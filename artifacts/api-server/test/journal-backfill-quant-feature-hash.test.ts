/**
 * Task #469 — backfill QUANT predictions silently dropped from
 * `prediction_journal` between the Task #406 brain flip and the Task #460
 * fix. The backfill must:
 *
 *   1. Identify QUANT-source rows in `predictions` that have no matching
 *      journal entry (the gap left by the silent-drop bug).
 *   2. Fetch the canonical feature vector + hash from `/ml/features` once
 *      per (coinId, timeframe) — bounded API call count regardless of how
 *      many predictions share the key.
 *   3. Insert a journal row preserving the ORIGINAL `created_at` so the
 *      audit/replay surface (abstain-rate denominators, journal-health
 *      sparkline) shows continuous coverage in the historical bucket.
 *   4. Skip QUANT rows that ALREADY have a journal entry — the helper is
 *      idempotent and re-running it on a healthy journal is a no-op.
 *   5. NEVER drop a row over an /ml/features failure: the live writer's
 *      labelled `missing:{source}:{modelVersion}:{coinId}:{timeframe}`
 *      placeholder is used as a last resort so the row still lands.
 */
import { describe, it, after, before } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import { db, predictionJournalTable, predictionsTable, agentsTable } from "@workspace/db";
import { eq, and } from "drizzle-orm";

import { backfillMissingQuantJournals } from "../src/lib/journal-writer";

const TRACER = "qfh-backfill-test";

type MockHandler = (req: http.IncomingMessage, parsed: { coinId: string; timeframe: string }) => unknown;

function startMockMlEngine(handler: MockHandler): Promise<{ url: string; close: () => Promise<void>; calls: Array<{ coinId: string; timeframe: string }> }> {
  const calls: Array<{ coinId: string; timeframe: string }> = [];
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      if (req.method === "POST" && req.url === "/ml/features") {
        let body = "";
        req.on("data", (c) => (body += c));
        req.on("end", () => {
          const parsed = JSON.parse(body) as { coinId: string; timeframe: string };
          calls.push(parsed);
          const result = handler(req, parsed);
          if (result === null) {
            res.writeHead(503, { "content-type": "application/json" });
            res.end(JSON.stringify({ error: "ml-engine down" }));
            return;
          }
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(result));
        });
        return;
      }
      res.writeHead(404).end();
    });
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as AddressInfo).port;
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise<void>((r) => server.close(() => r())),
        calls,
      });
    });
  });
}

let agentId: number;

async function ensureAgent(): Promise<number> {
  const [existing] = await db.select().from(agentsTable).where(eq(agentsTable.name, TRACER)).limit(1);
  if (existing) return existing.id;
  const [created] = await db
    .insert(agentsTable)
    .values({ name: TRACER, personality: "test-personality", systemPrompt: "test" })
    .returning({ id: agentsTable.id });
  return created.id;
}

async function cleanup(): Promise<void> {
  await db.delete(predictionJournalTable).where(eq(predictionJournalTable.coinId, TRACER));
  await db.delete(predictionsTable).where(eq(predictionsTable.coinId, TRACER));
}

describe("Task #469 — backfillMissingQuantJournals", () => {
  before(async () => {
    agentId = await ensureAgent();
    await cleanup();
  });
  after(async () => {
    await cleanup();
  });

  it("inserts a journal row for every dropped QUANT prediction, hydrates hash from /ml/features, preserves original createdAt, and skips already-journaled rows", async () => {
    await cleanup();

    // Two QUANT predictions on (TRACER, 1h) created an hour apart — both
    // dropped from the journal during the gap. One QUANT prediction on
    // (TRACER, 6h) — also dropped. One QUANT prediction on (TRACER, 1d)
    // that ALREADY has a journal entry — must not be re-inserted.
    const t0 = new Date(Date.now() - 3 * 60 * 60 * 1000);
    const t1 = new Date(Date.now() - 2 * 60 * 60 * 1000);
    const t2 = new Date(Date.now() - 1 * 60 * 60 * 1000);

    const dropped1 = await insertPrediction({ coinId: TRACER, timeframe: "1h", source: "lightgbm", createdAt: t0 });
    const dropped2 = await insertPrediction({ coinId: TRACER, timeframe: "1h", source: "lightgbm", createdAt: t1 });
    const dropped3 = await insertPrediction({ coinId: TRACER, timeframe: "6h", source: "lightgbm", createdAt: t2 });
    const alreadyJournaled = await insertPrediction({ coinId: TRACER, timeframe: "1d", source: "lightgbm", createdAt: t2 });

    // Pre-existing journal row for the 1d prediction (simulating a row
    // that wasn't lost — the backfill must NOT touch it).
    await db.insert(predictionJournalTable).values({
      predictionId: alreadyJournaled.id,
      brain: "QUANT",
      agentId,
      agentName: TRACER,
      coinId: TRACER,
      coinName: TRACER,
      timeframe: "1d",
      modelId: "lightgbm",
      modelVersion: "v1",
      source: "lightgbm",
      featureHash: "preexisting-hash",
      featureVector: null,
      regimeLabel: null,
      direction: "up",
      confidence: 0.6,
      rawConfidence: null,
      probUp: null, probDown: null, probStable: null,
      expectedReturnPct: null, predictionStdPct: null,
      priceAtPrediction: 100,
      predictedPrice: 101,
      gatesApplied: {},
      becameTrade: null,
      skipReason: null,
      tradeId: null,
      resolvesAt: null,
    });

    // Mock /ml/features returning a real hash + vector for both keys.
    const mock = await startMockMlEngine((_req, parsed) => ({
      coinId: parsed.coinId,
      timeframe: parsed.timeframe,
      candleCount: 200,
      features: { rsi14: 55.5, macdHist: 0.1, atrPct: 1.2 },
      insufficientData: false,
      featureHash: `hash:${parsed.coinId}:${parsed.timeframe}`,
      durationMs: 5,
    }));
    process.env.ML_ENGINE_URL = mock.url;

    try {
      const result = await backfillMissingQuantJournals({
        since: new Date(t0.getTime() - 60_000),
      });

      // Three QUANT predictions in window; one already journaled; two
      // unique (coinId,timeframe) keys hydrated (1h + 6h, NOT 1d because
      // that row was already journaled and skipped before /ml/features).
      assert.equal(result.scanned, 4, "scanned must include all four QUANT rows in the window");
      assert.equal(result.alreadyJournaled, 1, "must skip the row that already had a journal entry");
      assert.equal(result.inserted, 3, "must insert one journal row per dropped QUANT prediction");
      assert.equal(result.failed, 0);
      assert.equal(result.uniqueKeys, 2, "must call /ml/features once per (coinId, timeframe) — 1h and 6h");
      assert.equal(result.featuresHydrated, 2);
      assert.equal(result.featuresFailed, 0);

      // /ml/features call count is bounded by unique keys, not row count.
      assert.equal(mock.calls.length, 2, "/ml/features must be called once per unique (coinId,timeframe)");

      // Both 1h dropped rows landed with the hydrated hash + vector and
      // preserved their original createdAt (so the journal-health
      // sparkline puts them in the historical bucket, not "now").
      const r1 = await fetchJournal(dropped1.id);
      const r2 = await fetchJournal(dropped2.id);
      const r3 = await fetchJournal(dropped3.id);

      assert.equal(r1.featureHash, `hash:${TRACER}:1h`, "1h row must use the hydrated hash");
      assert.equal(r2.featureHash, `hash:${TRACER}:1h`);
      assert.equal(r3.featureHash, `hash:${TRACER}:6h`, "6h row must use the hydrated hash");
      assert.deepEqual(r1.featureVector, { rsi14: 55.5, macdHist: 0.1, atrPct: 1.2 });
      assert.equal(r1.brain, "QUANT");
      assert.equal(r1.source, "lightgbm");
      assert.equal(r1.createdAt.getTime(), t0.getTime(), "createdAt must be preserved");
      assert.equal(r2.createdAt.getTime(), t1.getTime());
      assert.equal(r3.createdAt.getTime(), t2.getTime());

      // The pre-existing 1d journal row was untouched — its hash is still
      // the one we wrote above, NOT a hydrated value.
      const preexisting = await fetchJournal(alreadyJournaled.id);
      assert.equal(preexisting.featureHash, "preexisting-hash", "must not touch already-journaled rows");

      // Re-running is a no-op (idempotency).
      const reRun = await backfillMissingQuantJournals({
        since: new Date(t0.getTime() - 60_000),
      });
      assert.equal(reRun.inserted, 0, "second run must insert nothing — backfill is idempotent");
      assert.equal(reRun.alreadyJournaled, 4, "all four rows must now appear journaled");
    } finally {
      await mock.close();
      delete process.env.ML_ENGINE_URL;
    }
  });

  it("falls back to the labelled `missing:` placeholder when /ml/features fails — never drops the row", async () => {
    await cleanup();

    const t0 = new Date(Date.now() - 30 * 60 * 1000);
    const dropped = await insertPrediction({
      coinId: TRACER,
      timeframe: "5m",
      source: "lightgbm",
      createdAt: t0,
      modelVersion: "lgbm-v7",
    });

    // /ml/features returns 503 — backfill MUST still write the row.
    const mock = await startMockMlEngine(() => null);
    process.env.ML_ENGINE_URL = mock.url;

    try {
      const result = await backfillMissingQuantJournals({
        since: new Date(t0.getTime() - 60_000),
      });
      assert.equal(result.inserted, 1, "row must land even when /ml/features fails");
      assert.equal(result.featuresHydrated, 0);
      assert.equal(result.featuresFailed, 1);

      const row = await fetchJournal(dropped.id);
      assert.equal(
        row.featureHash,
        `missing:lightgbm:lgbm-v7:${TRACER}:5m`,
        "must use the same labelled placeholder format as the live writer",
      );
      assert.equal(row.featureVector, null);
      assert.equal(row.brain, "QUANT");
    } finally {
      await mock.close();
      delete process.env.ML_ENGINE_URL;
    }
  });
});

// --- helpers ---------------------------------------------------------------

async function insertPrediction(opts: {
  coinId: string;
  timeframe: string;
  source: string;
  createdAt: Date;
  modelVersion?: string;
}): Promise<{ id: number }> {
  const [row] = await db
    .insert(predictionsTable)
    .values({
      agentId,
      agentName: TRACER,
      coinId: opts.coinId,
      coinName: opts.coinId,
      direction: "up",
      confidence: 0.65,
      reasoning: "test",
      priceAtPrediction: 100,
      predictedPrice: 101,
      timeframe: opts.timeframe,
      source: opts.source,
      patternContext: opts.modelVersion
        ? { quant: { modelVersion: opts.modelVersion, source: opts.source, featureHash: null } }
        : null,
      createdAt: opts.createdAt,
    })
    .returning({ id: predictionsTable.id });
  return row;
}

async function fetchJournal(predictionId: number): Promise<{
  featureHash: string | null;
  featureVector: unknown;
  brain: string;
  source: string | null;
  createdAt: Date;
}> {
  const [row] = await db
    .select({
      featureHash: predictionJournalTable.featureHash,
      featureVector: predictionJournalTable.featureVector,
      brain: predictionJournalTable.brain,
      source: predictionJournalTable.source,
      createdAt: predictionJournalTable.createdAt,
    })
    .from(predictionJournalTable)
    .where(and(
      eq(predictionJournalTable.predictionId, predictionId),
      eq(predictionJournalTable.coinId, TRACER),
    ))
    .limit(1);
  assert.ok(row, `expected journal row for predictionId=${predictionId}`);
  return row;
}
