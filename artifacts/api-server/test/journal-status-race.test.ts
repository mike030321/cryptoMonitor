import { describe, it, after } from "node:test";
import assert from "node:assert/strict";

import { db, predictionJournalTable } from "@workspace/db";
import { eq, sql } from "drizzle-orm";

import {
  writePredictionJournal,
  markPredictionJournalTrade,
} from "../src/lib/journal-writer";

const TRACER = "journal-status-race-test";

async function cleanup(): Promise<void> {
  await db
    .delete(predictionJournalTable)
    .where(eq(predictionJournalTable.coinId, TRACER));
}

describe("prediction_journal status race-safety", () => {
  after(cleanup);

  it("status update lands when called BEFORE the insert (retry succeeds)", async () => {
    await cleanup();
    const predictionId = Math.floor(Math.random() * 1_000_000_000) + 800_000_000;

    // Kick off the status update first; it must NOT find a row yet, so it
    // must retry. Then we insert mid-retry. The final state must reflect
    // the status update — proving the writer is race-safe.
    const statusPromise = markPredictionJournalTrade(predictionId, {
      becameTrade: true,
      tradeId: 12345,
    });

    // Sleep briefly so the first retry attempt fires against an empty
    // table, then insert.
    await new Promise((r) => setTimeout(r, 50));
    await writePredictionJournal({
      predictionId,
      brain: "QUANT",
      agentId: null,
      agentName: null,
      coinId: TRACER,
      coinName: "race-test",
      timeframe: "1h",
      modelId: "test",
      modelVersion: null,
      source: "model",
      featureHash: null,
      featureVector: null,
      regimeLabel: null,
      direction: "up",
      confidence: 0.5,
      rawConfidence: null,
      probUp: null, probDown: null, probStable: null,
      expectedReturnPct: null, predictionStdPct: null,
      priceAtPrediction: 100,
      predictedPrice: 101,
      gatesApplied: {},
      becameTrade: null,
      skipReason: null,
      tradeId: null,
      resolvesAt: new Date(),
    });

    await statusPromise;

    const [row] = await db
      .select()
      .from(predictionJournalTable)
      .where(eq(predictionJournalTable.predictionId, predictionId))
      .limit(1);
    assert.ok(row, "expected row to exist");
    assert.equal(row.becameTrade, true, "status update must have landed");
    assert.equal(row.tradeId, 12345);
  });

  it("NO-TRADE branch records becameTrade=false + skipReason", async () => {
    await cleanup();
    const predictionId = Math.floor(Math.random() * 1_000_000_000) + 700_000_000;

    await writePredictionJournal({
      predictionId,
      brain: "QUANT",
      agentId: null, agentName: null,
      coinId: TRACER, coinName: "race-test",
      timeframe: "1h",
      modelId: "test", modelVersion: null, source: "model",
      featureHash: null, featureVector: null, regimeLabel: null,
      direction: "up", confidence: 0.5, rawConfidence: null,
      probUp: null, probDown: null, probStable: null,
      expectedReturnPct: null, predictionStdPct: null,
      priceAtPrediction: 100, predictedPrice: 101,
      gatesApplied: {},
      becameTrade: null, skipReason: null, tradeId: null,
      resolvesAt: new Date(),
    });

    await markPredictionJournalTrade(predictionId, {
      becameTrade: false,
      skipReason: "no_trade_zone",
    });

    const [row] = await db
      .select()
      .from(predictionJournalTable)
      .where(eq(predictionJournalTable.predictionId, predictionId))
      .limit(1);
    assert.ok(row);
    assert.equal(row.becameTrade, false);
    assert.equal(row.skipReason, "no_trade_zone");
    assert.equal(row.tradeId, null);
  });

  it("Task #460 — QUANT row missing feature_hash lands with synthesized placeholder, not silently dropped", async () => {
    // Regression for Task #460 — the previous behaviour returned null and
    // dropped the row when a QUANT prediction arrived without an upstream
    // feature_hash. After Task #406's brain flip every freshly-retrained
    // 1d/6h LightGBM prediction hit this path and disappeared from the
    // journal silently, while the dashboard kept rendering them. The
    // contract is now: synthesize a `missing:{source}:{modelVersion}:
    // {coinId}:{timeframe}` placeholder, log a warn, and write the row.
    await cleanup();
    const predictionId = Math.floor(Math.random() * 1_000_000_000) + 500_000_000;

    const insertedId = await writePredictionJournal({
      predictionId,
      brain: "QUANT",
      agentId: null, agentName: null,
      coinId: TRACER, coinName: "race-test",
      timeframe: "1h",
      modelId: "lightgbm", modelVersion: "v9", source: "lightgbm",
      featureHash: null, // <-- the contract violation under test
      featureVector: null, regimeLabel: null,
      direction: "up", confidence: 0.7, rawConfidence: null,
      probUp: null, probDown: null, probStable: null,
      expectedReturnPct: null, predictionStdPct: null,
      priceAtPrediction: 100, predictedPrice: 101,
      gatesApplied: {},
      becameTrade: null, skipReason: null, tradeId: null,
      resolvesAt: new Date(),
    });

    // Row must land — NEVER silently dropped — so the audit trail is
    // continuous even under upstream contract violations.
    assert.ok(insertedId, "writer must NOT return null on missing feature_hash");

    const [row] = await db
      .select()
      .from(predictionJournalTable)
      .where(eq(predictionJournalTable.predictionId, predictionId))
      .limit(1);
    assert.ok(row, "row must exist in prediction_journal");
    // Placeholder is clearly labelled and deterministic so future log /
    // dashboard surfaces can spot upstream contract violations without
    // grepping the warn stream.
    assert.equal(
      row.featureHash,
      `missing:lightgbm:v9:${TRACER}:1h`,
      "synthesized hash must encode source/version/coin/timeframe",
    );
    // The row is otherwise a normal QUANT row — direction, confidence,
    // and brain are preserved exactly as the caller passed them.
    assert.equal(row.brain, "QUANT");
    assert.equal(row.direction, "up");
  });

  it("merges gatesApplied without clobbering existing keys", async () => {
    await cleanup();
    const predictionId = Math.floor(Math.random() * 1_000_000_000) + 600_000_000;

    await writePredictionJournal({
      predictionId,
      brain: "QUANT",
      agentId: null, agentName: null,
      coinId: TRACER, coinName: "race-test",
      timeframe: "1h",
      modelId: "lgbm", modelVersion: "v1", source: "lgbm",
      featureHash: null, featureVector: null, regimeLabel: null,
      direction: "up", confidence: 0.7, rawConfidence: null,
      probUp: null, probDown: null, probStable: null,
      expectedReturnPct: null, predictionStdPct: null,
      priceAtPrediction: 100, predictedPrice: 101,
      gatesApplied: { noTradeZone: false },
      becameTrade: null, skipReason: null, tradeId: null,
      resolvesAt: new Date(),
    });

    await markPredictionJournalTrade(predictionId, {
      becameTrade: false,
      skipReason: "fee_gate_ev",
      gatesApplied: { fee_gate_ev: true },
    });

    const [row] = await db
      .select()
      .from(predictionJournalTable)
      .where(eq(predictionJournalTable.predictionId, predictionId))
      .limit(1);
    assert.ok(row);
    const gates = row.gatesApplied as Record<string, unknown>;
    // Merge: original noTradeZone preserved, new fee_gate_ev added.
    assert.equal(gates.noTradeZone, false);
    assert.equal(gates.fee_gate_ev, true);
    assert.equal(row.skipReason, "fee_gate_ev");
  });
});

// Silence unused import lint (sql is intentionally available for future
// raw queries if the schema gains generated columns).
void sql;
