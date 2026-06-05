import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { eq, inArray } from "drizzle-orm";

// Task #619 — integration coverage: when `evaluateMttmAutoDisable`
// trips on the rising edge, it must invoke the notifier exactly once
// and the dedup row written by the notifier must reflect the trip.
//
// Mirrors the seeding strategy used by `mttm-consecutive-loss.test.ts`
// (real DB, scoped tracer agent, MTTM seeded via the public setters).

import {
  db,
  agentsTable,
  paperTradesTable,
  appSettingsTable,
} from "@workspace/db";
import {
  evaluateMttmAutoDisable,
  setMttmEnabled,
  setMttmUniverse,
  invalidateMttmCache,
  MTTM_KEYS,
  DEFAULT_MTTM_UNIVERSE,
} from "../src/lib/mttm.ts";
import { MTTM_DISABLE_ALERTS_SENT_KEY } from "../src/lib/mttm-disable-notifier.ts";

const TRACER = "mttm-auto-disable-notifier-integration-test";
let agentId: number;
let savedFetch: typeof fetch;
let savedSlackEnv: string | undefined;
let savedSlackEnv2: string | undefined;
let savedGenericEnv: string | undefined;

async function ensureAgent(): Promise<number> {
  const [existing] = await db
    .select()
    .from(agentsTable)
    .where(eq(agentsTable.name, TRACER))
    .limit(1);
  if (existing) return existing.id;
  const [created] = await db
    .insert(agentsTable)
    .values({ name: TRACER, personality: "test-personality" })
    .returning({ id: agentsTable.id });
  return created.id;
}

async function clearMttmSettings(): Promise<void> {
  await db
    .delete(appSettingsTable)
    .where(inArray(appSettingsTable.key, [...MTTM_KEYS]));
  invalidateMttmCache();
}

async function clearTracerTrades(): Promise<void> {
  await db.delete(paperTradesTable).where(eq(paperTradesTable.agentId, agentId));
}

async function clearDedup(): Promise<void> {
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MTTM_DISABLE_ALERTS_SENT_KEY));
}

describe("Task #619 — evaluateMttmAutoDisable triggers the notifier", () => {
  before(async () => {
    agentId = await ensureAgent();
    await clearTracerTrades();
    await clearMttmSettings();
    await clearDedup();
    savedFetch = globalThis.fetch;
    savedSlackEnv = process.env.SLACK_WEBHOOK_URL;
    savedSlackEnv2 = process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL;
    savedGenericEnv = process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL;
    delete process.env.SLACK_WEBHOOK_URL;
    delete process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL;
    process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL =
      "https://example.invalid/mttm-trip";
  });

  after(async () => {
    globalThis.fetch = savedFetch;
    if (savedSlackEnv === undefined) delete process.env.SLACK_WEBHOOK_URL;
    else process.env.SLACK_WEBHOOK_URL = savedSlackEnv;
    if (savedSlackEnv2 === undefined)
      delete process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL;
    else process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL = savedSlackEnv2;
    if (savedGenericEnv === undefined)
      delete process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL;
    else process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL = savedGenericEnv;
    await clearTracerTrades();
    await clearMttmSettings();
    await clearDedup();
  });

  beforeEach(async () => {
    await clearTracerTrades();
    await clearMttmSettings();
    await clearDedup();
  });

  it("auto-disable on consecutive losses fires exactly one webhook and records dedup", async () => {
    const calls: Array<{ url: string; body: string }> = [];
    globalThis.fetch = (async (
      url: string | URL | Request,
      init?: RequestInit,
    ) => {
      calls.push({ url: String(url), body: String(init?.body ?? "") });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const enabledAtPast = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000);
    await setMttmUniverse(DEFAULT_MTTM_UNIVERSE);
    await setMttmEnabled(true, {
      enabledAt: enabledAtPast,
      clearDisableReason: true,
    });

    const now = Date.now();
    const rows = [];
    for (let i = 0; i < 5; i++) {
      rows.push({
        agentId,
        agentName: TRACER,
        coinId: "bonk",
        coinName: "Bonk",
        action: "buy" as const,
        entryPrice: 100,
        exitPrice: 90,
        quantity: 1,
        positionSize: 100,
        entryFee: 0.1,
        pnl: -10,
        pnlPercent: -0.1,
        timeframe: "6h",
        status: "closed",
        closedAt: new Date(now - (5 - i) * 60_000),
      });
    }
    await db.insert(paperTradesTable).values(rows);

    const reason = await evaluateMttmAutoDisable();
    assert.ok(reason, "evaluator must return a reason on the rising edge");

    // The notifier is fire-and-forget. Poll for both side-effects (the
    // webhook POST and the dedup row write) up to 2s so the test is
    // robust to db latency when run alongside other test files.
    const expectedKey = `mttm_disable@${reason!.trippedAt}`;
    const deadline = Date.now() + 2000;
    let dedupRow: typeof appSettingsTable.$inferSelect | undefined;
    while (Date.now() < deadline) {
      if (calls.length >= 1) {
        const r = await db
          .select()
          .from(appSettingsTable)
          .where(eq(appSettingsTable.key, MTTM_DISABLE_ALERTS_SENT_KEY));
        if (r.length > 0) {
          const v = r[0].value as { keys?: string[] };
          if (Array.isArray(v.keys) && v.keys.includes(expectedKey)) {
            dedupRow = r[0];
            break;
          }
        }
      }
      await new Promise((r) => setTimeout(r, 25));
    }

    // Exactly one webhook fired (only the generic channel was configured).
    assert.equal(
      calls.length,
      1,
      `expected one webhook call, got ${calls.length}`,
    );
    assert.ok(calls[0].url.includes("/mttm-trip"));
    assert.ok(calls[0].body.includes("consecutive_losses"));
    assert.ok(calls[0].body.includes("\"consecutiveLosses\":5"));

    // Dedup row written and keyed by the reason's trippedAt.
    assert.ok(dedupRow, "dedup row must be persisted by the notifier");
    const v = dedupRow!.value as { keys: string[] };
    assert.ok(
      v.keys.includes(expectedKey),
      `dedup keys must include ${expectedKey}, got ${JSON.stringify(v.keys)}`,
    );

    // A second evaluator call (lane already disabled) must NOT
    // re-fire — the evaluator short-circuits and the dedup ensures
    // the notifier is a noop even if the evaluator changes.
    const callsBefore = calls.length;
    await evaluateMttmAutoDisable();
    await new Promise((r) => setTimeout(r, 200));
    assert.equal(
      calls.length,
      callsBefore,
      "no extra webhook calls after re-evaluating an already-tripped lane",
    );
  });
});
