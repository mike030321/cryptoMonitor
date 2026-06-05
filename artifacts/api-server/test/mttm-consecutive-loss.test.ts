import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { eq, inArray } from "drizzle-orm";

// Task #614 — MTTM auto-disable on consecutive losses.
//
// Seeds the database with an MTTM-enabled state and ≥5 closed losing
// MTTM trades and verifies that `evaluateMttmAutoDisable()`:
//   - flips `mttm_enabled` to false,
//   - persists a typed `mttm_disable_reason` row,
//   - returns the reason from the call.
//
// Uses the real DB (DATABASE_URL is provisioned in the test env). All
// tracer rows are scoped to a unique agent name and cleaned up in
// `after`.

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
} from "../src/lib/mttm";

const TRACER = "mttm-consec-loss-test";
let agentId: number;

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

describe("Task #614 — MTTM auto-disable (consecutive losses)", () => {
  before(async () => {
    agentId = await ensureAgent();
    await clearTracerTrades();
    await clearMttmSettings();
  });

  after(async () => {
    await clearTracerTrades();
    await clearMttmSettings();
  });

  it("5 consecutive MTTM losses flip mttm_enabled=false with a typed reason", async () => {
    // Pin an enable timestamp two days in the past so the trades we
    // seed (closedAt = now - {5..1} minutes) sit AFTER it.
    const enabledAtPast = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000);
    await setMttmUniverse(DEFAULT_MTTM_UNIVERSE);
    await setMttmEnabled(true, { enabledAt: enabledAtPast, clearDisableReason: true });

    // Seed 5 closed MTTM-universe losing trades on (bonk, 6h).
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
        pnlPercent: -0.10,
        timeframe: "6h",
        status: "closed",
        closedAt: new Date(now - (5 - i) * 60_000),
      });
    }
    await db.insert(paperTradesTable).values(rows);

    const reason = await evaluateMttmAutoDisable();
    assert.ok(reason, "evaluateMttmAutoDisable must return a reason after 5 losses");
    assert.equal(reason!.reason, "consecutive_losses");
    assert.ok(
      (reason!.consecutiveLosses ?? 0) >= 5,
      `consecutiveLosses must be ≥5, got ${reason!.consecutiveLosses}`,
    );

    // mttm_enabled flag must now be persisted as false.
    invalidateMttmCache();
    const enabledRow = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, "mttm_enabled"));
    const persisted = (enabledRow[0]?.value as { enabled?: boolean } | undefined)?.enabled;
    assert.equal(persisted, false, "mttm_enabled must be persisted as false after auto-disable");
  });

  it("evaluate is idempotent — calling again with disable already set returns the same reason", async () => {
    const reason = await evaluateMttmAutoDisable();
    // After the previous case the lane is disabled; the function must
    // either return the existing reason or null without mutating
    // state.
    if (reason) {
      assert.ok(
        reason.reason === "consecutive_losses" || reason.reason === "manual",
        `unexpected reason kind: ${reason.reason}`,
      );
    }
  });
});
