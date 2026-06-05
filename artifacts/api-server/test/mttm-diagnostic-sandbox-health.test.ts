/**
 * Task #670 — DS health probe.
 *
 * Covers the `getDiagnosticSandboxHealth` helper end-to-end against a
 * real `paper_trades` table. The helper is the early-warning surface
 * the dashboard renders BEFORE the auto-disable evaluator's full-since
 * -enable drawdown trips. Regressions in the trailing-window slicing
 * (e.g. forgetting to take just the most-recent N), the warn-line
 * arithmetic, or the `needs_refit` / `floor_breached` flags would
 * silently let the lane drift over the floor without the soft warning
 * the dashboard was added to render — exactly the failure mode this
 * task was opened to prevent. Each scenario seeds a deterministic trade
 * stream, pins an in-memory DS config, and asserts the health snapshot.
 */
import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { eq, inArray } from "drizzle-orm";

import {
  db,
  agentsTable,
  paperTradesTable,
  appSettingsTable,
} from "@workspace/db";
import {
  getDiagnosticSandboxHealth,
  invalidateDiagnosticSandboxHealthCache,
  invalidateMttmCache,
  __setMttmCache,
  slotKey,
  MTTM_KEYS,
  MTTM_DIAGNOSTIC_SANDBOX_COIN,
  MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
  MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
  MTTM_DIAGNOSTIC_SANDBOX_HEALTH_WARN_FRACTION,
  type MttmConfig,
} from "../src/lib/mttm";

const TRACER = "mttm-ds-health-test-670";
const FAKE_BTC_VERSION = "20260501T000000Z-test670-fake";
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

async function clearTracerTrades(): Promise<void> {
  await db.delete(paperTradesTable).where(eq(paperTradesTable.agentId, agentId));
}

async function clearMttmSettings(): Promise<void> {
  await db
    .delete(appSettingsTable)
    .where(inArray(appSettingsTable.key, [...MTTM_KEYS]));
  invalidateMttmCache();
  invalidateDiagnosticSandboxHealthCache();
}

function dsConfigFixture(
  enabledAt: Date,
  overrides: { drawdownPct?: number; nNegPnl?: number; mode?: "default" | "diagnostic_sandbox"; enabled?: boolean } = {},
): MttmConfig {
  const universe = [
    {
      coinId: MTTM_DIAGNOSTIC_SANDBOX_COIN,
      timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      version: FAKE_BTC_VERSION,
    },
  ];
  const universeKeys = new Set<string>();
  for (const u of universe) universeKeys.add(slotKey(u.coinId, u.timeframe));
  return {
    enabled: overrides.enabled ?? true,
    enabledAt: enabledAt.toISOString(),
    universe,
    maxPositionPct: MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
    consecutiveLossCap: 5,
    n10PostFeeCapPct: -0.02,
    disableReason: null,
    universeKeys,
    mode: overrides.mode ?? "diagnostic_sandbox",
    diagnosticSandbox: {
      btcVersion: FAKE_BTC_VERSION,
      drawdownPct: overrides.drawdownPct ?? MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      nNegPnl: overrides.nNegPnl ?? MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
      coinId: MTTM_DIAGNOSTIC_SANDBOX_COIN,
      timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      fixedPositionPct: MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
    },
  };
}

interface TradeSeed {
  pnlPercent: number;
  closedAt: Date;
}

async function seedDsTrades(seeds: TradeSeed[]): Promise<void> {
  const rows = seeds.map((s) => {
    const positionSize = 50;
    const pnl = positionSize * s.pnlPercent;
    return {
      agentId,
      agentName: TRACER,
      coinId: MTTM_DIAGNOSTIC_SANDBOX_COIN,
      coinName: "Bitcoin",
      action: "buy" as const,
      entryPrice: 100,
      exitPrice: 100 * (1 + s.pnlPercent),
      quantity: positionSize / 100,
      positionSize,
      entryFee: positionSize * 0.001,
      pnl,
      pnlPercent: s.pnlPercent,
      timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      status: "closed",
      closedAt: s.closedAt,
    };
  });
  await db.insert(paperTradesTable).values(rows);
}

describe("Task #670 — diagnostic-sandbox health probe", () => {
  before(async () => {
    agentId = await ensureAgent();
    await clearTracerTrades();
    await clearMttmSettings();
  });

  after(async () => {
    await clearTracerTrades();
    await clearMttmSettings();
  });

  beforeEach(async () => {
    await clearTracerTrades();
    invalidateDiagnosticSandboxHealthCache();
    invalidateMttmCache();
  });

  it("returns a non-evaluable snapshot when DS mode is off", async () => {
    const cfg = dsConfigFixture(new Date(), { mode: "default" });
    __setMttmCache(cfg);
    const h = await getDiagnosticSandboxHealth(cfg, { force: true });
    assert.equal(h.evaluable, false, "DS off ⇒ evaluable=false");
    assert.equal(h.needsRefit, false);
    assert.equal(h.floorBreached, false);
    assert.equal(h.nTradesObserved, 0);
    // Floor + warn line are still echoed so the dashboard can render
    // the policy preview.
    assert.equal(h.drawdownFloorPct, MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT);
    assert.equal(
      h.warnThresholdPct,
      MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT * MTTM_DIAGNOSTIC_SANDBOX_HEALTH_WARN_FRACTION,
    );
    assert.equal(h.warnFraction, MTTM_DIAGNOSTIC_SANDBOX_HEALTH_WARN_FRACTION);
  });

  it("returns a non-evaluable snapshot when DS is enabled but no trades have closed", async () => {
    const enabledAt = new Date(Date.now() - 1_000);
    const cfg = dsConfigFixture(enabledAt);
    __setMttmCache(cfg);
    const h = await getDiagnosticSandboxHealth(cfg, { force: true });
    assert.equal(h.evaluable, false, "no trades ⇒ evaluable=false");
    assert.equal(h.nTradesObserved, 0);
    assert.equal(h.trailingDrawdownPct, 0);
    assert.equal(h.needsRefit, false);
    assert.equal(h.floorBreached, false);
  });

  it("computes trailing-window drawdown over only the last N trades", async () => {
    // Seed an EARLY catastrophic loss block followed by a tail of small
    // wins. If the helper accidentally uses the full since-enable window
    // (like the auto-disable evaluator does) the early loss would
    // dominate and trip needs_refit. The trailing-window slice should
    // see only the tail and return a healthy snapshot.
    const enabledAt = new Date(Date.now() - 60_000);
    const baseClose = enabledAt.getTime() + 100;

    // 30 catastrophic losses at -10% pnl (account ret = -0.05% each)
    // followed by 10 small wins at +1% pnl. With nNegPnl=10 the
    // trailing window is just the wins.
    const seeds: TradeSeed[] = [];
    for (let i = 0; i < 30; i++) {
      seeds.push({ pnlPercent: -0.10, closedAt: new Date(baseClose + i * 10) });
    }
    for (let i = 0; i < 10; i++) {
      seeds.push({
        pnlPercent: 0.01,
        closedAt: new Date(baseClose + (30 + i) * 10),
      });
    }
    await seedDsTrades(seeds);

    const cfg = dsConfigFixture(enabledAt, { nNegPnl: 10 });
    __setMttmCache(cfg);
    const h = await getDiagnosticSandboxHealth(cfg, { force: true });

    assert.equal(h.evaluable, true);
    assert.equal(
      h.windowTrades,
      10,
      "windowTrades must mirror the configured nNegPnl",
    );
    assert.equal(
      h.nTradesObserved,
      10,
      "trailing window must observe exactly the last N trades",
    );
    // 10 wins at +0.005% each → equity walks monotonically up, trough = 0.
    assert.equal(
      h.trailingDrawdownPct,
      0,
      "monotonic-up tail must produce zero trailing drawdown",
    );
    assert.equal(h.needsRefit, false, "monotonic-up tail must not trip needs_refit");
    assert.equal(h.floorBreached, false);
    // Headroom is the full distance from 0 to the floor.
    assert.equal(h.headroomPct, 0 - MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT);
  });

  it("flips needs_refit when trailing drawdown crosses the warn line but stays above the floor", async () => {
    // Build a trailing window where the equity curve drops from peak by
    // ~4.2% — between the -4% warn line (0.8 * -5%) and the -5% floor.
    // We size each loss so 9 in a row push trough past -0.04 but stay
    // above -0.05.
    //
    //   per-trade account return = -0.10 * 0.005 = -0.0005
    //   equity after k trades    = (1 - 0.0005)^k
    //   90 trades                 ⇒ equity ≈ 0.95599 ⇒ dd ≈ -4.40%
    //   100 trades                ⇒ equity ≈ 0.95124 ⇒ dd ≈ -4.88%
    // Pick 90 trades so trough sits at ~-4.4%: past -4% warn but inside
    // -5% floor.
    const enabledAt = new Date(Date.now() - 60_000);
    const baseClose = enabledAt.getTime() + 100;
    const seeds: TradeSeed[] = [];
    for (let i = 0; i < 90; i++) {
      seeds.push({ pnlPercent: -0.10, closedAt: new Date(baseClose + i * 10) });
    }
    await seedDsTrades(seeds);

    const cfg = dsConfigFixture(enabledAt, { nNegPnl: 90 });
    __setMttmCache(cfg);
    const h = await getDiagnosticSandboxHealth(cfg, { force: true });

    assert.equal(h.evaluable, true);
    assert.equal(h.nTradesObserved, 90);
    // Sanity: trough is between the warn line and the floor.
    const warnLine =
      MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT *
      MTTM_DIAGNOSTIC_SANDBOX_HEALTH_WARN_FRACTION;
    assert.ok(
      h.trailingDrawdownPct <= warnLine,
      `trough ${h.trailingDrawdownPct} must be at-or-below the warn line ${warnLine}`,
    );
    assert.ok(
      h.trailingDrawdownPct > MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      `trough ${h.trailingDrawdownPct} must stay above the floor ${MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT}`,
    );
    assert.equal(h.needsRefit, true, "warn-line breach must flip needs_refit");
    assert.equal(
      h.floorBreached,
      false,
      "floor must NOT be breached when trough is above floor",
    );
    // Headroom is positive (= trailing - floor) when above the floor.
    assert.ok(
      h.headroomPct > 0,
      `headroom ${h.headroomPct} must be > 0 when above the floor`,
    );
  });

  it("flips floor_breached AND needs_refit when trailing drawdown crosses the floor", async () => {
    // 200 losses at -10% pnl ⇒ equity ≈ (1 - 0.0005)^200 ≈ 0.9048 ⇒ dd
    // ≈ -9.52%, comfortably past the -5% floor.
    const enabledAt = new Date(Date.now() - 60_000);
    const baseClose = enabledAt.getTime() + 100;
    const seeds: TradeSeed[] = [];
    for (let i = 0; i < 200; i++) {
      seeds.push({ pnlPercent: -0.10, closedAt: new Date(baseClose + i * 10) });
    }
    await seedDsTrades(seeds);

    const cfg = dsConfigFixture(enabledAt, { nNegPnl: 200 });
    __setMttmCache(cfg);
    const h = await getDiagnosticSandboxHealth(cfg, { force: true });

    assert.equal(h.evaluable, true);
    assert.equal(h.nTradesObserved, 200);
    assert.ok(
      h.trailingDrawdownPct <= MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      `trough ${h.trailingDrawdownPct} must be at-or-below floor`,
    );
    assert.equal(h.floorBreached, true);
    assert.equal(h.needsRefit, true, "floor breach implies needs_refit");
    // Headroom is non-positive when at-or-past the floor.
    assert.ok(
      h.headroomPct <= 0,
      `headroom ${h.headroomPct} must be <= 0 when at-or-past floor`,
    );
  });

  it("caches identical inputs and re-computes when the cache is invalidated", async () => {
    const enabledAt = new Date(Date.now() - 60_000);
    const baseClose = enabledAt.getTime() + 100;
    await seedDsTrades([
      { pnlPercent: -0.05, closedAt: new Date(baseClose + 0) },
      { pnlPercent: -0.05, closedAt: new Date(baseClose + 10) },
    ]);

    const cfg = dsConfigFixture(enabledAt, { nNegPnl: 50 });
    __setMttmCache(cfg);
    invalidateDiagnosticSandboxHealthCache();
    const first = await getDiagnosticSandboxHealth(cfg);
    assert.equal(first.evaluable, true);
    assert.equal(first.nTradesObserved, 2);

    // Add a new trade. Without invalidation, the cache must return the
    // same `computed_at` and `nTradesObserved`.
    await seedDsTrades([
      { pnlPercent: -0.05, closedAt: new Date(baseClose + 20) },
    ]);
    const cached = await getDiagnosticSandboxHealth(cfg);
    assert.equal(
      cached.computedAt,
      first.computedAt,
      "cached call must reuse the prior snapshot",
    );
    assert.equal(cached.nTradesObserved, first.nTradesObserved);

    // Force re-compute: the new trade must surface.
    invalidateDiagnosticSandboxHealthCache();
    const fresh = await getDiagnosticSandboxHealth(cfg, { force: true });
    assert.equal(fresh.nTradesObserved, 3);
    assert.notEqual(
      fresh.computedAt,
      first.computedAt,
      "post-invalidation call must produce a fresh snapshot",
    );
  });
});
