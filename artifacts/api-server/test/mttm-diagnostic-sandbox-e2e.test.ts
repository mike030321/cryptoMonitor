import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { eq, inArray } from "drizzle-orm";

// Task #661 — end-to-end coverage of the diagnostic-sandbox (DS) lane.
//
// Existing tests prove the building blocks individually:
//   - `diagnostic-sandbox-sizing.test.ts` pins the sizing constants and
//     paper-trader source-level branches.
//   - `mttm-position-size-override.test.ts` proves the cap-pick math.
//   - `mttm-universe-guard-e2e.test.ts` proves the whitelist guard via
//     `executePaperTrade`, but only for the default 16-slot lane.
//
// Nothing wires the FULL DS-mode lane together: mode flip → BTC version
// stamp → cache-driven universe collapse → simulated trade stream →
// `evaluateDiagnosticSandboxAutoDisable` returning the typed reason.
// This file does that. A regression where a future change to the cache
// primer or paper-trader override accidentally lets a non-BTC slot
// trade while `mttm_mode='diagnostic_sandbox'` would either change the
// collapsed universe or change which trades the evaluator filters in,
// and would surface here as a failed assertion.
//
// Reason-code mapping note: the task spec uses informal aliases
// (`diagnostic_sandbox_n_neg_pnl`, `diagnostic_sandbox_dd_floor`); the
// actual production reason codes emitted by mttm.ts are
// `diagnostic_negative_pnl_at_review` and `diagnostic_drawdown_exceeded`
// (see `MttmDiagnosticSandboxBreach`). We assert the production names.

import {
  db,
  agentsTable,
  paperTradesTable,
  appSettingsTable,
} from "@workspace/db";
import {
  evaluateDiagnosticSandboxAutoDisable,
  tripDiagnosticSandboxDrift,
  setMttmMode,
  setDiagnosticSandboxBtcVersion,
  getMttmConfig,
  clearMttmDisableReason,
  invalidateMttmCache,
  __setMttmCache,
  slotKey,
  MTTM_KEYS,
  MTTM_DIAGNOSTIC_SANDBOX_KEY,
  MTTM_DISABLE_REASON_KEY,
  MTTM_DIAGNOSTIC_SANDBOX_COIN,
  MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
  MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
  type MttmConfig,
} from "../src/lib/mttm";

const TRACER = "mttm-ds-e2e-test-661";
const FAKE_BTC_VERSION = "20260430T000000Z-test661-fake";
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

/**
 * Build an in-memory DS-mode config that mirrors what `getMttmConfig`
 * would return after `setMttmMode("diagnostic_sandbox")` +
 * `setDiagnosticSandboxBtcVersion(FAKE_BTC_VERSION)`. We pass this
 * directly into the evaluator so the lane state under test is
 * deterministic and decoupled from the test ordering of the live DB
 * row reads. `enabledAt` is set fresh per scenario so the evaluator's
 * `closedAt > enabledAt` filter only sees the trades we just inserted.
 */
function dsConfigFixture(enabledAt: Date): MttmConfig {
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
    enabled: true,
    enabledAt: enabledAt.toISOString(),
    universe,
    maxPositionPct: MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
    consecutiveLossCap: 5,
    n10PostFeeCapPct: -0.02,
    disableReason: null,
    universeKeys,
    mode: "diagnostic_sandbox",
    diagnosticSandbox: {
      btcVersion: FAKE_BTC_VERSION,
      drawdownPct: MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      nNegPnl: MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
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
  // Position size on the row is the gross notional; the evaluator does
  // NOT use it for DS — it weights each `pnlPercent` by the 0.5%
  // sizing pin to derive the per-trade account return. We still set a
  // non-zero notional so the row passes `notNull` constraints.
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

describe("Task #661 — diagnostic-sandbox lane end-to-end", () => {
  before(async () => {
    agentId = await ensureAgent();
    await clearTracerTrades();
    await clearMttmSettings();
  });

  after(async () => {
    await clearTracerTrades();
    await clearMttmSettings();
  });

  it("mode flip + BTC version stamp collapses the live universe to bitcoin/5m only and pins maxPositionPct to 0.5%", async () => {
    // Real DB round-trip: flip the lane, stamp the calibrated version,
    // then re-read getMttmConfig() so the cache is populated by the
    // production parser (not __setMttmCache).
    await setMttmMode("diagnostic_sandbox");
    const stamped = await setDiagnosticSandboxBtcVersion(FAKE_BTC_VERSION);

    assert.equal(stamped.mode, "diagnostic_sandbox");
    assert.equal(stamped.enabled, true, "DS mode flip must enable the lane");
    assert.equal(stamped.diagnosticSandbox.btcVersion, FAKE_BTC_VERSION);

    invalidateMttmCache();
    const cfg = await getMttmConfig();

    assert.equal(cfg.mode, "diagnostic_sandbox");
    assert.equal(cfg.enabled, true);
    assert.equal(
      cfg.universe.length,
      1,
      `DS lane universe must collapse to a single slot, got ${cfg.universe.length}`,
    );
    assert.equal(cfg.universe[0].coinId, MTTM_DIAGNOSTIC_SANDBOX_COIN);
    assert.equal(cfg.universe[0].timeframe, MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME);
    assert.equal(cfg.universe[0].version, FAKE_BTC_VERSION);
    assert.equal(cfg.universeKeys.size, 1);
    assert.ok(
      cfg.universeKeys.has(slotKey(
        MTTM_DIAGNOSTIC_SANDBOX_COIN,
        MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      )),
      "universeKeys must contain bitcoin|5m after DS mode flip",
    );
    assert.equal(
      cfg.maxPositionPct,
      MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
      `DS lane must pin maxPositionPct to ${MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT}`,
    );
    assert.equal(cfg.maxPositionPct, 0.005, "DS sizing pin must be 0.5%");
  });

  it("50 simulated BTC/5m losses with cumulative PnL<0 trip diagnostic_negative_pnl_at_review (the n≥50 + neg-PnL rule)", async () => {
    await clearTracerTrades();
    await clearMttmDisableReason();

    // enabledAt sits ~1s before our seeded trades so the evaluator's
    // `closedAt > enabledAt` filter sees ONLY this scenario's rows.
    const enabledAt = new Date(Date.now() - 1_000);
    const baseClose = enabledAt.getTime() + 100;
    const seeds: TradeSeed[] = [];
    for (let i = 0; i < 50; i++) {
      // pnlPercent = -0.10 → accountReturn = -0.10 * 0.005 = -0.0005.
      // Equity walk = (1 - 0.0005)^50 ≈ 0.9753 (drawdown ≈ -2.47%, which
      // sits ABOVE the -5% floor so the dd-rule does NOT pre-empt the
      // n + neg-PnL rule). cumulativePnlPct = 50 * -0.0005 = -0.025 < 0,
      // so the n≥50 rule fires.
      seeds.push({ pnlPercent: -0.10, closedAt: new Date(baseClose + i * 10) });
    }
    await seedDsTrades(seeds);

    const cfg = dsConfigFixture(enabledAt);
    __setMttmCache(cfg);

    // Sanity: the cached lane state is exactly what the evaluator
    // depends on — single-slot universe, 0.5% sizing pin.
    assert.equal(cfg.universeKeys.size, 1);
    assert.equal(cfg.maxPositionPct, 0.005);

    const reason = await evaluateDiagnosticSandboxAutoDisable(cfg);
    assert.ok(
      reason,
      "evaluator must return a typed reason after 50 losing BTC/5m trades",
    );
    assert.equal(
      reason!.reason,
      "diagnostic_negative_pnl_at_review",
      `expected n+neg-PnL trip (task spec alias 'diagnostic_sandbox_n_neg_pnl'), got: ${reason!.reason}`,
    );
    assert.equal(reason!.nTrades, 50);
    assert.ok(
      typeof reason!.cumulativePnlPct === "number" && reason!.cumulativePnlPct < 0,
      `cumulativePnlPct must be negative, got ${reason!.cumulativePnlPct}`,
    );
    // The drawdown rule must NOT have pre-empted: trough should sit
    // above (closer to 0 than) the -5% floor.
    assert.ok(
      typeof reason!.drawdownPct === "number" &&
        reason!.drawdownPct > MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      `drawdownPct ${reason!.drawdownPct} must be above the -5% floor; otherwise the dd-rule would have pre-empted`,
    );
  });

  it("full single-flow: live mode flip + version stamp + getMttmConfig + 50 BTC/5m losses trips n+neg-PnL with NO injected config", async () => {
    // This scenario complements the two evaluator-level cases by
    // exercising the full lane WITHOUT any test seam: real DB writes
    // for the lane state, getMttmConfig() repopulates the cache from
    // those writes, and `evaluateDiagnosticSandboxAutoDisable()` is
    // called with no injected config so it pulls everything itself.
    // This is the "would a refactor that decouples cache primer from
    // paper-trader override break things" check the task asked for.
    await clearTracerTrades();
    await clearMttmSettings();
    await setMttmMode("diagnostic_sandbox");
    await setDiagnosticSandboxBtcVersion(FAKE_BTC_VERSION);

    invalidateMttmCache();
    const liveCfg = await getMttmConfig();
    assert.equal(liveCfg.mode, "diagnostic_sandbox");
    assert.equal(liveCfg.enabled, true);
    assert.equal(liveCfg.universeKeys.size, 1);
    assert.equal(liveCfg.maxPositionPct, 0.005);
    assert.ok(liveCfg.enabledAt, "live config must have enabledAt populated");

    // Anchor seeded trades just after the live enabledAt so the
    // evaluator's `closedAt > enabledAt` filter scopes to them.
    const baseClose = new Date(liveCfg.enabledAt!).getTime() + 100;
    const seeds: TradeSeed[] = [];
    for (let i = 0; i < 50; i++) {
      seeds.push({ pnlPercent: -0.10, closedAt: new Date(baseClose + i * 10) });
    }
    await seedDsTrades(seeds);

    // No injected config — evaluator reads its own state.
    const reason = await evaluateDiagnosticSandboxAutoDisable();
    assert.ok(
      reason,
      "evaluator (no injected config) must return a typed reason after 50 losing BTC/5m trades against the live DS lane",
    );
    assert.equal(
      reason!.reason,
      "diagnostic_negative_pnl_at_review",
      `expected n+neg-PnL trip via the live cache primer, got: ${reason!.reason}`,
    );
    assert.equal(reason!.nTrades, 50);
  });

  it("a sequence of larger BTC/5m losses pushing equity drawdown to ≈ -6% trips diagnostic_drawdown_exceeded (the dd-floor rule)", async () => {
    await clearTracerTrades();
    await clearMttmDisableReason();

    const enabledAt = new Date(Date.now() - 1_000);
    const baseClose = enabledAt.getTime() + 100;
    // pnlPercent = -1.0 → accountReturn = -1.0 * 0.005 = -0.005 each.
    // (0.995)^13 ≈ 0.9369 → drawdown ≈ -6.31%, comfortably below the
    // -5% floor. Length is 13 (well under the n=50 threshold) so the
    // dd-rule is unambiguously the trigger.
    const seeds: TradeSeed[] = [];
    for (let i = 0; i < 13; i++) {
      seeds.push({ pnlPercent: -1.0, closedAt: new Date(baseClose + i * 10) });
    }
    await seedDsTrades(seeds);

    const cfg = dsConfigFixture(enabledAt);
    __setMttmCache(cfg);

    const reason = await evaluateDiagnosticSandboxAutoDisable(cfg);
    assert.ok(
      reason,
      "evaluator must return a typed reason once equity drawdown breaches the -5% floor",
    );
    assert.equal(
      reason!.reason,
      "diagnostic_drawdown_exceeded",
      `expected dd-floor trip (task spec alias 'diagnostic_sandbox_dd_floor'), got: ${reason!.reason}`,
    );
    assert.ok(
      typeof reason!.drawdownPct === "number" &&
        reason!.drawdownPct <= MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      `drawdownPct ${reason!.drawdownPct} must be at or below the -5% floor (${MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT})`,
    );
    assert.ok(
      typeof reason!.drawdownPct === "number" && reason!.drawdownPct <= -0.06,
      `drawdownPct ${reason!.drawdownPct} should reach ≈ -6% with the seeded sequence`,
    );
    assert.equal(
      reason!.nTrades,
      13,
      "n must be the full seeded sequence length (13), proving the dd rule fires before the n=50 rule could",
    );
  });

  // Task #666 — once a DS trip happens, the v1 row must reflect the
  // disable on its own. The legacy `mttm_enabled` row is no longer the
  // sole source of truth in DS mode (getMttmConfig overrides
  // `cfg.enabled` from the v1 row's `enabled` bit), so a trip that
  // only flipped `mttm_enabled` would silently re-arm if anything
  // ever cleared `mttm_disable_reason`. These cases pin both DS trip
  // entrypoints (the evaluator and the drift trip) to that contract.
  describe("Task #666 — DS auto-disable persists in the v1 row", () => {
    async function readV1Row(): Promise<{
      enabled: unknown;
      review: { auto_disabled: unknown; disable_reason: unknown } | null;
    } | null> {
      const rows = await db
        .select()
        .from(appSettingsTable)
        .where(eq(appSettingsTable.key, MTTM_DIAGNOSTIC_SANDBOX_KEY));
      if (rows.length === 0) return null;
      const v = rows[0].value as
        | {
            enabled?: unknown;
            review?: { auto_disabled?: unknown; disable_reason?: unknown };
          }
        | null
        | undefined;
      if (!v || typeof v !== "object") return null;
      return {
        enabled: v.enabled,
        review: v.review
          ? {
              auto_disabled: v.review.auto_disabled,
              disable_reason: v.review.disable_reason,
            }
          : null,
      };
    }

    it("evaluateDiagnosticSandboxAutoDisable trip flips v1 row's enabled→false and review.auto_disabled→true, and getMttmConfig returns enabled:false even with no mttm_disable_reason row present", async () => {
      await clearTracerTrades();
      await clearMttmSettings();
      await setMttmMode("diagnostic_sandbox");
      await setDiagnosticSandboxBtcVersion(FAKE_BTC_VERSION);

      // Sanity: v1 row is enabled before the trip.
      const before = await readV1Row();
      assert.ok(before, "v1 row must exist after mode flip");
      assert.equal(before!.enabled, true, "v1 row enabled must be true pre-trip");
      assert.equal(
        before!.review?.auto_disabled,
        false,
        "v1 row review.auto_disabled must be false pre-trip",
      );

      invalidateMttmCache();
      const liveCfg = await getMttmConfig();
      const baseClose = new Date(liveCfg.enabledAt!).getTime() + 100;
      const seeds: TradeSeed[] = [];
      for (let i = 0; i < 13; i++) {
        seeds.push({ pnlPercent: -1.0, closedAt: new Date(baseClose + i * 10) });
      }
      await seedDsTrades(seeds);

      const reason = await evaluateDiagnosticSandboxAutoDisable();
      assert.ok(reason, "evaluator must trip on the seeded losses");
      assert.equal(reason!.reason, "diagnostic_drawdown_exceeded");

      // Core regression: the v1 row itself must reflect the disable so
      // a stale operator action that clears `mttm_disable_reason`
      // cannot silently re-arm the lane.
      const after = await readV1Row();
      assert.ok(after, "v1 row must still exist after trip");
      assert.equal(
        after!.enabled,
        false,
        "v1 row enabled must flip to false after auto-disable trip",
      );
      assert.equal(
        after!.review?.auto_disabled,
        true,
        "v1 row review.auto_disabled must flip to true after auto-disable trip",
      );
      assert.equal(
        after!.review?.disable_reason,
        "diagnostic_drawdown_exceeded",
        "v1 row review.disable_reason must capture the trip reason code",
      );

      // Now simulate an operator (or bug) clearing mttm_disable_reason
      // and prove the lane stays disabled because the v1 row says so.
      await db
        .delete(appSettingsTable)
        .where(eq(appSettingsTable.key, MTTM_DISABLE_REASON_KEY));
      invalidateMttmCache();
      const reread = await getMttmConfig();
      assert.equal(reread.mode, "diagnostic_sandbox");
      assert.equal(
        reread.enabled,
        false,
        "DS lane must remain disabled via the v1 row even after mttm_disable_reason is cleared",
      );
      assert.equal(
        reread.disableReason,
        null,
        "this scenario specifically asserts disableReason is absent so it can't be the thing keeping the lane off",
      );
    });

    it("tripDiagnosticSandboxDrift also flips v1 row's enabled→false and review.auto_disabled→true", async () => {
      await clearTracerTrades();
      await clearMttmSettings();
      await setMttmMode("diagnostic_sandbox");
      await setDiagnosticSandboxBtcVersion(FAKE_BTC_VERSION);

      invalidateMttmCache();
      await getMttmConfig(); // warm cache so the trip uses live state

      const reason = await tripDiagnosticSandboxDrift(
        "diagnostic_universe_drift_detected",
        "test #666 — synthetic drift to verify v1-row persistence",
      );
      assert.ok(reason, "drift trip must return a typed reason");
      assert.equal(reason!.reason, "diagnostic_universe_drift_detected");

      const after = await readV1Row();
      assert.ok(after, "v1 row must exist after drift trip");
      assert.equal(
        after!.enabled,
        false,
        "v1 row enabled must flip to false after drift trip",
      );
      assert.equal(
        after!.review?.auto_disabled,
        true,
        "v1 row review.auto_disabled must flip to true after drift trip",
      );
      assert.equal(
        after!.review?.disable_reason,
        "diagnostic_universe_drift_detected",
        "v1 row review.disable_reason must capture the drift reason code",
      );

      // Same v1-row-is-authoritative check as the evaluator case.
      await db
        .delete(appSettingsTable)
        .where(eq(appSettingsTable.key, MTTM_DISABLE_REASON_KEY));
      invalidateMttmCache();
      const reread = await getMttmConfig();
      assert.equal(reread.mode, "diagnostic_sandbox");
      assert.equal(
        reread.enabled,
        false,
        "DS lane must remain disabled via the v1 row after a drift trip even if mttm_disable_reason is cleared",
      );
    });
  });
});
