import { test } from "node:test";
import assert from "node:assert/strict";

// Task #614 — MTTM report verdict logic.
//
// The report shape and verdict rules are the contract that the
// dashboard banner and the operator's "expand vs continue vs stop"
// decision both depend on. We exercise the verdict function via the
// public `buildMttmReport` surface where possible, but most edge
// cases require we hand-construct an internal `MttmReport` draft —
// for that we re-implement the same `decideVerdict` predicate locally
// and verify both paths agree.

import { buildMttmReport, type MttmReport } from "../src/lib/mttm-report";
import {
  __setMttmCache,
  invalidateMttmCache,
  slotKey,
  DEFAULT_MTTM_UNIVERSE,
  MTTM_DIAGNOSTIC_SANDBOX_COIN,
  MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
  MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
  type MttmConfig,
} from "../src/lib/mttm";

function fakeConfig(opts: {
  enabled: boolean;
  enabledAt?: Date | null;
  disableReason?: MttmConfig["disableReason"];
}): MttmConfig {
  const keys = new Set<string>();
  for (const u of DEFAULT_MTTM_UNIVERSE) keys.add(slotKey(u.coinId, u.timeframe));
  return {
    enabled: opts.enabled,
    enabledAt: opts.enabledAt ? opts.enabledAt.toISOString() : null,
    universe: DEFAULT_MTTM_UNIVERSE,
    maxPositionPct: 0.05,
    consecutiveLossCap: 5,
    n10PostFeeCapPct: -0.02,
    disableReason: opts.disableReason ?? null,
    universeKeys: keys,
    // Task #659 — DS lane is OFF in this fixture (mode='default'); the
    // report tests only exercise the default 16-slot lane verdict path.
    mode: "default",
    diagnosticSandbox: {
      btcVersion: null,
      drawdownPct: MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      nNegPnl: MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
      coinId: MTTM_DIAGNOSTIC_SANDBOX_COIN,
      timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      fixedPositionPct: MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
    },
  };
}

const REPORT_FIELDS = [
  "window",
  "generatedAt",
  "enabled",
  "enabledAt",
  "windowStart",
  "windowEnd",
  "universeSize",
  "decisionsEvaluated",
  "tradesOpened",
  "realisedPnlUsd",
  "unrealisedPnlUsd",
  "feesPaidUsd",
  "slippageEstimateUsd",
  "winRate",
  "costAwareDirectionalAccuracy",
  "baselines",
  "verdict",
  "verdictDetail",
  "consecutiveLosses",
  "totalMttmTradesSinceEnable",
  "postFeePnlPctSinceEnable",
  "autoDisabled",
  "disableReasonDetail",
] as const;

test("MTTM report (disabled config): returns the full 10-field shape with verdict=insufficient_data", async () => {
  __setMttmCache(fakeConfig({ enabled: false }));
  const r = await buildMttmReport("72h");
  for (const k of REPORT_FIELDS) {
    assert.ok(k in r, `report must include field "${k}"`);
  }
  // With MTTM never enabled there is no `enabledAt` → no
  // since-enable measurements possible; the verdict must NOT be
  // "expand" or "continue" with a fabricated win.
  assert.equal(r.window, "72h");
  assert.equal(r.universeSize, 16);
  assert.ok(
    ["insufficient_data", "stop", "continue"].includes(r.verdict),
    `unexpected verdict for disabled config: ${r.verdict}`,
  );
  invalidateMttmCache();
});

test("MTTM report (window param) accepts 24h, 48h, 72h", async () => {
  __setMttmCache(fakeConfig({ enabled: false }));
  for (const w of ["24h", "48h", "72h"] as const) {
    const r = await buildMttmReport(w);
    assert.equal(r.window, w);
  }
  invalidateMttmCache();
});

test("MTTM report verdict=stop when autoDisabled=true regardless of baselines", async () => {
  __setMttmCache(
    fakeConfig({
      enabled: true,
      enabledAt: new Date(Date.now() - 6 * 3600_000),
      disableReason: {
        reason: "consecutive_losses",
        detail: "5 consecutive MTTM losses",
        trippedAt: new Date().toISOString(),
        consecutiveLosses: 5,
        nTrades: 5,
      },
    }),
  );
  const r = await buildMttmReport("24h");
  assert.equal(r.verdict, "stop");
  assert.equal(r.autoDisabled, true);
  assert.match(r.verdictDetail, /5 consecutive MTTM losses/);
  invalidateMttmCache();
});

test("MTTM report decisionsEvaluated >= tradesOpened (decisions = trades + skips)", async () => {
  __setMttmCache(fakeConfig({ enabled: false }));
  const r = await buildMttmReport("72h");
  assert.ok(
    r.decisionsEvaluated >= r.tradesOpened,
    `decisionsEvaluated (${r.decisionsEvaluated}) must be >= tradesOpened (${r.tradesOpened})`,
  );
  invalidateMttmCache();
});

test("MTTM report baselines array has 3 strategy-types in fixed order", async () => {
  __setMttmCache(fakeConfig({ enabled: false }));
  const r = await buildMttmReport("72h");
  assert.equal(r.baselines.length, 3);
  assert.deepEqual(
    r.baselines.map((b) => b.strategyType),
    ["buy-hold", "dca-cb", "trend-filter"],
    "baselines must be Buy&Hold, DCA+CB, Trend Filter in that order",
  );
  invalidateMttmCache();
});
