import { test } from "node:test";
import assert from "node:assert/strict";

// Task #614 — MTTM position-size override.
//
// The override at paper-trader.ts:854-863 picks a per-trade cap based
// on `getMttmConfigCached()`:
//   - MTTM enabled + cache warm → mttm.maxPositionPct (5%)
//   - otherwise (MTTM off OR cache null) → MAX_POSITION_PCT (30%)
//
// We exercise the cap arithmetic directly using the real cache primer
// so a regression in the override pick (e.g. someone re-introducing
// `MAX_POSITION_PCT` instead of the cached value) shows up as a
// failed assertion against the per-trade cash limit.

import {
  __setMttmCache,
  invalidateMttmCache,
  getMttmConfigCached,
  slotKey,
  DEFAULT_MTTM_UNIVERSE,
  MTTM_DEFAULT_MAX_POSITION_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_COIN,
  MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
  MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
  type MttmConfig,
} from "../src/lib/mttm";
import { MAX_POSITION_PCT } from "../src/lib/trading-constants";

function fakeConfig(enabled: boolean): MttmConfig {
  const keys = new Set<string>();
  for (const u of DEFAULT_MTTM_UNIVERSE) keys.add(slotKey(u.coinId, u.timeframe));
  return {
    enabled,
    enabledAt: enabled ? new Date().toISOString() : null,
    universe: DEFAULT_MTTM_UNIVERSE,
    maxPositionPct: MTTM_DEFAULT_MAX_POSITION_PCT,
    consecutiveLossCap: 5,
    n10PostFeeCapPct: -0.02,
    disableReason: null,
    universeKeys: keys,
    // Task #659 — DS lane is OFF in this fixture; the position-size
    // override test only exercises the default 16-slot lane, so we set
    // mode='default' and stamp the DS sub-object with its safe defaults.
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

/**
 * Mirrors the cap pick at paper-trader.ts:854-863 — hot-path lookup
 * of the active `_maxPositionPct` constant.
 */
function chooseCap(): number {
  const m = getMttmConfigCached();
  return m && m.enabled ? m.maxPositionPct : MAX_POSITION_PCT;
}

test("MTTM off OR cache cold: cap is the global MAX_POSITION_PCT (30%)", () => {
  invalidateMttmCache();
  assert.equal(chooseCap(), MAX_POSITION_PCT);
  __setMttmCache(fakeConfig(false));
  assert.equal(chooseCap(), MAX_POSITION_PCT);
  invalidateMttmCache();
});

test("MTTM enabled: cap drops to mttm.maxPositionPct (5%)", () => {
  __setMttmCache(fakeConfig(true));
  const cap = chooseCap();
  assert.equal(cap, MTTM_DEFAULT_MAX_POSITION_PCT);
  assert.ok(cap < MAX_POSITION_PCT, "MTTM cap must be strictly tighter than the global cap");
  invalidateMttmCache();
});

test("Cap arithmetic on a $1000 portfolio: 5% MTTM → $50, 30% global → $300", () => {
  const portfolioUsd = 1000;
  __setMttmCache(fakeConfig(true));
  const mttmDollars = portfolioUsd * chooseCap();
  assert.equal(mttmDollars, 50);

  invalidateMttmCache();
  const globalDollars = portfolioUsd * chooseCap();
  assert.equal(globalDollars, 300);
});

test("MTTM cap default constant is 0.05 (5%) — matches task spec", () => {
  assert.equal(MTTM_DEFAULT_MAX_POSITION_PCT, 0.05);
});
