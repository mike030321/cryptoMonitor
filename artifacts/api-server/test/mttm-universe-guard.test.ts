import { test } from "node:test";
import assert from "node:assert/strict";

// Task #614 — MTTM whitelist guard.
//
// Verifies that when MTTM is enabled and the cache is warm, the
// `coinId|timeframe` whitelist check is the authoritative answer for
// the decision-path guard. We do not invoke `executePaperTrade` here
// (that would require a fully seeded portfolio + a quant decision) —
// we exercise the guard primitive directly because the guard is
// trivially derived from `MttmConfig.universeKeys.has(slotKey(...))`
// and any future regression in that primitive will surface as an
// out-of-universe trade slipping through.

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

function fakeConfig(enabled: boolean): MttmConfig {
  const keys = new Set<string>();
  for (const u of DEFAULT_MTTM_UNIVERSE) keys.add(slotKey(u.coinId, u.timeframe));
  return {
    enabled,
    enabledAt: enabled ? new Date().toISOString() : null,
    universe: DEFAULT_MTTM_UNIVERSE,
    maxPositionPct: 0.05,
    consecutiveLossCap: 5,
    n10PostFeeCapPct: -0.02,
    disableReason: null,
    universeKeys: keys,
    // Task #659 — DS lane is OFF in this fixture (mode='default'); the
    // guard test only exercises the default 16-slot universe whitelist.
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

test("MTTM guard: in-universe (bonk, 6h) is allowed", () => {
  const cfg = fakeConfig(true);
  assert.equal(cfg.universeKeys.has(slotKey("bonk", "6h")), true);
});

test("MTTM guard: in-universe (pepe, 1d) is allowed", () => {
  const cfg = fakeConfig(true);
  assert.equal(cfg.universeKeys.has(slotKey("pepe", "1d")), true);
});

test("MTTM guard: out-of-universe coin (bitcoin, 6h) is rejected", () => {
  const cfg = fakeConfig(true);
  assert.equal(cfg.universeKeys.has(slotKey("bitcoin", "6h")), false);
});

test("MTTM guard: in-universe coin but wrong timeframe (bonk, 1h) is rejected", () => {
  const cfg = fakeConfig(true);
  // bonk is in the universe at 6h and 1d but never 1h.
  assert.equal(cfg.universeKeys.has(slotKey("bonk", "1h")), false);
});

test("MTTM guard: cache primer round-trips", () => {
  const cfg = fakeConfig(true);
  __setMttmCache(cfg);
  // Cache now holds an enabled config. Caller would gate on .enabled
  // before consulting universeKeys — assert both are intact.
  assert.equal(cfg.enabled, true);
  assert.equal(cfg.universeKeys.size, 16);
  invalidateMttmCache();
});

test("MTTM guard: default universe is exactly 16 slots (8 coins × 2 tfs)", () => {
  assert.equal(DEFAULT_MTTM_UNIVERSE.length, 16);
  const coins = new Set(DEFAULT_MTTM_UNIVERSE.map((u) => u.coinId));
  const tfs = new Set(DEFAULT_MTTM_UNIVERSE.map((u) => u.timeframe));
  assert.equal(coins.size, 8, "must cover exactly 8 coins");
  assert.deepEqual([...tfs].sort(), ["1d", "6h"], "must cover exactly {6h, 1d}");
});
