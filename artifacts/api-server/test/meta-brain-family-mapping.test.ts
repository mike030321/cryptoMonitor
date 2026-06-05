/**
 * Task #381 — strategy family resolution is the single source of truth
 * for (agent personality → strategy_family). Snapshot-style test so
 * any drift is caught.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { resolveStrategyFamily } from "../src/lib/meta-brain/adapter";

describe("resolveStrategyFamily (Task #381)", () => {
  const cases: Array<[string, string | undefined, string]> = [
    ["MomentumMike", undefined, "momentum"],
    ["TrendTrader", undefined, "momentum"],
    ["ContrarianCarol", undefined, "mean_reversion"],
    ["MeanReverter", undefined, "mean_reversion"],
    ["BreakoutBob", undefined, "breakout"],
    ["VolScalper", undefined, "volatility_forecaster"],
    ["Anonymous", undefined, "baseline"],
    ["", "momentum_specialist", "momentum"],
    ["", "mean_specialist", "mean_reversion"],
    ["", "breakout_head", "breakout"],
    ["", "vol_head", "volatility_forecaster"],
  ];
  for (const [personality, specialist, expected] of cases) {
    it(`(${personality || "_"}, ${specialist || "_"}) → ${expected}`, () => {
      assert.equal(resolveStrategyFamily(personality, specialist), expected);
    });
  }

  it("falls back to baseline on unknown input", () => {
    assert.equal(resolveStrategyFamily(null), "baseline");
    assert.equal(resolveStrategyFamily(undefined), "baseline");
    assert.equal(resolveStrategyFamily("xx"), "baseline");
  });
});
