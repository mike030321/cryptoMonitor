/**
 * Task #381 — invariant: the meta-brain never inverts trade direction.
 * The brain only emits trust/allocation/suppression. Direction is
 * decided upstream by the quant brain. This test verifies the
 * adapter exposes no direction-mutating surface and that suppression
 * is the strongest signal it can emit (via mult=0 in hard mode and
 * via isFamilySuppressed).
 */
import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";

import {
  __resetAdapterState,
  __setActiveDirectiveForTest,
  getFamilySizeMultiplier,
  isFamilySuppressed,
} from "../src/lib/meta-brain/adapter";
import { neutralDirective, type MetaBrainDirective } from "../src/lib/meta-brain/contract";

const base: MetaBrainDirective = {
  tick_id: "real-uuid-dir",
  trust_multiplier: {
    momentum: 1.5, mean_reversion: 0.5, breakout: 1.0,
    volatility_forecaster: 1.0, baseline: 1.0,
  },
  allocation_weight: {
    momentum: 0.4, mean_reversion: 0.15, breakout: 0.15,
    volatility_forecaster: 0.15, baseline: 0.15,
  },
  caution_level: 0.7,
  exploration_budget: 0.1,
  suppress_signal: false,
  defensive_mode: "off",
  suppressed_families: [],
  paused_slices: [],
  reason_codes: [],
};

describe("meta-brain direction invariant (Task #381)", () => {
  beforeEach(() => __resetAdapterState());

  it("multiplier is always non-negative — brain cannot flip direction sign", () => {
    for (const mode of ["off", "soft", "hard"] as const) {
      __setActiveDirectiveForTest({ ...base, defensive_mode: mode });
      for (const fam of [
        "momentum",
        "mean_reversion",
        "breakout",
        "volatility_forecaster",
        "baseline",
      ] as const) {
        const m = getFamilySizeMultiplier(fam);
        assert.ok(m >= 0, `mode=${mode} family=${fam} mult=${m} must be >=0`);
      }
    }
  });

  it("suppress_signal suppresses every family", () => {
    __setActiveDirectiveForTest({ ...base, suppress_signal: true });
    for (const fam of [
      "momentum",
      "mean_reversion",
      "breakout",
      "volatility_forecaster",
      "baseline",
    ] as const) {
      assert.equal(isFamilySuppressed(fam), true, `${fam} should be suppressed`);
    }
  });

  it("paused_slices match (coin,timeframe)", () => {
    __setActiveDirectiveForTest({ ...base, paused_slices: ["btc/5m"] });
    assert.equal(isFamilySuppressed("momentum", "btc", "5m"), true);
    assert.equal(isFamilySuppressed("momentum", "btc", "1h"), false);
    assert.equal(isFamilySuppressed("momentum", "eth", "5m"), false);
  });

  it("neutral / shadow → never suppressed", () => {
    __setActiveDirectiveForTest(neutralDirective("test"));
    assert.equal(isFamilySuppressed("momentum"), false);
    __setActiveDirectiveForTest({
      ...neutralDirective("test"),
      tick_id: "shadow:xyz",
      suppress_signal: true,
    });
    // Shadow tick_id passes isNeutralDirective so suppression cannot fire.
    assert.equal(isFamilySuppressed("momentum"), false);
  });
});
