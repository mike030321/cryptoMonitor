/**
 * Task #381 step 6 — sizing clamp invariants.
 *
 * The composite multiplier (quant × brain) is bounded:
 *  - default: clamped to [0.5, 1.5] — neither factor can wipe the other.
 *  - defensive_mode === "hard": floor relaxes to 0.0 so the brain can
 *    route to suppression via mult=0; ceiling stays at 1.5.
 *
 * `getFamilySizeMultiplier` itself returns 1.0 (no-op) for neutral and
 * shadow directives so sizing is unaffected when the brain is
 * disabled or in shadow mode.
 */
import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";

import {
  __resetAdapterState,
  __setActiveDirectiveForTest,
  getFamilySizeMultiplier,
} from "../src/lib/meta-brain/adapter";
import { neutralDirective, type MetaBrainDirective } from "../src/lib/meta-brain/contract";

function makeDirective(overrides: Partial<MetaBrainDirective> = {}): MetaBrainDirective {
  return {
    tick_id: "real-uuid-clamp",
    trust_multiplier: {
      momentum: 1.0,
      mean_reversion: 1.0,
      breakout: 1.0,
      volatility_forecaster: 1.0,
      baseline: 1.0,
    },
    allocation_weight: {
      momentum: 0.2,
      mean_reversion: 0.2,
      breakout: 0.2,
      volatility_forecaster: 0.2,
      baseline: 0.2,
    },
    caution_level: 0.5,
    exploration_budget: 0.1,
    suppress_signal: false,
    defensive_mode: "off",
    suppressed_families: [],
    paused_slices: [],
    reason_codes: [],
    ...overrides,
  };
}

describe("meta-brain sizing clamp (Task #381)", () => {
  beforeEach(() => {
    __resetAdapterState();
  });

  it("neutral directive → multiplier 1.0", () => {
    __setActiveDirectiveForTest(neutralDirective("test"));
    assert.equal(getFamilySizeMultiplier("momentum"), 1.0);
  });

  it("shadow tick_id → multiplier 1.0", () => {
    __setActiveDirectiveForTest({ ...neutralDirective("test"), tick_id: "shadow:abc" });
    assert.equal(getFamilySizeMultiplier("momentum"), 1.0);
  });

  it("equal allocation, trust=1 → ~1.0 multiplier", () => {
    __setActiveDirectiveForTest(makeDirective());
    const m = getFamilySizeMultiplier("momentum");
    assert.ok(Math.abs(m - 1.0) < 1e-9, `expected ~1.0 got ${m}`);
  });

  it("over-weighted family → multiplier > 1", () => {
    __setActiveDirectiveForTest(
      makeDirective({
        allocation_weight: {
          momentum: 0.5,
          mean_reversion: 0.125,
          breakout: 0.125,
          volatility_forecaster: 0.125,
          baseline: 0.125,
        },
      }),
    );
    const m = getFamilySizeMultiplier("momentum");
    assert.ok(m > 1, `expected >1 got ${m}`);
  });

  it("soft defensive damps multiplier", () => {
    __setActiveDirectiveForTest(makeDirective({ defensive_mode: "soft" }));
    const m = getFamilySizeMultiplier("momentum");
    assert.ok(m < 1, `soft mode should damp, got ${m}`);
  });

  it("hard defensive zeroes multiplier", () => {
    __setActiveDirectiveForTest(makeDirective({ defensive_mode: "hard" }));
    assert.equal(getFamilySizeMultiplier("momentum"), 0);
  });
});
