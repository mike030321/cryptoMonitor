import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { defaultMinExpRetPct, getMinExpectedReturnPct } from "../src/lib/quant-brain";
import { getQuantEvGateRequiredPct } from "../src/lib/paper-trader";
import { QUANT_MIN_EXP_RET_PCT_FACTOR, ROUND_TRIP_COST_PCT } from "../src/lib/trading-constants";

/**
 * The brain's |expectedReturnPct| floor used to be locked to the
 * paper-trader's EV gate (`getMinEvVsCost() * round_trip_cost * 100`).
 * Task #94 enforced that lock so the brain wouldn't emit signals the
 * trader then drops as `quant_ev_below_costs`. But the two gate
 * fundamentally different quantities — the brain compares the model's
 * regression head (|expRet|) to the floor, the trader compares
 * `confidence × tp_distance_pct` to MIN_EV_VS_COST × round_trip_cost.
 * Their numeric equality was coincidental and made the brain vacuously
 * tight whenever the model's regression head was small.
 *
 * Task #130 decouples them: the brain's floor is now derived from
 * `quant_brain.decision_thresholds.min_expected_return_pct_factor` in
 * shared/trading-frictions.json, mirrored on the Python backtester via
 * app/backtest/contract.py. This test now asserts the *new* contract:
 *   1. The brain uses the JSON factor (not getMinEvVsCost).
 *   2. With no env override, getMinExpectedReturnPct == defaultMinExpRetPct.
 *   3. The floor is a positive, finite percent.
 *   4. The brain's floor is permissive enough to actually emit signals
 *      against the model's typical |expRet| range (~0.05–0.20%).
 */
describe("quant-brain |expRet| floor — JSON-driven, decoupled from trader EV gate", () => {
  it("brain's default floor = QUANT_MIN_EXP_RET_PCT_FACTOR × ROUND_TRIP_COST_PCT × 100", () => {
    const expected = QUANT_MIN_EXP_RET_PCT_FACTOR * ROUND_TRIP_COST_PCT * 100;
    assert.equal(defaultMinExpRetPct(), expected);
  });

  it("with no QUANT_MIN_EXP_RET_PCT env override, getMinExpectedReturnPct == defaultMinExpRetPct", () => {
    assert.equal(process.env.QUANT_MIN_EXP_RET_PCT, undefined);
    assert.equal(getMinExpectedReturnPct(), defaultMinExpRetPct());
  });

  it("threshold is a positive finite percent (sanity)", () => {
    const v = getMinExpectedReturnPct();
    assert.ok(Number.isFinite(v) && v > 0, `expected positive finite percent, got ${v}`);
  });

  it("calibrated floor (v3) sits below the trader EV gate so the brain is the looser side", () => {
    // Brain: factor × round_trip × 100. Trader EV gate (different formula:
    // confidence × tp_distance) shares the round-trip cost units. As of v3
    // factor=0.5 and MIN_EV_VS_COST=3.0, so brain ≤ trader_gate. If anyone
    // bumps the brain's factor above MIN_EV_VS_COST without re-running the
    // calibration study, this test fires — that combination would re-create
    // the task #130 failure where the brain blocks 100% of signals.
    assert.ok(
      defaultMinExpRetPct() <= getQuantEvGateRequiredPct(),
      `brain floor ${defaultMinExpRetPct()}% must stay ≤ trader EV gate ${getQuantEvGateRequiredPct()}% — see shared/trading-frictions.json#quant_brain.decision_thresholds._doc`,
    );
  });
});
