import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  MAKER_FEE_PCT,
  TAKER_FEE_PCT,
  SLIPPAGE_PCT,
  ROUND_TRIP_COST_PCT,
  OUTCOME_THRESHOLDS_PERCENT,
  getOutcomeThresholdPercent,
  INITIAL_BALANCE_USD,
  MAX_OPEN_POSITIONS_PER_AGENT,
  MAX_PORTFOLIO_AT_RISK,
  DAILY_LOSS_LIMIT_PCT,
  DRAWDOWN_HALT_PCT,
  MAX_POSITION_PCT,
  KELLY_MIN_TRADES,
  KELLY_RAMP_END,
  KELLY_FRACTION,
  ASYMMETRIC_LONG_MIN_CONFIDENCE,
  TRADEABLE_TIMEFRAMES_SET,
  FLEET_BRAKE_MIN_OPEN,
  FLEET_BRAKE_DOMINANCE,
  RECENT_LOSS_BLOCK_COUNT,
  GATES_BASELINE,
  TRAILING_STOP,
  tieredPositionPct,
  getSlMultiplier,
  getTpMultiplier,
  getAtrFloorPct,
} from "../src/lib/trading-constants";

// Load the JSON contract directly so the test asserts that the in-process
// constants are the SAME bytes the Python backtester reads. Drift here =
// silent divergence between live trader and backtester = correctness bug.
function loadContract(): any {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const root = path.resolve(here, "..", "..", "..");
  return JSON.parse(
    readFileSync(path.join(root, "shared", "trading-frictions.json"), "utf8"),
  );
}

test("trading-frictions: file exists and parses", () => {
  const c = loadContract();
  assert.equal(typeof c, "object");
  assert.ok(c.fees);
});

test("trading-frictions: fees + slippage match JSON", () => {
  const c = loadContract();
  assert.equal(MAKER_FEE_PCT, c.fees.maker_fee_pct);
  assert.equal(TAKER_FEE_PCT, c.fees.taker_fee_pct);
  assert.equal(SLIPPAGE_PCT,  c.fees.slippage_pct);
  assert.equal(
    ROUND_TRIP_COST_PCT,
    2 * (c.fees.taker_fee_pct + c.fees.slippage_pct),
  );
});

test("trading-frictions: outcome thresholds match JSON", () => {
  const c = loadContract();
  for (const tf of ["1m", "5m", "1h", "2h", "6h", "1d"]) {
    assert.equal(
      OUTCOME_THRESHOLDS_PERCENT[tf],
      c.outcome_thresholds_percent[tf],
      `mismatch for ${tf}`,
    );
    assert.equal(getOutcomeThresholdPercent(tf), c.outcome_thresholds_percent[tf]);
  }
  assert.equal(getOutcomeThresholdPercent("__missing__"), c.outcome_thresholds_percent._default);
});

test("trading-frictions: risk constants match JSON", () => {
  const c = loadContract();
  assert.equal(INITIAL_BALANCE_USD,            c.risk.initial_balance_usd);
  assert.equal(MAX_OPEN_POSITIONS_PER_AGENT,   c.risk.max_open_positions_per_agent);
  assert.equal(MAX_PORTFOLIO_AT_RISK,          c.risk.max_portfolio_at_risk);
  assert.equal(DAILY_LOSS_LIMIT_PCT,           c.risk.daily_loss_limit_pct);
  assert.equal(DRAWDOWN_HALT_PCT,              c.risk.drawdown_halt_pct);
  assert.equal(MAX_POSITION_PCT,               c.risk.max_position_pct);
  assert.equal(KELLY_MIN_TRADES,               c.risk.kelly_min_trades);
  assert.equal(KELLY_RAMP_END,                 c.risk.kelly_ramp_end);
  assert.equal(KELLY_FRACTION,                 c.risk.kelly_fraction);
  assert.equal(ASYMMETRIC_LONG_MIN_CONFIDENCE, c.risk.asymmetric_long_min_confidence);
});

test("trading-frictions: tiered position pct matches JSON", () => {
  const c = loadContract();
  // Asserts both the data and the lookup function. JSON tiers are sorted
  // descending by min_confidence.
  const sorted = [...c.tiered_position_pct].sort(
    (a: any, b: any) => b.min_confidence - a.min_confidence,
  );
  for (const tier of sorted) {
    assert.equal(tieredPositionPct(tier.min_confidence), tier.pct);
  }
  // Below the lowest tier, returns the lowest pct.
  assert.equal(tieredPositionPct(0), sorted[sorted.length - 1].pct);
});

test("trading-frictions: fleet brake + recent-loss block match JSON", () => {
  const c = loadContract();
  assert.equal(FLEET_BRAKE_MIN_OPEN,    c.fleet_brake.min_open_positions);
  assert.equal(FLEET_BRAKE_DOMINANCE,   c.fleet_brake.same_side_dominance_share);
  assert.equal(RECENT_LOSS_BLOCK_COUNT, c.recent_loss_block.consecutive_losses);
});

test("trading-frictions: tradeable timeframes match JSON", () => {
  const c = loadContract();
  assert.deepEqual(
    Array.from(TRADEABLE_TIMEFRAMES_SET).sort(),
    [...c.tradeable_timeframes].sort(),
  );
});

test("trading-frictions: SL/TP/ATR-floor lookups match JSON", () => {
  const c = loadContract();
  for (const tf of c.tradeable_timeframes) {
    assert.equal(getSlMultiplier(tf),  c.tf_sl_multiplier[tf],   `sl mult ${tf}`);
    assert.equal(getTpMultiplier(tf),  c.tf_tp_multiplier[tf],   `tp mult ${tf}`);
    assert.equal(getAtrFloorPct(tf),   c.tf_atr_floor_pct[tf],   `atr floor ${tf}`);
  }
  assert.equal(getSlMultiplier("__none__"), c.tf_sl_multiplier._default);
  assert.equal(getTpMultiplier("__none__"), c.tf_tp_multiplier._default);
  assert.equal(getAtrFloorPct("__none__"),  c.tf_atr_floor_pct._default);
});

test("trading-frictions: trailing stop constants match JSON", () => {
  const c = loadContract();
  assert.equal(TRAILING_STOP.minPeakPnlPctToTrail,
               c.trailing_stop.min_peak_pnl_pct_to_trail);
  assert.equal(TRAILING_STOP.trailGivebackFraction,
               c.trailing_stop.trail_giveback_fraction);
  for (const tf of Object.keys(c.trailing_stop.expiry_extension_ms)) {
    assert.equal(
      TRAILING_STOP.expiryExtensionMs[tf],
      c.trailing_stop.expiry_extension_ms[tf],
      `extension ms ${tf}`,
    );
  }
});

test("trading-frictions: gates baseline matches JSON", () => {
  const c = loadContract();
  for (const k of Object.keys(c.gates_baseline)) {
    assert.equal(GATES_BASELINE[k].value, c.gates_baseline[k].value, `gate ${k} value`);
    assert.equal(GATES_BASELINE[k].floor, c.gates_baseline[k].floor, `gate ${k} floor`);
  }
});
