// One-shot script that mirrors the live quant-brain emission rule
// (artifacts/api-server/src/lib/quant-brain.ts:142-172) on a fixed
// probability/expectedReturnPct grid and emits the expected outputs
// as JSON. The Python parity test (test_backtest.py::test_decide_direction_matches_ts_reference)
// loads the same fixture and asserts the Python implementation produces
// identical decisions for every row.
//
// Run: node artifacts/ml-engine/tests/_gen_quant_brain_reference.mjs > artifacts/ml-engine/tests/fixtures/quant_brain_parity.json
//
// The math here is copy-pasted line-for-line from quant-brain.ts so any
// future drift in the TS rule is caught by re-running this script and
// regenerating the fixture.

import { readFileSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

function findContractPath() {
  const here = path.dirname(fileURLToPath(import.meta.url));
  let cur = here;
  for (let i = 0; i < 8; i++) {
    if (existsSync(path.join(cur, "pnpm-workspace.yaml"))) {
      return path.join(cur, "shared", "trading-frictions.json");
    }
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  throw new Error("could not locate workspace root");
}

const contract = JSON.parse(readFileSync(findContractPath(), "utf8"));
const dt = contract.quant_brain.decision_thresholds;
const MIN_DIRECTIONAL_PROB = dt.min_directional_prob;
const MIN_DIRECTIONAL_EDGE = dt.min_directional_edge;
const ROUND_TRIP_COST_PCT = 2 * (contract.fees.taker_fee_pct + contract.fees.slippage_pct);
// As of policy v3 (task #130) the brain's |expRet| floor is derived from the
// dedicated `min_expected_return_pct_factor` knob, NOT gates_baseline.
// MIN_EV_VS_COST: those two were "by construction" equal in v2, but v2's
// 0.9% floor blocked 100% of signals (model |expRet| caps at ~0.19%) which
// made the backtest report vacuous. The factor is now tunable independently.
const MIN_EXP_RET_FACTOR = dt.min_expected_return_pct_factor;
const MIN_EXPECTED_RETURN_PCT = MIN_EXP_RET_FACTOR * ROUND_TRIP_COST_PCT * 100;

// Mirror of quant-brain.ts:142-172 — directional preference (NOT argmax).
function decide(probDown, probStable, probUp, expectedReturnPct) {
  const dirSide = probUp >= probDown ? "up" : "down";
  const dirProb = Math.max(probUp, probDown);
  const dirEdge = Math.abs(probUp - probDown);
  const expRetSign = expectedReturnPct > 0 ? "up" : expectedReturnPct < 0 ? "down" : "stable";

  if (dirProb < MIN_DIRECTIONAL_PROB) {
    return { side: null, confidence: probStable, reason: "abstain_low_directional_prob" };
  }
  if (dirEdge < MIN_DIRECTIONAL_EDGE) {
    return { side: null, confidence: probStable, reason: "abstain_no_directional_edge" };
  }
  if (Math.abs(expectedReturnPct) < MIN_EXPECTED_RETURN_PCT) {
    return { side: null, confidence: probStable, reason: "abstain_exp_ret_below_cost" };
  }
  if (expRetSign !== dirSide) {
    return { side: null, confidence: probStable, reason: "abstain_exp_ret_disagrees" };
  }
  const confidence = dirSide === "up" ? probUp : probDown;
  return { side: dirSide, confidence, reason: null };
}

// Representative grid: covers the four abstain branches plus the emit
// path for both directions, plus boundary conditions.
function gridRow(p_down, p_stable, p_up, exp_ret_pct, label) {
  const r = decide(p_down, p_stable, p_up, exp_ret_pct);
  return {
    label,
    p_down, p_stable, p_up, expected_return_pct: exp_ret_pct,
    expected_side: r.side,
    expected_confidence: r.confidence,
    expected_reason: r.reason,
  };
}

const cases = [
  // Emit path
  gridRow(0.10, 0.35, 0.55,  1.50, "emit_up_strong"),
  gridRow(0.55, 0.35, 0.10, -1.50, "emit_down_strong"),
  gridRow(0.03, 0.85, 0.12,  1.50, "emit_up_thin_edge"),       // edge=0.09 ≥ 0.05, p_up=0.12 ≥ 0.08
  gridRow(0.20, 0.65, 0.07, -1.50, "emit_down_thin_edge"),     // edge=0.13 ≥ 0.05, p_down=0.20 ≥ 0.08
  // 1) MIN_DIRECTIONAL_PROB fail
  gridRow(0.05, 0.92, 0.03,  1.50, "abstain_low_dir_prob_up_max"),
  gridRow(0.07, 0.86, 0.07,  0.16, "abstain_low_dir_prob_tie"),
  // 2) MIN_DIRECTIONAL_EDGE fail
  gridRow(0.40, 0.20, 0.40,  1.50, "abstain_no_edge_equal"),
  gridRow(0.30, 0.39, 0.31,  1.50, "abstain_no_edge_close"),    // edge=0.01 < 0.05
  gridRow(0.10, 0.78, 0.12,  1.50, "abstain_no_edge_just_below"), // edge=0.02 < 0.05
  // 3) |expRet| below cost (floor = 0.5 × round_trip × 100 = 0.15%)
  gridRow(0.10, 0.35, 0.55,  0.05, "abstain_exp_ret_tiny_pos"),
  gridRow(0.55, 0.35, 0.10, -0.05, "abstain_exp_ret_tiny_neg"),
  gridRow(0.10, 0.35, 0.55,  0.0,  "abstain_exp_ret_zero"),
  gridRow(0.10, 0.35, 0.55,  0.14, "abstain_exp_ret_just_below_floor"),
  // 4) expRet sign disagrees
  gridRow(0.10, 0.35, 0.55, -1.50, "abstain_disagree_up_negret"),
  gridRow(0.55, 0.35, 0.10,  1.50, "abstain_disagree_down_posret"),
  // Boundary: dirProb exactly at threshold, edge exactly at threshold
  gridRow(0.03, 0.84, 0.08,  0.16, "boundary_dir_prob_at_threshold"), // dirProb=0.08, edge=0.05
  gridRow(0.08, 0.79, 0.13,  0.16, "boundary_dir_edge_at_threshold"), // dirProb=0.13, edge=0.05
  // Boundary: |expRet| exactly at floor (0.5 * 0.003 * 100 = 0.15)
  gridRow(0.10, 0.35, 0.55,  0.15, "boundary_exp_ret_at_floor"),
];

const out = {
  policy_version: dt.policy_version,
  thresholds: {
    min_directional_prob: MIN_DIRECTIONAL_PROB,
    min_directional_edge: MIN_DIRECTIONAL_EDGE,
    min_expected_return_pct: MIN_EXPECTED_RETURN_PCT,
  },
  cases,
};

console.log(JSON.stringify(out, null, 2));
