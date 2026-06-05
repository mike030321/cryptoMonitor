// All numeric trading constants live in `shared/trading-frictions.json` at
// the workspace root. This module is the TypeScript adapter — it loads the
// file once at startup and re-exports the values under their existing names
// so callers (paper-trader, calibration loop, tests) don't need to change.
//
// The Python backtester (artifacts/ml-engine/app/backtest/contract.py) loads
// the same JSON file. Drift between the two sides is a correctness bug —
// adjust the JSON, never the language-specific constants.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

interface Frictions {
  fees: { maker_fee_pct: number; taker_fee_pct: number; slippage_pct: number };
  quant_brain?: {
    enabled: boolean;
    auto_revert: {
      consecutive_drift_cycles: number;
      drift_share_threshold: number;
      min_evaluable_coins: number;
    };
    decision_thresholds?: {
      min_directional_prob: number;
      min_directional_edge: number;
      min_expected_return_pct_factor: number;
      policy_version: string;
    };
  };
  outcome_thresholds_percent: Record<string, number>;
  risk: {
    initial_balance_usd: number;
    max_open_positions_per_agent: number;
    max_portfolio_at_risk: number;
    daily_loss_limit_pct: number;
    drawdown_halt_pct: number;
    max_position_pct: number;
    default_position_pct: number;
    kelly_min_trades: number;
    kelly_ramp_end: number;
    kelly_fraction: number;
    asymmetric_long_min_confidence: number;
  };
  tiered_position_pct: { min_confidence: number; pct: number }[];
  fleet_brake: { min_open_positions: number; same_side_dominance_share: number };
  recent_loss_block: { consecutive_losses: number };
  tradeable_timeframes: string[];
  timeframe_ms: Record<string, number>;
  tf_sl_multiplier: Record<string, number>;
  tf_tp_multiplier: Record<string, number>;
  tf_atr_floor_pct: Record<string, number>;
  trailing_stop: {
    min_peak_pnl_pct_to_trail: number;
    trail_giveback_fraction: number;
    expiry_extension_ms: Record<string, number>;
  };
  gates_baseline: Record<string, { value: number; floor: number }>;
  regime_classifier: unknown;
  backtest_deploy_gate: unknown;
}

function findContractPath(): string {
  // Resolve relative to this source file. tsx (test runner) preserves
  // import.meta.url against the .ts source; esbuild bundles this module
  // into dist/index.mjs, in which case import.meta.url points at dist/.
  // Walk upward looking for `pnpm-workspace.yaml`, then read the file.
  const here = path.dirname(fileURLToPath(import.meta.url));
  let cur = here;
  for (let i = 0; i < 8; i++) {
    try {
      const candidate = path.join(cur, "pnpm-workspace.yaml");
      readFileSync(candidate);
      return path.join(cur, "shared", "trading-frictions.json");
    } catch {
      const parent = path.dirname(cur);
      if (parent === cur) break;
      cur = parent;
    }
  }
  throw new Error(
    `[trading-constants] could not locate workspace root (started at ${here})`,
  );
}

const FRICTIONS: Frictions = JSON.parse(readFileSync(findContractPath(), "utf8"));

// ── Fees & slippage ────────────────────────────────────────────────────────
export const MAKER_FEE_PCT = FRICTIONS.fees.maker_fee_pct;
export const TAKER_FEE_PCT = FRICTIONS.fees.taker_fee_pct;
export const SLIPPAGE_PCT  = FRICTIONS.fees.slippage_pct;

export const ROUND_TRIP_COST_PCT = 2 * (TAKER_FEE_PCT + SLIPPAGE_PCT);
export const ROUND_TRIP_COST_PERCENT = ROUND_TRIP_COST_PCT * 100;

// ── Outcome thresholds (percent) ──────────────────────────────────────────
// Every threshold is set STRICTLY above ROUND_TRIP_COST_PERCENT (0.30%) so a
// "correct" prediction is also net-profitable after fees + slippage. This is
// load-bearing for the calibration loop.
export const OUTCOME_THRESHOLDS_PERCENT: Record<string, number> =
  Object.fromEntries(
    Object.entries(FRICTIONS.outcome_thresholds_percent).filter(
      ([k]) => !k.startsWith("_"),
    ),
  );

// Task #349 — fail-fast per-timeframe lookup. The JSON `_default` cascade
// is allowed; falling back to a TS-side hard-coded literal is forbidden so
// a missing `_default` in shared/trading-frictions.json cannot silently
// substitute a number that the Python backtester would never see. Mirrors
// `_require_tf_lookup` in app/backtest/contract.py.
export function _requireTfLookup(
  map: Record<string, number>,
  tf: string,
  ctx: string,
): number {
  const v = map[tf];
  if (v !== undefined && v !== null) return v;
  const dflt = map._default;
  if (dflt !== undefined && dflt !== null) return dflt;
  throw new Error(
    `[trading-constants] '${ctx}' has no entry for tf='${tf}' and no ` +
    `'_default' key in shared/trading-frictions.json. Refusing to fall ` +
    `back to a TS-side literal — see task #349.`,
  );
}
export function getOutcomeThresholdPercent(timeframe: string): number {
  return _requireTfLookup(
    FRICTIONS.outcome_thresholds_percent,
    timeframe,
    "outcome_thresholds_percent",
  );
}

// ── Risk / position sizing ────────────────────────────────────────────────
export const INITIAL_BALANCE_USD = FRICTIONS.risk.initial_balance_usd;
export const MAX_OPEN_POSITIONS_PER_AGENT = FRICTIONS.risk.max_open_positions_per_agent;
export const MAX_PORTFOLIO_AT_RISK = FRICTIONS.risk.max_portfolio_at_risk;
export const DAILY_LOSS_LIMIT_PCT = FRICTIONS.risk.daily_loss_limit_pct;
export const DRAWDOWN_HALT_PCT = FRICTIONS.risk.drawdown_halt_pct;
export const MAX_POSITION_PCT = FRICTIONS.risk.max_position_pct;
export const DEFAULT_POSITION_PCT = FRICTIONS.risk.default_position_pct;
export const KELLY_MIN_TRADES = FRICTIONS.risk.kelly_min_trades;
export const KELLY_RAMP_END = FRICTIONS.risk.kelly_ramp_end;
export const KELLY_FRACTION = FRICTIONS.risk.kelly_fraction;
export const ASYMMETRIC_LONG_MIN_CONFIDENCE = FRICTIONS.risk.asymmetric_long_min_confidence;

// Tiered position sizing — sorted descending by min_confidence.
const TIERED = [...FRICTIONS.tiered_position_pct].sort(
  (a, b) => b.min_confidence - a.min_confidence,
);
export function tieredPositionPct(confidence: number): number {
  for (const t of TIERED) if (confidence >= t.min_confidence) return t.pct;
  return TIERED[TIERED.length - 1].pct;
}

// ── Fleet correlation brake ───────────────────────────────────────────────
export const FLEET_BRAKE_MIN_OPEN = FRICTIONS.fleet_brake.min_open_positions;
export const FLEET_BRAKE_DOMINANCE = FRICTIONS.fleet_brake.same_side_dominance_share;

// ── Recent-loss block ─────────────────────────────────────────────────────
export const RECENT_LOSS_BLOCK_COUNT = FRICTIONS.recent_loss_block.consecutive_losses;

// ── Timeframes ────────────────────────────────────────────────────────────
export const TRADEABLE_TIMEFRAMES_SET: Set<string> = new Set(FRICTIONS.tradeable_timeframes);

// ── ATR / SL / TP geometry ────────────────────────────────────────────────
// Task #349 — fail-fast. The previous `tfLookup(map, tf, fallback)` helper
// took a TS-side literal as a third arg (1.8 / 3.0 / 0.008). If the JSON
// lost both the per-tf entry AND its `_default`, the live trader silently
// switched to a multiplier the Python backtester would never have used —
// breaking the SL/TP/ATR-floor parity contract. We now require either the
// per-tf entry or the JSON `_default`, and throw otherwise.
export function getSlMultiplier(tf: string): number {
  return _requireTfLookup(FRICTIONS.tf_sl_multiplier, tf, "tf_sl_multiplier");
}
export function getTpMultiplier(tf: string): number {
  return _requireTfLookup(FRICTIONS.tf_tp_multiplier, tf, "tf_tp_multiplier");
}
export function getAtrFloorPct(tf: string): number {
  return _requireTfLookup(FRICTIONS.tf_atr_floor_pct, tf, "tf_atr_floor_pct");
}

// ── Trailing stop ─────────────────────────────────────────────────────────
export const TRAILING_STOP = {
  minPeakPnlPctToTrail: FRICTIONS.trailing_stop.min_peak_pnl_pct_to_trail,
  trailGivebackFraction: FRICTIONS.trailing_stop.trail_giveback_fraction,
  expiryExtensionMs: FRICTIONS.trailing_stop.expiry_extension_ms,
};

// ── Gates baseline (consumed by tuning-tracker) ───────────────────────────
export const GATES_BASELINE = FRICTIONS.gates_baseline;

// ── Outcome judgement (shared between LLM live resolver and shadow resolver) ─
// Pure function — same neutral-zone band logic used by resolvePendingPredictions.
// Returns the same shape so both callers can apply identical adjudication.
export type Direction = "up" | "down" | "stable";
export interface DirectionJudgement { correct: boolean; neutral: boolean }

export function judgeDirection(
  direction: Direction,
  priceChangePct: number,
  timeframe: string,
): DirectionJudgement {
  const threshold = getOutcomeThresholdPercent(timeframe);
  const neutralMultiplier = ["1h", "2h", "6h", "1d"].includes(timeframe) ? 1.2 : 1.0;
  const neutralThreshold = threshold * neutralMultiplier;
  if (direction === "up") {
    if (priceChangePct > threshold) return { correct: true, neutral: false };
    if (Math.abs(priceChangePct) < neutralThreshold && priceChangePct >= 0) return { correct: false, neutral: true };
  } else if (direction === "down") {
    if (priceChangePct < -threshold) return { correct: true, neutral: false };
    if (Math.abs(priceChangePct) < neutralThreshold && priceChangePct <= 0) return { correct: false, neutral: true };
  } else {
    if (Math.abs(priceChangePct) < threshold) return { correct: true, neutral: false };
  }
  return { correct: false, neutral: false };
}

export const RESOLVE_GRACE_PERIOD_MS = 5 * 60 * 1000;

// Stuck-cycle timeout (forceUnlockCycle) and per-position cash cap.
export const CYCLE_STUCK_TIMEOUT_MS = 4 * 60 * 1000;
export const MAX_CASH_PER_POSITION_PCT = 0.80;

// Horizon weak-signal thresholds (shared by /crypto/brain/accuracy
// response, dashboard DISABLED badge, and horizon-gate trade skip).
export const HORIZON_WEAK_DIRECTIONAL_FLOOR_PCT = 45;
export const HORIZON_WEAK_MIN_RESOLVED = 50;

// ── Phase 5 quant brain config ────────────────────────────────────────────
// `QUANT_BRAIN_ENABLED` is the persisted default that brain-flag.ts uses
// when no row has been written to app_settings.quant_brain_enabled yet.
// The DB row, if present, overrides this constant; the env var
// QUANT_BRAIN_FORCE_OFF=1 overrides everything (final kill-switch).
//
// `QUANT_BRAIN_AUTO_REVERT` controls the safety guard that disables the
// quant brain when the live shadow tracker reports drift_low on the majority
// of evaluable coins for `consecutive_drift_cycles` cycles in a row. Tune all
// three knobs in shared/trading-frictions.json — never here.
// Task #343 — fail-fast loader. The previous version silently fell back
// to a hard-coded {enabled:false, ...} when `quant_brain` was missing
// from the JSON, which let a config-management bug (typo, deleted key,
// wrong file path) ship to prod looking like the brain was just turned
// off on purpose. We now raise at module load time so the bad config is
// impossible to miss.
export function _requireConfig<T>(value: T | undefined | null, key: string): T {
  if (value === undefined || value === null) {
    throw new Error(
      `[trading-constants] required key '${key}' missing from ` +
      `shared/trading-frictions.json. Refusing to start with a silent ` +
      `default — see task #343.`,
    );
  }
  return value;
}

// Task #356 — exposed for unit tests so the missing-key throws can be
// exercised against mutated copies of the contract without spawning a
// subprocess to reload the module. Mirrors EXACTLY the inline
// `_requireConfig` calls in the module-level wiring below; if a check is
// added/removed at the top level, update this function too.
export function assertContractRequiredKeys(raw: unknown): void {
  const r =
    raw && typeof raw === "object"
      ? (raw as Record<string, unknown>)
      : {};
  const qb = _requireConfig(
    r.quant_brain as Record<string, unknown> | undefined,
    "quant_brain",
  );
  _requireConfig(qb.enabled as boolean | undefined, "quant_brain.enabled");
  _requireConfig(
    qb.auto_revert as Record<string, unknown> | undefined,
    "quant_brain.auto_revert",
  );
  const qd = _requireConfig(
    qb.decision_thresholds as Record<string, unknown> | undefined,
    "quant_brain.decision_thresholds",
  );
  _requireConfig(
    qd.min_directional_prob as number | undefined,
    "quant_brain.decision_thresholds.min_directional_prob",
  );
  _requireConfig(
    qd.min_directional_edge as number | undefined,
    "quant_brain.decision_thresholds.min_directional_edge",
  );
  _requireConfig(
    qd.min_expected_return_pct_factor as number | undefined,
    "quant_brain.decision_thresholds.min_expected_return_pct_factor",
  );
  _requireConfig(
    qd.policy_version as string | undefined,
    "quant_brain.decision_thresholds.policy_version",
  );
}
const _quantBrain = _requireConfig(FRICTIONS.quant_brain, "quant_brain");
export const QUANT_BRAIN_ENABLED: boolean = _requireConfig(
  _quantBrain.enabled, "quant_brain.enabled",
);
export const QUANT_BRAIN_AUTO_REVERT = _requireConfig(
  _quantBrain.auto_revert, "quant_brain.auto_revert",
);

// Decision-rule thresholds — the live brain (quant-brain.ts) and the
// Python backtester (artifacts/ml-engine/app/backtest/simulator.py) BOTH
// read these from the same shared/trading-frictions.json so any tuning
// change updates both sides at once. Drift between the two is a
// correctness bug. Env vars (QUANT_MIN_DIR_PROB, QUANT_MIN_DIR_EDGE) still
// override at runtime for diagnostics — see quant-brain.ts.
const _quantDecision = _requireConfig(
  _quantBrain.decision_thresholds, "quant_brain.decision_thresholds",
);
export const QUANT_MIN_DIRECTIONAL_PROB: number = _requireConfig(
  _quantDecision.min_directional_prob,
  "quant_brain.decision_thresholds.min_directional_prob",
);
export const QUANT_MIN_DIRECTIONAL_EDGE: number = _requireConfig(
  _quantDecision.min_directional_edge,
  "quant_brain.decision_thresholds.min_directional_edge",
);
// Multiplier on `ROUND_TRIP_COST_PCT * 100` that derives the brain's
// |expectedReturnPct| floor. Calibrated 2026-04-22 (task #130) — see the
// _doc field of quant_brain.decision_thresholds in
// shared/trading-frictions.json. The Python backtester reads the same JSON
// field via app/backtest/contract.py so live and offline behaviour stay
// in lock-step.
export const QUANT_MIN_EXP_RET_PCT_FACTOR: number = _requireConfig(
  _quantDecision.min_expected_return_pct_factor,
  "quant_brain.decision_thresholds.min_expected_return_pct_factor",
);
export const QUANT_POLICY_VERSION: string = _requireConfig(
  _quantDecision.policy_version,
  "quant_brain.decision_thresholds.policy_version",
);

// ── Fleet reset cutoff (TS-only metadata; not part of JSON contract) ──────
// Every prediction created STRICTLY before this timestamp was produced with
// the wrong sign convention, sub-cost outcome thresholds, or synthetic price
// data. The single source of truth that any calibration / accuracy /
// pattern-learning query must use to scope itself to the post-reset run.
export const PREDICTION_FLEET_RESET_AT = new Date("2026-04-21T00:00:00.000Z");
