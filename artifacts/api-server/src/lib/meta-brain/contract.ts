/**
 * Market Meta-Brain ⇄ quant contract (TypeScript side).
 *
 * Single source of truth for the JSON payload exchanged with the
 * ml-engine's `/ml/meta-brain/evaluate` endpoint. Mirrors the vendored
 * package's `QuantBridge.from_payload` input contract and the
 * `BrainDirective.to_dict()` output plus our `tick_id` handle.
 *
 * Responses are Zod-validated on the boundary — malformed responses
 * trigger the neutral-directive fallback (see ./fallback.ts) and never
 * reach the trading path.
 *
 * Telemetry honesty (Task #381): per-slice and portfolio fields that
 * the api-server cannot truthfully compute yet are sent as numeric
 * `0` together with a `missing:<field>` flag in `anomaly_flags`. The
 * vendored Python `QuantSliceTelemetry` is a strict dataclass that
 * does not accept None, so we down-weight via the flag instead of
 * sending null. Real values are populated wherever the data already
 * exists (recent_accuracy, pnl_state, drawdown_state, exposure,
 * slippage_bps).
 */

import { z } from "zod";

export const STRATEGY_FAMILIES = [
  "momentum",
  "mean_reversion",
  "breakout",
  "volatility_forecaster",
  "baseline",
] as const;
export type StrategyFamily = (typeof STRATEGY_FAMILIES)[number];

// ────────────────────────── payload IN ───────────────────────────────

// Task #550 — per-timeframe role pass-through. Sourced from
// `shared/timeframe-roles.json` via `getRoleForTimeframe(timeframe)`
// at slice-collection time. The Python side does NOT yet consume this
// field — that's a separate downstream task. Adding the field now so
// the next task is purely a Python change. Mirrors the 4-role enum
// defined in `artifacts/api-server/src/lib/timeframe-roles.ts`; if a
// role is added there, this type must be updated too (and a missing-
// role default chosen on the Python side).
export const SLICE_ROLES = ["trade", "shadow", "context", "disabled"] as const;
export type SliceRole = (typeof SLICE_ROLES)[number];

export interface MetaBrainSlice {
  coin: string;
  timeframe: string;
  strategy_family: StrategyFamily;
  /** Task #550 — pass-through; consumption is a separate task. */
  slice_role: SliceRole;
  edge: number;
  confidence: number;
  calibrated_confidence: number;
  risk_score: number;
  recent_accuracy: number;
  pnl_state: number;
  drawdown_state: number;
  disagreement: number;
  prediction_error: number;
  regime: string;
  volatility: number;
  correlation_shift: number;
  exposure: number;
  turnover: number;
  slippage_bps: number;
  anomaly_flags: string[];
}

export interface MetaBrainPortfolio {
  total_drawdown: number;
  realized_vol: number;
  concentration: number;
  leverage: number;
  liquidity_stress: number;
  correlation_shift: number;
  active_risk_budget: number;
  kill_switch_distance: number;
  anomaly_flags: string[];
}

/** Task #390 — read-only governance benchmark. Strategy Lab → meta-
 * brain only. NEVER referenced by the quant predictor / `/ml/decide`
 * payload / training features / trade-decision rows. The forbidden-
 * prefix parity tests
 * (`no-llm-fields-in-trade-decisions.test.ts` /
 *  `no-llm-fields-runtime.test.ts`) keep the boundary honest. */
export interface MetaBrainBenchmark {
  aiReturn7d: number;
  bestBaselineReturn7d: number;
  relativeAlpha7d: number;
  relativeAlpha14d: number;
  drawdownRatioVsBest: number;
  sustainedUnderperformance: boolean;
  sampleCount: number;
  stale: boolean;
}

export interface MetaBrainBatch {
  slices: MetaBrainSlice[];
  portfolio: MetaBrainPortfolio;
  timestamp: string;
  /** Optional. Sent as a populated object only when Strategy Lab
   * snapshots are non-empty AND non-stale. Otherwise the adapter
   * sends an explicit `null` (NOT omission) so ml-engine's
   * `/ml/meta-brain/evaluate` can distinguish "Strategy Lab is
   * present but neutral this cycle" from "field never wired" —
   * either case yields the neutral-default governance signal. */
  benchmark?: MetaBrainBenchmark | null;
}

// ────────────────────────── payload OUT ──────────────────────────────

export const DEFENSIVE_MODES = ["off", "soft", "hard"] as const;
export type DefensiveMode = (typeof DEFENSIVE_MODES)[number];

// Task #381: the executing surface — fields whose values may affect the
// next tick's trade decisions — is exactly:
//   trust_multiplier, allocation_weight, defensive_mode,
//   suppress_signal, suppressed_families, paused_slices.
// Everything else (caution_level, exploration_budget, reason_codes,
// tick_id) is OBSERVABILITY-ONLY and must not be read by the
// adapter / paper-trader / monitor for sizing or gating. The
// `assertExecutingSurface` runtime guard in adapter.ts pins this so
// any future drift fails loud.
export const EXECUTING_DIRECTIVE_FIELDS = [
  "trust_multiplier",
  "allocation_weight",
  "defensive_mode",
  "suppress_signal",
  "suppressed_families",
  "paused_slices",
] as const;

export interface MetaBrainDirective {
  tick_id: string;
  trust_multiplier: Record<string, number>;
  allocation_weight: Record<string, number>;
  /** Observability-only; not consumed by execution. */
  caution_level: number;
  /** Observability-only; not consumed by execution. */
  exploration_budget: number;
  suppress_signal: boolean;
  defensive_mode: DefensiveMode;
  suppressed_families: string[];
  paused_slices: string[];
  /** Observability-only; not consumed by execution. */
  reason_codes: string[];
}

const bounded = (min: number, max: number) =>
  z
    .number()
    .min(min)
    .max(max)
    .refine((v) => Number.isFinite(v), "not_finite");

export const MetaBrainDirectiveSchema = z.object({
  tick_id: z.string().min(1),
  trust_multiplier: z.record(z.string(), bounded(0, 1.5)),
  allocation_weight: z.record(z.string(), bounded(0, 1.0)),
  caution_level: bounded(0, 1),
  exploration_budget: bounded(0, 0.1),
  suppress_signal: z.boolean(),
  defensive_mode: z.enum(DEFENSIVE_MODES),
  suppressed_families: z.array(z.string()),
  paused_slices: z.array(z.string()),
  reason_codes: z.array(z.string()),
});
export type MetaBrainDirectiveFromSchema = z.infer<typeof MetaBrainDirectiveSchema>;
const _typeCheck: MetaBrainDirective = {} as MetaBrainDirectiveFromSchema;
void _typeCheck;

export function allocationSumsToOne(
  allocation: Record<string, number>,
): boolean {
  const values = Object.values(allocation);
  if (values.length === 0) return true;
  const total = values.reduce((s, v) => s + v, 0);
  return Math.abs(total - 1.0) < 1e-6;
}

// ───────────────────── neutral / shadow directives ──────────────────
// Returned whenever the brain is disabled, unreachable, times out, or
// returns malformed output. Multiplies to 1.0 at the sizing hook,
// suppresses nothing, no defensive mode, no exploration. The live
// trader behaves exactly as it did before this module existed.
//
// Shadow directives (tick_id `shadow:<uuid>`) are produced by the
// client when META_BRAIN_SHADOW=1 and the brain returned a real
// directive: shape is neutral so sizing is unaffected, but the real
// uuid is preserved so record_outcome can still feed the brain's
// learning loop (the brain saw the entry, just didn't authorize the
// size shaping). The TICK_ID PREFIX is the design contract — see
// `isNeutralDirective` and `getFamilySizeMultiplier`.

export function neutralDirective(reason: string): MetaBrainDirective {
  return {
    tick_id: `neutral:${reason}`,
    trust_multiplier: {},
    allocation_weight: {},
    caution_level: 0,
    exploration_budget: 0,
    suppress_signal: false,
    defensive_mode: "off",
    suppressed_families: [],
    paused_slices: [],
    reason_codes: [`fallback:${reason}`],
  };
}

/** True iff the directive must NOT shape sizing or suppression.
 * Both `neutral:*` (brain failed/disabled) and `shadow:*` (brain ran
 * but we are in shadow mode) qualify. Sizing always uses this; only
 * record_outcome cares about the underlying uuid (so it skips
 * `neutral:*` and accepts `shadow:*`).
 */
export function isNeutralDirective(d: MetaBrainDirective): boolean {
  return (
    d.tick_id.startsWith("neutral:") || d.tick_id.startsWith("shadow:")
  );
}
