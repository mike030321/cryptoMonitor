/**
 * Typed client for the Python ML engine sidecar (Phase 1 — wiring only).
 *
 * The ml-engine artifact runs as a separate process inside the same workspace
 * container, listening on `ML_ENGINE_URL` (default http://localhost:8000) and
 * mounting all routes under `/ml`. Phase 1 returns a stub /predict response
 * so callers can be wired and tested end-to-end before any real model exists.
 *
 * Nothing in the production trading path uses this yet — it is added so
 * Phases 2-5 have a stable seam to swap in the real LightGBM-backed
 * predictor without further changes to the Node side.
 */

import type { TimeframeKey } from "./pattern-analyzer";

const DEFAULT_TIMEOUT_MS = 5_000;

// Resolved per-call so tests (and production env hot-swaps) can override
// ML_ENGINE_URL without having to import this module after setting it.
function mlBaseUrl(): string {
  return process.env.ML_ENGINE_URL || "http://localhost:8000";
}

export interface MlFeatureVector {
  candleCount: number;
  lastPrice: number;
  ret1: number;
  ret5: number;
  ret10: number;
  momentum: number;
  realizedVol: number;
  rsi14: number;
  macdLine: number;
  macdSignal: number;
  macdHist: number;
  atr14: number;
  atrPct: number;
  ema9: number;
  ema21: number;
  emaSpreadPct: number;
  distFromEma9Pct: number;
  distFromEma21Pct: number;
  bbUpper: number;
  bbMiddle: number;
  bbLower: number;
  bbWidth: number;
  bbPctB: number;
  bbWidthPct: number;
}

export interface MlFeaturesResponse {
  coinId: string;
  timeframe: string;
  candleCount: number;
  features: MlFeatureVector | null;
  insufficientData: boolean;
  featureHash: string | null;
  durationMs: number;
}

export interface MlPredictResponse {
  coinId: string;
  timeframe: string;
  probUp: number;
  probDown: number;
  // Optional fields populated only when the LightGBM path is used; the stub
  // path returned by the ml-engine before any model is trained omits them.
  probStable?: number;
  expectedReturnPct: number;
  predictionStdPct?: number | null;
  confidence?: number;
  modelVersion?: string;
  modelCoinId?: string;
  featureHash?: string | null;
  /**
   * Provenance of the prediction.
   *  - "lightgbm" / "model": real per-(coin,tf) LightGBM booster
   *  - "prior":   pooled-prior fallback (Laplace-smoothed marginals only)
   *  - "stub":    in-memory stub used pre-Phase-3 / unit tests
   *
   * Headline accuracy & P&L scoreboards filter `"prior"` out by default —
   * see predictions.source / model_predictions.source for details.
   */
  source: "stub" | "model" | "lightgbm" | "prior";
  /**
   * Phase 3 — live 6-class regime label classified from the same feature
   * vector used by the model. `null` when /ml/predict couldn't classify
   * (e.g. the prior-only fallback path).
   */
  regime?: string | null;
  /**
   * Phase 3 — per-regime specialist heads' view of the same bar. One
   * entry per specialist kind (momentum, mean_reversion, breakout,
   * volatility_forecaster); a kind without a trained model still
   * appears with `error: "not_trained"` so consumers can render an
   * "awaiting training" cell. NOT used to gate live trades — Phase 4's
   * meta-model is the gate. Forwarded into the prediction journal so
   * the diagnostics page can compute per-specialist per-regime accuracy.
   */
  specialists?: MlSpecialistPrediction[];
  /**
   * Task #418 — attribution flag set to `"pooled"` when the response
   * was served by the pooled fallback for a coin whose per-coin slot
   * was missing, quarantined, or failed the verification gate.
   * `null`/absent when the per-coin slot served the prediction or the
   * caller targeted the pooled slot directly. Surfaced in the
   * dashboard so operators can tell which head decided.
   */
  fallback?: "pooled" | null;
}

export interface MlSpecialistPrediction {
  kind: string;
  modelVersion: string;
  modelCoinId: string;
  regimeSubset: string[];
  applicable: boolean;
  probUp?: number | null;
  probDown?: number | null;
  probStable?: number | null;
  expectedReturnPct?: number | null;
  confidence?: number | null;
  error?: string | null;
}

async function postJson<T>(path: string, body: unknown, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${mlBaseUrl()}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`ml-engine ${path} ${res.status}: ${text.slice(0, 200)}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export async function mlHealth(timeoutMs = DEFAULT_TIMEOUT_MS): Promise<{ status: string; service: string }> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${mlBaseUrl()}/ml/health`, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`ml-engine health ${res.status}`);
    return (await res.json()) as { status: string; service: string };
  } finally {
    clearTimeout(timer);
  }
}

export interface MlModelsResponse {
  available: { coinId: string; timeframe: string }[];
  count: number;
}

export async function getMlModels(timeoutMs = DEFAULT_TIMEOUT_MS): Promise<MlModelsResponse> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${mlBaseUrl()}/ml/models`, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`ml-engine /ml/models ${res.status}`);
    return (await res.json()) as MlModelsResponse;
  } finally {
    clearTimeout(timer);
  }
}

export function getMlPrediction(coinId: string, timeframe: TimeframeKey): Promise<MlPredictResponse> {
  return postJson<MlPredictResponse>("/ml/predict", { coinId, timeframe });
}

/**
 * Phase 4 — meta-model action / size / abstain decision served by the
 * Python ml-engine. Falls back to a deterministic specialist-agreement
 * heuristic on the server side when no trained meta-model exists yet,
 * so this call always returns something useful.
 */
export type MetaAction = "long" | "short" | "no_trade";
export type MetaAbstainReason =
  | "low_edge"
  | "bad_regime_fit"
  | "specialist_disagreement"
  | "low_calibration";

export interface MlMetaPredictRequest {
  coinId: string;
  timeframe: TimeframeKey;
  base: {
    probUp: number;
    probDown: number;
    probStable: number;
    expectedReturnPct: number;
    rawConfidence?: number;
    predictionStdPct?: number | null;
  };
  regime?: string | null;
  specialists?: Array<Record<string, unknown>>;
}

export interface MlMetaPredictResponse {
  timeframe: string;
  action: MetaAction;
  sizeMultiplier: number;
  expectedEdgePct: number;
  abstainReason: MetaAbstainReason | null;
  metaKind: "lightgbm" | "heuristic" | "heuristic_fallback";
  metaVersion: string;
  actionProbs: Record<MetaAction, number>;
}

export function getMlMetaPrediction(
  body: MlMetaPredictRequest,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<MlMetaPredictResponse> {
  return postJson<MlMetaPredictResponse>("/ml/meta/predict", body, timeoutMs);
}

/**
 * Phase 1 — single source of truth for journaled features.
 *
 * The Python ml-engine owns canonical feature generation (RSI, MACD, ATR,
 * EMA, Bollinger, returns, vol). The TS pattern-analyzer.ts is a *thin
 * mirror* used only on the live LLM-prompt path, kept in lockstep via the
 * parity test in artifacts/ml-engine/tests. When we journal a prediction
 * we ask the Python service for the authoritative vector + featureHash so
 * the learner trains on identical numbers regardless of which brain made
 * the call.
 */
export function getMlFeatures(
  coinId: string,
  timeframe: TimeframeKey,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<MlFeaturesResponse> {
  return postJson<MlFeaturesResponse>("/ml/features", { coinId, timeframe }, timeoutMs);
}

/**
 * Phase 2 — first-class regime label set served by the Python ml-engine.
 * Live, training, and backtest all read from the same classifier so the
 * label space cannot drift between paths.
 */
export type RegimeLabel =
  | "trending_up"
  | "trending_down"
  | "range_chop"
  | "high_vol_breakout"
  | "low_vol_compression"
  | "panic_liquidation";

export const REGIME_LABELS: RegimeLabel[] = [
  "trending_up",
  "trending_down",
  "range_chop",
  "high_vol_breakout",
  "low_vol_compression",
  "panic_liquidation",
];

export interface MlRegimeResponse {
  coinId: string;
  timeframe: string;
  label: RegimeLabel;
  confidence: number;
  inputs: Record<string, number>;
  insufficientData: boolean;
  candleCount: number;
  durationMs: number;
}

export interface MlBasketRegimeResponse {
  label: RegimeLabel;
  confidence: number;
  inputs: Record<string, number>;
}

export function getMlRegime(coinId: string, timeframe: TimeframeKey): Promise<MlRegimeResponse> {
  return postJson<MlRegimeResponse>("/ml/regime", { coinId, timeframe });
}

/**
 * Phase 5 — unified decision engine HTTP wrapper. The Python ml-engine
 * exposes the SAME `decide()` function the offline backtester uses, so a
 * live trading decision and an offline backtest decision cannot drift.
 * Returns an action ("long"/"short"/"no_trade"), confidence, suggested
 * position size, SL/TP geometry, and the portfolio-constraint breakdown.
 */
export interface MlDecideOpenPosition {
  coinId: string;
  direction: string;
  notionalUsd: number;
  regimeAtEntry?: string | null;
  betaToBtc?: number | null;
}
export interface MlDecidePortfolio {
  equityUsd: number;
  cashUsd: number;
  openPositions: MlDecideOpenPosition[];
}
export interface MlDecideRequest {
  coinId: string;
  timeframe: TimeframeKey;
  lastPrice: number;
  atrValue: number;
  probUp: number;
  probDown: number;
  probStable: number;
  expectedReturnPct: number;
  regime?: string | null;
  trendBias?: "bullish" | "bearish" | null;
  portfolio?: MlDecidePortfolio | null;
  recentOutcomes?: number[];
  gateMinConfidence?: number;
  gateMinTpDistancePct?: number;
  gateMinEvVsCost?: number;
  gateCounterTrendMinConfidence?: number;
}
export interface MlDecideResponse {
  action: "long" | "short" | "no_trade";
  confidence: number;
  sizeMultiplier: number;
  positionSizeUsd: number;
  direction: "up" | "down" | null;
  slPrice: number | null;
  tpPrice: number | null;
  expiresInMs: number;
  skipReason: string | null;
  skipDetail: string | null;
  gatesApplied: Record<string, unknown>;
  portfolioCheck: Record<string, unknown>;
  raw: Record<string, unknown>;
}

export function getMlDecision(
  body: MlDecideRequest,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<MlDecideResponse> {
  return postJson<MlDecideResponse>("/ml/decide", body, timeoutMs);
}

export function getMlBasketRegime(
  changes24h: number[],
  crossCoinVol?: number | null,
): Promise<MlBasketRegimeResponse> {
  return postJson<MlBasketRegimeResponse>("/ml/regime/basket", {
    changes24h,
    crossCoinVol: crossCoinVol ?? null,
  });
}
