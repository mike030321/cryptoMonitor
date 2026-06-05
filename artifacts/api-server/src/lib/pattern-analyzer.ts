/**
 * pattern-analyzer.ts — THIN TS MIRROR of artifacts/ml-engine/app/features.py.
 *
 * Adaptive engine (Phase 1) makes Python the single source of truth for
 * feature generation. The ml-engine `/ml/features` endpoint is what the
 * journal/learner reads. This file only exists to feed the live LLM-prompt
 * path with structurally identical numbers (RSI/MACD/ATR/EMA/Bollinger
 * + the trend / support / resistance derivations Python doesn't replicate).
 *
 * Lockstep parity with features.py is enforced by:
 *   - artifacts/ml-engine/tests/_gen_reference.mjs (regenerates reference
 *     values from THIS file)
 *   - artifacts/ml-engine/tests/test_features.py (asserts Python matches
 *     those reference values)
 *
 * If you change indicator math here, you MUST regenerate the reference
 * fixture and re-run the parity test, otherwise the journal and the live
 * prompts will silently disagree.
 */
import { db, priceHistoryTable, predictionsTable } from "@workspace/db";
import { eq, desc, gte, lte, and, sql } from "drizzle-orm";
import { logger } from "./logger";

// lookbackMultiple now drives resampled-candle counts (one candle per ms bucket).
// MACD(12,26,9) needs >=35 candles to emit a non-default signal line; RSI(14)
// and EMA periods are well-covered by >=35. Pre-resampling these values were
// fine for raw ticks (~30s cadence) but post-resampling they produced too few
// candles on 1h/2h/6h/1d, silently defaulting MACD/EMA and degrading signal
// quality. Bumped so every timeframe has >=35 candles available.
export const TIMEFRAMES = {
  "1m": { label: "1 Minute", ms: 60_000, lookbackMultiple: 60 },
  "5m": { label: "5 Minutes", ms: 300_000, lookbackMultiple: 60 },
  "1h": { label: "1 Hour", ms: 3_600_000, lookbackMultiple: 60 },
  "2h": { label: "2 Hours", ms: 7_200_000, lookbackMultiple: 60 },
  "6h": { label: "6 Hours", ms: 21_600_000, lookbackMultiple: 40 },
  "1d": { label: "1 Day", ms: 86_400_000, lookbackMultiple: 35 },
} as const;

export type TimeframeKey = keyof typeof TIMEFRAMES;

export interface MACDResult {
  macdLine: number;
  signalLine: number;
  histogram: number;
  crossover: "bullish" | "bearish" | "none";
}

export interface BollingerBands {
  upper: number;
  middle: number;
  lower: number;
  width: number;
  percentB: number;
}

export interface EMAResult {
  ema9: number;
  ema21: number;
  crossover: "bullish" | "bearish" | "none";
  spread: number;
}

export interface PatternAnalysis {
  trend: "bullish" | "bearish" | "sideways";
  trendStrength: number;
  volatility: number;
  avgPriceChange: number;
  support: number;
  resistance: number;
  rsi: number;
  momentum: number;
  volumeTrend: "increasing" | "decreasing" | "stable";
  priceHistory: number[];
  recentChanges: number[];
  patternType: string;
  macd: MACDResult;
  bollingerBands: BollingerBands;
  ema: EMAResult;
  atr: number;
  signalStrength: number;
}

export interface DirectionAccuracy {
  up: { correct: number; total: number; accuracy: number };
  down: { correct: number; total: number; accuracy: number };
}

export interface PastAccuracy {
  totalPredictions: number;
  correctPredictions: number;
  accuracy: number;
  avgConfidence: number;
  bestTimeframe: string;
  bestTimeframeAccuracy: number;
  coinAccuracy: Record<string, { correct: number; total: number; accuracy: number }>;
  directionBias: { up: number; down: number; stable: number };
  directionAccuracyByTimeframe: Record<string, DirectionAccuracy>;
  avgWin: number;
  avgLoss: number;
  winRate: number;
}

const MIN_SAMPLES: Record<TimeframeKey, number> = {
  "1m": 3,
  "5m": 5,
  "1h": 8,
  "2h": 10,
  "6h": 12,
  "1d": 15,
};

function calculateEMAValues(prices: number[], period: number): number[] {
  if (prices.length === 0) return [];
  const k = 2 / (period + 1);
  const emaValues: number[] = [prices[0]];
  for (let i = 1; i < prices.length; i++) {
    emaValues.push(prices[i] * k + emaValues[i - 1] * (1 - k));
  }
  return emaValues;
}

function calculateMACD(prices: number[]): MACDResult {
  if (prices.length < 26) {
    return { macdLine: 0, signalLine: 0, histogram: 0, crossover: "none" };
  }

  const ema12 = calculateEMAValues(prices, 12);
  const ema26 = calculateEMAValues(prices, 26);
  const macdLine = ema12.map((v, i) => v - ema26[i]);
  const signalEma = calculateEMAValues(macdLine, 9);

  const currentMacd = macdLine[macdLine.length - 1];
  const currentSignal = signalEma[signalEma.length - 1];
  const histogram = currentMacd - currentSignal;

  const prevMacd = macdLine.length >= 2 ? macdLine[macdLine.length - 2] : currentMacd;
  const prevSignal = signalEma.length >= 2 ? signalEma[signalEma.length - 2] : currentSignal;

  let crossover: "bullish" | "bearish" | "none" = "none";
  if (prevMacd <= prevSignal && currentMacd > currentSignal) crossover = "bullish";
  else if (prevMacd >= prevSignal && currentMacd < currentSignal) crossover = "bearish";

  return { macdLine: currentMacd, signalLine: currentSignal, histogram, crossover };
}

function calculateBollingerBands(prices: number[], period = 20, stdDevMultiplier = 2): BollingerBands {
  if (prices.length < period) {
    const last = prices[prices.length - 1] || 0;
    return { upper: last, middle: last, lower: last, width: 0, percentB: 0.5 };
  }

  const slice = prices.slice(-period);
  const middle = slice.reduce((s, p) => s + p, 0) / period;
  const variance = slice.reduce((s, p) => s + (p - middle) ** 2, 0) / (period - 1);
  const stdDev = Math.sqrt(variance);

  const upper = middle + stdDevMultiplier * stdDev;
  const lower = middle - stdDevMultiplier * stdDev;
  const width = upper - lower;
  const currentPrice = prices[prices.length - 1];
  const percentB = width > 0 ? (currentPrice - lower) / width : 0.5;

  return { upper, middle, lower, width, percentB };
}

function calculateEMACrossover(prices: number[]): EMAResult {
  if (prices.length < 21) {
    const last = prices[prices.length - 1] || 0;
    return { ema9: last, ema21: last, crossover: "none", spread: 0 };
  }

  const ema9Values = calculateEMAValues(prices, 9);
  const ema21Values = calculateEMAValues(prices, 21);

  const ema9 = ema9Values[ema9Values.length - 1];
  const ema21 = ema21Values[ema21Values.length - 1];
  const spread = ((ema9 - ema21) / ema21) * 100;

  const prevEma9 = ema9Values.length >= 2 ? ema9Values[ema9Values.length - 2] : ema9;
  const prevEma21 = ema21Values.length >= 2 ? ema21Values[ema21Values.length - 2] : ema21;

  let crossover: "bullish" | "bearish" | "none" = "none";
  if (prevEma9 <= prevEma21 && ema9 > ema21) crossover = "bullish";
  else if (prevEma9 >= prevEma21 && ema9 < ema21) crossover = "bearish";

  return { ema9, ema21, crossover, spread };
}

function calculateATR(prices: number[], period = 14): number {
  if (prices.length < period + 1) return 0;

  const trueRanges: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    const prevClose = prices[i - 1];
    const currClose = prices[i];
    const estimatedHigh = Math.max(currClose, prevClose) * (1 + 0.001); // quant-only-allow: 10bp candle high/low jitter for missing OHLC reconstruction, not a fee
    const estimatedLow = Math.min(currClose, prevClose) * (1 - 0.001); // quant-only-allow: 10bp candle high/low jitter for missing OHLC reconstruction, not a fee
    const tr = Math.max(
      estimatedHigh - estimatedLow,
      Math.abs(estimatedHigh - prevClose),
      Math.abs(estimatedLow - prevClose)
    );
    trueRanges.push(tr);
  }

  let atr = 0;
  for (let i = 0; i < period; i++) {
    atr += trueRanges[i];
  }
  atr /= period;

  for (let i = period; i < trueRanges.length; i++) {
    atr = (atr * (period - 1) + trueRanges[i]) / period;
  }

  return atr;
}

function calculateSignalStrength(
  rsi: number,
  macd: MACDResult,
  bb: BollingerBands,
  ema: EMAResult,
  momentum: number,
  trend: string,
  trendStrength: number
): number {
  const MAX_RAW = 120;
  let strength = 0;

  if (rsi > 70 || rsi < 30) strength += 25;
  else if (rsi > 60 || rsi < 40) strength += 10;

  if (macd.crossover !== "none") strength += 30;
  else if (Math.abs(macd.histogram) > Math.abs(macd.signalLine) * 0.1) strength += 10;

  if (bb.percentB > 0.95 || bb.percentB < 0.05) strength += 25;
  else if (bb.percentB > 0.8 || bb.percentB < 0.2) strength += 15;

  if (ema.crossover !== "none") strength += 20;
  else if (Math.abs(ema.spread) > 0.5) strength += 10;

  if (Math.abs(momentum) > 0.3) strength += 10;
  if (trendStrength > 0.6) strength += 10;

  return Math.round((strength / MAX_RAW) * 100);
}

export interface RawPricePoint {
  price: number;
  timestamp: Date | number;
}

/**
 * Resample raw tick data into per-timeframe candle closes. Each bucket is the
 * window [bucketStart, bucketStart + bucketMs). The "close" of a bucket is the
 * LAST observed tick price within that bucket. Empty buckets are skipped (we
 * never invent prices). This matches how real candlestick data is computed by
 * exchanges and is what every classical indicator (RSI, MACD, EMA, BB, ATR)
 * was designed for. Feeding raw 30s ticks straight into a 14-period RSI on a
 * "1h" timeframe was the silent bug that made every short-timeframe signal
 * meaningless.
 */
/**
 * Strip synthetic seed rows so they cannot leak into RSI/MACD/EMA/ATR. The
 * indicator pipeline calls this; if too few REAL ticks remain, the analyzer
 * downstream returns patternType="insufficient_data" and the engine skips
 * the coin/timeframe rather than trading on fabricated structure.
 */
export function filterRealTicksOnly<T extends { isSynthetic: boolean | null }>(
  rows: T[],
): T[] {
  return rows.filter((r) => !r.isSynthetic);
}

export function resampleToCandles(points: RawPricePoint[], bucketMs: number): number[] {
  if (points.length === 0 || bucketMs <= 0) return [];
  const closes: number[] = [];
  let currentBucketStart = -1;
  let lastPriceInBucket = 0;
  for (const p of points) {
    const ts = p.timestamp instanceof Date ? p.timestamp.getTime() : Number(p.timestamp);
    if (!Number.isFinite(ts) || !Number.isFinite(p.price) || p.price <= 0) continue;
    const bucketStart = Math.floor(ts / bucketMs) * bucketMs;
    if (bucketStart !== currentBucketStart) {
      if (currentBucketStart !== -1) closes.push(lastPriceInBucket);
      currentBucketStart = bucketStart;
    }
    lastPriceInBucket = p.price;
  }
  if (currentBucketStart !== -1) closes.push(lastPriceInBucket);
  return closes;
}

// Phase 1 telemetry — counts of how often pattern analysis used the
// canonical Python feature pipeline vs the local TS fallback. Read by the
// journal-health diagnostics card so any drift toward fallback is visible.
//
// Buckets:
//  - python:       served a fresh cached overlay from the Python service
//  - python_stale: served a stale cached overlay while a background
//                  refresh is in flight (still numerically authoritative,
//                  just slightly older than FRESH_MS)
//  - ts_fallback:  no cached overlay was available (cold cache or the
//                  background refresh has been failing) — local TS values
//                  were used. This is the safety net the task description
//                  refers to and the bucket to alarm on.
const featureSourceCounts = { python: 0, python_stale: 0, ts_fallback: 0 };

function recordFeatureSource(source: "python" | "python_stale" | "ts_fallback"): void {
  featureSourceCounts[source]++;
}

export function getFeatureSourceTelemetry(): {
  python: number;
  pythonStale: number;
  tsFallback: number;
  pythonPct: number;
} {
  const fresh = featureSourceCounts.python;
  const stale = featureSourceCounts.python_stale;
  const fallback = featureSourceCounts.ts_fallback;
  const total = fresh + stale + fallback;
  return {
    python: fresh,
    pythonStale: stale,
    tsFallback: fallback,
    // pythonPct counts both fresh + stale because both come from the
    // canonical Python pipeline; only ts_fallback represents drift.
    pythonPct: total > 0 ? ((fresh + stale) / total) * 100 : 0,
  };
}

export async function analyzePatterns(coinId: string, timeframe: TimeframeKey): Promise<PatternAnalysis> {
  const tf = TIMEFRAMES[timeframe];
  const lookbackMs = tf.ms * tf.lookbackMultiple;
  const now = new Date();
  const since = new Date(now.getTime() - lookbackMs);

  const history = await db
    .select({
      price: priceHistoryTable.price,
      timestamp: priceHistoryTable.timestamp,
      isSynthetic: priceHistoryTable.isSynthetic,
    })
    .from(priceHistoryTable)
    .where(and(eq(priceHistoryTable.coinId, coinId), gte(priceHistoryTable.timestamp, since), lte(priceHistoryTable.timestamp, now)))
    .orderBy(priceHistoryTable.timestamp);

  // Synthetic seed rows (interpolated hourly fills used to bootstrap the
  // system at startup) are NOT a substitute for real candles — their structure
  // leaks into RSI/MACD/EMA/ATR and biases every signal. We refuse to fall
  // back to them: when real data is insufficient the function below returns
  // patternType="insufficient_data" and the engine skips trading on that
  // coin/timeframe until enough real ticks have accumulated.
  const ticks = filterRealTicksOnly(history);

  // Resample into proper per-timeframe candles. We do NOT fall back to raw
  // sub-timeframe ticks: feeding a 14-period RSI on a "1h" timeframe with
  // 30-second tick noise was the silent bug this task is meant to eliminate.
  // If the resampled candle count is below the indicator minimum, the block
  // below returns patternType="insufficient_data" and the engine refuses to
  // trade that coin/timeframe until enough real candles have accumulated.
  const resampled = resampleToCandles(
    ticks.map((t) => ({ price: t.price, timestamp: t.timestamp })),
    tf.ms,
  );
  const prices = resampled;
  const minRequired = MIN_SAMPLES[timeframe] || 3;
  const defaultMACD: MACDResult = { macdLine: 0, signalLine: 0, histogram: 0, crossover: "none" };
  const lastPrice = prices[prices.length - 1] || 0;
  const defaultBB: BollingerBands = { upper: lastPrice, middle: lastPrice, lower: lastPrice, width: 0, percentB: 0.5 };
  const defaultEMA: EMAResult = { ema9: lastPrice, ema21: lastPrice, crossover: "none", spread: 0 };

  if (prices.length < minRequired) {
    return {
      trend: "sideways",
      trendStrength: 0,
      volatility: 0,
      avgPriceChange: 0,
      support: prices[0] || 0,
      resistance: prices[0] || 0,
      rsi: 50,
      momentum: 0,
      volumeTrend: "stable",
      priceHistory: prices,
      recentChanges: [],
      patternType: "insufficient_data",
      macd: defaultMACD,
      bollingerBands: defaultBB,
      ema: defaultEMA,
      atr: 0,
      signalStrength: 0,
    };
  }

  const changes = prices.slice(1).map((p, i) => ((p - prices[i]) / prices[i]) * 100);

  const avgChange = changes.reduce((s, c) => s + c, 0) / changes.length;
  const volatility = Math.sqrt(changes.reduce((s, c) => s + (c - avgChange) ** 2, 0) / changes.length);

  const recentHalf = changes.slice(Math.floor(changes.length / 2));
  const olderHalf = changes.slice(0, Math.floor(changes.length / 2));
  const recentAvg = recentHalf.length > 0 ? recentHalf.reduce((s, c) => s + c, 0) / recentHalf.length : 0;
  const olderAvg = olderHalf.length > 0 ? olderHalf.reduce((s, c) => s + c, 0) / olderHalf.length : 0;

  let trend: "bullish" | "bearish" | "sideways";
  let trendStrength: number;
  if (recentAvg > 0.05 && avgChange > 0) {
    trend = "bullish";
    trendStrength = Math.min(1, Math.abs(recentAvg) / volatility || 0.5);
  } else if (recentAvg < -0.05 && avgChange < 0) {
    trend = "bearish";
    trendStrength = Math.min(1, Math.abs(recentAvg) / volatility || 0.5);
  } else {
    trend = "sideways";
    trendStrength = Math.max(0, 1 - volatility * 2);
  }

  const support = Math.min(...prices.slice(-Math.min(20, prices.length)));
  const resistance = Math.max(...prices.slice(-Math.min(20, prices.length)));

  const rsi = calculateRSI(prices);
  const momentum = recentAvg - olderAvg;

  const recentVol = recentHalf.reduce((s, c) => s + Math.abs(c), 0) / (recentHalf.length || 1);
  const olderVol = olderHalf.reduce((s, c) => s + Math.abs(c), 0) / (olderHalf.length || 1);
  const activityTrend = recentVol > olderVol * 1.2 ? "increasing" : recentVol < olderVol * 0.8 ? "decreasing" : "stable";
  const volumeTrend = activityTrend;

  const macd = calculateMACD(prices);
  const bollingerBands = calculateBollingerBands(prices);
  const ema = calculateEMACrossover(prices);
  const rawAtr = calculateATR(prices);
  const recentReturns = changes.slice(-Math.min(20, changes.length));
  const realizedVol = recentReturns.length > 1
    ? Math.sqrt(recentReturns.reduce((s, c) => s + (c / 100) ** 2, 0) / recentReturns.length)
    : 0.005;
  const dynamicFloor = lastPrice * Math.max(realizedVol * 0.5, 0.003);
  const atr = Math.max(rawAtr, dynamicFloor);

  let patternType = "ranging";
  if (macd.crossover === "bullish" && rsi < 60) patternType = "macd_bullish_cross";
  else if (macd.crossover === "bearish" && rsi > 40) patternType = "macd_bearish_cross";
  else if (bollingerBands.percentB > 0.95) patternType = "bb_upper_touch";
  else if (bollingerBands.percentB < 0.05) patternType = "bb_lower_touch";
  else if (ema.crossover === "bullish") patternType = "ema_golden_cross";
  else if (ema.crossover === "bearish") patternType = "ema_death_cross";
  else if (trendStrength > 0.6 && trend === "bullish") patternType = "strong_uptrend";
  else if (trendStrength > 0.6 && trend === "bearish") patternType = "strong_downtrend";
  else if (rsi > 70) patternType = "overbought";
  else if (rsi < 30) patternType = "oversold";
  else if (volatility > 2) patternType = "high_volatility_breakout";
  else if (volatility < 0.1) patternType = "consolidation";
  else if (momentum > 0.3) patternType = "momentum_surge";
  else if (momentum < -0.3) patternType = "momentum_drop";

  // Python features.py is the canonical source of indicator math (Phase 1
  // adaptive engine). When ml-engine is reachable we OVERLAY its values
  // onto the ones we just computed locally — the parity test guarantees
  // they're numerically identical, so no behavior change, but every
  // downstream consumer (LLM prompt, journal, fingerprint, signal
  // strength) is now reading the same numbers the Python service used.
  // If ml-engine is unreachable we keep the local TS values so trading
  // is never blocked by ML-engine downtime; the fallback is counted in
  // featureSourceTelemetry so diagnostics can surface the gap.
  const overlayResult = getCanonicalIndicatorsNonBlocking(coinId, timeframe);
  const pythonOverlay = overlayResult.value;
  recordFeatureSource(overlayResult.source);
  const finalRsi = pythonOverlay?.rsi14 ?? rsi;
  const finalMacd: MACDResult = pythonOverlay
    ? {
        macdLine: pythonOverlay.macdLine,
        signalLine: pythonOverlay.macdSignal,
        histogram: pythonOverlay.macdHist,
        crossover: macd.crossover,
      }
    : macd;
  const finalAtr = pythonOverlay?.atr14 ?? atr;
  const finalEma: EMAResult = pythonOverlay
    ? {
        ema9: pythonOverlay.ema9,
        ema21: pythonOverlay.ema21,
        crossover: ema.crossover,
        spread: pythonOverlay.emaSpreadPct,
      }
    : ema;
  const finalBb: BollingerBands = pythonOverlay
    ? {
        upper: pythonOverlay.bbUpper,
        middle: pythonOverlay.bbMiddle,
        lower: pythonOverlay.bbLower,
        width: pythonOverlay.bbWidth,
        percentB: pythonOverlay.bbPctB,
      }
    : bollingerBands;

  // Compute signal strength AFTER overlay so downstream LLM prompts and
  // gate logic use the SAME numbers the Python feature pipeline produced.
  const signalStrength = calculateSignalStrength(
    finalRsi, finalMacd, finalBb, finalEma, momentum, trend, trendStrength,
  );

  return {
    trend,
    trendStrength,
    volatility,
    avgPriceChange: avgChange,
    support,
    resistance,
    rsi: finalRsi,
    momentum,
    volumeTrend,
    priceHistory: prices.slice(-20),
    recentChanges: changes.slice(-10),
    patternType,
    macd: finalMacd,
    bollingerBands: finalBb,
    ema: finalEma,
    atr: finalAtr,
    signalStrength,
  };
}

/**
 * Canonical indicator vector overlaid on the locally-computed TS values.
 * Returned shape mirrors the subset of MlFeatureVector that
 * pattern-analyzer actually consumes.
 */
type CanonicalIndicators = {
  rsi14: number;
  macdLine: number;
  macdSignal: number;
  macdHist: number;
  atr14: number;
  ema9: number;
  ema21: number;
  emaSpreadPct: number;
  bbUpper: number;
  bbMiddle: number;
  bbLower: number;
  bbWidth: number;
  bbPctB: number;
};

// ---------------------------------------------------------------------------
// Non-blocking ml-engine feature overlay
// ---------------------------------------------------------------------------
//
// Trading must never wait on the Python ml-engine. The previous
// implementation awaited a 5s-timeout HTTP call inside the analysis path,
// so any ml-engine slowdown directly throttled prediction cadence.
//
// Strategy:
//  1. Keep an in-memory cache of the most recently fetched canonical
//     indicators per (coinId, timeframe).
//  2. Pattern analysis NEVER awaits the network — it reads from the cache
//     synchronously and kicks off a background refresh when the entry is
//     stale or missing.
//  3. The background refresher uses the normal long timeout (so it is
//     resilient to a momentarily slow service) but runs off the hot path,
//     deduplicated so we don't pile up requests for the same key.
//
// Net effect: an ml-engine slowdown only delays cache freshness; it never
// stalls a single analysis cycle. Cold caches (or sustained ml-engine
// outages) fall through to the TS values exactly as before, counted as
// `ts_fallback` in featureSourceTelemetry so diagnostics still surface
// the gap.
const FEATURE_FRESH_MS = 10_000;
const FEATURE_STALE_OK_MS = 5 * 60_000;
const FEATURE_BG_TIMEOUT_MS = 5_000;

type FeatureCacheEntry = { value: CanonicalIndicators; fetchedAt: number };

const featureCache = new Map<string, FeatureCacheEntry>();
const featureRefreshInflight = new Map<string, Promise<void>>();
let lastFallbackLogAt = 0;

function featureKey(coinId: string, timeframe: TimeframeKey): string {
  return `${coinId}::${timeframe}`;
}

function refreshCanonicalIndicators(coinId: string, timeframe: TimeframeKey): Promise<void> {
  const key = featureKey(coinId, timeframe);
  const existing = featureRefreshInflight.get(key);
  if (existing) return existing;

  const p = (async () => {
    try {
      // Lazy-imported to avoid a circular dep with ml-client (which
      // imports TimeframeKey from this file).
      const { getMlFeatures } = await import("./ml-client");
      const resp = await getMlFeatures(coinId, timeframe, FEATURE_BG_TIMEOUT_MS);
      const f = resp.features;
      if (!f || resp.insufficientData) return;
      featureCache.set(key, {
        value: {
          rsi14: f.rsi14,
          macdLine: f.macdLine,
          macdSignal: f.macdSignal,
          macdHist: f.macdHist,
          atr14: f.atr14,
          ema9: f.ema9,
          ema21: f.ema21,
          emaSpreadPct: f.emaSpreadPct,
          bbUpper: f.bbUpper,
          bbMiddle: f.bbMiddle,
          bbLower: f.bbLower,
          bbWidth: f.bbWidth,
          bbPctB: f.bbPctB,
        },
        fetchedAt: Date.now(),
      });
    } catch {
      // Swallow — telemetry already records the ts_fallback bucket and
      // the next analysis cycle will retry the background refresh.
    } finally {
      featureRefreshInflight.delete(key);
    }
  })();

  featureRefreshInflight.set(key, p);
  return p;
}

/**
 * Synchronous, non-blocking overlay lookup. Returns the cached canonical
 * indicators (and whether they were fresh or stale) and schedules a
 * background refresh when needed. Never awaits the ml-engine.
 */
function getCanonicalIndicatorsNonBlocking(
  coinId: string,
  timeframe: TimeframeKey,
): { value: CanonicalIndicators | null; source: "python" | "python_stale" | "ts_fallback" } {
  const key = featureKey(coinId, timeframe);
  const cached = featureCache.get(key);
  const now = Date.now();

  if (cached && now - cached.fetchedAt <= FEATURE_FRESH_MS) {
    return { value: cached.value, source: "python" };
  }

  // Either stale or missing — kick off a background refresh.
  void refreshCanonicalIndicators(coinId, timeframe);

  if (cached && now - cached.fetchedAt <= FEATURE_STALE_OK_MS) {
    return { value: cached.value, source: "python_stale" };
  }

  // Cold cache (or sustained outage). Fall back to TS values. Log at
  // most once a minute so an extended outage is obvious without
  // spamming the logs.
  if (now - lastFallbackLogAt > 60_000) {
    lastFallbackLogAt = now;
    console.warn(
      `[pattern-analyzer] ml-engine feature overlay unavailable for ${coinId}/${timeframe} — using TS fallback (cache cold or refresh failing)`,
    );
  }
  return { value: null, source: "ts_fallback" };
}

/**
 * Test-only: clear the in-memory feature cache & inflight tracking so
 * unit tests can exercise cold-cache behavior deterministically.
 */
export function __resetFeatureCacheForTests(): void {
  featureCache.clear();
  featureRefreshInflight.clear();
  lastFallbackLogAt = 0;
  featureSourceCounts.python = 0;
  featureSourceCounts.python_stale = 0;
  featureSourceCounts.ts_fallback = 0;
}

function calculateRSI(prices: number[], period = 14): number {
  if (prices.length < period + 1) return 50;

  const changes: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i];
    else avgLoss += Math.abs(changes[i]);
  }
  avgGain /= period;
  avgLoss /= period;

  for (let i = period; i < changes.length; i++) {
    const gain = changes[i] > 0 ? changes[i] : 0;
    const loss = changes[i] < 0 ? Math.abs(changes[i]) : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
  }

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

export async function getAgentPastAccuracy(agentId: number): Promise<PastAccuracy> {
  const resolved = await db
    .select()
    .from(predictionsTable)
    .where(and(
      eq(predictionsTable.agentId, agentId),
      sql`${predictionsTable.outcome} IN ('correct', 'wrong')`
    ))
    .orderBy(desc(predictionsTable.createdAt))
    .limit(200);

  const totalPredictions = resolved.length;
  const correctPredictions = resolved.filter((p) => p.outcome === "correct").length;
  const accuracy = totalPredictions > 0 ? correctPredictions / totalPredictions : 0;
  const avgConfidence = totalPredictions > 0
    ? resolved.reduce((s, p) => s + p.confidence, 0) / totalPredictions
    : 0.5;

  const byTimeframe: Record<string, { correct: number; total: number }> = {};
  const coinAccuracy: Record<string, { correct: number; total: number; accuracy: number }> = {};
  const directionBias = { up: 0, down: 0, stable: 0 };
  const directionAccuracyByTimeframe: Record<string, DirectionAccuracy> = {};

  for (const p of resolved) {
    const tf = p.timeframe || "1m";
    if (!byTimeframe[tf]) byTimeframe[tf] = { correct: 0, total: 0 };
    byTimeframe[tf].total++;
    if (p.outcome === "correct") byTimeframe[tf].correct++;

    if (!coinAccuracy[p.coinId]) coinAccuracy[p.coinId] = { correct: 0, total: 0, accuracy: 0 };
    coinAccuracy[p.coinId].total++;
    if (p.outcome === "correct") coinAccuracy[p.coinId].correct++;

    directionBias[p.direction as keyof typeof directionBias]++;

    if (!directionAccuracyByTimeframe[tf]) {
      directionAccuracyByTimeframe[tf] = {
        up: { correct: 0, total: 0, accuracy: 0 },
        down: { correct: 0, total: 0, accuracy: 0 },
      };
    }
    if (p.direction === "up" || p.direction === "down") {
      directionAccuracyByTimeframe[tf][p.direction].total++;
      if (p.outcome === "correct") directionAccuracyByTimeframe[tf][p.direction].correct++;
    }
  }

  for (const coin of Object.values(coinAccuracy)) {
    coin.accuracy = coin.total > 0 ? coin.correct / coin.total : 0;
  }

  for (const da of Object.values(directionAccuracyByTimeframe)) {
    da.up.accuracy = da.up.total > 0 ? da.up.correct / da.up.total : 0;
    da.down.accuracy = da.down.total > 0 ? da.down.correct / da.down.total : 0;
  }

  let bestTimeframe = "1m";
  let bestTimeframeAccuracy = 0;
  for (const [tf, stats] of Object.entries(byTimeframe)) {
    const acc = stats.total > 0 ? stats.correct / stats.total : 0;
    if (acc > bestTimeframeAccuracy && stats.total >= 3) {
      bestTimeframeAccuracy = acc;
      bestTimeframe = tf;
    }
  }

  let totalWinPnl = 0;
  let totalLossPnl = 0;
  let winCount = 0;
  let lossCount = 0;
  for (const p of resolved) {
    if (p.actualPrice != null && p.priceAtPrediction > 0) {
      const pctMove = Math.abs((p.actualPrice - p.priceAtPrediction) / p.priceAtPrediction) * 100;
      if (p.outcome === "correct") {
        totalWinPnl += pctMove;
        winCount++;
      } else {
        totalLossPnl += pctMove;
        lossCount++;
      }
    }
  }

  const avgWin = winCount > 0 ? totalWinPnl / winCount : 0.1;
  const avgLoss = lossCount > 0 ? totalLossPnl / lossCount : 0.1;
  const winRate = totalPredictions > 0 ? correctPredictions / totalPredictions : 0;

  return {
    totalPredictions,
    correctPredictions,
    accuracy,
    avgConfidence,
    bestTimeframe,
    bestTimeframeAccuracy,
    coinAccuracy,
    directionBias,
    directionAccuracyByTimeframe,
    avgWin,
    avgLoss,
    winRate,
  };
}

export async function getPastPredictionPatterns(coinId: string, timeframe: TimeframeKey): Promise<{
  historicalAccuracy: number;
  totalPast: number;
  commonOutcome: string;
  avgPriceMove: number;
}> {
  const pastPreds = await db
    .select()
    .from(predictionsTable)
    .where(and(
      eq(predictionsTable.coinId, coinId),
      eq(predictionsTable.timeframe, timeframe),
      sql`${predictionsTable.outcome} IN ('correct', 'wrong')`
    ))
    .orderBy(desc(predictionsTable.createdAt))
    .limit(50);

  if (pastPreds.length === 0) {
    return { historicalAccuracy: 0.5, totalPast: 0, commonOutcome: "unknown", avgPriceMove: 0 };
  }

  const correct = pastPreds.filter((p) => p.outcome === "correct").length;
  const historicalAccuracy = correct / pastPreds.length;

  const upCount = pastPreds.filter((p) => p.direction === "up" && p.outcome === "correct").length;
  const downCount = pastPreds.filter((p) => p.direction === "down" && p.outcome === "correct").length;
  const commonOutcome = upCount > downCount ? "up" : downCount > upCount ? "down" : "stable";

  const withActual = pastPreds.filter((p) => p.actualPrice != null);
  const avgPriceMove = withActual.length > 0
    ? withActual.reduce((s, p) => s + ((p.actualPrice! - p.priceAtPrediction) / p.priceAtPrediction) * 100, 0) / withActual.length
    : 0;

  return { historicalAccuracy, totalPast: pastPreds.length, commonOutcome, avgPriceMove };
}

export interface CompositeFingerprint {
  key: string;
  rsiBin: string;
  macdBin: string;
  bbBin: string;
  trendBin: string;
  volBin: string;
  emaBin: string;
  activityBin: string;
}

function binRSI(rsi: number): string {
  if (rsi < 20) return "extreme-oversold";
  if (rsi < 30) return "oversold";
  if (rsi < 40) return "weak";
  if (rsi < 60) return "neutral";
  if (rsi < 70) return "strong";
  if (rsi < 80) return "overbought";
  return "extreme-overbought";
}

function binMACD(macd: MACDResult): string {
  if (macd.crossover === "bullish") return "bull-cross";
  if (macd.crossover === "bearish") return "bear-cross";
  if (macd.histogram > 0 && macd.macdLine > 0) return "bull-momentum";
  if (macd.histogram < 0 && macd.macdLine < 0) return "bear-momentum";
  if (macd.histogram > 0) return "bull-diverging";
  if (macd.histogram < 0) return "bear-diverging";
  return "neutral";
}

function binBB(percentB: number): string {
  if (percentB < 0.05) return "lower-touch";
  if (percentB < 0.2) return "lower-zone";
  if (percentB < 0.4) return "mid-lower";
  if (percentB < 0.6) return "middle";
  if (percentB < 0.8) return "mid-upper";
  if (percentB < 0.95) return "upper-zone";
  return "upper-touch";
}

function binTrend(trend: string, strength: number): string {
  if (trend === "bullish" && strength > 0.6) return "strong-bull";
  if (trend === "bullish") return "weak-bull";
  if (trend === "bearish" && strength > 0.6) return "strong-bear";
  if (trend === "bearish") return "weak-bear";
  return "sideways";
}

function binVolatility(vol: number): string {
  if (vol < 0.1) return "very-low";
  if (vol < 0.5) return "low";
  if (vol < 1.5) return "normal";
  if (vol < 3.0) return "high";
  return "extreme";
}

function binEMA(ema: EMAResult): string {
  if (ema.crossover === "bullish") return "golden-cross";
  if (ema.crossover === "bearish") return "death-cross";
  if (ema.spread > 0.5) return "bull-spread";
  if (ema.spread < -0.5) return "bear-spread";
  return "neutral";
}

function binActivity(volumeTrend: string): string {
  return volumeTrend;
}

export function generateFingerprint(analysis: PatternAnalysis): CompositeFingerprint {
  const rsiBin = binRSI(analysis.rsi);
  const macdBin = binMACD(analysis.macd);
  const bbBin = binBB(analysis.bollingerBands.percentB);
  const trendBin = binTrend(analysis.trend, analysis.trendStrength);
  const volBin = binVolatility(analysis.volatility);
  const emaBin = binEMA(analysis.ema);
  const activityBin = binActivity(analysis.volumeTrend);

  const key = `${rsiBin}|${macdBin}|${bbBin}|${trendBin}|${volBin}|${emaBin}|${activityBin}`;

  return { key, rsiBin, macdBin, bbBin, trendBin, volBin, emaBin, activityBin };
}

export function fingerprintDistance(a: CompositeFingerprint, bHash: string): number {
  const bParts = bHash.split("|");
  const aParts = [a.rsiBin, a.macdBin, a.bbBin, a.trendBin, a.volBin, a.emaBin, a.activityBin];
  let diff = 0;
  for (let i = 0; i < Math.min(aParts.length, bParts.length); i++) {
    if (aParts[i] !== bParts[i]) diff++;
  }
  return diff;
}

export function formatPatternContext(analysis: PatternAnalysis, pastPatterns: { historicalAccuracy: number; totalPast: number; commonOutcome: string; avgPriceMove: number }): string {
  const lines = [
    `TREND: ${analysis.trend} (strength: ${(analysis.trendStrength * 100).toFixed(0)}%)`,
    `RSI(14): ${analysis.rsi.toFixed(1)} | Momentum: ${analysis.momentum > 0 ? "+" : ""}${analysis.momentum.toFixed(4)}`,
    `MACD(12,26,9): Line=${analysis.macd.macdLine.toFixed(8)} Signal=${analysis.macd.signalLine.toFixed(8)} Hist=${analysis.macd.histogram.toFixed(8)} Cross=${analysis.macd.crossover}`,
    `Bollinger(20,2): Upper=$${analysis.bollingerBands.upper.toFixed(8)} Mid=$${analysis.bollingerBands.middle.toFixed(8)} Lower=$${analysis.bollingerBands.lower.toFixed(8)} %B=${(analysis.bollingerBands.percentB * 100).toFixed(1)}%`,
    `EMA: 9=$${analysis.ema.ema9.toFixed(8)} 21=$${analysis.ema.ema21.toFixed(8)} Spread=${analysis.ema.spread.toFixed(3)}% Cross=${analysis.ema.crossover}`,
    `ATR(14): $${analysis.atr.toFixed(8)} | Volatility: ${analysis.volatility.toFixed(3)}%`,
    `Price Activity Trend: ${analysis.volumeTrend} (based on price movement magnitude, not exchange volume)`,
    `Support: $${analysis.support.toFixed(8)} | Resistance: $${analysis.resistance.toFixed(8)}`,
    `Pattern: ${analysis.patternType} | Signal Strength: ${analysis.signalStrength}/100`,
    `Recent changes: ${analysis.recentChanges.slice(-5).map((c) => `${c > 0 ? "+" : ""}${c.toFixed(3)}%`).join(", ")}`,
  ];

  if (pastPatterns.totalPast > 0) {
    lines.push(`HISTORICAL: ${pastPatterns.totalPast} past predictions, ${(pastPatterns.historicalAccuracy * 100).toFixed(0)}% accuracy`);
    lines.push(`Common correct direction: ${pastPatterns.commonOutcome} | Avg price move: ${pastPatterns.avgPriceMove > 0 ? "+" : ""}${pastPatterns.avgPriceMove.toFixed(4)}%`);
  }

  return lines.join("\n");
}
