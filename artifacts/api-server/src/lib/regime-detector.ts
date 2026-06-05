import { db, coinInsightsTable, fingerprintBuffersTable } from "@workspace/db";
import { eq, and, desc, sql, gte } from "drizzle-orm";
import { logger } from "./logger";
import type { CoinPrice, FearGreedData, BtcDominanceData } from "./coins";
import type { CompositeFingerprint } from "./pattern-analyzer";
import type { TimeframeKey } from "./pattern-analyzer";
import { getMlBasketRegime, getMlRegime, type RegimeLabel } from "./ml-client";

export type MarketRegime = "bull" | "bear" | "sideways" | "volatile";

export type TrendBias = "bullish" | "bearish" | "neutral";

/**
 * Phase 2 — first-class 6-class regime label set served by ml-engine.
 * Re-exported here so existing modules import it from one place.
 */
export type { RegimeLabel } from "./ml-client";
export { REGIME_LABELS } from "./ml-client";

/**
 * Map a 4-class legacy regime to the 6-class label so historical /
 * fallback paths that only have the legacy state can still stamp a
 * label. Loses some information (volatile collapses to high_vol_breakout)
 * but is good enough for backfill and for the basket fallback when the
 * Python regime endpoint is unreachable.
 */
export function legacyToRegimeLabel(legacy: MarketRegime): RegimeLabel {
  switch (legacy) {
    case "bull": return "trending_up";
    case "bear": return "trending_down";
    case "sideways": return "range_chop";
    case "volatile": return "high_vol_breakout";
  }
}

/**
 * Phase 2 — reverse map from the canonical 6-class label (single source
 * of truth in the Python ml-engine) back down to the legacy 4-class
 * `MarketRegime`. This is what lets every downstream gate (paper-trader
 * counter-trend, agent-evolution, regime adjustments) keep working
 * unchanged while the new label rides alongside.
 */
export function regimeLabelToLegacy(label: RegimeLabel): MarketRegime {
  switch (label) {
    case "trending_up": return "bull";
    case "trending_down": return "bear";
    case "range_chop": return "sideways";
    case "low_vol_compression": return "sideways";
    case "high_vol_breakout": return "volatile";
    case "panic_liquidation": return "volatile";
  }
}

export interface RegimeState {
  regime: MarketRegime;
  /**
   * Phase 2 — 6-class regime label served by ml-engine. Populated from
   * the basket-regime endpoint each cycle. Falls back to a deterministic
   * mapping from `regime` if the Python service is unreachable so the
   * field is never null on the cached state.
   */
  regimeLabel: RegimeLabel;
  confidence: number;
  avgChange24h: number;
  crossCoinVolatility: number;
  fearGreedValue: number | null;
  bullishRatio: number;
  btcDominance: number | null;
  btcDominanceDirection: "rising" | "falling" | "stable" | null;
  trendBias: TrendBias;
  timestamp: number;
}

export interface SequenceMatch {
  sequence: string[];
  outcomeAfter: string;
  avgPriceMove: number;
  winRate: number;
  sampleCount: number;
  regime: string | null;
  sameRegimeCount: number;
}

export interface TemporalContext {
  regime: RegimeState;
  recentFingerprints: string[];
  sequenceMatches: SequenceMatch[];
  regimeAdjustment: { slMultiplier: number; tpMultiplier: number; confidenceModifier: number };
}

let cachedRegime: RegimeState | null = null;
let regimeCacheTime = 0;
const REGIME_CACHE_TTL = 5 * 60 * 1000;

const fingerprintBuffers = new Map<string, string[]>();
const MAX_FINGERPRINT_HISTORY = 10;
let fingerprintsLoaded = false;

export async function loadFingerprintBuffers(): Promise<void> {
  try {
    const rows = await db.select().from(fingerprintBuffersTable);
    for (const row of rows) {
      const key = `${row.coinId}:${row.timeframe}`;
      const fps = Array.isArray(row.fingerprints) ? row.fingerprints as string[] : [];
      fingerprintBuffers.set(key, fps);
    }
    fingerprintsLoaded = true;
    if (rows.length > 0) {
      logger.info({ entries: rows.length }, "Loaded fingerprint buffers from database");
    }
  } catch (err) {
    logger.warn("Failed to load fingerprint buffers from database");
  }
}

async function persistFingerprintBuffer(coinId: string, timeframe: string, fingerprints: string[]): Promise<void> {
  try {
    await db
      .insert(fingerprintBuffersTable)
      .values({
        coinId,
        timeframe,
        fingerprints: JSON.stringify(fingerprints),
      })
      .onConflictDoUpdate({
        target: [fingerprintBuffersTable.coinId, fingerprintBuffersTable.timeframe],
        set: {
          fingerprints: JSON.stringify(fingerprints),
          updatedAt: new Date(),
        },
      });
  } catch (err) {
    logger.debug({ err, coinId, timeframe }, "Fingerprint buffer persist failed");
  }
}

export function updateFingerprintBuffer(coinId: string, timeframe: string, fingerprint: string): void {
  const key = `${coinId}:${timeframe}`;
  let buffer = fingerprintBuffers.get(key);
  if (!buffer) {
    buffer = [];
    fingerprintBuffers.set(key, buffer);
  }
  if (buffer.length > 0 && buffer[buffer.length - 1] === fingerprint) return;
  buffer.push(fingerprint);
  if (buffer.length > MAX_FINGERPRINT_HISTORY) {
    buffer.shift();
  }
  persistFingerprintBuffer(coinId, timeframe, buffer);
}

export function getRecentFingerprints(coinId: string, timeframe: string): string[] {
  return fingerprintBuffers.get(`${coinId}:${timeframe}`) || [];
}

/**
 * Phase 2 — minimal price-only fallback when the Python ml-engine is
 * unreachable. Kept intentionally small so it cannot drift far from the
 * canonical classifier; logs a warn each time it is used so an outage
 * is visible. NOT the live decision path.
 */
function fallbackRegimeLabelFromBasket(
  changes: number[],
  crossCoinVol: number,
  avgChange: number,
  bullishRatio: number,
): RegimeLabel {
  const avgVol = changes.length > 0
    ? changes.reduce((s, c) => s + Math.abs(c), 0) / changes.length
    : 0;
  // Panic: deeply negative average AND broad participation in the drop.
  if (avgChange < -4 && bullishRatio < 0.2) return "panic_liquidation";
  // High-vol breakout: cross-coin volatility very wide.
  if (crossCoinVol > 6 || avgVol > 5) return "high_vol_breakout";
  // Low-vol compression: everything basically flat.
  if (avgVol < 0.5 && crossCoinVol < 1) return "low_vol_compression";
  if (avgChange > 2 && bullishRatio > 0.6) return "trending_up";
  if (avgChange < -2 && bullishRatio < 0.4) return "trending_down";
  return "range_chop";
}

/**
 * Phase 2 — thin client for the canonical Python ml-engine basket
 * regime classifier. Returns `RegimeState` whose `regimeLabel` is
 * sourced from `/ml/regime/basket` (single source of truth across
 * live, training, and backtest), and whose legacy 4-class `regime`
 * field is derived from that label so every downstream gate keeps
 * working unchanged.
 *
 * Basket statistics (avgChange, crossCoinVol, fear&greed, BTC
 * dominance, trendBias) are still computed in TS — they're context for
 * downstream consumers, not regime classification.
 *
 * If the ml-engine is unreachable we use a tiny price-only fallback
 * and log a warning so the outage is visible; we do NOT silently
 * substitute the legacy 4-class output.
 */
export async function classifyRegime(
  prices: CoinPrice[],
  fearGreed: FearGreedData | null,
  btcDominance: BtcDominanceData | null = null,
): Promise<RegimeState> {
  if (Date.now() - regimeCacheTime < REGIME_CACHE_TTL && cachedRegime) {
    return cachedRegime;
  }

  const changes = prices.map(p => p.priceChange24h ?? 0);
  const avgChange = changes.length > 0 ? changes.reduce((s, c) => s + c, 0) / changes.length : 0;

  const variance = changes.length > 1
    ? changes.reduce((s, c) => s + (c - avgChange) ** 2, 0) / (changes.length - 1)
    : 0;
  const crossCoinVol = Math.sqrt(variance);

  const bullish = changes.filter(c => c > 1).length;
  const bullishRatio = changes.length > 0 ? bullish / changes.length : 0.5;

  const fgValue = fearGreed?.value ?? null;

  let btcDomDirection: "rising" | "falling" | "stable" | null = null;
  let btcDomValue: number | null = null;
  if (btcDominance) {
    btcDomValue = btcDominance.dominance;
    if (btcDominance.dominanceChange > 0.1 || (btcDominance.btcChange24h > 1 && btcDominance.dominanceChange >= 0)) {
      btcDomDirection = "rising";
    } else if (btcDominance.dominanceChange < -0.1 || (btcDominance.btcChange24h < -1 && btcDominance.dominanceChange <= 0)) {
      btcDomDirection = "falling";
    } else {
      btcDomDirection = "stable";
    }
  }

  // Canonical 6-class label from the Python ml-engine. This is the
  // single source of truth that live / training / backtest all share.
  let regimeLabel: RegimeLabel;
  let confidence: number;
  let source: "ml-engine" | "fallback";
  try {
    const resp = await getMlBasketRegime(changes, crossCoinVol);
    regimeLabel = resp.label;
    confidence = resp.confidence;
    source = "ml-engine";
  } catch (err) {
    regimeLabel = fallbackRegimeLabelFromBasket(changes, crossCoinVol, avgChange, bullishRatio);
    confidence = 0.4;
    source = "fallback";
    logger.warn(
      { err: String(err), regimeLabel },
      "regime-detector: ml-engine unreachable, used local fallback (Phase 2 outage)",
    );
  }

  const regime = regimeLabelToLegacy(regimeLabel);

  let trendBias: TrendBias = "neutral";
  if (avgChange > 2 && bullishRatio > 0.55) {
    trendBias = "bullish";
  } else if (avgChange < -2 && bullishRatio < 0.45) {
    trendBias = "bearish";
  } else if (avgChange > 1 && bullishRatio > 0.5) {
    trendBias = "bullish";
  } else if (avgChange < -1 && bullishRatio < 0.5) {
    trendBias = "bearish";
  }

  const state: RegimeState = {
    regime,
    regimeLabel,
    confidence,
    avgChange24h: avgChange,
    crossCoinVolatility: crossCoinVol,
    fearGreedValue: fgValue,
    bullishRatio,
    btcDominance: btcDomValue,
    btcDominanceDirection: btcDomDirection,
    trendBias,
    timestamp: Date.now(),
  };

  logger.info(
    {
      regimeLabel,
      regime,
      source,
      trendBias,
      confidence: confidence.toFixed(2),
      avgChange24h: avgChange.toFixed(2),
      bullishRatio: bullishRatio.toFixed(2),
      crossCoinVol: crossCoinVol.toFixed(2),
    },
    "Regime classified",
  );

  cachedRegime = state;
  regimeCacheTime = Date.now();

  return state;
}

/**
 * Phase 2 — per-coin per-timeframe regime client. Thin wrapper around the
 * Python ml-engine /ml/regime endpoint (which uses the shared Phase-1
 * feature vector). Used to stamp predictions journal rows with a per-coin
 * label that's more specific than the basket-level state. Callers must
 * tolerate `null` (the engine may be unreachable or the coin may not have
 * enough candle history yet).
 */
export async function fetchCoinRegimeLabel(
  coinId: string,
  timeframe: TimeframeKey,
): Promise<RegimeLabel | null> {
  try {
    const r = await getMlRegime(coinId, timeframe);
    if (r.insufficientData) return null;
    return r.label;
  } catch (err) {
    logger.debug({ err: String(err), coinId, timeframe }, "fetchCoinRegimeLabel failed");
    return null;
  }
}

export function getRegimeAdjustments(regime: MarketRegime): { slMultiplier: number; tpMultiplier: number; confidenceModifier: number } {
  switch (regime) {
    case "volatile":
      return { slMultiplier: 1.4, tpMultiplier: 1.3, confidenceModifier: -0.05 };
    case "bull":
      return { slMultiplier: 1.0, tpMultiplier: 1.2, confidenceModifier: 0.03 };
    case "bear":
      return { slMultiplier: 1.1, tpMultiplier: 0.9, confidenceModifier: -0.03 };
    case "sideways":
      return { slMultiplier: 0.9, tpMultiplier: 0.8, confidenceModifier: 0 };
  }
}

export async function findSequenceMatches(
  coinId: string,
  timeframe: string,
  currentRegime: MarketRegime
): Promise<SequenceMatch[]> {
  const recentFps = getRecentFingerprints(coinId, timeframe);
  if (recentFps.length < 2) return [];

  const last2 = recentFps.slice(-2);
  const last3 = recentFps.length >= 3 ? recentFps.slice(-3) : null;

  const cutoff = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  const historicalInsights = await db
    .select({
      fingerprint: coinInsightsTable.fingerprint,
      direction: coinInsightsTable.direction,
      priceChangePercent: coinInsightsTable.priceChangePercent,
      regime: coinInsightsTable.regime,
      createdAt: coinInsightsTable.createdAt,
    })
    .from(coinInsightsTable)
    .where(and(
      eq(coinInsightsTable.coinId, coinId),
      eq(coinInsightsTable.timeframe, timeframe),
      gte(coinInsightsTable.createdAt, cutoff),
      sql`${coinInsightsTable.fingerprint} IS NOT NULL`
    ))
    .orderBy(coinInsightsTable.createdAt)
    .limit(500);

  if (historicalInsights.length < 3) return [];

  const matches: SequenceMatch[] = [];

  const fpList = historicalInsights.map(h => ({
    fp: h.fingerprint!,
    move: h.priceChangePercent,
    regime: h.regime,
  }));

  const twoStepResults = matchNStepSequence(fpList, last2, 2, currentRegime);
  if (twoStepResults) matches.push(twoStepResults);

  if (last3) {
    const threeStepResults = matchNStepSequence(fpList, last3, 3, currentRegime);
    if (threeStepResults) matches.push(threeStepResults);
  }

  return matches;
}

function matchNStepSequence(
  history: { fp: string; move: number; regime: string | null }[],
  targetSequence: string[],
  n: number,
  currentRegime: MarketRegime
): SequenceMatch | null {
  const allOutcomes: { move: number; sameRegime: boolean }[] = [];
  const sameRegimeOutcomes: { move: number }[] = [];

  for (let i = 0; i <= history.length - n - 1; i++) {
    let seqMatch = true;
    for (let j = 0; j < n; j++) {
      if (history[i + j].fp !== targetSequence[j]) {
        seqMatch = false;
        break;
      }
    }
    if (seqMatch) {
      const next = history[i + n];
      if (next) {
        const isSameRegime = next.regime === currentRegime;
        allOutcomes.push({ move: next.move, sameRegime: isSameRegime });
        if (isSameRegime) sameRegimeOutcomes.push({ move: next.move });
      }
    }
  }

  if (allOutcomes.length < 2) return null;

  const useSameRegime = sameRegimeOutcomes.length >= 2;
  const useSet = useSameRegime ? sameRegimeOutcomes : allOutcomes;

  const bullishMoves = useSet.filter(o => o.move > 0.1).length;
  const bearishMoves = useSet.filter(o => o.move < -0.1).length;
  const directionalTotal = bullishMoves + bearishMoves;
  if (directionalTotal === 0) return null;

  const avgMove = useSet.reduce((s, o) => s + o.move, 0) / useSet.length;
  const winRate = bullishMoves / directionalTotal;

  return {
    sequence: targetSequence,
    outcomeAfter: avgMove > 0 ? "bullish" : "bearish",
    avgPriceMove: avgMove,
    winRate,
    sampleCount: useSet.length,
    regime: currentRegime,
    sameRegimeCount: sameRegimeOutcomes.length,
  };
}

export function formatTemporalContext(ctx: TemporalContext): string {
  const parts: string[] = [];

  parts.push(`MARKET REGIME: ${ctx.regime.regime.toUpperCase()} (confidence: ${(ctx.regime.confidence * 100).toFixed(0)}%)`);
  parts.push(`  Avg 24h change: ${ctx.regime.avgChange24h.toFixed(2)}% | Cross-coin volatility: ${ctx.regime.crossCoinVolatility.toFixed(2)}% | Bullish ratio: ${(ctx.regime.bullishRatio * 100).toFixed(0)}%`);

  if (ctx.regime.fearGreedValue != null) {
    parts.push(`  Fear & Greed: ${ctx.regime.fearGreedValue}/100`);
  }

  if (ctx.regime.btcDominance != null) {
    parts.push(`  BTC Dominance: ${ctx.regime.btcDominance.toFixed(1)}% (${ctx.regime.btcDominanceDirection ?? "unknown"}) — ${ctx.regime.btcDominanceDirection === "rising" ? "capital flowing to BTC (bearish for alts)" : ctx.regime.btcDominanceDirection === "falling" ? "capital rotating into alts (bullish for alts)" : "stable dominance"}`);
  }

  const adj = ctx.regimeAdjustment;
  if (ctx.regime.regime === "volatile") {
    parts.push(`  REGIME GUIDANCE: High volatility — SL×${adj.slMultiplier}, TP×${adj.tpMultiplier}, confidence${adj.confidenceModifier >= 0 ? "+" : ""}${(adj.confidenceModifier * 100).toFixed(0)}%`);
  } else if (ctx.regime.regime === "bull") {
    parts.push(`  REGIME GUIDANCE: Bullish regime — use as tiebreaker only when this coin's own structure is mixed; do NOT auto-default. SL×${adj.slMultiplier}, TP×${adj.tpMultiplier}, confidence${adj.confidenceModifier >= 0 ? "+" : ""}${(adj.confidenceModifier * 100).toFixed(0)}%`);
  } else if (ctx.regime.regime === "bear") {
    parts.push(`  REGIME GUIDANCE: Bearish regime — use as tiebreaker only when this coin's own structure is mixed; do NOT auto-default to short. Coins decouple from the basket; trust THIS coin's indicators first. SL×${adj.slMultiplier}, TP×${adj.tpMultiplier}, confidence${adj.confidenceModifier >= 0 ? "+" : ""}${(adj.confidenceModifier * 100).toFixed(0)}%`);
  } else {
    parts.push(`  REGIME GUIDANCE: Sideways — reduce size, SL×${adj.slMultiplier}, TP×${adj.tpMultiplier}, mean-reversion expected`);
  }

  if (ctx.recentFingerprints.length > 1) {
    const shown = ctx.recentFingerprints.slice(-4);
    parts.push(`  Recent fingerprint chain (${shown.length} steps): ${shown.join(" → ")}`);
  }

  if (ctx.sequenceMatches.length > 0) {
    parts.push("");
    parts.push("TEMPORAL PATTERN CHAINS:");
    for (const m of ctx.sequenceMatches) {
      const stepLabel = m.sequence.length === 2 ? "2-step" : "3-step";
      const regimeNote = m.sameRegimeCount > 0
        ? ` (${m.sameRegimeCount}/${m.sampleCount} in same ${m.regime} regime)`
        : "";
      parts.push(`  ${stepLabel} sequence [${m.sequence.join(" → ")}]:`);
      parts.push(`    Outcome: ${m.outcomeAfter} | Bullish ratio: ${(m.winRate * 100).toFixed(0)}% over ${m.sampleCount} occurrences | Avg move: ${m.avgPriceMove > 0 ? "+" : ""}${m.avgPriceMove.toFixed(2)}%${regimeNote}`);
    }
  }

  return parts.join("\n");
}

export function getCurrentRegime(): RegimeState | null {
  return cachedRegime;
}

export async function clearRegimeCache(): Promise<void> {
  cachedRegime = null;
  regimeCacheTime = 0;
  fingerprintBuffers.clear();
  fingerprintsLoaded = false;
  try {
    await db.delete(fingerprintBuffersTable);
  } catch {
  }
}
