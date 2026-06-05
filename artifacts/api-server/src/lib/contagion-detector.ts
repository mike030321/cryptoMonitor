import { db, priceHistoryTable, coinCorrelationsTable } from "@workspace/db";
import { eq, and, gte, sql } from "drizzle-orm";
import { logger } from "./logger";
import { MONITORED_COINS, type CoinPrice } from "./coins";

export interface ContagionAlert {
  sourceCoinId: string;
  sourceCoinName: string;
  sourceMove: number;
  targetCoinId: string;
  targetCoinName: string;
  correlation: number;
  followProbability: number;
  expectedMove: number;
  lagHours: number;
  direction: "bullish" | "bearish";
  sampleSize: number;
  createdAt: number;
}

interface HistoricalContagionStats {
  followRate: number;
  medianLagHours: number;
  avgFollowMove: number;
  sampleSize: number;
}

const SIGNIFICANT_MOVE_PCT = 2.0;
const ALERT_TTL = 2 * 60 * 60 * 1000;
const DETECTION_LOOKBACK = 60 * 60 * 1000;
const MIN_CORRELATION = 0.3;
const MIN_HISTORICAL_SAMPLES = 3;
const HISTORY_LOOKBACK_DAYS = 7;

let activeAlerts: ContagionAlert[] = [];
let lastDetectionTime = 0;
const DETECTION_COOLDOWN = 60_000;

const contagionStatsCache = new Map<string, { stats: HistoricalContagionStats; time: number }>();
const STATS_CACHE_TTL = 30 * 60 * 1000;

async function computeHistoricalContagion(
  sourceCoinId: string,
  targetCoinId: string,
  sourceDirection: "up" | "down",
  correlationSign: number
): Promise<HistoricalContagionStats> {
  const expectTargetUp = (sourceDirection === "up" && correlationSign >= 0) || (sourceDirection === "down" && correlationSign < 0);
  const expectedTargetDir = expectTargetUp ? "up" : "down";
  const cacheKey = `${sourceCoinId}:${targetCoinId}:${sourceDirection}:${expectedTargetDir}`;
  const cached = contagionStatsCache.get(cacheKey);
  if (cached && Date.now() - cached.time < STATS_CACHE_TTL) return cached.stats;

  const lookbackStart = new Date(Date.now() - HISTORY_LOOKBACK_DAYS * 24 * 60 * 60 * 1000);

  const sourcePrices = await db
    .select({ price: priceHistoryTable.price, timestamp: priceHistoryTable.timestamp })
    .from(priceHistoryTable)
    .where(and(
      eq(priceHistoryTable.coinId, sourceCoinId),
      gte(priceHistoryTable.timestamp, lookbackStart)
    ))
    .orderBy(priceHistoryTable.timestamp);

  if (sourcePrices.length < 10) {
    const fallback: HistoricalContagionStats = { followRate: 0.4, medianLagHours: 2, avgFollowMove: 0, sampleSize: 0 };
    contagionStatsCache.set(cacheKey, { stats: fallback, time: Date.now() });
    return fallback;
  }

  const ROLLING_WINDOW_MS = 60 * 60 * 1000;
  const sourceEvents: { timestamp: Date; movePct: number }[] = [];
  let lastEventTs = 0;
  for (let i = 0; i < sourcePrices.length; i++) {
    const curr = sourcePrices[i];
    const currTs = curr.timestamp.getTime();
    if (currTs - lastEventTs < ROLLING_WINDOW_MS / 2) continue;

    let windowStart = sourcePrices.findIndex(
      p => p.timestamp.getTime() >= currTs - ROLLING_WINDOW_MS
    );
    if (windowStart < 0 || windowStart >= i) continue;

    const basePrice = sourcePrices[windowStart].price;
    const movePct = ((curr.price - basePrice) / basePrice) * 100;
    if (Math.abs(movePct) >= SIGNIFICANT_MOVE_PCT) {
      const isUp = movePct > 0;
      if ((sourceDirection === "up" && isUp) || (sourceDirection === "down" && !isUp)) {
        sourceEvents.push({ timestamp: curr.timestamp, movePct });
        lastEventTs = currTs;
      }
    }
  }

  if (sourceEvents.length < 2) {
    const fallback: HistoricalContagionStats = { followRate: 0.4, medianLagHours: 2, avgFollowMove: 0, sampleSize: 0 };
    contagionStatsCache.set(cacheKey, { stats: fallback, time: Date.now() });
    return fallback;
  }

  const targetPrices = await db
    .select({ price: priceHistoryTable.price, timestamp: priceHistoryTable.timestamp })
    .from(priceHistoryTable)
    .where(and(
      eq(priceHistoryTable.coinId, targetCoinId),
      gte(priceHistoryTable.timestamp, lookbackStart)
    ))
    .orderBy(priceHistoryTable.timestamp);

  if (targetPrices.length < 10) {
    const fallback: HistoricalContagionStats = { followRate: 0.4, medianLagHours: 2, avgFollowMove: 0, sampleSize: 0 };
    contagionStatsCache.set(cacheKey, { stats: fallback, time: Date.now() });
    return fallback;
  }

  let followCount = 0;
  const lagHoursList: number[] = [];
  const followMoves: number[] = [];
  const FOLLOW_WINDOW_MS = 6 * 60 * 60 * 1000;
  const FOLLOW_THRESHOLD_PCT = 1.0;

  for (const event of sourceEvents) {
    const eventTs = event.timestamp.getTime();
    const windowEnd = eventTs + FOLLOW_WINDOW_MS;

    const targetAtEvent = targetPrices.find(p => p.timestamp.getTime() >= eventTs);
    if (!targetAtEvent) continue;

    const targetAfter = targetPrices.filter(
      p => p.timestamp.getTime() > eventTs && p.timestamp.getTime() <= windowEnd
    );

    if (targetAfter.length === 0) continue;

    const basePrice = targetAtEvent.price;
    let followed = false;

    for (const tp of targetAfter) {
      const targetMove = ((tp.price - basePrice) / basePrice) * 100;
      const sameDirection = expectedTargetDir === "up" ? targetMove > FOLLOW_THRESHOLD_PCT : targetMove < -FOLLOW_THRESHOLD_PCT;

      if (sameDirection) {
        followed = true;
        const lagMs = tp.timestamp.getTime() - eventTs;
        lagHoursList.push(lagMs / (60 * 60 * 1000));
        followMoves.push(targetMove);
        break;
      }
    }

    if (followed) followCount++;
  }

  const sampleSize = sourceEvents.length;
  const followRate = sampleSize > 0 ? followCount / sampleSize : 0;

  lagHoursList.sort((a, b) => a - b);
  const medianLagHours = lagHoursList.length > 0
    ? lagHoursList[Math.floor(lagHoursList.length / 2)]
    : 2;

  const avgFollowMove = followMoves.length > 0
    ? followMoves.reduce((s, v) => s + v, 0) / followMoves.length
    : 0;

  const stats: HistoricalContagionStats = {
    followRate: Math.max(0.05, Math.min(0.95, followRate)),
    medianLagHours: Math.round(medianLagHours * 10) / 10,
    avgFollowMove: Math.round(avgFollowMove * 100) / 100,
    sampleSize,
  };

  contagionStatsCache.set(cacheKey, { stats, time: Date.now() });
  return stats;
}

export async function detectContagionEvents(prices: CoinPrice[]): Promise<ContagionAlert[]> {
  const now = Date.now();
  if (now - lastDetectionTime < DETECTION_COOLDOWN) return activeAlerts;
  lastDetectionTime = now;

  activeAlerts = activeAlerts.filter(a => now - a.createdAt < ALERT_TTL);

  const since = new Date(now - DETECTION_LOOKBACK);
  const significantMovers: { coinId: string; coinName: string; move: number }[] = [];

  for (const coin of prices) {
    try {
      const recentPrices = await db
        .select({ price: priceHistoryTable.price, timestamp: priceHistoryTable.timestamp })
        .from(priceHistoryTable)
        .where(and(
          eq(priceHistoryTable.coinId, coin.id),
          gte(priceHistoryTable.timestamp, since)
        ))
        .orderBy(priceHistoryTable.timestamp)
        .limit(100);

      if (recentPrices.length < 2) continue;

      const oldestPrice = recentPrices[0].price;
      const currentPrice = coin.currentPrice;
      const movePct = ((currentPrice - oldestPrice) / oldestPrice) * 100;

      if (Math.abs(movePct) >= SIGNIFICANT_MOVE_PCT) {
        significantMovers.push({ coinId: coin.id, coinName: coin.name, move: movePct });
      }
    } catch (err) {
      logger.warn({ err, coinId: coin.id }, "Contagion: failed to read recent prices for coin");
      continue;
    }
  }

  if (significantMovers.length === 0) return activeAlerts;

  for (const mover of significantMovers) {
    const sourceDir: "up" | "down" = mover.move > 0 ? "up" : "down";

    try {
      const correlationsA = await db
        .select({
          coinB: coinCorrelationsTable.coinB,
          correlation: coinCorrelationsTable.correlation,
        })
        .from(coinCorrelationsTable)
        .where(and(
          eq(coinCorrelationsTable.coinA, mover.coinId),
          sql`ABS(${coinCorrelationsTable.correlation}) > ${MIN_CORRELATION}`,
          gte(coinCorrelationsTable.updatedAt, new Date(now - 24 * 60 * 60 * 1000))
        ));

      const correlationsB = await db
        .select({
          coinA: coinCorrelationsTable.coinA,
          correlation: coinCorrelationsTable.correlation,
        })
        .from(coinCorrelationsTable)
        .where(and(
          eq(coinCorrelationsTable.coinB, mover.coinId),
          sql`ABS(${coinCorrelationsTable.correlation}) > ${MIN_CORRELATION}`,
          gte(coinCorrelationsTable.updatedAt, new Date(now - 24 * 60 * 60 * 1000))
        ));

      const correlated: { coinId: string; correlation: number }[] = [];
      for (const c of correlationsA) correlated.push({ coinId: c.coinB, correlation: c.correlation });
      for (const c of correlationsB) correlated.push({ coinId: c.coinA, correlation: c.correlation });

      for (const target of correlated) {
        const corrBasedMove = mover.move * target.correlation * 0.6;
        const targetDir = corrBasedMove > 0 ? "bullish" : "bearish";
        const alreadyHasAlert = activeAlerts.find(
          a => a.sourceCoinId === mover.coinId && a.targetCoinId === target.coinId && a.direction === targetDir
        );
        if (alreadyHasAlert) continue;

        const historicalStats = await computeHistoricalContagion(
          mover.coinId,
          target.coinId,
          sourceDir,
          target.correlation
        );

        const followProb = historicalStats.sampleSize >= MIN_HISTORICAL_SAMPLES
          ? historicalStats.followRate
          : Math.min(0.85, 0.4 + Math.abs(target.correlation) * 0.4);

        const lagHours = historicalStats.sampleSize >= MIN_HISTORICAL_SAMPLES
          ? Math.max(0.5, historicalStats.medianLagHours)
          : (Math.abs(target.correlation) > 0.7 ? 1 : Math.abs(target.correlation) > 0.5 ? 2 : 4);

        const expectedMove = historicalStats.sampleSize >= MIN_HISTORICAL_SAMPLES && historicalStats.avgFollowMove !== 0
          ? historicalStats.avgFollowMove
          : corrBasedMove;

        const direction = targetDir;

        const targetCoin = MONITORED_COINS.find(c => c.id === target.coinId);
        if (!targetCoin) continue;

        const alert: ContagionAlert = {
          sourceCoinId: mover.coinId,
          sourceCoinName: mover.coinName,
          sourceMove: mover.move,
          targetCoinId: target.coinId,
          targetCoinName: targetCoin.name,
          correlation: target.correlation,
          followProbability: followProb,
          expectedMove,
          lagHours,
          direction,
          sampleSize: historicalStats.sampleSize,
          createdAt: now,
        };

        activeAlerts.push(alert);

        logger.info({
          source: mover.coinName,
          sourceMove: `${mover.move > 0 ? "+" : ""}${mover.move.toFixed(1)}%`,
          target: targetCoin.name,
          prob: `${(followProb * 100).toFixed(0)}%`,
          expectedMove: `${expectedMove > 0 ? "+" : ""}${expectedMove.toFixed(1)}%`,
          lag: `${lagHours}h`,
          sampleSize: historicalStats.sampleSize,
          empirical: historicalStats.sampleSize >= MIN_HISTORICAL_SAMPLES,
        }, "Contagion alert generated");
      }
    } catch (err) {
      logger.warn({ err, sourceCoinId: mover.coinId }, "Contagion: failed to compute alerts for mover");
      continue;
    }
  }

  return activeAlerts;
}

export function getContagionAlertsForCoin(coinId: string): ContagionAlert[] {
  const now = Date.now();
  return activeAlerts.filter(a => a.targetCoinId === coinId && now - a.createdAt < ALERT_TTL);
}

export function formatContagionContext(alerts: ContagionAlert[]): string {
  if (alerts.length === 0) return "";

  const lines = ["CONTAGION ALERTS (correlated coin significant moves):"];
  for (const a of alerts.slice(0, 3)) {
    const ageMin = Math.round((Date.now() - a.createdAt) / 60_000);
    const basis = a.sampleSize >= MIN_HISTORICAL_SAMPLES ? `empirical (${a.sampleSize} events)` : "correlation-based estimate";
    lines.push(`  ${a.sourceCoinName} moved ${a.sourceMove > 0 ? "+" : ""}${a.sourceMove.toFixed(1)}% (${ageMin}m ago)`);
    lines.push(`    → Follow probability: ${(a.followProbability * 100).toFixed(0)}% [${basis}] | Expected move: ${a.expectedMove > 0 ? "+" : ""}${a.expectedMove.toFixed(1)}% within ${a.lagHours}h`);
    lines.push(`    → Signal: ${a.direction.toUpperCase()} — consider this when setting direction and confidence`);
  }
  return lines.join("\n");
}

export function getActiveAlerts(): ContagionAlert[] {
  const now = Date.now();
  return activeAlerts.filter(a => now - a.createdAt < ALERT_TTL);
}

export function clearContagionCache(): void {
  activeAlerts = [];
  lastDetectionTime = 0;
  contagionStatsCache.clear();
}
