/**
 * Coin insights & correlations (Task #444 deterministic-quant edition).
 *
 * The legacy module also held an LLM-context layer (formatInsightsContext,
 * formatCorrelationsContext, matchFingerprint, MarketContextSnapshot,
 * deriveContextTags, analyzeContextDifferences) used to assemble prompt
 * strings. After the LLM/news cutover those formatters are deleted —
 * regime-detector and contagion-detector still consume the structured
 * tables (`coin_insights` for fingerprint+outcome history,
 * `coin_correlations` for cross-coin lead/lag).
 */
import { db, coinInsightsTable, coinCorrelationsTable, priceHistoryTable } from "@workspace/db";
import { eq, and, desc, gte, inArray } from "drizzle-orm";
import { logger } from "./logger";
import { MONITORED_COINS } from "./coins";
import type { TimeframeKey } from "./pattern-analyzer";

export interface CoinInsightSummary {
  coinId: string;
  timeframe: string;
  totalInsights: number;
  winRate: number;
  avgPriceMove: number;
  bestPatterns: { pattern: string; winRate: number; count: number }[];
  directionBias: { up: number; down: number };
  rsiZones: {
    oversold: { count: number; winRate: number };
    overbought: { count: number; winRate: number };
  };
  macdSignalAccuracy: { bullish: number; bearish: number };
}

export async function storeCoinInsight(
  coinId: string,
  timeframe: string,
  patternType: string,
  direction: string,
  outcome: string,
  priceChangePercent: number,
  rsiAtPrediction: number | null,
  macdSignal: string | null,
  bbPercentB: number | null,
  volatility: number | null,
  agentId: number | null,
  fingerprint: string | null = null,
  regime: string | null = null,
): Promise<void> {
  try {
    await db.insert(coinInsightsTable).values({
      coinId,
      timeframe,
      patternType,
      direction,
      outcome,
      priceChangePercent,
      rsiAtPrediction,
      macdSignal,
      bbPercentB,
      volatility,
      agentId,
      fingerprint,
      regime,
    });
  } catch (err) {
    logger.error({ err, coinId, timeframe }, "Failed to store coin insight");
  }
}

export async function getCoinInsights(
  coinId: string,
  timeframe: string,
  agentId?: number,
  limit = 100,
): Promise<CoinInsightSummary> {
  const conditions = [
    eq(coinInsightsTable.coinId, coinId),
    eq(coinInsightsTable.timeframe, timeframe),
  ];
  if (agentId != null) {
    conditions.push(eq(coinInsightsTable.agentId, agentId));
  }

  const insights = await db
    .select()
    .from(coinInsightsTable)
    .where(and(...conditions))
    .orderBy(desc(coinInsightsTable.createdAt))
    .limit(limit);

  if (insights.length === 0) {
    return {
      coinId,
      timeframe,
      totalInsights: 0,
      winRate: 0,
      avgPriceMove: 0,
      bestPatterns: [],
      directionBias: { up: 0, down: 0 },
      rsiZones: {
        oversold: { count: 0, winRate: 0 },
        overbought: { count: 0, winRate: 0 },
      },
      macdSignalAccuracy: { bullish: 0, bearish: 0 },
    };
  }

  const nonNeutral = insights.filter((i) => i.outcome !== "neutral");
  const correct = insights.filter((i) => i.outcome === "correct").length;
  const winRate = nonNeutral.length > 0 ? correct / nonNeutral.length : 0;
  const avgPriceMove =
    insights.reduce((s, i) => s + i.priceChangePercent, 0) / insights.length;

  const patternStats: Record<string, { correct: number; total: number }> = {};
  for (const i of insights) {
    if (!patternStats[i.patternType]) patternStats[i.patternType] = { correct: 0, total: 0 };
    patternStats[i.patternType].total++;
    if (i.outcome === "correct") patternStats[i.patternType].correct++;
  }
  const bestPatterns = Object.entries(patternStats)
    .map(([pattern, stats]) => ({
      pattern,
      winRate: stats.total > 0 ? stats.correct / stats.total : 0,
      count: stats.total,
    }))
    .filter((p) => p.count >= 3)
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, 5);

  const upCount = insights.filter((i) => i.direction === "up").length;
  const downCount = insights.filter((i) => i.direction === "down").length;

  const oversoldInsights = insights.filter(
    (i) => i.rsiAtPrediction != null && i.rsiAtPrediction < 30,
  );
  const overboughtInsights = insights.filter(
    (i) => i.rsiAtPrediction != null && i.rsiAtPrediction > 70,
  );

  const rsiZones = {
    oversold: {
      count: oversoldInsights.length,
      winRate:
        oversoldInsights.length > 0
          ? oversoldInsights.filter((i) => i.outcome === "correct").length /
            oversoldInsights.length
          : 0,
    },
    overbought: {
      count: overboughtInsights.length,
      winRate:
        overboughtInsights.length > 0
          ? overboughtInsights.filter((i) => i.outcome === "correct").length /
            overboughtInsights.length
          : 0,
    },
  };

  const bullishMacd = insights.filter((i) => i.macdSignal === "bullish");
  const bearishMacd = insights.filter((i) => i.macdSignal === "bearish");

  const macdSignalAccuracy = {
    bullish:
      bullishMacd.length > 0
        ? bullishMacd.filter((i) => i.outcome === "correct").length /
          bullishMacd.length
        : 0,
    bearish:
      bearishMacd.length > 0
        ? bearishMacd.filter((i) => i.outcome === "correct").length /
          bearishMacd.length
        : 0,
  };

  return {
    coinId,
    timeframe,
    totalInsights: insights.length,
    winRate,
    avgPriceMove,
    bestPatterns,
    directionBias: { up: upCount, down: downCount },
    rsiZones,
    macdSignalAccuracy,
  };
}

export async function computeCoinCorrelations(timeframe: TimeframeKey): Promise<void> {
  const tf = {
    "1m": 60_000,
    "5m": 300_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "6h": 21_600_000,
    "1d": 86_400_000,
  };
  const lookbackMs = (tf[timeframe] || 3_600_000) * 24;
  const since = new Date(Date.now() - lookbackMs);

  const coinIds = MONITORED_COINS.map((c) => c.id);
  const priceData: Record<string, number[]> = {};

  for (const coinId of coinIds) {
    const history = await db
      .select({ price: priceHistoryTable.price })
      .from(priceHistoryTable)
      .where(
        and(
          eq(priceHistoryTable.coinId, coinId),
          gte(priceHistoryTable.timestamp, since),
        ),
      )
      .orderBy(priceHistoryTable.timestamp);

    if (history.length >= 10) {
      priceData[coinId] = history.map((h) => h.price);
    }
  }

  const coins = Object.keys(priceData);
  for (let i = 0; i < coins.length; i++) {
    for (let j = i + 1; j < coins.length; j++) {
      const coinA = coins[i];
      const coinB = coins[j];
      const pricesA = priceData[coinA];
      const pricesB = priceData[coinB];

      const minLen = Math.min(pricesA.length, pricesB.length);
      if (minLen < 10) continue;

      const returnsA: number[] = [];
      const returnsB: number[] = [];
      for (let k = 1; k < minLen; k++) {
        returnsA.push((pricesA[k] - pricesA[k - 1]) / pricesA[k - 1]);
        returnsB.push((pricesB[k] - pricesB[k - 1]) / pricesB[k - 1]);
      }

      const correlation = pearsonCorrelation(returnsA, returnsB);

      await db
        .insert(coinCorrelationsTable)
        .values({
          coinA,
          coinB,
          correlation,
          timeframe,
          sampleSize: returnsA.length,
        })
        .onConflictDoUpdate({
          target: [
            coinCorrelationsTable.coinA,
            coinCorrelationsTable.coinB,
            coinCorrelationsTable.timeframe,
          ],
          set: { correlation, sampleSize: returnsA.length, updatedAt: new Date() },
        });
    }
  }

  logger.info({ timeframe, coins: coins.length }, "Computed coin correlations");
}

export interface CorrelationWithPrice {
  coinId: string;
  correlation: number;
  recentPriceChange?: number;
  coinName?: string;
}

export async function getCorrelationsForCoin(
  coinId: string,
  timeframe: string,
): Promise<CorrelationWithPrice[]> {
  const correlationsA = await db
    .select()
    .from(coinCorrelationsTable)
    .where(
      and(
        eq(coinCorrelationsTable.coinA, coinId),
        eq(coinCorrelationsTable.timeframe, timeframe),
      ),
    );

  const correlationsB = await db
    .select()
    .from(coinCorrelationsTable)
    .where(
      and(
        eq(coinCorrelationsTable.coinB, coinId),
        eq(coinCorrelationsTable.timeframe, timeframe),
      ),
    );

  const result: CorrelationWithPrice[] = [];
  for (const c of correlationsA) {
    result.push({ coinId: c.coinB, correlation: c.correlation });
  }
  for (const c of correlationsB) {
    result.push({ coinId: c.coinA, correlation: c.correlation });
  }

  const coinIds = result.map((r) => r.coinId);
  if (coinIds.length > 0) {
    try {
      const tfMs = {
        "1m": 60_000,
        "5m": 300_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "6h": 21_600_000,
        "1d": 86_400_000,
      };
      const lookbackMs = (tfMs[timeframe as keyof typeof tfMs] || 3_600_000) * 2;
      const since = new Date(Date.now() - lookbackMs);

      const allPrices = await db
        .select({
          coinId: priceHistoryTable.coinId,
          price: priceHistoryTable.price,
          timestamp: priceHistoryTable.timestamp,
        })
        .from(priceHistoryTable)
        .where(
          and(
            inArray(priceHistoryTable.coinId, coinIds),
            gte(priceHistoryTable.timestamp, since),
          ),
        )
        .orderBy(priceHistoryTable.coinId, priceHistoryTable.timestamp);

      const pricesByCoin: Record<string, number[]> = {};
      for (const row of allPrices) {
        if (!pricesByCoin[row.coinId]) pricesByCoin[row.coinId] = [];
        pricesByCoin[row.coinId].push(row.price);
      }

      for (const corr of result) {
        const prices = pricesByCoin[corr.coinId];
        if (prices && prices.length >= 2) {
          corr.recentPriceChange =
            ((prices[prices.length - 1] - prices[0]) / prices[0]) * 100;
        }
        const coin = MONITORED_COINS.find((c) => c.id === corr.coinId);
        if (coin) corr.coinName = coin.name;
      }
    } catch {
      for (const corr of result) {
        const coin = MONITORED_COINS.find((c) => c.id === corr.coinId);
        if (coin) corr.coinName = coin.name;
      }
    }
  }

  return result.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
}

function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n === 0) return 0;

  const sumX = x.reduce((s, v) => s + v, 0);
  const sumY = y.reduce((s, v) => s + v, 0);
  const sumXY = x.reduce((s, v, i) => s + v * y[i], 0);
  const sumX2 = x.reduce((s, v) => s + v * v, 0);
  const sumY2 = y.reduce((s, v) => s + v * v, 0);

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  if (denominator === 0) return 0;
  return Math.max(-1, Math.min(1, numerator / denominator));
}
