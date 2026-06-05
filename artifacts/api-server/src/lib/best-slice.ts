/**
 * Best-slice computer (Task #444 deterministic-quant edition).
 *
 * Replaces the legacy `_legacy/ai-engine#generateBestPick` which combined
 * agent votes with news/sentiment/copy and an LLM-shaped reasoning blurb.
 * After the LLM/news cutover the dashboard's "best pick" is just the
 * coin with the strongest weighted directional consensus across recent
 * deterministic predictions — no news factors, no narrative reasoning,
 * no sentiment string, no `brain: "LLM"` branch.
 *
 * The resulting `BestPickResult` is the canonical shape used by the
 * dashboard contract (`/crypto/best-pick`), kept structurally compatible
 * with the legacy export so client code keeps rendering. LLM-only fields
 * (`newsFactors`, `newsTags`, `whyExplanation`) are omitted from the
 * payload — the React client already treats them as optional.
 */
import type { CoinPrice } from "./coins";
import { TIMEFRAMES, type TimeframeKey } from "./pattern-analyzer";
import { buildWhatExplanation, type BestPickResult } from "./quant-types";

export type { BestPickResult };

interface AgentRow {
  id: number;
  name: string;
  score: number;
  totalPredictions: number;
  correctPredictions: number;
  status: string;
}

interface PredictionRow {
  agentId: number;
  agentName: string;
  coinId: string;
  coinName: string;
  direction: string;
  confidence: number;
  reasoning: string;
  priceAtPrediction: number;
  predictedPrice: number;
  outcome: string;
  timeframe: string | null;
  createdAt: Date;
}

const TIMEFRAME_WEIGHTS: Record<string, number> = {
  "1m": 0.5,
  "5m": 0.8,
  "1h": 1.2,
  "2h": 1.5,
  "6h": 1.8,
  "1d": 2.0,
};

function defaultPick(coin: CoinPrice): BestPickResult {
  return {
    coinId: coin.id,
    coinName: coin.name,
    coinSymbol: coin.symbol,
    currentPrice: coin.currentPrice,
    action: "hold",
    successProbability: 0.3,
    holdTimeframe: TIMEFRAMES["1h"].label,
    holdMinutes: 60,
    expectedPriceChange: 0,
    reasoning:
      "[BRAIN=ABSTAIN] No recent directional consensus across active agents — holding.",
    agentConsensus: [],
    riskLevel: "medium",
    timeframeBreakdown: [],
    whatExplanation: buildWhatExplanation({
      bestAction: "hold",
      coinSymbol: coin.symbol,
      brain: "ABSTAIN",
      modelProbability: null,
      holdTimeframe: TIMEFRAMES["1h"].label,
    }),
    brain: "ABSTAIN",
    updatedAt: new Date().toISOString(),
  };
}

export async function computeBestSlice(
  agents: AgentRow[],
  recentPredictions: PredictionRow[],
  coins: CoinPrice[],
): Promise<BestPickResult> {
  if (coins.length === 0) {
    // Genuinely empty fleet — synthesise a neutral pick so the UI has a row.
    return defaultPick({
      id: "unknown",
      name: "Unknown",
      symbol: "?",
      currentPrice: 0,
      priceChange24h: 0,
      volume24h: 0,
      marketCap: 0,
      lastUpdated: new Date().toISOString(),
      // Synthetic placeholder used only when the fleet is genuinely empty;
      // marked non-live so downstream cards can render an "—" state.
      isLiveData: false,
    });
  }

  const tenMinAgo = Date.now() - 10 * 60 * 1000;
  const fresh = recentPredictions.filter((p) => {
    const ts =
      p.createdAt instanceof Date
        ? p.createdAt.getTime()
        : new Date(p.createdAt).getTime();
    return ts > tenMinAgo;
  });
  const working = fresh.length > 5 ? fresh : recentPredictions.slice(0, 80);

  type CoinScore = {
    buyScore: number;
    sellScore: number;
    totalWeight: number;
    predictions: PredictionRow[];
  };
  const coinScores: Record<string, CoinScore> = {};

  for (const pred of working) {
    const agent = agents.find((a) => a.id === pred.agentId);
    if (!agent) continue;
    const accuracy =
      agent.totalPredictions > 0
        ? agent.correctPredictions / agent.totalPredictions
        : 0.5;
    const scoreWeight = Math.max(agent.score, 10) / 100;
    const tfWeight = TIMEFRAME_WEIGHTS[pred.timeframe || "1m"] || 1;
    const weight = scoreWeight * accuracy * pred.confidence * tfWeight;

    if (!coinScores[pred.coinId]) {
      coinScores[pred.coinId] = {
        buyScore: 0,
        sellScore: 0,
        totalWeight: 0,
        predictions: [],
      };
    }
    if (pred.direction === "up") {
      coinScores[pred.coinId].buyScore += weight;
      coinScores[pred.coinId].totalWeight += weight;
    } else if (pred.direction === "down") {
      coinScores[pred.coinId].sellScore += weight;
      coinScores[pred.coinId].totalWeight += weight;
    }
    // "stable" votes are abstentions — they MUST NOT contribute to either
    // directional score nor to totalWeight.
    coinScores[pred.coinId].predictions.push(pred);
  }

  let bestCoinId = "";
  let bestScore = -1;
  let bestAction: "buy" | "sell" | "hold" = "hold";

  for (const [coinId, scores] of Object.entries(coinScores)) {
    const netScore = Math.abs(scores.buyScore - scores.sellScore);
    if (netScore > bestScore) {
      bestScore = netScore;
      bestCoinId = coinId;
      if (scores.buyScore > scores.sellScore) bestAction = "buy";
      else if (scores.sellScore > scores.buyScore) bestAction = "sell";
      else bestAction = "hold";
    }
  }

  if (!bestCoinId) {
    return defaultPick(coins[0]);
  }

  const bestCoin = coins.find((c) => c.id === bestCoinId);
  if (!bestCoin) return defaultPick(coins[0]);

  const coinPreds = coinScores[bestCoinId];

  // Per-agent latest vote.
  const agentVoteMap = new Map<
    number,
    {
      agentName: string;
      direction: string;
      confidence: number;
      score: number;
      accuracy: number;
    }
  >();
  for (const p of coinPreds.predictions) {
    const agent = agents.find((a) => a.id === p.agentId);
    const existing = agentVoteMap.get(p.agentId);
    if (!existing || p.confidence > existing.confidence) {
      agentVoteMap.set(p.agentId, {
        agentName: p.agentName,
        direction: p.direction,
        confidence: p.confidence,
        score: agent?.score ?? 0,
        accuracy:
          agent && agent.totalPredictions > 0
            ? Math.round((agent.correctPredictions / agent.totalPredictions) * 100)
            : 0,
      });
    }
  }
  const agentConsensus = Array.from(agentVoteMap.values());

  // Per-timeframe direction summary.
  const tfBreakdownAcc: Record<
    string,
    { directions: string[]; confidences: number[] }
  > = {};
  for (const p of coinPreds.predictions) {
    const tf = p.timeframe || "1m";
    if (!tfBreakdownAcc[tf]) tfBreakdownAcc[tf] = { directions: [], confidences: [] };
    tfBreakdownAcc[tf].directions.push(p.direction);
    tfBreakdownAcc[tf].confidences.push(p.confidence);
  }
  const tfOrder = ["1m", "5m", "1h", "2h", "6h", "1d"];
  const timeframeBreakdown = Object.entries(tfBreakdownAcc)
    .map(([tf, data]) => {
      const upCount = data.directions.filter((d) => d === "up").length;
      const downCount = data.directions.filter((d) => d === "down").length;
      return {
        timeframe: tf,
        direction:
          upCount > downCount ? "up" : downCount > upCount ? "down" : "stable",
        avgConfidence:
          data.confidences.reduce((s, c) => s + c, 0) / data.confidences.length,
        agentCount: data.directions.length,
      };
    })
    .sort((a, b) => tfOrder.indexOf(a.timeframe) - tfOrder.indexOf(b.timeframe));

  const totalWeight = coinPreds.totalWeight || 1;
  const dominantWeight =
    bestAction === "buy" ? coinPreds.buyScore : bestAction === "sell" ? coinPreds.sellScore : 0;
  const consensus = dominantWeight / totalWeight;
  const avgAgentAccuracy =
    agents.length > 0
      ? agents.reduce(
          (s, a) =>
            s + (a.totalPredictions > 0 ? a.correctPredictions / a.totalPredictions : 0),
          0,
        ) / agents.length
      : 0.3;
  const tfAlignment =
    timeframeBreakdown.length > 1
      ? timeframeBreakdown.filter(
          (t) => t.direction === (bestAction === "buy" ? "up" : "down"),
        ).length / timeframeBreakdown.length
      : 0.5;
  // No news term — pure deterministic blend of consensus, accuracy,
  // timeframe alignment, and the raw best-score margin.
  const rawProbability =
    consensus * 0.4 +
    avgAgentAccuracy * 0.3 +
    tfAlignment * 0.2 +
    Math.min(bestScore * 0.15, 0.1);
  const successProbability = Math.min(0.85, Math.max(0.15, rawProbability));

  const avgPriceChange =
    coinPreds.predictions.length > 0
      ? coinPreds.predictions.reduce(
          (sum, p) =>
            sum +
            ((p.predictedPrice - p.priceAtPrediction) / p.priceAtPrediction) * 100,
          0,
        ) / coinPreds.predictions.length
      : 0;

  const longestAlignedTf = [...timeframeBreakdown]
    .reverse()
    .find((t) => t.direction === (bestAction === "buy" ? "up" : "down"));
  const volatility = Math.abs(bestCoin.priceChange24h ?? 0);

  let holdMinutes: number;
  let holdTimeframe: string;
  if (longestAlignedTf) {
    const tfMs = TIMEFRAMES[longestAlignedTf.timeframe as TimeframeKey]?.ms || 60000;
    holdMinutes = Math.round(tfMs / 60000);
    const tfLabel =
      TIMEFRAMES[longestAlignedTf.timeframe as TimeframeKey]?.label ||
      longestAlignedTf.timeframe;
    holdTimeframe = `${tfLabel} (aligned across ${
      timeframeBreakdown.filter((t) => t.direction === longestAlignedTf.direction).length
    } timeframes)`;
  } else if (volatility > 10) {
    holdMinutes = 5;
    holdTimeframe = "5 minutes (high volatility)";
  } else if (volatility > 5) {
    holdMinutes = 30;
    holdTimeframe = "30 minutes (moderate volatility)";
  } else {
    holdMinutes = 60;
    holdTimeframe = "1 hour (normal conditions)";
  }

  const riskLevel: BestPickResult["riskLevel"] =
    volatility > 15
      ? "extreme"
      : volatility > 8
        ? "high"
        : volatility > 3
          ? "medium"
          : "low";

  const reasoning = `[BRAIN=QUANT] ${agentConsensus.length} agent vote${
    agentConsensus.length === 1 ? "" : "s"
  } produced a ${(consensus * 100).toFixed(0)}% directional consensus on ${
    bestCoin.symbol
  } (${bestAction}). Timeframe alignment ${(tfAlignment * 100).toFixed(0)}%; fleet accuracy ${(
    avgAgentAccuracy * 100
  ).toFixed(0)}%.`;

  return {
    coinId: bestCoin.id,
    coinName: bestCoin.name,
    coinSymbol: bestCoin.symbol,
    currentPrice: bestCoin.currentPrice,
    action: bestAction,
    successProbability,
    holdTimeframe,
    holdMinutes,
    expectedPriceChange: avgPriceChange,
    reasoning,
    agentConsensus,
    riskLevel,
    timeframeBreakdown,
    whatExplanation: buildWhatExplanation({
      bestAction,
      coinSymbol: bestCoin.symbol,
      brain: "QUANT",
      modelProbability: successProbability,
      holdTimeframe,
    }),
    brain: "QUANT",
    updatedAt: new Date().toISOString(),
  };
}
