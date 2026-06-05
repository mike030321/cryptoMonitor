import { db, predictionsTable, agentsTable } from "@workspace/db";
import { eq, and, desc, sql, gte } from "drizzle-orm";
import { logger } from "./logger";
import { MONITORED_COINS, type CoinPrice } from "./coins";

export interface CoinAffinity {
  coinId: string;
  coinName: string;
  accuracy: number;
  predictionCount: number;
  avgPnl: number;
  affinityScore: number;
}

export interface AgentSpecialization {
  agentId: number;
  agentName: string;
  topCoins: CoinAffinity[];
  weakCoins: CoinAffinity[];
  allAffinities: CoinAffinity[];
}

const specializationCache = new Map<number, { data: AgentSpecialization; time: number }>();
const CACHE_TTL = 5 * 60 * 1000;

export async function getAgentSpecialization(agentId: number, agentName: string): Promise<AgentSpecialization> {
  const cached = specializationCache.get(agentId);
  if (cached && Date.now() - cached.time < CACHE_TTL) return cached.data;

  const cutoff = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);

  const resolved = await db
    .select({
      coinId: predictionsTable.coinId,
      coinName: predictionsTable.coinName,
      outcome: predictionsTable.outcome,
      direction: predictionsTable.direction,
      priceAtPrediction: predictionsTable.priceAtPrediction,
      actualPrice: predictionsTable.actualPrice,
    })
    .from(predictionsTable)
    .where(and(
      eq(predictionsTable.agentId, agentId),
      sql`${predictionsTable.outcome} IN ('correct', 'wrong')`,
      gte(predictionsTable.createdAt, cutoff)
    ))
    .orderBy(desc(predictionsTable.createdAt))
    .limit(300);

  const coinStats: Record<string, { correct: number; total: number; pnlSum: number; coinName: string }> = {};

  for (const p of resolved) {
    if (!coinStats[p.coinId]) {
      coinStats[p.coinId] = { correct: 0, total: 0, pnlSum: 0, coinName: p.coinName };
    }
    const stats = coinStats[p.coinId];
    stats.total++;
    if (p.outcome === "correct") stats.correct++;
    if (p.actualPrice != null && p.priceAtPrediction > 0) {
      const rawPnl = ((p.actualPrice - p.priceAtPrediction) / p.priceAtPrediction) * 100;
      stats.pnlSum += p.direction === "down" ? -rawPnl : rawPnl;
    }
  }

  const affinities: CoinAffinity[] = [];
  for (const [coinId, stats] of Object.entries(coinStats)) {
    const accuracy = stats.total > 0 ? stats.correct / stats.total : 0;
    const avgPnl = stats.total > 0 ? stats.pnlSum / stats.total : 0;
    const countFactor = Math.min(1, stats.total / 10);
    const affinityScore = (accuracy * 0.5 + Math.max(0, avgPnl / 5) * 0.3 + countFactor * 0.2) * 100;

    affinities.push({
      coinId,
      coinName: stats.coinName,
      accuracy,
      predictionCount: stats.total,
      avgPnl,
      affinityScore: Math.round(affinityScore * 10) / 10,
    });
  }

  affinities.sort((a, b) => b.affinityScore - a.affinityScore);

  const spec: AgentSpecialization = {
    agentId,
    agentName,
    topCoins: affinities.slice(0, 3),
    weakCoins: affinities.filter(a => a.predictionCount >= 3 && a.accuracy < 0.35).slice(0, 3),
    allAffinities: affinities,
  };

  specializationCache.set(agentId, { data: spec, time: Date.now() });
  return spec;
}

export interface CoinAllocation {
  coin: CoinPrice;
  signalStrength: number;
  isExploration: boolean;
}

export function allocateCoinsForAgent(
  agentId: number,
  specialization: AgentSpecialization,
  rankedCoins: { coin: CoinPrice; signalStrength: number }[],
  maxCoins: number
): CoinAllocation[] {
  const EXPLORE_SLOTS = Math.min(2, Math.max(1, Math.floor(maxCoins * 0.2)));
  const specializedSlots = maxCoins - EXPLORE_SLOTS;

  const affinityMap = new Map(specialization.allAffinities.map(a => [a.coinId, a]));

  const scored = rankedCoins.map(rc => {
    const aff = affinityMap.get(rc.coin.id);
    let boost = 0;
    let penalty = 0;
    if (aff) {
      boost = aff.affinityScore * 0.5;
      if (aff.predictionCount >= 3 && aff.accuracy < 0.35) {
        penalty = 15;
      }
    }
    return {
      ...rc,
      adjustedScore: rc.signalStrength + boost - penalty,
      predictionCount: aff?.predictionCount ?? 0,
    };
  });

  scored.sort((a, b) => b.adjustedScore - a.adjustedScore);

  const selected: CoinAllocation[] = [];
  const usedIds = new Set<string>();

  for (const s of scored) {
    if (selected.length >= specializedSlots) break;
    selected.push({ coin: s.coin, signalStrength: s.signalStrength, isExploration: false });
    usedIds.add(s.coin.id);
  }

  const remainingForExplore = scored.filter(s => !usedIds.has(s.coin.id));
  const underexploredCoins = remainingForExplore.filter(s => s.predictionCount < 5);

  const explorePool = underexploredCoins.length > 0 ? underexploredCoins : remainingForExplore;
  for (let i = 0; i < EXPLORE_SLOTS && explorePool.length > 0; i++) {
    const idx = Math.floor(Math.random() * explorePool.length);
    const pick = explorePool.splice(idx, 1)[0];
    selected.push({ coin: pick.coin, signalStrength: pick.signalStrength, isExploration: true });
    usedIds.add(pick.coin.id);
  }

  return selected;
}

export async function getAllAgentSpecializations(): Promise<AgentSpecialization[]> {
  const agents = await db
    .select({ id: agentsTable.id, name: agentsTable.name })
    .from(agentsTable)
    .limit(20);

  const results: AgentSpecialization[] = [];
  for (const agent of agents) {
    const spec = await getAgentSpecialization(agent.id, agent.name);
    results.push(spec);
  }
  return results;
}
