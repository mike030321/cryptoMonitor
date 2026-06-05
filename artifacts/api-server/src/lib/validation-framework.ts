import { db, predictionsTable, paperTradesTable, paperPortfoliosTable, priceHistoryTable } from "@workspace/db";
import { eq, and, gte, desc, sql, ne } from "drizzle-orm";
import { logger } from "./logger";
import { MONITORED_COINS } from "./coins";

interface BaselineResult {
  strategy: string;
  totalTrades: number;
  winRate: number;
  avgReturn: number;
  totalReturn: number;
  sharpeProxy: number;
  maxDrawdown: number;
}

interface ValidationReport {
  reportGeneratedAt: string;
  periodHours: number;
  systemPerformance: {
    totalPredictions: number;
    correctPredictions: number;
    wrongPredictions: number;
    neutralPredictions: number;
    accuracy: number;
    avgConfidence: number;
    avgConfidenceWhenCorrect: number;
    avgConfidenceWhenWrong: number;
    calibrationError: number;
    totalTrades: number;
    tradingWinRate: number;
    totalPnl: number;
    avgTradeReturn: number;
    bestTrade: number;
    worstTrade: number;
    profitFactor: number;
  };
  baselines: BaselineResult[];
  systemVsBaselines: {
    strategy: string;
    systemEdge: number;
    systemBetter: boolean;
  }[];
  byTimeframe: Record<string, {
    predictions: number;
    accuracy: number;
    trades: number;
    winRate: number;
    avgReturn: number;
  }>;
  byAgent: {
    name: string;
    predictions: number;
    accuracy: number;
    trades: number;
    winRate: number;
    pnl: number;
    uniqueBehavior: number;
  }[];
  ablationImpact: Record<string, {
    enabled: boolean;
    description: string;
  }>;
  warnings: string[];
}

async function computeBaselines(periodHours: number): Promise<BaselineResult[]> {
  const since = new Date(Date.now() - periodHours * 60 * 60 * 1000);
  const baselines: BaselineResult[] = [];

  const priceSnapshots: Record<string, { price: number; timestamp: Date }[]> = {};
  for (const coin of MONITORED_COINS) {
    const history = await db
      .select({ price: priceHistoryTable.price, timestamp: priceHistoryTable.timestamp })
      .from(priceHistoryTable)
      .where(and(eq(priceHistoryTable.coinId, coin.id), gte(priceHistoryTable.timestamp, since)))
      .orderBy(priceHistoryTable.timestamp);
    if (history.length >= 10) {
      priceSnapshots[coin.id] = history;
    }
  }

  const buyHoldResults: number[] = [];
  for (const [, snapshots] of Object.entries(priceSnapshots)) {
    if (snapshots.length < 2) continue;
    const startPrice = snapshots[0].price;
    const endPrice = snapshots[snapshots.length - 1].price;
    buyHoldResults.push(((endPrice - startPrice) / startPrice) * 100);
  }

  if (buyHoldResults.length > 0) {
    const avgReturn = buyHoldResults.reduce((a, b) => a + b, 0) / buyHoldResults.length;
    const totalReturn = buyHoldResults.reduce((a, b) => a + b, 0);
    const stdDev = Math.sqrt(buyHoldResults.reduce((s, r) => s + (r - avgReturn) ** 2, 0) / buyHoldResults.length) || 1;
    let maxDD = 0;
    for (const [, snapshots] of Object.entries(priceSnapshots)) {
      let peak = snapshots[0].price;
      for (const s of snapshots) {
        if (s.price > peak) peak = s.price;
        const dd = ((peak - s.price) / peak) * 100;
        if (dd > maxDD) maxDD = dd;
      }
    }
    baselines.push({
      strategy: "Buy & Hold (equal weight)",
      totalTrades: Object.keys(priceSnapshots).length,
      winRate: buyHoldResults.filter(r => r > 0).length / buyHoldResults.length,
      avgReturn,
      totalReturn,
      sharpeProxy: avgReturn / stdDev,
      maxDrawdown: maxDD,
    });
  }

  const rsiResults: { wins: number; total: number; returns: number[] } = { wins: 0, total: 0, returns: [] };
  for (const [, snapshots] of Object.entries(priceSnapshots)) {
    if (snapshots.length < 20) continue;
    for (let i = 14; i < snapshots.length - 5; i++) {
      const window = snapshots.slice(i - 14, i);
      let gains = 0, losses = 0;
      for (let j = 1; j < window.length; j++) {
        const change = window[j].price - window[j - 1].price;
        if (change > 0) gains += change;
        else losses += Math.abs(change);
      }
      const avgGain = gains / 14;
      const avgLoss = losses / 14 || 0.001; // quant-only-allow: divide-by-zero epsilon for RSI computation, not a fee
      const rs = avgGain / avgLoss;
      const rsi = 100 - 100 / (1 + rs);

      if (rsi < 30 || rsi > 70) {
        const direction = rsi < 30 ? "up" : "down";
        const exitIdx = Math.min(i + 5, snapshots.length - 1);
        const entryPrice = snapshots[i].price;
        const exitPrice = snapshots[exitIdx].price;
        const ret = direction === "up"
          ? ((exitPrice - entryPrice) / entryPrice) * 100 - 0.25
          : ((entryPrice - exitPrice) / entryPrice) * 100 - 0.25;
        rsiResults.total++;
        if (ret > 0) rsiResults.wins++;
        rsiResults.returns.push(ret);
      }
    }
  }

  if (rsiResults.total > 0) {
    const avgRet = rsiResults.returns.reduce((a, b) => a + b, 0) / rsiResults.returns.length;
    const stdDev = Math.sqrt(rsiResults.returns.reduce((s, r) => s + (r - avgRet) ** 2, 0) / rsiResults.returns.length) || 1;
    baselines.push({
      strategy: "RSI-only (30/70 threshold)",
      totalTrades: rsiResults.total,
      winRate: rsiResults.wins / rsiResults.total,
      avgReturn: avgRet,
      totalReturn: rsiResults.returns.reduce((a, b) => a + b, 0),
      sharpeProxy: avgRet / stdDev,
      maxDrawdown: Math.abs(Math.min(...rsiResults.returns, 0)),
    });
  }

  const randomResults: number[] = [];
  for (const [, snapshots] of Object.entries(priceSnapshots)) {
    if (snapshots.length < 10) continue;
    for (let trial = 0; trial < 20; trial++) {
      const entryIdx = Math.floor(Math.random() * (snapshots.length - 6));
      const exitIdx = entryIdx + 5;
      const direction = Math.random() > 0.5 ? "up" : "down";
      const entryPrice = snapshots[entryIdx].price;
      const exitPrice = snapshots[exitIdx].price;
      const ret = direction === "up"
        ? ((exitPrice - entryPrice) / entryPrice) * 100 - 0.25
        : ((entryPrice - exitPrice) / entryPrice) * 100 - 0.25;
      randomResults.push(ret);
    }
  }

  if (randomResults.length > 0) {
    const avgRet = randomResults.reduce((a, b) => a + b, 0) / randomResults.length;
    const stdDev = Math.sqrt(randomResults.reduce((s, r) => s + (r - avgRet) ** 2, 0) / randomResults.length) || 1;
    baselines.push({
      strategy: "Random (coin flip + fees)",
      totalTrades: randomResults.length,
      winRate: randomResults.filter(r => r > 0).length / randomResults.length,
      avgReturn: avgRet,
      totalReturn: randomResults.reduce((a, b) => a + b, 0),
      sharpeProxy: avgRet / stdDev,
      maxDrawdown: Math.abs(Math.min(...randomResults, 0)),
    });
  }

  const momentumResults: number[] = [];
  for (const [, snapshots] of Object.entries(priceSnapshots)) {
    if (snapshots.length < 15) continue;
    for (let i = 10; i < snapshots.length - 5; i++) {
      const lookback = snapshots.slice(i - 10, i);
      const momentum = (lookback[lookback.length - 1].price - lookback[0].price) / lookback[0].price;
      if (Math.abs(momentum) > 0.01) {
        const direction = momentum > 0 ? "up" : "down";
        const exitIdx = Math.min(i + 5, snapshots.length - 1);
        const entryPrice = snapshots[i].price;
        const exitPrice = snapshots[exitIdx].price;
        const ret = direction === "up"
          ? ((exitPrice - entryPrice) / entryPrice) * 100 - 0.25
          : ((entryPrice - exitPrice) / entryPrice) * 100 - 0.25;
        momentumResults.push(ret);
      }
    }
  }

  if (momentumResults.length > 0) {
    const avgRet = momentumResults.reduce((a, b) => a + b, 0) / momentumResults.length;
    const stdDev = Math.sqrt(momentumResults.reduce((s, r) => s + (r - avgRet) ** 2, 0) / momentumResults.length) || 1;
    baselines.push({
      strategy: "Simple momentum (10-period)",
      totalTrades: momentumResults.length,
      winRate: momentumResults.filter(r => r > 0).length / momentumResults.length,
      avgReturn: avgRet,
      totalReturn: momentumResults.reduce((a, b) => a + b, 0),
      sharpeProxy: avgRet / stdDev,
      maxDrawdown: Math.abs(Math.min(...momentumResults, 0)),
    });
  }

  return baselines;
}

export async function generateValidationReport(periodHours: number = 24): Promise<ValidationReport> {
  const since = new Date(Date.now() - periodHours * 60 * 60 * 1000);
  const warnings: string[] = [];

  const allPredictions = await db
    .select()
    .from(predictionsTable)
    .where(and(
      gte(predictionsTable.createdAt, since),
      ne(predictionsTable.outcome, "pending")
    ))
    .orderBy(desc(predictionsTable.createdAt));

  const correct = allPredictions.filter(p => p.outcome === "correct");
  const wrong = allPredictions.filter(p => p.outcome === "wrong");
  const neutral = allPredictions.filter(p => p.outcome === "neutral");
  const decidedCount = correct.length + wrong.length;
  const accuracy = decidedCount > 0 ? correct.length / decidedCount : 0;

  const avgConfidence = allPredictions.length > 0
    ? allPredictions.reduce((s, p) => s + p.confidence, 0) / allPredictions.length
    : 0;
  const avgConfCorrect = correct.length > 0
    ? correct.reduce((s, p) => s + p.confidence, 0) / correct.length
    : 0;
  const avgConfWrong = wrong.length > 0
    ? wrong.reduce((s, p) => s + p.confidence, 0) / wrong.length
    : 0;

  const calibrationError = Math.abs(avgConfidence - accuracy);

  const allTrades = await db
    .select()
    .from(paperTradesTable)
    .where(and(
      gte(paperTradesTable.createdAt, since),
      eq(paperTradesTable.status, "closed")
    ))
    .orderBy(desc(paperTradesTable.createdAt));

  const winningTrades = allTrades.filter(t => (t.pnl ?? 0) > 0);
  const losingTrades = allTrades.filter(t => (t.pnl ?? 0) <= 0);
  const totalPnl = allTrades.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const avgTradeReturn = allTrades.length > 0 ? totalPnl / allTrades.length : 0;
  const bestTrade = allTrades.length > 0 ? Math.max(...allTrades.map(t => t.pnl ?? 0)) : 0;
  const worstTrade = allTrades.length > 0 ? Math.min(...allTrades.map(t => t.pnl ?? 0)) : 0;
  const grossProfit = winningTrades.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const grossLoss = Math.abs(losingTrades.reduce((s, t) => s + (t.pnl ?? 0), 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;

  const baselines = await computeBaselines(periodHours);

  const systemVsBaselines = baselines.map(b => ({
    strategy: b.strategy,
    systemEdge: (accuracy * 100) - (b.winRate * 100),
    systemBetter: accuracy > b.winRate,
  }));

  const byTimeframe: Record<string, { predictions: number; accuracy: number; trades: number; winRate: number; avgReturn: number }> = {};
  const tfGroups = new Map<string, typeof allPredictions>();
  for (const p of allPredictions) {
    const tf = p.timeframe || "unknown";
    if (!tfGroups.has(tf)) tfGroups.set(tf, []);
    tfGroups.get(tf)!.push(p);
  }
  for (const [tf, preds] of tfGroups) {
    const c = preds.filter(p => p.outcome === "correct").length;
    const w = preds.filter(p => p.outcome === "wrong").length;
    const decided = c + w;
    const tfTrades = allTrades.filter(t => t.timeframe === tf);
    const tfWins = tfTrades.filter(t => (t.pnl ?? 0) > 0);
    const tfPnl = tfTrades.reduce((s, t) => s + (t.pnl ?? 0), 0);
    byTimeframe[tf] = {
      predictions: preds.length,
      accuracy: decided > 0 ? (c / decided) * 100 : 0,
      trades: tfTrades.length,
      winRate: tfTrades.length > 0 ? (tfWins.length / tfTrades.length) * 100 : 0,
      avgReturn: tfTrades.length > 0 ? tfPnl / tfTrades.length : 0,
    };
  }

  const agentGroups = new Map<string, typeof allPredictions>();
  for (const p of allPredictions) {
    if (!agentGroups.has(p.agentName)) agentGroups.set(p.agentName, []);
    agentGroups.get(p.agentName)!.push(p);
  }

  const byAgent: ValidationReport["byAgent"] = [];
  for (const [name, preds] of agentGroups) {
    const c = preds.filter(p => p.outcome === "correct").length;
    const w = preds.filter(p => p.outcome === "wrong").length;
    const decided = c + w;
    const agentTrades = allTrades.filter(t => t.agentName === name);
    const agentWins = agentTrades.filter(t => (t.pnl ?? 0) > 0);
    const agentPnl = agentTrades.reduce((s, t) => s + (t.pnl ?? 0), 0);

    const directionSet = new Set(preds.map(p => `${p.coinId}:${p.direction}`));
    const uniqueBehavior = directionSet.size / Math.max(preds.length, 1);

    byAgent.push({
      name,
      predictions: preds.length,
      accuracy: decided > 0 ? (c / decided) * 100 : 0,
      trades: agentTrades.length,
      winRate: agentTrades.length > 0 ? (agentWins.length / agentTrades.length) * 100 : 0,
      pnl: agentPnl,
      uniqueBehavior: Math.round(uniqueBehavior * 100),
    });
  }

  byAgent.sort((a, b) => b.pnl - a.pnl);

  const { getAblationConfig } = await import("./ablation-config");
  const ablation = getAblationConfig();
  const ablationImpact: Record<string, { enabled: boolean; description: string }> = {
    dualConsensus: { enabled: ablation.dualConsensus, description: "Multi-model consensus (GPT + Gemini)" },
    fingerprintMatching: { enabled: ablation.fingerprintMatching, description: "Historical pattern fingerprint lookup" },
    contagionDetection: { enabled: ablation.contagionDetection, description: "Cross-coin contagion alerts" },
    confidenceCalibration: { enabled: ablation.confidenceCalibration, description: "Confidence calibration adjustment" },
    regimeDetection: { enabled: ablation.regimeDetection, description: "Market regime classification" },
    agentSpecialization: { enabled: ablation.agentSpecialization, description: "Per-agent coin allocation optimization" },
  };

  if (decidedCount < 30) warnings.push(`Only ${decidedCount} decided predictions — insufficient for statistical significance (need 30+)`);
  if (allTrades.length < 20) warnings.push(`Only ${allTrades.length} closed trades — insufficient sample size for reliable metrics`);
  if (calibrationError > 0.15) warnings.push(`High calibration error (${(calibrationError * 100).toFixed(1)}%) — confidence scores don't match actual accuracy`);
  if (neutral.length > decidedCount * 0.5) warnings.push(`High neutral rate (${neutral.length}/${allPredictions.length}) — many predictions are expiring without clear outcomes`);
  if (profitFactor < 1) warnings.push(`Profit factor below 1.0 (${profitFactor.toFixed(2)}) — system is net losing money`);

  const lowDiffAgents = byAgent.filter(a => a.uniqueBehavior < 20);
  if (lowDiffAgents.length > 3) warnings.push(`${lowDiffAgents.length} agents have low behavioral differentiation (<20%) — may be acting as themed wrappers`);

  return {
    reportGeneratedAt: new Date().toISOString(),
    periodHours,
    systemPerformance: {
      totalPredictions: allPredictions.length,
      correctPredictions: correct.length,
      wrongPredictions: wrong.length,
      neutralPredictions: neutral.length,
      accuracy: accuracy * 100,
      avgConfidence: avgConfidence * 100,
      avgConfidenceWhenCorrect: avgConfCorrect * 100,
      avgConfidenceWhenWrong: avgConfWrong * 100,
      calibrationError: calibrationError * 100,
      totalTrades: allTrades.length,
      tradingWinRate: allTrades.length > 0 ? (winningTrades.length / allTrades.length) * 100 : 0,
      totalPnl,
      avgTradeReturn,
      bestTrade,
      worstTrade,
      profitFactor,
    },
    baselines,
    systemVsBaselines,
    byTimeframe,
    byAgent,
    ablationImpact,
    warnings,
  };
}
