import { Router, type IRouter } from "express";
import { eq, desc, and, gte, sql, inArray, isNull } from "drizzle-orm";
// Task #444 — `llmEventSignalsTable`, `llmDriftExplanationsTable`,
// `llmTradePostmortemsTable`, `llmOperatorAnswersTable` were dropped
// alongside the llm-sidecar directory; the four sidecar routes that
// consumed them are deleted lower down in this file.
import { db, agentsTable, predictionsTable, priceHistoryTable, monitoringStateTable, paperPortfoliosTable, paperTradesTable, paperPositionsTable, skipEventsTable, predictionJournalTable, tradeJournalTable, marketSignalsTable } from "@workspace/db";

// Task #468 — admin reset routes seed one row per registry-executor
// profile. Each row carries the canonical `profile_id` so the live
// trader's registry gate (paper-trader.ts) resolves directly through
// `getAgentProfile()` and the boot-time sweep is a no-op for these
// rows. The seed is timeframe-agnostic — the monitor's per-(coin,
// timeframe) loop fans out on its own.
const DETERMINISTIC_AGENTS: ReadonlyArray<{ name: string; personality: string; timeframe: string; profileId: string }> = [
  { name: "Momentum Core",        personality: "Momentum Core (registry executor)",        timeframe: "1h", profileId: "momentum_core" },
  { name: "Mean Reversion Core",  personality: "Mean Reversion Core (registry executor)",  timeframe: "1h", profileId: "mean_reversion_core" },
  { name: "Breakout Core",        personality: "Breakout Core (registry executor)",        timeframe: "1h", profileId: "breakout_core" },
  { name: "Volatility Defensive", personality: "Volatility Defensive (registry executor)", timeframe: "1h", profileId: "volatility_defensive" },
];
import { getMarketSignalsPollerStatus, getMarketSignalsPollerTargets } from "../../lib/market-signals-poller";
import {
  clearMarketSignalsWatcherSnooze,
  getMarketSignalsAlertChannels,
  getMarketSignalsWatcherHistory,
  getMarketSignalsWatcherSnooze,
  getMarketSignalsWatcherState,
  isSnoozeDuration,
  setMarketSignalsWatcherSnooze,
} from "../../lib/market-signals-poller-watcher";
import {
  ERROR_STREAK_THRESHOLD as TOPUP_5M_ERROR_STREAK_THRESHOLD,
  getTopup5mAlertChannels,
  getTopup5mNotifierState,
} from "../../lib/topup-5m-notifier";
import { getDisabledOutcomeBannerState } from "../../lib/disabled-outcome-notifier";
// Task #444 — llm-sidecar / askCopilot imports removed; the four
// `/crypto/llm/sidecar/*` and `/crypto/llm/copilot` routes below are
// deleted. Quant brain is the sole authority for trade decisions.
import {
  ListCoinsResponse,
  ListAgentsResponse,
  GetAgentParams,
  GetAgentResponse,
  ListPredictionsQueryParams,
  ListPredictionsResponse,
  GetDashboardResponse,
  TriggerAnalysisResponse,
  GetAgentPerformanceQueryParams,
  GetAgentPerformanceResponse,
  GetPriceHistoryParams,
  GetPriceHistoryQueryParams,
  GetPriceHistoryResponse,
  GetMonitoringStatusResponse,
  GetDiagnosticSandboxStatusResponse,
  GetDiagnosticSandboxHealthResponse,
  GetCoinDetailParams,
  GetCoinDetailQueryParams,
  GetCoinDetailResponse,
  TimeframeRolesResponse,
} from "@workspace/api-zod";
import {
  loadTimeframeRoles,
  summarizeTimeframeRoles,
} from "../../lib/timeframe-roles";
import { fetchCoinPrices, MONITORED_COINS } from "../../lib/coins";
import { TIMEFRAMES } from "../../lib/pattern-analyzer";
import { getMlModels } from "../../lib/ml-client";
import { getStrategyComparison } from "../../lib/strategy-lab";
import { getPortfolioSummaries, getAutoDeployedLast24hUsd, getAutoDeployAttribution, getAutoDeployAttributionHistory } from "../../lib/paper-trader";
import { getSectorForCoin, PORTFOLIO_CONSTRAINTS_CONFIG } from "../../lib/portfolio-constraints";
import { getSkipReasonsSummary, getSkipTimeline, getSkipsInBucket, getSkipPersistHealth, type SkipReason } from "../../lib/skip-tracker";
import {
  applyLoosen,
  applyTighten,
  computeSuggestion,
  getAutoApplyTightenStatus,
  getCachedSuggestion,
  getTuningState,
  revertChange,
  setAutoApplyTightenOverride,
  setCachedSuggestion,
  getPendingTighten,
  getAutoApplyTightenTicks,
  isAutoApplyTightenEnabled,
  type GateKey,
} from "../../lib/tuning-tracker";
import { runAnalysisCycle, getCycleCount, getLatestBestPick, getRetrainSchedulerState } from "../../lib/monitor";
import { getAgentCalibration } from "../../lib/confidence-calibrator";
import { getBrainState, setBrainState } from "../../lib/brain-flag";
import { computeBrainRuntimeState } from "../../lib/brain-runtime-state";
import { isMlAvailabilitySnapshotReady } from "../../lib/ml-availability";
import {
  hasPromotedSlice,
  getPromotionGateRetryStats,
  type PromotionGateRetryStats,
} from "../../lib/brain-promotion-gate";
import { createBrainStatePostHandler } from "../../lib/brain-state-route";
import { getMeasurementModeState, setMeasurementModeState } from "../../lib/measurement-mode";
import {
  getMttmConfig,
  setMttmEnabled,
  setMttmUniverse,
  clearMttmDisableReason,
  evaluateMttmAutoDisable,
  // Task #659 (C-BTC) — diagnostic sandbox lane.
  evaluateDiagnosticSandboxAutoDisable,
  evaluateDiagnosticSandboxDrift,
  getDiagnosticSandboxMetrics,
  getDiagnosticSandboxHealth,
  getDiagnosticSandboxLabel,
  isDiagnosticSandboxReady,
  setMttmMode,
  setDiagnosticSandboxBtcVersion,
  MTTM_DIAGNOSTIC_SANDBOX_COIN,
  MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
  MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
  type MttmSlot,
  type MttmMode,
} from "../../lib/mttm";
import { buildMttmReport, type MttmWindow } from "../../lib/mttm-report";
import { runMttmAudit } from "../../lib/mttm-audit";
import { getBrainRevertLog, getAutoRevertCounter } from "../../lib/brain-auto-revert";
// Task #444 — getRecentNewsTags removed; the LLM news classifier and its
// `news_tags` table are gone. The /crypto/brain/news-tags route below
// is deleted alongside this import.
import {
  getJournalHealth,
  getJournalHealthSeries,
  backfillJournals,
  backfillMissingQuantJournals,
  writeBacktestJournalRows,
  type InsertBacktestJournalArgs,
  type BacktestSimulatedTrade,
} from "../../lib/journal-writer";
// Task #444 — llm-bias-monitor + llm-bias-demote-tracker removed; the
// LLM brain no longer authors trade decisions, so the bias safety net
// has nothing to monitor.

const router: IRouter = Router();

function requireAdminApiKey(req: import("express").Request, res: import("express").Response): boolean {
  const adminKey = req.headers["x-admin-key"];
  const expected = process.env.ADMIN_API_KEY;
  if (!expected || !adminKey || adminKey !== expected) {
    res.status(403).json({ error: "Forbidden: invalid admin API key. This endpoint requires the ADMIN_API_KEY header." });
    return false;
  }
  return true;
}

router.get("/crypto/coins", async (req, res): Promise<void> => {
  const prices = await fetchCoinPrices();
  res.json(ListCoinsResponse.parse(prices));
});

const SKIP_REASON_LABELS: Record<string, string> = {
  confidence_below_threshold: "Confidence below 0.40",
  counter_trend_regime: "Counter-trend in confirmed regime",
  daily_loss_limit: "Daily loss limit hit",
  drawdown_halt: "Max drawdown halt",
  risk_recheck_halt: "Risk re-check halt",
  consecutive_losses: "3 consecutive losses on coin",
  fee_gate_tp_floor: "TP distance below fee floor",
  fee_gate_ev: "Expected value below fee threshold",
  both_models_down: "Both AI models unavailable",
  no_trade_zone: "No-trade zone (model disagreement)",
  single_model_penalty: "Single-model penalty (no consensus)",
};

router.get("/crypto/coins/:coinId", async (req, res): Promise<void> => {
  const params = GetCoinDetailParams.safeParse(req.params);
  if (!params.success) {
    res.status(400).json({ error: params.error.message });
    return;
  }
  const queryParams = GetCoinDetailQueryParams.safeParse(req.query);
  const hours = queryParams.success ? queryParams.data.hours ?? 24 : 24;

  const allCoins = await fetchCoinPrices();
  const coin = allCoins.find((c) => c.id === params.data.coinId);
  if (!coin) {
    res.status(404).json({ error: "Coin not found" });
    return;
  }

  const since = new Date(Date.now() - hours * 60 * 60 * 1000);

  const recentPredictionsRaw = await db
    .select()
    .from(predictionsTable)
    .where(and(
      eq(predictionsTable.coinId, coin.id),
      gte(predictionsTable.createdAt, since),
    ))
    .orderBy(desc(predictionsTable.createdAt))
    .limit(50);

  const recentPredictions = recentPredictionsRaw.map((p) => ({
    ...p,
    timeframe: p.timeframe || "1m",
    resolvesAt: p.resolvesAt?.toISOString() ?? null,
    createdAt: p.createdAt.toISOString(),
    resolvedAt: p.resolvedAt?.toISOString() ?? null,
  }));

  let up = 0, down = 0, stable = 0;
  let confSum = 0, confCount = 0;
  for (const p of recentPredictionsRaw) {
    if (p.direction === "up") up++;
    else if (p.direction === "down") down++;
    else stable++;
    if (typeof p.confidence === "number") {
      confSum += p.confidence;
      confCount++;
    }
  }
  const total = up + down + stable;
  let dominant: "up" | "down" | "stable" | "none" = "none";
  if (total > 0) {
    if (up >= down && up >= stable) dominant = "up";
    else if (down >= up && down >= stable) dominant = "down";
    else dominant = "stable";
  }
  const agentAgreement = {
    up,
    down,
    stable,
    total,
    dominant,
    avgConfidence: confCount > 0 ? confSum / confCount : 0,
  };

  const skipRows = await db
    .select()
    .from(skipEventsTable)
    .where(gte(skipEventsTable.ts, since))
    .orderBy(desc(skipEventsTable.ts));

  const matchKeys = [coin.id.toLowerCase(), coin.symbol.toLowerCase(), coin.name.toLowerCase()];
  const recentSkipEvents = skipRows
    .filter((row) => {
      const details = (row.details ?? {}) as Record<string, unknown>;
      for (const k of ["coin", "symbol", "asset", "ticker", "coinId"]) {
        const v = details[k];
        if (typeof v === "string" && matchKeys.includes(v.toLowerCase())) return true;
      }
      return false;
    })
    .slice(0, 100)
    .map((row) => ({
      ts: row.ts.toISOString(),
      reason: row.reason,
      reasonLabel: SKIP_REASON_LABELS[row.reason] ?? row.reason,
      agentName: row.agentName,
      message: row.message,
    }));

  res.json(GetCoinDetailResponse.parse({
    coin,
    recentPredictions,
    agentAgreement,
    recentSkipEvents,
    windowHours: hours,
  }));
});

router.get("/crypto/agents", async (req, res): Promise<void> => {
  const agents = await db
    .select()
    .from(agentsTable)
    .where(eq(agentsTable.isActive, true))
    .orderBy(desc(agentsTable.score));

  // Task #468 — surface the resolved strategy-profile inline on the
  // existing /crypto/agents response. The spec explicitly forbids
  // adding new endpoints, so the dashboard enriches each row with:
  //   • `registryProfile` — the boot-loaded AgentProfile (null only
  //     when the legacy-name sweep could not resolve, in which case
  //     the dashboard renders a "needs review" badge);
  //   • `profileSubId` — the Strategy-Lab variant tag retained per
  //     spec lines 39-41 (e.g. "baseline_buy_hold"), null for any
  //     non-baseline-variant agent;
  //   • `retirementMetric` — live retirement-rule values (DA,
  //     Sharpe, activation count) sourced from the latest nightly
  //     snapshot so the dashboard surfaces per-agent metric values
  //     alongside `status` per the dashboard step.
  // Profile + snapshot data are read from in-memory caches — no
  // per-request DB lookup.
  const { tryGetCachedEntry, mapLegacyNameToSubId, getRetirementSnapshot } =
    await import("../../lib/agents-registry");
  const snapshot = getRetirementSnapshot();
  // Index the snapshot's candidates by every numeric agent id they
  // affect. A single candidate row (one per profile) can cover many
  // agents in the DB sharing that `profile_id`, so we replicate the
  // candidate per affected agent for O(1) lookup below.
  const candidateByAgentId = new Map<
    number,
    NonNullable<typeof snapshot>["candidates"][number]
  >();
  if (snapshot) {
    for (const c of snapshot.candidates) {
      for (const aid of c.affected_agent_ids) candidateByAgentId.set(aid, c);
    }
  }
  const result = agents.map((a) => {
    const candidate = candidateByAgentId.get(a.id);
    const retirementMetric = candidate
      ? {
          ruleKind: candidate.rule_kind,
          closedTrades: candidate.closed_trades,
          directionalAccuracy: candidate.directional_accuracy,
          costAwareSharpe: candidate.cost_aware_sharpe,
          daThreshold: candidate.da_threshold,
          sharpeThreshold: candidate.sharpe_threshold,
          activationThreshold: candidate.activation_threshold,
          triggeredBy: candidate.triggered_by,
          autoFlipped: candidate.auto_flipped,
          evaluatedAt: candidate.evaluated_at,
        }
      : null;
    return {
      ...a,
      accuracy:
        a.totalPredictions > 0
          ? (a.correctPredictions / a.totalPredictions) * 100
          : 0,
      registryProfile: tryGetCachedEntry(a.id)?.profile ?? null,
      profileSubId: mapLegacyNameToSubId(a.name),
      retirementMetric,
    };
  });

  res.json(ListAgentsResponse.parse(result));
});

// Task #512 — non-numeric `:id` values fall through to the next route
// so the more-specific sibling paths registered later in this file —
// `/agents/families`, `/agents/families/:profileId/coins`, and
// `/agents/archived` — are not shadowed by this parameterised handler.
// path-to-regexp v8 does not support inline regex constraints
// (e.g. `:id(\\d+)`), so we use a runtime `next()` guard instead.
router.get("/crypto/agents/:id", async (req, res, next): Promise<void> => {
  if (!/^\d+$/.test(String(req.params.id ?? ""))) {
    return next();
  }
  const params = GetAgentParams.safeParse(req.params);
  if (!params.success) {
    res.status(400).json({ error: params.error.message });
    return;
  }

  const [agent] = await db
    .select()
    .from(agentsTable)
    .where(eq(agentsTable.id, params.data.id));

  if (!agent) {
    res.status(404).json({ error: "Agent not found" });
    return;
  }

  const recentPredictions = await db
    .select()
    .from(predictionsTable)
    .where(eq(predictionsTable.agentId, agent.id))
    .orderBy(desc(predictionsTable.createdAt))
    .limit(20);

  const allPredictions = await db
    .select()
    .from(predictionsTable)
    .where(and(
      eq(predictionsTable.agentId, agent.id),
      sql`${predictionsTable.outcome} != 'pending'`
    ))
    .orderBy(predictionsTable.createdAt);

  let runningCorrect = 0;
  let runningTotal = 0;
  let runningScore = 100;
  const accuracyHistory = allPredictions.map((p) => {
    if (p.outcome === "correct" || p.outcome === "wrong") {
      runningTotal++;
      if (p.outcome === "correct") runningCorrect++;
    }
    runningScore += p.scoreChange || 0;
    return {
      timestamp: p.createdAt.toISOString(),
      accuracy: runningTotal > 0 ? (runningCorrect / runningTotal) * 100 : 0,
      cumulativeScore: runningScore,
    };
  });

  // Pull this agent's paper portfolio summary (if any) so the UI can show
  // cash, P&L, open positions, and recent trades alongside the agent's logic.
  const summaries = await getPortfolioSummaries();
  const paperPortfolio = summaries.find((p) => p.agentId === agent.id) ?? null;

  const result = {
    agent: {
      ...agent,
      accuracy: agent.totalPredictions > 0 ? (agent.correctPredictions / agent.totalPredictions) * 100 : 0,
    },
    recentPredictions: recentPredictions.map((p) => ({
      ...p,
      timeframe: p.timeframe || "1m",
      resolvesAt: p.resolvesAt?.toISOString() ?? null,
      createdAt: p.createdAt.toISOString(),
      resolvedAt: p.resolvedAt?.toISOString() ?? null,
    })),
    accuracyHistory,
    paperPortfolio: paperPortfolio
      ? {
          cashBalance: paperPortfolio.cashBalance,
          totalValue: paperPortfolio.totalValue,
          startingCapital: paperPortfolio.startingCapital,
          totalTrades: paperPortfolio.totalTrades,
          winningTrades: paperPortfolio.winningTrades,
          losingTrades: paperPortfolio.losingTrades,
          winRate: paperPortfolio.winRate,
          openPositions: paperPortfolio.openPositions,
          recentTrades: paperPortfolio.recentTrades,
        }
      : null,
  };

  res.json(GetAgentResponse.parse(result));
});

router.get("/crypto/predictions", async (req, res): Promise<void> => {
  const queryParams = ListPredictionsQueryParams.safeParse(req.query);
  const limit = queryParams.success ? queryParams.data.limit ?? 50 : 50;
  const agentId = queryParams.success ? queryParams.data.agentId : undefined;
  const coinId = queryParams.success ? queryParams.data.coinId : undefined;

  let query = db
    .select()
    .from(predictionsTable)
    .orderBy(desc(predictionsTable.createdAt))
    .limit(limit);

  const conditions = [];
  if (agentId) conditions.push(eq(predictionsTable.agentId, agentId));
  if (coinId) conditions.push(eq(predictionsTable.coinId, coinId));

  let predictions;
  if (conditions.length > 0) {
    predictions = await db
      .select()
      .from(predictionsTable)
      .where(and(...conditions))
      .orderBy(desc(predictionsTable.createdAt))
      .limit(limit);
  } else {
    predictions = await query;
  }

  const result = predictions.map((p) => ({
    ...p,
    timeframe: p.timeframe || "1m",
    resolvesAt: p.resolvesAt?.toISOString() ?? null,
    createdAt: p.createdAt.toISOString(),
    resolvedAt: p.resolvedAt?.toISOString() ?? null,
  }));

  res.json(ListPredictionsResponse.parse(result));
});

router.get("/crypto/dashboard", async (req, res): Promise<void> => {
  const agents = await db
    .select()
    .from(agentsTable)
    .where(eq(agentsTable.isActive, true))
    .orderBy(desc(agentsTable.score));

  const calibrations = await Promise.all(
    agents.map(a => getAgentCalibration(a.id).catch(() => null))
  );
  const calibrationMap = new Map(agents.map((a, i) => [a.id, calibrations[i]]));

  const agentsWithAccuracy = agents.map((a) => ({
    ...a,
    accuracy: a.totalPredictions > 0 ? (a.correctPredictions / a.totalPredictions) * 100 : 0,
    calibrationQuality: calibrationMap.get(a.id)?.calibrationQuality ?? undefined,
    confidenceBias: calibrationMap.get(a.id)?.overallBias ?? undefined,
    generation: a.generation ?? 1,
    evolutionMethod: a.evolutionMethod ?? "original",
    parentIds: a.parentIds ?? null,
    isActive: a.isActive ?? true,
  }));

  const resolvedPredictions = await db
    .select()
    .from(predictionsTable)
    .where(sql`${predictionsTable.outcome} != 'pending'`)
    .orderBy(desc(predictionsTable.resolvedAt))
    .limit(6);

  const pendingPredictions = await db
    .select()
    .from(predictionsTable)
    .where(eq(predictionsTable.outcome, "pending"))
    .orderBy(desc(predictionsTable.createdAt))
    .limit(4);

  const recentPredictions = [...pendingPredictions, ...resolvedPredictions]
    .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
    .slice(0, 10);

  const correctCount = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(predictionsTable)
    .where(eq(predictionsTable.outcome, "correct"));

  const wrongCount = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(predictionsTable)
    .where(eq(predictionsTable.outcome, "wrong"));

  const neutralCount = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(predictionsTable)
    .where(eq(predictionsTable.outcome, "neutral"));

  const pendingCount = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(predictionsTable)
    .where(eq(predictionsTable.outcome, "pending"));

  const totalResolved = (correctCount[0]?.count || 0) + (wrongCount[0]?.count || 0);
  const totalAll = totalResolved + (pendingCount[0]?.count || 0) + (neutralCount[0]?.count || 0);

  // Task #444 — the all-time `overallAccuracy` field was removed from
  // the dashboard payload. It folded together every resolved prediction
  // since launch, including months of legacy personality/LLM-era picks
  // and `quant_disabled` stable rows that resolve as trivially-correct,
  // producing a misleading ~70% headline that the dashboard rendered as
  // a quant-brain win-rate. The 24h window is the only honest reading
  // and is what the card now shows; all-time totals stay available on
  // the per-agent detail pages where their scope is explicit.
  //
  // Task #405 / B-AVG-HIDES — original audit context:
  // hides current model degradation; the audit found 24h directional
  // accuracy of 19.5 % was being masked by an all-time 56.5 %.
  const dir24hRow = await db.execute(sql`
    SELECT
      COUNT(*)::int AS resolved,
      SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END)::int AS correct
    FROM ${predictionsTable}
    WHERE outcome IN ('correct','wrong')
      AND direction IN ('up','down')
      AND resolved_at >= NOW() - INTERVAL '24 hours'
  `);
  const dir24h = (dir24hRow as unknown as { rows?: Array<{ resolved: number; correct: number }> }).rows?.[0]
    ?? { resolved: 0, correct: 0 };
  const directional24h = {
    resolved: Number(dir24h.resolved ?? 0),
    correct: Number(dir24h.correct ?? 0),
    accuracyPct: Number(dir24h.resolved ?? 0) > 0
      ? (Number(dir24h.correct ?? 0) / Number(dir24h.resolved)) * 100
      : null as number | null,
    coinFlipFloorPct: 45,
  } as const;

  const monState = await db.select().from(monitoringStateTable).limit(1);

  // Task #444 — `agent-evolution` (LLM-driven personality mutation) is
  // gone. The dashboard payload kept the field for back-compat and just
  // reports the static "no evolution" stub; the dashboard hook accepts
  // it and renders an empty cycle history.
  const evolutionStatus: Record<string, unknown> = {
    generation: 1,
    lastEvolution: null,
    currentFitness: [],
    history: [],
    agentLineage: {},
  };

  // Task #405 / B-AVG-HIDES — demote any "best agent" candidate whose 24h
  // directional accuracy is below the coin-flip floor of 45 % when the
  // sample size is non-trivial (>= 8 directional outcomes). This stops
  // the headline from celebrating an all-time 83 % winRate while the same
  // agent's recent directional accuracy is at 19 %.
  //
  // Task #532 / C-5 — exclude `quant_disabled` abstain rows from the
  // best/worst computation. While the brain is offline every prediction
  // emits `direction=stable` with `reasoning='quant_disabled'` and is
  // trivially "correct" (predicted stable, realized inside the neutral
  // band → wins). The agents.score / lifetime accuracy fields are
  // dominated by these rows, so the dashboard reports an 88 % win-streak
  // for an agent that has authored zero real trades. We recompute the
  // ranking from the predictions table with abstain rows excluded and
  // require ≥30 non-abstain resolved samples; if that bar is not met
  // the dashboard renders an "insufficient signal" placeholder instead
  // of a fake leaderboard.
  const MIN_NON_ABSTAIN_SAMPLE_FOR_RANKING = 30;
  const COIN_FLIP_FLOOR_PCT = 45;
  const MIN_DIRECTIONAL_SAMPLE = 8;
  let nonAbstainPerAgent = new Map<number, { resolved: number; correct: number }>();
  try {
    const ranking = await db.execute(sql`
      SELECT
        agent_id::int                                                  AS agent_id,
        COUNT(*)::int                                                  AS resolved,
        SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END)::int        AS correct
      FROM ${predictionsTable}
      WHERE outcome IN ('correct','wrong')
        AND direction IN ('up','down')
        AND resolved_at >= NOW() - INTERVAL '7 days'
        AND COALESCE(reasoning,'') NOT ILIKE '%quant_disabled%'
        AND COALESCE(reasoning,'') NOT ILIKE '%quant_abstain_%'
      GROUP BY agent_id
    `);
    for (const r of (ranking as unknown as { rows?: Array<{ agent_id: number; resolved: number; correct: number }> }).rows ?? []) {
      nonAbstainPerAgent.set(Number(r.agent_id), { resolved: Number(r.resolved), correct: Number(r.correct) });
    }
  } catch {
    nonAbstainPerAgent = new Map();
  }
  const candidatesWithRealSignal = agentsWithAccuracy.filter((a) => {
    const s = nonAbstainPerAgent.get(a.id);
    return s !== undefined && s.resolved >= MIN_NON_ABSTAIN_SAMPLE_FOR_RANKING;
  }).map((a) => {
    const s = nonAbstainPerAgent.get(a.id)!;
    return { ...a, accuracy: (s.correct / s.resolved) * 100 };
  });

  let bestAgent: typeof agentsWithAccuracy[number] | null = null;
  let worstAgent: typeof agentsWithAccuracy[number] | null = null;
  let dashboardSignal: "ok" | "insufficient_signal" = "insufficient_signal";
  if (candidatesWithRealSignal.length > 0) {
    dashboardSignal = "ok";
    const sorted = [...candidatesWithRealSignal].sort((a, b) => b.accuracy - a.accuracy);
    // Apply legacy 24h coin-flip floor demotion using all-time table only
    // when we have real-signal candidates above the bar.
    const candidates = await db.execute(sql`
      SELECT
        agent_id::int                                                          AS agent_id,
        COUNT(*)::int                                                          AS resolved,
        SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END)::int                AS correct
      FROM ${predictionsTable}
      WHERE outcome IN ('correct','wrong')
        AND direction IN ('up','down')
        AND resolved_at >= NOW() - INTERVAL '24 hours'
        AND COALESCE(reasoning,'') NOT ILIKE '%quant_disabled%'
        AND COALESCE(reasoning,'') NOT ILIKE '%quant_abstain_%'
      GROUP BY agent_id
    `);
    const perAgent = new Map<number, { resolved: number; correct: number }>();
    for (const r of (candidates as unknown as { rows?: Array<{ agent_id: number; resolved: number; correct: number }> }).rows ?? []) {
      perAgent.set(Number(r.agent_id), { resolved: Number(r.resolved), correct: Number(r.correct) });
    }
    const passes = (a: { id: number }) => {
      const s = perAgent.get(a.id);
      if (!s || s.resolved < MIN_DIRECTIONAL_SAMPLE) return true;
      return (s.correct / s.resolved) * 100 >= COIN_FLIP_FLOOR_PCT;
    };
    bestAgent = sorted.find(passes) ?? sorted[0] ?? null;
    worstAgent = sorted[sorted.length - 1] ?? null;
  }

  // Task #532 / Rev 2.2 — `agentsWithAccuracy` carries the agents-row
  // `streak`, `streakType`, and `score` fields, which were inflated by
  // the abstain-era predictions (every quant_disabled row scored as a
  // "correct stable" call). The leaderboard re-ranks on the
  // non-abstain accuracy, but the streak/score badges would still
  // surface the contaminated values. Null them out on the leaderboard
  // payloads so the dashboard cannot render a fake "12-day streak"
  // alongside the recomputed accuracy. The full agentRankings list
  // still preserves the raw fields for the per-agent detail page,
  // which is honest about its all-time scope.
  const cleanRankingPayload = <T extends { streak?: unknown; streakType?: unknown; score?: unknown }>(
    a: T | null,
  ): (T & { streak: null; streakType: null; score: null }) | null => {
    if (!a) return null;
    return { ...a, streak: null, streakType: null, score: null };
  };

  const dashboard = {
    totalPredictions: totalAll,
    directional24h,
    // Task #532 — when there is no real signal we return `null` and
    // surface `signal: "insufficient_signal"` so the dashboard can
    // hide the leaderboard instead of rendering an "N/A / 0 % / 0
    // streak" placeholder agent that operators read as a real agent
    // ranked first on the board. Code review correctly flagged that
    // the previous N/A placeholder was the same lie the rest of the
    // audit was repairing.
    bestAgent: cleanRankingPayload(bestAgent),
    worstAgent: cleanRankingPayload(worstAgent),
    signal: dashboardSignal,
    recentPredictions: recentPredictions.map((p) => ({
      ...p,
      timeframe: p.timeframe || "1m",
      resolvesAt: p.resolvesAt?.toISOString() ?? null,
      createdAt: p.createdAt.toISOString(),
      resolvedAt: p.resolvedAt?.toISOString() ?? null,
    })),
    agentRankings: agentsWithAccuracy,
    predictionsByOutcome: {
      correct: correctCount[0]?.count || 0,
      wrong: wrongCount[0]?.count || 0,
      pending: pendingCount[0]?.count || 0,
    },
    activeCoinCount: 10,
    monitoringCycles: monState[0]?.cycleCount || getCycleCount(),
    evolution: evolutionStatus,
  };

  res.json(GetDashboardResponse.parse(dashboard));
});

router.post("/crypto/trigger-analysis", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const result = await runAnalysisCycle();
  res.json(TriggerAnalysisResponse.parse({
    success: true,
    message: `Analysis cycle #${result.cycleNumber} completed with ${result.predictionsCreated} predictions`,
    predictionsCreated: result.predictionsCreated,
    cycleNumber: result.cycleNumber,
  }));
});

router.get("/crypto/agent-performance", async (req, res): Promise<void> => {
  const queryParams = GetAgentPerformanceQueryParams.safeParse(req.query);
  const hours = queryParams.success ? queryParams.data.hours ?? 24 : 24;

  const since = new Date(Date.now() - hours * 60 * 60 * 1000);

  const predictions = await db
    .select()
    .from(predictionsTable)
    .where(and(
      gte(predictionsTable.createdAt, since),
      sql`${predictionsTable.outcome} != 'pending'`
    ))
    .orderBy(predictionsTable.createdAt);

  const agentStats = new Map<number, { name: string; correct: number; total: number; score: number }>();

  const performancePoints = predictions.map((p) => {
    if (!agentStats.has(p.agentId)) {
      agentStats.set(p.agentId, { name: p.agentName, correct: 0, total: 0, score: 100 });
    }
    const stats = agentStats.get(p.agentId)!;
    if (p.outcome === "correct" || p.outcome === "wrong") {
      stats.total++;
      if (p.outcome === "correct") stats.correct++;
    }
    stats.score += p.scoreChange || 0;

    return {
      timestamp: p.createdAt.toISOString(),
      agentId: p.agentId,
      agentName: p.agentName,
      accuracy: stats.total > 0 ? (stats.correct / stats.total) * 100 : 0,
      score: stats.score,
    };
  });

  res.json(GetAgentPerformanceResponse.parse(performancePoints));
});

router.get("/crypto/price-history/:coinId", async (req, res): Promise<void> => {
  const params = GetPriceHistoryParams.safeParse(req.params);
  if (!params.success) {
    res.status(400).json({ error: params.error.message });
    return;
  }

  const queryParams = GetPriceHistoryQueryParams.safeParse(req.query);
  const hours = queryParams.success ? queryParams.data.hours ?? 1 : 1;

  const since = new Date(Date.now() - hours * 60 * 60 * 1000);

  const history = await db
    .select()
    .from(priceHistoryTable)
    .where(and(
      eq(priceHistoryTable.coinId, params.data.coinId),
      gte(priceHistoryTable.timestamp, since)
    ))
    .orderBy(priceHistoryTable.timestamp);

  const result = history.map((h) => ({
    timestamp: h.timestamp.toISOString(),
    price: h.price,
    coinId: h.coinId,
  }));

  res.json(GetPriceHistoryResponse.parse(result));
});

router.get("/crypto/measurement-mode", async (_req, res): Promise<void> => {
  // Reads the persisted flag (or env-var fallback) so the UI banner mirrors
  // the same value the monitor cycle uses. The full state object includes
  // the source ("default" | "env" | "manual") so the UI can show whether
  // the current value came from the operator toggle or the legacy env var.
  try {
    const state = await getMeasurementModeState();
    res.json({
      enabled: state.enabled,
      source: state.source,
      lastChangedAt: state.lastChangedAt,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// ── Task #550 ─ per-timeframe role registry (read-only) ──
// Returns the contents of `shared/timeframe-roles.json` plus a small
// counts summary so the dashboard can render badges next to each
// timeframe (`trade` / `shadow` / `context` / `disabled`). No admin
// key required — this is the same trust level as `/measurement-mode`
// (operational visibility, not control). Fail-closed: if the loader
// throws (missing/malformed JSON) every TF will already have been
// coerced to `disabled` (by_safety) inside `loadTimeframeRoles`, so
// this route always returns a well-formed payload.
router.get("/crypto/timeframe-roles", (_req, res): void => {
  try {
    const doc = loadTimeframeRoles();
    const summary = summarizeTimeframeRoles(doc);
    res.json(TimeframeRolesResponse.parse({ document: doc, summary }));
  } catch (err) {
    res
      .status(500)
      .json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

router.post("/crypto/measurement-mode/toggle", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const enabled = (req.body ?? {}).enabled;
  if (typeof enabled !== "boolean") {
    res.status(400).json({ error: "Body must be { enabled: boolean }" });
    return;
  }
  try {
    const state = await setMeasurementModeState(enabled);
    res.json({
      enabled: state.enabled,
      source: state.source,
      lastChangedAt: state.lastChangedAt,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// ── Task #614 — Minimum Truthful Trading Mode (MTTM) ──────────────────
//
// Three read endpoints + two admin write endpoints:
//   GET  /mttm/state               → current config (enabled, universe, caps,
//                                    disable reason). Public.
//   GET  /mttm/report?window=24h|48h|72h → 10-field success-criteria report
//                                    + verdict. Public.
//   POST /mttm/state/toggle        → flip the enable flag. Body { enabled,
//                                    clearDisableReason? }. Admin only —
//                                    clearing a tripped reason requires the
//                                    explicit `clearDisableReason: true` so
//                                    auto-disable cannot be silently bypassed.
//   POST /mttm/state/universe      → replace the pinned 16-slot list. Body
//                                    { universe: MttmSlot[] }. Admin only.
//   POST /mttm/state/evaluate      → manually re-run the auto-disable rule.
//                                    Returns the disable reason if tripped,
//                                    else null. Admin only.

router.get("/mttm/state", async (_req, res): Promise<void> => {
  try {
    const cfg = await getMttmConfig();
    res.json({
      enabled: cfg.enabled,
      enabledAt: cfg.enabledAt,
      universe: cfg.universe,
      universeSize: cfg.universe.length,
      maxPositionPct: cfg.maxPositionPct,
      consecutiveLossCap: cfg.consecutiveLossCap,
      n10PostFeeCapPct: cfg.n10PostFeeCapPct,
      disableReason: cfg.disableReason,
      autoDisabled: !!cfg.disableReason,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

router.get("/mttm/report", async (req, res): Promise<void> => {
  const raw = typeof req.query.window === "string" ? req.query.window : "72h";
  const allowed: MttmWindow[] = ["24h", "48h", "72h"];
  if (!allowed.includes(raw as MttmWindow)) {
    res.status(400).json({ error: `window must be one of ${allowed.join(", ")}` });
    return;
  }
  try {
    const report = await buildMttmReport(raw as MttmWindow);
    res.json(report);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

router.post("/mttm/state/toggle", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const body = req.body ?? {};
  if (typeof body.enabled !== "boolean") {
    res.status(400).json({ error: "Body must be { enabled: boolean }" });
    return;
  }
  try {
    if (body.enabled) {
      // Re-enabling after a trip requires explicit acknowledgement.
      const current = await getMttmConfig();
      if (current.disableReason && body.clearDisableReason !== true) {
        res.status(409).json({
          error:
            "MTTM auto-disable still active — re-send with clearDisableReason: true to acknowledge",
          disableReason: current.disableReason,
        });
        return;
      }
      // Hard precondition: every slot in the configured universe must
      // pass `mttm-audit` before MTTM can be turned on. This closes
      // the loophole where the toggle would happily flip to true even
      // though the underlying registry still had unpromoted or
      // missing slots — the spec's "Done looks like" gate.
      const audit = await runMttmAudit();
      if (!audit.ok) {
        res.status(409).json({
          error:
            "MTTM cannot enable — audit failed. Every slot in mttm_universe " +
            "must be a promoted lightgbm model with `latest` matching the " +
            "pinned version. Run `pnpm --filter @workspace/api-server exec " +
            "tsx src/scripts/mttm-audit.ts` for the full table.",
          auditSource: audit.source,
          modelsRoot: audit.modelsRoot,
          failingSlots: audit.failingSlots.map((r) => ({
            coinId: r.coinId,
            timeframe: r.timeframe,
            expectedVersion: r.expectedVersion,
            latestVersion: r.latestVersion,
            servedPredictorKind: r.servedPredictorKind,
            promoted: r.promoted,
            problems: r.problems,
          })),
        });
        return;
      }
    }
    const cfg = await setMttmEnabled(body.enabled, {
      clearDisableReason: body.clearDisableReason === true,
    });
    res.json({
      enabled: cfg.enabled,
      enabledAt: cfg.enabledAt,
      disableReason: cfg.disableReason,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

router.post("/mttm/state/universe", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const body = req.body ?? {};
  if (!Array.isArray(body.universe)) {
    res.status(400).json({ error: "Body must be { universe: MttmSlot[] }" });
    return;
  }
  const universe: MttmSlot[] = [];
  for (const e of body.universe) {
    if (
      e &&
      typeof e === "object" &&
      typeof e.coinId === "string" &&
      typeof e.timeframe === "string" &&
      typeof e.version === "string"
    ) {
      universe.push({ coinId: e.coinId, timeframe: e.timeframe, version: e.version });
    } else {
      res.status(400).json({
        error:
          "Each universe entry must be { coinId: string, timeframe: string, version: string }",
      });
      return;
    }
  }
  try {
    const cfg = await setMttmUniverse(universe);
    res.json({ universe: cfg.universe, universeSize: cfg.universe.length });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

router.post("/mttm/state/clear-disable", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    await clearMttmDisableReason();
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

router.post("/mttm/state/evaluate", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const reason = await evaluateMttmAutoDisable();
    res.json({ tripped: !!reason, reason });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #659 — diagnostic-sandbox API. One public read + three admin
// writes (mode flip, btc-version stamp, evaluator re-run). Status
// shape is governed by the OpenAPI spec (DiagnosticSandboxStatus).

router.get("/diagnostic-sandbox/status", async (_req, res): Promise<void> => {
  try {
    const cfg = await getMttmConfig();
    const metrics = await getDiagnosticSandboxMetrics(cfg);
    const since = cfg.enabledAt;
    const sinceMs = since ? Date.parse(since) : NaN;
    const hoursSince = Number.isFinite(sinceMs)
      ? (Date.now() - sinceMs) / 3_600_000
      : null;
    const payload = {
      label: getDiagnosticSandboxLabel(),
      mode: cfg.mode,
      enabled: cfg.enabled && cfg.mode === "diagnostic_sandbox",
      ready: isDiagnosticSandboxReady(cfg),
      universe: cfg.universe.map((s) => ({
        coin_id: s.coinId,
        timeframe: s.timeframe,
        version: s.version ?? null,
      })),
      fixed_position_pct: MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
      btc_version: cfg.diagnosticSandbox.btcVersion,
      meta_shadow: cfg.mode === "diagnostic_sandbox",
      drawdown_floor_pct: cfg.diagnosticSandbox.drawdownPct,
      n_neg_pnl_threshold: cfg.diagnosticSandbox.nNegPnl,
      since,
      hours_since: hoursSince,
      closed_trades_since: metrics?.nTrades ?? 0,
      current_drawdown_pct: metrics?.drawdownPct ?? 0,
      net_pnl_pct: metrics?.cumulativePnlPct ?? 0,
      reviews_remaining:
        metrics?.reviewsRemaining ?? cfg.diagnosticSandbox.nNegPnl,
      auto_disable_status: {
        // Snake_case external contract: `disabled` + `disabled_at`
        // (the internal MttmDisableReason still carries `trippedAt`,
        // which we map onto `disabled_at` here at the API boundary).
        disabled: !!cfg.disableReason,
        reason: cfg.disableReason?.reason ?? null,
        detail: cfg.disableReason?.detail ?? null,
        disabled_at: cfg.disableReason?.trippedAt ?? null,
      },
    };
    res.json(GetDiagnosticSandboxStatusResponse.parse(payload));
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : "unknown",
    });
  }
});

// Task #670 — DS health probe. Cheap trailing-window drawdown read so
// operators have an early-warning surface before the auto-disable
// evaluator's full-since-enable drawdown trips and shuts the lane off.
// Public read (same as /status); the helper does its own 30s in-process
// caching so a noisy dashboard poll doesn't translate to a paper_trades
// scan per request.
router.get("/diagnostic-sandbox/health", async (_req, res): Promise<void> => {
  try {
    const health = await getDiagnosticSandboxHealth();
    const payload = {
      evaluable: health.evaluable,
      label: health.label,
      coin_id: health.coinId,
      timeframe: health.timeframe,
      btc_version: health.btcVersion,
      window_trades: health.windowTrades,
      n_trades_observed: health.nTradesObserved,
      trailing_drawdown_pct: health.trailingDrawdownPct,
      trailing_pnl_pct: health.trailingPnlPct,
      drawdown_floor_pct: health.drawdownFloorPct,
      warn_threshold_pct: health.warnThresholdPct,
      warn_fraction: health.warnFraction,
      headroom_pct: health.headroomPct,
      needs_refit: health.needsRefit,
      floor_breached: health.floorBreached,
      computed_at: health.computedAt,
    };
    res.json(GetDiagnosticSandboxHealthResponse.parse(payload));
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : "unknown",
    });
  }
});

router.post("/diagnostic-sandbox/mode", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const body = req.body ?? {};
  const requested = typeof body.mode === "string" ? body.mode : null;
  if (requested !== "default" && requested !== "diagnostic_sandbox") {
    res.status(400).json({
      error: "Body must be { mode: 'default' | 'diagnostic_sandbox' }",
    });
    return;
  }
  try {
    const cfg = await setMttmMode(requested as MttmMode);
    res.json({
      mode: cfg.mode,
      universe: cfg.universe,
      maxPositionPct: cfg.maxPositionPct,
      ready: isDiagnosticSandboxReady(cfg),
    });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : "unknown",
    });
  }
});

router.post(
  "/diagnostic-sandbox/btc-version",
  async (req, res): Promise<void> => {
    if (!requireAdminApiKey(req, res)) return;
    const body = req.body ?? {};
    const v = body.version;
    if (v !== null && (typeof v !== "string" || v.length === 0)) {
      res.status(400).json({
        error: "Body must be { version: string | null }",
      });
      return;
    }
    try {
      const cfg = await setDiagnosticSandboxBtcVersion(v as string | null);
      res.json({
        btcVersion: cfg.diagnosticSandbox.btcVersion,
        ready: isDiagnosticSandboxReady(cfg),
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : "unknown",
      });
    }
  },
);

router.post(
  "/diagnostic-sandbox/evaluate",
  async (req, res): Promise<void> => {
    if (!requireAdminApiKey(req, res)) return;
    try {
      // Drift first (universe / promotion / scope), then trade-tally
      // breach. Either path can flip the lane; report whichever fired.
      const driftReason = await evaluateDiagnosticSandboxDrift();
      const tallyReason = driftReason
        ? null
        : await evaluateDiagnosticSandboxAutoDisable();
      const reason = driftReason ?? tallyReason;
      res.json({
        tripped: !!reason,
        reason,
        kind: driftReason ? "drift" : tallyReason ? "tally" : null,
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : "unknown",
      });
    }
  },
);

router.get("/crypto/monitoring-status", async (req, res): Promise<void> => {
  const state = await db.select().from(monitoringStateTable).limit(1);
  const agents = await db.select().from(agentsTable).where(eq(agentsTable.isActive, true));

  const status = {
    isRunning: state[0]?.isRunning || false,
    cycleCount: state[0]?.cycleCount || getCycleCount(),
    lastCycleAt: state[0]?.lastCycleAt?.toISOString() ?? null,
    nextCycleAt: state[0]?.nextCycleAt?.toISOString() ?? null,
    activeAgents: agents.filter((a) => a.status === "active").length,
    monitoredCoins: 10,
  };

  res.json(GetMonitoringStatusResponse.parse(status));
});

// Health surface for skip-event persistence. Lives outside the typed
// monitoring-status response so we don't have to bump the OpenAPI/Zod
// contract. Used by the dashboard to alert when silent DB writes are
// failing in the background loop.
router.get("/crypto/skip-tracker-health", (_req, res): void => {
  res.json(getSkipPersistHealth());
});

// Health surface for the market_signals poller (Task #285). Combines the
// in-process poller status (last poll timestamp / ok / error) with a
// per-coin row count over the trailing hour so operators can see at a
// glance whether the new external signal streams (funding, OI,
// liquidations, spread, BTC/ETH lead prices) are flowing without
// running SQL by hand. `isStale` flips true when the last poll is older
// than 3x the configured poll interval (or has never run).
router.get("/crypto/market-signals-health", async (_req, res): Promise<void> => {
  try {
    const status = getMarketSignalsPollerStatus();
    const since = new Date(Date.now() - 60 * 60 * 1000);
    // Per-coin row count + per-field non-null count over the trailing
    // hour. The per-field counts let operators spot a silent partial
    // outage where rows are still being written but a single upstream
    // stream (e.g. funding) is returning null. Without this, a coin's
    // total row count looks healthy even when 4/5 of its signals are
    // dead.
    const rows = await db
      .select({
        coinId: marketSignalsTable.coinId,
        count: sql<number>`count(*)::int`,
        lastTimestamp: sql<Date>`max(${marketSignalsTable.timestamp})`,
        fundingRateCount: sql<number>`count(${marketSignalsTable.fundingRate})::int`,
        openInterestUsdCount: sql<number>`count(${marketSignalsTable.openInterestUsd})::int`,
        liquidations1hUsdCount: sql<number>`count(${marketSignalsTable.liquidations1hUsd})::int`,
        bidAskSpreadBpsCount: sql<number>`count(${marketSignalsTable.bidAskSpreadBps})::int`,
        midPriceCount: sql<number>`count(${marketSignalsTable.midPrice})::int`,
      })
      .from(marketSignalsTable)
      .where(gte(marketSignalsTable.timestamp, since))
      .groupBy(marketSignalsTable.coinId);

    // Build the per-coin response from the canonical poller target list
    // (monitored coins on OKX SWAP + btc/eth lead refs) and zero-fill any
    // expected coin that wrote no rows in the last hour. Without this,
    // a silently broken stream would simply be absent from the response
    // and the dashboard would still look healthy. Coins that wrote rows
    // but aren't in the canonical list (e.g. retired coin still present
    // historically) are appended so they're not lost.
    type FieldCounts = {
      fundingRate: number;
      openInterestUsd: number;
      liquidations1hUsd: number;
      bidAskSpreadBps: number;
      midPrice: number;
    };
    type PerCoinRow = {
      coinId: string;
      rowsLastHour: number;
      lastTimestamp: string | null;
      fieldNonNullCounts: FieldCounts;
    };
    const rowMap = new Map<string, Omit<PerCoinRow, "coinId">>(
      rows.map((r) => [r.coinId, {
        rowsLastHour: r.count,
        lastTimestamp: r.lastTimestamp ? new Date(r.lastTimestamp).toISOString() : null,
        fieldNonNullCounts: {
          fundingRate: r.fundingRateCount,
          openInterestUsd: r.openInterestUsdCount,
          liquidations1hUsd: r.liquidations1hUsdCount,
          bidAskSpreadBps: r.bidAskSpreadBpsCount,
          midPrice: r.midPriceCount,
        },
      }]),
    );
    const emptyCounts: FieldCounts = {
      fundingRate: 0,
      openInterestUsd: 0,
      liquidations1hUsd: 0,
      bidAskSpreadBps: 0,
      midPrice: 0,
    };
    const targets = getMarketSignalsPollerTargets();
    const seen = new Set<string>();
    const perCoin: PerCoinRow[] = [];
    for (const coinId of targets) {
      const r = rowMap.get(coinId);
      perCoin.push({
        coinId,
        rowsLastHour: r?.rowsLastHour ?? 0,
        lastTimestamp: r?.lastTimestamp ?? null,
        fieldNonNullCounts: r?.fieldNonNullCounts ?? { ...emptyCounts },
      });
      seen.add(coinId);
    }
    for (const r of rows) {
      if (seen.has(r.coinId)) continue;
      const m = rowMap.get(r.coinId)!;
      perCoin.push({
        coinId: r.coinId,
        rowsLastHour: m.rowsLastHour,
        lastTimestamp: m.lastTimestamp,
        fieldNonNullCounts: m.fieldNonNullCounts,
      });
    }
    perCoin.sort((a, b) => a.coinId.localeCompare(b.coinId));

    const totalRowsLastHour = perCoin.reduce((acc, r) => acc + r.rowsLastHour, 0);
    const staleThresholdMs = status.intervalMs * 3;
    const isStale =
      status.lastPollAt == null || Date.now() - status.lastPollAt > staleThresholdMs;

    // Task #290 — surface off-dashboard alert hook configuration + watcher
    // state so the dashboard can show whether outages will page someone
    // when nobody has the page open. Failures here must never break the
    // health route, so we read state best-effort.
    const channels = getMarketSignalsAlertChannels();
    let watcherState: Awaited<ReturnType<typeof getMarketSignalsWatcherState>> | null = null;
    try {
      watcherState = await getMarketSignalsWatcherState();
    } catch {
      watcherState = null;
    }
    // Task #303 — surface a bounded recent-alert history (last 20) so
    // operators can spot flapping outages from the dashboard. Best
    // effort; never break the health route.
    let recentAlerts: Awaited<ReturnType<typeof getMarketSignalsWatcherHistory>> = [];
    try {
      recentAlerts = await getMarketSignalsWatcherHistory(20);
    } catch {
      recentAlerts = [];
    }
    let snooze: Awaited<ReturnType<typeof getMarketSignalsWatcherSnooze>> | null = null;
    try {
      snooze = await getMarketSignalsWatcherSnooze();
    } catch {
      snooze = null;
    }

    res.json({
      lastPollAt: status.lastPollAt ? new Date(status.lastPollAt).toISOString() : null,
      lastPollOk: status.lastPollOk,
      lastPollError: status.lastPollError,
      intervalMs: status.intervalMs,
      staleThresholdMs,
      isStale,
      totalRowsLastHour,
      perCoin,
      alertHook: {
        configured: channels.genericConfigured || channels.slackConfigured,
        genericConfigured: channels.genericConfigured,
        slackConfigured: channels.slackConfigured,
        activeIncident: !!watcherState?.activeIncidentKey,
        activeIncidentReason: watcherState?.activeIncidentReason ?? null,
        activeIncidentSince: watcherState?.activeIncidentSince ?? null,
        lastAlertAt: watcherState?.lastAlertAt ?? null,
        lastAlertKind: watcherState?.lastAlertKind ?? null,
        lastAlertReason: watcherState?.lastAlertReason ?? null,
        recentAlerts,
        // Task #302 — per-coin silent-stream sub-incidents that the
        // watcher is currently paging on. Surfacing this lets the
        // dashboard render which coins have an active sub-incident
        // (badge), and when each was first observed.
        silentCoinIncidents: Object.entries(watcherState?.silentCoinIncidents ?? {})
          .map(([coinId, info]) => ({ coinId, since: info.since }))
          .sort((a, b) => a.coinId.localeCompare(b.coinId)),
        lastSilentCoinAlertAt: watcherState?.lastSilentCoinAlertAt ?? null,
        lastSilentCoinAlertKind: watcherState?.lastSilentCoinAlertKind ?? null,
        lastSilentCoinAlertReason: watcherState?.lastSilentCoinAlertReason ?? null,
        // Task #301 — snooze surface. While snoozed, dispatches are
        // suppressed; the dashboard surfaces what *would* have alerted
        // via the pendingIncident* fields below.
        snoozed: !!snooze,
        snoozedUntil: snooze?.snoozedUntil ?? null,
        snoozedAt: snooze?.snoozedAt ?? null,
        snoozeDuration: snooze?.duration ?? null,
        pendingIncident: !!watcherState?.pendingIncidentKey,
        pendingIncidentReason: watcherState?.pendingIncidentReason ?? null,
        pendingIncidentSince: watcherState?.pendingIncidentSince ?? null,
      },
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Per-exchange breakdown of the aggregated `liquidations_1h_usd` field
// (Task #286 wrote `source_breakdown` but nothing surfaced it). For each
// coin that has any liquidations data in the last 6 hours we return the
// latest total + the latest per-source split so the UI can render
// "OKX 42% / Gate 58%". For each source that was contributing within
// the lookback window but stopped, we expose `silent: true` once it's
// been quiet for more than `silentThresholdMinutes` (15) so the UI can
// chip a warning when a venue goes dark.
router.get("/crypto/liquidations-breakdown", async (_req, res): Promise<void> => {
  try {
    const now = Date.now();
    const lookback = new Date(now - 6 * 60 * 60 * 1000);
    const rows = await db
      .select({
        coinId: marketSignalsTable.coinId,
        timestamp: marketSignalsTable.timestamp,
        liquidations1hUsd: marketSignalsTable.liquidations1hUsd,
        sourceBreakdown: marketSignalsTable.sourceBreakdown,
      })
      .from(marketSignalsTable)
      .where(and(
        gte(marketSignalsTable.timestamp, lookback),
        sql`${marketSignalsTable.liquidations1hUsd} IS NOT NULL`,
      ))
      .orderBy(desc(marketSignalsTable.timestamp));

    interface SrcAgg { lastSeenAt: number }
    interface CoinAgg {
      coinId: string;
      latestTimestamp: number;
      latestTotalUsd: number;
      latestBreakdown: Record<string, number>;
      sources: Map<string, SrcAgg>;
    }
    const coins = new Map<string, CoinAgg>();
    for (const row of rows) {
      const ts = row.timestamp.getTime();
      let agg = coins.get(row.coinId);
      if (!agg) {
        agg = {
          coinId: row.coinId,
          latestTimestamp: ts,
          latestTotalUsd: row.liquidations1hUsd ?? 0,
          latestBreakdown: (row.sourceBreakdown ?? {}) as Record<string, number>,
          sources: new Map(),
        };
        coins.set(row.coinId, agg);
      }
      const breakdown = (row.sourceBreakdown ?? {}) as Record<string, number>;
      for (const [src, val] of Object.entries(breakdown)) {
        if (typeof val !== "number" || val <= 0) continue;
        const cur = agg.sources.get(src);
        if (!cur || cur.lastSeenAt < ts) {
          agg.sources.set(src, { lastSeenAt: ts });
        }
      }
    }

    const SILENT_THRESHOLD_MIN = 15;
    const SILENT_THRESHOLD_MS = SILENT_THRESHOLD_MIN * 60 * 1000;
    const result = Array.from(coins.values())
      .filter((c) => c.sources.size > 0)
      .map((c) => {
        const totalUsd = c.latestTotalUsd;
        const sources = Array.from(c.sources.entries()).map(([source, info]) => {
          const rawCurrent = c.latestBreakdown[source];
          const currentUsd = typeof rawCurrent === "number" && rawCurrent > 0 ? rawCurrent : 0;
          const sharePct = totalUsd > 0 ? (currentUsd / totalUsd) * 100 : 0;
          const silentMs = now - info.lastSeenAt;
          return {
            source,
            currentUsd,
            sharePct,
            lastSeenAt: new Date(info.lastSeenAt).toISOString(),
            silent: silentMs > SILENT_THRESHOLD_MS,
            silentMinutes: Math.floor(silentMs / 60000),
          };
        }).sort((a, b) => b.sharePct - a.sharePct || a.source.localeCompare(b.source));
        return {
          coinId: c.coinId,
          totalUsd,
          latestTimestamp: new Date(c.latestTimestamp).toISOString(),
          sources,
        };
      })
      .sort((a, b) => b.totalUsd - a.totalUsd || a.coinId.localeCompare(b.coinId));

    res.json({
      generatedAt: new Date().toISOString(),
      silentThresholdMinutes: SILENT_THRESHOLD_MIN,
      lookbackHours: 6,
      coins: result,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "liquidations-breakdown failed" });
  }
});

// Task #301 — operator-controlled mute for the market signals alert hook.
// Suppresses off-dashboard dispatches (webhook + Slack) for a bounded
// window during planned maintenance. Body: { duration: "15m" | "1h" |
// "until_midnight" }. Persisted in app_settings so a backend restart
// during the maintenance window does not silently un-mute.
router.post("/crypto/market-signals/snooze", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const duration = (req.body ?? {}).duration;
  if (!isSnoozeDuration(duration)) {
    res.status(400).json({
      error: "Body must be { duration: '15m' | '1h' | 'until_midnight' }",
    });
    return;
  }
  try {
    const snooze = await setMarketSignalsWatcherSnooze(duration);
    res.json(snooze);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #301 — operator-initiated unmute. Clears any active snooze; the
// next watcher tick resumes normal dispatch behavior.
router.post("/crypto/market-signals/snooze/clear", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    await clearMarketSignalsWatcherSnooze();
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #411 — combined surface for the dashboard's 5m top-up banner.
// Joins the live ml-engine `scheduled_5m_topup.state` (so the dashboard
// always sees the same per-coin contiguous_days the watcher does) with
// the api-server's notifier dedup state (active incident, last alert,
// channel config). The dashboard banner is "visible when any coin's
// contiguous_days < threshold OR the daily tick has failed twice in a
// row" — both inputs come from this endpoint.
router.get("/crypto/5m-topup-health", async (_req, res): Promise<void> => {
  try {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
    let topup: Record<string, unknown> | null = null;
    let fetchError: string | null = null;
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 4_000);
    try {
      const r = await fetch(`${base}/ml/admin/5m-topup/status`, { signal: ctrl.signal });
      if (!r.ok) {
        fetchError = `ml-engine HTTP ${r.status}`;
      } else {
        topup = (await r.json()) as Record<string, unknown>;
      }
    } catch (err) {
      fetchError = err instanceof Error ? err.message : String(err);
    } finally {
      clearTimeout(t);
    }

    const notifier = await getTopup5mNotifierState();
    const channels = getTopup5mAlertChannels();

    // Surface the banner trigger inputs in a normalized shape so the
    // dashboard can render without re-implementing the threshold logic.
    const alertCoins: string[] = Array.isArray(topup?.last_alerts)
      ? (topup!.last_alerts as unknown[]).filter((x): x is string => typeof x === "string")
      : [];
    const lastOutcome = (topup?.last_attempt_outcome as string | null) ?? null;
    const consecutiveErrors = notifier.consecutiveErrors;
    const errorStreakTriggered = consecutiveErrors >= TOPUP_5M_ERROR_STREAK_THRESHOLD;
    const bannerVisible = errorStreakTriggered || alertCoins.length > 0;

    res.json({
      generatedAt: new Date().toISOString(),
      fetchError,
      bannerVisible,
      errorStreakTriggered,
      errorStreakThreshold: TOPUP_5M_ERROR_STREAK_THRESHOLD,
      consecutiveErrors,
      alertCoins,
      thresholdDays: typeof topup?.alert_below_days === "number" ? topup.alert_below_days : null,
      perCoinContiguousDays: (topup?.last_health_per_coin as Record<string, number> | null) ?? null,
      lastAttemptOutcome: lastOutcome,
      lastCheckAt: typeof topup?.last_check_at === "number"
        ? new Date((topup.last_check_at as number) * 1000).toISOString()
        : null,
      lastFinishedAt: typeof topup?.last_finished_at === "number"
        ? new Date((topup.last_finished_at as number) * 1000).toISOString()
        : null,
      lastError: (topup?.last_error as string | null) ?? null,
      schedulerEnabled: typeof topup?.enabled === "boolean" ? topup.enabled : null,
      ticksTotal: typeof topup?.ticks_total === "number" ? topup.ticks_total : null,
      runsTotal: typeof topup?.runs_total === "number" ? topup.runs_total : null,
      alertHook: {
        configured: channels.configured,
        genericConfigured: channels.genericConfigured,
        slackConfigured: channels.slackConfigured,
        activeIncident: notifier.activeIncidentKey != null,
        activeIncidentReason: notifier.activeIncidentReason,
        activeIncidentSince: notifier.activeIncidentSince,
        lastAlertAt: notifier.lastAlertAt,
        lastAlertKind: notifier.lastAlertKind,
        lastAlertReason: notifier.lastAlertReason,
      },
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "5m-topup-health failed" });
  }
});

// Task #577 — banner state for the disabled-outcome notifier.
//
// Surfaces the "outcome landed on a disabled timeframe" alarm. The
// upstream signal is the brain returning `{ok:false, reason:
// "disabled_role_rejected"}` from `/ml/meta-brain/record-outcome`;
// the api-server's meta-brain client pushes each rejection into a
// process-local ring buffer and dispatches webhooks on rising-edge.
// This endpoint just exposes the same recent-event summary so the
// dashboard banner can show what the operator would have been paged
// for. The banner is visible whenever any rejection has been observed
// in the active window (configurable via DISABLED_OUTCOME_WINDOW_MINUTES,
// default 5).
router.get("/crypto/disabled-outcome-health", async (_req, res): Promise<void> => {
  try {
    const state = await getDisabledOutcomeBannerState();
    res.json({
      generatedAt: new Date().toISOString(),
      ...state,
    });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : "disabled-outcome-health failed",
    });
  }
});

// GET /crypto/quant-abstain-reasons?hours=24
// Aggregates prediction_journal rows where the bot abstained from
// trading because the quant brain had no usable model. Skip reasons of
// the form `quant_abstain_<reason>` are written by the trade loop in
// monitor.ts when prediction.brain === "ABSTAIN". Operators see fewer
// trades; this card explains *why* and points at which (coin,
// timeframe) pairs need a model trained.
router.get("/crypto/quant-abstain-reasons", async (req, res): Promise<void> => {
  try {
    const rawHours = Number(req.query.hours ?? 24);
    const hours = Number.isFinite(rawHours) ? Math.min(168, Math.max(1, rawHours)) : 24;
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);
    const rows = await db
      .select({
        coinId: predictionJournalTable.coinId,
        coinName: predictionJournalTable.coinName,
        timeframe: predictionJournalTable.timeframe,
        skipReason: predictionJournalTable.skipReason,
      })
      .from(predictionJournalTable)
      .where(
        and(
          gte(predictionJournalTable.createdAt, since),
          eq(predictionJournalTable.becameTrade, false),
          sql`${predictionJournalTable.skipReason} LIKE 'quant_abstain_%'`,
        ),
      );

    interface PairAgg {
      coinId: string;
      coinName: string | null;
      timeframe: string;
      total: number;
      reasons: Record<string, number>;
    }
    const pairKey = (coinId: string, timeframe: string) => `${coinId}::${timeframe}`;
    const pairs = new Map<string, PairAgg>();
    const reasonTotals: Record<string, number> = {};

    for (const r of rows) {
      const reason = (r.skipReason ?? "quant_abstain_no_model").replace(/^quant_abstain_/, "") || "no_model";
      const k = pairKey(r.coinId, r.timeframe);
      let cell = pairs.get(k);
      if (!cell) {
        cell = {
          coinId: r.coinId,
          coinName: r.coinName,
          timeframe: r.timeframe,
          total: 0,
          reasons: {},
        };
        pairs.set(k, cell);
      }
      cell.total += 1;
      cell.reasons[reason] = (cell.reasons[reason] ?? 0) + 1;
      reasonTotals[reason] = (reasonTotals[reason] ?? 0) + 1;
    }

    const byPair = Array.from(pairs.values())
      .map((p) => ({
        ...p,
        topReason: Object.entries(p.reasons).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "no_model",
      }))
      .sort((a, b) => b.total - a.total);

    const byReason = Object.entries(reasonTotals)
      .map(([reason, count]) => ({ reason, count }))
      .sort((a, b) => b.count - a.count);

    res.json({
      hours,
      generatedAt: new Date().toISOString(),
      totalAbstains: rows.length,
      uniquePairs: byPair.length,
      byReason,
      byPair,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "quant-abstain-reasons failed" });
  }
});


// Task #453 — `/crypto/api-budget` endpoint removed alongside
// `rate-limiter.ts`. After Task #444 nothing called `recordApiCall`, so
// this endpoint always reported zero gpt/gemini usage with a permanent
// "dual" mode. The matching frontend hook (`useApiBudget`) had no
// consumers.

/**
 * Reality-check P&L: subtracts realistic exchange costs and benchmarks
 * each agent against a naive equal-weighted buy-and-hold of the monitored coin basket.
 *
 * Cost model (per side):
 *   - 0.20% taker fee (Binance VIP 0)
 *   - 0.10% slippage on these small/mid-cap altcoins
 * Round-trip = 0.60% of position size per closed trade.
 * Open positions are charged one side (entry) only — they haven't been sold yet.
 */
router.get("/crypto/reality-check", async (req, res): Promise<void> => {
  // Task #365 — fees & slippage MUST come from shared/trading-frictions.json
  // via trading-constants.ts. The previous local literals (0.0010 + 0.0005)
  // happened to match paper-trader at the time, but a future tuning change
  // to the shared contract would have left this endpoint silently
  // misreporting net P&L. The shared module is the single source of truth.
  const { TAKER_FEE_PCT, SLIPPAGE_PCT, INITIAL_BALANCE_USD } = await import(
    "../../lib/trading-constants"
  );
  const FEE_PER_SIDE = TAKER_FEE_PCT;
  const SLIPPAGE_PER_SIDE = SLIPPAGE_PCT;
  const COST_PER_SIDE = FEE_PER_SIDE + SLIPPAGE_PER_SIDE;
  const STARTING_CASH = INITIAL_BALANCE_USD;

  // Only count ACTIVE ai-bots so the Fleet (15 × $1000 = $15,000) and
  // Quant Brain Performance cards report against the same denominator.
  // Including deactivated bots inflated starting capital to $20k+ and made
  // the % return look smaller than the dashboard's Fleet Performance card.
  const aiAgents = await db
    .select({ id: agentsTable.id })
    .from(agentsTable)
    .where(
      and(
        eq(agentsTable.strategyType, "ai-bots"),
        eq(agentsTable.isActive, true),
        isNull(agentsTable.archivedAt),
        sql`${agentsTable.profileId} IS DISTINCT FROM 'legacy_archived'`,
      ),
    );
  const aiAgentIds = new Set(aiAgents.map(a => a.id));
  const portfoliosAll = await db.select().from(paperPortfoliosTable);
  const portfolios = portfoliosAll.filter(p => aiAgentIds.has(p.agentId));
  const allTrades = await db.select().from(paperTradesTable);
  const tradesAll = allTrades.filter((t) => aiAgentIds.has(t.agentId));
  const allOpenPositions = await db.select().from(paperPositionsTable);
  const openPositions = allOpenPositions.filter((p) => aiAgentIds.has(p.agentId));

  // Prior-only pooled-fallback predictions return the same Laplace-smoothed
  // marginals on every call, so any P&L attributed to them is structural
  // noise rather than skill. Default: drop trades that came from a
  // source='prior' prediction. Pass ?includePrior=1 to fold them back in.
  const includePriorRaw = String((req.query as { includePrior?: string }).includePrior ?? "");
  const includePrior = includePriorRaw === "1" || includePriorRaw.toLowerCase() === "true";
  const tradePredictionIds = Array.from(
    new Set(tradesAll.map((t) => t.predictionId).filter((v): v is number => typeof v === "number")),
  );
  const priorPredIds = new Set<number>();
  if (tradePredictionIds.length > 0) {
    // Parameterized `inArray` (not raw string concat) so non-integer
    // predictionIds can never slip into the SQL. tradePredictionIds is
    // already filtered to `typeof === "number"` above, but defense in
    // depth is free here. Always computed (regardless of includePrior)
    // so the response can report the total count of prior-source trades
    // — the dashboard badge needs that to stay visible after the user
    // toggles includePrior=1.
    const rows = await db
      .select({ id: predictionsTable.id })
      .from(predictionsTable)
      .where(and(
        eq(predictionsTable.source, "prior"),
        inArray(predictionsTable.id, tradePredictionIds),
      ));
    for (const r of rows) {
      if (Number.isFinite(r.id)) priorPredIds.add(r.id);
    }
  }
  const priorTradesTotal = tradesAll.filter(
    (t) => t.predictionId != null && priorPredIds.has(t.predictionId),
  ).length;
  const trades = includePrior
    ? tradesAll
    : tradesAll.filter((t) => t.predictionId == null || !priorPredIds.has(t.predictionId));
  const priorTradesExcluded = tradesAll.length - trades.length;
  const prices = await fetchCoinPrices();
  const priceById = new Map(prices.map((p) => [p.id, p.currentPrice]));

  // Buy-and-hold benchmark: use the FROZEN Strategy Lab "Buy & Hold" bucket
  // as the source of truth. That bucket bought $1,000 of an equal-weight basket
  // at the start of measurement mode and has held it ever since — that's the
  // honest, cumulative benchmark to beat. The previous 24h-rolling proxy was
  // misleading because the window kept sliding.
  const stratComparison = await getStrategyComparison();
  const hodlBucket = stratComparison.buckets.find((b) => b.strategyType === "buy-hold");
  const benchmarkPctSinceStart = hodlBucket?.totalPnlPct ?? 0;

  const agentResults = portfolios.map((p) => {
    const agentTrades = trades.filter((t) => t.agentId === p.agentId);
    const closedTrades = agentTrades.filter((t) => t.status === "closed");
    const openTradesForAgent = agentTrades.filter((t) => t.status === "open");
    const positions = openPositions.filter((pos) => pos.agentId === p.agentId);

    // Round-trip costs for closed trades + entry-only costs for still-open trades
    const closedTurnover = closedTrades.reduce((s, t) => s + Number(t.positionSize ?? 0), 0);
    const openTurnover = openTradesForAgent.reduce((s, t) => s + Number(t.positionSize ?? 0), 0);
    const totalCosts = closedTurnover * COST_PER_SIDE * 2 + openTurnover * COST_PER_SIDE;

    // Reported total value already includes unrealized gains on open positions.
    // Recompute it independently so we don't depend on potentially stale `total_value`.
    const openMarketValue = positions.reduce((s, pos) => {
      const px = priceById.get(pos.coinId) ?? Number(pos.entryPrice ?? 0);
      const qty = Number(pos.quantity ?? 0);
      return s + px * qty;
    }, 0);
    const grossTotal = Number(p.cashBalance ?? 0) + openMarketValue;
    const netTotal = grossTotal - totalCosts;

    const grossReturnPct = ((grossTotal - STARTING_CASH) / STARTING_CASH) * 100;
    const netReturnPct = ((netTotal - STARTING_CASH) / STARTING_CASH) * 100;
    const alphaVsBenchmark = netReturnPct - benchmarkPctSinceStart;

    return {
      agentId: p.agentId,
      agentName: p.agentName,
      grossTotalValue: grossTotal,
      netTotalValue: netTotal,
      grossPnlUsd: grossTotal - STARTING_CASH,
      netPnlUsd: netTotal - STARTING_CASH,
      grossReturnPct,
      netReturnPct,
      alphaVsBenchmark,
      totalCosts,
      closedTrades: closedTrades.length,
      openTrades: openTradesForAgent.length,
      turnover: closedTurnover + openTurnover,
    };
  });

  agentResults.sort((a, b) => b.netPnlUsd - a.netPnlUsd);

  const fleetStartingCapital = agentResults.length * STARTING_CASH;
  const fleetGrossPnlUsd = agentResults.reduce((s, a) => s + a.grossPnlUsd, 0);
  const fleetNetPnlUsd = agentResults.reduce((s, a) => s + a.netPnlUsd, 0);
  const fleetGross = agentResults.length
    ? agentResults.reduce((s, a) => s + a.grossReturnPct, 0) / agentResults.length
    : 0;
  const fleetNet = agentResults.length
    ? agentResults.reduce((s, a) => s + a.netReturnPct, 0) / agentResults.length
    : 0;
  const fleetCosts = agentResults.reduce((s, a) => s + a.totalCosts, 0);
  const fleetTurnover = agentResults.reduce((s, a) => s + a.turnover, 0);
  const totalClosedTrades = agentResults.reduce((s, a) => s + a.closedTrades, 0);
  const winners = agentResults.filter((a) => a.netReturnPct > benchmarkPctSinceStart).length;
  const benchmarkPnlUsd = (benchmarkPctSinceStart / 100) * fleetStartingCapital;

  res.json({
    assumptions: {
      feePerSidePct: FEE_PER_SIDE * 100,
      slippagePerSidePct: SLIPPAGE_PER_SIDE * 100,
      roundTripCostPct: COST_PER_SIDE * 2 * 100,
      benchmarkLabel: "Frozen Strategy Lab Buy & Hold ($1k bought once at measurement start)",
    },
    benchmark: {
      returnPct: benchmarkPctSinceStart,
      pnlUsd: benchmarkPnlUsd,
    },
    fleet: {
      startingCapital: fleetStartingCapital,
      grossPnlUsd: fleetGrossPnlUsd,
      netPnlUsd: fleetNetPnlUsd,
      alphaPnlUsd: fleetNetPnlUsd - benchmarkPnlUsd,
      avgGrossReturnPct: fleetGross,
      avgNetReturnPct: fleetNet,
      avgAlphaVsBenchmark: fleetNet - benchmarkPctSinceStart,
      totalCosts: fleetCosts,
      totalTurnover: fleetTurnover,
      totalClosedTrades,
      agentsBeatingBenchmark: winners,
      totalAgents: agentResults.length,
    },
    autoDeploy: {
      deployedLast24hUsd: getAutoDeployedLast24hUsd(),
      attribution: await getAutoDeployAttribution(prices),
    },
    priorFallback: {
      included: includePrior,
      tradesExcluded: priorTradesExcluded,
      tradesTotal: priorTradesTotal,
    },
    agents: agentResults,
  });
});

router.get("/crypto/auto-deploy-attribution-history", async (req, res): Promise<void> => {
  const hoursParam = Number(req.query.hours);
  const hours = Number.isFinite(hoursParam) && hoursParam > 0 && hoursParam <= 24 * 30
    ? hoursParam
    : 168;
  const points = await getAutoDeployAttributionHistory(hours);
  let delta24h: number | null = null;
  if (points.length >= 2) {
    const latest = points[points.length - 1];
    const cutoff = latest.capturedAt - 24 * 60 * 60 * 1000;
    let baselineIdx = -1;
    for (let i = points.length - 2; i >= 0; i--) {
      if (points[i].capturedAt <= cutoff) { baselineIdx = i; break; }
    }
    if (baselineIdx === -1) baselineIdx = 0;
    delta24h = latest.totalNetPnlUsd - points[baselineIdx].totalNetPnlUsd;
  }
  res.json({ hours, points, delta24h });
});

router.get("/crypto/skip-reasons", async (req, res): Promise<void> => {
  const hoursParam = Number(req.query.hours);
  const hours = Number.isFinite(hoursParam) && hoursParam > 0 ? hoursParam : 24;
  const summary = await getSkipReasonsSummary(hours * 60 * 60 * 1000);
  const suggestion = getCachedSuggestion();
  res.json({ ...summary, suggestion });
});

router.get("/crypto/tuning", async (_req, res): Promise<void> => {
  res.json({
    ...getTuningState(),
    suggestion: getCachedSuggestion(),
    autoApplyTighten: getAutoApplyTightenStatus(),
    pendingTighten: getPendingTighten(),
    autoApplyTightenTicks: getAutoApplyTightenTicks(),
    autoApplyTightenEnabled: isAutoApplyTightenEnabled(),
  });
});

router.post("/crypto/tuning/auto-apply-tighten", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const body = (req.body ?? {}) as { enabled?: unknown };
  let value: boolean | null;
  if (body.enabled === null) {
    value = null;
  } else if (typeof body.enabled === "boolean") {
    value = body.enabled;
  } else {
    res.status(400).json({
      error: "Body must be { enabled: boolean | null }. Use null to clear the override and fall back to the env default.",
    });
    return;
  }
  try {
    const status = await setAutoApplyTightenOverride(value);
    res.json({ autoApplyTighten: status });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({
      error: `Failed to persist auto-apply tighten override: ${message}`,
    });
  }
});

router.post("/crypto/tuning/apply", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const body = (req.body ?? {}) as { gate?: string; source?: string; direction?: string };
    const gate = body.gate as GateKey | undefined;
    if (!gate) {
      res.status(400).json({ error: "Missing gate" });
      return;
    }
    const direction = body.direction === "tighten" ? "tighten" : "loosen";
    let source: "auto-suggest" | "auto-tighten" | "manual";
    if (body.source === "auto-suggest" || body.source === "auto-tighten") {
      source = body.source;
    } else if (body.source === "manual") {
      source = "manual";
    } else {
      // Default the source from direction when not explicitly provided.
      source = direction === "tighten" ? "auto-tighten" : "auto-suggest";
    }
    const change =
      direction === "tighten" ? applyTighten(gate, source) : applyLoosen(gate, source);
    setCachedSuggestion(null);
    res.json({ change, state: getTuningState() });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    res.status(400).json({ error: message });
  }
});

router.post("/crypto/tuning/revert", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const body = (req.body ?? {}) as { changeId?: string };
    if (!body.changeId) {
      res.status(400).json({ error: "Missing changeId" });
      return;
    }
    const change = revertChange(body.changeId);
    res.json({ change, state: getTuningState() });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    res.status(400).json({ error: message });
  }
});

router.get("/crypto/skip-timeline", async (req, res): Promise<void> => {
  const hoursParam = Number(req.query.hours);
  const hours = Number.isFinite(hoursParam) && hoursParam > 0 ? hoursParam : 24;
  const bucketHoursParam = Number(req.query.bucketHours);
  const bucketMs = Number.isFinite(bucketHoursParam) && bucketHoursParam > 0
    ? bucketHoursParam * 60 * 60 * 1000
    : undefined;
  const timeline = await getSkipTimeline(hours * 60 * 60 * 1000, bucketMs);
  res.json(timeline);
});

router.get("/crypto/skip-events", async (req, res): Promise<void> => {
  const reason = String(req.query.reason ?? "") as SkipReason;
  const bucketTs = Number(req.query.bucketTs);
  const bucketMs = Number(req.query.bucketMs);
  if (!reason || !Number.isFinite(bucketTs) || !Number.isFinite(bucketMs) || bucketMs <= 0) {
    res.status(400).json({ error: "Missing or invalid reason/bucketTs/bucketMs" });
    return;
  }
  const events = await getSkipsInBucket(reason, bucketTs, bucketMs);
  res.json({ reason, bucketTs, bucketMs, count: events.length, events });
});

router.get("/crypto/coin-signals", async (req, res): Promise<void> => {
  const agents = await db.select().from(agentsTable);
  const recentPreds = await db
    .select()
    .from(predictionsTable)
    .where(gte(predictionsTable.createdAt, new Date(Date.now() - 15 * 60 * 1000)))
    .orderBy(desc(predictionsTable.createdAt));

  const prices = await fetchCoinPrices();
  const coinSignals: Record<string, { buyScore: number; sellScore: number; count: number; directions: string[]; confidences: number[] }> = {};

  for (const pred of recentPreds) {
    if (!coinSignals[pred.coinId]) {
      coinSignals[pred.coinId] = { buyScore: 0, sellScore: 0, count: 0, directions: [], confidences: [] };
    }
    const agent = agents.find((a) => a.id === pred.agentId);
    const accuracy = agent && agent.totalPredictions > 0 ? agent.correctPredictions / agent.totalPredictions : 0.5;
    const weight = (agent?.score || 50) / 100 * accuracy * pred.confidence;

    if (pred.direction === "up") coinSignals[pred.coinId].buyScore += weight;
    else if (pred.direction === "down") coinSignals[pred.coinId].sellScore += weight;
    coinSignals[pred.coinId].count++;
    coinSignals[pred.coinId].directions.push(pred.direction);
    coinSignals[pred.coinId].confidences.push(pred.confidence);
  }

  const signals = prices.map((coin) => {
    const s = coinSignals[coin.id];
    if (!s || s.count === 0) {
      return { coinId: coin.id, coinName: coin.name, coinSymbol: coin.symbol, currentPrice: coin.currentPrice, priceChange24h: coin.priceChange24h || 0, signal: "hold" as const, strength: 0, confidence: 0, agentAgreement: 0 };
    }
    const total = s.buyScore + s.sellScore;
    const signal = s.buyScore > s.sellScore * 1.2 ? "buy" as const : s.sellScore > s.buyScore * 1.2 ? "sell" as const : "hold" as const;
    const dominant = signal === "buy" ? s.buyScore : signal === "sell" ? s.sellScore : Math.max(s.buyScore, s.sellScore);
    const strength = total > 0 ? Math.min(100, Math.round((dominant / total) * 100)) : 0;
    const avgConf = s.confidences.reduce((a, b) => a + b, 0) / s.confidences.length;
    const agreementDir = signal === "buy" ? "up" : signal === "sell" ? "down" : "stable";
    const agreement = Math.round((s.directions.filter((d) => d === agreementDir).length / s.directions.length) * 100);
    return { coinId: coin.id, coinName: coin.name, coinSymbol: coin.symbol, currentPrice: coin.currentPrice, priceChange24h: coin.priceChange24h || 0, signal, strength, confidence: Math.round(avgConf * 100), agentAgreement: agreement };
  });

  signals.sort((a, b) => b.strength - a.strength);
  res.json(signals);
});

router.get("/crypto/best-pick", async (req, res): Promise<void> => {
  // Task #532 / C-2 + C-12 — when the brain is offline, every agent
  // returns `direction=stable` with `confidence=0`, but the cached
  // `getLatestBestPick()` payload still ranks a coin like "HOLD SEI 37%"
  // because the consensus aggregator falls back to whichever coin had
  // the smallest residual disagreement. Surfacing that as a real
  // "consensus pick" misleads operators into thinking the brain is
  // authoring something. Short-circuit with an honest no-signal payload
  // whenever the brain is not in the `online` runtime state — that
  // covers all three cases the audit identified:
  //   * `offline_disabled`  — kill-switch off / env force-off / default
  //   * `offline_no_model`  — kill-switch on, but ml-engine has no
  //     trained model so every prediction is a no_model abstain
  //   * runtime-status read failure — fail closed, never serve a
  //     consensus when we cannot prove the brain is actually authoring.
  // The dashboard renders this as a "no live consensus" banner instead
  // of a fake recommendation; the cached `bestPick` is preserved and
  // automatically returns once the brain is back online.
  try {
    const runtime = await computeBrainRuntimeState();
    if (runtime.state !== "online") {
      const reasonByState: Record<string, string> = {
        offline_disabled: "brain_offline",
        offline_no_model: "brain_offline_no_model",
      };
      const suppressedReason = reasonByState[runtime.state] ?? "brain_offline";
      const human =
        runtime.state === "offline_no_model"
          ? "Quant brain is enabled but ml-engine has no trained model loaded — every recent prediction was a no_model abstain. No live consensus pick is available."
          : "Quant brain is OFF (source: " +
            runtime.brainSource +
            "). No live consensus pick is available. Existing positions are unaffected.";
      res.json({
        coinId: "",
        coinName: "No live consensus",
        coinSymbol: "",
        currentPrice: 0,
        action: "hold",
        successProbability: 0,
        holdTimeframe: "n/a — brain offline",
        holdMinutes: 0,
        expectedPriceChange: 0,
        reasoning: human,
        agentConsensus: [],
        newsFactors: [],
        riskLevel: "medium",
        timeframeBreakdown: [],
        brain: null,
        suppressedReason,
        brainRuntimeState: runtime.state,
        updatedAt: new Date().toISOString(),
      });
      return;
    }
  } catch {
    // Fail-closed: if we cannot prove the brain is online, suppress
    // the cached pick. The BrainStateBanner separately surfaces the
    // "we can't read the flag" case.
    res.json({
      coinId: "",
      coinName: "No live consensus",
      coinSymbol: "",
      currentPrice: 0,
      action: "hold",
      successProbability: 0,
      holdTimeframe: "n/a — brain status unknown",
      holdMinutes: 0,
      expectedPriceChange: 0,
      reasoning:
        "Brain runtime status could not be read. Refusing to surface a cached consensus pick as live.",
      agentConsensus: [],
      newsFactors: [],
      riskLevel: "medium",
      timeframeBreakdown: [],
      brain: null,
      suppressedReason: "brain_status_unknown",
      brainRuntimeState: "unknown",
      updatedAt: new Date().toISOString(),
    });
    return;
  }
  const bestPick = getLatestBestPick();
  if (!bestPick) {
    res.json({
      coinId: "",
      coinName: "Waiting...",
      coinSymbol: "",
      currentPrice: 0,
      action: "hold",
      successProbability: 0,
      holdTimeframe: "Agents are still warming up — need at least 1 complete cycle",
      holdMinutes: 0,
      expectedPriceChange: 0,
      reasoning: "The monitoring engine hasn't completed enough cycles yet to generate a reliable recommendation. Please wait for more data.",
      agentConsensus: [],
      newsFactors: [],
      riskLevel: "medium",
      timeframeBreakdown: [],
      updatedAt: new Date().toISOString(),
    });
    return;
  }
  res.json(bestPick);
});

router.get("/crypto/strategy-lab", async (req, res): Promise<void> => {
  const { getStrategyComparison } = await import("../../lib/strategy-lab");
  const data = await getStrategyComparison();
  res.json(data);
});

router.get("/crypto/strategy-lab/settings", async (req, res): Promise<void> => {
  const { getStrategySettings, DCA_DEFAULTS } = await import("../../lib/strategy-lab");
  const settings = await getStrategySettings();
  res.json({ settings, defaults: DCA_DEFAULTS });
});

router.put("/crypto/strategy-lab/settings", async (req, res): Promise<void> => {
  const { updateStrategySettings, DCA_DEFAULTS } = await import("../../lib/strategy-lab");
  const body = req.body ?? {};
  const settings = await updateStrategySettings({
    drawdownTriggerPct: typeof body.drawdownTriggerPct === "number" ? body.drawdownTriggerPct : undefined,
    resumeLookbackDays: typeof body.resumeLookbackDays === "number" ? body.resumeLookbackDays : undefined,
    cycleDeployUsd: typeof body.cycleDeployUsd === "number" ? body.cycleDeployUsd : undefined,
    buyIntervalHours: typeof body.buyIntervalHours === "number" ? body.buyIntervalHours : undefined,
  });
  res.json({ settings, defaults: DCA_DEFAULTS });
});

router.get("/crypto/paper-portfolios", async (req, res): Promise<void> => {
  const { getPortfolioSummaries } = await import("../../lib/paper-trader");
  const portfolios = await getPortfolioSummaries();

  // Task #512 — exclude archived/legacy rows from the live executor
  // surface. The dashboard's "Quant Fleet / Bots in profit X/Y" math
  // and the paper-portfolio leaderboard MUST count only currently-
  // executing agents (the 4 deterministic executors + Strategy-Lab
  // baselines). Legacy personalities still live in the DB so trade
  // history stays intact, but they no longer dilute the live counters.
  const liveAgents = await db
    .select({
      id: agentsTable.id,
      profileId: agentsTable.profileId,
      archivedAt: agentsTable.archivedAt,
      isActive: agentsTable.isActive,
    })
    .from(agentsTable);
  // Task #518 — `baseline_reference` rows have `is_active = false` in
  // the DB by design (Strategy-Lab's seed step writes them that way so
  // the analysis cycle does not iterate them as agents). The original
  // `isActive === true` filter therefore stripped every baseline out
  // of `/paper-portfolios`, leaving the Benchmarks panel permanently
  // empty even though baselines were trading. Loosen the filter to
  // include any non-archived `baseline_reference` row regardless of
  // `is_active`. Executors and other live agents still require
  // `is_active = true`.
  const liveIds = new Set(
    liveAgents
      .filter(
        (a) =>
          a.archivedAt === null &&
          a.profileId !== "legacy_archived" &&
          (a.isActive === true || a.profileId === "baseline_reference"),
      )
      .map((a) => a.id),
  );
  // Tag baseline_reference rows so the frontend can sort them into a
  // Benchmarks panel (separate from the 4 executor cards). The shape
  // of every other field stays the same so existing consumers that
  // ignore `kind` continue to work.
  const profileById = new Map(liveAgents.map((a) => [a.id, a.profileId]));
  const filtered = portfolios
    .filter((p) => liveIds.has(p.agentId))
    .map((p) => ({
      ...p,
      kind:
        profileById.get(p.agentId) === "baseline_reference"
          ? "benchmark"
          : "executor",
      profileId: profileById.get(p.agentId) ?? null,
    }));

  res.json(filtered);
});

// Task #512 — executor-fleet summary endpoint. Aggregates the 4
// deterministic executor agents (one row per profile_id) into a single
// payload the dashboard's FamilyFleet renders as 4 cards. Numbers are
// derived from `paper_portfolios` + `paper_trades` so they match the
// existing leaderboard exactly. Baseline + legacy rows are excluded
// here — baselines have their own panel; legacy rows are archived.
router.get("/crypto/agents/families", async (_req, res): Promise<void> => {
  const { getPortfolioSummaries } = await import("../../lib/paper-trader");
  const { listExecutingProfileIds, getAgentProfile } = await import(
    "../../lib/agents-registry"
  );
  const { getRetirementSnapshot } = await import(
    "../../lib/agents-registry/retirement"
  );
  const { getActiveDirective, resolveStrategyFamilyForProfile } = await import(
    "../../lib/meta-brain/adapter"
  );
  const { getSkipReasonsSummary } = await import("../../lib/skip-tracker");

  const summaries = await getPortfolioSummaries();
  const liveAgents = await db
    .select({
      id: agentsTable.id,
      name: agentsTable.name,
      profileId: agentsTable.profileId,
      isActive: agentsTable.isActive,
      archivedAt: agentsTable.archivedAt,
    })
    .from(agentsTable);

  // peakValue is stored on paper_portfolios but not exposed by
  // PortfolioSummary; pull it directly so each family card can render
  // its high-water-mark drawdown.
  const peakRows = await db
    .select({
      agentId: paperPortfoliosTable.agentId,
      peakValue: paperPortfoliosTable.peakValue,
    })
    .from(paperPortfoliosTable);
  const peakByAgentId = new Map(peakRows.map((r) => [r.agentId, r.peakValue ?? 0]));

  let retirementSnapshot: Awaited<ReturnType<typeof getRetirementSnapshot>> | null = null;
  try {
    retirementSnapshot = await getRetirementSnapshot();
  } catch {
    retirementSnapshot = null;
  }

  // Skip-tracker breakdown is keyed by reason → { byAgent: [{agentName,count}] }.
  // We total per agent name once so each family can sum its members'
  // skip counts in O(members) instead of re-walking the buckets.
  let skipsByAgent = new Map<string, number>();
  try {
    const skipSummary = await getSkipReasonsSummary(60 * 60 * 1000);
    for (const reasonRow of skipSummary.byReason ?? []) {
      for (const a of reasonRow.byAgent ?? []) {
        skipsByAgent.set(
          a.agentName,
          (skipsByAgent.get(a.agentName) ?? 0) + a.count,
        );
      }
    }
  } catch {
    skipsByAgent = new Map();
  }

  const directive = getActiveDirective();

  const families = await Promise.all(listExecutingProfileIds().map(async (profileId) => {
    const profile = getAgentProfile(profileId);
    const familyKey = resolveStrategyFamilyForProfile(profileId);
    const memberAgents = liveAgents.filter(
      (a) =>
        a.profileId === profileId &&
        a.isActive === true &&
        a.archivedAt === null,
    );
    const memberAgentIds = memberAgents.map((a) => a.id);
    const memberSummaries = summaries.filter((s) =>
      memberAgentIds.includes(s.agentId),
    );

    const equity = memberSummaries.reduce((sum, s) => sum + s.totalValue, 0);
    const startingCapital = memberSummaries.reduce(
      (sum, s) => sum + s.startingCapital,
      0,
    );
    const peakValue = memberSummaries.reduce(
      (sum, s) => sum + (peakByAgentId.get(s.agentId) ?? s.totalValue),
      0,
    );
    const realizedPnl = equity - startingCapital;
    const totalTrades = memberSummaries.reduce(
      (sum, s) => sum + s.totalTrades,
      0,
    );
    const winningTrades = memberSummaries.reduce(
      (sum, s) => sum + s.winningTrades,
      0,
    );
    const losingTrades = memberSummaries.reduce(
      (sum, s) => sum + s.losingTrades,
      0,
    );
    const openPositions = memberSummaries.reduce(
      (sum, s) => sum + s.openPositions.length,
      0,
    );
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

    // Drawdown from current peak. We do not yet persist all-time max
    // drawdown so this is the operational "how far below the high-
    // water mark are we right now" figure the dashboard surfaces.
    const maxDrawdown =
      peakValue > 0 ? Math.max(0, (peakValue - equity) / peakValue) * 100 : 0;

    // Cost-aware directional accuracy from the nightly retirement
    // snapshot. One candidate per profile_id, so this is just a
    // direct lookup.
    const retirement =
      retirementSnapshot?.candidates?.find?.(
        (c: { profile_id: string }) => c.profile_id === profileId,
      ) ?? null;
    const cada = retirement?.directional_accuracy ?? null;
    const costAwareSharpe = retirement?.cost_aware_sharpe ?? null;

    // Abstain rate = skips / (skips + trades) over the last hour
    // window the skip-tracker exposes. Family-level by summing each
    // member agent's skip count.
    //
    // Task #532 / C-6 — when the brain is offline the orchestrator
    // emits `quant_disabled` abstains *before* the gate-level skip
    // tracker sees them, so `skipsByAgent` reads 0 even when 100 % of
    // recent predictions are abstains. Recompute from the predictions
    // table directly so the family card cannot read "Abstain 0 %"
    // while the brain is dark. We use a 1-hour window to match the
    // skip-tracker's existing horizon.
    const familySkips = memberAgents.reduce(
      (sum, a) => sum + (skipsByAgent.get(a.name) ?? 0),
      0,
    );
    let predAbstainCount = 0;
    let predTotalCount = 0;
    if (memberAgentIds.length > 0) {
      try {
        const predRow = await db.execute(sql`
          SELECT
            COUNT(*)::int                                                                AS total,
            SUM(CASE WHEN COALESCE(reasoning,'') ILIKE '%quant_disabled%'
                       OR COALESCE(reasoning,'') ILIKE '%quant_abstain_%'
                     THEN 1 ELSE 0 END)::int                                             AS abstains
          FROM ${predictionsTable}
          WHERE agent_id IN (${sql.join(memberAgentIds.map((id) => sql`${id}`), sql`, `)})
            AND created_at >= NOW() - INTERVAL '1 hour'
        `);
        const r = (predRow as unknown as { rows?: Array<{ total: number; abstains: number }> }).rows?.[0];
        if (r) {
          predTotalCount = Number(r.total ?? 0);
          predAbstainCount = Number(r.abstains ?? 0);
        }
      } catch {
        predTotalCount = 0;
        predAbstainCount = 0;
      }
    }
    const recentTradeCount = memberSummaries.reduce(
      (sum, s) => sum + s.recentTrades.length,
      0,
    );
    // Take the maximum of the two abstain measurements: the gate-level
    // skip tracker AND the orchestrator-level abstain rows in
    // `predictions`. Either signal alone has blind spots; the maximum
    // is the honest "what fraction of decisions did this family
    // refuse to author in the last hour".
    const skipBased = familySkips + recentTradeCount > 0
      ? (familySkips / (familySkips + recentTradeCount)) * 100
      : 0;
    const predBased = predTotalCount > 0
      ? (predAbstainCount / predTotalCount) * 100
      : 0;
    const abstainRate = Math.max(skipBased, predBased);
    const abstainCount = Math.max(familySkips, predAbstainCount);

    // Trust multiplier from the active meta-brain directive.
    const trustMultiplier = directive.trust_multiplier?.[familyKey] ?? 1;

    // Status pill enum the dashboard renders. Order of precedence:
    //   suppressed > quarantined > cautious > active.
    const isSuppressed =
      directive.suppress_signal === true ||
      (directive.suppressed_families ?? []).includes(familyKey);
    const isQuarantined = Boolean(
      retirement?.triggered_by?.length && retirement?.auto_flipped,
    );
    let statusPill: "active" | "cautious" | "suppressed" | "quarantined" =
      "active";
    if (isSuppressed) statusPill = "suppressed";
    else if (isQuarantined) statusPill = "quarantined";
    else if (trustMultiplier < 1) statusPill = "cautious";

    return {
      profileId,
      displayName: profile.display_name,
      thesis: profile.thesis,
      strategyFamily: profile.strategy_family,
      preferredRegimes: profile.preferred_regimes,
      blockedRegimes: profile.blocked_regimes,
      memberAgentIds,
      memberCount: memberAgentIds.length,
      equity,
      startingCapital,
      peakValue,
      realizedPnl,
      realizedPnlPct:
        startingCapital > 0 ? (realizedPnl / startingCapital) * 100 : 0,
      maxDrawdown,
      totalTrades,
      winningTrades,
      losingTrades,
      openPositions,
      winRate,
      costAwareDirectionalAccuracy: cada,
      costAwareSharpe,
      abstainRate,
      abstainCount,
      trustMultiplier,
      statusPill,
      retirement: retirement ?? null,
    };
  }));

  res.json({ families });
});

// Task #512 — per-family per-coin drill-down. Returns one row per
// (coin) for the supplied executor profile, joining open positions
// and recent closed trades from the underlying agents. The dashboard
// renders this as a table when an operator clicks into a family card.
router.get(
  "/crypto/agents/families/:profileId/coins",
  async (req, res): Promise<void> => {
    const profileId = String(req.params.profileId || "");
    const { listExecutingProfileIds } = await import("../../lib/agents-registry");
    if (!listExecutingProfileIds().includes(profileId)) {
      res.status(404).json({ error: "Unknown executor profile_id" });
      return;
    }
    const { getPortfolioSummaries } = await import("../../lib/paper-trader");
    const {
      getActiveDirective,
      resolveStrategyFamilyForProfile,
      isFamilySuppressed,
    } = await import("../../lib/meta-brain/adapter");
    const { getSkipsForReason } = await import("../../lib/skip-tracker");
    const { fetchCoinPrices } = await import("../../lib/coins");

    const summaries = await getPortfolioSummaries();
    const memberAgents = await db
      .select({ id: agentsTable.id, name: agentsTable.name })
      .from(agentsTable)
      .where(
        and(
          eq(agentsTable.profileId, profileId),
          eq(agentsTable.isActive, true),
          isNull(agentsTable.archivedAt),
          sql`${agentsTable.profileId} IS DISTINCT FROM 'legacy_archived'`,
        ),
      );
    const memberIds = new Set(memberAgents.map((a) => a.id));
    const memberNames = new Set(memberAgents.map((a) => a.name));
    const memberSummaries = summaries.filter((s) => memberIds.has(s.agentId));

    // Live prices keyed by coinId — used to derive the per-coin
    // benchmark-relative behavior (executor return vs. buy-and-hold from
    // the family's first entry on that coin to right now).
    const livePriceById = new Map<string, number>();
    try {
      const live = await fetchCoinPrices();
      for (const p of live) {
        if (p.currentPrice > 0) livePriceById.set(p.id, p.currentPrice);
      }
    } catch {
      // upstream price feed offline → benchmark fields will fall back to
      // the most recent trade price.
    }

    const familyKey = resolveStrategyFamilyForProfile(profileId);
    const directive = getActiveDirective();
    const trustMultiplier = directive.trust_multiplier?.[familyKey] ?? 1;

    // Recent decisions (predictions) + recent trades for the family,
    // joined into per-coin buckets below. We cap at the most recent 200
    // rows each so the response stays bounded.
    const memberIdList = [...memberIds];
    let recentPredictions: Array<{
      id: number;
      agentId: number;
      agentName: string;
      coinId: string;
      coinName: string;
      direction: string;
      confidence: number;
      outcome: string;
      createdAt: Date;
      timeframe: string;
    }> = [];
    let recentTradesByMember: Array<{
      id: number;
      agentId: number;
      agentName: string;
      coinId: string;
      coinName: string;
      action: string;
      entryPrice: number;
      exitPrice: number | null;
      positionSize: number;
      pnl: number | null;
      pnlPercent: number | null;
      status: string;
      createdAt: Date;
    }> = [];
    if (memberIdList.length > 0) {
      recentPredictions = await db
        .select({
          id: predictionsTable.id,
          agentId: predictionsTable.agentId,
          agentName: predictionsTable.agentName,
          coinId: predictionsTable.coinId,
          coinName: predictionsTable.coinName,
          direction: predictionsTable.direction,
          confidence: predictionsTable.confidence,
          outcome: predictionsTable.outcome,
          createdAt: predictionsTable.createdAt,
          timeframe: predictionsTable.timeframe,
        })
        .from(predictionsTable)
        .where(inArray(predictionsTable.agentId, memberIdList))
        .orderBy(desc(predictionsTable.createdAt))
        .limit(200);
      recentTradesByMember = await db
        .select({
          id: paperTradesTable.id,
          agentId: paperTradesTable.agentId,
          agentName: paperTradesTable.agentName,
          coinId: paperTradesTable.coinId,
          coinName: paperTradesTable.coinName,
          action: paperTradesTable.action,
          entryPrice: paperTradesTable.entryPrice,
          exitPrice: paperTradesTable.exitPrice,
          positionSize: paperTradesTable.positionSize,
          pnl: paperTradesTable.pnl,
          pnlPercent: paperTradesTable.pnlPercent,
          status: paperTradesTable.status,
          createdAt: paperTradesTable.createdAt,
        })
        .from(paperTradesTable)
        .where(inArray(paperTradesTable.agentId, memberIdList))
        .orderBy(desc(paperTradesTable.createdAt))
        .limit(200);
    }

    // Pooled-fallback / EV-below-cost skip events over the last 24h.
    // We bucket per coin so the drill-down can show "X fallback skips
    // on BTC in the last day".
    const fallbackByCoin = new Map<string, number>();
    try {
      const events = await getSkipsForReason("quant_ev_below_costs", 24 * 60 * 60 * 1000);
      for (const e of events) {
        if (!e.coinId || !memberNames.has(e.agentName)) continue;
        fallbackByCoin.set(e.coinId, (fallbackByCoin.get(e.coinId) ?? 0) + 1);
      }
    } catch {
      // skip-tracker offline → leave fallbackUsage at 0 per coin.
    }

    type CoinAgg = {
      coinId: string;
      coinName: string;
      openPositions: number;
      openNotional: number;
      unrealizedPnl: number;
      closedTrades: number;
      winningTrades: number;
      realizedPnl: number;
      drawdown: number;
      // operational metrics
      predictionCount: number;
      correctPredictions: number;
      recentAccuracy: number;
      fallbackUsage: number;
      suppressionState: "active" | "suppressed";
      trustMultiplier: number;
      // benchmark-relative behavior — buy-and-hold the coin from the
      // family's first entry on it through to the live price, compared
      // to the family's actual capital-weighted return on the same coin.
      firstEntryAt: Date | null;
      firstEntryPrice: number;
      latestPrice: number;
      totalCostBasis: number;
      benchmarkBuyHoldPct: number | null;
      executorReturnPct: number | null;
      vsBenchmarkPct: number | null;
      benchmarkRelative: "outperforming" | "tracking" | "underperforming" | "no_data";
      recentDecisions: typeof recentPredictions;
      recentTrades: typeof recentTradesByMember;
    };
    const byCoin = new Map<string, CoinAgg>();
    const ensure = (coinId: string, coinName: string): CoinAgg => {
      let row = byCoin.get(coinId);
      if (!row) {
        // Suppression is timeframe-specific in meta-brain; "1h" is the
        // default executor cadence so we report that as the canonical
        // per-coin state. Coins that are paused on any executor TF
        // surface as "suppressed".
        const suppressed =
          isFamilySuppressed(familyKey, coinId, "1h") ||
          isFamilySuppressed(familyKey, coinId, "15m") ||
          isFamilySuppressed(familyKey, coinId, "5m");
        row = {
          coinId,
          coinName,
          openPositions: 0,
          openNotional: 0,
          unrealizedPnl: 0,
          closedTrades: 0,
          winningTrades: 0,
          realizedPnl: 0,
          drawdown: 0,
          predictionCount: 0,
          correctPredictions: 0,
          recentAccuracy: 0,
          fallbackUsage: fallbackByCoin.get(coinId) ?? 0,
          suppressionState: suppressed ? "suppressed" : "active",
          trustMultiplier,
          firstEntryAt: null,
          firstEntryPrice: 0,
          latestPrice: livePriceById.get(coinId) ?? 0,
          totalCostBasis: 0,
          benchmarkBuyHoldPct: null,
          executorReturnPct: null,
          vsBenchmarkPct: null,
          benchmarkRelative: "no_data",
          recentDecisions: [],
          recentTrades: [],
        };
        byCoin.set(coinId, row);
      }
      return row;
    };

    for (const s of memberSummaries) {
      for (const op of s.openPositions) {
        const row = ensure(op.coinId, op.coinName);
        row.openPositions++;
        row.openNotional += op.positionSize;
        row.unrealizedPnl += op.unrealizedPnl;
        row.totalCostBasis += op.positionSize;
        if (row.latestPrice <= 0) row.latestPrice = op.entryPrice;
      }
      for (const t of s.recentTrades) {
        const row = ensure(t.coinId, t.coinName);
        row.closedTrades++;
        if ((t.pnl ?? 0) > 0) row.winningTrades++;
        row.realizedPnl += t.pnl ?? 0;
        if (row.latestPrice <= 0 && (t.exitPrice ?? t.entryPrice) > 0) {
          row.latestPrice = t.exitPrice ?? t.entryPrice;
        }
      }
    }

    // First-entry walk over the full member trade history (recentTradesByMember
    // is sorted DESC by createdAt, so the LAST element we visit is the
    // earliest entry on each coin). totalCostBasis also picks up closed
    // trades' notional so executorReturnPct uses real capital deployed.
    for (const t of recentTradesByMember) {
      const row = ensure(t.coinId, t.coinName);
      if (t.status === "closed" || row.totalCostBasis === 0) {
        row.totalCostBasis += t.positionSize;
      }
      if (
        t.entryPrice > 0 &&
        (row.firstEntryAt === null || t.createdAt < row.firstEntryAt)
      ) {
        row.firstEntryAt = t.createdAt;
        row.firstEntryPrice = t.entryPrice;
      }
    }

    for (const p of recentPredictions) {
      const row = ensure(p.coinId, p.coinName);
      row.predictionCount++;
      if (p.outcome === "correct") row.correctPredictions++;
      if (row.recentDecisions.length < 25) row.recentDecisions.push(p);
    }
    for (const t of recentTradesByMember) {
      const row = ensure(t.coinId, t.coinName);
      if (row.recentTrades.length < 25) row.recentTrades.push(t);
    }

    // Drawdown per coin: open notional vs current unrealized loss.
    // For closed-only coins, drawdown is min(realizedPnl,0) / |notional|.
    for (const row of byCoin.values()) {
      const denom = row.openNotional > 0 ? row.openNotional : Math.abs(row.realizedPnl) || 1;
      row.drawdown = row.unrealizedPnl < 0 ? Math.abs(row.unrealizedPnl) / denom * 100 : 0;
      row.recentAccuracy =
        row.predictionCount > 0
          ? (row.correctPredictions / row.predictionCount) * 100
          : 0;

      // Benchmark-relative behavior: compare the family's actual return
      // on this coin to "what would have happened if you'd just bought
      // the coin at our first entry and held to now".
      if (
        row.firstEntryPrice > 0 &&
        row.latestPrice > 0 &&
        row.totalCostBasis > 0
      ) {
        row.benchmarkBuyHoldPct =
          ((row.latestPrice - row.firstEntryPrice) / row.firstEntryPrice) * 100;
        row.executorReturnPct =
          ((row.realizedPnl + row.unrealizedPnl) / row.totalCostBasis) * 100;
        row.vsBenchmarkPct =
          row.executorReturnPct - row.benchmarkBuyHoldPct;
        // ±50 bps band around buy-and-hold counts as "tracking"; outside
        // that, the family is meaningfully diverging from the benchmark.
        if (row.vsBenchmarkPct > 0.5) {
          row.benchmarkRelative = "outperforming";
        } else if (row.vsBenchmarkPct < -0.5) {
          row.benchmarkRelative = "underperforming";
        } else {
          row.benchmarkRelative = "tracking";
        }
      }
    }

    const coins = Array.from(byCoin.values())
      .map((c) => ({
        ...c,
        recentDecisions: c.recentDecisions.map((d) => ({
          ...d,
          createdAt: d.createdAt.toISOString(),
        })),
        recentTrades: c.recentTrades.map((t) => ({
          ...t,
          createdAt: t.createdAt.toISOString(),
        })),
      }))
      .sort(
        (a, b) =>
          b.realizedPnl + b.unrealizedPnl - (a.realizedPnl + a.unrealizedPnl),
      );
    res.json({
      profileId,
      strategyFamily: familyKey,
      trustMultiplier,
      coins,
    });
  },
);

// Task #512 — Archived Agents page. Surfaces every legacy row that the
// boot archive sweep flipped to `archivedAt != null OR
// profile_id='legacy_archived'`. History (predictions, trades) stays
// queryable through the existing /crypto/agents/:id endpoint; this
// route only enumerates the rows so the hidden `/agents/archived`
// page can list them.
router.get("/crypto/agents/archived", async (_req, res): Promise<void> => {
  const { paperPortfoliosTable } = await import("@workspace/db");
  const rows = await db
    .select({
      id: agentsTable.id,
      name: agentsTable.name,
      personality: agentsTable.personality,
      profileId: agentsTable.profileId,
      isActive: agentsTable.isActive,
      archivedAt: agentsTable.archivedAt,
      score: agentsTable.score,
      totalPredictions: agentsTable.totalPredictions,
      correctPredictions: agentsTable.correctPredictions,
      wrongPredictions: agentsTable.wrongPredictions,
      createdAt: agentsTable.createdAt,
      updatedAt: agentsTable.updatedAt,
    })
    .from(agentsTable);

  // Pull paper-portfolio history per agent so we can surface lifetime
  // P&L, drawdown, and trade count alongside the archived row.
  const portfolios = await db
    .select({
      agentId: paperPortfoliosTable.agentId,
      totalValue: paperPortfoliosTable.totalValue,
      peakValue: paperPortfoliosTable.peakValue,
      totalTrades: paperPortfoliosTable.totalTrades,
      winningTrades: paperPortfoliosTable.winningTrades,
      losingTrades: paperPortfoliosTable.losingTrades,
      updatedAt: paperPortfoliosTable.updatedAt,
    })
    .from(paperPortfoliosTable);
  const portfolioByAgentId = new Map(portfolios.map((p) => [p.agentId, p]));

  // The legacy 15-personality bots all carry an `personality` blurb
  // describing their original strategy ("Aggressive momentum trader",
  // "Volume-focused analyst", ...). That blurb IS the legacy type
  // label the dashboard surfaces — we expose it on a stable
  // `legacyType` field so the frontend doesn't have to guess.
  const archived = rows
    .filter(
      (r) => r.archivedAt !== null || r.profileId === "legacy_archived",
    )
    .map((r) => {
      const p = portfolioByAgentId.get(r.id) ?? null;
      const startingCapital = 1000; // historical seed for legacy bots
      const totalValue = p?.totalValue ?? startingCapital;
      const peakValue = p?.peakValue ?? startingCapital;
      const lifetimePnl = totalValue - startingCapital;
      const lifetimePnlPct = (lifetimePnl / startingCapital) * 100;
      const maxDrawdown =
        peakValue > 0
          ? Math.max(0, (peakValue - totalValue) / peakValue) * 100
          : 0;
      const tradeCount = p?.totalTrades ?? 0;
      // Last active = most recent update we observed on either the
      // agent row or its paper-portfolio row. Falls back to created.
      const lastActiveCandidates: Array<Date | null | undefined> = [
        r.updatedAt,
        p?.updatedAt,
      ];
      const lastActiveDate = lastActiveCandidates
        .filter((d): d is Date => d instanceof Date)
        .sort((a, b) => b.getTime() - a.getTime())[0] ?? r.createdAt;

      return {
        id: r.id,
        name: r.name,
        legacyType: r.personality,
        personality: r.personality,
        profileId: r.profileId,
        isActive: r.isActive,
        archivedAt: r.archivedAt ? r.archivedAt.toISOString() : null,
        archivedOn: r.archivedAt ? r.archivedAt.toISOString() : null,
        score: r.score,
        totalPredictions: r.totalPredictions,
        correctPredictions: r.correctPredictions,
        wrongPredictions: r.wrongPredictions,
        createdAt: r.createdAt ? r.createdAt.toISOString() : null,
        lastActiveAt: lastActiveDate ? lastActiveDate.toISOString() : null,
        lifetimePnl,
        lifetimePnlPct,
        maxDrawdown,
        tradeCount,
        winningTrades: p?.winningTrades ?? 0,
        losingTrades: p?.losingTrades ?? 0,
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
  res.json({ archived });
});

// Task #512 + #518 — activity banner data source. Returns last-hour
// trade counts SPLIT by source (executor vs Strategy-Lab baseline) so
// the dashboard does not conflate passive baseline rebalances with
// quant-brain executor decisions. The audit in Task #518 found the
// previous combined `tradesLastHour` was inflated by ~60 baseline
// position flips that operators read as "the brain is working" when
// the brain was actually in `offline_no_model` abstain. The split
// makes that situation impossible to misread.
//
// `executorTradesLastHour` counts trades from agents whose registry
// profile is one of the deterministic executors (NOT
// `baseline_reference`, NOT `legacy_archived`).
// `baselineTradesLastHour` counts trades from `baseline_reference`
// agents only. Both numbers exclude archived agents entirely.
router.get("/crypto/dashboard-activity", async (_req, res): Promise<void> => {
  const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);

  const liveAgents = await db
    .select({
      id: agentsTable.id,
      profileId: agentsTable.profileId,
      archivedAt: agentsTable.archivedAt,
    })
    .from(agentsTable);
  const baselineAgentIds = new Set(
    liveAgents
      .filter((a) => a.archivedAt === null && a.profileId === "baseline_reference")
      .map((a) => a.id),
  );
  const executorAgentIds = new Set(
    liveAgents
      .filter(
        (a) =>
          a.archivedAt === null &&
          a.profileId !== "baseline_reference" &&
          a.profileId !== "legacy_archived",
      )
      .map((a) => a.id),
  );

  const recent = await db
    .select({
      agentId: paperTradesTable.agentId,
      createdAt: paperTradesTable.createdAt,
    })
    .from(paperTradesTable)
    .where(gte(paperTradesTable.createdAt, oneHourAgo));

  let executorTradesLastHour = 0;
  let baselineTradesLastHour = 0;
  let lastExecutorAt: Date | null = null;
  let lastBaselineAt: Date | null = null;
  for (const r of recent) {
    if (executorAgentIds.has(r.agentId)) {
      executorTradesLastHour += 1;
      if (!lastExecutorAt || (r.createdAt && r.createdAt > lastExecutorAt)) {
        lastExecutorAt = r.createdAt ?? null;
      }
    } else if (baselineAgentIds.has(r.agentId)) {
      baselineTradesLastHour += 1;
      if (!lastBaselineAt || (r.createdAt && r.createdAt > lastBaselineAt)) {
        lastBaselineAt = r.createdAt ?? null;
      }
    }
  }

  // For the case where the executor has not traded in the last hour
  // but did trade earlier (e.g. brain went offline 90 min ago), still
  // surface its most recent timestamp so the banner can show "last
  // executor decision 2h ago" instead of "—".
  if (!lastExecutorAt && executorAgentIds.size > 0) {
    const latestExec = await db
      .select({ createdAt: paperTradesTable.createdAt })
      .from(paperTradesTable)
      .where(inArray(paperTradesTable.agentId, Array.from(executorAgentIds)))
      .orderBy(desc(paperTradesTable.createdAt))
      .limit(1);
    lastExecutorAt = latestExec[0]?.createdAt ?? null;
  }
  if (!lastBaselineAt && baselineAgentIds.size > 0) {
    const latestBase = await db
      .select({ createdAt: paperTradesTable.createdAt })
      .from(paperTradesTable)
      .where(inArray(paperTradesTable.agentId, Array.from(baselineAgentIds)))
      .orderBy(desc(paperTradesTable.createdAt))
      .limit(1);
    lastBaselineAt = latestBase[0]?.createdAt ?? null;
  }

  // Task #532 / C-1b — surface `lastValidQuantDecisionAt`: the most
  // recent prediction whose `reasoning` is *not* a quant abstain. The
  // executor-trade timestamp alone hides that the brain produced no
  // actual non-abstain prediction in the window — the activity strip
  // could read "last exec 4h ago" while the last *valid quant
  // decision* is days back. Surfacing both makes that gap visible.
  let lastValidQuantDecisionAt: string | null = null;
  try {
    const validRow = await db.execute(sql`
      SELECT MAX(created_at) AS ts
      FROM ${predictionsTable}
      WHERE COALESCE(reasoning,'') NOT ILIKE '%quant_disabled%'
        AND COALESCE(reasoning,'') NOT ILIKE '%quant_abstain_%'
        AND direction <> 'stable'
    `);
    const tsCell = (validRow as unknown as { rows?: Array<{ ts: Date | string | null }> }).rows?.[0]?.ts ?? null;
    if (tsCell) {
      lastValidQuantDecisionAt = tsCell instanceof Date ? tsCell.toISOString() : new Date(String(tsCell)).toISOString();
    }
  } catch {
    lastValidQuantDecisionAt = null;
  }

  res.json({
    executorTradesLastHour,
    baselineTradesLastHour,
    lastExecutorTradeAt: lastExecutorAt ? lastExecutorAt.toISOString() : null,
    lastBaselineTradeAt: lastBaselineAt ? lastBaselineAt.toISOString() : null,
    lastValidQuantDecisionAt,
    // Back-compat: old clients that still poll the v1 fields keep
    // working. Combined total + most-recent timestamp across both
    // sources. Removable once the dashboard ships the v2 banner.
    tradesLastHour: executorTradesLastHour + baselineTradesLastHour,
    lastTradeAt:
      lastExecutorAt && lastBaselineAt
        ? (lastExecutorAt > lastBaselineAt ? lastExecutorAt : lastBaselineAt).toISOString()
        : (lastExecutorAt ?? lastBaselineAt)?.toISOString() ?? null,
  });
});

async function performFullReset(): Promise<void> {
  const { paperPositionsTable, paperTradesTable, paperPortfoliosTable } = await import("@workspace/db");
  await db.delete(paperPositionsTable);
  await db.delete(paperTradesTable);
  await db.delete(paperPortfoliosTable);
  await db.delete(predictionsTable);
  const existingAgents = await db.select().from(agentsTable);
  const existingNames = existingAgents.map(a => a.name);
  for (const persona of DETERMINISTIC_AGENTS) {
    if (!existingNames.includes(persona.name)) {
      await db.insert(agentsTable).values({
        name: persona.name,
        personality: persona.personality,
        score: 100,
        totalPredictions: 0,
        correctPredictions: 0,
        wrongPredictions: 0,
        streak: 0,
        streakType: "none",
        status: "active",
        profileId: persona.profileId,
      });
    } else {
      await db.update(agentsTable)
        .set({ profileId: persona.profileId })
        .where(eq(agentsTable.name, persona.name));
    }
  }
  await db.update(agentsTable).set({
    score: 100,
    totalPredictions: 0,
    correctPredictions: 0,
    wrongPredictions: 0,
    streak: 0,
    streakType: "none",
    status: "active",
  });
  const { initializePaperPortfolios } = await import("../../lib/paper-trader");
  await initializePaperPortfolios();
  const { seedHistoricalPriceData, resetCycleCount, forceUnlockCycle } = await import("../../lib/monitor");
  forceUnlockCycle();
  await seedHistoricalPriceData();
  const { invalidateCalibrationCache } = await import("../../lib/confidence-calibrator");
  invalidateCalibrationCache();
  const { clearRegimeCache } = await import("../../lib/regime-detector");
  await clearRegimeCache();
  const { clearContagionCache } = await import("../../lib/contagion-detector");
  clearContagionCache();
  resetCycleCount();
  await db.update(monitoringStateTable).set({ cycleCount: 0, lastCycleAt: null, isRunning: false }).where(eq(monitoringStateTable.id, 1));
}

router.post("/crypto/admin/reset-bots", async (req, res): Promise<void> => {
  const adminKey = req.headers["x-admin-key"];
  if (!adminKey || !process.env.ADMIN_RESET_KEY || adminKey !== process.env.ADMIN_RESET_KEY) {
    res.status(403).json({ error: "Forbidden: invalid admin reset key" });
    return;
  }
  const confirm = (req.body && (req.body as { confirm?: string }).confirm) || "";
  if (confirm !== "RESET BOTS") {
    res.status(400).json({ error: "Confirmation required. POST { confirm: \"RESET BOTS\" } to wipe all bots." });
    return;
  }
  try {
    await performFullReset();
    res.json({ success: true, message: "All bots reset to $1,000. History wiped. Fresh start." });
  } catch (err) {
    res.status(500).json({ error: "Reset failed", details: String(err) });
  }
});

router.post("/crypto/admin/reset", async (req, res): Promise<void> => {
  const adminKey = req.headers["x-admin-key"];
  if (!adminKey || !process.env.ADMIN_RESET_KEY || adminKey !== process.env.ADMIN_RESET_KEY) {
    res.status(403).json({ error: "Forbidden: invalid admin reset key" });
    return;
  }
  const { paperPositionsTable, paperTradesTable, paperPortfoliosTable } = await import("@workspace/db");
  await db.delete(paperPositionsTable);
  await db.delete(paperTradesTable);
  await db.delete(paperPortfoliosTable);
  await db.delete(predictionsTable);
  const existingAgents = await db.select().from(agentsTable);
  const existingNames = existingAgents.map(a => a.name);
  for (const persona of DETERMINISTIC_AGENTS) {
    if (!existingNames.includes(persona.name)) {
      await db.insert(agentsTable).values({
        name: persona.name,
        personality: persona.personality,
        score: 100,
        totalPredictions: 0,
        correctPredictions: 0,
        wrongPredictions: 0,
        streak: 0,
        streakType: "none",
        status: "active",
        preferredTimeframes: JSON.stringify([persona.timeframe]),
        profileId: persona.profileId,
      });
    } else {
      await db.update(agentsTable)
        .set({
          preferredTimeframes: JSON.stringify([persona.timeframe]),
          profileId: persona.profileId,
        })
        .where(eq(agentsTable.name, persona.name));
    }
  }
  await db.update(agentsTable).set({
    score: 100,
    totalPredictions: 0,
    correctPredictions: 0,
    wrongPredictions: 0,
    streak: 0,
    streakType: "none",
    status: "active",
  });
  const { initializePaperPortfolios } = await import("../../lib/paper-trader");
  await initializePaperPortfolios();
  const { seedHistoricalPriceData, resetCycleCount } = await import("../../lib/monitor");
  await seedHistoricalPriceData();
  const { invalidateCalibrationCache } = await import("../../lib/confidence-calibrator");
  invalidateCalibrationCache();
  const { clearRegimeCache } = await import("../../lib/regime-detector");
  await clearRegimeCache();
  const { clearContagionCache } = await import("../../lib/contagion-detector");
  clearContagionCache();
  resetCycleCount();
  await db.update(monitoringStateTable).set({ cycleCount: 0, lastCycleAt: null, isRunning: false }).where(eq(monitoringStateTable.id, 1));
  res.json({ success: true, message: "Full reset complete. All data cleared, caches purged, cycle count reset. 10 agents initialized fresh." });
});

router.get("/crypto/agent-specializations", async (req, res): Promise<void> => {
  try {
    const { getAllAgentSpecializations } = await import("../../lib/agent-specialization");
    const specializations = await getAllAgentSpecializations();
    res.json(specializations.map(({ allAffinities, ...rest }) => rest));
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch agent specializations" });
  }
});

// Task #444 — `/crypto/llm-bias-demote-rate` removed (LLM brain gone).
// The dashboard panel that consumed it is deleted in the same task.

router.get("/crypto/contagion-alerts", async (req, res): Promise<void> => {
  try {
    const { getActiveAlerts } = await import("../../lib/contagion-detector");
    const alerts = getActiveAlerts();
    res.json(alerts);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch contagion alerts" });
  }
});

router.get("/crypto/validation-report", async (req, res): Promise<void> => {
  try {
    const hours = parseInt(req.query.hours as string) || 24;
    const { generateValidationReport } = await import("../../lib/validation-framework");
    const report = await generateValidationReport(hours);
    res.json(report);
  } catch (err) {
    res.status(500).json({ error: "Failed to generate validation report" });
  }
});

router.get("/crypto/ablation-config", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const { getAblationConfig } = await import("../../lib/ablation-config");
  res.json(getAblationConfig());
});

router.post("/crypto/ablation-config", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const { updateAblationConfig } = await import("../../lib/ablation-config");
  const updated = updateAblationConfig(req.body);
  res.json(updated);
});

router.post("/crypto/ablation-reset", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const { resetAblationConfig } = await import("../../lib/ablation-config");
  const config = resetAblationConfig();
  res.json(config);
});

// Task #444 — `agent-evolution` (LLM-driven personality mutation) is gone.
// The two evolution endpoints stay as stable stubs so the admin and
// agent-detail pages keep loading; `POST /crypto/admin/evolve` returns
// 410 Gone, `GET /crypto/evolution-status` returns an empty cycle log.
router.post("/crypto/admin/evolve", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  res.status(410).json({
    error: "gone",
    message:
      "Task #444: agent-evolution (LLM-driven personality mutation) has been removed. " +
      "Quant brain is the sole authority; nothing to evolve.",
  });
});

router.get("/crypto/evolution-status", async (_req, res): Promise<void> => {
  res.json({
    currentGeneration: 1,
    lastEvolutionTime: 0,
    nextEvolutionEta: 0,
    tradesSinceLastEvolution: 0,
    tradeThreshold: 0,
    history: [],
    agentLineage: {},
    currentFitness: [],
  });
});

// ── Phase 4: Quant Shadow ──────────────────────────────────────────────
import { modelPredictionsTable } from "@workspace/db";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { judgeDirection, HORIZON_WEAK_DIRECTIONAL_FLOOR_PCT, HORIZON_WEAK_MIN_RESOLVED, ROUND_TRIP_COST_PERCENT } from "../../lib/trading-constants";

function loadShadowGate(): { min_resolved_per_timeframe: number; min_directional_accuracy_lift_over_llm: number; max_brier_score: number } {
  // Walk up from the current module until we find the workspace root (the
  // dir that contains shared/trading-frictions.json). Works for both the
  // tsx dev path and the bundled dist/ output.
  let dir = path.dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 8; i++) {
    const candidate = path.join(dir, "shared", "trading-frictions.json");
    try {
      const json = JSON.parse(readFileSync(candidate, "utf8"));
      return json.shadow_cutover_gate;
    } catch { /* keep walking */ }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error("trading-frictions.json not found");
}

interface ShadowMetricsRow {
  timeframe: string;
  totalResolved: number;
  modelDirectionalAccuracy: number;
  brierScore: number;
  meanConfidence: number;
}

// Quant-live calibration aggregator. Read-only over `model_predictions`.
// The legacy `llmDirectionalAccuracy` / `agreementShare` fields were
// dropped when the LLM stopped being a trade-decision authority (Task
// #255). The residual `llm_direction` / `llm_confidence` /
// `llm_prediction_id` columns themselves were dropped from the table in
// Task #506.
async function computeShadowMetrics(includePrior = false): Promise<ShadowMetricsRow[]> {
  const rows = await db.select().from(modelPredictionsTable);
  const byTf = new Map<string, typeof rows>();
  for (const r of rows) {
    if (r.outcome === "pending") continue;
    // Drop prior-only fallback rows from the headline scoreboard — they all
    // share the same Laplace-smoothed marginal distribution and would drag
    // accuracy / Brier toward the empirical prior.
    if (!includePrior && r.source === "prior") continue;
    const arr = byTf.get(r.timeframe) ?? [];
    arr.push(r); byTf.set(r.timeframe, arr);
  }
  const out: ShadowMetricsRow[] = [];
  for (const [tf, arr] of byTf) {
    const total = arr.length;
    if (total === 0) continue;
    let modelCorrect = 0;
    let brier = 0;
    let confSum = 0;
    for (const r of arr) {
      const realized = r.resolvedOutcomePct ?? 0;
      // Use the SAME canonical adjudication (neutral-zone band) the live
      // resolver uses — never re-derive direction from raw sign.
      const modelJudge = judgeDirection(r.modelDirection as "up"|"down"|"stable", realized, r.timeframe);
      if (modelJudge.correct) modelCorrect++;
      // 3-class Brier scored against the canonical realized class label.
      const realizedClass: "up" | "down" | "stable" =
        judgeDirection("stable", realized, r.timeframe).correct
          ? "stable"
          : realized > 0 ? "up" : "down";
      const pUp = r.probUp ?? 0, pDn = r.probDown ?? 0, pSt = r.probStable ?? 0;
      brier += (pUp - (realizedClass === "up" ? 1 : 0)) ** 2
             + (pDn - (realizedClass === "down" ? 1 : 0)) ** 2
             + (pSt - (realizedClass === "stable" ? 1 : 0)) ** 2;
      confSum += r.confidence ?? 0;
    }
    out.push({
      timeframe: tf,
      totalResolved: total,
      modelDirectionalAccuracy: modelCorrect / total,
      brierScore: brier / total,
      meanConfidence: confSum / total,
    });
  }
  out.sort((a, b) => a.timeframe.localeCompare(b.timeframe));
  return out;
}

// ── Live vs Backtest tracking ─────────────────────────────────────────
// Phase 3 produced models/backtest_report.json with per-coin expectancy,
// win-rate and trade-count baselines. Phase 4 tracks whether the live
// shadow brain's realized win-rate stays inside a confidence band around
// that baseline. If it drifts outside the band on enough samples we surface
// a "drift" warning so the user knows the live and offline distributions
// have diverged before any cutover.
interface BacktestBaseline {
  winRate: number;
  expectancyUsd: number;
  nTrades: number;
}
interface LiveVsBacktestRow {
  coinId: string;
  baseline: BacktestBaseline | null;
  liveResolved: number;
  liveWinRate: number;
  liveMinusBaseline: number;
  withinBand: boolean | null;   // null while we don't have enough samples
  status: "tracking" | "drift_high" | "drift_low" | "insufficient_data" | "no_baseline";
}

interface BacktestPerCoinPayload {
  metrics?: { win_rate?: unknown; expectancy_usd?: unknown; n_trades?: unknown };
}
interface BacktestRun { per_coin?: Record<string, BacktestPerCoinPayload> }
interface BacktestReport { runs?: BacktestRun[] }

function asNumber(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

function loadBacktestBaselines(): Record<string, BacktestBaseline> {
  let dir = path.dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 8; i++) {
    const candidate = path.join(dir, "artifacts", "ml-engine", "models", "backtest_report.json");
    if (existsSync(candidate)) {
      try {
        const raw = JSON.parse(readFileSync(candidate, "utf8")) as BacktestReport;
        const out: Record<string, BacktestBaseline> = {};
        const runs: BacktestRun[] = Array.isArray(raw.runs) ? raw.runs : [];
        for (const run of runs) {
          const perCoin: Record<string, BacktestPerCoinPayload> = run.per_coin ?? {};
          for (const [coinId, payload] of Object.entries(perCoin)) {
            const m = payload?.metrics ?? {};
            out[coinId] = {
              winRate: asNumber(m.win_rate),
              expectancyUsd: asNumber(m.expectancy_usd),
              nTrades: asNumber(m.n_trades),
            };
          }
        }
        return out;
      } catch { return {}; }
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return {};
}

const LIVE_VS_BACKTEST_BAND_PCT = 0.10; // ±10pp confidence band on win-rate
const LIVE_VS_BACKTEST_MIN_SAMPLES = 30;

async function computeLiveVsBacktest(): Promise<{ rows: LiveVsBacktestRow[]; band: number; minSamples: number }> {
  const baselines = loadBacktestBaselines();
  const rows = await db.select().from(modelPredictionsTable);
  const byCoin = new Map<string, typeof rows>();
  for (const r of rows) {
    if (r.outcome === "pending") continue;
    const arr = byCoin.get(r.coinId) ?? [];
    arr.push(r); byCoin.set(r.coinId, arr);
  }
  const out: LiveVsBacktestRow[] = [];
  const coinIds = new Set<string>([...byCoin.keys(), ...Object.keys(baselines)]);
  for (const coinId of coinIds) {
    const arr = byCoin.get(coinId) ?? [];
    let wins = 0;
    for (const r of arr) {
      const judge = judgeDirection(r.modelDirection as "up"|"down"|"stable",
        r.resolvedOutcomePct ?? 0, r.timeframe);
      if (judge.correct) wins++;
    }
    const total = arr.length;
    const liveWinRate = total > 0 ? wins / total : 0;
    const baseline = baselines[coinId] ?? null;
    let withinBand: boolean | null = null;
    let status: LiveVsBacktestRow["status"] = "no_baseline";
    let liveMinusBaseline = 0;
    if (baseline && baseline.nTrades > 0) {
      liveMinusBaseline = liveWinRate - baseline.winRate;
      if (total < LIVE_VS_BACKTEST_MIN_SAMPLES) {
        status = "insufficient_data";
      } else if (Math.abs(liveMinusBaseline) <= LIVE_VS_BACKTEST_BAND_PCT) {
        withinBand = true; status = "tracking";
      } else {
        withinBand = false;
        status = liveMinusBaseline > 0 ? "drift_high" : "drift_low";
      }
    } else if (total > 0) {
      status = "no_baseline";
    }
    out.push({ coinId, baseline, liveResolved: total, liveWinRate, liveMinusBaseline, withinBand, status });
  }
  out.sort((a, b) => a.coinId.localeCompare(b.coinId));
  return { rows: out, band: LIVE_VS_BACKTEST_BAND_PCT, minSamples: LIVE_VS_BACKTEST_MIN_SAMPLES };
}

router.get("/crypto/shadow/vs-backtest", async (_req, res): Promise<void> => {
  try {
    res.json(await computeLiveVsBacktest());
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

router.get("/crypto/shadow/metrics", async (req, res): Promise<void> => {
  try {
    const includePriorRaw = String((req.query as { includePrior?: string }).includePrior ?? "");
    const includePrior = includePriorRaw === "1" || includePriorRaw.toLowerCase() === "true";
    const metrics = await computeShadowMetrics(includePrior);
    const gate = loadShadowGate();
    // Task #532 / C-8 — surface a "live calibration sample" timestamp so
    // the QuantShadow page can render an honest staleness banner. The
    // audit found `model_predictions.MAX(created_at)` was 3 days stale
    // because the brain has been off, but the page rendered the rows
    // as if they were current. Returning `freshness.lastSampleAt` +
    // `freshness.ageMinutes` lets the UI surface "Last live sample 3d
    // ago — calibration is stale" without re-querying.
    let lastSampleAtIso: string | null = null;
    let ageMinutes: number | null = null;
    try {
      const row = await db.execute(sql`SELECT MAX(created_at) AS ts FROM ${modelPredictionsTable}`);
      const ts = (row as unknown as { rows?: Array<{ ts: Date | string | null }> }).rows?.[0]?.ts ?? null;
      if (ts) {
        const d = ts instanceof Date ? ts : new Date(String(ts));
        lastSampleAtIso = d.toISOString();
        ageMinutes = Math.round((Date.now() - d.getTime()) / 60_000);
      }
    } catch {
      lastSampleAtIso = null;
      ageMinutes = null;
    }
    const STALE_AFTER_MINUTES = 60;
    res.json({
      metrics,
      gate,
      priorFallback: { included: includePrior },
      freshness: {
        lastSampleAt: lastSampleAtIso,
        ageMinutes,
        staleAfterMinutes: STALE_AFTER_MINUTES,
        isStale: ageMinutes !== null ? ageMinutes > STALE_AFTER_MINUTES : true,
      },
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #255 / Quant Live Health repurpose — removed
// `GET /crypto/shadow/disagreements`. The "LLM vs Quant" comparison axis
// no longer exists once the LLM stopped authoring trade decisions, and
// the page that consumed this endpoint was reframed as quant calibration.
router.get("/crypto/shadow/disagreements", (_req, res): void => {
  res.status(410).json({
    error: "gone",
    message:
      "The LLM is no longer a trade-decision authority (Task #255). The shadow disagreements view was retired with the Quant Live Health page repurpose.",
  });
});

// ── Phase 5 — Brain control plane ──────────────────────────────────────
// The quant brain feature flag defaults OFF and persists in app_settings.
// Flipping it ON routes every prediction through the LightGBM ml-engine and
// keeps the removed LLM path silent. The auto-revert guard watches the live-
// vs-backtest tracker and disables the quant brain if drift_low fires on the
// majority of coins for several consecutive cycles.
router.get("/crypto/brain/state", async (_req, res): Promise<void> => {
  try {
    const state = await getBrainState();
    const revertLog = await getBrainRevertLog();
    res.json({
      ...state,
      autoRevert: {
        consecutiveDriftCycles: getAutoRevertCounter(),
        recentEvents: revertLog.slice(0, 10),
      },
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #406 — admin-gated brain flip. Distinct from the deleted fleet
// `/crypto/brain/toggle` (Task #255): that route used to swap the entire
// trade-decision authority between LLM and QUANT, which no longer makes
// sense now the LLM is advisory-only. This endpoint instead controls the
// `quant_brain_enabled` kill-switch that gates whether the QUANT brain
// authors trades or every cycle abstains. Operators flip it ON only after
// the verification gate has promoted at least one slice — the audit in
// docs/remediation/2026-04-24-full-system-remediation.md is explicit that
// enabling over a known-bad model set is a safety violation.
//
// Enable path: consult `hasPromotedSlice()` against the latest
// verification-history record. If no promoted slice, refuse with 409 and
// surface the verdict so the operator sees exactly which gate failed.
// Disable path: always allowed (kill-switching down is never gated).
//
// The auto-revert path (`brain-auto-revert.ts`) writes the same flag with
// `source: "auto_revert"` when drift_low fires; manual operator flips
// recorded here use `source: "manual"`. `QUANT_BRAIN_FORCE_OFF=1` still
// hard-blocks any enable.
// Factory keeps the gate-then-toggle policy testable in isolation. The
// admin-key check stays at the route layer so requireAdminApiKey can read
// process.env.ADMIN_API_KEY directly without being threaded through the
// handler factory.
const brainStatePostHandler = createBrainStatePostHandler();
router.post("/crypto/brain/state", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  await brainStatePostHandler(req, res);
});

/**
 * Task #518 — single source of truth for the dashboard's brain
 * status pill and the new "BRAIN OFFLINE" banner. Distinct from
 * `/brain/state` (which is the operator-facing kill-switch view):
 *
 *   - `online`            → quant brain enabled AND has produced at
 *                           least one non-abstain prediction in the
 *                           last 30 min.
 *   - `offline_no_model`  → quant brain enabled but every recent
 *                           prediction abstained with `no_model`
 *                           (ml-engine has no trained per-coin or
 *                           pooled model for the requested pair).
 *                           This is the "looks online but is silently
 *                           dead" failure the audit caught.
 *   - `offline_disabled`  → kill-switch off (operator flip OR
 *                           QUANT_BRAIN_FORCE_OFF=1 OR auto-revert).
 *
 * The recentAbstainReasons rollup is the same shape the
 * `/quant-abstain-reasons` panel uses but bucketed over 30 min so the
 * banner can show "127 no_model in last 30 min" without a second
 * round-trip.
 */
// Task #532 — extracted helper so `/best-pick` (and any future
// surface that needs to gate on "is the brain authoring decisions
// right now?") can ask the same question without duplicating the
// abstain-reason logic. Returns the same `state` discriminant the
// `/brain/runtime-status` route surfaces, plus the brain source so
// callers can write honest copy ("source: default" vs "source: env").
//
// Task #680 — the implementation now lives in
// `../../lib/brain-runtime-state.ts` so the prediction-journal
// scan can be unit-tested with a stubbed db. The route's outer
// behavior (`BrainRuntimeStatePayload` shape, `state` discriminant,
// `since = now − 30 min` window) is unchanged; the SELECT is
// paginated internally to bound memory under prediction-journal
// surges.

/**
 * Task #554 — dataset-refresher health for the dashboard.
 *
 * The `dataset-refresher` workflow (Task #540) writes two files to disk
 * under `artifacts/ml-engine/models/datasets/`:
 *   - `_freshness_status.json`   — per-tf last_success_at / last_error /
 *                                  next_due_at / cadence_hours
 *   - `_freshness_alerts.jsonl`  — append-only failure log
 *
 * Until this route existed the only way to know whether the refresher
 * was healthy was to SSH-grep those files. This endpoint mirrors the
 * `/brain/runtime-status` shape: it returns a per-tf health pill
 * (green/amber/red), the most recent failure messages, an `unread`
 * flag for alerts that arrived after the last successful tick, and a
 * top-level `state` discriminant the dashboard banner uses to decide
 * whether to render loud red chrome.
 *
 *   - red    → at least one timeframe is past `cadence_hours * 1.5`
 *              since its last success, OR there is at least one
 *              alert newer than that timeframe's last_success_at.
 *   - amber  → at least one timeframe is past its `next_due_at` but
 *              still inside the 1.5x grace window.
 *   - green  → every managed timeframe is within cadence and there
 *              are no unread alerts.
 *   - unknown → status file missing entirely (refresher has never
 *              ticked, or files were wiped).
 *
 * The route is intentionally tolerant: missing files, malformed JSON,
 * or unreadable lines collapse to `unknown` rather than 500ing — the
 * dashboard prefers an honest "n/a" pill to a hard crash.
 */
export type DatasetFreshnessHealth = "green" | "amber" | "red" | "unknown";
export interface DatasetFreshnessAlertEntry {
  at: string | null;
  timeframe: string | null;
  status: string | null;
  error: string | null;
  cadenceHours: number | null;
  unread: boolean;
  raw: string;
}
export interface DatasetFreshnessTimeframe {
  timeframe: string;
  health: DatasetFreshnessHealth;
  cadenceHours: number | null;
  lastSuccessAt: string | null;
  lastAttemptAt: string | null;
  lastError: string | null;
  lastStatus: string | null;
  nextDueAt: string | null;
  mtimeOfNewestSnapshot: string | null;
  ageSeconds: number | null;
  pastDueSeconds: number | null;
  unreadAlertCount: number;
}
export interface DatasetFreshnessRollup {
  state: DatasetFreshnessHealth;
  writtenAt: string | null;
  timeframes: DatasetFreshnessTimeframe[];
  pastDueTimeframes: string[];
  alerts: DatasetFreshnessAlertEntry[];
  totalAlerts: number;
  totalUnreadAlerts: number;
}

function _parseIsoOrNull(v: unknown): number | null {
  if (typeof v !== "string" || !v) return null;
  const t = Date.parse(v);
  return Number.isFinite(t) ? t : null;
}

/**
 * Pure dataset-freshness rollup: takes the parsed status object and the
 * already-parsed alert entries (newest-first) and emits the per-tf health
 * + top-level state. Disk I/O lives in the route handler; this function
 * is what the unit tests pin.
 *
 * Health rules (per tf):
 *   red   — no last_success, OR cadence-derived age is > 1.5×cadence
 *           past last success, OR there are unread failure alerts.
 *   amber — past `next_due_at` (writer's own schedule) but not yet red.
 *           Falls back to `age > cadence` when next_due_at is missing.
 *   green — otherwise.
 *
 * Top-level rollup: red wins, then amber, then green; "unknown" is
 * reserved for "status object missing or empty" and is set by the caller.
 */
export function computeDatasetFreshness(
  statusObj: Record<string, unknown> | null,
  alerts: DatasetFreshnessAlertEntry[],
  now: number,
): DatasetFreshnessRollup {
  const tfMap: Record<string, Record<string, unknown>> =
    statusObj &&
    typeof statusObj["timeframes"] === "object" &&
    statusObj["timeframes"] != null &&
    !Array.isArray(statusObj["timeframes"])
      ? (statusObj["timeframes"] as Record<string, Record<string, unknown>>)
      : {};
  const writtenAt =
    statusObj && typeof statusObj["written_at"] === "string"
      ? (statusObj["written_at"] as string)
      : null;

  const timeframes: DatasetFreshnessTimeframe[] = [];
  const orderedTfs = Object.keys(tfMap).sort((a, b) => a.localeCompare(b));
  for (const tf of orderedTfs) {
    const entry = tfMap[tf] ?? {};
    const cadenceRaw = entry["cadence_hours"];
    const cadenceHours =
      typeof cadenceRaw === "number" && Number.isFinite(cadenceRaw)
        ? cadenceRaw
        : null;
    const lastSuccessAt =
      typeof entry["last_success_at"] === "string"
        ? (entry["last_success_at"] as string)
        : null;
    const lastAttemptAt =
      typeof entry["last_attempt_at"] === "string"
        ? (entry["last_attempt_at"] as string)
        : null;
    const lastError =
      typeof entry["last_error"] === "string"
        ? (entry["last_error"] as string)
        : null;
    const lastStatus =
      typeof entry["last_status"] === "string"
        ? (entry["last_status"] as string)
        : null;
    const nextDueAt =
      typeof entry["next_due_at"] === "string"
        ? (entry["next_due_at"] as string)
        : null;
    const mtimeOfNewestSnapshot =
      typeof entry["mtime_of_newest_snapshot"] === "string"
        ? (entry["mtime_of_newest_snapshot"] as string)
        : null;

    const lastSuccessTs = _parseIsoOrNull(lastSuccessAt);
    const nextDueTs = _parseIsoOrNull(nextDueAt);
    const cadenceMs = cadenceHours != null ? cadenceHours * 3600 * 1000 : null;
    const ageMs = lastSuccessTs != null ? now - lastSuccessTs : null;
    const pastDueMs =
      nextDueTs != null
        ? now - nextDueTs
        : cadenceMs != null && ageMs != null
          ? ageMs - cadenceMs
          : null;

    let unreadCount = 0;
    for (const a of alerts) {
      if (a.timeframe !== tf) continue;
      const aTs = _parseIsoOrNull(a.at);
      if (aTs == null) continue;
      if (lastSuccessTs == null || aTs > lastSuccessTs) {
        a.unread = true;
        unreadCount += 1;
      }
    }

    let health: DatasetFreshnessHealth;
    if (cadenceMs == null || lastSuccessTs == null) {
      health = "red";
    } else if (ageMs != null && ageMs > cadenceMs * 1.5) {
      health = "red";
    } else if (unreadCount > 0) {
      health = "red";
    } else if (nextDueTs != null && now > nextDueTs) {
      health = "amber";
    } else if (
      nextDueTs == null &&
      ageMs != null &&
      cadenceMs != null &&
      ageMs > cadenceMs
    ) {
      health = "amber";
    } else {
      health = "green";
    }

    timeframes.push({
      timeframe: tf,
      health,
      cadenceHours,
      lastSuccessAt,
      lastAttemptAt,
      lastError,
      lastStatus,
      nextDueAt,
      mtimeOfNewestSnapshot,
      ageSeconds: ageMs != null ? Math.round(ageMs / 1000) : null,
      pastDueSeconds: pastDueMs != null ? Math.round(pastDueMs / 1000) : null,
      unreadAlertCount: unreadCount,
    });
  }

  let state: DatasetFreshnessHealth;
  if (timeframes.length === 0) {
    state = "unknown";
  } else if (timeframes.some((t) => t.health === "red")) {
    state = "red";
  } else if (timeframes.some((t) => t.health === "amber")) {
    state = "amber";
  } else {
    state = "green";
  }

  const totalUnreadAlerts = alerts.reduce(
    (n, a) => n + (a.unread ? 1 : 0),
    0,
  );
  const pastDueTimeframes = timeframes
    .filter((t) => t.health === "red" || t.health === "amber")
    .map((t) => t.timeframe);

  return {
    state,
    writtenAt,
    timeframes,
    pastDueTimeframes,
    alerts,
    totalAlerts: alerts.length,
    totalUnreadAlerts,
  };
}

router.get("/crypto/datasets/freshness", async (_req, res): Promise<void> => {
  try {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const { fileURLToPath } = await import("node:url");
    // Walk up from the current module until we find the workspace
    // root (the dir that contains `artifacts/ml-engine/models`). This
    // mirrors `loadShadowGate` and works for both the tsx dev path
    // (.../api-server/src/routes/crypto/index.ts) and the bundled
    // dist path (.../api-server/dist/index.mjs) — naive `../../../..`
    // resolves differently for those two and breaks the bundled
    // build silently (the file just appears "missing").
    let dir = path.dirname(fileURLToPath(import.meta.url));
    let datasetsDir: string | null = null;
    for (let i = 0; i < 8; i++) {
      const candidate = path.join(
        dir,
        "artifacts",
        "ml-engine",
        "models",
        "datasets",
      );
      if (fs.existsSync(candidate)) {
        datasetsDir = candidate;
        break;
      }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
    if (!datasetsDir) {
      // The datasets dir does not exist on this host (no ml-engine
      // checked out locally). Return the same "unknown" shape we
      // emit when the status file is missing so the dashboard pill
      // renders "n/a" instead of erroring.
      res.json({
        state: "unknown" as const,
        statusFileExists: false,
        alertsFileExists: false,
        statusReadError: null,
        writtenAt: null,
        timeframes: [] as DatasetFreshnessTimeframe[],
        pastDueTimeframes: [] as string[],
        alerts: [] as DatasetFreshnessAlertEntry[],
        totalAlerts: 0,
        totalUnreadAlerts: 0,
        fetchedAt: new Date().toISOString(),
      });
      return;
    }
    const statusPath = path.join(datasetsDir, "_freshness_status.json");
    const alertsPath = path.join(datasetsDir, "_freshness_alerts.jsonl");

    // ── Load status JSON ──────────────────────────────────────────
    let statusRaw: unknown = null;
    let statusFileExists = false;
    let statusReadError: string | null = null;
    try {
      const text = fs.readFileSync(statusPath, "utf8");
      statusFileExists = true;
      try {
        statusRaw = JSON.parse(text);
      } catch (e) {
        statusReadError = e instanceof Error ? e.message : String(e);
      }
    } catch {
      statusFileExists = false;
    }

    // ── Tail alerts JSONL (last 200 lines, newest first) ──────────
    // We tail the file so a long-running refresher with thousands of
    // alert lines does not blow up the response. 200 is plenty for
    // the operator panel; a deeper history would belong in a
    // dedicated diagnostics page.
    const ALERT_TAIL = 200;
    let alertLines: string[] = [];
    let alertsFileExists = false;
    try {
      const text = fs.readFileSync(alertsPath, "utf8");
      alertsFileExists = true;
      alertLines = text
        .split("\n")
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .slice(-ALERT_TAIL)
        .reverse();
    } catch {
      alertsFileExists = false;
    }

    const alerts: DatasetFreshnessAlertEntry[] = alertLines.map((line) => {
      let parsed: Record<string, unknown> | null = null;
      try {
        const obj = JSON.parse(line);
        if (obj && typeof obj === "object" && !Array.isArray(obj)) {
          parsed = obj as Record<string, unknown>;
        }
      } catch {
        parsed = null;
      }
      const cadence = parsed?.["cadence_hours"];
      return {
        at: typeof parsed?.["at"] === "string" ? (parsed["at"] as string) : null,
        timeframe:
          typeof parsed?.["timeframe"] === "string"
            ? (parsed["timeframe"] as string)
            : null,
        status:
          typeof parsed?.["status"] === "string"
            ? (parsed["status"] as string)
            : null,
        error:
          typeof parsed?.["error"] === "string"
            ? (parsed["error"] as string)
            : null,
        cadenceHours:
          typeof cadence === "number" && Number.isFinite(cadence)
            ? cadence
            : null,
        unread: false,
        raw: line,
      };
    });

    // ── Per-tf + top-level rollup (pure function, see tests) ──────
    const statusObj =
      statusRaw && typeof statusRaw === "object" && !Array.isArray(statusRaw)
        ? (statusRaw as Record<string, unknown>)
        : null;
    const rollup = computeDatasetFreshness(statusObj, alerts, Date.now());
    const { writtenAt, timeframes, pastDueTimeframes, totalUnreadAlerts } =
      rollup;
    // "unknown" is reserved for "no readable status file"; the pure
    // rollup only knows about "no timeframes parsed". Promote here so
    // a missing/unreadable file always lands on unknown.
    const state: DatasetFreshnessHealth =
      !statusFileExists || statusReadError ? "unknown" : rollup.state;

    res.json({
      state,
      statusFileExists,
      alertsFileExists,
      statusReadError,
      writtenAt,
      timeframes,
      pastDueTimeframes,
      alerts,
      totalAlerts: alerts.length,
      totalUnreadAlerts,
      fetchedAt: new Date().toISOString(),
    });
  } catch (err) {
    res
      .status(500)
      .json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// ── Task #599 ────────────────────────────────────────────────────────
// Surface the latest 1h/2h calibration ON/OFF verdict that the
// dataset-refresher auto-triggers after each short-tf snapshot. The
// loop writes a `calibration_verdict.short_tf` block into
// `models/datasets/_freshness_status.json` with summary numbers,
// timestamps, and the path to the most recent verdict markdown +
// JSON under `artifacts/ml-engine/reports/`. We tail the markdown
// here (last ~8KB) so the dashboard can render the human-readable
// verdict without having to ship a separate file-server.
router.get(
  "/crypto/calibration-verdict/short-tf",
  async (_req, res): Promise<void> => {
    try {
      const fs = await import("node:fs");
      const path = await import("node:path");
      const { fileURLToPath } = await import("node:url");

      // Walk up from this module to find the workspace root that
      // contains `artifacts/ml-engine`. Mirrors the freshness route
      // above so dev (tsx) and bundled (dist) paths both resolve.
      let dir = path.dirname(fileURLToPath(import.meta.url));
      let mlEngineDir: string | null = null;
      for (let i = 0; i < 8; i++) {
        const candidate = path.join(dir, "artifacts", "ml-engine");
        if (fs.existsSync(candidate)) {
          mlEngineDir = candidate;
          break;
        }
        const parent = path.dirname(dir);
        if (parent === dir) break;
        dir = parent;
      }
      if (!mlEngineDir) {
        res.json({
          state: "unknown" as const,
          statusFileExists: false,
          statusReadError: null,
          shortTf: null,
          markdownPath: null,
          jsonPath: null,
          markdownTail: null,
          markdownReadError: null,
          fetchedAt: new Date().toISOString(),
        });
        return;
      }

      const statusPath = path.join(
        mlEngineDir,
        "models",
        "datasets",
        "_freshness_status.json",
      );

      let statusRaw: unknown = null;
      let statusFileExists = false;
      let statusReadError: string | null = null;
      try {
        const text = fs.readFileSync(statusPath, "utf8");
        statusFileExists = true;
        try {
          statusRaw = JSON.parse(text);
        } catch (e) {
          statusReadError = e instanceof Error ? e.message : String(e);
        }
      } catch {
        statusFileExists = false;
      }

      const statusObj =
        statusRaw && typeof statusRaw === "object" && !Array.isArray(statusRaw)
          ? (statusRaw as Record<string, unknown>)
          : null;
      const verdictContainer =
        (statusObj?.["calibration_verdict"] as
          | Record<string, unknown>
          | undefined) ?? null;
      const shortTfRaw =
        (verdictContainer?.["short_tf"] as
          | Record<string, unknown>
          | undefined) ?? null;

      const asString = (v: unknown): string | null =>
        typeof v === "string" ? v : null;
      const asNumber = (v: unknown): number | null =>
        typeof v === "number" && Number.isFinite(v) ? v : null;
      const asStringArray = (v: unknown): string[] | null =>
        Array.isArray(v) && v.every((x) => typeof x === "string")
          ? (v as string[])
          : null;

      type StageSummary = {
        nSlices: number | null;
        nPassingGate: number | null;
        nInTradeShareBand: number | null;
        meanTradeShare: number | null;
        meanDaLift: number | null;
        sumPnlPctTotalAug: number | null;
      };
      type VerdictSummary = {
        capturedAt: string | null;
        roundTripCostPct: number | null;
        timeframesSubset: string[] | null;
        wallTimeSeconds: number | null;
        nWorkers: number | null;
        nWorkUnits: number | null;
        off: StageSummary | null;
        on: StageSummary | null;
      };
      type ShortTfBlock = {
        lastStatus: string | null;
        lastAttemptAt: string | null;
        lastSuccessAt: string | null;
        lastError: string | null;
        lastElapsedSeconds: number | null;
        triggerTimeframes: string[] | null;
        timeoutSeconds: number | null;
        command: string | null;
        lastMdPath: string | null;
        lastJsonPath: string | null;
        summary: VerdictSummary | null;
      };

      function summarizeStage(stage: unknown): StageSummary | null {
        if (!stage || typeof stage !== "object") return null;
        const s = stage as Record<string, unknown>;
        return {
          nSlices: asNumber(s["n_slices"]),
          nPassingGate: asNumber(s["n_passing_gate"]),
          nInTradeShareBand: asNumber(s["n_in_trade_share_band"]),
          meanTradeShare: asNumber(s["mean_trade_share"]),
          meanDaLift: asNumber(s["mean_da_lift"]),
          sumPnlPctTotalAug: asNumber(s["sum_pnl_pct_total_aug"]),
        };
      }

      let shortTf: ShortTfBlock | null = null;
      if (shortTfRaw) {
        const summaryRaw = shortTfRaw["summary"];
        let summaryOut: VerdictSummary | null = null;
        if (summaryRaw && typeof summaryRaw === "object") {
          const s = summaryRaw as Record<string, unknown>;
          summaryOut = {
            capturedAt: asString(s["captured_at"]),
            roundTripCostPct: asNumber(s["round_trip_cost_pct"]),
            timeframesSubset: asStringArray(s["timeframes_subset"]),
            wallTimeSeconds: asNumber(s["wall_time_seconds"]),
            nWorkers: asNumber(s["n_workers"]),
            nWorkUnits: asNumber(s["n_work_units"]),
            off: summarizeStage(s["off"]),
            on: summarizeStage(s["on"]),
          };
        }
        shortTf = {
          lastStatus: asString(shortTfRaw["last_status"]),
          lastAttemptAt: asString(shortTfRaw["last_attempt_at"]),
          lastSuccessAt: asString(shortTfRaw["last_success_at"]),
          lastError: asString(shortTfRaw["last_error"]),
          lastElapsedSeconds: asNumber(shortTfRaw["last_elapsed_seconds"]),
          triggerTimeframes: asStringArray(shortTfRaw["trigger_timeframes"]),
          timeoutSeconds: asNumber(shortTfRaw["timeout_seconds"]),
          command: asString(shortTfRaw["command"]),
          lastMdPath: asString(shortTfRaw["last_md_path"]),
          lastJsonPath: asString(shortTfRaw["last_json_path"]),
          summary: summaryOut,
        };
      }

      // ── Tail the most recent verdict markdown (last ~8KB) ───────
      // Prefer the path stored in the status block (last_md_path);
      // fall back to globbing reports/ for the newest matching file
      // so a missing/stale status block does not blank the panel.
      const reportsDir = path.join(mlEngineDir, "reports");
      let markdownPath: string | null = null;
      let jsonPath: string | null = null;
      let markdownTail: string | null = null;
      let markdownReadError: string | null = null;

      const tryReadTail = (absPath: string): string | null => {
        try {
          const buf = fs.readFileSync(absPath, "utf8");
          const MAX_TAIL = 8 * 1024;
          if (buf.length <= MAX_TAIL) return buf;
          // Trim to a clean line boundary so the rendered markdown
          // does not start mid-paragraph.
          const trimmed = buf.slice(buf.length - MAX_TAIL);
          const nl = trimmed.indexOf("\n");
          return nl >= 0 ? trimmed.slice(nl + 1) : trimmed;
        } catch (e) {
          markdownReadError = e instanceof Error ? e.message : String(e);
          return null;
        }
      };

      // The loop stores `last_md_path` / `last_json_path` relative
      // to `artifacts/ml-engine` (its `ROOT`), e.g.
      // "reports/20260429T040000Z-task592-1h2h-stage2-verdict.md".
      if (shortTf?.lastMdPath) {
        const abs = path.isAbsolute(shortTf.lastMdPath)
          ? shortTf.lastMdPath
          : path.join(mlEngineDir, shortTf.lastMdPath);
        if (fs.existsSync(abs)) {
          markdownPath = shortTf.lastMdPath;
          markdownTail = tryReadTail(abs);
        }
      }
      if (shortTf?.lastJsonPath) {
        jsonPath = shortTf.lastJsonPath;
      }
      // Fallback: glob reports/ for the newest md report so a missing
      // status block (or stale path) does not blank the panel.
      if (markdownTail == null && fs.existsSync(reportsDir)) {
        try {
          const candidates = fs
            .readdirSync(reportsDir)
            .filter((f) => f.endsWith("-task592-1h2h-stage2-verdict.md"))
            .map((f) => ({
              name: f,
              full: path.join(reportsDir, f),
              mtime: fs.statSync(path.join(reportsDir, f)).mtimeMs,
            }))
            .sort((a, b) => b.mtime - a.mtime);
          if (candidates.length > 0) {
            const newest = candidates[0]!;
            markdownPath = path.join("reports", newest.name);
            markdownTail = tryReadTail(newest.full);
          }
        } catch (e) {
          markdownReadError = e instanceof Error ? e.message : String(e);
        }
      }

      // Map the loop's last_status into a coarse health pill the
      // dashboard can colour without recomputing thresholds.
      const state: "ok" | "error" | "timeout" | "unknown" = (() => {
        if (!shortTf?.lastStatus) return "unknown";
        if (shortTf.lastStatus === "ok") return "ok";
        if (shortTf.lastStatus === "timeout") return "timeout";
        return "error";
      })();

      res.json({
        state,
        statusFileExists,
        statusReadError,
        shortTf,
        markdownPath,
        jsonPath,
        markdownTail,
        markdownReadError,
        fetchedAt: new Date().toISOString(),
      });
    } catch (err) {
      res
        .status(500)
        .json({ error: err instanceof Error ? err.message : "unknown" });
    }
  },
);

router.get("/crypto/brain/runtime-status", async (_req, res): Promise<void> => {
  try {
    const runtime = await computeBrainRuntimeState();
    const { state, brainEnabled, brainSource, recentAbstainReasons, recentNonAbstainCount, lastSuccessfulAt } = runtime;

    // Best-effort current run dir — derived from the latest
    // training_run_* directory in artifacts/ml-engine/models/ so the
    // banner can show "model: training_run_20260428T113310Z". Failing
    // to read the directory is not fatal; the banner just omits the
    // field.
    let currentRunDir: string | null = null;
    try {
      const { readdirSync } = await import("node:fs");
      const path = await import("node:path");
      const { fileURLToPath } = await import("node:url");
      const here = path.dirname(fileURLToPath(import.meta.url));
      const modelsDir = path.resolve(here, "../../../../ml-engine/models");
      const runs = readdirSync(modelsDir).filter((d) => d.startsWith("training_run_"));
      runs.sort();
      currentRunDir = runs.at(-1) ?? null;
    } catch {
      currentRunDir = null;
    }

    // Task #686 — surface the promotion-gate retry roll-up so the
    // dashboard chip can show "the brain-enable check had to wait
    // for the ml-engine N times in the last hour, most recent
    // reason: <retry_failure_reason>". Pure read of the in-memory
    // ring buffer; no extra fetch / DB round-trip on this hot path.
    const promotionGateRetries: PromotionGateRetryStats =
      getPromotionGateRetryStats();

    res.json({
      state,
      brainEnabled,
      brainSource,
      mlAvailabilitySnapshotReady: isMlAvailabilitySnapshotReady(),
      recentAbstainReasons,
      recentNonAbstainCount,
      lastSuccessfulDecisionAt: lastSuccessfulAt
        ? lastSuccessfulAt.toISOString()
        : null,
      currentRunDir,
      windowMinutes: 30,
      promotionGateRetries,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

/**
 * Task #532 / C-11 — operator-visible meta-brain status. The
 * ml-engine exposes `/ml/meta-brain/health`, `/ml/meta-brain/stats`,
 * and `/ml/meta-brain/last_replay` but the dashboard had no surface
 * to read them — the audit found the meta-brain card on
 * `/diagnostics` was reading `/api/crypto/meta/action-distribution`
 * (the post-action distribution, not the learning state) and showing
 * an empty card. This proxy fans out to the three ml-engine
 * endpoints in parallel and returns a single `learningState` payload
 * so the frontend can render "next replay attempt", thresholds met,
 * and the trust-by-family map without three round-trips. Failures of
 * any single sub-call surface as `null` on that field rather than
 * collapsing the whole response.
 */
router.get("/crypto/meta-brain/status", async (_req, res): Promise<void> => {
  const mlBase = process.env.ML_ENGINE_URL || "http://localhost:8000";
  async function safeGet<T>(path: string): Promise<T | null> {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 4000);
      const resp = await fetch(`${mlBase}${path}`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (!resp.ok) return null;
      return (await resp.json()) as T;
    } catch {
      return null;
    }
  }
  // Task #532 — also count paper_trades closed in the last 24h so the
  // proxy can render an explicit "zero activity in 24h because no
  // closed trades" message. The supervisory brain only learns from
  // realized outcomes (record_outcome calls), so 0 closed trades →
  // 0 record_outcome calls → 0 trust-state changes by construction.
  // Without this surface the dashboard's "0 evaluate / 0 outcome"
  // counters look like a meta-brain bug instead of "the executors
  // haven't traded".
  // Task #532 / Rev 2.2 — when the DB query fails we surface `null`
  // (not 0) so the dashboard renders "n/a" instead of falsely
  // claiming "0 closed trades in 24h". The "no fake zero" contract
  // applies to every counter sourced from a query/probe that can
  // legitimately fail.
  let closedTrades24h: number | null = 0;
  try {
    const r = await db.execute(sql`
      SELECT COUNT(*)::int AS n
      FROM paper_trades
      WHERE closed_at IS NOT NULL
        AND closed_at >= NOW() - INTERVAL '24 hours'
    `);
    const rows = (r as unknown as { rows?: Array<{ n: number }> }).rows ?? [];
    closedTrades24h = Number(rows[0]?.n ?? 0);
  } catch {
    closedTrades24h = null;
  }

  const [health, stats, lastReplay] = await Promise.all([
    safeGet<Record<string, unknown>>("/ml/meta-brain/health"),
    safeGet<Record<string, unknown>>("/ml/meta-brain/stats"),
    safeGet<Record<string, unknown>>("/ml/meta-brain/last_replay"),
  ]);
  const reachable = health !== null;
  const ok = reachable && (health as { ok?: unknown })?.ok === true;

  // Task #532 / C-11 — derive operator-truth fields the audit
  // required from the raw stats / lastReplay payloads:
  //   * lastEvaluateAt  — most recent reward entry timestamp (every
  //     evaluate that was followed by record_outcome ends up here).
  //   * lastRecordOutcomeAt — alias for lastEvaluateAt (in this
  //     architecture each reward IS a record_outcome call); kept as
  //     a separate field so the UI can render both rows even when
  //     the upstream collapses them later.
  //   * trustStateChanges24h — count of recent_rewards entries whose
  //     timestamp is within the last 24h. Each reward corresponds to
  //     a trust-model update, so this is the literal "trust-state
  //     change count over 24h" the audit asked for.
  //   * lastDirective — last_replay.last_run.directive (or
  //     last_committed_run.directive) so the operator can see the
  //     last decision the supervisory brain authorized.
  //   * activityNote — explicit prose that explains why the counters
  //     are zero, so the operator never has to guess.
  function asObj(o: unknown): Record<string, unknown> | null {
    return o && typeof o === "object" && !Array.isArray(o)
      ? (o as Record<string, unknown>)
      : null;
  }
  function asArr(o: unknown): unknown[] {
    return Array.isArray(o) ? o : [];
  }
  function pickStr(o: Record<string, unknown> | null, k: string): string | null {
    if (!o) return null;
    const v = o[k];
    return typeof v === "string" ? v : null;
  }

  const recentRewards = asArr(stats ? stats["recent_rewards"] : null);
  // Reward entries are ordered oldest→newest by summarize_rewards();
  // pick the last one with a timestamp-ish field.
  // Task #532 / Rev 2.2 — when the ml-engine `/stats` probe failed
  // (`stats === null`) we cannot prove the count is zero — emit
  // `null` so the dashboard renders "n/a" instead of falsely
  // claiming "0 trust-state changes in 24h".
  let lastEvaluateAt: string | null = null;
  let trustStateChanges24h: number | null = stats === null ? null : 0;
  const cutoff24h = Date.now() - 24 * 60 * 60 * 1000;
  for (const entry of recentRewards) {
    const e = asObj(entry);
    if (!e) continue;
    const ts =
      pickStr(e, "timestamp") ??
      pickStr(e, "ts") ??
      pickStr(e, "at") ??
      pickStr(e, "recorded_at");
    if (ts) {
      const t = Date.parse(ts);
      if (Number.isFinite(t)) {
        if (!lastEvaluateAt || t > Date.parse(lastEvaluateAt)) {
          lastEvaluateAt = new Date(t).toISOString();
        }
        if (t >= cutoff24h) trustStateChanges24h += 1;
      }
    }
  }

  const lastReplayObj = asObj(lastReplay);
  const lastRun = asObj(lastReplayObj?.["last_run"] ?? null);
  const lastCommittedRun = asObj(lastReplayObj?.["last_committed_run"] ?? null);
  const lastDirective =
    asObj(lastRun?.["directive"] ?? null) ??
    asObj(lastCommittedRun?.["directive"] ?? null);

  // Explicit zero-activity explanation. Build only when we genuinely
  // have zero learning events in the window — otherwise omit so the
  // UI doesn't render a misleading note alongside live data.
  //
  // Task #551 / Step 5 — wording hardened to the user-directive item
  // 6 baseline ("replaying telemetry, not committing trust updates,
  // not enough attributed trades, no live learning yet"). The
  // previous "Zero meta-brain learning events…" copy was technically
  // correct but softer than the directive requires; the truth row now
  // says the four mandated phrases on the dashboard so the operator
  // cannot misread "no events yet" as "warming up".
  let activityNote: string | null = null;
  if (trustStateChanges24h === 0) {
    if (closedTrades24h === 0) {
      activityNote =
        "Replaying telemetry, not committing trust updates: not enough attributed trades, no live learning yet. paper_trades.closed_at is empty for the last 24h, so no realized outcome reached the supervisory brain.";
    } else {
      activityNote =
        `Replaying telemetry, not committing trust updates: not enough attributed trades, no live learning yet. ${closedTrades24h} closed paper trade(s) in the last 24h were not attributed — most likely the authorizing tick_id has been evicted from the meta-brain tick cache (record-outcome no-ops with reason=tick_id_not_in_cache).`;
    }
  }

  res.json({
    available: ok,
    reachable,
    mlEngineUrl: mlBase,
    health,
    stats,
    lastReplay,
    learningTruth: {
      lastEvaluateAt,
      lastRecordOutcomeAt: lastEvaluateAt,
      trustStateChanges24h,
      closedTrades24h,
      lastDirective,
      activityNote,
    },
    fetchedAt: new Date().toISOString(),
  });
});

/**
 * Task #518 — `/crypto/brain/quant-status` was removed in an earlier
 * sweep but a stale frontend bundle (and the deployment-log audit)
 * still polls it, producing a noisy stream of 404s. Return a stable
 * 410 Gone with a pointer to its replacement so old clients can be
 * fixed at the network tab without grepping git history.
 */
router.get("/crypto/brain/quant-status", (_req, res): void => {
  res.status(410).json({
    error: "gone",
    message:
      "Task #518: `/crypto/brain/quant-status` was removed. Use `/crypto/brain/runtime-status` for the dashboard's brain state pill, or `/crypto/brain/state` for the operator kill-switch view.",
    replacement: "/crypto/brain/runtime-status",
  });
});

/**
 * Task #255 / #444 — Fleet brain flip is permanently disabled and the
 * advisory LLM sidecar replacement was removed in #444. The route stays
 * as a stable 410 Gone for any old client still polling it.
 */
router.post("/crypto/brain/toggle", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  res.status(410).json({
    error: "gone",
    message:
      "Task #444: the LLM/news/sidecar pipeline has been removed end-to-end. " +
      "Quant brain is the sole authority for trade decisions; nothing to toggle.",
    replacement: null,
  });
});

router.get("/crypto/brain/accuracy", async (req, res): Promise<void> => {
  try {
    const hours = Math.min(168, Math.max(1, Number((req.query as { hours?: string }).hours ?? 24)));
    const sinceSql = sql.raw(`NOW() - INTERVAL '${hours} hours'`);

    // Prior-only pooled-fallback predictions (source='prior') always return
    // the same Laplace-smoothed marginals per timeframe. Counting them in
    // the headline scoreboard would drag accuracy toward the empirical
    // prior and look like a regression. Default: excluded. Pass
    // ?includePrior=1 (or =true) to include them — useful for diagnosing
    // how much of recent traffic is on the fallback.
    const includePriorRaw = String((req.query as { includePrior?: string }).includePrior ?? "");
    const includePrior = includePriorRaw === "1" || includePriorRaw.toLowerCase() === "true";
    // `IS DISTINCT FROM` so legacy NULL rows (pre-Phase-5) stay included.
    const priorFilterSql = includePrior ? sql.raw("") : sql.raw("AND source IS DISTINCT FROM 'prior'");

    // Per-brain × timeframe rollup of resolved predictions, with direction
    // breakdown so the "stable-bias" is visible (a brain that calls 'stable'
    // 100% of the time is correct most of the time on small timeframes for
    // structural reasons, not because of skill).
    const rollupRows = await db.execute(sql`
      SELECT
        CASE
          WHEN reasoning LIKE '%[BRAIN=QUANT]%' THEN 'QUANT'
          WHEN reasoning LIKE '%[BRAIN=LLM]%'   THEN 'LLM'
          ELSE 'untagged'
        END AS brain,
        timeframe,
        COUNT(*)                                                   AS total,
        COUNT(*) FILTER (WHERE outcome IS NOT NULL AND outcome::text != 'pending') AS resolved,
        COUNT(*) FILTER (WHERE outcome::text = 'correct')          AS correct,
        COUNT(*) FILTER (WHERE direction = 'stable')               AS stable_calls,
        COUNT(*) FILTER (WHERE direction = 'up')                   AS up_calls,
        COUNT(*) FILTER (WHERE direction = 'down')                 AS down_calls,
        COUNT(*) FILTER (WHERE direction != 'stable' AND outcome::text = 'correct') AS directional_correct,
        COUNT(*) FILTER (WHERE direction != 'stable' AND outcome IS NOT NULL AND outcome::text != 'pending') AS directional_resolved,
        -- Realised market direction per row (same window, same brain × timeframe slice):
        -- compare actual_price at resolution vs price_at_prediction. Lets the
        -- dashboard show "the brain called 87↓/14↑ but the market actually
        -- printed 132↑/143↓ bars" so directional bias vs. genuine signal
        -- weakness is diagnosable without dropping into SQL (Task #172).
        COUNT(*) FILTER (WHERE actual_price IS NOT NULL AND price_at_prediction IS NOT NULL AND actual_price > price_at_prediction) AS realized_up,
        COUNT(*) FILTER (WHERE actual_price IS NOT NULL AND price_at_prediction IS NOT NULL AND actual_price < price_at_prediction) AS realized_down,
        COUNT(*) FILTER (WHERE actual_price IS NOT NULL AND price_at_prediction IS NOT NULL AND actual_price = price_at_prediction) AS realized_flat
      FROM predictions
      WHERE created_at > ${sinceSql}
        ${priorFilterSql}
      GROUP BY 1, 2
      ORDER BY 1, 2
    `);

    // Hourly trend for QUANT only — accuracy + directional accuracy per hour
    // bucket so we can see if it's improving as retrains land.
    const trendRows = await db.execute(sql`
      SELECT
        date_trunc('hour', resolved_at) AS bucket,
        COUNT(*)                                                            AS resolved,
        COUNT(*) FILTER (WHERE outcome::text = 'correct')                   AS correct,
        COUNT(*) FILTER (WHERE direction != 'stable')                       AS directional,
        COUNT(*) FILTER (WHERE direction != 'stable' AND outcome::text = 'correct') AS directional_correct
      FROM predictions
      WHERE reasoning LIKE '%[BRAIN=QUANT]%'
        AND resolved_at IS NOT NULL
        AND resolved_at > ${sinceSql}
        ${priorFilterSql}
      GROUP BY 1
      ORDER BY 1
    `);

    // Companion count: how many prior-source rows the headline number is
    // currently hiding (or, when includePrior is on, how many it folded in).
    // Surfaces in the JSON so the dashboard can render a "X prior-fallback
    // predictions excluded" footnote next to the toggle.
    const priorCountRow = await db.execute(sql`
      SELECT COUNT(*)::int AS prior_total,
             COUNT(*) FILTER (WHERE outcome IS NOT NULL AND outcome::text != 'pending')::int AS prior_resolved
      FROM predictions
      WHERE created_at > ${sinceSql}
        AND source = 'prior'
    `);
    const priorRow = (priorCountRow.rows[0] ?? {}) as { prior_total?: unknown; prior_resolved?: unknown };
    const priorTotal = Number(priorRow.prior_total ?? 0);
    const priorResolved = Number(priorRow.prior_resolved ?? 0);

    const byBrain = (rollupRows.rows as Array<Record<string, unknown>>).map((r) => {
      const total = Number(r.total ?? 0);
      const resolved = Number(r.resolved ?? 0);
      const correct = Number(r.correct ?? 0);
      const stable = Number(r.stable_calls ?? 0);
      const dirResolved = Number(r.directional_resolved ?? 0);
      const dirCorrect = Number(r.directional_correct ?? 0);
      const dirAccuracyPct =
        dirResolved > 0 ? Number(((100 * dirCorrect) / dirResolved).toFixed(1)) : null;
      const upCalls = Number(r.up_calls ?? 0);
      const downCalls = Number(r.down_calls ?? 0);
      const realizedUp = Number(r.realized_up ?? 0);
      const realizedDown = Number(r.realized_down ?? 0);
      const realizedFlat = Number(r.realized_flat ?? 0);
      // Directional-bias check (Task #172): compare the brain's up:down call
      // odds against the realised up:down bar odds in the same slice. Spec
      // says >2× divergence flags directional bias as a likely cause of a
      // weak-signal verdict. We use Laplace-smoothed odds (+1 to each side)
      // so a zero count on either side still yields a finite ratio rather
      // than collapsing to 0/Infinity.
      const callDirTotal = upCalls + downCalls;
      const realizedDirTotal = realizedUp + realizedDown;
      let directionalBias = false;
      if (callDirTotal > 0 && realizedDirTotal > 0) {
        const callOdds = (upCalls + 1) / (downCalls + 1);
        const realizedOdds = (realizedUp + 1) / (realizedDown + 1);
        const ratio =
          callOdds >= realizedOdds ? callOdds / realizedOdds : realizedOdds / callOdds;
        directionalBias = ratio > 2;
      }
      // Server-side weak-signal verdict shared with the dashboard badge
      // and the horizon-gate trade skip (single source of truth).
      const weakSignal =
        dirAccuracyPct != null &&
        dirResolved >= HORIZON_WEAK_MIN_RESOLVED &&
        dirAccuracyPct < HORIZON_WEAK_DIRECTIONAL_FLOOR_PCT;
      const baseReason = weakSignal
        ? `Directional accuracy ${dirAccuracyPct}% on n=${dirResolved} is below the ${HORIZON_WEAK_DIRECTIONAL_FLOOR_PCT}% coin-flip floor.`
        : null;
      const biasNote =
        weakSignal && directionalBias
          ? ` Likely cause: directional bias — brain called ${upCalls}↑/${downCalls}↓ while the market printed ${realizedUp}↑/${realizedDown}↓.`
          : "";
      const weakSignalReason = baseReason ? baseReason + biasNote : null;
      return {
        brain: String(r.brain),
        timeframe: String(r.timeframe),
        total,
        resolved,
        correct,
        accuracyPct: resolved > 0 ? Number(((100 * correct) / resolved).toFixed(1)) : null,
        stableCalls: stable,
        upCalls,
        downCalls,
        stableSharePct: total > 0 ? Number(((100 * stable) / total).toFixed(1)) : null,
        directionalResolved: dirResolved,
        directionalCorrect: dirCorrect,
        directionalAccuracyPct: dirAccuracyPct,
        realizedUp,
        realizedDown,
        realizedFlat,
        directionalBias,
        weakSignal,
        weakSignalReason,
      };
    });

    const trend = (trendRows.rows as Array<Record<string, unknown>>).map((r) => {
      const resolved = Number(r.resolved ?? 0);
      const correct = Number(r.correct ?? 0);
      const directional = Number(r.directional ?? 0);
      const dirCorrect = Number(r.directional_correct ?? 0);
      return {
        bucket: r.bucket instanceof Date ? r.bucket.toISOString() : String(r.bucket),
        resolved,
        correct,
        accuracyPct: resolved > 0 ? Number(((100 * correct) / resolved).toFixed(1)) : null,
        directional,
        directionalCorrect: dirCorrect,
        directionalAccuracyPct:
          directional > 0 ? Number(((100 * dirCorrect) / directional).toFixed(1)) : null,
      };
    });

    // Task #444 — `llmBias` is gone. The directional-bias safety net only
    // applied to LLM-authored predictions; with the LLM brain removed,
    // the snapshot is structurally meaningless and stays null for back-
    // compat with any client still reading the field.
    res.json({
      windowHours: hours,
      byBrain,
      quantHourlyTrend: trend,
      // Provenance footer for the dashboard's "prior excluded" footnote.
      priorFallback: {
        included: includePrior,
        total: priorTotal,
        resolved: priorResolved,
      },
      llmBias: null,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #424 — proxy the ml-engine's daily 5m head top-up status so the
// dashboard can render which replica actually fetched the most recent
// pull (and when). The ml-engine endpoint is read-only and unauthenticated;
// we just forward and surface a stable shape with the fields the UI
// reads. Failures fall back to `null` so an ml-engine outage never
// blanks the rest of the dashboard.
router.get("/crypto/brain/5m-topup-status", async (_req, res): Promise<void> => {
  const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(
    /\/$/,
    "",
  );
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 2000);
    const r = await fetch(`${base}/ml/admin/5m-topup/status`, {
      signal: ctrl.signal,
    });
    clearTimeout(t);
    if (!r.ok) {
      res.status(502).json({ error: `ml-engine ${r.status}` });
      return;
    }
    const body = await r.json();
    res.json(body);
  } catch (err) {
    res.status(502).json({
      error: err instanceof Error ? err.message : "unknown",
    });
  }
});

// Task #435 — proxy the ml-engine's recent-winners endpoint so the
// dashboard can render the last N successful-tick winners (replica +
// timestamp, newest-first). Without this list the operator can only
// see the latest pull and would never spot "host-a is winning 14 days
// in a row" until something already broke. Mirrors the no-auth pattern
// of the sibling /5m-topup-status proxy above. The optional `?limit`
// query is forwarded verbatim — the ml-engine clamps it server-side.
router.get(
  "/crypto/brain/5m-topup-recent-winners",
  async (req, res): Promise<void> => {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(
      /\/$/,
      "",
    );
    const limitParam = typeof req.query.limit === "string" ? req.query.limit : null;
    const qs = limitParam !== null ? `?limit=${encodeURIComponent(limitParam)}` : "";
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 2000);
      const r = await fetch(`${base}/ml/admin/5m-topup/recent-winners${qs}`, {
        signal: ctrl.signal,
      });
      clearTimeout(t);
      if (!r.ok) {
        res.status(502).json({ error: `ml-engine ${r.status}` });
        return;
      }
      const body = await r.json();
      res.json(body);
    } catch (err) {
      res.status(502).json({
        error: err instanceof Error ? err.message : "unknown",
      });
    }
  },
);

router.get("/crypto/brain/retrain", async (_req, res): Promise<void> => {
  try {
    const sched = getRetrainSchedulerState();
    let mlStatus: unknown = null;
    try {
      const base = process.env.ML_ENGINE_URL || "http://localhost:8000";
      const r = await fetch(`${base.replace(/\/$/, "")}/ml/admin/retrain/status`);
      if (r.ok) mlStatus = await r.json();
    } catch {
      mlStatus = null;
    }
    res.json({ scheduler: sched, mlEngine: mlStatus });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #412 — read-only proxy of the daily 5m head top-up state. Mirrors
// the no-auth pattern of /crypto/brain/retrain so the dashboard can render
// a freshness tile (last_finished_at, per-coin rows inserted, alerts,
// per-coin contiguous_days) without exposing ml-engine directly.
router.get("/crypto/brain/5m-topup/status", async (_req, res): Promise<void> => {
  try {
    const base = process.env.ML_ENGINE_URL || "http://localhost:8000";
    const r = await fetch(`${base.replace(/\/$/, "")}/ml/admin/5m-topup/status`);
    if (!r.ok) {
      res.status(502).json({ error: `ml-engine ${r.status}` });
      return;
    }
    const body = await r.json();
    res.json(body);
  } catch (err) {
    res.status(502).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

router.post("/crypto/brain/retrain/run", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const adminKey = process.env.ADMIN_API_KEY || process.env.ML_ADMIN_TOKEN;
  if (!adminKey) {
    res.status(500).json({ error: "ADMIN_API_KEY not set" });
    return;
  }
  try {
    const base = process.env.ML_ENGINE_URL || "http://localhost:8000";
    const r = await fetch(`${base.replace(/\/$/, "")}/ml/admin/retrain`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Admin-Key": adminKey },
      body: JSON.stringify(req.body ?? {}),
    });
    const text = await r.text();
    res.status(r.status).type("application/json").send(text);
  } catch (err) {
    res.status(502).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #324 — thin per-coin retrain entry point. Validates the requested
// coin against the monitored list (typo guard) and forwards a narrowly-
// scoped body to /ml/admin/retrain so a single coin's slices can be
// retrained in minutes instead of the full ~4-5h cycle.
const PER_COIN_RETRAIN_TIMEFRAMES = ["1m", "5m", "1h", "2h", "6h", "1d"] as const;
type PerCoinRetrainTimeframe = (typeof PER_COIN_RETRAIN_TIMEFRAMES)[number];
router.post("/crypto/brain/retrain/coin", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const adminKey = process.env.ADMIN_API_KEY || process.env.ML_ADMIN_TOKEN;
  if (!adminKey) {
    res.status(500).json({ error: "ADMIN_API_KEY not set" });
    return;
  }
  const body = (req.body ?? {}) as { coinId?: unknown; timeframes?: unknown };
  const coinId = typeof body.coinId === "string" ? body.coinId.trim() : "";
  if (!coinId) {
    res.status(400).json({ error: "coinId is required (string)" });
    return;
  }
  const monitored = MONITORED_COINS.find((c) => c.id === coinId);
  if (!monitored) {
    res.status(400).json({
      error: `Unknown coinId '${coinId}'. Must be one of the ${MONITORED_COINS.length} monitored coins.`,
      validCoinIds: MONITORED_COINS.map((c) => c.id),
    });
    return;
  }
  let timeframes: PerCoinRetrainTimeframe[] | undefined;
  if (body.timeframes !== undefined && body.timeframes !== null) {
    if (!Array.isArray(body.timeframes) || body.timeframes.length === 0) {
      res.status(400).json({ error: "timeframes must be a non-empty array of strings" });
      return;
    }
    const cleaned: PerCoinRetrainTimeframe[] = [];
    for (const tf of body.timeframes) {
      if (typeof tf !== "string" || !PER_COIN_RETRAIN_TIMEFRAMES.includes(tf as PerCoinRetrainTimeframe)) {
        res.status(400).json({
          error: `Invalid timeframe '${String(tf)}'. Allowed: ${PER_COIN_RETRAIN_TIMEFRAMES.join(", ")}`,
        });
        return;
      }
      if (!cleaned.includes(tf as PerCoinRetrainTimeframe)) {
        cleaned.push(tf as PerCoinRetrainTimeframe);
      }
    }
    timeframes = cleaned;
  }
  const forwardBody: { coins: string[]; timeframes?: string[] } = { coins: [coinId] };
  if (timeframes) forwardBody.timeframes = timeframes;
  try {
    const base = process.env.ML_ENGINE_URL || "http://localhost:8000";
    const r = await fetch(`${base.replace(/\/$/, "")}/ml/admin/retrain`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Admin-Key": adminKey },
      body: JSON.stringify(forwardBody),
    });
    const text = await r.text();
    // Mirror /crypto/brain/retrain/run exactly: forward the upstream
    // status + raw JSON body. The validated coinId / timeframes scope
    // is communicated via the request, not the response — this keeps
    // both retrain entry points wire-compatible (202 accepted /
    // 409 retrain already in progress, etc.).
    res.status(r.status).type("application/json").send(text);
  } catch (err) {
    res.status(502).json({ error: err instanceof Error ? err.message : "unknown" });
  }
});

// Task #325 — read-only proxy of the watchdog history. No auth (mirrors
// /admin/retrain/status). Surfaces the per-retrain convergence snapshots
// (status: converged/improving/regressed/flat/stalled/first_run) so the
// dashboard can show whether the brain is making progress between runs.
router.get("/crypto/brain/verification-history", async (_req, res): Promise<void> => {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 8000);
  try {
    const base = process.env.ML_ENGINE_URL || "http://localhost:8000";
    const r = await fetch(
      `${base.replace(/\/$/, "")}/ml/admin/verification-history?limit=20`,
      { signal: ctrl.signal },
    );
    const text = await r.text();
    const upstreamCt = r.headers.get("content-type");
    if (upstreamCt) res.type(upstreamCt);
    else res.type("application/json");
    res.status(r.status).send(text);
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    res
      .status(aborted ? 504 : 502)
      .json({ error: aborted ? "ml-engine timeout" : err instanceof Error ? err.message : "unknown" });
  } finally {
    clearTimeout(timer);
  }
});

// Task #453 — `/crypto/brain/cost` endpoint removed alongside
// `rate-limiter.ts`. Computed gpt/gemini USD spend from
// `getBudgetStatus()`, which always returned zero usage post-#444.
// No frontend consumer existed.

/**
 * Phase 3 — per-specialist per-regime accuracy on the live journal.
 *
 * Reads the last `limit` resolved prediction_journal rows whose
 * `specialist_scores` column is populated, then aggregates each
 * specialist's directional call (the
 * argmax of probUp/probDown) against the realized outcome
 * (`up` / `down` / `stable` from realizedReturnPct vs the per-tf label
 * threshold — same convention as predictions.outcome). Returns the
 * shape consumed by `<SpecialistAccuracyCard />` on the diagnostics
 * page; never throws into the trading path. Defaults bound `limit` so
 * a curious operator can't accidentally pull the whole journal.
 */
router.get("/crypto/brain/specialists", async (req, res): Promise<void> => {
  const limit = Math.min(Math.max(Number(req.query.limit) || 1000, 50), 5000);
  try {
    // Read only rows whose dedicated `specialist_scores` column is
    // populated. The legacy `gates_applied ? 'specialists'` sidecar
    // fallback was dropped post-Phase-4 so the partial index
    // `prediction_journal_specialist_scores_created_at_idx` can serve
    // this query directly — keeping the diagnostics page O(limit) as
    // the journal grows. Any pre-Phase-4 rows that never got backfilled
    // are accepted as not visible here.
    const rows = await db
      .select({
        regimeLabel: predictionJournalTable.regimeLabel,
        outcome: predictionJournalTable.outcome,
        realizedReturnPct: predictionJournalTable.realizedReturnPct,
        specialistScores: predictionJournalTable.specialistScores,
        timeframe: predictionJournalTable.timeframe,
      })
      .from(predictionJournalTable)
      .where(sql`${predictionJournalTable.outcome} IS NOT NULL AND ${predictionJournalTable.specialistScores} IS NOT NULL`)
      .orderBy(desc(predictionJournalTable.createdAt))
      .limit(limit);

    type SpecCell = {
      kind: string;
      regime: string;
      n: number;
      directionalCalls: number;        // rows where specialist called up/down
      directionalCorrect: number;      // of those, how many matched outcome
      meanProbCorrect: number;         // running sum of p(direction)
      meanRealizedAbs: number;         // running sum of |realizedReturnPct|
    };
    const cells = new Map<string, SpecCell>();
    let withSpecialists = 0;
    let consideredRows = 0;

    for (const row of rows) {
      // The query filter guarantees specialist_scores IS NOT NULL, so
      // we read the dedicated column directly — no legacy
      // gates_applied.specialists fallback.
      const specs = Array.isArray(row.specialistScores)
        ? (row.specialistScores as unknown[])
        : null;
      if (!specs || specs.length === 0) continue;
      withSpecialists += 1;
      consideredRows += 1;
      const regime = row.regimeLabel ?? "unknown";
      // Realized direction from the resolved return — threshold-free
      // sign comparison so a specialist's "up" vs "down" call has a
      // ground-truth side regardless of the per-tf label threshold the
      // main 3-class head uses for its `outcome` text. `predictionJournal.outcome`
      // is "correct"/"wrong"/"neutral" relative to the MAIN head's
      // direction, which isn't useful here.
      const realized = typeof row.realizedReturnPct === "number" ? row.realizedReturnPct : null;
      const realizedSide: "up" | "down" | null =
        realized === null || realized === 0 ? null : realized > 0 ? "up" : "down";
      for (const sp of specs as Array<Record<string, unknown>>) {
        const kind = String(sp.kind ?? "unknown");
        const probUp = typeof sp.probUp === "number" ? sp.probUp : null;
        const probDown = typeof sp.probDown === "number" ? sp.probDown : null;
        // applicable=false means the live regime wasn't in this specialist's
        // training subset; we still record the row under its regime so an
        // operator can spot a specialist firing where it shouldn't.
        const key = `${kind}::${regime}`;
        const cell = cells.get(key) ?? {
          kind, regime, n: 0,
          directionalCalls: 0, directionalCorrect: 0,
          meanProbCorrect: 0, meanRealizedAbs: 0,
        };
        cell.n += 1;
        if (realized !== null) cell.meanRealizedAbs += Math.abs(realized);
        if (probUp !== null && probDown !== null && realizedSide !== null) {
          const dir: "up" | "down" = probUp >= probDown ? "up" : "down";
          cell.directionalCalls += 1;
          if (dir === realizedSide) {
            cell.directionalCorrect += 1;
            cell.meanProbCorrect += dir === "up" ? probUp : probDown;
          }
        }
        cells.set(key, cell);
      }
    }

    const specialists = Array.from(cells.values()).map((c) => ({
      kind: c.kind,
      regime: c.regime,
      n: c.n,
      directionalCalls: c.directionalCalls,
      directionalCorrect: c.directionalCorrect,
      directionalAccuracy:
        c.directionalCalls > 0 ? c.directionalCorrect / c.directionalCalls : null,
      meanProbWhenCorrect:
        c.directionalCorrect > 0 ? c.meanProbCorrect / c.directionalCorrect : null,
      meanRealizedAbsPct: c.n > 0 ? c.meanRealizedAbs / c.n : null,
    }));
    specialists.sort((a, b) =>
      a.kind === b.kind ? a.regime.localeCompare(b.regime) : a.kind.localeCompare(b.kind)
    );

    res.json({
      sampledRows: rows.length,
      rowsWithSpecialists: withSpecialists,
      consideredRows,
      specialists,
      generatedAt: new Date().toISOString(),
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "specialists query failed" });
  }
});

/**
 * Phase 4 — abstain rate per regime for the specialist meta-gate.
 *
 * Reads recent prediction_journal rows whose gates_applied JSON carries
 * the Phase-4 `metaGate` sidecar (every quant prediction since Phase 4
 * shipped) and groups by the live regime. For each regime returns:
 *   - total: number of journal rows seen
 *   - abstained: rows where the meta-gate skipped the trade
 *   - traded: rows where the meta-gate let the trade through
 *   - passthrough: rows where the gate had no quorum / main was already
 *                  abstaining and didn't actively block
 *   - reasonCounts: breakdown of meta-gate reasons (specialist_dissent_high,
 *                  specialist_meta_counter_signal, specialist_agree, ...)
 *   - abstainRate: abstained / (abstained + traded), null when denom=0
 *
 * Defaults bound `limit` so a curious operator can't pull the whole
 * journal. Never throws into the trading path.
 */
router.get("/crypto/brain/abstain-rate", async (req, res): Promise<void> => {
  const limit = Math.min(Math.max(Number(req.query.limit) || 2000, 50), 10000);
  try {
    const rows = await db
      .select({
        regimeLabel: predictionJournalTable.regimeLabel,
        gatesApplied: predictionJournalTable.gatesApplied,
        becameTrade: predictionJournalTable.becameTrade,
        skipReason: predictionJournalTable.skipReason,
      })
      .from(predictionJournalTable)
      .where(sql`${predictionJournalTable.gatesApplied} ? 'metaGate'`)
      .orderBy(desc(predictionJournalTable.createdAt))
      .limit(limit);

    type Cell = {
      regime: string;
      total: number;
      abstained: number;
      traded: number;
      passthrough: number;
      reasonCounts: Record<string, number>;
    };
    const cells = new Map<string, Cell>();
    let withMetaGate = 0;

    for (const row of rows) {
      const gates = (row.gatesApplied as Record<string, unknown> | null) ?? {};
      const meta = gates.metaGate as Record<string, unknown> | undefined;
      if (!meta) continue;
      withMetaGate += 1;
      const regime =
        (typeof meta.regime === "string" && meta.regime) ||
        row.regimeLabel ||
        "unknown";
      const reason = typeof meta.reason === "string" ? meta.reason : "unknown";
      const abstained = meta.abstained === true;
      const cell: Cell = cells.get(regime) ?? {
        regime, total: 0, abstained: 0, traded: 0, passthrough: 0, reasonCounts: {} as Record<string, number>,
      };
      cell.total += 1;
      cell.reasonCounts[reason] = (cell.reasonCounts[reason] ?? 0) + 1;
      if (abstained) {
        cell.abstained += 1;
      } else if (row.becameTrade === true) {
        cell.traded += 1;
      } else {
        // Direction collapsed to stable for a non-meta reason (no edge,
        // EV gate, fee gate, etc.) — count separately so the abstain rate
        // numerator only reflects the meta-gate's contribution.
        cell.passthrough += 1;
      }
      cells.set(regime, cell);
    }

    const regimes = Array.from(cells.values()).map((c) => {
      const denom = c.abstained + c.traded;
      return {
        regime: c.regime,
        total: c.total,
        abstained: c.abstained,
        traded: c.traded,
        passthrough: c.passthrough,
        abstainRate: denom > 0 ? c.abstained / denom : null,
        reasonCounts: c.reasonCounts,
      };
    });
    regimes.sort((a, b) => b.total - a.total);

    const totals = regimes.reduce(
      (acc, r) => {
        acc.total += r.total;
        acc.abstained += r.abstained;
        acc.traded += r.traded;
        acc.passthrough += r.passthrough;
        for (const [k, v] of Object.entries(r.reasonCounts)) {
          acc.reasonCounts[k] = (acc.reasonCounts[k] ?? 0) + v;
        }
        return acc;
      },
      { total: 0, abstained: 0, traded: 0, passthrough: 0, reasonCounts: {} as Record<string, number> },
    );
    const overallDenom = totals.abstained + totals.traded;

    res.json({
      sampledRows: rows.length,
      rowsWithMetaGate: withMetaGate,
      overall: {
        ...totals,
        abstainRate: overallDenom > 0 ? totals.abstained / overallDenom : null,
      },
      regimes,
      generatedAt: new Date().toISOString(),
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "abstain-rate query failed" });
  }
});

// Task #444 — `/crypto/brain/news-tags` removed (LLM news classifier and
// its `news_tags` table are gone). The dashboard panel that consumed
// this endpoint is deleted in the same task.

// Task #255 / Quant Live Health repurpose — removed
// `GET /crypto/shadow/readiness`. The cutover gate compared quant
// directional accuracy against the LLM baseline; the LLM no longer
// produces trade-level directions, so the gate degenerated to "always
// pass" and the UI card it backed was misleading.
router.get("/crypto/shadow/readiness", (_req, res): void => {
  res.status(410).json({
    error: "gone",
    message:
      "Quant brain cutover already happened (Task #91 / #255). The readiness gate was retired with the Quant Live Health page repurpose.",
  });
});

// Quant model coverage — surfaces which (coin, timeframe) pairs the
// ml-engine actually has a trained model for, so an operator can see at a
// glance which timeframes the quant brain is serving versus silently
// falling through to the LLM. Mirrors ml-availability's resolution
// (per-coin first, then __pooled__) but returns the full matrix instead of
// a single yes/no, so the dashboard can render a coverage grid.
router.get("/crypto/quant-coverage", async (_req, res): Promise<void> => {
  try {
    const r = await getMlModels();
    const perCoin = new Set<string>();
    const pooled = new Set<string>();
    for (const m of r.available || []) {
      if (m.coinId === "__pooled__") pooled.add(m.timeframe);
      else perCoin.add(`${m.coinId}:${m.timeframe}`);
    }
    const timeframes = Object.keys(TIMEFRAMES);
    const coins = MONITORED_COINS.map(c => ({ id: c.id, symbol: c.symbol, name: c.name }));
    const cells: { coinId: string; timeframe: string; source: "per-coin" | "pooled" | "none" }[] = [];
    for (const c of coins) {
      for (const tf of timeframes) {
        const source = perCoin.has(`${c.id}:${tf}`)
          ? "per-coin"
          : pooled.has(tf)
            ? "pooled"
            : "none";
        cells.push({ coinId: c.id, timeframe: tf, source });
      }
    }
    res.json({
      coins,
      timeframes,
      cells,
      pooled: timeframes.filter(tf => pooled.has(tf)),
      darkTimeframes: timeframes.filter(tf => !pooled.has(tf) && !coins.some(c => perCoin.has(`${c.id}:${tf}`))),
      fetchedAt: new Date().toISOString(),
    });
  } catch (err) {
    res.status(503).json({ error: err instanceof Error ? err.message : "ml-engine unreachable" });
  }
});

// Anomaly-cancel safety net visibility (task #118).
// The exit path in paper-trader marks a trade row as status='cancelled' with
// pnl=0 *only* when the price ratio falls outside [0.33, 3.0] x entry — the
// "this looks like a feed glitch, pretend the trade never happened" branch.
// No other code path writes status='cancelled', so this query is an
// unambiguous count of how often that safety net actually fired.
router.get("/crypto/anomaly-cancels", async (_req, res): Promise<void> => {
  try {
    const now = Date.now();
    const windowMs = 24 * 60 * 60 * 1000;
    const since = new Date(now - windowMs);
    const rows = await db
      .select({
        coinId: paperTradesTable.coinId,
        coinName: paperTradesTable.coinName,
        closedAt: paperTradesTable.closedAt,
      })
      .from(paperTradesTable)
      .where(and(
        eq(paperTradesTable.status, "cancelled"),
        gte(paperTradesTable.closedAt, since),
      ));

    const total = rows.length;
    const byCoinMap = new Map<string, { coinId: string; coinName: string; count: number; lastAt: string | null }>();
    for (const r of rows) {
      const prev = byCoinMap.get(r.coinId);
      const closedIso = r.closedAt ? new Date(r.closedAt).toISOString() : null;
      if (prev) {
        prev.count += 1;
        if (closedIso && (!prev.lastAt || closedIso > prev.lastAt)) prev.lastAt = closedIso;
      } else {
        byCoinMap.set(r.coinId, { coinId: r.coinId, coinName: r.coinName, count: 1, lastAt: closedIso });
      }
    }
    const byCoin = [...byCoinMap.values()].sort((a, b) => b.count - a.count);

    // Hourly buckets for the sparkline. 24 rolling 1-hour buckets ending at
    // "now" (not aligned to clock-hour boundaries) so the rightmost bar
    // always represents the most recent hour and a fresh spike is visible
    // immediately rather than waiting for the next clock hour.
    const bucketMs = 60 * 60 * 1000;
    const bucketCount = 24;
    const firstBucketStart = now - bucketCount * bucketMs;
    const buckets: { ts: number; count: number }[] = [];
    for (let i = 0; i < bucketCount; i++) {
      buckets.push({ ts: firstBucketStart + i * bucketMs, count: 0 });
    }
    for (const r of rows) {
      if (!r.closedAt) continue;
      const t = new Date(r.closedAt).getTime();
      const idx = Math.floor((t - firstBucketStart) / bucketMs);
      if (idx >= 0 && idx < bucketCount) buckets[idx].count += 1;
    }

    // Threshold: anomaly cancels are meant to be *rare*. More than 5 in any
    // single hour bucket is loud enough to mean "the feed is probably
    // misbehaving" — surface it as an alert. Threshold lives here so the UI
    // doesn't have to re-derive it.
    const hourlyThreshold = 5;
    const spikingBuckets = buckets.filter(b => b.count >= hourlyThreshold);
    const alert = spikingBuckets.length > 0
      ? {
          hourlyThreshold,
          peakHourCount: Math.max(...buckets.map(b => b.count)),
          spikingHourCount: spikingBuckets.length,
        }
      : null;

    res.json({
      windowHours: 24,
      total,
      byCoin,
      buckets,
      bucketMs,
      hourlyThreshold,
      alert,
      fetchedAt: new Date(now).toISOString(),
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "anomaly-cancels query failed" });
  }
});

// Directional-call share history (task #101). Thin proxy so the dashboard
// doesn't talk to ml-engine directly. We surface the alert list verbatim
// so the UI can render its loud banner without re-deriving the threshold.
router.get("/crypto/quant-directional-history", async (req, res): Promise<void> => {
  try {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
    const qs = new URLSearchParams();
    if (typeof req.query.timeframe === "string") qs.set("timeframe", req.query.timeframe);
    if (typeof req.query.limit === "string") qs.set("limit", req.query.limit);
    const url = `${base}/ml/training/history/directional-call-share${qs.toString() ? `?${qs}` : ""}`;
    const r = await fetch(url);
    const text = await r.text();
    res.status(r.status).type("application/json").send(text);
  } catch (err) {
    res.status(503).json({ error: err instanceof Error ? err.message : "ml-engine unreachable" });
  }
});

// REMOVED 2026-04-24 (B-CALIB-CARD): the `/crypto/quant-calibration-history`
// proxy route has been retired alongside the dashboard card it served. The
// underlying `quant_calibration_history` table is unprovisioned in this
// environment and there is no source feeding it, so the route only ever
// returned an empty payload. Re-introduce only when the schema +
// ingestion are in place. Audit ref: docs/audits/2026-04-23-full-system-audit.md.

// Latest auto-generated failure-analysis pair (task #327). Thin proxy so
// the diagnostics page can render the bucket counts and the markdown
// summary written by `failure_analysis.generate_for_report` after every
// retrain.
router.get("/crypto/failure-analysis-latest", async (_req, res): Promise<void> => {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 8000);
  try {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
    const r = await fetch(`${base}/ml/admin/failure-analysis/latest`, { signal: ctrl.signal });
    const text = await r.text();
    const upstreamCt = r.headers.get("content-type");
    if (upstreamCt) res.type(upstreamCt);
    else res.type("application/json");
    res.status(r.status).send(text);
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    res
      .status(aborted ? 504 : 502)
      .json({ error: aborted ? "ml-engine timeout" : err instanceof Error ? err.message : "ml-engine unreachable" });
  } finally {
    clearTimeout(timer);
  }
});

// Failure-analysis history (task #336). Thin proxy so the diagnostics
// page can sparkline how each bucket trends across the last N retrains.
router.get("/crypto/failure-analysis-history", async (req, res): Promise<void> => {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 8000);
  try {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
    const qs = new URLSearchParams();
    if (typeof req.query.limit === "string") qs.set("limit", req.query.limit);
    const url = `${base}/ml/admin/failure-analysis/history${qs.toString() ? `?${qs}` : ""}`;
    const r = await fetch(url, { signal: ctrl.signal });
    const text = await r.text();
    const upstreamCt = r.headers.get("content-type");
    if (upstreamCt) res.type(upstreamCt);
    else res.type("application/json");
    res.status(r.status).send(text);
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    res
      .status(aborted ? 504 : 502)
      .json({ error: aborted ? "ml-engine timeout" : err instanceof Error ? err.message : "ml-engine unreachable" });
  } finally {
    clearTimeout(timer);
  }
});

// Latest training report (task #147). Thin proxy so the dashboard can read
// per-slice diagnostics — currently the gates-alignment share — without
// scraping the HTML report.
router.get("/crypto/quant-training-report", async (_req, res): Promise<void> => {
  try {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
    const r = await fetch(`${base}/ml/training/report`);
    const text = await r.text();
    res.status(r.status).type("application/json").send(text);
  } catch (err) {
    res.status(503).json({ error: err instanceof Error ? err.message : "ml-engine unreachable" });
  }
});

// Task #615 — surface the per-slice live-gated replay block from the
// most recent campaign's `phase7_summary.json`. Thin proxy so the
// dashboard can read the four-way verdict pill (bleeding/dormant/
// tradeable/inconclusive), loose-vs-live PnL, and dominant rejection
// reason without an operator opening the run folder by hand.
router.get(
  "/crypto/training/live-gated-replay",
  async (_req, res): Promise<void> => {
    try {
      const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
      const r = await fetch(`${base}/ml/training/live-gated-replay`);
      const text = await r.text();
      res.status(r.status).type("application/json").send(text);
    } catch (err) {
      res.status(503).json({
        error: err instanceof Error ? err.message : "ml-engine unreachable",
      });
    }
  },
);

// Live training-progress heartbeat (task #393). Thin proxy so the dashboard
// can show "currently fitting <coin>/<tf>" plus an N/total counter and a
// "no row in 5 min" stale-warning while a training campaign is running.
router.get("/crypto/training/progress", async (_req, res): Promise<void> => {
  try {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
    const r = await fetch(`${base}/ml/training/progress`);
    const text = await r.text();
    res.status(r.status).type("application/json").send(text);
  } catch (err) {
    res.status(503).json({ error: err instanceof Error ? err.message : "ml-engine unreachable" });
  }
});

// Dataset cache freshness + footprint (task #559). Thin proxy of the
// ml-engine `/ml/dataset-freshness` endpoint, which itself reads
// `models/datasets/_freshness_status.json`. Powers the dashboard
// freshness panel that surfaces per-tf cache size, total cache size,
// last-N size history (sparkline), and the > 5 GB soft warning chip.
router.get("/crypto/dataset-freshness", async (_req, res): Promise<void> => {
  try {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
    const r = await fetch(`${base}/ml/dataset-freshness`);
    const text = await r.text();
    res.status(r.status).type("application/json").send(text);
  } catch (err) {
    res.status(503).json({ error: err instanceof Error ? err.message : "ml-engine unreachable" });
  }
});

// Trainer's stale-threshold warning (task #358). Thin proxy so the dashboard
// can show a banner when a recent training run silently fell back to the
// hardcoded label thresholds because shared/trading-frictions.json was
// unreachable. The flag is set by app.training.labels._record_fallback (#357).
router.get("/crypto/threshold-fallback-status", async (_req, res): Promise<void> => {
  try {
    const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
    const r = await fetch(`${base}/ml/threshold-fallback-status`);
    const text = await r.text();
    res.status(r.status).type("application/json").send(text);
  } catch (err) {
    res.status(503).json({ error: err instanceof Error ? err.message : "ml-engine unreachable" });
  }
});

// Phase 1 — adaptive engine journals.
//
// GET  /crypto/journal-health: write rate, resolution coverage, MAE/MFE
//                              coverage. Powers the diagnostics card.
// POST /crypto/journal-backfill: idempotently copy legacy predictions /
//                                paper_trades rows into the new journals so
//                                dashboards aren't empty on day one. Admin
//                                gated.
router.get("/crypto/journal-health", async (req, res): Promise<void> => {
  try {
    const hours = Math.min(168, Math.max(1, Number(req.query.hours ?? 24)));
    const health = await getJournalHealth(hours);
    res.json(health);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "journal-health failed" });
  }
});

// GET /crypto/journal-health-series: bucketed time-series companion that
// powers the diagnostics card sparkline. Bucket size adapts to the window
// (5m for 1h, 1h for 24h, 6h for 7d) so the point count stays sane.
router.get("/crypto/journal-health-series", async (req, res): Promise<void> => {
  try {
    const hours = Math.min(168, Math.max(1, Number(req.query.hours ?? 24)));
    let bucketSeconds: number;
    if (hours <= 1) bucketSeconds = 5 * 60;
    else if (hours <= 24) bucketSeconds = 60 * 60;
    else bucketSeconds = 6 * 60 * 60;
    const series = await getJournalHealthSeries(hours, bucketSeconds);
    res.json(series);
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : "journal-health-series failed",
    });
  }
});

router.post("/crypto/journal-backfill", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const counts = await backfillJournals();
    res.json({ ok: true, ...counts });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "backfill failed" });
  }
});

// POST /crypto/journal-backfill/quant-feature-hash — Task #469.
//
// One-shot backfill that re-walks the legacy `predictions` table for
// QUANT-source rows silently dropped from `prediction_journal` between the
// Task #406 brain flip and the Task #460 fix. For each missing row, fetches
// the canonical feature hash + vector from the Python ml-engine
// `/ml/features` (cached per coin/timeframe so the call count stays bounded)
// and inserts a journal row preserving the original `created_at`.
//
// Idempotent — re-running on a healthy journal is a no-op. Bounded by the
// optional `?sinceMs=`, `?untilMs=`, and `?limit=` query params so an
// operator can replay a specific window without scanning the whole history.
// Admin gated.
router.post("/crypto/journal-backfill/quant-feature-hash", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const sinceMs = req.query.sinceMs ? Number(req.query.sinceMs) : null;
    const untilMs = req.query.untilMs ? Number(req.query.untilMs) : null;
    const limit = req.query.limit ? Number(req.query.limit) : undefined;
    const result = await backfillMissingQuantJournals({
      since: sinceMs && Number.isFinite(sinceMs) ? new Date(sinceMs) : undefined,
      until: untilMs && Number.isFinite(untilMs) ? new Date(untilMs) : undefined,
      limit: limit !== undefined && Number.isFinite(limit) ? limit : undefined,
    });
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : "quant-feature-hash backfill failed",
    });
  }
});

// POST /crypto/journal/backtest-batch: ingestion seam for the Python
// backtester in artifacts/ml-engine/app/backtest. Each row lands in
// prediction_journal with brain="BACKTEST" so simulated decisions sit
// alongside live ones for replay / counterfactual analysis but stay
// trivially filterable. Admin gated.
// Phase 2 — backfill regime label on historical price_history rows.
// Walks rows with NULL regime in time-bucketed batches, computes the
// basket regime via the canonical Python ml-engine
// (/ml/regime/basket) for each bucket using the basket of 24h % changes
// reconstructed from price_history itself, and stamps the resulting
// 6-class label on every coin row that falls in the bucket. Admin
// gated. Idempotent — re-running only touches rows still NULL.
router.post("/crypto/diagnostics/regime/backfill", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const bucketMinutes = Math.min(60, Math.max(5, Number(req.query.bucketMinutes ?? 15)));
    const maxBuckets = Math.min(2000, Math.max(1, Number(req.query.maxBuckets ?? 200)));

    // True 24h basket returns: for each (coin, bucket) join the same
    // coin's bucket from ~24h prior and compute the % change. The
    // canonical classifier expects 24h-percent changes (matching the
    // live `priceChange24h` field), NOT bucket-to-bucket returns.
    type RawRow = { coin_id: string; bucket: Date; price: number; price24hAgo: number | null };
    const bucketSql = sql`
      date_trunc('hour', timestamp)
        + floor(extract(minute from timestamp) / ${bucketMinutes}) * (${bucketMinutes}::text || ' minutes')::interval
    `;
    // Restrict to buckets that still contain at least one NULL `regime`
    // row. Without this filter, repeated calls re-process the oldest
    // already-labeled buckets and never reach later NULL ranges when
    // the bucket count exceeds `maxBuckets`. With the filter, each call
    // makes forward progress regardless of pagination size.
    const raw = await db.execute<RawRow>(sql`
      WITH pending AS (
        SELECT DISTINCT ${bucketSql} AS bucket
        FROM ${priceHistoryTable}
        WHERE timestamp >= NOW() - INTERVAL '30 days'
          AND regime IS NULL
      ),
      bucketed AS (
        SELECT
          coin_id,
          ${bucketSql} AS bucket,
          AVG(price) AS price
        FROM ${priceHistoryTable}
        WHERE timestamp >= NOW() - INTERVAL '32 days'
        GROUP BY 1, 2
      )
      SELECT
        b.coin_id,
        b.bucket,
        b.price,
        prev.price AS "price24hAgo"
      FROM bucketed b
      INNER JOIN pending pn ON pn.bucket = b.bucket
      LEFT JOIN LATERAL (
        SELECT price
        FROM bucketed p
        WHERE p.coin_id = b.coin_id
          AND p.bucket <= b.bucket - INTERVAL '24 hours'
          AND p.bucket >= b.bucket - INTERVAL '26 hours'
        ORDER BY p.bucket DESC
        LIMIT 1
      ) prev ON TRUE
      ORDER BY b.bucket ASC, b.coin_id ASC
    `);
    const rows = (raw as unknown as { rows: RawRow[] }).rows ?? [];

    const buckets = new Map<number, { coins: string[]; changes: number[] }>();
    for (const r of rows) {
      if (r.price24hAgo == null || r.price24hAgo <= 0) continue;
      const ts = new Date(r.bucket).getTime();
      const pctChange = ((r.price - r.price24hAgo) / r.price24hAgo) * 100;
      if (!buckets.has(ts)) buckets.set(ts, { coins: [], changes: [] });
      const b = buckets.get(ts)!;
      b.coins.push(r.coin_id);
      b.changes.push(pctChange);
    }

    const sortedBuckets = [...buckets.entries()]
      .sort((a, b) => a[0] - b[0])
      .slice(0, maxBuckets);

    const { getMlBasketRegime } = await import("../../lib/ml-client");

    let labeled = 0;
    let bucketsProcessed = 0;
    let bucketsFailed = 0;
    for (const [ts, b] of sortedBuckets) {
      if (b.changes.length < 3) continue;
      const avg = b.changes.reduce((s, c) => s + c, 0) / b.changes.length;
      const variance =
        b.changes.reduce((s, c) => s + (c - avg) ** 2, 0) / Math.max(1, b.changes.length - 1);
      const cross = Math.sqrt(variance);

      let label: string;
      try {
        const resp = await getMlBasketRegime(b.changes, cross);
        label = resp.label;
      } catch {
        bucketsFailed++;
        continue;
      }
      bucketsProcessed++;

      const start = new Date(ts);
      const end = new Date(ts + bucketMinutes * 60 * 1000);
      const upd = await db.execute(sql`
        UPDATE ${priceHistoryTable}
        SET regime = ${label}
        WHERE regime IS NULL
          AND timestamp >= ${start}
          AND timestamp < ${end}
      `);
      labeled += (upd as unknown as { rowCount?: number }).rowCount ?? 0;
    }

    // Phase 2 — once price_history is freshly labeled, sweep
    // trade_journal rows whose regime_label is still NULL and stamp them
    // with the surrounding price_history regime for the same coin. We
    // pick the price_history row nearest in time to entry_time within a
    // ±30-minute window so a brief gap in candles doesn't strand the
    // trade. Idempotent — only NULL rows are touched.
    let tradesLabeled = 0;
    let tradeBackfillError: string | null = null;
    try {
      const upd = await db.execute(sql`
        WITH matched AS (
          SELECT tj.id, (
            SELECT p.regime
            FROM ${priceHistoryTable} p
            WHERE p.coin_id = tj.coin_id
              AND p.regime IS NOT NULL
              AND p.timestamp BETWEEN tj.entry_time - INTERVAL '30 minutes'
                                  AND tj.entry_time + INTERVAL '30 minutes'
            ORDER BY ABS(EXTRACT(EPOCH FROM (p.timestamp - tj.entry_time)))
            LIMIT 1
          ) AS regime
          FROM ${tradeJournalTable} tj
          WHERE tj.regime_label IS NULL
        )
        UPDATE ${tradeJournalTable} tj
        SET regime_label = m.regime
        FROM matched m
        WHERE m.id = tj.id AND m.regime IS NOT NULL
      `);
      tradesLabeled = (upd as unknown as { rowCount?: number }).rowCount ?? 0;
    } catch (err) {
      // Non-fatal: surface the failure on the response (and to stderr)
      // so the operator can see partial success without losing the
      // price_history progress already reported.
      tradeBackfillError = err instanceof Error ? err.message : String(err);
      console.warn("regime backfill: trade_journal sweep failed", tradeBackfillError);
    }

    res.json({
      ok: true,
      bucketMinutes,
      bucketsProcessed,
      bucketsFailed,
      bucketsConsidered: sortedBuckets.length,
      rowsLabeled: labeled,
      tradesLabeled,
      tradeBackfillError,
    });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : "regime backfill failed",
    });
  }
});

// Phase 2 — regime distribution diagnostics. Returns rolling per-label
// counts over multiple windows (1h, 24h, 7d) over price_history.regime,
// plus per-regime trade counts from trade_journal in the same windows,
// all compared against a 30-day baseline so an operator can spot drift
// (e.g. `panic_liquidation` going silent after a threshold tweak).
// Powers the diagnostics page RegimeDistributionCard.
router.get("/crypto/diagnostics/regime", async (_req, res): Promise<void> => {
  try {
    const now = new Date();
    const WINDOWS: { key: string; label: string; hours: number }[] = [
      { key: "1h", label: "Last hour", hours: 1 },
      { key: "24h", label: "Last 24h", hours: 24 },
      { key: "7d", label: "Last 7d", hours: 24 * 7 },
    ];
    const baselineHours = 24 * 30;
    const since = (h: number) => new Date(now.getTime() - h * 60 * 60 * 1000);

    const REGIMES = [
      "trending_up",
      "trending_down",
      "range_chop",
      "high_vol_breakout",
      "low_vol_compression",
      "panic_liquidation",
    ];

    type CountRow = { regime: string | null; count: number };
    const candleQuery = (h: number) =>
      db
        .select({
          regime: priceHistoryTable.regime,
          count: sql<number>`count(*)`,
        })
        .from(priceHistoryTable)
        .where(gte(priceHistoryTable.timestamp, since(h)))
        .groupBy(priceHistoryTable.regime);
    const tradeQuery = (h: number) =>
      db
        .select({
          regime: tradeJournalTable.regimeLabel,
          count: sql<number>`count(*)`,
        })
        .from(tradeJournalTable)
        .where(gte(tradeJournalTable.createdAt, since(h)))
        .groupBy(tradeJournalTable.regimeLabel);

    const baselineHoursList = [baselineHours];
    const allCandle = await Promise.all(
      [...WINDOWS.map((w) => w.hours), ...baselineHoursList].map(candleQuery),
    );
    const allTrade = await Promise.all(WINDOWS.map((w) => tradeQuery(w.hours)));
    const baselineRows: CountRow[] = allCandle[allCandle.length - 1] as CountRow[];

    const lookup = (rows: CountRow[]) => {
      const m = new Map<string, number>();
      for (const r of rows) {
        const key = r.regime ?? "unlabeled";
        m.set(key, (m.get(key) ?? 0) + Number(r.count ?? 0));
      }
      return m;
    };
    const sumCounts = (rows: CountRow[]) =>
      rows.reduce((s, r) => s + Number(r.count ?? 0), 0);

    const baselineMap = lookup(baselineRows);
    const totalBaseline = sumCounts(baselineRows);
    const unlabeledBaseline = baselineMap.get("unlabeled") ?? 0;
    const baselineDistribution = REGIMES.map((label) => {
      const c = baselineMap.get(label) ?? 0;
      return {
        label,
        count: c,
        pct: totalBaseline > 0 ? (c / totalBaseline) * 100 : 0,
      };
    });

    const windows = WINDOWS.map((w, i) => {
      const candleRows = allCandle[i] as CountRow[];
      const tradeRows = allTrade[i] as CountRow[];
      const candleMap = lookup(candleRows);
      const tradeMap = lookup(tradeRows);
      const totalCandles = sumCounts(candleRows);
      const totalTrades = sumCounts(tradeRows);
      const unlabeled = candleMap.get("unlabeled") ?? 0;
      const distribution = REGIMES.map((label) => {
        const c = candleMap.get(label) ?? 0;
        const baselineCount = baselineMap.get(label) ?? 0;
        const baselinePct =
          totalBaseline > 0 ? (baselineCount / totalBaseline) * 100 : 0;
        const pct = totalCandles > 0 ? (c / totalCandles) * 100 : 0;
        return {
          label,
          count: c,
          pct,
          baselineCount,
          baselinePct,
          driftPct: pct - baselinePct,
          tradeCount: tradeMap.get(label) ?? 0,
        };
      });
      return {
        key: w.key,
        label: w.label,
        hours: w.hours,
        totals: {
          candles: totalCandles,
          trades: totalTrades,
          unlabeledPct: totalCandles > 0 ? (unlabeled / totalCandles) * 100 : 0,
        },
        distribution,
      };
    });

    // Legacy fields preserved for any older client still pinned to the
    // 24h-only response shape.
    const w24 = windows.find((w) => w.key === "24h")!;
    res.json({
      windows,
      baseline: {
        days: 30,
        totals: {
          candles: totalBaseline,
          unlabeledPct:
            totalBaseline > 0 ? (unlabeledBaseline / totalBaseline) * 100 : 0,
        },
        distribution: baselineDistribution,
      },
      labels: REGIMES,
      windowHours: 24,
      baselineDays: 30,
      distribution: w24.distribution.map((d) => ({
        label: d.label,
        todayCount: d.count,
        todayPct: d.pct,
        baselineCount: d.baselineCount,
        baselinePct: d.baselinePct,
        tradeCount: d.tradeCount,
      })),
      totals: {
        todayCandles: w24.totals.candles,
        baselineCandles: totalBaseline,
        todayTrades: w24.totals.trades,
        unlabeledTodayPct: w24.totals.unlabeledPct,
        unlabeledBaselinePct:
          totalBaseline > 0 ? (unlabeledBaseline / totalBaseline) * 100 : 0,
      },
    });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : "regime diagnostics failed",
    });
  }
});

router.post("/crypto/journal/backtest-batch", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const body = req.body as { rows?: unknown };
    const raw = Array.isArray(body?.rows) ? body.rows : [];
    const rows: (InsertBacktestJournalArgs & { simulatedTrade?: BacktestSimulatedTrade | null })[] = [];
    for (const r of raw) {
      if (!r || typeof r !== "object") continue;
      const row = r as Record<string, unknown>;
      if (typeof row.coinId !== "string" || typeof row.timeframe !== "string") continue;
      if (typeof row.priceAtPrediction !== "number") continue;
      const dir = row.direction;
      if (dir !== "up" && dir !== "down" && dir !== "stable") continue;
      let simulatedTrade: BacktestSimulatedTrade | null = null;
      const st = row.simulatedTrade;
      if (st && typeof st === "object") {
        const s = st as Record<string, unknown>;
        if (
          typeof s.entryTime === "string" &&
          typeof s.exitTime === "string" &&
          typeof s.entryPriceAdj === "number" &&
          typeof s.exitPriceAdj === "number" &&
          typeof s.positionSizeUsd === "number" &&
          typeof s.exitReason === "string" &&
          typeof s.realizedPnlUsd === "number" &&
          typeof s.realizedPnlPct === "number"
        ) {
          simulatedTrade = {
            entryTime: new Date(s.entryTime),
            exitTime: new Date(s.exitTime),
            entryPriceRaw: (s.entryPriceRaw as number | null) ?? null,
            entryPriceAdj: s.entryPriceAdj,
            exitPriceRaw: (s.exitPriceRaw as number | null) ?? null,
            exitPriceAdj: s.exitPriceAdj,
            entryFee: (s.entryFee as number | null) ?? null,
            exitFee: (s.exitFee as number | null) ?? null,
            slippagePct: (s.slippagePct as number | null) ?? null,
            positionSizeUsd: s.positionSizeUsd,
            mfePct: (s.mfePct as number | null) ?? null,
            maePct: (s.maePct as number | null) ?? null,
            exitReason: s.exitReason,
            realizedPnlUsd: s.realizedPnlUsd,
            realizedPnlPct: s.realizedPnlPct,
          };
        }
      }
      rows.push({
        simulatedTrade,
        coinId: row.coinId,
        timeframe: row.timeframe,
        modelId: (row.modelId as string | null) ?? null,
        modelVersion: (row.modelVersion as string | null) ?? null,
        featureHash: (row.featureHash as string | null) ?? null,
        featureVector: (row.featureVector as Record<string, number> | null) ?? null,
        direction: dir,
        confidence: typeof row.confidence === "number" ? row.confidence : 0,
        probUp: (row.probUp as number | null) ?? null,
        probDown: (row.probDown as number | null) ?? null,
        probStable: (row.probStable as number | null) ?? null,
        expectedReturnPct: (row.expectedReturnPct as number | null) ?? null,
        predictionStdPct: (row.predictionStdPct as number | null) ?? null,
        priceAtPrediction: row.priceAtPrediction,
        predictedPrice: (row.predictedPrice as number | null) ?? null,
        actualPrice: (row.actualPrice as number | null) ?? null,
        realizedReturnPct: (row.realizedReturnPct as number | null) ?? null,
        outcome: (row.outcome as string | null) ?? null,
        resolvesAt: row.resolvesAt ? new Date(row.resolvesAt as string) : null,
        resolvedAt: row.resolvedAt ? new Date(row.resolvedAt as string) : null,
        // Sanitize: legacy callers may still tuck `regime` /
        // `specialists` into gatesApplied. Pull them out and route to
        // the dedicated columns so gates_applied stays focused on
        // actual gating decisions.
        gatesApplied: (() => {
          const g = { ...((row.gatesApplied as Record<string, unknown>) ?? {}) };
          delete g.regime;
          delete g.specialists;
          return g as Record<string, boolean | undefined>;
        })(),
        specialistScores:
          row.specialistScores ??
          (row.gatesApplied && typeof row.gatesApplied === "object"
            ? (row.gatesApplied as Record<string, unknown>).specialists ?? null
            : null),
        regimeLabel:
          (row.regimeLabel as string | null) ??
          (row.gatesApplied && typeof row.gatesApplied === "object"
            ? ((row.gatesApplied as Record<string, unknown>).regime as
                | string
                | undefined) ?? null
            : null),
      });
    }
    const inserted = await writeBacktestJournalRows(rows);
    res.json({ ok: true, accepted: rows.length, inserted });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "backtest-batch failed" });
  }
});

// ───────────────────────────────────────────────────────────────────────
// Phase 4 — meta-model diagnostics.
//
// Three small endpoints that power the MetaModelDiagnosticsCard on the
// diagnostics page. They read the prediction_journal + skip_events
// tables directly so they keep working even when ml-engine is down.
// ───────────────────────────────────────────────────────────────────────

// GET /crypto/meta/action-distribution
// Counts of meta action (long / short / no_trade) by regime, derived from
// prediction_journal rows where brain=QUANT in the last N hours. The
// meta action is read from gates_applied.meta_action (written by the
// quant brain alongside the legacy gate flags).
router.get("/crypto/meta/action-distribution", async (req, res): Promise<void> => {
  try {
    const hours = Math.min(168, Math.max(1, Number(req.query.hours ?? 24)));
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);
    const rows = await db
      .select({
        regime: predictionJournalTable.regimeLabel,
        gates: predictionJournalTable.gatesApplied,
        direction: predictionJournalTable.direction,
        becameTrade: predictionJournalTable.becameTrade,
      })
      .from(predictionJournalTable)
      .where(and(eq(predictionJournalTable.brain, "QUANT"), gte(predictionJournalTable.createdAt, since)));
    const out: Record<string, { long: number; short: number; no_trade: number }> = {};
    // Task #455 — also surface counts split by which meta-model produced
    // the action (heuristic vs lightgbm) and by trained version, so the
    // diagnostics card can show whether the served meta is the heuristic
    // baseline or a promoted candidate. Reads gates_applied.meta_kind
    // and gates_applied.meta_version (already written by the quant brain
    // alongside meta_action — no schema change required).
    const byMetaKind: Record<string, { long: number; short: number; no_trade: number }> = {};
    const byMetaVersion: Record<string, { long: number; short: number; no_trade: number }> = {};
    let total = 0;
    for (const r of rows) {
      const regime = (r.regime as string | null) ?? "unknown";
      const gates = (r.gates as Record<string, unknown> | null) ?? {};
      let action: "long" | "short" | "no_trade";
      const ma = gates.meta_action as string | undefined;
      if (ma === "long" || ma === "short" || ma === "no_trade") {
        action = ma;
      } else {
        // Fall back to direction so the card still renders meaningful
        // counts on rows journaled before /ml/meta/predict was wired.
        action = r.direction === "up" ? "long" : r.direction === "down" ? "short" : "no_trade";
      }
      if (!out[regime]) out[regime] = { long: 0, short: 0, no_trade: 0 };
      out[regime][action] += 1;
      const mk = typeof gates.meta_kind === "string" && gates.meta_kind.length > 0 ? (gates.meta_kind as string) : "unknown";
      if (!byMetaKind[mk]) byMetaKind[mk] = { long: 0, short: 0, no_trade: 0 };
      byMetaKind[mk][action] += 1;
      const mv = typeof gates.meta_version === "string" && gates.meta_version.length > 0 ? (gates.meta_version as string) : "unversioned";
      if (!byMetaVersion[mv]) byMetaVersion[mv] = { long: 0, short: 0, no_trade: 0 };
      byMetaVersion[mv][action] += 1;
      total += 1;
    }
    res.json({ hours, total, byRegime: out, byMetaKind, byMetaVersion });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "action-distribution failed" });
  }
});

// GET /crypto/meta/abstain-reasons
// Counts of skip_events whose reason is in the meta_no_trade_* family
// over the last N hours. Powers the "why did meta abstain" pie/bar.
router.get("/crypto/meta/abstain-reasons", async (req, res): Promise<void> => {
  try {
    const hours = Math.min(168, Math.max(1, Number(req.query.hours ?? 24)));
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);
    const metaReasons: SkipReason[] = [
      "meta_no_trade_low_edge",
      "meta_no_trade_bad_regime_fit",
      "meta_no_trade_specialist_disagreement",
      "meta_no_trade_low_calibration",
    ];
    const rows = await db
      .select({ reason: skipEventsTable.reason })
      .from(skipEventsTable)
      .where(and(gte(skipEventsTable.ts, since), inArray(skipEventsTable.reason, metaReasons)));
    const counts: Record<string, number> = {
      meta_no_trade_low_edge: 0,
      meta_no_trade_bad_regime_fit: 0,
      meta_no_trade_specialist_disagreement: 0,
      meta_no_trade_low_calibration: 0,
    };
    for (const r of rows) {
      const k = r.reason as string;
      if (k in counts) counts[k] += 1;
    }
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    res.json({ hours, total, counts });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "abstain-reasons failed" });
  }
});

// GET /crypto/meta/edge-deciles
// Realized vs predicted edge by decile of the meta-model's expected
// edge, computed on resolved prediction_journal rows where the trade
// was actually opened. realized = sign-adjusted realizedReturnPct minus
// the round-trip cost; predicted = gates_applied.meta_expected_edge_pct.
router.get("/crypto/meta/edge-deciles", async (req, res): Promise<void> => {
  try {
    const hours = Math.min(7 * 24 * 4, Math.max(24, Number(req.query.hours ?? 7 * 24)));
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);
    // Task #343 — was a local 0.3 literal that drifted out of step with
    // trading-constants. Use the shared `ROUND_TRIP_COST_PERCENT` (note
    // the _PERCENT suffix; ROUND_TRIP_COST_PCT is a fraction). Unit
    // test asserts parity in test/edge-deciles-cost-parity.test.ts.
    const rows = await db
      .select({
        gates: predictionJournalTable.gatesApplied,
        direction: predictionJournalTable.direction,
        realizedReturnPct: predictionJournalTable.realizedReturnPct,
        becameTrade: predictionJournalTable.becameTrade,
      })
      .from(predictionJournalTable)
      .where(
        and(
          eq(predictionJournalTable.brain, "QUANT"),
          eq(predictionJournalTable.becameTrade, true),
          gte(predictionJournalTable.createdAt, since),
        ),
      );
    const samples: Array<{ predicted: number; realized: number }> = [];
    for (const r of rows) {
      const gates = (r.gates as Record<string, unknown> | null) ?? {};
      const predicted = Number(gates.meta_expected_edge_pct);
      if (!Number.isFinite(predicted)) continue;
      const realRaw = r.realizedReturnPct;
      if (realRaw === null || realRaw === undefined) continue;
      const signed = r.direction === "down" ? -Number(realRaw) : Number(realRaw);
      const realized = signed - ROUND_TRIP_COST_PERCENT;
      samples.push({ predicted, realized });
    }
    samples.sort((a, b) => a.predicted - b.predicted);
    const decileCount = 10;
    const deciles: Array<{ decile: number; n: number; predictedAvg: number; realizedAvg: number }> = [];
    if (samples.length > 0) {
      const per = Math.max(1, Math.floor(samples.length / decileCount));
      for (let i = 0; i < decileCount; i++) {
        const start = i * per;
        const end = i === decileCount - 1 ? samples.length : start + per;
        const slice = samples.slice(start, end);
        if (slice.length === 0) continue;
        const pAvg = slice.reduce((s, x) => s + x.predicted, 0) / slice.length;
        const rAvg = slice.reduce((s, x) => s + x.realized, 0) / slice.length;
        deciles.push({ decile: i + 1, n: slice.length, predictedAvg: pAvg, realizedAvg: rAvg });
      }
    }
    res.json({ hours, totalSamples: samples.length, deciles });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "edge-deciles failed" });
  }
});

// ── Phase 5 — Model registry lifecycle ───────────────────────────────────
// Read endpoints are public (just inspect state); write endpoints
// (register, promote, rollback, set-state) require X-Admin-Key.
import {
  listRegistry as _listRegistry,
  registerModel as _registerModel,
  dryRunPromotion as _dryRunPromotion,
  promoteToChampion as _promoteToChampion,
  rollbackChampion as _rollbackChampion,
  setRegistryState as _setRegistryState,
} from "../../lib/model-registry";

router.get("/crypto/model-registry", async (_req, res): Promise<void> => {
  try {
    const rows = await _listRegistry();
    // Task #233 — augment with the resolver's *effective* served version per
    // (coin, timeframe) so the dashboard can flag slots being served by an
    // older fallback because the latest champion is quarantined. ml-engine
    // owns the resolver logic; we just consume.
    type EffectiveServingSlot = {
      coinId: string;
      timeframe: string;
      latestVersion: string | null;
      servedCoinId: string | null;
      servedVersion: string | null;
      fallback: boolean;
      fallbackReason: string | null;
      quarantinedVersions: string[];
    };
    let effectiveServing: EffectiveServingSlot[] = [];
    try {
      const mlBase = process.env.ML_ENGINE_URL || "http://localhost:8000";
      const r = await fetch(`${mlBase}/ml/registry/effective-serving`);
      if (r.ok) {
        const j = await r.json() as Record<string, unknown>;
        const slots = j.slots;
        if (Array.isArray(slots)) effectiveServing = slots as EffectiveServingSlot[];
      }
    } catch {
      // best-effort: a transient ml-engine blip should not break the panel
    }
    res.json({ rows, effectiveServing });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "registry list failed" });
  }
});

router.get("/crypto/model-registry/:id/promotion-preview", async (req, res): Promise<void> => {
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "invalid id" });
    return;
  }
  try {
    const out = await _dryRunPromotion(id);
    res.json(out);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "preview failed" });
  }
});

router.post("/crypto/model-registry", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const body = (req.body ?? {}) as Record<string, unknown>;
    if (typeof body.modelId !== "string" || typeof body.modelVersion !== "string") {
      res.status(400).json({ error: "modelId and modelVersion required" });
      return;
    }
    const row = await _registerModel({
      modelId: body.modelId,
      modelVersion: body.modelVersion,
      coinId: typeof body.coinId === "string" ? body.coinId : undefined,
      timeframe: typeof body.timeframe === "string" ? body.timeframe : undefined,
      state: (body.state as never) ?? "shadow",
      note: typeof body.note === "string" ? body.note : undefined,
    });
    res.status(201).json({ row });
  } catch (err) {
    res.status(400).json({ error: err instanceof Error ? err.message : "register failed" });
  }
});

router.post("/crypto/model-registry/:id/promote", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "invalid id" });
    return;
  }
  const body = (req.body ?? {}) as Record<string, unknown>;
  try {
    const row = await _promoteToChampion(id, {
      force: body.force === true,
      note: typeof body.note === "string" ? body.note : undefined,
    });
    res.json({ row });
  } catch (err) {
    const e = err as { code?: string; verdict?: unknown; message?: string };
    if (e.code === "PROMOTION_GATES_FAILED") {
      res.status(409).json({ error: e.message, verdict: e.verdict });
      return;
    }
    res.status(500).json({ error: e.message ?? "promote failed" });
  }
});

router.post("/crypto/model-registry/rollback", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const body = (req.body ?? {}) as Record<string, unknown>;
  try {
    const row = await _rollbackChampion(
      {
        modelId: typeof body.modelId === "string" ? body.modelId : undefined,
        coinId: typeof body.coinId === "string" ? body.coinId : undefined,
        timeframe: typeof body.timeframe === "string" ? body.timeframe : undefined,
      },
      { note: typeof body.note === "string" ? body.note : undefined },
    );
    res.json({ row });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "rollback failed" });
  }
});

router.post("/crypto/model-registry/:id/state", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "invalid id" });
    return;
  }
  const body = (req.body ?? {}) as Record<string, unknown>;
  if (typeof body.state !== "string") {
    res.status(400).json({ error: "state required" });
    return;
  }
  try {
    const row = await _setRegistryState(
      id,
      body.state as never,
      typeof body.note === "string" ? body.note : undefined,
    );
    res.json({ row });
  } catch (err) {
    res.status(400).json({ error: err instanceof Error ? err.message : "set-state failed" });
  }
});

/**
 * GET /crypto/portfolio-risk
 *
 * Live snapshot of the Phase 5 portfolio constraints across the active ai-bot
 * fleet. Aggregates every open paper position, groups by sector and regime,
 * and reports current exposure vs the caps in
 * `shared/trading-frictions.json:portfolio_constraints`.
 *
 * The dashboard card refreshes this every 15s so an operator can see when
 * the next trade is about to be blocked (and which gate would block it)
 * before it fires as a typed `portfolio_*` skip reason.
 */
router.get("/crypto/portfolio-risk", async (_req, res): Promise<void> => {
  const cfg = PORTFOLIO_CONSTRAINTS_CONFIG;
  const aiAgents = await db
    .select({ id: agentsTable.id, name: agentsTable.name })
    .from(agentsTable)
    .where(and(eq(agentsTable.strategyType, "ai-bots"), eq(agentsTable.isActive, true)));
  const aiAgentIds = new Set(aiAgents.map((a) => a.id));
  const agentNameById = new Map(aiAgents.map((a) => [a.id, a.name]));

  const portfoliosAll = await db.select().from(paperPortfoliosTable);
  const portfolios = portfoliosAll.filter((p) => aiAgentIds.has(p.agentId));
  const equityUsd = portfolios.reduce((s, p) => s + Number(p.totalValue ?? 0), 0);

  const positionsAll = await db.select().from(paperPositionsTable);
  const positions = positionsAll.filter((pos) => aiAgentIds.has(pos.agentId));

  type RowPos = {
    coinId: string;
    coinName: string;
    agentName: string;
    direction: string;
    notionalUsd: number;
    regimeAtEntry: string | null;
  };

  const rows: RowPos[] = positions.map((pos) => ({
    coinId: pos.coinId,
    coinName: pos.coinName,
    agentName: agentNameById.get(pos.agentId) ?? pos.agentName ?? "unknown",
    direction: pos.direction,
    notionalUsd: Number(pos.positionSize ?? 0),
    regimeAtEntry: pos.entryRegimeLabel ?? null,
  }));

  const totalNotionalUsd = rows.reduce((s, r) => s + r.notionalUsd, 0);
  const safeEquity = equityUsd > 0 ? equityUsd : 1e-9;

  // Sectors
  const sectorMap = new Map<string, RowPos[]>();
  for (const r of rows) {
    const sector = getSectorForCoin(r.coinId);
    const arr = sectorMap.get(sector) ?? [];
    arr.push(r);
    sectorMap.set(sector, arr);
  }
  const sectors = Array.from(sectorMap.entries())
    .map(([sector, sectorRows]) => {
      const notionalUsd = sectorRows.reduce((s, r) => s + r.notionalUsd, 0);
      const sharePct = notionalUsd / safeEquity;
      const distinctCoins = new Set(sectorRows.map((r) => r.coinId)).size;
      const cap =
        distinctCoins >= 2
          ? Math.min(cfg.max_sector_exposure_pct, cfg.max_correlated_exposure_pct)
          : cfg.max_sector_exposure_pct;
      return {
        sector,
        notionalUsd,
        sharePct,
        cap,
        distinctCoins,
        positions: sectorRows
          .slice()
          .sort((a, b) => b.notionalUsd - a.notionalUsd),
      };
    })
    .sort((a, b) => b.sharePct - a.sharePct);

  // Regimes
  const regimeMap = new Map<string, RowPos[]>();
  for (const r of rows) {
    const k = r.regimeAtEntry ?? "unknown";
    const arr = regimeMap.get(k) ?? [];
    arr.push(r);
    regimeMap.set(k, arr);
  }
  const regimes = Array.from(regimeMap.entries())
    .map(([regime, regimeRows]) => {
      const notionalUsd = regimeRows.reduce((s, r) => s + r.notionalUsd, 0);
      return {
        regime,
        notionalUsd,
        sharePct: notionalUsd / safeEquity,
        cap: cfg.regime_budget_pct,
        positions: regimeRows
          .slice()
          .sort((a, b) => b.notionalUsd - a.notionalUsd),
      };
    })
    .sort((a, b) => b.sharePct - a.sharePct);

  // Book beta — notional-weighted with default 1.0 since per-position beta
  // isn't tracked yet (matches checkPortfolioConstraints which uses 1.0 when
  // betaToBtc is null).
  let bookBeta: number | null = null;
  if (totalNotionalUsd > 0) {
    bookBeta = rows.reduce((s, r) => s + 1.0 * r.notionalUsd, 0) / totalNotionalUsd;
  }

  res.json({
    enabled: cfg?.enabled ?? false,
    caps: {
      maxSectorExposurePct: cfg?.max_sector_exposure_pct ?? null,
      maxCorrelatedExposurePct: cfg?.max_correlated_exposure_pct ?? null,
      maxBetaToBtc: cfg?.max_beta_to_btc ?? null,
      regimeBudgetPct: cfg?.regime_budget_pct ?? null,
    },
    equityUsd,
    totalNotionalUsd,
    bookBeta,
    bookBetaSource: "default-1.0", // per-position beta not tracked yet
    openPositionCount: rows.length,
    sectors,
    regimes,
    fetchedAt: new Date().toISOString(),
  });
});

// ── Phase 6 — Diagnostics, drift, quarantine, feature lab ───────────────
import { getPnlBreakdown as _getPnlBreakdown } from "../../lib/diagnostics-pnl";
import {
  computeCalibrationDrift as _computeCalibrationDrift,
  computeDistributionDrift as _computeDistributionDrift,
  computeFeatureDrift as _computeFeatureDrift,
  getRecentDriftSnapshots as _getRecentDriftSnapshots,
} from "../../lib/drift-tracker";
import {
  runQuarantineSweep as _runQuarantineSweep,
  listRecentQuarantineEvents as _listRecentQuarantineEvents,
} from "../../lib/auto-quarantine";
import {
  recomputeSlotStarvationAlerts as _recomputeSlotStarvationAlerts,
  getActiveSlotStarvationAlerts as _getActiveSlotStarvationAlerts,
} from "../../lib/quarantine-slot-alerts";
import {
  listCandidates as _listFlCandidates,
  createCandidate as _createFlCandidate,
  listReports as _listFlReports,
  runAblation as _runFlAblation,
  evaluatePromotion as _evalFlPromotion,
  approveCandidate as _approveFlCandidate,
  summarizeApprovedFeatureApplications as _summarizeFlApplied,
  rejectCandidate as _rejectFlCandidate,
  unquarantineCandidate as _unquarantineFlCandidate,
  assessUnquarantineRegression as _assessUnquarantineRegression,
  UnquarantineRegressionStillPresentError as _UnquarantineRegressionStillPresentError,
  getApprovedFeatures as _getApprovedFlFeatures,
  getQuarantinedFeatures as _getQuarantinedFlFeatures,
  listUnquarantineEvents as _listFlUnquarantineEvents,
  summarizeUnquarantineOverrides as _summarizeFlUnquarantineOverrides,
} from "../../lib/feature-lab";
import {
  dispatchAutoRetireNotifications as _dispatchAutoRetireNotifications,
} from "../../lib/auto-retire-notifier";
import { FEATURE_TRANSFORM_KINDS } from "@workspace/db";
import {
  assembleBenchmarkTelemetry as _assembleMetaBrainBenchmark,
  getCycleStats as _getMetaBrainCycleStats,
} from "../../lib/meta-brain";

router.get("/crypto/diagnostics/pnl-breakdown", async (req, res): Promise<void> => {
  try {
    const windowHours = Math.max(1, Math.min(24 * 90, Number(req.query.windowHours ?? 168) || 168));
    res.json(await _getPnlBreakdown(windowHours));
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "pnl-breakdown failed" });
  }
});

router.get("/crypto/diagnostics/drift", async (req, res): Promise<void> => {
  try {
    const windowHours = Math.max(1, Math.min(24 * 90, Number(req.query.windowHours ?? 168) || 168));
    const registryId = req.query.registryId != null ? Number(req.query.registryId) : null;
    const [calibration, distribution, feature, calSnaps, featSnaps] = await Promise.all([
      _computeCalibrationDrift({ windowHours, registryId }),
      _computeDistributionDrift({ windowHours, registryId }),
      _computeFeatureDrift({ windowHours, registryId }),
      _getRecentDriftSnapshots("calibration", 100),
      _getRecentDriftSnapshots("feature", 100),
    ]);
    res.json({
      windowHours,
      registryId,
      calibration,
      distribution,
      feature,
      history: { calibration: calSnaps, feature: featSnaps },
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "drift failed" });
  }
});

router.get("/crypto/diagnostics/quarantine", async (req, res): Promise<void> => {
  try {
    const events = await _listRecentQuarantineEvents(50);
    // `?refreshAlerts=1` re-derives the slot-starvation list from the
    // registry (cheap read-only query). Otherwise we hand back the
    // cached snapshot from the last sweep so this stays a fast GET.
    const refresh = req.query.refreshAlerts === "1" || req.query.refreshAlerts === "true";
    const starvedSlots = refresh
      ? await _recomputeSlotStarvationAlerts()
      : _getActiveSlotStarvationAlerts();
    res.json({ events, starvedSlots });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "quarantine list failed" });
  }
});

router.post("/crypto/diagnostics/quarantine/run", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const dryRun = body.dryRun === true;
    const windowHours = typeof body.windowHours === "number" ? body.windowHours : 168;
    const out = await _runQuarantineSweep({ windowHours, dryRun });
    res.json(out);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "quarantine run failed" });
  }
});

// Task #258 — admin-triggered immediate dispatch of any new auto-retire
// notifications. The 60s background loop covers the steady state; this
// endpoint exists so an operator can force a flush after configuring a
// new webhook URL or to confirm wiring during setup.
router.post(
  "/crypto/feature-lab/auto-retire-alerts/dispatch",
  async (req, res): Promise<void> => {
    if (!requireAdminApiKey(req, res)) return;
    try {
      const out = await _dispatchAutoRetireNotifications();
      res.json(out);
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : "dispatch failed",
      });
    }
  },
);

router.get("/crypto/feature-lab/candidates", async (_req, res): Promise<void> => {
  try {
    const [candidates, approvedFeatures, quarantinedFeatures, unquarantineEvents] = await Promise.all([
      _listFlCandidates(),
      _getApprovedFlFeatures(),
      _getQuarantinedFlFeatures(),
      _listFlUnquarantineEvents(25),
    ]);
    res.json({
      candidates,
      approvedFeatures,
      quarantinedFeatures,
      unquarantineEvents,
      transformKinds: FEATURE_TRANSFORM_KINDS,
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "candidates list failed" });
  }
});

router.get("/crypto/feature-lab/quarantined", async (_req, res): Promise<void> => {
  // Task #242 — surface auto-retired features written by the ml-engine
  // after a training run regressed on validation log_loss. Read-only;
  // operators see name, transformKind, regressing timeframes, and the
  // before/after log_loss snapshot from `feature_lab.quarantined`.
  try {
    const features = await _getQuarantinedFlFeatures();
    res.json({ features });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "quarantined list failed" });
  }
});

router.get("/crypto/feature-lab/unquarantine-events", async (req, res): Promise<void> => {
  // Task #247 — dedicated endpoint for the un-quarantine override
  // audit log so future tooling (e.g. an "all overrides" drawer) can
  // page through it without re-fetching the full candidates payload.
  try {
    const limit = req.query.limit != null ? Number(req.query.limit) || 25 : 25;
    const events = await _listFlUnquarantineEvents(limit);
    res.json({ events });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unquarantine events failed" });
  }
});

router.get("/crypto/feature-lab/unquarantine-summary", async (req, res): Promise<void> => {
  // Task #256 — group-by roll-up of operator un-quarantine overrides
  // over a configurable window. The Feature Lab card surfaces this so
  // reviewers can spot repeat patterns at a glance instead of scrolling
  // the per-event audit log.
  try {
    const windowDays = req.query.windowDays != null
      ? Number(req.query.windowDays) || 30
      : 30;
    const summary = await _summarizeFlUnquarantineOverrides(windowDays);
    res.json(summary);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "unquarantine summary failed" });
  }
});

router.get("/crypto/feature-lab/applied-models", async (req, res): Promise<void> => {
  // Task #235 — surface, per approved feature, which freshly-trained
  // model versions actually baked it in. Powers the Feature Lab UI's
  // "applied in" badges.
  try {
    const limitPerFeature = req.query.limitPerFeature != null
      ? Math.max(1, Math.min(50, Number(req.query.limitPerFeature) || 12))
      : 12;
    const features = await _summarizeFlApplied({ limitPerFeature });
    res.json({ features });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "applied-models failed" });
  }
});

router.post("/crypto/feature-lab/candidates", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  try {
    const body = (req.body ?? {}) as Record<string, unknown>;
    if (typeof body.name !== "string" || typeof body.transformKind !== "string") {
      res.status(400).json({ error: "name and transformKind required" });
      return;
    }
    const row = await _createFlCandidate({
      name: body.name,
      transformKind: body.transformKind as never,
      sourceColumn: typeof body.sourceColumn === "string" ? body.sourceColumn : null,
      description: typeof body.description === "string" ? body.description : null,
      proposedBy: typeof body.proposedBy === "string" ? body.proposedBy : null,
    });
    res.status(201).json({ row });
  } catch (err) {
    res.status(400).json({ error: err instanceof Error ? err.message : "create failed" });
  }
});

router.get("/crypto/feature-lab/reports", async (req, res): Promise<void> => {
  try {
    const candidateId = req.query.candidateId != null
      ? Number(req.query.candidateId)
      : undefined;
    const reports = await _listFlReports(candidateId);
    res.json({ reports });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "reports list failed" });
  }
});

router.post("/crypto/feature-lab/candidates/:id/ablate", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "invalid id" });
    return;
  }
  try {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const timeframe = typeof body.timeframe === "string" ? body.timeframe : "1h";
    const coinId = typeof body.coinId === "string" ? body.coinId : "__pooled__";
    const report = await _runFlAblation({ candidateId: id, timeframe, coinId });
    res.json({ report });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "ablate failed" });
  }
});

router.get("/crypto/feature-lab/candidates/:id/promotion-eligibility", async (req, res): Promise<void> => {
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "invalid id" });
    return;
  }
  try {
    res.json(await _evalFlPromotion(id));
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : "eligibility failed" });
  }
});

router.post("/crypto/feature-lab/candidates/:id/approve", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "invalid id" });
    return;
  }
  try {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const out = await _approveFlCandidate({
      candidateId: id,
      approvedBy: typeof body.approvedBy === "string" ? body.approvedBy : "operator",
      note: typeof body.note === "string" ? body.note : undefined,
      force: body.force === true,
    });
    res.json(out);
  } catch (err) {
    res.status(400).json({ error: err instanceof Error ? err.message : "approve failed" });
  }
});

router.get(
  "/crypto/feature-lab/candidates/:id/unquarantine-assessment",
  async (req, res): Promise<void> => {
    // Task #248 — re-checks the regression that drove the auto-retire
    // against the latest training report so the dashboard can warn the
    // operator before they confirm an un-quarantine. Read-only; no
    // admin-key gate (same as the rest of /feature-lab GETs).
    const id = Number(req.params.id);
    if (!Number.isFinite(id)) {
      res.status(400).json({ error: "invalid id" });
      return;
    }
    try {
      const assessment = await _assessUnquarantineRegression(id);
      res.json({ assessment });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : "assessment failed",
      });
    }
  },
);

router.post("/crypto/feature-lab/candidates/:id/unquarantine", async (req, res): Promise<void> => {
  // Task #243 — operator override for an auto-retired feature. Same
  // admin-key gate as approve; un-quarantines the candidate and
  // re-appends the spec to the approved bucket so the next training
  // run picks it up again.
  //
  // Task #248 — if the validation regression that drove the auto-retire
  // still looks present in the latest training report, refuse with HTTP
  // 409 + the assessment payload unless the caller passes
  // `acknowledgement` (typed reason) or `force=true`.
  if (!requireAdminApiKey(req, res)) return;
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "invalid id" });
    return;
  }
  try {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const out = await _unquarantineFlCandidate({
      candidateId: id,
      approvedBy: typeof body.approvedBy === "string" ? body.approvedBy : "operator",
      note: typeof body.note === "string" ? body.note : undefined,
      force: body.force === true,
      acknowledgement:
        typeof body.acknowledgement === "string" ? body.acknowledgement : undefined,
    });
    res.json(out);
  } catch (err) {
    if (err instanceof _UnquarantineRegressionStillPresentError) {
      res.status(409).json({
        error: err.message,
        code: err.code,
        assessment: err.assessment,
      });
      return;
    }
    res.status(400).json({ error: err instanceof Error ? err.message : "unquarantine failed" });
  }
});

router.post("/crypto/feature-lab/candidates/:id/reject", async (req, res): Promise<void> => {
  if (!requireAdminApiKey(req, res)) return;
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "invalid id" });
    return;
  }
  try {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const row = await _rejectFlCandidate(
      id,
      typeof body.note === "string" ? body.note : undefined,
    );
    res.json({ row });
  } catch (err) {
    res.status(400).json({ error: err instanceof Error ? err.message : "reject failed" });
  }
});

// ─────────────────────────────────────────────────────────────────────────
// Task #444 — LLM sidecar routes deleted.
//
// The four sidecar endpoints (`/crypto/llm/sidecar/state`, `/toggle`,
// `/recent`, `/crypto/llm/copilot`) and the `gatherCopilotContext` helper
// were removed end-to-end with the rest of the LLM/news/sentiment plane.
// Quant brain is the sole authority for trade decisions. The dashboard
// cards that consumed these routes are deleted in the same task; the
// contract test in `test/no-llm-fields-runtime.test.ts` asserts the
// endpoints stay 404 to catch regressions.
// ─────────────────────────────────────────────────────────────────────────

// Task #384 — meta-brain cycle stats for the dashboard. Pure
// observability: returns the most recent `meta_brain_cycle_stats`
// snapshot from the api-server adapter plus a small history ring
// buffer, the current enable mode, and a best-effort proxy of
// ml-engine `/ml/meta-brain/health` + `/ml/meta-brain/stats`. The
// trading path never reads from this endpoint.
router.get("/crypto/meta-brain/cycle-stats", async (_req, res): Promise<void> => {
  const enabled = process.env.META_BRAIN_ENABLED === "1";
  const shadow = process.env.META_BRAIN_SHADOW === "1";
  const mode = enabled ? "live" : shadow ? "shadow" : "off";
  const cycle = _getMetaBrainCycleStats();
  // Task #390 — read-only Strategy Lab benchmark telemetry assembled
  // from `strategy_snapshots`. Pure governance / observability — never
  // routed through the predictor or the trade-decision path.
  let benchmarkBlock: unknown = null;
  try {
    const bench = await _assembleMetaBrainBenchmark();
    benchmarkBlock = bench;
  } catch {
    // Swallow — observability only.
  }

  // Best-effort proxy. Never throws into the response.
  const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
  let mlHealth: unknown = null;
  let mlStats: unknown = null;
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 1500);
    const [hRes, sRes] = await Promise.allSettled([
      fetch(`${base}/ml/meta-brain/health`, { signal: ctrl.signal }),
      fetch(`${base}/ml/meta-brain/stats`, { signal: ctrl.signal }),
    ]);
    clearTimeout(t);
    if (hRes.status === "fulfilled" && hRes.value.ok) {
      mlHealth = await hRes.value.json().catch(() => null);
    }
    if (sRes.status === "fulfilled" && sRes.value.ok) {
      mlStats = await sRes.value.json().catch(() => null);
    }
  } catch {
    // Swallow — observability only.
  }

  // Task #390 — fold the brain's current `benchmark` family trust
  // weight (from `mlEngine.stats.trust_by_family.benchmark.trust`)
  // into the assembled benchmark block so the dashboard shows BOTH
  // the input signal and the brain's learned weight on it. Pure
  // observability — never written back into any decision payload.
  let trustWeight: number | null = null;
  try {
    const stats = mlStats as
      | { trust_by_family?: Record<string, { trust?: unknown }> }
      | null;
    const t = stats?.trust_by_family?.benchmark?.trust;
    if (typeof t === "number" && Number.isFinite(t)) trustWeight = t;
  } catch {
    // ignore — observability only
  }
  const benchmarkOut =
    benchmarkBlock && typeof benchmarkBlock === "object"
      ? { ...(benchmarkBlock as Record<string, unknown>), trustWeight }
      : benchmarkBlock;

  res.json({
    mode,
    enabled,
    shadow,
    last: cycle.last,
    history: cycle.history,
    mlEngine: { health: mlHealth, stats: mlStats },
    // Task #390 — `benchmark` is the api-server's locally-assembled
    // Strategy Lab telemetry, augmented with the brain's learned
    // `benchmark` family trust weight (`trustWeight`).
    benchmark: benchmarkOut,
    fetchedAt: new Date().toISOString(),
  });
});

export default router;
