import { db, agentsTable, predictionsTable, priceHistoryTable, monitoringStateTable, paperPositionsTable, paperPortfoliosTable } from "@workspace/db";
import { assertNativeCadence } from "./cadence-guard";
import { eq, and, desc, sql, lte, gte, isNull } from "drizzle-orm";
import { logger } from "./logger";
import { fetchCoinPrices, isPriceDataFresh, MONITORED_COINS, fetchHistoricalOHLCV, fetchFearGreedIndex, fetchBtcDominance, type CoinPrice, type FearGreedData } from "./coins";
import { computeBestSlice, type BestPickResult } from "./best-slice";
import { getPrediction as getBrainPrediction, resetCycleCache as resetBrainCycleCache } from "./prediction-orchestrator";
import { isQuantBrainEnabled } from "./brain-flag";
import { isMeasurementModeEnabled } from "./measurement-mode";
import { getMttmConfig } from "./mttm";
import { evaluateBrainAutoRevert } from "./brain-auto-revert";
import { analyzePatterns, getAgentPastAccuracy, getPastPredictionPatterns, TIMEFRAMES, type TimeframeKey, generateFingerprint } from "./pattern-analyzer";
import { initializePaperPortfolios, executePaperTrade, closeExpiredPositions, updatePortfolioValues, autoDeployIdleCash, getAutoDeployAttribution, recordAutoDeployAttributionSnapshot } from "./paper-trader";
import { writePredictionJournal, markPredictionJournalTrade, resolvePredictionJournal, lookupGateForPrediction } from "./journal-writer";
import { listRegistry } from "./model-registry";
import { runJournalRetention } from "./journal-retention";
import { pruneUnquarantineEventsIfDue } from "./feature-lab";
import { pruneMarketSignalsIfDue } from "./market-signals-retention";
import {
  seedDedupSetFromCurrentQuarantine,
  startAutoRetireNotifierLoop,
} from "./auto-retire-notifier";
import {
  seedDedupSetFromCurrentReport as seedTrainingContractDedup,
  startTrainingContractNotifierLoop,
} from "./training-contract-notifier";
import { startMarketSignalsPollerWatcher } from "./market-signals-poller-watcher";
import { startTopup5mNotifierLoop } from "./topup-5m-notifier";
import { startDisabledOutcomeNotifierLoop } from "./disabled-outcome-notifier";
import { paperTradesTable } from "@workspace/db";
import { runStrategyLabCycle, ensureStrategyLabAgents } from "./strategy-lab";
import {
  syncAgentProfileIds,
  loadAgentRegistryCache,
  installSighupReload,
  startRetirementLoop,
  seedExecutorAgents,
} from "./agents-registry";
import {
  fetchShadowPrediction,
  recordShadowPrediction,
  resolveShadowPredictions,
} from "./shadow-recorder";
import { judgeDirection } from "./trading-constants";
import { storeCoinInsight, getCoinInsights, computeCoinCorrelations } from "./coin-insights";
import { getRecommendedAgentCoins } from "./agent-fleet-coins";
import { invalidateCalibrationCache } from "./confidence-calibrator";
import { classifyRegime, updateFingerprintBuffer, getRegimeAdjustments, getCurrentRegime, type RegimeState } from "./regime-detector";
import { detectContagionEvents } from "./contagion-detector";
import { getAgentSpecialization, allocateCoinsForAgent } from "./agent-specialization";
import { getAblationConfig } from "./ablation-config";
import {
  computeSuggestion,
  computeTightenSuggestion,
  recordHealthyObservation,
  setCachedSuggestion,
  recordTightenSuggestionTick,
  isAutoApplyTightenEnabled,
  getAutoApplyTightenTicks,
  applyTighten,
} from "./tuning-tracker";
// Task #373 — supervisory meta-brain. Never predicts price, never
// places trades. evaluate(batch) runs once per tick; directive is
// cached for the next tick's sizing. See audit/meta-brain-integration.md.
import {
  assembleBenchmarkTelemetry as metaBrainAssembleBenchmark,
  collectSlice as metaBrainCollectSlice,
  flushTick as metaBrainFlushTick,
  resetBenchmarkCache as metaBrainResetBenchmarkCache,
  resolveStrategyFamily as metaBrainResolveFamily,
  resolveStrategyFamilyForProfile as metaBrainResolveFamilyForProfile,
  setBenchmarkTelemetry as metaBrainSetBenchmark,
  setPortfolioTelemetry as metaBrainSetPortfolio,
} from "./meta-brain";

let monitorInterval: ReturnType<typeof setInterval> | null = null;
let resolutionInterval: ReturnType<typeof setInterval> | null = null;
let paperTradeInterval: ReturnType<typeof setInterval> | null = null;
let tuningSuggestionInterval: ReturnType<typeof setInterval> | null = null;
let retrainInterval: ReturnType<typeof setInterval> | null = null;
let journalRetentionInterval: ReturnType<typeof setInterval> | null = null;
let lastRetrainAt: number | null = null;
let lastRetrainStatus: "ok" | "error" | "skipped" | null = null;
let lastRetrainError: string | null = null;

const RETRAIN_INTERVAL_MS = 8 * 60 * 60 * 1000; // 3 retrains per 24h

async function triggerScheduledRetrain(): Promise<void> {
  const adminKey = process.env.ADMIN_API_KEY || process.env.ML_ADMIN_TOKEN;
  if (!adminKey) {
    lastRetrainStatus = "skipped";
    lastRetrainError = "ADMIN_API_KEY not set";
    logger.warn("Scheduled retrain skipped: ADMIN_API_KEY not set");
    return;
  }
  const base = process.env.ML_ENGINE_URL || "http://localhost:8000";
  const url = `${base.replace(/\/$/, "")}/ml/admin/retrain`;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Admin-Key": adminKey },
      body: JSON.stringify({}),
    });
    if (res.status === 202 || res.status === 200) {
      lastRetrainAt = Date.now();
      lastRetrainStatus = "ok";
      lastRetrainError = null;
      logger.info("Scheduled retrain enqueued");
    } else if (res.status === 409) {
      lastRetrainStatus = "skipped";
      lastRetrainError = "already in progress";
      logger.info("Scheduled retrain skipped: already in progress");
    } else {
      lastRetrainStatus = "error";
      lastRetrainError = `HTTP ${res.status}`;
      logger.warn({ status: res.status }, "Scheduled retrain failed");
    }
  } catch (err) {
    lastRetrainStatus = "error";
    lastRetrainError = err instanceof Error ? err.message : String(err);
    logger.error({ err }, "Scheduled retrain request failed");
  }
}

export function getRetrainSchedulerState(): {
  intervalHours: number;
  lastRetrainAt: number | null;
  lastRetrainStatus: "ok" | "error" | "skipped" | null;
  lastRetrainError: string | null;
  nextRetrainAt: number | null;
} {
  return {
    intervalHours: RETRAIN_INTERVAL_MS / (60 * 60 * 1000),
    lastRetrainAt,
    lastRetrainStatus,
    lastRetrainError,
    nextRetrainAt: lastRetrainAt ? lastRetrainAt + RETRAIN_INTERVAL_MS : null,
  };
}

async function refreshTuningSuggestion(): Promise<void> {
  try {
    const [openPositions, allAgents] = await Promise.all([
      db.select().from(paperPositionsTable),
      db.select().from(agentsTable).where(eq(agentsTable.isActive, true)),
    ]);
    const agentCount = allAgents.length || 1;
    const healthyOpenFloor = Math.max(1, agentCount);
    recordHealthyObservation(openPositions.length >= healthyOpenFloor);

    // Loosen takes priority — if bots are starving for trades we want to
    // open the gates first. Only when no loosen suggestion fires do we
    // consider walking previously loosened gates back toward baseline.
    let suggestion = await computeSuggestion(openPositions.length, agentCount);
    if (!suggestion) {
      suggestion = await computeTightenSuggestion(openPositions.length, agentCount);
    }

    // Track per-gate continuous validity of tighten suggestions so the engine
    // can auto-apply them after a sustained window. Loosen suggestions and
    // gaps reset the counter.
    const tightenGate =
      suggestion && suggestion.direction === "tighten" ? suggestion.gate : null;
    const pending = recordTightenSuggestionTick(tightenGate);

    if (
      suggestion &&
      suggestion.direction === "tighten" &&
      pending &&
      isAutoApplyTightenEnabled() &&
      pending.ticks >= getAutoApplyTightenTicks()
    ) {
      try {
        const change = applyTighten(suggestion.gate, "auto-tighten");
        logger.info(
          {
            gate: change.gate,
            oldValue: change.oldValue,
            newValue: change.newValue,
            ticks: pending.ticks,
          },
          `Tuning: auto-applied tighten after ${pending.ticks} sustained ticks`,
        );
        setCachedSuggestion(null);
        return;
      } catch (err) {
        logger.error(
          { err, gate: suggestion.gate },
          "Failed to auto-apply tighten suggestion",
        );
      }
    }

    setCachedSuggestion(suggestion);
    if (suggestion) {
      logger.info(
        {
          direction: suggestion.direction,
          gate: suggestion.gate,
          share: suggestion.shareOfSkips.toFixed(2),
          openPositions: suggestion.openPositionCount,
          proposed: suggestion.proposedValue,
          tightenTicks: suggestion.direction === "tighten" ? pending?.ticks : undefined,
        },
        `Tuning ${suggestion.direction} suggestion ready`,
      );
    }
  } catch (err) {
    logger.error({ err }, "Failed to refresh tuning suggestion");
  }
}
let cycleCount = 0;
let latestBestPick: BestPickResult | null = null;

interface PatternCacheEntry {
  patternType: string;
  rsi: number;
  macdCrossover: string;
  bbPercentB: number;
  volatility: number;
  fingerprint?: string;
  regime?: string;
}

let isRunningCycle = false;
let cycleStartedAt: number | null = null;
let latestFearGreed: FearGreedData | null = null;

export function forceUnlockCycle(): void {
  isRunningCycle = false;
  cycleStartedAt = null;
}

export async function seedHistoricalPriceData(): Promise<void> {
  // Task #343 — the legacy synthetic seeder fabricated 24 hourly points
  // per CMC daily candle and wrote them into `price_history` flagged
  // `is_synthetic=true`. Those rows are at HOURLY cadence, not the 1m
  // native cadence `price_history` is contracted to hold. The trainer's
  // resampler would silently bucket them into 5m / 1h closes — exactly
  // the contamination this task hardens against. Real historical bars
  // now come from `artifacts/ml-engine/scripts/backfill_history.py`
  // (1m → price_history, 5m+ → price_candles), so this seeder is now a
  // no-op. The export is kept so any caller that has already imported
  // it doesn't crash. Cadence regression coverage lives in
  // `test/cadence-correctness.test.ts` and the new
  // `test_per_coin_retrain_isolation.py` — not in a runtime self-test.
  logger.info(
    "seedHistoricalPriceData is a no-op since task #343 — synthetic " +
    "hourly seeding contaminates price_history's 1m contract. Use " +
    "ml-engine/scripts/backfill_history.py for real OHLCV.",
  );
}

/**
 * Task #444 — the legacy initialiser seeded one DB agent per LLM
 * personality (Momentum Max, Contrarian Clara, …) and kept their
 * `personality` / `systemPrompt` columns synced with the in-memory
 * AGENT_PERSONALITIES list. Both the personality module and the auto-
 * evolution path are gone, so this function now only ensures the
 * `monitoring_state` row exists. Existing agent rows are preserved
 * verbatim so historical journals/joins keep resolving by id.
 */
export async function initializeAgents(): Promise<void> {
  const state = await db.select().from(monitoringStateTable);
  if (state.length === 0) {
    await db.insert(monitoringStateTable).values({
      isRunning: false,
      cycleCount: 0,
    });
  }
}

// Outcome thresholds are shared with confidence-calibrator via trading-constants.
// They are intentionally set ABOVE the round-trip trading cost so a "correct"
// prediction is also net-profitable after fees + slippage. See trading-constants.ts.
import { OUTCOME_THRESHOLDS_PERCENT, getOutcomeThresholdPercent, CYCLE_STUCK_TIMEOUT_MS, SLIPPAGE_PCT } from "./trading-constants";
const TIMEFRAME_THRESHOLDS = OUTCOME_THRESHOLDS_PERCENT;

const TIMEFRAME_SCORE_MULTIPLIERS: Record<string, number> = {
  "1m": 0.3,
  "5m": 0.5,
  "1h": 0.8,
  "2h": 1.0,
  "6h": 1.2,
  "1d": 1.5,
};

async function resolvePendingPredictions(): Promise<number> {
  const now = new Date();
  const GRACE_PERIOD_MS = 5 * 60 * 1000;
  const pendingPredictions = await db
    .select()
    .from(predictionsTable)
    .where(and(
      eq(predictionsTable.outcome, "pending"),
      sql`(${predictionsTable.resolvesAt} IS NOT NULL AND ${predictionsTable.resolvesAt} <= ${now}) OR (${predictionsTable.resolvesAt} IS NULL AND ${predictionsTable.createdAt} <= ${new Date(now.getTime() - 60000)})`
    ));

  if (pendingPredictions.length === 0) return 0;

  // Force a fresh price fetch (bypass the 30s cache) so 5-minute prediction
  // outcomes are judged against the actual current market, not data that
  // could be up to 90s old.
  const prices = await fetchCoinPrices(true);
  if (prices.length === 0) {
    logger.warn("Price fetch returned no data — deferring prediction resolution");
    return 0;
  }
  // Even when fetchCoinPrices returns data, on a failed live fetch it may
  // hand back the cached snapshot. Refuse to resolve against stale prices.
  if (!isPriceDataFresh()) {
    logger.warn("Price data is stale after force-fetch — deferring prediction resolution");
    return 0;
  }
  let resolved = 0;

  for (const prediction of pendingPredictions) {
    const currentCoin = prices.find((p) => p.id === prediction.coinId);
    if (!currentCoin) continue;

    const actualPrice = currentCoin.currentPrice;
    const priceChange = ((actualPrice - prediction.priceAtPrediction) / prediction.priceAtPrediction) * 100;
    const threshold = getOutcomeThresholdPercent(prediction.timeframe || "1m");

    const tf = prediction.timeframe || "1m";

    const resolvesAtTime = prediction.resolvesAt ? new Date(prediction.resolvesAt).getTime() : 0;
    const timeSinceResolve = now.getTime() - resolvesAtTime;
    const withinGracePeriod = resolvesAtTime > 0 && timeSinceResolve < GRACE_PERIOD_MS;
    const pastGracePeriod = resolvesAtTime > 0 && timeSinceResolve >= GRACE_PERIOD_MS;

    const checkDirectionCorrect = (change: number) =>
      judgeDirection(prediction.direction as "up" | "down" | "stable", change, tf);

    const t0Check = checkDirectionCorrect(priceChange);
    let isCorrect = t0Check.correct;
    let isNeutral = t0Check.neutral;

    if (!isCorrect && withinGracePeriod) {
      if (prediction.graceT0Price === null || prediction.graceT0Price === undefined) {
        await db.update(predictionsTable).set({
          graceT0Price: actualPrice,
          graceT0PriceChange: priceChange,
        }).where(eq(predictionsTable.id, prediction.id));
      }
      continue;
    }

    const hasGraceT0 = prediction.graceT0Price !== null && prediction.graceT0Price !== undefined;
    if (pastGracePeriod && hasGraceT0) {
      if (!isCorrect) {
        const t0Result = checkDirectionCorrect(prediction.graceT0PriceChange!);
        if (t0Result.correct) {
          isCorrect = true;
          isNeutral = false;
        } else if (t0Result.neutral && !isNeutral) {
          isNeutral = true;
        }

        if (!isCorrect) {
          const graceResult = checkDirectionCorrect(priceChange);
          if (graceResult.correct) {
            isCorrect = true;
            isNeutral = false;
          } else if (graceResult.neutral && !isNeutral) {
            isNeutral = true;
          }
        }
      }
    }

    const priceAccuracy = prediction.predictedPrice > 0
      ? 1 - Math.abs(actualPrice - prediction.predictedPrice) / prediction.predictedPrice
      : 0;
    const accuracyBonus = Math.max(0, priceAccuracy - 0.5) * 5;

    const tfMultiplier = TIMEFRAME_SCORE_MULTIPLIERS[prediction.timeframe || "5m"] || 1;
    const outcome = isCorrect ? "correct" : isNeutral ? "neutral" : "wrong";
    const scoreChange = isCorrect
      ? (prediction.confidence * 8 + accuracyBonus * 2) * tfMultiplier
      : isNeutral
        ? 0
        : -(prediction.confidence * 6) * tfMultiplier;

    const updated = await db
      .update(predictionsTable)
      .set({ outcome, actualPrice, scoreChange, resolvedAt: now })
      .where(and(eq(predictionsTable.id, prediction.id), eq(predictionsTable.outcome, "pending")))
      .returning({ id: predictionsTable.id });

    if (updated.length === 0) continue;

    // Phase 1 — keep the adaptive prediction-journal row in lock-step with
    // the legacy resolution. Fire-and-forget so a journal hiccup never
    // blocks resolution of the underlying prediction.
    void resolvePredictionJournal(prediction.id, {
      actualPrice,
      outcome,
      realizedReturnPct: priceChange,
      resolvedAt: now,
    });

    try {
      const cachedPattern = (prediction.patternContext as PatternCacheEntry | null) ?? null;
      // Task #444 — coin_insights stays as the regime-detector's pattern
      // memory, but the LLM-only `marketContext` / `contextTags` columns
      // are gone. We persist the deterministic fields only.
      await storeCoinInsight(
        prediction.coinId,
        prediction.timeframe || "1m",
        cachedPattern?.patternType || "unknown",
        prediction.direction,
        outcome,
        priceChange,
        cachedPattern?.rsi ?? null,
        cachedPattern?.macdCrossover ?? null,
        cachedPattern?.bbPercentB ?? null,
        cachedPattern?.volatility ?? null,
        prediction.agentId,
        cachedPattern?.fingerprint ?? null,
        cachedPattern?.regime ?? getCurrentRegime()?.regime ?? null
      );
    } catch (insightErr) {
      logger.error({ err: insightErr, coinId: prediction.coinId }, "Failed to store coin insight on resolution");
    }

    invalidateCalibrationCache(prediction.agentId);

    const agent = await db
      .select()
      .from(agentsTable)
      .where(eq(agentsTable.id, prediction.agentId));

    if (agent.length > 0) {
      const a = agent[0];
      const countedCorrect = isCorrect;
      const newScore = Math.max(5, Math.min(200, a.score + scoreChange));
      const newStreak = isNeutral
        ? a.streak
        : (a.streakType === (countedCorrect ? "win" : "loss"))
          ? a.streak + 1
          : 1;

      const streakBonus = countedCorrect && newStreak >= 3 ? newStreak * 0.5 : 0;
      const streakPenalty = !countedCorrect && !isNeutral && newStreak >= 5 ? newStreak * 0.3 : 0;
      const finalScore = Math.max(5, Math.min(200, newScore + streakBonus - streakPenalty));

      const recoveryBoost = finalScore < 40 ? 0.5 : 0;
      const adjustedScore = Math.max(15, Math.min(200, finalScore + recoveryBoost));
      const newStatus = adjustedScore < 30 ? "degraded" : adjustedScore < 60 ? "resting" : "active";

      await db
        .update(agentsTable)
        .set({
          score: adjustedScore,
          totalPredictions: isNeutral ? a.totalPredictions : a.totalPredictions + 1,
          correctPredictions: countedCorrect ? a.correctPredictions + 1 : a.correctPredictions,
          wrongPredictions: (!countedCorrect && !isNeutral) ? a.wrongPredictions + 1 : a.wrongPredictions,
          streak: newStreak,
          streakType: isNeutral ? a.streakType : (countedCorrect ? "win" : "loss"),
          status: newStatus,
        })
        .where(eq(agentsTable.id, a.id));
    }
    resolved++;
  }

  if (resolved > 0) {
    logger.info({ resolved }, "Resolved predictions");
  }
  return resolved;
}

function getTimeframesForCycle(cycle: number): TimeframeKey[] {
  const tfs: TimeframeKey[] = ["2h"];
  if (cycle % 3 === 0) tfs.push("1h");
  if (cycle % 4 === 0) tfs.push("5m");
  if (cycle % 2 === 0) tfs.push("6h");
  if (cycle % 6 === 0) tfs.push("1d");
  return tfs;
}

async function selectCoinsBySignalStrength(
  prices: CoinPrice[],
  timeframes: TimeframeKey[],
  maxCoins: number
): Promise<{ coin: CoinPrice; signalStrength: number; bestTimeframe: TimeframeKey }[]> {
  const scored: { coin: CoinPrice; signalStrength: number; bestTimeframe: TimeframeKey }[] = [];

  for (const coin of prices) {
    let maxStrength = 0;
    let bestTf: TimeframeKey = timeframes[0];

    for (const tf of timeframes) {
      try {
        const analysis = await analyzePatterns(coin.id, tf);
        if (analysis.signalStrength > maxStrength) {
          maxStrength = analysis.signalStrength;
          bestTf = tf;
        }
      } catch {
        continue;
      }
    }

    const dayChange = Math.abs(coin.priceChange24h ?? 0);
    const volumeBoost = coin.volume24h > 50_000_000 ? 3 : coin.volume24h > 10_000_000 ? 1 : 0;
    const changeBoost = dayChange > 5 ? 5 : dayChange > 2 ? 2 : 0;

    // Task #444 — the win-rate accuracy multiplier used to read
    // `coin_insights.totalInsights` / `winRate` for LLM-era weighting.
    // After the LLM cutover the per-coin signal score is the
    // deterministic technical-strength + volume + day-change combo.
    let accuracyMultiplier = 1.0;
    try {
      const coinInsightSummary = await getCoinInsights(coin.id, bestTf);
      if (coinInsightSummary.totalInsights >= 5) {
        accuracyMultiplier = 0.5 + coinInsightSummary.winRate;
      }
    } catch (e) { logger.debug({ err: e, coinId: coin.id }, "Coin accuracy lookup failed"); }

    scored.push({
      coin,
      signalStrength: (maxStrength + volumeBoost + changeBoost) * accuracyMultiplier,
      bestTimeframe: bestTf,
    });
  }

  scored.sort((a, b) => b.signalStrength - a.signalStrength);
  return scored.slice(0, maxCoins);
}

export async function runAnalysisCycle(): Promise<{ predictionsCreated: number; cycleNumber: number }> {
  if (isRunningCycle) {
    const stuckMs = cycleStartedAt ? Date.now() - cycleStartedAt : 0;
    if (stuckMs > CYCLE_STUCK_TIMEOUT_MS) {
      logger.error({ stuckMs }, "Cycle lock stuck >4min — force unlocking");
      isRunningCycle = false;
      cycleStartedAt = null;
    } else {
      logger.warn({ stuckMs }, "Analysis cycle already running — skipping");
      return { predictionsCreated: 0, cycleNumber: cycleCount };
    }
  }
  isRunningCycle = true;
  cycleStartedAt = Date.now();

  try {
  const prices = await fetchCoinPrices();

  if (prices.length === 0) {
    logger.warn("No price data available — skipping analysis cycle");
    cycleCount++;
    return { predictionsCreated: 0, cycleNumber: cycleCount };
  }

  if (!isPriceDataFresh()) {
    logger.warn("Price data is stale (>2min old) — skipping predictions, only resolving");
    await resolvePendingPredictions();
    cycleCount++;
    return { predictionsCreated: 0, cycleNumber: cycleCount };
  }

  // MEASUREMENT_MODE: when ON, blocks auto-deploy idle cash so we can
  // observe the consensus engine in isolation. Default is now ON so the
  // legacy momentum auto-deploy lane stays gated unless an operator
  // explicitly flips it off. Operators can flip this from the Strategy
  // Lab UI; fresh persisted DB values win (manual ON expires after 24h
  // unless `pinned`), with the MEASUREMENT_MODE env var as the fallback
  // default when no row exists yet.
  const MEASUREMENT_MODE = await isMeasurementModeEnabled();

  // Task #614 — warm the MTTM config cache once per cycle. The whitelist
  // guard and position-size override in paper-trader.ts read from the
  // sync `getMttmConfigCached()` accessor; this primes it so the first
  // executePaperTrade of the cycle doesn't fail-open on a cold cache.
  // Errors here are non-fatal — the helper logs and keeps the previous
  // snapshot.
  try {
    await getMttmConfig();
  } catch (err) {
    logger.warn({ err }, "monitor: failed to warm MTTM config cache (non-fatal)");
  }

  // Task #659 — DS drift+tally evaluators (no-ops outside DS mode).
  try {
    const {
      evaluateDiagnosticSandboxDrift,
      evaluateDiagnosticSandboxAutoDisable,
    } = await import("./mttm");
    await evaluateDiagnosticSandboxDrift();
    await evaluateDiagnosticSandboxAutoDisable();
  } catch (err) {
    logger.warn({ err }, "monitor: DS evaluators failed (non-fatal)");
  }

  // Task #444 — agent-evolution (LLM-driven personality mutation) is gone.
  // The active agent roster is the DB roster, full stop. Live executor
  // scope: archived / `legacy_archived` agents are excluded so the per-
  // cycle prediction loop and downstream fleet aggregations only count
  // bots that can still authorize live paper trades.
  const agents = await db.select().from(agentsTable).where(and(
    eq(agentsTable.isActive, true),
    eq(agentsTable.strategyType, "ai-bots"),
    isNull(agentsTable.archivedAt),
    sql`${agentsTable.profileId} IS DISTINCT FROM 'legacy_archived'`,
  ));
  const liveExecutorAgentIds = new Set(agents.map((a) => a.id));

  // Phase 2 — regime FIRST. Compute the canonical 6-class label for
  // this cycle BEFORE we ingest any candles, so every row of
  // price_history we write below carries the correct regime label of
  // the cycle that produced it (no off-by-one cycle, no startup nulls
  // beyond the very first cycle's failed-classifier path).
  await resolvePendingPredictions();

  // Task #444 — generateRealMarketNews + computeMarketSentiment were the
  // last LLM-era inputs. The quant brain consumes price/regime features
  // only; fear & greed is still useful as a regime input.
  latestFearGreed = await fetchFearGreedIndex();

  const ablation = getAblationConfig();

  const btcDominance = await fetchBtcDominance();
  const regimeState: RegimeState = ablation.regimeDetection
    ? await classifyRegime(prices, latestFearGreed, btcDominance)
    : { regime: "sideways" as const, regimeLabel: "range_chop" as const, confidence: 0, avgChange24h: 0, btcDominance: null, btcDominanceDirection: null, trendBias: "neutral" as const, crossCoinVolatility: 0, fearGreedValue: null, bullishRatio: 0.5, timestamp: Date.now() };

  // Stamp the cycle's regime label (6-class, ml-engine taxonomy) on
  // every tick we ingest so historical analyses can join
  // (price_history, regime) without re-classifying after the fact.
  const tickRegime = regimeState.regimeLabel;
  // Task #343 — live polling cadence is the 1m base contract for
  // price_history; the guard fails loud if a future change ever fires
  // this loop at a coarser interval.
  assertNativeCadence("1m", "monitor.runMonitoringCycle", "price_history");
  for (const coin of prices) {
    await db.insert(priceHistoryTable).values({
      coinId: coin.id,
      price: coin.currentPrice,
      timestamp: new Date(),
      regime: tickRegime,
      // Task #347 — explicit cadence marker for the DB-level CHECK
      // constraint. Defense-in-depth on top of `assertNativeCadence`
      // above; defaults to '1m' in the schema so this line is belt-
      // and-braces clarity, not strictly required for correctness.
      cadence: "1m",
    });
  }
  if (ablation.regimeDetection) {
    logger.info({ regime: regimeState.regime, confidence: (regimeState.confidence * 100).toFixed(0), avgChange: regimeState.avgChange24h.toFixed(2), btcDom: regimeState.btcDominance?.toFixed(1) ?? "n/a", btcDomDir: regimeState.btcDominanceDirection ?? "n/a" }, "Market regime classified");
  }

  if (ablation.contagionDetection) {
    try {
      await detectContagionEvents(prices);
    } catch (err) {
      logger.error({ err }, "Contagion detection failed");
    }
  }

  const timeframesThisCycle = getTimeframesForCycle(cycleCount);
  let predictionsCreated = 0;

  // Phase 5 — fresh brain cache for this cycle. Quant predictions are
  // shared across agents (one per coin×timeframe). The brain orchestrator
  // is the sole authority for direction/confidence after Task #444.
  resetBrainCycleCache();

  const agentAccuracyCache: Record<number, Awaited<ReturnType<typeof getAgentPastAccuracy>>> = {};

  const topCoins = await selectCoinsBySignalStrength(prices, timeframesThisCycle, 6);

  const pendingPredictions = await db
    .select({
      agentId: predictionsTable.agentId,
      coinId: predictionsTable.coinId,
      timeframe: predictionsTable.timeframe,
    })
    .from(predictionsTable)
    .where(eq(predictionsTable.outcome, "pending"));

  const pendingKeys = new Set(
    pendingPredictions.map(p => `${p.agentId}:${p.coinId}:${p.timeframe}`)
  );

  // In MEASUREMENT_MODE, never manufacture trades to "stay deployed". An
  // agent with no signal should hold cash. This is the core of honest
  // measurement — the aggregate AI bucket now only trades when the consensus
  // engine emits a real signal.
  if (!MEASUREMENT_MODE) {
    try {
      await autoDeployIdleCash(prices);
    } catch (err) {
      logger.error({ err }, "Auto-deploy idle cash failed (pre-cycle)");
    }
  }

  // Task #383 — pre-cycle aggregates for per-slice meta-brain telemetry.
  // Computed once here so the inner loop is a pure Map lookup (no extra
  // hot-path DB reads per coin/timeframe). All values are nullable: a
  // missing key means "no data yet" and the slice collector emits the
  // `missing:<field>` flag.
  const exposureByCoin = new Map<string, number>();
  const slotPnlState = new Map<
    string,
    { pnlState: number; drawdownState: number }
  >();
  try {
    const [openPositionsAll, fleetPortfoliosAll] = await Promise.all([
      db.select().from(paperPositionsTable),
      db.select().from(paperPortfoliosTable),
    ]);
    const openPositionsForExposure = openPositionsAll.filter((p) =>
      liveExecutorAgentIds.has(p.agentId),
    );
    const fleetPortfolios = fleetPortfoliosAll.filter((p) =>
      liveExecutorAgentIds.has(p.agentId),
    );
    const fleetEquity = fleetPortfolios.reduce(
      (s, p) => s + (p.totalValue ?? 0),
      0,
    );
    if (fleetEquity > 0) {
      const notionalByCoin = new Map<string, number>();
      for (const pos of openPositionsForExposure) {
        notionalByCoin.set(
          pos.coinId,
          (notionalByCoin.get(pos.coinId) ?? 0) + (pos.positionSize ?? 0),
        );
      }
      for (const [coinId, notional] of notionalByCoin) {
        exposureByCoin.set(
          coinId,
          Math.max(0, Math.min(1, notional / fleetEquity)),
        );
      }
    }

    // Recent closed trades per (coinId, timeframe). 7-day lookback caps
    // the row count; per-slot we keep at most the most-recent 20.
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    const recentClosed = await db
      .select({
        coinId: paperTradesTable.coinId,
        timeframe: paperTradesTable.timeframe,
        pnl: paperTradesTable.pnl,
        positionSize: paperTradesTable.positionSize,
        closedAt: paperTradesTable.closedAt,
      })
      .from(paperTradesTable)
      .where(
        and(
          eq(paperTradesTable.status, "closed"),
          gte(paperTradesTable.closedAt, sevenDaysAgo),
        ),
      )
      .orderBy(desc(paperTradesTable.closedAt));

    const bySlot = new Map<
      string,
      Array<{ pnl: number; size: number }>
    >();
    for (const row of recentClosed) {
      if (
        row.pnl === null ||
        row.pnl === undefined ||
        !row.positionSize ||
        row.positionSize <= 0
      ) {
        continue;
      }
      const key = `${row.coinId}|${row.timeframe}`;
      let bucket = bySlot.get(key);
      if (!bucket) {
        bucket = [];
        bySlot.set(key, bucket);
      }
      if (bucket.length < 20) {
        bucket.push({ pnl: row.pnl, size: row.positionSize });
      }
    }
    for (const [key, trades] of bySlot) {
      // Reverse to chronological order (older → newer) so the cumulative
      // running PnL fraction lets us measure peak-to-trough drawdown.
      const ordered = trades.slice().reverse();
      const totalSize = ordered.reduce((s, t) => s + t.size, 0);
      const totalPnl = ordered.reduce((s, t) => s + t.pnl, 0);
      const pnlState = totalSize > 0 ? totalPnl / totalSize : 0;

      let cum = 0;
      let peak = 0;
      let maxDrawdown = 0;
      for (const t of ordered) {
        cum += t.pnl / t.size; // per-trade fractional return
        if (cum > peak) peak = cum;
        const dd = peak - cum;
        if (dd > maxDrawdown) maxDrawdown = dd;
      }
      slotPnlState.set(key, {
        pnlState,
        drawdownState: Math.max(0, maxDrawdown),
      });
    }
  } catch (err) {
    logger.debug(
      { err: String(err) },
      "meta-brain: pre-cycle slot aggregates skipped (non-blocking)",
    );
  }

  for (const agent of agents) {
    if (agent.status === "degraded" && Math.random() > 0.5) continue;

    // Task #444 — slice-driven monitor. Each deterministic "Slice <tf>"
    // agent owns exactly one timeframe (encoded in `preferred_timeframes`
    // by the seed in routes/crypto/index.ts). We intersect the cycle's
    // timeframe set with the agent's owned slice so we get one prediction
    // lane per slice row instead of the old agent×timeframe cross-product.
    let ownedTimeframes: string[] | null = null;
    if (agent.preferredTimeframes) {
      try {
        const parsed = JSON.parse(agent.preferredTimeframes);
        if (Array.isArray(parsed) && parsed.every(x => typeof x === "string")) {
          ownedTimeframes = parsed;
        }
      } catch { /* fall through to legacy behavior */ }
    }
    const agentTimeframes = ownedTimeframes
      ? timeframesThisCycle.filter(tf => ownedTimeframes!.includes(tf))
      : timeframesThisCycle;
    if (agentTimeframes.length === 0) continue;

    if (!agentAccuracyCache[agent.id]) {
      agentAccuracyCache[agent.id] = await getAgentPastAccuracy(agent.id);
    }
    const agentAccuracy = agentAccuracyCache[agent.id];

    const budgetCoins = getRecommendedAgentCoins();
    const agentCoinCount = budgetCoins + Math.floor(Math.random() * 2);

    let specialization;
    if (ablation.agentSpecialization) {
      try {
        specialization = await getAgentSpecialization(agent.id, agent.name);
      } catch { specialization = null; }
    } else {
      specialization = null;
    }

    let coinsForAgent: { coin: CoinPrice; signalStrength: number }[];
    if (specialization) {
      const allocations = allocateCoinsForAgent(agent.id, specialization, topCoins, agentCoinCount);
      coinsForAgent = allocations.map(a => ({ coin: a.coin, signalStrength: a.signalStrength }));
    } else {
      coinsForAgent = topCoins.slice(0, agentCoinCount);
    }

    if (coinsForAgent.length === 0) continue;

    for (const { coin } of coinsForAgent) {
      for (const timeframe of agentTimeframes) {
        const dedupKey = `${agent.id}:${coin.id}:${timeframe}`;
        if (pendingKeys.has(dedupKey)) continue;
        try {
          const patternAnalysis = await analyzePatterns(coin.id, timeframe);
          const pastPatterns = await getPastPredictionPatterns(coin.id, timeframe);

          const fp = generateFingerprint(patternAnalysis);
          updateFingerprintBuffer(coin.id, timeframe, fp.key);

          // Task #444 — regime adjustments still inform stop/take-profit
          // multipliers downstream (see paper-trader). The fingerprint /
          // sequence-match / contagion-context strings used to be packed
          // into the LLM prompt; with the LLM gone those formatters are
          // deleted and we read the structured `regimeAdj` only.
          const regimeAdj = ablation.regimeDetection
            ? getRegimeAdjustments(regimeState.regime)
            : { slMultiplier: 1, tpMultiplier: 1, confidenceModifier: 0 };

          // Phase 4 — fire the Python ml-engine /ml/predict call CONCURRENTLY
          // with the live brain call so the strict 4s shadow timeout never
          // serializes onto the live trading path. Errors / timeouts /
          // circuit-open all resolve to null and are silently dropped.
          const shadowPromise = fetchShadowPrediction(coin.id, timeframe);

          const prediction = await getBrainPrediction({
            coin,
            timeframe,
            patternAnalysis,
          });

          const resolvesAt = new Date(Date.now() + TIMEFRAMES[timeframe].ms);

          const patternCtx = {
            patternType: patternAnalysis.patternType,
            rsi: patternAnalysis.rsi,
            macdCrossover: patternAnalysis.macd.crossover,
            bbPercentB: patternAnalysis.bollingerBands.percentB,
            volatility: patternAnalysis.volatility,
            fingerprint: fp.key,
            regime: regimeState.regime,
            // Phase 5 — when the quant brain ran, persist the raw payload
            // (probabilities, expected return, model version) so the
            // Best-Pick UI can show the EXACT numbers that drove the trade
            // instead of re-calling ml-engine with synthetic inputs.
            quant: prediction.quant ?? null,
          };

          const [inserted] = await db.insert(predictionsTable).values({
            agentId: agent.id,
            agentName: agent.name,
            coinId: coin.id,
            coinName: coin.name,
            direction: prediction.direction,
            confidence: prediction.confidence,
            reasoning: prediction.reasoning,
            priceAtPrediction: coin.currentPrice,
            predictedPrice: prediction.predictedPrice,
            outcome: "pending",
            timeframe,
            resolvesAt,
            patternContext: patternCtx,
            rawConfidence: prediction.rawConfidence ?? null,
            // Tag the model provenance so dashboards can exclude prior-only
            // pooled-fallback predictions from headline accuracy / P&L.
            // NULL for LLM-brain rows.
            source: prediction.quant?.source ?? null,
          }).returning({ id: predictionsTable.id });

          pendingKeys.add(dedupKey);

          // Phase 1 — adaptive engine: emit a prediction-journal row with the
          // full feature snapshot, model provenance, and predicted class
          // probabilities. Fire-and-forget — failures here MUST NOT affect
          // trading behavior.
          const featureVector: Record<string, number> = {
            rsi14: patternAnalysis.rsi,
            macdLine: patternAnalysis.macd.macdLine,
            macdSignal: patternAnalysis.macd.signalLine,
            macdHist: patternAnalysis.macd.histogram,
            bbPctB: patternAnalysis.bollingerBands.percentB,
            bbWidth: patternAnalysis.bollingerBands.width,
            ema9: patternAnalysis.ema.ema9,
            ema21: patternAnalysis.ema.ema21,
            emaSpread: patternAnalysis.ema.spread,
            atr14: patternAnalysis.atr,
            volatility: patternAnalysis.volatility,
            trendStrength: patternAnalysis.trendStrength,
            signalStrength: patternAnalysis.signalStrength,
          };
          const q = prediction.quant;
          // Phase 5 — look up the registry BEFORE the live insert so the
          // primary (non-shadow) row carries the active champion's
          // registry id. summarizeShadowMetrics(champion.id) then has
          // real live rows to compute its baseline against, instead of
          // the empty set the previous design produced. Best-effort —
          // never blocks the trading path; on any failure we simply
          // write the live row with a null registryId.
          let championRegistryId: number | null = null;
          let cachedRegistryRows: Awaited<ReturnType<typeof listRegistry>> | null = null;
          if (q && (q.source || q.modelVersion)) {
            const slotModelId = q.source ?? "lightgbm";
            try {
              cachedRegistryRows = await listRegistry();
              const champ = cachedRegistryRows.find(
                (r) =>
                  r.modelId === slotModelId &&
                  (r.coinId === coin.id || r.coinId === "*") &&
                  (r.timeframe === timeframe || r.timeframe === "*") &&
                  r.state === "champion" &&
                  r.isActive,
              );
              if (champ) championRegistryId = champ.id;
            } catch {
              cachedRegistryRows = null;
            }
          }
          // AWAIT the insert so the subsequent markPredictionJournalTrade
          // status update is guaranteed to find the row (no race). The
          // writer swallows errors internally — never throws into the
          // trading path. Status update is still race-safe via retry,
          // but awaiting eliminates the common case entirely.
          // Task #405 / B-LLM-AUTHORSHIP — stamp the brain that
           // ACTUALLY produced the prediction. The orchestrator returns
           // `prediction.brain === "ABSTAIN"` when no quant model is
           // available; the previous `q ? "QUANT" : "LLM"` collapse
           // silently re-labelled abstains as LLM-authored, violating
           // the "LLM cannot author trades" contract. The downstream
           // writer (writePredictionJournal) refuses any non-QUANT row
           // with a non-stable direction, so a regression here fails loud.
          const stampedBrain: "QUANT" | "ABSTAIN" | "LLM" =
            prediction.brain === "QUANT"
              ? "QUANT"
              : prediction.brain === "ABSTAIN"
                ? "ABSTAIN"
                : "LLM";
          await writePredictionJournal({
            shadow: false,
            registryId: championRegistryId,
            predictionId: inserted.id,
            brain: stampedBrain,
            refreshFeaturesFor: { coinId: coin.id, timeframe },
            agentId: agent.id,
            agentName: agent.name,
            coinId: coin.id,
            coinName: coin.name,
            timeframe,
            modelId: q?.source ?? (stampedBrain === "ABSTAIN" ? "abstain" : "legacy_llm"),
            modelVersion: q?.modelVersion ?? null,
            source: q?.source ?? null,
            featureHash: q?.featureHash ?? null,
            featureVector,
            // Phase 2 — first-class 6-class regime label from ml-engine
            // (e.g. "trending_up" / "panic_liquidation"). Falls back to
            // a deterministic mapping from the legacy 4-class regime so
            // we never write a null label when a regime exists.
            regimeLabel: regimeState.regimeLabel ?? regimeState.regime ?? null,
            direction: prediction.direction,
            confidence: prediction.confidence,
            rawConfidence: prediction.rawConfidence ?? q?.rawConfidence ?? null,
            probUp: q?.probUp ?? null,
            probDown: q?.probDown ?? null,
            probStable: q?.probStable ?? null,
            expectedReturnPct: q?.expectedReturnPct ?? null,
            predictionStdPct: q?.predictionStdPct ?? null,
            priceAtPrediction: coin.currentPrice,
            predictedPrice: prediction.predictedPrice,
            // Phase 4 — regime label and specialist sidecar are now
            // first-class columns on prediction_journal (regime_label /
            // specialist_scores). gates_applied stays focused on actual
            // gating decisions. Specialists do NOT gate the trade —
            // Phase 4's meta-model is the gate; this is observability-only.
            gatesApplied: {
              noTradeZone: prediction.isNoTradeZone === true,
              // Phase 4a — specialist meta-gate decision (always present
              // on quant rows) so /crypto/brain/abstain-rate can compute
              // abstain rate per regime straight from the journal.
              ...(q?.metaGate ? { metaGate: q.metaGate } : {}),
              // Phase 4b — learned meta-model decision in the gates JSON.
              // The /crypto/meta/* diagnostics endpoints read these keys
              // to build the action-distribution + edge-decile cards
              // without re-running inference.
              ...(q?.metaAction ? { meta_action: q.metaAction } : {}),
              ...(q?.metaSizeMultiplier !== null && q?.metaSizeMultiplier !== undefined
                ? { meta_size_multiplier: q.metaSizeMultiplier }
                : {}),
              ...(q?.metaExpectedEdgePct !== null && q?.metaExpectedEdgePct !== undefined
                ? { meta_expected_edge_pct: q.metaExpectedEdgePct }
                : {}),
              ...(q?.metaAbstainReason ? { meta_abstain_reason: q.metaAbstainReason } : {}),
              ...(q?.metaKind ? { meta_kind: q.metaKind } : {}),
              ...(q?.metaVersion ? { meta_version: q.metaVersion } : {}),
            },
            specialistScores:
              q?.specialists && q.specialists.length > 0
                ? q.specialists
                : null,
            becameTrade: null, // updated after executePaperTrade below
            skipReason: null,
            tradeId: null,
            resolvesAt,
          });

          // Phase 5 — registry-aware shadow journaling. For every
          // non-champion (shadow / challenger) entry registered for this
          // (modelId, coinId, timeframe) slot, write an extra prediction
          // journal row tagged `shadow=true` + `registryId`. The promotion
          // gate evaluator (`summarizeShadowMetrics`) reads these to score
          // each challenger against the champion without trading on it.
          // Fire-and-forget; never blocks the trading path.
          if (q && (q.source || q.modelVersion)) {
            const slotModelId = q.source ?? "lightgbm";
            void (cachedRegistryRows
              ? Promise.resolve(cachedRegistryRows)
              : listRegistry()
            )
              .then(async (rows) => {
                const candidates = rows.filter(
                  (r) =>
                    r.modelId === slotModelId &&
                    (r.coinId === coin.id || r.coinId === "*") &&
                    (r.timeframe === timeframe || r.timeframe === "*") &&
                    r.state !== "champion" &&
                    r.state !== "retired" &&
                    r.state !== "quarantined" &&
                    r.isActive,
                );
                for (const cand of candidates) {
                  await writePredictionJournal({
                    predictionId: inserted.id,
                    brain: "QUANT",
                    agentId: agent.id,
                    agentName: agent.name,
                    coinId: coin.id,
                    coinName: coin.name,
                    timeframe,
                    modelId: cand.modelId,
                    modelVersion: cand.modelVersion,
                    source: cand.modelId,
                    featureHash: q.featureHash ?? null,
                    featureVector,
                    regimeLabel: regimeState.regimeLabel ?? regimeState.regime ?? null,
                    direction: prediction.direction,
                    confidence: prediction.confidence,
                    rawConfidence: q.rawConfidence ?? null,
                    probUp: q.probUp ?? null,
                    probDown: q.probDown ?? null,
                    probStable: q.probStable ?? null,
                    expectedReturnPct: q.expectedReturnPct ?? null,
                    predictionStdPct: q.predictionStdPct ?? null,
                    priceAtPrediction: coin.currentPrice,
                    predictedPrice: prediction.predictedPrice,
                    gatesApplied: { noTradeZone: false, shadow: true },
                    specialistScores: q.specialists && q.specialists.length > 0 ? q.specialists : null,
                    becameTrade: false,
                    skipReason: "shadow_only",
                    tradeId: null,
                    resolvesAt,
                    shadow: true,
                    registryId: cand.id,
                  });
                }
              })
              .catch((err) => {
                logger.warn(
                  { err, coinId: coin.id, timeframe },
                  "shadow journal write failed",
                );
              });
          }

          // Persist shadow row (read-only — never affects the trade path).
          shadowPromise.then((shadow) =>
            recordShadowPrediction({
              shadow,
              coinId: coin.id,
              coinName: coin.name,
              timeframe,
              priceAtPrediction: coin.currentPrice,
            }),
          ).catch(() => { /* swallow — never touch trading path */ });

          // Task #373 — push this slice into the meta-brain tick
          // collector. Fully bounded, never throws on this path. The
          // batch is flushed once per cycle at end-of-tick, and the
          // returned directive shapes the NEXT tick's sizing (with
          // one-cycle lag; 30s << shortest active timeframe).
          try {
            const q = prediction.quant;
            const regimeLabel =
              regimeState.regimeLabel ?? regimeState.regime ?? "unknown";
            // Task #468 — strategy_family is resolved through the
            // typed strategy-profile registry. Every agents row carries
            // a `profile_id` (sweep-populated at boot via the legacy
            // compatibility map). Unknown ids throw — there is no
            // silent default. The legacy name-based resolver is kept
            // only for the historical contract test.
            const family = metaBrainResolveFamilyForProfile(agent.profileId);
            const probMax = q
              ? Math.max(q.probUp, q.probDown, q.probStable)
              : prediction.confidence;
            const disagreement = q && q.specialists
              ? (() => {
                  const votes = q.specialists
                    .filter((s: { applicable?: boolean }) => s.applicable)
                    .map((s: { probUp?: number | null; probDown?: number | null }) =>
                      (s.probUp ?? 0) - (s.probDown ?? 0),
                    );
                  if (votes.length < 2) return 0;
                  const mean =
                    votes.reduce((a: number, b: number) => a + b, 0) / votes.length;
                  const variance =
                    votes.reduce(
                      (a: number, b: number) => a + (b - mean) ** 2,
                      0,
                    ) / votes.length;
                  return Math.min(1, Math.sqrt(variance));
                })()
              : 0;
            // Task #381 step 4 — real telemetry. Anything we can
            // truthfully compute we send; anything we cannot, we send
            // as `null` and the adapter records a `missing:<field>`
            // flag so the brain's trust updater can down-weight it.
            const recentAccuracy =
              agentAccuracy && agentAccuracy.totalPredictions > 0
                ? agentAccuracy.accuracy
                : null;
            // calibrated_confidence: `prediction.confidence` is the
            // calibrated probability max produced by the calibrator.
            // The previous code passed `q.rawConfidence` (the
            // pre-calibration value), which mis-labelled the brain's
            // input. Use the calibrated value here.
            const calibrated = prediction.confidence;
            // Slippage applied at fill time is a constant pct in this
            // codebase — convert to basis points.
            const slippageBps = Number.isFinite(SLIPPAGE_PCT)
              ? SLIPPAGE_PCT * 10000
              : null;
            metaBrainCollectSlice({
              coin: coin.id,
              timeframe,
              strategy_family: family,
              edge: q?.expectedReturnPct ?? 0,
              confidence: prediction.confidence,
              calibrated_confidence: calibrated,
              risk_score: 1 - prediction.confidence,
              recent_accuracy: recentAccuracy,
              // Task #383 — per-slot pnl_state / drawdown_state from
              // the most recent ≤20 closed trades on this (coin,
              // timeframe) over a 7-day window. Null until the slot
              // has at least one closed trade with non-null pnl.
              pnl_state: slotPnlState.get(`${coin.id}|${timeframe}`)?.pnlState ?? null,
              drawdown_state: slotPnlState.get(`${coin.id}|${timeframe}`)?.drawdownState ?? null,
              disagreement,
              prediction_error: null, // no rolling tracker yet
              regime: regimeLabel,
              volatility: patternAnalysis.atr ?? 0,
              correlation_shift: null, // fleet-level tracker pending
              // Task #383 — per-coin exposure as a fraction of fleet
              // equity (timeframe-agnostic). Null until the coin has
              // an open position.
              exposure: exposureByCoin.get(coin.id) ?? null,
              turnover: null, // requires open/close churn tracker
              slippage_bps: slippageBps,
            });
          } catch (err) {
            // Collection is advisory. Swallow — never affect trades.
            logger.debug(
              { err: String(err), coinId: coin.id, timeframe },
              "meta-brain: slice collection failed (non-blocking)",
            );
          }

          const isNoTradeZone = prediction.isNoTradeZone === true || prediction.reasoning?.includes("[NO-TRADE ZONE");
          const metaAbstained = prediction.quant?.metaGate?.abstained === true;
          // Task #255 — quant brain authoritative. ABSTAIN means no model
          // is available for this (coin, timeframe). We deliberately do
          // NOT fall through to LLM-as-trade-author. The journal carries
          // the abstain reason for diagnostics.
          const quantAbstain = prediction.brain === "ABSTAIN";
          if (quantAbstain) {
            logger.info(
              { agent: agent.name, coin: coin.name, timeframe, reason: prediction.abstainReason ?? "no_model" },
              "Skipping paper trade — quant brain abstained (LLM cannot author trades)",
            );
            await markPredictionJournalTrade(inserted.id, {
              becameTrade: false,
              skipReason: `quant_abstain_${prediction.abstainReason ?? "no_model"}`,
            });
          } else if (isNoTradeZone) {
            logger.info({ agent: agent.name, coin: coin.name, timeframe, confidence: prediction.confidence }, "Skipping paper trade — NO-TRADE ZONE (model disagreement with similar confidence)");
            await markPredictionJournalTrade(inserted.id, {
              becameTrade: false,
              skipReason: "no_trade_zone",
            });
          } else if (metaAbstained) {
            // Phase 4 — specialist meta-gate told us to skip. The
            // direction is already collapsed to "stable" by the brain,
            // so executePaperTrade would silently no-op. We mark the
            // journal explicitly so the diagnostics page can count this
            // as an abstain (not a generic "gated") and so the skip
            // reason chips show what actually happened.
            const meta = prediction.quant!.metaGate!;
            logger.info(
              {
                agent: agent.name,
                coin: coin.name,
                timeframe,
                regime: meta.regime,
                reason: meta.reason,
                applicable: meta.applicable,
                agreement: meta.agreementVotes,
                meanScore: meta.meanDirectionalScore.toFixed(3),
              },
              "Skipping paper trade — quant meta-gate abstained on specialist signals",
            );
            await markPredictionJournalTrade(inserted.id, {
              becameTrade: false,
              skipReason: "quant_meta_abstain",
              gatesApplied: { quant_meta_abstain: true },
            });
          } else if (prediction.brain !== "QUANT") {
            // Task #255 boundary guard. The orchestrator already refuses to
            // emit an LLM-authored trade decision, but we add a second guard
            // here so any future regression that produces a non-QUANT,
            // non-ABSTAIN brain still cannot reach `executePaperTrade`.
            logger.warn(
              { agent: agent.name, coin: coin.name, timeframe, brain: prediction.brain },
              "Refusing to execute paper trade — only the QUANT brain may author trades (Task #255)",
            );
            await markPredictionJournalTrade(inserted.id, {
              becameTrade: false,
              skipReason: `non_quant_brain_${prediction.brain}`,
            });
          } else {
            try {
              await executePaperTrade(
                agent.id, agent.name, coin.id, coin.name,
                prediction.direction, prediction.confidence,
                coin.currentPrice, timeframe, inserted.id,
                patternAnalysis.atr,
                // Phase 5 — pass the quant payload through so the trader
                // can apply an EV-after-fees gate.
                prediction.quant,
              );
            } catch (tradeErr) {
              logger.error({ err: tradeErr }, "Paper trade failed");
            }

            // Detect whether the trade actually opened by looking up the
            // paper_trades row keyed by predictionId. Either updates the
            // journal with the trade id or marks it as gated.
            try {
              const [openedTrade] = await db
                .select({ id: paperTradesTable.id })
                .from(paperTradesTable)
                .where(eq(paperTradesTable.predictionId, inserted.id))
                .limit(1);
              if (openedTrade) {
                await markPredictionJournalTrade(inserted.id, {
                  becameTrade: true,
                  tradeId: openedTrade.id,
                });
              } else {
                // Pull the structured gate reason from the skip-tracker
                // telemetry the paper-trader just emitted; fall back to a
                // generic "gated" only if no skip event was recorded.
                const gate = await lookupGateForPrediction(agent.id, coin.id, new Date());
                await markPredictionJournalTrade(inserted.id, {
                  becameTrade: false,
                  skipReason: gate.skipReason ?? "gated",
                  gatesApplied: gate.gatesApplied,
                });
              }
            } catch (lookupErr) {
              logger.warn({ err: lookupErr, predictionId: inserted.id }, "journal trade-status lookup failed");
            }
          }

          predictionsCreated++;
        } catch (err) {
          logger.error({ err, agent: agent.name, coin: coin.id, timeframe }, "Prediction failed");
        }
      }
    }
  }

  cycleCount++;

  const recentPredictions = await db
    .select()
    .from(predictionsTable)
    .orderBy(desc(predictionsTable.createdAt))
    .limit(150);

  try {
    const projectedPredictions = recentPredictions.map(p => ({
      agentId: p.agentId,
      agentName: p.agentName,
      coinId: p.coinId,
      coinName: p.coinName,
      direction: p.direction,
      confidence: p.confidence,
      reasoning: p.reasoning,
      priceAtPrediction: p.priceAtPrediction,
      predictedPrice: p.predictedPrice,
      outcome: p.outcome,
      timeframe: p.timeframe,
      createdAt: p.createdAt,
    }));
    latestBestPick = await computeBestSlice(agents, projectedPredictions, prices);
  } catch (err) {
    logger.error({ err }, "Failed to compute best slice");
  }

  if (cycleCount % 10 === 0) {
    try {
      const corrTimeframe = timeframesThisCycle.includes("1h") ? "1h" : timeframesThisCycle[0];
      await computeCoinCorrelations(corrTimeframe);
    } catch (err) {
      logger.error({ err }, "Correlation computation failed");
    }
  }

  try {
    await runStrategyLabCycle(prices);
  } catch (err) {
    logger.error({ err }, "Strategy lab cycle failed");
  }

  // Task #444 — `classifyCycleNews` was the last surviving LLM call site
  // (one Gemini batch per cycle to tag market news). Both the news-fetcher
  // and the classifier are deleted in this task. The auto-revert guard
  // stays — it now only watches the quant brain's live shadow win-rate
  // versus the offline backtest baseline.
  try {
    await evaluateBrainAutoRevert();
  } catch (err) {
    logger.warn({ err }, "brain-auto-revert evaluation failed (non-critical)");
  }

  try {
    const attribution = await getAutoDeployAttribution(prices);
    const wrote = await recordAutoDeployAttributionSnapshot(attribution);
    if (wrote) {
      logger.info({
        net7d: attribution.window7d.total.netPnlUsd.toFixed(2),
        openUnrealized: attribution.open.total.unrealizedPnlUsd.toFixed(2),
      }, "Recorded auto-deploy attribution snapshot");
    }
  } catch (err) {
    logger.error({ err }, "Auto-deploy attribution snapshot failed");
  }

  // Task #381 step 4 — push fleet-level portfolio telemetry into the
  // meta-brain's tick batch BEFORE flushing. Aggregated across all
  // agents so the brain sees one fleet-wide snapshot per tick.
  try {
    const portfoliosAll = await db.select().from(paperPortfoliosTable);
    const portfolios = portfoliosAll.filter((p) =>
      liveExecutorAgentIds.has(p.agentId),
    );
    if (portfolios.length > 0) {
      const totalValue = portfolios.reduce((s, p) => s + p.totalValue, 0);
      const totalPeak = portfolios.reduce(
        (s, p) => s + (p.peakValue ?? p.totalValue),
        0,
      );
      const totalInitial = portfolios.length * 1000; // INITIAL_BALANCE
      const openPositionsAll = await db
        .select()
        .from(paperPositionsTable);
      const openPositions = openPositionsAll.filter((p) =>
        liveExecutorAgentIds.has(p.agentId),
      );
      const totalExposure = openPositions.reduce(
        (s, pos) => s + pos.positionSize,
        0,
      );
      const drawdown =
        totalPeak > 0 ? Math.max(0, (totalPeak - totalValue) / totalPeak) : 0;
      const exposureRatio = totalValue > 0 ? totalExposure / totalValue : 0;
      const pnlState =
        totalInitial > 0 ? (totalValue - totalInitial) / totalInitial : 0;
      metaBrainSetPortfolio({
        total_drawdown: drawdown,
        realized_vol: null, // requires rolling vol tracker
        concentration: Math.min(1, exposureRatio),
        leverage: null, // no leverage in paper trading
        liquidity_stress: null, // requires orderbook depth
        correlation_shift: null, // fleet-level rolling tracker pending
        active_risk_budget: Math.max(0, Math.min(1, 1 - drawdown)),
        kill_switch_distance: Math.max(0, Math.min(1, 1 - drawdown / 0.2)),
        anomaly_flags: [
          `pnl_state:${pnlState.toFixed(4)}`,
          `exposure:${exposureRatio.toFixed(4)}`,
        ],
      });
    }
  } catch (err) {
    logger.debug(
      { err: String(err) },
      "meta-brain: portfolio telemetry skipped (non-blocking)",
    );
  }

  // Task #390 — assemble Strategy Lab benchmark telemetry (read-only,
  // 7d / 14d windows over `strategy_snapshots`) and stash it on the
  // meta-brain adapter for inclusion in the next flush batch. Pure
  // governance — never touches the predictor or trade-decision path.
  try {
    metaBrainResetBenchmarkCache();
    const benchmark = await metaBrainAssembleBenchmark();
    metaBrainSetBenchmark(benchmark);
  } catch (err) {
    metaBrainSetBenchmark(null);
    logger.debug(
      { err: String(err) },
      "meta-brain: benchmark telemetry skipped (non-blocking)",
    );
  }

  // Task #373 — flush the meta-brain's per-tick telemetry batch. Runs
  // exactly once per cycle, AFTER all predictions have been journaled
  // and any trade decisions have fired. The returned directive is
  // cached and shapes the NEXT tick's sizing. Never blocks; on any
  // failure the cached directive stays at the previous tick's value
  // (or neutral on first boot / disabled).
  try {
    await metaBrainFlushTick();
  } catch (err) {
    logger.warn(
      { err: String(err) },
      "meta-brain: flushTick threw (non-blocking — sizing stays at last directive)",
    );
  }

  await db
    .update(monitoringStateTable)
    .set({
      cycleCount,
      lastCycleAt: new Date(),
      nextCycleAt: new Date(Date.now() + 30000),
      isRunning: true,
    })
    .where(eq(monitoringStateTable.id, 1));

  // Task #453 — apiMode/gptUsed/geminiUsed log fields removed alongside
  // rate-limiter.ts. Nothing has called `recordApiCall` since the LLM
  // plane was retired in Task #444, so those fields were structurally
  // pinned at "dual" / 0 / 0 every cycle and added pure noise.
  logger.info({
    cycleCount,
    predictionsCreated,
    timeframes: timeframesThisCycle,
    topCoins: topCoins.map(c => `${c.coin.symbol}(${c.signalStrength})`).join(", "),
  }, "Analysis cycle completed");

  return { predictionsCreated, cycleNumber: cycleCount };
  } finally {
    isRunningCycle = false;
    cycleStartedAt = null;
  }
}

export function startMonitoring(): void {
  if (monitorInterval) return;

  logger.info("Starting crypto monitoring engine with multi-timeframe analysis");

  const CYCLE_INTERVAL = 30000;
  monitorInterval = setInterval(async () => {
    try {
      await runAnalysisCycle();
    } catch (err) {
      logger.error({ err }, "Analysis cycle failed");
    }
  }, CYCLE_INTERVAL);

  resolutionInterval = setInterval(async () => {
    try {
      await resolvePendingPredictions();
      await resolveShadowPredictions();
    } catch (err) {
      logger.error({ err }, "Resolution check failed");
    }
  }, 15000);

  paperTradeInterval = setInterval(async () => {
    try {
      await closeExpiredPositions();
      await updatePortfolioValues();
    } catch (err) {
      logger.error({ err }, "Paper trade update failed");
    }
  }, 15000);

  // Auto-loosen suggestion: refresh every 5 min
  void refreshTuningSuggestion();
  tuningSuggestionInterval = setInterval(() => {
    void refreshTuningSuggestion();
  }, 5 * 60 * 1000);

  // Scheduled ML retrain: 3 times per 24h (every 8h). The model improves
  // automatically as price_history accumulates more real ticks.
  retrainInterval = setInterval(() => {
    void triggerScheduledRetrain();
  }, RETRAIN_INTERVAL_MS);

  // Journal retention: roll up + trim prediction_journal / trade_journal rows
  // older than JOURNAL_RETENTION_DAYS (default 90) into journal_rollup.
  // The function self-throttles to once per hour even if invoked more often.
  journalRetentionInterval = setInterval(() => {
    void runJournalRetention().catch((err) =>
      logger.error({ err }, "journal-retention: scheduled run failed"),
    );
    // Task #257 — piggy-back on the hourly retention tick to also trim the
    // un-quarantine audit log (default 180d retention, keep most recent N
    // per candidate). The inner helper self-throttles to once/hour.
    void pruneUnquarantineEventsIfDue().catch((err) =>
      logger.error({ err }, "feature-lab: scheduled unquarantine prune failed"),
    );
    // Task #292 — also trim `market_signals` rows older than
    // MARKET_SIGNALS_RETENTION_DAYS (default 30) so the table doesn't grow
    // unbounded (~17k rows/day) and the per-coin health aggregate stays
    // fast. The inner helper self-throttles to once/hour.
    void pruneMarketSignalsIfDue().catch((err) =>
      logger.error({ err }, "market-signals-retention: scheduled run failed"),
    );
  }, 60 * 60 * 1000);

  // Task #258 — push-channel notifications for auto-retired features.
  // Seed the dedup set on first boot so we don't blast every existing
  // quarantine entry, then start a 60s poll. Channels (Slack / email
  // webhook) are configured via env; with no env vars set the loop is
  // a no-op so this is safe to run unconditionally.
  void seedDedupSetFromCurrentQuarantine().catch((err) =>
    logger.warn({ err }, "auto-retire-notifier: startup seed failed"),
  );
  startAutoRetireNotifierLoop(60 * 1000);

  // Task #279 — push-channel notifications for training-contract failures.
  // Same pattern as #258: seed the dedup set on first boot so we don't
  // page on a stale failing run, then poll once a minute. Channels are
  // env-driven (TRAINING_CONTRACT_SLACK_WEBHOOK_URL / SLACK_WEBHOOK_URL /
  // TRAINING_CONTRACT_EMAIL_WEBHOOK_URL); no env => no-op loop.
  void seedTrainingContractDedup().catch((err) =>
    logger.warn({ err }, "training-contract-notifier: startup seed failed"),
  );
  startTrainingContractNotifierLoop(60 * 1000);

  // Task #290 — off-dashboard alerting for the market signals poller.
  // A 60s background tick checks the in-process poller status and fires
  // a debounced webhook when the poller is stale (>3x interval) or its
  // last poll errored. Recovery sends a one-shot ping. Channels (generic
  // webhook + Slack) are env-driven; with no env vars set the loop just
  // tracks state for dashboard surfacing.
  startMarketSignalsPollerWatcher(60 * 1000);

  // Task #411 — off-dashboard alerting for the daily 5m head top-up
  // (#410). Polls /ml/admin/5m-topup/status every 60s and fires a
  // webhook when last_attempt_outcome=="error" for two consecutive
  // scheduler ticks OR last_alerts is non-empty. Channels are env-driven
  // (TOPUP_5M_ALERT_WEBHOOK_URL / TOPUP_5M_ALERT_SLACK_WEBHOOK_URL /
  // SLACK_WEBHOOK_URL); with no env vars set the loop still maintains
  // dedup state for the dashboard banner.
  startTopup5mNotifierLoop(60 * 1000);

  // Task #577 — sweep loop for the disabled-outcome notifier. Incident
  // dispatch is push-driven (see disabled-outcome-notifier.ts), but the
  // recovery dispatch needs a periodic re-check to fire when the
  // active window finally clears (no rejection event arrives at that
  // moment to trigger inline). 60s mirrors the cadence of the other
  // notifier loops.
  startDisabledOutcomeNotifierLoop(60 * 1000);

  setTimeout(async () => {
    try {
      await seedHistoricalPriceData();
      // Task #512 — make sure the 4 deterministic executor agent rows
      // exist BEFORE paper portfolios are initialized so each new
      // executor immediately gets a $1000 starting balance from
      // initializePaperPortfolios. Idempotent.
      await seedExecutorAgents();
      await initializePaperPortfolios();
      await ensureStrategyLabAgents();
      // Task #468 — sweep every agents row through the legacy
      // compatibility map BEFORE the first analysis cycle so the
      // trade-path's registry gate sees a populated profile_id.
      await syncAgentProfileIds();
      // Then load the agent → profile cache that the trade-execution
      // path reads from. The cache is rebuilt only on SIGHUP — the
      // gate must NEVER do a per-decision DB lookup (spec line 50).
      await loadAgentRegistryCache();
      installSighupReload();
      // Nightly retirement evaluation — first pass runs ~10s after
      // boot so the dashboard has a snapshot quickly.
      startRetirementLoop();
      await runAnalysisCycle();
      // Kick off a journal cleanup pass shortly after startup so a freshly
      // restarted process (e.g. after a backfill) doesn't have to wait an
      // hour for the first scheduled tick.
      void runJournalRetention(true).catch((err) =>
        logger.error({ err }, "journal-retention: startup run failed"),
      );
      void pruneUnquarantineEventsIfDue(true).catch((err) =>
        logger.error({ err }, "feature-lab: startup unquarantine prune failed"),
      );
      void pruneMarketSignalsIfDue(true).catch((err) =>
        logger.error({ err }, "market-signals-retention: startup run failed"),
      );
    } catch (err) {
      logger.error({ err }, "Initial analysis cycle failed");
    }
  }, 5000);
}

export function getCycleCount(): number {
  return cycleCount;
}

export function resetCycleCount(): void {
  cycleCount = 0;
}

export function getLatestBestPick(): BestPickResult | null {
  return latestBestPick;
}
