import { db, paperPortfoliosTable, paperTradesTable, paperPositionsTable, paperPositionMarksTable, agentsTable, autoDeployAttributionSnapshotsTable } from "@workspace/db";
import { eq, and, lte, gte, desc, isNull, inArray, sql } from "drizzle-orm";
import { logger } from "./logger";
import { recordSkip, type SkipReason } from "./skip-tracker";
import { isHorizonDisabled } from "./horizon-gate";
import { getRoleForTimeframe } from "./timeframe-roles";
import {
  getMinConfidenceToTrade,
  getQuantMinConfidenceToTrade,
  getCounterTrendMinConfidence,
  getMinTpDistancePct,
  getMinEvVsCost,
} from "./tuning-tracker";
import type { CoinPrice } from "./coins";
import { fetchCoinPrices, isPriceDataFresh } from "./coins";
import { TIMEFRAMES, type TimeframeKey } from "./pattern-analyzer";
import { getCurrentRegime } from "./regime-detector";

// Fee/slippage constants are now imported from the shared trading-constants
// module so the live paper-trader, the calibration scoring code, the
// Python backtester, and the unit test suite can never disagree about what
// a round trip costs. ALL numeric values come from
// shared/trading-frictions.json — see trading-constants.ts header.
import {
  MAKER_FEE_PCT,
  TAKER_FEE_PCT,
  SLIPPAGE_PCT,
  ROUND_TRIP_COST_PCT,
  INITIAL_BALANCE_USD as SHARED_INITIAL_BALANCE,
  MAX_OPEN_POSITIONS_PER_AGENT as SHARED_MAX_OPEN_POSITIONS_PER_AGENT,
  MAX_PORTFOLIO_AT_RISK as SHARED_MAX_PORTFOLIO_AT_RISK,
  MAX_CASH_PER_POSITION_PCT,
  KELLY_MIN_TRADES as SHARED_KELLY_MIN_TRADES,
  KELLY_RAMP_END as SHARED_KELLY_RAMP_END,
  KELLY_FRACTION as SHARED_KELLY_FRACTION,
  DEFAULT_POSITION_PCT as SHARED_DEFAULT_POSITION_PCT,
  MAX_POSITION_PCT as SHARED_MAX_POSITION_PCT,
  DAILY_LOSS_LIMIT_PCT as SHARED_DAILY_LOSS_LIMIT_PCT,
  DRAWDOWN_HALT_PCT as SHARED_DRAWDOWN_HALT_PCT,
  ASYMMETRIC_LONG_MIN_CONFIDENCE,
  TRADEABLE_TIMEFRAMES_SET,
  FLEET_BRAKE_MIN_OPEN as SHARED_FLEET_BRAKE_MIN_OPEN,
  FLEET_BRAKE_DOMINANCE as SHARED_FLEET_BRAKE_DOMINANCE,
  RECENT_LOSS_BLOCK_COUNT,
  tieredPositionPct as sharedTieredPositionPct,
  getSlMultiplier,
  getTpMultiplier,
  getAtrFloorPct,
  TRAILING_STOP,
} from "./trading-constants";
// Pure helpers shared with the test suite. The live trader MUST call these
// (not reimplement the math) so any regression to the SL/TP trigger or the
// fee/slippage application is caught by `pnpm --filter @workspace/api-server test`.
import {
  applyEntrySlippage,
  applyExitSlippage,
  entryFee as computeEntryFee,
  exitFee as computeExitFee,
  recoverEntryFee,
  isStopLossHit,
  isTakeProfitHit,
} from "./trade-math";
import { writeTradeJournal } from "./journal-writer";
import { checkPortfolioConstraints } from "./portfolio-constraints";
import { getMlDecision } from "./ml-client";
import {
  AgentNotExecutableError,
  getCachedProfileForAgentId,
  profileAllowsRegime,
  type AgentProfile,
} from "./agents-registry";
// Task #373 — supervisory meta-brain. Never predicts price or places
// trades; only shapes sizing (clamped 0.5–1.5 by the existing
// metaSizeMultiplier clamp), and feeds bounded-learning reward signal
// on close. See audit/meta-brain-integration.md.
import {
  bindTradeToTick,
  clearTickBinding,
  getActiveDirective,
  getFamilySizeMultiplier,
  isFamilySuppressed as metaBrainIsFamilySuppressed,
  isNeutralDirective as metaBrainIsNeutralDirective,
  peekTickForTrade,
  resolveStrategyFamily as metaBrainResolveFamily,
  sendRecordOutcome,
} from "./meta-brain";
// Task #614 — Minimum Truthful Trading Mode. Self-contained lane: a
// pre-decision whitelist guard, a position-size override, and a
// per-trade-close auto-disable check. All other risk math is unchanged.
import {
  evaluateMttmAutoDisable,
  getMttmConfig,
  getMttmConfigCached,
  slotKey as mttmSlotKey,
} from "./mttm";

// Trade-entry gate thresholds (MIN_CONFIDENCE_TO_TRADE, MIN_TP_DISTANCE_PCT,
// MIN_EV_VS_COST, COUNTER_TREND_MIN_CONFIDENCE) are sourced from
// `tuning-tracker.ts` so they can be auto-loosened when bots are starving for
// trades and reverted from the dashboard. Baselines live in
// shared/trading-frictions.json under `gates_baseline`.
/**
 * Required quant `expectedReturnPct` for the EV gate around L246-268. Exported
 * so the brain↔trader parity test (and quant-brain's `defaultMinExpRetPct`)
 * compute against the SAME source of truth — any future drift on either side
 * is caught by `test/quant-brain-trader-parity.test.ts`.
 *
 * `ROUND_TRIP_COST_PCT` is a fraction (0.003) and quant.expectedReturnPct is a
 * percent (0.30), so we multiply by 100 to compare on the same scale.
 */
export function getQuantEvGateRequiredPct(): number {
  return getMinEvVsCost() * ROUND_TRIP_COST_PCT * 100;
}

export const HODLER_AGENT_NAME = "Hodler Hank";
const HODLER_TARGET_DEPLOYMENT = 0.95;
const HODLER_REBALANCE_THRESHOLD = 0.05;

// Re-bind the shared (json-sourced) constants to the local names already used
// throughout this file so we don't have to rewrite every call site.
const INITIAL_BALANCE = SHARED_INITIAL_BALANCE;
const TRADEABLE_TIMEFRAMES = TRADEABLE_TIMEFRAMES_SET;
const MAX_OPEN_POSITIONS_PER_AGENT = SHARED_MAX_OPEN_POSITIONS_PER_AGENT;
const MAX_PORTFOLIO_AT_RISK = SHARED_MAX_PORTFOLIO_AT_RISK;
const KELLY_MIN_TRADES = SHARED_KELLY_MIN_TRADES;
const KELLY_RAMP_END = SHARED_KELLY_RAMP_END;
const KELLY_FRACTION = SHARED_KELLY_FRACTION;
const DEFAULT_POSITION_PCT = SHARED_DEFAULT_POSITION_PCT;
const MAX_POSITION_PCT = SHARED_MAX_POSITION_PCT;
const DAILY_LOSS_LIMIT_PCT = SHARED_DAILY_LOSS_LIMIT_PCT;
const DRAWDOWN_HALT_PCT = SHARED_DRAWDOWN_HALT_PCT;

const tieredPositionPct = sharedTieredPositionPct;

let isClosingPositions = false;

interface RiskCheckResult {
  dailyLossHit: boolean;
  drawdownHit: boolean;
  updatedPeak: number;
  updatedDayStart: number;
  updatedDayDate: string;
}

export function checkRiskLimits(
  currentValue: number,
  peakValue: number | null,
  dayStartValue: number | null,
  dayStartDate: string | null
): RiskCheckResult {
  const today = new Date().toISOString().slice(0, 10);
  const peak = peakValue ?? currentValue;
  const effectivePeak = currentValue > peak ? currentValue : peak;

  let effectiveDayStart = dayStartValue ?? currentValue;
  let effectiveDayDate = dayStartDate ?? today;
  if (effectiveDayDate !== today) {
    effectiveDayStart = currentValue;
    effectiveDayDate = today;
  }

  const dayLoss = (effectiveDayStart - currentValue) / effectiveDayStart;
  const drawdown = (effectivePeak - currentValue) / effectivePeak;

  return {
    dailyLossHit: dayLoss >= DAILY_LOSS_LIMIT_PCT,
    drawdownHit: drawdown >= DRAWDOWN_HALT_PCT,
    updatedPeak: effectivePeak,
    updatedDayStart: effectiveDayStart,
    updatedDayDate: effectiveDayDate,
  };
}

function calculateKellySize(
  winRate: number,
  avgWin: number,
  avgLoss: number,
  portfolioValue: number,
  confidence: number,
  timeframe: string
): number {
  if (avgLoss <= 0) avgLoss = 0.1;
  if (avgWin <= 0) avgWin = 0.1;

  const R = avgWin / avgLoss;
  const kellyPct = (winRate * R - (1 - winRate)) / R;

  if (kellyPct <= 0) return 0;

  const fractionalKelly = kellyPct * KELLY_FRACTION;

  const confidenceMultiplier = 0.8 + confidence * 0.6;

  const timeframeMultiplier: Record<string, number> = {
    "5m": 0.7,
    "1h": 0.8,
    "2h": 1.2,
    "6h": 1.5,
    "1d": 1.7,
  };
  const tfMult = timeframeMultiplier[timeframe] || 1.0;

  let positionPct = fractionalKelly * confidenceMultiplier * tfMult;
  positionPct = Math.min(positionPct, MAX_POSITION_PCT);

  if (positionPct < 0.03) return 0;

  return portfolioValue * positionPct;
}

export async function initializePaperPortfolios(): Promise<void> {
  await removeHodlerHankIfExists();
  const agents = await db.select().from(agentsTable);
  for (const agent of agents) {
    const existing = await db
      .select()
      .from(paperPortfoliosTable)
      .where(eq(paperPortfoliosTable.agentId, agent.id));

    if (existing.length === 0) {
      await db.insert(paperPortfoliosTable).values({
        agentId: agent.id,
        agentName: agent.name,
        cashBalance: INITIAL_BALANCE,
        totalValue: INITIAL_BALANCE,
        totalTrades: 0,
        winningTrades: 0,
        losingTrades: 0,
      });
      logger.info({ agentName: agent.name, balance: INITIAL_BALANCE }, "Initialized paper portfolio");
    }
  }
}

export interface QuantTradePayload {
  probUp: number;
  probDown: number;
  probStable: number;
  expectedReturnPct: number;
  modelVersion?: string;
  source?: string;
  // Task #468 — when the per-coin slot was missing/quarantined and
  // the prediction was served by the pooled prior, this carries
  // "pooled" so the trade gate can apply the agent's
  // pooled_fallback_penalty multiplier.
  fallback?: "pooled" | null;
  // Phase 4 — meta-model decision surface. When `metaAbstainReason` is
  // set the trader records a typed skip and exits before the legacy
  // gates run. `metaSizeMultiplier` is clamped to [0.5, 1.5] and scales
  // the computed position size before any portfolio caps. Both come
  // straight from /ml/meta/predict via quant-brain.
  metaAction?: "long" | "short" | "no_trade" | null;
  metaSizeMultiplier?: number | null;
  metaExpectedEdgePct?: number | null;
  metaAbstainReason?: string | null;
  metaKind?: string | null;
  metaVersion?: string | null;
}

const META_ABSTAIN_SKIP_REASONS: Record<string, "meta_no_trade_low_edge" | "meta_no_trade_bad_regime_fit" | "meta_no_trade_specialist_disagreement" | "meta_no_trade_low_calibration"> = {
  low_edge: "meta_no_trade_low_edge",
  bad_regime_fit: "meta_no_trade_bad_regime_fit",
  specialist_disagreement: "meta_no_trade_specialist_disagreement",
  low_calibration: "meta_no_trade_low_calibration",
};

/** Clamp the composite (quant × brain) size multiplier into the
 * permitted band. Single source of truth — there is no per-factor
 * clamp anywhere else.
 *
 * Task #381 step 6: when the supervisory meta-brain is in
 * `defensive_mode === "hard"`, the floor relaxes to 0.0 so the brain
 * can route to suppression via `mult = 0` (see
 * `getFamilySizeMultiplier`). In every other case the floor stays at
 * 0.5 so the brain cannot wipe out sizing from a momentary blip.
 */
function clampMetaSizeMultiplier(x: number | null | undefined): number {
  if (x === null || x === undefined || !Number.isFinite(x)) return 1.0;
  let floor = 0.5;
  try {
    const d = getActiveDirective();
    if (
      !metaBrainIsNeutralDirective(d) &&
      d.defensive_mode === "hard"
    ) {
      floor = 0.0;
    }
  } catch {
    /* keep default floor */
  }
  return Math.max(floor, Math.min(1.5, x));
}

export async function executePaperTrade(
  agentId: number,
  agentName: string,
  coinId: string,
  coinName: string,
  direction: "up" | "down" | "stable",
  confidence: number,
  entryPrice: number,
  timeframe: string,
  predictionId: number,
  atrValue: number,
  // Phase 5 — when present (quant brain ON), the trader runs an EV gate:
  // the absolute expected return in the trade direction must exceed the
  // round-trip cost (taker fee + slippage, both ways) by a margin, else
  // we skip. LLM trades pass undefined and the gate is bypassed. This is
  // the last line of defence between the model and real-money sizing.
  quant?: QuantTradePayload,
): Promise<void> {
  // ── Task #468 — strict registry gate ─────────────────────────────
  // Resolve the agent's strategy profile from the startup-loaded
  // cache (NEVER a per-decision DB lookup). The cache throws a
  // typed `AgentNotExecutableError` for any of:
  //   - unknown agent id (row not in cache → never default)
  //   - DB status not "active" (e.g. quarantine_review / disabled
  //     after a retirement flip)
  //   - profile.executes=false (baseline / legacy_archived)
  // Each typed reason lands as a structured skip; the trade NEVER
  // falls through to the generic gates. This is the spec-required
  // "no silent default agent" contract (`task-468.md` line 68).
  let profile: AgentProfile;
  try {
    profile = getCachedProfileForAgentId(agentId).profile;
  } catch (err) {
    if (err instanceof AgentNotExecutableError) {
      recordSkip(
        "agent_not_executable",
        agentName,
        `Skipping trade — agent ${agentId} not executable: ${err.reason} (${err.message})`,
        {
          agentName,
          coinName,
          profileId: err.profileId ?? "unknown",
          profileStatus: err.dbStatus ?? "unknown",
        },
        { agentId, coinId },
      );
      return;
    }
    throw err;
  }
  // Hard-veto blocked regimes BEFORE any meta-model call (spec line
  // 44). The meta-model is never consulted for an agent that cannot
  // trade in the current regime — the test suite enforces this.
  const liveRegime = getCurrentRegime();
  const regimeLabel = liveRegime?.regimeLabel ?? null;
  if (!profileAllowsRegime(profile, regimeLabel)) {
    recordSkip(
      "agent_blocked_regime",
      agentName,
      `Skipping trade — current regime ${regimeLabel ?? "unknown"} is blocked / non-preferred for ${profile.agent_id}`,
      {
        agentName,
        coinName,
        profileId: profile.agent_id,
        regime: regimeLabel ?? "unknown",
        blockedRegimes: profile.blocked_regimes.join(",") || "(none)",
      },
      { agentId, coinId },
    );
    return;
  }
  if (
    profile.min_confidence !== null &&
    direction !== "stable"
  ) {
    // Task #468 — abstain_bias > 1 raises the effective confidence
    // floor (more cautious agents demand higher conviction); < 1
    // lowers it. The lift is intentionally small (5 bps per unit
    // above 1.0) so a 1.30 abstain_bias only nudges the floor by
    // ~1.5%, never inverting the gate. min_confidence is clamped
    // to [0, 1].
    const ab = profile.abstain_bias ?? 1.0;
    const effectiveMinConfidence = Math.max(
      0,
      Math.min(1, profile.min_confidence + (ab - 1.0) * 0.05),
    );
    if (confidence < effectiveMinConfidence) {
      recordSkip(
        "confidence_below_threshold",
        agentName,
        `Skipping trade — confidence below profile floor (${profile.agent_id})`,
        {
          agentName,
          coinName,
          timeframe,
          direction,
          confidence: (confidence * 100).toFixed(0) + "%",
          profileFloor: (effectiveMinConfidence * 100).toFixed(1) + "%",
          baseFloor: (profile.min_confidence * 100).toFixed(0) + "%",
          abstainBias: ab.toFixed(2),
          profileId: profile.agent_id,
        },
        { agentId, coinId },
      );
      return;
    }
  }
  if (
    profile.min_expected_edge_after_costs !== null &&
    quant !== undefined
  ) {
    const directionalReturnFrac =
      direction === "up"
        ? quant.expectedReturnPct / 100
        : -quant.expectedReturnPct / 100;
    const requiredEdge = profile.min_expected_edge_after_costs;
    if (directionalReturnFrac < requiredEdge) {
      recordSkip(
        "quant_ev_below_costs",
        agentName,
        `Skipping trade — directional edge below profile floor (${profile.agent_id})`,
        {
          agentName,
          coinName,
          direction,
          directionalReturnFrac: directionalReturnFrac.toFixed(5),
          profileMinEdge: requiredEdge.toFixed(5),
          profileId: profile.agent_id,
        },
        { agentId, coinId },
      );
      return;
    }
  }
  // Phase 4 — meta-model abstain is a first-class terminal outcome.
  // Record the typed reason BEFORE the legacy `direction === "stable"`
  // early-out so the diagnostics page can break down WHY the meta sat
  // out (low_edge / bad_regime_fit / specialist_disagreement /
  // low_calibration) instead of seeing an unaccounted skip.
  if (quant?.metaAction === "no_trade" || quant?.metaAbstainReason) {
    const mapped = quant.metaAbstainReason
      ? META_ABSTAIN_SKIP_REASONS[quant.metaAbstainReason]
      : null;
    const skipReason = mapped ?? "quant_meta_abstain";
    const reason = quant.metaAbstainReason ?? "no_trade";
    await recordSkip(
      skipReason,
      agentName,
      `Meta-model abstained — ${reason}`,
      {
        agentName,
        coinName,
        timeframe,
        metaAction: quant.metaAction ?? "no_trade",
        metaKind: quant.metaKind ?? "unknown",
        metaVersion: quant.metaVersion ?? "unknown",
        probUp: (quant.probUp * 100).toFixed(0) + "%",
        probDown: (quant.probDown * 100).toFixed(0) + "%",
        expectedReturnPct: quant.expectedReturnPct.toFixed(3) + "%",
        metaExpectedEdgePct:
          quant.metaExpectedEdgePct === null || quant.metaExpectedEdgePct === undefined
            ? "n/a"
            : quant.metaExpectedEdgePct.toFixed(3) + "%",
      },
      { agentId, coinId },
    );
    return;
  }
  if (direction === "stable") return;
  // Asymmetric confidence gate: BUY trades historically win only ~24% in this
  // fleet vs ~56% for SELL, so longs require an extra 5pt of conviction.
  // Quant-driven trades bypass the asymmetry and use the explicit gate floor
  // because the LLM winrate skew that motivated the bump doesn't describe the
  // calibrated quant probabilities — see getQuantMinConfidenceToTrade() for
  // the full rationale. LLM trades (no `quant` payload) keep the old gate.
  let minConfidence: number;
  if (quant) {
    minConfidence = getQuantMinConfidenceToTrade();
  } else {
    const baseMinConfidence = getMinConfidenceToTrade();
    minConfidence = direction === "up"
      ? Math.max(baseMinConfidence, 0.55)
      : baseMinConfidence;
  }
  if (confidence < minConfidence) {
    recordSkip(
      "confidence_below_threshold",
      agentName,
      "Skipping trade — confidence below MIN_CONFIDENCE_TO_TRADE",
      {
        agentName,
        coinName,
        timeframe,
        direction,
        confidence: (confidence * 100).toFixed(0) + "%",
        required: (minConfidence * 100).toFixed(0) + "%",
      },
      { agentId, coinId },
    );
    return;
  }
  // Task #550 — second of the two required gates. The static
  // `TRADEABLE_TIMEFRAMES_SET` stays as the "supported timeframe
  // universe" check; layered on top is the per-timeframe role from
  // `shared/timeframe-roles.json`. Both must pass: a TF whose role is
  // not `trade` (shadow / context / disabled) is refused here even if
  // it is in the static tradeable set, and an unsupported TF is
  // refused even if its role were somehow `trade`. The first gate
  // (per-slice promotion via `isModelAvailable` in quant-brain.ts) is
  // unchanged — both gates are required, neither is sufficient alone.
  if (!TRADEABLE_TIMEFRAMES.has(timeframe)) return;
  const tfRole = getRoleForTimeframe(timeframe);
  if (tfRole !== "trade") {
    logger.warn(
      {
        agentName,
        coinName,
        timeframe,
        role: tfRole,
        reason: "trade_blocked_by_role",
      },
      "paper-trader: trade refused — timeframe role is not 'trade'",
    );
    return;
  }

  // ── Task #614 — MTTM whitelist guard ────────────────────────────
  // When Minimum Truthful Trading Mode is enabled, the decision
  // pipeline is restricted to the 16 promoted-lightgbm slots in
  // `mttm_universe`. Anything outside the whitelist is recorded as a
  // typed skip so the dashboard can show how many trades MTTM is
  // gating away.
  //
  // Fail-CLOSED on a cache miss: we cannot let an out-of-universe
  // trade slip through just because the cache hasn't been warmed yet.
  // `await getMttmConfig()` returns instantly when the cache is hot
  // (the common path after `monitor.ts.runAnalysisCycle` warms it at
  // the start of every cycle) and falls through to a single
  // `app_settings` SELECT on a cold start. We capture the same
  // snapshot here and reuse it for the position-size cap below so
  // both the guard and the cap see the same authoritative config
  // and there is no fail-open window.
  const _mttmActiveCfg = await getMttmConfig();
  if (_mttmActiveCfg.enabled) {
    const sk = mttmSlotKey(coinId, timeframe);
    if (!_mttmActiveCfg.universeKeys.has(sk)) {
      // Task #659 — DS lane uses a stricter skip reason than the
      // generic 16-slot universe gate. When the diagnostic sandbox is
      // the active mode, the universe is locked to bitcoin/5m, so any
      // off-scope signal is a *diagnostic_universe_locked* skip, not a
      // generic "outside-universe" skip. The distinction matters for
      // operator log audits (DS lane lockdowns are its own bucket).
      const isDsLane = _mttmActiveCfg.mode === "diagnostic_sandbox";
      const reason = isDsLane ? "diagnostic_universe_locked" : "mttm_outside_universe";
      const msg = isDsLane
        ? `Skipping trade — diagnostic sandbox locked to bitcoin/5m, ${sk} is off-scope`
        : `Skipping trade — MTTM on, ${sk} not in 16-slot universe`;
      recordSkip(
        reason,
        agentName,
        msg,
        { agentName, coinName, timeframe },
        { agentId, coinId },
      );
      return;
    }
  }

  // Skip when this (brain, timeframe) horizon is disabled by the
  // weak-signal gate (same threshold as the dashboard WEAK badge).
  const brainTag: "QUANT" | "LLM" = quant ? "QUANT" : "LLM";
  try {
    if (await isHorizonDisabled(brainTag, timeframe)) {
      recordSkip(
        "horizon_disabled_weak_signal",
        agentName,
        "Skipping trade — horizon disabled by weak-signal gate",
        {
          agentName,
          coinName,
          brain: brainTag,
          timeframe,
          direction,
        },
        { agentId, coinId },
      );
      return;
    }
  } catch (err) {
    logger.warn({ err }, "horizon-disable gate check failed — allowing trade");
  }

  // Phase 5 quant EV gate. When the quant brain is driving the trade we
  // require its expected return in the trade direction to clear the round-
  // trip cost (taker fee + slippage, both ways) by `getMinEvVsCost()`x.
  // ROUND_TRIP_COST_PCT is a fraction (e.g. 0.003 == 0.30%); the model's
  // expectedReturnPct is a percent (e.g. 0.50 == 0.50%), so we compare on
  // the same scale by multiplying ROUND_TRIP_COST_PCT by 100. LLM trades
  // (no quant payload) bypass this gate; legacy gates still protect them.
  if (quant) {
    const directionalReturnPct = direction === "up" ? quant.expectedReturnPct : -quant.expectedReturnPct;
    const requiredPct = getQuantEvGateRequiredPct();
    if (directionalReturnPct < requiredPct) {
      recordSkip(
        "quant_ev_below_costs",
        agentName,
        "Skipping trade — quant expected return below round-trip cost",
        {
          agentName,
          coinName,
          direction,
          expectedReturnPct: quant.expectedReturnPct.toFixed(3) + "%",
          directionalReturnPct: directionalReturnPct.toFixed(3) + "%",
          requiredPct: requiredPct.toFixed(3) + "%",
          probUp: (quant.probUp * 100).toFixed(0) + "%",
          probDown: (quant.probDown * 100).toFixed(0) + "%",
          modelVersion: quant.modelVersion ?? "unknown",
        },
        { agentId, coinId },
      );
      return;
    }
  }

  // Fleet correlation brake: if too many open positions across the whole fleet
  // are already on the same side, block new entries on that side. Prevents
  // cluster wipeouts where one market move stops out the entire fleet at once.
  // Only kicks in once the fleet has at least 6 open positions (small samples
  // are too noisy to act on).
  try {
    const openPositions = await db
      .select({ direction: paperPositionsTable.direction })
      .from(paperPositionsTable);
    if (openPositions.length >= SHARED_FLEET_BRAKE_MIN_OPEN) {
      const sameSide = openPositions.filter((p) => p.direction === direction).length;
      const sameSideShare = sameSide / openPositions.length;
      if (sameSideShare >= SHARED_FLEET_BRAKE_DOMINANCE) {
        recordSkip(
          "fleet_direction_imbalance",
          agentName,
          "Skipping trade — fleet correlation brake (too many positions on same side)",
          {
            agentName,
            coinName,
            direction,
            sameSide,
            totalOpen: openPositions.length,
            sameSideShare: (sameSideShare * 100).toFixed(0) + "%",
            threshold: (SHARED_FLEET_BRAKE_DOMINANCE * 100).toFixed(0) + "%",
          },
          { agentId, coinId },
        );
        return;
      }
    }
  } catch (err) {
    logger.warn({ err }, "fleet correlation brake check failed — allowing trade");
  }

  const regime = getCurrentRegime();
  if (regime) {
    const counterTrendShort = direction === "down" && regime.trendBias === "bullish";
    const counterTrendLong = direction === "up" && regime.trendBias === "bearish";
    const counterTrendMin = getCounterTrendMinConfidence();
    if ((counterTrendShort || counterTrendLong) && confidence < counterTrendMin) {
      recordSkip("counter_trend_regime", agentName,
        "Skipping trade — counter-trend in confirmed regime, confidence below override threshold",
        {
          agentName,
          coinName,
          direction,
          trendBias: regime.trendBias,
          avgChange24h: regime.avgChange24h.toFixed(2) + "%",
          confidence: (confidence * 100).toFixed(0) + "%",
          required: (counterTrendMin * 100).toFixed(0) + "%",
        },
        { agentId, coinId });
      return;
    }
  }

  await db.transaction(async (tx) => {
    const portfolio = await tx
      .select()
      .from(paperPortfoliosTable)
      .where(eq(paperPortfoliosTable.agentId, agentId))
      .for("update");

    if (portfolio.length === 0) return;
    const p = portfolio[0];

    const riskCheck = checkRiskLimits(p.totalValue, p.peakValue, p.dayStartValue, p.dayStartDate);

    await tx.update(paperPortfoliosTable).set({
      peakValue: riskCheck.updatedPeak,
      dayStartValue: riskCheck.updatedDayStart,
      dayStartDate: riskCheck.updatedDayDate,
    }).where(eq(paperPortfoliosTable.agentId, agentId));

    if (riskCheck.dailyLossHit) {
      recordSkip("daily_loss_limit", agentName, "Skipping trade — daily loss limit hit",
        { agentName, coinName, dailyLossLimit: `${DAILY_LOSS_LIMIT_PCT * 100}%` },
        { agentId, coinId });
      return;
    }
    // Task #468 — drawdown_sensitivity scales the effective halt
    // threshold per agent. sensitivity > 1 = MORE sensitive = halts at
    // a lower drawdown (e.g. volatility_defensive uses 1.5 → halt at
    // 2/3 of the fleet-default threshold). null = inherit fleet
    // default verbatim.
    {
      const ds = profile.drawdown_sensitivity;
      const effectiveDrawdownThreshold =
        ds !== null && ds > 0 ? DRAWDOWN_HALT_PCT / ds : DRAWDOWN_HALT_PCT;
      const peak = p.peakValue ?? 0;
      const drawdownNow = peak > 0 ? (peak - p.totalValue) / peak : 0;
      const profileDrawdownHit = drawdownNow >= effectiveDrawdownThreshold;
      if (riskCheck.drawdownHit || profileDrawdownHit) {
        recordSkip(
          "drawdown_halt",
          agentName,
          "Skipping trade — max drawdown halt triggered",
          {
            agentName,
            coinName,
            drawdownHalt: `${(effectiveDrawdownThreshold * 100).toFixed(2)}%`,
            drawdownNow: `${(drawdownNow * 100).toFixed(2)}%`,
            profileId: profile.agent_id,
            drawdownSensitivity: ds ?? "fleet_default",
          },
          { agentId, coinId },
        );
        return;
      }
    }

    const existingPositions = await tx
      .select()
      .from(paperPositionsTable)
      .where(eq(paperPositionsTable.agentId, agentId));

    if (existingPositions.length >= MAX_OPEN_POSITIONS_PER_AGENT) return;

    const alreadyHasCoin = existingPositions.some((pos) => pos.coinId === coinId);
    if (alreadyHasCoin) return;

    const recentCoinTrades = await tx.select().from(paperTradesTable)
      .where(and(eq(paperTradesTable.agentId, agentId), eq(paperTradesTable.coinId, coinId), eq(paperTradesTable.status, "closed")))
      .orderBy(desc(paperTradesTable.closedAt))
      .limit(3);
    if (recentCoinTrades.length >= 3) {
      const allLosses = recentCoinTrades.every(t => (t.pnl ?? 0) <= 0);
      if (allLosses) {
        recordSkip("consecutive_losses", agentName,
          "Skipping trade — 3 consecutive losses on this coin", { agentName, coinName },
          { agentId, coinId });
        return;
      }
    }

    let positionSize: number;

    // Task #659 — DS lane uses strict fixed sizing. Kelly, confidence
    // tiers, meta multiplier, profile bias, pooled penalty, family
    // multiplier are all bypassed.
    const _isDsLane = _mttmActiveCfg.enabled
      && _mttmActiveCfg.mode === "diagnostic_sandbox";
    if (_isDsLane) {
      positionSize = p.totalValue * _mttmActiveCfg.diagnosticSandbox.fixedPositionPct;
    } else if (p.totalTrades >= KELLY_MIN_TRADES && p.winningTrades > 0) {
      const winRate = p.winningTrades / p.totalTrades;
      const trades = await tx.select().from(paperTradesTable)
        .where(and(eq(paperTradesTable.agentId, agentId), eq(paperTradesTable.status, "closed")))
        .orderBy(desc(paperTradesTable.closedAt))
        .limit(50);

      let totalWinPnl = 0, totalLossPnl = 0, winCount = 0, lossCount = 0;
      for (const t of trades) {
        if (t.pnl !== null && t.positionSize > 0) {
          const pctPnl = Math.abs(t.pnl / t.positionSize) * 100;
          if (t.pnl > 0) { totalWinPnl += pctPnl; winCount++; }
          else { totalLossPnl += pctPnl; lossCount++; }
        }
      }

      const avgWin = winCount > 0 ? totalWinPnl / winCount : 0.1;
      const avgLoss = lossCount > 0 ? totalLossPnl / lossCount : 0.1;

      let kellySize = calculateKellySize(winRate, avgWin, avgLoss, p.totalValue, confidence, timeframe);
      if (p.totalTrades < KELLY_RAMP_END) {
        const rampProgress = (p.totalTrades - KELLY_MIN_TRADES) / (KELLY_RAMP_END - KELLY_MIN_TRADES);
        const fixedSize = p.totalValue * tieredPositionPct(confidence);
        positionSize = fixedSize * (1 - rampProgress) + kellySize * rampProgress;
      } else {
        positionSize = kellySize;
      }
    } else {
      positionSize = p.totalValue * tieredPositionPct(confidence);
    }

    // Phase 4 — apply meta-model size multiplier. Task #373 composes the
    // supervisory meta-brain's family multiplier with the quant meta's
    // multiplier, then applies a SINGLE final clamp to [0.5, 1.5] so the
    // combined effect can never widen sizing beyond the existing risk
    // band. (Clamping each factor independently would allow a combined
    // 2.25x / 0.25x excursion — the architect caught this.) The brain
    // contribution is 1.0 (no-op) when disabled / neutral / unreachable.
    //
    // Task #468 — strategy family comes from the typed registry via
    // the agent's profile_id (already loaded into `profile` above).
    // For rows whose profile lookup failed (legacy / unswept) we fall
    // back to the legacy name-based resolver to keep the meta-brain
    // sizing path live during the rollout.
    let compositeSizeMult = 1.0;
    if (quant && quant.metaSizeMultiplier !== null && quant.metaSizeMultiplier !== undefined) {
      compositeSizeMult *= quant.metaSizeMultiplier;
    }
    if (profile && profile.size_bias !== null) {
      // Profile size_bias is a multiplicative tilt around 1.0 (defensive
      // agents pull below 1, aggressive agents pull above). Composed
      // with the meta-brain multiplier and clamped together below.
      compositeSizeMult *= profile.size_bias;
    }
    // Task #468 — pooled_fallback_penalty: when the prediction was
    // served by the pooled prior (per-coin slot missing/quarantined),
    // shrink the position by the agent's penalty factor. v1 spec
    // values range 0.50 (volatility_defensive) .. 0.85 (momentum_core).
    if (
      profile &&
      profile.pooled_fallback_penalty !== null &&
      quant &&
      (quant.fallback === "pooled" || quant.source === "prior")
    ) {
      compositeSizeMult *= profile.pooled_fallback_penalty;
    }
    try {
      const family = profile
        ? profile.strategy_family
        : metaBrainResolveFamily(agentName);
      // Task #381 step 4 — wire isFamilySuppressed at the trade gate.
      // The supervisory brain may emit suppress_signal, family-level
      // suppression, or a paused_slices entry covering this
      // (coin, timeframe). Route through the existing skip path —
      // no new execution branch. Shadow-mode directives are neutral
      // and never trigger this branch.
      if (metaBrainIsFamilySuppressed(family, coinId, timeframe)) {
        await recordSkip(
          "meta_brain_suppress",
          agentName,
          "Skipping trade — meta-brain supervisory suppression",
          { agentName, coinName, timeframe },
          { agentId, coinId },
        );
        return;
      }
      // Task #468 — benchmark_sensitivity scales the agent's
      // exposure to the supervisory brain's family-multiplier
      // signal (which is itself a function of the Strategy-Lab
      // benchmark cohort, see meta-brain/benchmark-telemetry.ts).
      // sensitivity 1.0 = full effect, 0.3 (volatility_defensive)
      // = 30% of effect. Only the *shrink* portion (mult < 1.0)
      // is sensitivity-scaled; the brain may still expand size.
      const familyMult = getFamilySizeMultiplier(family);
      const bs = profile?.benchmark_sensitivity ?? null;
      let scaledFamilyMult = familyMult;
      if (bs !== null && familyMult < 1.0) {
        scaledFamilyMult = 1.0 - (1.0 - familyMult) * bs;
      }
      compositeSizeMult *= scaledFamilyMult;
    } catch (err) {
      // Advisory; never affect trade.
      logger.debug(
        { err: String(err), agentId, coinId },
        "meta-brain: size shaping skipped (non-blocking)",
      );
    }
    // Hoist the clamped meta-multiplier into a stable const so the engine-
    // override path below can re-apply the same supervisory shaping after
    // `/ml/decide` returns its base sizing. The DS shadow lane skips the
    // multiplier application below (DS uses an exact-pin sizer instead).
    const executionSizeMultiplier = clampMetaSizeMultiplier(compositeSizeMult);
    if (!_isDsLane && executionSizeMultiplier !== 1.0) {
      positionSize = positionSize * executionSizeMultiplier;
    }

    // Task #614 — MTTM-scoped per-position cap (5%) instead of the
    // global 30%. Hoisted so the engine-override path below can reuse it
    // after `/ml/decide` returns its base sizing.
    const _maxPositionPct =
      _mttmActiveCfg.enabled ? _mttmActiveCfg.maxPositionPct : MAX_POSITION_PCT;
    if (_isDsLane) {
      // Task #659 — strict exact 0.5% sizing in DS mode. We bypass the
      // generic cash/portfolio-at-risk caps entirely and either place the
      // exact pin or skip with reason `dsl_insufficient_cash` — never
      // silently shrink. The MTTM-cap path below is unreachable here.
      const dsPin = p.totalValue * _mttmActiveCfg.diagnosticSandbox.fixedPositionPct;
      if (coinId !== _mttmActiveCfg.diagnosticSandbox.coinId
          || timeframe !== _mttmActiveCfg.diagnosticSandbox.timeframe) {
        throw new Error(
          `DS invariant: cap-stage reached with (${coinId},${timeframe}); pin is (`
          + `${_mttmActiveCfg.diagnosticSandbox.coinId},`
          + `${_mttmActiveCfg.diagnosticSandbox.timeframe}).`,
        );
      }
      if (p.cashBalance < dsPin) {
        await recordSkip(
          "ds_insufficient_cash",
          agentName,
          `Skipping DS trade — cash ${p.cashBalance.toFixed(2)} < pin ${dsPin.toFixed(2)}`,
          { agentName, coinName, timeframe },
          { agentId, coinId },
        );
        return;
      }
      positionSize = dsPin;
    } else {
      // Cash + portfolio-at-risk caps unchanged.
      positionSize = Math.min(
        positionSize,
        p.cashBalance * MAX_CASH_PER_POSITION_PCT,
        p.totalValue * _maxPositionPct,
      );
    }

    const totalInvested = existingPositions.reduce((s, pos) => s + pos.positionSize, 0);
    if (!_isDsLane && (totalInvested + positionSize) / p.totalValue >= MAX_PORTFOLIO_AT_RISK) {
      positionSize = Math.max(0, p.totalValue * MAX_PORTFOLIO_AT_RISK - totalInvested);
    }

    let entryFee = computeEntryFee(positionSize);
    if (!_isDsLane) {
      // Default lane: fee comes out of the executed notional.
      positionSize = positionSize - entryFee;
      const postFeePortfolioCheck = (totalInvested + positionSize) / p.totalValue;
      if (postFeePortfolioCheck >= MAX_PORTFOLIO_AT_RISK) {
        positionSize = Math.max(0, p.totalValue * MAX_PORTFOLIO_AT_RISK - totalInvested);
        entryFee = computeEntryFee(positionSize);
        positionSize = positionSize - entryFee;
      }
    } else {
      // Task #659 — DS executed notional is exactly the 0.5% pin. The
      // entry fee is paid OUT of cash separately (so reported notional
      // == equity * 0.005); if cash cannot cover pin + fee, skip.
      if (p.cashBalance < positionSize + entryFee) {
        await recordSkip(
          "ds_insufficient_cash",
          agentName,
          `Skipping DS trade — cash ${p.cashBalance.toFixed(2)} < pin+fee ${(positionSize + entryFee).toFixed(2)}`,
          { agentName, coinName, timeframe },
          { agentId, coinId },
        );
        return;
      }
    }

    const freshPortfolio = await tx
      .select()
      .from(paperPortfoliosTable)
      .where(eq(paperPortfoliosTable.agentId, agentId));
    if (freshPortfolio.length > 0) {
      const fp = freshPortfolio[0];
      const freshRisk = checkRiskLimits(fp.totalValue, fp.peakValue, fp.dayStartValue, fp.dayStartDate);
      if (freshRisk.dailyLossHit || freshRisk.drawdownHit) {
        recordSkip("risk_recheck_halt", agentName,
          "Skipping trade — risk limit hit on re-check before insert",
          { agentName, coinName, dailyLossHit: freshRisk.dailyLossHit, drawdownHit: freshRisk.drawdownHit },
          { agentId, coinId });
        return;
      }
    }

    // Phase 5 — fleet-level portfolio constraints (sector cap, correlated
    // exposure, beta-to-BTC, regime budget). Same gate the unified Python
    // decision engine applies in the offline backtester.
    {
      const portfolioCheck = checkPortfolioConstraints({
        coinId,
        newNotionalUsd: positionSize,
        equityUsd: p.totalValue,
        regime: regime?.regimeLabel ?? null,
        openPositions: existingPositions.map((pos) => {
          const entryRegime = (pos as { entryRegimeLabel?: string | null })
            .entryRegimeLabel ?? null;
          return {
            coinId: pos.coinId,
            direction: (pos.direction as "up" | "down") ?? "up",
            notionalUsd: pos.positionSize,
            regimeAtEntry: entryRegime,
            betaToBtc: null,
          };
        }),
      });
      if (!portfolioCheck.ok) {
        recordSkip(
          portfolioCheck.skipReason ?? "portfolio_constraint",
          agentName,
          `Skipping trade — ${portfolioCheck.detail ?? "portfolio constraint hit"}`,
          { agentName, coinName, ...portfolioCheck.breakdown },
          { agentId, coinId },
        );
        return;
      }
    }

    // Phase 5 — route the decision through the unified Python decision
    // engine (`/ml/decide`). The engine is the single source of truth for
    // direction emit, confidence floor, counter-trend, consec-loss, EV
    // gate, sizing, SL/TP geometry, and portfolio constraints — the same
    // pure function the offline backtester runs through `decide()`. We
    // also OVERRIDE the locally-computed positionSize/SL/TP further down
    // with the engine's values so live execution matches backtest output
    // bit-for-bit. If `/ml/decide` is unreachable on a quant trade we
    // skip the trade rather than fall back to the legacy stack — the
    // engine being the gate is the whole point.
    let engineDecision: Awaited<ReturnType<typeof getMlDecision>> | null = null;
    if (quant && atrValue !== null && atrValue !== undefined && atrValue > 0) {
      try {
        engineDecision = await getMlDecision({
          coinId,
          timeframe: timeframe as TimeframeKey,
          lastPrice: entryPrice,
          atrValue: atrValue,
          probUp: quant.probUp,
          probDown: quant.probDown,
          probStable: quant.probStable,
          expectedReturnPct: quant.expectedReturnPct,
          regime: regime?.regimeLabel ?? null,
          trendBias: (regime?.trendBias as "bullish" | "bearish" | null | undefined) ?? null,
          portfolio: {
            equityUsd: p.totalValue,
            cashUsd: p.cashBalance,
            openPositions: existingPositions.map((pos) => ({
              coinId: pos.coinId,
              direction: pos.direction ?? "up",
              notionalUsd: pos.positionSize,
              regimeAtEntry: (pos as { entryRegimeLabel?: string | null })
                .entryRegimeLabel ?? null,
              betaToBtc: null,
            })),
          },
        });
      } catch (err) {
        recordSkip(
          "quant_meta_abstain",
          agentName,
          "Skipping trade — unified decision engine /ml/decide unreachable",
          { agentName, coinName, err: String(err) },
          { agentId, coinId },
        );
        return;
      }
      if (engineDecision.action === "no_trade") {
        recordSkip(
          (engineDecision.skipReason as SkipReason | null) ?? "portfolio_constraint",
          agentName,
          `Skipping trade — unified decision engine: ${engineDecision.skipReason ?? "unknown"} ${engineDecision.skipDetail ?? ""}`,
          {
            agentName,
            coinName,
            direction,
            confidence,
            source: "ml-decide",
            gates: engineDecision.gatesApplied,
            portfolio: engineDecision.portfolioCheck,
          },
          { agentId, coinId },
        );
        return;
      }
      // Engine approved — adopt its decision as authoritative: trade SIDE,
      // sizing, and SL/TP all come from the unified engine. The previously
      // locally-computed direction/positionSize/entryFee are replaced.
      if (
        (engineDecision.direction === "up" || engineDecision.direction === "down") &&
        engineDecision.direction !== direction
      ) {
        direction = engineDecision.direction as "up" | "down";
      }
      positionSize = engineDecision.positionSizeUsd;
      // `/ml/decide` is the authority for side + base size + SL/TP, but
      // the api-server owns live supervisory overlays that Python does not
      // know about: meta-brain family sizing, registry size_bias,
      // pooled-fallback penalty, and MTTM's tighter max-position cap. Apply
      // those AFTER the engine override so quant trades cannot erase them.
      // DS lane is excluded — its 0.5% pin already ran above.
      if (!_isDsLane) {
        if (executionSizeMultiplier !== 1.0) {
          positionSize *= executionSizeMultiplier;
        }
        positionSize = Math.min(
          positionSize,
          p.cashBalance * MAX_CASH_PER_POSITION_PCT,
          p.totalValue * _maxPositionPct,
        );
        if ((totalInvested + positionSize) / p.totalValue >= MAX_PORTFOLIO_AT_RISK) {
          positionSize = Math.max(0, p.totalValue * MAX_PORTFOLIO_AT_RISK - totalInvested);
        }
      }
      entryFee = positionSize * TAKER_FEE_PCT / Math.max(1e-12, 1 - TAKER_FEE_PCT);
    }

    if (positionSize < 1.0 || entryPrice <= 0) return;

    const totalCashNeeded = positionSize + entryFee;
    const liveCash = freshPortfolio.length > 0 ? freshPortfolio[0].cashBalance : p.cashBalance;
    if (liveCash < totalCashNeeded) {
      logger.warn({ agentName, coinName, liveCash, want: totalCashNeeded }, "AI trade aborted — insufficient cash");
      return;
    }

    const livePositionsCheck = await tx.select().from(paperPositionsTable).where(eq(paperPositionsTable.agentId, agentId));
    if (livePositionsCheck.length >= MAX_OPEN_POSITIONS_PER_AGENT) return;
    if (livePositionsCheck.some(pos => pos.coinId === coinId)) return;

    // direction has been narrowed to "up" | "down" by the early
    // `direction === "stable"` return above; the closure boundary just
    // re-widens the inferred type, so reassert it here.
    const adjustedEntryPrice = applyEntrySlippage(entryPrice, direction as "up" | "down");

    const quantity = positionSize / adjustedEntryPrice;

    const tf = TIMEFRAMES[timeframe as TimeframeKey];
    const expiresAt = new Date(Date.now() + (tf?.ms || 300000));

    // ATR floor + SL/TP geometry are sourced from
    // shared/trading-frictions.json so the Python backtester sees the same
    // numbers. Risk/reward geometry: TP distance is 1.5-1.95x the SL distance
    // so wins are bigger than losses; breakeven win-rate drops from ~56% to
    // ~34-40%, a defensible bar even if direction calls are only marginally
    // better than random.
    const floorPct = getAtrFloorPct(timeframe);
    const fallbackFloor = adjustedEntryPrice * floorPct;
    const effectiveAtr = Math.max(atrValue, fallbackFloor);
    const slMult = getSlMultiplier(timeframe);
    const tpMult = getTpMultiplier(timeframe);
    const slDistance = effectiveAtr * slMult;
    const tpDistance = effectiveAtr * tpMult;

    // Two fee-aware rejects.
    // (a) Hard TP-distance floor: if ATR-derived take-profit distance is below
    //     2× round-trip cost, EV is negative even on a winning hit. Skip.
    // (b) Expected-value floor: confidence × tpDistancePct must clear 3× the
    //     round-trip cost, else the long-run edge cannot beat fees + slippage.
    const tpDistancePct = tpDistance / adjustedEntryPrice;
    const evScore = confidence * tpDistancePct;
    const minTpDistance = getMinTpDistancePct();
    const evRequired = getMinEvVsCost() * ROUND_TRIP_COST_PCT;
    // Phase 5 — when the unified decision engine approved this trade it
    // has already applied the same fee/EV gates with identical inputs.
    // Skipping the duplicate gate here keeps the live path as a thin
    // execution layer over `decide()`.
    if (engineDecision && engineDecision.action !== "no_trade") {
      // already gated upstream
    } else if (tpDistancePct < minTpDistance) {
      recordSkip("fee_gate_tp_floor", agentName,
        "Skipping trade — TP distance below fee-aware hard floor (negative EV)",
        {
          agentName, coinName, timeframe,
          confidence: (confidence * 100).toFixed(0) + "%",
          tpDistancePct: (tpDistancePct * 100).toFixed(3) + "%",
          required: (minTpDistance * 100).toFixed(2) + "%",
          atrPct: ((atrValue / adjustedEntryPrice) * 100).toFixed(3) + "%",
        },
        { agentId, coinId });
      return;
    }
    if (engineDecision && engineDecision.action !== "no_trade") {
      // already gated upstream by /ml/decide
    } else if (evScore < evRequired) {
      recordSkip("fee_gate_ev", agentName,
        "Skipping trade — expected value below fee threshold (confidence × TP-distance < EV multiple × round-trip cost)",
        {
          agentName, coinName, timeframe,
          confidence: (confidence * 100).toFixed(0) + "%",
          tpDistancePct: (tpDistancePct * 100).toFixed(3) + "%",
          evScore: (evScore * 100).toFixed(3) + "%",
          evRequired: (evRequired * 100).toFixed(3) + "%",
        },
        { agentId, coinId });
      return;
    }

    let stopLossPrice: number;
    let takeProfitPrice: number;

    if (engineDecision && engineDecision.slPrice !== null && engineDecision.tpPrice !== null) {
      // Use the engine's SL/TP geometry verbatim so live execution matches
      // backtest output bit-for-bit.
      stopLossPrice = engineDecision.slPrice;
      takeProfitPrice = engineDecision.tpPrice;
    } else if (direction === "up") {
      stopLossPrice = adjustedEntryPrice - slDistance;
      takeProfitPrice = adjustedEntryPrice + tpDistance;
    } else {
      stopLossPrice = adjustedEntryPrice + slDistance;
      takeProfitPrice = adjustedEntryPrice - tpDistance;
    }

    const [trade] = await tx.insert(paperTradesTable).values({
      agentId,
      agentName,
      coinId,
      coinName,
      action: direction === "up" ? "buy" : "sell",
      entryPrice: adjustedEntryPrice,
      quantity,
      positionSize,
      entryFee,
      timeframe,
      predictionId,
      status: "open",
    }).returning({ id: paperTradesTable.id });

    await tx.insert(paperPositionsTable).values({
      agentId,
      agentName,
      coinId,
      coinName,
      direction,
      entryPrice: adjustedEntryPrice,
      quantity,
      positionSize,
      timeframe,
      tradeId: trade.id,
      expiresAt,
      stopLossPrice,
      takeProfitPrice,
      peakPrice: adjustedEntryPrice,
      // Phase 2 — capture the regime active at the moment the trade is
      // taken. Carried through to trade_journal.regime_label on close so
      // per-regime PnL reflects the regime AT decision time, not the
      // (possibly different) regime cached at close time.
      entryRegimeLabel: regime?.regimeLabel ?? null,
    });

    // Task #373 — bind this trade to the current tick's meta-brain
    // directive so record_outcome can close the learning loop on the
    // same directive that authorized the entry. In-memory map;
    // bounded and lost-on-restart by design (any trades opened
    // before a restart simply don't feed record_outcome).
    // Task #381 — bind for both real and shadow ticks. `bindTradeToTick`
    // skips `neutral:*` internally; `shadow:*` is preserved so
    // record_outcome still feeds the brain's learning loop.
    bindTradeToTick(trade.id, getActiveDirective().tick_id);

    const totalCostWithFee = positionSize + entryFee;
    await tx
      .update(paperPortfoliosTable)
      .set({ cashBalance: p.cashBalance - totalCostWithFee, updatedAt: new Date() })
      .where(eq(paperPortfoliosTable.agentId, agentId));

    logger.info({
      agentName, coinName, direction, timeframe,
      confidence: (confidence * 100).toFixed(0) + "%",
      positionSize: "$" + positionSize.toFixed(2),
      entryFee: "$" + entryFee.toFixed(4),
      slippage: (SLIPPAGE_PCT * 100).toFixed(2) + "%",
      entryPrice: adjustedEntryPrice,
      stopLoss: stopLossPrice.toFixed(8),
      takeProfit: takeProfitPrice.toFixed(8),
    }, "Paper trade opened");
  });
}

export async function closeExpiredPositions(): Promise<number> {
  if (isClosingPositions) return 0;
  isClosingPositions = true;

  try {
    const now = new Date();

    const allPositions = await db.select().from(paperPositionsTable);
    if (allPositions.length === 0) return 0;

    if (!isPriceDataFresh()) {
      logger.warn("Price data stale — deferring position close");
      return 0;
    }

    const prices = await fetchCoinPrices();
    if (prices.length === 0) return 0;
    let closed = 0;

    for (const pos of allPositions) {
      const currentCoin = prices.find((c) => c.id === pos.coinId);
      if (!currentCoin) continue;

      let exitPrice = currentCoin.currentPrice;
      const isExpired = pos.expiresAt <= now;

      const currentPeak = pos.peakPrice ?? pos.entryPrice;
      let newPeak = currentPeak;
      if (pos.direction === "up") {
        newPeak = Math.max(currentPeak, exitPrice);
      } else {
        newPeak = Math.min(currentPeak, exitPrice);
      }

      if (newPeak !== currentPeak) {
        await db.update(paperPositionsTable).set({ peakPrice: newPeak })
          .where(eq(paperPositionsTable.id, pos.id));
      }

      let hitStopLoss = false;
      let hitTakeProfit = false;

      if (pos.stopLossPrice && pos.takeProfitPrice) {
        const dir: "up" | "down" = pos.direction === "up" ? "up" : "down";
        hitStopLoss = isStopLossHit(dir, exitPrice, pos.stopLossPrice);
        hitTakeProfit = isTakeProfitHit(dir, exitPrice, pos.takeProfitPrice);
      }

      if (!isExpired && !hitStopLoss && !hitTakeProfit) continue;

      if (isExpired && !hitStopLoss && !hitTakeProfit) {
        const peakPnlPct = pos.direction === "up"
          ? (newPeak - pos.entryPrice) / pos.entryPrice
          : (pos.entryPrice - newPeak) / pos.entryPrice;

        const unrealizedPnlPct = pos.direction === "up"
          ? (exitPrice - pos.entryPrice) / pos.entryPrice
          : (pos.entryPrice - exitPrice) / pos.entryPrice;

        if (peakPnlPct > TRAILING_STOP.minPeakPnlPctToTrail) {
          const trailingStopPct = TRAILING_STOP.trailGivebackFraction;
          const newStopPrice = pos.direction === "up"
            ? newPeak * (1 - peakPnlPct * trailingStopPct)
            : newPeak * (1 + peakPnlPct * trailingStopPct);
          const extensionMs = TRAILING_STOP.expiryExtensionMs[pos.timeframe]
            ?? TRAILING_STOP.expiryExtensionMs._default;
          const newExpiry = new Date(Date.now() + extensionMs);

          await db.update(paperPositionsTable).set({
            expiresAt: newExpiry,
            stopLossPrice: newStopPrice,
            peakPrice: newPeak,
          }).where(eq(paperPositionsTable.id, pos.id));

          logger.info({
            agentName: pos.agentName || "unknown",
            coinName: pos.coinName,
            peakPnlPct: (peakPnlPct * 100).toFixed(2) + "%",
            currentPnlPct: (unrealizedPnlPct * 100).toFixed(2) + "%",
            newStopPrice: newStopPrice.toFixed(8),
            peakPrice: newPeak.toFixed(8),
            extensionMin: (extensionMs / 60000).toFixed(0),
          }, "Profitable position extended with trailing stop (peak-based)");
          continue;
        }
      }

      const priceRatio = exitPrice / pos.entryPrice;
      if (priceRatio > 3 || priceRatio < 0.33) {
        // Anomaly cancel: feed glitch, "this trade never happened".
        // The open path deducted positionSize + entryFee from cash, so the
        // full reversal must refund both. Prefer the entry_fee persisted on
        // the paper_trades row; fall back to recoverEntryFee for historical
        // rows opened before the column existed (NULL). Mark the trade row
        // pnl=0 to record that the open-time cash impact was fully reversed
        // (audit reconciles).
        const [tradeRow] = await db.select({ entryFee: paperTradesTable.entryFee })
          .from(paperTradesTable)
          .where(eq(paperTradesTable.id, pos.tradeId));
        const refundedEntryFee = tradeRow?.entryFee ?? recoverEntryFee(pos.positionSize);
        const refund = pos.positionSize + refundedEntryFee;
        await db.transaction(async (tx) => {
          await tx.delete(paperPositionsTable).where(eq(paperPositionsTable.id, pos.id));
          await tx.update(paperTradesTable).set({
            status: "cancelled",
            closedAt: now,
            pnl: 0,
            pnlPercent: 0,
          }).where(eq(paperTradesTable.id, pos.tradeId));
          const portfolio = await tx.select().from(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, pos.agentId));
          if (portfolio.length > 0) {
            await tx.update(paperPortfoliosTable).set({ cashBalance: portfolio[0].cashBalance + refund, updatedAt: now }).where(eq(paperPortfoliosTable.agentId, pos.agentId));
          }
        });
        logger.warn({
          agentName: pos.agentName,
          coinName: pos.coinName,
          entryPrice: pos.entryPrice,
          exitPrice,
          priceRatio: priceRatio.toFixed(4),
          refunded: "$" + refund.toFixed(4),
          entryFeeReversed: "$" + refundedEntryFee.toFixed(6),
        }, "Paper trade anomaly-cancelled — full open-time reversal (positionSize + entryFee)");

        // Phase 1 — adaptive trade journal: log the cancel event so abstain
        // / risk-management metrics can count it correctly. PnL is forced
        // to 0 to mirror the cash reversal.
        const [cancelledTradeRow] = await db
          .select()
          .from(paperTradesTable)
          .where(eq(paperTradesTable.id, pos.tradeId));
        void writeTradeJournal({
          tradeId: pos.tradeId,
          predictionId: cancelledTradeRow?.predictionId ?? null,
          agentId: pos.agentId,
          agentName: pos.agentName ?? "unknown",
          coinId: pos.coinId,
          coinName: pos.coinName,
          timeframe: pos.timeframe,
          direction: pos.direction,
          entryTime: cancelledTradeRow?.createdAt ?? now,
          exitTime: now,
          entryPriceRaw: pos.entryPrice,
          entryPriceAdj: pos.entryPrice,
          exitPriceRaw: exitPrice,
          exitPriceAdj: exitPrice,
          entryFee: refundedEntryFee,
          exitFee: 0,
          slippagePct: SLIPPAGE_PCT,
          positionSizeUsd: pos.positionSize,
          mfePct: null,
          maePct: null,
          exitReason: "anomaly-cancel",
          realizedPnlUsd: 0,
          realizedPnlPct: 0,
          // Phase 2 — prefer the regime captured at trade-open time
          // (decision time). Falls back to the current regime only if
          // the position pre-dates the column being added.
          regimeLabel: pos.entryRegimeLabel ?? getCurrentRegime()?.regimeLabel ?? null,
        });
        // Cancelled positions are treated as data glitches: no realized
        // outcome should train the meta-brain, and the trade->tick binding
        // should not linger until the TTL sweep.
        clearTickBinding(pos.tradeId);
        continue;
      }

      const exitPriceRaw = exitPrice;
      exitPrice = applyExitSlippage(exitPrice, pos.direction === "up" ? "up" : "down");

      let rawPnl: number;
      if (pos.direction === "up") {
        rawPnl = (exitPrice - pos.entryPrice) / pos.entryPrice * pos.positionSize;
      } else {
        rawPnl = (pos.entryPrice - exitPrice) / pos.entryPrice * pos.positionSize;
      }

      const exitFee = computeExitFee(pos.positionSize + rawPnl);
      const pnl = rawPnl - exitFee;

      const pnlPercent = (pnl / pos.positionSize) * 100;
      const closeReason = hitTakeProfit ? "take-profit" : hitStopLoss ? "stop-loss" : "expired";

      // MAE/MFE estimated from the tracked peakPrice. peakPrice is the best
      // price seen for the position, so MFE is straightforward; MAE is
      // approximated from the worst of (current price vs entry) since we
      // don't currently track a separate trough. Future Phase 1.x can add a
      // troughPrice column to make this exact.
      const peakPrice = pos.peakPrice ?? pos.entryPrice;
      let mfePct: number;
      let maePct: number;
      if (pos.direction === "up") {
        mfePct = ((peakPrice - pos.entryPrice) / pos.entryPrice) * 100;
        maePct = Math.max(0, ((pos.entryPrice - exitPriceRaw) / pos.entryPrice) * 100);
      } else {
        mfePct = ((pos.entryPrice - peakPrice) / pos.entryPrice) * 100;
        maePct = Math.max(0, ((exitPriceRaw - pos.entryPrice) / pos.entryPrice) * 100);
      }

      await db.transaction(async (tx) => {
        const stillExists = await tx.select().from(paperPositionsTable).where(eq(paperPositionsTable.id, pos.id));
        if (stillExists.length === 0) return;

        await tx
          .update(paperTradesTable)
          .set({
            exitPrice,
            pnl,
            pnlPercent,
            status: "closed",
            closedAt: now,
          })
          .where(and(eq(paperTradesTable.id, pos.tradeId), eq(paperTradesTable.status, "open")));

        const portfolio = await tx
          .select()
          .from(paperPortfoliosTable)
          .where(eq(paperPortfoliosTable.agentId, pos.agentId));

        if (portfolio.length > 0) {
          const p = portfolio[0];
          const newCash = p.cashBalance + pos.positionSize + pnl;

          await tx
            .update(paperPortfoliosTable)
            .set({
              cashBalance: newCash,
              totalTrades: p.totalTrades + 1,
              winningTrades: pnl > 0 ? p.winningTrades + 1 : p.winningTrades,
              losingTrades: pnl <= 0 ? p.losingTrades + 1 : p.losingTrades,
              updatedAt: now,
            })
            .where(eq(paperPortfoliosTable.agentId, pos.agentId));
        }

        await tx
          .delete(paperPositionsTable)
          .where(eq(paperPositionsTable.id, pos.id));

        closed++;

        logger.info({
          agentName: pos.agentName || "unknown",
          coinName: pos.coinName,
          direction: pos.direction,
          timeframe: pos.timeframe,
          entryPrice: pos.entryPrice,
          exitPrice,
          rawPnl: rawPnl.toFixed(4),
          exitFee: "$" + exitFee.toFixed(4),
          netPnl: pnl.toFixed(4),
          pnlPercent: pnlPercent.toFixed(2) + "%",
          closeReason,
        }, "Paper trade closed");
      });

      // Task #614 — re-evaluate the MTTM auto-disable rule after every
      // close. The evaluator (`evaluateMttmAutoDisable`) reads the
      // freshest config itself and short-circuits when MTTM is
      // disabled, already tripped, or the universe is empty, so it is
      // always safe to call. We deliberately do NOT pre-gate on
      // `getMttmConfigCached()` here: a cold or expired cache must
      // never cause the auto-disable rule to silently skip an
      // evaluation. We DO pre-check universe membership against an
      // awaited fresh snapshot purely to avoid an unnecessary DB scan
      // for unrelated coins; if that fetch fails we fall through and
      // call the evaluator anyway so the rule cannot be bypassed.
      try {
        const mttmFresh = await getMttmConfig();
        const inUniverse =
          mttmFresh.enabled &&
          !mttmFresh.disableReason &&
          mttmFresh.universeKeys.has(mttmSlotKey(pos.coinId, pos.timeframe));
        if (inUniverse) {
          // Fire-and-forget so a slow DB scan never blocks the close
          // sweep. Errors are non-fatal — the next close re-checks.
          evaluateMttmAutoDisable().catch((err) => {
            logger.warn({ err }, "mttm: auto-disable evaluation failed (non-fatal)");
          });
        }
      } catch (err) {
        // If we couldn't read fresh config, evaluate unconditionally.
        // The evaluator itself short-circuits when MTTM is off — this
        // is the safe-by-default branch.
        logger.warn(
          { err },
          "mttm: getMttmConfig() failed in close-path; evaluating auto-disable unconditionally",
        );
        evaluateMttmAutoDisable().catch((err2) => {
          logger.warn({ err: err2 }, "mttm: auto-disable evaluation failed (non-fatal)");
        });
      }

      // Phase 1 — adaptive trade journal: emit one row per close with full
      // PnL / fee / slippage / MAE / MFE / exit-reason detail. Outside the
      // transaction so a journal failure can never roll back the cash
      // settlement.
      const [closedTradeRow] = await db
        .select()
        .from(paperTradesTable)
        .where(eq(paperTradesTable.id, pos.tradeId));

      // Task #373 — feed the realized outcome back into the meta-brain's
      // bounded-learning loop. Fire-and-forget; timeouts and 4xx/5xx are
      // swallowed by the client. The directive is resolved by the
      // tick_id that authorized entry, stored at open time. Missing
      // outcome sub-metrics (correct_defense, correct_suppression,
      // missed_edge_cost) are passed as 0.0 (neutral) — the brain's
      // bounded plasticity handles sparse reward signal.
      try {
        // Task #381 step 5 — peek (do NOT consume) so transient
        // record_outcome failures can be retried by the next close
        // attempt or reaped by the TTL sweep. The binding is cleared
        // only on confirmed `{ok: true}` from the brain, by
        // sendRecordOutcome itself.
        const tickId = peekTickForTrade(pos.tradeId);
        if (tickId) {
          // pnl normalized to position size in [-1, +1]-ish range.
          const realizedPnl = pos.positionSize > 0 ? pnl / pos.positionSize : 0;
          // MAE is already a % of entry; convert to fraction.
          const realizedDrawdown = maePct / 100;
          // Stability proxy: 1 - |mfe-mae|/(mfe+mae). Closer to 1 = exit
          // near peak with low adverse excursion; closer to 0 = whipsaw.
          const spread = Math.max(1e-9, Math.abs(mfePct) + Math.abs(maePct));
          const realizedStability = Math.max(
            0,
            Math.min(1, 1 - Math.abs(Math.abs(mfePct) - Math.abs(maePct)) / spread),
          );
          const entryFeePaid = closedTradeRow?.entryFee ?? 0;
          const turnoverCost = pos.positionSize > 0
            ? (entryFeePaid + exitFee) / pos.positionSize
            : 0;
          // Task #381 step 5 — outcome honesty. We have first-hand
          // measurements for {pnl, drawdown, stability, turnover_cost};
          // everything else requires counterfactual or fleet-level
          // tracking we have not yet built, so we send `null` (the
          // wire layer translates to 0.0 for the vendored dataclass)
          // and rely on the brain's bounded plasticity to be robust
          // to sparse signal. Honest > pretending.
          void sendRecordOutcome(
            {
              tick_id: tickId,
              // Task #574 — pass the position's timeframe so the
              // wire layer can resolve `slice_role` from the
              // per-timeframe role registry AT SUBMISSION time.
              // Resolving here (rather than capturing the role at
              // open time) means a role flip mid-trade is honoured:
              // an outcome from a TF that has since been demoted
              // to `shadow`/`context`/`disabled` will not feed the
              // trust model.
              timeframe: pos.timeframe,
              timestamp: now.toISOString(),
              outcome: {
                realized_pnl: realizedPnl,
                realized_drawdown: realizedDrawdown,
                realized_stability: realizedStability,
                turnover_cost: turnoverCost,
                action_churn: null,
                correct_defense: null,
                correct_suppression: null,
                missed_edge_cost: null,
              },
            },
            { tradeId: pos.tradeId },
          ).catch(() => {
            // sendRecordOutcome already swallows; double safety so
            // an unhandled rejection cannot crash the close path.
          });
        } else {
          // No binding — either entry pre-dated meta-brain, or it was
          // already swept. Nothing to learn from. Idempotent no-op.
          clearTickBinding(pos.tradeId);
        }
      } catch (err) {
        // Never block trade close. The record_outcome client already
        // swallows most failures; this is belt-and-suspenders.
        logger.debug(
          { err: String(err), tradeId: pos.tradeId },
          "meta-brain record_outcome skipped (non-blocking)",
        );
      }

      void writeTradeJournal({
        tradeId: pos.tradeId,
        predictionId: closedTradeRow?.predictionId ?? null,
        agentId: pos.agentId,
        agentName: pos.agentName ?? "unknown",
        coinId: pos.coinId,
        coinName: pos.coinName,
        timeframe: pos.timeframe,
        direction: pos.direction,
        entryTime: closedTradeRow?.createdAt ?? now,
        exitTime: now,
        entryPriceRaw: pos.entryPrice,
        entryPriceAdj: pos.entryPrice,
        exitPriceRaw,
        exitPriceAdj: exitPrice,
        entryFee: closedTradeRow?.entryFee ?? null,
        exitFee,
        slippagePct: SLIPPAGE_PCT,
        positionSizeUsd: pos.positionSize,
        mfePct,
        maePct,
        exitReason: closeReason,
        realizedPnlUsd: pnl,
        realizedPnlPct: pnlPercent,
        // Phase 2 — decision-time regime captured when the position was
        // opened. Live current regime is only a backstop for legacy
        // rows.
        regimeLabel: pos.entryRegimeLabel ?? getCurrentRegime()?.regimeLabel ?? null,
      });

      // Task #444 — the LLM sidecar (postmortem narrator) was deleted.
      // Trade closes used to fan out to writeTradePostmortem here for an
      // operator-facing narrative; the deterministic-quant cutover removed
      // that surface entirely.
    }

    if (closed > 0) {
      logger.info({ closed }, "Closed paper positions");
    }

    return closed;
  } finally {
    isClosingPositions = false;
  }
}

export async function updatePortfolioValues(): Promise<void> {
  if (!isPriceDataFresh()) return;
  const prices = await fetchCoinPrices();
  if (prices.length === 0) return;
  const portfolios = await db.select().from(paperPortfoliosTable);

  // Task #491 — accumulate per-position marks across the whole sweep
  // so they can be persisted in a single bulk insert. Each mark gives
  // the meta-brain replay a real intra-trade price sample to compute
  // honest MAE / mark-to-market stability instead of falling back to
  // `max(0, -pnl_pct)` and a neutral 0.5. Skipped when the venue
  // returned no price for the coin or when the price is in the
  // anomaly band — those cases were never marked-to-market in
  // `openValue` either, so emitting a mark would lie.
  const marksToInsert: {
    positionId: number;
    tradeId: number;
    agentId: number;
    coinId: string;
    markPrice: number;
    pnlPct: number | null;
  }[] = [];

  for (const p of portfolios) {
    const positions = await db
      .select()
      .from(paperPositionsTable)
      .where(eq(paperPositionsTable.agentId, p.agentId));

    let openValue = 0;
    for (const pos of positions) {
      const coin = prices.find((c) => c.id === pos.coinId);
      if (!coin) {
        openValue += pos.positionSize;
        continue;
      }

      const priceRatio = coin.currentPrice / pos.entryPrice;
      if (priceRatio > 3 || priceRatio < 0.33) {
        openValue += pos.positionSize;
        continue;
      }

      let rawUnrealizedPnl: number;
      if (pos.direction === "up") {
        rawUnrealizedPnl = (coin.currentPrice - pos.entryPrice) / pos.entryPrice * pos.positionSize;
      } else {
        rawUnrealizedPnl = (pos.entryPrice - coin.currentPrice) / pos.entryPrice * pos.positionSize;
      }
      const exitValue = Math.max(0, pos.positionSize + rawUnrealizedPnl);
      const estExitFee = exitValue * MAKER_FEE_PCT;
      const estSlippageCost = exitValue * SLIPPAGE_PCT;
      const unrealizedPnl = rawUnrealizedPnl - estExitFee - estSlippageCost;
      openValue += pos.positionSize + unrealizedPnl;

      // Mark-to-market sample for the replay. Only persisted when the
      // open position is linked to a `paper_trades` row (tradeId > 0)
      // — strategy-lab synthetic positions occasionally use 0 as a
      // placeholder and aren't replay-eligible.
      if (pos.tradeId > 0) {
        marksToInsert.push({
          positionId: pos.id,
          tradeId: pos.tradeId,
          agentId: pos.agentId,
          coinId: pos.coinId,
          markPrice: coin.currentPrice,
          pnlPct:
            pos.positionSize > 0
              ? (unrealizedPnl / pos.positionSize) * 100
              : null,
        });
      }
    }

    const totalValue = p.cashBalance + openValue;

    const riskUpdate = checkRiskLimits(
      totalValue,
      p.peakValue,
      p.dayStartValue,
      p.dayStartDate
    );

    await db
      .update(paperPortfoliosTable)
      .set({
        totalValue,
        updatedAt: new Date(),
        peakValue: riskUpdate.updatedPeak,
        dayStartValue: riskUpdate.updatedDayStart,
        dayStartDate: riskUpdate.updatedDayDate,
      })
      .where(eq(paperPortfoliosTable.agentId, p.agentId));
  }

  // Bulk-insert all marks at the end so a single tick costs at most one
  // round trip regardless of how many open positions the fleet has. A
  // failure here is logged and swallowed — the marks table is purely a
  // feedback enrichment for the offline replay; we must never let it
  // block live mark-to-market portfolio updates.
  if (marksToInsert.length > 0) {
    try {
      await db.insert(paperPositionMarksTable).values(marksToInsert);
    } catch (err) {
      logger.warn(
        { err, count: marksToInsert.length },
        "Failed to persist paper_position_marks (replay enrichment) — portfolio update succeeded",
      );
    }
  }
}

/**
 * Task #505 — pull every `paper_position_marks` row whose `trade_id` is
 * in the supplied set, grouped by trade id and ordered chronologically
 * within each group. The reference implementation lives in the meta-brain
 * replay (`replay_meta_brain.py:_mae_from_marks` / `_stability_from_marks`),
 * which expects marks pre-sorted by `marked_at` so the consecutive-return
 * stdev is well defined. We mirror that ordering here.
 *
 * Returns an empty map when `tradeIds` is empty, so callers can call
 * unconditionally without an `if (ids.length)` guard.
 */
async function fetchMarksByTradeId(
  tradeIds: number[],
): Promise<Map<number, { markPrice: number }[]>> {
  const grouped = new Map<number, { markPrice: number }[]>();
  if (tradeIds.length === 0) return grouped;
  const rows = await db
    .select({
      tradeId: paperPositionMarksTable.tradeId,
      markPrice: paperPositionMarksTable.markPrice,
    })
    .from(paperPositionMarksTable)
    .where(inArray(paperPositionMarksTable.tradeId, tradeIds))
    .orderBy(paperPositionMarksTable.tradeId, paperPositionMarksTable.markedAt);
  for (const r of rows) {
    let arr = grouped.get(r.tradeId);
    if (!arr) {
      arr = [];
      grouped.set(r.tradeId, arr);
    }
    arr.push({ markPrice: r.markPrice });
  }
  return grouped;
}

/**
 * Task #505 — true intra-trade max-adverse-excursion as a fraction of
 * entry price. Mirrors `replay_meta_brain.py:_mae_from_marks` so the
 * dashboard never disagrees with the meta-brain replay's `realized_drawdown`.
 *
 *   buy  → max(0, (entry − min(mark_price)) / entry)
 *   sell → max(0, (max(mark_price) − entry) / entry)
 *
 * Returns `null` when there are no usable marks, when the entry price
 * is non-positive, or when the action is not "buy"/"sell" — those rows
 * render as "—" on the dashboard.
 */
export function maePctFromMarks(
  entry: number,
  action: string,
  marks: { markPrice: number }[],
): number | null {
  if (!marks.length || !(entry > 0)) return null;
  const a = action.toLowerCase();
  if (a !== "buy" && a !== "sell") return null;
  const prices = marks
    .map((m) => m.markPrice)
    .filter((p): p is number => Number.isFinite(p));
  if (prices.length === 0) return null;
  if (a === "buy") {
    let lo = prices[0];
    for (let i = 1; i < prices.length; i++) if (prices[i] < lo) lo = prices[i];
    return Math.max(0, (entry - lo) / entry);
  }
  let hi = prices[0];
  for (let i = 1; i < prices.length; i++) if (prices[i] > hi) hi = prices[i];
  return Math.max(0, (hi - entry) / entry);
}

/**
 * Task #505 — bounded stability score in [0, 1] derived from the
 * population standard deviation of consecutive mark-to-mark returns.
 * Mirrors `replay_meta_brain.py:_stability_from_marks`:
 *
 *   stability = clamp01(1 / (1 + 5 * sigma))
 *
 * Higher values mean a smoother price path during the hold. Returns
 * `null` when fewer than two valid returns exist (the replay uses 0.5
 * as a neutral fallback in that case, but the dashboard prefers an
 * explicit "—" so the operator can tell "we don't know" from "neutral").
 */
export function stabilityFromMarks(marks: { markPrice: number }[]): number | null {
  if (marks.length < 2) return null;
  const prices = marks
    .map((m) => m.markPrice)
    .filter((p): p is number => Number.isFinite(p));
  if (prices.length < 2) return null;
  const rets: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    const prev = prices[i - 1];
    if (prev > 0) rets.push((prices[i] - prev) / prev);
  }
  if (rets.length < 2) return null;
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance =
    rets.reduce((a, b) => a + (b - mean) * (b - mean), 0) / rets.length;
  const sigma = Math.sqrt(variance);
  const stability = 1 / (1 + 5 * sigma);
  if (!Number.isFinite(stability)) return null;
  return Math.max(0, Math.min(1, stability));
}

export interface PortfolioSummary {
  agentId: number;
  agentName: string;
  cashBalance: number;
  totalValue: number;
  /**
   * Starting capital this bot was seeded with (USD). Sourced from
   * `shared/trading-frictions.json:risk.initial_balance_usd` via
   * INITIAL_BALANCE so the dashboard cannot drift from the API. The
   * dashboard derives per-bot net P&L as `totalValue - startingCapital`
   * (Task #362 — fixes leaderboard P&L mixing realized-only with
   * mark-to-market equity).
   */
  startingCapital: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  openPositions: {
    coinId: string;
    coinName: string;
    direction: string;
    entryPrice: number;
    positionSize: number;
    unrealizedPnl: number;
    timeframe: string;
    expiresAt: string;
  }[];
  recentTrades: {
    /** paper_trades.id — durable join key for paper_position_marks. */
    id: number;
    coinId: string;
    coinName: string;
    action: string;
    entryPrice: number;
    exitPrice: number | null;
    pnl: number | null;
    pnlPercent: number | null;
    status: string;
    timeframe: string;
    createdAt: string;
    /**
     * Task #505 — true intra-trade max-adverse-excursion as a fraction
     * of entry price (0..∞, typically 0..0.2). Derived from
     * `paper_position_marks` joined on `trade_id`, mirroring the math in
     * `replay_meta_brain.py:_mae_from_marks`. `null` when the trade
     * predates the mark stream (Task #491 migration) so the dashboard
     * can render "—" instead of a misleading zero.
     */
    maePct: number | null;
    /**
     * Task #505 — bounded stability score [0, 1] derived from the
     * standard deviation of consecutive mark-to-mark returns
     * (`1 / (1 + 5*sigma)`), mirroring
     * `replay_meta_brain.py:_stability_from_marks`. `null` when there
     * are fewer than two usable returns.
     */
    stability: number | null;
  }[];
}

export async function getPortfolioSummaries(): Promise<PortfolioSummary[]> {
  const prices = await fetchCoinPrices();
  const portfolios = await db.select().from(paperPortfoliosTable).orderBy(paperPortfoliosTable.totalValue);
  const results: PortfolioSummary[] = [];

  for (const p of portfolios) {
    const positions = await db
      .select()
      .from(paperPositionsTable)
      .where(eq(paperPositionsTable.agentId, p.agentId));

    const openPositions = positions.map((pos) => {
      const coin = prices.find((c) => c.id === pos.coinId);
      const currentPrice = coin?.currentPrice || pos.entryPrice;
      const rawPnl = pos.direction === "up"
        ? (currentPrice - pos.entryPrice) / pos.entryPrice * pos.positionSize
        : (pos.entryPrice - currentPrice) / pos.entryPrice * pos.positionSize;
      const exitValue = Math.max(0, pos.positionSize + rawPnl);
      const estExitFee = exitValue * MAKER_FEE_PCT;
      const estSlippageCost = exitValue * SLIPPAGE_PCT;
      const unrealizedPnl = rawPnl - estExitFee - estSlippageCost;

      return {
        coinId: pos.coinId,
        coinName: pos.coinName,
        direction: pos.direction,
        entryPrice: pos.entryPrice,
        positionSize: pos.positionSize,
        unrealizedPnl,
        timeframe: pos.timeframe,
        expiresAt: pos.expiresAt.toISOString(),
      };
    });

    const recentTrades = await db
      .select()
      .from(paperTradesTable)
      .where(
        and(
          eq(paperTradesTable.agentId, p.agentId),
          eq(paperTradesTable.status, "closed"),
        ),
      )
      .orderBy(desc(paperTradesTable.createdAt))
      .limit(10);

    // Task #505 — surface true intra-trade max-adverse-excursion and a
    // bounded stability score per row, derived from the per-tick
    // `paper_position_marks` stream (Task #491). Marks are pulled in
    // one batch per portfolio (indexed by trade_id) and grouped in
    // memory rather than issuing one round trip per trade. Trades that
    // predate the mark stream simply yield no rows here, in which case
    // the helpers return null and the dashboard renders "—".
    const tradeIds = recentTrades.map((t) => t.id);
    const marksByTrade = await fetchMarksByTradeId(tradeIds);

    results.push({
      agentId: p.agentId,
      agentName: p.agentName,
      cashBalance: p.cashBalance,
      totalValue: p.totalValue,
      startingCapital: INITIAL_BALANCE,
      totalTrades: p.totalTrades,
      winningTrades: p.winningTrades,
      losingTrades: p.losingTrades,
      winRate: p.totalTrades > 0 ? (p.winningTrades / p.totalTrades) * 100 : 0,
      openPositions,
      recentTrades: recentTrades.map((t) => {
        const marks = marksByTrade.get(t.id) ?? [];
        return {
          id: t.id,
          coinId: t.coinId,
          coinName: t.coinName,
          action: t.action,
          entryPrice: t.entryPrice,
          exitPrice: t.exitPrice,
          pnl: t.pnl,
          pnlPercent: t.pnlPercent,
          status: t.status,
          timeframe: t.timeframe,
          createdAt: t.createdAt.toISOString(),
          maePct: maePctFromMarks(t.entryPrice, t.action, marks),
          stability: stabilityFromMarks(marks),
        };
      }),
    });
  }

  results.sort((a, b) => b.totalValue - a.totalValue);
  return results;
}

async function removeHodlerHankIfExists(): Promise<void> {
  const existing = await db.select().from(agentsTable).where(eq(agentsTable.name, HODLER_AGENT_NAME));
  if (existing.length === 0) return;
  const hankId = existing[0].id;
  await db.delete(paperPositionsTable).where(eq(paperPositionsTable.agentId, hankId));
  await db.delete(paperTradesTable).where(eq(paperTradesTable.agentId, hankId));
  await db.delete(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, hankId));
  await db.delete(agentsTable).where(eq(agentsTable.id, hankId));
  logger.info({ agentId: hankId }, "Hodler Hank removed — auto-deploy now serves as in-market benchmark");
}

async function _legacyEnsureHodlerHankAgent(): Promise<{ id: number; name: string } | null> {
  const existing = await db.select().from(agentsTable).where(eq(agentsTable.name, HODLER_AGENT_NAME));
  if (existing.length > 0) {
    if (existing[0].isActive === false) {
      await db.update(agentsTable).set({ isActive: true }).where(eq(agentsTable.id, existing[0].id));
    }
    return { id: existing[0].id, name: existing[0].name };
  }

  const [created] = await db.insert(agentsTable).values({
    name: HODLER_AGENT_NAME,
    personality: "Passive equal-weight basket holder — the live buy-and-hold benchmark",
    status: "active",
    isActive: true,
    evolutionMethod: "baseline",
    preferredTimeframes: JSON.stringify(["1d"]),
  }).returning({ id: agentsTable.id, name: agentsTable.name });

  await db.insert(paperPortfoliosTable).values({
    agentId: created.id,
    agentName: created.name,
    cashBalance: INITIAL_BALANCE,
    totalValue: INITIAL_BALANCE,
    totalTrades: 0,
    winningTrades: 0,
    losingTrades: 0,
  });

  logger.info({ agentId: created.id }, "Hodler Hank baseline agent created");
  return created;
}

// NOTE: Legacy `ensureHodlerInvested` was removed in Phase 5. It was dead code
// (never called) referencing undefined symbols (HODLER_BASKET_SIZE,
// HODLER_FAR_FUTURE, ensureHodlerHankAgent without underscore) that broke
// typecheck and would have crashed at runtime if ever wired up. The Hodler
// Hank baseline is now seeded by `_legacyEnsureHodlerHankAgent` only.

const ANCHOR_BASKET_SIZE = 4;
const ANCHOR_MIN_MCAP_USD = 50_000_000;
const ANCHOR_MIN_24H_PCT_BULL = 1.0;
const ANCHOR_MIN_24H_PCT_NEUTRAL = 1.0;
const SHORT_BASKET_SIZE = 2;
const SHORT_MAX_24H_PCT = -1.5;
const SHORT_BEAR_CONFIDENCE = 0.6;

const autoDeployHistory: { ts: number; usd: number }[] = [];

function recordAutoDeploy(usd: number): void {
  const now = Date.now();
  autoDeployHistory.push({ ts: now, usd });
  const cutoff = now - 24 * 60 * 60 * 1000;
  while (autoDeployHistory.length > 0 && autoDeployHistory[0].ts < cutoff) {
    autoDeployHistory.shift();
  }
}

export function getAutoDeployedLast24hUsd(): number {
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  return autoDeployHistory.filter(e => e.ts >= cutoff).reduce((s, e) => s + e.usd, 0);
}

export async function autoDeployIdleCash(prices: CoinPrice[]): Promise<void> {
  const regime = getCurrentRegime();
  if (!regime) return;

  // Decide which side to deploy based on regime.
  // - bullish/neutral trendBias: long momentum basket (stricter momentum filter in neutral)
  // - confirmed bear regime: small short basket of weakest coins
  const trendBias = regime.trendBias;
  const isBear = regime.regime === "bear" && regime.confidence >= SHORT_BEAR_CONFIDENCE;

  let direction: "up" | "down";
  let basket: CoinPrice[];
  let basketCashFraction: number;
  let stopLossMult: number;
  let takeProfitMult: number;

  if (trendBias === "bullish" || trendBias === "neutral") {
    const minMomentum = trendBias === "bullish" ? ANCHOR_MIN_24H_PCT_BULL : ANCHOR_MIN_24H_PCT_NEUTRAL;
    const eligible = prices.filter(p =>
      p.currentPrice > 0 &&
      p.marketCap >= ANCHOR_MIN_MCAP_USD &&
      (p.priceChange24h ?? 0) >= minMomentum
    );
    const sortedByMomentum = [...eligible].sort((a, b) => (b.priceChange24h ?? 0) - (a.priceChange24h ?? 0));
    basket = sortedByMomentum.slice(0, ANCHOR_BASKET_SIZE);
    if (basket.length === 0) {
      logger.info({ trendBias, minMomentum, eligibleCount: eligible.length, totalCoins: prices.length }, "Auto-deploy skipped — no positive-momentum coins meet criteria");
      return;
    }
    direction = "up";
    basketCashFraction = 0.85;
    stopLossMult = 0.82;
    takeProfitMult = 1.25;
    logger.info({
      trendBias,
      basket: basket.map(c => `${c.name}(+${(c.priceChange24h ?? 0).toFixed(2)}%)`).join(", "),
    }, "Auto-deploy long momentum basket selected");
  } else if (isBear) {
    const eligible = prices.filter(p =>
      p.currentPrice > 0 &&
      p.marketCap >= ANCHOR_MIN_MCAP_USD &&
      (p.priceChange24h ?? 0) <= SHORT_MAX_24H_PCT
    );
    const sortedByWeakness = [...eligible].sort((a, b) => (a.priceChange24h ?? 0) - (b.priceChange24h ?? 0));
    basket = sortedByWeakness.slice(0, SHORT_BASKET_SIZE);
    if (basket.length === 0) {
      logger.info({ regime: regime.regime, eligibleCount: eligible.length, totalCoins: prices.length }, "Auto-deploy skipped — no negative-momentum coins meet short criteria");
      return;
    }
    direction = "down";
    // Shorts in bear regime are riskier — deploy only a small slice of cash.
    basketCashFraction = 0.25;
    stopLossMult = 1.18;
    takeProfitMult = 0.80;
    logger.info({
      regime: regime.regime,
      confidence: regime.confidence.toFixed(2),
      basket: basket.map(c => `${c.name}(${(c.priceChange24h ?? 0).toFixed(2)}%)`).join(", "),
    }, "Auto-deploy short basket selected (bear regime)");
  } else {
    return;
  }

  // The opener must use the same live-executor scope as attribution below:
  // active ai-bots only, not archived, and not the legacy archive bucket.
  // Otherwise old personality portfolios can receive fresh auto-deploy
  // trades while the dashboard correctly excludes them, creating invisible
  // risk and fake "live fleet" behavior.
  const portfolios = await db.select().from(paperPortfoliosTable);
  const liveAiBotAgents = await db
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
  const liveAiBotIds = new Set(liveAiBotAgents.map((a) => a.id));

  for (const portfolio of portfolios) {
    if (portfolio.agentName === HODLER_AGENT_NAME) continue;
    if (!liveAiBotIds.has(portfolio.agentId)) continue;

    const cashRatio = portfolio.totalValue > 0 ? portfolio.cashBalance / portfolio.totalValue : 0;
    if (cashRatio < 0.30) continue;

    const positions = await db.select().from(paperPositionsTable).where(eq(paperPositionsTable.agentId, portfolio.agentId));
    const slotsOpen = MAX_OPEN_POSITIONS_PER_AGENT - positions.length;
    if (slotsOpen <= 0) continue;

    const cooldownCutoff = new Date(Date.now() - 6 * 60 * 60 * 1000);
    const recentClosed = await db
      .select({ coinId: paperTradesTable.coinId })
      .from(paperTradesTable)
      .where(
        and(
          eq(paperTradesTable.agentId, portfolio.agentId),
          eq(paperTradesTable.status, "closed"),
          gte(paperTradesTable.closedAt, cooldownCutoff),
        ),
      );
    const cooldownCoins = new Set(recentClosed.map(r => r.coinId));

    // Coin-level exclusivity: don't open a new auto-deploy on a coin we
    // already hold in either direction (avoids accidental hedges).
    const heldCoinIds = new Set(positions.map(p => p.coinId));
    const missingCoins = basket.filter(c =>
      !heldCoinIds.has(c.id) &&
      !cooldownCoins.has(c.id),
    );
    const toOpen = missingCoins.slice(0, slotsOpen);
    if (toOpen.length === 0) continue;

    // Long deploys lean in (0.75 of equity); short deploys stay small (0.20 of equity).
    const equityCap = direction === "up" ? 0.75 : 0.20;
    const targetDeploy = Math.min(portfolio.totalValue * equityCap, portfolio.cashBalance * basketCashFraction);
    const perCoin = targetDeploy / toOpen.length;
    if (perCoin < 10) continue;

    const tradeAction = direction === "up" ? "buy" : "sell";

    for (const coin of toOpen) {
      const jitter = 0.97 + Math.random() * 0.06;
      const allocate = perCoin * jitter;
      // Symmetric slippage so the auto-deploy lane uses the same
      // execution-realistic entry as openPaperTrade above. This matters
      // for: stop/take prices, peakPrice baseline, and the realized P&L
      // attribution that downstream meta-brain learning consumes.
      const adjustedEntryPrice = applyEntrySlippage(coin.currentPrice, direction);
      const quantity = allocate / adjustedEntryPrice;
      const expiresAt = new Date(Date.now() + 6 * 60 * 60 * 1000);

      try {
        let opened = false;
        // Task #405 / B-AUTODEPLOY-NOFEE — the auto-deploy basket buy used
        // to insert the trade row WITHOUT `entry_fee` and to debit cash
        // by `allocate` only, leaving every basket buy NULL on entry_fee
        // and skipping the maker-fee charge entirely. Cash debit and the
        // stored row are now both fee-aware so:
        //   * recordOutcome's entryFeePaid != 0 and the meta-brain learning
        //     loop sees the real `turnover_cost` for these strategies, and
        //   * the bot's cash position no longer overstates idle balance.
        const entryFee = computeEntryFee(allocate);
        const totalDebit = allocate + entryFee;
        await db.transaction(async (tx) => {
          const fresh = await tx.select().from(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, portfolio.agentId)).for("update");
          if (fresh.length === 0) return;
          if (fresh[0].cashBalance < totalDebit) {
            logger.warn({ agentName: portfolio.agentName, coin: coin.name, cash: fresh[0].cashBalance, want: totalDebit, allocate, entryFee }, "Auto-deploy aborted — insufficient cash");
            return;
          }

          const livePositions = await tx.select().from(paperPositionsTable).where(eq(paperPositionsTable.agentId, portfolio.agentId));
          if (livePositions.length >= MAX_OPEN_POSITIONS_PER_AGENT) return;
          if (livePositions.some(p => p.coinId === coin.id)) return;

          const [trade] = await tx.insert(paperTradesTable).values({
            agentId: portfolio.agentId,
            agentName: portfolio.agentName,
            coinId: coin.id,
            coinName: coin.name,
            action: tradeAction,
            entryPrice: adjustedEntryPrice,
            quantity,
            positionSize: allocate,
            entryFee,
            timeframe: "6h",
            predictionId: null,
            status: "open",
          }).returning({ id: paperTradesTable.id });

          await tx.insert(paperPositionsTable).values({
            agentId: portfolio.agentId,
            agentName: portfolio.agentName,
            coinId: coin.id,
            coinName: coin.name,
            direction,
            entryPrice: adjustedEntryPrice,
            quantity,
            positionSize: allocate,
            timeframe: "6h",
            tradeId: trade.id,
            expiresAt,
            stopLossPrice: adjustedEntryPrice * stopLossMult,
            takeProfitPrice: adjustedEntryPrice * takeProfitMult,
            peakPrice: adjustedEntryPrice,
            // Phase 2 — capture decision-time regime; see same field in
            // openPaperTrade.
            entryRegimeLabel: getCurrentRegime()?.regimeLabel ?? null,
          });

          await tx.update(paperPortfoliosTable).set({
            cashBalance: fresh[0].cashBalance - totalDebit,
            updatedAt: new Date(),
          }).where(eq(paperPortfoliosTable.agentId, portfolio.agentId));
          opened = true;
        });

        if (opened) {
          recordAutoDeploy(allocate);
          logger.info({
            agentName: portfolio.agentName,
            coin: coin.name,
            direction,
            allocate: "$" + allocate.toFixed(2),
            cashRatio: (cashRatio * 100).toFixed(0) + "%",
          }, "Auto-deployed idle cash to diversified anchor basket");
        }
      } catch (err) {
        logger.warn({ err, agentName: portfolio.agentName, coin: coin.name }, "Auto-deploy failed");
      }
    }
  }
}
// Suppress unused-var lint for HODLER_REBALANCE_THRESHOLD reserved for future drift rebalancing
void HODLER_REBALANCE_THRESHOLD;

interface SideAttribution {
  realizedPnlUsd: number;
  unrealizedPnlUsd: number;
  netPnlUsd: number;
  closedTrades: number;
  openPositions: number;
  deployedUsd: number;
}

export interface CoinAttribution {
  coinId: string;
  symbol: string;
  name: string;
  realizedPnlUsd: number;
  unrealizedPnlUsd: number;
  netPnlUsd: number;
  closedTrades: number;
  openPositions: number;
  deployedUsd: number;
  longCount: number;
  shortCount: number;
}

interface WindowAttribution {
  long: SideAttribution;
  short: SideAttribution;
  total: SideAttribution;
  coins: CoinAttribution[];
}

export interface AutoDeployAttribution {
  window24h: WindowAttribution;
  window7d: WindowAttribution;
  open: WindowAttribution;
}

function emptySide(): SideAttribution {
  return {
    realizedPnlUsd: 0,
    unrealizedPnlUsd: 0,
    netPnlUsd: 0,
    closedTrades: 0,
    openPositions: 0,
    deployedUsd: 0,
  };
}

function emptyWindow(): WindowAttribution {
  return { long: emptySide(), short: emptySide(), total: emptySide(), coins: [] };
}

function getOrCreateCoin(map: Map<string, CoinAttribution>, coinId: string, prices: CoinPrice[]): CoinAttribution {
  let coin = map.get(coinId);
  if (!coin) {
    const meta = prices.find(p => p.id === coinId);
    coin = {
      coinId,
      symbol: meta?.symbol ?? coinId,
      name: meta?.name ?? coinId,
      realizedPnlUsd: 0,
      unrealizedPnlUsd: 0,
      netPnlUsd: 0,
      closedTrades: 0,
      openPositions: 0,
      deployedUsd: 0,
      longCount: 0,
      shortCount: 0,
    };
    map.set(coinId, coin);
  }
  return coin;
}

function finalizeCoins(map: Map<string, CoinAttribution>): CoinAttribution[] {
  const arr = Array.from(map.values());
  for (const c of arr) {
    c.netPnlUsd = c.realizedPnlUsd + c.unrealizedPnlUsd;
  }
  arr.sort((a, b) => b.netPnlUsd - a.netPnlUsd);
  return arr;
}

function rollupTotals(w: WindowAttribution): void {
  w.total.realizedPnlUsd = w.long.realizedPnlUsd + w.short.realizedPnlUsd;
  w.total.unrealizedPnlUsd = w.long.unrealizedPnlUsd + w.short.unrealizedPnlUsd;
  w.total.netPnlUsd = w.total.realizedPnlUsd + w.total.unrealizedPnlUsd;
  w.total.closedTrades = w.long.closedTrades + w.short.closedTrades;
  w.total.openPositions = w.long.openPositions + w.short.openPositions;
  w.total.deployedUsd = w.long.deployedUsd + w.short.deployedUsd;
  w.long.netPnlUsd = w.long.realizedPnlUsd + w.long.unrealizedPnlUsd;
  w.short.netPnlUsd = w.short.realizedPnlUsd + w.short.unrealizedPnlUsd;
}

// Identifies "auto-deploy" trades as those originated by autoDeployIdleCash:
// they are inserted with predictionId = null on agents whose strategyType is
// "ai-bots". The Hodler Hank baseline agent is no longer present (it was
// removed in favour of auto-deploy as the in-market benchmark), and other
// strategy-lab agents (dca-cb, buy-hold, trend-filter) use different
// strategyType values, so the (ai-bots, predictionId IS NULL) pair uniquely
// fingerprints auto-deploy trades.
//
// Task #532 / C-9 — historic `legacy_archived` agents still have
// `strategy_type='ai-bots'` so an unfiltered query attributes their old
// trades to the live executor fleet. The Phase-0 truth audit found this
// inflated `realityCheck.autoDeploy.window7d.long.realizedPnlUsd` to $3.24
// while the 4 live executors had **zero** closed trades over the same 7d
// window. Filter to active, non-archived, non-legacy rows so the dashboard
// reports honest live executor attribution.
export async function getAutoDeployAttribution(prices: CoinPrice[]): Promise<AutoDeployAttribution> {
  // Task #532 / Rev 2.1 — additionally enforce `isActive=true` so an
  // operator-paused executor cannot fabricate live attribution. The
  // schema allows non-archived rows with `is_active=false` (the boot
  // path can leave such rows in place); without this filter the live
  // P&L surface would silently drift again the moment a bot is
  // deactivated without being archived. The contract test asserts
  // this invariant so it cannot regress.
  const aiBotAgents = await db
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
  const aiBotIds = aiBotAgents.map(a => a.id);

  const result: AutoDeployAttribution = {
    window24h: emptyWindow(),
    window7d: emptyWindow(),
    open: emptyWindow(),
  };

  if (aiBotIds.length === 0) {
    rollupTotals(result.window24h);
    rollupTotals(result.window7d);
    rollupTotals(result.open);
    return result;
  }

  const now = Date.now();
  const cutoff7d = new Date(now - 7 * 24 * 60 * 60 * 1000);
  const cutoff24h = now - 24 * 60 * 60 * 1000;

  const coins24h = new Map<string, CoinAttribution>();
  const coins7d = new Map<string, CoinAttribution>();
  const coinsOpen = new Map<string, CoinAttribution>();

  // Closed auto-deploy trades within the last 7 days.
  const closedTrades = await db
    .select({
      id: paperTradesTable.id,
      coinId: paperTradesTable.coinId,
      action: paperTradesTable.action,
      pnl: paperTradesTable.pnl,
      positionSize: paperTradesTable.positionSize,
      closedAt: paperTradesTable.closedAt,
      status: paperTradesTable.status,
    })
    .from(paperTradesTable)
    .where(
      and(
        inArray(paperTradesTable.agentId, aiBotIds),
        isNull(paperTradesTable.predictionId),
        eq(paperTradesTable.status, "closed"),
        gte(paperTradesTable.closedAt, cutoff7d),
      ),
    );

  for (const t of closedTrades) {
    const isLong = t.action === "buy";
    const pnl = t.pnl ?? 0;
    const side7d = isLong ? result.window7d.long : result.window7d.short;
    side7d.realizedPnlUsd += pnl;
    side7d.closedTrades += 1;
    side7d.deployedUsd += t.positionSize;

    const coin7d = getOrCreateCoin(coins7d, t.coinId, prices);
    coin7d.realizedPnlUsd += pnl;
    coin7d.closedTrades += 1;
    coin7d.deployedUsd += t.positionSize;
    if (isLong) coin7d.longCount += 1; else coin7d.shortCount += 1;

    const closedAtMs = t.closedAt ? t.closedAt.getTime() : 0;
    if (closedAtMs >= cutoff24h) {
      const side24h = isLong ? result.window24h.long : result.window24h.short;
      side24h.realizedPnlUsd += pnl;
      side24h.closedTrades += 1;
      side24h.deployedUsd += t.positionSize;

      const coin24h = getOrCreateCoin(coins24h, t.coinId, prices);
      coin24h.realizedPnlUsd += pnl;
      coin24h.closedTrades += 1;
      coin24h.deployedUsd += t.positionSize;
      if (isLong) coin24h.longCount += 1; else coin24h.shortCount += 1;
    }
  }

  // Open auto-deploy positions: trade's predictionId is null and the agent is
  // an ai-bots agent. Compute unrealized P&L using current prices.
  const openTrades = await db
    .select({ id: paperTradesTable.id })
    .from(paperTradesTable)
    .where(
      and(
        inArray(paperTradesTable.agentId, aiBotIds),
        isNull(paperTradesTable.predictionId),
        eq(paperTradesTable.status, "open"),
      ),
    );
  const openTradeIds = openTrades.map(t => t.id);

  if (openTradeIds.length > 0) {
    const openPositions = await db
      .select()
      .from(paperPositionsTable)
      .where(inArray(paperPositionsTable.tradeId, openTradeIds));

    const priceById = new Map(prices.map(p => [p.id, p.currentPrice]));
    for (const pos of openPositions) {
      const isLong = pos.direction === "up";
      const current = priceById.get(pos.coinId) ?? pos.entryPrice;
      const pnl = pos.entryPrice > 0
        ? (isLong
          ? (current - pos.entryPrice) / pos.entryPrice
          : (pos.entryPrice - current) / pos.entryPrice) * pos.positionSize
        : 0;

      const side = isLong ? result.open.long : result.open.short;
      side.unrealizedPnlUsd += pnl;
      side.openPositions += 1;
      side.deployedUsd += pos.positionSize;

      const side24h = isLong ? result.window24h.long : result.window24h.short;
      side24h.unrealizedPnlUsd += pnl;
      side24h.openPositions += 1;

      const side7d = isLong ? result.window7d.long : result.window7d.short;
      side7d.unrealizedPnlUsd += pnl;
      side7d.openPositions += 1;

      const coinOpen = getOrCreateCoin(coinsOpen, pos.coinId, prices);
      coinOpen.unrealizedPnlUsd += pnl;
      coinOpen.openPositions += 1;
      coinOpen.deployedUsd += pos.positionSize;
      if (isLong) coinOpen.longCount += 1; else coinOpen.shortCount += 1;

      const coin24h = getOrCreateCoin(coins24h, pos.coinId, prices);
      coin24h.unrealizedPnlUsd += pnl;
      coin24h.openPositions += 1;
      if (isLong) coin24h.longCount += 1; else coin24h.shortCount += 1;

      const coin7d = getOrCreateCoin(coins7d, pos.coinId, prices);
      coin7d.unrealizedPnlUsd += pnl;
      coin7d.openPositions += 1;
      if (isLong) coin7d.longCount += 1; else coin7d.shortCount += 1;
    }
  }

  result.window24h.coins = finalizeCoins(coins24h);
  result.window7d.coins = finalizeCoins(coins7d);
  result.open.coins = finalizeCoins(coinsOpen);

  rollupTotals(result.window24h);
  rollupTotals(result.window7d);
  rollupTotals(result.open);
  return result;
}

const SNAPSHOT_INTERVAL_MS = 60 * 60 * 1000;
let lastSnapshotAt = 0;

export interface AutoDeployAttributionHistoryPoint {
  capturedAt: number;
  totalNetPnlUsd: number;
  longRealizedPnlUsd: number;
  longUnrealizedPnlUsd: number;
  shortRealizedPnlUsd: number;
  shortUnrealizedPnlUsd: number;
  deployedUsd: number;
  closedTrades: number;
  openPositions: number;
}

export async function recordAutoDeployAttributionSnapshot(
  attribution: AutoDeployAttribution,
  opts: { force?: boolean } = {},
): Promise<boolean> {
  const now = Date.now();
  if (!opts.force && now - lastSnapshotAt < SNAPSHOT_INTERVAL_MS) return false;

  const w7d = attribution.window7d;
  const open = attribution.open;
  try {
    await db.insert(autoDeployAttributionSnapshotsTable).values({
      longRealizedPnlUsd: w7d.long.realizedPnlUsd,
      longUnrealizedPnlUsd: open.long.unrealizedPnlUsd,
      shortRealizedPnlUsd: w7d.short.realizedPnlUsd,
      shortUnrealizedPnlUsd: open.short.unrealizedPnlUsd,
      totalNetPnlUsd: w7d.total.realizedPnlUsd + open.total.unrealizedPnlUsd,
      deployedUsd: open.total.deployedUsd,
      closedTrades: w7d.total.closedTrades,
      openPositions: open.total.openPositions,
    });
    lastSnapshotAt = now;
    return true;
  } catch (err) {
    logger.error({ err }, "Failed to record auto-deploy attribution snapshot");
    return false;
  }
}

export async function getAutoDeployAttributionHistory(
  hours: number = 168,
): Promise<AutoDeployAttributionHistoryPoint[]> {
  const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
  const rows = await db
    .select()
    .from(autoDeployAttributionSnapshotsTable)
    .where(gte(autoDeployAttributionSnapshotsTable.capturedAt, cutoff))
    .orderBy(autoDeployAttributionSnapshotsTable.capturedAt);
  return rows.map(r => ({
    capturedAt: r.capturedAt.getTime(),
    totalNetPnlUsd: r.totalNetPnlUsd,
    longRealizedPnlUsd: r.longRealizedPnlUsd,
    longUnrealizedPnlUsd: r.longUnrealizedPnlUsd,
    shortRealizedPnlUsd: r.shortRealizedPnlUsd,
    shortUnrealizedPnlUsd: r.shortUnrealizedPnlUsd,
    deployedUsd: r.deployedUsd,
    closedTrades: r.closedTrades,
    openPositions: r.openPositions,
  }));
}


