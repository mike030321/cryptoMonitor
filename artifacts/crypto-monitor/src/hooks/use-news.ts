import { useQuery } from "@tanstack/react-query";

export interface BestPick {
  coinId: string;
  coinName: string;
  coinSymbol: string;
  currentPrice: number;
  action: "buy" | "sell" | "hold";
  successProbability: number;
  holdTimeframe: string;
  holdMinutes: number;
  expectedPriceChange: number;
  reasoning: string;
  agentConsensus: {
    agentName: string;
    direction: string;
    confidence: number;
    score: number;
    accuracy: number;
  }[];
  riskLevel: "low" | "medium" | "high" | "extreme";
  timeframeBreakdown: {
    timeframe: string;
    direction: string;
    avgConfidence: number;
  }[];
  // Task #444 — separated explainer + quant context. The legacy
  // `newsFactors`, `newsTags`, `whyExplanation` fields belonged to the
  // LLM/news plane and are removed; `brain` is now `"QUANT" | "ABSTAIN"`.
  whatExplanation?: string;
  modelProbability?: number;
  evAfterFeesPct?: number;
  brain?: "QUANT" | "ABSTAIN" | null;
  // Task #532 / C-2 — when the brain runtime is anything other than
  // `online`, the API short-circuits and emits this suppressedReason
  // so the dashboard can render an honest "no live consensus" card
  // instead of a fake recommendation. Backend (api-server
  // computeBrainRuntimeState) emits one of:
  //   "brain_offline"          – flag/source explicitly disabled
  //   "brain_offline_no_model" – flag enabled but ml-engine has no model
  //   "brain_status_unknown"   – ml-engine unreachable / status indeterminate
  // We also accept `string` so a future backend value cannot silently
  // fall through to the recommendation layout.
  suppressedReason?:
    | "brain_offline"
    | "brain_offline_no_model"
    | "brain_status_unknown"
    | (string & {})
    | null;
  // Task #532 / Rev 2 — backend also surfaces the raw runtime state for
  // observability/badging. Same enum as /brain/runtime-status.state.
  brainRuntimeState?:
    | "online"
    | "offline_disabled"
    | "offline_no_model"
    | "unknown"
    | (string & {})
    | null;
  updatedAt: string;
}

export interface CoinSignal {
  coinId: string;
  coinName: string;
  coinSymbol: string;
  currentPrice: number;
  priceChange24h: number;
  signal: "buy" | "sell" | "hold";
  strength: number;
  confidence: number;
  agentAgreement: number;
}

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

export function useBestPick() {
  return useQuery<BestPick>({
    queryKey: ["crypto-best-pick"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/best-pick`);
      if (!res.ok) throw new Error("Failed to fetch best pick");
      return res.json();
    },
    refetchInterval: 30000,
  });
}

export interface PaperPortfolio {
  agentId: number;
  agentName: string;
  /**
   * Task #512 — `executor` for the 4 deterministic family agents
   * (Momentum Core, Mean Reversion Core, Breakout Core, Volatility
   * Defensive); `benchmark` for Strategy-Lab `baseline_reference`
   * rows. `null` for legacy rows that pre-date the registry — these
   * never reach the dashboard once the boot archive sweep has run.
   * The dashboard splits the leaderboard into a Family Fleet (4
   * cards) and a Benchmarks panel using this tag.
   */
  kind?: "executor" | "benchmark" | null;
  /** Task #512 — registry profile_id for the underlying agent row, or `null`. */
  profileId?: string | null;
  cashBalance: number;
  totalValue: number;
  /**
   * Starting capital seeded into this bot (USD). The dashboard derives
   * net P&L per bot as `totalValue - startingCapital` so the displayed
   * P&L always reconciles with the displayed equity (Task #362). The
   * legacy realized-only `totalPnl` / `totalPnlPercent` fields were
   * dropped from the API in Task #370 — derive net P&L from this seed.
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
     * of entry price (e.g. 0.012 = 1.2% drawdown), derived from
     * `paper_position_marks` joined on `trade_id`. `null` for trades
     * that predate the mark stream — render as "—".
     */
    maePct: number | null;
    /**
     * Task #505 — bounded stability score in [0, 1] from mark-to-mark
     * return stdev (`1 / (1 + 5*sigma)`). Higher means a smoother hold.
     * `null` when fewer than two usable returns exist.
     */
    stability: number | null;
  }[];
}

export function usePaperPortfolios() {
  return useQuery<PaperPortfolio[]>({
    queryKey: ["crypto-paper-portfolios"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/paper-portfolios`);
      if (!res.ok) throw new Error("Failed to fetch paper portfolios");
      return res.json();
    },
    refetchInterval: 30000,
  });
}

export function useCoinSignals() {
  return useQuery<CoinSignal[]>({
    queryKey: ["crypto-coin-signals"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/coin-signals`);
      if (!res.ok) throw new Error("Failed to fetch coin signals");
      return res.json();
    },
    refetchInterval: 30000,
  });
}

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
}

export function useAgentSpecializations() {
  return useQuery<AgentSpecialization[]>({
    queryKey: ["crypto-agent-specializations"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/agent-specializations`);
      if (!res.ok) throw new Error("Failed to fetch agent specializations");
      return res.json();
    },
    refetchInterval: 30000,
  });
}

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

export function useContagionAlerts() {
  return useQuery<ContagionAlert[]>({
    queryKey: ["crypto-contagion-alerts"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/contagion-alerts`);
      if (!res.ok) throw new Error("Failed to fetch contagion alerts");
      return res.json();
    },
    refetchInterval: 30000,
  });
}

// Task #453 — `BurnStatus`, `ApiBudget`, and `useApiBudget` removed
// alongside the api-server's rate-limiter. The matching `/crypto/api-budget`
// endpoint is gone; nothing in the UI rendered the data anymore (the LLM
// plane was retired in Task #444 and the budget always reported zeros).

// Task #444 — `agent-evolution` (LLM-driven personality mutation) is gone.
// The `EvolutionFitness` / `EvolutionStatus` interfaces and the
// `useEvolutionStatus` hook were removed. The admin and agent-detail
// pages no longer render evolution sections; the server endpoint
// `/crypto/evolution-status` remains as an empty stub for back-compat.

export interface RealityCheck {
  assumptions: {
    feePerSidePct: number;
    slippagePerSidePct: number;
    roundTripCostPct: number;
    benchmarkLabel: string;
  };
  benchmark: { returnPct: number; pnlUsd: number };
  fleet: {
    startingCapital: number;
    grossPnlUsd: number;
    netPnlUsd: number;
    alphaPnlUsd: number;
    avgGrossReturnPct: number;
    avgNetReturnPct: number;
    avgAlphaVsBenchmark: number;
    totalCosts: number;
    totalTurnover: number;
    totalClosedTrades: number;
    agentsBeatingBenchmark: number;
    totalAgents: number;
  };
  agents: {
    agentId: number;
    agentName: string;
    grossTotalValue: number;
    netTotalValue: number;
    grossPnlUsd: number;
    netPnlUsd: number;
    grossReturnPct: number;
    netReturnPct: number;
    alphaVsBenchmark: number;
    totalCosts: number;
    closedTrades: number;
    openTrades: number;
    turnover: number;
  }[];
  autoDeploy?: {
    deployedLast24hUsd: number;
    attribution: AutoDeployAttribution;
  };
}

export interface AutoDeploySide {
  realizedPnlUsd: number;
  unrealizedPnlUsd: number;
  netPnlUsd: number;
  closedTrades: number;
  openPositions: number;
  deployedUsd: number;
}

export interface AutoDeployCoin {
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

export interface AutoDeployWindow {
  long: AutoDeploySide;
  short: AutoDeploySide;
  total: AutoDeploySide;
  coins: AutoDeployCoin[];
}

export interface AutoDeployAttribution {
  window24h: AutoDeployWindow;
  window7d: AutoDeployWindow;
  open: AutoDeployWindow;
}

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

export interface AutoDeployAttributionHistory {
  hours: number;
  points: AutoDeployAttributionHistoryPoint[];
  delta24h: number | null;
}

export interface SkipEvent {
  ts: number;
  reason: string;
  agentName: string;
  agentId: number | null;
  coinId: string | null;
  message: string;
  details: Record<string, unknown>;
}

export type TuningGateKey =
  | "MIN_CONFIDENCE_TO_TRADE"
  | "COUNTER_TREND_MIN_CONFIDENCE"
  | "MIN_TP_DISTANCE_PCT"
  | "MIN_EV_VS_COST";

export interface TuningSuggestion {
  direction: "loosen" | "tighten";
  gate: TuningGateKey;
  label: string;
  reason: string | null;
  reasonLabel: string;
  shareOfSkips: number;
  totalSkips: number;
  openPositionCount: number;
  healthyOpenFloor: number;
  currentValue: number;
  proposedValue: number;
  loosenPct: number;
  unit: "ratio" | "pct" | "multiple";
  projectedAdditionalTrades: number;
  projectedAdditionalTradesCapped: boolean;
  projectedSampleSize: number;
  sustainedHealthyTicks?: number;
  requiredHealthyTicks?: number;
}

export type TuningChangeKind = "gate" | "auto-tighten-toggle";

export interface TuningChange {
  id: string;
  ts: number;
  kind: TuningChangeKind;
  gate?: TuningGateKey;
  label: string;
  oldValue?: number;
  newValue?: number;
  pctChange?: number;
  source: "auto-suggest" | "auto-tighten" | "manual" | "env";
  reverted: boolean;
  revertedAt: number | null;
  /** Populated for auto-tighten-toggle entries: the new effective enabled state. */
  enabled?: boolean;
}

export interface TuningGateState {
  key: TuningGateKey;
  label: string;
  baseline: number;
  current: number;
  minFloor: number;
  unit: "ratio" | "pct" | "multiple";
  pctFromBaseline: number;
  canLoosenMore: boolean;
  belowBaselineSince: number | null;
}

export interface AutoApplyTightenStatus {
  enabled: boolean;
  envDefault: boolean;
  override: boolean | null;
  source: "override" | "env";
}

export interface TuningPendingTighten {
  gate: TuningGateKey;
  ticks: number;
}

export interface TuningState {
  gates: TuningGateState[];
  history: TuningChange[];
  suggestion: TuningSuggestion | null;
  autoApplyTighten: AutoApplyTightenStatus;
  pendingTighten: TuningPendingTighten | null;
  autoApplyTightenTicks: number;
  autoApplyTightenEnabled: boolean;
}

export interface SkipReasonsSummary {
  windowMs: number;
  generatedAt: string;
  totalSkips: number;
  byReason: {
    reason: string;
    label: string;
    count: number;
    byAgent: { agentName: string; count: number }[];
    recent: SkipEvent[];
  }[];
  suggestion: TuningSuggestion | null;
}

export function useSkipReasons() {
  return useQuery<SkipReasonsSummary>({
    queryKey: ["crypto-skip-reasons"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/skip-reasons?hours=24`);
      if (!res.ok) throw new Error("Failed to fetch skip reasons");
      return res.json();
    },
    refetchInterval: 30000,
  });
}

export function useTuningState() {
  return useQuery<TuningState>({
    queryKey: ["crypto-tuning"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/tuning`);
      if (!res.ok) throw new Error("Failed to fetch tuning state");
      return res.json();
    },
    refetchInterval: 30000,
  });
}

export interface SkipTimelineBucket {
  ts: number;
  total: number;
  byReason: Record<string, number>;
  spikeReasons: string[];
}

export interface SkipSpike {
  ts: number;
  reason: string;
  label: string;
  count: number;
  mean: number;
  stdDev: number;
  zScore: number;
}

export interface SkipTimeline {
  windowMs: number;
  bucketMs: number;
  generatedAt: string;
  totalSkips: number;
  reasons: { reason: string; label: string; total: number }[];
  buckets: SkipTimelineBucket[];
  spikes: SkipSpike[];
  spikeThreshold: { zScore: number; minCount: number };
}

export function useSkipTimeline(hours: number) {
  return useQuery<SkipTimeline>({
    queryKey: ["crypto-skip-timeline", hours],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/skip-timeline?hours=${hours}`);
      if (!res.ok) throw new Error("Failed to fetch skip timeline");
      return res.json();
    },
    refetchInterval: 60000,
  });
}

export interface SkipEventsResponse {
  reason: string;
  bucketTs: number;
  bucketMs: number;
  count: number;
  events: SkipEvent[];
}

export function useSkipEventsForBucket(
  params: { reason: string; bucketTs: number; bucketMs: number } | null,
) {
  return useQuery<SkipEventsResponse>({
    queryKey: ["crypto-skip-events", params?.reason, params?.bucketTs, params?.bucketMs],
    queryFn: async () => {
      if (!params) throw new Error("missing params");
      const url = `${apiBase}/crypto/skip-events?reason=${encodeURIComponent(params.reason)}&bucketTs=${params.bucketTs}&bucketMs=${params.bucketMs}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch skip events");
      return res.json();
    },
    enabled: params !== null,
  });
}

export function useAutoDeployAttributionHistory() {
  return useQuery<AutoDeployAttributionHistory>({
    queryKey: ["crypto-auto-deploy-attribution-history"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/auto-deploy-attribution-history?hours=168`);
      if (!res.ok) throw new Error("Failed to fetch auto-deploy attribution history");
      return res.json();
    },
    refetchInterval: 60000,
  });
}

export function useRealityCheck() {
  return useQuery<RealityCheck>({
    queryKey: ["crypto-reality-check"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/reality-check`);
      if (!res.ok) throw new Error("Failed to fetch reality check");
      return res.json();
    },
    refetchInterval: 30000,
  });
}

// ────────────────────────── Task #512 ──────────────────────────
// Family-fleet, drill-down, archived-agents, and dashboard-activity
// hooks. All four endpoints follow the existing raw-fetch pattern
// (no openapi.yaml entry) so the build stays decoupled from spec
// regeneration. The dashboard reworks the legacy "Bots Leaderboard"
// around these hooks; the legacy `usePaperPortfolios` hook is kept
// because the QuantFleetCard and Strategy-Lab benchmark panel still
// read it (now with `kind` / `profileId` populated by the API).

export interface FamilyCard {
  profileId: string;
  displayName: string;
  thesis: string;
  strategyFamily: string;
  preferredRegimes: string[] | "all";
  blockedRegimes: string[];
  memberAgentIds: number[];
  memberCount: number;
  equity: number;
  startingCapital: number;
  peakValue: number;
  realizedPnl: number;
  realizedPnlPct: number;
  maxDrawdown: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  openPositions: number;
  winRate: number;
  costAwareDirectionalAccuracy: number | null;
  costAwareSharpe: number | null;
  abstainRate: number;
  abstainCount: number;
  trustMultiplier: number;
  statusPill: "active" | "cautious" | "suppressed" | "quarantined";
  retirement: {
    profile_id: string;
    directional_accuracy: number | null;
    cost_aware_sharpe: number | null;
    auto_flipped: boolean;
    triggered_by?: string[];
    notes?: string;
  } | null;
}

export function useFamilies() {
  return useQuery<{ families: FamilyCard[] }>({
    queryKey: ["crypto-agents-families"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/agents/families`);
      if (!res.ok) throw new Error("Failed to fetch executor families");
      return res.json();
    },
    refetchInterval: 30_000,
  });
}

export interface FamilyDecision {
  id: number;
  agentId: number;
  agentName: string;
  coinId: string;
  coinName: string;
  direction: string;
  confidence: number;
  outcome: string;
  createdAt: string;
  timeframe: string;
}

export interface FamilyTrade {
  id: number;
  agentId: number;
  agentName: string;
  coinId: string;
  coinName: string;
  action: string;
  entryPrice: number;
  exitPrice: number | null;
  pnl: number | null;
  pnlPercent: number | null;
  status: string;
  createdAt: string;
}

export interface FamilyCoinRow {
  coinId: string;
  coinName: string;
  openPositions: number;
  openNotional: number;
  unrealizedPnl: number;
  closedTrades: number;
  winningTrades: number;
  realizedPnl: number;
  drawdown: number;
  predictionCount: number;
  correctPredictions: number;
  recentAccuracy: number;
  fallbackUsage: number;
  suppressionState: "active" | "suppressed";
  trustMultiplier: number;
  // benchmark-relative behavior — buy-and-hold from the family's first
  // entry on the coin to live price vs. its actual capital-weighted return
  firstEntryAt: string | null;
  firstEntryPrice: number;
  latestPrice: number;
  totalCostBasis: number;
  benchmarkBuyHoldPct: number | null;
  executorReturnPct: number | null;
  vsBenchmarkPct: number | null;
  benchmarkRelative: "outperforming" | "tracking" | "underperforming" | "no_data";
  recentDecisions: FamilyDecision[];
  recentTrades: FamilyTrade[];
}

export interface FamilyCoinsResponse {
  profileId: string;
  strategyFamily: string;
  trustMultiplier: number;
  coins: FamilyCoinRow[];
}

export function useFamilyCoins(profileId: string | null) {
  return useQuery<FamilyCoinsResponse>({
    queryKey: ["crypto-agents-family-coins", profileId],
    enabled: typeof profileId === "string" && profileId.length > 0,
    queryFn: async () => {
      const res = await fetch(
        `${apiBase}/crypto/agents/families/${encodeURIComponent(profileId!)}/coins`,
      );
      if (!res.ok) throw new Error("Failed to fetch family coin breakdown");
      return res.json();
    },
    refetchInterval: 30_000,
  });
}

export interface ArchivedAgentRow {
  id: number;
  name: string;
  legacyType: string;
  personality: string;
  profileId: string | null;
  isActive: boolean;
  archivedAt: string | null;
  archivedOn: string | null;
  score: number;
  totalPredictions: number;
  correctPredictions: number;
  wrongPredictions: number;
  createdAt: string | null;
  lastActiveAt: string | null;
  lifetimePnl: number;
  lifetimePnlPct: number;
  maxDrawdown: number;
  tradeCount: number;
  winningTrades: number;
  losingTrades: number;
}

export function useArchivedAgents() {
  return useQuery<{ archived: ArchivedAgentRow[] }>({
    queryKey: ["crypto-agents-archived"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/agents/archived`);
      if (!res.ok) throw new Error("Failed to fetch archived agents");
      return res.json();
    },
    refetchInterval: 60_000,
  });
}

export interface DashboardActivity {
  // Task #518 — split executor vs baseline so the dashboard banner
  // does not conflate Strategy-Lab passive rebalances with quant
  // brain executor decisions. v1 fields (`tradesLastHour`,
  // `lastTradeAt`) are still emitted for back-compat.
  executorTradesLastHour: number;
  baselineTradesLastHour: number;
  lastExecutorTradeAt: string | null;
  lastBaselineTradeAt: string | null;
  // Task #532 / C-1b — most recent prediction whose `reasoning` is
  // not a quant abstain. Lets the activity strip surface the gap
  // between "last executor trade" and "last *valid quant decision*".
  lastValidQuantDecisionAt?: string | null;
  tradesLastHour: number;
  lastTradeAt: string | null;
}

export function useDashboardActivity() {
  return useQuery<DashboardActivity>({
    queryKey: ["crypto-dashboard-activity"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/dashboard-activity`);
      if (!res.ok) throw new Error("Failed to fetch dashboard activity");
      return res.json();
    },
    refetchInterval: 30_000,
  });
}

// Task #518 — brain runtime status. Distinct from the
// `/brain/state` operator kill-switch view: this is the
// dashboard-facing status pill ("online" / "offline_no_model" /
// "offline_disabled") plus the recent-abstain rollup so the
// dashboard banner can show *why* the brain is dark.
export type BrainRuntimeState = "online" | "offline_no_model" | "offline_disabled";

// Task #686 — promotion-gate retry roll-up for the dashboard chip.
// Mirrors `PromotionGateRetryStats` in
// `artifacts/api-server/src/lib/brain-promotion-gate.ts`. Shape is
// optional on the wire so an older api-server (one that hasn't shipped
// #686 yet) still satisfies the type without a contract bump.
export interface PromotionGateRetries {
  count: number;
  windowMs: number;
  mostRecentAt: string | null;
  mostRecentReason: string | null;
  mostRecentAttempt: number | null;
}

export interface BrainRuntimeStatus {
  state: BrainRuntimeState;
  brainEnabled: boolean;
  brainSource: "default" | "manual" | "auto_revert" | "env";
  mlAvailabilitySnapshotReady: boolean;
  recentAbstainReasons: Record<string, number>;
  recentNonAbstainCount: number;
  lastSuccessfulDecisionAt: string | null;
  currentRunDir: string | null;
  windowMinutes: number;
  promotionGateRetries?: PromotionGateRetries;
}

export function useBrainRuntimeStatus() {
  return useQuery<BrainRuntimeStatus>({
    queryKey: ["crypto-brain-runtime-status"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/brain/runtime-status`);
      if (!res.ok) throw new Error(`brain/runtime-status ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}

// ────────────────────────── Task #554 ──────────────────────────
// Dataset-refresher health for the dashboard. Backed by
// `/crypto/datasets/freshness`, which reads
// `artifacts/ml-engine/models/datasets/_freshness_status.json` and
// tails `_freshness_alerts.jsonl`. Used by `<DatasetFreshnessBanner/>`
// (sticky banner when a tf is past-due) and `<DatasetFreshnessCard/>`
// (per-tf pills + expandable alert log) on the dashboard.
export type DatasetFreshnessHealth = "green" | "amber" | "red" | "unknown";

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

export interface DatasetFreshnessAlert {
  at: string | null;
  timeframe: string | null;
  status: string | null;
  error: string | null;
  cadenceHours: number | null;
  unread: boolean;
  raw: string;
}

export interface DatasetFreshnessStatus {
  state: DatasetFreshnessHealth;
  statusFileExists: boolean;
  alertsFileExists: boolean;
  statusReadError: string | null;
  writtenAt: string | null;
  timeframes: DatasetFreshnessTimeframe[];
  pastDueTimeframes: string[];
  alerts: DatasetFreshnessAlert[];
  totalAlerts: number;
  totalUnreadAlerts: number;
  fetchedAt: string;
}

export function useDatasetFreshness() {
  return useQuery<DatasetFreshnessStatus>({
    queryKey: ["crypto-datasets-freshness"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/datasets/freshness`);
      if (!res.ok) throw new Error(`datasets/freshness ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

// ────────────────────────── Task #599 ──────────────────────────
// 1h/2h calibration ON/OFF verdict. The dataset-refresher
// auto-runs `task592_parallel_stage2` after every successful 1h
// or 2h snapshot; the API tails the result + the most recent
// `*-task592-1h2h-stage2-verdict.md` from `artifacts/ml-engine/reports/`.
// Powers `<ShortTfCalibrationVerdictCard/>` on the dashboard so an
// operator can see at a glance whether ON or OFF is currently the
// better calibration for the short timeframes.
export type ShortTfCalibrationVerdictState =
  | "ok"
  | "error"
  | "timeout"
  | "unknown";

export interface ShortTfCalibrationStageSummary {
  nSlices: number | null;
  nPassingGate: number | null;
  nInTradeShareBand: number | null;
  meanTradeShare: number | null;
  meanDaLift: number | null;
  sumPnlPctTotalAug: number | null;
}

export interface ShortTfCalibrationVerdictSummary {
  capturedAt: string | null;
  roundTripCostPct: number | null;
  timeframesSubset: string[] | null;
  wallTimeSeconds: number | null;
  nWorkers: number | null;
  nWorkUnits: number | null;
  off: ShortTfCalibrationStageSummary | null;
  on: ShortTfCalibrationStageSummary | null;
}

export interface ShortTfCalibrationVerdictBlock {
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
  summary: ShortTfCalibrationVerdictSummary | null;
}

export interface ShortTfCalibrationVerdictResponse {
  state: ShortTfCalibrationVerdictState;
  statusFileExists: boolean;
  statusReadError: string | null;
  shortTf: ShortTfCalibrationVerdictBlock | null;
  markdownPath: string | null;
  jsonPath: string | null;
  markdownTail: string | null;
  markdownReadError: string | null;
  fetchedAt: string;
}

export function useShortTfCalibrationVerdict() {
  return useQuery<ShortTfCalibrationVerdictResponse>({
    queryKey: ["crypto-calibration-verdict-short-tf"],
    queryFn: async () => {
      const res = await fetch(
        `${apiBase}/crypto/calibration-verdict/short-tf`,
      );
      if (!res.ok) {
        throw new Error(`calibration-verdict/short-tf ${res.status}`);
      }
      return res.json();
    },
    // The verdict only updates when a 1h/2h refresh lands, so a 60s
    // poll is plenty (matches the dataset-freshness card cadence).
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}
