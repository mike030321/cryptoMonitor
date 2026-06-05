/**
 * Quant brain types — the deterministic shapes that flow between
 * `quant-brain.ts` (adapter on /ml/predict + /ml/meta/predict),
 * `prediction-orchestrator.ts` (live-trade gate), `journal-writer.ts`
 * (storage), `monitor.ts` (cycle driver), and `paper-trader.ts`
 * (execution gate).
 *
 * Carved out of the (now-deleted) `_legacy/ai-engine.ts` as part of
 * Task #444 so no live module imports from `_legacy/`. Anything LLM-,
 * news-, or personality-shaped is GONE; this module only contains the
 * quant payload contract and the deterministic best-pick prose
 * builders that survive into the slice-driven flow.
 */
export interface PredictionResult {
  direction: "up" | "down" | "stable";
  confidence: number;
  rawConfidence?: number;
  reasoning: string;
  predictedPrice: number;
  stopLoss: number;
  takeProfit: number;
  isNoTradeZone?: boolean;
  /**
   * The raw quant-brain payload travels alongside the legacy fields so
   * monitor.ts can persist it (in patternContext.quant) and paper-trader.ts
   * can run an EV-vs-cost gate without re-querying ml-engine.
   */
  quant?: QuantPayload;
}

/**
 * Strict type for the quant-brain payload duplicated onto PredictionResult.
 * Keep in sync with quant-brain.ts and journal-writer.ts — every field
 * here lands in `prediction_journal` so we can replay decisions and run
 * counterfactuals later.
 */
export interface QuantPayload {
  probUp: number;
  probDown: number;
  probStable: number;
  expectedReturnPct: number;
  predictionStdPct: number | null;
  modelVersion: string;
  source: string;
  /**
   * Task #468 — pooled-fallback flag forwarded from /ml/predict.
   * "pooled" iff the per-coin slot was missing/quarantined and the
   * pooled prior served the prediction; null otherwise. The trade
   * gate uses this to apply the agent profile's
   * pooled_fallback_penalty.
   */
  fallback?: "pooled" | null;
  featureHash: string | null;
  rawConfidence: number | null;
  /**
   * Live 6-class regime label classified from the same feature
   * vector used by the model. Stashed in the prediction journal's
   * gates_applied JSON (under `regime`) so per-regime backtests on the
   * journal don't have to re-run inference. `null` when /ml/predict
   * couldn't classify (prior-only fallback path).
   */
  regime?: string | null;
  /**
   * Per-regime specialist heads' view of this bar. Forwarded
   * verbatim from /ml/predict and stored on the journal in the dedicated
   * `specialist_scores` jsonb column so the diagnostics page can compute
   * per-specialist per-regime accuracy without re-querying the model.
   */
  specialists?: QuantSpecialistView[];
  /**
   * Specialist meta-gate (deterministic vote across the per-regime
   * specialists). Always emitted on quant rows so the diagnostics page
   * can compute abstain rate per regime without re-running inference.
   */
  metaGate?: QuantMetaGate;
  /**
   * Learned meta-model decision surface (/ml/meta/predict). When present,
   * this owns the trade gate; the legacy directional floors and the
   * specialist `metaGate` act as a safety-net only.
   */
  metaAction?: "long" | "short" | "no_trade" | null;
  metaSizeMultiplier?: number | null;
  metaExpectedEdgePct?: number | null;
  metaAbstainReason?: string | null;
  metaKind?: string | null;
  metaVersion?: string | null;
}

export interface QuantMetaGate {
  abstained: boolean;
  reason: string;
  regime: string | null;
  applicable: number;
  agreementVotes: number;
  dissentVotes: number;
  meanDirectionalScore: number;
  mainDirection: "up" | "down" | "stable";
}

/**
 * Single specialist's view as journaled. Mirrors `MlSpecialistPrediction`
 * in ml-client.ts but shaped for storage: lossy/null fields collapse to
 * `null` so the JSON column stays compact.
 */
export interface QuantSpecialistView {
  kind: string;
  modelVersion: string;
  modelCoinId: string;
  regimeSubset: string[];
  applicable: boolean;
  probUp: number | null;
  probDown: number | null;
  probStable: number | null;
  expectedReturnPct: number | null;
  confidence: number | null;
  error: string | null;
}

// ── Deterministic best-slice result ───────────────────────────────────
// Replaces the LLM/personality `BestPickResult`. The `agentConsensus`
// list is now slice-shaped (one entry per (coin, timeframe) slice that
// voted), and `newsFactors` / `newsTags` are GONE. Optional fields
// remain for back-compat with the dashboard payload.
export interface BestPickResult {
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
    agentCount: number;
  }[];
  /** Summarises the raw quant verdict (the "what"). */
  whatExplanation?: string;
  /** Deterministic "why" — same prose builder, no LLM. */
  whyExplanation?: string;
  /** Quant brain probability (probUp for buy, probDown for sell). 0-1. */
  modelProbability?: number;
  /** Expected return after fees+slippage, in percent. Negative => skip. */
  evAfterFeesPct?: number;
  /**
   * Brain identity — kept so the badge keeps rendering. After Task #444
   * removed the LLM path this is always "QUANT" or "ABSTAIN" but the
   * union shape stays back-compat with stored journal rows.
   */
  brain?: "QUANT" | "ABSTAIN";
  updatedAt: string;
}

// ── Best-pick prose builders ─────────────────────────────────────────
// Pure functions extracted so the truthfulness invariants (prose
// probability matches the tile; "consensus across 1 timeframes" never
// appears) can be pinned with unit tests without spinning up the DB.

export interface FallbackReasoningArgs {
  bestAction: "buy" | "sell" | "hold";
  coinName: string;
  timeframeCount: number;
  /** Label of the dominant aligned timeframe, e.g. "4h". Used when count === 1. */
  primaryTimeframe: string | null;
  /** LightGBM model probability for the chosen direction (0-1). null => omit. */
  modelProbability: number | null | undefined;
}

export function buildFallbackReasoning(args: FallbackReasoningArgs): string {
  const action = args.bestAction.toUpperCase();
  const head = `${action} ${args.coinName}`;
  let basis: string;
  if (args.timeframeCount >= 2) {
    basis = ` — consensus across ${args.timeframeCount} timeframes`;
  } else if (args.timeframeCount === 1 && args.primaryTimeframe) {
    basis = ` — based on the ${args.primaryTimeframe} timeframe`;
  } else if (args.timeframeCount === 1) {
    basis = ` — based on a single timeframe`;
  } else {
    basis = "";
  }
  const probSuffix = args.modelProbability != null
    ? ` with ${(args.modelProbability * 100).toFixed(0)}% model probability.`
    : ".";
  return `${head}${basis}${probSuffix}`;
}

export interface WhatExplanationArgs {
  bestAction: "buy" | "sell" | "hold";
  coinSymbol: string;
  brain: "QUANT" | "ABSTAIN" | undefined;
  modelProbability: number | null | undefined;
  holdTimeframe: string;
}

export function buildWhatExplanation(args: WhatExplanationArgs): string {
  const action = args.bestAction.toUpperCase();
  const brainLabel = args.brain ?? "QUANT";
  if (args.modelProbability != null) {
    return `${action} ${args.coinSymbol}: ${brainLabel} brain ${(args.modelProbability * 100).toFixed(0)}% over ${args.holdTimeframe}.`;
  }
  return `${action} ${args.coinSymbol}: ${brainLabel} brain over ${args.holdTimeframe}.`;
}
