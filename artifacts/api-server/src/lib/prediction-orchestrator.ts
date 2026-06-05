/**
 * Brain orchestrator (Task #444 deterministic-quant edition).
 *
 * The quant brain (`/ml/predict` + `/ml/meta/predict`) is the SOLE authority
 * for live trade decisions. There is NO LLM execution path: when the quant
 * brain has no model for a (coin, timeframe) — or quant is globally
 * disabled — this function ABSTAINS (returns `brain: "ABSTAIN"` with
 * `direction: "stable"` and `confidence: 0`). The caller MUST treat
 * ABSTAIN as "do not open a new position".
 *
 * The `__setLlmOverride` test seam below is preserved as a no-op so the
 * existing parity tests can keep installing a throwing override and
 * asserting it is never invoked. There is no production code path that
 * could call it: the LLM module it used to point at (`_legacy/ai-engine`)
 * was deleted as part of #444.
 */
import type { PredictionResult } from "./quant-types";
import { getQuantPrediction, type QuantPrediction } from "./quant-brain";
import {
  isQuantBrainEnabled,
  isDiagnosticSandboxEnabled,
} from "./brain-flag";
import {
  MTTM_DIAGNOSTIC_SANDBOX_COIN,
  MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
} from "./mttm";
import { isModelAvailable } from "./ml-availability";
import type { CoinPrice } from "./coins";
import type { TimeframeKey, PatternAnalysis } from "./pattern-analyzer";

interface OrchestratorArgs {
  coin: CoinPrice;
  timeframe: TimeframeKey;
  patternAnalysis: PatternAnalysis;
}

export interface OrchestratorResult extends PredictionResult {
  brain: "QUANT" | "ABSTAIN";
  /** Populated when brain === "QUANT" — features the explainer can surface. */
  quantMeta?: {
    modelVersion: string;
    source: QuantPrediction["source"];
    expectedReturnPct: number;
    probUp: number;
    probDown: number;
    probStable: number;
  };
  /** Populated when brain === "ABSTAIN" — short reason code for the journal. */
  abstainReason?: "no_model" | "quant_disabled";
}

const cycleCache = new Map<string, OrchestratorResult>();
const cycleNoModelKeys = new Set<string>();

export function resetCycleCache(): void {
  cycleCache.clear();
  cycleNoModelKeys.clear();
}

export function getCycleCacheSize(): number {
  return cycleCache.size;
}

// ── Test seams (NODE_ENV=test only) ──────────────────────────────────────
// `LlmFn` used to be `typeof getAgentPrediction`. After Task #444 deleted
// the LLM brain module, the seam stays as a no-op so the parity tests
// (which install a throwing override and assert it is never invoked) keep
// passing without modification. Production code never reads `llmOverride`.
type LlmFn = (...args: unknown[]) => Promise<unknown>;
type QuantFn = typeof getQuantPrediction;
let llmOverride: LlmFn | null = null;
let quantOverride: QuantFn | null = null;
export function __setLlmOverride(fn: LlmFn | null): void { llmOverride = fn; }
export function __setQuantOverride(fn: QuantFn | null): void { quantOverride = fn; }
export function __getLlmOverride(): LlmFn | null { return llmOverride; }

function tagReasoning(brain: "QUANT" | "ABSTAIN", reasoning: string): string {
  if (reasoning.includes(`[BRAIN=${brain}]`)) return reasoning;
  return `[BRAIN=${brain}] ${reasoning}`;
}

function abstainResult(reason: "no_model" | "quant_disabled"): OrchestratorResult {
  const r: OrchestratorResult = {
    direction: "stable",
    confidence: 0,
    predictedPrice: 0,
    stopLoss: 0,
    takeProfit: 0,
    reasoning: tagReasoning("ABSTAIN", `quant brain has no decision (${reason})`),
    brain: "ABSTAIN",
    abstainReason: reason,
  };
  return r;
}

export async function getPrediction(args: OrchestratorArgs): Promise<OrchestratorResult> {
  const cacheKey = `${args.coin.id}:${args.timeframe}`;
  const hit = cycleCache.get(cacheKey);
  if (hit) return hit;

  // Task #659 (C-BTC) — quant lane reachability follows the strict
  // contract: `(isQuantBrainEnabled() || isDiagnosticSandboxEnabled())`.
  // The DS lane is the per-lane diagnostic exception, it MUST be
  // reachable while the global flag stays off (otherwise the
  // beta-calibrated paper sandbox is dead code).
  //
  // Per-slot filtering is enforced HERE: when the production flag
  // is off but DS is on, only the pinned `(bitcoin, 5m)` slot is
  // allowed through — every other slot abstains as `quant_disabled`
  // exactly as before #659. This preserves the hard rule that DS
  // does NOT widen quant access for any other coin or timeframe.
  let quant: boolean;
  if (process.env.QUANT_BRAIN_TEST_FORCE_ON === "1") {
    quant = true;
  } else if (await isQuantBrainEnabled()) {
    quant = true;
  } else if (await isDiagnosticSandboxEnabled()) {
    // DS lane is on globally, but only the pinned slot may execute.
    quant = (
      args.coin.id === MTTM_DIAGNOSTIC_SANDBOX_COIN
      && args.timeframe === MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME
    );
  } else {
    quant = false;
  }

  if (quant) {
    if (!cycleNoModelKeys.has(cacheKey) && !isModelAvailable(args.coin.id, args.timeframe)) {
      cycleNoModelKeys.add(cacheKey);
    }
    const skipQuant = cycleNoModelKeys.has(cacheKey);
    if (!skipQuant) {
      const atrPct = args.patternAnalysis.atr / Math.max(args.coin.currentPrice, 1e-9);
      const quantFn = quantOverride ?? getQuantPrediction;
      const q = await quantFn(args.coin, args.timeframe, atrPct);

      if (q.noModelAvailable) {
        cycleNoModelKeys.add(cacheKey);
      } else {
        const result: OrchestratorResult = {
          ...q,
          reasoning: tagReasoning("QUANT", q.reasoning),
          brain: "QUANT",
          quantMeta: {
            modelVersion: q.modelVersion,
            source: q.source,
            expectedReturnPct: q.expectedReturnPct,
            probUp: q.probUp,
            probDown: q.probDown,
            probStable: q.probStable,
          },
        };
        cycleCache.set(cacheKey, result);
        return result;
      }
    }

    // Quant said "no model" → ABSTAIN. The orchestrator NEVER falls through
    // to anything else: Task #444 (and #255 before it) forbids LLM-authored
    // trade decisions, and the LLM brain module is gone.
    const result = abstainResult("no_model");
    cycleCache.set(cacheKey, result);
    return result;
  }

  // Quant globally disabled → ABSTAIN. Same rationale as above.
  const result = abstainResult("quant_disabled");
  cycleCache.set(cacheKey, result);
  return result;
}

/**
 * Test-only invariant guard. Returns false unconditionally — the orchestrator
 * has no LLM execution path. Kept as an exported function so tests and
 * monitoring can assert the invariant cheaply.
 */
export function __isLlmExecutionAllowed(): boolean {
  return false;
}
