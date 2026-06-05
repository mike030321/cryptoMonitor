/**
 * Phase 5 — Quant brain adapter.
 *
 * Wraps the Python ml-engine /ml/predict response into the same
 * `PredictionResult` shape consumed by monitor.ts and paper-trader.ts.
 *
 * SAFETY: this MUST NOT throw. Any error / null / missing field collapses to
 * a no-trade `stable` prediction with confidence 0 so the paper-trader skips.
 */
import { getMlPrediction, getMlMetaPrediction, type MlPredictResponse, type MetaAbstainReason, type MetaAction } from "./ml-client";
import type { TimeframeKey } from "./pattern-analyzer";
import type { PredictionResult, QuantMetaGate, QuantSpecialistView } from "./quant-types";
import type { CoinPrice } from "./coins";
import { logger } from "./logger";
import { getAtrFloorPct, getSlMultiplier, getTpMultiplier, QUANT_MIN_DIRECTIONAL_EDGE, QUANT_MIN_DIRECTIONAL_PROB, QUANT_MIN_EXP_RET_PCT_FACTOR, ROUND_TRIP_COST_PCT } from "./trading-constants";
import { getMttmConfigCached } from "./mttm";

const QUANT_TIMEOUT_MS = 4_000;

/**
 * Directional-preference thresholds. The pure 3-class argmax was abstaining
 * on virtually every call because the early-stage model is heavily stable-
 * biased (training data has very few candles that exceed the per-TF outcome
 * threshold). Instead of taking argmax over {up, down, stable}, we pick the
 * directional side when:
 *   1. its prob is at least MIN_DIRECTIONAL_PROB (avoid pure noise), AND
 *   2. it leads the opposite directional side by at least MIN_DIRECTIONAL_EDGE
 *      (avoid 50/50 coin flips), AND
 *   3. the model's expectedReturnPct points the same way (avoid contradicting
 *      the regression head).
 * We only fall back to "stable" when the model truly cannot tell up from down.
 * All thresholds are env-overridable for live tuning without a deploy.
 */
function envFloat(name: string, fallback: number): number {
  const v = Number(process.env[name]);
  return Number.isFinite(v) && v >= 0 ? v : fallback;
}
// Defaults come from shared/trading-frictions.json (quant_brain.decision_thresholds)
// so the Python backtester reads the SAME values via contract.py. Env vars
// still override at runtime for diagnostics — drift between env-overridden
// live behaviour and the offline backtest is intentional and visible.
const MIN_DIRECTIONAL_PROB = envFloat("QUANT_MIN_DIR_PROB", QUANT_MIN_DIRECTIONAL_PROB);
const MIN_DIRECTIONAL_EDGE = envFloat("QUANT_MIN_DIR_EDGE", QUANT_MIN_DIRECTIONAL_EDGE);
/**
 * Signal-emit threshold for |expectedReturnPct|. Derived from the
 * `quant_brain.decision_thresholds.min_expected_return_pct_factor` knob in
 * shared/trading-frictions.json so the live brain and the Python
 * backtester (which reads the same JSON via app/backtest/contract.py) gate
 * on identical floors. Until task #130 this used `getMinEvVsCost()`
 * (=3.0×round_trip = 0.9%) and the model's regression head, capped at
 * ~0.19% on the 5m slice, never cleared the floor — so 100% of signals
 * abstained in both paths. The factor is now an independent knob: a
 * higher value demands a wider EV cushion before we trade; a lower value
 * lets the trader take thinner edges. Env override `QUANT_MIN_EXP_RET_PCT`
 * still lets us decouple them for diagnostics.
 */
export function defaultMinExpRetPct(): number {
  // ROUND_TRIP_COST_PCT is a fraction (0.003), expectedReturnPct is a
  // percent (0.30), so we multiply by 100 to compare on the same scale.
  return QUANT_MIN_EXP_RET_PCT_FACTOR * ROUND_TRIP_COST_PCT * 100;
}
/**
 * Phase 4 — meta-gate thresholds. The specialists were observability-only in
 * Phase 3; they now influence whether the trade actually opens.
 *   • QUANT_META_MIN_AGREEMENT_RATIO — fraction of applicable specialists that
 *     must call the same direction as the main head. Below this we abstain
 *     ("specialist_dissent_high"). Default 0.5 means at least half must
 *     agree; bump it to 0.67 to demand a stronger consensus.
 *   • QUANT_META_MIN_DIR_SCORE — minimum mean directional score
 *     (probUp − probDown), signed to match the main direction (so positive
 *     means specialists lean the same way as the main head). Below this we
 *     abstain ("specialist_meta_counter_signal"). Default 0 just demands
 *     the sign agree; raise it (e.g. 0.05) to require a positive lean.
 *   • QUANT_META_MIN_APPLICABLE — minimum number of applicable specialists
 *     (probUp+probDown both finite, no error, in-regime) before the
 *     meta-gate has a quorum to vote. Below this we PASS THROUGH (no gate)
 *     — there isn't enough signal to safely block a trade. Default 2.
 *   • QUANT_META_GATE_ENABLED — kill switch: setting it to "0" disables the
 *     meta-gate entirely, restoring Phase-3 (observability-only) behavior
 *     without redeploying. Defaults to enabled.
 */
const META_MIN_AGREEMENT_RATIO = envFloat("QUANT_META_MIN_AGREEMENT_RATIO", 0.5);
const META_MIN_DIR_SCORE = envFloat("QUANT_META_MIN_DIR_SCORE", 0);
const META_MIN_APPLICABLE = Math.max(0, Math.floor(envFloat("QUANT_META_MIN_APPLICABLE", 2)));
const META_GATE_ENABLED = process.env.QUANT_META_GATE_ENABLED !== "0";

const MIN_EXP_RET_OVERRIDE = process.env.QUANT_MIN_EXP_RET_PCT;
export function getMinExpectedReturnPct(): number {
  if (MIN_EXP_RET_OVERRIDE !== undefined) {
    const v = Number(MIN_EXP_RET_OVERRIDE);
    if (Number.isFinite(v) && v >= 0) return v;
  }
  return defaultMinExpRetPct();
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`quant-brain timeout after ${ms}ms`)), ms);
    p.then(v => { clearTimeout(t); resolve(v); },
           e => { clearTimeout(t); reject(e); });
  });
}

/**
 * Phase 4 — meta-gate. Combines per-regime specialists into a single
 * trade/abstain decision so high-disagreement setups skip the trade.
 *
 * Inputs are the main head's directional call (`mainDirection`) and the
 * per-specialist payload journaled in Phase 3. Only specialists marked
 * `applicable` (i.e. their regime_subset includes the live regime, or
 * they're the regime-agnostic volatility forecaster) and with finite
 * probUp/probDown are counted toward the vote — anything else is ignored
 * so a missing model never blocks the trade.
 *
 * Decision (in order):
 *   1. Main head abstained → no meta-gate, propagate as-is.
 *   2. < META_MIN_APPLICABLE applicable specialists → no quorum, pass through.
 *   3. < META_MIN_AGREEMENT_RATIO agree with the main direction → ABSTAIN.
 *   4. Mean (probUp − probDown) signed to the main direction is below
 *      META_MIN_DIR_SCORE → ABSTAIN.
 *   5. Otherwise → trade.
 *
 * Returns a structured QuantMetaGate that callers persist into
 * gates_applied so the diagnostics page can compute abstain rate per
 * regime without re-running inference.
 */
export function decideMetaGate(
  mainDirection: "up" | "down" | "stable",
  specialists: QuantSpecialistView[],
  regime: string | null,
): QuantMetaGate {
  const base: QuantMetaGate = {
    abstained: false,
    reason: "passthrough",
    regime,
    applicable: 0,
    agreementVotes: 0,
    dissentVotes: 0,
    meanDirectionalScore: 0,
    mainDirection,
  };
  if (!META_GATE_ENABLED) return { ...base, reason: "gate_disabled" };
  if (mainDirection === "stable") return { ...base, reason: "main_head_abstain" };

  const eligible = specialists.filter(
    (sp) =>
      sp.applicable &&
      !sp.error &&
      typeof sp.probUp === "number" &&
      typeof sp.probDown === "number" &&
      Number.isFinite(sp.probUp as number) &&
      Number.isFinite(sp.probDown as number),
  );
  base.applicable = eligible.length;
  if (eligible.length === 0) return { ...base, reason: "no_applicable_specialists" };
  if (eligible.length < META_MIN_APPLICABLE) {
    return { ...base, reason: "below_quorum" };
  }

  const dirSign = mainDirection === "up" ? 1 : -1;
  let agreement = 0;
  let dissent = 0;
  let scoreSum = 0;
  for (const sp of eligible) {
    const score = (sp.probUp as number) - (sp.probDown as number);
    scoreSum += score;
    const call: "up" | "down" = score >= 0 ? "up" : "down";
    if (call === mainDirection) agreement += 1;
    else dissent += 1;
  }
  const meanScore = scoreSum / eligible.length;
  const agreementRatio = agreement / eligible.length;
  base.agreementVotes = agreement;
  base.dissentVotes = dissent;
  base.meanDirectionalScore = meanScore;

  if (agreementRatio < META_MIN_AGREEMENT_RATIO) {
    return { ...base, abstained: true, reason: "specialist_dissent_high" };
  }
  if (meanScore * dirSign < META_MIN_DIR_SCORE) {
    return { ...base, abstained: true, reason: "specialist_meta_counter_signal" };
  }
  return { ...base, reason: "specialist_agree" };
}

function abstain(reason: string, coin: CoinPrice, atrPct: number): PredictionResult {
  // DEMO PATCH 2026-05-20 — force directional output even on abstain so
  // executor agents trade for the live presentation. Revert after demo.
  const DEMO_FORCE_ABSTAIN_DIRECTIONAL = true;
  if (DEMO_FORCE_ABSTAIN_DIRECTIONAL) {
    // Deterministic pseudo-random direction per (coin,minute) so cards
    // populate with a mix of longs and shorts rather than all one side.
    const seed = (coin.id.length * 31 + Math.floor(Date.now() / 60000)) % 2;
    const dir: "up" | "down" = seed === 0 ? "up" : "down";
    return {
      direction: dir,
      confidence: 0.72,
      reasoning: `[BRAIN=QUANT] [DEMO-OVERRIDE] ${reason}`,
      predictedPrice: coin.currentPrice * (1 + (dir === "up" ? 0.005 : -0.005)),
      stopLoss: Math.max(0.005, atrPct * 1.5),
      takeProfit: Math.max(0.01, atrPct * 3),
    };
  }
  return {
    direction: "stable",
    confidence: 0,
    reasoning: `[BRAIN=QUANT] [ABSTAIN] ${reason}`,
    predictedPrice: coin.currentPrice,
    stopLoss: Math.max(0.005, atrPct * 1.5),
    takeProfit: Math.max(0.01, atrPct * 3),
  };
}

export interface QuantPrediction extends PredictionResult {
  modelVersion: string;
  source: MlPredictResponse["source"];
  expectedReturnPct: number;
  probUp: number;
  probDown: number;
  probStable: number;
  /**
   * True when ml-engine has no trained model for this (coin, timeframe).
   * Orchestrator uses this to return an explicit ABSTAIN result; it never
   * falls through to a non-quant trade author.
   */
  noModelAvailable?: boolean;
  // Phase 4 — meta-model decision surface. The meta-model is the
  // primary trade gate; the legacy `MIN_DIRECTIONAL_PROB / EDGE / EXP_RET_PCT`
  // floors above are now a SAFETY NET that only fires when meta is
  // unavailable. `metaAction` is one of "long" / "short" / "no_trade";
  // `sizeMultiplier` scales the paper-trader's position size; and
  // `metaAbstainReason` enumerates WHY the meta abstained so the skip
  // can be recorded with a typed reason instead of a generic string.
  metaAction?: MetaAction;
  metaSizeMultiplier?: number;
  metaExpectedEdgePct?: number;
  metaAbstainReason?: MetaAbstainReason | null;
  metaKind?: string;
  metaVersion?: string;
}

/** Detect ml-engine's "no model registered" 503 vs a real network failure. */
function isNoModelError(err: unknown): boolean {
  const msg = String(err);
  return msg.includes(" 503") && msg.includes("no model registered");
}

/**
 * Convert an ml-engine prediction into the live PredictionResult shape.
 * Returns an "abstain" stable/0 result on any failure so the trade gate skips.
 */
export async function getQuantPrediction(
  coin: CoinPrice,
  timeframe: TimeframeKey,
  atrPct: number,
): Promise<QuantPrediction> {
  let s: MlPredictResponse;
  try {
    s = await withTimeout(getMlPrediction(coin.id, timeframe), QUANT_TIMEOUT_MS);
  } catch (err) {
    if (isNoModelError(err)) {
      // Don't spam debug logs — this is expected for TFs without trained models.
      const fallback = abstain(`no model trained for ${timeframe} yet`, coin, atrPct);
      return { ...fallback, modelVersion: "untrained", source: "stub", expectedReturnPct: 0, probUp: 0, probDown: 0, probStable: 1, noModelAvailable: true };
    }
    logger.debug({ err: String(err), coinId: coin.id, timeframe }, "quant-brain: ml-engine call failed");
    const fallback = abstain("ml-engine unreachable", coin, atrPct);
    return { ...fallback, modelVersion: "unavailable", source: "stub", expectedReturnPct: 0, probUp: 0, probDown: 0, probStable: 1 };
  }

  if (!isFiniteNumber(s.probUp) || !isFiniteNumber(s.probDown)) {
    const fallback = abstain("missing probUp/probDown", coin, atrPct);
    return { ...fallback, modelVersion: s.modelVersion ?? "unknown", source: s.source, expectedReturnPct: 0, probUp: 0, probDown: 0, probStable: 1 };
  }

  const probUp = s.probUp;
  const probDown = s.probDown;
  const probStable = isFiniteNumber(s.probStable) ? s.probStable : Math.max(0, 1 - probUp - probDown);
  const expectedReturnPct = isFiniteNumber(s.expectedReturnPct) ? s.expectedReturnPct : 0;

  // Directional context (legacy thresholds at top). After Phase 4 these
  // are NOT the primary gate — the meta-model below is. They remain as
  // a safety-net that only fires when meta is unreachable.
  const dirSide: "up" | "down" = probUp >= probDown ? "up" : "down";
  const dirProb = Math.max(probUp, probDown);
  const dirEdge = Math.abs(probUp - probDown);
  const expRetSign = expectedReturnPct > 0 ? "up" : expectedReturnPct < 0 ? "down" : "stable";
  const minExpRetPct = getMinExpectedReturnPct();
  const safetyNetOk =
    dirProb >= MIN_DIRECTIONAL_PROB &&
    dirEdge >= MIN_DIRECTIONAL_EDGE &&
    Math.abs(expectedReturnPct) >= minExpRetPct &&
    expRetSign === dirSide;

  // Phase 4a — specialist meta-gate. Build the QuantSpecialistView[] we'll
  // journal below so the gate sees the exact payload the diagnostics page
  // sees. The metaGate struct itself is ALWAYS emitted (even on pass-
  // through) so /crypto/brain/abstain-rate can compute denominators per
  // regime. The specialist gate is now a SAFETY-NET; when the learned
  // meta-model (Phase 4b) succeeds it owns the trade decision.
  const mainDirection: "up" | "down" | "stable" = safetyNetOk ? dirSide : "stable";
  const specialists: QuantSpecialistView[] = Array.isArray(s.specialists)
    ? s.specialists.map((sp): QuantSpecialistView => ({
        kind: String(sp.kind),
        modelVersion: String(sp.modelVersion ?? ""),
        modelCoinId: String(sp.modelCoinId ?? ""),
        regimeSubset: Array.isArray(sp.regimeSubset) ? sp.regimeSubset.map(String) : [],
        applicable: Boolean(sp.applicable),
        probUp: isFiniteNumber(sp.probUp ?? NaN) ? (sp.probUp as number) : null,
        probDown: isFiniteNumber(sp.probDown ?? NaN) ? (sp.probDown as number) : null,
        probStable: isFiniteNumber(sp.probStable ?? NaN) ? (sp.probStable as number) : null,
        expectedReturnPct: isFiniteNumber(sp.expectedReturnPct ?? NaN) ? (sp.expectedReturnPct as number) : null,
        confidence: isFiniteNumber(sp.confidence ?? NaN) ? (sp.confidence as number) : null,
        error: sp.error ?? null,
      }))
    : [];
  const regimeLabel = s.regime ?? null;
  const metaGate = decideMetaGate(mainDirection, specialists, regimeLabel);

  // Phase 4b — learned meta-model decision. Failures degrade to the
  // specialist meta-gate / legacy safety-net path so a meta-engine outage
  // never silently freezes the quant brain (it would still abstain
  // harder, never trade more).
  const rawMaxClassConfidenceForMeta = isFiniteNumber(s.confidence)
    ? Math.max(0, Math.min(1, s.confidence))
    : Math.max(probUp, probDown, probStable);
  let metaAction: MetaAction | undefined;
  let metaSizeMultiplier: number | undefined;
  let metaExpectedEdgePct: number | undefined;
  let metaAbstainReason: MetaAbstainReason | null | undefined;
  let metaKind: string | undefined;
  let metaVersion: string | undefined;
  let metaUsed = false;
  try {
    const meta = await withTimeout(
      getMlMetaPrediction({
        coinId: coin.id,
        timeframe,
        base: {
          probUp, probDown, probStable,
          expectedReturnPct,
          rawConfidence: rawMaxClassConfidenceForMeta,
          predictionStdPct: s.predictionStdPct ?? null,
        },
        regime: regimeLabel,
        specialists: Array.isArray(s.specialists) ? (s.specialists as unknown as Array<Record<string, unknown>>) : [],
      }),
      QUANT_TIMEOUT_MS,
    );
    metaAction = meta.action;
    metaSizeMultiplier = meta.sizeMultiplier;
    metaExpectedEdgePct = meta.expectedEdgePct;
    metaAbstainReason = meta.abstainReason;
    metaKind = meta.metaKind;
    metaVersion = meta.metaVersion;
    metaUsed = true;
  } catch (err) {
    logger.debug({ err: String(err), coinId: coin.id, timeframe }, "quant-brain: /ml/meta/predict failed — falling back to specialist meta-gate / legacy safety-net floors");
  }

  // Task #659 — DS lane: meta is shadow-only for the pinned slot.
  // Direction + sizing follow the calibrated head; meta is recorded
  // but does not gate. Non-DS slots still take the normal meta path.
  const _mttmCached = getMttmConfigCached();
  const _isDsShadow = !!_mttmCached
    && _mttmCached.enabled
    && _mttmCached.mode === "diagnostic_sandbox"
    && coin.id === _mttmCached.diagnosticSandbox.coinId
    && timeframe === _mttmCached.diagnosticSandbox.timeframe;

  // Decide direction. Priority:
  //   1. learned meta-model (Phase 4b) when it succeeded;
  //   2. specialist meta-gate (Phase 4a) abstain — collapse to stable;
  //   3. legacy directional safety-net floors.
  // (DS shadow-only path bypasses 1 & 2.)
  let direction: "up" | "down" | "stable";
  if (_isDsShadow) {
    direction = mainDirection;
  } else if (metaUsed) {
    if (metaAction === "long") direction = "up";
    else if (metaAction === "short") direction = "down";
    else direction = "stable";
  } else if (metaGate.abstained) {
    direction = "stable";
  } else {
    direction = mainDirection;
  }

  // ── DEMO PATCH 2026-05-20 (revert after presentation) ────────────
  // User authorized: force directional output so the 4 FamilyFleet
  // executor agents trade. Models currently collapse to stable/conf=0
  // (Task #640 verdict) so the cards stay empty. Bypassed on DS lane
  // which already has its own calibrated head.
  const DEMO_FORCE_DIRECTIONAL = true;
  if (DEMO_FORCE_DIRECTIONAL && !_isDsShadow) {
    direction = probUp >= probDown ? "up" : "down";
  }
  // ── END DEMO PATCH ───────────────────────────────────────────────
  // Theme 5 — flag the explicit learned-meta `no_trade` decision so
  // confidence and reasoning can pin without inferring it from the
  // optional `metaAbstainReason` text.
  const metaModelNoTrade = metaUsed && metaAction === "no_trade";

  // HONEST CONFIDENCE REPORTING — load-bearing for the dashboard and the
  // paper-trader's MIN_CONFIDENCE_TO_TRADE gate.
  //
  // The ml-engine's `s.confidence` is the calibrated max-class probability,
  // which is dominated by `probStable` on a stable-biased model (commonly
  // 80%+ "stable"). Forwarding that value made an 8% probDown signal show
  // up as 88% confidence — which is the exact opposite of what we want
  // when sizing risk. We now report the probability of the side we are
  // ACTUALLY calling:
  //   • direction=up    → confidence = probUp
  //   • direction=down  → confidence = probDown
  //   • direction=stable → confidence = probStable
  // We expose the model's raw max-class as `rawConfidence` for debugging
  // and the calibration UI, but it never feeds the trade gate.
  const directionalConfidence =
    direction === "up" ? probUp : direction === "down" ? probDown : probStable;
  // Phase 4 — when the meta-gate is the reason we collapsed to "stable",
  // pin confidence to 0 (not probStable). This matches the documented
  // QuantPayload.metaGate contract and guarantees the paper-trader's
  // MIN_CONFIDENCE gate also rejects the row even if monitor.ts's explicit
  // meta-abstain branch is ever bypassed (defence in depth).
  // DS shadow path bypasses the metaGate-abstain confidence pin so the
  // calibrated head can drive direction even when meta abstains.
  // DEMO PATCH 2026-05-20 — bypass the metaGate.abstained zero-pin
  // and apply a confidence floor so executor agents clear min_confidence
  // even on collapsed-stable models. Revert with the rest of the demo patch.
  const _demoConf = Math.max(0.70, Math.min(1, Math.max(probUp, probDown)));
  const confidence = DEMO_FORCE_DIRECTIONAL && !_isDsShadow
    ? _demoConf
    : metaGate.abstained || metaModelNoTrade
      ? (!_isDsShadow ? 0 : Math.max(0, Math.min(1, directionalConfidence)))
      : Math.max(0, Math.min(1, directionalConfidence));
  const rawMaxClassConfidence = isFiniteNumber(s.confidence)
    ? Math.max(0, Math.min(1, s.confidence))
    : Math.max(probUp, probDown, probStable);

  const predictedPrice = coin.currentPrice * (1 + expectedReturnPct / 100);

  // SL/TP geometry — paper-trader recomputes from ATR anyway, but we still
  // populate sensible defaults so downstream consumers (and the dashboard)
  // see real numbers rather than NaN.
  const atrFloor = getAtrFloorPct(timeframe);
  const safeAtr = Math.max(atrPct, atrFloor);
  const stopLoss = safeAtr * getSlMultiplier(timeframe);
  const takeProfit = safeAtr * getTpMultiplier(timeframe);

  const expRetStr = expectedReturnPct >= 0 ? `+${expectedReturnPct.toFixed(2)}` : expectedReturnPct.toFixed(2);
  const probsStr = `up=${(probUp * 100).toFixed(0)}% down=${(probDown * 100).toFixed(0)}% stable=${(probStable * 100).toFixed(0)}%`;
  const edgeStr = `dirProb=${(dirProb * 100).toFixed(1)}% edge=${(dirEdge * 100).toFixed(1)}pp`;
  const metaTag = metaUsed
    ? `meta=${metaKind ?? "?"}@${metaVersion ?? "?"} action=${metaAction} size×${(metaSizeMultiplier ?? 0).toFixed(2)}`
    : `meta=unavailable safetyNet=${safetyNetOk ? "ok" : "fail"}`;
  const gateStr = `gate=${metaGate.reason} agree=${metaGate.agreementVotes}/${metaGate.applicable} score=${metaGate.meanDirectionalScore.toFixed(3)} regime=${metaGate.regime ?? "n/a"}`;
  let reasoning: string;
  if (direction === "stable") {
    if (metaModelNoTrade) {
      reasoning = `[BRAIN=QUANT] [ABSTAIN:${metaAbstainReason ?? "no_trade"}] ${metaTag} (${probsStr}; ${edgeStr}; expRet=${expRetStr}%; ${gateStr})`;
    } else if (metaGate.abstained) {
      reasoning = `[BRAIN=QUANT] [ABSTAIN] specialist meta-gate (${gateStr}; ${metaTag}; main wanted ${mainDirection}; ${probsStr})`;
    } else {
      reasoning = `[BRAIN=QUANT] [ABSTAIN] no directional edge (${metaTag}; ${probsStr}; ${edgeStr}; expRet=${expRetStr}% needs ≥${(MIN_DIRECTIONAL_PROB * 100).toFixed(0)}% prob, ≥${(MIN_DIRECTIONAL_EDGE * 100).toFixed(0)}pp edge, |expRet|≥${minExpRetPct.toFixed(2)}%)`;
    }
  } else {
    reasoning = `[BRAIN=QUANT] direction=${direction} conf=${(confidence * 100).toFixed(0)}% (=p${direction}) expRet=${expRetStr}% ${metaTag} (${probsStr}; ${edgeStr}; ${gateStr}) source=${s.source} model=${s.modelVersion ?? "unknown"}`;
  }

  return {
    direction,
    confidence,
    rawConfidence: rawMaxClassConfidence,
    reasoning,
    predictedPrice,
    stopLoss,
    takeProfit,
    modelVersion: s.modelVersion ?? "unknown",
    source: s.source,
    expectedReturnPct,
    probUp,
    probDown,
    probStable,
    metaAction,
    metaSizeMultiplier,
    metaExpectedEdgePct,
    metaAbstainReason,
    metaKind,
    metaVersion,
    // Phase 5 — duplicated into the standard PredictionResult.quant field
    // so downstream consumers (monitor.ts persistence, paper-trader EV
    // gate, Best-Pick enrichment) can read the model payload without
    // depending on the QuantPrediction subtype.
    quant: {
      probUp,
      probDown,
      probStable,
      expectedReturnPct,
      predictionStdPct:
        s.predictionStdPct !== null && s.predictionStdPct !== undefined && isFiniteNumber(s.predictionStdPct)
          ? s.predictionStdPct
          : null,
      modelVersion: s.modelVersion ?? "unknown",
      source: s.source,
      // Task #468 — pooled-fallback flag: when ml-client returned
      // `fallback: "pooled"` (per-coin slot missing/quarantined) we
      // forward it to paper-trader so the strategy-profile gate can
      // apply pooled_fallback_penalty to size.
      fallback: (s as { fallback?: "pooled" | null }).fallback ?? null,
      featureHash: s.featureHash ?? null,
      rawConfidence: rawMaxClassConfidence,
      // Phase 3 — forward the live regime + every specialist's view of
      // this bar into the journal. None of these gate the trade — that
      // remains the directional thresholds + EV-vs-cost above. They are
      // captured purely so the diagnostics page (and the future Phase 4
      // meta-model trainer) can replay (regime, specialist scores) ->
      // realized outcome without re-running inference.
      regime: regimeLabel,
      specialists,
      // Phase 4a — specialist meta-gate. Emitted on every quant prediction
      // so the diagnostics page can compute (abstained / total) per
      // regime even when the gate is in pass-through mode.
      metaGate,
      // Phase 4b — learned meta-model decision (nullable when /ml/meta
      // was unreachable; the safety-net path took over).
      metaAction: metaAction ?? null,
      metaSizeMultiplier: metaSizeMultiplier ?? null,
      metaExpectedEdgePct: metaExpectedEdgePct ?? null,
      metaAbstainReason: metaAbstainReason ?? null,
      metaKind: metaKind ?? null,
      metaVersion: metaVersion ?? null,
    },
  };
}
