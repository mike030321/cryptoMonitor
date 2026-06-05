/**
 * Quant prediction recorder (originally Phase 4 shadow recorder).
 *
 * Calls the Python ml-engine /ml/predict endpoint on every analysis cycle
 * and persists the result to `model_predictions` for live calibration
 * tracking on the Quant Live Health page. NEVER affects the trading path:
 * callers must treat all errors as "no row recorded" and continue.
 *
 * Includes a tiny in-memory circuit breaker so a flapping ml-engine cannot
 * starve the live trading loop with timeouts.
 *
 * Task #255 / Quant Live Health repurpose: the legacy `llmDirection` /
 * `llmConfidence` arguments were dropped — the LLM no longer authors
 * trade-level directions, so there is nothing to compare against. The
 * residual `llmPredictionId` FK back to `predictions` was dropped in
 * Task #506 alongside the underlying columns.
 */
import { db, modelPredictionsTable } from "@workspace/db";
import { eq, and, lte } from "drizzle-orm";
import { logger } from "./logger";
import { getMlPrediction } from "./ml-client";
import { TIMEFRAMES, type TimeframeKey } from "./pattern-analyzer";
import { judgeDirection, RESOLVE_GRACE_PERIOD_MS } from "./trading-constants";
import { fetchCoinPrices, isPriceDataFresh } from "./coins";

const SHADOW_TIMEOUT_MS = 4_000;
const CB_FAILURE_THRESHOLD = 5;
const CB_OPEN_MS = 60_000;

let consecutiveFailures = 0;
let openUntil = 0;

function shadowCircuitOpen(): boolean {
  return Date.now() < openUntil;
}

function recordFailure(): void {
  consecutiveFailures++;
  if (consecutiveFailures >= CB_FAILURE_THRESHOLD) {
    openUntil = Date.now() + CB_OPEN_MS;
    logger.warn({ failures: consecutiveFailures, openMs: CB_OPEN_MS },
      "shadow-recorder circuit breaker OPEN — pausing ml-engine calls");
    consecutiveFailures = 0;
  }
}
function recordSuccess(): void { consecutiveFailures = 0; }

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`shadow-recorder timeout after ${ms}ms`)), ms);
    p.then(v => { clearTimeout(t); resolve(v); },
           e => { clearTimeout(t); reject(e); });
  });
}

/** Fire the ml call WITHOUT awaiting — caller awaits the returned promise
 * concurrently with the LLM call. Resolves to null on any error/timeout/
 * circuit-open; never throws. */
export function fetchShadowPrediction(coinId: string, timeframe: TimeframeKey) {
  if (shadowCircuitOpen()) return Promise.resolve(null);
  return withTimeout(getMlPrediction(coinId, timeframe), SHADOW_TIMEOUT_MS)
    .then((r) => { recordSuccess(); return r; })
    .catch((err) => {
      recordFailure();
      logger.debug({ err: String(err), coinId, timeframe }, "shadow ml-engine call failed");
      return null;
    });
}

interface RecordArgs {
  shadow: Awaited<ReturnType<typeof getMlPrediction>> | null;
  coinId: string;
  coinName: string;
  timeframe: TimeframeKey;
  priceAtPrediction: number;
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export async function recordShadowPrediction(args: RecordArgs): Promise<void> {
  const s = args.shadow;
  if (!s) return;
  // Wire-boundary validation — refuse to insert garbage. The MlPredictResponse
  // type already declares the optional fields; we still verify probabilities
  // are finite numbers because the ml-engine could ship a buggy build.
  if (!isFiniteNumber(s.probUp) || !isFiniteNumber(s.probDown)) {
    logger.warn({ coinId: args.coinId, timeframe: args.timeframe },
      "shadow prediction missing probUp/probDown — skipping insert");
    return;
  }
  const probUp = s.probUp;
  const probDown = s.probDown;
  const probStable = isFiniteNumber(s.probStable)
    ? s.probStable
    : Math.max(0, 1 - probUp - probDown);
  const max = Math.max(probUp, probDown, probStable);
  const modelDirection: "up" | "down" | "stable" =
    max === probUp ? "up" : max === probDown ? "down" : "stable";
  const expectedReturnPct = isFiniteNumber(s.expectedReturnPct) ? s.expectedReturnPct : 0;
  const predictionStdPct = isFiniteNumber(s.predictionStdPct) ? s.predictionStdPct : null;
  const confidence = isFiniteNumber(s.confidence) ? s.confidence : max;
  const tfMs = TIMEFRAMES[args.timeframe]?.ms ?? 300_000;

  try {
    await db.insert(modelPredictionsTable).values({
      coinId: args.coinId,
      coinName: args.coinName,
      timeframe: args.timeframe,
      modelVersion: s.modelVersion ?? "unknown",
      modelCoinId: s.modelCoinId ?? args.coinId,
      featureHash: s.featureHash ?? null,
      probUp, probDown, probStable,
      expectedReturnPct,
      predictionStdPct,
      confidence,
      modelDirection,
      priceAtPrediction: args.priceAtPrediction,
      resolvesAt: new Date(Date.now() + tfMs),
      // Tag the ml-engine provenance so shadow / accuracy dashboards can
      // exclude the prior-only pooled fallback from headline metrics.
      source: s.source,
    });
  } catch (err) {
    logger.warn({ err, coinId: args.coinId, timeframe: args.timeframe },
      "Failed to insert shadow prediction row (non-fatal)");
  }
}

/** Resolution mirror of resolvePendingPredictions — uses the SAME shared
 * `judgeDirection` adjudication (neutral-zone band) and the SAME 5-minute
 * grace-period two-pass second-chance logic so shadow outcomes are directly
 * comparable to live LLM outcomes. Grading drift here would invalidate the
 * cutover gate, so this MUST stay in lock-step with monitor.ts. */
export async function resolveShadowPredictions(): Promise<number> {
  const now = new Date();
  const due = await db
    .select()
    .from(modelPredictionsTable)
    .where(and(
      eq(modelPredictionsTable.outcome, "pending"),
      lte(modelPredictionsTable.resolvesAt, now),
    ));
  if (due.length === 0) return 0;

  const prices = await fetchCoinPrices(true);
  if (prices.length === 0 || !isPriceDataFresh()) {
    logger.warn("shadow resolve: stale prices, deferring");
    return 0;
  }

  let resolved = 0;
  for (const row of due) {
    const cur = prices.find(p => p.id === row.coinId);
    if (!cur) continue;
    const actualPrice = cur.currentPrice;
    const priceChange = ((actualPrice - row.priceAtPrediction) / row.priceAtPrediction) * 100;

    const resolvesAtTime = new Date(row.resolvesAt).getTime();
    const since = now.getTime() - resolvesAtTime;
    const withinGrace = since < RESOLVE_GRACE_PERIOD_MS;
    const pastGrace = since >= RESOLVE_GRACE_PERIOD_MS;

    const t0 = judgeDirection(row.modelDirection as "up"|"down"|"stable", priceChange, row.timeframe);
    let isCorrect = t0.correct;
    let isNeutral = t0.neutral;

    // First miss inside grace → snapshot t0 price, defer until grace expires.
    if (!isCorrect && withinGrace) {
      if (row.graceT0Price == null) {
        await db.update(modelPredictionsTable).set({
          graceT0Price: actualPrice,
          graceT0PriceChange: priceChange,
        }).where(eq(modelPredictionsTable.id, row.id));
      }
      continue;
    }

    // Second-chance look at the snapshotted t0 + the post-grace price.
    if (pastGrace && row.graceT0Price != null && !isCorrect) {
      const t0Snap = judgeDirection(row.modelDirection as "up"|"down"|"stable", row.graceT0PriceChange ?? 0, row.timeframe);
      if (t0Snap.correct) { isCorrect = true; isNeutral = false; }
      else if (t0Snap.neutral && !isNeutral) { isNeutral = true; }

      if (!isCorrect) {
        const reCheck = judgeDirection(row.modelDirection as "up"|"down"|"stable", priceChange, row.timeframe);
        if (reCheck.correct) { isCorrect = true; isNeutral = false; }
        else if (reCheck.neutral && !isNeutral) { isNeutral = true; }
      }
    }

    const outcome: "correct" | "wrong" | "neutral" =
      isCorrect ? "correct" : isNeutral ? "neutral" : "wrong";

    await db.update(modelPredictionsTable).set({
      actualPrice, resolvedOutcomePct: priceChange, outcome, resolvedAt: now,
    }).where(and(
      eq(modelPredictionsTable.id, row.id),
      eq(modelPredictionsTable.outcome, "pending"),
    ));
    resolved++;
  }
  if (resolved > 0) logger.info({ resolved }, "Resolved shadow predictions");
  return resolved;
}
