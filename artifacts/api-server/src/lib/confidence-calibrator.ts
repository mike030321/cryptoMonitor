import { db, predictionsTable } from "@workspace/db";
import { eq, sql, and, desc, gte } from "drizzle-orm";
import { PREDICTION_FLEET_RESET_AT } from "./trading-constants";

export interface CalibrationBucket {
  bucketMin: number;
  bucketMax: number;
  totalPredictions: number;
  correctPredictions: number;
  actualWinRate: number;
}

export interface AgentCalibration {
  agentId: number;
  buckets: CalibrationBucket[];
  overallBias: number;
  calibrationQuality: number;
}

const BUCKET_COUNT = 10;
const MIN_BUCKET_SAMPLES = 3;

const calibrationCache = new Map<number, { data: AgentCalibration; expiresAt: number }>();
const CACHE_TTL = 5 * 60 * 1000;

export async function getAgentCalibration(agentId: number): Promise<AgentCalibration> {
  const cached = calibrationCache.get(agentId);
  if (cached && Date.now() < cached.expiresAt) return cached.data;

  const resolved = await db
    .select({
      confidence: predictionsTable.confidence,
      rawConfidence: predictionsTable.rawConfidence,
      outcome: predictionsTable.outcome,
    })
    .from(predictionsTable)
    .where(and(
      eq(predictionsTable.agentId, agentId),
      sql`${predictionsTable.outcome} IN ('correct', 'wrong')`,
      // Defensive scoping: only use predictions from the post-reset run.
      // See PREDICTION_FLEET_RESET_AT for why pre-reset rows are poison.
      gte(predictionsTable.createdAt, PREDICTION_FLEET_RESET_AT)
    ))
    .orderBy(desc(predictionsTable.createdAt))
    .limit(300);

  const buckets: CalibrationBucket[] = [];
  for (let i = 0; i < BUCKET_COUNT; i++) {
    const bucketMin = i / BUCKET_COUNT;
    const bucketMax = (i + 1) / BUCKET_COUNT;
    const inBucket = resolved.filter(p => {
      const conf = p.rawConfidence ?? p.confidence;
      return conf >= bucketMin && conf < bucketMax;
    });
    const correct = inBucket.filter(p => p.outcome === "correct").length;
    buckets.push({
      bucketMin,
      bucketMax,
      totalPredictions: inBucket.length,
      correctPredictions: correct,
      actualWinRate: inBucket.length >= MIN_BUCKET_SAMPLES ? correct / inBucket.length : (bucketMin + bucketMax) / 2,
    });
  }

  let totalWeightedError = 0;
  let totalSamples = 0;
  let biasSum = 0;
  let biasCount = 0;

  for (const bucket of buckets) {
    if (bucket.totalPredictions >= MIN_BUCKET_SAMPLES) {
      const midpoint = (bucket.bucketMin + bucket.bucketMax) / 2;
      const error = Math.abs(bucket.actualWinRate - midpoint);
      totalWeightedError += error * bucket.totalPredictions;
      totalSamples += bucket.totalPredictions;
      biasSum += (bucket.actualWinRate - midpoint) * bucket.totalPredictions;
      biasCount += bucket.totalPredictions;
    }
  }

  const calibrationQuality = totalSamples > 0
    ? Math.max(0, 1 - (totalWeightedError / totalSamples) * 2)
    : 0.5;

  const overallBias = biasCount > 0 ? biasSum / biasCount : 0;

  const result: AgentCalibration = { agentId, buckets, overallBias, calibrationQuality };
  calibrationCache.set(agentId, { data: result, expiresAt: Date.now() + CACHE_TTL });
  return result;
}

export function applyCalibration(rawConfidence: number, calibration: AgentCalibration): number {
  const bucketIndex = Math.min(BUCKET_COUNT - 1, Math.floor(rawConfidence * BUCKET_COUNT));
  const bucket = calibration.buckets[bucketIndex];

  const totalSamples = calibration.buckets.reduce((sum, b) => sum + b.totalPredictions, 0);

  if (bucket.totalPredictions < MIN_BUCKET_SAMPLES) {
    if (totalSamples >= 30) {
      // overallBias = mean(actualWinRate - statedMidpoint).
      // Positive bias  => actual > stated => agent is UNDERconfident => raise stated confidence.
      // Negative bias  => actual < stated => agent is OVERconfident  => lower stated confidence.
      // Therefore we ADD the bias (scaled), not subtract it. Subtracting (the prior bug) made
      // an underconfident agent even MORE underconfident, killing legitimate trade signals.
      const adjustedByBias = rawConfidence + calibration.overallBias * 0.15;
      return Math.min(0.95, Math.max(0.25, adjustedByBias));
    }
    return Math.min(0.95, Math.max(0.25, rawConfidence));
  }

  const blendFactor = Math.min(0.4, bucket.totalPredictions / 30);
  const calibrated = rawConfidence * (1 - blendFactor) + bucket.actualWinRate * blendFactor;

  return Math.min(0.95, Math.max(0.25, calibrated));
}

export function formatCalibrationContext(calibration: AgentCalibration): string {
  if (calibration.buckets.every(b => b.totalPredictions < MIN_BUCKET_SAMPLES)) return "";

  const lines: string[] = ["=== CONFIDENCE CALIBRATION ==="];
  lines.push(`Calibration Quality: ${(calibration.calibrationQuality * 100).toFixed(0)}%`);

  // overallBias = mean(actualWinRate - statedMidpoint). Positive => actual exceeds stated
  // (UNDERconfident). Negative => actual falls short of stated (OVERconfident). The previous
  // wording had the labels reversed and was telling overconfident agents to be MORE confident.
  if (calibration.overallBias > 0.05) {
    lines.push(`NOTE: You tend to be UNDERCONFIDENT by ~${(calibration.overallBias * 100).toFixed(0)}%. You can be slightly more confident.`);
  } else if (calibration.overallBias < -0.05) {
    lines.push(`WARNING: You tend to be OVERCONFIDENT by ~${(Math.abs(calibration.overallBias) * 100).toFixed(0)}%. Reduce your stated confidence.`);
  }

  const significantBuckets = calibration.buckets.filter(b => b.totalPredictions >= MIN_BUCKET_SAMPLES);
  if (significantBuckets.length > 0) {
    lines.push("Confidence vs Reality:");
    for (const b of significantBuckets) {
      const statedRange = `${(b.bucketMin * 100).toFixed(0)}-${(b.bucketMax * 100).toFixed(0)}%`;
      const actual = `${(b.actualWinRate * 100).toFixed(0)}%`;
      const diff = b.actualWinRate - (b.bucketMin + b.bucketMax) / 2;
      const indicator = Math.abs(diff) < 0.05 ? "✓" : diff > 0 ? "↑" : "↓";
      lines.push(`  ${statedRange} stated → ${actual} actual ${indicator} (${b.totalPredictions} samples)`);
    }
  }

  return lines.join("\n");
}

export function invalidateCalibrationCache(agentId?: number): void {
  if (agentId !== undefined) {
    calibrationCache.delete(agentId);
  } else {
    calibrationCache.clear();
  }
}
