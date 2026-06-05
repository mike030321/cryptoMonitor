/**
 * Phase 6 — Drift trackers.
 *
 * Three orthogonal kinds, all derived from `prediction_journal`:
 *
 *   • calibration            — bucketed expected-calibration-error: bin
 *                              predictions by max(probUp, probDown,
 *                              probStable), compute |confidence - empirical
 *                              accuracy| weighted by bucket count.
 *   • prediction_distribution — KL divergence between the recent-window
 *                              direction histogram and a baseline window.
 *   • feature                — mean / std drift of each numeric feature
 *                              column in `feature_vector`, summarised as
 *                              the max absolute z-score across columns.
 *
 * All three return a single scalar `score` (higher = more drift) plus a
 * structured `detail` payload so the UI can render the breakdown without
 * re-aggregating.
 *
 * `persistDriftSnapshot` writes one row to `drift_snapshots` so the
 * auto-quarantine routine and the dashboard sparkline have a stable
 * historical baseline without re-aggregating the journal on every
 * refresh.
 */
import {
  db,
  predictionJournalTable,
  driftSnapshotsTable,
  type DriftKind,
} from "@workspace/db";
import { and, gte, sql, desc } from "drizzle-orm";

export interface CalibrationBucket {
  bucket: number;          // 0..9 for deciles
  meanConfidence: number;
  empiricalAccuracy: number;
  count: number;
}

export interface CalibrationDriftResult {
  score: number;           // ECE
  threshold: number;
  breached: boolean;
  nSamples: number;
  buckets: CalibrationBucket[];
}

export interface DistributionDriftResult {
  score: number;           // KL divergence (recent || baseline)
  threshold: number;
  breached: boolean;
  nSamples: number;
  recent: Record<string, number>;     // direction -> share
  baseline: Record<string, number>;
}

export interface FeatureDriftResult {
  score: number;           // max absolute z-score across columns
  threshold: number;
  breached: boolean;
  nSamples: number;
  perFeature: { name: string; recentMean: number; baselineMean: number; recentStd: number; baselineStd: number; zScore: number }[];
}

export const DRIFT_THRESHOLDS = {
  calibration: 0.20,                // ECE > 20% = breach
  prediction_distribution: 0.50,   // KL > 0.5 nats
  feature: 3.0,                     // |z| > 3
} as const;

const MIN_SAMPLES = 50;

function bucketize(p: number): number {
  if (!Number.isFinite(p)) return 0;
  const b = Math.min(9, Math.max(0, Math.floor(p * 10)));
  return b;
}

export async function computeCalibrationDrift(opts: {
  windowHours?: number;
  registryId?: number | null;
}): Promise<CalibrationDriftResult> {
  const windowHours = opts.windowHours ?? 168;
  const since = new Date(Date.now() - windowHours * 3600 * 1000);
  const conds = [
    gte(predictionJournalTable.createdAt, since),
    sql`${predictionJournalTable.outcome} IS NOT NULL AND ${predictionJournalTable.outcome} <> 'pending'`,
  ];
  if (opts.registryId != null) {
    conds.push(sql`${predictionJournalTable.registryId} = ${opts.registryId}`);
  }
  const rows = await db
    .select({
      probUp: predictionJournalTable.probUp,
      probDown: predictionJournalTable.probDown,
      probStable: predictionJournalTable.probStable,
      direction: predictionJournalTable.direction,
      outcome: predictionJournalTable.outcome,
    })
    .from(predictionJournalTable)
    .where(and(...conds))
    .limit(20000);

  const buckets: { sumConf: number; sumCorrect: number; n: number }[] =
    Array.from({ length: 10 }, () => ({ sumConf: 0, sumCorrect: 0, n: 0 }));
  let total = 0;
  for (const r of rows) {
    const probs = [r.probUp ?? 0, r.probDown ?? 0, r.probStable ?? 0];
    const conf = Math.max(...probs);
    if (!Number.isFinite(conf) || conf <= 0) continue;
    const correct = r.outcome === "correct" ? 1 : 0;
    const b = bucketize(conf);
    buckets[b].sumConf += conf;
    buckets[b].sumCorrect += correct;
    buckets[b].n += 1;
    total += 1;
  }
  let ece = 0;
  const out: CalibrationBucket[] = [];
  for (let i = 0; i < 10; i++) {
    const b = buckets[i];
    if (b.n === 0) {
      out.push({ bucket: i, meanConfidence: 0, empiricalAccuracy: 0, count: 0 });
      continue;
    }
    const meanConf = b.sumConf / b.n;
    const acc = b.sumCorrect / b.n;
    ece += (b.n / total) * Math.abs(meanConf - acc);
    out.push({
      bucket: i,
      meanConfidence: Number(meanConf.toFixed(4)),
      empiricalAccuracy: Number(acc.toFixed(4)),
      count: b.n,
    });
  }
  const score = total > 0 ? Number(ece.toFixed(4)) : 0;
  return {
    score,
    threshold: DRIFT_THRESHOLDS.calibration,
    breached: total >= MIN_SAMPLES && score > DRIFT_THRESHOLDS.calibration,
    nSamples: total,
    buckets: out,
  };
}

export async function computeDistributionDrift(opts: {
  windowHours?: number;
  registryId?: number | null;
}): Promise<DistributionDriftResult> {
  const windowHours = opts.windowHours ?? 168;
  const baselineHours = windowHours * 4;
  const recentSince = new Date(Date.now() - windowHours * 3600 * 1000);
  const baselineSince = new Date(Date.now() - baselineHours * 3600 * 1000);

  const conds = [gte(predictionJournalTable.createdAt, baselineSince)];
  if (opts.registryId != null) {
    conds.push(sql`${predictionJournalTable.registryId} = ${opts.registryId}`);
  }
  const rows = await db
    .select({
      direction: predictionJournalTable.direction,
      createdAt: predictionJournalTable.createdAt,
    })
    .from(predictionJournalTable)
    .where(and(...conds))
    .limit(50000);

  const recent: Record<string, number> = { up: 0, down: 0, stable: 0 };
  const baseline: Record<string, number> = { up: 0, down: 0, stable: 0 };
  for (const r of rows) {
    const d = r.direction;
    if (!(d in recent)) continue;
    if (r.createdAt >= recentSince) {
      recent[d] += 1;
    } else {
      baseline[d] += 1;
    }
  }
  const recentN = Object.values(recent).reduce((a, b) => a + b, 0);
  const baselineN = Object.values(baseline).reduce((a, b) => a + b, 0);
  const recentP: Record<string, number> = {};
  const baselineP: Record<string, number> = {};
  // Laplace smoothing so KL is finite even if a class is empty.
  const eps = 0.01;
  for (const k of Object.keys(recent)) {
    recentP[k] = recentN > 0 ? (recent[k] + eps) / (recentN + 3 * eps) : 1 / 3;
    baselineP[k] = baselineN > 0 ? (baseline[k] + eps) / (baselineN + 3 * eps) : 1 / 3;
  }
  let kl = 0;
  for (const k of Object.keys(recentP)) {
    kl += recentP[k] * Math.log(recentP[k] / baselineP[k]);
  }
  const nSamples = recentN;
  return {
    score: Number(kl.toFixed(4)),
    threshold: DRIFT_THRESHOLDS.prediction_distribution,
    breached:
      nSamples >= MIN_SAMPLES && kl > DRIFT_THRESHOLDS.prediction_distribution,
    nSamples,
    recent: Object.fromEntries(
      Object.entries(recentP).map(([k, v]) => [k, Number(v.toFixed(4))]),
    ),
    baseline: Object.fromEntries(
      Object.entries(baselineP).map(([k, v]) => [k, Number(v.toFixed(4))]),
    ),
  };
}

const NUMERIC_FEATURE_KEYS = [
  "ret1", "ret5", "ret10", "momentum", "realizedVol",
  "rsi14", "macdLine", "macdSignal", "macdHist",
  "atr14", "atrPct", "ema9", "ema21", "emaSpreadPct",
  "distFromEma9Pct", "distFromEma21Pct",
  "bbWidth", "bbPctB", "bbWidthPct",
];

export async function computeFeatureDrift(opts: {
  windowHours?: number;
  registryId?: number | null;
}): Promise<FeatureDriftResult> {
  const windowHours = opts.windowHours ?? 168;
  const baselineHours = windowHours * 4;
  const recentSince = new Date(Date.now() - windowHours * 3600 * 1000);
  const baselineSince = new Date(Date.now() - baselineHours * 3600 * 1000);

  const conds = [
    gte(predictionJournalTable.createdAt, baselineSince),
    sql`${predictionJournalTable.featureVector} IS NOT NULL`,
  ];
  if (opts.registryId != null) {
    conds.push(sql`${predictionJournalTable.registryId} = ${opts.registryId}`);
  }
  const rows = await db
    .select({
      fv: predictionJournalTable.featureVector,
      createdAt: predictionJournalTable.createdAt,
    })
    .from(predictionJournalTable)
    .where(and(...conds))
    .limit(20000);

  const recentSums = new Map<string, { sum: number; sumSq: number; n: number }>();
  const baselineSums = new Map<string, { sum: number; sumSq: number; n: number }>();
  function add(target: Map<string, { sum: number; sumSq: number; n: number }>, k: string, v: number) {
    const acc = target.get(k) ?? { sum: 0, sumSq: 0, n: 0 };
    acc.sum += v;
    acc.sumSq += v * v;
    acc.n += 1;
    target.set(k, acc);
  }
  let recentN = 0;
  for (const r of rows) {
    const fv = r.fv as Record<string, unknown> | null;
    if (!fv) continue;
    const target = r.createdAt >= recentSince ? recentSums : baselineSums;
    if (target === recentSums) recentN += 1;
    for (const k of NUMERIC_FEATURE_KEYS) {
      const v = fv[k];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      add(target, k, v);
    }
  }
  const perFeature: FeatureDriftResult["perFeature"] = [];
  let maxZ = 0;
  for (const k of NUMERIC_FEATURE_KEYS) {
    const r = recentSums.get(k);
    const b = baselineSums.get(k);
    if (!r || !b || r.n < 5 || b.n < 5) continue;
    const rMean = r.sum / r.n;
    const bMean = b.sum / b.n;
    const rVar = Math.max(0, r.sumSq / r.n - rMean * rMean);
    const bVar = Math.max(0, b.sumSq / b.n - bMean * bMean);
    const rStd = Math.sqrt(rVar);
    const bStd = Math.sqrt(bVar);
    const pooledStd = Math.sqrt((rVar + bVar) / 2) || 1e-9;
    const z = (rMean - bMean) / pooledStd;
    const zAbs = Math.abs(z);
    if (zAbs > maxZ) maxZ = zAbs;
    perFeature.push({
      name: k,
      recentMean: Number(rMean.toFixed(6)),
      baselineMean: Number(bMean.toFixed(6)),
      recentStd: Number(rStd.toFixed(6)),
      baselineStd: Number(bStd.toFixed(6)),
      zScore: Number(z.toFixed(4)),
    });
  }
  perFeature.sort((a, b) => Math.abs(b.zScore) - Math.abs(a.zScore));
  return {
    score: Number(maxZ.toFixed(4)),
    threshold: DRIFT_THRESHOLDS.feature,
    breached: recentN >= MIN_SAMPLES && maxZ > DRIFT_THRESHOLDS.feature,
    nSamples: recentN,
    perFeature,
  };
}

export async function persistDriftSnapshot(args: {
  registryId?: number | null;
  coinId?: string;
  timeframe?: string;
  kind: DriftKind;
  nSamples: number;
  score: number;
  threshold: number;
  breached: boolean;
  detail?: unknown;
}): Promise<void> {
  await db.insert(driftSnapshotsTable).values({
    registryId: args.registryId ?? null,
    coinId: args.coinId ?? "*",
    timeframe: args.timeframe ?? "*",
    kind: args.kind,
    nSamples: args.nSamples,
    score: args.score,
    threshold: args.threshold,
    breached: args.breached,
    detail: (args.detail as object | undefined) ?? null,
  });
}

export async function getRecentDriftSnapshots(
  kind: DriftKind,
  limit = 200,
): Promise<unknown[]> {
  const rows = await db
    .select()
    .from(driftSnapshotsTable)
    .where(sql`${driftSnapshotsTable.kind} = ${kind}`)
    .orderBy(desc(driftSnapshotsTable.createdAt))
    .limit(limit);
  return rows;
}
