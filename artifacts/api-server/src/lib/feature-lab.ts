/**
 * Phase 6 — Feature Lab.
 *
 * Manages candidate feature lifecycle: draft → ablated → approved.
 * The walk-forward ablation runner lives in the ml-engine — this lib
 * just orchestrates the call and persists the resulting report rows.
 *
 * Approval flow:
 *   1. Operator POSTs a candidate (name + transformKind + optional
 *      sourceColumn) — state = "draft".
 *   2. Operator triggers an ablation run; the ml-engine returns
 *      walk-forward metrics with vs without the candidate. Result is
 *      persisted to `feature_lab_reports`; candidate state transitions
 *      to "ablated".
 *   3. If the report's `delta_log_loss` (baseline - augmented) is
 *      positive AND `n_samples` ≥ MIN_SAMPLES, the "Promote feature"
 *      button is enabled. Approving writes the candidate spec to
 *      `app_settings.feature_lab.approved` so the next training run
 *      includes it, and transitions the candidate state to "approved".
 */
import {
  db,
  featureLabCandidatesTable,
  featureLabReportsTable,
  featureLabUnquarantineEventsTable,
  appSettingsTable,
  modelRegistryTable,
  type FeatureLabCandidateRow,
  type FeatureLabReportRow,
  type FeatureLabUnquarantineEventRow,
  FEATURE_TRANSFORM_KINDS,
  type FeatureTransformKind,
} from "@workspace/db";
import { and, eq, desc, gte, sql } from "drizzle-orm";
import { logger } from "./logger";

const ML_BASE = () => process.env.ML_ENGINE_URL || "http://localhost:8000";

export const APPROVED_FEATURES_SETTING_KEY = "feature_lab.approved";
export const QUARANTINED_FEATURES_SETTING_KEY = "feature_lab.quarantined";

export const MIN_ABLATION_SAMPLES = 200;

/**
 * Task #248 — when un-quarantining we re-check the validation regression
 * delta against the most recent training report. The same +0.05 log_loss
 * band that triggered auto-retire (see ml-engine `auto_retire.py`) gates
 * whether we still consider the regression "present". Operators can flip
 * the threshold via env var to match a future ml-engine change without a
 * code edit.
 */
export const UNQUARANTINE_REGRESSION_THRESHOLD = (() => {
  const raw = process.env.UNQUARANTINE_REGRESSION_THRESHOLD;
  const v = raw == null ? NaN : Number(raw);
  return Number.isFinite(v) && v > 0 ? v : 0.05;
})();

export interface QuarantinedFeatureRecord {
  name: string;
  transformKind: string;
  sourceColumn: string | null;
  quarantinedAt: string;
  reason: string;
  detail?: Record<string, unknown> | null;
}

export interface AblationRunnerResult {
  nSamples: number;
  nFolds: number;
  baselineLogLoss: number | null;
  augmentedLogLoss: number | null;
  baselineAccuracy: number | null;
  augmentedAccuracy: number | null;
  extra?: Record<string, unknown>;
}

export async function listCandidates(): Promise<FeatureLabCandidateRow[]> {
  return db
    .select()
    .from(featureLabCandidatesTable)
    .orderBy(desc(featureLabCandidatesTable.createdAt));
}

export async function createCandidate(args: {
  name: string;
  description?: string | null;
  transformKind: FeatureTransformKind;
  sourceColumn?: string | null;
  proposedBy?: string | null;
}): Promise<FeatureLabCandidateRow> {
  if (!FEATURE_TRANSFORM_KINDS.includes(args.transformKind)) {
    throw new Error(
      `invalid transformKind '${args.transformKind}' (must be one of ${FEATURE_TRANSFORM_KINDS.join(",")})`,
    );
  }
  if (args.transformKind === "passthrough_existing" && !args.sourceColumn) {
    throw new Error("passthrough_existing requires sourceColumn");
  }
  const [row] = await db
    .insert(featureLabCandidatesTable)
    .values({
      name: args.name,
      description: args.description ?? null,
      transformKind: args.transformKind,
      sourceColumn: args.sourceColumn ?? null,
      proposedBy: args.proposedBy ?? null,
      state: "draft",
    })
    .returning();
  return row;
}

export async function listReports(
  candidateId?: number,
): Promise<FeatureLabReportRow[]> {
  const q = db.select().from(featureLabReportsTable);
  const out = candidateId == null
    ? await q.orderBy(desc(featureLabReportsTable.createdAt)).limit(200)
    : await db
        .select()
        .from(featureLabReportsTable)
        .where(eq(featureLabReportsTable.candidateId, candidateId))
        .orderBy(desc(featureLabReportsTable.createdAt));
  return out;
}

export async function runAblation(args: {
  candidateId: number;
  timeframe: string;
  coinId?: string;
}): Promise<FeatureLabReportRow> {
  const [cand] = await db
    .select()
    .from(featureLabCandidatesTable)
    .where(eq(featureLabCandidatesTable.id, args.candidateId));
  if (!cand) throw new Error(`candidate ${args.candidateId} not found`);
  const coinId = args.coinId ?? "__pooled__";
  let runnerStatus: "ok" | "error" = "ok";
  let runnerError: string | null = null;
  let result: AblationRunnerResult = {
    nSamples: 0,
    nFolds: 0,
    baselineLogLoss: null,
    augmentedLogLoss: null,
    baselineAccuracy: null,
    augmentedAccuracy: null,
  };
  try {
    const res = await fetch(`${ML_BASE()}/ml/feature-lab/ablate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        name: cand.name,
        transformKind: cand.transformKind,
        sourceColumn: cand.sourceColumn,
        timeframe: args.timeframe,
        coinId,
      }),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`ml-engine ablate ${res.status}: ${text.slice(0, 300)}`);
    }
    result = (await res.json()) as AblationRunnerResult;
  } catch (err) {
    runnerStatus = "error";
    runnerError = err instanceof Error ? err.message : String(err);
    logger.warn(
      { candidateId: args.candidateId, err: runnerError },
      "Feature-lab ablation failed",
    );
  }
  const dLogLoss =
    result.baselineLogLoss != null && result.augmentedLogLoss != null
      ? result.baselineLogLoss - result.augmentedLogLoss
      : null;
  const dAcc =
    result.baselineAccuracy != null && result.augmentedAccuracy != null
      ? result.augmentedAccuracy - result.baselineAccuracy
      : null;
  const [report] = await db
    .insert(featureLabReportsTable)
    .values({
      candidateId: args.candidateId,
      timeframe: args.timeframe,
      coinId,
      nFolds: result.nFolds,
      nSamples: result.nSamples,
      baselineLogLoss: result.baselineLogLoss,
      augmentedLogLoss: result.augmentedLogLoss,
      deltaLogLoss: dLogLoss,
      baselineAccuracy: result.baselineAccuracy,
      augmentedAccuracy: result.augmentedAccuracy,
      deltaAccuracy: dAcc,
      extra: (result.extra as object | undefined) ?? null,
      runnerStatus,
      runnerError,
    })
    .returning();
  if (runnerStatus === "ok" && cand.state === "draft") {
    await db
      .update(featureLabCandidatesTable)
      .set({ state: "ablated", updatedAt: new Date() })
      .where(eq(featureLabCandidatesTable.id, args.candidateId));
  }
  return report;
}

export interface PromotionEligibility {
  eligible: boolean;
  reasons: string[];
  bestReport: FeatureLabReportRow | null;
}

export async function evaluatePromotion(
  candidateId: number,
): Promise<PromotionEligibility> {
  const reports = await listReports(candidateId);
  const reasons: string[] = [];
  if (reports.length === 0) {
    reasons.push("no_ablation_reports");
    return { eligible: false, reasons, bestReport: null };
  }
  // Pick the most recent OK report.
  const okReports = reports.filter((r) => r.runnerStatus === "ok");
  if (okReports.length === 0) {
    reasons.push("no_successful_ablation");
    return { eligible: false, reasons, bestReport: null };
  }
  const best = okReports[0]; // already DESC by createdAt
  if ((best.nSamples ?? 0) < MIN_ABLATION_SAMPLES) {
    reasons.push(`insufficient_samples (${best.nSamples} < ${MIN_ABLATION_SAMPLES})`);
  }
  if ((best.deltaLogLoss ?? 0) <= 0) {
    reasons.push("delta_log_loss_not_positive");
  }
  return {
    eligible: reasons.length === 0,
    reasons,
    bestReport: best,
  };
}

/**
 * Approve a candidate. Validates promotion eligibility (unless `force`),
 * appends the candidate spec to `app_settings.feature_lab.approved`,
 * and transitions the candidate state to `approved`. The next training
 * run reads this setting to extend its feature schema (and thereby
 * bumps `feature_schema_hash` on the resulting manifests, so any model
 * trained against the old schema must re-enter validation).
 */
export async function approveCandidate(args: {
  candidateId: number;
  approvedBy?: string;
  note?: string;
  force?: boolean;
}): Promise<{
  candidate: FeatureLabCandidateRow;
  approvedFeatures: { name: string; transformKind: string; sourceColumn: string | null }[];
}> {
  const verdict = await evaluatePromotion(args.candidateId);
  if (!verdict.eligible && !args.force) {
    throw new Error(
      `Promotion ineligible: ${verdict.reasons.join("; ")}`,
    );
  }
  const [cand] = await db
    .select()
    .from(featureLabCandidatesTable)
    .where(eq(featureLabCandidatesTable.id, args.candidateId));
  if (!cand) throw new Error(`candidate ${args.candidateId} not found`);

  const existing = await getApprovedFeatures();
  const filtered = existing.filter((f) => f.name !== cand.name);
  filtered.push({
    name: cand.name,
    transformKind: cand.transformKind,
    sourceColumn: cand.sourceColumn,
  });
  await db
    .insert(appSettingsTable)
    .values({
      key: APPROVED_FEATURES_SETTING_KEY,
      value: { features: filtered },
    })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: { features: filtered }, updatedAt: new Date() },
    });

  const [updated] = await db
    .update(featureLabCandidatesTable)
    .set({
      state: "approved",
      approvedAt: new Date(),
      approvedBy: args.approvedBy ?? null,
      approvalNote: args.note ?? null,
      updatedAt: new Date(),
    })
    .where(eq(featureLabCandidatesTable.id, args.candidateId))
    .returning();

  logger.info(
    {
      candidateId: args.candidateId,
      name: cand.name,
      approvedBy: args.approvedBy,
      forced: args.force === true,
    },
    "Feature lab: candidate approved",
  );
  return { candidate: updated, approvedFeatures: filtered };
}

/**
 * Task #243 — operator override for an auto-retired feature. Auto-retire
 * (task #236) is a one-way move: the feature is yanked from the approved
 * bucket and appended to `feature_lab.quarantined`. When the operator
 * disagrees (e.g. the validation regression was caused by a coincident
 * data outage, not the feature), this brings it back:
 *
 *   - Drops the record from `feature_lab.quarantined`.
 *   - Re-appends the spec to `feature_lab.approved` so the next training
 *     run picks it up again. We prefer the spec recorded on the
 *     quarantined row (which preserves the original transformKind /
 *     sourceColumn even if approved was rewritten in the meantime) and
 *     fall back to the candidate row when the quarantined record is
 *     missing those fields.
 *   - Flips `feature_lab_candidates.state` back to "approved" and stamps
 *     who did it + the optional note onto `approvedBy` / `approvalNote`.
 *
 * Throws if the candidate is not currently quarantined or if no
 * quarantined record exists for it — un-quarantining a feature that
 * was never quarantined would be a silent no-op otherwise.
 */
/**
 * Task #248 — re-check the original validation regression that drove an
 * auto-retire against the most recent training report. We pull the
 * pooled log_loss for each implicated timeframe out of
 * `${ML_BASE}/ml/training/report` and compare it to the `prior_log_loss`
 * baked into the quarantined record. If `latest - prior` is still above
 * the regression threshold, the regression has *not* recovered and the
 * un-quarantine UI should warn the operator (and the API will refuse
 * unless the caller passes a typed `acknowledgement` or `force=true`).
 *
 * Best-effort: missing report / unreachable ml-engine / missing pooled
 * slot for a timeframe never blocks an un-quarantine — those cases
 * surface as `status: "unknown"` so the UI can degrade to the original
 * delta + a banner explaining the latest report could not be read.
 */
export interface UnquarantineRegressionPerTimeframe {
  timeframe: string;
  originalPriorLogLoss: number | null;
  originalCurrentLogLoss: number | null;
  originalDelta: number | null;
  latestPooledLogLoss: number | null;
  latestDelta: number | null;
  recovered: boolean | null;
}

export interface UnquarantineRegressionAssessment {
  status: "ok" | "no_quarantine_record" | "no_training_report" | "unknown";
  threshold: number;
  reason: string | null;
  quarantinedAt: string | null;
  originalReason: string | null;
  worstOriginalDelta: number | null;
  worstLatestDelta: number | null;
  regressionStillPresent: boolean;
  perTimeframe: UnquarantineRegressionPerTimeframe[];
  trainingReportError?: string;
}

function _extractPooledLogLoss(report: unknown, timeframe: string): number | null {
  if (!report || typeof report !== "object") return null;
  const tfs = (report as { timeframes?: unknown }).timeframes;
  if (!tfs || typeof tfs !== "object") return null;
  const tfReport = (tfs as Record<string, unknown>)[timeframe];
  if (!tfReport || typeof tfReport !== "object") return null;
  const pooled = (tfReport as { pooled?: unknown }).pooled;
  if (!pooled || typeof pooled !== "object") return null;
  if ((pooled as { status?: unknown }).status !== "trained") return null;
  const metrics = (pooled as { metrics?: unknown }).metrics;
  if (!metrics || typeof metrics !== "object") return null;
  const v = (metrics as { log_loss?: unknown }).log_loss;
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return null;
  return n;
}

interface QuarantineDetailTimeframe {
  timeframe?: unknown;
  current_log_loss?: unknown;
  prior_log_loss?: unknown;
  delta_log_loss?: unknown;
}

function _readDetailTimeframes(rec: QuarantinedFeatureRecord): QuarantineDetailTimeframe[] {
  const detail = rec.detail as { timeframes?: unknown } | null | undefined;
  const list = detail?.timeframes;
  if (!Array.isArray(list)) return [];
  return list.filter((x): x is QuarantineDetailTimeframe => !!x && typeof x === "object");
}

/**
 * Pure (no I/O) version of the regression assessment. Exported for
 * testing — production callers should use `assessUnquarantineRegression`
 * which wires up the DB and ml-engine fetch.
 */
export function buildUnquarantineRegressionAssessment(
  qRecord: QuarantinedFeatureRecord | null,
  trainingReport: unknown | null,
  reportError: string | null,
  threshold: number = UNQUARANTINE_REGRESSION_THRESHOLD,
): UnquarantineRegressionAssessment {
  if (!qRecord) {
    return {
      status: "no_quarantine_record",
      threshold,
      reason: null,
      quarantinedAt: null,
      originalReason: null,
      worstOriginalDelta: null,
      worstLatestDelta: null,
      regressionStillPresent: false,
      perTimeframe: [],
    };
  }
  const tfRows = _readDetailTimeframes(qRecord);
  const perTimeframe: UnquarantineRegressionPerTimeframe[] = tfRows.map((t) => {
    const tf = typeof t.timeframe === "string" ? t.timeframe : "";
    const prior = typeof t.prior_log_loss === "number" ? t.prior_log_loss : null;
    const origCur = typeof t.current_log_loss === "number" ? t.current_log_loss : null;
    const origDelta = typeof t.delta_log_loss === "number" ? t.delta_log_loss : null;
    const latest = trainingReport ? _extractPooledLogLoss(trainingReport, tf) : null;
    const latestDelta =
      latest != null && prior != null ? latest - prior : null;
    const recovered =
      latestDelta == null ? null : latestDelta <= threshold;
    return {
      timeframe: tf,
      originalPriorLogLoss: prior,
      originalCurrentLogLoss: origCur,
      originalDelta: origDelta,
      latestPooledLogLoss: latest,
      latestDelta,
      recovered,
    };
  });

  const worstOrig = perTimeframe
    .map((p) => p.originalDelta)
    .filter((v): v is number => v != null)
    .reduce<number | null>((m, v) => (m == null || v > m ? v : m), null);
  const worstLatest = perTimeframe
    .map((p) => p.latestDelta)
    .filter((v): v is number => v != null)
    .reduce<number | null>((m, v) => (m == null || v > m ? v : m), null);

  let status: UnquarantineRegressionAssessment["status"];
  let regressionStillPresent: boolean;
  if (perTimeframe.length === 0) {
    // No structured detail to re-check (older quarantine record). Treat
    // as unknown so the UI shows a generic warning rather than a green
    // light.
    status = "unknown";
    regressionStillPresent = true;
  } else if (trainingReport == null) {
    status = "no_training_report";
    regressionStillPresent = true;
  } else if (worstLatest == null) {
    // Report present but no pooled slot for any of the implicated
    // timeframes — we can't tell either way.
    status = "unknown";
    regressionStillPresent = true;
  } else {
    status = "ok";
    regressionStillPresent = worstLatest > threshold;
  }

  return {
    status,
    threshold,
    reason: reportError ?? null,
    quarantinedAt: qRecord.quarantinedAt,
    originalReason: qRecord.reason,
    worstOriginalDelta: worstOrig,
    worstLatestDelta: worstLatest,
    regressionStillPresent,
    perTimeframe,
    ...(reportError ? { trainingReportError: reportError } : {}),
  };
}

export async function assessUnquarantineRegression(
  candidateId: number,
): Promise<UnquarantineRegressionAssessment> {
  const [cand] = await db
    .select()
    .from(featureLabCandidatesTable)
    .where(eq(featureLabCandidatesTable.id, candidateId));
  let qRecord: QuarantinedFeatureRecord | null = null;
  if (cand) {
    const quarantined = await getQuarantinedFeatures();
    qRecord = quarantined.find((q) => q.name === cand.name) ?? null;
  }
  let trainingReport: unknown | null = null;
  let reportError: string | null = null;
  if (qRecord) {
    try {
      const res = await fetch(`${ML_BASE()}/ml/training/report`);
      if (!res.ok) {
        reportError = `ml-engine /ml/training/report ${res.status}`;
      } else {
        const body = (await res.json()) as { status?: unknown };
        const status = body?.status;
        if (status === "missing" || status === "error") {
          reportError = `training report status=${String(status)}`;
        } else {
          trainingReport = body;
        }
      }
    } catch (err) {
      reportError = err instanceof Error ? err.message : String(err);
    }
  }
  return buildUnquarantineRegressionAssessment(
    qRecord,
    trainingReport,
    reportError,
  );
}

/**
 * Thrown by `unquarantineCandidate` when the regression that drove the
 * auto-retire still looks present in the latest training report and the
 * caller did not pass `force` or an `acknowledgement` reason. Carries
 * the assessment so the API layer can return it to the dashboard.
 */
export class UnquarantineRegressionStillPresentError extends Error {
  readonly code = "regression_still_present";
  readonly assessment: UnquarantineRegressionAssessment;
  constructor(assessment: UnquarantineRegressionAssessment) {
    super(
      `validation regression still present (latest delta=${
        assessment.worstLatestDelta ?? "unknown"
      } > threshold=${assessment.threshold}); pass acknowledgement or force=true to override`,
    );
    this.assessment = assessment;
  }
}

export async function unquarantineCandidate(args: {
  candidateId: number;
  approvedBy?: string;
  note?: string;
  force?: boolean;
  acknowledgement?: string;
}): Promise<{
  candidate: FeatureLabCandidateRow;
  approvedFeatures: { name: string; transformKind: string; sourceColumn: string | null }[];
  quarantined: QuarantinedFeatureRecord[];
  assessment: UnquarantineRegressionAssessment;
}> {
  const [cand] = await db
    .select()
    .from(featureLabCandidatesTable)
    .where(eq(featureLabCandidatesTable.id, args.candidateId));
  if (!cand) throw new Error(`candidate ${args.candidateId} not found`);
  if (cand.state !== "quarantined") {
    throw new Error(
      `candidate ${cand.name} is not quarantined (state=${cand.state})`,
    );
  }

  const quarantined = await getQuarantinedFeatures();
  const qRecord = quarantined.find((q) => q.name === cand.name) ?? null;
  if (!qRecord) {
    throw new Error(
      `no quarantined record found for ${cand.name} — refusing to un-quarantine`,
    );
  }
  const remainingQuarantined = quarantined.filter((q) => q.name !== cand.name);

  // Task #248 — re-check the regression before letting the operator
  // restore the feature. A non-empty `acknowledgement` (typed reason)
  // or an explicit `force=true` lets the caller proceed even when the
  // regression is still present; otherwise we reject so the dashboard
  // can prompt the operator.
  const ack = (args.acknowledgement ?? "").trim();
  const assessment = await assessUnquarantineRegression(args.candidateId);
  if (assessment.regressionStillPresent && !args.force && !ack) {
    throw new UnquarantineRegressionStillPresentError(assessment);
  }

  const transformKind = qRecord.transformKind || cand.transformKind;
  const sourceColumn = qRecord.sourceColumn ?? cand.sourceColumn ?? null;

  const existing = await getApprovedFeatures();
  const filtered = existing.filter((f) => f.name !== cand.name);
  filtered.push({ name: cand.name, transformKind, sourceColumn });

  await db
    .insert(appSettingsTable)
    .values({
      key: APPROVED_FEATURES_SETTING_KEY,
      value: { features: filtered },
    })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: { features: filtered }, updatedAt: new Date() },
    });
  await db
    .insert(appSettingsTable)
    .values({
      key: QUARANTINED_FEATURES_SETTING_KEY,
      value: { features: remainingQuarantined },
    })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: { features: remainingQuarantined }, updatedAt: new Date() },
    });

  const ackTrim = (args.acknowledgement ?? "").trim();
  const ackSuffix = ackTrim
    ? ` (regression-still-present ack: ${ackTrim})`
    : args.force && assessment.regressionStillPresent
      ? " (forced despite regression still present)"
      : "";
  const noteSuffix = `un-quarantined by ${args.approvedBy ?? "operator"}${
    args.note ? `: ${args.note}` : ""
  }${ackSuffix}`;
  const newNote = cand.approvalNote ? `${cand.approvalNote}\n${noteSuffix}` : noteSuffix;
  const [updated] = await db
    .update(featureLabCandidatesTable)
    .set({
      state: "approved",
      approvedAt: new Date(),
      approvedBy: args.approvedBy ?? null,
      approvalNote: newNote,
      updatedAt: new Date(),
    })
    .where(eq(featureLabCandidatesTable.id, args.candidateId))
    .returning();

  // Task #247 — append a time-ordered audit row capturing who did the
  // override and what the prior quarantine reason was, so the next
  // reviewer can spot patterns when the same feature regresses again.
  try {
    await db.insert(featureLabUnquarantineEventsTable).values({
      candidateId: args.candidateId,
      candidateName: cand.name,
      operator: args.approvedBy ?? "operator",
      note: args.note ?? null,
      priorReason: qRecord.reason ?? null,
      priorReasonDetail: (qRecord.detail as object | undefined) ?? null,
      priorQuarantinedAt: qRecord.quarantinedAt ? new Date(qRecord.quarantinedAt) : null,
    });
  } catch (err) {
    // Audit logging must not block the operator action — log and move on.
    logger.warn(
      { candidateId: args.candidateId, err: err instanceof Error ? err.message : String(err) },
      "Feature lab: failed to record unquarantine audit event",
    );
  }

  logger.info(
    {
      candidateId: args.candidateId,
      name: cand.name,
      approvedBy: args.approvedBy,
    },
    "Feature lab: candidate un-quarantined",
  );
  return {
    candidate: updated,
    approvedFeatures: filtered,
    quarantined: remainingQuarantined,
    assessment,
  };
}

export async function rejectCandidate(
  candidateId: number,
  note?: string,
): Promise<FeatureLabCandidateRow> {
  const [row] = await db
    .update(featureLabCandidatesTable)
    .set({
      state: "rejected",
      approvalNote: note ?? null,
      updatedAt: new Date(),
    })
    .where(eq(featureLabCandidatesTable.id, candidateId))
    .returning();
  return row;
}

export async function getApprovedFeatures(): Promise<
  { name: string; transformKind: string; sourceColumn: string | null }[]
> {
  const [row] = await db
    .select()
    .from(appSettingsTable)
    .where(eq(appSettingsTable.key, APPROVED_FEATURES_SETTING_KEY));
  if (!row?.value) return [];
  const v = row.value as { features?: unknown };
  if (!Array.isArray(v.features)) return [];
  return v.features as { name: string; transformKind: string; sourceColumn: string | null }[];
}

/**
 * Task #235 — for each approved feature, look up the most recent
 * model_registry rows whose `metricsSnapshot.approved_features_applied`
 * actually includes that feature, grouped per timeframe. This is what
 * closes the operator feedback loop: approving a feature kicks it into
 * the next training run, and this query proves a specific model
 * version baked it in. Empty `appliedIn` means the feature has been
 * approved but no trained model has carried it yet (e.g. waiting for
 * the next retrain or training failed).
 *
 * Reads only base lightgbm rows — meta models don't carry the
 * approved-features schema (they operate on top of base predictions).
 * Within a timeframe we keep the most recent model per coin slot so the
 * UI can show "champion + recent shadows" without exploding into every
 * coin row.
 */
export interface AppliedModelEntry {
  registryId: number;
  modelId: string;
  modelVersion: string;
  coinId: string;
  timeframe: string;
  state: string;
  promotedAt: string | null;
  createdAt: string;
}
export interface ApprovedFeatureAppliedSummary {
  name: string;
  transformKind: string;
  sourceColumn: string | null;
  appliedIn: AppliedModelEntry[];
}

export async function summarizeApprovedFeatureApplications(opts: {
  limitPerFeature?: number;
} = {}): Promise<ApprovedFeatureAppliedSummary[]> {
  const limitPerFeature = Math.max(1, Math.min(50, opts.limitPerFeature ?? 12));
  const approved = await getApprovedFeatures();
  if (approved.length === 0) return [];
  const rows = await db
    .select({
      id: modelRegistryTable.id,
      modelId: modelRegistryTable.modelId,
      modelVersion: modelRegistryTable.modelVersion,
      coinId: modelRegistryTable.coinId,
      timeframe: modelRegistryTable.timeframe,
      state: modelRegistryTable.state,
      promotedAt: modelRegistryTable.promotedAt,
      createdAt: modelRegistryTable.createdAt,
      metricsSnapshot: modelRegistryTable.metricsSnapshot,
    })
    .from(modelRegistryTable)
    .where(eq(modelRegistryTable.modelId, "lightgbm"))
    .orderBy(desc(modelRegistryTable.createdAt));
  const byFeature = new Map<string, AppliedModelEntry[]>();
  for (const a of approved) byFeature.set(a.name, []);
  for (const r of rows) {
    const snap = (r.metricsSnapshot ?? {}) as { approved_features_applied?: unknown };
    const applied = Array.isArray(snap.approved_features_applied)
      ? (snap.approved_features_applied as unknown[]).filter(
          (x): x is string => typeof x === "string",
        )
      : [];
    if (applied.length === 0) continue;
    for (const featName of applied) {
      const bucket = byFeature.get(featName);
      if (!bucket) continue;
      if (bucket.length >= limitPerFeature) continue;
      bucket.push({
        registryId: r.id,
        modelId: r.modelId,
        modelVersion: r.modelVersion,
        coinId: r.coinId,
        timeframe: r.timeframe,
        state: r.state,
        promotedAt: r.promotedAt ? r.promotedAt.toISOString() : null,
        createdAt: r.createdAt.toISOString(),
      });
    }
  }
  return approved.map((a) => ({
    name: a.name,
    transformKind: a.transformKind,
    sourceColumn: a.sourceColumn,
    appliedIn: byFeature.get(a.name) ?? [],
  }));
}

void and;

/**
 * Task #236 — list features the ml-engine auto-retired after a training
 * run regressed beyond the validation guardrail. The ml-engine writes
 * this bucket directly via SQL after each `run_training`; this helper
 * is the read-side for the api-server / dashboard.
 */
export async function getQuarantinedFeatures(): Promise<QuarantinedFeatureRecord[]> {
  const [row] = await db
    .select()
    .from(appSettingsTable)
    .where(eq(appSettingsTable.key, QUARANTINED_FEATURES_SETTING_KEY));
  if (!row?.value) return [];
  const v = row.value as { features?: unknown };
  if (!Array.isArray(v.features)) return [];
  return v.features as QuarantinedFeatureRecord[];
}


/**
 * Task #247 — read side for the un-quarantine override audit log. The
 * Feature Lab card in the dashboard pulls the most recent N events so
 * reviewers can spot patterns (same operator overriding the same
 * feature multiple times, etc.).
 */
export async function listUnquarantineEvents(
  limit = 25,
): Promise<FeatureLabUnquarantineEventRow[]> {
  const cap = Math.max(1, Math.min(200, limit));
  return db
    .select()
    .from(featureLabUnquarantineEventsTable)
    .orderBy(desc(featureLabUnquarantineEventsTable.createdAt))
    .limit(cap);
}

/**
 * Task #257 — retention sweep for `feature_lab_unquarantine_events`.
 *
 * Mirrors the pattern used by skip-tracker / journal-retention: deletes
 * rows older than `retentionDays` (default 180), but preserves at least
 * the most recent `minPerCandidate` events per candidate so reviewers can
 * always see at least one prior override even on long-quiet candidates.
 *
 * Implemented as a single SQL statement using a window function so the
 * "keep most recent N per candidate" rule is evaluated atomically against
 * the same snapshot the DELETE acts on (no race with concurrent inserts
 * promoting a row past the keep threshold mid-prune).
 */
export const DEFAULT_UNQUARANTINE_RETENTION_DAYS = 180;
export const DEFAULT_UNQUARANTINE_MIN_PER_CANDIDATE = 1;

export interface PruneUnquarantineEventsResult {
  ranAt: string;
  retentionDays: number;
  minPerCandidate: number;
  cutoff: string;
  deleted: number;
  durationMs: number;
}

function getUnquarantineRetentionDays(): number {
  const raw = process.env["UNQUARANTINE_EVENTS_RETENTION_DAYS"];
  const parsed = raw !== undefined ? Number(raw) : NaN;
  return Number.isFinite(parsed) && parsed > 0
    ? Math.floor(parsed)
    : DEFAULT_UNQUARANTINE_RETENTION_DAYS;
}

function getUnquarantineMinPerCandidate(): number {
  const raw = process.env["UNQUARANTINE_EVENTS_MIN_PER_CANDIDATE"];
  const parsed = raw !== undefined ? Number(raw) : NaN;
  return Number.isFinite(parsed) && parsed >= 1
    ? Math.floor(parsed)
    : DEFAULT_UNQUARANTINE_MIN_PER_CANDIDATE;
}

export async function pruneUnquarantineEvents(
  opts: { retentionDays?: number; minPerCandidate?: number } = {},
): Promise<PruneUnquarantineEventsResult> {
  const startedAt = Date.now();
  const retentionDays = Math.max(1, opts.retentionDays ?? getUnquarantineRetentionDays());
  const minPerCandidate = Math.max(
    1,
    opts.minPerCandidate ?? getUnquarantineMinPerCandidate(),
  );
  const cutoff = new Date(startedAt - retentionDays * 24 * 60 * 60 * 1000);

  let deleted = 0;
  try {
    const result = await db.execute(sql`
      WITH ranked AS (
        SELECT id,
          row_number() OVER (
            PARTITION BY candidate_id ORDER BY created_at DESC, id DESC
          ) AS rn
        FROM feature_lab_unquarantine_events
      )
      DELETE FROM feature_lab_unquarantine_events e
      USING ranked r
      WHERE e.id = r.id
        AND r.rn > ${minPerCandidate}
        AND e.created_at < ${cutoff}
      RETURNING e.id
    `);
    deleted = Array.isArray(result.rows) ? result.rows.length : 0;
  } catch (err) {
    logger.error(
      { err: err instanceof Error ? err.message : String(err) },
      "feature-lab: unquarantine events prune failed",
    );
    throw err;
  }

  const out: PruneUnquarantineEventsResult = {
    ranAt: new Date(startedAt).toISOString(),
    retentionDays,
    minPerCandidate,
    cutoff: cutoff.toISOString(),
    deleted,
    durationMs: Date.now() - startedAt,
  };
  if (deleted > 0) {
    logger.info(out, "feature-lab: pruned unquarantine events");
  } else {
    logger.debug(out, "feature-lab: no unquarantine events to prune");
  }
  return out;
}

const UNQUARANTINE_PRUNE_INTERVAL_MS = 60 * 60 * 1000;
let lastUnquarantinePruneSuccessAt = 0;

export async function pruneUnquarantineEventsIfDue(
  force = false,
): Promise<PruneUnquarantineEventsResult | null> {
  const now = Date.now();
  // Gate on last *successful* run — a failed sweep should be retried on
  // the next tick rather than suppressed for an hour.
  if (
    !force &&
    now - lastUnquarantinePruneSuccessAt < UNQUARANTINE_PRUNE_INTERVAL_MS
  ) {
    return null;
  }
  const result = await pruneUnquarantineEvents();
  lastUnquarantinePruneSuccessAt = Date.now();
  return result;
}

/**
 * Task #256 — roll-up of un-quarantine override events over a window
 * (default 30 days). Powers a small group-by panel on the Feature Lab
 * card so reviewers can spot repeat patterns ("operator X overrode 6
 * times this month", "feature Y has been un-quarantined 3 times")
 * without scrolling the raw audit log.
 */
export interface UnquarantineOverrideOperatorEntry {
  operator: string;
  count: number;
  lastAt: string;
  candidates: string[];
}
export interface UnquarantineOverrideCandidateEntry {
  candidateId: number;
  candidateName: string;
  count: number;
  lastAt: string;
  operators: string[];
}
export interface UnquarantineOverrideSummary {
  windowDays: number;
  windowStart: string;
  totalEvents: number;
  byOperator: UnquarantineOverrideOperatorEntry[];
  byCandidate: UnquarantineOverrideCandidateEntry[];
}

export async function summarizeUnquarantineOverrides(
  windowDays = 30,
): Promise<UnquarantineOverrideSummary> {
  const days = Math.max(1, Math.min(365, Math.floor(Number(windowDays) || 30)));
  const windowStart = new Date(Date.now() - days * 86_400_000);

  const rows = await db
    .select()
    .from(featureLabUnquarantineEventsTable)
    .where(gte(featureLabUnquarantineEventsTable.createdAt, windowStart))
    .orderBy(desc(featureLabUnquarantineEventsTable.createdAt));

  const opMap = new Map<
    string,
    { count: number; lastAt: Date; candidates: Set<string> }
  >();
  const candMap = new Map<
    number,
    { name: string; count: number; lastAt: Date; operators: Set<string> }
  >();

  for (const r of rows) {
    const opKey = r.operator || "(unknown)";
    const op = opMap.get(opKey);
    if (op) {
      op.count++;
      if (r.createdAt > op.lastAt) op.lastAt = r.createdAt;
      op.candidates.add(r.candidateName);
    } else {
      opMap.set(opKey, {
        count: 1,
        lastAt: r.createdAt,
        candidates: new Set([r.candidateName]),
      });
    }
    const cand = candMap.get(r.candidateId);
    if (cand) {
      cand.count++;
      if (r.createdAt > cand.lastAt) cand.lastAt = r.createdAt;
      cand.operators.add(opKey);
    } else {
      candMap.set(r.candidateId, {
        name: r.candidateName,
        count: 1,
        lastAt: r.createdAt,
        operators: new Set([opKey]),
      });
    }
  }

  return {
    windowDays: days,
    windowStart: windowStart.toISOString(),
    totalEvents: rows.length,
    byOperator: Array.from(opMap.entries())
      .map(([operator, v]) => ({
        operator,
        count: v.count,
        lastAt: v.lastAt.toISOString(),
        candidates: Array.from(v.candidates).sort(),
      }))
      .sort((a, b) => b.count - a.count || a.operator.localeCompare(b.operator)),
    byCandidate: Array.from(candMap.entries())
      .map(([candidateId, v]) => ({
        candidateId,
        candidateName: v.name,
        count: v.count,
        lastAt: v.lastAt.toISOString(),
        operators: Array.from(v.operators).sort(),
      }))
      .sort(
        (a, b) => b.count - a.count || a.candidateName.localeCompare(b.candidateName),
      ),
  };
}

// silence unused-import warnings if the tree-shaker can't infer
void sql;
