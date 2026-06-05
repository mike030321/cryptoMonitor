/**
 * Phase 6 — Automatic model quarantine.
 *
 * Wired into the registry: scans every active row in `model_registry`
 * (excluding already-quarantined / retired) and applies three guardrails:
 *
 *   1. probability collapse — share of rows whose argmax probability ≥ 0.95
 *      and direction == "stable" exceeds COLLAPSE_THRESHOLD. Means the
 *      model has lost the ability to emit directional calls.
 *   2. calibration drift   — ECE above DRIFT_THRESHOLDS.calibration over
 *      a recent window.
 *   3. feature drift       — max |z-score| > DRIFT_THRESHOLDS.feature.
 *
 * On breach, the registry row is moved to "quarantined", a row is written
 * to `quarantine_events`, and the most recent drift snapshot is persisted
 * for forensic context. Quarantined rows are never auto-unquarantined —
 * an operator must `setRegistryState` back to `shadow`/`challenger`/
 * `champion` after fixing the underlying issue.
 */
import {
  db,
  modelRegistryTable,
  predictionJournalTable,
  quarantineEventsTable,
  type ModelRegistryRow,
} from "@workspace/db";
import { and, eq, gte, sql, ne } from "drizzle-orm";
import { logger } from "./logger";
import {
  computeCalibrationDrift,
  computeFeatureDrift,
  persistDriftSnapshot,
} from "./drift-tracker";
import {
  recomputeSlotStarvationAlerts,
  type SlotStarvationAlert,
} from "./quarantine-slot-alerts";

const COLLAPSE_THRESHOLD = 0.85;
const COLLAPSE_MIN_SAMPLES = 50;

export interface QuarantineDecision {
  registryId: number;
  modelId: string;
  modelVersion: string;
  coinId: string;
  timeframe: string;
  state: string;
  decision: "kept" | "quarantined";
  reasonCodes: string[];
  detail: Record<string, unknown>;
}

async function probCollapseCheck(
  registryId: number,
  windowHours: number,
): Promise<{ breached: boolean; share: number; n: number }> {
  const since = new Date(Date.now() - windowHours * 3600 * 1000);
  const rows = await db
    .select({
      probUp: predictionJournalTable.probUp,
      probDown: predictionJournalTable.probDown,
      probStable: predictionJournalTable.probStable,
      direction: predictionJournalTable.direction,
    })
    .from(predictionJournalTable)
    .where(
      and(
        eq(predictionJournalTable.registryId, registryId),
        gte(predictionJournalTable.createdAt, since),
      ),
    )
    .limit(5000);
  let collapsed = 0;
  let n = 0;
  for (const r of rows) {
    const probs = [r.probUp ?? 0, r.probDown ?? 0, r.probStable ?? 0];
    const maxP = Math.max(...probs);
    if (!Number.isFinite(maxP) || maxP <= 0) continue;
    n += 1;
    if (maxP >= 0.95 && r.direction === "stable") collapsed += 1;
  }
  const share = n > 0 ? collapsed / n : 0;
  return {
    breached: n >= COLLAPSE_MIN_SAMPLES && share >= COLLAPSE_THRESHOLD,
    share,
    n,
  };
}

async function evaluateRow(
  row: ModelRegistryRow,
  windowHours: number,
): Promise<QuarantineDecision> {
  const reasons: string[] = [];
  const detail: Record<string, unknown> = {};

  const collapse = await probCollapseCheck(row.id, windowHours);
  detail.probCollapse = collapse;
  if (collapse.breached) reasons.push("prob_collapse");

  const calib = await computeCalibrationDrift({
    windowHours,
    registryId: row.id,
  });
  detail.calibration = { score: calib.score, n: calib.nSamples, breached: calib.breached };
  if (calib.breached) reasons.push("calibration_drift");
  await persistDriftSnapshot({
    registryId: row.id,
    coinId: row.coinId,
    timeframe: row.timeframe,
    kind: "calibration",
    nSamples: calib.nSamples,
    score: calib.score,
    threshold: calib.threshold,
    breached: calib.breached,
    detail: { buckets: calib.buckets },
  });

  const feat = await computeFeatureDrift({
    windowHours,
    registryId: row.id,
  });
  detail.feature = { score: feat.score, n: feat.nSamples, breached: feat.breached };
  if (feat.breached) reasons.push("feature_drift");
  await persistDriftSnapshot({
    registryId: row.id,
    coinId: row.coinId,
    timeframe: row.timeframe,
    kind: "feature",
    nSamples: feat.nSamples,
    score: feat.score,
    threshold: feat.threshold,
    breached: feat.breached,
    detail: { perFeature: feat.perFeature.slice(0, 8) },
  });

  return {
    registryId: row.id,
    modelId: row.modelId,
    modelVersion: row.modelVersion,
    coinId: row.coinId,
    timeframe: row.timeframe,
    state: row.state,
    decision: reasons.length > 0 ? "quarantined" : "kept",
    reasonCodes: reasons,
    detail,
  };
}

/**
 * Run one quarantine sweep over every non-terminal registry row. Returns
 * the per-row decision matrix the dashboard renders.
 */
export async function runQuarantineSweep(opts?: {
  windowHours?: number;
  dryRun?: boolean;
}): Promise<{
  generatedAt: string;
  windowHours: number;
  dryRun: boolean;
  decisions: QuarantineDecision[];
  starvedSlots: SlotStarvationAlert[];
}> {
  const windowHours = opts?.windowHours ?? 168;
  const dryRun = opts?.dryRun === true;
  const rows = await db
    .select()
    .from(modelRegistryTable)
    .where(
      and(
        eq(modelRegistryTable.isActive, true),
        ne(modelRegistryTable.state, "quarantined"),
        ne(modelRegistryTable.state, "retired"),
      ),
    );

  const decisions: QuarantineDecision[] = [];
  for (const row of rows) {
    try {
      const d = await evaluateRow(row, windowHours);
      decisions.push(d);
      if (d.decision === "quarantined" && !dryRun) {
        const fromState = row.state;
        await db.transaction(async (tx) => {
          await tx
            .update(modelRegistryTable)
            .set({
              state: "quarantined",
              updatedAt: new Date(),
              note:
                (row.note ? row.note + " | " : "") +
                `auto-quarantine: ${d.reasonCodes.join(",")}`,
            })
            .where(eq(modelRegistryTable.id, row.id));
          await tx.insert(quarantineEventsTable).values({
            registryId: row.id,
            fromState,
            toState: "quarantined",
            reasonCode: d.reasonCodes[0],
            triggeredBy: "auto",
            detail: d.detail,
          });
        });
        logger.warn(
          {
            registryId: row.id,
            modelId: row.modelId,
            modelVersion: row.modelVersion,
            reasons: d.reasonCodes,
          },
          "Auto-quarantine: model moved to quarantined",
        );
        // Task #444 — the LLM sidecar drift explainer was deleted along
        // with the rest of the LLM surface. Quarantine reasons live on
        // the registry row + the structured event we just inserted.
      }
    } catch (err) {
      logger.warn(
        {
          registryId: row.id,
          err: err instanceof Error ? err.message : String(err),
        },
        "Quarantine sweep: row evaluation failed",
      );
    }
  }
  // Post-sweep aggregation: re-derive the set of (coin, timeframe) slots
  // whose every active version is now quarantined (per-coin AND pooled),
  // so the operator gets a proactive alert instead of having to wait for
  // a `/ml/predict` 503.
  let starvedSlots: SlotStarvationAlert[] = [];
  try {
    starvedSlots = await recomputeSlotStarvationAlerts();
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "Quarantine sweep: slot-starvation aggregation failed",
    );
  }

  return {
    generatedAt: new Date().toISOString(),
    windowHours,
    dryRun,
    decisions,
    starvedSlots,
  };
}

export async function listRecentQuarantineEvents(limit = 50) {
  return db
    .select()
    .from(quarantineEventsTable)
    .orderBy(sql`${quarantineEventsTable.createdAt} DESC`)
    .limit(limit);
}
