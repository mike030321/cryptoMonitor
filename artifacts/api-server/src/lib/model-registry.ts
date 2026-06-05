/**
 * Phase 5 — Model registry lifecycle (DB-backed).
 *
 * State machine:
 *   shadow      — predictions journaled with `shadow=true`, never trades.
 *   challenger  — eligible for promotion once gates pass.
 *   champion    — actively serves live trade decisions. At most one per
 *                 (model_id, coin_id, timeframe) slot.
 *   quarantined — manually disabled after a runtime failure / drift event.
 *   retired     — previously a champion, now rolled back; preserved for
 *                 audit and as a rollback target.
 *
 * Promote and rollback are one-click: promote evaluates the gates via the
 * ml-engine `/ml/registry/evaluate-promotion` endpoint (so the gate logic
 * lives in ONE place — Python `registry_lifecycle.py`), then atomically
 * demotes the current champion to "retired" and elevates the challenger to
 * "champion". Rollback is unconditional and restores the previous champion.
 */
import {
  db,
  modelRegistryTable,
  predictionJournalTable,
  type ModelRegistryRow,
  type ModelRegistryState,
  MODEL_REGISTRY_STATES,
} from "@workspace/db";
import { and, eq, sql, desc } from "drizzle-orm";
import { logger } from "./logger";

const ML_BASE = () => process.env.ML_ENGINE_URL || "http://localhost:8000";

export interface RegistrySlot {
  modelId: string;
  coinId: string;
  timeframe: string;
}

function normalizeSlot(s: Partial<RegistrySlot>): RegistrySlot {
  return {
    modelId: s.modelId ?? "lightgbm",
    coinId: s.coinId ?? "*",
    timeframe: s.timeframe ?? "*",
  };
}

export async function listRegistry(): Promise<ModelRegistryRow[]> {
  return db
    .select()
    .from(modelRegistryTable)
    .orderBy(desc(modelRegistryTable.updatedAt));
}

export async function registerModel(args: {
  modelId: string;
  modelVersion: string;
  coinId?: string;
  timeframe?: string;
  state?: ModelRegistryState;
  note?: string;
  metricsSnapshot?: unknown;
}): Promise<ModelRegistryRow> {
  const slot = normalizeSlot(args);
  const state: ModelRegistryState = args.state ?? "shadow";
  if (!MODEL_REGISTRY_STATES.includes(state)) {
    throw new Error(`invalid state: ${state}`);
  }
  const [row] = await db
    .insert(modelRegistryTable)
    .values({
      modelId: slot.modelId,
      modelVersion: args.modelVersion,
      coinId: slot.coinId,
      timeframe: slot.timeframe,
      state,
      note: args.note ?? null,
      metricsSnapshot: (args.metricsSnapshot as object | undefined) ?? null,
      isActive: true,
      promotedAt: state === "champion" ? new Date() : null,
    })
    .returning();
  return row;
}

export async function getCurrentChampion(
  slot: Partial<RegistrySlot>,
): Promise<ModelRegistryRow | null> {
  const s = normalizeSlot(slot);
  const rows = await db
    .select()
    .from(modelRegistryTable)
    .where(
      and(
        eq(modelRegistryTable.modelId, s.modelId),
        eq(modelRegistryTable.coinId, s.coinId),
        eq(modelRegistryTable.timeframe, s.timeframe),
        eq(modelRegistryTable.state, "champion"),
        eq(modelRegistryTable.isActive, true),
      ),
    )
    .limit(1);
  return rows[0] ?? null;
}

/** Aggregate resolved shadow journal rows for a registry entry into the
 * metrics the promotion-gate evaluator needs. Net edge here is the mean
 * realized_return_pct (already a percent), with zero-cost approximation —
 * the simulator's friction model is the precise version. For the live
 * promotion gate this aggregate is the right magnitude and is cheap. */
export async function summarizeShadowMetrics(
  registryId: number,
): Promise<{
  samples: number;
  netEdgePct: number;
  drawdownPct: number;
  perRegimeNetEdgePct: Record<string, number>;
}> {
  // Phase 5 — for the champion, the comparable baseline is its NON-shadow
  // resolved predictions (the rows that actually drove live trades). For
  // challengers / shadows, the baseline is their SHADOW rows. Looking up
  // the registry row's own state lets us pick the right filter so that
  // promotion edge-lift compares like-for-like.
  const slot = (
    await db
      .select({ state: modelRegistryTable.state })
      .from(modelRegistryTable)
      .where(eq(modelRegistryTable.id, registryId))
      .limit(1)
  )[0];
  const isChampion = slot?.state === "champion";
  const rows = await db
    .select({
      realizedReturnPct: predictionJournalTable.realizedReturnPct,
      regime: predictionJournalTable.regimeLabel,
      direction: predictionJournalTable.direction,
      outcome: predictionJournalTable.outcome,
    })
    .from(predictionJournalTable)
    .where(
      and(
        eq(predictionJournalTable.registryId, registryId),
        isChampion
          ? eq(predictionJournalTable.shadow, false)
          : eq(predictionJournalTable.shadow, true),
        sql`${predictionJournalTable.outcome} IS NOT NULL AND ${predictionJournalTable.outcome} <> 'pending'`,
      ),
    );
  let samples = 0;
  let edgeSum = 0;
  let equity = 100; // % units; arbitrary basis for drawdown
  let peak = equity;
  let maxDd = 0;
  const byRegime = new Map<string, { sum: number; n: number }>();
  for (const r of rows) {
    if (r.realizedReturnPct == null) continue;
    samples += 1;
    // Signed edge: realizedReturnPct already encodes the price move; multiply
    // by the call sign so an "up" prediction whose price went up = +edge,
    // and a "down" prediction whose price went down = +edge.
    const sign = r.direction === "up" ? 1 : r.direction === "down" ? -1 : 0;
    const edge = sign * r.realizedReturnPct;
    edgeSum += edge;
    equity *= 1 + edge / 100;
    if (equity > peak) peak = equity;
    const dd = peak > 0 ? (peak - equity) / peak : 0;
    if (dd > maxDd) maxDd = dd;
    const regimeKey = r.regime ?? "unknown";
    const acc = byRegime.get(regimeKey) ?? { sum: 0, n: 0 };
    acc.sum += edge;
    acc.n += 1;
    byRegime.set(regimeKey, acc);
  }
  const perRegimeNetEdgePct: Record<string, number> = {};
  for (const [k, v] of byRegime.entries()) {
    perRegimeNetEdgePct[k] = v.n > 0 ? v.sum / v.n : 0;
  }
  return {
    samples,
    netEdgePct: samples > 0 ? edgeSum / samples : 0,
    drawdownPct: maxDd * 100,
    perRegimeNetEdgePct,
  };
}

export class PromotionGatesFailedError extends Error {
  readonly code = "PROMOTION_GATES_FAILED" as const;
  constructor(readonly verdict: PromotionVerdict) {
    super(`promotion gates failed: ${verdict.reasons.join("; ")}`);
    this.name = "PromotionGatesFailedError";
  }
}

export interface PromotionVerdict {
  eligible: boolean;
  samplesOk: boolean;
  edgeLiftOk: boolean;
  drawdownOk: boolean;
  regimeRobustnessOk: boolean;
  reasons: string[];
  thresholds: Record<string, number>;
  metricsSummary: Record<string, unknown>;
}

async function callEvaluate(body: object): Promise<PromotionVerdict> {
  const res = await fetch(`${ML_BASE()}/ml/registry/evaluate-promotion`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`ml-engine evaluate-promotion ${res.status}: ${text.slice(0, 200)}`);
  }
  return (await res.json()) as PromotionVerdict;
}

/** Compute a promotion verdict for a single registry row WITHOUT mutating
 * state. UI uses this to render the per-gate pass/fail before the user
 * clicks "Promote". */
export async function dryRunPromotion(
  registryId: number,
): Promise<{ row: ModelRegistryRow; verdict: PromotionVerdict; champion: ModelRegistryRow | null }> {
  const [row] = await db
    .select()
    .from(modelRegistryTable)
    .where(eq(modelRegistryTable.id, registryId));
  if (!row) throw new Error(`registry row ${registryId} not found`);
  const champion = await getCurrentChampion({
    modelId: row.modelId,
    coinId: row.coinId,
    timeframe: row.timeframe,
  });
  const challenger = await summarizeShadowMetrics(registryId);
  const championMetrics = champion
    ? await summarizeShadowMetrics(champion.id)
    : { netEdgePct: 0, samples: 0, drawdownPct: 0, perRegimeNetEdgePct: {} };
  const verdict = await callEvaluate({
    samples: challenger.samples,
    netEdgePct: challenger.netEdgePct,
    championNetEdgePct: championMetrics.netEdgePct,
    drawdownPct: challenger.drawdownPct,
    perRegimeNetEdgePct: challenger.perRegimeNetEdgePct,
  });
  return { row, verdict, champion };
}

/** One-click promote. Re-evaluates the gate verdict, then atomically
 * demotes the current champion (if any) and elevates the row to champion.
 * Throws if the verdict is not eligible. */
export async function promoteToChampion(
  registryId: number,
  opts: { force?: boolean; note?: string } = {},
): Promise<ModelRegistryRow> {
  const { row, verdict, champion } = await dryRunPromotion(registryId);
  if (!verdict.eligible && !opts.force) {
    throw new PromotionGatesFailedError(verdict);
  }
  const now = new Date();
  return db.transaction(async (tx) => {
    if (champion) {
      await tx
        .update(modelRegistryTable)
        .set({
          state: "retired",
          demotedAt: now,
          updatedAt: now,
          isActive: true,
        })
        .where(eq(modelRegistryTable.id, champion.id));
    }
    const [updated] = await tx
      .update(modelRegistryTable)
      .set({
        state: "champion",
        promotedAt: now,
        updatedAt: now,
        previousChampionId: champion?.id ?? null,
        metricsSnapshot: { verdict },
        note: opts.note ?? row.note,
        isActive: true,
      })
      .where(eq(modelRegistryTable.id, registryId))
      .returning();
    logger.info(
      {
        registryId,
        modelId: updated.modelId,
        modelVersion: updated.modelVersion,
        previousChampionId: champion?.id,
        force: !!opts.force,
      },
      "Model promoted to champion",
    );
    return updated;
  });
}

/** Unconditional rollback: demotes the current champion to "retired" and
 * re-promotes its `previousChampionId` (if present and still active) to
 * "champion". Returns the rolled-back-to row, or null if no rollback target. */
export async function rollbackChampion(
  slot: Partial<RegistrySlot>,
  opts: { note?: string } = {},
): Promise<ModelRegistryRow | null> {
  const champion = await getCurrentChampion(slot);
  if (!champion) {
    throw new Error("no active champion to roll back");
  }
  const now = new Date();
  return db.transaction(async (tx) => {
    await tx
      .update(modelRegistryTable)
      .set({
        state: "retired",
        demotedAt: now,
        updatedAt: now,
        note: opts.note ?? champion.note,
      })
      .where(eq(modelRegistryTable.id, champion.id));
    if (!champion.previousChampionId) {
      logger.warn({ id: champion.id }, "Rollback: no previous champion");
      return null;
    }
    const [restored] = await tx
      .update(modelRegistryTable)
      .set({
        state: "champion",
        promotedAt: now,
        updatedAt: now,
        demotedAt: null,
      })
      .where(eq(modelRegistryTable.id, champion.previousChampionId))
      .returning();
    logger.info(
      { fromId: champion.id, toId: restored?.id },
      "Champion rolled back to previous version",
    );
    return restored ?? null;
  });
}

export async function setRegistryState(
  registryId: number,
  state: ModelRegistryState,
  note?: string,
): Promise<ModelRegistryRow> {
  if (!MODEL_REGISTRY_STATES.includes(state)) {
    throw new Error(`invalid state: ${state}`);
  }
  const [row] = await db
    .update(modelRegistryTable)
    .set({ state, updatedAt: new Date(), note: note ?? null })
    .where(eq(modelRegistryTable.id, registryId))
    .returning();
  return row;
}
