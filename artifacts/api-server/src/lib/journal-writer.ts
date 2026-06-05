import { db, predictionJournalTable, tradeJournalTable, predictionsTable, paperTradesTable, skipEventsTable } from "@workspace/db";
import { eq, and, gte, lte, desc, inArray, sql } from "drizzle-orm";
import { logger } from "./logger";
import type { PatternAnalysis, TimeframeKey } from "./pattern-analyzer";
import type { QuantPayload } from "./quant-types";
import { getMlFeatures } from "./ml-client";

/**
 * Sources emitted by the quant brain (see ml-client `MlPredictResponse.source`
 * and quant-brain.ts fallbacks). Anything in this set means the row was
 * produced by the QUANT brain; everything else (including null) is the LLM.
 */
const QUANT_SOURCES = new Set(["lightgbm", "model", "prior", "stub", "unavailable", "untrained"]);

function classifyBrain(source: string | null | undefined): "LLM" | "QUANT" {
  if (!source) return "LLM";
  return QUANT_SOURCES.has(source) ? "QUANT" : "LLM";
}

/**
 * Look back over the last `windowMs` skip_events for this (agentId, coinId)
 * and return the most recent reason as a structured `{ skipReason,
 * gatesApplied }` patch the journal can record. Returns nulls when no
 * recent skip is found — caller should fall back to a generic "gated"
 * label so we never silently lose the row.
 *
 * This is the bridge that turns paper-trader's existing recordSkip()
 * telemetry into structured gate attribution on the journal row, so
 * downstream learners see WHICH gate fired (fee_gate_ev, daily_loss_limit,
 * counter_trend_regime, …) rather than an opaque "gated" string.
 */
export async function lookupGateForPrediction(
  agentId: number,
  coinId: string,
  predictionCreatedAt: Date,
  windowMs = 30_000,
): Promise<{ skipReason: string | null; gatesApplied: GatesApplied }> {
  try {
    const since = new Date(predictionCreatedAt.getTime() - windowMs);
    const rows = await db
      .select({ reason: skipEventsTable.reason, ts: skipEventsTable.ts })
      .from(skipEventsTable)
      .where(and(
        eq(skipEventsTable.agentId, agentId),
        eq(skipEventsTable.coinId, coinId),
        gte(skipEventsTable.ts, since),
      ))
      .orderBy(desc(skipEventsTable.ts))
      .limit(1);
    if (rows.length === 0) return { skipReason: null, gatesApplied: {} };
    const reason = rows[0].reason;
    return { skipReason: reason, gatesApplied: { [reason]: true } };
  } catch (err) {
    logger.debug({ err: String(err), agentId, coinId }, "journal-writer: gate lookup failed");
    return { skipReason: null, gatesApplied: {} };
  }
}

/**
 * Phase 1 — Journal write helpers.
 *
 * Centralised so the prediction-orchestrator path, the paper-trader, and
 * the backtester all produce identical row shapes. Failures here MUST NOT
 * affect trading behavior — every helper is wrapped to swallow errors and
 * emit a logger.warn so a journal hiccup never aborts a real cycle.
 */

// Phase 4 — meta-model fields (non-boolean) live in the same JSON
// payload because they are written by the same orchestrator pass that
// fills the gate booleans, and the read-side endpoints consume them
// straight from `gates_applied`. Typing them explicitly keeps the
// fast-path call sites in `monitor.ts` honest.
export type QuantMetaGate = import("./quant-types").QuantMetaGate;
export type GatesApplied = {
  noTradeZone?: boolean;
  feeGateTpFloor?: boolean;
  feeGateEv?: boolean;
  regimeFilter?: boolean;
  confidenceBelowThreshold?: boolean;
  consecutiveLossesHalt?: boolean;
  drawdownHalt?: boolean;
  riskRecheckHalt?: boolean;
  dailyLossLimit?: boolean;
  insufficientCash?: boolean;
  duplicatePosition?: boolean;
  maxOpenPositions?: boolean;
  bothModelsDown?: boolean;
  singleModelPenalty?: boolean;
  metaGate?: QuantMetaGate;
  meta_version?: string;
  meta_kind?: string;
  meta_abstain_reason?: string;
  meta_expected_edge_pct?: number;
  meta_size_multiplier?: number;
  meta_action?: "long" | "short" | "no_trade" | "abstain";
  // free-form for future gates
  [k: string]: boolean | string | number | QuantMetaGate | undefined;
};

export interface InsertPredictionJournalArgs {
  predictionId: number | null;
  /**
   * Task #405 / B-LLM-AUTHORSHIP — the journal writer enforces the
   * "LLM cannot author trades" contract. The orchestrator returns
   * `brain: "ABSTAIN"` when no quant model is available; the monitor
   * caller MUST forward that brand-name through here so the row is
   * stamped honestly. Any non-QUANT brain that arrives carrying a
   * `direction !== "stable"` is refused at write time (see
   * `writePredictionJournal`).
   */
  brain: "LLM" | "QUANT" | "BACKTEST" | "ABSTAIN";
  /**
   * When set, the writer will fetch the canonical feature vector + hash
   * from the Python ml-engine /ml/features endpoint after the row is
   * inserted and patch them in. This makes Python the single source of
   * truth for journaled features regardless of which brain (LLM or QUANT)
   * produced the prediction. Optional and fire-and-forget — failures here
   * never affect trading or the row's existence.
   */
  refreshFeaturesFor?: { coinId: string; timeframe: TimeframeKey } | null;
  agentId: number | null;
  agentName: string | null;
  coinId: string;
  coinName: string | null;
  timeframe: string;
  modelId: string | null;
  modelVersion: string | null;
  source: string | null;
  featureHash: string | null;
  featureVector: Record<string, number> | null;
  regimeLabel: string | null;
  direction: "up" | "down" | "stable";
  confidence: number;
  rawConfidence: number | null;
  probUp: number | null;
  probDown: number | null;
  probStable: number | null;
  expectedReturnPct: number | null;
  predictionStdPct: number | null;
  priceAtPrediction: number;
  predictedPrice: number | null;
  gatesApplied: GatesApplied;
  /**
   * Phase 4 — promoted out of `gates_applied`. Per-specialist views of
   * this bar (probUp/probDown/etc per kind) for diagnostics. Pass
   * `null` for LLM rows or when the specialist ensemble didn't run.
   */
  specialistScores: unknown | null;
  becameTrade: boolean | null;
  skipReason: string | null;
  tradeId: number | null;
  resolvesAt: Date | null;
  /**
   * Phase 5 — model registry linkage. When the prediction came from a
   * non-champion entry in `model_registry` (state=shadow|challenger), set
   * `shadow=true` and `registryId` to that row's id so the promotion gate
   * evaluator can score it without trading on it.
   */
  shadow?: boolean | null;
  registryId?: number | null;
}

export async function writePredictionJournal(
  args: InsertPredictionJournalArgs,
): Promise<number | null> {
  // Task #405 / B-LLM-AUTHORSHIP — runtime guard. The "LLM cannot
  // author trades" contract means no non-QUANT brain may EVER stamp a
  // directional (`up`/`down`) row in the journal. If a caller forgot
  // to coerce direction to "stable" on the abstain path, refuse the
  // write loudly and skip the row — better an audible miss than a
  // silently-laundered LLM-authored trade intent. BACKTEST rows come
  // from the Python backtester whose direction is its own ground-truth
  // research signal and is allowed to be directional.
  if (
    (args.brain === "LLM" || args.brain === "ABSTAIN") &&
    args.direction !== "stable"
  ) {
    logger.error(
      {
        brain: args.brain, direction: args.direction,
        coinId: args.coinId, timeframe: args.timeframe,
        agentId: args.agentId, source: args.source,
      },
      "journal-writer: refused non-QUANT directional row (LLM-authorship guard)",
    );
    return null;
  }
  // Task #405 / B-QUANT-FEATHASH — every QUANT row should carry a
  // feature_hash so downstream replay can reconstruct the exact feature
  // vector the model saw, and learning loops can de-duplicate "same bar,
  // same features" rows. The orchestrator + ml-engine /ml/predict are
  // both expected to attach one (see Task #460 — `featureHash` was added
  // to the Python /ml/predict response so the live trading path always
  // forwards a real hash).
  //
  // BUT: silently dropping the row when the upstream omits it (the
  // previous behaviour) hid every freshly-retrained 1d/6h QUANT
  // prediction from the journal during the Task #406 brain flip — the
  // dashboard kept rendering them while the audit trail had no record.
  // We now NEVER drop a QUANT row over a missing hash. Instead:
  //   1. synthesize a clearly-labelled placeholder
  //      (`missing:{source}:{modelVersion}:{coinId}:{timeframe}`) so the
  //      row still has a stable, traceable identifier;
  //   2. log a single warn (not error) so the upstream contract
  //      violation is still visible without flooding logs at error
  //      severity every cycle;
  //   3. let `refreshFeaturesFor` (when the caller passes it — the live
  //      monitor.ts path always does) overwrite the placeholder with
  //      the canonical hash from /ml/features post-insert.
  let effectiveFeatureHash = args.featureHash;
  if (args.brain === "QUANT" && !effectiveFeatureHash) {
    effectiveFeatureHash = `missing:${args.source ?? "unknown"}:${args.modelVersion ?? "unknown"}:${args.coinId}:${args.timeframe}`;
    logger.warn(
      {
        brain: args.brain, direction: args.direction,
        coinId: args.coinId, timeframe: args.timeframe,
        agentId: args.agentId, source: args.source,
        modelId: args.modelId, modelVersion: args.modelVersion,
        synthesizedFeatureHash: effectiveFeatureHash,
        willRefresh: Boolean(args.refreshFeaturesFor),
      },
      "journal-writer: QUANT row missing feature_hash — wrote with synthesized placeholder",
    );
  }
  try {
    const [row] = await db.insert(predictionJournalTable).values({
      predictionId: args.predictionId,
      brain: args.brain,
      agentId: args.agentId,
      agentName: args.agentName,
      coinId: args.coinId,
      coinName: args.coinName,
      timeframe: args.timeframe,
      modelId: args.modelId,
      modelVersion: args.modelVersion,
      source: args.source,
      featureHash: effectiveFeatureHash,
      featureVector: args.featureVector,
      regimeLabel: args.regimeLabel,
      direction: args.direction,
      confidence: args.confidence,
      rawConfidence: args.rawConfidence,
      probUp: args.probUp,
      probDown: args.probDown,
      probStable: args.probStable,
      expectedReturnPct: args.expectedReturnPct,
      predictionStdPct: args.predictionStdPct,
      priceAtPrediction: args.priceAtPrediction,
      predictedPrice: args.predictedPrice,
      gatesApplied: args.gatesApplied,
      specialistScores: args.specialistScores,
      becameTrade: args.becameTrade,
      skipReason: args.skipReason,
      tradeId: args.tradeId,
      resolvesAt: args.resolvesAt,
      shadow: args.shadow ?? false,
      registryId: args.registryId ?? null,
    }).returning({ id: predictionJournalTable.id });

    // Single Python source of truth for journaled features. Fire-and-forget;
    // never blocks the caller and never throws into the trading path.
    if (args.refreshFeaturesFor) {
      const { coinId, timeframe } = args.refreshFeaturesFor;
      const insertedId = row.id;
      void getMlFeatures(coinId, timeframe)
        .then(async (resp) => {
          if (!resp.features && !resp.featureHash) return;
          await db.update(predictionJournalTable)
            .set({
              featureHash: resp.featureHash ?? null,
              featureVector: (resp.features as unknown as Record<string, number> | null) ?? null,
            })
            .where(eq(predictionJournalTable.id, insertedId));
        })
        .catch((err) => {
          logger.debug(
            { err: String(err), coinId, timeframe },
            "journal-writer: ml-features refresh skipped",
          );
        });
    }

    return row.id;
  } catch (err) {
    logger.warn({ err, predictionId: args.predictionId }, "journal-writer: prediction insert failed");
    return null;
  }
}

export async function markPredictionJournalTrade(
  predictionId: number | null,
  patch: {
    becameTrade: boolean;
    skipReason?: string | null;
    tradeId?: number | null;
    /**
     * Structured per-gate booleans merged into the existing gatesApplied
     * jsonb column. Use this to record explicit gate decisions (e.g.
     * fee_gate_ev, daily_loss_limit) so downstream learners can replay
     * counterfactuals instead of guessing from a generic "gated" string.
     */
    gatesApplied?: GatesApplied;
  },
): Promise<void> {
  if (!predictionId) return;
  // Idempotent + race-safe. If the prediction_journal row hasn't been
  // committed yet (insert is in flight), the UPDATE will affect zero
  // rows. Retry a few times with backoff so the trade/skip status
  // ALWAYS lands. The row is keyed by predictionId (unique 1:1 with
  // predictionsTable.id), so retrying is safe.
  const maxAttempts = 4;
  const backoffMs = [25, 75, 200, 500];
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      let mergedGates: GatesApplied | null = null;
      if (patch.gatesApplied && Object.keys(patch.gatesApplied).length > 0) {
        // Phase 5 — must scope to the LIVE row (shadow=false). With shadow
        // rows now sharing the same predictionId, an unscoped read could
        // return a shadow row's gates and merge them into the live row,
        // contaminating the trade-decision audit trail.
        const [existing] = await db
          .select({ gatesApplied: predictionJournalTable.gatesApplied })
          .from(predictionJournalTable)
          .where(
            and(
              eq(predictionJournalTable.predictionId, predictionId),
              eq(predictionJournalTable.shadow, false),
            ),
          )
          .limit(1);
        mergedGates = {
          ...((existing?.gatesApplied as GatesApplied | null) ?? {}),
          ...patch.gatesApplied,
        };
      }
      // Phase 5 — restrict to the LIVE row (shadow=false). Shadow rows
      // share the same predictionId but represent challenger-eyes-only
      // entries that never trade and must not be marked with becameTrade
      // /tradeId, otherwise they contaminate the promotion gate sample.
      const updated = await db.update(predictionJournalTable)
        .set({
          becameTrade: patch.becameTrade,
          skipReason: patch.skipReason ?? null,
          tradeId: patch.tradeId ?? null,
          ...(mergedGates ? { gatesApplied: mergedGates } : {}),
        })
        .where(
          and(
            eq(predictionJournalTable.predictionId, predictionId),
            eq(predictionJournalTable.shadow, false),
          ),
        )
        .returning({ id: predictionJournalTable.id });
      if (updated.length > 0) return; // status landed
      // Row not yet present — back off and retry. After the final
      // attempt, log so the gap is visible in journal-health drift.
      if (attempt < maxAttempts - 1) {
        await new Promise((r) => setTimeout(r, backoffMs[attempt]));
        continue;
      }
      logger.warn(
        { predictionId, becameTrade: patch.becameTrade, skipReason: patch.skipReason },
        "journal-writer: status update found no journal row after retries",
      );
      return;
    } catch (err) {
      logger.warn({ err, predictionId, attempt }, "journal-writer: prediction update failed");
      return;
    }
  }
}

export async function resolvePredictionJournal(
  predictionId: number,
  patch: { actualPrice: number; outcome: string; realizedReturnPct: number; resolvedAt: Date },
): Promise<void> {
  try {
    // Phase 5 — resolve BOTH live and shadow rows for this predictionId.
    // The realized price/outcome is the same physical event regardless of
    // which model "saw" it, so champion + every challenger gets the same
    // ground-truth row. Their direction/confidence differ (when shadow
    // inference is wired), but the realized return is shared.
    await db.update(predictionJournalTable)
      .set({
        actualPrice: patch.actualPrice,
        outcome: patch.outcome,
        realizedReturnPct: patch.realizedReturnPct,
        resolvedAt: patch.resolvedAt,
      })
      .where(eq(predictionJournalTable.predictionId, predictionId));
  } catch (err) {
    logger.warn({ err, predictionId }, "journal-writer: resolve failed");
  }
}

export interface InsertTradeJournalArgs {
  tradeId: number;
  predictionId: number | null;
  agentId: number;
  agentName: string;
  coinId: string;
  coinName: string;
  timeframe: string;
  direction: string;
  entryTime: Date;
  exitTime: Date;
  entryPriceRaw: number | null;
  entryPriceAdj: number;
  exitPriceRaw: number | null;
  exitPriceAdj: number | null;
  entryFee: number | null;
  exitFee: number | null;
  slippagePct: number | null;
  positionSizeUsd: number;
  mfePct: number | null;
  maePct: number | null;
  exitReason: string;
  realizedPnlUsd: number | null;
  realizedPnlPct: number | null;
  /**
   * Phase 2 — 6-class regime label captured at the moment the trade is
   * journaled. Nullable so callers that haven't been migrated yet still
   * compile; a follow-up will require it once all sites pass it.
   */
  regimeLabel?: string | null;
}

export async function writeTradeJournal(args: InsertTradeJournalArgs): Promise<void> {
  try {
    // Look up the LIVE matching prediction_journal row id (if any) so the
    // row is fully joined on insert. Shadow/challenger rows can share the
    // same predictionId, but they did not authorize the live paper trade and
    // must not receive trade attribution.
    let predictionJournalId: number | null = null;
    if (args.predictionId !== null) {
      const [pj] = await db
        .select({ id: predictionJournalTable.id })
        .from(predictionJournalTable)
        .where(and(
          eq(predictionJournalTable.predictionId, args.predictionId),
          eq(predictionJournalTable.shadow, false),
        ))
        .limit(1);
      predictionJournalId = pj?.id ?? null;
    }

    const counterfactualBetter =
      args.realizedPnlUsd !== null && args.realizedPnlUsd <= 0;

    await db.insert(tradeJournalTable).values({
      tradeId: args.tradeId,
      predictionId: args.predictionId,
      predictionJournalId,
      agentId: args.agentId,
      agentName: args.agentName,
      coinId: args.coinId,
      coinName: args.coinName,
      timeframe: args.timeframe,
      direction: args.direction,
      entryTime: args.entryTime,
      exitTime: args.exitTime,
      entryPriceRaw: args.entryPriceRaw,
      entryPriceAdj: args.entryPriceAdj,
      exitPriceRaw: args.exitPriceRaw,
      exitPriceAdj: args.exitPriceAdj,
      entryFee: args.entryFee,
      exitFee: args.exitFee,
      slippagePct: args.slippagePct,
      positionSizeUsd: args.positionSizeUsd,
      mfePct: args.mfePct,
      maePct: args.maePct,
      exitReason: args.exitReason,
      realizedPnlUsd: args.realizedPnlUsd,
      realizedPnlPct: args.realizedPnlPct,
      regimeLabel: args.regimeLabel ?? null,
      counterfactualBetter,
    });
  } catch (err) {
    logger.warn(
      { err, tradeId: args.tradeId, predictionId: args.predictionId },
      "journal-writer: trade insert failed",
    );
  }
}

/**
 * Backfill helper — copy rows from legacy `predictions` and `paper_trades`
 * into the new journals. Idempotent: skips rows that already exist (keyed
 * by predictionId / tradeId).
 *
 * Returns counts so the admin endpoint can surface progress.
 */
export async function backfillJournals(): Promise<{
  predictionsInserted: number;
  tradesInserted: number;
}> {
  let predictionsInserted = 0;
  let tradesInserted = 0;

  // ----- one-shot Phase 4 promotion: copy any legacy
  // gates_applied.specialists / gates_applied.regime sidecar into the
  // dedicated specialist_scores / regime_label columns. Idempotent:
  // only touches rows where the destination column is still null and
  // the legacy key exists. Cleans the keys out of gates_applied so the
  // jsonb column converges on actual gating decisions only.
  try {
    await db.execute(sql`
      UPDATE prediction_journal
         SET specialist_scores = gates_applied -> 'specialists'
       WHERE specialist_scores IS NULL
         AND gates_applied ? 'specialists'
    `);
    await db.execute(sql`
      UPDATE prediction_journal
         SET regime_label = gates_applied ->> 'regime'
       WHERE regime_label IS NULL
         AND gates_applied ? 'regime'
    `);
    await db.execute(sql`
      UPDATE prediction_journal
         SET gates_applied = gates_applied - 'specialists' - 'regime'
       WHERE gates_applied ?| array['specialists','regime']
    `);
  } catch (err) {
    logger.warn({ err }, "backfillJournals: phase-4 sidecar promotion failed");
  }

  // ----- predictions -> prediction_journal -----
  const existingPj = await db
    .select({ predictionId: predictionJournalTable.predictionId })
    .from(predictionJournalTable);
  const existingPjSet = new Set(
    existingPj.map((r) => r.predictionId).filter((id): id is number => id !== null),
  );

  const allPredictions = await db.select().from(predictionsTable);
  for (const p of allPredictions) {
    if (existingPjSet.has(p.id)) continue;

    const ctx = (p.patternContext ?? null) as
      | (Partial<PatternAnalysis> & { regime?: string | null; quant?: Partial<QuantPayload> | null })
      | null;
    const quant = ctx?.quant ?? null;

    const realizedReturnPct =
      p.actualPrice !== null && p.actualPrice !== undefined && p.priceAtPrediction
        ? ((p.actualPrice - p.priceAtPrediction) / p.priceAtPrediction) * 100
        : null;

    try {
      await db.insert(predictionJournalTable).values({
        predictionId: p.id,
        brain: classifyBrain(p.source),
        agentId: p.agentId,
        agentName: p.agentName,
        coinId: p.coinId,
        coinName: p.coinName,
        timeframe: p.timeframe,
        modelId: p.source ?? "llm",
        modelVersion: quant?.modelVersion ?? null,
        source: p.source,
        featureHash: quant?.featureHash ?? null,
        featureVector: null, // not retained on legacy rows
        regimeLabel: ctx?.regime ?? null,
        direction: p.direction,
        confidence: p.confidence,
        rawConfidence: p.rawConfidence ?? null,
        probUp: quant?.probUp ?? null,
        probDown: quant?.probDown ?? null,
        probStable: quant?.probStable ?? null,
        expectedReturnPct: quant?.expectedReturnPct ?? null,
        predictionStdPct: quant?.predictionStdPct ?? null,
        priceAtPrediction: p.priceAtPrediction,
        predictedPrice: p.predictedPrice,
        gatesApplied: {},
        becameTrade: null, // unknown from legacy data; updated via trade backfill below
        skipReason: null,
        tradeId: null,
        resolvesAt: p.resolvesAt,
        actualPrice: p.actualPrice,
        realizedReturnPct,
        outcome: p.outcome,
        resolvedAt: p.resolvedAt,
        createdAt: p.createdAt,
      });
      predictionsInserted++;
    } catch (err) {
      logger.warn({ err, predictionId: p.id }, "backfillJournals: prediction insert failed");
    }
  }

  // ----- paper_trades -> trade_journal -----
  const existingTj = await db
    .select({ tradeId: tradeJournalTable.tradeId })
    .from(tradeJournalTable);
  const existingTjSet = new Set(
    existingTj.map((r) => r.tradeId).filter((id): id is number => id !== null),
  );

  const allTrades = await db.select().from(paperTradesTable);
  for (const t of allTrades) {
    if (existingTjSet.has(t.id)) continue;
    if (t.status === "open") continue; // skip open positions — journal them at close

    let predictionJournalId: number | null = null;
    if (t.predictionId !== null && t.predictionId !== undefined) {
      const [pj] = await db
        .select({ id: predictionJournalTable.id })
        .from(predictionJournalTable)
        .where(and(
          eq(predictionJournalTable.predictionId, t.predictionId),
          eq(predictionJournalTable.shadow, false),
        ))
        .limit(1);
      predictionJournalId = pj?.id ?? null;

      // Mark the prediction as having become a trade.
      try {
        await db.update(predictionJournalTable)
          .set({ becameTrade: true, tradeId: t.id })
          .where(and(
            eq(predictionJournalTable.predictionId, t.predictionId),
            eq(predictionJournalTable.shadow, false),
          ));
      } catch { /* non-critical */ }
    }

    const exitReason =
      t.status === "cancelled"
        ? "anomaly-cancel"
        : t.pnl !== null && t.pnl > 0
          ? "take-profit-or-trailing"
          : "stop-loss-or-expired";

    const counterfactualBetter = t.pnl !== null && t.pnl <= 0;

    try {
      await db.insert(tradeJournalTable).values({
        tradeId: t.id,
        predictionId: t.predictionId,
        predictionJournalId,
        agentId: t.agentId,
        agentName: t.agentName,
        coinId: t.coinId,
        coinName: t.coinName,
        timeframe: t.timeframe,
        direction: t.action === "buy" ? "up" : "down",
        entryTime: t.createdAt,
        exitTime: t.closedAt ?? t.createdAt,
        entryPriceRaw: null,
        entryPriceAdj: t.entryPrice,
        exitPriceRaw: null,
        exitPriceAdj: t.exitPrice,
        entryFee: t.entryFee,
        exitFee: null,
        slippagePct: null,
        positionSizeUsd: t.positionSize,
        mfePct: null,
        maePct: null,
        exitReason,
        realizedPnlUsd: t.pnl,
        realizedPnlPct: t.pnlPercent,
        // Phase 2 backfill: legacy paper_trades didn't store a regime
        // label, so we leave this null. A separate one-shot backfill job
        // can derive it from the prediction's stored patternContext.
        regimeLabel: null,
        counterfactualBetter,
      });
      tradesInserted++;
    } catch (err) {
      logger.warn({ err, tradeId: t.id }, "backfillJournals: trade insert failed");
    }
  }

  return { predictionsInserted, tradesInserted };
}

/**
 * Backtest journal entry — produced by the Python backtester (ml-engine
 * `app/backtest/`) and POSTed to /crypto/journal/backtest-batch. Keeping
 * a dedicated helper means the row shape stays in lockstep with the live
 * QUANT/LLM rows even though the backtester runs out-of-process.
 *
 * brain is hard-coded to "BACKTEST" so the diagnostics card and any
 * downstream learners can filter simulated rows out (or in) cleanly.
 */
export interface InsertBacktestJournalArgs {
  coinId: string;
  timeframe: string;
  modelId: string | null;
  modelVersion: string | null;
  featureHash: string | null;
  featureVector: Record<string, number> | null;
  regimeLabel: string | null;
  direction: "up" | "down" | "stable";
  confidence: number;
  probUp: number | null;
  probDown: number | null;
  probStable: number | null;
  expectedReturnPct: number | null;
  predictionStdPct: number | null;
  priceAtPrediction: number;
  predictedPrice: number | null;
  actualPrice: number | null;
  realizedReturnPct: number | null;
  outcome: string | null;
  resolvesAt: Date | null;
  resolvedAt: Date | null;
  gatesApplied: GatesApplied;
  /**
   * Phase 4 — promoted out of `gates_applied`. Optional sidecar so
   * older backtest payloads (which never set it) keep working.
   */
  specialistScores?: unknown | null;
}

/**
 * Optional simulated-trade payload riding alongside the prediction row. When
 * present we ALSO insert a `trade_journal` entry so the backtester contributes
 * to MAE/MFE/fees coverage just like the live paper-trader does.
 */
export interface BacktestSimulatedTrade {
  entryTime: Date;
  exitTime: Date;
  entryPriceRaw: number | null;
  entryPriceAdj: number;
  exitPriceRaw: number | null;
  exitPriceAdj: number;
  entryFee: number | null;
  exitFee: number | null;
  slippagePct: number | null;
  positionSizeUsd: number;
  mfePct: number | null;
  maePct: number | null;
  exitReason: string;
  realizedPnlUsd: number;
  realizedPnlPct: number;
}

export async function writeBacktestJournalRows(
  rows: (InsertBacktestJournalArgs & { simulatedTrade?: BacktestSimulatedTrade | null })[],
): Promise<{ predictionsInserted: number; tradesInserted: number }> {
  if (rows.length === 0) return { predictionsInserted: 0, tradesInserted: 0 };
  let predictionsInserted = 0;
  let tradesInserted = 0;
  // Keys we don't want to mistake for the "real" gate reason when picking
  // a structured skipReason out of gatesApplied.
  const META_GATE_KEYS = new Set(["backtest", "detail", "noTradeZone"]);
  for (const r of rows) {
    const becameTrade = !!r.simulatedTrade;
    // For SKIP rows, propagate the structured gate reason into the
    // first-class `skipReason` column so it can be queried/learned from
    // directly (matching live `lookupGateForPrediction` semantics). The
    // reason is the first truthy gate key in `gatesApplied` that isn't a
    // metadata key. Falls back to "backtest_skipped" only when no
    // structured reason was provided.
    let resolvedSkipReason: string | null = null;
    if (!becameTrade) {
      const gates = (r.gatesApplied ?? {}) as Record<string, unknown>;
      for (const [key, val] of Object.entries(gates)) {
        if (META_GATE_KEYS.has(key)) continue;
        if (val === true) { resolvedSkipReason = key; break; }
      }
      resolvedSkipReason = resolvedSkipReason ?? "backtest_skipped";
    }
    let predictionJournalId: number | null = null;
    try {
      const [inserted] = await db.insert(predictionJournalTable).values({
        predictionId: null,
        brain: "BACKTEST",
        agentId: null,
        agentName: null,
        coinId: r.coinId,
        coinName: null,
        timeframe: r.timeframe,
        modelId: r.modelId,
        modelVersion: r.modelVersion,
        source: r.modelId,
        featureHash: r.featureHash,
        featureVector: r.featureVector,
        regimeLabel: r.regimeLabel,
        direction: r.direction,
        confidence: r.confidence,
        rawConfidence: null,
        probUp: r.probUp,
        probDown: r.probDown,
        probStable: r.probStable,
        expectedReturnPct: r.expectedReturnPct,
        predictionStdPct: r.predictionStdPct,
        priceAtPrediction: r.priceAtPrediction,
        predictedPrice: r.predictedPrice,
        gatesApplied: r.gatesApplied,
        specialistScores: r.specialistScores ?? null,
        // Truthful semantics: simulated executed trades are trades. Only
        // predictions with no simulatedTrade payload are recorded as
        // skipped (e.g. when the backtest's gating rule rejected entry).
        becameTrade,
        skipReason: becameTrade ? null : resolvedSkipReason,
        tradeId: null,
        resolvesAt: r.resolvesAt,
        actualPrice: r.actualPrice,
        realizedReturnPct: r.realizedReturnPct,
        outcome: r.outcome,
        resolvedAt: r.resolvedAt,
      }).returning({ id: predictionJournalTable.id });
      predictionsInserted++;
      predictionJournalId = inserted.id;
    } catch (err) {
      logger.warn({ err, coinId: r.coinId, timeframe: r.timeframe }, "journal-writer: backtest prediction insert failed");
      continue;
    }

    if (r.simulatedTrade) {
      const t = r.simulatedTrade;
      const counterfactualBetter = t.realizedPnlUsd <= 0;
      try {
        await db.insert(tradeJournalTable).values({
          tradeId: null,
          predictionId: null,
          predictionJournalId,
          agentId: null,
          agentName: "backtest",
          coinId: r.coinId,
          coinName: r.coinId,
          timeframe: r.timeframe,
          direction: r.direction,
          entryTime: t.entryTime,
          exitTime: t.exitTime,
          entryPriceRaw: t.entryPriceRaw,
          entryPriceAdj: t.entryPriceAdj,
          exitPriceRaw: t.exitPriceRaw,
          exitPriceAdj: t.exitPriceAdj,
          entryFee: t.entryFee,
          exitFee: t.exitFee,
          slippagePct: t.slippagePct,
          positionSizeUsd: t.positionSizeUsd,
          mfePct: t.mfePct,
          maePct: t.maePct,
          exitReason: t.exitReason,
          realizedPnlUsd: t.realizedPnlUsd,
          realizedPnlPct: t.realizedPnlPct,
          counterfactualBetter,
        });
        tradesInserted++;
      } catch (err) {
        logger.warn({ err, coinId: r.coinId, timeframe: r.timeframe }, "journal-writer: backtest trade insert failed");
      }
    }
  }
  return { predictionsInserted, tradesInserted };
}

export interface JournalHealth {
  windowHours: number;
  predictions: {
    total: number;
    perDay: number;
    resolved: number;
    resolvedPct: number;
    becameTrade: number;
    becameTradePct: number;
  };
  trades: {
    total: number;
    perDay: number;
    withMaeMfe: number;
    maeMfeCoveragePct: number;
    withFees: number;
    feesCoveragePct: number;
  };
  features: {
    pythonCalls: number;
    pythonStaleCalls: number;
    tsFallbackCalls: number;
    pythonPct: number;
  };
  /**
   * Task #470 — operator surface for synthesized feature_hash placeholders
   * written by {@link writePredictionJournal} when a QUANT row arrives
   * without an upstream `featureHash`. Always counted over the last hour
   * (independent of `windowHours`) so a fresh contract regression in the
   * predictor lights up immediately instead of being diluted across a 24h
   * or 7d aggregate. `total > threshold` is the soft-alert trigger the
   * dashboard renders as a warning banner.
   */
  synthesizedFeatureHashes: {
    windowHours: number;
    total: number;
    threshold: number;
    byKey: Array<{
      source: string;
      timeframe: string;
      count: number;
      lastSeen: string;
    }>;
  };
}

/**
 * Bucketed time-series companion to {@link getJournalHealth}. Each bucket
 * reports predictions written, resolution coverage %, and MAE/MFE coverage %
 * for a fixed-width slice of the selected window so the diagnostics card can
 * draw a sparkline of how the journal has trended (rather than just the
 * latest aggregate snapshot). Bucket size is chosen by the caller so the
 * point count stays sane (~12-30 buckets per window).
 */
export interface JournalHealthSeriesPoint {
  bucketStart: string; // ISO timestamp at start of bucket
  predictions: number;
  resolvedPct: number;
  trades: number;
  maeMfeCoveragePct: number;
}

export interface JournalHealthSeries {
  windowHours: number;
  bucketSeconds: number;
  points: JournalHealthSeriesPoint[];
}

export async function getJournalHealthSeries(
  windowHours: number,
  bucketSeconds: number,
): Promise<JournalHealthSeries> {
  const since = new Date(Date.now() - windowHours * 3_600_000);
  const bucketExpr = sql`to_timestamp(floor(extract(epoch from created_at) / ${bucketSeconds}) * ${bucketSeconds})`;

  const pjRows = (await db.execute(sql`
    SELECT
      ${bucketExpr} AS bucket,
      COUNT(*)::int AS total,
      COUNT(*) FILTER (
        WHERE outcome IS NOT NULL AND outcome::text <> 'pending'
      )::int AS resolved
    FROM prediction_journal
    WHERE created_at >= ${since}
    GROUP BY 1
    ORDER BY 1
  `)) as unknown as { rows: Array<{ bucket: string | Date; total: number; resolved: number }> };

  const tjRows = (await db.execute(sql`
    SELECT
      ${bucketExpr} AS bucket,
      COUNT(*)::int AS total,
      COUNT(*) FILTER (
        WHERE mfe_pct IS NOT NULL AND mae_pct IS NOT NULL
      )::int AS with_mae_mfe
    FROM trade_journal
    WHERE created_at >= ${since}
    GROUP BY 1
    ORDER BY 1
  `)) as unknown as { rows: Array<{ bucket: string | Date; total: number; with_mae_mfe: number }> };

  const pjMap = new Map<number, { total: number; resolved: number }>();
  for (const r of pjRows.rows) {
    const ts = new Date(r.bucket).getTime();
    pjMap.set(ts, { total: Number(r.total), resolved: Number(r.resolved) });
  }
  const tjMap = new Map<number, { total: number; withMaeMfe: number }>();
  for (const r of tjRows.rows) {
    const ts = new Date(r.bucket).getTime();
    tjMap.set(ts, { total: Number(r.total), withMaeMfe: Number(r.with_mae_mfe) });
  }

  // Generate every bucket in the window so gaps render as zeros, not as a
  // skipped X tick. Aligns to the same floor() boundary used in SQL.
  const bucketMs = bucketSeconds * 1000;
  const nowAligned = Math.floor(Date.now() / bucketMs) * bucketMs;
  const startAligned = Math.floor(since.getTime() / bucketMs) * bucketMs;
  const points: JournalHealthSeriesPoint[] = [];
  for (let t = startAligned; t <= nowAligned; t += bucketMs) {
    const pj = pjMap.get(t) ?? { total: 0, resolved: 0 };
    const tj = tjMap.get(t) ?? { total: 0, withMaeMfe: 0 };
    points.push({
      bucketStart: new Date(t).toISOString(),
      predictions: pj.total,
      resolvedPct: pj.total > 0 ? (pj.resolved / pj.total) * 100 : 0,
      trades: tj.total,
      maeMfeCoveragePct: tj.total > 0 ? (tj.withMaeMfe / tj.total) * 100 : 0,
    });
  }

  return { windowHours, bucketSeconds, points };
}

/**
 * Task #469 — one-shot backfill for QUANT predictions silently dropped from
 * `prediction_journal` between the Task #406 brain flip and the Task #460
 * fix.
 *
 * Until Task #460, the journal writer returned `null` whenever a QUANT row
 * arrived without a `feature_hash`. Every freshly-retrained 1d/6h LightGBM
 * prediction emitted during the brain flip hit that path and disappeared
 * from the journal silently — the dashboard kept rendering them while the
 * audit trail had a gap. Task #460 fixed the path forward (synthesized
 * placeholder + warn) but the historical hole remains.
 *
 * This helper re-walks the legacy `predictions` table for QUANT-source rows
 * (`source IN QUANT_SOURCES`) that have no matching `prediction_journal`
 * entry, fetches the canonical feature vector + hash from `/ml/features`
 * once per (coinId, timeframe), and inserts a journal row preserving the
 * original `created_at` so the audit/replay surface (abstain-rate
 * denominators, journal-health drift dashboard) shows continuous coverage
 * during the affected window.
 *
 * Idempotent: only inserts rows whose `predictionId` isn't already present
 * in `prediction_journal`. Re-running on a healthy journal is a no-op.
 *
 * Best-effort on hash recovery: `/ml/features` returns the LATEST feature
 * vector for the coin/timeframe, not the historical bar that produced the
 * original prediction. That's the price of having no cached vector for the
 * dropped rows — the row at least gets a real, current hash that's
 * traceable. When the call fails or returns null, we fall back to the same
 * `missing:{source}:{modelVersion}:{coinId}:{timeframe}` placeholder the
 * live writer uses (Task #460), so the row still exists and is greppable.
 */
export interface BackfillMissingQuantJournalsOptions {
  /** Earliest predictions.created_at to consider. Omitted = no lower bound. */
  since?: Date;
  /** Latest predictions.created_at to consider. Omitted = no upper bound. */
  until?: Date;
  /**
   * Cap on rows processed per call. Default: 5000, max 50000. Rows are
   * scanned newest-first; for windows containing more than `limit`
   * QUANT predictions, narrow the window via `until` and re-run rather
   * than raising the cap, so the call stays bounded.
   */
  limit?: number;
}

export interface BackfillMissingQuantJournalsResult {
  /** QUANT predictions inspected in the window. */
  scanned: number;
  /** Skipped because a prediction_journal row already exists. */
  alreadyJournaled: number;
  /** Journal rows inserted by this call. */
  inserted: number;
  /** Rows where the insert threw and the gap remains. */
  failed: number;
  /** Distinct (coinId, timeframe) keys touched. */
  uniqueKeys: number;
  /** Keys where /ml/features returned a usable hash. */
  featuresHydrated: number;
  /** Keys where /ml/features failed or returned null (placeholder used). */
  featuresFailed: number;
}

export async function backfillMissingQuantJournals(
  opts: BackfillMissingQuantJournalsOptions = {},
): Promise<BackfillMissingQuantJournalsResult> {
  const limit = Math.min(Math.max(opts.limit ?? 5000, 1), 50_000);

  const conditions = [inArray(predictionsTable.source, Array.from(QUANT_SOURCES))];
  if (opts.since) conditions.push(gte(predictionsTable.createdAt, opts.since));
  if (opts.until) conditions.push(lte(predictionsTable.createdAt, opts.until));

  const candidates = await db
    .select()
    .from(predictionsTable)
    .where(and(...conditions))
    .orderBy(desc(predictionsTable.createdAt))
    .limit(limit);

  // Pre-fetch existing journal entries for the candidate set so the gap
  // detection is O(1) per row regardless of journal size.
  const candidateIds = candidates.map((p) => p.id);
  const existingPjSet = new Set<number>();
  if (candidateIds.length > 0) {
    const rows = await db
      .select({ predictionId: predictionJournalTable.predictionId })
      .from(predictionJournalTable)
      .where(inArray(predictionJournalTable.predictionId, candidateIds));
    for (const r of rows) {
      if (r.predictionId !== null) existingPjSet.add(r.predictionId);
    }
  }

  const featuresCache = new Map<
    string,
    { hash: string | null; vector: Record<string, number> | null }
  >();
  const cacheKey = (c: string, t: string) => `${c}::${t}`;

  let scanned = 0;
  let alreadyJournaled = 0;
  let inserted = 0;
  let failed = 0;
  let featuresHydrated = 0;
  let featuresFailed = 0;

  for (const p of candidates) {
    scanned += 1;
    if (existingPjSet.has(p.id)) {
      alreadyJournaled += 1;
      continue;
    }

    const ctx = (p.patternContext ?? null) as
      | (Partial<PatternAnalysis> & { regime?: string | null; quant?: Partial<QuantPayload> | null })
      | null;
    const quant = ctx?.quant ?? null;

    // One /ml/features call per (coinId, timeframe) — recovers the
    // canonical hash + vector from the single Python source of truth.
    const key = cacheKey(p.coinId, p.timeframe);
    let recovered = featuresCache.get(key);
    if (!recovered) {
      try {
        const resp = await getMlFeatures(p.coinId, p.timeframe as TimeframeKey);
        recovered = {
          hash: resp.featureHash ?? null,
          vector: (resp.features as unknown as Record<string, number> | null) ?? null,
        };
        if (recovered.hash) featuresHydrated += 1;
        else featuresFailed += 1;
      } catch (err) {
        logger.debug(
          { err: String(err), coinId: p.coinId, timeframe: p.timeframe },
          "backfillMissingQuantJournals: ml-features fetch failed",
        );
        recovered = { hash: null, vector: null };
        featuresFailed += 1;
      }
      featuresCache.set(key, recovered);
    }

    const modelVersion = quant?.modelVersion ?? null;
    // Mirror the live writer's contract (Task #460): a real hash when we
    // have one, the cached on-prediction hash from patternContext.quant
    // as a second-best, and the same labelled placeholder as the live
    // path when neither is available — never silently drop the row.
    const featureHash =
      recovered.hash
      ?? quant?.featureHash
      ?? `missing:${p.source ?? "unknown"}:${modelVersion ?? "unknown"}:${p.coinId}:${p.timeframe}`;

    const realizedReturnPct =
      p.actualPrice !== null && p.actualPrice !== undefined && p.priceAtPrediction
        ? ((p.actualPrice - p.priceAtPrediction) / p.priceAtPrediction) * 100
        : null;

    try {
      await db.insert(predictionJournalTable).values({
        predictionId: p.id,
        brain: classifyBrain(p.source),
        agentId: p.agentId,
        agentName: p.agentName,
        coinId: p.coinId,
        coinName: p.coinName,
        timeframe: p.timeframe,
        modelId: p.source ?? "lightgbm",
        modelVersion,
        source: p.source,
        featureHash,
        featureVector: recovered.vector,
        regimeLabel: ctx?.regime ?? null,
        direction: p.direction,
        confidence: p.confidence,
        rawConfidence: p.rawConfidence ?? null,
        probUp: quant?.probUp ?? null,
        probDown: quant?.probDown ?? null,
        probStable: quant?.probStable ?? null,
        expectedReturnPct: quant?.expectedReturnPct ?? null,
        predictionStdPct: quant?.predictionStdPct ?? null,
        priceAtPrediction: p.priceAtPrediction,
        predictedPrice: p.predictedPrice,
        gatesApplied: {},
        becameTrade: null,
        skipReason: null,
        tradeId: null,
        resolvesAt: p.resolvesAt,
        actualPrice: p.actualPrice,
        realizedReturnPct,
        outcome: p.outcome,
        resolvedAt: p.resolvedAt,
        // Preserve the original prediction timestamp so the journal-health
        // sparkline + abstain-rate denominators see the row in its actual
        // historical bucket, not when this backfill ran.
        createdAt: p.createdAt,
      });
      inserted += 1;
    } catch (err) {
      failed += 1;
      logger.warn(
        { err, predictionId: p.id, coinId: p.coinId, timeframe: p.timeframe },
        "backfillMissingQuantJournals: insert failed",
      );
    }
  }

  return {
    scanned,
    alreadyJournaled,
    inserted,
    failed,
    uniqueKeys: featuresCache.size,
    featuresHydrated,
    featuresFailed,
  };
}

export async function getJournalHealth(windowHours = 24): Promise<JournalHealth> {
  const since = new Date(Date.now() - windowHours * 3_600_000);

  // SQL aggregates with a time-window predicate so the endpoint stays
  // O(window) instead of O(table). Indexed on created_at.
  const [pjAgg] = await db
    .select({
      total: sql<number>`count(*)::int`,
      resolved: sql<number>`count(*) filter (where ${predictionJournalTable.outcome} is not null and ${predictionJournalTable.outcome} <> 'pending')::int`,
      becameTrade: sql<number>`count(*) filter (where ${predictionJournalTable.becameTrade} = true)::int`,
    })
    .from(predictionJournalTable)
    .where(gte(predictionJournalTable.createdAt, since));

  const total = pjAgg?.total ?? 0;
  const resolved = pjAgg?.resolved ?? 0;
  const becameTrade = pjAgg?.becameTrade ?? 0;

  const [tjAgg] = await db
    .select({
      total: sql<number>`count(*)::int`,
      withMaeMfe: sql<number>`count(*) filter (where ${tradeJournalTable.mfePct} is not null and ${tradeJournalTable.maePct} is not null)::int`,
      withFees: sql<number>`count(*) filter (where ${tradeJournalTable.entryFee} is not null and ${tradeJournalTable.exitFee} is not null)::int`,
    })
    .from(tradeJournalTable)
    .where(gte(tradeJournalTable.createdAt, since));

  const totalTrades = tjAgg?.total ?? 0;
  const withMaeMfe = tjAgg?.withMaeMfe ?? 0;
  const withFees = tjAgg?.withFees ?? 0;

  const days = Math.max(windowHours / 24, 0.0001);
  // Lazy-imported to avoid a circular dep with pattern-analyzer.ts.
  const { getFeatureSourceTelemetry } = await import("./pattern-analyzer.js");
  const featureTel = getFeatureSourceTelemetry();

  // Task #470 — synthesized feature_hash diagnostic. Pinned to the last
  // hour regardless of `windowHours` so a freshly-broken predictor
  // contract surfaces immediately. Healthy operation = 0 rows; any
  // non-zero count means writePredictionJournal wrote at least one
  // QUANT row whose upstream forgot to send a featureHash.
  const synthWindowHours = 1;
  const synthSince = new Date(Date.now() - synthWindowHours * 3_600_000);
  const synthRows = (await db.execute(sql`
    SELECT
      coalesce(source, 'unknown') AS source,
      timeframe,
      COUNT(*)::int AS count,
      MAX(created_at) AS last_seen
    FROM prediction_journal
    WHERE created_at >= ${synthSince}
      AND feature_hash LIKE 'missing:%'
    GROUP BY 1, 2
    ORDER BY count DESC, last_seen DESC
  `)) as unknown as {
    rows: Array<{
      source: string;
      timeframe: string;
      count: number;
      last_seen: string | Date;
    }>;
  };
  const synthByKey = synthRows.rows.map((r) => ({
    source: r.source,
    timeframe: r.timeframe,
    count: Number(r.count),
    lastSeen: new Date(r.last_seen).toISOString(),
  }));
  const synthTotal = synthByKey.reduce((s, r) => s + r.count, 0);

  return {
    windowHours,
    predictions: {
      total,
      perDay: Math.round(total / days),
      resolved,
      resolvedPct: total > 0 ? (resolved / total) * 100 : 0,
      becameTrade,
      becameTradePct: total > 0 ? (becameTrade / total) * 100 : 0,
    },
    trades: {
      total: totalTrades,
      perDay: Math.round(totalTrades / days),
      withMaeMfe,
      maeMfeCoveragePct: totalTrades > 0 ? (withMaeMfe / totalTrades) * 100 : 0,
      withFees,
      feesCoveragePct: totalTrades > 0 ? (withFees / totalTrades) * 100 : 0,
    },
    features: {
      pythonCalls: featureTel.python,
      pythonStaleCalls: featureTel.pythonStale,
      tsFallbackCalls: featureTel.tsFallback,
      pythonPct: featureTel.pythonPct,
    },
    synthesizedFeatureHashes: {
      windowHours: synthWindowHours,
      total: synthTotal,
      // Soft-alert at 5/hr — small enough to catch a single stuck
      // (source, timeframe) pair, large enough to absorb the occasional
      // one-off race (e.g. fresh deploy, ml-engine restart) without
      // crying wolf.
      threshold: 5,
      byKey: synthByKey,
    },
  };
}
