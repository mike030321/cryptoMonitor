import {
  pgTable,
  serial,
  text,
  integer,
  real,
  boolean,
  timestamp,
  jsonb,
  index,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

/**
 * Phase 1 — Adaptive engine prediction journal.
 *
 * One row per prediction emitted (LLM or QUANT brain). Captures the full
 * audit trail every later phase reads from: feature vector + hash, predicted
 * class probs, predicted forward return, regime label (placeholder for
 * Phase 2), model id + version, timeframe, coin, decision gates that were
 * applied, whether the prediction became a trade, and resolution state.
 *
 * `predictionId` joins back to the legacy `predictions` table so dashboards
 * that already key off it keep working during the migration window.
 */
export const predictionJournalTable = pgTable(
  "prediction_journal",
  {
    id: serial("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),

    // Identification
    predictionId: integer("prediction_id"), // FK to legacy predictions.id (nullable for backtest rows)
    brain: text("brain").notNull(), // "LLM" | "QUANT" | "BACKTEST"
    agentId: integer("agent_id"),
    agentName: text("agent_name"),
    coinId: text("coin_id").notNull(),
    coinName: text("coin_name"),
    timeframe: text("timeframe").notNull(),

    // Model provenance
    modelId: text("model_id"),         // e.g. "lightgbm" / "llm:gpt-5-mini"
    modelVersion: text("model_version"),
    source: text("source"),            // matches predictions.source: prior|stub|model|lightgbm|llm

    // Feature snapshot
    featureHash: text("feature_hash"),
    featureVector: jsonb("feature_vector"), // raw {colName: value} dict
    regimeLabel: text("regime_label"),      // placeholder for Phase 2 regime model

    // Prediction outputs
    direction: text("direction").notNull(), // "up" | "down" | "stable"
    confidence: real("confidence").notNull(),
    rawConfidence: real("raw_confidence"),
    probUp: real("prob_up"),
    probDown: real("prob_down"),
    probStable: real("prob_stable"),
    expectedReturnPct: real("expected_return_pct"),
    predictionStdPct: real("prediction_std_pct"),
    priceAtPrediction: real("price_at_prediction").notNull(),
    predictedPrice: real("predicted_price"),

    // Decision gates that were applied (jsonb so the schema can grow without
    // migrations). Every later phase reads from this when scoring abstain
    // performance.
    gatesApplied: jsonb("gates_applied"), // { noTradeZone, regimeFilter, feeGateTp, feeGateEv, ... }

    // Phase 3 — observability-only sidecar. Per-regime ensemble's view of
    // this bar (one entry per specialist kind). Promoted out of
    // `gates_applied` (Phase 4) so the diagnostics page can compute
    // per-specialist per-regime accuracy via a typed jsonb column instead
    // of overloading a column whose name implies gating decisions.
    // Specialists do NOT gate live trades — the meta-model is the gate.
    specialistScores: jsonb("specialist_scores"),

    // Trade lifecycle
    becameTrade: boolean("became_trade"), // null=undecided, true=traded, false=skipped
    skipReason: text("skip_reason"),
    tradeId: integer("trade_id"),         // FK to paper_trades.id when traded

    // Phase 5 — champion/challenger lifecycle marker. When `shadow` is true,
    // this prediction was emitted by a model in `shadow`/`challenger` state
    // (model_registry.state) and is journaled for later evaluation but did
    // NOT influence the live trading decision. Resolved shadow rows feed
    // the promotion-gate evaluator. NULL on legacy rows; defaults false on
    // new inserts.
    shadow: boolean("shadow").default(false),
    registryId: integer("registry_id"),   // FK to model_registry.id (nullable)

    // Resolution (mirrors predictions row when resolved)
    resolvesAt: timestamp("resolves_at", { withTimezone: true }),
    resolvedAt: timestamp("resolved_at", { withTimezone: true }),
    actualPrice: real("actual_price"),
    realizedReturnPct: real("realized_return_pct"),
    outcome: text("outcome"), // "correct" | "wrong" | "neutral" | "pending"
  },
  (t) => ({
    createdAtIdx: index("prediction_journal_created_at_idx").on(t.createdAt),
    predictionIdIdx: index("prediction_journal_prediction_id_idx").on(t.predictionId),
    coinTimeframeIdx: index("prediction_journal_coin_timeframe_idx").on(t.coinId, t.timeframe),
    becameTradeIdx: index("prediction_journal_became_trade_idx").on(t.becameTrade),
    // Partial btree index to make the diagnostics specialists endpoint
    // (/crypto/brain/specialists) O(limit) instead of a sequential scan
    // over the full journal. Covers only rows where the typed
    // specialist_scores column is populated, ordered newest-first.
    specialistScoresCreatedAtIdx: index(
      "prediction_journal_specialist_scores_created_at_idx",
    )
      .on(t.createdAt.desc())
      .where(sql`${t.specialistScores} IS NOT NULL`),
  }),
);

export const insertPredictionJournalSchema = createInsertSchema(predictionJournalTable).omit({
  id: true,
  createdAt: true,
});
export type InsertPredictionJournal = z.infer<typeof insertPredictionJournalSchema>;
export type PredictionJournalRow = typeof predictionJournalTable.$inferSelect;
