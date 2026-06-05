import { pgTable, text, serial, timestamp, real, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { directionEnum, outcomeEnum } from "./predictions";

export const modelPredictionsTable = pgTable("model_predictions", {
  id: serial("id").primaryKey(),
  coinId: text("coin_id").notNull(),
  coinName: text("coin_name").notNull(),
  timeframe: text("timeframe").notNull(),
  modelVersion: text("model_version").notNull(),
  modelCoinId: text("model_coin_id").notNull(),
  featureHash: text("feature_hash"),
  probUp: real("prob_up").notNull(),
  probDown: real("prob_down").notNull(),
  probStable: real("prob_stable").notNull(),
  expectedReturnPct: real("expected_return_pct").notNull(),
  predictionStdPct: real("prediction_std_pct"),
  confidence: real("confidence").notNull(),
  modelDirection: directionEnum("model_direction").notNull(),
  priceAtPrediction: real("price_at_prediction").notNull(),
  resolvesAt: timestamp("resolves_at", { withTimezone: true }).notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  resolvedAt: timestamp("resolved_at", { withTimezone: true }),
  actualPrice: real("actual_price"),
  resolvedOutcomePct: real("resolved_outcome_pct"),
  outcome: outcomeEnum("outcome").notNull().default("pending"),
  graceT0Price: real("grace_t0_price"),
  graceT0PriceChange: real("grace_t0_price_change"),
  /**
   * Provenance reported by the ml-engine (`"prior" | "stub" | "model" |
   * "lightgbm"`). Lets shadow / accuracy aggregates filter the prior-only
   * pooled fallback out of the headline scoreboard. NULL for rows written
   * before this column existed.
   */
  source: text("source"),
}, (t) => ({
  resolvesAtIdx: index("model_predictions_resolves_at_idx").on(t.resolvesAt),
  tfIdx: index("model_predictions_tf_idx").on(t.timeframe),
  createdAtIdx: index("model_predictions_created_at_idx").on(t.createdAt),
}));

export const insertModelPredictionSchema = createInsertSchema(modelPredictionsTable).omit({ id: true, createdAt: true });
export type InsertModelPrediction = z.infer<typeof insertModelPredictionSchema>;
export type ModelPrediction = typeof modelPredictionsTable.$inferSelect;
