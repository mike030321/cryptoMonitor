import { pgTable, text, serial, timestamp, integer, real, pgEnum, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { agentsTable } from "./agents";

export const directionEnum = pgEnum("prediction_direction", ["up", "down", "stable"]);
export const outcomeEnum = pgEnum("prediction_outcome", ["correct", "wrong", "pending", "neutral"]);
export const timeframeEnum = pgEnum("prediction_timeframe", ["1m", "5m", "1h", "2h", "6h", "1d"]);

export const predictionsTable = pgTable("predictions", {
  id: serial("id").primaryKey(),
  agentId: integer("agent_id").notNull().references(() => agentsTable.id),
  agentName: text("agent_name").notNull(),
  coinId: text("coin_id").notNull(),
  coinName: text("coin_name").notNull(),
  direction: directionEnum("direction").notNull(),
  confidence: real("confidence").notNull(),
  reasoning: text("reasoning").notNull(),
  priceAtPrediction: real("price_at_prediction").notNull(),
  predictedPrice: real("predicted_price").notNull(),
  actualPrice: real("actual_price"),
  outcome: outcomeEnum("outcome").notNull().default("pending"),
  scoreChange: real("score_change"),
  timeframe: text("timeframe").notNull().default("1m"),
  resolvesAt: timestamp("resolves_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  resolvedAt: timestamp("resolved_at", { withTimezone: true }),
  patternContext: jsonb("pattern_context"),
  rawConfidence: real("raw_confidence"),
  graceT0Price: real("grace_t0_price"),
  graceT0PriceChange: real("grace_t0_price_change"),
  /**
   * Provenance of the model that produced this prediction. Populated for
   * QUANT-brain rows from `MlPredictResponse.source` (`"prior" | "stub" |
   * "model" | "lightgbm"`); NULL for legacy rows and for LLM-brain rows.
   *
   * Headline accuracy / P&L scoreboards filter `source = 'prior'` out by
   * default — the prior-only pooled fallback returns the same Laplace-
   * smoothed marginals every call, so counting it toward calibrated
   * accuracy would drag the headline number toward the empirical prior
   * and look like a regression.
   */
  source: text("source"),
});

export const insertPredictionSchema = createInsertSchema(predictionsTable).omit({ id: true, createdAt: true });
export type InsertPrediction = z.infer<typeof insertPredictionSchema>;
export type Prediction = typeof predictionsTable.$inferSelect;
