import { pgTable, text, serial, timestamp, integer, real, uniqueIndex } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";

// Task #444 — `marketContext` (jsonb) and `contextTags` (text) were the
// LLM-prompt cache columns; both are dropped now that the LLM brain is
// gone. The remaining columns feed regime-detector / contagion-detector.
export const coinInsightsTable = pgTable("coin_insights", {
  id: serial("id").primaryKey(),
  coinId: text("coin_id").notNull(),
  timeframe: text("timeframe").notNull(),
  patternType: text("pattern_type").notNull(),
  direction: text("direction").notNull(),
  outcome: text("outcome").notNull(),
  priceChangePercent: real("price_change_percent").notNull(),
  rsiAtPrediction: real("rsi_at_prediction"),
  macdSignal: text("macd_signal"),
  bbPercentB: real("bb_percent_b"),
  volatility: real("volatility"),
  agentId: integer("agent_id"),
  fingerprint: text("fingerprint"),
  regime: text("regime"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const coinCorrelationsTable = pgTable("coin_correlations", {
  id: serial("id").primaryKey(),
  coinA: text("coin_a").notNull(),
  coinB: text("coin_b").notNull(),
  correlation: real("correlation").notNull(),
  timeframe: text("timeframe").notNull(),
  sampleSize: integer("sample_size").notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("coin_correlations_pair_tf_idx").on(table.coinA, table.coinB, table.timeframe),
]);

export const insertCoinInsightSchema = createInsertSchema(coinInsightsTable).omit({ id: true, createdAt: true });
export const insertCoinCorrelationSchema = createInsertSchema(coinCorrelationsTable).omit({ id: true, updatedAt: true });

export type CoinInsight = typeof coinInsightsTable.$inferSelect;
export type CoinCorrelation = typeof coinCorrelationsTable.$inferSelect;
