// Export your models here. Add one export per file
// export * from "./posts";
//
// Each model/table should ideally be split into different files.
// Each model/table should define a Drizzle table, insert schema, and types:
//
//   import { pgTable, text, serial } from "drizzle-orm/pg-core";
//   import { createInsertSchema } from "drizzle-zod";
//   import { z } from "zod/v4";
//
//   export const postsTable = pgTable("posts", {
//     id: serial("id").primaryKey(),
//     title: text("title").notNull(),
//   });
//
//   export const insertPostSchema = createInsertSchema(postsTable).omit({ id: true });
//   export type InsertPost = z.infer<typeof insertPostSchema>;
//   export type Post = typeof postsTable.$inferSelect;

export * from "./agents";
export * from "./predictions";
export * from "./price_history";
export * from "./price_candles";
export * from "./monitoring";
export * from "./paper_trades";
export * from "./paper_position_marks";
export * from "./coin_insights";
export * from "./fingerprint_buffers";
// Task #444 — `evolution` (LLM-driven personality mutation log) is gone.
export * from "./strategy_lab";
export * from "./skip_events";
export * from "./auto_deploy_attribution_snapshots";
export * from "./tuning_gate_state";
export * from "./app_settings";
export * from "./model_predictions";
// Task #444 — `news_tags` (LLM news-classifier output) is gone.
export * from "./prediction_journal";
export * from "./trade_journal";
export * from "./journal_rollup";
export * from "./model_registry";
export * from "./feature_lab";
// Task #444 — the four `llm_*` sidecar tables are dropped along with the
// llm-sidecar directory. Schema files remain physically present until
// the next housekeeping pass; they are simply not exported so Drizzle
// won't re-create them.
export * from "./market_signals";
