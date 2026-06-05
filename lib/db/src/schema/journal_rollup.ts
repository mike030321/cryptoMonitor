import {
  pgTable,
  serial,
  text,
  integer,
  real,
  timestamp,
  date,
  uniqueIndex,
  index,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

/**
 * Phase 1 — Daily rollup of `prediction_journal` and `trade_journal` rows
 * older than the retention window (default 90 days). The retention job
 * aggregates raw rows into one row per
 * (bucketDate, kind, brain, coinId, timeframe, direction) and then deletes
 * the source rows so storage doesn't grow forever.
 *
 * The recent-window queries (journal-health, dashboards) keep reading the
 * raw tables; only operational/historical queries hit this rollup.
 */
export const journalRollupTable = pgTable(
  "journal_rollup",
  {
    id: serial("id").primaryKey(),
    rolledUpAt: timestamp("rolled_up_at", { withTimezone: true })
      .notNull()
      .defaultNow(),

    bucketDate: date("bucket_date").notNull(),
    kind: text("kind").notNull(), // "prediction" | "trade"
    brain: text("brain").notNull().default(""), // "LLM"|"QUANT"|"BACKTEST" or "" for trades
    coinId: text("coin_id").notNull(),
    timeframe: text("timeframe").notNull(),
    direction: text("direction").notNull().default(""), // "" when not applicable

    count: integer("count").notNull(),

    // prediction-only aggregates
    becameTradeCount: integer("became_trade_count"),
    resolvedCount: integer("resolved_count"),
    correctCount: integer("correct_count"),
    wrongCount: integer("wrong_count"),
    neutralCount: integer("neutral_count"),
    sumRealizedReturnPct: real("sum_realized_return_pct"),

    // trade-only aggregates
    sumRealizedPnlUsd: real("sum_realized_pnl_usd"),
    sumRealizedPnlPct: real("sum_realized_pnl_pct"),
    sumEntryFee: real("sum_entry_fee"),
    sumExitFee: real("sum_exit_fee"),
    withMaeMfeCount: integer("with_mae_mfe_count"),
    withFeesCount: integer("with_fees_count"),
  },
  (t) => ({
    uniq: uniqueIndex("journal_rollup_bucket_uniq").on(
      t.bucketDate,
      t.kind,
      t.brain,
      t.coinId,
      t.timeframe,
      t.direction,
    ),
    bucketDateIdx: index("journal_rollup_bucket_date_idx").on(t.bucketDate),
    kindIdx: index("journal_rollup_kind_idx").on(t.kind),
  }),
);

export const insertJournalRollupSchema = createInsertSchema(journalRollupTable).omit({
  id: true,
  rolledUpAt: true,
});
export type InsertJournalRollup = z.infer<typeof insertJournalRollupSchema>;
export type JournalRollupRow = typeof journalRollupTable.$inferSelect;
