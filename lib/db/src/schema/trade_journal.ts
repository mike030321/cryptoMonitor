import {
  pgTable,
  serial,
  text,
  integer,
  real,
  boolean,
  timestamp,
  index,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

/**
 * Phase 1 — Adaptive engine trade journal.
 *
 * One row per closed paper trade (or anomaly-cancel). Joined to its
 * prediction via `predictionId` (matches predictions.id and the legacy
 * paper_trades.predictionId column) and `predictionJournalId` for the
 * Phase-1 journal row. Captures everything needed to score adaptive
 * strategies in later phases: full price + fee + slippage breakdown,
 * MAE / MFE for risk analysis, exit reason taxonomy, realized PnL,
 * and a counterfactual flag indicating whether abstaining would have
 * been better.
 */
export const tradeJournalTable = pgTable(
  "trade_journal",
  {
    id: serial("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),

    // Identification
    tradeId: integer("trade_id"),                  // FK to paper_trades.id
    predictionId: integer("prediction_id"),        // FK to predictions.id
    predictionJournalId: integer("prediction_journal_id"), // FK to prediction_journal.id
    agentId: integer("agent_id"),
    agentName: text("agent_name"),
    coinId: text("coin_id").notNull(),
    coinName: text("coin_name"),
    timeframe: text("timeframe").notNull(),
    direction: text("direction").notNull(),

    // Entry / exit
    entryTime: timestamp("entry_time", { withTimezone: true }).notNull(),
    exitTime: timestamp("exit_time", { withTimezone: true }),
    entryPriceRaw: real("entry_price_raw"),       // pre-slippage (best estimate)
    entryPriceAdj: real("entry_price_adj").notNull(), // slippage-adjusted (= paper_trades.entryPrice)
    exitPriceRaw: real("exit_price_raw"),         // pre-slippage exit price
    exitPriceAdj: real("exit_price_adj"),         // slippage-adjusted exit

    // Fees / slippage
    entryFee: real("entry_fee"),
    exitFee: real("exit_fee"),
    slippagePct: real("slippage_pct"),
    positionSizeUsd: real("position_size_usd").notNull(),

    // Excursion stats (% of entry price). Long: MFE = (peakHigh-entry)/entry,
    // MAE = (entry-peakLow)/entry. Short: signs flip.
    mfePct: real("mfe_pct"),
    maePct: real("mae_pct"),

    // Outcome
    exitReason: text("exit_reason"), // "take-profit" | "stop-loss" | "expired" | "trailing-stop" | "anomaly-cancel"
    realizedPnlUsd: real("realized_pnl_usd"),
    realizedPnlPct: real("realized_pnl_pct"),

    // Phase 2 — regime label captured at trade time (one of the 6-class
    // labels emitted by ml-engine /ml/regime). Nullable for backfill of
    // historical rows that pre-date the regime classifier.
    regimeLabel: text("regime_label"),

    // True iff the realised outcome was worse than abstaining (i.e. PnL <= 0
    // net of fees on a position that was opened). Lets the meta-model in
    // Phase 4 score abstain decisions against actual losses.
    counterfactualBetter: boolean("counterfactual_better"),
  },
  (t) => ({
    createdAtIdx: index("trade_journal_created_at_idx").on(t.createdAt),
    predictionIdIdx: index("trade_journal_prediction_id_idx").on(t.predictionId),
    tradeIdIdx: index("trade_journal_trade_id_idx").on(t.tradeId),
    coinTimeframeIdx: index("trade_journal_coin_timeframe_idx").on(t.coinId, t.timeframe),
  }),
);

export const insertTradeJournalSchema = createInsertSchema(tradeJournalTable).omit({
  id: true,
  createdAt: true,
});
export type InsertTradeJournal = z.infer<typeof insertTradeJournalSchema>;
export type TradeJournalRow = typeof tradeJournalTable.$inferSelect;
