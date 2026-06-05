import { pgTable, text, serial, timestamp, integer, real, pgEnum } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { agentsTable } from "./agents";

export const tradeActionEnum = pgEnum("trade_action", ["buy", "sell", "close"]);

export const paperPortfoliosTable = pgTable("paper_portfolios", {
  id: serial("id").primaryKey(),
  agentId: integer("agent_id").notNull().references(() => agentsTable.id).unique(),
  agentName: text("agent_name").notNull(),
  cashBalance: real("cash_balance").notNull().default(100),
  totalValue: real("total_value").notNull().default(100),
  totalTrades: integer("total_trades").notNull().default(0),
  winningTrades: integer("winning_trades").notNull().default(0),
  losingTrades: integer("losing_trades").notNull().default(0),
  peakValue: real("peak_value").default(1000),
  dayStartValue: real("day_start_value").default(1000),
  dayStartDate: text("day_start_date"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const paperTradesTable = pgTable("paper_trades", {
  id: serial("id").primaryKey(),
  agentId: integer("agent_id").notNull().references(() => agentsTable.id),
  agentName: text("agent_name").notNull(),
  coinId: text("coin_id").notNull(),
  coinName: text("coin_name").notNull(),
  action: tradeActionEnum("action").notNull(),
  entryPrice: real("entry_price").notNull(),
  exitPrice: real("exit_price"),
  quantity: real("quantity").notNull(),
  positionSize: real("position_size").notNull(),
  entryFee: real("entry_fee"),
  pnl: real("pnl"),
  pnlPercent: real("pnl_percent"),
  timeframe: text("timeframe").notNull().default("5m"),
  predictionId: integer("prediction_id"),
  status: text("status").notNull().default("open"),
  closedAt: timestamp("closed_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const paperPositionsTable = pgTable("paper_positions", {
  id: serial("id").primaryKey(),
  agentId: integer("agent_id").notNull().references(() => agentsTable.id),
  agentName: text("agent_name").notNull().default("unknown"),
  coinId: text("coin_id").notNull(),
  coinName: text("coin_name").notNull(),
  direction: text("direction").notNull(),
  entryPrice: real("entry_price").notNull(),
  quantity: real("quantity").notNull(),
  positionSize: real("position_size").notNull(),
  timeframe: text("timeframe").notNull(),
  tradeId: integer("trade_id").notNull(),
  expiresAt: timestamp("expires_at", { withTimezone: true }).notNull(),
  stopLossPrice: real("stop_loss_price"),
  takeProfitPrice: real("take_profit_price"),
  peakPrice: real("peak_price"),
  // Phase 2 — 6-class regime label captured at decision/open time so the
  // trade journal records the regime that was active when the trade was
  // taken, not whatever happens to be cached at close. Carried through to
  // `trade_journal.regime_label` on close.
  entryRegimeLabel: text("entry_regime_label"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const insertPaperPortfolioSchema = createInsertSchema(paperPortfoliosTable).omit({ id: true, createdAt: true, updatedAt: true });
export const insertPaperTradeSchema = createInsertSchema(paperTradesTable).omit({ id: true, createdAt: true });
export const insertPaperPositionSchema = createInsertSchema(paperPositionsTable).omit({ id: true, createdAt: true });

export type PaperPortfolio = typeof paperPortfoliosTable.$inferSelect;
export type PaperTrade = typeof paperTradesTable.$inferSelect;
export type PaperPosition = typeof paperPositionsTable.$inferSelect;
