import { pgTable, serial, integer, real, text, timestamp, boolean } from "drizzle-orm/pg-core";
import { agentsTable } from "./agents";

export const strategySnapshotsTable = pgTable("strategy_snapshots", {
  id: serial("id").primaryKey(),
  strategyType: text("strategy_type").notNull(),
  equity: real("equity").notNull(),
  cashBalance: real("cash_balance").notNull(),
  investedValue: real("invested_value").notNull(),
  timestamp: timestamp("timestamp", { withTimezone: true }).notNull().defaultNow(),
});

export const strategyStateTable = pgTable("strategy_state", {
  agentId: integer("agent_id").primaryKey().references(() => agentsTable.id),
  peakValue: real("peak_value").notNull().default(1000),
  circuitBreakerActive: boolean("circuit_breaker_active").notNull().default(false),
  circuitBreakerActivatedAt: timestamp("circuit_breaker_activated_at", { withTimezone: true }),
  lastBuyAt: timestamp("last_buy_at", { withTimezone: true }),
  totalFees: real("total_fees").notNull().default(0),
  initialDeployDone: boolean("initial_deploy_done").notNull().default(false),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const strategySettingsTable = pgTable("strategy_settings", {
  strategyType: text("strategy_type").primaryKey(),
  dcaDrawdownTriggerPct: real("dca_drawdown_trigger_pct").notNull().default(20),
  dcaResumeLookbackDays: integer("dca_resume_lookback_days").notNull().default(14),
  dcaCycleDeployUsd: real("dca_cycle_deploy_usd").notNull().default(33.33),
  dcaBuyIntervalHours: integer("dca_buy_interval_hours").notNull().default(24),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const strategySettingsHistoryTable = pgTable("strategy_settings_history", {
  id: serial("id").primaryKey(),
  strategyType: text("strategy_type").notNull(),
  timestamp: timestamp("timestamp", { withTimezone: true }).notNull().defaultNow(),
  drawdownTriggerBefore: real("drawdown_trigger_before"),
  drawdownTriggerAfter: real("drawdown_trigger_after"),
  resumeLookbackBefore: integer("resume_lookback_before"),
  resumeLookbackAfter: integer("resume_lookback_after"),
  cycleDeployBefore: real("cycle_deploy_before"),
  cycleDeployAfter: real("cycle_deploy_after"),
});

export type StrategySnapshot = typeof strategySnapshotsTable.$inferSelect;
export type StrategyState = typeof strategyStateTable.$inferSelect;
export type StrategySettings = typeof strategySettingsTable.$inferSelect;
export type StrategySettingsHistory = typeof strategySettingsHistoryTable.$inferSelect;
