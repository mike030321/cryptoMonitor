import { pgTable, text, serial, timestamp, integer, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const monitoringStateTable = pgTable("monitoring_state", {
  id: serial("id").primaryKey(),
  isRunning: boolean("is_running").notNull().default(false),
  cycleCount: integer("cycle_count").notNull().default(0),
  lastCycleAt: timestamp("last_cycle_at", { withTimezone: true }),
  nextCycleAt: timestamp("next_cycle_at", { withTimezone: true }),
});

export const insertMonitoringStateSchema = createInsertSchema(monitoringStateTable).omit({ id: true });
export type InsertMonitoringState = z.infer<typeof insertMonitoringStateSchema>;
export type MonitoringState = typeof monitoringStateTable.$inferSelect;
