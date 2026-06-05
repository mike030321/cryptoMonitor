import { pgTable, text, timestamp, doublePrecision } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const tuningGateStateTable = pgTable("tuning_gate_state", {
  gate: text("gate").primaryKey(),
  currentValue: doublePrecision("current_value").notNull(),
  belowBaselineSince: timestamp("below_baseline_since", { withTimezone: true }),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const insertTuningGateStateSchema = createInsertSchema(tuningGateStateTable);
export type InsertTuningGateState = z.infer<typeof insertTuningGateStateSchema>;
export type TuningGateState = typeof tuningGateStateTable.$inferSelect;
