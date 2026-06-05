import { pgTable, serial, text, integer, timestamp, jsonb, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const skipEventsTable = pgTable(
  "skip_events",
  {
    id: serial("id").primaryKey(),
    ts: timestamp("ts", { withTimezone: true }).notNull().defaultNow(),
    reason: text("reason").notNull(),
    agentName: text("agent_name").notNull(),
    agentId: integer("agent_id"),
    coinId: text("coin_id"),
    message: text("message").notNull(),
    details: jsonb("details").notNull().default({}),
  },
  (t) => ({
    tsIdx: index("skip_events_ts_idx").on(t.ts),
    reasonIdx: index("skip_events_reason_idx").on(t.reason),
  }),
);

export const insertSkipEventSchema = createInsertSchema(skipEventsTable).omit({ id: true });
export type InsertSkipEvent = z.infer<typeof insertSkipEventSchema>;
export type SkipEventRow = typeof skipEventsTable.$inferSelect;
