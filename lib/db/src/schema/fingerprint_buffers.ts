import { pgTable, text, serial, timestamp, jsonb, uniqueIndex } from "drizzle-orm/pg-core";

export const fingerprintBuffersTable = pgTable("fingerprint_buffers", {
  id: serial("id").primaryKey(),
  coinId: text("coin_id").notNull(),
  timeframe: text("timeframe").notNull(),
  fingerprints: jsonb("fingerprints").notNull().default("[]"),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  uniqueIndex("fingerprint_buffers_coin_timeframe_idx").on(table.coinId, table.timeframe),
]);

export type FingerprintBuffer = typeof fingerprintBuffersTable.$inferSelect;
