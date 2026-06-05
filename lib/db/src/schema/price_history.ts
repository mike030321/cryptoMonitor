import {
  pgTable,
  text,
  serial,
  timestamp,
  real,
  boolean,
  check,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

/**
 * Per-tick close-only price store. Contract: every row is a 1m-cadence
 * tick from the live poller (or a 1m close-only bar from the historical
 * backfill). Aggregated bars at any other cadence belong in
 * `price_candles` (see `schema-audit.md` and task #343).
 *
 * The `cadence` column + CHECK constraint is the database-level safety
 * net for that contract (task #347). Application writers still call
 * `assertNativeCadence(...)` / `assert_native_cadence(...)` as the
 * defense-in-depth first line — but a future ETL script, manual SQL
 * session, or third service that bypasses those guards is now stopped at
 * the database boundary instead of silently corrupting the table.
 *
 * The column defaults to '1m', so existing writers and any caller that
 * does not set it explicitly produce a row that satisfies the constraint.
 * Any caller that explicitly sets cadence to anything else fails loud
 * (Postgres raises 23514 / check_violation).
 */
export const priceHistoryTable = pgTable(
  "price_history",
  {
    id: serial("id").primaryKey(),
    coinId: text("coin_id").notNull(),
    price: real("price").notNull(),
    timestamp: timestamp("timestamp", { withTimezone: true })
      .notNull()
      .defaultNow(),
    isSynthetic: boolean("is_synthetic").notNull().default(false),
    // Phase 2 — first-class regime label attached at write time. Nullable
    // so legacy rows (predating the regime classifier) stay valid until a
    // backfill job lands a label on them.
    regime: text("regime"),
    // Task #347 — DB-level cadence marker. Constrained below to '1m'.
    cadence: text("cadence").notNull().default("1m"),
  },
  (t) => ({
    cadenceIs1m: check(
      "price_history_cadence_is_1m",
      sql`${t.cadence} = '1m'`,
    ),
  }),
);

export const insertPriceHistorySchema = createInsertSchema(priceHistoryTable).omit({ id: true });
export type InsertPriceHistory = z.infer<typeof insertPriceHistorySchema>;
export type PriceHistory = typeof priceHistoryTable.$inferSelect;
