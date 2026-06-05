import {
  pgTable,
  text,
  real,
  timestamp,
  uniqueIndex,
  index,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

/**
 * Aggregated OHLCV candles, written by historical-backfill modules
 * (CMC daily, OKX hourly) and any future native-cadence sources.
 *
 * `price_history` stays the raw-tick store for the live poller. Bars at
 * a known timeframe live here so a tick consumer cannot accidentally
 * read a daily bar through `price_history` (see schema-audit.md task
 * #315 / fix task #317).
 */
export const priceCandlesTable = pgTable(
  "price_candles",
  {
    coinId: text("coin_id").notNull(),
    timeframe: text("timeframe").notNull(), // "1m" | "5m" | "1h" | "2h" | "6h" | "1d"
    bucketStart: timestamp("bucket_start", { withTimezone: true }).notNull(),
    open: real("open").notNull(),
    high: real("high").notNull(),
    low: real("low").notNull(),
    close: real("close").notNull(),
    volume: real("volume"),
    source: text("source").notNull(), // "cmc" | "okx" | "coincap" | "live_poll" | "synthetic"
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (t) => ({
    pk: uniqueIndex("price_candles_pk_idx").on(
      t.coinId,
      t.timeframe,
      t.bucketStart,
    ),
    byCoinTf: index("price_candles_coin_tf_idx").on(
      t.coinId,
      t.timeframe,
      t.bucketStart,
    ),
  }),
);

export const insertPriceCandleSchema = createInsertSchema(priceCandlesTable);
export type InsertPriceCandle = z.infer<typeof insertPriceCandleSchema>;
export type PriceCandle = typeof priceCandlesTable.$inferSelect;
