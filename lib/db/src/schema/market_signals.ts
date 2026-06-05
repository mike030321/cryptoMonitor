import { pgTable, serial, text, timestamp, real, jsonb, index } from "drizzle-orm/pg-core";

/**
 * Task #271 — per-coin snapshots of external exchange signals registered
 * by the training contract (rule 5).
 *
 * One row per coin per poll cycle. Rows for the special pseudo-coin ids
 * `btc` and `eth` carry the cross-market reference price used to compute
 * the `btc_lead_ret_5m` / `eth_lead_ret_5m` features for every other coin.
 *
 * Columns are nullable so a partial fetch (e.g. funding succeeded but
 * OI request failed) still yields a usable row. The ml-engine treats
 * NULL/missing as the registered safe default (zero) — see
 * `EXTERNAL_STREAM_DEFAULTS` in labels.py.
 *
 * Retention (Task #292): rows older than `MARKET_SIGNALS_RETENTION_DAYS`
 * (default 30) are pruned hourly by `market-signals-retention.ts`. The
 * trainer's lookback window is well inside that horizon; raise the env
 * var if you ever extend the lookback past ~25 days.
 */
export const marketSignalsTable = pgTable(
  "market_signals",
  {
    id: serial("id").primaryKey(),
    coinId: text("coin_id").notNull(),
    timestamp: timestamp("timestamp", { withTimezone: true }).notNull().defaultNow(),
    fundingRate: real("funding_rate"),                  // perp funding (fraction, e.g. 0.0001 = 1bps)
    openInterestUsd: real("open_interest_usd"),         // notional OI in USD
    liquidations1hUsd: real("liquidations_1h_usd"),     // sum of large-trade USD over trailing 1h
    bidAskSpreadBps: real("bid_ask_spread_bps"),        // top-of-book spread in basis points
    midPrice: real("mid_price"),                        // (bid + ask) / 2 — used for BTC/ETH lead returns
    source: text("source"),                             // e.g. "okx_swap" or "okx_swap+gate_swap"
    // Task #286 — per-source USD breakdown for the aggregated
    // `liquidations_1h_usd` field, e.g. {"okx": 12345.6, "gate": 23456.7}.
    // Lets us audit which exchange contributed what and detect aggregator
    // outages from the data itself. NULL for older rows / non-liquidation
    // snapshots.
    sourceBreakdown: jsonb("source_breakdown").$type<Record<string, number>>(),
  },
  (t) => ({
    byCoinTs: index("market_signals_coin_ts_idx").on(t.coinId, t.timestamp),
    byTs: index("market_signals_ts_idx").on(t.timestamp),
  }),
);

export type MarketSignalRow = typeof marketSignalsTable.$inferSelect;
