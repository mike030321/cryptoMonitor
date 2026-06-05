import { describe, it } from "node:test";
import assert from "node:assert/strict";
import * as schema from "@workspace/db/schema";
import { getTableColumns, getTableName, is } from "drizzle-orm";
import { PgTable } from "drizzle-orm/pg-core";

/**
 * Cadence-correctness contract for the price store.
 *
 * Today's `price_history` table stores a single column for `price` with no
 * native-cadence / source / timeframe column. Once task #306's CMC-daily
 * and OKX-hourly backfill modules land, real (non-synthetic) daily and
 * hourly bars will be written into the same table as live-poll ticks. The
 * trainer's resampler (artifacts/ml-engine/app/features.py) walks the row
 * stream and assigns rows to buckets by `floor(ts / bucket_ms) * bucket_ms`
 * — it has no way to distinguish a daily bar from a 60-second tick, so a
 * 5m or 1h bucket can silently close at a daily-bar price.
 *
 * The repair (see artifacts/ml-engine/reports/20260423T000000Z-schema-audit.md)
 * is a separate `price_candles` table keyed by (coin_id, timeframe,
 * bucket_start) with `source` ('cmc' | 'okx' | 'live_poll' | 'synthetic').
 * The tests below assert the post-fix invariants. They are EXPECTED TO
 * FAIL on `main` today — the absence of these guarantees IS the problem
 * this task documents — and to start passing the moment task #317's
 * migration ships.
 */

interface PriceCandlesSchema {
  priceCandlesTable?: PgTable;
}

describe("price store cadence-correctness contract (expected-to-fail until #317 ships)", () => {
  it("price_candles table is registered in the Drizzle schema", () => {
    const candidate = (schema as PriceCandlesSchema).priceCandlesTable;
    assert.ok(
      candidate !== undefined,
      "schema.priceCandlesTable is missing — schema-audit.md proposes it; ship task #317's migration to satisfy this contract",
    );
    assert.ok(
      is(candidate, PgTable),
      "schema.priceCandlesTable must be a Drizzle pgTable",
    );
    assert.equal(getTableName(candidate), "price_candles");
  });

  it("price_candles carries the (coin_id, timeframe, bucket_start, source) columns", () => {
    const candidate = (schema as PriceCandlesSchema).priceCandlesTable;
    assert.ok(candidate !== undefined, "price_candles table missing (see test above)");
    const cols = getTableColumns(candidate);
    for (const required of ["coinId", "timeframe", "bucketStart", "source"]) {
      assert.ok(
        Object.prototype.hasOwnProperty.call(cols, required),
        `price_candles is missing column '${required}'`,
      );
    }
    // Verify the underlying SQL column name (not the Drizzle property name)
    // for coinId / bucketStart so the migration produces the snake_case
    // names the trainer's read path will look up.
    assert.equal(cols.coinId.name, "coin_id");
    assert.equal(cols.bucketStart.name, "bucket_start");
    assert.equal(cols.timeframe.name, "timeframe");
    assert.equal(cols.source.name, "source");
  });

  it("price_history retains a tick-only contract (no timeframe column leaking cadence info)", () => {
    // The post-fix contract is that price_history stays the raw-tick store
    // and price_candles owns aggregated bars. If someone instead bolts a
    // `timeframe` column onto price_history, this test fails so the choice
    // is forced into review.
    const cols = getTableColumns(schema.priceHistoryTable);
    assert.ok(
      !Object.prototype.hasOwnProperty.call(cols, "timeframe"),
      "price_history must not gain a 'timeframe' column — aggregated bars belong in price_candles per schema-audit.md",
    );
  });
});
