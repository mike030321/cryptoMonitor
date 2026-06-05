import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { sql, eq } from "drizzle-orm";

import { db, marketSignalsTable } from "@workspace/db";
import {
  pruneMarketSignals,
  pruneMarketSignalsIfDue,
  getMarketSignalsRetentionDays,
  DEFAULT_MARKET_SIGNALS_RETENTION_DAYS,
} from "../src/lib/market-signals-retention";

// Task #292 — verify the market_signals retention sweep deletes rows
// older than the configured horizon and leaves fresh rows alone, plus
// that the env-var override is honoured and the throttling gate works.

const TEST_COIN = `__retention_test__${process.pid}_${Date.now()}`;

async function insertAt(ts: Date): Promise<void> {
  await db.insert(marketSignalsTable).values({
    coinId: TEST_COIN,
    timestamp: ts,
    midPrice: 1,
    source: "test",
  });
}

async function countTestRows(): Promise<number> {
  const rows = await db
    .select({ id: marketSignalsTable.id })
    .from(marketSignalsTable)
    .where(eq(marketSignalsTable.coinId, TEST_COIN));
  return rows.length;
}

async function cleanup(): Promise<void> {
  await db.execute(
    sql`DELETE FROM market_signals WHERE coin_id = ${TEST_COIN}`,
  );
}

describe("market-signals retention", () => {
  it("default horizon comes from DEFAULT_MARKET_SIGNALS_RETENTION_DAYS", () => {
    delete process.env["MARKET_SIGNALS_RETENTION_DAYS"];
    assert.equal(
      getMarketSignalsRetentionDays(),
      DEFAULT_MARKET_SIGNALS_RETENTION_DAYS,
    );
  });

  it("MARKET_SIGNALS_RETENTION_DAYS env var overrides the default", () => {
    process.env["MARKET_SIGNALS_RETENTION_DAYS"] = "7";
    try {
      assert.equal(getMarketSignalsRetentionDays(), 7);
    } finally {
      delete process.env["MARKET_SIGNALS_RETENTION_DAYS"];
    }
  });

  it("invalid env value falls back to the default", () => {
    process.env["MARKET_SIGNALS_RETENTION_DAYS"] = "not-a-number";
    try {
      assert.equal(
        getMarketSignalsRetentionDays(),
        DEFAULT_MARKET_SIGNALS_RETENTION_DAYS,
      );
    } finally {
      delete process.env["MARKET_SIGNALS_RETENTION_DAYS"];
    }
  });

  it("deletes rows older than the cutoff and keeps fresh rows", async () => {
    const now = Date.now();
    const oldTs = new Date(now - 10 * 24 * 60 * 60 * 1000); // 10d old
    const freshTs = new Date(now - 60 * 1000); // 1m old

    await cleanup();
    try {
      await insertAt(oldTs);
      await insertAt(oldTs);
      await insertAt(freshTs);
      assert.equal(await countTestRows(), 3);

      // 5-day retention: both old rows should go, fresh row should stay.
      const result = await pruneMarketSignals({ retentionDays: 5 });
      assert.equal(result.success, true);
      assert.equal(result.retentionDays, 5);
      assert.ok(
        result.deleted >= 2,
        `expected >=2 deletes, got ${result.deleted}`,
      );

      const remaining = await countTestRows();
      assert.equal(
        remaining,
        1,
        `expected exactly 1 fresh row to survive, got ${remaining}`,
      );
    } finally {
      await cleanup();
    }
  });

  it("exempts okx_backfill_* rows from retention (task #586)", async () => {
    const now = Date.now();
    const oldTs = new Date(now - 200 * 24 * 60 * 60 * 1000); // ~200d old

    await cleanup();
    try {
      // One row tagged as a backfill source (must survive), one row
      // tagged with a non-backfill source (must be deleted), both at the
      // same old timestamp.
      await db.insert(marketSignalsTable).values([
        {
          coinId: TEST_COIN,
          timestamp: oldTs,
          midPrice: 1,
          source: "okx_backfill_funding_v1",
        },
        {
          coinId: TEST_COIN,
          timestamp: oldTs,
          midPrice: 1,
          source: "okx_swap",
        },
      ]);
      assert.equal(await countTestRows(), 2);

      const result = await pruneMarketSignals({ retentionDays: 30 });
      assert.equal(result.success, true);

      const survivors = await db
        .select({
          source: marketSignalsTable.source,
        })
        .from(marketSignalsTable)
        .where(eq(marketSignalsTable.coinId, TEST_COIN));
      assert.equal(
        survivors.length,
        1,
        `expected exactly the backfill row to survive, got ${survivors.length}`,
      );
      assert.equal(survivors[0]?.source, "okx_backfill_funding_v1");
    } finally {
      await cleanup();
    }
  });

  it("pruneMarketSignalsIfDue self-throttles after a successful run", async () => {
    // First run with force=true sets lastSuccessAt.
    const r1 = await pruneMarketSignalsIfDue(true);
    assert.ok(r1 && r1.success);
    // Immediate second call without force should be skipped (returns null).
    const r2 = await pruneMarketSignalsIfDue();
    assert.equal(r2, null, "second call within an hour should be throttled");
  });
});
