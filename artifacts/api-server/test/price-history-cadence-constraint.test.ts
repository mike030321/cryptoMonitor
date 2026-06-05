// Task #347 — DB-level safety net for `price_history`.
//
// Asserts the running database has the cadence column + CHECK constraint
// applied (via `runMigrations`) and that an explicit non-1m insert is
// rejected by Postgres (sqlstate 23514). The two app-side guards
// (`assertNativeCadence` in TS, `assert_native_cadence` in Python) are
// still the first line of defense; this test exists to prove the second
// line (the database itself) actually fires when a future writer
// bypasses them.
import { test } from "node:test";
import assert from "node:assert/strict";

import { db, pool, priceHistoryTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { runMigrations } from "../src/lib/migrate";

const TEST_COIN_ID = "__cadence_constraint_test__";

async function cleanupRows(): Promise<void> {
  await db.delete(priceHistoryTable).where(eq(priceHistoryTable.coinId, TEST_COIN_ID));
}

test("runMigrations applies the price_history cadence guard idempotently", async () => {
  // Run twice — second call must be a no-op, not a duplicate-constraint
  // error, because the runner is invoked on every boot.
  await runMigrations();
  await runMigrations();

  const { rows } = await pool.query<{ conname: string }>(
    `SELECT conname FROM pg_constraint WHERE conname = 'price_history_cadence_is_1m'`,
  );
  assert.equal(rows.length, 1, "CHECK constraint must exist after migrations run");

  const colCheck = await pool.query<{ column_name: string; column_default: string | null }>(
    `SELECT column_name, column_default FROM information_schema.columns
     WHERE table_name = 'price_history' AND column_name = 'cadence'`,
  );
  assert.equal(colCheck.rows.length, 1, "cadence column must exist");
  assert.match(
    colCheck.rows[0]!.column_default ?? "",
    /'1m'/,
    "cadence must default to '1m'",
  );
});

test("default-cadence insert is accepted (existing writers keep working)", async () => {
  await runMigrations();
  await cleanupRows();
  try {
    await db.insert(priceHistoryTable).values({
      coinId: TEST_COIN_ID,
      price: 1.0,
      timestamp: new Date(),
    });
    const inserted = await db
      .select({ cadence: priceHistoryTable.cadence })
      .from(priceHistoryTable)
      .where(eq(priceHistoryTable.coinId, TEST_COIN_ID));
    assert.equal(inserted.length, 1);
    assert.equal(inserted[0]!.cadence, "1m");
  } finally {
    await cleanupRows();
  }
});

test("explicit cadence='5m' insert is rejected by the DB CHECK constraint", async () => {
  await runMigrations();
  await cleanupRows();
  let rejected = false;
  let sqlstate: string | undefined;
  let constraintName: string | undefined;
  try {
    // Bypass the Drizzle insert (which would type-narrow cadence) and
    // simulate the failure mode this guard exists for: a foreign writer
    // (manual SQL session, future ETL script) that ignores the app-side
    // assertNativeCadence and writes a non-1m row directly.
    await pool.query(
      `INSERT INTO price_history (coin_id, price, timestamp, cadence)
       VALUES ($1, $2, NOW(), $3)`,
      [TEST_COIN_ID, 1.0, "5m"],
    );
  } catch (err) {
    rejected = true;
    const e = err as { code?: string; constraint?: string };
    sqlstate = e.code;
    constraintName = e.constraint;
  } finally {
    await cleanupRows();
  }
  assert.ok(rejected, "non-1m write must be rejected");
  assert.equal(sqlstate, "23514", "must be a check_violation");
  assert.equal(constraintName, "price_history_cadence_is_1m");
});
