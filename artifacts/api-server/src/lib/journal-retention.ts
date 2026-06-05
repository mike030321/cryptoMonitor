import { sql } from "drizzle-orm";
import { db } from "@workspace/db";
import { logger } from "./logger";

/**
 * Retention + rollup for prediction_journal and trade_journal.
 *
 * Policy: keep raw rows for the last `JOURNAL_RETENTION_DAYS` (default 90).
 * Anything older is aggregated into one row per
 * (bucket_date, kind, brain, coin_id, timeframe, direction) in
 * `journal_rollup`, then the source rows are deleted.
 *
 * Concurrency-safe: rollup and delete operate on the EXACT SAME rowset
 * via a single SQL statement that uses a `DELETE ... RETURNING` CTE as
 * the input to the rollup INSERT. Concurrently inserted rows that match
 * `created_at < cutoff` arriving between two separate snapshots can no
 * longer be deleted-without-being-rolled-up, because there is only one
 * snapshot — the DELETE's own. The whole thing also runs inside a
 * `db.transaction()` so a failure rolls everything back.
 *
 * Idempotent: rerunning merges into existing buckets via
 * INSERT ... ON CONFLICT DO UPDATE that adds counts/sums on top.
 *
 * Health/diagnostics endpoints read a 24h window and so are unaffected.
 */

const DEFAULT_RETENTION_DAYS = 90;
const PRUNE_INTERVAL_MS = 60 * 60 * 1000; // run at most once per hour

let lastAttemptAt = 0;
let lastSuccessAt = 0;
let lastResult: JournalRetentionResult | null = null;

export interface JournalRetentionResult {
  ranAt: string;
  retentionDays: number;
  cutoff: string;
  predictionsRolledUp: number;
  predictionsDeleted: number;
  tradesRolledUp: number;
  tradesDeleted: number;
  rollupBuckets: number;
  durationMs: number;
  success: boolean;
  error: string | null;
}

function getRetentionDays(): number {
  const raw = process.env["JOURNAL_RETENTION_DAYS"];
  const parsed = raw !== undefined ? Number(raw) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_RETENTION_DAYS;
}

export function getJournalRetentionDays(): number {
  return getRetentionDays();
}

export function getLastJournalRetentionResult(): JournalRetentionResult | null {
  return lastResult;
}

type Tx = Parameters<Parameters<typeof db.transaction>[0]>[0];

interface PassResult {
  rolledUp: number;
  deleted: number;
  buckets: number;
}

async function rollupPredictions(tx: Tx, cutoff: Date): Promise<PassResult> {
  // Single-statement DELETE+ROLLUP: the DELETE's RETURNING is the only
  // snapshot of "old rows" — aggregation feeds the INSERT directly, so
  // concurrently inserted rows cannot be deleted without being counted.
  const result = await tx.execute(sql`
    WITH deleted AS (
      DELETE FROM prediction_journal
      WHERE created_at < ${cutoff}
      RETURNING brain, coin_id, timeframe, direction, created_at,
                became_trade, outcome, realized_return_pct
    ),
    agg AS (
      SELECT
        date_trunc('day', created_at)::date AS bucket_date,
        coalesce(brain, '') AS brain,
        coin_id,
        timeframe,
        coalesce(direction, '') AS direction,
        count(*)::int AS cnt,
        count(*) FILTER (WHERE became_trade = true)::int AS became_trade,
        count(*) FILTER (
          WHERE outcome IS NOT NULL AND outcome <> 'pending'
        )::int AS resolved,
        count(*) FILTER (WHERE outcome = 'correct')::int AS correct,
        count(*) FILTER (WHERE outcome = 'wrong')::int AS wrong,
        count(*) FILTER (WHERE outcome = 'neutral')::int AS neutral,
        coalesce(sum(realized_return_pct), 0)::real AS sum_ret
      FROM deleted
      GROUP BY 1, 2, 3, 4, 5
    ),
    upserted AS (
      INSERT INTO journal_rollup (
        bucket_date, kind, brain, coin_id, timeframe, direction,
        count, became_trade_count, resolved_count, correct_count,
        wrong_count, neutral_count, sum_realized_return_pct
      )
      SELECT
        bucket_date, 'prediction', brain, coin_id, timeframe, direction,
        cnt, became_trade, resolved, correct, wrong, neutral, sum_ret
      FROM agg
      ON CONFLICT (bucket_date, kind, brain, coin_id, timeframe, direction)
      DO UPDATE SET
        count = journal_rollup.count + EXCLUDED.count,
        became_trade_count = coalesce(journal_rollup.became_trade_count, 0) + EXCLUDED.became_trade_count,
        resolved_count = coalesce(journal_rollup.resolved_count, 0) + EXCLUDED.resolved_count,
        correct_count = coalesce(journal_rollup.correct_count, 0) + EXCLUDED.correct_count,
        wrong_count = coalesce(journal_rollup.wrong_count, 0) + EXCLUDED.wrong_count,
        neutral_count = coalesce(journal_rollup.neutral_count, 0) + EXCLUDED.neutral_count,
        sum_realized_return_pct = coalesce(journal_rollup.sum_realized_return_pct, 0) + EXCLUDED.sum_realized_return_pct,
        rolled_up_at = now()
      RETURNING 1
    )
    SELECT
      (SELECT count(*)::int FROM deleted)            AS deleted,
      (SELECT count(*)::int FROM agg)                AS buckets,
      (SELECT coalesce(sum(cnt), 0)::int FROM agg)   AS rolled_up
  `);

  const row = (result.rows?.[0] ?? {}) as { deleted?: number; buckets?: number; rolled_up?: number };
  return {
    deleted: row.deleted ?? 0,
    buckets: row.buckets ?? 0,
    rolledUp: row.rolled_up ?? 0,
  };
}

async function rollupTrades(tx: Tx, cutoff: Date): Promise<PassResult> {
  const result = await tx.execute(sql`
    WITH deleted AS (
      DELETE FROM trade_journal
      WHERE created_at < ${cutoff}
      RETURNING coin_id, timeframe, direction, created_at,
                realized_pnl_usd, realized_pnl_pct,
                entry_fee, exit_fee, mfe_pct, mae_pct
    ),
    agg AS (
      SELECT
        date_trunc('day', created_at)::date AS bucket_date,
        coin_id,
        timeframe,
        direction,
        count(*)::int AS cnt,
        coalesce(sum(realized_pnl_usd), 0)::real AS sum_pnl_usd,
        coalesce(sum(realized_pnl_pct), 0)::real AS sum_pnl_pct,
        coalesce(sum(entry_fee), 0)::real AS sum_entry_fee,
        coalesce(sum(exit_fee), 0)::real AS sum_exit_fee,
        count(*) FILTER (
          WHERE mfe_pct IS NOT NULL AND mae_pct IS NOT NULL
        )::int AS with_mae_mfe,
        count(*) FILTER (
          WHERE entry_fee IS NOT NULL AND exit_fee IS NOT NULL
        )::int AS with_fees
      FROM deleted
      GROUP BY 1, 2, 3, 4
    ),
    upserted AS (
      INSERT INTO journal_rollup (
        bucket_date, kind, brain, coin_id, timeframe, direction,
        count, sum_realized_pnl_usd, sum_realized_pnl_pct,
        sum_entry_fee, sum_exit_fee, with_mae_mfe_count, with_fees_count
      )
      SELECT
        bucket_date, 'trade', '', coin_id, timeframe, direction,
        cnt, sum_pnl_usd, sum_pnl_pct, sum_entry_fee, sum_exit_fee,
        with_mae_mfe, with_fees
      FROM agg
      ON CONFLICT (bucket_date, kind, brain, coin_id, timeframe, direction)
      DO UPDATE SET
        count = journal_rollup.count + EXCLUDED.count,
        sum_realized_pnl_usd = coalesce(journal_rollup.sum_realized_pnl_usd, 0) + EXCLUDED.sum_realized_pnl_usd,
        sum_realized_pnl_pct = coalesce(journal_rollup.sum_realized_pnl_pct, 0) + EXCLUDED.sum_realized_pnl_pct,
        sum_entry_fee = coalesce(journal_rollup.sum_entry_fee, 0) + EXCLUDED.sum_entry_fee,
        sum_exit_fee = coalesce(journal_rollup.sum_exit_fee, 0) + EXCLUDED.sum_exit_fee,
        with_mae_mfe_count = coalesce(journal_rollup.with_mae_mfe_count, 0) + EXCLUDED.with_mae_mfe_count,
        with_fees_count = coalesce(journal_rollup.with_fees_count, 0) + EXCLUDED.with_fees_count,
        rolled_up_at = now()
      RETURNING 1
    )
    SELECT
      (SELECT count(*)::int FROM deleted)            AS deleted,
      (SELECT count(*)::int FROM agg)                AS buckets,
      (SELECT coalesce(sum(cnt), 0)::int FROM agg)   AS rolled_up
  `);

  const row = (result.rows?.[0] ?? {}) as { deleted?: number; buckets?: number; rolled_up?: number };
  return {
    deleted: row.deleted ?? 0,
    buckets: row.buckets ?? 0,
    rolledUp: row.rolled_up ?? 0,
  };
}

/**
 * Run the rollup + delete pass. Safe to call repeatedly; an internal
 * timestamp gate skips runs more often than `PRUNE_INTERVAL_MS` unless
 * `force` is set.
 */
export async function runJournalRetention(
  force = false,
): Promise<JournalRetentionResult | null> {
  const now = Date.now();
  // Throttle gates the next *successful* run, but a previous failure should
  // not block retries for an hour. We compare against `lastSuccessAt` so a
  // crashed/aborted run can be retried on the next tick.
  if (!force && now - lastSuccessAt < PRUNE_INTERVAL_MS) return null;
  lastAttemptAt = now;

  const startedAt = Date.now();
  const retentionDays = getRetentionDays();
  const cutoff = new Date(now - retentionDays * 24 * 60 * 60 * 1000);

  let predictionsRolledUp = 0;
  let predictionsDeleted = 0;
  let tradesRolledUp = 0;
  let tradesDeleted = 0;
  let rollupBuckets = 0;
  let success = true;
  let errorMsg: string | null = null;

  try {
    await db.transaction(async (tx) => {
      const p = await rollupPredictions(tx, cutoff);
      predictionsRolledUp = p.rolledUp;
      predictionsDeleted = p.deleted;
      rollupBuckets += p.buckets;

      const t = await rollupTrades(tx, cutoff);
      tradesRolledUp = t.rolledUp;
      tradesDeleted = t.deleted;
      rollupBuckets += t.buckets;
    });
  } catch (err) {
    success = false;
    errorMsg = err instanceof Error ? err.message : String(err);
    logger.error({ err }, "journal-retention: rollup failed (transaction rolled back)");
  }

  if (success) lastSuccessAt = now;

  const result: JournalRetentionResult = {
    ranAt: new Date(now).toISOString(),
    retentionDays,
    cutoff: cutoff.toISOString(),
    predictionsRolledUp,
    predictionsDeleted,
    tradesRolledUp,
    tradesDeleted,
    rollupBuckets,
    durationMs: Date.now() - startedAt,
    success,
    error: errorMsg,
  };
  lastResult = result;

  if (!success) {
    logger.warn(result, "journal-retention: run failed");
  } else if (predictionsDeleted > 0 || tradesDeleted > 0) {
    logger.info(result, "journal-retention: rollup completed");
  } else {
    logger.debug(result, "journal-retention: nothing to roll up");
  }
  return result;
}

export function getJournalRetentionStatus(): {
  lastAttemptAt: string | null;
  lastSuccessAt: string | null;
  lastResult: JournalRetentionResult | null;
} {
  return {
    lastAttemptAt: lastAttemptAt > 0 ? new Date(lastAttemptAt).toISOString() : null,
    lastSuccessAt: lastSuccessAt > 0 ? new Date(lastSuccessAt).toISOString() : null,
    lastResult,
  };
}
