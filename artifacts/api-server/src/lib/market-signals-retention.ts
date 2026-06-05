import { sql } from "drizzle-orm";
import { db } from "@workspace/db";
import { logger } from "./logger";

/**
 * Task #292 — retention sweep for `market_signals`.
 *
 * The poller writes ~12 rows every 60s (10 monitored coins + btc/eth lead
 * refs) → ~17k rows/day → ~6M rows/year of mostly historical data the
 * trainer no longer needs once the lookback window has passed. Without a
 * retention policy the table silently bloats DB storage and slows the
 * per-coin aggregate query that backs the health endpoint.
 *
 * Policy: delete rows older than `MARKET_SIGNALS_RETENTION_DAYS`
 * (default 30). The horizon is chosen to comfortably cover the trainer's
 * lookback window with margin; tune via env var if the model starts using
 * a longer window.
 *
 * Mirrors the pattern used by journal-retention / feature-lab unquarantine
 * prune: a single `DELETE ... WHERE timestamp < cutoff` that benefits from
 * the existing `market_signals_ts_idx` index on `timestamp`. Self-throttled
 * to once per hour via `pruneMarketSignalsIfDue`.
 *
 * Task #586 — rows whose `source` matches `okx_backfill_%` are EXEMPT from
 * retention. These are the historical funding/OI/mid-price snapshots
 * pulled by `artifacts/ml-engine/scripts/backfill_market_signals.py` so
 * the dataset refresher's asof-join sees real values across the full
 * 365d+ trainer lookback. Their footprint is small (~1.6M rows / year for
 * all monitored coins, ~20MB on disk) and pruning them after 30d would
 * defeat the entire backfill within an hour of a re-deploy. Live-poller
 * rows (any other `source` label, including NULL) keep the original 30d
 * horizon.
 */

export const DEFAULT_MARKET_SIGNALS_RETENTION_DAYS = 30;
const PRUNE_INTERVAL_MS = 60 * 60 * 1000;

let lastAttemptAt = 0;
let lastSuccessAt = 0;
let lastResult: PruneMarketSignalsResult | null = null;

export interface PruneMarketSignalsResult {
  ranAt: string;
  retentionDays: number;
  cutoff: string;
  deleted: number;
  durationMs: number;
  success: boolean;
  error: string | null;
}

function getRetentionDays(): number {
  const raw = process.env["MARKET_SIGNALS_RETENTION_DAYS"];
  const parsed = raw !== undefined ? Number(raw) : NaN;
  return Number.isFinite(parsed) && parsed > 0
    ? Math.floor(parsed)
    : DEFAULT_MARKET_SIGNALS_RETENTION_DAYS;
}

export function getMarketSignalsRetentionDays(): number {
  return getRetentionDays();
}

export async function pruneMarketSignals(opts: {
  retentionDays?: number;
} = {}): Promise<PruneMarketSignalsResult> {
  const startedAt = Date.now();
  const retentionDays = Math.max(1, opts.retentionDays ?? getRetentionDays());
  const cutoff = new Date(startedAt - retentionDays * 24 * 60 * 60 * 1000);

  let deleted = 0;
  let success = true;
  let errorMsg: string | null = null;
  try {
    // Use a CTE so the driver only ships back a single count(*) row even
    // when the first cleanup pass deletes millions of rows — avoids the
    // memory/transfer overhead of `DELETE ... RETURNING id`.
    const result = await db.execute<{ deleted: number }>(sql`
      WITH d AS (
        DELETE FROM market_signals
        WHERE timestamp < ${cutoff}
          AND (source IS NULL OR source NOT LIKE 'okx_backfill_%')
        RETURNING 1
      )
      SELECT count(*)::int AS deleted FROM d
    `);
    const row = (result.rows?.[0] ?? {}) as { deleted?: number };
    deleted = row.deleted ?? 0;
  } catch (err) {
    success = false;
    errorMsg = err instanceof Error ? err.message : String(err);
    logger.error({ err }, "market-signals-retention: prune failed");
  }

  const out: PruneMarketSignalsResult = {
    ranAt: new Date(startedAt).toISOString(),
    retentionDays,
    cutoff: cutoff.toISOString(),
    deleted,
    durationMs: Date.now() - startedAt,
    success,
    error: errorMsg,
  };
  lastResult = out;
  if (success && deleted > 0) {
    logger.info(out, "market-signals-retention: pruned old rows");
  } else if (success) {
    logger.debug(out, "market-signals-retention: nothing to prune");
  }
  return out;
}

export async function pruneMarketSignalsIfDue(
  force = false,
): Promise<PruneMarketSignalsResult | null> {
  const now = Date.now();
  // Gate on last *successful* run — a failed sweep should be retried on
  // the next tick rather than suppressed for an hour.
  if (!force && now - lastSuccessAt < PRUNE_INTERVAL_MS) return null;
  lastAttemptAt = now;
  const result = await pruneMarketSignals();
  if (result.success) lastSuccessAt = Date.now();
  return result;
}

export function getMarketSignalsRetentionStatus(): {
  lastAttemptAt: string | null;
  lastSuccessAt: string | null;
  lastResult: PruneMarketSignalsResult | null;
} {
  return {
    lastAttemptAt: lastAttemptAt > 0 ? new Date(lastAttemptAt).toISOString() : null,
    lastSuccessAt: lastSuccessAt > 0 ? new Date(lastSuccessAt).toISOString() : null,
    lastResult,
  };
}
