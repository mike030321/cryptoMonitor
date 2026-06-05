/**
 * Task #390 — Strategy Lab → Meta-Brain benchmark telemetry assembler.
 *
 * Read-only helper that queries `strategy_snapshots` for the four
 * Strategy Lab buckets (`ai-bots`, `dca-cb`, `buy-hold`,
 * `trend-filter`) and summarizes the AI fleet's *relative alpha* vs.
 * the best deterministic baseline over rolling 7d / 14d windows. The
 * result is consumed by the meta-brain's governance layer ONLY — it
 * never reaches the quant predictor, `/ml/decide`, training features,
 * or any trade-decision payload (enforced by the
 * `no-llm-fields-in-trade-decisions` parity guards).
 *
 * Architectural rules (non-negotiable, see .local/tasks/task-390.md):
 *   - Read-only. Zero writes to any `strategy_*` table.
 *   - Long lookback only (7d / 14d). Never same-tick / 1d values.
 *   - Slow cadence: computed once per ~30s monitor cycle.
 *   - Fail-safe: missing / stale snapshots return a neutral struct
 *     flagged `stale: true` so the brain behaves exactly as today.
 *   - Cap query window at ~30 days so the assembler stays cheap even
 *     as the snapshot table grows.
 */

import { and, asc, eq, gte } from "drizzle-orm";
import { db, strategySnapshotsTable } from "@workspace/db";
import { logger } from "../logger";

/** The four Strategy Lab bucket identifiers. AI Bots = the quant fleet
 * aggregate; the other three are deterministic baselines that the
 * meta-brain reads as a benchmark cohort. */
export const BENCHMARK_AI_BUCKET = "ai-bots" as const;
export const BENCHMARK_BASELINE_BUCKETS = [
  "dca-cb",
  "buy-hold",
  "trend-filter",
] as const;

/** Compact telemetry struct sent to the meta-brain alongside portfolio
 * + slice batch. Eight numeric fields (per the task spec) plus the
 * boolean / stale flags and a sample-count audit field. */
export interface BenchmarkTelemetry {
  /** AI fleet trailing 7-day return, expressed as a fraction
   * (e.g. 0.025 = +2.5%). 0 when stale. */
  aiReturn7d: number;
  /** Best-of-baselines trailing 7-day return (same convention). */
  bestBaselineReturn7d: number;
  /** aiReturn7d − bestBaselineReturn7d. */
  relativeAlpha7d: number;
  /** aiReturn14d − bestBaselineReturn14d. */
  relativeAlpha14d: number;
  /** AI peak-to-trough drawdown over the 14d window divided by the
   * best baseline's drawdown over the same window. Clamped to [0, 5].
   * 1.0 means equal pain; > 1 means the fleet hurts more. */
  drawdownRatioVsBest: number;
  /** True iff `relativeAlpha7d < 0` AND `relativeAlpha14d < 0` AND we
   * have enough samples — i.e. the fleet has been losing alpha for
   * long enough to act on. */
  sustainedUnderperformance: boolean;
  /** Number of AI snapshot rows used in the 14d computation. */
  sampleCount: number;
  /** True when too few snapshots / oldest sample too recent — the
   * brain is expected to fall back to neutral behaviour. */
  stale: boolean;
}

const WINDOW_QUERY_MS = 30 * 24 * 60 * 60 * 1000; // 30d cap
const WINDOW_7D_MS = 7 * 24 * 60 * 60 * 1000;
const WINDOW_14D_MS = 14 * 24 * 60 * 60 * 1000;
/** Below this many snapshots in a window we cannot make a defensible
 * relative-alpha claim → fall back to neutral + stale. */
const MIN_SAMPLES_PER_WINDOW = 10;
/** If the most recent AI snapshot is older than this, the data is
 * effectively dark and we report stale. */
const MAX_LAST_SAMPLE_AGE_MS = 6 * 60 * 60 * 1000; // 6h

/** Per-cycle memoization. Cleared at start of every monitor cycle via
 * `resetBenchmarkCache()`. Keeps multiple readers (assembler + cycle-
 * stats endpoint) from re-querying the snapshot table. */
let cycleCache: BenchmarkTelemetry | null = null;
let cycleCacheToken = 0;
let lastResetToken = 0;

export function resetBenchmarkCache(): void {
  cycleCache = null;
  lastResetToken = ++cycleCacheToken;
}

interface SnapshotRow {
  timestamp: Date;
  equity: number;
}

function neutralStale(sampleCount = 0): BenchmarkTelemetry {
  return {
    aiReturn7d: 0,
    bestBaselineReturn7d: 0,
    relativeAlpha7d: 0,
    relativeAlpha14d: 0,
    drawdownRatioVsBest: 1,
    sustainedUnderperformance: false,
    sampleCount,
    stale: true,
  };
}

/** Trailing return over the last `windowMs`: (last - first) / first. */
function trailingReturn(rows: SnapshotRow[], windowMs: number): {
  ret: number;
  count: number;
} {
  if (rows.length === 0) return { ret: 0, count: 0 };
  const cutoff = Date.now() - windowMs;
  const window = rows.filter((r) => r.timestamp.getTime() >= cutoff);
  if (window.length < 2) return { ret: 0, count: window.length };
  const first = window[0].equity;
  const last = window[window.length - 1].equity;
  if (!Number.isFinite(first) || first <= 0 || !Number.isFinite(last)) {
    return { ret: 0, count: window.length };
  }
  return { ret: (last - first) / first, count: window.length };
}

/** Peak-to-trough drawdown over the window (positive number). */
function peakDrawdown(rows: SnapshotRow[], windowMs: number): number {
  const cutoff = Date.now() - windowMs;
  const window = rows.filter((r) => r.timestamp.getTime() >= cutoff);
  let peak = -Infinity;
  let dd = 0;
  for (const r of window) {
    if (!Number.isFinite(r.equity)) continue;
    if (r.equity > peak) peak = r.equity;
    if (peak > 0) {
      const local = (peak - r.equity) / peak;
      if (local > dd) dd = local;
    }
  }
  return Number.isFinite(dd) ? dd : 0;
}

async function fetchBucketRows(
  bucket: string,
  cutoff: Date,
): Promise<SnapshotRow[]> {
  const rows = await db
    .select({
      timestamp: strategySnapshotsTable.timestamp,
      equity: strategySnapshotsTable.equity,
    })
    .from(strategySnapshotsTable)
    .where(
      and(
        eq(strategySnapshotsTable.strategyType, bucket),
        gte(strategySnapshotsTable.timestamp, cutoff),
      ),
    )
    .orderBy(asc(strategySnapshotsTable.timestamp));
  return rows.map((r) => ({ timestamp: r.timestamp, equity: r.equity }));
}

export async function assembleBenchmarkTelemetry(): Promise<BenchmarkTelemetry> {
  if (cycleCache !== null && lastResetToken === cycleCacheToken) {
    return cycleCache;
  }
  try {
    const cutoff = new Date(Date.now() - WINDOW_QUERY_MS);
    const aiRows = await fetchBucketRows(BENCHMARK_AI_BUCKET, cutoff);

    if (aiRows.length === 0) {
      cycleCache = neutralStale(0);
      return cycleCache;
    }

    const lastAi = aiRows[aiRows.length - 1];
    const ageMs = Date.now() - lastAi.timestamp.getTime();
    const ai7 = trailingReturn(aiRows, WINDOW_7D_MS);
    const ai14 = trailingReturn(aiRows, WINDOW_14D_MS);

    if (
      ageMs > MAX_LAST_SAMPLE_AGE_MS ||
      ai14.count < MIN_SAMPLES_PER_WINDOW
    ) {
      cycleCache = neutralStale(ai14.count);
      return cycleCache;
    }

    const baselineRows = await Promise.all(
      BENCHMARK_BASELINE_BUCKETS.map((b) => fetchBucketRows(b, cutoff)),
    );
    let best7 = -Infinity;
    let best14 = -Infinity;
    let bestDd14 = Infinity;
    let baselineCount14 = 0;
    for (const rows of baselineRows) {
      const r7 = trailingReturn(rows, WINDOW_7D_MS);
      const r14 = trailingReturn(rows, WINDOW_14D_MS);
      const dd14 = peakDrawdown(rows, WINDOW_14D_MS);
      if (r7.count >= 2 && r7.ret > best7) best7 = r7.ret;
      if (r14.count >= 2) {
        if (r14.ret > best14) best14 = r14.ret;
        if (dd14 < bestDd14) bestDd14 = dd14;
        baselineCount14 += r14.count;
      }
    }

    if (!Number.isFinite(best14) || baselineCount14 < MIN_SAMPLES_PER_WINDOW) {
      cycleCache = neutralStale(ai14.count);
      return cycleCache;
    }

    const aiDd14 = peakDrawdown(aiRows, WINDOW_14D_MS);
    const ddRatio =
      bestDd14 > 1e-6
        ? Math.min(5, Math.max(0, aiDd14 / bestDd14))
        : aiDd14 > 1e-6
          ? 5
          : 1;

    const alpha7 = ai7.ret - (Number.isFinite(best7) ? best7 : 0);
    const alpha14 = ai14.ret - best14;
    const sustained =
      alpha7 < 0 && alpha14 < 0 && ai7.count >= MIN_SAMPLES_PER_WINDOW;

    cycleCache = {
      aiReturn7d: ai7.ret,
      bestBaselineReturn7d: Number.isFinite(best7) ? best7 : 0,
      relativeAlpha7d: alpha7,
      relativeAlpha14d: alpha14,
      drawdownRatioVsBest: ddRatio,
      sustainedUnderperformance: sustained,
      sampleCount: ai14.count,
      stale: false,
    };
    return cycleCache;
  } catch (err) {
    logger.debug(
      { err: String(err), subsystem: "meta-brain" },
      "benchmark telemetry assembler failed (non-blocking)",
    );
    cycleCache = neutralStale(0);
    return cycleCache;
  }
}

// Test-only export.
export function __setBenchmarkCacheForTest(t: BenchmarkTelemetry | null): void {
  cycleCache = t;
}
