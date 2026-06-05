/**
 * TTL-cached snapshot of which (coinId, timeframe) pairs the ml-engine
 * actually has a trained model for.
 *
 * Why this exists: /ml/predict returns 503 ("no model registered") for any
 * (coin, tf) without a trained per-coin or pooled model. Models currently
 * exist for `1m` (most coins, plus pooled) and `5m` (pooled only). Every
 * cycle the orchestrator was probing all 6 coins x 4 untrained TFs
 * (1h/2h/6h/1d) = ~24 503s/cycle, which:
 *   1. flooded ml-engine logs with 503 entries (the symptom this fixes), and
 *   2. silently undermined the quant-as-primary cutover — every miss becomes
 *      an explicit ABSTAIN, so the dashboard's no-model counts should reflect
 *      only genuinely unavailable model slots.
 *
 * The snapshot is refreshed in the background every REFRESH_MS (60s by
 * default) so a freshly-trained model is picked up within one TTL window
 * without the orchestrator paying a per-request HTTP cost. Resolution
 * mirrors the ml-engine's: per-coin first, then `__pooled__`.
 *
 * Failure mode: if /ml/models has never returned successfully we report
 * "available" so we don't accidentally turn off the quant brain on an
 * ml-engine cold-start. The eventual /predict 503 is then handled by the
 * existing `isNoModelError` path in quant-brain.ts.
 */
import { getMlModels } from "./ml-client";
import { logger } from "./logger";

const REFRESH_MS = 60_000;
const POOLED = "__pooled__";

interface Snapshot {
  keys: Set<string>;
  fetchedAt: number;
}

let snapshot: Snapshot | null = null;
let pollerStarted = false;
let inflight: Promise<void> | null = null;

function key(coinId: string, timeframe: string): string {
  return `${coinId}:${timeframe}`;
}

async function refresh(): Promise<void> {
  try {
    const r = await getMlModels();
    const keys = new Set<string>();
    for (const m of r.available || []) keys.add(key(m.coinId, m.timeframe));
    snapshot = { keys, fetchedAt: Date.now() };
  } catch (err) {
    logger.debug(
      { err: String(err) },
      "ml-availability: /ml/models fetch failed; keeping previous snapshot",
    );
  } finally {
    inflight = null;
  }
}

function startPollerIfNeeded(): void {
  if (pollerStarted) return;
  if (process.env.NODE_ENV === "test") return; // tests inject snapshot manually
  pollerStarted = true;
  inflight = refresh();
  const handle = setInterval(() => {
    if (!inflight) inflight = refresh();
  }, REFRESH_MS);
  // Don't keep the event loop alive on shutdown.
  if (typeof handle.unref === "function") handle.unref();
}

/**
 * Returns true when ml-engine has a per-coin OR pooled model for (coinId, tf),
 * or when the cache has not yet been populated (fail-open). Synchronous so
 * the orchestrator's hot path stays cheap.
 */
export function isModelAvailable(coinId: string, timeframe: string): boolean {
  startPollerIfNeeded();
  const s = snapshot;
  if (!s || s.fetchedAt === 0) return true;
  return s.keys.has(key(coinId, timeframe)) || s.keys.has(key(POOLED, timeframe));
}

/** True only after at least one successful /ml/models fetch. */
export function isMlAvailabilitySnapshotReady(): boolean {
  return snapshot !== null && snapshot.fetchedAt > 0;
}

/** Test seam — populate the snapshot deterministically. */
export function __setMlAvailabilitySnapshot(
  pairs: Array<{ coinId: string; timeframe: string }>,
): void {
  const keys = new Set<string>();
  for (const m of pairs) keys.add(key(m.coinId, m.timeframe));
  snapshot = { keys, fetchedAt: Date.now() };
}

/** Test seam — clear the snapshot and poller flag. */
export function __resetMlAvailabilityCache(): void {
  snapshot = null;
  pollerStarted = false;
  inflight = null;
}
