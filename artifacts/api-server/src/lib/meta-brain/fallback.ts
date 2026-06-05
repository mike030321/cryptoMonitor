/**
 * Rate-limited fallback logging.
 *
 * When the brain is unreachable / times out / returns invalid JSON we
 * must fall back to the neutral directive silently on the trading path,
 * but we also need enough signal in the logs to debug the cause. This
 * helper collapses repeated fallbacks of the same cause to at most one
 * log line per minute per cause.
 */

import { logger } from "../logger";

const lastLoggedAtByCause: Map<string, number> = new Map();
const ONE_MINUTE_MS = 60_000;

export function logFallbackOnce(
  cause: string,
  extra: Record<string, unknown> = {},
): void {
  const now = Date.now();
  const last = lastLoggedAtByCause.get(cause) ?? 0;
  if (now - last < ONE_MINUTE_MS) return;
  lastLoggedAtByCause.set(cause, now);
  logger.warn(
    { ...extra, cause, subsystem: "meta-brain" },
    "meta-brain fallback to neutral directive",
  );
}

// Test-only. Resets the dedup state so each test starts clean.
export function __resetFallbackLogState(): void {
  lastLoggedAtByCause.clear();
}
