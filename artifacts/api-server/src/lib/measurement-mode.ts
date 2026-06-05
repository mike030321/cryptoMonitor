/**
 * Measurement Mode flag.
 *
 * Persists `measurement_mode` (boolean) in the existing app_settings table so
 * an operator can flip it from the Strategy Lab UI without editing env vars
 * or restarting the API server.
 *
 * Order of precedence when reading the effective state:
 *   1. The DB row in app_settings (if present) — this is what the UI writes to.
 *      Manual ON rows expire after 24h unless value.pinned === true, so a
 *      stale observation window cannot permanently block real-price paper
 *      auto-deploy after restarts.
 *   2. The MEASUREMENT_MODE env var (legacy fallback default) — only used
 *      when no DB row exists yet so existing deployments keep their behavior.
 *   3. OFF.
 *
 * Reads are cached briefly so the per-cycle check in monitor.ts doesn't hit
 * the DB on every iteration. Writes invalidate the cache immediately.
 */
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";

export const MEASUREMENT_MODE_KEY = "measurement_mode";

const CACHE_TTL_MS = 5_000;
const MANUAL_ON_TTL_MS = 24 * 60 * 60 * 1000;

export type MeasurementModeSource = "default" | "env" | "manual";

export interface MeasurementModeState {
  enabled: boolean;
  source: MeasurementModeSource;
  lastChangedAt: string;
}

interface CacheEntry {
  state: MeasurementModeState;
  expiresAt: number;
}

let cache: CacheEntry | null = null;

function envDefault(): { enabled: boolean; source: MeasurementModeSource } {
  // Quant-only is the architecturally correct default: only the
  // prediction-orchestrator (quant brain) may open trades. The legacy
  // momentum auto-deploy path (paper-trader.ts:autoDeployIdleCash) and
  // auto-evolution are gated behind this same flag, so OFF-by-default
  // would silently re-enable a non-quant trade source on any deployment
  // without a DB row. Operators can still flip this back via the
  // Strategy Lab UI if they explicitly want to observe the legacy paths.
  if (process.env.MEASUREMENT_MODE === "true") {
    return { enabled: true, source: "env" };
  }
  if (process.env.MEASUREMENT_MODE === "false") {
    return { enabled: false, source: "env" };
  }
  return { enabled: true, source: "default" };
}

function resolvePersistedState(
  value: { enabled?: unknown; pinned?: unknown } | null,
  updatedAt: Date,
  nowMs = Date.now(),
): MeasurementModeState {
  const enabled = !!(value && typeof value === "object" && value.enabled === true);
  const pinned = !!(value && typeof value === "object" && value.pinned === true);
  const updatedAtMs = updatedAt.getTime();
  const lastChangedAt = Number.isFinite(updatedAtMs)
    ? updatedAt.toISOString()
    : new Date(0).toISOString();

  if (
    enabled &&
    !pinned &&
    Number.isFinite(updatedAtMs) &&
    nowMs - updatedAtMs > MANUAL_ON_TTL_MS
  ) {
    // Stale manual ON: collapse back to whatever the env-default is so the
    // legacy momentum auto-deploy lane does not stay frozen forever after
    // an operator forgets to flip the toggle back.
    const fallback = envDefault();
    return {
      enabled: fallback.enabled,
      source: fallback.source,
      lastChangedAt,
    };
  }

  return {
    enabled,
    source: "manual",
    lastChangedAt,
  };
}

export async function getMeasurementModeState(): Promise<MeasurementModeState> {
  if (cache && cache.expiresAt > Date.now()) return cache.state;
  let state: MeasurementModeState;
  try {
    const rows = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, MEASUREMENT_MODE_KEY))
      .limit(1);
    if (rows.length === 0) {
      const fallback = envDefault();
      state = {
        enabled: fallback.enabled,
        source: fallback.source,
        lastChangedAt: new Date(0).toISOString(),
      };
    } else {
      const v = rows[0].value as { enabled?: unknown; pinned?: unknown } | null;
      state = resolvePersistedState(
        v,
        rows[0].updatedAt instanceof Date ? rows[0].updatedAt : new Date(0),
      );
    }
  } catch (err) {
    logger.warn(
      { err },
      "measurement-mode: failed to read app_settings — falling back to env default",
    );
    const fallback = envDefault();
    state = {
      enabled: fallback.enabled,
      source: fallback.source,
      lastChangedAt: new Date(0).toISOString(),
    };
  }
  cache = { state, expiresAt: Date.now() + CACHE_TTL_MS };
  return state;
}

export async function isMeasurementModeEnabled(): Promise<boolean> {
  return (await getMeasurementModeState()).enabled;
}

export async function setMeasurementModeState(
  enabled: boolean,
): Promise<MeasurementModeState> {
  const value = { enabled, source: "manual" as const };
  await db
    .insert(appSettingsTable)
    .values({ key: MEASUREMENT_MODE_KEY, value })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value, updatedAt: new Date() },
    });
  cache = null;
  const state = await getMeasurementModeState();
  logger.info({ enabled }, "measurement-mode: updated");
  return state;
}

export function invalidateMeasurementModeCache(): void {
  cache = null;
}
