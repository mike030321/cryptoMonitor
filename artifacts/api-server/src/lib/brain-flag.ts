/**
 * Phase 5 — Brain feature flag.
 *
 * Persists `quant_brain_enabled` (boolean) in the existing app_settings table.
 * Default is OFF — the LightGBM models in artifacts/ml-engine/models/ have not
 * been trained on real data yet, so flipping the flag ON before training is
 * complete means trading on stub predictions. The default and the env override
 * `QUANT_BRAIN_FORCE_OFF=1` both exist to make accidental cutover impossible.
 *
 * Reads are cached for BRAIN_CACHE_TTL_MS so the analysis cycle does not hit
 * the DB once per (coin, timeframe). Writes invalidate the cache immediately.
 */
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";
import { QUANT_BRAIN_ENABLED } from "./trading-constants";

export const BRAIN_FLAG_KEY = "quant_brain_enabled";
export const BRAIN_REVERT_LOG_KEY = "brain_revert_log";

const BRAIN_CACHE_TTL_MS = 5_000;

export type BrainSource = "default" | "manual" | "auto_revert" | "env";

interface BrainState {
  enabled: boolean;
  source: BrainSource;
  lastChangedAt: string;
}

interface CacheEntry {
  state: BrainState;
  expiresAt: number;
}

let cache: CacheEntry | null = null;

function defaultState(source: BrainSource = "default"): BrainState {
  // The persisted-default lives in shared/trading-frictions.json so the
  // Python backtester and TS live trader read the same value. Even when the
  // JSON ships with `enabled: true`, the env force-off and the stored DB
  // row both still override this.
  return { enabled: QUANT_BRAIN_ENABLED, source, lastChangedAt: new Date(0).toISOString() };
}

function envForceOff(): boolean {
  return process.env.QUANT_BRAIN_FORCE_OFF === "1" || process.env.QUANT_BRAIN_FORCE_OFF === "true";
}

export async function getBrainState(): Promise<BrainState> {
  // Env force-off is the final, non-negotiable kill-switch. It must hard-
  // return enabled:false regardless of the JSON default or any DB row,
  // otherwise an operator who shipped trading-frictions.json with
  // `quant_brain.enabled: true` could not stop the brain via env. Do NOT
  // route through defaultState() here — that reads the JSON value.
  if (envForceOff()) {
    return { enabled: false, source: "env", lastChangedAt: new Date().toISOString() };
  }
  if (cache && cache.expiresAt > Date.now()) return cache.state;
  try {
    const rows = await db.select().from(appSettingsTable).where(eq(appSettingsTable.key, BRAIN_FLAG_KEY)).limit(1);
    let state: BrainState;
    if (rows.length === 0) {
      state = defaultState();
    } else {
      const v = rows[0].value as { enabled?: unknown; source?: unknown } | null;
      const enabled: boolean = !!(v && typeof v === "object" && v.enabled === true);
      const source: BrainSource =
        v && typeof v.source === "string" && ["default", "manual", "auto_revert", "env"].includes(v.source)
          ? (v.source as BrainSource)
          : "manual";
      state = {
        enabled,
        source,
        lastChangedAt: rows[0].updatedAt instanceof Date ? rows[0].updatedAt.toISOString() : new Date(0).toISOString(),
      };
    }
    cache = { state, expiresAt: Date.now() + BRAIN_CACHE_TTL_MS };
    return state;
  } catch (err) {
    logger.warn({ err }, "brain-flag: failed to read app_settings — using configured quant default");
    return defaultState();
  }
}

export async function isQuantBrainEnabled(): Promise<boolean> {
  return (await getBrainState()).enabled;
}

/**
 * Task #659 — DS lane gate. True iff env kill-switch is off AND the
 * MTTM cache is warm AND enabled+mode==='diagnostic_sandbox' AND
 * disableReason is null AND universe is exactly {bitcoin|5m}.
 * Fail-closed on cold cache. Global quant_brain_enabled stays false.
 */
export async function isDiagnosticSandboxEnabled(): Promise<boolean> {
  if (envForceOff()) return false;
  const { getMttmConfigCached, MTTM_DIAGNOSTIC_SANDBOX_COIN,
    MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME, slotKey } = await import("./mttm");
  const m = getMttmConfigCached();
  if (!m) return false;
  if (m.enabled !== true) return false;
  if (m.mode !== "diagnostic_sandbox") return false;
  if (m.disableReason !== null) return false;
  if (m.universeKeys.size !== 1) return false;
  const pinned = slotKey(
    MTTM_DIAGNOSTIC_SANDBOX_COIN,
    MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
  );
  return m.universeKeys.has(pinned);
}

/** Task #659 — composite reachability: global brain OR DS+slot match. */
export async function isQuantBrainReachable(
  coinId: string,
  timeframe: string,
): Promise<boolean> {
  if (await isQuantBrainEnabled()) return true;
  if (!(await isDiagnosticSandboxEnabled())) return false;
  const { MTTM_DIAGNOSTIC_SANDBOX_COIN, MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME }
    = await import("./mttm");
  return (
    coinId === MTTM_DIAGNOSTIC_SANDBOX_COIN
    && timeframe === MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME
  );
}

export async function setBrainState(enabled: boolean, source: Exclude<BrainSource, "env" | "default">): Promise<BrainState> {
  if (envForceOff() && enabled) {
    logger.warn({ source }, "brain-flag: refusing to enable — QUANT_BRAIN_FORCE_OFF is set");
    return getBrainState();
  }
  const value = { enabled, source };
  await db
    .insert(appSettingsTable)
    .values({ key: BRAIN_FLAG_KEY, value })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value, updatedAt: new Date() },
    });
  cache = null;
  const state = await getBrainState();
  logger.info({ enabled, source }, "brain-flag: updated");
  return state;
}

export function invalidateBrainCache(): void {
  cache = null;
}
