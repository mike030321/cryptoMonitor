/**
 * Task #468 — startup-loaded agent → profile cache.
 *
 * The trade-execution path resolves an agent's strategy profile via
 * its numeric `agentId`. Doing a per-decision DB lookup would let the
 * agents.profile_id column flap mid-cycle, so the spec mandates a
 * boot-time mapping that is rebuilt only on admin reload (SIGHUP).
 *
 * This module owns that mapping. `loadAgentRegistryCache()` is called
 * from the boot sequence (after `syncAgentProfileIds()` populates
 * profile_id) and from the SIGHUP handler. `getCachedProfileForAgentId()`
 * THROWS on unknown — there is no silent default.
 */

import { db, agentsTable } from "@workspace/db";
import { logger } from "../logger";
import { getAgentProfile } from "./registry";
import type { AgentProfile } from "./schema";

interface CacheEntry {
  agentId: number;
  agentName: string;
  profile: AgentProfile;
  /** DB-level status (active / resting / degraded / quarantine_review /
   * disabled). The trade gate refuses any value that is not "active",
   * even if the profile itself is `executes=true`. */
  dbStatus: string;
}

let CACHE: Map<number, CacheEntry> = new Map();
let LOADED_AT: string | null = null;

/**
 * Rebuild the cache from the live agents table. Idempotent and safe to
 * call concurrently — the new map is constructed in a local then
 * swapped in atomically.
 */
export async function loadAgentRegistryCache(): Promise<{
  total: number;
  loaded_at: string;
}> {
  const rows = await db.select().from(agentsTable);
  const next = new Map<number, CacheEntry>();
  let resolveErrors = 0;
  for (const row of rows) {
    const profileId = (row as { profileId?: string | null }).profileId ?? null;
    if (!profileId) {
      // The boot sweep populates profile_id BEFORE this call. A null
      // here means a row was inserted between the sweep and this load
      // — log loudly but do not crash; the trade gate will throw.
      resolveErrors++;
      continue;
    }
    let profile: AgentProfile;
    try {
      profile = getAgentProfile(profileId);
    } catch (err) {
      resolveErrors++;
      logger.error(
        { err: String(err), agentId: row.id, profileId },
        "agents-registry: cache load could not resolve profile (row skipped — trade gate will throw)",
      );
      continue;
    }
    next.set(row.id, {
      agentId: row.id,
      agentName: row.name,
      profile,
      dbStatus: row.status,
    });
  }
  CACHE = next;
  LOADED_AT = new Date().toISOString();
  logger.info(
    { total: rows.length, cached: next.size, resolveErrors, loaded_at: LOADED_AT },
    "Task #468 — agent registry cache (re)loaded",
  );
  return { total: rows.length, loaded_at: LOADED_AT };
}

/**
 * Resolve the profile for a numeric agent id. THROWS if the agent is
 * not in the cache, or if its DB status is not "active". The trade
 * gate must call this — never fall back to a generic gate, never
 * silently default. Errors are typed so the trade gate can record a
 * structured skip reason.
 */
export class AgentNotExecutableError extends Error {
  constructor(
    public readonly reason:
      | "unknown_agent_id"
      | "missing_profile_id"
      | "non_active_db_status"
      | "profile_executes_false",
    public readonly agentId: number,
    public readonly profileId: string | null,
    public readonly dbStatus: string | null,
    detail: string,
  ) {
    super(detail);
    this.name = "AgentNotExecutableError";
  }
}

export function getCachedProfileForAgentId(agentId: number): {
  profile: AgentProfile;
  dbStatus: string;
  agentName: string;
} {
  const entry = CACHE.get(agentId);
  if (!entry) {
    throw new AgentNotExecutableError(
      "unknown_agent_id",
      agentId,
      null,
      null,
      `agentId ${agentId} not in registry cache (cache loaded_at=${LOADED_AT ?? "never"}). ` +
        `Reload via SIGHUP or restart.`,
    );
  }
  if (entry.dbStatus !== "active") {
    throw new AgentNotExecutableError(
      "non_active_db_status",
      agentId,
      entry.profile.agent_id,
      entry.dbStatus,
      `agent ${entry.agentName} (id=${agentId}, profile=${entry.profile.agent_id}) ` +
        `has db status=${entry.dbStatus}; only "active" rows may trade.`,
    );
  }
  if (!entry.profile.executes) {
    throw new AgentNotExecutableError(
      "profile_executes_false",
      agentId,
      entry.profile.agent_id,
      entry.dbStatus,
      `profile ${entry.profile.agent_id} has executes=false (status=${entry.profile.status})`,
    );
  }
  return {
    profile: entry.profile,
    dbStatus: entry.dbStatus,
    agentName: entry.agentName,
  };
}

/** Non-throwing variant — used by dashboard read paths and tests where
 * "no entry" is a legitimate state. */
export function tryGetCachedEntry(agentId: number): CacheEntry | null {
  return CACHE.get(agentId) ?? null;
}

export function getCacheStats(): { total: number; loaded_at: string | null } {
  return { total: CACHE.size, loaded_at: LOADED_AT };
}

/** Test helper — clears the cache between tests. */
export function _resetCacheForTests(): void {
  CACHE = new Map();
  LOADED_AT = null;
}

/** Test helper — pre-seed the cache without touching the DB. */
export function _seedCacheForTests(entries: CacheEntry[]): void {
  CACHE = new Map(entries.map((e) => [e.agentId, e]));
  LOADED_AT = new Date().toISOString();
}

// ────────────────────────── SIGHUP reload ──────────────────────────

let sighupRegistered = false;

/**
 * Install a SIGHUP handler that triggers a non-blocking cache reload.
 * Idempotent. Operators send SIGHUP to the api-server process to pick
 * up new agents.profile_id assignments without restarting (per task
 * spec — "no per-decision DB lookup, reload on admin signal").
 */
export function installSighupReload(): void {
  if (sighupRegistered) return;
  sighupRegistered = true;
  process.on("SIGHUP", () => {
    logger.info("Task #468 — SIGHUP received, reloading agent registry cache");
    void loadAgentRegistryCache().catch((err) =>
      logger.error({ err }, "agents-registry: SIGHUP reload failed"),
    );
  });
}
