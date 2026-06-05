/**
 * Task #468 — boot-time DB sweep that populates `agents.profile_id` on
 * every row using the legacy-name compatibility map. The sweep is
 * idempotent and runs unconditionally on each boot — rows whose
 * `profile_id` already resolves to a known registry id are left
 * untouched.
 *
 * After the sweep, every consumer must read the profile through
 * `getAgentProfile(row.profile_id)` rather than `row.personality`.
 * `personality` remains a notNull text column for back-compat and is
 * kept synced with `display_name` (informational only).
 */

import { eq } from "drizzle-orm";
import { db, agentsTable } from "@workspace/db";
import { logger } from "../logger";
import { mapLegacyNameToProfileId } from "./compat";
import { getAgentProfile } from "./registry";

interface SweepResult {
  total: number;
  updated: number;
  unchanged: number;
  byProfile: Record<string, number>;
}

export async function syncAgentProfileIds(): Promise<SweepResult> {
  const rows = await db.select().from(agentsTable);
  const result: SweepResult = {
    total: rows.length,
    updated: 0,
    unchanged: 0,
    byProfile: {},
  };

  for (const row of rows) {
    const current = (row as { profileId?: string | null }).profileId ?? null;
    const resolved = mapLegacyNameToProfileId(row.name, current);
    // Defensive — confirm the resolved id is actually in the registry
    // (would only fail if the compat map drifted). throws on miss.
    getAgentProfile(resolved);

    result.byProfile[resolved] = (result.byProfile[resolved] ?? 0) + 1;

    if (current === resolved) {
      result.unchanged++;
      continue;
    }

    await db
      .update(agentsTable)
      .set({ profileId: resolved })
      .where(eq(agentsTable.id, row.id));
    result.updated++;
  }

  logger.info(
    {
      total: result.total,
      updated: result.updated,
      unchanged: result.unchanged,
      byProfile: result.byProfile,
    },
    "Task #468 — agents.profile_id sweep complete",
  );
  return result;
}
