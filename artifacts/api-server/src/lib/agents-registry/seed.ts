/**
 * Task #512 — boot-time seed for the deterministic executor fleet.
 *
 * Inserts one `agents` row per registry-executor profile (momentum_core,
 * mean_reversion_core, breakout_core, volatility_defensive) if missing.
 * Idempotent: rows whose `profile_id` already matches an executor are
 * left untouched. Uses each profile's `display_name` and a `1h`
 * preferred timeframe (the live monitor fans out per-(coin, timeframe)
 * on its own — the seed is timeframe-agnostic).
 *
 * Runs in the boot sequence BEFORE `initializePaperPortfolios()` so the
 * paper-portfolio init step seeds a $1000 starting balance for each new
 * executor row in the same boot.
 *
 * The seed never touches legacy rows. The Task #512 archive sweep
 * (migration 009) is the only place that flips legacy rows to
 * `archivedAt = NOW(), isActive = false`.
 */

import { eq } from "drizzle-orm";
import { db, agentsTable } from "@workspace/db";
import { logger } from "../logger";
import { listExecutingProfileIds, getAgentProfile } from "./registry";

interface SeedResult {
  inserted: number;
  existed: number;
  byProfile: Record<string, "inserted" | "existed">;
}

/**
 * Idempotent boot seed for the 4 deterministic executor agents. Returns
 * a structured summary so the boot sequence can log how many rows were
 * created vs already present.
 */
export async function seedExecutorAgents(): Promise<SeedResult> {
  const result: SeedResult = { inserted: 0, existed: 0, byProfile: {} };

  for (const profileId of listExecutingProfileIds()) {
    const profile = getAgentProfile(profileId);
    const existing = await db
      .select()
      .from(agentsTable)
      .where(eq(agentsTable.profileId, profileId));

    if (existing.length > 0) {
      result.existed++;
      result.byProfile[profileId] = "existed";
      continue;
    }

    await db.insert(agentsTable).values({
      name: profile.display_name,
      personality: `${profile.display_name} (registry executor)`,
      score: 100,
      totalPredictions: 0,
      correctPredictions: 0,
      wrongPredictions: 0,
      streak: 0,
      streakType: "none",
      status: "active",
      isActive: true,
      preferredTimeframes: "1h",
      strategyType: "ai-bots",
      profileId,
    });
    result.inserted++;
    result.byProfile[profileId] = "inserted";
  }

  logger.info(
    { inserted: result.inserted, existed: result.existed, byProfile: result.byProfile },
    "Task #512 — executor agent seed complete",
  );
  return result;
}
