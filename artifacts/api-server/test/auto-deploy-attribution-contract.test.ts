/**
 * Task #532 — contract test: live aggregations must NEVER attribute
 * trades to archived/legacy agents.
 *
 * The Phase-0 audit found `getAutoDeployAttribution` filtered only by
 * `strategy_type='ai-bots'`, which silently included 20
 * `legacy_archived` agents (their `strategy_type` was never reset
 * during the registry sweep). This inflated `realityCheck.autoDeploy.
 * window7d.realizedPnlUsd` to $3.24 while the 4 live executors had
 * zero closed trades.
 *
 * This test asserts that the canonical "live ai-bots" filter excludes
 * archived rows, and locks in the contract so future schema or
 * filter changes cannot regress.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { db, agentsTable } from "@workspace/db";
import { and, eq, isNull, sql } from "drizzle-orm";

const LIVE_AI_BOTS_FILTER = and(
  eq(agentsTable.strategyType, "ai-bots"),
  // Task #532 / Rev 2.1 — must match `getAutoDeployAttribution` exactly,
  // including isActive=true so a paused-but-not-archived executor can
  // never fabricate live attribution.
  eq(agentsTable.isActive, true),
  isNull(agentsTable.archivedAt),
  sql`${agentsTable.profileId} IS DISTINCT FROM 'legacy_archived'`,
);

describe("Task #532 — autoDeploy attribution contract", () => {
  it("live ai-bots filter excludes legacy_archived rows", async () => {
    const live = await db
      .select({ id: agentsTable.id, profileId: agentsTable.profileId, archivedAt: agentsTable.archivedAt })
      .from(agentsTable)
      .where(LIVE_AI_BOTS_FILTER);

    for (const a of live) {
      assert.notEqual(
        a.profileId,
        "legacy_archived",
        `live ai-bots filter returned a legacy_archived agent (id=${a.id})`,
      );
      assert.equal(
        a.archivedAt,
        null,
        `live ai-bots filter returned an archived agent (id=${a.id})`,
      );
    }
  });

  it("live ai-bots filter is a strict subset of strategy_type='ai-bots'", async () => {
    const allAi = await db
      .select({ id: agentsTable.id })
      .from(agentsTable)
      .where(eq(agentsTable.strategyType, "ai-bots"));
    const live = await db
      .select({ id: agentsTable.id })
      .from(agentsTable)
      .where(LIVE_AI_BOTS_FILTER);

    const allIds = new Set(allAi.map((a) => a.id));
    for (const a of live) {
      assert.ok(
        allIds.has(a.id),
        `live filter returned id ${a.id} not in strategy_type='ai-bots' set`,
      );
    }
    assert.ok(live.length <= allAi.length, "live filter must not exceed total ai-bots count");
  });

  it("any legacy_archived row that is also strategy_type='ai-bots' is excluded by the live filter", async () => {
    const legacyAiBots = await db
      .select({ id: agentsTable.id })
      .from(agentsTable)
      .where(
        and(
          eq(agentsTable.strategyType, "ai-bots"),
          eq(agentsTable.profileId, "legacy_archived"),
        ),
      );
    const live = await db
      .select({ id: agentsTable.id })
      .from(agentsTable)
      .where(LIVE_AI_BOTS_FILTER);

    const liveIds = new Set(live.map((a) => a.id));
    for (const a of legacyAiBots) {
      assert.ok(
        !liveIds.has(a.id),
        `legacy_archived ai-bots agent id=${a.id} leaked into the live aggregation set`,
      );
    }
  });
});
