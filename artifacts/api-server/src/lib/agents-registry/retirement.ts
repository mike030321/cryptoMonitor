/**
 * Task #468 — nightly retirement evaluator.
 *
 * For every executor profile this job:
 *   1. Computes the rolling 30d cost-aware Sharpe + directional
 *      accuracy (DA) for the agent across all closed paper trades
 *      (the `30d_cost_aware_threshold` rule), or counts 90d closed-
 *      trade activations (the `90d_activation_count` rule for
 *      `volatility_defensive`).
 *   2. Compares against the profile's retirement_rule thresholds.
 *   3. If the rule triggers AND `auto_flip` is true, FLIPS every
 *      `agents` row carrying that `profile_id` from `status="active"`
 *      to `status="quarantine_review"` and emits a structured alert
 *      log row (the spec's "structured alert" — kept lightweight in
 *      v1, the operator restores via SIGHUP after manual review).
 *   4. The evaluator never auto-deletes rows.
 *
 * After any flip, `loadAgentRegistryCache()` is invoked so the trade
 * gate sees the new DB status before its next call.
 */

import { and, eq, gte, isNotNull } from "drizzle-orm";
import { db, agentsTable, paperTradesTable } from "@workspace/db";
import { logger } from "../logger";
import { ROUND_TRIP_COST_PCT } from "../trading-constants";
import { listProfiles } from "./registry";
import { loadAgentRegistryCache } from "./cache";

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;
const NINETY_DAYS_MS = 90 * 24 * 60 * 60 * 1000;

export interface RetirementCandidate {
  profile_id: string;
  display_name: string;
  rule_kind: "30d_cost_aware_threshold" | "90d_activation_count";
  closed_trades: number;
  directional_accuracy: number | null;
  cost_aware_sharpe: number | null;
  da_threshold: number | null;
  sharpe_threshold: number | null;
  activation_threshold: number | null;
  triggered_by: string[];
  auto_flipped: boolean;
  affected_agent_ids: number[];
  evaluated_at: string;
}

let lastSnapshot: { evaluated_at: string; candidates: RetirementCandidate[] } | null = null;

/**
 * Compute one nightly pass. Safe to invoke from a startup test.
 *
 * The function is intentionally side-effect-light when no rule
 * triggers: it only writes the in-memory snapshot. When a triggering
 * profile has `auto_flip=true`, it ALSO updates `agents.status` and
 * reloads the registry cache.
 */
export async function evaluateRetirementCandidates(): Promise<RetirementCandidate[]> {
  const evalStart = new Date();
  const candidates: RetirementCandidate[] = [];
  const allAgents = await db.select().from(agentsTable);

  let didFlip = false;

  for (const profile of listProfiles()) {
    if (profile.status !== "active") continue;
    if (profile.retirement_rule.kind === "never") continue;

    const matchedAgents = allAgents.filter((a) => {
      const id = (a as { profileId?: string | null }).profileId ?? null;
      return id === profile.agent_id;
    });
    if (matchedAgents.length === 0) continue;

    if (profile.retirement_rule.kind === "30d_cost_aware_threshold") {
      const cutoff = new Date(evalStart.getTime() - THIRTY_DAYS_MS);
      const da_threshold = profile.retirement_rule.da_threshold;
      const sharpe_threshold = profile.retirement_rule.sharpe_threshold;
      const auto_flip = profile.retirement_rule.auto_flip;

      let wins = 0;
      let total = 0;
      const returns: number[] = [];

      for (const agent of matchedAgents) {
        const trades = await db
          .select()
          .from(paperTradesTable)
          .where(
            and(
              eq(paperTradesTable.agentId, agent.id),
              eq(paperTradesTable.status, "closed"),
              isNotNull(paperTradesTable.pnl),
              gte(paperTradesTable.closedAt, cutoff),
            ),
          );
        for (const t of trades) {
          const pnl = (t as { pnl: number | null }).pnl;
          const positionSize = (t as { positionSize: number }).positionSize;
          if (pnl === null || positionSize <= 0) continue;
          total++;
          if (pnl > 0) wins++;
          const grossRet = pnl / positionSize;
          const costAdj = grossRet - ROUND_TRIP_COST_PCT;
          returns.push(costAdj);
        }
      }

      let da: number | null = null;
      let sharpe: number | null = null;
      if (total > 0) da = wins / total;
      if (returns.length > 1) {
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance =
          returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
        const stdev = Math.sqrt(variance);
        sharpe = stdev > 0 ? mean / stdev : null;
      }

      const triggered: string[] = [];
      if (da !== null && da < da_threshold) triggered.push("da");
      if (sharpe !== null && sharpe < sharpe_threshold) triggered.push("sharpe");

      // Spec step 8 — the dashboard surfaces LIVE retirement metric
      // values for every evaluable agent (DA, Sharpe, thresholds),
      // not only triggered exceptions. We always record an entry per
      // evaluable profile and let `triggered_by` indicate whether
      // the rule fired this evaluation.
      const affectedIds = matchedAgents
        .filter((a) => a.status === "active")
        .map((a) => a.id);
      let flipped = false;
      if (triggered.length > 0 && auto_flip && affectedIds.length > 0) {
        for (const aid of affectedIds) {
          await db
            .update(agentsTable)
            .set({ status: "quarantine_review" })
            .where(eq(agentsTable.id, aid));
        }
        didFlip = true;
        flipped = true;
        logger.warn(
          {
            event: "agent_retirement_quarantine",
            profile_id: profile.agent_id,
            display_name: profile.display_name,
            rule_kind: profile.retirement_rule.kind,
            triggered_by: triggered,
            directional_accuracy: da,
            cost_aware_sharpe: sharpe,
            da_threshold,
            sharpe_threshold,
            affected_agent_ids: affectedIds,
            evaluated_at: evalStart.toISOString(),
          },
          `Task #468 — retirement rule triggered, flipped ${affectedIds.length} agents to quarantine_review`,
        );
      }

      candidates.push({
        profile_id: profile.agent_id,
        display_name: profile.display_name,
        rule_kind: "30d_cost_aware_threshold",
        closed_trades: total,
        directional_accuracy: da,
        cost_aware_sharpe: sharpe,
        da_threshold,
        sharpe_threshold,
        activation_threshold: null,
        triggered_by: triggered,
        auto_flipped: flipped,
        // Spec step 8 — every DB row matched to this profile gets
        // the live metric values surfaced on the dashboard, not
        // just the active ones. Quarantined / disabled rows still
        // show "DA 0.42 / threshold 0.50" so operators can see WHY
        // a row was flipped.
        affected_agent_ids: matchedAgents.map((a) => a.id),
        evaluated_at: evalStart.toISOString(),
      });
      continue;
    }

    if (profile.retirement_rule.kind === "90d_activation_count") {
      // Volatility-defensive uses this rule. Spec line 64 says
      // "→ quarantine_review (not auto-retire)" — so we surface the
      // candidate but never flip status. The schema default keeps
      // `auto_flip=false`; we still respect it explicitly.
      const cutoff = new Date(evalStart.getTime() - NINETY_DAYS_MS);
      const threshold = profile.retirement_rule.activation_threshold;
      const auto_flip = profile.retirement_rule.auto_flip;

      let activations = 0;
      for (const agent of matchedAgents) {
        const trades = await db
          .select()
          .from(paperTradesTable)
          .where(
            and(
              eq(paperTradesTable.agentId, agent.id),
              eq(paperTradesTable.status, "closed"),
              gte(paperTradesTable.closedAt, cutoff),
            ),
          );
        activations += trades.length;
      }
      // Spec step 8 — always emit a per-profile candidate row so the
      // dashboard can surface the live `activations vs threshold`
      // values for every evaluable agent. `triggered_by` is only
      // populated when the rule actually fires.
      const ruleTriggered = activations < threshold;

      const affectedIds = matchedAgents
        .filter((a) => a.status === "active")
        .map((a) => a.id);
      let flipped = false;
      if (ruleTriggered && auto_flip && affectedIds.length > 0) {
        for (const aid of affectedIds) {
          await db
            .update(agentsTable)
            .set({ status: "quarantine_review" })
            .where(eq(agentsTable.id, aid));
        }
        didFlip = true;
        flipped = true;
      }
      if (ruleTriggered) {
        logger.warn(
          {
            event: "agent_retirement_review",
            profile_id: profile.agent_id,
            display_name: profile.display_name,
            rule_kind: profile.retirement_rule.kind,
            activations_90d: activations,
            activation_threshold: threshold,
            affected_agent_ids: affectedIds,
            auto_flipped: flipped,
            evaluated_at: evalStart.toISOString(),
          },
          `Task #468 — 90d activation count below threshold (review-only)`,
        );
      }

      candidates.push({
        profile_id: profile.agent_id,
        display_name: profile.display_name,
        rule_kind: "90d_activation_count",
        closed_trades: activations,
        directional_accuracy: null,
        cost_aware_sharpe: null,
        da_threshold: null,
        sharpe_threshold: null,
        activation_threshold: threshold,
        triggered_by: ruleTriggered ? ["activation_count"] : [],
        auto_flipped: flipped,
        affected_agent_ids: matchedAgents.map((a) => a.id),
        evaluated_at: evalStart.toISOString(),
      });
    }
  }

  if (didFlip) {
    // Refresh the trade-gate cache so the next decision sees the
    // newly-quarantined agents as non-executable.
    await loadAgentRegistryCache().catch((err) =>
      logger.error({ err }, "agents-registry: post-flip cache reload failed"),
    );
  }

  lastSnapshot = {
    evaluated_at: evalStart.toISOString(),
    candidates,
  };
  logger.info(
    {
      count: candidates.length,
      flipped: candidates.filter((c) => c.auto_flipped).length,
      candidates: candidates.map((c) => c.profile_id),
    },
    "Task #468 — nightly retirement evaluation complete",
  );
  return candidates;
}

/** Returns the most recent snapshot, or `null` if the job has not yet run. */
export function getRetirementSnapshot():
  | { evaluated_at: string; candidates: RetirementCandidate[] }
  | null {
  return lastSnapshot;
}

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
let timer: NodeJS.Timeout | null = null;

/**
 * Start the nightly loop. Idempotent; subsequent calls are no-ops.
 * The first pass runs ~10 seconds after boot so the dashboard has a
 * snapshot quickly without blocking startup.
 */
export function startRetirementLoop(): void {
  if (timer) return;
  setTimeout(() => {
    void evaluateRetirementCandidates().catch((err) =>
      logger.error({ err }, "retirement evaluation failed (startup pass)"),
    );
  }, 10_000);
  timer = setInterval(() => {
    void evaluateRetirementCandidates().catch((err) =>
      logger.error({ err }, "retirement evaluation failed (nightly pass)"),
    );
  }, ONE_DAY_MS);
}

/** Test helper — clears the in-memory snapshot. */
export function _resetRetirementForTests(): void {
  lastSnapshot = null;
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}
