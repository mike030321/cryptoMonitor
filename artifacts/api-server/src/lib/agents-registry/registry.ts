/**
 * Task #468 — v1 strategy-profile registry. Code/config-only; no LLM.
 *
 * The registry below is the single source of truth for the 5 v1 agents
 * plus the `legacy_archived` and `baseline_reference` non-executor
 * profiles. Behaviour values are copied verbatim from the task spec
 * v1 behaviour table — see `.local/tasks/task-468.md` lines 60-65.
 *
 * Adding or modifying a profile requires a code change (and a passing
 * contract test). There is no DB-mutable behaviour and no fallback to
 * a default profile. Lookups for unknown agent_ids throw — see
 * `getAgentProfile`. Profile values are read once at boot (and again on
 * SIGHUP-style admin reload via `reloadAgentRegistryCache`); there is
 * no per-decision DB lookup of these values.
 */

import {
  agentProfileListSchema,
  type AgentProfile,
} from "./schema";

// ────────────────────────── v1 profiles ──────────────────────────
// Values below are copied byte-for-byte from `.local/tasks/task-468.md`
// lines 60-65. Tests pin the values; do not edit without updating both.

const PROFILES_RAW: AgentProfile[] = [
  // ── Executor #1 — momentum_core ──────────────────────────────
  {
    agent_id: "momentum_core",
    display_name: "Momentum Core",
    thesis:
      "Trade with the dominant trend. Long when price action and " +
      "model edge agree on continuation; abstain in chop and panic " +
      "liquidations.",
    strategy_family: "momentum",
    preferred_regimes: ["trending_up", "trending_down", "high_vol_breakout"],
    blocked_regimes: [],
    min_confidence: 0.05,
    min_expected_edge_after_costs: 0.0001,
    abstain_bias: 1.00,
    size_bias: 1.00,
    pooled_fallback_penalty: 0.85,
    drawdown_sensitivity: 1.0,
    benchmark_sensitivity: 0.5,
    status: "active",
    executes: true,
    retirement_rule: {
      kind: "30d_cost_aware_threshold",
      da_threshold: 0.50,
      sharpe_threshold: 0.0,
      auto_flip: true,
    },
    created_at: "2026-04-24",
  },

  // ── Executor #2 — mean_reversion_core ────────────────────────
  {
    agent_id: "mean_reversion_core",
    display_name: "Mean Reversion Core",
    thesis:
      "Fade short-term extremes inside range-bound regimes. Sit out " +
      "trending markets where reversion does not work.",
    strategy_family: "mean_reversion",
    preferred_regimes: ["range_chop", "low_vol_compression"],
    blocked_regimes: [],
    min_confidence: 0.05,
    min_expected_edge_after_costs: 0.0001,
    abstain_bias: 1.15,
    size_bias: 0.85,
    pooled_fallback_penalty: 0.70,
    drawdown_sensitivity: 1.0,
    benchmark_sensitivity: 0.5,
    status: "active",
    executes: true,
    retirement_rule: {
      kind: "30d_cost_aware_threshold",
      da_threshold: 0.50,
      sharpe_threshold: 0.0,
      auto_flip: true,
    },
    created_at: "2026-04-24",
  },

  // ── Executor #3 — breakout_core ──────────────────────────────
  {
    agent_id: "breakout_core",
    display_name: "Breakout Core",
    thesis:
      "Buy expansions out of compression / consolidation when the " +
      "model confirms a directional thrust. Stand down in established " +
      "ranges and panic liquidations where breakouts are stale.",
    strategy_family: "breakout",
    preferred_regimes: ["low_vol_compression", "high_vol_breakout"],
    blocked_regimes: [],
    min_confidence: 0.05,
    min_expected_edge_after_costs: 0.0001,
    abstain_bias: 1.30,
    size_bias: 1.10,
    pooled_fallback_penalty: 0.60,
    drawdown_sensitivity: 1.0,
    benchmark_sensitivity: 0.5,
    status: "active",
    executes: true,
    retirement_rule: {
      kind: "30d_cost_aware_threshold",
      da_threshold: 0.50,
      sharpe_threshold: 0.0,
      auto_flip: true,
    },
    created_at: "2026-04-24",
  },

  // ── Executor #4 — volatility_defensive ───────────────────────
  // Note: the 90d_activation_count rule is review-only — the
  // nightly evaluator flags candidates on this profile but never
  // auto-flips status (see retirement.ts).
  {
    agent_id: "volatility_defensive",
    display_name: "Volatility Defensive",
    thesis:
      "Smaller, defensive sizing in elevated-vol regimes. Trades only " +
      "with high model conviction; abstains aggressively when edge is " +
      "marginal.",
    strategy_family: "volatility_forecaster",
    preferred_regimes: ["panic_liquidation", "high_vol_breakout"],
    blocked_regimes: [],
    min_confidence: 0.05,
    min_expected_edge_after_costs: 0.0001,
    abstain_bias: 1.30,
    size_bias: 0.50,
    pooled_fallback_penalty: 0.50,
    drawdown_sensitivity: 1.5,
    benchmark_sensitivity: 0.3,
    status: "active",
    executes: true,
    retirement_rule: {
      kind: "90d_activation_count",
      activation_threshold: 5,
      auto_flip: false,
      notes:
        "Review-only — flag on the dashboard when 90d closed-trade " +
        "count is below 5; do not auto-flip status.",
    },
    created_at: "2026-04-24",
  },

  // ── Non-executor: shared baseline reference ──────────────────
  // The single umbrella profile for ALL Strategy-Lab basket
  // strategies. Per spec (.local/tasks/task-468.md lines 39-41) the
  // v1 registry locks to exactly 5 family agents + this baseline +
  // `legacy_archived`; Strategy-Lab variant identity (Buy & Hold,
  // DCA+Circuit Breaker, Trend Filter) is retained as a SEPARATE
  // sub-id field — see `mapLegacyNameToSubId()` in compat.ts and the
  // `profileSubId` column on the dashboard payload — never as new
  // registry profile IDs. Cannot execute.
  {
    agent_id: "baseline_reference",
    display_name: "Baseline Reference",
    thesis:
      "Passive / mechanical reference strategy. Surfaced for " +
      "comparison only — never trades through the live decision " +
      "engine, and the dashboard treats it as a benchmark.",
    strategy_family: "baseline",
    preferred_regimes: "all",
    blocked_regimes: [],
    min_confidence: null,
    min_expected_edge_after_costs: null,
    abstain_bias: null,
    size_bias: null,
    pooled_fallback_penalty: null,
    drawdown_sensitivity: null,
    benchmark_sensitivity: null,
    status: "baseline",
    executes: false,
    retirement_rule: { kind: "never" },
    created_at: "2026-04-24",
  },

  // ── Non-executor: legacy archive bucket ──────────────────────
  // Catches every legacy personality that pre-dates this registry
  // (Momentum Mike, Sentiment Sarah, Pattern Pete, Hybrid-*, …)
  // plus anything the compat map cannot resolve. These rows still
  // exist in the DB so historical journals/predictions remain
  // queryable, but they cannot trade.
  {
    agent_id: "legacy_archived",
    display_name: "Legacy (archived)",
    thesis:
      "Legacy personality-style agent retained for historical " +
      "analytics. Cannot trade through the live decision engine.",
    strategy_family: "baseline",
    preferred_regimes: "all",
    blocked_regimes: [],
    min_confidence: null,
    min_expected_edge_after_costs: null,
    abstain_bias: null,
    size_bias: null,
    pooled_fallback_penalty: null,
    drawdown_sensitivity: null,
    benchmark_sensitivity: null,
    status: "legacy_archived",
    executes: false,
    retirement_rule: { kind: "never" },
    created_at: "2026-04-24",
  },
];

// Validate at module load — a typo or out-of-range number crashes the
// boot before a single agent can run a trade. This is the only place
// validation happens; consumers consume the typed result.
const PROFILES = agentProfileListSchema.parse(PROFILES_RAW);
const PROFILES_BY_ID = new Map<string, AgentProfile>(
  PROFILES.map((p) => [p.agent_id, p]),
);

// ────────────────────────── public API ──────────────────────────

/**
 * Resolve an agent profile by id. Throws if the id is not registered —
 * unknown ids must NEVER silently inherit a default profile (Task #468
 * acceptance test). The thrown error message includes a hint pointing
 * the operator at the registry file.
 */
export function getAgentProfile(agent_id: string | null | undefined): AgentProfile {
  if (!agent_id) {
    throw new Error(
      `getAgentProfile: empty agent_id — every agents row must carry a registry profile_id ` +
      `(see artifacts/api-server/src/lib/agents-registry/registry.ts; v1 ids: ` +
      `${listProfileIds().join(", ")})`,
    );
  }
  const p = PROFILES_BY_ID.get(agent_id);
  if (!p) {
    throw new Error(
      `getAgentProfile: unknown agent_id "${agent_id}" — registry lookups never default. ` +
      `Add the profile to artifacts/api-server/src/lib/agents-registry/registry.ts ` +
      `(or map the legacy name in compat.ts to legacy_archived). Known ids: ` +
      `${listProfileIds().join(", ")}`,
    );
  }
  return p;
}

/** Non-throwing variant — used only by the boot sweep / dashboard-listing
 * code where unknown ids are *expected* (e.g. legacy rows mid-migration).
 * The trade-execution path must call `getAgentProfile` (throws). */
export function tryGetAgentProfile(agent_id: string | null | undefined): AgentProfile | null {
  if (!agent_id) return null;
  return PROFILES_BY_ID.get(agent_id) ?? null;
}

export function listProfiles(): readonly AgentProfile[] {
  return PROFILES;
}

export function listProfileIds(): readonly string[] {
  return PROFILES.map((p) => p.agent_id);
}

export function listExecutingProfileIds(): readonly string[] {
  return PROFILES.filter((p) => p.executes).map((p) => p.agent_id);
}

/** True iff the regime is allowed by the profile's preferred/blocked list. */
export function profileAllowsRegime(
  profile: AgentProfile,
  regime: string | null | undefined,
): boolean {
  if (!regime) return true;
  if (profile.blocked_regimes.includes(regime as never)) return false;
  if (profile.preferred_regimes === "all") return true;
  return profile.preferred_regimes.includes(regime as never);
}
