/**
 * Task #468 — Strategy-profile registry: typed contract.
 *
 * The registry replaces the legacy "personality" string column as the
 * single source of truth for what an agent does, how it sizes, and
 * what regimes it should/should not trade in. Profiles are code/config
 * only — there is no LLM, no DB-mutable behaviour, no fallback to a
 * default profile when a lookup fails.
 *
 * The Zod schema below validates every profile in `registry.ts` at
 * module-load time. A typo or out-of-range value crashes the boot —
 * this is intentional. Strategy behaviour silently going wrong because
 * a config field accidentally became `null` is exactly the failure
 * mode this module exists to prevent.
 */

import { z } from "zod";
import { STRATEGY_FAMILIES } from "../meta-brain/contract";
import { REGIME_LABELS } from "../ml-client";

// ────────────────────────── enums ──────────────────────────

export const AGENT_STATUSES = [
  "active",
  "baseline",
  "quarantine_review",
  "disabled",
  "legacy_archived",
] as const;
export type AgentStatus = (typeof AGENT_STATUSES)[number];

// Profile family must be byte-for-byte identical to the meta-brain's
// frozen STRATEGY_FAMILIES list. The Zod schema enforces this so any
// drift between the two enums fails at boot, not in production.
export const agentFamilySchema = z.enum(STRATEGY_FAMILIES);
export type AgentFamily = z.infer<typeof agentFamilySchema>;

export const agentRegimeSchema = z.enum(
  REGIME_LABELS as [string, ...string[]],
);
export type AgentRegime = z.infer<typeof agentRegimeSchema>;

// ────────────────────────── retirement rule ──────────────────────────

const retirementRuleSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("30d_cost_aware_threshold"),
    da_threshold: z.number(),         // directional accuracy floor
    sharpe_threshold: z.number(),     // cost-aware sharpe floor
    // When true (default), nightly evaluator FLIPS agents.status to
    // `quarantine_review` on trigger. When false, the candidate is
    // surfaced for review only — used by `volatility_defensive` per
    // the v1 spec.
    auto_flip: z.boolean().optional().default(true),
    notes: z.string().optional(),
  }),
  z.object({
    kind: z.literal("90d_activation_count"),
    activation_threshold: z.number().int().nonnegative(),
    auto_flip: z.boolean().optional().default(false),
    notes: z.string().optional(),
  }),
  z.object({
    kind: z.literal("never"),
    notes: z.string().optional(),
  }),
]);

// ────────────────────────── profile ──────────────────────────

export const agentProfileSchema = z.object({
  // Stable identifier persisted in agents.profile_id and used as the
  // sole resolver key. Must never be reused for a different policy.
  agent_id: z.string().min(1),

  // Human-readable label surfaced on the dashboard.
  display_name: z.string().min(1),

  // One-paragraph description of the strategy intent.
  thesis: z.string().min(1),

  // Frozen family enum, enforced byte-for-byte with the meta-brain
  // contract via `agentFamilySchema`.
  strategy_family: agentFamilySchema,

  // Regime preferences — `preferred_regimes: "all"` means the agent is
  // willing to consider every regime. `blocked_regimes` is a hard veto
  // applied at the trade gate before the meta-model is consulted.
  preferred_regimes: z.union([
    z.literal("all"),
    z.array(agentRegimeSchema),
  ]),
  blocked_regimes: z.array(agentRegimeSchema),

  // Per-field policy. `null` means "inherit the fleet default" — the
  // gate code reads the fleet default rather than substituting 0.
  // Disabled / baseline / archived agents typically leave numeric
  // fields null because they cannot trade.
  min_confidence: z.number().min(0).max(1).nullable(),
  min_expected_edge_after_costs: z.number().nullable(),
  // abstain_bias is an "appetite-to-abstain" multiplier — values >1.0
  // bias the agent toward abstaining (used by the more cautious
  // executors per the v1 spec table).
  abstain_bias: z.number().min(0).max(2).nullable(),
  size_bias: z.number().min(0).max(2).nullable(),
  pooled_fallback_penalty: z.number().min(0).max(1).nullable(),
  drawdown_sensitivity: z.number().min(0).max(2).nullable(),
  benchmark_sensitivity: z.number().min(0).max(2).nullable(),

  // Lifecycle. `executes` MUST be false whenever status is anything
  // other than "active" — enforced by the post-parse refinement below.
  status: z.enum(AGENT_STATUSES),
  executes: z.boolean(),

  // Nightly retirement rule.
  retirement_rule: retirementRuleSchema,

  // Provenance.
  created_at: z.string(),
  notes: z.string().optional(),
})
.superRefine((p, ctx) => {
  if (p.executes && p.status !== "active") {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: `agent ${p.agent_id}: executes=true requires status="active" (got ${p.status})`,
    });
  }
  if (!p.executes && p.status === "active") {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: `agent ${p.agent_id}: status="active" requires executes=true`,
    });
  }
  // Active agents must declare numeric policy values (no null
  // inheritance for an executor — tests assert this contract).
  if (p.status === "active") {
    const required = [
      "min_confidence",
      "min_expected_edge_after_costs",
      "size_bias",
    ] as const;
    for (const f of required) {
      if (p[f] === null) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `agent ${p.agent_id}: status=active requires ${f} to be set (not null)`,
        });
      }
    }
  }
});

export type AgentProfile = z.infer<typeof agentProfileSchema>;

export const agentProfileListSchema = z
  .array(agentProfileSchema)
  .superRefine((profiles, ctx) => {
    const seen = new Set<string>();
    for (const p of profiles) {
      if (seen.has(p.agent_id)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `duplicate agent_id: ${p.agent_id}`,
        });
      }
      seen.add(p.agent_id);
    }
  });
