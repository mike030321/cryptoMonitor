/**
 * Hand-authored zod contract for `GET /api/crypto/timeframe-roles`.
 *
 * Lives outside the orval-generated `./generated/` tree so it survives
 * codegen runs. Kept in lockstep with the runtime loader's enum lists
 * in `artifacts/api-server/src/lib/timeframe-roles.ts` — both the
 * server-side validator and this client-facing schema must accept the
 * same 4 roles, 3 context subkinds, and 4 disabled reasons. Drifting
 * either side without the other would let a server response that the
 * loader accepts fail at the API boundary, or vice versa.
 */
import { z } from "zod";

export const TimeframeRoleEnum = z.enum([
  "trade",
  "shadow",
  "context",
  "disabled",
]);
export type TimeframeRoleApi = z.infer<typeof TimeframeRoleEnum>;

export const TimeframeContextSubkindEnum = z.enum([
  "filter",
  "regime",
  "risk_state",
]);

export const TimeframeDisabledReasonEnum = z.enum([
  "by_data",
  "by_gate",
  "by_operator",
  "by_safety",
]);

export const TimeframeRoleEntry = z.object({
  role: TimeframeRoleEnum,
  context_subkind: TimeframeContextSubkindEnum.nullable(),
  disabled_reason: TimeframeDisabledReasonEnum.nullable(),
  reason: z.string().min(1),
  evidence_ref: z.string().min(1),
  last_reviewed_at: z.string().min(1),
  promoted_slices_in_tf: z.array(z.string()),
});

export const TimeframeRolesDocument = z.object({
  schema_version: z.literal(1),
  generated_at: z.string().min(1),
  generated_by_task: z.string().min(1),
  timeframes: z.record(z.string(), TimeframeRoleEntry),
});

export const TimeframeRolesSummary = z.object({
  trade: z.number().int().nonnegative(),
  shadow: z.number().int().nonnegative(),
  context: z.number().int().nonnegative(),
  disabled: z.number().int().nonnegative(),
});

export const TimeframeRolesResponse = z.object({
  document: TimeframeRolesDocument,
  summary: TimeframeRolesSummary,
});

export type TimeframeRolesResponseT = z.infer<typeof TimeframeRolesResponse>;
