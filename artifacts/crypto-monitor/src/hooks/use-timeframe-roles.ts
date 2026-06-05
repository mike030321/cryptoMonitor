/**
 * Task #551 — per-timeframe role registry hook.
 *
 * Reads `/api/crypto/timeframe-roles` (added in #550) and feeds the
 * dashboard's `<HorizonRolesCard/>`. The endpoint already fail-closes
 * (every TF reduced to `disabled, by_safety`) when the underlying
 * shared/timeframe-roles.json is missing or malformed, so the hook
 * does not need its own fallback document — it only has to keep the
 * react-query error path honest so the card can render the
 * fail-closed banner instead of a bogus empty list.
 *
 * Cadence matches the meta-brain status card (30s) so the truth row
 * (brain → meta-brain → horizon roles) refreshes in lockstep.
 */
import { useQuery } from "@tanstack/react-query";

export type TimeframeRole = "trade" | "shadow" | "context" | "disabled";
export type TimeframeContextSubkind = "filter" | "regime" | "risk_state";
export type TimeframeDisabledReason =
  | "by_data"
  | "by_gate"
  | "by_operator"
  | "by_safety";

export interface TimeframeRoleEntry {
  role: TimeframeRole;
  context_subkind: TimeframeContextSubkind | null;
  disabled_reason: TimeframeDisabledReason | null;
  reason: string;
  evidence_ref: string;
  last_reviewed_at: string;
  promoted_slices_in_tf: string[];
}

export interface TimeframeRolesDoc {
  schema_version: 1;
  generated_at: string;
  generated_by_task: string;
  timeframes: Record<string, TimeframeRoleEntry>;
}

export interface TimeframeRolesSummary {
  trade: number;
  shadow: number;
  context: number;
  disabled: number;
}

export interface TimeframeRolesResponse {
  document: TimeframeRolesDoc;
  summary: TimeframeRolesSummary;
}

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

export function useTimeframeRoles() {
  return useQuery<TimeframeRolesResponse>({
    queryKey: ["crypto-timeframe-roles"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/timeframe-roles`);
      if (!res.ok) throw new Error(`timeframe-roles ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
    staleTime: 30_000,
  });
}
