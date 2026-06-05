/**
 * Task #550 — per-timeframe role layer.
 *
 * Source of truth for "what is this timeframe allowed to do today".
 * Read from shared/timeframe-roles.json; enforced by the brain
 * promotion gate (brain-promotion-gate.ts) AND the trade-execution
 * path (paper-trader.ts). The meta-brain adapter passes the role
 * through to ml-engine as `slice_role` (consumption is a separate
 * task — this module does not gate Python-side trust updates).
 *
 * 4-role enum, locked in this TS union AND the Zod schema below so
 * inflation by JSON edit is impossible:
 *
 *   - trade   — gate may permit trade execution (still requires
 *               per-slice promotion in the verification record).
 *   - shadow  — predictions logged but never executed.
 *   - context — predictions used as filter/regime/risk_state inputs
 *               but never themselves traded.
 *   - disabled — no predictions, no shadow, no context use.
 *
 * Sub-classifications are SEPARATE FIELDS (not new roles) so the
 * 4-role enum cannot be inflated by adding a new context flavor or
 * disabled reason.
 *
 * Failure mode: missing or malformed JSON ⇒ every TF is `disabled`
 * with `disabled_reason: by_safety`. Both gates refuse everything
 * until the file is restored. This is the "fail-closed" rule from
 * the task description (no-op default cannot be `trade` for any TF).
 *
 * Hot-reload: the file's mtime is checked on every read. Operators
 * can edit the file and both gates pick up the change on the next
 * request — no restart needed.
 */

import { readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { z } from "zod";
import { logger } from "./logger";

// ─────────────────── 4-role enum (locked) ────────────────────────

export const TIMEFRAME_ROLES = [
  "trade",
  "shadow",
  "context",
  "disabled",
] as const;
export type TimeframeRole = (typeof TIMEFRAME_ROLES)[number];

export const CONTEXT_SUBKINDS = ["filter", "regime", "risk_state"] as const;
export type ContextSubkind = (typeof CONTEXT_SUBKINDS)[number];

export const DISABLED_REASONS = [
  "by_data",
  "by_gate",
  "by_operator",
  "by_safety",
] as const;
export type DisabledReason = (typeof DISABLED_REASONS)[number];

// ─────────────────── Zod schema (validated on load) ──────────────

const timeframeEntrySchema = z
  .object({
    role: z.enum(TIMEFRAME_ROLES),
    context_subkind: z.enum(CONTEXT_SUBKINDS).nullable(),
    disabled_reason: z.enum(DISABLED_REASONS).nullable(),
    reason: z.string().min(1, "reason must be non-empty"),
    evidence_ref: z.string().min(1, "evidence_ref must be non-empty"),
    last_reviewed_at: z.string().min(1),
    promoted_slices_in_tf: z.array(z.string()),
  })
  .superRefine((entry, ctx) => {
    // context_subkind MUST be non-null iff role === 'context'.
    if (entry.role === "context" && entry.context_subkind === null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "context_subkind must be set when role='context'",
        path: ["context_subkind"],
      });
    }
    if (entry.role !== "context" && entry.context_subkind !== null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "context_subkind must be null when role!='context'",
        path: ["context_subkind"],
      });
    }
    // disabled_reason MUST be non-null iff role === 'disabled'.
    if (entry.role === "disabled" && entry.disabled_reason === null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "disabled_reason must be set when role='disabled'",
        path: ["disabled_reason"],
      });
    }
    if (entry.role !== "disabled" && entry.disabled_reason !== null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "disabled_reason must be null when role!='disabled'",
        path: ["disabled_reason"],
      });
    }
  });

export type TimeframeRoleEntry = z.infer<typeof timeframeEntrySchema>;

export const timeframeRolesDocSchema = z.object({
  schema_version: z.literal(1),
  generated_at: z.string().min(1),
  generated_by_task: z.string().min(1),
  timeframes: z.record(z.string(), timeframeEntrySchema),
});

export type TimeframeRolesDoc = z.infer<typeof timeframeRolesDocSchema>;

// ─────────────────── path resolution ─────────────────────────────

function findRolesFilePath(): string {
  // Walk up from this source file until we find pnpm-workspace.yaml,
  // then resolve shared/timeframe-roles.json under the workspace root.
  // Mirrors trading-constants.ts so the two configs live next to each
  // other and never disagree about which workspace they belong to.
  const here = path.dirname(fileURLToPath(import.meta.url));
  let cur = here;
  for (let i = 0; i < 8; i++) {
    try {
      readFileSync(path.join(cur, "pnpm-workspace.yaml"));
      return path.join(cur, "shared", "timeframe-roles.json");
    } catch {
      const parent = path.dirname(cur);
      if (parent === cur) break;
      cur = parent;
    }
  }
  throw new Error(
    `[timeframe-roles] could not locate workspace root (started at ${here})`,
  );
}

// ─────────────────── fail-closed default ─────────────────────────

// The static universe of supported timeframes — must match the JSON's
// keys for the steady state. If the file is missing/malformed, EVERY
// timeframe in this list defaults to `disabled (by_safety)`.
export const SUPPORTED_TIMEFRAMES = [
  "1m",
  "5m",
  "1h",
  "2h",
  "6h",
  "1d",
] as const;

function makeFailClosedDocument(): TimeframeRolesDoc {
  const nowIso = new Date().toISOString();
  const timeframes: Record<string, TimeframeRoleEntry> = {};
  for (const tf of SUPPORTED_TIMEFRAMES) {
    timeframes[tf] = {
      role: "disabled",
      context_subkind: null,
      disabled_reason: "by_safety",
      reason:
        "Fail-closed default: shared/timeframe-roles.json is missing or " +
        "malformed. Every timeframe is disabled until the file is restored.",
      evidence_ref: "fail-closed-default",
      last_reviewed_at: nowIso,
      promoted_slices_in_tf: [],
    };
  }
  return {
    schema_version: 1,
    generated_at: nowIso,
    generated_by_task: "fail-closed",
    timeframes,
  };
}

// ─────────────────── load + cache (mtime-watched) ────────────────

interface CacheEntry {
  doc: TimeframeRolesDoc;
  mtimeMs: number | null;
  loadedAt: number;
  failClosed: boolean;
}

let cache: CacheEntry | null = null;
// Cached resolved path — resolved once at module load. The workspace
// root never moves at runtime, so re-resolving on every read would be
// wasteful (and would mask a missing-workspace bug somewhere unhelpful).
let cachedPath: string | null = null;

function resolvedPath(): string {
  if (cachedPath === null) cachedPath = findRolesFilePath();
  return cachedPath;
}

function readAndValidate(filePath: string): TimeframeRolesDoc {
  const raw = readFileSync(filePath, "utf8");
  const parsed = JSON.parse(raw) as unknown;
  return timeframeRolesDocSchema.parse(parsed);
}

/**
 * Load the timeframe-roles document with mtime-watched caching.
 *
 * - First call: read from disk, validate, cache.
 * - Subsequent calls: stat the file; if mtime unchanged, return the
 *   cached doc; otherwise re-read and re-validate.
 * - On any failure (missing file, JSON parse error, schema rejection):
 *   warn-log and return the fail-closed document so both gates refuse
 *   every timeframe.
 *
 * This function NEVER throws — every error path produces a usable
 * fail-closed document so the trading-path code doesn't have to wrap
 * each call in a try/catch.
 */
export function loadTimeframeRoles(): TimeframeRolesDoc {
  let filePath: string;
  try {
    filePath = resolvedPath();
  } catch (err) {
    if (!cache?.failClosed) {
      logger.warn(
        { err: String(err) },
        "timeframe-roles: workspace root unreachable, returning fail-closed document",
      );
    }
    cache = {
      doc: makeFailClosedDocument(),
      mtimeMs: null,
      loadedAt: Date.now(),
      failClosed: true,
    };
    return cache.doc;
  }

  let mtimeMs: number | null;
  try {
    mtimeMs = statSync(filePath).mtimeMs;
  } catch (err) {
    if (!cache?.failClosed) {
      logger.warn(
        { err: String(err), filePath },
        "timeframe-roles: stat failed, returning fail-closed document",
      );
    }
    cache = {
      doc: makeFailClosedDocument(),
      mtimeMs: null,
      loadedAt: Date.now(),
      failClosed: true,
    };
    return cache.doc;
  }

  if (cache && cache.mtimeMs === mtimeMs && !cache.failClosed) {
    return cache.doc;
  }

  try {
    const doc = readAndValidate(filePath);
    if (cache?.failClosed) {
      logger.info(
        { filePath },
        "timeframe-roles: file restored, exiting fail-closed mode",
      );
    }
    cache = {
      doc,
      mtimeMs,
      loadedAt: Date.now(),
      failClosed: false,
    };
    return doc;
  } catch (err) {
    if (!cache?.failClosed) {
      logger.warn(
        { err: String(err), filePath },
        "timeframe-roles: load/validate failed, returning fail-closed document",
      );
    }
    cache = {
      doc: makeFailClosedDocument(),
      mtimeMs,
      loadedAt: Date.now(),
      failClosed: true,
    };
    return cache.doc;
  }
}

/**
 * Resolve the role for a given timeframe. Unknown timeframes (not
 * present in the loaded document) are treated as `disabled` so a
 * stray request can never silently bypass the role gate.
 */
export function getRoleForTimeframe(timeframe: string): TimeframeRole {
  const doc = loadTimeframeRoles();
  return doc.timeframes[timeframe]?.role ?? "disabled";
}

/**
 * Full role entry (with context_subkind / disabled_reason / reason)
 * for surfaced error text. Returns a synthesized fail-closed entry
 * when the timeframe is missing from the document.
 */
export function getRoleEntryForTimeframe(timeframe: string): TimeframeRoleEntry {
  const doc = loadTimeframeRoles();
  const entry = doc.timeframes[timeframe];
  if (entry) return entry;
  return {
    role: "disabled",
    context_subkind: null,
    disabled_reason: "by_safety",
    reason: `Timeframe ${timeframe} is not present in shared/timeframe-roles.json (treated as disabled).`,
    evidence_ref: "fail-closed-default",
    last_reviewed_at: new Date().toISOString(),
    promoted_slices_in_tf: [],
  };
}

/** Timeframes whose role is currently `trade`. */
export function getTradeRoleTimeframes(): string[] {
  const doc = loadTimeframeRoles();
  return Object.entries(doc.timeframes)
    .filter(([, e]) => e.role === "trade")
    .map(([tf]) => tf)
    .sort();
}

/** Per-role count summary for the read-only API endpoint. */
export interface TimeframeRolesSummary {
  trade: number;
  shadow: number;
  context: number;
  disabled: number;
}

export function summarizeTimeframeRoles(
  doc: TimeframeRolesDoc,
): TimeframeRolesSummary {
  const summary: TimeframeRolesSummary = {
    trade: 0,
    shadow: 0,
    context: 0,
    disabled: 0,
  };
  for (const entry of Object.values(doc.timeframes)) {
    summary[entry.role] += 1;
  }
  return summary;
}

/**
 * Test-only: reset the in-process cache. Production code never needs
 * to call this — the mtime watch handles real-world edits. Tests use
 * it to swap the underlying file between cases.
 */
export function _resetTimeframeRolesCacheForTests(): void {
  cache = null;
  cachedPath = null;
}
