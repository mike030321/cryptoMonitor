/**
 * Task #468 — legacy name → registry profile_id compatibility map.
 *
 * The DB still holds rows keyed on the old personality-style names
 * ("Slice 1m", "Strategy Lab Buy & Hold", "Hybrid-foo-bar", …). Boot
 * sweeps every agents row through this function and writes the
 * resolved profile_id back into the new `profile_id` column. After
 * the sweep, every consumer reads `agent.profile_id` and looks up the
 * profile through `getAgentProfile()` — never `agent.personality`.
 *
 * Unknown names map to `legacy_archived` (a non-executor profile) so
 * historical journals stay queryable but the agent cannot trade. This
 * is the ONE place where unknown-name handling defaults; every OTHER
 * lookup site must use `getAgentProfile()`, which throws.
 *
 * Per spec (.local/tasks/task-468.md lines 39-41) the v1 registry
 * locks to exactly the 5 family agents + `baseline_reference` +
 * `legacy_archived`. Strategy-Lab variant identity (Buy & Hold,
 * DCA + Circuit Breaker, Trend Filter) is preserved as a SEPARATE
 * sub-id string — see `mapLegacyNameToSubId()` below — never as new
 * registry profile IDs. The dashboard surfaces the sub-id alongside
 * the resolved profile so operators see "baseline_reference /
 * baseline_buy_hold" without losing variant attribution.
 */

import { listProfileIds } from "./registry";

const LEGACY_ARCHIVED = "legacy_archived";
const BASELINE_REFERENCE = "baseline_reference";

// ── exact-name matches (case-insensitive) ────────────────────────
//
// Every Strategy-Lab variant resolves to the umbrella
// `baseline_reference` profile id; the variant tag is recovered via
// `mapLegacyNameToSubId()` so the dashboard can render it.
const EXACT_MATCHES = new Map<string, string>([
  // Strategy-Lab variants — all umbrella'd under baseline_reference.
  ["dca + circuit breaker", BASELINE_REFERENCE],
  ["strategy lab dca + circuit breaker", BASELINE_REFERENCE],
  ["dca circuit breaker", BASELINE_REFERENCE],
  ["strategy lab buy & hold", BASELINE_REFERENCE],
  ["strategy lab buy and hold", BASELINE_REFERENCE],
  ["buy & hold", BASELINE_REFERENCE],
  ["buy and hold", BASELINE_REFERENCE],
  ["trend filter (30d basket)", BASELINE_REFERENCE],
  ["strategy lab trend filter (30d basket)", BASELINE_REFERENCE],
  ["trend filter", BASELINE_REFERENCE],
  // Generic baseline / benchmark name (no variant identity implied).
  ["baseline", BASELINE_REFERENCE],
  ["benchmark", BASELINE_REFERENCE],
]);

// Sub-id retention table — maps the same legacy names to a stable
// variant tag string (per spec lines 39-41). The sub-id string is
// NOT a registered profile_id; it's metadata surfaced on the
// dashboard alongside the umbrella profile so analysts can still
// distinguish baselines.
const SUB_ID_MATCHES = new Map<string, string>([
  ["strategy lab buy & hold", "baseline_buy_hold"],
  ["strategy lab buy and hold", "baseline_buy_hold"],
  ["buy & hold", "baseline_buy_hold"],
  ["buy and hold", "baseline_buy_hold"],
  ["dca + circuit breaker", "baseline_dca_cb"],
  ["strategy lab dca + circuit breaker", "baseline_dca_cb"],
  ["dca circuit breaker", "baseline_dca_cb"],
  ["trend filter (30d basket)", "baseline_trend_filter"],
  ["strategy lab trend filter (30d basket)", "baseline_trend_filter"],
  ["trend filter", "baseline_trend_filter"],
]);

// Production agent names that are known legacy LLM personalities or
// deterministic placeholders. Listed explicitly so the prod-name
// coverage test enumerates them and so future renames stay flagged.
// All map to legacy_archived (cannot trade).
const KNOWN_LEGACY_NAMES = new Set<string>([
  "sentiment sarah",
  "pattern pete",
  "momentum mike",
  "contrarian carol",
  "slice 1m",
  "slice 5m",
  "slice 15m",
  "slice 1h",
  "slice 4h",
  "slice 1d",
]);

const KNOWN_PROFILE_IDS = new Set<string>(listProfileIds());

/**
 * Resolve a legacy agent name → registry profile_id. Never throws;
 * unknown names fall through to `legacy_archived` so the boot-time
 * sweep can complete on any historical DB.
 *
 * Optionally pass `currentProfileId` (the value already stored on the
 * row) — if it's a known registry id, it is preserved verbatim. This
 * keeps the compat sweep idempotent across boots and lets operators
 * promote a row by writing the new id directly.
 */
export function mapLegacyNameToProfileId(
  name: string | null | undefined,
  currentProfileId?: string | null | undefined,
): string {
  if (currentProfileId && KNOWN_PROFILE_IDS.has(currentProfileId)) {
    return currentProfileId;
  }
  // If the row already carries one of the v1 ids verbatim in `name`
  // (e.g. agents seeded after this task lands), preserve it.
  if (name && KNOWN_PROFILE_IDS.has(name)) {
    return name;
  }

  const n = (name ?? "").trim().toLowerCase();
  if (!n) return LEGACY_ARCHIVED;

  const exact = EXACT_MATCHES.get(n);
  if (exact) return exact;

  // Pre-#468 LLM personalities (Sentiment Sarah, Pattern Pete,
  // Hybrid-*, Momentum Mike, …) and the deterministic "Slice Xm"
  // placeholders are all archived. These rows still exist for
  // historical journal queries but cannot trade.
  if (KNOWN_LEGACY_NAMES.has(n)) return LEGACY_ARCHIVED;
  if (n.startsWith("hybrid-")) return LEGACY_ARCHIVED;
  if (/^slice\s+\d+[mhd]$/.test(n)) return LEGACY_ARCHIVED;

  // Anything we have never seen — still archive (the registry never
  // silently defaults a NEW name into an executor).
  return LEGACY_ARCHIVED;
}

/**
 * Resolve a legacy agent name → variant sub-id (the spec's "with
 * sub-id retained as baseline_buy_hold / baseline_dca_cb /
 * baseline_trend_filter"). Returns null for any name that is not a
 * Strategy-Lab variant — the dashboard shows the sub-id only when
 * present.
 *
 * Sub-ids are NOT registry profile IDs; they are metadata surfaced
 * on the dashboard alongside the umbrella `baseline_reference`
 * profile so analysts retain variant attribution.
 */
export function mapLegacyNameToSubId(
  name: string | null | undefined,
): string | null {
  const n = (name ?? "").trim().toLowerCase();
  if (!n) return null;
  return SUB_ID_MATCHES.get(n) ?? null;
}

/** Test / dashboard helper — exposes the production-name catalog so
 * the prod-name coverage test can enumerate every legacy name we
 * know about. */
export function listKnownLegacyNamesForTests(): readonly string[] {
  return [...EXACT_MATCHES.keys(), ...KNOWN_LEGACY_NAMES];
}
