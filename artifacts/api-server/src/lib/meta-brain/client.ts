/**
 * HTTP client for the ml-engine's /meta-brain/* endpoints.
 *
 * All failures (disabled, timeout, network, 4xx/5xx, schema-invalid
 * response, allocation drift) collapse to the neutral directive via
 * ./fallback. The trading path is never stalled.
 *
 * Observability (Task #381 step 9): per-cause fallback counters and
 * sampled evaluate-latency p50/p95 are exposed via consume*() helpers
 * called by the adapter at flush time.
 */

import { logFallbackOnce } from "./fallback";
import {
  allocationSumsToOne,
  MetaBrainBatch,
  MetaBrainDirective,
  MetaBrainDirectiveSchema,
  neutralDirective,
  type SliceRole,
} from "./contract";
import { getRoleForTimeframe } from "../timeframe-roles";
import { recordDisabledOutcomeRejection } from "../disabled-outcome-notifier";

const DEFAULT_TIMEOUT_MS = 250;

function mlBaseUrl(): string {
  return process.env.ML_ENGINE_URL || "http://localhost:8000";
}

function isEnabled(): boolean {
  return process.env.META_BRAIN_ENABLED === "1" ||
    process.env.META_BRAIN_SHADOW === "1";
}

function shadowOnly(): boolean {
  return (
    process.env.META_BRAIN_SHADOW === "1" &&
    process.env.META_BRAIN_ENABLED !== "1"
  );
}

// ─────────────── per-cycle observability counters ───────────────────

const fallbackCounters: Record<string, number> = {};
function bumpCause(cause: string): void {
  fallbackCounters[cause] = (fallbackCounters[cause] ?? 0) + 1;
}
export function consumeFallbackCounters(): Record<string, number> {
  const snap = { ...fallbackCounters };
  for (const k of Object.keys(fallbackCounters)) delete fallbackCounters[k];
  return snap;
}

const LATENCY_SAMPLE_CAP = 256;
const latencySamples: number[] = [];
function sampleLatency(ms: number): void {
  if (latencySamples.length >= LATENCY_SAMPLE_CAP) latencySamples.shift();
  latencySamples.push(ms);
}
export function consumeLatencySamples(): {
  p50: number;
  p95: number;
  count: number;
} {
  const n = latencySamples.length;
  if (n === 0) return { p50: 0, p95: 0, count: 0 };
  const sorted = [...latencySamples].sort((a, b) => a - b);
  const p50 = sorted[Math.floor((n - 1) * 0.5)];
  const p95 = sorted[Math.floor((n - 1) * 0.95)];
  latencySamples.length = 0;
  return { p50, p95, count: n };
}

export async function postEvaluate(
  batch: MetaBrainBatch,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<MetaBrainDirective> {
  if (!isEnabled()) {
    bumpCause("disabled");
    return neutralDirective("disabled");
  }
  const start = Date.now();
  let response: Response;
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    response = await fetch(
      `${mlBaseUrl()}/ml/meta-brain/evaluate`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(batch),
        signal: ctrl.signal,
      },
    );
  } catch (err) {
    bumpCause("fetch_failed");
    logFallbackOnce("fetch_failed", { err: String(err) });
    return neutralDirective("fetch_failed");
  } finally {
    clearTimeout(t);
    sampleLatency(Date.now() - start);
  }

  if (!response.ok) {
    bumpCause(`http_${response.status}`);
    logFallbackOnce("http_not_ok", { status: response.status });
    return neutralDirective(`http_${response.status}`);
  }

  let raw: unknown;
  try {
    raw = await response.json();
  } catch (err) {
    bumpCause("bad_json");
    logFallbackOnce("bad_json", { err: String(err) });
    return neutralDirective("bad_json");
  }

  const parsed = MetaBrainDirectiveSchema.safeParse(raw);
  if (!parsed.success) {
    bumpCause("schema_invalid");
    logFallbackOnce("schema_invalid", {
      issues: parsed.error.issues.slice(0, 3),
    });
    return neutralDirective("schema_invalid");
  }

  if (!allocationSumsToOne(parsed.data.allocation_weight)) {
    bumpCause("allocation_drift");
    logFallbackOnce("allocation_drift", {
      sum: Object.values(parsed.data.allocation_weight).reduce(
        (s, v) => s + v,
        0,
      ),
    });
    return neutralDirective("allocation_drift");
  }

  if (shadowOnly()) {
    // Shadow mode design contract (Task #381 step 8): re-key the
    // tick_id with a `shadow:` prefix so that
    //   - `isNeutralDirective` returns true (sizing is unaffected),
    //   - `bindTradeToTick` still records the binding (record_outcome
    //     can close the loop),
    //   - the underlying real uuid is preserved for record_outcome.
    return {
      ...neutralDirective("shadow"),
      tick_id: `shadow:${parsed.data.tick_id}`,
    };
  }

  return parsed.data;
}

export interface MetaBrainOutcome {
  tick_id: string;
  /**
   * Task #574 — the timeframe whose role registry entry is resolved at
   * outcome-submission time (NOT at trade-open time) and sent on the
   * wire as `slice_role`. The Python brain uses it to gate trust
   * updates: only `trade` outcomes feed the trust model; `shadow` and
   * `context` are stored in `inputs_by_role` only; `disabled` is
   * rejected with `{ok:false, reason:"disabled_role_rejected"}`.
   *
   * Resolving at submission time (rather than capturing the role at
   * open time) is intentional: if the operator has flipped the
   * timeframe's role mid-trade (e.g. `trade` → `shadow` or `disabled`)
   * the realized outcome must respect the CURRENT role, not the role
   * that was in effect when the trade was opened.
   */
  timeframe: string;
  /**
   * Task #577 — optional slice id captured at trade-open time. The
   * Python brain echoes this back in its `[disabled_outcome_received]`
   * warn and the api-server's disabled-outcome notifier surfaces it
   * in the dashboard banner + webhook payload so an operator can
   * correlate a leaking outcome to the originating slice in one hop.
   * The api-server doesn't currently track slice ids end-to-end, so
   * this is permitted to be omitted; downstream surfaces will display
   * "—" instead.
   */
  sliceId?: string | null;
  timestamp?: string;
  outcome: {
    realized_pnl: number;
    realized_drawdown: number;
    realized_stability: number;
    turnover_cost: number;
    action_churn: number | null;
    correct_defense: number | null;
    correct_suppression: number | null;
    missed_edge_cost: number | null;
  };
}

/**
 * Wire payload for `/ml/meta-brain/record-outcome`. Mirrors the
 * Python-side `record_outcome` payload contract documented in
 * `artifacts/ml-engine/app/meta_brain.py`. `slice_role` is required
 * on the wire (Task #574) — sending it as missing triggers the
 * one-time `[ROLE_BACKCOMPAT_DEFAULT]` warn on the brain side and
 * silently defaults to `trade`, defeating role isolation.
 */
interface RecordOutcomeWirePayload {
  tick_id: string;
  slice_role: SliceRole;
  /** Task #577 — optional, forwarded through so the brain can echo it
   *  back in the `[disabled_outcome_received]` warn. */
  slice_id?: string;
  timestamp?: string;
  outcome: {
    realized_pnl: number;
    realized_drawdown: number;
    realized_stability: number;
    turnover_cost: number;
    action_churn: number;
    correct_defense: number;
    correct_suppression: number;
    missed_edge_cost: number;
  };
}

/** Returns true iff the brain confirmed `{ok: true}`. The adapter uses
 * this to clear the trade→tick binding only on success, enabling
 * retries on transient failures. */
export async function postRecordOutcome(
  payload: MetaBrainOutcome,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<boolean> {
  if (!isEnabled()) return false;
  if (payload.tick_id.startsWith("neutral:")) {
    // Fabricated neutral tick — the brain never saw this one.
    return false;
  }
  // Strip the shadow: prefix on the wire — the brain expects the
  // underlying uuid it issued.
  const wireTickId = payload.tick_id.startsWith("shadow:")
    ? payload.tick_id.slice("shadow:".length)
    : payload.tick_id;

  // Task #574 — resolve `slice_role` at outcome-submission time
  // (not at trade-open time) so a role flip mid-trade is honoured.
  // `getRoleForTimeframe` is fail-closed: an unknown timeframe or a
  // missing/malformed roles file yields `disabled`, which the brain
  // rejects — strictly safer than silently defaulting to `trade`.
  const sliceRole = getRoleForTimeframe(payload.timeframe);

  // Send any null-valued outcome dimensions as 0.0 for the wire (the
  // vendored Python dataclass requires float). The brain's bounded
  // plasticity is robust to sparse signal; real trackers for these
  // dimensions are explicit follow-ups.
  const wireBody: RecordOutcomeWirePayload = {
    tick_id: wireTickId,
    slice_role: sliceRole,
    timestamp: payload.timestamp,
    outcome: {
      realized_pnl: payload.outcome.realized_pnl,
      realized_drawdown: payload.outcome.realized_drawdown,
      realized_stability: payload.outcome.realized_stability,
      turnover_cost: payload.outcome.turnover_cost,
      action_churn: payload.outcome.action_churn ?? 0,
      correct_defense: payload.outcome.correct_defense ?? 0,
      correct_suppression: payload.outcome.correct_suppression ?? 0,
      missed_edge_cost: payload.outcome.missed_edge_cost ?? 0,
    },
  };
  // Task #577 — forward the optional slice_id so the brain's
  // `[disabled_outcome_received]` warn includes it. Omit when null/
  // absent rather than send an explicit null, matching the Python
  // contract's "optional" shape.
  if (typeof payload.sliceId === "string" && payload.sliceId.length > 0) {
    wireBody.slice_id = payload.sliceId;
  }

  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(
      `${mlBaseUrl()}/ml/meta-brain/record-outcome`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(wireBody),
        signal: ctrl.signal,
      },
    );
    if (!res.ok) {
      bumpCause(`record_outcome_http_${res.status}`);
      logFallbackOnce("record_outcome_not_ok", { status: res.status });
      return false;
    }
    let body: { ok?: boolean; reason?: string } = {};
    try {
      body = (await res.json()) as { ok?: boolean; reason?: string };
    } catch {
      bumpCause("record_outcome_bad_json");
      return false;
    }
    // Task #577 — surface the brain's `disabled_role_rejected` reason
    // as a structured operator-facing event. The brain's only signal
    // today is a `[disabled_outcome_received]` warn in /var/log; this
    // hand-off feeds the dashboard banner + off-dashboard webhook so
    // a leaking timeframe is visible within minutes, not days.
    if (body.ok === false && body.reason === "disabled_role_rejected") {
      bumpCause("record_outcome_disabled_role_rejected");
      try {
        recordDisabledOutcomeRejection({
          tickId: wireTickId,
          sliceId: payload.sliceId ?? null,
          timeframe: payload.timeframe,
        });
      } catch (err) {
        // Notifier failures must never crash the trade-close path.
        logFallbackOnce("disabled_outcome_notifier_throw", {
          err: err instanceof Error ? err.message : String(err),
        });
      }
    }
    return body.ok === true;
  } catch (err) {
    bumpCause("record_outcome_failed");
    logFallbackOnce("record_outcome_failed", { err: String(err) });
    return false;
  } finally {
    clearTimeout(t);
  }
}
