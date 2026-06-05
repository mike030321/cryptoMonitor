/**
 * Task #406 — promotion gate for the manual brain enable endpoint.
 *
 * The audit in docs/remediation/2026-04-24-full-system-remediation.md is
 * explicit: an operator MUST NOT flip `quant_brain_enabled` to true while
 * the verification gate is rejecting every slice. The route
 * `POST /api/crypto/brain/state` therefore consults this helper before
 * calling `setBrainState(true, "manual")`. Disabling (enabled:false) is
 * always allowed — the gate only guards the enable direction.
 *
 * "Promoted" here means the most recent verification-history entry from the
 * ML engine reports at least one slice with promoted:true (i.e.
 * counts.slices_promoted > 0 AND coins_with_promotion is non-empty AND the
 * batch verification_status is "ok"). All three are required so a stale
 * "passed:true" with a structurally-failed batch cannot waive the gate.
 *
 * Task #678 — bounded retry/backoff on the verification-history fetch.
 * A single transient ML-engine outage used to refuse the manual enable
 * with `history_unreachable`. The fail-closed semantic is preserved
 * (ALL retries must fail before we still return `history_unreachable`,
 * with NO caching / no last-known-good fallback — that would weaken the
 * gate and was explicitly rejected during triage of the Codex audit
 * Finding #1; see reports/codex-audit-triage-20260501T083546Z.md §2.1).
 * The change is purely "be more patient about a transient outage", not
 * "ever serve a stale verification".
 */
import { logger } from "./logger";
import {
  getRoleEntryForTimeframe,
  getTradeRoleTimeframes,
  type TimeframeRoleEntry,
} from "./timeframe-roles";

export interface PromotionGateVerdict {
  /** Whether the most-recent verification record contains a promoted slice. */
  ok: boolean;
  /** Short machine-readable reason when ok=false. */
  reason?:
    | "no_history"
    | "history_unreachable"
    | "verification_status_not_ok"
    | "no_promoted_slices"
    | "no_coins_with_promotion"
    // Task #550 — role gate refusal: requested enable scope contained
    // one or more timeframes whose role is not `trade`. The reason
    // text in the route response includes the offending timeframe(s).
    | "gate_pre_check_failed_by_role";
  /** Best-effort detail surfaced to the operator alongside the 409. */
  evidence?: {
    verification_status?: string | null;
    passed?: boolean | null;
    slices_promoted?: number | null;
    coins_with_promotion?: string[];
    recorded_at?: number | null;
    /** Task #550 — TFs with role==='trade' at the time of the check. */
    permitted_timeframes?: string[];
    /** Task #550 — TFs requested but refused because role!=='trade'. */
    refused_timeframes?: Array<{
      timeframe: string;
      role: TimeframeRoleEntry["role"];
      context_subkind: TimeframeRoleEntry["context_subkind"];
      disabled_reason: TimeframeRoleEntry["disabled_reason"];
      reason: string;
    }>;
  };
  /** Task #550 — TFs whose role currently allows trade execution. */
  permitted_timeframes?: string[];
}

/**
 * Task #678 — retry/backoff machinery for the verification-history fetch.
 *
 * - `GATE_RETRY_MAX_ATTEMPTS` total attempts (1 initial + N-1 retries).
 * - `GATE_RETRY_TOTAL_BUDGET_MS` is the HARD upper bound on wall-clock
 *   time the helper is allowed to spend across ALL attempts + backoffs.
 * - `GATE_RETRY_BACKOFF_BASE_MS` is the first backoff delay; subsequent
 *   delays double (exponential, no jitter — schedule must be deterministic
 *   per the task requirements).
 * - `GATE_PER_ATTEMPT_TIMEOUT_MS` matches today's 8 s AbortController
 *   timeout per attempt; effective per-attempt timeout is further capped
 *   by the remaining budget so we never overshoot the wall-clock bound.
 *
 * Worst-case schedule with the production constants below:
 *   attempt 1 (≤ 8 s) + backoff 500 ms + attempt 2 (≤ 8 s) +
 *   backoff 1000 ms + attempt 3 (capped to remaining budget = 6.5 s)
 *   = 24 s, exactly the budget.
 */
export const GATE_RETRY_MAX_ATTEMPTS = 3;
export const GATE_RETRY_TOTAL_BUDGET_MS = 24_000;
export const GATE_RETRY_BACKOFF_BASE_MS = 500;
export const GATE_PER_ATTEMPT_TIMEOUT_MS = 8_000;

/**
 * Task #686 — in-memory retry-event ring buffer.
 *
 * The bounded retry loop logs each failed attempt at warn level
 * (`attempt`, `retry_failure_reason`, `elapsed_ms`), but until now
 * those signals only existed in the workflow log stream. The audit
 * for #686 wants an at-a-glance "the gate had to wait for the
 * ml-engine N times in the last hour, the most recent reason was X"
 * surface in the dashboard so operators can tell a single-shot
 * `history_unreachable` apart from "all 3 attempts failed".
 *
 * We mirror the same warn-level events into a small, bounded ring
 * buffer (500 entries; ~1 day of every-minute hammering is well under
 * that). The buffer is process-local on purpose — it tracks "what
 * has THIS api-server seen" and intentionally clears on restart, so
 * an operator restart-to-fix-it action also resets the chip. No DB
 * write on the hot path, no change to the gate's external contract.
 */
export interface PromotionGateRetryEvent {
  /** Wall-clock time of the warn event in ms-since-epoch. */
  at: number;
  /** Which retry attempt failed (1-indexed). */
  attempt: number;
  /** Same `retry_failure_reason` string emitted to the warn log. */
  retryFailureReason: string;
  /** Milliseconds elapsed inside the retry loop when this attempt failed. */
  elapsedMs: number;
}

const RETRY_EVENT_RING_SIZE = 500;
const retryEvents: PromotionGateRetryEvent[] = [];

function recordRetryEvent(ev: PromotionGateRetryEvent): void {
  retryEvents.push(ev);
  if (retryEvents.length > RETRY_EVENT_RING_SIZE) {
    retryEvents.splice(0, retryEvents.length - RETRY_EVENT_RING_SIZE);
  }
}

export interface PromotionGateRetryStats {
  /** Number of retry-failure warn events recorded inside the window. */
  count: number;
  /** Width of the lookback window used, in milliseconds. */
  windowMs: number;
  /** ISO timestamp of the most recent retry-failure event in the window, if any. */
  mostRecentAt: string | null;
  /** `retry_failure_reason` of the most recent in-window event, if any. */
  mostRecentReason: string | null;
  /** Attempt index of the most recent in-window event, if any. */
  mostRecentAttempt: number | null;
}

/**
 * Task #686 — operator-visible roll-up of recent retry warn events.
 * Default window is 1 hour, matching the dashboard chip copy
 * ("promotion-gate retries in the last hour"). Pure read; never
 * mutates the buffer.
 */
export function getPromotionGateRetryStats(
  windowMs = 60 * 60 * 1000,
): PromotionGateRetryStats {
  const cutoff = Date.now() - windowMs;
  let count = 0;
  let mostRecent: PromotionGateRetryEvent | null = null;
  for (const ev of retryEvents) {
    if (ev.at < cutoff) continue;
    count += 1;
    if (!mostRecent || ev.at > mostRecent.at) {
      mostRecent = ev;
    }
  }
  return {
    count,
    windowMs,
    mostRecentAt: mostRecent ? new Date(mostRecent.at).toISOString() : null,
    mostRecentReason: mostRecent ? mostRecent.retryFailureReason : null,
    mostRecentAttempt: mostRecent ? mostRecent.attempt : null,
  };
}

/** Test-only — empty the in-memory ring buffer between cases. */
export function _resetPromotionGateRetryEventsForTest(): void {
  retryEvents.length = 0;
}

export interface VerificationGateOptions {
  /** Override for tests; defaults to global fetch. */
  fetcher?: typeof fetch;
  /** Override for tests; defaults to ML_ENGINE_URL || http://localhost:8000. */
  mlEngineBaseUrl?: string;
  /** Per-attempt abort timeout in ms; default 8000 (matches /verification-history route). */
  timeoutMs?: number;
  /**
   * Task #550 — when present, the gate also refuses if any of these
   * timeframes has a role other than `trade`. The route handler
   * currently has no per-request scope (it's a global enable), so this
   * is opt-in for callers (and unit tests) that DO scope the request.
   */
  requestedTimeframes?: string[];
  /**
   * Task #678 — overrides for the retry/backoff machinery. Production
   * code should NEVER pass this; it exists so unit tests can keep the
   * suite fast without re-implementing the loop. Anything not set
   * inherits the module-level constants above.
   */
  retry?: {
    maxAttempts?: number;
    totalBudgetMs?: number;
    backoffBaseMs?: number;
  };
}

const HISTORY_PATH = "/ml/admin/verification-history?limit=1";

/**
 * Outcome classification for a single fetch attempt. Drives the retry
 * decision tree without leaking fetch internals up to the gate body.
 */
type AttemptOutcome =
  | { kind: "ok"; response: Response }
  | { kind: "non_retryable_status"; status: number }
  | { kind: "retryable_status"; status: number }
  | { kind: "timeout" }
  | { kind: "network_error"; error: unknown };

async function attemptFetch(
  fetcher: typeof fetch,
  url: string,
  attemptTimeoutMs: number,
): Promise<AttemptOutcome> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), attemptTimeoutMs);
  let timedOut = false;
  // We can't directly tell an AbortError caused by our own timer apart
  // from any other AbortError, so flag it ourselves.
  const onAbort = () => {
    timedOut = true;
  };
  ctrl.signal.addEventListener("abort", onAbort, { once: true });
  try {
    const resp = await fetcher(url, { signal: ctrl.signal });
    if (resp.ok) {
      return { kind: "ok", response: resp };
    }
    if (resp.status >= 400 && resp.status < 500) {
      // 4xx is a configuration error, not a transient outage — do NOT retry.
      return { kind: "non_retryable_status", status: resp.status };
    }
    return { kind: "retryable_status", status: resp.status };
  } catch (err) {
    if (timedOut) {
      return { kind: "timeout" };
    }
    return { kind: "network_error", error: err };
  } finally {
    clearTimeout(timer);
  }
}

interface FetchSuccess {
  ok: true;
  response: Response;
}

interface FetchFailure {
  ok: false;
  /** Always returned to the gate so verdict.reason stays `history_unreachable`. */
  externalReason: "history_unreachable";
}

/**
 * Bounded retry loop. The helper "owns" the retry decision; it never
 * returns a stale or cached record, and its only failure mode is the
 * fail-closed `history_unreachable` (preserving the byte-identical
 * external contract documented in the Task #406 remediation note).
 */
async function fetchVerificationHistoryWithRetry(opts: {
  fetcher: typeof fetch;
  url: string;
  perAttemptTimeoutMs: number;
  maxAttempts: number;
  totalBudgetMs: number;
  backoffBaseMs: number;
}): Promise<FetchSuccess | FetchFailure> {
  const start = Date.now();
  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    const elapsedBefore = Date.now() - start;
    const remainingBefore = opts.totalBudgetMs - elapsedBefore;
    if (remainingBefore <= 0) {
      logger.warn(
        { attempt, elapsed_ms: elapsedBefore, budget_ms: opts.totalBudgetMs },
        "brain-promotion-gate: retry budget exhausted before attempt",
      );
      // Task #686 — surface this exhaustion to the dashboard chip too.
      recordRetryEvent({
        at: Date.now(),
        attempt,
        retryFailureReason: "budget_exhausted_before_attempt",
        elapsedMs: elapsedBefore,
      });
      return { ok: false, externalReason: "history_unreachable" };
    }
    const attemptTimeout = Math.min(opts.perAttemptTimeoutMs, remainingBefore);
    const outcome = await attemptFetch(opts.fetcher, opts.url, attemptTimeout);
    const elapsedAfter = Date.now() - start;

    if (outcome.kind === "ok") {
      // Successful attempt — gate's external contract is unchanged, so we
      // do NOT signal "we retried" up to the route. Operator only sees
      // success; the per-attempt warn-logs above remain for ops triage.
      return { ok: true, response: outcome.response };
    }

    if (outcome.kind === "non_retryable_status") {
      // 4xx is a configuration error (e.g. 401, 404). Do NOT retry — that
      // would just hammer the ml-engine with a request it will keep
      // refusing. Fail closed immediately.
      logger.warn(
        {
          attempt,
          status: outcome.status,
          elapsed_ms: elapsedAfter,
          retry_failure_reason: `non_retryable_status_${outcome.status}`,
        },
        `brain-promotion-gate: history fetch attempt ${attempt} returned non-retryable status ${outcome.status}`,
      );
      recordRetryEvent({
        at: Date.now(),
        attempt,
        retryFailureReason: `non_retryable_status_${outcome.status}`,
        elapsedMs: elapsedAfter,
      });
      return { ok: false, externalReason: "history_unreachable" };
    }

    const retryFailureReason =
      outcome.kind === "timeout"
        ? "timeout"
        : outcome.kind === "retryable_status"
          ? `non_2xx_status_${outcome.status}`
          : "network_error";
    logger.warn(
      {
        attempt,
        retry_failure_reason: retryFailureReason,
        elapsed_ms: elapsedAfter,
        ...(outcome.kind === "network_error" ? { err: outcome.error } : {}),
      },
      `brain-promotion-gate: history fetch attempt ${attempt} failed (${retryFailureReason})`,
    );
    recordRetryEvent({
      at: Date.now(),
      attempt,
      retryFailureReason,
      elapsedMs: elapsedAfter,
    });

    if (attempt === opts.maxAttempts) {
      // All attempts exhausted — fail closed.
      return { ok: false, externalReason: "history_unreachable" };
    }

    const backoffMs = opts.backoffBaseMs * Math.pow(2, attempt - 1);
    if (elapsedAfter + backoffMs >= opts.totalBudgetMs) {
      logger.warn(
        {
          attempt,
          elapsed_ms: elapsedAfter,
          next_backoff_ms: backoffMs,
          budget_ms: opts.totalBudgetMs,
          retry_failure_reason: "budget_exhausted",
        },
        "brain-promotion-gate: retry budget exhausted before next backoff",
      );
      // Task #686 — intentionally NOT recordRetryEvent'd here. The
      // failed attempt that brought us to this point was already
      // mirrored into the ring buffer (~25 lines above). Adding a
      // second event would double-count the same retry from the
      // operator's chip-counter perspective. The warn log still fires
      // for ops triage; the chip stays a strict "failed-attempt count".
      return { ok: false, externalReason: "history_unreachable" };
    }
    await new Promise<void>((resolve) => setTimeout(resolve, backoffMs));
  }
  // Defensive fall-through. Loop body always returns; this exists so the
  // function is total in TypeScript's eyes even if maxAttempts is 0.
  return { ok: false, externalReason: "history_unreachable" };
}

export async function hasPromotedSlice(
  options: VerificationGateOptions = {},
): Promise<PromotionGateVerdict> {
  // ── Task #550 — role gate (the second of two required gates) ──
  // Compute permitted_timeframes (TFs whose role==='trade') from the
  // loaded `shared/timeframe-roles.json`. If the caller supplied a
  // requested scope, refuse the enable when any requested TF's role
  // is not `trade`. This runs BEFORE the verification-history fetch
  // so a fail-closed roles file (every TF disabled (by_safety))
  // correctly refuses without needing the ml-engine to be reachable.
  const permittedTimeframes = getTradeRoleTimeframes();
  if (options.requestedTimeframes && options.requestedTimeframes.length > 0) {
    const refused: NonNullable<PromotionGateVerdict["evidence"]>["refused_timeframes"] = [];
    for (const tf of options.requestedTimeframes) {
      const entry = getRoleEntryForTimeframe(tf);
      if (entry.role !== "trade") {
        refused!.push({
          timeframe: tf,
          role: entry.role,
          context_subkind: entry.context_subkind,
          disabled_reason: entry.disabled_reason,
          reason: entry.reason,
        });
      }
    }
    if (refused && refused.length > 0) {
      const tfList = refused.map((r) => r.timeframe).join(", ");
      logger.warn(
        { refused, permittedTimeframes },
        `brain-promotion-gate: refused by role for timeframe(s) ${tfList}`,
      );
      return {
        ok: false,
        reason: "gate_pre_check_failed_by_role",
        evidence: {
          permitted_timeframes: permittedTimeframes,
          refused_timeframes: refused,
        },
        permitted_timeframes: permittedTimeframes,
      };
    }
  }

  const fetcher = options.fetcher ?? fetch;
  const baseUrl = (
    options.mlEngineBaseUrl ?? process.env.ML_ENGINE_URL ?? "http://localhost:8000"
  ).replace(/\/$/, "");
  const url = `${baseUrl}${HISTORY_PATH}`;
  const perAttemptTimeoutMs = options.timeoutMs ?? GATE_PER_ATTEMPT_TIMEOUT_MS;
  const maxAttempts = options.retry?.maxAttempts ?? GATE_RETRY_MAX_ATTEMPTS;
  const totalBudgetMs = options.retry?.totalBudgetMs ?? GATE_RETRY_TOTAL_BUDGET_MS;
  const backoffBaseMs = options.retry?.backoffBaseMs ?? GATE_RETRY_BACKOFF_BASE_MS;

  const fetchResult = await fetchVerificationHistoryWithRetry({
    fetcher,
    url,
    perAttemptTimeoutMs,
    maxAttempts,
    totalBudgetMs,
    backoffBaseMs,
  });
  if (!fetchResult.ok) {
    // External contract unchanged: a transient outage that survives the
    // retry budget still fails closed exactly as it did pre-#678.
    return {
      ok: false,
      reason: "history_unreachable",
      permitted_timeframes: permittedTimeframes,
    };
  }
  const resp = fetchResult.response;
  try {
    const body = (await resp.json()) as {
      rows?: Array<{
        verification_status?: string;
        passed?: boolean;
        coins_with_promotion?: string[];
        counts?: { slices_promoted?: number };
        recorded_at?: number;
      }>;
    };
    const row = body?.rows?.[0];
    if (!row) {
      return { ok: false, reason: "no_history", permitted_timeframes: permittedTimeframes };
    }
    const evidence = {
      verification_status: row.verification_status ?? null,
      passed: typeof row.passed === "boolean" ? row.passed : null,
      slices_promoted:
        typeof row.counts?.slices_promoted === "number" ? row.counts.slices_promoted : null,
      coins_with_promotion: Array.isArray(row.coins_with_promotion)
        ? row.coins_with_promotion
        : [],
      recorded_at: typeof row.recorded_at === "number" ? row.recorded_at : null,
      permitted_timeframes: permittedTimeframes,
    };
    if (row.verification_status !== "ok") {
      return {
        ok: false,
        reason: "verification_status_not_ok",
        evidence,
        permitted_timeframes: permittedTimeframes,
      };
    }
    if (!evidence.slices_promoted || evidence.slices_promoted <= 0) {
      return {
        ok: false,
        reason: "no_promoted_slices",
        evidence,
        permitted_timeframes: permittedTimeframes,
      };
    }
    if (!evidence.coins_with_promotion || evidence.coins_with_promotion.length === 0) {
      return {
        ok: false,
        reason: "no_coins_with_promotion",
        evidence,
        permitted_timeframes: permittedTimeframes,
      };
    }
    return { ok: true, evidence, permitted_timeframes: permittedTimeframes };
  } catch (err) {
    // JSON-parse failure on a 2xx response is treated as an unreachable
    // history (the body is structurally garbage; we have no verification
    // to reason about). Same fail-closed result, no caching, no fallback.
    logger.warn({ err }, "brain-promotion-gate: history body parse failed");
    return {
      ok: false,
      reason: "history_unreachable",
      permitted_timeframes: permittedTimeframes,
    };
  }
}
