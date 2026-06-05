import { test } from "node:test";
import assert from "node:assert/strict";

// Task #406 — promotion-gate unit tests for the manual brain enable.
//
// These exercise hasPromotedSlice() in isolation by injecting a fake
// fetcher that returns canned /ml/admin/verification-history payloads.
// The route handler in routes/crypto/index.ts calls this same helper
// before flipping `quant_brain_enabled` to true; refusing the enable
// when no slice is promoted is the safety property the audit
// (docs/remediation/2026-04-24-full-system-remediation.md) requires.

import {
  GATE_PER_ATTEMPT_TIMEOUT_MS,
  GATE_RETRY_BACKOFF_BASE_MS,
  GATE_RETRY_MAX_ATTEMPTS,
  GATE_RETRY_TOTAL_BUDGET_MS,
  _resetPromotionGateRetryEventsForTest,
  getPromotionGateRetryStats,
  hasPromotedSlice,
} from "../src/lib/brain-promotion-gate";

function fakeFetcher(payload: unknown, init: { status?: number } = {}) {
  return (async (_url: string | URL | Request, _opts?: RequestInit) => {
    return new Response(JSON.stringify(payload), {
      status: init.status ?? 200,
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch;
}

/**
 * Build a fetcher that returns canned outcomes per call. Each `step` is
 * either { status, body? } (resolves a Response with that status), or
 * { error: Error } (throws as if the network failed). Tracks call count
 * and per-call elapsed-since-construction for timing assertions used by
 * the Task #678 retry/backoff tests.
 */
type CannedStep =
  | { status: number; body?: unknown }
  | { error: Error }
  // "hang" simulates an attempt that never returns until it is aborted
  // by the gate's per-attempt AbortController. Used for the worst-case
  // "all timeouts" wall-clock-bound test.
  | { hang: true };

function scriptedFetcher(steps: CannedStep[]) {
  const start = Date.now();
  const calls: Array<{ at: number; index: number }> = [];
  const fn = (async (_url: string | URL | Request, init?: RequestInit) => {
    const index = calls.length;
    calls.push({ at: Date.now() - start, index });
    const step = steps[index];
    if (!step) {
      throw new Error(`scriptedFetcher: no canned step for call #${index + 1}`);
    }
    if ("error" in step) {
      throw step.error;
    }
    if ("hang" in step) {
      // Resolve only when our caller's AbortController fires. fetch() is
      // expected to reject with an AbortError once the signal is aborted.
      return await new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal as AbortSignal | undefined;
        if (!signal) {
          reject(new Error("scriptedFetcher: hang step requires an AbortSignal"));
          return;
        }
        if (signal.aborted) {
          const e = new Error("aborted");
          (e as Error & { name: string }).name = "AbortError";
          reject(e);
          return;
        }
        signal.addEventListener(
          "abort",
          () => {
            const e = new Error("aborted");
            (e as Error & { name: string }).name = "AbortError";
            reject(e);
          },
          { once: true },
        );
      });
    }
    return new Response(JSON.stringify(step.body ?? {}), {
      status: step.status,
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch;
  return { fn, calls, startedAt: start };
}

test("hasPromotedSlice: refuses when verification-history has zero rows", async () => {
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher({ rows: [] }),
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "no_history");
});

test("hasPromotedSlice: refuses when latest row has slices_promoted = 0", async () => {
  // This is the exact shape we observed under the current data + gate
  // configuration: a clean retrain where every slice tripped
  // directional_call_regression and nothing got promoted.
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher({
      rows: [
        {
          recorded_at: 1777030969,
          verification_status: "ok",
          passed: false,
          coins_with_promotion: [],
          counts: { slices_promoted: 0 },
        },
      ],
    }),
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "no_promoted_slices");
  assert.equal(verdict.evidence?.slices_promoted, 0);
  assert.equal(verdict.evidence?.passed, false);
});

test("hasPromotedSlice: refuses when verification_status is not 'ok'", async () => {
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher({
      rows: [
        {
          recorded_at: 1,
          verification_status: "error",
          passed: false,
          coins_with_promotion: [],
          counts: { slices_promoted: 5 },
        },
      ],
    }),
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "verification_status_not_ok");
});

test("hasPromotedSlice: refuses when coins_with_promotion is empty even if count > 0", async () => {
  // Defence in depth: counts.slices_promoted could disagree with the
  // per-coin attribution if upstream ever changes shape. Demand both.
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher({
      rows: [
        {
          recorded_at: 1,
          verification_status: "ok",
          passed: false,
          coins_with_promotion: [],
          counts: { slices_promoted: 3 },
        },
      ],
    }),
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "no_coins_with_promotion");
});

test("hasPromotedSlice: allows enable when latest row has a promoted slice and a coin", async () => {
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher({
      rows: [
        {
          recorded_at: 1777030969,
          verification_status: "ok",
          passed: true,
          coins_with_promotion: ["bitcoin"],
          counts: { slices_promoted: 1 },
        },
      ],
    }),
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, true);
  assert.equal(verdict.reason, undefined);
  assert.deepEqual(verdict.evidence?.coins_with_promotion, ["bitcoin"]);
});

test("hasPromotedSlice: refuses when ml-engine returns non-2xx", async () => {
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher({}, { status: 502 }),
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "history_unreachable");
});

test("hasPromotedSlice: refuses when fetcher throws (network unreachable)", async () => {
  const verdict = await hasPromotedSlice({
    fetcher: (async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch,
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "history_unreachable");
});

test("POST /crypto/brain/state handler factory consults hasPromotedSlice and emits 409 on the enable=true branch only", async () => {
  // Code-shape regression: the handler factory in
  // src/lib/brain-state-route.ts must call hasPromotedSlice() before
  // invoking setBrainState(true, "manual"), short-circuit with HTTP 409
  // when the gate refuses, and gate ONLY on the enabled:true branch.
  // The end-to-end behaviour is also exercised by
  // brain-state-route-integration.test.ts; this test locks the source
  // shape so the production wiring cannot regress without breaking a
  // test even if a dev forgets to wire the integration test.
  const fs = await import("node:fs/promises");
  const handler = await fs.readFile(
    new URL("../src/lib/brain-state-route.ts", import.meta.url),
    "utf8",
  );
  assert.match(handler, /hasPromotedSlice\s*\(/, "handler must call hasPromotedSlice() before enabling");
  assert.match(handler, /\.status\(409\)/, "handler must emit HTTP 409 when the gate refuses");
  assert.match(handler, /promotion_gate_blocked/, "handler must surface a promotion_gate_blocked error code");
  // The gate must be checked specifically on the enable-true branch
  // (kill-switching down to false should never be gated).
  assert.match(
    handler,
    /enabled\s*===\s*true[\s\S]*hasPromotedSlice/,
    "handler must consult the gate only on the enabled:true branch",
  );

  // The route in routes/crypto/index.ts must wire to this factory and
  // keep the admin-key gate at the route layer.
  const route = await fs.readFile(
    new URL("../src/routes/crypto/index.ts", import.meta.url),
    "utf8",
  );
  const start = route.indexOf('router.post("/crypto/brain/state"');
  assert.ok(start > 0, "expected POST /crypto/brain/state route");
  const after = route.slice(start);
  const end = after.indexOf("\n});\n");
  assert.ok(end > 0, "could not find end of /crypto/brain/state route");
  const routeHandler = after.slice(0, end);
  assert.match(routeHandler, /requireAdminApiKey\s*\(/, "route must enforce admin-key auth");
  assert.match(routeHandler, /brainStatePostHandler\s*\(/, "route must delegate to the factory-built handler");
});

// ────────────────────────────────────────────────────────────────────
// Task #678 — bounded retry/backoff on the verification-history fetch
// ────────────────────────────────────────────────────────────────────
//
// These tests pin the four properties enumerated in
// `.local/tasks/task-678.md` "Done looks like":
//   1. `history_unreachable` is still returned after retries are exhausted
//      (fail-closed contract preserved).
//   2. Total wall-clock time is bounded by the retry budget under the
//      worst-case "all timeouts" scenario.
//   3. Retry count and per-attempt backoff schedule are deterministic.
//   4. A successful retry on attempt 2 or 3 returns the verification
//      record without changing the gate's external contract.
//
// We also pin the audit-required "non-retryable 4xx is not retried"
// behaviour (Task #678 hard-guardrail #4 — a 401/404 is a configuration
// error, NOT a transient outage).

test("Task #678: production retry constants stay within the documented budget", () => {
  // Sanity-check the constants the gate ships with: 1 initial + 2 retries,
  // 24 s wall-clock budget, base backoff 500 ms doubling. The worst case
  // (3 timed-out attempts at 8 s + 500 ms + 1 s backoff) hits the budget
  // boundary exactly; if any constant drifts off-spec, this test loudly
  // tells the next agent before they wonder why the gate hangs.
  assert.equal(GATE_RETRY_MAX_ATTEMPTS, 3);
  assert.equal(GATE_RETRY_TOTAL_BUDGET_MS, 24_000);
  assert.equal(GATE_RETRY_BACKOFF_BASE_MS, 500);
  assert.equal(GATE_PER_ATTEMPT_TIMEOUT_MS, 8_000);
  // 1 backoff after attempt 1 + 1 backoff after attempt 2.
  const backoffSum = GATE_RETRY_BACKOFF_BASE_MS + GATE_RETRY_BACKOFF_BASE_MS * 2;
  // Per-attempt timeout × max attempts must fit in the budget once the
  // backoffs are subtracted; otherwise attempt 3 would overshoot.
  assert.ok(
    GATE_PER_ATTEMPT_TIMEOUT_MS * GATE_RETRY_MAX_ATTEMPTS <=
      GATE_RETRY_TOTAL_BUDGET_MS,
    "per-attempt timeout × max attempts must fit in the wall-clock budget",
  );
  assert.ok(
    backoffSum < GATE_RETRY_TOTAL_BUDGET_MS,
    "backoff sum must leave room for at least the first attempt",
  );
});

test("Task #678: history_unreachable is still returned when ALL retries exhaust (fail-closed preserved)", async () => {
  // Mock returns 503 three times. Gate must observe exactly 3 attempts
  // (1 initial + 2 retries) and STILL return the same history_unreachable
  // it would have returned pre-#678 — no caching, no last-known-good.
  const scripted = scriptedFetcher([
    { status: 503 },
    { status: 503 },
    { status: 503 },
  ]);
  const t0 = Date.now();
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
  });
  const elapsed = Date.now() - t0;
  assert.equal(verdict.ok, false);
  assert.equal(
    verdict.reason,
    "history_unreachable",
    "external contract must stay history_unreachable even after retries",
  );
  assert.equal(scripted.calls.length, 3, "must make exactly 3 attempts");
  // Backoff schedule is deterministic: 500 ms + 1000 ms = 1500 ms total
  // between attempts. Add a generous epsilon for scheduler jitter.
  assert.ok(
    elapsed < 4_000,
    `elapsed ${elapsed}ms must be much less than the 24s budget when each fetch returns instantly`,
  );
});

test("Task #678: per-attempt backoff schedule is deterministic (500 ms then 1000 ms, no jitter)", async () => {
  // Mock fails with a network error twice, then succeeds. We measure the
  // gap between recorded fetch start times to verify the gate slept the
  // EXACT documented backoff between attempts (no random jitter is allowed).
  const scripted = scriptedFetcher([
    { error: new Error("ECONNRESET") },
    { error: new Error("ECONNRESET") },
    {
      status: 200,
      body: {
        rows: [
          {
            recorded_at: 42,
            verification_status: "ok",
            passed: true,
            coins_with_promotion: ["bitcoin"],
            counts: { slices_promoted: 1 },
          },
        ],
      },
    },
  ]);
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, true, "third attempt succeeds");
  assert.equal(scripted.calls.length, 3);
  const gap1 = scripted.calls[1].at - scripted.calls[0].at;
  const gap2 = scripted.calls[2].at - scripted.calls[1].at;
  // Allow ±150 ms scheduler jitter on each side of the deterministic delay.
  assert.ok(
    gap1 >= GATE_RETRY_BACKOFF_BASE_MS - 50 &&
      gap1 <= GATE_RETRY_BACKOFF_BASE_MS + 200,
    `gap after attempt 1 was ${gap1}ms, expected ~${GATE_RETRY_BACKOFF_BASE_MS}ms`,
  );
  assert.ok(
    gap2 >= GATE_RETRY_BACKOFF_BASE_MS * 2 - 50 &&
      gap2 <= GATE_RETRY_BACKOFF_BASE_MS * 2 + 200,
    `gap after attempt 2 was ${gap2}ms, expected ~${GATE_RETRY_BACKOFF_BASE_MS * 2}ms`,
  );
});

test("Task #678: success on the 2nd attempt returns the verification record (external contract unchanged)", async () => {
  const scripted = scriptedFetcher([
    { error: new Error("ECONNREFUSED") },
    {
      status: 200,
      body: {
        rows: [
          {
            recorded_at: 1777031000,
            verification_status: "ok",
            passed: true,
            coins_with_promotion: ["bitcoin"],
            counts: { slices_promoted: 1 },
          },
        ],
      },
    },
  ]);
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, true);
  assert.equal(verdict.reason, undefined);
  assert.deepEqual(verdict.evidence?.coins_with_promotion, ["bitcoin"]);
  assert.equal(verdict.evidence?.slices_promoted, 1);
  // The verdict shape must NOT leak any "we retried" signal — the route
  // handler must see the same payload it would for a single-shot success.
  // The shape is locked by PromotionGateVerdict (no retry/attempt fields).
  const verdictKeys = Object.keys(verdict).sort();
  for (const k of verdictKeys) {
    assert.ok(
      ["ok", "reason", "evidence", "permitted_timeframes"].includes(k),
      `verdict must not expose new key '${k}' (would break the external contract)`,
    );
  }
  assert.equal(scripted.calls.length, 2);
});

test("Task #678: a non-retryable 4xx is NOT retried (configuration error, not transient outage)", async () => {
  // 401 is a misconfigured admin token — retrying just hammers the
  // ml-engine with a request it will keep refusing. Gate must fail
  // closed on the FIRST attempt with no retries.
  const scripted = scriptedFetcher([{ status: 401 }]);
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "history_unreachable");
  assert.equal(
    scripted.calls.length,
    1,
    "non-retryable 4xx must not trigger any retries",
  );
});

test("Task #678: total wall-clock is bounded by the retry budget under the worst-case 'all timeouts' scenario", async () => {
  // Use a small, scaled-down budget so the test stays fast: 600 ms total
  // budget, 200 ms per-attempt timeout, 50 ms backoff base. Under three
  // hangs the gate is forced through every retry path AND the final
  // budget-exhaustion check; total wall time must NEVER exceed
  // budget + small epsilon. Production constants are exercised at the
  // arithmetic level by the "production retry constants" test above.
  const scripted = scriptedFetcher([
    { hang: true },
    { hang: true },
    { hang: true },
  ]);
  const t0 = Date.now();
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
    timeoutMs: 200,
    retry: {
      maxAttempts: 3,
      totalBudgetMs: 600,
      backoffBaseMs: 50,
    },
  });
  const elapsed = Date.now() - t0;
  assert.equal(verdict.ok, false);
  assert.equal(
    verdict.reason,
    "history_unreachable",
    "fail-closed preserved under exhaustion-via-timeout",
  );
  // Allow ~200 ms of scheduler/timer slack on top of the budget.
  assert.ok(
    elapsed <= 600 + 250,
    `elapsed ${elapsed}ms must not exceed totalBudgetMs (600) + epsilon`,
  );
  // Should have made at least one attempt; the exact count depends on
  // whether attempt 3 fits inside the remaining budget. Either way,
  // calls.length must be ≤ maxAttempts.
  assert.ok(
    scripted.calls.length >= 1 && scripted.calls.length <= 3,
    `attempts (${scripted.calls.length}) must be in [1, maxAttempts]`,
  );
});

// ────────────────────────────────────────────────────────────────────
// Task #686 — operator-visible roll-up of promotion-gate retry warns
// ────────────────────────────────────────────────────────────────────
//
// The retry loop already emits warn-level logs with `attempt`,
// `retry_failure_reason`, `elapsed_ms`. Task #686 mirrors those into
// a bounded in-memory ring buffer so the dashboard banner can show
// "promotion-gate retries: N in last 60m · most recent: <reason>".
// The chip's whole purpose is to let an operator distinguish a
// single-shot `history_unreachable` from "all 3 attempts failed"
// without tailing logs.

test("Task #686: getPromotionGateRetryStats reports zero when no retries have happened", async () => {
  _resetPromotionGateRetryEventsForTest();
  const stats = getPromotionGateRetryStats();
  assert.equal(stats.count, 0);
  assert.equal(stats.mostRecentReason, null);
  assert.equal(stats.mostRecentAt, null);
  assert.equal(stats.mostRecentAttempt, null);
});

test("Task #686: a single-shot transient outage records exactly one event with the right reason", async () => {
  _resetPromotionGateRetryEventsForTest();
  const scripted = scriptedFetcher([
    { error: new Error("ECONNRESET") },
    {
      status: 200,
      body: {
        rows: [
          {
            recorded_at: 1,
            verification_status: "ok",
            passed: true,
            coins_with_promotion: ["bitcoin"],
            counts: { slices_promoted: 1 },
          },
        ],
      },
    },
  ]);
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
    retry: { backoffBaseMs: 5 },
  });
  assert.equal(verdict.ok, true, "second attempt succeeds");
  const stats = getPromotionGateRetryStats();
  assert.equal(stats.count, 1, "exactly one failed attempt was recorded");
  assert.equal(stats.mostRecentReason, "network_error");
  assert.equal(stats.mostRecentAttempt, 1);
  assert.ok(stats.mostRecentAt, "mostRecentAt is populated");
});

test("Task #686: an all-attempts-failed enable refusal records every retry warn (count === maxAttempts)", async () => {
  _resetPromotionGateRetryEventsForTest();
  const scripted = scriptedFetcher([
    { status: 503 },
    { status: 503 },
    { status: 503 },
  ]);
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
    retry: { backoffBaseMs: 5 },
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "history_unreachable");
  const stats = getPromotionGateRetryStats();
  assert.equal(stats.count, 3, "every failed attempt was recorded");
  // Most recent reason should be from the final attempt.
  assert.equal(stats.mostRecentReason, "non_2xx_status_503");
  assert.equal(stats.mostRecentAttempt, 3);
});

test("Task #686: a non-retryable 4xx still records exactly one event so the chip surfaces the misconfiguration", async () => {
  _resetPromotionGateRetryEventsForTest();
  const scripted = scriptedFetcher([{ status: 401 }]);
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "history_unreachable");
  const stats = getPromotionGateRetryStats();
  assert.equal(stats.count, 1);
  assert.equal(stats.mostRecentReason, "non_retryable_status_401");
  assert.equal(stats.mostRecentAttempt, 1);
});

test("Task #686: events older than the requested window are excluded (in-window slicing is correct)", async () => {
  _resetPromotionGateRetryEventsForTest();
  // Generate three failed-attempt events now.
  const scripted = scriptedFetcher([{ status: 502 }, { status: 502 }, { status: 502 }]);
  await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
    retry: { backoffBaseMs: 5 },
  });
  // Wait long enough that a small lookback window strictly precedes
  // the recorded events (50 ms slack on top of the 5 ms backoff sum).
  await new Promise((resolve) => setTimeout(resolve, 75));
  const stats = getPromotionGateRetryStats(10);
  assert.equal(stats.count, 0, "events outside the 10 ms window are excluded");
  assert.equal(stats.mostRecentReason, null);
  // …but the default 1 hour window still sees them.
  const stats1h = getPromotionGateRetryStats();
  assert.ok(stats1h.count >= 3, "default window still includes recent events");
  assert.equal(stats1h.windowMs, 60 * 60 * 1000, "default window is 1h");
});

test("Task #686: a successful enable with no retries records nothing (chip stays hidden)", async () => {
  _resetPromotionGateRetryEventsForTest();
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher({
      rows: [
        {
          recorded_at: 1,
          verification_status: "ok",
          passed: true,
          coins_with_promotion: ["bitcoin"],
          counts: { slices_promoted: 1 },
        },
      ],
    }),
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, true);
  const stats = getPromotionGateRetryStats();
  assert.equal(stats.count, 0, "happy path must not surface a retry chip");
});

test("Task #678: retry budget exhaustion before next backoff returns history_unreachable without a further attempt", async () => {
  // Sized so that the second attempt's failure leaves elapsed_ms close to
  // the budget; the next backoff would overshoot, so the gate must
  // budget-exhaust without a third attempt. We use scaled-down numbers
  // for test speed; the LOGIC under test is the production logic.
  const scripted = scriptedFetcher([
    { hang: true },
    { hang: true },
    { hang: true }, // should never be reached
  ]);
  const verdict = await hasPromotedSlice({
    fetcher: scripted.fn,
    mlEngineBaseUrl: "http://fake",
    timeoutMs: 100,
    retry: {
      maxAttempts: 3,
      // Budget = 250 ms. Attempt 1 hangs ≈ 100 ms, then 50 ms backoff =>
      // elapsed ≈ 150 ms. Attempt 2 hangs ≈ 100 ms => elapsed ≈ 250 ms.
      // Next backoff would be 100 ms, 250 + 100 ≥ 250 → budget exhausted.
      totalBudgetMs: 250,
      backoffBaseMs: 50,
    },
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "history_unreachable");
  assert.ok(
    scripted.calls.length <= 2,
    `expected ≤ 2 attempts before budget exhaustion, got ${scripted.calls.length}`,
  );
});
