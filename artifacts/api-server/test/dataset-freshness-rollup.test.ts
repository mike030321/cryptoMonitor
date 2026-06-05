import { test } from "node:test";
import assert from "node:assert/strict";

import {
  computeDatasetFreshness,
  type DatasetFreshnessAlertEntry,
} from "../src/routes/crypto/index";

// Task #554 — pin the rollup logic that powers /crypto/datasets/freshness.
// This is the function that decides green/amber/red/unknown per timeframe
// (and the top-level state) and feeds the dashboard banner. The rules are
// documented next to the function; these tests pin each rule.

const T0 = Date.parse("2026-04-28T14:00:00Z");
const HOUR = 3600 * 1000;

function makeStatus(
  tfs: Record<string, Record<string, unknown>>,
  writtenAt = "2026-04-28T13:59:00Z",
): Record<string, unknown> {
  return { written_at: writtenAt, timeframes: tfs };
}

function makeAlert(
  at: string,
  timeframe: string,
  error = "boom",
): DatasetFreshnessAlertEntry {
  return {
    at,
    timeframe,
    status: "error",
    error,
    cadenceHours: 6,
    unread: false,
    raw: JSON.stringify({ at, timeframe, status: "error", error }),
  };
}

test("green: fresh success, next_due_at in the future, no alerts", () => {
  const status = makeStatus({
    "1h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T13:00:00Z",
      last_attempt_at: "2026-04-28T13:00:00Z",
      last_status: "ok",
      last_error: null,
      next_due_at: "2026-04-28T19:00:00Z",
    },
  });
  const out = computeDatasetFreshness(status, [], T0);
  assert.equal(out.state, "green");
  assert.equal(out.timeframes[0].health, "green");
  assert.equal(out.pastDueTimeframes.length, 0);
  assert.equal(out.totalUnreadAlerts, 0);
});

test("amber: past next_due_at but inside 1.5×cadence", () => {
  // last_success 5h ago (cadence 6h, so age < cadence), but next_due is
  // 30min in the past — amber per the writer's own schedule.
  const status = makeStatus({
    "1h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T09:00:00Z", // 5h ago
      last_attempt_at: "2026-04-28T09:00:00Z",
      last_status: "ok",
      last_error: null,
      next_due_at: "2026-04-28T13:30:00Z", // 30min ago
    },
  });
  const out = computeDatasetFreshness(status, [], T0);
  assert.equal(out.timeframes[0].health, "amber");
  assert.equal(out.state, "amber");
  assert.deepEqual(out.pastDueTimeframes, ["1h"]);
  assert.ok(
    (out.timeframes[0].pastDueSeconds ?? 0) >= 30 * 60 - 1,
    "pastDueSeconds should be derived from next_due_at",
  );
});

test("amber fallback: no next_due_at, age between cadence and 1.5×cadence", () => {
  const status = makeStatus({
    "1h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T06:30:00Z", // 7.5h ago, > 6h, < 9h
      last_attempt_at: "2026-04-28T06:30:00Z",
      last_status: "ok",
      last_error: null,
      next_due_at: null, // writer omitted it
    },
  });
  const out = computeDatasetFreshness(status, [], T0);
  assert.equal(out.timeframes[0].health, "amber");
  assert.equal(out.state, "amber");
});

test("red: age > 1.5×cadence, even when next_due_at says we're fine", () => {
  // Writer's next_due is in the future (it just bumped the schedule)
  // but the actual last success is 12h ago for a 6h cadence — a stale
  // cache should still scream red regardless of the writer's optimism.
  const status = makeStatus({
    "1h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T02:00:00Z", // 12h ago, > 1.5×6h = 9h
      last_attempt_at: "2026-04-28T02:00:00Z",
      last_status: "ok",
      last_error: null,
      next_due_at: "2026-04-28T20:00:00Z", // future
    },
  });
  const out = computeDatasetFreshness(status, [], T0);
  assert.equal(out.timeframes[0].health, "red");
  assert.equal(out.state, "red");
});

test("red: unread alert (newer than last_success_at) escalates green→red", () => {
  const status = makeStatus({
    "1h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T13:00:00Z", // 1h ago — would be green
      last_attempt_at: "2026-04-28T13:30:00Z",
      last_status: "error",
      last_error: "DBPoolExhausted",
      next_due_at: "2026-04-28T19:00:00Z",
    },
  });
  const alerts = [makeAlert("2026-04-28T13:30:00Z", "1h", "DBPoolExhausted")];
  const out = computeDatasetFreshness(status, alerts, T0);
  assert.equal(out.timeframes[0].health, "red");
  assert.equal(out.state, "red");
  assert.equal(out.totalUnreadAlerts, 1);
  assert.equal(out.alerts[0].unread, true);
});

test("alerts older than last_success_at do not count as unread", () => {
  const status = makeStatus({
    "1h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T13:00:00Z",
      last_attempt_at: "2026-04-28T13:00:00Z",
      last_status: "ok",
      last_error: null,
      next_due_at: "2026-04-28T19:00:00Z",
    },
  });
  // Failure happened 2h ago, then a successful tick happened 1h ago.
  const alerts = [makeAlert("2026-04-28T12:00:00Z", "1h", "since-resolved")];
  const out = computeDatasetFreshness(status, alerts, T0);
  assert.equal(out.timeframes[0].health, "green");
  assert.equal(out.totalUnreadAlerts, 0);
  assert.equal(out.alerts[0].unread, false);
});

test("red: missing last_success_at → red (never-run timeframe)", () => {
  const status = makeStatus({
    "1h": {
      cadence_hours: 6,
      last_success_at: null,
      last_attempt_at: null,
      last_status: null,
      last_error: null,
      next_due_at: null,
    },
  });
  const out = computeDatasetFreshness(status, [], T0);
  assert.equal(out.timeframes[0].health, "red");
  assert.equal(out.state, "red");
});

test("unknown: no timeframes parsed (empty / missing status object)", () => {
  assert.equal(computeDatasetFreshness(null, [], T0).state, "unknown");
  assert.equal(computeDatasetFreshness({}, [], T0).state, "unknown");
  assert.equal(
    computeDatasetFreshness({ timeframes: {} }, [], T0).state,
    "unknown",
  );
});

test("rollup: red wins over amber, amber wins over green", () => {
  const status = makeStatus({
    "5m": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T13:00:00Z",
      next_due_at: "2026-04-28T19:00:00Z",
    },
    "1h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T09:00:00Z",
      next_due_at: "2026-04-28T13:30:00Z", // amber
    },
    "2h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T02:00:00Z", // 12h, red
      next_due_at: "2026-04-28T20:00:00Z",
    },
  });
  const out = computeDatasetFreshness(status, [], T0);
  assert.equal(out.state, "red");
  // pastDueTimeframes lists both red and amber.
  assert.deepEqual(out.pastDueTimeframes.sort(), ["1h", "2h"]);
  // Map by tf for clarity.
  const byTf = Object.fromEntries(out.timeframes.map((t) => [t.timeframe, t]));
  assert.equal(byTf["5m"].health, "green");
  assert.equal(byTf["1h"].health, "amber");
  assert.equal(byTf["2h"].health, "red");
});

test("ageSeconds and pastDueSeconds reflect the writer's clock", () => {
  const status = makeStatus({
    "1h": {
      cadence_hours: 6,
      last_success_at: "2026-04-28T12:00:00Z", // 2h ago
      next_due_at: "2026-04-28T13:30:00Z", // 30min ago
    },
  });
  const out = computeDatasetFreshness(status, [], T0);
  const tf = out.timeframes[0];
  assert.equal(tf.ageSeconds, 2 * 3600);
  assert.equal(tf.pastDueSeconds, 30 * 60);
});

void HOUR; // keep the constant in scope for future cases
