import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import {
  ERROR_STREAK_THRESHOLD,
  TOPUP_5M_NOTIFIER_STATE_KEY,
  evaluateTopupStatus,
  getTopup5mAlertChannels,
  getTopup5mNotifierState,
  runTopup5mNotifierTick,
  type Topup5mNotifierState,
  type TopupStatus,
} from "../src/lib/topup-5m-notifier.ts";

const NOW = 1_777_777_777_000;
const TICK_T0 = 1_777_700_000;

const baseStatus = (overrides: Partial<TopupStatus> = {}): TopupStatus => ({
  enabled: true,
  alert_below_days: 310,
  last_check_at: TICK_T0,
  last_attempt_outcome: "ok",
  last_finished_at: TICK_T0 + 5,
  last_error: null,
  last_alerts: [],
  last_health_per_coin: { BTC: 365.0, ETH: 360.0, SOL: 350.0 },
  ticks_total: 1,
  runs_total: 1,
  ...overrides,
});

const emptyPrev = (): Topup5mNotifierState => ({
  activeIncidentKey: null,
  activeIncidentReason: null,
  activeIncidentSince: null,
  lastAlertAt: null,
  lastAlertKind: null,
  lastAlertReason: null,
  consecutiveErrors: 0,
  lastObservedCheckAt: null,
});

let savedFetch: typeof fetch;
let savedGenericEnv: string | undefined;
let savedSlackEnv: string | undefined;
let savedSlackFallback: string | undefined;

before(() => {
  savedFetch = globalThis.fetch;
  savedGenericEnv = process.env.TOPUP_5M_ALERT_WEBHOOK_URL;
  savedSlackEnv = process.env.TOPUP_5M_ALERT_SLACK_WEBHOOK_URL;
  savedSlackFallback = process.env.SLACK_WEBHOOK_URL;
});

after(async () => {
  globalThis.fetch = savedFetch;
  if (savedGenericEnv === undefined) delete process.env.TOPUP_5M_ALERT_WEBHOOK_URL;
  else process.env.TOPUP_5M_ALERT_WEBHOOK_URL = savedGenericEnv;
  if (savedSlackEnv === undefined) delete process.env.TOPUP_5M_ALERT_SLACK_WEBHOOK_URL;
  else process.env.TOPUP_5M_ALERT_SLACK_WEBHOOK_URL = savedSlackEnv;
  if (savedSlackFallback === undefined) delete process.env.SLACK_WEBHOOK_URL;
  else process.env.SLACK_WEBHOOK_URL = savedSlackFallback;
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, TOPUP_5M_NOTIFIER_STATE_KEY));
});

beforeEach(async () => {
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, TOPUP_5M_NOTIFIER_STATE_KEY));
  delete process.env.TOPUP_5M_ALERT_WEBHOOK_URL;
  delete process.env.TOPUP_5M_ALERT_SLACK_WEBHOOK_URL;
  delete process.env.SLACK_WEBHOOK_URL;
});

describe("topup-5m-notifier — evaluateTopupStatus", () => {
  it("returns healthy on ok outcome with no alerts", () => {
    const r = evaluateTopupStatus(baseStatus(), emptyPrev());
    assert.equal(r.unhealthy, false);
    assert.equal(r.incidentKey, null);
    assert.equal(r.nextConsecutiveErrors, 0);
    assert.equal(r.freshTick, true);
  });

  it("does not flag a single error tick — needs two in a row", () => {
    const r = evaluateTopupStatus(
      baseStatus({ last_attempt_outcome: "error", last_error: "boom" }),
      emptyPrev(),
    );
    assert.equal(r.unhealthy, false);
    assert.equal(r.nextConsecutiveErrors, 1);
    assert.equal(r.incidentKey, null);
  });

  it("flags errors@active after two consecutive error ticks", () => {
    const prev = { ...emptyPrev(), consecutiveErrors: 1, lastObservedCheckAt: TICK_T0 - 86400 };
    const r = evaluateTopupStatus(
      baseStatus({ last_attempt_outcome: "error", last_error: "boom2" }),
      prev,
    );
    assert.equal(r.unhealthy, true);
    assert.equal(r.nextConsecutiveErrors, 2);
    assert.equal(r.incidentKey, "errors@active");
    assert.match(r.reason ?? "", /failed 2 times/);
    assert.match(r.reason ?? "", /boom2/);
  });

  it("ERROR_STREAK_THRESHOLD is exposed and is 2 (matches task spec)", () => {
    assert.equal(ERROR_STREAK_THRESHOLD, 2);
  });

  it("does NOT advance counter on a stale poll (same last_check_at)", () => {
    const prev = { ...emptyPrev(), consecutiveErrors: 1, lastObservedCheckAt: TICK_T0 };
    const r = evaluateTopupStatus(
      baseStatus({ last_attempt_outcome: "error", last_check_at: TICK_T0 }),
      prev,
    );
    assert.equal(r.freshTick, false);
    assert.equal(r.nextConsecutiveErrors, 1, "watcher poll cadence must not inflate streak");
    assert.equal(r.unhealthy, false);
  });

  it("ok outcome on a fresh tick resets the streak", () => {
    const prev = { ...emptyPrev(), consecutiveErrors: 5, lastObservedCheckAt: TICK_T0 - 86400 };
    const r = evaluateTopupStatus(baseStatus({ last_attempt_outcome: "ok" }), prev);
    assert.equal(r.nextConsecutiveErrors, 0);
    assert.equal(r.unhealthy, false);
  });

  it("disabled outcome does NOT reset the streak (preserves existing incident)", () => {
    const prev = { ...emptyPrev(), consecutiveErrors: 2, lastObservedCheckAt: TICK_T0 - 86400 };
    const r = evaluateTopupStatus(baseStatus({ last_attempt_outcome: "disabled" }), prev);
    assert.equal(r.nextConsecutiveErrors, 2);
    assert.equal(r.unhealthy, true);
    assert.equal(r.incidentKey, "errors@active");
  });

  it("non-empty last_alerts flags an alerts@<coins> incident", () => {
    const r = evaluateTopupStatus(
      baseStatus({ last_alerts: ["SOL", "BTC"] }),
      emptyPrev(),
    );
    assert.equal(r.unhealthy, true);
    assert.equal(r.incidentKey, "alerts@BTC,SOL", "coins must be sorted for stable dedup");
    assert.deepEqual(r.alertCoins, ["BTC", "SOL"]);
    assert.match(r.reason ?? "", /below threshold for 2 coin/);
  });

  it("errors take precedence over alerts when both present", () => {
    const prev = { ...emptyPrev(), consecutiveErrors: 1, lastObservedCheckAt: TICK_T0 - 86400 };
    const r = evaluateTopupStatus(
      baseStatus({
        last_attempt_outcome: "error",
        last_alerts: ["BTC"],
        last_error: "ml-engine OOM",
      }),
      prev,
    );
    assert.equal(r.incidentKey, "errors@active");
    assert.match(r.reason ?? "", /failed 2 times/);
  });

  it("changing the alert coin set yields a new incident key", () => {
    const r1 = evaluateTopupStatus(
      baseStatus({ last_alerts: ["BTC"] }),
      emptyPrev(),
    );
    const r2 = evaluateTopupStatus(
      baseStatus({ last_alerts: ["BTC", "ETH"], last_check_at: TICK_T0 + 86400 }),
      { ...emptyPrev(), lastObservedCheckAt: TICK_T0 },
    );
    assert.notEqual(r1.incidentKey, r2.incidentKey);
  });

  it("first observation (no prior tick observed) counts as a fresh tick", () => {
    const r = evaluateTopupStatus(baseStatus({ last_check_at: TICK_T0 }), emptyPrev());
    assert.equal(r.freshTick, true);
  });

  it("last_check_at == null is not a fresh tick", () => {
    const r = evaluateTopupStatus(
      baseStatus({ last_check_at: null, last_attempt_outcome: null }),
      emptyPrev(),
    );
    assert.equal(r.freshTick, false);
    assert.equal(r.nextConsecutiveErrors, 0);
  });

  // ── Task #442 — stuck-replica branch ──────────────────────────────────
  it("flags stuck_replica@<name> when ml-engine reports stuck_replica", () => {
    const r = evaluateTopupStatus(
      baseStatus({
        stuck_replica: "host-a/pid=1",
        stuck_replica_streak: 9,
        stuck_replica_threshold: 7,
      }),
      emptyPrev(),
    );
    assert.equal(r.unhealthy, true);
    assert.equal(r.incidentKey, "stuck_replica@host-a/pid=1");
    assert.match(r.reason ?? "", /host-a\/pid=1/);
    assert.match(r.reason ?? "", /9 ticks in a row/);
    assert.match(r.reason ?? "", /threshold: 7/);
  });

  it("stuck_replica is healthy when ml-engine leaves the field null", () => {
    // The ml-engine pre-applies the threshold check — `stuck_replica:
    // null` means "below threshold OR not stuck". The notifier must
    // not re-derive that decision from `stuck_replica_streak` alone.
    const r = evaluateTopupStatus(
      baseStatus({
        stuck_replica: null,
        stuck_replica_streak: 6,
        stuck_replica_threshold: 7,
      }),
      emptyPrev(),
    );
    assert.equal(r.unhealthy, false);
    assert.equal(r.incidentKey, null);
  });

  it("errors take precedence over stuck-replica when both present", () => {
    // A stuck replica AND two consecutive errors — operators care
    // first about the gate-erosion failure, the fairness concern can
    // wait until the engine is healthy again. The dedup key reflects
    // the more severe condition so a recovery clears the right one.
    const prev = { ...emptyPrev(), consecutiveErrors: 1, lastObservedCheckAt: TICK_T0 - 86400 };
    const r = evaluateTopupStatus(
      baseStatus({
        last_attempt_outcome: "error",
        last_error: "boom",
        stuck_replica: "host-a/pid=1",
        stuck_replica_streak: 9,
        stuck_replica_threshold: 7,
      }),
      prev,
    );
    assert.equal(r.incidentKey, "errors@active");
  });

  it("alert coins take precedence over stuck-replica", () => {
    // Same logic — a coin already below the gate threshold is a
    // stronger signal than "this one box has been doing all the
    // pulling". Stuck-replica falls through last.
    const r = evaluateTopupStatus(
      baseStatus({
        last_alerts: ["BTC"],
        stuck_replica: "host-a/pid=1",
        stuck_replica_streak: 9,
        stuck_replica_threshold: 7,
      }),
      emptyPrev(),
    );
    assert.equal(r.incidentKey, "alerts@BTC");
  });

  it("changing the stuck replica name yields a new incident key", () => {
    const r1 = evaluateTopupStatus(
      baseStatus({
        stuck_replica: "host-a/pid=1",
        stuck_replica_streak: 9,
        stuck_replica_threshold: 7,
      }),
      emptyPrev(),
    );
    const r2 = evaluateTopupStatus(
      baseStatus({
        stuck_replica: "host-b/pid=2",
        stuck_replica_streak: 9,
        stuck_replica_threshold: 7,
        last_check_at: TICK_T0 + 86400,
      }),
      { ...emptyPrev(), lastObservedCheckAt: TICK_T0 },
    );
    assert.notEqual(r1.incidentKey, r2.incidentKey);
  });
});

describe("topup-5m-notifier — runTopup5mNotifierTick", () => {
  it("noop on healthy with no prior incident — no webhooks fired", async () => {
    let called = 0;
    globalThis.fetch = (async () => {
      called++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;
    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "healthy");
    assert.equal(out.unhealthy, false);
    assert.equal(called, 0);
    assert.equal(out.state.lastObservedCheckAt, TICK_T0);
  });

  it("does NOT fire on a single error tick (needs two)", async () => {
    let called = 0;
    globalThis.fetch = (async () => {
      called++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;
    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({ last_attempt_outcome: "error", last_error: "x" }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "healthy");
    assert.equal(out.state.consecutiveErrors, 1);
    assert.equal(called, 0);
  });

  it("fires on the 2nd consecutive error tick, dedupes on the 3rd", async () => {
    const calls: Array<{ url: string; body: string }> = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      calls.push({ url: String(url), body: String(init?.body ?? "") });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runTopup5mNotifierTick({
      statusOverride: baseStatus({
        last_attempt_outcome: "error",
        last_error: "engine OOM",
        last_check_at: TICK_T0,
      }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(calls.length, 0);

    const out2 = await runTopup5mNotifierTick({
      statusOverride: baseStatus({
        last_attempt_outcome: "error",
        last_error: "engine OOM",
        last_check_at: TICK_T0 + 86400,
      }),
      now: NOW + 86_400_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out2.status, "alerted");
    assert.equal(calls.length, 1);
    assert.match(calls[0].body, /incident/);
    assert.match(calls[0].body, /engine OOM/);

    const out3 = await runTopup5mNotifierTick({
      statusOverride: baseStatus({
        last_attempt_outcome: "error",
        last_error: "engine OOM still",
        last_check_at: TICK_T0 + 2 * 86400,
      }),
      now: NOW + 2 * 86_400_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out3.status, "deduped", "continued failing tick must not re-page");
    assert.equal(calls.length, 1);
  });

  it("fires immediately on first non-empty last_alerts even with ok outcome", async () => {
    const calls: string[] = [];
    globalThis.fetch = (async (_u, init?: RequestInit) => {
      calls.push(String(init?.body ?? ""));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;
    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({
        last_alerts: ["SOL"],
        last_health_per_coin: { SOL: 308.5, BTC: 320 },
      }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "alerted");
    assert.equal(calls.length, 1);
    assert.match(calls[0], /SOL/);
    assert.match(calls[0], /308\.5/);
  });

  it("fires recovery when alert coins clear", async () => {
    const calls: string[] = [];
    globalThis.fetch = (async (_u, init?: RequestInit) => {
      calls.push(String(init?.body ?? ""));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runTopup5mNotifierTick({
      statusOverride: baseStatus({ last_alerts: ["SOL"] }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({ last_alerts: [], last_check_at: TICK_T0 + 86400 }),
      now: NOW + 86_400_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "recovered");
    assert.equal(out.state.activeIncidentKey, null);
    assert.equal(out.state.lastAlertKind, "recovery");
    assert.equal(calls.length, 2);
    assert.match(calls[1], /recovery/);
  });

  // Task #442 — end-to-end dispatch carries the new stuck-replica fields
  // through the full pipeline: status → evaluate → buildPayload → fetch
  // body. This is what an on-call engineer actually sees in their pager.
  it("end-to-end dispatch payload carries stuck-replica fields on the wire", async () => {
    const calls: string[] = [];
    globalThis.fetch = (async (_u, init?: RequestInit) => {
      calls.push(String(init?.body ?? ""));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({
        stuck_replica: "host-a/pid=1",
        stuck_replica_streak: 9,
        stuck_replica_threshold: 7,
      }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });

    assert.equal(out.status, "alerted");
    assert.equal(out.state.activeIncidentKey, "stuck_replica@host-a/pid=1");
    assert.equal(calls.length, 1);
    const body = JSON.parse(calls[0]);
    // The webhook receives a structured payload — confirm both the
    // human-readable surface (title/reason) and the machine-readable
    // fields downstream PagerDuty/Slack routers depend on.
    assert.equal(body.payload.stuckReplica, "host-a/pid=1");
    assert.equal(body.payload.stuckReplicaStreak, 9);
    assert.equal(body.payload.stuckReplicaThreshold, 7);
    assert.match(body.payload.reason, /host-a\/pid=1/);
    assert.match(body.payload.reason, /9 ticks in a row/);
  });

  // Recovery for a stuck-replica incident: once the head replica
  // changes (or the streak otherwise drops back below threshold), the
  // ml-engine sets `stuck_replica` back to null and the notifier must
  // fire a recovery and clear the incident.
  it("clears stuck-replica incident when ml-engine reports null again", async () => {
    const calls: string[] = [];
    globalThis.fetch = (async (_u, init?: RequestInit) => {
      calls.push(String(init?.body ?? ""));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runTopup5mNotifierTick({
      statusOverride: baseStatus({
        stuck_replica: "host-a/pid=1",
        stuck_replica_streak: 9,
        stuck_replica_threshold: 7,
      }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(calls.length, 1);

    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({
        stuck_replica: null,
        stuck_replica_streak: 1,
        stuck_replica_threshold: 7,
        last_check_at: TICK_T0 + 86400,
      }),
      now: NOW + 86_400_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "recovered");
    assert.equal(out.state.activeIncidentKey, null);
    assert.equal(calls.length, 2);
    assert.match(calls[1], /recovery/);
  });

  it("a new set of alert coins during an active incident re-pages", async () => {
    let count = 0;
    globalThis.fetch = (async () => {
      count++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runTopup5mNotifierTick({
      statusOverride: baseStatus({ last_alerts: ["SOL"] }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({
        last_alerts: ["SOL", "ETH"],
        last_check_at: TICK_T0 + 86400,
      }),
      now: NOW + 86_400_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "alerted");
    assert.equal(count, 2);
  });

  it("dispatches to both generic and slack channels when configured", async () => {
    const seen: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request) => {
      seen.push(String(url));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({ last_alerts: ["BTC"] }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: "https://hooks.slack.example/services/x",
    });
    assert.equal(out.status, "alerted");
    assert.equal(seen.length, 2);
    assert.ok(seen.some((u) => u.endsWith("/hook")));
    assert.ok(seen.some((u) => u.includes("hooks.slack.example")));
  });

  it("with no channels configured, still tracks state for the dashboard banner", async () => {
    let called = 0;
    globalThis.fetch = (async () => {
      called++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;
    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({ last_alerts: ["BTC"] }),
      now: NOW,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    assert.equal(called, 0);
    assert.equal(out.unhealthy, true);
    assert.equal(out.state.activeIncidentKey, "alerts@BTC");
    const persisted = await getTopup5mNotifierState();
    assert.equal(persisted.activeIncidentKey, "alerts@BTC");
  });

  it("logs but does not throw on webhook failure", async () => {
    globalThis.fetch = (async () =>
      new Response("nope", { status: 500 })) as typeof fetch;
    const out = await runTopup5mNotifierTick({
      statusOverride: baseStatus({ last_alerts: ["BTC"] }),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "channel_error");
    assert.equal(out.state.activeIncidentKey, "alerts@BTC", "active incident still recorded");
  });

  it("ml-engine fetch failure yields skipped status, not crash", async () => {
    globalThis.fetch = (async () => {
      throw new Error("connection refused");
    }) as typeof fetch;
    const out = await runTopup5mNotifierTick({
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "skipped");
  });
});

describe("topup-5m-notifier — getTopup5mAlertChannels", () => {
  it("falls back to SLACK_WEBHOOK_URL when topup-specific slack url is unset", () => {
    process.env.SLACK_WEBHOOK_URL = "https://slack.example/x";
    const channels = getTopup5mAlertChannels();
    assert.equal(channels.configured, true);
    assert.equal(channels.slackConfigured, true);
    assert.equal(channels.genericConfigured, false);
  });

  it("reports unconfigured when nothing is set", () => {
    const channels = getTopup5mAlertChannels();
    assert.equal(channels.configured, false);
    assert.equal(channels.slackConfigured, false);
    assert.equal(channels.genericConfigured, false);
  });
});
