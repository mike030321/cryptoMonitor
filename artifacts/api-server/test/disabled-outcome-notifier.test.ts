import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import {
  DEFAULT_WINDOW_MINUTES,
  DISABLED_OUTCOME_NOTIFIER_STATE_KEY,
  DISABLED_OUTCOME_RING_CAP,
  __resetRingBuffer,
  __setInlineDispatchDisabled,
  __snapshotRingBuffer,
  buildIncidentKey,
  buildIncidentReason,
  formatAlertTitle,
  getDisabledOutcomeAlertChannels,
  getDisabledOutcomeBannerState,
  recordDisabledOutcomeRejection,
  resolveWindowMs,
  runDisabledOutcomeNotifierTick,
  summarizeRecentEvents,
  type DisabledOutcomeEvent,
} from "../src/lib/disabled-outcome-notifier.ts";

const NOW = 1_777_777_777_000;
const WIN_5M = 5 * 60_000;

const ev = (
  tickId: string,
  timeframe: string,
  observedAt: number,
  sliceId: string | null = null,
): DisabledOutcomeEvent => ({
  tickId,
  sliceId,
  timeframe,
  observedAt,
});

let savedFetch: typeof fetch;
let savedGenericEnv: string | undefined;
let savedSlackEnv: string | undefined;
let savedSlackFallback: string | undefined;
let savedWindowEnv: string | undefined;

before(() => {
  savedFetch = globalThis.fetch;
  savedGenericEnv = process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL;
  savedSlackEnv = process.env.DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL;
  savedSlackFallback = process.env.SLACK_WEBHOOK_URL;
  savedWindowEnv = process.env.DISABLED_OUTCOME_WINDOW_MINUTES;
});

after(async () => {
  globalThis.fetch = savedFetch;
  if (savedGenericEnv === undefined)
    delete process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL;
  else process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL = savedGenericEnv;
  if (savedSlackEnv === undefined)
    delete process.env.DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL;
  else process.env.DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL = savedSlackEnv;
  if (savedSlackFallback === undefined) delete process.env.SLACK_WEBHOOK_URL;
  else process.env.SLACK_WEBHOOK_URL = savedSlackFallback;
  if (savedWindowEnv === undefined)
    delete process.env.DISABLED_OUTCOME_WINDOW_MINUTES;
  else process.env.DISABLED_OUTCOME_WINDOW_MINUTES = savedWindowEnv;
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, DISABLED_OUTCOME_NOTIFIER_STATE_KEY));
});

beforeEach(async () => {
  __resetRingBuffer();
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, DISABLED_OUTCOME_NOTIFIER_STATE_KEY));
  delete process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL;
  delete process.env.DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL;
  delete process.env.SLACK_WEBHOOK_URL;
  delete process.env.DISABLED_OUTCOME_WINDOW_MINUTES;
  // Reset fetch to a never-called default so tests that forget to
  // stub it fail loudly instead of hitting the network.
  globalThis.fetch = (async () => {
    throw new Error("test forgot to stub globalThis.fetch");
  }) as typeof fetch;
});

describe("disabled-outcome-notifier — pure helpers", () => {
  it("summarizeRecentEvents only counts events inside the window", () => {
    const events = [
      ev("a", "5m", NOW - WIN_5M - 1), // older than window — excluded
      ev("b", "5m", NOW - 60_000),
      ev("c", "1h", NOW - 30_000),
      ev("d", "5m", NOW - 1),
    ];
    const s = summarizeRecentEvents(events, NOW, WIN_5M);
    assert.equal(s.count, 3);
    assert.deepEqual(s.timeframes, ["1h", "5m"]);
    assert.deepEqual(s.perTimeframe, { "5m": 2, "1h": 1 });
    assert.equal(s.lastObservedAtIso, new Date(NOW - 1).toISOString());
  });

  it("summarizeRecentEvents returns empty summary when nothing is recent", () => {
    const s = summarizeRecentEvents(
      [ev("old", "5m", NOW - 10 * 60_000)],
      NOW,
      WIN_5M,
    );
    assert.equal(s.count, 0);
    assert.deepEqual(s.timeframes, []);
    assert.equal(s.lastObservedAtIso, null);
  });

  it("buildIncidentKey is null when there are no timeframes", () => {
    assert.equal(buildIncidentKey([]), null);
  });

  it("buildIncidentKey is stable for the same set regardless of input order", () => {
    assert.equal(
      buildIncidentKey(["1h", "5m", "1d"]),
      buildIncidentKey(["1d", "1h", "5m"]),
    );
    assert.equal(
      buildIncidentKey(["1h", "5m", "1d"]),
      "disabled_outcome@1d,1h,5m",
    );
  });

  it("buildIncidentKey changes when a NEW timeframe enters the set", () => {
    assert.notEqual(
      buildIncidentKey(["5m"]),
      buildIncidentKey(["5m", "1h"]),
      "fresh tf joining the leak must be a fresh incident",
    );
  });

  it("buildIncidentReason mentions count and per-tf breakdown", () => {
    const s = summarizeRecentEvents(
      [
        ev("a", "5m", NOW - 1),
        ev("b", "5m", NOW - 2),
        ev("c", "1h", NOW - 3),
      ],
      NOW,
      WIN_5M,
    );
    const reason = buildIncidentReason(s);
    assert.match(reason, /3 disabled-role outcomes/);
    assert.match(reason, /5m \(2\)/);
    assert.match(reason, /1h \(1\)/);
  });

  it("resolveWindowMs respects DISABLED_OUTCOME_WINDOW_MINUTES env", () => {
    process.env.DISABLED_OUTCOME_WINDOW_MINUTES = "12";
    assert.equal(resolveWindowMs(), 12 * 60_000);
  });

  it("resolveWindowMs floors to 60s for absurdly small overrides", () => {
    // 0.1 minute = 6s — should be clamped up to 60s.
    assert.equal(resolveWindowMs(6_000), 60_000);
  });

  it("resolveWindowMs falls back to default minutes when env is bad", () => {
    process.env.DISABLED_OUTCOME_WINDOW_MINUTES = "not-a-number";
    assert.equal(resolveWindowMs(), DEFAULT_WINDOW_MINUTES * 60_000);
  });
});

describe("disabled-outcome-notifier — concurrency", () => {
  it("a burst of inline ticks (production path: no eventsOverride) collapses to a single page on the rising edge", async () => {
    let webhookCalls = 0;
    globalThis.fetch = (async () => {
      webhookCalls += 1;
      // Add a small delay so concurrent ticks have time to overlap.
      await new Promise((r) => setTimeout(r, 30));
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;
    process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL = "https://example.test/hook";

    // Simulate a burst of 10 inline ticks racing against each other —
    // exactly what happens if 10 trades close at the same second and
    // each rejection triggers a fire-and-forget tick. Without the
    // singleflight every one of them would fire its own webhook.
    recordDisabledOutcomeRejection({
      tickId: "burst-1",
      sliceId: null,
      timeframe: "5m",
    });
    const ticks = await Promise.all(
      Array.from({ length: 10 }, () => runDisabledOutcomeNotifierTick()),
    );
    // Drain the inline tick from recordDisabledOutcomeRejection too.
    await new Promise((r) => setTimeout(r, 100));

    assert.equal(
      webhookCalls,
      1,
      "10 concurrent ticks on the same incident key must collapse to 1 webhook",
    );
    // The inline fire-and-forget tick from recordDisabledOutcomeRejection
    // got the rising edge first; every one of OUR 10 explicit ticks
    // should therefore see the active incident already set and dedupe.
    // We allow at most one "alerted" in case scheduling lets one of our
    // ticks beat the inline tick to the lock — what matters is that
    // they don't ALL fire (we already proved that with webhookCalls==1).
    const alerted = ticks.filter((t) => t.status === "alerted").length;
    assert.ok(
      alerted <= 1,
      `at most one of our ticks should claim the rising edge; got ${alerted}`,
    );
    for (const t of ticks) {
      assert.ok(
        t.status === "alerted" || t.status === "deduped",
        `every tick must report alerted or deduped; got ${t.status}`,
      );
    }
  });
});

describe("disabled-outcome-notifier — recordDisabledOutcomeRejection", () => {
  before(() => {
    __setInlineDispatchDisabled(true);
  });
  after(() => {
    __setInlineDispatchDisabled(false);
  });
  it("appends to the in-process ring buffer", () => {
    recordDisabledOutcomeRejection({
      tickId: "t1",
      sliceId: "s1",
      timeframe: "5m",
      observedAt: NOW,
    });
    const snap = __snapshotRingBuffer();
    assert.equal(snap.length, 1);
    assert.deepEqual(snap[0], {
      tickId: "t1",
      sliceId: "s1",
      timeframe: "5m",
      observedAt: NOW,
    });
  });

  it("caps the ring buffer at DISABLED_OUTCOME_RING_CAP", () => {
    for (let i = 0; i < DISABLED_OUTCOME_RING_CAP + 50; i++) {
      recordDisabledOutcomeRejection({
        tickId: `t${i}`,
        sliceId: null,
        timeframe: "5m",
        observedAt: NOW + i,
      });
    }
    const snap = __snapshotRingBuffer();
    assert.equal(snap.length, DISABLED_OUTCOME_RING_CAP);
    // Oldest entries dropped.
    assert.equal(snap[0].tickId, `t${50}`);
  });
});

describe("disabled-outcome-notifier — dispatch tick", () => {
  it("returns healthy and dispatches nothing when the window is empty", async () => {
    const r = await runDisabledOutcomeNotifierTick({
      now: NOW,
      eventsOverride: [],
    });
    assert.equal(r.status, "healthy");
    assert.equal(r.unhealthy, false);
    assert.equal(r.reason, null);
    assert.equal(r.summary.count, 0);
  });

  it("rising edge: dispatches incident webhook + persists active state", async () => {
    const captured: Array<{ url: string; body: unknown }> = [];
    globalThis.fetch = (async (url: string, init?: { body?: string }) => {
      captured.push({ url, body: init?.body ? JSON.parse(init.body) : null });
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;
    const r = await runDisabledOutcomeNotifierTick({
      now: NOW,
      webhookUrl: "https://example.test/hook",
      slackWebhookUrl: null,
      eventsOverride: [ev("tick-1", "5m", NOW - 30_000, "slice-A")],
    });

    assert.equal(r.status, "alerted");
    assert.equal(r.unhealthy, true);
    assert.match(r.reason ?? "", /1 disabled-role outcome rejected/);
    assert.equal(r.dispatched.find((d) => d.channel === "generic")?.outcome, "ok");
    assert.equal(r.dispatched.find((d) => d.channel === "slack")?.outcome, "skipped");
    assert.equal(r.state.activeIncidentKey, "disabled_outcome@5m");
    assert.equal(r.state.activeIncidentSince, new Date(NOW).toISOString());

    // Re-read persisted state.
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, DISABLED_OUTCOME_NOTIFIER_STATE_KEY));
    assert.ok(row, "state must be persisted");
    assert.equal(captured.length, 1, "exactly one webhook POST");
    assert.equal(captured[0].url, "https://example.test/hook");
  });

  it("dedupes when the same incident key persists", async () => {
    // First tick — rising edge.
    let captures = 0;
    globalThis.fetch = async () => {
      captures += 1;
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    };
    process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL = "https://example.test/hook";
    await runDisabledOutcomeNotifierTick({
      now: NOW,
      eventsOverride: [ev("t1", "5m", NOW - 60_000)],
    });
    assert.equal(captures, 1);

    // Second tick — same set of timeframes still leaking.
    await runDisabledOutcomeNotifierTick({
      now: NOW + 30_000,
      eventsOverride: [
        ev("t1", "5m", NOW - 60_000),
        ev("t2", "5m", NOW + 10_000),
      ],
    });
    // Should NOT have re-paged.
    assert.equal(captures, 1, "same set of leaking tfs should dedupe");
  });

  it("re-pages when a NEW timeframe enters the leaking set", async () => {
    let captures = 0;
    globalThis.fetch = async () => {
      captures += 1;
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    };
    process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL = "https://example.test/hook";
    await runDisabledOutcomeNotifierTick({
      now: NOW,
      eventsOverride: [ev("t1", "5m", NOW - 60_000)],
    });
    assert.equal(captures, 1);

    // A new tf joins.
    const r = await runDisabledOutcomeNotifierTick({
      now: NOW + 60_000,
      eventsOverride: [
        ev("t1", "5m", NOW - 60_000),
        ev("t2", "1h", NOW + 30_000),
      ],
    });
    assert.equal(captures, 2, "new tf joining set is a fresh incident");
    assert.equal(r.state.activeIncidentKey, "disabled_outcome@1h,5m");
  });

  it("recovery path: fires when the active window clears", async () => {
    let captures: string[] = [];
    globalThis.fetch = async (input, init) => {
      const body = init?.body as string | undefined;
      const parsed = body ? (JSON.parse(body) as { payload?: { kind?: string } }) : {};
      captures.push(parsed.payload?.kind ?? "unknown");
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    };
    process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL = "https://example.test/hook";

    // Rising edge.
    await runDisabledOutcomeNotifierTick({
      now: NOW,
      eventsOverride: [ev("t1", "5m", NOW - 60_000)],
    });
    assert.deepEqual(captures, ["incident"]);

    // Window cleared (event is now > 5m old).
    captures = [];
    const r = await runDisabledOutcomeNotifierTick({
      now: NOW + 10 * 60_000,
      eventsOverride: [ev("t1", "5m", NOW - 60_000)],
    });
    assert.equal(r.status, "recovered");
    assert.deepEqual(captures, ["recovery"]);
    assert.equal(r.state.activeIncidentKey, null);
    assert.equal(r.state.activeTimeframes.length, 0);
    assert.equal(r.state.lastAlertKind, "recovery");
  });

  it("tolerates webhook 5xx without throwing — channel_error reported and incident NOT marked alerted (so we retry)", async () => {
    globalThis.fetch = async () =>
      new Response("boom", { status: 500 });
    const r = await runDisabledOutcomeNotifierTick({
      now: NOW,
      webhookUrl: "https://example.test/broken",
      slackWebhookUrl: null,
      eventsOverride: [ev("t1", "5m", NOW - 1)],
    });
    assert.equal(r.status, "channel_error");
    assert.equal(r.dispatched.find((d) => d.channel === "generic")?.outcome, "error");
    // Liveliness fields refresh so the dashboard banner is accurate
    // (timeframes leaking, count) even while we retry. But the
    // dedup key is NOT advanced — otherwise the next tick would
    // dedupe and we'd never retry the page.
    assert.deepEqual(r.state.activeTimeframes, ["5m"]);
    assert.equal(r.state.activeEventCount, 1);
    assert.equal(
      r.state.activeIncidentKey,
      null,
      "must NOT mark incident alerted when all channels failed",
    );
    assert.equal(
      r.state.lastAlertKind,
      null,
      "lastAlertKind must NOT advance when all channels failed",
    );
  });

  it("retries dispatch on the next tick after a transient failure, then converges (exactly one webhook per incident key)", async () => {
    let attempt = 0;
    let successfulCalls = 0;
    globalThis.fetch = (async () => {
      attempt += 1;
      if (attempt === 1) {
        // First attempt fails — outbound network blip.
        return new Response("boom", { status: 500 });
      }
      successfulCalls += 1;
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;

    const events = [ev("t1", "5m", NOW - 1)];
    // Tick 1: rising edge, channels fail → status=channel_error.
    const r1 = await runDisabledOutcomeNotifierTick({
      now: NOW,
      webhookUrl: "https://example.test/flaky",
      slackWebhookUrl: null,
      eventsOverride: events,
    });
    assert.equal(r1.status, "channel_error");
    assert.equal(r1.state.activeIncidentKey, null);

    // Tick 2: same incident key still leaking, channel recovered →
    // we must RETRY the page (not dedupe).
    const r2 = await runDisabledOutcomeNotifierTick({
      now: NOW + 30_000,
      webhookUrl: "https://example.test/flaky",
      slackWebhookUrl: null,
      eventsOverride: events,
    });
    assert.equal(r2.status, "alerted", "must retry after channel recovers");
    assert.equal(r2.state.activeIncidentKey, "disabled_outcome@5m");
    assert.equal(successfulCalls, 1);

    // Tick 3: same incident key still leaking, channels healthy → now
    // dedupe (we already paged successfully).
    const r3 = await runDisabledOutcomeNotifierTick({
      now: NOW + 60_000,
      webhookUrl: "https://example.test/flaky",
      slackWebhookUrl: null,
      eventsOverride: events,
    });
    assert.equal(r3.status, "deduped");
    assert.equal(successfulCalls, 1, "exactly one webhook per incident key");
  });

  it("payload includes tickId, sliceId, and timeframe for each recent event", async () => {
    let captured: { payload?: { recentEvents?: unknown[] } } = {};
    globalThis.fetch = async (_url, init) => {
      const body = init?.body as string | undefined;
      captured = body ? (JSON.parse(body) as typeof captured) : {};
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    };
    await runDisabledOutcomeNotifierTick({
      now: NOW,
      webhookUrl: "https://example.test/hook",
      slackWebhookUrl: null,
      eventsOverride: [
        ev("tick-AAA", "5m", NOW - 60_000, "slice-XYZ"),
        ev("tick-BBB", "1h", NOW - 30_000, null),
      ],
    });
    const recent = captured.payload?.recentEvents as Array<{
      tickId: string;
      sliceId: string | null;
      timeframe: string;
    }>;
    assert.ok(Array.isArray(recent));
    assert.equal(recent.length, 2);
    assert.equal(recent[0].tickId, "tick-AAA");
    assert.equal(recent[0].sliceId, "slice-XYZ");
    assert.equal(recent[0].timeframe, "5m");
    assert.equal(recent[1].tickId, "tick-BBB");
    assert.equal(recent[1].sliceId, null);
    assert.equal(recent[1].timeframe, "1h");
  });
});

describe("disabled-outcome-notifier — getDisabledOutcomeBannerState", () => {
  it("bannerVisible=false when there are no events", async () => {
    const s = await getDisabledOutcomeBannerState({ now: NOW });
    assert.equal(s.bannerVisible, false);
    assert.equal(s.eventCount, 0);
    assert.deepEqual(s.timeframes, []);
  });

  it("surfaces recent events with all three correlation fields", async () => {
    const s = await getDisabledOutcomeBannerState({
      now: NOW,
      eventsOverride: [
        ev("tick-1", "5m", NOW - 60_000, "slice-1"),
        ev("tick-2", "1h", NOW - 30_000, null),
      ],
    });
    assert.equal(s.bannerVisible, true);
    assert.equal(s.eventCount, 2);
    assert.deepEqual(s.timeframes, ["1h", "5m"]);
    assert.equal(s.recentEvents[0].tickId, "tick-1");
    assert.equal(s.recentEvents[0].sliceId, "slice-1");
    assert.equal(s.recentEvents[1].sliceId, null);
  });

  it("surfaces alertHook configuration", async () => {
    process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL = "https://example.test/h";
    const s = await getDisabledOutcomeBannerState({
      now: NOW,
      eventsOverride: [ev("t", "5m", NOW)],
    });
    assert.equal(s.alertHook.configured, true);
    assert.equal(s.alertHook.genericConfigured, true);
    assert.equal(s.alertHook.slackConfigured, false);
  });
});

describe("disabled-outcome-notifier — alert channels & title formatting", () => {
  it("getDisabledOutcomeAlertChannels reflects env presence", () => {
    assert.deepEqual(getDisabledOutcomeAlertChannels(), {
      configured: false,
      genericConfigured: false,
      slackConfigured: false,
    });
    process.env.DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL = "https://x";
    assert.equal(getDisabledOutcomeAlertChannels().configured, true);
    assert.equal(getDisabledOutcomeAlertChannels().slackConfigured, true);
  });

  it("formatAlertTitle returns distinct strings for incident vs recovery", () => {
    assert.match(
      formatAlertTitle({
        kind: "incident",
        reason: "x",
        windowMinutes: 5,
        timeframes: ["5m"],
        perTimeframe: { "5m": 1 },
        eventCount: 1,
        recentEvents: [],
        incidentSince: null,
      }),
      /rejected/i,
    );
    assert.match(
      formatAlertTitle({
        kind: "recovery",
        reason: "x",
        windowMinutes: 5,
        timeframes: [],
        perTimeframe: {},
        eventCount: 0,
        recentEvents: [],
        incidentSince: null,
      }),
      /recovered/i,
    );
  });
});
