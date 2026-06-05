import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import {
  MARKET_SIGNALS_WATCHER_HISTORY_KEY,
  MARKET_SIGNALS_WATCHER_HISTORY_MAX,
  MARKET_SIGNALS_WATCHER_SNOOZE_KEY,
  MARKET_SIGNALS_WATCHER_STATE_KEY,
  clearMarketSignalsWatcherSnooze,
  computeSnoozeUntil,
  evaluatePollerStatus,
  getMarketSignalsAlertChannels,
  getMarketSignalsWatcherHistory,
  getMarketSignalsWatcherSnooze,
  getMarketSignalsWatcherState,
  resolveChannels,
  runMarketSignalsWatcherTick,
  setMarketSignalsWatcherSnooze,
} from "../src/lib/market-signals-poller-watcher.ts";

const NOW = 1_777_777_777_000;
const INTERVAL = 60_000;
const STALE_THRESHOLD = INTERVAL * 3;

const healthyStatus = (overrides: Partial<{ lastPollAt: number | null; lastPollOk: boolean; lastPollError: string | null }> = {}) => ({
  lastPollAt: NOW - 30_000,
  lastPollOk: true,
  lastPollError: null,
  intervalMs: INTERVAL,
  ...overrides,
});

const staleStatus = () => ({
  lastPollAt: NOW - (STALE_THRESHOLD + 60_000),
  lastPollOk: true,
  lastPollError: null,
  intervalMs: INTERVAL,
});

const erroredStatus = (msg = "OKX 500") => ({
  lastPollAt: NOW - 30_000,
  lastPollOk: false,
  lastPollError: msg,
  intervalMs: INTERVAL,
});

let savedFetch: typeof fetch;
let savedGenericEnv: string | undefined;
let savedSlackEnv: string | undefined;
let savedSlackFallback: string | undefined;

before(() => {
  savedFetch = globalThis.fetch;
  savedGenericEnv = process.env.MARKET_SIGNALS_ALERT_WEBHOOK_URL;
  savedSlackEnv = process.env.MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL;
  savedSlackFallback = process.env.SLACK_WEBHOOK_URL;
});

after(async () => {
  globalThis.fetch = savedFetch;
  if (savedGenericEnv === undefined) delete process.env.MARKET_SIGNALS_ALERT_WEBHOOK_URL;
  else process.env.MARKET_SIGNALS_ALERT_WEBHOOK_URL = savedGenericEnv;
  if (savedSlackEnv === undefined) delete process.env.MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL;
  else process.env.MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL = savedSlackEnv;
  if (savedSlackFallback === undefined) delete process.env.SLACK_WEBHOOK_URL;
  else process.env.SLACK_WEBHOOK_URL = savedSlackFallback;
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_STATE_KEY));
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_HISTORY_KEY));
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_SNOOZE_KEY));
});

beforeEach(async () => {
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_STATE_KEY));
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_HISTORY_KEY));
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_SNOOZE_KEY));
  delete process.env.MARKET_SIGNALS_ALERT_WEBHOOK_URL;
  delete process.env.MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL;
  delete process.env.SLACK_WEBHOOK_URL;
});

describe("market-signals-poller-watcher — evaluatePollerStatus", () => {
  it("flags stale when lastPollAt is older than 3x interval", () => {
    const r = evaluatePollerStatus(staleStatus(), NOW);
    assert.equal(r.unhealthy, true);
    assert.equal(r.isStale, true);
    assert.match(r.reason ?? "", /Poller stale/);
    assert.match(r.incidentKey ?? "", /^stale@/);
  });

  it("flags 'never' incident when lastPollAt is null", () => {
    const r = evaluatePollerStatus({
      lastPollAt: null,
      lastPollOk: false,
      lastPollError: null,
      intervalMs: INTERVAL,
    }, NOW);
    assert.equal(r.unhealthy, true);
    assert.equal(r.incidentKey, "stale@never");
  });

  it("flags errored when last poll errored even if recent", () => {
    const r = evaluatePollerStatus(erroredStatus(), NOW);
    assert.equal(r.unhealthy, true);
    assert.equal(r.isStale, false);
    assert.match(r.reason ?? "", /Last poll errored/);
    assert.match(r.incidentKey ?? "", /^error:/);
  });

  it("returns healthy when fresh and ok", () => {
    const r = evaluatePollerStatus(healthyStatus(), NOW);
    assert.equal(r.unhealthy, false);
    assert.equal(r.incidentKey, null);
  });
});

describe("market-signals-poller-watcher — runMarketSignalsWatcherTick", () => {
  it("noop on healthy with no prior incident", async () => {
    let called = 0;
    globalThis.fetch = (async () => {
      called++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;
    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "healthy");
    assert.equal(out.unhealthy, false);
    assert.equal(called, 0);
  });

  it("fires generic webhook on first stale tick, debounces on second", async () => {
    const calls: Array<{ url: string; body: string }> = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      calls.push({ url: String(url), body: String(init?.body ?? "") });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out1 = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out1.status, "alerted");
    assert.equal(out1.unhealthy, true);
    assert.equal(calls.length, 1);
    assert.match(calls[0].url, /\/hook$/);
    assert.match(calls[0].body, /incident/);

    const out2 = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW + 30_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out2.status, "deduped");
    assert.equal(calls.length, 1, "second tick during same outage must not re-fire");
  });

  it("fires recovery alert when poller becomes healthy after an incident", async () => {
    const urls: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      urls.push(String(url) + ":" + String(init?.body ?? ""));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "recovered");
    assert.equal(out.state.activeIncidentKey, null);
    assert.equal(out.state.lastAlertKind, "recovery");
    assert.equal(urls.length, 2);
    assert.ok(urls[1].includes("recovery"));
  });

  it("after recovery, a fresh outage fires another alert", async () => {
    const calls: number[] = [];
    globalThis.fetch = (async () => {
      calls.push(1);
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    // New outage with a different stale bucket (different lastPollAt).
    const out = await runMarketSignalsWatcherTick({
      statusOverride: {
        lastPollAt: NOW + 60_000,
        lastPollOk: true,
        lastPollError: null,
        intervalMs: INTERVAL,
      },
      now: NOW + 60_000 + STALE_THRESHOLD + 30_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "alerted");
    assert.equal(calls.length, 3, "expected incident + recovery + new incident");
  });

  it("debounces a continuous error outage even though lastPollAt advances every tick", async () => {
    // Repro for the regression where the incident key embedded
    // `lastPollAt`. The poller bumps lastPollAt on every poll attempt
    // (failed or not), so a stuck OKX would re-page every minute.
    const calls: string[] = [];
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      calls.push(String(init?.body ?? ""));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const errMsg = "OKX 503 unavailable";
    const tick1 = await runMarketSignalsWatcherTick({
      statusOverride: { lastPollAt: NOW, lastPollOk: false, lastPollError: errMsg, intervalMs: INTERVAL },
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(tick1.status, "alerted");
    assert.equal(calls.length, 1);

    // Same outage, same error, but lastPollAt has advanced (poller
    // tried again, got the same error). Must NOT re-page.
    for (let i = 1; i <= 5; i++) {
      const out = await runMarketSignalsWatcherTick({
        statusOverride: {
          lastPollAt: NOW + i * 60_000,
          lastPollOk: false,
          lastPollError: errMsg,
          intervalMs: INTERVAL,
        },
        now: NOW + i * 60_000,
        webhookUrl: "https://example.invalid/hook",
        slackWebhookUrl: null,
      });
      assert.equal(out.status, "deduped", `tick ${i + 1} must dedupe (got ${out.status})`);
    }
    assert.equal(calls.length, 1, "continuous same-error outage must produce exactly one alert");
  });

  it("a *different* error message during an outage fires a new alert", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runMarketSignalsWatcherTick({
      statusOverride: { lastPollAt: NOW, lastPollOk: false, lastPollError: "OKX 503", intervalMs: INTERVAL },
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(calls, 1);

    const out = await runMarketSignalsWatcherTick({
      statusOverride: {
        lastPollAt: NOW + 60_000,
        lastPollOk: false,
        lastPollError: "OKX 429 rate limited", // different message → new incident
        intervalMs: INTERVAL,
      },
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "alerted");
    assert.equal(calls, 2);
  });

  it("after a recovery, the same error message can re-alert", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runMarketSignalsWatcherTick({
      statusOverride: { lastPollAt: NOW, lastPollOk: false, lastPollError: "OKX 503", intervalMs: INTERVAL },
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus({ lastPollAt: NOW + 60_000 }),
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    const out = await runMarketSignalsWatcherTick({
      statusOverride: { lastPollAt: NOW + 120_000, lastPollOk: false, lastPollError: "OKX 503", intervalMs: INTERVAL },
      now: NOW + 120_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "alerted");
    assert.equal(calls, 3, "expected incident + recovery + new incident with same error message");
  });

  it("dispatches both generic and Slack channels when both configured", async () => {
    const urls: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request) => {
      urls.push(String(url));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: erroredStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/generic",
      slackWebhookUrl: "https://example.invalid/slack",
    });
    assert.equal(out.status, "alerted");
    assert.equal(urls.length, 2);
    assert.ok(urls.some((u) => u.endsWith("/generic")));
    assert.ok(urls.some((u) => u.endsWith("/slack")));
  });

  it("with no channels configured, still tracks state and never POSTs", async () => {
    let called = 0;
    globalThis.fetch = (async () => {
      called++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    assert.equal(out.status, "alerted");
    assert.equal(called, 0);
    assert.equal(out.state.activeIncidentKey?.startsWith("stale@"), true);
    // Persisted, so the next tick is a noop / dedup.
    const persisted = await getMarketSignalsWatcherState();
    assert.equal(persisted.activeIncidentKey, out.state.activeIncidentKey);

    const out2 = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW + 30_000,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    assert.equal(out2.status, "deduped");
  });

  it("isolates channel errors — one channel failing still flips dedup state", async () => {
    globalThis.fetch = (async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes("/slack")) return new Response("oops", { status: 500 });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/generic",
      slackWebhookUrl: "https://example.invalid/slack",
    });
    assert.equal(out.status, "channel_error");
    const generic = out.dispatched.find((d) => d.channel === "generic");
    const slack = out.dispatched.find((d) => d.channel === "slack");
    assert.equal(generic?.outcome, "ok");
    assert.equal(slack?.outcome, "error");

    // Even with the slack failure the dedup state was persisted, so the
    // next tick is a noop — we don't re-page once per minute.
    const out2 = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW + 30_000,
      webhookUrl: "https://example.invalid/generic",
      slackWebhookUrl: "https://example.invalid/slack",
    });
    assert.equal(out2.status, "deduped");
  });
});

describe("market-signals-poller-watcher — per-coin silent streams (#302)", () => {
  const TARGETS = ["btc", "eth", "pepe", "sol", "wld"];

  it("fires one incident webhook per newly-silent coin while poller is healthy", async () => {
    const calls: Array<{ url: string; body: string }> = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      calls.push({ url: String(url), body: String(init?.body ?? "") });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe", "wld"],
    });

    assert.equal(out.silentCoins.silent.length, 2);
    assert.deepEqual(out.silentCoins.silent, ["pepe", "wld"]);
    assert.deepEqual(out.silentCoins.newIncidents, ["pepe", "wld"]);
    assert.equal(calls.length, 2, "one webhook per newly-silent coin");
    assert.ok(calls.every((c) => /coin/.test(c.body)), "payload tagged scope coin");
    assert.ok(calls.some((c) => /pepe/.test(c.body)));
    assert.ok(calls.some((c) => /wld/.test(c.body)));
    assert.equal(out.state.silentCoinIncidents.pepe?.since != null, true);
    assert.equal(out.state.silentCoinIncidents.wld?.since != null, true);
    assert.equal(out.state.lastSilentCoinAlertKind, "incident");
  });

  it("debounces a coin that stays silent across ticks", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe"],
    });
    assert.equal(calls, 1);

    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe"],
    });
    assert.deepEqual(out.silentCoins.deduped, ["pepe"]);
    assert.equal(out.silentCoins.newIncidents.length, 0);
    assert.equal(calls, 1, "still-silent coin must not re-page");
  });

  it("recovers a previously-silent coin without touching other still-silent coins", async () => {
    const calls: string[] = [];
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      calls.push(String(init?.body ?? ""));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    // Tick 1: pepe + wld both go silent.
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe", "wld"],
    });
    assert.equal(calls.length, 2);

    // Tick 2: pepe recovers, wld still silent.
    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW + 120_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["wld"],
    });
    assert.deepEqual(out.silentCoins.recovered, ["pepe"]);
    assert.deepEqual(out.silentCoins.deduped, ["wld"]);
    assert.equal(calls.length, 3, "pepe recovery is one extra POST; wld is deduped");
    assert.ok(calls[2].includes("recovery"));
    assert.ok(calls[2].includes("pepe"));
    // wld sub-incident remains active
    assert.equal(out.state.silentCoinIncidents.wld?.since != null, true);
    assert.equal(out.state.silentCoinIncidents.pepe, undefined);
  });

  it("a fixed coin clearing its sub-incident does not silence other coins' alerts", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    // pepe goes silent
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe"],
    });
    // pepe recovers
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: [],
    });
    // wld now goes silent — must alert (sub-incident keys are per-coin).
    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW + 120_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["wld"],
    });
    assert.deepEqual(out.silentCoins.newIncidents, ["wld"]);
    assert.equal(calls, 3, "incident(pepe) + recovery(pepe) + incident(wld)");
  });

  it("suppresses per-coin alerts while the whole poller is unhealthy", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: erroredStatus("OKX 500"),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe", "wld"], // would alert if not suppressed
    });
    assert.equal(out.silentCoins.suppressedByGlobalIncident, true);
    assert.equal(out.silentCoins.newIncidents.length, 0);
    // Exactly one POST: the global incident, no per-coin.
    assert.equal(calls, 1);
  });

  it("preserves an existing per-coin sub-incident across a global incident & recovery", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    // pepe goes silent (healthy poller)
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe"],
    });
    assert.equal(calls, 1);

    // global poller breaks; per-coin processing must be suppressed and
    // pepe's sub-incident must remain in state (NOT recover spuriously).
    const out1 = await runMarketSignalsWatcherTick({
      statusOverride: erroredStatus("OKX 500"),
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe"],
    });
    assert.equal(out1.silentCoins.suppressedByGlobalIncident, true);
    assert.equal(out1.state.silentCoinIncidents.pepe?.since != null, true);
    assert.equal(calls, 2, "global incident only");

    // global recovers; pepe still silent — must dedupe, not re-fire.
    const out2 = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus({ lastPollAt: NOW + 120_000 }),
      now: NOW + 120_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe"],
    });
    assert.deepEqual(out2.silentCoins.deduped, ["pepe"]);
    assert.equal(out2.silentCoins.newIncidents.length, 0);
    assert.equal(out2.silentCoins.recovered.length, 0);
    // global recovery webhook fired — but no per-coin re-page.
    assert.equal(calls, 3);
  });

  it("filters silent coin overrides to known poller targets", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe", "unknown-retired-coin"],
    });
    assert.deepEqual(out.silentCoins.silent, ["pepe"]);
    assert.equal(calls, 1);
  });

  it("when no channels are configured, still tracks silent-coin sub-incidents in state", async () => {
    let called = 0;
    globalThis.fetch = (async () => {
      called++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: null,
      slackWebhookUrl: null,
      pollerTargetsOverride: TARGETS,
      silentCoinsOverride: ["pepe"],
    });
    assert.equal(called, 0);
    assert.deepEqual(out.silentCoins.newIncidents, ["pepe"]);
    assert.equal(out.state.silentCoinIncidents.pepe?.since != null, true);
  });
});

describe("market-signals-poller-watcher — channel/env wiring", () => {
  it("resolveChannels picks env vars by default", () => {
    process.env.MARKET_SIGNALS_ALERT_WEBHOOK_URL = "https://x.invalid/g";
    process.env.MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL = "https://x.invalid/s";
    const c = resolveChannels();
    assert.equal(c.webhookUrl, "https://x.invalid/g");
    assert.equal(c.slackWebhookUrl, "https://x.invalid/s");
  });

  it("falls back to SLACK_WEBHOOK_URL when the dedicated var is unset", () => {
    delete process.env.MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL;
    process.env.SLACK_WEBHOOK_URL = "https://x.invalid/fallback";
    const c = resolveChannels();
    assert.equal(c.slackWebhookUrl, "https://x.invalid/fallback");
  });

  it("getMarketSignalsAlertChannels reports both flags", () => {
    process.env.MARKET_SIGNALS_ALERT_WEBHOOK_URL = "https://x.invalid/g";
    delete process.env.MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL;
    delete process.env.SLACK_WEBHOOK_URL;
    const c = getMarketSignalsAlertChannels();
    assert.equal(c.genericConfigured, true);
    assert.equal(c.slackConfigured, false);
  });
});

describe("market-signals-poller-watcher — Task #301 snooze", () => {
  it("computeSnoozeUntil 15m / 1h are exact offsets", () => {
    assert.equal(computeSnoozeUntil("15m", NOW).getTime(), NOW + 15 * 60_000);
    assert.equal(computeSnoozeUntil("1h", NOW).getTime(), NOW + 60 * 60_000);
  });

  it("computeSnoozeUntil 'until_midnight' lands on the next local midnight", () => {
    const out = computeSnoozeUntil("until_midnight", NOW);
    assert.equal(out.getHours(), 0);
    assert.equal(out.getMinutes(), 0);
    assert.equal(out.getSeconds(), 0);
    assert.ok(out.getTime() > NOW);
    assert.ok(out.getTime() - NOW <= 24 * 60 * 60_000);
  });

  it("setMarketSignalsWatcherSnooze persists; expired snoozes read as null", async () => {
    await setMarketSignalsWatcherSnooze("15m", NOW);
    const live = await getMarketSignalsWatcherSnooze(NOW + 60_000);
    assert.ok(live);
    assert.equal(live!.duration, "15m");
    const expired = await getMarketSignalsWatcherSnooze(NOW + 30 * 60_000);
    assert.equal(expired, null);
  });

  it("clearMarketSignalsWatcherSnooze removes the snooze immediately", async () => {
    await setMarketSignalsWatcherSnooze("1h", NOW);
    assert.ok(await getMarketSignalsWatcherSnooze(NOW + 1000));
    await clearMarketSignalsWatcherSnooze();
    assert.equal(await getMarketSignalsWatcherSnooze(NOW + 1000), null);
  });

  it("snoozed unhealthy tick suppresses dispatch but tracks pending incident", async () => {
    let called = 0;
    globalThis.fetch = (async () => {
      called++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      snoozeOverride: {
        snoozedAt: new Date(NOW - 60_000).toISOString(),
        snoozedUntil: new Date(NOW + 30 * 60_000).toISOString(),
        duration: "1h",
      },
    });
    assert.equal(out.status, "snoozed");
    assert.equal(out.unhealthy, true);
    assert.equal(called, 0, "must not dispatch while snoozed");
    assert.equal(out.state.activeIncidentKey, null, "dispatch dedupe key must NOT be promoted under snooze");
    assert.ok(out.state.pendingIncidentKey?.startsWith("stale@"));
    assert.ok(out.state.pendingIncidentSince);
  });

  it("after snooze expires, an unhealthy poller fires the alert (not deduped)", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    // Tick 1: snoozed + stale → no dispatch, pending tracked.
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      snoozeOverride: {
        snoozedAt: new Date(NOW - 60_000).toISOString(),
        snoozedUntil: new Date(NOW + 60_000).toISOString(),
        duration: "15m",
      },
    });
    assert.equal(calls, 0);

    // Tick 2: snooze gone, still stale → MUST alert.
    const out = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW + 120_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      snoozeOverride: null,
    });
    assert.equal(out.status, "alerted");
    assert.equal(calls, 1, "post-snooze still-stale must page exactly once");
    // Pending bookkeeping cleared after snooze ends.
    assert.equal(out.state.pendingIncidentKey, null);
  });

  it("snoozed + healthy after a prior incident clears state silently (no recovery ping)", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    // Pre-seed: incident was active before snooze.
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      snoozeOverride: null,
    });
    assert.equal(calls, 1);

    // Now operator snoozes and poller recovers — must not page recovery.
    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus({ lastPollAt: NOW + 60_000 }),
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      snoozeOverride: {
        snoozedAt: new Date(NOW + 30_000).toISOString(),
        snoozedUntil: new Date(NOW + 30 * 60_000).toISOString(),
        duration: "1h",
      },
    });
    assert.equal(out.status, "snoozed_recovered");
    assert.equal(calls, 1, "no recovery dispatch under snooze");
    assert.equal(out.state.activeIncidentKey, null);
  });

  it("snoozed + healthy with no prior incident reports snoozed_healthy", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      snoozeOverride: {
        snoozedAt: new Date(NOW - 60_000).toISOString(),
        snoozedUntil: new Date(NOW + 60_000).toISOString(),
        duration: "15m",
      },
    });
    assert.equal(out.status, "snoozed_healthy");
    assert.equal(calls, 0);
  });

  it("repeated snoozed unhealthy ticks keep pendingIncidentSince stable for the same key", async () => {
    globalThis.fetch = (async () => new Response("ok", { status: 200 })) as typeof fetch;

    const snooze = {
      snoozedAt: new Date(NOW - 60_000).toISOString(),
      snoozedUntil: new Date(NOW + 30 * 60_000).toISOString(),
      duration: "1h" as const,
    };
    const t1 = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      snoozeOverride: snooze,
    });
    const t2 = await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW + 30_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
      snoozeOverride: snooze,
    });
    assert.equal(t1.state.pendingIncidentKey, t2.state.pendingIncidentKey);
    assert.equal(t1.state.pendingIncidentSince, t2.state.pendingIncidentSince);
  });
});

describe("market-signals-poller-watcher — alert history (Task #303)", () => {
  before(() => {
    globalThis.fetch = (async () => new Response("ok", { status: 200 })) as typeof fetch;
  });

  it("starts empty", async () => {
    const h = await getMarketSignalsWatcherHistory();
    assert.deepEqual(h, []);
  });

  it("appends an incident row when the watcher first alerts", async () => {
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    const h = await getMarketSignalsWatcherHistory();
    assert.equal(h.length, 1);
    assert.equal(h[0].kind, "incident");
    assert.match(h[0].reason, /Poller stale/);
    assert.equal(h[0].at, new Date(NOW).toISOString());
    assert.ok(h[0].incidentKey?.startsWith("stale@"));
  });

  it("does not append on a debounced tick (same outage)", async () => {
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW + 30_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    const h = await getMarketSignalsWatcherHistory();
    assert.equal(h.length, 1, "deduped tick must not append a row");
  });

  it("appends a recovery row, newest first", async () => {
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus(),
      now: NOW + 60_000,
      webhookUrl: "https://example.invalid/hook",
      slackWebhookUrl: null,
    });
    const h = await getMarketSignalsWatcherHistory();
    assert.equal(h.length, 2);
    assert.equal(h[0].kind, "recovery", "newest first");
    assert.equal(h[1].kind, "incident");
    assert.match(h[0].reason, /Recovered from/);
  });

  it("captures a flap as multiple rows so operators can spot it", async () => {
    // incident → recovery → fresh incident → recovery
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus({ lastPollAt: NOW + 60_000 }),
      now: NOW + 60_000,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    await runMarketSignalsWatcherTick({
      statusOverride: {
        lastPollAt: NOW + 60_000,
        lastPollOk: false,
        lastPollError: "OKX 503",
        intervalMs: INTERVAL,
      },
      now: NOW + 120_000,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus({ lastPollAt: NOW + 180_000 }),
      now: NOW + 180_000,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    const h = await getMarketSignalsWatcherHistory();
    assert.equal(h.length, 4, "flap should produce 4 rows");
    assert.deepEqual(
      h.map((r) => r.kind),
      ["recovery", "incident", "recovery", "incident"],
    );
  });

  it("respects the max cap (drops oldest)", async () => {
    // Seed history above the cap directly to avoid running ~50 ticks.
    const seeded = Array.from({ length: MARKET_SIGNALS_WATCHER_HISTORY_MAX }, (_, i) => ({
      at: new Date(NOW - (i + 1) * 60_000).toISOString(),
      kind: i % 2 === 0 ? "incident" : "recovery",
      reason: `seed ${i}`,
      incidentKey: `seed-${i}`,
      incidentSince: new Date(NOW - (i + 1) * 60_000).toISOString(),
    }));
    await db
      .insert(appSettingsTable)
      .values({
        key: MARKET_SIGNALS_WATCHER_HISTORY_KEY,
        value: { entries: seeded },
      });

    // One real alert should push the oldest off the end.
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    const h = await getMarketSignalsWatcherHistory();
    assert.equal(h.length, MARKET_SIGNALS_WATCHER_HISTORY_MAX);
    assert.equal(h[0].kind, "incident");
    assert.match(h[0].reason, /Poller stale/);
    // Oldest seed (index 49) is gone; the second-oldest is now the last.
    assert.equal(h[h.length - 1].reason, "seed 48");
  });

  it("getMarketSignalsWatcherHistory(limit) trims results", async () => {
    await runMarketSignalsWatcherTick({
      statusOverride: staleStatus(),
      now: NOW,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    await runMarketSignalsWatcherTick({
      statusOverride: healthyStatus({ lastPollAt: NOW + 60_000 }),
      now: NOW + 60_000,
      webhookUrl: null,
      slackWebhookUrl: null,
    });
    const trimmed = await getMarketSignalsWatcherHistory(1);
    assert.equal(trimmed.length, 1);
    assert.equal(trimmed[0].kind, "recovery");
  });
});
