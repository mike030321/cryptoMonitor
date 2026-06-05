import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import {
  MTTM_DISABLE_ALERTS_SENT_KEY,
  buildAlertPayload,
  formatAlertBody,
  formatAlertTitle,
  notifyMttmAutoDisabled,
} from "../src/lib/mttm-disable-notifier.ts";
import type { MttmDisableReason } from "../src/lib/mttm.ts";

const consecutiveReason = (
  trippedAt = "2026-04-29T12:00:00.000Z",
): MttmDisableReason => ({
  reason: "consecutive_losses",
  detail: "MTTM auto-disabled — 5 consecutive MTTM losses (cap = 5).",
  trippedAt,
  consecutiveLosses: 5,
  nTrades: 7,
});

const n10Reason = (
  trippedAt = "2026-04-29T13:00:00.000Z",
): MttmDisableReason => ({
  reason: "n10_post_fee",
  detail:
    "MTTM auto-disabled — 12 trades, post-fee PnL -3.40% < cap -2.00%.",
  trippedAt,
  consecutiveLosses: 2,
  nTrades: 12,
  postFeePnlPct: -0.034,
});

let savedFetch: typeof fetch;
let savedSlackEnv: string | undefined;
let savedSlackEnv2: string | undefined;
let savedGenericEnv: string | undefined;

before(() => {
  savedFetch = globalThis.fetch;
  savedSlackEnv = process.env.SLACK_WEBHOOK_URL;
  savedSlackEnv2 = process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL;
  savedGenericEnv = process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL;
  // Wipe envs so tests only use injected URLs.
  delete process.env.SLACK_WEBHOOK_URL;
  delete process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL;
  delete process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL;
});

after(async () => {
  globalThis.fetch = savedFetch;
  if (savedSlackEnv === undefined) delete process.env.SLACK_WEBHOOK_URL;
  else process.env.SLACK_WEBHOOK_URL = savedSlackEnv;
  if (savedSlackEnv2 === undefined)
    delete process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL;
  else process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL = savedSlackEnv2;
  if (savedGenericEnv === undefined)
    delete process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL;
  else process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL = savedGenericEnv;
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MTTM_DISABLE_ALERTS_SENT_KEY));
});

beforeEach(async () => {
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MTTM_DISABLE_ALERTS_SENT_KEY));
});

describe("mttm-disable-notifier — pure helpers", () => {
  it("buildAlertPayload preserves all required fields for consecutive_losses", () => {
    const p = buildAlertPayload(consecutiveReason());
    assert.equal(p.reasonKind, "consecutive_losses");
    assert.equal(p.consecutiveLosses, 5);
    assert.equal(p.nTrades, 7);
    assert.equal(p.postFeePnlPct, null);
    assert.equal(p.trippedAt, "2026-04-29T12:00:00.000Z");
  });

  it("buildAlertPayload preserves all required fields for n10_post_fee", () => {
    const p = buildAlertPayload(n10Reason());
    assert.equal(p.reasonKind, "n10_post_fee");
    assert.equal(p.nTrades, 12);
    assert.equal(p.postFeePnlPct, -0.034);
  });

  it("formatAlertTitle distinguishes the two trip kinds", () => {
    assert.equal(
      formatAlertTitle(buildAlertPayload(consecutiveReason())),
      "MTTM auto-disabled — consecutive losses",
    );
    assert.equal(
      formatAlertTitle(buildAlertPayload(n10Reason())),
      "MTTM auto-disabled — post-fee PnL floor breached",
    );
  });

  it("formatAlertBody includes reason kind, consecutive losses, total trades, and post-fee PnL%", () => {
    const body = formatAlertBody(buildAlertPayload(n10Reason()));
    // Spec requirement: alert must surface all four.
    assert.ok(body.includes("post-fee PnL"));
    assert.ok(body.includes("-3.40%"));
    assert.ok(body.includes("total trades=12"));
    assert.ok(body.includes("consecutive losses=2"));
  });
});

describe("mttm-disable-notifier — dispatch", () => {
  it("fires both webhooks on first call and dedups on a second call with the same trippedAt", async () => {
    const calls: Array<{ url: string; body: string }> = [];
    globalThis.fetch = (async (
      url: string | URL | Request,
      init?: RequestInit,
    ) => {
      calls.push({ url: String(url), body: String(init?.body ?? "") });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const reason = consecutiveReason("2026-04-29T14:00:00.000Z");
    const out = await notifyMttmAutoDisabled(reason, {
      webhookUrl: "https://example.invalid/generic",
      slackWebhookUrl: "https://example.invalid/slack",
    });
    assert.equal(out.status, "dispatched");
    assert.equal(out.generic, "ok");
    assert.equal(out.slack, "ok");
    assert.equal(calls.length, 2);
    // Body must contain the four required fields somewhere.
    const allBody = calls.map((c) => c.body).join("\n");
    assert.ok(allBody.includes("consecutive_losses"));
    assert.ok(allBody.includes("\"consecutiveLosses\":5"));
    assert.ok(allBody.includes("\"nTrades\":7"));

    // Second call with same trippedAt is a noop.
    const out2 = await notifyMttmAutoDisabled(reason, {
      webhookUrl: "https://example.invalid/generic",
      slackWebhookUrl: "https://example.invalid/slack",
    });
    assert.equal(out2.status, "noop");
    assert.equal(calls.length, 2, "no extra dispatch on second call");
  });

  it("a fresh trippedAt re-fires after a previous trip", async () => {
    let count = 0;
    globalThis.fetch = (async () => {
      count++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    await notifyMttmAutoDisabled(consecutiveReason("2026-04-29T15:00:00.000Z"), {
      webhookUrl: "https://example.invalid/generic",
      slackWebhookUrl: null,
    });
    assert.equal(count, 1);
    // Second auto-disable later (different trippedAt) — must page again.
    const out = await notifyMttmAutoDisabled(
      consecutiveReason("2026-04-29T16:30:00.000Z"),
      {
        webhookUrl: "https://example.invalid/generic",
        slackWebhookUrl: null,
      },
    );
    assert.equal(out.status, "dispatched");
    assert.equal(count, 2);
  });

  it("manual disables are skipped (no webhook fires, no dedup write)", async () => {
    let count = 0;
    globalThis.fetch = (async () => {
      count++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await notifyMttmAutoDisabled(
      {
        reason: "manual",
        detail: "operator flipped switch",
        trippedAt: "2026-04-29T17:00:00.000Z",
      },
      {
        webhookUrl: "https://example.invalid/generic",
        slackWebhookUrl: "https://example.invalid/slack",
      },
    );
    assert.equal(out.status, "skipped");
    assert.equal(out.reason, "manual_disable");
    assert.equal(count, 0);

    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, MTTM_DISABLE_ALERTS_SENT_KEY));
    assert.equal(row, undefined, "no dedup row should be written for manual");
  });

  it("with no channels configured, status is 'no_channel' but dedup is still recorded", async () => {
    let count = 0;
    globalThis.fetch = (async () => {
      count++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await notifyMttmAutoDisabled(
      consecutiveReason("2026-04-29T18:00:00.000Z"),
      { webhookUrl: null, slackWebhookUrl: null },
    );
    assert.equal(out.status, "no_channel");
    assert.equal(out.generic, "skipped");
    assert.equal(out.slack, "skipped");
    assert.equal(count, 0);

    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, MTTM_DISABLE_ALERTS_SENT_KEY));
    assert.ok(row, "dedup row should still be written");
    const v = row.value as { keys: string[] };
    assert.ok(v.keys.includes("mttm_disable@2026-04-29T18:00:00.000Z"));
  });

  it("partial failure (Slack 500, generic ok) reports 'dispatched' and records dedup", async () => {
    globalThis.fetch = (async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes("/slack")) return new Response("oops", { status: 500 });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await notifyMttmAutoDisabled(
      consecutiveReason("2026-04-29T19:00:00.000Z"),
      {
        webhookUrl: "https://example.invalid/generic",
        slackWebhookUrl: "https://example.invalid/slack",
      },
    );
    assert.equal(out.status, "dispatched");
    assert.equal(out.generic, "ok");
    assert.equal(out.slack, "error");
    assert.ok(out.error && out.error.includes("HTTP 500"));

    // Second call with same reason must be a noop, not a retry.
    const out2 = await notifyMttmAutoDisabled(
      consecutiveReason("2026-04-29T19:00:00.000Z"),
      {
        webhookUrl: "https://example.invalid/generic",
        slackWebhookUrl: "https://example.invalid/slack",
      },
    );
    assert.equal(out2.status, "noop");
  });

  it("total failure (every configured channel errors) reports 'failed' but still records dedup", async () => {
    globalThis.fetch = (async () => {
      return new Response("oops", { status: 500 });
    }) as typeof fetch;

    const out = await notifyMttmAutoDisabled(
      consecutiveReason("2026-04-29T19:30:00.000Z"),
      {
        webhookUrl: "https://example.invalid/generic",
        slackWebhookUrl: "https://example.invalid/slack",
      },
    );
    assert.equal(out.status, "failed");
    assert.equal(out.generic, "error");
    assert.equal(out.slack, "error");
    assert.ok(out.error && out.error.includes("HTTP 500"));

    // Subsequent call still dedups — no retry storm.
    const out2 = await notifyMttmAutoDisabled(
      consecutiveReason("2026-04-29T19:30:00.000Z"),
      {
        webhookUrl: "https://example.invalid/generic",
        slackWebhookUrl: "https://example.invalid/slack",
      },
    );
    assert.equal(out2.status, "noop");
  });

  it("env-driven Slack webhook (with SLACK_WEBHOOK_URL fallback) is honoured", async () => {
    const calls: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request) => {
      calls.push(String(url));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    process.env.SLACK_WEBHOOK_URL = "https://example.invalid/slack-fallback";
    try {
      const out = await notifyMttmAutoDisabled(
        consecutiveReason("2026-04-29T20:00:00.000Z"),
      );
      assert.equal(out.status, "dispatched");
      assert.equal(out.slack, "ok");
      assert.ok(calls.some((u) => u.includes("/slack-fallback")));
    } finally {
      delete process.env.SLACK_WEBHOOK_URL;
    }
  });
});
