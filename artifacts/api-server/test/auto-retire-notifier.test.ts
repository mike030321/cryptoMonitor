import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import {
  AUTO_RETIRE_ALERTS_SENT_KEY,
  buildAlertPayload,
  dispatchAutoRetireNotifications,
  formatAlertBody,
  formatAlertTitle,
  pickWorstTimeframe,
} from "../src/lib/auto-retire-notifier.ts";
import {
  QUARANTINED_FEATURES_SETTING_KEY,
  type QuarantinedFeatureRecord,
} from "../src/lib/feature-lab.ts";

const sampleRecord = (
  name: string,
  quarantinedAt = "2026-04-22T10:00:00Z",
): QuarantinedFeatureRecord => ({
  name,
  transformKind: "passthrough_existing",
  sourceColumn: "rsi_14",
  quarantinedAt,
  reason: "validation_regression",
  detail: {
    trigger: "auto_retire_after_training",
    threshold: 0.05,
    timeframes: [
      { timeframe: "1h", current_log_loss: 0.62, prior_log_loss: 0.55, delta_log_loss: 0.07 },
      { timeframe: "4h", current_log_loss: 0.58, prior_log_loss: 0.50, delta_log_loss: 0.08 },
    ],
  },
});

let savedFetch: typeof fetch;
let savedSlackEnv: string | undefined;
let savedSlackEnv2: string | undefined;
let savedEmailEnv: string | undefined;

before(() => {
  savedFetch = globalThis.fetch;
  savedSlackEnv = process.env.SLACK_WEBHOOK_URL;
  savedSlackEnv2 = process.env.AUTO_RETIRE_SLACK_WEBHOOK_URL;
  savedEmailEnv = process.env.AUTO_RETIRE_EMAIL_WEBHOOK_URL;
});

after(async () => {
  globalThis.fetch = savedFetch;
  if (savedSlackEnv === undefined) delete process.env.SLACK_WEBHOOK_URL;
  else process.env.SLACK_WEBHOOK_URL = savedSlackEnv;
  if (savedSlackEnv2 === undefined) delete process.env.AUTO_RETIRE_SLACK_WEBHOOK_URL;
  else process.env.AUTO_RETIRE_SLACK_WEBHOOK_URL = savedSlackEnv2;
  if (savedEmailEnv === undefined) delete process.env.AUTO_RETIRE_EMAIL_WEBHOOK_URL;
  else process.env.AUTO_RETIRE_EMAIL_WEBHOOK_URL = savedEmailEnv;
  // Clean up dedup row written during tests so we don't leak state.
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, AUTO_RETIRE_ALERTS_SENT_KEY));
});

beforeEach(async () => {
  // Reset dedup state between tests.
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, AUTO_RETIRE_ALERTS_SENT_KEY));
});

describe("auto-retire-notifier — pure helpers", () => {
  it("pickWorstTimeframe returns the largest delta_log_loss", () => {
    const r = sampleRecord("feat_a");
    const tfs = (r.detail as { timeframes?: unknown }).timeframes as Array<{
      timeframe: string;
      delta_log_loss: number;
    }>;
    const worst = pickWorstTimeframe(tfs);
    assert.equal(worst?.timeframe, "4h");
  });

  it("pickWorstTimeframe handles empty / null safely", () => {
    assert.equal(pickWorstTimeframe([]), null);
    assert.equal(pickWorstTimeframe(null), null);
    assert.equal(pickWorstTimeframe(undefined), null);
  });

  it("buildAlertPayload extracts worst timeframe + threshold + delta", () => {
    const p = buildAlertPayload(sampleRecord("feat_a"));
    assert.equal(p.name, "feat_a");
    assert.equal(p.worstTimeframe, "4h");
    assert.equal(p.deltaLogLoss, 0.08);
    assert.equal(p.threshold, 0.05);
    assert.equal(p.reason, "validation_regression");
  });

  it("formatAlertTitle / Body match the in-app toast format", () => {
    const p = buildAlertPayload(sampleRecord("feat_a"));
    assert.equal(formatAlertTitle(p), "Feature auto-retired: feat_a");
    assert.equal(
      formatAlertBody(p),
      "Worst timeframe 4h · Δlog_loss +0.0800 · reason validation_regression",
    );
  });
});

describe("auto-retire-notifier — dispatch", () => {
  it("noop when no new entries", async () => {
    const out = await dispatchAutoRetireNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
      quarantinedOverride: [],
      skipPersist: true,
    });
    assert.equal(out.status, "noop");
    assert.equal(out.newAlerts, 0);
  });

  it("dispatches Slack only when slack URL configured, marks dedup", async () => {
    const calls: Array<{ url: string; body: string }> = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      calls.push({ url: String(url), body: String(init?.body ?? "") });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await dispatchAutoRetireNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
      quarantinedOverride: [sampleRecord("feat_a"), sampleRecord("feat_b")],
    });
    assert.equal(out.status, "dispatched");
    assert.equal(out.newAlerts, 2);
    assert.equal(calls.length, 2);
    assert.ok(calls[0].url.includes("/slack"));
    assert.ok(calls[0].body.includes("feat_a"));
    for (const r of out.sent) {
      assert.equal(r.slack, "ok");
      assert.equal(r.email, "skipped");
    }

    // Second pass with the same records is a noop (dedup persisted).
    const out2 = await dispatchAutoRetireNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
      quarantinedOverride: [sampleRecord("feat_a"), sampleRecord("feat_b")],
    });
    assert.equal(out2.status, "noop");
    assert.equal(out2.newAlerts, 0);
  });

  it("dispatches both Slack and email when both configured", async () => {
    const urls: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request) => {
      urls.push(String(url));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await dispatchAutoRetireNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: "https://example.invalid/email",
      quarantinedOverride: [sampleRecord("feat_dual")],
    });
    assert.equal(out.newAlerts, 1);
    assert.equal(urls.length, 2);
    assert.ok(urls.some((u) => u.includes("/slack")));
    assert.ok(urls.some((u) => u.includes("/email")));
    assert.equal(out.sent[0].slack, "ok");
    assert.equal(out.sent[0].email, "ok");
  });

  it("isolates dispatch errors (slack fails, email succeeds, dedup still records)", async () => {
    globalThis.fetch = (async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes("/slack")) return new Response("oops", { status: 500 });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await dispatchAutoRetireNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: "https://example.invalid/email",
      quarantinedOverride: [sampleRecord("feat_err")],
    });
    assert.equal(out.newAlerts, 1);
    assert.equal(out.sent[0].slack, "error");
    assert.equal(out.sent[0].email, "ok");

    // Dedup still flipped — second pass must be noop, not an infinite retry.
    const out2 = await dispatchAutoRetireNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: "https://example.invalid/email",
      quarantinedOverride: [sampleRecord("feat_err")],
    });
    assert.equal(out2.status, "noop");
  });

  it("with no channels configured, still updates dedup so future configs don't blast history", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await dispatchAutoRetireNotifications({
      slackWebhookUrl: null,
      emailWebhookUrl: null,
      quarantinedOverride: [sampleRecord("feat_quiet")],
    });
    assert.equal(out.status, "dispatched");
    assert.equal(out.newAlerts, 1);
    assert.equal(calls, 0);
    assert.equal(out.sent[0].slack, "skipped");
    assert.equal(out.sent[0].email, "skipped");

    // Persisted as already-sent.
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, AUTO_RETIRE_ALERTS_SENT_KEY));
    assert.ok(row);
    const v = row.value as { keys: string[] };
    assert.ok(v.keys.includes("feat_quiet@2026-04-22T10:00:00Z"));
  });

  it("gracefully handles a missing quarantined bucket (DB read returns empty list)", async () => {
    // Ensure bucket not set
    await db
      .delete(appSettingsTable)
      .where(eq(appSettingsTable.key, QUARANTINED_FEATURES_SETTING_KEY));
    const out = await dispatchAutoRetireNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
    });
    // Either noop (empty bucket) or skipped with reason.
    assert.ok(out.status === "noop" || out.status === "skipped");
    assert.equal(out.newAlerts, 0);
  });
});
