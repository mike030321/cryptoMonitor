import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import {
  MlEngineUnreachableError,
  TRAINING_CONTRACT_ALERTS_SENT_KEY,
  UNREACHABLE_ALERT_EVERY_N,
  _getUnreachableCounterForTests,
  _resetUnreachableCounterForTests,
  buildAlertPayload,
  dispatchTrainingContractNotifications,
  findFailedTimeframes,
  formatAlertBody,
  formatAlertTitle,
  type TrainingReport,
} from "../src/lib/training-contract-notifier.ts";
import { logger } from "../src/lib/logger.ts";

const failingReport = (
  generatedAt = "2026-04-22T10:00:00Z",
): TrainingReport => ({
  status: "ok",
  generated_at: generatedAt,
  timeframes: {
    "1h": {
      leakage_audit: { passed: true },
      provenance: { rejected_synthetic: false, coins_rejected: [] },
    },
    "4h": {
      leakage_audit: { passed: false, violations: [{ kind: "future_peek" }] },
      provenance: { rejected_synthetic: false, coins_rejected: [] },
    },
    "1d": {
      leakage_audit: { passed: true },
      provenance: {
        rejected_synthetic: true,
        coins_rejected: ["BTC", "ETH"],
      },
    },
  },
});

const passingReport = (
  generatedAt = "2026-04-22T11:00:00Z",
): TrainingReport => ({
  status: "ok",
  generated_at: generatedAt,
  timeframes: {
    "1h": {
      leakage_audit: { passed: true },
      provenance: { rejected_synthetic: false, coins_rejected: [] },
    },
  },
});

let savedFetch: typeof fetch;
let savedSlackEnv: string | undefined;
let savedSlackEnv2: string | undefined;
let savedEmailEnv: string | undefined;

before(() => {
  savedFetch = globalThis.fetch;
  savedSlackEnv = process.env.SLACK_WEBHOOK_URL;
  savedSlackEnv2 = process.env.TRAINING_CONTRACT_SLACK_WEBHOOK_URL;
  savedEmailEnv = process.env.TRAINING_CONTRACT_EMAIL_WEBHOOK_URL;
});

after(async () => {
  globalThis.fetch = savedFetch;
  if (savedSlackEnv === undefined) delete process.env.SLACK_WEBHOOK_URL;
  else process.env.SLACK_WEBHOOK_URL = savedSlackEnv;
  if (savedSlackEnv2 === undefined) delete process.env.TRAINING_CONTRACT_SLACK_WEBHOOK_URL;
  else process.env.TRAINING_CONTRACT_SLACK_WEBHOOK_URL = savedSlackEnv2;
  if (savedEmailEnv === undefined) delete process.env.TRAINING_CONTRACT_EMAIL_WEBHOOK_URL;
  else process.env.TRAINING_CONTRACT_EMAIL_WEBHOOK_URL = savedEmailEnv;
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, TRAINING_CONTRACT_ALERTS_SENT_KEY));
});

beforeEach(async () => {
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, TRAINING_CONTRACT_ALERTS_SENT_KEY));
});

describe("training-contract-notifier — pure helpers", () => {
  it("findFailedTimeframes flags leakage and provenance failures", () => {
    const failed = findFailedTimeframes(failingReport());
    const tfs = failed.map((f) => f.timeframe).sort();
    assert.deepEqual(tfs, ["1d", "4h"]);
    const fourH = failed.find((f) => f.timeframe === "4h")!;
    assert.equal(fourH.leakageFailed, true);
    assert.equal(fourH.provenanceRejected, false);
    const oneD = failed.find((f) => f.timeframe === "1d")!;
    assert.equal(oneD.provenanceRejected, true);
    assert.deepEqual(oneD.coinsRejected, ["BTC", "ETH"]);
  });

  it("findFailedTimeframes returns [] for a passing report", () => {
    assert.deepEqual(findFailedTimeframes(passingReport()), []);
    assert.deepEqual(findFailedTimeframes(null), []);
    assert.deepEqual(findFailedTimeframes(undefined), []);
    assert.deepEqual(findFailedTimeframes({}), []);
  });

  it("buildAlertPayload aggregates rejected coin counts", () => {
    const r = failingReport();
    const p = buildAlertPayload(r, findFailedTimeframes(r));
    assert.equal(p.generatedAt, "2026-04-22T10:00:00Z");
    assert.equal(p.failedTimeframes.length, 2);
    assert.equal(p.totalRejectedCoins, 2);
  });

  it("formatAlertTitle / Body summarise the failure", () => {
    const r = failingReport();
    const p = buildAlertPayload(r, findFailedTimeframes(r));
    assert.equal(formatAlertTitle(p), "Training contract failed: 2 timeframe(s)");
    const body = formatAlertBody(p);
    assert.ok(body.includes("Run: 2026-04-22T10:00:00Z"));
    assert.ok(body.includes("4h: leakage"));
    assert.ok(body.includes("1d: synthetic-rejected"));
    assert.ok(body.includes("(2 coins rejected)"));
    assert.ok(body.includes("Total rejected coins"));
  });
});

describe("training-contract-notifier — dispatch", () => {
  it("noop when report is missing or has no failures", async () => {
    const out1 = await dispatchTrainingContractNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
      reportOverride: null,
      skipPersist: true,
    });
    assert.equal(out1.status, "noop");

    const out2 = await dispatchTrainingContractNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
      reportOverride: passingReport(),
      skipPersist: true,
    });
    assert.equal(out2.status, "noop");
    assert.equal(out2.reason, "contract_passed");
  });

  it("dispatches Slack only when slack URL configured, dedups by generated_at", async () => {
    const calls: Array<{ url: string; body: string }> = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      calls.push({ url: String(url), body: String(init?.body ?? "") });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await dispatchTrainingContractNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
      reportOverride: failingReport(),
    });
    assert.equal(out.status, "dispatched");
    assert.equal(calls.length, 1);
    assert.ok(calls[0].url.includes("/slack"));
    assert.ok(calls[0].body.includes("Training contract failed"));
    assert.equal(out.sent?.slack, "ok");
    assert.equal(out.sent?.email, "skipped");

    // Second pass with the same generated_at is a noop.
    const out2 = await dispatchTrainingContractNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
      reportOverride: failingReport(),
    });
    assert.equal(out2.status, "noop");
    assert.equal(out2.reason, "already_sent");

    // A new run with a different generated_at fires a fresh alert.
    const out3 = await dispatchTrainingContractNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: null,
      reportOverride: failingReport("2026-04-22T12:00:00Z"),
    });
    assert.equal(out3.status, "dispatched");
  });

  it("dispatches both Slack and email when both configured", async () => {
    const urls: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request) => {
      urls.push(String(url));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await dispatchTrainingContractNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: "https://example.invalid/email",
      reportOverride: failingReport(),
    });
    assert.equal(out.status, "dispatched");
    assert.equal(urls.length, 2);
    assert.ok(urls.some((u) => u.includes("/slack")));
    assert.ok(urls.some((u) => u.includes("/email")));
    assert.equal(out.sent?.slack, "ok");
    assert.equal(out.sent?.email, "ok");
  });

  it("isolates dispatch errors and still records dedup", async () => {
    globalThis.fetch = (async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes("/slack")) return new Response("oops", { status: 500 });
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await dispatchTrainingContractNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: "https://example.invalid/email",
      reportOverride: failingReport(),
    });
    assert.equal(out.status, "dispatched");
    assert.equal(out.sent?.slack, "error");
    assert.equal(out.sent?.email, "ok");

    const out2 = await dispatchTrainingContractNotifications({
      slackWebhookUrl: "https://example.invalid/slack",
      emailWebhookUrl: "https://example.invalid/email",
      reportOverride: failingReport(),
    });
    assert.equal(out2.status, "noop");
  });

  interface CapturedWarn {
    obj: Record<string, unknown> | undefined;
    msg: string;
  }

  function spyOnWarn(): {
    calls: CapturedWarn[];
    restore: () => void;
  } {
    const calls: CapturedWarn[] = [];
    const origWarn: typeof logger.warn = logger.warn.bind(logger);
    const spy = ((first: unknown, second?: unknown): void => {
      if (first !== null && typeof first === "object") {
        calls.push({
          obj: first as Record<string, unknown>,
          msg: typeof second === "string" ? second : "",
        });
      } else {
        calls.push({
          obj: undefined,
          msg: typeof first === "string" ? first : String(first ?? ""),
        });
      }
    }) as typeof logger.warn;
    logger.warn = spy;
    return {
      calls,
      restore: () => {
        logger.warn = origWarn;
      },
    };
  }

  function alarmsIn(calls: CapturedWarn[]): CapturedWarn[] {
    return calls.filter(
      (c) => c.obj?.event === "training_contract_notifier_unreachable",
    );
  }

  it("ml-engine unreachable: increments counter and emits structured WARN every Nth cycle", async () => {
    _resetUnreachableCounterForTests();
    const spy = spyOnWarn();

    const failingFetch = async (): Promise<TrainingReport | null> => {
      throw new MlEngineUnreachableError("ml-engine unreachable (test)");
    };

    try {
      // First N-1 cycles: counter increments but no alarm event.
      for (let i = 1; i < UNREACHABLE_ALERT_EVERY_N; i++) {
        const out = await dispatchTrainingContractNotifications({
          slackWebhookUrl: null,
          emailWebhookUrl: null,
          fetchOverride: failingFetch,
          skipPersist: true,
        });
        assert.equal(out.status, "skipped");
        assert.equal(out.reason, "fetch_failed");
        assert.equal(_getUnreachableCounterForTests(), i);
      }
      assert.equal(alarmsIn(spy.calls).length, 0);

      // Nth cycle: alarm fires.
      const outN = await dispatchTrainingContractNotifications({
        slackWebhookUrl: null,
        emailWebhookUrl: null,
        fetchOverride: failingFetch,
        skipPersist: true,
      });
      assert.equal(outN.status, "skipped");
      assert.equal(_getUnreachableCounterForTests(), UNREACHABLE_ALERT_EVERY_N);
      const alarms = alarmsIn(spy.calls);
      assert.equal(alarms.length, 1);
      assert.equal(
        alarms[0].obj?.consecutiveFailures,
        UNREACHABLE_ALERT_EVERY_N,
      );
      assert.equal(alarms[0].obj?.everyN, UNREACHABLE_ALERT_EVERY_N);

      // Another N-1 cycles: still no new alarm. Then 2N triggers a second.
      for (let i = 1; i < UNREACHABLE_ALERT_EVERY_N; i++) {
        await dispatchTrainingContractNotifications({
          slackWebhookUrl: null,
          emailWebhookUrl: null,
          fetchOverride: failingFetch,
          skipPersist: true,
        });
      }
      assert.equal(alarmsIn(spy.calls).length, 1);
      await dispatchTrainingContractNotifications({
        slackWebhookUrl: null,
        emailWebhookUrl: null,
        fetchOverride: failingFetch,
        skipPersist: true,
      });
      const alarms2 = alarmsIn(spy.calls);
      assert.equal(alarms2.length, 2);
      assert.equal(
        alarms2[1].obj?.consecutiveFailures,
        UNREACHABLE_ALERT_EVERY_N * 2,
      );
    } finally {
      spy.restore();
      _resetUnreachableCounterForTests();
    }
  });

  it("ml-engine unreachable: counter resets to 0 on first successful fetch", async () => {
    _resetUnreachableCounterForTests();
    const failingFetch = async (): Promise<TrainingReport | null> => {
      throw new MlEngineUnreachableError("ml-engine unreachable (test)");
    };

    for (let i = 0; i < 3; i++) {
      await dispatchTrainingContractNotifications({
        slackWebhookUrl: null,
        emailWebhookUrl: null,
        fetchOverride: failingFetch,
        skipPersist: true,
      });
    }
    assert.equal(_getUnreachableCounterForTests(), 3);

    // A successful fetch (passing report) resets the counter.
    const out = await dispatchTrainingContractNotifications({
      slackWebhookUrl: null,
      emailWebhookUrl: null,
      reportOverride: passingReport(),
      skipPersist: true,
    });
    assert.equal(out.status, "noop");
    assert.equal(_getUnreachableCounterForTests(), 0);

    // Subsequent unreachable cycle starts counting from 1, not 4.
    await dispatchTrainingContractNotifications({
      slackWebhookUrl: null,
      emailWebhookUrl: null,
      fetchOverride: failingFetch,
      skipPersist: true,
    });
    assert.equal(_getUnreachableCounterForTests(), 1);
    _resetUnreachableCounterForTests();
  });

  it("ml-engine unreachable: a non-unreachable fetch error breaks the streak (counter resets)", async () => {
    _resetUnreachableCounterForTests();
    const spy = spyOnWarn();
    const failingFetch = async (): Promise<TrainingReport | null> => {
      throw new MlEngineUnreachableError("ml-engine unreachable (test)");
    };
    const otherErrorFetch = async (): Promise<TrainingReport | null> => {
      throw new Error("ml-engine HTTP 500");
    };

    try {
      // Build up an unreachable streak just shy of the alarm.
      for (let i = 0; i < UNREACHABLE_ALERT_EVERY_N - 1; i++) {
        await dispatchTrainingContractNotifications({
          slackWebhookUrl: null,
          emailWebhookUrl: null,
          fetchOverride: failingFetch,
          skipPersist: true,
        });
      }
      assert.equal(
        _getUnreachableCounterForTests(),
        UNREACHABLE_ALERT_EVERY_N - 1,
      );
      assert.equal(alarmsIn(spy.calls).length, 0);

      // A different (non-unreachable) error means the engine answered.
      // The streak must reset.
      const out = await dispatchTrainingContractNotifications({
        slackWebhookUrl: null,
        emailWebhookUrl: null,
        fetchOverride: otherErrorFetch,
        skipPersist: true,
      });
      assert.equal(out.status, "skipped");
      assert.equal(out.reason, "fetch_failed");
      assert.equal(_getUnreachableCounterForTests(), 0);

      // One more unreachable: counter restarts at 1 — no alarm fires.
      await dispatchTrainingContractNotifications({
        slackWebhookUrl: null,
        emailWebhookUrl: null,
        fetchOverride: failingFetch,
        skipPersist: true,
      });
      assert.equal(_getUnreachableCounterForTests(), 1);
      assert.equal(alarmsIn(spy.calls).length, 0);
    } finally {
      spy.restore();
      _resetUnreachableCounterForTests();
    }
  });

  it("with no channels configured, still records dedup so a future config doesn't blast history", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;

    const out = await dispatchTrainingContractNotifications({
      slackWebhookUrl: null,
      emailWebhookUrl: null,
      reportOverride: failingReport(),
    });
    assert.equal(out.status, "dispatched");
    assert.equal(calls, 0);
    assert.equal(out.sent?.slack, "skipped");
    assert.equal(out.sent?.email, "skipped");

    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, TRAINING_CONTRACT_ALERTS_SENT_KEY));
    assert.ok(row);
    const v = row.value as { keys: string[] };
    assert.ok(v.keys.includes("2026-04-22T10:00:00Z"));
  });
});
