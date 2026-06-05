/**
 * Task #619 — off-dashboard alerting for MTTM auto-disable trips.
 *
 * Background: when MTTM trips its consecutive-loss or n≥10 post-fee
 * floor, `evaluateMttmAutoDisable` (in `./mttm.ts`) writes a typed
 * `mttm_disable_reason` row and flips `mttm_enabled=false`. The
 * dashboard banner surfaces this, but operators who aren't actively
 * looking at the dashboard only find out next time they check.
 *
 * This module closes that gap by firing a single push-channel
 * notification on the rising edge of an auto-disable. The hook is
 * fire-and-forget so a slow webhook never blocks the trade-close
 * sweep, and dedup is keyed by the disable reason's `trippedAt` ISO
 * timestamp so a process restart cannot re-fire for an already-paged
 * trip. Manual disables are skipped — operators flipping the switch
 * by hand don't need to be paged about themselves.
 *
 * Channels mirror the convention used by the other notifiers in this
 * directory (`auto-retire-notifier`, `disabled-outcome-notifier`,
 * `topup-5m-notifier`):
 *   `MTTM_DISABLE_ALERT_WEBHOOK_URL`        — generic JSON webhook
 *   `MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL`  — Slack incoming webhook
 *                                              (falls back to
 *                                              `SLACK_WEBHOOK_URL`)
 * With nothing set the dispatcher still records dedup so that
 * configuring a webhook later does not retroactively page on the
 * historical trip. No new infra is introduced beyond what the other
 * notifiers already use.
 */
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";
import type { MttmDisableReason } from "./mttm";

export const MTTM_DISABLE_ALERTS_SENT_KEY = "mttm.disable_alerts_sent";

/** Hard cap on the persisted dedup set so it cannot grow forever. */
const MAX_TRACKED_KEYS = 200;

export interface MttmDisableAlertPayload {
  reasonKind: MttmDisableReason["reason"];
  detail: string;
  trippedAt: string;
  consecutiveLosses: number | null;
  nTrades: number | null;
  postFeePnlPct: number | null;
}

/**
 * Possible outcomes of a single `notifyMttmAutoDisabled` call:
 *   - "noop"        — duplicate trippedAt; nothing to do.
 *   - "dispatched"  — at least one configured channel delivered.
 *   - "no_channel"  — no channel configured; dedup still recorded so a
 *                     future webhook config doesn't blast history.
 *   - "failed"      — every configured channel errored. Dedup is still
 *                     recorded to avoid re-page storms; ops should
 *                     scrape these as a real-delivery failure.
 *   - "skipped"     — input did not warrant an alert (manual disable,
 *                     missing trippedAt).
 */
export type MttmDisableDispatchStatus =
  | "noop"
  | "dispatched"
  | "no_channel"
  | "failed"
  | "skipped";

export interface MttmDisableDispatchSummary {
  status: MttmDisableDispatchStatus;
  reason?: string;
  /** Stable dedup key used (`mttm_disable@<trippedAt>`) or null when skipped. */
  key: string | null;
  generic: "ok" | "skipped" | "error";
  slack: "ok" | "skipped" | "error";
  error: string | null;
}

export function buildAlertPayload(
  reason: MttmDisableReason,
): MttmDisableAlertPayload {
  return {
    reasonKind: reason.reason,
    detail: reason.detail,
    trippedAt: reason.trippedAt,
    consecutiveLosses:
      typeof reason.consecutiveLosses === "number"
        ? reason.consecutiveLosses
        : null,
    nTrades: typeof reason.nTrades === "number" ? reason.nTrades : null,
    postFeePnlPct:
      typeof reason.postFeePnlPct === "number" ? reason.postFeePnlPct : null,
  };
}

export function formatAlertTitle(p: MttmDisableAlertPayload): string {
  if (p.reasonKind === "consecutive_losses") {
    return "MTTM auto-disabled — consecutive losses";
  }
  if (p.reasonKind === "n10_post_fee") {
    return "MTTM auto-disabled — post-fee PnL floor breached";
  }
  return "MTTM disabled";
}

export function formatAlertBody(p: MttmDisableAlertPayload): string {
  const parts: string[] = [];
  if (p.consecutiveLosses != null) {
    parts.push(`consecutive losses=${p.consecutiveLosses}`);
  }
  if (p.nTrades != null) {
    parts.push(`total trades=${p.nTrades}`);
  }
  if (p.postFeePnlPct != null) {
    parts.push(`post-fee PnL=${(p.postFeePnlPct * 100).toFixed(2)}%`);
  }
  const stats = parts.length > 0 ? ` (${parts.join(", ")})` : "";
  return `${p.detail}${stats}`;
}

function dedupKey(reason: MttmDisableReason): string {
  return `mttm_disable@${reason.trippedAt}`;
}

async function readDedupSet(): Promise<Set<string>> {
  try {
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, MTTM_DISABLE_ALERTS_SENT_KEY));
    if (!row?.value) return new Set();
    const v = row.value as { keys?: unknown };
    if (!Array.isArray(v.keys)) return new Set();
    return new Set(v.keys.filter((k): k is string => typeof k === "string"));
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "mttm-disable-notifier: failed to read dedup set; treating as empty",
    );
    return new Set();
  }
}

async function writeDedupSet(keys: Set<string>): Promise<void> {
  let arr = Array.from(keys);
  if (arr.length > MAX_TRACKED_KEYS) {
    arr = arr.slice(arr.length - MAX_TRACKED_KEYS);
  }
  await db
    .insert(appSettingsTable)
    .values({
      key: MTTM_DISABLE_ALERTS_SENT_KEY,
      value: { keys: arr },
    })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: { keys: arr }, updatedAt: new Date() },
    });
}

async function postJson(url: string, body: unknown): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
  }
}

async function dispatchGeneric(
  url: string,
  p: MttmDisableAlertPayload,
): Promise<void> {
  await postJson(url, {
    title: formatAlertTitle(p),
    text: formatAlertBody(p),
    payload: p,
  });
}

async function dispatchSlack(
  url: string,
  p: MttmDisableAlertPayload,
): Promise<void> {
  const title = formatAlertTitle(p);
  const body = formatAlertBody(p);
  // Include the explicit reasonKind in the body so an operator scanning
  // a Slack channel doesn't have to infer it from the title alone.
  const bodyWithKind = `Reason: \`${p.reasonKind}\`\n${body}`;
  await postJson(url, {
    text: `${title} (${p.reasonKind}) — ${body}`,
    blocks: [
      {
        type: "section",
        text: { type: "mrkdwn", text: `:rotating_light: *${title}*` },
      },
      { type: "section", text: { type: "mrkdwn", text: bodyWithKind } },
    ],
  });
}

export interface DispatchOptions {
  /** Override env-driven generic webhook (test seam). */
  webhookUrl?: string | null;
  /** Override env-driven Slack webhook (test seam). */
  slackWebhookUrl?: string | null;
  /** When true, skip persisting the dedup set (test seam). */
  skipPersist?: boolean;
}

function resolveChannels(opts: DispatchOptions): {
  webhookUrl: string | null;
  slackWebhookUrl: string | null;
} {
  const webhookUrl =
    opts.webhookUrl !== undefined
      ? opts.webhookUrl
      : process.env.MTTM_DISABLE_ALERT_WEBHOOK_URL || null;
  const slackWebhookUrl =
    opts.slackWebhookUrl !== undefined
      ? opts.slackWebhookUrl
      : process.env.MTTM_DISABLE_ALERT_SLACK_WEBHOOK_URL ||
        process.env.SLACK_WEBHOOK_URL ||
        null;
  return { webhookUrl, slackWebhookUrl };
}

/**
 * Fire a single auto-disable alert. Idempotent on `reason.trippedAt`
 * — the second call with the same `trippedAt` is a noop. Manual
 * disables (`reason.reason === "manual"`) are skipped: they are
 * always operator-initiated, so we never page on them.
 *
 * Never throws — all failures are reported in the returned summary so
 * callers in the trade-close hot path can use this in fire-and-forget
 * mode without risk of crashing.
 */
export async function notifyMttmAutoDisabled(
  reason: MttmDisableReason,
  opts: DispatchOptions = {},
): Promise<MttmDisableDispatchSummary> {
  if (reason.reason === "manual") {
    return {
      status: "skipped",
      reason: "manual_disable",
      key: null,
      generic: "skipped",
      slack: "skipped",
      error: null,
    };
  }
  if (!reason.trippedAt) {
    return {
      status: "skipped",
      reason: "missing_tripped_at",
      key: null,
      generic: "skipped",
      slack: "skipped",
      error: null,
    };
  }

  const key = dedupKey(reason);
  const sent = await readDedupSet();
  if (sent.has(key)) {
    return {
      status: "noop",
      key,
      generic: "skipped",
      slack: "skipped",
      error: null,
    };
  }

  const { webhookUrl, slackWebhookUrl } = resolveChannels(opts);
  const payload = buildAlertPayload(reason);

  let generic: "ok" | "skipped" | "error" = "skipped";
  let slack: "ok" | "skipped" | "error" = "skipped";
  let error: string | null = null;

  if (webhookUrl) {
    try {
      await dispatchGeneric(webhookUrl, payload);
      generic = "ok";
    } catch (err) {
      generic = "error";
      const msg = err instanceof Error ? err.message : String(err);
      error = msg;
      logger.warn(
        { err: msg, key },
        "mttm-disable-notifier: generic webhook dispatch failed",
      );
    }
  }
  if (slackWebhookUrl) {
    try {
      await dispatchSlack(slackWebhookUrl, payload);
      slack = "ok";
    } catch (err) {
      slack = "error";
      const msg = err instanceof Error ? err.message : String(err);
      error = error ? `${error}; slack: ${msg}` : msg;
      logger.warn(
        { err: msg, key },
        "mttm-disable-notifier: Slack dispatch failed",
      );
    }
  }

  // Always mark as sent — even on dispatch errors. Otherwise a
  // persistent webhook outage would re-page on every subsequent
  // trade close that calls evaluateMttmAutoDisable. The warn logs
  // above ensure dispatch outages aren't silent.
  sent.add(key);
  if (!opts.skipPersist) {
    try {
      await writeDedupSet(sent);
    } catch (err) {
      logger.warn(
        { err: err instanceof Error ? err.message : String(err), key },
        "mttm-disable-notifier: failed to persist dedup set; next call may resend",
      );
    }
  }

  // Compute a precise outcome so ops telemetry can distinguish "we
  // delivered", "no channel was configured", and "every channel
  // errored". The dedup row is written in all three cases above so a
  // restart cannot re-page on the same trip.
  const anyConfigured = generic !== "skipped" || slack !== "skipped";
  const anyOk = generic === "ok" || slack === "ok";
  let status: MttmDisableDispatchStatus;
  if (!anyConfigured) status = "no_channel";
  else if (anyOk) status = "dispatched";
  else status = "failed";

  if (anyOk) {
    logger.info(
      {
        reasonKind: payload.reasonKind,
        consecutiveLosses: payload.consecutiveLosses,
        nTrades: payload.nTrades,
        postFeePnlPct: payload.postFeePnlPct,
        generic,
        slack,
      },
      "mttm-disable-notifier: dispatched alert",
    );
  } else if (status === "failed") {
    logger.warn(
      {
        reasonKind: payload.reasonKind,
        generic,
        slack,
        err: error,
      },
      "mttm-disable-notifier: every configured channel failed; dedup still recorded",
    );
  }

  return {
    status,
    key,
    generic,
    slack,
    error,
  };
}
