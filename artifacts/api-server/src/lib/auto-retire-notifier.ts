/**
 * Task #258 — push-channel notifications for ml-engine auto-retire
 * events.
 *
 * Background: task #236 added the auto-retire pipeline in the
 * ml-engine, and task #252 added an in-app toast on the dashboard so
 * an operator with the page open sees a new entry the moment it
 * appears in `app_settings.feature_lab.quarantined`. That toast only
 * fires while the dashboard is open. Operators away from the screen
 * still miss the regression window — by the time they come back the
 * next training run may have buried the alert.
 *
 * This module closes the gap by polling the same bucket from the
 * api-server (which runs 24/7) and dispatching out-of-band
 * notifications to a Slack webhook AND/OR a generic email-style
 * webhook. The notification body matches the toast: feature name,
 * worst-regressing timeframe, Δlog_loss, reason.
 *
 * Dedup: alerted entries are persisted in
 * `app_settings.feature_lab.auto_retire_alerts_sent` keyed by
 * `name@quarantinedAt` so a backend restart doesn't re-fire an alert
 * for an entry we already sent. The bucket is bounded to the most
 * recent 200 keys so it doesn't grow forever.
 *
 * Channels are opt-in. With no env vars set the poller is still
 * "armed" — it seeds the dedup set so a future webhook configuration
 * doesn't suddenly blast every historical record. Channel dispatch
 * failures are isolated: a Slack outage never blocks the email
 * webhook, and neither blocks the dedup write so we don't loop on
 * the same entry forever.
 */
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";
import {
  getQuarantinedFeatures,
  type QuarantinedFeatureRecord,
} from "./feature-lab";

export const AUTO_RETIRE_ALERTS_SENT_KEY = "feature_lab.auto_retire_alerts_sent";

/** Cap on the persisted dedup set; one entry per (name, quarantinedAt). */
const MAX_TRACKED_KEYS = 200;

export interface AutoRetireAlertPayload {
  name: string;
  worstTimeframe: string | null;
  deltaLogLoss: number | null;
  currentLogLoss: number | null;
  priorLogLoss: number | null;
  reason: string;
  quarantinedAt: string;
  threshold: number | null;
}

export interface AutoRetireDispatchSummary {
  status: "noop" | "dispatched" | "skipped";
  reason?: string;
  checked: number;
  newAlerts: number;
  sent: Array<{
    key: string;
    slack: "ok" | "skipped" | "error";
    email: "ok" | "skipped" | "error";
    error?: string | null;
  }>;
}

interface QuarantinedTimeframeDetail {
  timeframe?: string;
  current_log_loss?: number | null;
  prior_log_loss?: number | null;
  delta_log_loss?: number | null;
}

function keyOf(q: QuarantinedFeatureRecord): string {
  return `${q.name}@${q.quarantinedAt}`;
}

/** Pick the timeframe with the largest (worst) Δlog_loss. Mirrors the
 *  same picker used by the in-app toast in `phase6-diagnostics-cards`. */
export function pickWorstTimeframe(
  tfs: ReadonlyArray<QuarantinedTimeframeDetail> | undefined | null,
): QuarantinedTimeframeDetail | null {
  if (!tfs || tfs.length === 0) return null;
  let worst: QuarantinedTimeframeDetail | null = null;
  for (const t of tfs) {
    const d = t?.delta_log_loss;
    if (d === null || d === undefined || !Number.isFinite(d)) continue;
    if (!worst || (worst.delta_log_loss ?? -Infinity) < d) worst = t;
  }
  return worst ?? tfs[0] ?? null;
}

export function buildAlertPayload(
  q: QuarantinedFeatureRecord,
): AutoRetireAlertPayload {
  const detail = (q.detail ?? {}) as {
    timeframes?: QuarantinedTimeframeDetail[];
    threshold?: number | null;
  };
  const worst = pickWorstTimeframe(detail.timeframes);
  return {
    name: q.name,
    worstTimeframe: worst?.timeframe ?? null,
    deltaLogLoss: worst?.delta_log_loss ?? null,
    currentLogLoss: worst?.current_log_loss ?? null,
    priorLogLoss: worst?.prior_log_loss ?? null,
    reason: q.reason,
    quarantinedAt: q.quarantinedAt,
    threshold:
      typeof detail.threshold === "number" && Number.isFinite(detail.threshold)
        ? detail.threshold
        : null,
  };
}

export function formatAlertTitle(p: AutoRetireAlertPayload): string {
  return `Feature auto-retired: ${p.name}`;
}

export function formatAlertBody(p: AutoRetireAlertPayload): string {
  const tfLabel = p.worstTimeframe ?? "unknown timeframe";
  const deltaLabel = p.deltaLogLoss != null
    ? `Δlog_loss ${p.deltaLogLoss >= 0 ? "+" : ""}${p.deltaLogLoss.toFixed(4)}`
    : "Δlog_loss n/a";
  return `Worst timeframe ${tfLabel} · ${deltaLabel} · reason ${p.reason}`;
}

async function readDedupSet(): Promise<Set<string>> {
  try {
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, AUTO_RETIRE_ALERTS_SENT_KEY));
    if (!row?.value) return new Set();
    const v = row.value as { keys?: unknown };
    if (!Array.isArray(v.keys)) return new Set();
    return new Set(v.keys.filter((k): k is string => typeof k === "string"));
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "auto-retire-notifier: failed to read dedup set; treating as empty",
    );
    return new Set();
  }
}

async function writeDedupSet(keys: Set<string>): Promise<void> {
  // Bound the set to the most recent MAX_TRACKED_KEYS (insertion order).
  let arr = Array.from(keys);
  if (arr.length > MAX_TRACKED_KEYS) {
    arr = arr.slice(arr.length - MAX_TRACKED_KEYS);
  }
  await db
    .insert(appSettingsTable)
    .values({
      key: AUTO_RETIRE_ALERTS_SENT_KEY,
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

async function dispatchSlack(
  url: string,
  p: AutoRetireAlertPayload,
): Promise<void> {
  // Slack incoming-webhook payload. Use blocks so the worst-tf line
  // wraps cleanly in the Slack client; fall back text drives mobile
  // push previews.
  const title = formatAlertTitle(p);
  const body = formatAlertBody(p);
  await postJson(url, {
    text: `${title} — ${body}`,
    blocks: [
      {
        type: "section",
        text: { type: "mrkdwn", text: `:warning: *${title}*` },
      },
      {
        type: "section",
        text: { type: "mrkdwn", text: body },
      },
    ],
  });
}

async function dispatchEmail(
  url: string,
  p: AutoRetireAlertPayload,
): Promise<void> {
  // Generic email webhook contract: POSTs a JSON body that the
  // operator's relay (Zapier / Make / a tiny Lambda) forwards as an
  // email. Keeps this module free of SMTP / nodemailer wiring while
  // still satisfying the "email" leg of the task.
  await postJson(url, {
    subject: formatAlertTitle(p),
    text: `${formatAlertTitle(p)}\n\n${formatAlertBody(p)}\n\n` +
      `Quarantined at: ${p.quarantinedAt}\n` +
      (p.currentLogLoss != null && p.priorLogLoss != null
        ? `log_loss prior=${p.priorLogLoss.toFixed(4)} current=${p.currentLogLoss.toFixed(4)}\n`
        : "") +
      (p.threshold != null ? `threshold=${p.threshold}\n` : ""),
    payload: p,
  });
}

export interface DispatchOptions {
  /** Override the env-driven Slack webhook URL (test seam). */
  slackWebhookUrl?: string | null;
  /** Override the env-driven email webhook URL (test seam). */
  emailWebhookUrl?: string | null;
  /** Inject the quarantine source for tests (skips DB read). */
  quarantinedOverride?: QuarantinedFeatureRecord[];
  /** When true, skip persisting the dedup set (test seam). */
  skipPersist?: boolean;
}

/**
 * Single poll tick: read quarantine bucket, diff against dedup set,
 * dispatch new entries to Slack + email webhooks (whichever are
 * configured), persist updated dedup set.
 *
 * Never throws — partial failures are reported in the summary so the
 * scheduled loop never crashes the api-server.
 */
export async function dispatchAutoRetireNotifications(
  opts: DispatchOptions = {},
): Promise<AutoRetireDispatchSummary> {
  const slackUrl = opts.slackWebhookUrl !== undefined
    ? opts.slackWebhookUrl
    : (process.env.AUTO_RETIRE_SLACK_WEBHOOK_URL
        || process.env.SLACK_WEBHOOK_URL
        || null);
  const emailUrl = opts.emailWebhookUrl !== undefined
    ? opts.emailWebhookUrl
    : (process.env.AUTO_RETIRE_EMAIL_WEBHOOK_URL || null);

  let quarantined: QuarantinedFeatureRecord[];
  try {
    quarantined = opts.quarantinedOverride
      ?? (await getQuarantinedFeatures());
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "auto-retire-notifier: failed to read quarantine bucket",
    );
    return {
      status: "skipped",
      reason: "read_failed",
      checked: 0,
      newAlerts: 0,
      sent: [],
    };
  }

  const sent = await readDedupSet();
  const newEntries = quarantined.filter((q) => !sent.has(keyOf(q)));

  if (newEntries.length === 0) {
    return {
      status: "noop",
      checked: quarantined.length,
      newAlerts: 0,
      sent: [],
    };
  }

  const results: AutoRetireDispatchSummary["sent"] = [];
  for (const q of newEntries) {
    const payload = buildAlertPayload(q);
    let slack: "ok" | "skipped" | "error" = "skipped";
    let email: "ok" | "skipped" | "error" = "skipped";
    let error: string | null = null;

    if (slackUrl) {
      try {
        await dispatchSlack(slackUrl, payload);
        slack = "ok";
      } catch (err) {
        slack = "error";
        error = err instanceof Error ? err.message : String(err);
        logger.warn(
          { err: error, name: q.name },
          "auto-retire-notifier: Slack dispatch failed",
        );
      }
    }
    if (emailUrl) {
      try {
        await dispatchEmail(emailUrl, payload);
        email = "ok";
      } catch (err) {
        email = "error";
        const msg = err instanceof Error ? err.message : String(err);
        error = error ? `${error}; email: ${msg}` : msg;
        logger.warn(
          { err: msg, name: q.name },
          "auto-retire-notifier: email dispatch failed",
        );
      }
    }

    results.push({ key: keyOf(q), slack, email, error });
    // Always mark as sent — even on dispatch errors. The retry would
    // re-queue the same entry forever, and the operator can see the
    // record in-app via the existing Quarantined Features card. We
    // log loudly above so dispatch outages aren't silent.
    sent.add(keyOf(q));

    if (slack === "ok" || email === "ok") {
      logger.info(
        {
          name: q.name,
          worstTimeframe: payload.worstTimeframe,
          deltaLogLoss: payload.deltaLogLoss,
          slack,
          email,
        },
        "auto-retire-notifier: dispatched alert",
      );
    }
  }

  if (!opts.skipPersist) {
    try {
      await writeDedupSet(sent);
    } catch (err) {
      logger.warn(
        { err: err instanceof Error ? err.message : String(err) },
        "auto-retire-notifier: failed to persist dedup set; next poll may resend",
      );
    }
  }

  return {
    status: "dispatched",
    checked: quarantined.length,
    newAlerts: newEntries.length,
    sent: results,
  };
}

/**
 * Startup seeding: on first boot we don't want to blast every existing
 * quarantined feature into Slack/email. Read the bucket once, mark
 * every record as already-sent, and only alert on entries that appear
 * AFTER this point. Subsequent restarts are protected by the persisted
 * dedup set.
 */
export async function seedDedupSetFromCurrentQuarantine(): Promise<number> {
  try {
    const existing = await readDedupSet();
    if (existing.size > 0) return 0; // already seeded on a prior run
    const records = await getQuarantinedFeatures();
    if (records.length === 0) return 0;
    const seeded = new Set(records.map(keyOf));
    await writeDedupSet(seeded);
    logger.info(
      { seeded: seeded.size },
      "auto-retire-notifier: seeded dedup set from existing quarantine",
    );
    return seeded.size;
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "auto-retire-notifier: seeding failed",
    );
    return 0;
  }
}

let pollInterval: ReturnType<typeof setInterval> | null = null;

/** Start the periodic poll. Idempotent. */
export function startAutoRetireNotifierLoop(intervalMs = 60_000): void {
  if (pollInterval) return;
  pollInterval = setInterval(() => {
    void dispatchAutoRetireNotifications().catch((err) =>
      logger.error(
        { err },
        "auto-retire-notifier: unexpected dispatch error",
      ),
    );
  }, intervalMs);
}

/** Test seam — stop the loop. */
export function stopAutoRetireNotifierLoop(): void {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}
