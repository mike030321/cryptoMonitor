/**
 * Task #279 — push-channel notifications for training-contract failures.
 *
 * Background: the dashboard's `training-contract-card` already toasts
 * when a fresh training report has any timeframe with
 * `leakage_audit.passed=false` or `provenance.rejected_synthetic=true`.
 * That toast only fires while an operator has the dashboard open. When
 * nobody's watching, a contract failure can sit unacknowledged until
 * the next training run buries it.
 *
 * This module closes that gap by polling the same training report from
 * the api-server (which runs 24/7) and dispatching out-of-band
 * notifications to a Slack webhook AND/OR a generic email-style
 * webhook. The notification body lists which timeframes failed and
 * why so the on-call can decide whether to roll back without opening
 * the dashboard.
 *
 * Dedup: each training run is uniquely identified by its
 * `generated_at` timestamp. Once we fire an alert for a given run we
 * record the timestamp in `app_settings.training_contract_alerts_sent`
 * so a backend restart doesn't re-fire. The bucket is bounded to the
 * most recent 200 keys so it doesn't grow forever.
 *
 * Channels are opt-in. With no env vars set the poller is still
 * "armed" — it seeds the dedup set so a future webhook configuration
 * doesn't suddenly blast every historical failed run. Channel dispatch
 * failures are isolated: a Slack outage never blocks the email
 * webhook, and neither blocks the dedup write so we don't loop on
 * the same run forever. Mirrors the pattern in
 * `auto-retire-notifier.ts` (task #258).
 */
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";

export const TRAINING_CONTRACT_ALERTS_SENT_KEY =
  "training_contract.alerts_sent";

const MAX_TRACKED_KEYS = 200;

/**
 * Task #407 — paged-alarm threshold for ml-engine unreachability.
 * The poller fires every 60s (default), so 5 consecutive failures
 * means the ml-engine has been silent for ~5 minutes. We emit a
 * structured WARN at every Nth consecutive failure (i.e. 5, 10, 15…)
 * with the explicit `event` name `training_contract_notifier_unreachable`
 * so an external log monitor (e.g. Replit deployment-log filter) can
 * page on it. The counter resets on the first successful report fetch.
 */
export const UNREACHABLE_ALERT_EVERY_N = 5;

/**
 * Task #407 — sentinel error class. Thrown by `fetchTrainingReport`
 * when `waitForMlEngineHealth` times out. The dispatch catch block
 * keys off `instanceof MlEngineUnreachableError` to decide whether
 * to bump the consecutive-failure counter (vs. other transient
 * fetch failures that don't indicate an unreachable engine).
 */
export class MlEngineUnreachableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MlEngineUnreachableError";
  }
}

let consecutiveUnreachable = 0;

/** Test seam — reset the consecutive-unreachable counter. */
export function _resetUnreachableCounterForTests(): void {
  consecutiveUnreachable = 0;
}

/** Test seam — read the consecutive-unreachable counter. */
export function _getUnreachableCounterForTests(): number {
  return consecutiveUnreachable;
}

interface LeakageAudit {
  passed?: boolean;
  violations?: unknown[];
}

interface ProvenanceSummary {
  rows_real?: number;
  rows_synthetic?: number;
  coins_rejected?: string[];
  rejected_synthetic?: boolean;
}

interface TimeframeReport {
  status?: string;
  leakage_audit?: LeakageAudit | null;
  provenance?: ProvenanceSummary | null;
}

export interface TrainingReport {
  status?: string;
  generated_at?: string;
  timeframes?: Record<string, TimeframeReport>;
}

export interface FailedTimeframe {
  timeframe: string;
  leakageFailed: boolean;
  provenanceRejected: boolean;
  coinsRejected: string[];
}

export interface TrainingContractAlertPayload {
  generatedAt: string;
  failedTimeframes: FailedTimeframe[];
  totalRejectedCoins: number;
}

export interface TrainingContractDispatchSummary {
  status: "noop" | "dispatched" | "skipped";
  reason?: string;
  generatedAt?: string;
  sent?: {
    slack: "ok" | "skipped" | "error";
    email: "ok" | "skipped" | "error";
    error?: string | null;
  };
}

/** Pure helper: identify the failed timeframes in a training report.
 *  Returns an empty array if nothing failed. Mirrors the same checks
 *  the in-app `training-contract-card` uses to render its red banner. */
export function findFailedTimeframes(
  report: TrainingReport | null | undefined,
): FailedTimeframe[] {
  if (!report?.timeframes) return [];
  const out: FailedTimeframe[] = [];
  for (const [tf, rep] of Object.entries(report.timeframes)) {
    const leakageFailed =
      rep?.leakage_audit != null && rep.leakage_audit.passed === false;
    const provenanceRejected = rep?.provenance?.rejected_synthetic === true;
    if (!leakageFailed && !provenanceRejected) continue;
    out.push({
      timeframe: tf,
      leakageFailed,
      provenanceRejected,
      coinsRejected: rep?.provenance?.coins_rejected ?? [],
    });
  }
  return out;
}

export function buildAlertPayload(
  report: TrainingReport,
  failed: FailedTimeframe[],
): TrainingContractAlertPayload {
  const totalRejectedCoins = failed.reduce(
    (n, f) => n + f.coinsRejected.length,
    0,
  );
  return {
    generatedAt: report.generated_at ?? "unknown",
    failedTimeframes: failed,
    totalRejectedCoins,
  };
}

export function formatAlertTitle(p: TrainingContractAlertPayload): string {
  return `Training contract failed: ${p.failedTimeframes.length} timeframe(s)`;
}

export function formatAlertBody(p: TrainingContractAlertPayload): string {
  const lines: string[] = [`Run: ${p.generatedAt}`];
  for (const f of p.failedTimeframes) {
    const flags: string[] = [];
    if (f.leakageFailed) flags.push("leakage");
    if (f.provenanceRejected) flags.push("synthetic-rejected");
    let line = `• ${f.timeframe}: ${flags.join(" + ")}`;
    if (f.coinsRejected.length > 0) {
      line += ` (${f.coinsRejected.length} coin${f.coinsRejected.length === 1 ? "" : "s"} rejected)`;
    }
    lines.push(line);
  }
  if (p.totalRejectedCoins > 0) {
    lines.push(`Total rejected coins across timeframes: ${p.totalRejectedCoins}`);
  }
  return lines.join("\n");
}

async function readDedupSet(): Promise<Set<string>> {
  try {
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, TRAINING_CONTRACT_ALERTS_SENT_KEY));
    if (!row?.value) return new Set();
    const v = row.value as { keys?: unknown };
    if (!Array.isArray(v.keys)) return new Set();
    return new Set(v.keys.filter((k): k is string => typeof k === "string"));
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "training-contract-notifier: failed to read dedup set; treating as empty",
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
      key: TRAINING_CONTRACT_ALERTS_SENT_KEY,
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
  p: TrainingContractAlertPayload,
): Promise<void> {
  const title = formatAlertTitle(p);
  const body = formatAlertBody(p);
  await postJson(url, {
    text: `${title}\n${body}`,
    blocks: [
      {
        type: "section",
        text: { type: "mrkdwn", text: `:rotating_light: *${title}*` },
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
  p: TrainingContractAlertPayload,
): Promise<void> {
  await postJson(url, {
    subject: formatAlertTitle(p),
    text: `${formatAlertTitle(p)}\n\n${formatAlertBody(p)}\n`,
    payload: p,
  });
}

/**
 * Task #405 / B-NOTIFIER-BOOT — wait for the ml-engine to come up
 * before issuing a real fetch. Without this, the API process boots
 * faster than the Python ml-engine and the very first poll throws a
 * connection-refused, which (a) emits a noisy WARN on every clean
 * boot and (b) wastes a polling tick. Returns true when /ml/health
 * answers, false on timeout — caller uses the false case to skip the
 * poll quietly.
 */
async function waitForMlEngineHealth(opts: { timeoutMs: number; intervalMs: number }): Promise<boolean> {
  const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(/\/$/, "");
  const deadline = Date.now() + opts.timeoutMs;
  while (Date.now() < deadline) {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), Math.min(2000, opts.intervalMs));
      const r = await fetch(`${base}/ml/health`, { signal: ctrl.signal });
      clearTimeout(t);
      if (r.ok) return true;
    } catch {
      // swallow — try again
    }
    await new Promise((res) => setTimeout(res, opts.intervalMs));
  }
  return false;
}

async function fetchTrainingReport(): Promise<TrainingReport | null> {
  const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(
    /\/$/,
    "",
  );
  // Task #405 / B-NOTIFIER-BOOT — short health-wait so a colocated ml
  // engine that's still booting doesn't cause this poll to throw a
  // noisy connection-refused on every clean restart. The wait is bounded
  // (2.5s) so a genuinely-down ml-engine still surfaces a fetch error
  // promptly and the existing skip path takes over.
  const healthy = await waitForMlEngineHealth({ timeoutMs: 2500, intervalMs: 250 });
  if (!healthy) {
    throw new MlEngineUnreachableError(
      "ml-engine unreachable (no /ml/health within 2.5s)",
    );
  }
  const r = await fetch(`${base}/ml/training/report`);
  if (!r.ok) {
    throw new Error(`ml-engine HTTP ${r.status}`);
  }
  const json = (await r.json()) as TrainingReport;
  return json;
}

export interface DispatchOptions {
  slackWebhookUrl?: string | null;
  emailWebhookUrl?: string | null;
  /** Inject the report for tests (skips ml-engine fetch). */
  reportOverride?: TrainingReport | null;
  /** When true, skip persisting the dedup set (test seam). */
  skipPersist?: boolean;
  /**
   * Test seam: replace the default `fetchTrainingReport` call. Used by
   * the unreachable-alarm tests to simulate `MlEngineUnreachableError`
   * without going through the (slow) 2.5s health-check.
   */
  fetchOverride?: () => Promise<TrainingReport | null>;
}

/**
 * Single poll tick: fetch the latest training report, check for
 * contract failures, dispatch one alert per (slack + email) per
 * `generated_at`, persist dedup. Never throws — partial failures are
 * reported in the summary.
 */
export async function dispatchTrainingContractNotifications(
  opts: DispatchOptions = {},
): Promise<TrainingContractDispatchSummary> {
  const slackUrl = opts.slackWebhookUrl !== undefined
    ? opts.slackWebhookUrl
    : (process.env.TRAINING_CONTRACT_SLACK_WEBHOOK_URL
        || process.env.SLACK_WEBHOOK_URL
        || null);
  const emailUrl = opts.emailWebhookUrl !== undefined
    ? opts.emailWebhookUrl
    : (process.env.TRAINING_CONTRACT_EMAIL_WEBHOOK_URL || null);

  let report: TrainingReport | null;
  try {
    if (opts.reportOverride !== undefined) {
      report = opts.reportOverride;
    } else if (opts.fetchOverride) {
      report = await opts.fetchOverride();
    } else {
      report = await fetchTrainingReport();
    }
    // Task #407 — successful fetch (any non-throw) clears the
    // unreachable counter. Other transient fetch failures (e.g. HTTP
    // 500 on /ml/training/report) do not increment, but we don't want
    // a passing report to leave a stale counter behind either.
    consecutiveUnreachable = 0;
  } catch (err) {
    if (err instanceof MlEngineUnreachableError) {
      consecutiveUnreachable += 1;
      const shouldAlarm =
        consecutiveUnreachable % UNREACHABLE_ALERT_EVERY_N === 0;
      if (shouldAlarm) {
        logger.warn(
          {
            event: "training_contract_notifier_unreachable",
            consecutiveFailures: consecutiveUnreachable,
            everyN: UNREACHABLE_ALERT_EVERY_N,
            err: err.message,
          },
          `training-contract-notifier: ml-engine unreachable for ${consecutiveUnreachable} consecutive cycles`,
        );
      } else {
        logger.warn(
          {
            err: err.message,
            consecutiveFailures: consecutiveUnreachable,
          },
          "training-contract-notifier: failed to fetch training report",
        );
      }
    } else {
      // Non-unreachable error (e.g. /ml/health passed but
      // /ml/training/report returned 500) breaks the unreachable
      // streak — the engine answered, so the count of *consecutive
      // unreachable* cycles must reset to 0.
      consecutiveUnreachable = 0;
      logger.warn(
        { err: err instanceof Error ? err.message : String(err) },
        "training-contract-notifier: failed to fetch training report",
      );
    }
    return { status: "skipped", reason: "fetch_failed" };
  }

  if (!report || report.status === "missing" || !report.generated_at) {
    return { status: "noop", reason: "no_report" };
  }

  const failed = findFailedTimeframes(report);
  if (failed.length === 0) {
    return { status: "noop", reason: "contract_passed", generatedAt: report.generated_at };
  }

  const dedup = await readDedupSet();
  const key = report.generated_at;
  if (dedup.has(key)) {
    return { status: "noop", reason: "already_sent", generatedAt: key };
  }

  const payload = buildAlertPayload(report, failed);
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
        { err: error, generatedAt: key },
        "training-contract-notifier: Slack dispatch failed",
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
        { err: msg, generatedAt: key },
        "training-contract-notifier: email dispatch failed",
      );
    }
  }

  // Mark as sent unconditionally (matches auto-retire-notifier): we
  // log dispatch errors loudly above, the operator can still see the
  // failure in the dashboard's training-contract card, and we don't
  // want to retry-loop on a broken webhook.
  dedup.add(key);
  if (!opts.skipPersist) {
    try {
      await writeDedupSet(dedup);
    } catch (err) {
      logger.warn(
        { err: err instanceof Error ? err.message : String(err) },
        "training-contract-notifier: failed to persist dedup set; next poll may resend",
      );
    }
  }

  if (slack === "ok" || email === "ok") {
    logger.info(
      {
        generatedAt: key,
        failedCount: failed.length,
        slack,
        email,
      },
      "training-contract-notifier: dispatched alert",
    );
  }

  return {
    status: "dispatched",
    generatedAt: key,
    sent: { slack, email, error },
  };
}

/**
 * Startup seeding: on first boot, if the current training report is
 * already failing the contract we don't want to immediately page on
 * historical state. Mark its `generated_at` as already-sent so we only
 * fire on the NEXT failing run. Subsequent restarts are protected by
 * the persisted dedup set.
 */
export async function seedDedupSetFromCurrentReport(): Promise<boolean> {
  try {
    const existing = await readDedupSet();
    if (existing.size > 0) return false;
    const report = await fetchTrainingReport();
    if (!report || !report.generated_at) return false;
    const failed = findFailedTimeframes(report);
    if (failed.length === 0) return false;
    const seeded = new Set([report.generated_at]);
    await writeDedupSet(seeded);
    logger.info(
      { generatedAt: report.generated_at },
      "training-contract-notifier: seeded dedup set from current failing report",
    );
    return true;
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "training-contract-notifier: seeding failed",
    );
    return false;
  }
}

let pollInterval: ReturnType<typeof setInterval> | null = null;

/** Start the periodic poll. Idempotent. */
export function startTrainingContractNotifierLoop(intervalMs = 60_000): void {
  if (pollInterval) return;
  pollInterval = setInterval(() => {
    void dispatchTrainingContractNotifications().catch((err) =>
      logger.error(
        { err },
        "training-contract-notifier: unexpected dispatch error",
      ),
    );
  }, intervalMs);
}

/** Test seam — stop the loop. */
export function stopTrainingContractNotifierLoop(): void {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}
