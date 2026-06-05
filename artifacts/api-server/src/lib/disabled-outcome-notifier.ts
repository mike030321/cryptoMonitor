/**
 * Task #577 — off-dashboard alerting + dashboard banner state for the
 * "outcome landed on a disabled timeframe" event.
 *
 * Background: the Python meta-brain (`artifacts/ml-engine/app/meta_brain.py`)
 * rejects record-outcome calls whose resolved `slice_role == "disabled"`
 * with `{ok:false, reason:"disabled_role_rejected"}` and a structured
 * `[disabled_outcome_received]` warn. That is the ONLY operator-facing
 * signal that some upstream gate has leaked a trade from a turned-off
 * timeframe. Until now the warn lived in `/var/log` only — operators
 * had to grep to find leaks, which means leaks of "trades from a
 * disabled timeframe" sit undetected for days.
 *
 * This module closes the loop:
 *   1. `recordDisabledOutcomeRejection({tickId, sliceId, timeframe})`
 *      is called inline by the api-server's meta-brain client whenever
 *      it sees the rejection on the wire. The event lands in a bounded
 *      in-process ring buffer.
 *   2. A push-driven dispatcher fires a webhook (generic + Slack)
 *      on the rising edge of an incident. The incident key is the
 *      sorted set of distinct timeframes seen in the active window;
 *      a fresh timeframe entering the set re-pages, but a continuing
 *      stream from the same timeframe(s) dedupes.
 *   3. A periodic sweep loop fires the recovery webhook when the
 *      active window has been clear for one full poll.
 *   4. `getDisabledOutcomeBannerState()` exposes the same recent-event
 *      summary the dispatcher uses, so the dashboard banner shows
 *      what the operator would have been paged for.
 *
 * Channels are env-driven and opt-in:
 *   `DISABLED_OUTCOME_ALERT_WEBHOOK_URL`        (generic JSON webhook)
 *   `DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL`  (Slack incoming webhook;
 *                                                falls back to
 *                                                `SLACK_WEBHOOK_URL`)
 * With nothing set the dispatcher still tracks state so the dashboard
 * banner is accurate, but does not POST anywhere.
 *
 * Window: configurable via `DISABLED_OUTCOME_WINDOW_MINUTES` (default
 * 5). The task spec says "alert when one or more disabled-role outcomes
 * have been rejected by the brain in the last N minutes"; 5 minutes is
 * a sensible default that keeps an operator inside the "minutes, not
 * days" deadline while not pageing on a single one-off race.
 */
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";

export const DISABLED_OUTCOME_NOTIFIER_STATE_KEY =
  "disabled_outcome_notifier.state";

/** Hard cap on the in-process ring buffer. We never need more than a
 *  few minutes of events; 256 is more than enough headroom even at
 *  pathological leak rates without making the dashboard payload huge. */
export const DISABLED_OUTCOME_RING_CAP = 256;

/** Default window the banner / paging logic asks "any events in the
 *  last N minutes?". Overridable via `DISABLED_OUTCOME_WINDOW_MINUTES`. */
export const DEFAULT_WINDOW_MINUTES = 5;

/** Hard floor on the window so a misconfigured env can't accidentally
 *  reduce the alert to a single-tick race. */
const MIN_WINDOW_MS = 60_000;

export interface DisabledOutcomeEvent {
  /** The tick_id the api-server sent on the wire (post `shadow:` strip). */
  tickId: string;
  /** Slice id if the api-server captured one; null when not tracked. */
  sliceId: string | null;
  /** Resolved timeframe whose role was `disabled` at submission time. */
  timeframe: string;
  /** Wall-clock ms when the rejection was observed. */
  observedAt: number;
}

export interface DisabledOutcomeNotifierState {
  /** Stable identity of the active incident. Null when no events in
   *  the active window. The key is bucketed by the sorted set of
   *  distinct timeframes seen in the window so a NEW timeframe joining
   *  the leak (e.g. another disabled tf starts leaking too) is a fresh
   *  incident worth re-paging on. */
  activeIncidentKey: string | null;
  /** ISO timestamp the active incident was first observed. */
  activeIncidentSince: string | null;
  /** Reason summary string for the active incident. */
  activeIncidentReason: string | null;
  /** Distinct timeframes seen in the active window at last dispatch. */
  activeTimeframes: string[];
  /** Total event count seen in the active window at last dispatch. */
  activeEventCount: number;
  /** ISO timestamp of the last alert dispatched (incident OR recovery). */
  lastAlertAt: string | null;
  /** Last alert kind dispatched. */
  lastAlertKind: "incident" | "recovery" | null;
  /** Last alert reason summary. */
  lastAlertReason: string | null;
}

const EMPTY_STATE: DisabledOutcomeNotifierState = {
  activeIncidentKey: null,
  activeIncidentSince: null,
  activeIncidentReason: null,
  activeTimeframes: [],
  activeEventCount: 0,
  lastAlertAt: null,
  lastAlertKind: null,
  lastAlertReason: null,
};

// ─────────────────────── ring buffer (in-process) ────────────────────

const ringBuffer: DisabledOutcomeEvent[] = [];

/** Test seam — restore the ring to empty between tests. */
export function __resetRingBuffer(): void {
  ringBuffer.length = 0;
}

/** Test seam — read a snapshot of the current ring buffer. */
export function __snapshotRingBuffer(): DisabledOutcomeEvent[] {
  return ringBuffer.slice();
}

/** Test seam — when true, `recordDisabledOutcomeRejection` skips the
 *  fire-and-forget inline dispatch. Used by unit tests that want to
 *  exercise the ring buffer without bleeding state into other tests. */
let inlineDispatchDisabled = false;
export function __setInlineDispatchDisabled(v: boolean): void {
  inlineDispatchDisabled = v;
}

/**
 * Synchronously record a rejection. Cheap (just a push + bounded
 * shift). Does NOT block on dispatch — the caller is the trade-close
 * path and we never want to add latency there. The async tick is
 * scheduled fire-and-forget; recovery is handled by the sweep loop.
 */
export function recordDisabledOutcomeRejection(
  event: Omit<DisabledOutcomeEvent, "observedAt"> & { observedAt?: number },
): void {
  const observedAt = event.observedAt ?? Date.now();
  ringBuffer.push({
    tickId: event.tickId,
    sliceId: event.sliceId ?? null,
    timeframe: event.timeframe,
    observedAt,
  });
  while (ringBuffer.length > DISABLED_OUTCOME_RING_CAP) ringBuffer.shift();

  if (inlineDispatchDisabled) return;

  // Dispatch is fire-and-forget. A failed dispatch is logged and the
  // sweep loop will retry. We never want to throw from the trade-close
  // path. The function below short-circuits on stale window.
  void runDisabledOutcomeNotifierTick().catch((err) =>
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "disabled-outcome-notifier: inline dispatch failed",
    ),
  );
}

// ─────────────────────────── pure helpers ────────────────────────────

export interface RecentEventSummary {
  /** Events in the active window, oldest-first. */
  recent: DisabledOutcomeEvent[];
  /** Total event count in the active window. */
  count: number;
  /** Distinct timeframes seen in the active window, sorted. */
  timeframes: string[];
  /** Per-timeframe rejection count, by timeframe. */
  perTimeframe: Record<string, number>;
  /** ISO timestamp of the most recent event, or null. */
  lastObservedAtIso: string | null;
}

export function summarizeRecentEvents(
  events: readonly DisabledOutcomeEvent[],
  now: number,
  windowMs: number,
): RecentEventSummary {
  const cutoff = now - windowMs;
  const recent = events.filter((e) => e.observedAt >= cutoff);
  // Stable: ring buffer is naturally chronological. We don't sort.
  const perTimeframe: Record<string, number> = {};
  for (const e of recent) {
    perTimeframe[e.timeframe] = (perTimeframe[e.timeframe] ?? 0) + 1;
  }
  const timeframes = Object.keys(perTimeframe).sort();
  const lastObservedAt = recent.length > 0
    ? recent[recent.length - 1].observedAt
    : null;
  return {
    recent,
    count: recent.length,
    timeframes,
    perTimeframe,
    lastObservedAtIso:
      lastObservedAt !== null ? new Date(lastObservedAt).toISOString() : null,
  };
}

export function buildIncidentKey(timeframes: readonly string[]): string | null {
  if (timeframes.length === 0) return null;
  // Bucket by the sorted set of distinct timeframes — a new tf entering
  // the active window is a fresh incident; the same set continuing to
  // leak dedupes. We do NOT include the count in the key, otherwise
  // every event would re-page during a sustained leak.
  return `disabled_outcome@${[...timeframes].sort().join(",")}`;
}

export function buildIncidentReason(summary: RecentEventSummary): string {
  const tfs = summary.timeframes;
  if (tfs.length === 0) return "no events";
  const tfList = tfs
    .map((tf) => `${tf} (${summary.perTimeframe[tf]})`)
    .join(", ");
  return `${summary.count} disabled-role outcome${summary.count === 1 ? "" : "s"} rejected from timeframe${tfs.length === 1 ? "" : "s"}: ${tfList}`;
}

// ───────────────────────────── window cfg ────────────────────────────

export function resolveWindowMs(
  override?: number,
): number {
  if (typeof override === "number" && Number.isFinite(override) && override > 0) {
    return Math.max(MIN_WINDOW_MS, Math.floor(override));
  }
  const raw = process.env.DISABLED_OUTCOME_WINDOW_MINUTES;
  const parsed = raw != null ? Number(raw) : NaN;
  const minutes =
    Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_WINDOW_MINUTES;
  return Math.max(MIN_WINDOW_MS, Math.floor(minutes * 60_000));
}

// ───────────────────────────── persistence ───────────────────────────

function isState(v: unknown): v is Partial<DisabledOutcomeNotifierState> {
  return !!v && typeof v === "object";
}

async function readState(): Promise<DisabledOutcomeNotifierState> {
  try {
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, DISABLED_OUTCOME_NOTIFIER_STATE_KEY));
    if (!row?.value || !isState(row.value)) return { ...EMPTY_STATE };
    const v = row.value as Partial<DisabledOutcomeNotifierState>;
    return {
      activeIncidentKey:
        typeof v.activeIncidentKey === "string" ? v.activeIncidentKey : null,
      activeIncidentSince:
        typeof v.activeIncidentSince === "string"
          ? v.activeIncidentSince
          : null,
      activeIncidentReason:
        typeof v.activeIncidentReason === "string"
          ? v.activeIncidentReason
          : null,
      activeTimeframes:
        Array.isArray(v.activeTimeframes)
          ? v.activeTimeframes.filter((s): s is string => typeof s === "string")
          : [],
      activeEventCount:
        typeof v.activeEventCount === "number" &&
        Number.isFinite(v.activeEventCount) &&
        v.activeEventCount >= 0
          ? Math.floor(v.activeEventCount)
          : 0,
      lastAlertAt:
        typeof v.lastAlertAt === "string" ? v.lastAlertAt : null,
      lastAlertKind:
        v.lastAlertKind === "incident" || v.lastAlertKind === "recovery"
          ? v.lastAlertKind
          : null,
      lastAlertReason:
        typeof v.lastAlertReason === "string" ? v.lastAlertReason : null,
    };
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "disabled-outcome-notifier: failed to read state; treating as empty",
    );
    return { ...EMPTY_STATE };
  }
}

async function writeState(
  state: DisabledOutcomeNotifierState,
): Promise<void> {
  await db
    .insert(appSettingsTable)
    .values({ key: DISABLED_OUTCOME_NOTIFIER_STATE_KEY, value: state })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: state, updatedAt: new Date() },
    });
}

export async function getDisabledOutcomeNotifierState(): Promise<DisabledOutcomeNotifierState> {
  return await readState();
}

// ───────────────────────────── channels ──────────────────────────────

export interface DisabledOutcomeAlertChannels {
  configured: boolean;
  genericConfigured: boolean;
  slackConfigured: boolean;
}

export function getDisabledOutcomeAlertChannels(): DisabledOutcomeAlertChannels {
  const generic = !!process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL;
  const slack =
    !!process.env.DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL ||
    !!process.env.SLACK_WEBHOOK_URL;
  return {
    configured: generic || slack,
    genericConfigured: generic,
    slackConfigured: slack,
  };
}

// ───────────────────────────── payload ───────────────────────────────

export interface DisabledOutcomeAlertPayload {
  kind: "incident" | "recovery";
  reason: string;
  /** Window the dispatcher is using, in minutes. */
  windowMinutes: number;
  /** Distinct timeframes seen in the active window at evaluation time. */
  timeframes: string[];
  /** Per-timeframe rejection count in the window. */
  perTimeframe: Record<string, number>;
  /** Total events in the window. */
  eventCount: number;
  /** Up to 10 most recent offending events with the trio operators
   *  need to find the leaking caller fast. */
  recentEvents: Array<{
    tickId: string;
    sliceId: string | null;
    timeframe: string;
    observedAt: string;
  }>;
  /** ISO timestamp the active incident began, or null on recovery rows. */
  incidentSince: string | null;
}

export function formatAlertTitle(p: DisabledOutcomeAlertPayload): string {
  return p.kind === "incident"
    ? "Meta-brain rejected disabled-role outcome(s)"
    : "Meta-brain disabled-role outcome alert recovered";
}

const RECENT_EVENTS_CAP = 10;

function buildPayload(
  kind: "incident" | "recovery",
  reason: string,
  summary: RecentEventSummary,
  windowMinutes: number,
  incidentSince: string | null,
): DisabledOutcomeAlertPayload {
  const tail = summary.recent.slice(-RECENT_EVENTS_CAP);
  return {
    kind,
    reason,
    windowMinutes,
    timeframes: summary.timeframes,
    perTimeframe: summary.perTimeframe,
    eventCount: summary.count,
    recentEvents: tail.map((e) => ({
      tickId: e.tickId,
      sliceId: e.sliceId,
      timeframe: e.timeframe,
      observedAt: new Date(e.observedAt).toISOString(),
    })),
    incidentSince,
  };
}

// ───────────────────────────── dispatch ──────────────────────────────

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
  payload: DisabledOutcomeAlertPayload,
): Promise<void> {
  await postJson(url, {
    title: formatAlertTitle(payload),
    text: payload.reason,
    payload,
  });
}

async function dispatchSlack(
  url: string,
  payload: DisabledOutcomeAlertPayload,
): Promise<void> {
  const title = formatAlertTitle(payload);
  const icon =
    payload.kind === "incident" ? ":rotating_light:" : ":white_check_mark:";
  const blocks: unknown[] = [
    { type: "section", text: { type: "mrkdwn", text: `${icon} *${title}*` } },
    { type: "section", text: { type: "mrkdwn", text: payload.reason } },
  ];
  if (payload.kind === "incident" && payload.recentEvents.length > 0) {
    const lines = payload.recentEvents
      .slice(-5)
      .map(
        (e) =>
          `• \`${e.timeframe}\` tick=\`${e.tickId}\` slice=\`${e.sliceId ?? "—"}\``,
      )
      .join("\n");
    blocks.push({
      type: "section",
      text: { type: "mrkdwn", text: `Most recent leaking events:\n${lines}` },
    });
  }
  await postJson(url, {
    text: `${title} — ${payload.reason}`,
    blocks,
  });
}

export interface DispatchOptions {
  /** Override env-driven generic webhook (test seam). */
  webhookUrl?: string | null;
  /** Override env-driven Slack webhook (test seam). */
  slackWebhookUrl?: string | null;
  /** Inject "now" for deterministic tests. */
  now?: number;
  /** Override window length, ms (test seam). */
  windowMsOverride?: number;
  /** When true, skip persisting the resulting state (test seam). */
  skipPersist?: boolean;
  /** Inject a ring buffer (test seam) instead of using the process-local one. */
  eventsOverride?: readonly DisabledOutcomeEvent[];
}

export interface DisabledOutcomeWatcherTickSummary {
  status:
    | "healthy"
    | "alerted"
    | "deduped"
    | "recovered"
    | "channel_error";
  unhealthy: boolean;
  reason: string | null;
  summary: RecentEventSummary;
  dispatched: Array<{
    channel: "generic" | "slack";
    outcome: "ok" | "error" | "skipped";
    error?: string | null;
  }>;
  state: DisabledOutcomeNotifierState;
}

function resolveChannels(opts: DispatchOptions): {
  webhookUrl: string | null;
  slackWebhookUrl: string | null;
} {
  const webhookUrl =
    opts.webhookUrl !== undefined
      ? opts.webhookUrl
      : process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL || null;
  const slackWebhookUrl =
    opts.slackWebhookUrl !== undefined
      ? opts.slackWebhookUrl
      : process.env.DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL ||
        process.env.SLACK_WEBHOOK_URL ||
        null;
  return { webhookUrl, slackWebhookUrl };
}

/**
 * Single dispatcher tick. Re-evaluates the active window against the
 * persisted dedup state and fires incident / recovery webhooks when
 * the active-incident key transitions. Never throws — partial failures
 * are reported in the summary so neither the inline path nor the sweep
 * loop can crash the api-server.
 *
 * Concurrency: bursts of `recordDisabledOutcomeRejection` calls each
 * schedule a fire-and-forget tick. Without serialization those ticks
 * could read stale pre-incident state in parallel and double-page on
 * the rising edge. We serialize by chaining ticks onto a single
 * in-flight promise so reads/writes of the persistent dedup state
 * happen one-at-a-time within this process. Test paths that pass
 * `eventsOverride` skip the lock so existing tests stay deterministic.
 */
let inflightTick: Promise<DisabledOutcomeWatcherTickSummary> | null = null;

export async function runDisabledOutcomeNotifierTick(
  opts: DispatchOptions = {},
): Promise<DisabledOutcomeWatcherTickSummary> {
  // Test paths use eventsOverride to drive a deterministic event set
  // and assert the summary directly; they don't share global state
  // with concurrent callers, so they bypass the singleflight.
  if (opts.eventsOverride !== undefined) {
    return await runDisabledOutcomeNotifierTickInner(opts);
  }
  // Singleflight: chain onto the in-flight tick (if any) so the next
  // tick reads the state our current write produced. This collapses
  // bursts of rejections on the SAME incident key into one webhook.
  const next: Promise<DisabledOutcomeWatcherTickSummary> = (
    inflightTick ?? Promise.resolve(null)
  ).then(
    () => runDisabledOutcomeNotifierTickInner(opts),
    () => runDisabledOutcomeNotifierTickInner(opts),
  );
  inflightTick = next;
  try {
    return await next;
  } finally {
    // Clear only if our chained promise is still the head — another
    // caller may have already chained on top of us.
    if (inflightTick === next) inflightTick = null;
  }
}

async function runDisabledOutcomeNotifierTickInner(
  opts: DispatchOptions,
): Promise<DisabledOutcomeWatcherTickSummary> {
  const now = opts.now ?? Date.now();
  const windowMs = resolveWindowMs(opts.windowMsOverride);
  const events = opts.eventsOverride ?? ringBuffer;
  const summary = summarizeRecentEvents(events, now, windowMs);
  const incidentKey = buildIncidentKey(summary.timeframes);
  const reason = incidentKey ? buildIncidentReason(summary) : null;

  const prev = await readState();
  const nowIso = new Date(now).toISOString();
  const { webhookUrl, slackWebhookUrl } = resolveChannels(opts);

  const next: DisabledOutcomeNotifierState = { ...prev };
  const dispatched: DisabledOutcomeWatcherTickSummary["dispatched"] = [];
  let summaryStatus: DisabledOutcomeWatcherTickSummary["status"];

  if (incidentKey !== null && reason !== null) {
    if (prev.activeIncidentKey === incidentKey) {
      // Same set of leaking timeframes — refresh the bookkeeping but
      // do NOT re-page. Operators have already been notified.
      next.activeTimeframes = summary.timeframes;
      next.activeEventCount = summary.count;
      next.activeIncidentReason = reason;
      summaryStatus = "deduped";
    } else {
      // Rising edge — either no prior incident, or the leaking set
      // changed. Either way: page.
      const incidentSince = nowIso;
      const payload = buildPayload(
        "incident",
        reason,
        summary,
        Math.round(windowMs / 60_000),
        incidentSince,
      );
      let anyOk = false;
      let anyErr = false;
      if (webhookUrl) {
        try {
          await dispatchGeneric(webhookUrl, payload);
          dispatched.push({ channel: "generic", outcome: "ok" });
          anyOk = true;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          dispatched.push({ channel: "generic", outcome: "error", error: msg });
          anyErr = true;
          logger.warn(
            { err: msg, key: incidentKey },
            "disabled-outcome-notifier: generic webhook dispatch failed",
          );
        }
      } else {
        dispatched.push({ channel: "generic", outcome: "skipped" });
      }
      if (slackWebhookUrl) {
        try {
          await dispatchSlack(slackWebhookUrl, payload);
          dispatched.push({ channel: "slack", outcome: "ok" });
          anyOk = true;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          dispatched.push({ channel: "slack", outcome: "error", error: msg });
          anyErr = true;
          logger.warn(
            { err: msg, key: incidentKey },
            "disabled-outcome-notifier: slack dispatch failed",
          );
        }
      } else {
        dispatched.push({ channel: "slack", outcome: "skipped" });
      }
      // Liveliness fields refresh either way so the dashboard banner
      // shows the correct count/timeframes even while we retry.
      next.activeTimeframes = summary.timeframes;
      next.activeEventCount = summary.count;
      next.activeIncidentReason = reason;
      const allChannelsFailed = anyErr && !anyOk;
      if (allChannelsFailed) {
        // Do NOT advance activeIncidentKey on the rising edge if every
        // configured channel failed. Otherwise the next tick would
        // dedupe and operators might never receive the page despite
        // an ongoing leak. We keep prev.activeIncidentKey (or null)
        // unchanged so the next tick retries the dispatch.
        summaryStatus = "channel_error";
        logger.warn(
          {
            key: incidentKey,
            reason,
            timeframes: summary.timeframes,
            count: summary.count,
          },
          "disabled-outcome-notifier: incident dispatch failed on all channels; will retry next tick",
        );
      } else {
        next.activeIncidentKey = incidentKey;
        next.activeIncidentSince = incidentSince;
        next.lastAlertAt = nowIso;
        next.lastAlertKind = "incident";
        next.lastAlertReason = reason;
        summaryStatus = "alerted";
        logger.info(
          {
            key: incidentKey,
            reason,
            timeframes: summary.timeframes,
            count: summary.count,
          },
          "disabled-outcome-notifier: dispatched incident",
        );
      }
    }
  } else if (prev.activeIncidentKey != null) {
    // Falling edge — window is empty, recover.
    const payload = buildPayload(
      "recovery",
      "Disabled-role outcome leak recovered (no events in window)",
      summary,
      Math.round(windowMs / 60_000),
      prev.activeIncidentSince,
    );
    if (webhookUrl) {
      try {
        await dispatchGeneric(webhookUrl, payload);
        dispatched.push({ channel: "generic", outcome: "ok" });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        dispatched.push({ channel: "generic", outcome: "error", error: msg });
      }
    } else {
      dispatched.push({ channel: "generic", outcome: "skipped" });
    }
    if (slackWebhookUrl) {
      try {
        await dispatchSlack(slackWebhookUrl, payload);
        dispatched.push({ channel: "slack", outcome: "ok" });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        dispatched.push({ channel: "slack", outcome: "error", error: msg });
      }
    } else {
      dispatched.push({ channel: "slack", outcome: "skipped" });
    }
    next.activeIncidentKey = null;
    next.activeIncidentSince = null;
    next.activeIncidentReason = null;
    next.activeTimeframes = [];
    next.activeEventCount = 0;
    next.lastAlertAt = nowIso;
    next.lastAlertKind = "recovery";
    next.lastAlertReason = "recovered";
    summaryStatus = "recovered";
    logger.info(
      { recoveredFromKey: prev.activeIncidentKey },
      "disabled-outcome-notifier: dispatched recovery",
    );
  } else {
    summaryStatus = "healthy";
  }

  if (!opts.skipPersist) {
    try {
      await writeState(next);
    } catch (err) {
      logger.warn(
        { err: err instanceof Error ? err.message : String(err) },
        "disabled-outcome-notifier: failed to persist state; next tick may resend",
      );
    }
  }

  return {
    status: summaryStatus,
    unhealthy: incidentKey !== null,
    reason,
    summary,
    dispatched,
    state: next,
  };
}

// ───────────────────────────── public read ───────────────────────────

export interface DisabledOutcomeBannerState {
  /** True iff there are any events in the active window. */
  bannerVisible: boolean;
  /** Window the banner is summarizing, in minutes. */
  windowMinutes: number;
  /** Distinct timeframes leaking in the window, sorted. */
  timeframes: string[];
  /** Per-timeframe rejection count in the window. */
  perTimeframe: Record<string, number>;
  /** Total events in the window. */
  eventCount: number;
  /** Most recent up-to-10 events for the banner card. */
  recentEvents: Array<{
    tickId: string;
    sliceId: string | null;
    timeframe: string;
    observedAt: string;
  }>;
  /** ISO timestamp of the most recent rejection, or null when none. */
  lastObservedAt: string | null;
  /** The dedup state from the persistent store. */
  notifier: DisabledOutcomeNotifierState;
  /** Webhook channel configuration for the banner footer. */
  alertHook: DisabledOutcomeAlertChannels & {
    activeIncident: boolean;
    activeIncidentReason: string | null;
    activeIncidentSince: string | null;
    lastAlertAt: string | null;
    lastAlertKind: "incident" | "recovery" | null;
    lastAlertReason: string | null;
  };
}

export async function getDisabledOutcomeBannerState(opts: {
  now?: number;
  windowMsOverride?: number;
  eventsOverride?: readonly DisabledOutcomeEvent[];
} = {}): Promise<DisabledOutcomeBannerState> {
  const now = opts.now ?? Date.now();
  const windowMs = resolveWindowMs(opts.windowMsOverride);
  const events = opts.eventsOverride ?? ringBuffer;
  const summary = summarizeRecentEvents(events, now, windowMs);
  const notifier = await readState();
  const channels = getDisabledOutcomeAlertChannels();
  const tail = summary.recent.slice(-RECENT_EVENTS_CAP);
  return {
    bannerVisible: summary.count > 0,
    windowMinutes: Math.round(windowMs / 60_000),
    timeframes: summary.timeframes,
    perTimeframe: summary.perTimeframe,
    eventCount: summary.count,
    recentEvents: tail.map((e) => ({
      tickId: e.tickId,
      sliceId: e.sliceId,
      timeframe: e.timeframe,
      observedAt: new Date(e.observedAt).toISOString(),
    })),
    lastObservedAt: summary.lastObservedAtIso,
    notifier,
    alertHook: {
      ...channels,
      activeIncident: notifier.activeIncidentKey != null,
      activeIncidentReason: notifier.activeIncidentReason,
      activeIncidentSince: notifier.activeIncidentSince,
      lastAlertAt: notifier.lastAlertAt,
      lastAlertKind: notifier.lastAlertKind,
      lastAlertReason: notifier.lastAlertReason,
    },
  };
}

// ───────────────────────────── sweep loop ────────────────────────────

let pollInterval: ReturnType<typeof setInterval> | null = null;

/**
 * Start the periodic sweep. Idempotent. The sweep handles the recovery
 * dispatch (which the inline push path can't do — there's no event to
 * push when the window finally clears). Default 60s mirrors the other
 * notifier loops.
 */
export function startDisabledOutcomeNotifierLoop(intervalMs = 60_000): void {
  if (pollInterval) return;
  pollInterval = setInterval(() => {
    void runDisabledOutcomeNotifierTick().catch((err) =>
      logger.error(
        { err },
        "disabled-outcome-notifier: unexpected dispatch error",
      ),
    );
  }, intervalMs);
  logger.info({ intervalMs }, "disabled-outcome-notifier loop started");
}

/** Test seam — stop the loop. */
export function stopDisabledOutcomeNotifierLoop(): void {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}
