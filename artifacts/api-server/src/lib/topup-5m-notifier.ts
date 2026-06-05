/**
 * Task #411 — off-dashboard alerting + banner state for the daily 5m
 * head top-up scheduler (#410).
 *
 * Background: `scheduled_5m_topup.py` runs once per `INTERVAL_SECONDS`
 * (24h default) inside the ml-engine. It writes a structured warning
 * to the ml-engine logs and a JSONL progress entry whenever any
 * coin's contiguous_days falls below `ALERT_BELOW_DAYS` (310 default
 * — one day of headroom over the 305-day hard gate). The current
 * surface is `GET /ml/admin/5m-topup/status` and the log line. Both
 * require an operator to be either tailing logs or watching the
 * dashboard.
 *
 * This module closes the loop. A 60s watcher in the api-server polls
 * the same status endpoint and dispatches an off-dashboard alert when:
 *   • `last_attempt_outcome == "error"` for two consecutive scheduler
 *     ticks (i.e. two real days running) — the daily top-up has
 *     stopped advancing the head and contiguous_days WILL erode if we
 *     don't intervene; OR
 *   • `last_alerts` is non-empty — at least one coin is already below
 *     the alert threshold and headed toward the gate.
 *
 * Channels are env-driven and opt-in:
 *   `TOPUP_5M_ALERT_WEBHOOK_URL`        (generic JSON webhook)
 *   `TOPUP_5M_ALERT_SLACK_WEBHOOK_URL`  (Slack incoming-webhook;
 *                                        falls back to
 *                                        `SLACK_WEBHOOK_URL`)
 * With no env set the watcher still tracks state so the dashboard
 * banner is still accurate, but does not POST anywhere.
 *
 * Dedup: the active incident key is held in
 * `app_settings.topup_5m_notifier.state`. The key changes when the
 * set of below-threshold coins changes OR when an error streak ends
 * and a fresh streak starts, so transient flaps page once but do not
 * spam. A recovery (back to OK with no alerts) fires a one-shot
 * recovery ping and clears the active key.
 *
 * "Two consecutive ticks" is measured against the scheduler's own
 * `last_check_at` — we only advance the consecutive counter when a
 * NEW ml-engine tick is observed (so the 60s watcher poll cadence
 * doesn't inflate the streak past the real 24h cadence).
 */
import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";

export const TOPUP_5M_NOTIFIER_STATE_KEY = "topup_5m_notifier.state";

/** Threshold for the "two consecutive failed ticks" condition. The task
 *  spec is explicit at 2; exposed as a const so tests can read the same
 *  value the production path enforces. */
export const ERROR_STREAK_THRESHOLD = 2;

/** Mirror of the ml-engine `scheduled_5m_topup.state` shape we depend
 *  on. Other fields exist but we only consume what we need so a future
 *  ml-engine field addition does not break the type. */
export interface TopupStatus {
  enabled?: boolean;
  alert_below_days?: number | null;
  last_check_at?: number | null;
  last_attempt_outcome?:
    | "disabled"
    | "skipped_busy"
    | "ok"
    | "error"
    | null;
  last_finished_at?: number | null;
  last_error?: string | null;
  last_alerts?: string[] | null;
  last_health_per_coin?: Record<string, number> | null;
  ticks_total?: number | null;
  runs_total?: number | null;
  // Task #442 — stuck-replica detection. The ml-engine status endpoint
  // overlays the fleet-wide streak from the shared recent_winners row
  // so every replica reports the same values regardless of which one
  // we polled. `stuck_replica` is non-null exactly when the head
  // replica's consecutive-win count is at-or-above the configured
  // threshold; below that the ml-engine leaves it null so we do not
  // have to re-implement the threshold check here.
  stuck_replica?: string | null;
  stuck_replica_streak?: number | null;
  stuck_replica_threshold?: number | null;
}

export interface Topup5mNotifierState {
  /** Stable identity of the current outage. Null when healthy. */
  activeIncidentKey: string | null;
  /** Reason summary for the active incident (for dashboard surfacing). */
  activeIncidentReason: string | null;
  /** ISO timestamp when the active incident was first observed. */
  activeIncidentSince: string | null;
  /** ISO timestamp of the last alert (incident or recovery) we sent. */
  lastAlertAt: string | null;
  /** Last alert kind. */
  lastAlertKind: "incident" | "recovery" | null;
  /** Last alert reason summary. */
  lastAlertReason: string | null;
  /**
   * Number of consecutive `last_attempt_outcome == "error"` scheduler
   * ticks observed. A "tick" here is a unique `last_check_at` value
   * — the 60s watcher poll never bumps this on its own, only a fresh
   * scheduler tick does. Reset to 0 by any non-error outcome.
   */
  consecutiveErrors: number;
  /**
   * The most recent scheduler `last_check_at` we have already accounted
   * for in `consecutiveErrors`. Prevents the 60s watcher from
   * double-counting the same tick.
   */
  lastObservedCheckAt: number | null;
}

const EMPTY_STATE: Topup5mNotifierState = {
  activeIncidentKey: null,
  activeIncidentReason: null,
  activeIncidentSince: null,
  lastAlertAt: null,
  lastAlertKind: null,
  lastAlertReason: null,
  consecutiveErrors: 0,
  lastObservedCheckAt: null,
};

function isState(v: unknown): v is Partial<Topup5mNotifierState> {
  return !!v && typeof v === "object";
}

async function readState(): Promise<Topup5mNotifierState> {
  try {
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, TOPUP_5M_NOTIFIER_STATE_KEY));
    if (!row?.value || !isState(row.value)) return { ...EMPTY_STATE };
    const v = row.value as Partial<Topup5mNotifierState>;
    return {
      activeIncidentKey:
        typeof v.activeIncidentKey === "string" ? v.activeIncidentKey : null,
      activeIncidentReason:
        typeof v.activeIncidentReason === "string"
          ? v.activeIncidentReason
          : null,
      activeIncidentSince:
        typeof v.activeIncidentSince === "string"
          ? v.activeIncidentSince
          : null,
      lastAlertAt: typeof v.lastAlertAt === "string" ? v.lastAlertAt : null,
      lastAlertKind:
        v.lastAlertKind === "incident" || v.lastAlertKind === "recovery"
          ? v.lastAlertKind
          : null,
      lastAlertReason:
        typeof v.lastAlertReason === "string" ? v.lastAlertReason : null,
      consecutiveErrors:
        typeof v.consecutiveErrors === "number" &&
        Number.isFinite(v.consecutiveErrors) &&
        v.consecutiveErrors >= 0
          ? Math.floor(v.consecutiveErrors)
          : 0,
      lastObservedCheckAt:
        typeof v.lastObservedCheckAt === "number" &&
        Number.isFinite(v.lastObservedCheckAt)
          ? v.lastObservedCheckAt
          : null,
    };
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "topup-5m-notifier: failed to read state; treating as empty",
    );
    return { ...EMPTY_STATE };
  }
}

async function writeState(state: Topup5mNotifierState): Promise<void> {
  await db
    .insert(appSettingsTable)
    .values({ key: TOPUP_5M_NOTIFIER_STATE_KEY, value: state })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: state, updatedAt: new Date() },
    });
}

export async function getTopup5mNotifierState(): Promise<Topup5mNotifierState> {
  return await readState();
}

/** Public read helper for the dashboard surface. */
export interface Topup5mAlertChannels {
  configured: boolean;
  genericConfigured: boolean;
  slackConfigured: boolean;
}

export function getTopup5mAlertChannels(): Topup5mAlertChannels {
  const generic = !!process.env.TOPUP_5M_ALERT_WEBHOOK_URL;
  const slack =
    !!process.env.TOPUP_5M_ALERT_SLACK_WEBHOOK_URL ||
    !!process.env.SLACK_WEBHOOK_URL;
  return {
    configured: generic || slack,
    genericConfigured: generic,
    slackConfigured: slack,
  };
}

export interface Topup5mAlertPayload {
  kind: "incident" | "recovery";
  reason: string;
  /** Coins below the alert threshold at evaluation time. */
  alertCoins: string[];
  /** Per-coin contiguous_days for the below-threshold coins. */
  perCoinContiguousDays: Record<string, number>;
  /** Threshold used by the scheduler (e.g. 310 days). */
  thresholdDays: number | null;
  /** Consecutive failed scheduler ticks seen at evaluation time. */
  consecutiveErrors: number;
  /** Most recent scheduler outcome string. */
  lastAttemptOutcome: TopupStatus["last_attempt_outcome"];
  /** ISO timestamp of the most recent scheduler tick (or null). */
  lastCheckAt: string | null;
  /** Last scheduler error message (truncated by the engine). */
  lastError: string | null;
  /** ISO timestamp the active incident began (or null on recovery rows). */
  incidentSince: string | null;
  /** Task #442 — replica that has won the daily pull `stuckReplicaStreak`
   *  ticks in a row, or null when the streak is below the threshold or
   *  the incident is not a stuck-replica one. */
  stuckReplica: string | null;
  /** Current head replica's consecutive-win count at evaluation time. */
  stuckReplicaStreak: number | null;
  /** Threshold the ml-engine was configured with (e.g. 7 days). */
  stuckReplicaThreshold: number | null;
}

/** Pure helper: classify a status snapshot + previous state into the
 *  next active-incident shape. Tested directly. */
export function evaluateTopupStatus(
  status: TopupStatus,
  prev: Topup5mNotifierState,
): {
  unhealthy: boolean;
  reason: string | null;
  incidentKey: string | null;
  /** Coins below threshold (sorted, used for the dashboard banner). */
  alertCoins: string[];
  /** Consecutive error count after accounting for THIS status snapshot.
   *  Not yet persisted — caller decides whether to write. */
  nextConsecutiveErrors: number;
  /** Whether this snapshot represents a fresh scheduler tick (i.e. a
   *  `last_check_at` we haven't seen before). Drives both
   *  consecutive-error accounting and the "noop on stale poll" path. */
  freshTick: boolean;
} {
  const lastCheckAt =
    typeof status.last_check_at === "number" ? status.last_check_at : null;
  const freshTick =
    lastCheckAt != null &&
    (prev.lastObservedCheckAt == null ||
      lastCheckAt > prev.lastObservedCheckAt);
  const outcome = status.last_attempt_outcome ?? null;

  // Consecutive-error counter — only advance on a fresh scheduler tick.
  let nextConsecutiveErrors = prev.consecutiveErrors;
  if (freshTick) {
    if (outcome === "error") {
      nextConsecutiveErrors = prev.consecutiveErrors + 1;
    } else if (outcome === "ok" || outcome === "skipped_busy") {
      // "disabled" intentionally does NOT reset the streak — a turned-off
      // scheduler shouldn't silently clear an existing incident.
      nextConsecutiveErrors = 0;
    }
  }

  const alertCoins = Array.isArray(status.last_alerts)
    ? [...status.last_alerts].filter((c): c is string => typeof c === "string").sort()
    : [];

  // Two conditions, errors take precedence (it's the more severe failure).
  if (nextConsecutiveErrors >= ERROR_STREAK_THRESHOLD) {
    const errMsg = status.last_error ? `: ${status.last_error}` : "";
    return {
      unhealthy: true,
      reason: `Daily 5m top-up tick has failed ${nextConsecutiveErrors} times in a row${errMsg}`,
      // Bucket by the count itself so a flap (error → ok → error) starts
      // a new incident key when the count resets and re-crosses 2. We
      // intentionally DON'T include lastCheckAt in the key — every new
      // failed tick during the same outage would otherwise re-page.
      incidentKey: "errors@active",
      alertCoins,
      nextConsecutiveErrors,
      freshTick,
    };
  }

  if (alertCoins.length > 0) {
    return {
      unhealthy: true,
      reason: `5m contiguous_days below threshold for ${alertCoins.length} coin(s): ${alertCoins.join(", ")}`,
      // Bucket by the sorted coin list — a different set of coins is a
      // separate incident worth re-paging on.
      incidentKey: `alerts@${alertCoins.join(",")}`,
      alertCoins,
      nextConsecutiveErrors,
      freshTick,
    };
  }

  // Task #442 — stuck-replica condition. The ml-engine has already
  // applied the threshold check before populating `stuck_replica`, so
  // its presence alone signals "alert". The incident key is bucketed
  // by the replica name (and not the streak) so a continued streak
  // dedupes after the first page; an operator who muted the alert by
  // restarting the stuck box will see the incident clear naturally
  // when a different replica next wins. Lower precedence than the
  // error and alert-coin branches above — those are direct gate-erosion
  // failures, while a stuck replica is a fairness/availability concern.
  const stuckReplica =
    typeof status.stuck_replica === "string" && status.stuck_replica.length > 0
      ? status.stuck_replica
      : null;
  if (stuckReplica !== null) {
    const streak =
      typeof status.stuck_replica_streak === "number" &&
      Number.isFinite(status.stuck_replica_streak)
        ? status.stuck_replica_streak
        : 0;
    const threshold =
      typeof status.stuck_replica_threshold === "number" &&
      Number.isFinite(status.stuck_replica_threshold)
        ? status.stuck_replica_threshold
        : null;
    const thresholdSuffix = threshold !== null ? ` (threshold: ${threshold})` : "";
    return {
      unhealthy: true,
      reason: `Replica ${stuckReplica} has won the daily 5m top-up ${streak} ticks in a row${thresholdSuffix}`,
      incidentKey: `stuck_replica@${stuckReplica}`,
      alertCoins,
      nextConsecutiveErrors,
      freshTick,
    };
  }

  return {
    unhealthy: false,
    reason: null,
    incidentKey: null,
    alertCoins,
    nextConsecutiveErrors,
    freshTick,
  };
}

function buildPayload(
  kind: "incident" | "recovery",
  reason: string,
  status: TopupStatus,
  alertCoins: string[],
  consecutiveErrors: number,
  incidentSince: string | null,
): Topup5mAlertPayload {
  const lastCheckAt =
    typeof status.last_check_at === "number"
      ? new Date(status.last_check_at * 1000).toISOString()
      : null;
  const perCoin: Record<string, number> = {};
  if (status.last_health_per_coin && typeof status.last_health_per_coin === "object") {
    for (const c of alertCoins) {
      const v = status.last_health_per_coin[c];
      if (typeof v === "number" && Number.isFinite(v)) perCoin[c] = v;
    }
  }
  return {
    kind,
    reason,
    alertCoins,
    perCoinContiguousDays: perCoin,
    thresholdDays:
      typeof status.alert_below_days === "number" ? status.alert_below_days : null,
    consecutiveErrors,
    lastAttemptOutcome: status.last_attempt_outcome ?? null,
    lastCheckAt,
    lastError: status.last_error ?? null,
    incidentSince,
    stuckReplica:
      typeof status.stuck_replica === "string" && status.stuck_replica.length > 0
        ? status.stuck_replica
        : null,
    stuckReplicaStreak:
      typeof status.stuck_replica_streak === "number" &&
      Number.isFinite(status.stuck_replica_streak)
        ? status.stuck_replica_streak
        : null,
    stuckReplicaThreshold:
      typeof status.stuck_replica_threshold === "number" &&
      Number.isFinite(status.stuck_replica_threshold)
        ? status.stuck_replica_threshold
        : null,
  };
}

export function formatAlertTitle(p: Topup5mAlertPayload): string {
  return p.kind === "incident"
    ? "5m top-up scheduler — alert"
    : "5m top-up scheduler — recovered";
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
  payload: Topup5mAlertPayload,
): Promise<void> {
  await postJson(url, {
    title: formatAlertTitle(payload),
    text: payload.reason,
    payload,
  });
}

async function dispatchSlack(
  url: string,
  payload: Topup5mAlertPayload,
): Promise<void> {
  const title = formatAlertTitle(payload);
  const icon =
    payload.kind === "incident" ? ":rotating_light:" : ":white_check_mark:";
  await postJson(url, {
    text: `${title} — ${payload.reason}`,
    blocks: [
      { type: "section", text: { type: "mrkdwn", text: `${icon} *${title}*` } },
      { type: "section", text: { type: "mrkdwn", text: payload.reason } },
    ],
  });
}

export interface DispatchOptions {
  /** Override env-driven generic webhook (test seam). */
  webhookUrl?: string | null;
  /** Override env-driven Slack webhook (test seam). */
  slackWebhookUrl?: string | null;
  /** Inject the ml-engine status snapshot (test seam). */
  statusOverride?: TopupStatus;
  /** Inject "now" for deterministic tests. */
  now?: number;
  /** When true, skip persisting state (test seam). */
  skipPersist?: boolean;
}

export interface Topup5mWatcherTickSummary {
  status: "healthy" | "alerted" | "deduped" | "recovered" | "channel_error" | "skipped";
  unhealthy: boolean;
  reason: string | null;
  freshTick: boolean;
  dispatched: Array<{
    channel: "generic" | "slack";
    outcome: "ok" | "error" | "skipped";
    error?: string | null;
  }>;
  state: Topup5mNotifierState;
}

function resolveChannelUrls(opts: DispatchOptions): {
  webhookUrl: string | null;
  slackWebhookUrl: string | null;
} {
  const webhookUrl =
    opts.webhookUrl !== undefined
      ? opts.webhookUrl
      : process.env.TOPUP_5M_ALERT_WEBHOOK_URL || null;
  const slackWebhookUrl =
    opts.slackWebhookUrl !== undefined
      ? opts.slackWebhookUrl
      : process.env.TOPUP_5M_ALERT_SLACK_WEBHOOK_URL ||
        process.env.SLACK_WEBHOOK_URL ||
        null;
  return { webhookUrl, slackWebhookUrl };
}

async function fetchTopupStatus(): Promise<TopupStatus> {
  const base = (process.env.ML_ENGINE_URL || "http://localhost:8000").replace(
    /\/$/,
    "",
  );
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 5_000);
  try {
    const res = await fetch(`${base}/ml/admin/5m-topup/status`, {
      signal: ctrl.signal,
    });
    if (!res.ok) {
      throw new Error(`ml-engine /ml/admin/5m-topup/status ${res.status}`);
    }
    return (await res.json()) as TopupStatus;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Single watcher tick. Polls the ml-engine status, advances the
 * consecutive-error counter when a fresh scheduler tick is observed,
 * dispatches alerts (incident / recovery) when the active-incident
 * key changes, persists the new state. Never throws — partial
 * failures are reported in the summary so the scheduled loop never
 * crashes the api-server.
 */
export async function runTopup5mNotifierTick(
  opts: DispatchOptions = {},
): Promise<Topup5mWatcherTickSummary> {
  let status: TopupStatus;
  try {
    status = opts.statusOverride ?? (await fetchTopupStatus());
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "topup-5m-notifier: failed to fetch ml-engine status",
    );
    return {
      status: "skipped",
      unhealthy: false,
      reason: "fetch_failed",
      freshTick: false,
      dispatched: [],
      state: await readState(),
    };
  }

  const prev = await readState();
  const evalResult = evaluateTopupStatus(status, prev);
  const now = opts.now ?? Date.now();
  const nowIso = new Date(now).toISOString();

  const { webhookUrl, slackWebhookUrl } = resolveChannelUrls(opts);

  // Build the next state mirror so we always persist the consecutive
  // counter + last observed tick, even on dedupe / healthy paths.
  const next: Topup5mNotifierState = {
    ...prev,
    consecutiveErrors: evalResult.nextConsecutiveErrors,
    lastObservedCheckAt: evalResult.freshTick
      ? status.last_check_at ?? prev.lastObservedCheckAt
      : prev.lastObservedCheckAt,
  };

  const dispatched: Topup5mWatcherTickSummary["dispatched"] = [];
  let summaryStatus: Topup5mWatcherTickSummary["status"];

  // Active-incident transition logic — same shape as the
  // market-signals-poller-watcher: dispatch on rising edge, recovery
  // on falling edge, dedupe in between.
  if (evalResult.unhealthy && evalResult.incidentKey) {
    if (prev.activeIncidentKey === evalResult.incidentKey) {
      summaryStatus = "deduped";
    } else {
      // Rising edge or key change — fire incident alert. A different
      // incident key (e.g. alert coin set changed, or an error streak
      // started after we recovered) is a fresh incident with its own
      // `since` timestamp; only carry the prior `since` forward when
      // the key is genuinely the same one (which the dedupe branch
      // above already handles).
      const incidentSince = nowIso;
      const payload = buildPayload(
        "incident",
        evalResult.reason ?? "5m top-up alert",
        status,
        evalResult.alertCoins,
        evalResult.nextConsecutiveErrors,
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
            { err: msg, key: evalResult.incidentKey },
            "topup-5m-notifier: generic webhook dispatch failed",
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
            { err: msg, key: evalResult.incidentKey },
            "topup-5m-notifier: slack dispatch failed",
          );
        }
      } else {
        dispatched.push({ channel: "slack", outcome: "skipped" });
      }
      // Mark active unconditionally — same posture as the other
      // notifiers in this codebase. We loudly log dispatch errors above
      // and the operator can still see the incident in the dashboard
      // banner; we don't want to retry-loop on a broken webhook.
      next.activeIncidentKey = evalResult.incidentKey;
      next.activeIncidentReason = evalResult.reason;
      next.activeIncidentSince = incidentSince;
      next.lastAlertAt = nowIso;
      next.lastAlertKind = "incident";
      next.lastAlertReason = evalResult.reason;
      summaryStatus = anyErr && !anyOk ? "channel_error" : "alerted";
      logger.info(
        {
          key: evalResult.incidentKey,
          reason: evalResult.reason,
          alertCoins: evalResult.alertCoins,
          consecutiveErrors: evalResult.nextConsecutiveErrors,
        },
        "topup-5m-notifier: dispatched incident",
      );
    }
  } else if (!evalResult.unhealthy && prev.activeIncidentKey != null) {
    // Falling edge — recovery.
    const payload = buildPayload(
      "recovery",
      "5m top-up recovered",
      status,
      [],
      evalResult.nextConsecutiveErrors,
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
    next.activeIncidentReason = null;
    next.activeIncidentSince = null;
    next.lastAlertAt = nowIso;
    next.lastAlertKind = "recovery";
    next.lastAlertReason = "recovered";
    summaryStatus = "recovered";
    logger.info(
      { recoveredFromKey: prev.activeIncidentKey },
      "topup-5m-notifier: dispatched recovery",
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
        "topup-5m-notifier: failed to persist state; next tick may resend",
      );
    }
  }

  return {
    status: summaryStatus,
    unhealthy: evalResult.unhealthy,
    reason: evalResult.reason,
    freshTick: evalResult.freshTick,
    dispatched,
    state: next,
  };
}

let pollInterval: ReturnType<typeof setInterval> | null = null;

/** Start the periodic poll. Idempotent. */
export function startTopup5mNotifierLoop(intervalMs = 60_000): void {
  if (pollInterval) return;
  pollInterval = setInterval(() => {
    void runTopup5mNotifierTick().catch((err) =>
      logger.error(
        { err },
        "topup-5m-notifier: unexpected dispatch error",
      ),
    );
  }, intervalMs);
  logger.info({ intervalMs }, "topup-5m-notifier loop started");
}

/** Test seam — stop the loop. */
export function stopTopup5mNotifierLoop(): void {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}
