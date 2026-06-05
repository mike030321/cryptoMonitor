/**
 * Task #290 — off-dashboard alerting for the market signals poller.
 *
 * The market-signals health card (#285) surfaces poller staleness and
 * upstream errors on the crypto-monitor dashboard, but operators only
 * see it when the dashboard is open. This module closes the gap with a
 * background watcher that polls `getMarketSignalsPollerStatus()` and
 * fires an off-dashboard webhook when the poller has not written rows
 * for >3x the configured poll interval, or when the most recent poll
 * errored.
 *
 * Debounce contract: a single outage produces a single alert. The
 * "active incident" key is held in `app_settings.market_signals_watcher`
 * so a backend restart mid-outage does not re-fire the same alert. When
 * the poller recovers we send a one-shot recovery ping and clear the
 * incident, so the next outage will alert again.
 *
 * Channels are opt-in via env vars
 *   `MARKET_SIGNALS_ALERT_WEBHOOK_URL`        (generic JSON webhook)
 *   `MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL`  (Slack incoming-webhook;
 *                                              falls back to
 *                                              `SLACK_WEBHOOK_URL`)
 * With no env set the watcher still tracks state so dashboard surfacing
 * is accurate, but does not POST anywhere.
 */
import { db, appSettingsTable, marketSignalsTable } from "@workspace/db";
import { eq, gte, sql } from "drizzle-orm";
import { logger } from "./logger";
import {
  getMarketSignalsPollerStatus,
  getMarketSignalsPollerTargets,
} from "./market-signals-poller";

export const MARKET_SIGNALS_WATCHER_STATE_KEY = "market_signals_watcher.state";
export const MARKET_SIGNALS_WATCHER_HISTORY_KEY = "market_signals_watcher.history";
export const MARKET_SIGNALS_WATCHER_SNOOZE_KEY = "market_signals_watcher.snooze";

/**
 * Task #303 — bounded ring of recent alert events (incidents + recoveries)
 * so the dashboard can show a flapping history. Cap is intentionally
 * larger than the "last 20" we display so operators can scroll back a
 * little when correlating to deploys / OKX incidents without bloating
 * the app_settings row.
 */
export const MARKET_SIGNALS_WATCHER_HISTORY_MAX = 50;

export interface MarketSignalsAlertHistoryEntry {
  /** ISO timestamp the alert was dispatched. */
  at: string;
  /** "incident" when the watcher first saw an outage, "recovery" on clear. */
  kind: "incident" | "recovery";
  /** Human reason summary (incident reason or recovery-from text). */
  reason: string;
  /** Stable incident key if known — useful for grouping flaps. */
  incidentKey: string | null;
  /** ISO timestamp of when the incident began (null for recovery rows). */
  incidentSince: string | null;
}

export interface MarketSignalsWatcherState {
  /** Stable identity of the current outage. Null when poller is healthy. */
  activeIncidentKey: string | null;
  /** Reason summary for the active incident — for dashboard surfacing. */
  activeIncidentReason: string | null;
  /** ISO timestamp when the active incident was first observed. */
  activeIncidentSince: string | null;
  /** ISO timestamp of the last alert (incident or recovery) we dispatched. */
  lastAlertAt: string | null;
  /** What the last alert was about — "incident" | "recovery" | null. */
  lastAlertKind: "incident" | "recovery" | null;
  /** Last alert reason summary (incident reason or "recovered"). */
  lastAlertReason: string | null;
  /**
   * Task #302 — per-coin "silent stream" sub-incidents. The whole
   * poller can look healthy (rows landing for most coins, no error)
   * while a single coin's stream silently breaks for hours. We track
   * one debounced sub-incident per affected coin so a fixed coin's
   * recovery clears its own key without touching others.
   *
   * Map key is the coin id; value is the ISO timestamp the sub-incident
   * was first observed. A coin id present here means an "incident"
   * alert has already been dispatched for it and further ticks while
   * still silent must dedupe. When the coin starts writing rows again
   * we send a one-shot recovery and remove the key.
   */
  silentCoinIncidents: Record<string, { since: string }>;
  /** ISO timestamp of the last per-coin alert dispatched. */
  lastSilentCoinAlertAt: string | null;
  lastSilentCoinAlertKind: "incident" | "recovery" | null;
  lastSilentCoinAlertReason: string | null;
  /**
   * Task #301 — When a snooze is active and the poller is unhealthy, we
   * still want to surface the live incident on the dashboard but we must
   * NOT touch `activeIncidentKey` (which is the dedupe key for actual
   * dispatches). These pending fields capture the latest observed
   * unhealthy state during a snooze for surfacing only.
   */
  pendingIncidentKey: string | null;
  pendingIncidentReason: string | null;
  pendingIncidentSince: string | null;
}

const EMPTY_STATE: MarketSignalsWatcherState = {
  activeIncidentKey: null,
  activeIncidentReason: null,
  activeIncidentSince: null,
  lastAlertAt: null,
  lastAlertKind: null,
  lastAlertReason: null,
  silentCoinIncidents: {},
  lastSilentCoinAlertAt: null,
  lastSilentCoinAlertKind: null,
  lastSilentCoinAlertReason: null,
  pendingIncidentKey: null,
  pendingIncidentReason: null,
  pendingIncidentSince: null,
};

/**
 * Task #301 — operator-controlled mute for the alert hook. While a snooze
 * is active the watcher continues to *evaluate* poller health (so the
 * dashboard reflects reality), but it does not POST to any channel.
 * Persisted in `app_settings` so a backend restart during a maintenance
 * window does not silently un-mute.
 */
export type SnoozeDuration = "15m" | "1h" | "until_midnight";

export interface MarketSignalsWatcherSnooze {
  /** ISO timestamp when the snooze was created. */
  snoozedAt: string;
  /** ISO timestamp when the snooze expires. */
  snoozedUntil: string;
  /** Operator-selected duration label (for surfacing). */
  duration: SnoozeDuration;
}

export interface MarketSignalsAlertPayload {
  kind: "incident" | "recovery";
  /**
   * Task #302 — `"poller"` is the original whole-poller alert (#290).
   * `"coin"` is a per-coin silent-stream sub-incident; `coinId` is set.
   */
  scope: "poller" | "coin";
  coinId?: string | null;
  /**
   * Task #302 — for `scope: "coin"` alerts, the count of rows the coin
   * wrote in the trailing hour (always 0 for an incident, >=1 for a
   * recovery snapshot).
   */
  rowsLastHour?: number;
  reason: string;
  pollerStatus: {
    lastPollAt: string | null;
    lastPollOk: boolean;
    lastPollError: string | null;
    intervalMs: number;
    staleThresholdMs: number;
    isStale: boolean;
  };
  incidentSince: string | null;
}

export interface MarketSignalsWatcherTickSummary {
  status:
    | "healthy"
    | "alerted"
    | "deduped"
    | "recovered"
    | "channel_error"
    /** Suppressed by an operator snooze; tracked but not dispatched. */
    | "snoozed"
    /** Snooze active and poller is healthy. */
    | "snoozed_healthy"
    /** Snooze active and poller recovered (silent clear, no recovery ping). */
    | "snoozed_recovered";
  unhealthy: boolean;
  reason: string | null;
  dispatched: Array<{
    channel: "generic" | "slack";
    outcome: "ok" | "error" | "skipped";
    error?: string | null;
  }>;
  state: MarketSignalsWatcherState;
  /**
   * Task #302 — per-coin silent-stream sub-incident summary. Always
   * present so the route surface and tests can branch on it without
   * defensive checks.
   */
  silentCoins: {
    /** Coins whose row count over the trailing hour is zero. */
    silent: string[];
    /** Coins whose sub-incident fired this tick. */
    newIncidents: string[];
    /** Coins whose recovery fired this tick. */
    recovered: string[];
    /** Coins whose sub-incident was already active and is being deduped. */
    deduped: string[];
    /**
     * True when the whole-poller incident is active so per-coin
     * sub-incidents are intentionally suppressed (otherwise every coin
     * would page individually for the same global outage).
     */
    suppressedByGlobalIncident: boolean;
    dispatched: Array<{
      coinId: string;
      kind: "incident" | "recovery";
      channel: "generic" | "slack";
      outcome: "ok" | "error" | "skipped";
      error?: string | null;
    }>;
  };
  snooze: MarketSignalsWatcherSnooze | null;
}

interface PollerLikeStatus {
  lastPollAt: number | null;
  lastPollOk: boolean;
  lastPollError: string | null;
  intervalMs: number;
}

function isWatcherState(v: unknown): v is Partial<MarketSignalsWatcherState> {
  return !!v && typeof v === "object";
}

function isSnoozeRecord(v: unknown): v is Partial<MarketSignalsWatcherSnooze> {
  return !!v && typeof v === "object";
}

const SNOOZE_DURATIONS: ReadonlySet<SnoozeDuration> = new Set([
  "15m",
  "1h",
  "until_midnight",
]);

export function isSnoozeDuration(v: unknown): v is SnoozeDuration {
  return typeof v === "string" && SNOOZE_DURATIONS.has(v as SnoozeDuration);
}

/**
 * Compute the absolute expiry for a snooze duration. "until_midnight"
 * is the next local-midnight boundary in the server's TZ. We accept a
 * `now` for deterministic tests.
 */
export function computeSnoozeUntil(duration: SnoozeDuration, now: number = Date.now()): Date {
  if (duration === "15m") return new Date(now + 15 * 60_000);
  if (duration === "1h") return new Date(now + 60 * 60_000);
  // until_midnight — next 00:00 local time strictly after `now`.
  const d = new Date(now);
  d.setHours(24, 0, 0, 0);
  return d;
}

async function readSnoozeRaw(): Promise<MarketSignalsWatcherSnooze | null> {
  try {
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_SNOOZE_KEY));
    if (!row?.value || !isSnoozeRecord(row.value)) return null;
    const v = row.value as Partial<MarketSignalsWatcherSnooze>;
    if (
      typeof v.snoozedAt !== "string" ||
      typeof v.snoozedUntil !== "string" ||
      !isSnoozeDuration(v.duration)
    ) {
      return null;
    }
    return {
      snoozedAt: v.snoozedAt,
      snoozedUntil: v.snoozedUntil,
      duration: v.duration,
    };
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "market-signals-watcher: failed to read snooze; treating as unmuted",
    );
    return null;
  }
}

/**
 * Read the snooze and return it only when *currently* active. Expired
 * snoozes are returned as null so the watcher resumes dispatching.
 */
export async function readActiveSnooze(now: number = Date.now()): Promise<MarketSignalsWatcherSnooze | null> {
  const s = await readSnoozeRaw();
  if (!s) return null;
  const until = Date.parse(s.snoozedUntil);
  if (!Number.isFinite(until) || until <= now) return null;
  return s;
}

/** Set the snooze. Overwrites any existing snooze. */
export async function setMarketSignalsWatcherSnooze(
  duration: SnoozeDuration,
  now: number = Date.now(),
): Promise<MarketSignalsWatcherSnooze> {
  const snoozedUntil = computeSnoozeUntil(duration, now);
  const record: MarketSignalsWatcherSnooze = {
    snoozedAt: new Date(now).toISOString(),
    snoozedUntil: snoozedUntil.toISOString(),
    duration,
  };
  await db
    .insert(appSettingsTable)
    .values({ key: MARKET_SIGNALS_WATCHER_SNOOZE_KEY, value: record })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: record, updatedAt: new Date() },
    });
  return record;
}

/** Clear the snooze (operator-initiated unmute). */
export async function clearMarketSignalsWatcherSnooze(): Promise<void> {
  await db
    .delete(appSettingsTable)
    .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_SNOOZE_KEY));
}

/** Read snooze for dashboard surfacing (returns null if expired). */
export async function getMarketSignalsWatcherSnooze(
  now: number = Date.now(),
): Promise<MarketSignalsWatcherSnooze | null> {
  return await readActiveSnooze(now);
}

async function readState(): Promise<MarketSignalsWatcherState> {
  try {
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_STATE_KEY));
    if (!row?.value || !isWatcherState(row.value)) return { ...EMPTY_STATE };
    const v = row.value as Partial<MarketSignalsWatcherState>;
    const silentRaw = (v as { silentCoinIncidents?: unknown }).silentCoinIncidents;
    const silent: Record<string, { since: string }> = {};
    if (silentRaw && typeof silentRaw === "object") {
      for (const [k, val] of Object.entries(silentRaw as Record<string, unknown>)) {
        if (typeof k !== "string" || k.length === 0) continue;
        if (val && typeof val === "object") {
          const since = (val as { since?: unknown }).since;
          if (typeof since === "string") {
            silent[k] = { since };
          }
        }
      }
    }
    return {
      activeIncidentKey: typeof v.activeIncidentKey === "string" ? v.activeIncidentKey : null,
      activeIncidentReason:
        typeof v.activeIncidentReason === "string" ? v.activeIncidentReason : null,
      activeIncidentSince:
        typeof v.activeIncidentSince === "string" ? v.activeIncidentSince : null,
      lastAlertAt: typeof v.lastAlertAt === "string" ? v.lastAlertAt : null,
      lastAlertKind:
        v.lastAlertKind === "incident" || v.lastAlertKind === "recovery"
          ? v.lastAlertKind
          : null,
      lastAlertReason: typeof v.lastAlertReason === "string" ? v.lastAlertReason : null,
      silentCoinIncidents: silent,
      lastSilentCoinAlertAt:
        typeof v.lastSilentCoinAlertAt === "string" ? v.lastSilentCoinAlertAt : null,
      lastSilentCoinAlertKind:
        v.lastSilentCoinAlertKind === "incident" || v.lastSilentCoinAlertKind === "recovery"
          ? v.lastSilentCoinAlertKind
          : null,
      lastSilentCoinAlertReason:
        typeof v.lastSilentCoinAlertReason === "string" ? v.lastSilentCoinAlertReason : null,
      pendingIncidentKey:
        typeof v.pendingIncidentKey === "string" ? v.pendingIncidentKey : null,
      pendingIncidentReason:
        typeof v.pendingIncidentReason === "string" ? v.pendingIncidentReason : null,
      pendingIncidentSince:
        typeof v.pendingIncidentSince === "string" ? v.pendingIncidentSince : null,
    };
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "market-signals-watcher: failed to read state; treating as empty",
    );
    return { ...EMPTY_STATE };
  }
}

function isHistoryEntry(v: unknown): v is MarketSignalsAlertHistoryEntry {
  if (!v || typeof v !== "object") return false;
  const r = v as Record<string, unknown>;
  return (
    typeof r.at === "string" &&
    (r.kind === "incident" || r.kind === "recovery") &&
    typeof r.reason === "string"
  );
}

async function readHistory(): Promise<MarketSignalsAlertHistoryEntry[]> {
  try {
    const [row] = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, MARKET_SIGNALS_WATCHER_HISTORY_KEY));
    if (!row?.value) return [];
    // Stored as { entries: [...] } so we can extend the row later with
    // metadata (cap config, schema version) without a migration.
    const raw = row.value as Record<string, unknown>;
    const list = Array.isArray(raw.entries) ? raw.entries : Array.isArray(raw) ? raw : [];
    return (list as unknown[])
      .filter(isHistoryEntry)
      .map((e) => ({
        at: e.at,
        kind: e.kind,
        reason: e.reason,
        incidentKey: typeof e.incidentKey === "string" ? e.incidentKey : null,
        incidentSince: typeof e.incidentSince === "string" ? e.incidentSince : null,
      }));
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "market-signals-watcher: failed to read history; treating as empty",
    );
    return [];
  }
}

async function writeHistory(entries: MarketSignalsAlertHistoryEntry[]): Promise<void> {
  await db
    .insert(appSettingsTable)
    .values({
      key: MARKET_SIGNALS_WATCHER_HISTORY_KEY,
      value: { entries },
    })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: { entries }, updatedAt: new Date() },
    });
}

async function appendHistoryQuiet(entry: MarketSignalsAlertHistoryEntry): Promise<void> {
  try {
    const existing = await readHistory();
    // Newest first so the dashboard can render directly without sorting.
    const next = [entry, ...existing].slice(0, MARKET_SIGNALS_WATCHER_HISTORY_MAX);
    await writeHistory(next);
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "market-signals-watcher: failed to append to history",
    );
  }
}

async function writeState(state: MarketSignalsWatcherState): Promise<void> {
  await db
    .insert(appSettingsTable)
    .values({ key: MARKET_SIGNALS_WATCHER_STATE_KEY, value: state })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: state, updatedAt: new Date() },
    });
}

/** Compute the unhealthy reason and stable incident key, or null if healthy. */
export function evaluatePollerStatus(
  status: PollerLikeStatus,
  now: number = Date.now(),
): { unhealthy: boolean; reason: string | null; incidentKey: string | null; isStale: boolean; staleThresholdMs: number } {
  const staleThresholdMs = status.intervalMs * 3;
  const isStale =
    status.lastPollAt == null || now - status.lastPollAt > staleThresholdMs;
  if (isStale) {
    // Bucket the stale incident by the last-good poll timestamp so a
    // recovery + new outage produces a fresh key. "never" covers the
    // case where the poller has never written a row.
    const bucket = status.lastPollAt == null ? "never" : String(status.lastPollAt);
    return {
      unhealthy: true,
      reason: status.lastPollAt == null
        ? "Poller has never written a row"
        : `Poller stale: last write ${Math.round((now - status.lastPollAt) / 1000)}s ago (>${Math.round(staleThresholdMs / 1000)}s threshold)`,
      incidentKey: `stale@${bucket}`,
      isStale: true,
      staleThresholdMs,
    };
  }
  if (status.lastPollError) {
    return {
      unhealthy: true,
      reason: `Last poll errored: ${status.lastPollError}`,
      // Bucket on the error message ALONE so a continuous outage that
      // keeps producing the same error every minute is treated as one
      // incident. `lastPollAt` is bumped on every poll attempt
      // (including failed ones), so it must NOT be part of the key —
      // otherwise the key changes every tick and we'd re-page every
      // minute. A *different* error message produces a new key (and a
      // new alert), and a recovery clears the active key so the next
      // error after recovery alerts again.
      incidentKey: `error:${status.lastPollError}`,
      isStale: false,
      staleThresholdMs,
    };
  }
  return { unhealthy: false, reason: null, incidentKey: null, isStale: false, staleThresholdMs };
}

function formatPayload(
  kind: "incident" | "recovery",
  reason: string,
  status: PollerLikeStatus,
  incidentSince: string | null,
  staleThresholdMs: number,
  isStale: boolean,
  scope: "poller" | "coin" = "poller",
  coinExtras: { coinId?: string; rowsLastHour?: number } = {},
): MarketSignalsAlertPayload {
  return {
    kind,
    scope,
    coinId: coinExtras.coinId ?? null,
    rowsLastHour: coinExtras.rowsLastHour,
    reason,
    pollerStatus: {
      lastPollAt: status.lastPollAt ? new Date(status.lastPollAt).toISOString() : null,
      lastPollOk: status.lastPollOk,
      lastPollError: status.lastPollError,
      intervalMs: status.intervalMs,
      staleThresholdMs,
      isStale,
    },
    incidentSince,
  };
}

export function formatAlertTitle(p: MarketSignalsAlertPayload): string {
  if (p.scope === "coin") {
    return p.kind === "incident"
      ? `Market signals coin stream silent — ${p.coinId}`
      : `Market signals coin stream recovered — ${p.coinId}`;
  }
  return p.kind === "incident"
    ? "Market signals poller — alert"
    : "Market signals poller — recovered";
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

async function dispatchGeneric(url: string, payload: MarketSignalsAlertPayload): Promise<void> {
  await postJson(url, {
    title: formatAlertTitle(payload),
    text: payload.reason,
    payload,
  });
}

async function dispatchSlack(url: string, payload: MarketSignalsAlertPayload): Promise<void> {
  const title = formatAlertTitle(payload);
  const icon = payload.kind === "incident" ? ":rotating_light:" : ":white_check_mark:";
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
  /** Inject poller status (test seam). */
  statusOverride?: PollerLikeStatus;
  /** Inject "now" for deterministic tests. */
  now?: number;
  /** When true, skip persisting the watcher state (test seam). */
  skipPersist?: boolean;
  /**
   * Task #302 — inject the per-coin silent set (test seam). When
   * provided, the watcher skips the DB query and uses this list as the
   * authoritative set of expected coins that wrote zero rows in the
   * trailing hour. The list is filtered to known poller targets.
   */
  silentCoinsOverride?: string[];
  /**
   * Task #302 — inject the canonical poller target list (test seam).
   * Defaults to `getMarketSignalsPollerTargets()`.
   */
  pollerTargetsOverride?: string[];
  /** Override snooze read (test seam). null = unmuted; undefined = read DB. */
  snoozeOverride?: MarketSignalsWatcherSnooze | null;
}

/** Resolve env-driven channel URLs. Exported for the dashboard surface. */
export function resolveChannels(opts: DispatchOptions = {}): {
  webhookUrl: string | null;
  slackWebhookUrl: string | null;
} {
  const webhookUrl = opts.webhookUrl !== undefined
    ? opts.webhookUrl
    : (process.env.MARKET_SIGNALS_ALERT_WEBHOOK_URL || null);
  const slackWebhookUrl = opts.slackWebhookUrl !== undefined
    ? opts.slackWebhookUrl
    : (process.env.MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL
        || process.env.SLACK_WEBHOOK_URL
        || null);
  return { webhookUrl, slackWebhookUrl };
}

/**
 * Task #302 — query the trailing-hour row counts and return the set of
 * expected coins that wrote zero rows. Returns `null` (not `[]`) when
 * the DB query fails so callers can skip per-coin processing rather
 * than spuriously fire recoveries for in-flight sub-incidents.
 */
async function fetchSilentCoinsFromDb(
  targets: string[],
  now: number,
): Promise<string[] | null> {
  if (targets.length === 0) return [];
  try {
    const since = new Date(now - 60 * 60 * 1000);
    const rows = await db
      .select({ coinId: marketSignalsTable.coinId })
      .from(marketSignalsTable)
      .where(gte(marketSignalsTable.timestamp, since))
      .groupBy(marketSignalsTable.coinId);
    const writers = new Set(rows.map((r) => r.coinId));
    return targets.filter((id) => !writers.has(id)).sort();
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "market-signals-watcher: per-coin row count query failed; skipping silent-coin alerts this tick",
    );
    return null;
  }
}

/**
 * Single watcher tick: read poller status, decide whether to fire an
 * alert (or recovery), persist state, return a summary. Never throws —
 * partial failures are reported in the summary.
 *
 * Task #302 — the tick also detects per-coin silent streams. When the
 * whole poller looks healthy but one or more expected coins have
 * written zero rows in the trailing hour, we fire one debounced
 * "incident" alert per affected coin and a one-shot "recovery" when
 * each silent coin starts writing again. Per-coin sub-incidents are
 * suppressed while a whole-poller incident is active so a global
 * outage does not cause us to page once per coin for the same root
 * cause; the per-coin state is preserved across that suppression so
 * sub-incidents don't spuriously recover when the global comes back.
 */
export async function runMarketSignalsWatcherTick(
  opts: DispatchOptions = {},
): Promise<MarketSignalsWatcherTickSummary> {
  const status = opts.statusOverride ?? getMarketSignalsPollerStatus();
  const now = opts.now ?? Date.now();
  const evalResult = evaluatePollerStatus(status, now);
  const { webhookUrl, slackWebhookUrl } = resolveChannels(opts);

  const state = await readState();
  const dispatched: MarketSignalsWatcherTickSummary["dispatched"] = [];
  const silentSummary: MarketSignalsWatcherTickSummary["silentCoins"] = {
    silent: [],
    newIncidents: [],
    recovered: [],
    deduped: [],
    suppressedByGlobalIncident: false,
    dispatched: [],
  };

  // Resolve the per-coin silent set. Test seams take precedence; if the
  // poller status itself is mocked we default to "no silent coins" so
  // whole-poller-only tests don't accidentally hit the DB or page about
  // empty test data.
  const targets = opts.pollerTargetsOverride ?? getMarketSignalsPollerTargets();
  let silentCoins: string[] | null;
  if (opts.silentCoinsOverride !== undefined) {
    const allowed = new Set(targets);
    silentCoins = opts.silentCoinsOverride.filter((c) => allowed.has(c)).slice().sort();
  } else if (opts.statusOverride !== undefined) {
    silentCoins = [];
  } else {
    silentCoins = await fetchSilentCoinsFromDb(targets, now);
  }

  const snooze = opts.snoozeOverride !== undefined
    ? opts.snoozeOverride
    : await readActiveSnooze(now);

  // ------------------------------------------------------------------
  // Snooze short-circuit (Task #301). Operator-suppressed dispatch:
  // we still evaluate poller health (so the dashboard reflects reality)
  // but we do NOT POST to any channel and we do NOT touch
  // `activeIncidentKey` (the dispatch dedupe key). Per-coin alerts are
  // also suppressed; `silentCoinIncidents` is preserved as-is so that
  // when the snooze ends, the next tick fires whatever is still wrong
  // (poller and/or per-coin).
  // ------------------------------------------------------------------
  if (snooze) {
    silentSummary.silent = silentCoins ?? [];
    silentSummary.deduped = Object.keys(state.silentCoinIncidents).slice().sort();
    silentSummary.suppressedByGlobalIncident = false;
    if (!evalResult.unhealthy) {
      // Poller healthy under snooze. If we were tracking an active
      // incident from before the snooze, clear it silently — operator
      // muted us, don't fire a recovery ping either. Also clear any
      // pending-during-snooze incident.
      const wasTrackingIncident = state.activeIncidentKey != null;
      const hadPending = state.pendingIncidentKey != null;
      if (wasTrackingIncident || hadPending) {
        const newState: MarketSignalsWatcherState = {
          ...state,
          activeIncidentKey: null,
          activeIncidentReason: null,
          activeIncidentSince: null,
          pendingIncidentKey: null,
          pendingIncidentReason: null,
          pendingIncidentSince: null,
        };
        if (!opts.skipPersist) await persistQuiet(newState);
        return {
          status: "snoozed_recovered",
          unhealthy: false,
          reason: null,
          dispatched,
          state: newState,
          silentCoins: silentSummary,
          snooze,
        };
      }
      return {
        status: "snoozed_healthy",
        unhealthy: false,
        reason: null,
        dispatched,
        state,
        silentCoins: silentSummary,
        snooze,
      };
    }
    // Unhealthy under snooze — track the latest incident in the
    // `pending*` fields for dashboard surfacing without bumping the
    // dispatch dedupe key.
    const incidentKey = evalResult.incidentKey!;
    const incidentReason = evalResult.reason!;
    const pendingSince = state.pendingIncidentKey === incidentKey
      ? (state.pendingIncidentSince ?? new Date(now).toISOString())
      : new Date(now).toISOString();
    const newState: MarketSignalsWatcherState = {
      ...state,
      pendingIncidentKey: incidentKey,
      pendingIncidentReason: incidentReason,
      pendingIncidentSince: pendingSince,
    };
    if (
      newState.pendingIncidentKey !== state.pendingIncidentKey ||
      newState.pendingIncidentReason !== state.pendingIncidentReason ||
      newState.pendingIncidentSince !== state.pendingIncidentSince
    ) {
      if (!opts.skipPersist) await persistQuiet(newState);
    }
    return {
      status: "snoozed",
      unhealthy: true,
      reason: incidentReason,
      dispatched,
      state: newState,
      silentCoins: silentSummary,
      snooze,
    };
  }

  // Snooze just expired or was never set — clear any leftover
  // pending-incident bookkeeping from a prior snooze so it doesn't
  // appear on the dashboard alongside live state.
  let baseState = state;
  if (
    state.pendingIncidentKey != null ||
    state.pendingIncidentReason != null ||
    state.pendingIncidentSince != null
  ) {
    baseState = {
      ...state,
      pendingIncidentKey: null,
      pendingIncidentReason: null,
      pendingIncidentSince: null,
    };
  }

  // ------------------------------------------------------------------
  // Whole-poller branch (unchanged behaviour from #290).
  // ------------------------------------------------------------------
  let globalStatus: MarketSignalsWatcherTickSummary["status"];
  let globalReason: string | null;
  let nextState: MarketSignalsWatcherState;

  if (!evalResult.unhealthy) {
    if (baseState.activeIncidentKey == null) {
      globalStatus = "healthy";
      globalReason = null;
      nextState = { ...baseState };
    } else {
      const recoveryReason = `Recovered from: ${baseState.activeIncidentReason ?? "unknown incident"}`;
      const payload = formatPayload(
        "recovery",
        recoveryReason,
        status,
        baseState.activeIncidentSince,
        evalResult.staleThresholdMs,
        false,
      );
      await dispatchPayload(webhookUrl, slackWebhookUrl, payload, dispatched);
      const recoveredAt = new Date(now).toISOString();
      nextState = {
        ...baseState,
        activeIncidentKey: null,
        activeIncidentReason: null,
        activeIncidentSince: null,
        lastAlertAt: recoveredAt,
        lastAlertKind: "recovery",
        lastAlertReason: recoveryReason,
      };
      // Task #303 — record this recovery in the bounded history ring so
      // the dashboard can surface flapping. Best-effort; never breaks
      // the tick.
      if (!opts.skipPersist) {
        await appendHistoryQuiet({
          at: recoveredAt,
          kind: "recovery",
          reason: recoveryReason,
          incidentKey: baseState.activeIncidentKey,
          incidentSince: baseState.activeIncidentSince,
        });
      }
      globalStatus = dispatched.some((d) => d.outcome === "error") ? "channel_error" : "recovered";
      globalReason = recoveryReason;
    }
  } else {
    const incidentKey = evalResult.incidentKey!;
    const incidentReason = evalResult.reason!;
    if (baseState.activeIncidentKey === incidentKey) {
      globalStatus = "deduped";
      globalReason = incidentReason;
      nextState = { ...baseState };
    } else {
      const incidentSince = new Date(now).toISOString();
      const payload = formatPayload(
        "incident",
        incidentReason,
        status,
        incidentSince,
        evalResult.staleThresholdMs,
        evalResult.isStale,
      );
      await dispatchPayload(webhookUrl, slackWebhookUrl, payload, dispatched);
      const alertAt = new Date(now).toISOString();
      nextState = {
        ...baseState,
        activeIncidentKey: incidentKey,
        activeIncidentReason: incidentReason,
        activeIncidentSince: incidentSince,
        lastAlertAt: alertAt,
        lastAlertKind: "incident",
        lastAlertReason: incidentReason,
      };
      // Task #303 — record this incident in the bounded history ring.
      if (!opts.skipPersist) {
        await appendHistoryQuiet({
          at: alertAt,
          kind: "incident",
          reason: incidentReason,
          incidentKey,
          incidentSince,
        });
      }
      globalStatus = dispatched.some((d) => d.outcome === "error") ? "channel_error" : "alerted";
      globalReason = incidentReason;
    }
  }

  // ------------------------------------------------------------------
  // Per-coin silent stream branch (#302).
  // ------------------------------------------------------------------
  // Suppress while the global poller is unhealthy: every coin would be
  // silent for the same root cause, and we don't want to page once per
  // coin for that. Preserve the existing silent-coin state so an
  // ongoing sub-incident doesn't fire a spurious recovery when the
  // global outage clears.
  if (evalResult.unhealthy) {
    silentSummary.suppressedByGlobalIncident = true;
    silentSummary.silent = silentCoins ?? [];
    silentSummary.deduped = Object.keys(baseState.silentCoinIncidents).slice().sort();
  } else if (silentCoins == null) {
    // DB query failed; do nothing (preserve state).
    silentSummary.deduped = Object.keys(baseState.silentCoinIncidents).slice().sort();
  } else {
    const silentSet = new Set(silentCoins);
    const activeSet = new Set(Object.keys(baseState.silentCoinIncidents));

    const newIncidents = [...silentSet].filter((c) => !activeSet.has(c)).sort();
    const recovered = [...activeSet].filter((c) => !silentSet.has(c)).sort();
    const deduped = [...silentSet].filter((c) => activeSet.has(c)).sort();

    silentSummary.silent = [...silentSet].sort();
    silentSummary.newIncidents = newIncidents;
    silentSummary.recovered = recovered;
    silentSummary.deduped = deduped;

    const updatedSilentMap: Record<string, { since: string }> = { ...baseState.silentCoinIncidents };

    let lastAlertAt = nextState.lastSilentCoinAlertAt;
    let lastAlertKind = nextState.lastSilentCoinAlertKind;
    let lastAlertReason = nextState.lastSilentCoinAlertReason;

    for (const coinId of newIncidents) {
      const since = new Date(now).toISOString();
      const reason = `Coin "${coinId}" wrote 0 rows in the trailing hour while the poller is healthy`;
      const payload = formatPayload(
        "incident",
        reason,
        status,
        since,
        evalResult.staleThresholdMs,
        false,
        "coin",
        { coinId, rowsLastHour: 0 },
      );
      const sub: MarketSignalsWatcherTickSummary["dispatched"] = [];
      await dispatchPayload(webhookUrl, slackWebhookUrl, payload, sub);
      for (const d of sub) {
        silentSummary.dispatched.push({ coinId, kind: "incident", ...d });
      }
      updatedSilentMap[coinId] = { since };
      lastAlertAt = new Date(now).toISOString();
      lastAlertKind = "incident";
      lastAlertReason = reason;
    }

    for (const coinId of recovered) {
      const reason = `Coin "${coinId}" started writing rows again`;
      const since = baseState.silentCoinIncidents[coinId]?.since ?? null;
      const payload = formatPayload(
        "recovery",
        reason,
        status,
        since,
        evalResult.staleThresholdMs,
        false,
        "coin",
        { coinId, rowsLastHour: 1 },
      );
      const sub: MarketSignalsWatcherTickSummary["dispatched"] = [];
      await dispatchPayload(webhookUrl, slackWebhookUrl, payload, sub);
      for (const d of sub) {
        silentSummary.dispatched.push({ coinId, kind: "recovery", ...d });
      }
      delete updatedSilentMap[coinId];
      lastAlertAt = new Date(now).toISOString();
      lastAlertKind = "recovery";
      lastAlertReason = reason;
    }

    nextState = {
      ...nextState,
      silentCoinIncidents: updatedSilentMap,
      lastSilentCoinAlertAt: lastAlertAt,
      lastSilentCoinAlertKind: lastAlertKind,
      lastSilentCoinAlertReason: lastAlertReason,
    };
  }

  // Persist only when state actually changed, to keep healthy-tick load
  // off the DB and avoid no-op updatedAt churn on app_settings.
  const dirty =
    JSON.stringify(state) !== JSON.stringify(nextState) ||
    (dispatched.length > 0 && dispatched.some((d) => d.outcome !== "skipped")) ||
    silentSummary.dispatched.length > 0;
  if (!opts.skipPersist && dirty) {
    await persistQuiet(nextState);
  }

  // Promote channel_error if any sub-incident dispatch errored.
  let finalStatus = globalStatus;
  if (
    finalStatus !== "channel_error" &&
    silentSummary.dispatched.some((d) => d.outcome === "error")
  ) {
    finalStatus = "channel_error";
  }
  // If the global side was a no-op but we fired per-coin alerts this
  // tick, surface that in the top-level status so callers can tell the
  // tick wasn't entirely idle.
  if (
    finalStatus === "healthy" &&
    (silentSummary.newIncidents.length > 0 || silentSummary.recovered.length > 0)
  ) {
    finalStatus = silentSummary.newIncidents.length > 0 ? "alerted" : "recovered";
  }

  return {
    status: finalStatus,
    unhealthy: evalResult.unhealthy,
    reason: globalReason,
    dispatched,
    state: nextState,
    silentCoins: silentSummary,
    snooze: null,
  };
}

async function persistQuiet(state: MarketSignalsWatcherState): Promise<void> {
  try {
    await writeState(state);
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "market-signals-watcher: failed to persist state; next tick may re-alert",
    );
  }
}

async function dispatchPayload(
  webhookUrl: string | null,
  slackWebhookUrl: string | null,
  payload: MarketSignalsAlertPayload,
  out: MarketSignalsWatcherTickSummary["dispatched"],
): Promise<void> {
  if (webhookUrl) {
    try {
      await dispatchGeneric(webhookUrl, payload);
      out.push({ channel: "generic", outcome: "ok" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      out.push({ channel: "generic", outcome: "error", error: msg });
      logger.warn({ err: msg }, "market-signals-watcher: generic webhook failed");
    }
  } else {
    out.push({ channel: "generic", outcome: "skipped" });
  }
  if (slackWebhookUrl) {
    try {
      await dispatchSlack(slackWebhookUrl, payload);
      out.push({ channel: "slack", outcome: "ok" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      out.push({ channel: "slack", outcome: "error", error: msg });
      logger.warn({ err: msg }, "market-signals-watcher: Slack webhook failed");
    }
  } else {
    out.push({ channel: "slack", outcome: "skipped" });
  }
}

/** Read the persisted watcher state for dashboard surfacing. */
export async function getMarketSignalsWatcherState(): Promise<MarketSignalsWatcherState> {
  return await readState();
}

/**
 * Read the bounded recent-alert history (newest first) for dashboard
 * surfacing. Capped to {@link MARKET_SIGNALS_WATCHER_HISTORY_MAX}; pass
 * `limit` to trim further (e.g. the dashboard renders the last 20).
 */
export async function getMarketSignalsWatcherHistory(
  limit?: number,
): Promise<MarketSignalsAlertHistoryEntry[]> {
  const all = await readHistory();
  if (typeof limit === "number" && limit >= 0) return all.slice(0, limit);
  return all;
}

/** Snapshot of channel configuration for dashboard surfacing. */
export function getMarketSignalsAlertChannels(): {
  genericConfigured: boolean;
  slackConfigured: boolean;
} {
  const { webhookUrl, slackWebhookUrl } = resolveChannels();
  return {
    genericConfigured: !!webhookUrl,
    slackConfigured: !!slackWebhookUrl,
  };
}

let watcherInterval: ReturnType<typeof setInterval> | null = null;

/** Start the background watcher. Idempotent. */
export function startMarketSignalsPollerWatcher(intervalMs = 60_000): void {
  if (watcherInterval) return;
  watcherInterval = setInterval(() => {
    void runMarketSignalsWatcherTick().catch((err) =>
      logger.error({ err }, "market-signals-watcher: unexpected tick error"),
    );
  }, intervalMs);
  logger.info({ intervalMs }, "Market signals poller watcher started");
}

/** Test seam — stop the loop. */
export function stopMarketSignalsPollerWatcher(): void {
  if (watcherInterval) {
    clearInterval(watcherInterval);
    watcherInterval = null;
  }
}
