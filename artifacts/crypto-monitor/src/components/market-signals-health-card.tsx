import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import {
  Activity,
  AlertTriangle,
  BellOff,
  BellRing,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Moon,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAdminKey } from "@/hooks/use-admin-key";
import { AdminKeyField } from "@/components/admin-key-field";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

type SnoozeDuration = "15m" | "1h" | "until_midnight";

const SNOOZE_OPTIONS: Array<{ value: SnoozeDuration; label: string }> = [
  { value: "15m", label: "15m" },
  { value: "1h", label: "1h" },
  { value: "until_midnight", label: "until midnight" },
];

interface FieldNonNullCounts {
  fundingRate: number;
  openInterestUsd: number;
  liquidations1hUsd: number;
  bidAskSpreadBps: number;
  midPrice: number;
}

interface PerCoinRow {
  coinId: string;
  rowsLastHour: number;
  lastTimestamp: string | null;
  fieldNonNullCounts?: FieldNonNullCounts;
}

const SIGNAL_FIELDS: Array<{
  key: keyof FieldNonNullCounts;
  label: string;
  short: string;
}> = [
  { key: "fundingRate", label: "funding_rate", short: "fund" },
  { key: "openInterestUsd", label: "open_interest_usd", short: "OI" },
  { key: "liquidations1hUsd", label: "liquidations_1h_usd", short: "liq" },
  { key: "bidAskSpreadBps", label: "bid_ask_spread_bps", short: "sprd" },
  { key: "midPrice", label: "mid_price", short: "mid" },
];

interface AlertHistoryEntry {
  at: string;
  kind: "incident" | "recovery";
  reason: string;
  incidentKey: string | null;
  incidentSince: string | null;
}

interface SilentCoinIncident {
  coinId: string;
  since: string;
}

interface AlertHookInfo {
  configured: boolean;
  genericConfigured: boolean;
  slackConfigured: boolean;
  activeIncident: boolean;
  activeIncidentReason: string | null;
  activeIncidentSince: string | null;
  lastAlertAt: string | null;
  lastAlertKind: "incident" | "recovery" | null;
  lastAlertReason: string | null;
  recentAlerts?: AlertHistoryEntry[];
  // Task #302 — per-coin sub-incidents the watcher is currently paging
  // on. Empty array when no coin streams are silently broken.
  silentCoinIncidents?: SilentCoinIncident[];
  lastSilentCoinAlertAt?: string | null;
  lastSilentCoinAlertKind?: "incident" | "recovery" | null;
  lastSilentCoinAlertReason?: string | null;
  // Task #301 — operator snooze surface.
  snoozed?: boolean;
  snoozedUntil?: string | null;
  snoozedAt?: string | null;
  snoozeDuration?: SnoozeDuration | null;
  pendingIncident?: boolean;
  pendingIncidentReason?: string | null;
  pendingIncidentSince?: string | null;
}

interface MarketSignalsHealth {
  lastPollAt: string | null;
  lastPollOk: boolean;
  lastPollError: string | null;
  intervalMs: number;
  staleThresholdMs: number;
  isStale: boolean;
  totalRowsLastHour: number;
  perCoin: PerCoinRow[];
  alertHook?: AlertHookInfo;
}

function formatRelative(iso: string | null): string {
  if (!iso) return "never";
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return "unknown";
  const sec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (sec < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const h = Math.round(min / 60);
  return `${h}h ago`;
}

function formatRemaining(iso: string | null | undefined): string {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return "";
  const sec = Math.max(0, Math.round((t - Date.now()) / 1000));
  if (sec < 60) return `${sec}s left`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m left`;
  const h = Math.floor(min / 60);
  const rem = min % 60;
  return rem === 0 ? `${h}h left` : `${h}h ${rem}m left`;
}

export function MarketSignalsHealthCard() {
  const [historyOpen, setHistoryOpen] = useState(false);
  const queryClient = useQueryClient();
  const admin = useAdminKey();
  const [snoozeBusy, setSnoozeBusy] = useState(false);
  const [snoozeErr, setSnoozeErr] = useState<string | null>(null);
  const [showKeyField, setShowKeyField] = useState(false);

  const { data, isLoading, isError } = useQuery<MarketSignalsHealth>({
    queryKey: ["market-signals-health"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/market-signals-health`);
      if (!res.ok) throw new Error(`market-signals-health ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
    staleTime: 15_000,
  });

  async function applySnooze(duration: SnoozeDuration | null) {
    setSnoozeErr(null);
    if (!admin.hasKey) {
      setShowKeyField(true);
      setSnoozeErr("Admin key required to change snooze.");
      return;
    }
    setSnoozeBusy(true);
    try {
      const res = await admin.adminFetch(
        duration
          ? `${apiBase}/crypto/market-signals/snooze`
          : `${apiBase}/crypto/market-signals/snooze/clear`,
        {
          action: duration ? `snooze ${duration}` : "clear snooze",
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: duration ? JSON.stringify({ duration }) : undefined,
        },
      );
      if (!res) {
        setShowKeyField(true);
        throw new Error("Admin key was rejected or missing.");
      }
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error((body as { error?: string }).error || `HTTP ${res.status}`);
      }
      await queryClient.invalidateQueries({ queryKey: ["market-signals-health"] });
      // Successful action — collapse the inline key field to reduce clutter.
      setShowKeyField(false);
    } catch (err) {
      setSnoozeErr(err instanceof Error ? err.message : String(err));
    } finally {
      setSnoozeBusy(false);
    }
  }

  const stale = data?.isStale ?? false;
  const hasError = !!data?.lastPollError;
  // A single coin writing zero rows in the last hour means an upstream
  // stream is silently broken even when the global poller status is OK
  // (the poller marks lastPollOk = true if any target succeeds).
  const zeroCount = data?.perCoin.filter((r) => r.rowsLastHour === 0).length ?? 0;
  // Also count (coin, field) pairs where the field is mostly null
  // despite rows being written — a silent partial outage on a single
  // upstream stream. Without this, the top-line status reads "OK"
  // while the matrix shows red cells.
  const mostlyNullFieldCount =
    data?.perCoin.reduce((acc, r) => {
      if (r.rowsLastHour <= 0 || !r.fieldNonNullCounts) return acc;
      let n = 0;
      for (const f of SIGNAL_FIELDS) {
        const nn = r.fieldNonNullCounts[f.key];
        if (nn / r.rowsLastHour < 0.5) n += 1;
      }
      return acc + n;
    }, 0) ?? 0;
  const partialOutage = zeroCount > 0 || mostlyNullFieldCount > 0;
  const unhealthy = stale || hasError || partialOutage;
  const partialLabel =
    zeroCount > 0
      ? `PARTIAL (${zeroCount} silent)`
      : `PARTIAL (${mostlyNullFieldCount} field${mostlyNullFieldCount === 1 ? "" : "s"} null)`;

  return (
    <Card
      className={cn(
        "border-border/40",
        unhealthy
          ? "bg-rose-500/10 ring-1 ring-rose-500/30"
          : "bg-card/30",
      )}
      data-testid="market-signals-health-card"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Market signals poller
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            funding · open interest · liquidations · spread · BTC/ETH lead
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-24 w-full" />}
        {isError && (
          <div
            className="text-sm text-rose-300 font-mono"
            data-testid="market-signals-health-error"
          >
            Couldn't load market signals poller health.
          </div>
        )}
        {data && (
          <div className="space-y-3">
            <div className="flex items-end justify-between gap-4 flex-wrap">
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground">
                  Last poll
                </div>
                <div
                  className={cn(
                    "text-2xl font-display font-bold mt-1 flex items-center gap-2",
                    unhealthy ? "text-rose-300" : "text-emerald-300",
                  )}
                  data-testid="market-signals-health-status"
                >
                  {unhealthy ? (
                    <AlertTriangle className="w-5 h-5" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5" />
                  )}
                  {stale
                    ? "STALE"
                    : hasError
                      ? "ERROR"
                      : partialOutage
                        ? partialLabel
                        : "OK"}
                </div>
                <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
                  {formatRelative(data.lastPollAt)}
                  {" · "}interval {Math.round(data.intervalMs / 1000)}s
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] uppercase font-mono text-muted-foreground">
                  Rows last hour
                </div>
                <div
                  className="text-2xl font-display font-bold mt-1"
                  data-testid="market-signals-health-total-rows"
                >
                  {data.totalRowsLastHour}
                </div>
                <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
                  across {data.perCoin.length} coin{data.perCoin.length === 1 ? "" : "s"}
                </div>
              </div>
            </div>

            {data.alertHook && (
              <div
                className={cn(
                  "p-2 rounded-md flex items-start gap-2 text-[11px] font-mono",
                  data.alertHook.snoozed
                    ? "bg-slate-500/10 ring-1 ring-slate-400/30 text-slate-200"
                    : data.alertHook.configured
                      ? "bg-emerald-500/10 ring-1 ring-emerald-500/20 text-emerald-200"
                      : "bg-amber-500/10 ring-1 ring-amber-500/20 text-amber-200",
                )}
                data-testid="market-signals-health-alert-hook"
              >
                {data.alertHook.snoozed ? (
                  <Moon className="w-4 h-4 mt-0.5 shrink-0" />
                ) : data.alertHook.configured ? (
                  <BellRing className="w-4 h-4 mt-0.5 shrink-0" />
                ) : (
                  <BellOff className="w-4 h-4 mt-0.5 shrink-0" />
                )}
                <div className="space-y-1 min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    {data.alertHook.snoozed ? (
                      <>
                        <span data-testid="market-signals-health-snoozed-pill">
                          Alerts snoozed ({data.alertHook.snoozeDuration ?? "active"})
                          {" — "}
                          <span className="text-slate-100 font-semibold">
                            {formatRemaining(data.alertHook.snoozedUntil)}
                          </span>
                        </span>
                      </>
                    ) : data.alertHook.configured ? (
                      <>
                        Off-dashboard alert hook configured
                        {" ("}
                        {[
                          data.alertHook.genericConfigured ? "webhook" : null,
                          data.alertHook.slackConfigured ? "slack" : null,
                        ]
                          .filter(Boolean)
                          .join(" + ")}
                        {")"}
                      </>
                    ) : (
                      <>
                        No off-dashboard alert hook — set{" "}
                        <code className="text-amber-100">MARKET_SIGNALS_ALERT_WEBHOOK_URL</code>
                        {" or "}
                        <code className="text-amber-100">MARKET_SIGNALS_ALERT_SLACK_WEBHOOK_URL</code>
                        {" to page when the poller goes stale."}
                      </>
                    )}
                  </div>
                  {data.alertHook.activeIncident && !data.alertHook.snoozed && (
                    <div className="text-rose-200" data-testid="market-signals-health-alert-active">
                      Active incident: {data.alertHook.activeIncidentReason ?? "unknown"}
                      {data.alertHook.activeIncidentSince
                        ? ` (since ${formatRelative(data.alertHook.activeIncidentSince)})`
                        : ""}
                    </div>
                  )}
                  {(data.alertHook.silentCoinIncidents?.length ?? 0) > 0 && (
                    <div
                      className="text-rose-200"
                      data-testid="market-signals-health-silent-coin-incidents"
                    >
                      Silent coin sub-incidents:{" "}
                      {data.alertHook.silentCoinIncidents!.map((s, i) => (
                        <span key={s.coinId}>
                          {i > 0 ? ", " : ""}
                          <span className="font-semibold">{s.coinId}</span>
                          {" ("}
                          {formatRelative(s.since)}
                          {")"}
                        </span>
                      ))}
                    </div>
                  )}
                  {data.alertHook.snoozed && data.alertHook.pendingIncident && (
                    <div
                      className="text-amber-200"
                      data-testid="market-signals-health-snoozed-pending"
                    >
                      Suppressed incident: {data.alertHook.pendingIncidentReason ?? "unknown"}
                      {data.alertHook.pendingIncidentSince
                        ? ` (since ${formatRelative(data.alertHook.pendingIncidentSince)})`
                        : ""}
                      {" — will fire on unmute / snooze expiry if still bad."}
                    </div>
                  )}
                  {data.alertHook.recentAlerts && data.alertHook.recentAlerts.length > 0 && (
                    <div className="pt-1">
                      <button
                        type="button"
                        onClick={() => setHistoryOpen((v) => !v)}
                        className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
                        data-testid="market-signals-health-history-toggle"
                        aria-expanded={historyOpen}
                      >
                        {historyOpen ? (
                          <ChevronDown className="w-3 h-3" />
                        ) : (
                          <ChevronRight className="w-3 h-3" />
                        )}
                        Recent alerts ({data.alertHook.recentAlerts.length})
                      </button>
                      {historyOpen && (
                        <ul
                          className="mt-1.5 space-y-1 max-h-48 overflow-y-auto pr-1"
                          data-testid="market-signals-health-history-list"
                        >
                          {data.alertHook.recentAlerts.map((entry, idx) => (
                            <li
                              key={`${entry.at}-${idx}`}
                              className={cn(
                                "flex items-start gap-2 text-[11px] leading-tight rounded px-1.5 py-1",
                                entry.kind === "incident"
                                  ? "bg-rose-500/10 text-rose-100"
                                  : "bg-emerald-500/10 text-emerald-100",
                              )}
                              data-testid={`market-signals-health-history-row-${idx}`}
                            >
                              <span
                                className={cn(
                                  "shrink-0 inline-block px-1 rounded text-[9px] uppercase tracking-wider font-semibold mt-0.5",
                                  entry.kind === "incident"
                                    ? "bg-rose-500/30 text-rose-50"
                                    : "bg-emerald-500/30 text-emerald-50",
                                )}
                              >
                                {entry.kind}
                              </span>
                              <span className="shrink-0 text-muted-foreground/80 tabular-nums w-16">
                                {formatRelative(entry.at)}
                              </span>
                              <span className="min-w-0 break-words">{entry.reason}</span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                  {!data.alertHook.activeIncident &&
                    !data.alertHook.snoozed &&
                    data.alertHook.lastAlertAt && (
                      <div className="text-muted-foreground">
                        Last {data.alertHook.lastAlertKind ?? "alert"}:{" "}
                        {formatRelative(data.alertHook.lastAlertAt)}
                        {data.alertHook.lastAlertReason
                          ? ` — ${data.alertHook.lastAlertReason}`
                          : ""}
                      </div>
                    )}

                  {/* Snooze controls — operator mute for planned maintenance. */}
                  {data.alertHook.configured && (
                    <div
                      className="flex items-center gap-1.5 flex-wrap pt-1"
                      data-testid="market-signals-health-snooze-controls"
                    >
                      {data.alertHook.snoozed ? (
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={snoozeBusy}
                          onClick={() => void applySnooze(null)}
                          className="h-6 px-2 text-[10px] font-mono"
                          data-testid="market-signals-health-snooze-clear"
                        >
                          <X className="w-3 h-3 mr-1" />
                          Unmute now
                        </Button>
                      ) : (
                        <>
                          <span className="text-[10px] uppercase text-muted-foreground">
                            Snooze:
                          </span>
                          {SNOOZE_OPTIONS.map((opt) => (
                            <Button
                              key={opt.value}
                              size="sm"
                              variant="outline"
                              disabled={snoozeBusy}
                              onClick={() => void applySnooze(opt.value)}
                              className="h-6 px-2 text-[10px] font-mono"
                              data-testid={`market-signals-health-snooze-${opt.value}`}
                            >
                              {opt.label}
                            </Button>
                          ))}
                        </>
                      )}
                    </div>
                  )}

                  {snoozeErr && (
                    <div
                      className="text-[10px] text-rose-300"
                      data-testid="market-signals-health-snooze-error"
                    >
                      {snoozeErr}
                    </div>
                  )}
                  {showKeyField && (
                    <AdminKeyField
                      admin={admin}
                      autoFocus
                      label="Admin key"
                      helpText="Required to snooze or unmute the alert hook."
                      className="mt-1"
                      testIdPrefix="market-signals-health-snooze-key"
                    />
                  )}
                </div>
              </div>
            )}

            {hasError && data.lastPollError && (
              <div
                className="p-2 rounded-md bg-rose-500/10 ring-1 ring-rose-500/20 text-xs"
                data-testid="market-signals-health-last-error"
              >
                <div className="text-[10px] uppercase font-mono text-rose-200/80 mb-1">
                  Last poll error
                </div>
                <div className="font-mono text-rose-200 break-words">
                  {data.lastPollError}
                </div>
              </div>
            )}

            {data.perCoin.length === 0 ? (
              <div
                className="text-xs font-mono text-rose-200"
                data-testid="market-signals-health-empty"
              >
                No rows written in the last hour.
              </div>
            ) : (
              <div
                className="overflow-x-auto"
                data-testid="market-signals-health-per-coin"
              >
                <table className="w-full text-[11px] font-mono border-separate border-spacing-y-0.5">
                  <thead>
                    <tr className="text-[10px] uppercase text-muted-foreground">
                      <th className="text-left font-normal pr-2">coin</th>
                      <th className="text-right font-normal px-1">rows</th>
                      {SIGNAL_FIELDS.map((f) => (
                        <th
                          key={f.key}
                          className="text-right font-normal px-1"
                          title={f.label}
                        >
                          {f.short}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.perCoin.map((row) => {
                      const total = row.rowsLastHour;
                      const counts = row.fieldNonNullCounts;
                      return (
                        <tr
                          key={row.coinId}
                          className="bg-card/40"
                          data-testid={`market-signals-health-coin-${row.coinId}`}
                        >
                          <td className="pl-1.5 pr-2 py-1 truncate text-muted-foreground rounded-l-md">
                            {row.coinId}
                          </td>
                          <td
                            className={cn(
                              "text-right tabular-nums px-1 py-1",
                              total === 0 ? "text-rose-300" : "text-foreground",
                            )}
                          >
                            {total}
                          </td>
                          {SIGNAL_FIELDS.map((f, idx) => {
                            const nn = counts ? counts[f.key] : 0;
                            // Fill rate vs total rows written this
                            // hour. <50% non-null = silent partial
                            // outage on this specific upstream stream.
                            const hasRows = total > 0;
                            const rate = hasRows ? nn / total : 0;
                            const isMostlyNull = hasRows && rate < 0.5;
                            const isPartial = hasRows && !isMostlyNull && rate < 1;
                            return (
                              <td
                                key={f.key}
                                className={cn(
                                  "text-right tabular-nums px-1 py-1",
                                  idx === SIGNAL_FIELDS.length - 1 && "rounded-r-md",
                                  !hasRows
                                    ? "text-muted-foreground/50"
                                    : isMostlyNull
                                      ? "bg-rose-500/30 text-rose-100 font-semibold"
                                      : isPartial
                                        ? "bg-amber-500/20 text-amber-100"
                                        : "text-emerald-300",
                                )}
                                title={
                                  hasRows
                                    ? `${f.label}: ${nn}/${total} non-null (${Math.round(rate * 100)}%)`
                                    : `${f.label}: no rows written`
                                }
                                data-testid={`market-signals-health-cell-${row.coinId}-${f.key}`}
                              >
                                {hasRows ? `${nn}/${total}` : "—"}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <div className="mt-2 flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-2.5 h-2.5 rounded-sm bg-emerald-500/60" />
                    full
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-2.5 h-2.5 rounded-sm bg-amber-500/40" />
                    partial
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-2.5 h-2.5 rounded-sm bg-rose-500/50" />
                    mostly null
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
