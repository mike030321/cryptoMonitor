/**
 * Task #554 — dataset-refresher health surface for the dashboard.
 *
 * The `dataset-refresher` workflow (Task #540) auto-refreshes the
 * cached training datasets and writes per-tf health to disk. Until
 * this card existed, an operator had to SSH-grep
 * `models/datasets/_freshness_status.json` and tail
 * `_freshness_alerts.jsonl` to know whether the refresher was alive.
 *
 * This module exports two React components, both fed by the shared
 * `useDatasetFreshness` hook so a single 60s poll feeds them both:
 *
 *   * <DatasetFreshnessBanner/>  Sticky red/amber banner that renders
 *     ONLY when the refresher is unhealthy (any tf past-due or any
 *     unread alert). Clicking it scrolls to the failure-log panel
 *     inside <DatasetFreshnessCard/> and opens it. Modeled on the
 *     Task #518 BrainStateBanner.
 *
 *   * <DatasetFreshnessCard/>  Always-visible card that lists every
 *     managed timeframe as a green/amber/red pill with its
 *     last-success timestamp, plus a collapsible "Recent failures"
 *     panel that surfaces the alert log with unread alerts pinned to
 *     the top. The collapsible is the panel the banner links to.
 */

import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Clock,
  Database,
  FileX,
} from "lucide-react";
import {
  useDatasetFreshness,
  type DatasetFreshnessAlert,
  type DatasetFreshnessHealth,
  type DatasetFreshnessStatus,
  type DatasetFreshnessTimeframe,
} from "@/hooks/use-news";
import { formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

const FAILURE_PANEL_ID = "dataset-freshness-failures";

function healthStyles(health: DatasetFreshnessHealth): {
  ring: string;
  bg: string;
  fg: string;
  dot: string;
  label: string;
} {
  switch (health) {
    case "green":
      return {
        ring: "ring-emerald-500/30",
        bg: "bg-emerald-500/10",
        fg: "text-emerald-300",
        dot: "bg-emerald-400",
        label: "FRESH",
      };
    case "amber":
      return {
        ring: "ring-amber-500/40",
        bg: "bg-amber-500/10",
        fg: "text-amber-300",
        dot: "bg-amber-400",
        label: "DUE",
      };
    case "red":
      return {
        ring: "ring-red-500/40",
        bg: "bg-red-500/10",
        fg: "text-red-300",
        dot: "bg-red-400",
        label: "STALE",
      };
    case "unknown":
    default:
      return {
        ring: "ring-muted-foreground/20",
        bg: "bg-muted/20",
        fg: "text-muted-foreground",
        dot: "bg-muted-foreground/60",
        label: "N/A",
      };
  }
}

function formatAge(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const abs = Math.abs(seconds);
  if (abs < 60) return `${Math.round(abs)}s`;
  if (abs < 3600) return `${Math.round(abs / 60)}m`;
  if (abs < 86400) return `${Math.round((abs / 3600) * 10) / 10}h`;
  return `${Math.round((abs / 86400) * 10) / 10}d`;
}

function focusFailurePanel(): void {
  // Open the failure-log collapsible (its `data-state` is driven by
  // local state, so we expose it via a hash + scroll). The card's
  // <Collapsible/> defaults to open when the URL hash matches.
  const el = document.getElementById(FAILURE_PANEL_ID);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  // Dispatch a custom event the card listens for — keeps the open
  // state lifted to React rather than doing DOM surgery here.
  window.dispatchEvent(new CustomEvent("dataset-freshness-open-failures"));
}

/* ────────────────────────────────────────────────────────────── */
/*  Sticky banner — renders only on unhealthy state                */
/* ────────────────────────────────────────────────────────────── */

export function DatasetFreshnessBanner() {
  const { data, isLoading, isError } = useDatasetFreshness();
  if (isLoading || !data) return null;

  // `unknown` (status file missing) is NOT escalated to a banner —
  // a fresh checkout has never run the refresher and we don't want
  // to add chrome on first boot. The card below still renders the
  // "n/a" state honestly. Errors fetching the route also stay quiet
  // because the brain banner already handles "API offline" surfaces.
  if (isError) return null;
  if (data.state === "green" || data.state === "unknown") return null;

  const isRed = data.state === "red";
  const styles = healthStyles(data.state);
  const Icon = isRed ? AlertTriangle : Clock;

  const pastDue = data.pastDueTimeframes;
  const subtitle = isRed
    ? `Dataset cache is stale. ${pastDue.length > 0 ? `Past-due timeframes: ${pastDue.join(", ")}.` : ""} ${
        data.totalUnreadAlerts > 0
          ? `${data.totalUnreadAlerts} unread refresh alert${data.totalUnreadAlerts === 1 ? "" : "s"}.`
          : ""
      } Retrains will run on stale data until this clears.`
    : `Dataset refresh is overdue but inside the 1.5× grace window. Watching: ${pastDue.join(", ") || "—"}.`;

  return (
    <div
      className={`sticky top-0 z-30 flex flex-col gap-2 px-4 py-3 rounded-xl ring-1 backdrop-blur-md shadow-lg ${styles.ring} ${styles.bg}`}
      data-testid="dataset-freshness-banner"
      data-freshness-state={data.state}
    >
      <button
        type="button"
        className="flex items-start gap-3 text-left"
        onClick={focusFailurePanel}
        data-testid="dataset-freshness-banner-link"
      >
        <Icon className={`w-5 h-5 mt-0.5 shrink-0 ${styles.fg}`} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <div className={`text-sm font-mono font-semibold uppercase tracking-wider ${styles.fg}`}>
              {isRed
                ? "DATASET REFRESHER UNHEALTHY"
                : "DATASET REFRESH OVERDUE"}
            </div>
            <span className={`text-[11px] font-mono uppercase tracking-wider underline-offset-2 hover:underline ${styles.fg}`}>
              View failure log →
            </span>
          </div>
          <div className="text-xs text-foreground/80 mt-1">{subtitle}</div>
        </div>
      </button>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  Per-tf pill                                                    */
/* ────────────────────────────────────────────────────────────── */

function TimeframePill({ tf }: { tf: DatasetFreshnessTimeframe }) {
  const styles = healthStyles(tf.health);
  const cadence =
    tf.cadenceHours != null
      ? `every ${tf.cadenceHours < 1 ? `${Math.round(tf.cadenceHours * 60)}m` : `${tf.cadenceHours}h`}`
      : "cadence n/a";
  const lastSuccess = tf.lastSuccessAt
    ? formatTimeAgo(tf.lastSuccessAt)
    : "never";
  const pastDueNote =
    tf.pastDueSeconds != null && tf.pastDueSeconds > 0
      ? ` · ${formatAge(tf.pastDueSeconds)} past due`
      : "";

  return (
    <div
      className={cn(
        "flex flex-col gap-1 rounded-lg px-3 py-2 ring-1",
        styles.ring,
        styles.bg,
      )}
      data-testid={`dataset-freshness-pill-${tf.timeframe}`}
      data-freshness-health={tf.health}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={cn("w-2 h-2 rounded-full", styles.dot)} />
          <span className="font-mono text-sm font-semibold uppercase tracking-wider">
            {tf.timeframe}
          </span>
        </div>
        <span className={cn("text-[10px] font-mono uppercase tracking-wider", styles.fg)}>
          {styles.label}
        </span>
      </div>
      <div className="text-[11px] font-mono text-muted-foreground">
        last success {lastSuccess}
        {pastDueNote}
      </div>
      <div className="text-[10px] font-mono text-muted-foreground/80">
        {cadence}
        {tf.unreadAlertCount > 0 ? ` · ${tf.unreadAlertCount} new alert${tf.unreadAlertCount === 1 ? "" : "s"}` : ""}
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  Failure log row                                                */
/* ────────────────────────────────────────────────────────────── */

function AlertRow({ alert }: { alert: DatasetFreshnessAlert }) {
  return (
    <div
      className={cn(
        "flex flex-col gap-1 rounded-md px-3 py-2 text-xs font-mono border",
        alert.unread
          ? "border-red-500/40 bg-red-500/5"
          : "border-border/40 bg-card/40",
      )}
      data-testid="dataset-freshness-alert-row"
      data-unread={alert.unread ? "true" : "false"}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <span className="text-foreground">
          {alert.timeframe ?? "—"} ·{" "}
          <span className={alert.unread ? "text-red-300" : "text-muted-foreground"}>
            {alert.status ?? "unknown"}
          </span>
        </span>
        <span className="text-muted-foreground">
          {alert.at ? formatTimeAgo(alert.at) : "no timestamp"}
          {alert.unread && (
            <span className="ml-2 text-red-300 uppercase tracking-wider">
              new
            </span>
          )}
        </span>
      </div>
      {alert.error && (
        <div className="text-foreground/80 break-words">{alert.error}</div>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  Always-visible card                                            */
/* ────────────────────────────────────────────────────────────── */

function StatusHeadline({ data }: { data: DatasetFreshnessStatus }) {
  if (!data.statusFileExists) {
    return (
      <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
        <FileX className="w-3.5 h-3.5" />
        <span>
          status file not found — dataset-refresher has never written
          a tick on this host
        </span>
      </div>
    );
  }
  if (data.statusReadError) {
    return (
      <div className="flex items-center gap-2 text-xs font-mono text-red-300">
        <AlertTriangle className="w-3.5 h-3.5" />
        <span>status file unreadable: {data.statusReadError}</span>
      </div>
    );
  }
  if (data.timeframes.length === 0) {
    return (
      <div className="text-xs font-mono text-muted-foreground">
        no timeframes recorded yet
      </div>
    );
  }
  if (data.state === "green") {
    return (
      <div className="flex items-center gap-2 text-xs font-mono text-emerald-300">
        <CheckCircle2 className="w-3.5 h-3.5" />
        <span>
          all {data.timeframes.length} timeframes within cadence ·
          last write {data.writtenAt ? formatTimeAgo(data.writtenAt) : "—"}
        </span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 text-xs font-mono text-foreground/80">
      <Clock className="w-3.5 h-3.5" />
      <span>
        {data.pastDueTimeframes.length} timeframe
        {data.pastDueTimeframes.length === 1 ? "" : "s"} past due ·{" "}
        {data.totalUnreadAlerts} unread alert
        {data.totalUnreadAlerts === 1 ? "" : "s"}
      </span>
    </div>
  );
}

export function DatasetFreshnessCard() {
  const { data, isLoading, isError, error } = useDatasetFreshness();
  // Sort alerts: unread first, then newest first. The hook already
  // returns newest-first; we just promote unread to the top.
  const [failuresOpen, setFailuresOpen] = useState(false);

  // Banner click handler dispatches `dataset-freshness-open-failures`
  // — listen and pop the collapsible open. We also auto-open whenever
  // the URL hash points at our anchor so a deep link works too.
  useEffect(() => {
    const open = () => setFailuresOpen(true);
    window.addEventListener("dataset-freshness-open-failures", open);
    if (window.location.hash === `#${FAILURE_PANEL_ID}`) setFailuresOpen(true);
    return () =>
      window.removeEventListener("dataset-freshness-open-failures", open);
  }, []);

  if (isLoading) {
    return (
      <div
        className="rounded-xl border border-border/40 bg-card/40 p-4 text-xs font-mono text-muted-foreground"
        data-testid="dataset-freshness-card-loading"
      >
        loading dataset freshness…
      </div>
    );
  }
  if (isError || !data) {
    return (
      <div
        className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-xs font-mono text-red-300"
        data-testid="dataset-freshness-card-error"
      >
        failed to load dataset freshness
        {error instanceof Error ? `: ${error.message}` : ""}
      </div>
    );
  }

  const sortedAlerts: DatasetFreshnessAlert[] = [
    ...data.alerts.filter((a) => a.unread),
    ...data.alerts.filter((a) => !a.unread),
  ];

  return (
    <div
      className="rounded-xl border border-border/40 bg-card/40 p-4 space-y-3"
      data-testid="dataset-freshness-card"
      data-freshness-state={data.state}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-mono uppercase tracking-wider">
            Dataset Refresher
          </span>
        </div>
        <span
          className={cn(
            "text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full ring-1",
            healthStyles(data.state).ring,
            healthStyles(data.state).bg,
            healthStyles(data.state).fg,
          )}
          data-testid="dataset-freshness-card-state"
        >
          {healthStyles(data.state).label}
        </span>
      </div>

      <StatusHeadline data={data} />

      {data.timeframes.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {data.timeframes.map((tf) => (
            <TimeframePill key={tf.timeframe} tf={tf} />
          ))}
        </div>
      )}

      <Collapsible
        open={failuresOpen}
        onOpenChange={setFailuresOpen}
        id={FAILURE_PANEL_ID}
      >
        <CollapsibleTrigger
          className="w-full group"
          data-testid="dataset-freshness-failures-toggle"
        >
          <div className="flex items-center justify-between p-2 rounded-md bg-card/40 border border-border/40 hover:bg-card/60 transition-colors">
            <span className="text-xs font-mono uppercase tracking-wider flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5" />
              Recent failures
              {data.totalAlerts > 0 && (
                <span className="text-muted-foreground">
                  ({data.totalAlerts}
                  {data.totalUnreadAlerts > 0 && (
                    <span className="text-red-300">
                      {" "}
                      · {data.totalUnreadAlerts} unread
                    </span>
                  )}
                  )
                </span>
              )}
            </span>
            <ChevronDown className="w-4 h-4 text-muted-foreground transition-transform group-data-[state=closed]:-rotate-90" />
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-2">
          {!data.alertsFileExists ? (
            <div className="text-xs font-mono text-muted-foreground px-1 py-2">
              no alert log on disk yet — the refresher has not recorded
              a failure on this host
            </div>
          ) : sortedAlerts.length === 0 ? (
            <div className="text-xs font-mono text-muted-foreground px-1 py-2">
              alert log is empty
            </div>
          ) : (
            <div
              className="flex flex-col gap-2 max-h-72 overflow-y-auto"
              data-testid="dataset-freshness-failures-list"
            >
              {sortedAlerts.map((a, i) => (
                <AlertRow key={`${a.at ?? "no-ts"}-${i}`} alert={a} />
              ))}
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

