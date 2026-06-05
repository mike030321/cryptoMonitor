/**
 * Task #577 — dashboard banner for the "disabled-role outcome leaked"
 * alarm. Shown whenever the api-server has observed at least one
 * record-outcome rejected by the brain with reason
 * `disabled_role_rejected` in the active window (default last 5
 * minutes).
 *
 * The same condition fires an off-dashboard webhook from the
 * api-server's disabled-outcome-notifier on the rising edge. This
 * banner is the in-page counterpart so an operator who DOES have the
 * dashboard open sees the same alarm without grep-ing /var/log.
 */
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  AlertTriangle,
  BellOff,
  BellRing,
  CheckCircle2,
  Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface AlertHook {
  configured: boolean;
  genericConfigured: boolean;
  slackConfigured: boolean;
  activeIncident: boolean;
  activeIncidentReason: string | null;
  activeIncidentSince: string | null;
  lastAlertAt: string | null;
  lastAlertKind: "incident" | "recovery" | null;
  lastAlertReason: string | null;
}

interface DisabledOutcomeEvent {
  tickId: string;
  sliceId: string | null;
  timeframe: string;
  observedAt: string;
}

interface DisabledOutcomeHealth {
  generatedAt: string;
  bannerVisible: boolean;
  windowMinutes: number;
  timeframes: string[];
  perTimeframe: Record<string, number>;
  eventCount: number;
  recentEvents: DisabledOutcomeEvent[];
  lastObservedAt: string | null;
  alertHook: AlertHook;
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
  if (h < 48) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}

export function DisabledOutcomeBanner() {
  const { data, isLoading, isError } = useQuery<DisabledOutcomeHealth>({
    queryKey: ["disabled-outcome-health"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/disabled-outcome-health`);
      if (!res.ok) throw new Error(`disabled-outcome-health ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
    staleTime: 15_000,
  });

  // Hide the banner entirely when the api-server can't be reached
  // (fail-closed — the existing health cards will already be screaming
  // if there's a real api-server outage) or when there are no events
  // in the active window.
  if (isLoading)
    return (
      <Skeleton
        className="h-20 w-full"
        data-testid="disabled-outcome-banner-loading"
      />
    );
  if (isError || !data) return null;
  if (!data.bannerVisible) return null;

  return (
    <Card
      className="border-rose-500/40 bg-rose-500/10 ring-1 ring-rose-500/30"
      data-testid="disabled-outcome-banner"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2 text-rose-200">
          <AlertTriangle className="w-4 h-4" />
          Disabled-timeframe outcome leak
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            brain rejected outcome(s) for a turned-off timeframe — upstream is
            leaking
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div
            className="text-sm font-mono text-rose-100 flex items-start gap-2"
            data-testid="disabled-outcome-banner-reason"
          >
            <span className="text-rose-400">•</span>
            <span>
              {data.eventCount} disabled-role outcome
              {data.eventCount === 1 ? "" : "s"} rejected in the last{" "}
              {data.windowMinutes}m from{" "}
              {data.timeframes.length} timeframe
              {data.timeframes.length === 1 ? "" : "s"}
            </span>
          </div>

          {data.timeframes.length > 0 && (
            <div
              className="rounded-md bg-rose-500/10 ring-1 ring-rose-500/20 p-2 text-[11px] font-mono"
              data-testid="disabled-outcome-banner-tfs"
            >
              <div className="text-[10px] uppercase tracking-wider text-rose-200/80 mb-1">
                Leaking timeframes
              </div>
              <div className="flex flex-wrap gap-1.5">
                {data.timeframes.map((tf) => (
                  <span
                    key={tf}
                    className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-100"
                    data-testid={`disabled-outcome-banner-tf-${tf}`}
                  >
                    {tf} ({data.perTimeframe[tf] ?? 0})
                  </span>
                ))}
              </div>
            </div>
          )}

          {data.recentEvents.length > 0 && (
            <div
              className="rounded-md bg-rose-500/10 ring-1 ring-rose-500/20 p-2 text-[11px] font-mono"
              data-testid="disabled-outcome-banner-events"
            >
              <div className="text-[10px] uppercase tracking-wider text-rose-200/80 mb-1">
                Most recent leaking events (so you can find the caller fast)
              </div>
              <div className="space-y-1">
                {[...data.recentEvents]
                  .reverse()
                  .slice(0, 5)
                  .map((e) => (
                    <div
                      key={`${e.tickId}-${e.observedAt}`}
                      className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-rose-100"
                      data-testid={`disabled-outcome-banner-event-${e.tickId}`}
                    >
                      <span className="px-1 rounded bg-rose-500/20">
                        tf=<span className="font-bold">{e.timeframe}</span>
                      </span>
                      <span className="text-rose-200/80">
                        tick=
                        <span className="text-rose-100 break-all">
                          {e.tickId}
                        </span>
                      </span>
                      <span className="text-rose-200/80">
                        slice=
                        <span className="text-rose-100 break-all">
                          {e.sliceId ?? "—"}
                        </span>
                      </span>
                      <span className="text-rose-200/60 ml-auto flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatRelative(e.observedAt)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          <div
            className={cn(
              "p-2 rounded-md flex items-start gap-2 text-[11px] font-mono",
              data.alertHook.configured
                ? "bg-emerald-500/10 ring-1 ring-emerald-500/20 text-emerald-200"
                : "bg-amber-500/10 ring-1 ring-amber-500/20 text-amber-200",
            )}
            data-testid="disabled-outcome-banner-alert-hook"
          >
            {data.alertHook.configured ? (
              <BellRing className="w-4 h-4 mt-0.5 shrink-0" />
            ) : (
              <BellOff className="w-4 h-4 mt-0.5 shrink-0" />
            )}
            <div className="space-y-1 min-w-0 flex-1">
              <div>
                {data.alertHook.configured ? (
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
                    <code className="text-amber-100">
                      DISABLED_OUTCOME_ALERT_WEBHOOK_URL
                    </code>
                    {" or "}
                    <code className="text-amber-100">
                      DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL
                    </code>
                    {" to page off-dashboard when an outcome lands on a"}
                    {" disabled timeframe."}
                  </>
                )}
              </div>
              {data.alertHook.activeIncident && (
                <div
                  className="text-rose-200"
                  data-testid="disabled-outcome-banner-active-incident"
                >
                  Active incident:{" "}
                  {data.alertHook.activeIncidentReason ?? "unknown"}
                  {data.alertHook.activeIncidentSince
                    ? ` (since ${formatRelative(data.alertHook.activeIncidentSince)})`
                    : ""}
                </div>
              )}
              {data.alertHook.lastAlertAt && !data.alertHook.activeIncident && (
                <div
                  className="text-emerald-200/80"
                  data-testid="disabled-outcome-banner-last-alert"
                >
                  <CheckCircle2 className="inline w-3 h-3 mr-1" />
                  Last {data.alertHook.lastAlertKind ?? "alert"}:{" "}
                  {data.alertHook.lastAlertReason ?? "unknown"} ·{" "}
                  {formatRelative(data.alertHook.lastAlertAt)}
                </div>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
