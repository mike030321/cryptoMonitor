/**
 * Task #411 — dashboard banner for the daily 5m head top-up scheduler
 * (#410). Shown whenever:
 *   • any coin's 5m contiguous_days is below the alert threshold
 *     (310 days by default), OR
 *   • the daily top-up tick has failed twice in a row.
 *
 * The same conditions trigger an off-dashboard webhook from the
 * api-server (`topup-5m-notifier`). This banner is the in-page
 * counterpart so an operator who DOES have the dashboard open sees
 * the same alarm without scraping ml-engine logs or hitting
 * `/ml/admin/5m-topup/status` by hand.
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
  Calendar,
  CheckCircle2,
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

interface Topup5mHealth {
  generatedAt: string;
  fetchError: string | null;
  bannerVisible: boolean;
  errorStreakTriggered: boolean;
  errorStreakThreshold: number;
  consecutiveErrors: number;
  alertCoins: string[];
  thresholdDays: number | null;
  perCoinContiguousDays: Record<string, number> | null;
  lastAttemptOutcome: string | null;
  lastCheckAt: string | null;
  lastFinishedAt: string | null;
  lastError: string | null;
  schedulerEnabled: boolean | null;
  ticksTotal: number | null;
  runsTotal: number | null;
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

export function Topup5mBanner() {
  const { data, isLoading, isError } = useQuery<Topup5mHealth>({
    queryKey: ["5m-topup-health"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/5m-topup-health`);
      if (!res.ok) throw new Error(`5m-topup-health ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  // Hide the banner entirely when:
  //   • we couldn't load it (fail closed — surfacing nothing is better
  //     than surfacing a confusing "?" during an api-server outage; the
  //     existing skip-event/market-signals cards will already be
  //     screaming if there's a real outage), OR
  //   • the scheduler is healthy and no coins are below threshold.
  if (isLoading) return <Skeleton className="h-20 w-full" data-testid="5m-topup-banner-loading" />;
  if (isError || !data) return null;
  if (!data.bannerVisible) return null;

  const threshold = data.thresholdDays ?? 310;
  const belowCoins = data.alertCoins;
  const reasons: string[] = [];
  if (data.errorStreakTriggered) {
    reasons.push(
      `Daily 5m top-up has failed ${data.consecutiveErrors} ticks in a row`,
    );
  }
  if (belowCoins.length > 0) {
    reasons.push(
      `${belowCoins.length} coin${belowCoins.length === 1 ? "" : "s"} below ${threshold}d 5m contiguous gate`,
    );
  }

  return (
    <Card
      className="border-rose-500/40 bg-rose-500/10 ring-1 ring-rose-500/30"
      data-testid="5m-topup-banner"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2 text-rose-200">
          <AlertTriangle className="w-4 h-4" />
          5m head top-up alert
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            daily scheduler keeps the 305d 5m contiguous gate cleared
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex flex-col gap-1">
            {reasons.map((r) => (
              <div
                key={r}
                className="text-sm font-mono text-rose-100 flex items-start gap-2"
                data-testid="5m-topup-banner-reason"
              >
                <span className="text-rose-400">•</span>
                <span>{r}</span>
              </div>
            ))}
          </div>

          {belowCoins.length > 0 && (
            <div
              className="rounded-md bg-rose-500/10 ring-1 ring-rose-500/20 p-2 text-[11px] font-mono"
              data-testid="5m-topup-banner-coins"
            >
              <div className="text-[10px] uppercase tracking-wider text-rose-200/80 mb-1">
                Coins under {threshold} day threshold
              </div>
              <div className="flex flex-wrap gap-1.5">
                {belowCoins.map((c) => {
                  const days = data.perCoinContiguousDays?.[c];
                  return (
                    <span
                      key={c}
                      className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-100"
                      data-testid={`5m-topup-banner-coin-${c}`}
                    >
                      {c}
                      {typeof days === "number"
                        ? ` (${days.toFixed(1)}d)`
                        : ""}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {data.errorStreakTriggered && data.lastError && (
            <div
              className="rounded-md bg-rose-500/10 ring-1 ring-rose-500/20 p-2 text-[11px] font-mono"
              data-testid="5m-topup-banner-error"
            >
              <div className="text-[10px] uppercase tracking-wider text-rose-200/80 mb-1">
                Last scheduler error
              </div>
              <div className="text-rose-100 break-words">{data.lastError}</div>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-muted-foreground">
            <span className="flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              Last tick {formatRelative(data.lastCheckAt)}
              {data.lastAttemptOutcome
                ? ` (${data.lastAttemptOutcome})`
                : ""}
            </span>
            {data.schedulerEnabled === false && (
              <span className="text-amber-300">scheduler disabled</span>
            )}
          </div>

          <div
            className={cn(
              "p-2 rounded-md flex items-start gap-2 text-[11px] font-mono",
              data.alertHook.configured
                ? "bg-emerald-500/10 ring-1 ring-emerald-500/20 text-emerald-200"
                : "bg-amber-500/10 ring-1 ring-amber-500/20 text-amber-200",
            )}
            data-testid="5m-topup-banner-alert-hook"
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
                    <code className="text-amber-100">TOPUP_5M_ALERT_WEBHOOK_URL</code>
                    {" or "}
                    <code className="text-amber-100">TOPUP_5M_ALERT_SLACK_WEBHOOK_URL</code>
                    {" to page when the daily top-up fails or skips."}
                  </>
                )}
              </div>
              {data.alertHook.activeIncident && (
                <div className="text-rose-200" data-testid="5m-topup-banner-active-incident">
                  Active incident: {data.alertHook.activeIncidentReason ?? "unknown"}
                  {data.alertHook.activeIncidentSince
                    ? ` (since ${formatRelative(data.alertHook.activeIncidentSince)})`
                    : ""}
                </div>
              )}
              {data.alertHook.lastAlertAt && !data.alertHook.activeIncident && (
                <div
                  className="text-emerald-200/80"
                  data-testid="5m-topup-banner-last-alert"
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
