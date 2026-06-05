import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Clock, AlertTriangle, CheckCircle2, Database } from "lucide-react";
import { cn } from "@/lib/utils";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface FiveMTopupStatus {
  enabled?: boolean;
  interval_seconds?: number;
  window_days?: number;
  alert_below_days?: number;
  last_check_at?: number | null;
  last_attempt_outcome?: string | null;
  last_finished_at?: number | null;
  last_error?: string | null;
  last_topup_inserted?: number;
  last_topup_per_coin?: Record<string, number>;
  last_health_per_coin?: Record<string, number>;
  last_alerts?: string[];
  ticks_total?: number;
  runs_total?: number;
  rows_inserted_total?: number;
  alerts_emitted_total?: number;
}

type Freshness = "fresh" | "stale" | "alert" | "never";

function classifyFreshness(
  status: FiveMTopupStatus | undefined,
  nowMs: number,
): Freshness {
  if (!status) return "never";
  if ((status.last_alerts?.length ?? 0) > 0) return "alert";
  const finishedAtSec = status.last_finished_at;
  if (!finishedAtSec) return "never";
  const ageHours = (nowMs / 1000 - finishedAtSec) / 3600;
  if (ageHours > 48) return "alert";
  if (ageHours >= 26) return "stale";
  return "fresh";
}

function formatRelative(epochSec: number | null | undefined, nowMs: number): string {
  if (!epochSec) return "never";
  const ageSec = Math.max(0, nowMs / 1000 - epochSec);
  if (ageSec < 60) return `${Math.round(ageSec)}s ago`;
  const ageMin = ageSec / 60;
  if (ageMin < 60) return `${Math.round(ageMin)}m ago`;
  const ageH = ageMin / 60;
  if (ageH < 48) return `${ageH.toFixed(1)}h ago`;
  return `${(ageH / 24).toFixed(1)}d ago`;
}

function formatAbsolute(epochSec: number | null | undefined): string {
  if (!epochSec) return "—";
  return new Date(epochSec * 1000).toLocaleString();
}

const FRESHNESS_LABEL: Record<Freshness, string> = {
  fresh: "Fresh",
  stale: "Stale",
  alert: "Alerting",
  never: "No tick yet",
};

const FRESHNESS_HELP: Record<Freshness, string> = {
  fresh: "last top-up under 26h ago",
  stale: "last top-up 26–48h ago",
  alert: "last top-up >48h ago or coins below alert threshold",
  never: "scheduler hasn't completed a tick since boot",
};

export function FiveMTopupStatusCard() {
  const { data, isLoading, isError } = useQuery<FiveMTopupStatus>({
    queryKey: ["five-m-topup-status"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/brain/5m-topup/status`);
      if (!res.ok) throw new Error(`5m-topup status ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  const nowMs = Date.now();
  const freshness = classifyFreshness(data, nowMs);
  const tone =
    freshness === "fresh"
      ? "bg-emerald-500/10 ring-1 ring-emerald-500/30"
      : freshness === "stale"
      ? "bg-amber-500/10 ring-1 ring-amber-500/30"
      : freshness === "alert"
      ? "bg-rose-500/10 ring-1 ring-rose-500/30"
      : "bg-card/30";
  const iconTone =
    freshness === "fresh"
      ? "text-emerald-300"
      : freshness === "stale"
      ? "text-amber-300"
      : freshness === "alert"
      ? "text-rose-300"
      : "text-muted-foreground";
  const FreshnessIcon = freshness === "fresh" ? CheckCircle2 : AlertTriangle;

  const perCoin = data?.last_topup_per_coin ?? {};
  const health = data?.last_health_per_coin ?? {};
  const alerts = data?.last_alerts ?? [];
  const alertSet = new Set(alerts);
  const coinIds = Array.from(
    new Set([...Object.keys(perCoin), ...Object.keys(health)]),
  ).sort();

  return (
    <Card className={cn("border-border/40", tone)} data-testid="five-m-topup-status-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Database className="w-4 h-4" />
          Daily 5m top-up
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            keeps `price_candles` head fresh so the 305-day gate stays clear
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-24 w-full" />}
        {isError && (
          <div
            className="text-sm text-rose-300 font-mono"
            data-testid="five-m-topup-status-error"
          >
            Couldn't load 5m top-up status.
          </div>
        )}
        {data && (
          <div className="space-y-3">
            <div className="flex items-end justify-between gap-4 flex-wrap">
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground">
                  Last finished
                </div>
                <div
                  className={cn(
                    "text-2xl font-display font-bold mt-1 flex items-center gap-2",
                    iconTone,
                  )}
                  data-testid="five-m-topup-status-freshness"
                >
                  <FreshnessIcon className="w-5 h-5" />
                  <span data-testid="five-m-topup-status-relative">
                    {formatRelative(data.last_finished_at, nowMs)}
                  </span>
                </div>
                <div
                  className="text-[11px] font-mono text-muted-foreground mt-0.5"
                  data-testid="five-m-topup-status-absolute"
                >
                  {formatAbsolute(data.last_finished_at)}
                </div>
              </div>
              <div className="flex flex-col items-end gap-1">
                <Badge
                  variant="outline"
                  className={cn(
                    "font-mono text-[10px]",
                    iconTone,
                  )}
                  data-testid="five-m-topup-status-badge"
                >
                  {FRESHNESS_LABEL[freshness]}
                </Badge>
                <div className="text-[10px] font-mono text-muted-foreground/80 max-w-[14rem] text-right">
                  {FRESHNESS_HELP[freshness]}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
              <div className="p-2 rounded-md bg-card/40 border border-border/30">
                <div className="uppercase text-muted-foreground text-[10px]">
                  Rows last tick
                </div>
                <div
                  className="text-base font-bold mt-0.5"
                  data-testid="five-m-topup-status-rows-last"
                >
                  {(data.last_topup_inserted ?? 0).toLocaleString()}
                </div>
              </div>
              <div className="p-2 rounded-md bg-card/40 border border-border/30">
                <div className="uppercase text-muted-foreground text-[10px]">
                  Outcome
                </div>
                <div
                  className="text-base font-bold mt-0.5 flex items-center gap-1"
                  data-testid="five-m-topup-status-outcome"
                >
                  <Clock className="w-3.5 h-3.5 text-muted-foreground" />
                  {data.last_attempt_outcome ?? "—"}
                </div>
              </div>
              <div className="p-2 rounded-md bg-card/40 border border-border/30">
                <div className="uppercase text-muted-foreground text-[10px]">
                  Total runs
                </div>
                <div
                  className="text-base font-bold mt-0.5"
                  data-testid="five-m-topup-status-runs-total"
                >
                  {(data.runs_total ?? 0).toLocaleString()}
                </div>
              </div>
              <div className="p-2 rounded-md bg-card/40 border border-border/30">
                <div className="uppercase text-muted-foreground text-[10px]">
                  Alert threshold
                </div>
                <div className="text-base font-bold mt-0.5">
                  {(data.alert_below_days ?? 0).toFixed(0)}d
                </div>
              </div>
            </div>

            {alerts.length > 0 && (
              <div
                className="p-2 rounded-md bg-rose-500/10 ring-1 ring-rose-500/20 text-xs"
                data-testid="five-m-topup-status-alerts"
              >
                <div className="text-[10px] uppercase font-mono text-rose-200/80 mb-1">
                  Coins below {(data.alert_below_days ?? 0).toFixed(0)}d contiguous
                </div>
                <div className="flex flex-wrap gap-1">
                  {alerts.map((c) => (
                    <Badge
                      key={c}
                      variant="outline"
                      className="font-mono text-[10px] text-rose-200 border-rose-400/40"
                      data-testid={`five-m-topup-status-alert-${c}`}
                    >
                      {c}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {data.last_error && (
              <div
                className="p-2 rounded-md bg-rose-500/10 ring-1 ring-rose-500/20 text-xs"
                data-testid="five-m-topup-status-last-error"
              >
                <div className="text-[10px] uppercase font-mono text-rose-200/80 mb-1">
                  Last error
                </div>
                <div className="font-mono text-rose-200 break-words">
                  {data.last_error}
                </div>
              </div>
            )}

            {coinIds.length > 0 && (
              <div className="overflow-x-auto">
                <table
                  className="w-full text-[11px] font-mono"
                  data-testid="five-m-topup-status-per-coin"
                >
                  <thead>
                    <tr className="text-[10px] uppercase text-muted-foreground">
                      <th className="text-left py-1 pr-3">Coin</th>
                      <th className="text-right py-1 pr-3">Rows added</th>
                      <th className="text-right py-1">Contiguous (d)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {coinIds.map((coin) => {
                      const rows = perCoin[coin] ?? 0;
                      const days = health[coin];
                      const isAlerting = alertSet.has(coin);
                      return (
                        <tr
                          key={coin}
                          className="border-t border-border/20"
                          data-testid={`five-m-topup-status-row-${coin}`}
                        >
                          <td className={cn("py-1 pr-3", isAlerting && "text-rose-300")}>
                            {coin}
                          </td>
                          <td className="py-1 pr-3 text-right">
                            {rows.toLocaleString()}
                          </td>
                          <td
                            className={cn(
                              "py-1 text-right",
                              isAlerting && "text-rose-300 font-bold",
                            )}
                            data-testid={`five-m-topup-status-row-${coin}-days`}
                          >
                            {typeof days === "number" ? days.toFixed(2) : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
