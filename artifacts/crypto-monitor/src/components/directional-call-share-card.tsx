import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertTriangle, RefreshCw, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

interface HistoryPoint {
  generated_at: string;
  coin_id: string;
  timeframe: string;
  version: string | null;
  directional_call_share_pct: number;
  n_predictions: number;
  source: string | null;
  n_train_rows: number;
  tradeable_timeframe: boolean;
  below_threshold: boolean;
  threshold_pct: number;
}

interface Series {
  timeframe: string;
  coinId: string;
  points: HistoryPoint[];
  latestSharePct: number | null;
  latestVersion: string | null;
  latestAt: string | null;
  tradeableTimeframe: boolean;
}

interface AlertEntry {
  timeframe: string;
  coinId: string;
  sharePct: number;
  version: string | null;
  thresholdPct: number;
  generatedAt: string;
}

interface DirectionalShareResp {
  thresholdPct: number;
  tradeableTimeframes: string[];
  totalRecords: number;
  returnedRecords: number;
  series: Series[];
  alerts: AlertEntry[];
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;
const TF_ORDER = ["1m", "5m", "1h", "2h", "6h", "1d"];

function tfRank(tf: string) {
  const i = TF_ORDER.indexOf(tf);
  return i === -1 ? 99 : i;
}

function Sparkline({
  points,
  threshold,
  tradeable,
}: {
  points: HistoryPoint[];
  threshold: number;
  tradeable: boolean;
}) {
  if (points.length === 0) {
    return <span className="text-[10px] text-muted-foreground">no history</span>;
  }
  const w = 120;
  const h = 32;
  const max = 100;
  const min = 0;
  const xStep = points.length > 1 ? w / (points.length - 1) : 0;
  const path = points
    .map((p, i) => {
      const x = points.length === 1 ? w / 2 : i * xStep;
      const y = h - ((p.directional_call_share_pct - min) / (max - min)) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const thresholdY = h - (threshold / max) * h;
  const last = points[points.length - 1];
  const lastX = points.length === 1 ? w / 2 : (points.length - 1) * xStep;
  const lastY = h - (last.directional_call_share_pct / max) * h;
  const stroke = last.below_threshold ? "#ef4444" : tradeable ? "#10b981" : "#9ca3af";
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="block">
      {tradeable && (
        <line
          x1={0}
          y1={thresholdY}
          x2={w}
          y2={thresholdY}
          stroke="#f59e0b"
          strokeDasharray="2,2"
          strokeWidth={1}
          opacity={0.6}
        />
      )}
      <path d={path} stroke={stroke} strokeWidth={1.5} fill="none" />
      <circle cx={lastX} cy={lastY} r={2.5} fill={stroke} />
    </svg>
  );
}

export function DirectionalCallShareCard() {
  const [data, setData] = useState<DirectionalShareResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const res = await fetch(apiUrl(`/crypto/quant-directional-history?limit=2000`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
    const id = setInterval(() => void refresh(), 60_000);
    return () => clearInterval(id);
  }, []);

  const grouped = useMemo(() => {
    const byTf = new Map<string, Series[]>();
    for (const s of data?.series ?? []) {
      if (!byTf.has(s.timeframe)) byTf.set(s.timeframe, []);
      byTf.get(s.timeframe)!.push(s);
    }
    return Array.from(byTf.entries())
      .map(([tf, rows]) => ({
        timeframe: tf,
        rows: rows.sort((a, b) => a.coinId.localeCompare(b.coinId)),
      }))
      .sort((a, b) => tfRank(a.timeframe) - tfRank(b.timeframe));
  }, [data]);

  const alerts = data?.alerts ?? [];

  return (
    <Card data-testid="directional-call-share-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Directional-call share over time
          {data && (
            <Badge variant="outline" className="ml-2">
              floor {data.thresholdPct}%
            </Badge>
          )}
        </CardTitle>
        <Button size="sm" variant="ghost" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {err && <div className="text-xs text-red-400">Error: {err}</div>}

        {alerts.length > 0 && (
          <div
            className="rounded-md border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-300 flex gap-2"
            data-testid="directional-share-alert"
          >
            <AlertTriangle className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-semibold">
                Directional-call share regression on {alerts.length} slice
                {alerts.length === 1 ? "" : "s"}.
              </div>
              <div className="text-[11px] text-red-200/80">
                Latest training run dropped these tradeable timeframes below{" "}
                {data?.thresholdPct}%. The model is mostly emitting STABLE again — check
                the last label/threshold change before deploying.
              </div>
              <ul className="text-[11px] text-red-200 list-disc list-inside">
                {alerts.slice(0, 8).map((a) => (
                  <li key={`${a.timeframe}-${a.coinId}`}>
                    <span className="font-mono">{a.timeframe}</span> · {a.coinId} ={" "}
                    {a.sharePct}% (v{a.version})
                  </li>
                ))}
                {alerts.length > 8 && <li>…and {alerts.length - 8} more</li>}
              </ul>
            </div>
          </div>
        )}

        {grouped.length === 0 && !err && (
          <div className="text-xs text-muted-foreground">
            No training-history entries yet. Run a training pass to seed the chart.
          </div>
        )}

        {grouped.map(({ timeframe, rows }) => (
          <div key={timeframe} className="space-y-1">
            <div className="flex items-center gap-2 text-xs">
              <span className="font-mono text-muted-foreground">{timeframe}</span>
              {rows[0]?.tradeableTimeframe ? (
                <Badge variant="outline" className="text-[10px] border-emerald-500/40 text-emerald-300">
                  tradeable
                </Badge>
              ) : (
                <Badge variant="outline" className="text-[10px] text-muted-foreground">
                  not tradeable
                </Badge>
              )}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-muted-foreground">
                    <th className="py-1 pr-3">Coin</th>
                    <th className="py-1 pr-3 text-right">Latest %</th>
                    <th className="py-1 pr-3 text-right">Samples</th>
                    <th className="py-1 pr-3">Trend</th>
                    <th className="py-1 pr-3 text-right">Runs</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((s) => {
                    const latest = s.points[s.points.length - 1];
                    const isAlert = latest?.below_threshold;
                    return (
                      <tr
                        key={`${s.timeframe}-${s.coinId}`}
                        className="border-t border-border/30"
                        data-testid={`dcs-row-${s.timeframe}-${s.coinId}`}
                      >
                        <td className="py-1 pr-3 font-mono">{s.coinId}</td>
                        <td
                          className={cn(
                            "py-1 pr-3 text-right tabular-nums font-semibold",
                            isAlert ? "text-red-400" : "",
                          )}
                        >
                          {s.latestSharePct != null ? `${s.latestSharePct}%` : "—"}
                        </td>
                        <td className="py-1 pr-3 text-right tabular-nums text-muted-foreground">
                          {latest?.n_predictions ?? 0}
                        </td>
                        <td className="py-1 pr-3">
                          <Sparkline
                            points={s.points}
                            threshold={data?.thresholdPct ?? 15}
                            tradeable={s.tradeableTimeframe}
                          />
                        </td>
                        <td className="py-1 pr-3 text-right tabular-nums text-muted-foreground">
                          {s.points.length}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ))}

        <div className="text-[10px] text-muted-foreground">
          Each point = one training run. The model's directional-call share is the
          fraction of holdout predictions where it picks UP or DOWN over STABLE. The
          dashed line marks the regression floor.
        </div>
      </CardContent>
    </Card>
  );
}
