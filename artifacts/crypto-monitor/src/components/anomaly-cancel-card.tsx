import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ShieldAlert, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface AnomalyCancelBucket {
  ts: number;
  count: number;
}

interface AnomalyCancelByCoin {
  coinId: string;
  coinName: string;
  count: number;
  lastAt: string | null;
}

interface AnomalyCancelResponse {
  windowHours: number;
  total: number;
  byCoin: AnomalyCancelByCoin[];
  buckets: AnomalyCancelBucket[];
  bucketMs: number;
  hourlyThreshold: number;
  alert: {
    hourlyThreshold: number;
    peakHourCount: number;
    spikingHourCount: number;
  } | null;
  fetchedAt: string;
}

function Sparkline({ buckets, threshold }: { buckets: AnomalyCancelBucket[]; threshold: number }) {
  const width = 240;
  const height = 44;
  const max = Math.max(threshold, ...buckets.map(b => b.count), 1);
  const barWidth = width / buckets.length;
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="block"
      data-testid="anomaly-cancel-sparkline"
    >
      <line
        x1={0}
        x2={width}
        y1={height - (threshold / max) * height}
        y2={height - (threshold / max) * height}
        stroke="currentColor"
        strokeOpacity={0.25}
        strokeDasharray="3 3"
        className="text-rose-400"
      />
      {buckets.map((b, i) => {
        const h = (b.count / max) * height;
        const isSpike = b.count >= threshold;
        return (
          <rect
            key={b.ts}
            x={i * barWidth + 0.5}
            y={height - h}
            width={Math.max(barWidth - 1, 1)}
            height={h}
            className={cn(isSpike ? "fill-rose-400" : "fill-amber-400/70")}
          >
            <title>{`${new Date(b.ts).toLocaleTimeString([], { hour: "numeric" })}: ${b.count} cancels`}</title>
          </rect>
        );
      })}
    </svg>
  );
}

export function AnomalyCancelCard() {
  const { data, isLoading, isError } = useQuery<AnomalyCancelResponse>({
    queryKey: ["anomaly-cancels"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/anomaly-cancels`);
      if (!res.ok) throw new Error(`anomaly-cancels ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  return (
    <Card className="bg-card/50 border-border/40" data-testid="anomaly-cancel-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          Anomaly-cancel safety net
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            trades reversed because the exit price looked like a feed glitch (outside 0.33×–3× entry)
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-24 w-full" />}
        {isError && (
          <div className="text-sm text-rose-300 font-mono" data-testid="anomaly-cancel-error">
            Couldn't load anomaly-cancel counts.
          </div>
        )}
        {data && (
          <div className="space-y-3">
            {data.alert && (
              <div
                className="flex items-start gap-2 p-2 rounded-md bg-rose-500/10 ring-1 ring-rose-500/30 text-rose-200 text-xs"
                data-testid="anomaly-cancel-alert"
              >
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <div>
                  <div className="font-medium">
                    Spike: {data.alert.peakHourCount} cancels in a single hour
                    {data.alert.spikingHourCount > 1 ? ` (${data.alert.spikingHourCount} hours over the line)` : ""}.
                  </div>
                  <div className="mt-0.5 text-rose-200/80">
                    Threshold is {data.alert.hourlyThreshold}/hr. A burst this size usually means a price feed is misbehaving, not real moves.
                  </div>
                </div>
              </div>
            )}

            <div className="flex items-end justify-between gap-4 flex-wrap">
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground">
                  Last {data.windowHours}h
                </div>
                <div
                  className={cn(
                    "text-3xl font-display font-bold mt-1",
                    data.total === 0 ? "text-muted-foreground" : data.alert ? "text-rose-300" : "text-amber-300",
                  )}
                  data-testid="anomaly-cancel-total"
                >
                  {data.total}
                </div>
                <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
                  {data.total === 0
                    ? "safety net hasn't fired — feed looks clean"
                    : data.total === 1
                      ? "1 cancel"
                      : `${data.total} cancels`}
                </div>
              </div>
              <div className="text-rose-300/80" data-testid="anomaly-cancel-spark-wrap">
                <Sparkline buckets={data.buckets} threshold={data.hourlyThreshold} />
                <div className="flex justify-between text-[10px] font-mono text-muted-foreground/70 mt-0.5">
                  <span>−24h</span>
                  <span>now</span>
                </div>
              </div>
            </div>

            {data.byCoin.length > 0 && (
              <div className="pt-2 border-t border-border/20">
                <div className="text-[10px] uppercase font-mono text-muted-foreground mb-1.5">
                  By coin
                </div>
                <div className="flex flex-wrap gap-1.5" data-testid="anomaly-cancel-by-coin">
                  {data.byCoin.map(c => (
                    <Badge
                      key={c.coinId}
                      variant="outline"
                      className="font-mono text-[11px] border-border/40"
                      data-testid={`anomaly-cancel-coin-${c.coinId}`}
                      title={c.lastAt ? `last cancel ${new Date(c.lastAt).toLocaleString()}` : undefined}
                    >
                      {c.coinName} <span className="ml-1 text-rose-300">×{c.count}</span>
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            <div className="text-[10px] font-mono text-muted-foreground/60">
              refreshes every 60s · last {new Date(data.fetchedAt).toLocaleTimeString()}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
