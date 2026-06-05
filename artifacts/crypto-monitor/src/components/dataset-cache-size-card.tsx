import { useEffect, useMemo, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  AlertTriangle,
  HardDrive,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/format";

interface RetentionSummary {
  policy?: string;
  retain_n?: number;
  retained?: number;
  deleted?: number;
  oldest_kept_at?: string | null;
  bytes_on_disk?: number;
  pinned_kept?: string[];
  deleted_files?: string[];
  skipped?: string;
}

interface TimeframeEntry {
  last_success_at?: string | null;
  last_attempt_at?: string | null;
  last_error?: string | null;
  next_due_at?: string | null;
  cadence_hours?: number;
  mtime_of_newest_snapshot?: string | null;
  retention?: RetentionSummary;
}

interface SizeHistoryPoint {
  at?: string;
  total_bytes?: number;
  per_tf?: Record<string, number>;
}

interface DatasetFreshness {
  status?: "ok" | "missing" | "error";
  error?: string;
  written_at?: string;
  total_bytes_on_disk?: number;
  cache_size_warn_bytes?: number;
  cache_size_warning?: boolean;
  cache_size_history?: SizeHistoryPoint[];
  timeframes?: Record<string, TimeframeEntry>;
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

// Stable timeframe ordering matches the trainer/refresher cadence
// table — keeps the rows visually consistent across refreshes.
const TF_ORDER = ["1m", "5m", "1h", "2h", "6h", "1d"];

function orderedTfs(tfs: string[]): string[] {
  const known = TF_ORDER.filter((k) => tfs.includes(k));
  const extras = tfs.filter((k) => !TF_ORDER.includes(k)).sort();
  return [...known, ...extras];
}

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  // Show one decimal below 100 for readability, integer beyond.
  const fmt = v >= 100 ? v.toFixed(0) : v.toFixed(1);
  return `${fmt} ${units[i]}`;
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  testId?: string;
  ariaLabel?: string;
}

function Sparkline({
  values,
  width = 120,
  height = 28,
  testId,
  ariaLabel,
}: SparklineProps) {
  if (values.length < 2) {
    return (
      <div
        className="text-[10px] text-muted-foreground/70 italic"
        data-testid={testId ? `${testId}-empty` : undefined}
      >
        {values.length === 1 ? "1 sample" : "no history yet"}
      </div>
    );
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  const stepX = width / (values.length - 1);
  const points = values
    .map((v, i) => {
      const x = i * stepX;
      // Invert Y so larger = higher in the SVG.
      const y = height - ((v - min) / span) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  // Trend colour: green if shrinking (post-trim), amber if growing,
  // grey if flat. Compares first vs last sample only.
  const first = values[0];
  const last = values[values.length - 1];
  let stroke = "rgba(148,163,184,0.85)";
  if (last > first * 1.05) stroke = "rgba(251,191,36,0.95)";
  else if (last < first * 0.95) stroke = "rgba(52,211,153,0.95)";
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
      data-testid={testId}
    >
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

export function DatasetCacheSizeCard() {
  const [data, setData] = useState<DatasetFreshness | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const r = await fetch(apiUrl("/crypto/dataset-freshness"), {
        cache: "no-store",
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = (await r.json()) as DatasetFreshness;
      setData(j);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
    // 60s poll matches the dataset-refresher's slowest sane cadence
    // (the scheduler ticks every 30 min by default, so anything faster
    // than 30s is wasted bandwidth — but 60s keeps the panel feeling
    // alive without hammering the ml-engine).
    const t = window.setInterval(() => void refresh(), 60_000);
    return () => window.clearInterval(t);
  }, []);

  const tfs = useMemo(() => {
    if (!data?.timeframes) return [] as string[];
    return orderedTfs(Object.keys(data.timeframes));
  }, [data]);

  const totalSeries = useMemo(() => {
    const hist = data?.cache_size_history ?? [];
    return hist
      .map((h) => h.total_bytes ?? 0)
      .filter((v) => Number.isFinite(v));
  }, [data]);

  const perTfSeries = useMemo(() => {
    const hist = data?.cache_size_history ?? [];
    const out: Record<string, number[]> = {};
    for (const tf of tfs) out[tf] = [];
    for (const h of hist) {
      for (const tf of tfs) {
        const v = h.per_tf?.[tf];
        if (Number.isFinite(v)) out[tf].push(v as number);
      }
    }
    return out;
  }, [data, tfs]);

  const totalBytes = data?.total_bytes_on_disk ?? null;
  const warnBytes = data?.cache_size_warn_bytes ?? 5 * 1024 * 1024 * 1024;
  const overWarn = !!data?.cache_size_warning ||
    (totalBytes != null && totalBytes > warnBytes);

  return (
    <Card data-testid="dataset-cache-size-card">
      <CardHeader className="flex flex-row items-start justify-between gap-2">
        <div>
          <CardTitle className="flex items-center gap-2 text-base">
            <HardDrive className="w-4 h-4" />
            Dataset cache footprint
          </CardTitle>
          <p className="text-xs text-muted-foreground mt-1 max-w-prose">
            Bytes-on-disk for every cached training-dataset timeframe
            after each scheduler trim. Confirms the cache stays capped
            at the documented ~14-day footprint and surfaces per-tf
            bloat (e.g. 1m parquets ballooning) before it bites.
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => void refresh()}
          disabled={loading}
          data-testid="dataset-cache-size-refresh"
          className="h-8"
        >
          <RefreshCw
            className={cn("w-3.5 h-3.5 mr-1", loading && "animate-spin")}
          />
          Refresh
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {err && (
          <div
            className="text-xs text-red-400"
            data-testid="dataset-cache-size-error"
          >
            Failed to load: {err}
          </div>
        )}

        {data?.status === "missing" && (
          <div
            className="text-xs text-muted-foreground"
            data-testid="dataset-cache-size-missing"
          >
            The <code>dataset-refresher</code> workflow has not written
            a freshness status yet. Once it ticks, per-tf cache sizes
            will appear here.
          </div>
        )}

        {data?.status === "error" && (
          <div
            className="text-xs text-red-400"
            data-testid="dataset-cache-size-server-error"
          >
            Server could not read the freshness status: {data.error ?? "unknown"}
          </div>
        )}

        {data?.status === "ok" && (
          <>
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border/60 bg-muted/20 p-3">
              <div className="flex flex-col">
                <span className="text-[10px] uppercase font-mono tracking-wider text-muted-foreground">
                  Total cache on disk
                </span>
                <div className="flex items-center gap-2 mt-0.5">
                  <span
                    className="text-2xl font-display font-bold"
                    data-testid="dataset-cache-size-total"
                  >
                    {formatBytes(totalBytes)}
                  </span>
                  {overWarn ? (
                    <Badge
                      variant="outline"
                      className="border-amber-500/50 bg-amber-500/10 text-amber-300 font-mono text-[10px]"
                      data-testid="dataset-cache-size-warning-chip"
                    >
                      <AlertTriangle className="w-3 h-3 mr-1" />
                      &gt; {formatBytes(warnBytes)}
                    </Badge>
                  ) : (
                    <Badge
                      variant="outline"
                      className="border-emerald-500/40 bg-emerald-500/10 text-emerald-300 font-mono text-[10px]"
                      data-testid="dataset-cache-size-ok-chip"
                    >
                      within {formatBytes(warnBytes)} cap
                    </Badge>
                  )}
                </div>
                <span className="text-[11px] text-muted-foreground mt-0.5">
                  {data.written_at
                    ? `Updated ${formatTimeAgo(data.written_at)}`
                    : "never updated"}
                </span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[10px] uppercase font-mono tracking-wider text-muted-foreground">
                  Total trend ({totalSeries.length} samples)
                </span>
                <Sparkline
                  values={totalSeries}
                  width={160}
                  height={36}
                  testId="dataset-cache-size-total-sparkline"
                  ariaLabel="Total cache size sparkline"
                />
              </div>
            </div>

            {tfs.length === 0 && (
              <div
                className="text-xs text-muted-foreground"
                data-testid="dataset-cache-size-no-tfs"
              >
                No timeframes recorded yet — wait for the next refresher
                tick to populate per-tf rows.
              </div>
            )}

            {tfs.length > 0 && (
              <div className="overflow-x-auto">
                <table
                  className="w-full text-xs"
                  data-testid="dataset-cache-size-table"
                >
                  <thead>
                    <tr className="text-left text-muted-foreground border-b border-border/40">
                      <th className="py-1.5 pr-3 font-mono uppercase text-[10px]">
                        TF
                      </th>
                      <th className="py-1.5 pr-3 font-mono uppercase text-[10px] text-right">
                        Cache size
                      </th>
                      <th className="py-1.5 pr-3 font-mono uppercase text-[10px] text-right">
                        Snapshots
                      </th>
                      <th className="py-1.5 pr-3 font-mono uppercase text-[10px]">
                        Oldest kept
                      </th>
                      <th className="py-1.5 pr-3 font-mono uppercase text-[10px]">
                        Trend
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tfs.map((tf) => {
                      const entry = data.timeframes?.[tf] ?? {};
                      const ret = entry.retention ?? {};
                      const series = perTfSeries[tf] ?? [];
                      return (
                        <tr
                          key={tf}
                          className="border-b border-border/20 last:border-b-0"
                          data-testid={`dataset-cache-size-row-${tf}`}
                        >
                          <td className="py-1.5 pr-3 font-mono">{tf}</td>
                          <td
                            className="py-1.5 pr-3 font-mono text-right"
                            data-testid={`dataset-cache-size-bytes-${tf}`}
                          >
                            {formatBytes(ret.bytes_on_disk)}
                          </td>
                          <td className="py-1.5 pr-3 font-mono text-right text-muted-foreground">
                            {ret.retained ?? "—"}
                            {ret.retain_n != null && ret.retain_n > 0 && (
                              <span className="text-muted-foreground/60">
                                /{ret.retain_n}
                              </span>
                            )}
                          </td>
                          <td className="py-1.5 pr-3 text-muted-foreground">
                            {formatTime(ret.oldest_kept_at)}
                          </td>
                          <td className="py-1.5 pr-3">
                            <Sparkline
                              values={series}
                              width={120}
                              height={22}
                              testId={`dataset-cache-size-sparkline-${tf}`}
                              ariaLabel={`${tf} cache size trend`}
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            <div className="text-[11px] text-muted-foreground">
              Sparklines show the last {totalSeries.length || 0} samples
              of post-trim cache size — green = shrinking, amber =
              growing, grey = flat. The full series caps at{" "}
              {data.cache_size_history?.length ?? 0} entries (override
              with <code>ML_REFRESH_SIZE_HISTORY_LEN</code>).
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
