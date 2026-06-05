import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Database, RefreshCw, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface JournalHealth {
  windowHours: number;
  predictions: {
    total: number;
    perDay: number;
    resolved: number;
    resolvedPct: number;
    becameTrade: number;
    becameTradePct: number;
  };
  trades: {
    total: number;
    perDay: number;
    withMaeMfe: number;
    maeMfeCoveragePct: number;
    withFees: number;
    feesCoveragePct: number;
  };
  features?: {
    pythonCalls: number;
    tsFallbackCalls: number;
    pythonPct: number;
  };
  synthesizedFeatureHashes?: {
    windowHours: number;
    total: number;
    threshold: number;
    byKey: Array<{
      source: string;
      timeframe: string;
      count: number;
      lastSeen: string;
    }>;
  };
}

interface JournalHealthSeriesPoint {
  bucketStart: string;
  predictions: number;
  resolvedPct: number;
  trades: number;
  maeMfeCoveragePct: number;
}

interface JournalHealthSeries {
  windowHours: number;
  bucketSeconds: number;
  points: JournalHealthSeriesPoint[];
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

function pctBadge(pct: number, label: string) {
  const tone =
    pct >= 90
      ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/40"
      : pct >= 60
        ? "bg-amber-500/10 text-amber-400 border-amber-500/40"
        : "bg-red-500/10 text-red-400 border-red-500/40";
  return (
    <Badge variant="outline" className={cn("rounded-full px-2 py-0.5 text-xs", tone)}>
      {pct.toFixed(0)}% {label}
    </Badge>
  );
}

const WINDOW_OPTIONS: { value: number; label: string }[] = [
  { value: 1, label: "1h" },
  { value: 24, label: "24h" },
  { value: 168, label: "7d" },
];
const STORAGE_KEY = "journal-health-window-hours";

function loadStoredHours(): number {
  if (typeof window === "undefined") return 24;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return 24;
    const parsed = Number.parseInt(raw, 10);
    if (WINDOW_OPTIONS.some((o) => o.value === parsed)) return parsed;
  } catch {
    // ignore storage errors
  }
  return 24;
}

function formatBucketSize(seconds: number): string {
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86_400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86_400)}d`;
}

function formatTickLabel(iso: string, bucketSeconds: number): string {
  const d = new Date(iso);
  if (bucketSeconds >= 86_400 || bucketSeconds >= 6 * 3600) {
    return d.toLocaleDateString([], { month: "numeric", day: "numeric" });
  }
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

interface SparklineSeriesSpec {
  key: keyof Pick<JournalHealthSeriesPoint, "predictions" | "resolvedPct" | "maeMfeCoveragePct">;
  label: string;
  color: string;
  unit: "count" | "pct";
}

const SERIES_SPECS: SparklineSeriesSpec[] = [
  { key: "predictions", label: "predictions", color: "#60a5fa", unit: "count" },
  { key: "resolvedPct", label: "resolved %", color: "#34d399", unit: "pct" },
  { key: "maeMfeCoveragePct", label: "MAE/MFE %", color: "#f59e0b", unit: "pct" },
];

function JournalHealthSparkline({
  series,
}: {
  series: JournalHealthSeries;
}) {
  const width = 560;
  const height = 96;
  const padX = 4;
  const padY = 6;

  const points = series.points;
  const maxPredictions = Math.max(1, ...points.map((p) => p.predictions));

  const xFor = (i: number) => {
    if (points.length <= 1) return padX;
    return padX + (i / (points.length - 1)) * (width - padX * 2);
  };

  const yForPct = (pct: number) =>
    height - padY - (Math.max(0, Math.min(100, pct)) / 100) * (height - padY * 2);
  const yForCount = (n: number) =>
    height - padY - (Math.max(0, n) / maxPredictions) * (height - padY * 2);

  function pathFor(spec: SparklineSeriesSpec): string {
    return points
      .map((p, i) => {
        const x = xFor(i);
        const v = p[spec.key] as number;
        const y = spec.unit === "pct" ? yForPct(v) : yForCount(v);
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }

  // X tick labels: first, middle, last
  const tickIdxs = useMemo(() => {
    if (points.length === 0) return [] as number[];
    if (points.length === 1) return [0];
    return [0, Math.floor(points.length / 2), points.length - 1];
  }, [points.length]);

  if (points.length === 0) {
    return (
      <div
        className="text-xs text-muted-foreground py-4 text-center"
        data-testid="journal-health-sparkline-empty"
      >
        No journal activity in the selected window.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">
          Trend over last {series.windowHours}h
        </div>
        <div className="flex flex-wrap gap-3">
          {SERIES_SPECS.map((s) => (
            <div key={s.key} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <span
                className="inline-block h-1.5 w-3 rounded-sm"
                style={{ backgroundColor: s.color }}
              />
              {s.label}
            </div>
          ))}
        </div>
      </div>
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="block w-full"
        data-testid="journal-health-sparkline"
      >
        {/* gridline at 50% / 100% for the % axis */}
        <line
          x1={padX}
          x2={width - padX}
          y1={yForPct(100)}
          y2={yForPct(100)}
          stroke="currentColor"
          strokeOpacity={0.08}
        />
        <line
          x1={padX}
          x2={width - padX}
          y1={yForPct(50)}
          y2={yForPct(50)}
          stroke="currentColor"
          strokeOpacity={0.08}
          strokeDasharray="2 3"
        />
        {SERIES_SPECS.map((s) => (
          <path
            key={s.key}
            d={pathFor(s)}
            fill="none"
            stroke={s.color}
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
            data-testid={`journal-health-sparkline-${s.key}`}
          />
        ))}
      </svg>
      <div className="flex justify-between text-[10px] text-muted-foreground/80 font-mono">
        {tickIdxs.map((i) => (
          <span key={i}>{formatTickLabel(points[i].bucketStart, series.bucketSeconds)}</span>
        ))}
      </div>
      <div className="text-[10px] text-muted-foreground/70">
        Bucket size: {formatBucketSize(series.bucketSeconds)} · {points.length} points · % axis is
        0–100; predictions axis scales to peak ({maxPredictions}).
      </div>
    </div>
  );
}

function formatRelativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return iso;
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86_400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86_400)}d ago`;
}

function SynthesizedFeatureHashBanner({
  info,
}: {
  info: NonNullable<JournalHealth["synthesizedFeatureHashes"]>;
}) {
  const overThreshold = info.total >= info.threshold;
  const tone = overThreshold
    ? "border-red-500/40 bg-red-500/10 text-red-300"
    : "border-amber-500/40 bg-amber-500/10 text-amber-300";
  const iconColor = overThreshold ? "text-red-400" : "text-amber-400";
  return (
    <div
      className={cn(
        "rounded-md border px-3 py-2 text-xs space-y-2",
        tone,
      )}
      data-testid="journal-health-synth-feature-hash-banner"
    >
      <div className="flex items-start gap-2">
        <AlertTriangle className={cn("h-3.5 w-3.5 mt-0.5 shrink-0", iconColor)} />
        <div className="space-y-0.5">
          <div className="font-medium">
            Synthesized feature_hash placeholders:{" "}
            <span data-testid="journal-health-synth-feature-hash-total">
              {info.total}
            </span>{" "}
            in last {info.windowHours}h
          </div>
          <div className="text-[11px] opacity-80">
            QUANT rows landed without an upstream feature_hash and were
            written with a <code>missing:&hellip;</code> placeholder.
            Healthy operation = 0;{" "}
            {overThreshold
              ? `above the soft-alert threshold of ${info.threshold}/h — check the predictor contract.`
              : `below the soft-alert threshold of ${info.threshold}/h.`}
          </div>
        </div>
      </div>
      <ul className="space-y-1 pl-5 font-mono text-[11px]">
        {info.byKey.map((row) => (
          <li
            key={`${row.source}:${row.timeframe}`}
            className="flex items-center justify-between gap-3"
            data-testid={`journal-health-synth-feature-hash-row-${row.source}-${row.timeframe}`}
          >
            <span className="truncate">
              {row.source}
              <span className="opacity-60"> · </span>
              {row.timeframe}
            </span>
            <span className="opacity-90 whitespace-nowrap">
              {row.count} ·{" "}
              <span className="opacity-70">last {formatRelativeTime(row.lastSeen)}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function JournalHealthCard() {
  const [data, setData] = useState<JournalHealth | null>(null);
  const [series, setSeries] = useState<JournalHealthSeries | null>(null);
  const [hours, setHours] = useState<number>(() => loadStoredHours());
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(STORAGE_KEY, String(hours));
    } catch {
      // ignore storage errors
    }
  }, [hours]);

  async function refresh(h = hours) {
    setLoading(true);
    try {
      const [healthRes, seriesRes] = await Promise.all([
        fetch(apiUrl(`/crypto/journal-health?hours=${h}`)),
        fetch(apiUrl(`/crypto/journal-health-series?hours=${h}`)),
      ]);
      if (!healthRes.ok) throw new Error(`HTTP ${healthRes.status}`);
      setData(await healthRes.json());
      if (seriesRes.ok) {
        setSeries(await seriesRes.json());
      } else {
        setSeries(null);
      }
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh(hours);
    const id = setInterval(() => void refresh(hours), 30_000);
    return () => clearInterval(id);
  }, [hours]);

  return (
    <Card data-testid="journal-health-card">
      <CardHeader className="flex flex-row items-start justify-between gap-3 pb-3">
        <div className="space-y-1">
          <CardTitle className="flex items-center gap-2 text-lg font-display">
            <Database className="h-5 w-5" />
            Journal Health
          </CardTitle>
          <p className="text-xs text-muted-foreground max-w-md">
            Adaptive-engine prediction & trade journals. Tracks write rate,
            resolution coverage, and MAE/MFE coverage so we know dashboards
            in later phases will have complete data.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {WINDOW_OPTIONS.map((opt) => (
            <Button
              key={opt.value}
              size="sm"
              variant={hours === opt.value ? "default" : "outline"}
              className="h-7 rounded-full text-xs"
              onClick={() => setHours(opt.value)}
              data-testid={`journal-health-window-${opt.value}`}
            >
              {opt.label}
            </Button>
          ))}
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7"
            onClick={() => void refresh(hours)}
            disabled={loading}
            aria-label="Refresh"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {err && (
          <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-400">
            <AlertTriangle className="h-3.5 w-3.5" />
            {err}
          </div>
        )}

        {data?.synthesizedFeatureHashes &&
          data.synthesizedFeatureHashes.total > 0 && (
            <SynthesizedFeatureHashBanner
              info={data.synthesizedFeatureHashes}
            />
          )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-lg border border-border bg-muted/20 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
              Predictions
            </div>
            <div className="text-2xl font-semibold mb-1" data-testid="journal-health-pred-perday">
              {data?.predictions.perDay ?? "—"}
              <span className="text-sm text-muted-foreground font-normal"> /day</span>
            </div>
            <div className="text-xs text-muted-foreground mb-2">
              {data?.predictions.total ?? 0} total in last {data?.windowHours ?? hours}h
            </div>
            <div className="flex flex-wrap gap-2">
              {data && pctBadge(data.predictions.resolvedPct, "resolved")}
              {data && pctBadge(data.predictions.becameTradePct, "→ trade")}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-muted/20 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
              Trades
            </div>
            <div className="text-2xl font-semibold mb-1" data-testid="journal-health-trade-perday">
              {data?.trades.perDay ?? "—"}
              <span className="text-sm text-muted-foreground font-normal"> /day</span>
            </div>
            <div className="text-xs text-muted-foreground mb-2">
              {data?.trades.total ?? 0} closed in last {data?.windowHours ?? hours}h
            </div>
            <div className="flex flex-wrap gap-2">
              {data && pctBadge(data.trades.maeMfeCoveragePct, "MAE/MFE")}
              {data && pctBadge(data.trades.feesCoveragePct, "fees")}
            </div>
          </div>
        </div>

        {series && (
          <div className="rounded-lg border border-border bg-muted/20 p-3">
            <JournalHealthSparkline series={series} />
          </div>
        )}

        {data?.features && (
          <div className="rounded-lg border border-border bg-muted/20 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
              Feature Source (Python vs TS Fallback)
            </div>
            <div className="flex flex-wrap items-center gap-3">
              {pctBadge(data.features.pythonPct, "Python canonical")}
              <span
                className="text-xs text-muted-foreground"
                data-testid="journal-health-feature-counts"
              >
                {data.features.pythonCalls} Python · {data.features.tsFallbackCalls} TS fallback
                (since server start)
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
