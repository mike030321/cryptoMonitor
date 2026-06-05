import { useEffect, useMemo, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Activity, AlertTriangle, CheckCircle2, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";

interface CurrentlyFitting {
  coin?: string | null;
  timeframe?: string | null;
  index?: number | null;
  total?: number | null;
  started_at?: string | null;
  headline?: string | null;
}

interface ProgressEvent {
  emitted_at?: string;
  phase?: string;
  status?: string;
  headline?: string;
  coin?: string;
  timeframe?: string;
}

interface ProgressSlice {
  coin?: string | null;
  timeframe?: string | null;
  index?: number | null;
  total?: number | null;
  started_at?: string | null;
  ended_at?: string | null;
  elapsed_sec?: number | null;
  status?: string | null;
}

interface ProgressIdleGap {
  from?: string | null;
  to?: string | null;
  duration_sec?: number | null;
  after_coin?: string | null;
  before_coin?: string | null;
}

interface TrainingProgress {
  status?: string;
  latest_emitted_at?: string | null;
  latest_phase?: string | null;
  latest_status?: string | null;
  latest_headline?: string | null;
  stale?: boolean;
  stale_seconds?: number | null;
  current_timeframe?: string | null;
  currently_fitting?: CurrentlyFitting | null;
  run_finished?: boolean;
  recent?: ProgressEvent[];
  slices?: ProgressSlice[];
  idle_gaps?: ProgressIdleGap[];
  error?: string;
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

function formatDuration(secs: number | null | undefined): string {
  if (secs == null || !isFinite(secs)) return "—";
  const s = Math.max(0, Math.floor(secs));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return mm ? `${h}h ${mm}m` : `${h}h`;
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

export function TrainingProgressCard() {
  const [data, setData] = useState<TrainingProgress | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const r = await fetch(apiUrl("/crypto/training/progress"), {
        cache: "no-store",
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = (await r.json()) as TrainingProgress;
      setData(j);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // 15s poll matches the trainer's slowest heartbeat cadence so the
    // "currently fitting <coin>/<tf>" line stays fresh without hammering
    // the ml-engine. The server-side stale check uses 5 minutes.
    const t = window.setInterval(refresh, 15_000);
    return () => window.clearInterval(t);
  }, []);

  const fitting = data?.currently_fitting;
  const stale = !!data?.stale;
  const runFinished = !!data?.run_finished;

  const counterText = useMemo(() => {
    if (!fitting) return null;
    const { index, total } = fitting;
    if (index == null || total == null) return null;
    return `${index} / ${total}`;
  }, [fitting]);

  const progressPct = useMemo(() => {
    if (!fitting?.index || !fitting?.total) return null;
    return Math.max(0, Math.min(100, (fitting.index / fitting.total) * 100));
  }, [fitting]);

  return (
    <Card data-testid="training-progress-card">
      <CardHeader className="flex flex-row items-start justify-between gap-2">
        <div>
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="w-4 h-4" />
            Live training progress
          </CardTitle>
          <p className="text-xs text-muted-foreground mt-1 max-w-prose">
            Per-coin heartbeat from the active training campaign — the
            same stream operators see when tailing{" "}
            <code className="text-[11px]">models/progress_updates.jsonl</code>.
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={refresh}
          disabled={loading}
          data-testid="training-progress-refresh"
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
            data-testid="training-progress-error"
          >
            Failed to load: {err}
          </div>
        )}

        {data?.status === "missing" && (
          <div
            className="text-xs text-muted-foreground"
            data-testid="training-progress-missing"
          >
            No training campaign has run yet —{" "}
            <code>models/progress_updates.jsonl</code> doesn't exist.
          </div>
        )}

        {data?.status === "empty" && (
          <div
            className="text-xs text-muted-foreground"
            data-testid="training-progress-empty"
          >
            Progress journal exists but has no rows yet.
          </div>
        )}

        {data?.status === "error" && (
          <div
            className="text-xs text-red-400"
            data-testid="training-progress-server-error"
          >
            Server could not read the journal: {data.error ?? "unknown error"}
          </div>
        )}

        {data?.status === "ok" && (
          <>
            {stale && !runFinished && (
              <div
                className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3"
                data-testid="training-progress-stale-warning"
              >
                <AlertTriangle className="w-4 h-4 text-amber-300 mt-0.5 shrink-0" />
                <div className="text-xs text-amber-200">
                  <div className="font-semibold">
                    Run looks stalled — no heartbeat for{" "}
                    {formatDuration(data.stale_seconds)}.
                  </div>
                  <div className="text-amber-200/80 mt-0.5">
                    The trainer normally writes a row per coin slice. If
                    nothing new appears for over 5 minutes the campaign
                    is likely hung; check the ml-engine logs before
                    trusting the progress shown below.
                  </div>
                </div>
              </div>
            )}

            {runFinished && (
              <div
                className="flex items-start gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 p-3"
                data-testid="training-progress-finished"
              >
                <CheckCircle2 className="w-4 h-4 text-emerald-300 mt-0.5 shrink-0" />
                <div className="text-xs text-emerald-200">
                  <div className="font-semibold">
                    Last campaign finished.
                  </div>
                  <div className="text-emerald-200/80 mt-0.5">
                    Latest event:{" "}
                    {data.latest_headline ?? data.latest_phase ?? "done"}
                  </div>
                </div>
              </div>
            )}

            {fitting && !runFinished && (
              <div
                className="rounded-md border border-border/60 bg-muted/20 p-3 space-y-2"
                data-testid="training-progress-fitting"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-sm">
                    <span className="text-muted-foreground">
                      Currently fitting:
                    </span>{" "}
                    <span
                      className="font-mono font-semibold"
                      data-testid="training-progress-fitting-slice"
                    >
                      {fitting.coin ?? "?"}
                      {fitting.timeframe ? `/${fitting.timeframe}` : ""}
                    </span>
                  </div>
                  {counterText && (
                    <Badge
                      variant="secondary"
                      className="font-mono"
                      data-testid="training-progress-fitting-counter"
                    >
                      {counterText}
                    </Badge>
                  )}
                </div>
                {progressPct != null && (
                  <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all"
                      style={{ width: `${progressPct}%` }}
                      data-testid="training-progress-fitting-bar"
                    />
                  </div>
                )}
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
                  <span>
                    Started:{" "}
                    <span data-testid="training-progress-fitting-started">
                      {formatTime(fitting.started_at)}
                    </span>
                  </span>
                  {data.current_timeframe && (
                    <span>
                      Timeframe context:{" "}
                      <span className="font-mono">
                        {data.current_timeframe}
                      </span>
                    </span>
                  )}
                </div>
              </div>
            )}

            {!fitting && !runFinished && data.latest_headline && (
              <div
                className="text-xs text-muted-foreground"
                data-testid="training-progress-between-slices"
              >
                Between per-coin slices — last event:{" "}
                <span className="font-mono">{data.latest_headline}</span>
              </div>
            )}

            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
              <span>
                Last heartbeat:{" "}
                <span data-testid="training-progress-latest-at">
                  {formatTime(data.latest_emitted_at)}
                </span>
                {data.stale_seconds != null && (
                  <>
                    {" "}
                    <span data-testid="training-progress-stale-seconds">
                      ({formatDuration(data.stale_seconds)} ago)
                    </span>
                  </>
                )}
              </span>
              {data.latest_phase && (
                <span>
                  Latest phase:{" "}
                  <span className="font-mono">{data.latest_phase}</span>
                </span>
              )}
            </div>

            {data.slices && data.slices.length > 0 && (
              <TrainingTimelineChart
                slices={data.slices}
                idleGaps={data.idle_gaps ?? []}
              />
            )}

            {data.recent && data.recent.length > 0 && (
              <details className="text-xs">
                <summary
                  className="cursor-pointer text-muted-foreground select-none"
                  data-testid="training-progress-recent-toggle"
                >
                  Recent events ({data.recent.length})
                </summary>
                <ol
                  className="mt-2 space-y-1 max-h-64 overflow-auto"
                  data-testid="training-progress-recent-list"
                >
                  {data.recent.map((ev, i) => (
                    <li
                      key={`${ev.emitted_at ?? i}-${ev.phase ?? i}-${i}`}
                      className="flex gap-2 font-mono text-[11px] text-muted-foreground"
                      data-testid={`training-progress-recent-row-${i}`}
                    >
                      <span className="shrink-0 text-muted-foreground/70">
                        {ev.emitted_at
                          ? new Date(ev.emitted_at).toLocaleTimeString()
                          : "—"}
                      </span>
                      <span className="shrink-0 text-foreground/80">
                        {ev.phase ?? "?"}
                      </span>
                      <span className="truncate">{ev.headline ?? ""}</span>
                    </li>
                  ))}
                </ol>
              </details>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// Tailwind colour pairs (fill + border) per timeframe band so the run-shape
// across 5m / 1h / 2h / 6h / 1d is readable at a glance.
const TIMEFRAME_COLOURS: Record<string, { fill: string; stroke: string; label: string }> = {
  "5m": { fill: "#f59e0b", stroke: "#b45309", label: "5m" },
  "1h": { fill: "#3b82f6", stroke: "#1d4ed8", label: "1h" },
  "2h": { fill: "#8b5cf6", stroke: "#6d28d9", label: "2h" },
  "6h": { fill: "#10b981", stroke: "#047857", label: "6h" },
  "1d": { fill: "#f43f5e", stroke: "#be123c", label: "1d" },
};
const UNKNOWN_TF_COLOUR = { fill: "#6b7280", stroke: "#374151", label: "?" };

function colourFor(tf: string | null | undefined) {
  if (!tf) return UNKNOWN_TF_COLOUR;
  return TIMEFRAME_COLOURS[tf] ?? UNKNOWN_TF_COLOUR;
}

interface TimelineProps {
  slices: ProgressSlice[];
  idleGaps: ProgressIdleGap[];
}

function TrainingTimelineChart({ slices, idleGaps }: TimelineProps) {
  const parsed = useMemo(() => {
    const out: Array<{
      coin: string;
      timeframe: string | null;
      start: number;
      end: number;
      status: string;
    }> = [];
    const now = Date.now();
    for (const s of slices) {
      if (!s.started_at) continue;
      const start = Date.parse(s.started_at);
      if (Number.isNaN(start)) continue;
      const endRaw = s.ended_at ? Date.parse(s.ended_at) : NaN;
      // For still-running slices anchor the bar to "now" so it grows live.
      const end = Number.isFinite(endRaw) ? endRaw : now;
      out.push({
        coin: s.coin ?? "?",
        timeframe: s.timeframe ?? null,
        start,
        end: Math.max(end, start + 1_000),
        status: s.status ?? "done",
      });
    }
    return out;
  }, [slices]);

  const gaps = useMemo(() => {
    const out: Array<{ start: number; end: number; durationSec: number }> = [];
    for (const g of idleGaps) {
      if (!g.from || !g.to) continue;
      const start = Date.parse(g.from);
      const end = Date.parse(g.to);
      if (Number.isNaN(start) || Number.isNaN(end) || end <= start) continue;
      out.push({ start, end, durationSec: g.duration_sec ?? (end - start) / 1000 });
    }
    return out;
  }, [idleGaps]);

  // Stable timeframe row order — known timeframes first in canonical
  // order, then any extras alphabetically. Keeps the chart visually
  // consistent across refreshes.
  const rows = useMemo(() => {
    const known = ["5m", "1h", "2h", "6h", "1d"];
    const seen = new Set<string>();
    for (const p of parsed) seen.add(p.timeframe ?? "?");
    const ordered = known.filter((k) => seen.has(k));
    const extras = Array.from(seen)
      .filter((k) => !known.includes(k))
      .sort();
    return [...ordered, ...extras];
  }, [parsed]);

  if (parsed.length === 0) return null;

  const minT = Math.min(...parsed.map((p) => p.start));
  const maxT = Math.max(...parsed.map((p) => p.end));
  const span = Math.max(1, maxT - minT);

  const width = 720;
  const rowHeight = 22;
  const rowGap = 6;
  const leftPad = 44;
  const rightPad = 8;
  const topPad = 8;
  const innerWidth = width - leftPad - rightPad;
  const height = topPad + rows.length * (rowHeight + rowGap);

  const xFor = (t: number) => leftPad + ((t - minT) / span) * innerWidth;
  const yFor = (tf: string) => {
    const idx = rows.indexOf(tf);
    return topPad + idx * (rowHeight + rowGap);
  };

  // Tick labels at start, midpoint, and end of the run window.
  const ticks = [0, 0.5, 1].map((f) => {
    const t = minT + span * f;
    const x = leftPad + f * innerWidth;
    const label = new Date(t).toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    return { x, label };
  });

  const totalSliceMs = parsed.reduce((acc, p) => acc + (p.end - p.start), 0);
  const idleMs = gaps.reduce((acc, g) => acc + (g.end - g.start), 0);

  return (
    <div
      className="rounded-md border border-border/60 bg-muted/10 p-3 space-y-2"
      data-testid="training-progress-timeline"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs text-muted-foreground">
          Per-slice timeline ({parsed.length} slices,{" "}
          {gaps.length} idle gap{gaps.length === 1 ? "" : "s"} &gt; 1m)
        </div>
        <div className="flex flex-wrap gap-2 text-[10px]">
          {rows.map((tf) => {
            const c = colourFor(tf);
            return (
              <span
                key={tf}
                className="inline-flex items-center gap-1 font-mono"
                data-testid={`training-progress-timeline-legend-${tf}`}
              >
                <span
                  className="inline-block w-3 h-3 rounded-sm"
                  style={{ backgroundColor: c.fill, border: `1px solid ${c.stroke}` }}
                />
                {c.label}
              </span>
            );
          })}
          <span className="inline-flex items-center gap-1 font-mono text-amber-300/90">
            <span
              className="inline-block w-3 h-3 rounded-sm border border-amber-400"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(45deg, rgba(251,191,36,0.35) 0 3px, transparent 3px 6px)",
              }}
            />
            idle &gt; 1m
          </span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height + 18}`}
          width="100%"
          preserveAspectRatio="none"
          style={{ minWidth: 480, height: height + 18 }}
          role="img"
          aria-label="Training campaign per-slice timeline"
        >
          {/* row backgrounds + labels */}
          {rows.map((tf) => {
            const y = yFor(tf);
            return (
              <g key={`row-${tf}`}>
                <rect
                  x={leftPad}
                  y={y}
                  width={innerWidth}
                  height={rowHeight}
                  fill="rgba(255,255,255,0.02)"
                  stroke="rgba(255,255,255,0.06)"
                />
                <text
                  x={leftPad - 6}
                  y={y + rowHeight / 2 + 3}
                  textAnchor="end"
                  fontSize={10}
                  fontFamily="ui-monospace, SFMono-Regular, monospace"
                  fill="rgba(255,255,255,0.65)"
                >
                  {colourFor(tf).label}
                </text>
              </g>
            );
          })}
          {/* idle-gap shaded bands across the full chart height */}
          {gaps.map((g, i) => {
            const x1 = xFor(g.start);
            const x2 = xFor(g.end);
            const w = Math.max(2, x2 - x1);
            return (
              <rect
                key={`gap-${i}`}
                x={x1}
                y={topPad}
                width={w}
                height={rows.length * (rowHeight + rowGap) - rowGap}
                fill="rgba(251,191,36,0.18)"
                stroke="rgba(251,191,36,0.5)"
                strokeDasharray="3 3"
                data-testid={`training-progress-timeline-gap-${i}`}
              >
                <title>
                  Idle {formatDuration(g.durationSec)} between coins
                </title>
              </rect>
            );
          })}
          {/* slice bars */}
          {parsed.map((p, i) => {
            const c = colourFor(p.timeframe);
            const y = yFor(p.timeframe ?? "?");
            const x1 = xFor(p.start);
            const x2 = xFor(p.end);
            const w = Math.max(2, x2 - x1);
            const isRunning = p.status === "running";
            const isSkipped = p.status === "skipped";
            return (
              <rect
                key={`slc-${i}`}
                x={x1}
                y={y + 3}
                width={w}
                height={rowHeight - 6}
                rx={2}
                ry={2}
                fill={isSkipped ? "rgba(120,120,120,0.35)" : c.fill}
                stroke={isRunning ? "#fff" : c.stroke}
                strokeWidth={isRunning ? 1.5 : 1}
                strokeDasharray={isRunning ? "3 2" : undefined}
                data-testid={`training-progress-timeline-slice-${i}`}
              >
                <title>
                  {p.coin}/{p.timeframe ?? "?"} — {p.status}
                  {"\n"}
                  {new Date(p.start).toLocaleTimeString()} →{" "}
                  {new Date(p.end).toLocaleTimeString()}
                  {"\n"}
                  elapsed {formatDuration((p.end - p.start) / 1000)}
                </title>
              </rect>
            );
          })}
          {/* x-axis ticks */}
          {ticks.map((t, i) => (
            <g key={`tick-${i}`}>
              <line
                x1={t.x}
                x2={t.x}
                y1={topPad}
                y2={height}
                stroke="rgba(255,255,255,0.12)"
                strokeDasharray="2 3"
              />
              <text
                x={t.x}
                y={height + 12}
                textAnchor={i === 0 ? "start" : i === ticks.length - 1 ? "end" : "middle"}
                fontSize={9}
                fontFamily="ui-monospace, SFMono-Regular, monospace"
                fill="rgba(255,255,255,0.55)"
              >
                {t.label}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span data-testid="training-progress-timeline-window">
          Window: {formatDuration((maxT - minT) / 1000)}
        </span>
        <span data-testid="training-progress-timeline-fitting-total">
          Fitting time: {formatDuration(totalSliceMs / 1000)}
        </span>
        <span data-testid="training-progress-timeline-idle-total">
          Idle &gt; 1m: {formatDuration(idleMs / 1000)}
        </span>
      </div>
    </div>
  );
}
