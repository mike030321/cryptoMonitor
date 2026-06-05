import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, RefreshCw, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface WindowRow {
  label: string;
  count: number;
  pct: number;
  baselineCount: number;
  baselinePct: number;
  driftPct: number;
  tradeCount: number;
}

interface WindowBlock {
  key: string;
  label: string;
  hours: number;
  totals: {
    candles: number;
    trades: number;
    unlabeledPct: number;
  };
  distribution: WindowRow[];
}

interface BaselineBlock {
  days: number;
  totals: { candles: number; unlabeledPct: number };
  distribution: { label: string; count: number; pct: number }[];
}

interface RegimeDistribution {
  windows: WindowBlock[];
  baseline: BaselineBlock;
  labels: string[];
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

const LABEL_TONES: Record<string, string> = {
  trending_up: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  trending_down: "bg-red-500/15 text-red-300 border-red-500/40",
  range_chop: "bg-slate-500/15 text-slate-300 border-slate-500/40",
  high_vol_breakout: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  low_vol_compression: "bg-sky-500/15 text-sky-300 border-sky-500/40",
  panic_liquidation: "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/40",
};

const LABEL_FILL: Record<string, string> = {
  trending_up: "bg-emerald-500",
  trending_down: "bg-red-500",
  range_chop: "bg-slate-500",
  high_vol_breakout: "bg-amber-500",
  low_vol_compression: "bg-sky-500",
  panic_liquidation: "bg-fuchsia-500",
};

function prettyLabel(s: string): string {
  return s.replaceAll("_", " ");
}

function formatDrift(pct: number): string {
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}pp`;
}

function driftTone(row: WindowRow, totalCandles: number): string {
  // Only flag drift when there's enough sample to be meaningful.
  if (totalCandles < 30) return "text-muted-foreground";
  const abs = Math.abs(row.driftPct);
  // Going silent (label firing in baseline but not in window) is the
  // most operationally interesting case the task calls out.
  if (row.baselinePct >= 1 && row.pct === 0) return "text-red-400";
  if (abs >= 10) return "text-amber-300";
  if (abs >= 5) return "text-sky-300";
  return "text-muted-foreground";
}

export function RegimeDistributionCard() {
  const [data, setData] = useState<RegimeDistribution | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeWindow, setActiveWindow] = useState<string>("24h");

  async function refresh() {
    setLoading(true);
    try {
      const res = await fetch(apiUrl("/crypto/diagnostics/regime"));
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

  const active = useMemo<WindowBlock | null>(() => {
    if (!data) return null;
    return (
      data.windows.find((w) => w.key === activeWindow) ??
      data.windows[0] ??
      null
    );
  }, [data, activeWindow]);

  return (
    <Card data-testid="regime-distribution-card">
      <CardHeader className="flex flex-row items-start justify-between gap-3 pb-3">
        <div className="space-y-1">
          <CardTitle className="flex items-center gap-2 text-lg font-display">
            <Activity className="h-5 w-5" />
            Regime Distribution (Phase 2)
          </CardTitle>
          <p className="text-xs text-muted-foreground max-w-md">
            Rolling regime mix over the last hour, 24h, and 7d, each
            compared to the 30-day baseline so you can spot a label going
            silent (e.g. <span className="font-mono">panic_liquidation</span>{" "}
            after a threshold tweak). Per-window trade counts come from the
            trade journal.
          </p>
        </div>
        <Button
          size="icon"
          variant="ghost"
          className="h-7 w-7"
          onClick={() => void refresh()}
          disabled={loading}
          aria-label="Refresh"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {err && (
          <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-400">
            <AlertTriangle className="h-3.5 w-3.5" />
            {err}
          </div>
        )}

        {data && active && (
          <>
            <div
              className="flex flex-wrap gap-1"
              role="tablist"
              aria-label="Rolling window"
            >
              {data.windows.map((w) => (
                <button
                  key={w.key}
                  role="tab"
                  aria-selected={w.key === active.key}
                  onClick={() => setActiveWindow(w.key)}
                  data-testid={`regime-window-${w.key}`}
                  className={cn(
                    "rounded-md border px-2.5 py-1 text-xs transition-colors",
                    w.key === active.key
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border text-muted-foreground hover:text-foreground",
                  )}
                >
                  {w.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-3 gap-3 text-xs">
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">Window candles</div>
                <div className="text-lg font-semibold">
                  {active.totals.candles.toLocaleString()}
                </div>
              </div>
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">30d baseline</div>
                <div className="text-lg font-semibold">
                  {data.baseline.totals.candles.toLocaleString()}
                </div>
              </div>
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">Window trades</div>
                <div className="text-lg font-semibold">
                  {active.totals.trades.toLocaleString()}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {active.distribution.map((row) => {
                const tone = LABEL_TONES[row.label] ?? "";
                const fill = LABEL_FILL[row.label] ?? "bg-slate-500";
                const drift = driftTone(row, active.totals.candles);
                const silenced =
                  row.baselinePct >= 1 &&
                  row.pct === 0 &&
                  active.totals.candles >= 30;
                return (
                  <div
                    key={row.label}
                    className="space-y-1"
                    data-testid={`regime-row-${row.label}`}
                  >
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-medium",
                          tone,
                        )}
                      >
                        {prettyLabel(row.label)}
                        {silenced && (
                          <AlertTriangle
                            className="h-3 w-3"
                            aria-label="silent vs baseline"
                          />
                        )}
                      </span>
                      <span className="text-muted-foreground">
                        <span
                          className="text-foreground font-medium"
                          data-testid={`regime-count-${row.label}`}
                        >
                          {row.count.toLocaleString()} candles
                        </span>{" "}
                        · {row.pct.toFixed(1)}% · baseline{" "}
                        {row.baselinePct.toFixed(1)}% ·{" "}
                        <span className={cn("font-medium", drift)}>
                          {formatDrift(row.driftPct)}
                        </span>{" "}
                        ·{" "}
                        <span className="text-foreground font-medium">
                          {row.tradeCount} trades
                        </span>
                      </span>
                    </div>
                    <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted/30">
                      <div
                        className={cn("absolute inset-y-0 left-0", fill)}
                        style={{
                          width: `${Math.min(100, row.pct).toFixed(1)}%`,
                        }}
                      />
                      <div
                        className="absolute inset-y-0 w-px bg-foreground/40"
                        style={{
                          left: `${Math.min(100, row.baselinePct).toFixed(1)}%`,
                        }}
                        title="30-day baseline marker"
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {(active.totals.unlabeledPct > 1 ||
              data.baseline.totals.unlabeledPct > 1) && (
              <div className="text-xs text-muted-foreground">
                Unlabeled candles: window {active.totals.unlabeledPct.toFixed(1)}
                %, baseline {data.baseline.totals.unlabeledPct.toFixed(1)}%
                {" — "}rows ingested before the regime classifier shipped or
                while the ml-engine was unreachable.
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
