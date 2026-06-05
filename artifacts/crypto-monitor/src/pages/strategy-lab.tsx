import { useStrategyLab, type StrategyBucket, type SettingsHistoryEvent } from "@/hooks/use-strategy-lab";
import { useStrategyLabSettings, useUpdateStrategyLabSettings, type DcaSettings } from "@/hooks/use-strategy-lab-settings";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Beaker, ShieldCheck, ShieldAlert, TrendingUp, TrendingDown, Activity, Settings2, RotateCcw, Save, Clock } from "lucide-react";
import { ComposedChart, Line, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, ReferenceLine } from "recharts";
import type { TooltipProps } from "recharts";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useAdminKey } from "@/hooks/use-admin-key";
import { AdminKeyField } from "@/components/admin-key-field";

interface MeasurementModeState {
  enabled: boolean;
  source: "default" | "env" | "manual";
  lastChangedAt: string;
}

// Reads measurement mode state from the backend so the banner reflects
// reality instead of always claiming the system is in measurement mode.
// Returns null while loading or on error (banner is then hidden).
function useMeasurementMode(): {
  state: MeasurementModeState | null;
  loaded: boolean;
  refresh: () => void;
  setState: (s: MeasurementModeState) => void;
} {
  const [state, setState] = useState<MeasurementModeState | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let cancelled = false;
    fetch(`${import.meta.env.BASE_URL}api/crypto/measurement-mode`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: MeasurementModeState) => {
        if (cancelled) return;
        setState({
          enabled: !!d.enabled,
          source: d.source ?? "default",
          lastChangedAt: d.lastChangedAt ?? new Date(0).toISOString(),
        });
        setLoaded(true);
      })
      .catch(() => { if (!cancelled) setLoaded(true); });
    return () => { cancelled = true; };
  }, [tick]);
  const refresh = useCallback(() => setTick((t) => t + 1), []);
  return { state, loaded, refresh, setState };
}

function MeasurementModeToggle({
  state,
  onUpdated,
}: {
  state: MeasurementModeState | null;
  onUpdated: (s: MeasurementModeState) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showKeyField, setShowKeyField] = useState(false);
  const admin = useAdminKey({
    onRejected: () => setErr("Admin key was rejected — paste it again below."),
  });

  if (!state) return null;
  const enabled = state.enabled;
  const targetLabel = enabled ? "Turn measurement mode OFF" : "Turn measurement mode ON";
  const confirmMsg = enabled
    ? "Turn measurement mode OFF? Auto-deploy of idle cash will resume on the next monitor cycle."
    : "Turn measurement mode ON? This stops auto-deploy of idle cash so the consensus engine can be observed in isolation.";

  async function flip() {
    if (!window.confirm(confirmMsg)) return;
    if (!admin.hasKey) {
      setShowKeyField(true);
      setErr("Admin key required to flip measurement mode.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const res = await admin.adminFetch(
        `${import.meta.env.BASE_URL}api/crypto/measurement-mode/toggle`,
        {
          action: targetLabel,
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: !enabled }),
        },
      );
      if (!res) {
        setShowKeyField(true);
        throw new Error("Admin key was rejected or missing.");
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error((data as { error?: string }).error || `HTTP ${res.status}`);
      onUpdated({
        enabled: !!(data as MeasurementModeState).enabled,
        source: (data as MeasurementModeState).source ?? "manual",
        lastChangedAt: (data as MeasurementModeState).lastChangedAt ?? new Date().toISOString(),
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Toggle failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-2" data-testid="measurement-mode-toggle">
      <div className="flex items-center gap-2 flex-wrap">
        <Badge
          variant="outline"
          className={cn(
            "text-[10px] uppercase tracking-wider",
            enabled
              ? "border-cyan-500/40 text-cyan-300"
              : "border-white/20 text-muted-foreground",
          )}
          data-testid="measurement-mode-status"
        >
          {enabled ? "On" : "Off"}
          {state.source === "env" && " (env)"}
          {state.source === "manual" && " (manual)"}
        </Badge>
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={flip}
          className={cn(
            enabled
              ? "border-amber-500/40 text-amber-300 hover:bg-amber-500/10 hover:text-amber-200"
              : "border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/10 hover:text-cyan-200",
          )}
          data-testid="button-toggle-measurement-mode"
        >
          {busy ? "Saving…" : targetLabel}
        </Button>
      </div>
      {err && (
        <div className="text-xs text-rose-400" data-testid="measurement-mode-error">
          {err}
        </div>
      )}
      {showKeyField && !admin.hasKey && (
        <AdminKeyField
          admin={admin}
          autoFocus
          helpText="Paste ADMIN_API_KEY to flip measurement mode. The key stays in this tab only."
          testIdPrefix="measurement-mode-admin-key"
        />
      )}
    </div>
  );
}

type SeriesKey = "ai-bots" | "dca-cb" | "buy-hold" | "trend-filter";

interface ChartRow {
  timestamp: string;
  t: number;
  "ai-bots"?: number;
  "dca-cb"?: number;
  "buy-hold"?: number;
  "trend-filter"?: number;
  settingChange?: SettingsHistoryEvent;
  dcaSettingMarker?: number;
}

const BUCKET_COLORS: Record<string, string> = {
  "ai-bots": "#8b5cf6",
  "dca-cb": "#06b6d4",
  "buy-hold": "#10b981",
  "trend-filter": "#f59e0b",
};

const BUCKET_DESCRIPTIONS: Record<string, string> = {
  "ai-bots": "Aggregate of all live quant agents — LightGBM-driven decisions, paper trading on every monitor cycle.",
  "dca-cb": "Daily equal-slice DCA across the basket. Liquidates fully on 20% drawdown from peak; resumes when 14-day basket return is positive.",
  "buy-hold": "One-time equal-weight buy across the monitored basket. No further actions. The benchmark to beat.",
  "trend-filter": "Equal-weight basket buy when the 30-day basket return is positive; full exit to cash when it turns negative.",
};

function fmtUsd(n: number, opts: { showSign?: boolean } = {}) {
  const sign = opts.showSign && n > 0 ? "+" : n < 0 ? "-" : "";
  const abs = Math.abs(n);
  const str = abs >= 1000 ? abs.toFixed(0) : abs.toFixed(2);
  return `${sign}$${str}`;
}

function fmtPct(n: number, opts: { showSign?: boolean } = {}) {
  const sign = opts.showSign && n > 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function BucketCard({ bucket, leader, drawdownWarnThreshold }: { bucket: StrategyBucket; leader: boolean; drawdownWarnThreshold?: number }) {
  const positive = bucket.totalPnl >= 0;
  const color = BUCKET_COLORS[bucket.strategyType];
  return (
    <Card className={cn(
      "border-border/50 bg-card/50 backdrop-blur ring-1 transition",
      leader ? "ring-emerald-500/40 shadow-lg shadow-emerald-500/10" : "ring-white/5",
    )} data-testid={`bucket-${bucket.strategyType}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
              <CardTitle className="text-base font-display tracking-tight">{bucket.label}</CardTitle>
              {leader && <Badge variant="outline" className="text-[10px] border-emerald-500/40 text-emerald-400">LEADING</Badge>}
              {bucket.circuitBreakerActive && (
                <Badge variant="outline" className="text-[10px] border-amber-500/40 text-amber-400">
                  <ShieldAlert className="w-3 h-3 mr-1" /> Breaker tripped
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1.5 max-w-xl">{BUCKET_DESCRIPTIONS[bucket.strategyType]}</p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-display font-bold tabular-nums">{fmtUsd(bucket.currentEquity)}</div>
            <div className={cn("text-sm font-mono tabular-nums", positive ? "text-emerald-400" : "text-rose-400")}>
              {fmtUsd(bucket.totalPnl, { showSign: true })} ({fmtPct(bucket.totalPnlPct, { showSign: true })})
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {bucket.latestDecision && (
          <div
            className={cn(
              "mb-3 rounded-lg border p-2.5",
              bucket.latestDecision.tone === "good" && "border-emerald-500/30 bg-emerald-500/5",
              bucket.latestDecision.tone === "warn" && "border-amber-500/30 bg-amber-500/5",
              bucket.latestDecision.tone === "neutral" && "border-white/[0.08] bg-white/[0.03]",
            )}
            data-testid={`decision-${bucket.strategyType}`}
          >
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-semibold text-muted-foreground">
              <Clock className="w-3 h-3" /> Latest decision
            </div>
            <div
              className={cn(
                "text-sm font-medium mt-1",
                bucket.latestDecision.tone === "good" && "text-emerald-300",
                bucket.latestDecision.tone === "warn" && "text-amber-300",
                bucket.latestDecision.tone === "neutral" && "text-foreground",
              )}
            >
              {bucket.latestDecision.headline}
            </div>
            {bucket.latestDecision.detail && (
              <div className="text-[11px] text-muted-foreground mt-0.5">{bucket.latestDecision.detail}</div>
            )}
          </div>
        )}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
          <Stat label="Capital" value={fmtUsd(bucket.startingCapital)} />
          <Stat label="Cash" value={fmtUsd(bucket.cash)} />
          <Stat label="Invested" value={fmtUsd(bucket.invested)} />
          <Stat label="Trades" value={bucket.totalTrades.toString()} />
          <Stat label="Fees" value={fmtUsd(bucket.totalFees)} />
          <Stat label="Peak equity" value={fmtUsd(bucket.peakValue)} />
          <Stat
            label="Max drawdown"
            value={fmtPct(bucket.maxDrawdownPct)}
            tone={
              drawdownWarnThreshold !== undefined && bucket.maxDrawdownPct >= drawdownWarnThreshold
                ? "warn"
                : undefined
            }
          />
          <Stat label="Agents" value={bucket.agentCount.toString()} />
        </div>
      </CardContent>
    </Card>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "warn" | "good" }) {
  const toneClass = tone === "warn" ? "text-amber-400" : tone === "good" ? "text-emerald-400" : "text-foreground";
  return (
    <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-2.5">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">{label}</div>
      <div className={cn("text-sm font-mono tabular-nums mt-0.5", toneClass)}>{value}</div>
    </div>
  );
}

export default function StrategyLab() {
  const { data, isLoading } = useStrategyLab();
  const { data: settingsData } = useStrategyLabSettings();
  const measurementMode = useMeasurementMode();
  const drawdownTriggerPct = settingsData?.settings.drawdownTriggerPct;
  const dcaDrawdownWarnThreshold =
    drawdownTriggerPct !== undefined ? Math.max(0, drawdownTriggerPct - 2) : undefined;

  const dcaSettingsHistory = data?.settingsHistory?.["dca-cb"] ?? [];

  const chartData = useMemo<ChartRow[]>(() => {
    if (!data) return [];
    const allTimestamps = new Set<string>();
    for (const series of Object.values(data.equityCurves)) {
      for (const pt of series) allTimestamps.add(pt.timestamp);
    }
    const eventByTs = new Map<string, SettingsHistoryEvent>();
    for (const ev of dcaSettingsHistory) {
      allTimestamps.add(ev.timestamp);
      eventByTs.set(ev.timestamp, ev);
    }
    const sorted = Array.from(allTimestamps).sort();
    const seriesMaps: Record<SeriesKey, Map<string, number>> = {
      "ai-bots": new Map(),
      "dca-cb": new Map(),
      "buy-hold": new Map(),
      "trend-filter": new Map(),
    };
    for (const [k, v] of Object.entries(data.equityCurves)) {
      if (k in seriesMaps) {
        seriesMaps[k as SeriesKey] = new Map(v.map(pt => [pt.timestamp, pt.equity]));
      }
    }
    const lastVals: Record<SeriesKey, number | null> = {
      "ai-bots": null,
      "dca-cb": null,
      "buy-hold": null,
      "trend-filter": null,
    };
    return sorted.map(ts => {
      const row: ChartRow = { timestamp: ts, t: new Date(ts).getTime() };
      (Object.keys(seriesMaps) as SeriesKey[]).forEach(k => {
        const v = seriesMaps[k].get(ts);
        if (v !== undefined) lastVals[k] = v;
        const last = lastVals[k];
        if (last !== null) row[k] = last;
      });
      const ev = eventByTs.get(ts);
      if (ev) {
        row.settingChange = ev;
        if (lastVals["dca-cb"] !== null) {
          row.dcaSettingMarker = lastVals["dca-cb"];
        }
      }
      return row;
    });
  }, [data, dcaSettingsHistory]);

  if (isLoading || !data) {
    return (
      <div className="space-y-6" data-testid="strategy-lab-loading">
        <Skeleton className="h-12 w-72" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  const sortedByPnl = [...data.buckets].sort((a, b) => b.totalPnlPct - a.totalPnlPct);
  const leader = sortedByPnl[0];
  const runnerUp = sortedByPnl[1];
  const lastPlace = sortedByPnl[sortedByPnl.length - 1];
  const leaderSpreadVsRunnerUp = leader && runnerUp ? leader.totalPnlPct - runnerUp.totalPnlPct : null;
  const leaderSpreadVsLast = leader && lastPlace && leader !== lastPlace ? leader.totalPnlPct - lastPlace.totalPnlPct : null;

  return (
    <div className="space-y-6" data-testid="strategy-lab-page">
      {measurementMode.loaded && measurementMode.state && (
        <div
          className={cn(
            "rounded-lg border p-3 text-xs flex items-start gap-3",
            measurementMode.state.enabled
              ? "border-cyan-500/30 bg-cyan-500/5 text-cyan-100/90"
              : "border-white/[0.08] bg-white/[0.03] text-muted-foreground",
          )}
          data-testid="measurement-mode-banner"
        >
          <ShieldCheck
            className={cn(
              "w-4 h-4 mt-0.5 flex-shrink-0",
              measurementMode.state.enabled ? "text-cyan-300" : "text-muted-foreground",
            )}
          />
          <div className="flex-1 min-w-0">
            {measurementMode.state.enabled ? (
              <>
                <span className="font-semibold text-cyan-200">Measurement mode active.</span>{" "}
                Auto-deploy of idle cash is off and agent evolution is frozen. Bots only trade when the consensus engine emits a real signal.
                This keeps the 30–90 day comparison honest — we want to know which strategy truly wins before any real money is deployed.
              </>
            ) : (
              <>
                <span className="font-semibold text-foreground">Measurement mode is off.</span>{" "}
                Agent evolution and auto-deploy of idle cash are running normally. Flip it on to freeze the system for an isolated measurement window.
              </>
            )}
          </div>
          <div className="shrink-0">
            <MeasurementModeToggle
              state={measurementMode.state}
              onUpdated={(s) => measurementMode.setState(s)}
            />
          </div>
        </div>
      )}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-display font-bold tracking-tight flex items-center gap-3">
            <Beaker className="w-7 h-7 text-cyan-400" />
            Strategy Lab
          </h1>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-3xl">
            Parallel measurement of the quant fleet against naive baselines. Same prices, same fees, same coins — different decision rules.
            Each portfolio starts at $1,000. All four buckets are deterministic; the only thing that varies is what triggers a buy or sell.
          </p>
        </div>
        {leader && (
          <Card className="border-border/50 ring-1 ring-emerald-500/30 bg-emerald-500/5" data-testid="leader-callout">
            <CardContent className="p-4 min-w-[220px]">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                Current leader
              </div>
              <div className="text-lg font-display font-bold mt-1 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ background: BUCKET_COLORS[leader.strategyType] }} />
                {leader.label}
              </div>
              <div className={cn("text-xl font-display font-bold tabular-nums mt-0.5", leader.totalPnlPct >= 0 ? "text-emerald-400" : "text-rose-400")}>
                {fmtPct(leader.totalPnlPct, { showSign: true })}
              </div>
              {leaderSpreadVsRunnerUp !== null && runnerUp && (
                <div className="text-[11px] text-muted-foreground mt-1">
                  +{leaderSpreadVsRunnerUp.toFixed(2)}% over {runnerUp.label}
                  {leaderSpreadVsLast !== null && lastPlace && lastPlace !== runnerUp && (
                    <> · +{leaderSpreadVsLast.toFixed(2)}% over {lastPlace.label}</>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>

      <DcaSettingsPanel />

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {data.buckets.map(bucket => (
          <BucketCard
            key={bucket.strategyType}
            bucket={bucket}
            leader={leader && bucket.strategyType === leader.strategyType}
            drawdownWarnThreshold={bucket.strategyType === "dca-cb" ? dcaDrawdownWarnThreshold : undefined}
          />
        ))}
      </div>

      <Card className="border-border/50 bg-card/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" /> Equity curves
          </CardTitle>
        </CardHeader>
        <CardContent>
          {chartData.length < 2 ? (
            <div className="h-72 flex items-center justify-center text-sm text-muted-foreground">
              Collecting data — equity curves will appear after a few cycles.
            </div>
          ) : (
            <div className="h-80" data-testid="strategy-lab-chart">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="t"
                    type="number"
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={(ms) => new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                    stroke="rgba(255,255,255,0.4)"
                    fontSize={11}
                  />
                  <YAxis
                    domain={["dataMin - 20", "dataMax + 20"]}
                    tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
                    stroke="rgba(255,255,255,0.4)"
                    fontSize={11}
                  />
                  <Tooltip content={<EquityTooltip />} />
                  <Legend
                    formatter={(v) => labelFor(v)}
                    wrapperStyle={{ fontSize: 12 }}
                    payload={[
                      { value: "ai-bots", type: "line", color: BUCKET_COLORS["ai-bots"] },
                      { value: "dca-cb", type: "line", color: BUCKET_COLORS["dca-cb"] },
                      { value: "buy-hold", type: "line", color: BUCKET_COLORS["buy-hold"] },
                      { value: "trend-filter", type: "line", color: BUCKET_COLORS["trend-filter"] },
                      ...(dcaSettingsHistory.length > 0
                        ? [{ value: "dca-setting-change", type: "line" as const, color: "#fbbf24" }]
                        : []),
                    ]}
                  />
                  {dcaSettingsHistory.map(ev => (
                    <ReferenceLine
                      key={`refline-${ev.id}`}
                      x={new Date(ev.timestamp).getTime()}
                      stroke="#fbbf24"
                      strokeDasharray="4 4"
                      strokeOpacity={0.55}
                      ifOverflow="extendDomain"
                    />
                  ))}
                  <Line type="monotone" dataKey="ai-bots" stroke={BUCKET_COLORS["ai-bots"]} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="dca-cb" stroke={BUCKET_COLORS["dca-cb"]} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="buy-hold" stroke={BUCKET_COLORS["buy-hold"]} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="trend-filter" stroke={BUCKET_COLORS["trend-filter"]} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Scatter
                    dataKey="dcaSettingMarker"
                    fill="#fbbf24"
                    stroke="#0a0a0f"
                    strokeWidth={2}
                    shape="circle"
                    isAnimationActive={false}
                    legendType="none"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}
          {dcaSettingsHistory.length > 0 && (
            <p className="text-[11px] text-muted-foreground/70 mt-3" data-testid="dca-history-hint">
              <span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1.5 align-middle" />
              Amber markers on the DCA curve show where you tuned the knobs. Hover one to see the before/after values.
            </p>
          )}
        </CardContent>
      </Card>

      <div className="text-[11px] text-muted-foreground/70 leading-relaxed max-w-3xl">
        <p>
          <span className="text-foreground/70 font-medium">Honest accounting:</span> All strategies pay the same simulated 0.10% fee + 0.05% slippage per side.
          Equity = cash + mark-to-market of holdings net of estimated exit costs. Snapshots taken every monitor cycle.
          DCA deploys the per-cycle amount on the configured cadence across the basket until cash runs out or the breaker trips.
        </p>
      </div>
    </div>
  );
}

function EquityTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;
  const row = (payload[0]?.payload ?? {}) as ChartRow;
  const change = row.settingChange;
  const seen = new Set<string>();
  const entries: { key: string; value: number }[] = [];
  for (const p of payload) {
    if (!p) continue;
    const key = typeof p.dataKey === "string" ? p.dataKey : "";
    if (!key || key === "dcaSettingMarker") continue;
    if (typeof p.value !== "number") continue;
    if (seen.has(key)) continue;
    seen.add(key);
    entries.push({ key, value: p.value });
  }
  return (
    <div
      className="rounded-md border border-white/10 bg-[rgba(15,15,20,0.95)] p-2.5 text-xs shadow-lg max-w-[260px]"
      data-testid={change ? "dca-setting-tooltip" : undefined}
    >
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
        {new Date(Number(label)).toLocaleString()}
      </div>
      {entries.length > 0 && (
        <div className="mt-1.5 space-y-0.5">
          {entries.map(e => (
            <div key={e.key} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: BUCKET_COLORS[e.key] }} />
                <span className="text-foreground/80">{labelFor(e.key)}</span>
              </span>
              <span className="font-mono tabular-nums">${e.value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
      {change && (
        <div className="mt-2 pt-2 border-t border-white/10">
          <div className="text-[10px] uppercase tracking-wider text-amber-400 font-semibold flex items-center gap-1">
            <Settings2 className="w-3 h-3" /> DCA setting change
          </div>
          <div className="mt-1 space-y-1">
            {change.changes.map(c => (
              <div key={c.field} className="flex items-center justify-between gap-3" data-testid={`tooltip-change-${c.field}`}>
                <span className="text-foreground/80">{c.label}</span>
                <span className="font-mono tabular-nums">
                  <span className="text-muted-foreground line-through">{c.formattedBefore}</span>
                  <span className="mx-1 text-muted-foreground">→</span>
                  <span className="text-amber-300">{c.formattedAfter}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DcaSettingsPanel() {
  const { data, isLoading } = useStrategyLabSettings();
  const update = useUpdateStrategyLabSettings();
  const [draft, setDraft] = useState<DcaSettings | null>(null);

  useEffect(() => {
    if (data?.settings && draft === null) {
      setDraft(data.settings);
    }
  }, [data, draft]);

  if (isLoading || !data || !draft) {
    return (
      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="dca-settings-loading">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-cyan-400" /> DCA + Circuit Breaker tuning
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  const defaults = data.defaults;
  const dirty =
    draft.drawdownTriggerPct !== data.settings.drawdownTriggerPct ||
    draft.resumeLookbackDays !== data.settings.resumeLookbackDays ||
    Math.abs(draft.cycleDeployUsd - data.settings.cycleDeployUsd) > 0.001 ||
    draft.buyIntervalHours !== data.settings.buyIntervalHours;

  const handleReset = () => {
    setDraft({ ...defaults });
  };

  const handleSave = () => {
    update.mutate(draft);
  };

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="dca-settings-panel">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
              <Settings2 className="w-4 h-4 text-cyan-400" /> DCA + Circuit Breaker tuning
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1.5 max-w-2xl">
              Adjust how aggressive the daily DCA strategy is. Changes apply on the next monitor cycle and persist across restarts.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleReset}
              disabled={update.isPending}
              data-testid="dca-settings-reset"
              className="text-xs"
            >
              <RotateCcw className="w-3.5 h-3.5 mr-1.5" /> Defaults
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={!dirty || update.isPending}
              data-testid="dca-settings-save"
              className="text-xs"
            >
              <Save className="w-3.5 h-3.5 mr-1.5" />
              {update.isPending ? "Saving…" : dirty ? "Save changes" : "Saved"}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
          <SettingControl
            label="Drawdown trigger"
            value={`${draft.drawdownTriggerPct.toFixed(0)}%`}
            help={`Liquidate the basket when equity falls this far below peak. Default ${defaults.drawdownTriggerPct.toFixed(0)}%.`}
            testId="dca-setting-drawdown"
          >
            <Slider
              min={5}
              max={50}
              step={1}
              value={[draft.drawdownTriggerPct]}
              onValueChange={([v]) => setDraft({ ...draft, drawdownTriggerPct: v })}
              data-testid="dca-slider-drawdown"
            />
            <RangeHint min="5%" max="50%" />
          </SettingControl>

          <SettingControl
            label="Resume lookback"
            value={`${draft.resumeLookbackDays} day${draft.resumeLookbackDays === 1 ? "" : "s"}`}
            help={`After tripping, only resume buying when the basket's return over this window turns positive. Default ${defaults.resumeLookbackDays} days.`}
            testId="dca-setting-resume"
          >
            <Slider
              min={1}
              max={30}
              step={1}
              value={[draft.resumeLookbackDays]}
              onValueChange={([v]) => setDraft({ ...draft, resumeLookbackDays: v })}
              data-testid="dca-slider-resume"
            />
            <RangeHint min="1d" max="30d" />
          </SettingControl>

          <SettingControl
            label="Per-cycle deploy"
            value={`$${draft.cycleDeployUsd.toFixed(2)}`}
            help={`USD spent each daily cycle, equally split across the basket. Default $${defaults.cycleDeployUsd.toFixed(2)}.`}
            testId="dca-setting-deploy"
          >
            <Slider
              min={5}
              max={200}
              step={1}
              value={[draft.cycleDeployUsd]}
              onValueChange={([v]) => setDraft({ ...draft, cycleDeployUsd: v })}
              data-testid="dca-slider-deploy"
            />
            <RangeHint min="$5" max="$200" />
          </SettingControl>

          <SettingControl
            label="Buy cadence"
            value={formatInterval(draft.buyIntervalHours)}
            help={`How often the DCA strategy deploys a cycle. Default ${formatInterval(defaults.buyIntervalHours)}.`}
            testId="dca-setting-interval"
          >
            <Slider
              min={1}
              max={168}
              step={1}
              value={[draft.buyIntervalHours]}
              onValueChange={([v]) => setDraft({ ...draft, buyIntervalHours: v })}
              data-testid="dca-slider-interval"
            />
            <RangeHint min="1h" max="7d" />
          </SettingControl>
        </div>
        {update.isError && (
          <p className="text-xs text-rose-400 mt-3" data-testid="dca-settings-error">
            Couldn't save: {update.error.message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function SettingControl({ label, value, help, testId, children }: { label: string; value: string; help: string; testId: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-3" data-testid={testId}>
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">{label}</div>
        <div className="text-base font-mono tabular-nums text-foreground">{value}</div>
      </div>
      {children}
      <p className="text-[11px] text-muted-foreground mt-2 leading-snug">{help}</p>
    </div>
  );
}

function RangeHint({ min, max }: { min: string; max: string }) {
  return (
    <div className="flex justify-between text-[10px] font-mono text-muted-foreground/60 mt-1.5">
      <span>{min}</span>
      <span>{max}</span>
    </div>
  );
}

function formatInterval(hours: number): string {
  if (hours < 24 || hours % 24 !== 0) return `${hours}h`;
  const days = hours / 24;
  return `${days} day${days === 1 ? "" : "s"}`;
}

function labelFor(key: string): string {
  if (key === "ai-bots") return "Quant Fleet";
  if (key === "dca-cb") return "DCA + Breaker";
  if (key === "buy-hold") return "Buy & Hold";
  if (key === "trend-filter") return "Trend Filter";
  return key;
}
