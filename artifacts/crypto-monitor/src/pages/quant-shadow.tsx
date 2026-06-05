import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

interface ShadowMetric {
  timeframe: string;
  totalResolved: number;
  modelDirectionalAccuracy: number;
  brierScore: number;
  meanConfidence: number;
}
interface MetricsResp {
  metrics: ShadowMetric[];
  priorFallback?: { included: boolean };
  // Task #532 / C-8 — staleness window: when ageMinutes >
  // staleAfterMinutes, the calibration table is showing stale numbers
  // and we render an honest banner above the rows instead of letting
  // operators read 19.5 % directional accuracy as if it were current.
  freshness?: {
    lastSampleAt: string | null;
    ageMinutes: number | null;
    staleAfterMinutes: number;
    isStale: boolean;
  };
}
interface VsBacktestRow {
  coinId: string;
  baseline: { winRate: number; expectancyUsd: number; nTrades: number } | null;
  liveResolved: number;
  liveWinRate: number;
  liveMinusBaseline: number;
  withinBand: boolean | null;
  status: "tracking" | "drift_high" | "drift_low" | "insufficient_data" | "no_baseline";
}
interface VsBacktestResp { rows: VsBacktestRow[]; band: number; minSamples: number }

const base = import.meta.env.BASE_URL;
async function fetchJson<T>(p: string): Promise<T> {
  const r = await fetch(`${base}api${p}`);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function QuantShadow() {
  const metricsQ = useQuery({ queryKey: ["shadow-metrics"], queryFn: () => fetchJson<MetricsResp>("/crypto/shadow/metrics"), refetchInterval: 30000 });
  const vsBtQ = useQuery({ queryKey: ["shadow-vs-backtest"], queryFn: () => fetchJson<VsBacktestResp>("/crypto/shadow/vs-backtest"), refetchInterval: 60000 });

  if (metricsQ.isLoading) {
    return <div className="space-y-4"><Skeleton className="h-10 w-64" /><Skeleton className="h-64" /></div>;
  }

  const metrics = metricsQ.data?.metrics ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Quant Live Health</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Calibration and live-vs-backtest tracking for the quant brain that authorises every trade.
          All numbers below come from resolved predictions written to <span className="font-mono">model_predictions</span>.
        </p>
        {metricsQ.data?.priorFallback?.included && (
          <Badge variant="outline" className="mt-2">includes prior-fallback rows</Badge>
        )}
      </div>

      {/* Task #532 / C-8 — staleness banner. The audit found
          model_predictions.MAX(created_at) was 3d stale because the
          brain has been off, but this page rendered the rows as if
          they were current. When `freshness.isStale` we render a red
          banner with the exact age + last-sample timestamp. */}
      {metricsQ.data?.freshness && (
        (() => {
          const f = metricsQ.data!.freshness!;
          const ageStr = f.ageMinutes === null
            ? "never"
            : f.ageMinutes < 60
              ? `${f.ageMinutes} min ago`
              : f.ageMinutes < 60 * 24
                ? `${(f.ageMinutes / 60).toFixed(1)} h ago`
                : `${(f.ageMinutes / (60 * 24)).toFixed(1)} d ago`;
          if (f.isStale) {
            return (
              <Card
                className="border-red-500/40 bg-red-500/10"
                data-testid="shadow-staleness-banner"
              >
                <CardContent className="py-3 text-sm">
                  <span className="font-semibold text-red-300">Calibration is stale.</span>{" "}
                  Last live sample{" "}
                  <span className="font-mono text-red-200">{ageStr}</span>
                  {f.lastSampleAt && (
                    <>
                      {" "}(<span className="font-mono">{f.lastSampleAt}</span>)
                    </>
                  )}
                  . Threshold for "live" is &le; {f.staleAfterMinutes} min. The
                  numbers below describe the brain's most recent active
                  window — they do not reflect the current cycle.
                </CardContent>
              </Card>
            );
          }
          return (
            <Card className="border-emerald-500/30 bg-emerald-500/5">
              <CardContent className="py-2 text-xs text-emerald-300/90">
                Live calibration sample {ageStr} (threshold: {f.staleAfterMinutes} min).
              </CardContent>
            </Card>
          );
        })()
      )}

      {metrics.length === 0 ? (
        <Card><CardContent className="py-8 text-center text-muted-foreground">
          No resolved quant predictions yet. Rows resolve once each timeframe horizon elapses.
        </CardContent></Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Calibration per timeframe</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground mb-3">
              Directional accuracy uses the same neutral-zone adjudication as the live resolver.
              Brier is the 3-class score against the canonical realized class (lower is better).
              Mean confidence is the model's self-reported max-class probability across the window.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="border-b text-left text-muted-foreground">
                  <th className="py-2">Timeframe</th>
                  <th>Resolved</th>
                  <th>Directional accuracy</th>
                  <th>Brier (lower=better)</th>
                  <th>Mean confidence</th>
                </tr></thead>
                <tbody>
                  {metrics.map(m => (
                    <tr key={m.timeframe} className="border-b">
                      <td className="py-2 font-mono">{m.timeframe}</td>
                      <td>{m.totalResolved}</td>
                      <td>{pct(m.modelDirectionalAccuracy)}</td>
                      <td>{m.brierScore.toFixed(3)}</td>
                      <td>{pct(m.meanConfidence)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {vsBtQ.data && vsBtQ.data.rows.length > 0 && (
        <Card>
          <CardHeader><CardTitle>Live vs Phase 3 backtest tracking</CardTitle></CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground mb-3">
              Live quant win-rate per coin against the offline backtest baseline. "Tracking" means the live brain is staying within ±{(vsBtQ.data.band * 100).toFixed(0)} percentage points of its backtest expectation on at least {vsBtQ.data.minSamples} resolved trades.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="border-b text-left text-muted-foreground">
                  <th className="py-2">Coin</th>
                  <th>Live resolved</th>
                  <th>Live win-rate</th>
                  <th>Backtest baseline</th>
                  <th>Δ</th>
                  <th>Status</th>
                </tr></thead>
                <tbody>
                  {vsBtQ.data.rows.map(r => {
                    const badge = r.status === "tracking"
                      ? <Badge className="bg-emerald-600 text-white">tracking</Badge>
                      : r.status === "drift_high"
                        ? <Badge className="bg-amber-600 text-white">live above backtest</Badge>
                        : r.status === "drift_low"
                          ? <Badge className="bg-red-600 text-white">live below backtest</Badge>
                          : r.status === "insufficient_data"
                            ? <Badge variant="outline">need more samples</Badge>
                            : <Badge variant="outline">no baseline</Badge>;
                    return (
                      <tr key={r.coinId} className="border-b">
                        <td className="py-2">{r.coinId}</td>
                        <td>{r.liveResolved}</td>
                        <td>{pct(r.liveWinRate)}</td>
                        <td>{r.baseline ? `${pct(r.baseline.winRate)} (${r.baseline.nTrades} trades)` : "—"}</td>
                        <td className={r.withinBand === true ? "text-emerald-400" : r.withinBand === false ? "text-red-400" : ""}>
                          {r.baseline ? `${r.liveMinusBaseline >= 0 ? "+" : ""}${pct(r.liveMinusBaseline)}` : "—"}
                        </td>
                        <td>{badge}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
