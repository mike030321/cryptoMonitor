import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrendingUp, AlertTriangle, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { PriorFallbackBadge } from "@/components/prior-fallback-badge";

interface BrainRow {
  brain: string;
  timeframe: string;
  total: number;
  resolved: number;
  correct: number;
  accuracyPct: number | null;
  stableCalls: number;
  upCalls: number;
  downCalls: number;
  stableSharePct: number | null;
  directionalResolved: number;
  directionalCorrect: number;
  directionalAccuracyPct: number | null;
  // Realised market direction in the same window/slice — counted from
  // actual_price vs price_at_prediction on resolved rows. Lets operators
  // compare the brain's call ratio against what the market actually did
  // (Task #172).
  realizedUp?: number;
  realizedDown?: number;
  realizedFlat?: number;
  // True when the call ratio diverges from the realised bar ratio by
  // more than 2× — a likely cause of a weak-signal verdict.
  directionalBias?: boolean;
  // Server-driven weak-signal verdict (preferred); client-side rule
  // is used as a fallback when the field is absent.
  weakSignal?: boolean;
  weakSignalReason?: string | null;
}

interface TrendPoint {
  bucket: string;
  resolved: number;
  correct: number;
  accuracyPct: number | null;
  directional: number;
  directionalCorrect: number;
  directionalAccuracyPct: number | null;
}

interface AccuracyResp {
  windowHours: number;
  byBrain: BrainRow[];
  quantHourlyTrend: TrendPoint[];
  priorFallback?: {
    included: boolean;
    total: number;
    resolved: number;
  };
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

// Canonical chronological order of supported timeframes. Anything returned by
// the API that isn't in this list still renders, but appears after the known
// timeframes (preserving the audit's "new TFs auto-appear" requirement).
const TF_CHRONOLOGICAL_ORDER = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"];

function tfRank(tf: string) {
  const i = TF_CHRONOLOGICAL_ORDER.indexOf(tf);
  return i === -1 ? TF_CHRONOLOGICAL_ORDER.length : i;
}

function tfSort(a: BrainRow, b: BrainRow) {
  const ra = tfRank(a.timeframe);
  const rb = tfRank(b.timeframe);
  if (ra !== rb) return ra - rb;
  return a.timeframe.localeCompare(b.timeframe);
}

export function QuantAccuracyCard() {
  const [data, setData] = useState<AccuracyResp | null>(null);
  const [hours, setHours] = useState(24);
  const [includePrior, setIncludePrior] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh(h = hours, ip = includePrior) {
    setLoading(true);
    try {
      const res = await fetch(
        apiUrl(`/crypto/brain/accuracy?hours=${h}${ip ? "&includePrior=1" : ""}`),
      );
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
    void refresh(hours, includePrior);
    const id = setInterval(() => void refresh(hours, includePrior), 30_000);
    return () => clearInterval(id);
  }, [hours, includePrior]);

  // Post-LLM-removal (Task #255): the LLM is no longer a trade-decision
  // brain. The /crypto/brain/accuracy endpoint may still return historical
  // LLM-tagged rows from before the cutover, but they are not rendered here
  // because this card is the live-quant scorecard.
  const quantRows = (data?.byBrain ?? []).filter((r) => r.brain === "QUANT").sort(tfSort);

  const quantTotals = quantRows.reduce(
    (acc, r) => {
      acc.resolved += r.resolved;
      acc.correct += r.correct;
      acc.dirResolved += r.directionalResolved;
      acc.dirCorrect += r.directionalCorrect;
      acc.stable += r.stableCalls;
      acc.total += r.total;
      return acc;
    },
    { resolved: 0, correct: 0, dirResolved: 0, dirCorrect: 0, stable: 0, total: 0 },
  );
  const headlineAcc =
    quantTotals.resolved > 0
      ? Number(((100 * quantTotals.correct) / quantTotals.resolved).toFixed(1))
      : null;
  const directionalAcc =
    quantTotals.dirResolved > 0
      ? Number(((100 * quantTotals.dirCorrect) / quantTotals.dirResolved).toFixed(1))
      : null;
  const stableShare =
    quantTotals.total > 0
      ? Number(((100 * quantTotals.stable) / quantTotals.total).toFixed(1))
      : null;

  const trend = data?.quantHourlyTrend ?? [];
  const lastTrend = trend.slice(-12);

  return (
    <Card data-testid="quant-accuracy-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Quant Accuracy
          <Badge variant="outline" className="ml-2">last {hours}h</Badge>
        </CardTitle>
        <div className="flex gap-2">
          {[24, 72, 168].map((h) => (
            <Button
              key={h}
              size="sm"
              variant={hours === h ? "default" : "outline"}
              onClick={() => setHours(h)}
              data-testid={`quant-acc-window-${h}`}
            >
              {h === 168 ? "7d" : `${h}h`}
            </Button>
          ))}
          <Button size="sm" variant="ghost" onClick={() => void refresh()} disabled={loading}>
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {err && <div className="text-xs text-red-400">Error: {err}</div>}

        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-xs text-muted-foreground">Headline accuracy</div>
            <div className="text-2xl font-semibold" data-testid="quant-headline-acc">
              {headlineAcc != null ? `${headlineAcc}%` : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground">
              {quantTotals.correct}/{quantTotals.resolved} resolved
            </div>
          </div>
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-xs text-muted-foreground flex items-center gap-1">
              Directional accuracy
              <Badge variant="outline" className="ml-1 text-[10px]">excludes stable</Badge>
            </div>
            <div className="text-2xl font-semibold" data-testid="quant-directional-acc">
              {directionalAcc != null ? `${directionalAcc}%` : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground">
              {quantTotals.dirCorrect}/{quantTotals.dirResolved} non-stable resolved
            </div>
          </div>
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-xs text-muted-foreground flex items-center gap-1">
              Stable-call share
              {stableShare != null && stableShare >= 90 && (
                <AlertTriangle className="h-3 w-3 text-amber-400" />
              )}
            </div>
            <div className="text-2xl font-semibold" data-testid="quant-stable-share">
              {stableShare != null ? `${stableShare}%` : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground">
              {quantTotals.stable}/{quantTotals.total} predictions
            </div>
          </div>
        </div>

        {data?.priorFallback && (
          <PriorFallbackBadge
            excludedCount={data.priorFallback.total}
            unit="predictions"
            included={data.priorFallback.included}
            onToggle={() => setIncludePrior((v) => !v)}
            testId="quant-accuracy-prior-fallback"
          />
        )}

        {stableShare != null && stableShare >= 90 && (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-2 text-[11px] text-amber-300 flex gap-2">
            <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <span>
              Quant is calling <b>stable</b> on {stableShare}% of predictions — its
              headline accuracy is structurally inflated because small TF moves are
              usually under the threshold. Watch the <b>directional accuracy</b> column;
              that's the real skill metric. Numbers should improve as the model retrains
              on more accumulated history.
            </span>
          </div>
        )}

        <div>
          <div className="text-xs font-medium mb-1 text-muted-foreground">By timeframe</div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-3">TF</th>
                  <th className="py-1 pr-3 text-right">Resolved</th>
                  <th className="py-1 pr-3 text-right">Acc</th>
                  <th className="py-1 pr-3 text-right">Dir-only</th>
                  <th className="py-1 pr-3 text-right" title="Quant's directional calls: up / down / stable">Calls ↑/↓/=</th>
                  <th className="py-1 pr-3 text-right" title="Realised market bars in the same window: up / down / flat (from actual_price vs price_at_prediction)">Bars ↑/↓/=</th>
                </tr>
              </thead>
              <tbody>
                {quantRows.map((r) => (
                  <tr key={`${r.brain}-${r.timeframe}`} className="border-t border-border/30">
                    <td className="py-1 pr-3 font-mono">
                      <span className="inline-flex items-center gap-1">
                        {r.timeframe}
                        {/* Audit (Task #165): the 1h / 2h / 6h / 1d horizons
                            measured below coin-flip directional accuracy on
                            live data (41% / 37% / 14% / 34%). Surface a
                            "weak signal" warning whenever a horizon falls
                            below 45% directional with at least 50 resolved
                            calls, so operators can see which horizons are
                            unreliable instead of trusting the headline
                            accuracy column. Investigation of the underlying
                            labelling / threshold / baseline cause is tracked
                            separately. */}
                        {(r.weakSignal === true ||
                          (r.weakSignal === undefined &&
                            r.directionalAccuracyPct != null &&
                            r.directionalResolved >= 50 &&
                            r.directionalAccuracyPct < 45)) && (
                          <span
                            title={
                              // Server-provided reason already appends the
                              // "directional bias" diagnosis when call ratio
                              // diverges from realised bar ratio by >2×
                              // (Task #172).
                              r.weakSignalReason ??
                              `Directional accuracy ${r.directionalAccuracyPct}% on n=${r.directionalResolved} is below coin-flip — treat predictions on this horizon as low-confidence.${
                                r.directionalBias
                                  ? ` Likely cause: directional bias — brain called ${r.upCalls}↑/${r.downCalls}↓ while the market printed ${r.realizedUp ?? 0}↑/${r.realizedDown ?? 0}↓.`
                                  : ""
                              }`
                            }
                            data-testid={`horizon-weak-${r.brain}-${r.timeframe}`}
                            className="inline-flex items-center gap-0.5 rounded border border-amber-500/40 bg-amber-500/10 px-1 py-0 text-[9px] uppercase tracking-wide text-amber-300"
                          >
                            <AlertTriangle className="h-2.5 w-2.5" /> disabled
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">{r.resolved}</td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {r.accuracyPct != null ? `${r.accuracyPct}%` : "—"}
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {r.directionalAccuracyPct != null
                        ? `${r.directionalAccuracyPct}% (${r.directionalCorrect}/${r.directionalResolved})`
                        : "—"}
                    </td>
                    <td
                      className={cn(
                        "py-1 pr-3 text-right tabular-nums",
                        r.directionalBias ? "text-amber-300 font-medium" : "text-muted-foreground",
                      )}
                      title={
                        r.directionalBias
                          ? "Brain's call ratio diverges from realised bars by more than 2× — directional bias."
                          : undefined
                      }
                      data-testid={`call-ratio-${r.brain}-${r.timeframe}`}
                    >
                      {r.upCalls} / {r.downCalls} / {r.stableCalls}
                    </td>
                    <td
                      className="py-1 pr-3 text-right tabular-nums text-muted-foreground"
                      data-testid={`bar-ratio-${r.brain}-${r.timeframe}`}
                    >
                      {(r.realizedUp ?? 0) + (r.realizedDown ?? 0) + (r.realizedFlat ?? 0) > 0
                        ? `${r.realizedUp ?? 0} / ${r.realizedDown ?? 0} / ${r.realizedFlat ?? 0}`
                        : "—"}
                    </td>
                  </tr>
                ))}
                {quantRows.length === 0 && (
                  <tr>
                    <td colSpan={6} className="py-2 text-center text-muted-foreground">
                      No quant predictions resolved in this window yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <div className="text-xs font-medium mb-1 text-muted-foreground">
            Quant trend — last {lastTrend.length} hourly buckets
          </div>
          <div className="flex items-end gap-0.5 h-16">
            {lastTrend.length === 0 ? (
              <div className="text-[11px] text-muted-foreground self-center">
                No resolved quant predictions in this window yet.
              </div>
            ) : (
              lastTrend.map((p) => {
                const acc = p.directionalAccuracyPct ?? p.accuracyPct ?? 0;
                const isDir = p.directionalAccuracyPct != null;
                return (
                  <div
                    key={p.bucket}
                    title={`${new Date(p.bucket).toLocaleString()}\nresolved=${p.resolved} correct=${p.correct}\ndirectional=${p.directional} dir-correct=${p.directionalCorrect}`}
                    className={cn(
                      "flex-1 rounded-t",
                      isDir ? "bg-emerald-500/70" : "bg-emerald-500/25",
                    )}
                    style={{ height: `${Math.max(4, acc)}%` }}
                  />
                );
              })
            )}
          </div>
          <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
            <span>0%</span>
            <span>solid = directional buckets · faint = stable-only</span>
            <span>100%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

