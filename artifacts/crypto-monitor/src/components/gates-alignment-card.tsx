import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { GitCompareArrows, RefreshCw, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface GatesAlignment {
  n: number;
  source: string;
  min_directional_edge: number;
  min_expected_return_pct: number;
  aligned_share: number;
  aligned_loud_share: number;
  aligned_quiet_share: number;
  loud_classifier_quiet_regressor_share: number;
  quiet_classifier_loud_regressor_share: number;
  dir_edge_p50: number;
  dir_edge_p95: number;
  abs_magnitude_pct_p50: number;
  abs_magnitude_pct_p95: number;
}

interface SliceReport {
  status?: string;
  version?: string;
  gates_alignment?: GatesAlignment | null;
}

interface TimeframeReport {
  per_coin?: Record<string, SliceReport>;
  pooled?: SliceReport | null;
}

interface TrainingReport {
  status?: string;
  generated_at?: string;
  timeframes?: Record<string, TimeframeReport>;
}

interface Row {
  timeframe: string;
  coinId: string;
  version: string | null;
  gates: GatesAlignment;
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;
const TF_ORDER = ["1m", "5m", "1h", "2h", "6h", "1d"];
const tfRank = (tf: string) => {
  const i = TF_ORDER.indexOf(tf);
  return i === -1 ? 99 : i;
};

// A slice is "misaligned enough to flag" when more than a third of its
// holdout sits in one of the wasted-budget buckets. Tunable threshold so
// the banner only fires when it actually means something.
const FLAG_THRESHOLD = 0.33;

function pct(x: number) {
  return `${(x * 100).toFixed(1)}%`;
}

export function GatesAlignmentCard() {
  const [data, setData] = useState<TrainingReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const res = await fetch(apiUrl(`/crypto/quant-training-report`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData((await res.json()) as TrainingReport);
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

  const rows: Row[] = useMemo(() => {
    const out: Row[] = [];
    for (const [tf, tfRep] of Object.entries(data?.timeframes ?? {})) {
      for (const [coin, slc] of Object.entries(tfRep.per_coin ?? {})) {
        if (slc?.status === "trained" && slc.gates_alignment) {
          out.push({ timeframe: tf, coinId: coin, version: slc.version ?? null, gates: slc.gates_alignment });
        }
      }
      const pooled = tfRep.pooled;
      if (pooled?.status === "trained" && pooled.gates_alignment) {
        out.push({ timeframe: tf, coinId: "__pooled__", version: pooled.version ?? null, gates: pooled.gates_alignment });
      }
    }
    out.sort((a, b) => tfRank(a.timeframe) - tfRank(b.timeframe) || a.coinId.localeCompare(b.coinId));
    return out;
  }, [data]);

  const flagged = rows.filter(
    r =>
      r.gates.loud_classifier_quiet_regressor_share >= FLAG_THRESHOLD ||
      r.gates.quiet_classifier_loud_regressor_share >= FLAG_THRESHOLD,
  );

  const isMissing = data?.status === "missing";

  return (
    <Card data-testid="gates-alignment-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <GitCompareArrows className="h-5 w-5" />
          Gates aligned (sign vs magnitude)
          {data?.generated_at && (
            <Badge variant="outline" className="ml-2 text-[10px]">
              {new Date(data.generated_at).toLocaleString()}
            </Badge>
          )}
        </CardTitle>
        <Button size="sm" variant="ghost" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {err && <div className="text-xs text-red-400">Error: {err}</div>}

        {isMissing && (
          <div className="text-xs text-muted-foreground">
            No training report yet. Run a training pass to populate this card.
          </div>
        )}

        {!isMissing && rows.length === 0 && !err && (
          <div className="text-xs text-muted-foreground">
            No trained slices report a gates-alignment number yet. Older models
            are surfaced after the next retrain.
          </div>
        )}

        {flagged.length > 0 && (
          <div
            className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-300 flex gap-2"
            data-testid="gates-alignment-alert"
          >
            <AlertTriangle className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-semibold">
                {flagged.length} slice{flagged.length === 1 ? "" : "s"} have heads
                disagreeing on more than {Math.round(FLAG_THRESHOLD * 100)}% of
                holdout predictions.
              </div>
              <div className="text-[11px] text-amber-200/80">
                Either the classifier is confident while the regressor is below
                the cost floor, or the regressor screams while the classifier is
                near 50/50. Both patterns point at gate-budget misallocation.
              </div>
            </div>
          </div>
        )}

        {rows.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-3">Slice</th>
                  <th className="py-1 pr-3 text-right">N</th>
                  <th className="py-1 pr-3 text-right" title="Both heads agree (loud+quiet)">
                    Aligned
                  </th>
                  <th className="py-1 pr-3 text-right" title="Classifier confident, regressor below cost floor">
                    Loud cls / quiet reg
                  </th>
                  <th className="py-1 pr-3 text-right" title="Regressor screams, classifier near 50/50">
                    Quiet cls / loud reg
                  </th>
                  <th className="py-1 pr-3 text-right">|edge| p95</th>
                  <th className="py-1 pr-3 text-right">|mag%| p95</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(r => {
                  const lcQr = r.gates.loud_classifier_quiet_regressor_share;
                  const qcLr = r.gates.quiet_classifier_loud_regressor_share;
                  const flagged =
                    lcQr >= FLAG_THRESHOLD || qcLr >= FLAG_THRESHOLD;
                  return (
                    <tr
                      key={`${r.timeframe}-${r.coinId}`}
                      className="border-t border-border/30"
                      data-testid={`gates-row-${r.timeframe}-${r.coinId}`}
                    >
                      <td className="py-1 pr-3 font-mono">
                        <span className="text-muted-foreground">{r.timeframe}</span>{" "}
                        {r.coinId}
                      </td>
                      <td className="py-1 pr-3 text-right tabular-nums text-muted-foreground">
                        {r.gates.n}
                      </td>
                      <td
                        className={cn(
                          "py-1 pr-3 text-right tabular-nums font-semibold",
                          r.gates.aligned_share < 0.5 ? "text-amber-400" : "text-emerald-400",
                        )}
                      >
                        {pct(r.gates.aligned_share)}
                      </td>
                      <td
                        className={cn(
                          "py-1 pr-3 text-right tabular-nums",
                          lcQr >= FLAG_THRESHOLD ? "text-amber-400" : "",
                        )}
                      >
                        {pct(lcQr)}
                      </td>
                      <td
                        className={cn(
                          "py-1 pr-3 text-right tabular-nums",
                          qcLr >= FLAG_THRESHOLD ? "text-amber-400" : "",
                        )}
                      >
                        {pct(qcLr)}
                      </td>
                      <td className="py-1 pr-3 text-right tabular-nums text-muted-foreground">
                        {r.gates.dir_edge_p95.toFixed(3)}
                      </td>
                      <td className="py-1 pr-3 text-right tabular-nums text-muted-foreground">
                        {r.gates.abs_magnitude_pct_p95.toFixed(3)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="text-[10px] text-muted-foreground">
          Heads are "loud" when they clear the live gates: classifier edge
          {data && rows[0] ? ` (≥ ${rows[0].gates.min_directional_edge.toFixed(3)})` : ""},
          regressor magnitude
          {data && rows[0] ? ` (≥ ${rows[0].gates.min_expected_return_pct.toFixed(3)}%)` : ""}.
          "Aligned" = both loud or both quiet. Computed on the calibration
          holdout per slice, refreshed every retrain.
        </div>
      </CardContent>
    </Card>
  );
}
