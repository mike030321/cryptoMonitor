import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, RefreshCw, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ActionDistribution {
  hours: number;
  total: number;
  byRegime: Record<string, { long: number; short: number; no_trade: number }>;
  // Task #455 — counts split by which meta-model produced the action
  // (heuristic vs lightgbm) and by trained version. Optional because
  // older API builds may not return them.
  byMetaKind?: Record<string, { long: number; short: number; no_trade: number }>;
  byMetaVersion?: Record<string, { long: number; short: number; no_trade: number }>;
}

interface AbstainReasons {
  hours: number;
  total: number;
  counts: Record<string, number>;
}

interface EdgeDecile {
  decile: number;
  n: number;
  predictedAvg: number;
  realizedAvg: number;
}

interface EdgeDeciles {
  hours: number;
  totalSamples: number;
  deciles: EdgeDecile[];
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

const REASON_LABELS: Record<string, string> = {
  meta_no_trade_low_edge: "Low edge",
  meta_no_trade_bad_regime_fit: "Bad regime fit",
  meta_no_trade_specialist_disagreement: "Specialist disagreement",
  meta_no_trade_low_calibration: "Low calibration",
};

const ACTION_FILL: Record<string, string> = {
  long: "bg-emerald-500",
  short: "bg-red-500",
  no_trade: "bg-slate-500",
};

export function MetaModelDiagnosticsCard() {
  const [actions, setActions] = useState<ActionDistribution | null>(null);
  const [reasons, setReasons] = useState<AbstainReasons | null>(null);
  const [deciles, setDeciles] = useState<EdgeDeciles | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const [a, r, d] = await Promise.all([
        fetch(apiUrl("/crypto/meta/action-distribution?hours=24")),
        fetch(apiUrl("/crypto/meta/abstain-reasons?hours=24")),
        fetch(apiUrl("/crypto/meta/edge-deciles?hours=168")),
      ]);
      if (!a.ok || !r.ok || !d.ok) throw new Error("HTTP error");
      setActions(await a.json());
      setReasons(await r.json());
      setDeciles(await d.json());
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

  return (
    <Card data-testid="meta-model-diagnostics-card">
      <CardHeader className="flex flex-row items-start justify-between gap-3 pb-3">
        <div className="space-y-1">
          <CardTitle className="flex items-center gap-2 text-lg font-display">
            <Brain className="h-5 w-5" />
            Meta-Model Decision (Phase 4)
          </CardTitle>
          <p className="text-xs text-muted-foreground max-w-md">
            The learned meta-model is the primary trade gate: action
            (long/short/no-trade), size multiplier, expected edge. Below: action
            mix by regime over 24h, abstain reasons over 24h, and the realized-vs-predicted
            edge calibration by decile over 7d.
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
      <CardContent className="space-y-6">
        {err && (
          <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-400">
            <AlertTriangle className="h-3.5 w-3.5" />
            {err}
          </div>
        )}

        {actions && (actions.byMetaKind || actions.byMetaVersion) && (
          <div className="space-y-2" data-testid="meta-served-by">
            <div className="text-xs font-semibold text-muted-foreground">
              Served by · last 24h
            </div>
            <div className="flex flex-wrap gap-1.5 text-[11px]">
              {actions.byMetaKind &&
                Object.entries(actions.byMetaKind)
                  .sort((a, b) => {
                    const at = a[1].long + a[1].short + a[1].no_trade;
                    const bt = b[1].long + b[1].short + b[1].no_trade;
                    return bt - at;
                  })
                  .map(([kind, c]) => {
                    const tot = c.long + c.short + c.no_trade;
                    const tone =
                      kind === "lightgbm"
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                        : kind === "heuristic"
                          ? "border-amber-500/40 bg-amber-500/10 text-amber-400"
                          : "border-border bg-muted/20 text-muted-foreground";
                    return (
                      <span
                        key={`mk-${kind}`}
                        data-testid={`meta-kind-chip-${kind}`}
                        className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5", tone)}
                      >
                        <span className="font-semibold">{kind}</span>
                        <span className="text-muted-foreground">({tot})</span>
                      </span>
                    );
                  })}
              {actions.byMetaVersion &&
                Object.entries(actions.byMetaVersion)
                  .sort((a, b) => {
                    const at = a[1].long + a[1].short + a[1].no_trade;
                    const bt = b[1].long + b[1].short + b[1].no_trade;
                    return bt - at;
                  })
                  .slice(0, 6)
                  .map(([version, c]) => {
                    const tot = c.long + c.short + c.no_trade;
                    return (
                      <span
                        key={`mv-${version}`}
                        data-testid={`meta-version-chip-${version}`}
                        className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/20 px-2 py-0.5 text-muted-foreground"
                        title={version}
                      >
                        <span className="font-mono">{version.slice(0, 16)}</span>
                        <span>({tot})</span>
                      </span>
                    );
                  })}
            </div>
          </div>
        )}

        {actions && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-muted-foreground">
              Action distribution by regime · {actions.total} predictions / 24h
            </div>
            {Object.keys(actions.byRegime).length === 0 ? (
              <div className="text-xs text-muted-foreground italic">
                No quant predictions in the last 24h.
              </div>
            ) : (
              Object.entries(actions.byRegime).map(([regime, c]) => {
                const tot = c.long + c.short + c.no_trade || 1;
                return (
                  <div key={regime} className="space-y-1" data-testid={`meta-action-row-${regime}`}>
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <span className="font-medium">{regime.replaceAll("_", " ")}</span>
                      <span className="text-muted-foreground">
                        long {c.long} · short {c.short} · skip {c.no_trade}
                      </span>
                    </div>
                    <div className="flex h-2 w-full overflow-hidden rounded-full bg-muted/30">
                      <div className={cn(ACTION_FILL.long)} style={{ width: `${(c.long / tot) * 100}%` }} />
                      <div className={cn(ACTION_FILL.short)} style={{ width: `${(c.short / tot) * 100}%` }} />
                      <div className={cn(ACTION_FILL.no_trade)} style={{ width: `${(c.no_trade / tot) * 100}%` }} />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {reasons && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-muted-foreground">
              Abstain reasons · {reasons.total} skips / 24h
            </div>
            {reasons.total === 0 ? (
              <div className="text-xs text-muted-foreground italic">
                Meta hasn't recorded any abstains yet.
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2 text-xs">
                {Object.entries(reasons.counts).map(([reason, count]) => {
                  const pct = (count / reasons.total) * 100;
                  return (
                    <div
                      key={reason}
                      className="rounded-md border border-border bg-muted/20 p-2"
                      data-testid={`meta-abstain-${reason}`}
                    >
                      <div className="text-muted-foreground">{REASON_LABELS[reason] ?? reason}</div>
                      <div className="text-base font-semibold">
                        {count} <span className="text-xs text-muted-foreground">({pct.toFixed(0)}%)</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {deciles && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-muted-foreground">
              Realized vs predicted edge by decile · {deciles.totalSamples} resolved trades / 7d
            </div>
            {deciles.deciles.length === 0 ? (
              <div className="text-xs text-muted-foreground italic">
                Not enough resolved trades yet for edge calibration.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-left text-muted-foreground">
                      <th className="py-1 pr-3">Decile</th>
                      <th className="py-1 pr-3">N</th>
                      <th className="py-1 pr-3">Predicted edge</th>
                      <th className="py-1 pr-3">Realized edge</th>
                    </tr>
                  </thead>
                  <tbody>
                    {deciles.deciles.map((d) => {
                      const realizedTone =
                        d.realizedAvg > 0 ? "text-emerald-400" : d.realizedAvg < 0 ? "text-red-400" : "";
                      return (
                        <tr key={d.decile} className="border-t border-border/40">
                          <td className="py-1 pr-3">{d.decile}</td>
                          <td className="py-1 pr-3">{d.n}</td>
                          <td className="py-1 pr-3">{d.predictedAvg.toFixed(3)}%</td>
                          <td className={cn("py-1 pr-3 font-medium", realizedTone)}>
                            {d.realizedAvg.toFixed(3)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
