import { useEffect, useMemo, useRef, useState } from "react";
import { useToast } from "@/hooks/use-toast";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  ShieldCheck,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface LeakageAudit {
  passed: boolean;
  violations?: unknown[];
}

interface PerCoinProvenance {
  rows_real?: number;
  rows_synthetic?: number;
  rejected_synthetic?: boolean;
}

interface ProvenanceSummary {
  rows_real?: number;
  rows_synthetic?: number;
  coins_rejected?: string[];
  rejected_synthetic?: boolean;
  per_coin?: Record<string, PerCoinProvenance>;
}

interface TimeframeReport {
  status?: string;
  leakage_audit?: LeakageAudit | null;
  provenance?: ProvenanceSummary | null;
  feature_coverage?: Record<string, number> | null;
  feature_density?: Record<string, number> | null;
}

interface LoopInfo {
  kind?: string;
  reason?: string;
}

interface TrainingReport {
  status?: string;
  generated_at?: string;
  loop?: LoopInfo;
  timeframes?: Record<string, TimeframeReport>;
}

interface ThresholdFallbackStatus {
  used_fallback?: boolean;
  reason?: string | null;
  path_tried?: string | null;
  error?: string;
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

const TF_ORDER = ["1m", "5m", "1h", "2h", "6h", "1d"];
const tfRank = (tf: string) => {
  const i = TF_ORDER.indexOf(tf);
  return i === -1 ? 99 : i;
};

// Coverage <80% non-null is the operator-facing threshold for "this market
// signal is barely showing up". Same bar that's used in the existing quant
// coverage card so the two views agree at a glance.
const COVERAGE_WARN = 0.8;

function pct(x: number) {
  return `${(x * 100).toFixed(0)}%`;
}

interface SortedTf {
  tf: string;
  rep: TimeframeReport;
}

interface CoverageDrillDown {
  tf: string;
  signal: string;
  coverage: number;
}

export function TrainingContractCard() {
  const [data, setData] = useState<TrainingReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();
  const notifiedRunRef = useRef<string | null>(null);
  const fallbackNotifiedRef = useRef<string | null>(null);
  const [drill, setDrill] = useState<CoverageDrillDown | null>(null);
  const [fallback, setFallback] = useState<ThresholdFallbackStatus | null>(
    null,
  );

  async function refresh() {
    setLoading(true);
    try {
      const [reportRes, fallbackRes] = await Promise.all([
        fetch(apiUrl(`/crypto/quant-training-report`)),
        fetch(apiUrl(`/crypto/threshold-fallback-status`)),
      ]);
      if (!reportRes.ok) throw new Error(`HTTP ${reportRes.status}`);
      setData((await reportRes.json()) as TrainingReport);
      if (fallbackRes.ok) {
        setFallback((await fallbackRes.json()) as ThresholdFallbackStatus);
      } else {
        setFallback({ error: `HTTP ${fallbackRes.status}` });
      }
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

  const usedFallback = fallback?.used_fallback === true;

  // Push-notify operators the first time we see the trainer flip to
  // fallback thresholds. Dedup by (reason + path) so a single ongoing
  // fallback only fires one toast even though we poll every minute.
  useEffect(() => {
    if (!usedFallback) return;
    const key = `${fallback?.reason ?? ""}::${fallback?.path_tried ?? ""}`;
    if (fallbackNotifiedRef.current === key) return;
    fallbackNotifiedRef.current = key;
    toast({
      variant: "destructive",
      title: "Trainer fell back to hardcoded label thresholds",
      description: `${fallback?.reason ?? "unknown reason"} — tried ${
        fallback?.path_tried ?? "unknown path"
      }`,
    });
  }, [usedFallback, fallback?.reason, fallback?.path_tried, toast]);

  const timeframes: SortedTf[] = useMemo(() => {
    return Object.entries(data?.timeframes ?? {})
      .map(([tf, rep]) => ({ tf, rep }))
      .sort((a, b) => tfRank(a.tf) - tfRank(b.tf));
  }, [data]);

  const isMissing = data?.status === "missing";

  const anyLeakageFailed = timeframes.some(
    ({ rep }) => rep.leakage_audit && rep.leakage_audit.passed === false,
  );
  const anyProvenanceRejected = timeframes.some(
    ({ rep }) => rep.provenance?.rejected_synthetic === true,
  );

  const totalRejectedCoins = timeframes.reduce(
    (n, { rep }) => n + (rep.provenance?.coins_rejected?.length ?? 0),
    0,
  );

  const banner = anyLeakageFailed || anyProvenanceRejected;

  // Push-notify operators when the latest training run violates the contract.
  // Dedup by `generated_at` so a single failing run only fires one toast even
  // though we poll every minute.
  useEffect(() => {
    if (!banner) return;
    const runId = data?.generated_at;
    if (!runId) return;
    if (notifiedRunRef.current === runId) return;
    notifiedRunRef.current = runId;

    const reasons: string[] = [];
    if (anyLeakageFailed) reasons.push("leakage audit failed");
    if (anyProvenanceRejected) {
      reasons.push(
        `provenance rejected synthetic rows for ${totalRejectedCoins} coin slice${
          totalRejectedCoins === 1 ? "" : "s"
        }`,
      );
    }

    toast({
      variant: "destructive",
      title: "Training contract violated",
      description: reasons.join(" · "),
    });
  }, [
    banner,
    data?.generated_at,
    anyLeakageFailed,
    anyProvenanceRejected,
    totalRejectedCoins,
    toast,
  ]);

  return (
    <Card data-testid="training-contract-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5" />
          Training contract
          {data?.generated_at && (
            <Badge variant="outline" className="ml-2 text-[10px]">
              {new Date(data.generated_at).toLocaleString()}
            </Badge>
          )}
          {data?.loop?.kind && (
            <Badge
              variant="outline"
              className="ml-1 text-[10px]"
              data-testid="training-contract-loop"
            >
              loop: {data.loop.kind}
              {data.loop.reason ? ` · ${data.loop.reason}` : ""}
            </Badge>
          )}
        </CardTitle>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => void refresh()}
          disabled={loading}
        >
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

        {usedFallback && (
          <div
            className="rounded-md border border-red-500/50 bg-red-500/10 p-3 text-xs text-red-300 flex gap-2"
            data-testid="training-threshold-fallback-banner"
          >
            <AlertTriangle className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-semibold">
                Trainer fell back to hardcoded label thresholds.
              </div>
              <div className="text-[11px] text-red-200/80 space-y-0.5">
                <div>
                  A worker couldn't read{" "}
                  <span className="font-mono">
                    shared/trading-frictions.json
                  </span>{" "}
                  and silently used the in-code mirror. Training is now at risk
                  of drifting from the live trader/backtester until the file is
                  reachable again.
                </div>
                <div>
                  Reason:{" "}
                  <span className="font-mono">
                    {fallback?.reason ?? "unknown"}
                  </span>
                </div>
                <div>
                  Path tried:{" "}
                  <span className="font-mono break-all">
                    {fallback?.path_tried ?? "unknown"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {banner && (
          <div
            className="rounded-md border border-red-500/50 bg-red-500/10 p-3 text-xs text-red-300 flex gap-2"
            data-testid="training-contract-banner"
          >
            <AlertTriangle className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-semibold">
                Training contract violated on the latest run.
              </div>
              <div className="text-[11px] text-red-200/80">
                {anyLeakageFailed && (
                  <div>
                    Leakage audit failed on at least one timeframe — a feature
                    leaked future information into the training frame.
                  </div>
                )}
                {anyProvenanceRejected && (
                  <div>
                    Provenance rejected synthetic rows for{" "}
                    {totalRejectedCoins} coin slice
                    {totalRejectedCoins === 1 ? "" : "s"}. Those slices will
                    train on real candles only.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {!isMissing && timeframes.length === 0 && !err && (
          <div className="text-xs text-muted-foreground">
            Latest report has no per-timeframe slices.
          </div>
        )}

        {timeframes.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-3">Timeframe</th>
                  <th className="py-1 pr-3">Provenance</th>
                  <th className="py-1 pr-3">Leakage audit</th>
                  <th className="py-1 pr-3">Rejected coins</th>
                  <th className="py-1 pr-3">Coverage (low signals)</th>
                </tr>
              </thead>
              <tbody>
                {timeframes.map(({ tf, rep }) => {
                  const prov = rep.provenance;
                  const leak = rep.leakage_audit;
                  const rejected = prov?.coins_rejected ?? [];
                  const provFail = prov?.rejected_synthetic === true;
                  const leakFail = leak ? leak.passed === false : false;
                  const lowCoverage = Object.entries(
                    rep.feature_coverage ?? {},
                  )
                    .filter(
                      ([col, v]) =>
                        col !== "coin_idx" &&
                        typeof v === "number" &&
                        v < COVERAGE_WARN,
                    )
                    .sort((a, b) => a[1] - b[1]);
                  return (
                    <tr
                      key={tf}
                      className="border-t border-border/30 align-top"
                      data-testid={`training-contract-row-${tf}`}
                    >
                      <td className="py-2 pr-3 font-mono">{tf}</td>
                      <td className="py-2 pr-3">
                        {prov ? (
                          <span
                            className={cn(
                              "inline-flex items-center gap-1",
                              provFail
                                ? "text-red-400"
                                : "text-emerald-400",
                            )}
                            data-testid={`training-contract-provenance-${tf}`}
                          >
                            {provFail ? (
                              <XCircle className="h-3.5 w-3.5" />
                            ) : (
                              <CheckCircle2 className="h-3.5 w-3.5" />
                            )}
                            {provFail ? "fail" : "pass"}
                            <span className="text-muted-foreground ml-1">
                              ({prov.rows_real ?? 0} real /{" "}
                              {prov.rows_synthetic ?? 0} synth)
                            </span>
                          </span>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                      <td className="py-2 pr-3">
                        {leak ? (
                          <span
                            className={cn(
                              "inline-flex items-center gap-1",
                              leakFail
                                ? "text-red-400"
                                : "text-emerald-400",
                            )}
                            data-testid={`training-contract-leakage-${tf}`}
                          >
                            {leakFail ? (
                              <XCircle className="h-3.5 w-3.5" />
                            ) : (
                              <CheckCircle2 className="h-3.5 w-3.5" />
                            )}
                            {leakFail ? "fail" : "pass"}
                            {leakFail && leak.violations && (
                              <span className="text-muted-foreground ml-1">
                                ({leak.violations.length})
                              </span>
                            )}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                      <td
                        className="py-2 pr-3"
                        data-testid={`training-contract-rejected-${tf}`}
                      >
                        {rejected.length === 0 ? (
                          <span className="text-muted-foreground">0</span>
                        ) : (
                          <span className="text-amber-400">
                            {rejected.length}
                            <span className="text-muted-foreground ml-1">
                              ({rejected.slice(0, 4).join(", ")}
                              {rejected.length > 4 ? "…" : ""})
                            </span>
                          </span>
                        )}
                      </td>
                      <td className="py-2 pr-3">
                        {Object.keys(rep.feature_coverage ?? {}).length === 0 ? (
                          <span className="text-muted-foreground">—</span>
                        ) : lowCoverage.length === 0 ? (
                          <span className="text-emerald-400">
                            all ≥ {pct(COVERAGE_WARN)}
                          </span>
                        ) : (
                          <div className="flex flex-wrap gap-1">
                            {lowCoverage.slice(0, 6).map(([col, v]) => (
                              <button
                                key={col}
                                type="button"
                                onClick={() =>
                                  setDrill({ tf, signal: col, coverage: v })
                                }
                                className="cursor-pointer"
                                data-testid={`training-contract-coverage-chip-${tf}-${col}`}
                                title={`Click to see which coins are causing the ${col} coverage gap`}
                              >
                                <Badge
                                  variant="outline"
                                  className="text-[10px] border-amber-500/50 text-amber-300 hover:bg-amber-500/10"
                                >
                                  {col}: {pct(v)}
                                </Badge>
                              </button>
                            ))}
                            {lowCoverage.length > 6 && (
                              <span className="text-muted-foreground text-[10px]">
                                +{lowCoverage.length - 6} more
                              </span>
                            )}
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="text-[10px] text-muted-foreground">
          Provenance fails when a coin slice is dropped because only synthetic
          rows were available. Leakage audit fails when a feature can see
          beyond the prediction horizon. Coverage chips list market signals
          present on fewer than {pct(COVERAGE_WARN)} of training rows. Click a
          chip to see which coins are causing the gap.
        </div>
      </CardContent>

      <CoverageDrillDownDialog
        drill={drill}
        timeframes={timeframes}
        onClose={() => setDrill(null)}
      />
    </Card>
  );
}

interface CoverageDrillDownDialogProps {
  drill: CoverageDrillDown | null;
  timeframes: SortedTf[];
  onClose: () => void;
}

interface CoinRow {
  coin: string;
  rows_real: number;
  rows_synthetic: number;
  rejected_synthetic: boolean;
  share: number;
}

function CoverageDrillDownDialog({
  drill,
  timeframes,
  onClose,
}: CoverageDrillDownDialogProps) {
  const tfRep = drill
    ? timeframes.find((t) => t.tf === drill.tf)?.rep
    : undefined;
  const perCoin = tfRep?.provenance?.per_coin ?? {};
  const coinsRejected = tfRep?.provenance?.coins_rejected ?? [];

  const rows: CoinRow[] = useMemo(() => {
    const entries = Object.entries(perCoin);
    const totalReal = entries.reduce(
      (n, [, p]) => n + (p.rows_real ?? 0),
      0,
    );
    const out: CoinRow[] = entries.map(([coin, p]) => ({
      coin,
      rows_real: p.rows_real ?? 0,
      rows_synthetic: p.rows_synthetic ?? 0,
      rejected_synthetic: Boolean(p.rejected_synthetic),
      share: totalReal > 0 ? (p.rows_real ?? 0) / totalReal : 0,
    }));
    // Worst contributors first: rejected coins, then lowest real-row share.
    out.sort((a, b) => {
      if (a.rejected_synthetic !== b.rejected_synthetic) {
        return a.rejected_synthetic ? -1 : 1;
      }
      return a.rows_real - b.rows_real;
    });
    return out;
  }, [perCoin]);

  const hasPerCoin = rows.length > 0;

  return (
    <Dialog open={!!drill} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className="max-w-lg"
        data-testid="training-contract-coverage-drilldown"
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span className="font-mono text-amber-300">
              {drill?.signal}
            </span>
            <span className="text-muted-foreground text-sm font-normal">
              · {drill?.tf} · {drill ? pct(drill.coverage) : ""} coverage
            </span>
          </DialogTitle>
          <DialogDescription>
            Per-coin breakdown of which slices contributed real rows to this
            timeframe. Coins rejected by the provenance guard contribute zero
            rows to <span className="font-mono">{drill?.signal}</span> (and
            every other signal), so they are the most likely cause of a
            coverage gap. Coins with a small share of real rows can also drag
            coverage down if a provider isn't backfilling that signal for
            them.
          </DialogDescription>
        </DialogHeader>

        {!hasPerCoin && (
          <div className="text-xs text-muted-foreground py-4">
            No per-coin provenance recorded for this timeframe.
          </div>
        )}

        {hasPerCoin && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-3">Coin</th>
                  <th className="py-1 pr-3">Status</th>
                  <th className="py-1 pr-3 text-right">Real rows</th>
                  <th className="py-1 pr-3 text-right">Synth rows</th>
                  <th className="py-1 pr-3 text-right">Share</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.coin}
                    className="border-t border-border/30"
                    data-testid={`training-contract-coverage-coin-${r.coin}`}
                  >
                    <td className="py-1 pr-3 font-mono">{r.coin}</td>
                    <td className="py-1 pr-3">
                      {r.rejected_synthetic ? (
                        <span className="inline-flex items-center gap-1 text-red-400">
                          <XCircle className="h-3.5 w-3.5" />
                          rejected
                        </span>
                      ) : r.rows_real === 0 ? (
                        <span className="inline-flex items-center gap-1 text-amber-400">
                          <AlertTriangle className="h-3.5 w-3.5" />
                          no rows
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-emerald-400">
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          contributing
                        </span>
                      )}
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {r.rows_real}
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {r.rows_synthetic}
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {pct(r.share)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {coinsRejected.length > 0 && (
          <div className="text-[11px] text-red-300/80 border-t border-border/30 pt-2">
            <span className="font-semibold">
              {coinsRejected.length} coin
              {coinsRejected.length === 1 ? "" : "s"} rejected by the
              provenance guard:
            </span>{" "}
            <span className="font-mono">{coinsRejected.join(", ")}</span>
          </div>
        )}

        <div className="text-[10px] text-muted-foreground">
          This view is built from the per-coin provenance in the latest
          training report — no extra training output is needed.
        </div>
      </DialogContent>
    </Dialog>
  );
}
