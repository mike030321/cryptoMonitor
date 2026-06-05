import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, RefreshCw, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface SpecialistCell {
  kind: string;
  regime: string;
  n: number;
  directionalCalls: number;
  directionalCorrect: number;
  directionalAccuracy: number | null;
  meanProbWhenCorrect: number | null;
  meanRealizedAbsPct: number | null;
}

interface SpecialistResponse {
  sampledRows: number;
  rowsWithSpecialists: number;
  consideredRows: number;
  specialists: SpecialistCell[];
  generatedAt: string;
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

const KIND_TONES: Record<string, string> = {
  momentum: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  mean_reversion: "bg-sky-500/15 text-sky-300 border-sky-500/40",
  breakout: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  volatility_forecaster: "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/40",
};

function pretty(s: string): string {
  return s.replaceAll("_", " ");
}

function fmtPct(v: number | null, digits = 1): string {
  if (v === null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtSignedPct(v: number | null, digits = 2): string {
  if (v === null || !Number.isFinite(v)) return "—";
  return `${v.toFixed(digits)}%`;
}

export function SpecialistAccuracyCard() {
  const [data, setData] = useState<SpecialistResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const res = await fetch(apiUrl("/crypto/brain/specialists"));
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

  // Group by kind for the panel layout — easier to scan than a flat list.
  const grouped = useMemo(() => {
    const m = new Map<string, SpecialistCell[]>();
    for (const c of data?.specialists ?? []) {
      const arr = m.get(c.kind) ?? [];
      arr.push(c);
      m.set(c.kind, arr);
    }
    for (const arr of m.values()) {
      arr.sort((a, b) => b.n - a.n);
    }
    return Array.from(m.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [data]);

  return (
    <Card data-testid="specialist-accuracy-card">
      <CardHeader className="flex flex-row items-start justify-between gap-3 pb-3">
        <div className="space-y-1">
          <CardTitle className="flex items-center gap-2 text-lg font-display">
            <Brain className="h-5 w-5" />
            Specialist Accuracy (Phase 3)
          </CardTitle>
          <p className="text-xs text-muted-foreground max-w-xl">
            Per-regime specialist heads (momentum, mean-reversion, breakout,
            volatility) scored from the prediction journal. Direction = sign
            of probUp − probDown vs realized return sign. Specialists do not
            gate live trades yet — Phase 4 wires the meta-model in.
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

        {data && (
          <>
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">Journal rows sampled</div>
                <div className="text-lg font-semibold">
                  {data.sampledRows.toLocaleString()}
                </div>
              </div>
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">With specialists</div>
                <div className="text-lg font-semibold">
                  {data.rowsWithSpecialists.toLocaleString()}
                </div>
              </div>
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">Considered</div>
                <div className="text-lg font-semibold">
                  {data.consideredRows.toLocaleString()}
                </div>
              </div>
            </div>

            {grouped.length === 0 ? (
              <div className="rounded-md border border-dashed border-border bg-muted/10 p-3 text-xs text-muted-foreground">
                No resolved journal rows carry specialist scores yet. Run a
                training cycle and let the live brain emit a few predictions —
                this card populates as rows resolve.
              </div>
            ) : (
              <div className="space-y-3">
                {grouped.map(([kind, cells]) => {
                  const tone = KIND_TONES[kind] ?? "bg-slate-500/15 text-slate-300 border-slate-500/40";
                  return (
                    <div
                      key={kind}
                      className="space-y-2"
                      data-testid={`specialist-block-${kind}`}
                    >
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <span
                          className={cn(
                            "inline-flex items-center rounded-full border px-2 py-0.5 font-medium",
                            tone,
                          )}
                        >
                          {pretty(kind)}
                        </span>
                        <span className="text-muted-foreground">
                          {cells.reduce((s, c) => s + c.n, 0)} rows across{" "}
                          {cells.length} regimes
                        </span>
                      </div>
                      <div className="overflow-x-auto rounded-md border border-border">
                        <table className="w-full text-xs">
                          <thead className="bg-muted/30 text-muted-foreground">
                            <tr>
                              <th className="px-2 py-1 text-left font-normal">Regime</th>
                              <th className="px-2 py-1 text-right font-normal">N</th>
                              <th className="px-2 py-1 text-right font-normal">Dir calls</th>
                              <th className="px-2 py-1 text-right font-normal">Dir acc</th>
                              <th className="px-2 py-1 text-right font-normal">p̄(correct)</th>
                              <th className="px-2 py-1 text-right font-normal">|realized|</th>
                            </tr>
                          </thead>
                          <tbody>
                            {cells.map((c) => (
                              <tr
                                key={`${kind}-${c.regime}`}
                                className="border-t border-border"
                                data-testid={`specialist-row-${kind}-${c.regime}`}
                              >
                                <td className="px-2 py-1">{pretty(c.regime)}</td>
                                <td className="px-2 py-1 text-right">{c.n}</td>
                                <td className="px-2 py-1 text-right">{c.directionalCalls}</td>
                                <td
                                  className={cn(
                                    "px-2 py-1 text-right font-medium",
                                    c.directionalAccuracy !== null
                                      ? c.directionalAccuracy >= 0.55
                                        ? "text-emerald-300"
                                        : c.directionalAccuracy <= 0.45
                                          ? "text-red-300"
                                          : "text-foreground"
                                      : "text-muted-foreground",
                                  )}
                                >
                                  {fmtPct(c.directionalAccuracy)}
                                </td>
                                <td className="px-2 py-1 text-right">
                                  {fmtPct(c.meanProbWhenCorrect)}
                                </td>
                                <td className="px-2 py-1 text-right">
                                  {fmtSignedPct(c.meanRealizedAbsPct)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="text-[10px] text-muted-foreground">
              Generated {new Date(data.generatedAt).toLocaleTimeString()}.
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
