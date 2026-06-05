import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ShieldOff, RefreshCw, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface RegimeCell {
  regime: string;
  total: number;
  abstained: number;
  traded: number;
  passthrough: number;
  abstainRate: number | null;
  reasonCounts: Record<string, number>;
}

interface AbstainRateResponse {
  sampledRows: number;
  rowsWithMetaGate: number;
  overall: {
    total: number;
    abstained: number;
    traded: number;
    passthrough: number;
    abstainRate: number | null;
    reasonCounts: Record<string, number>;
  };
  regimes: RegimeCell[];
  generatedAt: string;
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

function pretty(s: string): string {
  return s.replaceAll("_", " ");
}

function fmtPct(v: number | null, digits = 1): string {
  if (v === null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function MetaAbstainRateCard() {
  const [data, setData] = useState<AbstainRateResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const res = await fetch(apiUrl("/crypto/brain/abstain-rate"));
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

  const topReasons = useMemo(() => {
    if (!data) return [] as { reason: string; count: number }[];
    return Object.entries(data.overall.reasonCounts)
      .map(([reason, count]) => ({ reason, count }))
      .sort((a, b) => b.count - a.count);
  }, [data]);

  return (
    <Card data-testid="meta-abstain-rate-card">
      <CardHeader className="flex flex-row items-start justify-between gap-3 pb-3">
        <div className="space-y-1">
          <CardTitle className="flex items-center gap-2 text-lg font-display">
            <ShieldOff className="h-5 w-5" />
            Specialist Meta-Gate Abstain Rate (Phase 4)
          </CardTitle>
          <p className="text-xs text-muted-foreground max-w-xl">
            Per-regime share of quant predictions skipped by the specialist
            meta-gate. Denominator is (abstained + traded); pass-through rows
            (no quorum, or main head already abstained) are shown separately
            so they don&apos;t skew the rate.
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
            <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">Rows w/ meta-gate</div>
                <div className="text-lg font-semibold">
                  {data.rowsWithMetaGate.toLocaleString()}
                </div>
              </div>
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">Abstained</div>
                <div className="text-lg font-semibold text-amber-300">
                  {data.overall.abstained.toLocaleString()}
                </div>
              </div>
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">Traded</div>
                <div className="text-lg font-semibold text-emerald-300">
                  {data.overall.traded.toLocaleString()}
                </div>
              </div>
              <div className="rounded-md border border-border bg-muted/20 p-2">
                <div className="text-muted-foreground">Overall rate</div>
                <div className="text-lg font-semibold">
                  {fmtPct(data.overall.abstainRate)}
                </div>
              </div>
            </div>

            {data.regimes.length === 0 ? (
              <div className="rounded-md border border-dashed border-border bg-muted/10 p-3 text-xs text-muted-foreground">
                No prediction journal rows carry a meta-gate decision yet.
                Once the quant brain runs a few cycles this card populates
                automatically.
              </div>
            ) : (
              <div className="overflow-x-auto rounded-md border border-border">
                <table className="w-full text-xs">
                  <thead className="bg-muted/30 text-muted-foreground">
                    <tr>
                      <th className="px-2 py-1 text-left font-normal">Regime</th>
                      <th className="px-2 py-1 text-right font-normal">Total</th>
                      <th className="px-2 py-1 text-right font-normal">Abstained</th>
                      <th className="px-2 py-1 text-right font-normal">Traded</th>
                      <th className="px-2 py-1 text-right font-normal">Pass-through</th>
                      <th className="px-2 py-1 text-right font-normal">Abstain rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.regimes.map((r) => (
                      <tr
                        key={r.regime}
                        className="border-t border-border"
                        data-testid={`abstain-row-${r.regime}`}
                      >
                        <td className="px-2 py-1">{pretty(r.regime)}</td>
                        <td className="px-2 py-1 text-right">{r.total}</td>
                        <td className="px-2 py-1 text-right text-amber-300">
                          {r.abstained}
                        </td>
                        <td className="px-2 py-1 text-right text-emerald-300">
                          {r.traded}
                        </td>
                        <td className="px-2 py-1 text-right text-muted-foreground">
                          {r.passthrough}
                        </td>
                        <td
                          className={cn(
                            "px-2 py-1 text-right font-medium",
                            r.abstainRate !== null
                              ? r.abstainRate >= 0.5
                                ? "text-amber-300"
                                : "text-foreground"
                              : "text-muted-foreground",
                          )}
                        >
                          {fmtPct(r.abstainRate)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {topReasons.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">
                  Meta-gate reason breakdown
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {topReasons.map((r) => (
                    <span
                      key={r.reason}
                      className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/20 px-2 py-0.5 text-[11px]"
                      data-testid={`reason-chip-${r.reason}`}
                    >
                      <span className="font-medium">{pretty(r.reason)}</span>
                      <span className="text-muted-foreground">{r.count}</span>
                    </span>
                  ))}
                </div>
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
