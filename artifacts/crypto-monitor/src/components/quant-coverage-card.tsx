import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Boxes, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface QuantCoverage {
  coins: { id: string; symbol: string; name: string }[];
  timeframes: string[];
  cells: { coinId: string; timeframe: string; source: "per-coin" | "pooled" | "none" }[];
  pooled: string[];
  darkTimeframes: string[];
  fetchedAt: string;
}

function cellStyle(source: "per-coin" | "pooled" | "none") {
  if (source === "per-coin") return "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30";
  if (source === "pooled") return "bg-sky-500/15 text-sky-300 ring-sky-500/30";
  return "bg-rose-500/10 text-rose-300/80 ring-rose-500/30";
}

function cellLabel(source: "per-coin" | "pooled" | "none") {
  if (source === "per-coin") return "●";
  if (source === "pooled") return "○";
  return "—";
}

export function QuantCoverageCard() {
  const { data, isLoading, isError } = useQuery<QuantCoverage>({
    queryKey: ["quant-coverage"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/quant-coverage`);
      if (!res.ok) throw new Error(`quant-coverage ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  return (
    <Card className="bg-card/50 border-border/40" data-testid="quant-coverage-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Boxes className="w-4 h-4" />
          Quant coverage
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            which timeframes the quant brain actually has a model for
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-32 w-full" />}
        {isError && (
          <div className="text-sm text-rose-300 font-mono" data-testid="quant-coverage-error">
            ml-engine unreachable — coverage unknown.
          </div>
        )}
        {data && (
          <div className="space-y-3">
            {data.darkTimeframes.length > 0 && (
              <div
                className="flex items-start gap-2 p-2 rounded-md bg-rose-500/10 ring-1 ring-rose-500/30 text-rose-200 text-xs"
                data-testid="quant-coverage-dark"
              >
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <div>
                  <div className="font-medium">
                    {data.darkTimeframes.length} timeframe{data.darkTimeframes.length === 1 ? "" : "s"} dark — every prediction here falls through to the LLM:
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {data.darkTimeframes.map(tf => (
                      <Badge key={tf} variant="outline" className="border-rose-500/40 text-rose-200 text-[10px] uppercase font-mono">
                        {tf}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-muted-foreground">
                    <th className="text-left font-normal py-1 pr-2">coin</th>
                    {data.timeframes.map(tf => {
                      const isDark = data.darkTimeframes.includes(tf);
                      const isPooledOnly = data.pooled.includes(tf);
                      return (
                        <th
                          key={tf}
                          className={cn(
                            "text-center font-normal py-1 px-1",
                            isDark && "text-rose-300",
                            !isDark && isPooledOnly && "text-sky-300/80",
                          )}
                        >
                          {tf}
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {data.coins.map(c => (
                    <tr key={c.id} data-testid={`quant-coverage-row-${c.id}`}>
                      <td className="py-1 pr-2 text-foreground/90">{c.symbol}</td>
                      {data.timeframes.map(tf => {
                        const cell = data.cells.find(
                          x => x.coinId === c.id && x.timeframe === tf,
                        );
                        const source = cell?.source ?? "none";
                        return (
                          <td key={tf} className="py-1 px-1 text-center">
                            <span
                              title={`${c.symbol} ${tf}: ${source}`}
                              className={cn(
                                "inline-flex items-center justify-center w-6 h-6 rounded ring-1 text-[11px]",
                                cellStyle(source),
                              )}
                              data-testid={`quant-coverage-cell-${c.id}-${tf}`}
                            >
                              {cellLabel(source)}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground font-mono">
              <span className="inline-flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded ring-1 bg-emerald-500/15 ring-emerald-500/30" />
                per-coin model
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded ring-1 bg-sky-500/15 ring-sky-500/30" />
                pooled fallback
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded ring-1 bg-rose-500/10 ring-rose-500/30" />
                no model (LLM only)
              </span>
              <span className="ml-auto text-muted-foreground/60">
                refreshes every 60s · last {new Date(data.fetchedAt).toLocaleTimeString()}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
