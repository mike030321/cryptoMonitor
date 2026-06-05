/**
 * Task #512 — Benchmarks panel.
 *
 * Renders Strategy-Lab `baseline_reference` rows (DCA + Circuit Breaker,
 * Buy & Hold, Trend Filter, …) as a compact comparison list, separate
 * from the 4 executor cards. Reads the existing `/paper-portfolios`
 * payload and filters by the new `kind === "benchmark"` tag the API
 * sets in Task #512.
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { LineChart } from "lucide-react";
import { Link } from "wouter";
import { usePaperPortfolios, type PaperPortfolio } from "@/hooks/use-news";
import { derivePnl } from "@/lib/derive-pnl";

function BenchmarkRow({ b }: { b: PaperPortfolio }) {
  const { netPnl, netPnlPct } = derivePnl(b);
  const profitable = netPnl >= 0;
  return (
    <Link
      href={`/agents/${b.agentId}`}
      className="block px-3 py-2 rounded-lg border border-border/30 hover:bg-background/40 transition-colors"
      data-testid={`benchmark-row-${b.agentId}`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="font-medium text-sm truncate">{b.agentName}</div>
          <div className="text-[10px] uppercase font-mono text-muted-foreground">
            {b.totalTrades} trades · {b.winRate.toFixed(0)}% win
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className="text-sm font-mono">${b.totalValue.toFixed(0)}</div>
          <div
            className={cn(
              "text-[11px] font-mono",
              profitable ? "text-emerald-300" : "text-red-300",
            )}
          >
            {profitable ? "+" : ""}${netPnl.toFixed(2)} ({profitable ? "+" : ""}{netPnlPct.toFixed(2)}%)
          </div>
        </div>
      </div>
    </Link>
  );
}

export function BenchmarksPanel() {
  const { data, isLoading } = usePaperPortfolios();
  const benchmarks = (data ?? [])
    .filter((p) => p.kind === "benchmark")
    .sort((a, b) => b.totalValue - a.totalValue);

  if (!isLoading && benchmarks.length === 0) return null;

  return (
    <Card className="bg-card/50 border-border/40" data-testid="benchmarks-panel">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <LineChart className="w-4 h-4" />
          Benchmarks
          <span className="text-[10px] font-normal text-muted-foreground normal-case tracking-normal">
            (Strategy-Lab passive baselines)
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        ) : (
          <div className="space-y-2">
            {benchmarks.map((b) => (
              <BenchmarkRow key={b.agentId} b={b} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
