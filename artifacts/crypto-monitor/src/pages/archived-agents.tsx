/**
 * Task #512 — Archived Agents page (`/agents/archived`).
 *
 * Hidden-by-default page that lists every legacy personality row that
 * the boot archive sweep flipped to `archivedAt != null OR
 * profile_id='legacy_archived'`. Linked from the sidebar so operators
 * can still inspect historical predictions and trade journals via the
 * existing `/agents/:id` detail page.
 */

import { Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Archive, ChevronLeft } from "lucide-react";
import { useArchivedAgents } from "@/hooks/use-news";
import { formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

export default function ArchivedAgentsPage() {
  const { data, isLoading } = useArchivedAgents();
  const rows = data?.archived ?? [];

  return (
    <div className="space-y-6" data-testid="archived-agents-page">
      <div>
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-xs font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="w-3.5 h-3.5" />
          Back to dashboard
        </Link>
        <h1 className="mt-2 text-3xl md:text-4xl font-display font-bold tracking-tight gradient-text">
          Archived Agents
        </h1>
        <p className="mt-2 text-sm text-muted-foreground max-w-3xl">
          Legacy personality bots retained for historical analytics. They
          no longer trade through the live decision engine — every trade
          and prediction is preserved in their detail page.
        </p>
      </div>

      <Card className="bg-card/50 border-border/40">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <Archive className="w-4 h-4" />
            {isLoading ? "Loading…" : `${rows.length} archived agents`}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading && <Skeleton className="h-40 w-full" />}
          {!isLoading && rows.length === 0 && (
            <div className="text-center py-8 text-muted-foreground font-mono text-sm">
              No archived agents.
            </div>
          )}
          {!isLoading && rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[10px] uppercase font-mono text-muted-foreground border-b border-border/40">
                    <th className="text-left py-2 pr-3">Name</th>
                    <th className="text-left py-2 px-3">Legacy type</th>
                    <th className="text-right py-2 px-3">Trades</th>
                    <th className="text-right py-2 px-3">Win rate</th>
                    <th className="text-right py-2 px-3">Lifetime P&amp;L</th>
                    <th className="text-right py-2 px-3">Max DD</th>
                    <th className="text-right py-2 px-3">Last active</th>
                    <th className="text-right py-2 pl-3">Archived on</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const totalPreds = r.totalPredictions || 0;
                    const tradeWinRate =
                      r.tradeCount > 0
                        ? (r.winningTrades / r.tradeCount) * 100
                        : totalPreds > 0
                          ? (r.correctPredictions / totalPreds) * 100
                          : 0;
                    const pnlColor =
                      r.lifetimePnl > 0
                        ? "text-emerald-300"
                        : r.lifetimePnl < 0
                          ? "text-red-300"
                          : "text-muted-foreground";
                    return (
                      <tr
                        key={r.id}
                        className="border-b border-border/20 hover:bg-background/30"
                        data-testid={`archived-agent-row-${r.id}`}
                      >
                        <td className="py-2 pr-3">
                          <Link
                            href={`/agents/${r.id}`}
                            className="hover:underline underline-offset-2"
                          >
                            {r.name}
                          </Link>
                          <div className="text-[10px] font-mono text-muted-foreground">
                            score {r.score}
                          </div>
                        </td>
                        <td className="py-2 px-3">
                          <Badge
                            variant="outline"
                            className="text-[10px] font-mono border-border/40 text-muted-foreground max-w-[280px] whitespace-normal text-left"
                            data-testid={`archived-agent-type-${r.id}`}
                          >
                            {r.legacyType ?? r.personality ?? "—"}
                          </Badge>
                        </td>
                        <td className="py-2 px-3 text-right font-mono">
                          {r.tradeCount}
                          <div className="text-[10px] text-muted-foreground">
                            {totalPreds} preds
                          </div>
                        </td>
                        <td className="py-2 px-3 text-right font-mono">
                          {r.tradeCount > 0 || totalPreds > 0
                            ? `${tradeWinRate.toFixed(0)}%`
                            : "—"}
                        </td>
                        <td
                          className={cn("py-2 px-3 text-right font-mono", pnlColor)}
                          data-testid={`archived-agent-pnl-${r.id}`}
                        >
                          {r.lifetimePnl >= 0 ? "+" : ""}${r.lifetimePnl.toFixed(2)}
                          <div className="text-[10px] text-muted-foreground">
                            {r.lifetimePnlPct >= 0 ? "+" : ""}
                            {r.lifetimePnlPct.toFixed(2)}%
                          </div>
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-amber-300/90">
                          {r.maxDrawdown.toFixed(2)}%
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-muted-foreground">
                          {r.lastActiveAt ? formatTimeAgo(r.lastActiveAt) : "—"}
                        </td>
                        <td className="py-2 pl-3 text-right font-mono text-muted-foreground">
                          {r.archivedOn ? formatTimeAgo(r.archivedOn) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
