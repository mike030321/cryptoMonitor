/**
 * Task #512 — Family drill-down page (`/agents/families/:profileId`).
 *
 * Renders the executor family's per-coin breakdown plus the underlying
 * member-agent list. Reached by clicking a card in the dashboard's
 * Family Fleet. The page reads two endpoints:
 *   - `/crypto/agents/families`            — for the family header
 *   - `/crypto/agents/families/:id/coins`  — for the coin table
 */

import { useRoute, Link } from "wouter";
import { useFamilies, useFamilyCoins, type FamilyCard } from "@/hooks/use-news";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ChevronLeft, Trophy } from "lucide-react";
import { cn } from "@/lib/utils";

const STATUS_PILL_STYLES: Record<FamilyCard["statusPill"], string> = {
  active: "border-emerald-400/40 text-emerald-300 bg-emerald-400/10",
  cautious: "border-amber-400/40 text-amber-300 bg-amber-400/10",
  suppressed: "border-red-500/40 text-red-300 bg-red-500/10",
  quarantined: "border-purple-500/40 text-purple-300 bg-purple-500/10",
};

function FamilyHeader({ family }: { family: FamilyCard }) {
  const profitable = family.realizedPnl >= 0;
  return (
    <Card className="bg-card/50 border-border/40">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-display font-semibold flex items-center gap-2">
          <Trophy className="w-5 h-5" />
          {family.displayName}
          <Badge variant="outline" className="ml-2 text-[10px] font-mono uppercase tracking-wider">
            {family.strategyFamily}
          </Badge>
          <Badge
            variant="outline"
            className={cn(
              "text-[10px] font-mono uppercase tracking-wider",
              STATUS_PILL_STYLES[family.statusPill],
            )}
            data-testid="family-detail-status"
          >
            {family.statusPill}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground max-w-3xl">{family.thesis}</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Equity</div>
            <div className="text-2xl font-display font-bold mt-1">${family.equity.toFixed(0)}</div>
            <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
              vs ${family.startingCapital.toFixed(0)} seed
            </div>
          </div>
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Net P&amp;L</div>
            <div
              className={cn(
                "text-2xl font-display font-bold mt-1",
                profitable ? "text-emerald-300" : "text-red-300",
              )}
              data-testid="family-detail-pnl"
            >
              {profitable ? "+" : ""}${family.realizedPnl.toFixed(2)}
            </div>
            <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
              {profitable ? "+" : ""}{family.realizedPnlPct.toFixed(2)}%
            </div>
          </div>
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Trades</div>
            <div className="text-2xl font-display font-bold mt-1">{family.totalTrades}</div>
            <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
              {family.winRate.toFixed(0)}% win
            </div>
          </div>
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Open</div>
            <div className="text-2xl font-display font-bold mt-1">{family.openPositions}</div>
            <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
              {family.memberCount} agent{family.memberCount === 1 ? "" : "s"}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          <div className="rounded-md border border-border/30 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Max DD</div>
            <div className="text-base font-mono mt-1 text-amber-300/90">
              {family.maxDrawdown.toFixed(2)}%
            </div>
          </div>
          <div className="rounded-md border border-border/30 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">CADA</div>
            <div className="text-base font-mono mt-1">
              {family.costAwareDirectionalAccuracy != null
                ? `${(family.costAwareDirectionalAccuracy * 100).toFixed(1)}%`
                : "—"}
            </div>
            <div className="text-[10px] font-mono text-muted-foreground mt-0.5">
              cost-aware Sharpe{" "}
              {family.costAwareSharpe != null
                ? family.costAwareSharpe.toFixed(2)
                : "—"}
            </div>
          </div>
          <div className="rounded-md border border-border/30 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Abstain rate</div>
            <div className="text-base font-mono mt-1">
              {family.abstainRate.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono text-muted-foreground mt-0.5">
              {family.abstainCount} skips · last hour
            </div>
          </div>
          <div className="rounded-md border border-border/30 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Trust × Sizing</div>
            <div
              className={cn(
                "text-base font-mono mt-1",
                family.trustMultiplier >= 1 ? "text-emerald-300" : "text-amber-300",
              )}
              data-testid="family-detail-trust"
            >
              {family.trustMultiplier.toFixed(2)}x
            </div>
            <div className="text-[10px] font-mono text-muted-foreground mt-0.5">
              meta-brain directive
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function FamilyDrillDownPage() {
  const [, params] = useRoute("/agents/families/:profileId");
  const profileId = params?.profileId ?? null;
  const { data: familiesData } = useFamilies();
  const family = (familiesData?.families ?? []).find((f) => f.profileId === profileId) ?? null;

  const { data: coinsData, isLoading } = useFamilyCoins(profileId);
  const coins = coinsData?.coins ?? [];

  return (
    <div className="space-y-6" data-testid="family-drill-down-page">
      <div>
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-xs font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground"
          data-testid="link-back-dashboard"
        >
          <ChevronLeft className="w-3.5 h-3.5" />
          Back to dashboard
        </Link>
        <h1 className="mt-2 text-3xl md:text-4xl font-display font-bold tracking-tight gradient-text">
          {family?.displayName ?? "Executor Family"}
        </h1>
      </div>

      {family ? <FamilyHeader family={family} /> : <Skeleton className="h-44 w-full" />}

      <Card className="bg-card/50 border-border/40">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono uppercase tracking-wider">
            Per-coin breakdown
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading && <Skeleton className="h-40 w-full" />}
          {!isLoading && coins.length === 0 && (
            <div className="text-center py-8 text-muted-foreground font-mono text-sm">
              No trades or open positions for this family yet.
            </div>
          )}
          {!isLoading && coins.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[10px] uppercase font-mono text-muted-foreground border-b border-border/40">
                    <th className="text-left py-2 pr-3">Coin</th>
                    <th className="text-left py-2 px-3">Status</th>
                    <th className="text-right py-2 px-3">Open / Notional</th>
                    <th className="text-right py-2 px-3">Unrealized</th>
                    <th className="text-right py-2 px-3">Realized</th>
                    <th className="text-right py-2 px-3">Drawdown</th>
                    <th className="text-right py-2 px-3">Recent acc.</th>
                    <th className="text-right py-2 px-3">vs Buy &amp; Hold</th>
                    <th className="text-right py-2 pl-3">Fallback / Trust</th>
                  </tr>
                </thead>
                <tbody>
                  {coins.map((c) => {
                    const realizedColor =
                      c.realizedPnl > 0
                        ? "text-emerald-300"
                        : c.realizedPnl < 0
                          ? "text-red-300"
                          : "text-muted-foreground";
                    const unrealizedColor =
                      c.unrealizedPnl > 0
                        ? "text-emerald-300"
                        : c.unrealizedPnl < 0
                          ? "text-red-300"
                          : "text-muted-foreground";
                    return (
                      <tr
                        key={c.coinId}
                        className="border-b border-border/20 hover:bg-background/30"
                        data-testid={`family-coin-row-${c.coinId}`}
                      >
                        <td className="py-2 pr-3">
                          <Link
                            href={`/coins/${c.coinId}`}
                            className="hover:underline underline-offset-2"
                          >
                            {c.coinName}
                          </Link>
                        </td>
                        <td className="py-2 px-3">
                          <Badge
                            variant="outline"
                            className={cn(
                              "text-[9px] uppercase tracking-wider font-mono",
                              c.suppressionState === "suppressed"
                                ? "border-red-500/40 text-red-300 bg-red-500/10"
                                : "border-emerald-400/40 text-emerald-300 bg-emerald-400/10",
                            )}
                            data-testid={`family-coin-status-${c.coinId}`}
                          >
                            {c.suppressionState}
                          </Badge>
                        </td>
                        <td className="py-2 px-3 text-right font-mono">
                          {c.openPositions} · ${c.openNotional.toFixed(0)}
                        </td>
                        <td className={cn("py-2 px-3 text-right font-mono", unrealizedColor)}>
                          {c.unrealizedPnl >= 0 ? "+" : ""}${c.unrealizedPnl.toFixed(2)}
                        </td>
                        <td className={cn("py-2 px-3 text-right font-mono", realizedColor)}>
                          {c.realizedPnl >= 0 ? "+" : ""}${c.realizedPnl.toFixed(2)}
                          <div className="text-[9px] text-muted-foreground">
                            {c.closedTrades} closed
                          </div>
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-amber-300/90">
                          {c.drawdown.toFixed(1)}%
                        </td>
                        <td className="py-2 px-3 text-right font-mono">
                          {c.predictionCount > 0 ? `${c.recentAccuracy.toFixed(0)}%` : "—"}
                          <div className="text-[9px] text-muted-foreground">
                            {c.predictionCount} preds
                          </div>
                        </td>
                        <td
                          className="py-2 px-3 text-right font-mono"
                          data-testid={`family-coin-vs-benchmark-${c.coinId}`}
                        >
                          {c.benchmarkRelative === "no_data" ||
                          c.vsBenchmarkPct === null ||
                          c.benchmarkBuyHoldPct === null ? (
                            <span className="text-muted-foreground">—</span>
                          ) : (
                            <>
                              <span
                                className={cn(
                                  c.benchmarkRelative === "outperforming"
                                    ? "text-emerald-300"
                                    : c.benchmarkRelative === "underperforming"
                                      ? "text-red-300"
                                      : "text-muted-foreground",
                                )}
                              >
                                {c.vsBenchmarkPct >= 0 ? "+" : ""}
                                {c.vsBenchmarkPct.toFixed(2)}%
                              </span>
                              <div className="text-[9px] text-muted-foreground">
                                B&amp;H {c.benchmarkBuyHoldPct >= 0 ? "+" : ""}
                                {c.benchmarkBuyHoldPct.toFixed(2)}%
                              </div>
                            </>
                          )}
                        </td>
                        <td className="py-2 pl-3 text-right font-mono">
                          <span
                            className={cn(
                              c.fallbackUsage > 0 ? "text-amber-300" : "text-muted-foreground",
                            )}
                          >
                            {c.fallbackUsage} skips
                          </span>
                          <div className="text-[9px] text-muted-foreground">
                            trust {c.trustMultiplier.toFixed(2)}x
                          </div>
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

      <RecentActivity coins={coins} />

      {family && family.memberAgentIds.length > 0 && (
        <Card className="bg-card/50 border-border/40">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-mono uppercase tracking-wider">
              Member agents
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {family.memberAgentIds.map((id) => (
                <Link
                  key={id}
                  href={`/agents/${id}`}
                  className="px-2.5 py-1 rounded-md border border-border/40 text-xs font-mono hover:bg-background/40"
                  data-testid={`family-member-link-${id}`}
                >
                  agent #{id}
                </Link>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function RecentActivity({ coins }: { coins: import("@/hooks/use-news").FamilyCoinRow[] }) {
  const decisions = coins
    .flatMap((c) => c.recentDecisions)
    .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
    .slice(0, 15);
  const trades = coins
    .flatMap((c) => c.recentTrades)
    .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
    .slice(0, 15);
  if (decisions.length === 0 && trades.length === 0) return null;
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card className="bg-card/50 border-border/40">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono uppercase tracking-wider">
            Recent decisions
          </CardTitle>
        </CardHeader>
        <CardContent>
          {decisions.length === 0 ? (
            <div className="text-xs font-mono text-muted-foreground">No recent predictions.</div>
          ) : (
            <ul className="space-y-1.5 text-xs font-mono" data-testid="recent-decisions">
              {decisions.map((d) => (
                <li
                  key={d.id}
                  className="flex items-center justify-between gap-3 border-b border-border/20 pb-1.5"
                >
                  <span className="truncate">
                    <span className="text-muted-foreground">[{d.timeframe}]</span> {d.coinName}{" "}
                    <span className="text-foreground/80">{d.direction}</span>
                  </span>
                  <span
                    className={cn(
                      "shrink-0",
                      d.outcome === "correct"
                        ? "text-emerald-300"
                        : d.outcome === "wrong"
                          ? "text-red-300"
                          : "text-muted-foreground",
                    )}
                  >
                    {d.outcome}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
      <Card className="bg-card/50 border-border/40">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono uppercase tracking-wider">
            Recent trades
          </CardTitle>
        </CardHeader>
        <CardContent>
          {trades.length === 0 ? (
            <div className="text-xs font-mono text-muted-foreground">No recent trades.</div>
          ) : (
            <ul className="space-y-1.5 text-xs font-mono" data-testid="recent-trades">
              {trades.map((t) => (
                <li
                  key={t.id}
                  className="flex items-center justify-between gap-3 border-b border-border/20 pb-1.5"
                >
                  <span className="truncate">
                    <span className="text-muted-foreground">{t.action}</span> {t.coinName} @ ${t.entryPrice.toFixed(4)}
                  </span>
                  <span
                    className={cn(
                      "shrink-0",
                      (t.pnl ?? 0) > 0
                        ? "text-emerald-300"
                        : (t.pnl ?? 0) < 0
                          ? "text-red-300"
                          : "text-muted-foreground",
                    )}
                  >
                    {t.pnl != null
                      ? `${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}`
                      : t.status}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
