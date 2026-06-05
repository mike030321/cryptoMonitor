import { useGetAgent, getGetAgentQueryKey } from "@workspace/api-client-react";
import { useParams } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { formatCurrency, formatPercentage, formatTimeAgo } from "@/lib/format";
import { derivePnl } from "@/lib/derive-pnl";
import { Brain, TrendingUp, TrendingDown, Minus, ArrowLeft, CheckCircle, XCircle, Clock, Dna, GitBranch, Cpu, Thermometer, Layers, Code2, DollarSign, Wallet } from "lucide-react";
import { Link } from "wouter";
import { cn } from "@/lib/utils";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
// Task #444 — `useEvolutionStatus` is gone; the evolution lineage card
// below is rendered as a no-op fallback so the page keeps mounting.

export default function AgentDetail() {
  const params = useParams<{ id: string }>();
  const agentId = parseInt(params.id || "0", 10);
  const { data, isLoading } = useGetAgent(agentId, { query: { enabled: !!agentId, queryKey: getGetAgentQueryKey(agentId), refetchInterval: 10000 } });

  if (isLoading) {
    return (
      <div className="space-y-6" data-testid="agent-detail-loading">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-64" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12" data-testid="agent-not-found">
        <p className="text-muted-foreground font-mono">Agent not found</p>
        <Link href="/agents" className="text-primary font-mono text-sm mt-2 inline-block">Back to agents</Link>
      </div>
    );
  }

  const { agent, recentPredictions, accuracyHistory, paperPortfolio } = data;
  const a = agent;

  // Task #512 — classify the agent so the detail page shows whether it is
  // part of the live deterministic fleet or an archived legacy bot, and
  // points operators back to the right surface (family drill-down vs.
  // archived index) instead of treating every agent as a live "personality".
  const EXECUTOR_NAMES = new Set([
    "Momentum Core",
    "Mean Reversion Core",
    "Breakout Core",
    "Volatility Defensive",
  ]);
  const isLiveExecutor = EXECUTOR_NAMES.has(agent.name);
  const isArchivedLegacy = !isLiveExecutor;

  const directionIcon = (dir: string) => {
    if (dir === "up") return <TrendingUp className="w-4 h-4 text-emerald-400" />;
    if (dir === "down") return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-yellow-400" />;
  };

  const outcomeIcon = (outcome: string | null) => {
    if (outcome === "correct") return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    if (outcome === "wrong") return <XCircle className="w-4 h-4 text-red-400" />;
    return <Clock className="w-4 h-4 text-yellow-400" />;
  };

  return (
    <div className="space-y-6" data-testid="agent-detail-page">
      <div className="flex items-center gap-4">
        <Link href="/agents">
          <div className="p-2 rounded-lg border border-border/30 hover:border-primary/30 transition-colors cursor-pointer" data-testid="link-back-agents">
            <ArrowLeft className="w-4 h-4" />
          </div>
        </Link>
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <Brain className="w-6 h-6 text-primary" />
            <h1 className="text-2xl font-display font-bold tracking-wider" data-testid="text-agent-name">{agent.name}</h1>
            <Badge variant="outline" className={cn(
              "font-mono text-xs",
              agent.status === "active" ? "border-emerald-400/30 text-emerald-400" :
              agent.status === "resting" ? "border-yellow-400/30 text-yellow-400" :
              "border-red-400/30 text-red-400"
            )}>
              {agent.status}
            </Badge>
            <Badge
              variant="outline"
              className={cn(
                "font-mono text-[10px] uppercase tracking-wider",
                isLiveExecutor
                  ? "border-emerald-400/40 text-emerald-300 bg-emerald-400/10"
                  : "border-muted-foreground/40 text-muted-foreground bg-muted/10",
              )}
              data-testid="badge-agent-fleet-status"
            >
              {isLiveExecutor ? "Live executor" : "Archived legacy"}
            </Badge>
          </div>
          <p className="text-xs font-mono text-muted-foreground mt-1">
            {isLiveExecutor ? (
              <>
                Member of the deterministic v1 executor fleet — see the full{" "}
                <Link
                  href={`/agents/families/${agent.name
                    .toLowerCase()
                    .replace(/ /g, "_")}`}
                  className="text-primary hover:underline"
                  data-testid="link-family-drill-down"
                >
                  family drill-down
                </Link>{" "}
                for live operational metrics.
              </>
            ) : (
              <>
                Legacy personality bot — read-only, no longer trades through the
                live decision engine. See{" "}
                <Link
                  href="/agents/archived"
                  className="text-primary hover:underline"
                  data-testid="link-archived-index"
                >
                  Archived Agents
                </Link>{" "}
                for the full archive.
              </>
            )}
          </p>
        </div>
      </div>

      {/* Task #444 — Evolution lineage card removed with `agent-evolution`.
          Slice/strategy lineage will return in Task #468 (deterministic
          5-agent strategy registry). Until then this slot is intentionally
          blank rather than rendering placeholder data. */}

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-agent-logic">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <Cpu className="h-4 w-4 text-cyan-400" /> Agent Logic
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-2 rounded border border-border/20 bg-background/30">
              <div className="text-muted-foreground uppercase text-[10px] flex items-center gap-1"><Layers className="h-3 w-3" /> Archetype</div>
              <div className="mt-1">{a.personality}</div>
            </div>
            <div className="p-2 rounded border border-border/20 bg-background/30">
              <div className="text-muted-foreground uppercase text-[10px] flex items-center gap-1"><Thermometer className="h-3 w-3" /> Temperature</div>
              <div className="mt-1 font-bold">{a.temperature != null ? a.temperature.toFixed(2) : "—"}</div>
            </div>
            <div className="p-2 rounded border border-border/20 bg-background/30">
              <div className="text-muted-foreground uppercase text-[10px] flex items-center gap-1"><Clock className="h-3 w-3" /> Timeframes</div>
              <div className="mt-1">{a.preferredTimeframes ? a.preferredTimeframes.split(",").join(", ") : "all"}</div>
            </div>
            <div className="p-2 rounded border border-border/20 bg-background/30">
              <div className="text-muted-foreground uppercase text-[10px]">Active</div>
              {/* Render Yes only when explicitly true; missing/null → No. */}
              <div className="mt-1">{a.isActive === true ? <span className="text-emerald-400">Yes</span> : <span className="text-red-400">No</span>}</div>
            </div>
          </div>
          {a.systemPrompt && (
            <details className="group" data-testid="agent-system-prompt">
              <summary className="cursor-pointer text-xs font-mono text-muted-foreground flex items-center gap-1 hover:text-foreground transition-colors">
                <Code2 className="h-3 w-3" />
                <span className="uppercase tracking-wider">System Prompt</span>
                <span className="text-[10px] ml-1 group-open:hidden">(click to view)</span>
              </summary>
              <pre className="mt-2 p-3 rounded border border-border/20 bg-background/50 text-[11px] font-mono whitespace-pre-wrap leading-relaxed text-foreground/90 max-h-96 overflow-auto">
{a.systemPrompt}
              </pre>
            </details>
          )}
        </CardContent>
      </Card>

      {paperPortfolio && (
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-agent-portfolio">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
              <Wallet className="h-4 w-4 text-emerald-400" /> Paper Trading
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs font-mono">
              <div className="p-2 rounded border border-border/20 bg-background/30">
                <div className="text-muted-foreground uppercase text-[10px]">Cash</div>
                <div className="mt-1 font-bold">{formatCurrency(paperPortfolio.cashBalance)}</div>
              </div>
              <div className="p-2 rounded border border-border/20 bg-background/30">
                <div className="text-muted-foreground uppercase text-[10px]">Total Value</div>
                <div className="mt-1 font-bold">{formatCurrency(paperPortfolio.totalValue)}</div>
              </div>
              {/*
                Task #365 / #368 — match the dashboard leaderboard's
                equity-derived P&L via the shared `derivePnl` helper.
                The legacy `paperPortfolio.totalPnl` field is REALIZED-
                ONLY (closed positions only, ignores unrealized P&L on
                open trades) and disagreed with the equity tile next to
                it by several dollars per bot. The helper enforces
                `totalValue - startingCapital` everywhere so this tile
                cannot drift from the dashboard.
              */}
              <div className="p-2 rounded border border-border/20 bg-background/30">
                <div className="text-muted-foreground uppercase text-[10px]">P&amp;L</div>
                {(() => {
                  const { netPnl, netPnlPct } = derivePnl(paperPortfolio);
                  return (
                    <div className={cn("mt-1 font-bold", netPnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {netPnl >= 0 ? "+" : ""}{formatCurrency(netPnl)}
                      <span className="ml-1 text-[10px] opacity-70">
                        ({netPnlPct >= 0 ? "+" : ""}{netPnlPct.toFixed(2)}%)
                      </span>
                    </div>
                  );
                })()}
              </div>
              <div className="p-2 rounded border border-border/20 bg-background/30">
                <div className="text-muted-foreground uppercase text-[10px]">Trades</div>
                <div className="mt-1 font-bold">
                  <span className="text-emerald-400">{paperPortfolio.winningTrades}</span>
                  <span className="text-muted-foreground/50 mx-1">/</span>
                  <span className="text-red-400">{paperPortfolio.losingTrades}</span>
                </div>
              </div>
              <div className="p-2 rounded border border-border/20 bg-background/30">
                <div className="text-muted-foreground uppercase text-[10px]">Win Rate</div>
                <div className="mt-1 font-bold">{paperPortfolio.winRate.toFixed(1)}%</div>
              </div>
            </div>

            {paperPortfolio.openPositions.length > 0 && (
              <div>
                <div className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2">
                  Open Positions ({paperPortfolio.openPositions.length})
                </div>
                <div className="space-y-1.5">
                  {paperPortfolio.openPositions.map((pos: typeof paperPortfolio.openPositions[number], i: number) => (
                    <div key={i} className="flex items-center justify-between text-xs font-mono p-2 rounded border border-border/20 bg-background/30">
                      <div className="flex items-center gap-2">
                        {pos.direction === "up" ? <TrendingUp className="w-3 h-3 text-emerald-400" /> : <TrendingDown className="w-3 h-3 text-red-400" />}
                        <span>{pos.coinName}</span>
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5">{pos.timeframe}</Badge>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-muted-foreground">{formatCurrency(pos.entryPrice)}</span>
                        <span className={cn("font-bold w-20 text-right", pos.unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                          {pos.unrealizedPnl >= 0 ? "+" : ""}{formatCurrency(pos.unrealizedPnl)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {paperPortfolio.recentTrades.length > 0 && (
              <div>
                <div className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2">
                  Recent Closed Trades
                </div>
                {/*
                 * Task #505 — column header row. The "Worst DD" column shows
                 * the true intra-trade max-adverse-excursion derived from
                 * the per-tick `paper_position_marks` stream (Task #491),
                 * and the "Stab" badge is a bounded 0..1 stability score
                 * from the same source. Both fall back to "—" for trades
                 * that predate the mark stream so older rows render
                 * gracefully instead of showing a misleading 0.
                 */}
                <div
                  className="hidden sm:flex items-center justify-between text-[9px] font-mono uppercase tracking-wider text-muted-foreground/70 px-2 pb-1"
                  data-testid="recent-trades-header"
                >
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 shrink-0" />
                    <span>Trade</span>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <span className="w-28 text-right">Price</span>
                    <span className="w-16 text-right" title="True intra-trade max-adverse-excursion (worst drawdown vs entry) from the per-tick mark stream">Worst DD</span>
                    <span className="w-10 text-right" title="Stability score 0..1 derived from mark-to-mark return stdev. Higher = smoother hold.">Stab</span>
                    <span className="w-20 text-right">P&amp;L</span>
                  </div>
                </div>
                <div className="space-y-1.5 max-h-64 overflow-y-auto">
                  {paperPortfolio.recentTrades.slice(0, 15).map((t: typeof paperPortfolio.recentTrades[number]) => (
                    <div
                      key={t.id}
                      className="flex items-center justify-between text-xs font-mono p-2 rounded border border-border/20 bg-background/30"
                      data-testid={`recent-trade-row-${t.id}`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <DollarSign className="w-3 h-3 text-muted-foreground shrink-0" />
                        <span className="truncate">{t.coinName}</span>
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 shrink-0">{t.timeframe}</Badge>
                        <span className="text-muted-foreground text-[10px] shrink-0">{formatTimeAgo(t.createdAt)}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className="text-muted-foreground text-[10px] w-28 text-right">
                          {formatCurrency(t.entryPrice)} {t.exitPrice != null ? `→ ${formatCurrency(t.exitPrice)}` : ""}
                        </span>
                        <span
                          className={cn(
                            "w-16 text-right text-[10px] tabular-nums",
                            // Color the MAE so the operator can spot
                            // "won by luck" trades (large drawdown then
                            // recovered) at a glance. Thresholds match
                            // the rough loss bands the tuning gate uses.
                            t.maePct == null
                              ? "text-muted-foreground/60"
                              : t.maePct >= 0.05
                                ? "text-red-400"
                                : t.maePct >= 0.02
                                  ? "text-amber-400"
                                  : "text-muted-foreground",
                          )}
                          data-testid={`recent-trade-mae-${t.id}`}
                          title={
                            t.maePct == null
                              ? "No per-tick mark data for this trade (predates the mark stream)"
                              : `Worst point during the hold was ${(t.maePct * 100).toFixed(2)}% adverse vs entry`
                          }
                        >
                          {t.maePct == null ? "—" : `${(t.maePct * 100).toFixed(2)}%`}
                        </span>
                        <span className="w-10 flex justify-end" data-testid={`recent-trade-stability-${t.id}`}>
                          {t.stability == null ? (
                            <Badge
                              variant="outline"
                              className="text-[9px] px-1 py-0 h-3.5 text-muted-foreground/60 border-border/30"
                              title="No per-tick mark data for this trade"
                            >
                              —
                            </Badge>
                          ) : (
                            <Badge
                              variant="outline"
                              className={cn(
                                "text-[9px] px-1 py-0 h-3.5 tabular-nums",
                                t.stability >= 0.7
                                  ? "border-emerald-500/40 text-emerald-400"
                                  : t.stability >= 0.4
                                    ? "border-amber-500/40 text-amber-400"
                                    : "border-red-500/40 text-red-400",
                              )}
                              title={`Stability ${t.stability.toFixed(2)} — derived from mark-to-mark return stdev (1 / (1 + 5σ))`}
                            >
                              {t.stability.toFixed(2)}
                            </Badge>
                          )}
                        </span>
                        {t.pnl != null ? (
                          <span className={cn("font-bold w-20 text-right", t.pnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                            {t.pnl >= 0 ? "+" : ""}{formatCurrency(t.pnl)}
                          </span>
                        ) : (
                          <span className="w-20 text-right text-muted-foreground/60">—</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-agent-score">
          <CardContent className="pt-6">
            <div className="text-3xl font-display font-bold">{agent.score.toFixed(1)}</div>
            <div className="text-xs font-mono text-muted-foreground uppercase mt-1">Score</div>
          </CardContent>
        </Card>
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-agent-accuracy">
          <CardContent className="pt-6">
            <div className="text-3xl font-display font-bold">{formatPercentage(agent.accuracy)}</div>
            <div className="text-xs font-mono text-muted-foreground uppercase mt-1">Accuracy</div>
            <Progress value={agent.accuracy} className="mt-2 h-1" />
          </CardContent>
        </Card>
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-agent-record">
          <CardContent className="pt-6">
            <div className="text-3xl font-display font-bold">
              <span className="text-emerald-400">{agent.correctPredictions}</span>
              <span className="text-muted-foreground/50 mx-1">/</span>
              <span className="text-red-400">{agent.wrongPredictions}</span>
            </div>
            <div className="text-xs font-mono text-muted-foreground uppercase mt-1">W/L Record</div>
          </CardContent>
        </Card>
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-agent-streak">
          <CardContent className="pt-6">
            <div className="text-3xl font-display font-bold flex items-center gap-2">
              {agent.streak}
              {agent.streakType === "win" ? <TrendingUp className="w-5 h-5 text-emerald-400" /> : agent.streakType === "loss" ? <TrendingDown className="w-5 h-5 text-red-400" /> : null}
            </div>
            <div className="text-xs font-mono text-muted-foreground uppercase mt-1">
              {agent.streakType !== "none" ? `${agent.streakType} Streak` : "No Streak"}
            </div>
          </CardContent>
        </Card>
      </div>

      {accuracyHistory.length > 0 && (
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-accuracy-chart">
          <CardHeader>
            <CardTitle className="text-sm font-mono uppercase tracking-wider">Performance Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={accuracyHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="timestamp" tick={false} stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <Tooltip
                    contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontFamily: "monospace", fontSize: "12px" }}
                    labelFormatter={() => ""}
                  />
                  <Line type="monotone" dataKey="accuracy" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} name="Accuracy %" />
                  <Line type="monotone" dataKey="cumulativeScore" stroke="hsl(var(--secondary))" strokeWidth={2} dot={false} name="Score" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-recent-predictions">
        <CardHeader>
          <CardTitle className="text-sm font-mono uppercase tracking-wider">Recent Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {recentPredictions.map((pred) => (
              <div
                key={pred.id}
                className="flex items-center justify-between p-3 rounded-lg border border-border/20 bg-background/30"
                data-testid={`card-prediction-${pred.id}`}
              >
                <div className="flex items-center gap-3">
                  {directionIcon(pred.direction)}
                  <div>
                    <div className="text-sm font-medium">{pred.coinName}</div>
                    <div className="text-xs font-mono text-muted-foreground">
                      {formatCurrency(pred.priceAtPrediction)} {"\u2192"} {formatCurrency(pred.predictedPrice)}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right text-xs font-mono">
                    <div>{(pred.confidence * 100).toFixed(0)}% conf</div>
                    <div className="text-muted-foreground">{formatTimeAgo(pred.createdAt)}</div>
                  </div>
                  <div className="flex items-center gap-1">
                    {outcomeIcon(pred.outcome)}
                    {pred.scoreChange !== null && pred.scoreChange !== undefined && (
                      <span className={cn("text-xs font-mono font-bold", pred.scoreChange > 0 ? "text-emerald-400" : "text-red-400")}>
                        {pred.scoreChange > 0 ? "+" : ""}{pred.scoreChange.toFixed(1)}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {recentPredictions.length === 0 && (
              <div className="text-center py-6 text-muted-foreground font-mono text-sm">
                No predictions yet
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
