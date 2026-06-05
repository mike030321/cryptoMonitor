import {
  useGetCoinDetail,
  getGetCoinDetailQueryKey,
  useGetPriceHistory,
  getGetPriceHistoryQueryKey,
} from "@workspace/api-client-react";
import { useParams, Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { formatCurrency, formatCompactCurrency, formatTimeAgo } from "@/lib/format";
import { ArrowLeft, TrendingUp, TrendingDown, Minus, CheckCircle, XCircle, Clock, AlertTriangle, Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { useState } from "react";

const RANGE_OPTIONS = [
  { key: "24h", label: "24h", hours: 24 },
  { key: "7d", label: "7d", hours: 24 * 7 },
  { key: "30d", label: "30d", hours: 24 * 30 },
] as const;
type RangeKey = typeof RANGE_OPTIONS[number]["key"];

export default function CoinDetail() {
  const params = useParams<{ id: string }>();
  const coinId = params.id ?? "";

  const [rangeKey, setRangeKey] = useState<RangeKey>("24h");
  const range = RANGE_OPTIONS.find((r) => r.key === rangeKey) ?? RANGE_OPTIONS[0];
  const rangeHours = range.hours;

  const { data, isLoading } = useGetCoinDetail(
    coinId,
    { hours: rangeHours },
    {
      query: {
        enabled: !!coinId,
        queryKey: getGetCoinDetailQueryKey(coinId, { hours: rangeHours }),
        refetchInterval: 15000,
      },
    },
  );

  const { data: priceHistory } = useGetPriceHistory(
    coinId,
    { hours: rangeHours },
    {
      query: {
        enabled: !!coinId,
        queryKey: getGetPriceHistoryQueryKey(coinId, { hours: rangeHours }),
        refetchInterval: 30000,
      },
    },
  );

  const isMultiDay = rangeHours > 48;
  const formatChartTick = (t: string | number | Date) => {
    const d = new Date(t);
    return isMultiDay
      ? d.toLocaleDateString([], { month: "short", day: "numeric" })
      : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  if (isLoading) {
    return (
      <div className="space-y-6" data-testid="coin-detail-loading">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-64" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12" data-testid="coin-not-found">
        <p className="text-muted-foreground font-mono">Coin not found</p>
        <Link href="/coins" className="text-primary font-mono text-sm mt-2 inline-block">
          Back to markets
        </Link>
      </div>
    );
  }

  const { coin, recentPredictions, agentAgreement, recentSkipEvents, windowHours } = data;

  const directionIcon = (dir: string) => {
    if (dir === "up") return <TrendingUp className="w-4 h-4 text-emerald-400" />;
    if (dir === "down") return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-yellow-400" />;
  };

  const outcomeIcon = (outcome: string | null | undefined) => {
    if (outcome === "correct") return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    if (outcome === "wrong") return <XCircle className="w-4 h-4 text-red-400" />;
    return <Clock className="w-4 h-4 text-yellow-400" />;
  };

  const changeIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="w-4 h-4 text-emerald-400" />;
    if (change < 0) return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-yellow-400" />;
  };

  const total = agentAgreement.total;
  const upPct = total > 0 ? (agentAgreement.up / total) * 100 : 0;
  const downPct = total > 0 ? (agentAgreement.down / total) * 100 : 0;
  const stablePct = total > 0 ? (agentAgreement.stable / total) * 100 : 0;

  return (
    <div className="space-y-6" data-testid="coin-detail-page">
      <div className="flex items-center gap-4">
        <Link href="/coins">
          <div className="p-2 rounded-lg border border-border/30 hover:border-primary/30 transition-colors cursor-pointer" data-testid="link-back-coins">
            <ArrowLeft className="w-4 h-4" />
          </div>
        </Link>
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center font-display font-bold text-base text-primary">
            {coin.symbol.slice(0, 2)}
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-display font-bold tracking-wider" data-testid="text-coin-name">
                {coin.name}
              </h1>
              <Badge variant="outline" className="font-mono text-xs">{coin.symbol}</Badge>
            </div>
            <p className="text-sm font-mono text-muted-foreground mt-1">
              Updated {formatTimeAgo(coin.lastUpdated)}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-coin-price">
          <CardContent className="pt-6">
            <div className="text-3xl font-display font-bold">{formatCurrency(coin.currentPrice)}</div>
            <div className="text-xs font-mono text-muted-foreground uppercase mt-1">Current Price</div>
          </CardContent>
        </Card>
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-coin-change">
          <CardContent className="pt-6">
            <div className={cn(
              "text-3xl font-display font-bold flex items-center gap-2",
              coin.priceChange24h > 0 ? "text-emerald-400" :
              coin.priceChange24h < 0 ? "text-red-400" : "text-yellow-400",
            )}>
              {changeIcon(coin.priceChange24h)}
              {coin.priceChange24h > 0 ? "+" : ""}{coin.priceChange24h.toFixed(2)}%
            </div>
            <div className="text-xs font-mono text-muted-foreground uppercase mt-1">24h Change</div>
          </CardContent>
        </Card>
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-coin-volume">
          <CardContent className="pt-6">
            <div className="text-3xl font-display font-bold">{formatCompactCurrency(coin.volume24h)}</div>
            <div className="text-xs font-mono text-muted-foreground uppercase mt-1">Volume 24h</div>
          </CardContent>
        </Card>
        <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-coin-mcap">
          <CardContent className="pt-6">
            <div className="text-3xl font-display font-bold">{formatCompactCurrency(coin.marketCap)}</div>
            <div className="text-xs font-mono text-muted-foreground uppercase mt-1">Market Cap</div>
          </CardContent>
        </Card>
      </div>

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-price-history">
        <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <Activity className="h-4 w-4 text-cyan-400" />
            Price History · last {range.label}
          </CardTitle>
          <div className="flex items-center gap-1" data-testid="price-history-range-selector">
            {RANGE_OPTIONS.map((opt) => (
              <Button
                key={opt.key}
                size="sm"
                variant={opt.key === rangeKey ? "default" : "outline"}
                className="h-7 px-2 text-xs font-mono"
                onClick={() => setRangeKey(opt.key)}
                data-testid={`button-range-${opt.key}`}
                aria-pressed={opt.key === rangeKey}
              >
                {opt.label}
              </Button>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          {priceHistory && priceHistory.length >= 2 ? (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={priceHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis
                    dataKey="timestamp"
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={11}
                    tickFormatter={formatChartTick}
                    minTickGap={40}
                  />
                  <YAxis
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={11}
                    domain={["auto", "auto"]}
                    tickFormatter={(v) => formatCurrency(v)}
                    width={80}
                  />
                  <Tooltip
                    contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontFamily: "monospace", fontSize: "12px" }}
                    formatter={(value: number) => [formatCurrency(value), "Price"]}
                    labelFormatter={(t) => new Date(t).toLocaleString()}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke={priceHistory[priceHistory.length - 1].price >= priceHistory[0].price ? "#00ff9d" : "#ff3366"}
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-72 flex items-center justify-center text-sm font-mono text-muted-foreground">
              Collecting data...
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-agent-agreement">
        <CardHeader>
          <CardTitle className="text-sm font-mono uppercase tracking-wider">
            Agent Agreement · last {windowHours}h
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {total === 0 ? (
            <div className="text-sm font-mono text-muted-foreground text-center py-4">
              No predictions in the last {windowHours}h.
            </div>
          ) : (
            <>
              <div className="grid grid-cols-3 gap-3">
                <div className="p-3 rounded border border-emerald-400/20 bg-emerald-400/5" data-testid="agreement-up">
                  <div className="flex items-center gap-2 text-emerald-400">
                    <TrendingUp className="w-4 h-4" />
                    <span className="text-xs font-mono uppercase">Up</span>
                  </div>
                  <div className="text-2xl font-display font-bold mt-1">{agentAgreement.up}</div>
                  <Progress value={upPct} className="mt-2 h-1" />
                  <div className="text-[10px] font-mono text-muted-foreground mt-1">{upPct.toFixed(0)}%</div>
                </div>
                <div className="p-3 rounded border border-red-400/20 bg-red-400/5" data-testid="agreement-down">
                  <div className="flex items-center gap-2 text-red-400">
                    <TrendingDown className="w-4 h-4" />
                    <span className="text-xs font-mono uppercase">Down</span>
                  </div>
                  <div className="text-2xl font-display font-bold mt-1">{agentAgreement.down}</div>
                  <Progress value={downPct} className="mt-2 h-1" />
                  <div className="text-[10px] font-mono text-muted-foreground mt-1">{downPct.toFixed(0)}%</div>
                </div>
                <div className="p-3 rounded border border-yellow-400/20 bg-yellow-400/5" data-testid="agreement-stable">
                  <div className="flex items-center gap-2 text-yellow-400">
                    <Minus className="w-4 h-4" />
                    <span className="text-xs font-mono uppercase">Stable</span>
                  </div>
                  <div className="text-2xl font-display font-bold mt-1">{agentAgreement.stable}</div>
                  <Progress value={stablePct} className="mt-2 h-1" />
                  <div className="text-[10px] font-mono text-muted-foreground mt-1">{stablePct.toFixed(0)}%</div>
                </div>
              </div>
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-muted-foreground">
                  Dominant signal:{" "}
                  <span className="text-foreground font-bold uppercase" data-testid="text-dominant-signal">
                    {agentAgreement.dominant}
                  </span>
                </span>
                <span className="text-muted-foreground">
                  Avg confidence:{" "}
                  <span className="text-foreground font-bold" data-testid="text-avg-confidence">
                    {(agentAgreement.avgConfidence * 100).toFixed(0)}%
                  </span>
                </span>
                <span className="text-muted-foreground">
                  Total: <span className="text-foreground font-bold">{total}</span>
                </span>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-recent-predictions">
        <CardHeader>
          <CardTitle className="text-sm font-mono uppercase tracking-wider">
            Recent Predictions
          </CardTitle>
        </CardHeader>
        <CardContent>
          {recentPredictions.length === 0 ? (
            <div className="text-center py-6 text-muted-foreground font-mono text-sm">
              No predictions for this coin in the last {windowHours}h.
            </div>
          ) : (
            <ScrollArea className="max-h-[60vh]">
              <div className="space-y-2 pr-2">
                {recentPredictions.map((pred: typeof recentPredictions[number]) => (
                  <div
                    key={pred.id}
                    className="flex items-center justify-between p-3 rounded-lg border border-border/20 bg-background/30"
                    data-testid={`card-prediction-${pred.id}`}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      {directionIcon(pred.direction)}
                      <div className="min-w-0">
                        <Link
                          href={`/agents/${pred.agentId}`}
                          className="text-sm font-medium hover:text-primary hover:underline"
                          data-testid={`link-prediction-agent-${pred.id}`}
                        >
                          {pred.agentName}
                        </Link>
                        <div className="text-xs font-mono text-muted-foreground">
                          {formatCurrency(pred.priceAtPrediction)} {"\u2192"} {formatCurrency(pred.predictedPrice)}
                          <Badge variant="outline" className="ml-2 text-[9px] px-1 py-0 h-3.5">{pred.timeframe}</Badge>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <div className="text-right text-xs font-mono">
                        <div>{(pred.confidence * 100).toFixed(0)}% conf</div>
                        <div className="text-muted-foreground">{formatTimeAgo(pred.createdAt)}</div>
                      </div>
                      <div className="flex items-center gap-1">
                        {outcomeIcon(pred.outcome)}
                        {pred.scoreChange !== null && pred.scoreChange !== undefined && (
                          <span className={cn(
                            "text-xs font-mono font-bold",
                            pred.scoreChange > 0 ? "text-emerald-400" : "text-red-400",
                          )}>
                            {pred.scoreChange > 0 ? "+" : ""}{pred.scoreChange.toFixed(1)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-skip-events">
        <CardHeader>
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-400" />
            Skip Events · last {windowHours}h
          </CardTitle>
        </CardHeader>
        <CardContent>
          {recentSkipEvents.length === 0 ? (
            <div className="text-center py-6 text-muted-foreground font-mono text-sm" data-testid="text-no-skip-events">
              No skip events for this coin in the last {windowHours}h.
            </div>
          ) : (
            <ScrollArea className="max-h-[50vh]">
              <ul className="space-y-2 pr-2" data-testid="list-coin-skip-events">
                {recentSkipEvents.map((ev: typeof recentSkipEvents[number], i: number) => (
                  <li
                    key={`${ev.ts}-${i}`}
                    className="rounded border border-border/50 bg-card/40 p-2 font-mono text-xs"
                    data-testid={`row-coin-skip-event-${i}`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span className="text-amber-400">
                        {new Date(ev.ts).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                      </span>
                      <span className="text-muted-foreground">
                        <span className="text-foreground">{ev.agentName}</span>
                        <span className="mx-1">·</span>
                        <span>{ev.reasonLabel}</span>
                      </span>
                    </div>
                    <div className="text-foreground/90 break-words">{ev.message}</div>
                  </li>
                ))}
              </ul>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
