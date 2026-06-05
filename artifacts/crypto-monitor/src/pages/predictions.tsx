import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { useListPredictions, useListAgents, useListCoins, getListPredictionsQueryKey } from "@workspace/api-client-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { formatCurrency, formatPercentage, formatTimeAgo } from "@/lib/format";
import { TrendingUp, TrendingDown, Minus, CheckCircle, XCircle, Clock, Filter } from "lucide-react";
import { cn } from "@/lib/utils";

export default function Predictions() {
  const [agentFilter, setAgentFilter] = useState<string>("all");
  const [coinFilter, setCoinFilter] = useState<string>("all");
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const params = {
    limit: 100,
    ...(agentFilter !== "all" ? { agentId: parseInt(agentFilter) } : {}),
    ...(coinFilter !== "all" ? { coinId: coinFilter } : {}),
  };

  const { data: predictions, isLoading } = useListPredictions(params, {
    query: { refetchInterval: 10000, queryKey: getListPredictionsQueryKey(params) },
  });
  const { data: agents } = useListAgents();
  const { data: coins } = useListCoins();

  if (isLoading) {
    return (
      <div className="space-y-6" data-testid="predictions-loading">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-96" />
      </div>
    );
  }

  const directionIcon = (dir: string) => {
    if (dir === "up") return <TrendingUp className="w-4 h-4 text-emerald-400" />;
    if (dir === "down") return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-yellow-400" />;
  };

  const outcomeIcon = (outcome: string | null) => {
    if (outcome === "correct") return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    if (outcome === "wrong") return <XCircle className="w-4 h-4 text-red-400" />;
    return <Clock className="w-4 h-4 text-yellow-400 animate-pulse" />;
  };

  return (
    <div className="space-y-6" data-testid="predictions-page">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-4xl md:text-5xl font-display font-bold tracking-tight gradient-text" data-testid="text-page-title">Live Feed</h1>
          <p className="text-sm text-muted-foreground mt-2 flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20 text-xs font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 live-dot" />
              Streaming
            </span>
            <span>{predictions?.length || 0} predictions loaded</span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <Select value={agentFilter} onValueChange={setAgentFilter}>
            <SelectTrigger className="w-[180px] font-mono text-sm" data-testid="select-agent-filter">
              <SelectValue placeholder="All Agents" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Agents</SelectItem>
              {agents?.map((a) => (
                <SelectItem key={a.id} value={a.id.toString()}>{a.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={coinFilter} onValueChange={setCoinFilter}>
            <SelectTrigger className="w-[180px] font-mono text-sm" data-testid="select-coin-filter">
              <SelectValue placeholder="All Coins" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Coins</SelectItem>
              {coins?.map((c) => (
                <SelectItem key={c.id} value={c.id}>{c.name} ({c.symbol})</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <Card className="border-border/50 bg-card/50 backdrop-blur" data-testid="card-predictions-list">
        <CardContent className="pt-6">
          <div className="space-y-2">
            {predictions?.map((pred) => {
              const isExpanded = expandedId === pred.id;
              return (
              <div
                key={pred.id}
                className={cn(
                  "rounded-lg border transition-all duration-200 cursor-pointer",
                  pred.outcome === "correct" ? "border-emerald-400/20 bg-emerald-400/5" :
                  pred.outcome === "wrong" ? "border-red-400/20 bg-red-400/5" :
                  "border-border/20 bg-background/30",
                  "hover:border-primary/30"
                )}
                data-testid={`card-prediction-${pred.id}`}
                onClick={() => setExpandedId(isExpanded ? null : pred.id)}
              >
                <div className="flex items-center justify-between p-4">
                <div className="flex items-center gap-4 min-w-0 flex-1">
                  {outcomeIcon(pred.outcome)}
                  {directionIcon(pred.direction)}
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium flex items-center gap-2">
                      <span className="text-primary/80">{pred.agentName}</span>
                      <span className="text-muted-foreground/50">|</span>
                      <span>{pred.coinName}</span>
                    </div>
                    <div className={cn("text-xs font-mono text-muted-foreground mt-1 max-w-md", !isExpanded && "truncate")}>
                      {pred.reasoning}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-6 text-right">
                  <div>
                    <div className="text-xs font-mono">
                      {formatCurrency(pred.priceAtPrediction)} {"\u2192"} {formatCurrency(pred.predictedPrice)}
                    </div>
                    {pred.actualPrice !== null && pred.actualPrice !== undefined && (
                      <div className="text-xs font-mono text-muted-foreground">
                        Actual: {formatCurrency(pred.actualPrice)}
                      </div>
                    )}
                  </div>
                  <div className="text-center">
                    {(pred as any).timeframe && (
                      <Badge variant="secondary" className="font-mono text-[10px] mb-1 block">
                        {(pred as any).timeframe}
                      </Badge>
                    )}
                    <Badge variant="outline" className="font-mono text-xs">
                      {(pred.confidence * 100).toFixed(0)}%
                    </Badge>
                    {pred.scoreChange !== null && pred.scoreChange !== undefined && (
                      <div className={cn("text-xs font-mono font-bold mt-1", pred.scoreChange > 0 ? "text-emerald-400" : "text-red-400")}>
                        {pred.scoreChange > 0 ? "+" : ""}{pred.scoreChange.toFixed(1)} pts
                      </div>
                    )}
                  </div>
                  <div className="text-xs font-mono text-muted-foreground min-w-[60px]">
                    {formatTimeAgo(pred.createdAt)}
                  </div>
                  <ChevronDown className={cn("w-4 h-4 text-muted-foreground transition-transform", isExpanded && "rotate-180")} />
                </div>
                </div>
                {isExpanded && (
                  <div className="px-4 pb-4 pt-0 border-t border-border/20 mt-1" data-testid={`card-prediction-${pred.id}-expanded`}>
                    <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground mt-3 mb-1">Full Reasoning</div>
                    <div className="text-xs font-mono text-foreground/90 whitespace-pre-wrap leading-relaxed">{pred.reasoning}</div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-3 text-[11px] font-mono">
                      <div className="p-2 rounded border border-border/20"><div className="text-muted-foreground">Direction</div><div className="font-bold capitalize">{pred.direction}</div></div>
                      <div className="p-2 rounded border border-border/20"><div className="text-muted-foreground">Confidence</div><div className="font-bold">{(pred.confidence * 100).toFixed(1)}%</div></div>
                      <div className="p-2 rounded border border-border/20"><div className="text-muted-foreground">Entry → Target</div><div className="font-bold">{formatCurrency(pred.priceAtPrediction)} → {formatCurrency(pred.predictedPrice)}</div></div>
                      <div className="p-2 rounded border border-border/20"><div className="text-muted-foreground">Outcome</div><div className="font-bold capitalize">{pred.outcome ?? "pending"}</div></div>
                    </div>
                  </div>
                )}
              </div>
            );})}
            {(!predictions || predictions.length === 0) && (
              <div className="text-center py-12 text-muted-foreground font-mono text-sm">
                No predictions yet. The AI agents will start analyzing shortly...
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
