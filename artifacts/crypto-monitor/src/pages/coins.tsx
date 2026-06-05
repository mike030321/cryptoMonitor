import { useEffect } from "react";
import { Link } from "wouter";
import { useListCoins, useGetPriceHistory, getGetPriceHistoryQueryKey, getListCoinsQueryKey } from "@workspace/api-client-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { formatCurrency, formatCompactCurrency, formatPercentage } from "@/lib/format";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";

function CoinPriceChart({ coinId }: { coinId: string }) {
  const { data: history } = useGetPriceHistory(coinId, { hours: 1 }, {
    query: { refetchInterval: 30000, queryKey: getGetPriceHistoryQueryKey(coinId, { hours: 1 }) },
  });

  if (!history || history.length < 2) {
    return (
      <div className="h-16 flex items-center justify-center text-xs font-mono text-muted-foreground">
        Collecting data...
      </div>
    );
  }

  const isUp = history[history.length - 1].price >= history[0].price;

  return (
    <div className="h-16">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={history}>
          <YAxis domain={["auto", "auto"]} hide />
          <Line
            type="monotone"
            dataKey="price"
            stroke={isUp ? "#00ff9d" : "#ff3366"}
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Coins() {
  const { data: coins, isLoading } = useListCoins({
    query: { queryKey: getListCoinsQueryKey(), refetchInterval: 15000 },
  });

  useEffect(() => {
    if (isLoading || !coins?.length) return undefined;
    if (typeof window === "undefined") return undefined;
    const hash = window.location.hash;
    if (!hash || !hash.startsWith("#coin-")) return undefined;
    const id = hash.slice("#coin-".length);
    if (!id) return undefined;
    const target = document.getElementById(`coin-${id}`);
    if (!target) return undefined;
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("ring-2", "ring-primary/60");
    const t = window.setTimeout(() => {
      target.classList.remove("ring-2", "ring-primary/60");
    }, 2000);
    return () => window.clearTimeout(t);
  }, [isLoading, coins]);

  if (isLoading) {
    return (
      <div className="space-y-6" data-testid="coins-loading">
        <Skeleton className="h-10 w-48" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-48" />)}
        </div>
      </div>
    );
  }

  const changeIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="w-4 h-4 text-emerald-400" />;
    if (change < 0) return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-yellow-400" />;
  };

  return (
    <div className="space-y-6" data-testid="coins-page">
      <div>
        <h1 className="text-4xl md:text-5xl font-display font-bold tracking-tight gradient-text" data-testid="text-page-title">Markets</h1>
        <p className="text-sm text-muted-foreground mt-2 flex items-center gap-2 flex-wrap">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-secondary/15 text-secondary ring-1 ring-secondary/25 text-xs font-semibold">
            {coins?.length || 0} altcoins
          </span>
          <span>High-risk, high-reward · under live surveillance</span>
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {coins?.map((coin) => (
          <Card
            key={coin.id}
            id={`coin-${coin.id}`}
            className="border-border/50 bg-card/50 backdrop-blur hover:border-primary/20 transition-all duration-300 scroll-mt-24"
            data-testid={`card-coin-${coin.id}`}
          >
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <Link
                  href={`/coins/${coin.id}`}
                  className="flex items-center gap-3 group"
                  data-testid={`link-coin-detail-${coin.id}`}
                >
                  <div className="w-10 h-10 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center font-display font-bold text-sm text-primary group-hover:border-primary/50 transition-colors">
                    {coin.symbol.slice(0, 2)}
                  </div>
                  <div>
                    <CardTitle className="text-base group-hover:text-primary transition-colors">{coin.name}</CardTitle>
                    <div className="text-xs font-mono text-muted-foreground">{coin.symbol}</div>
                  </div>
                </Link>
                <Badge
                  variant="outline"
                  className={cn(
                    "font-mono text-xs flex items-center gap-1",
                    coin.priceChange24h > 0 ? "border-emerald-400/30 text-emerald-400" :
                    coin.priceChange24h < 0 ? "border-red-400/30 text-red-400" :
                    "border-yellow-400/30 text-yellow-400"
                  )}
                >
                  {changeIcon(coin.priceChange24h)}
                  {coin.priceChange24h > 0 ? "+" : ""}{coin.priceChange24h.toFixed(2)}%
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-2xl font-display font-bold" data-testid={`text-price-${coin.id}`}>
                {formatCurrency(coin.currentPrice)}
              </div>

              <CoinPriceChart coinId={coin.id} />

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-2 rounded bg-background/50 border border-border/20">
                  <div className="text-xs font-mono text-muted-foreground uppercase">Volume 24h</div>
                  <div className="text-sm font-mono font-medium mt-0.5">{formatCompactCurrency(coin.volume24h)}</div>
                </div>
                <div className="p-2 rounded bg-background/50 border border-border/20">
                  <div className="text-xs font-mono text-muted-foreground uppercase">Market Cap</div>
                  <div className="text-sm font-mono font-medium mt-0.5">{formatCompactCurrency(coin.marketCap)}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
