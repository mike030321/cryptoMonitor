import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Flame, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface SourceRow {
  source: string;
  currentUsd: number;
  sharePct: number;
  lastSeenAt: string;
  silent: boolean;
  silentMinutes: number;
}

interface CoinRow {
  coinId: string;
  totalUsd: number;
  latestTimestamp: string;
  sources: SourceRow[];
}

interface LiquidationsBreakdown {
  generatedAt: string;
  silentThresholdMinutes: number;
  lookbackHours: number;
  coins: CoinRow[];
}

const SOURCE_LABEL: Record<string, string> = {
  okx: "OKX",
  gate: "Gate",
  binance: "Binance",
  bybit: "Bybit",
};

function labelFor(source: string): string {
  return SOURCE_LABEL[source] ?? source.toUpperCase();
}

function formatUsd(usd: number): string {
  if (usd >= 1_000_000) return `$${(usd / 1_000_000).toFixed(2)}M`;
  if (usd >= 1_000) return `$${(usd / 1_000).toFixed(1)}k`;
  return `$${usd.toFixed(0)}`;
}

function formatSilent(min: number): string {
  if (min < 60) return `${min}m`;
  const h = Math.floor(min / 60);
  const m = min % 60;
  return m > 0 ? `${h}h${m}m` : `${h}h`;
}

export function LiquidationsBreakdownCard() {
  const { data, isLoading, isError } = useQuery<LiquidationsBreakdown>({
    queryKey: ["liquidations-breakdown"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/liquidations-breakdown`);
      if (!res.ok) throw new Error(`liquidations-breakdown ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  const silentCount =
    data?.coins.reduce(
      (acc, c) => acc + c.sources.filter((s) => s.silent).length,
      0,
    ) ?? 0;
  const anySilent = silentCount > 0;

  return (
    <Card
      className={cn(
        "border-border/40",
        anySilent ? "bg-amber-500/5 ring-1 ring-amber-500/30" : "bg-card/30",
      )}
      data-testid="liquidations-breakdown-card"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Flame className="w-4 h-4" />
          Liquidations by exchange
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            trailing 1h · per-source split
          </span>
          {anySilent && (
            <span
              className="ml-auto inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/40"
              data-testid="liquidations-breakdown-silent-summary"
            >
              <AlertTriangle className="w-3 h-3" />
              {silentCount} silent
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-24 w-full" />}
        {isError && (
          <div
            className="text-sm text-rose-300 font-mono"
            data-testid="liquidations-breakdown-error"
          >
            Couldn't load liquidation breakdown.
          </div>
        )}
        {data && data.coins.length === 0 && (
          <div
            className="text-xs font-mono text-muted-foreground"
            data-testid="liquidations-breakdown-empty"
          >
            No liquidation breakdown data in the last {data.lookbackHours}h.
          </div>
        )}
        {data && data.coins.length > 0 && (
          <div
            className="space-y-2"
            data-testid="liquidations-breakdown-list"
          >
            {data.coins.map((coin) => (
              <div
                key={coin.coinId}
                className="p-2 rounded-md bg-card/40 border border-border/40"
                data-testid={`liquidations-breakdown-coin-${coin.coinId}`}
              >
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
                      {coin.coinId}
                    </span>
                    <span
                      className="text-sm font-display font-bold tabular-nums"
                      data-testid={`liquidations-breakdown-total-${coin.coinId}`}
                    >
                      {formatUsd(coin.totalUsd)}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {coin.sources.map((s) => (
                      <span
                        key={s.source}
                        className={cn(
                          "inline-flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded",
                          s.silent
                            ? "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/40"
                            : "bg-card/60 text-foreground/80 ring-1 ring-border/40",
                        )}
                        data-testid={`liquidations-breakdown-source-${coin.coinId}-${s.source}`}
                        title={
                          s.silent
                            ? `${labelFor(s.source)} silent for ${formatSilent(
                                s.silentMinutes,
                              )} (last contributed ${new Date(
                                s.lastSeenAt,
                              ).toLocaleTimeString()})`
                            : `${labelFor(s.source)} contributed ${formatUsd(
                                s.currentUsd,
                              )}`
                        }
                      >
                        {s.silent && <AlertTriangle className="w-3 h-3" />}
                        <span>{labelFor(s.source)}</span>
                        {s.silent ? (
                          <span className="tabular-nums">
                            silent {formatSilent(s.silentMinutes)}
                          </span>
                        ) : (
                          <span className="tabular-nums">
                            {s.sharePct.toFixed(0)}%
                          </span>
                        )}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
            <div className="text-[10px] font-mono text-muted-foreground/70 pt-1">
              warning when an exchange that was contributing has been silent for &gt;{" "}
              {data.silentThresholdMinutes}m
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
