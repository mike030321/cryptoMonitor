import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { CircleSlash, ArrowRight } from "lucide-react";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface QuantAbstainReasons {
  hours: number;
  generatedAt: string;
  totalAbstains: number;
  uniquePairs: number;
  byReason: Array<{ reason: string; count: number }>;
  byPair: Array<{
    coinId: string;
    coinName: string | null;
    timeframe: string;
    total: number;
    topReason: string;
    reasons: Record<string, number>;
  }>;
}

const REASON_LABELS: Record<string, string> = {
  no_model: "No model trained for this pair",
  model_unavailable: "Model temporarily unavailable",
  ml_engine_down: "ML engine unreachable",
  feature_vector_missing: "Feature vector unavailable",
  insufficient_history: "Not enough history",
  low_confidence: "Confidence below abstain floor",
};

function reasonLabel(reason: string): string {
  return REASON_LABELS[reason] ?? reason.replace(/_/g, " ");
}

const MAX_PAIRS_VISIBLE = 6;

export function QuantAbstainReasonsCard() {
  const { data, isLoading, isError } = useQuery<QuantAbstainReasons>({
    queryKey: ["quant-abstain-reasons", 24],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/quant-abstain-reasons?hours=24`);
      if (!res.ok) throw new Error(`quant-abstain-reasons ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  return (
    <Card className="bg-card/50 border-border/40" data-testid="quant-abstain-reasons-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <CircleSlash className="w-4 h-4" />
          Abstain reasons (last 24h)
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            why the bot skipped trading — usually no model for that pair
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-32 w-full" />}
        {isError && (
          <div
            className="text-sm text-rose-300 font-mono"
            data-testid="quant-abstain-reasons-error"
          >
            Couldn't load abstain reasons.
          </div>
        )}
        {data && data.totalAbstains === 0 && (
          <div
            className="text-sm text-muted-foreground font-mono"
            data-testid="quant-abstain-reasons-empty"
          >
            No quant-brain abstains in the last 24 hours — every (coin, timeframe) the bot looked at had a usable model.
          </div>
        )}
        {data && data.totalAbstains > 0 && (
          <div className="space-y-4">
            <div className="flex items-end justify-between gap-4 flex-wrap">
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground">
                  Abstain events
                </div>
                <div
                  className="text-3xl font-display font-bold mt-1"
                  data-testid="quant-abstain-reasons-total"
                >
                  {data.totalAbstains}
                </div>
                <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
                  across {data.uniquePairs} (coin, timeframe) pair{data.uniquePairs === 1 ? "" : "s"}
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {data.byReason.map((r) => (
                  <Badge
                    key={r.reason}
                    variant="outline"
                    className="font-mono text-[11px] bg-amber-500/10 text-amber-200 border-amber-500/30"
                    data-testid={`quant-abstain-reason-chip-${r.reason}`}
                  >
                    {reasonLabel(r.reason)} · {r.count}
                  </Badge>
                ))}
              </div>
            </div>

            <div className="space-y-1.5" data-testid="quant-abstain-pair-list">
              {data.byPair.slice(0, MAX_PAIRS_VISIBLE).map((p) => {
                const symbol = p.coinName ?? p.coinId;
                const testId = `quant-abstain-pair-${p.coinId}-${p.timeframe}`;
                return (
                  <div
                    key={`${p.coinId}-${p.timeframe}`}
                    className="flex items-center justify-between gap-3 p-2 rounded-md bg-background/40 ring-1 ring-border/30"
                    data-testid={testId}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="font-mono text-sm truncate">
                        <span className="font-semibold">{symbol}</span>
                        <span className="text-muted-foreground"> · {p.timeframe}</span>
                      </div>
                      <div className="text-[11px] text-muted-foreground font-mono mt-0.5 truncate">
                        {p.total} abstain{p.total === 1 ? "" : "s"} · top reason: {reasonLabel(p.topReason)}
                      </div>
                    </div>
                    <Link href="/diagnostics#quant-coverage">
                      <Button
                        size="sm"
                        variant="outline"
                        className="rounded-full text-[11px] h-7 px-3 border-sky-500/40 text-sky-200 hover:bg-sky-500/10"
                        data-testid={`${testId}-train-link`}
                      >
                        Coverage
                        <ArrowRight className="w-3 h-3 ml-1" />
                      </Button>
                    </Link>
                  </div>
                );
              })}
              {data.byPair.length > MAX_PAIRS_VISIBLE && (
                <div className="text-[11px] text-muted-foreground font-mono pt-1">
                  +{data.byPair.length - MAX_PAIRS_VISIBLE} more pair{data.byPair.length - MAX_PAIRS_VISIBLE === 1 ? "" : "s"} not shown.
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
