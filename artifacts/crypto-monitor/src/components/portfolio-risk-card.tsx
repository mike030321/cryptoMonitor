import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { ShieldCheck, AlertTriangle, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatCurrency } from "@/lib/format";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface PortfolioRiskPosition {
  coinId: string;
  coinName: string;
  agentName: string;
  direction: string;
  notionalUsd: number;
  regimeAtEntry: string | null;
}

interface PortfolioRiskSector {
  sector: string;
  notionalUsd: number;
  sharePct: number;
  cap: number;
  distinctCoins: number;
  positions: PortfolioRiskPosition[];
}

interface PortfolioRiskRegime {
  regime: string;
  notionalUsd: number;
  sharePct: number;
  cap: number;
  positions: PortfolioRiskPosition[];
}

interface PortfolioRiskResponse {
  enabled: boolean;
  caps: {
    maxSectorExposurePct: number | null;
    maxCorrelatedExposurePct: number | null;
    maxBetaToBtc: number | null;
    regimeBudgetPct: number | null;
  };
  equityUsd: number;
  totalNotionalUsd: number;
  bookBeta: number | null;
  bookBetaSource: string;
  openPositionCount: number;
  sectors: PortfolioRiskSector[];
  regimes: PortfolioRiskRegime[];
  fetchedAt: string;
}

function pct(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

function severity(share: number, cap: number): "ok" | "warn" | "over" {
  if (cap <= 0) return "ok";
  const r = share / cap;
  if (r >= 1) return "over";
  if (r >= 0.8) return "warn";
  return "ok";
}

function barColors(sev: "ok" | "warn" | "over"): { bar: string; text: string } {
  switch (sev) {
    case "over":
      return { bar: "bg-rose-400", text: "text-rose-300" };
    case "warn":
      return { bar: "bg-amber-400", text: "text-amber-300" };
    default:
      return { bar: "bg-emerald-400", text: "text-emerald-300" };
  }
}

function PositionList({ positions }: { positions: PortfolioRiskPosition[] }) {
  if (positions.length === 0) {
    return <div className="text-xs text-muted-foreground font-mono">no open positions</div>;
  }
  return (
    <div className="space-y-1">
      {positions.map((p, i) => {
        const isUp = p.direction === "up" || p.direction === "long";
        return (
          <div
            key={`${p.agentName}-${p.coinId}-${i}`}
            className="flex items-center justify-between gap-3 text-[11px] font-mono"
            data-testid={`portfolio-risk-position-${p.coinId}-${i}`}
          >
            <div className="flex items-center gap-1.5 min-w-0">
              {isUp ? (
                <ArrowUpRight className="w-3 h-3 text-emerald-400 shrink-0" />
              ) : (
                <ArrowDownRight className="w-3 h-3 text-rose-400 shrink-0" />
              )}
              <span className="truncate">{p.coinName}</span>
              <span className="text-muted-foreground/70 truncate">· {p.agentName}</span>
            </div>
            <span className="text-foreground/80 shrink-0">{formatCurrency(p.notionalUsd)}</span>
          </div>
        );
      })}
    </div>
  );
}

interface ExposureRowProps {
  label: string;
  sublabel?: string;
  share: number;
  cap: number;
  notionalUsd: number;
  positions: PortfolioRiskPosition[];
  testId: string;
}

function ExposureRow({ label, sublabel, share, cap, notionalUsd, positions, testId }: ExposureRowProps) {
  const sev = severity(share, cap);
  const { bar, text } = barColors(sev);
  const widthPct = Math.min(100, Math.max(0, (share / Math.max(cap, 1e-9)) * 100));
  return (
    <HoverCard openDelay={120} closeDelay={80}>
      <HoverCardTrigger asChild>
        <div
          className="cursor-default group"
          data-testid={testId}
        >
          <div className="flex items-baseline justify-between gap-2 mb-1">
            <div className="flex items-baseline gap-2 min-w-0">
              <span className="text-xs font-mono uppercase tracking-wider truncate">{label}</span>
              {sublabel && (
                <span className="text-[10px] text-muted-foreground/70 font-mono">{sublabel}</span>
              )}
            </div>
            <div className={cn("text-xs font-mono shrink-0", text)}>
              {pct(share)} <span className="text-muted-foreground/60">/ {pct(cap)} cap</span>
            </div>
          </div>
          <div className="h-1.5 w-full rounded-full bg-muted/30 overflow-hidden">
            <div
              className={cn("h-full transition-all", bar)}
              style={{ width: `${widthPct}%` }}
            />
          </div>
          <div className="text-[10px] font-mono text-muted-foreground/60 mt-0.5">
            {formatCurrency(notionalUsd)} notional · {positions.length}{" "}
            {positions.length === 1 ? "position" : "positions"} · hover to inspect
          </div>
        </div>
      </HoverCardTrigger>
      <HoverCardContent className="w-72" align="end">
        <div className="text-[10px] uppercase font-mono text-muted-foreground mb-2">
          {label} · open positions
        </div>
        <PositionList positions={positions} />
      </HoverCardContent>
    </HoverCard>
  );
}

export function PortfolioRiskCard() {
  const { data, isLoading, isError } = useQuery<PortfolioRiskResponse>({
    queryKey: ["portfolio-risk"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/portfolio-risk`);
      if (!res.ok) throw new Error(`portfolio-risk ${res.status}`);
      return res.json();
    },
    refetchInterval: 15_000,
    staleTime: 10_000,
  });

  return (
    <Card className="bg-card/50 border-border/40" data-testid="portfolio-risk-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <ShieldCheck className="w-4 h-4" />
          Portfolio Risk
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            live exposure vs the Phase 5 sector / beta / regime caps · refreshes every 15s
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-32 w-full" />}
        {isError && (
          <div className="text-sm text-rose-300 font-mono" data-testid="portfolio-risk-error">
            Couldn't load portfolio risk snapshot.
          </div>
        )}
        {data && (
          <div className="space-y-4">
            {!data.enabled && (
              <div
                className="flex items-start gap-2 p-2 rounded-md bg-amber-500/10 ring-1 ring-amber-500/30 text-amber-200 text-xs"
                data-testid="portfolio-risk-disabled"
              >
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <div>
                  Portfolio constraints are <span className="font-semibold">disabled</span> in
                  <code className="mx-1">trading-frictions.json</code>. The numbers below are
                  informational only — no trades are being blocked.
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3" data-testid="portfolio-risk-summary">
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground">Fleet equity</div>
                <div className="text-base font-display font-bold mt-0.5">
                  {formatCurrency(data.equityUsd)}
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground">Open notional</div>
                <div className="text-base font-display font-bold mt-0.5">
                  {formatCurrency(data.totalNotionalUsd)}
                </div>
                <div className="text-[10px] font-mono text-muted-foreground/70">
                  {data.openPositionCount}{" "}
                  {data.openPositionCount === 1 ? "position" : "positions"}
                </div>
              </div>
              <div data-testid="portfolio-risk-book-beta">
                <div className="text-[10px] uppercase font-mono text-muted-foreground">
                  Book β vs BTC
                </div>
                <div
                  className={cn(
                    "text-base font-display font-bold mt-0.5",
                    data.bookBeta !== null && data.caps.maxBetaToBtc !== null && data.bookBeta > data.caps.maxBetaToBtc
                      ? "text-rose-300"
                      : "text-foreground",
                  )}
                >
                  {data.bookBeta !== null ? data.bookBeta.toFixed(2) : "—"}
                  <span className="text-xs text-muted-foreground/60 font-mono ml-1">
                    / {data.caps.maxBetaToBtc?.toFixed(2) ?? "—"}
                  </span>
                </div>
                <div className="text-[10px] font-mono text-muted-foreground/70">
                  default β=1.0 (per-coin β not tracked)
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground">Sector cap</div>
                <div className="text-base font-display font-bold mt-0.5">
                  {pct(data.caps.maxSectorExposurePct, 0)}
                </div>
                <div className="text-[10px] font-mono text-muted-foreground/70">
                  correlated cap {pct(data.caps.maxCorrelatedExposurePct, 0)}
                </div>
              </div>
            </div>

            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground mb-2">
                By sector
              </div>
              {data.sectors.length === 0 ? (
                <div className="text-xs text-muted-foreground font-mono" data-testid="portfolio-risk-no-sectors">
                  no open positions
                </div>
              ) : (
                <div className="space-y-3" data-testid="portfolio-risk-sectors">
                  {data.sectors.map((s) => (
                    <ExposureRow
                      key={s.sector}
                      label={s.sector}
                      sublabel={`${s.distinctCoins} ${s.distinctCoins === 1 ? "coin" : "coins"}`}
                      share={s.sharePct}
                      cap={s.cap}
                      notionalUsd={s.notionalUsd}
                      positions={s.positions}
                      testId={`portfolio-risk-sector-${s.sector}`}
                    />
                  ))}
                </div>
              )}
            </div>

            {data.regimes.length > 0 && (
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground mb-2">
                  By regime budget
                </div>
                <div className="space-y-3" data-testid="portfolio-risk-regimes">
                  {data.regimes.map((r) => (
                    <ExposureRow
                      key={r.regime}
                      label={r.regime}
                      share={r.sharePct}
                      cap={r.cap}
                      notionalUsd={r.notionalUsd}
                      positions={r.positions}
                      testId={`portfolio-risk-regime-${r.regime}`}
                    />
                  ))}
                </div>
              </div>
            )}

            <div className="text-[10px] font-mono text-muted-foreground/60">
              refreshes every 15s · last {new Date(data.fetchedAt).toLocaleTimeString()}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
