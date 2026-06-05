/**
 * Task #512 — Family Fleet card cluster.
 *
 * Renders one card per executor profile (Momentum Core, Mean Reversion
 * Core, Breakout Core, Volatility Defensive). Each card shows the
 * family's combined equity, realized P&L, win rate, open positions and
 * a clickable link into the per-coin drill-down at
 * `/agents/families/:profileId`. Replaces the legacy
 * 15-personality "Bots Leaderboard" — operators no longer need to
 * scroll a leaderboard, just look at 4 strategy outcomes.
 */

import { Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  useFamilies,
  useBrainRuntimeStatus,
  type FamilyCard,
  type BrainRuntimeStatus,
} from "@/hooks/use-news";
import { Trophy, ChevronRight, Crown, Activity, Shield, Zap, PowerOff } from "lucide-react";

const FAMILY_ICONS: Record<string, typeof Crown> = {
  momentum_core: Crown,
  mean_reversion_core: Activity,
  breakout_core: Zap,
  volatility_defensive: Shield,
};

const STATUS_PILL_STYLES: Record<FamilyCard["statusPill"], string> = {
  active: "border-emerald-400/40 text-emerald-300 bg-emerald-400/10",
  cautious: "border-amber-400/40 text-amber-300 bg-amber-400/10",
  suppressed: "border-red-500/40 text-red-300 bg-red-500/10",
  quarantined: "border-purple-500/40 text-purple-300 bg-purple-500/10",
};

// Task #518 — when the brain is offline, every family is by definition
// not making decisions, so the operational `statusPill` ("active",
// "cautious", etc.) is misleading on these cards. Override every card to
// a "BRAIN OFFLINE" pill (red for offline_no_model, amber for
// offline_disabled) so a glance at the fleet immediately matches the
// truth that the runtime banner is also showing at the top of the page.
type BrainOverridePill = {
  label: string;
  className: string;
} | null;

function brainOverridePill(status: BrainRuntimeStatus | undefined): BrainOverridePill {
  if (!status || status.state === "online") return null;
  if (status.state === "offline_no_model") {
    return {
      label: "brain offline · no model",
      className: "border-red-500/50 text-red-300 bg-red-500/15",
    };
  }
  return {
    label: "brain offline · disabled",
    className: "border-amber-500/50 text-amber-300 bg-amber-500/15",
  };
}

function FamilyCardView({
  family,
  brainPill,
}: {
  family: FamilyCard;
  brainPill: BrainOverridePill;
}) {
  const Icon = FAMILY_ICONS[family.profileId] ?? Trophy;
  const profitable = family.realizedPnl >= 0;
  return (
    <Link
      href={`/agents/families/${family.profileId}`}
      className={cn(
        "block rounded-xl border p-4 transition-colors hover:bg-background/40",
        profitable
          ? "bg-emerald-400/5 border-emerald-400/25"
          : "bg-red-400/5 border-red-400/25",
      )}
      data-testid={`family-card-${family.profileId}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <span
            className={cn(
              "w-9 h-9 rounded-lg flex items-center justify-center shrink-0",
              profitable
                ? "bg-emerald-400/15 text-emerald-300"
                : "bg-red-400/15 text-red-300",
            )}
          >
            <Icon className="w-5 h-5" />
          </span>
          <div className="min-w-0">
            <div className="font-display font-semibold text-base truncate">
              {family.displayName}
            </div>
            <div className="text-[10px] uppercase font-mono tracking-wider text-muted-foreground">
              {family.strategyFamily}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {brainPill ? (
            <Badge
              variant="outline"
              className={cn(
                "text-[9px] uppercase tracking-wider font-mono inline-flex items-center gap-1",
                brainPill.className,
              )}
              data-testid={`family-status-${family.profileId}`}
              title="Brain runtime is offline — see the banner above for details"
            >
              <PowerOff className="w-3 h-3" />
              {brainPill.label}
            </Badge>
          ) : (
            <Badge
              variant="outline"
              className={cn(
                "text-[9px] uppercase tracking-wider font-mono",
                STATUS_PILL_STYLES[family.statusPill],
              )}
              data-testid={`family-status-${family.profileId}`}
            >
              {family.statusPill}
            </Badge>
          )}
          <ChevronRight className="w-4 h-4 text-muted-foreground" />
        </div>
      </div>

      <p className="text-xs text-muted-foreground mt-2 line-clamp-2">
        {family.thesis}
      </p>

      <div className="grid grid-cols-2 gap-3 mt-3">
        <div className="rounded-md border border-border/30 p-2">
          <div className="text-[10px] uppercase font-mono text-muted-foreground">
            Equity
          </div>
          <div className="text-lg font-mono mt-0.5">
            ${family.equity.toFixed(0)}
          </div>
        </div>
        <div className="rounded-md border border-border/30 p-2">
          <div className="text-[10px] uppercase font-mono text-muted-foreground">
            Net P&amp;L
          </div>
          <div
            className={cn(
              "text-lg font-mono mt-0.5",
              profitable ? "text-emerald-300" : "text-red-300",
            )}
            data-testid={`family-pnl-${family.profileId}`}
          >
            {profitable ? "+" : ""}${family.realizedPnl.toFixed(2)}
            <span className="ml-1 text-[10px] text-muted-foreground">
              ({profitable ? "+" : ""}{family.realizedPnlPct.toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mt-3 text-[10px] font-mono">
        <div className="rounded border border-border/20 p-1.5">
          <div className="uppercase text-muted-foreground">Max DD</div>
          <div className="text-amber-300/90 mt-0.5">
            {family.maxDrawdown.toFixed(2)}%
          </div>
        </div>
        <div className="rounded border border-border/20 p-1.5">
          <div className="uppercase text-muted-foreground">CADA</div>
          <div className="text-foreground/90 mt-0.5">
            {family.costAwareDirectionalAccuracy != null
              ? `${(family.costAwareDirectionalAccuracy * 100).toFixed(0)}%`
              : "—"}
          </div>
        </div>
        <div className="rounded border border-border/20 p-1.5">
          <div className="uppercase text-muted-foreground">Abstain</div>
          <div className="text-foreground/90 mt-0.5">
            {family.abstainRate.toFixed(0)}%
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between mt-3 text-[11px] font-mono text-muted-foreground">
        <span>
          {family.totalTrades} trades · {family.winRate.toFixed(0)}% win · trust{" "}
          <span
            className={cn(
              family.trustMultiplier >= 1
                ? "text-emerald-300"
                : "text-amber-300",
            )}
            data-testid={`family-trust-${family.profileId}`}
          >
            {family.trustMultiplier.toFixed(2)}x
          </span>
        </span>
        <span>{family.openPositions} open</span>
      </div>

      {family.retirement && family.retirement.auto_flipped && (
        <div className="mt-2 flex items-center gap-1.5 flex-wrap">
          <Badge
            variant="outline"
            className="text-[9px] uppercase tracking-wider border-amber-500/40 text-amber-300"
          >
            auto-flipped
          </Badge>
          {family.retirement.triggered_by?.map((t) => (
            <Badge
              key={t}
              variant="outline"
              className="text-[9px] font-mono border-border/40 text-muted-foreground"
            >
              {t}
            </Badge>
          ))}
        </div>
      )}
    </Link>
  );
}

export function FamilyFleet() {
  const { data, isLoading, error } = useFamilies();
  const families = data?.families ?? [];
  // Task #518 — pull brain runtime status so each card can override its
  // operational pill with "BRAIN OFFLINE" when the brain isn't deciding.
  const { data: brainStatus } = useBrainRuntimeStatus();
  const brainPill = brainOverridePill(brainStatus);

  return (
    <Card className="bg-card/50 border-border/40" data-testid="family-fleet-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Trophy className="w-4 h-4" />
          Executor Families
          <span className="text-[10px] font-normal text-muted-foreground normal-case tracking-normal">
            (4 deterministic strategies — click for per-coin breakdown)
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-44 w-full" />
            ))}
          </div>
        )}
        {error && (
          <div className="text-sm text-red-400 font-mono" data-testid="family-fleet-error">
            Could not load executor families: {String(error)}
          </div>
        )}
        {!isLoading && !error && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {families.map((f) => (
              <FamilyCardView key={f.profileId} family={f} brainPill={brainPill} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
