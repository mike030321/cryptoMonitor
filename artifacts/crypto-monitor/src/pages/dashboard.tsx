import { useGetDashboard, useGetMonitoringStatus, useGetDiagnosticSandboxStatus, getGetDashboardQueryKey, getGetMonitoringStatusQueryKey, getGetDiagnosticSandboxStatusQueryKey } from "@workspace/api-client-react";
import { SkipTrackerHealthCard } from "@/components/skip-tracker-health-card";
import { FiveMTopupCard } from "@/components/five-m-topup-card";
import { MarketSignalsHealthCard } from "@/components/market-signals-health-card";
import { FiveMTopupStatusCard } from "@/components/five-m-topup-status-card";
import { Topup5mBanner } from "@/components/topup-5m-banner";
import { DisabledOutcomeBanner } from "@/components/disabled-outcome-banner";
import { LiquidationsBreakdownCard } from "@/components/liquidations-breakdown-card";
import { QuantAbstainReasonsCard } from "@/components/quant-abstain-reasons-card";
import { PortfolioRiskCard } from "@/components/portfolio-risk-card";
import { useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { formatCurrency, formatTimeAgo } from "@/lib/format";
import { derivePnl } from "@/lib/derive-pnl";
import { Activity, DollarSign, TrendingUp, TrendingDown, Minus, Zap, RotateCcw, Crown, Trophy, ChevronDown, Stethoscope } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { useBestPick, usePaperPortfolios, useBrainRuntimeStatus, type PaperPortfolio } from "@/hooks/use-news";
// Task #512 — executor-fleet rework: 4 family cards + benchmarks panel +
// activity banner replace the legacy 15-personality leaderboard.
// Task #518 — brain-state banner so an offline/abstaining brain cannot
// hide behind a "Live" StatusPill.
import { FamilyFleet } from "@/components/family-fleet";
import { BenchmarksPanel } from "@/components/benchmarks-panel";
import { ActivityBanner } from "@/components/activity-banner";
import { BrainStateBanner } from "@/components/brain-state-banner";
// Task #614 — MTTM lane banner. Renders nothing when MTTM is off so
// the dashboard chrome is unchanged in the default state.
import { MttmBanner } from "@/components/mttm-banner";
import { DiagnosticSandboxBanner } from "@/components/diagnostic-sandbox-banner";
// Task #551 — per-timeframe role card. Renders directly under the
// brain-state banner so the dashboard can no longer imply that every
// horizon is `trade`-eligible.
import { HorizonRolesCard } from "@/components/horizon-roles-card";
// Task #554 — dataset-refresher health surfaces. The sticky banner
// only renders when the refresher is unhealthy; the card is always
// visible and shows per-tf freshness pills + an expandable failure
// log (the panel the banner deep-links to).
import {
  DatasetFreshnessBanner,
  DatasetFreshnessCard,
} from "@/components/dataset-freshness-card";
// Task #599 — surfaces the latest 1h/2h ON/OFF calibration verdict
// the dataset-refresher auto-runs after each short-tf snapshot.
import { ShortTfCalibrationVerdictCard } from "@/components/short-tf-calibration-verdict-card";
import { PriorFallbackBadge } from "@/components/prior-fallback-badge";
import { useQuery } from "@tanstack/react-query";
import { useAdminKey } from "@/hooks/use-admin-key";
import { AdminKeyField } from "@/components/admin-key-field";
// Task #444 — `LlmSidecarPanel` deleted with the rest of the LLM plane.
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Link } from "wouter";

function ResetBotsButton() {
  const queryClient = useQueryClient();
  const { adminFetch, hasKey, ensureKey } = useAdminKey({
    keyName: "ADMIN_RESET_KEY",
    onRejected: () => window.alert("Reset key was rejected. Fix the typo in the field above and try again."),
  });
  const [isResetting, setIsResetting] = useState(false);

  const handleReset = async () => {
    if (!hasKey) {
      ensureKey("authorize this reset");
      window.alert("Paste your reset key in the field at the top, then click Reset Bots again.");
      return;
    }
    const phrase = window.prompt(
      "This wipes ALL bots back to $1,000 and deletes every trade and prediction.\n\nType RESET BOTS to confirm:",
    );
    if (phrase !== "RESET BOTS") {
      if (phrase !== null) window.alert("Reset cancelled — phrase did not match.");
      return;
    }
    setIsResetting(true);
    try {
      const res = await adminFetch(`${import.meta.env.BASE_URL}api/crypto/admin/reset-bots`, {
        action: "authorize this reset",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: "RESET BOTS" }),
      });
      if (!res) return;
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Reset failed");
      window.alert(data.message || "Reset complete.");
      await queryClient.invalidateQueries();
    } catch (err) {
      window.alert(`Reset failed: ${String(err)}`);
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <Button
      onClick={handleReset}
      disabled={isResetting}
      variant="outline"
      className="rounded-full px-4 border-rose-500/40 text-rose-300 hover:bg-rose-500/10 hover:text-rose-200"
      data-testid="button-reset-bots"
    >
      <RotateCcw className="w-4 h-4 mr-2" />
      {isResetting ? "Resetting…" : "Reset Bots"}
    </Button>
  );
}

function ForceAnalysisButton() {
  const queryClient = useQueryClient();
  const { adminFetch, hasKey, ensureKey } = useAdminKey({
    onRejected: () => window.alert("Admin key was rejected. Fix the typo in the field above and try again."),
  });
  const [isTriggering, setIsTriggering] = useState(false);

  const handleTrigger = async () => {
    if (!hasKey) {
      ensureKey("trigger an analysis");
      return;
    }
    setIsTriggering(true);
    try {
      const res = await adminFetch(`${import.meta.env.BASE_URL}api/crypto/trigger-analysis`, {
        action: "trigger an analysis",
        method: "POST",
      });
      if (res?.ok) await queryClient.invalidateQueries();
    } finally {
      setIsTriggering(false);
    }
  };

  return (
    <Button
      onClick={handleTrigger}
      disabled={isTriggering}
      className="rounded-full px-5 gradient-bg-primary text-white border-0 hover:opacity-90 shadow-lg shadow-primary/30"
      data-testid="button-trigger-analysis"
    >
      <Zap className="w-4 h-4 mr-2" />
      {isTriggering ? "Analyzing…" : "Run Analysis Now"}
    </Button>
  );
}

function StatusPill({ status }: { status: { isRunning?: boolean; cycleCount?: number; nextCycleAt?: string | null } | undefined }) {
  if (!status?.isRunning) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/20 text-xs font-medium">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
        Warming up
      </span>
    );
  }
  const nextLabel = (() => {
    if (!status.nextCycleAt) return "scheduling…";
    const ms = new Date(status.nextCycleAt).getTime() - Date.now();
    if (ms > 0) return `next in ${Math.ceil(ms / 1000)}s`;
    return "running now";
  })();
  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20 font-medium">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 live-dot" />
        Live
      </span>
      <span>Cycle #{status.cycleCount ?? 0}</span>
      <span className="text-muted-foreground/50">•</span>
      <span>{nextLabel}</span>
    </span>
  );
}

interface RealityCheckLite {
  fleet: {
    startingCapital: number;
    netPnlUsd: number;
    grossPnlUsd: number;
    avgNetReturnPct: number;
    totalClosedTrades: number;
  };
  priorFallback?: { included: boolean; tradesExcluded: number; tradesTotal?: number };
}

// Card removed — merged into QuantFleetCard below. The prior split between
// "Fleet Performance" (mark-to-market) and "Quant Brain Realized P&L"
// (closed only) was confusing operators because both cards now describe the
// same fleet. The unified card surfaces the realized vs unrealized split
// directly so the gap is self-explanatory.

interface MeasurementModeState {
  enabled: boolean;
  source: "default" | "env" | "manual";
  lastChangedAt: string;
}

function useMeasurementMode() {
  return useQuery<MeasurementModeState>({
    queryKey: ["measurement-mode"],
    queryFn: async () => {
      const res = await fetch(`${import.meta.env.BASE_URL}api/crypto/measurement-mode`, { cache: "no-store" });
      if (!res.ok) throw new Error(`measurement-mode ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}

function QuantFleetCard({ bots }: { bots: PaperPortfolio[] }) {
  const [includePrior, setIncludePrior] = useState(false);
  const { data: measurementMode } = useMeasurementMode();
  const measurementActive = measurementMode?.enabled ?? null;
  // Task #518 — brain runtime status drives a SEPARATE pill from the
  // operator-toggle measurement pill. Measurement=ON means auto-deploy is
  // gated off; it says nothing about whether the brain is currently authoring
  // decisions. Showing both pills makes the two states orthogonal and
  // unambiguous.
  const { data: brainStatus } = useBrainRuntimeStatus();
  const brainState = brainStatus?.state ?? null;

  // Task #659 (C-BTC) — diagnostic sandbox pill. Polls the OpenAPI-
  // generated hook so the FE and the server share one schema.
  const { data: dsStatus } = useGetDiagnosticSandboxStatus({
    query: {
      queryKey: getGetDiagnosticSandboxStatusQueryKey(),
      refetchInterval: 30_000,
      staleTime: 25_000,
    },
  });
  const dsActive = dsStatus?.mode === "diagnostic_sandbox";
  const dsAutoDisabled = !!dsStatus?.auto_disable_status?.disabled;
  const dsAutoDisabledReason = dsStatus?.auto_disable_status?.reason ?? null;

  // Mark-to-market view from per-bot portfolios (cash + open positions
  // valued at last known price). The fleet aggregates `totalValue` and
  // each bot's `startingCapital` (Task #362). Per-bot rendering in
  // BotLeaderboard derives net P&L the same way (`totalValue -
  // startingCapital`) so the fleet card and the per-bot rows are
  // guaranteed to agree by construction. `winners` counts bots whose
  // equity exceeds their seed — the prior `totalPnl > 0` definition was
  // realized-only and disagreed with the visible per-bot rows.
  const start = bots.reduce((s, p) => s + (p.startingCapital ?? 1000), 0);
  const total = bots.reduce((s, p) => s + p.totalValue, 0);
  const cash = bots.reduce((s, p) => s + p.cashBalance, 0);
  const openMarketValue = total - cash;
  const totalChange = total - start;
  const totalChangePct = start > 0 ? (totalChange / start) * 100 : 0;
  const trades = bots.reduce((s, p) => s + p.totalTrades, 0);
  const wins = bots.reduce((s, p) => s + p.winningTrades, 0);
  const winRate = trades > 0 ? (wins / trades) * 100 : 0;
  const winners = bots.filter(b => b.totalValue > (b.startingCapital ?? 1000)).length;

  // Realized view from the reality-check endpoint (closed trades, after
  // fees, prior fallback toggle). Unrealized = mark-to-market change minus
  // realized — a derivation rather than a separate query, so it always
  // reconciles by construction.
  const { data: realityData } = useQuery<RealityCheckLite>({
    queryKey: ["crypto-reality-check", includePrior],
    queryFn: async () => {
      const res = await fetch(
        `${import.meta.env.BASE_URL}api/crypto/reality-check${includePrior ? "?includePrior=1" : ""}`,
      );
      if (!res.ok) throw new Error("Failed to load reality check");
      return res.json();
    },
    refetchInterval: 60_000,
  });
  const realized = realityData?.fleet?.netPnlUsd ?? 0;
  const realizedGross = realityData?.fleet?.grossPnlUsd ?? 0;
  const realizedFees = realizedGross - realized;
  const closedTrades = realityData?.fleet?.totalClosedTrades ?? 0;
  const avgNetPct = realityData?.fleet?.avgNetReturnPct ?? 0;
  const unrealized = totalChange - realized;
  const priorTotal =
    realityData?.priorFallback?.tradesTotal ?? realityData?.priorFallback?.tradesExcluded ?? 0;
  const showPriorBadge = priorTotal > 0;

  return (
    <Card className="bg-card/50 border-border/40" data-testid="quant-fleet-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2 flex-wrap">
          <Crown className="w-4 h-4" />
          Quant Fleet P&amp;L
          <span className="text-[10px] font-normal text-muted-foreground normal-case tracking-normal">
            (one source of truth · realized + unrealized = total equity change)
          </span>
          {/* Task #518 — brain runtime pill (left of Measurement) so the
              two orthogonal states never get visually fused. */}
          {brainState === "online" && (
            <Badge
              variant="outline"
              className="ml-auto text-[10px] font-mono uppercase tracking-wider border-emerald-400/40 text-emerald-300 bg-emerald-400/5"
              data-testid="brain-state-pill-online"
              title={
                brainStatus?.lastSuccessfulDecisionAt
                  ? `Brain online · last decision ${new Date(brainStatus.lastSuccessfulDecisionAt).toLocaleTimeString()}`
                  : "Brain online — recent non-abstain predictions exist."
              }
            >
              Brain: Online
            </Badge>
          )}
          {brainState === "offline_no_model" && (
            <Badge
              variant="outline"
              className="ml-auto text-[10px] font-mono uppercase tracking-wider border-red-400/50 text-red-300 bg-red-400/10"
              data-testid="brain-state-pill-offline-no-model"
              title="Quant brain enabled but ml-engine has no trained model — every cycle is abstaining."
            >
              Brain: Offline (no_model)
            </Badge>
          )}
          {brainState === "offline_disabled" && (
            <Badge
              variant="outline"
              className="ml-auto text-[10px] font-mono uppercase tracking-wider border-amber-400/50 text-amber-300 bg-amber-400/10"
              data-testid="brain-state-pill-offline-disabled"
              title="Quant brain kill-switch is OFF (operator flip, env force-off, or auto-revert)."
            >
              Brain: Disabled
            </Badge>
          )}
          {measurementActive === true && (
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] font-mono uppercase tracking-wider border-amber-400/50 text-amber-300 bg-amber-400/10",
                brainState === null && "ml-auto",
              )}
              data-testid="quant-only-status-on"
              title="Measurement Mode is active: auto-deploy is blocked so signal quality can be observed."
            >
              Measurement: ON
            </Badge>
          )}
          {measurementActive === false && (
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] font-mono uppercase tracking-wider border-emerald-400/40 text-emerald-300 bg-emerald-400/5",
                brainState === null && "ml-auto",
              )}
              data-testid="quant-only-status-off"
              title="Measurement Mode is off: real-price auto-deploy may open paper trades."
            >
              Auto-deploy: ON
            </Badge>
          )}
          {/* Task #659 (C-BTC) — Diagnostic Sandbox status pill. Per the
              operator-facing contract this pill renders one of three
              explicit states: "Diagnostic Sandbox: ON" when the lane is
              actively authoring trades, "Diagnostic Sandbox: OFF" when
              the lane exists but is not the active mode, and
              "Diagnostic Sandbox: Auto-disabled(<reason>)" when the
              auto-disable verdict has tripped. The reason is shown
              verbatim so operators can act without opening the banner. */}
          {dsAutoDisabled ? (
            <Badge
              variant="outline"
              className="text-[10px] font-mono uppercase tracking-wider border-red-400/50 text-red-300 bg-red-400/10"
              data-testid="diagnostic-sandbox-pill-disabled"
              title="Diagnostic Paper Sandbox auto-disabled — see banner above for the breach detail."
            >
              Diagnostic Sandbox: Auto-disabled
              {dsAutoDisabledReason ? ` (${dsAutoDisabledReason})` : ""}
            </Badge>
          ) : dsActive ? (
            <Badge
              variant="outline"
              className="text-[10px] font-mono uppercase tracking-wider border-teal-400/40 text-teal-200 bg-teal-400/10"
              data-testid={
                dsStatus?.ready
                  ? "diagnostic-sandbox-pill-on"
                  : "diagnostic-sandbox-pill-pending"
              }
              title={
                dsStatus?.ready
                  ? `Diagnostic Paper Sandbox is the only lane authoring trades · BTC version ${dsStatus.btc_version ?? "unknown"}.`
                  : "Diagnostic Paper Sandbox staged but no calibrated BTC/5m version is promoted yet — lane is read-only."
              }
            >
              Diagnostic Sandbox: {dsStatus?.ready ? "ON" : "ON (pending)"}
            </Badge>
          ) : (
            <Badge
              variant="outline"
              className="text-[10px] font-mono uppercase tracking-wider border-slate-500/40 text-slate-300 bg-slate-500/10"
              data-testid="diagnostic-sandbox-pill-off"
              title="Diagnostic Paper Sandbox is OFF — the default 16-slot MTTM lane is active."
            >
              Diagnostic Sandbox: OFF
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Fleet Equity</div>
            <div className="text-2xl font-display font-bold mt-1">${total.toFixed(2)}</div>
            <div className={cn("text-[11px] font-mono mt-0.5", totalChange >= 0 ? "text-emerald-400/70" : "text-red-400/70")}>
              {totalChange >= 0 ? "+" : ""}${totalChange.toFixed(2)} ({totalChangePct >= 0 ? "+" : ""}{totalChangePct.toFixed(2)}%) vs ${start.toLocaleString()} start
            </div>
          </div>
          <div className="rounded-md border border-border/40 p-3" title="Realized P&L on closed trades, after fees.">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Realized (closed)</div>
            <div className={cn("text-2xl font-display font-bold mt-1", realized >= 0 ? "text-emerald-400" : "text-red-400")} data-testid="reality-check-net-pnl">
              {realized >= 0 ? "+" : ""}${realized.toFixed(2)}
            </div>
            <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
              {closedTrades} closed · gross {realizedGross >= 0 ? "+" : ""}${realizedGross.toFixed(2)} · fees ${realizedFees.toFixed(2)}
            </div>
          </div>
          <div className="rounded-md border border-border/40 p-3" title="Floating P&L on currently-open positions valued at last known price.">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Unrealized (open)</div>
            <div className={cn("text-2xl font-display font-bold mt-1", unrealized >= 0 ? "text-emerald-400" : "text-red-400")}>
              {unrealized >= 0 ? "+" : ""}${unrealized.toFixed(2)}
            </div>
            <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
              ${openMarketValue.toFixed(0)} open · ${cash.toFixed(0)} cash idle
            </div>
          </div>
          <div className="rounded-md border border-border/40 p-3">
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Bots in Profit</div>
            <div className="text-2xl font-display font-bold mt-1">{winners}<span className="text-base font-mono text-muted-foreground"> / {bots.length}</span></div>
            <div className="text-[11px] font-mono text-muted-foreground mt-0.5">{winRate.toFixed(0)}% win · {avgNetPct.toFixed(2)}% avg per bot</div>
          </div>
        </div>

        {showPriorBadge && (
          <PriorFallbackBadge
            excludedCount={priorTotal}
            unit="trades"
            included={realityData?.priorFallback?.included ?? includePrior}
            onToggle={() => setIncludePrior((v) => !v)}
            testId="reality-check-prior-fallback"
          />
        )}
      </CardContent>
    </Card>
  );
}

function TopPickCard({ pick }: { pick: NonNullable<ReturnType<typeof useBestPick>["data"]> }) {
  // Task #532 / C-2 + Rev 2 — whenever the API short-circuits with a
  // suppressedReason (any non-null value emitted by
  // computeBrainRuntimeState — currently "brain_offline",
  // "brain_offline_no_model", or "brain_status_unknown") render an
  // honest no-signal card. We deliberately treat *every* non-null
  // suppressedReason as "no live consensus" so a future backend-side
  // addition cannot silently fall through to the recommendation
  // layout (which would render "HOLD <coin> XX%" and a coin link with
  // an empty coinId).
  if (pick.suppressedReason) {
    const reasonCopy: Record<string, { headline: string; explainer: string }> = {
      brain_offline: {
        headline: "No live consensus",
        explainer: "Quant brain is OFF. No live consensus pick is available.",
      },
      brain_offline_no_model: {
        headline: "No live consensus",
        explainer:
          "Quant brain is enabled in config but the ml-engine has no promoted model loaded — nothing to recommend yet.",
      },
      brain_status_unknown: {
        headline: "No live consensus",
        explainer:
          "Quant brain runtime status is currently unreachable, so no live recommendation can be trusted.",
      },
    };
    const fallbackCopy = {
      headline: "No live consensus",
      explainer:
        "Quant brain is not online. No live consensus pick is available.",
    };
    const copy = reasonCopy[pick.suppressedReason] ?? fallbackCopy;
    return (
      <Card
        className="border bg-amber-400/5 border-amber-400/30 backdrop-blur"
        data-testid="top-pick-card-suppressed"
      >
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <Crown className="w-4 h-4 text-yellow-400" />
            What the bots agree on right now
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-2">
            <div className="text-2xl font-display font-bold text-amber-300 flex items-center gap-2">
              <Minus className="w-6 h-6" />
              {copy.headline}
            </div>
            <p className="text-sm text-muted-foreground max-w-2xl">
              {pick.reasoning || copy.explainer}
            </p>
            <div className="text-[11px] font-mono text-muted-foreground/70">
              suppressedReason: {pick.suppressedReason}
              {pick.brainRuntimeState ? ` · brainRuntimeState: ${pick.brainRuntimeState}` : ""}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }
  const actionColor = pick.action === "buy" ? "text-emerald-400" : pick.action === "sell" ? "text-red-400" : "text-yellow-400";
  const actionBg = pick.action === "buy" ? "bg-emerald-400/5 border-emerald-400/30" : pick.action === "sell" ? "bg-red-400/5 border-red-400/30" : "bg-yellow-400/5 border-yellow-400/30";
  const Icon = pick.action === "buy" ? TrendingUp : pick.action === "sell" ? TrendingDown : Minus;
  return (
    <Card className={cn("border backdrop-blur", actionBg)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Crown className="w-4 h-4 text-yellow-400" />
          What the bots agree on right now
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className={cn("text-3xl font-display font-bold flex items-center gap-2", actionColor)}>
              <Icon className="w-7 h-7" />
              {pick.action.toUpperCase()} {pick.coinName}
              {pick.brain && (
                <Badge
                  variant="outline"
                  className={cn(
                    "ml-1 text-[10px] uppercase tracking-wider",
                    pick.brain === "QUANT" ? "border-emerald-500/40 text-emerald-400" : "border-blue-500/40 text-blue-400",
                  )}
                  data-testid="best-pick-brain"
                >
                  {pick.brain}
                </Badge>
              )}
            </div>
            <div className="text-sm text-muted-foreground mt-1">
              <Link href={`/coins/${pick.coinId}`} className="hover:text-foreground underline-offset-2 hover:underline">{pick.coinSymbol} @ {formatCurrency(pick.currentPrice)}</Link>
            </div>
            {pick.whatExplanation && (
              <div className="mt-2 max-w-2xl" data-testid="best-pick-what">
                <div className="text-[10px] uppercase font-mono text-muted-foreground">What</div>
                <p className="text-sm text-foreground/90 font-medium">{pick.whatExplanation}</p>
              </div>
            )}
            <div className="mt-2 max-w-2xl" data-testid="best-pick-why">
              <div className="text-[10px] uppercase font-mono text-muted-foreground">Why</div>
              <p className="text-sm text-foreground/80">{pick.reasoning}</p>
            </div>
            {/* Task #444 — `newsTags` badges removed; news classifier is gone. */}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-5 text-center">
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">
                {pick.modelProbability != null ? "Model prob." : "Confidence"}
              </div>
              <div className="text-xl font-display font-bold mt-1" data-testid="best-pick-probability">
                {pick.modelProbability != null
                  ? `${(pick.modelProbability * 100).toFixed(0)}%`
                  : "—"}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">Hold for</div>
              <div className="text-xl font-display font-bold mt-1">{pick.holdTimeframe}</div>
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">Expected</div>
              <div className={cn("text-xl font-display font-bold mt-1", pick.expectedPriceChange >= 0 ? "text-emerald-400" : "text-red-400")}>
                {pick.expectedPriceChange >= 0 ? "+" : ""}{pick.expectedPriceChange.toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">EV after fees</div>
              <div
                className={cn(
                  "text-xl font-display font-bold mt-1",
                  (pick.evAfterFeesPct ?? 0) >= 0 ? "text-emerald-400" : "text-red-400",
                )}
                data-testid="best-pick-ev"
              >
                {pick.evAfterFeesPct != null
                  ? `${pick.evAfterFeesPct >= 0 ? "+" : ""}${pick.evAfterFeesPct.toFixed(2)}%`
                  : "—"}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Task #405 / B-AVG-HIDES — render the trailing 24h directional accuracy
// alongside the all-time figure, with a coin-flip-floor warning when the
// 24h sample is below 45 %. The audit found a 24h directional accuracy of
// 19.5 % was being masked by an all-time 56.5 %; this card flags
// degradation in red and now reports the 24h figure only.
// Task #444 — the "All-time" half of this card was removed. It averaged
// `outcome='correct'` over every prediction since launch, which folded
// in months of legacy personality/LLM-era picks (and quant_disabled
// stable rows that resolve as trivially-correct). Operators were
// reading the resulting headline (~70%) as a quant brain win-rate
// when the brain has authored zero recent trades. The 24h window
// (with the coin-flip floor degradation badge) is the only honest
// reading and is what this card now shows.
function Directional24hCard({
  directional24h,
}: {
  directional24h?: { resolved: number; correct: number; accuracyPct: number | null; coinFlipFloorPct: number };
}) {
  const d = directional24h;
  const recent = d && d.accuracyPct !== null ? d.accuracyPct : null;
  const floor = d?.coinFlipFloorPct ?? 45;
  const degraded = recent !== null && recent < floor;
  return (
    <Card className="bg-card/50 border-border/40" data-testid="directional-24h-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Directional Accuracy (last 24 h)
          {degraded && (
            <Badge variant="destructive" className="text-[9px] py-0 px-1.5" data-testid="directional-degraded-badge">
              below {floor.toFixed(0)}% floor
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className={cn(
            "text-2xl font-mono",
            degraded ? "text-red-400" : "text-foreground",
          )}
          data-testid="directional-24h"
        >
          {recent !== null
            ? `${recent.toFixed(1)}%`
            : d && d.resolved === 0
              ? "n/a"
              : "—"}
        </div>
        <div className="text-[10px] text-muted-foreground mt-1">
          {d
            ? `${d.correct.toLocaleString()} / ${d.resolved.toLocaleString()} directional outcomes in the last 24 h. Excludes "abstain"/no-trade rows. Coin-flip floor ${floor.toFixed(0)}%.`
            : "Loading…"}
        </div>
      </CardContent>
    </Card>
  );
}

// Task #512 — `BotLeaderboard` removed. The legacy 15-personality
// leaderboard was replaced by 4 deterministic executor cards
// (<FamilyFleet/>) plus a separate Benchmarks panel. Per-bot detail
// pages remain reachable through `/agents/:id` from the Family
// drill-down + Archived Agents pages.

function directionIcon(dir: string) {
  if (dir === "up") return <TrendingUp className="w-4 h-4 text-emerald-400" />;
  if (dir === "down") return <TrendingDown className="w-4 h-4 text-red-400" />;
  return <Minus className="w-4 h-4 text-yellow-400" />;
}

function outcomeColor(outcome: string | null | undefined) {
  if (outcome === "correct") return "text-emerald-400";
  if (outcome === "wrong") return "text-red-400";
  return "text-muted-foreground";
}

function RecentPredictions({ preds }: { preds: Array<{ id: number; agentName: string; coinName: string; direction: string; priceAtPrediction: number; predictedPrice: number; confidence: number; outcome: string | null; createdAt: string; timeframe?: string | null; reasoning?: string | null }> }) {
  // Task #532 / C-4 — hide quant_disabled / quant_abstain rows from
  // the "Live Bot Predictions" feed. With the brain offline, 100 % of
  // predictions in the last hour have `direction='stable'`,
  // `confidence=0`, and `reasoning ~ 'quant_disabled_…'`. Rendering
  // them as "ABSTAIN ($0 confidence)" rows made the panel look like
  // the brain was *making* decisions when in reality it was refusing
  // to. Filter them out and surface a single "n abstain rows hidden"
  // count so operators still know they exist.
  const isAbstain = (p: { reasoning?: string | null; direction: string; confidence: number }) => {
    const r = (p.reasoning ?? "").toLowerCase();
    if (r.includes("quant_disabled") || r.includes("quant_abstain")) return true;
    return false;
  };
  const abstainCount = preds.filter(isAbstain).length;
  const visible = preds.filter((p) => !isAbstain(p)).slice(0, 12);
  return (
    <Card className="bg-card/50 border-border/40">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Live Bot Predictions
          {abstainCount > 0 && (
            <Badge
              variant="outline"
              className="ml-auto text-[10px] py-0 px-1.5 border-amber-400/40 text-amber-300"
              data-testid="recent-predictions-abstain-count"
            >
              {abstainCount} abstain hidden
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {visible.length === 0 && (
            <div className="text-center py-8 text-muted-foreground font-mono text-sm">
              {abstainCount > 0
                ? `n/a — last ${abstainCount} predictions were quant abstains (brain offline). No real directions to show.`
                : "Waiting for the first analysis cycle…"}
            </div>
          )}
          {visible.map((p) => (
            <div
              key={p.id}
              className="flex items-center justify-between p-2.5 rounded-lg border border-border/20 bg-background/40"
              data-testid={`prediction-${p.id}`}
            >
              <div className="flex items-center gap-3 min-w-0">
                {directionIcon(p.direction)}
                <div className="min-w-0">
                  <div className="text-sm flex items-center gap-2">
                    <span className="text-muted-foreground truncate">{p.agentName}</span>
                    <span className="font-medium truncate">{p.coinName}</span>
                  </div>
                  <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
                    {formatCurrency(p.priceAtPrediction)} → {formatCurrency(p.predictedPrice)}
                    <span className="ml-2 text-muted-foreground/60">{formatTimeAgo(p.createdAt)}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Badge variant="outline" className="font-mono text-[10px]">{(p.confidence * 100).toFixed(0)}%</Badge>
                <span className={cn("text-[11px] font-mono font-bold uppercase w-14 text-right", outcomeColor(p.outcome))}>
                  {p.outcome || "pending"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const adminApi = useAdminKey({});
  const adminReset = useAdminKey({ keyName: "ADMIN_RESET_KEY" });
  const { data: dashboard, isLoading } = useGetDashboard({ query: { refetchInterval: 30000, queryKey: getGetDashboardQueryKey() } });
  const { data: status } = useGetMonitoringStatus({ query: { refetchInterval: 10000, queryKey: getGetMonitoringStatusQueryKey() } });
  const { data: bestPick } = useBestPick();
  const { data: paperPortfolios } = usePaperPortfolios();

  // Task #512 — keep only executor portfolios for the QuantFleet card
  // so "Bots in profit X/Y" denominator excludes Strategy-Lab
  // benchmarks. The API already filters out archived legacy rows in
  // Task #512; this client-side split keeps benchmarks out of the
  // executor math while leaving them visible in <BenchmarksPanel/>.
  const executorBots = (paperPortfolios ?? [])
    .filter((b) => b.kind !== "benchmark")
    .sort((a, b) => b.totalValue - a.totalValue);

  return (
    <div className="space-y-6" data-testid="dashboard-page">
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <h1 className="text-4xl md:text-5xl font-display font-bold tracking-tight gradient-text" data-testid="text-page-title">
            Crypto AI Trading
          </h1>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <StatusPill status={status} />
            {/* Task #512 — activity banner: live trade rate + last decision age. */}
            <ActivityBanner />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ResetBotsButton />
          <ForceAnalysisButton />
        </div>
      </div>

      {/* Task #518 — brain runtime banner. Renders nothing when the
          brain is online; renders a red/amber banner with the
          abstain-reason rollup when the brain is silently dark or
          operator-disabled. Sits above the admin-key prompt so it is
          impossible to miss even on a fresh-load tab. */}
      <BrainStateBanner />

      {/* Task #614 — MTTM lane banner. Renders nothing when MTTM is
          off; pinned at the top above HorizonRolesCard when active so
          the operator sees the 16-slot guarantee + verdict line on
          every dashboard load. */}
      <MttmBanner />

      {/* Task #659 (C-BTC) — Diagnostic paper sandbox banner. Renders
          nothing unless `mttm_mode='diagnostic_sandbox'`; pinned just
          below the default MTTM banner so when both lanes happen to be
          shown side-by-side in a dev environment the operator can see
          which is which. The DS banner itself enforces mutual
          exclusivity at the data level (the universe / sizing pins are
          mode-gated in mttm.ts), so co-existence is purely visual. */}
      <DiagnosticSandboxBanner />

      {/* Task #551 — per-timeframe role card. Unconditionally visible
          (no collapse) so a trader scanning the dashboard sees, at a
          glance, which horizons are trade-eligible vs shadow vs
          context vs disabled. Renders directly under the brain-state
          banner. */}
      <HorizonRolesCard />

      {/* Task #554 — dataset-refresher banner. Renders nothing when
          the refresher is healthy; renders a red/amber sticky banner
          when any tf is past-due or there are unread alerts. Clicking
          it scrolls to the failure-log panel inside
          DatasetFreshnessCard below. */}
      <DatasetFreshnessBanner />

      {(!adminApi.hasKey || !adminReset.hasKey) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3" data-testid="dashboard-admin-keys">
          {!adminApi.hasKey && (
            <AdminKeyField
              admin={adminApi}
              label="Admin key (ADMIN_API_KEY)"
              helpText="Needed to run an analysis on demand. Stays in this tab only."
              testIdPrefix="dashboard-admin-api-key"
            />
          )}
          {!adminReset.hasKey && (
            <AdminKeyField
              admin={adminReset}
              label="Reset key (ADMIN_RESET_KEY)"
              helpText="Needed to reset the bots back to $1,000. Stays in this tab only."
              testIdPrefix="dashboard-admin-reset-key"
            />
          )}
        </div>
      )}

      {/* Task #444 — `LlmSidecarPanel` removed with the LLM/news plane. */}

      {/* Task #512 — Quant Fleet card now feeds executor-only portfolios
          so "Bots in profit X/Y" denominator excludes Strategy-Lab
          benchmarks. Family Fleet renders the 4 deterministic strategy
          cards underneath; Benchmarks panel surfaces baseline rows. */}
      {executorBots.length > 0 ? <QuantFleetCard bots={executorBots} /> : <Skeleton className="h-28 w-full" />}

      <FamilyFleet />

      <BenchmarksPanel />

      {/* Realized + unrealized now live inside QuantFleetCard above. */}

      <Directional24hCard directional24h={dashboard?.directional24h} />

      <PortfolioRiskCard />

      <Topup5mBanner />

      <DisabledOutcomeBanner />

      {/* Task #554 — dataset-refresher card. Always-visible per-tf
          freshness pills + an expandable failure log. Sits next to
          the other health cards so an operator scanning the
          dashboard sees cache health without leaving the page. */}
      <DatasetFreshnessCard />

      {/* Task #599 — 1h/2h ON/OFF calibration verdict. Re-runs
          automatically after every successful 1h or 2h dataset
          snapshot so an operator can see at a glance whether the
          short-tf gate is still tuned correctly. Sits next to the
          freshness card because they share the same underlying
          status file. */}
      <ShortTfCalibrationVerdictCard />

      <SkipTrackerHealthCard />

      <FiveMTopupStatusCard />

      <FiveMTopupCard />

      <MarketSignalsHealthCard />

      <LiquidationsBreakdownCard />

      <QuantAbstainReasonsCard />

      {bestPick && <TopPickCard pick={bestPick} />}

      {/* Task #512 — the legacy 15-personality "Bots Leaderboard"
          collapsible was removed. The 4 deterministic executor cards
          live in <FamilyFleet/> above; Strategy-Lab baselines live in
          <BenchmarksPanel/>; archived legacy bots live on the
          /agents/archived page (sidebar link). */}

      <Collapsible defaultOpen>
        <CollapsibleTrigger className="w-full group" data-testid="toggle-recent-predictions">
          <div className="flex items-center justify-between p-3 rounded-lg bg-card/40 border border-border/40 hover:bg-card/60 transition-colors">
            <span className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Live Bot Predictions
            </span>
            <ChevronDown className="w-4 h-4 text-muted-foreground transition-transform group-data-[state=closed]:-rotate-90" />
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-3">
          {isLoading ? <Skeleton className="h-48 w-full" /> : <RecentPredictions preds={dashboard?.recentPredictions ?? []} />}
        </CollapsibleContent>
      </Collapsible>

      <Link href="/diagnostics">
        <div
          className="flex items-center justify-between p-4 rounded-lg bg-gradient-to-r from-primary/5 to-secondary/5 border border-primary/20 hover:border-primary/40 transition-colors cursor-pointer"
          data-testid="link-diagnostics"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Stethoscope className="w-5 h-5 text-primary" />
            </div>
            <div>
              <div className="font-medium">Brain Diagnostics</div>
              <div className="text-xs text-muted-foreground">
                Model accuracy, calibration, coverage, gates &amp; anomalies — for tuning the quant brain
              </div>
            </div>
          </div>
          <ChevronDown className="w-5 h-5 text-muted-foreground -rotate-90" />
        </div>
      </Link>
    </div>
  );
}
