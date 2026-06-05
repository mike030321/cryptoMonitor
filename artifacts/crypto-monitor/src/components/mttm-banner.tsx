/**
 * Task #614 — Minimum Truthful Trading Mode (MTTM) banner.
 *
 * Renders nothing when MTTM is off so the dashboard chrome is
 * unchanged in the default state. When MTTM is enabled it pins the
 * banner at the top of the dashboard with the exact wording and
 * fields the spec requires:
 *
 *   Title:  "Minimum Truthful Trading Mode — paper only"
 *   Line 1: Universe: 8 coins × {6h,1d}
 *   Line 2: Realised PnL Δ vs Buy & Hold (last 72h)
 *   Line 3: Trades in last 72h · win rate · post-fee PnL%
 *   Line 4: Auto-disable armed at: ≥{cap} consecutive losses
 *           OR n≥10 with post-fee < {n10PostFeeCapPct}% (the two
 *           triggers actually wired in `evaluateMttmAutoDisable`).
 *           We accurately surface only the rules the code enforces;
 *           inventing additional triggers in the banner would violate
 *           the "Minimum Truthful" contract of this mode.
 *   Right:  verdict pill (continue / expand / stop / insufficient data)
 *
 * If `autoDisabled=true`, the banner switches to a red "AUTO-DISABLED"
 * variant showing `disableReason.reason / trippedAt / detail` so the
 * operator cannot miss the breach.
 *
 * Polling is intentionally cheap (30s) and entirely client-side.
 */

import type { ReactElement, ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ShieldCheck, OctagonX, Hourglass } from "lucide-react";

interface MttmStateResponse {
  enabled: boolean;
  enabledAt: string | null;
  universe: Array<{ coinId: string; timeframe: string; version: string }>;
  universeSize: number;
  maxPositionPct: number;
  consecutiveLossCap: number;
  /**
   * Stored as a decimal fraction (e.g. -0.02 for the -2% threshold).
   * Multiply by 100 for display.
   */
  n10PostFeeCapPct: number;
  disableReason: {
    reason: "consecutive_losses" | "n10_post_fee" | "manual";
    detail: string;
    trippedAt: string;
    consecutiveLosses?: number;
    nTrades?: number;
    postFeePnlPct?: number;
  } | null;
  autoDisabled: boolean;
}

interface MttmReportResponse {
  window: "24h" | "48h" | "72h";
  decisionsEvaluated: number;
  /** Trades opened *in window* — this is the "trades in last 72h" figure. */
  tradesOpened: number;
  realisedPnlUsd: number;
  unrealisedPnlUsd: number;
  feesPaidUsd: number;
  slippageEstimateUsd: number;
  winRate: number | null;
  costAwareDirectionalAccuracy: number | null;
  baselines: Array<{
    strategyType: string;
    label: string;
    deltaPct: number | null;
  }>;
  consecutiveLosses: number;
  totalMttmTradesSinceEnable: number;
  postFeePnlPctSinceEnable: number | null;
  autoDisabled: boolean;
  disableReasonDetail: string | null;
  verdict: "continue" | "expand" | "stop" | "insufficient_data";
  verdictDetail: string;
}

function pct(v: number | null, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

/**
 * Count distinct coins in the configured universe so the spec's
 * "8 coins × {6h,1d}" wording is computed from the live config rather
 * than hard-coded — if the operator narrows or widens the universe
 * via /mttm/state/universe, the banner stays accurate.
 */
function describeUniverse(state: MttmStateResponse): string {
  const coinSet = new Set(state.universe.map((s) => s.coinId));
  const tfSet = new Set(state.universe.map((s) => s.timeframe));
  const tfList = Array.from(tfSet).sort().join(",");
  return `${coinSet.size} coins × {${tfList}}`;
}

export function MttmBanner(): ReactElement | null {
  const stateQ = useQuery({
    queryKey: ["mttm", "state"],
    queryFn: async (): Promise<MttmStateResponse> => {
      const r = await fetch(`${import.meta.env.BASE_URL}api/mttm/state`);
      if (!r.ok) throw new Error(`mttm/state HTTP ${r.status}`);
      return r.json();
    },
    refetchInterval: 30_000,
    staleTime: 25_000,
  });

  const reportQ = useQuery({
    queryKey: ["mttm", "report", "72h"],
    queryFn: async (): Promise<MttmReportResponse> => {
      const r = await fetch(`${import.meta.env.BASE_URL}api/mttm/report?window=72h`);
      if (!r.ok) throw new Error(`mttm/report HTTP ${r.status}`);
      return r.json();
    },
    refetchInterval: 30_000,
    staleTime: 25_000,
    enabled: stateQ.data?.enabled === true,
  });

  // Render nothing when MTTM is off — keeps the dashboard chrome
  // identical to today for the default operator path.
  if (!stateQ.data || !stateQ.data.enabled) return null;

  const state = stateQ.data;
  const report = reportQ.data;

  // Auto-disabled banner takes priority — operator MUST see this.
  if (state.autoDisabled && state.disableReason) {
    return (
      <div
        className="rounded-lg ring-1 ring-red-500/40 bg-red-500/10 text-red-200 px-4 py-3 flex items-start gap-3"
        data-testid="mttm-banner-disabled"
      >
        <OctagonX className="size-5 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0 text-sm">
          <div className="font-semibold">
            Minimum Truthful Trading Mode — AUTO-DISABLED ({state.disableReason.reason})
          </div>
          <div className="text-xs opacity-90 mt-0.5">
            Tripped at {new Date(state.disableReason.trippedAt).toLocaleString()}.{" "}
            {state.disableReason.detail} Trade authority for the {state.universeSize}-slot
            paper-only lane is OFF until an operator acknowledges from the admin
            panel. No new MTTM trades will open.
          </div>
        </div>
      </div>
    );
  }

  // n10PostFeeCapPct is a decimal fraction in the API (e.g. -0.02);
  // display it as a percent. `toFixed(0)` on -0.02 produces "-0",
  // which is wrong — multiply by 100 first.
  const n10ThresholdPctText = `${(state.n10PostFeeCapPct * 100).toFixed(0)}%`;
  const buyHoldDeltaPct =
    report?.baselines.find((b) => b.strategyType === "buy-hold")?.deltaPct ?? null;

  const verdictBadge = (() => {
    if (!report) return null;
    const map: Record<MttmReportResponse["verdict"], { cls: string; label: string; icon: ReactNode }> = {
      continue: {
        cls: "bg-amber-500/20 text-amber-200 ring-amber-500/40",
        label: "continue",
        icon: <ShieldCheck className="size-3.5" />,
      },
      expand: {
        cls: "bg-emerald-500/20 text-emerald-200 ring-emerald-500/40",
        label: "expand",
        icon: <ShieldCheck className="size-3.5" />,
      },
      stop: {
        cls: "bg-red-500/20 text-red-200 ring-red-500/40",
        label: "stop",
        icon: <OctagonX className="size-3.5" />,
      },
      insufficient_data: {
        cls: "bg-slate-500/20 text-slate-200 ring-slate-500/40",
        label: "insufficient data",
        icon: <Hourglass className="size-3.5" />,
      },
    };
    const m = map[report.verdict];
    return (
      <span
        className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 ${m.cls}`}
        data-testid="mttm-banner-verdict"
        title={report.verdictDetail}
      >
        {m.icon}
        verdict: {m.label}
      </span>
    );
  })();

  return (
    <div
      className="rounded-lg ring-1 ring-amber-500/40 bg-amber-500/10 text-amber-100 px-4 py-3"
      data-testid="mttm-banner-active"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className="size-5 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0 text-sm">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="font-semibold">
              Minimum Truthful Trading Mode — paper only
            </div>
            {verdictBadge}
          </div>

          <div
            className="text-xs opacity-90 mt-1"
            data-testid="mttm-banner-universe"
          >
            Universe: {describeUniverse(state)} ·{" "}
            {(state.maxPositionPct * 100).toFixed(0)}% per-position cap
          </div>

          {report ? (
            <>
              <div
                className="text-xs opacity-90 mt-1 flex flex-wrap items-center gap-x-3 gap-y-1"
                data-testid="mttm-banner-stats"
              >
                <span data-testid="mttm-banner-trades-72h">
                  Trades (last 72h): {report.tradesOpened}
                </span>
                <span data-testid="mttm-banner-win">
                  Win rate:{" "}
                  {report.winRate === null
                    ? "—"
                    : `${(report.winRate * 100).toFixed(0)}%`}
                </span>
                <span data-testid="mttm-banner-postfee">
                  Post-fee PnL: {pct(report.postFeePnlPctSinceEnable)}
                </span>
                <span data-testid="mttm-banner-bh-delta">
                  Δ vs Buy &amp; Hold (72h): {pct(buyHoldDeltaPct)}
                </span>
              </div>
              <div
                className="text-[11px] opacity-75 mt-1"
                data-testid="mttm-banner-risk"
              >
                Auto-disable armed at: ≥{state.consecutiveLossCap} consecutive
                losses · n≥10 with post-fee &lt; {n10ThresholdPctText}
              </div>
            </>
          ) : (
            <div className="text-xs opacity-75 mt-1">
              {reportQ.isLoading ? "Loading 72h report…" : "No report data yet."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
