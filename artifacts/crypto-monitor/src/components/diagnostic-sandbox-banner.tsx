/**
 * Task #659 (C-BTC) — Diagnostic Paper Sandbox banner.
 *
 * Renders nothing when MTTM is NOT in `diagnostic_sandbox` mode. When
 * the sandbox mode is active it pins a teal banner at the top of the
 * dashboard. Auto-disabled variant (red) takes precedence and surfaces
 * the disable reason verbatim. Polls the OpenAPI-generated
 * `useGetDiagnosticSandboxStatus` hook so the FE consumes the same
 * contract the server publishes.
 */

import type { ReactElement } from "react";
import { FlaskConical, OctagonX, Hourglass, AlertTriangle } from "lucide-react";
import {
  useGetDiagnosticSandboxStatus,
  getGetDiagnosticSandboxStatusQueryKey,
  useGetDiagnosticSandboxHealth,
  getGetDiagnosticSandboxHealthQueryKey,
} from "@workspace/api-client-react";

export function DiagnosticSandboxBanner(): ReactElement | null {
  const { data: status } = useGetDiagnosticSandboxStatus({
    query: {
      queryKey: getGetDiagnosticSandboxStatusQueryKey(),
      refetchInterval: 30_000,
      staleTime: 25_000,
    },
  });
  // Task #670 — DS health probe. Polls the trailing-window drawdown so
  // the banner can surface a soft "needs refit" warning before the
  // auto-disable evaluator's full-since-enable drawdown trips.
  const { data: health } = useGetDiagnosticSandboxHealth({
    query: {
      queryKey: getGetDiagnosticSandboxHealthQueryKey(),
      refetchInterval: 30_000,
      staleTime: 25_000,
    },
  });

  if (!status || status.mode !== "diagnostic_sandbox") return null;

  const ad = status.auto_disable_status;

  if (ad.disabled && ad.reason && ad.detail && ad.disabled_at) {
    return (
      <div
        className="rounded-lg ring-1 ring-red-500/40 bg-red-500/10 text-red-200 px-4 py-3 flex items-start gap-3"
        data-testid="diagnostic-sandbox-banner-disabled"
      >
        <OctagonX className="size-5 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0 text-sm">
          <div
            className="font-semibold"
            data-testid="diagnostic-sandbox-banner-label"
          >
            {status.label} — AUTO-DISABLED ({ad.reason})
          </div>
          <div className="text-xs opacity-90 mt-0.5">
            Disabled at {new Date(ad.disabled_at).toLocaleString()}. {ad.detail}{" "}
            The BTC/5m beta-calibrated lane is OFF until an operator
            acknowledges from the admin panel.
          </div>
        </div>
      </div>
    );
  }

  const ddPctText = `${(status.drawdown_floor_pct * 100).toFixed(1)}%`;
  const sizingText = `${(status.fixed_position_pct * 100).toFixed(2)}%`;
  const liveDdText = `${(status.current_drawdown_pct * 100).toFixed(2)}%`;
  const livePnlText = `${status.net_pnl_pct >= 0 ? "+" : ""}${(
    status.net_pnl_pct * 100
  ).toFixed(2)}%`;
  const pinnedSlot = status.universe[0];
  const slotText = pinnedSlot
    ? `${pinnedSlot.coin_id}/${pinnedSlot.timeframe}`
    : "bitcoin/5m";
  const readyBadge = status.ready ? (
    <span
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 bg-emerald-500/20 text-emerald-200 ring-emerald-500/40"
      data-testid="diagnostic-sandbox-banner-ready"
    >
      ready
    </span>
  ) : (
    <span
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 bg-slate-500/20 text-slate-200 ring-slate-500/40"
      data-testid="diagnostic-sandbox-banner-pending"
    >
      <Hourglass className="size-3.5" />
      pending promotion
    </span>
  );
  return (
    <div
      className="rounded-lg ring-1 ring-teal-500/40 bg-teal-500/10 text-teal-100 px-4 py-3"
      data-testid="diagnostic-sandbox-banner-active"
    >
      <div className="flex items-start gap-3">
        <FlaskConical className="size-5 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0 text-sm">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div
              className="font-semibold"
              data-testid="diagnostic-sandbox-banner-label"
            >
              {status.label}
            </div>
            {readyBadge}
          </div>
          <div className="text-[11px] opacity-75 mt-0.5">
            Universe pin: {slotText} · sizing {sizingText} (fixed)
          </div>
          <div
            className="text-xs opacity-90 mt-1"
            data-testid="diagnostic-sandbox-banner-config"
          >
            Calibrator: beta · meta-brain:{" "}
            {status.meta_shadow ? "SHADOW" : "live"} · scope-pinned to{" "}
            {slotText}
          </div>
          <div
            className="text-[11px] opacity-75 mt-1"
            data-testid="diagnostic-sandbox-banner-risk"
          >
            Auto-disable armed at: peak-to-trough drawdown ≤ {ddPctText} · n≥
            {status.n_neg_pnl_threshold} trades with cumulative PnL &lt; 0
          </div>
          <div
            className="text-[11px] opacity-90 mt-1"
            data-testid="diagnostic-sandbox-banner-metrics"
          >
            Live: {status.closed_trades_since}/{status.n_neg_pnl_threshold}{" "}
            trades · cum PnL{" "}
            <span
              className={
                status.net_pnl_pct >= 0 ? "text-emerald-300" : "text-red-300"
              }
            >
              {livePnlText}
            </span>{" "}
            · drawdown{" "}
            <span
              className={
                status.current_drawdown_pct <= status.drawdown_floor_pct
                  ? "text-red-300"
                  : "text-teal-100"
              }
            >
              {liveDdText}
            </span>{" "}
            · reviews remaining {status.reviews_remaining}
          </div>
          {health && health.evaluable ? (
            <div
              className={`text-[11px] mt-1 inline-flex items-center gap-1 rounded px-1.5 py-0.5 ring-1 ${
                health.floor_breached
                  ? "bg-red-500/20 text-red-200 ring-red-500/40"
                  : health.needs_refit
                    ? "bg-amber-500/20 text-amber-100 ring-amber-500/40"
                    : "bg-emerald-500/15 text-emerald-200 ring-emerald-500/30"
              }`}
              data-testid="diagnostic-sandbox-banner-health"
            >
              {(health.needs_refit || health.floor_breached) && (
                <AlertTriangle className="size-3.5" />
              )}
              <span>
                DS health (last {health.n_trades_observed}/
                {health.window_trades} trades): drawdown{" "}
                {(health.trailing_drawdown_pct * 100).toFixed(2)}% · headroom{" "}
                {(health.headroom_pct * 100).toFixed(2)}% to{" "}
                {(health.drawdown_floor_pct * 100).toFixed(1)}% floor
                {health.floor_breached
                  ? " · floor breached — re-fit now"
                  : health.needs_refit
                    ? ` · past ${(health.warn_threshold_pct * 100).toFixed(1)}% warn line — stage a re-fit`
                    : " · healthy"}
              </span>
            </div>
          ) : null}
          <div
            className="text-[11px] opacity-75 mt-1 font-mono"
            data-testid="diagnostic-sandbox-banner-version"
          >
            BTC version:{" "}
            {status.btc_version ?? <span className="italic">none staged</span>}
            {status.since
              ? ` · enabled ${new Date(status.since).toLocaleString()}`
              : ""}
          </div>
        </div>
      </div>
    </div>
  );
}
