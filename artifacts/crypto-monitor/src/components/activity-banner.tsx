/**
 * Task #512 + #518 — dashboard activity banner.
 *
 * Surfaces a single-line "X executor / Y baseline trades in last
 * hour, last decision N min ago" pill at the top of the dashboard.
 * The split (Task #518) prevents the audit's failure mode where 60
 * Strategy-Lab baseline rebalances looked like a healthy executor
 * while the brain was actually in `offline_no_model` abstain.
 *
 * Pulls from `/dashboard-activity` and re-fetches every 30 s.
 */

import { useDashboardActivity } from "@/hooks/use-news";
import { Activity } from "lucide-react";
import { formatTimeAgo } from "@/lib/format";

export function ActivityBanner() {
  const { data } = useDashboardActivity();
  const exec = data?.executorTradesLastHour ?? 0;
  const baseline = data?.baselineTradesLastHour ?? 0;
  const lastExec = data?.lastExecutorTradeAt ?? null;
  const lastBaseline = data?.lastBaselineTradeAt ?? null;
  // Task #532 / C-1b — surface the gap between "last executor trade"
  // and "last *valid* (non-abstain) quant decision". The audit found
  // the brain emitted only `quant_disabled` abstains for days while
  // the activity strip read "last exec 4h ago" — operators couldn't
  // see that the brain hadn't authored a real direction in the
  // meantime. When the two timestamps disagree, the strip renders
  // both pills.
  const lastQuantDecision = data?.lastValidQuantDecisionAt ?? null;
  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-card/40 border border-border/30 text-xs font-mono text-muted-foreground"
      data-testid="activity-banner"
    >
      <Activity className="w-3.5 h-3.5 text-primary" />
      <span data-testid="activity-banner-executor">
        <span className="text-foreground font-semibold">{exec}</span> executor
      </span>
      <span className="text-muted-foreground/50">·</span>
      <span data-testid="activity-banner-baseline">
        <span className="text-foreground font-semibold">{baseline}</span> baseline
      </span>
      <span className="text-muted-foreground/50">·</span>
      <span title="Last executor trade timestamp — distinct from baseline rebalances">
        last exec{" "}
        <span className="text-foreground font-semibold" data-testid="activity-banner-last-exec">
          {lastExec ? formatTimeAgo(lastExec) : "—"}
        </span>
      </span>
      {baseline > 0 && lastBaseline && (
        <>
          <span className="text-muted-foreground/50">·</span>
          <span title="Last baseline rebalance — Strategy-Lab passive baselines">
            base{" "}
            <span className="text-foreground/80">
              {formatTimeAgo(lastBaseline)}
            </span>
          </span>
        </>
      )}
      <span className="text-muted-foreground/50">·</span>
      <span
        title="Most recent prediction whose reasoning was NOT a quant_disabled / quant_abstain abstain. If this is much older than 'last exec', the brain has only been emitting abstains."
        data-testid="activity-banner-last-valid-quant-decision"
      >
        last quant{" "}
        <span className="text-foreground font-semibold">
          {lastQuantDecision ? formatTimeAgo(lastQuantDecision) : "n/a"}
        </span>
      </span>
    </div>
  );
}
