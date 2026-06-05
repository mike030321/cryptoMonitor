/**
 * Task #518 — brain runtime state banner.
 *
 * Surfaces the dashboard's "is the brain actually authoring decisions
 * right now?" answer at the top of the page. Renders nothing when the
 * brain is online so the dashboard chrome stays unchanged in the
 * happy path; renders a red banner with the abstain-reason rollup
 * when the brain is silently dark or operator-disabled.
 *
 * The audit that opened Task #518 found that an `offline_no_model`
 * brain looked indistinguishable from `online` on the dashboard:
 * StatusPill said "Live · Cycle #N · next in 30s" because the
 * monitoring loop was still ticking, and the activity banner showed
 * inflated trade counts because Strategy-Lab baselines kept
 * rebalancing. This banner exists so that situation cannot recur.
 */

import { Link } from "wouter";
import { useBrainRuntimeStatus, type PromotionGateRetries } from "@/hooks/use-news";
import { AlertTriangle, Clock, PowerOff, Stethoscope } from "lucide-react";
import { formatTimeAgo } from "@/lib/format";

type OfflineState = "offline_no_model" | "offline_disabled";

// Task #681 — known brainSource values the banner has explicit copy
// for. The OpenAPI / hook contract for `brainSource` is currently a
// fixed enum (see `BrainRuntimeStatus.brainSource` in
// `src/hooks/use-news.ts`), but the api-server route already types
// `brainSource` as a plain `string` on the wire (see
// `BrainRuntimeStatePayload` in
// `artifacts/api-server/src/routes/crypto/index.ts`). The banner's
// local `BrainSource` typedef is intentionally widened to `string` so
// that if a new server-side value is ever added (e.g. `"shadow"`)
// before this component is updated, the banner falls back to a
// "source unknown" copy that surfaces the raw value verbatim instead
// of silently rendering the generic "quant_brain_enabled is false"
// headline. The OpenAPI / hook contract is intentionally NOT widened
// here — that would be a contract change requiring codegen.
type KnownBrainSource = "default" | "manual" | "auto_revert" | "env";
type BrainSource = KnownBrainSource | (string & {});

interface StateCopy {
  title: string;
  subtitle: string;
  icon: typeof AlertTriangle;
  ring: string;
  bg: string;
  fg: string;
}

// Task #532 / C-3 — source-aware "why is the brain off?" copy. The
// previous version called every offline_disabled state a "kill-switch
// is OFF" failure, but the audit found that on the live workspace
// the `app_settings.quant_brain_enabled` row does not exist at all
// (default OFF, source="default"). "Kill-switch is OFF" implies an
// operator flipped it; the truth is the brain has never been turned
// on. The copy below renders the actual gating reason inferred from
// `brainSource`.
function disabledCopyForSource(source: BrainSource | undefined): StateCopy {
  switch (source) {
    case "default":
      return {
        title: "QUANT DISABLED — quant brain has never been enabled",
        subtitle:
          "There is no `app_settings.quant_brain_enabled=true` row in the database. The default is OFF; the brain has never been flipped on. Cycles are running but nothing has trade authority.",
        icon: PowerOff,
        ring: "ring-amber-500/40",
        bg: "bg-amber-500/10",
        fg: "text-amber-300",
      };
    case "env":
      return {
        title: "QUANT DISABLED — env force-off",
        subtitle:
          "QUANT_BRAIN_FORCE_OFF=1 in the environment. This overrides any DB row or operator flip. Cycles are running but nothing has trade authority.",
        icon: PowerOff,
        ring: "ring-amber-500/40",
        bg: "bg-amber-500/10",
        fg: "text-amber-300",
      };
    case "auto_revert":
      return {
        title: "QUANT DISABLED — auto-reverted",
        subtitle:
          "Auto-revert tripped (drift_low or consecutive losses) and flipped quant_brain_enabled back to OFF. Inspect `/diagnostics` for the most recent revert reason.",
        icon: AlertTriangle,
        ring: "ring-red-500/40",
        bg: "bg-red-500/10",
        fg: "text-red-300",
      };
    case "manual":
      return {
        title: "QUANT DISABLED — operator flipped it OFF",
        subtitle:
          "An operator set quant_brain_enabled=false. Cycles are running but nothing has trade authority. Re-enable from the brain control plane after the verification gate has promoted a slice.",
        icon: PowerOff,
        ring: "ring-amber-500/40",
        bg: "bg-amber-500/10",
        fg: "text-amber-300",
      };
    default:
      // Task #681 — explicit "source unknown" fallback. If the
      // brainSource is a defined string but not one of the four known
      // values above (e.g. server adds a new enum value before the
      // banner is updated), surface the raw value verbatim in the
      // headline rather than silently rendering the generic
      // "quant_brain_enabled is false" copy. The undefined case
      // (source not yet known on the wire) keeps the original
      // generic copy so existing behaviour is preserved.
      if (typeof source === "string") {
        return {
          title: "QUANT DISABLED — source unknown",
          subtitle: `quant_brain_enabled is false. Reported source: ${source}`,
          icon: PowerOff,
          ring: "ring-amber-500/40",
          bg: "bg-amber-500/10",
          fg: "text-amber-300",
        };
      }
      return {
        title: "QUANT DISABLED — quant brain disabled",
        subtitle:
          "quant_brain_enabled is false. Cycles are running but nothing has trade authority.",
        icon: PowerOff,
        ring: "ring-amber-500/40",
        bg: "bg-amber-500/10",
        fg: "text-amber-300",
      };
  }
}

const NO_MODEL_COPY: StateCopy = {
  title: "QUANT DISABLED — no trained model available",
  subtitle:
    "ml-engine has no per-coin or pooled model for the requested pairs. Every cycle is abstaining; nothing the executor sees is a real quant decision.",
  icon: AlertTriangle,
  ring: "ring-red-500/40",
  bg: "bg-red-500/10",
  fg: "text-red-300",
};

function copyFor(state: OfflineState, source: BrainSource | undefined): StateCopy {
  return state === "offline_no_model" ? NO_MODEL_COPY : disabledCopyForSource(source);
}

// Task #686 — render a small "the brain-enable check had to wait for
// the ml-engine N times in the last hour" chip inside the offline
// banner. Source signal is the warn-level retry events emitted by
// `brain-promotion-gate.ts`; the api-server route mirrors those into
// the `promotionGateRetries` field on `/crypto/brain/runtime-status`.
//
// Visibility rules:
//   * Hidden when `promotionGateRetries` is missing on the wire (older
//     api-server) or when `count === 0` (no recent retry pressure).
//   * Always shown otherwise — operator must be able to tell apart a
//     single-shot `history_unreachable` (count === 1) from "all 3
//     attempts failed" (count >= 3) at a glance.
function PromotionGateRetryChip({
  retries,
}: {
  retries: PromotionGateRetries | undefined;
}) {
  if (!retries || retries.count <= 0) return null;
  const windowMinutes = Math.round(retries.windowMs / 60_000);
  const reasonText = retries.mostRecentReason
    ? `most recent: ${retries.mostRecentReason}${
        retries.mostRecentAttempt ? ` (attempt ${retries.mostRecentAttempt})` : ""
      }`
    : null;
  const ago = retries.mostRecentAt ? formatTimeAgo(retries.mostRecentAt) : null;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-300"
      data-testid="brain-state-banner-promotion-gate-retries"
      data-promotion-gate-retry-count={retries.count}
      title="Bounded retries the manual brain-enable check has performed against the ml-engine /verification-history endpoint."
    >
      <Clock className="w-3 h-3" />
      promotion-gate retries: {retries.count} in last {windowMinutes}m
      {reasonText && <> · {reasonText}</>}
      {ago && <> · {ago}</>}
    </span>
  );
}

export function BrainStateBanner() {
  const { data, isLoading } = useBrainRuntimeStatus();
  if (isLoading || !data) return null;
  if (data.state === "online") return null;

  const copy = copyFor(data.state, data.brainSource as BrainSource | undefined);
  const Icon = copy.icon;
  const reasons = Object.entries(data.recentAbstainReasons).sort(
    (a, b) => b[1] - a[1],
  );
  // Sticky positioning so the banner stays visible while operators
  // scroll the dashboard — the audit's whole point is that this state
  // must not be possible to scroll past unnoticed. `top-0 z-30` keeps
  // it pinned under any global header without overlapping the sidebar
  // (which is z-40+ via the layout shell).
  return (
    <div
      className={`sticky top-0 z-30 flex flex-col gap-2 px-4 py-3 rounded-xl ring-1 backdrop-blur-md shadow-lg ${copy.ring} ${copy.bg}`}
      data-testid="brain-state-banner"
      data-brain-state={data.state}
    >
      <div className="flex items-start gap-3">
        <Icon className={`w-5 h-5 mt-0.5 shrink-0 ${copy.fg}`} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <div className={`text-sm font-mono font-semibold uppercase tracking-wider ${copy.fg}`}>
              {copy.title}
            </div>
            <Link
              href="/diagnostics"
              className={`inline-flex items-center gap-1 text-[11px] font-mono uppercase tracking-wider underline-offset-2 hover:underline ${copy.fg}`}
              data-testid="brain-state-banner-diagnostics-link"
            >
              <Stethoscope className="w-3 h-3" />
              View Brain Diagnostics →
            </Link>
          </div>
          <div className="text-xs text-foreground/80 mt-1">{copy.subtitle}</div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono text-muted-foreground">
            {reasons.length > 0 && (
              <span>
                {reasons
                  .map(([reason, count]) => `${count} ${reason}`)
                  .join(" · ")}{" "}
                in last {data.windowMinutes}m
              </span>
            )}
            {data.lastSuccessfulDecisionAt && (
              <span data-testid="brain-state-banner-last-decision">
                last successful decision{" "}
                <span className="text-foreground">
                  {formatTimeAgo(data.lastSuccessfulDecisionAt)}
                </span>
              </span>
            )}
            {data.currentRunDir && (
              <span title="Latest training_run_* dir on the ml-engine">
                model: {data.currentRunDir}
              </span>
            )}
            {data.brainSource && (
              <span>source: {data.brainSource}</span>
            )}
            <PromotionGateRetryChip retries={data.promotionGateRetries} />
          </div>
        </div>
      </div>
    </div>
  );
}
