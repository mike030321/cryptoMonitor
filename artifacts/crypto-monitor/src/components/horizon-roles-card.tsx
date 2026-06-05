// Per-timeframe role card. Trade badge wording is gated on the brain
// kill-switch: "TRADE-ELIGIBLE" unless `quant_brain_enabled === true`,
// then "TRADING". See task #551.
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";
import { useBrainRuntimeStatus } from "@/hooks/use-news";
import {
  useTimeframeRoles,
  type TimeframeRole,
  type TimeframeRoleEntry,
  type TimeframeRolesDoc,
} from "@/hooks/use-timeframe-roles";

export const TIMEFRAME_ORDER = ["1m", "5m", "1h", "2h", "6h", "1d"] as const;

export function isClickableEvidence(ref: string): boolean {
  return ref.length > 0 && ref !== "fail-closed-default";
}

export function badgeClassesFor(role: TimeframeRole): string {
  switch (role) {
    case "trade":
      return "bg-emerald-500/15 text-emerald-300 border border-emerald-400/40";
    case "shadow":
      return "bg-amber-500/15 text-amber-300 border border-amber-400/40";
    case "context":
      return "bg-slate-500/15 text-slate-300 border border-slate-400/40";
    case "disabled":
      return "bg-muted text-muted-foreground/70 border border-border/40";
  }
}

export function tradeBadgeText(quantBrainEnabled: boolean): string {
  return quantBrainEnabled ? "TRADING" : "TRADE-ELIGIBLE";
}

export function badgeTextFor(role: TimeframeRole, quantBrainEnabled: boolean): string {
  switch (role) {
    case "trade":
      return tradeBadgeText(quantBrainEnabled);
    case "shadow":
      return "SHADOW";
    case "context":
      return "CONTEXT";
    case "disabled":
      return "DISABLED";
  }
}

interface SummaryCounts {
  trade: number;
  shadow: number;
  context: number;
  disabled: number;
}

export function summaryLine(
  doc: TimeframeRolesDoc,
  counts: SummaryCounts,
  quantBrainEnabled: boolean,
): string {
  const tradeTfs = TIMEFRAME_ORDER.filter(
    (tf) => doc.timeframes[tf]?.role === "trade",
  );
  const verb = quantBrainEnabled ? "Trading on" : "Trade-eligible on";
  const tradeFragment = tradeTfs.length
    ? `${verb} ${tradeTfs.length} timeframe(s) (${tradeTfs.join(", ")}).`
    : `${verb} 0 timeframes.`;
  return `${tradeFragment} ${counts.shadow} shadow. ${counts.context} context. ${counts.disabled} disabled.`;
}

export interface RenderedEntry {
  tf: string;
  entry: TimeframeRoleEntry;
  synthesized: boolean;
}

// Always returns six rows in TIMEFRAME_ORDER. Missing TFs get a
// fail-closed disabled entry so the dashboard mirrors the server gate
// (a missing TF is refused, not absent). Counts are recomputed from
// the rendered six rows whenever any TF was synthesized so a partial
// document cannot under-report disabled horizons.
export function normalizeForRender(
  doc: TimeframeRolesDoc,
  serverSummary: SummaryCounts,
): {
  renderedEntries: RenderedEntry[];
  synthesizedMissingTfs: string[];
  effectiveDoc: TimeframeRolesDoc;
  effectiveSummary: SummaryCounts;
} {
  const synthesizedMissingTfs: string[] = [];
  const renderedEntries: RenderedEntry[] = TIMEFRAME_ORDER.map((tf) => {
    const real = doc.timeframes[tf];
    if (real) return { tf, entry: real, synthesized: false };
    synthesizedMissingTfs.push(tf);
    return {
      tf,
      synthesized: true,
      entry: {
        role: "disabled",
        context_subkind: null,
        disabled_reason: "by_safety",
        reason: `No entry for ${tf} in the role registry; fail-closed defaults applied.`,
        evidence_ref: "fail-closed-default",
        last_reviewed_at: doc.generated_at,
        promoted_slices_in_tf: [],
      } satisfies TimeframeRoleEntry,
    };
  });
  const renderedCounts: SummaryCounts = renderedEntries.reduce(
    (acc, { entry }) => {
      acc[entry.role] += 1;
      return acc;
    },
    { trade: 0, shadow: 0, context: 0, disabled: 0 } as SummaryCounts,
  );
  const effectiveSummary =
    synthesizedMissingTfs.length > 0 ? renderedCounts : serverSummary;
  const effectiveDoc: TimeframeRolesDoc = {
    ...doc,
    timeframes: Object.fromEntries(
      renderedEntries.map(({ tf, entry }) => [tf, entry]),
    ),
  };
  return { renderedEntries, synthesizedMissingTfs, effectiveDoc, effectiveSummary };
}

function HorizonRoleRow({
  timeframe,
  entry,
  quantBrainEnabled,
  synthesized = false,
}: {
  timeframe: string;
  entry: TimeframeRoleEntry;
  quantBrainEnabled: boolean;
  synthesized?: boolean;
}) {
  const badgeText = badgeTextFor(entry.role, quantBrainEnabled);
  const subLabel =
    entry.role === "context" && entry.context_subkind
      ? entry.context_subkind
      : entry.role === "disabled" && entry.disabled_reason
        ? entry.disabled_reason
        : null;
  const promotedCount = entry.promoted_slices_in_tf.length;
  const evidenceClickable = isClickableEvidence(entry.evidence_ref);
  const tooltip = `last reviewed: ${entry.last_reviewed_at} · evidence: ${entry.evidence_ref}`;

  return (
    <div
      className="flex flex-wrap items-start gap-x-3 gap-y-1 py-2 border-b border-border/30 last:border-b-0"
      data-testid={`horizon-role-row-${timeframe}`}
      data-role={entry.role}
      data-synthesized={synthesized ? "true" : "false"}
      title={tooltip}
    >
      <div className="flex items-center gap-2 shrink-0 min-w-[180px]">
        <Badge
          className={cn(
            "text-[10px] font-mono uppercase tracking-wider px-2 py-0.5",
            badgeClassesFor(entry.role),
          )}
          data-testid={`horizon-role-badge-${timeframe}`}
        >
          {badgeText}
        </Badge>
        {subLabel && (
          <span
            className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground/80"
            data-testid={`horizon-role-sublabel-${timeframe}`}
          >
            {subLabel}
          </span>
        )}
        <span className="text-sm font-mono font-semibold text-foreground/90">
          {timeframe}
        </span>
      </div>
      <div className="flex-1 min-w-[200px] text-xs text-muted-foreground">
        {entry.reason}
      </div>
      <div className="flex items-center gap-3 text-[11px] font-mono text-muted-foreground shrink-0">
        <span data-testid={`horizon-role-promoted-${timeframe}`}>
          {promotedCount} promoted
        </span>
        {evidenceClickable ? (
          <a
            href={
              /^https?:\/\//i.test(entry.evidence_ref)
                ? entry.evidence_ref
                : `/${entry.evidence_ref.replace(/^\/+/, "")}`
            }
            target="_blank"
            rel="noreferrer"
            className="underline underline-offset-2 hover:text-foreground text-muted-foreground/80 truncate max-w-[280px]"
            data-testid={`horizon-role-evidence-${timeframe}`}
            title={entry.evidence_ref}
          >
            {entry.evidence_ref}
          </a>
        ) : (
          <span
            className="text-muted-foreground/60 truncate max-w-[280px]"
            data-testid={`horizon-role-evidence-${timeframe}`}
          >
            {entry.evidence_ref}
          </span>
        )}
      </div>
    </div>
  );
}

export function HorizonRolesCard() {
  const rolesQ = useTimeframeRoles();
  const brainQ = useBrainRuntimeStatus();
  // The `trade` badge wording flips only when we have positive
  // confirmation that the brain is enabled. An undefined brain status
  // (loading or fetch error) falls back to the safer "TRADE-ELIGIBLE"
  // wording so the dashboard cannot accidentally claim "TRADING".
  const quantBrainEnabled = brainQ.data?.brainEnabled === true;

  if (rolesQ.isLoading) {
    return (
      <Card data-testid="horizon-roles-card">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <ShieldCheck className="w-4 h-4" /> Horizon Roles
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-32 w-full" data-testid="horizon-roles-card-loading" />
        </CardContent>
      </Card>
    );
  }

  if (rolesQ.isError || !rolesQ.data) {
    return (
      <Card data-testid="horizon-roles-card">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <ShieldCheck className="w-4 h-4" /> Horizon Roles
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className="text-sm text-amber-300 bg-amber-500/10 border border-amber-400/30 rounded-md px-3 py-2"
            data-testid="horizon-roles-card-error"
          >
            Could not read horizon roles — defaulting to refused. The
            promotion gate and the trade-execution path will refuse every
            timeframe until the role registry is reachable.
          </div>
        </CardContent>
      </Card>
    );
  }

  const { document: doc, summary } = rolesQ.data;
  const { renderedEntries, synthesizedMissingTfs, effectiveDoc, effectiveSummary } =
    normalizeForRender(doc, summary);
  const isEmptyDoc = Object.keys(doc.timeframes).length === 0;
  const showFailClosedBanner = renderedEntries.every(
    ({ entry }) =>
      entry.role === "disabled" &&
      entry.disabled_reason === "by_safety" &&
      entry.evidence_ref === "fail-closed-default",
  );

  return (
    <Card data-testid="horizon-roles-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2 flex-wrap">
          <ShieldCheck className="w-4 h-4" /> Horizon Roles
          <span className="text-[10px] font-normal text-muted-foreground normal-case tracking-normal">
            (per-timeframe trade authority — read from
            shared/timeframe-roles.json)
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {showFailClosedBanner && (
          <div
            className="text-sm text-amber-300 bg-amber-500/10 border border-amber-400/30 rounded-md px-3 py-2"
            data-testid="horizon-roles-fail-closed-banner"
          >
            {isEmptyDoc
              ? "All timeframes refused (fail-closed). The role registry is empty; both the promotion gate and the trade-execution path are refusing every timeframe until at least one is approved."
              : "All timeframes refused (fail-closed). The role registry file is missing or malformed; both the promotion gate and the trade-execution path are refusing every timeframe until it is restored."}
          </div>
        )}
        {synthesizedMissingTfs.length > 0 && !showFailClosedBanner && (
          <div
            className="text-xs text-amber-300 bg-amber-500/10 border border-amber-400/30 rounded-md px-3 py-2"
            data-testid="horizon-roles-partial-doc-banner"
          >
            Role registry is missing {synthesizedMissingTfs.length} required
            timeframe(s) ({synthesizedMissingTfs.join(", ")}). Those rows are
            shown as fail-closed disabled (gate refuses promotion and trading
            until each is published).
          </div>
        )}
        <div
          className="text-xs text-muted-foreground"
          data-testid="horizon-roles-summary"
        >
          {summaryLine(effectiveDoc, effectiveSummary, quantBrainEnabled)}
        </div>
        <div className="rounded-md border border-border/30 px-3">
          {renderedEntries.map(({ tf, entry, synthesized }) => (
            <HorizonRoleRow
              key={tf}
              timeframe={tf}
              entry={entry}
              quantBrainEnabled={quantBrainEnabled}
              synthesized={synthesized}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
