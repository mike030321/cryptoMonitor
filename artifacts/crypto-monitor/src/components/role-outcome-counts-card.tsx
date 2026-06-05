/**
 * Task #578 — Role outcome counts.
 *
 * The Python brain partitions every accepted record-outcome call by
 * the slice's `slice_role` (`trade | shadow | context | disabled`)
 * and exposes the cumulative per-role counter on
 * `/ml/meta-brain/stats` as `inputs_by_role`. With the wire now
 * reliably stamping `slice_role` on every call this counter is
 * trustworthy — but the dashboard had no surface that read it, so
 * operators couldn't tell whether the per-timeframe role split they
 * configured (e.g. "1h is shadow, 1d is trade") was actually being
 * honoured by live traffic.
 *
 * This card renders four tiles (trade / shadow / context / disabled)
 * showing the cumulative count, the count over the last 24h, and a
 * compact 24-bar hourly sparkline so an operator can spot a sudden
 * spike in any single bucket. Each tile carries a plain-language
 * explanation of what that role means in the brain's learning loop.
 *
 * Data source: `/api/crypto/meta-brain/cycle-stats` already proxies
 * the ml-engine `/ml/meta-brain/stats` payload through
 * `mlEngine.stats`, so no new api-server route is needed.
 */
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Users } from "lucide-react";
import { cn } from "@/lib/utils";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

const ROLES = ["trade", "shadow", "context", "disabled"] as const;
type Role = (typeof ROLES)[number];

interface HourlyBucket {
  hour_start: string;
  counts: Record<Role, number>;
}

interface MetaBrainStats {
  inputs_by_role?: Partial<Record<Role, number>> | null;
  inputs_by_role_24h?: Partial<Record<Role, number>> | null;
  inputs_by_role_hourly?: HourlyBucket[] | null;
  inputs_by_role_window_hours?: number | null;
}

interface CycleStatsResponse {
  mlEngine: {
    stats: MetaBrainStats | null;
  };
  fetchedAt: string;
}

const ROLE_META: Record<
  Role,
  { label: string; tagline: string; tone: string; bar: string }
> = {
  trade: {
    label: "Trade",
    tagline:
      "Outcomes that updated the brain's trust scores. Real learning happens here.",
    tone: "border-emerald-500/40 bg-emerald-500/5",
    bar: "bg-emerald-400",
  },
  shadow: {
    label: "Shadow",
    tagline:
      "Outcomes observed for shadow-mode audit. Stored, but never touch trust scores.",
    tone: "border-amber-500/40 bg-amber-500/5",
    bar: "bg-amber-400",
  },
  context: {
    label: "Context",
    tagline:
      "Stored for governance and context analysis only. Never touches trust scores.",
    tone: "border-sky-500/40 bg-sky-500/5",
    bar: "bg-sky-400",
  },
  disabled: {
    label: "Disabled",
    tagline:
      "Rejected — a 'disabled' arrival means an upstream caller is misconfigured for this slice.",
    tone: "border-rose-500/40 bg-rose-500/5",
    bar: "bg-rose-400",
  },
};

function Sparkline({
  buckets,
  role,
  barClass,
}: {
  buckets: HourlyBucket[];
  role: Role;
  barClass: string;
}) {
  const values = buckets.map((b) => Number(b.counts?.[role] ?? 0));
  const max = values.reduce((m, v) => (v > m ? v : m), 0);
  return (
    <div
      className="flex items-end gap-[2px] h-8 w-full"
      data-testid={`role-trend-sparkline-${role}`}
      title={
        max === 0
          ? `No ${role} outcomes recorded in the last ${buckets.length}h`
          : `Peak hour: ${max} ${role} outcomes`
      }
    >
      {values.map((v, i) => {
        const heightPct = max > 0 ? Math.max(6, (v / max) * 100) : 4;
        return (
          <div
            key={`${role}-${i}`}
            className={cn(
              "flex-1 rounded-sm",
              v > 0 ? barClass : "bg-muted/40",
            )}
            style={{ height: `${heightPct}%` }}
          />
        );
      })}
    </div>
  );
}

export function RoleOutcomeCountsCard() {
  const { data, isLoading, isError } = useQuery<CycleStatsResponse>({
    queryKey: ["meta-brain-cycle-stats"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/meta-brain/cycle-stats`);
      if (!res.ok) throw new Error(`meta-brain cycle stats ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
    staleTime: 15_000,
  });

  const stats = data?.mlEngine?.stats ?? null;
  const totals = stats?.inputs_by_role ?? null;
  const last24h = stats?.inputs_by_role_24h ?? null;
  const hourly = stats?.inputs_by_role_hourly ?? null;
  const windowHours = stats?.inputs_by_role_window_hours ?? 24;
  const grandTotal = ROLES.reduce(
    (s, r) => s + Number(totals?.[r] ?? 0),
    0,
  );
  const grand24h = ROLES.reduce((s, r) => s + Number(last24h?.[r] ?? 0), 0);

  return (
    <Card
      className="bg-card/50 border-border/40"
      data-testid="role-outcome-counts-card"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Users className="w-4 h-4" />
          Outcomes by Role
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            per-role record-outcome arrivals from ml-engine ·
            confirms the timeframe-role split is being honoured
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading && (
          <Skeleton
            className="h-32 w-full"
            data-testid="role-outcome-counts-loading"
          />
        )}
        {isError && (
          <div
            className="text-sm text-rose-300 font-mono"
            data-testid="role-outcome-counts-error"
          >
            Couldn't load role outcome counts.
          </div>
        )}
        {!isLoading && !isError && stats === null && (
          <div
            className="text-sm text-muted-foreground"
            data-testid="role-outcome-counts-unavailable"
          >
            n/a — ml-engine /stats was unreachable.
          </div>
        )}
        {!isLoading && !isError && stats !== null && totals === null && (
          <div
            className="text-sm text-muted-foreground"
            data-testid="role-outcome-counts-missing-field"
          >
            ml-engine /stats responded but did not include the
            <code className="mx-1 text-xs">inputs_by_role</code>
            field — upgrade the ml-engine to surface per-role counters.
          </div>
        )}
        {!isLoading && !isError && totals !== null && (
          <>
            <div
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3"
              data-testid="role-outcome-counts-grid"
            >
              {ROLES.map((role) => {
                const meta = ROLE_META[role];
                const total = Number(totals?.[role] ?? 0);
                const recent = Number(last24h?.[role] ?? 0);
                const sharePct = grandTotal > 0 ? (total / grandTotal) * 100 : 0;
                return (
                  <div
                    key={role}
                    className={cn(
                      "rounded-md border p-3 space-y-2",
                      meta.tone,
                    )}
                    data-testid={`role-tile-${role}`}
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <div className="text-[10px] uppercase font-mono tracking-wider text-muted-foreground">
                        {meta.label}
                      </div>
                      <div
                        className="text-[10px] font-mono text-muted-foreground/70"
                        data-testid={`role-share-${role}`}
                      >
                        {grandTotal > 0 ? `${sharePct.toFixed(1)}% of total` : "—"}
                      </div>
                    </div>
                    <div className="flex items-baseline gap-3">
                      <div
                        className="text-2xl font-display font-bold"
                        data-testid={`role-total-${role}`}
                      >
                        {total.toLocaleString()}
                      </div>
                      <div
                        className="text-xs font-mono text-muted-foreground"
                        data-testid={`role-24h-${role}`}
                      >
                        +{recent.toLocaleString()} in {windowHours}h
                      </div>
                    </div>
                    {hourly && hourly.length > 0 && (
                      <Sparkline
                        buckets={hourly}
                        role={role}
                        barClass={meta.bar}
                      />
                    )}
                    <p className="text-[11px] text-muted-foreground leading-snug">
                      {meta.tagline}
                    </p>
                  </div>
                );
              })}
            </div>
            <div className="text-[10px] text-muted-foreground/70 font-mono flex flex-wrap items-center gap-x-3 gap-y-1">
              <span data-testid="role-grand-total">
                {grandTotal.toLocaleString()} cumulative outcomes (since last
                checkpoint reload)
              </span>
              <span className="text-muted-foreground/40">·</span>
              <span data-testid="role-grand-24h">
                {grand24h.toLocaleString()} in last {windowHours}h
              </span>
              {hourly === null && totals !== null && (
                <>
                  <span className="text-muted-foreground/40">·</span>
                  <span className="text-amber-300/80">
                    trend buffer not yet exposed by ml-engine — totals only
                  </span>
                </>
              )}
            </div>
            {/* Task #578 — make the in-memory nature of the trend
                explicit so operators don't read a flat sparkline
                immediately after a deploy as a real lull in traffic. */}
            <div
              className="text-[10px] text-muted-foreground/60 font-mono leading-snug"
              data-testid="role-trend-memory-note"
            >
              Trend is held in the ml-engine's runtime memory and
              resets on every restart; cumulative counters above are
              durable across restarts.
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
