import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Server, AlertTriangle, CheckCircle2, Clock, History } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/format";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

// Shape mirrors `ml-engine/app/scheduled_5m_topup.py:state` plus the
// task #424 winner overlay applied by `/ml/admin/5m-topup/status`.
// Every field is optional because the ml-engine adds new keys over time
// and we don't want one stale field to break this panel.
interface FiveMTopupStatus {
  enabled?: boolean;
  last_attempt_outcome?: string | null;
  last_finished_at?: number | null;
  last_topup_inserted?: number | null;
  last_alerts?: string[] | null;
  skips_locked_total?: number | null;
  // Task #424 — winning replica attribution.
  last_winning_replica?: string | null;
  // ISO-8601 string when the most recent successful tick fired.
  last_winning_at?: string | null;
  error?: string;
}

// Task #435 — recent-winners list (newest-first). Each entry mirrors a
// row written by `_record_winning_replica` in the ml-engine: a stable
// replica identity + the ISO timestamp of that successful tick.
interface RecentWinner {
  replica: string;
  tick_at: string;
}
interface RecentWinnersResponse {
  winners?: RecentWinner[];
  limit?: number;
  max?: number;
  error?: string;
}

function formatOutcome(o: string | null | undefined): string {
  if (!o) return "—";
  return o.replace(/_/g, " ");
}

// Task #435 — color-tag each unique replica so an operator can eyeball
// "all three boxes are taking turns" vs "only one color showing up".
// We use a small, deterministic palette so the same replica always
// renders in the same color across the list and across renders.
const REPLICA_PALETTE = [
  "bg-emerald-500/15 text-emerald-200 ring-emerald-500/30",
  "bg-sky-500/15 text-sky-200 ring-sky-500/30",
  "bg-violet-500/15 text-violet-200 ring-violet-500/30",
  "bg-amber-500/15 text-amber-200 ring-amber-500/30",
  "bg-rose-500/15 text-rose-200 ring-rose-500/30",
  "bg-teal-500/15 text-teal-200 ring-teal-500/30",
] as const;

// Task #441 — solid-fill counterparts of REPLICA_PALETTE, indexed the
// same way, so the inline distribution bar and the list chips for the
// same replica use the same hue. Keep array length and order in sync.
const REPLICA_BAR_PALETTE = [
  "bg-emerald-500",
  "bg-sky-500",
  "bg-violet-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-teal-500",
] as const;

function replicaPaletteIndex(
  replica: string,
  lookup: Map<string, number>,
): number {
  let idx = lookup.get(replica);
  if (idx === undefined) {
    idx = lookup.size;
    lookup.set(replica, idx);
  }
  return idx;
}

function replicaColorClass(replica: string, lookup: Map<string, number>): string {
  const idx = replicaPaletteIndex(replica, lookup);
  return REPLICA_PALETTE[idx % REPLICA_PALETTE.length];
}

function replicaBarColorClass(
  replica: string,
  lookup: Map<string, number>,
): string {
  const idx = replicaPaletteIndex(replica, lookup);
  return REPLICA_BAR_PALETTE[idx % REPLICA_BAR_PALETTE.length];
}

export function FiveMTopupCard() {
  const { data, isLoading, isError } = useQuery<FiveMTopupStatus>({
    queryKey: ["5m-topup-status"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/brain/5m-topup-status`);
      if (!res.ok) throw new Error(`5m-topup-status ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  // Task #435 — fetch the recent-winners history alongside the status.
  // Kept as a separate query so a transient failure on either endpoint
  // doesn't blank the other half of the card.
  const recent = useQuery<RecentWinnersResponse>({
    queryKey: ["5m-topup-recent-winners"],
    queryFn: async () => {
      const res = await fetch(
        `${apiBase}/crypto/brain/5m-topup-recent-winners?limit=14`,
      );
      if (!res.ok) throw new Error(`5m-topup-recent-winners ${res.status}`);
      return res.json();
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  const replica = data?.last_winning_replica ?? null;
  const winningAt = data?.last_winning_at ?? null;
  const outcome = data?.last_attempt_outcome ?? null;
  const skipsLocked = data?.skips_locked_total ?? 0;
  const alerts = data?.last_alerts ?? [];
  const hasAlerts = alerts.length > 0;
  const haveEverWon = Boolean(replica && winningAt);

  const recentList: RecentWinner[] = recent.data?.winners ?? [];
  // Build a per-replica color lookup that's stable for the lifetime of
  // this render, then summarise so the header can say e.g. "host-a 9
  // of last 14, host-b 5 of last 14".
  // Task #441 — populate the lookup eagerly (in newest-first order) so
  // the distribution bar segments and the list chips agree on each
  // replica's hue regardless of which one renders first.
  const colorLookup = new Map<string, number>();
  const counts = new Map<string, number>();
  for (const w of recentList) {
    replicaPaletteIndex(w.replica, colorLookup);
    counts.set(w.replica, (counts.get(w.replica) ?? 0) + 1);
  }
  const distinctReplicas = counts.size;
  // Task #441 — distribution segments, biggest share first so the
  // dominant replica anchors the left edge and a "stuck box" stripe
  // is the obvious thing the eye lands on.
  const distribution = Array.from(counts.entries()).sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
  );
  // Flag a "stuck replica" suspicion when we have at least 5 entries
  // and one box owns >=80% of them. Mild heuristic — the actual
  // judgement call still belongs to the operator looking at the list.
  const stuckSuspected =
    recentList.length >= 5 &&
    Array.from(counts.values()).some((n) => n / recentList.length >= 0.8) &&
    distinctReplicas >= 1;

  return (
    <Card
      className={cn(
        "border-border/40 bg-card/30",
        hasAlerts && "ring-1 ring-amber-500/30 bg-amber-500/5",
      )}
      data-testid="five-m-topup-card"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Server className="w-4 h-4" />
          Daily 5m top-up
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            which ml-engine replica fetched the most recent pull
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-16 w-full" />}
        {isError && (
          <div
            className="text-sm text-rose-300 font-mono"
            data-testid="five-m-topup-card-error"
          >
            Couldn't load 5m top-up status.
          </div>
        )}
        {data && (
          <div className="space-y-3">
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">
                Today's pull
              </div>
              {haveEverWon ? (
                <div className="mt-1 flex items-center gap-2 flex-wrap">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400/80 shrink-0" />
                  <span
                    className="font-mono text-sm break-all"
                    data-testid="five-m-topup-winning-replica"
                  >
                    {replica}
                  </span>
                  <span
                    className="text-xs text-muted-foreground font-mono inline-flex items-center gap-1"
                    data-testid="five-m-topup-winning-at"
                  >
                    <Clock className="w-3 h-3" />
                    {formatTimeAgo(winningAt!)}
                    <span className="text-muted-foreground/60">
                      · {new Date(winningAt!).toLocaleString()}
                    </span>
                  </span>
                </div>
              ) : (
                <div
                  className="mt-1 text-sm font-mono text-muted-foreground"
                  data-testid="five-m-topup-no-winner"
                >
                  No replica has recorded a successful pull yet.
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div>
                <div className="text-[10px] uppercase text-muted-foreground">
                  This replica's last attempt
                </div>
                <div
                  className="mt-1"
                  data-testid="five-m-topup-last-outcome"
                >
                  {formatOutcome(outcome)}
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase text-muted-foreground">
                  Times this replica yielded
                </div>
                <div
                  className="mt-1"
                  data-testid="five-m-topup-skips-locked"
                >
                  {skipsLocked}
                </div>
              </div>
            </div>

            {/* Task #435 — recent-winners history so an operator can spot
                "host-a wins every day" vs "all three replicas take turns". */}
            <div data-testid="five-m-topup-recent-winners">
              <div className="text-[10px] uppercase font-mono text-muted-foreground flex items-center gap-2">
                <History className="w-3 h-3" />
                Recent pulls (newest first)
                {recentList.length > 0 && (
                  <span
                    className="text-muted-foreground/70 normal-case"
                    data-testid="five-m-topup-recent-summary"
                  >
                    · {recentList.length} entries · {distinctReplicas} distinct{" "}
                    {distinctReplicas === 1 ? "replica" : "replicas"}
                  </span>
                )}
              </div>
              {recent.isLoading && (
                <Skeleton className="h-12 w-full mt-1" />
              )}
              {recent.isError && (
                <div
                  className="mt-1 text-xs font-mono text-rose-300"
                  data-testid="five-m-topup-recent-error"
                >
                  Couldn't load recent winners.
                </div>
              )}
              {!recent.isLoading && !recent.isError && recentList.length === 0 && (
                <div
                  className="mt-1 text-xs font-mono text-muted-foreground"
                  data-testid="five-m-topup-recent-empty"
                >
                  No recent winners recorded yet.
                </div>
              )}
              {recentList.length > 0 && (
                <>
                  {/* Task #441 — inline stacked distribution bar so the
                      operator sees "one color owns the stripe" without
                      having to scan every row name. Each segment shares
                      its hue with the chip in the list below. */}
                  <div
                    className="mt-2 flex h-2 w-full overflow-hidden rounded-full ring-1 ring-border/40"
                    role="img"
                    aria-label={`Replica share over the last ${recentList.length} pulls: ${distribution
                      .map(
                        ([r, c]) =>
                          `${r} ${c} of ${recentList.length} (${Math.round(
                            (c / recentList.length) * 100,
                          )}%)`,
                      )
                      .join(", ")}`}
                    data-testid="five-m-topup-recent-distribution"
                  >
                    {distribution.map(([rep, count]) => {
                      const pct = (count / recentList.length) * 100;
                      return (
                        <div
                          key={rep}
                          className={cn(
                            "h-full",
                            replicaBarColorClass(rep, colorLookup),
                          )}
                          style={{ width: `${pct}%` }}
                          title={`${rep} · ${count} of ${recentList.length} (${pct.toFixed(0)}%)`}
                          data-testid="five-m-topup-recent-distribution-segment"
                          data-replica={rep}
                          data-count={count}
                        />
                      );
                    })}
                  </div>
                  {stuckSuspected && (
                    <div
                      className="mt-2 p-2 rounded-md bg-amber-500/10 ring-1 ring-amber-500/20 text-xs font-mono flex items-start gap-2"
                      data-testid="five-m-topup-stuck-warning"
                    >
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                      <span className="text-amber-100">
                        One replica is winning the lock disproportionately
                        often — clocks may have drifted on the others.
                      </span>
                    </div>
                  )}
                  <ul
                    className="mt-2 space-y-1 max-h-48 overflow-auto pr-1"
                    data-testid="five-m-topup-recent-list"
                  >
                    {recentList.map((w, idx) => (
                      <li
                        key={`${w.tick_at}-${idx}`}
                        className="flex items-center gap-2 text-xs font-mono"
                        data-testid="five-m-topup-recent-item"
                      >
                        <span className="text-muted-foreground/60 w-5 text-right shrink-0">
                          {idx + 1}.
                        </span>
                        <span
                          className={cn(
                            "px-1.5 py-0.5 rounded ring-1 break-all",
                            replicaColorClass(w.replica, colorLookup),
                          )}
                          data-testid="five-m-topup-recent-replica"
                        >
                          {w.replica}
                        </span>
                        <span
                          className="text-muted-foreground"
                          data-testid="five-m-topup-recent-when"
                        >
                          {formatTimeAgo(w.tick_at)}
                        </span>
                        <span className="text-muted-foreground/60">
                          · {new Date(w.tick_at).toLocaleString()}
                        </span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>

            {hasAlerts && (
              <div
                className="p-2 rounded-md bg-amber-500/10 ring-1 ring-amber-500/20 text-xs font-mono flex items-start gap-2"
                data-testid="five-m-topup-alerts"
              >
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <div className="text-[10px] uppercase text-amber-200/80">
                    Coins under the contiguous-days threshold
                  </div>
                  <div className="text-amber-100">{alerts.join(", ")}</div>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
