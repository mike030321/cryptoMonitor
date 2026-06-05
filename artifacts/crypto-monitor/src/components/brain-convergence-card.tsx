import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RefreshCw, Activity, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface VerificationCounts {
  slices_promoted: number;
  slices_no_lift: number;
  slices_below_coinflip: number;
  slices_insufficient_sample: number;
  slices_contract_failed: number;
  slices_untrained: number;
}

interface VerificationDiff {
  status: string;
  delta_promoted: number;
  stall_streak: number;
  prev_promoted: number;
  prev_passed: boolean;
  newly_promoted_coins: string[];
  newly_demoted_coins: string[];
}

interface VerificationRow {
  recorded_at: number;
  source_report_started_at?: string | null;
  source_report_completed_at?: string | null;
  verification_status: string;
  passed: boolean;
  active_coins: string[];
  coins_with_promotion: string[];
  coins_without_promotion: string[];
  promoted_by_coin: Record<string, number>;
  counts: VerificationCounts;
  diff: VerificationDiff;
  stall_streak: number;
}

interface VerificationHistoryResp {
  rows: VerificationRow[];
  count: number;
  error?: string;
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

const STATUS_TONE: Record<string, string> = {
  converged: "border-emerald-500/40 text-emerald-300",
  improving: "border-sky-500/40 text-sky-300",
  regressed: "border-rose-500/40 text-rose-300",
  flat: "border-zinc-500/40 text-zinc-300",
  stalled: "border-amber-500/40 text-amber-300",
  first_run: "border-zinc-500/40 text-zinc-300",
  no_verification: "border-zinc-500/40 text-zinc-400",
};

function fmtTime(ts: number | null | undefined): string {
  if (ts == null) return "—";
  try {
    return new Date(Number(ts) * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

function fmtDelta(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  const v = Number(n);
  if (v === 0) return "0";
  return v > 0 ? `+${v}` : String(v);
}

function MiniLine({ values, color }: { values: number[]; color: string }) {
  if (values.length === 0) return <span className="text-[10px] text-muted-foreground">no data</span>;
  const w = 120;
  const h = 28;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const xStep = values.length > 1 ? w / (values.length - 1) : 0;
  const path = values
    .map((v, i) => {
      const x = values.length === 1 ? w / 2 : i * xStep;
      const y = h - ((v - min) / span) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="block">
      <path d={path} stroke={color} strokeWidth={1.4} fill="none" />
    </svg>
  );
}

export function BrainConvergenceCard() {
  const [data, setData] = useState<VerificationHistoryResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const res = await fetch(apiUrl(`/crypto/brain/verification-history`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = (await res.json()) as VerificationHistoryResp;
      // ml-engine returns 200 with `{rows: [], count: 0, error: "..."}`
      // when the watchdog read fails. Surface that as an error rather
      // than letting it masquerade as a clean "no history yet" state.
      if (json.error) {
        setData(null);
        setErr(json.error);
        return;
      }
      setData(json);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
    const id = setInterval(() => void refresh(), 30_000);
    return () => clearInterval(id);
  }, []);

  // History endpoint returns newest-first. For sparklines we want
  // chronological order (oldest -> newest) so the line reads left-to-right.
  const rowsNewestFirst = data?.rows ?? [];
  const last10NewestFirst = rowsNewestFirst.slice(0, 10);
  const chronological = useMemo(
    () => last10NewestFirst.slice().reverse(),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data],
  );

  const latest = rowsNewestFirst[0];
  const passed = latest?.passed ?? false;
  const stallStreak = latest?.stall_streak ?? 0;
  const stalledChip = stallStreak >= 3;

  const promotedCount = latest?.coins_with_promotion?.length ?? 0;
  const activeCount = latest?.active_coins?.length ?? 0;
  const newlyPromoted = latest?.diff?.newly_promoted_coins ?? [];
  const newlyDemoted = latest?.diff?.newly_demoted_coins ?? [];

  const promotedSeries = chronological.map(
    (r) => r.counts?.slices_promoted ?? 0,
  );

  return (
    <Card data-testid="brain-convergence-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Brain convergence
          {data && (
            <Badge variant="outline" className="ml-2 text-[10px]">
              last {data.count} run{data.count === 1 ? "" : "s"}
            </Badge>
          )}
        </CardTitle>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => void refresh()}
          disabled={loading}
          aria-label="Refresh brain convergence"
          data-testid="brain-convergence-refresh"
        >
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
        </Button>
      </CardHeader>

      <CardContent className="space-y-4">
        {err && (
          <div
            className="text-xs text-rose-400"
            data-testid="brain-convergence-error"
          >
            Error: {err}
          </div>
        )}

        {!err && (data?.count ?? 0) === 0 && (
          <div
            className="text-xs text-muted-foreground"
            data-testid="brain-convergence-empty"
          >
            No retrain history yet. The watchdog records a snapshot after the
            first training run that produces a verification block.
          </div>
        )}

        {latest && (
          <div className="flex flex-wrap items-center gap-2">
            <Badge
              className={cn(
                "text-xs px-3 py-1 inline-flex items-center gap-1",
                passed
                  ? "bg-emerald-600/30 text-emerald-200 border-emerald-500/40"
                  : "bg-rose-600/30 text-rose-200 border-rose-500/40",
              )}
              variant="outline"
              data-testid="brain-convergence-passed-badge"
            >
              {passed ? (
                <CheckCircle2 className="h-3.5 w-3.5" />
              ) : (
                <XCircle className="h-3.5 w-3.5" />
              )}
              {passed ? "verification passed" : "verification failed"}
            </Badge>
            <Badge
              variant="outline"
              className={cn(
                "text-[10px]",
                STATUS_TONE[latest.diff?.status ?? ""] ??
                  "border-zinc-500/40 text-zinc-300",
              )}
              data-testid="brain-convergence-status-badge"
            >
              {latest.diff?.status ?? "—"}
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              promoted {promotedCount}/{activeCount} coins
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              Δ promoted {fmtDelta(latest.diff?.delta_promoted)}
            </Badge>
            {stalledChip && (
              <Badge
                variant="outline"
                className="text-[10px] border-amber-500/50 text-amber-300 inline-flex items-center gap-1"
                data-testid="brain-convergence-stall-chip"
              >
                <AlertTriangle className="h-3 w-3" />
                stalled for {stallStreak} runs
              </Badge>
            )}
            <span className="ml-auto text-[10px] text-muted-foreground">
              last run {fmtTime(latest.recorded_at)}
            </span>
          </div>
        )}

        {latest && (newlyPromoted.length > 0 || newlyDemoted.length > 0) && (
          <div className="text-[11px] flex flex-wrap gap-x-4 gap-y-1">
            {newlyPromoted.length > 0 && (
              <div data-testid="brain-convergence-newly-promoted">
                <span className="text-muted-foreground">newly promoted:</span>{" "}
                <span className="text-emerald-300 font-mono">
                  {newlyPromoted.join(", ")}
                </span>
              </div>
            )}
            {newlyDemoted.length > 0 && (
              <div data-testid="brain-convergence-newly-demoted">
                <span className="text-muted-foreground">newly demoted:</span>{" "}
                <span className="text-rose-300 font-mono">
                  {newlyDemoted.join(", ")}
                </span>
              </div>
            )}
          </div>
        )}

        {chronological.length > 0 && (
          <div className="flex items-center gap-3 text-[11px]">
            <span className="text-muted-foreground">slices promoted</span>
            <MiniLine values={promotedSeries} color="#34d399" />
            <span className="text-muted-foreground tabular-nums">
              {promotedSeries[0]} → {promotedSeries[promotedSeries.length - 1]}
            </span>
          </div>
        )}

        {last10NewestFirst.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-3">When</th>
                  <th className="py-1 pr-3">Status</th>
                  <th className="py-1 pr-3 text-right">Promoted</th>
                  <th className="py-1 pr-3 text-right">Δ</th>
                  <th className="py-1 pr-3 text-right">Stall</th>
                </tr>
              </thead>
              <tbody>
                {last10NewestFirst.map((r, i) => (
                  <tr
                    key={`${r.recorded_at}-${i}`}
                    className="border-t border-border/30"
                    data-testid={`brain-convergence-row-${i}`}
                  >
                    <td className="py-1 pr-3 text-muted-foreground">
                      {fmtTime(r.recorded_at)}
                    </td>
                    <td className="py-1 pr-3">
                      <Badge
                        variant="outline"
                        className={cn(
                          "text-[10px]",
                          STATUS_TONE[r.diff?.status ?? ""] ??
                            "border-zinc-500/40 text-zinc-300",
                        )}
                      >
                        {r.diff?.status ?? "—"}
                      </Badge>
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {r.counts?.slices_promoted ?? 0}
                    </td>
                    <td
                      className={cn(
                        "py-1 pr-3 text-right tabular-nums",
                        (r.diff?.delta_promoted ?? 0) > 0 && "text-emerald-300",
                        (r.diff?.delta_promoted ?? 0) < 0 && "text-rose-300",
                      )}
                    >
                      {fmtDelta(r.diff?.delta_promoted)}
                    </td>
                    <td
                      className={cn(
                        "py-1 pr-3 text-right tabular-nums",
                        (r.stall_streak ?? 0) >= 3 && "text-amber-300",
                      )}
                    >
                      {r.stall_streak ?? 0}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="text-[10px] text-muted-foreground">
          One row per retrain. <span className="font-mono">converged</span>{" "}
          means verification.passed=true; <span className="font-mono">improving</span>/
          <span className="font-mono">regressed</span> compares slices_promoted
          vs the previous run; <span className="font-mono">flat</span> for 3+
          runs in a row escalates to{" "}
          <span className="font-mono">stalled</span>.
        </div>
      </CardContent>
    </Card>
  );
}
