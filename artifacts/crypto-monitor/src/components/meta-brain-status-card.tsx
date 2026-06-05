/**
 * Task #532 / C-11 — Meta-brain learning state card on /diagnostics.
 *
 * Reads `/api/crypto/meta-brain/status` (the new ml-engine proxy)
 * and renders three honest answers:
 *   1. Is the ml-engine /ml/meta-brain/health endpoint reachable AND
 *      reporting ok=true? Pill: green online / amber reachable / red
 *      unreachable.
 *   2. Last replay attempt: timestamp + outcome (skipped /
 *      promoted / rolled_back) + reason if skipped.
 *   3. Cycle stats: number of decisions seen, abstain rate, current
 *      family trust multipliers.
 *
 * Every field gracefully renders "n/a" when the proxy returns null
 * for that sub-call rather than collapsing the card. No fake zeros.
 */
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Brain } from "lucide-react";
import { formatTimeAgo } from "@/lib/format";

interface LearningTruth {
  lastEvaluateAt: string | null;
  lastRecordOutcomeAt: string | null;
  trustStateChanges24h: number;
  closedTrades24h: number;
  lastDirective: Record<string, unknown> | null;
  activityNote: string | null;
}

interface MetaBrainStatus {
  available: boolean;
  reachable: boolean;
  mlEngineUrl: string;
  health: Record<string, unknown> | null;
  stats: Record<string, unknown> | null;
  lastReplay: Record<string, unknown> | null;
  learningTruth?: LearningTruth;
  fetchedAt: string;
}

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

function pickStr(o: Record<string, unknown> | null | undefined, k: string): string | null {
  if (!o) return null;
  const v = o[k];
  return typeof v === "string" ? v : null;
}
function pickNum(o: Record<string, unknown> | null | undefined, k: string): number | null {
  if (!o) return null;
  const v = o[k];
  return typeof v === "number" ? v : null;
}
function pickBool(o: Record<string, unknown> | null | undefined, k: string): boolean | null {
  if (!o) return null;
  const v = o[k];
  return typeof v === "boolean" ? v : null;
}

export function MetaBrainStatusCard() {
  const q = useQuery<MetaBrainStatus>({
    queryKey: ["meta-brain-status"],
    queryFn: async () => {
      const r = await fetch(`${apiBase}/crypto/meta-brain/status`);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    },
    refetchInterval: 30_000,
  });

  if (q.isLoading) {
    return <Skeleton className="h-40 w-full" data-testid="meta-brain-status-card-loading" />;
  }
  const d = q.data;
  if (!d) {
    return (
      <Card data-testid="meta-brain-status-card">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <Brain className="w-4 h-4" /> Meta-Brain Learning State
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">n/a — ml-engine status proxy unreachable.</p>
        </CardContent>
      </Card>
    );
  }

  const pill = !d.reachable
    ? <Badge className="bg-red-600 text-white">unreachable</Badge>
    : d.available
      ? <Badge className="bg-emerald-600 text-white">online</Badge>
      : <Badge className="bg-amber-600 text-white">reachable, not ok</Badge>;

  // Task #532 / Rev 2.1 — ml-engine `/last_replay` returns a nested
  // shape: { ok, last_run: {...}, last_committed_run: {...} }. Prefer
  // `last_committed_run` (the most recent promotion) and fall back to
  // `last_run` (the most recent attempt, which may have been gated by
  // thresholds_not_met). Only fall back to flat-shape lookups for
  // forward/backward compat with older payloads.
  const lastRun = (d.lastReplay && typeof d.lastReplay === "object")
    ? ((d.lastReplay as Record<string, unknown>).last_committed_run as Record<string, unknown> | null
        ?? (d.lastReplay as Record<string, unknown>).last_run as Record<string, unknown> | null
        ?? null)
    : null;
  const commitDetails = lastRun
    ? ((lastRun.commit_details as Record<string, unknown>) ?? null)
    : null;
  const lastReplayAt =
    pickStr(lastRun, "finished_at") ??
    pickStr(lastRun, "started_at") ??
    pickStr(d.lastReplay, "ts") ??
    pickStr(d.lastReplay, "timestamp");
  // Outcome: prefer the explicit promotion flag from commit_details,
  // then fall back to a free-form outcome/decision string.
  const promotedFlag = commitDetails ? pickBool(commitDetails, "promoted") : null;
  const committedFlag = lastRun ? pickBool(lastRun, "commit") : null;
  const lastReplayOutcome =
    promotedFlag === true || committedFlag === true
      ? "promoted"
      : promotedFlag === false || committedFlag === false
        ? "thresholds_not_met"
        : pickStr(d.lastReplay, "outcome") ?? pickStr(d.lastReplay, "decision");
  const lastReplayReason =
    pickStr(commitDetails, "reason") ??
    pickStr(lastRun, "reason") ??
    pickStr(d.lastReplay, "reason") ??
    pickStr(d.lastReplay, "skipReason");
  const decisionsSeen = pickNum(d.stats, "decisions") ?? pickNum(d.stats, "totalDecisions");
  const abstainRate = pickNum(d.stats, "abstainRate") ?? pickNum(d.stats, "abstain_rate");
  const eligibleNow = pickBool(d.stats, "eligibleForReplay") ?? pickBool(d.stats, "eligible_for_replay");
  // Task #532 / C-11 — operator-truth fields. Rendered as "n/a" when
  // the proxy returned null rather than fake-zero them.
  const lt = d.learningTruth;
  const lastEvaluateAt = lt?.lastEvaluateAt ?? null;
  const lastRecordOutcomeAt = lt?.lastRecordOutcomeAt ?? null;
  const trustChanges24h = lt?.trustStateChanges24h ?? null;
  const closedTrades24h = lt?.closedTrades24h ?? null;
  const lastDirective = lt?.lastDirective ?? null;
  const activityNote = lt?.activityNote ?? null;
  const directiveCaution = pickNum(lastDirective, "caution_level");
  const directiveDefensive = pickStr(lastDirective, "defensive_mode");
  const directiveReasonsRaw = lastDirective?.["reason_codes"];
  const directiveReasons = Array.isArray(directiveReasonsRaw)
    ? directiveReasonsRaw.filter((x): x is string => typeof x === "string")
    : [];

  return (
    <Card data-testid="meta-brain-status-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Brain className="w-4 h-4" /> Meta-Brain Learning State
          <span className="ml-auto">{pill}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-[10px] uppercase font-mono text-muted-foreground">ml-engine</div>
            <div className="font-mono text-xs mt-1 break-all" data-testid="meta-brain-status-url">
              {d.mlEngineUrl}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Decisions seen</div>
            <div className="font-mono text-base mt-1" data-testid="meta-brain-status-decisions">
              {decisionsSeen !== null ? decisionsSeen.toLocaleString() : "n/a"}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono text-muted-foreground">Abstain rate</div>
            <div className="font-mono text-base mt-1" data-testid="meta-brain-status-abstain">
              {abstainRate !== null ? `${(abstainRate * 100).toFixed(1)}%` : "n/a"}
            </div>
          </div>
        </div>
        <div className="border-t border-border/30 pt-3">
          <div className="text-[10px] uppercase font-mono text-muted-foreground mb-1">Last replay</div>
          <div className="text-sm" data-testid="meta-brain-status-last-replay">
            {lastReplayAt ? (
              <>
                <span className="font-mono text-xs text-muted-foreground">{formatTimeAgo(lastReplayAt)}</span>
                {lastReplayOutcome && (
                  <Badge
                    variant="outline"
                    className={`ml-2 text-[10px] uppercase ${
                      lastReplayOutcome === "promoted"
                        ? "border-emerald-500/40 text-emerald-300"
                        : lastReplayOutcome === "rolled_back"
                          ? "border-red-500/40 text-red-300"
                          : "border-amber-500/40 text-amber-300"
                    }`}
                  >
                    {lastReplayOutcome}
                  </Badge>
                )}
                {lastReplayReason && (
                  <div className="text-xs text-muted-foreground mt-1">{lastReplayReason}</div>
                )}
              </>
            ) : (
              <span className="text-muted-foreground">n/a — ml-engine returned no replay history.</span>
            )}
          </div>
          {eligibleNow !== null && (
            <div className="text-xs text-muted-foreground mt-2">
              Eligible to replay now:{" "}
              <span className={eligibleNow ? "text-emerald-300" : "text-amber-300"}>
                {eligibleNow ? "yes" : "no"}
              </span>
            </div>
          )}
        </div>
        {/* Task #532 / C-11 — operator-truth section. Shows last
            evaluate, last record-outcome, trust-state changes over
            24h, last directive, and an explicit zero-activity note
            so the operator never has to guess why the counters are
            zero. */}
        <div className="border-t border-border/30 pt-3 space-y-2" data-testid="meta-brain-learning-truth">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">Last evaluate</div>
              <div className="font-mono text-xs mt-1" data-testid="meta-brain-last-evaluate">
                {lastEvaluateAt ? formatTimeAgo(lastEvaluateAt) : "n/a"}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">Last record-outcome</div>
              <div className="font-mono text-xs mt-1" data-testid="meta-brain-last-record-outcome">
                {lastRecordOutcomeAt ? formatTimeAgo(lastRecordOutcomeAt) : "n/a"}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">Trust changes 24h</div>
              <div className="font-mono text-base mt-1" data-testid="meta-brain-trust-changes-24h">
                {trustChanges24h !== null ? trustChanges24h.toLocaleString() : "n/a"}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-muted-foreground">Closed trades 24h</div>
              <div className="font-mono text-base mt-1" data-testid="meta-brain-closed-trades-24h">
                {closedTrades24h !== null ? closedTrades24h.toLocaleString() : "n/a"}
              </div>
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono text-muted-foreground mb-1">Last directive</div>
            {lastDirective ? (
              <div className="text-xs font-mono text-muted-foreground" data-testid="meta-brain-last-directive">
                {directiveDefensive && (
                  <span className="mr-2">defensive_mode: <span className="text-foreground">{directiveDefensive}</span></span>
                )}
                {directiveCaution !== null && (
                  <span className="mr-2">caution: <span className="text-foreground">{directiveCaution.toFixed(2)}</span></span>
                )}
                {directiveReasons.length > 0 && (
                  <span>reason_codes: <span className="text-foreground">{directiveReasons.slice(0, 4).join(", ")}</span></span>
                )}
                {!directiveDefensive && directiveCaution === null && directiveReasons.length === 0 && (
                  <span>directive captured but with no reason_codes / caution / defensive_mode fields.</span>
                )}
              </div>
            ) : (
              <span className="text-xs text-muted-foreground" data-testid="meta-brain-last-directive">
                n/a — no directive surfaced by the most recent replay run.
              </span>
            )}
          </div>
          {activityNote && (
            <div
              className="text-xs text-amber-300/90 border-l-2 border-amber-500/40 pl-2 mt-2"
              data-testid="meta-brain-activity-note"
            >
              {activityNote}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
