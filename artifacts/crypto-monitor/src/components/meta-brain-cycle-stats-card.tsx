import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Brain } from "lucide-react";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface CycleStats {
  sliceCount: number;
  bindings: number;
  cacheHits: number;
  cacheMisses: number;
  suppressedTick: number;
  defensiveOff: number;
  defensiveSoft: number;
  defensiveHard: number;
  fallbacksByCause: Record<string, number>;
  evalLatencyP50Ms: number;
  evalLatencyP95Ms: number;
  evalCount: number;
  timeoutCount: number;
  activeTickId: string;
  activeDefensiveMode: "off" | "soft" | "hard";
  activeSuppressedFamilies: string[];
  activePausedSlices: string[];
  emittedAt: string;
}

interface BenchmarkBlock {
  aiReturn7d: number;
  bestBaselineReturn7d: number;
  relativeAlpha7d: number;
  relativeAlpha14d: number;
  drawdownRatioVsBest: number;
  sustainedUnderperformance: boolean;
  sampleCount: number;
  stale: boolean;
  /** Brain's currently-learned trust weight on the synthetic
   * `benchmark` family — folded in by the cycle-stats route from
   * `mlEngine.stats.trust_by_family.benchmark.trust`. */
  trustWeight?: number | null;
}

interface CycleStatsResponse {
  mode: "live" | "shadow" | "off";
  enabled: boolean;
  shadow: boolean;
  last: CycleStats | null;
  history: CycleStats[];
  mlEngine: {
    health: { ok?: boolean; tick_cache_size?: number } | null;
    stats: {
      ok?: boolean;
      trust_by_family?: Record<string, { trust: number; stability: number }>;
      benchmark_trust?: { trust: number; stability: number } | null;
    } | null;
  };
  benchmark: BenchmarkBlock | null;
  fetchedAt: string;
}

function fmtPct(x: number): string {
  return `${(x * 100).toFixed(2)}%`;
}

function modeBadgeClass(mode: CycleStatsResponse["mode"]) {
  if (mode === "live") return "bg-emerald-500/15 text-emerald-200 ring-emerald-500/40";
  if (mode === "shadow") return "bg-amber-500/15 text-amber-200 ring-amber-500/40";
  return "bg-zinc-700/30 text-zinc-300 ring-zinc-600/40";
}

function defensiveBadgeClass(mode: CycleStats["activeDefensiveMode"]) {
  if (mode === "hard") return "bg-rose-500/15 text-rose-200 ring-rose-500/40";
  if (mode === "soft") return "bg-amber-500/15 text-amber-200 ring-amber-500/40";
  return "bg-emerald-500/15 text-emerald-200 ring-emerald-500/40";
}

function tickIdSummary(tickId: string): { label: string; tone: string } {
  if (tickId.startsWith("neutral:")) return { label: "neutral (no directive)", tone: "text-zinc-400" };
  if (tickId.startsWith("shadow:")) return { label: "shadow (observability only)", tone: "text-amber-300" };
  return { label: "live directive", tone: "text-emerald-300" };
}

export function MetaBrainCycleStatsCard() {
  const { data, isLoading, isError } = useQuery<CycleStatsResponse>({
    queryKey: ["meta-brain-cycle-stats"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/meta-brain/cycle-stats`);
      if (!res.ok) throw new Error(`meta-brain cycle stats ${res.status}`);
      return res.json();
    },
    refetchInterval: 15_000,
    staleTime: 10_000,
  });

  return (
    <Card className="bg-card/50 border-border/40" data-testid="meta-brain-cycle-stats-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Brain className="w-4 h-4" />
          Meta-Brain Cycle Stats
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            per-cycle telemetry from the governance brain · refreshes every 15s
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-40 w-full" />}
        {isError && (
          <div className="text-sm text-rose-300 font-mono" data-testid="meta-brain-cycle-stats-error">
            Couldn't load meta-brain cycle stats.
          </div>
        )}
        {data && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge className={`ring-1 ${modeBadgeClass(data.mode)}`} data-testid="meta-brain-mode">
                mode: {data.mode}
              </Badge>
              {data.last ? (
                <>
                  <Badge
                    className={`ring-1 ${defensiveBadgeClass(data.last.activeDefensiveMode)}`}
                    data-testid="meta-brain-defensive-mode"
                  >
                    defensive: {data.last.activeDefensiveMode}
                  </Badge>
                  <span
                    className={`text-xs font-mono ${tickIdSummary(data.last.activeTickId).tone}`}
                    data-testid="meta-brain-tick-summary"
                  >
                    {tickIdSummary(data.last.activeTickId).label}
                  </span>
                </>
              ) : (
                <span className="text-xs text-muted-foreground">
                  no cycle has emitted stats yet (waiting for first flush)
                </span>
              )}
            </div>

            {data.last && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                <Stat label="slices/cycle" value={data.last.sliceCount} />
                <Stat label="open bindings" value={data.last.bindings} />
                <Stat label="cache hits" value={data.last.cacheHits} />
                <Stat label="cache misses" value={data.last.cacheMisses} />
                <Stat label="eval p50 (ms)" value={data.last.evalLatencyP50Ms} />
                <Stat label="eval p95 (ms)" value={data.last.evalLatencyP95Ms} />
                <Stat label="eval count" value={data.last.evalCount} />
                <Stat
                  label="timeouts"
                  value={data.last.timeoutCount}
                  warn={data.last.timeoutCount > 0}
                />
              </div>
            )}

            {data.last && (
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70">
                  fallbacks (last cycle)
                </div>
                {Object.keys(data.last.fallbacksByCause).length === 0 ? (
                  <div className="text-xs text-emerald-300 font-mono">
                    none — every evaluate call returned a directive
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(data.last.fallbacksByCause).map(([cause, n]) => (
                      <Badge
                        key={cause}
                        className="bg-amber-500/10 text-amber-200 ring-1 ring-amber-500/30"
                        data-testid={`meta-brain-fallback-${cause}`}
                      >
                        {cause}: {n}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            )}

            {data.last &&
              (data.last.activeSuppressedFamilies.length > 0 ||
                data.last.activePausedSlices.length > 0) && (
                <div className="space-y-1">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70">
                    active throttles
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs font-mono">
                    {data.last.activeSuppressedFamilies.map((f) => (
                      <Badge
                        key={`fam-${f}`}
                        className="bg-rose-500/10 text-rose-200 ring-1 ring-rose-500/30"
                      >
                        suppressed: {f}
                      </Badge>
                    ))}
                    {data.last.activePausedSlices.map((s) => (
                      <Badge
                        key={`slice-${s}`}
                        className="bg-rose-500/10 text-rose-200 ring-1 ring-rose-500/30"
                      >
                        paused: {s}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

            {data.benchmark && (
              <div className="space-y-1" data-testid="meta-brain-benchmark">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 flex items-center gap-2">
                  <span>Benchmark vs. baselines (Strategy Lab — observability only)</span>
                  {data.benchmark.stale && (
                    <Badge
                      variant="outline"
                      className="bg-muted/40 text-muted-foreground ring-1 ring-muted-foreground/20 text-[9px]"
                      data-testid="meta-brain-benchmark-stale"
                    >
                      stale
                    </Badge>
                  )}
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                  <Stat label="AI 7d" value={fmtPct(data.benchmark.aiReturn7d)} />
                  <Stat
                    label="best baseline 7d"
                    value={fmtPct(data.benchmark.bestBaselineReturn7d)}
                  />
                  <Stat
                    label="alpha 7d"
                    value={fmtPct(data.benchmark.relativeAlpha7d)}
                    warn={!data.benchmark.stale && data.benchmark.relativeAlpha7d < 0}
                  />
                  <Stat
                    label="alpha 14d"
                    value={fmtPct(data.benchmark.relativeAlpha14d)}
                    warn={!data.benchmark.stale && data.benchmark.relativeAlpha14d < 0}
                  />
                  <Stat
                    label="dd ratio vs best"
                    value={data.benchmark.drawdownRatioVsBest.toFixed(2)}
                    warn={!data.benchmark.stale && data.benchmark.drawdownRatioVsBest > 1.25}
                  />
                  <Stat label="samples" value={data.benchmark.sampleCount} />
                  {typeof data.benchmark.trustWeight === "number" && (
                    <Stat
                      label="benchmark trust"
                      value={data.benchmark.trustWeight.toFixed(3)}
                    />
                  )}
                </div>
                {data.benchmark.stale ? (
                  <div
                    className="text-xs text-muted-foreground font-mono"
                    data-testid="meta-brain-benchmark-stale-note"
                  >
                    benchmark stale — brain falls back to neutral handling
                  </div>
                ) : (
                  data.benchmark.sustainedUnderperformance && (
                    <div
                      className="text-xs text-amber-300 font-mono"
                      data-testid="meta-brain-benchmark-warn"
                    >
                      sustained underperformance — defensive_mode may be soft-clamped
                    </div>
                  )
                )}
              </div>
            )}

            {data.mlEngine.stats?.trust_by_family && (
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70">
                  trust by family (ml-engine)
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 text-xs font-mono">
                  {Object.entries(data.mlEngine.stats.trust_by_family).map(([fam, v]) => (
                    <div
                      key={fam}
                      className="flex items-center justify-between rounded px-2 py-1 bg-muted/30"
                      data-testid={`meta-brain-trust-${fam}`}
                    >
                      <span className="text-muted-foreground">{fam}</span>
                      <span>
                        trust {v.trust.toFixed(3)} · stability {v.stability.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="text-[10px] text-muted-foreground/70 font-mono">
              {data.history.length} cycle(s) buffered ·{" "}
              {data.last ? `last emitted ${new Date(data.last.emittedAt).toLocaleTimeString()}` : ""}
              {" · "}fetched {new Date(data.fetchedAt).toLocaleTimeString()}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({
  label,
  value,
  warn,
}: {
  label: string;
  value: number | string;
  warn?: boolean;
}) {
  return (
    <div className="rounded-md bg-muted/30 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70">
        {label}
      </div>
      <div className={warn ? "text-rose-300" : "text-foreground"}>{value}</div>
    </div>
  );
}
