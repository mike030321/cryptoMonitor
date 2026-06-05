import { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Activity, Beaker, ShieldAlert, TrendingUp, RefreshCw } from "lucide-react";
import { useAdminKey } from "@/hooks/use-admin-key";
import { toast } from "@/hooks/use-toast";
import { ToastAction } from "@/components/ui/toast";

const FEATURE_LAB_CARD_ID = "feature-lab-card-anchor";

function scrollToFeatureLab() {
  const el = document.getElementById(FEATURE_LAB_CARD_ID);
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function pickWorstTimeframe(
  tfs: QuarantinedTimeframeDetail[] | undefined,
): QuarantinedTimeframeDetail | null {
  if (!tfs || tfs.length === 0) return null;
  let worst: QuarantinedTimeframeDetail | null = null;
  for (const t of tfs) {
    const d = t.delta_log_loss;
    if (d === null || d === undefined || !Number.isFinite(d)) continue;
    if (!worst || (worst.delta_log_loss ?? -Infinity) < d) {
      worst = t;
    }
  }
  return worst ?? tfs[0];
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

function fmtUsd(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}$${n.toFixed(2)}`;
}
function fmtPct(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}
function fmtNum(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

interface PnlBucket {
  regime: string;
  modelVersion: string;
  coinId: string;
  timeframe: string;
  nTrades: number;
  pnlUsd: number;
  winRate: number;
  avgPnlPct: number;
}
interface PnlBreakdown {
  windowHours: number;
  generatedAt: string;
  totals: { nTrades: number; pnlUsd: number; winRate: number };
  buckets: PnlBucket[];
}

export function PnlBreakdownCard() {
  const [data, setData] = useState<PnlBreakdown | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [windowHours, setWindowHours] = useState(72);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(apiUrl(`/crypto/diagnostics/pnl-breakdown?windowHours=${windowHours}`));
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "load failed");
    } finally {
      setLoading(false);
    }
  }, [windowHours]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 60_000);
    return () => clearInterval(id);
  }, [fetchData]);

  return (
    <Card data-testid="pnl-breakdown-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-emerald-400" />PnL Breakdown by Regime / Model / Coin / Timeframe</span>
          <span className="flex items-center gap-2">
            <Input type="number" min={1} max={720} value={windowHours} onChange={(e) => setWindowHours(Math.max(1, Math.min(720, Number(e.target.value) || 72)))} className="w-20 h-8" data-testid="pnl-window-input" />
            <span className="text-xs text-muted-foreground">hrs</span>
            <Button variant="ghost" size="sm" onClick={fetchData} disabled={loading} data-testid="pnl-refresh">
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {error && <div className="text-rose-400 text-sm mb-2">Error: {error}</div>}
        {data && (
          <>
            <div className="flex flex-wrap gap-3 mb-3 text-sm">
              <Badge variant="outline" data-testid="pnl-total-trades">Trades: {data.totals.nTrades}</Badge>
              <Badge variant="outline" className={data.totals.pnlUsd >= 0 ? "text-emerald-400" : "text-rose-400"} data-testid="pnl-total-usd">PnL: {fmtUsd(data.totals.pnlUsd)}</Badge>
              <Badge variant="outline" data-testid="pnl-total-winrate">Win rate: {fmtPct(data.totals.winRate)}</Badge>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="text-left py-1">Regime</th>
                    <th className="text-left">Model</th>
                    <th className="text-left">Coin</th>
                    <th className="text-left">TF</th>
                    <th className="text-right">N</th>
                    <th className="text-right">PnL</th>
                    <th className="text-right">Win%</th>
                    <th className="text-right">Avg%</th>
                  </tr>
                </thead>
                <tbody data-testid="pnl-buckets-tbody">
                  {data.buckets.slice(0, 30).map((b, i) => (
                    <tr key={i} className="border-t border-border/30">
                      <td className="py-1">{b.regime}</td>
                      <td className="font-mono">{b.modelVersion}</td>
                      <td>{b.coinId}</td>
                      <td>{b.timeframe}</td>
                      <td className="text-right">{b.nTrades}</td>
                      <td className={`text-right ${b.pnlUsd >= 0 ? "text-emerald-400" : "text-rose-400"}`}>{fmtUsd(b.pnlUsd)}</td>
                      <td className="text-right">{fmtPct(b.winRate)}</td>
                      <td className="text-right">{(b.avgPnlPct * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                  {data.buckets.length === 0 && (
                    <tr><td colSpan={8} className="text-center text-muted-foreground py-3">No closed trades in window</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

interface DriftSnapshot {
  windowHours: number;
  registryId: number | null;
  calibration: {
    score: number;
    threshold: number;
    breached: boolean;
    nSamples: number;
    buckets: Array<{ bucket: number; meanConfidence: number; empiricalAccuracy: number; count: number }>;
  };
  distribution: {
    score: number;
    threshold: number;
    breached: boolean;
    nSamples: number;
    recent: Record<string, number>;
    baseline: Record<string, number>;
  };
  feature: {
    score: number;
    threshold: number;
    breached: boolean;
    nSamples: number;
    perFeature: Array<{ name: string; recentMean: number; baselineMean: number; recentStd: number; baselineStd: number; zScore: number }>;
  };
  history: { calibration: Array<unknown>; feature: Array<unknown> };
}

export function DriftTrackerCard() {
  const [data, setData] = useState<DriftSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(apiUrl("/crypto/diagnostics/drift?windowHours=24"));
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "load failed");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 60_000);
    return () => clearInterval(id);
  }, [fetchData]);

  function tone(v: number, warn: number, bad: number): string {
    if (v >= bad) return "text-rose-400";
    if (v >= warn) return "text-amber-400";
    return "text-emerald-400";
  }

  return (
    <Card data-testid="drift-tracker-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2"><Activity className="w-5 h-5 text-amber-400" />Drift Trackers — Calibration / Distribution / Features</span>
          <Button variant="ghost" size="sm" onClick={fetchData} disabled={loading} data-testid="drift-refresh">
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {error && <div className="text-rose-400 text-sm mb-2">Error: {error}</div>}
        {data && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
            <div className="p-3 rounded-lg border border-border/40" data-testid="drift-calibration">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-muted-foreground">Calibration (ECE)</span>
                {data.calibration.breached && <Badge variant="destructive">breach</Badge>}
              </div>
              <div className={`text-2xl font-mono ${tone(data.calibration.score, data.calibration.threshold * 0.5, data.calibration.threshold)}`}>{fmtNum(data.calibration.score, 4)}</div>
              <div className="text-xs text-muted-foreground mt-1">threshold {fmtNum(data.calibration.threshold, 2)} · n={data.calibration.nSamples}</div>
            </div>
            <div className="p-3 rounded-lg border border-border/40" data-testid="drift-distribution">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-muted-foreground">Prediction distribution KL</span>
                {data.distribution.breached && <Badge variant="destructive">breach</Badge>}
              </div>
              <div className={`text-2xl font-mono ${tone(data.distribution.score, data.distribution.threshold * 0.5, data.distribution.threshold)}`}>{fmtNum(data.distribution.score, 4)}</div>
              <div className="text-xs text-muted-foreground mt-1">
                threshold {fmtNum(data.distribution.threshold, 2)} · n={data.distribution.nSamples}
              </div>
              <div className="mt-1 text-[10px] text-muted-foreground space-y-0.5">
                {Object.keys(data.distribution.recent).map((k) => (
                  <div key={k} className="flex justify-between">
                    <span>{k}</span>
                    <span className="font-mono">{fmtNum(data.distribution.recent[k], 3)} vs {fmtNum(data.distribution.baseline[k], 3)}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="p-3 rounded-lg border border-border/40" data-testid="drift-feature">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-muted-foreground">Feature drift (max |z|)</span>
                {data.feature.breached && <Badge variant="destructive">breach</Badge>}
              </div>
              <div className={`text-2xl font-mono ${tone(data.feature.score, data.feature.threshold * 0.6, data.feature.threshold)}`}>{fmtNum(data.feature.score, 2)}</div>
              <div className="text-xs text-muted-foreground mt-1">threshold {fmtNum(data.feature.threshold, 1)}σ · n={data.feature.nSamples}</div>
              {data.feature.perFeature.length > 0 && (
                <div className="mt-2 space-y-0.5">
                  {[...data.feature.perFeature]
                    .sort((a, b) => Math.abs(b.zScore) - Math.abs(a.zScore))
                    .slice(0, 4)
                    .map((f, i) => (
                      <div key={i} className="flex justify-between text-xs">
                        <span className="font-mono truncate mr-2">{f.name}</span>
                        <span className={tone(Math.abs(f.zScore), data.feature.threshold * 0.6, data.feature.threshold)}>{f.zScore.toFixed(2)}σ</span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface QuarantineEvent {
  id: number;
  registryId: number;
  fromState: string;
  toState: string;
  reasonCode: string;
  triggeredBy: string;
  detail: Record<string, unknown> | null;
  createdAt: string;
}

export function QuarantineEventsCard() {
  const [events, setEvents] = useState<QuarantineEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<{ checked: number; quarantined: number } | null>(null);
  const admin = useAdminKey({});

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(apiUrl("/crypto/diagnostics/quarantine?limit=20"));
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setEvents(Array.isArray(j.events) ? j.events : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "load failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const runSweep = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await admin.adminFetch(apiUrl("/crypto/diagnostics/quarantine/run"), {
        method: "POST",
        action: "run quarantine sweep",
      });
      if (!r) { setError("Admin key required"); return; }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      const decisions = Array.isArray(j.decisions) ? j.decisions : [];
      const quarantined = decisions.filter((d: { decision: string }) => d.decision === "quarantined").length;
      setLastRun({ checked: decisions.length, quarantined });
      fetchData();
    } catch (e) {
      setError(e instanceof Error ? e.message : "sweep failed");
    } finally {
      setRunning(false);
    }
  }, [admin, fetchData]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 60_000);
    return () => clearInterval(id);
  }, [fetchData]);

  return (
    <Card data-testid="quarantine-events-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2"><ShieldAlert className="w-5 h-5 text-rose-400" />Model Quarantine Events</span>
          <span className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={runSweep} disabled={running} data-testid="quarantine-run-sweep">
              {running ? "Sweeping…" : "Run sweep"}
            </Button>
            <Button variant="ghost" size="sm" onClick={fetchData} disabled={loading} data-testid="quarantine-refresh">
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {error && <div className="text-rose-400 text-sm mb-2">Error: {error}</div>}
        {lastRun && (
          <div className="text-xs text-muted-foreground mb-2" data-testid="quarantine-last-run">
            Last sweep: checked {lastRun.checked}, quarantined {lastRun.quarantined}
          </div>
        )}
        {events.length === 0 ? (
          <div className="text-sm text-muted-foreground py-3 text-center">No quarantine events</div>
        ) : (
          <div className="space-y-1" data-testid="quarantine-events-list">
            {events.map((ev) => (
              <div key={ev.id} className="flex items-center justify-between text-xs p-2 rounded border border-border/30">
                <div className="flex flex-col">
                  <div className="flex items-center gap-2">
                    <Badge variant={ev.toState === "quarantined" ? "destructive" : "outline"}>{ev.fromState} → {ev.toState}</Badge>
                    <span className="font-mono">registry #{ev.registryId}</span>
                    <span className="text-amber-400">{ev.reasonCode}</span>
                    <Badge variant="outline" className="text-[10px]">{ev.triggeredBy}</Badge>
                  </div>
                  <span className="text-muted-foreground mt-0.5">{new Date(ev.createdAt).toLocaleString()}</span>
                </div>
                {ev.detail && (
                  <code className="text-[10px] text-muted-foreground max-w-md truncate">{JSON.stringify(ev.detail)}</code>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface FeatureCandidate {
  id: number;
  name: string;
  transformKind: string;
  sourceColumn: string | null;
  state: string;
  description: string | null;
  proposedBy: string | null;
  createdAt: string;
  approvedAt: string | null;
  approvedBy: string | null;
}

interface FeatureLabReport {
  id: number;
  candidateId: number;
  timeframe: string;
  coinId: string;
  nFolds: number;
  nSamples: number;
  baselineLogLoss: number | null;
  augmentedLogLoss: number | null;
  deltaLogLoss: number | null;
  baselineAccuracy: number | null;
  augmentedAccuracy: number | null;
  deltaAccuracy: number | null;
  runnerStatus?: string;
  createdAt: string;
}

interface UnquarantineEvent {
  id: number;
  createdAt: string;
  candidateId: number;
  candidateName: string;
  operator: string;
  note: string | null;
  priorReason: string | null;
  priorQuarantinedAt: string | null;
  priorReasonDetail?: Record<string, unknown> | null;
}

interface FeatureLabResponse {
  candidates: FeatureCandidate[];
  approvedFeatures: Array<{ name: string; transformKind: string; sourceColumn?: string | null }>;
  quarantinedFeatures?: QuarantinedFeature[];
  unquarantineEvents?: UnquarantineEvent[];
  transformKinds: string[];
}

// Task #256 — group-by roll-up of un-quarantine overrides.
interface UnquarantineOverrideOperatorEntry {
  operator: string;
  count: number;
  lastAt: string;
  candidates: string[];
}
interface UnquarantineOverrideCandidateEntry {
  candidateId: number;
  candidateName: string;
  count: number;
  lastAt: string;
  operators: string[];
}
interface UnquarantineOverrideSummary {
  windowDays: number;
  windowStart: string;
  totalEvents: number;
  byOperator: UnquarantineOverrideOperatorEntry[];
  byCandidate: UnquarantineOverrideCandidateEntry[];
}

// Task #235 — operator feedback loop: which model versions actually
// carry each approved feature, per timeframe.
interface AppliedModelEntry {
  registryId: number;
  modelVersion: string;
  coinId: string;
  timeframe: string;
  state: string;
  promotedAt: string | null;
  createdAt: string;
}
interface ApprovedFeatureApplied {
  name: string;
  transformKind: string;
  sourceColumn: string | null;
  appliedIn: AppliedModelEntry[];
}

// Task #242 — features auto-retired by the trainer after a validation
// regression. Mirrors `QuarantinedFeatureRecord` in api-server/lib.
interface QuarantinedTimeframeDetail {
  timeframe: string;
  current_log_loss: number | null;
  prior_log_loss: number | null;
  delta_log_loss: number | null;
}
interface QuarantinedFeature {
  name: string;
  transformKind: string | null;
  sourceColumn: string | null;
  quarantinedAt: string;
  reason: string;
  detail?: {
    trigger?: string;
    threshold?: number;
    timeframes?: QuarantinedTimeframeDetail[];
  } | null;
}

export function FeatureLabCard() {
  const [data, setData] = useState<FeatureLabResponse | null>(null);
  const [reports, setReports] = useState<Record<number, FeatureLabReport[]>>({});
  const [applied, setApplied] = useState<ApprovedFeatureApplied[]>([]);
  const [quarantined, setQuarantined] = useState<QuarantinedFeature[]>([]);
  // Tracks quarantined entries we've already surfaced as toasts so 60s
  // refreshes don't keep re-firing for the same auto-retire event.
  // Initialized lazily on the first poll so pre-existing entries don't
  // notify on every page load — only entries that appear *after* mount.
  const seenQuarantineKeysRef = useRef<Set<string> | null>(null);
  const [overrideSummary, setOverrideSummary] = useState<UnquarantineOverrideSummary | null>(null);
  const [overrideWindow, setOverrideWindow] = useState<number>(30);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [kind, setKind] = useState("rsi_squared");
  const [sourceColumn, setSourceColumn] = useState("");
  const admin = useAdminKey({});

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [r, rApplied, rQuar, rOverrides] = await Promise.all([
        fetch(apiUrl("/crypto/feature-lab/candidates")),
        fetch(apiUrl("/crypto/feature-lab/applied-models")),
        fetch(apiUrl("/crypto/feature-lab/quarantined")),
        fetch(apiUrl(`/crypto/feature-lab/unquarantine-summary?windowDays=${overrideWindow}`)),
      ]);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
      // Best-effort: applied-models is a UI nicety, not a hard
      // dependency of the candidate list, so swallow non-2xx.
      if (rApplied.ok) {
        const j = await rApplied.json();
        setApplied(Array.isArray(j.features) ? j.features : []);
      }
      if (rQuar.ok) {
        const j = await rQuar.json();
        const features: QuarantinedFeature[] = Array.isArray(j.features) ? j.features : [];
        setQuarantined(features);
        // Push-style notification: fire a toast the first time a new
        // entry appears in the quarantined list (Task #252).
        const keyOf = (q: QuarantinedFeature) => `${q.name}@${q.quarantinedAt}`;
        if (seenQuarantineKeysRef.current === null) {
          // First poll — seed the set so existing entries don't fire.
          seenQuarantineKeysRef.current = new Set(features.map(keyOf));
        } else {
          const seen = seenQuarantineKeysRef.current;
          for (const q of features) {
            const k = keyOf(q);
            if (seen.has(k)) continue;
            seen.add(k);
            const worst = pickWorstTimeframe(q.detail?.timeframes);
            const tfLabel = worst ? worst.timeframe : "unknown timeframe";
            const deltaLabel = worst && worst.delta_log_loss != null
              ? `Δlog_loss ${worst.delta_log_loss >= 0 ? "+" : ""}${worst.delta_log_loss.toFixed(4)}`
              : "Δlog_loss n/a";
            toast({
              variant: "destructive",
              title: `Feature auto-retired: ${q.name}`,
              description: `Worst timeframe ${tfLabel} · ${deltaLabel} · reason ${q.reason}`,
              duration: Infinity,
              action: (
                <ToastAction
                  altText="View in Feature Lab"
                  onClick={scrollToFeatureLab}
                  data-testid={`quarantine-toast-view-${q.name}`}
                >
                  View
                </ToastAction>
              ),
            });
          }
        }
      }
      if (rOverrides.ok) {
        const j = (await rOverrides.json()) as UnquarantineOverrideSummary;
        setOverrideSummary(j);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "load failed");
    } finally {
      setLoading(false);
    }
  }, [overrideWindow]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 60_000);
    return () => clearInterval(id);
  }, [fetchData]);

  const create = useCallback(async () => {
    if (!name.trim()) return;
    setBusy(-1);
    setError(null);
    try {
      const r = await admin.adminFetch(apiUrl("/crypto/feature-lab/candidates"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          transformKind: kind,
          sourceColumn: sourceColumn.trim() || null,
        }),
        action: "create feature candidate",
      });
      if (!r) { setError("Admin key required"); return; }
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      setName("");
      fetchData();
    } catch (e) {
      setError(e instanceof Error ? e.message : "create failed");
    } finally {
      setBusy(null);
    }
  }, [admin, name, kind, sourceColumn, fetchData]);

  const loadReports = useCallback(async (id: number) => {
    try {
      const r = await fetch(apiUrl(`/crypto/feature-lab/reports?candidateId=${id}`));
      if (!r.ok) return;
      const j = await r.json();
      setReports((prev) => ({ ...prev, [id]: Array.isArray(j.reports) ? j.reports : [] }));
    } catch {
      // ignore
    }
  }, []);

  const action = useCallback(async (id: number, op: "ablate" | "approve" | "reject" | "unquarantine") => {
    setBusy(id);
    setError(null);
    try {
      // Task #248 — un-quarantine has a pre-flight check: if the
      // validation regression that drove the auto-retire still looks
      // present in the latest training report, warn the operator and
      // require a typed reason before forcing it through.
      let body: Record<string, unknown> | undefined;
      if (op === "unquarantine") {
        const aRes = await fetch(
          apiUrl(`/crypto/feature-lab/candidates/${id}/unquarantine-assessment`),
        );
        const aJson = aRes.ok ? await aRes.json() : null;
        const a = (aJson?.assessment ?? null) as null | {
          status: string;
          threshold: number;
          worstOriginalDelta: number | null;
          worstLatestDelta: number | null;
          regressionStillPresent: boolean;
          trainingReportError?: string;
          perTimeframe: Array<{
            timeframe: string;
            originalDelta: number | null;
            latestDelta: number | null;
            recovered: boolean | null;
          }>;
        };
        const fmt = (n: number | null | undefined) =>
          n == null || !Number.isFinite(n) ? "n/a" : n.toFixed(4);
        if (a?.regressionStillPresent) {
          const lines: string[] = [
            "⚠️ Validation regression still looks real.",
            "",
            `Original Δlog-loss: ${fmt(a.worstOriginalDelta)}`,
            `Latest   Δlog-loss: ${fmt(a.worstLatestDelta)}`,
            `Threshold:          ${fmt(a.threshold)}`,
          ];
          if (a.perTimeframe.length > 0) {
            lines.push("", "Per timeframe (orig → latest):");
            for (const t of a.perTimeframe) {
              lines.push(
                `  ${t.timeframe}: ${fmt(t.originalDelta)} → ${fmt(t.latestDelta)}` +
                  (t.recovered === true ? " ✓ recovered" : t.recovered === false ? " ✗ still bad" : " (?)"),
              );
            }
          }
          if (a.trainingReportError) {
            lines.push("", `Note: latest training report could not be read (${a.trainingReportError}).`);
          }
          lines.push("", "Type a reason to un-quarantine anyway (or Cancel):");
          const reason = window.prompt(lines.join("\n"), "");
          if (reason == null) {
            setBusy(null);
            return;
          }
          const trimmed = reason.trim();
          if (!trimmed) {
            setError("Reason required to un-quarantine while the regression is still present.");
            setBusy(null);
            return;
          }
          body = { acknowledgement: trimmed };
        } else {
          const summary =
            a == null
              ? "Could not load regression assessment."
              : `Original Δlog-loss: ${fmt(a.worstOriginalDelta)}\nLatest   Δlog-loss: ${fmt(a.worstLatestDelta)}\nThreshold:          ${fmt(a.threshold)}\n\nThe regression appears to have recovered.`;
          if (!window.confirm(`Un-quarantine this feature?\n\n${summary}`)) {
            setBusy(null);
            return;
          }
        }
      }
      const postOnce = async (sendBody: Record<string, unknown> | undefined) =>
        admin.adminFetch(apiUrl(`/crypto/feature-lab/candidates/${id}/${op}`), {
          method: "POST",
          ...(sendBody
            ? {
                headers: { "content-type": "application/json" },
                body: JSON.stringify(sendBody),
              }
            : {}),
          action: `${op} feature candidate`,
        });
      let r = await postOnce(body);
      if (r && r.status === 409 && op === "unquarantine") {
        // Server says the regression is still present. This happens when
        // the preflight assessment fetch failed (or the operator skipped
        // it entirely) so we never collected a typed reason. Recover by
        // prompting now with the assessment the 409 carries.
        let payload: { assessment?: { worstOriginalDelta?: number | null; worstLatestDelta?: number | null; threshold?: number } } | null = null;
        try { payload = await r.json(); } catch { payload = null; }
        const a = payload?.assessment ?? null;
        const fmt = (n: number | null | undefined) =>
          n == null || !Number.isFinite(n) ? "n/a" : n.toFixed(4);
        const msg =
          "⚠️ Validation regression still looks real.\n\n" +
          `Original Δlog-loss: ${fmt(a?.worstOriginalDelta)}\n` +
          `Latest   Δlog-loss: ${fmt(a?.worstLatestDelta)}\n` +
          `Threshold:          ${fmt(a?.threshold)}\n\n` +
          "Type a reason to un-quarantine anyway (or Cancel):";
        const reason = window.prompt(msg, "");
        if (reason == null || !reason.trim()) {
          setError("Reason required to un-quarantine while the regression is still present.");
          setBusy(null);
          return;
        }
        r = await postOnce({ acknowledgement: reason.trim() });
      }
      if (!r) { setError("Admin key required"); return; }
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try {
          const j = await r.json();
          if (j?.code === "regression_still_present") {
            msg = "Regression still present — type a reason to override.";
          } else if (typeof j?.error === "string") {
            msg = j.error;
          }
        } catch {
          msg = `HTTP ${r.status}: ${await r.text().catch(() => "")}`;
        }
        throw new Error(msg);
      }
      fetchData();
      if (op === "ablate") await loadReports(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : `${op} failed`);
    } finally {
      setBusy(null);
    }
  }, [admin, fetchData, loadReports]);

  return (
    <Card id={FEATURE_LAB_CARD_ID} data-testid="feature-lab-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2"><Beaker className="w-5 h-5 text-emerald-400" />Feature Lab — Walk-Forward Ablation</span>
          <Button variant="ghost" size="sm" onClick={fetchData} disabled={loading} data-testid="feature-lab-refresh">
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {error && <div className="text-rose-400 text-sm mb-2">Error: {error}</div>}
        {data && (
          <>
            <div className="mb-3 space-y-2" data-testid="approved-features-applied">
              <div className="text-xs text-muted-foreground">
                Approved features — most recent model versions per timeframe that actually applied them:
              </div>
              {data.approvedFeatures.length === 0 ? (
                <div className="text-xs text-muted-foreground">none approved yet</div>
              ) : (
                <div className="space-y-1">
                  {data.approvedFeatures.map((f) => {
                    const summary = applied.find((a) => a.name === f.name);
                    const entries = summary?.appliedIn ?? [];
                    // Group by timeframe and pick the most recent (entries
                    // already arrive sorted desc by createdAt). Showing one
                    // per TF keeps the row compact while still confirming
                    // the approved feature reached the live schema.
                    const seenTf = new Set<string>();
                    const recentPerTf: AppliedModelEntry[] = [];
                    for (const e of entries) {
                      if (seenTf.has(e.timeframe)) continue;
                      seenTf.add(e.timeframe);
                      recentPerTf.push(e);
                    }
                    return (
                      <div key={f.name} className="flex flex-wrap items-center gap-2 text-xs" data-testid={`approved-feature-${f.name}`}>
                        <Badge variant="outline" className="font-mono">{f.name}</Badge>
                        <span className="text-muted-foreground">{f.transformKind}{f.sourceColumn ? `(${f.sourceColumn})` : ""}</span>
                        {recentPerTf.length === 0 ? (
                          <span className="text-amber-400" data-testid={`applied-empty-${f.name}`}>not yet in any trained model — waiting for next retrain</span>
                        ) : (
                          recentPerTf.map((e) => (
                            <Badge
                              key={`${e.timeframe}-${e.registryId}`}
                              variant="outline"
                              className="font-mono text-emerald-400 border-emerald-500/30"
                              title={`coin=${e.coinId} state=${e.state} created=${new Date(e.createdAt).toLocaleString()}`}
                              data-testid={`applied-in-${f.name}-${e.timeframe}`}
                            >
                              {e.timeframe}: {e.modelVersion}
                            </Badge>
                          ))
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            <div className="mb-3 space-y-2" data-testid="quarantined-features">
              <div className="text-xs text-muted-foreground">
                Auto-retired features — quarantined by the trainer after a validation regression:
              </div>
              {quarantined.length === 0 ? (
                <div className="text-xs text-muted-foreground" data-testid="quarantined-empty">none auto-retired</div>
              ) : (
                <div className="space-y-1">
                  {quarantined.map((q) => {
                    const tfs = q.detail?.timeframes ?? [];
                    return (
                      <div
                        key={`${q.name}-${q.quarantinedAt}`}
                        className="p-2 rounded border border-amber-500/30 bg-amber-500/5 text-xs"
                        data-testid={`quarantined-feature-${q.name}`}
                      >
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <Badge variant="outline" className="font-mono">{q.name}</Badge>
                          {q.transformKind && (
                            <span className="text-muted-foreground">
                              {q.transformKind}{q.sourceColumn ? `(${q.sourceColumn})` : ""}
                            </span>
                          )}
                          <Badge variant="destructive" data-testid={`quarantined-state-${q.name}`}>quarantined</Badge>
                          <Badge variant="outline" className="text-amber-400 border-amber-500/30">{q.reason}</Badge>
                          <span className="text-muted-foreground ml-auto">
                            {new Date(q.quarantinedAt).toLocaleString()}
                          </span>
                        </div>
                        {tfs.length > 0 ? (
                          <div className="flex flex-wrap gap-2 text-muted-foreground">
                            {tfs.map((t) => (
                              <span
                                key={t.timeframe}
                                className="font-mono"
                                data-testid={`quarantined-tf-${q.name}-${t.timeframe}`}
                              >
                                <span className="text-foreground">{t.timeframe}</span>:{" "}
                                {fmtNum(t.prior_log_loss, 4)} → <span className="text-rose-400">{fmtNum(t.current_log_loss, 4)}</span>
                                {" "}(Δ <span className="text-rose-400">{fmtNum(t.delta_log_loss, 4)}</span>)
                              </span>
                            ))}
                          </div>
                        ) : (
                          <div className="text-muted-foreground">no timeframe detail recorded</div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            <div className="mb-3 space-y-2" data-testid="unquarantine-override-summary">
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span>Un-quarantine overrides — roll-up over last</span>
                <select
                  value={overrideWindow}
                  onChange={(e) => setOverrideWindow(Number(e.target.value) || 30)}
                  className="bg-background border border-border/40 rounded px-2 text-xs"
                  data-testid="unquarantine-window-select"
                >
                  <option value={7}>7 days</option>
                  <option value={30}>30 days</option>
                  <option value={90}>90 days</option>
                </select>
                <span data-testid="unquarantine-window-total">
                  {overrideSummary ? `(${overrideSummary.totalEvents} total)` : ""}
                </span>
              </div>
              {overrideSummary && overrideSummary.totalEvents === 0 ? (
                <div className="text-xs text-muted-foreground" data-testid="unquarantine-summary-empty">
                  no overrides in this window
                </div>
              ) : overrideSummary ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  <div
                    className="p-2 rounded border border-border/30"
                    data-testid="unquarantine-summary-by-operator"
                  >
                    <div className="text-xs text-muted-foreground mb-1">By operator</div>
                    <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
                      {overrideSummary.byOperator.map((op) => (
                        <div
                          key={op.operator}
                          className="flex flex-wrap items-center gap-2 text-xs"
                          data-testid={`unquarantine-by-operator-${op.operator}`}
                        >
                          <Badge variant="outline" className="font-mono text-amber-400 border-amber-500/30">
                            {op.operator}
                          </Badge>
                          <span className="text-foreground">
                            {op.count} override{op.count === 1 ? "" : "s"}
                          </span>
                          <span className="text-muted-foreground">
                            across {op.candidates.length} feature{op.candidates.length === 1 ? "" : "s"}
                          </span>
                          <span
                            className="text-muted-foreground ml-auto font-mono"
                            title={op.candidates.join(", ")}
                          >
                            last {new Date(op.lastAt).toLocaleDateString()}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div
                    className="p-2 rounded border border-border/30"
                    data-testid="unquarantine-summary-by-candidate"
                  >
                    <div className="text-xs text-muted-foreground mb-1">By feature</div>
                    <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
                      {overrideSummary.byCandidate.map((c) => (
                        <div
                          key={c.candidateId}
                          className="flex flex-wrap items-center gap-2 text-xs"
                          data-testid={`unquarantine-by-candidate-${c.candidateId}`}
                        >
                          <Badge variant="outline" className="font-mono">{c.candidateName}</Badge>
                          <span className="text-foreground">
                            {c.count} override{c.count === 1 ? "" : "s"}
                          </span>
                          <span className="text-muted-foreground">
                            by {c.operators.length} operator{c.operators.length === 1 ? "" : "s"}
                          </span>
                          <span
                            className="text-muted-foreground ml-auto font-mono"
                            title={c.operators.join(", ")}
                          >
                            last {new Date(c.lastAt).toLocaleDateString()}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
            {data.unquarantineEvents && data.unquarantineEvents.length > 0 && (
              <div className="mb-3 space-y-1" data-testid="unquarantine-audit-log">
                <div className="text-xs text-muted-foreground">
                  Recent un-quarantine overrides (audit log) — last {data.unquarantineEvents.length}:
                </div>
                <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
                  {data.unquarantineEvents.slice(0, 10).map((ev) => (
                    <div
                      key={ev.id}
                      className="flex flex-wrap items-center gap-2 text-xs p-2 rounded border border-amber-500/20 bg-amber-500/5"
                      data-testid={`unquarantine-event-${ev.id}`}
                    >
                      <span className="text-muted-foreground font-mono">
                        {new Date(ev.createdAt).toLocaleString()}
                      </span>
                      <Badge variant="outline" className="font-mono">{ev.candidateName}</Badge>
                      <span className="text-amber-400">by {ev.operator}</span>
                      {ev.priorReason && (
                        <span className="text-muted-foreground">
                          prior reason: <span className="text-rose-400">{ev.priorReason}</span>
                        </span>
                      )}
                      {ev.note && (
                        <span className="text-muted-foreground italic">"{ev.note}"</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-2 mb-3 p-3 rounded border border-border/30">
              <Input placeholder="candidate name" value={name} onChange={(e) => setName(e.target.value)} data-testid="feature-lab-name-input" />
              <select value={kind} onChange={(e) => setKind(e.target.value)} className="bg-background border border-border/40 rounded px-2 text-sm" data-testid="feature-lab-kind-select">
                {data.transformKinds.map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
              <Input placeholder="source column (passthrough only)" value={sourceColumn} onChange={(e) => setSourceColumn(e.target.value)} data-testid="feature-lab-source-input" />
              <Button onClick={create} disabled={busy !== null || !name.trim()} data-testid="feature-lab-create">Create candidate</Button>
            </div>
            <div className="space-y-2" data-testid="feature-lab-candidates-list">
              {data.candidates.length === 0 ? (
                <div className="text-sm text-muted-foreground text-center py-3">No candidates yet — create one above</div>
              ) : (
                data.candidates.map((c) => {
                  const candidateReports = reports[c.id] ?? [];
                  const latest = candidateReports[0] ?? null;
                  const eligible = latest != null && (latest.deltaLogLoss ?? 0) > 0 && latest.nSamples >= 200;
                  return (
                    <div key={c.id} className="p-2 rounded border border-border/30 text-xs">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge variant="outline" className="font-mono">{c.name}</Badge>
                          <span className="text-muted-foreground">{c.transformKind}{c.sourceColumn ? `(${c.sourceColumn})` : ""}</span>
                          <Badge variant={c.state === "approved" ? "default" : c.state === "rejected" || c.state === "quarantined" ? "destructive" : "outline"}>{c.state}</Badge>
                        </div>
                        <div className="flex gap-1">
                          <Button size="sm" variant="outline" onClick={() => loadReports(c.id)} data-testid={`feature-lab-reports-${c.id}`}>Reports</Button>
                          <Button size="sm" variant="outline" onClick={() => action(c.id, "ablate")} disabled={busy === c.id} data-testid={`feature-lab-ablate-${c.id}`}>Ablate</Button>
                          <Button size="sm" variant="default" onClick={() => action(c.id, "approve")} disabled={busy === c.id || !eligible} data-testid={`feature-lab-approve-${c.id}`}>Approve</Button>
                          {c.state === "quarantined" && (
                            <Button
                              size="sm"
                              variant="default"
                              className="bg-amber-600 hover:bg-amber-700"
                              onClick={() => action(c.id, "unquarantine")}
                              disabled={busy === c.id}
                              title="Disagree with auto-retire — restore this feature to the approved bucket"
                              data-testid={`feature-lab-unquarantine-${c.id}`}
                            >
                              Un-quarantine
                            </Button>
                          )}
                          <Button size="sm" variant="destructive" onClick={() => action(c.id, "reject")} disabled={busy === c.id} data-testid={`feature-lab-reject-${c.id}`}>Reject</Button>
                        </div>
                      </div>
                      {c.state === "quarantined" && (() => {
                        const q = data.quarantinedFeatures?.find((x) => x.name === c.name);
                        if (!q) return null;
                        return (
                          <div className="text-xs text-amber-400 mb-1" data-testid={`feature-lab-quarantine-reason-${c.id}`}>
                            auto-retired ({q.reason}) at {new Date(q.quarantinedAt).toLocaleString()}
                          </div>
                        );
                      })()}
                      {latest && (
                        <div className="flex flex-wrap gap-3 text-muted-foreground">
                          <span>Δlog-loss: <span className={(latest.deltaLogLoss ?? 0) > 0 ? "text-emerald-400" : "text-rose-400"}>{fmtNum(latest.deltaLogLoss, 4)}</span></span>
                          <span>Δaccuracy: <span className={(latest.deltaAccuracy ?? 0) > 0 ? "text-emerald-400" : "text-rose-400"}>{fmtNum(latest.deltaAccuracy, 4)}</span></span>
                          <span>folds: {latest.nFolds}</span>
                          <span>n: {latest.nSamples}</span>
                          {latest.timeframe && <span>tf: {latest.timeframe}</span>}
                          {eligible && <Badge variant="default" className="bg-emerald-600">eligible</Badge>}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
