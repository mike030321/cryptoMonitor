import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useAdminKey } from "@/hooks/use-admin-key";
import { Boxes, CheckCircle2, XCircle, Crown, RotateCcw, FlaskConical, AlertTriangle, Radio } from "lucide-react";

interface RegistryRow {
  id: number;
  modelId: string;
  modelVersion: string;
  coinId: string;
  timeframe: string;
  state: "shadow" | "challenger" | "champion" | "quarantined" | "retired";
  promotedAt: string | null;
  demotedAt: string | null;
  previousChampionId: number | null;
  note: string | null;
  metricsSnapshot: Record<string, unknown> | null;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

interface EffectiveServingSlot {
  coinId: string;
  timeframe: string;
  latestVersion: string | null;
  servedCoinId: string | null;
  servedVersion: string | null;
  fallback: boolean;
  fallbackReason:
    | "quarantined-skip"
    | "pooled-fallback"
    | "quarantined-skip+pooled-fallback"
    | null;
  quarantinedVersions: string[];
}

interface PromotionVerdict {
  eligible: boolean;
  samplesOk: boolean;
  edgeLiftOk: boolean;
  drawdownOk: boolean;
  regimeRobustnessOk: boolean;
  reasons: string[];
  thresholds: Record<string, number>;
  metricsSummary: Record<string, unknown>;
}

const apiUrl = (p: string) => `${import.meta.env.BASE_URL}api${p}`;

const STATE_COLORS: Record<RegistryRow["state"], string> = {
  champion: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  challenger: "bg-sky-500/15 text-sky-400 border-sky-500/30",
  shadow: "bg-slate-500/15 text-slate-400 border-slate-500/30",
  quarantined: "bg-red-500/15 text-red-400 border-red-500/30",
  retired: "bg-zinc-500/15 text-zinc-400 border-zinc-500/30",
};

function GateRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {ok ? (
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
      ) : (
        <XCircle className="h-3.5 w-3.5 text-red-400" />
      )}
      <span className={ok ? "text-emerald-300" : "text-red-300"}>{label}</span>
    </div>
  );
}

export function ModelRegistryCard() {
  const admin = useAdminKey({});
  const [rows, setRows] = useState<RegistryRow[] | null>(null);
  const [serving, setServing] = useState<EffectiveServingSlot[]>([]);
  const [previews, setPreviews] = useState<Record<number, PromotionVerdict | undefined>>({});
  const [busyId, setBusyId] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function refresh() {
    try {
      const r = await fetch(apiUrl("/crypto/model-registry"), { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setRows(j.rows ?? []);
      setServing(j.effectiveServing ?? []);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load model registry");
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, []);

  async function loadPreview(id: number) {
    try {
      const r = await fetch(apiUrl(`/crypto/model-registry/${id}/promotion-preview`));
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setPreviews((p) => ({ ...p, [id]: j.verdict }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Preview failed");
    }
  }

  async function promote(id: number, force = false) {
    if (!window.confirm(force ? "Force promote this model to champion (gates failed)?" : "Promote this model to champion?")) return;
    setBusyId(id); setErr(null);
    try {
      const r = await admin.adminFetch(apiUrl(`/crypto/model-registry/${id}/promote`), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ force }),
        action: "promote model to champion",
      });
      if (!r) { setErr("Admin key required"); return; }
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error ?? `HTTP ${r.status}`);
      }
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Promote failed");
    } finally {
      setBusyId(null);
    }
  }

  async function rollback(row: RegistryRow) {
    if (!window.confirm(`Roll back champion ${row.modelId}/${row.coinId}/${row.timeframe} to previous version?`)) return;
    setBusyId(row.id); setErr(null);
    try {
      const r = await admin.adminFetch(apiUrl(`/crypto/model-registry/rollback`), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          modelId: row.modelId, coinId: row.coinId, timeframe: row.timeframe,
        }),
        action: "roll back champion",
      });
      if (!r) { setErr("Admin key required"); return; }
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error ?? `HTTP ${r.status}`);
      }
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Rollback failed");
    } finally {
      setBusyId(null);
    }
  }

  const grouped = useMemo(() => {
    const m = new Map<string, RegistryRow[]>();
    for (const r of rows ?? []) {
      const k = `${r.modelId}|${r.coinId}|${r.timeframe}`;
      if (!m.has(k)) m.set(k, []);
      m.get(k)!.push(r);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [rows]);

  const servingByCoinTf = useMemo(() => {
    const m = new Map<string, EffectiveServingSlot>();
    for (const s of serving) m.set(`${s.coinId}|${s.timeframe}`, s);
    return m;
  }, [serving]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Boxes className="h-4 w-4" />
          Model Registry
          <Badge variant="outline" className="ml-2 text-xs">Phase 5</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {err && (
          <div className="mb-3 rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
            {err}
          </div>
        )}
        {!rows && <div className="text-xs text-muted-foreground">Loading…</div>}
        {rows && rows.length === 0 && (
          <div className="text-xs text-muted-foreground">
            No models registered yet. Models enter the registry in <code>shadow</code>{" "}
            state when promoted from training and graduate to <code>challenger</code>{" "}
            once they have sample volume. The promotion gate (samples + net edge lift +
            drawdown + regime robustness) decides champion.
          </div>
        )}
        <div className="space-y-3">
          {grouped.map(([slot, slotRows]) => {
            const sample = slotRows[0];
            const eff = servingByCoinTf.get(`${sample.coinId}|${sample.timeframe}`);
            return (
            <div key={slot} className="rounded border border-border/40 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                <span className="font-mono text-muted-foreground">{slot}</span>
                {eff && (
                  <span className="flex items-center gap-1 text-[11px]">
                    <Radio className="h-3 w-3 text-emerald-400" />
                    <span className="text-muted-foreground">serving:</span>
                    <span className="font-mono text-foreground">
                      {eff.servedVersion ?? "—"}
                    </span>
                    {eff.servedCoinId && eff.servedCoinId !== sample.coinId && (
                      <span className="text-muted-foreground">
                        (from {eff.servedCoinId})
                      </span>
                    )}
                    {eff.fallback && (
                      <Badge
                        variant="outline"
                        className="border-amber-500/40 bg-amber-500/10 text-[10px] text-amber-300"
                        title={
                          eff.quarantinedVersions.length
                            ? `Skipped quarantined: ${eff.quarantinedVersions.join(", ")}`
                            : "Live trader is serving an older version than the latest pointer"
                        }
                      >
                        <AlertTriangle className="mr-1 h-3 w-3" />
                        {eff.fallbackReason === "pooled-fallback"
                          ? "pooled fallback"
                          : eff.fallbackReason ===
                            "quarantined-skip+pooled-fallback"
                          ? "quarantined → pooled"
                          : eff.fallbackReason === "quarantined-skip"
                          ? "quarantined fallback"
                          : "fallback"}
                      </Badge>
                    )}
                  </span>
                )}
              </div>
              <div className="space-y-2">
                {slotRows.map((row) => {
                  const verdict = previews[row.id];
                  const isQuarantinedSkip =
                    !!eff &&
                    row.state === "champion" &&
                    eff.fallback &&
                    eff.servedVersion !== row.modelVersion;
                  const isServing =
                    !!eff &&
                    eff.servedVersion === row.modelVersion &&
                    (eff.servedCoinId ?? row.coinId) === row.coinId;
                  return (
                    <div key={row.id} className="rounded border border-border/30 p-2">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge className={`border ${STATE_COLORS[row.state]}`}>
                            {row.state === "champion" && <Crown className="mr-1 h-3 w-3" />}
                            {row.state}
                          </Badge>
                          <span className="font-mono text-xs">{row.modelVersion}</span>
                          {isServing && (
                            <Badge
                              variant="outline"
                              className="border-emerald-500/40 bg-emerald-500/10 text-[10px] text-emerald-300"
                              title="Live trader is currently routing predictions to this version"
                            >
                              <Radio className="mr-1 h-3 w-3" />
                              live serving
                            </Badge>
                          )}
                          {isQuarantinedSkip && (
                            <Badge
                              variant="outline"
                              className="border-amber-500/40 bg-amber-500/10 text-[10px] text-amber-300"
                              title={`Champion is quarantined; live trader is serving ${eff?.servedVersion ?? "—"} instead`}
                            >
                              <AlertTriangle className="mr-1 h-3 w-3" />
                              not serving — fallback to {eff?.servedVersion ?? "—"}
                            </Badge>
                          )}
                          {row.previousChampionId && (
                            <span className="text-[10px] text-muted-foreground">
                              ← prev #{row.previousChampionId}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1">
                          {row.state === "challenger" && (
                            <>
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-7 text-xs"
                                onClick={() => loadPreview(row.id)}
                              >
                                <FlaskConical className="mr-1 h-3 w-3" />
                                Check gates
                              </Button>
                              <Button
                                size="sm"
                                className="h-7 text-xs"
                                disabled={busyId === row.id}
                                onClick={() => promote(row.id, false)}
                              >
                                Promote
                              </Button>
                            </>
                          )}
                          {row.state === "champion" && row.previousChampionId && (
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 text-xs"
                              disabled={busyId === row.id}
                              onClick={() => rollback(row)}
                            >
                              <RotateCcw className="mr-1 h-3 w-3" />
                              Rollback
                            </Button>
                          )}
                        </div>
                      </div>
                      {verdict && (
                        <div className="mt-2 grid grid-cols-2 gap-1 rounded bg-muted/30 p-2">
                          <GateRow ok={verdict.samplesOk} label="Samples" />
                          <GateRow ok={verdict.edgeLiftOk} label="Net edge lift" />
                          <GateRow ok={verdict.drawdownOk} label="Drawdown" />
                          <GateRow ok={verdict.regimeRobustnessOk} label="Regime robustness" />
                          {!verdict.eligible && (
                            <>
                              <div className="col-span-2 mt-1 text-[10px] text-red-300">
                                {verdict.reasons.join(" · ")}
                              </div>
                              <div className="col-span-2">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  className="h-6 text-[10px] text-amber-400"
                                  onClick={() => promote(row.id, true)}
                                >
                                  Force-promote anyway
                                </Button>
                              </div>
                            </>
                          )}
                        </div>
                      )}
                      {(() => {
                        // Task #235 — surface the approved feature-lab
                        // features that actually baked into THIS model
                        // version. Empty list = base FEATURE_COLUMNS only;
                        // missing key = pre-Task-#235 row.
                        const snap = row.metricsSnapshot ?? {};
                        const raw = (snap as { approved_features_applied?: unknown }).approved_features_applied;
                        if (!Array.isArray(raw)) return null;
                        const names = (raw as unknown[]).filter((x): x is string => typeof x === "string");
                        return (
                          <div className="mt-2 flex flex-wrap items-center gap-1 text-[10px]" data-testid={`registry-applied-features-${row.id}`}>
                            <span className="text-muted-foreground">approved features:</span>
                            {names.length === 0 ? (
                              <span className="text-muted-foreground italic">none</span>
                            ) : (
                              names.map((n) => (
                                <Badge
                                  key={n}
                                  variant="outline"
                                  className="font-mono text-emerald-400 border-emerald-500/30"
                                >
                                  {n}
                                </Badge>
                              ))
                            )}
                          </div>
                        );
                      })()}
                      {row.note && (
                        <div className="mt-1 text-[10px] text-muted-foreground italic">
                          {row.note}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
