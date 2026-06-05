import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AdminKeyField } from "@/components/admin-key-field";
import { useAdminKey } from "@/hooks/use-admin-key";
import { Zap, AlertCircle, CheckCircle2, Clock, Loader2 } from "lucide-react";

const apiUrl = (p: string) => `${import.meta.env.BASE_URL}api${p}`;

const TIMEFRAME_OPTIONS = ["1m", "5m", "1h", "2h", "6h", "1d"] as const;

interface MonitoredCoin {
  id: string;
  name: string;
  symbol: string;
}

interface MlEngineStatus {
  running?: boolean;
  last_started_at?: number | null;
  last_finished_at?: number | null;
  last_status?: string | null;
  last_report?: Record<string, unknown> | null;
  // Task #457 — auto-prune janitor (added in #451) writes these onto
  // `_retrain_state` after every training cycle. Surface them as a
  // small chip so operators can see the cleanup is healthy without
  // tailing server logs.
  last_pruned_count?: number | null;
  last_pruned_bytes?: number | null;
  last_pruned_error?: string | null;
}

// Format a byte count for the cleanup chip. Uses base-10 (KB/MB/GB) so
// the number matches `du -h --si` and operator intuition; the chip is
// for at-a-glance health, not byte-exact accounting.
function fmtBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i += 1;
  }
  // 1 decimal for KB+; bytes stay whole.
  const rounded = i === 0 ? Math.round(v) : Math.round(v * 10) / 10;
  return `${rounded} ${units[i]}`;
}

// Body shape mirrors /ml/admin/retrain exactly (per task #324 contract):
// 202 → { status: "accepted", started_at }
// 409 → { detail: "retrain already in progress" }
// 400 → { error, validCoinIds? } (api-server validation layer)
interface RetrainResponse {
  status?: string;
  started_at?: number;
  detail?: string;
  error?: string;
  validCoinIds?: string[];
}

interface ResultBanner {
  kind: "ok" | "busy" | "error";
  message: string;
  detail?: string;
}

function fmtAgo(ts: number | null | undefined): string {
  if (!ts) return "never";
  const sec = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

export function RetrainCoinCard() {
  const admin = useAdminKey({});
  const [coins, setCoins] = useState<MonitoredCoin[] | null>(null);
  const [coinId, setCoinId] = useState<string>("");
  const [selectedTfs, setSelectedTfs] = useState<Set<string>>(
    () => new Set(["5m"]),
  );
  const [submitting, setSubmitting] = useState(false);
  const [banner, setBanner] = useState<ResultBanner | null>(null);
  const [status, setStatus] = useState<MlEngineStatus | null>(null);

  // Load monitored coin list once on mount. We intentionally do not
  // depend on `coinId` here — selecting a different coin should not
  // re-fetch the list. The default-selection branch reads the latest
  // `coinId` via the functional setter to avoid a stale-closure trap.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(apiUrl("/crypto/coins"), { cache: "no-store" });
        if (!r.ok) return;
        const j = (await r.json()) as Array<{ id: string; name: string; symbol: string }>;
        if (cancelled) return;
        const list = j.map((c) => ({ id: c.id, name: c.name, symbol: c.symbol }));
        setCoins(list);
        if (list.length > 0) {
          setCoinId((current) => current || list[0].id);
        }
      } catch {
        // non-fatal — leave the picker empty; status polling and the
        // operator's next interaction will surface any persistent
        // network problem via the normal banner channel.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Poll the ml-engine retrain status so the card can show whether a
  // run is currently in progress (and disable the button accordingly).
  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const r = await fetch(apiUrl("/crypto/brain/retrain"), { cache: "no-store" });
        if (!r.ok) return;
        const j = await r.json();
        if (cancelled) return;
        setStatus((j?.mlEngine as MlEngineStatus | null) ?? null);
      } catch {
        // ignore — status polling is best-effort
      }
    }
    tick();
    const id = setInterval(tick, 8_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const toggleTf = (tf: string) => {
    setSelectedTfs((prev) => {
      const next = new Set(prev);
      if (next.has(tf)) next.delete(tf);
      else next.add(tf);
      return next;
    });
  };

  async function submit() {
    if (!coinId) {
      setBanner({ kind: "error", message: "Pick a coin first." });
      return;
    }
    setSubmitting(true);
    setBanner(null);
    try {
      const tfs = Array.from(selectedTfs);
      const r = await admin.adminFetch(apiUrl("/crypto/brain/retrain/coin"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          coinId,
          ...(tfs.length > 0 ? { timeframes: tfs } : {}),
        }),
        action: "trigger per-coin retrain",
      });
      if (!r) {
        setBanner({ kind: "error", message: "Admin key required — paste it above." });
        return;
      }
      const j = (await r.json().catch(() => ({}))) as RetrainResponse;
      if (r.status === 202) {
        const coinName = coins?.find((c) => c.id === coinId)?.name ?? coinId;
        const tfLabel = tfs.length > 0 ? tfs.join(", ") : "all timeframes";
        setBanner({
          kind: "ok",
          message: `Retrain accepted for ${coinName} (${tfLabel}).`,
          detail: "Track progress under the Brain Switch card or via the watchdog log.",
        });
      } else if (r.status === 409) {
        setBanner({
          kind: "busy",
          message: "ML engine is already running a retrain.",
          detail: j.detail ?? "Wait for it to finish, then try again.",
        });
      } else {
        setBanner({
          kind: "error",
          message: j.error ?? j.detail ?? `Retrain rejected (HTTP ${r.status}).`,
        });
      }
    } catch (e) {
      setBanner({
        kind: "error",
        message: e instanceof Error ? e.message : "Retrain request failed.",
      });
    } finally {
      setSubmitting(false);
    }
  }

  const running = status?.running === true;
  const disabled = submitting || running || !coinId;

  return (
    <Card data-testid="retrain-coin-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Zap className="h-4 w-4" />
          Retrain one coin
          <Badge variant="outline" className="ml-2 text-xs">Operator</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          A full retrain walks ~60 slices and takes ~4–5 hours. When iterating
          on a single coin (label thresholds, feature changes, manifest fixes)
          use this to retrain just that coin's slices in minutes.
        </p>

        <AdminKeyField
          admin={admin}
          label="Admin key (ADMIN_API_KEY)"
          helpText="Required to trigger a retrain. Same key as /admin/reload."
          testIdPrefix="retrain-coin-admin-key"
        />

        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="retrain-coin-picker" className="text-xs">Coin</Label>
            {coins == null ? (
              <Input
                id="retrain-coin-picker"
                value="Loading…"
                disabled
                className="h-9"
                data-testid="retrain-coin-picker-loading"
              />
            ) : (
              <Select value={coinId} onValueChange={setCoinId}>
                <SelectTrigger
                  id="retrain-coin-picker"
                  className="h-9"
                  data-testid="retrain-coin-picker"
                >
                  <SelectValue placeholder="Pick a coin" />
                </SelectTrigger>
                <SelectContent>
                  {coins.map((c) => (
                    <SelectItem
                      key={c.id}
                      value={c.id}
                      data-testid={`retrain-coin-option-${c.id}`}
                    >
                      {c.name} ({c.symbol})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Timeframes</Label>
            <div
              className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-md border border-border/50 bg-muted/20 px-3 py-2"
              data-testid="retrain-coin-tf-picker"
            >
              {TIMEFRAME_OPTIONS.map((tf) => {
                const checked = selectedTfs.has(tf);
                return (
                  <label
                    key={tf}
                    className="flex items-center gap-1.5 text-xs font-mono cursor-pointer select-none"
                    data-testid={`retrain-coin-tf-${tf}`}
                  >
                    <Checkbox
                      checked={checked}
                      onCheckedChange={() => toggleTf(tf)}
                      data-testid={`retrain-coin-tf-${tf}-checkbox`}
                    />
                    {tf}
                  </label>
                );
              })}
              <span className="text-[11px] text-muted-foreground ml-auto">
                {selectedTfs.size === 0 ? "all timeframes" : `${selectedTfs.size} selected`}
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
            <div className="text-[11px] font-mono text-muted-foreground flex items-center gap-2">
              <Clock className="h-3 w-3" />
              ML engine:{" "}
              {running ? (
                <span className="text-amber-300">running (started {fmtAgo(status?.last_started_at)})</span>
              ) : status?.last_finished_at ? (
                <span>idle — last finished {fmtAgo(status.last_finished_at)}{" "}
                  {status.last_status ? <Badge variant="outline" className="ml-1 text-[10px]">{status.last_status}</Badge> : null}
                </span>
              ) : (
                <span>idle</span>
              )}
            </div>
            {/* Task #457 — auto-prune chip. Renders as soon as the
                ml-engine has reported a janitor outcome (count is set,
                even if 0). On error, swap the chip for a red error chip
                so a regression is loud instead of silent. */}
            {status?.last_pruned_error ? (
              <Badge
                variant="outline"
                className="text-[10px] border-rose-500/40 bg-rose-500/[0.06] text-rose-200"
                data-testid="retrain-coin-cleanup-chip-error"
                title={status.last_pruned_error}
              >
                Last cleanup: failed
              </Badge>
            ) : status?.last_pruned_count != null ? (
              <Badge
                variant="outline"
                className="text-[10px]"
                data-testid="retrain-coin-cleanup-chip"
              >
                Last cleanup: {status.last_pruned_count} version
                {status.last_pruned_count === 1 ? "" : "s"}, {fmtBytes(status.last_pruned_bytes)} freed
              </Badge>
            ) : null}
          </div>
          <Button
            onClick={submit}
            disabled={disabled}
            size="sm"
            data-testid="retrain-coin-submit"
          >
            {submitting ? (
              <><Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />Submitting…</>
            ) : running ? (
              "Retrain in progress"
            ) : (
              "Retrain this coin"
            )}
          </Button>
        </div>

        {banner && (
          <div
            className={[
              "rounded-md border px-3 py-2 text-xs flex items-start gap-2",
              banner.kind === "ok"
                ? "border-emerald-500/40 bg-emerald-500/[0.06] text-emerald-200"
                : banner.kind === "busy"
                ? "border-amber-500/40 bg-amber-500/[0.06] text-amber-200"
                : "border-rose-500/40 bg-rose-500/[0.06] text-rose-200",
            ].join(" ")}
            data-testid={`retrain-coin-banner-${banner.kind}`}
          >
            {banner.kind === "ok" ? (
              <CheckCircle2 className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            ) : (
              <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            )}
            <div className="space-y-0.5">
              <div className="font-medium">{banner.message}</div>
              {banner.detail && <div className="text-[11px] opacity-80">{banner.detail}</div>}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
