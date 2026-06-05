import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, CheckCircle2, RotateCcw, ShieldAlert, FlaskConical, LogOut } from "lucide-react";
// Task #444 — `useEvolutionStatus`, the trigger-evolve action, and the
// recent-cycles list were all backed by `agent-evolution` (LLM-driven
// personality mutation) which has been removed. The evolution card is
// rendered as a "removed" notice below.
import { useQueryClient } from "@tanstack/react-query";
import { useAdminKey } from "@/hooks/use-admin-key";
import { AdminKeyField } from "@/components/admin-key-field";

interface AblationConfig {
  dualConsensus: boolean;
  fingerprintMatching: boolean;
  contagionDetection: boolean;
  confidenceCalibration: boolean;
  regimeDetection: boolean;
  agentSpecialization: boolean;
}

const TOGGLE_LABELS: Record<keyof AblationConfig, { label: string; description: string }> = {
  dualConsensus: {
    label: "Dual Consensus",
    description: "Require agreement between primary and secondary agents",
  },
  fingerprintMatching: {
    label: "Fingerprint Matching",
    description: "Match coin patterns against historical fingerprints",
  },
  contagionDetection: {
    label: "Contagion Detection",
    description: "Detect cross-coin contagion / correlated risk events",
  },
  confidenceCalibration: {
    label: "Confidence Calibration",
    description: "Calibrate raw model confidences against historical accuracy",
  },
  regimeDetection: {
    label: "Regime Detection",
    description: "Identify the current market regime to gate strategies",
  },
  agentSpecialization: {
    label: "Agent Specialization",
    description: "Route coins to specialized agents instead of broad ones",
  },
};

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

type ToastTone = "success" | "error";

export default function AdminPanel() {
  const [config, setConfig] = useState<AblationConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pendingKey, setPendingKey] = useState<keyof AblationConfig | null>(null);
  const [resetting, setResetting] = useState(false);
  const [toast, setToast] = useState<{ tone: ToastTone; message: string } | null>(null);
  const queryClient = useQueryClient();

  const showToast = (tone: ToastTone, message: string) => {
    setToast({ tone, message });
    window.setTimeout(() => setToast(null), 6000);
  };

  const admin = useAdminKey({
    onRejected: () => {
      showToast(
        "error",
        "Admin key was rejected. We kept your last attempt in the field below — fix the typo and re-save.",
      );
    },
  });
  const { hasKey, clearKey, adminFetch } = admin;

  const loadConfig = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await adminFetch(`${apiBase}/crypto/ablation-config`, {
        action: "view the experiment toggles",
      });
      if (!res) return;
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error((data as { error?: string }).error || `Failed to load config (HTTP ${res.status})`);
      }
      const data = (await res.json()) as AblationConfig;
      setConfig(data);
    } catch (err) {
      setLoadError(String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Intentionally do NOT auto-load. The operator must paste the admin key
    // first via the "Load config" button so that even viewing the toggles
    // requires the ADMIN_API_KEY.
  }, []);

  const updateToggle = async (key: keyof AblationConfig, nextValue: boolean) => {
    if (!config) return;
    setPendingKey(key);
    try {
      const res = await adminFetch(`${apiBase}/crypto/ablation-config`, {
        action: `${nextValue ? "enable" : "disable"} "${TOGGLE_LABELS[key].label}"`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [key]: nextValue }),
      });
      if (!res) return;
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error((data as { error?: string }).error || `HTTP ${res.status}`);
      setConfig(data as AblationConfig);
      showToast("success", `${TOGGLE_LABELS[key].label} ${nextValue ? "enabled" : "disabled"}.`);
    } catch (err) {
      showToast("error", `Update failed: ${String(err)}`);
    } finally {
      setPendingKey(null);
    }
  };

  const resetConfig = async () => {
    const confirmed = window.confirm("Reset all experiment toggles back to their defaults (everything ON)?");
    if (!confirmed) return;
    setResetting(true);
    try {
      const res = await adminFetch(`${apiBase}/crypto/ablation-reset`, {
        action: "reset experiment toggles",
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res) return;
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error((data as { error?: string }).error || `HTTP ${res.status}`);
      setConfig(data as AblationConfig);
      showToast("success", "Experiment toggles reset to defaults.");
    } catch (err) {
      showToast("error", `Reset failed: ${String(err)}`);
    } finally {
      setResetting(false);
    }
  };

  // Task #444 — `triggerEvolve` removed alongside the rest of the
  // agent-evolution plane. The server endpoint `/crypto/admin/evolve`
  // is still mounted but now returns 410 Gone.

  return (
    <div className="space-y-6 max-w-4xl" data-testid="admin-panel">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-amber-500/10 ring-1 ring-amber-500/30 flex items-center justify-center">
          <ShieldAlert className="w-5 h-5 text-amber-400" />
        </div>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-display font-bold tracking-tight">Admin Panel</h1>
          <p className="text-sm text-muted-foreground">
            Operator controls for experiments and the agent evolution loop. Each action requires the ADMIN_API_KEY.
          </p>
        </div>
        <div className="shrink-0">
          {hasKey && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                clearKey();
                showToast("success", "Admin key forgotten. Paste it again below to run admin actions.");
              }}
              className="border-amber-500/40 text-amber-300 hover:bg-amber-500/10 hover:text-amber-200"
              data-testid="button-forget-admin-key"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Forget admin key
            </Button>
          )}
        </div>
      </div>

      {!hasKey && (
        <AdminKeyField
          admin={admin}
          autoFocus
          helpText="Paste your ADMIN_API_KEY here to load the experiment toggles and trigger evolutions. The key stays in this tab only."
          testIdPrefix="admin-key-panel"
        />
      )}

      {toast && (
        <Alert
          variant={toast.tone === "error" ? "destructive" : "default"}
          data-testid={`admin-toast-${toast.tone}`}
        >
          {toast.tone === "error" ? (
            <AlertCircle className="h-4 w-4" />
          ) : (
            <CheckCircle2 className="h-4 w-4" />
          )}
          <AlertTitle>{toast.tone === "error" ? "Failed" : "Success"}</AlertTitle>
          <AlertDescription>{toast.message}</AlertDescription>
        </Alert>
      )}

      <Card className="bg-card/50 border-border/40">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-cyan-400" />
              Experiment Toggles (Ablation Config)
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={loadConfig}
                disabled={loading}
                data-testid="button-reload-ablation"
              >
                {loading ? "Loading…" : config ? "Reload" : "Load config"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={resetConfig}
                disabled={resetting || loading}
                className="border-amber-500/40 text-amber-300 hover:bg-amber-500/10 hover:text-amber-200"
                data-testid="button-reset-ablation"
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                {resetting ? "Resetting…" : "Reset to defaults"}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {loadError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Failed to load config</AlertTitle>
              <AlertDescription>{loadError}</AlertDescription>
            </Alert>
          )}

          {loading && !config && (
            <div className="space-y-2">
              {[0, 1, 2, 3, 4, 5].map((i) => (
                <Skeleton key={i} className="h-14 w-full" />
              ))}
            </div>
          )}

          {!loading && !config && !loadError && (
            <div
              className="text-sm text-muted-foreground p-4 rounded-md border border-dashed border-border/40"
              data-testid="admin-empty-state"
            >
              {hasKey ? (
                <>Click <span className="font-medium text-foreground">Load config</span> to view and edit the experiment toggles.</>
              ) : (
                <>Paste your ADMIN_API_KEY in the field above, then click <span className="font-medium text-foreground">Load config</span> to view the experiment toggles.</>
              )}
            </div>
          )}

          {config && (
            <ul className="space-y-2">
              {(Object.keys(TOGGLE_LABELS) as Array<keyof AblationConfig>).map((key) => {
                const meta = TOGGLE_LABELS[key];
                const value = config[key];
                const isPending = pendingKey === key;
                return (
                  <li
                    key={key}
                    className="flex items-start justify-between gap-4 p-3 rounded-lg border border-border/30 bg-white/[0.02]"
                    data-testid={`row-toggle-${key}`}
                  >
                    <div className="min-w-0">
                      <Label htmlFor={`toggle-${key}`} className="text-sm font-medium">
                        {meta.label}
                      </Label>
                      <p className="text-xs text-muted-foreground mt-0.5">{meta.description}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0 pt-0.5">
                      <span
                        className={`text-[10px] font-mono uppercase tracking-wider ${
                          value ? "text-emerald-400" : "text-muted-foreground"
                        }`}
                      >
                        {isPending ? "Saving…" : value ? "On" : "Off"}
                      </span>
                      <Switch
                        id={`toggle-${key}`}
                        checked={value}
                        disabled={isPending}
                        onCheckedChange={(next) => updateToggle(key, next)}
                        data-testid={`switch-${key}`}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card className="bg-card/50 border-border/40" data-testid="card-agent-evolution-removed">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-mono uppercase tracking-wider">
            Agent Evolution
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Removed in Task #444</AlertTitle>
            <AlertDescription>
              The LLM-driven evolution loop (personality mutation, prompt
              evolution, hybrid offspring) has been removed. The deterministic
              quant brain is now the sole authority. A deterministic
              5-agent strategy registry lands in Task #468.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    </div>
  );
}
