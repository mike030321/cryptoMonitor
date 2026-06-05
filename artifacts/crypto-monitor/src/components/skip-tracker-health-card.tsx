import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Database, AlertTriangle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "@/hooks/use-toast";

const STABLE_POLLS_BEFORE_CLEAR = 2;

type ToastHandle = ReturnType<typeof toast>;

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

interface SkipTrackerHealth {
  failures: number;
  successes: number;
  lastError: string | null;
  lastErrorAt: string | null;
}

export function SkipTrackerHealthCard() {
  const { data, isLoading, isError } = useQuery<SkipTrackerHealth>({
    queryKey: ["skip-tracker-health"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/skip-tracker-health`);
      if (!res.ok) throw new Error(`skip-tracker-health ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
    staleTime: 15_000,
  });

  const failures = data?.failures ?? 0;
  const hasFailures = failures > 0;

  const incidentToastRef = useRef<ToastHandle | null>(null);
  const lastFailuresRef = useRef<number | null>(null);
  const stablePollsRef = useRef(0);
  const incidentStartFailuresRef = useRef(0);

  useEffect(() => {
    if (!data) return;
    const prev = lastFailuresRef.current;
    const incidentActive = incidentToastRef.current !== null;

    const buildDescription = (delta: number, scope: "boot" | "incident") =>
      `${delta} failed write${delta === 1 ? "" : "s"} ${
        scope === "boot" ? "since boot" : `this incident (total ${failures} since boot)`
      }.${data.lastError ? ` Last error: ${data.lastError}` : ""}`;

    if (prev === null) {
      if (failures > 0) {
        incidentStartFailuresRef.current = 0;
        incidentToastRef.current = toast({
          variant: "destructive",
          title: "Skip-event DB writes failing",
          description: buildDescription(failures, "boot"),
          duration: Infinity,
        });
        stablePollsRef.current = 0;
      }
      lastFailuresRef.current = failures;
      return;
    }

    if (failures > prev) {
      stablePollsRef.current = 0;
      if (!incidentActive) {
        incidentStartFailuresRef.current = prev;
        incidentToastRef.current = toast({
          variant: "destructive",
          title: "Skip-event DB writes failing",
          description: buildDescription(failures - prev, "incident"),
          duration: Infinity,
        });
      } else if (incidentToastRef.current) {
        const delta = failures - incidentStartFailuresRef.current;
        incidentToastRef.current.update({
          id: incidentToastRef.current.id,
          variant: "destructive",
          title: "Skip-event DB writes failing",
          description: buildDescription(delta, "incident"),
          open: true,
          duration: Infinity,
        });
      }
    } else if (incidentActive) {
      stablePollsRef.current += 1;
      if (stablePollsRef.current >= STABLE_POLLS_BEFORE_CLEAR) {
        incidentToastRef.current?.dismiss();
        incidentToastRef.current = null;
        stablePollsRef.current = 0;
      }
    }

    lastFailuresRef.current = failures;
  }, [data, failures]);

  useEffect(() => {
    return () => {
      incidentToastRef.current?.dismiss();
      incidentToastRef.current = null;
    };
  }, []);

  return (
    <Card
      className={cn(
        "border-border/40",
        hasFailures
          ? "bg-rose-500/10 ring-1 ring-rose-500/30"
          : "bg-card/30 opacity-80",
      )}
      data-testid="skip-tracker-health-card"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
          <Database className="w-4 h-4" />
          Skip-event database health
          <span className="text-[10px] font-normal text-muted-foreground/70 normal-case">
            tracks silent DB drops when persisting trade-skip events
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-16 w-full" />}
        {isError && (
          <div
            className="text-sm text-rose-300 font-mono"
            data-testid="skip-tracker-health-error"
          >
            Couldn't load skip-event database health.
          </div>
        )}
        {data && (
          <div className="space-y-2">
            <div className="flex items-end justify-between gap-4 flex-wrap">
              <div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground">
                  Failed writes
                </div>
                <div
                  className={cn(
                    "text-3xl font-display font-bold mt-1 flex items-center gap-2",
                    hasFailures ? "text-rose-300" : "text-muted-foreground",
                  )}
                  data-testid="skip-tracker-health-failures"
                >
                  {hasFailures ? (
                    <AlertTriangle className="w-6 h-6" />
                  ) : (
                    <CheckCircle2 className="w-6 h-6 text-emerald-400/70" />
                  )}
                  {failures}
                </div>
                <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
                  {hasFailures
                    ? `${data.successes} successes since boot`
                    : `${data.successes} successes since boot — all writes landing`}
                </div>
              </div>
            </div>

            {hasFailures && data.lastError && (
              <div
                className="p-2 rounded-md bg-rose-500/10 ring-1 ring-rose-500/20 text-xs"
                data-testid="skip-tracker-health-last-error"
              >
                <div className="text-[10px] uppercase font-mono text-rose-200/80 mb-1">
                  Last error
                  {data.lastErrorAt && (
                    <span className="ml-1 normal-case text-muted-foreground">
                      · {new Date(data.lastErrorAt).toLocaleString()}
                    </span>
                  )}
                </div>
                <div
                  className="font-mono text-rose-200 break-words"
                  data-testid="skip-tracker-health-last-error-message"
                >
                  {data.lastError}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
