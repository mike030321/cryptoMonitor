/**
 * Task #599 — 1h/2h calibration ON/OFF verdict surface for the
 * dashboard.
 *
 * The dataset-refresher (Task #540) auto-runs
 * `scripts/task592_parallel_stage2.py` after every successful 1h or
 * 2h snapshot and writes the result into
 * `models/datasets/_freshness_status.json` under
 * `calibration_verdict.short_tf`. Until this card existed, an
 * operator had to SSH-tail the latest
 * `*-task592-1h2h-stage2-verdict.md` to find out whether the
 * pre-trade ON-vs-OFF gate was holding up.
 *
 * This card surfaces:
 *   * a colored health pill (ok / error / timeout / unknown) driven
 *     by the loop's `last_status`,
 *   * the per-stage aggregates (in-band slice count, mean trade
 *     share, mean DA-lift, summed PnL%) for OFF vs ON side-by-side,
 *   * an expandable section with the tail of the markdown report
 *     and the file paths the loop wrote it to.
 */

import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Clock,
  FileText,
  Gauge,
  Hourglass,
} from "lucide-react";
import {
  useShortTfCalibrationVerdict,
  type ShortTfCalibrationStageSummary,
  type ShortTfCalibrationVerdictResponse,
  type ShortTfCalibrationVerdictState,
} from "@/hooks/use-news";
import { formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

function stateStyles(state: ShortTfCalibrationVerdictState): {
  ring: string;
  bg: string;
  fg: string;
  label: string;
  Icon: typeof CheckCircle2;
} {
  switch (state) {
    case "ok":
      return {
        ring: "ring-emerald-500/30",
        bg: "bg-emerald-500/10",
        fg: "text-emerald-300",
        label: "OK",
        Icon: CheckCircle2,
      };
    case "error":
      return {
        ring: "ring-red-500/40",
        bg: "bg-red-500/10",
        fg: "text-red-300",
        label: "ERROR",
        Icon: AlertTriangle,
      };
    case "timeout":
      return {
        ring: "ring-amber-500/40",
        bg: "bg-amber-500/10",
        fg: "text-amber-300",
        label: "TIMEOUT",
        Icon: Hourglass,
      };
    case "unknown":
    default:
      return {
        ring: "ring-muted-foreground/20",
        bg: "bg-muted/20",
        fg: "text-muted-foreground",
        label: "N/A",
        Icon: Clock,
      };
  }
}

function fmtNum(
  v: number | null | undefined,
  digits: number = 3,
): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function fmtCount(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return String(Math.round(v));
}

function StageColumn({
  label,
  stage,
}: {
  label: "OFF" | "ON";
  stage: ShortTfCalibrationStageSummary | null;
}) {
  const inBand = stage?.nInTradeShareBand ?? null;
  const total = stage?.nSlices ?? null;
  // The verdict's primary signal is "how many evaluated slices fell
  // inside the [0.40, 0.85] trade-share band". A high in-band count
  // with positive PnL is the goal; ON should beat OFF here.
  const inBandPct =
    inBand != null && total != null && total > 0
      ? Math.round((inBand / total) * 100)
      : null;
  return (
    <div
      className="rounded-lg border border-border/40 bg-card/40 p-3 space-y-1.5 text-xs font-mono"
      data-testid={`short-tf-calibration-verdict-stage-${label.toLowerCase()}`}
    >
      <div className="flex items-center justify-between">
        <span className="uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        <span className="text-foreground/80">
          {fmtCount(stage?.nSlices)} slices
        </span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground">in-band</span>
        <span>
          {fmtCount(inBand)}
          {inBandPct != null && (
            <span className="text-muted-foreground"> ({inBandPct}%)</span>
          )}
        </span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground">gate-pass</span>
        <span>{fmtCount(stage?.nPassingGate)}</span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground">mean tradeshare</span>
        <span>{fmtNum(stage?.meanTradeShare, 3)}</span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground">mean DA lift</span>
        <span>{fmtNum(stage?.meanDaLift, 4)}</span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground">Σ PnL %</span>
        <span
          className={cn(
            stage?.sumPnlPctTotalAug != null && stage.sumPnlPctTotalAug > 0
              ? "text-emerald-300"
              : stage?.sumPnlPctTotalAug != null && stage.sumPnlPctTotalAug < 0
                ? "text-red-300"
                : undefined,
          )}
        >
          {fmtNum(stage?.sumPnlPctTotalAug, 1)}
        </span>
      </div>
    </div>
  );
}

function VerdictHeadline({
  data,
}: {
  data: ShortTfCalibrationVerdictResponse;
}) {
  const block = data.shortTf;
  if (!block) {
    return (
      <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
        <Clock className="w-3.5 h-3.5" />
        <span>
          no calibration verdict on disk yet — waiting for the next 1h/2h
          dataset refresh to trigger one
        </span>
      </div>
    );
  }
  if (data.state === "ok") {
    const summary = block.summary;
    const onIn = summary?.on?.nInTradeShareBand ?? null;
    const offIn = summary?.off?.nInTradeShareBand ?? null;
    const wall = block.lastElapsedSeconds;
    return (
      <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-emerald-300">
        <CheckCircle2 className="w-3.5 h-3.5" />
        <span>
          last verdict {block.lastSuccessAt
            ? formatTimeAgo(block.lastSuccessAt)
            : "—"}
          {onIn != null && offIn != null && (
            <>
              {" · "}
              <span className="text-foreground/80">
                in-band ON {onIn} vs OFF {offIn}
              </span>
            </>
          )}
          {wall != null && (
            <>
              {" · "}
              <span className="text-muted-foreground">
                ran in {Math.round(wall)}s
              </span>
            </>
          )}
        </span>
      </div>
    );
  }
  return (
    <div className="flex items-start gap-2 text-xs font-mono text-foreground/80">
      <AlertTriangle className="w-3.5 h-3.5 mt-0.5 text-red-300" />
      <div className="space-y-0.5">
        <div>
          <span className="text-red-300 uppercase">{data.state}</span>
          {block.lastAttemptAt && (
            <>
              {" · attempted "}
              {formatTimeAgo(block.lastAttemptAt)}
            </>
          )}
          {block.lastSuccessAt && (
            <span className="text-muted-foreground">
              {" · last good "}
              {formatTimeAgo(block.lastSuccessAt)}
            </span>
          )}
        </div>
        {block.lastError && (
          <div
            className="text-red-300/90 break-words"
            data-testid="short-tf-calibration-verdict-error"
          >
            {block.lastError}
          </div>
        )}
      </div>
    </div>
  );
}

export function ShortTfCalibrationVerdictCard() {
  const { data, isLoading, isError, error } = useShortTfCalibrationVerdict();
  const [open, setOpen] = useState(false);

  if (isLoading) {
    return (
      <div
        className="rounded-xl border border-border/40 bg-card/40 p-4 text-xs font-mono text-muted-foreground"
        data-testid="short-tf-calibration-verdict-card-loading"
      >
        loading 1h/2h calibration verdict…
      </div>
    );
  }
  if (isError || !data) {
    return (
      <div
        className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-xs font-mono text-red-300"
        data-testid="short-tf-calibration-verdict-card-error"
      >
        failed to load 1h/2h calibration verdict
        {error instanceof Error ? `: ${error.message}` : ""}
      </div>
    );
  }

  const styles = stateStyles(data.state);
  const StateIcon = styles.Icon;
  const summary = data.shortTf?.summary ?? null;
  const tfSubset =
    summary?.timeframesSubset ?? data.shortTf?.triggerTimeframes ?? null;

  return (
    <div
      className="rounded-xl border border-border/40 bg-card/40 p-4 space-y-3"
      data-testid="short-tf-calibration-verdict-card"
      data-verdict-state={data.state}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-mono uppercase tracking-wider">
            1h/2h Calibration Verdict
          </span>
          {tfSubset && tfSubset.length > 0 && (
            <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
              [{tfSubset.join(", ")}]
            </span>
          )}
        </div>
        <span
          className={cn(
            "text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full ring-1 inline-flex items-center gap-1",
            styles.ring,
            styles.bg,
            styles.fg,
          )}
          data-testid="short-tf-calibration-verdict-state"
        >
          <StateIcon className="w-3 h-3" />
          {styles.label}
        </span>
      </div>

      <VerdictHeadline data={data} />

      {summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <StageColumn label="OFF" stage={summary.off} />
          <StageColumn label="ON" stage={summary.on} />
        </div>
      )}

      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger
          className="w-full group"
          data-testid="short-tf-calibration-verdict-details-toggle"
        >
          <div className="flex items-center justify-between p-2 rounded-md bg-card/40 border border-border/40 hover:bg-card/60 transition-colors">
            <span className="text-xs font-mono uppercase tracking-wider flex items-center gap-2">
              <FileText className="w-3.5 h-3.5" />
              Verdict report
              {data.markdownPath && (
                <span className="text-muted-foreground normal-case tracking-normal">
                  · {data.markdownPath}
                </span>
              )}
            </span>
            <ChevronDown className="w-4 h-4 text-muted-foreground transition-transform group-data-[state=closed]:-rotate-90" />
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-2">
          {data.markdownReadError && (
            <div className="text-xs font-mono text-red-300 px-1 py-2">
              failed to read verdict markdown: {data.markdownReadError}
            </div>
          )}
          {data.markdownTail ? (
            <pre
              className="text-[11px] font-mono leading-relaxed text-foreground/80 bg-card/40 border border-border/40 rounded-md p-3 max-h-72 overflow-auto whitespace-pre-wrap"
              data-testid="short-tf-calibration-verdict-markdown"
            >
              {data.markdownTail}
            </pre>
          ) : (
            !data.markdownReadError && (
              <div className="text-xs font-mono text-muted-foreground px-1 py-2">
                no verdict report on disk yet
              </div>
            )
          )}
          {(data.shortTf?.command || data.shortTf?.timeoutSeconds) && (
            <div className="mt-2 text-[10px] font-mono text-muted-foreground space-y-0.5">
              {data.shortTf?.command && (
                <div>
                  <span className="uppercase tracking-wider">cmd: </span>
                  <span className="break-all">{data.shortTf.command}</span>
                </div>
              )}
              {data.shortTf?.timeoutSeconds != null && (
                <div>
                  <span className="uppercase tracking-wider">timeout: </span>
                  {data.shortTf.timeoutSeconds}s
                </div>
              )}
              {data.jsonPath && (
                <div>
                  <span className="uppercase tracking-wider">json: </span>
                  <span className="break-all">{data.jsonPath}</span>
                </div>
              )}
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
