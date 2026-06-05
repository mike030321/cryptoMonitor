import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, RefreshCw, AlertTriangle } from "lucide-react";

interface FailureAnalysisLatest {
  generated_at: string | null;
  bucket_counts: Record<string, number>;
  summary_md: string;
  json_path?: string | null;
  md_path?: string | null;
  error?: string;
}

interface FailureAnalysisHistoryRow {
  generated_at: string | null;
  bucket_counts: Record<string, number>;
  json_path?: string | null;
}

interface FailureAnalysisHistory {
  rows: FailureAnalysisHistoryRow[];
  count: number;
  error?: string;
}

const BUCKET_LABELS: Record<string, string> = {
  promoted: "Promoted",
  salvageable_with_schema_fix: "Schema fix",
  insufficient_sample: "Insufficient sample",
  structurally_noisy_retire: "Structurally noisy (retire)",
  salvageable_with_better_features_or_labels: "Better features / labels",
  unknown: "Unknown",
};

const BUCKET_TONE: Record<string, string> = {
  promoted: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  salvageable_with_schema_fix: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  insufficient_sample: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  structurally_noisy_retire: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  salvageable_with_better_features_or_labels:
    "bg-sky-500/15 text-sky-300 border-sky-500/30",
  unknown: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

// Solid fills for the stacked-bar history. Kept in sync with BUCKET_TONE
// so the legend chips match the bar colors.
const BUCKET_FILL: Record<string, string> = {
  promoted: "bg-emerald-500",
  salvageable_with_schema_fix: "bg-amber-500",
  insufficient_sample: "bg-slate-500",
  structurally_noisy_retire: "bg-rose-500",
  salvageable_with_better_features_or_labels: "bg-sky-500",
  unknown: "bg-slate-600",
};

// Stable bucket order so colors stay consistent across runs even when
// the per-row dict iteration order changes.
const BUCKET_ORDER = [
  "promoted",
  "salvageable_with_schema_fix",
  "salvageable_with_better_features_or_labels",
  "insufficient_sample",
  "structurally_noisy_retire",
  "unknown",
];

function formatTs(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function formatTsShort(iso: string | null): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return `${d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    })} ${d.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
    })}`;
  } catch {
    return iso;
  }
}

export function FailureAnalysisCard() {
  const [data, setData] = useState<FailureAnalysisLatest | null>(null);
  const [history, setHistory] = useState<FailureAnalysisHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    setHistoryError(null);
    // Run both fetches independently so a transient history outage
    // doesn't suppress the latest-snapshot card.
    const [latestResult, historyResult] = await Promise.allSettled([
      fetch(`${import.meta.env.BASE_URL}api/crypto/failure-analysis-latest`),
      fetch(
        `${import.meta.env.BASE_URL}api/crypto/failure-analysis-history?limit=30`,
      ),
    ]);

    if (latestResult.status === "fulfilled") {
      try {
        const r = latestResult.value;
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = (await r.json()) as FailureAnalysisLatest;
        setData(j);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load");
      }
    } else {
      setError(
        latestResult.reason instanceof Error
          ? latestResult.reason.message
          : "Failed to load",
      );
    }

    if (historyResult.status === "fulfilled") {
      try {
        const r = historyResult.value;
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = (await r.json()) as FailureAnalysisHistory;
        setHistory(j);
      } catch (err) {
        setHistory({ rows: [], count: 0 });
        setHistoryError(err instanceof Error ? err.message : "Failed to load");
      }
    } else {
      setHistory({ rows: [], count: 0 });
      setHistoryError(
        historyResult.reason instanceof Error
          ? historyResult.reason.message
          : "Failed to load",
      );
    }

    setLoading(false);
  };

  useEffect(() => {
    void load();
  }, []);

  const buckets = Object.entries(data?.bucket_counts ?? {})
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1]);
  const total = buckets.reduce((acc, [, n]) => acc + n, 0);
  const hasReport = !!data?.generated_at;

  // History is returned newest-first; reverse for left-to-right time axis.
  const historyRows = (history?.rows ?? []).slice().reverse();
  const hasHistory = historyRows.length >= 2;
  const historyMaxTotal = historyRows.reduce((m, row) => {
    const t = Object.values(row.bucket_counts).reduce(
      (a, n) => a + (Number.isFinite(n) ? n : 0),
      0,
    );
    return t > m ? t : m;
  }, 0);
  // Bucket keys present anywhere in the history, kept in BUCKET_ORDER
  // first then any unknowns appended.
  const historyBucketKeys = (() => {
    const seen = new Set<string>();
    historyRows.forEach((r) =>
      Object.keys(r.bucket_counts).forEach((k) => seen.add(k)),
    );
    const ordered = BUCKET_ORDER.filter((k) => seen.has(k));
    const extras = Array.from(seen).filter((k) => !BUCKET_ORDER.includes(k));
    return [...ordered, ...extras];
  })();

  return (
    <Card data-testid="failure-analysis-card">
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div>
          <CardTitle className="flex items-center gap-2 text-lg">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            Failure analysis (auto)
          </CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Generated automatically after each retrain from{" "}
            <code>models/report.json</code>. Buckets per the same rules as
            the hand-run failure-analysis script.
          </p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => void load()}
          disabled={loading}
          data-testid="failure-analysis-refresh"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-xs text-muted-foreground">
          <span className="font-medium text-foreground">Generated:</span>{" "}
          {formatTs(data?.generated_at ?? null)}
          {hasReport && total > 0 && (
            <>
              {" · "}
              <span className="font-medium text-foreground">{total}</span>{" "}
              slices classified
            </>
          )}
        </div>

        {error && (
          <div className="text-xs text-rose-400" data-testid="failure-analysis-error">
            Failed to load: {error}
          </div>
        )}
        {!error && data?.error && (
          <div className="text-xs text-rose-400">
            Engine error: {data.error}
          </div>
        )}
        {!loading && !error && !hasReport && (
          <div className="text-xs text-muted-foreground">
            No auto-generated failure-analysis report yet — one will be
            written after the next retrain.
          </div>
        )}

        {hasReport && buckets.length > 0 && (
          <div className="flex flex-wrap gap-2" data-testid="failure-analysis-buckets">
            {buckets.map(([bucket, n]) => (
              <Badge
                key={bucket}
                variant="outline"
                className={BUCKET_TONE[bucket] ?? BUCKET_TONE.unknown}
              >
                {BUCKET_LABELS[bucket] ?? bucket}: {n}
              </Badge>
            ))}
          </div>
        )}

        {hasReport && historyError && (
          <div
            className="text-xs text-amber-400"
            data-testid="failure-analysis-history-error"
          >
            History unavailable: {historyError}
          </div>
        )}

        {hasReport && hasHistory && historyMaxTotal > 0 && (
          <div data-testid="failure-analysis-history">
            <div className="text-xs text-muted-foreground mb-2">
              <span className="font-medium text-foreground">
                Bucket counts over the last {historyRows.length} retrains
              </span>{" "}
              (oldest → newest)
            </div>
            <div className="flex items-end gap-1 h-24">
              {historyRows.map((row, idx) => {
                const rowTotal = historyBucketKeys.reduce(
                  (a, k) => a + (Number(row.bucket_counts[k]) || 0),
                  0,
                );
                const heightPct =
                  historyMaxTotal > 0
                    ? (rowTotal / historyMaxTotal) * 100
                    : 0;
                const tooltip =
                  `${formatTs(row.generated_at)} — ${rowTotal} slices\n` +
                  historyBucketKeys
                    .map((k) => `${BUCKET_LABELS[k] ?? k}: ${row.bucket_counts[k] ?? 0}`)
                    .join("\n");
                return (
                  <div
                    key={`${row.generated_at ?? "row"}-${idx}`}
                    className="flex-1 min-w-[6px] flex flex-col justify-end h-full"
                    title={tooltip}
                    data-testid="failure-analysis-history-bar"
                  >
                    <div
                      className="w-full flex flex-col-reverse rounded-sm overflow-hidden"
                      style={{ height: `${heightPct}%` }}
                    >
                      {historyBucketKeys.map((k) => {
                        const n = Number(row.bucket_counts[k]) || 0;
                        if (n <= 0 || rowTotal <= 0) return null;
                        const segPct = (n / rowTotal) * 100;
                        return (
                          <div
                            key={k}
                            className={BUCKET_FILL[k] ?? BUCKET_FILL.unknown}
                            style={{ height: `${segPct}%` }}
                          />
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
              <span>{formatTsShort(historyRows[0]?.generated_at ?? null)}</span>
              <span>
                {formatTsShort(
                  historyRows[historyRows.length - 1]?.generated_at ?? null,
                )}
              </span>
            </div>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
              {historyBucketKeys.map((k) => (
                <span key={k} className="inline-flex items-center gap-1">
                  <span
                    className={`inline-block w-2 h-2 rounded-sm ${
                      BUCKET_FILL[k] ?? BUCKET_FILL.unknown
                    }`}
                  />
                  {BUCKET_LABELS[k] ?? k}
                </span>
              ))}
            </div>
          </div>
        )}

        {hasReport && data?.summary_md && (
          <div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded((v) => !v)}
              data-testid="failure-analysis-toggle"
              className="-ml-2 text-xs"
            >
              {expanded ? (
                <ChevronUp className="w-4 h-4 mr-1" />
              ) : (
                <ChevronDown className="w-4 h-4 mr-1" />
              )}
              {expanded ? "Hide markdown summary" : "Show markdown summary"}
            </Button>
            {expanded && (
              <pre
                className="mt-2 max-h-[600px] overflow-auto rounded-md bg-muted/40 p-3 text-[11px] leading-relaxed whitespace-pre-wrap font-mono"
                data-testid="failure-analysis-md"
              >
                {data.summary_md}
              </pre>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
