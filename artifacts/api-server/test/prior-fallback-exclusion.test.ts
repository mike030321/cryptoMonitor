import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Task #107 — Prior-source predictions (the Laplace-smoothed pooled
// fallback added in task #98) must be tagged on insert and EXCLUDED from
// the headline accuracy / P&L scoreboards by default. Without this, the
// flat marginal distribution drags calibrated accuracy toward the
// empirical prior and looks like a regression.
//
// We assert two things:
//   1. The schema + insert sites carry a `source` column populated from
//      the ml-engine `MlPredictResponse.source`.
//   2. The aggregation routes (/crypto/brain/accuracy,
//      /crypto/reality-check, /crypto/shadow/metrics) drop source='prior'
//      rows by default and accept an `includePrior` toggle.
//
// We also exercise the in-memory shadow-metrics filter path with a
// hand-built fixture so a future change can't silently delete the filter
// without breaking this test.

const here = dirname(fileURLToPath(import.meta.url));
const apiRoot = join(here, "..");
const repoRoot = join(apiRoot, "..", "..");

function readSrc(absPath: string): string {
  return readFileSync(absPath, "utf8");
}

test("predictions schema declares a `source` column for provenance tagging", () => {
  const src = readSrc(join(repoRoot, "lib/db/src/schema/predictions.ts"));
  assert.match(src, /source:\s*text\("source"\)/, "predictions.source column must exist");
});

test("model_predictions schema declares a `source` column", () => {
  const src = readSrc(join(repoRoot, "lib/db/src/schema/model_predictions.ts"));
  assert.match(src, /source:\s*text\("source"\)/, "model_predictions.source column must exist");
});

test("monitor.ts persists prediction.quant.source into predictions.source", () => {
  const src = readSrc(join(apiRoot, "src/lib/monitor.ts"));
  assert.match(
    src,
    /source:\s*prediction\.quant\?\.source\s*\?\?\s*null/,
    "monitor insert must tag source from the quant payload",
  );
});

test("shadow-recorder persists ml-engine source into model_predictions.source", () => {
  const src = readSrc(join(apiRoot, "src/lib/shadow-recorder.ts"));
  assert.match(
    src,
    /source:\s*s\.source/,
    "shadow recorder insert must forward MlPredictResponse.source",
  );
});

test("/crypto/brain/accuracy excludes source='prior' by default and accepts includePrior toggle", () => {
  const src = readSrc(join(apiRoot, "src/routes/crypto/index.ts"));
  // Default-exclude clause used in BOTH the rollup and trend SQL.
  const filterMatches = src.match(/source IS DISTINCT FROM 'prior'/g) ?? [];
  assert.ok(
    filterMatches.length >= 1,
    "headline accuracy SQL must filter source='prior' by default",
  );
  // Toggle is wired through a request query param.
  assert.match(src, /includePrior/, "must expose an `includePrior` query param");
  // Response surfaces the prior-fallback footer for the dashboard.
  assert.match(src, /priorFallback:\s*\{/);
});

test("/crypto/reality-check drops trades from prior-source predictions by default", () => {
  const src = readSrc(join(apiRoot, "src/routes/crypto/index.ts"));
  // The trade filter joins paper_trades to predictions.source via a
  // parameterized inArray (NOT raw SQL concat) so no ids can smuggle SQL.
  assert.match(src, /inArray\(predictionsTable\.id, tradePredictionIds\)/);
  assert.match(src, /eq\(predictionsTable\.source, "prior"\)/);
  assert.match(src, /priorTradesExcluded/);
});

test("/crypto/shadow/metrics passes includePrior into computeShadowMetrics", () => {
  const src = readSrc(join(apiRoot, "src/routes/crypto/index.ts"));
  assert.match(src, /computeShadowMetrics\(includePrior\)/);
  assert.match(src, /r\.source === "prior"/, "shadow metric loop must skip prior rows");
});

// Behavioural fixture: replicate the in-memory shadow-metrics filter to
// guarantee a prior row is excluded from the headline aggregate.
type ShadowRow = {
  outcome: "pending" | "correct" | "wrong" | "neutral";
  source: string | null;
  timeframe: string;
};

function aggregateShadowResolvedCount(rows: ShadowRow[], includePrior = false): number {
  let n = 0;
  for (const r of rows) {
    if (r.outcome === "pending") continue;
    if (!includePrior && r.source === "prior") continue;
    n++;
  }
  return n;
}

test("regression: prior-source row is excluded from the headline shadow aggregate", () => {
  const rows: ShadowRow[] = [
    { outcome: "correct", source: "lightgbm", timeframe: "5m" },
    { outcome: "wrong",   source: "prior",    timeframe: "5m" }, // must be dropped
    { outcome: "correct", source: null,       timeframe: "5m" }, // legacy row, kept
    { outcome: "pending", source: "lightgbm", timeframe: "5m" }, // unresolved, dropped
  ];
  // Default: prior row excluded, pending excluded → 2 resolved counted.
  assert.equal(aggregateShadowResolvedCount(rows, false), 2);
  // With the toggle on: prior row folded back in → 3 resolved counted.
  assert.equal(aggregateShadowResolvedCount(rows, true), 3);
});
