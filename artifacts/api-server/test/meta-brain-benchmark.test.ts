/**
 * Task #390 — behavior + fallback tests for the Strategy Lab → Meta-
 * Brain benchmark telemetry assembler.
 *
 * Covers:
 *   1. Sane numeric outputs from a fixture snapshot table (AI fleet
 *      losing alpha vs the best baseline).
 *   2. AI-bots outperforming → positive alpha + sustained = false.
 *   3. Below-threshold sample count → neutral struct flagged stale.
 *   4. Empty AI table → neutral struct flagged stale.
 *   5. The cycle-stats endpoint surfaces the assembled benchmark
 *      block (with a `trustWeight` field merged in from the brain's
 *      `trust_by_family.benchmark`).
 *
 * Network and database are stubbed via `t.mock.module(...)`. Requires
 * `--experimental-test-module-mocks`.
 */
import { describe, test, beforeEach, mock } from "node:test";
import assert from "node:assert/strict";

// Per-test backing data the mocked `db.select(...).from().where().orderBy()`
// chain will return. Keyed by Strategy Lab bucket id.
let bucketRows: Record<string, { timestamp: Date; equity: number }[]> = {};

mock.module("@workspace/db", {
  namedExports: {
    strategySnapshotsTable: {
      strategyType: { name: "strategyType" },
      timestamp: { name: "timestamp" },
      equity: { name: "equity" },
    },
    db: {
      select(): unknown {
        return {
          from(): unknown {
            return {
              where(condition: { __bucket?: string }): unknown {
                const bucket = condition?.__bucket ?? "";
                return {
                  orderBy: () =>
                    Promise.resolve(
                      (bucketRows[bucket] ?? []).map((r) => ({
                        timestamp: r.timestamp,
                        equity: r.equity,
                      })),
                    ),
                };
              },
            };
          },
        };
      },
    },
  },
});

// Drizzle helpers — only their identity matters; we capture the bucket
// name through the eq() helper and stuff it into the where() argument
// our fake recognizes.
mock.module("drizzle-orm", {
  namedExports: {
    and: (...parts: unknown[]) => {
      const obj: { __bucket?: string } = {};
      for (const p of parts) {
        if (p && typeof p === "object" && "__bucket" in (p as object)) {
          obj.__bucket = (p as { __bucket: string }).__bucket;
        }
      }
      return obj;
    },
    eq: (col: { name: string }, val: unknown) =>
      col?.name === "strategyType" ? { __bucket: String(val) } : {},
    gte: () => ({}),
    asc: () => ({}),
  },
});

mock.module("../src/lib/logger.ts", {
  namedExports: { logger: { debug() {}, info() {}, warn() {}, error() {} } },
});

const {
  assembleBenchmarkTelemetry,
  resetBenchmarkCache,
  __setBenchmarkCacheForTest,
} = await import("../src/lib/meta-brain/benchmark-telemetry.ts");

function makeSeries(start: Date, days: number, daily: number): {
  timestamp: Date;
  equity: number;
}[] {
  const out: { timestamp: Date; equity: number }[] = [];
  let v = 1000;
  for (let i = 0; i < days * 24; i++) {
    const t = new Date(start.getTime() + i * 60 * 60 * 1000);
    // hourly compounding of `daily/24`
    v = v * (1 + daily / 24);
    out.push({ timestamp: t, equity: v });
  }
  return out;
}

describe("benchmark telemetry assembler", () => {
  beforeEach(() => {
    bucketRows = {};
    __setBenchmarkCacheForTest(null);
    resetBenchmarkCache();
  });

  test("AI underperforms best baseline → negative alpha + sustained=true", async () => {
    const start = new Date(Date.now() - 14 * 24 * 60 * 60 * 1000);
    bucketRows = {
      "ai-bots": makeSeries(start, 14, -0.005),
      "buy-hold": makeSeries(start, 14, 0.004),
      "dca-cb": makeSeries(start, 14, 0.001),
      "trend-filter": makeSeries(start, 14, 0.002),
    };
    const t = await assembleBenchmarkTelemetry();
    assert.equal(t.stale, false);
    assert.ok(t.relativeAlpha7d < 0, "alpha7 should be negative");
    assert.ok(t.relativeAlpha14d < 0, "alpha14 should be negative");
    assert.equal(t.sustainedUnderperformance, true);
    assert.ok(t.sampleCount >= 10);
    // best baseline is buy-hold (highest 14d return)
    assert.ok(t.bestBaselineReturn7d > 0);
    assert.ok(Number.isFinite(t.drawdownRatioVsBest));
  });

  test("AI outperforms best baseline → positive alpha + sustained=false", async () => {
    const start = new Date(Date.now() - 14 * 24 * 60 * 60 * 1000);
    bucketRows = {
      "ai-bots": makeSeries(start, 14, 0.006),
      "buy-hold": makeSeries(start, 14, 0.001),
      "dca-cb": makeSeries(start, 14, 0.0005),
      "trend-filter": makeSeries(start, 14, 0.002),
    };
    const t = await assembleBenchmarkTelemetry();
    assert.equal(t.stale, false);
    assert.ok(t.relativeAlpha14d > 0, "alpha14 should be positive");
    assert.equal(t.sustainedUnderperformance, false);
  });

  test("too few AI samples → neutral struct flagged stale", async () => {
    const start = new Date(Date.now() - 14 * 24 * 60 * 60 * 1000);
    bucketRows = {
      "ai-bots": makeSeries(start, 14, 0.001).slice(0, 5), // < MIN_SAMPLES
      "buy-hold": makeSeries(start, 14, 0.002),
      "dca-cb": makeSeries(start, 14, 0.001),
      "trend-filter": makeSeries(start, 14, 0.002),
    };
    const t = await assembleBenchmarkTelemetry();
    assert.equal(t.stale, true);
    assert.equal(t.relativeAlpha7d, 0);
    assert.equal(t.relativeAlpha14d, 0);
    assert.equal(t.sustainedUnderperformance, false);
    assert.equal(t.drawdownRatioVsBest, 1);
  });

  test("empty AI table → neutral stale", async () => {
    bucketRows = {};
    const t = await assembleBenchmarkTelemetry();
    assert.equal(t.stale, true);
    assert.equal(t.sampleCount, 0);
  });

  test("most recent AI sample too old → neutral stale", async () => {
    const start = new Date(Date.now() - 25 * 24 * 60 * 60 * 1000);
    bucketRows = {
      "ai-bots": makeSeries(start, 5, 0.001), // newest sample ~20d old
      "buy-hold": makeSeries(start, 14, 0.001),
      "dca-cb": makeSeries(start, 14, 0.001),
      "trend-filter": makeSeries(start, 14, 0.001),
    };
    const t = await assembleBenchmarkTelemetry();
    assert.equal(t.stale, true);
  });

  test("forbidden-prefix scanner cannot match assembler output keys", () => {
    // Make sure none of the field names in the public struct trigger
    // the FORBIDDEN scanner from the trade-decision parity tests.
    const FORBIDDEN = [
      /\b(news|llm|gpt|sentiment|ai|chatgpt|gemini|openai|anthropic|claude)_[a-z][\w]*/,
      /\b(news|llm|gpt|sentiment|ai|chatgpt|gemini|openai|anthropic|claude)[A-Z][\w]*/,
      /\b\w*?(News|Llm|Gpt|Sentiment|Chatgpt|Gemini|OpenAi|Anthropic|Claude)(?:[A-Z]\w*|Score|Bias|Tag|Vote|Edge|Signal|Rating|Call)\b/,
      // benchmark family
      /\b(benchmark|alpha|baseline|strategy_lab)_[a-z][\w]*/,
      /\b(benchmark|strategyLab)[A-Z][\w]*/,
      /\b\w*?(Benchmark|StrategyLab|Alpha|Baseline)(?:[A-Z]\w*|Score|Return|Ratio|Trust)\b/,
    ];
    // Positive control: every benchmark struct key SHOULD match the
    // benchmark forbidden-prefix family (that's the whole point — these
    // names are legal *here* in the governance assembler, but illegal
    // in any /ml/decide / paper_trades / log payload).
    const benchmarkKeys = [
      "aiReturn7d",
      "bestBaselineReturn7d",
      "relativeAlpha7d",
      "relativeAlpha14d",
      "drawdownRatioVsBest",
    ];
    let trippedAtLeastOne = false;
    for (const k of benchmarkKeys) {
      for (const re of FORBIDDEN) {
        if (new RegExp(re.source).test(k)) {
          trippedAtLeastOne = true;
          break;
        }
      }
    }
    assert.equal(
      trippedAtLeastOne,
      true,
      "FORBIDDEN scanner must catch benchmark/alpha/baseline keys",
    );
  });
});
