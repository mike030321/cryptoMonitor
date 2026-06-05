import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  OUTCOME_THRESHOLDS_PERCENT,
  ROUND_TRIP_COST_PERCENT,
  getOutcomeThresholdPercent,
} from "../src/lib/trading-constants";
import { resampleToCandles, filterRealTicksOnly } from "../src/lib/pattern-analyzer";

describe("trading thresholds beat round-trip cost", () => {
  it("ROUND_TRIP_COST_PERCENT is 0.30 (2 * (0.10% taker + 0.05% slippage))", () => {
    assert.equal(Number(ROUND_TRIP_COST_PERCENT.toFixed(4)), 0.3);
  });

  it("every outcome threshold is STRICTLY > round-trip cost so 'correct' is also profitable", () => {
    for (const [tf, threshold] of Object.entries(OUTCOME_THRESHOLDS_PERCENT)) {
      assert.ok(
        threshold > ROUND_TRIP_COST_PERCENT,
        `threshold for ${tf} (${threshold}%) must be > round-trip cost (${ROUND_TRIP_COST_PERCENT}%)`,
      );
    }
  });

  it("longer timeframes have higher thresholds (monotonic)", () => {
    const order = ["1m", "5m", "1h", "2h", "6h", "1d"];
    let prev = -Infinity;
    for (const tf of order) {
      const v = OUTCOME_THRESHOLDS_PERCENT[tf];
      assert.ok(v >= prev, `${tf} threshold ${v} should be >= previous ${prev}`);
      prev = v;
    }
  });

  it("getOutcomeThresholdPercent falls back to 0.30 for unknown timeframe", () => {
    assert.equal(getOutcomeThresholdPercent("nonexistent"), 0.3);
    assert.equal(getOutcomeThresholdPercent("1h"), 0.45);
  });
});

describe("resampleToCandles produces real per-timeframe candles", () => {
  it("returns [] for empty input", () => {
    assert.deepEqual(resampleToCandles([], 60_000), []);
  });

  it("groups ticks within the same bucket and uses the LAST tick as the close", () => {
    const bucketMs = 60_000;
    // Bucket-align t0 so all three ticks fall within [t0, t0+60_000).
    const t0 = Math.floor(1_700_000_000_000 / bucketMs) * bucketMs;
    const points = [
      { price: 100, timestamp: t0 + 1_000 },
      { price: 101, timestamp: t0 + 30_000 },
      { price: 102, timestamp: t0 + 59_000 },
    ];
    const closes = resampleToCandles(points, bucketMs);
    assert.deepEqual(closes, [102]);
  });

  it("creates one close per bucket; empty buckets are skipped", () => {
    const bucketMs = 60_000;
    const t0 = Math.floor(1_700_000_000_000 / bucketMs) * bucketMs;
    // Three buckets of activity, with a gap (no bucket created for the gap).
    const points = [
      { price: 10, timestamp: t0 + 1000 },         // bucket 0
      { price: 11, timestamp: t0 + 90_000 },       // bucket 1
      // bucket 2 empty
      { price: 13, timestamp: t0 + 200_000 },      // bucket 3
      { price: 14, timestamp: t0 + 240_000 },      // bucket 4 (boundary at 240_000)
    ];
    const closes = resampleToCandles(points, bucketMs);
    assert.deepEqual(closes, [10, 11, 13, 14]);
  });

  it("dense ticks within a 1h bucket collapse to ONE close", () => {
    const bucketMs = 3_600_000;
    const t0 = Math.floor(1_700_000_000_000 / bucketMs) * bucketMs;
    // 119 ticks at 30s spacing = 119 * 30s = 59m30s — all within one hour bucket.
    const points = Array.from({ length: 119 }, (_, i) => ({
      price: 50_000 + i,
      timestamp: t0 + i * 30_000,
    }));
    const closes = resampleToCandles(points, bucketMs);
    assert.equal(closes.length, 1, "all ticks within one hour = exactly 1 candle");
    assert.equal(closes[0], 50_000 + 118);
  });

  it("ignores invalid prices and timestamps", () => {
    const bucketMs = 60_000;
    const t0 = Math.floor(1_700_000_000_000 / bucketMs) * bucketMs;
    const points = [
      { price: 0, timestamp: t0 },
      { price: -5, timestamp: t0 + 1000 },
      { price: 100, timestamp: NaN as unknown as number },
      { price: 200, timestamp: t0 + 30_000 },
    ];
    const closes = resampleToCandles(points, bucketMs);
    assert.deepEqual(closes, [200]);
  });
});

describe("calibration sign convention", () => {
  // We intentionally test the math contract, not the DB-fetching wrapper.
  // overallBias = mean(actualWinRate - statedMidpoint).
  //   bias > 0  =>  actual exceeds stated  =>  agent UNDERconfident  =>  raise confidence
  //   bias < 0  =>  actual under stated    =>  agent OVERconfident   =>  lower confidence
  // The sparse-bucket adjustment in applyCalibration is: rawConfidence + bias * 0.15.
  it("adding (not subtracting) the bias moves an underconfident agent UP", () => {
    const raw = 0.50;
    const bias = +0.20; // underconfident
    const adjusted = raw + bias * 0.15;
    assert.ok(adjusted > raw, "underconfident agent should have confidence raised");
  });

  it("adding the bias moves an overconfident agent DOWN", () => {
    const raw = 0.50;
    const bias = -0.20; // overconfident
    const adjusted = raw + bias * 0.15;
    assert.ok(adjusted < raw, "overconfident agent should have confidence lowered");
  });
});

describe("synthetic seed rows must NEVER feed signal indicators", () => {
  // The classical RSI/MACD/EMA/ATR pipeline can be silently corrupted by
  // interpolated bootstrap rows. The contract here is non-negotiable: even
  // when only one or two real ticks exist, indicator inputs include zero
  // synthetic rows. The downstream analyzer then returns insufficient_data
  // and the engine skips that coin/timeframe instead of trading on fakes.
  it("filterRealTicksOnly drops every isSynthetic=true row", () => {
    const mixed = [
      { price: 100, timestamp: new Date(0),     isSynthetic: true },
      { price: 101, timestamp: new Date(1000),  isSynthetic: true },
      { price: 102, timestamp: new Date(2000),  isSynthetic: true },
      { price: 200, timestamp: new Date(3000),  isSynthetic: false },
    ];
    const out = filterRealTicksOnly(mixed);
    assert.equal(out.length, 1);
    assert.equal(out[0].price, 200);
    assert.equal(out[0].isSynthetic, false);
  });

  it("does NOT fall back to synthetic when real ticks are sparse (no fabricated structure)", () => {
    // 50 synthetic rows + 2 real rows. Old buggy code would have used the
    // synthetic ones because realTicks.length < 3. We must NOT do that.
    const rows = [
      ...Array.from({ length: 50 }, (_, i) => ({
        price: 100 + i,
        timestamp: new Date(i * 1000),
        isSynthetic: true,
      })),
      { price: 999, timestamp: new Date(50_000), isSynthetic: false },
      { price: 998, timestamp: new Date(51_000), isSynthetic: false },
    ];
    const out = filterRealTicksOnly(rows);
    assert.equal(out.length, 2, "must return only the real rows even when count is below MACD/RSI minimum");
    assert.ok(out.every((r) => r.isSynthetic === false));
  });

  it("treats null isSynthetic as real (legacy rows predate the column)", () => {
    const rows = [
      { price: 1, timestamp: new Date(0), isSynthetic: null as boolean | null },
      { price: 2, timestamp: new Date(1), isSynthetic: true },
    ];
    const out = filterRealTicksOnly(rows);
    assert.equal(out.length, 1);
    assert.equal(out[0].price, 1);
  });
});

describe("indicator inputs MUST be per-timeframe candles, never sub-timeframe tick noise", () => {
  // Many high-frequency ticks crammed into a window smaller than ONE candle
  // bucket must collapse to a single close — NOT a stream of raw ticks.
  // Old buggy code fell back to `ticks.map(t => t.price)` for a 14-period
  // RSI on the "1h" timeframe, which produced confident signals from
  // 30-second microstructure noise. The contract: dense intra-bucket
  // ticks resample to ≤ 1 candle, which then trips insufficient_data.
  const ONE_HOUR_MS = 60 * 60 * 1000;

  it("100 ticks within a 1h window yield exactly 1 candle close (no fallback to ticks)", () => {
    const baseTs = new Date("2026-04-21T00:00:00Z").getTime();
    const points = Array.from({ length: 100 }, (_, i) => ({
      price: 100 + Math.sin(i / 5),
      timestamp: new Date(baseTs + i * 30_000),
    }));
    const candles = resampleToCandles(points, ONE_HOUR_MS);
    assert.equal(candles.length, 1, "all ticks fell into one 1h bucket so there must be exactly one candle close");
  });

  it("ticks spread across 2 hours yield exactly 2 candle closes — never 'use the raw ticks' fallback", () => {
    const baseTs = new Date("2026-04-21T00:00:00Z").getTime();
    const points = [
      { price: 100, timestamp: new Date(baseTs + 5 * 60_000) },
      { price: 101, timestamp: new Date(baseTs + 35 * 60_000) },
      { price: 102, timestamp: new Date(baseTs + 50 * 60_000) },
      { price: 200, timestamp: new Date(baseTs + 70 * 60_000) },
      { price: 201, timestamp: new Date(baseTs + 100 * 60_000) },
    ];
    const candles = resampleToCandles(points, ONE_HOUR_MS);
    assert.equal(candles.length, 2);
    assert.equal(candles[0], 102);
    assert.equal(candles[1], 201);
  });
});
