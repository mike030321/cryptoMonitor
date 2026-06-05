import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  computeCalibrationDrift,
  computeDistributionDrift,
  computeFeatureDrift,
  DRIFT_THRESHOLDS,
} from "../src/lib/drift-tracker.ts";

describe("drift-tracker", () => {
  it("DRIFT_THRESHOLDS expose calibration / distribution / feature scalars", () => {
    assert.equal(typeof DRIFT_THRESHOLDS.calibration, "number");
    assert.equal(typeof DRIFT_THRESHOLDS.prediction_distribution, "number");
    assert.equal(typeof DRIFT_THRESHOLDS.feature, "number");
    assert.ok(DRIFT_THRESHOLDS.calibration > 0 && DRIFT_THRESHOLDS.calibration < 1);
    assert.ok(DRIFT_THRESHOLDS.feature >= 1, "feature z-threshold should be >=1σ");
  });

  it("computeCalibrationDrift returns a non-breached result when sample count is small", async () => {
    const r = await computeCalibrationDrift({ windowHours: 1 });
    assert.ok(typeof r.score === "number");
    assert.ok(typeof r.nSamples === "number" && r.nSamples >= 0);
    if (r.nSamples < 50) assert.equal(r.breached, false, "sub-MIN_SAMPLES result must not breach");
    assert.ok(Array.isArray(r.buckets) && r.buckets.length === 10, "10 deciles always");
  });

  it("computeDistributionDrift returns smoothed shares that sum ~= 1", async () => {
    const r = await computeDistributionDrift({ windowHours: 24 });
    const recentSum = Object.values(r.recent).reduce((a, b) => a + b, 0);
    const baselineSum = Object.values(r.baseline).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(recentSum - 1) < 0.05, `recent shares should sum ~1, got ${recentSum}`);
    assert.ok(Math.abs(baselineSum - 1) < 0.05, `baseline shares should sum ~1, got ${baselineSum}`);
    assert.ok(r.score >= 0, "KL divergence is non-negative");
  });

  it("computeFeatureDrift surfaces per-feature breakdown", async () => {
    const r = await computeFeatureDrift({ windowHours: 24 });
    assert.ok(typeof r.score === "number" && r.score >= 0);
    assert.ok(Array.isArray(r.perFeature));
    for (const f of r.perFeature) {
      assert.equal(typeof f.name, "string");
      assert.equal(typeof f.zScore, "number");
    }
  });
});
