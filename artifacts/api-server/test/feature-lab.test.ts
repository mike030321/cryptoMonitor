import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  evaluatePromotion,
  listCandidates,
  getApprovedFeatures,
  summarizeApprovedFeatureApplications,
  unquarantineCandidate,
  assessUnquarantineRegression,
  buildUnquarantineRegressionAssessment,
  UnquarantineRegressionStillPresentError,
  UNQUARANTINE_REGRESSION_THRESHOLD,
  APPROVED_FEATURES_SETTING_KEY,
  QUARANTINED_FEATURES_SETTING_KEY,
  MIN_ABLATION_SAMPLES,
} from "../src/lib/feature-lab.ts";

describe("feature-lab promotion eligibility", () => {
  it("constants surface stable names", () => {
    assert.equal(APPROVED_FEATURES_SETTING_KEY, "feature_lab.approved");
    assert.equal(QUARANTINED_FEATURES_SETTING_KEY, "feature_lab.quarantined");
    assert.ok(MIN_ABLATION_SAMPLES > 0);
  });

  it("unquarantineCandidate refuses an unknown candidate", async () => {
    await assert.rejects(
      () => unquarantineCandidate({ candidateId: 0, approvedBy: "test" }),
      /not found/,
    );
  });

  it("UNQUARANTINE_REGRESSION_THRESHOLD has a sane default", () => {
    assert.ok(UNQUARANTINE_REGRESSION_THRESHOLD > 0);
    assert.ok(UNQUARANTINE_REGRESSION_THRESHOLD < 1);
  });

  it("assessUnquarantineRegression on an unknown candidate returns no_quarantine_record", async () => {
    const a = await assessUnquarantineRegression(0);
    assert.equal(a.status, "no_quarantine_record");
    assert.equal(a.regressionStillPresent, false);
    assert.equal(a.perTimeframe.length, 0);
  });

  it("buildUnquarantineRegressionAssessment: latest log_loss recovered → not present", () => {
    const a = buildUnquarantineRegressionAssessment(
      {
        name: "rsi_sq",
        transformKind: "rsi_squared",
        sourceColumn: null,
        quarantinedAt: "2026-01-01T00:00:00Z",
        reason: "validation_regression",
        detail: {
          timeframes: [
            { timeframe: "1h", prior_log_loss: 0.50, current_log_loss: 0.62, delta_log_loss: 0.12 },
            { timeframe: "4h", prior_log_loss: 0.48, current_log_loss: 0.55, delta_log_loss: 0.07 },
          ],
        },
      },
      {
        timeframes: {
          // Both pooled log_losses now sit AT prior — fully recovered.
          "1h": { pooled: { status: "trained", metrics: { log_loss: 0.50 } } },
          "4h": { pooled: { status: "trained", metrics: { log_loss: 0.48 } } },
        },
      },
      null,
      0.05,
    );
    assert.equal(a.status, "ok");
    assert.equal(a.regressionStillPresent, false);
    assert.equal(a.worstOriginalDelta, 0.12);
    assert.equal(a.worstLatestDelta, 0);
    assert.equal(a.perTimeframe[0].recovered, true);
  });

  it("buildUnquarantineRegressionAssessment: latest log_loss still high → still present", () => {
    const a = buildUnquarantineRegressionAssessment(
      {
        name: "rsi_sq",
        transformKind: "rsi_squared",
        sourceColumn: null,
        quarantinedAt: "2026-01-01T00:00:00Z",
        reason: "validation_regression",
        detail: {
          timeframes: [
            { timeframe: "1h", prior_log_loss: 0.50, current_log_loss: 0.62, delta_log_loss: 0.12 },
          ],
        },
      },
      {
        timeframes: {
          // Latest pooled log_loss is 0.59 — still 0.09 above prior, > 0.05.
          "1h": { pooled: { status: "trained", metrics: { log_loss: 0.59 } } },
        },
      },
      null,
      0.05,
    );
    assert.equal(a.status, "ok");
    assert.equal(a.regressionStillPresent, true);
    assert.ok(a.worstLatestDelta != null && a.worstLatestDelta > 0.05);
    assert.equal(a.perTimeframe[0].recovered, false);
  });

  it("buildUnquarantineRegressionAssessment: missing training report → status=no_training_report and still present", () => {
    const a = buildUnquarantineRegressionAssessment(
      {
        name: "rsi_sq",
        transformKind: "rsi_squared",
        sourceColumn: null,
        quarantinedAt: "2026-01-01T00:00:00Z",
        reason: "validation_regression",
        detail: {
          timeframes: [
            { timeframe: "1h", prior_log_loss: 0.50, current_log_loss: 0.62, delta_log_loss: 0.12 },
          ],
        },
      },
      null,
      "ml-engine /ml/training/report 503",
      0.05,
    );
    assert.equal(a.status, "no_training_report");
    assert.equal(a.regressionStillPresent, true);
    assert.equal(a.trainingReportError, "ml-engine /ml/training/report 503");
  });

  it("buildUnquarantineRegressionAssessment: report present but pooled slot missing → status=unknown", () => {
    const a = buildUnquarantineRegressionAssessment(
      {
        name: "rsi_sq",
        transformKind: "rsi_squared",
        sourceColumn: null,
        quarantinedAt: "2026-01-01T00:00:00Z",
        reason: "validation_regression",
        detail: {
          timeframes: [
            { timeframe: "1h", prior_log_loss: 0.50, current_log_loss: 0.62, delta_log_loss: 0.12 },
          ],
        },
      },
      // Pooled status is not "trained" → no latest log_loss extractable.
      { timeframes: { "1h": { pooled: { status: "insufficient_data" } } } },
      null,
      0.05,
    );
    assert.equal(a.status, "unknown");
    assert.equal(a.regressionStillPresent, true);
    assert.equal(a.perTimeframe[0].latestPooledLogLoss, null);
  });

  it("buildUnquarantineRegressionAssessment: legacy quarantine row without detail.timeframes → status=unknown", () => {
    const a = buildUnquarantineRegressionAssessment(
      {
        name: "legacy_feat",
        transformKind: "rsi_squared",
        sourceColumn: null,
        quarantinedAt: "2026-01-01T00:00:00Z",
        reason: "validation_regression",
        detail: null,
      },
      { timeframes: { "1h": { pooled: { status: "trained", metrics: { log_loss: 0.50 } } } } },
      null,
    );
    assert.equal(a.status, "unknown");
    assert.equal(a.regressionStillPresent, true);
    assert.equal(a.perTimeframe.length, 0);
  });

  it("UnquarantineRegressionStillPresentError carries an assessment payload", () => {
    const err = new UnquarantineRegressionStillPresentError({
      status: "ok",
      threshold: 0.05,
      reason: null,
      quarantinedAt: null,
      originalReason: "validation_regression",
      worstOriginalDelta: 0.1,
      worstLatestDelta: 0.09,
      regressionStillPresent: true,
      perTimeframe: [],
    });
    assert.equal(err.code, "regression_still_present");
    assert.equal(err.assessment.worstLatestDelta, 0.09);
    assert.match(err.message, /regression still present/);
  });

  it("evaluatePromotion on a never-ablated candidate is not eligible", async () => {
    // ID 0 cannot exist (serial starts at 1) — so reports = [], not eligible.
    const v = await evaluatePromotion(0);
    assert.equal(v.eligible, false);
    assert.ok(v.reasons.includes("no_ablation_reports"));
    assert.equal(v.bestReport, null);
  });

  it("listCandidates returns an array (possibly empty)", async () => {
    const rows = await listCandidates();
    assert.ok(Array.isArray(rows));
  });

  it("getApprovedFeatures returns an array", async () => {
    const arr = await getApprovedFeatures();
    assert.ok(Array.isArray(arr));
    for (const f of arr) {
      assert.equal(typeof f.name, "string");
      assert.equal(typeof f.transformKind, "string");
      assert.ok(Array.isArray(f.inputs));
    }
  });

  it("summarizeApprovedFeatureApplications returns one entry per approved feature", async () => {
    const summary = await summarizeApprovedFeatureApplications({ limitPerFeature: 5 });
    assert.ok(Array.isArray(summary));
    const approved = await getApprovedFeatures();
    assert.equal(summary.length, approved.length);
    for (const s of summary) {
      assert.equal(typeof s.name, "string");
      assert.ok(Array.isArray(s.appliedIn));
      for (const e of s.appliedIn) {
        assert.equal(typeof e.registryId, "number");
        assert.equal(typeof e.modelVersion, "string");
        assert.equal(typeof e.timeframe, "string");
      }
    }
  });
});
