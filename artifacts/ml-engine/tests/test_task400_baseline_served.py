"""Task #400 — register the multinomial-logistic baseline as the
slice's served predictor when (a) the booster lost head-to-head on
directional accuracy AND (b) the baseline cleared the verification
gate on its own (DA > 0.50, sum-of-folds n_test >= 200).

This module covers the full baseline-wins pipeline:

  1. `train_one_slice` — when forced down the baseline-served path,
     persists `baseline.joblib` (NOT `model.txt`) and stamps the
     manifest with `served_predictor_kind="baseline"` and metrics
     copied from the baseline CV.
  2. `registry.load_model` — round-trips the baseline-served slice
     and re-materialises the (encoder, lr, priors) triple onto the
     LoadedModel.
  3. `verification.classify_slice` — promotes baseline-served slices
     with the dedicated `lift_baseline_served` reason; the
     verification block bumps both `slices_promoted` (for the
     unified operational total) and `slices_promoted_baseline` (for
     the attribution view).
  4. `failure_analysis.assign_bucket` — attributes baseline-served
     promoted slices to their own cohort
     (`promoted_baseline_served`) so an operator can see which
     slots are riding on the baseline rather than the booster.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.training.labels import build_labeled_frame_for_coin


def _make_high_noise_frame(coins, *, n_per_coin: int = 220, seed: int = 400):
    """Synthetic ticks with enough rows-per-coin that the slice
    crosses MIN_HOLDOUT_ROWS=200 in the walk-forward sum.
    Mirrors the fixture used by the regression-head test (line 820).
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(seed)
    frames = []
    for coin, drift in zip(coins, [0.001, -0.0005]):
        p = 100.0
        ticks = []
        for i in range(n_per_coin):
            p *= (1.0 + drift + rng.normal(0, 0.006))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )


def test_train_one_slice_serves_baseline_when_forced(tmp_path, monkeypatch):
    """Force `_should_serve_baseline` -> True and verify the slice is
    registered as a baseline-served slot.

    Asserts:
      1. `baseline.joblib` exists in the model directory.
      2. `model.txt` was intentionally NOT written.
      3. `manifest.served_predictor_kind == "baseline"`.
      4. Manifest `metrics` were copied from the baseline CV (the
         baseline IS the served head — no separate booster metrics
         to serve).
      5. Returned report exposes `served_predictor_kind="baseline"`
         and a separate `lightgbm_cv_metrics` block (so the operator
         can audit what the booster scored even though we shipped
         the baseline).
      6. `load_model` round-trips the baseline artifact.
    """
    from app.training import registry as registry_module
    from app.training import train as train_mod

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    # Pin the served-baseline gate so the test doesn't depend on the
    # booster genuinely losing head-to-head on a synthetic frame —
    # the gate logic itself is unit-tested separately below.
    monkeypatch.setattr(
        train_mod, "_should_serve_baseline", lambda *a, **k: True,
    )

    coins = ["pepe", "bonk"]
    df = _make_high_noise_frame(coins)

    res = train_mod.train_one_slice(
        df, coin_id="__pooled__", timeframe="1m", vocab=sorted(coins),
    )
    assert res["status"] == "trained", res

    # 1 + 2: baseline persisted, booster file absent.
    model_dir = tmp_path / "__pooled__" / "1m" / res["version"]
    assert (model_dir / "baseline.joblib").exists(), (
        "served=baseline must persist the LR pipeline as baseline.joblib"
    )
    assert not (model_dir / "model.txt").exists(), (
        "booster file must NOT be written when served=baseline; the "
        "served predictor IS the baseline pipeline"
    )

    # 5: report surface for operators / dashboards.
    assert res["served_predictor_kind"] == "baseline"
    assert "lightgbm_cv_metrics" in res, (
        "the booster's CV metrics must still be surfaced even when "
        "the baseline serves, so operators can audit head-to-head"
    )
    # `metrics` (the served-head headline) was copied from baseline.
    assert res["metrics"] == res["baseline_metrics"]

    # 6: round-trip via load_model.
    loaded = registry_module.load_model("__pooled__", "1m", res["version"])
    assert loaded is not None
    # 3 + 4: manifest stamps and metric provenance.
    assert loaded.manifest.served_predictor_kind == "baseline"
    assert loaded.manifest.metrics == res["baseline_metrics"]
    assert loaded.booster is None, (
        "baseline-served slices must not load a booster — there's "
        "nothing on disk to load"
    )
    assert loaded.baseline_artifact is not None
    enc, lr, priors = loaded.baseline_artifact
    # Shape contract on the round-tripped pipeline. We don't invoke
    # `_baseline_predict` directly here — that's exercised by the
    # `/ml/predict` integration suite — but the artifact's structural
    # contract is what matters for the registry round-trip.
    assert priors.shape == (3,), (
        "baseline priors must be a 3-class vector"
    )
    assert hasattr(lr, "predict_proba"), (
        "baseline pipeline's classifier must expose predict_proba "
        "(it's invoked at inference inside _baseline_predict)"
    )
    assert hasattr(enc, "transform"), (
        "baseline pipeline's encoder must expose transform"
    )


def test_should_serve_baseline_gate_logic():
    """Pure-function check of the gate constants used at training time
    (Task #400). The trainer's gate must match the verification
    watchdog's bar exactly: baseline must beat booster on DA AND
    clear DA > 0.50 AND have >= 200 holdout rows.
    """
    from app.training.train import _should_serve_baseline
    from app.training.verification import (
        MIN_DIRECTIONAL_ACCURACY,
        MIN_HOLDOUT_ROWS,
    )

    # Baseline wins, both above the floor, sufficient sample.
    assert _should_serve_baseline(0.45, 0.55, MIN_HOLDOUT_ROWS)
    # Baseline ties booster — not a win.
    assert not _should_serve_baseline(0.55, 0.55, MIN_HOLDOUT_ROWS)
    # Baseline wins but sits at the coin-flip floor.
    assert not _should_serve_baseline(
        0.40, MIN_DIRECTIONAL_ACCURACY, MIN_HOLDOUT_ROWS,
    )
    # Baseline wins above the floor but the holdout is too small.
    assert not _should_serve_baseline(0.45, 0.55, MIN_HOLDOUT_ROWS - 1)
    # NaN sentinels disqualify the comparison.
    assert not _should_serve_baseline(float("nan"), 0.55, MIN_HOLDOUT_ROWS)
    assert not _should_serve_baseline(0.45, float("nan"), MIN_HOLDOUT_ROWS)


def test_verification_promotes_baseline_served_slice():
    """`classify_slice` must promote a slice whose served predictor is
    the baseline, emit the dedicated `lift_baseline_served` reason,
    and `build_verification_block` must bump both
    `slices_promoted` (operational total) and
    `slices_promoted_baseline` (attribution).
    """
    from app.training import verification as v

    # Synthetic slice that already cleared the trainer's served-baseline
    # gate. metrics == baseline_metrics by construction (the trainer
    # copies the baseline CV metrics into `metrics` for the served
    # head); the lift check would otherwise fail.
    slice_rep = {
        "status": "trained",
        "served_predictor_kind": "baseline",
        "metrics": {"directional_accuracy": 0.56},
        "baseline_metrics": {"directional_accuracy": 0.56},
        "fold_metrics": [
            {"n_test": 120}, {"n_test": 90},  # sum 210 > MIN_HOLDOUT_ROWS
        ],
        "directional_call_share": 0.40,  # well below the 0.95 ceiling
    }
    verdict = v.classify_slice(slice_rep)
    assert verdict["promoted"] is True
    assert verdict["reason"] == v.REASON_PROMOTED_BASELINE
    assert verdict["served_predictor_kind"] == "baseline"

    # A booster-served slice with metrics == baseline_metrics still
    # falls through to the lift check and is rejected — the served
    # kind is the only thing that flips the verdict.
    booster_slice = dict(slice_rep)
    booster_slice["served_predictor_kind"] = "lightgbm"
    booster_verdict = v.classify_slice(booster_slice)
    assert booster_verdict["promoted"] is False
    assert booster_verdict["reason"] == v.REASON_NO_LIFT

    # Aggregator: baseline-served promotion bumps BOTH the unified
    # promotion total and the attribution bucket.
    report = {"timeframes": {"1m": {"per_coin": {"pepe": slice_rep}}}}
    block = v.build_verification_block(report, active_coins=["pepe"])
    assert block["passed"] is True
    assert block["counts"]["slices_promoted"] == 1
    assert block["counts"]["slices_promoted_baseline"] == 1


def test_failure_analysis_attributes_baseline_served_to_separate_cohort():
    """`failure_analysis.assign_bucket` must route baseline-served
    promoted slices to `BUCKET_PROMOTED_BASELINE` so the report's
    cohort breakdown shows them separately from booster-served
    promotions.
    """
    from app.training import failure_analysis as fa

    served_baseline = {
        "status": "trained",
        "served_predictor_kind": "baseline",
        "metrics": {"directional_accuracy": 0.56},
        "lightgbm_cv_metrics": {"directional_accuracy": 0.49},
        "per_class_holdout_breakdown": {"down": 70, "stable": 90, "up": 70},
    }
    bucket, reason = fa.assign_bucket(served_baseline, timeframe="1h")
    assert bucket == fa.BUCKET_PROMOTED_BASELINE, (
        f"baseline-served promoted slice landed in {bucket!r} "
        f"instead of the dedicated cohort"
    )
    # Reason text must surface the booster's losing CV DA so the
    # operator can audit the head-to-head close-call without
    # cross-referencing another file.
    assert "served=baseline" in reason
    assert "0.490" in reason

    # Same numeric profile but served=lightgbm → standard promoted bucket.
    served_booster = dict(served_baseline)
    served_booster["served_predictor_kind"] = "lightgbm"
    bucket_b, _ = fa.assign_bucket(served_booster, timeframe="1h")
    assert bucket_b == fa.BUCKET_PROMOTED


def test_train_one_slice_propagates_threshold_source_end_to_end(tmp_path, monkeypatch):
    """Task #459 — integration check that the vol-scaled
    `threshold_pct_source` flows from labels.py → train_one_slice →
    slice report → ModelManifest → registry round-trip without being
    silently dropped at any layer. A future refactor that loses the
    field on any boundary fails this test loudly so the verification
    dashboard / failure-analysis pass never silently regresses to the
    pre-#459 "static-only" view.
    """
    from datetime import datetime, timedelta, timezone

    import numpy as np

    from app.training import labels as labels_mod
    from app.training import registry as registry_module
    from app.training import train as train_mod

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    # Synthetic 1m series with ~0.6 % per-row noise — well above the
    # 0.20 % static 1m baseline (LABEL_THRESHOLDS_PERCENT["1m"] = 0.20),
    # so labels.py MUST widen the threshold to the realized-vol floor.
    rng = np.random.default_rng(459)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    p = 100.0
    ticks: list[tuple[datetime, float]] = []
    for i in range(260):
        p *= 1.0 + rng.normal(0.0, 0.006)
        ticks.append((base + timedelta(seconds=i * 60), p))
    df = labels_mod.build_labeled_frame_for_coin("__t459_int__", "1m", ticks)
    assert not df.empty
    # Sanity: the labelled frame already records the dynamic source
    # (otherwise the integration assertion would test the wrong path).
    sources = set(df["directional_label_threshold_source"].unique().tolist())
    assert sources == {"vol_scaled"}, (
        f"fixture must produce a vol-scaled frame; got sources={sources}"
    )
    stamped_threshold = float(df["directional_label_threshold_pct"].iloc[0])
    static_baseline = labels_mod.LABEL_THRESHOLDS_PERCENT["1m"]
    assert stamped_threshold > static_baseline

    res = train_mod.train_one_slice(
        df, coin_id="__t459_int__", timeframe="1m", vocab=["__t459_int__"],
    )
    assert res["status"] == "trained", res

    # Slice report carries the source AND the chosen value (they back
    # the verification dashboard / failure-analysis bucketing).
    assert res["threshold_pct_source"] == "vol_scaled", (
        f"slice report must surface vol-scaled source; got "
        f"threshold_pct_source={res.get('threshold_pct_source')!r}"
    )
    assert res["threshold_pct"] == pytest.approx(stamped_threshold, rel=1e-6)

    # Round-trip the persisted manifest and assert the field landed on
    # disk (ModelManifest.to_dict() shape — what the registry browser
    # and verification consumers actually read).
    loaded = registry_module.load_model("__t459_int__", "1m", res["version"])
    assert loaded is not None
    assert loaded.manifest.threshold_pct_source == "vol_scaled"
    assert loaded.manifest.threshold_pct == pytest.approx(
        stamped_threshold, rel=1e-6,
    )
    serialized = loaded.manifest.to_dict()
    assert serialized.get("threshold_pct_source") == "vol_scaled", (
        "ModelManifest.to_dict() must include the new field so JSON "
        "consumers (verification-history endpoint, dashboards) see it"
    )
