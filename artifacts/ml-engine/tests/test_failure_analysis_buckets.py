"""Bucket-assignment unit tests for the auto failure-analysis report.

Covers the boundary semantics that must stay locked to the verification
gate so the dashboard and the verification block never disagree on
whether a slice is promoted (Task #401).
"""
from __future__ import annotations

from app.training.failure_analysis import (
    BUCKET_FEATURES_OR_LABELS,
    BUCKET_INSUFFICIENT_SAMPLE,
    BUCKET_PROMOTED,
    assign_bucket,
)


def _trained_slice(da: float, n_holdout: int) -> dict:
    """Mint a `trained` slice rep with the per-class breakdown the
    failure analyser reads to derive the holdout count."""
    # `_holdout_rows` sums `per_class_holdout_breakdown.values()`.
    return {
        "status": "trained",
        "metrics": {"directional_accuracy": da},
        "baseline_metrics": {"directional_accuracy": 0.40},
        "per_class_holdout_breakdown": {"-1": n_holdout // 3,
                                          "0": n_holdout // 3,
                                          "1": n_holdout - 2 * (n_holdout // 3)},
    }


def test_assign_bucket_1d_above_new_floor_is_promoted():
    """A 1d slice with DA strictly above the per-tf floor (0.530)
    promotes — sanity-check the upper happy-path."""
    bucket, _ = assign_bucket(_trained_slice(0.555, 265), timeframe="1d")
    assert bucket == BUCKET_PROMOTED


def test_assign_bucket_1d_at_new_floor_is_not_promoted():
    """Strict-greater-than parity with `verification.classify_slice`:
    `verification` retires `da == min_da` as `below_coinflip`, so the
    failure-analysis bucket assignment MUST also leave a slice that
    ties the 1d floor outside `BUCKET_PROMOTED`. Without strict
    semantics the dashboard would call this slice promoted while the
    verification block reported it `below_coinflip`."""
    bucket, _ = assign_bucket(_trained_slice(0.530, 265), timeframe="1d")
    assert bucket != BUCKET_PROMOTED


def test_assign_bucket_1d_below_new_floor_is_not_promoted():
    """The regression case the per-tf floor is designed to catch:
    `dogwifcoin/1d` cleared the legacy gate at DA 0.516. Under the
    0.530 1d floor it must NOT count as promoted in the failure
    analysis bucket counts either."""
    bucket, _ = assign_bucket(_trained_slice(0.516, 265), timeframe="1d")
    assert bucket != BUCKET_PROMOTED


def test_assign_bucket_1h_at_legacy_floor_is_not_promoted():
    """1h still uses the default 0.50 floor with strict-greater-than
    semantics. A 1h slice with DA exactly 0.50 must NOT be promoted —
    same boundary semantics as the verification gate."""
    bucket, _ = assign_bucket(_trained_slice(0.500, 250), timeframe="1h")
    assert bucket != BUCKET_PROMOTED


def test_assign_bucket_1h_just_above_legacy_floor_is_promoted():
    """1h with DA strictly above 0.50 promotes (default unchanged)."""
    bucket, _ = assign_bucket(_trained_slice(0.510, 250), timeframe="1h")
    assert bucket == BUCKET_PROMOTED


def test_assign_bucket_insufficient_holdout_skips_promotion():
    """Holdout below the floor is `insufficient_sample`, not promoted,
    independent of DA."""
    bucket, _ = assign_bucket(_trained_slice(0.620, 50), timeframe="1d")
    assert bucket == BUCKET_INSUFFICIENT_SAMPLE


def test_assign_bucket_missing_slice_is_insufficient_sample():
    """A None slice (untrained / missing) is always
    `insufficient_sample`."""
    bucket, _ = assign_bucket(None, timeframe="1d")
    assert bucket == BUCKET_INSUFFICIENT_SAMPLE


def test_assign_bucket_red_gate_below_floor_falls_to_features_or_labels():
    """A trained slice with adequate holdout but a DA below the floor,
    and no calibration / collapse signal, falls into the
    'features-or-labels' bucket (the default for 'red but not retired')
    — confirms the per-tf floor change doesn't leak into the residual
    bucket logic."""
    bucket, _ = assign_bucket(_trained_slice(0.520, 265), timeframe="1d")
    assert bucket == BUCKET_FEATURES_OR_LABELS
