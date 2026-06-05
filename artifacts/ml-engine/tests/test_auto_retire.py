"""Task #236 — auto-retire approved features that fail validation.

The persistence path needs a Postgres pool, which the unit-test suite
doesn't have, so these tests focus on the pure helper
`diagnose_regressions` and verify the `auto_retire_after_training`
entry point degrades gracefully when DATABASE_URL is unset (matching
`fetch_approved_features`'s contract in `approved_features.py`).
"""
from __future__ import annotations

import asyncio
import os

from app.training.auto_retire import (
    DEFAULT_LOG_LOSS_REGRESSION_THRESHOLD,
    auto_retire_after_training,
    diagnose_regressions,
)


def _tf_report(log_loss: float, applied: list[str]) -> dict:
    return {
        "pooled": {
            "status": "trained",
            "metrics": {"log_loss": log_loss, "auc": 0.6},
        },
        "per_coin": {},
        "approved_features_applied": list(applied),
    }


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_diagnose_flags_first_appearance_with_regression():
    prior = {
        "timeframes": {
            "1m": _tf_report(0.50, applied=["log_rv"]),
            "5m": _tf_report(0.45, applied=["log_rv"]),
        },
    }
    current = {
        "timeframes": {
            # 1m: regression > 0.05, NEW feature appears => quarantine
            "1m": _tf_report(0.60, applied=["log_rv", "rsi_sq_band"]),
            # 5m: tiny regression, no new features => no decision
            "5m": _tf_report(0.46, applied=["log_rv"]),
        },
    }
    decisions = diagnose_regressions(current, prior)
    assert len(decisions) == 1
    d = decisions[0]
    assert d["timeframe"] == "1m"
    assert d["feature_name"] == "rsi_sq_band"
    assert d["delta_log_loss"] > DEFAULT_LOG_LOSS_REGRESSION_THRESHOLD
    assert d["first_appearance"] is True


def test_diagnose_skips_when_regression_within_threshold():
    prior = {"timeframes": {"1m": _tf_report(0.50, applied=[])}}
    current = {
        "timeframes": {
            "1m": _tf_report(0.52, applied=["new_feat"]),
        },
    }
    # +0.02 < 0.05 default threshold
    assert diagnose_regressions(current, prior) == []


def test_diagnose_skips_when_no_new_features():
    prior = {"timeframes": {"1m": _tf_report(0.40, applied=["a", "b"])}}
    current = {
        "timeframes": {
            "1m": _tf_report(0.80, applied=["a", "b"]),  # huge regression
        },
    }
    # Regression is real but no NEW feature appeared this run, so the
    # auto-retire heuristic has no candidate to quarantine.
    assert diagnose_regressions(current, prior) == []


def test_diagnose_handles_missing_prior_report():
    current = {"timeframes": {"1m": _tf_report(0.60, applied=["x"])}}
    assert diagnose_regressions(current, None) == []
    assert diagnose_regressions(current, {}) == []


def test_diagnose_skips_when_pooled_not_trained():
    prior = {"timeframes": {"1m": _tf_report(0.50, applied=["a"])}}
    current = {
        "timeframes": {
            "1m": {
                "pooled": {"status": "prior_pooled"},
                "approved_features_applied": ["a", "newish"],
            },
        },
    }
    # No trained pooled metrics this run => no comparison possible.
    assert diagnose_regressions(current, prior) == []


def test_diagnose_handles_nan_log_loss():
    prior = {"timeframes": {"1m": _tf_report(float("nan"), applied=[])}}
    current = {"timeframes": {"1m": _tf_report(0.60, applied=["x"])}}
    assert diagnose_regressions(current, prior) == []


def test_diagnose_multiple_new_features_all_quarantined():
    prior = {"timeframes": {"1m": _tf_report(0.40, applied=[])}}
    current = {
        "timeframes": {
            "1m": _tf_report(0.55, applied=["alpha", "beta", "gamma"]),
        },
    }
    decs = diagnose_regressions(current, prior)
    names = sorted(d["feature_name"] for d in decs)
    assert names == ["alpha", "beta", "gamma"]
    for d in decs:
        assert d["timeframe"] == "1m"
        assert d["delta_log_loss"] > DEFAULT_LOG_LOSS_REGRESSION_THRESHOLD


def test_auto_retire_noop_when_no_decisions():
    prior = {"timeframes": {"1m": _tf_report(0.50, applied=["a"])}}
    current = {"timeframes": {"1m": _tf_report(0.51, applied=["a"])}}
    summary = _run(auto_retire_after_training(current, prior, []))
    assert summary["status"] == "noop"
    assert summary["decisions"] == []
    assert summary["quarantined_names"] == []


def test_auto_retire_skips_persist_without_database_url(monkeypatch):
    """When DATABASE_URL is unset (default in the unit-test suite) the
    persist step is intentionally skipped; we still surface the
    decisions so the report records that we *would* have quarantined
    the feature once the DB is available.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    prior = {"timeframes": {"1m": _tf_report(0.40, applied=[])}}
    current = {"timeframes": {"1m": _tf_report(0.55, applied=["new_feat"])}}
    approved = [
        {"name": "new_feat", "transform_kind": "rsi_squared",
         "source_column": None},
    ]
    summary = _run(auto_retire_after_training(current, prior, approved))
    assert summary["status"] == "skipped"
    assert summary["reason"] == "no_database_url"
    assert len(summary["decisions"]) == 1
    assert summary["decisions"][0]["feature_name"] == "new_feat"


def test_diagnose_uses_full_previous_run_report_not_prior_pooled_fallback():
    """Regression guard: in `run_training` a per-timeframe prior-pooled
    fallback must NOT clobber the previous-run report passed to
    `diagnose_regressions`. The helper should only ever consume the
    multi-timeframe report shape, and a fallback dict (no `timeframes`
    key) is treated as "no prior data" — never crashing.
    """
    # Shape produced by `_train_prior_pooled` — looks like a single
    # pooled slot, not a full run report. If `run_training` ever hands
    # this in by mistake (the bug fixed in this task) we must degrade
    # gracefully: zero decisions, no traceback.
    fallback_shaped = {
        "coin_id": "__pooled__", "timeframe": "1m",
        "status": "trained", "model_kind": "prior",
        "n_labels": 100, "prior_probs": [0.3, 0.4, 0.3],
    }
    current = {"timeframes": {"1m": _tf_report(0.99, applied=["new"])}}
    decisions = diagnose_regressions(current, fallback_shaped)
    assert decisions == []


def test_auto_retire_threshold_override_via_env(monkeypatch):
    # diagnose_regressions accepts an explicit threshold override; the
    # env var feeds the module-level default but we test the parameter
    # directly so the test is hermetic.
    prior = {"timeframes": {"1m": _tf_report(0.50, applied=[])}}
    current = {"timeframes": {"1m": _tf_report(0.53, applied=["new"])}}
    # Default threshold (0.05) wouldn't fire on +0.03; a tighter
    # threshold of 0.01 should.
    assert diagnose_regressions(current, prior) == []
    decs = diagnose_regressions(current, prior, threshold=0.01)
    assert len(decs) == 1
    assert decs[0]["feature_name"] == "new"
    assert decs[0]["threshold"] == 0.01
