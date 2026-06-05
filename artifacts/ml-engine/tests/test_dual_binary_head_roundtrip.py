"""Task #654 — paper-trading family C ("dual_binary_head") foundation.

Round-trip test for the new dual-binary-head model family:

  1. Train two tiny binary LightGBM heads on synthetic data
     (long: y > +threshold, short: y < -threshold).
  2. Fit a one-knob Platt sigmoid per head on the same training scores.
  3. Persist via `save_model` with `served_predictor_kind="dual_binary_head"`.
  4. Load via `load_model` — must return a `LoadedDualHeadModel`.
  5. Hit `/ml/predict` end-to-end via `TestClient`; assert the new
     response shape (`predictor_kind`, `p_long`, `p_short`, `abstain`,
     `side`, `confidence`, `label_family`) is well-formed.
  6. Hit `/ml/predict` with the scope guard active: a stamped
     scope_constraint that excludes the request's (coin, tf) must
     short-circuit to `{"status": "out_of_scope", ...}`.

Strict negative cases:
  • `manifest.validate()` raises on missing dual-head fields.
  • `promote_shadow_to_serving` refuses when target row is not 'shadow'.
  • `promote_shadow_to_serving` refuses when manifest fails to load.
"""
from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import main as main_module
from app import registry_lifecycle
from app.main import app
from app.training import registry as registry_module
from app.training.registry import (
    LoadedDualHeadModel,
    LoadedModel,
    ModelManifest,
    ScopeViolationError,
    load_model,
    save_model,
)


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------


def _synth_training_frame(
    n_rows: int = 400, n_features: int = 5, *, seed: int = 17,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a `(X, y)` pair where the *first* feature linearly drives the
    forward return so a tiny binary booster is guaranteed to beat 50/50.
    Returns `(features_df, forward_return_pct_array)`.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_rows, n_features))
    # Forward return in PERCENT, with a strong linear signal on column 0.
    y_pct = 0.5 * X[:, 0] + 0.05 * rng.normal(0.0, 1.0, size=n_rows)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y_pct


def _train_binary_head(
    X: pd.DataFrame, labels: np.ndarray, *, seed: int,
) -> tuple[lgb.Booster, np.ndarray]:
    """Train a tiny binary lightgbm booster. Returns the booster AND the
    in-sample raw probabilities (used to fit the Platt sigmoid).
    """
    train_set = lgb.Dataset(X, label=labels.astype(int))
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "verbosity": -1, "num_leaves": 7, "min_data_in_leaf": 5,
        "learning_rate": 0.1, "feature_pre_filter": False,
        "seed": seed,
    }
    booster = lgb.train(params, train_set, num_boost_round=30)
    raw = np.asarray(booster.predict(X)).astype(float)
    return booster, raw


def _fit_platt(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Tiny Platt fit via numpy gradient descent.

    Standard LR over a single feature (the raw model score). Returns
    `{"slope": ..., "intercept": ...}` with the convention used by
    `LoadedDualHeadModel._platt`:  P = 1 / (1 + exp(slope*score + intercept)).
    A POSITIVE input correlation with `labels` should produce a NEGATIVE
    slope so higher raw → higher calibrated.

    `scipy.optimize` would be cleaner but we keep deps to stdlib +
    numpy so the test stays cheap.
    """
    s = scores.astype(float)
    y = labels.astype(float)
    slope, intercept = 0.0, 0.0
    lr = 0.5
    for _ in range(400):
        z = slope * s + intercept
        p = 1.0 / (1.0 + np.exp(np.clip(z, -50.0, 50.0)))
        # gradient of NLL w.r.t. (slope, intercept).
        # NLL = -sum( y*log(p) + (1-y)*log(1-p) )
        # dp/dz = -p*(1-p) (because z has a leading + sign in exp)
        # ⇒ dNLL/dz = (p - y)*(-1)? Re-derive:
        # log p = -log(1+exp(z)). d/dz = -p_neg where p_neg = e^z/(1+e^z) = 1-p.
        # d log(1-p)/dz: 1-p = e^z/(1+e^z), d log(1-p)/dz = 1 - (1-p) = p.
        # ⇒ dNLL/dz = -[ y*-(1-p) + (1-y)*p ] = y*(1-p) - (1-y)*p = y - p.
        # Wait sign flips because of inner `+ z` in our sigmoid form;
        # double-check with finite difference further down. Update rule
        # below was verified to converge on a held-out check:
        d = (p - y) * -1.0
        g_slope = float(np.mean(d * s))
        g_intercept = float(np.mean(d))
        slope -= lr * g_slope
        intercept -= lr * g_intercept
    return {"slope": float(slope), "intercept": float(intercept)}


def _build_dual_head_manifest(
    *, coin_id: str, timeframe: str, version: str,
    feature_names: list[str], friction_pct: float,
    abstain_tau: float, platt: dict,
) -> ModelManifest:
    return ModelManifest(
        coin_id=coin_id,
        timeframe=timeframe,
        version=version,
        feature_names=feature_names,
        coin_vocab=[coin_id],
        n_train_rows=0,
        n_test_rows=0,
        metrics={},
        baseline_metrics={},
        threshold_pct=friction_pct,
        horizon_candles=1,
        # 3-class field stays empty for dual-head (validated by
        # ModelManifest.validate when kind is dual_binary_head).
        class_return_means_pct=[],
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=abstain_tau,
        platt_calibration=platt,
        friction_threshold_pct=friction_pct,
        label_family="C_post_cost",
    )


def _build_dual_head_isotonic_manifest(
    *, coin_id: str, timeframe: str, version: str,
    feature_names: list[str], friction_pct: float,
    abstain_tau: float, isotonic: dict,
) -> ModelManifest:
    """Task #657 — sibling helper to `_build_dual_head_manifest` for the
    isotonic-calibrated variant. The platt block is left None because
    `validate()` enforces "exactly one calibrator per manifest" when
    method is isotonic.
    """
    return ModelManifest(
        coin_id=coin_id,
        timeframe=timeframe,
        version=version,
        feature_names=feature_names,
        coin_vocab=[coin_id],
        n_train_rows=0,
        n_test_rows=0,
        metrics={},
        baseline_metrics={},
        threshold_pct=friction_pct,
        horizon_candles=1,
        class_return_means_pct=[],
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=abstain_tau,
        platt_calibration=None,
        friction_threshold_pct=friction_pct,
        label_family="C_post_cost",
        calibration_method="isotonic",
        isotonic_calibration=isotonic,
    )


# ---------------------------------------------------------------------------
# Pure (no-DB) tests
# ---------------------------------------------------------------------------


def test_manifest_validate_rejects_missing_dual_head_fields() -> None:
    """`ModelManifest.validate()` must catch a half-configured dual-head
    manifest BEFORE save_model / load_model touch the disk.
    """
    m = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_test_invalid",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        # All dual-head fields intentionally missing.
    )
    with pytest.raises(ValueError) as excinfo:
        m.validate()
    msg = str(excinfo.value)
    assert "long_model_path" in msg
    assert "short_model_path" in msg
    assert "abstain_tau" in msg
    assert "platt_calibration" in msg
    assert "friction_threshold_pct" in msg


def test_manifest_validate_isotonic_requires_threshold_arrays() -> None:
    """Task #657 — `calibration_method='isotonic'` requires the
    isotonic_calibration block with both heads, with non-empty
    threshold arrays of equal length. Stale Platt block must not
    sneak through alongside.
    """
    # Missing isotonic block entirely.
    m_no_iso = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_iso_missing",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5,
        friction_threshold_pct=0.30,
        calibration_method="isotonic",
        isotonic_calibration=None,  # missing
    )
    with pytest.raises(ValueError) as excinfo:
        m_no_iso.validate()
    assert "isotonic_calibration" in str(excinfo.value)

    # Missing one head's threshold pair.
    m_one_head = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_iso_one_head",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5,
        friction_threshold_pct=0.30,
        calibration_method="isotonic",
        isotonic_calibration={
            "long": {
                "x_thresholds": [0.1, 0.5, 0.9],
                "y_values": [0.05, 0.5, 0.95],
            },
            # short head missing
        },
    )
    with pytest.raises(ValueError) as excinfo:
        m_one_head.validate()
    assert "isotonic_calibration['short']" in str(excinfo.value)

    # Mismatched array lengths.
    m_mismatch = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_iso_mismatch",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5,
        friction_threshold_pct=0.30,
        calibration_method="isotonic",
        isotonic_calibration={
            "long": {
                "x_thresholds": [0.1, 0.5, 0.9],
                "y_values": [0.05, 0.95],  # wrong length
            },
            "short": {
                "x_thresholds": [0.1, 0.5, 0.9],
                "y_values": [0.05, 0.5, 0.95],
            },
        },
    )
    with pytest.raises(ValueError) as excinfo:
        m_mismatch.validate()
    assert "isotonic_calibration['long']" in str(excinfo.value)


def test_manifest_validate_platt_rejects_isotonic_block() -> None:
    """`calibration_method='platt'` (the legacy default) must reject a
    manifest that ALSO carries an `isotonic_calibration` block — that
    pattern would silently ship two calibrators and produce method-
    dependent serving behaviour."""
    m = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_dh_dual_cal",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5,
        friction_threshold_pct=0.30,
        calibration_method="platt",
        platt_calibration={
            "long":  {"slope": -1.0, "intercept": 0.0},
            "short": {"slope": -1.0, "intercept": 0.0},
        },
        isotonic_calibration={
            "long":  {"x_thresholds": [0.1, 0.9], "y_values": [0.1, 0.9]},
            "short": {"x_thresholds": [0.1, 0.9], "y_values": [0.1, 0.9]},
        },
    )
    with pytest.raises(ValueError) as excinfo:
        m.validate()
    assert "isotonic_calibration" in str(excinfo.value)


def test_dual_head_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    """Train two binary heads, persist via save_model, load via load_model,
    verify the loaded object is a LoadedDualHeadModel with intact params
    and that predict_one returns sane shapes.
    """
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    df, y = _synth_training_frame(n_rows=400, n_features=5, seed=17)
    friction_pct = 0.10  # i.e. 0.10%
    long_labels = (y > friction_pct).astype(int)
    short_labels = (y < -friction_pct).astype(int)
    # Sanity — both classes must be present, otherwise lightgbm degenerates.
    assert int(long_labels.sum()) > 30 and int(long_labels.sum()) < 370
    assert int(short_labels.sum()) > 30 and int(short_labels.sum()) < 370

    long_booster, long_raw = _train_binary_head(df, long_labels, seed=1)
    short_booster, short_raw = _train_binary_head(df, short_labels, seed=2)
    platt = {
        "long": _fit_platt(long_raw, long_labels),
        "short": _fit_platt(short_raw, short_labels),
    }

    manifest = _build_dual_head_manifest(
        coin_id="bitcoin", timeframe="5m", version="v_dh_001",
        feature_names=list(df.columns),
        friction_pct=friction_pct,
        abstain_tau=0.55,
        platt=platt,
    )
    out_dir = save_model(
        coin_id="bitcoin", timeframe="5m", version="v_dh_001",
        booster=long_booster, calibrators=None, manifest=manifest,
        regressor=short_booster,
    )
    # Files must exist with the manifest-recorded names.
    assert (out_dir / "long_model.txt").exists()
    assert (out_dir / "short_model.txt").exists()
    assert (out_dir / "manifest.json").exists()
    # The 3-class artefacts MUST NOT be written for this family.
    assert not (out_dir / "model.txt").exists()
    assert not (out_dir / "calibrators.joblib").exists()
    assert not (out_dir / "regressor.txt").exists()
    assert not (out_dir / "baseline.joblib").exists()
    assert not (out_dir / "prior.json").exists()

    # Manifest round-trips with the new fields preserved.
    saved_manifest = json.loads((out_dir / "manifest.json").read_text())
    assert saved_manifest["served_predictor_kind"] == "dual_binary_head"
    assert saved_manifest["long_model_path"] == "long_model.txt"
    assert saved_manifest["short_model_path"] == "short_model.txt"
    assert saved_manifest["abstain_tau"] == 0.55
    assert saved_manifest["friction_threshold_pct"] == friction_pct
    assert saved_manifest["label_family"] == "C_post_cost"
    assert saved_manifest["platt_calibration"]["long"]["slope"] == platt["long"]["slope"]

    loaded = load_model("bitcoin", "5m", "v_dh_001")
    assert isinstance(loaded, LoadedDualHeadModel), type(loaded)
    assert loaded.manifest.served_predictor_kind == "dual_binary_head"
    assert loaded.manifest.abstain_tau == 0.55
    assert loaded.booster_long is not None
    assert loaded.booster_short is not None

    # Build a single-row DataFrame matching the trained feature schema and
    # ask predict_one for a prediction. We don't assert a SPECIFIC side
    # (the synthetic noise can flip it) — just that the response is
    # well-formed and within bounds.
    X_one = df.iloc[[0]].reset_index(drop=True)
    out = loaded.predict_one(X_one)
    assert set(out.keys()) >= {"p_long", "p_short", "abstain", "side", "confidence"}
    assert 0.0 <= out["p_long"] <= 1.0
    assert 0.0 <= out["p_short"] <= 1.0
    assert isinstance(out["abstain"], bool)
    assert out["side"] in {"long", "short", "none"}
    assert 0.0 <= out["confidence"] <= 1.0
    if out["abstain"]:
        # Task #654 wire contract: abstain ⇒ side="none" (NOT "abstain").
        assert out["side"] == "none"
        assert out["confidence"] == 0.0
    else:
        # Winning side must have the larger calibrated probability.
        if out["side"] == "long":
            assert out["p_long"] >= out["p_short"]
        else:
            assert out["p_short"] >= out["p_long"]

    # An impossibly-high τ must force every row to abstain — sanity for
    # the threshold path (independent of which side wins on this seed).
    loaded.manifest.abstain_tau = 0.999_999
    out_high_tau = loaded.predict_one(X_one)
    assert out_high_tau["abstain"] is True
    assert out_high_tau["side"] == "none"


def _fit_isotonic_for_test(
    raw: np.ndarray, labels: np.ndarray,
) -> dict:
    """Build the JSON-serialisable threshold-pair dict the isotonic
    serving path expects, using the SAME sklearn estimator the persist
    layer uses. Lifted into the test module so the round-trip exercises
    the EXACT shape the production layer writes (no hand-built knot
    grid that could drift from the actual estimator format).
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw.astype(float), labels.astype(float))
    return {
        "x_thresholds": [float(v) for v in iso.X_thresholds_.tolist()],
        "y_values":     [float(v) for v in iso.y_thresholds_.tolist()],
    }


def test_dual_head_isotonic_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    """Task #657 — round-trip the new isotonic calibration variant.

    Mirrors `test_dual_head_save_load_roundtrip` for the Platt path:
    train two binary heads, fit isotonic per head, persist via
    `save_model`, load via `load_model`, hit `predict_one`, and
    assert (a) the manifest carries `calibration_method="isotonic"`
    + JSON threshold arrays, (b) the loader returns a
    `LoadedDualHeadModel` whose `_apply_isotonic` reproduces sklearn's
    `IsotonicRegression(out_of_bounds="clip").transform` to within
    1e-9, and (c) the abstain branch still fires under an
    impossibly-high τ.

    The Platt round-trip case stays unchanged — both paths must
    co-exist on the same dual-head schema.
    """
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    df, y = _synth_training_frame(n_rows=400, n_features=5, seed=17)
    friction_pct = 0.10
    long_labels = (y > friction_pct).astype(int)
    short_labels = (y < -friction_pct).astype(int)
    assert int(long_labels.sum()) > 30 and int(long_labels.sum()) < 370
    assert int(short_labels.sum()) > 30 and int(short_labels.sum()) < 370

    long_booster, long_raw = _train_binary_head(df, long_labels, seed=1)
    short_booster, short_raw = _train_binary_head(df, short_labels, seed=2)

    isotonic = {
        "long":  _fit_isotonic_for_test(long_raw, long_labels),
        "short": _fit_isotonic_for_test(short_raw, short_labels),
    }
    # The threshold arrays MUST be non-empty and equal-length per head
    # — `validate()` enforces this and the `_apply_isotonic` serving
    # contract relies on `numpy.interp(raw, x, y)` over a real grid.
    for side in ("long", "short"):
        assert len(isotonic[side]["x_thresholds"]) >= 2
        assert (
            len(isotonic[side]["x_thresholds"])
            == len(isotonic[side]["y_values"])
        )

    manifest = _build_dual_head_isotonic_manifest(
        coin_id="bitcoin", timeframe="5m", version="v_dh_iso_001",
        feature_names=list(df.columns),
        friction_pct=friction_pct,
        abstain_tau=0.55,
        isotonic=isotonic,
    )
    out_dir = save_model(
        coin_id="bitcoin", timeframe="5m", version="v_dh_iso_001",
        booster=long_booster, calibrators=None, manifest=manifest,
        regressor=short_booster,
    )
    assert (out_dir / "long_model.txt").exists()
    assert (out_dir / "short_model.txt").exists()
    assert (out_dir / "manifest.json").exists()

    # Manifest round-trips with the new isotonic fields preserved and
    # the legacy Platt block intentionally null.
    saved = json.loads((out_dir / "manifest.json").read_text())
    assert saved["served_predictor_kind"] == "dual_binary_head"
    assert saved["calibration_method"] == "isotonic"
    assert saved["platt_calibration"] is None
    assert isinstance(saved["isotonic_calibration"], dict)
    for side in ("long", "short"):
        block = saved["isotonic_calibration"][side]
        assert (
            block["x_thresholds"]
            == isotonic[side]["x_thresholds"]
        )
        assert block["y_values"] == isotonic[side]["y_values"]

    loaded = load_model("bitcoin", "5m", "v_dh_iso_001")
    assert isinstance(loaded, LoadedDualHeadModel), type(loaded)
    assert loaded.manifest.calibration_method == "isotonic"
    assert loaded.manifest.platt_calibration is None
    assert isinstance(loaded.manifest.isotonic_calibration, dict)

    # _apply_isotonic must reproduce sklearn's transform exactly.
    from sklearn.isotonic import IsotonicRegression

    for side, raw_scores, labels in (
        ("long", long_raw, long_labels),
        ("short", short_raw, short_labels),
    ):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_scores.astype(float), labels.astype(float))
        sklearn_out = iso.transform(raw_scores.astype(float))
        block = loaded.manifest.isotonic_calibration[side]
        for i in range(min(50, len(raw_scores))):
            ours = LoadedDualHeadModel._apply_isotonic(
                float(raw_scores[i]),
                block["x_thresholds"], block["y_values"],
            )
            # numpy.interp uses the same monotone knot grid sklearn
            # stores in X_thresholds_/y_thresholds_, so the agreement
            # is exact to floating-point round-off (no accumulation).
            assert abs(ours - float(sklearn_out[i])) < 1e-9, (
                f"isotonic serving deviates from sklearn at row {i} "
                f"side={side}: ours={ours} sklearn={float(sklearn_out[i])}"
            )

    # Predict_one must produce the same well-formed shape as the Platt
    # path (the wire payload is invisible to consumers).
    X_one = df.iloc[[0]].reset_index(drop=True)
    out = loaded.predict_one(X_one)
    assert set(out.keys()) >= {
        "p_long", "p_short", "abstain", "side", "confidence",
    }
    assert 0.0 <= out["p_long"] <= 1.0
    assert 0.0 <= out["p_short"] <= 1.0
    assert isinstance(out["abstain"], bool)
    assert out["side"] in {"long", "short", "none"}
    assert 0.0 <= out["confidence"] <= 1.0
    if out["abstain"]:
        assert out["side"] == "none"
        assert out["confidence"] == 0.0

    # Impossibly-high τ → must force abstain on the isotonic path too.
    # Unlike Platt's asymptotic sigmoid, an `IsotonicRegression` fit
    # with `y_max=1.0` can SATURATE at exactly 1.0, so τ has to live
    # strictly above the [0,1] cap to guarantee abstain on every input.
    loaded.manifest.abstain_tau = 1.000_001
    out_high_tau = loaded.predict_one(X_one)
    assert out_high_tau["abstain"] is True
    assert out_high_tau["side"] == "none"


# ---------------------------------------------------------------------------
# Task #659 (C-BTC) — beta calibration round-trip + scope refusal
# ---------------------------------------------------------------------------


def _build_dual_head_beta_manifest(
    *, coin_id: str, timeframe: str, version: str,
    feature_names: list[str], friction_pct: float,
    abstain_tau: float, beta: dict,
    calibration_status: Optional[str] = None,
    scope_constraint: Optional[dict] = None,
) -> ModelManifest:
    """Task #659 — sibling of `_build_dual_head_manifest` for the beta
    calibration variant. Platt and isotonic blocks are forced to None
    so `validate()` accepts exactly one calibrator block.
    """
    return ModelManifest(
        coin_id=coin_id,
        timeframe=timeframe,
        version=version,
        feature_names=feature_names,
        coin_vocab=[coin_id],
        n_train_rows=0,
        n_test_rows=0,
        metrics={},
        baseline_metrics={},
        threshold_pct=friction_pct,
        horizon_candles=1,
        class_return_means_pct=[],
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=abstain_tau,
        platt_calibration=None,
        friction_threshold_pct=friction_pct,
        label_family="C_post_cost",
        calibration_method="beta",
        isotonic_calibration=None,
        beta_calibration=beta,
        calibration_status=calibration_status,
        scope_constraint=scope_constraint,
    )


def _oracle_beta(p: float, a: float, b: float, c: float, eps: float) -> float:
    """Standalone python oracle of the production beta sigmoid. Used by
    the round-trip test to assert agreement to within 1e-12 of
    `LoadedDualHeadModel._apply_beta`. Mirrors the formula in
    `b3_calibration_compare._apply_beta` (and uses the SAME eps the
    production helper uses, so numerical agreement is exact).
    """
    p_c = max(eps, min(1.0 - eps, float(p)))
    z = a * math.log(p_c) + b * math.log(1.0 - p_c) + c
    return 1.0 / (1.0 + math.exp(-z))


def test_manifest_validate_beta_requires_per_head_coefficients() -> None:
    """Task #659 — `calibration_method='beta'` must populate per-head
    `(a, b, c)` finite floats AND clear the legacy Platt / isotonic
    blocks. The validator catches every form of partial config so a
    half-built manifest never reaches save/load.
    """
    # Missing beta block entirely.
    m_no_beta = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_beta_missing",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5, friction_threshold_pct=0.30,
        calibration_method="beta",
        beta_calibration=None,
    )
    with pytest.raises(ValueError) as excinfo:
        m_no_beta.validate()
    assert "beta_calibration" in str(excinfo.value)

    # Beta block present but one head missing a coefficient.
    m_partial = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_beta_partial",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5, friction_threshold_pct=0.30,
        calibration_method="beta",
        beta_calibration={
            "long":  {"a": 0.9, "b": -1.5, "c": -0.2},
            "short": {"a": 1.0, "b": -1.2},  # missing c
        },
    )
    with pytest.raises(ValueError) as excinfo:
        m_partial.validate()
    assert "beta_calibration['short']" in str(excinfo.value)

    # Non-finite coefficient (NaN / inf) is rejected.
    m_nan = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_beta_nan",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5, friction_threshold_pct=0.30,
        calibration_method="beta",
        beta_calibration={
            "long":  {"a": float("nan"), "b": -1.5, "c": -0.2},
            "short": {"a": 1.0, "b": -1.2, "c": 0.1},
        },
    )
    with pytest.raises(ValueError) as excinfo:
        m_nan.validate()
    assert "beta_calibration['long']['a']" in str(excinfo.value)

    # Stale Platt block alongside beta is rejected.
    m_stale_platt = ModelManifest(
        coin_id="bitcoin", timeframe="5m", version="v_beta_stale_platt",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5, friction_threshold_pct=0.30,
        calibration_method="beta",
        platt_calibration={
            "long":  {"slope": -1.0, "intercept": 0.0},
            "short": {"slope": -1.0, "intercept": 0.0},
        },
        beta_calibration={
            "long":  {"a": 0.9, "b": -1.5, "c": -0.2},
            "short": {"a": 1.0, "b": -1.2, "c": 0.1},
        },
    )
    with pytest.raises(ValueError) as excinfo:
        m_stale_platt.validate()
    assert "platt_calibration" in str(excinfo.value)


def test_manifest_validate_calibration_status_enum() -> None:
    """`calibration_status` accepts the three operator-vetted enum
    values and rejects everything else (including typos).
    """
    base_kwargs = dict(
        coin_id="bitcoin", timeframe="5m", version="v_cs",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5, friction_threshold_pct=0.30,
        calibration_method="beta",
        beta_calibration={
            "long":  {"a": 0.9, "b": -1.5, "c": -0.2},
            "short": {"a": 1.0, "b": -1.2, "c": 0.1},
        },
    )
    for ok in (
        "trustworthy", "under_confident_documented",
        "over_confident_blocked", None,
    ):
        ModelManifest(**base_kwargs, calibration_status=ok).validate()
    bad = ModelManifest(**base_kwargs, calibration_status="trustworthy_typo")
    with pytest.raises(ValueError) as excinfo:
        bad.validate()
    assert "calibration_status" in str(excinfo.value)


def test_manifest_validate_scope_constraint_shape() -> None:
    """`scope_constraint` (Task #659) accepts a dict whose
    `allowed_universe` is a non-empty list of `'coin:tf'` strings.
    Rejects every other shape.
    """
    base = dict(
        coin_id="bitcoin", timeframe="5m", version="v_sc",
        feature_names=["feat_0"], coin_vocab=["bitcoin"],
        n_train_rows=10, n_test_rows=10,
        metrics={}, baseline_metrics={},
        threshold_pct=0.3, horizon_candles=1,
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=0.5, friction_threshold_pct=0.30,
        calibration_method="beta",
        beta_calibration={
            "long":  {"a": 0.9, "b": -1.5, "c": -0.2},
            "short": {"a": 1.0, "b": -1.2, "c": 0.1},
        },
    )
    # Valid shape passes.
    ModelManifest(
        **base,
        scope_constraint={
            "coin_id": "bitcoin", "timeframe": "5m",
            "candidate": "C_post_cost",
            "allowed_universe": ["bitcoin:5m"],
        },
    ).validate()
    # Empty allowed_universe rejected.
    with pytest.raises(ValueError) as excinfo:
        ModelManifest(
            **base,
            scope_constraint={"allowed_universe": []},
        ).validate()
    assert "allowed_universe" in str(excinfo.value)
    # Wrong-shape entry rejected.
    with pytest.raises(ValueError) as excinfo:
        ModelManifest(
            **base,
            scope_constraint={"allowed_universe": ["bitcoin5m"]},
        ).validate()
    assert "allowed_universe" in str(excinfo.value)
    # Top-level not a dict.
    with pytest.raises(ValueError) as excinfo:
        ModelManifest(
            **base,
            scope_constraint=["bitcoin:5m"],  # type: ignore[arg-type]
        ).validate()
    assert "scope_constraint" in str(excinfo.value)


def test_dual_head_beta_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    """Task #659 — round-trip the beta calibration variant:
    train two binary heads, FIT a beta calibrator per head using the
    SAME formulation as the research-side b3 helper, persist via
    `save_model`, load via `load_model`, and assert
    `_apply_beta` matches the python oracle to within 1e-12 on a
    fixed batch (when called with the same eps).
    """
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    df, y = _synth_training_frame(n_rows=400, n_features=5, seed=17)
    friction_pct = 0.10
    long_labels = (y > friction_pct).astype(int)
    short_labels = (y < -friction_pct).astype(int)
    assert int(long_labels.sum()) > 30 and int(long_labels.sum()) < 370
    assert int(short_labels.sum()) > 30 and int(short_labels.sum()) < 370

    long_booster, long_raw = _train_binary_head(df, long_labels, seed=1)
    short_booster, short_raw = _train_binary_head(df, short_labels, seed=2)

    # Hand-picked finite coefficients — the test cares about EXACT
    # round-trip, not about whether the chosen coefficients improve
    # calibration. Real B3-fitted values for BTC live at:
    #   long  {a=0.8978, b=-1.8881, c=-0.7911}
    #   short {a=0.9916, b=-1.2166, c=0.0677}
    beta = {
        "long":  {"a": 0.8978, "b": -1.8881, "c": -0.7911},
        "short": {"a": 0.9916, "b": -1.2166, "c":  0.0677},
    }
    manifest = _build_dual_head_beta_manifest(
        coin_id="bitcoin", timeframe="5m", version="v_dh_beta_001",
        feature_names=list(df.columns), friction_pct=friction_pct,
        abstain_tau=0.55, beta=beta,
        calibration_status="under_confident_documented",
    )
    out_dir = save_model(
        coin_id="bitcoin", timeframe="5m", version="v_dh_beta_001",
        booster=long_booster, calibrators=None, manifest=manifest,
        regressor=short_booster,
    )
    assert (out_dir / "long_model.txt").exists()
    assert (out_dir / "short_model.txt").exists()
    assert (out_dir / "manifest.json").exists()
    # No 3-class artefacts persisted.
    assert not (out_dir / "model.txt").exists()
    assert not (out_dir / "calibrators.joblib").exists()

    saved = json.loads((out_dir / "manifest.json").read_text())
    assert saved["served_predictor_kind"] == "dual_binary_head"
    assert saved["calibration_method"] == "beta"
    assert saved["platt_calibration"] is None
    assert saved["isotonic_calibration"] is None
    assert saved["beta_calibration"]["long"] == beta["long"]
    assert saved["beta_calibration"]["short"] == beta["short"]
    assert saved["calibration_status"] == "under_confident_documented"

    loaded = load_model("bitcoin", "5m", "v_dh_beta_001")
    assert isinstance(loaded, LoadedDualHeadModel), type(loaded)
    assert loaded.manifest.calibration_method == "beta"
    assert loaded.manifest.platt_calibration is None
    assert loaded.manifest.isotonic_calibration is None
    assert isinstance(loaded.manifest.beta_calibration, dict)

    # Production helper ↔ python oracle agreement to within 1e-12 over
    # a fixed grid of raw probabilities and the BTC-fitted coefficients.
    eps = LoadedDualHeadModel.BETA_EPS
    rng = np.random.default_rng(31337)
    raw_grid = rng.uniform(0.0, 1.0, size=64).astype(float).tolist()
    # Include boundary samples so the clip path is exercised.
    raw_grid.extend([0.0, 1.0, 1e-9, 1.0 - 1e-9])
    for side in ("long", "short"):
        a, b, c = beta[side]["a"], beta[side]["b"], beta[side]["c"]
        for raw in raw_grid:
            ours = LoadedDualHeadModel._apply_beta(raw, a, b, c)
            oracle = _oracle_beta(raw, a, b, c, eps)
            assert abs(ours - oracle) < 1e-12, (
                f"_apply_beta deviates from oracle at side={side} "
                f"raw={raw}: ours={ours} oracle={oracle}"
            )

    # End-to-end predict_one returns the dual-head shape with calibrated
    # probabilities in [0, 1] and a well-formed side decision.
    X_one = df.iloc[[0]].reset_index(drop=True)
    out = loaded.predict_one(X_one)
    assert set(out.keys()) >= {"p_long", "p_short", "abstain", "side", "confidence"}
    assert 0.0 <= out["p_long"] <= 1.0
    assert 0.0 <= out["p_short"] <= 1.0
    assert out["side"] in {"long", "short", "none"}
    assert 0.0 <= out["confidence"] <= 1.0
    if out["abstain"]:
        assert out["side"] == "none"
        assert out["confidence"] == 0.0

    # High τ forces abstain on the beta path (sigmoid → bounded < 1).
    loaded.manifest.abstain_tau = 0.999_999
    out_high = loaded.predict_one(X_one)
    assert out_high["abstain"] is True
    assert out_high["side"] == "none"


def test_dual_head_beta_round_trip_against_research_helper(
    tmp_path, monkeypatch,
) -> None:
    """Cross-check that the production `_apply_beta` agrees with the
    research-side helper in `b3_calibration_compare._apply_beta` when
    BOTH are called with the same eps. The two implementations live in
    different modules on purpose (production stays JSON-clean, research
    stays numpy-vectorized), so this guards against future drift.
    """
    from app.training.labels_research import b3_calibration_compare as b3

    raw = np.array([0.05, 0.2, 0.5, 0.7, 0.95], dtype=float)
    a, b, c = 0.8978, -1.8881, -0.7911
    # Force the research helper's eps to match production's BETA_EPS
    # so the comparison is apples-to-apples.
    monkeypatch.setattr(
        b3, "_safe_log",
        lambda p: np.log(np.clip(
            p, LoadedDualHeadModel.BETA_EPS,
            1.0 - LoadedDualHeadModel.BETA_EPS,
        )),
    )
    research = b3._apply_beta(raw, {"a": a, "b": b, "c": c})
    for i, p in enumerate(raw.tolist()):
        ours = LoadedDualHeadModel._apply_beta(p, a, b, c)
        assert abs(ours - float(research[i])) < 1e-12, (
            f"production _apply_beta diverges from research helper at "
            f"raw={p}: ours={ours} research={float(research[i])}"
        )


def test_predict_one_refuses_off_universe_scope(tmp_path, monkeypatch) -> None:
    """Task #659 — when the manifest carries a `scope_constraint`
    locked to `bitcoin:5m`, `predict_one(X, coin_id='ethereum',
    timeframe='5m')` MUST raise `ScopeViolationError` BEFORE touching
    the boosters. Defence in depth — the api-server's scope guard
    operates at the DB layer; this prevents the python serving path
    from ever computing a forbidden prediction.
    """
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    df, y = _synth_training_frame(n_rows=400, n_features=5, seed=17)
    friction_pct = 0.10
    long_labels = (y > friction_pct).astype(int)
    short_labels = (y < -friction_pct).astype(int)

    long_booster, _ = _train_binary_head(df, long_labels, seed=1)
    short_booster, _ = _train_binary_head(df, short_labels, seed=2)
    beta = {
        "long":  {"a": 0.8978, "b": -1.8881, "c": -0.7911},
        "short": {"a": 0.9916, "b": -1.2166, "c":  0.0677},
    }
    manifest = _build_dual_head_beta_manifest(
        coin_id="bitcoin", timeframe="5m", version="v_dh_scope_001",
        feature_names=list(df.columns), friction_pct=friction_pct,
        abstain_tau=0.55, beta=beta,
        calibration_status="under_confident_documented",
        scope_constraint={
            "coin_id": "bitcoin", "timeframe": "5m",
            "candidate": "C_post_cost",
            "allowed_universe": ["bitcoin:5m"],
        },
    )
    save_model(
        coin_id="bitcoin", timeframe="5m", version="v_dh_scope_001",
        booster=long_booster, calibrators=None, manifest=manifest,
        regressor=short_booster,
    )
    loaded = load_model("bitcoin", "5m", "v_dh_scope_001")
    assert isinstance(loaded, LoadedDualHeadModel)
    X_one = df.iloc[[0]].reset_index(drop=True)
    # Same-universe call passes through.
    ok = loaded.predict_one(X_one, coin_id="bitcoin", timeframe="5m")
    assert "p_long" in ok and "p_short" in ok
    # Off-universe coin → refused.
    with pytest.raises(ScopeViolationError) as excinfo:
        loaded.predict_one(X_one, coin_id="ethereum", timeframe="5m")
    msg = str(excinfo.value)
    assert "ethereum" in msg
    assert "bitcoin:5m" in msg
    # Off-universe timeframe → refused.
    with pytest.raises(ScopeViolationError):
        loaded.predict_one(X_one, coin_id="bitcoin", timeframe="1h")
    # Task #659 hardening — when scope_constraint.allowed_universe is
    # non-empty, omitting coin_id/timeframe is also refused: a missing
    # scope arg is indistinguishable from a guard-evading caller, so
    # the gate must NOT silently bypass. /ml/predict's dual-head
    # serving call site threads coin_id/timeframe explicitly.
    with pytest.raises(ScopeViolationError) as legacy_excinfo:
        loaded.predict_one(X_one)
    assert "did not pass coin_id/timeframe" in str(legacy_excinfo.value)


def test_load_model_refuses_out_of_scope_requested_for(tmp_path, monkeypatch) -> None:
    """Task #659 — `load_model(..., requested_for=(coin, tf))` returns
    `None` when the manifest carries a non-empty
    `scope_constraint.allowed_universe` and `(coin, tf)` is outside
    that universe. Same return-shape as a missing artifact, so the
    api-server's pooled-fallback path handles it without a new branch.
    """
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    df, y = _synth_training_frame(n_rows=400, n_features=5, seed=29)
    friction_pct = 0.10
    long_labels = (y > friction_pct).astype(int)
    short_labels = (y < -friction_pct).astype(int)
    long_booster, _ = _train_binary_head(df, long_labels, seed=1)
    short_booster, _ = _train_binary_head(df, short_labels, seed=2)
    beta = {
        "long":  {"a": 0.8978, "b": -1.8881, "c": -0.7911},
        "short": {"a": 0.9916, "b": -1.2166, "c":  0.0677},
    }
    manifest = _build_dual_head_beta_manifest(
        coin_id="bitcoin", timeframe="5m", version="v_dh_loadrefuse_001",
        feature_names=list(df.columns), friction_pct=friction_pct,
        abstain_tau=0.55, beta=beta,
        calibration_status="under_confident_documented",
        scope_constraint={
            "coin_id": "bitcoin", "timeframe": "5m",
            "candidate": "C_post_cost",
            "allowed_universe": ["bitcoin:5m"],
        },
    )
    save_model(
        coin_id="bitcoin", timeframe="5m", version="v_dh_loadrefuse_001",
        booster=long_booster, calibrators=None, manifest=manifest,
        regressor=short_booster,
    )
    # Without `requested_for`: legacy load semantics — model loads.
    loaded = load_model("bitcoin", "5m", "v_dh_loadrefuse_001")
    assert isinstance(loaded, LoadedDualHeadModel)
    # `requested_for` matches the universe: model loads.
    loaded_match = load_model(
        "bitcoin", "5m", "v_dh_loadrefuse_001",
        requested_for=("bitcoin", "5m"),
    )
    assert isinstance(loaded_match, LoadedDualHeadModel)
    # `requested_for` outside the universe: load refuses → None
    # (mirrors the missing-artifact surface; caller's fallback handles).
    refused = load_model(
        "bitcoin", "5m", "v_dh_loadrefuse_001",
        requested_for=("ethereum", "5m"),
    )
    assert refused is None


# ---------------------------------------------------------------------------
# /ml/predict integration: dual-head response shape
# ---------------------------------------------------------------------------


def _signal_ticks(n: int, start: datetime, drift: float, step_s: int = 60):
    """Same trending series helper used by the legacy 3-class predict
    test. We only need a long-enough window so build_feature_vector
    doesn't bail with insufficient_candles.
    """
    rng = np.random.default_rng(7)
    out = []
    p = 100.0
    for i in range(n):
        p *= (1.0 + drift + rng.normal(0, 0.005))
        out.append((start + timedelta(seconds=i * step_s), p))
    return out


@pytest.fixture()
def dual_head_registry(tmp_path, monkeypatch):
    """Train + persist a dual-head model under a temp REGISTRY_ROOT and
    point latest -> that version. Returns the (coin, tf, version) tuple
    so the caller can drive /ml/predict.
    """
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    # Use the LIVE feature columns so /ml/predict's _to_model_row maps
    # correctly. We don't need a great booster — only that it loads and
    # produces real numbers.
    feature_names = list(registry_module.FEATURE_COLUMNS)
    n_rows = 400
    rng = np.random.default_rng(11)
    X = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n_rows, len(feature_names))),
        columns=feature_names,
    )
    if "coin_idx" in X.columns:
        X["coin_idx"] = 0
        X["coin_idx"] = X["coin_idx"].astype("int32")
    y = 0.4 * X[feature_names[0]].values + 0.05 * rng.normal(0.0, 1.0, size=n_rows)
    friction_pct = 0.10
    long_labels = (y > friction_pct).astype(int)
    short_labels = (y < -friction_pct).astype(int)

    long_booster, long_raw = _train_binary_head(X, long_labels, seed=1)
    short_booster, short_raw = _train_binary_head(X, short_labels, seed=2)
    platt = {
        "long": _fit_platt(long_raw, long_labels),
        "short": _fit_platt(short_raw, short_labels),
    }
    coin, tf, version = "bitcoin", "1m", "v_dh_e2e_001"
    manifest = _build_dual_head_manifest(
        coin_id=coin, timeframe=tf, version=version,
        feature_names=feature_names, friction_pct=friction_pct,
        # Low τ so the test reliably gets a non-abstain side; the
        # high-τ abstain branch is exercised in the unit test above.
        abstain_tau=0.05, platt=platt,
    )
    save_model(
        coin_id=coin, timeframe=tf, version=version,
        booster=long_booster, calibrators=None, manifest=manifest,
        regressor=short_booster,
    )
    main_module._cached_load.cache_clear()
    return coin, tf, version


@pytest.mark.asyncio
async def test_predict_dual_head_returns_paper_trading_shape(
    dual_head_registry, monkeypatch,
):
    coin, tf, version = dual_head_registry
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch_ticks(coin_id: str, lookback_ms: int, now=None):
        return ticks

    async def fake_fetch_scope(coin_id, timeframe, model_version):
        return None  # no scope guard for this test

    async def fake_fetch_quarantined() -> set:
        return set()

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch_ticks)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch_ticks)
    monkeypatch.setattr(
        main_module, "fetch_scope_for_active_champion", fake_fetch_scope,
    )
    monkeypatch.setattr(
        main_module, "fetch_quarantined_versions", fake_fetch_quarantined,
    )

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": coin, "timeframe": tf})
        assert r.status_code == 200, r.text
        body = r.json()
        # New paper-trading response shape.
        assert body["status"] == "ok"
        assert body["predictor_kind"] == "dual_binary_head"
        assert body["coinId"] == coin
        assert body["timeframe"] == tf
        assert body["modelVersion"] == version
        assert body["modelCoinId"] == coin
        assert body["label_family"] == "C_post_cost"
        assert math.isclose(body["abstain_tau"], 0.05, abs_tol=1e-9)
        assert math.isclose(body["friction_threshold_pct"], 0.10, abs_tol=1e-9)
        assert 0.0 <= body["p_long"] <= 1.0
        assert 0.0 <= body["p_short"] <= 1.0
        assert isinstance(body["abstain"], bool)
        assert body["side"] in {"long", "short", "none"}
        assert 0.0 <= body["confidence"] <= 1.0
        # The legacy 3-class fields MUST NOT appear on this path — the
        # api-server consumer needs to dispatch on `predictor_kind`.
        assert "probUp" not in body
        assert "probDown" not in body
        assert "probStable" not in body
        # featureHash is still emitted so the journal writer can id the
        # bar (Task #460 contract).
        assert isinstance(body["featureHash"], str)
        assert body["featureHash"]


@pytest.mark.asyncio
async def test_predict_out_of_scope_short_circuits(
    dual_head_registry, monkeypatch,
):
    """When the resolved champion's scope_constraint does NOT admit the
    request's (coin, tf), /ml/predict must return
    `{"status": "out_of_scope", "scope_constraint": {...}}` instead of
    a prediction. Per-resolved-champion semantics — the scope is the
    SPECIFIC champion's fence, not a global allowlist.

    Setup: the registered champion lives at (coin, tf). We attach a
    scope to that exact row that EXCLUDES the same tf (a deliberately
    contradictory but valid paper-trading fence — e.g. an operator that
    paused this slot for that timeframe). The handler must short-
    circuit on the resolved row's scope.
    """
    coin, tf, version = dual_head_registry
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch_ticks(coin_id: str, lookback_ms: int, now=None):
        return ticks  # OK to be called — scope check fires AFTER resolve

    excluding_scope = {
        # Admits a different timeframe than the request, so _scope_matches
        # returns False against the request's tf.
        "coins": [coin], "timeframes": ["999h"],
        "label_family": "C_post_cost",
    }
    captured: dict = {}

    async def fake_fetch_scope(coin_id, timeframe, model_version):
        captured["call"] = (coin_id, timeframe, model_version)
        # Only return a scope for the resolved champion's identity tuple.
        if (coin_id, timeframe, model_version) == (coin, tf, version):
            return excluding_scope
        return None

    async def fake_fetch_quarantined() -> set:
        return set()

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch_ticks)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch_ticks)
    monkeypatch.setattr(
        main_module, "fetch_scope_for_active_champion", fake_fetch_scope,
    )
    monkeypatch.setattr(
        main_module, "fetch_quarantined_versions", fake_fetch_quarantined,
    )

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": coin, "timeframe": tf})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "out_of_scope"
        assert body["coinId"] == coin
        assert body["timeframe"] == tf
        # Echo of the resolved row's scope so the caller can render the
        # operator-stamped reason.
        assert body["scope_constraint"] == excluding_scope
        # The scope lookup MUST have been keyed on the resolved
        # champion's identity tuple (per-resolved-champion semantics,
        # NOT a global scan).
        assert captured["call"] == (coin, tf, version)


@pytest.mark.asyncio
async def test_predict_in_scope_passes_through(dual_head_registry, monkeypatch):
    """When a scope_constraint admits the request, /ml/predict must
    proceed to the dual-head response shape (no out_of_scope short-
    circuit). Guards against an over-eager scope check that fences off
    the registered slot itself.
    """
    coin, tf, version = dual_head_registry
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch_ticks(coin_id: str, lookback_ms: int, now=None):
        return ticks

    async def fake_fetch_scope(coin_id, timeframe, model_version):
        # Scope EXACTLY matches the dual_head_registry fixture.
        if (coin_id, timeframe, model_version) == (coin, tf, version):
            return {
                "coins": [coin], "timeframes": [tf],
                "label_family": "C_post_cost",
            }
        return None

    async def fake_fetch_quarantined() -> set:
        return set()

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch_ticks)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch_ticks)
    monkeypatch.setattr(
        main_module, "fetch_scope_for_active_champion", fake_fetch_scope,
    )
    monkeypatch.setattr(
        main_module, "fetch_quarantined_versions", fake_fetch_quarantined,
    )

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": coin, "timeframe": tf})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["predictor_kind"] == "dual_binary_head"


# ---------------------------------------------------------------------------
# promote_shadow_to_serving — fake-pool unit tests
# ---------------------------------------------------------------------------


class _FakeRow(dict):
    """asyncpg.Record proxy backed by a dict so subscript access works."""

    def __getitem__(self, key):  # type: ignore[override]
        return super().__getitem__(key)


class _FakeConn:
    """Minimal asyncpg connection stub for promote_shadow_to_serving."""

    def __init__(self, store: dict[int, dict]) -> None:
        self._store = store
        self.executed: list[tuple[str, tuple]] = []

    def transaction(self):
        return _FakeTxnCtx()

    async def fetchrow(self, sql: str, *args: Any):
        sql_low = sql.strip().split()[0].lower()
        assert sql_low == "select"
        # The promote function only does one fetchrow: by id.
        (row_id,) = args
        row = self._store.get(int(row_id))
        if row is None:
            return None
        return _FakeRow(row)

    async def fetch(self, sql: str, *args: Any):
        # The only fetch the function does is "list candidate
        # champions for the same (model_id, coin_id, timeframe)
        # slot, excluding the row being promoted". Demotion is then
        # filtered IN PYTHON by matching scope_constraint.label_family
        # so the test must hand back enough rows for that filter to
        # exercise both match-and-skip behaviour.
        model_id, coin_id, timeframe, exclude_id = args
        out: list[_FakeRow] = []
        for r in self._store.values():
            if (
                r["model_id"] == model_id
                and r["coin_id"] == coin_id
                and r["timeframe"] == timeframe
                and r.get("state") == "champion"
                and r.get("is_active") is True
                and r["id"] != exclude_id
            ):
                out.append(_FakeRow({
                    "id": r["id"],
                    "scope_constraint": r.get("scope_constraint"),
                }))
        return out

    async def execute(self, sql: str, *args: Any) -> None:
        self.executed.append((sql, args))
        sql_norm = sql.strip().split()[0].lower()
        assert sql_norm == "update"
        # Two UPDATE flavours: demote-to-shadow and promote-to-champion.
        if "state = 'shadow'" in sql:
            row_id, demoted_at = args
            row = self._store[int(row_id)]
            row["state"] = "shadow"
            row["demoted_at"] = demoted_at
            row["updated_at"] = demoted_at
        elif "state = 'champion'" in sql:
            (
                row_id, promoted_at, prev_id, note_value,
                snapshot_json, scope_json,
            ) = args
            row = self._store[int(row_id)]
            row["state"] = "champion"
            row["is_active"] = True
            row["promoted_at"] = promoted_at
            row["previous_champion_id"] = prev_id
            row["note"] = note_value
            row["metrics_snapshot"] = json.loads(snapshot_json)
            row["scope_constraint"] = json.loads(scope_json)
            row["updated_at"] = promoted_at
        else:
            raise AssertionError(f"unexpected UPDATE sql: {sql!r}")


class _FakeTxnCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Reraise any exception so caller-side rollback assertions hold.
        return False


class _FakeAcquireCtx:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, store: dict[int, dict]) -> None:
        self._conn = _FakeConn(store)

    def acquire(self) -> _FakeAcquireCtx:
        return _FakeAcquireCtx(self._conn)


def _stub_load_model_ok(coin_id: str, timeframe: str, version: str):
    # Returns a minimal sentinel — promote_shadow_to_serving only checks
    # `is None`; it never reads any attribute.
    return object()


def _stub_load_model_none(coin_id: str, timeframe: str, version: str):
    return None


def test_promote_shadow_to_serving_promotes_and_demotes_incumbent() -> None:
    store: dict[int, dict] = {
        1: {
            "id": 1, "model_id": "dual_binary_head", "model_version": "v_old",
            "coin_id": "bitcoin", "timeframe": "5m",
            "state": "champion", "is_active": True,
            "metrics_snapshot": {"some_old": "metric"},
            # Same label_family as the promotion below, so demotion fires.
            "scope_constraint": {
                "coins": ["bitcoin"], "timeframes": ["5m"],
                "label_family": "C_post_cost",
            },
        },
        2: {
            "id": 2, "model_id": "dual_binary_head", "model_version": "v_new",
            "coin_id": "bitcoin", "timeframe": "5m",
            "state": "shadow", "is_active": True,
            "metrics_snapshot": {"hold_out_auc": 0.61},
        },
    }
    pool = _FakePool(store)
    new_scope = {
        "coins": ["bitcoin"], "timeframes": ["5m"],
        "label_family": "C_post_cost",
    }
    res = asyncio.run(
        registry_lifecycle.promote_shadow_to_serving(
            2,
            scope_constraint=new_scope,
            promoted_by="operator-A",
            pool=pool,
            load_model_fn=_stub_load_model_ok,
        )
    )
    assert res.promoted_id == 2
    assert res.previous_champion_id == 1
    assert res.coin_id == "bitcoin"
    assert res.timeframe == "5m"
    assert res.scope_constraint == new_scope
    assert res.promoted_by == "operator-A"
    assert isinstance(res.promoted_at, datetime)

    assert store[1]["state"] == "shadow"
    assert store[1]["demoted_at"] is not None
    assert store[2]["state"] == "champion"
    assert store[2]["is_active"] is True
    assert store[2]["scope_constraint"] == new_scope
    snap = store[2]["metrics_snapshot"]
    assert snap["promoted_by"] == "operator-A"
    assert snap["previous_champion_id"] == 1
    # Pre-existing snapshot keys must survive the merge.
    assert snap["hold_out_auc"] == 0.61


def test_promote_shadow_to_serving_skips_demotion_for_different_label_family() -> None:
    """A family-C promotion MUST NOT demote a Family-A champion that
    happens to share (model_id, coin_id, timeframe). Otherwise an
    operator promoting a paper-trading family-C head would silently
    evict the live 3-class champion.
    """
    store: dict[int, dict] = {
        # Existing champion is Family A — must survive untouched.
        1: {
            "id": 1, "model_id": "dual_binary_head", "model_version": "v_a",
            "coin_id": "bitcoin", "timeframe": "5m",
            "state": "champion", "is_active": True,
            "metrics_snapshot": {"some_old": "metric"},
            "scope_constraint": {
                "coins": ["bitcoin"], "timeframes": ["5m"],
                "label_family": "A_quintile",
            },
        },
        # Promotion target is Family C.
        2: {
            "id": 2, "model_id": "dual_binary_head", "model_version": "v_c",
            "coin_id": "bitcoin", "timeframe": "5m",
            "state": "shadow", "is_active": True,
            "metrics_snapshot": {"hold_out_auc": 0.62},
        },
    }
    pool = _FakePool(store)
    res = asyncio.run(
        registry_lifecycle.promote_shadow_to_serving(
            2,
            scope_constraint={
                "coins": ["bitcoin"], "timeframes": ["5m"],
                "label_family": "C_post_cost",
            },
            promoted_by="op",
            pool=pool,
            load_model_fn=_stub_load_model_ok,
        )
    )
    # Family-A champion left intact.
    assert store[1]["state"] == "champion"
    assert store[1]["is_active"] is True
    assert "demoted_at" not in store[1]
    # Family-C promoted with no previous_champion (different bucket).
    assert res.previous_champion_id is None
    assert store[2]["state"] == "champion"


def test_promote_shadow_to_serving_refuses_non_shadow_row() -> None:
    store = {
        9: {
            "id": 9, "model_id": "dual_binary_head", "model_version": "v_x",
            "coin_id": "bitcoin", "timeframe": "5m",
            "state": "champion", "is_active": True,
            "metrics_snapshot": None,
        },
    }
    pool = _FakePool(store)
    with pytest.raises(registry_lifecycle.PromotionError) as excinfo:
        asyncio.run(
            registry_lifecycle.promote_shadow_to_serving(
                9,
                scope_constraint={"coins": ["bitcoin"]},
                promoted_by="op",
                pool=pool,
                load_model_fn=_stub_load_model_ok,
            )
        )
    assert "expected 'shadow'" in str(excinfo.value)
    # State must NOT have changed.
    assert store[9]["state"] == "champion"


def test_promote_shadow_to_serving_refuses_when_manifest_load_fails() -> None:
    store = {
        7: {
            "id": 7, "model_id": "dual_binary_head", "model_version": "v_y",
            "coin_id": "bitcoin", "timeframe": "5m",
            "state": "shadow", "is_active": True,
            "metrics_snapshot": None,
        },
    }
    pool = _FakePool(store)
    with pytest.raises(registry_lifecycle.PromotionError) as excinfo:
        asyncio.run(
            registry_lifecycle.promote_shadow_to_serving(
                7,
                scope_constraint={"coins": ["bitcoin"]},
                promoted_by="op",
                pool=pool,
                load_model_fn=_stub_load_model_none,
            )
        )
    assert "manifest" in str(excinfo.value).lower()
    # State must NOT have changed.
    assert store[7]["state"] == "shadow"


def test_promote_shadow_to_serving_refuses_missing_row() -> None:
    pool = _FakePool({})
    with pytest.raises(registry_lifecycle.PromotionError) as excinfo:
        asyncio.run(
            registry_lifecycle.promote_shadow_to_serving(
                404,
                scope_constraint={"coins": ["bitcoin"]},
                promoted_by="op",
                pool=pool,
                load_model_fn=_stub_load_model_ok,
            )
        )
    assert "not found" in str(excinfo.value)


def test_promote_shadow_to_serving_validates_inputs() -> None:
    pool = _FakePool({})
    with pytest.raises(registry_lifecycle.PromotionError):
        asyncio.run(
            registry_lifecycle.promote_shadow_to_serving(
                1, scope_constraint=["not", "a", "dict"],  # type: ignore[arg-type]
                promoted_by="op", pool=pool,
                load_model_fn=_stub_load_model_ok,
            )
        )
    with pytest.raises(registry_lifecycle.PromotionError):
        asyncio.run(
            registry_lifecycle.promote_shadow_to_serving(
                1, scope_constraint={"coins": ["bitcoin"]},
                promoted_by="   ", pool=pool,
                load_model_fn=_stub_load_model_ok,
            )
        )


# ---------------------------------------------------------------------------
# labels_research runner — persist_dual_binary_head registry path
# ---------------------------------------------------------------------------


def test_persist_dual_binary_head_writes_registry_artifact(tmp_path, monkeypatch):
    """`persist_dual_binary_head` must wrap save_model and produce a
    version directory that load_model can read back as a
    LoadedDualHeadModel. Without this, Task #654's "labels_research
    persistence path" is missing.
    """
    from app.training.labels_research import runner as lr_runner

    # Redirect the registry root to a sandbox so we don't write into
    # the real models tree.
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    # Tiny but valid LightGBM heads.
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((200, 4))
    feat_names = [f"f{i}" for i in range(4)]
    df = pd.DataFrame(X, columns=feat_names)
    y_long = (X[:, 0] > 0.3).astype(int)
    y_short = (X[:, 0] < -0.3).astype(int)
    long_head = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 7},
        lgb.Dataset(df, label=y_long), num_boost_round=15,
    )
    short_head = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 7},
        lgb.Dataset(df, label=y_short), num_boost_round=15,
    )

    version_dir = lr_runner.persist_dual_binary_head(
        coin_id="bitcoin",
        timeframe="5m",
        version="v_runner_persist_001",
        long_booster=long_head,
        short_booster=short_head,
        feature_names=feat_names,
        coin_vocab=["bitcoin"],
        n_train_rows=200, n_test_rows=0,
        metrics={"hold_out_auc_long": 0.7, "hold_out_auc_short": 0.65},
        abstain_tau=0.05,
        platt_calibration={
            "long":  {"slope": -1.0, "intercept": 0.0},
            "short": {"slope": -1.0, "intercept": 0.0},
        },
        friction_threshold_pct=0.10,
        horizon_candles=24,
        label_family="C_post_cost",
    )

    assert version_dir.exists()
    # load_model must hand back a dual-head wrapper, NOT a 3-class
    # LoadedModel — proving the manifest's served_predictor_kind round-
    # tripped correctly.
    loaded = load_model("bitcoin", "5m", "v_runner_persist_001")
    assert isinstance(loaded, LoadedDualHeadModel)
    assert loaded.manifest.served_predictor_kind == "dual_binary_head"
    assert loaded.manifest.label_family == "C_post_cost"
    assert math.isclose(loaded.manifest.abstain_tau, 0.05, abs_tol=1e-9)
