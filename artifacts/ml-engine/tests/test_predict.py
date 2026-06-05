"""End-to-end test of the /ml/predict path with a real (tiny) trained model.

We synthesise a tiny labeled dataset, train one timeframe (per-coin + pooled
fallback paths), persist it to a temporary registry, then call /ml/predict
via TestClient and verify it returns a model-backed answer (not the
Phase-1 stub).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import main as main_module
from app.main import app
from app.training import registry as registry_module
from app.training.labels import build_labeled_frame_for_coin
from app.training.train import train_timeframe


def _signal_ticks(n: int, start: datetime, drift: float, step_s: int = 60):
    """Trending series with high-volatility noise so labels span all 3
    classes (DOWN/STABLE/UP) — needed for the per-class isotonic
    calibration to fit and the reliability diagram to populate.
    """
    rng = np.random.default_rng(7)
    out = []
    p = 100.0
    for i in range(n):
        p *= (1.0 + drift + rng.normal(0, 0.005))
        out.append((start + timedelta(seconds=i * step_s), p))
    return out


def _build_dataset(coins_drift: list[tuple[str, float, int]]) -> pd.DataFrame:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    frames = []
    for coin, drift, n in coins_drift:
        frames.append(build_labeled_frame_for_coin(coin, "1m", _signal_ticks(n, base, drift)))
    return pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)


@pytest.fixture()
def trained_registry(tmp_path, monkeypatch):
    """Train two per-coin models + (no pooled fallback needed) into a temp registry."""
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    df = _build_dataset([("pepe", 0.0010, 200), ("bonk", -0.0005, 200)])
    res = train_timeframe(df, "1m", coin_ids=["pepe", "bonk"])
    assert res["status"] == "trained", res
    main_module._cached_load.cache_clear()
    return res


@pytest.fixture()
def trained_registry_pooled_only(tmp_path, monkeypatch):
    """Train a pooled-only model (no per-coin slice has enough rows)."""
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    # Each coin alone is below MIN_PER_COIN_ROWS=80 — forces pooled fallback.
    df = _build_dataset([("pepe", 0.0010, 75), ("bonk", -0.0005, 75), ("doge", 0.0002, 75)])
    res = train_timeframe(df, "1m", coin_ids=["pepe", "bonk", "doge"])
    assert res["status"] == "trained", res
    assert res["pooled"] is not None and res["pooled"]["status"] == "trained"
    assert all(slc["status"] != "trained" for slc in res["per_coin"].values())
    main_module._cached_load.cache_clear()
    return res


@pytest.mark.asyncio
async def test_predict_uses_per_coin_model_when_available(trained_registry, monkeypatch):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch(coin_id: str, lookback_ms: int, now=None):
        return ticks

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch)

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": "pepe", "timeframe": "1m"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "lightgbm"
        assert body["modelCoinId"] == "pepe", "should resolve per-coin model first"
        # Probabilities form a valid distribution.
        assert 0.0 <= body["probUp"] <= 1.0
        assert 0.0 <= body["probDown"] <= 1.0
        assert 0.0 <= body["probStable"] <= 1.0
        assert math.isclose(
            body["probUp"] + body["probDown"] + body["probStable"], 1.0, abs_tol=1e-6,
        )
        assert 0.0 <= body["confidence"] <= 1.0
        # expectedReturnPct must be a real model-derived figure: weighted sum
        # of per-class mean returns. predictionStdPct must be non-negative.
        assert isinstance(body["expectedReturnPct"], float)
        assert body["predictionStdPct"] >= 0.0
        assert isinstance(body["featureImportanceTop5"], list)
        assert len(body["featureImportanceTop5"]) <= 5
        assert all("feature" in f and "gainPct" in f for f in body["featureImportanceTop5"])
        # Task #460 — every QUANT prediction must carry a feature_hash so
        # the api-server's journal-writer accepts it. The lightgbm path
        # computes it from the same feature dict that fed the booster, so
        # live and journal stay in lock-step. Length 12 mirrors the
        # `feature_hash()` helper used by /ml/features.
        assert isinstance(body["featureHash"], str) and len(body["featureHash"]) == 12


@pytest.mark.asyncio
async def test_predict_returns_specialists_and_regime_when_specialists_on_disk(
    trained_registry, monkeypatch,
):
    """Phase 3 contract — once `train_timeframe` has run, the registry
    contains at least the volatility-forecaster specialist on disk, and
    `/ml/predict` MUST surface (a) a non-empty `specialists[]` block
    with one entry per taxonomy kind and (b) a non-null `regime`
    string. This is the regression test for the reviewer's "a future
    change can silently strip specialist outputs from production
    predictions" finding — without it, anyone reshuffling the registry
    map or the response builder could drop the entire specialist block
    and the legacy 3-class head would still pass every other predict
    test.
    """
    from app.training.registry import (
        SPECIALIST_KINDS, latest_version, specialist_coin_id,
    )

    # Sanity: the fixture must have actually persisted at least one
    # specialist slot on disk. Otherwise the assertions below are
    # vacuous and we'd silently regress.
    persisted_specialists = [
        kind for kind in SPECIALIST_KINDS
        if latest_version(specialist_coin_id(kind), "1m") is not None
    ]
    assert persisted_specialists, (
        "fixture didn't persist any specialist slot; the rest of this "
        "test would be vacuous. Specialist trainer or fixture has drifted."
    )

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch(coin_id: str, lookback_ms: int, now=None):
        return ticks

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch)

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": "pepe", "timeframe": "1m"})
        assert r.status_code == 200, r.text
        body = r.json()

        # (a) regime is the live 6-class label, not None.
        assert isinstance(body.get("regime"), str) and body["regime"], (
            f"expected non-empty regime string, got {body.get('regime')!r}"
        )

        # (b) specialists[] is one entry per taxonomy kind.
        specialists = body.get("specialists")
        assert isinstance(specialists, list)
        assert len(specialists) == len(SPECIALIST_KINDS)
        kinds = [s["kind"] for s in specialists]
        assert set(kinds) == set(SPECIALIST_KINDS), (
            f"specialists block kind drift: got {sorted(kinds)}, "
            f"expected {sorted(SPECIALIST_KINDS)}"
        )

        # Every entry carries the canonical regime_subset and an
        # `applicable` flag. This is the shape the journal + diagnostics
        # page reads — strip it and downstream rendering breaks
        # silently.
        for s in specialists:
            assert "regimeSubset" in s and isinstance(s["regimeSubset"], list)
            assert "applicable" in s and isinstance(s["applicable"], bool)
            assert s["modelCoinId"] == specialist_coin_id(s["kind"])

        # At least the specialists that have models on disk MUST emit a
        # real prediction (no `error`, real probabilities). Without this
        # check, /ml/predict could regress to "always returns
        # not_trained" and the test above would still pass.
        live_specialists = [
            s for s in specialists if s["kind"] in persisted_specialists
        ]
        assert live_specialists, "no specialists on disk to score"
        for s in live_specialists:
            assert s.get("error") is None, (
                f"specialist {s['kind']} on disk but predict returned "
                f"error={s.get('error')!r}"
            )
            assert s.get("modelVersion"), (
                f"specialist {s['kind']} on disk but modelVersion is empty"
            )
            assert 0.0 <= s["probUp"] <= 1.0
            assert 0.0 <= s["probDown"] <= 1.0
            assert 0.0 <= s["probStable"] <= 1.0
            assert math.isclose(
                s["probUp"] + s["probDown"] + s["probStable"], 1.0, abs_tol=1e-6,
            )


@pytest.mark.asyncio
async def test_predict_falls_back_to_pooled_when_per_coin_missing(
    trained_registry_pooled_only, monkeypatch,
):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch(coin_id: str, lookback_ms: int, now=None):
        return ticks

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch)

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": "pepe", "timeframe": "1m"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "lightgbm"
        assert body["modelCoinId"] == "__pooled__", "should fall back to pooled"
        # Task #418 — even the existing missing-per-coin fallback must
        # surface the attribution flag so the dashboard can render a
        # "Pooled fallback" badge for any pooled-served coin.
        assert body["fallback"] == "pooled"


@pytest.mark.asyncio
async def test_predict_falls_back_to_pooled_when_per_coin_fails_verification(
    trained_registry, monkeypatch,
):
    """Task #418 — third structural option from the task #401 brief.

    `trained_registry` produces both a per-coin model (e.g. `pepe/1m`)
    AND a pooled model (`__pooled__/1m`); `_resolve_for_predict`
    normally serves the per-coin slice. After we stamp the per-coin
    slice's verification verdict as `promoted=False` (mirroring the
    real-world per-coin 1d slot whose holdout DA fell into the noise
    band raised by the tighter task #401 floor) AND the pooled slice
    as `promoted=True`, /ml/predict MUST switch to the pooled head and
    surface `fallback="pooled"` on the response so the dashboard can
    attribute the prediction.
    """
    from app.training.registry import (
        POOLED_COIN_ID, latest_version, write_verification_verdict,
    )

    pepe_v = latest_version("pepe", "1m")
    pooled_v = latest_version(POOLED_COIN_ID, "1m")
    assert pepe_v is not None and pooled_v is not None, (
        "fixture must have produced both a per-coin and pooled slot"
    )

    # Per-coin slice failed verification — mirrors the [0.500, 0.530]
    # noise-band per-coin 1d slot the task brief targets.
    write_verification_verdict("pepe", "1m", pepe_v, {
        "promoted": False, "reason": "below_coinflip",
        "directional_accuracy": 0.512,
        "baseline_directional_accuracy": 0.498,
        "lift": 0.014, "holdout_rows": 250,
        "min_directional_accuracy_applied": 0.530,
        "coin": "pepe", "timeframe": "1m", "kind": "per_coin",
    })
    # Pooled slice promoted — the safety net the trader should now use.
    write_verification_verdict(POOLED_COIN_ID, "1m", pooled_v, {
        "promoted": True, "reason": "lift",
        "directional_accuracy": 0.612,
        "baseline_directional_accuracy": 0.498,
        "lift": 0.114, "holdout_rows": 500,
        "min_directional_accuracy_applied": 0.500,
        "coin": "__pooled__", "timeframe": "1m", "kind": "pooled",
    })
    # Cache may have warmed from previous tests — make sure the
    # post-stamp verdict is the one /ml/predict reads.
    main_module._cached_load.cache_clear()

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch(coin_id: str, lookback_ms: int, now=None):
        return ticks

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch)

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": "pepe", "timeframe": "1m"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "lightgbm"
        # Even though pepe's per-coin model exists and loads, the failed
        # verification verdict forces the pooled head to serve.
        assert body["modelCoinId"] == POOLED_COIN_ID, (
            f"expected pooled fallback, got modelCoinId={body['modelCoinId']!r}"
        )
        assert body["fallback"] == "pooled", (
            f"expected fallback=pooled attribution flag, got {body.get('fallback')!r}"
        )

    # Sanity: bonk's per-coin slice still has no verdict on file, so
    # the resolver's no-verdict branch must keep serving the per-coin
    # head (no silent fallback for slices that simply predate stamping).
    bonk_v = latest_version("bonk", "1m")
    assert bonk_v is not None
    with TestClient(app) as client:
        r2 = client.post("/ml/predict", json={"coinId": "bonk", "timeframe": "1m"})
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["modelCoinId"] == "bonk"
        assert body2.get("fallback") in (None, ""), (
            f"slice without a verdict file must NOT be silently switched "
            f"to pooled; got fallback={body2.get('fallback')!r}"
        )


@pytest.mark.asyncio
async def test_predict_keeps_per_coin_when_pooled_also_fails_verification(
    trained_registry, monkeypatch,
):
    """Task #418 — when the per-coin slice failed verification but the
    pooled slice ALSO failed, the resolver must keep serving the
    per-coin slice. Falling back to an unverified pool would gain us
    nothing and would silently swap which head decides; the per-coin
    slice's verdict is already recorded for operator visibility.
    """
    from app.training.registry import (
        POOLED_COIN_ID, latest_version, write_verification_verdict,
    )

    pepe_v = latest_version("pepe", "1m")
    pooled_v = latest_version(POOLED_COIN_ID, "1m")
    assert pepe_v is not None and pooled_v is not None

    write_verification_verdict("pepe", "1m", pepe_v, {
        "promoted": False, "reason": "below_coinflip",
        "directional_accuracy": 0.510,
        "baseline_directional_accuracy": 0.500,
        "lift": 0.010, "holdout_rows": 250,
        "min_directional_accuracy_applied": 0.530,
        "coin": "pepe", "timeframe": "1m", "kind": "per_coin",
    })
    write_verification_verdict(POOLED_COIN_ID, "1m", pooled_v, {
        "promoted": False, "reason": "no_lift",
        "directional_accuracy": 0.495,
        "baseline_directional_accuracy": 0.498,
        "lift": -0.003, "holdout_rows": 500,
        "min_directional_accuracy_applied": 0.500,
        "coin": "__pooled__", "timeframe": "1m", "kind": "pooled",
    })
    main_module._cached_load.cache_clear()

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch(coin_id: str, lookback_ms: int, now=None):
        return ticks

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch)

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": "pepe", "timeframe": "1m"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["modelCoinId"] == "pepe", (
            "pool also failed verification — must NOT swap which head "
            "decides; per-coin slice stays in service"
        )
        assert body.get("fallback") in (None, ""), (
            f"no fallback when both heads fail verification; got "
            f"fallback={body.get('fallback')!r}"
        )


def test_predict_503_when_no_model(monkeypatch, tmp_path):
    """If neither per-coin nor pooled model is registered, predict refuses."""
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    main_module._cached_load.cache_clear()
    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": "pepe", "timeframe": "6h"})
        assert r.status_code == 503
        assert "no model registered" in r.json()["detail"]


def test_admin_reload_disabled_without_token(monkeypatch):
    monkeypatch.delenv("ML_ADMIN_TOKEN", raising=False)
    with TestClient(app) as client:
        r = client.post("/ml/admin/reload")
        assert r.status_code == 404


def test_admin_reload_requires_correct_token(monkeypatch):
    monkeypatch.setenv("ML_ADMIN_TOKEN", "secret-xyz")
    with TestClient(app) as client:
        r1 = client.post("/ml/admin/reload")
        assert r1.status_code == 401
        r2 = client.post("/ml/admin/reload", headers={"X-Admin-Token": "wrong"})
        assert r2.status_code == 401
        r3 = client.post("/ml/admin/reload", headers={"X-Admin-Token": "secret-xyz"})
        assert r3.status_code == 200


def test_report_endpoint_renders_html(trained_registry):
    """Real per-class reliability diagram + insufficient_data rows for all
    advertised timeframes appear in the HTML."""
    from app.training import report as report_module
    fake_report = {
        "generated_at": "2025-01-01T00:00:00Z",
        "coin_ids": ["pepe", "bonk"], "lookback_days": 7,
        "timeframes": {
            "1m": trained_registry,
            "6h": {"status": "insufficient_data", "n_rows": 5, "min_rows_required": 80},
            "1d": {"status": "insufficient_data", "n_rows": 0, "min_rows_required": 80},
        },
    }
    html_out = report_module.render_html(fake_report)
    assert "<html" in html_out
    assert "1m" in html_out
    assert "6h" in html_out and "1d" in html_out
    assert html_out.count("Insufficient data") >= 2
    assert "Reliability" in html_out
    # Per-class diagram legend — must show all 3 classes.
    assert "DOWN" in html_out and "UP" in html_out and "STABLE" in html_out
    # Per-coin sections rendered.
    assert "pepe" in html_out and "bonk" in html_out
    # Task #146 — regression head's holdout stats appear on each trained
    # slice (the fixture trains real per-coin LightGBM models, which
    # include the magnitude head from task #135).
    assert "Regression head (magnitude)" in html_out
    assert "holdout p95 |pred|" in html_out


def test_report_flags_degenerate_regression_head():
    """Task #146 — when a slice's regression-head p95 falls below the
    timeframe's label threshold, the report must flag it. That's the
    exact regression bug task #135 fixed (magnitude predictions
    collapsing toward 0)."""
    from app.training import report as report_module

    fake_report = {
        "generated_at": "2025-01-01T00:00:00Z",
        "coin_ids": ["pepe"], "lookback_days": 7,
        "timeframes": {
            "1m": {
                "status": "trained",
                "n_rows": 200,
                "per_coin": {
                    "pepe": {
                        "status": "trained",
                        "version": "vtest",
                        "n_rows": 200,
                        "n_folds": 1,
                        "metrics": {"auc": 0.6, "log_loss": 1.0,
                                    "directional_accuracy": 0.5},
                        "baseline_metrics": {"auc": 0.5, "log_loss": 1.1,
                                             "directional_accuracy": 0.5},
                        "lift_auc": 0.1,
                        "fold_metrics": [],
                        "calibration_diagram": [],
                        "class_return_means_pct": [-0.4, 0.0, 0.4],
                        "threshold_pct": 0.30,
                        "has_regression_head": True,
                        "regression_head_stats": {
                            "n_train_rows": 100,
                            "n_holdout_rows": 25,
                            "abs_pred_p50_pct": 0.01,
                            "abs_pred_p95_pct": 0.02,  # below threshold
                            "abs_pred_max_pct": 0.05,
                            "mae_pct": 0.04,
                            "best_iteration": 3,
                        },
                    },
                },
                "pooled": None,
                "dataset_path": "x.parquet",
            },
        },
    }
    html_out = report_module.render_html(fake_report)
    assert "Regression head (magnitude)" in html_out
    assert "Magnitude head looks degenerate" in html_out
    # Healthy slice (p95 well above threshold) should not be flagged.
    fake_report["timeframes"]["1m"]["per_coin"]["pepe"]["regression_head_stats"][
        "abs_pred_p95_pct"
    ] = 1.5
    html_ok = report_module.render_html(fake_report)
    assert "Magnitude head looks degenerate" not in html_ok
