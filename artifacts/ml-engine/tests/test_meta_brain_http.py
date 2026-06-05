"""Task #381 — meta-brain HTTP contract + safety tests.

Covers:
  * /evaluate accepts null numeric fields (auto-flagged missing)
  * /evaluate returns a directive whose allocation_weight sums to 1
  * /record-outcome with unknown tick_id returns {ok: false} (200, not 5xx)
  * /record-outcome roundtrip on the same tick_id returns {ok: true}
  * trust_model.json checkpoint is rehydrated on init()
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import meta_brain
from app.main import app


@pytest.fixture(autouse=True)
def _init_brain():
    # Reset state between tests by re-init'ing.
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain.init()
    yield
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()


def _slice(family="momentum"):
    return {
        "coin": "btc",
        "timeframe": "5m",
        "strategy_family": family,
        "edge": 0.01,
        "confidence": 0.6,
        "calibrated_confidence": 0.6,
        "risk_score": 0.4,
        "recent_accuracy": None,  # null — adapter must handle
        "pnl_state": None,
        "drawdown_state": None,
        "disagreement": 0.1,
        "prediction_error": 0.0,
        "regime": "trend_up",
        "volatility": 0.02,
        "correlation_shift": None,
        "exposure": None,
        "turnover": None,
        "slippage_bps": None,
        "anomaly_flags": [],
    }


def _portfolio_null():
    return {
        "total_drawdown": None,
        "realized_vol": None,
        "concentration": None,
        "leverage": None,
        "liquidity_stress": None,
        "correlation_shift": None,
        "active_risk_budget": None,
        "kill_switch_distance": None,
        "anomaly_flags": [],
    }


def test_health_endpoint_reports_ready():
    client = TestClient(app)
    r = client.get("/ml/meta-brain/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    assert "tick_cache_size" in body
    assert "state_root" in body


def test_evaluate_accepts_null_numeric_fields():
    client = TestClient(app)
    body = {
        "slices": [_slice()],
        "portfolio": _portfolio_null(),
        "timestamp": "2026-04-23T00:00:00Z",
    }
    r = client.post("/ml/meta-brain/evaluate", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert "tick_id" in out
    assert "allocation_weight" in out
    s = sum(out["allocation_weight"].values())
    assert abs(s - 1.0) < 1e-6, f"allocation must sum to 1, got {s}"


def test_record_outcome_unknown_tick_returns_ok_false():
    client = TestClient(app)
    r = client.post(
        "/ml/meta-brain/record-outcome",
        json={
            "tick_id": "never-issued",
            "outcome": {
                "realized_pnl": 0.01,
                "realized_drawdown": 0.005,
                "realized_stability": 0.7,
                "turnover_cost": 0.001,
                "action_churn": 0.0,
                "correct_defense": 0.0,
                "correct_suppression": 0.0,
                "missed_edge_cost": 0.0,
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is False
    assert "tick_id_not_in_cache" in body.get("reason", "")


def test_record_outcome_roundtrip():
    client = TestClient(app)
    body = {
        "slices": [_slice()],
        "portfolio": _portfolio_null(),
    }
    r = client.post("/ml/meta-brain/evaluate", json=body)
    tick_id = r.json()["tick_id"]
    r2 = client.post(
        "/ml/meta-brain/record-outcome",
        json={
            "tick_id": tick_id,
            "outcome": {
                "realized_pnl": 0.005,
                "realized_drawdown": 0.002,
                "realized_stability": 0.8,
                "turnover_cost": 0.0005,
                "action_churn": 0.0,
                "correct_defense": 0.0,
                "correct_suppression": 0.0,
                "missed_edge_cost": 0.0,
            },
        },
    )
    assert r2.status_code == 200, r2.text
    assert r2.json().get("ok") is True


def test_trust_model_hydrate_from_checkpoint(tmp_path, monkeypatch):
    # Point _STATE_ROOT at a temp dir, write a checkpoint, re-init.
    monkeypatch.setattr(meta_brain, "_STATE_ROOT", tmp_path)
    (tmp_path / "trust_model.json").write_text(
        json.dumps(
            {
                "momentum": {
                    "trust": 1.4,
                    "stability": 0.8,
                    "exploration_eligibility": 0.05,
                    "failure_streak": 0,
                    "recovery_score": 0.9,
                    "last_regime": "trend_up",
                }
            }
        ),
        encoding="utf-8",
    )
    meta_brain._service = None  # force re-init
    meta_brain.init()
    fams = meta_brain._service.trust_model.trust_by_family  # type: ignore[union-attr]
    assert "momentum" in fams
    assert abs(fams["momentum"].trust - 1.4) < 1e-9
    assert fams["momentum"].last_regime == "trend_up"


def test_normalize_payload_in_place_records_missing_flags():
    payload = {
        "slices": [_slice()],
        "portfolio": _portfolio_null(),
    }
    meta_brain._normalize_payload_in_place(payload)
    flags = payload["slices"][0]["anomaly_flags"]
    assert any(f.startswith("missing:recent_accuracy") for f in flags)
    assert any(f.startswith("missing:exposure") for f in flags)
    pflags = payload["portfolio"]["anomaly_flags"]
    assert any(f.startswith("missing:total_drawdown") for f in pflags)


def test_stats_returns_trust_table():
    client = TestClient(app)
    r = client.get("/ml/meta-brain/stats")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert "trust_by_family" in body
