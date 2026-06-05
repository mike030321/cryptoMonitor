"""Task #391 — focused regression test for benchmark→QuantBridge isolation
and the soft-clamp truth table.

Two invariants this file guards against silent regression of:

  1. The optional `benchmark` block must be popped from the payload
     BEFORE `QuantBridge.from_payload` is called. The bridge has no
     concept of benchmark and any leakage indicates the governance
     contract was broken.

  2. The defensive-mode soft-clamp (`off → soft`) must fire ONLY when
     BOTH `sustainedUnderperformance is True` AND `relativeAlpha14d < 0`.
     The other three cells of the truth table must leave `defensive_mode`
     at "off". Under no cell is `hard` an acceptable outcome — the
     benchmark path is forbidden from escalating past "soft".
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import meta_brain
from app.main import app


@pytest.fixture(autouse=True)
def _init_brain():
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain.init()
    yield
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()


def _slice(family: str = "momentum") -> dict:
    return {
        "coin": "btc",
        "timeframe": "5m",
        "strategy_family": family,
        "edge": 0.01,
        "confidence": 0.6,
        "calibrated_confidence": 0.6,
        "risk_score": 0.4,
        "recent_accuracy": 0.55,
        "pnl_state": 0.0,
        "drawdown_state": 0.0,
        "disagreement": 0.1,
        "prediction_error": 0.0,
        "regime": "trend_up",
        "volatility": 0.02,
        "correlation_shift": 0.0,
        "exposure": 0.1,
        "turnover": 0.05,
        "slippage_bps": 1.0,
        "anomaly_flags": [],
    }


def _portfolio() -> dict:
    return {
        "total_drawdown": 0.05,
        "realized_vol": 0.02,
        "concentration": 0.3,
        "leverage": 1.0,
        "liquidity_stress": 0.1,
        "correlation_shift": 0.05,
        "active_risk_budget": 0.6,
        "kill_switch_distance": 0.5,
        "anomaly_flags": [],
    }


def _benchmark(
    *, sustained: bool, alpha14: float, alpha7: float | None = None
) -> dict:
    return {
        "aiReturn7d": -0.04 if alpha14 < 0 else 0.03,
        "bestBaselineReturn7d": 0.02 if alpha14 < 0 else 0.005,
        "relativeAlpha7d": float(alpha7 if alpha7 is not None else alpha14),
        "relativeAlpha14d": float(alpha14),
        "drawdownRatioVsBest": 1.4 if alpha14 < 0 else 0.7,
        "sustainedUnderperformance": bool(sustained),
        "sampleCount": 80,
        "stale": False,
    }


def _post(body: dict) -> dict:
    client = TestClient(app)
    r = client.post("/ml/meta-brain/evaluate", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_benchmark_is_popped_before_quant_bridge_from_payload(monkeypatch):
    """Strict isolation: spy on the bridge and assert the payload it
    sees has no `benchmark` key — and that the synthetic benchmark
    family slice WAS appended to `slices`. This guards against future
    refactors that accidentally pass the raw benchmark struct through
    to the vendor bridge.
    """
    seen: dict = {}
    real_from_payload = meta_brain._bridge.from_payload

    def spy(payload):
        # Snapshot the payload shape the bridge actually receives.
        seen["has_benchmark_key"] = "benchmark" in payload
        seen["payload_keys"] = sorted(payload.keys())
        seen["families"] = [
            s.get("strategy_family") for s in payload.get("slices", [])
        ]
        return real_from_payload(payload)

    monkeypatch.setattr(meta_brain._bridge, "from_payload", spy)

    body = {
        "slices": [_slice("momentum")],
        "portfolio": _portfolio(),
        "benchmark": _benchmark(sustained=True, alpha14=-0.05),
        "timestamp": "2026-04-23T00:00:00Z",
    }
    _post(body)

    assert seen, "QuantBridge.from_payload was never called"
    assert seen["has_benchmark_key"] is False, (
        f"benchmark leaked into bridge payload: keys={seen['payload_keys']}"
    )
    assert "benchmark" not in seen["payload_keys"]
    # The synthetic benchmark family slice must be the only carrier of
    # benchmark telemetry into the planner.
    assert "benchmark" in seen["families"], (
        f"synthetic benchmark family slice missing: {seen['families']}"
    )


@pytest.mark.parametrize(
    "sustained,alpha14,expected_mode,expect_reason",
    [
        # The single soft-clamp cell.
        (True, -0.05, "soft", True),
        # The three off cells — neither flag, only one flag, or wrong sign.
        (True, 0.05, "off", False),
        (False, -0.05, "off", False),
        (False, 0.05, "off", False),
    ],
    ids=[
        "sustained_and_alpha_negative__soft",
        "sustained_but_alpha_positive__off",
        "alpha_negative_but_not_sustained__off",
        "neither_flag__off",
    ],
)
def test_soft_clamp_truth_table_and_never_hard(
    sustained, alpha14, expected_mode, expect_reason
):
    """Four-cell truth table for the `off → soft` gate. Only the
    `(sustained=True, alpha14<0)` cell may flip to "soft". No cell may
    escalate to "hard" — the benchmark path is forbidden from reaching
    `hard` (that mode is reserved for portfolio-only `risk_pressure`).
    """
    body = {
        "slices": [_slice("momentum"), _slice("mean_reversion")],
        "portfolio": _portfolio(),
        "benchmark": _benchmark(sustained=sustained, alpha14=alpha14),
        "timestamp": "2026-04-23T00:00:00Z",
    }
    out = _post(body)

    mode = out["defensive_mode"]
    assert mode != "hard", (
        f"benchmark path must never escalate to hard; "
        f"sustained={sustained} alpha14={alpha14} got {mode}"
    )
    assert mode == expected_mode, (
        f"sustained={sustained} alpha14={alpha14} expected {expected_mode}, "
        f"got {mode}"
    )
    has_reason = "benchmark_alpha_negative" in out.get("reason_codes", [])
    assert has_reason is expect_reason, (
        f"reason code mismatch for sustained={sustained} alpha14={alpha14}: "
        f"expected_present={expect_reason}, got_present={has_reason}"
    )
