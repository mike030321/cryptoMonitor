"""Task #390 — meta-brain benchmark governance behavior + fallback tests.

Covers:
  * /evaluate accepts an optional `benchmark` block and returns a
    directive whose allocation_weight still sums to 1 (no schema
    contamination of the directive surface).
  * A non-stale benchmark with sustained negative alpha NUDGES
    `defensive_mode` from "off" → "soft" via the soft-cap path.
  * The benchmark signal NEVER escalates `defensive_mode` to "hard"
    on its own — invariant the api-server relies on.
  * The benchmark block is recorded as an episode through the official
    `record_outcome` learning loop (regime memory observes a reward,
    episodic memory grows by 1) and the synthetic `benchmark` family
    appears in `trust_by_family` after one cycle.
  * `benchmark` keys are popped before QuantBridge.from_payload, so
    the payload schema seen by the bridge is identical to the
    pre-#390 shape.
  * Stale benchmark + absent benchmark produce identical directives —
    fail-safe contract.
"""
from __future__ import annotations

from copy import deepcopy

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


def _benchmark_underperforming() -> dict:
    return {
        "aiReturn7d": -0.04,
        "bestBaselineReturn7d": 0.02,
        "relativeAlpha7d": -0.06,
        "relativeAlpha14d": -0.05,
        "drawdownRatioVsBest": 1.4,
        "sustainedUnderperformance": True,
        "sampleCount": 80,
        "stale": False,
    }


def _benchmark_outperforming() -> dict:
    return {
        "aiReturn7d": 0.03,
        "bestBaselineReturn7d": 0.005,
        "relativeAlpha7d": 0.025,
        "relativeAlpha14d": 0.02,
        "drawdownRatioVsBest": 0.7,
        "sustainedUnderperformance": False,
        "sampleCount": 80,
        "stale": False,
    }


def _benchmark_stale() -> dict:
    return {
        "aiReturn7d": 0.0,
        "bestBaselineReturn7d": 0.0,
        "relativeAlpha7d": 0.0,
        "relativeAlpha14d": 0.0,
        "drawdownRatioVsBest": 1.0,
        "sustainedUnderperformance": False,
        "sampleCount": 0,
        "stale": True,
    }


def _evaluate(benchmark: dict | None) -> dict:
    client = TestClient(app)
    body: dict = {
        "slices": [_slice("momentum"), _slice("mean_reversion")],
        "portfolio": _portfolio(),
        "timestamp": "2026-04-23T00:00:00Z",
    }
    if benchmark is not None:
        body["benchmark"] = benchmark
    r = client.post("/ml/meta-brain/evaluate", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _alloc_sums_to_one(out: dict) -> None:
    s = sum(out["allocation_weight"].values())
    assert abs(s - 1.0) < 1e-6, f"allocation must sum to 1, got {s}"


def test_evaluate_accepts_benchmark_block_and_directive_is_clean():
    out = _evaluate(_benchmark_underperforming())
    # Directive surface stays clean — no benchmark/alpha/baseline keys
    # leak into the response.
    forbidden = ("benchmark", "alpha", "baseline", "strategy_lab")
    for k in out.keys():
        kl = k.lower()
        assert not any(p in kl for p in forbidden), (
            f"directive key {k!r} contains a forbidden benchmark prefix"
        )
    _alloc_sums_to_one(out)


def test_sustained_underperformance_soft_clamps_defensive_mode():
    out = _evaluate(_benchmark_underperforming())
    assert out["defensive_mode"] in ("soft", "hard"), out["defensive_mode"]
    # Reason code recorded so downstream observability can attribute.
    if out["defensive_mode"] == "soft":
        assert "benchmark_alpha_negative" in out.get("reason_codes", [])


def test_benchmark_alone_never_escalates_to_hard_defensive_mode():
    # Deterministic invariant: with neutral quant slices + portfolio
    # (so the no-benchmark baseline is "off"), even an extreme
    # punishing benchmark can never drive `defensive_mode` past "soft".
    bm = deepcopy(_benchmark_underperforming())
    bm["aiReturn7d"] = -0.99
    bm["bestBaselineReturn7d"] = 0.99
    bm["relativeAlpha7d"] = -0.99
    bm["relativeAlpha14d"] = -0.99
    bm["drawdownRatioVsBest"] = 9.0
    bm["sustainedUnderperformance"] = True
    out = _evaluate(bm)
    assert out["defensive_mode"] in ("off", "soft"), (
        f"benchmark alone must never escalate to hard, got {out['defensive_mode']}"
    )


def test_baseline_no_benchmark_is_off_for_neutral_inputs():
    # Sanity check the precondition for the soft-cap test above:
    # without any benchmark, the same neutral batch yields "off".
    out = _evaluate(None)
    assert out["defensive_mode"] == "off", (
        f"baseline must be off for neutral inputs, got {out['defensive_mode']}"
    )


def test_benchmark_flows_through_record_outcome_loop():
    """Spec Step 3: benchmark uses the existing `record_outcome` loop —
    one episode pushed per evaluate, benchmark family slot populated
    by the broadcast learn_from_outcome (which by spec also "trims
    trust on losing families")."""
    svc = meta_brain._service
    assert svc is not None
    ep_before = len(svc.episodic_memory._buffer)
    _evaluate(_benchmark_underperforming())
    ep_after = len(svc.episodic_memory._buffer)
    assert ep_after == ep_before + 1, (
        f"expected one new episode from record_outcome, got {ep_before} → {ep_after}"
    )
    assert "benchmark" in svc.trust_model.trust_by_family
    fts = svc.trust_model.trust_by_family["benchmark"]
    assert 0.0 <= fts.trust <= 5.0
    assert 0.0 <= fts.stability <= 1.0


def test_one_evaluate_request_runs_one_service_evaluate(monkeypatch):
    """Soft-cap is enforced by construction (synthetic family slice
    cannot push portfolio-only `risk_pressure`); no re-evaluation."""
    svc = meta_brain._service
    assert svc is not None
    real_eval = svc.evaluate
    calls = {"n": 0}

    def counting_eval(batch):
        calls["n"] += 1
        return real_eval(batch)

    monkeypatch.setattr(svc, "evaluate", counting_eval)
    _evaluate(_benchmark_underperforming())
    assert calls["n"] == 1, (
        f"benchmark governance must not re-evaluate; got {calls['n']} calls"
    )


def test_benchmark_keys_never_reach_quant_bridge():
    # Sanity: pop happens before QuantBridge.from_payload — if the
    # payload still contained a `benchmark` key, the bridge would
    # raise (its schema rejects unknown top-level keys).
    out = _evaluate(_benchmark_outperforming())
    _alloc_sums_to_one(out)


def test_stale_benchmark_and_absent_benchmark_produce_equivalent_directives():
    # Two cycles from a fresh brain. Compare structural fields the
    # planner exposes; we don't compare allocation_weight values
    # directly because record_outcome side-effects update trust state.
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain.init()
    out_absent = _evaluate(None)

    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain.init()
    out_stale = _evaluate(_benchmark_stale())

    # Same structural shape (no soft-clamp, no benchmark reason code).
    assert out_absent["defensive_mode"] == out_stale["defensive_mode"]
    assert (
        "benchmark_alpha_negative" not in out_absent.get("reason_codes", [])
    )
    assert (
        "benchmark_alpha_negative" not in out_stale.get("reason_codes", [])
    )
    # Both keep the directive surface clean.
    for o in (out_absent, out_stale):
        for k in o.keys():
            kl = k.lower()
            assert not any(
                p in kl
                for p in ("benchmark", "alpha", "baseline", "strategy_lab")
            )


def test_stats_endpoint_exposes_benchmark_trust():
    client = TestClient(app)
    _evaluate(_benchmark_underperforming())
    r = client.get("/ml/meta-brain/stats")
    assert r.status_code == 200, r.text
    body = r.json()
    # The synthetic benchmark trust slot is observable to api-server.
    tbf = body.get("trust_by_family", {})
    assert "benchmark" in tbf, f"expected benchmark in trust_by_family: {tbf}"
