"""Task #356 — fail-fast coverage for the shared/trading-frictions.json
loaders on the Python side.

Tasks #343 and #349 hardened every Python consumer of the contract so a
missing required key throws at startup instead of silently substituting
a hardcoded default. This file pins those throw paths so a future
refactor that quietly re-introduces a silent fallback fails CI here
instead of in production.

Mirrored TS coverage lives at
artifacts/api-server/test/trading-frictions-fail-fast.test.ts — both
sides must stay in lock-step (drift between live trader and
backtester is a correctness bug).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.backtest.contract import (
    Frictions,
    _require,
    _require_tf_lookup,
    get_frictions,
)
from app.decision_engine.engine import (
    DecisionRequest,
    OpenPosition,
    PortfolioState,
    _check_portfolio,
    decide,
)
from app.registry_lifecycle import PromotionMetrics, evaluate_promotion


def _load_real_contract() -> dict:
    here = Path(__file__).resolve()
    workspace_root = here.parents[3]
    return json.loads(
        (workspace_root / "shared" / "trading-frictions.json").read_text()
    )


def _delete_at(d: dict, path: list[str]) -> dict:
    """Return a deep copy of `d` with the leaf at `path` removed."""
    out = copy.deepcopy(d)
    cur = out
    for k in path[:-1]:
        cur = cur[k]
    del cur[path[-1]]
    return out


# ── champion_challenger (registry_lifecycle.py, task #349) ───────────────
CHAMPION_CHALLENGER_KEYS = [
    (["champion_challenger"], "raw.champion_challenger"),
    (
        ["champion_challenger", "min_shadow_samples"],
        "champion_challenger.min_shadow_samples",
    ),
    (
        ["champion_challenger", "min_net_edge_lift_vs_champion"],
        "champion_challenger.min_net_edge_lift_vs_champion",
    ),
    (
        ["champion_challenger", "max_drawdown_pct"],
        "champion_challenger.max_drawdown_pct",
    ),
    (
        ["champion_challenger", "min_passing_regimes_share"],
        "champion_challenger.min_passing_regimes_share",
    ),
]


@pytest.mark.parametrize("path,key_ctx", CHAMPION_CHALLENGER_KEYS)
def test_evaluate_promotion_raises_when_required_key_missing(
    monkeypatch, path, key_ctx
):
    """The lifecycle gate must refuse to load when ANY required
    `champion_challenger` knob is absent — the previous silent fallback
    to (200, 0.5, 20.0, 0.5) let a contract typo promote challengers
    against thresholds that exist nowhere in the JSON."""
    mutated = _delete_at(_load_real_contract(), path)
    monkeypatch.setattr(
        "app.registry_lifecycle.get_frictions",
        lambda: Frictions(raw=mutated),
    )
    metrics = PromotionMetrics(
        samples=500,
        net_edge_pct=0.7,
        champion_net_edge_pct=0.0,
        drawdown_pct=5.0,
        per_regime_net_edge_pct={"bull": 0.5, "bear": 0.6},
    )
    with pytest.raises(KeyError) as excinfo:
        evaluate_promotion(metrics)
    assert key_ctx in str(excinfo.value)
    assert "task #343" in str(excinfo.value)


# ── portfolio_constraints (engine.py:_check_portfolio, task #349) ────────
PORTFOLIO_CONSTRAINTS_KEYS = [
    (["portfolio_constraints"], "raw.portfolio_constraints"),
    (
        ["portfolio_constraints", "enabled"],
        "portfolio_constraints.enabled",
    ),
    (
        ["portfolio_constraints", "sector_map"],
        "portfolio_constraints.sector_map",
    ),
    (
        ["portfolio_constraints", "sector_map", "_default"],
        "portfolio_constraints.sector_map._default",
    ),
    (
        ["portfolio_constraints", "max_sector_exposure_pct"],
        "portfolio_constraints.max_sector_exposure_pct",
    ),
    (
        ["portfolio_constraints", "max_correlated_exposure_pct"],
        "portfolio_constraints.max_correlated_exposure_pct",
    ),
    (
        ["portfolio_constraints", "max_beta_to_btc"],
        "portfolio_constraints.max_beta_to_btc",
    ),
    (
        ["portfolio_constraints", "regime_budget_pct"],
        "portfolio_constraints.regime_budget_pct",
    ),
]


def _portfolio_with_two_open_positions() -> PortfolioState:
    # Two coins in the same sector so the correlated-exposure branch is
    # reachable (even though most tests trip earlier checks first).
    return PortfolioState(
        equity_usd=1000.0,
        cash_usd=500.0,
        open_positions=[
            OpenPosition(
                coin_id="bitcoin",
                direction="up",
                notional_usd=200.0,
                regime_at_entry="bull",
                beta_to_btc=1.0,
            ),
            OpenPosition(
                coin_id="ethereum",
                direction="up",
                notional_usd=200.0,
                regime_at_entry="bull",
                beta_to_btc=1.0,
            ),
        ],
    )


@pytest.mark.parametrize("path,key_ctx", PORTFOLIO_CONSTRAINTS_KEYS)
def test_check_portfolio_raises_when_required_key_missing(path, key_ctx):
    """The fleet-level gate must refuse to evaluate when ANY required
    `portfolio_constraints` knob is absent. Previously the loader used
    `fr.raw.get("portfolio_constraints") or {}` and downstream
    `pc.get("enabled", False)` made every fleet gate silently pass."""
    mutated = _delete_at(_load_real_contract(), path)
    fr = Frictions(raw=mutated)
    with pytest.raises(KeyError) as excinfo:
        _check_portfolio(
            coin_id="bitcoin",
            new_notional=100.0,
            regime="bull",
            portfolio=_portfolio_with_two_open_positions(),
            fr=fr,
        )
    assert key_ctx in str(excinfo.value)
    assert "task #343" in str(excinfo.value)


def test_check_portfolio_passes_with_unmodified_contract():
    """Sanity check — without this every "missing X" test could pass
    for the wrong reason (e.g. base contract already missing something)."""
    fr = Frictions(raw=_load_real_contract())
    ok, skip, breakdown = _check_portfolio(
        coin_id="bitcoin",
        new_notional=10.0,
        regime="bull",
        portfolio=PortfolioState(equity_usd=10_000.0, cash_usd=10_000.0),
        fr=fr,
    )
    # On a fresh book the small new position cannot trip any cap.
    assert ok is True
    assert skip is None
    assert breakdown["enabled"] is True


# ── decide() also surfaces the throw via _check_portfolio ─────────────────
def test_decide_propagates_portfolio_constraints_missing_key():
    """End-to-end coverage: `decide()` is the entry point used by both
    the live trader and the backtester. If the JSON contract loses
    `portfolio_constraints` it must raise out of `decide()` rather than
    silently emitting a trade against absent fleet gates."""
    mutated = _delete_at(_load_real_contract(), ["portfolio_constraints"])
    fr = Frictions(raw=mutated)
    req = DecisionRequest(
        coin_id="bitcoin",
        timeframe="5m",
        last_price=100.0,
        atr_value=1.0,
        prob_up=0.7,
        prob_down=0.2,
        prob_stable=0.1,
        expected_return_pct=0.5,
        regime="bull",
        portfolio=PortfolioState(equity_usd=1_000.0, cash_usd=1_000.0),
    )
    with pytest.raises(KeyError) as excinfo:
        decide(req, fr=fr)
    assert "raw.portfolio_constraints" in str(excinfo.value)


# ── per-tf `_default` cascade (contract.py, task #343) ────────────────────
TF_LOOKUP_BLOCKS_AND_METHODS = [
    ("outcome_thresholds_percent", "outcome_threshold_pct"),
    ("tf_sl_multiplier", "sl_mult"),
    ("tf_tp_multiplier", "tp_mult"),
    ("tf_atr_floor_pct", "atr_floor_pct"),
]


@pytest.mark.parametrize("block,method", TF_LOOKUP_BLOCKS_AND_METHODS)
def test_frictions_per_tf_lookup_raises_without_default(block, method):
    """Each per-tf map is allowed to use the JSON `_default` cascade,
    but losing the cascade entirely must throw — never silently fall
    back to a Python-side literal."""
    raw = _load_real_contract()
    raw[block] = {k: v for k, v in raw[block].items() if k != "_default"}
    fr = Frictions(raw=raw)
    with pytest.raises(KeyError) as excinfo:
        getattr(fr, method)("__no_such_tf__")
    assert block in str(excinfo.value)
    assert "_default" in str(excinfo.value)
    assert "__no_such_tf__" in str(excinfo.value)


def test_frictions_per_tf_lookup_throws_when_block_missing_entirely():
    """Removing the whole map (not just its `_default`) must also
    surface as a `_require` failure — covered by `_require` ahead of
    `_require_tf_lookup` in every accessor."""
    raw = _load_real_contract()
    del raw["outcome_thresholds_percent"]
    fr = Frictions(raw=raw)
    with pytest.raises(KeyError) as excinfo:
        fr.outcome_threshold_pct("5m")
    assert "raw.outcome_thresholds_percent" in str(excinfo.value)
    assert "task #343" in str(excinfo.value)


# ── quant_brain.decision_thresholds (contract.py, task #343) ──────────────
QUANT_BRAIN_DECISION_KEYS = [
    "min_directional_prob",
    "min_directional_edge",
    "min_expected_return_pct_factor",
    "policy_version",
]


@pytest.mark.parametrize("missing_key", QUANT_BRAIN_DECISION_KEYS)
def test_frictions_quant_decision_threshold_property_raises_when_key_missing(
    missing_key,
):
    """The properties on Frictions that surface quant-brain decision
    thresholds (consumed by both the live trader and the backtester
    simulator) must blow up loudly when their JSON knob is absent."""
    raw = _load_real_contract()
    del raw["quant_brain"]["decision_thresholds"][missing_key]
    fr = Frictions(raw=raw)
    if missing_key == "policy_version":
        with pytest.raises(KeyError):
            _ = fr.quant_policy_version
    elif missing_key == "min_expected_return_pct_factor":
        with pytest.raises(KeyError):
            _ = fr.min_expected_return_pct
    elif missing_key == "min_directional_prob":
        with pytest.raises(KeyError):
            _ = fr.min_directional_prob
    elif missing_key == "min_directional_edge":
        with pytest.raises(KeyError):
            _ = fr.min_directional_edge


def test_frictions_quant_decision_thresholds_block_missing_raises():
    raw = _load_real_contract()
    del raw["quant_brain"]["decision_thresholds"]
    fr = Frictions(raw=raw)
    with pytest.raises(KeyError):
        _ = fr.quant_decision_thresholds


# ── _require / _require_tf_lookup primitives ──────────────────────────────
def test_require_raises_with_precise_path():
    with pytest.raises(KeyError) as excinfo:
        _require({"a": 1}, "b", "ctx")
    msg = str(excinfo.value)
    assert "ctx.b" in msg
    assert "task #343" in msg


def test_require_raises_when_value_is_explicitly_none():
    """Explicit None is treated the same as a missing key — otherwise a
    JSON `null` would silently pass `key in d` and short-circuit the
    fail-fast guard."""
    with pytest.raises(KeyError):
        _require({"a": None}, "a", "ctx")


def test_require_tf_lookup_returns_default_when_tf_missing():
    """The cascade is allowed: a missing tf with a present `_default`
    is the documented fallback path."""
    assert _require_tf_lookup({"_default": 0.5}, "5m", "ctx") == 0.5


def test_require_tf_lookup_raises_when_neither_tf_nor_default_present():
    with pytest.raises(KeyError) as excinfo:
        _require_tf_lookup({"1m": 0.1}, "5m", "ctx")
    msg = str(excinfo.value)
    assert "ctx" in msg
    assert "5m" in msg
    assert "_default" in msg
