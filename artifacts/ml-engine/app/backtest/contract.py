"""Loader for shared/trading-frictions.json.

The TypeScript live trader (artifacts/api-server/src/lib/trading-constants.ts)
loads the same file. Drift between the two adapters is a correctness bug.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _find_workspace_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "pnpm-workspace.yaml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(f"Could not locate workspace root from {start}")


def _contract_path() -> Path:
    return _find_workspace_root(Path(__file__)) / "shared" / "trading-frictions.json"


@lru_cache(maxsize=1)
def load_contract() -> dict[str, Any]:
    return json.loads(_contract_path().read_text())


# Reload helper for tests that monkeypatch the file.
def _reset_cache() -> None:
    load_contract.cache_clear()


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    """Task #343 — fail-fast accessor. Raise a precise KeyError instead
    of silently substituting a hard-coded literal when a contract key
    is missing. See trading-constants.ts `_requireConfig` for the TS
    mirror."""
    if key not in d or d[key] is None:
        raise KeyError(
            f"[trading-frictions] required key '{ctx}.{key}' missing "
            f"from shared/trading-frictions.json. Refusing to fall back "
            f"to a hard-coded default — see task #343."
        )
    return d[key]


def _require_tf_lookup(m: dict[str, Any], tf: str, ctx: str) -> Any:
    """Per-timeframe lookup with `_default` cascade — but the cascade
    is allowed to use the JSON `_default` ONLY. Falling back to a
    hard-coded literal is a silent fallback and is forbidden."""
    if tf in m and m[tf] is not None:
        return m[tf]
    if "_default" in m and m["_default"] is not None:
        return m["_default"]
    raise KeyError(
        f"[trading-frictions] '{ctx}' has no entry for tf={tf!r} and no "
        f"'_default' key. Refusing to fall back to a hard-coded literal "
        f"— see task #343."
    )


@dataclass(frozen=True)
class Frictions:
    raw: dict[str, Any]

    # Fees & slippage
    @property
    def maker_fee_pct(self) -> float: return self.raw["fees"]["maker_fee_pct"]
    @property
    def taker_fee_pct(self) -> float: return self.raw["fees"]["taker_fee_pct"]
    @property
    def slippage_pct(self) -> float: return self.raw["fees"]["slippage_pct"]
    @property
    def round_trip_cost_pct(self) -> float:
        return 2 * (self.taker_fee_pct + self.slippage_pct)

    # Risk
    @property
    def initial_balance_usd(self) -> float: return self.raw["risk"]["initial_balance_usd"]
    @property
    def max_open_positions_per_agent(self) -> int: return self.raw["risk"]["max_open_positions_per_agent"]
    @property
    def max_portfolio_at_risk(self) -> float: return self.raw["risk"]["max_portfolio_at_risk"]
    @property
    def daily_loss_limit_pct(self) -> float: return self.raw["risk"]["daily_loss_limit_pct"]
    @property
    def drawdown_halt_pct(self) -> float: return self.raw["risk"]["drawdown_halt_pct"]
    @property
    def max_position_pct(self) -> float: return self.raw["risk"]["max_position_pct"]
    @property
    def asymmetric_long_min_confidence(self) -> float:
        return self.raw["risk"]["asymmetric_long_min_confidence"]

    # Lookups
    def outcome_threshold_pct(self, tf: str) -> float:
        return float(_require_tf_lookup(
            _require(self.raw, "outcome_thresholds_percent", "raw"),
            tf, "outcome_thresholds_percent",
        ))

    def sl_mult(self, tf: str) -> float:
        return float(_require_tf_lookup(
            _require(self.raw, "tf_sl_multiplier", "raw"),
            tf, "tf_sl_multiplier",
        ))

    def tp_mult(self, tf: str) -> float:
        return float(_require_tf_lookup(
            _require(self.raw, "tf_tp_multiplier", "raw"),
            tf, "tf_tp_multiplier",
        ))

    def atr_floor_pct(self, tf: str) -> float:
        return float(_require_tf_lookup(
            _require(self.raw, "tf_atr_floor_pct", "raw"),
            tf, "tf_atr_floor_pct",
        ))

    def timeframe_ms(self, tf: str) -> int:
        return int(_require(
            _require(self.raw, "timeframe_ms", "raw"),
            tf, "timeframe_ms",
        ))

    def tradeable_timeframes(self) -> list[str]:
        return list(_require(self.raw, "tradeable_timeframes", "raw"))

    def tiered_position_pct(self, confidence: float) -> float:
        # Tiers in JSON sorted descending by min_confidence; the lowest entry
        # (min_confidence=0) is the absolute fallback.
        tiers = sorted(
            self.raw["tiered_position_pct"],
            key=lambda t: t["min_confidence"],
            reverse=True,
        )
        for t in tiers:
            if confidence >= t["min_confidence"]:
                return float(t["pct"])
        return float(tiers[-1]["pct"])

    # Trailing
    @property
    def trailing(self) -> dict[str, Any]: return self.raw["trailing_stop"]

    def trailing_extension_ms(self, tf: str) -> int:
        return int(_require_tf_lookup(
            _require(
                _require(self.raw, "trailing_stop", "raw"),
                "expiry_extension_ms", "trailing_stop",
            ),
            tf, "trailing_stop.expiry_extension_ms",
        ))

    # Gates baseline
    def gate(self, key: str) -> dict[str, float]:
        return self.raw["gates_baseline"][key]

    # Regime classifier params
    @property
    def regime_classifier(self) -> dict[str, Any]: return self.raw["regime_classifier"]

    # Deploy gate
    @property
    def deploy_gate(self) -> dict[str, Any]: return self.raw["backtest_deploy_gate"]

    # Quant-brain decision rule (mirror of artifacts/api-server/src/lib/quant-brain.ts).
    @property
    def quant_decision_thresholds(self) -> dict[str, Any]:
        return self.raw["quant_brain"]["decision_thresholds"]

    @property
    def min_directional_prob(self) -> float:
        return float(self.quant_decision_thresholds["min_directional_prob"])

    @property
    def min_directional_edge(self) -> float:
        return float(self.quant_decision_thresholds["min_directional_edge"])

    @property
    def min_expected_return_pct(self) -> float:
        # Live formula: getMinEvVsCost() * ROUND_TRIP_COST_PCT * 100. The
        # backtester always uses the contract baseline (no runtime tuning),
        # which by construction equals the paper-trader EV-gate floor.
        factor = float(self.quant_decision_thresholds["min_expected_return_pct_factor"])
        return factor * self.round_trip_cost_pct * 100.0

    @property
    def quant_policy_version(self) -> str:
        return str(self.quant_decision_thresholds["policy_version"])


def get_frictions() -> Frictions:
    return Frictions(load_contract())
