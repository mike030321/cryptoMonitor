"""Test helper used by the TypeScript parity suite.

Reads a JSON array of decision fixtures from stdin, runs each through
`app.decision_engine.decide` (the SAME pure function the live trader hits
via `/ml/decide` and the offline backtester runs in-process), and writes
the resulting decisions back to stdout as a JSON array. Kept tiny on
purpose so the TS test (artifacts/api-server/test/decision-engine-parity.test.ts)
is the single place that owns the assertion.

Task #345 — extended with a post-decide wrapper that mirrors the two
live-trader-only safeguards the engine itself doesn't model:

  * Fleet correlation brake (paper-trader.ts:384-412): when ≥6 fleet
    positions are open and ≥60% sit on the same side as the candidate,
    the live trader skips with `fleet_direction_imbalance`.
  * Kelly sizing override (paper-trader.ts:493-522): once the agent has
    ≥KELLY_MIN_TRADES closed trades, sizing switches from
    tieredPositionPct(confidence) to a fractional-Kelly formula
    (linearly blended into the tier path between KELLY_MIN_TRADES and
    KELLY_RAMP_END).

Both wrappers are written here AND in the TS mirror with identical
math. The TS test asserts they agree fixture-by-fixture; any drift on
either side fires before the next backtest can ship a verdict.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from app.backtest.contract import load_contract
from app.decision_engine import (
    DecisionRequest,
    OpenPosition,
    PortfolioState,
    decide,
)


def _portfolio(p: dict[str, Any] | None) -> PortfolioState | None:
    if p is None:
        return None
    return PortfolioState(
        equity_usd=float(p["equityUsd"]),
        cash_usd=float(p["cashUsd"]),
        open_positions=[
            OpenPosition(
                coin_id=op["coinId"],
                direction=op["direction"],
                notional_usd=float(op["notionalUsd"]),
                regime_at_entry=op.get("regimeAtEntry"),
                beta_to_btc=op.get("betaToBtc"),
            )
            for op in (p.get("openPositions") or [])
        ],
    )


# ── Live-trader-only safeguards (mirrored in tsApplyLiveExtras) ─────────
_CONTRACT = load_contract()
_FLEET_MIN_OPEN = int(_CONTRACT["fleet_brake"]["min_open_positions"])
_FLEET_DOMINANCE = float(_CONTRACT["fleet_brake"]["same_side_dominance_share"])
_KELLY_MIN_TRADES = int(_CONTRACT["risk"]["kelly_min_trades"])
_KELLY_RAMP_END = int(_CONTRACT["risk"]["kelly_ramp_end"])
_KELLY_FRACTION = float(_CONTRACT["risk"]["kelly_fraction"])
_MAX_POSITION_PCT = float(_CONTRACT["risk"]["max_position_pct"])
_MAX_PORTFOLIO_AT_RISK = float(_CONTRACT["risk"]["max_portfolio_at_risk"])
_TAKER_FEE_PCT = float(_CONTRACT["fees"]["taker_fee_pct"])
_MAX_CASH_PER_POSITION_PCT = 0.80  # paper-trader.ts MAX_CASH_PER_POSITION_PCT

_TF_KELLY_MULT = {"5m": 0.7, "1h": 0.8, "2h": 1.2, "6h": 1.5, "1d": 1.7}


def _tiered_pct(confidence: float) -> float:
    tiers = sorted(
        _CONTRACT["tiered_position_pct"],
        key=lambda t: t["min_confidence"],
        reverse=True,
    )
    for t in tiers:
        if confidence >= t["min_confidence"]:
            return float(t["pct"])
    return float(tiers[-1]["pct"])


def _kelly_size(
    win_rate: float, avg_win: float, avg_loss: float,
    portfolio_value: float, confidence: float, timeframe: str,
) -> float:
    """Mirror of paper-trader.ts:calculateKellySize."""
    if avg_loss <= 0:
        avg_loss = 0.1
    if avg_win <= 0:
        avg_win = 0.1
    R = avg_win / avg_loss
    kelly_pct = (win_rate * R - (1.0 - win_rate)) / R
    if kelly_pct <= 0:
        return 0.0
    fractional = kelly_pct * _KELLY_FRACTION
    conf_mult = 0.8 + confidence * 0.6
    tf_mult = _TF_KELLY_MULT.get(timeframe, 1.0)
    pos_pct = fractional * conf_mult * tf_mult
    pos_pct = min(pos_pct, _MAX_POSITION_PCT)
    if pos_pct < 0.03:
        return 0.0
    return portfolio_value * pos_pct


def _apply_live_extras(fx: dict[str, Any], r) -> dict[str, Any]:
    """Apply fleet brake + Kelly override on top of `decide()`. The TS
    suite runs an identical wrapper; drift either way trips the test."""
    base = {
        "action": r.action,
        "confidence": r.confidence,
        "sizeMultiplier": r.size_multiplier,
        "positionSizeUsd": r.position_size_usd,
        "direction": r.direction,
        "slPrice": r.sl_price,
        "tpPrice": r.tp_price,
        "skipReason": r.skip_reason,
        "skipDetail": r.skip_detail,
    }
    if base["action"] == "no_trade":
        return base

    direction = base["direction"]

    # Fleet correlation brake (mirror of paper-trader.ts:384-412). Fires
    # BEFORE sizing in the live trader, so override the approve here
    # before any Kelly recompute can ride on top.
    fleet = fx.get("fleetState")
    if fleet is not None:
        up = int(fleet.get("up", 0))
        down = int(fleet.get("down", 0))
        total = up + down
        if total >= _FLEET_MIN_OPEN:
            same_side = up if direction == "up" else down
            share = same_side / total if total else 0.0
            if share >= _FLEET_DOMINANCE:
                return {
                    "action": "no_trade",
                    "confidence": 0,
                    "sizeMultiplier": 0,
                    "positionSizeUsd": 0,
                    "direction": None,
                    "slPrice": None,
                    "tpPrice": None,
                    "skipReason": "fleet_direction_imbalance",
                    "skipDetail": (
                        f"sameSideShare={share:.3f}>={_FLEET_DOMINANCE}"
                    ),
                }

    # Kelly sizing override (mirror of paper-trader.ts:491-522). Replaces
    # the engine's tier-based positionSizeUsd with the Kelly-blended size
    # once the agent has crossed KELLY_MIN_TRADES closed trades.
    kelly = fx.get("kellyState")
    if kelly is not None:
        total_trades = int(kelly.get("totalTrades", 0))
        winning = int(kelly.get("winningTrades", 0))
        if total_trades >= _KELLY_MIN_TRADES and winning > 0:
            portfolio = fx.get("portfolio") or {}
            equity = float(portfolio.get("equityUsd", 0))
            cash = float(portfolio.get("cashUsd", 0))
            invested = sum(
                float(p.get("notionalUsd", 0))
                for p in portfolio.get("openPositions") or []
            )
            confidence = float(base["confidence"])
            timeframe = fx["timeframe"]
            win_rate = winning / total_trades
            avg_win = float(kelly.get("avgWinPct", 0.1))
            avg_loss = float(kelly.get("avgLossPct", 0.1))
            kelly_size = _kelly_size(
                win_rate, avg_win, avg_loss, equity, confidence, timeframe,
            )
            if total_trades < _KELLY_RAMP_END:
                ramp = (total_trades - _KELLY_MIN_TRADES) / (
                    _KELLY_RAMP_END - _KELLY_MIN_TRADES
                )
                fixed_size = equity * _tiered_pct(confidence)
                pos = fixed_size * (1 - ramp) + kelly_size * ramp
            else:
                pos = kelly_size
            pos = min(
                pos, cash * _MAX_CASH_PER_POSITION_PCT,
                equity * _MAX_POSITION_PCT,
            )
            if equity > 0 and (
                (invested + pos) / equity >= _MAX_PORTFOLIO_AT_RISK
            ):
                pos = max(0.0, equity * _MAX_PORTFOLIO_AT_RISK - invested)
            entry_fee = pos * _TAKER_FEE_PCT
            base["positionSizeUsd"] = pos - entry_fee
    return base


def _run_one(fx: dict[str, Any]) -> dict[str, Any]:
    req = DecisionRequest(
        coin_id=fx["coinId"],
        timeframe=fx["timeframe"],
        last_price=float(fx["lastPrice"]),
        atr_value=float(fx["atrValue"]),
        prob_up=float(fx["probUp"]),
        prob_down=float(fx["probDown"]),
        prob_stable=float(fx["probStable"]),
        expected_return_pct=float(fx["expectedReturnPct"]),
        regime=fx.get("regime"),
        trend_bias=fx.get("trendBias"),
        portfolio=_portfolio(fx.get("portfolio")),
        recent_outcomes=list(fx.get("recentOutcomes") or []),
        gate_min_confidence=fx.get("gateMinConfidence"),
        gate_min_tp_distance_pct=fx.get("gateMinTpDistancePct"),
        gate_min_ev_vs_cost=fx.get("gateMinEvVsCost"),
        gate_counter_trend_min_confidence=fx.get("gateCounterTrendMinConfidence"),
        portfolio_constraints_override=fx.get("portfolioConstraintsOverride"),
    )
    r = decide(req)
    return _apply_live_extras(fx, r)


def main() -> None:
    fixtures = json.load(sys.stdin)
    out = [_run_one(fx) for fx in fixtures]
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
