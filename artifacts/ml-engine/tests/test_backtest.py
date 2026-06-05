"""Phase-3 backtester tests.

Cover the four task contracts:
* simulator gates / SL/TP/expiry/trailing / fee math
* metrics
* Monte Carlo determinism
* regime breakdown
* decision rule (pass + fail)
* end-to-end smoke run on a synthetic dataset
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtest.contract import _reset_cache, get_frictions, load_contract
from app.backtest.decision import decide
from app.backtest.metrics import compute_metrics, max_drawdown_pct
from app.backtest.monte_carlo import run_monte_carlo
from app.backtest.regime import classify_regime
from app.backtest.regime_breakdown import regime_breakdown
from app.backtest.simulator import (
    CoinTickStream,
    TradeRow,
    apply_entry_slippage,
    apply_exit_slippage,
    decide_direction,
    is_stop_loss_hit,
    is_take_profit_hit,
    round_trip_pnl_usd,
    simulate,
)


# ── Contract loader ────────────────────────────────────────────────────────
def test_contract_loads_and_matches_json():
    raw = load_contract()
    fr = get_frictions()
    assert fr.taker_fee_pct == raw["fees"]["taker_fee_pct"]
    assert fr.initial_balance_usd == raw["risk"]["initial_balance_usd"]
    assert fr.tradeable_timeframes() == raw["tradeable_timeframes"]
    # Round-trip cost matches the algebra: 2*(taker+slippage).
    assert math.isclose(
        fr.round_trip_cost_pct,
        2 * (raw["fees"]["taker_fee_pct"] + raw["fees"]["slippage_pct"]),
    )


def test_tiered_position_pct_picks_correct_tier():
    fr = get_frictions()
    # Highest tier
    assert fr.tiered_position_pct(0.95) == 0.22
    # Mid tier
    assert fr.tiered_position_pct(0.55) == 0.14
    # Lowest fallback
    assert fr.tiered_position_pct(0.01) == 0.05


# ── Pure math (mirror of trade-math.ts) ────────────────────────────────────
def test_slippage_directions():
    assert apply_entry_slippage(100.0, "up", 0.001) == pytest.approx(100.1)
    assert apply_entry_slippage(100.0, "down", 0.001) == pytest.approx(99.9)
    assert apply_exit_slippage(100.0, "up", 0.001) == pytest.approx(99.9)
    assert apply_exit_slippage(100.0, "down", 0.001) == pytest.approx(100.1)


def test_sl_tp_hit_predicates():
    assert is_stop_loss_hit("up", 90, 95) is True
    assert is_stop_loss_hit("down", 110, 105) is True
    assert is_take_profit_hit("up", 110, 105) is True
    assert is_take_profit_hit("down", 90, 95) is True


def test_round_trip_pnl_positive_long():
    fr = get_frictions()
    # 10 % move long; check sign + rough magnitude.
    pnl = round_trip_pnl_usd("up", 100.0, 110.0, 1000.0, fr=fr)
    assert pnl > 0
    # Gross approx 1000 * 0.10 = 100; minus ~0.3% costs ≈ 96.
    assert 90 < pnl < 100


def test_round_trip_pnl_short_loses_when_price_rises():
    fr = get_frictions()
    pnl = round_trip_pnl_usd("down", 100.0, 110.0, 1000.0, fr=fr)
    assert pnl < 0


# ── Live-brain emission rule (mirror of quant-brain.ts) ────────────────────
def test_decide_direction_emits_when_all_gates_pass():
    fr = get_frictions()
    # Strong probUp (0.55), strong edge (0.45), expRet positive and well
    # above the EV-vs-cost floor: emit "up" with confidence = probUp.
    side, conf, reason = decide_direction(0.10, 0.35, 0.55, 1.5, fr=fr)
    assert side == "up"
    assert conf == pytest.approx(0.55)
    assert reason is None


def test_decide_direction_emits_down():
    fr = get_frictions()
    side, conf, reason = decide_direction(0.55, 0.35, 0.10, -1.5, fr=fr)
    assert side == "down"
    assert conf == pytest.approx(0.55)
    assert reason is None


def test_decide_direction_matches_ts_reference():
    """Parity test: load the JSON fixture produced by
    artifacts/ml-engine/tests/_gen_quant_brain_reference.mjs (which is a
    line-for-line transliteration of artifacts/api-server/src/lib/quant-brain.ts)
    and assert the Python implementation returns identical decisions for
    every row of a representative grid (emit, every abstain branch,
    boundaries). If the live brain rule changes, regenerate the fixture
    via `node artifacts/ml-engine/tests/_gen_quant_brain_reference.mjs >
    artifacts/ml-engine/tests/fixtures/quant_brain_parity.json` and the
    test will surface any drift between the two implementations.
    """
    fr = get_frictions()
    fixture_path = Path(__file__).parent / "fixtures" / "quant_brain_parity.json"
    fx = json.loads(fixture_path.read_text())
    # Sanity: the fixture and the Python contract must agree on the
    # thresholds — if they drift, the parity claim is meaningless.
    assert fx["policy_version"] == fr.quant_policy_version
    assert fx["thresholds"]["min_directional_prob"] == pytest.approx(fr.min_directional_prob)
    assert fx["thresholds"]["min_directional_edge"] == pytest.approx(fr.min_directional_edge)
    assert fx["thresholds"]["min_expected_return_pct"] == pytest.approx(fr.min_expected_return_pct)
    for case in fx["cases"]:
        side, conf, reason = decide_direction(
            case["p_down"], case["p_stable"], case["p_up"],
            case["expected_return_pct"], fr=fr,
        )
        assert side == case["expected_side"], f"side mismatch for {case['label']}"
        assert conf == pytest.approx(case["expected_confidence"]), \
            f"confidence mismatch for {case['label']}"
        assert reason == case["expected_reason"], f"reason mismatch for {case['label']}"


def test_decide_direction_abstains_with_granular_reason():
    fr = get_frictions()
    # 1) Below MIN_DIRECTIONAL_PROB (max(p_up,p_down) < 0.08).
    s, c, r = decide_direction(0.05, 0.92, 0.03, 1.5, fr=fr)
    assert s is None and r == "abstain_low_directional_prob"
    assert c == pytest.approx(0.92)  # confidence = p_stable on abstain

    # 2) Edge below MIN_DIRECTIONAL_EDGE (|p_up - p_down| < 0.02).
    s, c, r = decide_direction(0.40, 0.20, 0.40, 1.5, fr=fr)
    assert s is None and r == "abstain_no_directional_edge"

    # 3) |expRet| below the EV-vs-cost floor.
    s, c, r = decide_direction(0.10, 0.35, 0.55, 0.05, fr=fr)
    assert s is None and r == "abstain_exp_ret_below_cost"

    # 4) expRet sign disagrees with directional side.
    s, c, r = decide_direction(0.10, 0.35, 0.55, -1.5, fr=fr)
    assert s is None and r == "abstain_exp_ret_disagrees"


# ── Regime classifier ──────────────────────────────────────────────────────
def test_classify_regime_bull_and_bear_and_volatile():
    bull = classify_regime([3.0, 4.0, 2.5, 3.5, 5.0])
    assert bull.regime == "bull" and bull.bullish_ratio > 0.6
    bear = classify_regime([-3.0, -4.0, -2.5, -3.5, -5.0])
    assert bear.regime == "bear"
    vol = classify_regime([10, -10, 8, -8, 12, -12])
    assert vol.regime == "volatile"
    assert classify_regime([]).regime == "sideways"


# ── Metrics ────────────────────────────────────────────────────────────────
def _make_trade(pnl: float, regime="bull", coin="x") -> TradeRow:
    return TradeRow(coin_id=coin, timeframe="5m", direction="up",
                    entry_ts_ms=0, exit_ts_ms=1, entry_price=100.0,
                    exit_price=100.0 + pnl / 10, position_size_usd=100.0,
                    pnl_usd=pnl, pnl_pct=pnl, exit_reason="tp",
                    regime_at_entry=regime, confidence=0.7)


def test_metrics_basic():
    trades = [_make_trade(10), _make_trade(-5), _make_trade(20), _make_trade(-2)]
    m = compute_metrics(trades, 1000.0)
    assert m.n_trades == 4
    assert m.win_rate == 0.5
    assert m.expectancy_usd == pytest.approx((10 - 5 + 20 - 2) / 4)
    assert m.profit_factor == pytest.approx((10 + 20) / (5 + 2))
    assert m.final_pnl_usd == pytest.approx(23)


def test_metrics_empty_safe():
    m = compute_metrics([], 1000.0)
    assert m.n_trades == 0
    assert m.expectancy_usd == 0.0


def test_max_drawdown():
    assert max_drawdown_pct([1000, 1200, 900, 1100]) == pytest.approx(25.0)


def test_metrics_emit_time_in_market_and_avg_hold():
    # Three non-overlapping trades over 30s span: trade durations 1+2+3 = 6s,
    # union span = 30s ⇒ time_in_market_pct = 20%.
    trades = [
        TradeRow("a", "5m", "up", 0,      1_000,  100, 100, 100, 1, 1, "tp", "bull", 0.7),
        TradeRow("a", "5m", "up", 5_000,  7_000,  100, 100, 100, 1, 1, "tp", "bull", 0.7),
        TradeRow("a", "5m", "up", 20_000, 23_000, 100, 100, 100, 1, 1, "tp", "bull", 0.7),
    ]
    m = compute_metrics(trades, 1000.0)
    assert m.time_in_market_pct == pytest.approx(6_000 / 23_000 * 100, rel=1e-9)
    assert m.avg_hold_ms == pytest.approx((1_000 + 2_000 + 3_000) / 3)


def test_metrics_time_in_market_handles_overlap():
    # Two overlapping trades: [0,5000] ∪ [3000,8000] = 8000ms over 8000ms span = 100%.
    trades = [
        TradeRow("a", "5m", "up", 0,    5_000, 100, 100, 100, 1, 1, "tp", "bull", 0.7),
        TradeRow("b", "5m", "up", 3_000, 8_000, 100, 100, 100, 1, 1, "tp", "bull", 0.7),
    ]
    m = compute_metrics(trades, 1000.0)
    assert m.time_in_market_pct == pytest.approx(100.0)


# ── Monte Carlo ────────────────────────────────────────────────────────────
def test_monte_carlo_deterministic():
    trades = [_make_trade(10), _make_trade(-5), _make_trade(20), _make_trade(-3),
              _make_trade(15), _make_trade(-7)]
    a = run_monte_carlo(trades, 1000.0, n_runs=200, seed=42)
    b = run_monte_carlo(trades, 1000.0, n_runs=200, seed=42)
    assert a.equity_band_p50 == b.equity_band_p50
    # Total final PnL preserved across permutations (sum of pnls).
    assert a.equity_band_p50[-1] == pytest.approx(1000 + sum(t.pnl_usd for t in trades))
    assert a.equity_band_p05[-1] == a.equity_band_p50[-1] == a.equity_band_p95[-1]


# ── Regime breakdown ───────────────────────────────────────────────────────
def test_regime_breakdown_buckets_correctly():
    trades = [_make_trade(10, "bull"), _make_trade(-5, "bear"),
              _make_trade(20, "bull"), _make_trade(2, "sideways")]
    rb = regime_breakdown(trades, 1000.0)
    assert rb["bull"]["n_trades"] == 2
    assert rb["bear"]["n_trades"] == 1
    assert rb["sideways"]["n_trades"] == 1
    assert rb["volatile"]["n_trades"] == 0


# ── Decision gate ──────────────────────────────────────────────────────────
def test_decision_pass_path():
    fr = get_frictions()
    overall = {"n_trades": 100, "expectancy_usd": 1.0, "sharpe_per_trade": 0.8}
    rb = {"bull":     {"n_trades": 60, "expectancy_usd": 1.5},
          "bear":     {"n_trades": 0,  "expectancy_usd": 0.0},
          "sideways": {"n_trades": 0,  "expectancy_usd": 0.0},
          "volatile": {"n_trades": 0,  "expectancy_usd": 0.0}}
    v = decide(overall, rb, fr=fr, mc_p05_drawdown_pct=10.0)
    assert v.deploy is True
    assert "bull" in v.passing_regimes


def test_decision_fail_paths():
    fr = get_frictions()
    rb = {r: {"n_trades": 0, "expectancy_usd": 0.0}
          for r in ("bull", "bear", "sideways", "volatile")}
    # Fail on every gate at once.
    overall = {"n_trades": 10, "expectancy_usd": -1.0, "sharpe_per_trade": -0.2}
    v = decide(overall, rb, fr=fr, mc_p05_drawdown_pct=80.0)
    assert v.deploy is False
    joined = " | ".join(v.reasons)
    assert "insufficient trades" in joined
    assert "expectancy_usd" in joined
    assert "sharpe_per_trade" in joined
    assert "MC p05 drawdown" in joined
    assert "no regime passes" in joined


# ── Simulator end-to-end (no model needed) ─────────────────────────────────
def _sideways_lookup(_ts):
    return classify_regime([0.1, -0.1, 0.05, -0.05])


def _bullish_oos_row(ts, coin, p_up=0.8, atr=0.5, atr_pct=0.5,
                     expected_return_pct=1.0):
    # expected_return_pct defaults well above the live brain's |expRet|
    # floor (calibrated 2026-04-22 to factor=0.5 → ~0.15% on the v3 policy)
    # so the directional gate emits a signal in the same way live inference
    # would when the regression head agrees with probUp.
    return {"timestamp_ms": ts, "coin_id": coin,
            "entry_price": 100.0, "atr14": atr, "atrPct": atr_pct,
            "true_label": 2, "p_down": 0.05, "p_stable": 0.15, "p_up": p_up,
            "expected_return_pct": expected_return_pct,
            "fold_id": 0}


def test_simulator_take_profit_path():
    fr = get_frictions()
    oos = pd.DataFrame([_bullish_oos_row(1_000, "btc", p_up=0.8, atr=2.0, atr_pct=2.0)])
    # Tape rallies from 100 → 120 over 60s.
    streams = {"btc": CoinTickStream("btc",
        [2_000, 5_000, 10_000, 60_000], [101.0, 105.0, 115.0, 125.0])}
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams=streams,
                   fr=fr, regime_lookup=_sideways_lookup,
                   gate_min_confidence=0.5, gate_min_tp_distance_pct=0.001,
                   gate_min_ev_vs_cost=0.0)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.direction == "up"
    assert t.exit_reason == "tp"
    assert t.pnl_usd > 0


def test_simulator_stop_loss_path():
    fr = get_frictions()
    oos = pd.DataFrame([_bullish_oos_row(1_000, "btc", p_up=0.8, atr=2.0, atr_pct=2.0)])
    # Tape drops sharply.
    streams = {"btc": CoinTickStream("btc",
        [2_000, 5_000, 10_000], [99.0, 95.0, 80.0])}
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams=streams,
                   fr=fr, regime_lookup=_sideways_lookup,
                   gate_min_confidence=0.5, gate_min_tp_distance_pct=0.001,
                   gate_min_ev_vs_cost=0.0)
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "sl"
    assert res.trades[0].pnl_usd < 0


def test_simulator_abstain_when_no_directional_edge():
    fr = get_frictions()
    # p_up == p_down → dir_edge = 0 → live brain abstains with the
    # granular reason `abstain_no_directional_edge` (NOT the obsolete
    # 3-class-argmax `abstain_stable`).
    oos = pd.DataFrame([{"timestamp_ms": 1_000, "coin_id": "btc",
                         "entry_price": 100.0, "atr14": 1.0, "atrPct": 1.0,
                         "true_label": 1, "p_down": 0.2, "p_stable": 0.6,
                         "p_up": 0.2, "expected_return_pct": 1.5, "fold_id": 0}])
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams={"btc":
        CoinTickStream("btc", [2_000], [100.0])},
        fr=fr, regime_lookup=_sideways_lookup)
    assert res.trades == []
    assert any(s.reason == "abstain_no_directional_edge" for s in res.skips)


def test_simulator_abstain_when_exp_ret_below_cost():
    fr = get_frictions()
    # Strong directional probs but expRet under the EV-vs-cost floor.
    oos = pd.DataFrame([{"timestamp_ms": 1_000, "coin_id": "btc",
                         "entry_price": 100.0, "atr14": 1.0, "atrPct": 1.0,
                         "true_label": 2, "p_down": 0.10, "p_stable": 0.35,
                         "p_up": 0.55, "expected_return_pct": 0.05, "fold_id": 0}])
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams={"btc":
        CoinTickStream("btc", [2_000], [100.0])},
        fr=fr, regime_lookup=_sideways_lookup)
    assert res.trades == []
    assert any(s.reason == "abstain_exp_ret_below_cost" for s in res.skips)


def test_simulator_abstain_when_exp_ret_disagrees():
    fr = get_frictions()
    # Strong probUp but expRet sign is negative — model heads disagree.
    oos = pd.DataFrame([{"timestamp_ms": 1_000, "coin_id": "btc",
                         "entry_price": 100.0, "atr14": 1.0, "atrPct": 1.0,
                         "true_label": 2, "p_down": 0.10, "p_stable": 0.35,
                         "p_up": 0.55, "expected_return_pct": -1.5, "fold_id": 0}])
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams={"btc":
        CoinTickStream("btc", [2_000], [100.0])},
        fr=fr, regime_lookup=_sideways_lookup)
    assert res.trades == []
    assert any(s.reason == "abstain_exp_ret_disagrees" for s in res.skips)


def test_simulator_low_confidence_skipped():
    fr = get_frictions()
    oos = pd.DataFrame([_bullish_oos_row(1_000, "btc", p_up=0.45)])
    res = simulate(timeframe="5m", oos_predictions=oos,
                   tick_streams={"btc": CoinTickStream("btc", [2_000], [100.0])},
                   fr=fr, regime_lookup=_sideways_lookup)
    assert res.trades == []
    assert any(s.reason == "confidence_below_threshold" for s in res.skips)


def test_simulator_max_open_positions_enforced():
    fr = get_frictions()
    rows = [_bullish_oos_row(1_000 + i*10, f"c{i}", atr=2.0, atr_pct=2.0)
            for i in range(fr.max_open_positions_per_agent + 2)]
    oos = pd.DataFrame(rows)
    streams = {f"c{i}": CoinTickStream(f"c{i}", [10**9], [100.0])
               for i in range(len(rows))}  # never trigger SL/TP
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams=streams,
                   fr=fr, regime_lookup=_sideways_lookup,
                   gate_min_confidence=0.5, gate_min_tp_distance_pct=0.001,
                   gate_min_ev_vs_cost=0.0)
    assert any(s.reason == "max_open_positions" for s in res.skips)


# ── Walk-forward OOS shape (uses small synthetic dataset) ─────────────────
def test_predict_oos_returns_empty_for_tiny_dataset():
    from app.backtest.walk_forward_oos import predict_oos_for_dataset
    df = pd.DataFrame()
    out = predict_oos_for_dataset(df)
    assert out.empty
    assert set(out.columns) >= {"timestamp_ms", "coin_id", "entry_price",
                                "p_down", "p_stable", "p_up"}


# ── Parity fixes for architect review ─────────────────────────────────────
def test_simulator_pnl_no_double_slippage_or_fee():
    """Live accounting: entry slippage + entry fee paid at OPEN; close path
    only applies exit slippage + exit fee. Backtester must match."""
    fr = get_frictions()
    # Walk a clean tape: open, then sit at the same raw price; expiry should
    # produce ~ -2*slippage*size - exit_fee, NOT 4x that.
    oos = pd.DataFrame([_bullish_oos_row(1_000, "btc", p_up=0.8,
                                          atr=2.0, atr_pct=2.0)])
    streams = {"btc": CoinTickStream("btc",
        [2_000, 5_000, 60_000_000_000], [100.0, 100.0, 100.0])}
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams=streams,
                   fr=fr, regime_lookup=_sideways_lookup,
                   gate_min_confidence=0.5, gate_min_tp_distance_pct=0.001,
                   gate_min_ev_vs_cost=0.0)
    assert len(res.trades) == 1
    t = res.trades[0]
    # raw exit = 100, adj_entry = 100*(1+slippage), adj_exit = 100*(1-slippage)
    # gross = (adj_exit - adj_entry) * (size/adj_entry)
    adj_entry = 100.0 * (1 + fr.slippage_pct)
    adj_exit  = 100.0 * (1 - fr.slippage_pct)
    qty = t.position_size_usd / adj_entry
    expected_gross = (adj_exit - adj_entry) * qty
    expected_exit_fee = max(0.0, t.position_size_usd + expected_gross) * fr.maker_fee_pct
    expected_pnl = expected_gross - expected_exit_fee
    assert t.pnl_usd == pytest.approx(expected_pnl, rel=1e-9)


def test_simulator_consecutive_loss_block():
    fr = get_frictions()
    n = int(fr.raw["recent_loss_block"]["consecutive_losses"])
    # Submit n losing trades on the same coin then one more — the (n+1)th
    # decision must be skipped with reason `consecutive_losses`.
    # Space decisions across UTC days so the daily-loss limit doesn't fire
    # before the consecutive-loss block can be observed.
    one_day = 24 * 60 * 60 * 1000
    decision_ts = [(i + 1) * one_day for i in range(n + 1)]
    rows = [_bullish_oos_row(t, "btc", p_up=0.8, atr=2.0, atr_pct=2.0)
            for t in decision_ts]
    oos = pd.DataFrame(rows)
    # Tape: brief SL hit shortly after each decision.
    tape_ts = sorted([t + d for t in decision_ts for d in (10, 50, 80)])
    # adj_entry ≈ 100.05; SL distance = atr*sl_mult = 2.0*1.8 = 3.6, so SL ≈ 96.45.
    # Tape at 90 triggers SL every time (still well above the 0.33 anomaly floor).
    streams = {"btc": CoinTickStream("btc", tape_ts,
                                     [90.0] * len(tape_ts))}
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams=streams,
                   fr=fr, regime_lookup=_sideways_lookup,
                   gate_min_confidence=0.5, gate_min_tp_distance_pct=0.001,
                   gate_min_ev_vs_cost=0.0)
    assert len(res.trades) == n  # not n+1
    assert any(s.reason == "consecutive_losses" for s in res.skips)


def test_simulator_anomaly_cancel_path():
    """Exit price > 3× entry → no PnL booked, full position size refunded."""
    fr = get_frictions()
    oos = pd.DataFrame([_bullish_oos_row(1_000, "btc", p_up=0.8, atr=2.0, atr_pct=2.0)])
    # Tape that explodes 5x — anomaly guard cancels.
    streams = {"btc": CoinTickStream("btc", [2_000, 60_000_000_000], [500.0, 500.0])}
    res = simulate(timeframe="5m", oos_predictions=oos, tick_streams=streams,
                   fr=fr, regime_lookup=_sideways_lookup,
                   gate_min_confidence=0.5, gate_min_tp_distance_pct=0.001,
                   gate_min_ev_vs_cost=0.0)
    assert res.trades == []  # no trade booked
    # Anomaly cancel must reverse the FULL open-time cash impact
    # (position_size + entry_fee) — equity equals the pre-trade value
    # exactly (within float epsilon), no leaked entry fee.
    assert res.final_equity == pytest.approx(res.initial_equity, rel=1e-9)


def test_run_backtest_is_deterministic(tmp_path, monkeypatch):
    """Same inputs → identical report bytes (no wall-clock)."""
    from app.backtest import run as run_module
    monkeypatch.setattr(run_module, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(run_module, "REPORT_JSON", tmp_path / "r.json")
    monkeypatch.setattr(run_module, "REPORT_HTML", tmp_path / "r.html")
    run_module.run_backtest()
    bytes_a = (tmp_path / "r.json").read_bytes()
    run_module.run_backtest()
    bytes_b = (tmp_path / "r.json").read_bytes()
    assert bytes_a == bytes_b
    # And input fingerprint is stable.
    assert json.loads(bytes_a)["input_fingerprint"] == json.loads(bytes_b)["input_fingerprint"]


# ── End-to-end smoke (no LightGBM run) ────────────────────────────────────
def test_run_backtest_without_snapshots(tmp_path, monkeypatch):
    """If no parquet snapshots exist for any tradeable timeframe, the
    runner must produce a valid report with all runs marked
    `no_dataset` and overall deploy=False — never crash, never fabricate.
    """
    from app.backtest import run as run_module
    monkeypatch.setattr(run_module, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(run_module, "REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr(run_module, "REPORT_HTML", tmp_path / "report.html")
    report = run_module.run_backtest()
    assert report["summary"]["deploy"] is False
    assert all(r["status"] == "no_dataset" for r in report["runs"])
    # JSON + HTML written.
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()
    # JSON must be loadable.
    json.loads((tmp_path / "report.json").read_text())
