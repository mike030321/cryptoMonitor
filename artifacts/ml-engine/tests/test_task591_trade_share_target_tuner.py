"""Task #591 — regression tests for the per-(coin, timeframe)
auto-tuner of `TRADE_SHARE_TARGET` in the feature-edge search runner.

Pin three invariants:

1. `_select_trade_share_target` picks the grid target whose fitted
   `delta` maximises cal-tail post-fee PnL on the same data.
2. The tuner's tie-break rule (smallest target wins) fires on
   degenerate-tie inputs.
3. `_cal_tail_pnl_at_delta`'s trade mask and signed-return accounting
   match `score_fold`'s on the same inputs, so the tuner cannot
   silently drift from the fold scoring economics the verdict reports.
"""

from __future__ import annotations

import os

os.environ.setdefault("ML_FEATURE_EDGE_CALIBRATE", "0")

import numpy as np
import pandas as pd
import pytest

from scripts.feature_edge_search import run_search as rs


def _synthetic(n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    p_up = rng.uniform(0.05, 0.95, n)
    p_dn = (1.0 - p_up) * rng.uniform(0.0, 1.0, n)
    p_neu = 1.0 - p_up - p_dn
    proba = np.stack([p_dn, p_neu, p_up], axis=1).clip(min=1e-6)
    proba /= proba.sum(axis=1, keepdims=True)
    margin_signed = proba[:, 2] - proba[:, 0]
    fwd = 0.01 * margin_signed + rng.normal(0, 0.005, n)
    cal_df = pd.DataFrame({"forward_return": fwd})
    return proba, fwd, cal_df


def test_select_trade_share_target_picks_max_pnl():
    proba, _, cal_df = _synthetic()
    rt_cost = 0.0010

    best_target, best_delta, best_pnl, sweep = rs._select_trade_share_target(
        cal_proba=proba, cal_df=cal_df, round_trip_cost_frac=rt_cost,
    )

    assert sweep, "sweep must contain one record per grid target"
    assert {r["trade_share_target"] for r in sweep} == set(rs.TRADE_SHARE_TARGET_GRID)

    truly_best = max(
        sweep,
        key=lambda r: (r["cal_tail_pnl_pct_total"], -r["trade_share_target"]),
    )
    assert best_target == truly_best["trade_share_target"]
    assert best_delta == truly_best["delta_fitted"]
    assert best_pnl == truly_best["cal_tail_pnl_pct_total"]


def test_select_trade_share_target_tiebreak_smallest_wins():
    """Degenerate input: zero forward returns and zero cost mean every
    grid target produces the same cal-tail PnL (zero). The tie-break
    must pick the SMALLEST target (largest neutral-band delta)."""
    n = 200
    proba = np.tile(np.array([0.30, 0.40, 0.30]), (n, 1))
    cal_df = pd.DataFrame({"forward_return": np.zeros(n)})

    best_target, _best_delta, _best_pnl, sweep = rs._select_trade_share_target(
        cal_proba=proba, cal_df=cal_df, round_trip_cost_frac=0.0,
    )

    pnls = {r["trade_share_target"]: r["cal_tail_pnl_pct_total"] for r in sweep}
    assert len(set(pnls.values())) == 1, (
        f"expected all-tie PnLs on degenerate input, got {pnls}"
    )
    assert best_target == min(rs.TRADE_SHARE_TARGET_GRID), (
        f"tie-break must pick the smallest target, got {best_target}"
    )


def test_cal_tail_pnl_matches_score_fold_economics():
    """`_cal_tail_pnl_at_delta` must use the same trade mask and signed
    return convention as `score_fold` so the tuner optimises what the
    verdict reports. We hand-compute the expected PnL using exactly the
    convention from `score_fold` and compare to the helper output."""
    proba, fwd_cal, cal_df = _synthetic(n=300, seed=11)
    rt_cost = 0.00075

    p_dn = proba[:, 0]
    p_st = proba[:, 1]
    p_up = proba[:, 2]
    margin = np.maximum(p_up, p_dn) - p_st
    delta = float(np.quantile(margin, 0.4))

    pnl, n_trades = rs._cal_tail_pnl_at_delta(
        cal_proba=proba, fwd_cal=fwd_cal,
        delta=delta, round_trip_cost_frac=rt_cost,
    )

    trade_mask = margin > delta if delta > 0.0 else margin > 0.0
    direction = np.where(p_up >= p_dn, 2, 0)
    signs = np.where(direction[trade_mask] == 2, 1.0, -1.0)
    signed_ret_pct = signs * fwd_cal[trade_mask] * 100.0
    expected = float(
        np.sum(signed_ret_pct - (rt_cost * 100.0))
    )

    assert n_trades == int(trade_mask.sum())
    assert pnl == pytest.approx(expected, abs=1e-9)


def test_cal_tail_pnl_zero_trades_returns_zero():
    """Edge case: when no row's margin clears delta, helper must
    return (0.0, 0) without dividing or indexing into empty arrays."""
    n = 50
    proba = np.tile(np.array([0.33, 0.34, 0.33]), (n, 1))
    fwd = np.full(n, 0.05)

    pnl, n_trades = rs._cal_tail_pnl_at_delta(
        cal_proba=proba, fwd_cal=fwd,
        delta=0.50, round_trip_cost_frac=0.001,
    )
    assert n_trades == 0
    assert pnl == 0.0
