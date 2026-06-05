"""Label producers for Task #643.

Three NEW objective families plus the existing 3-class label kept as a
direct comparison anchor. Each producer takes a numpy/pandas array of
forward returns expressed as **fractions** (NOT percent — caller
converts before calling) and returns the label vector of the same
length plus optional per-family metadata.

Family conventions (consistent across A/B/C):

* ``-1`` = short candidate
* ``+1`` = long candidate
* ``0``  = no-trade / abstain  (baseline 3-class also emits 0 = STABLE)

The 3-class baseline returns ``0`` for STABLE, ``+1`` for UP, ``-1``
for DOWN, matching the existing ``label_three_class`` semantics so the
metric tabulator can treat all four families uniformly when computing
n_trades / precision / etc.

For Family A (quintiles) the *sign-encoded* output is the natural
trade-side projection: Q5 → +1 (strongest long), Q1 → -1 (strongest
short), Q2..Q4 → 0 (no-trade). The trainer keeps the underlying 5-class
target separately so a multinomial-logistic head can be fit.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Friction sourcing — read once, cached. No edits to the JSON ever.
# ---------------------------------------------------------------------------

_FRICTIONS_PATH = (
    Path(__file__).resolve().parents[5] / "shared" / "trading-frictions.json"
)
_FRICTIONS_CACHE: Optional[dict] = None


def _load_frictions() -> dict:
    global _FRICTIONS_CACHE
    if _FRICTIONS_CACHE is None:
        with open(_FRICTIONS_PATH, "r", encoding="utf-8") as f:
            _FRICTIONS_CACHE = json.load(f)
    return _FRICTIONS_CACHE


def round_trip_cost_fraction() -> float:
    """Sum of entry slip + entry fee + exit slip + exit fee, expressed
    as a fraction (NOT percent). With current frictions this equals
    ``2 * 0.001 + 2 * 0.0005 = 0.003`` (= 0.30 %). Read from
    ``shared/trading-frictions.json`` so any future cost change is
    automatically picked up — no edits to that file from this task.
    """
    fees = _load_frictions()["fees"]
    return float(
        2.0 * fees["taker_fee_pct"] + 2.0 * fees["slippage_pct"]
    )


# Safety margin above round-trip cost for Family C — fixed by the spec
# at 0.10 % so the post-cost label fires only when the forward return
# clears cost AND a non-trivial buffer.
POST_COST_SAFETY_MARGIN_FRACTION: float = 0.001  # 0.10 %


# Forward horizon (in BARS) per timeframe. Same horizon across all
# three label families so their PnL numbers are comparable. Baseline
# 3-class `directional_label_forward_return` already runs on the
# trader-aware horizon — we override with this for A/B/C so all four
# families predict the *same* event.
HORIZON_BARS_PER_TF: dict[str, int] = {
    # 60 bars × 1m = 1h forward
    "1m": 60,
    # 12 bars × 5m = 1h forward
    "5m": 12,
}


def horizon_bars(timeframe: str) -> int:
    if timeframe not in HORIZON_BARS_PER_TF:
        raise ValueError(
            f"Task #643 horizon not configured for timeframe {timeframe!r}; "
            f"supported: {sorted(HORIZON_BARS_PER_TF)}"
        )
    return HORIZON_BARS_PER_TF[timeframe]


# ---------------------------------------------------------------------------
# Forward-return computation
# ---------------------------------------------------------------------------


def compute_forward_returns(
    closes: Sequence[float], horizon: int,
) -> np.ndarray:
    """Forward log/arith return over ``horizon`` bars, expressed as a
    fraction (e.g. 0.01 == 1 %). Uses arithmetic ``(c[t+h] - c[t]) /
    c[t]`` to match the existing labels.py convention so PnL numbers
    are directly comparable. Pads the final ``horizon`` rows with NaN
    so the output is the same length as ``closes``.
    """
    arr = np.asarray(closes, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if n <= horizon:
        return out
    out[: n - horizon] = (arr[horizon:] - arr[: n - horizon]) / arr[: n - horizon]
    return out


# ---------------------------------------------------------------------------
# Family A — quintile labels
# ---------------------------------------------------------------------------


@dataclass
class QuintileLabels:
    """Output of ``label_quintile``.

    * ``quintile`` — integers 1..5 (1 = strongest short, 5 = strongest
      long, 2..4 = no-trade band). NaN where the source forward-return
      is NaN (last ``horizon`` rows or training-window outliers).
    * ``trade_side`` — sign-encoded projection (+1 / -1 / 0) used by
      the unified metric tabulator.
    * ``edges`` — the four cut points actually applied. Stamped onto
      the model report so downstream verification can replay the
      quantile rule on a fresh holdout.
    """

    quintile: np.ndarray
    trade_side: np.ndarray
    edges: tuple[float, float, float, float]


def label_quintile(
    forward_ret: Sequence[float],
    *,
    train_mask: Optional[np.ndarray] = None,
) -> QuintileLabels:
    """Bucket forward returns into quintile bins. Quintile boundaries
    are computed on the TRAIN subset only when ``train_mask`` is
    supplied (no test-set leakage). When ``train_mask`` is ``None``
    every non-NaN row participates in the boundary computation — call
    sites that care about leakage MUST pass the mask.

    Q5 (top quintile) → trade_side = +1
    Q1 (bottom quintile) → trade_side = -1
    Q2..Q4 → trade_side = 0
    """
    arr = np.asarray(forward_ret, dtype=float)
    n = len(arr)
    out_q = np.full(n, np.nan, dtype=float)
    out_side = np.zeros(n, dtype=float)

    finite = np.isfinite(arr)
    if train_mask is not None:
        train_finite = finite & train_mask
    else:
        train_finite = finite
    train_vals = arr[train_finite]
    if len(train_vals) < 25:
        # Not enough training data to cut into quintiles; everything is
        # NaN / no-trade so the metric tabulator surfaces the slice as
        # "insufficient training data" rather than producing garbage
        # boundaries.
        edges = (float("nan"),) * 4
        return QuintileLabels(quintile=out_q, trade_side=out_side, edges=edges)

    qs = np.quantile(train_vals, [0.2, 0.4, 0.6, 0.8])
    e1, e2, e3, e4 = (float(qs[0]), float(qs[1]), float(qs[2]), float(qs[3]))

    # Bucket assignment: <e1 → Q1, [e1,e2) → Q2, [e2,e3) → Q3,
    # [e3,e4) → Q4, >=e4 → Q5. NaNs stay NaN.
    out_q = np.where(finite, np.full(n, np.nan), np.nan)
    sub = arr.copy()
    out_q[finite & (sub < e1)] = 1.0
    out_q[finite & (sub >= e1) & (sub < e2)] = 2.0
    out_q[finite & (sub >= e2) & (sub < e3)] = 3.0
    out_q[finite & (sub >= e3) & (sub < e4)] = 4.0
    out_q[finite & (sub >= e4)] = 5.0

    out_side[out_q == 5.0] = 1.0
    out_side[out_q == 1.0] = -1.0
    return QuintileLabels(
        quintile=out_q, trade_side=out_side, edges=(e1, e2, e3, e4),
    )


# ---------------------------------------------------------------------------
# Family B — sparse top-decile of |forward return|
# ---------------------------------------------------------------------------


@dataclass
class SparseLabels:
    """Output of ``label_sparse_top_decile``.

    * ``trade_side`` — +1 long-candidate, -1 short-candidate,
      0 no-trade.
    * ``threshold_abs`` — the ``|fwd_ret|`` cutoff applied (90th
      percentile of |fwd_ret| on the training subset).
    """

    trade_side: np.ndarray
    threshold_abs: float


def label_sparse_top_decile(
    forward_ret: Sequence[float],
    *,
    train_mask: Optional[np.ndarray] = None,
    decile_pct: float = 0.90,
) -> SparseLabels:
    """Label only the rows whose ``|forward_ret|`` exceeds the
    ``decile_pct``-quantile (default 90th percentile = top decile) of
    |fwd_ret| on the training subset. Sign of the forward return
    determines long vs short; everything else is no-trade.
    """
    arr = np.asarray(forward_ret, dtype=float)
    n = len(arr)
    out = np.zeros(n, dtype=float)
    finite = np.isfinite(arr)
    if train_mask is not None:
        train_finite = finite & train_mask
    else:
        train_finite = finite
    train_abs = np.abs(arr[train_finite])
    if len(train_abs) < 25:
        return SparseLabels(trade_side=out, threshold_abs=float("nan"))
    threshold = float(np.quantile(train_abs, decile_pct))
    abs_arr = np.abs(arr)
    pick = finite & (abs_arr >= threshold)
    out[pick & (arr > 0)] = 1.0
    out[pick & (arr < 0)] = -1.0
    return SparseLabels(trade_side=out, threshold_abs=threshold)


# ---------------------------------------------------------------------------
# Family C — post-cost opportunity labels
# ---------------------------------------------------------------------------


@dataclass
class PostCostLabels:
    """Output of ``label_post_cost``.

    * ``trade_side`` — +1 long-candidate (fwd_ret > +cost+margin),
      -1 short-candidate (fwd_ret < -(cost+margin)), 0 no-trade.
    * ``threshold_fraction`` — the absolute return floor applied
      (round_trip_cost + safety_margin).
    """

    trade_side: np.ndarray
    threshold_fraction: float


def label_post_cost(
    forward_ret: Sequence[float],
    *,
    margin_fraction: float = POST_COST_SAFETY_MARGIN_FRACTION,
) -> PostCostLabels:
    """Label rows whose forward return clears
    ``round_trip_cost + margin`` in absolute value. No training-set
    leakage concerns because the threshold is read from the immutable
    frictions JSON, not estimated from the data.
    """
    arr = np.asarray(forward_ret, dtype=float)
    n = len(arr)
    out = np.zeros(n, dtype=float)
    threshold = round_trip_cost_fraction() + float(margin_fraction)
    finite = np.isfinite(arr)
    out[finite & (arr > threshold)] = 1.0
    out[finite & (arr < -threshold)] = -1.0
    return PostCostLabels(trade_side=out, threshold_fraction=threshold)


# ---------------------------------------------------------------------------
# Baseline 3-class — exact mirror of ``labels.label_three_class`` so the
# tabulator can treat all four families uniformly. Threshold sourced
# the SAME way the production trainer sources it (per-coin override →
# timeframe default) via ``threshold_for``.
# ---------------------------------------------------------------------------


@dataclass
class ThreeClassLabels:
    """Output of ``label_three_class_baseline``.

    * ``trade_side`` — +1 UP, -1 DOWN, 0 STABLE — matching the
      existing 3-class production semantics.
    * ``threshold_fraction`` — the band actually applied (in fraction).
    """

    trade_side: np.ndarray
    threshold_fraction: float


def label_three_class_baseline(
    forward_ret: Sequence[float], threshold_fraction: float,
) -> ThreeClassLabels:
    """3-class wrapper. Exact policy mirror of ``labels.label_three_class``
    but operating on **fractions** (the producer convention), not
    percent."""
    arr = np.asarray(forward_ret, dtype=float)
    n = len(arr)
    out = np.zeros(n, dtype=float)
    finite = np.isfinite(arr)
    out[finite & (arr > threshold_fraction)] = 1.0
    out[finite & (arr < -threshold_fraction)] = -1.0
    return ThreeClassLabels(
        trade_side=out, threshold_fraction=float(threshold_fraction),
    )
