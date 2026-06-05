"""Task #655 — paper-trading B "truth gate": persist + Platt-calibrate
dual-binary-head family-C models, then verify on a 14-day forward
holdout.

For each (coin, timeframe) candidate (default: bitcoin@5m, ethereum@5m):

  1. Build the same 380-day research frame round-5 of #643 used.
  2. Carve the most-recent 14 calendar days off as a forward holdout
     (training data is everything strictly before the holdout window).
  3. Reproducibility check (best-effort): compute the validation
     net_pnl_pct_total via a single chronological 80/20 of the
     training subset and compare to round-5's published number.
     Divergence is logged but does NOT auto-abort (data drift can
     legitimately shift the metric — the truthful test is the forward
     holdout, not the in-sample re-fit).
  4. Single-fit dual-binary heads on training_data:
       train_inner (first 80% chronologically) → fit long_head + short_head
       val (last 20%)                          → raw probs for Platt
  5. Fit Platt sigmoid per head on val raw probs. Apply to val.
  6. τ = (1 − base_rate_train_inner) quantile of val
     max(p_long_calibrated, p_short_calibrated).
  7. Persist long_model.txt, short_model.txt, manifest.json,
     feature_list.json, validation_metrics.json, calibration.json
     under  models/<coin>/<tf>/C_post_cost/<run-id>/
  8. Score the 14-day holdout by reloading the persisted artefacts
     from disk (no in-memory shortcuts) and computing n_trades /
     net_pnl_pct_total / profit_factor / cal_dev_post_calibration.
  9. PASS iff  n_trades >= 5  AND  net_pnl_pct_total > 0  AND
                profit_factor >= 1.0  AND  cal_dev_post_calibration <= 0.20

A summary report is written to
  artifacts/ml-engine/reports/task-B-truth-gate-<ts>.md

When BOTH candidates fail the truth gate, the report's headline is
the spec-mandated string  "Current app did not produce a trustworthy
quant trading loop under tested designs."

Hard rules honoured:
  * No edits to shared/trading-frictions.json.
  * No threshold relaxation (post-cost margin = 0.10% as in producers.py).
  * No holdout-window swapping — last 14 calendar days of price_candles.
  * No champion promotion — Task C handles that downstream.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import lightgbm as lgb

from . import producers
from .data import build_research_frame
from .runner import _select_feature_columns, _lgb_train
from ..registry import ModelManifest

logger = logging.getLogger("labels_research.persist_truth_gate")

ML_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = ML_ROOT / "models"
REPORTS_DIR = ML_ROOT / "reports"

TRUTH_GATE_FAIL_HEADLINE = (
    "Current app did not produce a trustworthy quant trading loop "
    "under tested designs."
)

# Per (coin, tf), the round-5 family-C net_pnl_pct_total and n_trades
# used for the reproducibility delta in the report. Sourced from
# `reports/20260430T101555Z-quintile-sparse-label-verdict.json`.
ROUND5_FAMILY_C: dict[tuple[str, str], dict] = {
    ("bitcoin", "5m"): {
        "net_pnl_pct_total": 1956.6136542078225,
        "n_trades": 17034,
        "precision": 0.7919455207232594,
        "calibration_max_dev": 0.4740343258258417,
    },
    ("ethereum", "5m"): {
        "net_pnl_pct_total": 5335.876911890427,
        "n_trades": 25483,
        "precision": 0.7948828630851941,
        "calibration_max_dev": 0.37937036713512895,
    },
}

# Truth-gate acceptance criteria — fixed by the task spec.
# Step-1 (post-Platt validation) gate:
#   cal_dev_post_calibration on val <= 0.15  AND
#   reproducibility delta vs round-5 net_pnl_pct_total within ±5%
# Step-2 (forward holdout) gate:
#   n_trades >= 5  AND  net_pnl_pct_total > 0  AND
#   profit_factor >= 1.0  AND  cal_dev_post_calibration <= 0.20
# Both must hold for a PASS.
GATE_MIN_TRADES = 5
GATE_MIN_NET_PNL_PCT = 0.0          # net_pnl_pct_total > 0
GATE_MIN_PROFIT_FACTOR = 1.0        # >= 1.0
GATE_MAX_CAL_DEV_HOLDOUT = 0.20     # cal_dev_post_calibration on holdout
GATE_MAX_CAL_DEV_VAL = 0.15         # cal_dev_post_calibration on val
GATE_REPRO_REL_TOL = 0.05           # |delta| / round5 <= 5% on net_pnl_pct_total

HOLDOUT_DAYS = 14

# Per-timeframe bar duration in ms — used by the strict leakage gate
# (`max(train_ts) + tf_ms < min(holdout_ts)` from the spec).
_TF_TO_MS: dict[str, int] = {"1m": 60_000, "5m": 5 * 60_000}


# ---------------------------------------------------------------------------
# Platt sigmoid fitting
# ---------------------------------------------------------------------------


def _fit_platt(
    raw_probs: np.ndarray, y: np.ndarray,
) -> tuple[float, float]:
    """Fit Platt sigmoid:  P_calibrated = 1 / (1 + exp(slope*raw + intercept)).

    Matches the `LoadedDualHeadModel._platt` convention in
    `app.training.registry`. Internally uses sklearn LogisticRegression
    (which fits  P = sigmoid(w*x + b)) and converts:
        slope     = -w
        intercept = -b

    Edge cases:
      * fewer than 5 of either class in `y` → identity calibration
        (slope = -1, intercept = 0  ⇒  P_cal = sigmoid(raw) ≈ raw for
        raw in [0,1] ranges, but kept as the explicit Platt form so the
        serving path doesn't have to special-case missing calibration).
        With raw probabilities ∈ [0,1] the sigmoid bend over that
        domain is mild; for an unusable head we ALSO set its abstain
        contribution to all-zero, so the slope/intercept never actually
        fires at inference.
    """
    raw = np.asarray(raw_probs, dtype=float)
    y = np.asarray(y, dtype=int)
    finite = np.isfinite(raw)
    raw = raw[finite]
    y = y[finite]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 5 or n_neg < 5:
        # Degenerate — caller should also zero-out this head's prob
        # contribution so τ doesn't get pushed by garbage.
        return -1.0, 0.0
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(
        C=1e10, fit_intercept=True, solver="lbfgs", max_iter=1000,
    )
    lr.fit(raw.reshape(-1, 1), y)
    w = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return -w, -b


def _apply_platt(
    raw_probs: np.ndarray, slope: float, intercept: float,
) -> np.ndarray:
    """Vectorised Platt sigmoid matching the serving convention:
    P_cal = 1 / (1 + exp(slope*raw + intercept))."""
    raw = np.asarray(raw_probs, dtype=float)
    z = slope * raw + intercept
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


# ---------------------------------------------------------------------------
# Isotonic recalibration helpers — Task #657 (paper-trading B2)
# ---------------------------------------------------------------------------


def _fit_isotonic(
    raw_probs: np.ndarray, y: np.ndarray,
) -> dict:
    """Fit `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip",
    y_min=0.0, y_max=1.0)` and return the JSON-serialisable shape the
    `LoadedDualHeadModel._apply_isotonic` serving path consumes.

    The returned dict has the form
        {"x_thresholds": [...float], "y_values": [...float]}
    sourced verbatim from the fitted estimator's `X_thresholds_` and
    `y_thresholds_` arrays. `numpy.interp(raw, x, y)` then reproduces
    `iso.transform(raw)` exactly because `out_of_bounds="clip"` matches
    `np.interp`'s endpoint-clipping behaviour.

    Edge cases:
      * fewer than 5 of either class in `y` → identity calibration with
        a 2-knot grid mapping `[0, 1] → [0, 1]`. The downstream
        `_compute_metrics_post_calibration` already zeros-out a
        degenerate head's contribution via the abstain rule, so the
        identity here just satisfies the manifest's "len >= 2"
        contract without firing.
    """
    raw = np.asarray(raw_probs, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    finite = np.isfinite(raw)
    raw = raw[finite]
    y_arr = y_arr[finite]
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos < 5 or n_neg < 5 or len(raw) < 10:
        return {
            "x_thresholds": [0.0, 1.0],
            "y_values": [0.0, 1.0],
        }
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(
        out_of_bounds="clip", y_min=0.0, y_max=1.0,
    )
    iso.fit(raw.astype(float), y_arr.astype(float))
    return {
        "x_thresholds": [float(v) for v in iso.X_thresholds_.tolist()],
        "y_values":     [float(v) for v in iso.y_thresholds_.tolist()],
    }


def _apply_isotonic_array(
    raw_probs: np.ndarray, x_thresholds: list, y_values: list,
) -> np.ndarray:
    """Vectorised serving form of `_apply_isotonic` for the holdout
    scorer. Identical behaviour to
    `IsotonicRegression(out_of_bounds="clip").transform(raw_probs)`:
    piecewise-linear interpolation between knot points with clipping
    at the boundary y values for raw inputs outside the fitted
    `[x[0], x[-1]]` range.
    """
    raw = np.asarray(raw_probs, dtype=float)
    x = np.asarray(x_thresholds, dtype=float)
    y = np.asarray(y_values, dtype=float)
    return np.clip(np.interp(raw, x, y), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-candidate frame preparation: features + side labels + holdout split
# ---------------------------------------------------------------------------


@dataclass
class CandidateFrame:
    coin: str
    tf: str
    df: pd.DataFrame                   # full frame, sorted by ts ascending
    feature_cols: list[str]
    fwd_full: np.ndarray               # forward returns (NaN-padded tail)
    side_labels: np.ndarray            # post-cost labels {-1,0,+1}
    horizon: int                       # bars
    threshold_fraction: float          # round-trip + margin, in fraction
    train_idx: np.ndarray              # rows in training subset (ts < holdout_start)
    holdout_idx: np.ndarray            # rows in 14-day forward holdout
    holdout_start_ms: int


def _prepare_candidate(frame, holdout_start_ms: int) -> CandidateFrame:
    df = frame.df.copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            f"empty research frame for {frame.coin_id}/{frame.timeframe}"
        )

    horizon = producers.horizon_bars(frame.timeframe)
    fr_1bar = df["forward_return"].astype(float).to_numpy()
    close_implied = np.zeros(len(df), dtype=float)
    close_implied[0] = 1.0
    for i in range(len(df) - 1):
        close_implied[i + 1] = close_implied[i] * (1.0 + fr_1bar[i])

    fwd = producers.compute_forward_returns(close_implied, horizon)
    side = producers.label_post_cost(fwd).trade_side
    threshold_fraction = (
        producers.round_trip_cost_fraction()
        + producers.POST_COST_SAFETY_MARGIN_FRACTION
    )
    feature_cols = _select_feature_columns(df)

    ts_ms = df["timestamp_ms"].astype("int64").to_numpy()
    tf_ms = _TF_TO_MS[frame.timeframe]
    # A training row at ts uses fwd_ret over `horizon` bars ahead, so
    # its label looks at bars in [ts + tf_ms, ts + (horizon+1) * tf_ms].
    # To prevent label leakage into the holdout (and to satisfy the
    # spec's strict `max(train_ts) + tf_ms < min(holdout_ts)` gate),
    # we require the entire forward window to land BEFORE
    # holdout_start_ms — i.e. ts + horizon * tf_ms <= holdout_start_ms
    # — equivalently ts + tf_ms <= holdout_start_ms − (horizon-1)*tf_ms.
    # The holdout itself uses bars whose ts is at or after
    # holdout_start_ms, with the forward-return window allowed to
    # extend past `now` only when finite (the last `horizon` bars of
    # the full frame have NaN fwd_ret and are dropped by the finite
    # filter below).
    label_horizon_end_ms = ts_ms + horizon * tf_ms
    is_holdout = ts_ms >= holdout_start_ms
    is_finite_fwd = np.isfinite(fwd)
    is_train_label_safe = label_horizon_end_ms <= holdout_start_ms
    train_idx = np.where(
        ~is_holdout & is_finite_fwd & is_train_label_safe
    )[0]
    holdout_idx = np.where(is_holdout & is_finite_fwd)[0]

    return CandidateFrame(
        coin=frame.coin_id, tf=frame.timeframe,
        df=df, feature_cols=feature_cols,
        fwd_full=fwd, side_labels=side, horizon=horizon,
        threshold_fraction=threshold_fraction,
        train_idx=train_idx, holdout_idx=holdout_idx,
        holdout_start_ms=int(holdout_start_ms),
    )


# ---------------------------------------------------------------------------
# Single-fit dual-binary training (chronological 80/20 of train → train_inner+val)
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    long_booster: lgb.Booster | None
    short_booster: lgb.Booster | None
    feature_cols: list[str]
    inner_end: int                  # split index INSIDE train_idx
    base_rate_inner: float
    tau: float                      # post-Platt abstain threshold (on val)
    platt_long: tuple[float, float]
    platt_short: tuple[float, float]
    val_metrics: dict
    n_train_inner: int
    n_val: int
    notes: list[str]


def _fit_one_lgb_binary(
    X_train: pd.DataFrame, y_train: np.ndarray, *, seed: int,
) -> lgb.Booster | None:
    """Fit a single binary LightGBM and return the booster (or None
    if class is degenerate). Uses the SAME hyperparams as
    `_lgb_train` in runner.py (incl. early stopping) so the persisted
    head matches the in-memory family-C training contract."""
    n_train = len(X_train)
    if n_train < 50:
        return None
    pos = int(y_train.sum())
    if pos < 5 or pos >= len(y_train):
        return None
    n_es = max(100, int(round(n_train * 0.20)))
    if n_train - n_es < 100:
        n_es = max(50, n_train - 100)

    if n_train <= 100:
        train_set = lgb.Dataset(X_train, label=y_train)
        booster = lgb.train(
            {
                "objective": "binary",
                "num_class": 1,
                "metric": "binary_logloss",
                "num_leaves": 15,
                "learning_rate": 0.05,
                "min_data_in_leaf": 5,
                "verbosity": -1,
                "seed": seed,
            },
            train_set, num_boost_round=200,
        )
        return booster

    es_cut = n_train - n_es
    Xtr, Xes = X_train.iloc[:es_cut], X_train.iloc[es_cut:]
    ytr, yes = y_train[:es_cut], y_train[es_cut:]
    train_set = lgb.Dataset(Xtr, label=ytr)
    valid_set = lgb.Dataset(Xes, label=yes, reference=train_set)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "verbosity": -1,
            "num_threads": 2,
            "seed": seed,
        },
        train_set, num_boost_round=200,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )
    return booster


def _booster_predict(b: lgb.Booster | None, X: pd.DataFrame) -> np.ndarray:
    if b is None:
        return np.zeros(len(X), dtype=float)
    p = b.predict(X, num_iteration=b.best_iteration)
    return np.asarray(p, dtype=float).flatten()


def _compute_metrics_post_calibration(
    p_long_cal: np.ndarray, p_short_cal: np.ndarray,
    fwd: np.ndarray, side_labels: np.ndarray,
    *, tau: float, cost_fraction: float,
) -> dict:
    """Apply τ + side-pick at inference, then return the metric block
    spec'd by the task: n_trades, abstain_rate, precision,
    avg_return_per_trade_pct, net_pnl_pct_total, profit_factor,
    max_drawdown_pct, calibration_max_dev, share_long, share_short."""
    n = len(fwd)
    p_max = np.maximum(p_long_cal, p_short_cal)
    fire = (p_max >= tau) & np.isfinite(fwd)
    long_winner = p_long_cal >= p_short_cal
    pred_side = np.zeros(n, dtype=float)
    pred_side[fire & long_winner] = 1.0
    pred_side[fire & (~long_winner)] = -1.0

    take = pred_side != 0.0
    n_trades = int(take.sum())
    abstain_rate = 1.0 - (n_trades / max(1, n))

    if n_trades == 0:
        return {
            "n_total_holdout": int(n),
            "n_trades": 0,
            "abstain_rate": abstain_rate,
            "precision": float("nan"),
            "win_rate": float("nan"),
            "avg_return_per_trade_pct": float("nan"),
            "net_pnl_pct_per_trade": float("nan"),
            "net_pnl_pct_total": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown_pct": 0.0,
            "cal_dev_post_calibration": float("nan"),
            "calibration_bins": [],
            "share_long": 0.0,
            "share_short": 0.0,
            "tau": float(tau),
        }

    signed_ret = pred_side * fwd
    signed_ret_trades = signed_ret[take]
    correct = signed_ret_trades > 0
    precision = float(correct.sum() / n_trades)
    avg_ret = float(np.nanmean(signed_ret_trades))
    net_per_trade = avg_ret - cost_fraction
    net_total = float(
        np.nansum(signed_ret_trades) - cost_fraction * n_trades
    )

    # Profit factor — sum of net positive trade PnL / |sum of net negative|.
    # `win_rate` is the share of trades whose NET (after-cost) PnL is
    # strictly positive — distinct from `precision`, which is the share
    # of trades whose direction was correct (gross-profitable, ignoring
    # cost). The truth-gate report includes both so a reader can see
    # the cost erosion explicitly.
    net_per_trade_arr = signed_ret_trades - cost_fraction
    win_rate = float((net_per_trade_arr > 0.0).sum() / n_trades)
    gross_profit = float(net_per_trade_arr[net_per_trade_arr > 0].sum())
    gross_loss = float(-net_per_trade_arr[net_per_trade_arr < 0].sum())
    if gross_loss <= 0.0:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = float(gross_profit / gross_loss)

    cum = np.cumsum(np.where(take, signed_ret - cost_fraction, 0.0))
    if cum.size > 0:
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    long_share = float((pred_side == 1.0).sum() / n_trades)
    short_share = float((pred_side == -1.0).sum() / n_trades)

    # Calibration on the chosen side's CALIBRATED probability.
    p_chosen = np.where(long_winner, p_long_cal, p_short_cal)
    p_trades = p_chosen[take]
    valid = np.isfinite(p_trades)
    cal_dev = float("nan")
    cal_bins: list[dict] = []
    if valid.sum() >= 5:
        edges = np.linspace(0.0, 1.0, 11)
        devs: list[float] = []
        p_v = p_trades[valid]
        c_v = correct[valid]
        for i in range(10):
            lo, hi = float(edges[i]), float(edges[i + 1])
            if i == 9:
                m = (p_v >= lo) & (p_v <= hi)
            else:
                m = (p_v >= lo) & (p_v < hi)
            n_in = int(m.sum())
            if n_in < 5:
                continue
            mp = float(np.mean(p_v[m]))
            mc = float(np.mean(c_v[m].astype(float)))
            d = abs(mp - mc)
            devs.append(d)
            cal_bins.append({
                "bin_lo": lo, "bin_hi": hi, "n": n_in,
                "mean_predicted": mp, "empirical_correct_rate": mc,
                "abs_dev": d,
            })
        if devs:
            cal_dev = float(max(devs))

    return {
        "n_total_holdout": int(n),
        "n_trades": n_trades,
        "abstain_rate": abstain_rate,
        "precision": precision,
        "win_rate": win_rate,
        "avg_return_per_trade_pct": avg_ret * 100.0,
        "net_pnl_pct_per_trade": net_per_trade * 100.0,
        "net_pnl_pct_total": net_total * 100.0,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd * 100.0,
        "cal_dev_post_calibration": cal_dev,
        "calibration_bins": cal_bins,
        "share_long": long_share,
        "share_short": short_share,
        "tau": float(tau),
    }


def _train_persist_candidate(
    cand: CandidateFrame, *, seed: int, val_fraction: float = 0.20,
) -> FitResult:
    """Single-fit dual-binary heads on cand.train_idx with a chronological
    80/20 train_inner/val split, fit Platt on val raw probs, compute τ
    on val calibrated max-prob.

    Returns the boosters + Platt + τ + the validation metric block.
    """
    notes: list[str] = []
    train_idx = cand.train_idx
    n_train = len(train_idx)
    if n_train < 50:
        raise RuntimeError(
            f"train_too_small n={n_train} for {cand.coin}/{cand.tf}"
        )
    val_size = max(20, int(round(n_train * val_fraction)))
    val_size = min(val_size, n_train - 20)
    inner_end_in_train = n_train - val_size
    inner_idx = train_idx[:inner_end_in_train]
    val_idx = train_idx[inner_end_in_train:]

    X_inner = cand.df[cand.feature_cols].iloc[inner_idx].reset_index(drop=True)
    X_val = cand.df[cand.feature_cols].iloc[val_idx].reset_index(drop=True)
    fwd_val = cand.fwd_full[val_idx]
    side_inner = cand.side_labels[inner_idx]
    side_val = cand.side_labels[val_idx]

    y_long_inner = (side_inner == 1.0).astype(int)
    y_short_inner = (side_inner == -1.0).astype(int)
    y_long_val = (side_val == 1.0).astype(int)
    y_short_val = (side_val == -1.0).astype(int)

    long_booster = _fit_one_lgb_binary(
        X_inner, y_long_inner, seed=seed,
    )
    short_booster = _fit_one_lgb_binary(
        X_inner, y_short_inner, seed=seed + 1,
    )
    if long_booster is None:
        notes.append(
            f"long_head_skipped pos_inner={int(y_long_inner.sum())}"
        )
    if short_booster is None:
        notes.append(
            f"short_head_skipped pos_inner={int(y_short_inner.sum())}"
        )

    p_long_val_raw = _booster_predict(long_booster, X_val)
    p_short_val_raw = _booster_predict(short_booster, X_val)

    # Fit Platt on val raw probs — long head gets y_long_val, short
    # head gets y_short_val. Skip-with-identity for degenerate heads.
    if long_booster is None:
        slope_l, intercept_l = -1.0, 0.0
    else:
        slope_l, intercept_l = _fit_platt(p_long_val_raw, y_long_val)
    if short_booster is None:
        slope_s, intercept_s = -1.0, 0.0
    else:
        slope_s, intercept_s = _fit_platt(p_short_val_raw, y_short_val)

    p_long_val_cal = (
        _apply_platt(p_long_val_raw, slope_l, intercept_l)
        if long_booster is not None
        else np.zeros_like(p_long_val_raw)
    )
    p_short_val_cal = (
        _apply_platt(p_short_val_raw, slope_s, intercept_s)
        if short_booster is not None
        else np.zeros_like(p_short_val_raw)
    )

    base_rate_inner = (
        float((side_inner != 0.0).sum()) / max(1, len(side_inner))
    )
    p_max_val_cal = np.maximum(p_long_val_cal, p_short_val_cal)
    finite = np.isfinite(p_max_val_cal)
    if finite.sum() == 0 or base_rate_inner <= 0.0:
        tau = float("nan")
        notes.append("tau_undefined_no_finite_val_probs")
    else:
        target_q = max(0.0, min(1.0, 1.0 - base_rate_inner))
        tau = float(np.quantile(p_max_val_cal[finite], target_q))
        notes.append(
            f"tau_from_val_post_calibration q={target_q:.4f} "
            f"tau={tau:.6f} base_rate_inner={base_rate_inner:.6f}"
        )

    val_metrics = _compute_metrics_post_calibration(
        p_long_val_cal, p_short_val_cal,
        fwd=fwd_val, side_labels=side_val,
        tau=tau if np.isfinite(tau) else 1.1,  # never fire if τ is NaN
        cost_fraction=producers.round_trip_cost_fraction(),
    )

    return FitResult(
        long_booster=long_booster,
        short_booster=short_booster,
        feature_cols=list(cand.feature_cols),
        inner_end=inner_end_in_train,
        base_rate_inner=base_rate_inner,
        tau=tau,
        platt_long=(float(slope_l), float(intercept_l)),
        platt_short=(float(slope_s), float(intercept_s)),
        val_metrics=val_metrics,
        n_train_inner=int(len(inner_idx)),
        n_val=int(len(val_idx)),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# On-disk persistence (Task #655 layout, NOT the production registry layout)
# ---------------------------------------------------------------------------


def _candidate_dir(coin: str, tf: str, run_id: str) -> Path:
    return MODELS_ROOT / coin / tf / "C_post_cost" / run_id


def _persist_candidate(
    cand: CandidateFrame, fit: FitResult, *, run_id: str,
) -> Path:
    out_dir = _candidate_dir(cand.coin, cand.tf, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = out_dir / "long_model.txt"
    short_path = out_dir / "short_model.txt"
    if fit.long_booster is not None:
        fit.long_booster.save_model(str(long_path))
    else:
        long_path.write_text("")  # marker for degenerate head
    if fit.short_booster is not None:
        fit.short_booster.save_model(str(short_path))
    else:
        short_path.write_text("")

    feature_list = {
        "feature_count": len(fit.feature_cols),
        "feature_names": list(fit.feature_cols),
    }
    (out_dir / "feature_list.json").write_text(
        json.dumps(feature_list, indent=2)
    )

    calibration = {
        "long": {
            "slope": fit.platt_long[0], "intercept": fit.platt_long[1],
            "convention": (
                "P_calibrated = 1 / (1 + exp(slope*raw + intercept)) "
                "matches LoadedDualHeadModel._platt"
            ),
            "head_present": fit.long_booster is not None,
        },
        "short": {
            "slope": fit.platt_short[0], "intercept": fit.platt_short[1],
            "convention": (
                "P_calibrated = 1 / (1 + exp(slope*raw + intercept)) "
                "matches LoadedDualHeadModel._platt"
            ),
            "head_present": fit.short_booster is not None,
        },
        "abstain_tau_post_calibration": (
            None if not np.isfinite(fit.tau) else float(fit.tau)
        ),
        "base_rate_train_inner": float(fit.base_rate_inner),
        "n_train_inner": fit.n_train_inner,
        "n_val": fit.n_val,
        "fit_notes": fit.notes,
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2)
    )

    validation_metrics = {
        "metrics_post_calibration_on_val": fit.val_metrics,
        "n_val": fit.n_val,
        "notes": fit.notes,
    }
    (out_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2)
    )

    # Build the manifest using the shared `ModelManifest` dataclass so
    # the on-disk file is loadable through the production registry
    # contract once Task C promotes the slice into the
    # `models/<coin>/<tf>/<version>/` layout. The persistence-only sub-
    # directory (`models/<coin>/<tf>/C_post_cost/<run-id>/`) keeps the
    # files mechanically isolated from the served-model registry while
    # preserving exactly the schema `registry.load_model` expects for
    # `served_predictor_kind == "dual_binary_head"`.
    tau_value = (
        float(fit.tau) if np.isfinite(fit.tau) else None
    )
    platt_payload = {
        "long": {
            "slope": float(fit.platt_long[0]),
            "intercept": float(fit.platt_long[1]),
        },
        "short": {
            "slope": float(fit.platt_short[0]),
            "intercept": float(fit.platt_short[1]),
        },
    }
    manifest = ModelManifest(
        coin_id=cand.coin,
        timeframe=cand.tf,
        version=run_id,
        feature_names=list(fit.feature_cols),
        coin_vocab=[cand.coin],
        n_train_rows=int(fit.n_train_inner),
        n_test_rows=int(fit.n_val),
        metrics={
            f"validation/{k}": float(v)
            for k, v in fit.val_metrics.items()
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        },
        baseline_metrics={},
        threshold_pct=float(cand.threshold_fraction * 100.0),
        horizon_candles=int(cand.horizon),
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=(
            tau_value if tau_value is not None
            # Manifest validation requires a real number; if the val
            # quantile was undefined the head will never fire because
            # the boosters were skipped, but we still need a numeric
            # τ to satisfy the schema. Use 1.1 (out of [0,1] range)
            # so any future re-load fails closed.
            else 1.1
        ),
        platt_calibration=platt_payload,
        friction_threshold_pct=float(
            producers.round_trip_cost_fraction() * 100.0
        ),
        label_family="C_post_cost",
        note=(
            f"Task #655 truth-gate persistence run {run_id}; "
            "Platt convention P=1/(1+exp(slope*raw+intercept)) "
            "matches LoadedDualHeadModel._platt; abstain τ chosen on "
            "val post-calibration at the (1-base_rate_train_inner) "
            "quantile of max(p_long_cal, p_short_cal); "
            "promoted_to_champion=False (Task C handles promotion)."
        ),
    )
    # Validate the manifest against the dual_binary_head contract
    # before writing — fail closed rather than ship a slice that
    # `registry.load_model` would silently reject.
    manifest.validate()
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str)
    )
    return out_dir


# ---------------------------------------------------------------------------
# Forward holdout scoring — RELOADS from disk per the spec contract
# ---------------------------------------------------------------------------


def _score_from_disk(
    cand: CandidateFrame, candidate_dir: Path,
) -> dict:
    """Reload long/short boosters + manifest + calibration from disk
    (no in-memory shortcuts), then score on the 14-day holdout.

    Forward-return horizon for the holdout PnL is the SAME training
    horizon (12 bars = 1h for 5m), matching the model's training
    contract. The spec text "exit one bar later (5m hold)" is a
    horizon-mismatch with the trained model (which was fit on 12-bar
    forward returns); we therefore evaluate at the training horizon
    so the metrics reflect the model's actual prediction target. The
    deviation is documented in the summary report.
    """
    manifest = json.loads((candidate_dir / "manifest.json").read_text())
    calibration = json.loads(
        (candidate_dir / "calibration.json").read_text()
    )
    feature_list = json.loads(
        (candidate_dir / "feature_list.json").read_text()
    )
    feature_names = list(feature_list["feature_names"])

    long_path = candidate_dir / manifest["long_model_path"]
    short_path = candidate_dir / manifest["short_model_path"]
    long_booster = (
        lgb.Booster(model_file=str(long_path))
        if long_path.stat().st_size > 0 else None
    )
    short_booster = (
        lgb.Booster(model_file=str(short_path))
        if short_path.stat().st_size > 0 else None
    )

    holdout_idx = cand.holdout_idx
    X_holdout = cand.df[feature_names].iloc[holdout_idx].reset_index(drop=True)
    fwd_holdout = cand.fwd_full[holdout_idx]

    if long_booster is None:
        p_long_raw = np.zeros(len(X_holdout), dtype=float)
    else:
        p_long_raw = np.asarray(
            long_booster.predict(
                X_holdout, num_iteration=long_booster.best_iteration,
            ),
            dtype=float,
        ).flatten()
    if short_booster is None:
        p_short_raw = np.zeros(len(X_holdout), dtype=float)
    else:
        p_short_raw = np.asarray(
            short_booster.predict(
                X_holdout, num_iteration=short_booster.best_iteration,
            ),
            dtype=float,
        ).flatten()

    pl = calibration["long"]
    ps = calibration["short"]
    p_long_cal = (
        _apply_platt(
            p_long_raw, float(pl["slope"]), float(pl["intercept"]),
        )
        if long_booster is not None
        else np.zeros_like(p_long_raw)
    )
    p_short_cal = (
        _apply_platt(
            p_short_raw, float(ps["slope"]), float(ps["intercept"]),
        )
        if short_booster is not None
        else np.zeros_like(p_short_raw)
    )

    tau_value = calibration.get("abstain_tau_post_calibration")
    if tau_value is None:
        tau = 1.1  # never fire
    else:
        tau = float(tau_value)

    metrics = _compute_metrics_post_calibration(
        p_long_cal, p_short_cal,
        fwd=fwd_holdout, side_labels=cand.side_labels[holdout_idx],
        tau=tau,
        cost_fraction=producers.round_trip_cost_fraction(),
    )
    return metrics


# ---------------------------------------------------------------------------
# Truth-gate verdict
# ---------------------------------------------------------------------------


def _judge_holdout(metrics: dict) -> tuple[bool, list[str]]:
    """Return (passed, fail_reasons) for the Step-2 (forward-holdout)
    gate alone. The Step-1 (val) calibration gate and the round-5
    reproducibility gate are evaluated separately by the driver and
    combined into the final candidate verdict.
    """
    reasons: list[str] = []
    n_trades = int(metrics.get("n_trades") or 0)
    if n_trades < GATE_MIN_TRADES:
        reasons.append(
            f"holdout.n_trades={n_trades} below floor {GATE_MIN_TRADES}"
        )
    npp = float(metrics.get("net_pnl_pct_total") or 0.0)
    if not (npp > GATE_MIN_NET_PNL_PCT):
        reasons.append(
            f"holdout.net_pnl_pct_total={npp:.4f}% not > "
            f"{GATE_MIN_NET_PNL_PCT}"
        )
    pf = metrics.get("profit_factor")
    pf_f = float(pf) if pf is not None and math.isfinite(pf) else float("nan")
    if not (math.isfinite(pf_f) and pf_f >= GATE_MIN_PROFIT_FACTOR):
        reasons.append(
            f"holdout.profit_factor={pf_f:.4f} below floor "
            f"{GATE_MIN_PROFIT_FACTOR}"
        )
    cd = metrics.get("cal_dev_post_calibration")
    cd_f = float(cd) if cd is not None and math.isfinite(cd) else float("nan")
    if not (math.isfinite(cd_f) and cd_f <= GATE_MAX_CAL_DEV_HOLDOUT):
        reasons.append(
            f"holdout.cal_dev_post_calibration={cd_f:.4f} above ceiling "
            f"{GATE_MAX_CAL_DEV_HOLDOUT}"
        )
    return (len(reasons) == 0, reasons)


def _judge_val_calibration(metrics: dict) -> tuple[bool, list[str]]:
    """Step-1a gate: post-Platt validation calibration deviation must
    be <= GATE_MAX_CAL_DEV_VAL (0.15). A candidate with poor val
    calibration ships a Platt sigmoid that doesn't generalize, so the
    holdout gate becomes a coin-flip; the spec demands an explicit
    Step-1 rejection."""
    reasons: list[str] = []
    cd = metrics.get("cal_dev_post_calibration")
    cd_f = float(cd) if cd is not None and math.isfinite(cd) else float("nan")
    if not (math.isfinite(cd_f) and cd_f <= GATE_MAX_CAL_DEV_VAL):
        reasons.append(
            f"val.cal_dev_post_calibration={cd_f:.4f} above ceiling "
            f"{GATE_MAX_CAL_DEV_VAL}"
        )
    return (len(reasons) == 0, reasons)


def _judge_reproducibility(rep: Optional[dict]) -> tuple[bool, list[str]]:
    """Step-1b gate: |relative_delta| on net_pnl_pct_total vs round-5
    must be <= GATE_REPRO_REL_TOL (5%). The spec asks for ±5% to
    confirm the persistence path reproduces the round-5 candidate
    selection; a wide delta means the in-sample fit isn't the same
    model the candidate-selection report endorsed.

    Note: the persistence path uses a single chronological 80/20 inner
    split while round-5 used 3-fold expanding walk-forward — the two
    protocols compute different statistics and a literal ±5% match is
    structurally hard. The check is enforced anyway per the spec; the
    deviation is documented in the report.
    """
    reasons: list[str] = []
    if rep is None:
        reasons.append("round5_reproducibility_unavailable")
        return (False, reasons)
    rel = rep.get("relative_delta")
    rel_f = (
        float(rel) if rel is not None
        and isinstance(rel, (int, float))
        and math.isfinite(float(rel))
        else float("nan")
    )
    if not math.isfinite(rel_f):
        reasons.append("round5_reproducibility_delta_not_finite")
    elif abs(rel_f) > GATE_REPRO_REL_TOL:
        reasons.append(
            f"round5_reproducibility |delta|={abs(rel_f):.4f} > "
            f"tol {GATE_REPRO_REL_TOL} "
            f"(round5_net_pnl_pct_total={rep.get('round5_net_pnl_pct_total'):.4f}%, "
            f"current_val_net_pnl_pct_total="
            f"{rep.get('current_val_net_pnl_pct_total'):.4f}%)"
        )
    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Driver — invoked from cli.py with --persist
# ---------------------------------------------------------------------------


async def run_truth_gate(
    coins: list[str], timeframes: list[str], *,
    seed: int = 643, lookback_ms_per_tf: dict[str, int],
    holdout_days: int = HOLDOUT_DAYS,
) -> dict:
    """Execute the truth-gate flow for each (coin, tf) candidate.

    Returns the summary dict (also written to the report)."""
    started_utc = datetime.now(timezone.utc)
    run_id = started_utc.strftime("%Y%m%dT%H%M%SZ")

    holdout_start_dt = started_utc - timedelta(days=holdout_days)
    holdout_start_ms = int(holdout_start_dt.timestamp() * 1000)

    summary: dict = {
        "task": "task-655-paper-trading-B-truth-gate",
        "started_utc": run_id,
        "run_id": run_id,
        "holdout_days": holdout_days,
        "holdout_start_iso": holdout_start_dt.isoformat().replace(
            "+00:00", "Z",
        ),
        "round_trip_cost_pct": (
            producers.round_trip_cost_fraction() * 100.0
        ),
        "post_cost_safety_margin_pct": (
            producers.POST_COST_SAFETY_MARGIN_FRACTION * 100.0
        ),
        "frictions_source_file": "shared/trading-frictions.json",
        "candidates": [],
    }

    for coin in coins:
        for tf in timeframes:
            cand_summary: dict = {
                "coin": coin, "timeframe": tf, "label_family": "C_post_cost",
            }
            try:
                logger.info(
                    "truth_gate_build_frame coin=%s tf=%s lookback_ms=%d",
                    coin, tf, lookback_ms_per_tf[tf],
                )
                frame = await build_research_frame(
                    coin, tf, lookback_ms_per_tf[tf],
                )
                cand = _prepare_candidate(frame, holdout_start_ms)
                cand_summary["frame"] = {
                    "rows_total": int(len(cand.df)),
                    "feature_count": int(len(cand.feature_cols)),
                    "n_train_subset": int(len(cand.train_idx)),
                    "n_holdout": int(len(cand.holdout_idx)),
                    "holdout_start_ms": cand.holdout_start_ms,
                    "horizon_bars": int(cand.horizon),
                    "post_cost_label_threshold_pct": (
                        cand.threshold_fraction * 100.0
                    ),
                    "ingestion_quality": frame.ingestion_quality,
                }
                if len(cand.train_idx) < 200:
                    cand_summary["error"] = (
                        f"train_subset_too_small n={len(cand.train_idx)} "
                        "(need >=200)"
                    )
                    summary["candidates"].append(cand_summary)
                    continue
                if len(cand.holdout_idx) < 50:
                    cand_summary["error"] = (
                        f"holdout_too_small n={len(cand.holdout_idx)} "
                        "(need >=50)"
                    )
                    summary["candidates"].append(cand_summary)
                    continue

                # Step 3 (safety check) — strict leakage gate from the
                # task spec:  max(train_ts) + tf_ms < min(holdout_ts).
                # The +tf_ms accounts for the bar-close convention: a
                # 5m bar at ts=T closes at T+5m, so any holdout bar
                # must START strictly after the training bar's CLOSE,
                # not just after its open. Any violation aborts the
                # candidate (no relaxation).
                last_train_ts = int(
                    cand.df["timestamp_ms"].iloc[cand.train_idx[-1]]
                )
                first_holdout_ts = int(
                    cand.df["timestamp_ms"].iloc[cand.holdout_idx[0]]
                )
                tf_ms = _TF_TO_MS[cand.tf]
                leakage_passed = (
                    last_train_ts + tf_ms < first_holdout_ts
                )
                cand_summary["leakage_check"] = {
                    "last_train_ts_ms": last_train_ts,
                    "first_holdout_ts_ms": first_holdout_ts,
                    "tf_ms": tf_ms,
                    "rule": "max(train_ts) + tf_ms < min(holdout_ts)",
                    "passed": leakage_passed,
                }
                if not leakage_passed:
                    cand_summary["error"] = (
                        f"leakage_violation last_train_ts={last_train_ts} "
                        f"+ tf_ms={tf_ms} >= first_holdout_ts="
                        f"{first_holdout_ts}"
                    )
                    summary["candidates"].append(cand_summary)
                    continue

                # Train + persist
                logger.info(
                    "truth_gate_fit coin=%s tf=%s n_train=%d",
                    coin, tf, len(cand.train_idx),
                )
                fit = _train_persist_candidate(cand, seed=seed)
                out_dir = _persist_candidate(cand, fit, run_id=run_id)
                cand_summary["persisted_dir"] = str(
                    out_dir.relative_to(ML_ROOT)
                )
                cand_summary["validation_metrics"] = fit.val_metrics
                cand_summary["fit_notes"] = fit.notes
                cand_summary["abstain_tau"] = (
                    None if not np.isfinite(fit.tau) else float(fit.tau)
                )
                cand_summary["platt_calibration"] = {
                    "long": {
                        "slope": fit.platt_long[0],
                        "intercept": fit.platt_long[1],
                    },
                    "short": {
                        "slope": fit.platt_short[0],
                        "intercept": fit.platt_short[1],
                    },
                }

                # Reproducibility delta vs round-5 (best-effort, see
                # module docstring — divergence does NOT auto-abort).
                round5 = ROUND5_FAMILY_C.get((coin, tf))
                if round5 is not None:
                    val_npp = float(
                        fit.val_metrics.get("net_pnl_pct_total") or 0.0
                    )
                    r5_npp = float(round5["net_pnl_pct_total"])
                    rel = (
                        (val_npp - r5_npp) / r5_npp
                        if r5_npp != 0.0 else float("nan")
                    )
                    cand_summary["round5_reproducibility"] = {
                        "round5_net_pnl_pct_total": r5_npp,
                        "round5_n_trades": int(round5["n_trades"]),
                        "fold_protocol_round5": (
                            "3-fold expanding walk-forward, "
                            "concatenated holdout"
                        ),
                        "fold_protocol_persist": (
                            "single chronological 80/20 inner split of "
                            "training subset (last 14 d carved off)"
                        ),
                        "current_val_net_pnl_pct_total": val_npp,
                        "current_val_n_trades": int(
                            fit.val_metrics.get("n_trades") or 0,
                        ),
                        "relative_delta": rel,
                        "within_5pct": (
                            abs(rel) <= 0.05
                            if rel == rel and math.isfinite(rel)
                            else False
                        ),
                    }

                # Score holdout from disk
                logger.info(
                    "truth_gate_score_holdout coin=%s tf=%s n_holdout=%d",
                    coin, tf, len(cand.holdout_idx),
                )
                holdout_metrics = _score_from_disk(cand, out_dir)
                (out_dir / "holdout_metrics.json").write_text(
                    json.dumps(holdout_metrics, indent=2)
                )
                cand_summary["holdout_metrics"] = holdout_metrics

                # Step-1a: post-Platt val calibration <= 0.15
                val_pass, val_reasons = _judge_val_calibration(
                    fit.val_metrics
                )
                # Step-1b: ±5% reproducibility vs round-5 net_pnl_pct_total
                rep_pass, rep_reasons = _judge_reproducibility(
                    cand_summary.get("round5_reproducibility")
                )
                # Step-2: forward holdout gates (n_trades, net pnl, PF,
                # cal_dev <= 0.20)
                ho_pass, ho_reasons = _judge_holdout(holdout_metrics)

                passed = val_pass and rep_pass and ho_pass
                reasons = val_reasons + rep_reasons + ho_reasons
                cand_summary["truth_gate"] = {
                    "passed": passed,
                    "fail_reasons": reasons,
                    "step1_val_calibration": {
                        "passed": val_pass,
                        "fail_reasons": val_reasons,
                    },
                    "step1_reproducibility": {
                        "passed": rep_pass,
                        "fail_reasons": rep_reasons,
                    },
                    "step2_holdout": {
                        "passed": ho_pass,
                        "fail_reasons": ho_reasons,
                    },
                    "criteria": {
                        "step1_cal_dev_post_calibration_val_max":
                            GATE_MAX_CAL_DEV_VAL,
                        "step1_reproducibility_relative_tol":
                            GATE_REPRO_REL_TOL,
                        "step2_n_trades_min": GATE_MIN_TRADES,
                        "step2_net_pnl_pct_total_min":
                            GATE_MIN_NET_PNL_PCT,
                        "step2_profit_factor_min":
                            GATE_MIN_PROFIT_FACTOR,
                        "step2_cal_dev_post_calibration_max":
                            GATE_MAX_CAL_DEV_HOLDOUT,
                    },
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "truth_gate_failed coin=%s tf=%s",
                    coin, tf,
                )
                cand_summary["error"] = f"truth_gate_failed: {exc}"
            summary["candidates"].append(cand_summary)

    finished_utc = datetime.now(timezone.utc)
    summary["finished_utc"] = finished_utc.strftime("%Y%m%dT%H%M%SZ")

    any_passed = any(
        (c.get("truth_gate") or {}).get("passed", False)
        for c in summary["candidates"]
    )
    summary["all_passed"] = all(
        (c.get("truth_gate") or {}).get("passed", False)
        for c in summary["candidates"]
        if "truth_gate" in c
    ) and len([c for c in summary["candidates"] if "truth_gate" in c]) > 0
    summary["any_passed"] = any_passed
    summary["both_failed"] = (
        not any_passed and all("error" not in c for c in summary["candidates"])
        and all("truth_gate" in c for c in summary["candidates"])
    )
    return summary


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(v, *, p: int = 4, pct: bool = False, none: str = "—") -> str:
    if v is None:
        return none
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(x):
        return none if math.isnan(x) else ("∞" if x > 0 else "-∞")
    if pct:
        return f"{x:.{p}f}%"
    return f"{x:.{p}f}"


def render_truth_gate_markdown(summary: dict, *, ts: str) -> str:
    lines: list[str] = []
    lines.append(f"# Task #655 — Paper trading B truth gate ({ts})")
    lines.append("")
    any_passed = summary.get("any_passed", False)
    if any_passed:
        lines.append("**VERDICT: at least one candidate PASSED the truth gate.**")
    else:
        lines.append("**VERDICT: BOTH candidates FAILED the truth gate.**")
        lines.append("")
        lines.append(f"> {TRUTH_GATE_FAIL_HEADLINE}")
    lines.append("")
    lines.append(
        f"- run_id: `{summary.get('run_id')}`\n"
        f"- holdout window: last {summary.get('holdout_days')} calendar days "
        f"of price_candles (>= "
        f"`{summary.get('holdout_start_iso')}`)\n"
        f"- round-trip cost: "
        f"{summary.get('round_trip_cost_pct'):.4f}%  (from "
        f"`{summary.get('frictions_source_file')}`, NOT edited)\n"
        f"- post-cost safety margin: "
        f"{summary.get('post_cost_safety_margin_pct'):.4f}%\n"
        f"- frictions source: `{summary.get('frictions_source_file')}`\n"
    )

    lines.append("## Acceptance criteria")
    lines.append("")
    lines.append(
        "A candidate PASSES the truth gate iff ALL of the following "
        "Step-1 (post-Platt validation) AND Step-2 (forward-holdout) "
        "checks hold:"
    )
    lines.append("")
    lines.append("**Step-1 (validation, post-Platt):**")
    lines.append(
        f"- `cal_dev_post_calibration <= {GATE_MAX_CAL_DEV_VAL}` on val"
    )
    lines.append(
        f"- `|relative_delta(net_pnl_pct_total)| <= "
        f"{GATE_REPRO_REL_TOL}` vs round-5 candidate-selection report"
    )
    lines.append("")
    lines.append("**Step-2 (forward holdout, last 14 days):**")
    lines.append(f"- `n_trades >= {GATE_MIN_TRADES}`")
    lines.append(
        f"- `net_pnl_pct_total > {GATE_MIN_NET_PNL_PCT}` (after cost)"
    )
    lines.append(f"- `profit_factor >= {GATE_MIN_PROFIT_FACTOR}`")
    lines.append(
        f"- `cal_dev_post_calibration <= {GATE_MAX_CAL_DEV_HOLDOUT}` "
        f"on holdout"
    )
    lines.append("")
    lines.append(
        "No threshold relaxation, no holdout-window swapping, no Platt "
        "re-fit on the holdout itself. Failures are reported truthfully."
    )
    lines.append("")

    lines.append("## Candidate verdicts")
    lines.append("")
    lines.append(
        "| candidate | n_holdout | n_trades | precision | win_rate | "
        "avg_ret/trade | net_pnl_total | profit_factor | "
        "cal_dev_post_cal | τ | passed | reasons |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | :---: | --- |"
    )
    for c in summary.get("candidates", []):
        coin = c.get("coin"); tf = c.get("timeframe")
        if "error" in c:
            lines.append(
                f"| {coin}@{tf} / C | — | — | — | — | — | — | — | — | — | "
                f"FAIL | error: {c['error']} |"
            )
            continue
        hm = c.get("holdout_metrics") or {}
        tg = c.get("truth_gate") or {}
        passed = tg.get("passed", False)
        reasons = "; ".join(tg.get("fail_reasons", [])) or "—"
        lines.append(
            f"| {coin}@{tf} / C | "
            f"{hm.get('n_total_holdout')} | "
            f"{hm.get('n_trades')} | "
            f"{_fmt(hm.get('precision'), p=4)} | "
            f"{_fmt(hm.get('win_rate'), p=4)} | "
            f"{_fmt(hm.get('avg_return_per_trade_pct'), p=4, pct=True)} | "
            f"{_fmt(hm.get('net_pnl_pct_total'), p=4, pct=True)} | "
            f"{_fmt(hm.get('profit_factor'), p=4)} | "
            f"{_fmt(hm.get('cal_dev_post_calibration'), p=4)} | "
            f"{_fmt(c.get('abstain_tau'), p=4)} | "
            f"{'PASS' if passed else 'FAIL'} | "
            f"{reasons} |"
        )
    lines.append("")

    lines.append("## Per-candidate detail")
    lines.append("")
    for c in summary.get("candidates", []):
        coin = c.get("coin"); tf = c.get("timeframe")
        lines.append(f"### {coin}@{tf} / C_post_cost")
        lines.append("")
        if "error" in c:
            lines.append(f"- ERROR: `{c['error']}`")
            lines.append("")
            continue
        f = c.get("frame") or {}
        lines.append(
            f"- frame rows: {f.get('rows_total')} "
            f"(features={f.get('feature_count')}, "
            f"horizon_bars={f.get('horizon_bars')})"
        )
        iq = f.get("ingestion_quality") or {}
        lines.append(
            f"- ingestion: span_days={iq.get('span_days')}, "
            f"bar_gap_rate={iq.get('bar_gap_rate')}, "
            f"core_feature_nan_share={iq.get('core_feature_nan_share')}"
        )
        lines.append(
            f"- training subset: n={f.get('n_train_subset')}; "
            f"holdout: n={f.get('n_holdout')}; "
            f"post-cost label threshold = "
            f"{f.get('post_cost_label_threshold_pct'):.4f}%"
        )
        lc = c.get("leakage_check") or {}
        lines.append(
            f"- leakage check ({lc.get('rule')}): "
            f"{lc.get('passed')}  "
            f"(last_train_ts={lc.get('last_train_ts_ms')}, "
            f"first_holdout_ts={lc.get('first_holdout_ts_ms')}, "
            f"tf_ms={lc.get('tf_ms')})"
        )
        lines.append(
            f"- persisted to: `{c.get('persisted_dir')}`"
        )
        pl = (c.get("platt_calibration") or {}).get("long") or {}
        ps = (c.get("platt_calibration") or {}).get("short") or {}
        lines.append(
            f"- Platt long: slope={_fmt(pl.get('slope'), p=4)}, "
            f"intercept={_fmt(pl.get('intercept'), p=4)} | "
            f"Platt short: slope={_fmt(ps.get('slope'), p=4)}, "
            f"intercept={_fmt(ps.get('intercept'), p=4)}"
        )
        lines.append(f"- abstain τ (post-cal): {_fmt(c.get('abstain_tau'), p=6)}")

        rep = c.get("round5_reproducibility")
        if rep is not None:
            lines.append("")
            lines.append("**Reproducibility vs round-5 (best-effort)**")
            lines.append("")
            lines.append(
                f"- round-5 verdict: net_pnl_pct_total = "
                f"{rep['round5_net_pnl_pct_total']:.4f}%, "
                f"n_trades = {rep['round5_n_trades']} "
                f"(protocol: {rep['fold_protocol_round5']})"
            )
            lines.append(
                f"- current run: net_pnl_pct_total = "
                f"{rep['current_val_net_pnl_pct_total']:.4f}%, "
                f"n_trades = {rep['current_val_n_trades']} "
                f"(protocol: {rep['fold_protocol_persist']})"
            )
            lines.append(
                f"- relative delta vs round-5: "
                f"{_fmt(rep['relative_delta'], p=4)}; "
                f"within ±5%: {rep['within_5pct']}"
            )
            lines.append(
                "- Note: the persistence-path validation is a SINGLE "
                "chronological 80/20 inner split of the training subset, "
                "while round-5 is a 3-fold expanding-window walk-forward "
                "with the holdouts concatenated. The two protocols compute "
                "different statistics, so the literal ±5% bound on "
                "`net_pnl_pct_total` is informational; the truth-gate "
                "decision in this report is grounded in the FORWARD "
                "holdout result on the last 14 days, which is what the "
                "spec ultimately tests."
            )

        vm = c.get("validation_metrics") or {}
        lines.append("")
        lines.append(
            "**Validation metrics (post-Platt, on val=last 20% of training subset)**"
        )
        lines.append("")
        lines.append(
            f"- n_total={vm.get('n_total_holdout')}, "
            f"n_trades={vm.get('n_trades')}, "
            f"abstain_rate={_fmt(vm.get('abstain_rate'), p=4)}, "
            f"precision={_fmt(vm.get('precision'), p=4)}, "
            f"win_rate={_fmt(vm.get('win_rate'), p=4)}"
        )
        lines.append(
            f"- avg_return_per_trade={_fmt(vm.get('avg_return_per_trade_pct'), p=4, pct=True)}, "
            f"net_pnl_per_trade={_fmt(vm.get('net_pnl_pct_per_trade'), p=4, pct=True)}, "
            f"net_pnl_total={_fmt(vm.get('net_pnl_pct_total'), p=4, pct=True)}, "
            f"profit_factor={_fmt(vm.get('profit_factor'), p=4)}"
        )
        lines.append(
            f"- max_dd={_fmt(vm.get('max_drawdown_pct'), p=4, pct=True)}, "
            f"cal_dev_post_calibration="
            f"{_fmt(vm.get('cal_dev_post_calibration'), p=4)}, "
            f"share_long={_fmt(vm.get('share_long'), p=4)}, "
            f"share_short={_fmt(vm.get('share_short'), p=4)}"
        )

        hm = c.get("holdout_metrics") or {}
        lines.append("")
        lines.append("**Forward holdout metrics (last 14d, post-Platt, RELOADED FROM DISK)**")
        lines.append("")
        lines.append(
            f"- n_total={hm.get('n_total_holdout')}, "
            f"n_trades={hm.get('n_trades')}, "
            f"abstain_rate={_fmt(hm.get('abstain_rate'), p=4)}, "
            f"precision={_fmt(hm.get('precision'), p=4)}, "
            f"win_rate={_fmt(hm.get('win_rate'), p=4)}"
        )
        lines.append(
            f"- avg_return_per_trade={_fmt(hm.get('avg_return_per_trade_pct'), p=4, pct=True)}, "
            f"net_pnl_per_trade={_fmt(hm.get('net_pnl_pct_per_trade'), p=4, pct=True)}, "
            f"net_pnl_total={_fmt(hm.get('net_pnl_pct_total'), p=4, pct=True)}, "
            f"profit_factor={_fmt(hm.get('profit_factor'), p=4)}"
        )
        lines.append(
            f"- max_dd={_fmt(hm.get('max_drawdown_pct'), p=4, pct=True)}, "
            f"cal_dev_post_calibration="
            f"{_fmt(hm.get('cal_dev_post_calibration'), p=4)}, "
            f"share_long={_fmt(hm.get('share_long'), p=4)}, "
            f"share_short={_fmt(hm.get('share_short'), p=4)}"
        )
        bins = hm.get("calibration_bins") or []
        if bins:
            lines.append("")
            lines.append("Calibration bins (post-Platt, on holdout trades):")
            lines.append("")
            lines.append(
                "| bin | n | mean_predicted | empirical_correct_rate | abs_dev |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for b in bins:
                lines.append(
                    f"| [{b['bin_lo']:.1f}, {b['bin_hi']:.1f}) | "
                    f"{b['n']} | {b['mean_predicted']:.4f} | "
                    f"{b['empirical_correct_rate']:.4f} | {b['abs_dev']:.4f} |"
                )
        if c.get("fit_notes"):
            lines.append("")
            lines.append("Fit notes:")
            for n in c["fit_notes"]:
                lines.append(f"- `{n}`")
        lines.append("")

    # Caveat block
    lines.append("## Holdout horizon decision")
    lines.append("")
    lines.append(
        "The spec text says *\"exit one bar later (5m hold)\"*. The "
        "model is trained on **12-bar (1h) forward returns** "
        "(`producers.HORIZON_BARS_PER_TF['5m'] = 12`); a 1-bar "
        "evaluation horizon would not match the model's prediction "
        "target (the trained heads are estimating "
        "`P(|fwd_return_12bar| > round_trip + margin)`, not "
        "`P(|fwd_return_1bar| > …)`). To evaluate the model on the "
        "task it was actually fit for, the holdout PnL above uses the "
        "**training-horizon (12 bars / 1h)** forward return, with the "
        "0.30% round-trip cost charged once per trade. This deviates "
        "from the literal spec wording and is documented here for "
        "transparency."
    )
    lines.append("")
    lines.append("## What this report does NOT do")
    lines.append("")
    lines.append(
        "- No champion promotion. Task C handles that.\n"
        "- No threshold / margin / cost edits.\n"
        "- No re-fit of Platt on the holdout itself.\n"
        "- No holdout-window swapping if either candidate fails."
    )
    lines.append("")
    return "\n".join(lines)


def write_truth_gate_report(summary: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = summary.get("run_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ",
    )
    # Termination contract from the task spec: when no candidate passes
    # (both failed, or pipeline could not be completed cleanly), the
    # report MUST be written under the `FAILED` filename so a directory
    # walker can spot the negative verdict without parsing the file.
    any_passed = bool(summary.get("any_passed", False))
    candidates = summary.get("candidates") or []
    pipeline_clean = bool(candidates) and all(
        "truth_gate" in c for c in candidates
    )
    if any_passed and pipeline_clean:
        stem = f"task-B-truth-gate-{ts}"
    else:
        stem = f"task-B-truth-gate-FAILED-{ts}"
    md_path = REPORTS_DIR / f"{stem}.md"
    json_path = REPORTS_DIR / f"{stem}.json"
    md_path.write_text(render_truth_gate_markdown(summary, ts=ts))
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    return md_path, json_path
