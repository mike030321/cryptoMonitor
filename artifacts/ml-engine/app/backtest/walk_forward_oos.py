"""Re-uses the per-fold training machinery from `app.training.train` to
produce calibrated out-of-sample predictions for every row of a labeled
dataset, WITHOUT persisting a model.

The output is the input the simulator consumes: one row per OOS feature
candle with timestamp, coin, derived entry price, derived ATR, true label,
and the calibrated 3-class probability vector. By construction every row
is OOS for the booster + calibrators that produced it (no leakage).

Determinism: LightGBM is seeded (seed=42 in `_lgb_params`), Optuna is
seeded, and walk-forward splits are time-ordered. Two runs over the same
dataset return identical predictions.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from ..training.registry import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from ..training.train import (
    CALIBRATION_HOLDOUT_FRACTION,
    MIN_TRAIN_ROWS,
    _calibrate_per_class,
    _class_return_means_pct,
    _encode_coin_idx,
    _optuna_search_lgb,
    _prepare_xy,
    _train_regressor_head,
)
from ..training.walk_forward import WalkForwardConfig, walk_forward_splits

logger = logging.getLogger("ml-engine.backtest.oos")


def _derive_entry_price(row) -> float:
    """lastClose = atr14 / atrPct * 100. See features.py:198 — atrPct is
    defined as (atr14 / lastClose) * 100, so this is the exact inverse and
    not an approximation. Falls back to NaN if atrPct is zero.
    """
    atr14 = float(row["atr14"])
    atr_pct = float(row["atrPct"])
    if atr_pct <= 0:
        return float("nan")
    return atr14 / atr_pct * 100.0


def predict_oos_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward OOS prediction. Returns a DataFrame with columns:
    timestamp_ms, coin_id, entry_price, atr14, atr_pct, true_label,
    p_down, p_stable, p_up, fold_id.

    For datasets that fail the MIN_TRAIN_ROWS bar an empty frame is
    returned (matching the live training behaviour: do not fabricate).
    """
    if df.empty or len(df) < MIN_TRAIN_ROWS:
        return _empty_oos_frame()

    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    vocab = sorted(df["coin_id"].unique().tolist())
    df = _encode_coin_idx(df, vocab)
    X_all, y_all = _prepare_xy(df)

    # Mirror the policy in train.py:393.
    n_folds = max(2, min(5, len(df) // 30))
    cfg = WalkForwardConfig(
        n_folds=n_folds,
        min_train_size=max(50, len(df) // (n_folds + 1)),
    )

    oos_frames: list[pd.DataFrame] = []
    for fold_id, (tr_idx, te_idx) in enumerate(walk_forward_splits(df, cfg)):
        if len(te_idx) == 0:
            continue
        X_tr_full, y_tr_full = X_all.iloc[tr_idx], y_all[tr_idx]
        X_te,      y_te      = X_all.iloc[te_idx], y_all[te_idx]

        # Per-fold calibration tail: last CALIBRATION_HOLDOUT_FRACTION of train.
        cal_start = max(1, int(len(X_tr_full) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
        if cal_start >= len(X_tr_full) - 5:
            X_fit, y_fit = X_tr_full, y_tr_full
            calibrators = None
        else:
            X_fit, y_fit = X_tr_full.iloc[:cal_start], y_tr_full[:cal_start]
            X_cal, y_cal = X_tr_full.iloc[cal_start:], y_tr_full[cal_start:]
            _, booster_for_cal = _optuna_search_lgb(X_fit, y_fit)
            cal_pred_raw = booster_for_cal.predict(
                X_cal, num_iteration=booster_for_cal.best_iteration,
            )
            if cal_pred_raw.ndim == 1:
                cal_pred_raw = np.tile([0, 1, 0], (len(X_cal), 1)).astype(float)
            calibrators, _, _ = _calibrate_per_class(cal_pred_raw, y_cal)

        # Refit booster on the full (pre-cal-tail) training slice. We
        # intentionally use X_fit (not X_tr_full) for the FINAL booster too
        # so the calibrators we just fit are leak-free for OOS predictions.
        _, booster = _optuna_search_lgb(X_fit, y_fit)
        raw_preds = booster.predict(X_te, num_iteration=booster.best_iteration)
        if raw_preds.ndim == 1:
            raw_preds = np.tile([0, 1, 0], (len(X_te), 1)).astype(float)

        if calibrators is not None:
            cal = np.zeros_like(raw_preds)
            for k, c in enumerate(calibrators):
                cal[:, k] = c.predict(raw_preds[:, k]) if c is not None else raw_preds[:, k]
            s = cal.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            preds = cal / s
        else:
            preds = raw_preds

        # Task #135 — fit a magnitude head on non-stable rows of the FOLD's
        # training slice and use it for `expected_return_pct`. This mirrors
        # the live inference path (main.py uses model.regressor when
        # present), and is the OOS-side fix for the saturated-near-zero
        # regression head that task #130 documented. We fit on `tr_idx`
        # (the SAME slice the booster trained on) so each OOS row is still
        # a true holdout for the regressor too.
        train_slice = df.iloc[tr_idx]
        vocab_for_reg = sorted(train_slice["coin_id"].unique().tolist())
        regressor, _ = _train_regressor_head(train_slice, vocab_for_reg)
        if regressor is not None:
            X_te_for_reg = _encode_coin_idx(
                df.iloc[te_idx], vocab_for_reg,
            )[FEATURE_COLUMNS]
            magnitude = np.abs(regressor.predict(
                X_te_for_reg, num_iteration=regressor.best_iteration,
            ).astype(float))
            # Sign from the classifier — same composition as live
            # inference in main.py. The downstream gate uses
            # min_directional_prob/edge to filter weak directional
            # signals separately, so a tie (p_up == p_down) defaulting
            # to UP is a non-issue.
            sign = np.where(preds[:, 2] >= preds[:, 0], 1.0, -1.0)
            exp_ret = sign * magnitude
        else:
            # Fallback: not enough non-stable rows in this fold's training
            # slice. Use the legacy probability-weighted class-mean
            # expectation so the row still has a non-NaN expRet.
            means_pct = _class_return_means_pct(train_slice)
            if len(means_pct) != 3:
                means_pct = [0.0, 0.0, 0.0]
            exp_ret = (
                preds[:, 0] * means_pct[0]
                + preds[:, 1] * means_pct[1]
                + preds[:, 2] * means_pct[2]
            )

        oos_rows = df.iloc[te_idx].copy()
        oos_rows["entry_price"] = oos_rows.apply(_derive_entry_price, axis=1)
        oos_rows["true_label"] = y_te
        oos_rows["p_down"] = preds[:, 0]
        oos_rows["p_stable"] = preds[:, 1]
        oos_rows["p_up"] = preds[:, 2]
        oos_rows["expected_return_pct"] = exp_ret
        oos_rows["fold_id"] = fold_id
        keep = ["timestamp_ms", "coin_id", "entry_price", "atr14", "atrPct",
                "forward_return", "true_label", "p_down", "p_stable", "p_up",
                "expected_return_pct", "fold_id"]
        oos_frames.append(oos_rows[keep])

    if not oos_frames:
        return _empty_oos_frame()
    out = pd.concat(oos_frames, ignore_index=True)
    out = out.dropna(subset=["entry_price"]).reset_index(drop=True)
    return out


def _empty_oos_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_ms": pd.Series(dtype="int64"),
        "coin_id": pd.Series(dtype="object"),
        "entry_price": pd.Series(dtype="float64"),
        "atr14": pd.Series(dtype="float64"),
        "atrPct": pd.Series(dtype="float64"),
        "forward_return": pd.Series(dtype="float64"),
        "true_label": pd.Series(dtype="int64"),
        "p_down": pd.Series(dtype="float64"),
        "p_stable": pd.Series(dtype="float64"),
        "p_up": pd.Series(dtype="float64"),
        "expected_return_pct": pd.Series(dtype="float64"),
        "fold_id": pd.Series(dtype="int64"),
    })
