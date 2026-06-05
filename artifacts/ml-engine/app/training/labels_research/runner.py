"""Walk-forward trainer + holdout evaluator for Task #643.

Trains 4 distinct LightGBM model architectures per (coin, timeframe)
slice, one per label family:

* ``baseline_3class`` — 3-class multinomial on {-1, 0, +1} with the
  production-sourced threshold per (coin, tf). Abstain class is ``0``.
* ``A_quintile`` — true **5-class** multinomial on the quintile bucket
  (Q1..Q5) computed on the train subset only. At inference: argmax over
  5 classes; Q5 → +1 (long), Q1 → -1 (short), Q2..Q4 → 0 (abstain).
  The "abstain head" is the joint Q2+Q3+Q4 probability.
* ``B_sparse`` — TWO binary directional heads (``long_opp`` and
  ``short_opp``) trained on the sparse top-decile labels, with an
  explicit abstain calibration threshold τ chosen on the training fold
  to match the training base rate of opportunities. At inference: the
  side with the higher predicted prob wins iff that prob ≥ τ; else
  abstain.
* ``C_post_cost`` — same dual-binary-with-abstain architecture as B,
  but using the post-cost label (|fwd_ret| > round-trip + 0.10 %
  margin).

Models stay in-memory only; nothing is persisted to ``model_registry``,
nothing is promoted to champion.

Walk-forward splitting: **3-fold expanding-window walk-forward** with
identical training budget across all 4 families on every fold.

  fold 1: train rows [0%, 40%)  holdout rows [40%, 60%)
  fold 2: train rows [0%, 60%)  holdout rows [60%, 80%)
  fold 3: train rows [0%, 80%)  holdout rows [80%, 100%)

Total holdout coverage = 60 % of the slice, sourced exclusively from
post-train rows (no leakage in either direction). Within each fold the
inner ~20 % of the train segment is used for LightGBM early stopping.
The label edges (quintile cutpoints, sparse-top-decile threshold) are
re-fit on the train segment of each fold so a fold cannot peek at its
own holdout. The reported per-family metrics are computed on the
**concatenation** of the 3 fold holdouts so precision / net PnL /
calibration are integrated across the whole walk-forward window.

Round-trip cost is read from ``shared/trading-frictions.json`` via
``producers.round_trip_cost_fraction()``; no edits to that file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import producers
from .data import SliceFrame


EXCLUDE_COLS = {
    "coin_id", "timeframe", "timestamp_ms",
    "forward_return", "label_binary_up", "label_3class",
    "directional_label_horizon_candles",
    "directional_label_forward_return",
    "label_threshold_pct", "label_threshold_source",
    "label_threshold_source_per_coin_override",
    "label_threshold_source_timeframe_default",
    "directional_label_threshold_pct",
    "coin_idx",
}


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    numeric_kinds = {"i", "u", "f", "b"}
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if df[c].dtype.kind not in numeric_kinds:
            continue
        cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    holdout_idx: np.ndarray


def make_walk_forward_split(
    n: int, *, holdout_frac: float = 0.20,
) -> WalkForwardSplit:
    if n <= 0:
        return WalkForwardSplit(
            train_idx=np.array([], dtype=int),
            holdout_idx=np.array([], dtype=int),
        )
    n_holdout = max(1, int(round(n * holdout_frac)))
    n_train = n - n_holdout
    train_idx = np.arange(n_train, dtype=int)
    holdout_idx = np.arange(n_train, n, dtype=int)
    return WalkForwardSplit(train_idx=train_idx, holdout_idx=holdout_idx)


def make_walk_forward_splits(
    n: int, *, n_folds: int = 3, fold_frac: float = 0.20,
    initial_train_frac: float = 0.40,
) -> list[WalkForwardSplit]:
    """3-fold expanding-window walk-forward with identical fold size.

    For ``n_folds=3`` and the defaults this yields:

      fold 1: train [0, 0.40n)  holdout [0.40n, 0.60n)
      fold 2: train [0, 0.60n)  holdout [0.60n, 0.80n)
      fold 3: train [0, 0.80n)  holdout [0.80n, 1.00n)

    Folds are time-ordered; nothing in fold ``i``'s holdout ever
    appears in fold ``i``'s train, and the train sets grow with time
    so each fold's training budget is at least as informative as the
    previous one's.
    """
    if n <= 0 or n_folds <= 0:
        return []
    # Use floor() not round() for the fold size so that with the default
    # (initial_train_frac=0.40, fold_frac=0.20, n_folds=3) we always
    # land at 0.40n + 3*0.20n = n exactly when n is divisible by the
    # fold count, and stay strictly INSIDE the slice when it is not.
    # `round()` previously occasionally pushed the third fold's
    # ``hold_end`` one row past ``n`` and caused the splitter to
    # silently emit only 2 folds (e.g. BTC/ETH 5m at 92,128 rows).
    fold_n = max(1, int(n * fold_frac))
    initial_train = max(1, int(n * initial_train_frac))
    splits: list[WalkForwardSplit] = []
    for k in range(n_folds):
        train_end = initial_train + k * fold_n
        hold_end = train_end + fold_n
        if train_end <= 0 or hold_end > n:
            break
        train_idx = np.arange(0, train_end, dtype=int)
        holdout_idx = np.arange(train_end, hold_end, dtype=int)
        if len(train_idx) < 50 or len(holdout_idx) < 5:
            continue
        splits.append(
            WalkForwardSplit(train_idx=train_idx, holdout_idx=holdout_idx)
        )
    return splits


# ---------------------------------------------------------------------------
# LightGBM helpers
# ---------------------------------------------------------------------------


def _lgb_train(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_holdout: pd.DataFrame, *, num_class: int,
    seed: int = 643,
) -> np.ndarray:
    """Train a (multinomial or binary) LightGBM and return holdout
    probabilities of shape ``(n_holdout, num_class)`` (binary is
    expanded to two columns for a uniform shape).
    """
    import lightgbm as lgb

    n_train = len(X_train)
    n_es = max(100, int(round(n_train * 0.20)))
    if n_train - n_es < 100:
        n_es = max(50, n_train - 100)

    if n_train <= 100:
        train_set = lgb.Dataset(X_train, label=y_train)
        booster = lgb.train(
            {
                "objective": "multiclass" if num_class > 2 else "binary",
                "num_class": num_class if num_class > 2 else 1,
                "metric": (
                    "multi_logloss" if num_class > 2 else "binary_logloss"
                ),
                "num_leaves": 15,
                "learning_rate": 0.05,
                "min_data_in_leaf": 5,
                "verbosity": -1,
                "seed": seed,
            },
            train_set, num_boost_round=200,
        )
        preds = booster.predict(X_holdout)
        if num_class == 2:
            preds = np.column_stack([1 - preds, preds])
        return preds

    es_cut = n_train - n_es
    Xtr, Xes = X_train.iloc[:es_cut], X_train.iloc[es_cut:]
    ytr, yes = y_train[:es_cut], y_train[es_cut:]

    train_set = lgb.Dataset(Xtr, label=ytr)
    valid_set = lgb.Dataset(Xes, label=yes, reference=train_set)

    params = {
        "objective": "multiclass" if num_class > 2 else "binary",
        "metric": "multi_logloss" if num_class > 2 else "binary_logloss",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "num_threads": 2,
        "seed": seed,
    }
    if num_class > 2:
        params["num_class"] = num_class

    booster = lgb.train(
        params, train_set, num_boost_round=200,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )
    preds = booster.predict(X_holdout, num_iteration=booster.best_iteration)
    if num_class == 2:
        preds = np.column_stack([1 - preds, preds])
    return preds


# ---------------------------------------------------------------------------
# Family result container
# ---------------------------------------------------------------------------


@dataclass
class FamilyResult:
    family: str
    n_train: int
    n_holdout: int
    n_classes: int
    pred_side: np.ndarray            # {-1, 0, +1} per holdout row
    prob_chosen_side: np.ndarray     # NaN where pred_side == 0
    forward_ret_holdout: np.ndarray
    label_threshold_repr: str
    notes: list[str] = field(default_factory=list)
    abstain_threshold: float = float("nan")  # τ for B/C heads (NaN otherwise)


# ---------------------------------------------------------------------------
# Per-family trainers — distinct architectures per the task spec
# ---------------------------------------------------------------------------


def _train_family_three_class(
    *, side_labels_train: np.ndarray,
    X_train: pd.DataFrame, X_holdout: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Baseline 3-class multinomial (-1/0/+1). Abstain class = 0.

    Returns ``(pred_side, prob_chosen_side, notes, n_classes_present)``.
    """
    notes: list[str] = []
    # Map {-1, 0, +1} → {0, 1, 2}; abstain class is 1
    y_full = (side_labels_train + 1.0).astype(int)
    classes_present = np.unique(y_full)
    if len(classes_present) < 2:
        notes.append(
            f"degenerate_labels classes_present={classes_present.tolist()}"
        )
        return (
            np.zeros(len(X_holdout), dtype=float),
            np.full(len(X_holdout), np.nan, dtype=float),
            notes, int(len(classes_present)),
        )

    if len(classes_present) == 2:
        relabel_map = {int(c): i for i, c in enumerate(classes_present)}
        y_relabel = np.array([relabel_map[int(c)] for c in y_full])
        probs = _lgb_train(
            X_train, y_relabel, X_holdout, num_class=2, seed=seed,
        )
        full = np.zeros((len(X_holdout), 3), dtype=float)
        for i, c in enumerate(classes_present):
            full[:, int(c)] = probs[:, i]
        probs_full = full
    else:
        probs_full = _lgb_train(
            X_train, y_full, X_holdout, num_class=3, seed=seed,
        )

    chosen = np.argmax(probs_full, axis=1)
    pred_side = chosen.astype(float) - 1.0  # {0,1,2} → {-1,0,+1}

    # Probability of the chosen direction (for calibration). For
    # abstain rows we set NaN so the calibration metric reflects only
    # rows that actually traded.
    prob_chosen = np.full(len(X_holdout), np.nan, dtype=float)
    long_mask = pred_side == 1.0
    short_mask = pred_side == -1.0
    prob_chosen[long_mask] = probs_full[long_mask, 2]
    prob_chosen[short_mask] = probs_full[short_mask, 0]

    return pred_side, prob_chosen, notes, int(len(classes_present))


def _train_family_quintile_5class(
    *, quintile_labels_train: np.ndarray,
    X_train: pd.DataFrame, X_holdout: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """**True 5-class** multinomial on quintile buckets (Q1..Q5).

    At inference: argmax over 5 classes. Q5 → +1, Q1 → -1, Q2..Q4 → 0
    (abstain). The abstain head is the joint probability mass on
    Q2..Q4 — the model must concentrate probability on the extreme
    quintiles to commit to a trade.

    Returns ``(pred_side, prob_chosen_side, notes, n_classes_present)``.
    """
    notes: list[str] = []
    # quintile_labels_train uses values 1..5; remap to 0..4 for LGB
    finite_mask = np.isfinite(quintile_labels_train)
    if finite_mask.sum() < 25:
        notes.append("insufficient_train_quintiles")
        return (
            np.zeros(len(X_holdout), dtype=float),
            np.full(len(X_holdout), np.nan, dtype=float),
            notes, 0,
        )
    y_full = (quintile_labels_train[finite_mask] - 1.0).astype(int)
    X_train_used = X_train.iloc[finite_mask].reset_index(drop=True)
    classes_present = np.unique(y_full)
    if len(classes_present) < 2:
        notes.append(
            f"degenerate_quintile classes_present={classes_present.tolist()}"
        )
        return (
            np.zeros(len(X_holdout), dtype=float),
            np.full(len(X_holdout), np.nan, dtype=float),
            notes, int(len(classes_present)),
        )

    # If not all 5 classes are present in train, relabel contiguously
    # then reinflate to a 5-column probability matrix.
    if len(classes_present) < 5:
        relabel_map = {int(c): i for i, c in enumerate(classes_present)}
        y_relabel = np.array([relabel_map[int(c)] for c in y_full])
        if len(classes_present) == 2:
            probs = _lgb_train(
                X_train_used, y_relabel, X_holdout, num_class=2, seed=seed,
            )
        else:
            probs = _lgb_train(
                X_train_used, y_relabel, X_holdout,
                num_class=int(len(classes_present)), seed=seed,
            )
        full = np.zeros((len(X_holdout), 5), dtype=float)
        for i, c in enumerate(classes_present):
            full[:, int(c)] = probs[:, i]
        probs_full = full
    else:
        probs_full = _lgb_train(
            X_train_used, y_full, X_holdout, num_class=5, seed=seed,
        )

    chosen = np.argmax(probs_full, axis=1)  # 0..4
    pred_side = np.zeros(len(X_holdout), dtype=float)
    pred_side[chosen == 4] = 1.0   # Q5 → long
    pred_side[chosen == 0] = -1.0  # Q1 → short
    # chosen in {1,2,3} (= Q2/Q3/Q4) stays 0 (abstain)

    prob_chosen = np.full(len(X_holdout), np.nan, dtype=float)
    long_mask = pred_side == 1.0
    short_mask = pred_side == -1.0
    prob_chosen[long_mask] = probs_full[long_mask, 4]
    prob_chosen[short_mask] = probs_full[short_mask, 0]

    return pred_side, prob_chosen, notes, int(len(classes_present))


def _train_family_dual_binary_with_abstain(
    *, side_labels_train: np.ndarray,
    X_train: pd.DataFrame, X_holdout: pd.DataFrame,
    seed: int, family_tag: str,
    val_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, list[str], int, float]:
    """Dual binary directional heads + **validation-calibrated** abstain.

    * head_long  : binary classifier P(long_opportunity)  trained on
      target = (side_labels_train == +1)
    * head_short : binary classifier P(short_opportunity) trained on
      target = (side_labels_train == -1)

    Calibration protocol (round 4 fix — was previously in-sample on
    train predictions, which materially overstated head confidence):

      1. Split the train fold chronologically into ``train_inner`` (the
         first ``1 − val_fraction`` of rows) and ``val`` (the last
         ``val_fraction`` of rows). The split is index-based on the
         already-time-sorted train slice, so val is strictly later than
         train_inner — no peeking.
      2. Fit each head on ``train_inner`` only.
      3. Score the val rows with those heads → val ``max(p_long,
         p_short)`` distribution.
      4. τ := the ``1 − base_rate_train_inner`` quantile of the val
         max-prob distribution. This is the natural "match the empirical
         opportunity rate" rule but evaluated on **out-of-sample**
         validation predictions, so the τ value is not contaminated by
         train-set overfitting.
      5. Re-score the holdout with the **same** train_inner-fit heads.
         (We deliberately do not refit on the full train fold — that
         would invalidate the τ value, which is conditioned on the
         specific train_inner-fit head outputs.)

    At inference (holdout):
      * if max(p_long, p_short) < τ → abstain
      * else side := argmax(p_long, p_short) (long if tied)

    Returns
    -------
    (pred_side, prob_chosen_side, notes, n_classes_present, tau_used)
    """
    notes: list[str] = []
    n_holdout = len(X_holdout)
    pred_side = np.zeros(n_holdout, dtype=float)
    prob_chosen = np.full(n_holdout, np.nan, dtype=float)

    n_train_total = len(X_train)
    if n_train_total < 50:
        notes.append(f"train_too_small_for_val_split n_train={n_train_total}")
        return pred_side, prob_chosen, notes, 0, float("nan")

    # Chronological 80/20 split of train → train_inner + val.
    # X_train rows are already time-ordered upstream (slice driver
    # iterates per-fold over a time-sorted frame).
    val_size = max(20, int(round(n_train_total * val_fraction)))
    val_size = min(val_size, n_train_total - 20)  # keep ≥ 20 in train_inner
    inner_end = n_train_total - val_size
    X_inner = X_train.iloc[:inner_end].reset_index(drop=True)
    X_val = X_train.iloc[inner_end:].reset_index(drop=True)
    side_inner = side_labels_train[:inner_end]
    notes.append(
        f"val_calibration_split n_train_inner={inner_end} n_val={val_size}"
    )

    y_long_inner = (side_inner == 1.0).astype(int)
    y_short_inner = (side_inner == -1.0).astype(int)
    n_long = int(y_long_inner.sum())
    n_short = int(y_short_inner.sum())

    if n_long < 5 and n_short < 5:
        notes.append(
            f"degenerate_dual_binary positives_long_inner={n_long} "
            f"positives_short_inner={n_short}"
        )
        return pred_side, prob_chosen, notes, 1, float("nan")

    # Train each head on train_inner; score on val (for τ) AND on
    # holdout (for the final per-bar decision). Heads with one class
    # contribute all-zero probability so they never fire.
    p_long_val = np.zeros(len(X_val), dtype=float)
    p_short_val = np.zeros(len(X_val), dtype=float)
    p_long_holdout = np.zeros(n_holdout, dtype=float)
    p_short_holdout = np.zeros(n_holdout, dtype=float)
    n_classes_used = 0

    if n_long >= 5 and n_long < len(y_long_inner):
        p_long_val = _lgb_train(
            X_inner, y_long_inner, X_val, num_class=2, seed=seed,
        )[:, 1]
        p_long_holdout = _lgb_train(
            X_inner, y_long_inner, X_holdout, num_class=2, seed=seed,
        )[:, 1]
        n_classes_used += 1
    else:
        notes.append(f"long_head_skipped positives_inner={n_long}")

    if n_short >= 5 and n_short < len(y_short_inner):
        p_short_val = _lgb_train(
            X_inner, y_short_inner, X_val, num_class=2, seed=seed + 1,
        )[:, 1]
        p_short_holdout = _lgb_train(
            X_inner, y_short_inner, X_holdout, num_class=2, seed=seed + 1,
        )[:, 1]
        n_classes_used += 1
    else:
        notes.append(f"short_head_skipped positives_inner={n_short}")

    # Calibrate τ on **val** max-prob distribution (out-of-sample).
    # Quantile target := 1 − base_rate(train_inner).
    base_rate_inner = (
        float((side_inner != 0.0).sum()) / max(1, len(side_inner))
    )
    if base_rate_inner <= 0.0 or not np.isfinite(base_rate_inner):
        notes.append("base_rate_inner_zero_no_calibration")
        return pred_side, prob_chosen, notes, n_classes_used, float("nan")
    p_max_val = np.maximum(p_long_val, p_short_val)
    if len(p_max_val) == 0 or not np.isfinite(p_max_val).any():
        notes.append("val_max_prob_degenerate_no_calibration")
        return pred_side, prob_chosen, notes, n_classes_used, float("nan")
    target_quantile = max(0.0, min(1.0, 1.0 - base_rate_inner))
    tau = float(np.quantile(p_max_val, target_quantile))
    notes.append(
        f"tau_from_val quantile={target_quantile:.4f} tau={tau:.4f} "
        f"base_rate_inner={base_rate_inner:.4f}"
    )

    p_max_h = np.maximum(p_long_holdout, p_short_holdout)
    fire = p_max_h >= tau
    long_winner = p_long_holdout >= p_short_holdout
    pred_side[fire & long_winner] = 1.0
    pred_side[fire & (~long_winner)] = -1.0

    long_mask = pred_side == 1.0
    short_mask = pred_side == -1.0
    prob_chosen[long_mask] = p_long_holdout[long_mask]
    prob_chosen[short_mask] = p_short_holdout[short_mask]

    return pred_side, prob_chosen, notes, n_classes_used, tau


# ---------------------------------------------------------------------------
# Slice driver
# ---------------------------------------------------------------------------


def train_and_evaluate_slice(
    slice_frame: SliceFrame, *, seed: int = 643,
    progress_log: Optional[object] = None,
) -> dict:
    df = slice_frame.df
    timeframe = slice_frame.timeframe
    coin_id = slice_frame.coin_id

    if df.empty or len(df) < 200:
        return {
            "coin_id": coin_id, "timeframe": timeframe,
            "rows_total": int(len(df)),
            "bars_source": slice_frame.bars_source,
            "self_leak_columns_dropped": slice_frame.self_leak_columns_dropped,
            "skipped_reason": (
                f"insufficient_rows={len(df)} (need >=200 for "
                f"walk-forward + holdout)"
            ),
            "families": {},
        }

    horizon = producers.horizon_bars(timeframe)
    fr_1bar = df["forward_return"].astype(float).values
    close_implied = np.zeros(len(df), dtype=float)
    close_implied[0] = 1.0
    for i in range(len(df) - 1):
        close_implied[i + 1] = close_implied[i] * (1.0 + fr_1bar[i])

    fwd = producers.compute_forward_returns(close_implied, horizon)
    feature_cols = _select_feature_columns(df)
    X_all = df[feature_cols].copy()
    valid = np.isfinite(fwd)
    X_all = X_all.loc[valid].reset_index(drop=True)
    fwd_valid = fwd[valid]
    n = len(fwd_valid)

    if n < 250:
        return {
            "coin_id": coin_id, "timeframe": timeframe,
            "rows_total": int(len(df)), "rows_valid": int(n),
            "bars_source": slice_frame.bars_source,
            "self_leak_columns_dropped": slice_frame.self_leak_columns_dropped,
            "skipped_reason": (
                f"insufficient_valid_rows={n} after horizon mask "
                "(need >=250 for 3-fold walk-forward)"
            ),
            "families": {},
        }

    splits = make_walk_forward_splits(n, n_folds=3)
    if not splits:
        return {
            "coin_id": coin_id, "timeframe": timeframe,
            "rows_total": int(len(df)), "rows_valid": int(n),
            "bars_source": slice_frame.bars_source,
            "self_leak_columns_dropped": slice_frame.self_leak_columns_dropped,
            "skipped_reason": "no_valid_walk_forward_splits",
            "families": {},
        }

    families: dict[str, FamilyResult] = {}

    def _log(msg: str) -> None:
        if progress_log is not None:
            try:
                progress_log.write(msg + "\n")
                progress_log.flush()
            except Exception:
                pass

    import time as _time

    # Per-family accumulators across folds
    fam_acc: dict[str, dict] = {
        "baseline_3class": {"label_repr": "", "notes": [], "tau": [], "n_classes_per_fold": []},
        "A_quintile":     {"label_repr": "", "notes": [], "tau": [], "n_classes_per_fold": []},
        "B_sparse":       {"label_repr": "", "notes": [], "tau": [], "n_classes_per_fold": []},
        "C_post_cost":    {"label_repr": "", "notes": [], "tau": [], "n_classes_per_fold": []},
    }
    fam_pred: dict[str, list[np.ndarray]] = {k: [] for k in fam_acc}
    fam_prob: dict[str, list[np.ndarray]] = {k: [] for k in fam_acc}
    fam_fwd: list[np.ndarray] = []
    per_fold_meta: list[dict] = []

    bl_threshold_pct = base_threshold_pct(coin_id, timeframe)
    bl_threshold_frac = bl_threshold_pct / 100.0

    for fold_i, split in enumerate(splits, start=1):
        train_mask = np.zeros(n, dtype=bool)
        train_mask[split.train_idx] = True
        X_train = X_all.iloc[split.train_idx].reset_index(drop=True)
        X_holdout = X_all.iloc[split.holdout_idx].reset_index(drop=True)
        fwd_holdout = fwd_valid[split.holdout_idx]
        fam_fwd.append(fwd_holdout)
        per_fold_meta.append({
            "fold": fold_i,
            "train_start": int(split.train_idx[0]),
            "train_end_exclusive": int(split.train_idx[-1]) + 1,
            "holdout_start": int(split.holdout_idx[0]),
            "holdout_end_exclusive": int(split.holdout_idx[-1]) + 1,
            "n_train": int(len(split.train_idx)),
            "n_holdout": int(len(split.holdout_idx)),
        })

        # --- baseline 3-class
        _t0 = _time.time()
        _log(
            f"  fold{fold_i} baseline_3class "
            f"n_train={len(split.train_idx)} ..."
        )
        bl_labels = producers.label_three_class_baseline(
            fwd_valid, threshold_fraction=bl_threshold_frac,
        )
        pred, prob, notes, _n_cls = _train_family_three_class(
            side_labels_train=bl_labels.trade_side[split.train_idx],
            X_train=X_train, X_holdout=X_holdout, seed=seed + fold_i,
        )
        fam_pred["baseline_3class"].append(pred)
        fam_prob["baseline_3class"].append(prob)
        fam_acc["baseline_3class"]["notes"].extend(
            [f"fold{fold_i}:{nm}" for nm in notes]
        )
        fam_acc["baseline_3class"]["n_classes_per_fold"].append(int(_n_cls))
        fam_acc["baseline_3class"]["label_repr"] = (
            f"3-class threshold={bl_threshold_pct:.3f}% "
            f"(production source for {coin_id}/{timeframe})"
        )
        _log(
            f"  fold{fold_i} baseline_3class done in {_time.time()-_t0:.1f}s"
        )

        # --- A: true 5-class quintile
        _t0 = _time.time()
        qa = producers.label_quintile(fwd_valid, train_mask=train_mask)
        _log(f"  fold{fold_i} A_quintile_5class start ...")
        pred, prob, notes, _n_cls = _train_family_quintile_5class(
            quintile_labels_train=qa.quintile[split.train_idx],
            X_train=X_train, X_holdout=X_holdout, seed=seed + fold_i,
        )
        fam_pred["A_quintile"].append(pred)
        fam_prob["A_quintile"].append(prob)
        fam_acc["A_quintile"]["notes"].extend(
            [f"fold{fold_i}:{nm}" for nm in notes]
        )
        fam_acc["A_quintile"]["n_classes_per_fold"].append(int(_n_cls))
        if not fam_acc["A_quintile"]["label_repr"]:
            fam_acc["A_quintile"]["label_repr"] = (
                f"Q1<{qa.edges[0]*100:.4f}% "
                f"Q5>={qa.edges[3]*100:.4f}% "
                f"(quintile edges fit per-fold on train; 5-class multinomial)"
            )
        _log(
            f"  fold{fold_i} A_quintile_5class done in {_time.time()-_t0:.1f}s"
        )

        # --- B: dual-binary with abstain calibration
        _t0 = _time.time()
        sb = producers.label_sparse_top_decile(
            fwd_valid, train_mask=train_mask,
        )
        _log(f"  fold{fold_i} B_sparse_dual_binary start ...")
        pred, prob, notes, _n_cls, tau_b = (
            _train_family_dual_binary_with_abstain(
                side_labels_train=sb.trade_side[split.train_idx],
                X_train=X_train, X_holdout=X_holdout,
                seed=seed + fold_i, family_tag="B_sparse",
            )
        )
        fam_pred["B_sparse"].append(pred)
        fam_prob["B_sparse"].append(prob)
        fam_acc["B_sparse"]["notes"].extend(
            [f"fold{fold_i}:{nm}" for nm in notes]
        )
        fam_acc["B_sparse"]["tau"].append(tau_b)
        fam_acc["B_sparse"]["n_classes_per_fold"].append(int(_n_cls))
        if not fam_acc["B_sparse"]["label_repr"]:
            fam_acc["B_sparse"]["label_repr"] = (
                f"|fwd_ret| >= {sb.threshold_abs*100:.4f}% "
                f"(top-decile train-set cutoff fit per-fold; "
                f"dual binary heads, abstain τ calibrated on val "
                f"split — chronological 80/20 of train, heads fit on "
                f"train_inner only, τ = (1 − base_rate_inner) "
                f"quantile of val max(p_long,p_short))"
            )
        _log(
            f"  fold{fold_i} B_sparse_dual_binary done "
            f"in {_time.time()-_t0:.1f}s tau={tau_b:.4f}"
        )

        # --- C: dual-binary with abstain calibration on post-cost
        _t0 = _time.time()
        pc = producers.label_post_cost(fwd_valid)
        _log(f"  fold{fold_i} C_post_cost_dual_binary start ...")
        pred, prob, notes, _n_cls, tau_c = (
            _train_family_dual_binary_with_abstain(
                side_labels_train=pc.trade_side[split.train_idx],
                X_train=X_train, X_holdout=X_holdout,
                seed=seed + fold_i, family_tag="C_post_cost",
            )
        )
        fam_pred["C_post_cost"].append(pred)
        fam_prob["C_post_cost"].append(prob)
        fam_acc["C_post_cost"]["notes"].extend(
            [f"fold{fold_i}:{nm}" for nm in notes]
        )
        fam_acc["C_post_cost"]["tau"].append(tau_c)
        fam_acc["C_post_cost"]["n_classes_per_fold"].append(int(_n_cls))
        if not fam_acc["C_post_cost"]["label_repr"]:
            fam_acc["C_post_cost"]["label_repr"] = (
                f"|fwd_ret| > {pc.threshold_fraction*100:.3f}% "
                f"(round-trip cost "
                f"{producers.round_trip_cost_fraction()*100:.2f}% + "
                f"margin {producers.POST_COST_SAFETY_MARGIN_FRACTION*100:.2f}%;"
                f" dual binary heads, abstain τ calibrated on val "
                f"split — chronological 80/20 of train, heads fit on "
                f"train_inner only, τ = (1 − base_rate_inner) "
                f"quantile of val max(p_long,p_short))"
            )
        _log(
            f"  fold{fold_i} C_post_cost_dual_binary done "
            f"in {_time.time()-_t0:.1f}s tau={tau_c:.4f}"
        )

    fwd_concat = np.concatenate(fam_fwd) if fam_fwd else np.array([])
    for fname in ("baseline_3class", "A_quintile", "B_sparse", "C_post_cost"):
        pred_concat = (
            np.concatenate(fam_pred[fname])
            if fam_pred[fname] else np.array([])
        )
        prob_concat = (
            np.concatenate(fam_prob[fname])
            if fam_prob[fname] else np.array([])
        )
        taus = fam_acc[fname]["tau"]
        finite_taus = [t for t in taus if t == t and np.isfinite(t)]
        mean_tau = (
            float(np.mean(finite_taus)) if finite_taus else float("nan")
        )
        # Aggregate `n_classes_present_in_train` as the **maximum**
        # across folds — gives an honest "how many distinct label
        # classes did this family ever see in training" signal. A
        # fold-level minimum would falsely flag a degenerate fold for
        # the whole slice; the max captures whether the family had at
        # least one fold with full class coverage. Per-fold values are
        # still available in the family `notes` (they're emitted as
        # `degenerate_labels classes_present=[…]` notes when a fold
        # has < 2 classes).
        nc_per_fold = fam_acc[fname]["n_classes_per_fold"]
        n_classes_agg = int(max(nc_per_fold)) if nc_per_fold else 0
        families[fname] = FamilyResult(
            family=fname,
            n_train=int(sum(m["n_train"] for m in per_fold_meta)),
            n_holdout=int(sum(m["n_holdout"] for m in per_fold_meta)),
            n_classes=n_classes_agg, pred_side=pred_concat,
            prob_chosen_side=prob_concat,
            forward_ret_holdout=fwd_concat,
            label_threshold_repr=fam_acc[fname]["label_repr"],
            notes=fam_acc[fname]["notes"],
            abstain_threshold=mean_tau,
        )

    iq = slice_frame.ingestion_quality or {}
    return {
        "coin_id": coin_id, "timeframe": timeframe,
        "rows_total": int(len(df)),
        "rows_valid": int(n),
        "n_folds": len(splits),
        "fold_layout": per_fold_meta,
        "n_train_total": int(sum(m["n_train"] for m in per_fold_meta)),
        "n_holdout_total": int(sum(m["n_holdout"] for m in per_fold_meta)),
        "bars_source": slice_frame.bars_source,
        "self_leak_columns_dropped": slice_frame.self_leak_columns_dropped,
        "horizon_bars": int(horizon),
        "feature_count": int(len(feature_cols)),
        "round_trip_cost_pct": producers.round_trip_cost_fraction() * 100.0,
        "ingestion_quality": iq,
        "families": {k: _serialize_family(v) for k, v in families.items()},
    }


def base_threshold_pct(coin_id: str, timeframe: str) -> float:
    return float(base_labels_threshold_for(coin_id, timeframe))


def base_labels_threshold_for(coin_id: str, timeframe: str) -> float:
    from .. import labels as base_labels
    return float(base_labels.resolve_label_threshold_pct(coin_id, timeframe))


# ---------------------------------------------------------------------------
# Holdout metrics + reliability calibration
# ---------------------------------------------------------------------------


def _serialize_family(fr: FamilyResult) -> dict:
    metrics = compute_holdout_metrics(
        pred_side=fr.pred_side,
        prob_chosen_side=fr.prob_chosen_side,
        fwd_holdout=fr.forward_ret_holdout,
    )
    return {
        "family": fr.family,
        "n_train": fr.n_train,
        "n_holdout": fr.n_holdout,
        "n_classes_present_in_train": fr.n_classes,
        "label_rule": fr.label_threshold_repr,
        "abstain_threshold": (
            None if not np.isfinite(fr.abstain_threshold)
            else float(fr.abstain_threshold)
        ),
        "metrics": metrics,
        "notes": fr.notes,
    }


def _calibration_max_dev(
    prob_chosen: np.ndarray, correct: np.ndarray,
    *, n_bins: int = 10, min_per_bin: int = 5,
) -> tuple[float, list[dict]]:
    """Reliability deviation: bin the trades by predicted probability
    of the chosen direction, then compare bin-mean predicted prob to
    bin-mean realised correctness. Returns ``(max_abs_dev, table)``.

    Bins span [0.0, 1.0] in equal-width slices. Empty / under-sampled
    bins are skipped. ``max_abs_dev`` is the largest absolute gap on
    any populated bin (== 0 means perfect calibration). NaN if there
    are no populated bins.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: list[dict] = []
    devs: list[float] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (prob_chosen >= lo) & (prob_chosen <= hi)
        else:
            mask = (prob_chosen >= lo) & (prob_chosen < hi)
        n_in = int(mask.sum())
        if n_in < min_per_bin:
            continue
        mean_p = float(np.mean(prob_chosen[mask]))
        mean_c = float(np.mean(correct[mask].astype(float)))
        dev = abs(mean_p - mean_c)
        devs.append(dev)
        table.append({
            "bin_lo": lo, "bin_hi": hi, "n": n_in,
            "mean_predicted": mean_p, "empirical_correct_rate": mean_c,
            "abs_dev": dev,
        })
    if not devs:
        return float("nan"), table
    return float(max(devs)), table


def compute_holdout_metrics(
    pred_side: np.ndarray, prob_chosen_side: np.ndarray,
    fwd_holdout: np.ndarray,
    cost_fraction: Optional[float] = None,
) -> dict:
    if cost_fraction is None:
        cost_fraction = producers.round_trip_cost_fraction()
    n = len(pred_side)
    take = pred_side != 0.0
    n_trades = int(take.sum())
    n_total = int(n)
    abstain_rate = 1.0 - (n_trades / n_total) if n_total else 0.0

    if n_trades == 0:
        return {
            "n_total_holdout": n_total,
            "n_trades": 0,
            "abstain_rate": abstain_rate,
            "directional_call_share": 0.0,
            "precision": float("nan"),
            "directional_accuracy_on_trades": float("nan"),
            "avg_return_per_trade_pct": float("nan"),
            "net_pnl_pct_per_trade": float("nan"),
            "net_pnl_pct_total": 0.0,
            "max_drawdown_pct": 0.0,
            "calibration_max_dev": float("nan"),
            "calibration_bins": [],
            "share_long": 0.0,
            "share_short": 0.0,
        }

    signed_ret = pred_side * fwd_holdout
    signed_ret_trades = signed_ret[take]
    correct_trades = signed_ret_trades > 0
    precision = float(correct_trades.sum() / n_trades)

    avg_ret_per_trade = float(np.nanmean(signed_ret_trades))
    net_per_trade = avg_ret_per_trade - cost_fraction
    net_total = float(
        np.nansum(signed_ret_trades) - cost_fraction * n_trades
    )

    cum = np.cumsum(np.where(take, signed_ret - cost_fraction, 0.0))
    if cum.size > 0:
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    long_share = float((pred_side == 1.0).sum() / n_trades)
    short_share = float((pred_side == -1.0).sum() / n_trades)

    # Calibration is computed on rows that actually traded; a row's
    # predicted "probability of being a correct directional call" is
    # the model's probability assigned to the chosen side.
    prob_trades = prob_chosen_side[take]
    valid = np.isfinite(prob_trades)
    if valid.sum() >= 5:
        cal_dev, cal_bins = _calibration_max_dev(
            prob_trades[valid], correct_trades[valid],
        )
    else:
        cal_dev, cal_bins = float("nan"), []

    return {
        "n_total_holdout": n_total,
        "n_trades": n_trades,
        "abstain_rate": abstain_rate,
        "directional_call_share": 1.0 - abstain_rate,
        "precision": precision,
        "directional_accuracy_on_trades": precision,
        "avg_return_per_trade_pct": avg_ret_per_trade * 100.0,
        "net_pnl_pct_per_trade": net_per_trade * 100.0,
        "net_pnl_pct_total": net_total * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "calibration_max_dev": cal_dev,
        "calibration_bins": cal_bins,
        "share_long": long_share,
        "share_short": short_share,
    }


# ---------------------------------------------------------------------------
# Task #654 — paper-trading family C ("dual_binary_head") persistence path.
# ---------------------------------------------------------------------------
#
# Up to and including Task #643, the labels research runner is strictly
# in-memory: `_train_family_dual_binary_with_abstain` returns prediction
# arrays and discards the trained boosters after the fold. That's
# correct for *evaluation* — the per-fold heads must not be persisted
# because they're conditioned on a chronological train/val split that
# isn't representative of the full slice.
#
# Task #654 ("foundation") needs an actual on-disk artefact in the
# new `served_predictor_kind="dual_binary_head"` shape so the registry
# load + serve path can be exercised end-to-end. We deliberately
# implement persistence as a SEPARATE function (`persist_dual_binary_head`)
# that takes the FULLY-trained heads + Platt + τ + manifest fields the
# caller has already produced. We do NOT modify the existing in-memory
# evaluator, so the labels-research evaluation pipeline keeps its
# leakage-free contract.
#
# The function is a thin wrapper over `app.training.registry.save_model`
# that:
#   - constructs a `served_predictor_kind="dual_binary_head"` manifest
#     with all the family-C fields populated (validate() catches any
#     missing field before disk is touched)
#   - calls save_model with `booster=long_head, regressor=short_head`
#     (the kwargs save_model already understands for this family)
#
# No model is promoted by this function; that step is gated by the
# operator via `app.registry_lifecycle.promote_shadow_to_serving`.

from typing import Any  # noqa: E402  (intentionally late, after main module body)


def persist_dual_binary_head(
    *,
    coin_id: str,
    timeframe: str,
    version: str,
    long_booster: Any,
    short_booster: Any,
    feature_names: list[str],
    coin_vocab: list[str],
    n_train_rows: int,
    n_test_rows: int,
    metrics: dict,
    abstain_tau: float,
    platt_calibration: dict,
    friction_threshold_pct: float,
    horizon_candles: int,
    label_family: str = "C_post_cost",
    long_model_path: str = "long_model.txt",
    short_model_path: str = "short_model.txt",
    long_model_filename: Optional[str] = None,
    short_model_filename: Optional[str] = None,
):
    """Persist a fully-trained family-C dual-head model to the registry.

    Returns the on-disk version directory (`pathlib.Path`) so the caller
    can attach it to a registry row via the api-server's existing
    insert-shadow path. The slot is registered in `state='shadow'`
    (default in the registry's insert flow); it is NOT promoted to
    `champion` here — that is the explicit responsibility of
    `app.registry_lifecycle.promote_shadow_to_serving` and lives behind
    a separate operator action.

    Parameters mirror the registry manifest's family-C field set; see
    `ModelManifest` (`served_predictor_kind="dual_binary_head"`) for
    the validation rules applied on save.

    `long_model_filename` / `short_model_filename` are accepted for
    callers that want to override the default file names; they take
    precedence over the legacy `long_model_path` / `short_model_path`
    kwargs.
    """
    # Imported lazily so this module stays importable in environments
    # (e.g. light-weight test runs) that don't load the registry side.
    from app.training.registry import ModelManifest, save_model

    long_path = long_model_filename or long_model_path
    short_path = short_model_filename or short_model_path

    manifest = ModelManifest(
        coin_id=coin_id,
        timeframe=timeframe,
        version=version,
        feature_names=list(feature_names),
        coin_vocab=list(coin_vocab),
        n_train_rows=int(n_train_rows),
        n_test_rows=int(n_test_rows),
        metrics=dict(metrics),
        baseline_metrics={},
        threshold_pct=float(friction_threshold_pct),
        horizon_candles=int(horizon_candles),
        # 3-class field stays empty for dual-head (validate() will
        # accept this when served_predictor_kind == "dual_binary_head").
        class_return_means_pct=[],
        served_predictor_kind="dual_binary_head",
        long_model_path=long_path,
        short_model_path=short_path,
        abstain_tau=float(abstain_tau),
        platt_calibration=dict(platt_calibration),
        friction_threshold_pct=float(friction_threshold_pct),
        label_family=label_family,
    )
    # save_model() calls manifest.validate() inside its dual-head branch,
    # so a misconfigured manifest (missing Platt / τ / paths) will raise
    # ValueError BEFORE any file is written.
    return save_model(
        coin_id=coin_id,
        timeframe=timeframe,
        version=version,
        booster=long_booster,
        calibrators=None,
        manifest=manifest,
        regressor=short_booster,
    )
