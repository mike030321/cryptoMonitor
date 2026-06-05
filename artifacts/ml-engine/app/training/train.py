"""Walk-forward LightGBM trainer (multiclass DOWN/STABLE/UP).

For each timeframe in DEFAULT_TIMEFRAMES we:

1. Build the labeled dataset (one row per coin × bucket).
2. Persist the labeled frame to `models/datasets/{tf}_{version}.parquet` so a
   training run is reproducible.
3. For each tracked coin, attempt per-coin training (registry path
   `{coin}/{tf}/{version}/`). If the per-coin slice is too small, that coin
   is added to the pool that gets a single shared `__pooled__` model.
4. Per training: walk-forward CV with an expanding window. Each fold trains
   both a multinomial logistic regression baseline (one-hot coin) and a
   3-class LightGBM tuned by an Optuna study, then evaluates on the held-out
   test slice.
5. After CV, refit the chosen LightGBM on the first (1 - CALIBRATION_HOLDOUT_FRACTION)
   chronological share of the data, then fit one isotonic calibrator per class
   on the held-out tail. The deployed booster and the data the calibrators
   were fit on come from the same model, so calibration is leak-free.
6. The trainer also stores per-class mean forward returns in the manifest so
   `/ml/predict` can compute a *real* expectedReturnPct and predictionStdPct
   from the calibrated 3-class probability vector — no synthesis.

CLI:
    pnpm --filter @workspace/ml-engine train
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from .calibration import (
    TemperatureScaledClass,
    apply_single_temperature,
    fit_single_temperature,
)

from ..db import close_pool, init_pool
from . import labels as _labels_mod
from .labels import (
    FORWARD_HORIZON_CANDLES,
    LABEL_THRESHOLDS_PERCENT,
    MULTI_BAR_LABEL_THRESHOLDS_PERCENT,
    OUTCOME_THRESHOLDS_PERCENT,
    audit_leakage,
    build_labeled_dataset,
    resolve_directional_label_threshold_pct,
    resolve_label_threshold_pct,
)
from . import registry as registry_module
from .registry import (
    CATEGORICAL_FEATURES,
    CONTRACT_NEW_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_LINEAGE,
    FORWARD_TARGET_COLUMNS,
    ModelManifest,
    POOLED_COIN_ID,
    SPECIALIST_KINDS,
    SPECIALIST_REGIME_MAP,
    dataset_path,
    make_version,
    save_model,
    specialist_coin_id,
)
from .walk_forward import WalkForwardConfig, walk_forward_splits

logger = logging.getLogger("ml-engine.train")

# Silence Optuna's per-trial INFO chatter so the training log stays readable.
optuna.logging.set_verbosity(optuna.logging.WARNING)

NUM_CLASSES = 3  # 0=DOWN, 1=STABLE, 2=UP

# --- Default tracked coins -----------------------------------------------------
# `sei-network` was removed in task #409 because no free venue can deliver
# the 305-day strict-contiguous 5m history the hard gate requires:
#   - OKX free `history-candles` truncates SEI-USDT 5m at ~161 days back.
#   - Binance and Bybit are geo-blocked from Replit infrastructure.
#   - Coinbase serves SEI-USD 5m back to ~2025-06-10 but does not emit
#     zero-volume bars for thin 5-minute windows (643 small thinness gaps)
#     and had a real ~6h venue outage on 2025-10-25 that no other free
#     source covers. Longest contiguous run achievable: ~181 days < 305.
# Full rationale: reports/task-409-sei-5m-rationale.md
# To re-enable: add "sei-network" back to this list.
DEFAULT_COINS: list[str] = [
    "pepe", "bonk", "floki-inu", "dogwifcoin",
    "render-token", "injective-protocol", "celestia",
    "worldcoin-wld", "jupiter-exchange-solana",
]

# Real-data lookback in days. Defaults to 365 so the trainer absorbs the
# full year of OHLCV the backfill module pulls from OKX (and CMC for 1d).
# Override via env so an operator can dial it without redeploying.
# Historical default was 30 (no-op when only ~13h of ticks existed); the
# bigger window is safe because labels.py already filters synthetic rows
# at SQL and stamps `provenance.rejected_synthetic` for any leak.
LOOKBACK_DAYS = int(os.environ.get("ML_LOOKBACK_DAYS", "365"))

# Task #417 — per-timeframe lookback overrides. The 1d slice was bound by
# the same 365-day envelope as 5m/1h/2h/6h, which capped the holdout at
# ~265 rows. The binomial-noise σ on a directional-accuracy estimate at
# n=265, p=0.5 is ~0.031 (see reports/20260423T204419Z-task379-root-cause.md
# and reports/20260424T083741Z-task401-1d-da-floor.md). Widening 1d to a
# multi-year window grows the holdout to ~720+ rows and shrinks σ to
# ~0.019, which is what gives the original 0.50 / new 0.530 DA gates
# honest statistical power. The shorter timeframes deliberately keep the
# 365d envelope — pulling 3 years of 5m bars (~315k rows/coin) blows up
# fit time without a comparable noise-band benefit, since each holdout
# already has thousands of rows.
#
# Per-tf overrides are env-var-tunable via `ML_LOOKBACK_DAYS_<TF>` (e.g.
# `ML_LOOKBACK_DAYS_1D=1500`). When unset, the lookup falls back to
# `_DEFAULT_LOOKBACK_DAYS_PER_TF` and finally to the global LOOKBACK_DAYS,
# so legacy callers and tests that only set `ML_LOOKBACK_DAYS` continue
# to work unchanged for short timeframes.
_DEFAULT_LOOKBACK_DAYS_PER_TF: dict[str, int] = {
    # 1100 days ≈ 3 years; safely above the 1000-row floor the Phase-2
    # data audit checks for. The backfill_history default for 1d is
    # bumped to match (see scripts/backfill_history.py DEFAULT_DAYS_BY_TF).
    "1d": 1100,
}


def lookback_days_for(timeframe: str) -> int:
    """Resolve the per-timeframe lookback window in days.

    Priority order:
      1. `ML_LOOKBACK_DAYS_<TF>` env var (e.g. ``ML_LOOKBACK_DAYS_1D``).
      2. `_DEFAULT_LOOKBACK_DAYS_PER_TF[tf]` if present.
      3. The global `LOOKBACK_DAYS` (= ``ML_LOOKBACK_DAYS`` or 365).
    """
    env_key = f"ML_LOOKBACK_DAYS_{timeframe.upper()}"
    override = os.environ.get(env_key)
    if override is not None:
        try:
            parsed = int(override)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return _DEFAULT_LOOKBACK_DAYS_PER_TF.get(timeframe, LOOKBACK_DAYS)


# Every advertised timeframe MUST appear in the report — trained when there
# is enough data, otherwise marked `insufficient_data`. Never silently skipped.
DEFAULT_TIMEFRAMES = ["1m", "5m", "1h", "2h", "6h", "1d"]
MIN_TRAIN_ROWS = 80
MIN_PER_COIN_ROWS = 80          # below this we don't try a per-coin model
CALIBRATION_HOLDOUT_FRACTION = 0.2

# Tiny Optuna budget. With <10k rows, more trials chase noise. Both can be
# overridden via env so test suites can shrink the budget further.
OPTUNA_N_TRIALS = int(os.environ.get("ML_OPTUNA_N_TRIALS", "6"))
OPTUNA_TIMEOUT_SECONDS = int(os.environ.get("ML_OPTUNA_TIMEOUT_SECONDS", "30"))
LGB_NUM_BOOST_ROUND = int(os.environ.get("ML_LGB_NUM_BOOST_ROUND", "200"))
LGB_EARLY_STOPPING = 25


# --- Metrics helpers -----------------------------------------------------------
def _multiclass_macro_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """One-vs-rest macro AUC; NaN if fewer than 2 classes present."""
    present = [k for k in range(NUM_CLASSES) if (y_true == k).any() and (y_true != k).any()]
    if len(present) < 2:
        return float("nan")
    aucs = [roc_auc_score((y_true == k).astype(int), proba[:, k]) for k in present]
    return float(np.mean(aucs))


def _multiclass_log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    p = p / p.sum(axis=1, keepdims=True)
    return float(log_loss(y_true, p, labels=list(range(NUM_CLASSES))))


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean per-class Brier score (a.k.a. multiclass Brier)."""
    if len(y_true) == 0:
        return float("nan")
    scores = []
    for k in range(NUM_CLASSES):
        scores.append(brier_score_loss((y_true == k).astype(int), proba[:, k]))
    return float(np.mean(scores))


def _directional_accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    pred = proba.argmax(axis=1)
    return float((pred == y_true).mean())


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    auc: float
    log_loss: float
    brier: float
    directional_accuracy: float
    baseline_auc: float
    baseline_log_loss: float
    baseline_brier: float
    baseline_directional_accuracy: float
    best_params: dict
    # Task #316 — per-fold gain importances for the LightGBM booster, in
    # the same order as the active feature columns. Persisted on the
    # report so the failure-analysis pass can compute rank-correlation
    # of importances across adjacent folds without re-deriving anything.
    feature_importance: Optional[list[float]] = None
    feature_names: Optional[list[str]] = None


# --- Encoding ------------------------------------------------------------------
def _encode_coin_idx(df: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    idx = {c: i for i, c in enumerate(vocab)}
    out = df.copy()
    out["coin_idx"] = out["coin_id"].map(lambda c: idx.get(c, -1)).astype("int32")
    return out


def _prepare_xy(
    df: pd.DataFrame, feature_columns: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    cols = list(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    X = df[cols].copy()
    y = df["label_3class"].to_numpy().astype(int)
    return X, y


# --- Training core -------------------------------------------------------------
def _lgb_params(num_leaves: int, learning_rate: float, min_child_samples: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "deterministic": True,
        "seed": 42,
    }


CLASS_WEIGHT_ALPHA = float(os.environ.get("ML_CLASS_WEIGHT_ALPHA", "1.0"))

# Task #507 — tiny-slice booster recovery. The Task #482 single-T
# calibrator fixed the per-class isotonic rerank failure but left 4
# (coin, tf) slices stuck with raw STABLE-argmax in ``0.0-4.5 %`` —
# below the 5 % `MAX_DIRECTIONAL_CALL_SHARE = 0.95` floor the
# verification gate enforces (`app/training/verification.py:90`):
# bonk@1d, celestia@1d, dogwifcoin@1d, celestia@6h. Experiments under
# `scripts/diagnostic_482/run_507_focused.py` (full grid in
# `reports/<TS>-task507-booster-collapse-verification.md`) showed:
#
#   1. The brief's hypothesis (over-weighted STABLE rows on tiny
#      holdouts) is **the wrong direction**. Sweeping the
#      `_balanced_sample_weight` exponent across `alpha ∈ [0.0, 2.5]`
#      while keeping the default-`_lgb_params` + `LGB_EARLY_STOPPING`
#      path moves raw STABLE-argmax monotonically *up* with alpha
#      (uniform → 0 %, sqrt → 0 %, full-balance → 1.5 %, alpha=2 →
#      6 % on celestia@1d only). Sqrt-dampening makes the stuck
#      slices *worse*, not better.
#
#   2. The actual root cause is `best_iteration == 1` on every tiny
#      slice. LightGBM's `multi_logloss` validation loss spikes after
#      the first tree (the small holdout — n≈67 on 1d — is too noisy
#      for the early-stopping callback's 25-round patience to ride
#      out, so it locks in iteration 1). With one tree, the per-row
#      `P(STABLE)` is bounded by the marginal prior ± a small leaf
#      delta — `max P(STABLE) < 1/3` and STABLE never wins argmax.
#
#   3. The combination that recovers the booster on every tiny slice
#      is **softer hyperparameters + no early stopping + a moderate
#      class-balance boost**: `num_leaves=15, learning_rate=0.05,
#      min_child_samples=20`, fixed `LGB_NUM_BOOST_ROUND` budget,
#      `alpha=2.0`. The softer model can no longer carve a
#      single-leaf STABLE/non-STABLE split that the holdout punishes,
#      so the loss curve keeps decreasing past iteration 1 and the
#      booster gets to express STABLE.
#
# We branch on `n_train` so the recovery path only fires on slices
# small enough that early stopping is the binding lever — `1d` (~264
# rows) and `6h` (~1140) — leaving `1h` / `2h` (~6.9k / ~3.4k) and
# longer-history pooled fits on the original Optuna-driven path so
# their tuned hyperparameters are not silently overridden.
#
# Post-#633 update — the vol-scaled label rollout (Task #459) shifts
# the STABLE/UP/DOWN ratio toward more STABLE rows on 6h+ slots. With
# more STABLE rows in training, the alpha=2 minority-amplification
# recipe Task #507 settled on for n≈264 1d slots over-pushes 6h slots
# (n≈1396 here) into per-coin class collapse: post-#633 6h boosters
# predict DOWN 65–90% of holdout where actual DOWN is ~40%, while the
# pool-trained specialists on the same data hold balanced predictions
# (collapse_gap < 0.03). The cure is *not* a different alpha — Task
# #507's sweep already showed alpha-only knobs cap STABLE-argmax at
# ~6%.
#
# 2026-04-29 update — exporting ML_TINY_SLICE_THRESHOLD=600 was the
# obvious next try (push 6h slots out of the soft path and into the
# regular Optuna + early-stopping path so Optuna can pick the capacity
# for the slice's actual signal density). It was tested end-to-end
# against all 8 MTTM 6h slots. **It does not work.** Per-coin
# directional-accuracy stayed in the 0.39–0.43 band (mean Δ = +0.003,
# 0/8 crossed 0.50), per-coin call-share stayed in 0.91–0.98 for 7/8
# slots (mean Δ = -0.002, collapse signature unchanged, 4 better and
# 3 worse among the failing 7), and log-loss got *worse* on every
# single slot (mean Δ = +0.45 nats, range +0.12 to +0.72 — the
# regular-path booster is now confidently wrong instead of softly
# wrong). Verdict and full per-coin table at
# `models/reports/20260429T182338Z/mttm-6h-retrain-verdict.md`
# (includes a reproducible calc snippet for the aggregate deltas).
# The evidence is that the 6h directional edge under the current
# feature schema + vol-scaled labels is genuinely below 0.50,
# regardless of tiny-slice routing — so this lever is not the right
# place to keep pulling. Default (1500) preserved; ML_TINY_SLICE_THRESHOLD
# env var is intentionally not set in artifact.toml. Future operators:
# do not re-run the threshold-lever experiment without first changing
# the 6h feature set or the vol-scaled label thresholds.
TINY_SLICE_THRESHOLD = int(os.environ.get("ML_TINY_SLICE_THRESHOLD", "1500"))
TINY_SLICE_NUM_LEAVES = int(os.environ.get("ML_TINY_SLICE_NUM_LEAVES", "15"))
TINY_SLICE_LEARNING_RATE = float(os.environ.get("ML_TINY_SLICE_LEARNING_RATE", "0.05"))
TINY_SLICE_MIN_CHILD_SAMPLES = int(os.environ.get("ML_TINY_SLICE_MIN_CHILD_SAMPLES", "20"))
TINY_SLICE_CLASS_WEIGHT_ALPHA = float(
    os.environ.get("ML_TINY_SLICE_CLASS_WEIGHT_ALPHA", "2.0")
)


def _balanced_sample_weight(
    y: np.ndarray, *, alpha: float | None = None,
) -> np.ndarray:
    """Power-balanced inverse-frequency class weights.

    The original Task #95 scheme gave each present class equal total
    mass — the textbook "balanced" recipe (sklearn's
    ``class_weight="balanced"``). That is reproduced here at
    ``alpha = 1`` (the module default, retained after Task #507's
    investigation contradicted the original sqrt-dampening
    hypothesis).

    Power-balancing dampens or amplifies the exponent::

        w_k = (n_total / (K_present * count_k)) ** alpha

    so ``alpha = 0`` is uniform (no rebalancing), ``alpha = 1`` is
    the Task #95 full-balance recipe, and ``alpha > 1`` *amplifies*
    the rare class. The Task #507 tiny-slice booster-recovery path
    in ``_train_lgb`` calls this with ``alpha = 2`` to push the rare
    class hard enough that the booster's marginal STABLE prior on
    very small holdouts crosses the per-row argmax threshold; see
    the ``TINY_SLICE_*`` constants above for the full rationale.

    Missing classes still contribute zero weight. Empty input still
    returns an empty array. The output is NOT renormalized to sum to
    ``len(y)`` because LightGBM is invariant to a global weight scale
    (it shows up only as a constant gradient rescaling).
    """
    if alpha is None:
        alpha = CLASS_WEIGHT_ALPHA
    if len(y) == 0:
        return np.ones(0, dtype=float)
    weights = np.ones(len(y), dtype=float)
    counts = {k: int((y == k).sum()) for k in range(NUM_CLASSES)}
    present = [k for k, n in counts.items() if n > 0]
    if not present:
        return weights
    target = len(y) / len(present)  # full-balance per-row weight numerator
    for k, n in counts.items():
        if n > 0:
            ratio = target / n  # the alpha=1 (full-balance) row weight
            weights[y == k] = ratio ** alpha
    return weights


def _train_lgb(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    params: dict,
) -> tuple[lgb.Booster, float]:
    """Fit a multiclass booster, with a tiny-slice recovery branch.

    For ``n_train < TINY_SLICE_THRESHOLD`` the input ``params`` and the
    standard ``early_stopping`` callback are replaced by the
    tiny-slice recipe documented above the constants — softer
    ``num_leaves`` / ``learning_rate`` / ``min_child_samples``,
    no early stopping, and a more aggressive class-balance boost.
    Larger slices keep their caller-provided params and the standard
    ``LGB_EARLY_STOPPING`` patience.
    """
    n_train = len(X_train)
    is_tiny_slice = n_train < TINY_SLICE_THRESHOLD

    if is_tiny_slice:
        params = _lgb_params(
            num_leaves=TINY_SLICE_NUM_LEAVES,
            learning_rate=TINY_SLICE_LEARNING_RATE,
            min_child_samples=TINY_SLICE_MIN_CHILD_SAMPLES,
        )
        weight = _balanced_sample_weight(
            y_train, alpha=TINY_SLICE_CLASS_WEIGHT_ALPHA,
        )
        callbacks = [lgb.log_evaluation(0)]
    else:
        weight = _balanced_sample_weight(y_train)
        callbacks = [
            lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(0),
        ]

    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
        weight=weight,
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, reference=train_set,
        categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False,
    )
    booster = lgb.train(
        params, train_set,
        num_boost_round=LGB_NUM_BOOST_ROUND,
        valid_sets=[val_set],
        callbacks=callbacks,
    )
    # When early stopping is disabled, ``booster.best_iteration`` is 0,
    # which LightGBM's ``predict`` interprets as "use all trees" — so
    # the same call works for both branches.
    val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_loss = _multiclass_log_loss(y_val, val_pred)
    return booster, val_loss


def _optuna_search_lgb(
    X_train: pd.DataFrame, y_train: np.ndarray,
) -> tuple[dict, lgb.Booster]:
    """Tiny Optuna study with TPE sampler. Inner train/val split is the last
    20% of the (already chronological) training fold — never peeks at the
    walk-forward test fold.
    """
    n = len(X_train)
    cut = max(1, int(n * 0.8))
    if cut >= n:
        cut = n - 1
    X_tr, X_va = X_train.iloc[:cut], X_train.iloc[cut:]
    y_tr, y_va = y_train[:cut], y_train[cut:]

    if len(X_va) == 0 or len(np.unique(y_va)) < 2:
        params = _lgb_params(15, 0.1, 5)
        booster, _ = _train_lgb(X_train, y_train, X_train, y_train, params)
        return params, booster

    # Test/CI escape hatch: skip the search and use a single sane config so
    # the suite finishes inside the shell timeout. Production runs leave
    # this unset.
    if os.environ.get("ML_SKIP_OPTUNA") == "1":
        params = _lgb_params(31, 0.1, 5)
        booster, _ = _train_lgb(X_train, y_train, X_va, y_va, params)
        return params, booster

    def _objective(trial: optuna.Trial) -> float:
        params = _lgb_params(
            num_leaves=trial.suggest_categorical("num_leaves", [15, 31, 63]),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
        )
        try:
            _, loss = _train_lgb(X_tr, y_tr, X_va, y_va, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("optuna_trial_failed", extra={"params": params, "error": str(exc)})
            return float("inf")
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        _objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT_SECONDS,
        show_progress_bar=False,
    )
    best = study.best_params
    best_params = _lgb_params(best["num_leaves"], best["learning_rate"], best["min_child_samples"])
    final_booster, _ = _train_lgb(X_train, y_train, X_va, y_va, best_params)
    return best_params, final_booster


def _regressor_params() -> dict:
    """LightGBM params for the magnitude head.

    Task #135 — the classifier alone produced a near-degenerate
    `expectedReturnPct` distribution (5m OOS p95 ≈ 0.11%) because
    `expRet = sum(p_k * mean_pct_k)` is shrunk by p_stable (~0.55 mean)
    and the per-class means are clipped to ±label_threshold (~0.10% on 5m).
    A separate L2 regressor over non-stable rows learns the *magnitude*
    of the move when something happens, decoupled from the diluting
    p_stable mass.

    Target choice: `|forward_return| * 100` (always positive). Predicting
    SIGNED returns directly collapses to a near-constant prediction
    because the conditional mean of forward_return given features is
    close to zero (up and down moves balance), so an L1/L2 booster fits
    a flat tree and early-stops at iteration 1. Targeting magnitude
    sidesteps that — magnitude is large and positively correlated with
    realized vol/ATR-like features. Sign comes from the classifier's
    `p_up - p_down` at inference time.
    """
    return {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "deterministic": True,
        "seed": 42,
    }


# Below this many non-stable rows the regression head is skipped (the
# trainer falls back to the legacy probability-weighted expectation).
# 30 is the verdict's minimum-trade bar from task #130 — below it any
# regressor we fit is just memorizing noise.
MIN_REGRESSOR_ROWS = 30


def _directional_horizon_returns(df: pd.DataFrame) -> np.ndarray:
    """Per-row signed forward return at the SAME horizon as `label_3class`.

    Task #379 — when the directional-label horizon spans multiple bars
    (1h/2h/6h), `forward_return` (1-bar) is no longer the magnitude that
    matches the classifier's target. The labelled frame carries
    `directional_label_forward_return` (fractional, multi-bar) so the
    regressor head and per-class return means can stay coherent with the
    classifier. For legacy 1-bar timeframes this column equals
    `forward_return` (labels.py emits the same fraction), so the call is
    backward-compatible. Pre-#379 frames without the column fall back to
    `forward_return` exactly.
    """
    if "directional_label_forward_return" in df.columns:
        col = df["directional_label_forward_return"].astype(float)
        if col.notna().all():
            return col.to_numpy()
        fr = df["forward_return"].astype(float)
        return col.fillna(fr).to_numpy()
    return df["forward_return"].astype(float).to_numpy()


def _train_regressor_head(
    df: pd.DataFrame, vocab: list[str],
    feature_columns: Optional[Sequence[str]] = None,
) -> tuple[Optional[lgb.Booster], Optional[dict]]:
    """Fit a LightGBM regressor on the directional-horizon return (in %)
    over **non-stable** rows only. Returns (booster, holdout_stats) or
    (None, None) if there isn't enough non-stable data.

    The non-stable filter is the key trick: training on every row would
    drag predictions toward 0 (because ~55% of rows are STABLE with
    ~0% returns). Filtering to non-stable rows teaches the model what a
    real move looks like; combined with the classifier's STABLE prob the
    downstream gate naturally vetoes flat-market signals.

    Task #379 — the magnitude target uses
    `directional_label_forward_return` (the multi-bar return at 1h/2h/6h,
    1-bar elsewhere) so the magnitude head and the directional classifier
    are anchored to the same forward window. Falls back to `forward_return`
    on legacy frames that pre-date the column.
    """
    df = df.copy()
    df = _encode_coin_idx(df, vocab)
    nonstab = df[df["label_3class"] != 1].copy()
    if len(nonstab) < MIN_REGRESSOR_ROWS:
        return None, None

    nonstab = nonstab.sort_values("timestamp_ms").reset_index(drop=True)
    cols = list(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    X = nonstab[cols].copy()
    # Magnitude target — directional-horizon return so it matches the
    # window the classifier's UP/STABLE/DOWN label spans (task #379).
    y = np.abs(_directional_horizon_returns(nonstab)) * 100.0

    # Chronological 80/20 holdout — same shape as the classifier's
    # calibration tail so the persisted stats are directly comparable.
    cut = max(1, int(len(X) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
    if cut >= len(X) - 3:
        X_tr, y_tr = X, y
        X_va, y_va = X, y
    else:
        X_tr, y_tr = X.iloc[:cut], y[:cut]
        X_va, y_va = X.iloc[cut:], y[cut:]

    params = _regressor_params()
    train_set = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_va, label=y_va, reference=train_set,
        categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False,
    )
    booster = lgb.train(
        params, train_set,
        num_boost_round=LGB_NUM_BOOST_ROUND,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(0)],
    )

    pred_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    abs_pred = np.abs(pred_va)
    stats = {
        "n_train_rows": int(len(X_tr)),
        "n_holdout_rows": int(len(X_va)),
        "abs_pred_p50_pct": float(np.median(abs_pred)),
        "abs_pred_p95_pct": float(np.quantile(abs_pred, 0.95)),
        "abs_pred_max_pct": float(np.max(abs_pred)),
        "mae_pct": float(np.mean(np.abs(pred_va - y_va))),
        "best_iteration": int(booster.best_iteration or 0),
    }
    return booster, stats


def _gate_alignment_summary(
    probs: np.ndarray,
    magnitudes_pct: np.ndarray,
    mde: float,
    mer: float,
    source: str,
) -> Optional[dict]:
    """Joint distribution of the sign head (classifier) vs the magnitude
    head (regressor) on a sample, scored against the live decision gates
    (`min_directional_edge` and `min_expected_return_pct`).

    The "aligned" buckets are the cases where both heads agree on whether
    we should trade — either both above their respective gates ("loud") or
    both below ("quiet"). The other two buckets are the disagreements that
    motivated task #147:

      - `loud_classifier_quiet_regressor`: classifier picks UP/DOWN with
        edge >= mde, but the regressor magnitude is below the cost floor.
        The downstream gate will skip — the classifier's confidence is
        "wasted budget".
      - `quiet_classifier_loud_regressor`: regressor magnitude clears the
        cost floor, but |p_up - p_down| < mde so the directional gate is
        not confident enough to take the trade. The regressor's signal is
        also "wasted budget".

    Returns None if the input is empty or the inputs disagree on length —
    callers should treat that as "diagnostic not available", same as the
    regression-head stats.
    """
    n = len(probs)
    if n == 0 or len(magnitudes_pct) != n:
        return None
    dir_edge = np.abs(probs[:, 2] - probs[:, 0])
    abs_mag = np.abs(magnitudes_pct)
    cls_loud = dir_edge >= mde
    reg_loud = abs_mag >= mer
    aligned_loud = int((cls_loud & reg_loud).sum())
    aligned_quiet = int((~cls_loud & ~reg_loud).sum())
    loud_cls_quiet_reg = int((cls_loud & ~reg_loud).sum())
    quiet_cls_loud_reg = int((~cls_loud & reg_loud).sum())
    aligned = aligned_loud + aligned_quiet
    total = float(n)
    return {
        "n": int(n),
        "source": source,
        "min_directional_edge": float(mde),
        "min_expected_return_pct": float(mer),
        "aligned_share": aligned / total,
        "aligned_loud_share": aligned_loud / total,
        "aligned_quiet_share": aligned_quiet / total,
        "loud_classifier_quiet_regressor_share": loud_cls_quiet_reg / total,
        "quiet_classifier_loud_regressor_share": quiet_cls_loud_reg / total,
        "dir_edge_p50": float(np.median(dir_edge)),
        "dir_edge_p95": float(np.quantile(dir_edge, 0.95)),
        "abs_magnitude_pct_p50": float(np.median(abs_mag)),
        "abs_magnitude_pct_p95": float(np.quantile(abs_mag, 0.95)),
    }


def _train_baseline(
    X_train: pd.DataFrame, y_train: np.ndarray,
) -> tuple[OneHotEncoder, LogisticRegression, np.ndarray]:
    """Multinomial logistic regression on numeric features + one-hot coin.

    Returns (encoder, lr, fallback_priors) where fallback_priors is a (3,)
    array used when the encoder was fit on a single class.

    Task #633 — sklearn's `LogisticRegression` does not accept NaN inputs
    natively. After the NaN-as-zero cutover, the external-stream feature
    columns (funding_rate, liquidations, BTC/ETH/SOL liq pulses, …)
    legitimately carry NaN on the share of historical rows where no
    provider snapshot exists. We impute those NaNs with 0.0 *only for
    the LR baseline* — the LightGBM booster keeps the raw NaN signal so
    its `use_missing` path can split on "is this row missing?". The
    baseline historically saw 0.0 in exactly these slots, so the
    impute-to-zero choice keeps the baseline's served behaviour
    bit-identical to its pre-#633 form (no surprise lift / regress on
    the booster-vs-baseline served gate).
    """
    coin_idx = X_train[["coin_idx"]].to_numpy()
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    coin_oh = enc.fit_transform(coin_idx)
    numeric = X_train.drop(columns=CATEGORICAL_FEATURES).to_numpy()
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
    Xb = np.hstack([numeric, coin_oh])

    priors = np.zeros(NUM_CLASSES)
    for k in range(NUM_CLASSES):
        priors[k] = float((y_train == k).mean()) if len(y_train) else 1.0 / NUM_CLASSES
    if priors.sum() == 0:
        priors[:] = 1.0 / NUM_CLASSES
    priors /= priors.sum()

    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    if len(np.unique(y_train)) < 2:
        lr.coef_ = np.zeros((1, Xb.shape[1]))
        lr.intercept_ = np.array([0.0])
        lr.classes_ = np.array(np.unique(y_train) if len(y_train) else [1])
    else:
        lr.fit(Xb, y_train)
    return enc, lr, priors


# Task #400 — promotion gate constants used to decide whether the
# multinomial-logistic baseline (`_train_baseline` above) should be
# REGISTERED as the slice's served predictor. Sourced from
# `app.training.verification` so the trainer's served-baseline gate
# stays bit-identical to the watchdog's promotion gate (the whole
# point: a slice we register as baseline-served must clear the same
# bar a booster-served slice would have to clear). Imported lazily at
# call time inside `_should_serve_baseline` to avoid a circular import
# (`verification` imports nothing from `train`, but
# `failure_analysis`/sibling modules sometimes do).
def _should_serve_baseline(
    booster_directional_accuracy: float,
    baseline_directional_accuracy: float,
    total_holdout_rows: int,
) -> bool:
    """Decide whether the multinomial-logistic baseline should be
    registered as the slice's *served* predictor (Task #400).

    Returns True iff ALL of the following hold:
      1. The baseline beat the booster on directional accuracy
         (mean of walk-forward folds, same source the verification
         gate already reads).
      2. The baseline itself would clear the verification watchdog's
         coin-flip floor (DA > MIN_DIRECTIONAL_ACCURACY = 0.50).
      3. The summed walk-forward holdout meets the minimum sample
         size (sum of fold n_test >= MIN_HOLDOUT_ROWS = 200) so the
         comparison itself is statistically meaningful.

    Without this gate a slice whose booster lost head-to-head with
    the LR baseline used to be silently retired by the watchdog,
    throwing away a working baseline along with the booster. Now the
    baseline gets registered as the served predictor for that slice
    (`served_predictor_kind="baseline"` on the manifest) and the
    verification gate promotes it on its own merits.
    """
    from .verification import MIN_DIRECTIONAL_ACCURACY, MIN_HOLDOUT_ROWS

    try:
        booster_da = float(booster_directional_accuracy)
        baseline_da = float(baseline_directional_accuracy)
    except (TypeError, ValueError):
        return False
    # Reject NaN sentinels emitted by `_mean` when every fold's metric
    # was NaN — neither side has a real number to compare.
    if booster_da != booster_da or baseline_da != baseline_da:  # noqa: PLR0124
        return False
    if baseline_da <= booster_da:
        return False
    if baseline_da <= MIN_DIRECTIONAL_ACCURACY:
        return False
    if int(total_holdout_rows) < MIN_HOLDOUT_ROWS:
        return False
    return True


def _baseline_predict(
    enc: OneHotEncoder, lr: LogisticRegression, priors: np.ndarray, X: pd.DataFrame,
) -> np.ndarray:
    coin_idx = X[["coin_idx"]].to_numpy()
    coin_oh = enc.transform(coin_idx)
    numeric = X.drop(columns=CATEGORICAL_FEATURES).to_numpy()
    # Task #633 — mirror the impute-to-zero applied at fit time so the
    # served LR baseline does not raise on NaN inputs from the post-#633
    # external-stream NaN-default path. See `_train_baseline` for the
    # rationale.
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
    Xb = np.hstack([numeric, coin_oh])
    if not hasattr(lr, "n_features_in_") or lr.n_features_in_ != Xb.shape[1] or len(lr.classes_) < 2:
        return np.tile(priors, (len(X), 1))
    proba_lr = lr.predict_proba(Xb)
    full = np.zeros((len(X), NUM_CLASSES))
    for j, cls in enumerate(lr.classes_):
        full[:, int(cls)] = proba_lr[:, j]
    # Fill in missing classes with a tiny epsilon and renormalize.
    full = np.clip(full, 1e-6, 1.0)
    full /= full.sum(axis=1, keepdims=True)
    return full


def _calibrate_per_class(
    cal_pred_raw: np.ndarray, y_cal: np.ndarray,
) -> tuple[Optional[list[TemperatureScaledClass]], np.ndarray, list[dict]]:
    """Calibrate the booster's raw multinomial probabilities with
    single-scalar temperature scaling, then wrap each class as a
    `TemperatureScaledClass` so the inference path can apply the
    calibrator one column at a time (existing contract). Returns the
    list of per-class wrappers, the calibrated+renormalized
    probability matrix on the holdout, and a per-class reliability
    diagram suitable for the report.

    Task #482 — the legacy implementation fit per-class
    `IsotonicRegression` on `(raw_p_k, y == k)` independently and then
    renormalized rows to sum to one. That recipe destroys the
    booster's cross-class argmax ranking: raw `[0.30, 0.60, 0.10]`
    can become `[0.40, 0.25, 0.30]` because the per-class isotonic
    curves don't respect each other. Across the fleet diagnostic at
    `scripts/diagnostic_482/run_stage_collapse_diagnostic.py`, this
    surfaced as 33/40 (coin, tf) slices with raw STABLE-argmax
    20-90% and calibrated STABLE-argmax 0-3%, driving
    `directional_call_share` to ~1.0 and tripping the verification
    gate.

    Single-scalar temperature scaling (Guo et al. 2017) — `cal[k] =
    raw[k] ** (1 / T)` followed by per-row sum-to-one — solves this by
    construction: applying the same monotone exponent to every class
    leaves the row argmax invariant. So the calibrator can only
    sharpen / flatten the booster's confidence; it can NEVER re-rank
    classes within a row. With `T = 1` it's exactly identity. The
    scalar `T` is fit on the calibration tail by maximizing
    multinomial log-likelihood. Vector / per-class temperature
    scaling was rejected because the optimizer hits its bounds and
    extreme per-class exponents reproduce the calibrator-collapse
    failure mode (the booster trained with `_balanced_sample_weight`
    over-predicts STABLE on average — the natural class mass is
    ~16/42/42 — and any per-class fit on the natural-distribution
    holdout sharpens STABLE down hard enough to lose argmax).

    The function still returns one wrapper per class with a
    `.predict(p_k_raw)` contract; `app/main.py` and
    `app/backtest/walk_forward_oos.py` continue to call
    `calibrators[k].predict(...)` and sum-to-one per row, which
    reconstructs `softmax(logits / T)` exactly.
    """
    if len(np.unique(y_cal)) < 2:
        return None, cal_pred_raw, []

    inv_T = fit_single_temperature(cal_pred_raw, y_cal)
    # Single-scalar temperature: every class wrapper carries the SAME
    # exponent so the per-row argmax is invariant.
    calibrators = [TemperatureScaledClass(inv_T) for _ in range(NUM_CLASSES)]
    cal_calibrated = apply_single_temperature(cal_pred_raw, inv_T)

    # Per-class reliability diagram bins.
    n_bins = 5
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    diagram: list[dict] = []
    class_names = ["DOWN", "STABLE", "UP"]
    for k in range(NUM_CLASSES):
        ck = (y_cal == k).astype(int)
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (cal_calibrated[:, k] >= lo) & (
                cal_calibrated[:, k] < hi if b < n_bins - 1 else cal_calibrated[:, k] <= hi
            )
            if mask.sum() == 0:
                continue
            diagram.append({
                "class": class_names[k],
                "bin_low": float(lo), "bin_high": float(hi),
                "mean_predicted": float(cal_calibrated[mask, k].mean()),
                "fraction_positive": float(ck[mask].mean()),
                "n": int(mask.sum()),
            })
    return calibrators, cal_calibrated, diagram


# --- Task #316 — per-class / calibration / regime / PnL helpers --------------
# These compute the surface the failure-analysis pass needs to bucket every
# red slice without re-deriving anything from persisted artifacts. All
# helpers are pure (numpy + pandas only) so they're trivial to test.
_CLASS_NAMES = ["DOWN", "STABLE", "UP"]


def _per_class_breakdown(y: np.ndarray) -> dict[str, int]:
    """Row count per class. Always emits all three keys so a downstream
    consumer can do `bd['DOWN']` without an existence check, even when a
    class has zero rows on the slice.
    """
    return {_CLASS_NAMES[k]: int((y == k).sum()) for k in range(NUM_CLASSES)}


def _per_class_accuracy(
    y_true: np.ndarray, proba: np.ndarray,
) -> dict[str, dict]:
    """Per-class recall on the holdout: for each true class, the fraction
    of rows the calibrated model predicted as that class. Always emits
    all three keys so downstream consumers can read
    `acc['UP']['accuracy']` without existence checks; `accuracy` is
    `None` for classes with zero holdout rows.
    """
    out: dict[str, dict] = {}
    if len(y_true) == 0:
        return {cls: {"n": 0, "accuracy": None} for cls in _CLASS_NAMES}
    pred = proba.argmax(axis=1)
    for k in range(NUM_CLASSES):
        mask = y_true == k
        n = int(mask.sum())
        if n == 0:
            out[_CLASS_NAMES[k]] = {"n": 0, "accuracy": None}
        else:
            out[_CLASS_NAMES[k]] = {
                "n": n,
                "accuracy": round(float((pred[mask] == k).mean()), 4),
            }
    return out


def _class_balance_drift(
    train_counts: dict[str, int], holdout_counts: dict[str, int],
) -> dict[str, float]:
    """Per-class share difference (holdout - train), plus the L1 sum.
    The L1 score is what flags the "regime shifted between train and
    holdout" failure mode in the failure-analysis pass.
    """
    n_tr = sum(train_counts.values()) or 1
    n_ho = sum(holdout_counts.values()) or 1
    out: dict[str, float] = {}
    l1 = 0.0
    for cls in _CLASS_NAMES:
        d = (holdout_counts[cls] / n_ho) - (train_counts[cls] / n_tr)
        out[cls] = round(float(d), 4)
        l1 += abs(d)
    out["l1_drift"] = round(float(l1), 4)
    return out


def _per_class_brier(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Per-class Brier score (one-vs-rest). Returns NaN for the whole
    block when the holdout itself is empty, and NaN for any class that
    has zero positives in the holdout (sklearn computes a degenerate
    score on a single-label vector that isn't meaningful for our use).
    """
    out: dict[str, float] = {}
    if len(y_true) == 0:
        return {cls: float("nan") for cls in _CLASS_NAMES}
    for k in range(NUM_CLASSES):
        ck = (y_true == k).astype(int)
        if int(ck.sum()) == 0:
            out[_CLASS_NAMES[k]] = float("nan")
            continue
        out[_CLASS_NAMES[k]] = round(
            float(brier_score_loss(ck, proba[:, k])), 4
        )
    return out


def _reliability_max_dev_per_class(
    diagram: list[dict],
) -> dict[str, float]:
    """Max |mean_predicted - fraction_positive| per class across the
    reliability bins. The failure-analysis pass treats values >= 0.10 as
    "calibration broken". Returns NaN for classes with no diagram bins
    (insufficient holdout data for that class).
    """
    out: dict[str, float] = {cls: float("nan") for cls in _CLASS_NAMES}
    by_class: dict[str, list[float]] = {cls: [] for cls in _CLASS_NAMES}
    for entry in diagram or []:
        cls = entry.get("class")
        if cls in by_class:
            by_class[cls].append(
                abs(float(entry["mean_predicted"]) - float(entry["fraction_positive"]))
            )
    for cls, devs in by_class.items():
        if devs:
            out[cls] = round(max(devs), 4)
    return out


# Confidence buckets the failure-analysis pass expects, in (lo, hi) form.
# The lowest bucket starts at 1/NUM_CLASSES (the smallest possible top
# probability under uniform random) so an inverted argmax never lands
# outside the table.
_CONFIDENCE_BUCKETS: list[tuple[float, float]] = [
    (1.0 / NUM_CLASSES, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 1.0),
]


def _confidence_bucket_da(
    proba: np.ndarray, y_true: np.ndarray,
) -> list[dict]:
    """Directional accuracy bucketed by the top predicted probability.
    Buckets are [0.33, 0.5), [0.5, 0.6), [0.6, 0.7), [0.7, 0.8), [0.8, 1.0]
    matching the surface the failure-analysis pass queries.
    """
    out: list[dict] = []
    n = len(y_true)
    if n == 0:
        return out
    pred = proba.argmax(axis=1)
    top = proba.max(axis=1)
    correct = (pred == y_true).astype(int)
    for i, (lo, hi) in enumerate(_CONFIDENCE_BUCKETS):
        if i < len(_CONFIDENCE_BUCKETS) - 1:
            mask = (top >= lo) & (top < hi)
        else:
            mask = (top >= lo) & (top <= hi)
        m = int(mask.sum())
        out.append({
            "lo": round(float(lo), 2),
            "hi": round(float(hi), 2),
            "n": m,
            "share": round(m / n, 4) if n else 0.0,
            "da": round(float(correct[mask].mean()), 4) if m else None,
        })
    return out


def _predicted_class_entropy(proba: np.ndarray) -> float:
    """Mean Shannon entropy (natural log) of the per-row predicted
    distribution. Low entropy ≈ the model is confident on every row;
    high entropy (≈ ln(NUM_CLASSES)) ≈ near-uniform / "no opinion".
    """
    if len(proba) == 0:
        return float("nan")
    p = np.clip(proba, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    return round(float(ent.mean()), 4)


def _prediction_collapse(
    proba: np.ndarray,
    y_true: np.ndarray,
    baseline_proba: Optional[np.ndarray] = None,
) -> dict:
    """Predicted-class distribution vs the empirical class prior on the
    holdout, with the top-class share gap that the failure-analysis pass
    uses to flag prediction collapse (gap >= 0.15 OR predicted top-class
    share >= 0.85).

    `collapse_gap` is the model top-class share minus the **label** (=
    class prior) top-class share — i.e. "how much more concentrated than
    chance". When `baseline_proba` is provided, also exposes
    `top_class_share_gap_vs_baseline`, the model top-class share minus
    the **baseline LR model's** top-class share on the same rows. The
    failure-analysis pass uses the latter to attribute collapse to the
    LightGBM head specifically (vs an inherited prior from the data).
    """
    n = len(y_true)
    if n == 0:
        zeros = {cls: 0.0 for cls in _CLASS_NAMES}
        return {
            "predicted_class_share": zeros,
            "label_class_share": zeros,
            "predicted_top_class_share": 0.0,
            "label_top_class_share": 0.0,
            "collapse_gap": 0.0,
            "baseline_top_class_share": None,
            "top_class_share_gap_vs_baseline": None,
        }
    pred = proba.argmax(axis=1)
    pred_share = {
        _CLASS_NAMES[k]: round(float((pred == k).mean()), 4)
        for k in range(NUM_CLASSES)
    }
    label_share = {
        _CLASS_NAMES[k]: round(float((y_true == k).mean()), 4)
        for k in range(NUM_CLASSES)
    }
    pred_top = float(max(pred_share.values()))
    label_top = float(max(label_share.values()))
    out = {
        "predicted_class_share": pred_share,
        "label_class_share": label_share,
        "predicted_top_class_share": round(pred_top, 4),
        "label_top_class_share": round(label_top, 4),
        "collapse_gap": round(pred_top - label_top, 4),
        "baseline_top_class_share": None,
        "top_class_share_gap_vs_baseline": None,
    }
    if baseline_proba is not None and len(baseline_proba) == n:
        base_pred = baseline_proba.argmax(axis=1)
        base_share = {
            _CLASS_NAMES[k]: float((base_pred == k).mean())
            for k in range(NUM_CLASSES)
        }
        base_top = float(max(base_share.values()))
        out["baseline_top_class_share"] = round(base_top, 4)
        out["top_class_share_gap_vs_baseline"] = round(pred_top - base_top, 4)
    return out


# Task #337 — bucket sizes used by the inter-arrival cadence audit so a
# slice that mixes cadences without being caught by the per-row stamps
# (`bars_by_native_cadence`) still trips `contamination_flag`. Mirrors
# the table in `scripts/compute_failure_metrics.cadence_audit`.
_CADENCE_AUDIT_BUCKET_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "6h": 21_600_000,
    "1d": 86_400_000,
}

# Task #337 — L1 distance threshold for "predicted distribution within
# epsilon of the training class prior". Matches the default in
# `scripts/compute_failure_metrics.predictions_near_prior_share`.
_PREDICTIONS_NEAR_PRIOR_EPS = 0.05


def _compute_contamination_flag(df: pd.DataFrame, timeframe: str) -> bool:
    """True when the assembled slice carries a contamination signal that
    `manifest.cadence_mixed` (which only fires on >1
    `bars_by_native_cadence` keys) cannot see:

      * any row stamped with a synthetic-source label, or
      * inter-arrival timestamp gaps that look like mixed cadence — same
        rule as `scripts/compute_failure_metrics.cadence_audit`
        (p95 >= 2 × bucket AND p50 < 0.5 × bucket).

    For the pooled slice we run the gap audit per `coin_id` so cross-
    coin timestamp interleaving doesn't create a false positive.
    """
    if "bars_source" in df.columns:
        for s in df["bars_source"].dropna().astype(str).unique():
            if "synthetic" in s.lower():
                return True
    bucket_ms = _CADENCE_AUDIT_BUCKET_MS.get(timeframe)
    if not bucket_ms or "timestamp_ms" not in df.columns:
        return False
    if "coin_id" in df.columns:
        groups = (g for _, g in df.groupby("coin_id", sort=False))
    else:
        groups = [df]
    for g in groups:
        if len(g) < 3:
            continue
        ts = np.sort(g["timestamp_ms"].astype(np.int64).to_numpy())
        gaps = np.diff(ts)
        if len(gaps) == 0:
            continue
        p50 = float(np.median(gaps))
        p95 = float(np.percentile(gaps, 95))
        if (p95 >= 2 * bucket_ms) and (p50 < bucket_ms * 0.5):
            return True
    return False


def _predictions_near_prior(
    proba: np.ndarray,
    train_y: np.ndarray,
    eps: float = _PREDICTIONS_NEAR_PRIOR_EPS,
) -> dict:
    """Share of holdout rows whose predicted class distribution is within
    `eps` (L1) of the training class prior. High share = the calibrated
    head has effectively learned to output the prior on most rows; the
    failure-analysis pass uses this to flag a calibrated-but-collapsed
    head as `structurally_noisy_retire`. Mirrors
    `scripts/compute_failure_metrics.predictions_near_prior_share`.
    """
    if len(proba) == 0 or len(train_y) == 0:
        return {"share_within_eps": 0.0, "eps": float(eps)}
    prior = np.array(
        [float((train_y == k).mean()) for k in range(NUM_CLASSES)],
        dtype=float,
    )
    diff = np.abs(proba - prior[None, :]).sum(axis=1)
    return {
        "share_within_eps": round(float((diff <= eps).mean()), 4),
        "eps": float(eps),
    }


def _regime_bucketed_da(
    regimes: pd.Series, proba: np.ndarray, y_true: np.ndarray,
) -> dict[str, dict]:
    """DA stratified by the `regime` label on each holdout row. The
    `regime` column is stamped by `labels.py` (Phase 2). Rows with NaN
    regime are bucketed under "_unknown".
    """
    out: dict[str, dict] = {}
    n = len(y_true)
    if n == 0 or regimes is None or len(regimes) != n:
        return out
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(int)
    labels = regimes.fillna("_unknown").astype(str).to_numpy()
    for r in sorted(set(labels.tolist())):
        mask = labels == r
        m = int(mask.sum())
        if m == 0:
            continue
        out[r] = {
            "n": m,
            "share": round(m / n, 4),
            "da": round(float(correct[mask].mean()), 4),
        }
    return out


def _holdout_pnl_after_fees(
    proba: np.ndarray,
    forward_returns: np.ndarray,
    *,
    min_directional_edge: float,
    min_expected_return_pct: float,
    round_trip_cost_pct: float,
    magnitudes_pct: Optional[np.ndarray] = None,
) -> dict:
    """Apply the live entry rule to the calibrated holdout and return
    realized PnL after the production round-trip cost.

    Entry rule (mirrors `quant-brain.ts` / the live decision rule):
      1. argmax != STABLE
      2. |p_up - p_down| >= min_directional_edge
      3. expected magnitude >= min_expected_return_pct (only enforced when
         `magnitudes_pct` is provided — i.e. the slice has a regression head)

    Trade direction = sign(p_up - p_down). Gross PnL per trade =
    direction * forward_return * 100 (percent). Net = gross minus the
    contract round-trip cost.
    """
    n = len(forward_returns)
    if n == 0:
        return {
            "n_trades": 0, "trade_share": 0.0, "gross_pct_mean": 0.0,
            "round_trip_cost_pct": round(float(round_trip_cost_pct), 4),
            "net_pct_mean": 0.0, "net_pct_total": 0.0,
            "win_rate": None,
        }
    p_down = proba[:, 0]
    p_up = proba[:, 2]
    edge = p_up - p_down
    argmax = proba.argmax(axis=1)
    take = (argmax != 1) & (np.abs(edge) >= float(min_directional_edge))
    if magnitudes_pct is not None and len(magnitudes_pct) == n:
        take = take & (np.abs(magnitudes_pct) >= float(min_expected_return_pct))
    n_trades = int(take.sum())
    if n_trades == 0:
        return {
            "n_trades": 0, "trade_share": 0.0, "gross_pct_mean": 0.0,
            "round_trip_cost_pct": round(float(round_trip_cost_pct), 4),
            "net_pct_mean": 0.0, "net_pct_total": 0.0,
            "win_rate": None,
        }
    direction = np.sign(edge[take])
    direction[direction == 0] = 1.0  # tie-break: treat as long
    gross_pct = direction * np.asarray(forward_returns, dtype=float)[take] * 100.0
    net_pct = gross_pct - float(round_trip_cost_pct)
    # Win rate = share of simulated trades whose net % return is strictly
    # positive after fees. Task #344 — was previously omitted, leaving the
    # diagnostics UI permanently rendering an em-dash. Computed on the same
    # `take` mask the rest of the post-fee metrics use, so it is directly
    # comparable to `net_pct_mean` and `n_trades`.
    win_rate = float((net_pct > 0).mean())
    return {
        "n_trades": n_trades,
        "trade_share": round(n_trades / n, 4),
        "gross_pct_mean": round(float(gross_pct.mean()), 4),
        "round_trip_cost_pct": round(float(round_trip_cost_pct), 4),
        "net_pct_mean": round(float(net_pct.mean()), 4),
        "net_pct_total": round(float(net_pct.sum()), 4),
        "win_rate": round(win_rate, 4),
    }


def _feature_importance_rank_corr_across_folds(
    folds: Sequence["FoldResult"],
) -> dict:
    """Pairwise Spearman rank correlation of LightGBM gain importances
    between adjacent walk-forward folds, plus the mean. The failure
    analysis treats mean rank-corr < 0.5 as "feature importance is
    unstable across folds" (corroborating evidence that the slice is
    learning fold-specific noise rather than a stable signal).
    """
    pairs: list[dict] = []
    rank_corrs: list[float] = []
    importances = [
        np.asarray(f.feature_importance, dtype=float)
        for f in folds
        if f.feature_importance is not None
    ]
    if len(importances) < 2:
        return {
            "rank_corr_pairwise_mean": None,
            "n_pairs": 0,
            "pairs": [],
        }
    for i in range(len(importances) - 1):
        a, b = importances[i], importances[i + 1]
        if a.shape != b.shape or a.size == 0:
            continue
        ra = pd.Series(a).rank(method="average").to_numpy()
        rb = pd.Series(b).rank(method="average").to_numpy()
        if np.std(ra) == 0 or np.std(rb) == 0:
            continue
        rho = float(np.corrcoef(ra, rb)[0, 1])
        pairs.append({"fold_a": i, "fold_b": i + 1, "rank_corr": round(rho, 4)})
        rank_corrs.append(rho)
    mean_rho = round(float(np.mean(rank_corrs)), 4) if rank_corrs else None
    return {
        "rank_corr_pairwise_mean": mean_rho,
        "n_pairs": len(rank_corrs),
        "pairs": pairs,
    }


def _class_return_means_pct(df: pd.DataFrame) -> list[float]:
    """Mean directional-horizon return (in %) per class. Used at inference
    to compute expectedReturnPct from calibrated probs. Falls back to the
    timeframe threshold if a class has no rows.

    Task #379 — uses `directional_label_forward_return` so the per-class
    expected-return semantics match the horizon at which the classifier's
    label was assigned. For 1-bar legacy timeframes this column equals
    `forward_return` so the means are unchanged. Falls back to
    `forward_return` on pre-#379 frames that lack the column.
    """
    returns = _directional_horizon_returns(df)
    label = df["label_3class"].to_numpy()
    out: list[float] = []
    for k in range(NUM_CLASSES):
        mask = label == k
        if not mask.any():
            out.append(0.0)  # filled in by caller using threshold
        else:
            out.append(float(np.mean(returns[mask]) * 100.0))
    return out


def _feature_schema_hash(feature_names: list[str]) -> str:
    """Short content hash of the FEATURE_COLUMNS contract — lets a registry
    consumer (and the diagnostics card) tell at a glance whether two
    manifests were trained against the same input schema. Phase 3.
    """
    import hashlib
    h = hashlib.sha1("|".join(feature_names).encode("utf-8")).hexdigest()
    return h[:12]


def _training_window_iso(df: pd.DataFrame) -> Optional[dict]:
    """Earliest/latest `timestamp_ms` in `df` rendered as ISO8601 strings
    so the manifest captures the temporal coverage of the slice without
    keeping the parquet in scope.
    """
    if "timestamp_ms" not in df.columns or df.empty:
        return None
    try:
        ts_lo = int(df["timestamp_ms"].min())
        ts_hi = int(df["timestamp_ms"].max())
    except Exception:
        return None
    return {
        "start": datetime.fromtimestamp(ts_lo / 1000.0, tz=timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(ts_hi / 1000.0, tz=timezone.utc).isoformat(),
    }


def train_one_slice(
    df: pd.DataFrame, coin_id: str, timeframe: str, vocab: list[str],
    *, note: str = "",
    specialist_kind: Optional[str] = None,
    regime_subset: Optional[list[str]] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> dict:
    """Run walk-forward + final-fit + per-class calibration for one slice
    of data. `coin_id` is the registry key — pass POOLED_COIN_ID for the
    cross-coin model.
    """
    if df.empty or len(df) < MIN_TRAIN_ROWS:
        return {
            "coin_id": coin_id, "timeframe": timeframe, "status": "insufficient_data",
            "n_rows": int(len(df)), "min_rows_required": MIN_TRAIN_ROWS,
        }

    # Task #317 — pre-train cadence guard. If the labeled frame for this
    # slice was assembled from rows of more than one native cadence and
    # no operator-approved mitigation is recorded, refuse to train. The
    # verification gate would block promotion anyway, but skipping the
    # train+save here also keeps a contaminated booster off disk.
    if "bars_native_cadence_ms" in df.columns and "bars_source" in df.columns:
        precadence: dict[str, int] = {}
        for cad_ms, src in zip(
            df["bars_native_cadence_ms"].fillna(0).astype("int64").tolist(),
            df["bars_source"].fillna("unknown").astype(str).tolist(),
        ):
            precadence[f"{src}:{int(cad_ms)}ms"] = (
                precadence.get(f"{src}:{int(cad_ms)}ms", 0) + 1
            )
        if len(precadence) > 1:
            return {
                "coin_id": coin_id, "timeframe": timeframe,
                "status": "rejected_cadence_mixed",
                "n_rows": int(len(df)),
                "bars_by_native_cadence": precadence,
                "cadence_mixed": True,
                "cadence_mitigation": None,
            }

    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df = _encode_coin_idx(df, vocab)
    active_feature_columns = (
        list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    )
    X_all, y_all = _prepare_xy(df, feature_columns=active_feature_columns)

    # Adaptive walk-forward fold count, codified policy:
    #   - Target 5 folds (the documented default in README.md).
    #   - Cap each fold to ~30 rows of test data so noise doesn't dominate
    #     metrics on tiny datasets.
    #   - Floor at 2 folds so every trainable slice (>= MIN_TRAIN_ROWS=80)
    #     still gets a hold-out evaluation.
    # Pooled models always have enough data for the full 5; per-coin slices
    # at the 80-row floor degrade to 2 folds, which is intentional.
    n_folds = max(2, min(5, len(df) // 30))
    cfg = WalkForwardConfig(n_folds=n_folds, min_train_size=max(50, len(df) // (n_folds + 1)))
    folds: list[FoldResult] = []

    for k, (tr_idx, te_idx) in enumerate(walk_forward_splits(df, cfg)):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all.iloc[te_idx], y_all[te_idx]

        params, booster = _optuna_search_lgb(X_tr, y_tr)
        lgb_pred = booster.predict(X_te, num_iteration=booster.best_iteration)
        if lgb_pred.ndim == 1:  # degenerate single-class fold
            lgb_pred = np.tile([0, 1, 0], (len(X_te), 1)).astype(float)

        enc, lr, priors = _train_baseline(X_tr, y_tr)
        base_pred = _baseline_predict(enc, lr, priors, X_te)

        # Task #316 — capture gain importances so the failure-analysis
        # pass can compute rank-corr across adjacent folds without
        # re-training. `feature_importance` returns one entry per active
        # column in `X_tr.columns` order.
        try:
            importances = booster.feature_importance(importance_type="gain").tolist()
            importance_names = list(X_tr.columns)
        except Exception:  # noqa: BLE001 — degenerate booster
            importances = None
            importance_names = None

        folds.append(FoldResult(
            fold=k, n_train=len(tr_idx), n_test=len(te_idx),
            auc=_multiclass_macro_auc(y_te, lgb_pred),
            log_loss=_multiclass_log_loss(y_te, lgb_pred),
            brier=_multiclass_brier(y_te, lgb_pred),
            directional_accuracy=_directional_accuracy(y_te, lgb_pred),
            baseline_auc=_multiclass_macro_auc(y_te, base_pred),
            baseline_log_loss=_multiclass_log_loss(y_te, base_pred),
            baseline_brier=_multiclass_brier(y_te, base_pred),
            baseline_directional_accuracy=_directional_accuracy(y_te, base_pred),
            best_params=params,
            feature_importance=importances,
            feature_names=importance_names,
        ))

    if not folds:
        return {"coin_id": coin_id, "timeframe": timeframe, "status": "no_folds", "n_rows": int(len(df))}

    def _mean(attr: str) -> float:
        vals = [getattr(f, attr) for f in folds if not math.isnan(getattr(f, attr))]
        return float(np.mean(vals)) if vals else float("nan")

    booster_metrics = {
        "auc": _mean("auc"), "log_loss": _mean("log_loss"),
        "brier": _mean("brier"), "directional_accuracy": _mean("directional_accuracy"),
    }
    baseline_metrics = {
        "auc": _mean("baseline_auc"), "log_loss": _mean("baseline_log_loss"),
        "brier": _mean("baseline_brier"),
        "directional_accuracy": _mean("baseline_directional_accuracy"),
    }

    # Task #400 — decide BEFORE the final fit whether to register the
    # multinomial-logistic baseline as the served predictor for this
    # slice. We compare the walk-forward CV mean directional accuracies
    # (same source the verification gate consumes) and require the
    # baseline to also clear the watchdog floor on its own (DA > 0.50,
    # sum-of-folds n_test >= 200). When the baseline wins this gate the
    # slice's served predictor IS the baseline: the booster file is
    # NOT written to disk, the per-class isotonic calibration is fit
    # on the BASELINE's holdout predictions, and `metrics` /
    # `class_means` / diagnostics are sourced from the baseline. The
    # booster's CV metrics are still preserved on the report (under
    # `lightgbm_cv_metrics`) for transparency.
    total_holdout_rows = int(sum(f.n_test for f in folds))
    served_baseline = _should_serve_baseline(
        booster_metrics["directional_accuracy"],
        baseline_metrics["directional_accuracy"],
        total_holdout_rows,
    )
    served_predictor_kind = "baseline" if served_baseline else "lightgbm"
    baseline_artifact_for_save: Optional[tuple] = None

    # --- Final fit + per-class isotonic calibration on a held-out tail.
    cal_start = max(1, int(len(df) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
    if cal_start >= len(df) - 5:
        final_params, final_booster = _optuna_search_lgb(X_all, y_all)
        calibrators = None
        calibration_diagram: list[dict] = []
        # No calibration holdout — fall back to in-sample predictions for
        # the directional-call share so the metric is still tracked.
        booster_in_sample_raw = final_booster.predict(
            X_all, num_iteration=final_booster.best_iteration,
        )
        if booster_in_sample_raw.ndim == 1:
            booster_in_sample_raw = np.tile(
                [0, 1, 0], (len(X_all), 1),
            ).astype(float)
        if served_baseline:
            # Refit the baseline on the full slice (no calibration tail
            # available) and use its predictions as the served signal.
            enc_b_full, lr_b_full, priors_b_full = _train_baseline(X_all, y_all)
            baseline_artifact_for_save = (enc_b_full, lr_b_full, priors_b_full)
            in_sample_raw = _baseline_predict(
                enc_b_full, lr_b_full, priors_b_full, X_all,
            )
        else:
            in_sample_raw = booster_in_sample_raw
        directional_call_share = float((in_sample_raw.argmax(axis=1) != 1).mean())
        directional_call_share_n = int(len(X_all))
        directional_call_share_source = "in_sample"
    else:
        X_final_train, y_final_train = X_all.iloc[:cal_start], y_all[:cal_start]
        X_cal, y_cal = X_all.iloc[cal_start:], y_all[cal_start:]
        final_params, final_booster = _optuna_search_lgb(X_final_train, y_final_train)
        booster_cal_pred_raw = final_booster.predict(
            X_cal, num_iteration=final_booster.best_iteration,
        )
        if booster_cal_pred_raw.ndim == 1:
            booster_cal_pred_raw = np.tile(
                [0, 1, 0], (len(X_cal), 1),
            ).astype(float)
        if served_baseline:
            # Task #400 — refit the baseline on the same train slice the
            # booster used so the calibration tail is genuine OOS for
            # both heads. The (encoder, lr, priors) triple is what gets
            # persisted to `baseline.joblib`.
            enc_b_srv, lr_b_srv, priors_b_srv = _train_baseline(
                X_final_train, y_final_train,
            )
            baseline_artifact_for_save = (enc_b_srv, lr_b_srv, priors_b_srv)
            cal_pred_raw = _baseline_predict(
                enc_b_srv, lr_b_srv, priors_b_srv, X_cal,
            )
        else:
            cal_pred_raw = booster_cal_pred_raw
        calibrators, cal_calibrated, calibration_diagram = _calibrate_per_class(cal_pred_raw, y_cal)
        # Share of holdout predictions where the model picks UP or DOWN
        # over STABLE. Class index 1 is STABLE; see NUM_CLASSES doc above.
        directional_call_share = float((cal_calibrated.argmax(axis=1) != 1).mean())
        directional_call_share_n = int(len(cal_calibrated))
        directional_call_share_source = "holdout"

    # Task #400 — `metrics` is the slice's served-predictor headline.
    # When the baseline serves we copy the baseline CV metrics here so
    # downstream consumers (verification gate, dashboards, failure
    # analysis) read the actual served-head numbers; the booster's CV
    # metrics are preserved separately under `lightgbm_cv_metrics` for
    # head-to-head transparency.
    if served_baseline:
        metrics = dict(baseline_metrics)
    else:
        metrics = dict(booster_metrics)

    # Fill in missing class return means with ±threshold for DOWN/UP and 0
    # for STABLE. Use the LABEL threshold (the same one labels.py used to
    # assign the class) so the means stored in the manifest reflect the
    # band the model was actually trained on. For per-coin slices this
    # honors the per-coin override registered in trading-frictions.json
    # (task #120). Pooled slices fall back to the timeframe default since
    # the pool can mix coins with different overrides.
    #
    # Task #379 — at 1h/2h/6h the directional label is assigned at a
    # multi-bar threshold (`MULTI_BAR_LABEL_THRESHOLDS_PERCENT[tf]`), so
    # the fallback threshold for a class with no observations MUST be the
    # same multi-bar number; using the legacy 1-bar `LABEL_THRESHOLDS_PERCENT`
    # would advertise an expected return that is ~4× too small relative to
    # the horizon the classifier predicts, distorting `expectedReturnPct`
    # for empty-class slices.
    # Task #459 — when labels.py stamped a per-row directional-label
    # threshold (the actual band it used to assign STABLE / UP / DOWN),
    # mirror it onto the manifest so downstream consumers see the band
    # the model was actually trained on. For per-coin slices every row
    # carries the same value; for pooled slices we take the median so
    # the manifest scalar is robust to per-coin variation. The legacy
    # resolver chain stays as the fallback for pre-#459 cached frames.
    threshold_pct_source = "static"
    threshold_pct = None
    if "directional_label_threshold_pct" in df.columns:
        stamped = df["directional_label_threshold_pct"].dropna().astype(float)
        if not stamped.empty:
            threshold_pct = float(stamped.median())
            if "directional_label_threshold_source" in df.columns:
                src_series = (
                    df["directional_label_threshold_source"]
                    .dropna()
                    .astype(str)
                )
                if not src_series.empty:
                    mode = src_series.mode()
                    if not mode.empty:
                        threshold_pct_source = str(mode.iloc[0])
    if threshold_pct is None:
        # Pre-#459 frames had no per-row threshold stamp; fall back to
        # the historical resolver chain so cached parquet snapshots that
        # predate this task still get a plausible manifest scalar.
        if timeframe in MULTI_BAR_LABEL_THRESHOLDS_PERCENT:
            threshold_pct = resolve_directional_label_threshold_pct(
                coin_id if coin_id != POOLED_COIN_ID else "__pooled__",
                timeframe,
            )
        elif coin_id == POOLED_COIN_ID:
            threshold_pct = LABEL_THRESHOLDS_PERCENT.get(
                timeframe, OUTCOME_THRESHOLDS_PERCENT[timeframe],
            )
        else:
            threshold_pct = resolve_label_threshold_pct(coin_id, timeframe)
    class_means = _class_return_means_pct(df)
    if class_means[0] == 0.0:  # no DOWN observations
        class_means[0] = -threshold_pct
    if class_means[2] == 0.0:  # no UP observations
        class_means[2] = threshold_pct

    # Task #633 — UP/STABLE/DOWN row counts of `label_3class` over the
    # full training frame for this slice. Surfaced on the manifest so
    # future audits can read the realized label shape (and confirm the
    # vol-scaled threshold actually rebalanced the classes) without
    # reverse-engineering it from the parquet snapshot. Indices match
    # `LABEL_3CLASS`: 0=DOWN, 1=STABLE, 2=UP.
    label_distribution: Optional[dict] = None
    if "label_3class" in df.columns:
        try:
            counts = df["label_3class"].astype("int64").value_counts()
            label_distribution = {
                "DOWN": int(counts.get(0, 0)),
                "STABLE": int(counts.get(1, 0)),
                "UP": int(counts.get(2, 0)),
                "n": int(len(df)),
            }
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "label_distribution_summarize_failed",
                extra={"coin_id": coin_id, "timeframe": timeframe, "error": str(exc)},
            )

    # Task #135 — fit a magnitude head on non-stable rows of the FULL slice.
    # `df` is already sorted by timestamp; the helper handles encoding +
    # holdout split internally. Returns (None, None) when there isn't
    # enough non-stable data, in which case the saved model has no
    # regressor and inference falls back to the legacy class-mean expRet.
    regressor, regressor_stats = _train_regressor_head(
        df, vocab, feature_columns=active_feature_columns,
    )

    # Task #147 — gates-alignment diagnostic. We score the calibrated
    # holdout (the same slice `directional_call_share` uses) with both
    # heads and report how often the SIGN head (classifier edge) and the
    # MAGNITUDE head agree on whether to trade. Done on the holdout
    # whenever it exists; otherwise falls back to the in-sample sample
    # (same source as `directional_call_share_source`).
    gates_alignment: Optional[dict] = None
    if regressor is not None:
        try:
            from ..backtest.contract import get_frictions  # local: heavy import
            fr = get_frictions()
            mde = float(fr.min_directional_edge)
            mer = float(fr.min_expected_return_pct)
            if directional_call_share_source == "holdout":
                probs_for_align = cal_calibrated
                X_for_reg = X_cal
            else:
                probs_for_align = in_sample_raw
                X_for_reg = X_all
            magnitudes = np.abs(regressor.predict(
                X_for_reg, num_iteration=regressor.best_iteration,
            ).astype(float))
            gates_alignment = _gate_alignment_summary(
                probs_for_align, magnitudes, mde, mer,
                source=directional_call_share_source,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic is best-effort
            logger.warning(
                "gates_alignment_failed",
                extra={"coin_id": coin_id, "timeframe": timeframe, "error": str(exc)},
            )

    # Task #316 — per-class accuracy + calibration + regime + PnL surface.
    # Computed on the calibrated holdout when one exists; otherwise on
    # the in-sample tail (same shape as `directional_call_share`). All
    # fields default to safe shapes so the failure-analysis pass can rely
    # on the keys existing on every trained slice.
    if directional_call_share_source == "holdout":
        proba_for_diag = cal_calibrated
        y_for_diag = y_cal
        df_holdout = df.iloc[cal_start:].reset_index(drop=True)
        train_y = y_all[:cal_start]
    else:
        proba_for_diag = in_sample_raw
        y_for_diag = y_all
        df_holdout = df.reset_index(drop=True)
        train_y = y_all

    per_class_holdout_breakdown = _per_class_breakdown(y_for_diag)
    per_class_train_breakdown = _per_class_breakdown(train_y)
    per_class_accuracy = _per_class_accuracy(y_for_diag, proba_for_diag)
    train_vs_holdout_class_balance_drift = _class_balance_drift(
        per_class_train_breakdown, per_class_holdout_breakdown,
    )
    per_class_brier = _per_class_brier(y_for_diag, proba_for_diag)
    reliability_max_dev_per_class = _reliability_max_dev_per_class(
        calibration_diagram,
    )
    confidence_bucket_da = _confidence_bucket_da(proba_for_diag, y_for_diag)
    predicted_class_entropy = _predicted_class_entropy(proba_for_diag)
    # Refit the LR baseline on the final-train slice and score the same
    # holdout the calibrator was scored on. Lets `prediction_collapse`
    # report the gap vs the baseline model (per task #316 wording) as
    # well as vs the empirical class prior. Best-effort; on failure we
    # simply omit the baseline-gap field.
    baseline_proba_for_diag: Optional[np.ndarray] = None
    try:
        if directional_call_share_source == "holdout":
            X_final_train_b, y_final_train_b = X_all.iloc[:cal_start], y_all[:cal_start]
            X_for_base = X_all.iloc[cal_start:]
        else:
            X_final_train_b, y_final_train_b = X_all, y_all
            X_for_base = X_all
        enc_b, lr_b, priors_b = _train_baseline(X_final_train_b, y_final_train_b)
        baseline_proba_for_diag = _baseline_predict(enc_b, lr_b, priors_b, X_for_base)
    except Exception:  # noqa: BLE001 — diagnostic only
        baseline_proba_for_diag = None
    prediction_collapse = _prediction_collapse(
        proba_for_diag, y_for_diag, baseline_proba=baseline_proba_for_diag,
    )
    # Task #337 — surface a contamination signal that goes beyond the
    # `cadence_mixed` row-stamp check, plus the share of holdout rows
    # whose calibrated distribution sits within ε of the training class
    # prior. The auto failure-analysis pass already inspects both fields
    # for byte-for-byte parity with the hand-run script; emitting them
    # here is what flips those branches on.
    contamination_flag = _compute_contamination_flag(df, timeframe)
    predictions_near_prior = _predictions_near_prior(proba_for_diag, train_y)
    regime_series = (
        df_holdout["regime"]
        if "regime" in df_holdout.columns and len(df_holdout) == len(y_for_diag)
        else None
    )
    regime_bucketed_da = _regime_bucketed_da(
        regime_series, proba_for_diag, y_for_diag,
    )
    feature_importance_stability = _feature_importance_rank_corr_across_folds(folds)

    # Holdout PnL after the production round-trip cost. Uses the live
    # entry rule (mde gate; mer gate too when a regression head is
    # available) so the figure mirrors what the live trader would have
    # realized on the same rows the calibrator was scored on.
    pnl_after_fees: Optional[dict] = None
    try:
        from ..backtest.contract import get_frictions  # local: heavy import
        fr = get_frictions()
        mde_pnl = float(fr.min_directional_edge)
        mer_pnl = float(fr.min_expected_return_pct)
        rtc = float(fr.round_trip_cost_pct) * 100.0  # contract is fraction; report uses %
        if (
            "forward_return" in df_holdout.columns
            and len(df_holdout) == len(y_for_diag)
        ):
            forward_returns = df_holdout["forward_return"].to_numpy(dtype=float)
            magnitudes_pct: Optional[np.ndarray] = None
            if regressor is not None:
                try:
                    if directional_call_share_source == "holdout":
                        X_for_reg_pnl = X_cal
                    else:
                        X_for_reg_pnl = X_all
                    magnitudes_pct = np.abs(regressor.predict(
                        X_for_reg_pnl,
                        num_iteration=regressor.best_iteration,
                    ).astype(float))
                except Exception:  # noqa: BLE001 — best-effort
                    magnitudes_pct = None
            pnl_after_fees = _holdout_pnl_after_fees(
                proba_for_diag, forward_returns,
                min_directional_edge=mde_pnl,
                min_expected_return_pct=mer_pnl,
                round_trip_cost_pct=rtc,
                magnitudes_pct=magnitudes_pct,
            )
    except Exception as exc:  # noqa: BLE001 - diagnostic is best-effort
        logger.warning(
            "holdout_pnl_after_fees_failed",
            extra={"coin_id": coin_id, "timeframe": timeframe, "error": str(exc)},
        )

    version = make_version()
    manifest = ModelManifest(
        coin_id=coin_id,
        timeframe=timeframe,
        version=version,
        feature_names=list(active_feature_columns),
        coin_vocab=vocab,
        n_train_rows=int(len(df)),
        n_test_rows=int(sum(f.n_test for f in folds)),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        threshold_pct=threshold_pct,
        threshold_pct_source=threshold_pct_source,
        threshold_source=threshold_pct_source,
        label_distribution=label_distribution,
        # Task #379 — surface the actual directional-label horizon used
        # for this slice (1 for 1m/5m/1d, 4 for 1h/2h/6h by default). The
        # registry / verification layer has historically read 1; downstream
        # consumers tolerate any positive int.
        horizon_candles=int(_labels_mod.resolve_directional_label_horizon_candles(timeframe)),
        class_return_means_pct=class_means,
        fold_metrics=[f.__dict__ for f in folds],
        note=note,
        directional_call_share=directional_call_share,
        directional_call_share_n=directional_call_share_n,
        directional_call_share_source=directional_call_share_source,
        served_predictor_kind=served_predictor_kind,
    )
    if regressor_stats is not None:
        manifest.regression_head_stats = regressor_stats
    if gates_alignment is not None:
        manifest.gates_alignment = gates_alignment
    # Phase 3 — stamp specialist + provenance metadata. Always set the
    # feature_schema_hash and training_window so even legacy per-coin /
    # pooled slots get the new fields; specialist_kind / regime_subset
    # stay None for non-specialist slots.
    manifest.specialist_kind = specialist_kind
    manifest.regime_subset = list(regime_subset or [])
    manifest.feature_schema_hash = _feature_schema_hash(list(active_feature_columns))
    manifest.training_window = _training_window_iso(df)
    # Task #317 — cadence provenance. Pull the per-row stamps written by
    # `build_labeled_frame_for_coin` and aggregate; the verification gate
    # uses these fields (via
    # `verification.manifest_blocks_promotion_for_cadence_mix`) to refuse
    # any slice whose rows came from more than one native cadence and is
    # not explicitly mitigated.
    cadence_breakdown: dict[str, int] = {}
    if "bars_native_cadence_ms" in df.columns and "bars_source" in df.columns:
        for cad_ms, src in zip(
            df["bars_native_cadence_ms"].fillna(0).astype("int64").tolist(),
            df["bars_source"].fillna("unknown").astype(str).tolist(),
        ):
            label = f"{src}:{int(cad_ms)}ms"
            cadence_breakdown[label] = cadence_breakdown.get(label, 0) + 1
    if cadence_breakdown:
        manifest.bars_by_native_cadence = cadence_breakdown
        manifest.cadence_mixed = len(cadence_breakdown) > 1
        # Pick the dominant source/cadence for the scalar fields.
        dominant_label = max(cadence_breakdown.items(), key=lambda kv: kv[1])[0]
        try:
            dom_src, dom_cad = dominant_label.rsplit(":", 1)
            manifest.bars_source = (
                "mixed" if manifest.cadence_mixed else dom_src
            )
            manifest.bars_native_cadence_ms = int(dom_cad.rstrip("ms"))
        except (ValueError, AttributeError):
            manifest.bars_source = "mixed" if manifest.cadence_mixed else None
            manifest.bars_native_cadence_ms = None
    else:
        # No per-row stamps — record an empty (uniform) cadence breakdown
        # so the manifest still carries the contract fields the gate
        # inspects. cadence_mixed stays False; gate is a no-op.
        manifest.bars_by_native_cadence = {}
        manifest.cadence_mixed = False
        manifest.bars_source = manifest.bars_source or "resampled_ticks"
        manifest.bars_native_cadence_ms = manifest.bars_native_cadence_ms or 60_000
    save_model(
        coin_id, timeframe, version, final_booster, calibrators, manifest,
        regressor=regressor,
        baseline_artifact=baseline_artifact_for_save,
    )
    return {
        "coin_id": coin_id, "timeframe": timeframe, "status": "trained",
        "version": version,
        "n_rows": int(len(df)),
        "n_folds": len(folds),
        "metrics": metrics, "baseline_metrics": baseline_metrics,
        # Task #400 — surface served-predictor identity + the booster's
        # CV metrics for transparency. The verification gate and the
        # auto failure-analysis pass branch on `served_predictor_kind`
        # (baseline-served slices skip the lift check and land in their
        # own cohort). `lightgbm_cv_metrics` is the head-to-head record
        # of what the booster scored even when we ended up serving the
        # baseline, so an operator can audit the close cases.
        "served_predictor_kind": served_predictor_kind,
        "lightgbm_cv_metrics": booster_metrics,
        "lift_auc": (metrics["auc"] - baseline_metrics["auc"])
            if not math.isnan(metrics["auc"]) and not math.isnan(baseline_metrics["auc"]) else None,
        "fold_metrics": [f.__dict__ for f in folds],
        "best_params": final_params,
        "calibration_diagram": calibration_diagram,
        "class_return_means_pct": class_means,
        "directional_call_share": directional_call_share,
        "directional_call_share_n": directional_call_share_n,
        "directional_call_share_source": directional_call_share_source,
        # Task #146 — surface the magnitude head's holdout stats so the
        # training report can flag a future retrain that collapses the
        # regressor back near 0 (the bug task #135 fixed) without having
        # to crack open the manifest by hand. `threshold_pct` is included
        # so the report can compare p95(|pred|) against the timeframe's
        # label-threshold floor.
        "threshold_pct": threshold_pct,
        # Task #459 — surface whether the STABLE-class band was widened
        # to the realized-volatility floor. The verification dashboard
        # and failure-analysis bucketing read this so an operator can
        # tell at a glance which slices are running on the dynamic band
        # vs the static curated value. `threshold_source` is the
        # task-#633 alias the post-NaN-cutover MTTM verdict tooling
        # reads; both fields carry the same value.
        "threshold_pct_source": threshold_pct_source,
        "threshold_source": threshold_pct_source,
        "label_distribution": label_distribution,
        "has_regression_head": regressor is not None,
        "regression_head_stats": regressor_stats,
        "gates_alignment": gates_alignment,
        "specialist_kind": specialist_kind,
        "regime_subset": list(regime_subset or []),
        "feature_schema_hash": manifest.feature_schema_hash,
        "training_window": manifest.training_window,
        # Task #316 — per-class accuracy + calibration + regime + PnL
        # surface so the failure-analysis pass can fill in every
        # "NOT MEASURED" field in the most recent diagnostic without
        # re-deriving anything from persisted artifacts.
        "per_class_holdout_breakdown": per_class_holdout_breakdown,
        "per_class_train_breakdown": per_class_train_breakdown,
        "per_class_accuracy": per_class_accuracy,
        "train_vs_holdout_class_balance_drift": train_vs_holdout_class_balance_drift,
        "per_class_brier": per_class_brier,
        "reliability_max_dev_per_class": reliability_max_dev_per_class,
        "confidence_bucket_da": confidence_bucket_da,
        "predicted_class_entropy": predicted_class_entropy,
        "prediction_collapse": prediction_collapse,
        # Task #337 — emitted alongside `prediction_collapse` so the auto
        # failure-analysis pass (failure_analysis._is_prediction_collapsed
        # / _is_schema_fix_required) can fire its near-prior and
        # contamination branches without re-deriving anything from disk.
        "contamination_flag": contamination_flag,
        "predictions_near_prior": predictions_near_prior,
        "regime_bucketed_da": regime_bucketed_da,
        "feature_importance_stability": feature_importance_stability,
        "pnl_after_fees": pnl_after_fees,
        "diagnostics_source": directional_call_share_source,
        # Task #317 — surface cadence provenance on the per-slice report
        # so the verification gate (`classify_slice`) can refuse a slice
        # that was assembled from rows of more than one native cadence.
        "bars_source": manifest.bars_source,
        "bars_native_cadence_ms": manifest.bars_native_cadence_ms,
        "bars_by_native_cadence": dict(manifest.bars_by_native_cadence or {}),
        "cadence_mixed": bool(manifest.cadence_mixed),
        "cadence_mitigation": manifest.cadence_mitigation,
    }


def _specialist_target_3class(df: pd.DataFrame, kind: str) -> Optional[pd.Series]:
    """Phase 3 — derive a cost-aware 3-class target from the trade-aware
    label columns (`tp_before_sl_long/short`, `opportunity_score`,
    `forward_window_return_pct`) so each specialist optimizes a
    trade-relevant objective rather than the legacy raw-return-sign
    `label_3class`.

    Directional specialists (momentum, mean_reversion, breakout) learn
    "TP-before-SL on the long side" vs "TP-before-SL on the short side"
    vs "no edge" using the precomputed barrier flags. Rows where neither
    barrier is hit fall back to the sign of `opportunity_score` (already
    cost-adjusted) so we don't throw away the row.

    The volatility_forecaster is a magnitude head — its 3-class proxy is
    tercile-bucketed `|forward_window_return_pct|` (low / mid / high).
    Using a 3-class shape lets us reuse `train_one_slice`'s walk-forward
    + per-class isotonic calibration machinery without forking it; the
    binary `prob_move_gt_cost` regression target tracked in `labels.py`
    is still in the manifest for downstream consumers that want the raw
    column.

    Returns None when the slice is missing the required columns or has
    no resolvable target (caller treats as `insufficient_data`).
    """
    if df.empty:
        return None
    if kind == "volatility_forecaster":
        col = "forward_window_return_pct"
        if col not in df.columns:
            return None
        absret = df[col].abs()
        if absret.dropna().empty:
            return None
        try:
            buckets = pd.qcut(absret, q=3, labels=False, duplicates="drop")
        except ValueError:
            return None
        # qcut may yield <3 bins on tiny / constant slices. Coerce
        # NaN→1 (mid bucket) so every row keeps a target instead of
        # silently dropping rows from the train slice.
        return buckets.fillna(1).astype(int).clip(0, 2)
    # Directional specialists.
    if "tp_before_sl_long" not in df.columns or "tp_before_sl_short" not in df.columns:
        return None
    long_hit = df["tp_before_sl_long"]
    short_hit = df["tp_before_sl_short"]
    target = pd.Series(1, index=df.index, dtype=int)  # default = STABLE
    target[(long_hit == 1) & (short_hit != 1)] = 2  # UP wins barrier race
    target[(short_hit == 1) & (long_hit != 1)] = 0  # DOWN wins barrier race
    if "opportunity_score" in df.columns:
        unresolved = long_hit.isna() & short_hit.isna()
        opp = df["opportunity_score"]
        target[unresolved & (opp > 0)] = 2
        target[unresolved & (opp < 0)] = 0
    if target.nunique() < 2:
        return None
    return target


def train_specialists(
    df: pd.DataFrame, timeframe: str, vocab: list[str],
    feature_columns: Optional[Sequence[str]] = None,
) -> dict:
    """Phase 3 — train one pooled model per specialist kind.

    For each directional specialist, subset the labeled frame on that
    kind's regime block and reuse `train_one_slice`. The volatility
    forecaster trains on ALL rows (no regime filter) — its job is to
    estimate realized magnitude regardless of regime so the directional
    specialists can lean on it.

    Returns a `{kind -> result_dict}` map mirroring the per-coin map in
    `train_timeframe` so the report renderer + diagnostics card can walk
    a single shape. Each entry has the same status semantics as
    `train_one_slice` (`trained` / `insufficient_data` / etc.).

    Specialists are pooled — a single registry slot per (kind, timeframe)
    that all coins share. Per-coin specialists would be a future
    extension once we have enough per-coin/regime row mass.
    """
    out: dict[str, dict] = {}
    if df.empty:
        for kind in SPECIALIST_KINDS:
            out[kind] = {
                "status": "insufficient_data",
                "n_rows": 0,
                "specialist_kind": kind,
                "regime_subset": list(SPECIALIST_REGIME_MAP[kind]),
            }
        return out

    for kind, regimes in SPECIALIST_REGIME_MAP.items():
        if kind == "volatility_forecaster" or not regimes:
            sub = df.copy()
        else:
            if "regime" not in df.columns:
                out[kind] = {
                    "status": "insufficient_data",
                    "n_rows": 0,
                    "specialist_kind": kind,
                    "regime_subset": list(regimes),
                    "note": "labeled frame missing regime column",
                }
                continue
            sub = df[df["regime"].isin(regimes)].copy()

        slot = specialist_coin_id(kind)
        note = (
            f"Phase 3 specialist '{kind}' for {timeframe}. "
            f"Trained on regimes {regimes if regimes else 'ALL'} "
            f"({len(sub)} rows). Pooled across {len(vocab)} coins."
        )
        if len(sub) < MIN_TRAIN_ROWS:
            out[kind] = {
                "coin_id": slot,
                "timeframe": timeframe,
                "status": "insufficient_data",
                "n_rows": int(len(sub)),
                "min_rows_required": MIN_TRAIN_ROWS,
                "specialist_kind": kind,
                "regime_subset": list(regimes),
            }
            continue
        # Phase 3 — replace the legacy 3-class return-sign target with
        # a cost-aware target derived from the trade-aware label block.
        # See `_specialist_target_3class` for the per-kind derivation.
        # Without this override, specialists would all collapse to the
        # same single-model objective and the ensemble would be
        # observability theatre.
        spec_y = _specialist_target_3class(sub, kind)
        if spec_y is None:
            out[kind] = {
                "coin_id": slot, "timeframe": timeframe,
                "status": "insufficient_data",
                "n_rows": int(len(sub)),
                "specialist_kind": kind, "regime_subset": list(regimes),
                "note": "no resolvable trade-aware target",
            }
            continue
        sub = sub.copy()
        sub["label_3class"] = spec_y.astype(int).to_numpy()
        try:
            res = train_one_slice(
                sub, coin_id=slot, timeframe=timeframe, vocab=vocab,
                note=note,
                specialist_kind=kind, regime_subset=list(regimes),
                feature_columns=feature_columns,
            )
            # Annotate the per-class distribution of the cost-aware
            # target so the diagnostics card / report can compare
            # across specialists at a glance.
            res["specialist_target_distribution"] = {
                "down": int((spec_y == 0).sum()),
                "stable": int((spec_y == 1).sum()),
                "up": int((spec_y == 2).sum()),
            }
            res["specialist_target_kind"] = (
                "magnitude_terciles" if kind == "volatility_forecaster"
                else "tp_before_sl_cost_aware"
            )
        except Exception as exc:  # noqa: BLE001 — specialist failure is non-fatal
            logger.warning(
                "specialist_train_failed",
                extra={"kind": kind, "timeframe": timeframe, "error": str(exc)},
            )
            out[kind] = {
                "coin_id": slot, "timeframe": timeframe,
                "status": "error", "error": str(exc),
                "specialist_kind": kind, "regime_subset": list(regimes),
            }
            continue
        out[kind] = res
    return out


def _empirical_class_priors_from_ticks(
    ticks_by_coin: dict[str, list[tuple]],
    timeframe: str,
) -> tuple[list[float], list[float], int]:
    """Build a feature-free 3-class prior for a timeframe.

    For each coin we resample real ticks to the target timeframe's bucket
    and compute per-bucket forward returns. Each forward return is labeled
    DOWN/STABLE/UP using the timeframe's threshold. Returns:
      (prior_probs[DOWN, STABLE, UP], class_return_means_pct, n_labels)

    This is the data source for the prior pooled fallback used by 1h/2h/6h/1d
    when there isn't enough per-coin history to fit indicator features. Labels
    pool across coins so a TF with a single forward bucket per coin still
    yields a defensible prior (vs. a uniform [1/3, 1/3, 1/3] guess).
    Falls back to a uniform prior + threshold-mean returns if there are 0
    labels (e.g. 1d on a freshly-installed DB).
    """
    from ..features import TIMEFRAME_MS, resample_to_candles

    bucket_ms = TIMEFRAME_MS[timeframe]
    threshold_pct = OUTCOME_THRESHOLDS_PERCENT[timeframe]
    counts = [0, 0, 0]
    return_sums = [0.0, 0.0, 0.0]
    for coin, ticks in ticks_by_coin.items():
        closes = resample_to_candles(ticks, bucket_ms)
        for i in range(len(closes) - 1):
            if closes[i] <= 0:
                continue
            r = (closes[i + 1] - closes[i]) / closes[i]
            r_pct = r * 100.0
            if r_pct > threshold_pct:
                k = 2
            elif r_pct < -threshold_pct:
                k = 0
            else:
                k = 1
            counts[k] += 1
            return_sums[k] += r_pct
    total = sum(counts)
    if total == 0:
        # No data at all — emit a uniform prior with threshold-symmetric means.
        return ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [-threshold_pct, 0.0, threshold_pct], 0)
    # Laplace smoothing (alpha=1) so a class with 0 observations still gets
    # nonzero mass — important for higher timeframes with tiny n.
    smoothed = [(c + 1) / (total + 3) for c in counts]
    means_pct: list[float] = []
    for k in range(3):
        if counts[k] > 0:
            means_pct.append(return_sums[k] / counts[k])
        else:
            means_pct.append(-threshold_pct if k == 0 else (threshold_pct if k == 2 else 0.0))
    return smoothed, means_pct, total


async def _train_prior_pooled(
    timeframe: str, coin_ids: Sequence[str], lookback_ms: int,
) -> dict:
    """Build and persist a prior-only pooled model for `timeframe`.

    Called when `train_timeframe` couldn't fit a real LightGBM pooled model
    because no per-coin slice produced any feature rows (typically 1h/2h/6h/1d
    on a fresh DB). The prior is empirical (Laplace-smoothed class frequencies
    of forward returns at the target timeframe) — honest about what we know,
    and immediately replaceable by a real model on the next training run that
    finally has enough history.
    """
    from ..db import fetch_real_ticks

    ticks_by_coin: dict[str, list] = {}
    for c in coin_ids:
        try:
            ticks_by_coin[c] = await fetch_real_ticks(c, lookback_ms)
        except Exception as exc:  # noqa: BLE001
            logger.warning("prior_fetch_ticks_failed", extra={"coin": c, "tf": timeframe, "error": str(exc)})
            ticks_by_coin[c] = []

    priors, means_pct, n_labels = _empirical_class_priors_from_ticks(ticks_by_coin, timeframe)
    threshold_pct = OUTCOME_THRESHOLDS_PERCENT[timeframe]
    version = make_version()
    manifest = ModelManifest(
        coin_id=POOLED_COIN_ID,
        timeframe=timeframe,
        version=version,
        feature_names=[],          # prior models have no feature schema
        coin_vocab=sorted(coin_ids),
        n_train_rows=int(n_labels),
        n_test_rows=0,
        metrics={"auc": float("nan"), "log_loss": float("nan"),
                 "brier": float("nan"), "directional_accuracy": float("nan")},
        baseline_metrics={"auc": float("nan"), "log_loss": float("nan"),
                          "brier": float("nan"), "directional_accuracy": float("nan")},
        threshold_pct=threshold_pct,
        horizon_candles=1,
        class_return_means_pct=means_pct,
        fold_metrics=[],
        note=(
            "Prior-only pooled fallback — emitted because no per-coin slice "
            f"produced enough rows for an indicator-based fit at {timeframe}. "
            f"Built from {n_labels} forward-return labels across "
            f"{len(coin_ids)} coins (Laplace-smoothed). Will be auto-replaced "
            "by a real LightGBM model once each coin has "
            f">= {MIN_CANDLES_FOR_FEATURES_HINT} candles of history."
        ),
        model_kind="prior",
        prior_probs=list(priors),
    )
    # Task #317 — prior-only fallbacks consume raw ticks at every coin,
    # so the input cadence is uniformly the live-poll cadence. Stamp a
    # single-cadence breakdown so the verification gate sees the slice
    # as cadence-uniform (cadence_mixed=False) and the contract fields
    # are present on every manifest written by this trainer.
    manifest.bars_source = "resampled_ticks"
    manifest.bars_native_cadence_ms = 60_000
    manifest.bars_by_native_cadence = {
        "resampled_ticks:60000ms": int(n_labels),
    }
    manifest.cadence_mixed = False
    save_model(POOLED_COIN_ID, timeframe, version, booster=None, calibrators=None, manifest=manifest)
    return {
        "coin_id": POOLED_COIN_ID, "timeframe": timeframe, "status": "trained",
        "version": version, "model_kind": "prior",
        "n_labels": int(n_labels),
        "prior_probs": list(priors),
        "class_return_means_pct": means_pct,
        # Task #317 — same provenance fields as the LightGBM path so the
        # verification gate can apply the cadence-mix check uniformly.
        "bars_source": manifest.bars_source,
        "bars_native_cadence_ms": manifest.bars_native_cadence_ms,
        "bars_by_native_cadence": dict(manifest.bars_by_native_cadence or {}),
        "cadence_mixed": bool(manifest.cadence_mixed),
        "cadence_mitigation": manifest.cadence_mitigation,
    }


# Imported lazily-by-name so the registry import block doesn't grow another
# coupling. Mirrors features.MIN_CANDLES_FOR_FEATURES so the prior-model note
# stays accurate if that constant ever moves.
from ..features import MIN_CANDLES_FOR_FEATURES as MIN_CANDLES_FOR_FEATURES_HINT  # noqa: E402


def train_timeframe(
    df: pd.DataFrame, timeframe: str, coin_ids: Sequence[str],
    feature_columns: Optional[Sequence[str]] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Train per-coin models when each coin has enough rows; for any coin
    that doesn't, train a single pooled model that serves them as fallback.
    Returns a per-timeframe report dict.
    """
    def _emit(rec: dict) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({"timeframe": timeframe, **rec})
        except Exception as exc:  # noqa: BLE001 - progress is best-effort
            logger.warning("progress_callback_failed", extra={"error": str(exc)})

    full_vocab = sorted(coin_ids)
    df = df.sort_values("timestamp_ms").reset_index(drop=True) if not df.empty else df
    if df.empty:
        _emit({
            "phase": "train_timeframe_done",
            "status": "insufficient_data",
            "headline": f"{timeframe}: 0 rows, no training",
            "n_rows": 0,
        })
        return {
            "timeframe": timeframe, "status": "insufficient_data",
            "n_rows": 0, "min_rows_required": MIN_TRAIN_ROWS,
            "per_coin": {}, "pooled": None,
        }

    # Persist the full labeled frame for reproducibility BEFORE training.
    snapshot_version = make_version()
    out_path = dataset_path(timeframe, snapshot_version)
    try:
        df.to_parquet(out_path, index=False)
        dataset_persisted = str(out_path.relative_to(registry_module.REGISTRY_ROOT))
    except Exception as exc:  # pragma: no cover - parquet engine missing fallback
        logger.warning("dataset_parquet_failed", extra={"error": str(exc)})
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
        dataset_persisted = str(out_path.relative_to(registry_module.REGISTRY_ROOT))

    per_coin: dict[str, dict] = {}
    pooled_coin_ids: list[str] = []
    n_coins = len(full_vocab)
    for idx, coin in enumerate(full_vocab, start=1):
        sub = df[df["coin_id"] == coin]
        if len(sub) < MIN_PER_COIN_ROWS:
            pooled_coin_ids.append(coin)
            per_coin[coin] = {
                "status": "insufficient_data_per_coin",
                "n_rows": int(len(sub)),
                "min_rows_required": MIN_PER_COIN_ROWS,
                "fallback": "pooled",
            }
            _emit({
                "phase": "per_coin_skipped",
                "status": "insufficient_data_per_coin",
                "headline": (
                    f"{coin}/{timeframe} skipped ({idx}/{n_coins}): "
                    f"{len(sub)} rows < {MIN_PER_COIN_ROWS}"
                ),
                "coin": coin,
                "n_rows": int(len(sub)),
            })
            continue
        _emit({
            "phase": "per_coin_start",
            "status": "running",
            "headline": f"fitting {coin}/{timeframe} ({idx}/{n_coins})",
            "coin": coin,
            "n_rows_input": int(len(sub)),
        })
        slice_t0 = time.time()
        res = train_one_slice(
            sub.copy(), coin_id=coin, timeframe=timeframe, vocab=[coin],
            note=f"Per-coin model for {coin}/{timeframe}.",
            feature_columns=feature_columns,
        )
        per_coin[coin] = res
        if res["status"] != "trained":
            pooled_coin_ids.append(coin)
        _emit({
            "phase": "per_coin_done",
            "status": (res or {}).get("status") or "unknown",
            "headline": (
                f"{coin}/{timeframe} ({idx}/{n_coins}) "
                f"status={(res or {}).get('status')}"
            ),
            "coin": coin,
            "elapsed_sec": round(time.time() - slice_t0, 2),
        })

    # Task #121 — always refit the pooled fallback on the full dataset on
    # every training run, even when every tracked coin already has a per-coin
    # model. The pooled slot is the safety net for any coin that doesn't have
    # one (e.g. an 11th coin added after launch, or a coin whose history is
    # wiped). If we only retrain it when `pooled_coin_ids` is non-empty, the
    # pooled model can age out of sync with the live label thresholds /
    # feature schema and silently collapse to "always STABLE" the moment a
    # new coin starts routing through it. Refitting on every run keeps the
    # safety net fresh; the cost is one extra `train_one_slice` per timeframe.
    pooled_report: Optional[dict] = None
    if not df.empty:
        pooled_df = df.copy()
        if pooled_coin_ids:
            note = (
                "Pooled fallback model trained on ALL coins. Coins without a "
                f"per-coin model on this run: {sorted(pooled_coin_ids)}. "
                "Resolved by /ml/predict when a per-coin model is missing."
            )
        else:
            note = (
                "Pooled fallback model trained on ALL coins. Every tracked "
                "coin has a per-coin model on this run, but the pooled slot "
                "is refreshed anyway so the safety net stays in sync with "
                "the current label thresholds and feature schema (task #121)."
            )
        pooled_report = train_one_slice(
            pooled_df, coin_id=POOLED_COIN_ID, timeframe=timeframe,
            vocab=full_vocab,
            note=note,
            feature_columns=feature_columns,
        )

    overall_status = "trained" if any(
        r.get("status") == "trained" for r in per_coin.values()
    ) or (pooled_report and pooled_report.get("status") == "trained") else "insufficient_data"

    # Phase 3 — train pooled per-regime specialists alongside the
    # per-coin / pooled set. Failures here never tank the overall run;
    # legacy heads are still authoritative for live trading until
    # Phase 4 wires the meta-model in.
    try:
        specialists_report = train_specialists(
            df, timeframe, full_vocab, feature_columns=feature_columns,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "specialists_train_block_failed",
            extra={"timeframe": timeframe, "error": str(exc)},
        )
        specialists_report = {"error": str(exc)}

    return {
        "timeframe": timeframe,
        "status": overall_status,
        "n_rows": int(len(df)),
        "per_coin": per_coin,
        "pooled": pooled_report,
        "specialists": specialists_report,
        "dataset_path": dataset_persisted,
    }


# --- Directional-call share history (task #101) -----------------------------
# Persisted timeseries so a regression after a label/threshold change is
# visible even after restart. JSON-Lines so concurrent writers can append
# atomically and old training runs are never rewritten.
DIRECTIONAL_SHARE_HISTORY_DIR = registry_module.REGISTRY_ROOT / "training_history"
DIRECTIONAL_SHARE_HISTORY_PATH = DIRECTIONAL_SHARE_HISTORY_DIR / "directional_call_share.jsonl"
# A retrain that drops any tradeable timeframe below this share is loud.
# Configurable so an operator can tune without redeploying.
DIRECTIONAL_SHARE_MIN_PCT = float(os.environ.get("ML_DIRECTIONAL_SHARE_MIN_PCT", "15.0"))
# Tradeable timeframes — only these trigger the loud alert. Untradeable
# timeframes (1m especially) still get logged but don't raise the alarm.
DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES = {"5m", "1h", "2h", "6h", "1d"}
# Retention policy for the rolling history file. Keep at most N records per
# (coin, timeframe) and drop anything older than M days. Both are env-tunable
# so an operator can keep more history without redeploying.
DIRECTIONAL_SHARE_HISTORY_MAX_PER_SLICE = int(
    os.environ.get("ML_DIRECTIONAL_SHARE_HISTORY_MAX_PER_SLICE", "500")
)
DIRECTIONAL_SHARE_HISTORY_MAX_AGE_DAYS = int(
    os.environ.get("ML_DIRECTIONAL_SHARE_HISTORY_MAX_AGE_DAYS", "90")
)


def _append_directional_share_history(record: dict) -> None:
    """Append one (coin, timeframe) record to the rolling history file.
    Best-effort: a write failure is logged and swallowed so training never
    fails because the dashboard history is full.
    """
    try:
        DIRECTIONAL_SHARE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with DIRECTIONAL_SHARE_HISTORY_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 - history is non-essential
        logger.warning("directional_share_history_append_failed", extra={"error": str(exc)})


def _trim_directional_share_history(
    max_per_slice: int = DIRECTIONAL_SHARE_HISTORY_MAX_PER_SLICE,
    max_age_days: int = DIRECTIONAL_SHARE_HISTORY_MAX_AGE_DAYS,
) -> dict:
    """Apply the retention policy to the rolling history file.

    Keeps the newest `max_per_slice` records per (coin_id, timeframe) and
    drops anything older than `max_age_days`. Rewrites the file atomically
    via a sibling temp file so a crash mid-write can't corrupt it. A
    malformed line is dropped (it would never round-trip anyway).

    Best-effort: any failure is logged and swallowed so the training run's
    contract is preserved.
    """
    summary = {"kept": 0, "dropped": 0, "skipped": False}
    try:
        if not DIRECTIONAL_SHARE_HISTORY_PATH.exists():
            summary["skipped"] = True
            return summary
        cutoff = None
        if max_age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        rows: list[tuple[datetime, str, dict, str]] = []
        dropped_malformed = 0
        dropped_age = 0
        with DIRECTIONAL_SHARE_HISTORY_PATH.open("r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    dropped_malformed += 1
                    continue
                ts_raw = rec.get("generated_at")
                try:
                    ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    # Unparseable timestamp -> treat as epoch so it's pruned
                    # by the per-slice cap before any well-dated record.
                    ts = datetime.fromtimestamp(0, tz=timezone.utc)
                if cutoff is not None and ts < cutoff:
                    dropped_age += 1
                    continue
                slice_key = f"{rec.get('coin_id')}|{rec.get('timeframe')}"
                rows.append((ts, slice_key, rec, line))
        # Per-slice cap: keep newest N per (coin, timeframe).
        by_slice: dict[str, list[tuple[datetime, dict, str]]] = {}
        for ts, key, rec, line in rows:
            by_slice.setdefault(key, []).append((ts, rec, line))
        kept: list[tuple[datetime, str]] = []
        dropped_capped = 0
        for key, items in by_slice.items():
            items.sort(key=lambda t: t[0])
            if max_per_slice > 0 and len(items) > max_per_slice:
                dropped_capped += len(items) - max_per_slice
                items = items[-max_per_slice:]
            for ts, _rec, line in items:
                kept.append((ts, line))
        # Re-sort globally by timestamp so the file stays append-style ordered.
        kept.sort(key=lambda t: t[0])
        tmp_path = DIRECTIONAL_SHARE_HISTORY_PATH.with_suffix(
            DIRECTIONAL_SHARE_HISTORY_PATH.suffix + ".tmp"
        )
        with tmp_path.open("w") as f:
            for _ts, line in kept:
                f.write(line + "\n")
        os.replace(tmp_path, DIRECTIONAL_SHARE_HISTORY_PATH)
        summary["kept"] = len(kept)
        summary["dropped"] = dropped_malformed + dropped_capped + dropped_age
        summary["dropped_malformed"] = dropped_malformed
        summary["dropped_capped"] = dropped_capped
        summary["dropped_age"] = dropped_age
        return summary
    except Exception as exc:  # noqa: BLE001 - history trim is non-essential
        logger.warning("directional_share_history_trim_failed", extra={"error": str(exc)})
        summary["skipped"] = True
        return summary


def _record_directional_share_for_report(report: dict) -> list[dict]:
    """Walk a finished training report, append a history row per slice, and
    return the list of rows written. Also emits a loud warning log when a
    tradeable timeframe drops below the configured floor.
    """
    written: list[dict] = []
    generated_at = report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    for tf, tf_report in (report.get("timeframes") or {}).items():
        if not isinstance(tf_report, dict):
            continue
        slices: list[tuple[str, dict]] = []
        for coin, slc in (tf_report.get("per_coin") or {}).items():
            if isinstance(slc, dict) and slc.get("status") == "trained":
                slices.append((coin, slc))
        pooled = tf_report.get("pooled")
        if isinstance(pooled, dict) and pooled.get("status") == "trained":
            slices.append((POOLED_COIN_ID, pooled))
        for coin, slc in slices:
            share = slc.get("directional_call_share")
            if share is None:
                continue
            share_pct = round(float(share) * 100.0, 2)
            tradeable = tf in DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES
            below = tradeable and share_pct < DIRECTIONAL_SHARE_MIN_PCT
            record = {
                "generated_at": generated_at,
                "coin_id": coin,
                "timeframe": tf,
                "version": slc.get("version"),
                "directional_call_share_pct": share_pct,
                "n_predictions": int(slc.get("directional_call_share_n") or 0),
                "source": slc.get("directional_call_share_source"),
                "n_train_rows": int(slc.get("n_rows") or 0),
                "tradeable_timeframe": tradeable,
                "below_threshold": below,
                "threshold_pct": DIRECTIONAL_SHARE_MIN_PCT,
            }
            _append_directional_share_history(record)
            written.append(record)
            if below:
                # Loud — warning level so it stands out in service logs and
                # alert pipelines that scrape WARN+ from ml-engine.
                logger.warning(
                    "directional_call_share_regression",
                    extra={
                        "coin_id": coin,
                        "timeframe": tf,
                        "version": slc.get("version"),
                        "share_pct": share_pct,
                        "threshold_pct": DIRECTIONAL_SHARE_MIN_PCT,
                    },
                )
    return written


# --- Regression-head holdout-stats history (task #157) ----------------------
# Mirrors the directional-call-share history above: one JSONL row per
# (coin, timeframe) per training run, capturing the magnitude head's
# holdout p50/p95/max/MAE so a slow drift toward the label-threshold
# floor (the bug task #135 fixed) is visible BEFORE it crosses the line.
REGRESSION_HEAD_HISTORY_PATH = (
    DIRECTIONAL_SHARE_HISTORY_DIR / "regression_head_stats.jsonl"
)
REGRESSION_HEAD_HISTORY_MAX_PER_SLICE = int(
    os.environ.get("ML_REGRESSION_HEAD_HISTORY_MAX_PER_SLICE", "500")
)
REGRESSION_HEAD_HISTORY_MAX_AGE_DAYS = int(
    os.environ.get("ML_REGRESSION_HEAD_HISTORY_MAX_AGE_DAYS", "90")
)


def _append_regression_head_history(record: dict) -> None:
    """Append one (coin, timeframe) regression-head record to the rolling
    history file. Best-effort; failures are logged and swallowed.
    """
    try:
        DIRECTIONAL_SHARE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with REGRESSION_HEAD_HISTORY_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "regression_head_history_append_failed", extra={"error": str(exc)},
        )


def _trim_regression_head_history(
    max_per_slice: int = REGRESSION_HEAD_HISTORY_MAX_PER_SLICE,
    max_age_days: int = REGRESSION_HEAD_HISTORY_MAX_AGE_DAYS,
) -> dict:
    """Same retention policy as `_trim_directional_share_history`: keep the
    newest `max_per_slice` rows per (coin_id, timeframe) and drop anything
    older than `max_age_days`. Atomic rewrite via sibling temp file.
    """
    summary = {"kept": 0, "dropped": 0, "skipped": False}
    try:
        if not REGRESSION_HEAD_HISTORY_PATH.exists():
            summary["skipped"] = True
            return summary
        cutoff = None
        if max_age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        rows: list[tuple[datetime, str, dict, str]] = []
        dropped_malformed = 0
        dropped_age = 0
        with REGRESSION_HEAD_HISTORY_PATH.open("r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    dropped_malformed += 1
                    continue
                ts_raw = rec.get("generated_at")
                try:
                    ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    ts = datetime.fromtimestamp(0, tz=timezone.utc)
                if cutoff is not None and ts < cutoff:
                    dropped_age += 1
                    continue
                slice_key = f"{rec.get('coin_id')}|{rec.get('timeframe')}"
                rows.append((ts, slice_key, rec, line))
        by_slice: dict[str, list[tuple[datetime, dict, str]]] = {}
        for ts, key, rec, line in rows:
            by_slice.setdefault(key, []).append((ts, rec, line))
        kept: list[tuple[datetime, str]] = []
        dropped_capped = 0
        for key, items in by_slice.items():
            items.sort(key=lambda t: t[0])
            if max_per_slice > 0 and len(items) > max_per_slice:
                dropped_capped += len(items) - max_per_slice
                items = items[-max_per_slice:]
            for ts, _rec, line in items:
                kept.append((ts, line))
        kept.sort(key=lambda t: t[0])
        tmp_path = REGRESSION_HEAD_HISTORY_PATH.with_suffix(
            REGRESSION_HEAD_HISTORY_PATH.suffix + ".tmp"
        )
        with tmp_path.open("w") as f:
            for _ts, line in kept:
                f.write(line + "\n")
        os.replace(tmp_path, REGRESSION_HEAD_HISTORY_PATH)
        summary["kept"] = len(kept)
        summary["dropped"] = dropped_malformed + dropped_capped + dropped_age
        summary["dropped_malformed"] = dropped_malformed
        summary["dropped_capped"] = dropped_capped
        summary["dropped_age"] = dropped_age
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "regression_head_history_trim_failed", extra={"error": str(exc)},
        )
        summary["skipped"] = True
        return summary


def _record_regression_head_for_report(report: dict) -> list[dict]:
    """Walk a finished training report and append one history row per
    trained slice that has a regression head. Returns the rows written.

    Only slices with `has_regression_head=True` and a populated
    `regression_head_stats` dict produce a record. Prior-only fallbacks
    and legacy manifests are silently skipped.
    """
    written: list[dict] = []
    generated_at = report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    for tf, tf_report in (report.get("timeframes") or {}).items():
        if not isinstance(tf_report, dict):
            continue
        slices: list[tuple[str, dict]] = []
        for coin, slc in (tf_report.get("per_coin") or {}).items():
            if isinstance(slc, dict) and slc.get("status") == "trained":
                slices.append((coin, slc))
        pooled = tf_report.get("pooled")
        if isinstance(pooled, dict) and pooled.get("status") == "trained":
            slices.append((POOLED_COIN_ID, pooled))
        for coin, slc in slices:
            if not slc.get("has_regression_head"):
                continue
            stats = slc.get("regression_head_stats")
            if not isinstance(stats, dict) or not stats:
                continue
            record = {
                "generated_at": generated_at,
                "coin_id": coin,
                "timeframe": tf,
                "version": slc.get("version"),
                "abs_pred_p50_pct": stats.get("abs_pred_p50_pct"),
                "abs_pred_p95_pct": stats.get("abs_pred_p95_pct"),
                "abs_pred_max_pct": stats.get("abs_pred_max_pct"),
                "mae_pct": stats.get("mae_pct"),
                "n_holdout_rows": stats.get("n_holdout_rows"),
                "n_train_rows": stats.get("n_train_rows"),
                "best_iteration": stats.get("best_iteration"),
                "threshold_pct": slc.get("threshold_pct"),
            }
            _append_regression_head_history(record)
            written.append(record)
    return written


# --- Task #267 — provenance / coverage / slow-fast helpers ------------------
def _summarize_feature_coverage(
    df: pd.DataFrame, feature_columns: Sequence[str],
) -> dict[str, float]:
    """Per-feature share of NON-NULL rows, as required by rule 7 of the
    training contract (`% non-null`). A feature that defaults to 0 for
    every row still counts as fully covered by this metric — that is
    intentional: NaN means "no value at all", 0 may be a legitimate
    real value (e.g. funding_rate exactly 0). For an operator-facing
    "is this stream actually delivering signal?" view, see
    `_summarize_feature_density` which reports `% non-zero`.
    """
    if df.empty:
        return {col: 0.0 for col in feature_columns}
    n = float(len(df))
    out: dict[str, float] = {}
    for col in feature_columns:
        if col not in df.columns:
            out[col] = 0.0
            continue
        out[col] = round(float(df[col].notna().sum()) / n, 4)
    return out


def _summarize_feature_density(
    df: pd.DataFrame, feature_columns: Sequence[str],
) -> dict[str, float]:
    """Per-feature share of non-null AND non-zero rows. Surfaced
    alongside coverage so an operator can spot the case where a stream
    column is "covered" (every row has a value) but defaulting to 0
    because no provider is actually wired up. Rule 7 of the contract.

    NOTE (Task #633): for `EXTERNAL_STREAM_FEATURE_COLUMNS` the
    authoritative "is this stream actually delivering?" signal is now
    `_detect_unwired_external_streams`, which uses `notna` only — a
    legitimately observed `funding_rate == 0.0` is real signal, not a
    default. This `(notna & != 0)` density heuristic remains useful for
    backwards-compat dashboards but understates coverage for zero-valued
    observations, so prefer the unwired-stream list when reasoning about
    stream wiring health.
    """
    if df.empty:
        return {col: 0.0 for col in feature_columns}
    n = float(len(df))
    out: dict[str, float] = {}
    for col in feature_columns:
        if col not in df.columns:
            out[col] = 0.0
            continue
        s = df[col]
        try:
            populated = float((s.notna() & (s != 0)).sum())
        except Exception:  # noqa: BLE001 - non-numeric / categorical
            populated = float(s.notna().sum())
        out[col] = round(populated / n, 4)
    return out


def _detect_unwired_external_streams(
    df: pd.DataFrame,
) -> list[str]:
    """Task #287 / #633 — return the subset of
    `EXTERNAL_STREAM_FEATURE_COLUMNS` that received NO real provider data
    on this run.

    Post-Task #633 the contract is unambiguous: `EXTERNAL_STREAM_DEFAULTS`
    writes NaN (not 0.0) when an asof-join finds no match, so a
    legitimately observed `funding_rate == 0.0` is a real signal and
    must NOT be treated as "missing" by this diagnostic. The fingerprint
    is therefore strictly "every row is NaN" — i.e. no asof-match ever
    landed for this column. Counting `s.notna() & (s != 0)` would
    understate coverage for honest zero-valued observations and is no
    longer correct.

    Surfacing this list lets the next training run loudly assert that
    the contract columns (funding_rate, open_interest_z,
    liquidations_1h_usd, bid_ask_spread_bps, btc_lead_ret_5m,
    eth_lead_ret_5m, plus BTC/ETH/SOL liq pulses) actually pick up real
    signal once their providers land in `market_signals`.
    """
    from .registry import EXTERNAL_STREAM_FEATURE_COLUMNS as _ES
    out: list[str] = []
    if df is None or df.empty:
        return out
    for col in _ES:
        if col not in df.columns:
            continue
        s = df[col]
        non_default = int(s.notna().sum())
        if non_default == 0:
            out.append(col)
    return out


def _summarize_target_row_counts(df: pd.DataFrame) -> dict[str, int]:
    """Row count with a non-null value for each declared forward target.
    Surfaced in `report.json` so an operator can tell whether a refactor
    silently dropped a target column."""
    if df.empty:
        return {col: 0 for col in FORWARD_TARGET_COLUMNS}
    out: dict[str, int] = {}
    for col in FORWARD_TARGET_COLUMNS:
        if col not in df.columns:
            out[col] = 0
            continue
        out[col] = int(df[col].notna().sum())
    return out


def _provenance_summary(per_coin: dict[str, dict]) -> dict:
    """Aggregate per-coin provenance into a per-timeframe summary."""
    coins_rejected = sorted([
        c for c, p in per_coin.items() if p.get("rejected_synthetic")
    ])
    return {
        "rows_real": int(sum(p.get("rows_real", 0) for p in per_coin.values())),
        "rows_synthetic": int(
            sum(p.get("rows_synthetic", 0) for p in per_coin.values())
        ),
        "coins_rejected": coins_rejected,
        "rejected_synthetic": bool(coins_rejected),
        "per_coin": per_coin,
    }


# Fast-loop trigger threshold: number of newly-resolved meta rows that
# must arrive before a meta-only retrain fires. Keeps the cadence ~hourly
# under normal flow (defaults to 100 rows). Set via env so an operator
# can dial it without redeploying.
FAST_LOOP_MIN_NEW_ROWS = int(os.environ.get("ML_FAST_LOOP_MIN_NEW_ROWS", "100"))
# Slow-loop new-data threshold: number of additional real ticks across
# all coins since the last slow-loop retrain that should also trigger
# a slow-loop run (in addition to the cadence + drift / regime triggers).
SLOW_LOOP_NEW_TICKS_THRESHOLD = int(
    os.environ.get("ML_SLOW_LOOP_NEW_TICKS_THRESHOLD", "5000")
)


def should_run_slow_loop(
    *,
    seconds_since_last_slow: Optional[float] = None,
    cadence_seconds: int = 1800,
    drift_detected: bool = False,
    regime_shift_detected: bool = False,
    new_ticks_since_last: int = 0,
    new_ticks_threshold: int = SLOW_LOOP_NEW_TICKS_THRESHOLD,
) -> tuple[bool, str]:
    """Pure decision helper for the slow-loop scheduler. Returns
    (should_run, reason). The orchestrator persists `reason` into
    `report.json::loop.reason` so operators can audit what fired."""
    if drift_detected:
        return True, "drift_detected"
    if regime_shift_detected:
        return True, "regime_shift"
    if new_ticks_since_last >= new_ticks_threshold:
        return True, "new_data_threshold"
    if seconds_since_last_slow is None:
        return True, "cold_start"
    if seconds_since_last_slow >= cadence_seconds:
        return True, "cadence"
    return False, "cooldown"


def should_run_fast_loop(
    *,
    new_meta_rows_since_last: int,
    threshold: int = FAST_LOOP_MIN_NEW_ROWS,
) -> tuple[bool, str]:
    """Pure decision helper for the fast-loop scheduler. The fast loop
    fires when at least `threshold` newly-resolved meta rows have arrived
    since the last meta retrain."""
    if new_meta_rows_since_last >= threshold:
        return True, f"new_meta_rows>={threshold}"
    return False, "below_threshold"


async def run_meta_only(timeframes: Sequence[str]) -> dict:
    """Task #267 — fast-loop entry point. Refits ONLY the meta-model on
    the most recent resolved meta rows; never touches the base classifier
    or regressor. Mirrors the meta block at the end of `run_training` so
    a fast tick can run independently between slow ticks.

    Writes a slim `loop=fast` envelope into `report.json` (without
    overwriting per-timeframe base reports) so operators can see which
    loop most recently produced a meta refresh.
    """
    await init_pool()
    try:
        from .train_meta import run_meta_training
        meta_out = await run_meta_training(timeframes)
        envelope = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "loop": {"kind": "fast", "reason": "meta_only", "min_new_rows": FAST_LOOP_MIN_NEW_ROWS},
            "meta_models": meta_out,
        }
        try:
            registry_module.REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
            (registry_module.REGISTRY_ROOT / "fast_loop_report.json").write_text(
                json.dumps(envelope, indent=2, default=str),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fast_loop_report_write_failed", extra={"error": str(exc)})
        return envelope
    finally:
        await close_pool()


# --- Orchestration -------------------------------------------------------------
async def run_training(
    coin_ids: Sequence[str],
    timeframes: Sequence[str],
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Drive a slow-loop training pass.

    `progress_callback`, when provided, is invoked synchronously with a
    structured dict for major heartbeat events: per-timeframe
    `build_dataset_start` / `build_dataset_done`, per-coin slice
    start/done (forwarded by `train_timeframe`), and per-timeframe
    `train_done`. The orchestrator wires this to `_append_progress` so
    operators tailing `models/progress_updates.jsonl` see live progress
    instead of a long silent gap during the multi-timeframe inner loop
    (task #380). Failures inside the callback are caught and logged so a
    flaky writer never aborts a training run.
    """
    def _emit(rec: dict) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(rec)
        except Exception as exc:  # noqa: BLE001 - progress is best-effort
            logger.warning("progress_callback_failed", extra={"error": str(exc)})

    await init_pool()
    try:
        # Task #236 — snapshot the previous run's report BEFORE we
        # overwrite it, so the auto-retire guardrail can compare
        # this run's pooled validation metrics against the prior model.
        # NOTE: kept under a distinct name (`previous_run_report`) so
        # the per-timeframe `prior_pooled_report` fallback below cannot
        # clobber it inside the loop.
        previous_run_report: Optional[dict] = None
        try:
            previous_report_path = registry_module.REGISTRY_ROOT / "report.json"
            if previous_report_path.exists():
                previous_run_report = json.loads(previous_report_path.read_text())
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("previous_run_report_load_failed", extra={"error": str(exc)})
            previous_run_report = None
        # Task #231 — pull the operator-approved feature-lab specs once at
        # the start of the run so every timeframe sees the same schema.
        # The bridge tolerates a missing/empty/malformed app_settings row
        # and returns []; in that case the training pipeline behaves
        # exactly like before this task.
        from .approved_features import (
            apply_approved_features,
            extend_feature_columns,
            fetch_approved_features,
        )
        approved = await fetch_approved_features()
        if approved:
            logger.info(
                "approved_features_loaded count=%d names=%s",
                len(approved), [a["name"] for a in approved],
            )
        # Task #417 — record both the legacy global lookback (kept for
        # backwards compat with downstream report consumers / dashboards)
        # AND the per-timeframe map actually used to build each dataset.
        # The legacy field reports the maximum window in use so a glance
        # at the report still answers "how far back did we train?".
        lookback_per_tf = {tf: lookback_days_for(tf) for tf in timeframes}
        report: dict = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "coin_ids": list(coin_ids),
            "lookback_days": (
                max(lookback_per_tf.values()) if lookback_per_tf else LOOKBACK_DAYS
            ),
            "lookback_days_per_tf": lookback_per_tf,
            "timeframes": {},
            "approved_features": [
                {"name": a["name"],
                 "transform_kind": a["transform_kind"],
                 "source_column": a["source_column"]}
                for a in approved
            ],
        }
        # Task #267 — record which loop produced this run. The default
        # entry point is the slow loop (full base + meta retrain); the
        # fast-loop entry point lives at `run_meta_only`. The orchestrator
        # never silently runs both on the same tick — see
        # `should_run_slow_loop` / `should_run_fast_loop` for the
        # decision helpers consumed by the scheduler.
        report["loop"] = {
            "kind": "slow",
            "reason": os.environ.get("ML_SLOW_LOOP_REASON", "manual_or_scheduled"),
            "fast_loop_min_new_rows": FAST_LOOP_MIN_NEW_ROWS,
            "slow_loop_new_ticks_threshold": SLOW_LOOP_NEW_TICKS_THRESHOLD,
        }
        for tf_idx, tf in enumerate(timeframes, start=1):
            n_tfs = len(timeframes)
            logger.info(f"build_dataset_start timeframe={tf}")
            _emit({
                "phase": "build_dataset_start",
                "status": "running",
                "headline": f"build_dataset_start tf={tf} ({tf_idx}/{n_tfs})",
                "timeframe": tf,
            })
            build_t0 = time.time()
            tf_provenance: dict[str, dict] = {}
            # Task #417 — resolve the lookback window for THIS timeframe
            # (1d gets the wider 3-year envelope; others keep 1y) so the
            # trainer reads the right number of bars for each cadence.
            tf_lookback_days = lookback_per_tf[tf]
            tf_lookback_ms = tf_lookback_days * 24 * 3600 * 1000
            df = await build_labeled_dataset(
                coin_ids, tf, tf_lookback_ms, provenance_out=tf_provenance,
            )
            logger.info(f"build_dataset_done timeframe={tf} rows={len(df)}")
            _emit({
                "phase": "build_dataset_done",
                "status": "ok",
                "headline": (
                    f"build_dataset_done tf={tf} rows={len(df)} "
                    f"in {round(time.time() - build_t0, 1)}s"
                ),
                "timeframe": tf,
                "n_rows": int(len(df)),
                "elapsed_sec": round(time.time() - build_t0, 2),
            })
            # Apply approved feature transforms BEFORE training so per-coin,
            # pooled, and specialist heads all see the extended schema and
            # the resulting `feature_schema_hash` reflects them.
            df, added_feature_names = apply_approved_features(df, approved)
            active_feature_columns = extend_feature_columns(
                FEATURE_COLUMNS, added_feature_names,
                categorical=CATEGORICAL_FEATURES,
            )
            if added_feature_names:
                logger.info(
                    f"approved_features_applied timeframe={tf} "
                    f"added={added_feature_names}"
                )
            # Task #267 — leakage audit + per-feature coverage + per-target
            # row counts. The audit runs on the FULL pre-training frame so a
            # planted future-leak fails the slice before any LightGBM cycles
            # are spent.
            #
            # Approved features (from Feature Lab) are auto-registered as
            # safe (max_lookforward=0) for the audit. They are derived
            # algebraically from already-registered base features by
            # `apply_approved_features`, so they inherit the point-in-time
            # property; without this the lineage gate would fail every
            # approved feature and silently halt retraining. The schema
            # and numerical layers still apply unchanged.
            audit_lineage = {
                **FEATURE_LINEAGE,
                **{name: {"max_lookforward": 0,
                          "max_lookback": None,
                          "auto_registered_via": "apply_approved_features"}
                   for name in added_feature_names},
            }
            leakage = audit_leakage(
                df,
                feature_columns=[c for c in active_feature_columns if c != "coin_idx"],
                target_columns=FORWARD_TARGET_COLUMNS,
                expected_horizon=FORWARD_HORIZON_CANDLES,
                feature_lineage=audit_lineage,
            )
            coverage = _summarize_feature_coverage(df, active_feature_columns)
            density = _summarize_feature_density(df, active_feature_columns)
            target_counts = _summarize_target_row_counts(df)
            provenance = _provenance_summary(tf_provenance)
            # Task #287 — assert (warn) that every registered external
            # stream actually delivered non-zero values this run. A
            # fully-covered, zero-density column means the trainer fell
            # through to `EXTERNAL_STREAM_DEFAULTS` for that stream.
            unwired_streams = _detect_unwired_external_streams(df)
            if unwired_streams and not df.empty:
                logger.warning(
                    "external_stream_unwired",
                    extra={
                        "timeframe": tf,
                        "columns": unwired_streams,
                        "n_rows": int(len(df)),
                    },
                )
            if not leakage["passed"]:
                logger.warning(
                    "leakage_audit_failed",
                    extra={"timeframe": tf, "violations": leakage["violations"]},
                )
                tf_report = {
                    "timeframe": tf,
                    "status": "leakage_audit_failed",
                    "n_rows": int(len(df)),
                    "per_coin": {},
                    "pooled": None,
                    "leakage_audit": leakage,
                    "feature_coverage": coverage,
                    "feature_density": density,
                    "unwired_external_streams": unwired_streams,
                    "target_row_counts": target_counts,
                    "provenance": provenance,
                }
                report["timeframes"][tf] = tf_report
                logger.info(
                    f"train_done timeframe={tf} status={tf_report['status']}"
                )
                continue
            tf_report = train_timeframe(
                df, tf, coin_ids, feature_columns=active_feature_columns,
                progress_callback=progress_callback,
            )
            tf_report["approved_features_applied"] = list(added_feature_names)
            tf_report["leakage_audit"] = leakage
            tf_report["feature_coverage"] = coverage
            tf_report["feature_density"] = density
            tf_report["unwired_external_streams"] = unwired_streams
            tf_report["target_row_counts"] = target_counts
            tf_report["provenance"] = provenance
            # If neither a per-coin nor a pooled LightGBM fit succeeded for
            # this timeframe, deploy a prior-only pooled model so /ml/predict
            # returns something honest instead of 503-ing — and so the api-
            # server's ml-availability cache stops marking the TF as a dead
            # zone that always falls through to the LLM brain.
            pooled_ok = tf_report.get("pooled") and tf_report["pooled"].get("status") == "trained"
            per_coin_ok = any(
                r.get("status") == "trained" for r in tf_report.get("per_coin", {}).values()
            )
            if not pooled_ok and not per_coin_ok:
                logger.info(f"train_prior_pooled_start timeframe={tf}")
                prior_pooled_report = await _train_prior_pooled(tf, coin_ids, tf_lookback_ms)
                tf_report["pooled"] = prior_pooled_report
                tf_report["status"] = "prior_pooled"
                logger.info(
                    f"train_prior_pooled_done timeframe={tf} "
                    f"n_labels={prior_pooled_report['n_labels']} "
                    f"priors={prior_pooled_report['prior_probs']}"
                )
            report["timeframes"][tf] = tf_report
            logger.info(f"train_done timeframe={tf} status={tf_report['status']}")
            per_coin_map = tf_report.get("per_coin") or {}
            n_trained = sum(
                1 for r in per_coin_map.values()
                if isinstance(r, dict) and r.get("status") == "trained"
            )
            _emit({
                "phase": "train_done",
                "status": tf_report.get("status") or "unknown",
                "headline": (
                    f"train_done tf={tf} status={tf_report.get('status')} "
                    f"trained_per_coin={n_trained}/{len(per_coin_map)} "
                    f"pooled={(tf_report.get('pooled') or {}).get('status', 'n/a')}"
                ),
                "timeframe": tf,
                "n_rows": tf_report.get("n_rows"),
                "per_coin_status": {
                    c: (r.get("status") if isinstance(r, dict) else None)
                    for c, r in per_coin_map.items()
                },
            })
        registry_module.REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
        (registry_module.REGISTRY_ROOT / "report.json").write_text(json.dumps(report, indent=2, default=str))
        # Persist the directional-call share timeseries (task #101). Failures
        # here never bubble up — training output is the contract.
        try:
            history_rows = _record_directional_share_for_report(report)
            report["directional_call_share_history_appended"] = len(history_rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("directional_share_history_failed", extra={"error": str(exc)})
        # Apply the retention policy so the JSONL never grows unbounded
        # (task #103). Trim runs after append so a fresh record is always
        # present even if the cap is small.
        try:
            trim_summary = _trim_directional_share_history()
            report["directional_call_share_history_trim"] = trim_summary
        except Exception as exc:  # noqa: BLE001
            logger.warning("directional_share_history_trim_failed", extra={"error": str(exc)})
        # Persist the regression-head holdout-stats timeseries (task #157).
        # Same best-effort contract as the directional-call-share history:
        # any failure here is logged and swallowed so the training output
        # contract is preserved.
        try:
            reg_rows = _record_regression_head_for_report(report)
            report["regression_head_history_appended"] = len(reg_rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("regression_head_history_failed", extra={"error": str(exc)})
        try:
            reg_trim = _trim_regression_head_history()
            report["regression_head_history_trim"] = reg_trim
        except Exception as exc:  # noqa: BLE001
            logger.warning("regression_head_history_trim_failed", extra={"error": str(exc)})
        # Auto-recalibrate the live quant-brain decision thresholds against
        # the freshly-trained model (task #137). Off by default — a retrain
        # only triggers the sweep when ML_RECALIBRATE_THRESHOLDS=1 is set.
        # Production CLI runs with the flag on so a model whose regression
        # head shifted can't silently re-introduce the "0 trades" failure
        # that prompted task #130. The sweep always writes a reviewable
        # proposal to models/calibration_recommendation.json; the JSON is
        # only mutated when ML_RECALIBRATE_THRESHOLDS_APPLY=1 is *also* set.
        if os.environ.get("ML_RECALIBRATE_THRESHOLDS") == "1":
            try:
                from .threshold_calibration import (  # local import to keep cold start cheap
                    DEFAULT_TIMEFRAME, recalibrate_after_training,
                )
                tf = os.environ.get("ML_RECALIBRATE_TIMEFRAME", DEFAULT_TIMEFRAME)
                apply = os.environ.get("ML_RECALIBRATE_THRESHOLDS_APPLY") == "1"
                proposal = recalibrate_after_training(
                    report, timeframe=tf, apply=apply,
                )
                report["threshold_recalibration"] = {
                    "timeframe": proposal.get("timeframe"),
                    "status": proposal.get("status"),
                    "applied": proposal.get("applied", False),
                    "model_hash": proposal.get("model_hash"),
                    "recommendation_status": proposal.get("recommendation_status"),
                    "would_apply": (proposal.get("change") or {}).get("would_apply", False),
                }
            except Exception as exc:  # noqa: BLE001 - best-effort; training contract is sacred
                logger.warning("threshold_recalibration_failed", extra={"error": str(exc)})
                report["threshold_recalibration"] = {"status": "error", "error": str(exc)}
        # Phase 4 — train (or heuristically deploy) the meta-model that
        # decides action / size / abstain on top of the base + specialist
        # heads. Best-effort; never aborts the training contract.
        try:
            from .train_meta import run_meta_training
            meta_out = await run_meta_training(timeframes)
            report["meta_models"] = meta_out
        except Exception as exc:  # noqa: BLE001
            logger.warning("meta_training_failed", extra={"error": str(exc)})
            report["meta_models"] = {"status": "error", "error": str(exc)}
        # Task #236 — compare this run's pooled validation metrics
        # against the prior run's, and auto-quarantine any approved
        # feature whose first appearance in the trained schema
        # correlates with a log_loss regression beyond the threshold.
        # Best-effort; never aborts the training contract.
        try:
            from .auto_retire import auto_retire_after_training
            retire_summary = await auto_retire_after_training(
                report, previous_run_report, approved,
            )
            report["auto_retired_features"] = retire_summary
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto_retire_failed", extra={"error": str(exc)})
            report["auto_retired_features"] = {
                "status": "error", "reason": str(exc),
            }
        # Task #220 — auto-register every freshly-trained slice into the
        # `model_registry` table as a `shadow` row so promote/rollback
        # picks the new version up without an operator POST. Idempotent
        # and best-effort; never aborts the training contract.
        try:
            from .register_shadow import register_shadow_rows
            shadow_summary = await register_shadow_rows(report)
            report["registry_shadow_registration"] = shadow_summary
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "registry_shadow_registration_failed",
                extra={"error": str(exc)},
            )
            report["registry_shadow_registration"] = {
                "status": "error", "error": str(exc),
            }
        # Task #306 — on-real-data verification gate. Walk every per-coin
        # and pooled slice; classify as promoted / no_lift / below_coinflip
        # / insufficient_sample / contract_failed using the walk-forward
        # holdout metrics already computed by `train_one_slice`. The
        # top-level `verification.passed` is true only when every active
        # coin has at least one promoted slice. Best-effort — a failure
        # here writes `status=error` and never aborts training.
        try:
            from .verification import build_verification_block
            report["verification"] = build_verification_block(
                report, list(coin_ids),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("verification_block_failed", extra={"error": str(exc)})
            report["verification"] = {"status": "error", "error": str(exc)}
        # Task #418 — stamp each per-slice verdict next to its manifest so
        # `_resolve_for_predict` can decide at serve time whether to keep
        # serving a per-coin slice that landed in the noise band, or to
        # defer to the pooled fallback for the same timeframe. The
        # verification block's `per_slice` array is the canonical source
        # — we just project (coin, tf, kind) -> the slice's resolved
        # version on disk and persist the verdict dict verbatim. Best-
        # effort: a write failure logs and continues so the training
        # contract is never blocked by a verdict disk write.
        try:
            ver_block = report.get("verification") or {}
            per_slice = ver_block.get("per_slice") if isinstance(ver_block, dict) else None
            timeframes_block = report.get("timeframes") or {}
            n_written = 0
            n_skipped = 0
            for verdict in (per_slice or []):
                if not isinstance(verdict, dict):
                    continue
                coin = verdict.get("coin")
                tf = verdict.get("timeframe")
                if not coin or not tf:
                    continue
                tf_report = timeframes_block.get(tf)
                if not isinstance(tf_report, dict):
                    n_skipped += 1
                    continue
                if verdict.get("kind") == "pooled":
                    pooled_slice = tf_report.get("pooled")
                    slc_coin = registry_module.POOLED_COIN_ID
                    version = (pooled_slice or {}).get("version") if isinstance(pooled_slice, dict) else None
                else:
                    per_coin_slc = (tf_report.get("per_coin") or {}).get(coin)
                    slc_coin = coin
                    version = per_coin_slc.get("version") if isinstance(per_coin_slc, dict) else None
                if not version:
                    # Untrained / contract_failed shells have no version
                    # on disk to attach the verdict to. They are
                    # already represented in the report's verification
                    # block; nothing to write per-slice.
                    n_skipped += 1
                    continue
                written = registry_module.write_verification_verdict(
                    slc_coin, tf, version, verdict,
                )
                if written is not None:
                    n_written += 1
                else:
                    n_skipped += 1
            report["verification_verdicts_written"] = {
                "written": n_written, "skipped": n_skipped,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "verification_verdict_stamp_failed", extra={"error": str(exc)},
            )
            report["verification_verdicts_written"] = {
                "status": "error", "error": str(exc),
            }
        # Training watchdog — append the verification snapshot to the
        # history JSONL and compute the diff vs the previous run so
        # operators can watch the brain converge / stall / regress
        # across retrains without polling. Best-effort.
        try:
            from .watchdog import record_verification_snapshot
            report["verification_diff"] = record_verification_snapshot(
                report, registry_module.REGISTRY_ROOT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("watchdog_record_failed", extra={"error": str(exc)})
            report["verification_diff"] = {"status": "error", "error": str(exc)}
        # Task #327 — auto-generated failure-analysis pair from the
        # in-memory report (no offline re-inference). Best-effort; never
        # aborts training. Writes to artifacts/ml-engine/reports/.
        try:
            from .failure_analysis import generate_for_report
            report["failure_analysis"] = generate_for_report(
                report, registry_module.REGISTRY_ROOT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("failure_analysis_record_failed", extra={"error": str(exc)})
            report["failure_analysis"] = {"status": "error", "error": str(exc)}
        # Task #376 — assert every active (coin, tf) slot resolves to a
        # fresh, non-archived manifest after this run. The archive sweep
        # (scripts/archive_contaminated_models.py) renames the `latest`
        # pointer out of the way, and a future regression in `run_training`
        # could leave a slot un-replaced — silently pausing the agent for
        # that coin/timeframe (every prediction would 503 to the LLM
        # brain). Surfaced as a loud WARNING per missing slot in training
        # output, plus a structured block in the report.
        try:
            slot_audit = registry_module.audit_active_slots(
                list(coin_ids), list(timeframes),
            )
            report["active_slot_audit"] = slot_audit
            if not slot_audit["ok"]:
                logger.warning(
                    "active_slot_audit_failed n_missing=%d n_checked=%d",
                    len(slot_audit["missing"]), slot_audit["n_checked"],
                )
                for row in slot_audit["missing"]:
                    logger.warning(
                        "active_slot_missing coin=%s tf=%s state=%s",
                        row["coin_id"], row["timeframe"], row["slot_state"],
                    )
        except Exception as exc:  # noqa: BLE001 - best-effort
            logger.warning("active_slot_audit_failed", extra={"error": str(exc)})
            report["active_slot_audit"] = {"status": "error", "error": str(exc)}
        # Re-write report.json with the post-processing fields populated
        # (auto-retire, meta, threshold recalibration, shadow registration,
        # verification) so the next run's prior_report comparison and the
        # diagnostics card see the full picture, not just the per-timeframe
        # results.
        try:
            (registry_module.REGISTRY_ROOT / "report.json").write_text(
                json.dumps(report, indent=2, default=str)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("final_report_write_failed", extra={"error": str(exc)})
        return report
    finally:
        await close_pool()


def main():  # pragma: no cover - CLI
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="*", default=DEFAULT_COINS)
    parser.add_argument("--timeframes", nargs="*", default=DEFAULT_TIMEFRAMES)
    args = parser.parse_args()
    report = asyncio.run(run_training(args.coins, args.timeframes))
    print(json.dumps({tf: r.get("status") for tf, r in report["timeframes"].items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
