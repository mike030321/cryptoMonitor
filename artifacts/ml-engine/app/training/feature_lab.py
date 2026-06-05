"""Phase 6 — Feature Lab walk-forward ablation runner.

Given a candidate feature spec (name + transform_kind + optional source
column), trains a 3-fold walk-forward LightGBM classifier on the latest
persisted training-frame snapshot, with vs. without the candidate
column appended. Returns multi-class log-loss and directional accuracy
deltas so the api-server can store an ablation report.

The transforms are deterministic, allow-listed, and computed from the
existing FEATURE_COLUMNS — no user-supplied code is ever evaluated. This
keeps the lab safe to expose behind admin auth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .registry import FEATURE_COLUMNS, CATEGORICAL_FEATURES, REGISTRY_ROOT


SUPPORTED_TRANSFORMS = {
    "passthrough_existing",
    "log_realized_vol",
    "rsi_squared",
    "macd_x_atr",
    "ret5_minus_ret10",
    "bb_pctb_squared",
}


def _apply_transform(
    df: pd.DataFrame, transform_kind: str, source_column: Optional[str], name: str,
) -> pd.DataFrame:
    out = df.copy()
    if transform_kind == "passthrough_existing":
        if not source_column or source_column not in df.columns:
            raise ValueError(
                f"passthrough_existing requires a valid source_column (got {source_column!r})",
            )
        out[name] = pd.to_numeric(df[source_column], errors="coerce").fillna(0.0)
    elif transform_kind == "log_realized_vol":
        out[name] = np.log1p(
            np.abs(pd.to_numeric(df["realizedVol"], errors="coerce").fillna(0.0)),
        )
    elif transform_kind == "rsi_squared":
        v = pd.to_numeric(df["rsi14"], errors="coerce").fillna(50.0) - 50.0
        out[name] = v * v
    elif transform_kind == "macd_x_atr":
        macd = pd.to_numeric(df["macdLine"], errors="coerce").fillna(0.0)
        atr = pd.to_numeric(df["atrPct"], errors="coerce").fillna(0.0)
        out[name] = macd * atr
    elif transform_kind == "ret5_minus_ret10":
        a = pd.to_numeric(df["ret5"], errors="coerce").fillna(0.0)
        b = pd.to_numeric(df["ret10"], errors="coerce").fillna(0.0)
        out[name] = a - b
    elif transform_kind == "bb_pctb_squared":
        v = pd.to_numeric(df["bbPctB"], errors="coerce").fillna(0.5) - 0.5
        out[name] = v * v
    else:
        raise ValueError(f"unsupported transform_kind: {transform_kind}")
    return out


def _latest_dataset_path(timeframe: str) -> Optional[Path]:
    ds_dir = REGISTRY_ROOT / "datasets"
    if not ds_dir.exists():
        return None
    cands = sorted(ds_dir.glob(f"{timeframe}_*.parquet"))
    return cands[-1] if cands else None


def _load_dataset(timeframe: str) -> pd.DataFrame:
    p = _latest_dataset_path(timeframe)
    if p is None:
        raise FileNotFoundError(
            f"no dataset snapshot found for timeframe={timeframe}; train at least once first",
        )
    df = pd.read_parquet(p)
    if "label_3class" not in df.columns:
        raise ValueError(f"dataset {p} missing label_3class column")
    return df


def _walk_forward_metrics(
    df: pd.DataFrame, feature_names: list[str], n_folds: int = 3,
) -> tuple[float, float]:
    """Return (mean_log_loss, mean_directional_accuracy) over n_folds
    chronological walk-forward splits. Trained head is a small LightGBM
    multiclass — same family as the production model so deltas correspond
    to real production-time behaviour, just with a fixed compact config
    so the runner stays inside a few seconds.
    """
    import lightgbm as lgb

    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    n = len(df)
    if n < 200:
        raise ValueError(f"dataset too small for walk-forward ({n} rows)")
    fold_size = n // (n_folds + 1)
    losses: list[float] = []
    accs: list[float] = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_end = min(n, fold_size * (k + 1))
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        if len(train) < 50 or len(test) < 20:
            continue
        X_tr = train[feature_names].astype(float).fillna(0.0)
        y_tr = train["label_3class"].astype(int).to_numpy()
        X_te = test[feature_names].astype(float).fillna(0.0)
        y_te = test["label_3class"].astype(int).to_numpy()
        cats = [c for c in CATEGORICAL_FEATURES if c in feature_names]
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "num_leaves": 15,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": -1,
            "deterministic": True,
            "seed": 42,
        }
        train_set = lgb.Dataset(
            X_tr, label=y_tr,
            categorical_feature=cats if cats else "auto",
            free_raw_data=False,
        )
        booster = lgb.train(
            params, train_set,
            num_boost_round=80,
            callbacks=[lgb.log_evaluation(0)],
        )
        proba = booster.predict(X_te)
        # Clip to avoid log(0); 3-class log-loss.
        eps = 1e-12
        p = np.clip(proba, eps, 1 - eps)
        ll = -float(
            np.mean(np.log(p[np.arange(len(y_te)), y_te])),
        )
        losses.append(ll)
        # Directional accuracy: argmax over (DOWN=0, UP=2) only — ignore
        # rows where the test row was STABLE so a model that always
        # predicts STABLE doesn't get a free pass.
        nonstab = y_te != 1
        if nonstab.any():
            pred_dir = np.where(p[nonstab, 2] > p[nonstab, 0], 2, 0)
            accs.append(float(np.mean(pred_dir == y_te[nonstab])))
        else:
            accs.append(float("nan"))
    if not losses:
        raise ValueError("no usable walk-forward folds (dataset too small after splits)")
    return float(np.mean(losses)), float(np.nanmean(accs))


def run_ablation(
    name: str,
    transform_kind: str,
    source_column: Optional[str],
    timeframe: str,
    coin_id: str = "__pooled__",
    n_folds: int = 3,
) -> dict:
    if transform_kind not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"unsupported transform_kind {transform_kind!r}; allowed: {sorted(SUPPORTED_TRANSFORMS)}",
        )
    df = _load_dataset(timeframe)
    if coin_id != "__pooled__" and "coin_id" in df.columns:
        df = df[df["coin_id"] == coin_id]
        if df.empty:
            raise ValueError(f"no rows for coin_id={coin_id} in dataset {timeframe}")

    base_features = list(FEATURE_COLUMNS)
    aug_df = _apply_transform(df, transform_kind, source_column, name)
    aug_features = base_features + [name]

    base_loss, base_acc = _walk_forward_metrics(df, base_features, n_folds=n_folds)
    aug_loss, aug_acc = _walk_forward_metrics(aug_df, aug_features, n_folds=n_folds)
    return {
        "nSamples": int(len(df)),
        "nFolds": int(n_folds),
        "baselineLogLoss": float(base_loss),
        "augmentedLogLoss": float(aug_loss),
        "baselineAccuracy": float(base_acc) if not np.isnan(base_acc) else None,
        "augmentedAccuracy": float(aug_acc) if not np.isnan(aug_acc) else None,
        "extra": {
            "name": name,
            "transformKind": transform_kind,
            "sourceColumn": source_column,
            "timeframe": timeframe,
            "coinId": coin_id,
        },
    }
