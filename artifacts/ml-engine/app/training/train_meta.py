"""Phase 4 — meta-model trainer.

Persists a registry slot at `__meta__/{timeframe}/{version}/` containing:
  - meta_clf.txt   — LightGBM 3-class classifier (no_trade / long / short)
  - meta_reg.txt   — LightGBM regressor for `edge_after_costs`
  - manifest.json  — META manifest (see ModelManifest.meta_*)

When the prediction_journal has < MIN_ROWS_FOR_FIT resolved QUANT rows
for a timeframe, we deploy a HEURISTIC manifest instead of erroring —
the live `/ml/meta/predict` route reads `manifest.meta_kind == "heuristic"`
and applies a deterministic specialist-agreement + cost-cushion rule so
the wiring is testable from day-1, before any journal data exists.

`run_meta_training(timeframes)` is invoked from the end of
`run_training` so a single retrain refreshes both base + meta models.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import registry as registry_module
from .meta_dataset import META_FEATURE_COLUMNS, build_meta_dataset

logger = logging.getLogger(__name__)

META_COIN_ID = "__meta__"
ACTION_LABELS = ["no_trade", "long", "short"]   # index = class id
MIN_ROWS_FOR_FIT = 200      # below this we deploy the heuristic manifest
MIN_PER_CLASS_FOR_FIT = 25  # need each action class to be represented

# Calibration history (per-bucket reliability snapshot) is appended at
# the end of each meta-training run. Same JSONL convention as
# `directional_call_share` so the dashboard can plot drift over time.
META_CALIBRATION_HISTORY_PATH = (
    registry_module.REGISTRY_ROOT / "meta_calibration_history.jsonl"
)


def _make_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _meta_dir(timeframe: str, version: str) -> Path:
    return registry_module.REGISTRY_ROOT / META_COIN_ID / timeframe / version


def _write_latest(timeframe: str, version: str) -> None:
    p = registry_module.REGISTRY_ROOT / META_COIN_ID / timeframe / "latest"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(version)


def _calibration_buckets(df: pd.DataFrame, n_buckets: int = 10) -> list[dict]:
    """Per-decile realized-edge buckets used by both the dashboard and the
    /ml/training/history/meta-calibration endpoint.

    Bucket key = decile of |base_expected_return_pct|. For each bucket we
    record n, mean predicted edge, mean realized edge after costs, and
    the share of rows whose realized direction matched the predicted side.
    """
    if df.empty:
        return []
    abs_pred = df["base_expected_return_pct"].abs()
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(abs_pred, quantiles))
    if len(edges) < 2:
        return []
    out: list[dict] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (abs_pred >= lo) & (abs_pred <= hi if i == len(edges) - 2 else abs_pred < hi)
        sub = df[mask]
        if sub.empty:
            continue
        # Predicted direction was up iff base_prob_up >= base_prob_down
        pred_up = sub["base_prob_up"] >= sub["base_prob_down"]
        # Realized direction was up iff edge_after_costs aligned with up
        # (edge>0 when long action wins, <0 when short wins). For the hit
        # rate we use the underlying realized return sign: edge_after_costs
        # already had cost subtracted but the SIGN matches realized return.
        realized_up = sub["__edge_after_costs__"] > 0
        # Hit only counts if action == long matches realized up (or short
        # matches realized down). Since ACTION_LABELS is per-row, derive:
        action_long = (sub["__action__"] == "long").astype(int)
        action_short = (sub["__action__"] == "short").astype(int)
        side_match = ((pred_up & (action_long == 1)) | (~pred_up & (action_short == 1)))
        out.append({
            "bucket": i,
            "edge_lo_pct": lo,
            "edge_hi_pct": hi,
            "n": int(len(sub)),
            "mean_predicted_edge_pct": float(sub["base_expected_return_pct"].mean()),
            "mean_realized_edge_after_costs_pct": float(sub["__edge_after_costs__"].mean()),
            "hit_rate": float(side_match.mean()),
        })
    return out


def _save_meta_manifest(
    timeframe: str,
    version: str,
    *,
    meta_kind: str,
    n_train_rows: int,
    feature_names: list[str],
    metrics: dict,
    calibration_buckets: list[dict],
    note: str = "",
) -> None:
    out_dir = _meta_dir(timeframe, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "coin_id": META_COIN_ID,
        "timeframe": timeframe,
        "version": version,
        "model_kind": "meta",
        "meta_kind": meta_kind,            # "lightgbm" or "heuristic"
        "feature_names": list(feature_names),
        "n_train_rows": int(n_train_rows),
        "metrics": metrics,
        "calibration_buckets": calibration_buckets,
        "action_labels": ACTION_LABELS,
        "round_trip_cost_pct": 0.003,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _train_lightgbm(df: pd.DataFrame, timeframe: str, version: str) -> dict:
    import lightgbm as lgb
    X = df[META_FEATURE_COLUMNS].astype(float).to_numpy()
    y_action = df["__action__"].map({a: i for i, a in enumerate(ACTION_LABELS)}).to_numpy()
    y_edge = df["__edge_after_costs__"].to_numpy(dtype=float)

    # Time-ordered single split (not walk-forward — journal is small).
    split = max(1, int(len(df) * 0.8))
    Xtr, Xte = X[:split], X[split:]
    ytr_a, yte_a = y_action[:split], y_action[split:]
    ytr_e, yte_e = y_edge[:split], y_edge[split:]

    clf = lgb.train(
        params={
            "objective": "multiclass",
            "num_class": len(ACTION_LABELS),
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "verbosity": -1,
        },
        train_set=lgb.Dataset(Xtr, label=ytr_a),
        num_boost_round=200,
    )
    reg = lgb.train(
        params={
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "verbosity": -1,
        },
        train_set=lgb.Dataset(Xtr, label=ytr_e),
        num_boost_round=200,
    )

    out_dir = _meta_dir(timeframe, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(out_dir / "meta_clf.txt"))
    reg.save_model(str(out_dir / "meta_reg.txt"))

    if len(Xte) > 0:
        proba = clf.predict(Xte)
        pred_action = np.asarray(proba).argmax(axis=1)
        action_acc = float((pred_action == yte_a).mean())
        edge_pred = reg.predict(Xte)
        edge_mae = float(np.mean(np.abs(np.asarray(edge_pred) - yte_e)))
    else:
        action_acc = float("nan")
        edge_mae = float("nan")

    metrics = {
        "action_accuracy_holdout": action_acc,
        "edge_mae_holdout_pct": edge_mae,
        "n_train": int(split),
        "n_holdout": int(len(df) - split),
        "class_counts": {a: int((df["__action__"] == a).sum()) for a in ACTION_LABELS},
    }
    return metrics


def deploy_heuristic(timeframe: str, n_rows: int, note: str) -> str:
    """Deploy the day-1 heuristic meta-model. /ml/meta/predict reads
    `meta_kind=heuristic` and applies a deterministic specialist-
    agreement + cost-cushion rule.
    """
    version = _make_version()
    _save_meta_manifest(
        timeframe, version,
        meta_kind="heuristic",
        n_train_rows=n_rows,
        feature_names=list(META_FEATURE_COLUMNS),
        metrics={"reason": "insufficient_data"},
        calibration_buckets=[],
        note=note,
    )
    _write_latest(timeframe, version)
    logger.info(f"meta_heuristic_deployed timeframe={timeframe} version={version} rows={n_rows}")
    return version


def _append_calibration_history(record: dict) -> None:
    META_CALIBRATION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_CALIBRATION_HISTORY_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


async def run_meta_training(timeframes: Iterable[str]) -> dict:
    """Train (or heuristically deploy) one meta-model per timeframe.

    Best-effort: failure on one timeframe never aborts the others.
    """
    out: dict[str, dict] = {}
    for tf in timeframes:
        try:
            df = await build_meta_dataset(timeframe=tf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("meta_dataset_failed", extra={"timeframe": tf, "error": str(exc)})
            v = deploy_heuristic(tf, 0, f"dataset_build_failed: {exc}")
            out[tf] = {"status": "heuristic", "version": v, "reason": str(exc)}
            continue
        n = len(df)
        class_counts = {a: int((df["__action__"] == a).sum()) for a in ACTION_LABELS} if n else {}
        below_min = n < MIN_ROWS_FOR_FIT or any(
            c < MIN_PER_CLASS_FOR_FIT for c in class_counts.values()
        )
        if below_min:
            v = deploy_heuristic(
                tf, n,
                note=f"need >= {MIN_ROWS_FOR_FIT} rows and >= {MIN_PER_CLASS_FOR_FIT} per class; have {n} rows {class_counts}",
            )
            out[tf] = {
                "status": "heuristic",
                "version": v,
                "n_rows": n,
                "class_counts": class_counts,
            }
            continue
        try:
            version = _make_version()
            metrics = _train_lightgbm(df, tf, version)
            calibration = _calibration_buckets(df)
            _save_meta_manifest(
                tf, version,
                meta_kind="lightgbm",
                n_train_rows=n,
                feature_names=list(META_FEATURE_COLUMNS),
                metrics=metrics,
                calibration_buckets=calibration,
                note=f"trained on {n} resolved QUANT rows",
            )
            _write_latest(tf, version)
            _append_calibration_history({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "timeframe": tf,
                "version": version,
                "n_rows": n,
                "class_counts": metrics.get("class_counts", {}),
                "action_accuracy_holdout": metrics.get("action_accuracy_holdout"),
                "edge_mae_holdout_pct": metrics.get("edge_mae_holdout_pct"),
                "calibration_buckets": calibration,
            })
            out[tf] = {
                "status": "trained",
                "version": version,
                "n_rows": n,
                "metrics": metrics,
            }
            logger.info(f"meta_lgb_trained timeframe={tf} version={version} acc={metrics.get('action_accuracy_holdout')}")
        except Exception as exc:  # noqa: BLE001 - never abort the rest
            logger.warning("meta_train_failed", extra={"timeframe": tf, "error": str(exc)})
            v = deploy_heuristic(tf, n, f"train_failed: {exc}")
            out[tf] = {"status": "heuristic", "version": v, "reason": str(exc)}
    return out


def load_meta_manifest(timeframe: str) -> Optional[dict]:
    p = registry_module.REGISTRY_ROOT / META_COIN_ID / timeframe / "latest"
    if not p.exists():
        return None
    v = p.read_text().strip()
    if not v:
        return None
    mp = _meta_dir(timeframe, v) / "manifest.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except Exception:
        return None


def load_meta_models(timeframe: str):
    """Return (manifest, clf_booster, reg_booster) or (manifest, None, None)
    when the manifest is heuristic-only.
    """
    manifest = load_meta_manifest(timeframe)
    if manifest is None:
        return None, None, None
    if manifest.get("meta_kind") != "lightgbm":
        return manifest, None, None
    import lightgbm as lgb
    d = _meta_dir(timeframe, manifest["version"])
    clf_path = d / "meta_clf.txt"
    reg_path = d / "meta_reg.txt"
    if not clf_path.exists() or not reg_path.exists():
        return manifest, None, None
    return manifest, lgb.Booster(model_file=str(clf_path)), lgb.Booster(model_file=str(reg_path))
