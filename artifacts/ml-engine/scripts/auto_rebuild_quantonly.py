"""Task #367 — Auto-rebuild trading models on the new quant-only contract.

Task #365 archived 1,116 contaminated model directories and the runtime
guard (`registry.load_model`) now refuses to load any archived manifest.
Every `latest` pointer was moved to `latest.archived_pre_quantonly_...`,
so `resolve_model()` returns `None` for every (coin, timeframe) slot and
the quant brain abstains from trading.

This script stands in for the 30-minute auto-retrain loop in an offline
environment: for every slot that still has `latest.archived_*` with no
live `latest`, it fits a fresh LightGBM 3-class classifier on the post-
#365 `FEATURE_COLUMNS` contract using the most recent labelled parquet
for the timeframe, persists a clean `ModelManifest` via
`registry.save_model`, and records before/after holdout metrics against
the most recent archived baseline.

Design constraints:
  * Fast — one direct 80/20 holdout fit per slot (no Optuna, no
    calibration search). The production 30-minute loop runs the full
    `train_one_slice` pipeline; this run unblocks agents immediately.
  * Deterministic — fixed seed, fixed hyperparameters.
  * Schema-clean — `feature_names` = live `FEATURE_COLUMNS`; the
    runtime guard verifies zero forbidden prefixes at load time.

Output:
  * `audit/auto_rebuild_quantonly.json` — per-slot before/after metrics.
  * Fresh version dir + `latest` pointer per rebuilt slot under
    `artifacts/ml-engine/models/<coin>/<tf>/<version>/`.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "artifacts" / "ml-engine"))

from app.training import registry as registry_module  # noqa: E402
from app.training.registry import (  # noqa: E402
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_PREFIXES,
    POOLED_COIN_ID,
    SPECIALIST_KINDS,
    SPECIALIST_REGIME_MAP,
    ModelManifest,
    is_specialist_coin_id,
    make_version,
    save_model,
    specialist_coin_id,
)

DATASETS = REPO_ROOT / "artifacts" / "ml-engine" / "models" / "datasets"
AUDIT = REPO_ROOT / "audit"
AUDIT.mkdir(exist_ok=True)
TIMEFRAMES = ("1m", "5m", "1h", "2h", "6h", "1d")

MIN_TRAIN_ROWS = 80
RANDOM_SEED = 20260423
# Label threshold defaults mirror app.training.labels.OUTCOME_THRESHOLDS_PERCENT
# fallback bands used by train_one_slice when a coin has no forward returns
# observed in one of the directional classes. Values are retained so manifest
# `class_return_means_pct` degrades gracefully on thin slices.
FALLBACK_THRESHOLD_PCT = {
    "1m": 0.25, "5m": 0.40, "1h": 0.60,
    "2h": 0.80, "6h": 1.20, "1d": 2.00,
}


def latest_parquet_for(tf: str) -> Optional[Path]:
    cands = sorted(DATASETS.glob(f"{tf}_*.parquet"))
    return cands[-1] if cands else None


def aggregate_parquets_for(tf: str, max_files: int = 40) -> Optional[pd.DataFrame]:
    """Union the most recent parquets for a timeframe and dedupe on
    (coin_id, timestamp_ms). The individual daily parquets often
    contain a single coin at a time; the union gives the trainer a
    multi-coin frame so per-coin slots can all be fit in one pass.
    """
    cands = sorted(DATASETS.glob(f"{tf}_*.parquet"))[-max_files:]
    if not cands:
        return None
    frames = []
    for p in cands:
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "timestamp_ms" in df.columns and "coin_id" in df.columns:
        df = df.sort_values("timestamp_ms").drop_duplicates(
            subset=["coin_id", "timestamp_ms"], keep="last"
        ).reset_index(drop=True)
    return df


def load_archived_baselines() -> dict[tuple[str, str], dict]:
    p = AUDIT / "archived_models.json"
    if not p.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    inv = json.loads(p.read_text())
    for entry in inv.get("models", []):
        key = (entry["coin_id"], entry["timeframe"])
        prev = out.get(key)
        if prev is None or (entry.get("version") or "") > (prev.get("version") or ""):
            out[key] = entry
    return out


def enumerate_slots_needing_rebuild() -> list[tuple[str, str]]:
    """Every (coin, tf) where `latest` is missing but
    `latest.archived_pre_quantonly_20260423` (or similar) exists, OR the
    live `latest` points at a manifest whose feature_names still carry
    forbidden prefixes.
    """
    slots: list[tuple[str, str]] = []
    root = registry_module.REGISTRY_ROOT
    for coin_dir in sorted(root.iterdir()):
        if not coin_dir.is_dir() or coin_dir.name in {
            "datasets", "training_history", "__meta__",
        }:
            continue
        coin = coin_dir.name
        for tf_dir in sorted(coin_dir.iterdir()):
            if not tf_dir.is_dir():
                continue
            tf = tf_dir.name
            if tf not in TIMEFRAMES:
                continue
            latest = tf_dir / "latest"
            archived_latest = any(
                p.name.startswith("latest.archived") for p in tf_dir.iterdir()
            )
            # Candidate if latest missing OR archived_latest marker present.
            if not latest.exists() or archived_latest:
                slots.append((coin, tf))
                continue
            # Otherwise inspect current latest manifest; if contaminated,
            # also a rebuild candidate.
            v = latest.read_text().strip()
            mf = tf_dir / v / "manifest.json"
            if mf.exists():
                feats = json.loads(mf.read_text()).get("feature_names", [])
                if any(
                    c.startswith(p) for c in feats
                    for p in FORBIDDEN_FEATURE_PREFIXES
                ):
                    slots.append((coin, tf))
    return slots


def _encode_coin_idx(df: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    df = df.copy()
    lookup = {c: i for i, c in enumerate(vocab)}
    df["coin_idx"] = df["coin_id"].map(lookup).fillna(-1).astype(int)
    return df


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col == "coin_idx":
            continue
        if col not in df.columns:
            df[col] = 0.0
    return df


def _specialist_target(df: pd.DataFrame, kind: str) -> Optional[np.ndarray]:
    """Very small re-implementation of the production cost-aware
    specialist target. DOWN/STABLE/UP = 0/1/2. Returns None when the
    slice lacks the trade-aware columns.
    """
    if kind == "volatility_forecaster":
        col = "realized_vol_next_horizon"
        if col not in df.columns:
            return None
        v = df[col].astype(float).fillna(0.0).values
        q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
        y = np.ones(len(v), dtype=int)
        y[v <= q1] = 0
        y[v >= q2] = 2
        return y
    # Directional specialists: DOWN iff tp_before_sl_short, UP iff
    # tp_before_sl_long, else STABLE. This mirrors the spirit of the
    # live `_specialist_target_3class` without its full cost-aware
    # filter. For audit-grade parity the production pipeline still
    # governs; this script only needs to produce a usable training
    # target so the manifest holds non-degenerate metrics.
    need = {"tp_before_sl_long", "tp_before_sl_short"}
    if not need.issubset(df.columns):
        return None
    y = np.ones(len(df), dtype=int)
    y[df["tp_before_sl_short"].fillna(0).astype(int).to_numpy() == 1] = 0
    y[df["tp_before_sl_long"].fillna(0).astype(int).to_numpy() == 1] = 2
    return y


def _directional_accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    """Overall argmax accuracy across all three classes. Matches
    production `app.training.train._directional_accuracy` so deltas
    vs. archived baselines are apples-to-apples.
    """
    if len(y) == 0:
        return 0.0
    pred = probs.argmax(axis=1)
    return float((pred == y).mean())


def _multiclass_macro_auc(y: np.ndarray, probs: np.ndarray) -> float:
    """One-vs-rest macro AUC over the 3 classes; NaN if fewer than 2
    classes are present in the holdout. This matches the metric the
    production training pipeline (`app.training.train._multiclass_macro_auc`)
    stores in `manifest.metrics.auc`, so deltas vs. archived baselines
    are apples-to-apples.
    """
    present = [k for k in range(3) if (y == k).any() and (y != k).any()]
    if len(present) < 2:
        return float("nan")
    aucs = [roc_auc_score((y == k).astype(int), probs[:, k]) for k in present]
    return float(np.mean(aucs))


def _brier_multi(y: np.ndarray, probs: np.ndarray) -> float:
    """Mean per-class Brier score — matches production
    `app.training.train._multiclass_brier`: for each of the three
    classes take the binary `brier_score_loss((y==k), proba[:, k])`
    and average. This is half the sum-of-squares row Brier used in
    early drafts of this script; using the production formula keeps
    before/after deltas apples-to-apples.
    """
    if len(y) == 0:
        return float("nan")
    scores = [brier_score_loss((y == k).astype(int), probs[:, k]) for k in (0, 1, 2)]
    return float(np.mean(scores))


def _log_loss_multi(y: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return float(log_loss(y, probs, labels=[0, 1, 2]))


def _class_return_means_pct(df: pd.DataFrame, y: np.ndarray) -> list[float]:
    """Per-class average forward_return, in percentage points."""
    if "forward_return" not in df.columns:
        return [0.0, 0.0, 0.0]
    ret = df["forward_return"].astype(float).fillna(0.0).to_numpy() * 100.0
    means = []
    for k in (0, 1, 2):
        m = ret[y == k]
        means.append(float(np.mean(m)) if len(m) else 0.0)
    return means


def fit_and_save(
    coin: str, tf: str, df: pd.DataFrame, vocab: list[str],
    baseline_metrics: dict,
) -> dict:
    """Fit a 3-class LightGBM on 80/20 chronological split and persist
    a fresh ModelManifest via save_model.
    """
    specialist_kind: Optional[str] = None
    regime_subset: list[str] = []

    sub = df
    if is_specialist_coin_id(coin):
        for k in SPECIALIST_KINDS:
            if coin == specialist_coin_id(k):
                specialist_kind = k
                regime_subset = list(SPECIALIST_REGIME_MAP[k])
                break
        if specialist_kind is None:
            return {"status": "unknown_specialist"}
        if specialist_kind != "volatility_forecaster" and regime_subset:
            if "regime" not in df.columns:
                return {"status": "no_regime_column"}
            sub = df[df["regime"].isin(regime_subset)].copy()
        y = _specialist_target(sub, specialist_kind)
        if y is None:
            return {"status": "no_specialist_target"}
    elif coin == POOLED_COIN_ID:
        sub = df.copy()
        y = sub["label_3class"].astype(int).to_numpy()
    else:
        sub = df[df["coin_id"] == coin].copy()
        if sub.empty:
            return {"status": "no_rows_for_coin"}
        y = sub["label_3class"].astype(int).to_numpy()

    if len(sub) < MIN_TRAIN_ROWS:
        return {"status": "insufficient_data", "n_rows": int(len(sub))}

    sub = _ensure_feature_columns(sub)
    sub = _encode_coin_idx(sub, vocab)
    if "timestamp_ms" in sub.columns:
        order = np.argsort(sub["timestamp_ms"].to_numpy())
        sub = sub.iloc[order].reset_index(drop=True)
        y = y[order]

    X = sub[FEATURE_COLUMNS].astype(float).fillna(0.0).to_numpy()
    n = len(sub)
    cut = max(MIN_TRAIN_ROWS, int(n * 0.8))
    if n - cut < 10:
        return {"status": "holdout_too_small", "n_rows": n}
    Xtr, ytr = X[:cut], y[:cut]
    Xte, yte = X[cut:], y[cut:]
    if len(set(ytr)) < 2:
        return {"status": "single_class_train", "n_rows": n}

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "verbose": -1,
        "num_threads": 4,
        "seed": RANDOM_SEED,
    }
    train_ds = lgb.Dataset(
        Xtr, label=ytr, feature_name=FEATURE_COLUMNS,
        categorical_feature=["coin_idx"],
        free_raw_data=False,
    )
    booster = lgb.train(params, train_ds, num_boost_round=60)
    probs = booster.predict(Xte)
    if probs.ndim == 1:
        probs = np.tile([0, 1, 0], (len(Xte), 1)).astype(float)
    # Per-class baseline (majority-class prior on train fold):
    priors = np.array([
        (ytr == k).mean() if (ytr == k).any() else 1e-6 for k in (0, 1, 2)
    ])
    priors = priors / priors.sum()
    base_probs = np.tile(priors, (len(Xte), 1))

    metrics = {
        "auc": _multiclass_macro_auc(yte, probs),
        "log_loss": _log_loss_multi(yte, probs),
        "brier": _brier_multi(yte, probs),
        "directional_accuracy": _directional_accuracy(yte, probs),
    }
    base_metrics = {
        "auc": _multiclass_macro_auc(yte, base_probs),
        "log_loss": _log_loss_multi(yte, base_probs),
        "brier": _brier_multi(yte, base_probs),
        "directional_accuracy": _directional_accuracy(yte, base_probs),
    }
    directional_call_share = float((probs.argmax(axis=1) != 1).mean())
    fold_results = [{
        "fold": 0, "n_train": int(cut), "n_test": int(n - cut),
        "test_auc": metrics["auc"], "test_log_loss": metrics["log_loss"],
        "test_brier": metrics["brier"],
        "test_directional_accuracy": metrics["directional_accuracy"],
    }]

    class_means = _class_return_means_pct(sub, y)
    thr = FALLBACK_THRESHOLD_PCT.get(tf, 0.5)
    if class_means[0] == 0.0:
        class_means[0] = -thr
    if class_means[2] == 0.0:
        class_means[2] = thr

    version = make_version()
    manifest = ModelManifest(
        coin_id=coin,
        timeframe=tf,
        version=version,
        feature_names=list(FEATURE_COLUMNS),
        coin_vocab=list(vocab),
        n_train_rows=int(cut),
        n_test_rows=int(n - cut),
        metrics=metrics,
        baseline_metrics=base_metrics,
        threshold_pct=thr,
        horizon_candles=int(sub.get("forward_horizon_candles", pd.Series([1])).iloc[0])
        if "forward_horizon_candles" in sub.columns else 1,
        class_return_means_pct=class_means,
        fold_metrics=fold_results,
        note="task#367 auto-rebuild quant-only contract (single chronological 80/20 holdout)",
        directional_call_share=directional_call_share,
        directional_call_share_n=int(n - cut),
        directional_call_share_source="holdout",
        model_kind="lightgbm",
        specialist_kind=specialist_kind,
        regime_subset=regime_subset,
        training_window={
            "start": datetime.fromtimestamp(
                int(sub["timestamp_ms"].min()) / 1000, tz=timezone.utc
            ).isoformat() if "timestamp_ms" in sub.columns else None,
            "end": datetime.fromtimestamp(
                int(sub["timestamp_ms"].max()) / 1000, tz=timezone.utc
            ).isoformat() if "timestamp_ms" in sub.columns else None,
        },
    )
    save_model(coin, tf, version, booster, calibrators=None, manifest=manifest)

    return {
        "status": "ok",
        "version": version,
        "n_rows": n,
        "n_train": int(cut),
        "n_test": int(n - cut),
        "metrics": metrics,
        "baseline_metrics": base_metrics,
        "directional_call_share": directional_call_share,
    }


def manifest_is_clean(coin: str, tf: str) -> tuple[bool, Optional[str]]:
    latest_p = registry_module.REGISTRY_ROOT / coin / tf / "latest"
    if not latest_p.exists():
        return False, None
    v = latest_p.read_text().strip()
    mp = registry_module.REGISTRY_ROOT / coin / tf / v / "manifest.json"
    if not mp.exists():
        return False, v
    feats = json.loads(mp.read_text()).get("feature_names", [])
    forbidden = any(
        c.startswith(p) for c in feats for p in FORBIDDEN_FEATURE_PREFIXES
    )
    return (not forbidden), v


def main() -> int:
    forbidden_in_contract = [
        c for c in FEATURE_COLUMNS
        if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)
    ]
    assert not forbidden_in_contract, (
        f"FEATURE_COLUMNS still has forbidden cols: {forbidden_in_contract}"
    )

    baselines = load_archived_baselines()
    print(f"loaded {len(baselines)} archived baselines")

    slots = enumerate_slots_needing_rebuild()
    print(f"found {len(slots)} slots needing rebuild")

    by_tf: dict[str, list[str]] = defaultdict(list)
    for coin, tf in slots:
        by_tf[tf].append(coin)

    out: dict = {
        "task": 367,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "forbidden_prefixes": list(FORBIDDEN_FEATURE_PREFIXES),
        "min_train_rows": MIN_TRAIN_ROWS,
        "method": (
            "Direct 3-class LightGBM 80/20 chronological holdout on the "
            "latest labelled parquet per timeframe. Mirrors the feature "
            "contract enforced by registry.FEATURE_COLUMNS. Skips the "
            "Optuna search + isotonic calibration of the full "
            "train_one_slice pipeline so the whole fleet rebuilds in a "
            "bounded window; the live 30-minute auto-retrain loop will "
            "refit each slot with the full pipeline on its next tick."
        ),
        "slots": [],
    }

    rebuilt = 0
    skipped = 0
    errors = 0

    for tf in TIMEFRAMES:
        coins = sorted(set(by_tf.get(tf, [])))
        if not coins:
            continue
        df = aggregate_parquets_for(tf)
        if df is None or df.empty:
            for coin in coins:
                out["slots"].append({
                    "coin_id": coin, "timeframe": tf, "status": "no_parquet",
                })
                skipped += 1
            continue
        parquet = latest_parquet_for(tf)
        if "coin_id" not in df.columns:
            for coin in coins:
                out["slots"].append({
                    "coin_id": coin, "timeframe": tf,
                    "status": "parquet_missing_coin_id",
                })
                skipped += 1
            continue
        vocab = sorted(df["coin_id"].dropna().unique().tolist())
        for coin in coins:
            base = baselines.get((coin, tf)) or {}
            res = fit_and_save(coin, tf, df, vocab, base.get("metrics") or {})
            clean, latest_v = manifest_is_clean(coin, tf)
            row = {
                "coin_id": coin,
                "timeframe": tf,
                "parquet": parquet.name,
                **res,
                "manifest_clean": clean,
                "latest_version": latest_v,
            }
            if base:
                row["before_version"] = base.get("version")
                row["before_metrics"] = base.get("metrics")
                bm = base.get("metrics") or {}
                am = res.get("metrics") or {}
                if bm and am and res.get("status") == "ok":
                    row["delta"] = {
                        k: round((am.get(k) or 0.0) - (bm.get(k) or 0.0), 5)
                        for k in ("auc", "log_loss", "brier", "directional_accuracy")
                        if bm.get(k) is not None and am.get(k) is not None
                    }
            out["slots"].append(row)
            if res.get("status") == "ok" and clean:
                rebuilt += 1
            elif res.get("status") in {
                "insufficient_data", "no_specialist_target",
                "no_regime_column", "no_rows_for_coin", "holdout_too_small",
                "single_class_train", "no_parquet", "parquet_missing_coin_id",
            }:
                skipped += 1
            else:
                errors += 1
            print(
                f"[{tf}] {coin}: status={res.get('status')} "
                f"clean={clean} auc={(res.get('metrics') or {}).get('auc')} "
                f"delta_auc={(row.get('delta') or {}).get('auc')}"
            )

    out["summary"] = {
        "total_slots_attempted": len(slots),
        "rebuilt_clean": rebuilt,
        "skipped": skipped,
        "errors": errors,
    }

    # Emit strict JSON: convert NaN/Inf floats to null so downstream
    # non-Python parsers (jq, browsers, strict linters) accept the file.
    def _sanitize(obj):
        if isinstance(obj, float):
            if obj != obj or obj in (float("inf"), float("-inf")):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        return obj

    out_path = AUDIT / "auto_rebuild_quantonly.json"
    out_path.write_text(
        json.dumps(_sanitize(out), indent=2, default=str, allow_nan=False)
    )
    print(f"\nwrote {out_path}")
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
