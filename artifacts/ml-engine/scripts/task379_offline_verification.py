"""Task #379 offline verification — quantify the promotion-gate impact of
the new per-timeframe directional-label horizon WITHOUT re-running the
full training campaign (which is infeasible inside the agent shell:
single per-coin slices via `train_one_slice` exceed the 110s per-command
timeout because of the regressor head + per-class calibration the gate
does not even consume).

Approach
--------
Per-coin labeled parquets are already cached on disk (those are the
exact frames the campaign feeds into `train_one_slice`). Each cached
row carries:

  * `forward_return`           – legacy 1-bar pct change (fraction)
  * `forward_window_return_pct`– 4-bar cumulative pct change (percent),
                                  i.e. the trade-aware horizon the new
                                  directional label uses at 1h/2h/6h
  * `label_3class`             – the OLD 1-bar directional label

For each (coin, timeframe in {1h, 2h, 6h, 1d}) we:

  1. Score the OLD label (the cached `label_3class`) through the same
     walk-forward classifier the production gate consumes.
  2. Score the NEW label — for 1h/2h/6h this is
     `label_three_class(forward_window_return_pct,
                        MULTI_BAR_LABEL_THRESHOLDS_PERCENT[tf])`;
     for 1d the horizon doesn't change so NEW == OLD.
  3. Apply the verification watchdog gate
     (`directional_accuracy > 0.50` AND `> baseline_directional_accuracy`
     AND sum-of-folds `n_test >= 200`).
  4. Tally promoted-slice counts under each label scheme.

We invoke the SAME building blocks `train_one_slice` uses internally
(`walk_forward_splits`, `_optuna_search_lgb` with `ML_SKIP_OPTUNA=1`,
`_train_baseline`, `_baseline_predict`, `_directional_accuracy`) — but
we skip the regressor head and per-class calibration since the gate
does not depend on them. This gives an apples-to-apples preview of the
gate behaviour the next campaign run will produce, in single-digit
minutes instead of hours.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ML_SKIP_OPTUNA", "1")
os.environ.setdefault("ML_LGB_NUM_BOOST_ROUND", "40")

import warnings  # noqa: E402
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.training import labels as labels_mod  # noqa: E402
from app.training.train import (  # noqa: E402
    _baseline_predict, _directional_accuracy, _encode_coin_idx,
    _optuna_search_lgb, _prepare_xy, _train_baseline,
)
from app.training.walk_forward import WalkForwardConfig, walk_forward_splits  # noqa: E402
from app.training.registry import FEATURE_COLUMNS  # noqa: E402

DATASETS = {
    "1h": ROOT / "models/datasets/1h_20260423T014825Z.parquet",
    "2h": ROOT / "models/datasets/2h_20260423T030018Z.parquet",
    "6h": ROOT / "models/datasets/6h_20260423T035301Z.parquet",
    "1d": ROOT / "models/datasets/1d_20260423T043348Z.parquet",
}

# Verification gate (mirrors `app/training/verification.py`).
MIN_DA = 0.50
MIN_HOLDOUT_ROWS = 200
MIN_TRAIN_ROWS = 80


def _label_three_class(values: np.ndarray, threshold_pct: float) -> np.ndarray:
    out = np.full(values.shape[0], 1, dtype=np.int64)  # STABLE
    out[values >= threshold_pct] = 2  # UP
    out[values <= -threshold_pct] = 0  # DOWN
    return out


def _evaluate_slice(df: pd.DataFrame, vocab: list[str]) -> dict:
    """Run walk-forward + LGB classifier + multinomial baseline ONLY
    (skip the regressor head and per-class calibration that
    `train_one_slice` wires up — they do not feed the verification gate).
    Returns a dict shaped like the gate expects: status, metrics,
    baseline_metrics, folds (each with n_test).
    """
    if df is None or df.empty or len(df) < MIN_TRAIN_ROWS:
        return {"status": "insufficient_data", "n_rows": int(0 if df is None else len(df))}
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df = _encode_coin_idx(df, vocab)
    X_all, y_all = _prepare_xy(df, feature_columns=FEATURE_COLUMNS)
    n_folds = max(2, min(5, len(df) // 30))
    cfg = WalkForwardConfig(
        n_folds=n_folds, min_train_size=max(50, len(df) // (n_folds + 1)),
    )
    folds: list[dict] = []
    for k, (tr_idx, te_idx) in enumerate(walk_forward_splits(df, cfg)):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all.iloc[te_idx], y_all[te_idx]
        try:
            _, booster = _optuna_search_lgb(X_tr, y_tr)
            lgb_pred = booster.predict(X_te, num_iteration=booster.best_iteration)
            if lgb_pred.ndim == 1:
                lgb_pred = np.tile([0, 1, 0], (len(X_te), 1)).astype(float)
        except Exception as exc:  # noqa: BLE001
            return {"status": "lgb_failed", "error": str(exc), "n_rows": int(len(df))}
        try:
            enc, lr, priors = _train_baseline(X_tr, y_tr)
            base_pred = _baseline_predict(enc, lr, priors, X_te)
        except Exception as exc:  # noqa: BLE001
            return {"status": "baseline_failed", "error": str(exc), "n_rows": int(len(df))}
        folds.append({
            "fold": k, "n_test": int(len(te_idx)),
            "directional_accuracy": _directional_accuracy(y_te, lgb_pred),
            "baseline_directional_accuracy": _directional_accuracy(y_te, base_pred),
        })
    if not folds:
        return {"status": "no_folds", "n_rows": int(len(df))}
    da = float(np.mean([f["directional_accuracy"] for f in folds]))
    base_da = float(np.mean([f["baseline_directional_accuracy"] for f in folds]))
    return {
        "status": "trained", "n_rows": int(len(df)),
        "metrics": {"directional_accuracy": da},
        "baseline_metrics": {"directional_accuracy": base_da},
        "folds": folds,
    }


def _gate(result: dict) -> tuple[bool, str]:
    if result.get("status") != "trained":
        return False, f"status={result.get('status')}"
    folds = result.get("folds") or []
    n_test_total = int(sum(f.get("n_test", 0) for f in folds))
    if n_test_total < MIN_HOLDOUT_ROWS:
        return False, f"n_test={n_test_total}<{MIN_HOLDOUT_ROWS}"
    da = float(result["metrics"]["directional_accuracy"])
    base_da = float(result["baseline_metrics"]["directional_accuracy"])
    if not (da > MIN_DA):
        return False, f"DA={da:.4f}<=0.50"
    if not (da > base_da):
        return False, f"DA={da:.4f}<=baseline={base_da:.4f}"
    return True, f"DA={da:.4f}>base={base_da:.4f},n_test={n_test_total}"


def _run_for_tf(tf: str, parquet: Path, summary: dict) -> None:
    if not parquet.exists():
        print(f"[{tf}] missing cache {parquet.name}, skipping", flush=True)
        return
    full = pd.read_parquet(parquet)
    coins = sorted(full["coin_id"].dropna().astype(str).unique().tolist())
    new_thr = labels_mod.MULTI_BAR_LABEL_THRESHOLDS_PERCENT.get(tf)
    horizon_new = labels_mod.DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF.get(tf, 1)
    print(f"\n=== timeframe={tf} coins={len(coins)} new_thr={new_thr} "
          f"horizon_new={horizon_new} ===", flush=True)
    for coin in coins:
        sub = full[full["coin_id"] == coin].copy()
        if len(sub) < MIN_TRAIN_ROWS:
            print(f"  [{tf}/{coin}] insufficient_rows={len(sub)}", flush=True)
            continue
        old_df = sub.copy()
        new_df = sub.copy()
        if (horizon_new > 1 and new_thr is not None
                and "forward_window_return_pct" in new_df.columns):
            fwret = new_df["forward_window_return_pct"].astype(float).to_numpy()
            new_df["label_3class"] = _label_three_class(fwret, float(new_thr))
        t0 = time.time()
        old_res = _evaluate_slice(old_df, coins)
        t1 = time.time()
        new_res = _evaluate_slice(new_df, coins)
        t2 = time.time()
        old_pass, old_why = _gate(old_res)
        new_pass, new_why = _gate(new_res)
        print(
            f"  [{tf}/{coin}] OLD={'PASS' if old_pass else 'fail'}({old_why}) "
            f"| NEW={'PASS' if new_pass else 'fail'}({new_why}) "
            f"| t_old={t1-t0:.1f}s t_new={t2-t1:.1f}s", flush=True,
        )
        summary["per_slice"].append({
            "timeframe": tf, "coin": coin,
            "old_pass": old_pass, "old_reason": old_why,
            "new_pass": new_pass, "new_reason": new_why,
            "old_da": old_res.get("metrics", {}).get("directional_accuracy"),
            "new_da": new_res.get("metrics", {}).get("directional_accuracy"),
            "old_baseline_da": old_res.get("baseline_metrics", {}).get("directional_accuracy"),
            "new_baseline_da": new_res.get("baseline_metrics", {}).get("directional_accuracy"),
            "old_status": old_res.get("status"),
            "new_status": new_res.get("status"),
            "old_n_test_total": int(sum(f.get("n_test", 0) for f in old_res.get("folds", []))),
            "new_n_test_total": int(sum(f.get("n_test", 0) for f in new_res.get("folds", []))),
            "n_rows": int(len(sub)),
            "elapsed_seconds": round((t2 - t0), 2),
        })


def main() -> int:
    progress_path = ROOT / "reports" / "20260423T204419Z-task379-offline-progress.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_fp = progress_path.open("w", buffering=1)
    _orig = sys.stdout

    class _Tee:
        def write(self, s: str) -> int:
            try:
                progress_fp.write(s)
            except Exception:  # noqa: BLE001
                pass
            return _orig.write(s)
        def flush(self) -> None:
            progress_fp.flush()
            _orig.flush()

    sys.stdout = _Tee()
    summary: dict = {"per_slice": []}
    t0 = time.time()
    for tf, path in DATASETS.items():
        _run_for_tf(tf, path, summary)
    elapsed = time.time() - t0
    old_pass = sum(1 for s in summary["per_slice"] if s["old_pass"])
    new_pass = sum(1 for s in summary["per_slice"] if s["new_pass"])
    by_tf: dict = {}
    for s in summary["per_slice"]:
        b = by_tf.setdefault(s["timeframe"], {"old": 0, "new": 0, "n": 0})
        b["n"] += 1
        b["old"] += int(s["old_pass"])
        b["new"] += int(s["new_pass"])
    summary["totals"] = {
        "old_promoted": old_pass, "new_promoted": new_pass,
        "n_slices": len(summary["per_slice"]),
        "by_timeframe": by_tf,
        "elapsed_seconds": round(elapsed, 1),
        "lgb_num_boost_round": int(os.environ.get("ML_LGB_NUM_BOOST_ROUND", "40")),
        "skip_optuna": bool(os.environ.get("ML_SKIP_OPTUNA") == "1"),
    }
    out = ROOT / "reports" / "20260423T204419Z-task379-offline-verification.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print("\n========== TOTALS ==========")
    print(json.dumps(summary["totals"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
