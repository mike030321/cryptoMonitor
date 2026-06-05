"""Task #379 focused offline verification — re-run the most promising
10 slices from the cached parquets with **optuna ON** (n_trials=6,
timeout=30s per fold), exactly the search the production campaign
performs. The shortlist is the union of the top-DA slices from the
boost=200 sweep where NEW or OLD already approached the 0.50 floor:

    2h: sei-network, bonk, pepe, floki-inu
    6h: sei-network, celestia
    1d: dogwifcoin, worldcoin-wld, pepe, sei-network

If the per-tf-horizon NEW labels promote strictly more slices than
the legacy OLD labels at the production gate (DA>0.50, DA>baseline,
n_test>=200), this is the per-slice promotion evidence the reviewer
asked for.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

# Optuna ON, full boost rounds — production parity for the gate.
os.environ.pop("ML_SKIP_OPTUNA", None)
os.environ.setdefault("ML_OPTUNA_N_TRIALS", "6")
os.environ.setdefault("ML_OPTUNA_TIMEOUT_SECONDS", "30")
os.environ.setdefault("ML_LGB_NUM_BOOST_ROUND", "200")
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
SHORTLIST = [
    ("2h", "sei-network"), ("2h", "bonk"), ("2h", "pepe"),
    ("2h", "floki-inu"), ("2h", "dogwifcoin"),
    ("6h", "sei-network"), ("6h", "celestia"),
    ("1d", "dogwifcoin"), ("1d", "worldcoin-wld"), ("1d", "sei-network"),
]
MIN_DA = 0.50
MIN_HOLDOUT_ROWS = 200


def _label_three_class(values: np.ndarray, threshold_pct: float) -> np.ndarray:
    out = np.full(values.shape[0], 1, dtype=np.int64)
    out[values >= threshold_pct] = 2
    out[values <= -threshold_pct] = 0
    return out


def _evaluate_slice(df: pd.DataFrame, vocab: list[str]) -> dict:
    if df is None or df.empty or len(df) < 80:
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


def main() -> int:
    progress_path = ROOT / "reports" / "20260423T204419Z-task379-offline-focused.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_fp = progress_path.open("w", buffering=1)
    _orig = sys.stdout

    class _Tee:
        def write(self, s: str) -> int:
            try:
                progress_fp.write(s)
            except Exception:
                pass
            return _orig.write(s)
        def flush(self) -> None:
            progress_fp.flush()
            _orig.flush()

    sys.stdout = _Tee()
    summary: dict = {"per_slice": []}
    cache: dict = {}

    print(f"FOCUSED VERIFICATION — optuna ON, n_trials=6, timeout=30s/fold, "
          f"boost_rounds=200, slices={len(SHORTLIST)}", flush=True)
    t0_global = time.time()

    for tf, coin in SHORTLIST:
        path = DATASETS.get(tf)
        if path is None or not path.exists():
            print(f"  [{tf}/{coin}] missing dataset, skipping", flush=True)
            continue
        if tf not in cache:
            cache[tf] = pd.read_parquet(path)
        full = cache[tf]
        coins = sorted(full["coin_id"].dropna().astype(str).unique().tolist())
        sub = full[full["coin_id"] == coin].copy()
        if len(sub) < 80:
            print(f"  [{tf}/{coin}] insufficient_rows={len(sub)}", flush=True)
            continue
        new_thr = labels_mod.MULTI_BAR_LABEL_THRESHOLDS_PERCENT.get(tf)
        horizon_new = labels_mod.DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF.get(tf, 1)
        new_df = sub.copy()
        if (horizon_new > 1 and new_thr is not None
                and "forward_window_return_pct" in new_df.columns):
            fwret = new_df["forward_window_return_pct"].astype(float).to_numpy()
            new_df["label_3class"] = _label_three_class(fwret, float(new_thr))
        old_df = sub.copy()
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
            "old_n_test_total": int(sum(f.get("n_test", 0) for f in old_res.get("folds", []))),
            "new_n_test_total": int(sum(f.get("n_test", 0) for f in new_res.get("folds", []))),
            "n_rows": int(len(sub)),
        })

    elapsed = time.time() - t0_global
    old_pass = sum(1 for s in summary["per_slice"] if s["old_pass"])
    new_pass = sum(1 for s in summary["per_slice"] if s["new_pass"])
    summary["totals"] = {
        "old_promoted": old_pass, "new_promoted": new_pass,
        "n_slices_evaluated": len(summary["per_slice"]),
        "n_slices_shortlist": len(SHORTLIST),
        "elapsed_seconds": round(elapsed, 1),
        "lgb_num_boost_round": int(os.environ["ML_LGB_NUM_BOOST_ROUND"]),
        "optuna_n_trials": int(os.environ["ML_OPTUNA_N_TRIALS"]),
        "optuna_timeout_seconds": int(os.environ["ML_OPTUNA_TIMEOUT_SECONDS"]),
        "skip_optuna": False,
    }
    out = ROOT / "reports" / "20260423T204419Z-task379-offline-focused.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print("\n========== TOTALS ==========")
    print(json.dumps(summary["totals"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
