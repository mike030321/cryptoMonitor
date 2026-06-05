"""Task #482 — offline four-stage stable-class collapse diagnostic.

For each (coin, timeframe) slice in the latest pooled cached dataset, we
re-derive a small LightGBM booster (matching the trainer's
`_balanced_sample_weight` + per-class isotonic calibration recipe) and
record the STABLE-argmax share at four stages:

    1. label_stable_share   — fraction of holdout rows whose true label is STABLE
    2. raw_stable_share     — STABLE-argmax share of the booster's RAW predictions
    3. cal_stable_share     — STABLE-argmax share AFTER per-class isotonic
    4. cal_stable_prob_mean — mean calibrated P(STABLE) (sanity check)

The four-stage table tells us which lever is binding the
`directional_call_share` at the current 0.97-1.000 ceiling:

    * label_collapse      : label_stable_share <= 0.05 across the fleet
    * booster_collapse    : labels carry STABLE mass but raw_stable_share also
                             ~0; the booster never wants to predict STABLE
    * calibrator_collapse : raw STABLE survives but the per-class isotonic
                             flattens it post-cal
    * coupled_collapse    : both raw and calibrated STABLE share are very low,
                             but for distinct mechanical reasons

Output: a JSON dump under `reports/<TS>-task482-stage-collapse-diagnostic.json`
plus a markdown summary at `reports/<TS>-task482-stable-class-diagnostic.md`.

Cache datasets are taken from the most recent pooled snapshot per
timeframe (1h/2h/6h/1d). 5m / 1m skipped per `ML_CAMPAIGN_SKIP_5M=1`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Force the trainer's CI shortcuts so this script stays inside a normal
# shell timeout. ML_SKIP_OPTUNA=1 makes `_optuna_search_lgb` return a
# single sane LGBM config; we further cap num_boost_round for speed.
os.environ.setdefault("ML_SKIP_OPTUNA", "1")
os.environ.setdefault("ML_LGB_NUM_BOOST_ROUND", "80")

# Re-import after env is set so the trainer constants pick them up.
from app.training import train as _train_mod  # noqa: E402
from app.training.train import (  # noqa: E402
    CALIBRATION_HOLDOUT_FRACTION,
    NUM_CLASSES,
    _balanced_sample_weight,
    _calibrate_per_class,
    _encode_coin_idx,
    _lgb_params,
    _train_lgb,
)
from app.training.registry import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    POOLED_COIN_ID,
)


TRADEABLE_TFS = ["1h", "2h", "6h", "1d"]


def _latest_pooled_dataset(
    timeframe: str, *, max_age_hours: Optional[float] = None,
) -> Path:
    """Pick the *freshest* (newest mtime) dataset file for the timeframe.

    Task #540 — switched from a size-based picker to an mtime-based one
    once the auto-refresher (`scripts/scheduled_refresh_loop.py`) keeps
    every snapshot fully populated. Sorting by size used to be a proxy
    for coverage when older snapshots were sometimes missing coins; now
    that the refresher rewrites every (coin, tf) on cadence, mtime is
    the right "freshest" signal.

    Pass ``max_age_hours`` (or set ``ML_DATASET_MAX_AGE_HOURS``) to
    refuse stale snapshots — the retrain harness uses this to bail loud
    rather than silently consume a week-old cache when the refresher
    has been failing in the background. ``ML_DATASET_MAX_AGE_HOURS``
    overrides the per-call argument so an operator can force-allow
    stale data for a one-off rerun.
    """
    candidates = sorted(
        (ROOT / "models" / "datasets").glob(f"{timeframe}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no cached dataset for timeframe={timeframe}")
    chosen = candidates[0]
    env_override = os.environ.get("ML_DATASET_MAX_AGE_HOURS")
    if env_override is not None:
        try:
            max_age_hours = float(env_override)
        except ValueError:
            pass
    if max_age_hours is not None and max_age_hours > 0:
        age_hours = (time.time() - chosen.stat().st_mtime) / 3600.0
        if age_hours > max_age_hours:
            raise RuntimeError(
                f"latest cached dataset for timeframe={timeframe} is "
                f"{age_hours:.1f}h old (> max_age_hours={max_age_hours:.1f}); "
                f"newest file={chosen.name}. Run "
                f"`scripts.refresh_cached_datasets` (or ensure the "
                f"`dataset-refresher` workflow is running) and retry. "
                f"Set ML_DATASET_MAX_AGE_HOURS=0 to disable this check "
                f"for a one-off rerun."
            )
    return chosen


def _slice_for(df: pd.DataFrame, coin_id: str) -> pd.DataFrame:
    if coin_id == POOLED_COIN_ID:
        return df.copy()
    return df[df["coin_id"] == coin_id].copy()


def _diagnose_one(
    df_full: pd.DataFrame, coin_id: str, timeframe: str, vocab: list[str],
) -> dict:
    """Train a booster + calibrate the same way the trainer does and
    record per-stage STABLE-argmax shares. Returns one row of the
    diagnostic table.
    """
    df = _slice_for(df_full, coin_id)
    if df.empty or len(df) < 80:
        return {
            "coin_id": coin_id, "timeframe": timeframe,
            "status": "insufficient_data",
            "n_rows": int(len(df)),
        }
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df = _encode_coin_idx(df, vocab)

    X_all = df[FEATURE_COLUMNS].copy()
    y_all = df["label_3class"].to_numpy().astype(int)

    cal_start = max(1, int(len(df) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
    if cal_start >= len(df) - 5:
        return {
            "coin_id": coin_id, "timeframe": timeframe,
            "status": "tail_too_small",
            "n_rows": int(len(df)),
        }
    X_train, y_train = X_all.iloc[:cal_start], y_all[:cal_start]
    X_cal, y_cal = X_all.iloc[cal_start:], y_all[cal_start:]

    # Use the same LGBM config the ML_SKIP_OPTUNA shortcut would pick.
    params = _lgb_params(31, 0.1, 5)
    booster, _ = _train_lgb(X_train, y_train, X_cal, y_cal, params)

    raw_pred = booster.predict(X_cal, num_iteration=booster.best_iteration)
    if raw_pred.ndim == 1:
        raw_pred = np.tile([0, 1, 0], (len(X_cal), 1)).astype(float)

    calibrators, cal_pred, _diagram = _calibrate_per_class(raw_pred, y_cal)
    if calibrators is None:
        cal_pred = raw_pred  # no calibration fit (single-class everywhere)

    raw_argmax = raw_pred.argmax(axis=1)
    cal_argmax = cal_pred.argmax(axis=1)

    train_class = {
        f"train_{name}_share": float((y_train == k).mean())
        for k, name in enumerate(["DOWN", "STABLE", "UP"])
    }
    holdout_class = {
        f"label_{name}_share": float((y_cal == k).mean())
        for k, name in enumerate(["DOWN", "STABLE", "UP"])
    }
    raw_pred_class = {
        f"raw_{name}_share": float((raw_argmax == k).mean())
        for k, name in enumerate(["DOWN", "STABLE", "UP"])
    }
    cal_pred_class = {
        f"cal_{name}_share": float((cal_argmax == k).mean())
        for k, name in enumerate(["DOWN", "STABLE", "UP"])
    }

    raw_stable_prob_mean = float(raw_pred[:, 1].mean())
    cal_stable_prob_mean = float(cal_pred[:, 1].mean())
    raw_stable_prob_max = float(raw_pred[:, 1].max())
    cal_stable_prob_max = float(cal_pred[:, 1].max())

    # Compare STABLE column to the maxes of the other two columns —
    # a row gets argmax==STABLE iff p_stable strictly exceeds both.
    nonstable_max = np.maximum(cal_pred[:, 0], cal_pred[:, 2])
    cal_stable_margin_mean = float((cal_pred[:, 1] - nonstable_max).mean())

    weights = _balanced_sample_weight(y_train)
    weight_summary = {
        f"sample_weight_{name}": (
            float(weights[y_train == k].mean())
            if (y_train == k).any() else None
        )
        for k, name in enumerate(["DOWN", "STABLE", "UP"])
    }

    return {
        "coin_id": coin_id,
        "timeframe": timeframe,
        "status": "ok",
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_holdout": int(len(X_cal)),
        **train_class,
        **holdout_class,
        **raw_pred_class,
        **cal_pred_class,
        "raw_stable_prob_mean": raw_stable_prob_mean,
        "raw_stable_prob_max": raw_stable_prob_max,
        "cal_stable_prob_mean": cal_stable_prob_mean,
        "cal_stable_prob_max": cal_stable_prob_max,
        "cal_stable_margin_mean": cal_stable_margin_mean,
        "directional_call_share": float((cal_argmax != 1).mean()),
        **weight_summary,
        "calibrators_fit": calibrators is not None,
    }


def _bucket_for(row: dict) -> str:
    """Classify the binding lever for one slice given the four-stage row."""
    if row.get("status") != "ok":
        return row.get("status", "unknown")
    label_st = row["label_STABLE_share"]
    raw_st = row["raw_STABLE_share"]
    cal_st = row["cal_STABLE_share"]
    raw_pm = row["raw_stable_prob_mean"]
    if label_st <= 0.05:
        return "label_collapse"
    # Booster collapse: raw argmax STABLE share is essentially zero and
    # the average raw P(stable) is well below the marginal.
    booster_collapsed = raw_st <= 0.05 and raw_pm < 0.5 * label_st
    # Calibrator collapse: raw stage carries STABLE mass but the
    # calibrated stage drops it.
    calibrator_collapsed = (raw_st > 0.05) and (cal_st <= 0.05)
    if booster_collapsed and calibrator_collapsed:
        return "coupled_collapse"
    if booster_collapsed:
        return "booster_collapse"
    if calibrator_collapsed:
        return "calibrator_collapse"
    return "no_collapse"


def main() -> int:
    coins = list(_train_mod.DEFAULT_COINS) + [POOLED_COIN_ID]
    rows: list[dict] = []
    t0 = time.monotonic()
    # Allow restricting to a subset of timeframes via TFS env so the
    # full sweep can be split across multiple shell runs.
    selected_tfs = os.environ.get("TFS")
    tfs = (
        [t.strip() for t in selected_tfs.split(",") if t.strip()]
        if selected_tfs else list(TRADEABLE_TFS)
    )
    for tf in tfs:
        try:
            ds_path = _latest_pooled_dataset(tf)
        except FileNotFoundError as exc:
            rows.append({"timeframe": tf, "status": "no_dataset", "error": str(exc)})
            continue
        df = pd.read_parquet(ds_path)
        vocab = sorted(df["coin_id"].unique().tolist())
        for coin in coins:
            t = time.monotonic()
            try:
                row = _diagnose_one(df, coin, tf, vocab)
            except Exception as exc:  # noqa: BLE001 — one failed slice mustn't stop the diagnostic
                row = {
                    "coin_id": coin, "timeframe": tf,
                    "status": "error", "error": str(exc),
                }
            row["bucket"] = _bucket_for(row)
            row["elapsed_s"] = round(time.monotonic() - t, 2)
            row["dataset_path"] = str(ds_path.relative_to(ROOT))
            rows.append(row)
            print(
                f"  [{tf}] {coin:30} bucket={row['bucket']:>22} "
                f"elapsed={row['elapsed_s']}s",
                flush=True,
            )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"{ts}-task482-stage-collapse-diagnostic.json"
    json_path.write_text(json.dumps({
        "task": 482,
        "generated_at": ts,
        "coins": coins,
        "tradeable_tfs": TRADEABLE_TFS,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {json_path}")
    print(f"total elapsed: {round(time.monotonic() - t0, 1)}s")
    print(f"slices diagnosed: {sum(1 for r in rows if r.get('status') == 'ok')}")

    # Bucket counts
    from collections import Counter
    bc = Counter(r.get("bucket", "unknown") for r in rows)
    print("\nbucket counts:")
    for b, n in sorted(bc.items(), key=lambda kv: -kv[1]):
        print(f"  {b:>22}  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
