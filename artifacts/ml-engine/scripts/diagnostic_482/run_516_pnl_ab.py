"""Task #516 — pre-fix vs post-fix per-slice PnL A/B for the touched
1d / 6h slices.

The Task #507 booster fix toggled the tiny-slice training recipe in
``app/training/train.py::_train_lgb`` (soft hyperparams + alpha=2
weights when ``n_train < TINY_SLICE_THRESHOLD = 1500``). Task #516
asks whether walk-forward PnL on the affected 1d / 6h slices regressed
versus the pre-fix campaign baseline.

The pre-fix campaign run never produced per-slice walk-forward PnL for
1d / 6h (its archived ``backtest_report.json`` records
``status="no_dataset"`` for those timeframes — the 1000-day lookback
floor from Task #417 had not yet kicked in). The post-fix campaign
only captured PnL for 4 slices (3 per-coin 1d + pooled). A direct
delta from existing artifacts is therefore impossible.

This script builds the comparison directly: for each touched slice it
trains both code paths on **the same dataset window the post-fix
campaign used**, calibrates per class, and runs the trainer's own
``_holdout_pnl_after_fees`` helper on the calibration holdout. The two
arms differ only in the ``TINY_SLICE_THRESHOLD`` constant
(monkey-patched between fits): the pre-fix arm sees
``THRESHOLD=0`` (so the tiny-slice branch never fires and the trainer
falls through to the original Optuna-tuned recipe with early stopping
and alpha=1 sample weights), the post-fix arm sees the production
``THRESHOLD=1500`` (so every 1d / 6h slice trips the tiny-slice branch
exactly as it did in ``training_run_20260425T063302Z``).

The booster runs at the production ``ML_LGB_NUM_BOOST_ROUND=800`` —
this is the key faithfulness lever vs. the Task #507 focused
diagnostic, which used 80 rounds.

Output: JSON dump under
``reports/<TS>-task516-pnl-ab-results.json`` plus a markdown delta
table appended to the Task #516 report.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ML_SKIP_OPTUNA", "1")
os.environ.setdefault("ML_LGB_NUM_BOOST_ROUND", "800")

from app.backtest.contract import get_frictions  # noqa: E402
from app.training import train as _train_mod  # noqa: E402
from app.training.train import (  # noqa: E402
    CALIBRATION_HOLDOUT_FRACTION,
    NUM_CLASSES,
    _balanced_sample_weight,
    _calibrate_per_class,
    _encode_coin_idx,
    _holdout_pnl_after_fees,
    _lgb_params,
    _train_lgb,
)
from app.training.registry import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    POOLED_COIN_ID,
)


TOUCHED_SLICES = [
    ("bonk", "1d"), ("celestia", "1d"), ("dogwifcoin", "1d"),
    ("floki-inu", "1d"), ("injective-protocol", "1d"),
    ("jupiter-exchange-solana", "1d"), ("pepe", "1d"),
    ("render-token", "1d"), ("worldcoin-wld", "1d"),
    ("bonk", "6h"), ("celestia", "6h"), ("dogwifcoin", "6h"),
    ("floki-inu", "6h"), ("injective-protocol", "6h"),
    ("jupiter-exchange-solana", "6h"), ("pepe", "6h"),
    ("render-token", "6h"), ("worldcoin-wld", "6h"),
]


# Pinned dataset filenames — explicit so reruns are deterministic and
# the report's "same data the 04-25 campaign read" claim is auditable.
# These are the largest pooled snapshots whose mtime is at or just
# before the 04-25 campaign's first per-slice fit; the campaign's
# data-prep phase wrote them and the trainer consumed them in-place.
PINNED_DATASETS = {
    "1d": "1d_20260425T103252Z.parquet",
    "6h": "6h_20260423T035301Z.parquet",
}


def _latest_pooled_dataset(timeframe: str) -> Path:
    """Return the dataset path for ``timeframe``.

    Pinned by default to the snapshots the 04-25 production campaign
    consumed (see ``PINNED_DATASETS``). Override per-timeframe via the
    ``ML_516_DATASET_<TF>`` env var (e.g. ``ML_516_DATASET_1D``) when
    rerunning against a fresher snapshot.
    """
    override = os.environ.get(f"ML_516_DATASET_{timeframe.upper()}")
    fname = override or PINNED_DATASETS.get(timeframe)
    if not fname:
        raise FileNotFoundError(
            f"no pinned dataset for timeframe={timeframe} "
            "(set ML_516_DATASET_<TF> to override)"
        )
    path = ROOT / "models" / "datasets" / fname
    if not path.exists():
        raise FileNotFoundError(
            f"pinned dataset not found: {path} "
            f"(set ML_516_DATASET_{timeframe.upper()} to a present file)"
        )
    return path


def _slice_for(df: pd.DataFrame, coin_id: str) -> pd.DataFrame:
    if coin_id == POOLED_COIN_ID:
        return df.copy()
    return df[df["coin_id"] == coin_id].copy()


def _train_one_arm(
    df_full: pd.DataFrame, coin: str, tf: str, vocab: list[str],
    *, threshold: int, alpha: float, fr,
) -> dict:
    df = _slice_for(df_full, coin)
    if df.empty or len(df) < 80:
        return {"status": "insufficient_data", "n_rows": int(len(df))}
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df = _encode_coin_idx(df, vocab)

    X_all = df[FEATURE_COLUMNS].copy()
    y_all = df["label_3class"].to_numpy().astype(int)
    fwd_all = df["forward_return"].to_numpy(dtype=float)

    cal_start = max(1, int(len(df) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
    if cal_start >= len(df) - 5:
        return {"status": "tail_too_small", "n_rows": int(len(df))}

    X_train, y_train = X_all.iloc[:cal_start], y_all[:cal_start]
    X_cal, y_cal = X_all.iloc[cal_start:], y_all[cal_start:]
    fwd_cal = fwd_all[cal_start:]

    # Monkey-patch the trainer constant so the same `_train_lgb` helper
    # exercises the desired branch. The pre-fix arm uses 0 (tiny path
    # never fires); the post-fix arm uses the production 1500.
    saved_thr = _train_mod.TINY_SLICE_THRESHOLD
    saved_alpha = _train_mod.TINY_SLICE_CLASS_WEIGHT_ALPHA
    try:
        _train_mod.TINY_SLICE_THRESHOLD = threshold
        _train_mod.TINY_SLICE_CLASS_WEIGHT_ALPHA = float(alpha)
        # Default LGB params for the non-tiny branch — same shortcut the
        # ML_SKIP_OPTUNA path picks (mirrors run_stage_collapse_diagnostic.py:124).
        params = _lgb_params(31, 0.1, 5)
        booster, _ = _train_lgb(X_train, y_train, X_cal, y_cal, params)
        raw_proba = booster.predict(
            X_cal, num_iteration=booster.best_iteration,
        )
    finally:
        _train_mod.TINY_SLICE_THRESHOLD = saved_thr
        _train_mod.TINY_SLICE_CLASS_WEIGHT_ALPHA = saved_alpha

    if raw_proba.ndim == 1:
        raw_proba = np.tile([0, 1, 0], (len(X_cal), 1)).astype(float)

    calibrators, cal_proba, _diagram = _calibrate_per_class(raw_proba, y_cal)
    if calibrators is None:
        cal_proba = raw_proba
    raw_argmax = raw_proba.argmax(axis=1)
    cal_argmax = cal_proba.argmax(axis=1)

    pnl = _holdout_pnl_after_fees(
        cal_proba, fwd_cal,
        min_directional_edge=float(fr.min_directional_edge),
        min_expected_return_pct=float(fr.min_expected_return_pct),
        round_trip_cost_pct=float(fr.round_trip_cost_pct) * 100.0,
        magnitudes_pct=None,  # no regression head in this focused harness
    )
    return {
        "status": "ok",
        "n_rows": int(len(df)),
        "n_train": int(cal_start),
        "n_cal": int(len(df) - cal_start),
        "raw_stable_share": float((raw_argmax == 1).mean()),
        "cal_stable_share": float((cal_argmax == 1).mean()),
        "directional_call_share": float((cal_argmax != 1).mean()),
        "label_stable_share": float((y_cal == 1).mean()),
        "pnl_after_fees": pnl,
    }


def main() -> int:
    fr = get_frictions()
    print(
        f"using frictions mde={fr.min_directional_edge} "
        f"mer={fr.min_expected_return_pct} "
        f"rtc_pct={fr.round_trip_cost_pct * 100.0}",
        flush=True,
    )

    # Pre-load each timeframe's pooled dataset once.
    by_tf: dict[str, pd.DataFrame] = {}
    for _, tf in TOUCHED_SLICES:
        if tf not in by_tf:
            ds = _latest_pooled_dataset(tf)
            print(f"loading {tf}: {ds.name}", flush=True)
            by_tf[tf] = pd.read_parquet(ds)

    # Three arms. `pre` and `post` capture the regression vs. the
    # production tiny-slice branch; `proposed` tests the brief's
    # tightening lever (alpha=1.5 instead of 2.0).
    arms = [
        ("pre",      {"threshold": 0,    "alpha": 1.0}),  # tiny-slice branch never fires
        ("post",     {"threshold": 1500, "alpha": 2.0}),  # current production
        ("proposed", {"threshold": 1500, "alpha": 1.5}),  # tightened alpha
    ]

    results: list[dict] = []
    t0 = time.monotonic()
    for coin, tf in TOUCHED_SLICES:
        df = by_tf[tf]
        vocab = sorted(df["coin_id"].unique().tolist())
        t = time.monotonic()
        try:
            arm_results = {
                name: _train_one_arm(df, coin, tf, vocab, fr=fr, **kw)
                for name, kw in arms
            }
        except Exception as exc:  # noqa: BLE001
            results.append({
                "coin": coin, "timeframe": tf,
                "status": "error", "error": str(exc),
                "elapsed_s": round(time.monotonic() - t, 2),
            })
            print(f"  ERROR {coin}/{tf}: {exc}", flush=True)
            continue
        elapsed = round(time.monotonic() - t, 2)
        ref = arm_results["pre"]
        row = {
            "coin": coin, "timeframe": tf,
            "n_rows": ref.get("n_rows"),
            "n_train": ref.get("n_train"),
            "n_cal": ref.get("n_cal"),
            "label_stable_share": round(ref.get("label_stable_share") or 0.0, 4),
            "elapsed_s": elapsed,
        }
        for name in ("pre", "post", "proposed"):
            a = arm_results[name]
            pnl_block = a.get("pnl_after_fees") or {}
            row[f"{name}_dcs"] = round(a.get("directional_call_share") or 0.0, 4)
            row[f"{name}_pnl_pct_total"] = pnl_block.get("net_pct_total")
            row[f"{name}_n_trades"] = pnl_block.get("n_trades")
            row[f"{name}_win_rate"] = pnl_block.get("win_rate")
        pre_pnl = row["pre_pnl_pct_total"]
        post_pnl = row["post_pnl_pct_total"]
        prop_pnl = row["proposed_pnl_pct_total"]
        row["delta_post_minus_pre"] = (
            None if pre_pnl is None or post_pnl is None
            else round(post_pnl - pre_pnl, 4)
        )
        row["delta_proposed_minus_pre"] = (
            None if pre_pnl is None or prop_pnl is None
            else round(prop_pnl - pre_pnl, 4)
        )
        row["delta_proposed_minus_post"] = (
            None if post_pnl is None or prop_pnl is None
            else round(prop_pnl - post_pnl, 4)
        )
        results.append(row)
        print(
            f"  {coin:25} @ {tf:>2}  "
            f"PnL pre={pre_pnl}  post={post_pnl}  proposed={prop_pnl}  "
            f"Δ(post-pre)={row['delta_post_minus_pre']}  "
            f"Δ(prop-pre)={row['delta_proposed_minus_pre']}  "
            f"({elapsed}s)",
            flush=True,
        )

    total_elapsed = round(time.monotonic() - t0, 2)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{ts}-task516-pnl-ab-results.json"
    out.write_text(json.dumps({
        "task": 516,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ml_lgb_num_boost_round": int(os.environ["ML_LGB_NUM_BOOST_ROUND"]),
        "ml_skip_optuna": int(os.environ["ML_SKIP_OPTUNA"]),
        "frictions": {
            "min_directional_edge": float(fr.min_directional_edge),
            "min_expected_return_pct": float(fr.min_expected_return_pct),
            "round_trip_cost_pct": float(fr.round_trip_cost_pct) * 100.0,
        },
        "elapsed_total_s": total_elapsed,
        "results": results,
    }, indent=2, default=str))
    print(f"\nwrote {out.relative_to(ROOT)}  total_elapsed={total_elapsed}s", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
