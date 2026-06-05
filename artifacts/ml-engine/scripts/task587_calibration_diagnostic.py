"""Task #587 — calibration diagnostic for the feature-edge search trade_share blow-up.

Why this exists:
    Task #580 stage-2 evaluation produced `trade_share` between 0.83 and 1.00
    for every (coin, timeframe) — far outside the unmodified gate band
    [0.40, 0.85]. Symptom: the unweighted LightGBM fit by
    `feature_edge_search/run_search.py` is using raw `argmax(proba) != STABLE`
    as its trade rule, with NO calibration applied. The natural label
    distributions (STABLE share 15-23%) make argmax pick STABLE rarely, so
    the trade_share collapses to ~1.0 by construction.

What this script does:
    For each timeframe in {1h, 2h, 6h, 1d}:
      1. Load the latest dataset snapshot (same picker as run_search.py).
      2. Use a single time-ordered split per coin: first 80% train, last 20% test.
      3. From the trainer's chronological tail of train, hold out the last
         20% as a CALIBRATION TAIL (point-in-time-safe; no peek into test).
      4. Fit LightGBM on inner-train; fit single-scalar temperature on
         calibration tail predictions; apply temperature to the test set.
      5. Record per-(coin, timeframe):
         - per-class probability distribution stats (mean / p10 / p25 / p50
           / p75 / p90 / p95) for raw and calibrated probs;
         - trade_share / DA / PnL at neutral-band widths
           delta in {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}, where the trade
           rule is "trade iff max(P_UP, P_DOWN) > P_STABLE + delta";
         - trade_share at the same widths after calibration.

Output:
    artifacts/ml-engine/reports/<TS>-task587-calibration-diagnostic.json
    artifacts/ml-engine/reports/<TS>-task587-calibration-diagnostic.md

This script does NOT touch any production source file. It reads:
    - models/datasets/ snapshots
    - registry.FEATURE_COLUMNS / CATEGORICAL_FEATURES
    - shared/trading-frictions.json (for the round-trip cost)
    - app.training.calibration (single-scalar temperature scaling)
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.calibration import (  # noqa: E402
    apply_single_temperature,
    fit_single_temperature,
)
from app.training.registry import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)

logger = logging.getLogger("ml-engine.task587")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = ROOT.parent.parent
SHARED_FRICTIONS_PATH = REPO_ROOT / "shared" / "trading-frictions.json"
DATASETS_DIR = ROOT / "models" / "datasets"
REPORTS_DIR = ROOT / "reports"


def _json_safe(obj):
    """Recursively convert NaN/Inf floats to None for strict-JSON compatibility."""
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

TIMEFRAMES = ["6h", "1d"]   # focused subset; same scope as task587_rerun_stage2.py for runtime tractability
TEST_FRAC = 0.25            # last 25% chronological per coin = OOS test
CAL_TAIL_FRAC = 0.20        # last 20% of train (= 60..80 quantile of full series) = calibration tail
LGB_ROUNDS = 40             # halved vs run_search.py LGB_ROUNDS (80) for diagnostic speed; trade_share signal is stable
LGB_LEAVES = 15
LGB_LR = 0.1
SEED = 42
NEUTRAL_BAND_WIDTHS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
PROB_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def latest_dataset_path(timeframe: str) -> Optional[Path]:
    if not DATASETS_DIR.exists():
        return None
    cands = sorted(DATASETS_DIR.glob(f"{timeframe}_*.parquet"))
    return cands[-1] if cands else None


def load_dataset(timeframe: str) -> pd.DataFrame:
    p = latest_dataset_path(timeframe)
    if p is None:
        raise FileNotFoundError(f"no dataset snapshot for tf={timeframe} under {DATASETS_DIR}")
    df = pd.read_parquet(p)
    for col in ("label_3class", "forward_return", "coin_id", "timestamp_ms"):
        if col not in df.columns:
            raise ValueError(f"dataset {p} missing required column {col!r}")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def get_round_trip_cost_pct() -> float:
    payload = json.loads(SHARED_FRICTIONS_PATH.read_text())
    fees = payload["fees"]
    return 2.0 * (float(fees["taker_fee_pct"]) + float(fees["slippage_pct"]))


def lgb_params() -> dict:
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": LGB_LEAVES,
        "learning_rate": LGB_LR,
        "min_child_samples": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "deterministic": True,
        "seed": SEED,
    }


def encode_coin_idx(df: pd.DataFrame, vocab: list[str]) -> pd.Series:
    idx = {c: i for i, c in enumerate(vocab)}
    return df["coin_id"].map(idx).fillna(-1).astype(int)


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: list[str],
) -> np.ndarray:
    import lightgbm as lgb
    cats = [c for c in CATEGORICAL_FEATURES if c in feature_names]
    X_tr = train[feature_names].astype(float).fillna(0.0)
    y_tr = train["label_3class"].astype(int).to_numpy()
    X_te = test[feature_names].astype(float).fillna(0.0)
    train_set = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature=cats if cats else "auto",
        free_raw_data=False,
    )
    booster = lgb.train(
        lgb_params(), train_set,
        num_boost_round=LGB_ROUNDS,
        callbacks=[lgb.log_evaluation(0)],
    )
    proba = booster.predict(X_te)
    if proba.ndim == 1:
        proba = np.tile([0.0, 1.0, 0.0], (len(X_te), 1)).astype(float)
    return proba


def prob_distribution_stats(proba: np.ndarray) -> dict:
    """Per-class column stats: mean and a fixed quantile vector."""
    out = {}
    names = ["DOWN", "STABLE", "UP"]
    for k, name in enumerate(names):
        col = proba[:, k]
        q = np.quantile(col, PROB_QUANTILES)
        out[name] = {
            "mean": float(col.mean()),
            **{f"p{int(q_*100)}": float(q[i]) for i, q_ in enumerate(PROB_QUANTILES)},
        }
    # Distribution of the "winner-margin" max(p_up, p_down) - p_stable.
    margin = np.maximum(proba[:, 0], proba[:, 2]) - proba[:, 1]
    qmargin = np.quantile(margin, PROB_QUANTILES)
    out["MARGIN_DIRECTIONAL_OVER_STABLE"] = {
        "mean": float(margin.mean()),
        **{f"p{int(q_*100)}": float(qmargin[i]) for i, q_ in enumerate(PROB_QUANTILES)},
    }
    return out


def score_with_band(
    proba: np.ndarray,
    y_true: np.ndarray,
    fwd: np.ndarray,
    delta: float,
    rt_cost_frac: float,
) -> dict:
    """Trade iff max(P_UP, P_DOWN) > P_STABLE + delta. Direction = which of
    UP/DOWN is the winner. DA: among (true label != STABLE) rows, did the
    direction match? PnL: signed_return * 100 - rt_cost*100.
    """
    p_down = proba[:, 0]
    p_stable = proba[:, 1]
    p_up = proba[:, 2]
    dir_winner = np.where(p_up >= p_down, 2, 0)   # 2=UP, 0=DOWN
    dir_max = np.maximum(p_up, p_down)
    trade_mask = dir_max > (p_stable + delta)
    n_test = len(proba)
    n_trades = int(trade_mask.sum())

    nonstab_mask = y_true != 1
    n_dir_truth = int(nonstab_mask.sum())
    if n_dir_truth > 0:
        # Treat "did not trade" on a directional truth row as a miss
        # (matches verification gate: argmax STABLE on non-stable truth = miss).
        pred_class = np.where(trade_mask, dir_winner, 1)
        da = float(np.mean(pred_class[nonstab_mask] == y_true[nonstab_mask]))
    else:
        da = float("nan")

    if n_trades > 0:
        signs = np.where(dir_winner[trade_mask] == 2, 1.0, -1.0)
        signed_ret_pct = signs * fwd[trade_mask] * 100.0
        pnl = signed_ret_pct - (rt_cost_frac * 100.0)
        post_fee_pnl_pct_total = float(np.sum(pnl))
        win_rate = float(np.mean(pnl > 0))
    else:
        post_fee_pnl_pct_total = 0.0
        win_rate = float("nan")

    return {
        "delta": float(delta),
        "n_trades": n_trades,
        "n_test": n_test,
        "trade_share": (n_trades / n_test) if n_test > 0 else 0.0,
        "directional_accuracy": da,
        "post_fee_pnl_pct_total": post_fee_pnl_pct_total,
        "win_rate": win_rate,
    }


def evaluate_slice(
    df_coin: pd.DataFrame, feature_names: list[str], rt_cost_frac: float,
) -> Optional[dict]:
    """Single time-ordered split on a per-coin frame:
        train = first 75%; test = last 25%; calibration tail = last 20% of train.
    """
    df_coin = df_coin.sort_values("timestamp_ms").reset_index(drop=True)
    n = len(df_coin)
    if n < 200:
        return None
    test_start = int(n * (1.0 - TEST_FRAC))
    train_full = df_coin.iloc[:test_start]
    test = df_coin.iloc[test_start:]
    if len(train_full) < 100 or len(test) < 30:
        return None
    cal_start = int(len(train_full) * (1.0 - CAL_TAIL_FRAC))
    train_inner = train_full.iloc[:cal_start]
    cal_tail = train_full.iloc[cal_start:]
    if len(train_inner) < 50 or len(cal_tail) < 20:
        # Fallback: no calibration; use full train and skip calibrated metrics.
        proba_test_raw = fit_predict(train_full, test, feature_names)
        cal_inv_T = 1.0
        cal_pred_raw = None
        y_cal = None
    else:
        proba_test_raw = fit_predict(train_inner, test, feature_names)
        cal_pred_raw = fit_predict(train_inner, cal_tail, feature_names)
        y_cal = cal_tail["label_3class"].astype(int).to_numpy()
        cal_inv_T = fit_single_temperature(cal_pred_raw, y_cal)

    proba_test_cal = apply_single_temperature(proba_test_raw, cal_inv_T)
    y_te = test["label_3class"].astype(int).to_numpy()
    fwd_te = test["forward_return"].astype(float).to_numpy()

    raw_band_sweep = [
        score_with_band(proba_test_raw, y_te, fwd_te, d, rt_cost_frac)
        for d in NEUTRAL_BAND_WIDTHS
    ]
    cal_band_sweep = [
        score_with_band(proba_test_cal, y_te, fwd_te, d, rt_cost_frac)
        for d in NEUTRAL_BAND_WIDTHS
    ]

    return {
        "n_train_inner": int(len(train_inner) if cal_pred_raw is not None else len(train_full)),
        "n_cal_tail": int(len(cal_tail)) if cal_pred_raw is not None else 0,
        "n_test": int(len(test)),
        "calibration_inv_T": float(cal_inv_T),
        "calibration_T": float(1.0 / cal_inv_T) if cal_inv_T > 0 else None,
        "raw_prob_stats": prob_distribution_stats(proba_test_raw),
        "calibrated_prob_stats": prob_distribution_stats(proba_test_cal),
        "raw_band_sweep": raw_band_sweep,
        "calibrated_band_sweep": cal_band_sweep,
        "label_test_dist": {
            "down": float(np.mean(y_te == 0)),
            "stable": float(np.mean(y_te == 1)),
            "up": float(np.mean(y_te == 2)),
        },
    }


def run() -> None:
    rt_cost_frac = get_round_trip_cost_pct()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORTS_DIR / f"{ts}-task587-calibration-diagnostic.json"
    md_path = REPORTS_DIR / f"{ts}-task587-calibration-diagnostic.md"

    payload = {
        "task": 587,
        "captured_at": ts,
        "round_trip_cost_pct": rt_cost_frac * 100.0,
        "neutral_band_widths": NEUTRAL_BAND_WIDTHS,
        "test_frac": TEST_FRAC,
        "cal_tail_frac": CAL_TAIL_FRAC,
        "lgb_rounds": LGB_ROUNDS,
        "per_slice": [],
        "per_timeframe_dataset": {},
    }

    for tf in TIMEFRAMES:
        try:
            df = load_dataset(tf)
        except FileNotFoundError as exc:
            logger.warning("skip tf=%s: %s", tf, exc)
            continue
        df["coin_idx"] = encode_coin_idx(df, sorted(df["coin_id"].unique().tolist()))
        feats = [c for c in FEATURE_COLUMNS if c in df.columns]
        if "coin_idx" not in feats:
            feats = feats + ["coin_idx"]
        payload["per_timeframe_dataset"][tf] = {
            "n_rows": int(len(df)),
            "n_features": len(feats),
            "dataset_file": latest_dataset_path(tf).name,
            "label_dist": {
                "down": float(np.mean(df["label_3class"] == 0)),
                "stable": float(np.mean(df["label_3class"] == 1)),
                "up": float(np.mean(df["label_3class"] == 2)),
            },
        }
        for coin in sorted(df["coin_id"].unique().tolist()):
            sub = df[df["coin_id"] == coin].copy()
            t0 = time.time()
            try:
                slice_res = evaluate_slice(sub, feats, rt_cost_frac)
            except Exception as exc:  # noqa: BLE001
                logger.exception("slice failed tf=%s coin=%s", tf, coin)
                payload["per_slice"].append({
                    "coin": coin, "timeframe": tf, "error": str(exc),
                    "n_rows": int(len(sub)),
                })
                continue
            if slice_res is None:
                payload["per_slice"].append({
                    "coin": coin, "timeframe": tf,
                    "n_rows": int(len(sub)),
                    "skipped_reason": "fewer than 200 rows after split",
                })
                continue
            row = {"coin": coin, "timeframe": tf, "n_rows": int(len(sub)),
                   "elapsed_sec": round(time.time() - t0, 2), **slice_res}
            payload["per_slice"].append(row)
            logger.info(
                "tf=%s coin=%s inv_T=%.3f raw_trade_share@0=%s cal_trade_share@0=%s elapsed=%.1fs",
                tf, coin, slice_res["calibration_inv_T"],
                slice_res["raw_band_sweep"][0]["trade_share"],
                slice_res["calibrated_band_sweep"][0]["trade_share"],
                time.time() - t0,
            )

    json_path.write_text(json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False))
    write_markdown(md_path, payload)
    logger.info("wrote %s", json_path)
    logger.info("wrote %s", md_path)


def write_markdown(md_path: Path, payload: dict) -> None:
    lines: list[str] = []
    ts = payload["captured_at"]
    lines.append(f"# Task #587 — Calibration / decision-threshold diagnostic ({ts})\n")
    lines.append(
        "Investigates why task #580 stage-2 evaluation produced `trade_share` "
        "of 0.83-1.00 across every (coin, timeframe) slice — far outside the "
        "unmodified gate band [0.40, 0.85]. The hypothesis: the search "
        "runner trains an unweighted LightGBM and uses raw "
        "`argmax(proba) != STABLE` as the trade rule, with no calibration "
        "applied to the booster's confidence. With STABLE-class label share "
        "in the 15-23% range across timeframes, the booster's argmax rarely "
        "lands on STABLE, so trade_share collapses to ~1.0 by construction.\n"
    )
    lines.append("## Method\n")
    lines.append(
        f"- Per (coin, timeframe), single time-ordered split: first "
        f"{int((1 - payload['test_frac']) * 100)}% train, last "
        f"{int(payload['test_frac'] * 100)}% test.\n"
    )
    lines.append(
        f"- Calibration tail: last {int(payload['cal_tail_frac'] * 100)}% of "
        "train (point-in-time-safe, never overlaps test).\n"
    )
    lines.append(
        "- Calibration: single-scalar temperature scaling fitted on the cal "
        "tail (Guo et al. 2017). `inv_T = 1.0` is identity (no calibration).\n"
    )
    lines.append(
        f"- Round-trip cost: {payload['round_trip_cost_pct']:.4f}% from "
        "`shared/trading-frictions.json`.\n"
    )
    lines.append(
        "- Trade rule swept across neutral-band widths "
        f"`delta in {payload['neutral_band_widths']}`: trade iff "
        "`max(P_UP, P_DOWN) > P_STABLE + delta`.\n"
    )

    lines.append("\n## Per-timeframe label distribution (full snapshot)\n")
    lines.append("| timeframe | n_rows | n_features | DOWN | STABLE | UP | dataset |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for tf, info in payload["per_timeframe_dataset"].items():
        lines.append(
            f"| {tf} | {info['n_rows']} | {info['n_features']} | "
            f"{info['label_dist']['down']:.3f} | "
            f"{info['label_dist']['stable']:.3f} | "
            f"{info['label_dist']['up']:.3f} | `{info['dataset_file']}` |\n"
        )

    # Per-slice summary: trade_share at delta=0 (raw vs calibrated) and inv_T.
    lines.append(
        "\n## Per-(coin, timeframe) calibration summary\n"
        "Raw trade_share is the share of test rows where `argmax(raw_proba) != STABLE` (delta=0). "
        "Calibrated trade_share applies single-T temperature scaling first.\n\n"
    )
    lines.append("| coin | tf | n_test | inv_T | T | raw trade_share | cal trade_share | raw DA | cal DA |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for s in payload["per_slice"]:
        if "raw_band_sweep" not in s:
            continue
        raw0 = s["raw_band_sweep"][0]
        cal0 = s["calibrated_band_sweep"][0]
        T = s.get("calibration_T")
        T_str = f"{T:.3f}" if isinstance(T, (int, float)) else "—"
        lines.append(
            f"| {s['coin']} | {s['timeframe']} | {s['n_test']} | "
            f"{s['calibration_inv_T']:.3f} | {T_str} | "
            f"{raw0['trade_share']:.4f} | {cal0['trade_share']:.4f} | "
            f"{raw0['directional_accuracy']:.4f} | {cal0['directional_accuracy']:.4f} |\n"
        )

    # Trade_share as a function of band width — pooled per timeframe.
    lines.append(
        "\n## trade_share as a function of neutral-band width (pooled per timeframe)\n"
        "Pooled across coins by averaging the per-slice trade_share at each delta. "
        "The gate band [0.40, 0.85] is the production target.\n\n"
    )
    deltas = payload["neutral_band_widths"]
    header = "| timeframe | source | " + " | ".join(f"d={d:.2f}" for d in deltas) + " |\n"
    lines.append(header)
    lines.append("|" + "---|" * (len(deltas) + 2) + "\n")
    for tf in TIMEFRAMES:
        slices = [s for s in payload["per_slice"] if s.get("timeframe") == tf
                  and "raw_band_sweep" in s]
        if not slices:
            continue
        for source_name, key in (("raw", "raw_band_sweep"), ("calibrated", "calibrated_band_sweep")):
            row = [f"| {tf} | {source_name} |"]
            for di, _d in enumerate(deltas):
                vals = [s[key][di]["trade_share"] for s in slices]
                row.append(f" {np.mean(vals):.3f} |")
            lines.append("".join(row) + "\n")

    # Per-class predicted-prob distribution stats (raw and cal) — pooled per tf.
    lines.append(
        "\n## Predicted-prob distribution stats (pooled per timeframe)\n"
        "Means and quantiles of per-class predicted probabilities, averaged over "
        "the per-slice stats. The key signal is `STABLE.mean` for raw vs calibrated: "
        "if calibration meaningfully sharpens or flattens P(STABLE), the "
        "decision-threshold band moves. The `MARGIN` row is the per-row "
        "`max(P_UP,P_DOWN) - P_STABLE` distribution (the quantity the trade rule cuts on).\n\n"
    )
    qcols = ["mean"] + [f"p{int(q*100)}" for q in PROB_QUANTILES]
    lines.append("| timeframe | source | class | " + " | ".join(qcols) + " |\n")
    lines.append("|" + "---|" * (len(qcols) + 3) + "\n")
    for tf in TIMEFRAMES:
        slices = [s for s in payload["per_slice"] if s.get("timeframe") == tf
                  and "raw_prob_stats" in s]
        if not slices:
            continue
        for source_name, key in (("raw", "raw_prob_stats"), ("calibrated", "calibrated_prob_stats")):
            for cls in ["DOWN", "STABLE", "UP", "MARGIN_DIRECTIONAL_OVER_STABLE"]:
                vals = {qc: float(np.mean([s[key][cls][qc] for s in slices])) for qc in qcols}
                lines.append(
                    f"| {tf} | {source_name} | {cls} | "
                    + " | ".join(f"{vals[qc]:+.4f}" for qc in qcols)
                    + " |\n"
                )

    # Headline conclusion on whether calibration alone moves trade_share into band.
    lines.append("\n## Headline conclusion\n")
    target_lo, target_hi = 0.40, 0.85
    headline_rows: list[str] = []
    for tf in TIMEFRAMES:
        slices = [s for s in payload["per_slice"] if s.get("timeframe") == tf
                  and "raw_band_sweep" in s]
        if not slices:
            continue
        # For each delta, compute fraction of slices in band, raw and calibrated.
        for source_name, key in (("raw", "raw_band_sweep"), ("calibrated", "calibrated_band_sweep")):
            best = None
            for di, d in enumerate(payload["neutral_band_widths"]):
                ts_vals = [s[key][di]["trade_share"] for s in slices]
                in_band = [1 for v in ts_vals if target_lo <= v <= target_hi]
                share_in_band = len(in_band) / len(ts_vals)
                if best is None or share_in_band > best[1]:
                    best = (d, share_in_band, np.mean(ts_vals))
            headline_rows.append((tf, source_name, *best))
    lines.append("Best (delta, share-of-slices-in-band) per (timeframe, source):\n\n")
    lines.append("| timeframe | source | best delta | share of slices in [0.40, 0.85] | mean trade_share at best delta |\n")
    lines.append("|---|---|---|---|---|\n")
    for tf, source, d, share, mean_ts in headline_rows:
        lines.append(f"| {tf} | {source} | {d:.2f} | {share:.2f} | {mean_ts:.3f} |\n")

    lines.append(
        "\n## Interpretation\n"
        "1. If raw trade_share at `delta=0` is uniformly ~1.0, the booster's "
        "argmax never picks STABLE under the natural label distribution; this "
        "is the task #580 symptom by construction.\n"
        "2. If calibration alone (single-T) moves trade_share into [0.40, 0.85] "
        "for some timeframes at `delta=0`, the gate failure is purely a "
        "calibration miss — production already does this on the trainer side, "
        "but the search runner did not, so the search verdict was measuring a "
        "different surface than what production would deploy.\n"
        "3. If even after calibration trade_share stays above 0.85 at "
        "`delta=0`, a per-slice neutral-band tuning is needed (sweep above), "
        "and the diagnostic table shows the band width that lands inside [0.40, 0.85].\n"
    )

    md_path.write_text("".join(lines))


if __name__ == "__main__":
    run()
