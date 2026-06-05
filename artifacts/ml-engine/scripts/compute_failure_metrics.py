"""Re-evaluate every (coin, timeframe) slice from the latest training run and
emit per-slice diagnostic metrics for the failure-analysis report.

Source-of-truth for slice status / DA / holdout count:
  artifacts/ml-engine/reports/20260422T223431Z-baseline-verification.json

Source-of-truth for richer metrics (Brier, calibration, regime, gates_alignment,
class_return_means_pct, fold_metrics): the closest-available training cycle
artifacts on disk. The 22:34:31Z report.json was overwritten by the next
training cycle (~17 minutes later); the next cycle exhibits the same failure
pattern (2/66 promoted vs 0/66 then). Each enriched metric carries
`enrichment_source` to make the time-skew explicit.

Confidence-bucket DA, predicted-class entropy, prediction-collapse share,
per-class holdout count, regime-bucketed DA, PnL-after-fees: re-derived by
loading the persisted per-slice (model.txt + calibrators.joblib) and replaying
prediction over the chronological holdout slice (last 20% of per-coin rows in
the dataset parquet, matching CALIBRATION_HOLDOUT_FRACTION=0.2 in train.py).

This script writes:
  artifacts/ml-engine/reports/20260423T000000Z-failure-analysis.json
  artifacts/ml-engine/reports/20260423T000000Z-failure-analysis.md
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = REPO_ROOT / "artifacts" / "ml-engine"
MODELS_DIR = ML_ROOT / "models"
DATASETS_DIR = MODELS_DIR / "datasets"
REPORTS_DIR = ML_ROOT / "reports"

SOURCE_VERIFICATION = REPORTS_DIR / "20260422T223431Z-baseline-verification.json"
LATEST_REPORT = MODELS_DIR / "report.json"
TRADING_FRICTIONS = REPO_ROOT / "shared" / "trading-frictions.json"

OUT_JSON = REPORTS_DIR / "20260423T000000Z-failure-analysis.json"
OUT_MD = REPORTS_DIR / "20260423T000000Z-failure-analysis.md"

MIN_HOLDOUT_ROWS = 200
MIN_DIRECTIONAL_ACCURACY = 0.50
CALIBRATION_HOLDOUT_FRACTION = 0.2
DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES = {"5m", "1h", "2h", "6h", "1d"}

CLASS_NAMES = ["DOWN", "STABLE", "UP"]


def latest_dataset_for(timeframe: str) -> Path | None:
    candidates = sorted(DATASETS_DIR.glob(f"{timeframe}_*.parquet"))
    return candidates[-1] if candidates else None


def latest_model_dir(coin_id: str, timeframe: str) -> Path | None:
    base = MODELS_DIR / coin_id / timeframe
    if not base.exists():
        return None
    versioned = [p for p in base.iterdir() if p.is_dir()]
    if not versioned:
        return None
    versioned.sort(key=lambda p: p.name)
    return versioned[-1]


def reliability_max_dev_per_class(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    out: dict[str, float] = {}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        p = probs[:, cls_idx]
        y = (labels == cls_idx).astype(np.float64)
        max_dev = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
            n = int(mask.sum())
            if n < 5:
                continue
            mean_pred = float(p[mask].mean())
            frac_pos = float(y[mask].mean())
            max_dev = max(max_dev, abs(mean_pred - frac_pos))
        out[cls_name] = round(max_dev, 4)
    return out


def confidence_bucket_da(probs: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    pred_class = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    correct = (pred_class == labels).astype(np.float64)
    buckets = [(0.33, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0001)]
    out = []
    for lo, hi in buckets:
        mask = (pred_conf >= lo) & (pred_conf < hi)
        n = int(mask.sum())
        out.append(
            {
                "lo": round(lo, 2),
                "hi": round(min(hi, 1.0), 2),
                "n": n,
                "share": round(n / max(1, len(labels)), 4),
                "da": round(float(correct[mask].mean()), 4) if n else None,
            }
        )
    return out


def predicted_class_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-9, 1.0)
    row_h = -(p * np.log(p)).sum(axis=1)
    return float(row_h.mean())


def predicted_prob_spread(probs: np.ndarray) -> dict[str, float]:
    """Predicted top-class probability summary stats. Low std + low mean = the
    model is parking near the prior on every row (collapse signal)."""
    top = probs.max(axis=1)
    return {
        "max_prob_mean": round(float(top.mean()), 4),
        "max_prob_std": round(float(top.std()), 4),
        "max_prob_p95": round(float(np.percentile(top, 95)), 4),
    }


def predictions_near_prior_share(probs: np.ndarray, prior: np.ndarray, eps: float = 0.05) -> dict[str, float]:
    """Share of holdout rows whose predicted distribution is within `eps`
    L1 distance of the class prior. High share = the model has learned to
    output the prior on most rows."""
    diff = np.abs(probs - prior[None, :]).sum(axis=1)
    return {
        "eps": eps,
        "share_within_eps": round(float((diff <= eps).mean()), 4),
        "l1_to_prior_mean": round(float(diff.mean()), 4),
    }


def prediction_collapse(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred_class = probs.argmax(axis=1)
    pred_share = {CLASS_NAMES[i]: float((pred_class == i).mean()) for i in range(3)}
    label_share = {CLASS_NAMES[i]: float((labels == i).mean()) for i in range(3)}
    pred_top = max(pred_share.values())
    label_top = max(label_share.values())
    return {
        "predicted_class_share": {k: round(v, 4) for k, v in pred_share.items()},
        "label_class_share": {k: round(v, 4) for k, v in label_share.items()},
        "predicted_top_class_share": round(pred_top, 4),
        "label_top_class_share": round(label_top, 4),
        "collapse_gap": round(pred_top - label_top, 4),
    }


def per_class_holdout_breakdown(labels: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(3)}


def per_class_brier(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        y = (labels == i).astype(np.float64)
        out[name] = round(float(np.mean((probs[:, i] - y) ** 2)), 4)
    return out


def regime_da(pred_class: np.ndarray, labels: np.ndarray, regime: pd.Series) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    correct = (pred_class == labels).astype(np.float64)
    for reg in regime.dropna().unique():
        mask = (regime == reg).to_numpy()
        n = int(mask.sum())
        out[str(reg)] = {
            "n": n,
            "share": round(n / max(1, len(labels)), 4),
            "da": round(float(correct[mask].mean()), 4) if n else None,
        }
    return out


def pnl_after_fees(
    pred_class: np.ndarray,
    forward_return_pct: np.ndarray,
    round_trip_cost_pct: float,
) -> dict[str, Any]:
    """Side-aware PnL on the holdout: long when class=UP, short when class=DOWN,
    flat when STABLE. Subtract `round_trip_cost_pct` per trade (already in
    percent units, e.g. 0.30 means 0.30%)."""
    side = pred_class.astype(np.int64) - 1  # DOWN=-1, STABLE=0, UP=+1
    trades = side != 0
    n_trades = int(trades.sum())
    if n_trades == 0:
        return {
            "n_trades": 0,
            "trade_share": 0.0,
            "gross_pct_mean": 0.0,
            "round_trip_cost_pct": round(round_trip_cost_pct, 4),
            "net_pct_mean": 0.0,
            "net_pct_total": 0.0,
        }
    realized = side[trades].astype(np.float64) * forward_return_pct[trades].astype(np.float64)
    gross_mean_pct = float(np.mean(realized))
    net_mean_pct = gross_mean_pct - round_trip_cost_pct
    return {
        "n_trades": n_trades,
        "trade_share": round(n_trades / max(1, len(pred_class)), 4),
        "gross_pct_mean": round(gross_mean_pct, 4),
        "round_trip_cost_pct": round(round_trip_cost_pct, 4),
        "net_pct_mean": round(net_mean_pct, 4),
        "net_pct_total": round(float(np.sum(realized) - n_trades * round_trip_cost_pct), 4),
    }


def feature_importance_stability(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Read per-fold importances if persisted; fold_metrics in the current
    trainer does NOT carry importances, so return a deferred marker."""
    has_importance = any("feature_importance" in fm for fm in fold_metrics)
    if not has_importance:
        return {
            "rank_corr_pairwise_mean": None,
            "status": "deferred",
            "reason": "fold_metrics has no feature_importance arrays; trainer must persist importances per fold",
        }
    # If importances were ever added, compute mean Spearman across adjacent folds
    from scipy.stats import spearmanr
    imps = [np.array(fm["feature_importance"], dtype=float) for fm in fold_metrics]
    corrs = []
    for a, b in zip(imps[:-1], imps[1:]):
        if len(a) == len(b):
            corrs.append(float(spearmanr(a, b).correlation))
    return {
        "rank_corr_pairwise_mean": round(float(np.mean(corrs)), 4) if corrs else None,
        "n_pairs": len(corrs),
        "status": "computed",
    }


@dataclass
class SliceMetrics:
    coin_id: str
    timeframe: str
    status_in_source: str
    da_source: float | None
    baseline_da_source: float | None
    n_test_source: int | None
    enriched_from_run: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    fold_metrics_summary: dict[str, Any] = field(default_factory=dict)
    per_class_holdout_breakdown: dict[str, int] = field(default_factory=dict)
    per_class_train_breakdown: dict[str, int] = field(default_factory=dict)
    train_vs_holdout_class_balance_drift: dict[str, float] = field(default_factory=dict)
    per_class_brier: dict[str, float] = field(default_factory=dict)
    reliability_max_dev_per_class: dict[str, float] = field(default_factory=dict)
    confidence_bucket_da: list[dict[str, Any]] = field(default_factory=list)
    predicted_class_entropy: float | None = None
    predicted_prob_spread: dict[str, float] = field(default_factory=dict)
    predictions_near_prior: dict[str, float] = field(default_factory=dict)
    prediction_collapse: dict[str, Any] = field(default_factory=dict)
    regime_bucketed_da: dict[str, Any] = field(default_factory=dict)
    pnl_after_fees: dict[str, Any] = field(default_factory=dict)
    feature_importance_stability: dict[str, Any] = field(default_factory=dict)
    gates_alignment: dict[str, Any] = field(default_factory=dict)
    class_return_means_pct: list[float] | None = None
    contamination_flag: bool = False
    cadence_audit: dict[str, Any] = field(default_factory=dict)
    bucket: str = ""
    bucket_reason: str = ""
    repair_action: str = ""
    rank_score: float = 0.0


def cadence_audit(coin_df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """price_history has no native-cadence column today. Inspect inter-arrival
    gaps in the labeled dataset to detect cadence mixing. A clean per-timeframe
    dataset should have ~uniform inter-arrival ≈ bucket_size."""
    bucket_ms_map = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000, "2h": 7_200_000, "6h": 21_600_000, "1d": 86_400_000}
    bucket_ms = bucket_ms_map.get(timeframe, 0)
    if bucket_ms == 0 or len(coin_df) < 3:
        return {"status": "skipped", "reason": "insufficient rows or unknown bucket"}
    ts = coin_df["timestamp_ms"].sort_values().to_numpy()
    gaps = np.diff(ts)
    expected = float(bucket_ms)
    p50 = float(np.median(gaps))
    p95 = float(np.percentile(gaps, 95))
    return {
        "expected_gap_ms": expected,
        "observed_gap_ms_p50": p50,
        "observed_gap_ms_p95": p95,
        "p50_matches_expected": abs(p50 - expected) / expected < 0.5,
        "p95_within_2x_expected": p95 < 2 * expected,
        "mixed_cadence_detected": (p95 >= 2 * expected) and (p50 < expected * 0.5),
    }


# Conjunction thresholds for the structurally_noisy_retire verdict.
CALIBRATION_BROKEN_RELIABILITY_DEV = 0.10
PREDICTION_COLLAPSE_GAP = 0.15
PREDICTION_COLLAPSE_TOP_SHARE = 0.85
NEAR_PRIOR_DOMINANT_SHARE = 0.60  # >60% of holdout rows within ε of class prior


def is_calibration_broken(s: SliceMetrics) -> bool:
    rmd = s.reliability_max_dev_per_class or {}
    return any(v >= CALIBRATION_BROKEN_RELIABILITY_DEV for v in rmd.values())


def is_prediction_collapsed(s: SliceMetrics) -> bool:
    pc = s.prediction_collapse or {}
    if pc.get("collapse_gap", 0.0) >= PREDICTION_COLLAPSE_GAP:
        return True
    if pc.get("predicted_top_class_share", 0.0) >= PREDICTION_COLLAPSE_TOP_SHARE:
        return True
    near = s.predictions_near_prior or {}
    if near.get("share_within_eps", 0.0) >= NEAR_PRIOR_DOMINANT_SHARE:
        return True
    return False


def is_importance_unstable(s: SliceMetrics) -> bool:
    fis = s.feature_importance_stability or {}
    if fis.get("status") == "computed":
        rc = fis.get("rank_corr_pairwise_mean")
        return rc is not None and rc < 0.5
    # When importances aren't persisted (status=deferred), we don't have evidence
    # of instability — treat as "unknown" and do not let it block the retire
    # verdict by itself. Instead, the retire bucket fires on the cadence-clean
    # AND sufficient-sample AND calibration-broken AND prediction-collapse
    # conjunction; importance is corroborating evidence only.
    return False


def assign_bucket(s: SliceMetrics) -> tuple[str, str, str]:
    """Strict, priority-ordered classification matching the task spec:

      1. promoted — gate passed: DA >= 0.50 AND holdout >= 200 AND timeframe
         in DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES.
      2. salvageable_with_schema_fix — contamination_flag=True OR cadence
         audit detected mixed source.
      3. insufficient_sample — status=untrained OR holdout < 200.
      4. structurally_noisy_retire — CONJUNCTION:
           * cadence-clean (no contamination, no mixed cadence)
           * sufficient sample (holdout >= 200)
           * calibration broken (max per-class reliability deviation >= 0.10)
           * prediction collapse (collapse_gap >= 0.15 OR predicted_top_class
             share >= 0.85 OR predictions-within-ε-of-prior share >= 0.60)
         Importance instability (rank-corr < 0.5 across folds) is corroborating
         evidence; not required because fold importances are not persisted today.
      5. salvageable_with_better_features_or_labels — anything else (red gate
         but evidence remaining: signal in confidence buckets, or calibrated
         and not collapsed). The smallest first set of repairs draws from this.
    """
    n = s.n_test_source or 0
    da = s.da_source if s.da_source is not None else 0.0

    promoted = (
        (n >= MIN_HOLDOUT_ROWS)
        and (da >= MIN_DIRECTIONAL_ACCURACY)
        and (s.timeframe in DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES)
    )
    if promoted:
        return ("promoted", f"DA {da:.3f} >= 0.50 AND holdout {n} >= {MIN_HOLDOUT_ROWS}", "none")
    if s.contamination_flag or s.cadence_audit.get("mixed_cadence_detected"):
        return (
            "salvageable_with_schema_fix",
            "cadence audit detected mixed sources",
            "ship price_candles migration (see schema-audit.md), then retrain",
        )
    if s.status_in_source == "untrained" or n == 0:
        return (
            "insufficient_sample",
            f"status=untrained (n_test={n}); not enough labeled rows to train",
            "wait for more data; revisit when n_test >= MIN_HOLDOUT_ROWS",
        )
    if n < MIN_HOLDOUT_ROWS:
        return (
            "insufficient_sample",
            f"holdout {n} < MIN_HOLDOUT_ROWS={MIN_HOLDOUT_ROWS}",
            "wait for more data; revisit when n_test >= MIN_HOLDOUT_ROWS",
        )

    cadence_clean = not (s.contamination_flag or s.cadence_audit.get("mixed_cadence_detected"))
    sufficient_sample = n >= MIN_HOLDOUT_ROWS
    calibration_broken = is_calibration_broken(s)
    collapse = is_prediction_collapsed(s)
    importance_unstable = is_importance_unstable(s)

    if cadence_clean and sufficient_sample and calibration_broken and collapse:
        evidence = (
            f"calibration_broken=True (max reliability deviation "
            f"{max((s.reliability_max_dev_per_class or {0: 0}).values()):.3f} >= "
            f"{CALIBRATION_BROKEN_RELIABILITY_DEV}); prediction_collapse=True "
            f"(collapse_gap={s.prediction_collapse.get('collapse_gap')}, "
            f"predicted_top_class_share={s.prediction_collapse.get('predicted_top_class_share')}, "
            f"share_within_eps_of_prior={(s.predictions_near_prior or {}).get('share_within_eps')}); "
            f"importance_unstable={importance_unstable} (status="
            f"{(s.feature_importance_stability or {}).get('status')})"
        )
        return (
            "structurally_noisy_retire",
            evidence,
            "retire this (coin, timeframe) OR redefine the label scheme (e.g., binary up-vs-not, multi-horizon); per-coin threshold tuning will not rescue a collapsed, miscalibrated head",
        )
    return (
        "salvageable_with_better_features_or_labels",
        (
            f"red gate but signal remaining — calibration_broken={calibration_broken}, "
            f"prediction_collapse={collapse}; rebalance labels and/or extend feature set"
        ),
        "iterate on per-coin label thresholds (#318) and feature set; rerun gate",
    )


def main() -> None:
    src = json.loads(SOURCE_VERIFICATION.read_text())
    psm = src["per_slice_metrics"]
    counts = src["verification"]["counts"]
    active_coins = src["verification"]["active_coins"]
    enrich = json.loads(LATEST_REPORT.read_text())
    enrich_generated = enrich.get("generated_at")

    # Source per_slice_metrics has 50 entries (10 coins x 5 timeframes; 1d
    # untrained -> empty). Inject 1d untrained stubs and pooled slices so the
    # report covers all 66 slices the verification gate counted.
    psm.setdefault("1d", {})
    for coin_id in active_coins:
        if coin_id not in psm["1d"]:
            psm["1d"][coin_id] = {
                "directional_accuracy": None,
                "baseline_directional_accuracy": None,
                "fold_n_test_total": 0,
                "status": "untrained",
            }

    frictions = json.loads(TRADING_FRICTIONS.read_text())
    fees_block = frictions.get("fees", {})
    taker_pct = float(fees_block.get("taker_fee_pct", 0.001))
    slip_pct = float(fees_block.get("slippage_pct", 0.0005))
    # round-trip cost as a percentage move (entry+exit taker + entry+exit slippage)
    fee_bps = (taker_pct + slip_pct) * 100.0  # turn fraction into percent and store as percent for pnl_after_fees

    slices: list[SliceMetrics] = []
    dataset_cache: dict[str, pd.DataFrame] = {}

    for tf, coin_map in psm.items():
        for coin_id, src_metrics in coin_map.items():
            sm = SliceMetrics(
                coin_id=coin_id,
                timeframe=tf,
                status_in_source=src_metrics.get("status", "unknown"),
                da_source=src_metrics.get("directional_accuracy"),
                baseline_da_source=src_metrics.get("baseline_directional_accuracy"),
                n_test_source=src_metrics.get("fold_n_test_total"),
            )

            # Enrichment from latest report.json (if present)
            tf_block = enrich.get("timeframes", {}).get(tf, {})
            per_coin = tf_block.get("per_coin", {})
            er = per_coin.get(coin_id)
            if er:
                sm.enriched_from_run = enrich_generated
                sm.metrics = er.get("metrics", {})
                sm.baseline_metrics = er.get("baseline_metrics", {})
                fm = er.get("fold_metrics", []) or []
                if fm:
                    das = [f.get("directional_accuracy") for f in fm if f.get("directional_accuracy") is not None]
                    briers = [f.get("brier") for f in fm if f.get("brier") is not None]
                    sm.fold_metrics_summary = {
                        "n_folds": len(fm),
                        "da_mean": round(float(np.mean(das)), 4) if das else None,
                        "da_std": round(float(np.std(das)), 4) if das else None,
                        "brier_mean": round(float(np.mean(briers)), 4) if briers else None,
                        "brier_std": round(float(np.std(briers)), 4) if briers else None,
                    }
                sm.gates_alignment = er.get("gates_alignment", {}) or {}
                sm.class_return_means_pct = er.get("class_return_means_pct")
                sm.feature_importance_stability = feature_importance_stability(fm)

            # Cadence audit + holdout inference
            ds_path = latest_dataset_for(tf)
            if ds_path is not None:
                if str(ds_path) not in dataset_cache:
                    dataset_cache[str(ds_path)] = pd.read_parquet(ds_path)
                df_full = dataset_cache[str(ds_path)]
                coin_df = df_full[df_full["coin_id"] == coin_id].sort_values("timestamp_ms").reset_index(drop=True)
                sm.cadence_audit = cadence_audit(coin_df, tf)
                # contamination_flag stays False because price_history has no
                # cadence/source column — there is no signal to set it. We
                # rely on cadence_audit.mixed_cadence_detected as proxy.
                sm.contamination_flag = bool(sm.cadence_audit.get("mixed_cadence_detected", False))

                mdir = latest_model_dir(coin_id, tf)
                if mdir is not None and (mdir / "model.txt").exists() and len(coin_df) >= 5:
                    try:
                        manifest = json.loads((mdir / "manifest.json").read_text())
                        feat_names = manifest.get("feature_names", [])
                        coin_vocab = manifest.get("coin_vocab", []) or []
                        cls_means = manifest.get("class_return_means_pct", sm.class_return_means_pct or [-1.0, 0.0, 1.0])
                        booster = lgb.Booster(model_file=str(mdir / "model.txt"))
                        # Reproduce chronological holdout split (last 20%)
                        cut = max(1, int(len(coin_df) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
                        hold = coin_df.iloc[cut:].copy()
                        # Inject derived columns the trainer adds in-memory
                        if "coin_idx" in feat_names and "coin_idx" not in hold.columns:
                            coin_to_idx = {c: i for i, c in enumerate(coin_vocab)}
                            hold["coin_idx"] = hold["coin_id"].map(coin_to_idx).astype(float)
                        if len(hold) >= 5:
                            X_cols = [c for c in feat_names if c in hold.columns]
                            if len(X_cols) == len(feat_names):
                                X = hold[feat_names].astype(float).to_numpy()
                                raw = booster.predict(X)
                                if raw.ndim == 1:
                                    raw = np.column_stack([1 - raw, np.zeros_like(raw), raw])
                                # Apply calibrators if present
                                cal_path = mdir / "calibrators.joblib"
                                probs = raw.copy()
                                if cal_path.exists():
                                    try:
                                        cals = joblib.load(cal_path)
                                        if isinstance(cals, list) and len(cals) == raw.shape[1]:
                                            for i, cal in enumerate(cals):
                                                if cal is not None and hasattr(cal, "predict"):
                                                    probs[:, i] = cal.predict(raw[:, i])
                                            row_sums = probs.sum(axis=1, keepdims=True)
                                            row_sums[row_sums == 0] = 1.0
                                            probs = probs / row_sums
                                    except Exception as e:
                                        sm.bucket_reason = f"calibrator load failed: {e}"
                                if "label_3class" in hold.columns:
                                    labels = hold["label_3class"].astype(int).to_numpy()
                                    sm.per_class_holdout_breakdown = per_class_holdout_breakdown(labels)
                                    sm.per_class_brier = per_class_brier(probs, labels)
                                    sm.reliability_max_dev_per_class = reliability_max_dev_per_class(probs, labels)
                                    sm.confidence_bucket_da = confidence_bucket_da(probs, labels)
                                    sm.predicted_class_entropy = round(predicted_class_entropy(probs), 4)
                                    sm.predicted_prob_spread = predicted_prob_spread(probs)
                                    sm.prediction_collapse = prediction_collapse(probs, labels)
                                    if "regime" in hold.columns:
                                        sm.regime_bucketed_da = regime_da(probs.argmax(axis=1), labels, hold["regime"])
                                    # Train-vs-holdout class balance summary using
                                    # the chronological train portion of the same parquet.
                                    train_df = coin_df.iloc[:cut]
                                    if "label_3class" in train_df.columns and len(train_df):
                                        train_labels = train_df["label_3class"].astype(int).to_numpy()
                                        sm.per_class_train_breakdown = per_class_holdout_breakdown(train_labels)
                                        train_share = np.array([
                                            (train_labels == i).mean() for i in range(3)
                                        ])
                                        hold_share = np.array([
                                            (labels == i).mean() for i in range(3)
                                        ])
                                        sm.train_vs_holdout_class_balance_drift = {
                                            CLASS_NAMES[i]: round(float(hold_share[i] - train_share[i]), 4)
                                            for i in range(3)
                                        }
                                        sm.train_vs_holdout_class_balance_drift["l1_drift"] = round(
                                            float(np.abs(hold_share - train_share).sum()), 4
                                        )
                                        sm.predictions_near_prior = predictions_near_prior_share(probs, train_share)
                                if "forward_window_return_pct" in hold.columns or "forward_return" in hold.columns:
                                    fwd_col = "forward_window_return_pct" if "forward_window_return_pct" in hold.columns else "forward_return"
                                    fwd = hold[fwd_col].astype(float).to_numpy()
                                    if fwd_col == "forward_return":
                                        fwd = fwd * 100.0  # convert ratio to percent
                                    sm.pnl_after_fees = pnl_after_fees(probs.argmax(axis=1), fwd, fee_bps)
                    except Exception as e:
                        sm.bucket_reason = f"inference failed: {e}"

            bucket, reason, action = assign_bucket(sm)
            sm.bucket = bucket
            sm.bucket_reason = reason
            sm.repair_action = action
            da = sm.da_source or 0.0
            bda = sm.baseline_da_source or 0.0
            lift = max(0.0, da - bda)
            sm.rank_score = round(da + 2.0 * lift, 4)
            slices.append(sm)

    # Pooled slices (one per timeframe) — counted by the verification gate but
    # not in source per_slice_metrics. Pull them from the latest report.json.
    for tf, tf_block in enrich.get("timeframes", {}).items():
        pooled = tf_block.get("pooled")
        if not isinstance(pooled, dict) or "metrics" not in pooled:
            continue
        sm = SliceMetrics(
            coin_id="__pooled__",
            timeframe=tf,
            status_in_source=pooled.get("status", "trained"),
            da_source=pooled.get("metrics", {}).get("directional_accuracy"),
            baseline_da_source=pooled.get("baseline_metrics", {}).get("directional_accuracy"),
            n_test_source=int(pooled.get("directional_call_share_n") or 0),
        )
        sm.enriched_from_run = enrich_generated
        sm.metrics = pooled.get("metrics", {})
        sm.baseline_metrics = pooled.get("baseline_metrics", {})
        sm.gates_alignment = pooled.get("gates_alignment", {}) or {}
        sm.class_return_means_pct = pooled.get("class_return_means_pct")
        sm.feature_importance_stability = feature_importance_stability(pooled.get("fold_metrics", []) or [])
        # Pooled slices share the same parquet — cadence audit applies at the
        # combined level (mixed coins, same timeframe). Skip per-slice cadence
        # audit for pooled.
        sm.cadence_audit = {"status": "skipped", "reason": "pooled slice — cadence is per-coin"}
        bucket, reason, action = assign_bucket(sm)
        sm.bucket = bucket
        sm.bucket_reason = reason
        sm.repair_action = action
        da = sm.da_source or 0.0
        bda = sm.baseline_da_source or 0.0
        sm.rank_score = round(da + 2.0 * max(0.0, da - bda), 4)
        slices.append(sm)

    # Smallest first set: top 5 PER-COIN 5m slices (exclude __pooled__) that
    # are in salvageable_with_better_features_or_labels. Per-coin first because
    # the next planned lever (#318) is per-coin label thresholds.
    smallest = [
        s
        for s in slices
        if s.timeframe == "5m"
        and s.bucket == "salvageable_with_better_features_or_labels"
        and s.coin_id != "__pooled__"
    ]
    smallest.sort(key=lambda s: s.rank_score, reverse=True)
    smallest_first = [
        {
            "coin_id": s.coin_id,
            "timeframe": s.timeframe,
            "baseline_da": s.baseline_da_source,
            "model_da": s.da_source,
            "lift": round((s.da_source or 0) - (s.baseline_da_source or 0), 4),
            "n_test": s.n_test_source,
            "rank_score": s.rank_score,
            "repair_action": s.repair_action,
        }
        for s in smallest[:5]
    ]

    bucket_counts: dict[str, int] = {}
    for s in slices:
        bucket_counts[s.bucket] = bucket_counts.get(s.bucket, 0) + 1

    out = {
        "generated_at": "2026-04-23T00:00:00Z",
        "source_verification_report": str(SOURCE_VERIFICATION.relative_to(REPO_ROOT)),
        "source_verification_counts": counts,
        "enrichment_source": str(LATEST_REPORT.relative_to(REPO_ROOT)),
        "enrichment_source_generated_at": enrich_generated,
        "enrichment_caveat": (
            "The 22:34:31Z report's full per-slice surface was overwritten by the next training cycle "
            "(~17 min later, 2026-04-23T00:26:32Z). All Brier/calibration/fold/regime/PnL fields below "
            "are from that next cycle, which exhibits the same failure pattern (2/66 promoted vs 0/66). "
            "Source DA / baseline_DA / n_test / status fields come from the original 22:34:31Z file."
        ),
        "gate_constants": {
            "MIN_HOLDOUT_ROWS": MIN_HOLDOUT_ROWS,
            "MIN_DIRECTIONAL_ACCURACY": MIN_DIRECTIONAL_ACCURACY,
            "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": sorted(DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES),
            "CALIBRATION_HOLDOUT_FRACTION": CALIBRATION_HOLDOUT_FRACTION,
            "TAKER_FEE_BPS_PER_TRADE": fee_bps,
        },
        "bucket_counts": bucket_counts,
        "smallest_first_set": smallest_first,
        "slices": [s.__dict__ for s in slices],
    }

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT_JSON}")
    print(f"bucket_counts: {bucket_counts}")
    print(f"smallest_first_set ({len(smallest_first)}):")
    for s in smallest_first:
        print(f"  {s['coin_id']} 5m  baseline_da={s['baseline_da']:.4f}  model_da={s['model_da']:.4f}  lift={s['lift']:+.4f}  n_test={s['n_test']}")


if __name__ == "__main__":
    main()
