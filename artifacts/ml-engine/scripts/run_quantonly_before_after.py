"""Task #365 — direct per-slot before/after metrics for the audit.

Goal: produce a real numeric demonstration that the quant-only feature
set (no `news_*` columns) trains a model whose holdout metrics are at
least on par with the archived contaminated baseline. The full
production training pipeline (train_one_slice) runs an Optuna search +
isotonic calibration loop that takes minutes per slot — too slow to
run inline in the audit shell. This script therefore does a *direct*
LightGBM holdout evaluation that mirrors what
`train_one_slice` measures (binary directional AUC, Brier score, log
loss) but skips the calibration / Optuna / multi-fold CV machinery.
That is sufficient for an audit-grade before/after delta — production
retraining still uses the full pipeline via the 30-min auto-retrain
loop.

Output: `audit/before_after_metrics.json` with one entry per
(coin, timeframe) slot retrained.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
# Import the *canonical* feature contract from the live registry so
# this audit script and the production training pipeline share a
# single source of truth. The reviewer flagged a previous mirror as a
# drift risk — fixed by re-exporting here.
sys.path.insert(0, str(REPO_ROOT / "artifacts" / "ml-engine"))
from app.training.registry import (  # noqa: E402
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_PREFIXES,
    POOLED_COIN_ID,
)

DATASETS = REPO_ROOT / "artifacts" / "ml-engine" / "models" / "datasets"
AUDIT = REPO_ROOT / "audit"


def latest_parquet(tf: str) -> Path | None:
    cands = sorted(DATASETS.glob(f"{tf}_*.parquet"))
    return cands[-1] if cands else None


def fit_eval(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train LightGBM on the first 80% of the (chronologically sorted)
    rows and evaluate on the last 20% (the same split shape
    train_one_slice's final holdout uses). Returns binary directional
    AUC/Brier/log-loss against `direction_up_3c == 2` (UP)."""
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    # Coin index encoding (same scheme as registry: index into vocab).
    if "coin_idx" in feature_cols and "coin_idx" not in df.columns:
        vocab = {c: i for i, c in enumerate(sorted(df["coin_id"].unique()))}
        df["coin_idx"] = df["coin_id"].map(vocab).fillna(-1).astype(int)
    # Filter to the columns that actually exist in the parquet.
    use_cols = [c for c in feature_cols if c in df.columns]
    X = df[use_cols].astype(float).fillna(0.0).values
    # Label: parquets store either `label_3c` (0/1/2) or `forward_return`
    # — derive a binary UP label from whichever is present.
    if "label_3c" in df.columns:
        y_up = (df["label_3c"] == 2).astype(int).values
    elif "label" in df.columns and df["label"].dtype != object:
        y_up = (df["label"] > 0).astype(int).values
    elif "forward_return" in df.columns:
        y_up = (df["forward_return"] > 0).astype(int).values
    else:
        return {"status": "no_label", "n_features_used": len(use_cols)}
    n = len(df)
    cut = int(n * 0.8)
    if cut < 50 or (n - cut) < 20:
        return {"status": "too_small", "n_rows": n}
    Xtr, ytr = X[:cut], y_up[:cut]
    Xte, yte = X[cut:], y_up[cut:]
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return {"status": "single_class", "n_rows": n}
    booster = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "verbose": -1,
            "num_threads": 4,
        },
        lgb.Dataset(Xtr, label=ytr),
        num_boost_round=40,
    )
    p = booster.predict(Xte)
    return {
        "status": "ok",
        "n_rows": n,
        "n_train": int(cut),
        "n_test": int(n - cut),
        "n_features_used": len(use_cols),
        "metrics": {
            "auc": float(roc_auc_score(yte, p)),
            "brier": float(brier_score_loss(yte, p)),
            "log_loss": float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))),
        },
    }


def main() -> int:
    # Sanity: live FEATURE_COLUMNS contract has zero forbidden columns.
    bad = [c for c in FEATURE_COLUMNS if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)]
    assert not bad, f"FEATURE_COLUMNS still leaks: {bad}"

    before = json.loads((AUDIT / "archived_models.json").read_text())
    by_slot: dict[tuple[str, str], list] = defaultdict(list)
    for m in before["models"]:
        by_slot[(m["coin_id"], m["timeframe"])].append(m)

    out: dict = {
        "task": 365,
        "method": (
            "Direct LightGBM 80/20 chronological holdout on the latest "
            "labelled parquet per timeframe, using the post-#365 "
            "FEATURE_COLUMNS contract. Production training still uses "
            "train_one_slice (Optuna + isotonic calibration + multi-fold "
            "CV); this is the fast audit verification that the new "
            "feature set trains and yields a usable directional model."
        ),
        "feature_columns_after": FEATURE_COLUMNS,
        "forbidden_prefixes": list(FORBIDDEN_FEATURE_PREFIXES),
        "n_features_after": len(FEATURE_COLUMNS),
        "slots": [],
    }

    # All six production timeframes (`SUPPORTED_TIMEFRAMES`). The
    # script can be invoked with `--tf 1m,5m` to retrain a subset
    # (used to extend an existing audit table without re-doing the
    # longer timeframes).
    tfs = ("1d", "6h", "2h", "1h", "5m", "1m")
    for i, a in enumerate(sys.argv):
        if a == "--tf" and i + 1 < len(sys.argv):
            tfs = tuple(t.strip() for t in sys.argv[i + 1].split(",") if t.strip())
            break
    # If we're running a partial subset, merge with any existing entries
    # so the file always represents the full audit table.
    existing: dict = {}
    if (AUDIT / "before_after_metrics.json").exists() and tfs != (
        "1d", "6h", "2h", "1h", "5m", "1m"
    ):
        try:
            existing = json.loads((AUDIT / "before_after_metrics.json").read_text())
            for slot in existing.get("slots", []):
                if slot.get("timeframe") not in tfs:
                    out["slots"].append(slot)
        except Exception:
            pass
    def latest_baseline(coin: str, tf: str) -> dict:
        rows = sorted(by_slot.get((coin, tf), []),
                      key=lambda r: r.get("version", ""), reverse=True)
        return rows[0] if rows else {}

    def deltas(after: dict, before_metrics: dict) -> dict:
        d: dict = {}
        if after.get("status") == "ok" and before_metrics:
            for k in ("auc", "log_loss", "brier"):
                if before_metrics.get(k) is not None and after["metrics"].get(k) is not None:
                    d[k] = round(after["metrics"][k] - before_metrics[k], 5)
        return d

    for tf in tfs:
        parquet = latest_parquet(tf)
        if parquet is None:
            out["slots"].append({"timeframe": tf, "status": "no_parquet"})
            continue
        df = pd.read_parquet(parquet)
        # ---- POOLED SLOT (all coins) ----
        result = fit_eval(df, FEATURE_COLUMNS)
        baseline = latest_baseline(POOLED_COIN_ID, tf)
        before_metrics = (baseline or {}).get("metrics") or {}
        delta = deltas(result, before_metrics)
        slot = {
            "coin_id": POOLED_COIN_ID,
            "timeframe": tf,
            "parquet": parquet.name,
            "before": {
                "version": baseline.get("version"),
                "model_kind": baseline.get("model_kind"),
                "n_features_with_news": (
                    len(baseline.get("forbidden_features", [])) + len(FEATURE_COLUMNS)
                    if baseline else None
                ),
                "metrics": before_metrics,
                "directional_call_share": baseline.get("directional_call_share"),
            },
            "after": result,
            "delta_after_minus_before": delta,
        }
        out["slots"].append(slot)
        m = result.get("metrics") or {}
        print(
            f"[{tf} __pooled__] before_auc={before_metrics.get('auc')} "
            f"after_auc={m.get('auc')} delta_auc={delta.get('auc')} "
            f"after_brier={m.get('brier')}"
        )

        # ---- PER-COIN SLOTS ----
        # Retrain each per-coin specialist on its own slice of the
        # parquet, mirroring how the production training loop builds
        # per-coin models. We process every coin that (a) has a baseline
        # entry in archived_models.json for this timeframe AND (b) has
        # enough rows in the parquet to satisfy fit_eval's min-row gate.
        # This produces the per-coin AUC/Brier deltas the audit needs;
        # production training reproduces them via the full Optuna +
        # isotonic pipeline through the 30-min auto-retrain loop.
        per_coin_baselines = [
            (k[0], v) for k, v in by_slot.items()
            if k[1] == tf and k[0] != POOLED_COIN_ID
        ]
        # Sort by archived AUC so the first per-coin entries are the
        # most important historical specialists.
        per_coin_baselines.sort(
            key=lambda kv: -((sorted(kv[1], key=lambda r: r.get("version", ""), reverse=True)[0]).get("metrics", {}).get("auc") or 0)
        )
        coins_in_parquet = set(df["coin_id"].unique()) if "coin_id" in df.columns else set()
        per_coin_done = 0
        for coin, _rows in per_coin_baselines:
            if coin not in coins_in_parquet:
                continue
            sub = df[df["coin_id"] == coin]
            if len(sub) < 100:
                continue
            sub_result = fit_eval(sub, FEATURE_COLUMNS)
            sub_baseline = latest_baseline(coin, tf)
            sub_before = (sub_baseline or {}).get("metrics") or {}
            sub_delta = deltas(sub_result, sub_before)
            out["slots"].append({
                "coin_id": coin,
                "timeframe": tf,
                "parquet": parquet.name,
                "n_rows_for_coin": int(len(sub)),
                "before": {
                    "version": sub_baseline.get("version"),
                    "model_kind": sub_baseline.get("model_kind"),
                    "metrics": sub_before,
                },
                "after": sub_result,
                "delta_after_minus_before": sub_delta,
            })
            per_coin_done += 1
        print(f"[{tf}] per_coin_slices_retrained={per_coin_done}")

    (AUDIT / "before_after_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {AUDIT / 'before_after_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
