"""Task #365 — offline before/after comparison for the quant-only retrain.

Loads the most recent labeled parquet for each timeframe in
artifacts/ml-engine/models/datasets/ and trains a fresh booster using ONLY
the post-#365 FEATURE_COLUMNS (no `news_*` block). The result is written
to audit/before_after_quantonly.json so the audit report can cite the
holdout AUC / log-loss / Brier / directional-accuracy delta against the
archived baseline (from audit/archived_models.json).

Optuna is dialed down to 1 trial (ML_OPTUNA_N_TRIALS=1) so the run
finishes in seconds — the goal is to demonstrate that removing the
unconditionally-zero `news_*` columns has no negative impact on the
booster, not to ship a tuned production model. The full training cycle
will be auto-triggered on next ML retrain because the on-disk archived
manifests are now rejected by the load-time guard (registry.load_model).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Cap optuna BEFORE importing train.py — the constant is read at module
# import time.
os.environ.setdefault("ML_OPTUNA_N_TRIALS", "1")
os.environ.setdefault("ML_OPTUNA_TIMEOUT_SECONDS", "60")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "artifacts" / "ml-engine"))

import pandas as pd  # noqa: E402

from app.training.registry import (  # noqa: E402
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_PREFIXES,
    POOLED_COIN_ID,
)
from app.training.train import train_one_slice  # noqa: E402

DATASETS = REPO_ROOT / "artifacts" / "ml-engine" / "models" / "datasets"
AUDIT = REPO_ROOT / "audit"
AUDIT.mkdir(exist_ok=True)


def latest_parquet_for(tf: str) -> Path | None:
    candidates = sorted(DATASETS.glob(f"{tf}_*.parquet"))
    return candidates[-1] if candidates else None


def main() -> int:
    # Confirm the live FEATURE_COLUMNS contain no forbidden columns —
    # this is the contract we're training against.
    forbidden_in_contract = [
        c for c in FEATURE_COLUMNS
        if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)
    ]
    assert not forbidden_in_contract, (
        f"FEATURE_COLUMNS still contains forbidden cols: {forbidden_in_contract}"
    )

    out: dict = {
        "task": 365,
        "feature_columns_after": FEATURE_COLUMNS,
        "forbidden_prefixes": list(FORBIDDEN_FEATURE_PREFIXES),
        "n_features_after": len(FEATURE_COLUMNS),
        "slices": {},
    }

    # Train the pooled slot for the two highest-volume timeframes (1m, 5m)
    # plus 1h as the canonical sample. The remaining timeframes will be
    # retrained on the next ML training cycle.
    for tf in ("1m", "5m", "1h"):
        parquet = latest_parquet_for(tf)
        if parquet is None:
            out["slices"][tf] = {"status": "no_parquet"}
            continue
        df = pd.read_parquet(parquet)
        # Build coin vocab from the rows so coin_idx encodes correctly.
        vocab = sorted(df["coin_id"].dropna().unique().tolist())
        try:
            result = train_one_slice(
                df, POOLED_COIN_ID, tf, vocab,
                note="task#365 offline quant-only verification",
                feature_columns=FEATURE_COLUMNS,
            )
        except Exception as exc:  # noqa: BLE001
            out["slices"][tf] = {"status": "error", "error": str(exc), "parquet": str(parquet.name)}
            continue
        # Strip the heavy nested objects — we only want headline metrics.
        slim = {
            "parquet": str(parquet.name),
            "status": result.get("status", "ok"),
            "n_rows": result.get("n_rows"),
            "metrics": result.get("metrics"),
            "baseline_metrics": result.get("baseline_metrics"),
            "directional_call_share": result.get("directional_call_share"),
            "n_features_used": len(FEATURE_COLUMNS),
        }
        out["slices"][tf] = slim
        print(f"[{tf}] {slim}")

    out_path = AUDIT / "before_after_quantonly.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
