"""Task #507 — focused booster_collapse re-run diagnostic.

Re-runs the Task #482 stage-collapse diagnostic ONLY on the 4 slices
the calibrator fix could not promote (`bonk@1d`, `celestia@1d`,
`dogwifcoin@1d`, `celestia@6h`). With the full-fleet diagnostic taking
~10 minutes per timeframe, the focused restriction lets us iterate on
the fix and re-verify under a normal shell timeout.

Default behaviour (no env overrides) reflects the production booster
configuration: ``CLASS_WEIGHT_ALPHA=1.0`` plus the tiny-slice branch
in ``app/training/train.py::_train_lgb`` (soft hyperparameters, no
early stopping, ``alpha = 2`` weighting). The optional
``ML_CLASS_WEIGHT_ALPHA`` override is retained so the trainer
constants can still be swept by hand for ablation runs.

Usage:
    python scripts/diagnostic_482/run_507_focused.py
    ML_CLASS_WEIGHT_ALPHA=2.0 python scripts/diagnostic_482/run_507_focused.py

Output: a JSON dump under
`reports/<TS>-task507-booster-collapse-rerun-alpha{alpha}.json` with
one row per stuck slice, in the same schema as the Task #482
stage-collapse diagnostic for direct comparison. The corresponding
unit test
``test_task507_focused_diagnostic_recovers_stuck_slices_at_80_rounds``
asserts every stuck slice clears the verification gate on the latest
fixture in the 80-round regime. The production-rounds (800-round)
counterpart is
``scripts/diagnostic_482/run_519_focused_800rounds.py`` and its
companion test
``test_task519_focused_diagnostic_documents_800_round_regression``
— see ``reports/*-task519-booster-fix-rerun-800rounds-verification.md``
for why the 800-round regime fails on ``bonk@1d`` while the 80-round
regime passes.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ML_SKIP_OPTUNA", "1")
os.environ.setdefault("ML_LGB_NUM_BOOST_ROUND", "80")

# Re-import after env so the trainer constants pick up the override.
from app.training import train as _train_mod  # noqa: E402

# Reuse the diagnostic harness so JSON shape stays compatible with the
# Task #482 fixtures (the test harness already replays them).
from scripts.diagnostic_482.run_stage_collapse_diagnostic import (  # noqa: E402
    _bucket_for,
    _diagnose_one,
    _latest_pooled_dataset,
)
from app.training.registry import FEATURE_COLUMNS  # noqa: E402
import pandas as pd  # noqa: E402


# Task #539 — Task #537 regenerated every cached labeled-dataset
# parquet under `models/datasets/` so `volZScore60` is now part of
# the persisted schema natively. The retrofit shim that used to
# reconstruct the column from `lastPrice` is gone — if the harness
# ever encounters a parquet missing the column the error must
# surface immediately, so the operator refreshes the cache rather
# than silently feeding the booster a zero-filled column.
def _require_volzscore60(df: pd.DataFrame, source: Path) -> None:
    if "volZScore60" not in df.columns:
        raise RuntimeError(
            f"{source}: parquet is missing the `volZScore60` feature "
            "column. Refresh the cached dataset by running "
            "`scripts/refresh_cached_datasets.py` (Task #537) — the "
            "in-memory retrofit shim was removed in Task #539."
        )


STUCK_SLICES = [
    ("bonk", "1d"),
    ("celestia", "1d"),
    ("dogwifcoin", "1d"),
    ("celestia", "6h"),
]


def main() -> int:
    alpha = float(os.environ.get("ML_CLASS_WEIGHT_ALPHA", str(_train_mod.CLASS_WEIGHT_ALPHA)))
    print(f"running task #507 focused diagnostic with alpha={alpha}", flush=True)

    by_tf: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    t0 = time.monotonic()
    for coin, tf in STUCK_SLICES:
        if tf not in by_tf:
            # Task #517 — allow overriding the dataset path per-tf so the
            # focused diagnostic can be replayed against the SAME parquet
            # snapshot Task #507's verification produced its baseline on
            # (the daily writer will keep emitting larger snapshots, and
            # the single-coin signal can drift just from that). Pinning
            # makes the margin improvement a strictly feature-level
            # comparison rather than a feature ⊕ data-drift comparison.
            override = os.environ.get(f"ML_TASK507_{tf.upper()}_DATASET")
            if override:
                ds = Path(override)
                if not ds.is_absolute():
                    ds = ROOT / ds
                if not ds.exists():
                    raise FileNotFoundError(
                        f"ML_TASK507_{tf.upper()}_DATASET={override} not found",
                    )
            else:
                ds = _latest_pooled_dataset(tf)
            print(f"  loading {ds.name}", flush=True)
            df_loaded = pd.read_parquet(ds)
            _require_volzscore60(df_loaded, ds)
            missing = [c for c in FEATURE_COLUMNS if c not in df_loaded.columns
                       and c != "coin_idx"]
            if missing:
                print(f"  WARN: parquet missing feature columns {missing}", flush=True)
            by_tf[tf] = df_loaded
        df = by_tf[tf]
        vocab = sorted(df["coin_id"].unique().tolist())
        t = time.monotonic()
        try:
            row = _diagnose_one(df, coin, tf, vocab)
        except Exception as exc:  # noqa: BLE001
            row = {"coin_id": coin, "timeframe": tf, "status": "error", "error": str(exc)}
        row["bucket"] = _bucket_for(row)
        row["elapsed_s"] = round(time.monotonic() - t, 2)
        rows.append(row)
        print(
            f"  {coin:12} @ {tf:>2}  bucket={row['bucket']:>22}  "
            f"raw_S={row.get('raw_STABLE_share', float('nan')):.4f}  "
            f"DCS={row.get('directional_call_share', float('nan')):.4f}  "
            f"sw_S={row.get('sample_weight_STABLE', float('nan')):.3f}  "
            f"elapsed={row['elapsed_s']}s",
            flush=True,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{ts}-task507-booster-collapse-rerun-alpha{alpha}.json"
    out.write_text(json.dumps({
        "task": 507,
        "generated_at": ts,
        "class_weight_alpha": alpha,
        "stuck_slices": [list(s) for s in STUCK_SLICES],
        "rows": rows,
    }, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"total elapsed: {round(time.monotonic() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
