"""Task #519 — Re-run the Task #507 focused diagnostic at the production
``ML_LGB_NUM_BOOST_ROUND=800`` boosting budget.

This is a near-verbatim copy of ``run_507_focused.py`` with two
changes:

1. The ``ML_LGB_NUM_BOOST_ROUND`` default is bumped from 80 (the
   diagnostic's CI shortcut) to 800 (production training campaign
   setting). The Task #507 verification PASS verdict was obtained at
   80 rounds; the campaign run after the fix
   (``models/training_run_20260425T063302Z``) showed all 4 stuck
   slices STILL fail the directional-call-share gate. This script
   captures the actual outcome at production settings so the gap is
   documented.
2. Each slice is checkpointed to disk as soon as it finishes so a
   timeout / interrupt does not lose hours of work. The final output
   JSON file is named with the ``task519`` tag to keep the on-disk
   fixture distinct from the ``task507-*-alpha1.0.json`` files the
   ``test_task507_focused_diagnostic_recovers_stuck_slices_at_80_rounds``
   regression test reads, and is the fixture
   ``test_task519_focused_diagnostic_documents_800_round_regression``
   pins.

Output: ``reports/<TS>-task519-booster-collapse-rerun-800rounds.json``.
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
os.environ.setdefault("ML_LGB_NUM_BOOST_ROUND", "800")

from app.training import train as _train_mod  # noqa: E402

from scripts.diagnostic_482.run_507_focused import (  # noqa: E402
    _require_volzscore60,
)
from scripts.diagnostic_482.run_stage_collapse_diagnostic import (  # noqa: E402
    _bucket_for,
    _diagnose_one,
    _latest_pooled_dataset,
)
import pandas as pd  # noqa: E402


STUCK_SLICES = [
    ("bonk", "1d"),
    ("celestia", "1d"),
    ("dogwifcoin", "1d"),
    ("celestia", "6h"),
]


def main() -> int:
    alpha = float(os.environ.get("ML_CLASS_WEIGHT_ALPHA", str(_train_mod.CLASS_WEIGHT_ALPHA)))
    rounds = int(os.environ["ML_LGB_NUM_BOOST_ROUND"])
    print(f"running task #519 focused diagnostic with alpha={alpha} rounds={rounds}",
          flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{ts}-task519-booster-collapse-rerun-{rounds}rounds.json"

    by_tf: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    t0 = time.monotonic()
    for coin, tf in STUCK_SLICES:
        if tf not in by_tf:
            ds = _latest_pooled_dataset(tf)
            print(f"  loading {ds.name}", flush=True)
            df_loaded = pd.read_parquet(ds)
            # Task #539 — fail fast if a stale (pre-Task #537) parquet
            # without the `volZScore60` column ever sneaks back in,
            # rather than silently feeding the booster a zero-filled
            # column.
            _require_volzscore60(df_loaded, ds)
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
        # Checkpoint after every slice — the 800-round runs are slow
        # enough that losing them to a shell timeout would cost hours.
        out.write_text(json.dumps({
            "task": 519,
            "generated_at": ts,
            "class_weight_alpha": alpha,
            "num_boost_round": rounds,
            "stuck_slices": [list(s) for s in STUCK_SLICES],
            "rows": rows,
        }, indent=2))

    print(f"wrote {out.relative_to(ROOT)}")
    print(f"total elapsed: {round(time.monotonic() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
