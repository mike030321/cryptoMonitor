"""Task #537 — Refresh the cached training datasets.

The pooled labeled-dataset parquet snapshots under
`models/datasets/<tf>_<TS>.parquet` (the ones the diagnostic + retrain
harnesses read via `_latest_pooled_dataset`) had drifted up to ~5 days
behind today's market data. They were also missing the `volZScore60`
feature column added in Task #517 (so callers had to re-derive it via
`_augment_with_missing_feature_columns`), and several timeframes were
missing coverage for active coins:

  * 1d had only `bonk`, `floki-inu`, `pepe` (3 of 10).
  * 5m had only `bonk` (1 of 10).
  * 1m / 5m / 1h / 2h / 6h all lacked `sei-network`.

This script regenerates a fresh `<tf>_<TS>.parquet` for every supported
timeframe by calling the live `build_labeled_dataset` pipeline against
the production database. Each emitted frame contains every active
coin (`DEFAULT_COINS + ["sei-network"]`) the database still has data
for, and the `volZScore60` feature column is now part of the persisted
schema (it falls out of `build_feature_vector` natively).

Usage::

    cd artifacts/ml-engine && \\
        ../../.pythonlibs/bin/python -m scripts.refresh_cached_datasets

Environment overrides:
    ML_REFRESH_TIMEFRAMES=1d,6h,2h,1h,5m,1m   (default = all six)
    ML_REFRESH_COINS=pepe,bonk,...            (default = DEFAULT_COINS + sei)
    ML_REFRESH_LOOKBACK_DAYS_<TF>=N           (per-tf override; falls back
                                               to `lookback_days_for(tf)`)

The script stamps a manifest under
`reports/<TS>-task537-refresh-cached-datasets.json` summarising each
timeframe (input lookback days, rows produced, coins emitted, output
path). Failures on one timeframe never block the others — they are
recorded with `status="error"` so the next campaign can re-run only
the failing slices.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import close_pool, init_pool  # noqa: E402
from app.training.labels import build_labeled_dataset  # noqa: E402
from app.training.registry import dataset_path, make_version  # noqa: E402
from app.training.train import (  # noqa: E402
    DEFAULT_COINS,
    DEFAULT_TIMEFRAMES,
    lookback_days_for,
)


def _selected_timeframes() -> list[str]:
    raw = os.environ.get("ML_REFRESH_TIMEFRAMES")
    if not raw:
        # Process the cheaper / smaller-history timeframes first so any
        # OOM on the big 1m/5m frames still leaves the higher-tf caches
        # refreshed for the next retrain.
        return ["1d", "6h", "2h", "1h", "5m", "1m"]
    return [t.strip() for t in raw.split(",") if t.strip()]


def _selected_coins() -> list[str]:
    raw = os.environ.get("ML_REFRESH_COINS")
    if raw:
        return [c.strip() for c in raw.split(",") if c.strip()]
    # Task #537 — `sei-network` is intentionally NOT in
    # `DEFAULT_COINS` (Task #409 dropped it because the 5m
    # strict-contiguous gate could not be met from free venues).
    # The cached pooled snapshots, however, must still surface
    # whatever data the DB has so the per-coin diagnostic + retrain
    # harnesses can include it. The labeling pipeline already drops
    # any coin whose lookback window contains synthetic-only or
    # too-few rows, so adding it here is safe — it'll be emitted
    # only when there's real data to label.
    return list(DEFAULT_COINS) + ["sei-network"]


def _lookback_days(tf: str) -> int:
    raw = os.environ.get(f"ML_REFRESH_LOOKBACK_DAYS_{tf.upper()}")
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return lookback_days_for(tf)


async def _refresh_one(
    tf: str, coin_ids: list[str], version: str,
) -> dict:
    if tf not in DEFAULT_TIMEFRAMES:
        return {
            "timeframe": tf, "status": "unknown_timeframe",
            "error": f"{tf!r} not in DEFAULT_TIMEFRAMES={DEFAULT_TIMEFRAMES}",
        }
    lb_days = _lookback_days(tf)
    lb_ms = int(lb_days) * 24 * 3600 * 1000
    print(
        f"[task537] tf={tf:>2}  lookback_days={lb_days}  "
        f"coins={len(coin_ids)} ({coin_ids})",
        flush=True,
    )
    t0 = time.monotonic()
    provenance: dict = {}
    try:
        df = await build_labeled_dataset(
            coin_ids, tf, lb_ms, provenance_out=provenance,
        )
    except Exception as exc:  # noqa: BLE001 — one TF failure mustn't stop the rest
        elapsed = round(time.monotonic() - t0, 2)
        print(
            f"  ERROR  tf={tf}  elapsed={elapsed}s  err={exc!r}",
            flush=True,
        )
        return {
            "timeframe": tf, "status": "error",
            "error": str(exc),
            "elapsed_s": elapsed,
            "lookback_days": lb_days,
            "coin_ids_requested": coin_ids,
        }
    elapsed = round(time.monotonic() - t0, 2)
    if df.empty:
        print(
            f"  EMPTY  tf={tf}  elapsed={elapsed}s  (no labelable data)",
            flush=True,
        )
        return {
            "timeframe": tf, "status": "empty",
            "elapsed_s": elapsed,
            "lookback_days": lb_days,
            "coin_ids_requested": coin_ids,
            "provenance": provenance,
        }
    out_path = dataset_path(tf, version)
    df.to_parquet(out_path, index=False)
    coins_emitted = sorted(df["coin_id"].unique().tolist())
    has_volz = "volZScore60" in df.columns
    print(
        f"  WROTE  tf={tf}  rows={len(df):>7}  coins={len(coins_emitted)} "
        f"vol_z={has_volz}  elapsed={elapsed}s  ->  {out_path.name}",
        flush=True,
    )
    return {
        "timeframe": tf, "status": "ok",
        "elapsed_s": elapsed,
        "lookback_days": lb_days,
        "coin_ids_requested": coin_ids,
        "coins_emitted": coins_emitted,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "has_volZScore60": has_volz,
        "output_path": str(out_path.relative_to(ROOT)),
        "provenance": provenance,
    }


async def _amain() -> int:
    started_at = time.time()
    timeframes = _selected_timeframes()
    coins = _selected_coins()
    version = make_version()
    print(
        f"[task537] version={version} timeframes={timeframes} "
        f"coins={coins}",
        flush=True,
    )
    await init_pool()
    rows: list[dict] = []
    try:
        for tf in timeframes:
            row = await _refresh_one(tf, coins, version)
            rows.append(row)
    finally:
        await close_pool()

    summary = {
        "task": 537,
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "version_stamp": version,
        "timeframes": timeframes,
        "coin_ids": coins,
        "results": rows,
        "elapsed_s_total": round(time.time() - started_at, 1),
    }
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (
        f"{summary['generated_at']}-task537-refresh-cached-datasets.json"
    )
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[task537] wrote {out_path.relative_to(ROOT)}")
    print(f"[task537] total elapsed: {summary['elapsed_s_total']}s")
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_empty = sum(1 for r in rows if r.get("status") == "empty")
    # Task #537 — `unknown_timeframe` is an operator misconfiguration
    # (typo'd `ML_REFRESH_TIMEFRAMES`, etc.) and must surface a non-zero
    # exit code alongside `error`. Anything that's neither `ok` nor
    # `empty` is treated as a failure so the next campaign re-runs it
    # rather than silently leaving a tf un-refreshed.
    n_bad = sum(1 for r in rows if r.get("status") not in {"ok", "empty"})
    print(
        f"[task537] ok={n_ok} empty={n_empty} bad={n_bad}",
        flush=True,
    )
    return 0 if n_bad == 0 else 1


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
