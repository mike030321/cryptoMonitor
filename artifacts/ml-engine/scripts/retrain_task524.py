"""Task #524 — retrain every (coin, tf) slice against the live
FEATURE_COLUMNS contract (now containing `volZScore60`) and drive the
verification gate over the new versions.

Inputs are the cached labeled-dataset parquet snapshots under
`models/datasets/<tf>_<TS>.parquet` (the same snapshots the diagnostic
harness already trains against). For each timeframe the largest
snapshot is loaded — that file holds rows for every supported coin in
the last full data audit pass. As of Task #537 those snapshots include
the `volZScore60` feature column natively (Task #539 dropped the
in-memory retrofit shim that previously reconstructed it from
`lastPrice`); the loader now fails fast via `_require_volzscore60` if
a stale snapshot ever sneaks in. The frame is then fed through
`train_one_slice` once per coin and once for the pooled head.
`train_one_slice` writes the new model + manifest + calibrators (plus
a regressor head, when the slice has enough non-stable data) into
`models/<coin>/<tf>/<version>/` and updates the `latest` pointer.

After every slice has been retrained we synthesise the same `report`
shape `app.training.train.run_training` produces, hand it to
`build_verification_block`, persist the per-slice verdicts next to
each manifest via `write_verification_verdict`, and write a markdown +
JSON before/after summary to `reports/`. The before-side numbers are
computed by re-running the focused-diagnostic harness against the
unaugmented parquet (the same recipe Task #517 used in
`reports/20260428T120024Z-task517-volZScore60-full-fleet-regression.json`)
so the comparison is strictly feature-level for each (coin, tf).

Run:

    cd artifacts/ml-engine && \
        ML_SKIP_OPTUNA=1 ML_LGB_NUM_BOOST_ROUND=80 \
        ../../.pythonlibs/bin/python -m scripts.retrain_task524

Environment overrides:
    ML_TASK524_TIMEFRAMES=1d,6h,2h,1h   (default)
    ML_TASK524_COIN_FILTER=bonk,celestia (subset; default = all in parquet)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Trainer shortcuts so the full retrain finishes inside one shell budget.
os.environ.setdefault("ML_SKIP_OPTUNA", "1")
os.environ.setdefault("ML_LGB_NUM_BOOST_ROUND", "80")

from app.training import registry as registry_module  # noqa: E402
from app.training.registry import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    POOLED_COIN_ID,
    load_model,
)
from app.training.train import (  # noqa: E402
    CALIBRATION_HOLDOUT_FRACTION,
    DEFAULT_COINS,
    _encode_coin_idx,
    _lgb_params,
    _train_lgb,
    train_one_slice,
)
from app.training.verification import build_verification_block  # noqa: E402
from scripts.diagnostic_482.run_507_focused import (  # noqa: E402
    _require_volzscore60,
)
from scripts.diagnostic_482.run_stage_collapse_diagnostic import (  # noqa: E402
    _latest_pooled_dataset,
    _slice_for,
)


TIMEFRAMES = [
    t.strip() for t in os.environ.get(
        "ML_TASK524_TIMEFRAMES", "1d,6h,2h,1h",
    ).split(",") if t.strip()
]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _coin_filter() -> Optional[set[str]]:
    raw = os.environ.get("ML_TASK524_COIN_FILTER")
    if not raw:
        return None
    out = {c.strip() for c in raw.split(",") if c.strip()}
    return out or None


def _raw_stable_share_for_slice(
    df: pd.DataFrame, vocab: list[str], feature_columns: list[str],
) -> Optional[dict]:
    """Re-run the same diagnostic harness Task #517 used: train a
    booster on the head of the slice, predict on the calibration tail,
    return the raw STABLE-argmax share. `feature_columns` lets the
    caller score the same data with the OLD vs NEW schema for a strict
    feature-level comparison.
    """
    if df.empty or len(df) < 80:
        return None
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df = _encode_coin_idx(df, vocab)
    cal_start = max(1, int(len(df) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
    if cal_start >= len(df) - 5:
        return None
    cols = list(feature_columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return {"status": "missing_columns", "missing": missing}
    X = df[cols]
    y = df["label_3class"].to_numpy().astype(int)
    X_train, y_train = X.iloc[:cal_start], y[:cal_start]
    X_cal, y_cal = X.iloc[cal_start:], y[cal_start:]
    params = _lgb_params(31, 0.1, 5)
    booster, _ = _train_lgb(X_train, y_train, X_cal, y_cal, params)
    raw = booster.predict(X_cal, num_iteration=booster.best_iteration)
    if raw.ndim == 1:
        raw = np.tile([0, 1, 0], (len(X_cal), 1)).astype(float)
    argmax = raw.argmax(axis=1)
    return {
        "n_train": int(len(X_train)),
        "n_cal": int(len(X_cal)),
        "raw_STABLE_share": float((argmax == 1).mean()),
        "raw_DOWN_share": float((argmax == 0).mean()),
        "raw_UP_share": float((argmax == 2).mean()),
        "label_STABLE_share": float((y_cal == 1).mean()),
        "raw_stable_prob_mean": float(raw[:, 1].mean()),
    }


def _raw_stable_share_for_saved_model(
    df: pd.DataFrame, vocab: list[str], coin_id: str, timeframe: str,
) -> Optional[dict]:
    """Load the model just written by `train_one_slice`, slice the same
    calibration tail (last `1 - CALIBRATION_HOLDOUT_FRACTION` of the
    sorted frame), predict, and return the RAW (pre-calibration)
    STABLE-argmax share. This is the production-saved model's
    raw_STABLE_share on its own real holdout, which is what Task #524
    asks the report to surface.
    """
    loaded = load_model(coin_id, timeframe)
    if loaded is None or loaded.booster is None:
        return None
    feature_columns = list(loaded.manifest.feature_names or [])
    if "coin_idx" not in feature_columns:
        return None
    if df.empty or len(df) < 80:
        return None
    sub = df.sort_values("timestamp_ms").reset_index(drop=True)
    sub = _encode_coin_idx(sub, vocab)
    cal_start = max(1, int(len(sub) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
    if cal_start >= len(sub) - 5:
        return None
    missing = [c for c in feature_columns if c not in sub.columns]
    if missing:
        return {"status": "missing_columns", "missing": missing}
    X_cal = sub[feature_columns].iloc[cal_start:]
    y_cal = sub["label_3class"].to_numpy().astype(int)[cal_start:]
    booster = loaded.booster
    raw = booster.predict(X_cal, num_iteration=booster.best_iteration)
    if raw.ndim == 1:
        raw = np.tile([0, 1, 0], (len(X_cal), 1)).astype(float)
    argmax = raw.argmax(axis=1)
    return {
        "n_cal": int(len(X_cal)),
        "raw_STABLE_share": float((argmax == 1).mean()),
        "raw_DOWN_share": float((argmax == 0).mean()),
        "raw_UP_share": float((argmax == 2).mean()),
        "label_STABLE_share": float((y_cal == 1).mean()),
        "raw_stable_prob_mean": float(raw[:, 1].mean()),
        "version": loaded.manifest.version,
    }


def _strip_feature_for_before(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` whose columns mimic the pre-#517 schema:
    drop `volZScore60` so the BEFORE harness sees the exact feature set
    the legacy snapshot was trained on. This is the strict feature-level
    comparison Task #517's regression check used.
    """
    if "volZScore60" in df.columns:
        return df.drop(columns=["volZScore60"])
    return df


def _legacy_feature_columns() -> list[str]:
    """FEATURE_COLUMNS minus `volZScore60` — the schema in production
    immediately before Task #517 added the new column.
    """
    return [c for c in FEATURE_COLUMNS if c != "volZScore60"]


def _dataset_override(tf: str) -> Optional[Path]:
    """Per-tf dataset path override via env. Format:
    `ML_TASK524_DATASET_OVERRIDE=1d=models/datasets/foo.parquet,2h=...`.
    Lets a fill-in run target a specific older parquet that has coverage
    the size-picker (`_latest_pooled_dataset`) misses.
    """
    raw = os.environ.get("ML_TASK524_DATASET_OVERRIDE", "")
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        if k.strip() == tf:
            return Path(v.strip())
    return None


def _process_one_timeframe(
    tf: str, coin_filter: Optional[set[str]], legacy_cols: list[str],
) -> tuple[list[dict], list[dict], dict]:
    """Run BEFORE → RETRAIN → AFTER for a single timeframe and return
    (before_rows, after_rows, tf_report). Loading + processing the
    parquet inside this function lets the caller drop the dataframe
    between timeframes — the 1h snapshot alone is ~80MB which the
    container can't afford to keep around for all four tfs at once.
    """
    import gc

    # Task #540 — refuse to retrain on a snapshot older than the
    # auto-refresher's slowest cadence (24h for 1d/6h) plus a one-tick
    # grace window. The retrain pipeline must never silently consume a
    # week-old cache when the `dataset-refresher` workflow has been
    # broken in the background. ML_TASK524_MAX_AGE_HOURS overrides per
    # campaign run; ML_DATASET_MAX_AGE_HOURS=0 globally disables.
    max_age_h = float(os.environ.get("ML_TASK524_MAX_AGE_HOURS", "36"))
    ds_path = _dataset_override(tf) or _latest_pooled_dataset(
        tf, max_age_hours=max_age_h,
    )
    df_aug = pd.read_parquet(ds_path)
    # Task #539 — the cached parquets have `volZScore60` natively now
    # (Task #537 regenerated them and the #540 auto-refresher keeps
    # them that way). Surface a clear error if a stale snapshot ever
    # sneaks back in instead of silently zero-filling.
    _require_volzscore60(df_aug, ds_path)
    gc.collect()

    full_vocab = sorted(df_aug["coin_id"].unique().tolist())
    coins = [c for c in full_vocab if (coin_filter is None or c in coin_filter)]
    print(
        f"[task524] tf={tf:>2}  dataset={ds_path.name}  "
        f"rows={len(df_aug):,}  coins={len(coins)} ({coins})",
        flush=True,
    )

    # ── Phase A — BEFORE on the legacy feature schema.
    print(f"\n[task524] tf={tf} Phase A: BEFORE raw_STABLE_share (legacy schema)", flush=True)
    df_legacy = _strip_feature_for_before(df_aug)
    before_rows: list[dict] = []
    for coin in coins:
        sub = _slice_for(df_legacy, coin)
        t = time.monotonic()
        try:
            stats = _raw_stable_share_for_slice(sub, full_vocab, legacy_cols)
        except Exception as exc:  # noqa: BLE001
            stats = {"status": "error", "error": str(exc)}
        row = {
            "phase": "before", "coin_id": coin, "timeframe": tf,
            "elapsed_s": round(time.monotonic() - t, 2),
            **(stats or {"status": "no_data"}),
        }
        before_rows.append(row)
        r = row.get("raw_STABLE_share")
        print(
            f"  BEFORE {coin:25} @ {tf:>2}  raw_S={r if r is None else f'{r:.4f}'}",
            flush=True,
        )
    t = time.monotonic()
    try:
        stats = _raw_stable_share_for_slice(df_legacy, full_vocab, legacy_cols)
    except Exception as exc:  # noqa: BLE001
        stats = {"status": "error", "error": str(exc)}
    row = {
        "phase": "before", "coin_id": POOLED_COIN_ID, "timeframe": tf,
        "elapsed_s": round(time.monotonic() - t, 2),
        **(stats or {"status": "no_data"}),
    }
    before_rows.append(row)
    r = row.get("raw_STABLE_share")
    print(
        f"  BEFORE {POOLED_COIN_ID:25} @ {tf:>2}  raw_S={r if r is None else f'{r:.4f}'}",
        flush=True,
    )
    del df_legacy
    gc.collect()

    # ── Phase B — RETRAIN.
    print(f"\n[task524] tf={tf} Phase B: retrain", flush=True)
    per_coin: dict[str, dict] = {}
    for coin in coins:
        sub = _slice_for(df_aug, coin)
        t = time.monotonic()
        try:
            rep = train_one_slice(sub, coin, tf, vocab=[coin])
        except Exception as exc:  # noqa: BLE001
            rep = {"status": "error", "error": str(exc),
                   "coin_id": coin, "timeframe": tf}
        rep["elapsed_s"] = round(time.monotonic() - t, 2)
        per_coin[coin] = rep
        print(
            f"  RETRAIN {coin:25} @ {tf:>2}  status={rep.get('status'):<25}  "
            f"version={rep.get('version','?')}  elapsed={rep['elapsed_s']}s",
            flush=True,
        )
        gc.collect()
    t = time.monotonic()
    try:
        pooled_rep = train_one_slice(df_aug, POOLED_COIN_ID, tf, vocab=full_vocab)
    except Exception as exc:  # noqa: BLE001
        pooled_rep = {"status": "error", "error": str(exc),
                      "coin_id": POOLED_COIN_ID, "timeframe": tf}
    pooled_rep["elapsed_s"] = round(time.monotonic() - t, 2)
    print(
        f"  RETRAIN {POOLED_COIN_ID:25} @ {tf:>2}  status={pooled_rep.get('status'):<25}  "
        f"version={pooled_rep.get('version','?')}  elapsed={pooled_rep['elapsed_s']}s",
        flush=True,
    )
    gc.collect()

    tf_report = {
        "timeframe": tf,
        "status": "trained",
        "n_rows": int(len(df_aug)),
        "per_coin": per_coin,
        "pooled": pooled_rep,
    }

    # ── Phase C — AFTER from the saved booster.
    print(f"\n[task524] tf={tf} Phase C: AFTER raw_STABLE_share (saved)", flush=True)
    after_rows: list[dict] = []
    for coin in coins:
        sub = _slice_for(df_aug, coin)
        try:
            stats = _raw_stable_share_for_saved_model(sub, [coin], coin, tf)
        except Exception as exc:  # noqa: BLE001
            stats = {"status": "error", "error": str(exc)}
        row = {
            "phase": "after", "coin_id": coin, "timeframe": tf,
            **(stats or {"status": "no_data"}),
        }
        after_rows.append(row)
        r = row.get("raw_STABLE_share")
        print(
            f"  AFTER  {coin:25} @ {tf:>2}  raw_S={r if r is None else f'{r:.4f}'}  "
            f"version={row.get('version','?')}",
            flush=True,
        )
    try:
        stats = _raw_stable_share_for_saved_model(df_aug, full_vocab, POOLED_COIN_ID, tf)
    except Exception as exc:  # noqa: BLE001
        stats = {"status": "error", "error": str(exc)}
    row = {
        "phase": "after", "coin_id": POOLED_COIN_ID, "timeframe": tf,
        **(stats or {"status": "no_data"}),
    }
    after_rows.append(row)
    r = row.get("raw_STABLE_share")
    print(
        f"  AFTER  {POOLED_COIN_ID:25} @ {tf:>2}  raw_S={r if r is None else f'{r:.4f}'}  "
        f"version={row.get('version','?')}",
        flush=True,
    )

    tf_report["dataset"] = str(ds_path.relative_to(ROOT))
    tf_report["coins"] = coins
    tf_report["vocab"] = full_vocab

    del df_aug
    gc.collect()
    return before_rows, after_rows, tf_report


def main() -> int:
    started_at = time.time()
    coin_filter = _coin_filter()
    print(f"[task524] timeframes={TIMEFRAMES} coin_filter={coin_filter}", flush=True)

    legacy_cols = _legacy_feature_columns()
    before_rows: list[dict] = []
    after_rows: list[dict] = []
    timeframes_block: dict[str, dict] = {}

    # Persistence checkpoint between timeframes — if a later tf OOMs
    # we still have the earlier work on disk to inspect.
    checkpoint_path = ROOT / "reports" / "_task524_checkpoint.json"
    checkpoint_path.parent.mkdir(exist_ok=True)

    for tf in TIMEFRAMES:
        b_rows, a_rows, tf_rep = _process_one_timeframe(
            tf, coin_filter, legacy_cols,
        )
        before_rows.extend(b_rows)
        after_rows.extend(a_rows)
        timeframes_block[tf] = tf_rep
        # Persist incrementally.
        checkpoint_path.write_text(json.dumps({
            "before_rows": before_rows,
            "after_rows": after_rows,
            "timeframes_block": timeframes_block,
        }, indent=2, default=str))
        print(
            f"\n[task524] tf={tf} done. checkpoint -> "
            f"{checkpoint_path.relative_to(ROOT)}\n",
            flush=True,
        )

    # ── Phase D — verification gate over the synthesised report. We
    # use the union of coins seen across all timeframes as the active
    # set so a coin missing from one tf's parquet still shows up as
    # `untrained` rather than being silently dropped.
    print("\n[task524] Phase D: verification gate", flush=True)
    active_coins: list[str] = []
    seen: set[str] = set()
    for tf in TIMEFRAMES:
        tf_rep = timeframes_block.get(tf) or {}
        for coin in (tf_rep.get("coins") or list((tf_rep.get("per_coin") or {}).keys())):
            if coin not in seen:
                seen.add(coin)
                active_coins.append(coin)
    if coin_filter is None:
        for coin in DEFAULT_COINS:
            if coin not in seen:
                seen.add(coin)
                active_coins.append(coin)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coin_ids": active_coins,
        "timeframes": timeframes_block,
    }
    verification = build_verification_block(report, active_coins)
    print(
        f"  verification.passed={verification['passed']}  "
        f"slices_promoted={verification['counts']['slices_promoted']}  "
        f"slices_below_coinflip={verification['counts']['slices_below_coinflip']}  "
        f"slices_no_lift={verification['counts']['slices_no_lift']}  "
        f"slices_directional_call_regression={verification['counts']['slices_directional_call_regression']}",
        flush=True,
    )

    # Persist per-slice verdicts next to the manifest so the runtime
    # `_resolve_for_predict` path sees the same verdict.
    n_written = n_skipped = 0
    for verdict in verification.get("per_slice", []):
        if not isinstance(verdict, dict):
            continue
        coin = verdict.get("coin")
        tf = verdict.get("timeframe")
        kind = verdict.get("kind")
        if not coin or not tf:
            continue
        tf_rep = timeframes_block.get(tf) or {}
        if kind == "pooled":
            slc_coin = POOLED_COIN_ID
            version = (tf_rep.get("pooled") or {}).get("version")
        else:
            slc_coin = coin
            version = ((tf_rep.get("per_coin") or {}).get(coin) or {}).get("version")
        if not version:
            n_skipped += 1
            continue
        try:
            written = registry_module.write_verification_verdict(
                slc_coin, tf, version, verdict,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  verdict write failed for {coin}/{tf}: {exc}", flush=True)
            written = None
        if written is not None:
            n_written += 1
        else:
            n_skipped += 1
    print(f"  per-slice verdicts: written={n_written} skipped={n_skipped}", flush=True)

    # ── Write reports ──
    ts = _ts()
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"{ts}-task524-volZScore60-retrain.json"
    md_path = out_dir / f"{ts}-task524-volZScore60-retrain.md"

    # Build before/after merged rows for the report.
    after_index = {(r["timeframe"], r["coin_id"]): r for r in after_rows}
    merged_rows = []
    for br in before_rows:
        key = (br["timeframe"], br["coin_id"])
        ar = after_index.get(key, {})
        before_share = br.get("raw_STABLE_share")
        after_share = ar.get("raw_STABLE_share")
        delta = (
            None if (before_share is None or after_share is None)
            else round(after_share - before_share, 4)
        )
        merged_rows.append({
            "coin_id": br["coin_id"], "timeframe": br["timeframe"],
            "before_raw_STABLE_share": (
                None if before_share is None else round(before_share, 4)
            ),
            "after_raw_STABLE_share": (
                None if after_share is None else round(after_share, 4)
            ),
            "delta": delta,
            "after_label_STABLE_share": (
                None if ar.get("label_STABLE_share") is None
                else round(ar["label_STABLE_share"], 4)
            ),
            "after_version": ar.get("version"),
            "before_status": br.get("status"),
            "after_status": ar.get("status"),
        })

    # Locate the dogwifcoin@1d row for the headline assertion.
    target = next(
        (r for r in merged_rows
         if r["coin_id"] == "dogwifcoin" and r["timeframe"] == "1d"),
        None,
    )
    target_after_clears_target = (
        target is not None
        and target.get("after_raw_STABLE_share") is not None
        and target["after_raw_STABLE_share"] >= 0.10
    )

    payload = {
        "task": 524,
        "generated_at": ts,
        "harness": "scripts/retrain_task524.py",
        "feature_added": "volZScore60",
        "datasets": {
            tf: (timeframes_block.get(tf) or {}).get("dataset", "?")
            for tf in TIMEFRAMES
        },
        "active_coins": active_coins,
        "timeframes": TIMEFRAMES,
        "before_after_rows": merged_rows,
        "verification": verification,
        "verification_verdicts_written": {
            "written": n_written, "skipped": n_skipped,
        },
        "dogwifcoin_1d_target": {
            "after_raw_STABLE_share": (
                target.get("after_raw_STABLE_share") if target else None
            ),
            "before_raw_STABLE_share": (
                target.get("before_raw_STABLE_share") if target else None
            ),
            "delta": target.get("delta") if target else None,
            "clears_0_10_target_on_real_holdout": target_after_clears_target,
        },
        "elapsed_s_total": round(time.time() - started_at, 1),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    # Compact markdown
    lines: list[str] = []
    lines.append("# Task #524 — `volZScore60` retrain & verification report")
    lines.append("")
    lines.append(f"_Generated {ts} • elapsed {payload['elapsed_s_total']}s_")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for tf in TIMEFRAMES:
        tf_rep = timeframes_block.get(tf) or {}
        ds = tf_rep.get("dataset", "?")
        n_rows = tf_rep.get("n_rows", "?")
        coins_for_tf = tf_rep.get("coins") or list(
            (tf_rep.get("per_coin") or {}).keys()
        )
        lines.append(
            f"- `{tf}` → `{ds}` ({n_rows} rows, {len(coins_for_tf)} coins)"
        )
    lines.append("")
    lines.append("## Headline — dogwifcoin@1d")
    lines.append("")
    if target is not None:
        lines.append(
            f"- BEFORE raw_STABLE_share = "
            f"**{target['before_raw_STABLE_share']}**  "
            f"(legacy schema, no volZScore60)"
        )
        lines.append(
            f"- AFTER  raw_STABLE_share = "
            f"**{target['after_raw_STABLE_share']}**  "
            f"(saved booster on real holdout, version "
            f"`{target['after_version']}`)"
        )
        lines.append(f"- Δ = **{target['delta']}**")
        lines.append(
            f"- Clears 0.10 floor on real holdout: "
            f"**{target_after_clears_target}**"
        )
    else:
        lines.append("- (dogwifcoin@1d row missing — check inputs)")
    lines.append("")
    lines.append("## Verification gate")
    lines.append("")
    counts = verification["counts"]
    lines.append(f"- passed: **{verification['passed']}**")
    lines.append(f"- slices_promoted: {counts['slices_promoted']}")
    lines.append(f"- slices_no_lift: {counts['slices_no_lift']}")
    lines.append(f"- slices_below_coinflip: {counts['slices_below_coinflip']}")
    lines.append(
        f"- slices_insufficient_sample: {counts['slices_insufficient_sample']}"
    )
    lines.append(
        f"- slices_directional_call_regression: "
        f"{counts['slices_directional_call_regression']}"
    )
    lines.append(f"- coins_with_promotion: {verification['coins_with_promotion']}")
    lines.append(
        f"- coins_without_promotion: {verification['coins_without_promotion']}"
    )
    lines.append(
        f"- per-slice verdicts persisted: written={n_written} skipped={n_skipped}"
    )
    lines.append("")
    lines.append("## Per-slice raw_STABLE_share (BEFORE → AFTER)")
    lines.append("")
    lines.append(
        "| coin | tf | before | after | Δ | label_S | version |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in merged_rows:
        lines.append(
            f"| {r['coin_id']} | {r['timeframe']} | "
            f"{r['before_raw_STABLE_share']} | "
            f"{r['after_raw_STABLE_share']} | "
            f"{r['delta']} | "
            f"{r['after_label_STABLE_share']} | "
            f"`{r['after_version'] or '-'}` |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n")

    print(
        f"\n[task524] wrote {json_path.relative_to(ROOT)} and "
        f"{md_path.relative_to(ROOT)}",
        flush=True,
    )
    print(
        f"[task524] dogwifcoin@1d AFTER raw_STABLE_share = "
        f"{payload['dogwifcoin_1d_target']['after_raw_STABLE_share']}  "
        f"clears_0_10={target_after_clears_target}",
        flush=True,
    )
    print(
        f"[task524] verification.passed = {verification['passed']}  "
        f"total elapsed = {payload['elapsed_s_total']}s",
        flush=True,
    )

    # ── Stricter exit criteria (Task #524 review feedback).
    # This script retrains the timeframes listed in `TIMEFRAMES` (defaults
    # to ML_TASK524_TIMEFRAMES = "1d,6h,2h,1h"). The 5m and 1m timeframes
    # are intentionally OUT OF SCOPE for this script and are delegated to
    # `scripts/retrain_task524_fillin.py` (which uses different cached
    # parquets and runs as a separate workflow). Therefore, the exit
    # criteria check enforces every (coin, tf) in DEFAULT_COINS x
    # TIMEFRAMES (this script's actual coverage) plus __pooled__ x
    # TIMEFRAMES — not DEFAULT_COINS x DEFAULT_TIMEFRAMES.
    # The dogwifcoin@1d headline target is the second hard requirement.
    # The verification gate's pass/fail itself is *not* part of the exit
    # criteria — the 0.50/0.53 DA threshold is hardcoded by Task #401
    # (`MIN_DIRECTIONAL_ACCURACY` in app/training/verification.py) and is
    # historically very hard to clear on these tfs; promotion is a
    # separate workstream (follow-up #536).
    expected_slices = [
        (c, t) for c in DEFAULT_COINS for t in TIMEFRAMES
    ] + [(POOLED_COIN_ID, t) for t in TIMEFRAMES]
    trained_slices = {
        (r["coin_id"], r["timeframe"])
        for r in merged_rows if r.get("after_version")
    }
    missing_slices = [
        (c, t) for (c, t) in expected_slices if (c, t) not in trained_slices
    ]
    headline_ok = bool(target_after_clears_target)
    all_required_trained = not missing_slices
    if missing_slices:
        print(
            f"[task524] FAIL: {len(missing_slices)} required slice(s) "
            f"untrained: {missing_slices}",
            flush=True,
        )
    if not headline_ok:
        print(
            "[task524] FAIL: dogwifcoin@1d AFTER raw_STABLE_share did not "
            "clear the 0.10 floor on the real holdout.",
            flush=True,
        )
    return 0 if (headline_ok and all_required_trained) else 2


if __name__ == "__main__":
    raise SystemExit(main())
