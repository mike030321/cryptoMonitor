"""Task #524 — fill-in pass for slices the main retrain skipped.

The main retrain (`scripts/retrain_task524.py`) used the size-picker
`_latest_pooled_dataset(tf)` which selected sei-less parquets for
1d/2h/1h, and was scoped to tfs 1d/6h/2h/1h. This fills in:

  • sei-network @ 1d/2h/1h — using older parquets that have sei
  • every default coin + pooled @ 5m/1m — using the largest sei-bearing
    parquets for those tfs

For each retrained slice it persists a per-slice `verification.json`
verdict via `build_verification_block` + `write_verification_verdict`,
matching the main retrain's behaviour.

Outputs an additive report `reports/<TS>-task524-volZScore60-fillin.{json,md}`
and stamps the original report's JSON with a `fillin` block pointing at
the new artifacts so a single grep finds the full picture.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training import registry as registry_module  # noqa: E402
from app.training.registry import POOLED_COIN_ID  # noqa: E402
from app.training.train import (  # noqa: E402
    DEFAULT_COINS, FEATURE_COLUMNS, train_one_slice,
)
from app.training.verification import build_verification_block  # noqa: E402

from scripts.diagnostic_482.run_507_focused import (  # noqa: E402
    _require_volzscore60,
)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# Each entry: (tf, parquet_path_relative_to_ml_engine, coin_filter_or_None_for_all_default+pooled)
FILLIN_PLAN = [
    ("1d", "models/datasets/1d_20260423T043348Z.parquet", {"sei-network"}),
    ("2h", "models/datasets/2h_20260423T030018Z.parquet", {"sei-network"}),
    ("1h", "models/datasets/1h_20260423T014825Z.parquet", {"sei-network"}),
    ("5m", "models/datasets/5m_20260423T002113Z.parquet", None),
    ("1m", "models/datasets/1m_20260422T225300Z.parquet", None),
]


def _process_fillin_tf(
    tf: str, ds_path: Path, coin_filter,
) -> tuple[dict, list[dict]]:
    """Returns (timeframes_block_entry, per_coin_verdict_rows)."""
    df_aug = pd.read_parquet(ds_path)
    # Task #539 — the cached parquets have `volZScore60` natively now
    # (Task #537 regenerated them). Surface a clear error if a stale
    # snapshot ever sneaks back in instead of silently zero-filling.
    _require_volzscore60(df_aug, ds_path)
    gc.collect()

    full_vocab = sorted(df_aug["coin_id"].unique().tolist())
    if coin_filter is None:
        coins = [c for c in DEFAULT_COINS if c in full_vocab]
    else:
        coins = [c for c in full_vocab if c in coin_filter]
    print(
        f"[fillin] tf={tf:>2}  dataset={ds_path.name}  "
        f"rows={len(df_aug):,}  coins={coins}",
        flush=True,
    )

    per_coin: dict[str, dict] = {}
    for coin in coins:
        sub = df_aug[df_aug["coin_id"] == coin].copy()
        t = time.monotonic()
        try:
            rep = train_one_slice(sub, coin, tf, vocab=[coin])
        except Exception as exc:  # noqa: BLE001
            rep = {
                "status": "error", "error": str(exc),
                "coin_id": coin, "timeframe": tf,
            }
        rep["elapsed_s"] = round(time.monotonic() - t, 2)
        per_coin[coin] = rep
        print(
            f"  RETRAIN {coin:25} @ {tf:>2}  status={rep.get('status'):<25}  "
            f"version={rep.get('version','?')}  elapsed={rep['elapsed_s']}s",
            flush=True,
        )
        gc.collect()

    pooled_rep = None
    if coin_filter is None:
        t = time.monotonic()
        try:
            pooled_rep = train_one_slice(
                df_aug, POOLED_COIN_ID, tf, vocab=full_vocab,
            )
        except Exception as exc:  # noqa: BLE001
            pooled_rep = {
                "status": "error", "error": str(exc),
                "coin_id": POOLED_COIN_ID, "timeframe": tf,
            }
        pooled_rep["elapsed_s"] = round(time.monotonic() - t, 2)
        per_coin[POOLED_COIN_ID] = pooled_rep
        print(
            f"  RETRAIN {POOLED_COIN_ID:25} @ {tf:>2}  status={pooled_rep.get('status'):<25}  "
            f"version={pooled_rep.get('version','?')}  elapsed={pooled_rep['elapsed_s']}s",
            flush=True,
        )

    tf_block = {
        "dataset": str(ds_path),
        "n_rows": int(len(df_aug)),
        "coins": coins + ([POOLED_COIN_ID] if pooled_rep else []),
        "per_coin": per_coin,
    }
    if pooled_rep:
        tf_block["pooled"] = pooled_rep

    return tf_block, list(per_coin.values())


def main() -> int:
    started = time.time()
    ts = _ts()
    print(f"[fillin] started {ts}", flush=True)

    timeframes_block: dict[str, dict] = {}
    all_versions: list[dict] = []

    for tf, ds_rel, coin_filter in FILLIN_PLAN:
        ds_path = ROOT / ds_rel
        if not ds_path.exists():
            print(f"[fillin] SKIP tf={tf}: missing dataset {ds_path}", flush=True)
            continue
        tf_block, slice_reports = _process_fillin_tf(tf, ds_path, coin_filter)
        timeframes_block[tf] = tf_block
        for s in slice_reports:
            if s.get("version"):
                all_versions.append({
                    "coin": s.get("coin_id"),
                    "tf": tf,
                    "version": s["version"],
                    "status": s.get("status"),
                })

    # ── Verification + verdict persistence.
    active_coins: list[str] = []
    seen: set[str] = set()
    for tf, tf_block in timeframes_block.items():
        for coin in (tf_block.get("coins") or []):
            if coin == POOLED_COIN_ID:
                continue
            if coin not in seen:
                seen.add(coin)
                active_coins.append(coin)
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
    counts = verification["counts"]
    print(
        f"[fillin] verification.passed={verification['passed']}  "
        f"slices_promoted={counts['slices_promoted']}  "
        f"slices_below_coinflip={counts['slices_below_coinflip']}  "
        f"slices_directional_call_regression={counts.get('slices_directional_call_regression')}",
        flush=True,
    )

    n_written = n_skipped = 0
    for verdict in verification.get("per_slice", []):
        if not isinstance(verdict, dict):
            continue
        coin = verdict.get("coin")
        tf = verdict.get("timeframe")
        if not coin or not tf:
            n_skipped += 1
            continue
        # Only persist verdicts for slices we actually retrained in this pass.
        tf_block = timeframes_block.get(tf) or {}
        if coin not in (tf_block.get("coins") or []):
            n_skipped += 1
            continue
        version = ((tf_block.get("per_coin") or {}).get(coin) or {}).get("version")
        if not version:
            n_skipped += 1
            continue
        path = registry_module.write_verification_verdict(
            coin, tf, version, verdict,
        )
        if path is None:
            n_skipped += 1
        else:
            n_written += 1
    print(f"[fillin] verdicts written={n_written} skipped={n_skipped}", flush=True)

    # ── Persist report.
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{ts}-task524-volZScore60-fillin.json"
    md_path = out_dir / f"{ts}-task524-volZScore60-fillin.md"

    payload = {
        "task": 524,
        "kind": "fillin",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness": "scripts/retrain_task524_fillin.py",
        "fillin_plan": [
            {"tf": tf, "dataset": ds_rel, "coin_filter": sorted(list(cf or set()))}
            for tf, ds_rel, cf in FILLIN_PLAN
        ],
        "active_coins": active_coins,
        "timeframes": list(timeframes_block.keys()),
        "versions": all_versions,
        "verification": verification,
        "verification_verdicts_written": {
            "written": n_written, "skipped": n_skipped,
        },
        "elapsed_s_total": round(time.time() - started, 1),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    lines: list[str] = []
    lines.append("# Task #524 — fill-in retrain & verification report")
    lines.append("")
    lines.append(f"_Generated {ts} • elapsed {payload['elapsed_s_total']}s_")
    lines.append("")
    lines.append("Fills in the slices the main retrain skipped:")
    lines.append("- `sei-network` @ 1d/2h/1h (the size-picker chose sei-less parquets)")
    lines.append("- every default coin + `__pooled__` @ 5m/1m (not in the main retrain's tf list)")
    lines.append("")
    lines.append("## Versions trained")
    lines.append("")
    lines.append("| coin | tf | version | status |")
    lines.append("|---|---|---|---|")
    for v in all_versions:
        lines.append(f"| {v['coin']} | {v['tf']} | `{v['version']}` | {v['status']} |")
    lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append(f"- passed: **{verification['passed']}**")
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- per-slice verdicts persisted: written={n_written} skipped={n_skipped}")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"[fillin] wrote {json_path} and {md_path}", flush=True)

    # Stamp the latest task-524 main retrain report (discovered dynamically;
    # naming convention is `<TS>-task524-volZScore60-retrain.json`, written
    # by retrain_task524.py / retrain_task524_finish.py). If multiple exist,
    # pick the most recent by mtime.
    main_candidates = sorted(
        (ROOT / "reports").glob("*-task524-volZScore60-retrain.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    main_report = main_candidates[0] if main_candidates else None
    if main_report and main_report.exists():
        try:
            mr = json.loads(main_report.read_text())
            mr["fillin"] = {
                "report_json": str(json_path.relative_to(ROOT)),
                "report_md": str(md_path.relative_to(ROOT)),
                "versions_trained": len(all_versions),
                "verdicts_written": n_written,
            }
            main_report.write_text(json.dumps(mr, indent=2, default=str))
            print(f"[fillin] stamped main report {main_report.name} with fillin pointer", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fillin] WARN could not stamp main report: {exc}", flush=True)
    else:
        print("[fillin] no main retrain report found to stamp", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
