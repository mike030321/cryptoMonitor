"""Task #524 — finish step: load the `_task524_checkpoint.json` left
by `retrain_task524.py` after every (coin, tf) slice was retrained,
build the verification block, persist per-slice verdicts, and emit
the before/after JSON+markdown report.

This exists as a separate entry point so a crash in the verification
phase doesn't force us to re-train every slice. Re-runnable.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training import registry as registry_module  # noqa: E402
from app.training.registry import POOLED_COIN_ID  # noqa: E402
from app.training.train import DEFAULT_COINS  # noqa: E402
from app.training.verification import build_verification_block  # noqa: E402


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    cp = ROOT / "reports" / "_task524_checkpoint.json"
    if not cp.exists():
        print(f"checkpoint not found: {cp}", file=sys.stderr)
        return 1
    data = json.loads(cp.read_text())
    before_rows = data["before_rows"]
    after_rows = data["after_rows"]
    timeframes_block = data["timeframes_block"]
    timeframes = list(timeframes_block.keys())
    print(f"[finish] timeframes={timeframes}", flush=True)

    # ── Active coin set: union across timeframes, then DEFAULT_COINS.
    active_coins: list[str] = []
    seen: set[str] = set()
    for tf in timeframes:
        tf_rep = timeframes_block.get(tf) or {}
        for coin in (tf_rep.get("coins")
                     or list((tf_rep.get("per_coin") or {}).keys())):
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
        f"[finish] verification.passed={verification['passed']}  "
        f"slices_promoted={counts['slices_promoted']}  "
        f"slices_below_coinflip={counts['slices_below_coinflip']}  "
        f"slices_no_lift={counts['slices_no_lift']}  "
        f"slices_directional_call_regression={counts['slices_directional_call_regression']}",
        flush=True,
    )

    # Persist per-slice verdicts.
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
            print(f"[finish] verdict write failed for {coin}/{tf}: {exc}", flush=True)
            written = None
        if written is not None:
            n_written += 1
        else:
            n_skipped += 1
    print(f"[finish] per-slice verdicts: written={n_written} skipped={n_skipped}", flush=True)

    # ── before/after merge for the report.
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

    datasets = {tf: (timeframes_block[tf].get("dataset") or "?")
                for tf in timeframes}

    payload = {
        "task": 524,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness": "scripts/retrain_task524.py + retrain_task524_finish.py",
        "feature_added": "volZScore60",
        "datasets": datasets,
        "active_coins": active_coins,
        "timeframes": timeframes,
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
    }

    ts = _ts()
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"{ts}-task524-volZScore60-retrain.json"
    md_path = out_dir / f"{ts}-task524-volZScore60-retrain.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    # Markdown
    lines: list[str] = []
    lines.append("# Task #524 — `volZScore60` retrain & verification report")
    lines.append("")
    lines.append(
        f"_Generated {ts} • Retrains every (coin, tf) slice against the "
        f"live FEATURE_COLUMNS contract (now containing `volZScore60` from "
        f"Task #517), drives the verification gate, and demonstrates the "
        f"dogwifcoin@1d raw_STABLE_share recovery on a real holdout._"
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for tf in timeframes:
        n_rows = (timeframes_block[tf] or {}).get("n_rows", "?")
        coins = (timeframes_block[tf] or {}).get("coins") or list(
            ((timeframes_block[tf] or {}).get("per_coin") or {}).keys()
        )
        lines.append(
            f"- `{tf}` → `{datasets[tf]}` ({n_rows} rows, {len(coins)} coins)"
        )
    lines.append("")
    lines.append("## Headline — dogwifcoin @ 1d")
    lines.append("")
    if target is not None:
        lines.append(
            f"- BEFORE raw_STABLE_share = "
            f"**{target['before_raw_STABLE_share']}**  "
            f"(legacy schema, no `volZScore60`; same diagnostic harness "
            f"used by Task #517's `volZScore60-full-fleet-regression` report)"
        )
        lines.append(
            f"- AFTER  raw_STABLE_share = "
            f"**{target['after_raw_STABLE_share']}**  "
            f"(saved booster `{target['after_version']}` predicting on the "
            f"real walk-forward calibration tail)"
        )
        lines.append(f"- Δ = **{target['delta']}**")
        lines.append(
            f"- Clears the 0.10 floor on the real holdout: "
            f"**{target_after_clears_target}**"
        )
    else:
        lines.append("- (dogwifcoin @ 1d row missing — check inputs)")
    lines.append("")
    lines.append("## Verification gate")
    lines.append("")
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
    lines.append(
        f"- slices_contract_failed: {counts['slices_contract_failed']}"
    )
    lines.append(
        f"- coins_with_promotion: {verification['coins_with_promotion']}"
    )
    lines.append(
        f"- coins_without_promotion: {verification['coins_without_promotion']}"
    )
    lines.append(
        f"- per-slice verdicts persisted next to manifests: "
        f"written={n_written} skipped={n_skipped}"
    )
    lines.append("")
    lines.append("## Per-slice raw_STABLE_share — BEFORE → AFTER")
    lines.append("")
    lines.append(
        "BEFORE = booster trained on the legacy schema (FEATURE_COLUMNS "
        "minus `volZScore60`) using the focused diagnostic harness — "
        "matches the methodology in "
        "`reports/20260428T120024Z-task517-volZScore60-full-fleet-regression.json`."
    )
    lines.append("")
    lines.append(
        "AFTER = production-saved booster (the manifest `latest` pointer "
        "now points at) predicting on the same calibration tail. This is "
        "the metric Task #524 asks the report to surface."
    )
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
    md_path.write_text("\n".join(lines) + "\n")

    print(
        f"\n[finish] wrote {json_path.relative_to(ROOT)} and "
        f"{md_path.relative_to(ROOT)}",
        flush=True,
    )
    print(
        f"[finish] dogwifcoin@1d AFTER raw_STABLE_share = "
        f"{payload['dogwifcoin_1d_target']['after_raw_STABLE_share']}  "
        f"clears_0_10={target_after_clears_target}",
        flush=True,
    )
    print(
        f"[finish] verification.passed = {verification['passed']}",
        flush=True,
    )
    return 0 if target_after_clears_target else 2


if __name__ == "__main__":
    raise SystemExit(main())
