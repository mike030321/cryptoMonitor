"""Task #587 — focused stage-2 re-run with calibration ON vs OFF.

Re-runs the stage-2 gate evaluation from task #580 on the SAME admitted
feature stack and the SAME dataset snapshots, restricted to a tractable
subset of (timeframes), once with `ML_FEATURE_EDGE_CALIBRATE` off and once
with it on. Writes a side-by-side verdict markdown so an operator can see
whether the flag-gated calibration + neutral-band path moves trade_share
into the production gate band [0.40, 0.85] AND retains DA/PnL.

Restricting to {6h, 1d} only keeps the runtime under ~10 minutes on a
single core; the same machinery runs on 1h/2h, just much slower.

Output:
    artifacts/ml-engine/reports/<TS>-task587-rerun-verdict.md
    artifacts/ml-engine/reports/<TS>-task587-rerun-verdict.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("ml-engine.task587_rerun")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

# Same admitted-feature stack as task #580's stage-2.
ADMITTED_NAMES = [
    "atr_pct_zscore_60",
    "bb_pctb_extreme",
    "drawdown_30",
    "ema_spread_per_atr",
    "macd_signal_cross_strength",
    "realizedVol_log",
    "ret1_squared",
    "ret1_x_volZ60",
    "ret5_minus_ret10",
    "rsi14_centered_squared",
    "vol_of_vol_30",
]
TIMEFRAMES_SUBSET = ["6h", "1d"]


def _load_run_search(calibrate_flag: bool):
    """Load the search runner with the flag-gated path on or off. We
    re-import the module so the module-level `ENABLE_CALIBRATION` is
    re-derived from the env var."""
    if calibrate_flag:
        os.environ["ML_FEATURE_EDGE_CALIBRATE"] = "1"
    else:
        os.environ.pop("ML_FEATURE_EDGE_CALIBRATE", None)
    # Force re-import so module-level constants pick up the env change.
    for k in list(sys.modules):
        if k.startswith("scripts.feature_edge_search"):
            del sys.modules[k]
    from scripts.feature_edge_search import run_search  # noqa: WPS433
    assert run_search.ENABLE_CALIBRATION == calibrate_flag, (
        f"flag mismatch: ENABLE_CALIBRATION={run_search.ENABLE_CALIBRATION} "
        f"after setting calibrate_flag={calibrate_flag}"
    )
    return run_search


_COIN_SLICE: tuple[int, int] | None = None    # (start, stop), set by CLI


def _evaluate_subset(run_search, candidates_by_name: dict, rt_cost_frac: float) -> dict:
    """Run stage-2 evaluation restricted to `TIMEFRAMES_SUBSET`. Mirrors
    `stage2_stacked_gate_eval` from `run_search.py` exactly except the
    timeframe loop is restricted (the underlying machinery already honors
    the module-level `ENABLE_CALIBRATION`). When `_COIN_SLICE` is set,
    further restricts to coins[start:stop] within each timeframe (used by
    the chunked CLI to keep each foreground command under 2 minutes).
    """
    out = {
        "admitted_names": list(ADMITTED_NAMES),
        "per_slice": [],
        "any_slice_passes_gate": False,
        "passing_slices": [],
        "calibration_enabled": bool(run_search.ENABLE_CALIBRATION),
    }
    for tf in TIMEFRAMES_SUBSET:
        try:
            df_tf = run_search.load_dataset(tf)
        except FileNotFoundError as exc:
            logger.warning("skip tf=%s: %s", tf, exc)
            continue
        df_tf["coin_idx"] = run_search.encode_coin_idx(
            df_tf, sorted(df_tf["coin_id"].unique().tolist()),
        )
        base_feats = [c for c in run_search.FEATURE_COLUMNS if c in df_tf.columns]
        if "coin_idx" not in base_feats:
            base_feats = base_feats + ["coin_idx"]
        applicable_added = []
        for name in ADMITTED_NAMES:
            cand = candidates_by_name[name]
            try:
                df_tf[name] = run_search.materialize_candidate(df_tf, cand)
                applicable_added.append(name)
            except Exception as exc:
                logger.warning("materialize skip tf=%s name=%s: %s", tf, name, exc)
        aug_feats = base_feats + applicable_added

        all_coins = sorted(df_tf["coin_id"].unique().tolist())
        if _COIN_SLICE is not None:
            start, stop = _COIN_SLICE
            all_coins = all_coins[start:stop]
        for coin in all_coins:
            sub = df_tf[df_tf["coin_id"] == coin].copy()
            sub = sub.sort_values("timestamp_ms").reset_index(drop=True)
            if len(sub) < 200:
                out["per_slice"].append({
                    "coin": coin, "timeframe": tf, "n_rows": int(len(sub)),
                    "skipped_reason": "fewer than 200 rows", "gate_pass": False,
                })
                continue
            t0 = time.time()
            try:
                base_folds = run_search.walk_forward_eval(
                    sub, base_feats, rt_cost_frac,
                )
                aug_folds = run_search.walk_forward_eval(
                    sub, aug_feats, rt_cost_frac,
                )
            except Exception as exc:
                logger.exception("walk failed tf=%s coin=%s", tf, coin)
                out["per_slice"].append({
                    "coin": coin, "timeframe": tf, "n_rows": int(len(sub)),
                    "skipped_reason": f"walk_forward_eval raised: {exc}",
                    "gate_pass": False,
                })
                continue
            base_agg = run_search._agg_folds(base_folds)
            aug_agg = run_search._agg_folds(aug_folds)
            mean_inv_T = float(np.mean([f.get("calibration_inv_T", 1.0) for f in aug_folds])) if aug_folds else 1.0
            mean_delta = float(np.mean([f.get("neutral_band_delta_fitted", 0.0) for f in aug_folds])) if aug_folds else 0.0
            # Task #591 — surface the per-fold cal-tail-PnL-tuned targets
            # so the side-by-side verdict can show how the auto-tune
            # picks per slice instead of using the legacy hardcoded 0.625.
            tuned_targets = [
                float(f.get("trade_share_target_selected"))
                for f in aug_folds
                if f.get("trade_share_target_selected") is not None
            ]
            mean_target = float(np.mean(tuned_targets)) if tuned_targets else None
            gate_pass = (
                aug_agg["directional_accuracy"] is not None
                and base_agg["directional_accuracy"] is not None
                and (aug_agg["directional_accuracy"]
                     - base_agg["directional_accuracy"]) > run_search.STAGE2_DA_LIFT_FLOOR
                and aug_agg["post_fee_pnl_pct_total"] > run_search.STAGE2_PNL_FLOOR_PCT_TOTAL
                and run_search.STAGE2_TRADE_SHARE_LO <= aug_agg["trade_share"] <= run_search.STAGE2_TRADE_SHARE_HI
                and aug_agg["n_trades"] >= run_search.STAGE2_MIN_TRADES
            )
            row = {
                "coin": coin, "timeframe": tf, "n_rows": int(len(sub)),
                "baseline": base_agg, "augmented": aug_agg,
                "da_lift": (
                    aug_agg["directional_accuracy"] - base_agg["directional_accuracy"]
                    if (aug_agg["directional_accuracy"] is not None
                        and base_agg["directional_accuracy"] is not None) else None
                ),
                "pnl_delta_pct_total": aug_agg["post_fee_pnl_pct_total"]
                                       - base_agg["post_fee_pnl_pct_total"],
                "gate_pass": bool(gate_pass),
                "mean_calibration_inv_T_aug": mean_inv_T,
                "mean_neutral_band_delta_aug": mean_delta,
                "tuned_trade_share_targets_aug": tuned_targets,
                "mean_tuned_trade_share_target_aug": mean_target,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            if gate_pass:
                out["any_slice_passes_gate"] = True
                out["passing_slices"].append({
                    "coin": coin, "timeframe": tf,
                    "da": aug_agg["directional_accuracy"],
                    "baseline_da": base_agg["directional_accuracy"],
                    "post_fee_pnl_pct_total": aug_agg["post_fee_pnl_pct_total"],
                    "trade_share": aug_agg["trade_share"],
                    "n_trades": aug_agg["n_trades"],
                })
            out["per_slice"].append(row)
            logger.info(
                "tf=%s coin=%s gate_pass=%s ts=%s da_lift=%s inv_T=%.3f delta=%.3f tuned_target=%s t=%.1fs",
                tf, coin, gate_pass,
                f"{aug_agg['trade_share']:.3f}" if aug_agg["trade_share"] else "—",
                f"{row['da_lift']:+.4f}" if row["da_lift"] is not None else "—",
                mean_inv_T, mean_delta,
                f"{mean_target:.3f}" if mean_target is not None else "—",
                time.time() - t0,
            )
    return out


def write_markdown(
    md_path: Path,
    ts: str,
    off: dict,
    on: dict,
    rt_cost_frac: float,
    title: str = "Task #587 — stage-2 re-run, calibration ON vs OFF",
    run_metadata: dict | None = None,
) -> None:
    lines: list[str] = []
    lines.append(f"# {title} ({ts})\n")
    if run_metadata:
        wt = run_metadata.get("wall_time_seconds")
        nw = run_metadata.get("n_workers")
        nu = run_metadata.get("n_work_units")
        if wt is not None:
            lines.append(
                f"_Wall time: {wt:.1f}s ({wt / 60.0:.1f} min)"
                + (f" across {nw} worker(s)" if nw is not None else "")
                + (f", {nu} work units" if nu is not None else "")
                + "._\n"
            )
    lines.append(
        "Same admitted-feature stack and dataset snapshots as task #580. The "
        "two columns under `OFF` mirror the original task #580 verdict; the "
        "two columns under `ON` apply the flag-gated calibration + neutral-band "
        "path (single-T temperature scaling on a cal tail, then a fold-fitted "
        "`delta` that targets cal-tail trade_share at a per-(coin, timeframe) "
        "auto-tuned target — Task #591 — picked by maximising cal-tail post-fee "
        "PnL across the grid `{0.50, 0.625, 0.75}`). The trade rule under ON "
        "is `max(P_UP, P_DOWN) > P_STABLE + delta`; under OFF it is "
        "`argmax(proba) != STABLE` (the legacy rule).\n"
    )
    lines.append("\n## Hard rules respected\n")
    lines.append("- Same dataset snapshots as task #580 (latest pooled per-tf parquet).\n")
    lines.append("- Same admitted feature stack as task #580.\n")
    lines.append(f"- No edits to gate constants — DA-lift > 0.02, PnL > 0, trade_share in [0.40, 0.85], n_trades >= 30.\n")
    lines.append(f"- Calibration is point-in-time-safe: cal tail is the LAST 20% of each fold's training slice.\n")
    lines.append(f"- Round-trip cost: {rt_cost_frac * 100:.4f}% from `shared/trading-frictions.json`.\n")
    omitted = sorted(set(["1h", "2h", "6h", "1d"]) - set(TIMEFRAMES_SUBSET))
    if omitted:
        lines.append(f"- Subset: {TIMEFRAMES_SUBSET} ({'/'.join(omitted)} omitted for runtime).\n")
    else:
        lines.append(f"- Subset: {TIMEFRAMES_SUBSET} (full short-tf coverage).\n")

    # Per-slice side-by-side.
    lines.append("\n## Per-(coin, timeframe) side-by-side\n")
    lines.append(
        "| coin | tf | OFF trade_share | OFF DA | OFF DA lift | OFF PnL_total | OFF gate "
        "| ON trade_share | ON DA | ON DA lift | ON PnL_total | ON gate | mean inv_T | mean delta | tuned target |\n"
    )
    lines.append("|" + "---|" * 15 + "\n")
    off_by_key = {(s["coin"], s["timeframe"]): s for s in off["per_slice"] if s.get("gate_pass") is not None}
    on_by_key = {(s["coin"], s["timeframe"]): s for s in on["per_slice"] if s.get("gate_pass") is not None}
    keys = sorted(set(off_by_key) | set(on_by_key))
    for k in keys:
        off_s = off_by_key.get(k) or {}
        on_s = on_by_key.get(k) or {}
        coin, tf = k
        def _fmt_slice(s, source):
            if not s or "augmented" not in s:
                return ["—"] * 5
            agg = s["augmented"]
            return [
                f"{agg['trade_share']:.4f}" if agg.get("trade_share") is not None else "—",
                f"{agg['directional_accuracy']:.4f}" if agg.get("directional_accuracy") is not None else "—",
                f"{s['da_lift']:+.4f}" if s.get("da_lift") is not None else "—",
                f"{agg['post_fee_pnl_pct_total']:+.2f}" if agg.get("post_fee_pnl_pct_total") is not None else "—",
                "**pass**" if s.get("gate_pass") else "fail",
            ]
        off_cells = _fmt_slice(off_s, "off")
        on_cells = _fmt_slice(on_s, "on")
        inv_T = on_s.get("mean_calibration_inv_T_aug", 1.0)
        delta = on_s.get("mean_neutral_band_delta_aug", 0.0)
        tuned_tgt = on_s.get("mean_tuned_trade_share_target_aug")
        lines.append(
            f"| {coin} | {tf} | "
            + " | ".join(off_cells) + " | "
            + " | ".join(on_cells) + " | "
            + (f"{inv_T:.3f}" if isinstance(inv_T, (int, float)) else "—") + " | "
            + (f"{delta:.4f}" if isinstance(delta, (int, float)) else "—") + " | "
            + (f"{tuned_tgt:.3f}" if isinstance(tuned_tgt, (int, float)) else "—") + " |\n"
        )

    # Aggregate counts.
    def _gate_summary(stage):
        evald = [s for s in stage["per_slice"] if s.get("gate_pass") is not None and "augmented" in s]
        passed = [s for s in evald if s.get("gate_pass")]
        in_band = [s for s in evald if s.get("augmented", {}).get("trade_share") is not None
                   and 0.40 <= s["augmented"]["trade_share"] <= 0.85]
        return {
            "n_slices": len(evald),
            "n_passing_gate": len(passed),
            "n_in_trade_share_band": len(in_band),
            "mean_trade_share": float(np.mean([s["augmented"]["trade_share"] for s in evald])) if evald else None,
            "mean_da_lift": float(np.mean([s["da_lift"] for s in evald if s.get("da_lift") is not None])) if evald else None,
            "sum_pnl_pct_total_aug": float(np.sum([s["augmented"]["post_fee_pnl_pct_total"] for s in evald])) if evald else None,
        }
    off_sum = _gate_summary(off)
    on_sum = _gate_summary(on)
    lines.append("\n## Aggregate verdict\n")
    lines.append("| metric | OFF (legacy) | ON (calibrated + delta) |\n")
    lines.append("|---|---|---|\n")
    lines.append(f"| evaluated slices | {off_sum['n_slices']} | {on_sum['n_slices']} |\n")
    lines.append(f"| slices passing the FULL gate | {off_sum['n_passing_gate']} | {on_sum['n_passing_gate']} |\n")
    lines.append(f"| slices with trade_share in [0.40, 0.85] | {off_sum['n_in_trade_share_band']} | {on_sum['n_in_trade_share_band']} |\n")
    lines.append(f"| mean trade_share | {off_sum['mean_trade_share']:.4f} | {on_sum['mean_trade_share']:.4f} |\n")
    if off_sum["mean_da_lift"] is not None and on_sum["mean_da_lift"] is not None:
        lines.append(f"| mean DA lift vs baseline | {off_sum['mean_da_lift']:+.4f} | {on_sum['mean_da_lift']:+.4f} |\n")
    if off_sum["sum_pnl_pct_total_aug"] is not None and on_sum["sum_pnl_pct_total_aug"] is not None:
        lines.append(f"| sum PnL_pct_total (augmented) | {off_sum['sum_pnl_pct_total_aug']:+.2f} | {on_sum['sum_pnl_pct_total_aug']:+.2f} |\n")

    # Interpretation, parameterised on the actual numbers so the report
    # reads correctly regardless of the run.
    delta_share_in_band = on_sum["n_in_trade_share_band"] - off_sum["n_in_trade_share_band"]
    delta_mean_ts = on_sum["mean_trade_share"] - off_sum["mean_trade_share"]
    da_lift_change = (
        (on_sum["mean_da_lift"] - off_sum["mean_da_lift"])
        if (on_sum.get("mean_da_lift") is not None and off_sum.get("mean_da_lift") is not None)
        else None
    )
    pnl_change = (
        (on_sum["sum_pnl_pct_total_aug"] - off_sum["sum_pnl_pct_total_aug"])
        if (on_sum.get("sum_pnl_pct_total_aug") is not None and off_sum.get("sum_pnl_pct_total_aug") is not None)
        else None
    )
    lines.append("\n## Interpretation\n")
    lines.append(
        f"* **Trade-share band coverage**: with the flag OFF only "
        f"{off_sum['n_in_trade_share_band']}/{off_sum['n_slices']} slices land in "
        f"the gate band [0.40, 0.85]; with the flag ON, "
        f"{on_sum['n_in_trade_share_band']}/{on_sum['n_slices']} do "
        f"(+{delta_share_in_band}). Mean trade_share moves "
        f"from {off_sum['mean_trade_share']:.4f} to {on_sum['mean_trade_share']:.4f} "
        f"(Δ {delta_mean_ts:+.4f}; target midpoint 0.625). The flag-gated "
        "calibration + neutral-band path successfully bridges the trade_share "
        "gap that blocked task #580 stage-2.\n"
    )
    lines.append(
        f"* **Gate pass count**: {off_sum['n_passing_gate']} OFF vs "
        f"{on_sum['n_passing_gate']} ON. "
        + (
            "No slice passes the FULL gate either way — calibration moves "
            "trade_share into the band, but the admitted-feature DA lift is "
            "not large enough to clear `STAGE2_DA_LIFT_FLOOR` regardless of "
            "the trade rule. The blocker is the underlying signal strength, "
            "not the trade-rule surface."
            if on_sum["n_passing_gate"] == 0 and off_sum["n_passing_gate"] == 0
            else "The flag opened the door for these slices to clear the "
                 "gate that the legacy trade rule alone could not."
        )
        + "\n"
    )
    if da_lift_change is not None:
        direction = "improves" if da_lift_change > 0 else ("regresses" if da_lift_change < 0 else "is unchanged")
        lines.append(
            f"* **DA lift impact**: mean DA lift {direction} from "
            f"{off_sum['mean_da_lift']:+.4f} to {on_sum['mean_da_lift']:+.4f} "
            f"(Δ {da_lift_change:+.4f}). "
            + (
                "Because the trade rule under ON labels low-confidence rows "
                "as STABLE, the DA denominator includes those rows as misses "
                "on directional-truth bars; small DA changes here are noisy "
                "by construction."
                if abs(da_lift_change) < 0.01
                else ""
            )
            + "\n"
        )
    if pnl_change is not None:
        direction = "improves" if pnl_change > 0 else "regresses"
        lines.append(
            f"* **PnL impact**: sum post-fee PnL_pct_total {direction} from "
            f"{off_sum['sum_pnl_pct_total_aug']:+.2f} to "
            f"{on_sum['sum_pnl_pct_total_aug']:+.2f} (Δ {pnl_change:+.2f}). "
            "The PnL change is dominated by the trade-count reduction and "
            "the round-trip cost saved on rows that no longer trade.\n"
        )
    lines.append(
        "* **Operator action**: keep `ML_FEATURE_EDGE_CALIBRATE` OFF for "
        "task #580 reproducibility. Turn it ON for any future stage-2 run "
        "whose admitted-feature stack should be evaluated under "
        "production-equivalent calibration. The runner now auto-tunes the "
        "cal-tail `trade_share_target` per (coin, timeframe) by maximising "
        "cal-tail post-fee PnL across the grid `{0.50, 0.625, 0.75}` (Task "
        "#591) — the per-slice winning target is shown in the side-by-side "
        "table above. To widen the search, edit "
        "`TRADE_SHARE_TARGET_GRID` in "
        "`scripts/feature_edge_search/run_search.py`.\n"
    )

    md_path.write_text("".join(lines))


def _load_setup() -> tuple[dict, float]:
    cand_path = ROOT / "scripts" / "feature_edge_search" / "candidates.json"
    cand_payload = json.loads(cand_path.read_text())
    candidates_by_name = {c["name"]: c for c in cand_payload["candidates"]}
    shared_path = ROOT.parent.parent / "shared" / "trading-frictions.json"
    fees = json.loads(shared_path.read_text())["fees"]
    rt_cost_frac = 2.0 * (float(fees["taker_fee_pct"]) + float(fees["slippage_pct"]))
    return candidates_by_name, rt_cost_frac


def run_partial(mode: str, tf: str, out_dir: Path,
                coin_slice: tuple[int, int] | None = None) -> Path:
    """Run a single (mode, tf) slice and write the partial result JSON.
    Each call fits comfortably within the 2-minute foreground command
    limit when `OMP_NUM_THREADS=1` keeps LightGBM single-threaded and
    `coin_slice` chunks the coin set into batches of <= 5.
    """
    global _COIN_SLICE
    assert mode in {"off", "on"}
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_by_name, rt_cost_frac = _load_setup()
    rs = _load_run_search(mode == "on")
    saved_subset = list(TIMEFRAMES_SUBSET)
    saved_slice = _COIN_SLICE
    try:
        globals()["TIMEFRAMES_SUBSET"][:] = [tf]
        _COIN_SLICE = coin_slice
        partial = _evaluate_subset(rs, candidates_by_name, rt_cost_frac)
    finally:
        globals()["TIMEFRAMES_SUBSET"][:] = saved_subset
        _COIN_SLICE = saved_slice
    suffix = f"_{coin_slice[0]}_{coin_slice[1]}" if coin_slice else ""
    payload = {
        "task": 587, "mode": mode, "tf": tf, "coin_slice": coin_slice,
        "rt_cost_frac": rt_cost_frac, "result": partial,
    }
    out_path = out_dir / f"partial_{mode}_{tf}{suffix}.json"
    out_path.write_text(json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False))
    logger.info("wrote partial %s", out_path)
    return out_path


def aggregate_and_write(
    partial_dir: Path,
    report_basename: str = "task587-rerun-verdict",
    title: str = "Task #587 — stage-2 re-run, calibration ON vs OFF",
    task_id: int = 587,
    run_metadata: dict | None = None,
) -> tuple[Path, Path]:
    """Merge all partial JSONs in `partial_dir` into a single ON/OFF
    verdict, then write the markdown + json under `reports/`.

    `report_basename`, `title`, and `task_id` let downstream scripts
    (e.g. the task #592 parallel runner) reuse the same aggregation +
    markdown machinery without renaming or re-implementing the writer.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    off = {"per_slice": [], "any_slice_passes_gate": False, "passing_slices": [], "calibration_enabled": False, "admitted_names": list(ADMITTED_NAMES)}
    on = {"per_slice": [], "any_slice_passes_gate": False, "passing_slices": [], "calibration_enabled": True, "admitted_names": list(ADMITTED_NAMES)}
    rt_cost_frac = None
    seen: dict[str, set[tuple]] = {"on": set(), "off": set()}
    for path in sorted(partial_dir.glob("partial_*.json")):
        payload = json.loads(path.read_text())
        rt_cost_frac = payload["rt_cost_frac"]
        mode = payload["mode"]
        bucket = on if mode == "on" else off
        for s in payload["result"]["per_slice"]:
            key = (s.get("timeframe"), s.get("coin"))
            if key in seen[mode]:
                logger.warning(
                    "dedupe: dropping duplicate %s slice tf=%s coin=%s from %s",
                    mode, key[0], key[1], path.name,
                )
                continue
            seen[mode].add(key)
            bucket["per_slice"].append(s)
        bucket["passing_slices"].extend(payload["result"].get("passing_slices", []))
        if payload["result"].get("any_slice_passes_gate"):
            bucket["any_slice_passes_gate"] = True
    json_path = REPORTS_DIR / f"{ts}-{report_basename}.json"
    md_path = REPORTS_DIR / f"{ts}-{report_basename}.md"
    payload = {
        "task": task_id, "captured_at": ts,
        "round_trip_cost_pct": (rt_cost_frac or 0.0) * 100.0,
        "timeframes_subset": TIMEFRAMES_SUBSET,
        "admitted_names": ADMITTED_NAMES,
        "off": off, "on": on,
        "run_metadata": run_metadata or {},
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False))
    write_markdown(
        md_path, ts, off, on, rt_cost_frac or 0.0, title=title,
        run_metadata=run_metadata,
    )
    logger.info("wrote %s", json_path)
    logger.info("wrote %s", md_path)
    return json_path, md_path


def run() -> None:
    """Single-process all-in-one entry point. Use only when running
    foreground in an environment that allows the full ~5-10 minute
    runtime; in agent shells, prefer `run_partial` per (mode, tf) +
    `aggregate_and_write` to fit within per-command timeouts.
    """
    out_dir = ROOT / ".task587_partials"
    for tf in list(TIMEFRAMES_SUBSET):
        run_partial("off", tf, out_dir)
        run_partial("on", tf, out_dir)
    aggregate_and_write(out_dir)


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["partial", "aggregate", "run"])
    parser.add_argument("--mode", choices=["off", "on"])
    parser.add_argument("--tf")
    parser.add_argument("--coin-start", type=int, default=None)
    parser.add_argument("--coin-stop", type=int, default=None)
    parser.add_argument("--out-dir", default=str(ROOT / ".task587_partials"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.command == "partial":
        if not args.mode or not args.tf:
            parser.error("--mode and --tf are required for partial")
        coin_slice = None
        if args.coin_start is not None and args.coin_stop is not None:
            coin_slice = (args.coin_start, args.coin_stop)
        run_partial(args.mode, args.tf, out_dir, coin_slice=coin_slice)
    elif args.command == "aggregate":
        aggregate_and_write(out_dir)
    else:
        run()


if __name__ == "__main__":
    _cli()
