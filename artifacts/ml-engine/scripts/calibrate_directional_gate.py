"""Manual calibration study for shared/trading-frictions.json -> quant_brain.decision_thresholds.

Loads the most recent labeled snapshot for the timeframe, runs the
walk-forward OOS predictor ONCE (cached to ``oos_cache_{tf}.parquet``
between runs), then sweeps candidate values of ``min_directional_prob``,
``min_directional_edge``, and ``min_expected_return_pct_factor`` through
the existing live-mirroring simulator. For every combination it records
n_trades, realized PnL, win-rate, expectancy, sharpe, and skip count.

The actual sweep is implemented in
``artifacts/ml-engine/app/training/threshold_calibration.py`` (task #137
folded the logic into a reusable module so the training pipeline can
call the same code automatically). This script is the manual front-end:
print top combinations + write a JSON report for human review.

Usage::

    .pythonlibs/bin/python -m artifacts.ml-engine.scripts.calibrate_directional_gate

Or, equivalently, from the artifact dir::

    ../../.pythonlibs/bin/python scripts/calibrate_directional_gate.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Determinism, same as backtest.run.
os.environ.setdefault("ML_SKIP_OPTUNA", "1")

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACT_ROOT))

import pandas as pd  # noqa: E402

from app.backtest.contract import get_frictions  # noqa: E402
from app.backtest.run import (  # noqa: E402
    build_basket_change_lookup,
    build_tick_streams,
    latest_snapshot_for,
    regime_lookup_for,
)
from app.backtest.simulator import decide_direction  # noqa: E402
from app.backtest.walk_forward_oos import predict_oos_for_dataset  # noqa: E402
from app.training.threshold_calibration import (  # noqa: E402
    DEFAULT_GRID,
    recommend_thresholds,
    run_sweep,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("calibrate")

TIMEFRAME = "5m"


def gates_alignment_summary(oos: pd.DataFrame, fr) -> dict:
    """Task #147 — joint distribution of the SIGN head (classifier edge)
    vs the MAGNITUDE head (|expected_return_pct|) on the cached OOS frame.

    Mirrors the holdout-only summary that the trainer now writes into each
    slice manifest, but uses the leak-free walk-forward sample so we can
    spot disagreement patterns the live `decision_thresholds` will gate
    against. The 2x2 buckets (loud/quiet on each axis) are defined by
    `min_directional_edge` and `min_expected_return_pct` from frictions.

    A high `loud_classifier_quiet_regressor_share` means the classifier is
    "wasting" budget — picking UP/DOWN with edge but the regressor says
    the move is too small to clear the cost floor. The mirror image
    `quiet_classifier_loud_regressor_share` means the regressor is
    wasting budget the other direction. Both are signs the gates may be
    misallocating their budget.
    """
    n = len(oos)
    if n == 0:
        return {"n": 0}
    dir_edge = (oos["p_up"] - oos["p_down"]).abs().to_numpy()
    abs_mag = oos["expected_return_pct"].abs().to_numpy()
    mde = float(fr.min_directional_edge)
    mer = float(fr.min_expected_return_pct)
    cls_loud = dir_edge >= mde
    reg_loud = abs_mag >= mer
    aligned_loud = int((cls_loud & reg_loud).sum())
    aligned_quiet = int((~cls_loud & ~reg_loud).sum())
    loud_cls_quiet_reg = int((cls_loud & ~reg_loud).sum())
    quiet_cls_loud_reg = int((~cls_loud & reg_loud).sum())
    aligned = aligned_loud + aligned_quiet
    total = float(n)

    # 2x2 contingency over (mde, mer) buckets so a human can read the joint
    # distribution at a glance, alongside the share metrics.
    edge_buckets = [(0.0, mde, "edge<mde"), (mde, 1.0, "edge>=mde")]
    mag_buckets = [(0.0, mer, "mag<mer"), (mer, float("inf"), "mag>=mer")]
    contingency: list[dict] = []
    for lo_e, hi_e, e_label in edge_buckets:
        for lo_m, hi_m, m_label in mag_buckets:
            mask = ((dir_edge >= lo_e) & (dir_edge < hi_e)
                    & (abs_mag >= lo_m) & (abs_mag < hi_m))
            contingency.append({
                "edge": e_label, "magnitude": m_label,
                "n": int(mask.sum()),
                "share": float(mask.mean()),
            })
    return {
        "n": int(n),
        "min_directional_edge": mde,
        "min_expected_return_pct": mer,
        "aligned_share": aligned / total,
        "aligned_loud_share": aligned_loud / total,
        "aligned_quiet_share": aligned_quiet / total,
        "loud_classifier_quiet_regressor_share": loud_cls_quiet_reg / total,
        "quiet_classifier_loud_regressor_share": quiet_cls_loud_reg / total,
        "contingency": contingency,
    }


def describe_oos(oos: pd.DataFrame, fr) -> dict:
    """Summarize the distribution of directional signal mass in the OOS frame."""
    n = len(oos)
    p_up = oos["p_up"].to_numpy()
    p_down = oos["p_down"].to_numpy()
    p_stable = oos["p_stable"].to_numpy()
    exp_ret = oos["expected_return_pct"].to_numpy()
    dir_prob = pd.Series([max(u, d) for u, d in zip(p_up, p_down)])
    dir_edge = pd.Series([abs(u - d) for u, d in zip(p_up, p_down)])
    abs_exp_ret = pd.Series(exp_ret).abs()
    cost_floor = fr.round_trip_cost_pct * 100.0
    return {
        "n_rows": n,
        "p_up": {"mean": float(p_up.mean()), "p50": float(pd.Series(p_up).median()), "p95": float(pd.Series(p_up).quantile(0.95))},
        "p_down": {"mean": float(p_down.mean()), "p50": float(pd.Series(p_down).median()), "p95": float(pd.Series(p_down).quantile(0.95))},
        "p_stable": {"mean": float(p_stable.mean()), "p50": float(pd.Series(p_stable).median())},
        "dir_prob": {"p50": float(dir_prob.median()), "p75": float(dir_prob.quantile(0.75)), "p90": float(dir_prob.quantile(0.90)), "p95": float(dir_prob.quantile(0.95))},
        "dir_edge": {"p50": float(dir_edge.median()), "p75": float(dir_edge.quantile(0.75)), "p90": float(dir_edge.quantile(0.90)), "p95": float(dir_edge.quantile(0.95))},
        "abs_exp_ret_pct": {"p50": float(abs_exp_ret.median()), "p75": float(abs_exp_ret.quantile(0.75)), "p90": float(abs_exp_ret.quantile(0.90)), "p95": float(abs_exp_ret.quantile(0.95)), "max": float(abs_exp_ret.max())},
        "round_trip_cost_pct_x100": cost_floor,
        "current_min_exp_ret_floor_pct": fr.min_expected_return_pct,
        "rows_clearing_current_floor": int((abs_exp_ret >= fr.min_expected_return_pct).sum()),
    }


def gate_only_emit_count(oos: pd.DataFrame, fr, mdp: float, mde: float, mer: float) -> int:
    """Count how many OOS rows the directional-gate decision_direction()
    would EMIT a side for, ignoring the downstream paper-trader gates."""
    n_emit = 0
    for _, row in oos.iterrows():
        side, _, _ = decide_direction(
            float(row["p_down"]), float(row["p_stable"]), float(row["p_up"]),
            float(row["expected_return_pct"]),
            fr=fr,
            min_directional_prob=mdp,
            min_directional_edge=mde,
            min_expected_return_pct=mer,
        )
        if side is not None:
            n_emit += 1
    return n_emit


OOS_CACHE = Path(__file__).parent / f"oos_cache_{TIMEFRAME}.parquet"


def main():
    fr = get_frictions()
    snap = latest_snapshot_for(TIMEFRAME)
    if snap is None:
        raise SystemExit(f"No {TIMEFRAME} dataset snapshot found")
    logger.info("snapshot=%s", snap.name)

    df = pd.read_parquet(snap)
    logger.info("loaded %d rows, %d coins", len(df), df["coin_id"].nunique())

    if OOS_CACHE.exists() and os.environ.get("CALIB_REUSE_OOS", "1") == "1":
        oos = pd.read_parquet(OOS_CACHE)
        logger.info("reusing cached OOS at %s (%d rows)", OOS_CACHE, len(oos))
    else:
        oos = predict_oos_for_dataset(df)
        oos.to_parquet(OOS_CACHE)
        logger.info("wrote OOS cache to %s (%d rows)", OOS_CACHE, len(oos))
    logger.info("walk-forward produced %d OOS rows", len(oos))
    if oos.empty:
        raise SystemExit("walk-forward produced 0 OOS rows; can't calibrate")

    summary = describe_oos(oos, fr)
    logger.info("OOS summary: %s", json.dumps(summary, indent=2))

    # Task #147 — gates-alignment diagnostic on the cached OOS sample.
    align = gates_alignment_summary(oos, fr)
    logger.info("Gates alignment (OOS): %s", json.dumps(align, indent=2))
    print("\n# Gates alignment (sign head vs magnitude head, OOS sample)")
    print(f"  n={align['n']}  mde={align['min_directional_edge']:.3f}  mer={align['min_expected_return_pct']:.3f}%")
    print(f"  aligned_share              = {align['aligned_share']*100:6.2f}% "
          f"(loud={align['aligned_loud_share']*100:.2f}% quiet={align['aligned_quiet_share']*100:.2f}%)")
    print(f"  loud_cls / quiet_reg share = {align['loud_classifier_quiet_regressor_share']*100:6.2f}%  "
          f"(classifier confident, regressor below cost floor)")
    print(f"  quiet_cls / loud_reg share = {align['quiet_classifier_loud_regressor_share']*100:6.2f}%  "
          f"(regressor screams, classifier near 50/50)")
    print(f"  {'edge':>10} {'magnitude':>12} {'n':>8} {'share':>8}")
    for c in align.get("contingency", []):
        print(f"  {c['edge']:>10} {c['magnitude']:>12} {c['n']:>8d} {c['share']*100:>7.2f}%")

    streams = build_tick_streams(df)
    basket_changes = build_basket_change_lookup(streams, fr.timeframe_ms(TIMEFRAME))
    regime_fn = regime_lookup_for(basket_changes)

    sweep = run_sweep(
        oos=oos, streams=streams, regime_fn=regime_fn,
        base_fr=fr, timeframe=TIMEFRAME, grid=DEFAULT_GRID,
    )
    rows = [r.to_dict() for r in sweep]
    rows.sort(key=lambda r: (-r["n_trades"], -r["final_pnl_usd"]))

    rec, status = recommend_thresholds(sweep)
    logger.info(
        "recommendation status=%s rec=%s",
        status,
        {"mdp": rec.mdp, "mde": rec.mde, "factor": rec.factor,
         "n_trades": rec.n_trades, "pnl": rec.final_pnl_usd} if rec else None,
    )

    out_path = Path(__file__).parent / "calibration_results.json"
    out_path.write_text(json.dumps({
        "timeframe": TIMEFRAME,
        "snapshot": snap.name,
        "oos_summary": summary,
        "results": rows,
        "recommendation": {
            "status": status,
            "row": rec.to_dict() if rec else None,
        },
    }, indent=2, default=float))
    logger.info("wrote %s", out_path)

    # Print top 20 for human inspection.
    print("\n# Top combinations by trade count, then PnL")
    print(f"{'mdp':>5} {'mde':>5} {'fact':>5} {'mer%':>6} {'n_trades':>9} {'pnl_usd':>10} {'expect':>8} {'win%':>6} {'sharpe':>7}")
    for r in rows[:30]:
        print(f"{r['min_directional_prob']:>5.2f} {r['min_directional_edge']:>5.3f} {r['min_expected_return_pct_factor']:>5.2f} "
              f"{r['min_expected_return_pct']:>6.3f} {r['n_trades']:>9d} {r['final_pnl_usd']:>10.2f} "
              f"{r['expectancy_usd']:>8.3f} {r['win_rate']*100:>6.1f} {r['sharpe_per_trade']:>7.3f}")


if __name__ == "__main__":
    main()
