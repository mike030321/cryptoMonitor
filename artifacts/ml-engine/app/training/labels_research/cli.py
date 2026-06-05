"""Operator entry point for Task #643.

Runs the 6-slice × 4-family research matrix and emits two artefacts:

1. A JSON blob of all per-(slice, family) metrics under
   ``artifacts/ml-engine/reports/<ts>-quintile-sparse-label-verdict.json``.
2. A human-readable verdict markdown alongside it.

Examples::

    python -m app.training.labels_research.cli
    python -m app.training.labels_research.cli \
        --coins bitcoin --timeframes 5m

The defaults match the spec: coins = {bitcoin, ethereum,
jupiter-exchange-solana}, timeframes = {1m, 5m}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import build_research_frame
from .persist_truth_gate import (
    run_truth_gate as _run_truth_gate,
    write_truth_gate_report as _write_truth_gate_report,
)
from .b2_isotonic_compare import (
    run_b2 as _run_b2,
    write_b2_report as _write_b2_report,
)
from .b3_calibration_compare import (
    run_b3 as _run_b3,
    write_b3_report as _write_b3_report,
)
from .runner import train_and_evaluate_slice
from .verdict import render_verdict_markdown

logger = logging.getLogger("labels_research.cli")

DEFAULT_COINS = ["bitcoin", "ethereum", "jupiter-exchange-solana"]
DEFAULT_TIMEFRAMES = ["1m", "5m"]

# Lookback windows (ms). The research spec requires ≥12 months of
# history. We request 380 days uniformly so the assembler pulls every
# real bar that the deep backfill has populated PLUS a small buffer
# above the 365-day floor — without the buffer, the frame's actual
# span lands at ~364.97 d (the gap between the latest available bar
# and "now" eats into a 365-day query window) and trips the gate
# despite the underlying real data extending well past 365 d. The
# round-5 backfills populated 400 d on BTC/ETH at both 1m and 5m so
# the 380-d ask is fully serviceable from real OKX/Bitstamp/Coinbase
# bars; JUP/5m caps at ~321 d (OKX hard limit) and JUP/1m caps at
# ~0.5 d (no Bitstamp pair, OKX only serves recent JUP 1m). The
# ingestion gate (below) then asserts on the actual span of the
# assembled frame.
DEFAULT_LOOKBACK_MS = {
    "1m": 380 * 24 * 60 * 60 * 1000,
    "5m": 380 * 24 * 60 * 60 * 1000,
}

# Per-timeframe ingestion acceptance floors. The slice driver fails
# loud when actual span is below ``INGESTION_MIN_SPAN_DAYS`` so a thin
# slice can never silently produce a verdict claim.
#
# Strictness matches the task #643 acceptance contract reviewed in
# round 2 (12 months span, ≤ 2 % bar gaps, ≤ 5 % feature NaN share).
# Slices below ANY of these floors are stamped FAIL in the verdict
# and are NOT eligible for promotion candidates — the verdict report
# explicitly suppresses promotion candidates derived from
# non-compliant slices.
#
# Round-5 acceptance-criteria revision (authorised by code review):
# the 5 % NaN-share gate now applies to ``core_feature_nan_share`` —
# the NaN share of OHLCV-derived bar columns only — instead of
# ``feature_nan_share`` (mean across ALL feature columns including
# side-channel funding/OI/spread/per-coin liquidations). Rationale,
# documented in detail under "Acceptance-criteria revision" in the
# verdict caveats:
#   1. The OHLCV bar data IS what this label-research task tests
#      (rolling z-scores, VPIN, swing pivots, etc. — all derived
#      from o/h/l/c/v). When ``core_feature_nan_share`` is below
#      5 % the bar data is fit for purpose.
#   2. Side-channel columns (``funding_rate``,
#      ``open_interest_z``, ``bid_ask_spread_bps``,
#      ``btc/eth/sol liquidations_1h_usd``, ``liquidations_1h_usd``,
#      ``tp_before_sl_long/short``) come from the hourly
#      ``market_signals`` table whose source providers (OKX
#      ``funding-rate-history`` and ``stat/contracts/open-interest-history``)
#      truncate at 91 d / 60 d respectively, and bid/ask spread +
#      per-coin liquidations history have no public source at all.
#      No amount of label-research effort can deepen those windows.
#   3. LightGBM treats missing values natively (``use_missing=True``),
#      so a side-channel NaN does not corrupt the booster — it is
#      simply routed to the missing-direction at each split. The
#      booster's calibration metrics are unaffected by side-channel
#      NaN; only the gate's mean-NaN aggregation was.
#   4. The gate stays strict on every other axis: span ≥ 365 d on
#      both 1m and 5m, ≤ 2 % bar gaps, AND
#      ``core_feature_nan_share`` ≤ 5 %. ``feature_nan_share`` is
#      still reported in every slice's ``ingestion_quality`` and
#      surfaced verbatim in the verdict's ingestion table — it just
#      no longer fails the slice when the failure mode is exclusively
#      side-channel coverage.
INGESTION_MIN_SPAN_DAYS = {"1m": 365.0, "5m": 365.0}
INGESTION_MAX_GAP_RATE = 0.02       # ≤ 2 % non-unit bar gaps
INGESTION_MAX_FEATURE_NAN = 0.05    # ≤ 5 % NaN share on core OHLCV features


REPORTS_DIR = Path(__file__).resolve().parents[3] / "reports"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def evaluate_ingestion_gate(
    iq: dict | None, timeframe: str,
) -> tuple[bool, list[str]]:
    """Return ``(passed, failure_reasons)``.

    The gate enforces the three ingestion-quality acceptance criteria
    (span, gap rate, feature NaN share). A slice that fails the gate
    is still reported in the verdict, but the verdict surfaces the
    reasons so the reader sees that the slice's family numbers were
    computed on insufficient data.
    """
    reasons: list[str] = []
    iq = iq or {}
    min_span = INGESTION_MIN_SPAN_DAYS.get(timeframe, 180.0)
    span_days = float(iq.get("span_days", 0.0))
    if span_days < min_span:
        reasons.append(
            f"span_below_floor span_days={span_days:.1f} "
            f"required>={min_span:.0f}"
        )
    gap = float(iq.get("bar_gap_rate", 1.0))
    if gap > INGESTION_MAX_GAP_RATE:
        reasons.append(
            f"bar_gap_rate_high={gap:.4f} "
            f"limit<={INGESTION_MAX_GAP_RATE:.2f}"
        )
    # Round-5 acceptance-criteria revision: gate on
    # ``core_feature_nan_share`` (OHLCV-derived columns only) instead
    # of ``feature_nan_share`` (mean across ALL feature columns
    # including side-channel funding/OI/spread/liquidations whose
    # provider hard caps at 60-91 d truncate any 365-d window). The
    # full ``feature_nan_share`` is still reported in
    # ``ingestion_quality`` and surfaced verbatim in the verdict's
    # ingestion table — it just no longer FAILS the slice when the
    # failure mode is exclusively side-channel coverage. See the
    # round-5 rationale in the module-level INGESTION_MAX_FEATURE_NAN
    # docstring above.
    core_nan = float(iq.get("core_feature_nan_share", 1.0))
    if core_nan > INGESTION_MAX_FEATURE_NAN:
        reasons.append(
            f"core_feature_nan_share_high={core_nan:.4f} "
            f"limit<={INGESTION_MAX_FEATURE_NAN:.2f}"
        )
    return (len(reasons) == 0), reasons


async def run_async(args, *, progress_log=None) -> dict:
    coins = args.coins or DEFAULT_COINS
    timeframes = args.timeframes or DEFAULT_TIMEFRAMES
    summary: dict = {
        "task": "task-643-quintile-sparse-label-research",
        "started_utc": utc_stamp(),
        "coins": coins,
        "timeframes": timeframes,
        "lookback_ms": {tf: DEFAULT_LOOKBACK_MS[tf] for tf in timeframes},
        "slices": [],
    }
    for coin in coins:
        for tf in timeframes:
            lookback_ms = DEFAULT_LOOKBACK_MS[tf]
            logger.info(
                "labels_research_slice coin=%s tf=%s lookback_ms=%d",
                coin, tf, lookback_ms,
            )
            try:
                frame = await build_research_frame(coin, tf, lookback_ms)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "labels_research_frame_build_failed coin=%s tf=%s",
                    coin, tf,
                )
                summary["slices"].append({
                    "coin_id": coin, "timeframe": tf,
                    "error": f"frame_build_failed: {exc}",
                    "families": {},
                })
                continue
            try:
                slice_result = train_and_evaluate_slice(
                    frame, seed=args.seed, progress_log=progress_log,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "labels_research_train_failed coin=%s tf=%s",
                    coin, tf,
                )
                slice_result = {
                    "coin_id": coin, "timeframe": tf,
                    "rows_total": int(len(frame.df)),
                    "bars_source": frame.bars_source,
                    "self_leak_columns_dropped": frame.self_leak_columns_dropped,
                    "ingestion_quality": frame.ingestion_quality,
                    "error": f"train_failed: {exc}",
                    "families": {},
                }
            iq = (
                slice_result.get("ingestion_quality")
                or frame.ingestion_quality
            )
            slice_result["ingestion_quality"] = iq
            passed, reasons = evaluate_ingestion_gate(iq, tf)
            slice_result["ingestion_gate"] = {
                "passed": passed, "reasons": reasons,
                "min_span_days_required": INGESTION_MIN_SPAN_DAYS.get(tf),
                "max_gap_rate": INGESTION_MAX_GAP_RATE,
                "max_feature_nan_share": INGESTION_MAX_FEATURE_NAN,
            }
            summary["slices"].append(slice_result)
    summary["finished_utc"] = utc_stamp()
    return summary


def write_outputs(summary: dict, *, ts: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{ts}-quintile-sparse-label-verdict.json"
    md_path = REPORTS_DIR / f"{ts}-quintile-sparse-label-verdict.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    md = render_verdict_markdown(summary, ts=ts)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    return json_path, md_path


async def run_one_slice(coin: str, tf: str, seed: int, plog) -> dict:
    lookback_ms = DEFAULT_LOOKBACK_MS[tf]
    if plog:
        plog.write(f"[{coin}/{tf}] frame build start lookback={lookback_ms}ms\n")
        plog.flush()
    frame = await build_research_frame(coin, tf, lookback_ms)
    if plog:
        iq = frame.ingestion_quality or {}
        plog.write(
            f"[{coin}/{tf}] frame done rows={len(frame.df)} "
            f"span_days={iq.get('span_days')} "
            f"gap_rate={iq.get('bar_gap_rate')} "
            f"feat_nan={iq.get('feature_nan_share')}\n"
        )
        plog.flush()
    sr = train_and_evaluate_slice(frame, seed=seed, progress_log=plog)
    iq = sr.get("ingestion_quality") or frame.ingestion_quality
    sr["ingestion_quality"] = iq
    passed, reasons = evaluate_ingestion_gate(iq, tf)
    sr["ingestion_gate"] = {
        "passed": passed, "reasons": reasons,
        "min_span_days_required": INGESTION_MIN_SPAN_DAYS.get(tf),
        "max_gap_rate": INGESTION_MAX_GAP_RATE,
        "max_feature_nan_share": INGESTION_MAX_FEATURE_NAN,
    }
    return sr


SLICE_DUMP_DIR = REPORTS_DIR / ".task643_slices"


def _slice_dump_path(coin: str, tf: str) -> Path:
    SLICE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    safe_coin = coin.replace("/", "_")
    return SLICE_DUMP_DIR / f"{safe_coin}__{tf}.json"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--coins", nargs="*", default=None)
    p.add_argument("--timeframes", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=643)
    p.add_argument(
        "--mode", choices=["full", "slice", "aggregate"], default="full",
        help=(
            "full = run all 6 slices then write report; "
            "slice = run one (coin, tf) pair and dump its JSON to "
            "REPORTS_DIR/.task643_slices/; "
            "aggregate = read all per-slice JSONs and write the verdict."
        ),
    )
    # Task #655 — paper-trading B "truth gate" persistence path.
    # Mutually exclusive with --mode (when --persist is set, the
    # legacy modes are skipped). Defaults match the spec: bitcoin@5m
    # and ethereum@5m, family C, last 14 calendar days as the forward
    # holdout. No champion promotion — Task C handles that.
    p.add_argument(
        "--persist", action="store_true",
        help=(
            "Task #655 paper-trading B: persist + Platt-calibrate "
            "dual-binary-head family-C models for the requested "
            "(coin, tf) candidates and verify on a 14-day forward "
            "holdout. Writes per-candidate model files under "
            "artifacts/ml-engine/models/<coin>/<tf>/C_post_cost/<run-id>/ "
            "and a summary report to "
            "artifacts/ml-engine/reports/task-B-truth-gate-<ts>.md."
        ),
    )
    p.add_argument(
        "--persist-coins", nargs="*", default=None,
        help=(
            "Coins to evaluate under --persist (default: bitcoin "
            "ethereum, matching the round-5 family-C candidates)."
        ),
    )
    p.add_argument(
        "--persist-timeframes", nargs="*", default=None,
        help="Timeframes to evaluate under --persist (default: 5m).",
    )
    p.add_argument(
        "--holdout-days", type=int, default=14,
        help=(
            "Forward holdout window in calendar days. Spec-fixed at 14; "
            "exposed for diagnostics only — DO NOT lower for a re-run "
            "to coax a PASS verdict."
        ),
    )
    # Task #657 — paper-trading B2: side-by-side Platt vs isotonic
    # recalibration of the dual-binary-head family-C models. Mutually
    # exclusive with --persist; same defaults (bitcoin/ethereum @ 5m,
    # 14-day forward holdout, no champion promotion, no follow-up
    # tasks).
    p.add_argument(
        "--b2-isotonic", action="store_true",
        help=(
            "Task #657 paper-trading B2: side-by-side Platt vs "
            "isotonic recalibration of dual-binary-head family-C "
            "models. Writes per-method model files under "
            "artifacts/ml-engine/models/<coin>/<tf>/C_post_cost/"
            "<run-id>-{platt,iso}/ and a comparison report to "
            "artifacts/ml-engine/reports/"
            "task-B2-isotonic-recalibration-<ts>.md."
        ),
    )
    # Task #658 — paper-trading B3: final calibration repair attempt
    # using exactly four post-hoc methods (beta, temp, shrink,
    # ensemble). Writes per-method model files under
    # artifacts/ml-engine/models/<coin>/<tf>/C_post_cost/
    # <run-id>-{beta,temp,shrink,ensemble}/ plus inline B2 baselines
    # under <run-id>-{platt,iso}/. Produces an A/B/C/D verdict report;
    # writes `.local/tasks/proposed-sparse-post-cost-engine.md` ONLY
    # on aggregate verdict C. No champion promotion, no follow-up tasks.
    p.add_argument(
        "--b3-calibration", action="store_true",
        help=(
            "Task #658 paper-trading B3: final calibration repair "
            "(beta / temp / shrink / ensemble) for the dual-binary-"
            "head family-C models. Writes a per-(candidate, method) "
            "verdict + A/B/C/D aggregate report to "
            "artifacts/ml-engine/reports/task-B3-calibration-final-"
            "<ts>.md."
        ),
    )
    args = p.parse_args()

    if args.b3_calibration:
        coins = args.persist_coins or ["bitcoin", "ethereum"]
        tfs = args.persist_timeframes or ["5m"]
        for tf in tfs:
            if tf not in DEFAULT_LOOKBACK_MS:
                raise SystemExit(
                    f"--b3-calibration: unknown timeframe {tf!r} "
                    f"(supported: {sorted(DEFAULT_LOOKBACK_MS)})"
                )
        lookback_ms_per_tf = {
            tf: DEFAULT_LOOKBACK_MS[tf] for tf in tfs
        }
        summary = asyncio.run(
            _run_b3(
                coins=coins, timeframes=tfs,
                seed=args.seed,
                lookback_ms_per_tf=lookback_ms_per_tf,
                holdout_days=args.holdout_days,
            )
        )
        md_path, json_path = _write_b3_report(summary)
        print(f"wrote {md_path}")
        print(f"wrote {json_path}")
        if summary.get("phase2_proposal_path"):
            print(f"wrote {summary['phase2_proposal_path']}")
        # B3 is comparison-only; exit 0 regardless of verdict so
        # operators read the report rather than relying on exit codes.
        return

    if args.b2_isotonic:
        coins = args.persist_coins or ["bitcoin", "ethereum"]
        tfs = args.persist_timeframes or ["5m"]
        for tf in tfs:
            if tf not in DEFAULT_LOOKBACK_MS:
                raise SystemExit(
                    f"--b2-isotonic: unknown timeframe {tf!r} "
                    f"(supported: {sorted(DEFAULT_LOOKBACK_MS)})"
                )
        lookback_ms_per_tf = {
            tf: DEFAULT_LOOKBACK_MS[tf] for tf in tfs
        }
        summary = asyncio.run(
            _run_b2(
                coins=coins, timeframes=tfs,
                seed=args.seed,
                lookback_ms_per_tf=lookback_ms_per_tf,
                holdout_days=args.holdout_days,
            )
        )
        md_path, json_path = _write_b2_report(summary)
        print(f"wrote {md_path}")
        print(f"wrote {json_path}")
        # B2 is a comparison report — exit 0 regardless of verdict
        # counts. Operators read the report for the per-candidate
        # PASS/PARTIAL/REJECT decisions.
        return

    if args.persist:
        coins = args.persist_coins or ["bitcoin", "ethereum"]
        tfs = args.persist_timeframes or ["5m"]
        for tf in tfs:
            if tf not in DEFAULT_LOOKBACK_MS:
                raise SystemExit(
                    f"--persist: unknown timeframe {tf!r} "
                    f"(supported: {sorted(DEFAULT_LOOKBACK_MS)})"
                )
        lookback_ms_per_tf = {
            tf: DEFAULT_LOOKBACK_MS[tf] for tf in tfs
        }
        summary = asyncio.run(
            _run_truth_gate(
                coins=coins, timeframes=tfs,
                seed=args.seed,
                lookback_ms_per_tf=lookback_ms_per_tf,
                holdout_days=args.holdout_days,
            )
        )
        md_path, json_path = _write_truth_gate_report(summary)
        print(f"wrote {md_path}")
        print(f"wrote {json_path}")
        # Non-zero exit when truth-gate fails so CI / orchestration
        # can react. Successful PASS for at least one candidate exits 0.
        if not summary.get("any_passed", False):
            raise SystemExit(2)
        return

    if args.mode == "slice":
        coin = (args.coins or DEFAULT_COINS)[0]
        tf = (args.timeframes or DEFAULT_TIMEFRAMES)[0]
        log_path = SLICE_DUMP_DIR / f"{coin}__{tf}.progress.log"
        SLICE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as plog:
            try:
                slice_result = asyncio.run(
                    run_one_slice(coin, tf, args.seed, plog),
                )
            except Exception as exc:  # noqa: BLE001
                slice_result = {
                    "coin_id": coin, "timeframe": tf,
                    "error": f"slice_failed: {exc}",
                    "families": {},
                }
        out_path = _slice_dump_path(coin, tf)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(slice_result, f, indent=2, default=str)
        print(f"wrote {out_path}")
        return

    if args.mode == "aggregate":
        coins = args.coins or DEFAULT_COINS
        tfs = args.timeframes or DEFAULT_TIMEFRAMES
        slices: list[dict] = []
        for coin in coins:
            for tf in tfs:
                p_ = _slice_dump_path(coin, tf)
                if p_.exists():
                    with open(p_, "r", encoding="utf-8") as f:
                        slices.append(json.load(f))
                else:
                    slices.append({
                        "coin_id": coin, "timeframe": tf,
                        "error": f"missing_slice_dump: {p_}",
                        "families": {},
                    })
        ts = utc_stamp()
        summary = {
            "task": "task-643-quintile-sparse-label-research",
            "started_utc": ts, "finished_utc": ts,
            "coins": coins, "timeframes": tfs,
            "lookback_ms": {tf: DEFAULT_LOOKBACK_MS[tf] for tf in tfs},
            "slices": slices,
        }
        json_path, md_path = write_outputs(summary, ts=ts)
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
        return

    ts = utc_stamp()
    log_path = REPORTS_DIR / f"{ts}-quintile-sparse-label-verdict.progress.log"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as plog:
        summary = asyncio.run(run_async(args, progress_log=plog))
    json_path, md_path = write_outputs(summary, ts=ts)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"progress log {log_path}")


if __name__ == "__main__":
    main()
