"""Task #592 — parallel calibration ON/OFF stage-2 runner for short timeframes.

Re-runs the same admitted-feature stack and dataset snapshots as task #587 on
the 1h and 2h pooled datasets, parallelized across (mode, timeframe, coin)
work units so the full ON/OFF verdict completes well under an hour. Single-
threaded chunked execution from `task587_rerun_stage2.py` left 1h (~77k rows)
and 2h (~38k rows) at multi-hour foreground runtimes that no operator would
tolerate; this script reuses the same evaluation machinery and only changes
the orchestration to a process pool.

Each worker pins LightGBM + BLAS to a single thread (`OMP_NUM_THREADS=1`) so
the pool can hand each unit a dedicated CPU without oversubscription, and the
parent clears stale partials before each run so a previous half-run cannot
bleed into this verdict.

Output:
    artifacts/ml-engine/reports/<TS>-task592-1h2h-stage2-verdict.md
    artifacts/ml-engine/reports/<TS>-task592-1h2h-stage2-verdict.json
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("ml-engine.task592")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_OUT_DIR = ROOT / ".task592_partials"
DEFAULT_TFS = ("1h", "2h")
# Leave one core for the orchestrator + OS housekeeping. Capped at 6 because
# our datasets only have 10 coins per timeframe — beyond that, additional
# workers sit idle waiting for the last few units.
DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 2) - 1))

REPORT_BASENAME = "task592-1h2h-stage2-verdict"
REPORT_TITLE = "Task #592 — stage-2 calibration ON vs OFF, 1h + 2h (parallel)"


def _pin_single_thread() -> None:
    """Force every numerical library this script touches to use a single
    OS thread, so a process-pool worker cannot oversubscribe its CPU.
    Set in both the parent (before any heavy imports) and each worker.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"


def _list_coins(tf: str) -> list[str]:
    """Returns the sorted coin vocabulary for `tf`'s latest snapshot.
    Loaded in the parent so we know how many work units to queue and so
    we can map each `coin_idx` back to a coin id for logging.
    """
    from scripts.feature_edge_search import run_search as rs  # noqa: WPS433
    df = rs.load_dataset(tf)
    return sorted(df["coin_id"].unique().tolist())


def _run_unit(mode: str, tf: str, coin_idx: int, out_dir_str: str) -> str:
    """Worker entry point — runs a single (mode, tf, coins[idx:idx+1])
    slice and writes its partial JSON. Returns the partial's path.
    Imports happen inside the worker so each spawn re-derives the
    `ENABLE_CALIBRATION` module-level constant from its own env var.
    """
    from scripts import task587_rerun_stage2 as t587  # noqa: WPS433
    out_dir = Path(out_dir_str)
    p = t587.run_partial(
        mode, tf, out_dir, coin_slice=(coin_idx, coin_idx + 1),
    )
    return str(p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tfs", nargs="+", default=list(DEFAULT_TFS),
        help="Timeframes to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Process-pool size (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help="Directory for per-(mode, tf, coin) partial JSONs.",
    )
    parser.add_argument(
        "--coin", action="append", default=None,
        help="Restrict to specific coin id(s). Repeatable.",
    )
    parser.add_argument(
        "--keep-partials", action="store_true",
        help="Keep stale partial JSONs in --out-dir instead of clearing them.",
    )
    args = parser.parse_args(argv)

    _pin_single_thread()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_partials:
        for p in out_dir.glob("partial_*.json"):
            p.unlink()

    # Build the queue of (mode, tf, coin_idx) work units. The coin_idx is
    # the position in the FULL sorted coin vocabulary for `tf`, not the
    # filtered subset — this matters because `run_partial`'s coin_slice
    # is applied to `sorted(df_tf["coin_id"].unique())` inside the worker,
    # which always sees the full dataset's coin list.
    tasks: list[tuple[str, str, int, str]] = []
    queued_coins_by_tf: dict[str, list[str]] = {}
    for tf in args.tfs:
        full_coins = _list_coins(tf)
        if args.coin:
            allowed = set(args.coin)
            unknown = allowed - set(full_coins)
            if unknown:
                logger.error(
                    "tf=%s: --coin filter mentions unknown coin(s): %s",
                    tf, sorted(unknown),
                )
                return 2
            selected = [(i, c) for i, c in enumerate(full_coins) if c in allowed]
        else:
            selected = list(enumerate(full_coins))
        queued_coins_by_tf[tf] = [c for _i, c in selected]
        for mode in ("off", "on"):
            for full_idx, coin in selected:
                tasks.append((mode, tf, full_idx, coin))

    if not tasks:
        logger.error("no work units queued — check --tfs and --coin filters")
        return 2

    logger.info(
        "queued %d work units across %d worker(s); tfs=%s coins=%s",
        len(tasks), args.workers, args.tfs,
        {tf: len(c) for tf, c in queued_coins_by_tf.items()},
    )

    started = time.time()
    # `spawn` so each worker boots a fresh interpreter and re-derives
    # `ENABLE_CALIBRATION` from its env var (the legacy chunked CLI
    # relied on a fresh module import per (mode) flip).
    ctx = get_context("spawn")
    failures: list[tuple[str, str, int, str, str]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=_pin_single_thread,
    ) as pool:
        future_to_unit = {
            pool.submit(_run_unit, m, t, i, str(out_dir)): (m, t, i, c)
            for (m, t, i, c) in tasks
        }
        done = 0
        for fut in as_completed(future_to_unit):
            mode, tf, idx, coin = future_to_unit[fut]
            try:
                partial_path = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((mode, tf, idx, coin, str(exc)))
                logger.exception(
                    "worker failed mode=%s tf=%s coin=%s", mode, tf, coin,
                )
                continue
            done += 1
            logger.info(
                "[%d/%d] mode=%s tf=%s coin=%s -> %s (%.1fs elapsed)",
                done, len(tasks), mode, tf, coin,
                Path(partial_path).name, time.time() - started,
            )

    elapsed = time.time() - started
    logger.info("workers complete in %.1fs (%d failures)", elapsed, len(failures))
    if failures:
        # Refuse to write a verdict from a partially-failed run — the
        # operator should re-run the affected slices rather than read
        # an aggregate that silently dropped them.
        for mode, tf, idx, coin, exc in failures:
            logger.error("FAILED mode=%s tf=%s coin=%s: %s", mode, tf, coin, exc)
        return 3

    # Sanity-check that exactly the expected number of partials landed on
    # disk before we aggregate. This catches the "worker silently exited
    # without writing its partial" failure mode that as_completed cannot
    # detect (e.g. SIGKILL by an OOM killer between fit and write).
    expected_partials = len(tasks)
    actual_partials = sum(1 for _ in out_dir.glob("partial_*.json"))
    if actual_partials != expected_partials:
        logger.error(
            "partial-count mismatch: expected %d, found %d in %s — refusing "
            "to write a verdict from a half-complete run",
            expected_partials, actual_partials, out_dir,
        )
        return 4

    from scripts import task587_rerun_stage2 as t587  # noqa: WPS433
    # Make the verdict's "Subset" line and per-slice table reflect what
    # this script actually ran (not the task #587 default of {6h, 1d}).
    t587.TIMEFRAMES_SUBSET[:] = list(args.tfs)
    json_path, md_path = t587.aggregate_and_write(
        out_dir,
        report_basename=REPORT_BASENAME,
        title=REPORT_TITLE,
        task_id=592,
        run_metadata={
            "wall_time_seconds": round(elapsed, 1),
            "n_workers": args.workers,
            "n_work_units": expected_partials,
            "tfs": list(args.tfs),
            "coins_evaluated": queued_coins_by_tf,
        },
    )
    logger.info("wrote %s", md_path)
    logger.info("wrote %s", json_path)
    logger.info("total wall time %.1fs", time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
