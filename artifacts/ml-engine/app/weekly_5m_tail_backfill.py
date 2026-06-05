"""Task #603 — weekly deep-tail 5m backfill scheduler.

Runs `scripts.backfill_5m_extend.main()` from inside the ml-engine
process on a 7-day cadence. The first invocation fires shortly after
ml-engine startup (after the FastAPI app + DB pool are warm) so a
fresh deploy that has empty / shallow `price_candles` history starts
extending immediately without an operator having to manually launch a
workflow.

Why this lives in ml-engine instead of a standalone Replit workflow:
the project already sits at the 10-workflow ceiling, and the backfill
script's natural lifetime (≤2h then exit) plus the daily top-up's
distinct advisory lock mean running it as an in-process daemon is
strictly less infrastructure than another long-lived workflow. The
backfill script itself takes a Postgres advisory lock with a label
DISTINCT from the daily top-up's, so two replicas (or a manual ad-hoc
run) cannot double-pull the same coin.

Opt-outs that mirror the rest of the ml-engine schedulers:
  - `ML_WEEKLY_5M_TAIL_BACKFILL_ENABLED=0` (hard off)
  - `ML_AUTO_RETRAIN_TEST_DISABLE=1` (test environments — set in
    `tests/conftest.py`)

Tunables (env-overridable so an operator can throttle without code
changes):
  - `ML_WEEKLY_5M_TAIL_BACKFILL_INTERVAL_SECONDS` — default 7 days
  - `ML_WEEKLY_5M_TAIL_BACKFILL_STARTUP_DELAY_SECONDS` — default 120s
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("ml-engine")


ENABLED = os.environ.get("ML_WEEKLY_5M_TAIL_BACKFILL_ENABLED", "1") == "1"

# 7 days. The daily top-up keeps the HEAD of the 5m table fresh; this
# scheduler exists only to fill the deep TAIL when the historical bar
# slips below the 305-day gate (e.g. after a coin gets new history on
# Coinbase / OKX, or after a coin is added to the monitored set).
INTERVAL_SECONDS = int(os.environ.get(
    "ML_WEEKLY_5M_TAIL_BACKFILL_INTERVAL_SECONDS", str(7 * 24 * 3600),
))

# Wait for the FastAPI lifespan to finish + DB pool to warm + the
# daily top-up to settle into its loop before we kick off the first
# tail-backfill. 120s is comfortably more than the daily top-up's 30s
# startup delay so the two daemons never argue over `init_pool` race
# windows on a cold container.
STARTUP_DELAY_SECONDS = int(os.environ.get(
    "ML_WEEKLY_5M_TAIL_BACKFILL_STARTUP_DELAY_SECONDS", "120",
))

# In-process state surfaced by `/ml/admin/weekly-5m-tail-backfill/status`.
# Mirrors the shape of `scheduled_5m_topup.state` so the dashboard tile
# can render both daemons with the same component.
state: dict[str, Any] = {
    "enabled": ENABLED,
    "interval_seconds": INTERVAL_SECONDS,
    "startup_delay_seconds": STARTUP_DELAY_SECONDS,
    "runs_total": 0,
    "last_run_started_at": None,
    "last_run_finished_at": None,
    "last_run_outcome": None,    # "ok" | "error" | "skipped_locked" | "disabled"
    "last_run_elapsed_sec": None,
    "last_error": None,
}

_stop_event = threading.Event()
_thread: threading.Thread | None = None
_run_lock = threading.Lock()  # local re-entrancy guard


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_once_blocking() -> str:
    """Synchronous wrapper around `backfill_5m_extend.main()`. Returns
    the outcome string and updates `state` in place. Best-effort: any
    exception is caught and logged so the scheduler loop never dies."""
    started_iso = _utcnow_iso()
    started = time.monotonic()
    state["last_run_started_at"] = started_iso
    state["last_error"] = None
    # Defer the import so the scheduler module remains cheap to import
    # in test contexts where `ML_AUTO_RETRAIN_TEST_DISABLE=1` has
    # short-circuited `start_scheduler()` before any thread runs.
    from scripts import backfill_5m_extend as bf

    outcome = "ok"
    try:
        asyncio.run(bf.main())
    except Exception as exc:  # noqa: BLE001 — scheduler must never die
        outcome = "error"
        state["last_error"] = str(exc)
        logger.warning(
            "weekly_5m_tail_backfill_failed",
            extra={"error": str(exc)},
        )
    finally:
        state["last_run_finished_at"] = _utcnow_iso()
        state["last_run_elapsed_sec"] = round(time.monotonic() - started, 2)
        state["last_run_outcome"] = outcome
        state["runs_total"] = int(state.get("runs_total") or 0) + 1
    logger.info(
        "weekly_5m_tail_backfill_done",
        extra={
            "outcome": outcome,
            "elapsed_sec": state["last_run_elapsed_sec"],
            "runs_total": state["runs_total"],
        },
    )
    return outcome


def _tick() -> str:
    """Run one backfill iteration under the local re-entrancy lock.
    Returns the outcome string for the loop to log. The script itself
    holds a Postgres advisory lock so cross-replica safety is already
    handled — this lock just stops two ticks of THIS scheduler from
    overlapping inside one process."""
    if not ENABLED:
        state["last_run_outcome"] = "disabled"
        return "disabled"
    if not _run_lock.acquire(blocking=False):
        # Previous iteration still in flight (very unlikely on a 7-day
        # cadence but cheap to guard).
        state["last_run_outcome"] = "skipped_busy"
        return "skipped_busy"
    try:
        return _run_once_blocking()
    finally:
        _run_lock.release()


def _scheduler_loop() -> None:
    # Wait long enough for the FastAPI lifespan + DB pool + daily
    # top-up to settle.
    if _stop_event.wait(STARTUP_DELAY_SECONDS):
        return
    while not _stop_event.is_set():
        try:
            _tick()
        except Exception as exc:  # noqa: BLE001
            state["last_error"] = str(exc)
            logger.warning(
                "weekly_5m_tail_backfill_loop_failed",
                extra={"error": str(exc)},
            )
        if _stop_event.wait(INTERVAL_SECONDS):
            return


def start_scheduler() -> None:
    """Spawn the daemon thread. Honours both the dedicated
    `ML_WEEKLY_5M_TAIL_BACKFILL_ENABLED` switch and the shared
    `ML_AUTO_RETRAIN_TEST_DISABLE` opt-out so the FastAPI test client
    and `pytest` runs never accidentally fire the scheduler."""
    global _thread
    if not ENABLED:
        logger.info("weekly_5m_tail_backfill_disabled")
        return
    if os.environ.get("ML_AUTO_RETRAIN_TEST_DISABLE") == "1":
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(
        target=_scheduler_loop,
        daemon=True,
        name="ml-weekly-5m-tail-backfill",
    )
    _thread.start()
    logger.info(
        "weekly_5m_tail_backfill_started",
        extra={
            "interval_seconds": INTERVAL_SECONDS,
            "startup_delay_seconds": STARTUP_DELAY_SECONDS,
        },
    )


def stop_scheduler() -> None:
    global _thread
    _stop_event.set()
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _thread = None
