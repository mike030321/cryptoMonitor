"""Daily Meta-Brain replay scheduler — auto-warm the supervisory layer
from real post-#444 trading data on a sliding window.

Background (task #490):
  Task #467 built the manual replay pipeline at
  ``artifacts/ml-engine/scripts/replay_meta_brain.py``. It walks
  post-#444 ``prediction_journal`` rows, reconstructs 30-second cycles,
  binds closed ``paper_trades`` to the directives that authorized them,
  and warms trust / regime / episodic memory in a sandbox dir. Today an
  operator has to invoke it by hand; once we have ≥2000 closed trades
  and ≥30 days of post-cutoff history (the script's own gate for
  ``--commit``) the live supervisory layer should carry forward what
  the fleet has actually learned, not start most days near-neutral.

Contract:
  * Once per ``INTERVAL_SECONDS`` (24 h default), invoke
    ``replay_meta_brain.run_replay`` with ``--commit`` over a sliding
    window of ``WINDOW_DAYS`` ending at "now". The start is clamped
    to ``CUTOFF`` so we never cross the post-#444 brain-flip line.
  * The replay script promotes the sandbox into the canonical state
    dir ONLY when its three thresholds (``min-trades``, ``min-days``,
    ``min-regimes``) are all satisfied. Below threshold the run is
    honest about being ``pipeline_validation_only`` and the canonical
    state is untouched.
  * After every successful commit we rotate ``.bak.<ts>`` snapshots
    next to the canonical dir down to the most recent
    ``BACKUP_KEEP`` (3 by default) so they don't grow forever.
  * Failed runs (the script raised, or asyncio crashed) emit a
    structured ``meta_brain_replay_tick_failed`` log line and DO NOT
    touch the canonical state. The replay script's commit path is
    only taken on the success branch, so a partial-failure pre-commit
    can never corrupt canonical either.
  * The most recent run's manifest summary is persisted to a sibling
    ops dir, ``models/meta_brain_replay_status/last_replay.json`` —
    deliberately OUTSIDE the canonical state dir so a non-promoted run
    can never mutate canonical at the path level (the supervisory brain
    only loads files from ``models/meta_brain_state/``). Promoted ticks
    additionally write ``last_committed_replay.json`` in the same
    sibling dir, so ``/ml/meta-brain/last_replay`` can expose a
    dual-pointer view: ``last_run`` (latest attempt, committed or not)
    and ``last_committed_run`` (the manifest the canonical state dir is
    actually warmed from), without re-walking the sandbox tree.

Multi-replica safety:
  Same Postgres advisory-lock pattern as
  ``scheduled_5m_topup``: every replica imports this module and
  computes the same lock key, but only the replica that wins
  ``pg_try_advisory_lock`` actually runs the replay for the tick.
  Losers report ``skipped_locked_other_process`` and wait for the
  next tick. The lock is session-scoped, so a crashed worker can
  never leave it stuck.

Scheduler wiring:
  Started from ``app.main:lifespan`` alongside the auto-retrain,
  fast-loop, and 5m-topup daemons. Honours the same
  ``ML_AUTO_RETRAIN_TEST_DISABLE`` env switch so the FastAPI
  test client never actually fires this loop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db as db_mod  # noqa: E402
from scripts import replay_meta_brain  # noqa: E402

logger = logging.getLogger("ml-engine.meta-brain-replay")

# ── Config (env-overridable) ─────────────────────────────────────────────
ENABLED = os.environ.get("ML_META_BRAIN_REPLAY_ENABLED", "1") == "1"
INTERVAL_SECONDS = int(
    os.environ.get(
        "ML_META_BRAIN_REPLAY_INTERVAL_SECONDS", str(24 * 60 * 60)
    )
)  # nightly by default
WINDOW_DAYS = int(os.environ.get("ML_META_BRAIN_REPLAY_WINDOW_DAYS", "90"))
"""Width of the sliding window in days. Must be ≥ ``--min-days`` (30)
for the replay's commit gate to ever fire — 90 gives a comfortable
margin so a single missed tick doesn't drop us below the threshold."""

STARTUP_DELAY_SECONDS = int(
    os.environ.get("ML_META_BRAIN_REPLAY_STARTUP_DELAY_SECONDS", "120")
)
"""Wait this long after lifespan startup before the first tick. Lets
the DB pool, model registry, and fast-loop warm up first; the replay
is a heavy DB read so we'd rather it not race them."""

BACKUP_KEEP = int(os.environ.get("ML_META_BRAIN_REPLAY_BACKUP_KEEP", "3"))
"""Number of ``meta_brain_state.bak.<ts>`` snapshots to retain after a
successful commit. Older ones are deleted so the rollback footprint
stays bounded."""

# Replay-engine commit thresholds. Surfaced as env switches so an
# operator can tune them without redeploying. Defaults match the
# script's defaults so the replay's own contract doesn't shift.
MIN_TRADES = int(
    os.environ.get(
        "ML_META_BRAIN_REPLAY_MIN_TRADES",
        str(replay_meta_brain.DEFAULT_MIN_TRADES),
    )
)
MIN_DAYS = int(
    os.environ.get(
        "ML_META_BRAIN_REPLAY_MIN_DAYS",
        str(replay_meta_brain.DEFAULT_MIN_DAYS),
    )
)
MIN_REGIMES = int(
    os.environ.get(
        "ML_META_BRAIN_REPLAY_MIN_REGIMES",
        str(replay_meta_brain.DEFAULT_MIN_REGIMES),
    )
)
HOLDOUT_PCT = float(
    os.environ.get(
        "ML_META_BRAIN_REPLAY_HOLDOUT_PCT",
        str(replay_meta_brain.DEFAULT_HOLDOUT_PCT),
    )
)
WINDOW_S = int(
    os.environ.get(
        "ML_META_BRAIN_REPLAY_WINDOW_S",
        str(replay_meta_brain.DEFAULT_WINDOW_S),
    )
)

CANONICAL_STATE_DIR = replay_meta_brain.CANONICAL_STATE_DIR
# Sidecar lives in a sibling ops dir, NOT inside the canonical state
# dir. Two reasons:
#   1. The canonical state dir holds files the supervisory brain
#      actively loads on init (`trust_model.json`, `regime_memory.json`,
#      …). Dropping a non-loaded JSON next to them risks confusing a
#      future loader.
#   2. The spec is "canonical state is untouched on a failed/below-
#      threshold run". Writing a sidecar inside canonical — even an
#      ignored one — would technically violate that contract; keeping
#      it outside makes the contract enforceable at the path level.
STATUS_DIR = CANONICAL_STATE_DIR.parent / "meta_brain_replay_status"
LAST_REPLAY_PATH = STATUS_DIR / "last_replay.json"
"""Latest tick's manifest summary — committed OR below-threshold. Lets
ops see "the daemon ran tonight, here's why it didn't commit yet"."""
LAST_COMMITTED_REPLAY_PATH = STATUS_DIR / "last_committed_replay.json"
"""Latest tick whose manifest was actually promoted into canonical
state (`commit_details.promoted == True`). This is the "latest
manifest" in the strictest sense — it always corresponds to the
contents currently sitting in the canonical state dir, so an operator
can answer "what evidence is the live supervisory brain running on?"
without re-walking the sandbox tree."""

# Postgres advisory-lock label — stable across every ml-engine replica
# so they all contend on the same lock and only one wins per tick.
_LOCK_LABEL = "ml_engine.scheduled_meta_brain_replay"
_LOCK_KEY = db_mod.advisory_lock_key(_LOCK_LABEL)

# Match the timestamp suffix `commit_to_canonical` stamps onto each
# `.bak.<ts>` directory so backup rotation only ever touches the
# directories the replay script itself produced.
_BACKUP_SUFFIX_RE = re.compile(r"\.bak\.\d{8}T\d{6}Z$")


# ── State (read by /ml/admin/meta-brain-replay/status) ────────────────
state: dict = {
    "enabled": ENABLED,
    "interval_seconds": INTERVAL_SECONDS,
    "window_days": WINDOW_DAYS,
    "backup_keep": BACKUP_KEEP,
    "min_trades": MIN_TRADES,
    "min_days": MIN_DAYS,
    "min_regimes": MIN_REGIMES,
    "advisory_lock_label": _LOCK_LABEL,
    "advisory_lock_key": _LOCK_KEY,
    "last_check_at": None,
    "last_attempt_outcome": None,
    "last_finished_at": None,
    "last_error": None,
    "last_run_id": None,
    "last_run_state": None,
    "last_committed": False,
    "last_manifest_path": None,
    "ticks_total": 0,
    "runs_total": 0,
    "commits_total": 0,
    "skips_locked_total": 0,
    "errors_total": 0,
}


_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_run_lock = threading.Lock()


# ── Sliding window resolver ──────────────────────────────────────────────
def _compute_window(
    *, now: Optional[datetime] = None, window_days: int = WINDOW_DAYS,
) -> tuple[datetime, datetime]:
    """Return the ``(start, end)`` window for this tick.

    ``end`` is ``now`` (UTC). ``start`` is ``end - window_days``,
    clamped up to ``CUTOFF`` so the replay script's own
    ``start < CUTOFF`` guard can never trip. In the early days after
    the cutover the effective window is shorter than ``window_days`` —
    that is correct: the journal simply has no rows older than the
    cutoff to replay.
    """
    end = now if now is not None else datetime.now(timezone.utc)
    raw_start = end - timedelta(days=window_days)
    start = max(raw_start, replay_meta_brain.CUTOFF)
    return start, end


def _build_args(
    *, start: datetime, end: datetime, run_id: Optional[str] = None,
) -> argparse.Namespace:
    """Synthesize the argparse namespace ``run_replay`` expects.

    We reuse the script's own argparse defaults / commit thresholds so
    a tweak there propagates here without code changes. ``--commit`` is
    always set on the scheduled path; the replay's threshold gate
    decides whether the commit actually fires.
    """
    return argparse.Namespace(
        start=start.astimezone(timezone.utc).isoformat(),
        end=end.astimezone(timezone.utc).isoformat(),
        window_s=WINDOW_S,
        holdout_pct=HOLDOUT_PCT,
        max_cycles=None,
        min_trades=MIN_TRADES,
        min_days=MIN_DAYS,
        min_regimes=MIN_REGIMES,
        commit=True,
        dry_run=False,
        run_id=run_id,
        sandbox_dir=None,
    )


# ── Backup rotation ──────────────────────────────────────────────────────
def _rotate_backups(*, keep: int = BACKUP_KEEP) -> list[str]:
    """Delete all but the ``keep`` most recent ``.bak.<ts>`` snapshots
    that ``commit_to_canonical`` produces next to the canonical state
    dir.

    Returns the list of pruned directory names (newest-pruned first)
    so callers can record an audit line. Best-effort: a permission or
    OS error logs a warning and returns an empty list rather than
    crashing the tick.
    """
    if keep < 0:
        keep = 0
    parent = CANONICAL_STATE_DIR.parent
    name = CANONICAL_STATE_DIR.name
    if not parent.exists():
        return []
    backups: list[Path] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith(f"{name}.bak."):
            continue
        if not _BACKUP_SUFFIX_RE.search(child.name):
            continue
        backups.append(child)
    if not backups:
        return []
    backups.sort(key=lambda p: p.name, reverse=True)
    to_keep = backups[:keep]
    to_prune = backups[keep:]
    pruned: list[str] = []
    for p in to_prune:
        try:
            shutil.rmtree(p)
            pruned.append(p.name)
        except Exception as exc:  # noqa: BLE001 — best-effort rotation
            logger.warning(
                "meta_brain_replay_backup_prune_failed",
                extra={"path": str(p), "error": str(exc)},
            )
    if pruned:
        logger.info(
            "meta_brain_replay_backup_rotated",
            extra={
                "kept": [p.name for p in to_keep],
                "pruned": pruned,
            },
        )
    return pruned


# ── last_replay sidecar ──────────────────────────────────────────────────
def _summarize_manifest(manifest: dict, *, sandbox_dir: str) -> dict:
    """Project the (large) replay manifest into the small JSON the ops
    endpoint serves. Keeps only the fields callers actually need to
    answer "did the latest replay land, when, and on what evidence?".
    """
    commit_details = manifest.get("commit_details") or {}
    thresholds = commit_details.get("thresholds") or {}
    return {
        "run_id": manifest.get("run_id"),
        "task": manifest.get("task"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "state": manifest.get("state"),
        "commit": bool(manifest.get("commit")),
        "commit_details": {
            "requested": bool(commit_details.get("requested")),
            "promoted": bool(commit_details.get("promoted")),
            "thresholds_satisfied": bool(
                commit_details.get("thresholds_satisfied")
            ),
            "thresholds": thresholds,
            "target": commit_details.get("target"),
            "files": commit_details.get("files"),
            "backup": commit_details.get("backup"),
            "reason": commit_details.get("reason"),
        },
        "data_window": manifest.get("data_window"),
        "row_counts": manifest.get("row_counts"),
        "regimes_observed": manifest.get("regimes_observed"),
        "families_observed": manifest.get("families_observed"),
        "cycles_replayed": manifest.get("cycles_replayed"),
        "trades_attributed": manifest.get("trades_attributed"),
        "trades_unmatched": manifest.get("trades_unmatched"),
        "trades_skipped_holdout": manifest.get("trades_skipped_holdout"),
        "holdout_metrics": manifest.get("holdout_metrics"),
        "forbidden_columns_seen": manifest.get("forbidden_columns_seen"),
        "sandbox_dir": sandbox_dir,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic JSON write via tmp+rename so a partial write can never
    leave the sidecar half-populated for the polling endpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp.replace(path)


def _write_last_replay(summary: dict) -> None:
    """Persist BOTH the latest-run pointer and (when promoted) the
    latest-committed pointer.

    Two pointers because the task contract is "expose the latest
    manifest" but the replay's gates correctly produce many
    below-threshold runs during the early days post-cutover. Ops needs
    BOTH:
      * ``last_replay`` — "did the daemon run tonight at all?" — so a
        long stretch of below-threshold runs is debuggable.
      * ``last_committed_replay`` — "what evidence is the live
        supervisory brain actually running on?" — which by definition
        is the manifest that produced the contents of the canonical
        state dir.
    Best-effort: a write failure logs a warning but never breaks the
    tick (the canonical state itself remains the source of truth).
    """
    try:
        _atomic_write_json(LAST_REPLAY_PATH, summary)
        if summary.get("commit_details", {}).get("promoted"):
            _atomic_write_json(LAST_COMMITTED_REPLAY_PATH, summary)
    except Exception as exc:  # noqa: BLE001 — sidecar write is best-effort
        logger.warning(
            "meta_brain_replay_last_replay_write_failed",
            extra={"error": str(exc)},
        )


def _load_json_or_none(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "meta_brain_replay_sidecar_read_failed",
            extra={"path": str(path), "error": str(exc)},
        )
        return None


def load_last_replay() -> Optional[dict]:
    """Read the persisted latest-run summary for the
    ``/ml/meta-brain/last_replay`` endpoint. Returns ``None`` when no
    replay has ever run on this deployment (or the file is unreadable).
    """
    return _load_json_or_none(LAST_REPLAY_PATH)


def load_last_committed_replay() -> Optional[dict]:
    """Read the persisted latest-committed-run summary. Returns
    ``None`` when no committed run has ever happened on this deployment
    (i.e. the replay's three-gate has never been satisfied).
    """
    return _load_json_or_none(LAST_COMMITTED_REPLAY_PATH)


# ── Tick ─────────────────────────────────────────────────────────────────
async def _run_once_async() -> dict:
    """Acquire the cross-replica advisory lock and, if won, do one
    replay pass. When another ml-engine process holds the lock this
    returns ``{"skipped_locked": True, ...}`` immediately and never
    issues the replay's heavy DB read.
    """
    async with db_mod.try_advisory_lock(_LOCK_KEY) as got:
        if not got:
            return {"skipped_locked": True}
        start, end = _compute_window()
        if end <= start:
            # Brand-new deployment: now is at-or-before CUTOFF. Nothing
            # to replay yet; treat as a clean no-op so the scheduler
            # keeps ticking until journal rows accumulate.
            return {
                "skipped_locked": False,
                "no_window": True,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        args = _build_args(start=start, end=end)
        result = await replay_meta_brain.run_replay(args)
        manifest = result.get("manifest") or {}
        summary = _summarize_manifest(
            manifest, sandbox_dir=str(result.get("sandbox_dir") or ""),
        )
        # Sidecar write + backup rotation are orchestration concerns
        # owned by `_tick` so a single code path handles them whether
        # the replay was driven by the scheduler thread, a test stub,
        # or a future on-demand admin trigger.
        return {
            "skipped_locked": False,
            "no_window": False,
            "summary": summary,
        }


def _run_once_blocking() -> dict:
    """Drive ``_run_once_async`` on a fresh asyncio loop in this thread.

    Mirrors the pattern in ``scheduled_5m_topup._run_once_blocking`` /
    the auto-retrain runner so the FastAPI lifespan pool is never
    bound to the wrong loop. We save AND restore BOTH ``_pool`` and
    ``_pool_loop`` because the script's run_replay also opens its
    own asyncpg connection from ``DATABASE_URL`` (independent of
    db_mod's pool) and our advisory-lock acquire builds a per-thread
    pool here that must not bleed into the lifespan loop.
    """
    saved_pool = db_mod._pool
    saved_pool_loop = db_mod._pool_loop
    db_mod._pool = None
    db_mod._pool_loop = None
    try:
        return asyncio.run(_run_once_async())
    finally:
        db_mod._pool = saved_pool
        db_mod._pool_loop = saved_pool_loop


def _tick(now: Optional[float] = None) -> str:
    """One scheduler iteration. Returns the outcome string also written
    into ``state['last_attempt_outcome']`` so tests can drive the tick
    without spinning up the polling thread.
    """
    now = now if now is not None else time.time()
    state["ticks_total"] = int(state.get("ticks_total") or 0) + 1
    state["last_check_at"] = now

    if not ENABLED:
        state["last_attempt_outcome"] = "disabled"
        return "disabled"

    if not _run_lock.acquire(blocking=False):
        # Previous replay still in flight — extremely unlikely on a
        # 24h cadence, but cheap to guard against an over-eager test.
        state["last_attempt_outcome"] = "skipped_busy"
        return "skipped_busy"

    try:
        result = _run_once_blocking()
        if result.get("skipped_locked"):
            state["last_attempt_outcome"] = "skipped_locked_other_process"
            state["skips_locked_total"] = (
                int(state.get("skips_locked_total") or 0) + 1
            )
            logger.info(
                "meta_brain_replay_skipped_locked_other_process",
                extra={"advisory_lock_key": _LOCK_KEY},
            )
            return "skipped_locked_other_process"
        if result.get("no_window"):
            state["last_attempt_outcome"] = "no_window"
            logger.info(
                "meta_brain_replay_no_window",
                extra={
                    "start": result.get("start"),
                    "end": result.get("end"),
                },
            )
            return "no_window"
        summary = result["summary"]
        # Persist the sidecar BEFORE we touch state counters so a write
        # crash is reported as the tick error (and bumps `errors_total`)
        # rather than silently leaving `last_replay.json` stale.
        _write_last_replay(summary)
        pruned: list[str] = []
        if summary["commit_details"].get("promoted"):
            pruned = _rotate_backups(keep=BACKUP_KEEP)
        state["last_attempt_outcome"] = "ok"
        state["last_error"] = None
        state["last_run_id"] = summary.get("run_id")
        state["last_run_state"] = summary.get("state")
        state["last_committed"] = bool(summary.get("commit"))
        state["last_manifest_path"] = (
            f"{summary.get('sandbox_dir')}/manifest.json"
            if summary.get("sandbox_dir") else None
        )
        state["runs_total"] = int(state.get("runs_total") or 0) + 1
        if state["last_committed"]:
            state["commits_total"] = (
                int(state.get("commits_total") or 0) + 1
            )
            logger.info(
                "meta_brain_replay_committed",
                extra={
                    "run_id": summary.get("run_id"),
                    "trades_attributed": summary.get("trades_attributed"),
                    "cycles_replayed": summary.get("cycles_replayed"),
                    "thresholds": summary["commit_details"].get("thresholds"),
                    "pruned_backups": pruned,
                },
            )
        else:
            # Honest "not yet ready": canonical state is intentionally
            # untouched. NOT a structured error — the script's gate is
            # working as designed.
            logger.info(
                "meta_brain_replay_not_committed",
                extra={
                    "run_id": summary.get("run_id"),
                    "state": summary.get("state"),
                    "thresholds": summary["commit_details"].get("thresholds"),
                    "reason": summary["commit_details"].get("reason"),
                },
            )
        return "ok"
    except Exception as exc:  # noqa: BLE001 — scheduler must never die
        state["last_attempt_outcome"] = "error"
        state["last_error"] = str(exc)
        state["errors_total"] = int(state.get("errors_total") or 0) + 1
        # The structured failure log the task spec requires. Canonical
        # state was never touched on this branch — `commit_to_canonical`
        # is only called inside the success path of `run_replay`.
        logger.warning(
            "meta_brain_replay_tick_failed",
            extra={
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        return "error"
    finally:
        state["last_finished_at"] = time.time()
        _run_lock.release()


def _scheduler_loop() -> None:
    if _stop_event.wait(STARTUP_DELAY_SECONDS):
        return
    while not _stop_event.is_set():
        try:
            _tick()
        except Exception as exc:  # noqa: BLE001
            state["last_error"] = str(exc)
            logger.warning(
                "meta_brain_replay_loop_failed", extra={"error": str(exc)},
            )
        if _stop_event.wait(INTERVAL_SECONDS):
            return


def start_scheduler() -> None:
    """Spawn the daemon thread. Honours ``ML_AUTO_RETRAIN_TEST_DISABLE``
    so the FastAPI test client never actually fires this loop, matching
    the behaviour of every other ml-engine background daemon.
    """
    global _thread
    if not ENABLED:
        logger.info("meta_brain_replay_disabled")
        return
    if os.environ.get("ML_AUTO_RETRAIN_TEST_DISABLE") == "1":
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(
        target=_scheduler_loop,
        daemon=True,
        name="ml-meta-brain-replay",
    )
    _thread.start()
    logger.info(
        "meta_brain_replay_started",
        extra={
            "interval_seconds": INTERVAL_SECONDS,
            "window_days": WINDOW_DAYS,
            "min_trades": MIN_TRADES,
            "min_days": MIN_DAYS,
            "min_regimes": MIN_REGIMES,
        },
    )


def stop_scheduler() -> None:
    global _thread
    _stop_event.set()
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _thread = None
