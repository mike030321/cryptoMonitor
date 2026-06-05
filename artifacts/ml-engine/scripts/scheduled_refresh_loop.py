"""Task #540 / #599 — scheduled cron-style refresher for the cached training datasets.

Task #599 (this version) auto-runs the 1h+2h calibration ON/OFF verdict
(`scripts/task592_parallel_stage2.py`) after every successful 1h or 2h
refresh — the parallel runner completes in ~70s on this host so it's
cheap enough to do every tick. The result lands in
`_freshness_status.json` under the new `calibration_verdict.short_tf`
key (latest attempt + summary + report paths) and any failure also
appends to `_freshness_alerts.jsonl` so the dashboard alert log
surfaces it. Disable with ``ML_REFRESH_AUTO_CALIBRATION_VERDICT=0``;
override the subprocess timeout with
``ML_REFRESH_CALIBRATION_VERDICT_TIMEOUT_S=<seconds>`` (default 600).

Task #537 added `scripts/refresh_cached_datasets.py`, which regenerates a
fresh `<tf>_<TS>.parquet` snapshot for every supported timeframe. That
fixed the immediate staleness, but nothing scheduled it — so a week
later the cache would drift again and a human would have to re-run #537
by hand.

This module is the scheduler. It runs forever, ticks every
`ML_REFRESH_TICK_SECONDS` (default 1800s = 30 min), and on each tick
re-refreshes any timeframe whose newest cached parquet is older than
the documented per-tf cadence:

    1d, 6h    : 24 h     (slow-moving, daily refresh is plenty)
    2h, 1h    :  6 h     (mid-cadence, refresh four times a day)
    5m, 1m    :  6 h     (high-volume but capped to four refreshes/day
                          so the DB pool stays healthy)

Per-tf overrides are accepted via `ML_REFRESH_CADENCE_<TF>_HOURS`. The
list of timeframes the loop manages comes from
`ML_REFRESH_TIMEFRAMES` (default = all six).

Failures (DB unreachable, schema drift, OOM, etc.) are surfaced loudly:

  * a `[ALERT]` line is emitted on stderr (workflow log shows it bright)
  * a JSON line is appended to
    `models/datasets/_freshness_alerts.jsonl`
  * `models/datasets/_freshness_status.json` is rewritten with the
    per-tf `last_success_at`, `last_attempt_at`, `last_error`,
    `next_due_at`, and the mtime of whatever snapshot is on disk

The retrain harness (`scripts/retrain_task524.py`) reads the freshest
snapshot via `_latest_pooled_dataset(tf, max_age_hours=36)` and
*refuses to run* if the picker raises — so a silent multi-day
refresher outage now turns into a loud, fail-loud retrain rather than
training on week-old data.

Run (Replit workflow `dataset-refresher` runs this command)::

    cd artifacts/ml-engine && \\
        ../../.pythonlibs/bin/python -u -m scripts.scheduled_refresh_loop

Useful env knobs:

    ML_REFRESH_TICK_SECONDS=1800
    ML_REFRESH_TIMEFRAMES=1d,6h,2h,1h,5m,1m
    ML_REFRESH_CADENCE_1D_HOURS=24
    ML_REFRESH_CADENCE_6H_HOURS=24
    ML_REFRESH_CADENCE_2H_HOURS=6
    ML_REFRESH_CADENCE_1H_HOURS=6
    ML_REFRESH_CADENCE_5M_HOURS=6
    ML_REFRESH_CADENCE_1M_HOURS=6
    ML_REFRESH_RUN_ONCE=1            # exit after one tick (used by tests)
    ML_REFRESH_INITIAL_TICK=1        # refresh due tfs on startup (default)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import close_pool, init_pool  # noqa: E402
from app.training.registry import make_version  # noqa: E402
from app.training.train import DEFAULT_TIMEFRAMES  # noqa: E402

from scripts.refresh_cached_datasets import (  # noqa: E402
    _refresh_one,
    _selected_coins,
)

DEFAULT_CADENCE_HOURS: dict[str, float] = {
    "1d": 24.0,
    "6h": 24.0,
    "2h": 6.0,
    "1h": 6.0,
    "5m": 6.0,
    "1m": 6.0,
}

# Task #556 — keep ~2 weeks of history per timeframe so the cache
# never drifts into the multi-GB range. 24h-cadence tfs (1d, 6h)
# generate ~1 snapshot/day, 6h-cadence tfs (2h, 1h, 5m, 1m) generate
# ~4 snapshots/day, so 14 vs 28 both map to ~14 days of history.
# Pinned snapshots referenced by diagnostic harnesses are protected
# unconditionally (see ``_pinned_filenames``).
DEFAULT_RETENTION: dict[str, int] = {
    "1d": 14,
    "6h": 14,
    "2h": 28,
    "1h": 28,
    "5m": 28,
    "1m": 28,
}

DATASETS_DIR = ROOT / "models" / "datasets"
STATUS_PATH = DATASETS_DIR / "_freshness_status.json"
ALERTS_PATH = DATASETS_DIR / "_freshness_alerts.jsonl"

# Task #599 — short-tf calibration verdict trigger. The parallel
# stage-2 runner (`scripts.task592_parallel_stage2`) re-runs the
# 1h+2h ON/OFF verdict; we re-trigger it whenever a 1h or 2h
# refresh successfully lands a fresh pooled snapshot so operators
# see the verdict track the freshest data.
CALIBRATION_VERDICT_TFS: tuple[str, ...] = ("1h", "2h")
CALIBRATION_VERDICT_REPORT_GLOB = "*-task592-1h2h-stage2-verdict.json"
REPORTS_DIR = ROOT / "reports"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected_timeframes() -> list[str]:
    raw = os.environ.get("ML_REFRESH_TIMEFRAMES")
    if not raw:
        return ["1d", "6h", "2h", "1h", "5m", "1m"]
    return [t.strip() for t in raw.split(",") if t.strip()]


def _cadence_hours(tf: str) -> float:
    raw = os.environ.get(f"ML_REFRESH_CADENCE_{tf.upper()}_HOURS")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_CADENCE_HOURS.get(tf, 24.0)


def _tick_seconds() -> int:
    raw = os.environ.get("ML_REFRESH_TICK_SECONDS")
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return 1800


def _newest_snapshot_mtime(tf: str) -> Optional[float]:
    candidates = sorted(
        DATASETS_DIR.glob(f"{tf}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].stat().st_mtime


def _retention_count(tf: str) -> int:
    """Per-tf retention (how many freshest snapshots to keep). Override
    via ``ML_REFRESH_RETAIN_<TF>`` (a non-positive value disables
    trimming for that tf — operator escape hatch for forensic windows).
    """
    raw = os.environ.get(f"ML_REFRESH_RETAIN_{tf.upper()}")
    if raw is not None:
        try:
            n = int(raw)
            return n  # may be 0 / negative → trimming disabled
        except ValueError:
            pass
    return DEFAULT_RETENTION.get(tf, 14)


def _trim_spike_threshold() -> int:
    """Task #560 — how many deletions in a single tick is "way more
    than usual" and should fire a ``trim_spike`` alert.

    Steady state should delete 0–1 file per tick (the cadence is
    multiples of an hour, retention is days). A sudden spike almost
    always means something abnormal: clock skew that backdated a bunch
    of mtimes, an operator manually copying old snapshots in, or a
    retention env override that was just lowered. The alert lets an
    operator notice before historical-looking snapshots silently
    disappear.

    Override via ``ML_REFRESH_TRIM_SPIKE_THRESHOLD``. A non-positive
    value disables spike alerts entirely (operator escape hatch for
    intentional bulk-trim windows).
    """
    raw = os.environ.get("ML_REFRESH_TRIM_SPIKE_THRESHOLD")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return 5


def _maybe_alert_trim_spike(tf: str, summary: dict) -> None:
    """Emit a ``trim_spike`` alert if ``summary`` reports more deletions
    than ``_trim_spike_threshold()``. Called from ``_run_tick`` after
    each successful refresh's trim.
    """
    threshold = _trim_spike_threshold()
    if threshold <= 0:
        return
    deleted_count = int(summary.get("deleted") or 0)
    if deleted_count <= threshold:
        return
    _append_alert({
        "at": _now_iso(),
        "timeframe": tf,
        "status": "trim_spike",
        "deleted_count": deleted_count,
        "retain_n": summary.get("retain_n"),
        "threshold": threshold,
        "deleted_files": list(summary.get("deleted_files") or []),
    })


def _pinned_filenames() -> set[str]:
    """Return every parquet filename that a diagnostic harness pins for
    reproducibility. These must never be trimmed even if they age out
    of the per-tf retention window — the diagnostic scripts deliberately
    reference specific historical snapshots so reruns produce the same
    numbers as the archived report.

    Today only ``run_516_pnl_ab.PINNED_DATASETS`` exists; new diagnostic
    scripts that pin snapshots should add their dict to this aggregator.

    Raises ``RuntimeError`` if any pin source fails to import. The trim
    path treats this as fail-safe: it skips trimming for the tick
    rather than risk deleting snapshots that *should* have been pinned
    but weren't visible because of a transient import error. This
    preserves the "pinned snapshots are NEVER trimmed" guarantee.
    """
    pinned: set[str] = set()
    try:
        from scripts.diagnostic_482.run_516_pnl_ab import (  # noqa: WPS433
            PINNED_DATASETS as _ab_pins,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"failed to load pinned dataset manifest from "
            f"scripts.diagnostic_482.run_516_pnl_ab: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    pinned.update(_ab_pins.values())
    return pinned


def _bytes_on_disk(tf: str) -> int:
    """Total size in bytes of all currently-on-disk
    ``<tf>_*.parquet`` snapshots. Files that disappear mid-stat
    (e.g. another tick deleting them) are treated as 0 rather than
    crashing the trim path — the next tick will report the corrected
    total.
    """
    total = 0
    for p in DATASETS_DIR.glob(f"{tf}_*.parquet"):
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def _trim_old_snapshots(tf: str) -> dict:
    """Keep the freshest ``_retention_count(tf)`` snapshots for
    timeframe ``tf`` plus any pinned filenames; delete the rest.

    Returns a summary dict suitable for embedding in the per-tf
    ``_freshness_status.json`` entry::

        {
          "policy": "keep_newest_n",
          "retain_n": 14,
          "retained": 14,
          "deleted": 3,
          "pinned_kept": ["1d_20260425T103252Z.parquet"],
          "oldest_kept_at": "2026-04-15T10:32:52Z",
          "deleted_files": ["1d_20260401T...parquet", ...],
          "bytes_on_disk": 12345678,
        }

    ``bytes_on_disk`` (Task #559) is the post-trim sum of
    ``<tf>_*.parquet`` file sizes — exactly what the freshness
    dashboard panel needs to confirm the cache is staying within
    the documented ~14-day footprint and to spot per-tf bloat
    (e.g. 1m parquets ballooning).

    A non-positive ``retain_n`` short-circuits to a no-op (operator
    escape hatch — set ``ML_REFRESH_RETAIN_<TF>=0`` to freeze the
    cache).
    """
    retain_n = _retention_count(tf)
    summary: dict = {
        "policy": "keep_newest_n",
        "retain_n": retain_n,
        "retained": 0,
        "deleted": 0,
        "pinned_kept": [],
        "oldest_kept_at": None,
        "deleted_files": [],
        "bytes_on_disk": 0,
    }
    if retain_n <= 0:
        # Trimming disabled — still report on-disk counts so operators
        # can see the cache is unbounded.
        existing = sorted(DATASETS_DIR.glob(f"{tf}_*.parquet"))
        summary["retained"] = len(existing)
        if existing:
            mtimes = [p.stat().st_mtime for p in existing]
            summary["oldest_kept_at"] = _ts_to_iso(min(mtimes))
        summary["bytes_on_disk"] = _bytes_on_disk(tf)
        return summary

    try:
        pinned = _pinned_filenames()
    except Exception as exc:  # noqa: BLE001
        # Fail-safe: if we cannot enumerate pinned snapshots, refuse
        # to trim. The "never delete pinned datasets" guarantee is
        # more important than the "cache is bounded" goal — a missed
        # trim tick is harmless (next successful tick reclaims the
        # space), but deleting a pinned snapshot breaks reproducibility
        # of an archived diagnostic report.
        existing = sorted(DATASETS_DIR.glob(f"{tf}_*.parquet"))
        summary["retained"] = len(existing)
        summary["deleted"] = 0
        summary["skipped"] = "pinned_filenames_unavailable"
        summary["skip_error"] = f"{type(exc).__name__}: {exc}"
        if existing:
            mtimes = [p.stat().st_mtime for p in existing]
            summary["oldest_kept_at"] = _ts_to_iso(min(mtimes))
        summary["bytes_on_disk"] = _bytes_on_disk(tf)
        return summary

    candidates = sorted(
        DATASETS_DIR.glob(f"{tf}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,  # newest first
    )
    keep: list[Path] = []
    delete: list[Path] = []
    # Walk newest→oldest. The first ``retain_n`` snapshots are kept
    # unconditionally; everything older is deleted *unless* its filename
    # is pinned by a diagnostic harness.
    for idx, path in enumerate(candidates):
        if idx < retain_n:
            keep.append(path)
        elif path.name in pinned:
            keep.append(path)
            summary["pinned_kept"].append(path.name)
        else:
            delete.append(path)

    for path in delete:
        try:
            path.unlink()
            summary["deleted_files"].append(path.name)
        except OSError as exc:
            # Don't crash the tick if a snapshot is locked / already
            # gone — just log it onto the summary so operators can see.
            summary.setdefault("delete_errors", []).append(
                {"file": path.name, "error": f"{type(exc).__name__}: {exc}"},
            )

    summary["retained"] = len(keep)
    summary["deleted"] = len(summary["deleted_files"])
    if keep:
        oldest_mtime = min(p.stat().st_mtime for p in keep)
        summary["oldest_kept_at"] = _ts_to_iso(oldest_mtime)
    # Recompute bytes_on_disk *after* the unlink loop so the dashboard
    # reflects the post-trim cache size, not the pre-trim figure.
    summary["bytes_on_disk"] = _bytes_on_disk(tf)
    return summary


def _load_status() -> dict:
    if not STATUS_PATH.exists():
        return {"timeframes": {}}
    try:
        return json.loads(STATUS_PATH.read_text())
    except json.JSONDecodeError:
        return {"timeframes": {}}


def _save_status(status: dict) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    status["written_at"] = _now_iso()
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True))


def _append_alert(payload: dict) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True)
    with ALERTS_PATH.open("a") as fh:
        fh.write(line + "\n")
    # Loud stderr print — the `dataset-refresher` workflow log will
    # surface these bright at the top of the next operator inspection.
    print(f"[ALERT] dataset-refresher: {line}", file=sys.stderr, flush=True)


def _due_timeframes(
    timeframes: Iterable[str],
    status: dict,
    now: float,
    *,
    initial_tick: bool = False,
) -> list[str]:
    """Return the subset of ``timeframes`` whose newest snapshot is
    older than the per-tf cadence. ``initial_tick=True`` forces every
    tf with no on-disk snapshot to be refreshed immediately on startup.
    """
    tf_status = status.get("timeframes", {})
    due: list[str] = []
    for tf in timeframes:
        cadence_h = _cadence_hours(tf)
        cadence_s = cadence_h * 3600.0
        mtime = _newest_snapshot_mtime(tf)
        if mtime is None:
            # No snapshot at all — always refresh (this is the
            # cold-start path; refusing it would brick the retrain
            # harness forever).
            due.append(tf)
            continue
        # Use the most-recent of (last_success_at, snapshot mtime) so
        # an external manual refresh counts as a successful tick.
        last_succ_iso = tf_status.get(tf, {}).get("last_success_at")
        last_succ_ts = mtime
        if last_succ_iso:
            try:
                last_succ_ts = max(
                    last_succ_ts,
                    datetime.strptime(
                        last_succ_iso, "%Y-%m-%dT%H:%M:%SZ",
                    ).replace(tzinfo=timezone.utc).timestamp(),
                )
            except ValueError:
                pass
        age_s = now - last_succ_ts
        if age_s >= cadence_s:
            due.append(tf)
        elif initial_tick and not last_succ_iso:
            # Snapshot exists on disk but the scheduler has no record
            # of having refreshed it — kick a refresh so the status
            # file becomes the source of truth from this tick onward.
            due.append(tf)
    return due


async def _run_tick(
    timeframes: list[str],
    coins: list[str],
    *,
    initial_tick: bool = False,
) -> dict:
    """Execute one scheduler tick: refresh due tfs, write status +
    alerts, return the resulting status dict."""
    status = _load_status()
    status.setdefault("timeframes", {})
    now = time.time()
    due = _due_timeframes(timeframes, status, now, initial_tick=initial_tick)
    print(
        f"[task540] tick at {_now_iso()}  "
        f"managed={timeframes}  due={due}",
        flush=True,
    )

    if not due:
        # Still update next_due_at for visibility, and refresh the
        # bytes-on-disk reading so the dashboard stays current even
        # during quiet ticks (Task #559).
        for tf in timeframes:
            entry = status["timeframes"].setdefault(tf, {})
            entry["next_due_at"] = _format_next_due(tf, now)
            mtime = _newest_snapshot_mtime(tf)
            entry["mtime_of_newest_snapshot"] = (
                _ts_to_iso(mtime) if mtime else None
            )
            retention = entry.get("retention") or {}
            retention["bytes_on_disk"] = _bytes_on_disk(tf)
            entry["retention"] = retention
        _update_cache_size_telemetry(status, timeframes, now)
        _save_status(status)
        return status

    refreshed_ok: list[str] = []
    await init_pool()
    try:
        version = make_version()
        for tf in due:
            attempt_iso = _now_iso()
            try:
                row = await _refresh_one(tf, coins, version)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "timeframe": tf, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            entry = status["timeframes"].setdefault(tf, {})
            entry["last_attempt_at"] = attempt_iso
            entry["cadence_hours"] = _cadence_hours(tf)
            entry["last_status"] = row.get("status")
            entry["next_due_at"] = _format_next_due(tf, time.time())
            mtime = _newest_snapshot_mtime(tf)
            entry["mtime_of_newest_snapshot"] = (
                _ts_to_iso(mtime) if mtime else None
            )
            if row.get("status") == "ok":
                entry["last_success_at"] = attempt_iso
                entry["last_error"] = None
                entry["last_n_rows"] = row.get("n_rows")
                entry["last_coins_emitted"] = row.get("coins_emitted")
                entry["last_output_path"] = row.get("output_path")
                # Task #556 — trim old snapshots after each successful
                # refresh so the cache never grows past ~2 weeks of
                # history per timeframe. Pinned diagnostic snapshots
                # are protected unconditionally.
                entry["retention"] = _trim_old_snapshots(tf)
                # Task #560 — alert if this tick deleted way more files
                # than usual (signals clock skew, manual snapshot copy,
                # or a freshly-lowered retention env override).
                _maybe_alert_trim_spike(tf, entry["retention"])
                # Task #599 — record short-tf successes so we can
                # auto-trigger the calibration verdict once the DB
                # pool has been closed below.
                refreshed_ok.append(tf)
            elif row.get("status") == "empty":
                # No labelable data is *not* a failure (the labeling
                # pipeline drops coins with synthetic-only history).
                # Still mark a success so we don't loop on the same tf.
                entry["last_success_at"] = attempt_iso
                entry["last_error"] = None
                entry["last_n_rows"] = 0
                entry["last_coins_emitted"] = []
                # An "empty" refresh did not write a new snapshot, but
                # we still trim so a long string of empty refreshes
                # doesn't leave stale snapshots accumulating from a
                # previously-healthy run.
                entry["retention"] = _trim_old_snapshots(tf)
                _maybe_alert_trim_spike(tf, entry["retention"])
                # Task #599 — empty refresh still successfully wrote no
                # rows; do NOT trigger the verdict here because the
                # underlying parquet did not change.
            else:
                err = row.get("error") or row.get("status") or "unknown"
                entry["last_error"] = str(err)
                # Task #559 — even on error, refresh the per-tf
                # bytes-on-disk reading so the dashboard reflects
                # current cache footprint instead of going stale.
                retention = entry.get("retention") or {}
                retention["bytes_on_disk"] = _bytes_on_disk(tf)
                entry["retention"] = retention
                _append_alert({
                    "at": attempt_iso,
                    "timeframe": tf,
                    "status": row.get("status"),
                    "error": str(err),
                    "cadence_hours": _cadence_hours(tf),
                })
    finally:
        await close_pool()

    # Task #599 — re-run the 1h+2h calibration ON/OFF verdict whenever
    # a short-tf refresh just landed a fresh pooled snapshot. Runs in
    # a subprocess (after the DB pool is closed) so the verdict's
    # process-pool spawn doesn't trip over an open asyncio loop.
    _maybe_run_calibration_verdict(status, refreshed_ok)

    _update_cache_size_telemetry(status, timeframes, time.time())
    _save_status(status)
    return status


# Task #559 — keep the last N total-cache-size samples so the
# freshness dashboard can render a sparkline of cache footprint over
# time (catches "1m parquets ballooned overnight" without forcing
# operators to ssh in and `du -sh`). Default = 60 samples (≈30 hours
# at the 30-minute default tick).
DEFAULT_SIZE_HISTORY_LEN = 60

# Soft warning threshold: anything > 5 GB across all managed
# timeframes is suspicious for a cache that should top out around
# 14 days × ~6 timeframes of pooled snapshots. The dashboard turns
# the total chip amber when this trips.
CACHE_SIZE_WARN_BYTES = 5 * 1024 * 1024 * 1024


def _size_history_len() -> int:
    raw = os.environ.get("ML_REFRESH_SIZE_HISTORY_LEN")
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return DEFAULT_SIZE_HISTORY_LEN


def _update_cache_size_telemetry(
    status: dict,
    timeframes: list[str],
    now_ts: float,
) -> None:
    """Compute ``total_bytes_on_disk`` across all managed timeframes
    and append a single sample to ``cache_size_history`` (capped at
    ``ML_REFRESH_SIZE_HISTORY_LEN`` entries, default 60). Mutates
    ``status`` in place.

    The history sample stores the per-tf breakdown alongside the
    total so the dashboard sparkline can attribute spikes to the
    timeframe that caused them.
    """
    per_tf: dict[str, int] = {}
    total = 0
    for tf in timeframes:
        b = _bytes_on_disk(tf)
        per_tf[tf] = b
        total += b
    status["total_bytes_on_disk"] = total
    status["cache_size_warn_bytes"] = CACHE_SIZE_WARN_BYTES
    status["cache_size_warning"] = total > CACHE_SIZE_WARN_BYTES
    history = status.get("cache_size_history") or []
    if not isinstance(history, list):
        history = []
    history.append({
        "at": _ts_to_iso(now_ts),
        "total_bytes": total,
        "per_tf": per_tf,
    })
    cap = _size_history_len()
    if len(history) > cap:
        history = history[-cap:]
    status["cache_size_history"] = history


def _calibration_verdict_enabled() -> bool:
    """Task #599 — operator escape hatch. Set
    ``ML_REFRESH_AUTO_CALIBRATION_VERDICT=0`` (or ``false``) to skip the
    short-tf verdict re-run after a successful 1h/2h refresh. Default is
    enabled because the parallel runner is cheap (~70s).
    """
    raw = os.environ.get("ML_REFRESH_AUTO_CALIBRATION_VERDICT", "1")
    return raw not in {"0", "false", "False", ""}


def _calibration_verdict_timeout_s() -> int:
    """Task #599 — wall-clock cap on the verdict subprocess. Override
    via ``ML_REFRESH_CALIBRATION_VERDICT_TIMEOUT_S``. Default 600s
    (the parallel runner finishes in ~70s, so 10x is plenty of headroom
    for transient slowness without making a stuck subprocess hang the
    refresher loop forever).
    """
    raw = os.environ.get("ML_REFRESH_CALIBRATION_VERDICT_TIMEOUT_S")
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return 600


def _latest_calibration_verdict_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Return ``(json_path, md_path)`` for the freshest task592 verdict
    on disk, or ``(None, None)`` if none exists. Picked by mtime so an
    operator who reruns the script by hand still wins over a stale auto
    one if their run is newer.
    """
    if not REPORTS_DIR.exists():
        return None, None
    json_candidates = sorted(
        REPORTS_DIR.glob(CALIBRATION_VERDICT_REPORT_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_candidates:
        return None, None
    json_path = json_candidates[0]
    md_path = json_path.with_suffix(".md")
    return json_path, md_path if md_path.exists() else None


def _summarize_verdict_payload(payload: dict) -> dict:
    """Pull just the operator-relevant aggregate numbers out of the
    verdict JSON so the dashboard doesn't have to parse a 1000+-line
    file. Mirrors the ``_gate_summary`` reduction inside
    ``task587_rerun_stage2.aggregate_and_write`` — kept local so a future
    refactor of the verdict script does not silently break the loop.
    """
    def _stage_summary(stage: dict | None) -> dict:
        per_slice = list((stage or {}).get("per_slice") or [])
        evald = [
            s for s in per_slice
            if s.get("gate_pass") is not None and "augmented" in s
        ]
        in_band = [
            s for s in evald
            if (s.get("augmented") or {}).get("trade_share") is not None
            and 0.40 <= s["augmented"]["trade_share"] <= 0.85
        ]
        passed = [s for s in evald if s.get("gate_pass")]
        trade_shares = [
            s["augmented"]["trade_share"] for s in evald
            if (s.get("augmented") or {}).get("trade_share") is not None
        ]
        da_lifts = [
            s["da_lift"] for s in evald if s.get("da_lift") is not None
        ]
        pnls = [
            s["augmented"]["post_fee_pnl_pct_total"] for s in evald
            if (s.get("augmented") or {}).get("post_fee_pnl_pct_total")
            is not None
        ]
        return {
            "n_slices": len(evald),
            "n_passing_gate": len(passed),
            "n_in_trade_share_band": len(in_band),
            "mean_trade_share": (
                sum(trade_shares) / len(trade_shares)
                if trade_shares else None
            ),
            "mean_da_lift": (
                sum(da_lifts) / len(da_lifts) if da_lifts else None
            ),
            "sum_pnl_pct_total_aug": sum(pnls) if pnls else None,
        }
    return {
        "off": _stage_summary(payload.get("off")),
        "on": _stage_summary(payload.get("on")),
        "captured_at": payload.get("captured_at"),
        "round_trip_cost_pct": payload.get("round_trip_cost_pct"),
        "timeframes_subset": payload.get("timeframes_subset"),
        "wall_time_seconds": (
            (payload.get("run_metadata") or {}).get("wall_time_seconds")
        ),
        "n_workers": (
            (payload.get("run_metadata") or {}).get("n_workers")
        ),
        "n_work_units": (
            (payload.get("run_metadata") or {}).get("n_work_units")
        ),
    }


def _run_calibration_verdict(
    triggered_by: list[str],
) -> dict:
    """Run ``scripts.task592_parallel_stage2`` as a subprocess and
    return a status dict suitable for embedding in
    ``status['calibration_verdict']['short_tf']``.

    Subprocess (not in-process import) because:
      * task592 spawns its own process pool with the ``spawn`` start
        method; running it from inside an asyncio loop process is
        fragile (CUDA/BLAS reinit, OMP thread pinning, etc).
      * task592 mutates ``task587_rerun_stage2.TIMEFRAMES_SUBSET`` as
        a side-effect — keeping it isolated guarantees a clean run.

    The returned dict always carries ``last_attempt_at`` and either
    ``last_status='ok'`` (with summary + report paths) or
    ``last_status='error'`` / ``'timeout'`` (with ``last_error``).
    The caller is responsible for appending an alert on failure.
    """
    import subprocess
    attempt_iso = _now_iso()
    started = time.time()
    cmd = [
        sys.executable, "-u", "-m", "scripts.task592_parallel_stage2",
    ]
    timeout_s = _calibration_verdict_timeout_s()
    print(
        f"[task599] calibration verdict subprocess starting  "
        f"trigger={triggered_by}  timeout_s={timeout_s}  cmd={' '.join(cmd)}",
        flush=True,
    )
    base: dict = {
        "last_attempt_at": attempt_iso,
        "trigger_timeframes": list(triggered_by),
        "command": " ".join(cmd),
        "timeout_seconds": timeout_s,
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.time() - started, 1)
        return {
            **base,
            "last_status": "timeout",
            "last_error": f"verdict subprocess timed out after {timeout_s}s",
            "last_elapsed_seconds": elapsed,
            "stderr_tail": (exc.stderr or "")[-2000:] if exc.stderr else "",
        }
    elapsed = round(time.time() - started, 1)
    if proc.returncode != 0:
        return {
            **base,
            "last_status": "error",
            "last_error": (
                f"verdict subprocess exited with rc={proc.returncode}"
            ),
            "last_elapsed_seconds": elapsed,
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    json_path, md_path = _latest_calibration_verdict_paths()
    if json_path is None:
        return {
            **base,
            "last_status": "error",
            "last_error": (
                "verdict subprocess succeeded but no "
                f"{CALIBRATION_VERDICT_REPORT_GLOB} report was found in "
                f"{REPORTS_DIR}"
            ),
            "last_elapsed_seconds": elapsed,
        }
    try:
        payload = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "last_status": "error",
            "last_error": f"failed to parse {json_path.name}: {exc}",
            "last_elapsed_seconds": elapsed,
        }
    summary = _summarize_verdict_payload(payload)
    rel_md = md_path.relative_to(ROOT) if md_path else None
    rel_json = json_path.relative_to(ROOT)
    return {
        **base,
        "last_status": "ok",
        "last_success_at": attempt_iso,
        "last_error": None,
        "last_elapsed_seconds": elapsed,
        "last_md_path": str(rel_md) if rel_md else None,
        "last_json_path": str(rel_json),
        "summary": summary,
    }


def _maybe_run_calibration_verdict(
    status: dict,
    triggered_by: list[str],
) -> None:
    """If any short-tf refreshed successfully this tick (and the env
    knob hasn't disabled it), run the verdict subprocess and stash the
    result under ``status['calibration_verdict']['short_tf']``. Failures
    additionally append a ``calibration_verdict`` alert so the
    dashboard alert log surfaces them.
    """
    relevant = [tf for tf in triggered_by if tf in CALIBRATION_VERDICT_TFS]
    if not relevant:
        return
    if not _calibration_verdict_enabled():
        print(
            f"[task599] calibration verdict trigger fired ({relevant}) "
            "but ML_REFRESH_AUTO_CALIBRATION_VERDICT is disabled — skipping",
            flush=True,
        )
        return
    result = _run_calibration_verdict(relevant)
    container = status.setdefault("calibration_verdict", {})
    prev = container.get("short_tf") or {}
    # Preserve last_success_at + last_md_path across failed reruns so
    # the dashboard can still show the most recent good verdict
    # alongside the new failure.
    if result.get("last_status") != "ok":
        if prev.get("last_success_at") and not result.get("last_success_at"):
            result["last_success_at"] = prev["last_success_at"]
        if prev.get("last_md_path") and not result.get("last_md_path"):
            result["last_md_path"] = prev["last_md_path"]
        if prev.get("last_json_path") and not result.get("last_json_path"):
            result["last_json_path"] = prev["last_json_path"]
        if prev.get("summary") and not result.get("summary"):
            result["summary"] = prev["summary"]
        _append_alert({
            "at": result["last_attempt_at"],
            "timeframe": "+".join(relevant),
            "status": f"calibration_verdict_{result['last_status']}",
            "error": result.get("last_error") or "unknown",
            "elapsed_seconds": result.get("last_elapsed_seconds"),
        })
        print(
            f"[task599] calibration verdict FAILED  "
            f"status={result['last_status']}  "
            f"error={result.get('last_error')}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[task599] calibration verdict ok  "
            f"elapsed={result.get('last_elapsed_seconds')}s  "
            f"md={result.get('last_md_path')}",
            flush=True,
        )
    container["short_tf"] = result


def _format_next_due(tf: str, last_ts: float) -> str:
    next_ts = last_ts + _cadence_hours(tf) * 3600.0
    return datetime.fromtimestamp(next_ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )


def _funding_backfill_enabled() -> bool:
    """Task #628 — toggle for the on-startup OKX funding-rate backfill.

    Default ON. Set ``ML_REFRESH_FUNDING_BACKFILL=0`` to skip (e.g. on
    a fresh dev container with no outbound HTTP, or when a Coinglass
    backfill has already populated the table to a deeper history).
    The backfill itself is gap-fill mode by default (skips coins that
    already have ≥ 800 funding rows in the lookback window) so it is
    cheap to leave on permanently.
    """
    raw = os.environ.get("ML_REFRESH_FUNDING_BACKFILL", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


async def _maybe_run_funding_backfill() -> None:
    """Task #628 — run the OKX funding-rate-history backfill once on
    startup. Wrapped in its own try/except so any failure (HTTP, DB,
    schema) is loud but never blocks the dataset-refresh loop itself.

    Imports are local on purpose: the backfill module pulls
    ``urllib.request`` and walks an external HTTP endpoint, neither of
    which we want to import at scheduler module-load time (the loop
    is the long-running process that owns the lifecycle).
    """
    if not _funding_backfill_enabled():
        print(
            "[task628] funding-rate backfill disabled "
            "(ML_REFRESH_FUNDING_BACKFILL=0)",
            flush=True,
        )
        return
    try:
        # Default to gap-fill mode for the scheduler so a healthy
        # table doesn't re-fetch ~3000 rows every container restart.
        os.environ.setdefault("ML_FUNDING_BACKFILL_GAP_FILL_ONLY", "1")
        from scripts.backfill_funding_history import _amain as _backfill_amain  # noqa: WPS433
        rc = await _backfill_amain()
        if rc != 0:
            _append_alert({
                "at": _now_iso(),
                "timeframe": "(funding-backfill)",
                "status": "failed",
                "error": f"backfill returned non-zero exit code rc={rc}",
            })
    except Exception as exc:  # noqa: BLE001
        _append_alert({
            "at": _now_iso(),
            "timeframe": "(funding-backfill)",
            "status": "crashed",
            "error": f"{type(exc).__name__}: {exc}",
        })
        print(
            f"[task628] funding-rate backfill crashed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


async def _amain() -> int:
    timeframes = [t for t in _selected_timeframes() if t in DEFAULT_TIMEFRAMES]
    if not timeframes:
        print(
            f"[task540] no valid timeframes selected "
            f"(ML_REFRESH_TIMEFRAMES={os.environ.get('ML_REFRESH_TIMEFRAMES')!r}); exiting",
            file=sys.stderr,
            flush=True,
        )
        return 2
    coins = _selected_coins()
    tick_s = _tick_seconds()
    run_once = os.environ.get("ML_REFRESH_RUN_ONCE") == "1"
    initial_tick_env = os.environ.get("ML_REFRESH_INITIAL_TICK", "1")
    initial_tick = initial_tick_env not in {"0", "false", "False"}
    print(
        f"[task540] dataset-refresher started  "
        f"tfs={timeframes}  coins={len(coins)}  "
        f"tick_s={tick_s}  run_once={run_once}  "
        f"initial_tick={initial_tick}",
        flush=True,
    )
    # Task #628 — pull historical funding-rate data into market_signals
    # before the first dataset refresh so the labels.py asof-join
    # actually has microstructure values to attach. Runs in gap-fill
    # mode so it's a cheap no-op if the table already has coverage.
    await _maybe_run_funding_backfill()
    first = True
    while True:
        try:
            await _run_tick(
                timeframes, coins,
                initial_tick=initial_tick and first,
            )
        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            _append_alert({
                "at": _now_iso(),
                "timeframe": "(loop)",
                "status": "tick_crashed",
                "error": f"{type(exc).__name__}: {exc}",
            })
        first = False
        if run_once:
            return 0
        await asyncio.sleep(tick_s)


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
