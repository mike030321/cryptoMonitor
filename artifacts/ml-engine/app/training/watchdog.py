"""Training watchdog (Task #306 follow-up).

Every time a retrain produces a `verification` block, this module appends a
snapshot to `models/verification_history.jsonl` and computes a diff vs the
previous snapshot. The dashboard / operators can read the history via
`GET /ml/admin/verification-history` to see whether the brain is converging
(passed=true), improving (more slices promoted than last run), regressed
(fewer slices promoted), or stalled (no improvement for STALL_THRESHOLD
consecutive runs).

Hard rules:
- Never invents metrics. Reads only what `build_verification_block`
  produced.
- Never aborts training. All persistence is best-effort and wrapped.
- Idempotent within a single retrain — `record_verification_snapshot` is
  meant to be called once per `run_training`. Calling twice in the same
  second is harmless; it just appends a near-duplicate row.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("ml-engine.watchdog")

STALL_THRESHOLD = 3
STATUS_CONVERGED = "converged"
STATUS_IMPROVING = "improving"
STATUS_REGRESSED = "regressed"
STATUS_FLAT = "flat"
STATUS_STALLED = "stalled"
STATUS_FIRST_RUN = "first_run"
STATUS_NO_VERIFICATION = "no_verification"


def _history_path(registry_root: Path) -> Path:
    return registry_root / "verification_history.jsonl"


def _read_last_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        last_line: str | None = None
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return None
        return json.loads(last_line)
    except Exception as exc:  # noqa: BLE001
        logger.warning("watchdog_history_read_failed", extra={"error": str(exc)})
        return None


def _classify(prev_promoted: int, new_promoted: int, passed: bool) -> str:
    if passed:
        return STATUS_CONVERGED
    if new_promoted > prev_promoted:
        return STATUS_IMPROVING
    if new_promoted < prev_promoted:
        return STATUS_REGRESSED
    return STATUS_FLAT


def build_snapshot(
    report: dict[str, Any],
    *,
    now_ts: float | None = None,
) -> dict[str, Any] | None:
    """Distill the verification block into a snapshot row.

    Returns None when the report has no `verification` block (e.g. a
    fast-loop tick or a training run that errored before post-processing).
    """
    verification = report.get("verification")
    if not isinstance(verification, dict):
        return None
    counts = verification.get("counts") or {}
    promoted_by_coin = verification.get("promoted_by_coin") or {}
    return {
        "recorded_at": now_ts if now_ts is not None else time.time(),
        "source_report_started_at": report.get("started_at"),
        "source_report_completed_at": report.get("completed_at"),
        "verification_status": verification.get("status", "ok"),
        "passed": bool(verification.get("passed")),
        "active_coins": list(verification.get("active_coins") or []),
        "coins_with_promotion": list(
            verification.get("coins_with_promotion") or []
        ),
        "coins_without_promotion": list(
            verification.get("coins_without_promotion") or []
        ),
        "promoted_by_coin": dict(promoted_by_coin),
        "counts": {
            "slices_promoted": int(counts.get("slices_promoted", 0)),
            "slices_no_lift": int(counts.get("slices_no_lift", 0)),
            "slices_below_coinflip": int(
                counts.get("slices_below_coinflip", 0)
            ),
            "slices_insufficient_sample": int(
                counts.get("slices_insufficient_sample", 0)
            ),
            "slices_contract_failed": int(
                counts.get("slices_contract_failed", 0)
            ),
            "slices_untrained": int(counts.get("slices_untrained", 0)),
        },
    }


def diff_snapshots(
    prev: dict[str, Any] | None,
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Compare two snapshots and return a diff envelope.

    `prev` is None on the very first run.
    """
    curr_promoted = int(curr["counts"]["slices_promoted"])
    if prev is None:
        return {
            "status": STATUS_CONVERGED if curr["passed"] else STATUS_FIRST_RUN,
            "delta_promoted": curr_promoted,
            "stall_streak": 0,
            "prev_promoted": 0,
            "prev_passed": False,
            "newly_promoted_coins": list(curr["coins_with_promotion"]),
            "newly_demoted_coins": [],
        }
    prev_promoted = int(prev.get("counts", {}).get("slices_promoted", 0))
    prev_passed = bool(prev.get("passed"))
    prev_stall = int(prev.get("stall_streak", 0))
    status = _classify(prev_promoted, curr_promoted, curr["passed"])
    if status == STATUS_FLAT:
        stall_streak = prev_stall + 1
        if stall_streak >= STALL_THRESHOLD:
            status = STATUS_STALLED
    else:
        stall_streak = 0
    prev_set = set(prev.get("coins_with_promotion") or [])
    curr_set = set(curr["coins_with_promotion"])
    return {
        "status": status,
        "delta_promoted": curr_promoted - prev_promoted,
        "stall_streak": stall_streak,
        "prev_promoted": prev_promoted,
        "prev_passed": prev_passed,
        "newly_promoted_coins": sorted(curr_set - prev_set),
        "newly_demoted_coins": sorted(prev_set - curr_set),
    }


def record_verification_snapshot(
    report: dict[str, Any],
    registry_root: Path,
    *,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Append a snapshot to the history file and return the diff envelope.

    Always returns a dict so callers can stash it on the in-memory report
    even when the report has no verification block. Never raises — wraps
    every IO call.
    """
    snapshot = build_snapshot(report, now_ts=now_ts)
    if snapshot is None:
        return {
            "status": STATUS_NO_VERIFICATION,
            "delta_promoted": 0,
            "stall_streak": 0,
            "prev_promoted": 0,
            "prev_passed": False,
            "newly_promoted_coins": [],
            "newly_demoted_coins": [],
        }
    path = _history_path(registry_root)
    prev = _read_last_snapshot(path)
    diff = diff_snapshots(prev, snapshot)
    snapshot["diff"] = diff
    snapshot["stall_streak"] = diff["stall_streak"]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(snapshot, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("watchdog_history_write_failed", extra={"error": str(exc)})
    logger.info(
        "verification_watchdog",
        extra={
            "status": diff["status"],
            "passed": snapshot["passed"],
            "slices_promoted": snapshot["counts"]["slices_promoted"],
            "delta_promoted": diff["delta_promoted"],
            "stall_streak": diff["stall_streak"],
            "prev_promoted": diff["prev_promoted"],
        },
    )
    return diff


def read_history(
    registry_root: Path,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the last `limit` snapshots, newest first.

    Best-effort. Returns [] on any read error.
    """
    path = _history_path(registry_root)
    if not path.exists():
        return []
    try:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        rows.reverse()
        return rows[: max(1, int(limit))]
    except Exception as exc:  # noqa: BLE001
        logger.warning("watchdog_history_read_failed", extra={"error": str(exc)})
        return []
