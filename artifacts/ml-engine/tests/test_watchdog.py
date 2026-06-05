"""Unit tests for the training watchdog (Task #306 follow-up)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training.watchdog import (
    STALL_THRESHOLD,
    STATUS_CONVERGED,
    STATUS_FIRST_RUN,
    STATUS_FLAT,
    STATUS_IMPROVING,
    STATUS_NO_VERIFICATION,
    STATUS_REGRESSED,
    STATUS_STALLED,
    build_snapshot,
    diff_snapshots,
    read_history,
    record_verification_snapshot,
)


def _report(*, passed: bool, promoted: int, coins_with: list[str] | None = None) -> dict:
    return {
        "started_at": "2026-04-22T22:00:00Z",
        "verification": {
            "passed": passed,
            "active_coins": ["bonk", "pepe"],
            "coins_with_promotion": coins_with if coins_with is not None else (["bonk"] if promoted else []),
            "coins_without_promotion": [
                c for c in ["bonk", "pepe"] if c not in (coins_with if coins_with is not None else (["bonk"] if promoted else []))
            ],
            "promoted_by_coin": {"bonk": promoted, "pepe": 0},
            "counts": {
                "slices_promoted": promoted,
                "slices_no_lift": 0,
                "slices_below_coinflip": 60 - promoted,
                "slices_insufficient_sample": 0,
                "slices_contract_failed": 0,
                "slices_untrained": 0,
            },
        },
    }


def test_build_snapshot_returns_none_without_verification():
    assert build_snapshot({"started_at": "x"}) is None


def test_build_snapshot_normalizes_counts():
    snap = build_snapshot(_report(passed=False, promoted=2))
    assert snap is not None
    assert snap["counts"]["slices_promoted"] == 2
    assert snap["counts"]["slices_below_coinflip"] == 58
    assert snap["passed"] is False
    assert snap["coins_with_promotion"] == ["bonk"]


def test_first_run_diff():
    snap = build_snapshot(_report(passed=False, promoted=0))
    diff = diff_snapshots(None, snap)
    assert diff["status"] == STATUS_FIRST_RUN
    assert diff["delta_promoted"] == 0
    assert diff["stall_streak"] == 0


def test_first_run_already_passed():
    snap = build_snapshot(_report(passed=True, promoted=10, coins_with=["bonk", "pepe"]))
    diff = diff_snapshots(None, snap)
    assert diff["status"] == STATUS_CONVERGED


def test_improving_diff():
    prev = build_snapshot(_report(passed=False, promoted=2))
    curr = build_snapshot(_report(passed=False, promoted=5))
    diff = diff_snapshots(prev, curr)
    assert diff["status"] == STATUS_IMPROVING
    assert diff["delta_promoted"] == 3
    assert diff["stall_streak"] == 0


def test_regressed_diff():
    prev = build_snapshot(_report(passed=False, promoted=5))
    curr = build_snapshot(_report(passed=False, promoted=2))
    diff = diff_snapshots(prev, curr)
    assert diff["status"] == STATUS_REGRESSED
    assert diff["delta_promoted"] == -3


def test_converged_diff_overrides_flat():
    """passed=true wins even when the count is unchanged."""
    prev = build_snapshot(_report(passed=False, promoted=2))
    curr = build_snapshot(_report(passed=True, promoted=2, coins_with=["bonk", "pepe"]))
    diff = diff_snapshots(prev, curr)
    assert diff["status"] == STATUS_CONVERGED


def test_stall_streak_increments_then_fires():
    prev = build_snapshot(_report(passed=False, promoted=3))
    prev["stall_streak"] = 0
    diffs = []
    for _ in range(STALL_THRESHOLD):
        curr = build_snapshot(_report(passed=False, promoted=3))
        d = diff_snapshots(prev, curr)
        diffs.append(d)
        # Carry the streak forward like the recorder would.
        prev = curr
        prev["stall_streak"] = d["stall_streak"]
    assert diffs[0]["status"] == STATUS_FLAT
    assert diffs[-1]["status"] == STATUS_STALLED
    assert diffs[-1]["stall_streak"] == STALL_THRESHOLD


def test_stall_streak_resets_on_improvement():
    prev = build_snapshot(_report(passed=False, promoted=3))
    prev["stall_streak"] = 5
    curr = build_snapshot(_report(passed=False, promoted=4))
    diff = diff_snapshots(prev, curr)
    assert diff["status"] == STATUS_IMPROVING
    assert diff["stall_streak"] == 0


def test_newly_promoted_and_demoted_coins():
    prev = build_snapshot(_report(passed=False, promoted=1, coins_with=["bonk"]))
    curr = build_snapshot(_report(passed=False, promoted=1, coins_with=["pepe"]))
    diff = diff_snapshots(prev, curr)
    assert diff["newly_promoted_coins"] == ["pepe"]
    assert diff["newly_demoted_coins"] == ["bonk"]


def test_record_appends_jsonl_and_returns_diff(tmp_path: Path):
    diff1 = record_verification_snapshot(_report(passed=False, promoted=2), tmp_path)
    assert diff1["status"] == STATUS_FIRST_RUN
    diff2 = record_verification_snapshot(_report(passed=False, promoted=4), tmp_path)
    assert diff2["status"] == STATUS_IMPROVING
    assert diff2["delta_promoted"] == 2
    history_path = tmp_path / "verification_history.jsonl"
    lines = history_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(l) for l in lines]
    assert rows[0]["counts"]["slices_promoted"] == 2
    assert rows[1]["counts"]["slices_promoted"] == 4
    assert rows[1]["diff"]["status"] == STATUS_IMPROVING


def test_record_skips_when_no_verification(tmp_path: Path):
    diff = record_verification_snapshot({"started_at": "x"}, tmp_path)
    assert diff["status"] == STATUS_NO_VERIFICATION
    assert not (tmp_path / "verification_history.jsonl").exists()


def test_read_history_newest_first(tmp_path: Path):
    record_verification_snapshot(_report(passed=False, promoted=1), tmp_path)
    record_verification_snapshot(_report(passed=False, promoted=2), tmp_path)
    record_verification_snapshot(
        _report(passed=True, promoted=10, coins_with=["bonk", "pepe"]),
        tmp_path,
    )
    rows = read_history(tmp_path, limit=10)
    assert len(rows) == 3
    assert rows[0]["passed"] is True
    assert rows[-1]["counts"]["slices_promoted"] == 1


def test_read_history_respects_limit(tmp_path: Path):
    for i in range(5):
        record_verification_snapshot(_report(passed=False, promoted=i), tmp_path)
    rows = read_history(tmp_path, limit=2)
    assert len(rows) == 2
    assert rows[0]["counts"]["slices_promoted"] == 4
    assert rows[1]["counts"]["slices_promoted"] == 3


def test_read_history_handles_missing_file(tmp_path: Path):
    assert read_history(tmp_path, limit=10) == []


def test_record_survives_unwritable_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A write failure must not raise — training never aborts on watchdog IO."""
    bad = tmp_path / "nope"
    bad.write_text("")  # file, not dir — Path.parent.mkdir will succeed but open() will fail
    diff = record_verification_snapshot(
        _report(passed=False, promoted=1),
        bad / "history",
    )
    # The file path is bad/history/verification_history.jsonl ; bad is a
    # file so mkdir(parents=True, exist_ok=True) raises NotADirectoryError
    # which should be swallowed and still return a diff envelope.
    assert "status" in diff
