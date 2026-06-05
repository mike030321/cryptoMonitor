"""Contract tests for /ml/training/progress (task #393).

Covers the three states an operator cares about:
  * `currently_fitting` is parsed from the most recent unmatched
    `per_coin_start` headline (`fitting <coin>/<tf> (idx/total)`)
  * `stale=True` only when the newest row is older than 5 minutes
  * `run_finished=True` for terminal phases written by the campaign
    orchestrator and by `run_training`
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import main as ml_main
from app.training import registry as ml_registry


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture
def isolated_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ml_registry, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(ml_main, "REGISTRY_ROOT", tmp_path)
    return tmp_path / "progress_updates.jsonl"


def _run() -> dict:
    return asyncio.run(ml_main.training_progress())


def test_missing_journal_returns_missing_status(isolated_progress: Path) -> None:
    assert not isolated_progress.exists()
    assert _run() == {"status": "missing"}


def test_currently_fitting_is_parsed_from_unmatched_per_coin_start(
    isolated_progress: Path,
) -> None:
    now = datetime.now(timezone.utc)
    _write_rows(isolated_progress, [
        {"emitted_at": (now - timedelta(seconds=120)).isoformat(),
         "phase": "build_dataset_done", "status": "ok",
         "headline": "build_dataset_done tf=1h rows=12345 in 33.0s",
         "timeframe": "1h"},
        {"emitted_at": (now - timedelta(seconds=60)).isoformat(),
         "phase": "per_coin_start", "status": "running",
         "headline": "fitting bonk/1h (3/12)", "coin": "bonk"},
        {"emitted_at": (now - timedelta(seconds=30)).isoformat(),
         "phase": "per_coin_done", "status": "trained",
         "headline": "bonk/1h (3/12) status=trained", "coin": "bonk"},
        {"emitted_at": (now - timedelta(seconds=5)).isoformat(),
         "phase": "per_coin_start", "status": "running",
         "headline": "fitting pepe/1h (4/12)", "coin": "pepe"},
    ])
    res = _run()
    assert res["status"] == "ok"
    assert res["stale"] is False
    assert res["run_finished"] is False
    assert res["current_timeframe"] == "1h"
    fitting = res["currently_fitting"]
    assert fitting == {
        "coin": "pepe", "timeframe": "1h", "index": 4, "total": 12,
        "started_at": fitting["started_at"], "headline": "fitting pepe/1h (4/12)",
    }
    # newest first
    assert res["recent"][0]["coin"] == "pepe"


def test_stale_when_no_row_in_more_than_five_minutes(
    isolated_progress: Path,
) -> None:
    now = datetime.now(timezone.utc)
    _write_rows(isolated_progress, [
        {"emitted_at": (now - timedelta(minutes=12)).isoformat(),
         "phase": "build_dataset_done", "status": "ok",
         "headline": "build_dataset_done tf=1h", "timeframe": "1h"},
        {"emitted_at": (now - timedelta(minutes=10)).isoformat(),
         "phase": "per_coin_start", "status": "running",
         "headline": "fitting pepe/1h (4/12)", "coin": "pepe"},
    ])
    res = _run()
    assert res["status"] == "ok"
    assert res["stale"] is True
    assert res["stale_seconds"] is not None and res["stale_seconds"] > 5 * 60
    # The unmatched per_coin_start is still surfaced so the UI can show
    # what the trainer was working on when it went silent.
    assert res["currently_fitting"]["coin"] == "pepe"


@pytest.mark.parametrize(
    "terminal_phase",
    ["campaign_done", "campaign_failed", "final_summary", "training_done"],
)
def test_terminal_phases_clear_currently_fitting(
    isolated_progress: Path, terminal_phase: str,
) -> None:
    now = datetime.now(timezone.utc)
    _write_rows(isolated_progress, [
        {"emitted_at": (now - timedelta(seconds=30)).isoformat(),
         "phase": "per_coin_start", "status": "running",
         "headline": "fitting pepe/1h (12/12)", "coin": "pepe"},
        {"emitted_at": now.isoformat(),
         "phase": terminal_phase, "status": "ok",
         "headline": f"{terminal_phase} headline"},
    ])
    res = _run()
    assert res["status"] == "ok"
    assert res["run_finished"] is True
    assert res["currently_fitting"] is None
