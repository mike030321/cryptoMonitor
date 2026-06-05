"""Contract tests for /ml/training/live-gated-replay (Task #615).

The dashboard reads the per-slice live-gated replay block (verdict pill,
loose-vs-live PnL, dominant rejection reason) from the most recent
campaign's `phase7_summary.json`. The endpoint must:

* return `{status: "missing"}` when no `training_run_<TS>` folder has a
  `phase7_summary.json` on disk yet
* return `{status: "empty"}` when the newest summary has the block but
  with no per-slice entries (campaign predates Task #613 or phase 7 has
  not run)
* return `{status: "ok", per_slice, verdict_counts, run_dir, ...}` with
  the freshest run's data when multiple `training_run_<TS>` folders
  exist
* return `{status: "error"}` when the newest summary is corrupt
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app import main as ml_main
from app.training import registry as ml_registry


def _run() -> dict:
    return asyncio.run(ml_main.training_live_gated_replay())


@pytest.fixture
def isolated_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ml_registry, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(ml_main, "REGISTRY_ROOT", tmp_path)
    return tmp_path


def _write_summary(models_root: Path, ts: str, payload: dict) -> Path:
    run = models_root / f"training_run_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    p = run / "phase7_summary.json"
    p.write_text(json.dumps(payload))
    return p


def test_missing_when_no_training_run_folder(isolated_models: Path) -> None:
    assert _run() == {"status": "missing"}


def test_missing_when_training_run_folder_has_no_phase7_summary(
    isolated_models: Path,
) -> None:
    (isolated_models / "training_run_20260101T000000Z").mkdir(parents=True)
    # Other random files are fine but phase7_summary.json is the contract.
    (isolated_models / "training_run_20260101T000000Z" / "phase1_preflight.json").write_text("{}")
    assert _run() == {"status": "missing"}


def test_empty_when_summary_has_no_per_slice(isolated_models: Path) -> None:
    _write_summary(
        isolated_models,
        "20260101T000000Z",
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "live_gated_replay": {"per_slice": {}, "run_started_iso": None},
        },
    )
    out = _run()
    assert out["status"] == "empty"
    assert out["run_dir"] == "training_run_20260101T000000Z"
    assert out["per_slice"] == {}


def test_empty_when_summary_omits_live_gated_block(isolated_models: Path) -> None:
    # A campaign that ran before Task #613 added the block writes a
    # phase7_summary.json without `live_gated_replay`. The endpoint must
    # surface that as `empty`, not as an error, so the dashboard can
    # show a friendly footer instead of a red banner.
    _write_summary(
        isolated_models,
        "20260101T000000Z",
        {"generated_at": "2026-01-01T00:00:00+00:00"},
    )
    out = _run()
    assert out["status"] == "empty"
    assert out["per_slice"] == {}


def test_ok_returns_freshest_run(isolated_models: Path) -> None:
    # An older run that pre-dates the live-gated block, then a newer run
    # that has it. The endpoint must pick the newer one.
    _write_summary(
        isolated_models,
        "20260101T000000Z",
        {"live_gated_replay": {"per_slice": {}}},
    )
    populated = {
        "generated_at": "2026-04-29T00:00:00+00:00",
        "live_gated_replay": {
            "run_started_iso": "2026-04-28T00:00:00+00:00",
            "per_slice": {
                "bonk/5m": {
                    "loose_post_fee_pct_total": -1.23,
                    "live_trade_count": 0,
                    "live_net_pnl_pct": None,
                    "dominant_rejection_reason": "directional_edge",
                    "live_replay_status": "ok",
                    "economic_verdict": "dormant",
                    "economic_verdict_phrase": "dormant / no-edge under production gates",
                },
                "pepe/1h": {
                    "loose_post_fee_pct_total": 0.45,
                    "live_trade_count": 12,
                    "live_net_pnl_pct": 0.30,
                    "dominant_rejection_reason": None,
                    "live_replay_status": "ok",
                    "economic_verdict": "tradeable",
                    "economic_verdict_phrase": "tradeable / positive under production gates",
                },
            },
            "verdict_counts": {
                "bleeding": 0,
                "dormant": 1,
                "tradeable": 1,
                "inconclusive": 0,
            },
            "bleeding_slices": [],
            "dormant_slices": ["bonk/5m"],
            "tradeable_slices": ["pepe/1h"],
        },
    }
    _write_summary(isolated_models, "20260429T000000Z", populated)
    out = _run()
    assert out["status"] == "ok"
    assert out["run_dir"] == "training_run_20260429T000000Z"
    assert out["generated_at"] == "2026-04-29T00:00:00+00:00"
    assert out["run_started_iso"] == "2026-04-28T00:00:00+00:00"
    assert set(out["per_slice"].keys()) == {"bonk/5m", "pepe/1h"}
    assert out["per_slice"]["bonk/5m"]["economic_verdict"] == "dormant"
    assert out["per_slice"]["bonk/5m"]["dominant_rejection_reason"] == "directional_edge"
    assert out["per_slice"]["pepe/1h"]["live_net_pnl_pct"] == 0.30
    assert out["verdict_counts"] == {
        "bleeding": 0,
        "dormant": 1,
        "tradeable": 1,
        "inconclusive": 0,
    }
    assert out["dormant_slices"] == ["bonk/5m"]
    assert out["tradeable_slices"] == ["pepe/1h"]


def test_error_when_freshest_summary_is_corrupt(isolated_models: Path) -> None:
    run = isolated_models / "training_run_20260101T000000Z"
    run.mkdir(parents=True)
    (run / "phase7_summary.json").write_text("{not json")
    out = _run()
    assert out["status"] == "error"
    assert out["run_dir"] == "training_run_20260101T000000Z"
    assert "error" in out


def test_only_training_run_dirs_are_considered(isolated_models: Path) -> None:
    # A non-training_run folder with a phase7_summary.json should be
    # ignored — only `training_run_<TS>/` folders are part of the contract.
    other = isolated_models / "scratch"
    other.mkdir(parents=True)
    (other / "phase7_summary.json").write_text(
        json.dumps({"live_gated_replay": {"per_slice": {"x/1h": {}}}})
    )
    assert _run() == {"status": "missing"}
