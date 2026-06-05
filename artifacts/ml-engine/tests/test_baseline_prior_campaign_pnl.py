"""Task #520 — pre-run baseline archives every prior-run slice's PnL.

The campaign archives `models/_archive/<TS>_pre_full_run/baseline_snapshot.json`
before each new run. Historically that snapshot's `per_slice` block came
from the prior `report.json`, which after a partial run only carried 2-4
slices. Task #516 needed every prior 1d/6h slice's `pnl_after_fees` and
could not get it. The fix: harvest every `slice_done` event from the
prior campaign's window in `progress_updates.jsonl` and pin it under
`prior_campaign_per_slice_pnl`. These tests pin down the harvest
contract so the next "did PnL regress?" check stays a one-line diff.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def harvest():
    """Load `_harvest_prior_campaign_slice_pnl` without importing the
    full orchestrator (which pulls in the trainer + asyncpg pool).
    """
    src_path = Path(__file__).resolve().parents[1] / "scripts" / "run_full_training_campaign.py"
    src = src_path.read_text()
    import re

    m = re.search(
        r"(def _harvest_prior_campaign_slice_pnl\(.*?\n)(?=\n\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert m, "_harvest_prior_campaign_slice_pnl not found in orchestrator source"
    ns = {
        "Path": Path,
        "json": json,
        "Any": object,
        "Optional": type(None),
        "PROGRESS_PATH": Path("/tmp/__nonexistent__.jsonl"),
    }
    exec(m.group(0), ns)
    return ns["_harvest_prior_campaign_slice_pnl"]


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _slice_done(coin: str, tf: str, *, net: float, n: int, ts: str) -> dict:
    return {
        "emitted_at": ts,
        "phase": "slice_done",
        "status": "trained",
        "headline": f"{coin}/{tf}",
        "coin": coin,
        "timeframe": tf,
        "metrics": {"auc": 0.55, "directional_accuracy": 0.4},
        "baseline_metrics": {"auc": 0.50},
        "lift_auc": 0.05,
        "pnl_after_fees": {
            "n_trades": n,
            "net_pct_total": net,
            "net_pct_mean": net / max(n, 1),
            "gross_pct_mean": 0.0,
            "win_rate": 0.4,
            "trade_share": 0.5,
            "round_trip_cost_pct": 0.3,
        },
        "n_rows": 1000,
    }


def test_harvest_returns_empty_when_log_missing(harvest, tmp_path):
    out = harvest(tmp_path / "absent.jsonl")
    assert out["per_slice"] == {}
    assert out["slice_count"] == 0
    assert out["completed"] is False
    assert out["prior_run_dir"] is None


def test_harvest_returns_empty_when_only_current_campaign_started(harvest, tmp_path):
    """Bootstrap: only the *current* campaign_start exists. No prior."""
    log = tmp_path / "p.jsonl"
    _write_jsonl(log, [
        {"emitted_at": "2026-04-25T06:33:02+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_X"},
        # No slice_done yet because phase 3 runs before phase 4.
    ])
    out = harvest(log)
    assert out["slice_count"] == 0
    assert out["completed"] is False


def test_harvest_picks_up_completed_prior_campaign(harvest, tmp_path):
    log = tmp_path / "p.jsonl"
    _write_jsonl(log, [
        # Prior completed campaign.
        {"emitted_at": "2026-04-20T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_PRIOR"},
        _slice_done("bonk", "1h", net=-12.5, n=100, ts="2026-04-20T01:00:00+00:00"),
        _slice_done("pepe", "1d", net=42.0, n=200, ts="2026-04-20T02:00:00+00:00"),
        {"emitted_at": "2026-04-20T03:00:00+00:00", "phase": "campaign_done"},
        # Current campaign just started — phase 3 runs now, no slice_done yet.
        {"emitted_at": "2026-04-25T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_CURRENT"},
    ])
    out = harvest(log)
    assert out["completed"] is True
    assert out["prior_run_dir"] == "models/training_run_PRIOR"
    assert out["prior_campaign_started_at"] == "2026-04-20T00:00:00+00:00"
    assert out["prior_campaign_finished_at"] == "2026-04-20T03:00:00+00:00"
    assert out["slice_count"] == 2
    assert set(out["per_slice"].keys()) == {"bonk/1h", "pepe/1d"}
    bonk = out["per_slice"]["bonk/1h"]
    assert bonk["n_trades"] == 100
    assert bonk["net_pct_total"] == pytest.approx(-12.5)
    assert bonk["status"] == "trained"
    assert bonk["coin"] == "bonk"
    assert bonk["timeframe"] == "1h"


def test_harvest_keeps_last_slice_done_when_slice_retried(harvest, tmp_path):
    log = tmp_path / "p.jsonl"
    _write_jsonl(log, [
        {"emitted_at": "2026-04-20T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_PRIOR"},
        _slice_done("bonk", "1h", net=-99.9, n=1, ts="2026-04-20T01:00:00+00:00"),
        _slice_done("bonk", "1h", net=-12.5, n=100, ts="2026-04-20T02:00:00+00:00"),
        {"emitted_at": "2026-04-20T03:00:00+00:00", "phase": "campaign_done"},
        {"emitted_at": "2026-04-25T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_CURRENT"},
    ])
    out = harvest(log)
    assert out["slice_count"] == 1
    assert out["per_slice"]["bonk/1h"]["net_pct_total"] == pytest.approx(-12.5)
    assert out["per_slice"]["bonk/1h"]["n_trades"] == 100


def test_harvest_falls_back_to_interrupted_prior_campaign(harvest, tmp_path):
    """If the prior campaign never emitted `campaign_done` (e.g. process
    was killed), still capture its `slice_done` rows up to the current
    campaign_start so an operator can compare partial coverage.
    """
    log = tmp_path / "p.jsonl"
    _write_jsonl(log, [
        {"emitted_at": "2026-04-20T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_PRIOR"},
        _slice_done("bonk", "1h", net=-12.5, n=100, ts="2026-04-20T01:00:00+00:00"),
        # No campaign_done — the prior run was interrupted.
        {"emitted_at": "2026-04-25T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_CURRENT"},
    ])
    out = harvest(log)
    assert out["completed"] is False
    assert out["prior_run_dir"] == "models/training_run_PRIOR"
    assert out["slice_count"] == 1
    assert "bonk/1h" in out["per_slice"]


def test_harvest_prefers_completed_over_more_recent_interrupted(harvest, tmp_path):
    """Selection policy: when both a fully-bracketed prior campaign and
    a *more recent* interrupted campaign exist before the current
    `campaign_start`, the harvester picks the completed one. Pin that
    explicitly so a future change can't silently flip the policy.
    """
    log = tmp_path / "p.jsonl"
    _write_jsonl(log, [
        # Completed prior campaign (older).
        {"emitted_at": "2026-04-10T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_OLD"},
        _slice_done("bonk", "1h", net=-12.5, n=100, ts="2026-04-10T01:00:00+00:00"),
        {"emitted_at": "2026-04-10T03:00:00+00:00", "phase": "campaign_done"},
        # Newer but interrupted campaign — never emitted campaign_done.
        {"emitted_at": "2026-04-20T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_INTERRUPTED"},
        _slice_done("pepe", "1h", net=999.0, n=1, ts="2026-04-20T01:00:00+00:00"),
        # Current campaign — phase 3 just emitted this.
        {"emitted_at": "2026-04-25T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_CURRENT"},
    ])
    out = harvest(log)
    assert out["completed"] is True
    assert out["prior_run_dir"] == "models/training_run_OLD"
    # The interrupted run's slice_done must not contaminate the harvest.
    assert set(out["per_slice"].keys()) == {"bonk/1h"}


def test_harvest_skips_events_outside_prior_window(harvest, tmp_path):
    """`slice_done` events that belong to the *current* (about-to-start)
    campaign or to a maintenance loop *between* campaigns must not leak
    into the prior-campaign harvest.
    """
    log = tmp_path / "p.jsonl"
    _write_jsonl(log, [
        {"emitted_at": "2026-04-20T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_PRIOR"},
        _slice_done("bonk", "1h", net=-12.5, n=100, ts="2026-04-20T01:00:00+00:00"),
        {"emitted_at": "2026-04-20T03:00:00+00:00", "phase": "campaign_done"},
        # Maintenance retrain between campaigns (won't happen in real
        # life because slice_done is gated on the in-process hook, but
        # belt-and-suspenders).
        _slice_done("ghost", "1h", net=999.0, n=1, ts="2026-04-22T00:00:00+00:00"),
        {"emitted_at": "2026-04-25T00:00:00+00:00",
         "phase": "campaign_start", "run_dir": "models/training_run_CURRENT"},
        # Slice done in the new campaign — must not leak in either.
        _slice_done("ghost2", "1h", net=999.0, n=1, ts="2026-04-25T01:00:00+00:00"),
    ])
    out = harvest(log)
    assert set(out["per_slice"].keys()) == {"bonk/1h"}
