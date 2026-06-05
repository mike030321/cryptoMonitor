"""Task #490 — nightly Meta-Brain replay scheduler.

Covers the pure-logic surface of `app.scheduled_meta_brain_replay`
without touching Postgres or the heavy `replay_meta_brain.run_replay`
async path:

  1. `_compute_window` clamps `start` to CUTOFF when "now" is close to
     the post-#444 cutover, and slides correctly when it's far past.
  2. `_build_args` produces an argparse.Namespace whose fields match
     the script's defaults / our env overrides; `--commit` is always
     true on the scheduled path.
  3. `_rotate_backups` keeps the N newest `.bak.<ts>` snapshots and
     deletes the rest, ignoring siblings that don't match the suffix.
  4. `_write_last_replay` + `load_last_replay` round-trip the summary,
     and `load_last_replay` returns None when nothing has been written.
  5. `_summarize_manifest` projects only the contracted fields and
     never carries the giant `columns_used` allow-list.
  6. `_tick` honours the `ENABLED=False` short-circuit.
  7. `_tick` records a successful committed run into `state` and writes
     the sidecar; a non-committed (`pipeline_validation_only`) run
     leaves `last_committed=False` and does NOT touch the canonical
     state dir.
  8. `_tick` records a structured error and bumps `errors_total` on
     exception, AND leaves the canonical state dir untouched.
  9. `_tick` reports `skipped_locked_other_process` when another
     replica holds the advisory lock.
 10. The FastAPI app exposes `/ml/meta-brain/last_replay` and the
     body matches the persisted summary.
 11. `start_scheduler` is a no-op when `ML_AUTO_RETRAIN_TEST_DISABLE=1`
     (the same env switch the other ml-engine daemons honour).
"""
from __future__ import annotations

import contextlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import scheduled_meta_brain_replay as sched
from app.main import app
from scripts import replay_meta_brain


# ── 1. window resolver ────────────────────────────────────────────────────
def test_compute_window_clamps_to_cutoff_when_now_is_close():
    """Three days after the cutoff with a 90d window: start MUST clamp
    up to CUTOFF, not slide back into the deleted-LLM era. The replay
    script's own `start < CUTOFF` guard would otherwise SystemExit.
    """
    now = replay_meta_brain.CUTOFF + timedelta(days=3)
    start, end = sched._compute_window(now=now, window_days=90)
    assert start == replay_meta_brain.CUTOFF
    assert end == now


def test_compute_window_slides_when_now_is_far_past_cutoff():
    """A year past the cutoff with a 60d window: start MUST be
    `end - 60d`, not the cutoff (the replay would otherwise consume
    a year of journal rows on every nightly tick).
    """
    now = replay_meta_brain.CUTOFF + timedelta(days=365)
    start, end = sched._compute_window(now=now, window_days=60)
    assert start == now - timedelta(days=60)
    assert start > replay_meta_brain.CUTOFF


# ── 2. args builder ───────────────────────────────────────────────────────
def test_build_args_passes_commit_true_with_script_thresholds():
    start = replay_meta_brain.CUTOFF
    end = start + timedelta(days=45)
    args = sched._build_args(start=start, end=end)
    assert args.commit is True
    assert args.dry_run is False
    assert args.run_id is None
    assert args.sandbox_dir is None
    # Mirrors the script's own defaults so a tweak there propagates.
    assert args.min_trades == replay_meta_brain.DEFAULT_MIN_TRADES
    assert args.min_days == replay_meta_brain.DEFAULT_MIN_DAYS
    assert args.min_regimes == replay_meta_brain.DEFAULT_MIN_REGIMES
    assert args.window_s == replay_meta_brain.DEFAULT_WINDOW_S
    assert args.holdout_pct == replay_meta_brain.DEFAULT_HOLDOUT_PCT
    # ISO strings the script's `parse_iso` accepts.
    assert args.start.endswith("+00:00") or args.start.endswith("Z")
    assert args.end.endswith("+00:00") or args.end.endswith("Z")


# ── 3. backup rotation ────────────────────────────────────────────────────
def _make_backup(tmp_path: Path, ts: str) -> Path:
    """Mirror the directory layout `commit_to_canonical` produces.

    `meta_brain_state/` is the live canonical dir; siblings named
    `meta_brain_state.bak.<ts>` are the rotation snapshots.
    """
    p = tmp_path / f"meta_brain_state.bak.{ts}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "trust_model.json").write_text("{}", encoding="utf-8")
    return p


def test_rotate_backups_keeps_n_newest(tmp_path, monkeypatch):
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    # Five snapshots, oldest → newest by timestamp suffix.
    timestamps = [
        "20260101T000000Z",
        "20260102T000000Z",
        "20260103T000000Z",
        "20260104T000000Z",
        "20260105T000000Z",
    ]
    for ts in timestamps:
        _make_backup(tmp_path, ts)
    # Sibling that doesn't match the suffix MUST be ignored — operators
    # sometimes drop ad-hoc dirs alongside canonical state.
    (tmp_path / "meta_brain_state.manual").mkdir()
    pruned = sched._rotate_backups(keep=2)
    expected_pruned = sorted(
        f"meta_brain_state.bak.{ts}" for ts in timestamps[:3]
    )
    assert sorted(pruned) == expected_pruned
    surviving = sorted(
        p.name for p in tmp_path.iterdir() if p.is_dir()
    )
    assert "meta_brain_state" in surviving
    assert "meta_brain_state.manual" in surviving
    assert "meta_brain_state.bak.20260105T000000Z" in surviving
    assert "meta_brain_state.bak.20260104T000000Z" in surviving
    assert "meta_brain_state.bak.20260103T000000Z" not in surviving


def test_rotate_backups_no_op_when_under_threshold(tmp_path, monkeypatch):
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    _make_backup(tmp_path, "20260105T000000Z")
    pruned = sched._rotate_backups(keep=3)
    assert pruned == []


# ── 4. last_replay sidecar ────────────────────────────────────────────────
def test_last_replay_sidecar_round_trip(tmp_path, monkeypatch):
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    status_dir = tmp_path / "meta_brain_replay_status"
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    monkeypatch.setattr(sched, "STATUS_DIR", status_dir)
    monkeypatch.setattr(
        sched, "LAST_REPLAY_PATH", status_dir / "last_replay.json",
    )
    monkeypatch.setattr(
        sched,
        "LAST_COMMITTED_REPLAY_PATH",
        status_dir / "last_committed_replay.json",
    )
    assert sched.load_last_replay() is None
    payload = {
        "run_id": "replay-test",
        "state": "pipeline_validation_only",
        "commit": False,
    }
    sched._write_last_replay(payload)
    loaded = sched.load_last_replay()
    assert loaded == payload
    # Sidecar lives OUTSIDE canonical so a non-committed run can never
    # mutate the canonical state dir, even with an ignored JSON.
    assert (status_dir / "last_replay.json").exists()
    assert not any(p.name == "last_replay.json" for p in canonical.iterdir())


# ── 5. manifest projection ────────────────────────────────────────────────
def test_summarize_manifest_drops_full_allow_list_and_keeps_contract():
    manifest = {
        "run_id": "x",
        "task": "467",
        "started_at": "2026-04-21T00:00:00+00:00",
        "finished_at": "2026-04-21T00:01:00+00:00",
        "state": "production_ready",
        "commit": True,
        "commit_details": {
            "requested": True,
            "promoted": True,
            "thresholds_satisfied": True,
            "thresholds": {"trades_attributed": 2500, "min_trades": 2000},
            "target": "/x/meta_brain_state",
            "files": ["trust_model.json"],
            "backup": "/x/meta_brain_state.bak.20260421T000000Z",
        },
        "data_window": {"days_covered": 31.0},
        "row_counts": {"cycles_replayed": 9000},
        "regimes_observed": {"trend_up": 50, "range_chop": 60},
        "families_observed": {"momentum": 200},
        "cycles_replayed": 9000,
        "trades_attributed": 2500,
        "trades_unmatched": 100,
        "trades_skipped_holdout": 50,
        "holdout_metrics": {"avg_reward_proxy": 0.5},
        "forbidden_columns_seen": [],
        # Fields we INTENTIONALLY do not echo into the sidecar — they're
        # large and not interesting for the ops endpoint.
        "columns_used": {"feature_vector_keys": list(range(500))},
        "constants": {"PREDICTION_FLEET_RESET_AT": "2026-04-21T00:00:00+00:00"},
    }
    summary = sched._summarize_manifest(manifest, sandbox_dir="/sb")
    assert "columns_used" not in summary
    assert "constants" not in summary
    assert summary["run_id"] == "x"
    assert summary["commit"] is True
    assert summary["commit_details"]["promoted"] is True
    assert summary["data_window"]["days_covered"] == 31.0
    assert summary["sandbox_dir"] == "/sb"


# ── 6 / 11. ENABLED + test-disable short-circuits ─────────────────────────
def test_tick_disabled_short_circuit(monkeypatch):
    monkeypatch.setattr(sched, "ENABLED", False)
    out = sched._tick()
    assert out == "disabled"
    assert sched.state["last_attempt_outcome"] == "disabled"


def test_start_scheduler_no_op_when_test_disabled(monkeypatch):
    """Mirrors the contract for every other ml-engine daemon: the
    suite-wide `ML_AUTO_RETRAIN_TEST_DISABLE=1` env switch must keep
    the polling thread from ever spawning.
    """
    monkeypatch.setattr(sched, "ENABLED", True)
    monkeypatch.setenv("ML_AUTO_RETRAIN_TEST_DISABLE", "1")
    sched._thread = None
    sched.start_scheduler()
    assert sched._thread is None


# ── helpers for ticks that drive `_run_once_blocking` ─────────────────────
def _stub_run_once(monkeypatch, result: dict) -> None:
    monkeypatch.setattr(sched, "_run_once_blocking", lambda: result)


def _committed_summary(**overrides) -> dict:
    base = {
        "run_id": "replay-ok",
        "task": "467",
        "started_at": "2026-04-21T00:00:00+00:00",
        "finished_at": "2026-04-21T00:01:00+00:00",
        "state": "production_ready",
        "commit": True,
        "commit_details": {
            "requested": True,
            "promoted": True,
            "thresholds_satisfied": True,
            "thresholds": {"trades_attributed": 2500, "min_trades": 2000},
            "target": "/x/meta_brain_state",
            "files": ["trust_model.json", "regime_memory.json"],
            "backup": "/x/meta_brain_state.bak.20260421T000000Z",
        },
        "data_window": {"days_covered": 31.0},
        "row_counts": {"cycles_replayed": 9000},
        "regimes_observed": {"trend_up": 50, "range_chop": 60, "vol_spike": 5},
        "families_observed": {"momentum": 200},
        "cycles_replayed": 9000,
        "trades_attributed": 2500,
        "trades_unmatched": 100,
        "trades_skipped_holdout": 50,
        "holdout_metrics": {"avg_reward_proxy": 0.5},
        "forbidden_columns_seen": [],
        "sandbox_dir": "/x/sandbox/replay-ok",
    }
    base.update(overrides)
    return base


def _reset_state() -> None:
    sched.state.update(
        {
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
    )


# ── 7. successful committed run + non-committed run ──────────────────────
def test_tick_records_committed_run(monkeypatch, tmp_path):
    monkeypatch.setattr(sched, "ENABLED", True)
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    status_dir = tmp_path / "meta_brain_replay_status"
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    monkeypatch.setattr(sched, "STATUS_DIR", status_dir)
    monkeypatch.setattr(
        sched, "LAST_REPLAY_PATH", status_dir / "last_replay.json",
    )
    monkeypatch.setattr(
        sched,
        "LAST_COMMITTED_REPLAY_PATH",
        status_dir / "last_committed_replay.json",
    )
    _reset_state()
    summary = _committed_summary()
    _stub_run_once(
        monkeypatch,
        {
            "skipped_locked": False,
            "no_window": False,
            "summary": summary,
            "pruned_backups": [],
        },
    )
    out = sched._tick()
    assert out == "ok"
    assert sched.state["last_committed"] is True
    assert sched.state["last_run_id"] == "replay-ok"
    assert sched.state["last_run_state"] == "production_ready"
    assert sched.state["runs_total"] == 1
    assert sched.state["commits_total"] == 1
    sidecar = json.loads(
        (status_dir / "last_replay.json").read_text(encoding="utf-8")
    )
    assert sidecar["run_id"] == "replay-ok"
    # Sidecar must NOT land in canonical — it's an ops artifact, not a
    # supervisory-brain state file.
    assert not (canonical / "last_replay.json").exists()


def test_tick_non_committed_leaves_canonical_untouched(monkeypatch, tmp_path):
    """A run that doesn't satisfy the commit thresholds must NOT touch
    the canonical state. The replay script enforces this on its own
    branch; the scheduler must not re-promote behind its back.
    """
    monkeypatch.setattr(sched, "ENABLED", True)
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    (canonical / "trust_model.json").write_text(
        '{"momentum": {"trust": 1.0}}', encoding="utf-8"
    )
    canonical_mtime_before = (
        canonical / "trust_model.json"
    ).stat().st_mtime_ns
    canonical_listing_before = sorted(p.name for p in canonical.iterdir())
    status_dir = tmp_path / "meta_brain_replay_status"
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    monkeypatch.setattr(sched, "STATUS_DIR", status_dir)
    monkeypatch.setattr(
        sched, "LAST_REPLAY_PATH", status_dir / "last_replay.json",
    )
    _reset_state()
    summary = _committed_summary(
        commit=False,
        state="pipeline_validation_only",
        commit_details={
            "requested": True,
            "promoted": False,
            "thresholds_satisfied": False,
            "thresholds": {"trades_attributed": 5, "min_trades": 2000},
            "target": str(canonical),
            "reason": "thresholds_not_met",
        },
    )
    _stub_run_once(
        monkeypatch,
        {
            "skipped_locked": False,
            "no_window": False,
            "summary": summary,
            "pruned_backups": [],
        },
    )
    out = sched._tick()
    assert out == "ok"
    assert sched.state["last_committed"] is False
    assert sched.state["commits_total"] == 0
    assert sched.state["runs_total"] == 1
    # Canonical trust_model.json was NOT touched by the scheduler, AND
    # nothing new was added to the canonical dir (the sidecar lives in
    # the sibling status dir so a non-promoted run can never mutate
    # canonical at the path level).
    canonical_mtime_after = (
        canonical / "trust_model.json"
    ).stat().st_mtime_ns
    assert canonical_mtime_after == canonical_mtime_before
    canonical_listing_after = sorted(p.name for p in canonical.iterdir())
    assert canonical_listing_after == canonical_listing_before
    # Sidecar landed in the ops dir.
    assert (status_dir / "last_replay.json").exists()


# ── 8. exception path ────────────────────────────────────────────────────
def test_tick_records_error_and_leaves_canonical_untouched(
    monkeypatch, tmp_path, caplog,
):
    monkeypatch.setattr(sched, "ENABLED", True)
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    (canonical / "trust_model.json").write_text(
        '{"momentum": {"trust": 1.0}}', encoding="utf-8"
    )
    canonical_mtime_before = (
        canonical / "trust_model.json"
    ).stat().st_mtime_ns
    status_dir = tmp_path / "meta_brain_replay_status"
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    monkeypatch.setattr(sched, "STATUS_DIR", status_dir)
    monkeypatch.setattr(
        sched, "LAST_REPLAY_PATH", status_dir / "last_replay.json",
    )
    _reset_state()

    def _raise() -> dict:
        raise RuntimeError("simulated_db_outage")

    monkeypatch.setattr(sched, "_run_once_blocking", _raise)
    with caplog.at_level("WARNING", logger="ml-engine.meta-brain-replay"):
        out = sched._tick()
    assert out == "error"
    assert sched.state["last_attempt_outcome"] == "error"
    assert sched.state["last_error"] == "simulated_db_outage"
    assert sched.state["errors_total"] == 1
    # Structured failure log line — the spec requires it.
    assert any(
        "meta_brain_replay_tick_failed" in r.message for r in caplog.records
    )
    canonical_mtime_after = (
        canonical / "trust_model.json"
    ).stat().st_mtime_ns
    assert canonical_mtime_after == canonical_mtime_before
    # Sidecar must NOT have been written on the failure branch — neither
    # in canonical nor the ops dir — because we never reached the
    # success path that calls `_write_last_replay`.
    assert not (canonical / "last_replay.json").exists()
    assert not (status_dir / "last_replay.json").exists()


# ── 9. advisory-lock skip ────────────────────────────────────────────────
def test_tick_skipped_when_other_replica_holds_lock(monkeypatch):
    monkeypatch.setattr(sched, "ENABLED", True)
    _reset_state()
    _stub_run_once(monkeypatch, {"skipped_locked": True})
    out = sched._tick()
    assert out == "skipped_locked_other_process"
    assert sched.state["skips_locked_total"] == 1
    assert sched.state["runs_total"] == 0
    assert sched.state["commits_total"] == 0


# ── 10. /ml/meta-brain/last_replay endpoint ──────────────────────────────
def test_last_replay_endpoint_no_replay_yet(monkeypatch, tmp_path):
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    status_dir = tmp_path / "meta_brain_replay_status"
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    monkeypatch.setattr(sched, "STATUS_DIR", status_dir)
    monkeypatch.setattr(
        sched, "LAST_REPLAY_PATH", status_dir / "last_replay.json",
    )
    monkeypatch.setattr(
        sched,
        "LAST_COMMITTED_REPLAY_PATH",
        status_dir / "last_committed_replay.json",
    )
    client = TestClient(app)
    r = client.get("/ml/meta-brain/last_replay")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["reason"] == "no_replay_yet"


def test_last_replay_endpoint_returns_persisted_summary(
    monkeypatch, tmp_path,
):
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    status_dir = tmp_path / "meta_brain_replay_status"
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    monkeypatch.setattr(sched, "STATUS_DIR", status_dir)
    monkeypatch.setattr(
        sched, "LAST_REPLAY_PATH", status_dir / "last_replay.json",
    )
    monkeypatch.setattr(
        sched,
        "LAST_COMMITTED_REPLAY_PATH",
        status_dir / "last_committed_replay.json",
    )
    summary = _committed_summary()
    sched._write_last_replay(summary)
    client = TestClient(app)
    r = client.get("/ml/meta-brain/last_replay")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # Dual-pointer contract: a committed run populates both.
    assert body["last_run"]["run_id"] == "replay-ok"
    assert body["last_run"]["commit"] is True
    assert body["last_committed_run"]["run_id"] == "replay-ok"
    # Backwards-compatible alias.
    assert body["summary"] == body["last_run"]
    assert "scheduler" in body
    assert body["scheduler"]["window_days"] == sched.WINDOW_DAYS


def test_last_replay_endpoint_dual_pointer_split_when_committed_then_below_threshold(
    monkeypatch, tmp_path,
):
    """The dual-pointer contract: after a committed run we then have a
    below-threshold run, ``last_run`` advances to the new
    pipeline-validation summary while ``last_committed_run`` still
    reflects the genuinely promoted manifest. Without this, an
    operator looking at the endpoint after a single below-threshold
    tick would see "below threshold" and have no way to recover what
    the canonical state dir was actually warmed from.
    """
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    status_dir = tmp_path / "meta_brain_replay_status"
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    monkeypatch.setattr(sched, "STATUS_DIR", status_dir)
    monkeypatch.setattr(
        sched, "LAST_REPLAY_PATH", status_dir / "last_replay.json",
    )
    monkeypatch.setattr(
        sched,
        "LAST_COMMITTED_REPLAY_PATH",
        status_dir / "last_committed_replay.json",
    )
    # First: a committed run lands.
    sched._write_last_replay(
        _committed_summary(run_id="replay-promoted-1")
    )
    # Then: a below-threshold run lands the very next tick.
    sched._write_last_replay(
        _committed_summary(
            run_id="replay-not-yet-2",
            commit=False,
            state="pipeline_validation_only",
            commit_details={
                "requested": True,
                "promoted": False,
                "thresholds_satisfied": False,
                "thresholds": {"trades_attributed": 5, "min_trades": 2000},
                "target": str(canonical),
                "reason": "thresholds_not_met",
            },
        ),
    )
    client = TestClient(app)
    r = client.get("/ml/meta-brain/last_replay")
    body = r.json()
    assert body["ok"] is True
    assert body["last_run"]["run_id"] == "replay-not-yet-2"
    assert body["last_run"]["commit"] is False
    # The committed pointer still reflects the genuinely promoted run
    # — that's the manifest the live canonical state is running on.
    assert body["last_committed_run"]["run_id"] == "replay-promoted-1"
    assert body["last_committed_run"]["commit"] is True


def test_load_last_committed_replay_none_when_only_below_threshold_runs(
    monkeypatch, tmp_path,
):
    """A daemon that has only ever produced below-threshold runs must
    leave ``last_committed_replay.json`` absent so the endpoint clearly
    reports "the gates have never been satisfied".
    """
    canonical = tmp_path / "meta_brain_state"
    canonical.mkdir()
    status_dir = tmp_path / "meta_brain_replay_status"
    monkeypatch.setattr(sched, "CANONICAL_STATE_DIR", canonical)
    monkeypatch.setattr(sched, "STATUS_DIR", status_dir)
    monkeypatch.setattr(
        sched, "LAST_REPLAY_PATH", status_dir / "last_replay.json",
    )
    monkeypatch.setattr(
        sched,
        "LAST_COMMITTED_REPLAY_PATH",
        status_dir / "last_committed_replay.json",
    )
    sched._write_last_replay(
        _committed_summary(
            commit=False,
            state="pipeline_validation_only",
            commit_details={
                "requested": True,
                "promoted": False,
                "thresholds_satisfied": False,
                "thresholds": {"trades_attributed": 5, "min_trades": 2000},
                "target": str(canonical),
                "reason": "thresholds_not_met",
            },
        ),
    )
    assert sched.load_last_replay() is not None
    assert sched.load_last_committed_replay() is None
    assert not (status_dir / "last_committed_replay.json").exists()


# ── advisory-lock plumbing actually wires through `_run_once_async` ──────
def test_run_once_async_short_circuits_on_lock_loss(monkeypatch):
    """Verifies the cross-replica lock path: when `try_advisory_lock`
    yields False, we MUST NOT call `replay_meta_brain.run_replay`.
    """
    called = {"n": 0}

    @contextlib.asynccontextmanager
    async def _fake_lock(_key):
        yield False

    async def _fake_run_replay(_args):  # pragma: no cover - must not run
        called["n"] += 1
        return {"manifest": {}, "sandbox_dir": "/x"}

    monkeypatch.setattr(sched.db_mod, "try_advisory_lock", _fake_lock)
    monkeypatch.setattr(replay_meta_brain, "run_replay", _fake_run_replay)
    import asyncio
    out = asyncio.run(sched._run_once_async())
    assert out == {"skipped_locked": True}
    assert called["n"] == 0
