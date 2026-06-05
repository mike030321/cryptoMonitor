"""Task #613 — single-resume watchdog + per-slice live-gated replay.

These tests pin three behaviours added by the task:

  1. Under `ML_WATCHDOG_SINGLE_RESUME=1`, the SECOND watchdog halt
     writes a halt-and-report markdown and exits cleanly with
     `sys.exit(0)`. The first halt still waits like the legacy mode.
  2. The live-gated replay hook tolerates subprocess failures and
     reports them as `live_replay_status="error"` without ever
     raising — so the campaign cannot be blocked by a diagnostic
     regression.
  3. The bonk/5m back-fill from the existing post-fee diagnostic is
     idempotent — calling it twice writes exactly one
     `slice_live_replay_backfill` row.

The watchdog and live-replay helpers are unit-tested directly without
spinning up the full Phase-4 driver; we monkey-patch the orchestrator
module's `ROOT`, `REGISTRY_ROOT`, and `PROGRESS_PATH` to a tmp dir.
"""
from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest


def _load_campaign_module():
    return importlib.import_module("scripts.run_full_training_campaign")


def _redirect_to_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the orchestrator's filesystem-rooted globals at tmp_path so
    the unit under test never writes to the real registry / progress
    log."""
    mod = _load_campaign_module()
    monkeypatch.setattr(mod, "ROOT", tmp_path, raising=True)
    monkeypatch.setattr(mod, "REGISTRY_ROOT", tmp_path / "models", raising=True)
    monkeypatch.setattr(
        mod, "PROGRESS_PATH", tmp_path / "models" / "progress_updates.jsonl",
        raising=True,
    )
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "diagnostics").mkdir(parents=True, exist_ok=True)
    return mod


# ── Triggering-slice parser ────────────────────────────────────────────
def test_parse_triggering_slice_extracts_slug_or_returns_none():
    mod = _load_campaign_module()
    assert mod._parse_triggering_slice("slice_done:bonk/5m") == "bonk/5m"
    assert mod._parse_triggering_slice("slice_done:") is None
    assert mod._parse_triggering_slice("heartbeat:1800s_quiet") is None
    assert mod._parse_triggering_slice("") is None
    assert mod._parse_triggering_slice(None) is None  # type: ignore[arg-type]


# ── Economic-verdict mapping ────────────────────────────────────────────
def test_economic_verdict_table():
    mod = _load_campaign_module()
    # bleeding: loose<0, live n>=5, live<0
    assert mod._economic_verdict(-50.0, 12, -3.4) == "bleeding"
    # dormant: loose<0, live n<5
    assert mod._economic_verdict(-86.3, 1, 0.96) == "dormant"
    assert mod._economic_verdict(-86.3, 0, None) == "dormant"
    # tradeable: live>0 AND live n>=5 (regardless of loose)
    assert mod._economic_verdict(-10.0, 8, 1.2) == "tradeable"
    assert mod._economic_verdict(2.0, 6, 0.5) == "tradeable"
    # inconclusive: signals don't agree, or no observation
    assert mod._economic_verdict(None, None, None) == "inconclusive"
    # loose>0 AND live n<5 → inconclusive (live can't confirm)
    assert mod._economic_verdict(5.0, 2, None) == "inconclusive"
    # loose<0 AND live n>=5 AND live>0 — disagreement → inconclusive
    # (positive live PnL on >=5 trades wins the `tradeable` rule)
    assert mod._economic_verdict(-10.0, 5, 0.1) == "tradeable"


# ── Single-resume halt-and-exit ─────────────────────────────────────────
def test_first_halt_does_not_exit_under_single_resume_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """First halt under single-resume mode still consumes the resume
    via the env var path, increments `total_resumes`, and returns
    normally — only the SECOND halt should sys.exit(0)."""
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    monkeypatch.setenv("ML_WATCHDOG_SINGLE_RESUME", "1")
    monkeypatch.setenv("ML_WATCHDOG_RESUME", "1")  # immediate resume
    monkeypatch.setenv("ML_WATCHDOG_MAX_HALT_SEC", "5")
    state = {"total_resumes": 0, "consecutive_regress": 2}
    # Should NOT raise SystemExit — first halt is allowed to wait+resume
    mod._handle_watchdog_halt("slice_done:fake/5m", state)
    assert state["total_resumes"] == 1
    assert state["consecutive_regress"] == 0
    # No halt-report markdown is written on the first halt
    assert list((tmp_path / "diagnostics").glob("campaign_halt_*")) == []


def test_second_halt_writes_report_and_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    monkeypatch.setenv("ML_WATCHDOG_SINGLE_RESUME", "1")
    state = {
        "total_resumes": 1,
        "consecutive_regress": 2,
        "slices_done": [
            {
                "slice": "pepe/5m", "coin": "pepe", "timeframe": "5m",
                "status": "trained", "post_fee_pct_total": -3.4,
                "post_fee_n_trades": 100, "live_trade_count": 8,
                "live_net_pnl_pct": -1.2, "dominant_rejection_reason": None,
                "live_replay_status": "ok", "live_replay_error": None,
            },
            {
                "slice": "bonk/5m", "coin": "bonk", "timeframe": "5m",
                "status": "trained", "post_fee_pct_total": -86.35,
                "post_fee_n_trades": 200, "live_trade_count": 1,
                "live_net_pnl_pct": 0.965,
                "dominant_rejection_reason": "abstain_no_directional_edge",
                "live_replay_status": "ok", "live_replay_error": None,
            },
        ],
        "recent_snapshots": [
            {
                "at": "2026-04-29T11:00:00+00:00",
                "reason": "slice_done:pepe/5m",
                "watchdog_verdict": "regressed",
                "consecutive_regress": 1,
                "best_slice": "pepe/5m", "worst_slice": "bonk/5m",
                "halted": False,
            },
            {
                "at": "2026-04-29T11:05:00+00:00",
                "reason": "slice_done:bonk/5m",
                "watchdog_verdict": "regressed",
                "consecutive_regress": 2,
                "best_slice": "pepe/5m", "worst_slice": "bonk/5m",
                "halted": True,
            },
        ],
        "planned_slices": ["bonk/5m", "pepe/5m", "celestia/5m"],
    }
    with pytest.raises(SystemExit) as ei:
        mod._handle_watchdog_halt("slice_done:bonk/5m", state)
    assert ei.value.code == 0
    halt_dirs = sorted((tmp_path / "diagnostics").glob("campaign_halt_*"))
    assert len(halt_dirs) == 1
    report_path = halt_dirs[0] / "REPORT.md"
    assert report_path.exists()
    body = report_path.read_text()
    assert "Task #613" in body
    assert "slice_done:bonk/5m" in body
    assert "Total operator/auto-timeout resumes consumed:** 1" in body
    # Triggering-slice block carries the bonk/5m loose+live numbers and
    # the verdict phrase the task brief mandates.
    assert "## Triggering slice" in body
    assert "`bonk/5m`" in body
    assert "-86.3500" in body  # loose pct_total
    assert "abstain_no_directional_edge" in body  # dominant rejection
    assert "dormant / no-edge under production gates" in body
    # Last-3 snapshot trail and trained/pending lists are present.
    assert "## Last 3 snapshot trail" in body
    assert "regressed" in body
    assert "## Trained slices so far (2)" in body
    assert "## Pending slices (1)" in body
    assert "celestia/5m" in body
    assert "Slices trained this run:** 2 of 3 planned (1 pending)" in body
    # progress_updates.jsonl carries a `halted_final` row pointing at
    # the report path — the operator should be able to grep one line to
    # find the report.
    progress_lines = (
        (tmp_path / "models" / "progress_updates.jsonl").read_text()
        .strip().splitlines()
    )
    halt_rows = [
        json.loads(l) for l in progress_lines
        if '"halted_final"' in l
    ]
    assert len(halt_rows) == 1
    row = halt_rows[0]
    assert row["phase"] == "watchdog_halt"
    assert row["status"] == "halted_final"
    assert row["single_resume_mode"] is True
    assert row["total_resumes_consumed"] == 1
    assert row["halt_report_path"].endswith("REPORT.md")


def test_single_resume_disabled_falls_back_to_legacy_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Without the env var, even a 5th halt should NOT exit — legacy
    behaviour (auto-resume after the bounded wait) is preserved."""
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    monkeypatch.delenv("ML_WATCHDOG_SINGLE_RESUME", raising=False)
    monkeypatch.setenv("ML_WATCHDOG_RESUME", "1")
    monkeypatch.setenv("ML_WATCHDOG_MAX_HALT_SEC", "5")
    state = {"total_resumes": 4, "consecutive_regress": 3}
    mod._handle_watchdog_halt("slice_done:bonk/5m", state)  # no SystemExit
    assert state["total_resumes"] == 5
    assert list((tmp_path / "diagnostics").glob("campaign_halt_*")) == []


# ── Live-replay subprocess error fallback ────────────────────────────────
def test_live_replay_returns_skipped_for_pooled_or_specialist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    for coin in ("__pooled__", "__specialist_momentum__"):
        res = mod._run_live_replay(coin, "5m")
        assert res["live_replay_status"] == "skipped"
        assert res["live_trade_count"] is None
        assert res["live_net_pnl_pct"] is None


def test_live_replay_returns_skipped_when_no_latest_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    res = mod._run_live_replay("bonk", "5m")
    assert res["live_replay_status"] == "skipped"
    assert "no `latest`" in (res["live_replay_error"] or "")


def test_live_replay_subprocess_nonzero_exit_yields_error_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    coin_dir = tmp_path / "models" / "bonk" / "5m"
    coin_dir.mkdir(parents=True)
    (coin_dir / "latest").write_text("v_test\n")

    class _FakeProc:
        returncode = 2
        stdout = ""
        stderr = "fake-boom: holdout drift"

    def _fake_run(*args, **kwargs):
        return _FakeProc()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    res = mod._run_live_replay("bonk", "5m")
    assert res["live_replay_status"] == "error"
    assert "fake-boom" in (res["live_replay_error"] or "")
    assert res["live_trade_count"] is None
    assert res["live_net_pnl_pct"] is None


def test_live_replay_subprocess_timeout_yields_timeout_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    coin_dir = tmp_path / "models" / "bonk" / "5m"
    coin_dir.mkdir(parents=True)
    (coin_dir / "latest").write_text("v_test\n")

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=120)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    res = mod._run_live_replay("bonk", "5m")
    assert res["live_replay_status"] == "timeout"
    assert "120s" in (res["live_replay_error"] or "")


def test_live_replay_subprocess_unparseable_stdout_yields_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    coin_dir = tmp_path / "models" / "bonk" / "5m"
    coin_dir.mkdir(parents=True)
    (coin_dir / "latest").write_text("v_test\n")

    class _FakeProc:
        returncode = 0
        stdout = "not json"
        stderr = ""

    def _fake_run(*args, **kwargs):
        return _FakeProc()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    res = mod._run_live_replay("bonk", "5m")
    assert res["live_replay_status"] == "error"
    assert "parse subprocess output" in (res["live_replay_error"] or "")


def test_live_replay_subprocess_success_extracts_aggregate_and_dominant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    coin_dir = tmp_path / "models" / "bonk" / "5m"
    coin_dir.mkdir(parents=True)
    (coin_dir / "latest").write_text("v_test\n")
    diag_dir = tmp_path / "diagnostics" / "bonk_5m_post_fee_FAKE"
    diag_dir.mkdir(parents=True)
    (diag_dir / "summary.json").write_text(json.dumps({
        "aggregate": {"n_trades": 3, "net_pct_total": -1.234},
        "trade_distribution": {
            "skip_reason_counts": {
                "abstain_no_directional_edge": 999,
                "fee_gate_ev": 4,
            },
        },
    }))

    class _FakeProc:
        returncode = 0
        stderr = ""
        stdout = json.dumps({
            "output_dir": str(diag_dir),
            "n_trades": 3,
            "net_pct_total": -1.234,
            "win_rate": 0.0,
        })

    def _fake_run(*args, **kwargs):
        return _FakeProc()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    res = mod._run_live_replay("bonk", "5m")
    assert res["live_replay_status"] == "ok"
    assert res["live_trade_count"] == 3
    assert res["live_net_pnl_pct"] == -1.234
    assert res["dominant_rejection_reason"] == "abstain_no_directional_edge"


# ── Bonk/5m back-fill is idempotent ─────────────────────────────────────
def test_bonk_5m_backfill_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    diag_dir = (
        tmp_path / "diagnostics" / mod._BONK_5M_BACKFILL_DIAGNOSTIC_DIR
    )
    diag_dir.mkdir(parents=True)
    (diag_dir / "summary.json").write_text(json.dumps({
        "aggregate": {"n_trades": 1, "net_pct_total": 0.965},
        "trade_distribution": {
            "skip_reason_counts": {
                "abstain_no_directional_edge": 15433,
                "abstain_low_directional_prob": 2507,
            },
        },
    }))
    # First call: appends a row
    rec1 = mod._backfill_bonk_5m_diagnostic()
    assert rec1 is not None
    assert rec1["coin"] == "bonk"
    assert rec1["timeframe"] == "5m"
    assert rec1["live_trade_count"] == 1
    assert rec1["live_net_pnl_pct"] == pytest.approx(0.965)
    assert rec1["dominant_rejection_reason"] == "abstain_no_directional_edge"
    # Second call: no-op (returns None)
    rec2 = mod._backfill_bonk_5m_diagnostic()
    assert rec2 is None
    # Exactly one slice_live_replay_backfill row in the progress log
    progress_path = tmp_path / "models" / "progress_updates.jsonl"
    rows = [
        json.loads(l) for l in progress_path.read_text().strip().splitlines()
        if l.strip()
    ]
    backfill_rows = [
        r for r in rows if r.get("phase") == "slice_live_replay_backfill"
    ]
    assert len(backfill_rows) == 1


def test_bonk_5m_backfill_is_no_op_when_diagnostic_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    rec = mod._backfill_bonk_5m_diagnostic()
    assert rec is None
    progress_path = tmp_path / "models" / "progress_updates.jsonl"
    assert not progress_path.exists()


# ── Live-replay aggregator window filter ────────────────────────────────
def test_aggregate_live_replay_filters_by_run_window_but_honours_backfills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """`slice_done` rows older than the run start are filtered out, but
    `slice_live_replay_backfill` rows are always honoured (operator
    back-fills carry the previously-halted slice's evidence forward
    into a fresh campaign run)."""
    mod = _redirect_to_tmp(monkeypatch, tmp_path)
    progress_path = tmp_path / "models" / "progress_updates.jsonl"
    rows = [
        # OLD slice_done — should be filtered out
        {
            "emitted_at": "2026-04-01T00:00:00+00:00",
            "phase": "slice_done", "status": "trained",
            "coin": "celestia", "timeframe": "5m",
            "live_trade_count": 99, "live_net_pnl_pct": 1.0,
            "live_replay_status": "ok",
        },
        # NEW slice_done — kept
        {
            "emitted_at": "2026-04-30T00:00:00+00:00",
            "phase": "slice_done", "status": "trained",
            "coin": "pepe", "timeframe": "5m",
            "live_trade_count": 12, "live_net_pnl_pct": -2.5,
            "live_replay_status": "ok",
        },
        # OLD backfill — kept (operator-driven)
        {
            "emitted_at": "2026-04-01T00:00:00+00:00",
            "phase": "slice_live_replay_backfill", "status": "ok",
            "coin": "bonk", "timeframe": "5m",
            "live_trade_count": 1, "live_net_pnl_pct": 0.965,
            "live_replay_status": "ok",
        },
    ]
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = mod._aggregate_live_replay_per_slice("2026-04-29T00:00:00+00:00")
    assert "celestia/5m" not in out
    assert out["pepe/5m"]["live_trade_count"] == 12
    assert out["bonk/5m"]["live_trade_count"] == 1
    assert out["bonk/5m"]["source_phase"] == "slice_live_replay_backfill"
