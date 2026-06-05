"""Task #520 — `scripts/diff_campaign_pnl.py` smoke-test the resolver
and the empty/non-empty diff output paths so the helper stays usable
when an operator asks "did PnL regress vs the prior campaign?".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diff_mod():
    """Import scripts/diff_campaign_pnl.py without polluting sys.path
    permanently; the orchestrator script next door is heavyweight, so
    we add the scripts dir lazily and clean up after.
    """
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    added = False
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
        added = True
    try:
        import importlib

        if "diff_campaign_pnl" in sys.modules:
            mod = importlib.reload(sys.modules["diff_campaign_pnl"])
        else:
            mod = importlib.import_module("diff_campaign_pnl")
        yield mod
    finally:
        if added:
            sys.path.remove(str(scripts_dir))


def _write_snapshot(path: Path, per_slice: dict, *, completed: bool = True) -> None:
    payload = {
        "schema_version": "task520_v1",
        "per_slice": {},
        "prior_campaign_per_slice_pnl": {
            "prior_run_dir": "models/training_run_TEST",
            "prior_campaign_started_at": "2026-04-20T00:00:00+00:00",
            "prior_campaign_finished_at": "2026-04-20T03:00:00+00:00",
            "completed": completed,
            "slice_count": len(per_slice),
            "per_slice": per_slice,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _row(net: float, n: int, status: str = "trained") -> dict:
    return {
        "coin": "x", "timeframe": "1h", "status": status,
        "n_trades": n, "net_pct_total": net, "net_pct_mean": net / max(n, 1),
        "gross_pct_mean": 0.0, "win_rate": 0.4, "trade_share": 0.5,
        "round_trip_cost_pct": 0.3, "auc": 0.55, "baseline_auc": 0.5,
        "lift_auc": 0.05, "directional_accuracy": 0.4,
        "n_rows": 1000, "emitted_at": "2026-04-20T01:00:00+00:00",
    }


def test_resolve_snapshot_accepts_file_dir_and_run_dir(diff_mod, tmp_path):
    snap = tmp_path / "_archive" / "ARCH" / "baseline_snapshot.json"
    _write_snapshot(snap, {"bonk/1h": _row(net=-10.0, n=100)})

    # 1) direct file path
    assert diff_mod._resolve_snapshot(str(snap)) == snap
    # 2) parent archive directory
    assert diff_mod._resolve_snapshot(str(snap.parent)) == snap
    # 3) training_run_<TS>/ via phase3_baseline_pointer.json
    run_dir = tmp_path / "training_run_TEST"
    run_dir.mkdir()
    pointer = run_dir / "phase3_baseline_pointer.json"
    pointer.write_text(json.dumps({
        "archive_dir": str(snap.parent.relative_to(tmp_path)),
    }))
    # The resolver searches CWD/ROOT/_archive — point ROOT-relative
    # lookups at our tmp_path by patching the module constant.
    saved_root = diff_mod.ROOT
    diff_mod.ROOT = tmp_path
    diff_mod.ARCHIVE_ROOT = tmp_path / "_archive"
    try:
        assert diff_mod._resolve_snapshot(str(run_dir)) == snap
    finally:
        diff_mod.ROOT = saved_root
        diff_mod.ARCHIVE_ROOT = saved_root / "models" / "_archive"


def test_resolve_snapshot_raises_on_missing(diff_mod, tmp_path):
    with pytest.raises(FileNotFoundError):
        diff_mod._resolve_snapshot(str(tmp_path / "nope"))


def test_diff_main_prints_table_and_summary(diff_mod, tmp_path, capsys):
    snap_a = tmp_path / "A" / "baseline_snapshot.json"
    snap_b = tmp_path / "B" / "baseline_snapshot.json"
    _write_snapshot(snap_a, {
        "bonk/1h": _row(net=-100.0, n=100),
        "pepe/1d": _row(net=10.0, n=50),
    })
    _write_snapshot(snap_b, {
        "bonk/1h": _row(net=-50.0, n=120),    # improved by +50
        "pepe/1d": _row(net=-5.0, n=40),      # regressed by -15
        "celestia/6h": _row(net=2.0, n=30),   # only in B (no delta)
    })
    rc = diff_mod.main([str(snap_a), str(snap_b)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "# Per-slice PnL diff: net_pct_total" in captured
    assert "bonk/1h" in captured
    assert "pepe/1d" in captured
    assert "celestia/6h" in captured
    # Improved=1, regressed=1, undiffable=1 (celestia present only in B).
    assert "improved=1" in captured
    assert "regressed=1" in captured
    assert "undiffable=1" in captured


def test_diff_main_handles_both_empty(diff_mod, tmp_path, capsys):
    snap_a = tmp_path / "A" / "baseline_snapshot.json"
    snap_b = tmp_path / "B" / "baseline_snapshot.json"
    _write_snapshot(snap_a, {})
    _write_snapshot(snap_b, {})
    rc = diff_mod.main([str(snap_a), str(snap_b)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Both snapshots'" in captured
