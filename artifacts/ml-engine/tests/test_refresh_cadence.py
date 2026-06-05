"""Task #540 — auto-refresher cadence + freshness picker contract.

The cron-style scheduler (`scripts.scheduled_refresh_loop`) keeps the
cached `<tf>_<TS>.parquet` snapshots fresh on a documented cadence so
the ML retrain pipeline never silently consumes a week-old cache.
These tests pin down:

  * `_latest_pooled_dataset` picks the **newest by mtime**, not the
    biggest by size (the old behaviour, which silently selected stale
    snapshots when a fresh refresh wrote a smaller file because some
    coins had less history this week).
  * Passing `max_age_hours` to the picker raises a clear `RuntimeError`
    when the newest snapshot is older than the threshold — this is
    what makes `retrain_task524.py` fail loud rather than train on a
    stale cache.
  * The scheduler's `_due_timeframes` only refreshes timeframes whose
    newest snapshot is older than the per-tf cadence.
  * On refresher failure, the scheduler appends a JSON line to
    `_freshness_alerts.jsonl` and writes the failed status to
    `_freshness_status.json`.
  * The `_run_tick` happy path writes a success status entry.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import pytest


@pytest.fixture
def datasets_dir(tmp_path, monkeypatch):
    """Stand up a temp datasets dir and re-point the refresher modules
    at it. We monkeypatch the module-level `ROOT` and `DATASETS_DIR`
    constants so the production parquet directory is never touched.
    """
    from scripts import scheduled_refresh_loop as loop_mod
    from scripts.diagnostic_482 import run_stage_collapse_diagnostic as diag_mod

    ds_dir = tmp_path / "models" / "datasets"
    ds_dir.mkdir(parents=True)

    monkeypatch.setattr(loop_mod, "ROOT", tmp_path)
    monkeypatch.setattr(loop_mod, "DATASETS_DIR", ds_dir)
    monkeypatch.setattr(loop_mod, "STATUS_PATH", ds_dir / "_freshness_status.json")
    monkeypatch.setattr(loop_mod, "ALERTS_PATH", ds_dir / "_freshness_alerts.jsonl")
    monkeypatch.setattr(diag_mod, "ROOT", tmp_path)

    # Make sure no stale env override leaks across tests.
    monkeypatch.delenv("ML_DATASET_MAX_AGE_HOURS", raising=False)
    return ds_dir


def _touch(path: Path, *, mtime: float, size_bytes: int = 64) -> None:
    path.write_bytes(b"\0" * size_bytes)
    os.utime(path, (mtime, mtime))


# ── _latest_pooled_dataset picker ──────────────────────────────────────────


def test_latest_pooled_dataset_picks_newest_by_mtime(datasets_dir):
    from scripts.diagnostic_482.run_stage_collapse_diagnostic import (
        _latest_pooled_dataset,
    )

    now = time.time()
    # Older snapshot is much bigger — under the legacy size-based
    # picker this would win even though it is days stale.
    _touch(datasets_dir / "1d_old.parquet", mtime=now - 5 * 86400, size_bytes=10_000_000)
    _touch(datasets_dir / "1d_new.parquet", mtime=now - 600, size_bytes=10_000)

    chosen = _latest_pooled_dataset("1d")
    assert chosen.name == "1d_new.parquet", (
        "picker must select newest mtime, not biggest file"
    )


def test_latest_pooled_dataset_raises_when_stale(datasets_dir):
    from scripts.diagnostic_482.run_stage_collapse_diagnostic import (
        _latest_pooled_dataset,
    )

    now = time.time()
    _touch(datasets_dir / "1d_stale.parquet", mtime=now - 5 * 86400)

    with pytest.raises(RuntimeError, match="is .* old"):
        _latest_pooled_dataset("1d", max_age_hours=24)


def test_latest_pooled_dataset_accepts_fresh_snapshot(datasets_dir):
    from scripts.diagnostic_482.run_stage_collapse_diagnostic import (
        _latest_pooled_dataset,
    )

    now = time.time()
    _touch(datasets_dir / "1d_fresh.parquet", mtime=now - 600)

    chosen = _latest_pooled_dataset("1d", max_age_hours=24)
    assert chosen.name == "1d_fresh.parquet"


def test_latest_pooled_dataset_env_overrides_max_age(datasets_dir, monkeypatch):
    from scripts.diagnostic_482.run_stage_collapse_diagnostic import (
        _latest_pooled_dataset,
    )

    now = time.time()
    _touch(datasets_dir / "1d_stale.parquet", mtime=now - 5 * 86400)

    # Override of 0 disables the freshness check entirely (operator
    # escape hatch for one-off reruns).
    monkeypatch.setenv("ML_DATASET_MAX_AGE_HOURS", "0")
    chosen = _latest_pooled_dataset("1d", max_age_hours=24)
    assert chosen.name == "1d_stale.parquet"


def test_latest_pooled_dataset_no_candidates(datasets_dir):
    from scripts.diagnostic_482.run_stage_collapse_diagnostic import (
        _latest_pooled_dataset,
    )

    with pytest.raises(FileNotFoundError):
        _latest_pooled_dataset("1d", max_age_hours=24)


# ── scheduler cadence: _due_timeframes ──────────────────────────────────────


def test_due_timeframes_skips_recently_refreshed(datasets_dir, monkeypatch):
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    # 1d snapshot 1h old — well under the 24h cadence.
    _touch(datasets_dir / "1d_recent.parquet", mtime=now - 3600)
    # 6h snapshot 30h old — past its 24h cadence.
    _touch(datasets_dir / "6h_stale.parquet", mtime=now - 30 * 3600)

    status = {"timeframes": {
        "1d": {"last_success_at": loop._ts_to_iso(now - 3600)},
        "6h": {"last_success_at": loop._ts_to_iso(now - 30 * 3600)},
    }}

    due = loop._due_timeframes(["1d", "6h"], status, now)
    assert due == ["6h"]


def test_due_timeframes_refreshes_when_no_snapshot(datasets_dir):
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    due = loop._due_timeframes(["1d", "6h"], {"timeframes": {}}, now)
    # No snapshots on disk for either tf — both must be refreshed.
    assert sorted(due) == ["1d", "6h"]


def test_due_timeframes_per_tf_cadence_override(datasets_dir, monkeypatch):
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    _touch(datasets_dir / "1h_2h_old.parquet", mtime=now - 3 * 3600)
    # Make the file name well-formed so the picker matches.
    (datasets_dir / "1h_2h_old.parquet").rename(datasets_dir / "1h_recent.parquet")
    _touch(datasets_dir / "1h_recent.parquet", mtime=now - 3 * 3600)

    status = {"timeframes": {
        "1h": {"last_success_at": loop._ts_to_iso(now - 3 * 3600)},
    }}

    # Default 1h cadence is 6h — 3h old should NOT be due.
    due = loop._due_timeframes(["1h"], status, now)
    assert due == []

    # Tighten cadence to 1h — now the 3h-old snapshot IS due.
    monkeypatch.setenv("ML_REFRESH_CADENCE_1H_HOURS", "1")
    due = loop._due_timeframes(["1h"], status, now)
    assert due == ["1h"]


# ── scheduler tick: alerts + status writes ─────────────────────────────────


@pytest.mark.asyncio
async def test_run_tick_writes_alert_and_status_on_failure(
    datasets_dir, monkeypatch,
):
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    _touch(datasets_dir / "1d_stale.parquet", mtime=now - 30 * 3600)

    async def _fake_init() -> None:
        return None

    async def _fake_close() -> None:
        return None

    async def _fake_refresh(tf, coin_ids, version):
        raise RuntimeError("simulated DB outage")

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    monkeypatch.setattr(loop, "close_pool", _fake_close)
    monkeypatch.setattr(loop, "_refresh_one", _fake_refresh)
    monkeypatch.setattr(loop, "_selected_coins", lambda: ["bonk"])

    await loop._run_tick(["1d"], ["bonk"])

    alerts = (datasets_dir / "_freshness_alerts.jsonl").read_text().strip().splitlines()
    assert len(alerts) == 1
    payload = json.loads(alerts[0])
    assert payload["timeframe"] == "1d"
    assert payload["status"] == "error"
    assert "simulated DB outage" in payload["error"]
    assert payload["cadence_hours"] == 24.0

    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    entry = status["timeframes"]["1d"]
    assert entry["last_status"] == "error"
    assert entry["last_error"] and "simulated DB outage" in entry["last_error"]
    assert entry.get("last_success_at") is None
    assert entry["next_due_at"]
    assert entry["mtime_of_newest_snapshot"]


@pytest.mark.asyncio
async def test_run_tick_writes_success_status_on_ok(datasets_dir, monkeypatch):
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    _touch(datasets_dir / "1d_stale.parquet", mtime=now - 30 * 3600)

    async def _fake_init() -> None:
        return None

    async def _fake_close() -> None:
        return None

    async def _fake_refresh(tf, coin_ids, version):
        # Simulate the refresher writing a new snapshot before returning ok.
        out = datasets_dir / f"{tf}_NEWFRESH.parquet"
        _touch(out, mtime=time.time())
        return {
            "timeframe": tf, "status": "ok",
            "n_rows": 12345,
            "coins_emitted": list(coin_ids),
            "output_path": f"models/datasets/{out.name}",
        }

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    monkeypatch.setattr(loop, "close_pool", _fake_close)
    monkeypatch.setattr(loop, "_refresh_one", _fake_refresh)
    monkeypatch.setattr(loop, "_selected_coins", lambda: ["bonk", "pepe"])

    await loop._run_tick(["1d"], ["bonk", "pepe"])

    # No alert line should have been written on the happy path.
    assert not (datasets_dir / "_freshness_alerts.jsonl").exists()

    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    entry = status["timeframes"]["1d"]
    assert entry["last_status"] == "ok"
    assert entry["last_error"] is None
    assert entry["last_n_rows"] == 12345
    assert entry["last_coins_emitted"] == ["bonk", "pepe"]
    assert entry["last_success_at"]
    assert entry["next_due_at"]


@pytest.mark.asyncio
async def test_run_tick_skips_when_nothing_due(datasets_dir, monkeypatch):
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    _touch(datasets_dir / "1d_recent.parquet", mtime=now - 600)

    called: list[str] = []

    async def _fake_refresh(tf, coin_ids, version):  # noqa: ARG001
        called.append(tf)
        return {"timeframe": tf, "status": "ok", "n_rows": 1}

    async def _fake_init() -> None:
        return None

    async def _fake_close() -> None:
        return None

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    monkeypatch.setattr(loop, "close_pool", _fake_close)
    monkeypatch.setattr(loop, "_refresh_one", _fake_refresh)

    # Pre-seed status so `_due_timeframes` thinks 1d was just refreshed.
    (datasets_dir / "_freshness_status.json").write_text(json.dumps({
        "timeframes": {"1d": {"last_success_at": loop._ts_to_iso(now - 600)}},
    }))

    await loop._run_tick(["1d"], ["bonk"])

    assert called == [], "no refresh should fire when nothing is due"
    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    # next_due_at must still be updated for visibility.
    assert status["timeframes"]["1d"]["next_due_at"]


# ── Task #556: trim policy + pin protection ────────────────────────────────


def test_trim_keeps_n_freshest_and_deletes_older(datasets_dir, monkeypatch):
    """The trim policy must keep exactly the freshest N snapshots per
    timeframe and delete everything older. This pins the contract that
    keeps the dataset cache from growing unbounded.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    # 1d default retention is 14 → seed 18 snapshots so 4 must be deleted.
    for day_offset in range(18):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    # No pinned dataset names match anything we just wrote, so the
    # trim is purely retention-driven.
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())

    summary = loop._trim_old_snapshots("1d")

    assert summary["policy"] == "keep_newest_n"
    assert summary["retain_n"] == 14
    assert summary["retained"] == 14
    assert summary["deleted"] == 4
    # The 4 oldest must be gone, the 14 freshest must remain.
    remaining = sorted(p.name for p in datasets_dir.glob("1d_*.parquet"))
    assert len(remaining) == 14
    for day_offset in range(14):
        assert f"1d_2026-04-{day_offset:02d}.parquet" in remaining
    for day_offset in range(14, 18):
        assert f"1d_2026-04-{day_offset:02d}.parquet" not in remaining
    # oldest_kept_at reflects the 14th-freshest snapshot's mtime.
    assert summary["oldest_kept_at"] is not None


def test_trim_protects_pinned_diagnostic_snapshots(datasets_dir, monkeypatch):
    """Pinned snapshots referenced by diagnostic harnesses
    (e.g. ``run_516_pnl_ab.PINNED_DATASETS``) must NEVER be trimmed
    even if they are older than the retention window. Reproducibility
    of archived reports depends on this.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    pinned_name = "1d_PINNED_HISTORICAL.parquet"
    # Pin one ancient snapshot that would normally be deleted.
    _touch(datasets_dir / pinned_name, mtime=now - 60 * 86400)
    # And 16 fresh snapshots so the pinned one is well outside the
    # 14-snapshot retention window.
    for day_offset in range(16):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    monkeypatch.setattr(loop, "_pinned_filenames", lambda: {pinned_name})

    summary = loop._trim_old_snapshots("1d")

    # Pinned file survives even though it is the oldest on disk.
    assert (datasets_dir / pinned_name).exists(), (
        "pinned diagnostic snapshot must NEVER be trimmed"
    )
    assert pinned_name in summary["pinned_kept"]
    # Two non-pinned snapshots beyond the retention window are deleted.
    assert summary["deleted"] == 2
    # Sanity: the freshest 14 survive plus the pinned one.
    remaining = sorted(p.name for p in datasets_dir.glob("1d_*.parquet"))
    assert len(remaining) == 15
    assert pinned_name in remaining


def test_trim_pinned_filenames_aggregates_run_516(monkeypatch):
    """The aggregator must surface the actual ``PINNED_DATASETS`` from
    the diagnostic_482 module — the production trim path imports it,
    so a rename there must turn into a clear test failure rather than
    a silent loss of pin protection.
    """
    from scripts import scheduled_refresh_loop as loop
    from scripts.diagnostic_482 import run_516_pnl_ab

    pinned = loop._pinned_filenames()
    for fname in run_516_pnl_ab.PINNED_DATASETS.values():
        assert fname in pinned, (
            f"pinned filename {fname!r} from run_516_pnl_ab must be "
            "present in the trim aggregator"
        )


def test_trim_no_op_when_under_retention(datasets_dir, monkeypatch):
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())
    # Only 3 snapshots exist; default retention is 14.
    for day_offset in range(3):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    summary = loop._trim_old_snapshots("1d")
    assert summary["deleted"] == 0
    assert summary["retained"] == 3
    assert len(list(datasets_dir.glob("1d_*.parquet"))) == 3


def test_trim_per_tf_retention_override(datasets_dir, monkeypatch):
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())
    for day_offset in range(10):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    # Tighten retention to 3 — must delete 7 snapshots.
    monkeypatch.setenv("ML_REFRESH_RETAIN_1D", "3")
    summary = loop._trim_old_snapshots("1d")
    assert summary["retain_n"] == 3
    assert summary["retained"] == 3
    assert summary["deleted"] == 7
    assert len(list(datasets_dir.glob("1d_*.parquet"))) == 3


def test_trim_disabled_when_retention_zero(datasets_dir, monkeypatch):
    """Setting ``ML_REFRESH_RETAIN_<TF>=0`` is the operator escape
    hatch for forensic windows — it must short-circuit to a no-op so
    no snapshots are deleted.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    for day_offset in range(20):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    monkeypatch.setenv("ML_REFRESH_RETAIN_1D", "0")
    summary = loop._trim_old_snapshots("1d")
    assert summary["retain_n"] == 0
    assert summary["deleted"] == 0
    # All 20 snapshots remain on disk.
    assert len(list(datasets_dir.glob("1d_*.parquet"))) == 20
    # Reported retained count reflects on-disk reality.
    assert summary["retained"] == 20
    assert summary["oldest_kept_at"] is not None


def test_trim_fails_safe_when_pin_source_unavailable(datasets_dir, monkeypatch):
    """If the pinned-snapshot manifest cannot be loaded (e.g. a
    transient import error in a diagnostic module), the trim path
    must REFUSE to delete anything. The "never trim pinned snapshots"
    guarantee is more important than the "cache is bounded" goal —
    a missed trim tick is harmless, but deleting a pinned snapshot
    breaks reproducibility of an archived report.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    # Way over the 14-snapshot retention window.
    for day_offset in range(20):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    def _broken_pin_source() -> set[str]:
        raise ImportError("simulated diagnostic module import failure")

    monkeypatch.setattr(loop, "_pinned_filenames", _broken_pin_source)

    summary = loop._trim_old_snapshots("1d")

    # No deletions despite being well over the retention window.
    assert summary["deleted"] == 0
    assert summary["retained"] == 20
    assert summary["skipped"] == "pinned_filenames_unavailable"
    assert "ImportError" in summary["skip_error"]
    # All 20 snapshots remain on disk — fail-safe really held.
    assert len(list(datasets_dir.glob("1d_*.parquet"))) == 20


def test_trim_only_touches_target_timeframe(datasets_dir, monkeypatch):
    """Trimming 1h must NOT touch 1d snapshots even if the 1d cache
    is technically over its retention window. Each tick trims one tf
    after that tf's own successful refresh.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())
    # Way over 1d retention (14): 20 snapshots.
    for day_offset in range(20):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )
    # Way over 1h retention (28): 30 snapshots.
    for h_offset in range(30):
        _touch(
            datasets_dir / f"1h_2026-04-{h_offset:02d}.parquet",
            mtime=now - h_offset * 3600,
        )

    summary = loop._trim_old_snapshots("1h")
    assert summary["deleted"] == 2
    # 1d cache untouched.
    assert len(list(datasets_dir.glob("1d_*.parquet"))) == 20
    # 1h cache trimmed to 28.
    assert len(list(datasets_dir.glob("1h_*.parquet"))) == 28


# ── Task #560: trim-spike alert ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_tick_alerts_when_trim_deletes_exceed_threshold(
    datasets_dir, monkeypatch,
):
    """Task #560 — when one tick's trim deletes more files than the
    configurable threshold (default 5), the scheduler must append a
    ``trim_spike`` JSON line to ``_freshness_alerts.jsonl`` so an
    operator notices clock skew, manual snapshot copies, or a
    freshly-lowered retention env override before historical-looking
    snapshots silently disappear.
    """
    from scripts import scheduled_refresh_loop as loop

    # Make sure no leftover override from another test changes the
    # default threshold of 5.
    monkeypatch.delenv("ML_REFRESH_TRIM_SPIKE_THRESHOLD", raising=False)
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())

    now = time.time()
    # Seed 30 stale 1d snapshots. After the fake refresh writes the
    # 31st (freshest) one, default 1d retention is 14 → 17 deletions
    # in this tick — well over the default threshold of 5.
    for day_offset in range(1, 31):
        _touch(
            datasets_dir / f"1d_2026-03-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    async def _fake_init() -> None:
        return None

    async def _fake_close() -> None:
        return None

    async def _fake_refresh(tf, coin_ids, version):  # noqa: ARG001
        out = datasets_dir / f"{tf}_FRESH.parquet"
        _touch(out, mtime=time.time())
        return {
            "timeframe": tf, "status": "ok",
            "n_rows": 1,
            "coins_emitted": list(coin_ids),
            "output_path": f"models/datasets/{out.name}",
        }

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    monkeypatch.setattr(loop, "close_pool", _fake_close)
    monkeypatch.setattr(loop, "_refresh_one", _fake_refresh)
    monkeypatch.setattr(loop, "_selected_coins", lambda: ["bonk"])

    await loop._run_tick(["1d"], ["bonk"])

    alerts_path = datasets_dir / "_freshness_alerts.jsonl"
    assert alerts_path.exists(), (
        "trim spike must append a JSON line to _freshness_alerts.jsonl"
    )
    lines = alerts_path.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "trim_spike"
    assert payload["timeframe"] == "1d"
    assert payload["retain_n"] == 14
    assert payload["threshold"] == 5
    # 30 seeded + 1 fresh = 31 on-disk → keep 14 → 17 deleted.
    assert payload["deleted_count"] == 17
    assert isinstance(payload["deleted_files"], list)
    assert len(payload["deleted_files"]) == 17
    # The actual filenames must be in the payload so an operator can
    # see whether legit-looking historical snapshots got nuked.
    for fname in payload["deleted_files"]:
        assert fname.startswith("1d_2026-03-") and fname.endswith(".parquet")
    assert "at" in payload


@pytest.mark.asyncio
async def test_run_tick_no_trim_spike_alert_under_threshold(
    datasets_dir, monkeypatch,
):
    """Steady-state trims (0–1 deletion per tick, up to the threshold)
    must NOT fire ``trim_spike`` alerts — otherwise the alert loses
    all signal value.
    """
    from scripts import scheduled_refresh_loop as loop

    monkeypatch.delenv("ML_REFRESH_TRIM_SPIKE_THRESHOLD", raising=False)
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())

    now = time.time()
    # 16 seeded + 1 fresh = 17 on-disk → keep 14 → 3 deleted ≤ 5.
    for day_offset in range(1, 17):
        _touch(
            datasets_dir / f"1d_2026-03-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    async def _fake_init() -> None:
        return None

    async def _fake_close() -> None:
        return None

    async def _fake_refresh(tf, coin_ids, version):  # noqa: ARG001
        out = datasets_dir / f"{tf}_FRESH.parquet"
        _touch(out, mtime=time.time())
        return {
            "timeframe": tf, "status": "ok",
            "n_rows": 1, "coins_emitted": list(coin_ids),
            "output_path": f"models/datasets/{out.name}",
        }

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    monkeypatch.setattr(loop, "close_pool", _fake_close)
    monkeypatch.setattr(loop, "_refresh_one", _fake_refresh)
    monkeypatch.setattr(loop, "_selected_coins", lambda: ["bonk"])

    await loop._run_tick(["1d"], ["bonk"])

    # The trim happened (3 deletions appear in the status entry) but
    # 3 ≤ default threshold 5 so no alert was emitted.
    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    assert status["timeframes"]["1d"]["retention"]["deleted"] == 3
    assert not (datasets_dir / "_freshness_alerts.jsonl").exists()


@pytest.mark.asyncio
async def test_run_tick_trim_spike_threshold_env_override(
    datasets_dir, monkeypatch,
):
    """``ML_REFRESH_TRIM_SPIKE_THRESHOLD`` must let operators tighten
    the spike alert below the default of 5.
    """
    from scripts import scheduled_refresh_loop as loop

    monkeypatch.setenv("ML_REFRESH_TRIM_SPIKE_THRESHOLD", "2")
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())

    now = time.time()
    # 16 seeded + 1 fresh = 17 on-disk → keep 14 → 3 deleted > 2.
    for day_offset in range(1, 17):
        _touch(
            datasets_dir / f"1d_2026-03-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )

    async def _fake_init() -> None:
        return None

    async def _fake_close() -> None:
        return None

    async def _fake_refresh(tf, coin_ids, version):  # noqa: ARG001
        out = datasets_dir / f"{tf}_FRESH.parquet"
        _touch(out, mtime=time.time())
        return {
            "timeframe": tf, "status": "ok",
            "n_rows": 1, "coins_emitted": list(coin_ids),
            "output_path": f"models/datasets/{out.name}",
        }

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    monkeypatch.setattr(loop, "close_pool", _fake_close)
    monkeypatch.setattr(loop, "_refresh_one", _fake_refresh)
    monkeypatch.setattr(loop, "_selected_coins", lambda: ["bonk"])

    await loop._run_tick(["1d"], ["bonk"])

    lines = (
        (datasets_dir / "_freshness_alerts.jsonl").read_text().strip().splitlines()
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "trim_spike"
    assert payload["threshold"] == 2
    assert payload["deleted_count"] == 3


@pytest.mark.asyncio
async def test_run_tick_writes_retention_summary_on_success(
    datasets_dir, monkeypatch,
):
    """After a successful refresh, the per-tf status entry must
    contain a ``retention`` summary so operators can see the trim
    behaviour at a glance.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    # Seed 16 old 1d snapshots — 2 should be trimmed after the refresh
    # writes the 17th (which is the freshest).
    for day_offset in range(1, 17):
        _touch(
            datasets_dir / f"1d_2026-03-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
        )
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())

    async def _fake_init() -> None:
        return None

    async def _fake_close() -> None:
        return None

    async def _fake_refresh(tf, coin_ids, version):
        out = datasets_dir / f"{tf}_FRESH.parquet"
        _touch(out, mtime=time.time())
        return {
            "timeframe": tf, "status": "ok",
            "n_rows": 100,
            "coins_emitted": list(coin_ids),
            "output_path": f"models/datasets/{out.name}",
        }

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    monkeypatch.setattr(loop, "close_pool", _fake_close)
    monkeypatch.setattr(loop, "_refresh_one", _fake_refresh)
    monkeypatch.setattr(loop, "_selected_coins", lambda: ["bonk"])

    await loop._run_tick(["1d"], ["bonk"])

    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    retention = status["timeframes"]["1d"].get("retention")
    assert retention is not None, (
        "_run_tick must embed a retention summary after a successful refresh"
    )
    assert retention["policy"] == "keep_newest_n"
    assert retention["retain_n"] == 14
    assert retention["retained"] == 14
    assert retention["deleted"] == 3  # 17 on-disk → keep 14 → drop 3
    assert retention["oldest_kept_at"]
    # Cache really shrank to 14.
    assert len(list(datasets_dir.glob("1d_*.parquet"))) == 14
    # Task #559 — bytes_on_disk + total_bytes_on_disk + size history
    # must be populated after the refresh tick so the freshness
    # dashboard panel has something to render.
    assert "bytes_on_disk" in retention
    assert retention["bytes_on_disk"] > 0
    assert status.get("total_bytes_on_disk") == retention["bytes_on_disk"]
    assert "cache_size_warning" in status
    history = status.get("cache_size_history") or []
    assert len(history) >= 1
    last_sample = history[-1]
    assert last_sample["total_bytes"] == status["total_bytes_on_disk"]
    assert last_sample["per_tf"]["1d"] == retention["bytes_on_disk"]


# ── Task #559: cache-size telemetry on the freshness status file ───────────


def test_trim_summary_includes_bytes_on_disk(datasets_dir, monkeypatch):
    """``_trim_old_snapshots`` must report the post-trim bytes-on-disk
    for the timeframe so the dashboard can confirm the cache stays
    capped at the documented footprint.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())
    # Seed 18 1d snapshots, each 1 KiB, so the post-trim footprint
    # is exactly 14 KiB (14 retained × 1024 bytes).
    for day_offset in range(18):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
            size_bytes=1024,
        )

    summary = loop._trim_old_snapshots("1d")

    assert summary["bytes_on_disk"] == 14 * 1024, (
        "bytes_on_disk must be the post-trim sum of <tf>_*.parquet "
        "sizes — that is what the dashboard reports as the cache "
        "footprint for the timeframe."
    )


def test_trim_bytes_on_disk_when_disabled(datasets_dir, monkeypatch):
    """Even when trimming is disabled (``ML_REFRESH_RETAIN_<TF>=0``)
    the summary must still surface the on-disk footprint so the
    operator can see exactly how big the unbounded cache has grown.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    for day_offset in range(5):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
            size_bytes=2048,
        )
    monkeypatch.setenv("ML_REFRESH_RETAIN_1D", "0")
    summary = loop._trim_old_snapshots("1d")

    assert summary["bytes_on_disk"] == 5 * 2048
    # Sanity: nothing was deleted.
    assert summary["deleted"] == 0


def test_trim_bytes_on_disk_when_pin_source_unavailable(
    datasets_dir, monkeypatch,
):
    """Fail-safe path (pin source raised) must still surface the
    current footprint so operators can see the cache hasn't shrunk
    even though no trim happened.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    for day_offset in range(20):
        _touch(
            datasets_dir / f"1d_2026-04-{day_offset:02d}.parquet",
            mtime=now - day_offset * 86400,
            size_bytes=512,
        )

    def _broken() -> set[str]:
        raise ImportError("simulated")

    monkeypatch.setattr(loop, "_pinned_filenames", _broken)
    summary = loop._trim_old_snapshots("1d")
    assert summary["deleted"] == 0
    assert summary["bytes_on_disk"] == 20 * 512


@pytest.mark.asyncio
async def test_run_tick_quiet_tick_still_refreshes_cache_size(
    datasets_dir, monkeypatch,
):
    """When no timeframe is due, the tick must still refresh
    ``total_bytes_on_disk`` and append a ``cache_size_history``
    sample. Otherwise the dashboard sparkline would freeze whenever
    the refresher had nothing to do.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    # Recent snapshot so the 1d cadence (24h) is NOT yet due.
    _touch(
        datasets_dir / "1d_RECENT.parquet",
        mtime=now - 60,
        size_bytes=4096,
    )
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())

    async def _fail_init() -> None:  # pragma: no cover — must not fire
        raise AssertionError("init_pool must not run on a quiet tick")

    monkeypatch.setattr(loop, "init_pool", _fail_init)

    await loop._run_tick(["1d"], ["bonk"])

    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    assert status["total_bytes_on_disk"] == 4096
    assert status["cache_size_warning"] is False
    history = status.get("cache_size_history") or []
    assert len(history) == 1
    assert history[0]["per_tf"]["1d"] == 4096
    # Per-tf retention.bytes_on_disk gets refreshed on a quiet tick too.
    entry = status["timeframes"]["1d"]
    assert entry["retention"]["bytes_on_disk"] == 4096


@pytest.mark.asyncio
async def test_run_tick_cache_size_warning_trips_above_threshold(
    datasets_dir, monkeypatch,
):
    """When the total footprint exceeds the soft warning threshold
    the status must flag ``cache_size_warning`` so the dashboard
    can render the amber chip.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())
    # Drop the threshold to 1 KiB so we don't have to write a real
    # 5-GB file just to trip the chip.
    monkeypatch.setattr(loop, "CACHE_SIZE_WARN_BYTES", 1024)
    _touch(
        datasets_dir / "1d_RECENT.parquet",
        mtime=now - 60,
        size_bytes=4096,
    )
    async def _fake_init() -> None:
        return None

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    await loop._run_tick(["1d"], ["bonk"])

    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    assert status["cache_size_warning"] is True
    assert status["cache_size_warn_bytes"] == 1024


@pytest.mark.asyncio
async def test_run_tick_cache_size_history_caps_at_configured_len(
    datasets_dir, monkeypatch,
):
    """``cache_size_history`` must be capped at
    ``ML_REFRESH_SIZE_HISTORY_LEN`` entries so the status file does
    not grow unbounded.
    """
    from scripts import scheduled_refresh_loop as loop

    now = time.time()
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())
    monkeypatch.setenv("ML_REFRESH_SIZE_HISTORY_LEN", "3")
    _touch(
        datasets_dir / "1d_RECENT.parquet",
        mtime=now - 60,
        size_bytes=1024,
    )

    async def _fake_init() -> None:
        return None

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    # Five quiet ticks → history must keep only the last 3 samples.
    for _ in range(5):
        await loop._run_tick(["1d"], ["bonk"])

    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    history = status.get("cache_size_history") or []
    assert len(history) == 3, (
        "cache_size_history must be capped at "
        "ML_REFRESH_SIZE_HISTORY_LEN entries"
    )


# ── Task #599: auto-trigger calibration verdict on short-tf refresh ───────


def _install_short_tf_fakes(loop, datasets_dir, monkeypatch):
    """Wire the loop's DB + refresher hooks to no-op fakes that pretend
    every refresh succeeded. Returns nothing — caller installs the
    verdict subprocess fake separately so each test can pin its own
    success/error/timeout behaviour.
    """
    async def _fake_init() -> None:
        return None

    async def _fake_close() -> None:
        return None

    async def _fake_refresh(tf, coin_ids, version):  # noqa: ARG001
        out = datasets_dir / f"{tf}_FRESH.parquet"
        _touch(out, mtime=time.time())
        return {
            "timeframe": tf, "status": "ok",
            "n_rows": 1,
            "coins_emitted": list(coin_ids),
            "output_path": f"models/datasets/{out.name}",
        }

    monkeypatch.setattr(loop, "init_pool", _fake_init)
    monkeypatch.setattr(loop, "close_pool", _fake_close)
    monkeypatch.setattr(loop, "_refresh_one", _fake_refresh)
    monkeypatch.setattr(loop, "_pinned_filenames", lambda: set())
    monkeypatch.setattr(loop, "_selected_coins", lambda: ["bonk"])


@pytest.mark.asyncio
async def test_short_tf_refresh_triggers_calibration_verdict(
    datasets_dir, monkeypatch, tmp_path,
):
    """Task #599 — a successful 1h refresh must trigger the calibration
    verdict subprocess and embed the result in the status file under
    ``calibration_verdict.short_tf``.
    """
    from scripts import scheduled_refresh_loop as loop

    monkeypatch.delenv("ML_REFRESH_AUTO_CALIBRATION_VERDICT", raising=False)
    _install_short_tf_fakes(loop, datasets_dir, monkeypatch)

    triggers: list[list[str]] = []

    def _fake_run_verdict(triggered_by):
        # The loop must call us with the short-tf list that just
        # refreshed.
        triggers.append(list(triggered_by))
        return {
            "last_attempt_at": loop._now_iso(),
            "last_status": "ok",
            "last_success_at": loop._now_iso(),
            "last_error": None,
            "last_elapsed_seconds": 70.5,
            "last_md_path": "reports/X-task592-1h2h-stage2-verdict.md",
            "last_json_path": "reports/X-task592-1h2h-stage2-verdict.json",
            "summary": {
                "off": {"n_in_trade_share_band": 0, "mean_trade_share": 0.95},
                "on": {"n_in_trade_share_band": 19, "mean_trade_share": 0.65},
                "captured_at": "20260429T040000Z",
                "timeframes_subset": ["1h", "2h"],
            },
            "trigger_timeframes": list(triggered_by),
            "command": "fake",
            "timeout_seconds": 600,
        }

    monkeypatch.setattr(loop, "_run_calibration_verdict", _fake_run_verdict)

    await loop._run_tick(["1h"], ["bonk"])

    assert triggers == [["1h"]], (
        "verdict subprocess must fire exactly once with the short-tf "
        f"list that just refreshed; got {triggers}"
    )
    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    cv = status["calibration_verdict"]["short_tf"]
    assert cv["last_status"] == "ok"
    assert cv["last_md_path"].endswith(".md")
    assert cv["summary"]["on"]["n_in_trade_share_band"] == 19
    # No alert line should be written on the happy path.
    assert not (datasets_dir / "_freshness_alerts.jsonl").exists()


@pytest.mark.asyncio
async def test_non_short_tf_refresh_does_not_trigger_verdict(
    datasets_dir, monkeypatch,
):
    """Task #599 — a successful 1d refresh (or any tf outside
    ``CALIBRATION_VERDICT_TFS``) must NOT spawn the verdict subprocess.
    The verdict is short-tf-only; running it after a 1d refresh would
    burn ~70s for no operational signal.
    """
    from scripts import scheduled_refresh_loop as loop

    monkeypatch.delenv("ML_REFRESH_AUTO_CALIBRATION_VERDICT", raising=False)
    _install_short_tf_fakes(loop, datasets_dir, monkeypatch)

    calls: list[list[str]] = []

    def _fake_run_verdict(triggered_by):  # pragma: no cover
        calls.append(list(triggered_by))
        return {"last_status": "ok"}

    monkeypatch.setattr(loop, "_run_calibration_verdict", _fake_run_verdict)

    await loop._run_tick(["1d"], ["bonk"])

    assert calls == [], (
        "verdict must NOT run when the refreshed tf is outside "
        f"{loop.CALIBRATION_VERDICT_TFS}; got {calls}"
    )
    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    assert "calibration_verdict" not in status


@pytest.mark.asyncio
async def test_short_tf_refresh_skips_verdict_when_env_disabled(
    datasets_dir, monkeypatch,
):
    """Task #599 — operators must be able to disable the auto-trigger
    via ``ML_REFRESH_AUTO_CALIBRATION_VERDICT=0`` (e.g. when manually
    running the verdict during a tuning session and not wanting the
    refresher to stomp their state).
    """
    from scripts import scheduled_refresh_loop as loop

    monkeypatch.setenv("ML_REFRESH_AUTO_CALIBRATION_VERDICT", "0")
    _install_short_tf_fakes(loop, datasets_dir, monkeypatch)

    calls: list[list[str]] = []

    def _fake_run_verdict(triggered_by):  # pragma: no cover
        calls.append(list(triggered_by))
        return {"last_status": "ok"}

    monkeypatch.setattr(loop, "_run_calibration_verdict", _fake_run_verdict)

    await loop._run_tick(["1h"], ["bonk"])

    assert calls == [], "env knob disabled the verdict; subprocess must NOT run"


@pytest.mark.asyncio
async def test_short_tf_verdict_failure_appends_alert_and_keeps_prev_summary(
    datasets_dir, monkeypatch,
):
    """Task #599 — when the verdict subprocess fails (non-zero exit,
    timeout, or missing report), the loop must:
      * append a ``calibration_verdict_*`` JSON line to
        ``_freshness_alerts.jsonl`` so the dashboard alert log
        surfaces it,
      * record the failure under
        ``status['calibration_verdict']['short_tf']``,
      * preserve the previous successful run's ``last_success_at`` /
        ``last_md_path`` / ``summary`` so the dashboard still has a
        usable last-good verdict to show alongside the failure.
    """
    from scripts import scheduled_refresh_loop as loop

    monkeypatch.delenv("ML_REFRESH_AUTO_CALIBRATION_VERDICT", raising=False)
    _install_short_tf_fakes(loop, datasets_dir, monkeypatch)

    # Pre-seed a previous success in the status file. The loop must
    # not lose this when the new run fails.
    prev_success = {
        "calibration_verdict": {
            "short_tf": {
                "last_attempt_at": "2026-04-28T20:00:00Z",
                "last_status": "ok",
                "last_success_at": "2026-04-28T20:00:00Z",
                "last_md_path": "reports/old-task592-1h2h-stage2-verdict.md",
                "last_json_path": "reports/old-task592-1h2h-stage2-verdict.json",
                "summary": {"off": {"n_in_trade_share_band": 0}, "on": {"n_in_trade_share_band": 19}},
            },
        },
    }
    (datasets_dir / "_freshness_status.json").write_text(json.dumps(prev_success))

    def _fake_run_verdict(triggered_by):
        return {
            "last_attempt_at": loop._now_iso(),
            "last_status": "error",
            "last_error": "verdict subprocess exited with rc=3",
            "last_elapsed_seconds": 12.3,
            "trigger_timeframes": list(triggered_by),
            "command": "fake",
            "timeout_seconds": 600,
            "stderr_tail": "Traceback (most recent call last): ...",
        }

    monkeypatch.setattr(loop, "_run_calibration_verdict", _fake_run_verdict)

    await loop._run_tick(["2h"], ["bonk"])

    status = json.loads((datasets_dir / "_freshness_status.json").read_text())
    cv = status["calibration_verdict"]["short_tf"]
    assert cv["last_status"] == "error"
    assert "rc=3" in cv["last_error"]
    # Previous success must be preserved alongside the new failure.
    assert cv["last_success_at"] == "2026-04-28T20:00:00Z"
    assert cv["last_md_path"].startswith("reports/old-task592-")
    assert cv["summary"]["on"]["n_in_trade_share_band"] == 19

    alerts = (
        (datasets_dir / "_freshness_alerts.jsonl").read_text().strip().splitlines()
    )
    assert len(alerts) == 1
    payload = json.loads(alerts[0])
    assert payload["status"] == "calibration_verdict_error"
    assert payload["timeframe"] == "2h"
    assert "rc=3" in payload["error"]


def test_summarize_verdict_payload_extracts_aggregates(tmp_path):
    """Task #599 — the local summariser must produce the same
    aggregate counts that ``task587_rerun_stage2._gate_summary``
    produces, so the dashboard never disagrees with the markdown.
    """
    from scripts import scheduled_refresh_loop as loop

    payload = {
        "captured_at": "20260429T033030Z",
        "round_trip_cost_pct": 0.3,
        "timeframes_subset": ["1h", "2h"],
        "off": {
            "per_slice": [
                {  # in-band miss (gate_pass false), trade_share 0.95
                    "coin": "a", "timeframe": "1h", "gate_pass": False,
                    "augmented": {
                        "trade_share": 0.95,
                        "post_fee_pnl_pct_total": -100.0,
                    },
                    "da_lift": -0.005,
                },
                {  # in-band hit, trade_share 0.55
                    "coin": "b", "timeframe": "1h", "gate_pass": True,
                    "augmented": {
                        "trade_share": 0.55,
                        "post_fee_pnl_pct_total": 50.0,
                    },
                    "da_lift": 0.04,
                },
            ],
        },
        "on": {
            "per_slice": [
                {  # both in-band
                    "coin": "a", "timeframe": "1h", "gate_pass": False,
                    "augmented": {
                        "trade_share": 0.65,
                        "post_fee_pnl_pct_total": -10.0,
                    },
                    "da_lift": 0.0,
                },
                {  # in-band, gate pass
                    "coin": "b", "timeframe": "1h", "gate_pass": True,
                    "augmented": {
                        "trade_share": 0.6,
                        "post_fee_pnl_pct_total": 25.0,
                    },
                    "da_lift": 0.05,
                },
            ],
        },
        "run_metadata": {
            "wall_time_seconds": 70.5, "n_workers": 6, "n_work_units": 38,
        },
    }

    summary = loop._summarize_verdict_payload(payload)
    assert summary["captured_at"] == "20260429T033030Z"
    assert summary["timeframes_subset"] == ["1h", "2h"]
    assert summary["wall_time_seconds"] == 70.5
    # OFF: 1 in band ([0.40, 0.85]: 0.55), 1 out (0.95).
    assert summary["off"]["n_slices"] == 2
    assert summary["off"]["n_in_trade_share_band"] == 1
    assert summary["off"]["n_passing_gate"] == 1
    # ON: both slices in [0.40, 0.85].
    assert summary["on"]["n_in_trade_share_band"] == 2
    assert summary["on"]["sum_pnl_pct_total_aug"] == 15.0


def test_run_calibration_verdict_handles_subprocess_nonzero_exit(
    datasets_dir, monkeypatch, tmp_path,
):
    """Task #599 — when the subprocess exits non-zero, the helper must
    emit a structured status dict (no exception) so the calling tick
    can keep going.
    """
    from scripts import scheduled_refresh_loop as loop
    import subprocess

    class _FakeProc:
        def __init__(self):
            self.returncode = 4
            self.stdout = ""
            self.stderr = "boom from worker"

    def _fake_run(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = loop._run_calibration_verdict(["1h"])
    assert result["last_status"] == "error"
    assert "rc=4" in result["last_error"]
    assert "boom from worker" in (result.get("stderr_tail") or "")
    assert result["trigger_timeframes"] == ["1h"]


def test_run_calibration_verdict_handles_subprocess_timeout(
    datasets_dir, monkeypatch,
):
    """Task #599 — a hung subprocess must be killed by the timeout and
    surface as ``last_status='timeout'`` rather than wedging the loop.
    """
    from scripts import scheduled_refresh_loop as loop
    import subprocess

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("ML_REFRESH_CALIBRATION_VERDICT_TIMEOUT_S", "5")

    result = loop._run_calibration_verdict(["1h", "2h"])
    assert result["last_status"] == "timeout"
    assert "5s" in result["last_error"]
    assert result["timeout_seconds"] == 5


def test_run_calibration_verdict_missing_report_after_success(
    datasets_dir, monkeypatch,
):
    """Task #599 — if the subprocess exits 0 but no
    ``*-task592-1h2h-stage2-verdict.json`` is on disk afterwards,
    treat that as an error rather than silently writing an empty
    summary. Catches the "subprocess succeeded but failed to flush
    the report" failure mode.
    """
    from scripts import scheduled_refresh_loop as loop
    import subprocess

    class _FakeProc:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeProc())
    # Point the loop's REPORTS_DIR at an empty tmp dir.
    monkeypatch.setattr(loop, "REPORTS_DIR", datasets_dir / "_empty_reports")

    result = loop._run_calibration_verdict(["1h"])
    assert result["last_status"] == "error"
    assert "no" in result["last_error"] and "report" in result["last_error"]
