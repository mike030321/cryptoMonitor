"""Task #410 — daily 5m head top-up scheduler.

Covers the pure-logic surface of `app.scheduled_5m_topup` without
touching OKX or Postgres:

  1. The contiguous-day measurement matches the campaign's gate (longest
     run of consecutive 5m buckets, in days, gaps reset the run).
  2. `_emit_alerts` flags every coin under the threshold and writes a
     structured progress record; coins at-or-above stay silent.
  3. `_tick` honours the `ENABLED=False` short-circuit so tests / a
     production opt-out never accidentally hit the network.
  4. `start_scheduler` is a no-op when `ML_AUTO_RETRAIN_TEST_DISABLE=1`
     (the same env switch the other ml-engine daemons honour, set by
     `tests/conftest.py`).
  5. The FastAPI app exposes `/ml/admin/5m-topup/status` and the body
     matches the in-memory state dict.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import scheduled_5m_topup as topup
from app.main import app


# ── 1. contiguous_days measurement ────────────────────────────────────────
def _make_buckets(start: datetime, n: int, *, gap_at: int | None = None):
    """Return n consecutive 5m bucket_starts. If `gap_at` is given, skip
    that index so the first run length is `gap_at` (not n).
    """
    out = []
    for i in range(n):
        if gap_at is not None and i == gap_at:
            continue
        out.append(start + timedelta(minutes=5 * i))
    return out


def test_contiguous_days_longest_run_only(monkeypatch):
    """The measurement must report the LONGEST contiguous run, not the
    span between earliest and latest. A coin with a 100-bucket gap in
    the middle keeps only the longer of the two runs.
    """
    # Stand in for `db_mod.init_pool().acquire().fetch(...)` so the test
    # never touches Postgres. We feed back fabricated rows via a fake
    # connection / pool stack matching asyncpg's sync API surface.
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Run A: 5 contiguous buckets, then a gap (skip i=5..104), then run B
    # of 200 contiguous buckets. Longest = 200 → 200*300/86400 ≈ 0.69 d.
    rows_a = [{"bucket_start": b} for b in _make_buckets(start, 5)]
    rows_b = [
        {"bucket_start": (start + timedelta(minutes=5 * (105 + i)))}
        for i in range(200)
    ]
    rows = rows_a + rows_b

    class _Conn:
        async def fetch(self, *_args, **_kwargs):
            return rows

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_init_pool():
        return _Pool()

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)

    import asyncio
    health = asyncio.run(topup.measure_contiguous_5m(["foo"]))
    expected_days = round((200 * 300) / 86400.0, 2)
    assert health == {"foo": expected_days}


def test_contiguous_days_empty_table_is_zero(monkeypatch):
    class _Conn:
        async def fetch(self, *_args, **_kwargs):
            return []

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_init_pool():
        return _Pool()

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)

    import asyncio
    assert asyncio.run(topup.measure_contiguous_5m(["foo"])) == {"foo": 0.0}


# ── 2. alerting + progress journal ────────────────────────────────────────
def test_emit_alerts_flags_below_and_writes_progress(tmp_path, monkeypatch):
    """Every coin below `threshold` MUST land in the returned alert list,
    in the structured log, and in the progress JSONL. Coins at-or-above
    must NOT show up anywhere.
    """
    progress_path = tmp_path / "progress_updates.jsonl"
    monkeypatch.setattr(topup, "PROGRESS_PATH", progress_path)

    health = {
        "below_a": 280.0,    # alert
        "below_b": 309.99,   # alert (just under)
        "edge":    310.0,    # NOT an alert (at threshold)
        "above":   320.0,    # NOT an alert
    }
    alerts = topup._emit_alerts(health, threshold=310.0)
    assert alerts == ["below_a", "below_b"]

    # Progress entry should exist with the structured payload the
    # dashboard can read alongside the campaign's other entries.
    raw = progress_path.read_text().strip().splitlines()
    assert len(raw) == 1
    entry = json.loads(raw[0])
    assert entry["phase"] == "5m_topup_health"
    assert entry["status"] == "alert"
    assert entry["coins"] == ["below_a", "below_b"]
    assert entry["threshold_days"] == 310.0
    # only the offending coins make it into the per-coin payload
    assert set(entry["per_coin_contiguous_days"].keys()) == {"below_a", "below_b"}


def test_emit_alerts_no_progress_entry_when_all_healthy(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress_updates.jsonl"
    monkeypatch.setattr(topup, "PROGRESS_PATH", progress_path)

    alerts = topup._emit_alerts(
        {"a": 320.0, "b": 311.0}, threshold=310.0,
    )
    assert alerts == []
    # No file is created because `_append_progress` is only called when
    # there's something to alert on.
    assert not progress_path.exists()


# ── 3. tick honours ENABLED ──────────────────────────────────────────────
def test_tick_short_circuits_when_disabled(monkeypatch):
    """Disabling the loop via env must skip the OKX call entirely so a
    test / production opt-out can never accidentally hit the network."""
    monkeypatch.setattr(topup, "ENABLED", False)
    # Reset state so the assertion about outcome is unambiguous.
    topup.state["last_attempt_outcome"] = None
    out = topup._tick(now=1700000000.0)
    assert out == "disabled"
    assert topup.state["last_attempt_outcome"] == "disabled"
    assert topup.state["last_check_at"] == 1700000000.0


# ── 4. start_scheduler is inert in tests ─────────────────────────────────
def test_start_scheduler_skipped_when_test_env_disable_set():
    """conftest.py sets ML_AUTO_RETRAIN_TEST_DISABLE=1, which the new
    scheduler MUST honour — otherwise the FastAPI test client would
    spawn a daemon thread that hits OKX on every test run.
    """
    import os
    assert os.environ.get("ML_AUTO_RETRAIN_TEST_DISABLE") == "1"
    # Calling it directly must not spawn a thread.
    before = topup._thread
    topup.start_scheduler()
    assert topup._thread is before, (
        "start_scheduler must short-circuit when ML_AUTO_RETRAIN_TEST_DISABLE=1"
    )


# ── 5. _run_once_blocking restores BOTH _pool and _pool_loop ─────────────
def test_run_once_blocking_restores_pool_loop_metadata(monkeypatch):
    """Code-review fix: `_run_once_blocking` swaps the module-level
    `db_mod._pool` reference around `asyncio.run(_run_once_async())` so
    the lifespan pool is never bound to the scheduler's fresh loop. It
    MUST restore `db_mod._pool_loop` as well — otherwise the next
    lifespan-loop caller finds `_pool` set but `_pool_loop` pointing at
    the now-closed scheduler loop and pays a forced pool rebuild.
    """
    sentinel_pool = object()
    sentinel_loop = object()
    monkeypatch.setattr(topup.db_mod, "_pool", sentinel_pool)
    monkeypatch.setattr(topup.db_mod, "_pool_loop", sentinel_loop)

    # Stub out the actual work so we don't hit OKX or Postgres.
    async def _fake_run_once_async():
        # Inside the fresh asyncio loop the global _pool/_pool_loop
        # MUST be cleared so init_pool() builds a new pool against this
        # loop instead of trying to reuse the lifespan-bound pool.
        assert topup.db_mod._pool is None
        assert topup.db_mod._pool_loop is None
        # Simulate init_pool() stamping the fresh loop's metadata.
        topup.db_mod._pool = object()
        topup.db_mod._pool_loop = object()
        return {
            "coins": [], "inserted_per_coin": {},
            "contiguous_days_per_coin": {}, "alerts": [],
        }

    monkeypatch.setattr(topup, "_run_once_async", _fake_run_once_async)

    topup._run_once_blocking()

    # After the blocking call returns, BOTH module-level fields must be
    # back to the lifespan values — anything else means the next
    # lifespan-loop caller will see a stale loop binding and rebuild.
    assert topup.db_mod._pool is sentinel_pool
    assert topup.db_mod._pool_loop is sentinel_loop


# ── 6. multi-replica advisory lock (task #413) ───────────────────────────
def test_run_once_async_skips_when_lock_unavailable(monkeypatch):
    """When `try_advisory_lock` yields False (another ml-engine replica
    holds the lock) the coroutine MUST return `skipped_locked=True`
    without ever calling into `topup_5m_head` / `measure_contiguous_5m`.
    This is the whole point of task #413 — losers don't pound OKX.
    """
    import asyncio
    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_lock_unavailable(_key):
        yield False

    monkeypatch.setattr(topup.db_mod, "try_advisory_lock", _fake_lock_unavailable)

    # If the body actually ran, these would be called and the test would
    # fail loud — they're set to async functions that raise.
    async def _explode(*_a, **_k):
        raise AssertionError("must not run when advisory lock unavailable")

    monkeypatch.setattr(topup, "topup_5m_head", _explode)
    monkeypatch.setattr(topup, "measure_contiguous_5m", _explode)

    out = asyncio.run(topup._run_once_async())
    assert out["skipped_locked"] is True
    assert out["inserted_per_coin"] == {}
    assert out["contiguous_days_per_coin"] == {}
    assert out["alerts"] == []
    # Task #424 contract: skipped result must still carry the winner
    # keys (set to None) so the dashboard can rely on the response
    # shape without conditional defensive checks.
    assert out["winning_replica"] is None
    assert out["winning_at"] is None


def test_run_once_async_runs_body_when_lock_acquired(monkeypatch):
    """When the lock IS won the coroutine must do the normal work and
    report `skipped_locked=False`. Guards against a future refactor that
    accidentally inverts the lock check."""
    import asyncio
    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_lock_acquired(_key):
        yield True

    monkeypatch.setattr(topup.db_mod, "try_advisory_lock", _fake_lock_acquired)

    async def _fake_topup(coins, *, window_days):
        return {c: 7 for c in coins}

    async def _fake_health(coins):
        return {c: 320.0 for c in coins}

    monkeypatch.setattr(topup, "topup_5m_head", _fake_topup)
    monkeypatch.setattr(topup, "measure_contiguous_5m", _fake_health)
    monkeypatch.setattr(topup, "_coins", lambda: ["btc", "eth"])
    # silence the alert journal write
    monkeypatch.setattr(topup, "_emit_alerts", lambda h, threshold: [])

    # Stub the cross-replica winner write so the test never touches
    # Postgres. The recorded args are asserted below to confirm the
    # winner persistence happens *inside* the held-lock branch.
    recorded: list[dict] = []

    async def _fake_record(*, replica, tick_at):
        recorded.append({"replica": replica, "tick_at": tick_at})

    monkeypatch.setattr(topup, "_record_winning_replica", _fake_record)

    out = asyncio.run(topup._run_once_async())
    assert out["skipped_locked"] is False
    assert out["coins"] == ["btc", "eth"]
    assert out["inserted_per_coin"] == {"btc": 7, "eth": 7}
    assert out["contiguous_days_per_coin"] == {"btc": 320.0, "eth": 320.0}
    # The winner must be recorded exactly once per successful tick and
    # surfaced back to the caller so `_tick` can stamp it into state.
    assert len(recorded) == 1
    assert isinstance(out["winning_replica"], str) and out["winning_replica"]
    assert isinstance(out["winning_at"], str) and out["winning_at"]
    assert recorded[0]["replica"] == out["winning_replica"]


def test_run_once_async_skipped_branch_does_not_record_winner(monkeypatch):
    """Task #424: a replica that LOSES the advisory lock must NOT touch
    the shared `last_winner` row — otherwise a flapping loser could
    overwrite the actual winner's record. Result must still carry the
    keys (set to None) so callers can rely on the shape.
    """
    import asyncio
    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_lock_unavailable(_key):
        yield False

    monkeypatch.setattr(topup.db_mod, "try_advisory_lock", _fake_lock_unavailable)

    called = False

    async def _fake_record(*, replica, tick_at):
        nonlocal called
        called = True

    monkeypatch.setattr(topup, "_record_winning_replica", _fake_record)

    out = asyncio.run(topup._run_once_async())
    assert out["skipped_locked"] is True
    assert out["winning_replica"] is None
    assert out["winning_at"] is None
    assert called is False


def test_tick_records_skipped_locked_outcome(monkeypatch):
    """When `_run_once_blocking` reports the lock was held by another
    process, `_tick` must record the dedicated outcome string and bump
    the multi-replica skip counter — otherwise operators can't tell from
    the dashboard whether the daily top-up actually happened on this box.
    """
    monkeypatch.setattr(topup, "ENABLED", True)

    def _fake_run_once_blocking():
        return {
            "skipped_locked": True,
            "coins": [],
            "inserted_per_coin": {},
            "contiguous_days_per_coin": {},
            "alerts": [],
            "winning_replica": None,
            "winning_at": None,
        }

    monkeypatch.setattr(topup, "_run_once_blocking", _fake_run_once_blocking)
    topup.state["last_attempt_outcome"] = None
    before = int(topup.state.get("skips_locked_total") or 0)
    runs_before = int(topup.state.get("runs_total") or 0)
    rows_before = int(topup.state.get("rows_inserted_total") or 0)

    out = topup._tick(now=1700000001.0)
    assert out == "skipped_locked_other_process"
    assert topup.state["last_attempt_outcome"] == "skipped_locked_other_process"
    assert topup.state["skips_locked_total"] == before + 1
    # A skipped tick must NOT count as a successful run nor inflate the
    # rows-inserted totals — those track real OKX work only.
    assert topup.state["runs_total"] == runs_before
    assert topup.state["rows_inserted_total"] == rows_before


# ── 8. cross-replica winner attribution (task #424) ──────────────────────
def test_replica_identity_includes_pid():
    """The replica identity must contain the current process pid so two
    processes on the same hostname can still be told apart."""
    import os
    ident = topup._replica_identity()
    assert isinstance(ident, str) and ident
    assert f"pid={os.getpid()}" in ident


def test_tick_stamps_winning_replica_into_state(monkeypatch):
    """Task #424: a successful tick must write `last_winning_replica`
    and `last_winning_at` into the state dict so a single-replica
    deployment renders the field without needing a DB round-trip."""
    monkeypatch.setattr(topup, "ENABLED", True)

    def _fake_run_once_blocking():
        return {
            "skipped_locked": False,
            "coins": ["btc"],
            "inserted_per_coin": {"btc": 5},
            "contiguous_days_per_coin": {"btc": 320.0},
            "alerts": [],
            "winning_replica": "host-a/pid=42",
            "winning_at": "2026-04-24T12:00:00+00:00",
        }

    monkeypatch.setattr(topup, "_run_once_blocking", _fake_run_once_blocking)
    topup.state["last_winning_replica"] = None
    topup.state["last_winning_at"] = None

    out = topup._tick(now=1700000099.0)
    assert out == "ok"
    assert topup.state["last_winning_replica"] == "host-a/pid=42"
    assert topup.state["last_winning_at"] == "2026-04-24T12:00:00+00:00"


def test_load_last_winner_parses_jsonb_dict_and_string(monkeypatch):
    """`_load_last_winner` must accept both the dict-shaped jsonb codec
    output (asyncpg with `set_type_codec`) and the raw-string fallback
    so it works regardless of how the pool is configured."""
    import asyncio

    payloads = [
        {"replica": "host-x/pid=1", "tick_at": "2026-04-24T10:00:00+00:00"},
        json.dumps({"replica": "host-y/pid=2", "tick_at": "2026-04-24T11:00:00+00:00"}),
    ]

    for payload in payloads:
        class _Conn:
            def __init__(self, p):
                self._p = p

            async def fetchrow(self, *_args, **_kwargs):
                return {"value": self._p}

        class _Acquire:
            def __init__(self, p):
                self._p = p

            async def __aenter__(self):
                return _Conn(self._p)

            async def __aexit__(self, *_):
                return False

        class _Pool:
            def __init__(self, p):
                self._p = p

            def acquire(self):
                return _Acquire(self._p)

        async def _fake_init_pool(p=payload):
            return _Pool(p)

        monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)
        out = asyncio.run(topup._load_last_winner())
        assert out is not None
        assert out["replica"].startswith("host-")
        assert "tick_at" in out


def test_load_last_winner_returns_none_when_no_row(monkeypatch):
    """No tick has ever recorded a winner yet → `_load_last_winner`
    must return None so the status endpoint leaves the in-memory
    `last_winning_replica` (also None) untouched."""
    import asyncio

    class _Conn:
        async def fetchrow(self, *_args, **_kwargs):
            return None

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_init_pool():
        return _Pool()

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)
    assert asyncio.run(topup._load_last_winner()) is None


def test_status_endpoint_overlays_db_winner(monkeypatch):
    """Task #424: the status endpoint must merge the cross-replica
    winner row into the response, even when the local in-memory state
    has never seen a successful tick (because some other replica did
    every pull). This is the fix for "operator on replica B sees
    `last_winning_replica=None` even though replica A did the work"."""
    import app.scheduled_5m_topup as topup_mod

    async def _fake_load():
        return {
            "replica": "host-winner/pid=99",
            "tick_at": "2026-04-24T13:30:00+00:00",
        }

    monkeypatch.setattr(topup_mod, "_load_last_winner", _fake_load)
    # Local state mimics a replica that has only ever skipped.
    topup_mod.state["last_winning_replica"] = None
    topup_mod.state["last_winning_at"] = None

    with TestClient(app) as client:
        r = client.get("/ml/admin/5m-topup/status")
        assert r.status_code == 200
        body = r.json()
        assert body["last_winning_replica"] == "host-winner/pid=99"
        assert body["last_winning_at"] == "2026-04-24T13:30:00+00:00"


def test_status_endpoint_keeps_local_state_when_db_empty(monkeypatch):
    """Inverse of the overlay test: if the DB row doesn't exist yet, the
    endpoint must surface whatever the local state has (also None on a
    fresh boot) rather than overwriting with garbage."""
    import app.scheduled_5m_topup as topup_mod

    async def _fake_load_none():
        return None

    monkeypatch.setattr(topup_mod, "_load_last_winner", _fake_load_none)
    topup_mod.state["last_winning_replica"] = "host-local/pid=7"
    topup_mod.state["last_winning_at"] = "2026-04-24T09:00:00+00:00"

    with TestClient(app) as client:
        r = client.get("/ml/admin/5m-topup/status")
        assert r.status_code == 200
        body = r.json()
        # Local state survives untouched because the overlay had nothing
        # to merge in.
        assert body["last_winning_replica"] == "host-local/pid=7"
        assert body["last_winning_at"] == "2026-04-24T09:00:00+00:00"


def test_advisory_lock_key_is_stable_signed_bigint():
    """The advisory key must be deterministic (same label → same key on
    every replica) and stay inside Postgres's signed bigint range so
    `pg_try_advisory_lock(bigint)` accepts it without overflow."""
    from app import db as db_mod

    k1 = db_mod.advisory_lock_key("ml_engine.scheduled_5m_topup")
    k2 = db_mod.advisory_lock_key("ml_engine.scheduled_5m_topup")
    k3 = db_mod.advisory_lock_key("ml_engine.something_else")
    assert k1 == k2
    assert k1 != k3
    # signed int64 range
    assert -(2**63) <= k1 < 2**63
    assert -(2**63) <= k3 < 2**63
    # And the scheduler module exposes the resolved key in its state so
    # operators can correlate `pg_locks` rows with the dashboard.
    assert topup.state["advisory_lock_key"] == k1
    assert topup.state["advisory_lock_label"] == "ml_engine.scheduled_5m_topup"


# ── 9. recent-winners history (task #435) ────────────────────────────────
def test_parse_recent_winners_accepts_dict_and_string():
    """The asyncpg jsonb codec may parse the row into a Python list, or
    hand back the raw JSON string — both shapes must round-trip cleanly
    so the dashboard never silently shows an empty list when the row is
    actually populated."""
    raw_dict = [
        {"replica": "host-a/pid=1", "tick_at": "2026-04-24T00:00:00+00:00"},
        {"replica": "host-b/pid=2", "tick_at": "2026-04-23T00:00:00+00:00"},
    ]
    assert topup._parse_recent_winners(raw_dict) == raw_dict
    assert topup._parse_recent_winners(json.dumps(raw_dict)) == raw_dict


def test_parse_recent_winners_drops_garbage_silently():
    """A corrupt row must not crash the tick or the status endpoint —
    every malformed entry is dropped and the rest survive."""
    assert topup._parse_recent_winners(None) == []
    assert topup._parse_recent_winners("not json") == []
    assert topup._parse_recent_winners({"not": "a list"}) == []
    mixed = [
        {"replica": "ok/pid=1", "tick_at": "2026-04-24T00:00:00+00:00"},
        {"replica": 12345, "tick_at": "x"},      # non-string replica
        {"replica": "x"},                         # missing tick_at
        "string-entry",                           # not a dict
    ]
    out = topup._parse_recent_winners(mixed)
    assert out == [{"replica": "ok/pid=1", "tick_at": "2026-04-24T00:00:00+00:00"}]


def test_load_recent_winners_returns_empty_when_no_row(monkeypatch):
    """No tick has ever recorded a winner → the endpoint must return
    an empty list, not crash on the missing row."""
    import asyncio

    class _Conn:
        async def fetchrow(self, *_args, **_kwargs):
            return None

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_init_pool():
        return _Pool()

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)
    assert asyncio.run(topup._load_recent_winners(limit=14)) == []


def test_load_recent_winners_respects_limit(monkeypatch):
    """The list is stored newest-first; `_load_recent_winners` must
    slice the head so the dashboard's ``?limit=14`` only ever pays for
    14 rows even if the writer has accumulated the full cap of 50."""
    import asyncio

    payload = [
        {"replica": f"host-{i}/pid=1", "tick_at": f"2026-04-{i:02d}T00:00:00+00:00"}
        for i in range(1, 21)
    ]

    class _Conn:
        async def fetchrow(self, *_args, **_kwargs):
            return {"value": payload}

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_init_pool():
        return _Pool()

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)
    out = asyncio.run(topup._load_recent_winners(limit=5))
    assert len(out) == 5
    # newest-first slice means we keep the head, NOT the tail
    assert out == payload[:5]


def test_load_recent_winners_returns_empty_on_db_error(monkeypatch):
    """A DB hiccup on this best-effort read must not blow up the
    status endpoint — it just returns []."""
    import asyncio

    async def _fake_init_pool():
        raise RuntimeError("pool exploded")

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)
    assert asyncio.run(topup._load_recent_winners(limit=14)) == []


def test_record_winning_replica_appends_to_recent_list(monkeypatch):
    """Every successful tick must (a) UPSERT the single last-winner row
    AND (b) prepend the new winner to the recent-winners list inside
    one transaction. Three sequential ticks must leave the list in
    newest-first order with all three entries."""
    import asyncio

    state_rows: dict[str, str] = {}

    class _Tx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    class _Conn:
        def transaction(self):
            return _Tx()

        async def execute(self, sql: str, key: str, value: str):
            # The implementation issues an UPSERT against app_settings
            # twice per tick (last_winner + recent_winners). Both write
            # the JSON payload as the second positional arg.
            assert "INSERT INTO app_settings" in sql
            state_rows[key] = value

        async def fetchrow(self, sql: str, key: str):
            assert "SELECT value FROM app_settings" in sql
            v = state_rows.get(key)
            if v is None:
                return None
            return {"value": v}

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_init_pool():
        return _Pool()

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)

    async def _drive():
        await topup._record_winning_replica(
            replica="host-a/pid=1",
            tick_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        )
        await topup._record_winning_replica(
            replica="host-b/pid=2",
            tick_at=datetime(2026, 4, 23, tzinfo=timezone.utc),
        )
        await topup._record_winning_replica(
            replica="host-c/pid=3",
            tick_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        )

    asyncio.run(_drive())

    # last_winner row reflects the final tick.
    last = json.loads(state_rows[topup._LAST_WINNER_SETTING_KEY])
    assert last["replica"] == "host-c/pid=3"

    # recent_winners is newest-first with all three appended.
    recent = json.loads(state_rows[topup._RECENT_WINNERS_SETTING_KEY])
    assert [r["replica"] for r in recent] == [
        "host-c/pid=3", "host-b/pid=2", "host-a/pid=1",
    ]


def test_record_winning_replica_caps_recent_list(monkeypatch):
    """The recent-winners list must not grow without bound. After
    `_RECENT_WINNERS_MAX + 5` writes, the row must hold exactly
    `_RECENT_WINNERS_MAX` entries — the oldest 5 fall off."""
    import asyncio

    state_rows: dict[str, str] = {}

    class _Tx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    class _Conn:
        def transaction(self):
            return _Tx()

        async def execute(self, sql: str, key: str, value: str):
            state_rows[key] = value

        async def fetchrow(self, sql: str, key: str):
            v = state_rows.get(key)
            return None if v is None else {"value": v}

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_init_pool():
        return _Pool()

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)

    async def _drive():
        for i in range(topup._RECENT_WINNERS_MAX + 5):
            await topup._record_winning_replica(
                replica=f"host-{i}/pid=1",
                tick_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
            )

    asyncio.run(_drive())
    recent = json.loads(state_rows[topup._RECENT_WINNERS_SETTING_KEY])
    assert len(recent) == topup._RECENT_WINNERS_MAX
    # The newest entry is at the head; the oldest 5 have rolled off.
    assert recent[0]["replica"] == f"host-{topup._RECENT_WINNERS_MAX + 4}/pid=1"
    assert recent[-1]["replica"] == "host-5/pid=1"


def test_recent_winners_endpoint_returns_list(monkeypatch):
    """The new `/ml/admin/5m-topup/recent-winners` endpoint must surface
    whatever `_load_recent_winners` returned plus the resolved limit /
    cap so the dashboard can render the list and trust its ordering."""
    import app.scheduled_5m_topup as topup_mod

    payload = [
        {"replica": "host-a/pid=1", "tick_at": "2026-04-24T00:00:00+00:00"},
        {"replica": "host-b/pid=2", "tick_at": "2026-04-23T00:00:00+00:00"},
    ]

    async def _fake_load(*, limit):
        assert limit == 14
        return payload

    monkeypatch.setattr(topup_mod, "_load_recent_winners", _fake_load)

    with TestClient(app) as client:
        r = client.get("/ml/admin/5m-topup/recent-winners")
        assert r.status_code == 200
        body = r.json()
        assert body["winners"] == payload
        assert body["limit"] == 14
        assert body["max"] == topup_mod._RECENT_WINNERS_MAX


def test_recent_winners_endpoint_clamps_limit(monkeypatch):
    """A typo'd or malicious ``?limit`` must be clamped into the safe
    range — `0` becomes `1`, anything above the cap becomes the cap."""
    import app.scheduled_5m_topup as topup_mod

    seen_limits: list[int] = []

    async def _fake_load(*, limit):
        seen_limits.append(limit)
        return []

    monkeypatch.setattr(topup_mod, "_load_recent_winners", _fake_load)

    with TestClient(app) as client:
        r1 = client.get("/ml/admin/5m-topup/recent-winners?limit=0")
        assert r1.status_code == 200
        assert r1.json()["limit"] == 1

        r2 = client.get(
            f"/ml/admin/5m-topup/recent-winners?limit={topup_mod._RECENT_WINNERS_MAX + 100}",
        )
        assert r2.status_code == 200
        assert r2.json()["limit"] == topup_mod._RECENT_WINNERS_MAX

    assert seen_limits == [1, topup_mod._RECENT_WINNERS_MAX]


def test_recent_winners_endpoint_falls_back_on_non_numeric_limit(monkeypatch):
    """A non-numeric ``?limit=abc`` must fall back to the default of
    14 instead of returning a 422. Operator-facing dashboards should
    never see a 4xx error from a typo in a query string — they just
    want the recent list."""
    import app.scheduled_5m_topup as topup_mod

    seen_limits: list[int] = []

    async def _fake_load(*, limit):
        seen_limits.append(limit)
        return []

    monkeypatch.setattr(topup_mod, "_load_recent_winners", _fake_load)

    with TestClient(app) as client:
        r = client.get("/ml/admin/5m-topup/recent-winners?limit=oops")
        assert r.status_code == 200
        assert r.json()["limit"] == topup_mod._RECENT_WINNERS_DEFAULT_LIMIT

    assert seen_limits == [topup_mod._RECENT_WINNERS_DEFAULT_LIMIT]


def test_status_endpoint_returns_state_dict():
    """The dashboard needs a stable, read-only surface to render
    'last topped up at X, alerts: [...]'. The endpoint just mirrors the
    in-memory state dict — assert the contract shape here so a future
    refactor that drops a key fails loud.
    """
    with TestClient(app) as client:
        r = client.get("/ml/admin/5m-topup/status")
        assert r.status_code == 200
        body = r.json()
        for key in (
            "enabled", "interval_seconds", "window_days", "alert_below_days",
            "last_check_at", "last_attempt_outcome", "last_finished_at",
            "last_error", "last_topup_inserted", "last_topup_per_coin",
            "last_health_per_coin", "last_alerts",
            "ticks_total", "runs_total", "rows_inserted_total",
            "alerts_emitted_total",
            # Multi-replica advisory-lock telemetry (task #413)
            "skips_locked_total", "advisory_lock_label", "advisory_lock_key",
            # Cross-replica winner attribution (task #424)
            "last_winning_replica", "last_winning_at",
            # Stuck-replica detection (task #442)
            "stuck_replica_threshold", "stuck_replica_streak",
            "stuck_replica", "stuck_replica_alerts_total",
        ):
            assert key in body, f"status endpoint missing key {key}"


# ── 10. stuck-replica detection (task #442) ──────────────────────────────
def test_compute_winner_streak_counts_leading_run():
    """The streak is the count of leading entries whose ``replica``
    matches the head — earlier matches separated by a different replica
    do NOT extend it. This is the whole point: we want to detect "host-a
    has won the LAST N ticks", not "host-a has won N times total".
    """
    head, n = topup._compute_winner_streak([
        {"replica": "host-a/pid=1", "tick_at": "t3"},
        {"replica": "host-a/pid=1", "tick_at": "t2"},
        {"replica": "host-a/pid=1", "tick_at": "t1"},
        {"replica": "host-b/pid=2", "tick_at": "t0"},
        {"replica": "host-a/pid=1", "tick_at": "t-1"},
    ])
    assert head == "host-a/pid=1"
    assert n == 3


def test_compute_winner_streak_handles_empty_and_single():
    assert topup._compute_winner_streak([]) == (None, 0)
    head, n = topup._compute_winner_streak([
        {"replica": "host-z/pid=9", "tick_at": "t"},
    ])
    assert head == "host-z/pid=9"
    assert n == 1


def test_compute_winner_streak_drops_garbage():
    """A corrupt row (missing fields, wrong types) must not crash
    streak detection — the helper must defensively bail out."""
    assert topup._compute_winner_streak([{"not": "a replica"}]) == (None, 0)
    assert topup._compute_winner_streak([{"replica": 123}]) == (None, 0)
    # Non-dict head means we can't even pick a head — return zeroes.
    assert topup._compute_winner_streak(["string"]) == (None, 0)


def test_emit_stuck_replica_alert_writes_progress(tmp_path, monkeypatch):
    """Crossing the threshold must (a) log the structured warning the
    api-server notifier polls for and (b) append a progress journal
    entry the campaign reader can also see. The phase string is the
    contract the dashboard / notifier filter on, so assert it
    explicitly."""
    progress_path = tmp_path / "progress_updates.jsonl"
    monkeypatch.setattr(topup, "PROGRESS_PATH", progress_path)

    topup._emit_stuck_replica_alert(
        replica="host-a/pid=1", streak=9, threshold=7,
    )
    raw = progress_path.read_text().strip().splitlines()
    assert len(raw) == 1
    entry = json.loads(raw[0])
    assert entry["phase"] == "5m_topup_stuck_replica"
    assert entry["status"] == "alert"
    assert entry["replica"] == "host-a/pid=1"
    assert entry["consecutive_wins"] == 9
    assert entry["threshold"] == 7
    assert "9 ticks in a row" in entry["headline"]


def test_run_once_async_emits_stuck_replica_alert_when_threshold_crossed(monkeypatch):
    """The whole pipeline: a tick whose recent_winners list shows the
    same head replica >= threshold times in a row must (a) fire
    `_emit_stuck_replica_alert` AND (b) surface the streak + replica
    in the result dict so `_tick` can stamp it into state."""
    import asyncio
    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_lock(_key):
        yield True

    monkeypatch.setattr(topup.db_mod, "try_advisory_lock", _fake_lock)

    async def _fake_topup(coins, *, window_days):
        return {c: 0 for c in coins}

    async def _fake_health(coins):
        return {c: 320.0 for c in coins}

    monkeypatch.setattr(topup, "topup_5m_head", _fake_topup)
    monkeypatch.setattr(topup, "measure_contiguous_5m", _fake_health)
    monkeypatch.setattr(topup, "_coins", lambda: ["btc"])
    monkeypatch.setattr(topup, "_emit_alerts", lambda h, threshold: [])
    monkeypatch.setattr(topup, "STUCK_REPLICA_THRESHOLD", 3)

    # Pretend the recent_winners row already had 2 wins by host-a; this
    # tick (also host-a) makes it 3 — exactly the threshold.
    fake_after = [
        {"replica": "host-a/pid=1", "tick_at": "t3"},
        {"replica": "host-a/pid=1", "tick_at": "t2"},
        {"replica": "host-a/pid=1", "tick_at": "t1"},
    ]

    async def _fake_record(*, replica, tick_at):  # noqa: ARG001
        return fake_after

    monkeypatch.setattr(topup, "_record_winning_replica", _fake_record)
    monkeypatch.setattr(topup, "_replica_identity", lambda: "host-a/pid=1")

    emitted: list[dict] = []

    def _fake_emit(*, replica, streak, threshold):
        emitted.append({"replica": replica, "streak": streak, "threshold": threshold})

    monkeypatch.setattr(topup, "_emit_stuck_replica_alert", _fake_emit)

    out = asyncio.run(topup._run_once_async())
    assert out["skipped_locked"] is False
    assert out["stuck_replica"] == "host-a/pid=1"
    assert out["stuck_replica_streak"] == 3
    assert out["stuck_replica_threshold"] == 3
    assert out["stuck_replica_alert_emitted"] is True
    assert emitted == [{"replica": "host-a/pid=1", "streak": 3, "threshold": 3}]


def test_run_once_async_does_not_emit_below_threshold(monkeypatch):
    """A streak below threshold must NOT call the emitter — operators
    only want to be paged once a real run forms, not on every tick
    that adds to a 1-or-2-wide streak."""
    import asyncio
    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_lock(_key):
        yield True

    monkeypatch.setattr(topup.db_mod, "try_advisory_lock", _fake_lock)

    async def _fake_topup(coins, *, window_days):
        return {c: 0 for c in coins}

    async def _fake_health(coins):
        return {c: 320.0 for c in coins}

    monkeypatch.setattr(topup, "topup_5m_head", _fake_topup)
    monkeypatch.setattr(topup, "measure_contiguous_5m", _fake_health)
    monkeypatch.setattr(topup, "_coins", lambda: ["btc"])
    monkeypatch.setattr(topup, "_emit_alerts", lambda h, threshold: [])
    monkeypatch.setattr(topup, "STUCK_REPLICA_THRESHOLD", 7)

    async def _fake_record(*, replica, tick_at):  # noqa: ARG001
        # Only 2 entries — well below the threshold of 7.
        return [
            {"replica": "host-a/pid=1", "tick_at": "t2"},
            {"replica": "host-a/pid=1", "tick_at": "t1"},
        ]

    monkeypatch.setattr(topup, "_record_winning_replica", _fake_record)
    monkeypatch.setattr(topup, "_replica_identity", lambda: "host-a/pid=1")

    emitted: list[dict] = []

    def _fake_emit(*, replica, streak, threshold):
        emitted.append({"replica": replica})

    monkeypatch.setattr(topup, "_emit_stuck_replica_alert", _fake_emit)

    out = asyncio.run(topup._run_once_async())
    assert out["stuck_replica"] is None
    assert out["stuck_replica_streak"] == 2
    assert out["stuck_replica_alert_emitted"] is False
    assert emitted == []


def test_tick_bumps_stuck_alert_counter_only_when_emitted(monkeypatch):
    """`stuck_replica_alerts_total` MUST only increment on ticks that
    actually emitted the alert. Below-threshold ticks just update the
    streak field — the counter is the audit trail of how many pages
    the ml-engine fired off."""
    monkeypatch.setattr(topup, "ENABLED", True)

    def _fake_blocking_emit():
        return {
            "skipped_locked": False,
            "coins": [],
            "inserted_per_coin": {},
            "contiguous_days_per_coin": {},
            "alerts": [],
            "winning_replica": "host-a/pid=1",
            "winning_at": "2026-04-24T00:00:00+00:00",
            "stuck_replica_streak": 9,
            "stuck_replica": "host-a/pid=1",
            "stuck_replica_threshold": 7,
            "stuck_replica_alert_emitted": True,
        }

    def _fake_blocking_no_emit():
        return {
            "skipped_locked": False,
            "coins": [],
            "inserted_per_coin": {},
            "contiguous_days_per_coin": {},
            "alerts": [],
            "winning_replica": "host-a/pid=1",
            "winning_at": "2026-04-25T00:00:00+00:00",
            "stuck_replica_streak": 2,
            "stuck_replica": None,
            "stuck_replica_threshold": 7,
            "stuck_replica_alert_emitted": False,
        }

    topup.state["stuck_replica_alerts_total"] = 0
    topup.state["stuck_replica_streak"] = 0
    topup.state["stuck_replica"] = None

    monkeypatch.setattr(topup, "_run_once_blocking", _fake_blocking_emit)
    topup._tick(now=1700000300.0)
    assert topup.state["stuck_replica_alerts_total"] == 1
    assert topup.state["stuck_replica_streak"] == 9
    assert topup.state["stuck_replica"] == "host-a/pid=1"

    # A subsequent tick that does NOT cross the threshold must update
    # the streak/replica fields but leave the counter alone.
    monkeypatch.setattr(topup, "_run_once_blocking", _fake_blocking_no_emit)
    topup._tick(now=1700000400.0)
    assert topup.state["stuck_replica_alerts_total"] == 1, (
        "counter must only bump when an alert was actually emitted"
    )
    assert topup.state["stuck_replica_streak"] == 2
    assert topup.state["stuck_replica"] is None


def test_load_stuck_replica_overlay_returns_none_when_below_threshold(monkeypatch):
    """The overlay surfaces the streak verbatim, but only sets
    `stuck_replica` when the streak is at-or-above the configured
    threshold — the api-server notifier and the dashboard rely on
    that pre-applied check so they don't have to duplicate it."""
    import asyncio

    monkeypatch.setattr(topup, "STUCK_REPLICA_THRESHOLD", 7)

    async def _fake_load(*, limit):  # noqa: ARG001
        # Streak of 3 — below threshold of 7.
        return [
            {"replica": "host-a/pid=1", "tick_at": "t3"},
            {"replica": "host-a/pid=1", "tick_at": "t2"},
            {"replica": "host-a/pid=1", "tick_at": "t1"},
            {"replica": "host-b/pid=2", "tick_at": "t0"},
        ]

    monkeypatch.setattr(topup, "_load_recent_winners", _fake_load)
    out = asyncio.run(topup._load_stuck_replica_overlay())
    assert out["stuck_replica_streak"] == 3
    assert out["stuck_replica"] is None
    assert out["stuck_replica_threshold"] == 7


def test_load_stuck_replica_overlay_surfaces_replica_when_above(monkeypatch):
    """Inverse: the overlay must surface the head replica name once
    the streak crosses the threshold so the api-server notifier can
    fire its incident."""
    import asyncio

    monkeypatch.setattr(topup, "STUCK_REPLICA_THRESHOLD", 3)

    async def _fake_load(*, limit):  # noqa: ARG001
        return [
            {"replica": "host-a/pid=1", "tick_at": "t3"},
            {"replica": "host-a/pid=1", "tick_at": "t2"},
            {"replica": "host-a/pid=1", "tick_at": "t1"},
            {"replica": "host-b/pid=2", "tick_at": "t0"},
        ]

    monkeypatch.setattr(topup, "_load_recent_winners", _fake_load)
    out = asyncio.run(topup._load_stuck_replica_overlay())
    assert out["stuck_replica_streak"] == 3
    assert out["stuck_replica"] == "host-a/pid=1"


def test_status_endpoint_overlays_stuck_replica(monkeypatch):
    """A replica that has only ever LOST the lock still needs to surface
    the fleet-wide streak so its dashboard view doesn't lie. The status
    endpoint must overlay `_load_stuck_replica_overlay()` on top of the
    local state dict (which would otherwise show streak=0 for a perpetual
    loser)."""
    import app.scheduled_5m_topup as topup_mod

    async def _fake_load_winner():
        return None

    async def _fake_overlay():
        return {
            "stuck_replica_threshold": 7,
            "stuck_replica_streak": 11,
            "stuck_replica": "host-a/pid=42",
        }

    monkeypatch.setattr(topup_mod, "_load_last_winner", _fake_load_winner)
    monkeypatch.setattr(topup_mod, "_load_stuck_replica_overlay", _fake_overlay)
    topup_mod.state["stuck_replica_streak"] = 0
    topup_mod.state["stuck_replica"] = None

    with TestClient(app) as client:
        r = client.get("/ml/admin/5m-topup/status")
        assert r.status_code == 200
        body = r.json()
        assert body["stuck_replica_streak"] == 11
        assert body["stuck_replica"] == "host-a/pid=42"
        assert body["stuck_replica_threshold"] == 7


def test_status_endpoint_survives_overlay_failure(monkeypatch):
    """The overlay is best-effort — if the DB read raises, the endpoint
    must still return 200 with the local state values rather than
    crashing the dashboard."""
    import app.scheduled_5m_topup as topup_mod

    async def _fake_load_winner():
        return None

    async def _explode():
        raise RuntimeError("db down")

    monkeypatch.setattr(topup_mod, "_load_last_winner", _fake_load_winner)
    monkeypatch.setattr(topup_mod, "_load_stuck_replica_overlay", _explode)
    topup_mod.state["stuck_replica_streak"] = 0
    topup_mod.state["stuck_replica"] = None

    with TestClient(app) as client:
        r = client.get("/ml/admin/5m-topup/status")
        assert r.status_code == 200
        body = r.json()
        assert body["stuck_replica_streak"] == 0
        assert body["stuck_replica"] is None


def test_stuck_replica_threshold_is_env_configurable(monkeypatch):
    """Task spec: the threshold MUST be env-configurable. Re-import the
    module under a new env value and assert the constant + state mirror
    pick it up."""
    import importlib
    monkeypatch.setenv("ML_5M_TOPUP_STUCK_REPLICA_THRESHOLD", "11")
    import app.scheduled_5m_topup as topup_mod
    reloaded = importlib.reload(topup_mod)
    try:
        assert reloaded.STUCK_REPLICA_THRESHOLD == 11
        assert reloaded.state["stuck_replica_threshold"] == 11
    finally:
        # Restore so the rest of the suite sees the default — other
        # tests assert on the default state shape.
        monkeypatch.delenv("ML_5M_TOPUP_STUCK_REPLICA_THRESHOLD", raising=False)
        importlib.reload(reloaded)


# ── Task #603: counter persistence across ml-engine restarts ─────────────
class _FakeAppSettingsConn:
    """Minimal asyncpg-connection stub that backs `app_settings` with a
    plain Python dict. Returns rows shaped like a real fetchrow result
    (`{"value": <jsonb>}`) and accepts UPSERTs by writing the second
    placeholder param into the dict. Lets us pin the load + save paths
    of the counter helpers without touching Postgres.
    """

    def __init__(self, store: dict):
        self._store = store

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def fetchrow(self, sql, key):
        if key not in self._store:
            return None
        return {"value": self._store[key]}

    async def execute(self, sql, key, value):
        # asyncpg passes the JSON string as the second arg for our
        # `INSERT ... VALUES ($1, $2::jsonb, now()) ON CONFLICT ...`
        # statement; mirror that into the fake store so the next read
        # sees the new value.
        self._store[key] = value


class _FakeAppSettingsPool:
    """Pool stub returning the conn stub above from `acquire()`."""

    def __init__(self, store: dict):
        self._store = store

    def acquire(self):
        return _FakeAppSettingsConn(self._store)


def _install_fake_app_settings(monkeypatch, store: dict) -> None:
    """Replace `topup.db_mod.init_pool` with one returning a fake pool
    backed by `store`. Counter load/save flows through this without
    needing a real Postgres."""
    fake_pool = _FakeAppSettingsPool(store)

    async def _fake_init_pool():
        return fake_pool

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)


def test_load_counters_returns_zeros_when_no_row(monkeypatch):
    """Cold-start: app_settings has no `_COUNTERS_SETTING_KEY` row, so
    the loader must return a dict of zeros covering EVERY persisted
    key. Otherwise `_tick`'s mirror-into-state step would partial-update
    `state` and leak nondeterminism into the dashboard tiles."""
    import asyncio

    _install_fake_app_settings(monkeypatch, store={})
    out = asyncio.run(topup._load_counters_from_settings())
    assert set(out.keys()) == set(topup._PERSISTED_COUNTER_KEYS)
    assert all(v == 0 for v in out.values())


def test_load_counters_parses_jsonb_dict(monkeypatch):
    """asyncpg may decode jsonb as a Python dict already (codec
    enabled). Loader must accept that path and surface every persisted
    key, ignoring extras the row may carry from a future schema."""
    import asyncio

    store = {
        topup._COUNTERS_SETTING_KEY: {
            "ticks_total": 142,
            "runs_total": 7,
            "rows_inserted_total": 1989,
            "alerts_emitted_total": 3,
            "skips_locked_total": 12,
            "stuck_replica_alerts_total": 0,
            "future_extra_field": "ignored",
        }
    }
    _install_fake_app_settings(monkeypatch, store=store)

    out = asyncio.run(topup._load_counters_from_settings())
    assert out["ticks_total"] == 142
    assert out["runs_total"] == 7
    assert out["rows_inserted_total"] == 1989
    assert out["alerts_emitted_total"] == 3
    assert out["skips_locked_total"] == 12
    assert out["stuck_replica_alerts_total"] == 0
    assert "future_extra_field" not in out


def test_load_counters_parses_jsonb_string(monkeypatch):
    """Some asyncpg setups return jsonb as a JSON STRING (codec not
    registered). Loader must JSON-decode it transparently so callers
    don't have to care which path the codec took."""
    import asyncio
    import json as _json

    store = {
        topup._COUNTERS_SETTING_KEY: _json.dumps({
            "ticks_total": 99,
            "runs_total": 4,
            "rows_inserted_total": 500,
            "alerts_emitted_total": 1,
            "skips_locked_total": 6,
            "stuck_replica_alerts_total": 2,
        })
    }
    _install_fake_app_settings(monkeypatch, store=store)

    out = asyncio.run(topup._load_counters_from_settings())
    assert out["ticks_total"] == 99
    assert out["stuck_replica_alerts_total"] == 2


def test_save_counters_upserts_full_payload(monkeypatch):
    """Saver must UPSERT a JSON object covering EVERY persisted key,
    even ones that are zero. A subsequent load must round-trip the
    same values so `_tick`'s "save then re-hydrate on next process
    start" loop is non-lossy."""
    import asyncio
    import json as _json

    store: dict = {}
    _install_fake_app_settings(monkeypatch, store=store)

    asyncio.run(topup._save_counters_to_settings({
        "ticks_total": 5,
        "runs_total": 2,
        "rows_inserted_total": 123,
        "alerts_emitted_total": 1,
        "skips_locked_total": 0,
        "stuck_replica_alerts_total": 0,
        "ignored_extra": 999,
    }))

    raw = store[topup._COUNTERS_SETTING_KEY]
    parsed = _json.loads(raw) if isinstance(raw, str) else raw
    assert parsed["ticks_total"] == 5
    assert parsed["runs_total"] == 2
    assert parsed["rows_inserted_total"] == 123
    assert parsed["alerts_emitted_total"] == 1
    assert "ignored_extra" not in parsed
    # Every persisted key present (even the zeros) so a downstream
    # loader can rely on .get() returning a real number, not None.
    for k in topup._PERSISTED_COUNTER_KEYS:
        assert k in parsed


def test_save_counters_swallows_db_errors(monkeypatch):
    """Persistence is best-effort — a transient DB hiccup must NOT
    raise back into `_tick`'s finally clause where it would mask the
    actual outcome (or worse, prevent `_run_lock.release()`)."""
    import asyncio

    async def _fake_init_pool():
        raise RuntimeError("simulated db outage")

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)

    # Must not raise.
    asyncio.run(topup._save_counters_to_settings({
        k: 1 for k in topup._PERSISTED_COUNTER_KEYS
    }))


def test_tick_hydrates_counters_from_settings_on_first_call(monkeypatch):
    """First `_tick()` of the process must seed `state` from the
    persisted row BEFORE bumping `ticks_total`. The bump is then
    applied on top, so the visible counter equals
    `persisted + 1` after the first tick. Verifies the dashboard tiles
    survive an ml-engine restart."""
    import asyncio

    topup._reset_counter_state_for_tests()

    # Seed app_settings as if a previous process had already racked up
    # 100 ticks / 10 runs / 5000 inserted rows.
    store = {
        topup._COUNTERS_SETTING_KEY: {
            "ticks_total": 100,
            "runs_total": 10,
            "rows_inserted_total": 5000,
            "alerts_emitted_total": 2,
            "skips_locked_total": 50,
            "stuck_replica_alerts_total": 1,
        }
    }
    _install_fake_app_settings(monkeypatch, store=store)

    # Force the disabled short-circuit so `_tick` doesn't try to take
    # the advisory lock or call `topup_5m_head` — we only care about
    # the hydrate step here. The disabled outcome still bumps
    # `ticks_total` and runs the persist in `finally`.
    monkeypatch.setattr(topup, "ENABLED", False)

    out = topup._tick(now=1_000_000.0)
    assert out == "disabled"

    # Hydrated baseline + the +1 bump for THIS tick.
    assert topup.state["ticks_total"] == 101
    # Other counters mirrored straight from the persisted row.
    assert topup.state["runs_total"] == 10
    assert topup.state["rows_inserted_total"] == 5000
    assert topup.state["alerts_emitted_total"] == 2
    assert topup.state["skips_locked_total"] == 50
    assert topup.state["stuck_replica_alerts_total"] == 1
    # Hydration must flip the flag so the next tick doesn't re-hit DB.
    assert topup._counters_hydrated is True

    # Cleanup so subsequent tests in the suite see fresh state.
    topup._reset_counter_state_for_tests()


def test_tick_persists_counters_after_run(monkeypatch):
    """After `_tick` finishes (any outcome), the cumulative counter row
    in app_settings must reflect the new totals. This is the write half
    of the round-trip — without it a restart would forever read the
    same stale baseline."""
    topup._reset_counter_state_for_tests()

    store: dict = {}
    _install_fake_app_settings(monkeypatch, store=store)

    monkeypatch.setattr(topup, "ENABLED", False)  # disabled short-circuit
    topup._tick(now=2_000_000.0)

    # Persisted row must now exist and reflect the +1 bump on
    # ticks_total. Other counters stay at zero because the disabled
    # branch never reaches the OK path.
    raw = store[topup._COUNTERS_SETTING_KEY]
    import json as _json
    parsed = _json.loads(raw) if isinstance(raw, str) else raw
    assert parsed["ticks_total"] == 1
    assert parsed["runs_total"] == 0
    assert parsed["rows_inserted_total"] == 0

    topup._reset_counter_state_for_tests()


def test_topup_5m_head_uses_smart_fetcher_and_stamps_source(monkeypatch):
    """`topup_5m_head` must (1) call `fetch_5m_smart` per coin and (2)
    pass the returned source label into `insert_candles_batch` so the
    `price_candles.source` audit column reflects which upstream
    actually served each bar — never a silent fallback to "okx"."""
    import asyncio

    seen_inserts: list[dict] = []

    async def _fake_smart(client, coin_id, days, end_ts_ms=None):
        # Mapped coins return "coinbase"; JUP returns "okx".
        if coin_id == "jupiter-exchange-solana":
            return [("ts_jup",)], "okx"
        return [("ts_" + coin_id,)], "coinbase"

    async def _fake_insert(pool, coin, tf, candles, *, source, dry_run):
        seen_inserts.append({
            "coin": coin, "tf": tf, "source": source, "rows": len(candles),
        })
        return len(candles)

    async def _fake_init_pool():
        return object()  # `topup_5m_head` only forwards the pool

    monkeypatch.setattr(topup.db_mod, "init_pool", _fake_init_pool)

    # The function imports these lazily inside its body, so we need to
    # patch the module those imports resolve to.
    from scripts import backfill_history as bh
    monkeypatch.setattr(bh, "fetch_5m_smart", _fake_smart)
    monkeypatch.setattr(bh, "insert_candles_batch", _fake_insert)

    out = asyncio.run(topup.topup_5m_head(
        ["pepe", "jupiter-exchange-solana", "bonk"], window_days=7,
    ))

    assert out == {"pepe": 1, "jupiter-exchange-solana": 1, "bonk": 1}
    by_coin = {row["coin"]: row for row in seen_inserts}
    assert by_coin["pepe"]["source"] == "coinbase"
    assert by_coin["bonk"]["source"] == "coinbase"
    assert by_coin["jupiter-exchange-solana"]["source"] == "okx"
