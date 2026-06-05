"""Daily 5m head top-up — keep `price_candles` fresh so the 305-day
contiguous gate stays cleared without a campaign launch.

Background (task #410):
  Task #378 brought 9 monitored coins to ~310 contiguous days of 5m data,
  just barely above the 305-day hard gate in `_evaluate_5m_gate`. Because
  OKX `/api/v5/market/history-candles` is a fixed window relative to
  "now", the head of `price_candles` does NOT advance on its own — the
  live tick poller writes 1m close-only ticks into `price_history`, not
  5m OHLCV bars. Today the only paths that extend the head are (a) the
  orchestrator's per-iteration backfill loop in `phase2_data_audit`,
  which only runs when an operator launches a campaign, and (b) the
  ad-hoc driver in `scripts/backfill_5m_extend.py`. Neither runs on a
  schedule, so contiguous_days erodes by ~1 real day every day until the
  gate flips back to "skipped" mid-week.

Contract:
  * Once per `INTERVAL_SECONDS` (24h default), fetch the most recent
    `WINDOW_DAYS` (7 default) of 5m OHLCV from OKX for every monitored
    coin and idempotently insert into `price_candles`.
  * After the top-up, measure each coin's longest contiguous 5m run
    over the last year and emit a structured warning + a JSONL progress
    entry for any coin under `ALERT_BELOW_DAYS` (310 default — one day
    of headroom over the 305-day gate so we shout BEFORE the gate trips).
  * Real-data only. No synthetic candles, ever. Reuses the writer from
    `scripts.backfill_history` so the cadence guard
    (`assert_native_cadence`) and the OKX-source rule both apply at the
    same boundary every other path uses.

Scheduler wiring:
  Started from `app.main:lifespan` via `_start_5m_topup_scheduler`,
  alongside the existing auto-retrain and fast-loop daemons. Honours the
  same `ML_AUTO_RETRAIN_TEST_DISABLE` opt-out so the FastAPI test client
  never actually hits OKX.

Operational notes:
  * Multi-replica safety (task #413): each `_tick` first tries to take a
    Postgres advisory lock keyed on a constant label. Only the process
    that wins the lock actually fetches from OKX; losers report
    `skipped_locked_other_process` and wait for the next tick. The lock
    is session-scoped, so a crashed worker can never leave it stuck.
    Setting `ML_5M_TOPUP_ENABLED=0` is still supported as a hard opt-out.
  * Current state surfaced at GET `/ml/admin/5m-topup/status` for the
    dashboard.
  * Cross-replica winner attribution (task #424): the process that wins
    the advisory lock for a tick records its identity (nodename + pid)
    plus the tick timestamp into a single `app_settings` row keyed on
    `_LAST_WINNER_SETTING_KEY`. The status endpoint reads that row on
    every request and merges it into the response, so an operator
    inspecting any replica can see which box did the most recent pull
    even if the replica they're looking at is itself a `skipped_locked`
    loser. Best-effort: write/read failures fall back to the in-memory
    state and never break the tick.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db as db_mod  # noqa: E402
from app.contiguity import (  # noqa: E402
    CONTIGUITY_TOLERANCE_SECONDS,
    compute_longest_contiguous_run,
)

logger = logging.getLogger("ml-engine.5m-topup")

# ── Config (env-overridable) ─────────────────────────────────────────────
ENABLED = os.environ.get("ML_5M_TOPUP_ENABLED", "1") == "1"
INTERVAL_SECONDS = int(
    os.environ.get("ML_5M_TOPUP_INTERVAL_SECONDS", str(24 * 60 * 60))
)  # daily by default
WINDOW_DAYS = int(os.environ.get("ML_5M_TOPUP_WINDOW_DAYS", "7"))
ALERT_BELOW_DAYS = float(
    os.environ.get("ML_5M_TOPUP_ALERT_BELOW_DAYS", "310")
)
# Task #442 — fire a structured warning + progress entry when the same
# replica wins the daily pull this many ticks in a row. The api-server
# notifier surface (`topup-5m-notifier.ts`) polls the status endpoint
# and pages the operator off-dashboard so they don't have to be
# watching the dashboard to notice "host-a has owned the lock for two
# weeks straight". Default 7 — a full week of daily ticks is the
# smallest window where a stuck replica is meaningful (one bad day is
# not enough signal). Threshold is env-overridable so an operator can
# tune the sensitivity without a redeploy.
STUCK_REPLICA_THRESHOLD = int(
    os.environ.get("ML_5M_TOPUP_STUCK_REPLICA_THRESHOLD", "7")
)
STARTUP_DELAY_SECONDS = int(
    os.environ.get("ML_5M_TOPUP_STARTUP_DELAY_SECONDS", "30")
)
PROGRESS_PATH = ROOT / "models" / "progress_updates.jsonl"

# 5m bar duration in seconds — used for contiguous-run measurement.
_TF_SECONDS = 300

# Postgres advisory-lock label for the daily top-up. Stable across every
# ml-engine replica that imports this module so they all contend on the
# same lock and only one wins per tick. See `db.advisory_lock_key`.
_TOPUP_LOCK_LABEL = "ml_engine.scheduled_5m_topup"
_TOPUP_LOCK_KEY = db_mod.advisory_lock_key(_TOPUP_LOCK_LABEL)

# `app_settings` row that records the most recent winning replica + tick
# timestamp. One row, upserted on every successful tick; the status
# endpoint reads this back so an operator inspecting any replica sees
# the actual cross-fleet winner — not just whatever the local process's
# memory remembers. Task #424.
_LAST_WINNER_SETTING_KEY = "ml_engine.scheduled_5m_topup.last_winner"

# `app_settings` row that records the LAST N winning replicas + tick
# timestamps as a JSON array (newest-first). Task #435: the single
# `last_winner` row above answers "did SOMEONE pull today?" but not "is
# the lock evenly distributed across boxes?" — operators need the recent
# history to spot a stuck replica (e.g. host-a winning 14 days in a
# row). The list is appended to under the same advisory lock the tick
# already holds, so two replicas can never race to clobber each other's
# write. Capped at `_RECENT_WINNERS_MAX` so the row never grows
# unboundedly.
_RECENT_WINNERS_SETTING_KEY = "ml_engine.scheduled_5m_topup.recent_winners"
_RECENT_WINNERS_MAX = 50

# `app_settings` row that records the cumulative scheduler-tick counters
# (Task #603). Without persistence the dashboard's "Total runs / rows
# inserted / alerts emitted" tiles reset to zero on every ml-engine
# restart, making them functionally useless for "is the daily pull
# actually running?". The row holds a single JSON object whose keys are
# the entries in `_PERSISTED_COUNTER_KEYS`. Hydrated into `state` on
# the first tick and re-persisted at the end of every tick so the
# dashboard reflects fleet-wide totals across restarts and across
# replicas (the next replica to win the lock simply reads the latest
# value before bumping). Best-effort: DB hiccups log + fall through to
# the in-memory `state` so the tick can never die here.
_COUNTERS_SETTING_KEY = "ml_engine.scheduled_5m_topup.counters"
_PERSISTED_COUNTER_KEYS: tuple[str, ...] = (
    "ticks_total",
    "runs_total",
    "rows_inserted_total",
    "alerts_emitted_total",
    "skips_locked_total",
    "stuck_replica_alerts_total",
)
# Default number of recent winners the dashboard shows. Two weeks of
# daily ticks is enough to eyeball "all three replicas are taking turns"
# vs "host-a never lost" without flooding the card.
_RECENT_WINNERS_DEFAULT_LIMIT = 14


def _replica_identity() -> str:
    """Return a stable, human-readable identity for THIS process.

    Format: ``"<nodename>/pid=<pid>"``. The nodename is what the
    container/pod sees as its hostname, which is enough to tell
    "replica A vs replica B" apart in a multi-replica deployment;
    appending the pid disambiguates if a replica restarts within the
    same hostname between ticks.

    `os.uname()` is unavailable on Windows; the `getattr` fallback
    keeps the test suite green on the rare developer laptop that
    doesn't ship a POSIX `uname`.
    """
    uname = getattr(os, "uname", None)
    if uname is not None:
        try:
            nodename = uname().nodename or "unknown"
        except Exception:  # noqa: BLE001 - identity is best-effort
            nodename = "unknown"
    else:
        nodename = os.environ.get("HOSTNAME") or "unknown"
    return f"{nodename}/pid={os.getpid()}"

# ── State (read by /ml/admin/5m-topup/status) ────────────────────────────
state: dict = {
    "enabled": ENABLED,
    "interval_seconds": INTERVAL_SECONDS,
    "window_days": WINDOW_DAYS,
    "alert_below_days": ALERT_BELOW_DAYS,
    "last_check_at": None,           # epoch seconds when the last tick fired
    "last_attempt_outcome": None,    # disabled | skipped_busy | skipped_locked_other_process | ok | error
    "last_finished_at": None,
    "last_error": None,
    "last_topup_inserted": 0,        # rows actually written across all coins
    "last_topup_per_coin": {},       # {coin: rows_inserted}
    "last_health_per_coin": {},      # {coin: contiguous_days}
    "last_alerts": [],               # coins under ALERT_BELOW_DAYS at last tick
    "ticks_total": 0,
    "runs_total": 0,
    "rows_inserted_total": 0,
    "alerts_emitted_total": 0,
    # Multi-replica advisory-lock telemetry (task #413). Counts ticks
    # where another ml-engine process held the Postgres advisory lock so
    # we can confirm in production that only one replica is actually
    # firing OKX per tick.
    "skips_locked_total": 0,
    "advisory_lock_label": _TOPUP_LOCK_LABEL,
    "advisory_lock_key": _TOPUP_LOCK_KEY,
    # Cross-replica winner attribution (task #424). Identity of the
    # ml-engine process that actually completed the most recent
    # successful tick, plus the wall-clock time when that tick fired.
    # The status endpoint refreshes these from the shared `app_settings`
    # row on every read so any replica's dashboard view reflects the
    # actual fleet-wide winner.
    "last_winning_replica": None,
    "last_winning_at": None,
    # Stuck-replica detection (task #442). The current head replica's
    # consecutive-win streak (computed from the recent_winners row), the
    # replica name when the streak crosses `STUCK_REPLICA_THRESHOLD`
    # (else None), and a counter of how many times the ml-engine has
    # actually emitted the alert. The status endpoint overlays
    # `stuck_replica_streak` / `stuck_replica` from the shared row on
    # every read so any replica's dashboard view reflects the
    # fleet-wide streak — not just whatever this process did locally.
    "stuck_replica_threshold": STUCK_REPLICA_THRESHOLD,
    "stuck_replica_streak": 0,
    "stuck_replica": None,
    "stuck_replica_alerts_total": 0,
}

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_run_lock = threading.Lock()


def _coins() -> list[str]:
    """Resolve the canonical coin list lazily so a test can monkeypatch
    `app.training.train.DEFAULT_COINS` between import and first tick.

    Bitcoin is appended outside DEFAULT_COINS because it is consumed by
    the diagnostic-sandbox lane (BTC/5m fixed 0.5%) but is NOT a member
    of the trainer universe — adding it to DEFAULT_COINS would feed the
    training campaign without the contiguity/density audit having been
    done for BTC. Keeping it as a top-up-only target preserves the
    sandbox's fresh-data invariant while leaving the trainer untouched.
    """
    from app.training.train import DEFAULT_COINS
    coins = list(DEFAULT_COINS)
    if "bitcoin" not in coins:
        coins.append("bitcoin")
    return coins


# ── Top-up: fetch last N days of 5m and insert ───────────────────────────
async def topup_5m_head(
    coins: list[str], *, window_days: int = WINDOW_DAYS,
) -> dict[str, int]:
    """For each coin, pull the last `window_days` of 5m OHLCV from the
    deepest-history source available (Coinbase Exchange where listed,
    OKX otherwise) and idempotently insert into `price_candles`. Returns
    ``{coin: rows_inserted}``. ON CONFLICT DO NOTHING handles overlap
    with rows the previous tick already wrote.

    Per-coin source label is stamped on the inserted batch so an
    operator can audit which upstream actually served each bar — the
    canonical writer respects that label and never silently relabels.
    Task #603 — switched the head-pull from OKX-only to Coinbase-first
    because OKX's 5m history endpoint truncates at ~60-100d (well below
    the 305-day gate); Coinbase serves 5m back ≥335d for every
    monitored coin except JUP, which keeps OKX with its shorter window.
    """
    # Reuse the canonical writer + smart fetcher so the cadence guard
    # and the source-stamping rule both apply here exactly as they do
    # for the operator-driven backfill.
    from scripts.backfill_history import (  # noqa: WPS433
        fetch_5m_smart,
        insert_candles_batch,
    )

    pool = await db_mod.init_pool()
    inserted: dict[str, int] = {}
    async with httpx.AsyncClient() as client:
        for coin in coins:
            try:
                candles, source_label = await fetch_5m_smart(
                    client, coin, window_days,
                )
                n = await insert_candles_batch(
                    pool, coin, "5m", candles,
                    source=source_label, dry_run=False,
                )
            except Exception as exc:  # noqa: BLE001 - per-coin isolation
                logger.warning(
                    "topup_5m_coin_failed",
                    extra={"coin": coin, "error": str(exc)},
                )
                inserted[coin] = 0
                continue
            inserted[coin] = int(n)
    return inserted


# ── Health: longest contiguous 5m run per coin (last year window) ────────
async def measure_contiguous_5m(coins: list[str]) -> dict[str, float]:
    """Return {coin: contiguous_days} computed via
    `app.contiguity.compute_longest_contiguous_run` — the SAME helper
    the campaign's gate (`_coverage_per_slice`) uses, so the topup's
    health alert and the gate's verdict can never disagree about the
    same number.

    Task #604 — the returned value is the GAP-TOLERANT measure
    (5m tolerance = 7h / 84 missing buckets via
    `CONTIGUITY_TOLERANCE_SECONDS["5m"]`). The strict pre-tolerance
    metric is intentionally not surfaced here because the alert's
    threshold is meant to mirror the gate's threshold; if an operator
    needs the strict number they should run the campaign's coverage
    audit which carries both fields side-by-side.
    """
    pool = await db_mod.init_pool()
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    tolerance_sec = CONTIGUITY_TOLERANCE_SECONDS.get("5m", 0)
    out: dict[str, float] = {}
    for coin in coins:
        async with pool.acquire() as con:
            rows = await con.fetch(
                "SELECT bucket_start FROM price_candles "
                "WHERE coin_id=$1 AND timeframe='5m' AND bucket_start >= $2 "
                "ORDER BY bucket_start",
                coin, one_year_ago,
            )
        tol_days, _strict_days = compute_longest_contiguous_run(
            [r["bucket_start"] for r in rows], _TF_SECONDS, tolerance_sec,
        )
        out[coin] = round(tol_days, 2)
    return out


# ── Alerting / progress logging ──────────────────────────────────────────
def _append_progress(record: dict) -> None:
    """Append a structured entry to `models/progress_updates.jsonl` so the
    same surface the campaign uses for live updates also carries the
    daily top-up health. Never raises.
    """
    try:
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_PATH.open("a") as fh:
            import json
            payload = {"emitted_at": datetime.now(timezone.utc).isoformat(), **record}
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 - progress log is best-effort
        logger.warning("topup_5m_progress_append_failed", extra={"error": str(exc)})


def _emit_alerts(
    health: dict[str, float], *, threshold: float,
) -> list[str]:
    """Log + persist a structured alert for every coin whose 5m
    contiguous_days fell below `threshold`. Returns the alert list so
    callers (and tests) can assert on it.
    """
    below = sorted(
        [c for c, d in health.items() if d < threshold]
    )
    if below:
        logger.warning(
            "topup_5m_below_threshold",
            extra={
                "threshold_days": threshold,
                "coins": below,
                "per_coin_contiguous_days": {c: health[c] for c in below},
            },
        )
        _append_progress({
            "phase": "5m_topup_health",
            "status": "alert",
            "headline": (
                f"5m contiguous_days below {threshold:.0f} for "
                f"{len(below)} coin(s)"
            ),
            "threshold_days": threshold,
            "coins": below,
            "per_coin_contiguous_days": {c: health[c] for c in below},
        })
    else:
        logger.info(
            "topup_5m_health_ok",
            extra={
                "threshold_days": threshold,
                "min_contiguous_days": min(health.values()) if health else None,
            },
        )
    return below


# ── Cross-replica winner attribution (task #424) ─────────────────────────
def _parse_recent_winners(raw) -> list[dict]:
    """Normalise the raw value pulled from the recent-winners
    ``app_settings`` row into a list of ``{"replica", "tick_at"}``
    dicts. asyncpg may hand us either a parsed list (jsonb codec) or
    a JSON string; both shapes are accepted. Anything malformed (None,
    not a list, entries missing fields) is silently dropped — a corrupt
    row must never crash the tick or the status endpoint.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        replica = entry.get("replica")
        tick_at = entry.get("tick_at")
        if isinstance(replica, str) and isinstance(tick_at, str):
            out.append({"replica": replica, "tick_at": tick_at})
    return out


async def _record_winning_replica(
    *, replica: str, tick_at: datetime,
) -> list[dict]:
    """Persist the identity of the replica that just completed a tick
    into the shared ``app_settings`` rows. Called only after a successful
    OKX pull (the tick that won the advisory lock and finished cleanly).

    Two writes happen here, both inside one transaction so a partial
    failure can't leave the rows out of sync:

      1. ``_LAST_WINNER_SETTING_KEY`` — single-row UPSERT carrying the
         most recent winner. Task #424; preserved for any consumer that
         only wants the latest pull.
      2. ``_RECENT_WINNERS_SETTING_KEY`` — JSON array of the last
         ``_RECENT_WINNERS_MAX`` winners (newest-first). Task #435; lets
         operators see the recent history of who pulled so a stuck
         replica (one box winning N days in a row) is visible without
         scraping logs.

    The append is read-modify-write, but we only run inside the held
    advisory lock so two replicas can never race here. Best-effort: a
    DB hiccup must not break the tick, so failures log a warning and
    otherwise return silently.

    Returns the resulting recent-winners list (newest-first, capped at
    ``_RECENT_WINNERS_MAX``) so the caller can compute the head
    replica's streak without an extra DB round-trip. Returns an empty
    list on any DB error — the streak detector treats that the same
    as "not enough data to alert".
    """
    last_payload = json.dumps({
        "replica": replica,
        "tick_at": tick_at.isoformat(),
    })
    try:
        pool = await db_mod.init_pool()
        async with pool.acquire() as con:
            async with con.transaction():
                await con.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO UPDATE
                      SET value = EXCLUDED.value, updated_at = now()
                    """,
                    _LAST_WINNER_SETTING_KEY, last_payload,
                )
                row = await con.fetchrow(
                    "SELECT value FROM app_settings WHERE key = $1",
                    _RECENT_WINNERS_SETTING_KEY,
                )
                current = _parse_recent_winners(
                    row["value"] if row is not None else None
                )
                current.insert(0, {
                    "replica": replica,
                    "tick_at": tick_at.isoformat(),
                })
                if len(current) > _RECENT_WINNERS_MAX:
                    current = current[:_RECENT_WINNERS_MAX]
                await con.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO UPDATE
                      SET value = EXCLUDED.value, updated_at = now()
                    """,
                    _RECENT_WINNERS_SETTING_KEY, json.dumps(current),
                )
        return current
    except Exception as exc:  # noqa: BLE001 - winner write is best-effort
        logger.warning(
            "topup_5m_winner_record_failed",
            extra={"error": str(exc), "replica": replica},
        )
        return []


# ── Stuck-replica detection (task #442) ──────────────────────────────────
def _compute_winner_streak(
    recent_winners: list[dict],
) -> tuple[Optional[str], int]:
    """Return ``(replica, consecutive_count)`` for the head replica's
    current run. ``recent_winners`` is the newest-first list written by
    ``_record_winning_replica``: the streak is simply the count of
    leading entries whose ``replica`` equals the head entry's. Empty or
    malformed input yields ``(None, 0)`` so the caller never has to
    guard against the no-data case.

    Examples:
      ``[A, A, A, B, A]`` → ``("A", 3)`` (the early A's after the B
      don't extend the streak — only the head's contiguous prefix
      counts).
      ``[A]`` → ``("A", 1)``.
      ``[]`` → ``(None, 0)``.
    """
    if not recent_winners:
        return None, 0
    head_entry = recent_winners[0]
    if not isinstance(head_entry, dict):
        return None, 0
    head = head_entry.get("replica")
    if not isinstance(head, str):
        return None, 0
    streak = 0
    for entry in recent_winners:
        if not isinstance(entry, dict):
            break
        if entry.get("replica") == head:
            streak += 1
        else:
            break
    return head, streak


def _emit_stuck_replica_alert(
    *, replica: str, streak: int, threshold: int,
) -> None:
    """Log a structured warning + append a progress journal entry when
    a single replica's consecutive-win streak crosses ``threshold``.

    Mirrors the shape of ``_emit_alerts`` so the api-server notifier
    surface (`topup-5m-notifier.ts`) and the campaign progress reader
    can treat both kinds of 5m-topup alerts uniformly. Best-effort:
    progress write failures are swallowed by ``_append_progress`` so
    the tick itself can never die here.
    """
    logger.warning(
        "topup_5m_stuck_replica",
        extra={
            "replica": replica,
            "consecutive_wins": streak,
            "threshold": threshold,
        },
    )
    _append_progress({
        "phase": "5m_topup_stuck_replica",
        "status": "alert",
        "headline": (
            f"Replica {replica} has won the daily 5m top-up "
            f"{streak} ticks in a row (threshold: {threshold})"
        ),
        "replica": replica,
        "consecutive_wins": streak,
        "threshold": threshold,
    })


async def _load_stuck_replica_overlay() -> dict:
    """Compute the current head-replica streak by reading the recent
    winners row from ``app_settings``. The status endpoint calls this
    on every read so any replica's view of ``stuck_replica`` and
    ``stuck_replica_streak`` reflects the actual fleet-wide state —
    a replica that has only ever lost the lock would otherwise report
    ``stuck_replica_streak=0`` even when some other host has been
    winning every day for a fortnight.

    Returns a dict with the three keys the caller will overlay onto
    the status payload. ``stuck_replica`` is the head replica name
    iff its streak is at-or-above ``STUCK_REPLICA_THRESHOLD`` — below
    that we leave it None so the dashboard / notifier don't have to
    re-implement the threshold check.
    """
    winners = await _load_recent_winners(limit=_RECENT_WINNERS_MAX)
    head, streak = _compute_winner_streak(winners)
    return {
        "stuck_replica_threshold": STUCK_REPLICA_THRESHOLD,
        "stuck_replica_streak": streak,
        "stuck_replica": (
            head if (head is not None and streak >= STUCK_REPLICA_THRESHOLD)
            else None
        ),
    }


async def _load_last_winner() -> Optional[dict]:
    """Read the most recent winner record from ``app_settings``.

    Returns ``{"replica": str, "tick_at": str}`` or ``None`` if no tick
    has ever recorded a winner yet (or on any DB error). Used by the
    status endpoint so every replica's dashboard view shows the actual
    fleet-wide winner, not just whatever this process did locally.
    """
    try:
        pool = await db_mod.init_pool()
        async with pool.acquire() as con:
            row = await con.fetchrow(
                "SELECT value FROM app_settings WHERE key = $1",
                _LAST_WINNER_SETTING_KEY,
            )
    except Exception as exc:  # noqa: BLE001 - read is best-effort
        logger.warning(
            "topup_5m_winner_read_failed", extra={"error": str(exc)},
        )
        return None
    if row is None:
        return None
    raw = row["value"]
    # asyncpg may return jsonb as a parsed dict or as a JSON string
    # depending on codec setup — normalise.
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(raw, dict):
        return None
    replica = raw.get("replica")
    tick_at = raw.get("tick_at")
    if not isinstance(replica, str) or not isinstance(tick_at, str):
        return None
    return {"replica": replica, "tick_at": tick_at}


async def _load_recent_winners(
    *, limit: int = _RECENT_WINNERS_DEFAULT_LIMIT,
) -> list[dict]:
    """Read up to ``limit`` recent winner records from
    ``app_settings``. The list is stored newest-first, so we just slice
    the head. Task #435: lets the dashboard render the last ~14 winners
    so an operator can spot a stuck replica (one box winning every day)
    without scraping logs.

    Returns ``[]`` when no tick has ever recorded a winner yet, when
    the row is malformed, or on any DB error — the read must never
    block the rest of the status payload.
    """
    if limit <= 0:
        return []
    try:
        pool = await db_mod.init_pool()
        async with pool.acquire() as con:
            row = await con.fetchrow(
                "SELECT value FROM app_settings WHERE key = $1",
                _RECENT_WINNERS_SETTING_KEY,
            )
    except Exception as exc:  # noqa: BLE001 - read is best-effort
        logger.warning(
            "topup_5m_recent_winners_read_failed",
            extra={"error": str(exc)},
        )
        return []
    if row is None:
        return []
    return _parse_recent_winners(row["value"])[:limit]


# ── Counter persistence (Task #603) ──────────────────────────────────────
async def _load_counters_from_settings() -> dict[str, int]:
    """Read the persisted scheduler-tick counters from ``app_settings``.

    Returns a dict with EVERY key in ``_PERSISTED_COUNTER_KEYS`` present
    (zero by default) so callers can mirror straight into ``state``
    without partial-update worries. Never raises — DB hiccup logs and
    falls back to zeros so the tick can still run cleanly. asyncpg may
    decode the jsonb column either as a parsed dict or a JSON string;
    we normalise both.
    """
    out: dict[str, int] = {k: 0 for k in _PERSISTED_COUNTER_KEYS}
    try:
        pool = await db_mod.init_pool()
        async with pool.acquire() as con:
            row = await con.fetchrow(
                "SELECT value FROM app_settings WHERE key = $1",
                _COUNTERS_SETTING_KEY,
            )
    except Exception as exc:  # noqa: BLE001 - read is best-effort
        logger.warning(
            "topup_5m_counters_read_failed", extra={"error": str(exc)},
        )
        return out
    if row is None:
        return out
    raw = row["value"]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return out
    if not isinstance(raw, dict):
        return out
    for k in _PERSISTED_COUNTER_KEYS:
        v = raw.get(k)
        if isinstance(v, (int, float)):
            out[k] = int(v)
    return out


async def _save_counters_to_settings(counters: dict) -> None:
    """UPSERT the current scheduler-tick counters into ``app_settings``.

    Best-effort: write failures log but never raise so the tick can
    never die on a transient DB hiccup. Only the keys in
    ``_PERSISTED_COUNTER_KEYS`` are written; any extras passed in are
    silently dropped so a future caller can't accidentally bloat the row.
    """
    payload = {
        k: int(counters.get(k) or 0) for k in _PERSISTED_COUNTER_KEYS
    }
    try:
        pool = await db_mod.init_pool()
        async with pool.acquire() as con:
            await con.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES ($1, $2::jsonb, now())
                ON CONFLICT (key) DO UPDATE
                  SET value = EXCLUDED.value, updated_at = now()
                """,
                _COUNTERS_SETTING_KEY, json.dumps(payload),
            )
    except Exception as exc:  # noqa: BLE001 - write is best-effort
        logger.warning(
            "topup_5m_counters_write_failed", extra={"error": str(exc)},
        )


def _load_counters_blocking() -> dict[str, int]:
    """Drive `_load_counters_from_settings` on a fresh asyncio loop.

    Mirrors the same pool save/restore dance as `_run_once_blocking` so
    the FastAPI lifespan pool is never bound to the wrong loop. Used
    from the SYNC `_tick()` on the very first iteration to seed the
    in-memory counters from the persisted row.
    """
    saved_pool = db_mod._pool
    saved_pool_loop = db_mod._pool_loop
    db_mod._pool = None
    db_mod._pool_loop = None
    try:
        return asyncio.run(_load_counters_from_settings())
    finally:
        db_mod._pool = saved_pool
        db_mod._pool_loop = saved_pool_loop


def _save_counters_blocking(counters: dict) -> None:
    """Sync wrapper around `_save_counters_to_settings`. Same pool
    save/restore dance as `_load_counters_blocking`."""
    saved_pool = db_mod._pool
    saved_pool_loop = db_mod._pool_loop
    db_mod._pool = None
    db_mod._pool_loop = None
    try:
        asyncio.run(_save_counters_to_settings(counters))
    finally:
        db_mod._pool = saved_pool
        db_mod._pool_loop = saved_pool_loop


# Module-level flag so the very first `_tick()` call hydrates the
# persisted counters into `state` BEFORE bumping `ticks_total`. After
# the first hydration this flips to True and stays true for the
# lifetime of the process. Reset between tests via `_reset_counter_state_for_tests`.
_counters_hydrated: bool = False


def _reset_counter_state_for_tests() -> None:
    """Test-only hook: reset the hydration flag + zero all persisted
    counters in `state`. Production callers MUST NOT use this; it
    exists purely so the unit tests can drive `_tick()` from a known
    starting condition without depending on test-execution order.
    """
    global _counters_hydrated
    _counters_hydrated = False
    for k in _PERSISTED_COUNTER_KEYS:
        state[k] = 0


# ── Scheduler tick ───────────────────────────────────────────────────────
async def _run_once_async() -> dict:
    """Acquire the cross-replica advisory lock and, if won, do one
    top-up + health pass. When another ml-engine process holds the lock
    this returns ``{"skipped_locked": True, ...}`` immediately and never
    issues an OKX request.
    """
    async with db_mod.try_advisory_lock(_TOPUP_LOCK_KEY) as got:
        if not got:
            return {
                "skipped_locked": True,
                "coins": [],
                "inserted_per_coin": {},
                "contiguous_days_per_coin": {},
                "alerts": [],
                "winning_replica": None,
                "winning_at": None,
                "stuck_replica_streak": 0,
                "stuck_replica": None,
                "stuck_replica_threshold": STUCK_REPLICA_THRESHOLD,
                "stuck_replica_alert_emitted": False,
            }
        coins = _coins()
        inserted = await topup_5m_head(coins, window_days=WINDOW_DAYS)
        health = await measure_contiguous_5m(coins)
        alerts = _emit_alerts(health, threshold=ALERT_BELOW_DAYS)
        # Record cross-replica winner attribution (task #424). We do this
        # while still holding the advisory lock so two replicas can never
        # race to clobber each other's row inside the same tick window.
        # Writing AFTER the OKX pull finishes guarantees we only attribute
        # winners that actually completed the work.
        replica = _replica_identity()
        winning_at = datetime.now(timezone.utc)
        recent_winners_after = await _record_winning_replica(
            replica=replica, tick_at=winning_at,
        )
        # Stuck-replica detection (task #442). The recent_winners list we
        # just wrote already includes THIS tick at the head; compute the
        # head's streak from that list (no extra DB round-trip) and emit
        # a structured warning + progress entry when it crosses the
        # configured threshold. The api-server notifier polls the status
        # endpoint and pages the operator off-dashboard so they don't
        # have to be watching the dashboard to notice the imbalance.
        recent_list = recent_winners_after if recent_winners_after else []
        head_replica, streak = _compute_winner_streak(recent_list)
        stuck_replica: Optional[str] = None
        alert_emitted = False
        if (
            head_replica is not None
            and streak >= STUCK_REPLICA_THRESHOLD
        ):
            stuck_replica = head_replica
            _emit_stuck_replica_alert(
                replica=head_replica,
                streak=streak,
                threshold=STUCK_REPLICA_THRESHOLD,
            )
            alert_emitted = True
        return {
            "skipped_locked": False,
            "coins": coins,
            "inserted_per_coin": inserted,
            "contiguous_days_per_coin": health,
            "alerts": alerts,
            "winning_replica": replica,
            "winning_at": winning_at.isoformat(),
            "stuck_replica_streak": streak,
            "stuck_replica": stuck_replica,
            "stuck_replica_threshold": STUCK_REPLICA_THRESHOLD,
            "stuck_replica_alert_emitted": alert_emitted,
        }


def _run_once_blocking() -> dict:
    """Drive `_run_once_async` on a fresh asyncio loop in this thread.

    Mirrors the pattern in `_run_retrain_blocking` /
    `_run_fast_loop_blocking` so the FastAPI lifespan pool is never bound
    to the wrong loop. Crucially we save AND restore BOTH `_pool` and
    `_pool_loop` — `_run_once_async` runs on a fresh asyncio loop that
    `init_pool` will use to build a fresh pool and stamp `_pool_loop`
    with that fresh loop. If we only restored `_pool`, the next
    lifespan-loop caller would find `_pool` set but `_pool_loop` pointing
    at our now-closed scheduler loop and pay the penalty of a forced
    pool rebuild on its very next call. Restoring both keeps the
    lifespan pool's loop binding honest.
    """
    saved_pool = db_mod._pool
    saved_pool_loop = db_mod._pool_loop
    db_mod._pool = None
    db_mod._pool_loop = None
    try:
        return asyncio.run(_run_once_async())
    finally:
        db_mod._pool = saved_pool
        db_mod._pool_loop = saved_pool_loop


def _tick(now: Optional[float] = None) -> str:
    """One scheduler iteration. Returns the outcome string also written
    into `state['last_attempt_outcome']` so tests can drive the tick
    without spinning up the polling thread.
    """
    global _counters_hydrated
    now = now if now is not None else time.time()
    # Task #603 — on the very first tick of the process, hydrate the
    # persisted scheduler-tick counters from `app_settings` BEFORE we
    # bump `ticks_total` so the dashboard tiles show fleet-wide totals
    # across ml-engine restarts (and across replicas — the next replica
    # to enter `_tick` reads whatever the previous winner persisted).
    # Best-effort: a DB hiccup leaves `state` at its zero defaults and
    # the persistence path retries on the next tick.
    if not _counters_hydrated:
        try:
            loaded = _load_counters_blocking()
            for k, v in loaded.items():
                state[k] = int(v)
        except Exception as exc:  # noqa: BLE001 - hydrate is best-effort
            logger.warning(
                "topup_5m_counters_hydrate_failed",
                extra={"error": str(exc)},
            )
        # Flip even on failure so we don't pay the failed RTT every tick.
        _counters_hydrated = True
    state["ticks_total"] = int(state.get("ticks_total") or 0) + 1
    state["last_check_at"] = now

    # Task #603 — wrap the whole post-hydrate body in a try/finally so
    # the counter-persistence step ALWAYS runs, even on the
    # `disabled` / `skipped_busy` short-circuits. Otherwise the
    # `ticks_total` bump above would never reach app_settings on a
    # disabled box and the dashboard would forever show zero ticks for
    # that replica even though the scheduler IS being polled.
    try:
        if not ENABLED:
            state["last_attempt_outcome"] = "disabled"
            return "disabled"

        if not _run_lock.acquire(blocking=False):
            # Previous run still in flight — extremely unlikely on a 24h
            # cadence, but cheap to guard.
            state["last_attempt_outcome"] = "skipped_busy"
            return "skipped_busy"

        try:
            result = _run_once_blocking()
            if result.get("skipped_locked"):
                # Another ml-engine replica won the advisory lock for
                # this tick — it is doing the OKX pull, so we
                # explicitly do nothing. The next tick will try again.
                # No state mutation for `last_topup_*` / `last_health_*`
                # so the dashboard keeps showing whatever the most
                # recent winner produced.
                state["last_attempt_outcome"] = "skipped_locked_other_process"
                state["skips_locked_total"] = (
                    int(state.get("skips_locked_total") or 0) + 1
                )
                logger.info(
                    "topup_5m_skipped_locked_other_process",
                    extra={"advisory_lock_key": _TOPUP_LOCK_KEY},
                )
                return "skipped_locked_other_process"
            inserted_total = sum(result["inserted_per_coin"].values())
            state["last_topup_per_coin"] = result["inserted_per_coin"]
            state["last_topup_inserted"] = inserted_total
            state["last_health_per_coin"] = result["contiguous_days_per_coin"]
            state["last_alerts"] = result["alerts"]
            state["last_attempt_outcome"] = "ok"
            state["last_error"] = None
            # Local mirror of the cross-replica winner record (task
            # #424). The status endpoint also re-reads from the shared
            # `app_settings` row so any replica's view is fleet-accurate,
            # but stamping the state dict here means a single-replica
            # deployment doesn't need the DB round-trip to render the
            # field.
            state["last_winning_replica"] = result.get("winning_replica")
            state["last_winning_at"] = result.get("winning_at")
            # Stuck-replica detection (task #442). The successful tick
            # has the freshest view of the streak — record it locally
            # for the single-replica fast path. The status endpoint
            # also overlays `_load_stuck_replica_overlay()` on top so a
            # replica that has only ever lost the lock still surfaces
            # the fleet-wide value. `stuck_replica_alerts_total` is
            # bumped only when this tick actually emitted the alert —
            # that mirrors the existing `alerts_emitted_total`
            # accounting for the contiguous-days check above.
            state["stuck_replica_streak"] = int(
                result.get("stuck_replica_streak") or 0
            )
            state["stuck_replica"] = result.get("stuck_replica")
            if result.get("stuck_replica_alert_emitted"):
                state["stuck_replica_alerts_total"] = (
                    int(state.get("stuck_replica_alerts_total") or 0) + 1
                )
            state["runs_total"] = int(state.get("runs_total") or 0) + 1
            state["rows_inserted_total"] = (
                int(state.get("rows_inserted_total") or 0) + inserted_total
            )
            state["alerts_emitted_total"] = (
                int(state.get("alerts_emitted_total") or 0)
                + len(result["alerts"])
            )
            logger.info(
                "topup_5m_tick_done",
                extra={
                    "rows_inserted": inserted_total,
                    "alerts": result["alerts"],
                    "min_contiguous_days": (
                        min(result["contiguous_days_per_coin"].values())
                        if result["contiguous_days_per_coin"] else None
                    ),
                },
            )
            return "ok"
        except Exception as exc:  # noqa: BLE001 - scheduler must never die
            state["last_attempt_outcome"] = "error"
            state["last_error"] = str(exc)
            logger.warning(
                "topup_5m_tick_failed", extra={"error": str(exc)},
            )
            return "error"
        finally:
            state["last_finished_at"] = time.time()
            _run_lock.release()
    finally:
        # Task #603 — persist the cumulative counters to `app_settings`
        # so the dashboard tiles survive ml-engine restarts. We do this
        # after EVERY outcome (disabled / skipped_busy / skipped_locked
        # / ok / error) so even a tick that bumped only `ticks_total`
        # flushes that bump. Best-effort: a DB hiccup logs but never
        # breaks the tick loop.
        try:
            _save_counters_blocking({
                k: int(state.get(k) or 0) for k in _PERSISTED_COUNTER_KEYS
            })
        except Exception as exc:  # noqa: BLE001 - persist is best-effort
            logger.warning(
                "topup_5m_counters_persist_failed",
                extra={"error": str(exc)},
            )


def _scheduler_loop() -> None:
    # Small startup delay so the lifespan + DB pool finish initialising
    # before the first DB call.
    if _stop_event.wait(STARTUP_DELAY_SECONDS):
        return
    while not _stop_event.is_set():
        try:
            _tick()
        except Exception as exc:  # noqa: BLE001
            state["last_error"] = str(exc)
            logger.warning("topup_5m_loop_failed", extra={"error": str(exc)})
        if _stop_event.wait(INTERVAL_SECONDS):
            return


def start_scheduler() -> None:
    """Spawn the daemon thread. Honours the same `ML_AUTO_RETRAIN_TEST_DISABLE`
    env switch the other ml-engine daemons use so the FastAPI test client
    never actually fires this loop.
    """
    global _thread
    if not ENABLED:
        logger.info("topup_5m_disabled")
        return
    if os.environ.get("ML_AUTO_RETRAIN_TEST_DISABLE") == "1":
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(
        target=_scheduler_loop, daemon=True, name="ml-5m-topup",
    )
    _thread.start()
    logger.info(
        "topup_5m_started",
        extra={
            "interval_seconds": INTERVAL_SECONDS,
            "window_days": WINDOW_DAYS,
            "alert_below_days": ALERT_BELOW_DAYS,
        },
    )


def stop_scheduler() -> None:
    global _thread
    _stop_event.set()
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _thread = None
