"""Read-only access to the shared Postgres `price_history` table.

We use the same DATABASE_URL as the Node API server but only ever issue
SELECTs, and we always exclude `is_synthetic = true` rows at the SQL layer
so synthetic seed data can never leak into features.

Null-policy: legacy rows predate the `is_synthetic` column and have NULL.
The TS pattern-analyzer (`filterRealTicksOnly` -> `!r.isSynthetic`) treats
NULL as real, so we match that here (`is_synthetic IS NULL OR = false`)
to keep Python and Node feature inputs identical.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional

import asyncpg

logger = logging.getLogger("ml-engine.db")

_pool: Optional[asyncpg.Pool] = None
_pool_loop: Optional[asyncio.AbstractEventLoop] = None
_pool_locks: dict[int, asyncio.Lock] = {}

# Pool creation timeout. Replit's managed Postgres can be slow to TLS-handshake
# under load (we've seen 2-3s); the default asyncpg timeout (60s for the whole
# pool but per-connect ~ short) was racing under concurrent first-call traffic
# and producing CancelledError / TimeoutError 500s. Giving the create call a
# generous budget eliminates the cold-start race.
_POOL_CREATE_TIMEOUT_S = 30.0


def _lock_for_loop(loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
    # asyncio.Lock is loop-bound; cache one per loop so concurrent first
    # callers serialize on pool creation instead of all racing to create_pool.
    lock = _pool_locks.get(id(loop))
    if lock is None:
        lock = asyncio.Lock()
        _pool_locks[id(loop)] = lock
    return lock


async def init_pool() -> asyncpg.Pool:
    """Initialize (or return) the asyncpg pool, rebinding it to the current
    event loop when needed.

    asyncpg pools are bound to the loop that created them. If a different
    loop later tries to acquire a connection (e.g. uvicorn worker recycle,
    a startup task that ran on a different loop, or a training script that
    spun up its own loop) asyncpg raises:
        "got Future ... attached to a different loop".
    Detect that mismatch up front and recreate the pool on the active loop.

    A per-loop asyncio.Lock serializes concurrent pool creation so that the
    first burst of requests after startup (or after a loop swap) does not
    trigger N parallel `create_pool` attempts and N TLS handshakes against
    the upstream Postgres.
    """
    global _pool, _pool_loop
    running_loop = asyncio.get_running_loop()
    if _pool is not None and _pool_loop is running_loop:
        return _pool

    async with _lock_for_loop(running_loop):
        # Re-check inside the lock: another coroutine may have just built it.
        if _pool is not None and _pool_loop is running_loop:
            return _pool
        # Pool exists but is bound to a different (likely closed) loop —
        # discard it. Don't await its close(); it would dispatch onto the
        # dead loop and raise. Letting GC reap the underlying sockets is
        # safe here because we only hold read connections.
        if _pool is not None and _pool_loop is not running_loop:
            _pool = None
            _pool_loop = None
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL is required")
        _pool = await asyncio.wait_for(
            asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4),
            timeout=_POOL_CREATE_TIMEOUT_S,
        )
        _pool_loop = running_loop
        return _pool


def advisory_lock_key(label: str) -> int:
    """Derive a stable signed bigint advisory-lock key from a human label.

    Postgres `pg_try_advisory_lock(bigint)` takes a single int64. Using a
    blake2b-8 digest of a short label keeps the key recognisable in
    `pg_locks` (each scheduler picks its own label) while staying
    collision-resistant across the handful of schedulers we run.

    The key is deterministic, so every ml-engine replica that imports
    this module computes the same value for the same label and therefore
    contends on the same lock.
    """
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


@contextlib.asynccontextmanager
async def try_advisory_lock(key: int) -> AsyncIterator[bool]:
    """Try to acquire a session-scoped Postgres advisory lock.

    Yields ``True`` if this call holds the lock (released on exit) and
    ``False`` if another session already holds it. The lock is bound to
    a single asyncpg connection from the pool, so a crashed worker can
    never leave it stuck — Postgres releases all advisory locks when the
    session ends.

    Use this to coordinate process-local schedulers across multiple
    ml-engine replicas: wrap the body of a ``_tick`` so only one process
    actually fires the work per tick. The non-blocking
    ``pg_try_advisory_lock`` means losers return immediately and treat
    the tick as "someone else is doing it" instead of piling up.
    """
    pool = await init_pool()
    async with pool.acquire() as con:
        got = bool(await con.fetchval(
            "SELECT pg_try_advisory_lock($1::bigint)", key,
        ))
        try:
            yield got
        finally:
            if got:
                try:
                    await con.fetchval(
                        "SELECT pg_advisory_unlock($1::bigint)", key,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Connection is about to be returned to the pool /
                    # closed; swallow so the caller's outcome stands.
                    # Postgres will release on session end regardless.
                    logger.warning(
                        "advisory_unlock_failed",
                        extra={"key": key, "error": str(exc)},
                    )


async def close_pool() -> None:
    global _pool, _pool_loop
    if _pool is not None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is _pool_loop:
            await _pool.close()
        _pool = None
        _pool_loop = None


async def fetch_real_ticks(
    coin_id: str, lookback_ms: int, now: Optional[datetime] = None
) -> list[tuple[datetime, float]]:
    """Return real (non-synthetic) (timestamp, price) ticks for the lookback
    window. Returned oldest-first.
    """
    pool = await init_pool()
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(milliseconds=lookback_ms)
    rows = await pool.fetch(
        """
        SELECT timestamp, price
        FROM price_history
        WHERE coin_id = $1
          AND (is_synthetic IS NULL OR is_synthetic = false)
          AND timestamp >= $2
          AND timestamp <= $3
        ORDER BY timestamp ASC
        """,
        coin_id,
        start,
        end,
    )
    return [(r["timestamp"], float(r["price"])) for r in rows]


async def fetch_real_candles(
    coin_id: str,
    timeframe: str,
    lookback_ms: int,
    now: Optional[datetime] = None,
) -> list[tuple[datetime, float]]:
    """Task #317 — read native-cadence aggregated candles directly from
    `price_candles` for the given (coin, timeframe). Returns
    `(bucket_start, close)` oldest-first.

    Bars at a known timeframe live in `price_candles` (written by the
    backfill modules in #306) and never share storage with the raw-tick
    `price_history` stream. The trainer reads candles directly when a
    native cadence exists for the requested timeframe so the resampler
    can never silently merge a daily bar into a 5m bucket.

    Returns an empty list when no candles exist (caller falls back to
    `fetch_real_ticks` + `resample_to_candles` for that timeframe). A
    missing `price_candles` table (fresh DB / unit test environment) is
    also treated as "no candles".
    """
    try:
        pool = await init_pool()
        end = now or datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=lookback_ms)
        rows = await pool.fetch(
            """
            SELECT bucket_start, close
            FROM price_candles
            WHERE coin_id = $1
              AND timeframe = $2
              AND bucket_start >= $3
              AND bucket_start <= $4
            ORDER BY bucket_start ASC
            """,
            coin_id,
            timeframe,
            start,
            end,
        )
    except Exception:  # noqa: BLE001 — table missing in tests / fresh envs
        return []
    return [(r["bucket_start"], float(r["close"])) for r in rows]


async def fetch_real_ticks_with_provenance(
    coin_id: str, lookback_ms: int, now: Optional[datetime] = None,
) -> dict:
    """Task #267 — provenance-aware tick loader.

    Returns:
        {
            "ticks": list[(datetime, float)],   # real rows only, oldest-first
            "rows_real": int,                   # how many real rows we loaded
            "rows_synthetic": int,              # synthetic rows in the same window
            "rejected_synthetic": bool,         # True if any synthetic row would
                                                # have been included
            "window_start": datetime,
            "window_end": datetime,
        }

    The actual feature/training pipeline still consumes real rows only;
    the synthetic count is surfaced so the training contract's provenance
    guard can decide whether to abort the slice and stamp `report.json`
    with an explicit rejection reason instead of silently training on
    a window that contains dirty data.
    """
    pool = await init_pool()
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(milliseconds=lookback_ms)
    rows = await pool.fetch(
        """
        SELECT timestamp, price, COALESCE(is_synthetic, false) AS is_synthetic
        FROM price_history
        WHERE coin_id = $1
          AND timestamp >= $2
          AND timestamp <= $3
        ORDER BY timestamp ASC
        """,
        coin_id,
        start,
        end,
    )
    ticks: list[tuple[datetime, float]] = []
    n_real = 0
    n_synth = 0
    for r in rows:
        if r["is_synthetic"]:
            n_synth += 1
            continue
        n_real += 1
        ticks.append((r["timestamp"], float(r["price"])))
    return {
        "ticks": ticks,
        "rows_real": n_real,
        "rows_synthetic": n_synth,
        "rejected_synthetic": n_synth > 0,
        "window_start": start,
        "window_end": end,
    }


async def fetch_recent_news_tags(
    coin_id: str, limit: int = 20, now: Optional[datetime] = None
) -> list[str]:
    """Return the deduplicated set of news tags emitted by the LLM
    classifier for `coin_id` (or 'macro') within the last 24h, ordered
    most-recent-first. Used by the feature pipeline to feed structured
    news context into LightGBM.

    Returns an empty list if no tags are present so callers can pass it
    straight into `build_feature_vector`.
    """
    pool = await init_pool()
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(hours=24)
    rows = await pool.fetch(
        """
        SELECT tags
        FROM news_tags
        WHERE (coin_id = $1 OR coin_id = 'macro')
          AND created_at >= $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        coin_id,
        start,
        limit,
    )
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in rows:
        for t in (r["tags"] or []):
            if isinstance(t, str) and t not in seen_set:
                seen.append(t)
                seen_set.add(t)
    return seen


async def fetch_market_signals(
    coin_id: str, lookback_ms: int, now: Optional[datetime] = None,
) -> list[dict]:
    """Task #271 — return per-coin snapshots of the external exchange
    streams written by the api-server's `market-signals-poller` over the
    lookback window, oldest-first.

    Each row is a dict with keys: `timestamp_ms`, `funding_rate`,
    `open_interest_usd`, `liquidations_1h_usd`, `bid_ask_spread_bps`,
    `mid_price`. Missing columns / a missing table are silently treated
    as "no rows" so the trainer falls back to the registered safe
    defaults from `EXTERNAL_STREAM_DEFAULTS`.
    """
    try:
        pool = await init_pool()
        end = now or datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=lookback_ms)
        rows = await pool.fetch(
            """
            SELECT timestamp,
                   funding_rate,
                   open_interest_usd,
                   liquidations_1h_usd,
                   bid_ask_spread_bps,
                   mid_price
            FROM market_signals
            WHERE coin_id = $1
              AND timestamp >= $2
              AND timestamp <= $3
            ORDER BY timestamp ASC
            """,
            coin_id,
            start,
            end,
        )
    except Exception:  # noqa: BLE001 — table missing in tests / fresh envs
        return []
    out: list[dict] = []
    for r in rows:
        out.append({
            "timestamp_ms": int(r["timestamp"].timestamp() * 1000),
            "funding_rate": (
                float(r["funding_rate"]) if r["funding_rate"] is not None else None
            ),
            "open_interest_usd": (
                float(r["open_interest_usd"]) if r["open_interest_usd"] is not None else None
            ),
            "liquidations_1h_usd": (
                float(r["liquidations_1h_usd"]) if r["liquidations_1h_usd"] is not None else None
            ),
            "bid_ask_spread_bps": (
                float(r["bid_ask_spread_bps"]) if r["bid_ask_spread_bps"] is not None else None
            ),
            "mid_price": (
                float(r["mid_price"]) if r["mid_price"] is not None else None
            ),
        })
    return out


async def fetch_lead_price_series(
    reference_coin_id: str, lookback_ms: int, now: Optional[datetime] = None,
) -> list[tuple[int, float]]:
    """Task #271 — return BTC/ETH mid-price snapshots over the lookback
    window so the trainer can compute the cross-coin lead returns for
    every other coin.

    The poller writes BTC/ETH mid prices into `market_signals` under the
    pseudo-coin ids `btc` and `eth`. Each returned tuple is
    `(timestamp_ms, mid_price)` sorted oldest-first.
    """
    rows = await fetch_market_signals(reference_coin_id, lookback_ms, now=now)
    return [
        (r["timestamp_ms"], r["mid_price"])
        for r in rows
        if r["mid_price"] is not None and r["mid_price"] > 0
    ]


async def fetch_scope_for_active_champion(
    coin_id: str, timeframe: str, model_version: str,
) -> Optional[dict]:
    """Task #654 — return the `scope_constraint` payload (a dict) for the
    SPECIFIC active-champion registry row identified by
    `(coin_id, timeframe, model_version)`. The lookup is per-resolved-
    champion (NOT a global allowlist) so the scope acts as a per-model
    paper-trading fence: only that champion's recorded
    `(coins, timeframes)` are served by it; everything else short-
    circuits to `out_of_scope` so the caller renders an honest fence
    instead of inventing a prediction. Returns `None` when the row has
    no scope (legacy 3-class champions impose no restriction) or on
    any DB failure (table missing in tests, no DATABASE_URL,
    connectivity blip) — fail-open so the existing serving path stays
    intact.

    Authoritative writer: `app.registry_lifecycle.promote_shadow_to_serving`.
    """
    try:
        pool = await init_pool()
        row = await pool.fetchrow(
            """
            SELECT scope_constraint
            FROM model_registry
            WHERE coin_id = $1 AND timeframe = $2 AND model_version = $3
              AND state = 'champion' AND is_active = true
              AND scope_constraint IS NOT NULL
            LIMIT 1
            """,
            coin_id, timeframe, model_version,
        )
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    scope = row["scope_constraint"]
    if scope is None:
        return None
    if isinstance(scope, str):
        try:
            import json as _json

            scope = _json.loads(scope)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(scope, dict):
        return None
    return scope


async def fetch_quarantined_versions() -> set[tuple[str, str, str]]:
    """Task #232 — return the set of (coin_id, timeframe, model_version)
    triples that are currently quarantined in the api-server's
    `model_registry` table. Used by `/ml/predict`'s model resolver to
    EXCLUDE any registry slot whose state == 'quarantined' from live
    trade selection (with fallback to the previous champion / pooled).

    On any DB failure (table missing in tests, connectivity blip) returns
    an empty set so the caller falls back to the file-only resolution
    path. The api-server's quarantine writer is the source of truth; this
    is a read-only consumer.
    """
    try:
        pool = await init_pool()
        rows = await pool.fetch(
            """
            SELECT coin_id, timeframe, model_version
            FROM model_registry
            WHERE state = 'quarantined'
              AND is_active = true
            """
        )
    except Exception:  # noqa: BLE001
        return set()
    return {
        (str(r["coin_id"]), str(r["timeframe"]), str(r["model_version"]))
        for r in rows
    }
