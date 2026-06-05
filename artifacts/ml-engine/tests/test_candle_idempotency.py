"""Idempotent candle insertion under restart (task #345).

The 5m+ training pipeline reads aggregated bars from `price_candles`,
which is repopulated by `scripts/backfill_history.py:insert_candles_batch`.
That helper relies on the (coin_id, timeframe, bucket_start) unique
index and an `ON CONFLICT DO NOTHING` to be a no-op when the same
backfill window is re-run after a restart — without that property a
crashed-and-restarted trainer cron would either double-count bars
(silently inflating training-set weight on the overlap) or fail loudly
on every restart.

The test covers exactly that contract:

  1. Insert a batch of synthetic candles for an unused coin_id.
  2. "Restart": call insert_candles_batch with the SAME batch a second
     time. Assert the returned newly-inserted-row count is 0.
  3. Insert a batch that overlaps partially with the first. Assert the
     returned count equals the number of NEW (non-overlapping) buckets.
  4. Verify the row count in the table matches the union of all three
     batches — no duplicates anywhere.

The test only runs when DATABASE_URL is set (CI / local dev) and skips
otherwise. It cleans up the synthetic coin_id before and after so a
half-finished previous run cannot poison the assertions.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — idempotency test needs a real Postgres",
)


def _make_candles(start: datetime, n: int, *, base_price: float = 100.0):
    """Return n consecutive 1h OHLCV candles starting at `start`."""
    out = []
    for i in range(n):
        ts = start + timedelta(hours=i)
        o = base_price + i
        h = o + 0.5
        low = o - 0.5
        c = o + 0.25
        v = 1000.0 + i
        out.append((ts, float(o), float(h), float(low), float(c), float(v)))
    return out


@pytest.mark.asyncio
async def test_insert_candles_batch_is_idempotent_under_restart():
    import asyncpg  # noqa: F401  (ensure the driver is installed)
    from app.db import close_pool, init_pool
    from scripts.backfill_history import insert_candles_batch

    coin_id = f"__test_idempotency_{uuid.uuid4().hex[:8]}"
    timeframe = "1h"
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    pool = await init_pool()

    async def _cleanup() -> None:
        async with pool.acquire() as con:
            await con.execute(
                "DELETE FROM price_candles WHERE coin_id = $1", coin_id,
            )

    try:
        await _cleanup()

        # 1. Initial insert — all 10 buckets are new.
        first = _make_candles(start, 10)
        n1 = await insert_candles_batch(
            pool, coin_id, timeframe, first, source="test", dry_run=False,
        )
        assert n1 == 10, f"first insert should write all 10 rows, got {n1}"

        # Snapshot row count, OHLCV-sum, and per-bucket OHLC tuples after
        # the first insert. The post-restart snapshot MUST match exactly
        # — anything else means we either double-counted volume or
        # mutated existing rows.
        async def snapshot():
            async with pool.acquire() as con:
                rows = await con.fetch(
                    "SELECT bucket_start, open, high, low, close, volume "
                    "FROM price_candles WHERE coin_id = $1 AND timeframe = $2 "
                    "ORDER BY bucket_start",
                    coin_id, timeframe,
                )
            return {
                "count": len(rows),
                "vol_sum": float(sum(float(r["volume"]) for r in rows)),
                "ohlc": {
                    r["bucket_start"]: (
                        float(r["open"]), float(r["high"]),
                        float(r["low"]),  float(r["close"]),
                    )
                    for r in rows
                },
            }

        snap1 = await snapshot()
        assert snap1["count"] == 10
        # Expected SUM(volume) = 1000 + 1001 + ... + 1009 = 10045
        assert abs(snap1["vol_sum"] - 10045.0) < 1e-9, (
            f"first insert SUM(volume) wrong: got {snap1['vol_sum']}"
        )

        # 2. Restart: same batch re-applied. Must be a complete no-op:
        # row count, SUM(volume), and per-bucket OHLC all unchanged.
        n2 = await insert_candles_batch(
            pool, coin_id, timeframe, first, source="test", dry_run=False,
        )
        assert n2 == 0, (
            f"restart must be a no-op via ON CONFLICT, got {n2} new rows"
        )
        snap2 = await snapshot()
        assert snap2["count"] == snap1["count"], "restart changed row count"
        assert abs(snap2["vol_sum"] - snap1["vol_sum"]) < 1e-9, (
            f"restart leaked volume: pre={snap1['vol_sum']} post={snap2['vol_sum']} "
            "— ON CONFLICT path is silently double-counting"
        )
        assert snap2["ohlc"] == snap1["ohlc"], (
            "restart mutated existing OHLC values — ON CONFLICT must be DO NOTHING, "
            "not DO UPDATE on already-present buckets"
        )

        # 3. Overlapping batch — 5 old buckets + 5 new ones. The 5
        # overlapping buckets MUST keep their original OHLCV (DO NOTHING),
        # and SUM(volume) must increase by exactly the 5 new buckets'
        # contribution. _make_candles uses volume=1000+i for i in 0..n-1,
        # so the 5 new buckets (i=5..9 in the second batch) contribute
        # 1005+1006+1007+1008+1009 = 5035.
        overlapping = _make_candles(
            start + timedelta(hours=5), 10, base_price=999.0,
        )
        n3 = await insert_candles_batch(
            pool, coin_id, timeframe, overlapping, source="test", dry_run=False,
        )
        assert n3 == 5, (
            f"overlapping insert should add only the 5 new buckets, got {n3}"
        )
        snap3 = await snapshot()
        assert snap3["count"] == 15, (
            f"row count must equal union of inserts (15), got {snap3['count']}"
        )
        expected_vol = snap1["vol_sum"] + (1005 + 1006 + 1007 + 1008 + 1009)
        assert abs(snap3["vol_sum"] - expected_vol) < 1e-9, (
            f"overlap leaked volume: expected {expected_vol}, got {snap3['vol_sum']}"
        )
        # The 5 overlapping buckets (hours 5..9) keep their ORIGINAL OHLC,
        # NOT the `base_price=999` OHLC carried by the second batch.
        for hour in range(5, 10):
            ts = start + timedelta(hours=hour)
            assert snap3["ohlc"][ts] == snap1["ohlc"][ts], (
                f"bucket {ts} was mutated by overlapping insert — "
                f"pre={snap1['ohlc'][ts]} post={snap3['ohlc'][ts]}"
            )
    finally:
        await _cleanup()
        await close_pool()
