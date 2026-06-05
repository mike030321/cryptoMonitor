"""Round-5 helper: Bitstamp 1m OHLCV backfill into ``price_candles``.

Why this exists
---------------
Task #643 needs 365+ days of 1m bars for BTC and ETH in the
``price_candles`` table so the labels-research training pipeline can
read them via ``data.fetch_real_candles``. The default ingest path
(``scripts/backfill_history.py --target candles --timeframes 1m
--coins bitcoin ethereum``) sources from OKX
``v5/market/history-candles`` which serves ~500 days of history but
only at 100 bars per request and ~0.5 s per request — paging back 1
year of 1m data takes ~50 minutes per coin (5,256 requests × 0.5 s).

Bitstamp's ``/api/v2/ohlc/<pair>/`` endpoint serves identical 1m OHLCV
data at **1000 bars per request** with no API key required — empirically
~10× faster than OKX for the same coverage, finishing 365 days for one
coin in ~5 minutes instead of ~50.

Real-data only — every row written to ``price_candles`` carries
``source='bitstamp'`` so the campaign's source allow-list and the
research helper's source-attribution audit can both see exactly which
provider supplied each bar. No synthetic data, no fabricated OHLCV.

This is intentionally a small standalone helper inside the
``labels_research`` package (per the task #643 hard rule "new code
lives in app/training/labels_research/"). It does NOT modify
``scripts/backfill_history.py`` and does not touch the production
backfill scheduler. It calls ``insert_candles_batch`` from the
existing backfill helper so the row write goes through the same
cadence guard and idempotency contract as every other 1m / 5m
``price_candles`` writer.

Usage::

    python -m app.training.labels_research.bitstamp_1m_backfill \
        --coins bitcoin ethereum --days 400

The script paginates BACKWARD from "now" using Bitstamp's ``end=``
unix-second cursor, batching writes per page to keep memory low.
Stops when (a) ``days`` of history have been covered, (b) the venue
returns an empty page, or (c) a hard 1500-page safety cap is hit
(>= 100 days of 1m).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import asyncpg
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from scripts.backfill_history import insert_candles_batch  # noqa: E402

logger = logging.getLogger("labels_research.bitstamp_1m_backfill")


BITSTAMP_PAIRS: dict[str, str] = {
    # task #643 round-5 ingest. JUP is intentionally absent — Bitstamp
    # does not list JUP-USD.
    "bitcoin": "btcusd",
    "ethereum": "ethusd",
}

BITSTAMP_BASE = "https://www.bitstamp.net/api/v2/ohlc"
BITSTAMP_PAGE = 1000  # rows per request (max permitted by venue)
STEP_1M_SEC = 60


async def _fetch_page(
    client: httpx.AsyncClient, pair: str, end_unix_sec: int,
) -> list[dict]:
    url = f"{BITSTAMP_BASE}/{pair}/"
    params = {
        "step": str(STEP_1M_SEC),
        "limit": str(BITSTAMP_PAGE),
        "end": str(end_unix_sec),
    }
    r = await client.get(url, params=params, timeout=30.0)
    r.raise_for_status()
    body = r.json()
    return list(body.get("data", {}).get("ohlc", []))


def _to_candle_tuple(row: dict) -> tuple[datetime, float, float, float, float, float]:
    """Bitstamp returns one OHLC row as
    {timestamp, open, high, low, close, volume} (all strings,
    timestamp = bar OPEN unix seconds).

    ``insert_candles_batch`` expects ``(bucket_start_dt, open, high,
    low, close, volume)`` — bucket_start aligns with bar OPEN, same
    convention as the OKX OHLCV writer in ``scripts/backfill_history.py``.
    """
    ts = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
    return (
        ts,
        float(row["open"]), float(row["high"]),
        float(row["low"]),  float(row["close"]),
        float(row["volume"]),
    )


async def backfill_one(
    client: httpx.AsyncClient, pool: asyncpg.Pool,
    coin_id: str, days: int, *, dry_run: bool = False,
) -> dict:
    pair = BITSTAMP_PAIRS.get(coin_id)
    if pair is None:
        raise SystemExit(
            f"no Bitstamp pair mapping for {coin_id} "
            f"(supported: {list(BITSTAMP_PAIRS)})"
        )
    now_sec = int(datetime.now(tz=timezone.utc).timestamp())
    horizon_sec = now_sec - days * 86400
    end_cursor = now_sec
    total_pulled = 0
    total_inserted = 0
    pages = 0
    while True:
        try:
            rows = await _fetch_page(client, pair, end_cursor)
        except httpx.HTTPError as exc:
            logger.warning(
                "bitstamp_request_failed coin=%s pair=%s err=%s",
                coin_id, pair, exc,
            )
            break
        if not rows:
            break
        candles = [_to_candle_tuple(r) for r in rows]
        # Bitstamp returns oldest-first within a page; the oldest
        # bar in this page sets the next backward cursor.
        oldest_ts_sec = int(rows[0]["timestamp"])
        inserted = await insert_candles_batch(
            pool, coin_id, "1m", candles,
            source="bitstamp", dry_run=dry_run,
        )
        total_pulled += len(rows)
        total_inserted += inserted
        pages += 1
        if pages % 10 == 0:
            logger.info(
                "bitstamp_progress coin=%s pages=%d pulled=%d inserted=%d "
                "oldest=%s",
                coin_id, pages, total_pulled, total_inserted,
                datetime.fromtimestamp(oldest_ts_sec, tz=timezone.utc).isoformat(),
            )
        if oldest_ts_sec <= horizon_sec:
            break
        if pages >= 1500:
            logger.warning(
                "bitstamp_page_cap coin=%s pages=%d", coin_id, pages,
            )
            break
        # Step end backward by (page_size * step) so the next request
        # returns the bars immediately preceding the current oldest.
        end_cursor = oldest_ts_sec - 1
        await asyncio.sleep(0.05)
    logger.info(
        "bitstamp_done coin=%s pages=%d pulled=%d inserted=%d days=%d",
        coin_id, pages, total_pulled, total_inserted, days,
    )
    return {
        "coin": coin_id, "pair": pair, "pages": pages,
        "pulled": total_pulled, "inserted": total_inserted,
        "days_requested": days, "source": "bitstamp",
    }


async def main_async(args) -> None:
    dsn = os.environ["DATABASE_URL"]
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)
    try:
        async with httpx.AsyncClient() as client:
            for coin in args.coins:
                res = await backfill_one(
                    client, pool, coin, args.days, dry_run=args.dry_run,
                )
                logger.info("backfill_result %s", res)
    finally:
        await pool.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--coins", nargs="*", default=list(BITSTAMP_PAIRS),
        help="Coin ids to backfill (default: all configured: "
             f"{list(BITSTAMP_PAIRS)})",
    )
    p.add_argument(
        "--days", type=int, default=400,
        help="Days of history to fetch (default 400 — gives the "
             "task #643 ingestion gate one week of headroom over its "
             "365 d floor).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
