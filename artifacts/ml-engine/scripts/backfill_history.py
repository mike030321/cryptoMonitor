"""One-shot backfill of historical real OHLCV.

Real-data only. No synthetic ticks, ever. Every row written here
originated from a real exchange (OKX, Coinbase) or a licensed
aggregator (CoinMarketCap). Idempotent — re-running fills only missing
windows; never overwrites existing rows.

Two write targets, one script:
  * `price_history` (single un-typed close-only series) — 1m base
    cadence only. Used by the live poller and the resampler fallback.
  * `price_candles` (per-timeframe OHLCV table from task #317) —
    5m / 1h / 2h / 6h / 1d native bars. Read directly by the trainer
    via `db.fetch_real_candles` so the resampler can never silently
    merge a coarser bar into a finer bucket.

Sources:
  * OKX `/api/v5/market/history-candles` — primary, free, supports 1m
    through 1d back to listing date. Used for every tradeable timeframe
    we train on. Writes OHLCV into `price_candles` for higher
    timeframes; close-only ticks into `price_history` for 1m.
  * Coinbase Exchange `/products/{id}/candles` — secondary fallback for
    coins where OKX truncates `history-candles` short of the 5m hard
    gate (task #409). Free, public, no key required. Returns up to 300
    bars per request and is paginated backward by stepping `end` to the
    oldest bucket seen. Writes OHLCV into `price_candles` with
    `source="coinbase"` (whitelisted alongside OKX in the campaign's
    real-source allow-list).
  * CoinMarketCap `/v1/cryptocurrency/ohlcv/historical` — fallback for
    the 1d series only when OKX returns fewer than the requested number
    of days. Writes close-only ticks into `price_history`. (CMC is not
    written into `price_candles` in Phase 1; that's Phase 2 cross-source
    work.)

Per-timeframe windows (kept conservative so a fresh container can finish
in a few minutes per coin while still giving the trainer years of bars):
  1m  → 14 days    (≈ 20k candles / coin)   → price_history
  5m  → 310 days   (≈ 89k candles / coin)   → price_candles  [task #378]
  1h  → 365 days   (≈ 8.7k candles / coin)  → price_candles
  2h  → 365 days   (≈ 4.4k candles / coin)  → price_candles
  6h  → 365 days   (≈ 1.5k candles / coin)  → price_candles
  1d  → 1100 days  (~ 1.1k candles / coin)  → price_candles  [task #417]

The 1d window is wider than the others by design (task #417): the 1d
holdout was capped at ~265 rows under the legacy 365d envelope, leaving
the directional-accuracy gate fighting a binomial noise band of
σ ≈ 0.031. Pulling ≥1000 contiguous days of OKX 1d candles grows the
holdout to ~720+ rows and shrinks σ to ~0.019, giving the original 0.50
gate honest statistical power. OKX `history-candles` paginates 100 bars
per request, so 1100 days of 1d fits in ~11 pages — well under the
200-page single-invocation cap.

Usage:
    # Default: 1m → price_history, 5m/1h/2h/6h/1d → price_candles.
    python -m scripts.backfill_history
    # Subset of coins or timeframes:
    python -m scripts.backfill_history --coins bonk pepe
    python -m scripts.backfill_history --timeframes 5m 1h
    # Force one target only:
    python -m scripts.backfill_history --target candles --timeframes 5m
    python -m scripts.backfill_history --target history --timeframes 1m
    python -m scripts.backfill_history --dry-run        # don't write
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.training.train import DEFAULT_COINS  # noqa: E402
from app.cadence_guard import assert_native_cadence  # noqa: E402

logger = logging.getLogger("backfill")

OKX_BASE = "https://www.okx.com/api/v5/market"
CMC_BASE = "https://pro-api.coinmarketcap.com"
COINBASE_BASE = "https://api.exchange.coinbase.com"

OKX_SYMBOLS: dict[str, str] = {
    "pepe": "PEPE-USDT",
    "bonk": "BONK-USDT",
    "floki-inu": "FLOKI-USDT",
    "dogwifcoin": "WIF-USDT",
    "sei-network": "SEI-USDT",
    "render-token": "RENDER-USDT",
    "injective-protocol": "INJ-USDT",
    "celestia": "TIA-USDT",
    "worldcoin-wld": "WLD-USDT",
    "jupiter-exchange-solana": "JUP-USDT",
    # Task #643 — BTC/ETH added as ingestion targets for the
    # quintile/sparse opportunity-label research study. They are NOT
    # promoted to MONITORED_COINS; the study writes per-(coin, tf)
    # OHLCV into price_candles / price_history under the same contracts
    # as the alt-coin fleet so the labels_research training pipeline
    # can read them via fetch_real_candles. No live trader / no
    # paper-trader uses these series.
    "bitcoin": "BTC-USDT",
    "ethereum": "ETH-USDT",
}

# Coinbase Exchange product map. Originally introduced (Task #409) as
# a secondary 5m source for SEI when OKX `history-candles` truncated
# SEI-USDT 5m at ~161 days back. Empirically (April 2026) the same
# truncation applies to EVERY OKX 5m series — `_fetch_okx_raw` returns
# only the last ~60-100 days regardless of coin — while Coinbase
# Exchange's public `products/<id>/candles` serves 5m back ≥335 days
# with no API key from a non-restricted region. So as of Task #603
# Coinbase is the canonical 5m source for every monitored coin that
# lists on Coinbase Exchange; OKX remains the daily HEAD source for
# the lone holdout (JUP, not listed on Coinbase with usable history)
# and the canonical source for higher timeframes (1h+). See
# `prefer_coinbase_for_5m()` and `fetch_5m_smart()` below.
COINBASE_PRODUCTS: dict[str, str] = {
    "pepe": "PEPE-USD",
    "bonk": "BONK-USD",
    "floki-inu": "FLOKI-USD",
    "dogwifcoin": "WIF-USD",
    "sei-network": "SEI-USD",
    "render-token": "RENDER-USD",
    "injective-protocol": "INJ-USD",
    "celestia": "TIA-USD",
    "worldcoin-wld": "WLD-USD",
    # Task #643 — BTC/ETH research-only ingest. Coinbase serves 5m
    # back ~335 days for both, so prefer Coinbase as the deep-history
    # 5m source via prefer_coinbase_for_5m(). 1m still comes from OKX
    # (Coinbase truncates 1m at ~6h). See Task #643 plan.
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    # NB: `jupiter-exchange-solana` intentionally absent — Coinbase
    # Exchange does not list JUP-USD with usable history (zero bars at
    # 60-200d back per probe). JUP keeps OKX as its 5m source which
    # caps at ~66d coverage; the 5m gate must accommodate this or
    # exclude JUP explicitly.
}

# CMC slug overrides (must match coins.ts MONITORED_COINS.cmcSlug).
CMC_SLUGS: dict[str, str] = {
    "pepe": "pepe",
    "bonk": "bonk1",
    "floki-inu": "floki-inu",
    "dogwifcoin": "dogwifhat",
    "sei-network": "sei",
    "render-token": "render-token",
    "injective-protocol": "injective",
    "celestia": "celestia",
    "worldcoin-wld": "worldcoin-org",
    "jupiter-exchange-solana": "jupiter-ag",
}

# OKX `bar=` parameter per timeframe.
OKX_BAR: dict[str, str] = {
    "1m": "1m", "5m": "5m", "1h": "1H",
    "2h": "2H", "6h": "6H", "1d": "1D",
}
TF_MS: dict[str, int] = {
    "1m": 60_000, "5m": 5 * 60_000, "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000, "6h": 6 * 60 * 60_000, "1d": 24 * 60 * 60_000,
}
DEFAULT_DAYS_BY_TF: dict[str, int] = {
    # Task #417 — 1d bumped from 365 → 1100 so the trainer's 1d holdout
    # grows from ~265 rows to ~720+ rows. Other timeframes unchanged so
    # this fix doesn't pull a multi-year 5m/1h backfill we don't need.
    "1m": 14, "5m": 310, "1h": 365, "2h": 365, "6h": 365, "1d": 1100,
}

# Coinbase Exchange `granularity=` parameter per timeframe (seconds).
# Coinbase only supports a fixed set of granularities; any tf not listed
# here cannot be served by the Coinbase fallback path.
COINBASE_GRANULARITY: dict[str, int] = {
    "1m": 60, "5m": 300, "1h": 3600, "6h": 21600, "1d": 86400,
}
# Coinbase Exchange caps each `/candles` response at 300 datapoints.
# We use 290 per request for a small safety margin and to keep the
# round-trip count predictable.
_COINBASE_PAGE = 290


async def _fetch_okx_raw(
    client: httpx.AsyncClient, coin_id: str, timeframe: str, days: int,
    end_ts_ms: Optional[int] = None,
) -> list[list]:
    """Page OKX history-candles back `days` days. Returns raw rows
    `[ts, o, h, l, c, vol, ...]` newest-first as OKX returns them.
    Used by both the close-only and OHLCV variants below.

    When `end_ts_ms` is provided, the FIRST request seeds the OKX
    `after` cursor with that timestamp instead of "now", which lets a
    caller iteratively pull older windows than the 200-page hard cap
    permits in a single invocation. The OKX endpoint walks STRICTLY
    backward via `after`, so subsequent calls with progressively older
    `end_ts_ms` reach further back than the venue would otherwise return
    in a single sweep.
    """
    inst = OKX_SYMBOLS.get(coin_id)
    if inst is None:
        raise ValueError(f"no OKX symbol mapping for {coin_id}")
    bar = OKX_BAR[timeframe]
    end_ms = end_ts_ms if end_ts_ms is not None else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    horizon_ms = end_ms - days * 24 * 60 * 60 * 1000
    out: list[list] = []
    cursor: Optional[int] = end_ts_ms  # seed first request from the operator-supplied window
    pages = 0
    while True:
        params: dict[str, str] = {"instId": inst, "bar": bar, "limit": "100"}
        if cursor is not None:
            params["after"] = str(cursor)
        try:
            r = await client.get(
                f"{OKX_BASE}/history-candles", params=params, timeout=30.0,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "okx_request_failed coin=%s tf=%s err=%s", coin_id, timeframe, exc,
            )
            break
        body = r.json()
        if body.get("code") != "0":
            logger.warning(
                "okx_error coin=%s tf=%s body=%s", coin_id, timeframe, body,
            )
            break
        rows = body.get("data") or []
        if not rows:
            break
        oldest_open_ms = int(rows[-1][0])
        out.extend(rows)
        pages += 1
        if oldest_open_ms <= horizon_ms or pages > 1000:
            break
        cursor = oldest_open_ms
        await asyncio.sleep(0.18)  # well under 20 req/2s public limit
    return out


async def fetch_okx_candles(
    client: httpx.AsyncClient, coin_id: str, timeframe: str, days: int,
    end_ts_ms: Optional[int] = None,
) -> list[tuple[datetime, float]]:
    """Close-only fetcher for the legacy `price_history` writer.

    Stamps `ts` at bar CLOSE so the existing `price_history` consumers
    (which treat each row as a tick) keep their forward-walking
    semantics.
    """
    rows = await _fetch_okx_raw(client, coin_id, timeframe, days, end_ts_ms=end_ts_ms)
    bar_ms = TF_MS[timeframe]
    out: list[tuple[datetime, float]] = []
    for row in rows:
        open_time_ms = int(row[0])
        close_price = float(row[4])
        ts = datetime.fromtimestamp(
            (open_time_ms + bar_ms) / 1000.0, tz=timezone.utc,
        )
        out.append((ts, close_price))
    return out


async def fetch_okx_ohlcv(
    client: httpx.AsyncClient, coin_id: str, timeframe: str, days: int,
    end_ts_ms: Optional[int] = None,
) -> list[tuple[datetime, float, float, float, float, float]]:
    """OHLCV fetcher for the `price_candles` writer.

    Stamps `bucket_start` at bar OPEN — that's the canonical bucket key
    the `price_candles` unique index `(coin_id, timeframe, bucket_start)`
    expects, and what `db.fetch_real_candles` returns to the trainer.
    """
    rows = await _fetch_okx_raw(client, coin_id, timeframe, days, end_ts_ms=end_ts_ms)
    out: list[tuple[datetime, float, float, float, float, float]] = []
    for row in rows:
        try:
            open_time_ms = int(row[0])
            o = float(row[1]); h = float(row[2]); l = float(row[3])
            c = float(row[4]); v = float(row[5]) if len(row) > 5 else 0.0
        except (ValueError, IndexError):
            continue
        bucket_start = datetime.fromtimestamp(
            open_time_ms / 1000.0, tz=timezone.utc,
        )
        out.append((bucket_start, o, h, l, c, v))
    return out


async def _fetch_coinbase_raw(
    client: httpx.AsyncClient, coin_id: str, timeframe: str, days: int,
    end_ts_ms: Optional[int] = None,
) -> list[list]:
    """Page Coinbase Exchange `/candles` back `days` days. Returns rows
    `[time_s, low, high, open, close, vol]` newest-first (matches the
    Coinbase response ordering — note: NOT the OKX `[ts,o,h,l,c,v]`
    ordering, the caller must convert).

    Coinbase Exchange caps `/candles` at 300 datapoints per request, so
    we chunk by `_COINBASE_PAGE * granularity` seconds and walk `end`
    backward until we cross the requested horizon, the venue returns an
    empty page (listing date), or a safety cap fires. Public endpoint,
    no API key, ~10 req/s rate limit (we sleep 0.18 s between calls).

    Task #409 — secondary 5m source for SEI. Wired in via the
    `--source coinbase` switch on `scripts.backfill_history`.
    """
    product = COINBASE_PRODUCTS.get(coin_id)
    if product is None:
        raise ValueError(f"no Coinbase product mapping for {coin_id}")
    granularity = COINBASE_GRANULARITY.get(timeframe)
    if granularity is None:
        raise ValueError(
            f"timeframe {timeframe} not supported by Coinbase fallback "
            f"(supported: {sorted(COINBASE_GRANULARITY.keys())})"
        )
    page_secs = _COINBASE_PAGE * granularity
    end_s = (
        end_ts_ms // 1000 if end_ts_ms is not None
        else int(datetime.now(tz=timezone.utc).timestamp())
    )
    horizon_s = end_s - days * 24 * 60 * 60
    out: list[list] = []
    pages = 0
    cur_end_s = end_s
    while True:
        cur_start_s = cur_end_s - page_secs
        params = {
            "granularity": str(granularity),
            "start": datetime.fromtimestamp(cur_start_s, tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(cur_end_s, tz=timezone.utc).isoformat(),
        }
        try:
            r = await client.get(
                f"{COINBASE_BASE}/products/{product}/candles",
                params=params, timeout=30.0,
                headers={"User-Agent": "ml-engine-backfill/1.0"},
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "coinbase_request_failed coin=%s tf=%s err=%s",
                coin_id, timeframe, exc,
            )
            break
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            # Empty page = venue exhausted (listing date reached) or
            # Coinbase has no data for this window. Either way, stop.
            break
        # Coinbase returns rows in DESCENDING time order. The "oldest"
        # row in this batch is the LAST element.
        out.extend(rows)
        oldest_ts_s = int(rows[-1][0])
        pages += 1
        if oldest_ts_s <= horizon_s or pages > 1000:
            break
        # Step `cur_end_s` to one bar before the oldest seen so the next
        # request walks strictly older with no overlap.
        cur_end_s = oldest_ts_s - granularity
        await asyncio.sleep(0.18)
    return out


async def fetch_coinbase_ohlcv(
    client: httpx.AsyncClient, coin_id: str, timeframe: str, days: int,
    end_ts_ms: Optional[int] = None,
) -> list[tuple[datetime, float, float, float, float, float]]:
    """OHLCV fetcher for the `price_candles` writer via Coinbase.

    Stamps `bucket_start` at bar OPEN, matching the OKX path's contract
    so the `price_candles` unique index `(coin_id, timeframe,
    bucket_start)` interleaves cleanly with already-present OKX bars.
    Coinbase row layout is `[time_s, low, high, open, close, vol]`,
    NOT OKX's `[ts, o, h, l, c, v]` — we re-order at parse time.
    """
    rows = await _fetch_coinbase_raw(
        client, coin_id, timeframe, days, end_ts_ms=end_ts_ms,
    )
    out: list[tuple[datetime, float, float, float, float, float]] = []
    for row in rows:
        try:
            ts_s = int(row[0])
            l = float(row[1]); h = float(row[2])
            o = float(row[3]); c = float(row[4])
            v = float(row[5]) if len(row) > 5 else 0.0
        except (ValueError, IndexError, TypeError):
            continue
        bucket_start = datetime.fromtimestamp(ts_s, tz=timezone.utc)
        out.append((bucket_start, o, h, l, c, v))
    return out


def prefer_coinbase_for_5m(coin_id: str) -> bool:
    """Return True iff this coin should pull 5m bars from Coinbase
    Exchange instead of OKX.

    OKX `/api/v5/market/history-candles` empirically truncates 5m at
    ~60-100 days regardless of coin (see `COINBASE_PRODUCTS` doctring),
    well short of the 305-day hard gate in `_evaluate_5m_gate`. Coinbase
    Exchange's public `/products/<id>/candles` serves 5m back ≥335 days
    with no API key. So as of Task #603 every coin with a Coinbase
    product mapping prefers Coinbase as the canonical 5m source. Coins
    without a Coinbase listing (currently just `jupiter-exchange-solana`)
    stay on OKX with OKX's shorter window.
    """
    return coin_id in COINBASE_PRODUCTS


async def fetch_5m_smart(
    client: httpx.AsyncClient, coin_id: str, days: int,
    end_ts_ms: Optional[int] = None,
) -> tuple[
    list[tuple[datetime, float, float, float, float, float]],
    str,
]:
    """5m OHLCV fetcher that prefers Coinbase Exchange (deep history)
    where a product mapping exists, falling back to OKX otherwise.

    Returns ``(rows, source_label)`` so the caller can stamp the
    `price_candles.source` column with the actual upstream — never a
    silent fallback. ``source_label`` is one of ``"coinbase"`` or
    ``"okx"`` and matches the existing `source` enum on the writer side.
    Used by both the daily head top-up (`scheduled_5m_topup.topup_5m_head`)
    and the historical tail-extension driver (`backfill_5m_extend`).
    """
    if prefer_coinbase_for_5m(coin_id):
        rows = await fetch_coinbase_ohlcv(
            client, coin_id, "5m", days, end_ts_ms=end_ts_ms,
        )
        return rows, "coinbase"
    rows = await fetch_okx_ohlcv(
        client, coin_id, "5m", days, end_ts_ms=end_ts_ms,
    )
    return rows, "okx"


async def fetch_cmc_daily(
    client: httpx.AsyncClient, coin_id: str, days: int, api_key: str,
) -> list[tuple[datetime, float]]:
    """CMC daily OHLCV fallback (paid tier required, key already in env)."""
    slug = CMC_SLUGS.get(coin_id)
    if slug is None:
        return []
    url = f"{CMC_BASE}/v1/cryptocurrency/ohlcv/historical"
    params = {
        "slug": slug,
        "convert": "USD",
        "interval": "daily",
        "count": str(min(days, 365)),
    }
    try:
        r = await client.get(
            url, params=params, headers={"X-CMC_PRO_API_KEY": api_key}, timeout=30.0,
        )
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("cmc_request_failed coin=%s err=%s", coin_id, exc)
        return []
    body = r.json()
    if body.get("status", {}).get("error_code"):
        logger.warning(
            "cmc_error coin=%s msg=%s",
            coin_id, body["status"].get("error_message"),
        )
        return []
    quotes = body.get("data", {}).get("quotes") or []
    out: list[tuple[datetime, float]] = []
    for q in quotes:
        ts_str = q.get("time_close") or q.get("timestamp")
        usd = (q.get("quote") or {}).get("USD") or {}
        close = usd.get("close")
        if not ts_str or close is None:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            continue
        out.append((ts.astimezone(timezone.utc), float(close)))
    return out


async def insert_batch(
    pool: asyncpg.Pool, coin_id: str, points: list[tuple[datetime, float]],
    *, dry_run: bool = False, timeframe: str = "1m",
    source: str = "backfill_history",
) -> int:
    """Insert real (is_synthetic=false) rows. De-dupe by (coin_id, timestamp).

    Task #343 — `price_history` is the 1m-tick store. Any caller that
    tries to write a coarser cadence here (e.g. operator-forced
    `--target history --timeframes 5m`) is rejected at this boundary.
    Aggregated bars belong in `price_candles` via `insert_candles_batch`.
    """
    assert_native_cadence(timeframe, source, "price_history")
    if not points:
        return 0
    async with pool.acquire() as con:
        existing = await con.fetch(
            "SELECT timestamp FROM price_history "
            "WHERE coin_id=$1 AND timestamp = ANY($2::timestamptz[])",
            coin_id, [t for t, _ in points],
        )
        seen = {row["timestamp"] for row in existing}
        new_rows = [
            (coin_id, float(p), t, False) for t, p in points if t not in seen
        ]
        if not new_rows or dry_run:
            return len(new_rows)
        await con.executemany(
            "INSERT INTO price_history (coin_id, price, timestamp, is_synthetic) "
            "VALUES ($1, $2, $3, $4)",
            new_rows,
        )
        return len(new_rows)


async def insert_candles_batch(
    pool: asyncpg.Pool, coin_id: str, timeframe: str,
    candles: list[tuple[datetime, float, float, float, float, float]],
    *, source: str = "okx", dry_run: bool = False,
) -> int:
    """Insert OHLCV rows into `price_candles`. De-dupes against the
    `(coin_id, timeframe, bucket_start)` unique index via ON CONFLICT.

    Returns the count of rows actually inserted (post-conflict). Real-
    only — caller must vouch that `source` is a real provider.

    Task #343 — guard at the writer boundary. price_candles is the
    per-timeframe OHLCV store; the guard rejects any unknown cadence
    so a typo / new-coin onboarding that forgets the contract fails
    loud instead of silently writing untyped buckets.
    """
    assert_native_cadence(timeframe, f"backfill_history:{source}", "price_candles")
    if not candles:
        return 0
    if dry_run:
        # Even on dry-run we want an honest "would-be" count, so check
        # which buckets already exist and subtract them.
        async with pool.acquire() as con:
            existing = await con.fetch(
                "SELECT bucket_start FROM price_candles "
                "WHERE coin_id=$1 AND timeframe=$2 "
                "  AND bucket_start = ANY($3::timestamptz[])",
                coin_id, timeframe, [c[0] for c in candles],
            )
        seen = {row["bucket_start"] for row in existing}
        return sum(1 for c in candles if c[0] not in seen)
    rows = [
        (coin_id, timeframe, c[0], c[1], c[2], c[3], c[4], c[5], source)
        for c in candles
    ]
    async with pool.acquire() as con:
        # ON CONFLICT against the unique index keeps the operation
        # idempotent without an extra round-trip. RETURNING captures
        # only the rows that were actually inserted.
        inserted = await con.fetch(
            """
            INSERT INTO price_candles
              (coin_id, timeframe, bucket_start, open, high, low,
               close, volume, source)
            SELECT * FROM unnest(
              $1::text[], $2::text[], $3::timestamptz[],
              $4::real[],  $5::real[],  $6::real[],
              $7::real[],  $8::real[],  $9::text[]
            )
            ON CONFLICT (coin_id, timeframe, bucket_start) DO NOTHING
            RETURNING bucket_start
            """,
            [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
            [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows],
            [r[6] for r in rows], [r[7] for r in rows], [r[8] for r in rows],
        )
        return len(inserted)


async def backfill_history_one(
    client: httpx.AsyncClient, pool: asyncpg.Pool, coin_id: str, timeframe: str,
    days: int, *, dry_run: bool, cmc_key: Optional[str],
) -> dict:
    """Close-only path → `price_history`. Used for the 1m base cadence."""
    okx = await fetch_okx_candles(client, coin_id, timeframe, days)
    inserted_okx = await insert_batch(
        pool, coin_id, okx, dry_run=dry_run,
        timeframe=timeframe, source=f"backfill_history:okx[{timeframe}]",
    )
    inserted_cmc = 0
    cmc_pulled = 0
    used_cmc = False
    if timeframe == "1d" and cmc_key and len(okx) < int(0.9 * days):
        cmc = await fetch_cmc_daily(client, coin_id, days, cmc_key)
        cmc_pulled = len(cmc)
        if cmc:
            inserted_cmc = await insert_batch(
                pool, coin_id, cmc, dry_run=dry_run,
                timeframe=timeframe, source="backfill_history:cmc[1d]",
            )
            used_cmc = True
    return {
        "target": "price_history",
        "coin": coin_id, "timeframe": timeframe,
        "okx_pulled": len(okx), "okx_inserted": inserted_okx,
        "cmc_pulled": cmc_pulled, "cmc_inserted": inserted_cmc,
        "used_cmc": used_cmc,
    }


async def backfill_candles_one(
    client: httpx.AsyncClient, pool: asyncpg.Pool, coin_id: str, timeframe: str,
    days: int, *, dry_run: bool, end_ts_ms: Optional[int] = None,
    source: str = "okx",
) -> dict:
    """OHLCV path → `price_candles`. Used for 5m / 1h / 2h / 6h / 1d.

    `source` selects the venue:
      * `"okx"` (default) — primary fetch via OKX `history-candles`.
      * `"coinbase"` — Task #409 fallback for coins where OKX truncates
        the requested window short of the campaign's hard gate. Only
        coins listed in `COINBASE_PRODUCTS` are eligible.
    Rows land in `price_candles` with `source=<venue>` so the campaign's
    real-source allow-list can attribute them correctly.
    """
    if source == "coinbase":
        candles = await fetch_coinbase_ohlcv(
            client, coin_id, timeframe, days, end_ts_ms=end_ts_ms,
        )
        inserted = await insert_candles_batch(
            pool, coin_id, timeframe, candles,
            source="coinbase", dry_run=dry_run,
        )
        return {
            "target": "price_candles",
            "coin": coin_id, "timeframe": timeframe,
            "okx_pulled": 0, "okx_inserted": 0,
            "coinbase_pulled": len(candles), "coinbase_inserted": inserted,
            "cmc_pulled": 0, "cmc_inserted": 0,
            "used_cmc": False,
            "source": "coinbase",
        }
    if source != "okx":
        raise SystemExit(
            f"unknown --source {source!r} (expected 'okx' or 'coinbase')"
        )
    okx = await fetch_okx_ohlcv(client, coin_id, timeframe, days, end_ts_ms=end_ts_ms)
    inserted = await insert_candles_batch(
        pool, coin_id, timeframe, okx, source="okx", dry_run=dry_run,
    )
    return {
        "target": "price_candles",
        "coin": coin_id, "timeframe": timeframe,
        "okx_pulled": len(okx), "okx_inserted": inserted,
        "coinbase_pulled": 0, "coinbase_inserted": 0,
        "cmc_pulled": 0, "cmc_inserted": 0,
        "used_cmc": False,
        "source": "okx",
    }


# Default cadence routing — 1m is close-only ticks, everything else is
# native OHLCV bars. Caller can override via --target.
DEFAULT_HISTORY_TFS = ["1m"]
DEFAULT_CANDLES_TFS = ["5m", "1h", "2h", "6h", "1d"]


async def main_async(args) -> None:
    dsn = os.environ["DATABASE_URL"]
    cmc_key = os.environ.get("COINMARKETCAP_API_KEY")
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)
    try:
        async with httpx.AsyncClient() as client:
            results: list[dict] = []
            for coin in args.coins:
                for tf in args.timeframes:
                    days = args.days_override or DEFAULT_DAYS_BY_TF[tf]
                    if (
                        args.target == "history"
                        and tf not in DEFAULT_HISTORY_TFS
                    ):
                        # Task #343 — `--target history` for a non-1m
                        # timeframe is no longer permitted. The cadence
                        # guard inside `insert_batch` will raise
                        # `CadenceGuardError` the moment we try to write
                        # the first row. We refuse here too so the
                        # operator gets a clear message before any
                        # provider fetch fires.
                        raise SystemExit(
                            f"--target history is restricted to the 1m "
                            f"native cadence (got tf={tf}). Aggregated "
                            f"bars must go to price_candles via "
                            f"--target auto or --target candles. See "
                            f"task #343 / cadence_guard.py."
                        )
                    if args.target == "history" or (
                        args.target == "auto" and tf in DEFAULT_HISTORY_TFS
                    ):
                        res = await backfill_history_one(
                            client, pool, coin, tf, days,
                            dry_run=args.dry_run, cmc_key=cmc_key,
                        )
                    else:
                        res = await backfill_candles_one(
                            client, pool, coin, tf, days,
                            dry_run=args.dry_run,
                            end_ts_ms=getattr(args, "end_ts_ms", None),
                            source=args.source,
                        )
                    results.append(res)
                    logger.info(
                        "backfill target=%s coin=%s tf=%s days=%d "
                        "source=%s okx_pulled=%d okx_inserted=%d "
                        "coinbase_pulled=%d coinbase_inserted=%d "
                        "cmc_pulled=%d cmc_inserted=%d",
                        res["target"], coin, tf, days,
                        res.get("source", "n/a"),
                        res["okx_pulled"], res["okx_inserted"],
                        res.get("coinbase_pulled", 0),
                        res.get("coinbase_inserted", 0),
                        res["cmc_pulled"], res["cmc_inserted"],
                    )
            total_inserted = sum(
                r["okx_inserted"] + r.get("coinbase_inserted", 0)
                + r["cmc_inserted"] for r in results
            )
            logger.info(
                "backfill_done dry_run=%s total_inserted=%d",
                args.dry_run, total_inserted,
            )
    finally:
        await pool.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--coins", nargs="*", default=DEFAULT_COINS)
    p.add_argument(
        # Default routes 1m → price_history (close-only ticks) and any
        # higher cadence → price_candles (native OHLCV bars). The split
        # exists because `price_history` is a single un-typed series and
        # interleaving daily rows with 1m rows breaks the resampler in
        # features.py / labels.py. price_candles is keyed by timeframe
        # (task #317) so per-TF storage is safe.
        "--timeframes", nargs="*",
        default=DEFAULT_HISTORY_TFS + DEFAULT_CANDLES_TFS,
    )
    p.add_argument(
        "--target", choices=["auto", "candles", "history"], default="auto",
        help="auto = route by timeframe (1m → history, rest → candles); "
             "candles = force all into price_candles; "
             "history = force all into price_history.",
    )
    p.add_argument(
        "--days", dest="days_override", type=int, default=None,
        help="Override per-TF default. Otherwise uses DEFAULT_DAYS_BY_TF.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--end-ts-ms", dest="end_ts_ms", type=int, default=None,
        help="OKX `after` cursor seed (ms). When set, the first OKX "
             "request walks BACKWARD from this timestamp instead of now, "
             "letting an outer loop chain calls past the venue's 200-page "
             "single-invocation cap. Used by the Task #366 iterative 5m "
             "extension loop.",
    )
    p.add_argument(
        "--source", choices=["okx", "coinbase"], default="okx",
        help="Venue used by the candles target. 'okx' (default) is the "
             "primary; 'coinbase' is the Task #409 fallback for coins "
             "where OKX `history-candles` truncates the requested window "
             "short of the campaign's hard gate. Only coins listed in "
             "COINBASE_PRODUCTS are eligible for the coinbase fallback.",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
