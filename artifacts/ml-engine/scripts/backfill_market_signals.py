"""Task #586 — Backfill historical market_signals.

Background
==========
The api-server's `market-signals-poller` (task #271) writes per-minute
snapshots of `funding_rate`, `open_interest_usd`, `liquidations_1h_usd`,
`bid_ask_spread_bps`, and `mid_price` going forward from when the poller
was first started. The training datasets, however, span months/years of
historical candles. The asof-join in
`app.training.labels.build_labeled_dataset` therefore returns ``None`` for
nearly every historical bucket and the dataset columns
``funding_rate / open_interest_z / liquidations_1h_usd /
bid_ask_spread_bps / btc_lead_ret_5m / eth_lead_ret_5m`` fall through to
the registered safe-default of zero.

Task #580's feature-edge search rejected every cross-coin lead/lag and
liquidity/funding/OI/spread candidate because of this — the ablation
slices saw constant-zero columns, so the booster naturally learned the
features were noise. This script unblocks that bucket by pulling REAL
historical streams from the same OKX public endpoints the live poller
already uses, then writing them into ``market_signals`` with backfill
``source`` tags so the next dataset-refresh sees them via the existing
asof join (no labels.py changes required).

Streams backfilled
==================
* ``funding_rate``      — ``/api/v5/public/funding-rate-history`` (8h cadence)
* ``open_interest_usd`` — ``/api/v5/rubik/stat/contracts/open-interest-history`` (1h cadence)
* ``mid_price`` for ``btc`` / ``eth`` / ``sol`` — ``/api/v5/market/history-candles?bar=5m`` close

Streams NOT backfilled (forward-only coverage from live poller preserved)
=========================================================================
* ``liquidations_1h_usd`` — OKX ``/public/liquidation-orders`` only
  returns ~1000 most-recent records (already covered by the live poller).
  Coinglass aggregated history requires a paid API key. Forward coverage
  is preserved; older buckets remain at the registered safe default.
* ``bid_ask_spread_bps`` — historical top-of-book bid/ask is not exposed
  by any free public source. Forward coverage from the live poller is
  preserved; historical buckets remain at the registered safe default.

Idempotency
===========
For each backfill source tag the script DELETEs the existing rows in the
target window for the selected coin ids before inserting. Re-running the
script therefore replaces (rather than duplicates) the backfilled data.
The live poller's rows (with their own ``source`` labels) are NEVER
touched.

Usage
=====
    cd artifacts/ml-engine && \\
        ../../.pythonlibs/bin/python -m scripts.backfill_market_signals

Environment overrides
=====================
* ``ML_BACKFILL_LOOKBACK_DAYS``  total horizon to backfill (default: 365)
* ``ML_BACKFILL_COINS``           comma list of coin ids; default = the
                                  monitored set + lead refs
* ``ML_BACKFILL_STREAMS``         comma list of {funding,oi,mid}; default
                                  is all three
* ``ML_BACKFILL_DRY_RUN=1``       fetch only, do not write to DB

Each run stamps a manifest under ``reports/<TS>-task586-backfill-market-signals.json``.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import close_pool, init_pool  # noqa: E402

OKX_BASE = "https://www.okx.com"
HTTP_TIMEOUT = 12.0
PAGE_SLEEP_S = float(os.environ.get("ML_BACKFILL_PAGE_SLEEP_S", "0.05"))
MAX_PAGES_FUNDING = 200   # 200 pages * 100 rows * 8h = ~6.6 years (safe ceiling)
MAX_PAGES_OI = 100        # 100 pages * 100 rows * 1h = ~14 months
MAX_PAGES_CANDLES = 1500  # 1500 pages * 100 rows * 5m = ~17 months at 5m

# Mirrors api-server/src/lib/market-signals-poller.ts OKX_SWAP_BASE so
# the (coin_id, instId) mapping stays consistent between live polling and
# this backfill. Coins not listed here are silently skipped (same rule as
# the live poller).
OKX_SWAP_BASE: dict[str, str] = {
    "pepe": "PEPE",
    "floki-inu": "FLOKI",
    "bonk": "BONK",
    "dogwifcoin": "WIF",
    "render-token": "RENDER",
    "injective-protocol": "INJ",
    "sei-network": "SEI",
    "celestia": "TIA",
    "jupiter-exchange-solana": "JUP",
    "worldcoin-wld": "WLD",
    # Task #643 — BTC/ETH research targets. Funding / OI / mid rows
    # under coin_id="bitcoin" / "ethereum" so the labels_research
    # pipeline can opt into them. The labels.py self-leak guard drops
    # the matching feature columns when the TRAINING target is the
    # same coin (own funding rate predicting own future return is a
    # textbook leak). Live api-server poller is NOT updated; this is
    # ingest-only.
    "bitcoin": "BTC",
    "ethereum": "ETH",
}

# Pseudo-coin ids that store cross-market reference prices (mid_price is
# how `btc_lead_ret_5m` / `eth_lead_ret_5m` are derived in labels.py).
LEAD_REFS: list[tuple[str, str]] = [
    ("btc", "BTC"),
    ("eth", "ETH"),
    ("sol", "SOL"),
]

# `source` labels written onto backfilled rows so analytics can tell them
# apart from the live poller's rows. These are versioned so a future
# re-backfill can leave old runs in place if the operator wants.
SRC_FUNDING = "okx_backfill_funding_v1"
SRC_OI = "okx_backfill_oi_v1"
SRC_MID = "okx_backfill_mid_v1"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _selected_coins() -> list[str]:
    """Coin ids the run should process. Defaults to the live-poller's
    monitored set + the BTC/ETH/SOL lead refs.

    Setting ``ML_BACKFILL_COINS`` filters BOTH the OKX_SWAP_BASE loop and
    the LEAD_REFS loop, so an operator can chunk a run per-coin (useful
    when 1500 history-candles pages on a single lead exceed a tool
    timeout — e.g. just `ML_BACKFILL_COINS=btc ML_BACKFILL_STREAMS=mid`).
    """
    raw = os.environ.get("ML_BACKFILL_COINS")
    if raw:
        return [c.strip() for c in raw.split(",") if c.strip()]
    return list(OKX_SWAP_BASE.keys()) + [c for c, _ in LEAD_REFS]


def _selected_streams() -> set[str]:
    raw = os.environ.get("ML_BACKFILL_STREAMS")
    if raw:
        return {s.strip().lower() for s in raw.split(",") if s.strip()}
    return {"funding", "oi", "mid"}


def _lookback_days() -> int:
    raw = os.environ.get("ML_BACKFILL_LOOKBACK_DAYS")
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return 365


def _end_days_ago() -> int:
    """Number of days BEFORE `now` to anchor the right edge of the window.

    Default 0 = window ends at wall-clock now. A non-zero value lets an
    operator chunk a long backfill across several invocations:
    ``ML_BACKFILL_LOOKBACK_DAYS=180 ML_BACKFILL_END_DAYS_AGO=180`` covers
    the 180-360d range; the default run with offset=0 covers the 0-180d
    range. Each chunk owns its own deterministic source-tagged window so
    the idempotent DELETE+INSERT does not stomp the other chunk's rows.
    """
    raw = os.environ.get("ML_BACKFILL_END_DAYS_AGO")
    if raw:
        try:
            v = int(raw)
            if v >= 0:
                return v
        except ValueError:
            pass
    return 0


def _dry_run() -> bool:
    return os.environ.get("ML_BACKFILL_DRY_RUN", "").strip() in {"1", "true", "yes"}


async def _fetch_okx(
    client: httpx.AsyncClient, path: str,
) -> list:
    """Fetch ``OKX_BASE + path`` and return the ``data`` array.

    Raises on transport errors and on OKX-level error codes so the caller
    can decide how to degrade. Returns an empty list when the response
    contains no data array (e.g. unsupported instId) without raising.
    """
    r = await client.get(f"{OKX_BASE}{path}", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    body = r.json()
    code = body.get("code")
    if code is not None and code != "0":
        raise RuntimeError(f"OKX code={code} msg={body.get('msg')!r} path={path}")
    data = body.get("data")
    if not isinstance(data, list):
        return []
    return data


# --------------------------------------------------------------------------
# Funding rate history
# --------------------------------------------------------------------------

async def fetch_funding_history(
    client: httpx.AsyncClient, inst_id: str, since_ms: int,
    end_ms: Optional[int] = None,
) -> list[tuple[int, float]]:
    """Walk back the OKX funding-rate-history endpoint until ts < since_ms.

    OKX pagination semantics: ``after=<ts>`` returns records with
    ``fundingTime < ts``. We seed with ``after=end_ms+1`` (so the first
    page is anchored at end_ms instead of wall-clock now); when end_ms is
    None we use the default endpoint behaviour (newest-first).
    """
    out: list[tuple[int, float]] = []
    after: Optional[int] = end_ms + 1 if end_ms is not None else None
    pages = 0
    while pages < MAX_PAGES_FUNDING:
        path = (
            f"/api/v5/public/funding-rate-history?instId={inst_id}&limit=100"
            + (f"&after={after}" if after is not None else "")
        )
        try:
            rows = await _fetch_okx(client, path)
        except Exception as exc:
            print(f"  [{inst_id}] funding fetch failed (page {pages}): {exc}")
            break
        if not rows:
            break
        oldest_in_page: Optional[int] = None
        for r in rows:
            try:
                ts = int(r["fundingTime"])
                fr = float(r["fundingRate"])
            except (KeyError, TypeError, ValueError):
                continue
            if oldest_in_page is None or ts < oldest_in_page:
                oldest_in_page = ts
            if ts < since_ms:
                continue
            out.append((ts, fr))
        if oldest_in_page is None or oldest_in_page <= since_ms:
            break
        after = oldest_in_page
        pages += 1
        await asyncio.sleep(PAGE_SLEEP_S)
    # Deduplicate (in case pages overlapped) and sort oldest-first.
    seen: set[int] = set()
    deduped: list[tuple[int, float]] = []
    for ts, v in sorted(out):
        if ts in seen:
            continue
        seen.add(ts)
        deduped.append((ts, v))
    return deduped


# --------------------------------------------------------------------------
# Open interest history (USD-notional)
# --------------------------------------------------------------------------

async def fetch_oi_history(
    client: httpx.AsyncClient, inst_id: str, since_ms: int,
    end_ms: Optional[int] = None,
) -> list[tuple[int, float]]:
    """Walk back OKX open-interest-history (1H period) until ts < since_ms.

    Pagination here uses ``end=<ts>`` (records with ts <= end). The
    response is a 4-tuple ``[ts, oi(contracts), oiCcy(coin units),
    oiUsd]`` — we keep only the USD column so the downstream label code's
    z-score window is comparable across coins. ``end_ms`` lets the
    caller pin the right edge of the window when chunking a long
    backfill into multiple invocations.
    """
    out: list[tuple[int, float]] = []
    end: Optional[int] = end_ms
    pages = 0
    while pages < MAX_PAGES_OI:
        path = (
            f"/api/v5/rubik/stat/contracts/open-interest-history"
            f"?instId={inst_id}&period=1H&limit=100"
            + (f"&end={end}" if end is not None else "")
        )
        try:
            rows = await _fetch_okx(client, path)
        except Exception as exc:
            print(f"  [{inst_id}] OI fetch failed (page {pages}): {exc}")
            break
        if not rows:
            break
        oldest_in_page: Optional[int] = None
        for r in rows:
            if not isinstance(r, list) or len(r) < 4:
                continue
            try:
                ts = int(r[0])
                oi_usd = float(r[3])
            except (TypeError, ValueError):
                continue
            if oldest_in_page is None or ts < oldest_in_page:
                oldest_in_page = ts
            if ts < since_ms or oi_usd <= 0:
                continue
            out.append((ts, oi_usd))
        if oldest_in_page is None or oldest_in_page <= since_ms:
            break
        # Step back by 1 ms so we don't re-pull the boundary row.
        end = oldest_in_page - 1
        pages += 1
        await asyncio.sleep(PAGE_SLEEP_S)
    seen: set[int] = set()
    deduped: list[tuple[int, float]] = []
    for ts, v in sorted(out):
        if ts in seen:
            continue
        seen.add(ts)
        deduped.append((ts, v))
    return deduped


# --------------------------------------------------------------------------
# 5m mid-price history for the cross-coin lead refs (btc/eth/sol)
# --------------------------------------------------------------------------

async def fetch_mid_history(
    client: httpx.AsyncClient, inst_id: str, since_ms: int,
    end_ms: Optional[int] = None,
) -> list[tuple[int, float]]:
    """Pull 5m close prices from OKX history-candles.

    The labels pipeline computes ``btc_lead_ret_5m`` / ``eth_lead_ret_5m``
    by asof-joining mid_price snapshots over a 5m window. A 5m cadence
    is therefore the smallest cadence that meaningfully improves coverage
    while keeping the row count bounded. (1m would 5x the row count for
    no extra signal in the lead-return computation.)

    OKX pagination semantics for /market/history-candles: ``after=<ts>``
    returns candles with ts < after, oldest-first within the response.
    We treat the candle's bar OPEN ts as the asof timestamp and use the
    CLOSE price as the mid.
    """
    out: list[tuple[int, float]] = []
    after: Optional[int] = end_ms + 1 if end_ms is not None else None
    pages = 0
    while pages < MAX_PAGES_CANDLES:
        path = (
            f"/api/v5/market/history-candles?instId={inst_id}&bar=5m&limit=100"
            + (f"&after={after}" if after is not None else "")
        )
        try:
            rows = await _fetch_okx(client, path)
        except Exception as exc:
            print(f"  [{inst_id}] mid fetch failed (page {pages}): {exc}")
            break
        if not rows:
            break
        oldest_in_page: Optional[int] = None
        for r in rows:
            if not isinstance(r, list) or len(r) < 5:
                continue
            try:
                ts = int(r[0])
                close = float(r[4])
            except (TypeError, ValueError):
                continue
            if oldest_in_page is None or ts < oldest_in_page:
                oldest_in_page = ts
            if ts < since_ms or close <= 0:
                continue
            out.append((ts, close))
        if oldest_in_page is None or oldest_in_page <= since_ms:
            break
        after = oldest_in_page
        pages += 1
        await asyncio.sleep(PAGE_SLEEP_S)
    seen: set[int] = set()
    deduped: list[tuple[int, float]] = []
    for ts, v in sorted(out):
        if ts in seen:
            continue
        seen.add(ts)
        deduped.append((ts, v))
    return deduped


# --------------------------------------------------------------------------
# DB writers — idempotent per (coin_id, source, window).
# --------------------------------------------------------------------------

async def _replace_rows(
    pool, coin_id: str, source: str, since_ms: int, until_ms: int,
    rows_by_ts: dict[int, dict],
) -> int:
    """Delete then insert the backfilled rows for one (coin_id, source).

    `rows_by_ts` maps ts_ms -> dict of column overrides for that row.
    Every emitted row is stamped with `source` so re-runs of this script
    can DELETE just the rows it owns without disturbing the live poller's
    snapshots (which carry different source labels).
    """
    start = datetime.fromtimestamp(since_ms / 1000.0, tz=timezone.utc)
    end = datetime.fromtimestamp(until_ms / 1000.0, tz=timezone.utc)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                DELETE FROM market_signals
                WHERE coin_id = $1
                  AND source = $2
                  AND timestamp >= $3
                  AND timestamp <= $4
                """,
                coin_id, source, start, end,
            )
            if not rows_by_ts:
                return 0
            records = []
            for ts_ms, vals in sorted(rows_by_ts.items()):
                ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                records.append((
                    coin_id,
                    ts,
                    vals.get("funding_rate"),
                    vals.get("open_interest_usd"),
                    vals.get("liquidations_1h_usd"),
                    vals.get("bid_ask_spread_bps"),
                    vals.get("mid_price"),
                    source,
                ))
            await conn.executemany(
                """
                INSERT INTO market_signals (
                    coin_id, timestamp,
                    funding_rate, open_interest_usd, liquidations_1h_usd,
                    bid_ask_spread_bps, mid_price, source
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                """,
                records,
            )
            return len(records)


async def main() -> int:
    streams = _selected_streams()
    coins = _selected_coins()
    lookback_days = _lookback_days()
    end_days_ago = _end_days_ago()
    dry = _dry_run()

    now_ms = _now_ms()
    end_ms = now_ms - end_days_ago * 24 * 60 * 60 * 1000
    since_ms = end_ms - lookback_days * 24 * 60 * 60 * 1000

    print(
        f"task586 backfill streams={sorted(streams)} coins={coins} "
        f"lookback_days={lookback_days} end_days_ago={end_days_ago} dry_run={dry}",
    )

    pool = None
    if not dry:
        pool = await init_pool()

    manifest: dict = {
        "task": 586,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "streams": sorted(streams),
        "coins": coins,
        "lookback_days": lookback_days,
        "end_days_ago": end_days_ago,
        "since_ms": since_ms,
        "end_ms": end_ms,
        "now_ms": now_ms,
        "dry_run": dry,
        "per_coin": {},
        "anti_leak_check": {
            "future_ts_count": 0,
            "now_ms_at_check": None,
        },
    }

    lead_ids = {c for c, _ in LEAD_REFS}

    async with httpx.AsyncClient() as client:
        # Funding + OI for every monitored coin id (OKX SWAP-listed only).
        for coin_id in coins:
            base = OKX_SWAP_BASE.get(coin_id)
            if not base:
                # `btc/eth/sol` aren't in OKX_SWAP_BASE because the live
                # poller routes them via LEAD_REFERENCES; the lead loop
                # below picks them up. Anything else really is unsupported
                # — record it so the manifest tells the operator.
                if coin_id not in lead_ids:
                    print(f"  skip {coin_id}: not in OKX_SWAP_BASE")
                    manifest["per_coin"][coin_id] = {
                        "skipped": "not_in_okx_swap_base",
                    }
                continue
            inst_id = f"{base}-USDT-SWAP"
            entry: dict = {"inst_id": inst_id, "fetched": {}, "written": {}}

            if "funding" in streams:
                t0 = time.time()
                fr = await fetch_funding_history(
                    client, inst_id, since_ms, end_ms=end_ms,
                )
                entry["fetched"]["funding"] = {
                    "rows": len(fr),
                    "earliest_ms": fr[0][0] if fr else None,
                    "latest_ms": fr[-1][0] if fr else None,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if fr and not dry and pool is not None:
                    rows_by_ts = {ts: {"funding_rate": v} for ts, v in fr}
                    n = await _replace_rows(
                        pool, coin_id, SRC_FUNDING, since_ms, end_ms, rows_by_ts,
                    )
                    entry["written"]["funding"] = n

            if "oi" in streams:
                t0 = time.time()
                oi = await fetch_oi_history(
                    client, inst_id, since_ms, end_ms=end_ms,
                )
                entry["fetched"]["oi"] = {
                    "rows": len(oi),
                    "earliest_ms": oi[0][0] if oi else None,
                    "latest_ms": oi[-1][0] if oi else None,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if oi and not dry and pool is not None:
                    rows_by_ts = {ts: {"open_interest_usd": v} for ts, v in oi}
                    n = await _replace_rows(
                        pool, coin_id, SRC_OI, since_ms, end_ms, rows_by_ts,
                    )
                    entry["written"]["oi"] = n

            manifest["per_coin"][coin_id] = entry
            print(
                f"  {coin_id} ({inst_id}) funding={entry['fetched'].get('funding', {}).get('rows', 0)} "
                f"oi={entry['fetched'].get('oi', {}).get('rows', 0)}",
            )

        # Mid-price history for the cross-market lead refs (btc/eth/sol).
        if "mid" in streams:
            selected_set = set(coins)
            for coin_id, base in LEAD_REFS:
                if coin_id not in selected_set:
                    continue
                inst_id = f"{base}-USDT-SWAP"
                entry = manifest["per_coin"].setdefault(
                    coin_id, {"inst_id": inst_id, "fetched": {}, "written": {}},
                )
                t0 = time.time()
                mid = await fetch_mid_history(
                    client, inst_id, since_ms, end_ms=end_ms,
                )
                entry["fetched"]["mid"] = {
                    "rows": len(mid),
                    "earliest_ms": mid[0][0] if mid else None,
                    "latest_ms": mid[-1][0] if mid else None,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if mid and not dry and pool is not None:
                    rows_by_ts = {ts: {"mid_price": v} for ts, v in mid}
                    n = await _replace_rows(
                        pool, coin_id, SRC_MID, since_ms, end_ms, rows_by_ts,
                    )
                    entry["written"]["mid"] = n
                print(
                    f"  lead {coin_id} ({inst_id}) mid={entry['fetched'].get('mid', {}).get('rows', 0)}",
                )

    # Anti-leak: walk the in-memory manifest and confirm no row has a ts
    # later than the wall-clock `now` we recorded at start. The DB-side
    # check is a separate verification step (see report) but this catches
    # any mistake in the OKX response handling immediately.
    check_now = _now_ms()
    manifest["anti_leak_check"]["now_ms_at_check"] = check_now
    leak_count = 0
    for entry in manifest["per_coin"].values():
        for stream in ("funding", "oi", "mid"):
            f = entry.get("fetched", {}).get(stream) or {}
            latest = f.get("latest_ms")
            if latest is not None and int(latest) > check_now:
                leak_count += 1
    manifest["anti_leak_check"]["future_ts_count"] = leak_count

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{stamp}-task586-backfill-market-signals.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest -> {out_path}")

    if pool is not None:
        await close_pool()
    return 0 if leak_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
