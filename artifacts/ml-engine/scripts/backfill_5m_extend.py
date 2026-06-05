"""Task #378 / #581 — rate-limited concurrent 5m OKX backfill driver.

Pulls ≥320 days of 5m OHLCV for every monitored coin in parallel, while
serializing every OKX HTTP request through a single asyncio.Lock with a
~0.12s gap (≈8.3 req/s, well under OKX's 20-req/2s public budget). The
existing scripts.backfill_history primitives are reused for the actual
fetch + idempotent insert into `price_candles`.

Idempotent: re-running fills only missing windows; never overwrites
existing rows. Safe to invoke repeatedly until the 305-day gate clears.

Cross-replica lock isolation (task #581):
  This driver takes a Postgres advisory lock with a label DISTINCT from
  the daily top-up's `_TOPUP_LOCK_LABEL`. The two services therefore
  contend on different keys and can run concurrently without one
  starving the other or both writing the same coin from the same
  upstream at the same time. Losing the historical lock returns
  cleanly so a second invocation in a different replica becomes a
  no-op rather than racing.

Wall-clock ceiling (task #581):
  Each invocation honours a 2h hard ceiling (env-overridable via
  `BACKFILL_5M_DEADLINE_SECONDS`). When the deadline trips the per-coin
  fetch loop exits at the next page boundary so the partial pull is
  flushed and the worker exits cleanly. ON CONFLICT DO NOTHING means
  re-running picks up exactly where the previous invocation stopped.

Progress journal (task #581):
  Every coin emits a `phase="5m_historical_backfill"` entry into
  `models/progress_updates.jsonl` at start and finish. The dashboard
  freshness card already reads that file, so an operator can watch the
  pull live without tailing a workflow log.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

print("startup", flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncpg  # noqa: E402
import httpx  # noqa: E402

import scripts.backfill_history as bh  # noqa: E402
from app import db as db_mod  # noqa: E402

# Task #581 — bumped to 320 (one-week headroom over the 310-day alert
# bar in `scheduled_5m_topup.ALERT_BELOW_DAYS`, which itself sits one
# day above the 305-day hard gate in `_evaluate_5m_gate`). Operators
# can override via env without touching the file.
DAYS = int(os.environ.get("BACKFILL_5M_DAYS", "320"))
GLOBAL_DELAY_SEC = float(os.environ.get("BACKFILL_5M_DELAY_SEC", "0.12"))
# 2-hour hard ceiling per workflow run (task #581). Re-running resumes.
DEADLINE_SECONDS = int(
    os.environ.get("BACKFILL_5M_DEADLINE_SECONDS", str(2 * 60 * 60))
)

# Lock label is intentionally distinct from
# `app.scheduled_5m_topup._TOPUP_LOCK_LABEL` so the historical backfill
# and the daily top-up never contend on the same advisory key.
HISTORICAL_LOCK_LABEL = "ml_engine.scheduled_5m_topup.historical_backfill"
HISTORICAL_LOCK_KEY = db_mod.advisory_lock_key(HISTORICAL_LOCK_LABEL)

PROGRESS_PATH = Path(ROOT) / "models" / "progress_updates.jsonl"

_okx_lock = asyncio.Lock()
_deadline_monotonic: float | None = None


def _deadline_reached() -> bool:
    """Return True once the 2h wall-clock ceiling has elapsed."""
    return (
        _deadline_monotonic is not None
        and time.monotonic() >= _deadline_monotonic
    )


def _append_progress(record: dict) -> None:
    """Append one structured entry to `models/progress_updates.jsonl`.

    Same surface the daily top-up uses (`5m_topup_health`) so the
    dashboard freshness card and the api-server notifier see live
    progress for the historical pull too. Best-effort: never raises.
    """
    try:
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with PROGRESS_PATH.open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 - progress is best-effort
        print(f"  progress_append_failed err={exc}", flush=True)


async def gated_fetch_raw(client, coin_id, timeframe, days, end_ts_ms=None):
    """Drop-in replacement for bh._fetch_okx_raw that adds a process-wide
    serialization lock + post-request sleep so concurrent coins don't
    blow past OKX's public 20 req/2s rate limit (without the lock the
    parallel loop trivially earns 429s)."""
    inst = bh.OKX_SYMBOLS.get(coin_id)
    if inst is None:
        raise ValueError(f"no OKX symbol mapping for {coin_id}")
    bar = bh.OKX_BAR[timeframe]
    end_ms = (
        end_ts_ms if end_ts_ms is not None
        else int(time.time() * 1000)
    )
    horizon_ms = end_ms - days * 24 * 60 * 60 * 1000
    out: list[list] = []
    cursor = end_ts_ms
    pages = 0
    while True:
        # Wall-clock guard (task #581). Stop fetching at the next page
        # boundary so the partial result is still flushed by the caller.
        if _deadline_reached():
            print(
                f"  deadline_reached coin={coin_id} pages={pages} "
                f"rows={len(out)}", flush=True,
            )
            break
        params: dict[str, str] = {
            "instId": inst, "bar": bar, "limit": "100",
        }
        if cursor is not None:
            params["after"] = str(cursor)
        async with _okx_lock:
            try:
                r = await client.get(
                    f"{bh.OKX_BASE}/history-candles",
                    params=params, timeout=30.0,
                )
                r.raise_for_status()
                ok = True
            except httpx.HTTPError as exc:
                print(f"  okx_err coin={coin_id} pages={pages} err={exc}",
                      flush=True)
                ok = False
            await asyncio.sleep(GLOBAL_DELAY_SEC)
        if not ok:
            # one back-off retry outside the immediate window
            await asyncio.sleep(2.0)
            async with _okx_lock:
                try:
                    r = await client.get(
                        f"{bh.OKX_BASE}/history-candles",
                        params=params, timeout=30.0,
                    )
                    r.raise_for_status()
                    ok = True
                except httpx.HTTPError as exc2:
                    print(
                        f"  okx_err_retry coin={coin_id} pages={pages} "
                        f"err={exc2}", flush=True,
                    )
                await asyncio.sleep(GLOBAL_DELAY_SEC)
            if not ok:
                break
        body = r.json()
        if body.get("code") != "0":
            print(f"  okx_body_err coin={coin_id} body={body}", flush=True)
            break
        rows = body.get("data") or []
        if not rows:
            print(
                f"  okx_exhausted coin={coin_id} pages={pages} "
                f"oldest_reached_ms={cursor}", flush=True,
            )
            break
        oldest_open_ms = int(rows[-1][0])
        out.extend(rows)
        pages += 1
        if pages % 50 == 0:
            print(
                f"  progress coin={coin_id} pages={pages} rows={len(out)}",
                flush=True,
            )
        if oldest_open_ms <= horizon_ms or pages > 1500:
            break
        cursor = oldest_open_ms
    return out


bh._fetch_okx_raw = gated_fetch_raw


# ── Coinbase Exchange rate-limit gate (Task #603) ────────────────────────
# `gated_fetch_raw` above protects OKX from this script's parallel
# coin-fan-out. The Coinbase path now serves 9 of 10 coins (only JUP
# stays on OKX), so it needs the same cross-coin gate or we'd peg
# Coinbase's public 10 req/s limit instantly. We monkeypatch
# `bh._fetch_coinbase_raw` the same way `_fetch_okx_raw` is patched.
_coinbase_lock = asyncio.Lock()


async def gated_coinbase_fetch_raw(
    client, coin_id, timeframe, days, end_ts_ms=None,
):
    """Drop-in replacement for bh._fetch_coinbase_raw that adds a
    process-wide serialization lock + post-request sleep so concurrent
    coins don't blow past Coinbase's public ~10 req/s rate limit.
    Logic mirrors `bh._fetch_coinbase_raw` (same paging, same row
    layout) — only the locking + per-page deadline hook differ.
    """
    product = bh.COINBASE_PRODUCTS.get(coin_id)
    if product is None:
        raise ValueError(f"no Coinbase product mapping for {coin_id}")
    granularity = bh.COINBASE_GRANULARITY.get(timeframe)
    if granularity is None:
        raise ValueError(
            f"timeframe {timeframe} not supported by Coinbase fallback "
            f"(supported: {sorted(bh.COINBASE_GRANULARITY.keys())})"
        )
    page_secs = bh._COINBASE_PAGE * granularity
    end_s = (
        end_ts_ms // 1000 if end_ts_ms is not None
        else int(time.time())
    )
    horizon_s = end_s - days * 24 * 60 * 60
    out: list[list] = []
    pages = 0
    cur_end_s = end_s
    while True:
        if _deadline_reached():
            print(
                f"  deadline_reached coin={coin_id} pages={pages} "
                f"rows={len(out)}", flush=True,
            )
            break
        cur_start_s = cur_end_s - page_secs
        params = {
            "granularity": str(granularity),
            "start": datetime.fromtimestamp(
                cur_start_s, tz=timezone.utc,
            ).isoformat(),
            "end": datetime.fromtimestamp(
                cur_end_s, tz=timezone.utc,
            ).isoformat(),
        }
        async with _coinbase_lock:
            try:
                r = await client.get(
                    f"{bh.COINBASE_BASE}/products/{product}/candles",
                    params=params, timeout=30.0,
                    headers={"User-Agent": "ml-engine-backfill/1.0"},
                )
                r.raise_for_status()
                ok = True
            except httpx.HTTPError as exc:
                print(
                    f"  coinbase_err coin={coin_id} pages={pages} "
                    f"err={exc}", flush=True,
                )
                ok = False
            await asyncio.sleep(GLOBAL_DELAY_SEC)
        if not ok:
            await asyncio.sleep(2.0)
            async with _coinbase_lock:
                try:
                    r = await client.get(
                        f"{bh.COINBASE_BASE}/products/{product}/candles",
                        params=params, timeout=30.0,
                        headers={"User-Agent": "ml-engine-backfill/1.0"},
                    )
                    r.raise_for_status()
                    ok = True
                except httpx.HTTPError as exc2:
                    print(
                        f"  coinbase_err_retry coin={coin_id} pages={pages} "
                        f"err={exc2}", flush=True,
                    )
                await asyncio.sleep(GLOBAL_DELAY_SEC)
            if not ok:
                break
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            print(
                f"  coinbase_exhausted coin={coin_id} pages={pages} "
                f"oldest_reached_s={cur_end_s}", flush=True,
            )
            break
        out.extend(rows)
        oldest_ts_s = int(rows[-1][0])
        pages += 1
        if pages % 50 == 0:
            print(
                f"  progress coin={coin_id} pages={pages} rows={len(out)} "
                f"src=coinbase",
                flush=True,
            )
        if oldest_ts_s <= horizon_s or pages > 1500:
            break
        cur_end_s = oldest_ts_s - granularity
    return out


bh._fetch_coinbase_raw = gated_coinbase_fetch_raw


async def _measure_oldest_5m(pool, coin: str):
    """Return the oldest 5m bucket_start currently in price_candles for
    this coin (or None if no rows). Used to populate the per-coin
    progress entries with before/after marks."""
    async with pool.acquire() as con:
        return await con.fetchval(
            "SELECT MIN(bucket_start) FROM price_candles "
            "WHERE coin_id=$1 AND timeframe='5m'",
            coin,
        )


async def backfill_one(client, pool, coin):
    t0 = time.time()
    before_oldest = await _measure_oldest_5m(pool, coin)
    _append_progress({
        "phase": "5m_historical_backfill",
        "status": "start",
        "coin": coin,
        "days_requested": DAYS,
        "before_oldest_bucket": (
            before_oldest.isoformat() if before_oldest is not None else None
        ),
    })
    try:
        # Task #603 — use the smart fetcher so each coin pulls from
        # whichever upstream serves the deepest 5m history (Coinbase
        # for 9 of 10 coins, OKX for JUP). Source label flows into
        # both the writer and the progress journal so an operator can
        # audit who served what without grepping logs.
        candles, source_label = await bh.fetch_5m_smart(client, coin, DAYS)
        inserted = await bh.insert_candles_batch(
            pool, coin, "5m", candles, source=source_label, dry_run=False,
        )
        # Surface the actual oldest bucket we reached so the operator
        # can see how far back the chosen upstream actually serves.
        oldest_pulled = min((c[0] for c in candles), default=None)
        after_oldest = await _measure_oldest_5m(pool, coin)
        elapsed = round(time.time() - t0, 1)
        print(
            f"DONE coin={coin} src={source_label} pulled={len(candles)} "
            f"inserted={inserted} oldest_5m_bucket={oldest_pulled} "
            f"elapsed={elapsed}s",
            flush=True,
        )
        _append_progress({
            "phase": "5m_historical_backfill",
            "status": "ok",
            "coin": coin,
            "source": source_label,
            "pulled": len(candles),
            "inserted": int(inserted),
            "oldest_pulled_bucket": (
                oldest_pulled.isoformat() if oldest_pulled is not None
                else None
            ),
            "before_oldest_bucket": (
                before_oldest.isoformat() if before_oldest is not None
                else None
            ),
            "after_oldest_bucket": (
                after_oldest.isoformat() if after_oldest is not None
                else None
            ),
            "elapsed_s": elapsed,
            "deadline_reached": _deadline_reached(),
        })
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        print(
            f"FAIL coin={coin} err={type(exc).__name__}:{exc}",
            flush=True,
        )
        _append_progress({
            "phase": "5m_historical_backfill",
            "status": "fail",
            "coin": coin,
            "error": f"{type(exc).__name__}:{exc}",
            "elapsed_s": elapsed,
            "deadline_reached": _deadline_reached(),
        })


async def main():
    global _deadline_monotonic
    print("main_start", flush=True)
    print(
        f"config days={DAYS} delay={GLOBAL_DELAY_SEC}s "
        f"deadline={DEADLINE_SECONDS}s lock_label={HISTORICAL_LOCK_LABEL}",
        flush=True,
    )
    # Cross-replica advisory lock (task #581). A second invocation
    # while one is still running becomes a clean no-op instead of
    # double-pulling.
    async with db_mod.try_advisory_lock(HISTORICAL_LOCK_KEY) as got:
        if not got:
            print(
                f"ABORT another worker holds {HISTORICAL_LOCK_LABEL}",
                flush=True,
            )
            _append_progress({
                "phase": "5m_historical_backfill",
                "status": "skipped_locked",
                "lock_label": HISTORICAL_LOCK_LABEL,
            })
            return
        _deadline_monotonic = time.monotonic() + DEADLINE_SECONDS
        _append_progress({
            "phase": "5m_historical_backfill",
            "status": "run_start",
            "days_requested": DAYS,
            "deadline_seconds": DEADLINE_SECONDS,
            "coins": list(bh.OKX_SYMBOLS),
        })
        dsn = os.environ["DATABASE_URL"]
        pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=12)
        print("pool_ready", flush=True)
        try:
            async with httpx.AsyncClient() as client:
                print(
                    f"launching coins={list(bh.OKX_SYMBOLS)}", flush=True,
                )
                tasks = [
                    backfill_one(client, pool, c) for c in bh.OKX_SYMBOLS
                ]
                await asyncio.gather(*tasks)
        finally:
            await pool.close()
        _append_progress({
            "phase": "5m_historical_backfill",
            "status": "run_done",
            "deadline_reached": _deadline_reached(),
        })
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
