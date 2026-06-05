"""Task #628 — Backfill OKX funding-rate history into `market_signals`.

The `market_signals` table is populated minute-by-minute by the
api-server's `market-signals-poller`, but that poller only writes
"now" snapshots — i.e. the table only ever spans a few days of
history. The MTTM training labels in `app/training/labels.py` join
funding rates as-of every (coin, bucket) row in the per-coin/per-tf
feature frame, so when the lookback window is ~1 year (6h/1d slices)
the asof-join leaves `funding_rate=0` for ~75% of rows and
`liquidations_1h_usd=0` for 100% of rows. That is the data-side
reason MTTM verification gates currently fail across the 8 universe
coins (task-628.md).

OKX exposes a free, paginated funding-rate-history endpoint:

    GET /api/v5/public/funding-rate-history?instId=<inst>
        &after=<unix_ms>&limit=100

OKX pagination semantics (verified empirically — the public docs are
ambiguous): `after=<ts>` returns records with `fundingTime` STRICTLY
EARLIER than `<ts>` (older). `before=<ts>` returns records NEWER than
`<ts>`. To walk older history we pass the OLDEST `fundingTime` from
the prior page as the next `after`. The endpoint caps history at
~3 months regardless of pagination depth — that's an OKX-side limit,
not something this script can work around.

Funding cycles are every 8h on most pairs and every 4h on some, so
~91d × 3-6/day ≈ 273-546 rows per coin per backfill (~3 to 6 pages).
With 8 coins that's ~30 API calls — well under any rate limit.

This script is idempotent on `(coin_id, timestamp)` — rows already
present in `market_signals` for the exact funding bar are skipped on
insert (we use a SELECT-then-INSERT path because there is no unique
constraint on `(coin_id, timestamp)` and we don't want to add one
retroactively just for a backfiller).

Liquidations history is intentionally out of scope here:
  - OKX `/liquidation-orders` only exposes ~24h of recent fills
    (no historical pagination).
  - Coinglass aggregates exchange-wide history but requires
    `COINGLASS_API_KEY` (no key is set in this env).
  - The existing minute-cadence poller continues to fill liquidations
    going forward, so the gap closes naturally as the table ages.

Usage::

    cd artifacts/ml-engine && \\
        ../../.pythonlibs/bin/python -m scripts.backfill_funding_history

Env knobs::

    ML_FUNDING_BACKFILL_COINS=bonk,celestia,...   (default = 8 MTTM coins)
    ML_FUNDING_BACKFILL_LOOKBACK_DAYS=400         (default = 400)
    ML_FUNDING_BACKFILL_PAGE_LIMIT=100            (OKX page size, max 100)
    ML_FUNDING_BACKFILL_HTTP_TIMEOUT_S=20
    ML_FUNDING_BACKFILL_GAP_FILL_ONLY=1           (skip when a coin already
                                                   has >=THRESHOLD rows)
    ML_FUNDING_BACKFILL_GAP_THRESHOLD_ROWS=800    (gap-fill row floor)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import close_pool, init_pool  # noqa: E402

OKX_BASE = "https://www.okx.com"

# Mirrors `OKX_SWAP_BASE` in
# artifacts/api-server/src/lib/market-signals-poller.ts. Drift between
# this map and the api-server one is a correctness bug — keep them
# aligned.
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
}

# The 8 MTTM universe coins (from `DEFAULT_MTTM_UNIVERSE` in
# artifacts/api-server/src/lib/mttm.ts). Default backfill targets.
DEFAULT_MTTM_COINS: tuple[str, ...] = (
    "bonk",
    "celestia",
    "dogwifcoin",
    "floki-inu",
    "injective-protocol",
    "jupiter-exchange-solana",
    "pepe",
    "render-token",
)

# OKX caps funding-rate-history at ~92 days regardless of pagination
# depth. Setting the lookback higher just wastes API calls walking past
# the cap into empty pages — 95 leaves enough headroom for the most-
# recent boundary day plus a small buffer.
DEFAULT_LOOKBACK_DAYS = 95
DEFAULT_PAGE_LIMIT = 100
DEFAULT_HTTP_TIMEOUT_S = 20.0
DEFAULT_GAP_THRESHOLD_ROWS = 800
SOURCE_LABEL = "okx_funding_history_backfill"


def _selected_coins() -> list[str]:
    raw = os.environ.get("ML_FUNDING_BACKFILL_COINS")
    if raw:
        coins = [c.strip() for c in raw.split(",") if c.strip()]
    else:
        coins = list(DEFAULT_MTTM_COINS)
    # Filter to coins that have an OKX SWAP listing — anything else
    # cannot be backfilled and would silently no-op.
    out: list[str] = []
    for c in coins:
        if c not in OKX_SWAP_BASE:
            print(
                f"[funding-backfill] WARN  coin={c!r} has no OKX SWAP "
                "listing in OKX_SWAP_BASE — skipped",
                flush=True,
            )
            continue
        out.append(c)
    return out


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        n = int(raw)
        return n if n > 0 else default
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        f = float(raw)
        return f if f > 0 else default
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _http_get_json(url: str, timeout_s: float) -> dict:
    # OKX returns 403 for the default `Python-urllib/...` UA, so we
    # send a generic browser-shaped UA. Keep it identifiable so OKX
    # ops can trace traffic if they ever ask.
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (compatible; ReplitAgent-MTTM-Backfill/1.0; "
                "+https://replit.com)"
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (https URL)
        body = resp.read()
    return json.loads(body.decode("utf-8"))


async def _existing_funding_count(pool, coin_id: str, since_ms: int) -> int:
    row = await pool.fetchrow(
        """
        SELECT COUNT(*)::bigint AS n
        FROM market_signals
        WHERE coin_id = $1
          AND funding_rate IS NOT NULL
          AND timestamp >= to_timestamp($2 / 1000.0)
        """,
        coin_id,
        since_ms,
    )
    return int(row["n"]) if row else 0


async def _existing_timestamps(pool, coin_id: str, since_ms: int) -> set[int]:
    """Return the set of `timestamp_ms` already in `market_signals` for
    `coin_id` since `since_ms`. Used to skip duplicate rows on insert."""
    rows = await pool.fetch(
        """
        SELECT (EXTRACT(EPOCH FROM timestamp) * 1000)::bigint AS ts_ms
        FROM market_signals
        WHERE coin_id = $1
          AND timestamp >= to_timestamp($2 / 1000.0)
        """,
        coin_id,
        since_ms,
    )
    return {int(r["ts_ms"]) for r in rows}


def _fetch_funding_page(
    inst_id: str, after_ms: Optional[int], limit: int, timeout_s: float,
) -> list[dict]:
    """One page of funding-rate-history from OKX, oldest-end-cursor walk.

    OKX pagination semantics (verified empirically — the docs are
    misleading): `after=<ts>` returns records with `fundingTime`
    STRICTLY EARLIER than `<ts>` (i.e. older). `before=<ts>` returns
    records NEWER than `<ts>`. To walk older history we pass the
    OLDEST `fundingTime` from the prior page as the next `after`.
    """
    params = {"instId": inst_id, "limit": str(limit)}
    if after_ms is not None:
        params["after"] = str(after_ms)
    url = f"{OKX_BASE}/api/v5/public/funding-rate-history?" + urllib.parse.urlencode(params)
    body = _http_get_json(url, timeout_s)
    if not isinstance(body, dict) or body.get("code") != "0":
        msg = body.get("msg") if isinstance(body, dict) else "unknown"
        raise RuntimeError(f"OKX funding-rate-history error: code={body.get('code') if isinstance(body, dict) else '?'} msg={msg}")
    rows = body.get("data") or []
    out: list[dict] = []
    for r in rows:
        try:
            ts = int(r["fundingTime"])
            fr = float(r.get("realizedRate") or r.get("fundingRate"))
        except (KeyError, TypeError, ValueError):
            continue
        if ts <= 0:
            continue
        out.append({"ts_ms": ts, "funding_rate": fr})
    # Sort oldest-first for deterministic insert ordering.
    out.sort(key=lambda r: r["ts_ms"])
    return out


def _fetch_funding_history(
    coin_id: str, lookback_days: int, page_limit: int, timeout_s: float,
) -> list[dict]:
    inst_id = f"{OKX_SWAP_BASE[coin_id]}-USDT-SWAP"
    cutoff_ms = int((time.time() - lookback_days * 86400) * 1000)
    seen_ts: set[int] = set()
    rows: list[dict] = []
    cursor: Optional[int] = None  # `before=<ms>` walks older
    pages = 0
    # 13 pages of 100 rows = ~325 days of 8h funding bars per coin.
    # Cap at 25 pages to keep the backfill bounded if OKX returns
    # a longer history than expected.
    MAX_PAGES = 25
    while pages < MAX_PAGES:
        try:
            page = _fetch_funding_page(inst_id, cursor, page_limit, timeout_s)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[funding-backfill] ERROR coin={coin_id} page={pages} cursor={cursor} err={exc!r}",
                flush=True,
            )
            break
        if not page:
            break
        new_rows = 0
        oldest_in_page: Optional[int] = None
        for r in page:
            if r["ts_ms"] in seen_ts:
                continue
            if r["ts_ms"] < cutoff_ms:
                # We've walked past the lookback window — stop after
                # this page so we don't keep paginating into ancient
                # history.
                continue
            seen_ts.add(r["ts_ms"])
            rows.append(r)
            new_rows += 1
            if oldest_in_page is None or r["ts_ms"] < oldest_in_page:
                oldest_in_page = r["ts_ms"]
        pages += 1
        if oldest_in_page is None:
            # Page entirely outside the lookback window — done.
            break
        if oldest_in_page < cutoff_ms:
            # Reached the lookback boundary.
            break
        # Move cursor older for the next page.
        cursor = oldest_in_page
        if new_rows == 0:
            # Defensive: avoid an infinite loop if the API stops
            # returning new rows.
            break
    rows.sort(key=lambda r: r["ts_ms"])
    return rows


async def _backfill_coin(
    pool, coin_id: str, lookback_days: int, page_limit: int, timeout_s: float,
    gap_fill_only: bool, gap_threshold_rows: int,
) -> dict:
    inst_id = f"{OKX_SWAP_BASE[coin_id]}-USDT-SWAP"
    since_ms = int((time.time() - lookback_days * 86400) * 1000)
    existing_count = await _existing_funding_count(pool, coin_id, since_ms)
    if gap_fill_only and existing_count >= gap_threshold_rows:
        print(
            f"[funding-backfill] SKIP   coin={coin_id:>26}  existing={existing_count}  "
            f">= threshold={gap_threshold_rows}  (gap-fill mode)",
            flush=True,
        )
        return {
            "coin_id": coin_id,
            "instId": inst_id,
            "status": "skipped_gap_fill",
            "existing_funding_rows": existing_count,
            "fetched_rows": 0,
            "inserted_rows": 0,
        }
    t0 = time.monotonic()
    fetched = _fetch_funding_history(
        coin_id, lookback_days, page_limit, timeout_s,
    )
    fetch_elapsed = round(time.monotonic() - t0, 2)
    if not fetched:
        print(
            f"[funding-backfill] EMPTY  coin={coin_id:>26}  inst={inst_id}  elapsed={fetch_elapsed}s",
            flush=True,
        )
        return {
            "coin_id": coin_id,
            "instId": inst_id,
            "status": "no_rows",
            "existing_funding_rows": existing_count,
            "fetched_rows": 0,
            "inserted_rows": 0,
            "fetch_elapsed_s": fetch_elapsed,
        }
    existing_ts = await _existing_timestamps(pool, coin_id, since_ms)
    to_insert = [r for r in fetched if r["ts_ms"] not in existing_ts]
    if to_insert:
        # Build a single multi-row INSERT to keep the round-trips low.
        # We rely on `to_timestamp(epoch_ms / 1000.0)` for the
        # timestamptz cast.
        await pool.executemany(
            """
            INSERT INTO market_signals
                (coin_id, timestamp, funding_rate, source)
            VALUES ($1, to_timestamp($2 / 1000.0), $3, $4)
            """,
            [
                (coin_id, r["ts_ms"], float(r["funding_rate"]), SOURCE_LABEL)
                for r in to_insert
            ],
        )
    total_elapsed = round(time.monotonic() - t0, 2)
    span_days = round(
        (max(r["ts_ms"] for r in fetched) - min(r["ts_ms"] for r in fetched)) / 86400000.0, 1,
    ) if fetched else 0.0
    print(
        f"[funding-backfill] OK     coin={coin_id:>26}  inst={inst_id}  "
        f"existing={existing_count}  fetched={len(fetched):>4}  inserted={len(to_insert):>4}  "
        f"span={span_days}d  elapsed={total_elapsed}s",
        flush=True,
    )
    return {
        "coin_id": coin_id,
        "instId": inst_id,
        "status": "ok",
        "existing_funding_rows": existing_count,
        "fetched_rows": len(fetched),
        "inserted_rows": len(to_insert),
        "span_days": span_days,
        "fetch_elapsed_s": fetch_elapsed,
        "total_elapsed_s": total_elapsed,
    }


async def _amain() -> int:
    coins = _selected_coins()
    lookback_days = _int_env("ML_FUNDING_BACKFILL_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)
    page_limit = _int_env("ML_FUNDING_BACKFILL_PAGE_LIMIT", DEFAULT_PAGE_LIMIT)
    timeout_s = _float_env("ML_FUNDING_BACKFILL_HTTP_TIMEOUT_S", DEFAULT_HTTP_TIMEOUT_S)
    gap_fill_only = _bool_env("ML_FUNDING_BACKFILL_GAP_FILL_ONLY", False)
    gap_threshold_rows = _int_env(
        "ML_FUNDING_BACKFILL_GAP_THRESHOLD_ROWS", DEFAULT_GAP_THRESHOLD_ROWS,
    )
    started_at = time.time()
    print(
        f"[funding-backfill] start  coins={len(coins)}  lookback_days={lookback_days}  "
        f"page_limit={page_limit}  gap_fill_only={gap_fill_only}  threshold={gap_threshold_rows}",
        flush=True,
    )
    pool = await init_pool()
    rows: list[dict] = []
    try:
        for coin_id in coins:
            row = await _backfill_coin(
                pool, coin_id, lookback_days, page_limit, timeout_s,
                gap_fill_only, gap_threshold_rows,
            )
            rows.append(row)
    finally:
        await close_pool()
    summary = {
        "task": 628,
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "coin_ids": coins,
        "lookback_days": lookback_days,
        "page_limit": page_limit,
        "gap_fill_only": gap_fill_only,
        "gap_threshold_rows": gap_threshold_rows,
        "results": rows,
        "elapsed_s_total": round(time.time() - started_at, 1),
    }
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{summary['generated_at']}-task628-funding-backfill.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[funding-backfill] done   total_inserted={sum(r['inserted_rows'] for r in rows)}  "
        f"elapsed={summary['elapsed_s_total']}s  report={out_path.relative_to(ROOT)}",
        flush=True,
    )
    # Exit non-zero only if EVERY coin failed — partial coverage is
    # still useful and the next refresh tick will retry the misses.
    any_ok = any(r["status"] in {"ok", "skipped_gap_fill", "no_rows"} for r in rows)
    return 0 if any_ok else 1


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
