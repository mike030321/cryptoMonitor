"""Unit tests for the 5m source-routing helper introduced in Task #603.

Pin the predicate (every coin with a Coinbase product mapping prefers
Coinbase, JUP stays on OKX) and the smart fetcher's source-label flow
(``"coinbase"`` / ``"okx"`` returned alongside the rows so the writer
stamps the audit column with the actual upstream — never a silent
fallback).

These tests are pure-Python: they substitute the per-source fetchers
with fakes via monkeypatch so no HTTP is ever issued from CI.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backfill_history as bh  # noqa: E402


# ── prefer_coinbase_for_5m ───────────────────────────────────────────────
def test_prefer_coinbase_covers_every_coinbase_listed_coin():
    """Predicate must return True for every coin that has a Coinbase
    product mapping. Pinning the membership protects against an
    accidental delete of a mapping that would silently route a coin
    back to OKX (and re-trip the 5m gate without warning)."""
    for coin in bh.COINBASE_PRODUCTS:
        assert bh.prefer_coinbase_for_5m(coin) is True, (
            f"{coin} has a Coinbase product but predicate said False"
        )


def test_prefer_coinbase_excludes_jup_explicitly():
    """JUP is not on Coinbase Exchange with usable history (zero bars
    at 60-200d back per April 2026 probe). The predicate MUST return
    False so the smart fetcher routes JUP to OKX. If a future operator
    adds a Coinbase mapping for JUP without re-checking history, this
    test will start failing — at which point they should re-run the
    history probe before merging."""
    assert "jupiter-exchange-solana" not in bh.COINBASE_PRODUCTS
    assert bh.prefer_coinbase_for_5m("jupiter-exchange-solana") is False


def test_prefer_coinbase_returns_false_for_unknown_coin():
    """Defensive: an unmapped string must not accidentally route to
    Coinbase (which would raise inside the fetcher). Predicate is the
    cheap check; the fetcher's own ValueError is the safety net."""
    assert bh.prefer_coinbase_for_5m("not-a-real-coin") is False


def test_prefer_coinbase_covers_at_least_nine_coins():
    """Sanity floor: as of Task #603 we expect 9 of 10 monitored coins
    to route through Coinbase. If this drops below 9 something has been
    deleted from `COINBASE_PRODUCTS` without test-suite scrutiny."""
    assert len(bh.COINBASE_PRODUCTS) >= 9, (
        f"COINBASE_PRODUCTS shrank to {len(bh.COINBASE_PRODUCTS)}; "
        f"map: {bh.COINBASE_PRODUCTS}"
    )


# ── fetch_5m_smart ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fetch_5m_smart_routes_to_coinbase_when_mapped(monkeypatch):
    """For a coin in `COINBASE_PRODUCTS`, the smart fetcher must call
    `fetch_coinbase_ohlcv` (NOT the OKX path) and return the
    ``"coinbase"`` source label so the writer stamps `price_candles.source`
    accordingly."""
    calls: list[str] = []
    bar_open = datetime(2026, 4, 1, tzinfo=timezone.utc)
    fake_rows = [(bar_open, 1.0, 2.0, 0.5, 1.5, 100.0)]

    async def fake_coinbase(client, coin_id, tf, days, end_ts_ms=None):
        calls.append(f"coinbase:{coin_id}:{tf}:{days}")
        return fake_rows

    async def fake_okx(client, coin_id, tf, days, end_ts_ms=None):
        calls.append(f"okx:{coin_id}:{tf}:{days}")
        return fake_rows

    monkeypatch.setattr(bh, "fetch_coinbase_ohlcv", fake_coinbase)
    monkeypatch.setattr(bh, "fetch_okx_ohlcv", fake_okx)

    rows, source_label = await bh.fetch_5m_smart(
        client=None, coin_id="pepe", days=7,
    )

    assert rows == fake_rows
    assert source_label == "coinbase"
    assert calls == ["coinbase:pepe:5m:7"], (
        f"smart fetcher routed wrong; calls={calls}"
    )


@pytest.mark.asyncio
async def test_fetch_5m_smart_routes_to_okx_for_jup(monkeypatch):
    """JUP is not Coinbase-mapped, so the smart fetcher must call
    `fetch_okx_ohlcv` and return the ``"okx"`` source label. This is the
    canonical "fall back to OKX with shorter window" path that lets JUP
    keep some 5m coverage even when every other coin moves to Coinbase."""
    calls: list[str] = []
    fake_rows: list = []

    async def fake_coinbase(client, coin_id, tf, days, end_ts_ms=None):
        calls.append(f"coinbase:{coin_id}")
        return fake_rows

    async def fake_okx(client, coin_id, tf, days, end_ts_ms=None):
        calls.append(f"okx:{coin_id}:{tf}:{days}")
        return fake_rows

    monkeypatch.setattr(bh, "fetch_coinbase_ohlcv", fake_coinbase)
    monkeypatch.setattr(bh, "fetch_okx_ohlcv", fake_okx)

    rows, source_label = await bh.fetch_5m_smart(
        client=None, coin_id="jupiter-exchange-solana", days=7,
    )

    assert rows == fake_rows
    assert source_label == "okx"
    assert calls == ["okx:jupiter-exchange-solana:5m:7"], (
        f"smart fetcher should NOT have called Coinbase for JUP; calls={calls}"
    )


@pytest.mark.asyncio
async def test_fetch_5m_smart_propagates_end_ts_ms_to_chosen_source(
    monkeypatch,
):
    """The optional `end_ts_ms` kwarg (used by the historical tail driver
    to walk strictly older windows) must reach whichever per-source
    fetcher we chose. Otherwise the historical pull silently snaps back
    to "now" and the tail-extension never advances."""
    captured: dict = {}

    async def fake_coinbase(client, coin_id, tf, days, end_ts_ms=None):
        captured["src"] = "coinbase"
        captured["end_ts_ms"] = end_ts_ms
        return []

    async def fake_okx(client, coin_id, tf, days, end_ts_ms=None):
        captured["src"] = "okx"
        captured["end_ts_ms"] = end_ts_ms
        return []

    monkeypatch.setattr(bh, "fetch_coinbase_ohlcv", fake_coinbase)
    monkeypatch.setattr(bh, "fetch_okx_ohlcv", fake_okx)

    await bh.fetch_5m_smart(
        client=None, coin_id="bonk", days=30, end_ts_ms=1_700_000_000_000,
    )
    assert captured == {"src": "coinbase", "end_ts_ms": 1_700_000_000_000}

    await bh.fetch_5m_smart(
        client=None, coin_id="jupiter-exchange-solana", days=30,
        end_ts_ms=1_700_000_000_000,
    )
    assert captured == {"src": "okx", "end_ts_ms": 1_700_000_000_000}


# ── pytest-asyncio shim (the existing tests use a conftest that may or
# may not register asyncio mode; this fallback lets the file run
# standalone via `pytest tests/test_backfill_5m_source_routing.py`). ────
def pytest_collection_modifyitems(config, items):  # noqa: D401
    """Add the asyncio mark automatically so the suite works under both
    pytest-asyncio configurations (auto-mode and strict-mode)."""
    for item in items:
        if asyncio.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.asyncio)
