"""Phase 1 — Backtest journal emitter.

Pushes simulated decisions from the Python backtester into the api-server's
prediction_journal table via POST /crypto/journal/backtest-batch. Every row
lands with brain="BACKTEST" and skipReason="backtest_simulation" so they are
trivially separable from live LLM/QUANT rows but share the exact same row
shape — enabling apples-to-apples replay and counterfactual analysis later.

Failures are swallowed with a logger.warning. The backtest report is the
authoritative artifact; journal emission is a side-channel and must NEVER
abort or alter the simulation.
"""
from __future__ import annotations

import logging
import os
import urllib.request
import urllib.error
import json
import datetime as dt
from typing import Iterable

import pandas as pd

from .simulator import SimulationResult, SkipRow, TradeRow

logger = logging.getLogger("ml-engine.backtest.journal")


def _api_base_url() -> str:
    return os.environ.get("API_SERVER_URL", "http://localhost:8080")


def _admin_key() -> str | None:
    return os.environ.get("ADMIN_API_KEY")


def _ms_to_iso(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000.0, tz=dt.timezone.utc).isoformat()


def _mfe_pct(t: TradeRow) -> float:
    """Maximum favourable excursion as a % of entry price."""
    if t.entry_price <= 0 or t.peak_price <= 0:
        return 0.0
    if t.direction == "up":
        return (t.peak_price - t.entry_price) / t.entry_price * 100.0
    return (t.entry_price - t.peak_price) / t.entry_price * 100.0


def _mae_pct(t: TradeRow) -> float:
    """Maximum adverse excursion as a POSITIVE magnitude.

    Live convention (paper-trader.ts:817/820): `Math.max(0, ...)` — MAE is
    always stored as a non-negative percent so cross-source learning and
    sorting work consistently. We mirror that here so backtest rows share
    the EXACT same semantics as live trades in `trade_journal.maePct`.
    """
    if t.entry_price <= 0 or t.mae_price <= 0:
        return 0.0
    if t.direction == "up":
        # Adverse for a long = price went DOWN from entry.
        return max(0.0, (t.entry_price - t.mae_price) / t.entry_price * 100.0)
    # Adverse for a short = price went UP from entry.
    return max(0.0, (t.mae_price - t.entry_price) / t.entry_price * 100.0)


def _classify_outcome(direction: str, entry_price: float, exit_price: float) -> str:
    """Map (direction, price movement) -> live outcome taxonomy
    `correct` | `wrong` | `neutral`. Mirrors shadow-recorder.ts:193 and
    backfill semantics so backtest rows participate in cross-source
    learning without bias."""
    if entry_price <= 0:
        return "neutral"
    pct = (exit_price - entry_price) / entry_price
    # 0.05% deadband for "neutral" so rounding noise doesn't force a side.
    if abs(pct) < 0.0005:
        return "neutral"
    if direction == "up":
        return "correct" if pct > 0 else "wrong"
    if direction == "down":
        return "correct" if pct < 0 else "wrong"
    return "neutral"


def _trade_to_row(t: TradeRow, model_version: str | None) -> dict:
    """Map one simulated trade to the prediction_journal row shape the
    api-server endpoint expects, AND attach a `simulatedTrade` payload so
    the api-server materialises a matching `trade_journal` row.

    Critical semantics: simulated trades MUST land with `becameTrade=true`
    on the api-server side — the presence of `simulatedTrade` is the
    signal. Without it, the row would be incorrectly classified as a skip,
    polluting trade-conversion analytics.
    """
    # IMPORTANT: live semantics — realizedReturnPct is the SIGNED price
    # movement (not direction-adjusted PnL). Live writers compute it as
    # `(actualPrice - priceAtPrediction) / priceAtPrediction * 100`. We
    # mirror that exactly so a short trade that profited still records a
    # NEGATIVE realizedReturnPct (price went down). The directional
    # judgement lives in `outcome`, not in this field.
    price_return_pct = (
        (t.exit_price - t.entry_price) / t.entry_price * 100.0
        if t.entry_price > 0
        else 0.0
    )
    pnl_pct = float(t.pnl_pct)
    return {
        "coinId": t.coin_id,
        "timeframe": t.timeframe,
        "modelId": "lightgbm",
        "modelVersion": model_version,
        "featureHash": None,
        "featureVector": None,
        "regimeLabel": t.regime_at_entry,
        "direction": t.direction,
        "confidence": float(t.confidence),
        "probUp": None,
        "probDown": None,
        "probStable": None,
        "expectedReturnPct": None,
        "predictionStdPct": None,
        "priceAtPrediction": float(t.entry_price),
        "predictedPrice": float(t.exit_price),
        "actualPrice": float(t.exit_price),
        "realizedReturnPct": price_return_pct,
        "outcome": _classify_outcome(t.direction, t.entry_price, t.exit_price),
        "resolvesAt": _ms_to_iso(t.exit_ts_ms),
        "resolvedAt": _ms_to_iso(t.exit_ts_ms),
        "gatesApplied": {"backtest": True},
        "simulatedTrade": {
            "entryTime": _ms_to_iso(t.entry_ts_ms),
            "exitTime": _ms_to_iso(t.exit_ts_ms),
            "entryPriceRaw": float(t.raw_entry_price) if t.raw_entry_price > 0 else None,
            "entryPriceAdj": float(t.entry_price),
            "exitPriceRaw": float(t.raw_exit_price) if t.raw_exit_price > 0 else None,
            "exitPriceAdj": float(t.exit_price),
            "entryFee": float(t.entry_fee),
            "exitFee": float(t.exit_fee),
            "slippagePct": float(t.slippage_pct),
            "positionSizeUsd": float(t.position_size_usd),
            "mfePct": _mfe_pct(t),
            "maePct": _mae_pct(t),
            "exitReason": t.exit_reason,
            "realizedPnlUsd": float(t.pnl_usd),
            "realizedPnlPct": pnl_pct,
        },
    }


def _skip_to_row(s: SkipRow, model_version: str | None) -> dict:
    """Map one simulated SKIP (gate-rejected prediction) to the
    prediction_journal row shape with `becameTrade=false` and a structured
    skip reason. This is essential for cross-source learning: skipped
    predictions are first-class signals about WHEN gates fired, not noise
    to be discarded.
    """
    return {
        "coinId": s.coin_id,
        "timeframe": s.timeframe,
        "modelId": "lightgbm",
        "modelVersion": model_version,
        "featureHash": None,
        "featureVector": None,
        "regimeLabel": None,
        # Direction/confidence aren't carried in SkipRow today; the gate
        # decision (skip reason) is what matters here. We mark direction
        # as "stable" to signal "no trade", consistent with how live
        # abstain rows are recorded.
        "direction": "stable",
        "confidence": 0.0,
        "probUp": None, "probDown": None, "probStable": None,
        "expectedReturnPct": None,
        "predictionStdPct": None,
        "priceAtPrediction": 0.0,
        "predictedPrice": None,
        "actualPrice": None,
        "realizedReturnPct": None,
        "outcome": "neutral",
        "resolvesAt": _ms_to_iso(s.timestamp_ms),
        "resolvedAt": _ms_to_iso(s.timestamp_ms),
        # gatesApplied carries the structured reason — same convention
        # the live `lookupGateForPrediction` writes for live skips.
        "gatesApplied": {"backtest": True, s.reason: True, "detail": s.detail or ""},
        # No simulatedTrade -> the api-server records becameTrade=false
        # with skipReason="backtest_skipped". The structured reason is in
        # gatesApplied so it's not lost.
    }


def emit_backtest_journal(
    sim: SimulationResult,
    *,
    model_version: str | None = None,
    batch_size: int = 500,
) -> int:
    """POST every trade AND every skip in `sim` to
    /crypto/journal/backtest-batch.

    Returns the total predictions_inserted reported by the api-server, or
    0 on any failure (missing admin key, no api-server, etc).
    """
    admin_key = _admin_key()
    if not admin_key:
        logger.info(
            "skipping backtest journal emission — ADMIN_API_KEY not set"
        )
        return 0

    rows = [_trade_to_row(t, model_version) for t in sim.trades]
    rows.extend(_skip_to_row(s, model_version) for s in sim.skips)
    if not rows:
        return 0

    url = f"{_api_base_url()}/api/crypto/journal/backtest-batch"
    inserted_total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        payload = json.dumps({"rows": batch}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Admin-Key": admin_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                inserted = body.get("inserted")
                # Endpoint now returns {predictionsInserted, tradesInserted}.
                if isinstance(inserted, dict):
                    inserted_total += int(inserted.get("predictionsInserted", 0))
                else:
                    inserted_total += int(inserted or 0)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as err:
            logger.warning(
                "backtest journal POST failed (batch %d-%d): %s",
                i,
                i + len(batch),
                err,
            )
            return inserted_total
    return inserted_total
