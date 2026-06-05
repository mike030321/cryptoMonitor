"""Manual chronological replay of post-#444 prediction_journal rows
through the supervisory Meta-Brain, persisting trust / regime /
episodic state to a sandbox directory by default.

Three-name convention used throughout the codebase:
  * Quant Brain   — the LightGBM `/ml/predict` model that authors
    every paper trade. The sole price-direction predictor.
  * Meta-model    — the bounded supervisory governance layer (the
    vendored `market_meta_brain` package). Shapes trust / sizing /
    suppression. Never authors trades.
  * Meta-Brain    — colloquial name for the Meta-model + the api-
    server adapter that wires telemetry into it. Synonymous with
    "supervisory layer" in older docs.

This script touches *only* the Meta-model state. It never writes to
the database, never replays through the Quant Brain, and never
triggers a real or paper trade.

Run modes
─────────
sandbox (default):
    python scripts/replay_meta_brain.py
    Writes manifest, replay.jsonl, and holdout metrics under
      .local/cleanup/meta-brain-replay/<run_id>/

commit (only if thresholds met):
    python scripts/replay_meta_brain.py --commit
    Promotes the sandbox state into
      artifacts/ml-engine/models/meta_brain_state/
    Refuses to commit unless `--min-trades` (default 2000),
    `--min-days` (default 30), AND ≥ 3 distinct dominant regimes
    are all satisfied.

Required outputs
────────────────
Every run emits, under `<sandbox_dir>/`:
  * manifest.json   — see build_manifest() for the exact schema
  * replay.jsonl    — one JSON line per evaluated cycle
  * state/          — trust_model.json, regime_memory.json,
                      episodic_memory.json (full buffer)
  * holdout/metrics.json — avg reward, defensive-mode hit rate,
                      family-trust calibration error on the held
                      tail of cycles.

Allow-listed sources
────────────────────
Column references are pinned at scripts/dataset-columns.json.
SQL_COLUMNS below mirrors that allow-list and is enforced by the
contract test test_replay_meta_brain_no_leakage.py.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import hashlib
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import uuid
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_ENGINE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SRC = ML_ENGINE_ROOT / "vendor" / "market_meta_brain" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from market_meta_brain.domain.types import (  # noqa: E402
    GovernanceOutcome,
    PortfolioTelemetry,
    QuantSliceTelemetry,
    TelemetryBatch,
)
from market_meta_brain.runtime.checkpointing import Checkpointer  # noqa: E402
from market_meta_brain.runtime.service import MarketMetaBrainService  # noqa: E402

# ─────────────────────── constants & guards ────────────────────────

CUTOFF = datetime(2026, 4, 21, 0, 0, 0, tzinfo=timezone.utc)
"""Post-#444 brain-flip cutoff. Mirrors PREDICTION_FLEET_RESET_AT in
artifacts/api-server/src/lib/trading-constants.ts. Anything earlier
was authored under the deleted LLM ensemble and must not contaminate
the deterministic Meta-Brain."""

DEFAULT_WINDOW_S = 30
"""Monitor cycle period (artifacts/api-server/src/lib/monitor.ts)."""

DEFAULT_HOLDOUT_PCT = 0.20
"""Tail fraction withheld from learning. Spec calls for last 20%."""

DEFAULT_MIN_TRADES = 2000
DEFAULT_MIN_DAYS = 30
DEFAULT_MIN_REGIMES = 3
"""Three thresholds that must ALL be satisfied before --commit
promotes sandbox state into the canonical path. Below any one of
them the run is honest about being `pipeline_validation_only`."""

FORBIDDEN_PREFIXES = ("news_", "llm_", "gpt_", "sentiment_", "ai_")
"""Mirrors FORBIDDEN_FEATURE_PREFIXES in app/training/registry.py.
Any feature key matching one of these is dropped before reaching
the Meta-model and recorded under `forbidden_columns_seen` so the
contract test fails loudly."""

DEFAULT_SLIPPAGE_BPS = 5.0
"""Mirrors SLIPPAGE_PCT (0.0005) in api-server. Used when a slice
has no direct slippage measurement."""

ALLOWED_FAMILIES = (
    "momentum",
    "mean_reversion",
    "breakout",
    "volatility_forecaster",
    "baseline",
)

CANONICAL_STATE_DIR = ML_ENGINE_ROOT / "models" / "meta_brain_state"
SANDBOX_ROOT = REPO_ROOT / ".local" / "cleanup" / "meta-brain-replay"
ALLOW_LIST_PATH = ML_ENGINE_ROOT / "scripts" / "dataset-columns.json"

# ────────────────────────── SQL allow-list ─────────────────────────

# Single source of truth for every column actually projected by a SELECT
# in this script. The contract test enforces:
#   * each entry is a subset of the corresponding allow-list bucket in
#     dataset-columns.json,
#   * no entry carries a forbidden prefix,
#   * the SQL strings below reference only listed columns (regex
#     crosscheck).
# Aliases (`x AS y`) are listed under their underlying name.
SQL_COLUMNS: dict[str, list[str]] = {
    "prediction_journal": [
        "id", "created_at", "agent_id", "agent_name", "coin_id",
        "timeframe", "regime_label", "direction", "confidence",
        "raw_confidence", "prob_up", "prob_down", "prob_stable",
        "expected_return_pct", "prediction_std_pct",
        "price_at_prediction", "predicted_price", "feature_vector",
        "became_trade", "trade_id", "resolved_at", "actual_price",
        "realized_return_pct", "outcome", "shadow", "brain",
    ],
    "paper_trades": [
        "id", "agent_id", "agent_name", "coin_id", "coin_name",
        "action", "entry_price", "exit_price", "position_size",
        "quantity", "pnl", "pnl_percent", "entry_fee", "created_at",
        "closed_at", "status", "timeframe", "prediction_id",
    ],
    "paper_positions": [
        "id", "agent_id", "coin_id", "direction", "entry_price",
        "quantity", "position_size", "timeframe", "trade_id",
        "created_at", "peak_price", "entry_regime_label",
    ],
    "paper_position_marks": [
        "trade_id", "mark_price", "marked_at",
    ],
    "paper_portfolios": [
        "agent_id", "agent_name", "total_value", "cash_balance",
        "peak_value", "day_start_value",
    ],
    "agents": ["id", "name", "personality"],
    "strategy_snapshots": [
        "id", "strategy_type", "equity", "cash_balance",
        "invested_value", "timestamp",
    ],
    "market_signals": [
        "coin_id", "timestamp", "funding_rate", "open_interest_usd",
        "liquidations_1h_usd", "bid_ask_spread_bps", "mid_price",
    ],
    "price_candles": [
        "coin_id", "timeframe", "bucket_start", "open", "high",
        "low", "close", "volume",
    ],
}


logger = logging.getLogger("replay_meta_brain")


# ───────────────────────── allow-list ──────────────────────────────


def load_allow_list() -> dict[str, Any]:
    with ALLOW_LIST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_allow_list_full(allow_list: dict[str, Any]) -> dict[str, Any]:
    """Echo the FULL pinned dataset-columns.json contract into the
    manifest so the schema test can assert equality. This makes the
    manifest self-describing: anything not in this dict is, by
    contract, NOT used by the replay."""
    return {
        "_doc": (
            "Full pinned column allow-list copied verbatim from "
            "scripts/dataset-columns.json. The contract test "
            "(test_replay_meta_brain_no_leakage) enforces equality "
            "with that file and that no SQL string in the script "
            "references a column outside this set."
        ),
        "source_path": str(ALLOW_LIST_PATH.relative_to(REPO_ROOT)),
        **allow_list,
    }


def is_forbidden(key: str) -> bool:
    lk = key.lower()
    return any(lk.startswith(p) for p in FORBIDDEN_PREFIXES)


# ───────────────────────── family mapping ──────────────────────────


def resolve_strategy_family(personality: str | None) -> str:
    """Mirror of resolveStrategyFamily() in api-server adapter.ts.
    Bounded to the same five families the Meta-model recognises so a
    replay-derived directive shapes the same buckets the live path
    shapes.
    """
    p = (personality or "").lower()
    if "momentum" in p or "trend" in p:
        return "momentum"
    if "contrarian" in p or "reversion" in p or "revert" in p or "mean" in p:
        return "mean_reversion"
    if "breakout" in p:
        return "breakout"
    if "scalper" in p or "vol" in p:
        return "volatility_forecaster"
    return "baseline"


# ─────────────────────────── DB loading ────────────────────────────


async def fetch_journal_rows(
    conn: asyncpg.Connection, *, start: datetime, end: datetime
) -> list[asyncpg.Record]:
    return list(
        await conn.fetch(
            """
            SELECT id, created_at, agent_id, agent_name, coin_id, timeframe,
                   regime_label, direction, confidence, raw_confidence,
                   prob_up, prob_down, prob_stable, expected_return_pct,
                   prediction_std_pct, price_at_prediction, predicted_price,
                   feature_vector, became_trade, trade_id, resolved_at,
                   actual_price, realized_return_pct, outcome, shadow
            FROM prediction_journal
            WHERE created_at >= $1 AND created_at < $2
              AND brain = 'QUANT'
              AND COALESCE(shadow, false) = false
            ORDER BY created_at ASC, id ASC
            """,
            start,
            end,
        )
    )


async def fetch_closed_trades(
    conn: asyncpg.Connection, *, start: datetime, end: datetime
) -> list[asyncpg.Record]:
    """Closed paper_trades whose open AND close fall within the
    replay window. The schema stores the open time in `created_at`
    and the trade direction in `action` (`buy`/`sell`). Trades that
    opened before the cutoff or remain open at `end` are excluded —
    partial outcomes have no honest derivation."""
    return list(
        await conn.fetch(
            """
            SELECT id, agent_id, agent_name, coin_id, coin_name,
                   action, entry_price, exit_price,
                   position_size, quantity, pnl, pnl_percent, entry_fee,
                   created_at, closed_at, status, timeframe,
                   prediction_id
            FROM paper_trades
            WHERE created_at >= $1
              AND closed_at IS NOT NULL
              AND closed_at < $2
              AND status NOT IN ('open', 'pending')
            ORDER BY created_at ASC, id ASC
            """,
            start,
            end,
        )
    )


async def fetch_agents(conn: asyncpg.Connection) -> dict[int, dict[str, Any]]:
    rows = await conn.fetch("SELECT id, name, personality FROM agents")
    return {
        r["id"]: {
            "name": r["name"],
            "personality": r["personality"],
            "family": resolve_strategy_family(r["personality"]),
        }
        for r in rows
    }


async def fetch_portfolio_snapshots(
    conn: asyncpg.Connection,
) -> dict[int, dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT agent_id, agent_name, total_value, cash_balance,
               peak_value, day_start_value
        FROM paper_portfolios
        """
    )
    return {r["agent_id"]: dict(r) for r in rows}


async def fetch_strategy_snapshots(
    conn: asyncpg.Connection, *, start: datetime, end: datetime
) -> list[asyncpg.Record]:
    return list(
        await conn.fetch(
            """
            SELECT id, strategy_type, equity, cash_balance,
                   invested_value, timestamp
            FROM strategy_snapshots
            WHERE timestamp >= $1 AND timestamp < $2
            ORDER BY timestamp ASC
            """,
            start,
            end,
        )
    )


async def fetch_market_signals(
    conn: asyncpg.Connection, *, start: datetime, end: datetime
) -> list[asyncpg.Record]:
    return list(
        await conn.fetch(
            """
            SELECT coin_id, timestamp, funding_rate, open_interest_usd,
                   liquidations_1h_usd, bid_ask_spread_bps, mid_price
            FROM market_signals
            WHERE timestamp >= $1 AND timestamp < $2
            ORDER BY timestamp ASC
            """,
            start,
            end,
        )
    )


async def fetch_price_candles(
    conn: asyncpg.Connection,
    *,
    start: datetime,
    end: datetime,
    coin_ids: list[str],
) -> list[asyncpg.Record]:
    if not coin_ids:
        return []
    return list(
        await conn.fetch(
            """
            SELECT coin_id, timeframe, bucket_start, open, high,
                   low, close, volume
            FROM price_candles
            WHERE coin_id = ANY($1::text[])
              AND timeframe = '5m'
              AND bucket_start >= $2
              AND bucket_start < $3
            ORDER BY coin_id, bucket_start
            """,
            coin_ids,
            start,
            end,
        )
    )


async def fetch_position_marks(
    conn: asyncpg.Connection,
    *,
    start: datetime,
    end: datetime,
    trade_ids: list[int],
) -> list[asyncpg.Record]:
    """Task #491 — per-tick mark history written by the api-server's
    `updatePortfolioValues` loop. Restricted to trades that already
    appear in the replay window so a busy table doesn't blow the
    fetch up. Bound by `marked_at` for safety even though we filter
    by `trade_id`. Returns an empty list when there are no closed
    trades in the window — older runs (pre-#491) simply degrade to
    the previous price_candles-based MAE."""
    if not trade_ids:
        return []
    try:
        return list(
            await conn.fetch(
                """
                SELECT trade_id, mark_price, marked_at
                FROM paper_position_marks
                WHERE trade_id = ANY($1::int[])
                  AND marked_at >= $2
                  AND marked_at < $3
                ORDER BY trade_id, marked_at
                """,
                trade_ids,
                start,
                end,
            )
        )
    except asyncpg.UndefinedTableError:
        # Replaying a window that pre-dates the migration on a DB
        # snapshot that hasn't been migrated yet — degrade silently
        # to the candle-based path. Logged once at WARNING so the
        # operator notices but the replay still produces a manifest.
        logger.warning(
            "paper_position_marks table missing — replay falling back "
            "to candle-based MAE/stability for this run"
        )
        return []


async def fetch_positions(
    conn: asyncpg.Connection,
) -> list[asyncpg.Record]:
    """Open paper_positions. For a closed-trade replay these are
    informational only (open positions reflect the live cutover, not
    the replay window) but keep the call so the dataset surface and
    the contract test stay consistent with the spec."""
    return list(
        await conn.fetch(
            """
            SELECT id, agent_id, coin_id, direction, entry_price,
                   quantity, position_size, timeframe, trade_id,
                   created_at, peak_price, entry_regime_label
            FROM paper_positions
            """
        )
    )


# ─────────────────────── cycle reconstruction ──────────────────────


def cluster_into_cycles(
    rows: list[asyncpg.Record], window_s: int
) -> list[list[asyncpg.Record]]:
    if not rows:
        return []
    cycles: list[list[asyncpg.Record]] = [[rows[0]]]
    for prev, curr in zip(rows, rows[1:]):
        gap = (curr["created_at"] - prev["created_at"]).total_seconds()
        if gap > window_s:
            cycles.append([curr])
        else:
            cycles[-1].append(curr)
    return cycles


# ───────────────────────── helpers ─────────────────────────────────


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _disagreement(prob_up: Any, prob_down: Any) -> float:
    pu = _safe_float(prob_up)
    pd = _safe_float(prob_down)
    return _clamp01(1.0 - abs(pu - pd))


def _filter_features(
    feature_vector: Any, *, allowed_keys: set[str], forbidden_seen: set[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(feature_vector, dict):
        return out
    for k, v in feature_vector.items():
        if is_forbidden(k):
            forbidden_seen.add(k)
            continue
        if k not in allowed_keys:
            continue
        f = _safe_float(v, default=math.nan)
        if math.isfinite(f):
            out[k] = f
    return out


def _latest_at_or_before(
    sorted_index: list[datetime], records: list[Any], ts: datetime
) -> Any | None:
    """Binary search the most recent record whose timestamp ≤ ts."""
    if not sorted_index:
        return None
    pos = bisect.bisect_right(sorted_index, ts) - 1
    if pos < 0:
        return None
    return records[pos]


# ─────────────────────── slice construction ────────────────────────


def compute_recent_accuracy(
    journal_rows: list[dict[str, Any]],
    *,
    cycle_ts: datetime,
    window: int = 25,
) -> float | None:
    """Rolling directional accuracy for the (agent, coin, timeframe)
    derived from the last `window` resolved journal rows STRICTLY
    BEFORE `cycle_ts`. Uses the journal's `outcome` enum
    (`correct` / `wrong` / `neutral`); returns `correct / (correct +
    wrong)` so neutrals don't drag the ratio. Returns None if there
    are no resolved rows yet — the caller flags `missing:recent_accuracy`
    and the slice degrades gracefully (the trust model treats absent
    history as neutral)."""
    if not journal_rows:
        return None
    relevant = [r for r in journal_rows if r["created_at"] < cycle_ts and r.get("outcome") in ("correct", "wrong", "neutral")]
    if not relevant:
        return None
    tail = relevant[-window:]
    correct = sum(1 for r in tail if r["outcome"] == "correct")
    wrong = sum(1 for r in tail if r["outcome"] == "wrong")
    if correct + wrong == 0:
        return None
    return correct / (correct + wrong)


def build_slice(
    row: Any,
    *,
    family: str,
    portfolio_snap: dict[str, Any] | None,
    market_signal: dict[str, Any] | None,
    recent_accuracy: float | None,
    allowed_feature_keys: set[str],
    forbidden_seen: set[str],
) -> QuantSliceTelemetry:
    flags: list[str] = []
    feats = _filter_features(
        row["feature_vector"],
        allowed_keys=allowed_feature_keys,
        forbidden_seen=forbidden_seen,
    )
    confidence = _clamp01(_safe_float(row["confidence"]))
    edge = _safe_float(row["expected_return_pct"]) / 100.0
    risk_score = _clamp01(1.0 - confidence)
    realized = (
        _safe_float(row["realized_return_pct"]) / 100.0
        if row["realized_return_pct"] is not None
        else None
    )
    pred_pct = _safe_float(row["expected_return_pct"]) / 100.0
    if realized is None:
        prediction_error = 0.0
        flags.append("missing:prediction_error")
    else:
        prediction_error = abs(pred_pct - realized)
    pnl_state = 0.0
    drawdown_state = 0.0
    exposure = 0.0
    if portfolio_snap is not None:
        total = _safe_float(portfolio_snap.get("total_value"))
        peak = _safe_float(portfolio_snap.get("peak_value"), default=total)
        day_start = _safe_float(
            portfolio_snap.get("day_start_value"), default=total
        )
        if day_start > 0:
            pnl_state = (total - day_start) / day_start
        if peak > 0:
            drawdown_state = max(0.0, (peak - total) / peak)
        if total > 0:
            cash = _safe_float(portfolio_snap.get("cash_balance"))
            exposure = _clamp01(max(0.0, (total - cash) / total))
    else:
        flags.extend(
            ["missing:pnl_state", "missing:drawdown_state", "missing:exposure"]
        )
    volatility = _safe_float(feats.get("realizedVol"))
    if "realizedVol" not in feats:
        flags.append("missing:volatility")
    # Real per-coin signal enrichment (Task #467 review): correlation_shift
    # proxied by the contemporaneous funding rate (positive = crowded
    # longs, negative = crowded shorts) and slippage from the live
    # bid_ask_spread when present, falling back to the SLIPPAGE_PCT
    # default the api-server adapter uses when the venue is silent.
    correlation_shift = 0.0
    slippage_bps = DEFAULT_SLIPPAGE_BPS
    if market_signal is not None:
        funding = market_signal.get("funding_rate")
        if funding is not None:
            correlation_shift = float(funding)
        spread = market_signal.get("bid_ask_spread_bps")
        if spread is not None and float(spread) > 0:
            slippage_bps = float(spread)
        else:
            flags.append("missing:slippage_bps")
        if funding is None:
            flags.append("missing:correlation_shift")
    else:
        flags.extend(["missing:correlation_shift", "missing:slippage_bps"])
    if recent_accuracy is None:
        flags.append("missing:recent_accuracy")
        recent_accuracy_value = 0.5
    else:
        recent_accuracy_value = _clamp01(float(recent_accuracy))
    return QuantSliceTelemetry(
        coin=row["coin_id"],
        timeframe=row["timeframe"],
        strategy_family=family,
        edge=edge,
        confidence=confidence,
        calibrated_confidence=confidence,
        risk_score=risk_score,
        recent_accuracy=recent_accuracy_value,
        pnl_state=pnl_state,
        drawdown_state=drawdown_state,
        disagreement=_disagreement(row["prob_up"], row["prob_down"]),
        prediction_error=prediction_error,
        regime=row["regime_label"] or "unknown",
        volatility=volatility,
        correlation_shift=correlation_shift,
        exposure=exposure,
        turnover=0.0,
        slippage_bps=slippage_bps,
        anomaly_flags=flags,
    )


def aggregate_portfolio(
    *,
    cycle_ts: datetime,
    snapshot_index: list[datetime],
    snapshot_records: list[dict[str, Any]],
    cycle_signals: list[dict[str, Any]],
    fallback_portfolios: list[dict[str, Any]],
) -> PortfolioTelemetry:
    """Cycle-level portfolio telemetry. Sourced from the latest
    per-strategy `strategy_snapshots` row at or before `cycle_ts`
    (real time-series of equity / cash / invested_value), enriched
    with the cycle's contemporaneous market_signals for liquidity
    stress, and falling back to the per-agent paper_portfolios
    aggregate only when no strategy snapshot exists yet.
    """
    snap = _latest_at_or_before(snapshot_index, snapshot_records, cycle_ts)
    flags: list[str] = []
    if snap is not None:
        equity = _safe_float(snap.get("equity"))
        cash = _safe_float(snap.get("cash_balance"))
        invested = _safe_float(snap.get("invested_value"))
        # The replay does not retain a rolling peak across snapshots
        # (would require re-deriving from the full strategy_snapshots
        # series). Approximate fleet drawdown from invested vs equity:
        # an equity drop with high invested → real drawdown signal.
        drawdown = max(0.0, 1.0 - (equity / max(1.0, equity + abs(invested))))
        invested_share = _clamp01(invested / equity) if equity > 0 else 0.0
        flags.append("approx:total_drawdown_from_invested_share")
    else:
        # Pre-snapshot fallback: aggregate paper_portfolios so
        # downstream telemetry isn't structurally zero.
        if not fallback_portfolios:
            return PortfolioTelemetry(
                total_drawdown=0.0,
                realized_vol=0.0,
                concentration=0.0,
                leverage=0.0,
                liquidity_stress=0.0,
                correlation_shift=0.0,
                active_risk_budget=1.0,
                kill_switch_distance=1.0,
                anomaly_flags=["missing:portfolio"],
            )
        totals = [_safe_float(p.get("total_value")) for p in fallback_portfolios]
        peaks = [
            _safe_float(p.get("peak_value"), default=t)
            for p, t in zip(fallback_portfolios, totals)
        ]
        cashes = [_safe_float(p.get("cash_balance")) for p in fallback_portfolios]
        ft = sum(totals)
        fp = sum(peaks)
        fc = sum(cashes)
        drawdown = max(0.0, (fp - ft) / fp) if fp > 0 else 0.0
        invested_share = max(0.0, (ft - fc) / ft) if ft > 0 else 0.0
        flags.append("fallback:paper_portfolios")
    # Liquidity stress: mean(bid_ask_spread_bps) across the cycle's
    # observed signals, normalised to [0, 1] using a 30 bps reference.
    spreads = [
        float(s["bid_ask_spread_bps"])
        for s in cycle_signals
        if s.get("bid_ask_spread_bps") is not None
    ]
    liquidity_stress = _clamp01((statistics.mean(spreads) / 30.0)) if spreads else 0.0
    if not spreads:
        flags.append("missing:liquidity_stress")
    # Correlation shift: signed mean of funding_rates across cycle
    # signals — non-zero indicates the venue is crowded one way.
    fundings = [
        float(s["funding_rate"])
        for s in cycle_signals
        if s.get("funding_rate") is not None
    ]
    correlation_shift = statistics.mean(fundings) if fundings else 0.0
    if not fundings:
        flags.append("missing:correlation_shift")
    return PortfolioTelemetry(
        total_drawdown=drawdown,
        realized_vol=0.0,
        concentration=invested_share,
        leverage=invested_share,
        liquidity_stress=liquidity_stress,
        correlation_shift=correlation_shift,
        active_risk_budget=max(0.0, 1.0 - invested_share),
        kill_switch_distance=max(0.0, 1.0 - drawdown),
        anomaly_flags=flags,
    )


# ─────────────────────── outcome derivation ────────────────────────


def _mae_from_marks(
    *, entry: float, direction: str, marks: list[dict[str, Any]]
) -> float | None:
    """Task #491 — true intra-trade max-adverse-excursion from the
    per-tick mark stream. Returns None if the marks are unusable so
    the caller falls back to the candle-based path."""
    if not marks or entry <= 0 or direction not in ("buy", "sell"):
        return None
    prices = [
        _safe_float(m["mark_price"]) for m in marks if m.get("mark_price") is not None
    ]
    if not prices:
        return None
    if direction == "buy":
        return max(0.0, (entry - min(prices)) / entry)
    return max(0.0, (max(prices) - entry) / entry)


def _stability_from_marks(marks: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Mark-to-mark return stdev → bounded stability score.
    Returns (stability, sigma) or (None, None) when there are too
    few marks (≥ 2 returns required)."""
    if not marks:
        return None, None
    prices = [
        _safe_float(m["mark_price"]) for m in marks if m.get("mark_price") is not None
    ]
    rets = [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
        if prices[i - 1] > 0
    ]
    if len(rets) < 2:
        return None, None
    sigma = statistics.pstdev(rets)
    return _clamp01(1.0 / (1.0 + 5.0 * sigma)), sigma


def derive_outcome(
    trade: Any,
    *,
    coin_candles: list[dict[str, Any]] | None,
    journal_during_hold: list[dict[str, Any]] | None,
    position_marks: list[dict[str, Any]] | None = None,
) -> tuple[GovernanceOutcome, dict[str, Any]]:
    """Honest outcome reconstruction from a closed paper_trade plus
    its contemporaneous price_candles, journal rows, and (Task #491)
    per-tick `paper_position_marks`.

    realized_pnl       — pnl_percent / 100 (post-fee, post-slippage
                         in the trade-math path)
    realized_drawdown  — true intra-trade max-adverse-excursion.
                         Source preference (highest fidelity first):
                           1. paper_position_marks  (~15 s cadence)
                           2. price_candles low/high (5 m cadence)
                           3. fallback to max(0, -pnl_pct)
    realized_stability — 1 / (1 + 5*stdev) of consecutive returns
                         during the hold (bounded [0, 1]).
                         Same source preference as MAE.
    turnover_cost      — fees / position_size (fees doubled to cover
                         the symmetric exit fee the schema doesn't
                         store)
    action_churn       — count of consecutive direction flips in the
                         (agent, coin, timeframe) journal stream
                         during the hold

    The three counterfactual fields stay None — they require a
    what-if simulator we do not have.
    """
    derivation: dict[str, Any] = {}
    pnl_pct = _safe_float(trade["pnl_percent"]) / 100.0
    entry = _safe_float(trade["entry_price"])
    direction = (trade.get("action") if isinstance(trade, dict) else trade["action"]) or ""
    direction = direction.lower()

    realized_drawdown: float
    mark_mae = (
        _mae_from_marks(entry=entry, direction=direction, marks=position_marks)
        if position_marks
        else None
    )
    if mark_mae is not None:
        realized_drawdown = mark_mae
        derivation["mae_source"] = "position_marks"
        derivation["mae_marks_count"] = len(position_marks or [])
    elif coin_candles and entry > 0:
        if direction == "buy":
            min_low = min(_safe_float(c["low"]) for c in coin_candles if c.get("low") is not None)
            realized_drawdown = max(0.0, (entry - min_low) / entry)
            derivation["mae_source"] = "price_candles_low"
        elif direction == "sell":
            max_high = max(_safe_float(c["high"]) for c in coin_candles if c.get("high") is not None)
            realized_drawdown = max(0.0, (max_high - entry) / entry)
            derivation["mae_source"] = "price_candles_high"
        else:
            realized_drawdown = max(0.0, -pnl_pct)
            derivation["mae_source"] = "fallback_pnl_pct"
    else:
        realized_drawdown = max(0.0, -pnl_pct)
        derivation["mae_source"] = "fallback_pnl_pct_no_candles"

    mark_stability, mark_sigma = _stability_from_marks(position_marks or [])
    if mark_stability is not None:
        realized_stability = mark_stability
        derivation["stability_source"] = "position_marks_returns_stdev"
        derivation["stability_sigma"] = mark_sigma
        derivation["stability_marks_count"] = len(position_marks or [])
    elif coin_candles and len(coin_candles) >= 2:
        closes = [_safe_float(c["close"]) for c in coin_candles if c.get("close") is not None]
        rets = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if len(rets) >= 2:
            sigma = statistics.pstdev(rets)
            realized_stability = _clamp01(1.0 / (1.0 + 5.0 * sigma))
            derivation["stability_source"] = "price_candles_close_stdev"
            derivation["stability_sigma"] = sigma
        else:
            realized_stability = 0.5
            derivation["stability_source"] = "fallback_neutral_insufficient_returns"
    else:
        realized_stability = 0.5
        derivation["stability_source"] = "fallback_neutral_no_candles"

    position_size = _safe_float(trade["position_size"])
    fees = 2.0 * _safe_float(trade["entry_fee"])
    turnover_cost = (fees / position_size) if position_size > 0 else 0.0

    action_churn = 0.0
    if journal_during_hold:
        flips = 0
        prev = None
        for r in journal_during_hold:
            d = (r.get("direction") or "").lower()
            if prev is not None and d and d != prev:
                flips += 1
            if d:
                prev = d
        action_churn = float(flips)
        derivation["churn_source"] = "journal_direction_flips"
        derivation["churn_journal_rows"] = len(journal_during_hold)
    else:
        derivation["churn_source"] = "no_journal_rows_in_hold"

    return (
        GovernanceOutcome(
            realized_pnl=pnl_pct,
            realized_drawdown=realized_drawdown,
            realized_stability=realized_stability,
            turnover_cost=turnover_cost,
            action_churn=action_churn,
            correct_defense=None,
            correct_suppression=None,
            missed_edge_cost=None,
        ),
        derivation,
    )


# ───────────────────── replay engine ────────────────────────────────


class ReplayEngine:
    def __init__(
        self,
        *,
        service: MarketMetaBrainService,
        agents: dict[int, dict[str, Any]],
        portfolio_by_agent: dict[int, dict[str, Any]],
        snapshot_index: list[datetime],
        snapshot_records: list[dict[str, Any]],
        signals_by_coin_index: dict[str, list[datetime]],
        signals_by_coin_records: dict[str, list[dict[str, Any]]],
        candles_by_coin: dict[str, list[dict[str, Any]]],
        journal_by_key: dict[tuple[int, str, str], list[dict[str, Any]]],
        marks_by_trade: dict[int, list[dict[str, Any]]],
        allowed_feature_keys: set[str],
        window_s: int,
        replay_log: Any,
    ):
        self.service = service
        self.agents = agents
        self.portfolio_by_agent = portfolio_by_agent
        self.snapshot_index = snapshot_index
        self.snapshot_records = snapshot_records
        self.signals_by_coin_index = signals_by_coin_index
        self.signals_by_coin_records = signals_by_coin_records
        self.candles_by_coin = candles_by_coin
        self.journal_by_key = journal_by_key
        # Task #491 — per-trade mark stream (15s cadence) for honest
        # MAE / stability. Empty when the table is missing or the
        # window pre-dates the migration; replay degrades gracefully
        # to the candle-based path in that case.
        self.marks_by_trade = marks_by_trade
        self.allowed_feature_keys = allowed_feature_keys
        self.window_s = window_s
        self.replay_log = replay_log
        self.forbidden_seen: set[str] = set()
        self.regimes_observed: dict[str, int] = {}
        self.families_observed: dict[str, int] = {}
        self.cycles_replayed = 0
        self.holdout_cycles = 0
        self.trades_attributed = 0
        self.trades_unmatched = 0
        self.trades_skipped_holdout = 0
        # Holdout scoring buffers (filled during holdout pass)
        self.holdout_records: list[dict[str, Any]] = []
        self._directive_cache: deque[tuple[datetime, Any, str]] = deque(maxlen=4096)

    def _bind_trade_to_cycle(self, opened_at: datetime) -> tuple[Any, str] | None:
        match: tuple[Any, str] | None = None
        for ts, directive, iso in self._directive_cache:
            if ts <= opened_at:
                match = (directive, iso)
            else:
                break
        return match

    def _cycle_signal_for_coin(
        self, coin_id: str, ts: datetime
    ) -> dict[str, Any] | None:
        idx = self.signals_by_coin_index.get(coin_id)
        recs = self.signals_by_coin_records.get(coin_id)
        if not idx or not recs:
            return None
        return _latest_at_or_before(idx, recs, ts)

    def _cycle_signals_window(
        self, ts: datetime
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for coin_id, idx in self.signals_by_coin_index.items():
            rec = _latest_at_or_before(idx, self.signals_by_coin_records[coin_id], ts)
            if rec is not None:
                out.append(rec)
        return out

    def step_cycle(
        self, cycle_rows: list[Any], *, holdout: bool
    ) -> tuple[Any, str]:
        cycle_ts = cycle_rows[-1]["created_at"]
        cycle_iso = cycle_ts.astimezone(timezone.utc).isoformat()
        slices: list[QuantSliceTelemetry] = []
        for row in cycle_rows:
            agent_meta = self.agents.get(row["agent_id"], {})
            family = agent_meta.get("family") or "baseline"
            self.families_observed[family] = self.families_observed.get(family, 0) + 1
            regime_label = row["regime_label"] or "unknown"
            self.regimes_observed[regime_label] = self.regimes_observed.get(regime_label, 0) + 1
            portfolio_snap = self.portfolio_by_agent.get(row["agent_id"])
            market_signal = self._cycle_signal_for_coin(row["coin_id"], cycle_ts)
            recent_acc = compute_recent_accuracy(
                self.journal_by_key.get(
                    (row["agent_id"], row["coin_id"], row["timeframe"]),
                    [],
                ),
                cycle_ts=cycle_ts,
            )
            slices.append(
                build_slice(
                    row,
                    family=family,
                    portfolio_snap=portfolio_snap,
                    market_signal=market_signal,
                    recent_accuracy=recent_acc,
                    allowed_feature_keys=self.allowed_feature_keys,
                    forbidden_seen=self.forbidden_seen,
                )
            )
        portfolio = aggregate_portfolio(
            cycle_ts=cycle_ts,
            snapshot_index=self.snapshot_index,
            snapshot_records=self.snapshot_records,
            cycle_signals=self._cycle_signals_window(cycle_ts),
            fallback_portfolios=list(self.portfolio_by_agent.values()),
        )
        batch = TelemetryBatch(slices=slices, portfolio=portfolio, timestamp=cycle_iso)
        directive = self.service.evaluate(batch)
        self.cycles_replayed += 1
        if holdout:
            self.holdout_cycles += 1
            # Capture per-cycle scoring features for the holdout
            # metrics file (avg reward proxy, defensive-mode hit rate,
            # trust-vs-realised calibration error).
            self.holdout_records.append(
                {
                    "ts": cycle_iso,
                    "dominant_regime": directive.meta_state.dominant_regime
                    if directive.meta_state
                    else "unknown",
                    "caution_level": directive.caution_level,
                    "defensive_mode": directive.defensive_mode,
                    "suppressed_count": len(directive.suppressed_families),
                    "trust_map": dict(directive.trust_multiplier),
                }
            )
        else:
            self._directive_cache.append((cycle_ts, directive, cycle_iso))
        # Per-cycle replay log line — ALWAYS written so the operator
        # can audit both training and holdout paths.
        self.replay_log.write(
            json.dumps(
                {
                    "phase": "holdout" if holdout else "train",
                    "cycle_ts": cycle_iso,
                    "slice_count": len(slices),
                    "dominant_regime": directive.meta_state.dominant_regime
                    if directive.meta_state
                    else "unknown",
                    "caution_level": directive.caution_level,
                    "defensive_mode": directive.defensive_mode,
                    "exploration_budget": directive.exploration_budget,
                    "suppressed_families": list(directive.suppressed_families),
                    "trust_multiplier": dict(directive.trust_multiplier),
                    "reason_codes": list(directive.reason_codes),
                    "portfolio_anomaly_flags": list(portfolio.anomaly_flags),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        return directive, cycle_iso

    def step_trade(self, trade: Any) -> bool:
        opened_at: datetime = trade["opened_at"]
        bound = self._bind_trade_to_cycle(opened_at)
        if bound is None:
            self.trades_unmatched += 1
            return False
        directive, cycle_iso = bound
        # Pull in-window candles + journal for honest derivation
        candles_all = self.candles_by_coin.get(trade["coin_id"], [])
        coin_candles = [
            c
            for c in candles_all
            if trade["opened_at"] <= c["bucket_start"] <= trade["closed_at"]
        ]
        journal_key = (
            trade["agent_id"],
            trade["coin_id"],
            trade["timeframe"],
        )
        all_journal = self.journal_by_key.get(journal_key, [])
        journal_during_hold = [
            r
            for r in all_journal
            if trade["opened_at"] <= r["created_at"] <= trade["closed_at"]
        ]
        # Task #491 — narrow per-trade marks to the hold window so a
        # late mid-flight schema bug or stray row outside the trade
        # boundary can never poison the MAE.
        all_marks = self.marks_by_trade.get(int(trade["id"]), [])
        marks_during_hold = [
            m
            for m in all_marks
            if trade["opened_at"] <= m["marked_at"] <= trade["closed_at"]
        ]
        outcome, derivation = derive_outcome(
            trade,
            coin_candles=coin_candles,
            journal_during_hold=journal_during_hold,
            position_marks=marks_during_hold,
        )
        try:
            self.service.record_outcome(
                directive,
                outcome,
                timestamp=trade["closed_at"]
                .astimezone(timezone.utc)
                .isoformat(),
            )
        except ValueError as exc:
            logger.warning("record_outcome skipped: %s", exc)
            self.trades_unmatched += 1
            return False
        self.trades_attributed += 1
        self.replay_log.write(
            json.dumps(
                {
                    "phase": "outcome",
                    "trade_id": trade["id"],
                    "agent_id": trade["agent_id"],
                    "coin_id": trade["coin_id"],
                    "owning_cycle_ts": cycle_iso,
                    "closed_at": trade["closed_at"]
                    .astimezone(timezone.utc)
                    .isoformat(),
                    "realized_pnl": outcome.realized_pnl,
                    "realized_drawdown": outcome.realized_drawdown,
                    "realized_stability": outcome.realized_stability,
                    "action_churn": outcome.action_churn,
                    "turnover_cost": outcome.turnover_cost,
                    "derivation": derivation,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        return True


# ───────────────────────── persistence ─────────────────────────────


def state_hashes(state_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not state_dir.exists():
        return out
    for path in sorted(state_dir.glob("*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[path.name] = digest
    return out


def write_state(service: MarketMetaBrainService, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    cp = Checkpointer(state_dir)
    cp.save_json(
        "trust_model",
        {
            fam: {
                "trust": fts.trust,
                "stability": fts.stability,
                "exploration_eligibility": fts.exploration_eligibility,
                "failure_streak": fts.failure_streak,
                "recovery_score": fts.recovery_score,
                "last_regime": fts.last_regime,
            }
            for fam, fts in service.trust_model.trust_by_family.items()
        },
    )
    cp.save_json("regime_memory", service.regime_memory.state_dict())
    cp.save_json("episodic_memory", service.episodic_memory.state_dict())


def commit_to_canonical(sandbox_state: Path, *, dry_run: bool) -> dict[str, Any]:
    target = CANONICAL_STATE_DIR
    actions: dict[str, Any] = {"target": str(target), "files": [], "backup": None}
    if dry_run:
        actions["dry_run"] = True
        return actions
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = target.with_name(f"{target.name}.bak.{ts}")
        shutil.copytree(target, backup)
        actions["backup"] = str(backup)
    for src in sandbox_state.glob("*.json"):
        shutil.copy2(src, target / src.name)
        actions["files"].append(src.name)
    return actions


# ──────────────────────── holdout metrics ──────────────────────────


def compute_holdout_metrics(
    *,
    holdout_records: list[dict[str, Any]],
    final_trust_state: dict[str, dict[str, float]],
    holdout_dir: Path,
) -> dict[str, Any]:
    """Score the held-out tail. Per Task #467 §holdout: avg reward
    proxy, defensive-mode hit rate, family-trust calibration error.

    "Reward proxy" — without per-cycle realized P&L we approximate
    cycle reward by inverting caution_level (high caution = the brain
    expected adverse conditions; low caution = expected favourable
    conditions). This is honest about being a proxy.

    "Defensive-mode hit rate" — share of holdout cycles where the
    brain entered hard or soft defensive mode.

    "Family-trust calibration error" — for each family, compare the
    final trust value to the holdout-window mean of that family's
    trust_multiplier in the issued directives. Large gaps mean the
    trust state diverged from what the directive policy actually used.
    """
    holdout_dir.mkdir(parents=True, exist_ok=True)
    if not holdout_records:
        payload: dict[str, Any] = {
            "cycle_count": 0,
            "avg_reward_proxy": None,
            "defensive_mode_hit_rate": None,
            "family_trust_calibration_error": {},
            "notes": "no holdout cycles — increase --holdout-pct or run a longer window",
        }
        (holdout_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return payload
    avg_reward = statistics.mean(
        1.0 - _clamp01(r["caution_level"]) for r in holdout_records
    )
    defensive_hits = sum(
        1
        for r in holdout_records
        if r["defensive_mode"] in ("soft", "hard")
    )
    defensive_rate = defensive_hits / len(holdout_records)
    family_trust_window: dict[str, list[float]] = defaultdict(list)
    for r in holdout_records:
        for fam, mult in r["trust_map"].items():
            family_trust_window[fam].append(float(mult))
    calibration: dict[str, dict[str, float]] = {}
    for fam, observed in family_trust_window.items():
        mean_observed = statistics.mean(observed) if observed else 0.0
        final = final_trust_state.get(fam, {}).get("trust", 0.0)
        calibration[fam] = {
            "final_trust": round(float(final), 6),
            "holdout_mean_trust_multiplier": round(mean_observed, 6),
            "absolute_error": round(abs(final - mean_observed), 6),
            "samples": len(observed),
        }
    payload = {
        "cycle_count": len(holdout_records),
        "avg_reward_proxy": round(avg_reward, 6),
        "defensive_mode_hit_rate": round(defensive_rate, 6),
        "family_trust_calibration_error": calibration,
        "notes": "reward_proxy = mean(1 - caution_level); calibration = |final_trust - mean(trust_multiplier_during_holdout)|",
    }
    (holdout_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


# ──────────────────────── manifest writer ──────────────────────────


def build_manifest(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    args: argparse.Namespace,
    columns_used: dict[str, Any],
    forbidden_seen: list[str],
    source_counts: dict[str, int],
    engine: ReplayEngine,
    pre_hashes: dict[str, str],
    post_hashes: dict[str, str],
    warmed_hashes: dict[str, str],
    sandbox_state: Path,
    commit_details: dict[str, Any],
    state_label: str,
    final_trust_state: dict[str, dict[str, float]],
    final_trust_state_per_regime: dict[str, dict[str, float]],
    regime_prototype_count: int,
    episode_buffer_size: int,
    data_window: dict[str, Any],
    holdout_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Manifest schema follows Task #467 §manifest contract:

    * `data_window`     → {min_ts, max_ts, days_covered}
    * `row_counts`      → canonical counters: journal_rows_consumed,
                           cycles_replayed, outcomes_recorded,
                           outcomes_skipped_no_match,
                           outcomes_skipped_holdout, plus per-source
                           input row counts
    * `regimes_observed`→ {regime: count} dict (per-regime evidence)
    * `families_observed`→ {family: count} dict
    * `commit`          → boolean (true iff sandbox was promoted to
                           the canonical state path); details live in
                           `commit_details`
    * `columns_used`    → the FULL pinned allow-list from
                           dataset-columns.json — equality contract
    * `state`           → `production_ready` | `pipeline_validation_only`
    """
    canonical_row_counts = {
        "journal_rows_consumed": source_counts["prediction_journal_rows"],
        "cycles_replayed": engine.cycles_replayed,
        "outcomes_recorded": engine.trades_attributed,
        "outcomes_skipped_no_match": engine.trades_unmatched,
        "outcomes_skipped_holdout": engine.trades_skipped_holdout,
        "input_sources": source_counts,
    }
    return {
        "run_id": run_id,
        "task": "467",
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "finished_at": finished_at.astimezone(timezone.utc).isoformat(),
        "cutoff_utc": CUTOFF.isoformat(),
        "data_window": data_window,
        "window_seconds": args.window_s,
        "holdout_pct": args.holdout_pct,
        "replay_mode": "no_counterfactuals",
        "state": state_label,
        "columns_used": columns_used,
        "forbidden_columns_seen": sorted(forbidden_seen),
        "row_counts": canonical_row_counts,
        "regimes_observed": dict(engine.regimes_observed),
        "families_observed": dict(engine.families_observed),
        "cycles_replayed": engine.cycles_replayed,
        "holdout_cycle_count": engine.holdout_cycles,
        "trades_attributed": engine.trades_attributed,
        "trades_unmatched": engine.trades_unmatched,
        "trades_skipped_holdout": engine.trades_skipped_holdout,
        "final_trust_state": final_trust_state,
        "final_trust_state_per_regime": final_trust_state_per_regime,
        "regime_prototype_count": regime_prototype_count,
        "episode_buffer_size": episode_buffer_size,
        "holdout_metrics": holdout_metrics,
        "brain_state": {
            "pre_hashes": pre_hashes,
            "warmed_hashes_after_train_pass": warmed_hashes,
            "post_hashes": post_hashes,
            "sandbox_dir": str(sandbox_state),
        },
        "commit": bool(commit_details.get("promoted")),
        "commit_details": commit_details,
        "constants": {
            "PREDICTION_FLEET_RESET_AT": CUTOFF.isoformat(),
            "monitor_cycle_seconds": DEFAULT_WINDOW_S,
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "allowed_families": list(ALLOWED_FAMILIES),
            "min_trades_required_for_commit": args.min_trades,
            "min_days_required_for_commit": args.min_days,
            "min_regimes_required_for_commit": args.min_regimes,
        },
    }


# ──────────────────────────── main ─────────────────────────────────


def parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--start", default=CUTOFF.isoformat())
    p.add_argument("--end", default=None)
    p.add_argument("--window-s", type=int, default=DEFAULT_WINDOW_S)
    p.add_argument("--holdout-pct", type=float, default=DEFAULT_HOLDOUT_PCT)
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    p.add_argument("--min-days", type=int, default=DEFAULT_MIN_DAYS)
    p.add_argument("--min-regimes", type=int, default=DEFAULT_MIN_REGIMES)
    p.add_argument("--commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--sandbox-dir", default=None)
    return p.parse_args(argv)


def _index_signals(
    rows: list[asyncpg.Record],
) -> tuple[dict[str, list[datetime]], dict[str, list[dict[str, Any]]]]:
    by_coin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_coin[r["coin_id"]].append(dict(r))
    idx: dict[str, list[datetime]] = {}
    for coin, items in by_coin.items():
        items.sort(key=lambda x: x["timestamp"])
        idx[coin] = [x["timestamp"] for x in items]
    return idx, by_coin


def _index_candles(
    rows: list[asyncpg.Record],
) -> dict[str, list[dict[str, Any]]]:
    by_coin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_coin[r["coin_id"]].append(dict(r))
    for items in by_coin.values():
        items.sort(key=lambda x: x["bucket_start"])
    return by_coin


def _index_journal_by_key(
    rows: list[asyncpg.Record],
) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    out: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["agent_id"], r["coin_id"], r["timeframe"])
        out[key].append(dict(r))
    for items in out.values():
        items.sort(key=lambda x: x["created_at"])
    return out


async def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    run_id = args.run_id or f"replay-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    sandbox_root = Path(args.sandbox_dir) if args.sandbox_dir else SANDBOX_ROOT
    sandbox_dir = sandbox_root / run_id
    sandbox_state = sandbox_dir / "state"
    holdout_dir = sandbox_dir / "holdout"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    start = parse_iso(args.start)
    end = parse_iso(args.end) if args.end else datetime.now(timezone.utc)
    if start < CUTOFF:
        raise SystemExit(
            f"refusing to replay across the post-#444 cutoff ({CUTOFF.isoformat()})"
        )
    if end <= start:
        raise SystemExit("--end must be strictly after --start")
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL is not set")
    allow_list = load_allow_list()
    allowed_feature_keys = set(allow_list["feature_vector_keys"])
    conn = await asyncpg.connect(db_url)
    try:
        journal_rows = await fetch_journal_rows(conn, start=start, end=end)
        closed_trades = await fetch_closed_trades(conn, start=start, end=end)
        agents = await fetch_agents(conn)
        portfolios = await fetch_portfolio_snapshots(conn)
        snapshots = await fetch_strategy_snapshots(conn, start=start, end=end)
        signals = await fetch_market_signals(conn, start=start, end=end)
        coin_ids = sorted({t["coin_id"] for t in closed_trades})
        candles = await fetch_price_candles(
            conn, start=start, end=end, coin_ids=coin_ids
        )
        positions = await fetch_positions(conn)
        # Task #491 — fetch the per-tick mark stream for trades in
        # the window. Filter by trade_id (already known from
        # closed_trades) so the query is bounded even though the
        # table can be large in steady state.
        trade_ids = [int(t["id"]) for t in closed_trades if t.get("id") is not None]
        position_marks = await fetch_position_marks(
            conn, start=start, end=end, trade_ids=trade_ids
        )
    finally:
        await conn.close()
    # Build indices for fast lookups
    snapshot_records = [dict(r) for r in snapshots]
    snapshot_records.sort(key=lambda x: x["timestamp"])
    snapshot_index = [r["timestamp"] for r in snapshot_records]
    signals_index, signals_by_coin = _index_signals(signals)
    candles_by_coin = _index_candles(candles)
    journal_by_key = _index_journal_by_key(journal_rows)
    # Task #491 — bucket marks by trade_id once so step_trade is O(1).
    marks_by_trade: dict[int, list[dict[str, Any]]] = {}
    for r in position_marks:
        marks_by_trade.setdefault(int(r["trade_id"]), []).append(dict(r))
    cycles = cluster_into_cycles(journal_rows, args.window_s)
    if args.max_cycles is not None:
        cycles = cycles[: args.max_cycles]
    holdout_count = int(round(len(cycles) * args.holdout_pct))
    train_cycles = cycles[: len(cycles) - holdout_count]
    holdout_cycles = cycles[len(cycles) - holdout_count :] if holdout_count > 0 else []
    # Holdout boundary: any trade closing at or after this timestamp
    # MUST NOT call record_outcome — doing so would leak holdout data
    # into trust / regime / episodic state and invalidate the
    # holdout/metrics.json file as out-of-sample evidence.
    holdout_start_ts = (
        holdout_cycles[0][0]["created_at"] if holdout_cycles else None
    )
    service = MarketMetaBrainService()
    pre_hashes = state_hashes(sandbox_state)
    replay_log_path = sandbox_dir / "replay.jsonl"

    def _wrap(rec: asyncpg.Record) -> dict[str, Any]:
        return {
            "id": rec["id"],
            "agent_id": rec["agent_id"],
            "coin_id": rec["coin_id"],
            "action": rec["action"],
            "entry_price": rec["entry_price"],
            "pnl_percent": rec["pnl_percent"],
            "entry_fee": rec["entry_fee"],
            "position_size": rec["position_size"],
            "timeframe": rec["timeframe"],
            "opened_at": rec["created_at"],
            "closed_at": rec["closed_at"],
        }

    with replay_log_path.open("w", encoding="utf-8") as replay_log:
        engine = ReplayEngine(
            service=service,
            agents=agents,
            portfolio_by_agent=portfolios,
            snapshot_index=snapshot_index,
            snapshot_records=snapshot_records,
            signals_by_coin_index=signals_index,
            signals_by_coin_records=signals_by_coin,
            candles_by_coin=candles_by_coin,
            journal_by_key=journal_by_key,
            marks_by_trade=marks_by_trade,
            allowed_feature_keys=allowed_feature_keys,
            window_s=args.window_s,
            replay_log=replay_log,
        )
        trade_iter = iter(closed_trades)
        pending_trade = next(trade_iter, None)
        # Train pass: drain trades that close before each train cycle
        # boundary, but ONLY if they close before the holdout boundary
        # — anything closing in the holdout window is held back.
        for cycle in train_cycles:
            cycle_ts = cycle[-1]["created_at"]
            while pending_trade is not None and pending_trade["closed_at"] <= cycle_ts:
                if (
                    holdout_start_ts is not None
                    and pending_trade["closed_at"] >= holdout_start_ts
                ):
                    break
                engine.step_trade(_wrap(pending_trade))
                pending_trade = next(trade_iter, None)
            engine.step_cycle(cycle, holdout=False)
        # Final drain of pre-holdout trades (those that closed after
        # the last train cycle but BEFORE the holdout window starts).
        while pending_trade is not None and (
            holdout_start_ts is None
            or pending_trade["closed_at"] < holdout_start_ts
        ):
            engine.step_trade(_wrap(pending_trade))
            pending_trade = next(trade_iter, None)
        # Snapshot the warmed brain state right before the holdout
        # pass so the manifest can record the exact boundary.
        warmed_hashes_dir = sandbox_dir / "_warmed_state_snapshot"
        write_state(service, warmed_hashes_dir)
        warmed_hashes = state_hashes(warmed_hashes_dir)
        # Holdout pass: evaluate (read-only) but never record_outcome.
        for cycle in holdout_cycles:
            engine.step_cycle(cycle, holdout=True)
        # Account for trades whose closed_at fell inside the holdout
        # window — they are deliberately skipped to keep the holdout
        # honest. Tag them in the replay log for audit.
        while pending_trade is not None:
            engine.trades_skipped_holdout += 1
            replay_log.write(
                json.dumps(
                    {
                        "phase": "outcome_skipped_holdout",
                        "trade_id": pending_trade["id"],
                        "closed_at": pending_trade["closed_at"]
                        .astimezone(timezone.utc)
                        .isoformat(),
                        "reason": "trade_closed_in_holdout_window",
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            pending_trade = next(trade_iter, None)
        shutil.rmtree(warmed_hashes_dir, ignore_errors=True)
    write_state(service, sandbox_state)
    post_hashes = state_hashes(sandbox_state)

    span_days = (end - start).total_seconds() / 86400.0
    distinct_regimes = len(engine.regimes_observed)
    thresholds_ok = (
        engine.trades_attributed >= args.min_trades
        and span_days >= args.min_days
        and distinct_regimes >= args.min_regimes
    )
    state_label = "production_ready" if thresholds_ok else "pipeline_validation_only"
    final_trust_state = {
        fam: {
            "trust": round(fts.trust, 6),
            "stability": round(fts.stability, 6),
            "exploration_eligibility": round(fts.exploration_eligibility, 6),
            "failure_streak": fts.failure_streak,
            "recovery_score": round(fts.recovery_score, 6),
            "last_regime": fts.last_regime,
        }
        for fam, fts in service.trust_model.trust_by_family.items()
    }
    final_trust_state_per_regime = {
        fam: {"trust": round(fts.trust, 6), "regime": fts.last_regime}
        for fam, fts in service.trust_model.trust_by_family.items()
    }
    holdout_metrics = compute_holdout_metrics(
        holdout_records=engine.holdout_records,
        final_trust_state=final_trust_state,
        holdout_dir=holdout_dir,
    )
    commit_payload: dict[str, Any] = {
        "requested": bool(args.commit),
        "thresholds": {
            "trades_attributed": engine.trades_attributed,
            "min_trades": args.min_trades,
            "span_days": round(span_days, 4),
            "min_days": args.min_days,
            "distinct_regimes": distinct_regimes,
            "min_regimes": args.min_regimes,
        },
        "thresholds_satisfied": thresholds_ok,
        "promoted": False,
        "target": str(CANONICAL_STATE_DIR),
    }
    if args.commit and thresholds_ok:
        commit_payload.update(
            {"promoted": True, **commit_to_canonical(sandbox_state, dry_run=args.dry_run)}
        )
    elif args.commit:
        commit_payload["reason"] = "thresholds_not_met"
    finished_at = datetime.now(timezone.utc)
    columns_used = load_allow_list_full(allow_list)
    manifest = build_manifest(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        args=args,
        columns_used=columns_used,
        forbidden_seen=list(engine.forbidden_seen),
        source_counts={
            "prediction_journal_rows": len(journal_rows),
            "paper_trades_closed": len(closed_trades),
            "paper_positions_open": len(positions),
            "agents": len(agents),
            "paper_portfolios": len(portfolios),
            "strategy_snapshots": len(snapshots),
            "market_signals": len(signals),
            "price_candles_5m": len(candles),
            "paper_position_marks": len(position_marks),
            "cycles_total": len(cycles),
            "train_cycles": len(train_cycles),
            "holdout_cycles": len(holdout_cycles),
        },
        engine=engine,
        pre_hashes=pre_hashes,
        post_hashes=post_hashes,
        warmed_hashes=warmed_hashes,
        sandbox_state=sandbox_state,
        commit_details=commit_payload,
        state_label=state_label,
        final_trust_state=final_trust_state,
        final_trust_state_per_regime=final_trust_state_per_regime,
        regime_prototype_count=len(service.regime_memory.prototypes)
        if hasattr(service.regime_memory, "prototypes")
        else 0,
        episode_buffer_size=len(service.episodic_memory),
        data_window={
            "min_ts": start.astimezone(timezone.utc).isoformat(),
            "max_ts": end.astimezone(timezone.utc).isoformat(),
            "days_covered": round(span_days, 4),
        },
        holdout_metrics=holdout_metrics,
    )
    manifest_path = sandbox_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "sandbox_dir": str(sandbox_dir),
        "replay_jsonl": str(replay_log_path),
        "holdout_dir": str(holdout_dir),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("REPLAY_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    result = asyncio.run(run_replay(args))
    print(
        json.dumps(
            {
                "ok": True,
                "manifest_path": result["manifest_path"],
                "sandbox_dir": result["sandbox_dir"],
                "replay_jsonl": result["replay_jsonl"],
                "holdout_dir": result["holdout_dir"],
                "state": result["manifest"]["state"],
                "cycles_replayed": result["manifest"]["cycles_replayed"],
                "trades_attributed": result["manifest"]["trades_attributed"],
                "regimes_observed": result["manifest"]["regimes_observed"],
                "forbidden_columns_seen": result["manifest"]["forbidden_columns_seen"],
                "commit": result["manifest"]["commit"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
