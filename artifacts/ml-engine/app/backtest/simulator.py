"""Deterministic event-driven simulator that mirrors the live paper-trader.

Load-bearing module: every gate, every fee, every SL/TP rule the live
trader applies must be replicated here so a backtest result is a faithful
prediction of what the live system would have done. All numeric constants
come from `shared/trading-frictions.json` via `contract.py`.

Out of v1 scope (documented; mismatch is acceptable for v1 only because
the live system can match the same conditions):
* Kelly sizing — backtest assumes a cold-start trader (no >100-trade
  history), so `tieredPositionPct` is always used.
* Fleet correlation brake — single-agent backtest, not a fleet.
* Dynamic tuning gates — the backtester always uses the JSON `gates_baseline`
  values. Tests that want to sweep gate values pass them via the
  `gate_*` keyword overrides.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from .contract import Frictions
from .regime import RegimeState

logger = logging.getLogger("ml-engine.backtest.simulator")

# Anomaly cancel-path threshold — must match paper-trader.ts:687 exactly.
# When triggered, the open-time cash impact (position_size + entry_fee) must
# be reversed in full — refund both, never just position_size. Mirrors the
# `recoverEntryFee` helper in trade-math.ts.
ANOMALY_PRICE_RATIO_HIGH = 3.0
ANOMALY_PRICE_RATIO_LOW = 0.33


def _recover_entry_fee(stored_position_size: float, taker_fee_pct: float) -> float:
    """Inverse of `entry_fee = pre_fee * taker_fee_pct` given the stored
    post-fee position size (`pre_fee * (1 - taker_fee_pct)`). Used by the
    anomaly-cancel paths to fully reverse the open-time cash deduction.
    Mirrors `recoverEntryFee` in artifacts/api-server/src/lib/trade-math.ts."""
    return stored_position_size * taker_fee_pct / (1.0 - taker_fee_pct)


@dataclass
class TradeRow:
    coin_id: str
    timeframe: str
    direction: str            # "up" | "down"
    entry_ts_ms: int
    exit_ts_ms: int
    entry_price: float        # post-slippage (matches live paperPositionsTable.entryPrice)
    exit_price: float         # post-slippage
    position_size_usd: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str          # "sl" | "tp" | "expiry" | "trailing_sl" | "cancelled_anomaly"
    regime_at_entry: str
    confidence: float
    # Phase 1 — fields needed to materialise a `trade_journal` row alongside
    # the prediction_journal row when this backtest result is shipped to the
    # api-server. peak_price gives MFE; mae_price gives MAE; fees + slippage
    # mirror the live paper-trader's fee/slippage capture.
    peak_price: float = 0.0
    mae_price: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage_pct: float = 0.0
    raw_entry_price: float = 0.0
    raw_exit_price: float = 0.0


@dataclass
class SkipRow:
    coin_id: str
    timeframe: str
    timestamp_ms: int
    reason: str
    detail: str = ""


@dataclass
class SimulationResult:
    trades: list[TradeRow]
    skips: list[SkipRow]
    final_equity: float
    initial_equity: float
    timeframe: str

    @property
    def total_trades(self) -> int: return len(self.trades)
    @property
    def realized_pnl(self) -> float:
        return self.final_equity - self.initial_equity

    def to_records(self) -> list[dict]:
        return [t.__dict__ for t in self.trades]


# ── Pure math helpers (mirror artifacts/api-server/src/lib/trade-math.ts) ──
def apply_entry_slippage(raw_price: float, direction: str, slippage_pct: float) -> float:
    return raw_price * (1 + slippage_pct if direction == "up" else 1 - slippage_pct)


def apply_exit_slippage(raw_price: float, direction: str, slippage_pct: float) -> float:
    return raw_price * (1 - slippage_pct if direction == "up" else 1 + slippage_pct)


def is_stop_loss_hit(direction: str, current_price: float, sl_price: float) -> bool:
    return current_price <= sl_price if direction == "up" else current_price >= sl_price


def is_take_profit_hit(direction: str, current_price: float, tp_price: float) -> bool:
    return current_price >= tp_price if direction == "up" else current_price <= tp_price


def round_trip_pnl_usd(
    direction: str, raw_entry: float, raw_exit: float, position_size_usd: float,
    *, fr: Frictions,
) -> float:
    """End-to-end PnL helper used by external callers / tests. Applies BOTH
    entry and exit slippage + both fees. The simulator itself does NOT use
    this — it tracks open/close cash flows separately to match live
    bookkeeping (entry fee already deducted at open).
    """
    adj_entry = apply_entry_slippage(raw_entry, direction, fr.slippage_pct)
    adj_exit = apply_exit_slippage(raw_exit, direction, fr.slippage_pct)
    qty = position_size_usd / adj_entry
    if direction == "up":
        gross = (adj_exit - adj_entry) * qty
    else:
        gross = (adj_entry - adj_exit) * qty
    exit_notional = position_size_usd + gross
    entry_fee = position_size_usd * fr.taker_fee_pct
    exit_fee  = max(0.0, exit_notional) * fr.maker_fee_pct
    return gross - entry_fee - exit_fee


def _close_pnl(
    direction: str, adj_entry: float, raw_exit: float, position_size_usd: float,
    *, fr: Frictions,
) -> tuple[float, float]:
    """Live-faithful close-side accounting: entry slippage + entry fee were
    ALREADY applied at open, so this only applies exit slippage + exit fee.
    Returns (pnl_usd, adj_exit_price).
    """
    adj_exit = apply_exit_slippage(raw_exit, direction, fr.slippage_pct)
    qty = position_size_usd / adj_entry
    if direction == "up":
        gross = (adj_exit - adj_entry) * qty
    else:
        gross = (adj_entry - adj_exit) * qty
    exit_notional = position_size_usd + gross
    exit_fee = max(0.0, exit_notional) * fr.maker_fee_pct
    return (gross - exit_fee), adj_exit


# ── Decision: live quant-brain emission rule ───────────────────────────────
# Phase 5 — the directional emit + confidence-reporting rule lives in the
# unified `app.decision_engine` module so the live api-server (`/ml/decide`
# HTTP endpoint) and the offline backtester both call the SAME function.
# Re-exported here under the legacy name so existing callers keep working.
from app.decision_engine import (  # noqa: E402, F401
    decide,
    decide_direction,
    DecisionRequest,
    PortfolioState,
    OpenPosition,
)


# ── Tick stream ────────────────────────────────────────────────────────────
@dataclass
class CoinTickStream:
    coin_id: str
    timestamps_ms: list[int]
    prices: list[float]

    def closes_after(self, t0_ms: int) -> Iterable[tuple[int, float]]:
        for ts, px in zip(self.timestamps_ms, self.prices):
            if ts > t0_ms:
                yield ts, px


@dataclass
class _Position:
    coin_id: str
    direction: str
    entry_ts_ms: int
    entry_price: float        # post-slippage adj_entry (live: paperPositionsTable.entryPrice)
    position_size_usd: float
    sl_price: float
    tp_price: float
    expires_at_ms: int
    confidence: float
    regime_at_entry: str
    peak_price: float          # mirror of paperPositionsTable.peakPrice
    trailing_active: bool = False  # set once we extend with a trailing stop
    # MAE tracker — opposite of peak_price. For "up" trades this is the
    # lowest price seen since entry (worst adverse excursion); for "down"
    # trades it's the highest. Initialised to entry_price at open.
    mae_price: float = 0.0
    raw_entry_price: float = 0.0


def _close_position(
    pos: _Position, raw_exit_price: float, exit_ts_ms: int, exit_reason: str,
    timeframe: str, fr: Frictions,
) -> TradeRow:
    pnl, adj_exit = _close_pnl(pos.direction, pos.entry_price, raw_exit_price,
                               pos.position_size_usd, fr=fr)
    pnl_pct = (pnl / pos.position_size_usd) * 100.0 if pos.position_size_usd > 0 else 0.0
    # Fees + slippage capture for the trade_journal row. entry_fee was paid
    # against the pre-fee notional; exit_fee is the maker fee on the gross
    # exit notional (gross can be negative if the position blew through SL,
    # in which case the live formula clamps fee at 0).
    pre_fee_notional = pos.position_size_usd / max(1e-12, 1.0 - fr.taker_fee_pct)
    entry_fee = pre_fee_notional * fr.taker_fee_pct
    exit_notional = pos.position_size_usd + (pnl + entry_fee)  # gross before exit fee
    exit_fee = max(0.0, exit_notional) * fr.maker_fee_pct
    return TradeRow(
        coin_id=pos.coin_id, timeframe=timeframe, direction=pos.direction,
        entry_ts_ms=pos.entry_ts_ms, exit_ts_ms=exit_ts_ms,
        entry_price=pos.entry_price, exit_price=adj_exit,
        position_size_usd=pos.position_size_usd,
        pnl_usd=pnl, pnl_pct=pnl_pct,
        exit_reason=exit_reason, regime_at_entry=pos.regime_at_entry,
        confidence=pos.confidence,
        peak_price=pos.peak_price,
        mae_price=pos.mae_price if pos.mae_price > 0 else pos.entry_price,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        slippage_pct=fr.slippage_pct,
        raw_entry_price=pos.raw_entry_price if pos.raw_entry_price > 0 else pos.entry_price,
        raw_exit_price=raw_exit_price,
    )


def _is_anomalous(adj_entry: float, raw_exit: float) -> bool:
    """Mirror of paper-trader.ts:687 — exit_price/entry_price out of
    [0.33, 3.0] is treated as a feed glitch and the position is refunded
    with zero PnL (no trade booked). The refund must reverse BOTH
    position_size and the entry_fee paid at open — see _recover_entry_fee."""
    if adj_entry <= 0:
        return True
    ratio = raw_exit / adj_entry
    return ratio > ANOMALY_PRICE_RATIO_HIGH or ratio < ANOMALY_PRICE_RATIO_LOW


# ── Simulator ──────────────────────────────────────────────────────────────
def simulate(
    *,
    timeframe: str,
    oos_predictions: pd.DataFrame,
    tick_streams: dict[str, CoinTickStream],
    fr: Frictions,
    regime_lookup,
    gate_min_confidence: Optional[float] = None,
    gate_min_tp_distance_pct: Optional[float] = None,
    gate_min_ev_vs_cost: Optional[float] = None,
    gate_counter_trend_min_confidence: Optional[float] = None,
) -> SimulationResult:
    initial_equity = fr.initial_balance_usd
    equity = initial_equity
    peak_equity = initial_equity
    cash = initial_equity
    open_positions: list[_Position] = []
    trades: list[TradeRow] = []
    skips: list[SkipRow] = []

    g_min_conf  = gate_min_confidence            if gate_min_confidence            is not None else fr.gate("MIN_CONFIDENCE_TO_TRADE")["value"]
    g_min_tpd   = gate_min_tp_distance_pct       if gate_min_tp_distance_pct       is not None else fr.gate("MIN_TP_DISTANCE_PCT")["value"]
    g_min_ev    = gate_min_ev_vs_cost            if gate_min_ev_vs_cost            is not None else fr.gate("MIN_EV_VS_COST")["value"]
    g_ct_min    = gate_counter_trend_min_confidence if gate_counter_trend_min_confidence is not None else fr.gate("COUNTER_TREND_MIN_CONFIDENCE")["value"]

    asym_long_min = fr.asymmetric_long_min_confidence
    max_open = fr.max_open_positions_per_agent
    max_at_risk = fr.max_portfolio_at_risk
    max_pos_pct = fr.max_position_pct
    daily_loss_limit_pct = fr.daily_loss_limit_pct
    drawdown_halt_pct = fr.drawdown_halt_pct
    tf_ms = fr.timeframe_ms(timeframe)
    sl_mult = fr.sl_mult(timeframe)
    tp_mult = fr.tp_mult(timeframe)
    atr_floor_pct = fr.atr_floor_pct(timeframe)
    trailing = fr.trailing
    trailing_extension_ms = fr.trailing_extension_ms(timeframe)
    round_trip_cost = fr.round_trip_cost_pct
    consec_loss_n = int(fr.raw["recent_loss_block"]["consecutive_losses"])

    day_state: dict[str, float] = {}  # date_str -> equity at day start
    # Per-coin closed-trade ledger for the consecutive-loss block
    # (matches paper-trader.ts:349 — last N closed trades for THIS coin).
    coin_recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=consec_loss_n))

    def _book_trade(t: TradeRow) -> None:
        trades.append(t)
        coin_recent[t.coin_id].append(1 if t.pnl_usd > 0 else 0)

    events = oos_predictions.sort_values("timestamp_ms").reset_index(drop=True)

    def _advance_positions_to(ts_ms: int) -> None:
        nonlocal cash, equity, peak_equity, open_positions
        still_open: list[_Position] = []
        for pos in open_positions:
            stream = tick_streams.get(pos.coin_id)
            if stream is None:
                if ts_ms >= pos.expires_at_ms:
                    trade = _close_position(pos, pos.entry_price, pos.expires_at_ms,
                                            "expiry", timeframe, fr)
                    _book_trade(trade); cash += pos.position_size_usd + trade.pnl_usd
                else:
                    still_open.append(pos)
                continue
            closed = False
            for tick_ts, tick_px in stream.closes_after(pos.entry_ts_ms):
                if tick_ts > ts_ms:
                    break
                # Track peak price (mirror of live newPeak update).
                # Also track MAE (worst adverse excursion) for trade_journal.
                if pos.direction == "up":
                    pos.peak_price = max(pos.peak_price, tick_px)
                    pos.mae_price = min(pos.mae_price, tick_px) if pos.mae_price > 0 else tick_px
                else:
                    pos.peak_price = min(pos.peak_price, tick_px)
                    pos.mae_price = max(pos.mae_price, tick_px) if pos.mae_price > 0 else tick_px

                # SL hit — also covers the trailing stop after extension.
                if is_stop_loss_hit(pos.direction, tick_px, pos.sl_price):
                    if _is_anomalous(pos.entry_price, tick_px):
                        # Refund: no PnL booked, no trade row. Reverse BOTH
                        # position_size and the entry_fee paid at open
                        # (mirrors paper-trader.ts anomaly-cancel path).
                        cash += pos.position_size_usd + _recover_entry_fee(pos.position_size_usd, fr.taker_fee_pct)
                        closed = True; break
                    reason = "trailing_sl" if pos.trailing_active else "sl"
                    trade = _close_position(pos, tick_px, tick_ts, reason, timeframe, fr)
                    _book_trade(trade); cash += pos.position_size_usd + trade.pnl_usd
                    closed = True; break
                if is_take_profit_hit(pos.direction, tick_px, pos.tp_price):
                    if _is_anomalous(pos.entry_price, tick_px):
                        cash += pos.position_size_usd + _recover_entry_fee(pos.position_size_usd, fr.taker_fee_pct)
                        closed = True; break
                    trade = _close_position(pos, tick_px, tick_ts, "tp", timeframe, fr)
                    _book_trade(trade); cash += pos.position_size_usd + trade.pnl_usd
                    closed = True; break
                if tick_ts >= pos.expires_at_ms:
                    # Live formula (paper-trader.ts:611-633): peak-price-based.
                    if pos.direction == "up":
                        peak_pnl_pct = (pos.peak_price - pos.entry_price) / pos.entry_price
                    else:
                        peak_pnl_pct = (pos.entry_price - pos.peak_price) / pos.entry_price
                    if peak_pnl_pct > trailing["min_peak_pnl_pct_to_trail"] and not pos.trailing_active:
                        gb = trailing["trail_giveback_fraction"]
                        if pos.direction == "up":
                            pos.sl_price = pos.peak_price * (1 - peak_pnl_pct * gb)
                        else:
                            pos.sl_price = pos.peak_price * (1 + peak_pnl_pct * gb)
                        pos.expires_at_ms = tick_ts + trailing_extension_ms
                        pos.trailing_active = True
                        continue  # don't close — keep walking the tape
                    if _is_anomalous(pos.entry_price, tick_px):
                        cash += pos.position_size_usd + _recover_entry_fee(pos.position_size_usd, fr.taker_fee_pct)
                        closed = True; break
                    trade = _close_position(pos, tick_px, tick_ts, "expiry",
                                            timeframe, fr)
                    _book_trade(trade); cash += pos.position_size_usd + trade.pnl_usd
                    closed = True; break
            if not closed:
                still_open.append(pos)
        open_positions = still_open
        equity = cash + sum(p.position_size_usd for p in open_positions)
        peak_equity = max(peak_equity, equity)

    for _, row in events.iterrows():
        ts_ms = int(row["timestamp_ms"])
        coin_id = str(row["coin_id"])

        _advance_positions_to(ts_ms)

        date_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()
        if date_str not in day_state:
            day_state[date_str] = equity
        day_loss = (day_state[date_str] - equity) / day_state[date_str] if day_state[date_str] > 0 else 0
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if day_loss >= daily_loss_limit_pct:
            skips.append(SkipRow(coin_id, timeframe, ts_ms, "daily_loss_limit"))
            continue
        if drawdown >= drawdown_halt_pct:
            skips.append(SkipRow(coin_id, timeframe, ts_ms, "drawdown_halt"))
            continue

        # ── Phase 5: route the gate stack through the unified engine ─────
        # Both live and backtest now share `decide()` as the single source
        # of truth for direction emit, confidence floor, counter-trend gate,
        # consecutive-loss block, EV gate, sizing, and portfolio constraints.
        # The simulator only owns: per-tick advance, daily-loss/drawdown
        # halts, max-open-positions check, already-open coin check, and
        # the bookkeeping after the engine approves an entry.
        exp_ret_pct = float(row["expected_return_pct"]) if "expected_return_pct" in row.index else 0.0
        regime_state = regime_lookup(ts_ms)
        raw_entry_price = float(row["entry_price"])
        atr_value = float(row["atr14"])

        if any(p.coin_id == coin_id for p in open_positions):
            skips.append(SkipRow(coin_id, timeframe, ts_ms, "already_open"))
            continue

        portfolio_state = PortfolioState(
            equity_usd=equity, cash_usd=cash,
            open_positions=[
                OpenPosition(
                    coin_id=p.coin_id,
                    direction=p.direction,
                    notional_usd=p.position_size_usd,
                    regime_at_entry=p.regime_at_entry,
                ) for p in open_positions
            ],
        )
        recent = list(coin_recent[coin_id])

        result = decide(DecisionRequest(
            coin_id=coin_id, timeframe=timeframe,
            last_price=raw_entry_price, atr_value=atr_value,
            prob_up=float(row["p_up"]), prob_down=float(row["p_down"]),
            prob_stable=float(row["p_stable"]),
            expected_return_pct=exp_ret_pct,
            regime=regime_state.regime,
            trend_bias=regime_state.trend_bias,
            portfolio=portfolio_state,
            recent_outcomes=recent,
            gate_min_confidence=gate_min_confidence,
            gate_min_tp_distance_pct=gate_min_tp_distance_pct,
            gate_min_ev_vs_cost=gate_min_ev_vs_cost,
            gate_counter_trend_min_confidence=gate_counter_trend_min_confidence,
        ), fr=fr)

        if result.action == "no_trade":
            skips.append(SkipRow(coin_id, timeframe, ts_ms,
                                 result.skip_reason or "abstain_unknown",
                                 result.skip_detail or ""))
            continue

        position_size = result.position_size_usd
        # Recover the entry fee that the engine already netted out, so the
        # simulator's cash bookkeeping matches paper-trader's debit pattern.
        entry_fee = position_size * fr.taker_fee_pct / max(1e-12, 1.0 - fr.taker_fee_pct)

        # entry_price is the post-slippage adjusted entry; use the same
        # `apply_entry_slippage` formula the engine uses internally.
        adj_entry = apply_entry_slippage(raw_entry_price, result.direction or "up", fr.slippage_pct)
        pos = _Position(
            coin_id=coin_id,
            direction=result.direction or "up",
            entry_ts_ms=ts_ms,
            entry_price=adj_entry,
            position_size_usd=position_size,
            sl_price=result.sl_price or 0.0,
            tp_price=result.tp_price or 0.0,
            expires_at_ms=ts_ms + tf_ms,
            confidence=result.confidence,
            regime_at_entry=regime_state.regime,
            peak_price=adj_entry,
            mae_price=adj_entry,
            raw_entry_price=raw_entry_price,
        )
        open_positions.append(pos)
        cash -= (position_size + entry_fee)
        equity = cash + sum(p.position_size_usd for p in open_positions)

    if open_positions:
        last_ts = int(events["timestamp_ms"].max()) if len(events) else 0
        _advance_positions_to(last_ts + 10**12)
        for pos in open_positions:
            trade = _close_position(pos, pos.entry_price, last_ts, "expiry", timeframe, fr)
            _book_trade(trade); cash += pos.position_size_usd + trade.pnl_usd
        open_positions = []
        equity = cash

    return SimulationResult(
        trades=trades, skips=skips,
        final_equity=equity, initial_equity=initial_equity,
        timeframe=timeframe,
    )
