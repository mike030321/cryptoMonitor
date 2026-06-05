"""Per-trade metrics computed off the simulator's TradeRow ledger."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .simulator import TradeRow


@dataclass
class Metrics:
    n_trades: int
    win_rate: float
    avg_winner_usd: float
    avg_loser_usd: float
    expectancy_usd: float
    profit_factor: float
    sharpe_per_trade: float
    max_drawdown_pct: float
    final_pnl_usd: float
    time_in_market_pct: float       # share of backtest window with ≥1 open position
    avg_hold_ms: float              # mean trade duration (entry_ts → exit_ts)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Make NaN safely json-serializable.
        for k, v in d.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


def _equity_curve(initial: float, trades: Sequence[TradeRow]) -> list[float]:
    eq = [initial]
    for t in trades:
        eq.append(eq[-1] + t.pnl_usd)
    return eq


def max_drawdown_pct(curve: Sequence[float]) -> float:
    if not curve:
        return 0.0
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak <= 0:
            continue
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    return max_dd * 100.0


def _time_in_market_pct(trades: Sequence[TradeRow]) -> float:
    """Union of [entry_ts, exit_ts] intervals divided by total span. Two
    overlapping trades count once (we want fraction of time the strategy
    has *any* exposure, not gross exposure)."""
    if not trades:
        return 0.0
    intervals = sorted([(t.entry_ts_ms, t.exit_ts_ms) for t in trades])
    span_lo = intervals[0][0]
    span_hi = max(t.exit_ts_ms for t in trades)
    if span_hi <= span_lo:
        return 0.0
    merged_lo, merged_hi = intervals[0]
    union_ms = 0
    for lo, hi in intervals[1:]:
        if lo <= merged_hi:
            merged_hi = max(merged_hi, hi)
        else:
            union_ms += merged_hi - merged_lo
            merged_lo, merged_hi = lo, hi
    union_ms += merged_hi - merged_lo
    return union_ms / (span_hi - span_lo) * 100.0


def compute_metrics(trades: Sequence[TradeRow], initial_equity: float) -> Metrics:
    n = len(trades)
    if n == 0:
        return Metrics(
            n_trades=0, win_rate=0.0, avg_winner_usd=0.0, avg_loser_usd=0.0,
            expectancy_usd=0.0, profit_factor=0.0, sharpe_per_trade=0.0,
            max_drawdown_pct=0.0, final_pnl_usd=0.0,
            time_in_market_pct=0.0, avg_hold_ms=0.0,
        )
    pnls = [t.pnl_usd for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / n
    avg_w = sum(wins) / len(wins) if wins else 0.0
    avg_l = sum(losses) / len(losses) if losses else 0.0
    expectancy = sum(pnls) / n

    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    profit_factor = (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0)

    mean_p = expectancy
    if n > 1:
        var_p = sum((p - mean_p) ** 2 for p in pnls) / (n - 1)
        std_p = math.sqrt(var_p)
        sharpe = (mean_p / std_p) if std_p > 0 else 0.0
    else:
        sharpe = 0.0

    curve = _equity_curve(initial_equity, trades)
    avg_hold = sum(max(0, t.exit_ts_ms - t.entry_ts_ms) for t in trades) / n
    return Metrics(
        n_trades=n,
        win_rate=win_rate,
        avg_winner_usd=avg_w,
        avg_loser_usd=avg_l,
        expectancy_usd=expectancy,
        profit_factor=profit_factor if not math.isinf(profit_factor) else 1e9,
        sharpe_per_trade=sharpe,
        max_drawdown_pct=max_drawdown_pct(curve),
        final_pnl_usd=curve[-1] - initial_equity,
        time_in_market_pct=_time_in_market_pct(trades),
        avg_hold_ms=avg_hold,
    )
