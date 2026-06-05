"""Per-regime metric breakdown — bins trades by `regime_at_entry`."""
from __future__ import annotations

from typing import Sequence

from .metrics import Metrics, compute_metrics
from .simulator import TradeRow

# Phase 2 — canonical 6-class regime taxonomy. Matches REGIME_LABELS in
# app/regime.py so live, training, and backtest reports all use the same
# regime axis.
REGIMES = (
    "trending_up",
    "trending_down",
    "range_chop",
    "high_vol_breakout",
    "low_vol_compression",
    "panic_liquidation",
)


def regime_breakdown(
    trades: Sequence[TradeRow], initial_equity: float,
) -> dict[str, dict]:
    """Per-regime metrics. Initial equity passed through unchanged so each
    regime's drawdown is comparable; the metric is "if you'd ONLY taken
    these trades, what was the result". Regimes with zero trades are
    included with n_trades=0 so the report is self-describing.
    """
    out: dict[str, dict] = {}
    for r in REGIMES:
        bucket = [t for t in trades if t.regime_at_entry == r]
        out[r] = compute_metrics(bucket, initial_equity).to_dict()
    return out
