"""Trade-order Monte Carlo reshuffle.

Permuting the trade order tells you how much of the realised P&L curve was
order-luck vs systematic edge. We resample (without replacement) the trade
ledger N times and report the 5/50/95th percentile equity curves and the
distribution of final P&L and max drawdown.

Determinism: a fixed numpy seed yields identical bands across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from .metrics import max_drawdown_pct
from .simulator import TradeRow


@dataclass
class MonteCarloResult:
    n_runs: int
    final_pnl_p05: float
    final_pnl_p50: float
    final_pnl_p95: float
    max_drawdown_p05: float
    max_drawdown_p50: float
    max_drawdown_p95: float
    equity_band_p05: list[float]
    equity_band_p50: list[float]
    equity_band_p95: list[float]

    def to_dict(self) -> dict: return asdict(self)


def run_monte_carlo(
    trades: Sequence[TradeRow], initial_equity: float, *,
    n_runs: int = 2000, seed: int = 42,
) -> MonteCarloResult:
    n = len(trades)
    if n == 0:
        return MonteCarloResult(
            n_runs=0,
            final_pnl_p05=0.0, final_pnl_p50=0.0, final_pnl_p95=0.0,
            max_drawdown_p05=0.0, max_drawdown_p50=0.0, max_drawdown_p95=0.0,
            equity_band_p05=[initial_equity], equity_band_p50=[initial_equity],
            equity_band_p95=[initial_equity],
        )
    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl_usd for t in trades], dtype=float)
    # Build the (n_runs, n+1) curves matrix.
    curves = np.empty((n_runs, n + 1), dtype=float)
    curves[:, 0] = initial_equity
    for i in range(n_runs):
        order = rng.permutation(n)
        curves[i, 1:] = initial_equity + np.cumsum(pnls[order])

    finals = curves[:, -1] - initial_equity
    dds = np.array([max_drawdown_pct(curves[i].tolist()) for i in range(n_runs)])

    p05 = np.percentile(curves, 5, axis=0).tolist()
    p50 = np.percentile(curves, 50, axis=0).tolist()
    p95 = np.percentile(curves, 95, axis=0).tolist()

    return MonteCarloResult(
        n_runs=n_runs,
        final_pnl_p05=float(np.percentile(finals, 5)),
        final_pnl_p50=float(np.percentile(finals, 50)),
        final_pnl_p95=float(np.percentile(finals, 95)),
        max_drawdown_p05=float(np.percentile(dds, 5)),
        max_drawdown_p50=float(np.percentile(dds, 50)),
        max_drawdown_p95=float(np.percentile(dds, 95)),
        equity_band_p05=p05, equity_band_p50=p50, equity_band_p95=p95,
    )
