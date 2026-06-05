"""Phase 5 — unified decision engine.

Single source of truth for the trade-or-skip + size + portfolio decision.
The same module is consumed by:
  • the offline backtest simulator (`app/backtest/simulator.py`) — in-process
  • the live trader (`api-server/src/lib/paper-trader.ts`) — over HTTP via
    `/ml/decide`
so live and backtest behaviour cannot drift.
"""
from .engine import (
    DecisionRequest,
    DecisionResult,
    OpenPosition,
    PortfolioState,
    SpecialistView,
    decide,
    decide_direction,
)

__all__ = [
    "DecisionRequest",
    "DecisionResult",
    "OpenPosition",
    "PortfolioState",
    "SpecialistView",
    "decide",
    "decide_direction",
]
