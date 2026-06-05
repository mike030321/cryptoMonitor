from __future__ import annotations

from market_meta_brain.domain.types import MarketMetaState
from market_meta_brain.utils.math_utils import clamp


class SafetyGuardrails:
    def max_exploration_budget(self, state: MarketMetaState) -> float:
        if state.capital_safety < 0.4 or state.stress > 0.7:
            return 0.0
        if state.capital_safety < 0.6:
            return 0.02
        return 0.05

    def defensive_mode(self, state: MarketMetaState) -> str:
        if state.risk_pressure > 0.75 or state.capital_safety < 0.25:
            return "hard"
        if state.risk_pressure > 0.55 or state.stress > 0.60:
            return "soft"
        return "off"

    def caution_level(self, state: MarketMetaState) -> float:
        return clamp(0.45 * state.stress + 0.35 * state.risk_pressure + 0.20 * state.surprise, 0.0, 1.0)
