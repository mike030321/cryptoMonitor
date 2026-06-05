from __future__ import annotations

from dataclasses import dataclass

from market_meta_brain.domain.types import MarketMetaState


@dataclass(slots=True)
class DriveSnapshot:
    defend_capital: float
    exploit_edge: float
    probe_opportunity: float
    reduce_churn: float
    restore_coherence: float


class DriveMapper:
    """Translates meta-state into supervisory drives."""

    def map(self, state: MarketMetaState) -> DriveSnapshot:
        defend_capital = min(1.0, 0.50 * state.stress + 0.50 * state.risk_pressure)
        exploit_edge = min(1.0, 0.55 * state.opportunity_strength + 0.45 * state.coherence)
        probe_opportunity = min(1.0, state.curiosity)
        reduce_churn = min(1.0, 0.65 * state.fatigue + 0.35 * state.surprise)
        restore_coherence = min(1.0, 1.0 - state.coherence + 0.25 * state.surprise)
        return DriveSnapshot(
            defend_capital=defend_capital,
            exploit_edge=exploit_edge,
            probe_opportunity=probe_opportunity,
            reduce_churn=reduce_churn,
            restore_coherence=restore_coherence,
        )
