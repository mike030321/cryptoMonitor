from __future__ import annotations

from dataclasses import dataclass, field

from market_meta_brain.domain.types import FamilyState, GovernanceOutcome, MarketMetaState
from market_meta_brain.learning.bounded_plasticity import BoundedPlasticityController
from market_meta_brain.utils.math_utils import clamp


@dataclass(slots=True)
class FamilyTrustState:
    trust: float = 1.0
    stability: float = 0.5
    exploration_eligibility: float = 0.2
    failure_streak: int = 0
    recovery_score: float = 0.5
    last_regime: str = "unknown"


@dataclass(slots=True)
class StrategyTrustModel:
    learning_rate: float = 0.04
    trust_by_family: dict[str, FamilyTrustState] = field(default_factory=dict)
    plasticity: BoundedPlasticityController = field(default_factory=lambda: BoundedPlasticityController(max_step=0.05))

    def ensure_family(self, family: str) -> FamilyTrustState:
        if family not in self.trust_by_family:
            self.trust_by_family[family] = FamilyTrustState()
        return self.trust_by_family[family]

    def score_family(self, family_state: FamilyState, meta_state: MarketMetaState) -> float:
        internal = self.ensure_family(family_state.family)
        raw = (
            0.28 * family_state.avg_edge
            + 0.20 * family_state.avg_calibrated_confidence
            + 0.14 * family_state.avg_accuracy
            - 0.16 * family_state.avg_disagreement
            - 0.10 * family_state.avg_prediction_error
            - 0.06 * family_state.avg_drawdown_state
            - 0.06 * family_state.avg_risk_score
        )
        stabilized = 0.65 * internal.trust + 0.35 * raw
        if meta_state.stress > 0.65:
            stabilized -= 0.15 * family_state.avg_risk_score
        return clamp(stabilized, 0.0, 1.5)

    def produce_trust_map(self, meta_state: MarketMetaState) -> dict[str, float]:
        return {
            family: self.score_family(family_state, meta_state)
            for family, family_state in meta_state.family_states.items()
        }

    def learn_from_outcome(
        self,
        family_outcomes: dict[str, GovernanceOutcome],
        regime: str,
    ) -> None:
        """Bounded trust update keyed off a realized outcome.

        Replay (Task #467) cannot derive `correct_defense`,
        `correct_suppression`, or `missed_edge_cost` from real journal
        data — counterfactuals require a what-if simulator. When any of
        those three fields is `None` the corresponding trust term is
        skipped (treated as "no information"), never coerced to zero.
        Skipping zero would be wrong: zero says "we measured no
        defensive value", `None` says "we did not measure".
        """
        for family, outcome in family_outcomes.items():
            state = self.ensure_family(family)
            terms = (
                1.0
                + 0.45 * outcome.realized_pnl
                - 0.55 * outcome.realized_drawdown
            )
            if outcome.correct_suppression is not None:
                terms += 0.20 * outcome.correct_suppression
            if outcome.correct_defense is not None:
                terms += 0.20 * outcome.correct_defense
            if outcome.missed_edge_cost is not None:
                terms -= 0.35 * outcome.missed_edge_cost
            target = clamp(terms, 0.2, 1.8)
            state.trust = self.plasticity.update_value(state.trust, target)
            state.stability = self.plasticity.update_value(state.stability, outcome.realized_stability)
            state.exploration_eligibility = self.plasticity.update_value(
                state.exploration_eligibility,
                clamp(0.7 * state.stability + 0.3 * max(0.0, outcome.realized_pnl), 0.0, 1.0),
            )
            if outcome.realized_pnl < 0.0 or outcome.realized_drawdown > 0.0:
                state.failure_streak += 1
            else:
                state.failure_streak = max(0, state.failure_streak - 1)
            state.recovery_score = self.plasticity.update_value(
                state.recovery_score,
                clamp(0.5 * state.stability + 0.5 * max(0.0, outcome.realized_pnl), 0.0, 1.0),
            )
            state.last_regime = regime
