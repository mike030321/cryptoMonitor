from __future__ import annotations

from dataclasses import asdict

from market_meta_brain.domain.types import MarketMetaState, RegimePrototype
from market_meta_brain.utils.math_utils import clamp, cosine_similarity


class RegimePrototypeMemory:
    def __init__(self, capacity: int = 64, match_threshold: float = 0.92):
        self.capacity = capacity
        self.match_threshold = match_threshold
        self.prototypes: list[RegimePrototype] = []

    @staticmethod
    def state_to_vector(state: MarketMetaState) -> list[float]:
        return [
            state.stress,
            state.safety,
            state.fatigue,
            state.coherence,
            state.surprise,
            state.curiosity,
            state.regime_certainty,
            state.opportunity_strength,
            state.capital_safety,
            state.model_health,
            state.risk_pressure,
        ]

    def recall(self, state: MarketMetaState) -> tuple[RegimePrototype | None, float]:
        if not self.prototypes:
            return None, 0.0
        vector = self.state_to_vector(state)
        scored = [(p, cosine_similarity(vector, p.state_vector)) for p in self.prototypes]
        prototype, score = max(scored, key=lambda x: x[1])
        return prototype, score

    def update(self, state: MarketMetaState, reward: float) -> None:
        vector = self.state_to_vector(state)
        dominant_families = sorted(
            state.family_states.keys(),
            key=lambda k: state.family_states[k].avg_edge,
            reverse=True,
        )[:3]
        prototype, score = self.recall(state)
        if prototype is not None and score >= self.match_threshold:
            alpha = 1.0 / (prototype.supporting_count + 1)
            prototype.state_vector = [
                (1 - alpha) * old + alpha * new
                for old, new in zip(prototype.state_vector, vector)
            ]
            prototype.supporting_count += 1
            prototype.average_outcome = (1 - alpha) * prototype.average_outcome + alpha * reward
            prototype.stability_score = clamp(0.97 * prototype.stability_score + 0.03 * state.coherence, 0.0, 1.0)
            prototype.dominant_families = dominant_families
            return

        self.prototypes.append(
            RegimePrototype(
                regime_id=state.dominant_regime,
                state_vector=vector,
                supporting_count=1,
                average_outcome=reward,
                stability_score=state.coherence,
                dominant_families=dominant_families,
            )
        )
        if len(self.prototypes) > self.capacity:
            self.prototypes.sort(key=lambda p: (p.supporting_count, p.stability_score, p.average_outcome))
            self.prototypes.pop(0)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "match_threshold": self.match_threshold, "prototypes": [asdict(p) for p in self.prototypes]}
