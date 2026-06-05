from __future__ import annotations

from dataclasses import asdict

from market_meta_brain.domain.types import GovernanceEpisode, GovernanceOutcome, TelemetryBatch
from market_meta_brain.ingest.telemetry_adapter import TelemetryAdapter
from market_meta_brain.learning.reward_model import compute_meta_reward
from market_meta_brain.learning.trust_model import StrategyTrustModel
from market_meta_brain.memory.episodic_market import EpisodicMarketMemory
from market_meta_brain.memory.regime_prototypes import RegimePrototypeMemory
from market_meta_brain.memory.working_context import WorkingContext
from market_meta_brain.planning.supervisor_planner import SupervisorPlanner
from market_meta_brain.policy.directive_policy import DirectivePolicy
from market_meta_brain.runtime.logging import JsonlLogger
from market_meta_brain.state.drive_mapper import DriveMapper
from market_meta_brain.state.meta_state import MetaStateMapper


class MarketMetaBrainService:
    def __init__(self, logger: JsonlLogger | None = None):
        self.adapter = TelemetryAdapter()
        self.state_mapper = MetaStateMapper()
        self.drive_mapper = DriveMapper()
        self.trust_model = StrategyTrustModel()
        self.episodic_memory = EpisodicMarketMemory()
        self.regime_memory = RegimePrototypeMemory()
        self.working_context = WorkingContext()
        self.planner = SupervisorPlanner()
        self.policy = DirectivePolicy()
        self.logger = logger or JsonlLogger(None)

    def evaluate(self, batch: TelemetryBatch):
        family_states = self.adapter.aggregate_families(batch)
        dominant_regime = self.adapter.dominant_regime(batch)
        meta_state = self.state_mapper.map(family_states, batch.portfolio, dominant_regime)
        trust_map = self.trust_model.produce_trust_map(meta_state)
        drives = self.drive_mapper.map(meta_state)
        actions = self.planner.plan(meta_state, drives, trust_map)
        directive = self.policy.build(meta_state, actions, trust_map)

        self.working_context.push(
            {
                "stress": meta_state.stress,
                "coherence": meta_state.coherence,
                "risk_pressure": meta_state.risk_pressure,
                "caution_level": directive.caution_level,
                "defensive_mode": directive.defensive_mode,
            }
        )
        self.logger.log(
            {
                "type": "directive",
                "timestamp": batch.timestamp,
                "directive": directive.to_dict(),
            }
        )
        return directive

    def record_outcome(self, directive, outcome: GovernanceOutcome, timestamp: str | None = None) -> float:
        if directive.meta_state is None:
            raise ValueError("Directive must include meta_state to record an outcome.")
        reward = compute_meta_reward(outcome)
        state = directive.meta_state
        action_summary = {
            "caution_level": directive.caution_level,
            "exploration_budget": directive.exploration_budget,
            "suppressed_count": float(len(directive.suppressed_families)),
            "defensive_hard": 1.0 if directive.defensive_mode == "hard" else 0.0,
        }
        family_snapshot = {family: fs.avg_edge for family, fs in state.family_states.items()}
        episode = GovernanceEpisode(
            timestamp=timestamp,
            meta_state_vector=self.regime_memory.state_to_vector(state),
            dominant_regime=state.dominant_regime,
            family_snapshot=family_snapshot,
            action_summary=action_summary,
            reward=reward,
            outcome=outcome,
        )
        self.episodic_memory.push(episode)
        self.regime_memory.update(state, reward)
        family_outcomes = {
            family: GovernanceOutcome(
                realized_pnl=outcome.realized_pnl,
                realized_drawdown=outcome.realized_drawdown,
                realized_stability=outcome.realized_stability,
                turnover_cost=outcome.turnover_cost,
                action_churn=outcome.action_churn,
                correct_defense=outcome.correct_defense,
                correct_suppression=outcome.correct_suppression,
                missed_edge_cost=outcome.missed_edge_cost,
            )
            for family in state.family_states
        }
        self.trust_model.learn_from_outcome(family_outcomes, state.dominant_regime)
        self.logger.log({"type": "outcome", "timestamp": timestamp, "reward": reward, "outcome": asdict(outcome)})
        return reward
