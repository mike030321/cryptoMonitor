from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(slots=True)
class QuantSliceTelemetry:
    coin: str
    timeframe: str
    strategy_family: str
    edge: float
    confidence: float
    calibrated_confidence: float
    risk_score: float
    recent_accuracy: float
    pnl_state: float
    drawdown_state: float
    disagreement: float
    prediction_error: float
    regime: str
    volatility: float
    correlation_shift: float
    exposure: float
    turnover: float
    slippage_bps: float
    anomaly_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PortfolioTelemetry:
    total_drawdown: float
    realized_vol: float
    concentration: float
    leverage: float
    liquidity_stress: float
    correlation_shift: float
    active_risk_budget: float
    kill_switch_distance: float
    anomaly_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TelemetryBatch:
    slices: list[QuantSliceTelemetry]
    portfolio: PortfolioTelemetry
    timestamp: str | None = None


@dataclass(slots=True)
class FamilyState:
    family: str
    avg_edge: float
    avg_confidence: float
    avg_calibrated_confidence: float
    avg_disagreement: float
    avg_prediction_error: float
    avg_accuracy: float
    avg_pnl_state: float
    avg_drawdown_state: float
    avg_volatility: float
    avg_correlation_shift: float
    total_exposure: float
    total_turnover: float
    avg_risk_score: float
    avg_slippage_bps: float
    anomaly_count: int
    slice_count: int


@dataclass(slots=True)
class MarketMetaState:
    stress: float
    safety: float
    fatigue: float
    coherence: float
    surprise: float
    curiosity: float
    regime_certainty: float
    opportunity_strength: float
    capital_safety: float
    model_health: float
    risk_pressure: float
    family_states: dict[str, FamilyState] = field(default_factory=dict)
    dominant_regime: str = "unknown"


@dataclass(slots=True)
class RegimePrototype:
    regime_id: str
    state_vector: list[float]
    supporting_count: int
    average_outcome: float
    stability_score: float
    dominant_families: list[str]


@dataclass(slots=True)
class SupervisoryAction:
    action_type: str
    family_targets: list[str] = field(default_factory=list)
    strength: float = 0.0
    rationale: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BrainDirective:
    trust_multiplier: dict[str, float]
    allocation_weight: dict[str, float]
    caution_level: float
    exploration_budget: float
    suppress_signal: bool
    defensive_mode: str
    suppressed_families: list[str]
    paused_slices: list[str]
    reason_codes: list[str] = field(default_factory=list)
    meta_state: MarketMetaState | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class GovernanceOutcome:
    """Outcome record fed back into bounded learning.

    Live trading paths populate every field. Replay (Task #467) cannot
    derive the three counterfactual fields (`correct_defense`,
    `correct_suppression`, `missed_edge_cost`) without a what-if
    simulator and therefore passes them as `None`. Reward and trust
    updates treat `None` as "skip this term" — never as zero — so the
    replay degrades gracefully without fabricating signal. See
    `reward_model.compute_meta_reward` and
    `trust_model.learn_from_outcome`.
    """

    realized_pnl: float
    realized_drawdown: float
    realized_stability: float
    turnover_cost: float
    action_churn: float
    correct_defense: float | None = None
    correct_suppression: float | None = None
    missed_edge_cost: float | None = None


@dataclass(slots=True)
class GovernanceEpisode:
    timestamp: str | None
    meta_state_vector: list[float]
    dominant_regime: str
    family_snapshot: dict[str, float]
    action_summary: dict[str, float]
    reward: float
    outcome: GovernanceOutcome
