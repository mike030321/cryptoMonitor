from market_meta_brain.domain.types import MarketMetaState
from market_meta_brain.policy.safety_guardrails import SafetyGuardrails


def make_state(capital_safety: float, stress: float, risk_pressure: float):
    return MarketMetaState(
        stress=stress,
        safety=max(0.0, 1.0 - stress),
        fatigue=0.2,
        coherence=0.8,
        surprise=0.1,
        curiosity=0.2,
        regime_certainty=0.8,
        opportunity_strength=0.5,
        capital_safety=capital_safety,
        model_health=0.7,
        risk_pressure=risk_pressure,
        family_states={},
        dominant_regime="trend",
    )


def test_guardrails_disable_exploration_under_stress():
    g = SafetyGuardrails()
    assert g.max_exploration_budget(make_state(0.3, 0.8, 0.8)) == 0.0


def test_guardrails_enable_hard_defense_when_risk_is_high():
    g = SafetyGuardrails()
    assert g.defensive_mode(make_state(0.2, 0.6, 0.9)) == "hard"
