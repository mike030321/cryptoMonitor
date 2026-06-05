from __future__ import annotations

from market_meta_brain.domain.types import BrainDirective, MarketMetaState, SupervisoryAction
from market_meta_brain.policy.safety_guardrails import SafetyGuardrails
from market_meta_brain.utils.math_utils import clamp, softmax_dict


class DirectivePolicy:
    def __init__(self) -> None:
        self.guardrails = SafetyGuardrails()

    def build(
        self,
        state: MarketMetaState,
        actions: list[SupervisoryAction],
        trust_map: dict[str, float],
    ) -> BrainDirective:
        defensive_mode = self.guardrails.defensive_mode(state)
        caution_level = self.guardrails.caution_level(state)
        max_exploration = self.guardrails.max_exploration_budget(state)

        suppressed_families: list[str] = []
        reason_codes: list[str] = []
        exploration_budget = min(max_exploration, 0.02 + 0.03 * state.curiosity)

        for action in actions:
            reason_codes.extend(action.rationale)
            if action.action_type == "SUPPRESS_FAMILY":
                suppressed_families.extend(action.family_targets)
            elif action.action_type == "REDUCE_RISK":
                caution_level = clamp(caution_level + 0.15 * action.strength, 0.0, 1.0)
            elif action.action_type == "ENTER_DEFENSIVE_MODE":
                defensive_mode = "hard" if action.strength > 0.75 else max(defensive_mode, "soft")
            elif action.action_type == "ALLOW_MICRO_EXPLORATION":
                exploration_budget = min(max_exploration, exploration_budget + 0.02 * action.strength)

        adjusted_trust = {}
        for family, trust in trust_map.items():
            adjusted = trust
            if family in suppressed_families:
                adjusted *= 0.4
            if defensive_mode == "hard":
                adjusted *= 0.75
            adjusted_trust[family] = clamp(adjusted, 0.0, 1.5)

        allocation_weight = softmax_dict(adjusted_trust, temperature=max(0.25, 1.0 - state.coherence * 0.4))

        paused_slices: list[str] = []
        suppress_signal = bool(suppressed_families) or defensive_mode != "off"
        reason_codes = sorted(set(reason_codes))
        if defensive_mode != "off":
            reason_codes.append(f"DEFENSIVE_MODE_{defensive_mode.upper()}")
        if suppressed_families:
            reason_codes.append("FAMILY_SUPPRESSION_ACTIVE")

        return BrainDirective(
            trust_multiplier=adjusted_trust,
            allocation_weight=allocation_weight,
            caution_level=caution_level,
            exploration_budget=exploration_budget,
            suppress_signal=suppress_signal,
            defensive_mode=defensive_mode,
            suppressed_families=sorted(set(suppressed_families)),
            paused_slices=paused_slices,
            reason_codes=reason_codes,
            meta_state=state,
        )
