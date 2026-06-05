from __future__ import annotations

from market_meta_brain.domain.types import MarketMetaState, SupervisoryAction
from market_meta_brain.state.drive_mapper import DriveSnapshot
from market_meta_brain.utils.math_utils import clamp


class SupervisorPlanner:
    """Scores a small action set for supervisory portfolio control."""

    def plan(
        self,
        meta_state: MarketMetaState,
        drives: DriveSnapshot,
        trust_map: dict[str, float],
    ) -> list[SupervisoryAction]:
        actions: list[SupervisoryAction] = []
        low_trust = [family for family, score in trust_map.items() if score < 0.75]
        high_trust = [family for family, score in trust_map.items() if score > 1.05]

        if drives.defend_capital > 0.55:
            actions.append(
                SupervisoryAction(
                    action_type="ENTER_DEFENSIVE_MODE",
                    strength=drives.defend_capital,
                    rationale=["RISK_PRESSURE", "CAPITAL_DEFENSE"],
                )
            )
        if drives.restore_coherence > 0.45 and low_trust:
            actions.append(
                SupervisoryAction(
                    action_type="SUPPRESS_FAMILY",
                    family_targets=low_trust,
                    strength=clamp(drives.restore_coherence, 0.0, 1.0),
                    rationale=["LOW_COHERENCE", "MODEL_DISAGREEMENT"],
                )
            )
        if drives.exploit_edge > 0.45 and high_trust:
            actions.append(
                SupervisoryAction(
                    action_type="ROTATE_TRUST",
                    family_targets=high_trust,
                    strength=clamp(drives.exploit_edge, 0.0, 1.0),
                    rationale=["STABLE_EDGE", "CONFIDENCE_ALIGNMENT"],
                )
            )
        if drives.probe_opportunity > 0.20 and meta_state.capital_safety > 0.65:
            actions.append(
                SupervisoryAction(
                    action_type="ALLOW_MICRO_EXPLORATION",
                    family_targets=high_trust[:1] or list(trust_map.keys())[:1],
                    strength=clamp(drives.probe_opportunity, 0.0, 0.35),
                    rationale=["SAFE_EXPLORATION"],
                )
            )
        if drives.reduce_churn > 0.40:
            actions.append(
                SupervisoryAction(
                    action_type="REDUCE_RISK",
                    strength=clamp(drives.reduce_churn, 0.0, 1.0),
                    rationale=["OVERTRADING_RISK", "ACTION_CHURN"],
                )
            )

        if not actions:
            actions.append(SupervisoryAction(action_type="HOLD_POLICY", strength=0.25, rationale=["NO_OVERRIDE"]))
        return actions
