from __future__ import annotations

from market_meta_brain.domain.types import GovernanceOutcome


def compute_meta_reward(outcome: GovernanceOutcome) -> float:
    """Bounded scalar reward consumed by trust + episodic memory updates.

    Replay (Task #467) feeds outcomes whose three counterfactual fields
    (`correct_defense`, `correct_suppression`, `missed_edge_cost`) are
    `None` because they cannot be derived without a what-if simulator.
    Treat `None` as "term not contributed" — never as zero — so the
    reward is honestly partial instead of biased toward neutrality.
    """
    pnl_quality = outcome.realized_pnl
    drawdown_penalty = 1.4 * max(0.0, outcome.realized_drawdown)
    instability_penalty = 0.9 * max(0.0, 1.0 - outcome.realized_stability)
    churn_penalty = 0.7 * max(0.0, outcome.action_churn)
    turnover_penalty = 0.4 * max(0.0, outcome.turnover_cost)
    defense_bonus = (
        0.6 * max(0.0, outcome.correct_defense)
        if outcome.correct_defense is not None
        else 0.0
    )
    suppression_bonus = (
        0.5 * max(0.0, outcome.correct_suppression)
        if outcome.correct_suppression is not None
        else 0.0
    )
    missed_edge_penalty = (
        0.9 * max(0.0, outcome.missed_edge_cost)
        if outcome.missed_edge_cost is not None
        else 0.0
    )
    return float(
        pnl_quality
        - drawdown_penalty
        - instability_penalty
        - churn_penalty
        - turnover_penalty
        + defense_bonus
        + suppression_bonus
        - missed_edge_penalty
    )
