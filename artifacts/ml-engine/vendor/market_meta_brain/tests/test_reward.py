from market_meta_brain.domain.types import GovernanceOutcome
from market_meta_brain.learning.reward_model import compute_meta_reward


def test_reward_penalizes_drawdown():
    good = GovernanceOutcome(0.1, 0.0, 0.9, 0.01, 0.02, 0.2, 0.2, 0.0)
    bad = GovernanceOutcome(0.1, 0.2, 0.9, 0.01, 0.02, 0.2, 0.2, 0.0)
    assert compute_meta_reward(good) > compute_meta_reward(bad)
