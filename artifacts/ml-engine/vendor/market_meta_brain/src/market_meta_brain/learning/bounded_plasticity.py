from __future__ import annotations

from market_meta_brain.utils.math_utils import clamp


class BoundedPlasticityController:
    """Caps adaptation speed so the live supervisor cannot mutate too quickly."""

    def __init__(self, max_step: float = 0.05):
        self.max_step = max_step

    def update_value(self, current: float, target: float) -> float:
        delta = clamp(target - current, -self.max_step, self.max_step)
        return current + delta
