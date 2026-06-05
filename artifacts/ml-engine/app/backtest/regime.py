"""Backtest regime helpers — Phase 2.

Single source of truth: the canonical 6-class classifier in `app.regime`.
The backtest simulator stamps `RegimeState.regime` (one of the 6 labels:
trending_up, trending_down, range_chop, high_vol_breakout,
low_vol_compression, panic_liquidation) on every simulated trade so
backtest, live, and training all share one regime taxonomy.

The legacy 4-class classifier was removed when the canonical 6-class
classifier landed; downstream consumers (`regime_breakdown`,
`journal_client`) read the 6-class label directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..regime import (
    REGIME_LABELS as REGIME_LABELS_V2,
    RegimeDecision as RegimeDecisionV2,
    RegimeLabel,
    classify_regime_from_basket as classify_regime_v2_from_basket,
    classify_regime_from_features as classify_regime_v2_from_features,
)

REGIME_LABELS = REGIME_LABELS_V2

TrendBias = str  # "bullish" | "bearish" | "neutral"


@dataclass(frozen=True)
class RegimeState:
    """Backtest regime state. `regime` is the canonical 6-class label."""

    regime: RegimeLabel
    confidence: float
    avg_change_24h: float
    cross_coin_volatility: float
    bullish_ratio: float
    trend_bias: TrendBias

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "avg_change_24h": self.avg_change_24h,
            "cross_coin_volatility": self.cross_coin_volatility,
            "bullish_ratio": self.bullish_ratio,
            "trend_bias": self.trend_bias,
        }


def _trend_bias(avg_change: float, bullish_ratio: float) -> TrendBias:
    if avg_change > 2 and bullish_ratio > 0.55:
        return "bullish"
    if avg_change < -2 and bullish_ratio < 0.45:
        return "bearish"
    if avg_change > 1 and bullish_ratio > 0.5:
        return "bullish"
    if avg_change < -1 and bullish_ratio < 0.5:
        return "bearish"
    return "neutral"


def classify_regime(changes_24h: Sequence[float]) -> RegimeState:
    """Backtest entry point — delegates to the canonical 6-class
    classifier in `app.regime` so live, training, and backtest cannot
    drift. `changes_24h` is the per-coin 24h percent change vector at
    the evaluation timestamp.
    """
    changes = [c for c in changes_24h if c is not None and not math.isnan(c)]
    if not changes:
        return RegimeState("range_chop", 0.5, 0.0, 0.0, 0.5, "neutral")

    avg_change = sum(changes) / len(changes)
    if len(changes) > 1:
        var = sum((c - avg_change) ** 2 for c in changes) / (len(changes) - 1)
        cross_coin_vol = math.sqrt(var)
    else:
        cross_coin_vol = 0.0

    bullish = sum(1 for c in changes if c > 1)
    bullish_ratio = bullish / len(changes)

    decision: RegimeDecisionV2 = classify_regime_v2_from_basket(changes, cross_coin_vol)
    return RegimeState(
        regime=decision.label,
        confidence=decision.confidence,
        avg_change_24h=avg_change,
        cross_coin_volatility=cross_coin_vol,
        bullish_ratio=bullish_ratio,
        trend_bias=_trend_bias(avg_change, bullish_ratio),
    )


def regime_at(timestamp_ms: int, basket_changes: dict[int, list[float]]) -> RegimeState:
    """Look up the precomputed basket 24h change vector at-or-before
    `timestamp_ms` and classify with the canonical 6-class classifier.
    """
    if not basket_changes:
        return RegimeState("range_chop", 0.5, 0.0, 0.0, 0.5, "neutral")
    keys = sorted(basket_changes.keys())
    chosen = None
    for k in keys:
        if k <= timestamp_ms:
            chosen = k
        else:
            break
    if chosen is None:
        return RegimeState("range_chop", 0.5, 0.0, 0.0, 0.5, "neutral")
    return classify_regime(basket_changes[chosen])
