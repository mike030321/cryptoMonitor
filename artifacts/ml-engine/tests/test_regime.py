"""Tests for the 6-class regime classifier.

Pins the boundaries between the 6 labels for both entry points
(`classify_regime_from_features`, `classify_regime_from_basket`) so
threshold tweaks can't silently shift the label distribution that
live, training, and backtest all depend on.
"""
from __future__ import annotations

from collections import Counter

import pytest

from app.regime import (
    REGIME_LABELS,
    classify_regime_from_basket,
    classify_regime_from_features,
)


# ---------------------------------------------------------------------------
# classify_regime_from_features — one fixture per label
# ---------------------------------------------------------------------------

FEATURE_FIXTURES: dict[str, dict] = {
    "panic_liquidation": {
        # ret5 <= -3% AND atrPct >= 1.5% — overrides everything else.
        "emaSpreadPct": -2.0,
        "distFromEma21Pct": -3.0,
        "atrPct": 2.5,
        "bbWidthPct": 6.0,
        "ret5": -5.0,
        "ret10": -8.0,
        "macdHist": -0.5,
        "rsi14": 20.0,
    },
    "high_vol_breakout": {
        # bbWidth >= 4 AND |ret5| >= 2, but ret5 above panic threshold.
        "emaSpreadPct": 0.5,
        "distFromEma21Pct": 0.4,
        "atrPct": 1.0,
        "bbWidthPct": 5.0,
        "ret5": 2.5,
        "ret10": 1.2,
        "macdHist": 0.1,
        "rsi14": 58.0,
    },
    "low_vol_compression": {
        # bbWidth < 1 AND atrPct < 0.4 AND |ret10| < 1.5.
        "emaSpreadPct": 0.05,
        "distFromEma21Pct": 0.05,
        "atrPct": 0.2,
        "bbWidthPct": 0.5,
        "ret5": 0.1,
        "ret10": 0.3,
        "macdHist": 0.0,
        "rsi14": 50.0,
    },
    "trending_up": {
        # All 6 trend signals fire (score == 6, well above the >=4 cutoff).
        "emaSpreadPct": 1.5,
        "distFromEma21Pct": 1.2,
        "atrPct": 0.8,
        "bbWidthPct": 2.0,
        "ret5": 1.5,
        "ret10": 2.5,
        "macdHist": 0.4,
        "rsi14": 62.0,
    },
    "trending_down": {
        # Mirror image of trending_up — but stay above panic ret5 threshold.
        "emaSpreadPct": -1.5,
        "distFromEma21Pct": -1.2,
        "atrPct": 0.8,
        "bbWidthPct": 2.0,
        "ret5": -1.5,
        "ret10": -2.5,
        "macdHist": -0.4,
        "rsi14": 38.0,
    },
    "range_chop": {
        # Nothing fires: low directional signal, normal vol, no compression.
        "emaSpreadPct": 0.1,
        "distFromEma21Pct": 0.1,
        "atrPct": 0.6,
        "bbWidthPct": 1.5,
        "ret5": 0.2,
        "ret10": 0.3,
        "macdHist": 0.0,
        "rsi14": 50.0,
    },
}


@pytest.mark.parametrize("expected_label,features", list(FEATURE_FIXTURES.items()))
def test_classify_regime_from_features_label(expected_label, features):
    decision = classify_regime_from_features(features)
    assert decision.label == expected_label
    assert 0.0 <= decision.confidence <= 0.99
    # Every input the rule looked at must be echoed for operator debugging.
    for k in features:
        assert k in decision.inputs


def test_classify_regime_from_features_empty():
    decision = classify_regime_from_features({})
    assert decision.label == "range_chop"
    assert decision.confidence == 0.0


def test_classify_regime_from_features_covers_every_label():
    # Guard: if a new label is added to REGIME_LABELS, this fixture set
    # must be expanded so the snapshot test below stays meaningful.
    assert set(FEATURE_FIXTURES.keys()) == set(REGIME_LABELS)


# ---------------------------------------------------------------------------
# classify_regime_from_basket — one fixture per label
# ---------------------------------------------------------------------------

BASKET_FIXTURES: dict[str, tuple[list[float], float | None]] = {
    # avg <= -5 AND bearish_ratio >= 0.7 AND cross_coin_vol >= 4.
    "panic_liquidation": ([-12.0, -10.0, -8.0, -6.0, -5.0, -4.0, -1.0], 4.5),
    # cross_coin_vol >= 6 AND abs_avg >= 4, but doesn't satisfy panic rule.
    "high_vol_breakout": ([8.0, 7.0, -6.0, 5.0, -4.0, 6.0], 6.5),
    # abs_avg < 1 AND cross_coin_vol < 1.5.
    "low_vol_compression": ([0.2, -0.3, 0.1, -0.1, 0.4, -0.2], 0.5),
    # avg > 2 AND bullish_ratio > 0.6.
    "trending_up": ([3.0, 4.0, 2.5, 3.5, 0.5, 1.2], None),
    # avg < -2 AND bearish_ratio > 0.6.
    "trending_down": ([-3.0, -4.0, -2.5, -3.5, -0.5, -1.2], None),
    # Mixed and small — no rule fires.
    "range_chop": ([1.5, -1.5, 0.5, -0.5, 1.0, -1.0], None),
}


@pytest.mark.parametrize("expected_label,payload", list(BASKET_FIXTURES.items()))
def test_classify_regime_from_basket_label(expected_label, payload):
    changes, vol = payload
    decision = classify_regime_from_basket(changes, cross_coin_vol=vol)
    assert decision.label == expected_label
    assert 0.0 <= decision.confidence <= 0.99


def test_classify_regime_from_basket_empty():
    decision = classify_regime_from_basket([])
    assert decision.label == "range_chop"
    assert decision.confidence == 0.0


def test_classify_regime_from_basket_covers_every_label():
    assert set(BASKET_FIXTURES.keys()) == set(REGIME_LABELS)


# ---------------------------------------------------------------------------
# Snapshot / regression test — fixed basket -> fixed label distribution.
# Any threshold tweak that shifts a coin across a boundary will break this
# test, forcing a deliberate update instead of a silent regime drift.
# ---------------------------------------------------------------------------

SNAPSHOT_BASKET_24H: list[float] = [
    # 16-coin basket spanning the realistic range we see live: deep
    # drawdowns, modest pumps, sideways noise, a couple of dead-quiet coins.
    -12.0, -8.5, -6.0, -3.2, -2.1, -1.4, -0.8, -0.2,
    0.1, 0.6, 1.3, 2.4, 3.1, 4.8, 7.5, 11.0,
]


def _featureise(change_24h: float) -> dict:
    """Map a single 24h % change into a minimal feature vector.

    Just enough of the rule inputs to deterministically land each coin in
    one of the 6 buckets — keeps the snapshot test interpretable without
    depending on the live feature builder.
    """
    abs_change = abs(change_24h)
    # Volatility and band width grow with the magnitude of the move.
    atr_pct = max(0.1, abs_change / 5.0)
    bb_width = max(0.3, abs_change / 2.0)
    ret5 = change_24h / 2.0
    ret10 = change_24h
    sign = 1.0 if change_24h >= 0 else -1.0
    return {
        "emaSpreadPct": sign * min(2.0, abs_change / 5.0),
        "distFromEma21Pct": sign * min(2.0, abs_change / 4.0),
        "atrPct": atr_pct,
        "bbWidthPct": bb_width,
        "ret5": ret5,
        "ret10": ret10,
        "macdHist": sign * min(1.0, abs_change / 8.0),
        "rsi14": 50.0 + sign * min(25.0, abs_change * 2.5),
    }


# Pinned label distribution for SNAPSHOT_BASKET_24H. If you change a
# threshold in app/regime.py and this assertion fails, that is the test
# doing its job — review the new distribution and update intentionally.
EXPECTED_SNAPSHOT_DISTRIBUTION: dict[str, int] = {
    "panic_liquidation": 2,
    "trending_down": 3,
    "low_vol_compression": 6,
    "trending_up": 4,
    "high_vol_breakout": 1,
    "range_chop": 0,
}


def test_classifier_snapshot_distribution():
    counts = Counter(
        classify_regime_from_features(_featureise(c)).label
        for c in SNAPSHOT_BASKET_24H
    )
    # Compare full distribution including zero-count labels so a label
    # appearing where it shouldn't is also caught.
    actual = {label: counts.get(label, 0) for label in REGIME_LABELS}
    expected = {label: EXPECTED_SNAPSHOT_DISTRIBUTION.get(label, 0) for label in REGIME_LABELS}
    if actual != expected:
        diff_lines = []
        for label in REGIME_LABELS:
            old = expected[label]
            new = actual[label]
            if old != new:
                diff_lines.append(f"  {label:>22}: {old} -> {new} (delta {new - old:+d})")
        msg = (
            "\nRegime label distribution drifted on the pinned 16-coin snapshot.\n"
            "This CI gate exists so threshold tweaks in app/regime.py can't\n"
            "silently shift the live regime mix.\n\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n\n"
            "Per-label change:\n"
            + "\n".join(diff_lines)
            + "\n\nIf the new distribution is intentional, update\n"
            "EXPECTED_SNAPSHOT_DISTRIBUTION in artifacts/ml-engine/tests/test_regime.py\n"
            "in the same PR that changes the thresholds, and call out the shift\n"
            "in the PR description so reviewers can sanity-check it.\n"
        )
        raise AssertionError(msg)
    assert sum(actual.values()) == len(SNAPSHOT_BASKET_24H)
