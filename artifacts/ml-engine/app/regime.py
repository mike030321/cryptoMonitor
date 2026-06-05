"""Phase 2 — first-class regime classifier.

Outputs one of six regimes:
  - trending_up           strong directional uptrend
  - trending_down         strong directional downtrend
  - range_chop            no trend, normal-vol mean reversion
  - high_vol_breakout     vol expansion + directional break
  - low_vol_compression   vol contraction (Bollinger squeeze)
  - panic_liquidation     extreme drawdown + vol spike

Two entry points share the same taxonomy so live, training, and backtest
all agree on the label set:

  classify_regime_from_features(feature_vec) -> RegimeLabel
      Per-coin per-timeframe regime computed from the canonical feature
      vector built by `app.features.build_feature_vector` (Phase 1
      shared service). Used to stamp every candle / journal row.

  classify_regime_from_basket(changes_24h) -> RegimeLabel
      Market-wide regime from per-coin 24h % changes. Used by the live
      monitor cycle so the same 6-class label can be applied to the
      portfolio-level state surfaced to the trader/UI.

Rules are intentionally simple and deterministic — interpretable, no
training data needed for Phase 2. The downstream models in Phases 3-5
will learn per-regime conditional structure on top of these labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

RegimeLabel = Literal[
    "trending_up",
    "trending_down",
    "range_chop",
    "high_vol_breakout",
    "low_vol_compression",
    "panic_liquidation",
]

REGIME_LABELS: tuple[RegimeLabel, ...] = (
    "trending_up",
    "trending_down",
    "range_chop",
    "high_vol_breakout",
    "low_vol_compression",
    "panic_liquidation",
)


@dataclass(frozen=True)
class RegimeDecision:
    label: RegimeLabel
    confidence: float
    # Inputs the rule used; logged so an operator can see *why* a label fired.
    inputs: dict


def _safe(v: Optional[float], default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    return float(v)


def classify_regime_from_features(features: dict) -> RegimeDecision:
    """Per-coin per-timeframe regime from a Phase-1 feature vector.

    Inputs used (all per-coin, normalized so they're comparable across
    pairs of any base price):
      - emaSpreadPct      EMA9 vs EMA21 spread, % of price (trend strength)
      - distFromEma21Pct  distance from EMA21 (trend persistence)
      - atrPct            ATR/price (volatility regime)
      - bbWidthPct        Bollinger band width / mid (compression)
      - ret5, ret10       5- and 10-bar returns (drawdown / breakout)
      - macdHist          MACD histogram (trend acceleration)
      - rsi14             RSI-14 (overbought/oversold)
    """
    if not features:
        return RegimeDecision("range_chop", 0.0, {"reason": "empty_features"})

    ema_spread = _safe(features.get("emaSpreadPct"))
    dist_ema21 = _safe(features.get("distFromEma21Pct"))
    atr_pct = _safe(features.get("atrPct"))
    bb_width = _safe(features.get("bbWidthPct"))
    ret5 = _safe(features.get("ret5"))
    ret10 = _safe(features.get("ret10"))
    macd_hist = _safe(features.get("macdHist"))
    rsi14 = _safe(features.get("rsi14"), 50.0)

    inputs = {
        "emaSpreadPct": ema_spread,
        "distFromEma21Pct": dist_ema21,
        "atrPct": atr_pct,
        "bbWidthPct": bb_width,
        "ret5": ret5,
        "ret10": ret10,
        "macdHist": macd_hist,
        "rsi14": rsi14,
    }

    # 1) PANIC_LIQUIDATION — extreme drawdown + vol spike (overrides others).
    #    Threshold: 5-bar return <= -3% AND ATR%>1.5% — captures cascades.
    if ret5 <= -3.0 and atr_pct >= 1.5:
        confidence = min(0.99, 0.6 + abs(ret5) / 20 + atr_pct / 20)
        return RegimeDecision("panic_liquidation", confidence, inputs)

    # 2) HIGH_VOL_BREAKOUT — vol expansion (BB width > 4%) with a strong
    #    5-bar move in either direction (>= 2%). Direction is implied by
    #    the rule but the regime label is just "breakout"; trending labels
    #    handle steady directional regimes.
    if bb_width >= 4.0 and abs(ret5) >= 2.0:
        confidence = min(0.95, 0.55 + bb_width / 20 + abs(ret5) / 20)
        return RegimeDecision("high_vol_breakout", confidence, inputs)

    # 3) LOW_VOL_COMPRESSION — Bollinger squeeze (width < 1%) AND ATR%
    #    contracted (< 0.4%) AND no significant drift.
    if bb_width < 1.0 and atr_pct < 0.4 and abs(ret10) < 1.5:
        confidence = min(0.95, 0.6 + (1.0 - bb_width) / 4 + (0.4 - atr_pct) / 2)
        return RegimeDecision("low_vol_compression", confidence, inputs)

    # 4/5) Trend up / down — EMA spread + distance + ret10 agree on sign,
    #      with MACD histogram confirming acceleration.
    trend_score = 0.0
    if ema_spread > 0.3:
        trend_score += 1
    if ema_spread > 1.0:
        trend_score += 1
    if dist_ema21 > 0.5:
        trend_score += 1
    if ret10 > 1.0:
        trend_score += 1
    if macd_hist > 0:
        trend_score += 1
    if rsi14 > 55:
        trend_score += 1

    down_score = 0.0
    if ema_spread < -0.3:
        down_score += 1
    if ema_spread < -1.0:
        down_score += 1
    if dist_ema21 < -0.5:
        down_score += 1
    if ret10 < -1.0:
        down_score += 1
    if macd_hist < 0:
        down_score += 1
    if rsi14 < 45:
        down_score += 1

    if trend_score >= 4 and trend_score > down_score:
        confidence = min(0.95, 0.5 + trend_score / 12)
        return RegimeDecision("trending_up", confidence, inputs)
    if down_score >= 4 and down_score > trend_score:
        confidence = min(0.95, 0.5 + down_score / 12)
        return RegimeDecision("trending_down", confidence, inputs)

    # 6) Default — range / chop.
    confidence = max(
        0.3,
        min(0.85, 0.5 + (1.0 - min(abs(ret10), 5.0) / 5.0) * 0.3),
    )
    return RegimeDecision("range_chop", confidence, inputs)


def classify_regime_from_basket(
    changes_24h: Sequence[float],
    cross_coin_vol: Optional[float] = None,
) -> RegimeDecision:
    """Market-wide regime from a basket of per-coin 24h % changes.

    Used by the live cycle when no single per-coin feature vector is
    available (e.g. for stamping `price_history` rows when a per-coin
    feature set hasn't been computed yet on this cycle).
    """
    changes = [c for c in changes_24h if c is not None and not math.isnan(c)]
    if not changes:
        return RegimeDecision("range_chop", 0.0, {"reason": "empty_basket"})

    avg = sum(changes) / len(changes)
    if cross_coin_vol is None:
        if len(changes) > 1:
            var = sum((c - avg) ** 2 for c in changes) / (len(changes) - 1)
            cross_coin_vol = math.sqrt(var)
        else:
            cross_coin_vol = 0.0
    abs_avg = sum(abs(c) for c in changes) / len(changes)
    bullish_ratio = sum(1 for c in changes if c > 1) / len(changes)
    bearish_ratio = sum(1 for c in changes if c < -1) / len(changes)

    inputs = {
        "avgChange24h": avg,
        "absAvg24h": abs_avg,
        "crossCoinVol": cross_coin_vol,
        "bullishRatio": bullish_ratio,
        "bearishRatio": bearish_ratio,
    }

    # Panic — broad sell-off with high cross-coin vol.
    if avg <= -5 and bearish_ratio >= 0.7 and cross_coin_vol >= 4:
        return RegimeDecision(
            "panic_liquidation",
            min(0.99, 0.6 + abs(avg) / 20),
            inputs,
        )
    # Vol-expansion breakout — high cross-coin vol with a directional bias.
    if cross_coin_vol >= 6 and abs_avg >= 4:
        return RegimeDecision(
            "high_vol_breakout",
            min(0.95, 0.55 + cross_coin_vol / 20),
            inputs,
        )
    # Low-vol compression — calm market.
    if abs_avg < 1.0 and cross_coin_vol < 1.5:
        return RegimeDecision(
            "low_vol_compression",
            min(0.9, 0.6 + (1.0 - abs_avg) / 3),
            inputs,
        )
    if avg > 2 and bullish_ratio > 0.6:
        return RegimeDecision(
            "trending_up",
            min(0.95, 0.5 + avg / 10),
            inputs,
        )
    if avg < -2 and bearish_ratio > 0.6:
        return RegimeDecision(
            "trending_down",
            min(0.95, 0.5 + abs(avg) / 10),
            inputs,
        )
    return RegimeDecision(
        "range_chop",
        min(0.85, 0.5 + (1 - min(abs_avg, 5) / 5) * 0.3),
        inputs,
    )


# Mapping from the legacy 4-class taxonomy (bull/bear/sideways/volatile)
# used by older paths. Live trader callers can fall back to the 6-class
# label produced by the basket classifier; this map exists only for
# backfill of historical rows whose only stored signal is the 4-class.
LEGACY_TO_NEW: dict[str, RegimeLabel] = {
    "bull": "trending_up",
    "bear": "trending_down",
    "sideways": "range_chop",
    "volatile": "high_vol_breakout",
}


def map_legacy_label(legacy: Optional[str]) -> Optional[RegimeLabel]:
    if legacy is None:
        return None
    return LEGACY_TO_NEW.get(legacy)
