"""Pure feature-engineering functions.

Mirrors `artifacts/api-server/src/lib/pattern-analyzer.ts` for the indicators
the LLM brain currently sees, so Phase 2's LightGBM model trains on a
superset of what the LLMs already use. All functions are pure and take
plain lists/arrays so they are trivially unit-testable.

Indicator parameters match the TypeScript side:
- RSI(14)
- MACD(12, 26, 9)
- ATR(14) (close-only synthesised TR, same approximation as Node)
- EMA(9) / EMA(21)
- Bollinger Bands(20, 2σ)
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Iterable, Optional

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "6h": 21_600_000,
    "1d": 86_400_000,
}

MIN_CANDLES_FOR_FEATURES = 35  # MACD(12,26,9) signal needs >=35 candles.


class CadenceMismatchError(ValueError):
    """Raised by `resample_to_candles` when the input row stream's native
    cadence is coarser than the requested bucket width — i.e. resampling
    daily-cadence rows into 5m buckets would silently corrupt the bucket
    closes. See `artifacts/ml-engine/reports/20260423T000000Z-schema-audit.md`
    (task #315 / fix #317) for the full failure mode.
    """


def quarantine_rows_by_cadence(
    rows: list[tuple[int, float]],
    min_input_cadence_ms: int,
) -> list[tuple[int, float]]:
    """Task #317 — single source of truth for cadence quarantine.

    A row is kept iff EITHER the gap to the previous row OR the gap to
    the next row is within `min_input_cadence_ms`. This lets fine-cadence
    clusters survive while orphan coarse rows (whose only neighbour gaps
    exceed the cap, including a coarse row at index 0) are dropped.

    Both `resample_to_candles` and the trainer's bucket-start
    reconstruction in `app.training.labels` consume this function so the
    two streams cannot diverge — divergence would manifest as a
    `len(bucket_starts) != len(closes)` assertion in the trainer.
    """
    if min_input_cadence_ms <= 0 or len(rows) < 2:
        return list(rows)
    keep_flags = [False] * len(rows)
    for i in range(len(rows)):
        prev_ok = i > 0 and (rows[i][0] - rows[i - 1][0]) <= min_input_cadence_ms
        next_ok = i + 1 < len(rows) and (
            rows[i + 1][0] - rows[i][0]
        ) <= min_input_cadence_ms
        if prev_ok or next_ok:
            keep_flags[i] = True
    return [r for r, keep in zip(rows, keep_flags) if keep]


def resample_to_candles(
    ticks: Iterable[tuple[datetime, float]],
    bucket_ms: int,
    min_input_cadence_ms: Optional[int] = None,
) -> list[float]:
    """Resample raw ticks into per-bucket close prices.

    Matches `resampleToCandles` in pattern-analyzer.ts: each bucket is
    [floor(ts/bucket)*bucket, +bucket), close = last tick price in bucket,
    empty buckets are skipped. We never invent prices.

    `min_input_cadence_ms` (task #317 — cadence safety) caps the
    inter-arrival gap of consecutive INPUT rows. When set:
      * If every measurable inter-arrival gap is greater than the cap,
        the input stream is itself coarser than the requested bucket
        and `CadenceMismatchError` is raised — a daily-cadence row
        stream can never silently feed 5m or 1h buckets.
      * If only some rows arrive after a too-large gap, those individual
        rows are quarantined (skipped) before bucketing, so a single
        daily contaminant cannot become the close of a 5m bucket whose
        other contributors are fine-cadence ticks.
    """
    if bucket_ms <= 0:
        return []
    rows: list[tuple[int, float]] = []
    for ts, price in ticks:
        if not math.isfinite(price) or price <= 0:
            continue
        rows.append((int(ts.timestamp() * 1000), price))

    if min_input_cadence_ms is not None and min_input_cadence_ms > 0 and len(rows) >= 2:
        gaps = [rows[i][0] - rows[i - 1][0] for i in range(1, len(rows))]
        valid_gaps = sum(1 for g in gaps if g <= min_input_cadence_ms)
        if valid_gaps == 0:
            raise CadenceMismatchError(
                f"input row cadence is coarser than min_input_cadence_ms="
                f"{min_input_cadence_ms}ms (every inter-arrival gap exceeded "
                f"the cap; observed gaps: {gaps[:5]}{'...' if len(gaps) > 5 else ''})"
            )
        rows = quarantine_rows_by_cadence(rows, min_input_cadence_ms)
        if len(rows) < 2:
            raise CadenceMismatchError(
                f"input row cadence collapsed to {len(rows)} rows after "
                f"quarantining gaps > min_input_cadence_ms="
                f"{min_input_cadence_ms}ms"
            )

    closes: list[float] = []
    current_bucket = -1
    last_price = 0.0
    for ts_ms, price in rows:
        bucket = (ts_ms // bucket_ms) * bucket_ms
        if bucket != current_bucket:
            if current_bucket != -1:
                closes.append(last_price)
            current_bucket = bucket
        last_price = price
    if current_bucket != -1:
        closes.append(last_price)
    return closes


def ema_series(prices: list[float], period: int) -> list[float]:
    if not prices:
        return []
    k = 2.0 / (period + 1)
    out = [prices[0]]
    for p in prices[1:]:
        out.append(p * k + out[-1] * (1 - k))
    return out


def rsi_series(prices: list[float], period: int = 14) -> list[float]:
    """Per-bar Wilder RSI series.

    ``out[i]`` equals ``rsi(prices[:i+1], period)`` for every ``i``. The
    full series is computed in O(N) — without this the batched feature
    builder would still pay O(N²) cost across the per-bar slice.
    """
    n = len(prices)
    out = [50.0] * n
    if n < period + 1:
        return out
    changes = [prices[i] - prices[i - 1] for i in range(1, n)]
    avg_gain = sum(c for c in changes[:period] if c > 0) / period
    avg_loss = sum(-c for c in changes[:period] if c < 0) / period
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + (avg_gain / avg_loss))
    for k in range(period + 1, n):
        c = changes[k - 1]
        gain = c if c > 0 else 0.0
        loss = -c if c < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out[k] = 100.0
        else:
            out[k] = 100.0 - 100.0 / (1.0 + (avg_gain / avg_loss))
    return out


def atr_series(prices: list[float], period: int = 14) -> list[float]:
    """Per-bar ATR(period) series matching ``atr(prices[:i+1], period)``."""
    n = len(prices)
    out = [0.0] * n
    if n < period + 1:
        return out
    trs: list[float] = []
    for i in range(1, n):
        prev_close = prices[i - 1]
        curr_close = prices[i]
        est_high = max(curr_close, prev_close) * (1 + 0.001)
        est_low = min(curr_close, prev_close) * (1 - 0.001)
        tr = max(
            est_high - est_low,
            abs(est_high - prev_close),
            abs(est_low - prev_close),
        )
        trs.append(tr)
    a = sum(trs[:period]) / period
    out[period] = a
    for k in range(period + 1, n):
        tr = trs[k - 1]
        a = (a * (period - 1) + tr) / period
        out[k] = a
    return out


def bollinger_series(
    prices: list[float], period: int = 20, k: float = 2.0,
) -> dict[str, list[float]]:
    """Per-bar Bollinger band series matching ``bollinger(prices[:i+1])``.

    O(N × period) — for ``period=20`` this is effectively linear in N and
    avoids the O(N²) per-call slicing the legacy code path triggered.
    """
    n = len(prices)
    upper = [0.0] * n
    middle = [0.0] * n
    lower = [0.0] * n
    width = [0.0] * n
    pct_b = [0.5] * n
    for i in range(n):
        if i + 1 < period:
            last = prices[i]
            upper[i] = last
            middle[i] = last
            lower[i] = last
            width[i] = 0.0
            pct_b[i] = 0.5
            continue
        window = prices[i - period + 1 : i + 1]
        mid = sum(window) / period
        var = sum((p - mid) ** 2 for p in window) / (period - 1)
        sd = math.sqrt(var)
        up = mid + k * sd
        lo = mid - k * sd
        w = up - lo
        last = prices[i]
        upper[i] = up
        middle[i] = mid
        lower[i] = lo
        width[i] = w
        pct_b[i] = (last - lo) / w if w > 0 else 0.5
    return {
        "upper": upper, "middle": middle, "lower": lower,
        "width": width, "pctB": pct_b,
    }


def rsi(prices: list[float], period: int = 14) -> float:
    """Wilder's RSI matching the TypeScript implementation."""
    if len(prices) < period + 1:
        return 50.0
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    avg_gain = sum(c for c in changes[:period] if c > 0) / period
    avg_loss = sum(-c for c in changes[:period] if c < 0) / period
    for c in changes[period:]:
        gain = c if c > 0 else 0.0
        loss = -c if c < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(prices: list[float]) -> dict[str, float]:
    if len(prices) < 26:
        return {"line": 0.0, "signal": 0.0, "hist": 0.0}
    ema12 = ema_series(prices, 12)
    ema26 = ema_series(prices, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal = ema_series(macd_line, 9)
    line = macd_line[-1]
    sig = signal[-1]
    return {"line": line, "signal": sig, "hist": line - sig}


def atr(prices: list[float], period: int = 14) -> float:
    """Close-only ATR approximation matching pattern-analyzer.ts.

    The TS code synthesises a high/low band of ±0.1% around adjacent closes.
    We replicate it exactly so Phase 2's LightGBM features are bit-comparable
    to what the LLM brain saw historically.
    """
    if len(prices) < period + 1:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(prices)):
        prev_close = prices[i - 1]
        curr_close = prices[i]
        est_high = max(curr_close, prev_close) * (1 + 0.001)
        est_low = min(curr_close, prev_close) * (1 - 0.001)
        tr = max(
            est_high - est_low,
            abs(est_high - prev_close),
            abs(est_low - prev_close),
        )
        trs.append(tr)
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def bollinger(prices: list[float], period: int = 20, k: float = 2.0) -> dict[str, float]:
    if len(prices) < period:
        last = prices[-1] if prices else 0.0
        return {"upper": last, "middle": last, "lower": last, "width": 0.0, "pctB": 0.5}
    window = prices[-period:]
    middle = sum(window) / period
    var = sum((p - middle) ** 2 for p in window) / (period - 1)
    sd = math.sqrt(var)
    upper = middle + k * sd
    lower = middle - k * sd
    width = upper - lower
    last = prices[-1]
    pct_b = (last - lower) / width if width > 0 else 0.5
    return {"upper": upper, "middle": middle, "lower": lower, "width": width, "pctB": pct_b}


def realized_volatility(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return math.sqrt(sum(r * r for r in returns) / len(returns))


# Phase 5 — fixed news-tag vocabulary. Must stay in lock-step with the
# TypeScript classifier in artifacts/api-server/src/lib/news-classifier.ts.
# Order matters: it determines the column order in the feature vector so
# trained models see a stable schema. Adding a tag is forward-compatible
# (new column at the end); removing or reordering breaks every model.
# Order MUST match TS exactly so the column order in the model row stays
# stable across both sides. Source of truth lives in
# artifacts/api-server/src/lib/news-classifier.ts (TAG_VOCABULARY).
# `tests/test_news_tag_vocab_parity.py` enforces this with a parity check.
NEWS_TAG_VOCABULARY = (
    "etf_flow",
    "regulatory_risk",
    "exchange_outage",
    "macro_shock",
    "whale_move",
    "exploit_or_hack",
    "stablecoin_event",
    "narrative_rotation",
    "listing_or_delisting",
    "protocol_upgrade",
    "high_volume_breakout",
    "unusual_volatility",
)


def news_tag_features(tags: Optional[Iterable[str]]) -> dict[str, float]:
    """Convert a list of news tags into a stable one-hot feature dict.

    Unknown tags are silently dropped. Missing/empty input yields all-zero
    features so the column set is always present in the model row.
    """
    seen = set(tags or ())
    return {f"news_{t}": 1.0 if t in seen else 0.0 for t in NEWS_TAG_VOCABULARY}


def build_feature_vector(
    closes: list[float],
    news_tags: Optional[Iterable[str]] = None,
) -> Optional[dict[str, float]]:
    """Compute the full feature vector from a list of close prices.

    `news_tags` (Phase 5) is the per-coin tag list emitted by the LLM news
    classifier. When supplied, the canonical news-tag one-hot columns are
    appended to the feature vector so the LightGBM model can learn from
    headline structure without depending on raw text. Pass None for the
    legacy price-only feature set.

    Returns None when there aren't enough candles to compute MACD reliably.
    """
    if len(closes) < MIN_CANDLES_FOR_FEATURES:
        return None
    last = closes[-1]
    returns_pct = [
        (closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))
    ]
    recent20 = returns_pct[-20:] if len(returns_pct) >= 20 else returns_pct
    rv = realized_volatility(recent20)

    rsi14 = rsi(closes, 14)
    m = macd(closes)
    a = atr(closes, 14)
    bb = bollinger(closes, 20, 2.0)
    ema9 = ema_series(closes, 9)[-1]
    ema21 = ema_series(closes, 21)[-1]
    ema_spread_pct = ((ema9 - ema21) / ema21) * 100 if ema21 > 0 else 0.0

    ret1 = returns_pct[-1]
    ret5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0.0
    ret10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0.0
    momentum = ret5 - (
        (closes[-6] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0.0
    )

    dist_from_ema9_pct = ((last - ema9) / ema9) * 100 if ema9 > 0 else 0.0
    dist_from_ema21_pct = ((last - ema21) / ema21) * 100 if ema21 > 0 else 0.0

    # Task #517 — `volZScore60` is the rolling 60-bar z-score of the
    # absolute 1-bar return at the current bar. Captures whether the
    # latest move is anomalous relative to the recent vol regime, which
    # gives the booster a bar-magnitude signal that's normalized across
    # coins / timeframes (so the same feature works on bonk@2h and
    # dogwifcoin@1d). Strictly point-in-time: the window only contains
    # absolute returns observed at-or-before the current bar.
    abs_returns = [abs(r) for r in returns_pct]
    vol_zscore_60 = 0.0
    if len(abs_returns) >= 10:
        window = abs_returns[-60:]
        if len(window) >= 10:
            mean_w = sum(window) / len(window)
            var_w = sum((x - mean_w) ** 2 for x in window) / (len(window) - 1)
            std_w = math.sqrt(var_w)
            if std_w > 1e-12:
                vol_zscore_60 = (abs_returns[-1] - mean_w) / std_w

    feats = {
        "candleCount": float(len(closes)),
        "lastPrice": last,
        "ret1": ret1,
        "ret5": ret5,
        "ret10": ret10,
        "momentum": momentum,
        "realizedVol": rv,
        "rsi14": rsi14,
        "macdLine": m["line"],
        "macdSignal": m["signal"],
        "macdHist": m["hist"],
        "atr14": a,
        "atrPct": (a / last) * 100 if last > 0 else 0.0,
        "ema9": ema9,
        "ema21": ema21,
        "emaSpreadPct": ema_spread_pct,
        "distFromEma9Pct": dist_from_ema9_pct,
        "distFromEma21Pct": dist_from_ema21_pct,
        "bbUpper": bb["upper"],
        "bbMiddle": bb["middle"],
        "bbLower": bb["lower"],
        "bbWidth": bb["width"],
        "bbPctB": bb["pctB"],
        "bbWidthPct": (bb["width"] / bb["middle"]) * 100 if bb["middle"] > 0 else 0.0,
        "volZScore60": vol_zscore_60,
    }
    # Task #365 — Quant-Only Enforcement.
    # Phase 5 used to append the LLM-derived news-tag one-hot block here.
    # That channel is now permanently shut at the FEATURE level (not just
    # zero-filled at the call site): models trained from this builder
    # will not even see the columns. The `news_tag_features()` helper is
    # retained above for any backwards-compatible utility (e.g. parquet
    # backfills) but is intentionally NOT called from the live path.
    # The `news_tags` parameter is preserved on the public signature so
    # legacy callers don't break, and is silently ignored.
    _ = news_tags  # quant-only contract; see registry.FORBIDDEN_FEATURE_PREFIXES
    return feats


def build_feature_vectors_for_series(
    closes: list[float],
    news_tags: Optional[Iterable[str]] = None,
) -> list[Optional[dict[str, float]]]:
    """Vectorized batch builder: produce the per-bar feature dict for every
    position ``k`` in ``closes`` such that ``closes[:k+1]`` would yield a
    full feature vector.

    Equivalent to::

        [build_feature_vector(closes[:k+1], news_tags) for k in range(len(closes))]

    but computed in O(N) instead of O(N²): the rolling indicators
    (EMA9/12/21/26, MACD signal, RSI, ATR, Bollinger bands, realized vol)
    are derived once across the whole series, and each per-bar dict reads
    the indicator value at index ``k``.

    The batch path keeps the trainer's 5m dataset build bounded as the
    available candle history grows past tens of thousands of bars; without
    it, the per-bar ``build_feature_vector(closes[:k+1])`` call site is
    the O(N²) hot loop that hung the full-campaign 5m phase on bonk.

    Indices where ``closes[:k+1]`` is shorter than
    ``MIN_CANDLES_FOR_FEATURES`` yield ``None``, mirroring the per-call
    short-circuit in :func:`build_feature_vector`.
    """
    n = len(closes)
    out: list[Optional[dict[str, float]]] = [None] * n
    if n < MIN_CANDLES_FOR_FEATURES:
        return out

    ema9_full = ema_series(closes, 9)
    ema12_full = ema_series(closes, 12)
    ema21_full = ema_series(closes, 21)
    ema26_full = ema_series(closes, 26)
    macd_line_full = [a - b for a, b in zip(ema12_full, ema26_full)]
    macd_signal_full = ema_series(macd_line_full, 9)
    rsi14_full = rsi_series(closes, 14)
    atr14_full = atr_series(closes, 14)
    bb = bollinger_series(closes, 20, 2.0)
    bb_upper = bb["upper"]
    bb_middle = bb["middle"]
    bb_lower = bb["lower"]
    bb_width = bb["width"]
    bb_pct_b = bb["pctB"]

    # Per-bar simple returns: returns_pct[k] is the return from closes[k]
    # to closes[k+1], so it has length n-1. Mirrors the comprehension in
    # the per-call ``build_feature_vector`` exactly (no fallback when the
    # divisor is zero — we keep the same behaviour and let an upstream
    # filter catch invalid prices).
    returns_pct: list[float] = [0.0] * (n - 1) if n > 1 else []
    for k in range(1, n):
        prev = closes[k - 1]
        returns_pct[k - 1] = (closes[k] - prev) / prev

    # Rolling realized vol over the last 20 returns ending at bar k.
    # Per-call code computes ``realized_volatility(returns_pct_inner[-20:])``
    # with ``returns_pct_inner = returns_pct[:k]`` (k entries for the slice
    # ``closes[:k+1]``). We replicate that by looking at returns_pct[k-w:k].
    rv_window = 20
    rv_full: list[float] = [0.0] * n
    for k in range(2, n):
        recent_count = min(k, rv_window)
        if recent_count < 2:
            continue
        sub = returns_pct[k - recent_count : k]
        rv_full[k] = math.sqrt(sum(r * r for r in sub) / recent_count)

    # Task #517 — rolling 60-bar z-score of |ret1| at each bar. Mirrors
    # the per-call branch in ``build_feature_vector`` exactly: the window
    # at bar k is the absolute returns observed over the (up to) last 60
    # bars ending at k inclusive, and the z-score normalizes the
    # bar-magnitude reading the booster sees against that local regime.
    # Strictly point-in-time — only past returns enter the window.
    abs_returns_full: list[float] = [abs(r) for r in returns_pct]
    vol_z_full: list[float] = [0.0] * n
    for k in range(1, n):
        end = k  # exclusive: returns_pct[k-1] is the bar-k return
        start = max(0, end - 60)
        window = abs_returns_full[start:end]
        if len(window) < 10:
            continue
        mean_w = sum(window) / len(window)
        var_w = sum((x - mean_w) ** 2 for x in window) / (len(window) - 1)
        std_w = math.sqrt(var_w)
        if std_w > 1e-12:
            vol_z_full[k] = (abs_returns_full[k - 1] - mean_w) / std_w

    # Quant-only contract: news_tags ignored (kept on signature for
    # backwards compatibility, mirroring ``build_feature_vector``).
    _ = news_tags

    for k in range(MIN_CANDLES_FOR_FEATURES - 1, n):
        last = closes[k]
        # ret1/ret5/ret10/momentum match the per-call expressions where
        # ``closes`` is the slice ``closes[:k+1]``: closes[-1]==closes[k],
        # closes[-6]==closes[k-5], closes[-11]==closes[k-10]. The per-call
        # short-circuit branches gate on ``len(closes) >= 6 / 11``.
        ret1 = returns_pct[k - 1]
        if k + 1 >= 6:
            ret5 = (closes[k] - closes[k - 5]) / closes[k - 5]
        else:
            ret5 = 0.0
        if k + 1 >= 11:
            ret10 = (closes[k] - closes[k - 10]) / closes[k - 10]
            momentum = ret5 - ((closes[k - 5] - closes[k - 10]) / closes[k - 10])
        else:
            ret10 = 0.0
            momentum = ret5

        # MACD per-call returns zeros when len(closes_inner) < 26. With
        # MIN_CANDLES_FOR_FEATURES==35 the loop only enters at k>=34, so
        # k+1>=35>=26 and the branch is never taken in practice — kept
        # explicit for safety if MIN_CANDLES_FOR_FEATURES is ever lowered.
        if k + 1 < 26:
            macd_l = 0.0
            macd_s = 0.0
        else:
            macd_l = macd_line_full[k]
            macd_s = macd_signal_full[k]
        macd_h = macd_l - macd_s

        e9 = ema9_full[k]
        e21 = ema21_full[k]
        ema_spread_pct = ((e9 - e21) / e21) * 100 if e21 > 0 else 0.0
        dist9 = ((last - e9) / e9) * 100 if e9 > 0 else 0.0
        dist21 = ((last - e21) / e21) * 100 if e21 > 0 else 0.0

        atr_k = atr14_full[k]
        bb_mid = bb_middle[k]
        bb_w = bb_width[k]

        out[k] = {
            "candleCount": float(k + 1),
            "lastPrice": last,
            "ret1": ret1,
            "ret5": ret5,
            "ret10": ret10,
            "momentum": momentum,
            "realizedVol": rv_full[k],
            "rsi14": rsi14_full[k],
            "macdLine": macd_l,
            "macdSignal": macd_s,
            "macdHist": macd_h,
            "atr14": atr_k,
            "atrPct": (atr_k / last) * 100 if last > 0 else 0.0,
            "ema9": e9,
            "ema21": e21,
            "emaSpreadPct": ema_spread_pct,
            "distFromEma9Pct": dist9,
            "distFromEma21Pct": dist21,
            "bbUpper": bb_upper[k],
            "bbMiddle": bb_mid,
            "bbLower": bb_lower[k],
            "bbWidth": bb_w,
            "bbPctB": bb_pct_b[k],
            "bbWidthPct": (bb_w / bb_mid) * 100 if bb_mid > 0 else 0.0,
            "volZScore60": vol_z_full[k],
        }
    return out


def feature_hash(features: dict[str, float]) -> str:
    """Stable short hash of a feature vector for log auditing."""
    payload = json.dumps(
        {k: round(v, 10) for k, v in sorted(features.items())},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
