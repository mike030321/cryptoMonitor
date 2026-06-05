"""Label + feature-frame generation for Phase-2 training.

For each (coin, timeframe) we:
1. Pull real ticks from `price_history` (synthetic excluded at SQL).
2. Resample to per-bucket close prices using Phase-1's `resample_to_candles`.
3. At each candle index i >= MIN_CANDLES_FOR_FEATURES with i+1 < N, compute
   the feature vector from closes[:i+1] (no look-ahead) and label using the
   forward 1-bar return: r = (closes[i+1] - closes[i]) / closes[i].
4. Three-class label uses `LABEL_THRESHOLDS_PERCENT[timeframe]` (the
   training-only band, NOT the adjudication outcome threshold):
   - up    if r >  +thr/100
   - down  if r <  -thr/100
   - stable otherwise

Why two thresholds? The adjudication band in
`outcome_thresholds_percent` is set strictly above the round-trip cost
(0.30%) so a "correct" prediction is also net-profitable after fees.
But on real intraday data, almost no 1m/5m bar moves that much — so
labeling at the adjudication band collapses 88%+ of rows to STABLE and
the model never learns directional structure. The training-only
`label_thresholds_percent` is intentionally LOWER so the up/down
classes carry meaningful expected-return mass; adjudication is
unchanged. Task #95.

Returns a pandas DataFrame so the training loop can pool across coins.
The DataFrame includes a `timestamp_ms` column (the close-time of the
feature candle) so the walk-forward splitter can slice by time, never by
shuffled row index.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..db import close_pool, fetch_real_ticks, init_pool
from ..features import (
    CadenceMismatchError,
    MIN_CANDLES_FOR_FEATURES,
    TIMEFRAME_MS,
    build_feature_vectors_for_series,
    resample_to_candles,
)

# Task #317 — timeframes for which we ALWAYS prefer reading native-cadence
# candles directly from `price_candles` over resampling raw ticks. The 1m
# slice is the special case: no live source emits genuine 1m bars in our
# universe, so 1m always resamples from the 60s live-poll tick stream.
CANDLES_PREFERRED_TIMEFRAMES = {"5m", "1h", "2h", "6h", "1d"}

# Per-bucket inter-arrival cap passed into `resample_to_candles` to prevent
# a coarser-cadence row (e.g. a daily bar accidentally written to
# `price_history`) from contaminating a fine-cadence bucket close. The cap
# equals the bucket width itself: any row whose entry gap exceeds the
# bucket cannot legitimately have come from a same-cadence stream.
def _resample_cadence_cap_ms(timeframe: str) -> int:
    return TIMEFRAME_MS.get(timeframe, 0)


from ..regime import classify_regime_from_features
from ..backtest.contract import get_frictions

logger = logging.getLogger(__name__)

# Forward window (in candles) used to compute trade-aware labels — TP/SL
# adjudication, MAE/MFE, multi-bar return, and the cost-aware
# `prob_move_gt_cost` flag. Phase 3 chose 4 bars: long enough that an
# average ATR-based TP/SL has a real chance to hit, short enough that
# every label row still resolves (no left-censoring) and the existing
# per-(coin,timeframe) row count is preserved.
FORWARD_HORIZON_CANDLES = 4

# Adjudication thresholds — kept as a fall-back and to expose the value
# tests / external callers expect for the "outcome" view. Mirrors
# artifacts/api-server/src/lib/trading-constants.ts:OUTCOME_THRESHOLDS_PERCENT.
OUTCOME_THRESHOLDS_PERCENT: dict[str, float] = {
    "1m": 0.35,
    "5m": 0.35,
    "1h": 0.45,
    "2h": 0.55,
    "6h": 0.85,
    "1d": 1.50,
}


# Hardcoded mirrors of `shared/trading-frictions.json`. Used ONLY when the
# JSON file is unreachable (e.g. trainer worker process started without the
# workspace mounted). Drift between these mirrors and the JSON is a live
# correctness bug — when both sources are visible, `_load_*` checks the
# mirror equals the JSON and logs a loud WARN if it does not. When the JSON
# is unreachable we log a single visible WARN and flip
# `LABEL_THRESHOLDS_FALLBACK_STATUS["used_fallback"] = True` so the trainer
# entry point / metrics scraper can surface the drift risk in production.
# See task #357.
_LABEL_THRESHOLDS_MIRROR: dict[str, float] = {
    "1m": 0.03, "5m": 0.10, "1h": 0.20,
    "2h": 0.28, "6h": 0.45, "1d": 0.80,
}
_PER_COIN_LABEL_THRESHOLDS_MIRROR: dict[str, dict[str, float]] = {
    "bonk":                    {"5m": 0.22},
    "celestia":                {"5m": 0.21},
    "dogwifcoin":              {"5m": 0.24},
    "floki-inu":               {"5m": 0.04},
    "injective-protocol":      {"5m": 0.04},
    "jupiter-exchange-solana": {"5m": 0.04},
    "pepe":                    {"5m": 0.20},
    "render-token":            {"5m": 0.04},
    "sei-network":             {"5m": 0.04},
    "worldcoin-wld":           {"5m": 0.24},
}

# Module-level fallback status. Scraped by `app.training.metrics` so an
# operator can alert when a worker silently regressed to the mirror.
LABEL_THRESHOLDS_FALLBACK_STATUS: dict[str, object] = {
    "used_fallback": False,
    "reason": None,
    "path_tried": None,
}


def _frictions_path() -> Path:
    """Resolve the `trading-frictions.json` path.

    Honours `TRADING_FRICTIONS_PATH` so a trainer worker that runs outside
    the workspace checkout can be told exactly where the contract lives.
    Falls back to repo_root/shared/trading-frictions.json (this file is at
    artifacts/ml-engine/app/training/labels.py, so go up 4).
    """
    env_override = os.environ.get("TRADING_FRICTIONS_PATH")
    if env_override:
        return Path(env_override).expanduser()
    return Path(__file__).resolve().parents[4] / "shared" / "trading-frictions.json"


def _record_fallback(reason: str, path: Path) -> None:
    """Set the fallback status and log a single visible WARN. The status
    dict is module-level so a metrics scraper or trainer entry point can
    surface drift risk without re-reading the file.
    """
    if not LABEL_THRESHOLDS_FALLBACK_STATUS["used_fallback"]:
        logger.warning(
            "[labels] FALLBACK to hardcoded label thresholds — "
            "trading-frictions.json unreadable at %s (%s). The trainer is "
            "now at risk of drifting from the live trader/backtester. Set "
            "TRADING_FRICTIONS_PATH or mount the workspace to fix. (task #357)",
            path, reason,
        )
    LABEL_THRESHOLDS_FALLBACK_STATUS["used_fallback"] = True
    LABEL_THRESHOLDS_FALLBACK_STATUS["reason"] = reason
    LABEL_THRESHOLDS_FALLBACK_STATUS["path_tried"] = str(path)


def _load_label_thresholds_from_frictions() -> dict[str, float]:
    """Read `label_thresholds_percent` from shared/trading-frictions.json.

    On success, asserts the hardcoded mirror matches and logs a loud WARN
    on drift. On read failure, logs a single visible WARN, records the
    fallback in `LABEL_THRESHOLDS_FALLBACK_STATUS`, and returns the
    hardcoded mirror so the trainer can still run (e.g. unit tests, worker
    processes without the workspace mounted). See task #357.
    """
    path = _frictions_path()
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        _record_fallback(f"{type(exc).__name__}: {exc}", path)
        return dict(_LABEL_THRESHOLDS_MIRROR)
    block = raw.get("label_thresholds_percent", {})
    out = {k: float(v) for k, v in block.items() if not k.startswith("_")}
    if not out:
        _record_fallback("label_thresholds_percent missing/empty in JSON", path)
        return dict(_LABEL_THRESHOLDS_MIRROR)
    if out != _LABEL_THRESHOLDS_MIRROR:
        logger.warning(
            "[labels] DRIFT — _LABEL_THRESHOLDS_MIRROR=%s but "
            "trading-frictions.json says %s. Update the mirror in labels.py "
            "to match the JSON, otherwise worker processes that fall back "
            "will silently train on stale thresholds. (task #357)",
            _LABEL_THRESHOLDS_MIRROR, out,
        )
    return out


def _load_per_coin_label_thresholds_from_frictions() -> dict[str, dict[str, float]]:
    """Read `label_thresholds_percent_per_coin` from trading-frictions.json.

    Returns a {coin_id -> {timeframe -> percent}} dict. On read failure
    returns the hardcoded mirror (NOT an empty dict) so the trainer still
    honours the per-coin overrides, and records the same loud fallback
    status as `_load_label_thresholds_from_frictions`. Per-coin overrides
    exist so quiet coins (very low realized vol on a given timeframe) can
    still emit non-trivial up/down label mass — see task #120.
    """
    path = _frictions_path()
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        _record_fallback(f"{type(exc).__name__}: {exc}", path)
        return {c: dict(tf) for c, tf in _PER_COIN_LABEL_THRESHOLDS_MIRROR.items()}
    block = raw.get("label_thresholds_percent_per_coin")
    if not isinstance(block, dict) or not block:
        # Readable JSON but the per-coin block is missing/empty. Treat
        # this exactly like an unreadable file: surface a fallback so
        # the operator sees overrides silently disappearing rather than
        # the trainer training every coin on the timeframe default.
        _record_fallback(
            "label_thresholds_percent_per_coin missing/empty in JSON", path
        )
        return {c: dict(tf) for c, tf in _PER_COIN_LABEL_THRESHOLDS_MIRROR.items()}
    out: dict[str, dict[str, float]] = {}
    for coin, tf_map in block.items():
        if coin.startswith("_") or not isinstance(tf_map, dict):
            continue
        inner: dict[str, float] = {}
        for tf, val in tf_map.items():
            if tf.startswith("_"):
                continue
            try:
                inner[tf] = float(val)
            except (TypeError, ValueError):
                continue
        if inner:
            out[coin] = inner
    if out != _PER_COIN_LABEL_THRESHOLDS_MIRROR:
        logger.warning(
            "[labels] DRIFT — _PER_COIN_LABEL_THRESHOLDS_MIRROR has %d coins "
            "but trading-frictions.json has %d (per-coin overrides differ). "
            "Update the mirror in labels.py to match the JSON, otherwise "
            "worker processes that fall back will silently train on stale "
            "per-coin thresholds. (task #357)",
            len(_PER_COIN_LABEL_THRESHOLDS_MIRROR), len(out),
        )
    return out


# Training-only label band. Lower than OUTCOME_THRESHOLDS_PERCENT on every
# timeframe so the up/down classes contain enough rows for the model to
# learn directional structure. Drift is enforced by
# tests/test_training.py::test_label_thresholds_below_outcome_thresholds.
LABEL_THRESHOLDS_PERCENT: dict[str, float] = _load_label_thresholds_from_frictions()

# Per-(coin, timeframe) overrides on top of LABEL_THRESHOLDS_PERCENT. See
# `resolve_label_threshold_pct` for the resolution order. Used by
# `build_labeled_frame_for_coin` to honor a tighter band on quiet coins
# whose 5m bars rarely cross the timeframe default.
LABEL_THRESHOLDS_PERCENT_PER_COIN: dict[str, dict[str, float]] = (
    _load_per_coin_label_thresholds_from_frictions()
)


def resolve_label_threshold_pct(coin_id: str, timeframe: str) -> float:
    """Resolution order for the training-only label threshold (in percent):
    1. Per-coin override for (coin_id, timeframe), if set.
    2. The timeframe default in LABEL_THRESHOLDS_PERCENT.
    3. The adjudication threshold in OUTCOME_THRESHOLDS_PERCENT (last-resort
       fallback so an unknown timeframe never silently labels everything 0).
    """
    coin_block = LABEL_THRESHOLDS_PERCENT_PER_COIN.get(coin_id)
    if coin_block is not None and timeframe in coin_block:
        return float(coin_block[timeframe])
    if timeframe in LABEL_THRESHOLDS_PERCENT:
        return float(LABEL_THRESHOLDS_PERCENT[timeframe])
    return float(OUTCOME_THRESHOLDS_PERCENT[timeframe])


# ──────────────────────────────────────────────────────────────────────
# Task #379 — per-timeframe directional-label horizon.
#
# Diagnosis: at 1h/2h/6h, the legacy 1-bar directional label
# (`label_3class` derived from a single-bar forward return crossed against
# a sub-percent threshold) is dominated by noise. Across the full 1-year
# campaign every short-horizon per-coin slice came in below coin-flip on
# held-out directional accuracy and the multinomial-logistic baseline beat
# the LightGBM booster by 1–5 directional-accuracy points. The trade-aware
# label family already adjudicates over a 4-bar window
# (`FORWARD_HORIZON_CANDLES`) — using a 1-bar direction label alongside a
# 4-bar TP/SL adjudication is internally inconsistent: the booster is asked
# to predict the wrong horizon.
#
# Fix: at 1h/2h/6h, derive `label_3class` from the cumulative forward
# `FORWARD_HORIZON_CANDLES`-bar return so the directional target matches
# the trade-aware horizon. The 1-bar `forward_return` field is preserved
# unchanged so the existing PnL / regressor / per-class-mean code paths
# stay backward compatible. Threshold for the multi-bar label is set
# explicitly in `MULTI_BAR_LABEL_THRESHOLDS_PERCENT` (each value strictly
# below the corresponding `OUTCOME_THRESHOLDS_PERCENT`).
#
# 1m/5m/1d keep the legacy 1-bar label horizon — 1d data is too sparse to
# absorb a multi-bar label without re-introducing left-censoring on every
# row, and 1m/5m already have rich per-bar mass under the sub-1bp default.
#
# Set `ML_DIRECTIONAL_LABEL_HORIZON_MODE=legacy` to roll back to the
# pre-#379 1-bar behaviour for every timeframe.
# ──────────────────────────────────────────────────────────────────────

DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF: dict[str, int] = {
    "1m": 1,
    "5m": 1,
    "1h": FORWARD_HORIZON_CANDLES,
    "2h": FORWARD_HORIZON_CANDLES,
    "6h": FORWARD_HORIZON_CANDLES,
    "1d": 1,
}

# Multi-bar directional label thresholds (% over the multi-bar window).
# Calibrated to keep the up/down/stable class shares broadly comparable to
# the legacy 1-bar mix while the label captures a 4-bar directional move.
# Each value MUST stay strictly below OUTCOME_THRESHOLDS_PERCENT[tf] —
# enforced by `tests/test_training.py::test_multi_bar_label_thresholds_below_outcome`.
MULTI_BAR_LABEL_THRESHOLDS_PERCENT: dict[str, float] = {
    "1h": 0.40,
    "2h": 0.50,
    "6h": 0.80,
}


def _directional_horizon_legacy_mode() -> bool:
    return os.environ.get("ML_DIRECTIONAL_LABEL_HORIZON_MODE", "").lower() == "legacy"


def resolve_directional_label_horizon_candles(timeframe: str) -> int:
    """Return the number of forward bars the 3-class directional label
    spans for `timeframe`. Defaults to 1 for any unknown timeframe so an
    accidental new tf can never silently expand its label horizon.
    """
    if _directional_horizon_legacy_mode():
        return 1
    return int(DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF.get(timeframe, 1))


def resolve_directional_label_threshold_pct(
    coin_id: str, timeframe: str
) -> float:
    """Threshold (in percent) used to bucket the multi-bar forward return
    into UP/STABLE/DOWN. When the horizon is 1 we delegate to the legacy
    `resolve_label_threshold_pct` (per-coin overrides honoured). When the
    horizon is > 1 we use `MULTI_BAR_LABEL_THRESHOLDS_PERCENT[tf]`; per-coin
    overrides are intentionally not consulted here (the per-coin overrides
    were tuned against the 1-bar 5m label distribution and would clobber
    the multi-bar calibration).
    """
    horizon = resolve_directional_label_horizon_candles(timeframe)
    if horizon <= 1:
        return resolve_label_threshold_pct(coin_id, timeframe)
    if timeframe in MULTI_BAR_LABEL_THRESHOLDS_PERCENT:
        return float(MULTI_BAR_LABEL_THRESHOLDS_PERCENT[timeframe])
    # Defensive fallback: scale the 1-bar threshold by sqrt(horizon)
    # (independent-walk approximation), capped under the outcome band.
    base = resolve_label_threshold_pct(coin_id, timeframe)
    scaled = base * math.sqrt(horizon)
    cap = float(OUTCOME_THRESHOLDS_PERCENT.get(timeframe, scaled)) * 0.95
    return float(min(scaled, cap))


def label_three_class(forward_return_pct: float, threshold_pct: float) -> int:
    if forward_return_pct > threshold_pct:
        return 2  # up
    if forward_return_pct < -threshold_pct:
        return 0  # down
    return 1  # stable


# ──────────────────────────────────────────────────────────────────────
# Task #459 — volatility-scaled STABLE-class threshold.
#
# Diagnosis: the slow-loop trainer was using a fixed per-(coin, timeframe)
# percentage to decide STABLE vs UP/DOWN. On 1d / 6h slices for volatile
# alts (PEPE, BONK, FLOKI, …) virtually every bar moves more than the
# fixed band, so the STABLE class was structurally near-empty, the model
# never learned to emit it, and the `directional_call_regression` safety
# gate (`directional_call_share >= 0.95`) refused to promote every slice
# in the campaign. Full root cause:
#   docs/remediation/2026-04-24-full-system-remediation.md §C
#
# Fix: derive the threshold from the realized in-sample volatility of the
# directional-horizon return so the STABLE class always carries real
# mass. The static `resolve_directional_label_threshold_pct` value is
# kept as the FLOOR (we never go tighter than the curated band — that
# would re-introduce the over-broad 5m label problem task #95 fixed).
# The OUTCOME threshold (× `_VOL_SCALED_THRESHOLD_OUTCOME_HEADROOM`) is
# the CEILING — a labelled UP/DOWN row must still imply a move that
# could plausibly clear the round-trip-cost adjudication band.
# ──────────────────────────────────────────────────────────────────────
_VOL_SCALED_THRESHOLD_DEFAULT_FACTOR = 0.7
# Fraction of `OUTCOME_THRESHOLDS_PERCENT[tf]` we are willing to push
# the dynamic threshold up to. Matches the historical convention used
# in `resolve_directional_label_threshold_pct`'s sqrt-horizon fallback
# (line 358).
_VOL_SCALED_THRESHOLD_OUTCOME_HEADROOM = 0.95

THRESHOLD_SOURCE_STATIC = "static"
THRESHOLD_SOURCE_VOL_SCALED = "vol_scaled"


def compute_vol_scaled_threshold_pct(
    forward_returns_pct: Sequence[float],
    baseline_pct: float,
    ceiling_pct: float,
    vol_factor: float = _VOL_SCALED_THRESHOLD_DEFAULT_FACTOR,
) -> tuple[float, str]:
    """Pick a STABLE-class threshold (in %) that scales with realized
    volatility.

    Args:
        forward_returns_pct: per-row directional-horizon return in %.
            On a constant-ramp series these are roughly equal; on a
            real volatile series they span the empirical distribution.
        baseline_pct: the curated per-(coin, tf) static threshold (in
            %). Used as the FLOOR — the dynamic value can only widen
            it, never tighten it. Preserves the per-coin overrides
            wired in trading-frictions.json (task #120 / #318).
        ceiling_pct: hard cap on the chosen threshold (in %). The
            caller passes `OUTCOME_THRESHOLDS_PERCENT[tf] * 0.95` so a
            labelled UP/DOWN row still implies a move capable of
            clearing the adjudication band.
        vol_factor: multiplier on MAD(|forward returns|). 0.7 places
            the STABLE class at ~30–40 % share for symmetric heavy-
            tailed return distributions (Laplace-like crypto bars).

    Returns:
        (threshold_pct, source). Source is `vol_scaled` when MAD-floor
        bound and exceeds the static baseline; `static` otherwise. The
        threshold is always within `[baseline_pct, ceiling_pct]`.
    """
    baseline = float(baseline_pct)
    ceiling = float(ceiling_pct)
    if not forward_returns_pct:
        return baseline, THRESHOLD_SOURCE_STATIC
    abs_returns = [abs(float(r)) for r in forward_returns_pct
                   if r is not None and math.isfinite(float(r))]
    if not abs_returns:
        return baseline, THRESHOLD_SOURCE_STATIC
    mad = float(statistics.median(abs_returns))
    vol_floor = float(vol_factor) * mad
    if vol_floor <= baseline:
        return baseline, THRESHOLD_SOURCE_STATIC
    chosen = min(vol_floor, ceiling)
    if chosen <= baseline:
        # Ceiling pinned us back to (or below) the static value — no
        # actual widening happened. Record `static` so the manifest
        # reflects what the labels.py runtime actually used.
        return baseline, THRESHOLD_SOURCE_STATIC
    return chosen, THRESHOLD_SOURCE_VOL_SCALED


def _vol_scaled_threshold_ceiling_pct(timeframe: str, baseline_pct: float) -> float:
    """Cap on the dynamic threshold for `timeframe` (in %).

    Falls back to a multiple of the static baseline when the timeframe
    is not registered in `OUTCOME_THRESHOLDS_PERCENT`, mirroring the
    defensive shape of `resolve_directional_label_threshold_pct`.
    """
    outcome = OUTCOME_THRESHOLDS_PERCENT.get(timeframe)
    if outcome is None or outcome <= 0:
        return float(baseline_pct) * 5.0
    return float(outcome) * _VOL_SCALED_THRESHOLD_OUTCOME_HEADROOM


def _trade_aware_label(
    closes: list[float],
    i: int,
    horizon: int,
    sl_pct: float,
    tp_pct: float,
    cost_pct: float,
) -> dict:
    """Phase 3 — trade-aware labels for the entry candle at index i.

    Simulates the next `horizon` bars (closes[i+1 .. i+horizon]) against
    SL/TP barriers expressed as fractional moves from the entry close. The
    same forward window is used to compute MAE/MFE for a long trade and
    the cost-aware `prob_move_gt_cost` flag (did the absolute move at any
    point in the window exceed round-trip cost).

    Returns a dict with the new columns; missing right-side bars at the
    very end of the series produce NaN-friendly placeholders so the
    downstream walk-forward splitter still emits the row (existing
    label_3class is unaffected).
    """
    n = len(closes)
    entry = closes[i]
    if entry <= 0:
        return {
            "forward_window_return_pct": float("nan"),
            "prob_move_gt_cost": float("nan"),
            "tp_before_sl_long": float("nan"),
            "tp_before_sl_short": float("nan"),
            "mae_pct_long": float("nan"),
            "mfe_pct_long": float("nan"),
            "opportunity_score": float("nan"),
        }
    end = min(n - 1, i + horizon)
    if end <= i:
        return {
            "forward_window_return_pct": float("nan"),
            "prob_move_gt_cost": float("nan"),
            "tp_before_sl_long": float("nan"),
            "tp_before_sl_short": float("nan"),
            "mae_pct_long": float("nan"),
            "mfe_pct_long": float("nan"),
            "opportunity_score": float("nan"),
        }
    # Walk forward and track per-bar % moves vs entry. We can only "see"
    # bar closes (no intra-bar OHLC in this pipeline), so SL/TP hits are
    # adjudicated against the forward close path — same convention used
    # by the backtester so labels and live behaviour stay aligned.
    max_up = 0.0
    max_down = 0.0
    tp_long_hit_at: Optional[int] = None
    sl_long_hit_at: Optional[int] = None
    tp_short_hit_at: Optional[int] = None
    sl_short_hit_at: Optional[int] = None
    abs_move_exceeds_cost = False
    for j in range(i + 1, end + 1):
        r = (closes[j] - entry) / entry
        if r > max_up:
            max_up = r
        if r < max_down:
            max_down = r
        if abs(r) >= cost_pct and not abs_move_exceeds_cost:
            abs_move_exceeds_cost = True
        # Long trade: TP hits when r >= +tp; SL hits when r <= -sl.
        if tp_long_hit_at is None and r >= tp_pct:
            tp_long_hit_at = j
        if sl_long_hit_at is None and r <= -sl_pct:
            sl_long_hit_at = j
        # Short trade: TP hits when r <= -tp; SL hits when r >= +sl.
        if tp_short_hit_at is None and r <= -tp_pct:
            tp_short_hit_at = j
        if sl_short_hit_at is None and r >= sl_pct:
            sl_short_hit_at = j

    final_r = (closes[end] - entry) / entry

    def _resolve_tp_sl(tp_at: Optional[int], sl_at: Optional[int]) -> float:
        # 1.0 if TP hit strictly before SL; 0.0 if SL hit first; NaN if
        # neither barrier was touched within the horizon (left-censored
        # outcome — keep the row but exclude from per-side ROC/AUC).
        if tp_at is None and sl_at is None:
            return float("nan")
        if tp_at is not None and (sl_at is None or tp_at < sl_at):
            return 1.0
        return 0.0

    tp_long = _resolve_tp_sl(tp_long_hit_at, sl_long_hit_at)
    tp_short = _resolve_tp_sl(tp_short_hit_at, sl_short_hit_at)

    # Opportunity score: signed expected move scaled by the cost cushion.
    # Positive = long-side opportunity, negative = short-side. A move
    # smaller than cost contributes 0 so the metric mirrors the live
    # quant-brain's EV-vs-cost gate.
    final_pct = final_r * 100.0
    cushion_pct = max(0.0, abs(final_pct) - cost_pct * 100.0)
    sign = 1.0 if final_pct > 0 else (-1.0 if final_pct < 0 else 0.0)
    opp_score = sign * cushion_pct

    return {
        "forward_window_return_pct": final_pct,
        "prob_move_gt_cost": 1.0 if abs_move_exceeds_cost else 0.0,
        "tp_before_sl_long": tp_long,
        "tp_before_sl_short": tp_short,
        "mae_pct_long": max_down * 100.0,   # negative
        "mfe_pct_long": max_up * 100.0,     # positive
        "opportunity_score": opp_score,
    }


# --- Task #267 — null-safe enrichment for new feature streams + targets -----
# These columns are registered in `registry.FEATURE_COLUMNS` (and the
# forward-target list) but are not yet populated by any provider for
# every historical bar (OKX caps funding history at ~92 days; the
# liquidations / cross-market pulses are even thinner). We always emit
# them so the model contract is stable.
#
# Task #633 — defaults are `NaN`, NOT `0.0`. LightGBM's `use_missing`
# defaults to true, so it learns "this row had no funding data" cleanly
# rather than the spurious "funding was exactly zero" signal that 0-fill
# bakes into ~75 % of historical 6h rows. Per-feature coverage falls to
# the share of rows that actually have a real provider snapshot at-or-
# before the candle bucket; per-feature density is unchanged.
EXTERNAL_STREAM_DEFAULTS: dict[str, float] = {
    "funding_rate": float("nan"),
    "open_interest_z": float("nan"),
    "liquidations_1h_usd": float("nan"),
    "bid_ask_spread_bps": float("nan"),
    "btc_lead_ret_5m": float("nan"),
    "eth_lead_ret_5m": float("nan"),
    # Task #295 — cross-market liquidation pulses (BTC/ETH/SOL). Same
    # value is broadcast onto every per-coin training row.
    "btc_liquidations_1h_usd": float("nan"),
    "eth_liquidations_1h_usd": float("nan"),
    "sol_liquidations_1h_usd": float("nan"),
}

# Task #295 — pseudo-coin ids the api-server's market-signals poller
# writes the dominant-perp liquidation snapshots under (task #286). Each
# entry maps the coin_id to the trainer feature column the asof-joined
# value lands in.
CROSS_MARKET_LIQ_SOURCES: dict[str, str] = {
    "btc": "btc_liquidations_1h_usd",
    "eth": "eth_liquidations_1h_usd",
    "sol": "sol_liquidations_1h_usd",
}


# Task #643 — self-leak guard for BTC/ETH research training targets.
# Each entry maps a (coin_id) -> set of feature columns that MUST be
# overwritten with NaN when that coin is the training target, because
# the column is a future-derived self-reference (e.g. `btc_lead_ret_5m`
# is BTC's own forward 5m return — predicting bitcoin from that column
# is a textbook leak that inflates DA without producing real edge).
# Applied by `apply_self_leak_guard()` after a per-coin frame is built
# but before it is concat-ed into the cross-coin training matrix. The
# alt-coin slices retain the columns unchanged because for them BTC /
# ETH are genuinely cross-market signals.
SELF_LEAK_FEATURE_COLUMNS: dict[str, frozenset[str]] = {
    "bitcoin": frozenset({"btc_lead_ret_5m", "btc_liquidations_1h_usd"}),
    "ethereum": frozenset({"eth_lead_ret_5m", "eth_liquidations_1h_usd"}),
}


def apply_self_leak_guard(coin_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """Replace every self-leak feature column with NaN for the given
    target coin. Columns that are not present (older feature schema, or
    a slice with no cross-market features at all) are silently skipped
    so the helper is safe to call unconditionally. Returns the SAME
    frame mutated in-place — callers that want isolation should copy
    first. Idempotent.

    Used by `app/training/labels_research/` so the BTC/ETH targets
    cannot cheat off their own future returns / liquidations. Logs the
    columns that were actually overwritten for trainer-report stamping.
    """
    cols = SELF_LEAK_FEATURE_COLUMNS.get(coin_id)
    if not cols:
        return df
    overwritten: list[str] = []
    for c in cols:
        if c in df.columns:
            df[c] = float("nan")
            overwritten.append(c)
    if overwritten:
        logger.info(
            "self_leak_guard_applied coin=%s columns=%s",
            coin_id, sorted(overwritten),
        )
    return df


def _session_features_for_bucket(bucket_start_ms: int) -> dict[str, float]:
    """Compute always-populated session / time-of-day features from the
    UTC bucket start. Three exclusive session one-hots plus a sin/cos
    encoding of hour-of-day so the booster can learn intraday seasonality
    smoothly. Boundaries chosen to mirror the major desk-shifts:
    Asia 00:00–08:00 UTC, EU 08:00–16:00 UTC, US 16:00–24:00 UTC.
    """
    ts = datetime.fromtimestamp(bucket_start_ms / 1000.0, tz=timezone.utc)
    hour = ts.hour + ts.minute / 60.0
    if hour < 8.0:
        s_asia, s_eu, s_us = 1.0, 0.0, 0.0
    elif hour < 16.0:
        s_asia, s_eu, s_us = 0.0, 1.0, 0.0
    else:
        s_asia, s_eu, s_us = 0.0, 0.0, 1.0
    angle = 2.0 * math.pi * hour / 24.0
    return {
        "session_asia": s_asia,
        "session_eu": s_eu,
        "session_us": s_us,
        "hour_of_day_sin": math.sin(angle),
        "hour_of_day_cos": math.cos(angle),
    }


def _asof_signal_value(
    signals: Sequence[dict], bucket_ms: int, key: str,
) -> Optional[float]:
    """Return the most recent non-null `key` value with `timestamp_ms <=
    bucket_ms`. Signals are sorted oldest-first by `fetch_market_signals`,
    so we walk forward and remember the last non-null. Linear over rows
    but called per-bucket inside a per-coin builder, so still cheap given
    the poller's 60s cadence (≤ 1.5k rows / 24h).
    """
    last: Optional[float] = None
    for s in signals:
        ts = s.get("timestamp_ms")
        if ts is None or ts > bucket_ms:
            break
        v = s.get(key)
        if v is not None:
            last = float(v)
    return last


def _build_lead_return_lookup(
    series: Sequence[tuple[int, float]], window_ms: int = 5 * 60 * 1000,
) -> list[tuple[int, float]]:
    """Pre-compute (timestamp_ms -> lead_return_5m_pct) for the BTC/ETH
    reference series. Each return is `(price_t / price_{t-5m}) - 1`,
    expressed in percent. The trainer asof-joins this to candle bucket
    starts.
    """
    if len(series) < 2:
        return []
    out: list[tuple[int, float]] = []
    j = 0
    for i in range(len(series)):
        ts_i, p_i = series[i]
        target = ts_i - window_ms
        # advance j while the next sample is still <= target
        while j + 1 < len(series) and series[j + 1][0] <= target:
            j += 1
        ts_j, p_j = series[j]
        if p_j <= 0 or ts_j > ts_i:
            continue
        # Require we actually have ~5m of history; otherwise skip.
        if ts_i - ts_j < window_ms // 2:
            continue
        ret_pct = ((p_i - p_j) / p_j) * 100.0
        out.append((ts_i, ret_pct))
    return out


def _asof_lead_return(
    lookup: Sequence[tuple[int, float]], bucket_ms: int,
) -> Optional[float]:
    if not lookup:
        return None
    last: Optional[float] = None
    for ts, v in lookup:
        if ts > bucket_ms:
            break
        last = v
    return last


def _next_horizon_targets(
    closes: list[float], i: int, horizon: int, cost_pct: float,
) -> dict:
    """Task #267 — contract-locked next-horizon targets.

    Returns:
      - `realized_vol_next_horizon`: stdev of forward per-bar returns over
        the next `horizon` bars, in percent. NaN when fewer than 2 forward
        bars exist (so the row is excluded from any vol-head fit but the
        slice still trains).
      - `net_pnl_after_costs_pct`: `forward_window_return_pct` minus
        `round_trip_cost_pct * 100` in the direction of the move,
        floored at zero magnitude when the gross move did not clear cost.
        Mirrors the live EV gate so a label-time backtester sees the same
        PnL the live trader would book.
    """
    n = len(closes)
    end = min(n - 1, i + horizon)
    if end <= i:
        return {
            "realized_vol_next_horizon": float("nan"),
            "net_pnl_after_costs_pct": float("nan"),
        }
    fwd_returns_pct: list[float] = []
    for j in range(i + 1, end + 1):
        prev = closes[j - 1]
        if prev <= 0:
            continue
        fwd_returns_pct.append(((closes[j] - prev) / prev) * 100.0)
    if len(fwd_returns_pct) >= 2:
        mean = sum(fwd_returns_pct) / len(fwd_returns_pct)
        var = sum((r - mean) ** 2 for r in fwd_returns_pct) / (len(fwd_returns_pct) - 1)
        rv = math.sqrt(var)
    else:
        rv = float("nan")
    entry = closes[i]
    final_pct = ((closes[end] - entry) / entry) * 100.0 if entry > 0 else float("nan")
    cost_bps = cost_pct * 100.0
    if final_pct > cost_bps:
        net = final_pct - cost_bps
    elif final_pct < -cost_bps:
        net = final_pct + cost_bps
    else:
        net = 0.0
    return {
        "realized_vol_next_horizon": rv,
        "net_pnl_after_costs_pct": net,
    }


def audit_leakage(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    expected_horizon: int,
    *,
    feature_lineage: Optional[dict[str, dict]] = None,
    correlation_threshold: float = 0.99,
    reference_column: str = "lastPrice",
) -> dict:
    """Static + numerical leakage audit on a labeled frame produced by
    `build_labeled_frame_for_coin`.

    Returns:

        {
            "passed": bool,
            "violations": list[str],
            "expected_horizon": int,
            "n_rows": int,
            "lineage_unregistered": list[str],   # features missing lineage
            "future_corr_hits": list[dict],      # numeric leak detections
        }

    The audit enforces the training contract's "strictly point-in-time"
    rule with three independent layers, any of which can reject a slice:

      Layer 1 — Schema invariants:
        * No feature column shares a name with a declared target.
        * `forward_horizon_candles` matches the configured horizon.
        * `timestamp_ms` is monotonic non-decreasing (walk-forward needs
          this and the audit is the last gate before the splitter).
        * Every declared target is present in the frame.

      Layer 2 — Lineage gate:
        Every feature column MUST be present in `feature_lineage` (the
        registry's `FEATURE_LINEAGE` table) with `max_lookforward == 0`.
        An unregistered feature fails the audit so a developer can't
        sneak a new column in without declaring its provenance, and a
        registered column with `max_lookforward > 0` fails immediately.
        This catches the case the schema check cannot — a feature that
        is forward-looking but renamed away from any target name.

      Layer 3 — Numerical future-leak detection:
        For each numeric feature column, compute the Pearson correlation
        between the column and EACH numeric target column present in the
        frame. If |corr| >= `correlation_threshold` for any target,
        flag the column as a probable future-leak. This catches a
        registered column whose lineage entry says `max_lookforward=0`
        but whose values are a near-copy or near-renaming of a target
        (e.g. a feature assigned from `forward_return` under a different
        name). The check uses target columns (which are pure forward
        information) instead of shifted-price comparisons, because
        trending price series produce false positives against any
        smoothing of past price.

    The first two layers are deterministic; the third is statistical and
    will surface anything correlated almost-perfectly with the next bars.
    Together they make "lock to point-in-time" enforceable in CI.
    """
    violations: list[str] = []
    lineage_unregistered: list[str] = []
    future_corr_hits: list[dict] = []
    feat_list = list(feature_columns)
    feat_set = set(feat_list)
    tgt_set = set(target_columns)
    # Layer 1 — schema
    for col in feat_set & tgt_set:
        violations.append(f"feature '{col}' overlaps with declared target column")
    if df.empty:
        return {
            "passed": not violations,
            "violations": violations,
            "expected_horizon": int(expected_horizon),
            "n_rows": 0,
            "lineage_unregistered": lineage_unregistered,
            "future_corr_hits": future_corr_hits,
        }
    if "forward_horizon_candles" in df.columns:
        bad = df.loc[df["forward_horizon_candles"] != expected_horizon]
        if len(bad) > 0:
            violations.append(
                f"{len(bad)} row(s) carry forward_horizon_candles != "
                f"{expected_horizon}"
            )
    else:
        violations.append("frame missing forward_horizon_candles column")
    if "timestamp_ms" in df.columns:
        ts = df["timestamp_ms"].to_numpy()
        if len(ts) > 1 and not (ts[:-1] <= ts[1:]).all():
            violations.append("timestamp_ms is not monotonic non-decreasing")
    else:
        violations.append("frame missing timestamp_ms column")
    for tgt in target_columns:
        if tgt not in df.columns:
            violations.append(f"declared target column '{tgt}' is missing")
    # Layer 2 — lineage gate
    if feature_lineage is None:
        from .registry import FEATURE_LINEAGE as _LIN
        feature_lineage = _LIN
    for col in feat_list:
        meta = feature_lineage.get(col)
        if meta is None:
            lineage_unregistered.append(col)
            violations.append(
                f"feature '{col}' has no lineage entry — declare its "
                "max_lookforward in FEATURE_LINEAGE"
            )
            continue
        max_fwd = meta.get("max_lookforward")
        if max_fwd is None or max_fwd > 0:
            violations.append(
                f"feature '{col}' lineage declares max_lookforward="
                f"{max_fwd!r} (must be 0)"
            )
    # Layer 3 — numerical leak: high |corr| with any declared target.
    target_series: dict[str, pd.Series] = {}
    for tgt in target_columns:
        if tgt in df.columns:
            s = pd.to_numeric(df[tgt], errors="coerce")
            std = s.std(skipna=True)
            if s.notna().sum() >= 30 and std and std > 0:
                target_series[tgt] = s
    for col in feat_list:
        if col not in df.columns or col in target_series:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        std = s.std(skipna=True)
        if s.notna().sum() < 30 or not std or std == 0:
            continue
        for tgt_name, tgt_s in target_series.items():
            aligned = pd.concat([s, tgt_s], axis=1, join="inner").dropna()
            if len(aligned) < 30:
                continue
            try:
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            except Exception:  # noqa: BLE001
                continue
            if pd.notna(corr) and abs(corr) >= correlation_threshold:
                future_corr_hits.append({
                    "feature": col,
                    "target": tgt_name,
                    "corr": round(corr, 6),
                })
                violations.append(
                    f"feature '{col}' has |corr|={abs(corr):.4f} with target "
                    f"'{tgt_name}' — probable future-leak"
                )
                break  # one violation per feature is enough
    return {
        "passed": not violations,
        "violations": violations,
        "expected_horizon": int(expected_horizon),
        "n_rows": int(len(df)),
        "lineage_unregistered": lineage_unregistered,
        "future_corr_hits": future_corr_hits,
    }


def build_labeled_frame_for_coin(
    coin_id: str,
    timeframe: str,
    ticks: list[tuple[datetime, float]],
    news_tags: Optional[list[str]] = None,
    market_signals: Optional[Sequence[dict]] = None,
    btc_lead_lookup: Optional[Sequence[tuple[int, float]]] = None,
    eth_lead_lookup: Optional[Sequence[tuple[int, float]]] = None,
    cross_liq_signals: Optional[dict[str, Sequence[dict]]] = None,
    candles: Optional[list[tuple[datetime, float]]] = None,
) -> pd.DataFrame:
    """Pure function: turn a list of ticks into a labeled feature frame.

    No DB or async — easy to unit test.

    Task #317 — when `candles` is supplied (the caller has fetched
    native-cadence bars from `price_candles` for this (coin, timeframe)),
    we use those bars directly and skip resampling. Each emitted row is
    stamped with its `bars_source` and `bars_native_cadence_ms` so the
    trainer can refuse cadence-mixed slices in the manifest. When
    `candles` is None we fall back to the legacy resample-from-ticks path
    (with a `min_input_cadence_ms` cap so a coarser-cadence row in the
    tick stream cannot silently feed a fine-cadence bucket).
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    bucket_ms = TIMEFRAME_MS[timeframe]
    bars_source: str
    bars_native_cadence_ms: int
    # Use the per-(coin, timeframe) label band when set, else the
    # timeframe default, else the adjudication threshold (defensive — the
    # parity test in test_training.py keeps the two maps aligned). Per-coin
    # overrides exist so quiet coins still emit non-trivial directional
    # label mass — task #120.
    threshold_pct = resolve_label_threshold_pct(coin_id, timeframe) / 100.0

    # Task #379 — per-timeframe directional-label horizon. At 1h/2h/6h the
    # `label_3class` target spans a multi-bar forward window so it matches
    # the trade-aware adjudication horizon. `forward_return` (the 1-bar
    # field consumed by the regressor / PnL backtest) is preserved.
    direction_label_horizon = resolve_directional_label_horizon_candles(timeframe)
    # Task #459 — `direction_label_threshold_pct` and
    # `direction_label_threshold_source` are computed below, AFTER `closes`
    # is built, so we can scale the STABLE-class band to realized
    # volatility. The static `resolve_directional_label_threshold_pct`
    # value remains the FLOOR — see `compute_vol_scaled_threshold_pct`.
    direction_label_baseline_pct = resolve_directional_label_threshold_pct(
        coin_id, timeframe
    )

    # Phase 3 — pull SL/TP/cost percentages once. These come from the same
    # shared/trading-frictions.json that the live trader and backtester
    # read, so a label-time TP/SL adjudication uses the same geometry as
    # production. Failures degrade silently to zeros so unit tests with
    # no contract still produce the legacy label set.
    try:
        fr = get_frictions()
        sl_mult = float(fr.sl_mult(timeframe))
        tp_mult = float(fr.tp_mult(timeframe))
        atr_floor_pct = float(fr.atr_floor_pct(timeframe))
        cost_pct = float(fr.round_trip_cost_pct)
    except Exception:
        sl_mult, tp_mult, atr_floor_pct, cost_pct = 1.5, 3.0, 0.008, 0.003

    if candles is not None and len(candles) > 0:
        # Task #317 — native-cadence path. Use bars from `price_candles`
        # directly; no resampling, no risk of cross-cadence merging.
        closes = [float(p) for _, p in candles if p and p > 0]
        bucket_starts = [
            int(ts.timestamp() * 1000) for ts, p in candles if p and p > 0
        ]
        if len(closes) < MIN_CANDLES_FOR_FEATURES + 1:
            return pd.DataFrame()
        bars_source = "candles"
        bars_native_cadence_ms = bucket_ms
    else:
        # Legacy path: resample raw ticks. The cap forbids a coarser-
        # cadence row (e.g. a daily bar accidentally written to
        # price_history) from feeding a fine-cadence bucket close. A
        # CadenceMismatchError here means the input row stream is itself
        # too coarse for the requested timeframe — surface it; never
        # silently train on it.
        cap_ms = _resample_cadence_cap_ms(timeframe)
        try:
            closes = resample_to_candles(
                ticks, bucket_ms, min_input_cadence_ms=cap_ms,
            )
        except CadenceMismatchError as exc:
            logger.warning(
                "cadence_mismatch_in_resample",
                extra={
                    "coin_id": coin_id,
                    "timeframe": timeframe,
                    "error": str(exc),
                },
            )
            return pd.DataFrame()
        if len(closes) < MIN_CANDLES_FOR_FEATURES + 1:
            return pd.DataFrame()

        # Track the timestamp (start-of-bucket) of every emitted close so the
        # walk-forward splitter can sort by time across coins. Build the
        # bucket-start list from the SAME quarantine-aware row stream the
        # resampler used (via the shared `quarantine_rows_by_cadence`
        # helper) so `len(bucket_starts) == len(closes)` is guaranteed.
        from ..features import quarantine_rows_by_cadence
        kept_rows: list[tuple[int, float]] = []
        for ts, price in ticks:
            if not (price > 0) or not math.isfinite(price):
                continue
            kept_rows.append((int(ts.timestamp() * 1000), price))
        if cap_ms > 0:
            kept_rows = quarantine_rows_by_cadence(kept_rows, cap_ms)
        bucket_starts = []
        current_bucket = -1
        for ts_ms, _price in kept_rows:
            b = (ts_ms // bucket_ms) * bucket_ms
            if b != current_bucket:
                if current_bucket != -1:
                    bucket_starts.append(current_bucket)
                current_bucket = b
        if current_bucket != -1:
            bucket_starts.append(current_bucket)
        assert len(bucket_starts) == len(closes), (
            f"bucket start count {len(bucket_starts)} != closes count {len(closes)}"
        )
        bars_source = "resampled_ticks"
        # When resampling, the native cadence of the input is the live
        # poller's tick cadence (~60s). Recording the requested bucket
        # width here would mask the fact that this slice was synthesized
        # rather than read from a same-cadence bar source.
        bars_native_cadence_ms = TIMEFRAME_MS["1m"]

    # Phase 5 — news tags for this coin are fetched once by the async
    # caller (`build_labeled_dataset`) and threaded into the per-coin row
    # builder via the optional `news_tags` parameter. We use the CURRENT
    # snapshot (not per-bucket history) because the news classifier only
    # began emitting tags in Phase 5; back-filling a true point-in-time
    # tag history would require re-classifying every old headline. The
    # model sees zeros for rows whose tags didn't exist yet and the live
    # tag set for new bars.

    rows: list[dict] = []
    # Task #379 — when the directional-label horizon spans multiple
    # forward bars, the loop must stop early enough that
    # `closes[i + direction_label_horizon]` exists. The legacy 1-bar code
    # path (1m/5m/1d) keeps its existing `len(closes) - 1` upper bound.
    label_loop_end = len(closes) - max(1, direction_label_horizon)

    # Task #459 — derive the STABLE-class threshold from the realized
    # in-sample volatility of the directional-horizon return BEFORE we
    # enter the row loop. The static `direction_label_baseline_pct` is
    # the floor (we never go tighter than the curated band), the
    # outcome threshold (× 0.95) is the ceiling. The chosen value and
    # its source ("static" | "vol_scaled") are stamped on every row so
    # the trainer can mirror them onto the slice manifest without re-
    # deriving anything.
    direction_returns_pct: list[float] = []
    for i in range(MIN_CANDLES_FOR_FEATURES - 1, label_loop_end):
        entry_close = closes[i]
        if entry_close <= 0:
            continue
        if direction_label_horizon > 1:
            r = (closes[i + direction_label_horizon] - entry_close) / entry_close
        else:
            r = (closes[i + 1] - entry_close) / entry_close
        direction_returns_pct.append(r * 100.0)
    ceiling_pct = _vol_scaled_threshold_ceiling_pct(
        timeframe, direction_label_baseline_pct,
    )
    direction_label_threshold_pct_value, direction_label_threshold_source = (
        compute_vol_scaled_threshold_pct(
            direction_returns_pct,
            baseline_pct=direction_label_baseline_pct,
            ceiling_pct=ceiling_pct,
        )
    )
    direction_label_threshold_pct = direction_label_threshold_pct_value / 100.0

    # Task #481 — precompute the per-bar feature dicts ONCE for the whole
    # close series. The legacy site called
    # ``build_feature_vector(closes[: i + 1])`` per row, which re-walked
    # the full prefix inside every EMA/MACD/RSI/ATR helper and made the
    # 5m dataset build O(N²); for bonk's growing 5m candle history this
    # was the >30-minute hang that blocked the full-timeframe campaign.
    # ``build_feature_vectors_for_series`` produces bit-identical output
    # to the per-call path (verified by ``test_feature_batch_parity``),
    # so the row-level math downstream is unchanged.
    feature_vectors = build_feature_vectors_for_series(
        closes, news_tags=news_tags,
    )

    for i in range(MIN_CANDLES_FOR_FEATURES - 1, label_loop_end):
        feats = feature_vectors[i]
        if feats is None:
            continue
        forward_ret = (closes[i + 1] - closes[i]) / closes[i]
        # Multi-bar forward return powers `label_3class` when the
        # directional horizon > 1; for the 1-bar legacy path this equals
        # `forward_ret`.
        if direction_label_horizon > 1:
            direction_forward_ret = (
                closes[i + direction_label_horizon] - closes[i]
            ) / closes[i]
        else:
            direction_forward_ret = forward_ret
        row = dict(feats)
        row["coin_id"] = coin_id
        row["timeframe"] = timeframe
        row["timestamp_ms"] = bucket_starts[i]
        row["forward_return"] = forward_ret
        row["label_binary_up"] = 1 if forward_ret > 0 else 0
        row["label_3class"] = label_three_class(
            direction_forward_ret * 100.0,
            direction_label_threshold_pct * 100.0,
        )
        # Task #379 — surface the actual label horizon and the multi-bar
        # forward return on the row so failure-analysis / backtest tooling
        # can distinguish 1-bar vs multi-bar slices without re-deriving
        # them from the timeframe.
        row["directional_label_horizon_candles"] = int(direction_label_horizon)
        row["directional_label_forward_return"] = float(direction_forward_ret)
        # Task #459 — stamp the actual STABLE-class threshold (in %) and
        # its source on every row so the trainer can mirror them onto
        # the slice manifest and the verification dashboard can audit
        # which slices are running on the vol-scaled band.
        row["directional_label_threshold_pct"] = float(
            direction_label_threshold_pct_value,
        )
        row["directional_label_threshold_source"] = (
            direction_label_threshold_source
        )
        # Phase 2 — first-class regime label on every training row so the
        # downstream model can condition on it (and per-regime accuracy
        # can be reported). Computed from the SAME feature vector used by
        # the trainer so live and training cannot drift.
        row["regime"] = classify_regime_from_features(feats).label
        # Phase 3 — trade-aware labels. Geometry is taken from
        # `atrPct` on the entry bar, floored by `atr_floor_pct`, and
        # multiplied by SL/TP mults from the contract — same shape the
        # live paper-trader uses. We DO NOT replace the legacy
        # label_3class; both label families are persisted side by side so
        # the existing classifier stays comparable across phases while
        # the specialist heads consume the new columns.
        atr_pct_frac = max(float(feats.get("atrPct", 0.0)) / 100.0, atr_floor_pct)
        sl_pct_frac = atr_pct_frac * sl_mult
        tp_pct_frac = atr_pct_frac * tp_mult
        ta = _trade_aware_label(
            closes,
            i,
            FORWARD_HORIZON_CANDLES,
            sl_pct_frac,
            tp_pct_frac,
            cost_pct,
        )
        row["forward_horizon_candles"] = FORWARD_HORIZON_CANDLES
        # Task #317 — cadence provenance, carried per-row so the trainer
        # can detect (and refuse) cadence-mixed slices in the manifest.
        row["bars_source"] = bars_source
        row["bars_native_cadence_ms"] = bars_native_cadence_ms
        row.update(ta)
        # Task #267 — contract-locked next-horizon targets.
        row.update(_next_horizon_targets(
            closes, i, FORWARD_HORIZON_CANDLES, cost_pct,
        ))
        # Task #271 — populate external-stream columns from the market_
        # signals snapshots written by the api-server's poller. The asof
        # join takes the most recent snapshot at-or-before the candle's
        # bucket start, so the row is strictly point-in-time. Any column
        # whose stream is unavailable falls through to the registered
        # safe default below.
        bucket_ms = bucket_starts[i]
        if market_signals:
            fr = _asof_signal_value(market_signals, bucket_ms, "funding_rate")
            if fr is not None:
                row["funding_rate"] = fr
            oi = _asof_signal_value(market_signals, bucket_ms, "open_interest_usd")
            if oi is not None:
                # Z-score the raw OI against the per-coin window we have
                # in `market_signals` (mean / std over snapshots up to
                # bucket_ms). When the window is too short for a real
                # standardisation (only one snapshot, or every snapshot
                # equal to the same value), the z-score is mathematically
                # undefined — emit NaN rather than 0.0 so the model
                # treats it as missing, not as a literal "zero z-score"
                # signal (task #633).
                vals: list[float] = []
                for s in market_signals:
                    ts = s.get("timestamp_ms")
                    if ts is None or ts > bucket_ms:
                        break
                    v = s.get("open_interest_usd")
                    if v is not None and v > 0:
                        vals.append(float(v))
                if len(vals) >= 2:
                    mean = sum(vals) / len(vals)
                    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                    sd = math.sqrt(var)
                    if sd > 0:
                        row["open_interest_z"] = (oi - mean) / sd
                    else:
                        row["open_interest_z"] = float("nan")
                else:
                    row["open_interest_z"] = float("nan")
            liq = _asof_signal_value(market_signals, bucket_ms, "liquidations_1h_usd")
            if liq is not None:
                row["liquidations_1h_usd"] = liq
            sp = _asof_signal_value(market_signals, bucket_ms, "bid_ask_spread_bps")
            if sp is not None:
                row["bid_ask_spread_bps"] = sp
        if btc_lead_lookup:
            v = _asof_lead_return(btc_lead_lookup, bucket_ms)
            if v is not None:
                row["btc_lead_ret_5m"] = v
        if eth_lead_lookup:
            v = _asof_lead_return(eth_lead_lookup, bucket_ms)
            if v is not None:
                row["eth_lead_ret_5m"] = v
        # Task #295 — cross-market liquidation pulses, asof-joined from
        # the BTC/ETH/SOL pseudo-coin rows in `market_signals` (task
        # #286). Same value broadcasts onto every per-coin training row
        # because regime stress in the dominant perps usually leads
        # moves in the alts.
        if cross_liq_signals:
            for src_coin, col_name in CROSS_MARKET_LIQ_SOURCES.items():
                src_rows = cross_liq_signals.get(src_coin)
                if not src_rows:
                    continue
                liq_v = _asof_signal_value(
                    src_rows, bucket_ms, "liquidations_1h_usd",
                )
                if liq_v is not None:
                    row[col_name] = liq_v
        # Task #267 / #633 — null-safe external-stream defaults. The
        # columns are registered in `registry.FEATURE_COLUMNS`; an
        # unwired stream (or a row where the asof-join found no match)
        # falls through to its registered default of NaN — *not* 0.0 —
        # so LightGBM's native missing-value handling kicks in instead
        # of silently teaching the booster that funding/liquidations
        # were exactly zero on those bars. The model contract (column
        # presence + ordering) is unchanged.
        for col, default in EXTERNAL_STREAM_DEFAULTS.items():
            row.setdefault(col, default)
        # Task #267 — always-populated session / time-of-day features.
        row.update(_session_features_for_bucket(bucket_starts[i]))
        rows.append(row)

    return pd.DataFrame(rows)


async def build_labeled_dataset(
    coin_ids: Sequence[str], timeframe: str, lookback_ms: int,
    provenance_out: Optional[dict] = None,
) -> pd.DataFrame:
    """Fetch ticks for each coin and build the pooled labeled frame.

    Task #267 — when `provenance_out` is supplied, populates it with a
    per-coin record of `{rows_real, rows_synthetic, rejected_synthetic}`
    so the trainer can stamp `report.json` with the real-data provenance
    guard outcome. Coins whose lookback window contained ANY synthetic
    rows are EXCLUDED from the labeled frame and recorded with
    `rejected_synthetic=true`. The SQL filter in `fetch_real_ticks`
    already excludes those rows from the model's inputs; this guard is
    the second line of defence the contract requires.
    """
    from ..db import (
        fetch_real_candles,
        fetch_real_ticks_with_provenance,
        fetch_market_signals,
        fetch_lead_price_series,
    )

    # Task #271 — fetch BTC/ETH reference series ONCE per timeframe and
    # build the (timestamp -> 5m return) lookup. The same lookup feeds
    # every coin's `btc_lead_ret_5m` / `eth_lead_ret_5m` column.
    try:
        btc_series = await fetch_lead_price_series("btc", lookback_ms)
    except Exception:
        btc_series = []
    try:
        eth_series = await fetch_lead_price_series("eth", lookback_ms)
    except Exception:
        eth_series = []
    btc_lead_lookup = _build_lead_return_lookup(btc_series)
    eth_lead_lookup = _build_lead_return_lookup(eth_series)

    # Task #295 — pull BTC/ETH/SOL liquidation snapshots ONCE per
    # timeframe and asof-join the same value onto every coin's row. The
    # poller writes these under the pseudo-coin ids `btc`/`eth`/`sol`
    # (task #286). Failures degrade silently to an empty list so the
    # column falls through to the registered safe default.
    cross_liq_signals: dict[str, list[dict]] = {}
    for src_coin in CROSS_MARKET_LIQ_SOURCES:
        try:
            cross_liq_signals[src_coin] = await fetch_market_signals(
                src_coin, lookback_ms,
            )
        except Exception:
            cross_liq_signals[src_coin] = []

    frames: list[pd.DataFrame] = []
    for coin_id in coin_ids:
        try:
            prov = await fetch_real_ticks_with_provenance(coin_id, lookback_ms)
        except Exception as exc:  # noqa: BLE001 - degrade to legacy path
            logger.warning(
                "provenance_fetch_failed",
                extra={"coin_id": coin_id, "error": str(exc)},
            )
            ticks = await fetch_real_ticks(coin_id, lookback_ms)
            prov = {
                "ticks": ticks,
                "rows_real": len(ticks),
                "rows_synthetic": 0,
                "rejected_synthetic": False,
            }
        if provenance_out is not None:
            provenance_out[coin_id] = {
                "rows_real": int(prov["rows_real"]),
                "rows_synthetic": int(prov["rows_synthetic"]),
                "rejected_synthetic": bool(prov["rejected_synthetic"]),
            }
        if prov["rejected_synthetic"]:
            logger.warning(
                "provenance_guard_rejected_slice",
                extra={
                    "coin_id": coin_id,
                    "timeframe": timeframe,
                    "rows_synthetic": prov["rows_synthetic"],
                    "rows_real": prov["rows_real"],
                },
            )
            continue
        ticks = prov["ticks"]
        # Task #317 — for any timeframe with a native bar source
        # (5m / 1h / 2h / 6h / 1d), prefer reading aggregated candles
        # directly from `price_candles` so the trainer cannot silently
        # merge a daily bar into a finer bucket via the resampler. The
        # candle path is independent of `ticks` — a coin with a real
        # candle history but no live-poll ticks must still be trainable
        # from those candles. Only when neither candles nor ticks are
        # available do we skip the slice.
        candles_for_tf: Optional[list[tuple[datetime, float]]] = None
        if timeframe in CANDLES_PREFERRED_TIMEFRAMES:
            try:
                fetched = await fetch_real_candles(
                    coin_id, timeframe, lookback_ms,
                )
            except Exception:
                fetched = []
            # Per task contract: if ANY candles exist for a candle-
            # preferred timeframe, use them — never silently fall through
            # to the resampled-ticks path because the candle count is
            # below the feature-window minimum (the builder will return
            # an empty frame in that case, which is the correct signal
            # that the slice is not yet trainable).
            if fetched:
                candles_for_tf = fetched
        if candles_for_tf is None and not ticks:
            continue
        # LLM isolation contract (Tasks #91/#255/#344): the LLM must NOT
        # influence trade decisions, even at training time. The Phase 5
        # news-tag one-hot block was an LLM-derived input fed into the
        # LightGBM training matrix — that channel is now permanently
        # shut. We pass an empty list so the feature builder zero-fills
        # the tag columns, keeping the column schema stable for any
        # legacy training comparisons. New models will not see any
        # signal in those columns and therefore will not learn to
        # depend on them. The `news_tags` table is still WRITTEN by the
        # sidecar-gated news classifier for dashboard display; it is
        # just no longer READ here.
        news_tags: list[str] = []
        try:
            market_signals = await fetch_market_signals(coin_id, lookback_ms)
        except Exception:
            market_signals = []
        df = build_labeled_frame_for_coin(
            coin_id, timeframe, ticks,
            news_tags=news_tags,
            market_signals=market_signals,
            btc_lead_lookup=btc_lead_lookup,
            eth_lead_lookup=eth_lead_lookup,
            cross_liq_signals=cross_liq_signals,
            candles=candles_for_tf,
        )
        if provenance_out is not None and not df.empty:
            entry = provenance_out.setdefault(coin_id, {})
            entry["bars_source"] = (
                "candles" if candles_for_tf is not None else "resampled_ticks"
            )
            entry["bars_native_cadence_ms"] = (
                TIMEFRAME_MS[timeframe]
                if candles_for_tf is not None
                else TIMEFRAME_MS["1m"]
            )
            # Per-slice cadence breakdown so the timeframe-level
            # provenance surface mirrors what's stamped on the model
            # manifest. The dashboard / verification gate can read either
            # without re-deriving from the dataframe.
            label = (
                f"candles:{TIMEFRAME_MS[timeframe]}ms"
                if candles_for_tf is not None
                else "resampled_ticks:60000ms"
            )
            entry["bars_by_native_cadence"] = {label: int(len(df))}
            entry["cadence_mixed"] = False
            entry["cadence_mitigation"] = None
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)


async def build_labeled_dataset_standalone(
    coin_ids: Sequence[str], timeframe: str, lookback_ms: int
) -> pd.DataFrame:
    """Convenience for the CLI — opens its own DB pool."""
    await init_pool()
    try:
        return await build_labeled_dataset(coin_ids, timeframe, lookback_ms)
    finally:
        await close_pool()


def main_smoke():  # pragma: no cover - CLI helper
    coins = ["pepe", "bonk"]
    df = asyncio.run(build_labeled_dataset_standalone(coins, "1m", 7 * 24 * 3600 * 1000))
    print(df.head())
    print("rows:", len(df))


if __name__ == "__main__":  # pragma: no cover
    main_smoke()
