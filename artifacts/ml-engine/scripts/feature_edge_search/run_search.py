"""Task #580 — feature-edge search runner.

Stage 1: per-candidate walk-forward ablation against the unmodified base
schema, per (timeframe ∈ TIMEFRAMES). For each candidate we fit a baseline
LightGBM run AND a (baseline + candidate) LightGBM run on the SAME 3-fold
walk-forward splits, then record per-fold:
  - directional accuracy (DA), counted only on rows whose true label is
    UP or DOWN (matches the verification gate's definition);
  - post-fee P&L on the trades the model would have taken (argmax over
    UP/STABLE/DOWN, trade only when argmax != STABLE), priced with the
    canonical round-trip cost from `shared/trading-frictions.json`;
  - directional_call_share = trades / test_rows;
  - n_trades.

A candidate is admitted to stage 2 iff it produces a positive (DA-delta,
PnL-delta) on **at least 2** of the 3 OOS folds (per task spec).

Stage 2: stacked-candidate run. All admitted candidates are added to the
schema together; we re-evaluate per (coin, timeframe) slice with the
SAME walk-forward machinery and check the unmodified production
promotion gate:
    DA > baseline_DA + 0.02
    AND post_fee_pnl_pct_total > 0
    AND directional_call_share in [0.4, 0.85]
    AND n_trades >= 30

Verdict outputs:
    artifacts/ml-engine/reports/<TS>-task580-feature-edge-search.json
    artifacts/ml-engine/reports/<TS>-task580-feature-edge-verdict.md

This script does NOT touch any production source file (registry,
trading-frictions, gate constants, role JSON). It reads them.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.training.registry import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_PREFIXES,
)
from app.training.calibration import (  # noqa: E402
    apply_single_temperature,
    fit_single_temperature,
)

logger = logging.getLogger("ml-engine.feature_edge_search")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = ROOT.parent.parent
SHARED_FORBIDDEN_PATH = REPO_ROOT / "shared" / "forbidden-features.json"
SHARED_FRICTIONS_PATH = REPO_ROOT / "shared" / "trading-frictions.json"
CANDIDATES_PATH = Path(__file__).resolve().parent / "candidates.json"
DATASETS_DIR = ROOT / "models" / "datasets"
REPORTS_DIR = ROOT / "reports"

TIMEFRAMES = ["1h", "2h", "6h", "1d"]
N_FOLDS = 3
LGB_ROUNDS = 80
LGB_LEAVES = 15
LGB_LR = 0.1
SEED = 42

# Task #587 — fold-fitted single-T temperature scaling + learned neutral-band
# threshold on a calibration tail. Off by default to preserve task #580
# reproducibility; toggle with `ML_FEATURE_EDGE_CALIBRATE=1`. The tail is
# the LAST `CAL_TAIL_FRAC` of each fold's training slice (point-in-time-safe
# — the tail is older than the test slice by construction, since walk-forward
# expands `train_end` strictly before `test_start`).
#
# Two-step procedure when the flag is on:
#   1. Fit booster on the inner-train (the part of the training slice
#      BEFORE the cal tail), fit `inv_T` (Guo et al. 2017 single-scalar
#      temperature scaling) on the cal-tail predictions. Apply `inv_T` to
#      the test-set predictions.
#   2. Sweep `delta in [0, 1)` and pick the smallest `delta` whose cal-tail
#      `trade_share = mean(max(P_UP, P_DOWN) > P_STABLE + delta)` lands
#      AT OR BELOW the per-slice-tuned `trade_share_target` (see Task #591
#      below). The trade rule used by `score_fold` becomes "trade iff
#      `max(P_UP, P_DOWN) > P_STABLE + delta`" instead of the legacy
#      "trade iff `argmax != STABLE`". Critically, `delta` is fit on the
#      cal tail only — the test-set trade_share follows whatever
#      distribution the cal tail predicts.
#
# Why both steps: single-T calibration is argmax-preserving by construction
# (Task #482 — see `app/training/calibration.py`), so it cannot on its own
# move `trade_share` because `argmax != STABLE` is invariant under any
# positive monotone transform of the booster's logits. The
# decision-threshold step is the actual lever that moves trade_share into
# the gate band; calibration is what makes the cal-tail-fit `delta`
# generalize to the test slice (raw booster confidences are over-extreme
# under the natural label imbalance, so a `delta` fit on raw probs would
# generalize poorly).
#
# Task #591 — per-(coin, timeframe) auto-tuning of `trade_share_target`.
# The original task #587 implementation hardcoded a single target of 0.625
# (the gate-band midpoint) for every slice. The task #587 diagnostic showed
# per-slice trade-share-vs-PnL curves vary widely — some slices' best
# cal-tail PnL would land near the upper end of the band (~0.85), others
# nearer the lower end (~0.40). Replacing the single hardcoded target with
# a small grid sweep `{0.50, 0.625, 0.75}` and picking the value that
# maximizes cal-tail post-fee PnL (per fold, per slice) lets the
# calibration path squeeze more PnL out of the same booster without
# changing any gate constants. Selection is point-in-time-safe — the cal
# tail is strictly older than the test slice by walk-forward construction.
#
# With the flag off the behaviour is byte-identical to the pre-task-587
# path: full train + raw probabilities + argmax-trade rule.
ENABLE_CALIBRATION = os.environ.get("ML_FEATURE_EDGE_CALIBRATE", "0") == "1"
CAL_TAIL_FRAC = 0.20
# Default target used when no rt_cost-aware grid selection is available
# (e.g. callers that don't pass `round_trip_cost_frac`). Kept as the
# gate-band midpoint for backwards compatibility.
TRADE_SHARE_TARGET = 0.625
# Task #591 — small grid of cal-tail trade_share targets. The runner
# picks the one that maximizes post-fee PnL on the cal tail, per fold.
# Spans the production gate band [0.40, 0.85] symmetrically around the
# midpoint with three points so the selection has actual choice without
# blowing up runtime.
TRADE_SHARE_TARGET_GRID: tuple[float, ...] = (0.50, 0.625, 0.75)
NEUTRAL_BAND_DELTA_MAX = 0.99

# Stage-1 admission rule: ≥2 of 3 OOS folds positive in BOTH metrics.
STAGE1_MIN_FOLDS_POSITIVE = 2

# Stage-2 production gate (mirrors `task-580` spec, sourced from frictions).
STAGE2_DA_LIFT_FLOOR = 0.02
STAGE2_PNL_FLOOR_PCT_TOTAL = 0.0
STAGE2_TRADE_SHARE_LO = 0.40
STAGE2_TRADE_SHARE_HI = 0.85
STAGE2_MIN_TRADES = 30


# ── Forbidden-name guard ─────────────────────────────────────────────────
def load_forbidden_prefixes() -> tuple[str, ...]:
    """Cross-check that `shared/forbidden-features.json` and the registry
    tuple agree. Either disagreement is a contract bug; abort.
    """
    if not SHARED_FORBIDDEN_PATH.exists():
        raise RuntimeError(
            f"missing forbidden-features manifest at {SHARED_FORBIDDEN_PATH}"
        )
    payload = json.loads(SHARED_FORBIDDEN_PATH.read_text())
    file_prefixes = tuple(payload.get("prefixes") or ())
    registry_prefixes = tuple(FORBIDDEN_FEATURE_PREFIXES)
    if set(file_prefixes) != set(registry_prefixes):
        raise RuntimeError(
            "forbidden-features.json prefixes and registry "
            "FORBIDDEN_FEATURE_PREFIXES disagree: "
            f"file={file_prefixes} registry={registry_prefixes}. "
            "Update both before running this search."
        )
    return file_prefixes


# ── Candidate loader ─────────────────────────────────────────────────────
def load_candidates() -> list[dict]:
    payload = json.loads(CANDIDATES_PATH.read_text())
    cands = payload.get("candidates") or []
    if not cands:
        raise RuntimeError(f"candidates.json has no candidates: {CANDIDATES_PATH}")
    return cands


def assert_candidate_legal(cand: dict, forbidden_prefixes: tuple[str, ...]) -> None:
    name = cand.get("name") or ""
    if any(name.startswith(p) for p in forbidden_prefixes):
        raise RuntimeError(
            f"candidate name {name!r} starts with a forbidden prefix; refuse"
        )
    if not isinstance(cand.get("source_columns"), list) or not cand["source_columns"]:
        raise RuntimeError(f"candidate {name!r} missing source_columns")
    if not isinstance(cand.get("math"), str):
        raise RuntimeError(f"candidate {name!r} missing math")
    if cand.get("max_lookforward", 0) != 0:
        raise RuntimeError(f"candidate {name!r} has max_lookforward != 0")


# ── Candidate transform implementations ──────────────────────────────────
def _trailing_rolling(group_series: pd.Series, window: int, agg: str) -> pd.Series:
    # Strict trailing rolling (no peek into the current row's future bars).
    # min_periods=1 so early rows are still defined; downstream NaNs are
    # filled with 0 by the caller before training.
    rolling = group_series.rolling(window=window, min_periods=1)
    if agg == "std":
        return rolling.std(ddof=0)
    if agg == "mean":
        return rolling.mean()
    if agg == "max":
        return rolling.max()
    raise ValueError(f"unsupported agg {agg!r}")


def materialize_candidate(df: pd.DataFrame, cand: dict) -> pd.Series:
    """Compute the candidate column for `df`, RESPECTING per-coin grouping
    where the math involves a rolling window. Output is aligned to `df.index`.
    """
    name = cand["name"]
    if name == "vol_of_vol_30":
        out = df.groupby("coin_id", group_keys=False)["realizedVol"].apply(
            lambda s: _trailing_rolling(s.astype(float), 30, "std")
        )
    elif name == "realizedVol_log":
        out = np.log1p(np.abs(pd.to_numeric(df["realizedVol"], errors="coerce")))
    elif name == "ret1_squared":
        v = pd.to_numeric(df["ret1"], errors="coerce")
        out = v * v
    elif name == "ret5_minus_ret10":
        a = pd.to_numeric(df["ret5"], errors="coerce")
        b = pd.to_numeric(df["ret10"], errors="coerce")
        out = a - b
    elif name == "rsi14_centered_squared":
        v = pd.to_numeric(df["rsi14"], errors="coerce") - 50.0
        out = v * v
    elif name == "bb_pctb_extreme":
        v = pd.to_numeric(df["bbPctB"], errors="coerce") - 0.5
        out = v.abs()
    elif name == "macd_hist_norm_atr":
        h = pd.to_numeric(df["macdHist"], errors="coerce")
        a = pd.to_numeric(df["atr14"], errors="coerce").clip(lower=1e-9)
        out = h / a
    elif name == "vol_zscore60_squared":
        v = pd.to_numeric(df["volZScore60"], errors="coerce")
        out = v * v
    elif name == "ema_spread_per_atr":
        s = pd.to_numeric(df["emaSpreadPct"], errors="coerce")
        a = pd.to_numeric(df["atrPct"], errors="coerce").clip(lower=1e-9)
        out = s / a
    elif name == "atr_pct_zscore_60":
        def _zscore(s: pd.Series) -> pd.Series:
            v = s.astype(float)
            mean = _trailing_rolling(v, 60, "mean")
            std = _trailing_rolling(v, 60, "std").clip(lower=1e-9)
            return (v - mean) / std
        out = df.groupby("coin_id", group_keys=False)["atrPct"].apply(_zscore)
    elif name == "drawdown_30":
        def _dd(s: pd.Series) -> pd.Series:
            v = s.astype(float)
            mx = _trailing_rolling(v, 30, "max").clip(lower=1e-9)
            return (v - mx) / mx * 100.0
        out = df.groupby("coin_id", group_keys=False)["lastPrice"].apply(_dd)
    elif name == "btc_lead_x_self_ret":
        b = pd.to_numeric(df["btc_lead_ret_5m"], errors="coerce").fillna(0.0)
        s = np.sign(pd.to_numeric(df["ret1"], errors="coerce").fillna(0.0))
        out = b * s
    elif name == "eth_lead_minus_btc_lead":
        e = pd.to_numeric(df["eth_lead_ret_5m"], errors="coerce").fillna(0.0)
        b = pd.to_numeric(df["btc_lead_ret_5m"], errors="coerce").fillna(0.0)
        out = e - b
    elif name == "macd_signal_cross_strength":
        line = pd.to_numeric(df["macdLine"], errors="coerce")
        sig = pd.to_numeric(df["macdSignal"], errors="coerce")
        h = pd.to_numeric(df["macdHist"], errors="coerce").abs()
        out = np.where(line >= sig, 1.0, -1.0) * h
        out = pd.Series(out, index=df.index)
    elif name == "ret1_x_volZ60":
        a = pd.to_numeric(df["ret1"], errors="coerce")
        b = pd.to_numeric(df["volZScore60"], errors="coerce")
        out = a * b
    elif name == "log_liquidations_self":
        v = pd.to_numeric(df["liquidations_1h_usd"], errors="coerce").fillna(0.0)
        out = np.log1p(v.clip(lower=0.0))
    else:
        raise ValueError(f"no materializer registered for candidate {name!r}")

    out = pd.Series(np.asarray(out, dtype=float), index=df.index, name=name)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# ── Latest-snapshot picker ───────────────────────────────────────────────
def latest_dataset_path(timeframe: str) -> Optional[Path]:
    if not DATASETS_DIR.exists():
        return None
    cands = sorted(DATASETS_DIR.glob(f"{timeframe}_*.parquet"))
    return cands[-1] if cands else None


def load_dataset(timeframe: str) -> pd.DataFrame:
    p = latest_dataset_path(timeframe)
    if p is None:
        raise FileNotFoundError(f"no dataset snapshot for tf={timeframe} under {DATASETS_DIR}")
    df = pd.read_parquet(p)
    if "label_3class" not in df.columns:
        raise ValueError(f"dataset {p} missing label_3class column")
    if "forward_return" not in df.columns:
        raise ValueError(f"dataset {p} missing forward_return column")
    if "coin_id" not in df.columns:
        raise ValueError(f"dataset {p} missing coin_id column")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def get_round_trip_cost_pct() -> float:
    payload = json.loads(SHARED_FRICTIONS_PATH.read_text())
    fees = payload["fees"]
    # Exact mirror of artifacts/ml-engine/app/backtest/contract.py — the
    # round-trip cost is 2 * (taker_fee + slippage), expressed in fraction.
    return 2.0 * (float(fees["taker_fee_pct"]) + float(fees["slippage_pct"]))


# ── LightGBM trainer ─────────────────────────────────────────────────────
def lgb_params() -> dict:
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": LGB_LEAVES,
        "learning_rate": LGB_LR,
        "min_child_samples": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "deterministic": True,
        "seed": SEED,
    }


def encode_coin_idx(df: pd.DataFrame, vocab: list[str]) -> pd.Series:
    idx = {c: i for i, c in enumerate(vocab)}
    return df["coin_id"].map(idx).fillna(-1).astype(int)


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: list[str],
) -> np.ndarray:
    import lightgbm as lgb
    cats = [c for c in CATEGORICAL_FEATURES if c in feature_names]
    X_tr = train[feature_names].astype(float).fillna(0.0)
    y_tr = train["label_3class"].astype(int).to_numpy()
    X_te = test[feature_names].astype(float).fillna(0.0)
    train_set = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature=cats if cats else "auto",
        free_raw_data=False,
    )
    booster = lgb.train(
        lgb_params(), train_set,
        num_boost_round=LGB_ROUNDS,
        callbacks=[lgb.log_evaluation(0)],
    )
    proba = booster.predict(X_te)
    if proba.ndim == 1:
        # Single-class collapse — pad to 3-class uniform on STABLE.
        proba = np.tile([0.0, 1.0, 0.0], (len(X_te), 1)).astype(float)
    return proba


def _fit_neutral_band_delta(
    cal_proba: np.ndarray, target_trade_share: float,
) -> float:
    """Pick the smallest `delta in [0, 1)` whose cal-tail trade_share
    `mean(max(P_UP, P_DOWN) > P_STABLE + delta)` is `<= target_trade_share`.
    `target_trade_share` should be the desired test-time trade frequency
    (e.g. the midpoint of the gate's trade_share band). Returns 0.0 when
    the cal tail already trades at or below the target.
    """
    p_down = cal_proba[:, 0]
    p_stab = cal_proba[:, 1]
    p_up = cal_proba[:, 2]
    margin = np.maximum(p_up, p_down) - p_stab
    if len(margin) == 0:
        return 0.0
    base_share = float(np.mean(margin > 0.0))
    if base_share <= target_trade_share:
        return 0.0
    # We want share := mean(margin > delta) <= target. Equivalently, pick
    # delta as the (1 - target)-th quantile of the margin distribution
    # restricted to positive margins, but it's safer to just sweep on a
    # fine grid of empirical margin values to handle ties cleanly.
    sorted_margins = np.sort(margin)
    # The smallest delta achieving share <= target is the value at index
    # `cutoff_idx` of the sorted margins, where cutoff_idx is the count
    # of rows we ARE willing to keep (n - n_trade) shifted by 1 so the
    # threshold strictly excludes the cut row.
    n = len(sorted_margins)
    n_keep = int(np.ceil(target_trade_share * n))
    n_drop = n - n_keep                              # number with margin <= delta
    if n_drop <= 0:
        return 0.0
    if n_drop >= n:
        return float(min(sorted_margins[-1] + 1e-9, NEUTRAL_BAND_DELTA_MAX))
    # The (n_drop - 1)-th smallest margin is the largest one that should
    # FAIL the trade test. Pick delta == that value so that anything
    # strictly greater trades.
    delta = float(sorted_margins[n_drop - 1])
    delta = max(0.0, min(delta, NEUTRAL_BAND_DELTA_MAX))
    return delta


def _cal_tail_pnl_at_delta(
    cal_proba: np.ndarray,
    fwd_cal: np.ndarray,
    delta: float,
    round_trip_cost_frac: float,
) -> tuple[float, int]:
    """Cal-tail post-fee PnL (sum, in pct units) under the trade rule
    `max(P_UP, P_DOWN) > P_STABLE + delta`. Direction is the larger of
    (P_UP, P_DOWN). Returns `(pnl_pct_total, n_trades)`.

    Mirrors `score_fold` exactly so the per-target selection optimises
    the same objective the verdict reports — just measured on the cal
    tail instead of the test slice.
    """
    p_down = cal_proba[:, 0]
    p_stab = cal_proba[:, 1]
    p_up = cal_proba[:, 2]
    direction = np.where(p_up >= p_down, 2, 0)   # 2=UP, 0=DOWN
    margin = np.maximum(p_up, p_down) - p_stab
    if delta > 0.0:
        trade_mask = margin > delta
    else:
        # Equivalent to argmax != STABLE for the legacy delta == 0 case;
        # `margin > 0` matches the `score_fold` legacy branch.
        trade_mask = margin > 0.0
    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return 0.0, 0
    signs = np.where(direction[trade_mask] == 2, 1.0, -1.0)
    signed_ret_pct = signs * fwd_cal[trade_mask] * 100.0
    pnl_per_trade = signed_ret_pct - (round_trip_cost_frac * 100.0)
    return float(np.sum(pnl_per_trade)), n_trades


def _select_trade_share_target(
    cal_proba: np.ndarray,
    cal_df: pd.DataFrame,
    round_trip_cost_frac: float,
    grid: tuple[float, ...] = TRADE_SHARE_TARGET_GRID,
) -> tuple[float, float, float, list[dict]]:
    """Task #591 — pick the `trade_share_target` from `grid` that
    maximizes cal-tail post-fee PnL (in pct, summed). Returns
    `(best_target, best_delta, best_pnl_pct_total, sweep_records)`.

    Selection granularity: this helper picks ONE target for ONE fold's
    cal tail (the last `CAL_TAIL_FRAC` of that fold's training slice).
    Within each (coin, timeframe) slice, `walk_forward_eval` calls this
    helper independently per fold and persists the per-fold pick. The
    verdict then reports the slice-level mean of those per-fold picks
    plus the per-fold list (see `_agg_folds`). This per-fold variant is
    stricter than a single global per-slice tune — each fold gets to
    react to its own cal-tail surface — without losing the per-slice
    summary the operator-facing verdict needs.

    The cal tail is point-in-time-safe (older than the test slice by
    walk-forward construction), so optimising on it does NOT leak
    test-slice information.

    Tie-break rule: among targets that achieve the same best PnL (e.g.
    when several targets land in the same cal-tail trade region), pick
    the smallest target — that produces the largest neutral-band delta,
    which is the more conservative choice (fewer trades, lower
    realized-cost exposure on test).
    """
    fwd_cal = cal_df["forward_return"].astype(float).to_numpy()
    sweep: list[dict] = []
    for target in grid:
        delta = _fit_neutral_band_delta(cal_proba, target)
        pnl, n_trades = _cal_tail_pnl_at_delta(
            cal_proba, fwd_cal, delta, round_trip_cost_frac,
        )
        sweep.append({
            "trade_share_target": float(target),
            "delta_fitted": float(delta),
            "cal_tail_pnl_pct_total": float(pnl),
            "cal_tail_n_trades": int(n_trades),
            "cal_tail_trade_share": (
                float(n_trades) / float(len(cal_proba))
                if len(cal_proba) > 0 else 0.0
            ),
        })
    # argmax over PnL with the smallest-target tie-break described above.
    best = max(sweep, key=lambda r: (r["cal_tail_pnl_pct_total"],
                                       -r["trade_share_target"]))
    return (
        float(best["trade_share_target"]),
        float(best["delta_fitted"]),
        float(best["cal_tail_pnl_pct_total"]),
        sweep,
    )


def fit_predict_with_optional_calibration(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: list[str],
    round_trip_cost_frac: Optional[float] = None,
) -> tuple[np.ndarray, float, float, float, Optional[list[dict]]]:
    """Returns `(test_proba, inv_T, neutral_band_delta, trade_share_target,
    target_sweep)`.

    With `ENABLE_CALIBRATION` False, returns
    `(proba, 1.0, 0.0, TRADE_SHARE_TARGET, None)` — byte-identical to the
    pre-task-587 prediction path; `trade_share_target` is reported as the
    legacy hardcoded sentinel for verdict consistency.

    With the flag on:
        1. Split train into inner (first 1-CAL_TAIL_FRAC) + cal tail (last
           CAL_TAIL_FRAC).
        2. Fit booster on inner-train.
        3. Fit `inv_T` (single-scalar temperature) on cal-tail predictions.
        4. Apply temperature to BOTH cal-tail predictions and test
           predictions.
        5. Task #591 — sweep `TRADE_SHARE_TARGET_GRID` and pick the target
           whose cal-tail-fit `delta` maximises cal-tail post-fee PnL.
           Returns the winning target and its delta. When
           `round_trip_cost_frac` is `None`, falls back to the legacy
           single hardcoded `TRADE_SHARE_TARGET` (preserves backwards
           compat for any external caller that hasn't been updated to
           pass the cost).

    Falls back to the no-calibration path when the cal tail is too small
    to fit a reliable temperature.
    """
    if not ENABLE_CALIBRATION:
        return (
            fit_predict(train, test, feature_names),
            1.0, 0.0, TRADE_SHARE_TARGET, None,
        )
    n_train = len(train)
    cal_start = int(n_train * (1.0 - CAL_TAIL_FRAC))
    inner = train.iloc[:cal_start]
    cal = train.iloc[cal_start:]
    if len(inner) < 50 or len(cal) < 20:
        return (
            fit_predict(train, test, feature_names),
            1.0, 0.0, TRADE_SHARE_TARGET, None,
        )
    test_proba_raw = fit_predict(inner, test, feature_names)
    cal_proba_raw = fit_predict(inner, cal, feature_names)
    y_cal = cal["label_3class"].astype(int).to_numpy()
    inv_T = fit_single_temperature(cal_proba_raw, y_cal)
    cal_proba_cal = apply_single_temperature(cal_proba_raw, inv_T)
    test_proba_cal = apply_single_temperature(test_proba_raw, inv_T)
    if round_trip_cost_frac is None:
        # Legacy single-target path (kept for callers that haven't been
        # updated to the task #591 grid). Behaviour matches the original
        # task #587 implementation: pick the gate-band-midpoint target.
        target_used = TRADE_SHARE_TARGET
        delta = _fit_neutral_band_delta(cal_proba_cal, target_used)
        sweep_records = None
    else:
        target_used, delta, _best_pnl, sweep_records = _select_trade_share_target(
            cal_proba_cal, cal, round_trip_cost_frac, TRADE_SHARE_TARGET_GRID,
        )
    return (
        test_proba_cal,
        float(inv_T),
        float(delta),
        float(target_used),
        sweep_records,
    )


# ── Per-fold scorer ──────────────────────────────────────────────────────
def score_fold(
    test_df: pd.DataFrame,
    proba: np.ndarray,
    round_trip_cost_frac: float,
    neutral_band_delta: float = 0.0,
) -> dict:
    """DA, PnL, n_trades, trade_share computed on a fold's test slice.
    DA is the verification-gate definition: predicted direction in {DOWN, UP}
    matches the true label, counted ONLY on rows whose true label is UP or
    DOWN (rows whose true label is STABLE are EXCLUDED from the DA
    denominator).

    Trade rule:
        * Legacy (delta == 0.0): trade iff `argmax(proba) != STABLE`.
        * Task #587 (delta > 0): trade iff `max(P_UP, P_DOWN) > P_STABLE + delta`.
          The direction is the larger of (P_UP, P_DOWN); the rule reduces to
          the legacy argmax test when delta == 0.

    `trade_share` is the share of all test rows the trade rule fires on.
    PnL: trade only when the rule fires; signed_return = (+1 if direction=UP
    else -1) * forward_return; pnl_pct = signed_return * 100 - rt_cost*100.
    Sum of pnl_pct over all trades = post_fee_pnl_pct_total.
    """
    y_te = test_df["label_3class"].astype(int).to_numpy()
    fwd = test_df["forward_return"].astype(float).to_numpy()
    p_down = proba[:, 0]
    p_stab = proba[:, 1]
    p_up = proba[:, 2]
    direction = np.where(p_up >= p_down, 2, 0)   # 2=UP, 0=DOWN
    margin = np.maximum(p_up, p_down) - p_stab
    if neutral_band_delta > 0.0:
        trade_mask = margin > neutral_band_delta
    else:
        # Legacy argmax-trade rule. Equivalent to `margin > 0` because
        # direction-vs-stable comparison is exactly the same comparison
        # as the argmax-not-stable rule (ties to STABLE go to STABLE).
        trade_mask = np.argmax(proba, axis=1) != 1
    n_test = len(test_df)
    nonstab_mask = y_te != 1
    n_dir_truth = int(nonstab_mask.sum())
    if n_dir_truth > 0:
        # The "predicted class" for the DA denominator is STABLE (=1) on
        # rows the trade rule did NOT fire on, and the chosen direction
        # otherwise — matches the production verification gate, where
        # rows the model declined to trade are counted as STABLE for the
        # purpose of DA on directional-truth rows.
        pred_class = np.where(trade_mask, direction, 1)
        da = float(np.mean(pred_class[nonstab_mask] == y_te[nonstab_mask]))
    else:
        da = float("nan")
    n_trades = int(trade_mask.sum())
    if n_trades > 0:
        signs = np.where(direction[trade_mask] == 2, 1.0, -1.0)
        signed_ret_pct = signs * fwd[trade_mask] * 100.0
        pnl_per_trade = signed_ret_pct - (round_trip_cost_frac * 100.0)
        post_fee_pnl_pct_total = float(np.sum(pnl_per_trade))
        post_fee_pnl_pct_mean = float(np.mean(pnl_per_trade))
        win_rate = float(np.mean(pnl_per_trade > 0))
    else:
        post_fee_pnl_pct_total = 0.0
        post_fee_pnl_pct_mean = float("nan")
        win_rate = float("nan")
    trade_share = n_trades / n_test if n_test > 0 else 0.0
    return {
        "n_test": n_test,
        "directional_accuracy": da,
        "n_dir_truth_rows": n_dir_truth,
        "n_trades": n_trades,
        "trade_share": trade_share,
        "post_fee_pnl_pct_total": post_fee_pnl_pct_total,
        "post_fee_pnl_pct_mean": post_fee_pnl_pct_mean,
        "win_rate": win_rate,
        "neutral_band_delta_used": float(neutral_band_delta),
    }


# ── Walk-forward driver ──────────────────────────────────────────────────
def walk_forward_eval(
    df: pd.DataFrame,
    feature_names: list[str],
    round_trip_cost_frac: float,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    n = len(df)
    if n < 200:
        return []
    fold_size = n // (n_folds + 1)
    out: list[dict] = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_end = min(n, fold_size * (k + 1))
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        if len(train) < 100 or len(test) < 30:
            continue
        proba, inv_T, delta, target_used, target_sweep = (
            fit_predict_with_optional_calibration(
                train, test, feature_names,
                round_trip_cost_frac=round_trip_cost_frac,
            )
        )
        scored = score_fold(
            test, proba, round_trip_cost_frac, neutral_band_delta=delta,
        )
        scored["fold_id"] = k - 1
        scored["train_size"] = len(train)
        scored["test_size"] = len(test)
        # Task #587 — record `inv_T` and the cal-tail-fitted neutral-band
        # delta so the verdict can show how much each lever contributed
        # per fold. With the flag off both are at their identity values
        # (inv_T=1.0, delta=0.0).
        scored["calibration_inv_T"] = float(inv_T)
        scored["calibration_enabled"] = bool(ENABLE_CALIBRATION)
        scored["neutral_band_delta_fitted"] = float(delta)
        # Task #591 — record which `trade_share_target` from the grid
        # sweep won this fold's cal-tail PnL maximisation, plus the full
        # per-target sweep (None when calibration is off / when the cal
        # tail was too small to run the sweep).
        scored["trade_share_target_selected"] = float(target_used)
        scored["trade_share_target_grid"] = list(TRADE_SHARE_TARGET_GRID)
        scored["trade_share_target_sweep"] = target_sweep
        out.append(scored)
    return out


# ── Stage 1: per-candidate ablation ──────────────────────────────────────
def stage1_ablation(
    candidates: list[dict], round_trip_cost_frac: float,
) -> dict:
    out = {"per_candidate": [], "per_timeframe_baseline": {}}
    for tf in TIMEFRAMES:
        try:
            df = load_dataset(tf)
        except FileNotFoundError as exc:
            logger.warning("skip tf=%s: %s", tf, exc)
            continue
        df["coin_idx"] = encode_coin_idx(df, sorted(df["coin_id"].unique().tolist()))
        # The base feature set is the intersection of the registry's
        # FEATURE_COLUMNS with what's actually present in this snapshot.
        # In normal operation all timeframes carry the full schema (refreshed
        # by `scripts/refresh_cached_datasets.py`), but we keep the
        # intersection as a defensive guard against a stray older snapshot.
        base_feats = [c for c in FEATURE_COLUMNS if c in df.columns]
        if "coin_idx" not in base_feats:
            base_feats = base_feats + ["coin_idx"]

        logger.info("tf=%s n=%d base_feats=%d", tf, len(df), len(base_feats))
        t0 = time.time()
        baseline_folds = walk_forward_eval(
            df, base_feats, round_trip_cost_frac,
        )
        logger.info("baseline tf=%s folds=%d elapsed=%.1fs",
                    tf, len(baseline_folds), time.time() - t0)
        out["per_timeframe_baseline"][tf] = {
            "n_rows": int(len(df)),
            "n_features": len(base_feats),
            "folds": baseline_folds,
            "mean_da": _safe_mean([f["directional_accuracy"] for f in baseline_folds]),
            "sum_pnl_pct": _safe_sum([f["post_fee_pnl_pct_total"] for f in baseline_folds]),
        }
        for cand in candidates:
            t0 = time.time()
            try:
                aug_df = df.copy()
                aug_df[cand["name"]] = materialize_candidate(aug_df, cand)
                aug_feats = base_feats + [cand["name"]]
                aug_folds = walk_forward_eval(
                    aug_df, aug_feats, round_trip_cost_frac,
                )
                pair_folds = []
                folds_pos_da = 0
                folds_pos_pnl = 0
                folds_pos_both = 0
                for b, a in zip(baseline_folds, aug_folds):
                    da_delta = a["directional_accuracy"] - b["directional_accuracy"]
                    pnl_delta = a["post_fee_pnl_pct_total"] - b["post_fee_pnl_pct_total"]
                    pos_da = bool(da_delta > 0)
                    pos_pnl = bool(pnl_delta > 0)
                    folds_pos_da += int(pos_da)
                    folds_pos_pnl += int(pos_pnl)
                    folds_pos_both += int(pos_da and pos_pnl)
                    pair_folds.append({
                        "fold_id": a["fold_id"],
                        "baseline_da": b["directional_accuracy"],
                        "aug_da": a["directional_accuracy"],
                        "da_delta": da_delta,
                        "baseline_pnl_pct_total": b["post_fee_pnl_pct_total"],
                        "aug_pnl_pct_total": a["post_fee_pnl_pct_total"],
                        "pnl_delta": pnl_delta,
                        "baseline_n_trades": b["n_trades"],
                        "aug_n_trades": a["n_trades"],
                        "baseline_trade_share": b["trade_share"],
                        "aug_trade_share": a["trade_share"],
                    })
                admitted = folds_pos_both >= STAGE1_MIN_FOLDS_POSITIVE
                out["per_candidate"].append({
                    "name": cand["name"],
                    "bucket": cand["bucket"],
                    "timeframe": tf,
                    "folds": pair_folds,
                    "n_folds": len(pair_folds),
                    "folds_positive_da": folds_pos_da,
                    "folds_positive_pnl": folds_pos_pnl,
                    "folds_positive_both": folds_pos_both,
                    "stage1_admitted": admitted,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "error": None,
                })
                logger.info(
                    "ablation tf=%s name=%s admitted=%s pos_da/pnl/both=%d/%d/%d elapsed=%.1fs",
                    tf, cand["name"], admitted, folds_pos_da, folds_pos_pnl,
                    folds_pos_both, time.time() - t0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("ablation failed tf=%s name=%s", tf, cand["name"])
                out["per_candidate"].append({
                    "name": cand["name"],
                    "bucket": cand.get("bucket"),
                    "timeframe": tf,
                    "folds": [],
                    "stage1_admitted": False,
                    "error": str(exc),
                })
    return out


def _safe_mean(xs: list) -> Optional[float]:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(vals)) if vals else None


def _safe_sum(xs: list) -> Optional[float]:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.sum(vals)) if vals else None


# ── Stage 2: stacked, per-(coin, tf) gate evaluation ─────────────────────
def stage2_stacked_gate_eval(
    stage1: dict,
    candidates_by_name: dict,
    round_trip_cost_frac: float,
) -> dict:
    # Admitted set = any candidate with stage1_admitted=True on ANY timeframe.
    admitted_names = sorted({
        r["name"] for r in stage1["per_candidate"]
        if r.get("stage1_admitted")
    })
    out = {
        "admitted_names_for_stacking": admitted_names,
        "per_slice": [],
        "any_slice_passes_gate": False,
        "passing_slices": [],
    }
    if not admitted_names:
        return out
    for tf in TIMEFRAMES:
        try:
            df_tf = load_dataset(tf)
        except FileNotFoundError:
            continue
        df_tf["coin_idx"] = encode_coin_idx(
            df_tf, sorted(df_tf["coin_id"].unique().tolist()),
        )
        base_feats = [c for c in FEATURE_COLUMNS if c in df_tf.columns]
        if "coin_idx" not in base_feats:
            base_feats = base_feats + ["coin_idx"]

        # Materialize each admitted candidate on the FULL df_tf so the
        # rolling windows respect per-coin grouping with the most context.
        applicable_added = []
        for name in admitted_names:
            cand = candidates_by_name[name]
            try:
                df_tf[name] = materialize_candidate(df_tf, cand)
                applicable_added.append(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("stage2 materialize skip tf=%s name=%s: %s", tf, name, exc)
        aug_feats = base_feats + applicable_added

        for coin in sorted(df_tf["coin_id"].unique().tolist()):
            sub = df_tf[df_tf["coin_id"] == coin].copy()
            sub = sub.sort_values("timestamp_ms").reset_index(drop=True)
            if len(sub) < 200:
                out["per_slice"].append({
                    "coin": coin, "timeframe": tf,
                    "n_rows": int(len(sub)),
                    "skipped_reason": "fewer than 200 rows",
                    "gate_pass": False,
                })
                continue
            try:
                base_folds = walk_forward_eval(
                    sub, base_feats, round_trip_cost_frac,
                )
                aug_folds = walk_forward_eval(
                    sub, aug_feats, round_trip_cost_frac,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("stage2 walk failed tf=%s coin=%s", tf, coin)
                out["per_slice"].append({
                    "coin": coin, "timeframe": tf,
                    "n_rows": int(len(sub)),
                    "skipped_reason": f"walk_forward_eval raised: {exc}",
                    "gate_pass": False,
                })
                continue
            # Aggregate over folds: weighted DA by directional truth count;
            # sum of post-fee PnL; total trades; trade_share = sum trades /
            # sum test rows. This matches how the production verification
            # gate aggregates per-fold metrics into a slice verdict.
            base_agg = _agg_folds(base_folds)
            aug_agg = _agg_folds(aug_folds)
            gate_pass = (
                aug_agg["directional_accuracy"] is not None
                and base_agg["directional_accuracy"] is not None
                and (aug_agg["directional_accuracy"]
                     - base_agg["directional_accuracy"]) > STAGE2_DA_LIFT_FLOOR
                and aug_agg["post_fee_pnl_pct_total"] > STAGE2_PNL_FLOOR_PCT_TOTAL
                and STAGE2_TRADE_SHARE_LO <= aug_agg["trade_share"] <= STAGE2_TRADE_SHARE_HI
                and aug_agg["n_trades"] >= STAGE2_MIN_TRADES
            )
            row = {
                "coin": coin, "timeframe": tf,
                "n_rows": int(len(sub)),
                "baseline": base_agg,
                "augmented": aug_agg,
                "da_lift": (
                    aug_agg["directional_accuracy"] - base_agg["directional_accuracy"]
                    if (aug_agg["directional_accuracy"] is not None
                        and base_agg["directional_accuracy"] is not None)
                    else None
                ),
                "pnl_delta_pct_total": aug_agg["post_fee_pnl_pct_total"]
                                       - base_agg["post_fee_pnl_pct_total"],
                "gate_pass": bool(gate_pass),
                "gate_floor_da_lift": STAGE2_DA_LIFT_FLOOR,
                "gate_floor_pnl_pct_total": STAGE2_PNL_FLOOR_PCT_TOTAL,
                "gate_trade_share_band": [STAGE2_TRADE_SHARE_LO, STAGE2_TRADE_SHARE_HI],
                "gate_min_trades": STAGE2_MIN_TRADES,
            }
            if gate_pass:
                out["any_slice_passes_gate"] = True
                out["passing_slices"].append({
                    "coin": coin, "timeframe": tf,
                    "da": aug_agg["directional_accuracy"],
                    "baseline_da": base_agg["directional_accuracy"],
                    "post_fee_pnl_pct_total": aug_agg["post_fee_pnl_pct_total"],
                    "trade_share": aug_agg["trade_share"],
                    "n_trades": aug_agg["n_trades"],
                })
            out["per_slice"].append(row)
            logger.info(
                "stage2 tf=%s coin=%s gate_pass=%s da_lift=%s pnl_delta=%s",
                tf, coin, gate_pass, row["da_lift"], row["pnl_delta_pct_total"],
            )
    return out


def _agg_folds(folds: list[dict]) -> dict:
    if not folds:
        return {
            "directional_accuracy": None,
            "n_trades": 0,
            "trade_share": 0.0,
            "post_fee_pnl_pct_total": 0.0,
            "n_dir_truth_rows": 0,
            "n_test_rows": 0,
            "trade_share_targets_selected": [],
            "trade_share_target_mean": None,
        }
    total_dir_truth = sum(f["n_dir_truth_rows"] for f in folds)
    if total_dir_truth > 0:
        # Weighted DA = sum over folds of (DA_f * n_dir_truth_f) / total.
        weighted = sum(
            (f["directional_accuracy"] or 0.0) * f["n_dir_truth_rows"]
            for f in folds
        )
        da = weighted / total_dir_truth
    else:
        da = None
    n_trades = sum(f["n_trades"] for f in folds)
    n_test = sum(f["n_test"] for f in folds)
    pnl = sum(f["post_fee_pnl_pct_total"] for f in folds)
    # Task #591 — surface per-fold target selections at the slice level so
    # the verdict can show which targets the cal-tail PnL maximisation
    # picked. The legacy default 0.625 surfaces here when calibration is
    # off (see `fit_predict_with_optional_calibration`).
    targets_selected = [
        float(f["trade_share_target_selected"])
        for f in folds
        if f.get("trade_share_target_selected") is not None
    ]
    target_mean = (
        float(np.mean(targets_selected)) if targets_selected else None
    )
    return {
        "directional_accuracy": da,
        "n_trades": int(n_trades),
        "trade_share": (n_trades / n_test) if n_test > 0 else 0.0,
        "post_fee_pnl_pct_total": float(pnl),
        "n_dir_truth_rows": int(total_dir_truth),
        "n_test_rows": int(n_test),
        "trade_share_targets_selected": targets_selected,
        "trade_share_target_mean": target_mean,
    }


# ── Verdict writer ───────────────────────────────────────────────────────
def write_verdict(
    out_dir: Path, ts: str, stage1: dict, stage2: dict,
    candidates: list[dict], round_trip_cost_frac: float,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{ts}-task580-feature-edge-search.json"
    md_path = out_dir / f"{ts}-task580-feature-edge-verdict.md"

    # Collapse stage1 into per-candidate-overall verdicts.
    by_name: dict[str, dict] = {}
    for r in stage1["per_candidate"]:
        n = r["name"]
        by_name.setdefault(n, {
            "name": n,
            "bucket": r.get("bucket"),
            "timeframes_admitted": [],
            "timeframes_evaluated": [],
            "timeframes_with_error": [],
            "per_tf": {},
        })
        by_name[n]["timeframes_evaluated"].append(r["timeframe"])
        if r.get("error"):
            by_name[n]["timeframes_with_error"].append(r["timeframe"])
        if r.get("stage1_admitted"):
            by_name[n]["timeframes_admitted"].append(r["timeframe"])
        by_name[n]["per_tf"][r["timeframe"]] = {
            "folds_positive_both": r.get("folds_positive_both"),
            "folds_positive_da": r.get("folds_positive_da"),
            "folds_positive_pnl": r.get("folds_positive_pnl"),
            "n_folds": r.get("n_folds"),
            "stage1_admitted": r.get("stage1_admitted"),
            "error": r.get("error"),
        }

    candidate_verdicts = []
    for c in candidates:
        n = c["name"]
        rec = by_name.get(n) or {"name": n, "timeframes_admitted": [],
                                  "timeframes_evaluated": []}
        adm_tfs = rec.get("timeframes_admitted") or []
        if adm_tfs and stage2["any_slice_passes_gate"] and any(
            ps["coin"] and ps["timeframe"] for ps in stage2.get("passing_slices", [])
        ):
            verdict = "promote_to_schema"
        elif adm_tfs:
            verdict = "keep_observing"
        else:
            verdict = "reject"
        candidate_verdicts.append({
            "name": n,
            "bucket": c["bucket"],
            "timeframes_admitted_stage1": adm_tfs,
            "timeframes_evaluated": rec.get("timeframes_evaluated"),
            "verdict": verdict,
            "per_tf_summary": rec.get("per_tf", {}),
        })

    payload = {
        "task": 580,
        "captured_at": ts,
        "round_trip_cost_pct_used": round_trip_cost_frac * 100.0,
        "stage1_admit_rule": (
            f">= {STAGE1_MIN_FOLDS_POSITIVE} of {N_FOLDS} OOS folds positive in BOTH "
            "(directional_accuracy delta, post_fee_pnl_pct_total delta) vs same-fold baseline"
        ),
        "stage2_gate": {
            "da_lift_floor": STAGE2_DA_LIFT_FLOOR,
            "pnl_floor_pct_total": STAGE2_PNL_FLOOR_PCT_TOTAL,
            "trade_share_band": [STAGE2_TRADE_SHARE_LO, STAGE2_TRADE_SHARE_HI],
            "min_trades": STAGE2_MIN_TRADES,
        },
        "any_trade_eligible_slice": stage2["any_slice_passes_gate"],
        "passing_slices": stage2["passing_slices"],
        "stage1": stage1,
        "stage2": stage2,
        "candidate_verdicts": candidate_verdicts,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    # Markdown verdict.
    lines: list[str] = []
    lines.append(f"# Task #580 — Feature edge search verdict ({ts})\n")
    lines.append(
        "Successor to cancelled task #558. Searches deterministic, real-data-only "
        "candidate features for at least one (coin, timeframe) slice that earns "
        "`role=trade` under the **unmodified** promotion gate.\n"
    )
    lines.append("## Hard rules respected\n")
    lines.append("- No edits to `app/training/registry.py`, gate constants, or role JSON.\n")
    lines.append(f"- `shared/forbidden-features.json` cross-checked: prefixes match registry tuple {tuple(FORBIDDEN_FEATURE_PREFIXES)}.\n")
    lines.append("- Every candidate is point-in-time (`max_lookforward == 0`) and computed only from columns already in the persisted dataset snapshot.\n")
    lines.append(f"- Round-trip cost used: {round_trip_cost_frac * 100:.4f}% (sourced from `shared/trading-frictions.json`).\n")
    lines.append(f"- Stage-1 admit rule: {payload['stage1_admit_rule']}.\n")
    lines.append(f"- Stage-2 gate: DA > baseline_DA + {STAGE2_DA_LIFT_FLOOR}, post_fee_pnl_pct_total > {STAGE2_PNL_FLOOR_PCT_TOTAL}, trade_share in [{STAGE2_TRADE_SHARE_LO}, {STAGE2_TRADE_SHARE_HI}], n_trades >= {STAGE2_MIN_TRADES}.\n")

    lines.append("\n## Headline answer — is there ≥1 trade-worthy slice?\n")
    if stage2["any_slice_passes_gate"]:
        lines.append("**YES.** The following (coin, timeframe) slice(s) cleared the unmodified gate when the stacked stage-1-admitted feature set was added:\n")
        lines.append("| coin | timeframe | DA | baseline DA | DA lift | post_fee_pnl_pct_total | trade_share | n_trades |\n|---|---|---|---|---|---|---|---|\n")
        for ps in stage2["passing_slices"]:
            lines.append(
                f"| {ps['coin']} | {ps['timeframe']} | "
                f"{ps['da']:.4f} | {ps['baseline_da']:.4f} | "
                f"{ps['da'] - ps['baseline_da']:+.4f} | "
                f"{ps['post_fee_pnl_pct_total']:+.2f} | "
                f"{ps['trade_share']:.4f} | {ps['n_trades']} |\n"
            )
    else:
        lines.append("**NO.** Every (coin, timeframe) slice failed at least one gate condition under the stacked stage-1-admitted feature set. See per-slice table below.\n")

    lines.append("\n## Stage 1 — per-candidate ablation (pooled per timeframe)\n")
    lines.append("Per-fold deltas vs the same-fold baseline (LightGBM trained on the pooled per-coin frame for each timeframe). 'admitted' = admitted to stage 2.\n\n")
    lines.append("| candidate | bucket | tf | folds+ DA | folds+ PnL | folds+ both | admitted |\n|---|---|---|---|---|---|---|\n")
    for r in stage1["per_candidate"]:
        if r.get("error"):
            lines.append(f"| {r['name']} | {r.get('bucket')} | {r['timeframe']} | — | — | — | error: {r['error'][:80]} |\n")
            continue
        lines.append(
            f"| {r['name']} | {r['bucket']} | {r['timeframe']} | "
            f"{r['folds_positive_da']}/{r['n_folds']} | "
            f"{r['folds_positive_pnl']}/{r['n_folds']} | "
            f"{r['folds_positive_both']}/{r['n_folds']} | "
            f"{'**yes**' if r['stage1_admitted'] else 'no'} |\n"
        )

    lines.append("\n## Stage 1 — per-timeframe baseline reference\n")
    lines.append("| timeframe | n_rows | base_features | mean fold DA | sum fold pnl_pct |\n|---|---|---|---|---|\n")
    for tf, b in stage1.get("per_timeframe_baseline", {}).items():
        mean_da = b["mean_da"]
        pnl = b["sum_pnl_pct"]
        lines.append(
            f"| {tf} | {b['n_rows']} | {b['n_features']} | "
            f"{(mean_da if mean_da is not None else float('nan')):.4f} | "
            f"{(pnl if pnl is not None else float('nan')):+.2f} |\n"
        )

    lines.append("\n## Stage 2 — per-(coin, timeframe) gate evaluation (stacked admitted features)\n")
    lines.append(f"Admitted-for-stacking set: `{stage2['admitted_names_for_stacking']}`.\n\n")
    lines.append("| coin | tf | n_rows | base DA | aug DA | DA lift | base pnl_pct_total | aug pnl_pct_total | aug trade_share | aug n_trades | tuned target (mean of folds) | folds' targets | gate_pass |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for s in stage2["per_slice"]:
        if "skipped_reason" in s:
            lines.append(
                f"| {s['coin']} | {s['timeframe']} | {s['n_rows']} | — | — | — | — | — | — | — | — | — | skipped: {s['skipped_reason']} |\n"
            )
            continue
        b = s["baseline"]; a = s["augmented"]
        bda = b['directional_accuracy']
        ada = a['directional_accuracy']
        lift = s['da_lift']
        # Task #591 — per-slice tuned target (mean of fold-selected
        # targets) and the per-fold target list. Empty list / None mean
        # the cal-tail sweep didn't run for this slice (e.g. calibration
        # off or cal tail too small).
        tgt_mean = a.get("trade_share_target_mean")
        tgts = a.get("trade_share_targets_selected") or []
        tgt_mean_str = f"{tgt_mean:.3f}" if isinstance(tgt_mean, (int, float)) else "—"
        tgts_str = ",".join(f"{t:.3f}" for t in tgts) if tgts else "—"
        lines.append(
            f"| {s['coin']} | {s['timeframe']} | {s['n_rows']} | "
            f"{(bda if bda is not None else float('nan')):.4f} | "
            f"{(ada if ada is not None else float('nan')):.4f} | "
            f"{(lift if lift is not None else float('nan')):+.4f} | "
            f"{b['post_fee_pnl_pct_total']:+.2f} | "
            f"{a['post_fee_pnl_pct_total']:+.2f} | "
            f"{a['trade_share']:.4f} | {a['n_trades']} | "
            f"{tgt_mean_str} | {tgts_str} | "
            f"{'**PASS**' if s['gate_pass'] else 'fail'} |\n"
        )

    lines.append("\n## Per-candidate verdicts\n")
    lines.append("| candidate | bucket | timeframes admitted (stage 1) | verdict |\n|---|---|---|---|\n")
    for cv in candidate_verdicts:
        adm = ",".join(cv["timeframes_admitted_stage1"]) or "—"
        lines.append(f"| {cv['name']} | {cv['bucket']} | {adm} | **{cv['verdict']}** |\n")

    lines.append("\n## Out of scope (reaffirmed)\n")
    lines.append("- This task does NOT modify `FEATURE_COLUMNS`, gate floors, or role JSON.\n")
    lines.append("- Promotion of any `keep_observing` candidate into the production schema is a separate downstream task.\n")
    lines.append("- Re-running the full training campaign is owned by the existing rerun task.\n")

    md_path.write_text("".join(lines))
    return json_path, md_path


# ── Entry point ──────────────────────────────────────────────────────────
def main() -> None:
    forbidden_prefixes = load_forbidden_prefixes()
    candidates = load_candidates()
    for c in candidates:
        assert_candidate_legal(c, forbidden_prefixes)
    candidates_by_name = {c["name"]: c for c in candidates}

    rt_cost = get_round_trip_cost_pct()
    logger.info("loaded %d candidates; round_trip_cost_frac=%s", len(candidates), rt_cost)

    t0 = time.time()
    stage1 = stage1_ablation(candidates, rt_cost)
    logger.info("stage1 complete elapsed=%.1fs", time.time() - t0)

    t0 = time.time()
    stage2 = stage2_stacked_gate_eval(stage1, candidates_by_name, rt_cost)
    logger.info("stage2 complete elapsed=%.1fs", time.time() - t0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = write_verdict(REPORTS_DIR, ts, stage1, stage2,
                                        candidates, rt_cost)
    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {md_path}", flush=True)


if __name__ == "__main__":
    main()
