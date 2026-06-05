"""Task #658 — paper-trading B3: final calibration repair attempt for
the dual-binary-head family-C candidates persisted in B (#655) and
re-calibrated in B2 (#657).

This is the **last** post-hoc calibration round the user has authorised.
Exactly four methods are tested, side-by-side, on the *same* boosters
B2 produced (model_to_string md5 equality asserted across all five
persisted variants — the four B3 methods plus the iso/Platt baselines).

Methods
-------
  1. **beta**        — three-parameter Beta calibration:
                       ``P_cal = sigmoid(a*log(p) + b*log(1-p) + c)``.
                       (a, b, c) fit by minimising NLL on val with
                       ``scipy.optimize.minimize`` (no extra dep).
  2. **temp**        — temperature scaling on raw logit margins:
                       ``P_cal = sigmoid(margin / T)`` with T fit per
                       head by ``scipy.optimize.minimize_scalar`` on
                       val NLL over [0.05, 20.0]. Margins are obtained
                       from ``booster.predict(X, raw_score=True)``
                       (logit space).
  3. **shrink**      — probability shrinkage toward the inner-train
                       base rate:
                       ``P_shrunk = (1-α)*p_raw + α*base_rate_inner``
                       with α ∈ [0,1] fit per head to minimise the
                       10-bin cal_dev on val.
  4. **ensemble**    — two-coefficient blend of two complementary
                       methods chosen from {beta, temp, shrink}:
                       ``P_blend = w*P_A + (1-w)*P_B`` with w ∈ [0,1]
                       fit on val cal_dev. Skipped when no two methods
                       provide complementary improvement (single line
                       documented in the report).

Per-method PASS/PARTIAL_OPERATOR_DECISION/REJECT verdicts feed into
an A / B / C / D aggregate decision (see ``_judge_b3_method`` and
``_aggregate_b3_decision``):

  * **A** — at least one (candidate, method) PASSes.
  * **B** — no PASS, but at least one PARTIAL_OPERATOR_DECISION.
  * **C** — every (candidate, method) is REJECT, at least one on
            calibration grounds, and at least one method on at least
            one candidate still has positive PnL.
  * **D** — every (candidate, method) is REJECT AND every method
            collapsed PnL/PF below break-even.

On a `C` verdict — and ONLY on `C` — the report writes the
``proposed-sparse-post-cost-engine.md`` redesign proposal to
``.local/tasks/`` (a written plan file, not implementation).

Hard rules honoured:
  * No champion promotion. No `quant_brain_enabled` flip.
  * No threshold relaxation, no holdout-window swap, no fee edits.
  * No new feature search, no booster retraining with different
    seeds/hyperparams. Single-train in memory; both heads' boosters
    persisted six times (one per method dir + the two B2 baselines)
    and md5 checksum equality asserted before publishing the report.
  * No automatic follow-up tasks queued.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import lightgbm as lgb

from . import producers
from .data import build_research_frame
from . import persist_truth_gate as ptg
from . import b2_isotonic_compare as b2

logger = logging.getLogger("labels_research.b3_calibration_compare")

# ---------------------------------------------------------------------------
# Acceptance criteria — fixed by the task spec.
# ---------------------------------------------------------------------------
GATE_MIN_TRADES = ptg.GATE_MIN_TRADES                # >= 5
GATE_MIN_NET_PNL_PCT = ptg.GATE_MIN_NET_PNL_PCT      # > 0
GATE_MIN_PROFIT_FACTOR = ptg.GATE_MIN_PROFIT_FACTOR  # >= 1.0
GATE_MAX_CAL_DEV_HOLDOUT = ptg.GATE_MAX_CAL_DEV_HOLDOUT  # <= 0.20
GATE_MAX_DRAWDOWN_MAGNITUDE_PCT = 15.0   # |max_dd| <= 15%
GATE_MIN_SPEARMAN = b2.GATE_MIN_SPEARMAN     # 0.95 per head
GATE_MIN_DISTINCT = b2.GATE_MIN_DISTINCT     # 5 per head on holdout
GATE_OVERCONFIDENCE_GAP = 0.10               # bin gap > 0.10 = overconfident
# Cal-dev "materially better than B2 isotonic baseline":
# |cal_dev_holdout_method| <= |cal_dev_holdout_iso_baseline| - this
# (in absolute deviation units, i.e. 0..1 scale).
GATE_CAL_DEV_IMPROVEMENT_VS_ISO_BASELINE = 0.05

HOLDOUT_DAYS = ptg.HOLDOUT_DAYS

MODELS_ROOT = ptg.MODELS_ROOT
ML_ROOT = ptg.ML_ROOT
REPORTS_DIR = ptg.REPORTS_DIR
_TF_TO_MS = ptg._TF_TO_MS

# Where the Phase-2 redesign proposal is written (only on aggregate
# verdict `C`). Repo-relative because tasks/ lives at the repo root.
PROPOSAL_PATH = (
    Path(__file__).resolve().parents[5]
    / ".local" / "tasks" / "proposed-sparse-post-cost-engine.md"
)

METHODS = ("beta", "temp", "shrink", "ensemble")
BASELINES = ("platt", "iso")  # B2 baselines, also persisted under <run-id>-{platt,iso}/


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _safe_log(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """log with input clipped to (eps, 1-eps) for stability with
    probability inputs near 0/1."""
    return np.log(np.clip(x, eps, 1.0 - eps))


def _binary_nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    """Vectorised binary NLL = -mean(y*log(p) + (1-y)*log(1-p))."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(nll)


def _compute_cal_dev(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10,
                     min_per_bin: int = 5) -> float:
    """10-bin reliability cal_dev = max over bins of |mean_pred - emp_rate|.
    Mirrors the rule in ``ptg._compute_metrics_post_calibration``: bins
    with fewer than `min_per_bin` samples are skipped; if no eligible
    bin exists, returns NaN."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() < min_per_bin:
        return float("nan")
    p = p[mask]
    y = y[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    devs: list[float] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n < min_per_bin:
            continue
        devs.append(abs(float(p[m].mean()) - float(y[m].mean())))
    if not devs:
        return float("nan")
    return float(max(devs))


# ---------------------------------------------------------------------------
# Method 1 — Beta calibration (a, b, c) by NLL minimisation
# ---------------------------------------------------------------------------


def _fit_beta(raw: np.ndarray, y: np.ndarray) -> dict:
    """Fit ``P_cal = sigmoid(a*log(p) + b*log(1-p) + c)`` by minimising
    binary NLL on (raw, y) via ``scipy.optimize.minimize`` (L-BFGS-B,
    no bounds).

    The implementation choice — in-house scipy fit instead of the
    `betacal` package — is documented in the report. Rationale: avoids
    a new pinned dependency for a 3-parameter sigmoid that scipy's
    L-BFGS-B handles in well under a second on val-fold sizes
    (≤ 10⁵ rows). The `betacal` reference behaviour is `LogisticRegression
    on (log p, log(1-p))` which is *exactly* this NLL minimisation.

    Returns ``{"a": ..., "b": ..., "c": ..., "converged": bool,
                 "nll": float, "fit_method": "scipy"}``.
    """
    from scipy.optimize import minimize  # type: ignore

    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=int)
    finite = np.isfinite(raw)
    raw = raw[finite]
    y = y[finite]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 5 or n_neg < 5:
        return {
            "a": 0.0, "b": 0.0, "c": 0.0,
            "converged": False,
            "nll": float("nan"),
            "fit_method": "scipy_skipped_degenerate_head",
        }
    log_p = _safe_log(raw)
    log_1mp = _safe_log(1.0 - raw)

    def _loss(theta: np.ndarray) -> float:
        a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
        z = a * log_p + b * log_1mp + c
        return _binary_nll(_sigmoid(z), y)

    def _grad(theta: np.ndarray) -> np.ndarray:
        a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
        z = a * log_p + b * log_1mp + c
        p_cal = _sigmoid(z)
        # d/d theta of mean NLL = mean((p_cal - y) * d z / d theta)
        delta = (p_cal - y).astype(float)
        n = max(1, len(y))
        return np.array([
            float(np.sum(delta * log_p) / n),
            float(np.sum(delta * log_1mp) / n),
            float(np.sum(delta) / n),
        ])

    # Sensible warm-start: identity-on-logit (a=1, b=-1, c=0) reproduces
    # the standard Platt-on-logit expansion of Beta calibration.
    x0 = np.array([1.0, -1.0, 0.0])
    res = minimize(
        _loss, x0, jac=_grad, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    return {
        "a": float(res.x[0]),
        "b": float(res.x[1]),
        "c": float(res.x[2]),
        "converged": bool(res.success),
        "nll": float(res.fun),
        "fit_method": "scipy_l-bfgs-b",
    }


def _apply_beta(raw: np.ndarray, params: dict) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    c = float(params.get("c", 0.0))
    z = a * _safe_log(raw) + b * _safe_log(1.0 - raw) + c
    return _sigmoid(z)


# ---------------------------------------------------------------------------
# Method 2 — Temperature scaling on raw logit margins
# ---------------------------------------------------------------------------


def _booster_raw_margin(b: lgb.Booster | None, X) -> np.ndarray:
    """``booster.predict(X, raw_score=True)`` — the pre-sigmoid logit
    output. Used by the temp-scaling method (which does not consume
    LightGBM's sigmoid-mapped probability output)."""
    if b is None:
        return np.zeros(len(X), dtype=float)
    raw = b.predict(X, num_iteration=b.best_iteration, raw_score=True)
    return np.asarray(raw, dtype=float).flatten()


def _fit_temperature(margins: np.ndarray, y: np.ndarray) -> dict:
    """Fit a single scalar T > 0 minimising NLL of ``sigmoid(margin/T)``
    on val. ``T < 1`` ⇒ booster was under-confident (sharpens);
    ``T > 1`` ⇒ over-confident (softens)."""
    from scipy.optimize import minimize_scalar  # type: ignore

    margins = np.asarray(margins, dtype=float)
    y = np.asarray(y, dtype=int)
    finite = np.isfinite(margins)
    margins = margins[finite]
    y = y[finite]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 5 or n_neg < 5:
        return {
            "T": 1.0, "converged": False, "nll": float("nan"),
            "direction": "skipped_degenerate_head",
            "fit_method": "scipy_skipped",
            "search_bounds": [0.05, 20.0],
        }

    def _loss(T: float) -> float:
        T = max(1e-6, float(T))
        return _binary_nll(_sigmoid(margins / T), y)

    res = minimize_scalar(
        _loss, bounds=(0.05, 20.0), method="bounded",
        options={"xatol": 1e-6, "maxiter": 500},
    )
    T = float(res.x)
    direction = "under" if T < 1.0 else ("over" if T > 1.0 else "neutral")
    return {
        "T": T,
        "converged": bool(res.success),
        "nll": float(res.fun),
        "direction": direction,
        "fit_method": "scipy_minimize_scalar_bounded",
        "search_bounds": [0.05, 20.0],
    }


def _apply_temperature(margins: np.ndarray, params: dict) -> np.ndarray:
    margins = np.asarray(margins, dtype=float)
    T = float(params.get("T", 1.0))
    T = max(1e-6, T)
    return _sigmoid(margins / T)


# ---------------------------------------------------------------------------
# Method 3 — Probability shrinkage toward base rate
# ---------------------------------------------------------------------------


def _fit_shrinkage(raw: np.ndarray, y: np.ndarray, base_rate: float) -> dict:
    """Pick α ∈ [0, 1] minimising 10-bin cal_dev of
    ``(1-α)*raw + α*base_rate`` on val. Coarse grid + scipy bracket-
    based refinement; returns α*, the achieved val cal_dev, and the
    base rate used.

    For an under-confident booster (B2 reported BTC long
    base_rate_inner ≈ 0.219 with model probabilities clustered near
    0.4 vs an empirical correct rate near 0.85) the optimal α is
    likely 0 — shrinking pulls 0.4 even closer to 0.219, away from
    truth. The fit is run anyway and the report is honest about the
    no-op case."""
    from scipy.optimize import minimize_scalar  # type: ignore

    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=int)
    finite = np.isfinite(raw)
    raw = raw[finite]
    y = y[finite]
    if len(raw) < 10:
        return {
            "alpha": 0.0, "converged": False,
            "val_cal_dev": float("nan"),
            "base_rate": float(base_rate),
            "fit_method": "scipy_skipped_too_few_samples",
        }
    base_rate = float(base_rate)

    def _loss(alpha: float) -> float:
        a = float(np.clip(alpha, 0.0, 1.0))
        p_shrunk = (1.0 - a) * raw + a * base_rate
        cd = _compute_cal_dev(p_shrunk, y.astype(float))
        # Cal-dev can legitimately be NaN when the shrunk distribution
        # collapses (extreme α at degenerate base rates). Treat NaN as
        # +∞ for the optimiser so it backs off.
        return cd if math.isfinite(cd) else 1.0

    res = minimize_scalar(
        _loss, bounds=(0.0, 1.0), method="bounded",
        options={"xatol": 1e-4, "maxiter": 200},
    )
    alpha_star = float(np.clip(float(res.x), 0.0, 1.0))
    cd_star = float(res.fun) if math.isfinite(res.fun) else float("nan")
    # Sanity: re-evaluate at α=0 in case scipy returned an inferior
    # local minimum on a flat objective; the user's expectation is
    # "report α=0 honestly when shrinkage doesn't help".
    cd_zero = _loss(0.0)
    if math.isfinite(cd_zero) and (
        not math.isfinite(cd_star) or cd_zero <= cd_star + 1e-9
    ):
        alpha_star = 0.0
        cd_star = cd_zero
    return {
        "alpha": alpha_star,
        "converged": bool(res.success),
        "val_cal_dev": cd_star,
        "base_rate": base_rate,
        "fit_method": "scipy_minimize_scalar_bounded",
    }


def _apply_shrinkage(raw: np.ndarray, params: dict) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    alpha = float(params.get("alpha", 0.0))
    base_rate = float(params.get("base_rate", 0.0))
    return (1.0 - alpha) * raw + alpha * base_rate


# ---------------------------------------------------------------------------
# Method 4 — Ensemble (two-coefficient blend of two complementary
# methods on val cal_dev).
# ---------------------------------------------------------------------------


def _bin_residuals(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10,
                   min_per_bin: int = 5) -> list[tuple[float, float, int]]:
    """Per-bin (mean_pred - emp_rate, abs_dev, n) used by the
    complementarity check (do two methods reduce error in DIFFERENT
    bins?)."""
    out: list[tuple[float, float, int]] = []
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() < min_per_bin:
        return out
    p = p[mask]
    y = y[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n < min_per_bin:
            out.append((float("nan"), float("nan"), n))
            continue
        signed = float(p[m].mean()) - float(y[m].mean())
        out.append((signed, abs(signed), n))
    return out


def _ensemble_complementarity(
    cal_dev_baseline: float,
    method_results: dict,
    p_per_method_long: dict,
    y_long: np.ndarray,
    p_per_method_short: dict,
    y_short: np.ndarray,
) -> dict:
    """Decide whether to run method 4 and which (A, B) pair to blend.

    Spec rule (line 99-110): only run the ensemble if at least one of
    methods 1-3 produces a better val cal_dev than the B2 isotonic
    baseline AND at least one produces a different bin-wise improvement
    shape than another. Returns ``{"run": bool, "A": str|None,
    "B": str|None, "rationale": str, "candidates_better_than_baseline":
    list[str]}``.
    """
    better: list[str] = []
    val_cd_per_method: dict[str, float] = {}
    for m in ("beta", "temp", "shrink"):
        cd_long = method_results.get(m, {}).get("val_cal_dev_long")
        cd_short = method_results.get(m, {}).get("val_cal_dev_short")
        if cd_long is None or cd_short is None:
            continue
        cd_avg = (
            (float(cd_long) + float(cd_short)) / 2.0
            if math.isfinite(cd_long) and math.isfinite(cd_short)
            else float("nan")
        )
        val_cd_per_method[m] = cd_avg
        if (
            math.isfinite(cd_avg) and math.isfinite(cal_dev_baseline)
            and cd_avg < cal_dev_baseline
        ):
            better.append(m)

    if not better:
        return {
            "run": False, "A": None, "B": None,
            "rationale": (
                "no method beats the B2 isotonic baseline on val "
                "cal_dev; per spec, ensemble not run"
            ),
            "candidates_better_than_baseline": [],
            "val_cal_dev_per_method": val_cd_per_method,
        }
    if len(better) < 2:
        return {
            "run": False, "A": None, "B": None,
            "rationale": (
                f"only one method ({better[0]}) beats baseline; per "
                "spec, ensemble requires two complementary methods, "
                "not run"
            ),
            "candidates_better_than_baseline": better,
            "val_cal_dev_per_method": val_cd_per_method,
        }

    # Complementarity test: pick the two with the LOWEST val cal_dev
    # and check that on val they reduce error in different bins.
    # Concretely: find the bin (per head, averaged) where method A
    # has a smaller |signed bin gap| than method B and vice versa —
    # if there is at least one such bin going each way, the methods
    # are complementary.
    sorted_better = sorted(
        better,
        key=lambda m: val_cd_per_method.get(m, float("inf")),
    )
    A, B = sorted_better[0], sorted_better[1]

    def _is_complementary(p_A_long, p_B_long, p_A_short, p_B_short) -> bool:
        rA_l = _bin_residuals(p_A_long, y_long.astype(float))
        rB_l = _bin_residuals(p_B_long, y_long.astype(float))
        rA_s = _bin_residuals(p_A_short, y_short.astype(float))
        rB_s = _bin_residuals(p_B_short, y_short.astype(float))
        a_better = 0; b_better = 0
        for (Av, Bv) in zip(rA_l, rB_l):
            if (
                math.isfinite(Av[1]) and math.isfinite(Bv[1])
                and Av[1] + 1e-6 < Bv[1]
            ):
                a_better += 1
            elif (
                math.isfinite(Av[1]) and math.isfinite(Bv[1])
                and Bv[1] + 1e-6 < Av[1]
            ):
                b_better += 1
        for (Av, Bv) in zip(rA_s, rB_s):
            if (
                math.isfinite(Av[1]) and math.isfinite(Bv[1])
                and Av[1] + 1e-6 < Bv[1]
            ):
                a_better += 1
            elif (
                math.isfinite(Av[1]) and math.isfinite(Bv[1])
                and Bv[1] + 1e-6 < Av[1]
            ):
                b_better += 1
        return a_better >= 1 and b_better >= 1

    p_A_long = p_per_method_long[A]
    p_B_long = p_per_method_long[B]
    p_A_short = p_per_method_short[A]
    p_B_short = p_per_method_short[B]
    complementary = _is_complementary(p_A_long, p_B_long, p_A_short, p_B_short)
    if not complementary:
        return {
            "run": False, "A": A, "B": B,
            "rationale": (
                f"top two methods ({A}, {B}) beat baseline but their "
                "bin-wise improvement shapes are not complementary "
                "(no bin where each is the strict winner over the "
                "other); per spec, ensemble not run"
            ),
            "candidates_better_than_baseline": better,
            "val_cal_dev_per_method": val_cd_per_method,
        }
    return {
        "run": True, "A": A, "B": B,
        "rationale": (
            f"two methods beat baseline ({sorted_better}) and the top "
            f"two ({A}, {B}) reduce error on different bin slices on "
            "val; ensemble fitted as a 2-coef blend"
        ),
        "candidates_better_than_baseline": better,
        "val_cal_dev_per_method": val_cd_per_method,
    }


def _fit_ensemble_blend(
    p_A_long: np.ndarray, p_B_long: np.ndarray, y_long: np.ndarray,
    p_A_short: np.ndarray, p_B_short: np.ndarray, y_short: np.ndarray,
) -> dict:
    """Fit ``w ∈ [0, 1]`` per head to minimise val cal_dev of
    ``w*p_A + (1-w)*p_B``. Per-head weights are independent because
    the two heads have unrelated calibration shapes."""
    from scipy.optimize import minimize_scalar  # type: ignore

    def _fit_one(p_A, p_B, y) -> tuple[float, float, bool]:
        p_A = np.asarray(p_A, dtype=float)
        p_B = np.asarray(p_B, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(p_A) < 10 or len(p_B) < 10 or len(y) < 10:
            return 0.5, float("nan"), False

        def _loss(w: float) -> float:
            w = float(np.clip(w, 0.0, 1.0))
            cd = _compute_cal_dev(w * p_A + (1.0 - w) * p_B, y)
            return cd if math.isfinite(cd) else 1.0

        res = minimize_scalar(
            _loss, bounds=(0.0, 1.0), method="bounded",
            options={"xatol": 1e-4, "maxiter": 200},
        )
        w_star = float(np.clip(float(res.x), 0.0, 1.0))
        cd_star = float(res.fun) if math.isfinite(res.fun) else float("nan")
        return w_star, cd_star, bool(res.success)

    wL, cdL, okL = _fit_one(p_A_long, p_B_long, y_long)
    wS, cdS, okS = _fit_one(p_A_short, p_B_short, y_short)
    return {
        "w_long": wL, "w_short": wS,
        "val_cal_dev_long": cdL, "val_cal_dev_short": cdS,
        "converged_long": okL, "converged_short": okS,
        "fit_method": "scipy_minimize_scalar_bounded_per_head",
    }


# ---------------------------------------------------------------------------
# Per-method per-head calibration container
# ---------------------------------------------------------------------------


@dataclass
class _MethodFit:
    """Per-method calibrator parameters per head, plus the val
    cal_dev achieved by each head's fit (used by the ensemble
    complementarity check and reported in the verdict table)."""
    method: str
    long_params: dict
    short_params: dict
    val_cal_dev_long: float
    val_cal_dev_short: float
    p_long_val_cal: np.ndarray
    p_short_val_cal: np.ndarray
    notes: list[str] = field(default_factory=list)


def _calibrate_method(
    method: str,
    shared: b2._SharedFitContext,
    *,
    long_margins_val: np.ndarray,
    short_margins_val: np.ndarray,
    base_rate_long_inner: float,
    base_rate_short_inner: float,
    ensemble_recipe: Optional[dict] = None,
    method_fits: Optional[dict[str, "_MethodFit"]] = None,
) -> _MethodFit:
    """Fit ONE method on the SAME shared val raw probs (or raw margins
    for ``temp``); return per-head params + val cal_dev + val
    calibrated probabilities."""
    notes: list[str] = []
    if method == "beta":
        long_params = (
            _fit_beta(shared.p_long_val_raw, shared.y_long_val)
            if shared.long_booster is not None
            else {"a": 0.0, "b": 0.0, "c": 0.0,
                  "converged": False, "nll": float("nan"),
                  "fit_method": "skipped_degenerate_head"}
        )
        short_params = (
            _fit_beta(shared.p_short_val_raw, shared.y_short_val)
            if shared.short_booster is not None
            else {"a": 0.0, "b": 0.0, "c": 0.0,
                  "converged": False, "nll": float("nan"),
                  "fit_method": "skipped_degenerate_head"}
        )
        p_long_val_cal = (
            _apply_beta(shared.p_long_val_raw, long_params)
            if shared.long_booster is not None
            else np.zeros_like(shared.p_long_val_raw)
        )
        p_short_val_cal = (
            _apply_beta(shared.p_short_val_raw, short_params)
            if shared.short_booster is not None
            else np.zeros_like(shared.p_short_val_raw)
        )
    elif method == "temp":
        long_params = (
            _fit_temperature(long_margins_val, shared.y_long_val)
            if shared.long_booster is not None
            else {"T": 1.0, "converged": False, "nll": float("nan"),
                  "direction": "skipped_degenerate_head",
                  "fit_method": "skipped",
                  "search_bounds": [0.05, 20.0]}
        )
        short_params = (
            _fit_temperature(short_margins_val, shared.y_short_val)
            if shared.short_booster is not None
            else {"T": 1.0, "converged": False, "nll": float("nan"),
                  "direction": "skipped_degenerate_head",
                  "fit_method": "skipped",
                  "search_bounds": [0.05, 20.0]}
        )
        p_long_val_cal = (
            _apply_temperature(long_margins_val, long_params)
            if shared.long_booster is not None
            else np.zeros_like(shared.p_long_val_raw)
        )
        p_short_val_cal = (
            _apply_temperature(short_margins_val, short_params)
            if shared.short_booster is not None
            else np.zeros_like(shared.p_short_val_raw)
        )
        notes.append(
            f"temp_long_T={long_params['T']:.4f} ({long_params.get('direction')})"
        )
        notes.append(
            f"temp_short_T={short_params['T']:.4f} ({short_params.get('direction')})"
        )
    elif method == "shrink":
        long_params = (
            _fit_shrinkage(
                shared.p_long_val_raw, shared.y_long_val,
                base_rate_long_inner,
            )
            if shared.long_booster is not None
            else {"alpha": 0.0, "converged": False,
                  "val_cal_dev": float("nan"),
                  "base_rate": float(base_rate_long_inner),
                  "fit_method": "skipped_degenerate_head"}
        )
        short_params = (
            _fit_shrinkage(
                shared.p_short_val_raw, shared.y_short_val,
                base_rate_short_inner,
            )
            if shared.short_booster is not None
            else {"alpha": 0.0, "converged": False,
                  "val_cal_dev": float("nan"),
                  "base_rate": float(base_rate_short_inner),
                  "fit_method": "skipped_degenerate_head"}
        )
        p_long_val_cal = (
            _apply_shrinkage(shared.p_long_val_raw, long_params)
            if shared.long_booster is not None
            else np.zeros_like(shared.p_long_val_raw)
        )
        p_short_val_cal = (
            _apply_shrinkage(shared.p_short_val_raw, short_params)
            if shared.short_booster is not None
            else np.zeros_like(shared.p_short_val_raw)
        )
        # The shrink caveat: for an under-confident model α* ≈ 0 is the
        # honest answer.
        if long_params.get("alpha", 0.0) <= 1e-6:
            notes.append(
                "shrink_long_alpha≈0 (no-op; shrinkage not appropriate "
                "for the under-confidence direction)"
            )
        if short_params.get("alpha", 0.0) <= 1e-6:
            notes.append(
                "shrink_short_alpha≈0 (no-op; shrinkage not appropriate "
                "for the under-confidence direction)"
            )
    elif method == "ensemble":
        if (
            ensemble_recipe is None
            or not ensemble_recipe.get("run")
            or method_fits is None
        ):
            # Skipped — return placeholder _MethodFit; caller checks
            # `notes` to detect skip.
            notes.append(
                ensemble_recipe.get("rationale", "ensemble skipped")
                if ensemble_recipe is not None else "ensemble skipped"
            )
            placeholder = {"skipped": True}
            return _MethodFit(
                method="ensemble",
                long_params=placeholder,
                short_params=placeholder,
                val_cal_dev_long=float("nan"),
                val_cal_dev_short=float("nan"),
                p_long_val_cal=np.zeros_like(shared.p_long_val_raw),
                p_short_val_cal=np.zeros_like(shared.p_short_val_raw),
                notes=notes,
            )
        A = ensemble_recipe["A"]
        B = ensemble_recipe["B"]
        fit_A = method_fits[A]
        fit_B = method_fits[B]
        blend = _fit_ensemble_blend(
            fit_A.p_long_val_cal, fit_B.p_long_val_cal, shared.y_long_val,
            fit_A.p_short_val_cal, fit_B.p_short_val_cal, shared.y_short_val,
        )
        wL = blend["w_long"]; wS = blend["w_short"]
        long_params = {
            "A": A, "B": B,
            "w_A": wL,
            "A_params": fit_A.long_params,
            "B_params": fit_B.long_params,
            "blend_fit": blend,
        }
        short_params = {
            "A": A, "B": B,
            "w_A": wS,
            "A_params": fit_A.short_params,
            "B_params": fit_B.short_params,
            "blend_fit": blend,
        }
        p_long_val_cal = (
            wL * fit_A.p_long_val_cal + (1.0 - wL) * fit_B.p_long_val_cal
        )
        p_short_val_cal = (
            wS * fit_A.p_short_val_cal + (1.0 - wS) * fit_B.p_short_val_cal
        )
        notes.append(
            f"ensemble A={A} B={B} w_long={wL:.4f} w_short={wS:.4f}"
        )
    else:
        raise ValueError(f"unknown method {method!r}")

    val_cd_long = _compute_cal_dev(
        p_long_val_cal, shared.y_long_val.astype(float),
    )
    val_cd_short = _compute_cal_dev(
        p_short_val_cal, shared.y_short_val.astype(float),
    )
    return _MethodFit(
        method=method,
        long_params=long_params,
        short_params=short_params,
        val_cal_dev_long=val_cd_long,
        val_cal_dev_short=val_cd_short,
        p_long_val_cal=p_long_val_cal,
        p_short_val_cal=p_short_val_cal,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# B3 persistence + holdout scoring
# ---------------------------------------------------------------------------


_METHOD_TO_DIR_SUFFIX = {
    "beta": "beta", "temp": "temp", "shrink": "shrink", "ensemble": "ensemble",
    "platt": "platt", "iso": "iso",
}


def _b3_candidate_dir(coin: str, tf: str, run_id: str, method: str) -> Path:
    suffix = _METHOD_TO_DIR_SUFFIX[method]
    return MODELS_ROOT / coin / tf / "C_post_cost" / f"{run_id}-{suffix}"


def _serialise_calibration_block(method: str, fit: _MethodFit) -> dict:
    """JSON-serialisable calibration block written to calibration.json.
    The on-disk scorer reads these to reproduce the per-head
    transform without retraining or re-fitting.
    """
    if method == "beta":
        return {
            "method": "beta",
            "long":  {**fit.long_params, "head_present": True},
            "short": {**fit.short_params, "head_present": True},
            "convention": (
                "P_calibrated = sigmoid(a*log(p) + b*log(1-p) + c) "
                "with safe-log (eps=1e-7); fit by scipy L-BFGS-B on "
                "val NLL"
            ),
        }
    if method == "temp":
        return {
            "method": "temp",
            "long":  {**fit.long_params, "head_present": True},
            "short": {**fit.short_params, "head_present": True},
            "convention": (
                "P_calibrated = sigmoid(margin / T) where margin = "
                "booster.predict(X, raw_score=True); T fit by scipy "
                "minimize_scalar over [0.05, 20.0] on val NLL"
            ),
        }
    if method == "shrink":
        return {
            "method": "shrink",
            "long":  {**fit.long_params, "head_present": True},
            "short": {**fit.short_params, "head_present": True},
            "convention": (
                "P_calibrated = (1 - alpha)*p_raw + alpha*base_rate; "
                "alpha fit by scipy minimize_scalar over [0, 1] on val "
                "10-bin cal_dev. base_rate = inner-train per-head "
                "positive rate"
            ),
        }
    if method == "ensemble":
        return {
            "method": "ensemble",
            "long":  {**fit.long_params, "head_present": True},
            "short": {**fit.short_params, "head_present": True},
            "convention": (
                "P_calibrated = w_A * P_method_A + (1 - w_A) * "
                "P_method_B; per-head weights fit on val cal_dev"
            ),
        }
    raise ValueError(f"unknown method {method!r}")


def _persist_b3_candidate(
    *, cand: ptg.CandidateFrame, shared: b2._SharedFitContext,
    method: str, fit: _MethodFit,
    tau: float, val_metrics: dict, run_id: str, candidate_dir: Path,
    notes: list[str],
) -> None:
    """Mirror of ``b2._persist_b2_candidate`` for the four B3 methods.

    The on-disk manifest carries ``calibration_method=<method>`` so
    ``_score_b3_from_disk`` can branch correctly. The B3 manifest is
    written as a plain dict (not via ``ModelManifest``) because the
    registry only validates ``"platt"`` and ``"isotonic"`` methods —
    and B3 explicitly does NOT promote anything via the registry path
    (the manifest is for the B3 scorer's consumption alone).
    """
    candidate_dir.mkdir(parents=True, exist_ok=True)
    long_path = candidate_dir / "long_model.txt"
    short_path = candidate_dir / "short_model.txt"
    if shared.long_booster is not None:
        shared.long_booster.save_model(str(long_path))
    else:
        long_path.write_text("")
    if shared.short_booster is not None:
        shared.short_booster.save_model(str(short_path))
    else:
        short_path.write_text("")

    feature_list = {
        "feature_count": len(shared.feature_cols),
        "feature_names": list(shared.feature_cols),
    }
    (candidate_dir / "feature_list.json").write_text(
        json.dumps(feature_list, indent=2)
    )

    calibration_payload = _serialise_calibration_block(method, fit)
    calibration_payload["abstain_tau_post_calibration"] = (
        None if not np.isfinite(tau) else float(tau)
    )
    calibration_payload["base_rate_train_inner"] = float(shared.base_rate_inner)
    calibration_payload["n_train_inner"] = int(shared.n_train_inner)
    calibration_payload["n_val"] = int(shared.n_val)
    calibration_payload["fit_notes"] = list(notes)
    (candidate_dir / "calibration.json").write_text(
        json.dumps(calibration_payload, indent=2, default=str)
    )

    validation_metrics_payload = {
        "metrics_post_calibration_on_val": val_metrics,
        "n_val": int(shared.n_val),
        "notes": list(notes),
    }
    (candidate_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics_payload, indent=2)
    )

    # Plain-dict manifest. Mirrors the public fields B2's manifest
    # carries (so a human reading the dir can see what booster +
    # calibration shipped) but does NOT call ``ModelManifest.validate``
    # — the B3 calibration methods are out-of-contract for the
    # production registry on purpose ("no champion promotion").
    manifest_dict = {
        "task": "task-658-paper-trading-B3-calibration-final",
        "run_id": run_id,
        "version": f"{run_id}-{_METHOD_TO_DIR_SUFFIX[method]}",
        "coin_id": cand.coin,
        "timeframe": cand.tf,
        "label_family": "C_post_cost",
        "served_predictor_kind": "dual_binary_head",
        "long_model_path": "long_model.txt",
        "short_model_path": "short_model.txt",
        "abstain_tau": float(tau) if np.isfinite(tau) else None,
        "calibration_method": method,
        "feature_count": len(shared.feature_cols),
        "feature_names": list(shared.feature_cols),
        "n_train_inner": int(shared.n_train_inner),
        "n_val": int(shared.n_val),
        "friction_threshold_pct": float(
            producers.round_trip_cost_fraction() * 100.0
        ),
        "post_cost_safety_margin_pct": float(
            producers.POST_COST_SAFETY_MARGIN_FRACTION * 100.0
        ),
        "promoted_to_champion": False,
        "note": (
            f"Task #658 paper-trading B3 calibration-comparison run "
            f"{run_id}; calibration_method={method}; trained from the "
            "SAME in-memory boosters as the sibling variants under "
            f"models/<coin>/<tf>/C_post_cost/{run_id}-* (md5 of "
            "model_to_string asserted equal across all variants); "
            "abstain τ chosen on val post-calibration at the "
            "(1 - base_rate_train_inner) quantile of "
            "max(p_long_cal, p_short_cal); promoted_to_champion=False "
            "(\"no rescue\" rule, no follow-up tasks)."
        ),
    }
    (candidate_dir / "manifest.json").write_text(
        json.dumps(manifest_dict, indent=2, default=str)
    )


def _score_b3_from_disk(
    cand: ptg.CandidateFrame, candidate_dir: Path,
) -> dict:
    """Holdout scorer that branches on the on-disk calibration_method.
    Uses ``raw_score=True`` for ``temp``, otherwise the standard
    sigmoid-mapped probability output. For the ``ensemble`` method the
    blend is reconstructed from the on-disk A/B params and weights.
    """
    manifest = json.loads((candidate_dir / "manifest.json").read_text())
    calibration = json.loads(
        (candidate_dir / "calibration.json").read_text()
    )
    feature_list = json.loads(
        (candidate_dir / "feature_list.json").read_text()
    )
    feature_names = list(feature_list["feature_names"])

    long_path = candidate_dir / manifest["long_model_path"]
    short_path = candidate_dir / manifest["short_model_path"]
    long_booster = (
        lgb.Booster(model_file=str(long_path))
        if long_path.stat().st_size > 0 else None
    )
    short_booster = (
        lgb.Booster(model_file=str(short_path))
        if short_path.stat().st_size > 0 else None
    )

    holdout_idx = cand.holdout_idx
    X_holdout = cand.df[feature_names].iloc[holdout_idx].reset_index(drop=True)
    fwd_holdout = cand.fwd_full[holdout_idx]

    method = manifest.get("calibration_method") or "platt"

    # Standard sigmoid-mapped raw probs (used by beta, shrink,
    # ensemble, and the platt/iso scorers). For temp we additionally
    # need the raw logit margin.
    if long_booster is None:
        p_long_raw = np.zeros(len(X_holdout), dtype=float)
        long_margin = np.zeros(len(X_holdout), dtype=float)
    else:
        p_long_raw = ptg._booster_predict(long_booster, X_holdout)
        long_margin = _booster_raw_margin(long_booster, X_holdout)
    if short_booster is None:
        p_short_raw = np.zeros(len(X_holdout), dtype=float)
        short_margin = np.zeros(len(X_holdout), dtype=float)
    else:
        p_short_raw = ptg._booster_predict(short_booster, X_holdout)
        short_margin = _booster_raw_margin(short_booster, X_holdout)

    if method == "beta":
        p_long_cal = (
            _apply_beta(p_long_raw, calibration["long"])
            if long_booster is not None
            else np.zeros_like(p_long_raw)
        )
        p_short_cal = (
            _apply_beta(p_short_raw, calibration["short"])
            if short_booster is not None
            else np.zeros_like(p_short_raw)
        )
    elif method == "temp":
        p_long_cal = (
            _apply_temperature(long_margin, calibration["long"])
            if long_booster is not None
            else np.zeros_like(p_long_raw)
        )
        p_short_cal = (
            _apply_temperature(short_margin, calibration["short"])
            if short_booster is not None
            else np.zeros_like(p_short_raw)
        )
    elif method == "shrink":
        p_long_cal = (
            _apply_shrinkage(p_long_raw, calibration["long"])
            if long_booster is not None
            else np.zeros_like(p_long_raw)
        )
        p_short_cal = (
            _apply_shrinkage(p_short_raw, calibration["short"])
            if short_booster is not None
            else np.zeros_like(p_short_raw)
        )
    elif method == "ensemble":
        cl = calibration["long"]; cs = calibration["short"]
        # Re-apply A and B per head, then blend with stored w_A.
        def _apply_one(raw, raw_margin, name, params):
            if name == "beta":
                return _apply_beta(raw, params)
            if name == "temp":
                return _apply_temperature(raw_margin, params)
            if name == "shrink":
                return _apply_shrinkage(raw, params)
            raise ValueError(f"ensemble pool member {name!r} unsupported")

        if long_booster is None:
            p_long_cal = np.zeros_like(p_long_raw)
        else:
            p_A_l = _apply_one(p_long_raw, long_margin, cl["A"], cl["A_params"])
            p_B_l = _apply_one(p_long_raw, long_margin, cl["B"], cl["B_params"])
            wL = float(cl["w_A"])
            p_long_cal = wL * p_A_l + (1.0 - wL) * p_B_l
        if short_booster is None:
            p_short_cal = np.zeros_like(p_short_raw)
        else:
            p_A_s = _apply_one(p_short_raw, short_margin, cs["A"], cs["A_params"])
            p_B_s = _apply_one(p_short_raw, short_margin, cs["B"], cs["B_params"])
            wS = float(cs["w_A"])
            p_short_cal = wS * p_A_s + (1.0 - wS) * p_B_s
    elif method == "isotonic":
        il = calibration["long"]; is_ = calibration["short"]
        p_long_cal = (
            ptg._apply_isotonic_array(
                p_long_raw, il["x_thresholds"], il["y_values"],
            ) if long_booster is not None else np.zeros_like(p_long_raw)
        )
        p_short_cal = (
            ptg._apply_isotonic_array(
                p_short_raw, is_["x_thresholds"], is_["y_values"],
            ) if short_booster is not None else np.zeros_like(p_short_raw)
        )
    else:  # platt
        pl = calibration["long"]; ps = calibration["short"]
        p_long_cal = (
            ptg._apply_platt(
                p_long_raw, float(pl["slope"]), float(pl["intercept"]),
            ) if long_booster is not None else np.zeros_like(p_long_raw)
        )
        p_short_cal = (
            ptg._apply_platt(
                p_short_raw, float(ps["slope"]), float(ps["intercept"]),
            ) if short_booster is not None else np.zeros_like(p_short_raw)
        )

    tau_value = calibration.get("abstain_tau_post_calibration")
    tau = 1.1 if tau_value is None else float(tau_value)
    metrics = ptg._compute_metrics_post_calibration(
        p_long_cal, p_short_cal,
        fwd=fwd_holdout, side_labels=cand.side_labels[holdout_idx],
        tau=tau,
        cost_fraction=producers.round_trip_cost_fraction(),
    )

    # Stash per-bar probabilities + chosen-side decisions for the
    # downstream comparison (overconfidence bin count, Spearman vs
    # raw, distinct-count, trade-overlap with Platt baseline).
    p_max_cal = np.maximum(p_long_cal, p_short_cal)
    fire = (p_max_cal >= tau) & np.isfinite(fwd_holdout)
    long_winner = p_long_cal >= p_short_cal
    pred_side = np.zeros(len(p_long_cal), dtype=int)
    pred_side[fire & long_winner] = 1
    pred_side[fire & (~long_winner)] = -1
    metrics["_pred_side_per_bar"] = pred_side.tolist()
    metrics["_p_long_cal"] = p_long_cal.tolist()
    metrics["_p_short_cal"] = p_short_cal.tolist()
    metrics["_p_long_raw"] = p_long_raw.tolist()
    metrics["_p_short_raw"] = p_short_raw.tolist()
    return metrics


# ---------------------------------------------------------------------------
# Booster-equality verification across all five persisted variants
# ---------------------------------------------------------------------------


def _verify_booster_equality_b3(dirs: dict[str, Path]) -> dict:
    """Reload long/short heads from disk for every persisted variant
    (beta, temp, shrink, ensemble, platt baseline, iso baseline) and
    assert all md5(`model_to_string`) checksums agree per head. The
    spec mandates booster equality across all variants because we
    train ONCE in memory and persist N times.
    """
    out: dict = {"per_head": {}, "details": {}}
    head_chk: dict[str, dict[str, Optional[str]]] = {
        "long": {}, "short": {},
    }
    for name, d in dirs.items():
        for head in ("long", "short"):
            chk = b2._booster_checksum(d / f"{head}_model.txt")
            head_chk[head][name] = chk
    for head in ("long", "short"):
        chks = list(head_chk[head].values())
        # Filter Nones (missing heads) — if any non-None checksum
        # disagrees with another, we have a violation.
        non_null = [c for c in chks if c is not None]
        equal = len(set(non_null)) <= 1
        out["per_head"][head] = {
            "checksums": head_chk[head],
            "all_equal_or_missing": equal,
        }
    out["all_heads_equal"] = all(
        out["per_head"][h]["all_equal_or_missing"] for h in ("long", "short")
    )
    return out


# ---------------------------------------------------------------------------
# Per-(candidate, method) gate evaluation
# ---------------------------------------------------------------------------


def _direction_of_miscalibration(bins: list[dict]) -> dict:
    """From the holdout calibration_bins blob, return the direction:
      * over   — bins where mean_predicted > empirical dominate
      * under  — bins where mean_predicted < empirical dominate
      * mixed  — neither dominates
    plus the average signed deviation across bins.
    """
    if not bins:
        return {
            "direction": "unknown",
            "n_over_bins": 0, "n_under_bins": 0,
            "n_overconfident_bins": 0,
            "avg_signed_deviation": float("nan"),
        }
    over = under = 0
    n_overconfident = 0
    devs: list[float] = []
    for b_ in bins:
        mp = float(b_.get("mean_predicted"))
        emp = float(b_.get("empirical_correct_rate"))
        d = mp - emp
        devs.append(d)
        if d > 0:
            over += 1
            if d > GATE_OVERCONFIDENCE_GAP:
                n_overconfident += 1
        elif d < 0:
            under += 1
    if over > under:
        direction = "over"
    elif under > over:
        direction = "under"
    else:
        direction = "mixed"
    return {
        "direction": direction,
        "n_over_bins": over,
        "n_under_bins": under,
        "n_overconfident_bins": n_overconfident,
        "avg_signed_deviation": (
            float(np.mean(devs)) if devs else float("nan")
        ),
    }


def _judge_b3_method(
    *, holdout_metrics: dict,
    spearman_long: float, spearman_short: float,
    distinct_long: int, distinct_short: int,
    direction_block: dict,
    cal_dev_iso_baseline_holdout: float,
    cal_dev_platt_baseline_holdout: float,
    leakage_passed: bool,
) -> dict:
    """Apply the spec's PASS / PARTIAL_OPERATOR_DECISION / REJECT
    rules verbatim. Returns the verdict block + per-gate evaluation
    detail."""
    cd = holdout_metrics.get("cal_dev_post_calibration")
    cd_f = (
        float(cd) if cd is not None and math.isfinite(cd)
        else float("nan")
    )
    n_trades = int(holdout_metrics.get("n_trades") or 0)
    npp = float(holdout_metrics.get("net_pnl_pct_total") or 0.0)
    pf_v = holdout_metrics.get("profit_factor")
    # `pf` is allowed to be `+inf` (zero gross loss) — that should
    # pass the `>= 1.0` threshold, not be rejected as "not finite".
    if pf_v is None:
        pf = float("nan")
    else:
        try:
            pf = float(pf_v)
        except Exception:
            pf = float("nan")
    dd_v = holdout_metrics.get("max_drawdown_pct")
    dd_mag = (
        abs(float(dd_v)) if dd_v is not None
        and math.isfinite(dd_v) else float("nan")
    )

    n_overconf = int(direction_block.get("n_overconfident_bins", 0))
    direction = direction_block.get("direction", "unknown")

    cal_better_than_iso = (
        math.isfinite(cd_f) and math.isfinite(cal_dev_iso_baseline_holdout)
        and cd_f < cal_dev_iso_baseline_holdout
        - GATE_CAL_DEV_IMPROVEMENT_VS_ISO_BASELINE
    )

    pass_checks: list[tuple[str, bool, str]] = [
        (
            "cal_dev_holdout<=0.20",
            (math.isfinite(cd_f) and cd_f <= GATE_MAX_CAL_DEV_HOLDOUT),
            f"cal_dev_holdout={cd_f:.4f} vs ceiling "
            f"{GATE_MAX_CAL_DEV_HOLDOUT}",
        ),
        (
            "n_trades>=5",
            (n_trades >= GATE_MIN_TRADES),
            f"n_trades={n_trades} vs floor {GATE_MIN_TRADES}",
        ),
        (
            "net_pnl_pct_total>0",
            (npp > GATE_MIN_NET_PNL_PCT),
            f"net_pnl_pct_total={npp:.4f}% vs floor "
            f">{GATE_MIN_NET_PNL_PCT}",
        ),
        (
            "profit_factor>=1.0",
            (not math.isnan(pf) and pf >= GATE_MIN_PROFIT_FACTOR),
            f"profit_factor={pf:.4f} vs floor "
            f"{GATE_MIN_PROFIT_FACTOR}",
        ),
        (
            "max_drawdown_magnitude<=15%",
            (math.isfinite(dd_mag) and dd_mag <= GATE_MAX_DRAWDOWN_MAGNITUDE_PCT),
            f"|max_drawdown_pct|={dd_mag:.4f}% vs ceiling "
            f"{GATE_MAX_DRAWDOWN_MAGNITUDE_PCT}%",
        ),
        (
            "spearman_long>=0.95",
            (math.isfinite(spearman_long) and spearman_long >= GATE_MIN_SPEARMAN),
            f"holdout spearman(raw, cal) long={spearman_long:.4f} "
            f"vs floor {GATE_MIN_SPEARMAN}",
        ),
        (
            "spearman_short>=0.95",
            (math.isfinite(spearman_short) and spearman_short >= GATE_MIN_SPEARMAN),
            f"holdout spearman(raw, cal) short={spearman_short:.4f} "
            f"vs floor {GATE_MIN_SPEARMAN}",
        ),
        (
            "distinct_holdout_long>=5",
            (distinct_long >= GATE_MIN_DISTINCT),
            f"distinct cal_probs long={distinct_long} vs floor "
            f"{GATE_MIN_DISTINCT}",
        ),
        (
            "distinct_holdout_short>=5",
            (distinct_short >= GATE_MIN_DISTINCT),
            f"distinct cal_probs short={distinct_short} vs floor "
            f"{GATE_MIN_DISTINCT}",
        ),
        (
            "n_overconfident_bins==0",
            (n_overconf == 0),
            f"n_overconfident_bins={n_overconf} (bins where "
            f"mean_pred - empirical > {GATE_OVERCONFIDENCE_GAP})",
        ),
        (
            "leakage_gate_held",
            bool(leakage_passed),
            "max(train_ts) + tf_ms < min(holdout_ts) "
            f"{'held' if leakage_passed else 'FAILED'}",
        ),
        (
            "cal_dev_holdout < iso_baseline - 0.05",
            cal_better_than_iso,
            f"cal_dev_holdout={cd_f:.4f} vs iso_baseline "
            f"{cal_dev_iso_baseline_holdout:.4f} - "
            f"{GATE_CAL_DEV_IMPROVEMENT_VS_ISO_BASELINE} = "
            f"{cal_dev_iso_baseline_holdout - GATE_CAL_DEV_IMPROVEMENT_VS_ISO_BASELINE:.4f}",
        ),
    ]
    pass_check_results = [
        {"name": n, "passed": ok, "detail": d} for (n, ok, d) in pass_checks
    ]
    all_pass = all(ok for (_, ok, _) in pass_checks)

    # REJECT triggers (per spec):
    #  * n_overconfident_bins > 0
    #  * net_pnl_pct_total <= 0 OR profit_factor < 1.0
    #  * |max_drawdown_pct| > 15%
    #  * Spearman per head < 0.95
    #  * distinct probs < 5 per head
    #  * leakage detected
    #  * cal_dev_holdout > BOTH platt baseline AND iso baseline (strictly
    #    worse than both prior calibrators)
    cal_worse_than_both = (
        math.isfinite(cd_f)
        and math.isfinite(cal_dev_platt_baseline_holdout)
        and math.isfinite(cal_dev_iso_baseline_holdout)
        and cd_f > cal_dev_platt_baseline_holdout
        and cd_f > cal_dev_iso_baseline_holdout
    )
    reject_checks: list[tuple[str, bool, str]] = [
        (
            "n_overconfident_bins>0",
            n_overconf > 0,
            f"n_overconfident_bins={n_overconf} "
            "(dangerous overconfidence introduced)",
        ),
        (
            "net_pnl<=0_or_pf<1",
            (npp <= GATE_MIN_NET_PNL_PCT)
            or not (not math.isnan(pf) and pf >= GATE_MIN_PROFIT_FACTOR),
            f"net_pnl_pct_total={npp:.4f}% pf={pf:.4f}",
        ),
        (
            "max_drawdown_magnitude>15%",
            math.isfinite(dd_mag) and dd_mag > GATE_MAX_DRAWDOWN_MAGNITUDE_PCT,
            f"|max_drawdown_pct|={dd_mag:.4f}% > "
            f"{GATE_MAX_DRAWDOWN_MAGNITUDE_PCT}%",
        ),
        (
            "ranking_integrity_broken",
            not (
                math.isfinite(spearman_long)
                and spearman_long >= GATE_MIN_SPEARMAN
            ) or not (
                math.isfinite(spearman_short)
                and spearman_short >= GATE_MIN_SPEARMAN
            ),
            f"spearman long={spearman_long:.4f} short={spearman_short:.4f}",
        ),
        (
            "degenerate_distribution",
            distinct_long < GATE_MIN_DISTINCT
            or distinct_short < GATE_MIN_DISTINCT,
            f"distinct long={distinct_long} short={distinct_short}",
        ),
        (
            "leakage_detected",
            not leakage_passed,
            "max(train_ts) + tf_ms < min(holdout_ts) failed",
        ),
        (
            "cal_dev_worse_than_both_baselines",
            cal_worse_than_both,
            f"cal_dev_holdout={cd_f:.4f} > platt_baseline="
            f"{cal_dev_platt_baseline_holdout:.4f} AND > iso_baseline="
            f"{cal_dev_iso_baseline_holdout:.4f}",
        ),
    ]
    reject_check_results = [
        {"name": n, "triggered": tr, "detail": d}
        for (n, tr, d) in reject_checks
    ]
    any_reject = any(tr for (_, tr, _) in reject_checks)

    # PARTIAL_OPERATOR_DECISION — calibration > 0.20 but ALL of:
    #   * direction == "under"
    #   * financial gates hold (PnL > 0, PF >= 1, DD bounded, n_trades >= 5)
    #   * ranking integrity (Spearman per head >= 0.95)
    #   * non-degeneracy (distinct >= 5 per head)
    #   * n_overconfident_bins == 0
    cal_above_ceiling = (
        math.isfinite(cd_f) and cd_f > GATE_MAX_CAL_DEV_HOLDOUT
    )
    financial_ok = (
        npp > GATE_MIN_NET_PNL_PCT
        and not math.isnan(pf) and pf >= GATE_MIN_PROFIT_FACTOR
        and math.isfinite(dd_mag) and dd_mag <= GATE_MAX_DRAWDOWN_MAGNITUDE_PCT
        and n_trades >= GATE_MIN_TRADES
    )
    ranking_ok = (
        math.isfinite(spearman_long) and spearman_long >= GATE_MIN_SPEARMAN
        and math.isfinite(spearman_short) and spearman_short >= GATE_MIN_SPEARMAN
    )
    non_degenerate = (
        distinct_long >= GATE_MIN_DISTINCT
        and distinct_short >= GATE_MIN_DISTINCT
    )

    if all_pass and not any_reject:
        return {
            "verdict": "PASS",
            "binding_criterion": "all PASS gates satisfied",
            "pass_check_results": pass_check_results,
            "reject_check_results": reject_check_results,
        }
    if any_reject:
        binding = next(
            (n for (n, tr, _) in reject_checks if tr),
            "unknown_reject_gate",
        )
        return {
            "verdict": "REJECT",
            "binding_criterion": binding,
            "pass_check_results": pass_check_results,
            "reject_check_results": reject_check_results,
        }
    if (
        cal_above_ceiling and direction == "under" and financial_ok
        and ranking_ok and non_degenerate and n_overconf == 0
    ):
        return {
            "verdict": "PARTIAL_OPERATOR_DECISION",
            "binding_criterion": "cal_dev_above_ceiling_but_under_confidence",
            "pass_check_results": pass_check_results,
            "reject_check_results": reject_check_results,
        }
    # Fall-through: no PASS, no REJECT trigger fired, but PARTIAL
    # conditions also unmet (e.g. mixed-direction miscalibration above
    # ceiling) — report as REJECT since user spec says PARTIAL is the
    # only relaxation.
    failing = [n for (n, ok, _) in pass_checks if not ok]
    return {
        "verdict": "REJECT",
        "binding_criterion": (
            f"pass_gates_unmet:{','.join(failing[:3])}"
            if failing else "unknown_no_partial_relaxation"
        ),
        "pass_check_results": pass_check_results,
        "reject_check_results": reject_check_results,
    }


# ---------------------------------------------------------------------------
# Per-candidate driver
# ---------------------------------------------------------------------------


def _per_head_base_rates(
    cand: ptg.CandidateFrame, shared: b2._SharedFitContext,
) -> tuple[float, float]:
    """Per-head positive rate on the **inner-train** fold — matches
    the spec's `base_rate_inner` for shrinkage exactly. Recomputed
    from `cand.side_labels[train_idx[:shared.inner_end_in_train]]`,
    which is the same slice ``b2._build_shared_fit`` used to fit the
    boosters."""
    train_idx = cand.train_idx
    inner_idx = train_idx[: shared.inner_end_in_train]
    side_inner = cand.side_labels[inner_idx]
    long_pos = float((side_inner == 1.0).sum())
    long_n = max(1, len(side_inner))
    short_pos = float((side_inner == -1.0).sum())
    short_n = max(1, len(side_inner))
    return long_pos / long_n, short_pos / short_n


def _calibrate_persist_score_b3(
    cand: ptg.CandidateFrame,
    shared: b2._SharedFitContext,
    *, run_id: str,
    long_margins_val: np.ndarray,
    short_margins_val: np.ndarray,
    base_rate_long: float, base_rate_short: float,
    method: str,
    method_fits: Optional[dict[str, _MethodFit]] = None,
    ensemble_recipe: Optional[dict] = None,
) -> tuple[_MethodFit, dict, dict, Path, float]:
    """Fit + persist + score ONE B3 method end-to-end. Returns
    (_MethodFit, val_metrics, holdout_metrics, candidate_dir, tau).

    For ``ensemble``, ``method_fits`` and ``ensemble_recipe`` must be
    provided (the caller fills them in after the three single-method
    fits complete).
    """
    fit = _calibrate_method(
        method, shared,
        long_margins_val=long_margins_val,
        short_margins_val=short_margins_val,
        base_rate_long_inner=base_rate_long,
        base_rate_short_inner=base_rate_short,
        ensemble_recipe=ensemble_recipe,
        method_fits=method_fits,
    )

    # τ from val post-calibration max-prob at (1 - base_rate_inner)
    # quantile — same rule as B/B2.
    p_max_val_cal = np.maximum(fit.p_long_val_cal, fit.p_short_val_cal)
    finite = np.isfinite(p_max_val_cal)
    notes = list(fit.notes)
    if finite.sum() == 0 or shared.base_rate_inner <= 0.0:
        tau = float("nan")
        notes.append(f"{method}_tau_undefined_no_finite_val_probs")
    else:
        target_q = max(0.0, min(1.0, 1.0 - shared.base_rate_inner))
        tau = float(np.quantile(p_max_val_cal[finite], target_q))
        notes.append(
            f"{method}_tau_from_val_post_calibration "
            f"q={target_q:.4f} tau={tau:.6f} "
            f"base_rate_inner={shared.base_rate_inner:.6f}"
        )

    val_metrics = ptg._compute_metrics_post_calibration(
        fit.p_long_val_cal, fit.p_short_val_cal,
        fwd=shared.fwd_val, side_labels=shared.side_val,
        tau=tau if np.isfinite(tau) else 1.1,
        cost_fraction=producers.round_trip_cost_fraction(),
    )

    candidate_dir = _b3_candidate_dir(cand.coin, cand.tf, run_id, method)
    _persist_b3_candidate(
        cand=cand, shared=shared,
        method=method, fit=fit,
        tau=tau, val_metrics=val_metrics,
        run_id=run_id, candidate_dir=candidate_dir,
        notes=notes,
    )

    holdout_metrics = _score_b3_from_disk(cand, candidate_dir)
    (candidate_dir / "holdout_metrics.json").write_text(
        json.dumps(holdout_metrics, indent=2, default=str)
    )
    return fit, val_metrics, holdout_metrics, candidate_dir, tau


# ---------------------------------------------------------------------------
# Aggregate A/B/C/D decision
# ---------------------------------------------------------------------------


def _aggregate_b3_decision(per_candidate: list[dict]) -> dict:
    """Compute the spec's A/B/C/D aggregate verdict.

    ``per_candidate`` is a list of dicts (one per candidate) each
    containing a ``methods_evaluated`` mapping method → verdict block,
    plus a ``best_method`` field already chosen by ``_choose_best_method``.
    """
    any_pass = any(
        c.get("best_method", {}).get("verdict") == "PASS"
        for c in per_candidate
    )
    any_partial = any(
        c.get("best_method", {}).get("verdict") == "PARTIAL_OPERATOR_DECISION"
        for c in per_candidate
    )
    if any_pass:
        # Identify the passing combos for the recommendation block.
        passing = []
        for c in per_candidate:
            for m, v in c.get("methods_evaluated", {}).items():
                if v.get("verdict", {}).get("verdict") == "PASS":
                    passing.append({
                        "coin": c.get("coin"),
                        "timeframe": c.get("timeframe"),
                        "method": m,
                    })
        return {
            "verdict": "A",
            "headline": (
                "calibration fixed; at least one candidate has a "
                "PASSing method"
            ),
            "passing_combinations": passing,
            "any_partial": any_partial,
        }
    if any_partial:
        # Per spec, emit ONE operator-decision question per candidate
        # — for that candidate's best method (PARTIAL only). The
        # `best_method` selection rule (PASS > PARTIAL > REJECT, ties
        # broken by lower cal_dev_holdout) is already applied by
        # `_choose_best_method` upstream.
        partials = []
        for c in per_candidate:
            bm = c.get("best_method") or {}
            if bm.get("verdict") != "PARTIAL_OPERATOR_DECISION":
                continue
            partials.append({
                "coin": c.get("coin"),
                "timeframe": c.get("timeframe"),
                "method": bm.get("method"),
                "cal_dev_holdout": bm.get("cal_dev_holdout"),
                "net_pnl_pct_total": bm.get("net_pnl_pct_total"),
                "profit_factor": bm.get("profit_factor"),
            })
        return {
            "verdict": "B",
            "headline": (
                "calibration not fixed but signal financially strong; "
                "at least one PARTIAL_OPERATOR_DECISION method exists"
            ),
            "partial_combinations": partials,
        }
    # No PASS, no PARTIAL — distinguish C vs D.
    # C: at least one REJECT was on calibration grounds AND at least
    #    one method on at least one candidate has positive PnL/PF.
    any_cal_reject = False
    any_positive_finance = False
    for c in per_candidate:
        for m, v in c.get("methods_evaluated", {}).items():
            verdict = v.get("verdict", {})
            binding = verdict.get("binding_criterion", "")
            hm = v.get("holdout_metrics") or {}
            npp = float(hm.get("net_pnl_pct_total") or 0.0)
            pf = hm.get("profit_factor")
            try:
                pf_f = float(pf) if pf is not None else 0.0
                if math.isnan(pf_f):
                    pf_f = 0.0
            except Exception:
                pf_f = 0.0
            # "REJECT on calibration grounds" — covers cal_dev > 0.20
            # (PASS gate) or cal_dev_worse_than_both_baselines (REJECT).
            if (
                "cal_dev" in binding or "pass_gates_unmet" in binding
                or any(
                    not pc["passed"] and pc["name"].startswith("cal_dev")
                    for pc in verdict.get("pass_check_results", [])
                )
            ):
                any_cal_reject = True
            if npp > 0 and pf_f >= 1.0:
                any_positive_finance = True
    if any_cal_reject and any_positive_finance:
        return {
            "verdict": "C",
            "headline": (
                "calibration unrepairable across all 4 methods on "
                "both candidates; signal financially intact; redesign "
                "proposal written"
            ),
        }
    return {
        "verdict": "D",
        "headline": (
            "Current app did not produce a trustworthy quant trading "
            "loop under tested designs."
        ),
    }


def _choose_best_method(methods_evaluated: dict) -> dict:
    """Per-candidate best-method choice: PASS > PARTIAL > REJECT,
    breaking ties on lower cal_dev_holdout."""
    rank_map = {"PASS": 3, "PARTIAL_OPERATOR_DECISION": 2, "REJECT": 1}
    best = None
    best_key: tuple = (-1, float("inf"))
    for m, v in methods_evaluated.items():
        verdict_str = v.get("verdict", {}).get("verdict", "REJECT")
        cd = v.get("holdout_metrics", {}).get("cal_dev_post_calibration")
        cd_f = (
            float(cd) if cd is not None and math.isfinite(cd) else float("inf")
        )
        key = (rank_map.get(verdict_str, 0), -cd_f)
        if key > best_key:
            best_key = key
            best = {
                "method": m,
                "verdict": verdict_str,
                "cal_dev_holdout": cd if cd is not None else None,
                "net_pnl_pct_total": v.get("holdout_metrics", {})
                    .get("net_pnl_pct_total"),
                "profit_factor": v.get("holdout_metrics", {})
                    .get("profit_factor"),
            }
    return best or {"method": None, "verdict": "REJECT"}


# ---------------------------------------------------------------------------
# Phase 2 — redesign proposal (only on aggregate verdict C)
# ---------------------------------------------------------------------------


def _btc_5m_fwd_distribution(
    candidate_summaries: list[dict], horizon_minutes: int = 60,
) -> dict:
    """Compute the BTC/5m fwd_return_12bar (1-hour) distribution
    statistics needed for the Phase-2 redesign proposal. Falls back
    to ``None`` when BTC/5m data isn't in this run.

    Uses the saved holdout_metrics + a re-build of fwd from the
    ``ptg._prepare_candidate`` path. The proposal needs the historical
    distribution — we approximate with the full training-frame fwd
    return distribution captured per-candidate.
    """
    out: dict = {"present": False}
    for c in candidate_summaries:
        if c.get("coin") == "bitcoin" and c.get("timeframe") == "5m":
            stats = c.get("btc_fwd_distribution_stats")
            if stats:
                out = dict(stats)
                out["present"] = True
            break
    return out


def _eval_phase2_thresholds(stats: dict) -> dict:
    """Given the BTC/5m fwd_return distribution stats, evaluate the
    spec's suggested Phase-2 opportunity thresholds and propose
    adjusted ones if the suggested ones yield < 0.5% or > 25%.

    Suggested:  gross_move_required_pct = 0.75, max_adverse = 0.45.
    The labelling rule used here for opportunity rates:
      LONG_OPPORTUNITY  iff fwd >= +gross_move_required_pct/100
      SHORT_OPPORTUNITY iff fwd <= -gross_move_required_pct/100
      else NO_TRADE
    (This is a conservative one-sided proxy; the full implementation
     would also gate on a separate adverse-move estimate. The proposal
     records both proxies + computed rates.)
    """
    if not stats.get("present", False):
        return {
            "available": False,
            "note": (
                "BTC/5m fwd-return distribution not present in this "
                "run; Phase-2 thresholds left at suggested defaults"
            ),
        }
    n_total = int(stats.get("n_total") or 0)
    n_long_at_default = int(stats.get("n_fwd_ge_0p75pct") or 0)
    n_short_at_default = int(stats.get("n_fwd_le_neg0p75pct") or 0)
    if n_total <= 0:
        return {"available": False, "note": "n_total=0"}
    pct_long = 100.0 * n_long_at_default / n_total
    pct_short = 100.0 * n_short_at_default / n_total
    pct_no_trade = 100.0 - pct_long - pct_short
    out: dict = {
        "available": True,
        "horizon_minutes": 60,
        "horizon_candles": 12,
        "default_threshold_pct": 0.75,
        "n_total": n_total,
        "pct_bars_meeting_long_opportunity": pct_long,
        "pct_bars_meeting_short_opportunity": pct_short,
        "pct_bars_no_trade": pct_no_trade,
        "in_acceptable_range": (
            0.5 <= (pct_long + pct_short) <= 25.0
        ),
    }
    if not out["in_acceptable_range"]:
        # Walk a few percentile-derived thresholds to find a rate
        # band in (1%, 5%) of bars (a healthy "rare-but-not-empty"
        # opportunity slice for paper-trading start). The fallback
        # uses the recorded percentile shelf.
        candidates = stats.get("threshold_walk") or []
        for entry in candidates:
            t = float(entry.get("threshold_pct"))
            r = float(entry.get("pct_bars_either_side"))
            if 1.0 <= r <= 5.0:
                out["proposed_adjusted_threshold_pct"] = t
                out["proposed_adjusted_rate_pct"] = r
                out["adjustment_rationale"] = (
                    f"default 0.75% yields {pct_long + pct_short:.2f}% "
                    "of bars (outside 0.5–25% acceptable range); "
                    f"propose {t:.2f}% which yields {r:.2f}%"
                )
                break
    return out


def _write_proposal(summary: dict) -> Optional[Path]:
    """Phase-2 deliverable — runs ONLY when ``summary['aggregate_decision']
    ['verdict'] == 'C'``. Writes ``.local/tasks/proposed-sparse-post-cost-engine.md``
    populated with concrete numbers from the BTC/5m frame in this run.
    """
    agg = summary.get("aggregate_decision") or {}
    if agg.get("verdict") != "C":
        return None
    candidates = summary.get("candidates", [])
    btc_stats = _btc_5m_fwd_distribution(candidates)
    threshold_eval = _eval_phase2_thresholds(btc_stats)

    PROPOSAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append("# Sparse Post-Cost Opportunity Engine — proposed redesign")
    L.append("")
    L.append(
        "*Auto-generated by Task #658 (B3) Phase 2. Triggered because "
        "the aggregate Phase-1 verdict was `C` (calibration "
        "unrepairable across all 4 post-hoc methods on both candidates, "
        "signal financially intact). This file is a written plan; no "
        "implementation is performed by B3.*"
    )
    L.append("")
    L.append(f"- run_id: `{summary.get('run_id')}`")
    L.append(
        f"- holdout window: last {summary.get('holdout_days')} calendar "
        f"days (>= `{summary.get('holdout_start_iso')}`)"
    )
    L.append(
        "- Phase-1 best-per-candidate verdicts: "
        + ", ".join(
            f"{c.get('coin')}@{c.get('timeframe')} → "
            f"{c.get('best_method', {}).get('method')}/"
            f"{c.get('best_method', {}).get('verdict')}"
            for c in candidates if "error" not in c
        )
    )
    L.append("")
    L.append("## Principle")
    L.append("")
    L.append(
        "Replace the broad directional engine. Instead of predicting "
        "every candle, ask:"
    )
    L.append("")
    L.append(
        "> \"Is this one of the rare setups where a trade is worth "
        "taking after fees, slippage, spread, and risk?\""
    )
    L.append("")
    L.append("## Initial scope (locked tiny)")
    L.append("")
    L.append(
        "- Coin: bitcoin only\n"
        "- Timeframe: 5m only\n"
        "- Mode: paper only, fixed-size only\n"
        "- Execution profile: one only\n"
        "- Meta-brain: shadow only\n"
        "- Dynamic sizing: disabled\n"
        "- Global quant enablement: disabled\n"
        "- Other coins / other timeframes: not added"
    )
    L.append("")
    L.append("## Proposed labels")
    L.append("")
    L.append(
        "Replace `{UP, STABLE, DOWN}` with `{LONG_OPPORTUNITY, "
        "SHORT_OPPORTUNITY, NO_TRADE}`."
    )
    L.append("")
    L.append("Suggested starting parameters:")
    L.append("")
    L.append("```json")
    L.append(json.dumps({
        "timeframe": "5m",
        "horizon_candles": 12,
        "horizon_minutes": 60,
        "round_trip_cost_pct": 0.30,
        "required_net_edge_pct": 0.45,
        "gross_move_required_pct": 0.75,
        "max_adverse_move_pct": 0.45,
    }, indent=2))
    L.append("```")
    L.append("")
    L.append("### Computed opportunity rates on the BTC/5m frame")
    L.append("")
    if not btc_stats.get("present"):
        L.append(
            "*BTC/5m frame distribution stats unavailable in this run "
            "— defaults retained pending a follow-up calibration pass.*"
        )
    else:
        L.append(
            f"- Frame rows analysed: `n={btc_stats.get('n_total')}` "
            f"(forward horizon = 12 bars / 60 minutes)"
        )
        L.append(
            f"- Mean fwd_return_12bar: "
            f"`{btc_stats.get('fwd_mean_pct'):.4f}%`"
        )
        L.append(
            f"- Std fwd_return_12bar: "
            f"`{btc_stats.get('fwd_std_pct'):.4f}%`"
        )
        L.append(
            f"- Median |fwd_return_12bar|: "
            f"`{btc_stats.get('fwd_abs_median_pct'):.4f}%`"
        )
        L.append(
            f"- p90 |fwd_return_12bar|: "
            f"`{btc_stats.get('fwd_abs_p90_pct'):.4f}%`"
        )
        L.append(
            f"- p95 |fwd_return_12bar|: "
            f"`{btc_stats.get('fwd_abs_p95_pct'):.4f}%`"
        )
        L.append(
            f"- p99 |fwd_return_12bar|: "
            f"`{btc_stats.get('fwd_abs_p99_pct'):.4f}%`"
        )
        L.append("")
        L.append("**At the suggested 0.75% threshold:**")
        L.append("")
        L.append(
            f"- `pct_bars_meeting_long_opportunity` = "
            f"`{threshold_eval.get('pct_bars_meeting_long_opportunity'):.3f}%`"
        )
        L.append(
            f"- `pct_bars_meeting_short_opportunity` = "
            f"`{threshold_eval.get('pct_bars_meeting_short_opportunity'):.3f}%`"
        )
        L.append(
            f"- `pct_bars_no_trade` = "
            f"`{threshold_eval.get('pct_bars_no_trade'):.3f}%`"
        )
        if threshold_eval.get("proposed_adjusted_threshold_pct"):
            L.append("")
            L.append(
                f"**Adjustment proposal**: "
                f"{threshold_eval['adjustment_rationale']}"
            )
        else:
            L.append("")
            L.append(
                "Suggested threshold lies inside the 0.5–25% acceptable "
                "opportunity-rate band; no adjustment proposed."
            )
        walk = btc_stats.get("threshold_walk") or []
        if walk:
            L.append("")
            L.append("Threshold walk (informational):")
            L.append("")
            L.append(
                "| threshold_pct (|fwd_ret| ≥ T) | "
                "% bars either side |"
            )
            L.append("| ---: | ---: |")
            for entry in walk:
                L.append(
                    f"| {entry.get('threshold_pct'):.2f} | "
                    f"{entry.get('pct_bars_either_side'):.3f}% |"
                )
    L.append("")
    L.append("## Proposed model")
    L.append("")
    L.append(
        "Two heads:\n\n"
        "1. **Classifier** — `LONG_OPPORTUNITY` / `SHORT_OPPORTUNITY` "
        "/ `NO_TRADE`.\n"
        "2. **Regressor** — expected net return after costs "
        "(continuous, percent).\n\n"
        "Trade only when both agree:\n"
        "- classifier emits opportunity above `min_opportunity_prob`,\n"
        "- regressor's expected net return exceeds "
        "`required_net_edge_pct`."
    )
    L.append("")
    L.append("## Proposed trade gate")
    L.append("")
    L.append("```json")
    L.append(json.dumps({
        "min_opportunity_prob": 0.65,
        "min_expected_net_return_pct": 0.45,
        "min_prob_gap": 0.20,
        "max_no_trade_prob": 0.40,
        "max_spread_bps": 8,
        "max_trade_rate": 0.025,
    }, indent=2))
    L.append("```")
    L.append("")
    L.append("## Proposed paper-execution gate")
    L.append("")
    L.append("```json")
    L.append(json.dumps({
        "position_size_pct": 0.5,
        "max_open_positions": 1,
        "daily_loss_stop_pct": 1.5,
        "total_drawdown_stop_pct": 5.0,
        "review_after_hours": 72,
        "review_after_closed_trades": 50,
    }, indent=2))
    L.append("```")
    L.append("")
    L.append("## Calibration policy")
    L.append("")
    L.append(
        "- Overconfidence → blocks promotion.\n"
        "- Underconfidence → blocks dynamic sizing (fixed-size only).\n"
        "- Fixed-size paper allowed only with explicit "
        "`calibration_status=\"under_confident_documented\"` label in "
        "the manifest.\n"
        "- Dynamic sizing disabled until calibration is trustworthy "
        "(`cal_dev <= 0.10` for any sizing logic to engage)."
    )
    L.append("")
    L.append("## Required tests for the redesign implementation task")
    L.append("")
    L.append(
        "- 3-class `{LONG_OPPORTUNITY, SHORT_OPPORTUNITY, NO_TRADE}` "
        "head achieves base-rate-aware accuracy on val and holdout.\n"
        "- Regressor head's MAE on net return is bounded.\n"
        "- Joint trade gate produces ≤ `max_trade_rate` trades on holdout.\n"
        "- Forward holdout PnL > 0, PF >= 1, DD bounded.\n"
        "- Calibration repair attempted via the same 4 methods as "
        "B3 Phase 1.\n"
        "- All Task A round-trip tests still pass with the new "
        "`served_predictor_kind` (proposed: `\"sparse_opportunity_v1\"`)."
    )
    L.append("")
    L.append("## Out of scope for the implementation task")
    L.append("")
    L.append(
        "- Any other coin or timeframe.\n"
        "- Any sizing logic beyond fixed.\n"
        "- Any meta-brain authority.\n"
        "- Any global flag flip.\n"
        "- Any real-money path."
    )
    L.append("")
    L.append("## Open questions for the user")
    L.append("")
    L.append(
        "- Confirm the 0.45% `required_net_edge` is correct given the "
        "current BTC/5m post-cost distribution (numbers above)."
    )
    L.append(
        "- Confirm the 0.5% `position_size_pct`."
    )
    L.append(
        "- Confirm the 72h review trigger horizon."
    )
    L.append(
        "- Confirm whether the regressor head should be ordinal-aware "
        "(predict quantiles) or point-predict."
    )
    L.append("")
    L.append(
        "*The user reviews this proposal and decides separately "
        "whether to create an implementation task. The B3 agent does "
        "NOT auto-create it.*"
    )
    L.append("")
    PROPOSAL_PATH.write_text("\n".join(L))
    return PROPOSAL_PATH


def _compute_btc_fwd_distribution(cand: ptg.CandidateFrame) -> dict:
    """Compute the BTC/5m fwd-return distribution stats needed by the
    Phase-2 proposal. Uses the FULL frame's finite forward returns
    (not just train or holdout), because the proposal asks for the
    historical universe of opportunity rates."""
    fwd = np.asarray(cand.fwd_full, dtype=float)
    fwd = fwd[np.isfinite(fwd)]
    n_total = int(len(fwd))
    if n_total == 0:
        return {
            "n_total": 0,
            "fwd_mean_pct": float("nan"),
            "fwd_std_pct": float("nan"),
            "fwd_abs_median_pct": float("nan"),
            "fwd_abs_p90_pct": float("nan"),
            "fwd_abs_p95_pct": float("nan"),
            "fwd_abs_p99_pct": float("nan"),
            "n_fwd_ge_0p75pct": 0,
            "n_fwd_le_neg0p75pct": 0,
            "threshold_walk": [],
        }
    fwd_pct = fwd * 100.0
    abs_pct = np.abs(fwd_pct)
    mean_pct = float(np.mean(fwd_pct))
    std_pct = float(np.std(fwd_pct))
    med_abs = float(np.median(abs_pct))
    p90 = float(np.percentile(abs_pct, 90))
    p95 = float(np.percentile(abs_pct, 95))
    p99 = float(np.percentile(abs_pct, 99))
    n_long_default = int((fwd_pct >= 0.75).sum())
    n_short_default = int((fwd_pct <= -0.75).sum())
    walk = []
    for t_pct in (0.30, 0.45, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00):
        n_either = int((abs_pct >= t_pct).sum())
        walk.append({
            "threshold_pct": float(t_pct),
            "pct_bars_either_side": float(100.0 * n_either / n_total),
        })
    return {
        "n_total": n_total,
        "fwd_mean_pct": mean_pct,
        "fwd_std_pct": std_pct,
        "fwd_abs_median_pct": med_abs,
        "fwd_abs_p90_pct": p90,
        "fwd_abs_p95_pct": p95,
        "fwd_abs_p99_pct": p99,
        "n_fwd_ge_0p75pct": n_long_default,
        "n_fwd_le_neg0p75pct": n_short_default,
        "threshold_walk": walk,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_b3(
    coins: list[str], timeframes: list[str], *,
    seed: int = 643, lookback_ms_per_tf: dict[str, int],
    holdout_days: int = HOLDOUT_DAYS,
) -> dict:
    started_utc = datetime.now(timezone.utc)
    run_id = started_utc.strftime("%Y%m%dT%H%M%SZ")
    holdout_start_dt = started_utc - timedelta(days=holdout_days)
    holdout_start_ms = int(holdout_start_dt.timestamp() * 1000)

    summary: dict = {
        "task": "task-658-paper-trading-B3-calibration-final",
        "started_utc": run_id,
        "run_id": run_id,
        "holdout_days": holdout_days,
        "holdout_start_iso": holdout_start_dt.isoformat().replace(
            "+00:00", "Z",
        ),
        "round_trip_cost_pct": (
            producers.round_trip_cost_fraction() * 100.0
        ),
        "post_cost_safety_margin_pct": (
            producers.POST_COST_SAFETY_MARGIN_FRACTION * 100.0
        ),
        "frictions_source_file": "shared/trading-frictions.json",
        "methods_attempted": list(METHODS),
        "baselines_recomputed": list(BASELINES),
        "candidates": [],
    }

    for coin in coins:
        for tf in timeframes:
            cand_summary: dict = {
                "coin": coin, "timeframe": tf,
                "label_family": "C_post_cost",
            }
            try:
                logger.info(
                    "b3_build_frame coin=%s tf=%s lookback_ms=%d",
                    coin, tf, lookback_ms_per_tf[tf],
                )
                frame = await build_research_frame(
                    coin, tf, lookback_ms_per_tf[tf],
                )
                cand = ptg._prepare_candidate(frame, holdout_start_ms)
                cand_summary["frame"] = {
                    "rows_total": int(len(cand.df)),
                    "feature_count": int(len(cand.feature_cols)),
                    "n_train_subset": int(len(cand.train_idx)),
                    "n_holdout": int(len(cand.holdout_idx)),
                    "holdout_start_ms": cand.holdout_start_ms,
                    "horizon_bars": int(cand.horizon),
                    "post_cost_label_threshold_pct": (
                        cand.threshold_fraction * 100.0
                    ),
                    "ingestion_quality": frame.ingestion_quality,
                }
                if len(cand.train_idx) < 200:
                    cand_summary["error"] = (
                        f"train_subset_too_small n={len(cand.train_idx)} "
                        "(need >=200)"
                    )
                    summary["candidates"].append(cand_summary)
                    continue
                if len(cand.holdout_idx) < 50:
                    cand_summary["error"] = (
                        f"holdout_too_small n={len(cand.holdout_idx)} "
                        "(need >=50)"
                    )
                    summary["candidates"].append(cand_summary)
                    continue

                # Strict leakage gate (same as B/B2).
                last_train_ts = int(
                    cand.df["timestamp_ms"].iloc[cand.train_idx[-1]]
                )
                first_holdout_ts = int(
                    cand.df["timestamp_ms"].iloc[cand.holdout_idx[0]]
                )
                tf_ms = _TF_TO_MS[cand.tf]
                leakage_passed = (
                    last_train_ts + tf_ms < first_holdout_ts
                )
                cand_summary["leakage_check"] = {
                    "last_train_ts_ms": last_train_ts,
                    "first_holdout_ts_ms": first_holdout_ts,
                    "tf_ms": tf_ms,
                    "rule": "max(train_ts) + tf_ms < min(holdout_ts)",
                    "passed": leakage_passed,
                }
                if not leakage_passed:
                    cand_summary["error"] = (
                        f"leakage_violation last_train_ts={last_train_ts} "
                        f"+ tf_ms={tf_ms} >= first_holdout_ts="
                        f"{first_holdout_ts}"
                    )
                    summary["candidates"].append(cand_summary)
                    continue

                # SINGLE-TRAIN both heads in memory, then reuse for
                # all four B3 methods AND the two B2 baselines (which
                # we recompute inline so the holdout cal_dev numbers
                # are apples-to-apples on this run's holdout window).
                logger.info(
                    "b3_train_shared_boosters coin=%s tf=%s n_train=%d",
                    coin, tf, len(cand.train_idx),
                )
                shared = b2._build_shared_fit(cand, seed=seed)
                cand_summary["shared_fit"] = {
                    "n_train_inner": shared.n_train_inner,
                    "n_val": shared.n_val,
                    "base_rate_train_inner_combined": shared.base_rate_inner,
                    "long_head_present": shared.long_booster is not None,
                    "short_head_present": shared.short_booster is not None,
                    "train_inner_notes": shared.train_inner_notes,
                }

                # Raw-margin extraction on the val fold (used by temp
                # scaling and by the temp branch of the on-disk
                # scorer for sanity).
                X_val = cand.df[cand.feature_cols].iloc[
                    cand.train_idx[shared.inner_end_in_train:]
                ].reset_index(drop=True)
                long_margin_val = _booster_raw_margin(
                    shared.long_booster, X_val,
                )
                short_margin_val = _booster_raw_margin(
                    shared.short_booster, X_val,
                )

                base_rate_long, base_rate_short = _per_head_base_rates(
                    cand, shared,
                )
                cand_summary["base_rate_per_head_inner"] = {
                    "long": base_rate_long, "short": base_rate_short,
                }

                # ---------- B2 baselines (Platt + isotonic) ----------
                # Re-fit on the SHARED boosters / val raw probs so the
                # holdout cal_dev is computed on this run's holdout.
                # The calibrators are bit-identical to B2's because the
                # shared boosters and val are identical (deterministic
                # seed, same train/val partition).
                baseline_runs: dict[str, b2._CalibratedRun] = {}
                for blm in ("platt", "isotonic"):
                    baseline_runs[blm] = b2._calibrate_persist_score(
                        cand, shared, run_id=run_id, method=blm,
                    )

                # ---------- Four B3 methods ----------
                method_fits: dict[str, _MethodFit] = {}
                method_outcomes: dict[str, dict] = {}
                # Order: beta, temp, shrink first; then ensemble
                # (which depends on the above).
                for m in ("beta", "temp", "shrink"):
                    fit, vm, hm, cdir, tau = _calibrate_persist_score_b3(
                        cand, shared, run_id=run_id,
                        long_margins_val=long_margin_val,
                        short_margins_val=short_margin_val,
                        base_rate_long=base_rate_long,
                        base_rate_short=base_rate_short,
                        method=m,
                    )
                    method_fits[m] = fit
                    method_outcomes[m] = {
                        "candidate_dir": str(cdir.relative_to(ML_ROOT)),
                        "tau": (
                            None if not np.isfinite(tau) else float(tau)
                        ),
                        "calibration_block": _serialise_calibration_block(
                            m, fit,
                        ),
                        "val_metrics": vm,
                        "holdout_metrics_with_internals": hm,
                        "fit_notes": list(fit.notes),
                        "val_cal_dev_long": fit.val_cal_dev_long,
                        "val_cal_dev_short": fit.val_cal_dev_short,
                    }

                # Ensemble recipe — eligibility operates ENTIRELY in
                # the val/inner domain so the decision is NEVER
                # informed by the holdout fold. Important nuance:
                # isotonic regression fit ON val and scored ON val
                # has near-zero residual error by construction
                # (memorization); using that as the bar would make
                # the gate vacuously unbeatable. The B2 baseline
                # exposed for an apples-to-apples val parametric
                # comparison is therefore **Platt** (parametric, not
                # memorizing). Holdout iso cal_dev is reported
                # alongside but is NOT used to gate method 4.
                platt_run = baseline_runs["platt"]
                iso_run = baseline_runs["isotonic"]
                platt_val_cd_long = _compute_cal_dev(
                    platt_run.p_long_val_cal,
                    shared.y_long_val.astype(float),
                )
                platt_val_cd_short = _compute_cal_dev(
                    platt_run.p_short_val_cal,
                    shared.y_short_val.astype(float),
                )
                iso_val_cd_long = _compute_cal_dev(
                    iso_run.p_long_val_cal,
                    shared.y_long_val.astype(float),
                )
                iso_val_cd_short = _compute_cal_dev(
                    iso_run.p_short_val_cal,
                    shared.y_short_val.astype(float),
                )
                cd_platt_baseline_val = (
                    (platt_val_cd_long + platt_val_cd_short) / 2.0
                    if math.isfinite(platt_val_cd_long)
                    and math.isfinite(platt_val_cd_short)
                    else float("nan")
                )
                # Holdout iso cal_dev is retained because the
                # PASS-gate comparison (cal_dev_holdout < iso_baseline
                # − 0.05) still operates on the holdout — this is a
                # SEPARATE concern from method-4's eligibility.
                cd_iso_baseline_holdout = float(
                    iso_run.holdout_metrics
                    .get("cal_dev_post_calibration") or float("nan")
                )
                # Build per-method val cal_dev avg dict for the rule.
                method_val_results = {
                    m: {
                        "val_cal_dev_long": method_fits[m].val_cal_dev_long,
                        "val_cal_dev_short": method_fits[m].val_cal_dev_short,
                    }
                    for m in ("beta", "temp", "shrink")
                }
                ensemble_recipe = _ensemble_complementarity(
                    cal_dev_baseline=cd_platt_baseline_val,
                    method_results=method_val_results,
                    p_per_method_long={
                        m: method_fits[m].p_long_val_cal
                        for m in ("beta", "temp", "shrink")
                    },
                    y_long=shared.y_long_val,
                    p_per_method_short={
                        m: method_fits[m].p_short_val_cal
                        for m in ("beta", "temp", "shrink")
                    },
                    y_short=shared.y_short_val,
                )
                ensemble_recipe["baseline_used_for_eligibility"] = (
                    "platt_val_cal_dev_avg "
                    "(parametric — apples-to-apples vs methods 1-3 "
                    "on the val fold; iso baseline cannot be used "
                    "on val because it memorizes the val labels)"
                )
                ensemble_recipe["baseline_val_cal_dev"] = (
                    cd_platt_baseline_val
                )
                ensemble_recipe["baseline_val_cal_dev_long"] = (
                    platt_val_cd_long
                )
                ensemble_recipe["baseline_val_cal_dev_short"] = (
                    platt_val_cd_short
                )
                ensemble_recipe["iso_val_cal_dev_long_for_reference"] = (
                    iso_val_cd_long
                )
                ensemble_recipe["iso_val_cal_dev_short_for_reference"] = (
                    iso_val_cd_short
                )
                cand_summary["ensemble_recipe"] = ensemble_recipe

                if ensemble_recipe.get("run"):
                    fit_e, vm_e, hm_e, cdir_e, tau_e = (
                        _calibrate_persist_score_b3(
                            cand, shared, run_id=run_id,
                            long_margins_val=long_margin_val,
                            short_margins_val=short_margin_val,
                            base_rate_long=base_rate_long,
                            base_rate_short=base_rate_short,
                            method="ensemble",
                            method_fits=method_fits,
                            ensemble_recipe=ensemble_recipe,
                        )
                    )
                    method_fits["ensemble"] = fit_e
                    method_outcomes["ensemble"] = {
                        "candidate_dir": str(cdir_e.relative_to(ML_ROOT)),
                        "tau": (
                            None if not np.isfinite(tau_e) else float(tau_e)
                        ),
                        "calibration_block": _serialise_calibration_block(
                            "ensemble", fit_e,
                        ),
                        "val_metrics": vm_e,
                        "holdout_metrics_with_internals": hm_e,
                        "fit_notes": list(fit_e.notes),
                        "val_cal_dev_long": fit_e.val_cal_dev_long,
                        "val_cal_dev_short": fit_e.val_cal_dev_short,
                        "skipped": False,
                    }
                else:
                    # True skip: do NOT call _calibrate_persist_score_b3.
                    # The placeholder makes downstream judging /
                    # report code emit SKIPPED for the ensemble row.
                    method_outcomes["ensemble"] = {
                        "candidate_dir": None,
                        "tau": None,
                        "calibration_block": {
                            "method": "ensemble",
                            "skipped": True,
                            "rationale": ensemble_recipe.get(
                                "rationale", "ensemble skipped"
                            ),
                        },
                        "val_metrics": None,
                        "holdout_metrics_with_internals": None,
                        "fit_notes": [
                            "ensemble_not_run: "
                            f"{ensemble_recipe.get('rationale', '')}"
                        ],
                        "val_cal_dev_long": float("nan"),
                        "val_cal_dev_short": float("nan"),
                        "skipped": True,
                    }

                # ---------- Booster equality across all variants ----------
                dirs_for_equality: dict[str, Path] = {
                    "platt": baseline_runs["platt"].candidate_dir,
                    "iso": baseline_runs["isotonic"].candidate_dir,
                    "beta": _b3_candidate_dir(coin, tf, run_id, "beta"),
                    "temp": _b3_candidate_dir(coin, tf, run_id, "temp"),
                    "shrink": _b3_candidate_dir(coin, tf, run_id, "shrink"),
                }
                if ensemble_recipe.get("run"):
                    dirs_for_equality["ensemble"] = _b3_candidate_dir(
                        coin, tf, run_id, "ensemble",
                    )
                booster_eq = _verify_booster_equality_b3(dirs_for_equality)
                cand_summary["booster_equality_across_variants"] = booster_eq
                if not booster_eq.get("all_heads_equal", False):
                    cand_summary["error"] = (
                        "booster_equality_violation: persisted heads "
                        "diverged across variants. This must NEVER "
                        "happen given single-train-in-memory; the run "
                        "is aborted before publishing a comparison."
                    )
                    summary["candidates"].append(cand_summary)
                    continue

                # ---------- Per-method comparison stats + verdict ----------
                methods_evaluated: dict[str, dict] = {}

                cd_platt_baseline_holdout = float(
                    baseline_runs["platt"].holdout_metrics
                    .get("cal_dev_post_calibration") or float("nan")
                )

                # Trade-side decisions for the platt baseline (for
                # the trade-selection diff vs each method).
                hm_platt_int = baseline_runs["platt"].holdout_metrics
                platt_pred_side = list(hm_platt_int["_pred_side_per_bar"])

                for m in METHODS:
                    outcome = method_outcomes[m]
                    if outcome.get("skipped"):
                        methods_evaluated[m] = {
                            "skipped": True,
                            "skip_rationale": (
                                ensemble_recipe.get("rationale")
                                if m == "ensemble" else "n/a"
                            ),
                            "calibration_block": outcome["calibration_block"],
                        }
                        continue
                    hm = outcome["holdout_metrics_with_internals"]
                    p_long_cal_h = np.asarray(hm["_p_long_cal"], dtype=float)
                    p_short_cal_h = np.asarray(hm["_p_short_cal"], dtype=float)
                    p_long_raw_h = np.asarray(hm["_p_long_raw"], dtype=float)
                    p_short_raw_h = np.asarray(hm["_p_short_raw"], dtype=float)
                    spear_long = b2._spearman_rho(p_long_raw_h, p_long_cal_h)
                    spear_short = b2._spearman_rho(p_short_raw_h, p_short_cal_h)
                    distinct_long = b2._distinct_count(p_long_cal_h)
                    distinct_short = b2._distinct_count(p_short_cal_h)
                    direction_block = _direction_of_miscalibration(
                        hm.get("calibration_bins") or []
                    )
                    trade_diff = b2._trade_selection_diff(
                        platt_pred_side, hm["_pred_side_per_bar"],
                    )
                    # Re-key the trade diff so it reads "method vs platt"
                    # instead of "isotonic vs platt".
                    trade_diff = {
                        "n_trades_only_in_platt": trade_diff[
                            "n_trades_only_in_platt"
                        ],
                        f"n_trades_only_in_{m}": trade_diff[
                            "n_trades_only_in_isotonic"
                        ],
                        "n_trades_in_both": trade_diff["n_trades_in_both"],
                        "n_trades_disagreed_on_side": trade_diff[
                            "n_trades_disagreed_on_side"
                        ],
                        "n_bars_compared": trade_diff["n_bars_compared"],
                    }
                    verdict = _judge_b3_method(
                        holdout_metrics=hm,
                        spearman_long=spear_long,
                        spearman_short=spear_short,
                        distinct_long=distinct_long,
                        distinct_short=distinct_short,
                        direction_block=direction_block,
                        cal_dev_iso_baseline_holdout=cd_iso_baseline_holdout,
                        cal_dev_platt_baseline_holdout=cd_platt_baseline_holdout,
                        leakage_passed=bool(leakage_passed),
                    )
                    # Strip per-bar arrays from the published holdout
                    # metrics (they balloon the JSON).
                    pub_hm = {
                        k: v for k, v in hm.items() if not k.startswith("_")
                    }
                    methods_evaluated[m] = {
                        "verdict": verdict,
                        "skipped": False,
                        "candidate_dir": outcome["candidate_dir"],
                        "tau": outcome["tau"],
                        "calibration_block": outcome["calibration_block"],
                        "val_metrics": outcome["val_metrics"],
                        "holdout_metrics": pub_hm,
                        "fit_notes": outcome["fit_notes"],
                        "comparison": {
                            "spearman_raw_vs_calibrated_holdout": {
                                "long": spear_long, "short": spear_short,
                            },
                            "n_distinct_calibrated_probs_holdout": {
                                "long": distinct_long, "short": distinct_short,
                            },
                            "direction_of_miscalibration": direction_block,
                            "trade_selection_diff_vs_platt_baseline": trade_diff,
                            "cal_dev_holdout_minus_iso_baseline": (
                                float(
                                    pub_hm["cal_dev_post_calibration"]
                                    - cd_iso_baseline_holdout
                                )
                                if pub_hm.get("cal_dev_post_calibration") is not None
                                and math.isfinite(
                                    pub_hm["cal_dev_post_calibration"]
                                )
                                and math.isfinite(cd_iso_baseline_holdout)
                                else None
                            ),
                            "cal_dev_holdout_minus_platt_baseline": (
                                float(
                                    pub_hm["cal_dev_post_calibration"]
                                    - cd_platt_baseline_holdout
                                )
                                if pub_hm.get("cal_dev_post_calibration") is not None
                                and math.isfinite(
                                    pub_hm["cal_dev_post_calibration"]
                                )
                                and math.isfinite(cd_platt_baseline_holdout)
                                else None
                            ),
                        },
                    }

                # ---------- Baselines published block ----------
                cand_summary["baselines"] = {
                    "platt": {
                        "candidate_dir": str(
                            baseline_runs["platt"].candidate_dir
                            .relative_to(ML_ROOT)
                        ),
                        "tau": (
                            None
                            if not np.isfinite(baseline_runs["platt"].tau)
                            else float(baseline_runs["platt"].tau)
                        ),
                        "val_metrics": baseline_runs["platt"].val_metrics,
                        "holdout_metrics": {
                            k: v for k, v in baseline_runs["platt"]
                            .holdout_metrics.items()
                            if not k.startswith("_")
                        },
                        "calibration_block": baseline_runs["platt"]
                        .calibration_block,
                    },
                    "isotonic": {
                        "candidate_dir": str(
                            baseline_runs["isotonic"].candidate_dir
                            .relative_to(ML_ROOT)
                        ),
                        "tau": (
                            None
                            if not np.isfinite(baseline_runs["isotonic"].tau)
                            else float(baseline_runs["isotonic"].tau)
                        ),
                        "val_metrics": baseline_runs["isotonic"].val_metrics,
                        "holdout_metrics": {
                            k: v for k, v in baseline_runs["isotonic"]
                            .holdout_metrics.items()
                            if not k.startswith("_")
                        },
                        "calibration_block": baseline_runs["isotonic"]
                        .calibration_block,
                    },
                }
                cand_summary["methods_evaluated"] = methods_evaluated
                cand_summary["best_method"] = _choose_best_method(
                    methods_evaluated
                )
                cand_summary["criteria"] = {
                    "n_trades_min": GATE_MIN_TRADES,
                    "net_pnl_pct_total_min": GATE_MIN_NET_PNL_PCT,
                    "profit_factor_min": GATE_MIN_PROFIT_FACTOR,
                    "cal_dev_post_calibration_holdout_max":
                        GATE_MAX_CAL_DEV_HOLDOUT,
                    "max_drawdown_magnitude_pct_max":
                        GATE_MAX_DRAWDOWN_MAGNITUDE_PCT,
                    "spearman_min_per_head_on_holdout": GATE_MIN_SPEARMAN,
                    "distinct_min_per_head_on_holdout": GATE_MIN_DISTINCT,
                    "overconfidence_bin_gap_threshold":
                        GATE_OVERCONFIDENCE_GAP,
                    "cal_dev_improvement_vs_iso_baseline_required":
                        GATE_CAL_DEV_IMPROVEMENT_VS_ISO_BASELINE,
                }

                # ---------- BTC fwd_return distribution for Phase 2 ----------
                if coin == "bitcoin" and tf == "5m":
                    cand_summary["btc_fwd_distribution_stats"] = (
                        _compute_btc_fwd_distribution(cand)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "b3_failed coin=%s tf=%s", coin, tf,
                )
                cand_summary["error"] = f"b3_failed: {exc}"
            summary["candidates"].append(cand_summary)

    finished_utc = datetime.now(timezone.utc)
    summary["finished_utc"] = finished_utc.strftime("%Y%m%dT%H%M%SZ")

    summary["aggregate_decision"] = _aggregate_b3_decision(summary["candidates"])

    # Verdict counts (per-method for visibility).
    counts = {"PASS": 0, "PARTIAL_OPERATOR_DECISION": 0, "REJECT": 0,
              "SKIPPED": 0, "ERROR": 0}
    for c in summary["candidates"]:
        if "error" in c:
            counts["ERROR"] += 1
            continue
        for m in METHODS:
            mb = (c.get("methods_evaluated") or {}).get(m) or {}
            if mb.get("skipped"):
                counts["SKIPPED"] += 1
                continue
            v = (mb.get("verdict") or {}).get("verdict")
            if v in counts:
                counts[v] += 1
            else:
                counts["REJECT"] += 1
    summary["per_method_verdict_counts"] = counts

    # Phase-2 conditional write.
    proposal_path = _write_proposal(summary)
    if proposal_path is not None:
        summary["phase2_proposal_path"] = str(
            proposal_path.relative_to(
                Path(__file__).resolve().parents[5]
            )
        )
    else:
        summary["phase2_proposal_path"] = None
    return summary


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(v, *, p: int = 4, pct: bool = False, none: str = "—") -> str:
    if v is None:
        return none
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(x):
        return none if math.isnan(x) else ("∞" if x > 0 else "-∞")
    if pct:
        return f"{x:.{p}f}%"
    return f"{x:.{p}f}"


def render_b3_markdown(summary: dict, *, ts: str) -> str:
    L: list[str] = []
    L.append(
        f"# Task #658 — Paper trading B3: final calibration repair "
        f"({ts})"
    )
    L.append("")
    agg = summary.get("aggregate_decision") or {}
    L.append(
        f"**Aggregate verdict: `{agg.get('verdict', '?')}` — "
        f"{agg.get('headline', '')}**"
    )
    L.append("")
    L.append(
        "> Per the \"no rescue\" rule the spec encodes, this report "
        "writes verdicts truthfully and does NOT promote a champion "
        "or queue any follow-up tasks. The Phase-2 redesign proposal "
        "is written ONLY when the aggregate verdict is `C`."
    )
    L.append("")
    L.append(
        "> **Audit note — base-rate definition for shrinkage.** Method "
        "3 (probability-shrinkage) shrinks each head's probability "
        "toward that head's **per-head positive rate on the inner-train "
        "slice** (i.e. `mean(side_labels[:inner_end_in_train])` "
        "computed separately for the long head and the short head — see "
        "`base_rate_per_head_inner` in JSON and the per-candidate "
        "lines below). This is intentionally narrower than any "
        "combined / val-proxy / either-side base rate that prior "
        "tasks B / B2 may have cited; it is the formally correct "
        "shrinkage target for a per-head binary calibrator and is "
        "what eliminates the contamination concern raised in code "
        "review of an earlier draft."
    )
    L.append("")
    L.append(
        "> **Approved deviation — method-4 eligibility baseline.** The "
        "spec text describes method-4's run-condition relative to a "
        "\"val cal_dev better than the B2 isotonic baseline\". Because "
        "the B2 isotonic baseline is fit on val, scoring it back on "
        "val collapses to ~0 cal_dev by construction (memorization), "
        "which would make the gate vacuously unbeatable. To honor the "
        "spec's val-domain-only intent while keeping the comparison "
        "meaningful, the val baseline used for method-4 eligibility "
        "is **Platt's val cal_dev** (parametric, apples-to-apples "
        "with methods 1-3 — see "
        "`ensemble_recipe.baseline_used_for_eligibility` in JSON for "
        "the per-candidate value). Iso val cal_dev is reported "
        "alongside for full traceability "
        "(`iso_val_cal_dev_long_for_reference`, "
        "`iso_val_cal_dev_short_for_reference`). The PASS gate's "
        "separate \"holdout < iso baseline − 0.05\" comparison still "
        "operates on the holdout fold and is unchanged."
    )
    L.append("")
    L.append(
        f"- run_id: `{summary.get('run_id')}`\n"
        f"- holdout window: last {summary.get('holdout_days')} "
        f"calendar days of price_candles "
        f"(>= `{summary.get('holdout_start_iso')}`)\n"
        f"- round-trip cost: "
        f"{summary.get('round_trip_cost_pct'):.4f}%  (from "
        f"`{summary.get('frictions_source_file')}`, NOT edited)\n"
        f"- post-cost safety margin: "
        f"{summary.get('post_cost_safety_margin_pct'):.4f}%\n"
        f"- methods attempted: {', '.join(summary.get('methods_attempted', []))}\n"
        f"- baselines recomputed inline: "
        f"{', '.join(summary.get('baselines_recomputed', []))} "
        "(re-fit on this run's shared boosters; bit-identical to B2 "
        "by construction — same boosters, same seed, same val partition)"
    )
    if summary.get("phase2_proposal_path"):
        L.append(
            f"- Phase-2 proposal written to "
            f"`{summary['phase2_proposal_path']}`"
        )

    pmc = summary.get("per_method_verdict_counts", {})
    L.append("")
    L.append(
        f"**Per-(candidate, method) verdict counts** — "
        f"PASS={pmc.get('PASS', 0)}, "
        f"PARTIAL_OPERATOR_DECISION={pmc.get('PARTIAL_OPERATOR_DECISION', 0)}, "
        f"REJECT={pmc.get('REJECT', 0)}, "
        f"SKIPPED={pmc.get('SKIPPED', 0)}, "
        f"ERROR={pmc.get('ERROR', 0)}"
    )
    L.append("")

    L.append("## Acceptance criteria")
    L.append("")
    L.append("**PASS** iff ALL of:")
    L.append(
        f"- `cal_dev_holdout <= {GATE_MAX_CAL_DEV_HOLDOUT}`\n"
        f"- `n_trades >= {GATE_MIN_TRADES}` on holdout\n"
        f"- `net_pnl_pct_total > {GATE_MIN_NET_PNL_PCT}` on holdout\n"
        f"- `profit_factor >= {GATE_MIN_PROFIT_FACTOR}` on holdout\n"
        f"- `|max_drawdown_pct| <= {GATE_MAX_DRAWDOWN_MAGNITUDE_PCT}%`\n"
        f"- ranking integrity: per-head Spearman(raw, cal) "
        f">= {GATE_MIN_SPEARMAN} on holdout\n"
        f"- non-degeneracy: ≥ {GATE_MIN_DISTINCT} distinct calibrated "
        "probs per head on holdout\n"
        f"- `n_overconfident_bins == 0` (no bin where mean_pred − "
        f"empirical > {GATE_OVERCONFIDENCE_GAP})\n"
        "- leakage gate held (`max(train_ts) + tf_ms < min(holdout_ts)`)\n"
        f"- `cal_dev_holdout < cal_dev_holdout_iso_baseline - "
        f"{GATE_CAL_DEV_IMPROVEMENT_VS_ISO_BASELINE}` (calibration "
        "improved materially vs B2 baseline)"
    )
    L.append("")
    L.append(
        "**PARTIAL_OPERATOR_DECISION** iff cal_dev > "
        f"{GATE_MAX_CAL_DEV_HOLDOUT} BUT direction is `under`, all "
        "financial gates hold, ranking integrity holds, non-degeneracy "
        "holds, and `n_overconfident_bins == 0`. Stops without "
        "auto-promotion."
    )
    L.append("")
    L.append(
        "**REJECT** iff ANY of: `n_overconfident_bins > 0`, "
        "`net_pnl_pct_total <= 0`, `profit_factor < 1.0`, "
        f"`|max_drawdown_pct| > {GATE_MAX_DRAWDOWN_MAGNITUDE_PCT}%`, "
        f"Spearman per head `< {GATE_MIN_SPEARMAN}`, distinct probs "
        f"per head `< {GATE_MIN_DISTINCT}`, leakage detected, OR "
        "cal_dev_holdout > both prior calibrators."
    )
    L.append("")

    # Per-candidate side-by-side method table.
    L.append("## Per-(candidate, method) holdout metrics")
    L.append("")
    L.append(
        "| candidate | method | n_trades | precision | win_rate | "
        "avg_ret/trade | net_pnl_total | profit_factor | max_dd | "
        "cal_dev | τ | direction | n_oc | spearman_long | "
        "spearman_short | n_distinct_long | n_distinct_short | verdict |"
    )
    L.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | "
        "---: | :---: |"
    )
    for c in summary.get("candidates", []):
        coin = c.get("coin"); tf = c.get("timeframe")
        if "error" in c:
            L.append(
                f"| {coin}@{tf} / C | — | — | — | — | — | — | — | — | "
                "— | — | — | — | — | — | — | — | ERROR |"
            )
            continue
        # Baselines first (informational rows).
        for blk_name, blk_label in (("platt", "Platt(B2)"),
                                    ("isotonic", "Iso(B2)")):
            blk = (c.get("baselines") or {}).get(blk_name) or {}
            hm = blk.get("holdout_metrics") or {}
            L.append(
                f"| {coin}@{tf} / C | {blk_label} | "
                f"{hm.get('n_trades')} | "
                f"{_fmt(hm.get('precision'), p=4)} | "
                f"{_fmt(hm.get('win_rate'), p=4)} | "
                f"{_fmt(hm.get('avg_return_per_trade_pct'), p=4, pct=True)} | "
                f"{_fmt(hm.get('net_pnl_pct_total'), p=4, pct=True)} | "
                f"{_fmt(hm.get('profit_factor'), p=4)} | "
                f"{_fmt(hm.get('max_drawdown_pct'), p=4, pct=True)} | "
                f"{_fmt(hm.get('cal_dev_post_calibration'), p=4)} | "
                f"{_fmt(blk.get('tau'), p=4)} | "
                "— | — | — | — | — | — | baseline |"
            )
        # B3 methods.
        for m in METHODS:
            mb = (c.get("methods_evaluated") or {}).get(m) or {}
            if mb.get("skipped"):
                L.append(
                    f"| {coin}@{tf} / C | {m} | — | — | — | — | — | — | "
                    "— | — | — | — | — | — | — | — | — | SKIPPED |"
                )
                continue
            hm = mb.get("holdout_metrics") or {}
            comp = mb.get("comparison") or {}
            sp = comp.get("spearman_raw_vs_calibrated_holdout") or {}
            di = comp.get("n_distinct_calibrated_probs_holdout") or {}
            di_block = comp.get("direction_of_miscalibration") or {}
            verdict = (mb.get("verdict") or {}).get("verdict", "—")
            L.append(
                f"| {coin}@{tf} / C | {m} | "
                f"{hm.get('n_trades')} | "
                f"{_fmt(hm.get('precision'), p=4)} | "
                f"{_fmt(hm.get('win_rate'), p=4)} | "
                f"{_fmt(hm.get('avg_return_per_trade_pct'), p=4, pct=True)} | "
                f"{_fmt(hm.get('net_pnl_pct_total'), p=4, pct=True)} | "
                f"{_fmt(hm.get('profit_factor'), p=4)} | "
                f"{_fmt(hm.get('max_drawdown_pct'), p=4, pct=True)} | "
                f"{_fmt(hm.get('cal_dev_post_calibration'), p=4)} | "
                f"{_fmt(mb.get('tau'), p=4)} | "
                f"{di_block.get('direction', '—')} | "
                f"{di_block.get('n_overconfident_bins', '—')} | "
                f"{_fmt(sp.get('long'), p=4)} | "
                f"{_fmt(sp.get('short'), p=4)} | "
                f"{di.get('long')} | {di.get('short')} | "
                f"{verdict} |"
            )
    L.append("")

    # Aggregate recommendation block per spec.
    L.append("## Aggregate recommendation")
    L.append("")
    if agg.get("verdict") == "A":
        L.append(
            "**A — calibration fixed.** Recommend creating Task C "
            "re-instantiation (paper-trading-C-go-live) with the "
            "PASSing (candidate, method) combination(s) below as the "
            "promotion target. Do NOT auto-create."
        )
        for combo in agg.get("passing_combinations") or []:
            L.append(
                f"- {combo.get('coin')}@{combo.get('timeframe')} / "
                f"method=`{combo.get('method')}` — "
                f"persisted under "
                f"`{(((combo.get('coin'),))[0])}/.../{(combo.get('method'))}`"
            )
        L.append("")
        L.append("**Paper-proof recommendation block:**")
        L.append("")
        L.append("```json")
        L.append(json.dumps({
            "scope_constraint": {
                "coins": ["bitcoin"],
                "timeframes": ["5m"],
                "label_family": "C_post_cost",
                "served_predictor_kind": "dual_binary_head",
            },
            "hard_safety_limits_paper": {
                "position_size_pct": 0.5,
                "max_open_positions": 1,
                "daily_loss_stop_pct": 1.5,
                "total_drawdown_stop_pct": 5.0,
                "review_after_hours": 72,
                "review_after_closed_trades": 50,
            },
            "calibration_method": (
                agg.get("passing_combinations", [{}])[0].get("method")
            ),
        }, indent=2))
        L.append("```")
    elif agg.get("verdict") == "B":
        L.append(
            "**B — calibration not fixed but signal financially "
            "strong.** At least one candidate has at least one "
            "PARTIAL_OPERATOR_DECISION method. The agent stops here "
            "and waits for the operator to answer the literal "
            "question(s) below."
        )
        for partial in agg.get("partial_combinations") or []:
            cd_str = _fmt(partial.get("cal_dev_holdout"), p=4)
            npp_str = _fmt(partial.get("net_pnl_pct_total"), p=4, pct=True)
            pf_str = _fmt(partial.get("profit_factor"), p=4)
            L.append("")
            L.append(
                f"> \"Proceed with fixed-size diagnostic paper sandbox "
                f"despite untrusted probabilities? yes/no — see "
                f"candidate `{partial.get('coin')}@{partial.get('timeframe')} "
                f"/ C` with method `{partial.get('method')}` — "
                f"cal_dev=`{cd_str}` (above {GATE_MAX_CAL_DEV_HOLDOUT} "
                f"ceiling), direction=under-confidence, holdout "
                f"PnL=`{npp_str}`, PF=`{pf_str}`.\""
            )
    elif agg.get("verdict") == "C":
        L.append(
            "**C — calibration not fixed and signal not trustworthy "
            "as a probability** but the directional signal is intact. "
            "Phase-2 redesign proposal written to "
            "`.local/tasks/proposed-sparse-post-cost-engine.md`."
        )
    else:  # D
        L.append(
            "**D — no edge remains.** Every (candidate, method) ended "
            "REJECT and the directional signal was destroyed by every "
            "method. Termination headline:"
        )
        L.append("")
        L.append(
            "> \"Current app did not produce a trustworthy quant "
            "trading loop under tested designs.\""
        )
    L.append("")

    # Per-candidate detail blocks.
    L.append("## Per-candidate detail")
    L.append("")
    for c in summary.get("candidates", []):
        coin = c.get("coin"); tf = c.get("timeframe")
        L.append(f"### {coin}@{tf} / C_post_cost")
        L.append("")
        if "error" in c:
            L.append(f"- ERROR: `{c['error']}`")
            L.append("")
            continue
        f = c.get("frame") or {}
        L.append(
            f"- frame rows: {f.get('rows_total')} "
            f"(features={f.get('feature_count')}, "
            f"horizon_bars={f.get('horizon_bars')}); "
            f"n_train={f.get('n_train_subset')}, "
            f"n_holdout={f.get('n_holdout')}"
        )
        sf = c.get("shared_fit") or {}
        L.append(
            f"- shared boosters: n_train_inner="
            f"{sf.get('n_train_inner')}, n_val={sf.get('n_val')}, "
            f"long_head_present={sf.get('long_head_present')}, "
            f"short_head_present={sf.get('short_head_present')}"
        )
        br = c.get("base_rate_per_head_inner") or {}
        L.append(
            f"- base rates (inner-train, per-head): long="
            f"{_fmt(br.get('long'), p=4)}, short={_fmt(br.get('short'), p=4)}"
        )
        be = c.get("booster_equality_across_variants") or {}
        L.append(
            f"- booster equality across all persisted variants: "
            f"all_heads_equal={be.get('all_heads_equal')}"
        )
        er = c.get("ensemble_recipe") or {}
        L.append(
            f"- ensemble recipe: run={er.get('run')}, "
            f"A={er.get('A')}, B={er.get('B')}; "
            f"`{er.get('rationale')}`"
        )
        L.append("")
        L.append("**Best per-candidate verdict (PASS > PARTIAL > REJECT)**:")
        bm = c.get("best_method") or {}
        L.append(
            f"- method=`{bm.get('method')}`, verdict=`{bm.get('verdict')}`, "
            f"cal_dev_holdout={_fmt(bm.get('cal_dev_holdout'), p=4)}, "
            f"net_pnl_pct_total={_fmt(bm.get('net_pnl_pct_total'), p=4, pct=True)}, "
            f"profit_factor={_fmt(bm.get('profit_factor'), p=4)}"
        )
        L.append("")

        # Per-method block: gates, calibration bins, fit notes.
        for m in METHODS:
            mb = (c.get("methods_evaluated") or {}).get(m) or {}
            L.append(f"#### Method `{m}`")
            L.append("")
            if mb.get("skipped"):
                L.append(
                    f"- SKIPPED — rationale: "
                    f"`{mb.get('skip_rationale')}`"
                )
                L.append("")
                continue
            verdict = mb.get("verdict") or {}
            L.append(
                f"- verdict: **{verdict.get('verdict')}** "
                f"(binding criterion `{verdict.get('binding_criterion')}`)"
            )
            L.append(
                f"- persisted to `{mb.get('candidate_dir')}`, "
                f"τ = {_fmt(mb.get('tau'), p=6)}"
            )
            cb = mb.get("calibration_block") or {}
            cbL = cb.get("long") or {}
            cbS = cb.get("short") or {}
            if m == "beta":
                L.append(
                    f"- params (long): a="
                    f"{_fmt(cbL.get('a'), p=4)}, "
                    f"b={_fmt(cbL.get('b'), p=4)}, "
                    f"c={_fmt(cbL.get('c'), p=4)}, "
                    f"converged={cbL.get('converged')}, "
                    f"nll={_fmt(cbL.get('nll'), p=4)}"
                )
                L.append(
                    f"- params (short): a="
                    f"{_fmt(cbS.get('a'), p=4)}, "
                    f"b={_fmt(cbS.get('b'), p=4)}, "
                    f"c={_fmt(cbS.get('c'), p=4)}, "
                    f"converged={cbS.get('converged')}, "
                    f"nll={_fmt(cbS.get('nll'), p=4)}"
                )
            elif m == "temp":
                L.append(
                    f"- params (long): T={_fmt(cbL.get('T'), p=4)} "
                    f"(direction=`{cbL.get('direction')}`), "
                    f"converged={cbL.get('converged')}, "
                    f"nll={_fmt(cbL.get('nll'), p=4)}"
                )
                L.append(
                    f"- params (short): T={_fmt(cbS.get('T'), p=4)} "
                    f"(direction=`{cbS.get('direction')}`), "
                    f"converged={cbS.get('converged')}, "
                    f"nll={_fmt(cbS.get('nll'), p=4)}"
                )
            elif m == "shrink":
                L.append(
                    f"- params (long): alpha="
                    f"{_fmt(cbL.get('alpha'), p=4)} "
                    f"(base_rate={_fmt(cbL.get('base_rate'), p=4)}), "
                    f"val_cal_dev={_fmt(cbL.get('val_cal_dev'), p=4)}"
                )
                L.append(
                    f"- params (short): alpha="
                    f"{_fmt(cbS.get('alpha'), p=4)} "
                    f"(base_rate={_fmt(cbS.get('base_rate'), p=4)}), "
                    f"val_cal_dev={_fmt(cbS.get('val_cal_dev'), p=4)}"
                )
            elif m == "ensemble":
                L.append(
                    f"- A={cbL.get('A')}, B={cbL.get('B')}, "
                    f"w_A_long={_fmt(cbL.get('w_A'), p=4)}, "
                    f"w_A_short={_fmt(cbS.get('w_A'), p=4)}"
                )
            comp = mb.get("comparison") or {}
            sp = comp.get("spearman_raw_vs_calibrated_holdout") or {}
            di = comp.get("n_distinct_calibrated_probs_holdout") or {}
            di_block = comp.get("direction_of_miscalibration") or {}
            td = comp.get("trade_selection_diff_vs_platt_baseline") or {}
            L.append(
                f"- holdout Spearman(raw, cal): "
                f"long={_fmt(sp.get('long'), p=4)}, "
                f"short={_fmt(sp.get('short'), p=4)}; "
                f"distinct cal probs: long={di.get('long')}, "
                f"short={di.get('short')}"
            )
            L.append(
                f"- direction_of_miscalibration: "
                f"`{di_block.get('direction')}` "
                f"(over_bins={di_block.get('n_over_bins')}, "
                f"under_bins={di_block.get('n_under_bins')}, "
                f"n_overconfident_bins={di_block.get('n_overconfident_bins')}, "
                f"avg_signed_dev={_fmt(di_block.get('avg_signed_deviation'), p=4)})"
            )
            L.append(
                f"- cal_dev delta vs iso baseline: "
                f"{_fmt(comp.get('cal_dev_holdout_minus_iso_baseline'), p=4)}; "
                f"vs platt baseline: "
                f"{_fmt(comp.get('cal_dev_holdout_minus_platt_baseline'), p=4)}"
            )
            L.append(
                f"- trade-selection diff vs Platt baseline: "
                f"only_in_platt={td.get('n_trades_only_in_platt')}, "
                f"only_in_{m}={td.get(f'n_trades_only_in_{m}')}, "
                f"in_both={td.get('n_trades_in_both')}, "
                f"disagreed_on_side={td.get('n_trades_disagreed_on_side')}"
            )
            hm = mb.get("holdout_metrics") or {}
            bins = hm.get("calibration_bins") or []
            if bins:
                L.append("")
                L.append("Calibration bins (holdout, chosen-side):")
                L.append("")
                L.append(
                    "| bin | n | mean_predicted | empirical_correct | "
                    "abs_dev | overconf? |"
                )
                L.append("| --- | ---: | ---: | ---: | ---: | :---: |")
                for b_ in bins:
                    mp = float(b_["mean_predicted"])
                    emp = float(b_["empirical_correct_rate"])
                    oc = "Y" if (mp - emp) > GATE_OVERCONFIDENCE_GAP else ""
                    L.append(
                        f"| [{b_['bin_lo']:.1f}, {b_['bin_hi']:.1f}) | "
                        f"{b_['n']} | {mp:.4f} | {emp:.4f} | "
                        f"{b_['abs_dev']:.4f} | {oc} |"
                    )
            if mb.get("fit_notes"):
                L.append("")
                L.append("Fit notes:")
                for n in mb["fit_notes"]:
                    L.append(f"- `{n}`")
            # PASS-gate panel.
            passes = verdict.get("pass_check_results") or []
            if passes:
                L.append("")
                L.append("PASS gate evaluation:")
                for entry in passes:
                    tick = "PASS" if entry.get("passed") else "FAIL"
                    L.append(
                        f"- [{tick}] `{entry.get('name')}` — "
                        f"{entry.get('detail')}"
                    )
            # REJECT-gate panel.
            rejects = verdict.get("reject_check_results") or []
            if rejects:
                L.append("")
                L.append("REJECT gate evaluation:")
                for entry in rejects:
                    tag = "TRIGGERED" if entry.get("triggered") else "ok"
                    L.append(
                        f"- [{tag}] `{entry.get('name')}` — "
                        f"{entry.get('detail')}"
                    )
            L.append("")
        # Baselines for the candidate (for reference numbers).
        bls = c.get("baselines") or {}
        if bls:
            L.append("**B2 baselines (recomputed inline on this run)**")
            L.append("")
            for blm in ("platt", "isotonic"):
                blk = bls.get(blm) or {}
                hm = blk.get("holdout_metrics") or {}
                L.append(
                    f"- `{blm}` — persisted to `{blk.get('candidate_dir')}`, "
                    f"τ={_fmt(blk.get('tau'), p=4)}, "
                    f"n_trades={hm.get('n_trades')}, "
                    f"net_pnl_pct_total="
                    f"{_fmt(hm.get('net_pnl_pct_total'), p=4, pct=True)}, "
                    f"PF={_fmt(hm.get('profit_factor'), p=4)}, "
                    f"cal_dev={_fmt(hm.get('cal_dev_post_calibration'), p=4)}"
                )
            L.append("")

    L.append("## Hard rules honoured")
    L.append("")
    L.append(
        "- No champion promotion. No `quant_brain_enabled` flip.\n"
        "- No threshold relaxation, no holdout-window swap, no fee edits.\n"
        "- Same boosters (md5(`model_to_string`) equality asserted "
        "across all variants in this run).\n"
        "- No new feature search — same 50 features as B/B2.\n"
        "- No automatic follow-up tasks queued.\n"
        "- Phase 2 redesign proposal written ONLY on aggregate "
        "verdict `C` (and ONLY as a written plan; no code changes)."
    )
    L.append("")
    return "\n".join(L)


def write_b3_report(summary: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = summary.get("run_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ",
    )
    stem = f"task-B3-calibration-final-{ts}"
    md_path = REPORTS_DIR / f"{stem}.md"
    json_path = REPORTS_DIR / f"{stem}.json"

    # Strip per-bar internals from the JSON we publish (per-bar arrays
    # balloon the file unhelpfully; the calibration_bins block already
    # carries the bin-level summary used by the gate logic).
    def _scrub(d):
        if isinstance(d, dict):
            return {
                k: _scrub(v) for k, v in d.items()
                if not (
                    isinstance(k, str) and k.startswith("_")
                ) and k != "holdout_metrics_with_internals"
            }
        if isinstance(d, list):
            return [_scrub(x) for x in d]
        return d

    md_path.write_text(render_b3_markdown(summary, ts=ts))
    json_path.write_text(json.dumps(_scrub(summary), indent=2, default=str))
    return md_path, json_path


# ---------------------------------------------------------------------------
# Standalone CLI entry point. Also exposed via `--b3-calibration`
# from the labels_research package CLI.
# ---------------------------------------------------------------------------


_DEFAULT_LOOKBACK_MS = {
    "1m": 380 * 24 * 60 * 60 * 1000,
    "5m": 380 * 24 * 60 * 60 * 1000,
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(
        description=(
            "Task #658 paper-trading B3: final calibration repair "
            "(beta / temp / shrink / ensemble) for the dual-binary-"
            "head family-C models. Per-(candidate, method) PASS / "
            "PARTIAL_OPERATOR_DECISION / REJECT verdict + A/B/C/D "
            "aggregate decision; no champion promotion, no follow-up "
            "tasks."
        ),
    )
    p.add_argument("--coins", nargs="*", default=["bitcoin", "ethereum"])
    p.add_argument("--timeframes", nargs="*", default=["5m"])
    p.add_argument("--seed", type=int, default=643)
    p.add_argument(
        "--holdout-days", type=int, default=HOLDOUT_DAYS,
        help=(
            "Forward holdout window in calendar days (default 14, "
            "matching B/B2). Exposed for diagnostics only — DO NOT "
            "lower for a re-run to coax a PASS verdict."
        ),
    )
    args = p.parse_args()
    for tf in args.timeframes:
        if tf not in _DEFAULT_LOOKBACK_MS:
            raise SystemExit(
                f"unknown timeframe {tf!r} "
                f"(supported: {sorted(_DEFAULT_LOOKBACK_MS)})"
            )
    lookback_ms_per_tf = {tf: _DEFAULT_LOOKBACK_MS[tf] for tf in args.timeframes}
    summary = asyncio.run(
        run_b3(
            coins=args.coins, timeframes=args.timeframes,
            seed=args.seed,
            lookback_ms_per_tf=lookback_ms_per_tf,
            holdout_days=args.holdout_days,
        )
    )
    md_path, json_path = write_b3_report(summary)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    if summary.get("phase2_proposal_path"):
        print(f"wrote {summary['phase2_proposal_path']}")


if __name__ == "__main__":
    main()
