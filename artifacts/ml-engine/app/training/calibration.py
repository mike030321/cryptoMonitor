"""Calibration helpers for the multiclass training pipeline.

Task #482 — root cause of the stable-class collapse the verification
gate has been catching: the legacy per-class isotonic + per-row
sum-to-one recipe destroys the booster's cross-class argmax ranking.
Per-class isotonic learns an independent monotone curve for each class
column from `(raw_p_k, y == k)`. The marginal calibration is correct
(column means line up with class priors), but the curves can re-rank
classes within a row — a row with raw probs `[0.30, 0.60, 0.10]`
(booster prefers STABLE) gets mapped to e.g.
`iso_DOWN(0.30)=0.40`, `iso_STABLE(0.60)=0.25`, `iso_UP(0.10)=0.30`,
and after sum-to-one STABLE no longer wins the argmax. Across the
fleet diagnostic at
`scripts/diagnostic_482/run_stage_collapse_diagnostic.py`,
33/40 (coin, tf) slices showed raw STABLE-argmax 20-90% but cal
STABLE-argmax 0-3%, driving the verification gate's
`directional_call_share` to ~1.0 and tripping the
`MAX_DIRECTIONAL_CALL_SHARE = 0.95` ceiling on every slice the
1-year campaign tried to promote.

The fix is **single-scalar temperature scaling on the booster's raw
probabilities** (Guo et al. 2017, "On Calibration of Modern Neural
Networks"): `cal[k] = p[k] ** (1 / T)` followed by per-row sum-to-one,
with a single scalar `T` fit on the calibration tail by maximizing
multinomial log-likelihood. Compared with per-class isotonic this:

  * **preserves the booster's argmax exactly** because the same
    monotone exponent is applied to every class — the row's argmax
    is invariant under any positive monotone transform on the
    logits, so calibrator-induced re-ranking (the documented failure
    mode) cannot occur;
  * adjusts only the *confidence* of the booster, which is what the
    classical Platt / temperature-scaling literature defines
    calibration to do for a multiclass softmax classifier;
  * keeps the existing inference contract intact — each class still
    exposes a `.predict(p_k_raw)` method that the downstream
    `_calibrated_3class_probs` pipeline applies per column then sums
    to one per row. `TemperatureScaledClass(inv_T)` is just the same
    inv_T for all three columns.

We deliberately do NOT use a per-class (vector) temperature here.
Vector scaling DOES allow re-ranking — the optimizer hits its bounds
when the booster over-predicts a class on average (the case we have
because `_balanced_sample_weight` trains the booster on a smoothed
class-mass distribution that still over-weights the rare class
relative to the natural ~16/42/42 label distribution; see Task #507
for the sqrt-balanced dampening that softened, but did not eliminate,
this asymmetry), and the extreme per-class exponents collapse the
very class we want to recover. Single-T scaling avoids this trap by
construction.

The wrapper class lives in this module so joblib-pickled calibrators
can be loaded by both the trainer and the inference path without a
circular import. Old calibrator files (pure `IsotonicRegression`
lists) keep working because both expose the same `.predict` contract.
"""
from __future__ import annotations

import numpy as np

NUM_CLASSES = 3

# Hard caps on `inv_T` (= 1 / T). The fitted exponent should stay within
# a reasonable band so a tiny calibration tail can't produce a
# pathological multiplier. `inv_T = 1.0` is the identity (the raw
# booster probabilities pass through unchanged before per-row
# renormalization). Bounds correspond to T in [0.5, 4.0]; the upper
# bound (very flat calibration) is intentionally generous so a poorly
# discriminating booster can be flattened toward uniform without
# hitting a clip — the lower bound (sharp calibration) is tighter
# because over-sharpening can make the booster worse, not better.
INV_T_MIN = 0.25
INV_T_MAX = 2.0


class TemperatureScaledClass:
    """Per-class wrapper for single-scalar temperature scaling.

    `predict(p)` returns `clip(p, eps, 1) ** inv_T`. The downstream
    per-row sum-to-one in `app/main.py::_calibrated_3class_probs`
    recovers `softmax(logits / T)` once all three class wrappers have
    produced their column. All three wrappers carry the SAME `inv_T`
    (single-scalar temperature) so the booster's argmax is preserved
    by construction.
    """

    __slots__ = ("inv_T_k",)

    def __init__(self, inv_T_k: float) -> None:
        self.inv_T_k = float(inv_T_k)

    def predict(self, p: np.ndarray) -> np.ndarray:
        arr = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
        return arr ** self.inv_T_k

    def __getstate__(self) -> dict:
        return {"inv_T_k": self.inv_T_k}

    def __setstate__(self, state: dict) -> None:
        self.inv_T_k = float(state.get("inv_T_k", 1.0))

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"TemperatureScaledClass(inv_T_k={self.inv_T_k:.4f})"


def fit_single_temperature(
    raw: np.ndarray, y_cal: np.ndarray, max_iter: int = 200,
) -> float:
    """Return `inv_T` (scalar) that maximizes the multinomial
    log-likelihood of the labels under softmax-with-temperature applied
    to the raw multinomial probabilities. Falls back to `1.0`
    (identity calibration) on optimization failures or degenerate
    inputs (no class diversity, NaNs, wrong shape, ...).
    """
    raw_arr = np.asarray(raw, dtype=float)
    y_arr = np.asarray(y_cal, dtype=int)
    if raw_arr.ndim != 2 or raw_arr.shape[1] != NUM_CLASSES:
        return 1.0
    if y_arr.shape[0] != raw_arr.shape[0] or y_arr.shape[0] == 0:
        return 1.0
    if not np.isfinite(raw_arr).all():
        return 1.0
    if len(np.unique(y_arr)) < 2:
        return 1.0

    eps = 1e-12
    safe_raw = np.clip(raw_arr, eps, 1.0)
    log_raw = np.log(safe_raw)  # shape (n, K)
    n = log_raw.shape[0]
    rows = np.arange(n)

    def neg_log_likelihood(inv_T_arr: np.ndarray) -> float:
        inv_T = float(inv_T_arr[0])
        scaled = log_raw * inv_T                      # (n, K)
        m = scaled.max(axis=1, keepdims=True)
        log_norm = m.squeeze(-1) + np.log(
            np.exp(scaled - m).sum(axis=1) + eps
        )
        log_p_true = scaled[rows, y_arr] - log_norm
        return float(-log_p_true.sum())

    try:
        from scipy.optimize import minimize  # type: ignore[import-untyped]

        res = minimize(
            neg_log_likelihood,
            x0=np.array([1.0], dtype=float),
            method="L-BFGS-B",
            bounds=[(INV_T_MIN, INV_T_MAX)],
            options={"maxiter": max_iter},
        )
        if not getattr(res, "success", False):
            return 1.0
        inv_T = float(res.x[0])
    except Exception:
        return 1.0

    if not np.isfinite(inv_T):
        return 1.0
    return float(np.clip(inv_T, INV_T_MIN, INV_T_MAX))


def apply_single_temperature(raw: np.ndarray, inv_T: float) -> np.ndarray:
    """Apply `cal[k] = raw[k] ** inv_T` for every class and per-row
    sum-to-one. Matches the inference-time pipeline once each
    `TemperatureScaledClass` has produced its column in
    `_calibrated_3class_probs`.
    """
    raw_arr = np.clip(np.asarray(raw, dtype=float), 1e-12, 1.0)
    scaled = raw_arr ** float(inv_T)
    s = scaled.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return scaled / s
