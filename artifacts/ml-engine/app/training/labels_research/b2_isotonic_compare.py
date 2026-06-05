"""Task #657 — paper-trading B2: side-by-side Platt vs isotonic
recalibration of the dual-binary-head family-C models persisted in
Task B (#655).

Design contract (from `.local/tasks/task-657.md`):

  * Same boosters (LightGBM, same hyperparameters, same seed, same
    train_inner fold) as Task B. Booster equality is verified by
    asserting the reloaded `lgb.Booster.model_to_string()` checksum
    matches between the two persisted variants per head — a stronger
    guarantee than the spec wording because the boosters are trained
    EXACTLY ONCE in memory and persisted twice.
  * Same chronological 80/20 inner split → train_inner / val.
  * Same 14-day forward holdout (carved off the FULL frame BEFORE
    training; strict leakage gate `max(train_ts) + tf_ms <
    min(holdout_ts)`).
  * Same financial gates (frictions from `shared/trading-frictions.json`,
    NOT edited; round-trip = 0.30%; post-cost safety margin = 0.10%).
  * Two calibrators fit on the SAME val raw probabilities:
      - Platt:    `LogisticRegression` on (raw, label) per head;
                  serving form `1 / (1 + exp(slope*raw + intercept))`.
      - Isotonic: `IsotonicRegression(out_of_bounds="clip", y_min=0,
                  y_max=1)` per head; serving form
                  `numpy.interp(clip(raw, x[0], x[-1]), x, y)` over
                  the fitted `(X_thresholds_, y_thresholds_)` grid.
  * Per candidate τ is recomputed from the (1 - base_rate_train_inner)
    quantile of the calibrated `max(p_long_cal, p_short_cal)` ON VAL
    — independently per method. The decision rule (τ on max-prob,
    side = argmax, abstain otherwise) is IDENTICAL across methods.
  * NO threshold relaxation. NO holdout swap. NO Platt re-fit on
    holdout. NO champion promotion. NO automatic follow-up tasks
    ("no rescue" rule).

Acceptance per candidate:
  * PASS  — isotonic clears ALL of {n_trades >= 5, net_pnl_total > 0,
            profit_factor >= 1.0, cal_dev_holdout <= 0.20} AND
            Spearman(raw, isotonic_cal) >= 0.95 per head AND >= 5
            distinct calibrated probabilities per head on val (i.e.
            isotonic did not collapse to a step function).
  * PARTIAL — isotonic improves cal_dev_holdout vs Platt by >= 5
              percentage points but at least one of the gates above
              fails. Reported as "isotonic helps calibration but
              still doesn't ship".
  * REJECT  — neither PASS nor PARTIAL. Includes the case where
              isotonic is WORSE than Platt on cal_dev_holdout.

Outputs:
  * `artifacts/ml-engine/models/<coin>/<tf>/C_post_cost/<run-id>-platt/`
  * `artifacts/ml-engine/models/<coin>/<tf>/C_post_cost/<run-id>-iso/`
  * `artifacts/ml-engine/reports/task-B2-isotonic-recalibration-<ts>.{md,json}`
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import lightgbm as lgb

from . import producers
from .data import build_research_frame
from . import persist_truth_gate as ptg
from ..registry import ModelManifest

logger = logging.getLogger("labels_research.b2_isotonic_compare")

# Acceptance criteria for the B2 verdict — fixed by the task spec.
GATE_MIN_TRADES = ptg.GATE_MIN_TRADES                # >= 5
GATE_MIN_NET_PNL_PCT = ptg.GATE_MIN_NET_PNL_PCT      # > 0
GATE_MIN_PROFIT_FACTOR = ptg.GATE_MIN_PROFIT_FACTOR  # >= 1.0
GATE_MAX_CAL_DEV_HOLDOUT = ptg.GATE_MAX_CAL_DEV_HOLDOUT  # <= 0.20

# Spec-fixed monotonicity / collapse guards on the isotonic fit.
# Per the user spec acceptance text, both checks evaluate the
# CALIBRATED probabilities the model would actually serve on the
# forward holdout — not the val fit set. (Spec line: "post-isotonic
# probability distribution has at least 5 distinct values per head
# on the holdout".)
GATE_MIN_SPEARMAN = 0.95   # Spearman(raw, iso_cal) per head on holdout
GATE_MIN_DISTINCT = 5      # distinct calibrated probabilities per head on holdout
# REJECT criterion: a PnL drop versus Platt larger than this many
# percentage points (the spec example is "Platt 75% → isotonic 65%
# is a reject; 75% → 73% is acceptable noise" → threshold is 5
# percentage points on `net_pnl_pct_total`, additive).
GATE_REJECT_PNL_DROP_PCT_POINTS = 5.0

HOLDOUT_DAYS = ptg.HOLDOUT_DAYS

# `models/<coin>/<tf>/C_post_cost/<run-id>-{platt,iso}/`
MODELS_ROOT = ptg.MODELS_ROOT
ML_ROOT = ptg.ML_ROOT
REPORTS_DIR = ptg.REPORTS_DIR
_TF_TO_MS = ptg._TF_TO_MS


# ---------------------------------------------------------------------------
# In-memory single-train, dual-calibrate
# ---------------------------------------------------------------------------


@dataclass
class _SharedFitContext:
    """All quantities shared across the Platt and isotonic branches:
    the trained boosters, the val raw probabilities, the val labels,
    the val forward returns + side labels, plus the inner / val
    indices and the inner base rate. Constructed ONCE per candidate.
    """
    long_booster: lgb.Booster | None
    short_booster: lgb.Booster | None
    feature_cols: list[str]
    inner_end_in_train: int
    n_train_inner: int
    n_val: int
    base_rate_inner: float
    p_long_val_raw: np.ndarray
    p_short_val_raw: np.ndarray
    y_long_val: np.ndarray
    y_short_val: np.ndarray
    fwd_val: np.ndarray
    side_val: np.ndarray
    train_inner_notes: list[str]


def _build_shared_fit(
    cand: ptg.CandidateFrame, *, seed: int, val_fraction: float = 0.20,
) -> _SharedFitContext:
    """Single-train both binary heads on cand.train_idx, then return
    everything downstream calibration / scoring needs. Mirrors
    `ptg._train_persist_candidate` up to (but NOT including) the Platt
    fit — so both calibrators see the SAME boosters and the SAME val
    raw probabilities."""
    train_idx = cand.train_idx
    n_train = len(train_idx)
    if n_train < 50:
        raise RuntimeError(
            f"train_too_small n={n_train} for {cand.coin}/{cand.tf}"
        )
    val_size = max(20, int(round(n_train * val_fraction)))
    val_size = min(val_size, n_train - 20)
    inner_end_in_train = n_train - val_size
    inner_idx = train_idx[:inner_end_in_train]
    val_idx = train_idx[inner_end_in_train:]

    X_inner = cand.df[cand.feature_cols].iloc[inner_idx].reset_index(drop=True)
    X_val = cand.df[cand.feature_cols].iloc[val_idx].reset_index(drop=True)
    fwd_val = cand.fwd_full[val_idx]
    side_inner = cand.side_labels[inner_idx]
    side_val = cand.side_labels[val_idx]
    y_long_inner = (side_inner == 1.0).astype(int)
    y_short_inner = (side_inner == -1.0).astype(int)
    y_long_val = (side_val == 1.0).astype(int)
    y_short_val = (side_val == -1.0).astype(int)

    notes: list[str] = []
    long_booster = ptg._fit_one_lgb_binary(
        X_inner, y_long_inner, seed=seed,
    )
    short_booster = ptg._fit_one_lgb_binary(
        X_inner, y_short_inner, seed=seed + 1,
    )
    if long_booster is None:
        notes.append(
            f"long_head_skipped pos_inner={int(y_long_inner.sum())}"
        )
    if short_booster is None:
        notes.append(
            f"short_head_skipped pos_inner={int(y_short_inner.sum())}"
        )

    p_long_val_raw = ptg._booster_predict(long_booster, X_val)
    p_short_val_raw = ptg._booster_predict(short_booster, X_val)
    base_rate_inner = (
        float((side_inner != 0.0).sum()) / max(1, len(side_inner))
    )

    return _SharedFitContext(
        long_booster=long_booster, short_booster=short_booster,
        feature_cols=list(cand.feature_cols),
        inner_end_in_train=inner_end_in_train,
        n_train_inner=int(len(inner_idx)),
        n_val=int(len(val_idx)),
        base_rate_inner=base_rate_inner,
        p_long_val_raw=p_long_val_raw,
        p_short_val_raw=p_short_val_raw,
        y_long_val=y_long_val,
        y_short_val=y_short_val,
        fwd_val=fwd_val,
        side_val=side_val,
        train_inner_notes=notes,
    )


@dataclass
class _CalibratedRun:
    """Per-method calibration outcome — what gets persisted, scored
    and reported for ONE calibrator on ONE candidate."""
    method: str                   # "platt" | "isotonic"
    candidate_dir: Path
    tau: float
    val_metrics: dict             # post-calibration val metrics block
    holdout_metrics: dict         # post-calibration holdout metrics block
    p_long_val_cal: np.ndarray    # for spearman/distinct comparison
    p_short_val_cal: np.ndarray
    calibration_block: dict       # JSON-serialisable calibrator parameters
    notes: list[str]


def _calibrate_persist_score(
    cand: ptg.CandidateFrame,
    shared: _SharedFitContext,
    *, run_id: str, method: str,
) -> _CalibratedRun:
    """Apply ONE calibration method (Platt or isotonic) on top of the
    SHARED boosters + val raw probs, persist the slice under
    `<run_id>-{method}/`, and score the holdout from disk."""
    if method not in ("platt", "isotonic"):
        raise ValueError(f"unknown calibration method: {method!r}")

    notes: list[str] = list(shared.train_inner_notes)
    if method == "platt":
        # Skip-with-identity for degenerate heads matches the Task B
        # contract — the head's contribution is zeroed by the abstain
        # rule anyway, but the manifest needs valid (slope, intercept)
        # to satisfy `ModelManifest.validate()`.
        if shared.long_booster is None:
            slope_l, intercept_l = -1.0, 0.0
        else:
            slope_l, intercept_l = ptg._fit_platt(
                shared.p_long_val_raw, shared.y_long_val,
            )
        if shared.short_booster is None:
            slope_s, intercept_s = -1.0, 0.0
        else:
            slope_s, intercept_s = ptg._fit_platt(
                shared.p_short_val_raw, shared.y_short_val,
            )
        p_long_val_cal = (
            ptg._apply_platt(
                shared.p_long_val_raw, slope_l, intercept_l,
            )
            if shared.long_booster is not None
            else np.zeros_like(shared.p_long_val_raw)
        )
        p_short_val_cal = (
            ptg._apply_platt(
                shared.p_short_val_raw, slope_s, intercept_s,
            )
            if shared.short_booster is not None
            else np.zeros_like(shared.p_short_val_raw)
        )
        calibration_block = {
            "method": "platt",
            "long":  {"slope": float(slope_l), "intercept": float(intercept_l),
                      "head_present": shared.long_booster is not None},
            "short": {"slope": float(slope_s), "intercept": float(intercept_s),
                      "head_present": shared.short_booster is not None},
            "convention": (
                "P_calibrated = 1 / (1 + exp(slope*raw + intercept)) "
                "matches LoadedDualHeadModel._platt"
            ),
        }
    else:  # isotonic
        if shared.long_booster is None:
            iso_long = {"x_thresholds": [0.0, 1.0], "y_values": [0.0, 1.0]}
        else:
            iso_long = ptg._fit_isotonic(
                shared.p_long_val_raw, shared.y_long_val,
            )
        if shared.short_booster is None:
            iso_short = {"x_thresholds": [0.0, 1.0], "y_values": [0.0, 1.0]}
        else:
            iso_short = ptg._fit_isotonic(
                shared.p_short_val_raw, shared.y_short_val,
            )
        p_long_val_cal = (
            ptg._apply_isotonic_array(
                shared.p_long_val_raw,
                iso_long["x_thresholds"], iso_long["y_values"],
            )
            if shared.long_booster is not None
            else np.zeros_like(shared.p_long_val_raw)
        )
        p_short_val_cal = (
            ptg._apply_isotonic_array(
                shared.p_short_val_raw,
                iso_short["x_thresholds"], iso_short["y_values"],
            )
            if shared.short_booster is not None
            else np.zeros_like(shared.p_short_val_raw)
        )
        calibration_block = {
            "method": "isotonic",
            "long":  {"x_thresholds": iso_long["x_thresholds"],
                      "y_values":     iso_long["y_values"],
                      "head_present": shared.long_booster is not None},
            "short": {"x_thresholds": iso_short["x_thresholds"],
                      "y_values":     iso_short["y_values"],
                      "head_present": shared.short_booster is not None},
            "convention": (
                "P_calibrated = numpy.interp(clip(raw, x[0], x[-1]), x, y) "
                "reproduces sklearn IsotonicRegression(out_of_bounds='clip',"
                " y_min=0, y_max=1).transform; matches "
                "LoadedDualHeadModel._apply_isotonic"
            ),
        }

    # τ from val post-calibrated max-prob at the (1 - base_rate_inner)
    # quantile. SAME rule as Task B; recomputed independently per
    # method because the calibrator changes the distribution of
    # max(p_long_cal, p_short_cal).
    p_max_val_cal = np.maximum(p_long_val_cal, p_short_val_cal)
    finite = np.isfinite(p_max_val_cal)
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
        p_long_val_cal, p_short_val_cal,
        fwd=shared.fwd_val, side_labels=shared.side_val,
        tau=tau if np.isfinite(tau) else 1.1,
        cost_fraction=producers.round_trip_cost_fraction(),
    )

    # Persist this method's slice under its own subdir.
    candidate_dir = _b2_candidate_dir(cand.coin, cand.tf, run_id, method)
    _persist_b2_candidate(
        cand=cand, shared=shared,
        method=method,
        calibration_block=calibration_block,
        tau=tau, val_metrics=val_metrics, notes=notes,
        run_id=run_id, candidate_dir=candidate_dir,
    )

    # Score holdout from disk — strict per-spec contract (no in-memory
    # shortcut). The on-disk manifest dictates which calibrator the
    # scorer applies.
    holdout_metrics = _score_b2_from_disk(cand, candidate_dir)
    (candidate_dir / "holdout_metrics.json").write_text(
        json.dumps(holdout_metrics, indent=2)
    )

    return _CalibratedRun(
        method=method,
        candidate_dir=candidate_dir,
        tau=tau,
        val_metrics=val_metrics,
        holdout_metrics=holdout_metrics,
        p_long_val_cal=p_long_val_cal,
        p_short_val_cal=p_short_val_cal,
        calibration_block=calibration_block,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# B2 on-disk persistence: parallel to ptg._persist_candidate but
# parameterised on calibration method.
# ---------------------------------------------------------------------------


def _b2_candidate_dir(coin: str, tf: str, run_id: str, method: str) -> Path:
    suffix = "platt" if method == "platt" else "iso"
    return MODELS_ROOT / coin / tf / "C_post_cost" / f"{run_id}-{suffix}"


def _persist_b2_candidate(
    *, cand: ptg.CandidateFrame, shared: _SharedFitContext,
    method: str, calibration_block: dict,
    tau: float, val_metrics: dict, notes: list[str],
    run_id: str, candidate_dir: Path,
) -> None:
    """Mirror of `ptg._persist_candidate` but writes a manifest whose
    `calibration_method` matches the calibrator we just fit. Both
    calibration variants are persisted from the SAME boosters in
    memory so head-equality is tautological; the on-disk
    `model_to_string()` checksums of the reloaded boosters are
    asserted equal by the driver after both writes complete."""
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

    # On-disk calibration.json carries the method tag + the parameters
    # the holdout scorer needs. The platt convention text is preserved
    # verbatim from Task B for the platt branch; the isotonic branch
    # carries its own convention text for the same purpose.
    calibration_payload = dict(calibration_block)
    calibration_payload["abstain_tau_post_calibration"] = (
        None if not np.isfinite(tau) else float(tau)
    )
    calibration_payload["base_rate_train_inner"] = float(shared.base_rate_inner)
    calibration_payload["n_train_inner"] = shared.n_train_inner
    calibration_payload["n_val"] = shared.n_val
    calibration_payload["fit_notes"] = list(notes)
    (candidate_dir / "calibration.json").write_text(
        json.dumps(calibration_payload, indent=2)
    )

    validation_metrics_payload = {
        "metrics_post_calibration_on_val": val_metrics,
        "n_val": shared.n_val,
        "notes": list(notes),
    }
    (candidate_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics_payload, indent=2)
    )

    tau_value = float(tau) if np.isfinite(tau) else None
    if method == "platt":
        platt_payload = {
            "long":  {"slope": float(calibration_block["long"]["slope"]),
                      "intercept": float(calibration_block["long"]["intercept"])},
            "short": {"slope": float(calibration_block["short"]["slope"]),
                      "intercept": float(calibration_block["short"]["intercept"])},
        }
        isotonic_payload = None
    else:
        platt_payload = None
        isotonic_payload = {
            "long": {
                "x_thresholds": list(calibration_block["long"]["x_thresholds"]),
                "y_values":     list(calibration_block["long"]["y_values"]),
            },
            "short": {
                "x_thresholds": list(calibration_block["short"]["x_thresholds"]),
                "y_values":     list(calibration_block["short"]["y_values"]),
            },
        }

    manifest = ModelManifest(
        coin_id=cand.coin,
        timeframe=cand.tf,
        version=f"{run_id}-{('platt' if method == 'platt' else 'iso')}",
        feature_names=list(shared.feature_cols),
        coin_vocab=[cand.coin],
        n_train_rows=int(shared.n_train_inner),
        n_test_rows=int(shared.n_val),
        metrics={
            f"validation/{k}": float(v)
            for k, v in val_metrics.items()
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        },
        baseline_metrics={},
        threshold_pct=float(cand.threshold_fraction * 100.0),
        horizon_candles=int(cand.horizon),
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=(
            tau_value if tau_value is not None
            # Manifest validation requires a real number; degenerate
            # heads still need a numeric τ. Use 1.1 (out of [0,1]) so
            # any reload fails closed.
            else 1.1
        ),
        platt_calibration=platt_payload,
        friction_threshold_pct=float(
            producers.round_trip_cost_fraction() * 100.0
        ),
        label_family="C_post_cost",
        calibration_method=method,
        isotonic_calibration=isotonic_payload,
        note=(
            f"Task #657 paper-trading B2 isotonic-comparison run "
            f"{run_id}; calibration_method={method}; trained from "
            "the SAME in-memory boosters as the sibling variant in "
            f"models/<coin>/<tf>/C_post_cost/{run_id}-"
            f"{'iso' if method == 'platt' else 'platt'}/; abstain τ "
            "chosen on val post-calibration at the "
            "(1-base_rate_train_inner) quantile of "
            "max(p_long_cal, p_short_cal); promoted_to_champion=False "
            "(\"no rescue\" rule, no follow-up tasks)."
        ),
    )
    manifest.validate()
    (candidate_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str)
    )


def _score_b2_from_disk(
    cand: ptg.CandidateFrame, candidate_dir: Path,
) -> dict:
    """Holdout scorer that branches on the on-disk manifest's
    `calibration_method`. Mirrors `ptg._score_from_disk` but supports
    BOTH calibrators."""
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

    if long_booster is None:
        p_long_raw = np.zeros(len(X_holdout), dtype=float)
    else:
        p_long_raw = np.asarray(
            long_booster.predict(
                X_holdout, num_iteration=long_booster.best_iteration,
            ),
            dtype=float,
        ).flatten()
    if short_booster is None:
        p_short_raw = np.zeros(len(X_holdout), dtype=float)
    else:
        p_short_raw = np.asarray(
            short_booster.predict(
                X_holdout, num_iteration=short_booster.best_iteration,
            ),
            dtype=float,
        ).flatten()

    method = manifest.get("calibration_method") or "platt"
    if method == "isotonic":
        il = calibration["long"]
        is_ = calibration["short"]
        p_long_cal = (
            ptg._apply_isotonic_array(
                p_long_raw, il["x_thresholds"], il["y_values"],
            )
            if long_booster is not None
            else np.zeros_like(p_long_raw)
        )
        p_short_cal = (
            ptg._apply_isotonic_array(
                p_short_raw, is_["x_thresholds"], is_["y_values"],
            )
            if short_booster is not None
            else np.zeros_like(p_short_raw)
        )
    else:
        pl = calibration["long"]
        ps = calibration["short"]
        p_long_cal = (
            ptg._apply_platt(
                p_long_raw, float(pl["slope"]), float(pl["intercept"]),
            )
            if long_booster is not None
            else np.zeros_like(p_long_raw)
        )
        p_short_cal = (
            ptg._apply_platt(
                p_short_raw, float(ps["slope"]), float(ps["intercept"]),
            )
            if short_booster is not None
            else np.zeros_like(p_short_raw)
        )

    tau_value = calibration.get("abstain_tau_post_calibration")
    tau = 1.1 if tau_value is None else float(tau_value)
    metrics = ptg._compute_metrics_post_calibration(
        p_long_cal, p_short_cal,
        fwd=fwd_holdout, side_labels=cand.side_labels[holdout_idx],
        tau=tau,
        cost_fraction=producers.round_trip_cost_fraction(),
    )
    # Stash the per-bar trade-side decisions so the driver can compute
    # the trade-overlap (Jaccard) diff between Platt and isotonic on
    # the SAME holdout bars.
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
# Booster-equality verification (model_to_string checksum)
# ---------------------------------------------------------------------------


def _booster_checksum(path: Path) -> Optional[str]:
    """Return md5 of `lgb.Booster(model_file=...).model_to_string()`,
    or `None` for an empty (degenerate-head) marker file. Reloading
    the boosters from disk and re-emitting `model_to_string()` is the
    spec-mandated equality check across the two persisted variants.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    booster = lgb.Booster(model_file=str(path))
    s = booster.model_to_string()
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _verify_booster_equality(
    platt_dir: Path, iso_dir: Path,
) -> dict:
    """Reload both heads from disk for both methods and assert the
    `model_to_string()` checksums match. Returns a dict reporting the
    checksums + the equality verdict per head."""
    out: dict = {}
    for head in ("long", "short"):
        platt_chk = _booster_checksum(platt_dir / f"{head}_model.txt")
        iso_chk = _booster_checksum(iso_dir / f"{head}_model.txt")
        equal = (platt_chk == iso_chk)
        out[head] = {
            "platt_checksum": platt_chk,
            "isotonic_checksum": iso_chk,
            "equal": equal,
        }
    out["all_heads_equal"] = all(v["equal"] for v in out.values())
    return out


# ---------------------------------------------------------------------------
# Comparison statistics
# ---------------------------------------------------------------------------


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Uses scipy when available (handles
    ties exactly) and falls back to a tie-aware manual rank
    computation. Returns NaN when either input has zero variance after
    finite filtering."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xa = x[mask]; ya = y[mask]
    if np.std(xa) == 0.0 or np.std(ya) == 0.0:
        return float("nan")
    try:
        from scipy.stats import spearmanr  # type: ignore
        rho, _ = spearmanr(xa, ya)
        return float(rho)
    except Exception:  # noqa: BLE001
        # Manual fallback: average-rank tie handling, then Pearson on ranks.
        def _ranks(a: np.ndarray) -> np.ndarray:
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            i = 0
            sorted_a = a[order]
            n = len(a)
            while i < n:
                j = i
                while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
                    j += 1
                avg_rank = 0.5 * (i + j) + 1.0  # 1-based
                ranks[order[i:j + 1]] = avg_rank
                i = j + 1
            return ranks
        rx = _ranks(xa); ry = _ranks(ya)
        rx -= rx.mean(); ry -= ry.mean()
        denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
        if denom == 0.0:
            return float("nan")
        return float(np.sum(rx * ry) / denom)


def _distinct_count(values: np.ndarray, *, decimals: int = 6) -> int:
    """Distinct calibrated probability count after rounding to
    `decimals` decimal places. The rounding guards against the case
    where two raw probabilities differ by 1e-15 but isotonic should
    map them to the same step (the spec wants the "did the
    calibrator collapse to a constant" question, which a literal
    `np.unique` over float64 would over-count)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0
    return int(np.unique(np.round(arr, decimals=decimals)).size)


def _trade_selection_diff(
    platt_sides: list, iso_sides: list,
) -> dict:
    """Spec-mandated trade-selection diff (Step 5):
      * `n_trades_only_in_platt`     — bars Platt trades, isotonic abstains
      * `n_trades_only_in_isotonic`  — bars isotonic trades, Platt abstains
      * `n_trades_in_both`           — bars both calibrators trade
      * `n_trades_disagreed_on_side` — both trade but on opposite sides
        (sanity check; should be near-zero because both branch on the
        same `argmax(p_long, p_short)` rule on top of monotone-related
        calibrators).
    Each input is a per-bar `pred_side` array in {-1, 0, +1} where 0
    means "abstained".
    """
    p = np.asarray(platt_sides, dtype=int)
    i = np.asarray(iso_sides, dtype=int)
    if len(p) != len(i):
        return {
            "n_trades_only_in_platt": int((p != 0).sum()),
            "n_trades_only_in_isotonic": int((i != 0).sum()),
            "n_trades_in_both": 0,
            "n_trades_disagreed_on_side": 0,
            "n_bars_compared": 0,
            "warning": (
                f"length mismatch platt={len(p)} iso={len(i)}; "
                "diff fields fall back to per-array totals"
            ),
        }
    fired_p = (p != 0)
    fired_i = (i != 0)
    only_p = int((fired_p & ~fired_i).sum())
    only_i = int((fired_i & ~fired_p).sum())
    both = int((fired_p & fired_i).sum())
    disagree = int((fired_p & fired_i & (p != i)).sum())
    return {
        "n_trades_only_in_platt": only_p,
        "n_trades_only_in_isotonic": only_i,
        "n_trades_in_both": both,
        "n_trades_disagreed_on_side": disagree,
        "n_bars_compared": int(len(p)),
    }


# ---------------------------------------------------------------------------
# Verdict — three-way per spec (PASS / PARTIAL / REJECT)
# ---------------------------------------------------------------------------


def _judge_b2(
    *, platt_holdout: dict, iso_holdout: dict,
    spearman_long: float, spearman_short: float,
    distinct_long: int, distinct_short: int,
    leakage_passed: bool,
) -> dict:
    """Apply the B2 verdict rule per the user spec, verbatim.

    Returns a dict with:
      * `verdict`              — "PASS" | "PARTIAL" | "REJECT"
      * `binding_criterion`    — short text identifying THE gate that
                                 produced the verdict (the first PASS
                                 gate the isotonic variant misses for
                                 PARTIAL/REJECT, or "all PASS gates
                                 satisfied" for PASS)
      * `pass_check_results`   — per-gate PASS evaluation
      * `reject_check_results` — per-gate REJECT evaluation
      * `partial_explanation`  — brief text explaining why a PARTIAL
                                 candidate didn't trip a REJECT gate
    """
    cd_h = iso_holdout.get("cal_dev_post_calibration")
    cd_h_f = (
        float(cd_h) if cd_h is not None and math.isfinite(cd_h)
        else float("nan")
    )
    cd_p = platt_holdout.get("cal_dev_post_calibration")
    cd_p_f = (
        float(cd_p) if cd_p is not None and math.isfinite(cd_p)
        else float("nan")
    )
    npp_iso = float(iso_holdout.get("net_pnl_pct_total") or 0.0)
    npp_platt = float(platt_holdout.get("net_pnl_pct_total") or 0.0)
    pf_iso_v = iso_holdout.get("profit_factor")
    pf_iso = (
        float(pf_iso_v) if pf_iso_v is not None
        and math.isfinite(pf_iso_v) else float("nan")
    )
    n_trades = int(iso_holdout.get("n_trades") or 0)

    # ------------------------------------------------------------
    # PASS gates (every one of these must hold for PASS)
    # ------------------------------------------------------------
    pass_checks: list[tuple[str, bool, str]] = [
        (
            "cal_dev_holdout<=0.20",
            (math.isfinite(cd_h_f) and cd_h_f <= GATE_MAX_CAL_DEV_HOLDOUT),
            f"isotonic.holdout.cal_dev_post_calibration={cd_h_f:.4f} "
            f"vs ceiling {GATE_MAX_CAL_DEV_HOLDOUT}",
        ),
        (
            "n_trades>=5",
            (n_trades >= GATE_MIN_TRADES),
            f"isotonic.holdout.n_trades={n_trades} vs floor "
            f"{GATE_MIN_TRADES}",
        ),
        (
            "net_pnl_pct_total>0",
            (npp_iso > GATE_MIN_NET_PNL_PCT),
            f"isotonic.holdout.net_pnl_pct_total={npp_iso:.4f}% vs "
            f"floor >{GATE_MIN_NET_PNL_PCT}",
        ),
        (
            "profit_factor>=1.0",
            (math.isfinite(pf_iso) and pf_iso >= GATE_MIN_PROFIT_FACTOR),
            f"isotonic.holdout.profit_factor={pf_iso:.4f} vs floor "
            f"{GATE_MIN_PROFIT_FACTOR}",
        ),
        (
            "spearman_long>=0.95",
            (math.isfinite(spearman_long) and spearman_long >= GATE_MIN_SPEARMAN),
            f"holdout spearman(raw, iso_cal) long={spearman_long:.4f} "
            f"vs floor {GATE_MIN_SPEARMAN}",
        ),
        (
            "spearman_short>=0.95",
            (math.isfinite(spearman_short) and spearman_short >= GATE_MIN_SPEARMAN),
            f"holdout spearman(raw, iso_cal) short={spearman_short:.4f} "
            f"vs floor {GATE_MIN_SPEARMAN}",
        ),
        (
            "distinct_holdout_long>=5",
            (distinct_long >= GATE_MIN_DISTINCT),
            f"holdout distinct iso_cal probabilities long={distinct_long} "
            f"vs floor {GATE_MIN_DISTINCT}",
        ),
        (
            "distinct_holdout_short>=5",
            (distinct_short >= GATE_MIN_DISTINCT),
            f"holdout distinct iso_cal probabilities short={distinct_short} "
            f"vs floor {GATE_MIN_DISTINCT}",
        ),
    ]
    pass_check_results = [
        {"name": n, "passed": ok, "detail": d} for (n, ok, d) in pass_checks
    ]
    all_pass = all(ok for (_, ok, _) in pass_checks)

    # ------------------------------------------------------------
    # REJECT gates per spec — each ONE of these triggers REJECT.
    # ------------------------------------------------------------
    cal_worsened = (
        math.isfinite(cd_h_f) and math.isfinite(cd_p_f)
        and cd_h_f > cd_p_f
    )
    pnl_drop_pct_pts = (
        (npp_platt - npp_iso) if (
            math.isfinite(npp_platt) and math.isfinite(npp_iso)
        )
        else float("nan")
    )
    pnl_dropped = (
        math.isfinite(pnl_drop_pct_pts)
        and pnl_drop_pct_pts > GATE_REJECT_PNL_DROP_PCT_POINTS
    )
    pf_below_one = not (
        math.isfinite(pf_iso) and pf_iso >= GATE_MIN_PROFIT_FACTOR
    )
    spearman_broken = (
        not (math.isfinite(spearman_long) and spearman_long >= GATE_MIN_SPEARMAN)
        or
        not (math.isfinite(spearman_short) and spearman_short >= GATE_MIN_SPEARMAN)
    )

    reject_checks: list[tuple[str, bool, str]] = [
        (
            "cal_dev_holdout_iso_worse_than_platt",
            cal_worsened,
            f"iso_cal_dev_holdout={cd_h_f:.4f} > "
            f"platt_cal_dev_holdout={cd_p_f:.4f} "
            "(isotonic made calibration worse)",
        ),
        (
            "net_pnl_dropped_more_than_5pp_vs_platt",
            pnl_dropped,
            f"net_pnl_pct_total drop = "
            f"{pnl_drop_pct_pts:.4f}pp "
            f"(platt={npp_platt:.4f}% → iso={npp_iso:.4f}%); "
            f"threshold = {GATE_REJECT_PNL_DROP_PCT_POINTS}pp",
        ),
        (
            "profit_factor_iso_below_1.0",
            pf_below_one,
            f"isotonic.holdout.profit_factor={pf_iso:.4f} below "
            f"{GATE_MIN_PROFIT_FACTOR}",
        ),
        (
            "ranking_integrity_broken",
            spearman_broken,
            f"holdout spearman(raw, iso_cal) "
            f"long={spearman_long:.4f}, short={spearman_short:.4f}; "
            f"floor {GATE_MIN_SPEARMAN}",
        ),
        (
            "leakage_detected",
            (not leakage_passed),
            "leakage gate "
            "max(train_ts) + tf_ms < min(holdout_ts) failed",
        ),
    ]
    reject_check_results = [
        {"name": n, "triggered": tr, "detail": d}
        for (n, tr, d) in reject_checks
    ]
    any_reject = any(tr for (_, tr, _) in reject_checks)

    # ------------------------------------------------------------
    # PARTIAL conditions per spec:
    #   calibration improves vs Platt (cd_iso < cd_platt)
    #   AND financial metrics remain positive (npp > 0 AND pf >= 1.0)
    #   AND ranking integrity holds (Spearman per head >= 0.95)
    #   AND cal_dev > 0.20 on holdout (i.e. PASS just missed on cal).
    # The spec explicitly says PARTIAL "STOPS without proposing any
    # follow-up". The PnL drop ceiling is encoded as a REJECT gate so
    # a candidate that improves calibration but tanks PnL still
    # rejects.
    # ------------------------------------------------------------
    cal_improved = (
        math.isfinite(cd_h_f) and math.isfinite(cd_p_f)
        and cd_h_f < cd_p_f
    )
    cal_above_ceiling = (
        math.isfinite(cd_h_f) and cd_h_f > GATE_MAX_CAL_DEV_HOLDOUT
    )
    financial_ok = (
        npp_iso > GATE_MIN_NET_PNL_PCT
        and math.isfinite(pf_iso) and pf_iso >= GATE_MIN_PROFIT_FACTOR
        and n_trades >= GATE_MIN_TRADES
    )
    ranking_ok = (
        math.isfinite(spearman_long) and spearman_long >= GATE_MIN_SPEARMAN
        and math.isfinite(spearman_short) and spearman_short >= GATE_MIN_SPEARMAN
    )

    if all_pass and not any_reject:
        return {
            "verdict": "PASS",
            "binding_criterion": "all PASS gates satisfied",
            "pass_check_results": pass_check_results,
            "reject_check_results": reject_check_results,
            "partial_explanation": "",
            "deltas": {
                "cal_dev_holdout_iso_minus_platt": (
                    cd_h_f - cd_p_f
                    if math.isfinite(cd_h_f) and math.isfinite(cd_p_f)
                    else None
                ),
                "net_pnl_pct_total_iso_minus_platt": (
                    npp_iso - npp_platt
                    if math.isfinite(npp_iso) and math.isfinite(npp_platt)
                    else None
                ),
            },
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
            "partial_explanation": "",
            "deltas": {
                "cal_dev_holdout_iso_minus_platt": (
                    cd_h_f - cd_p_f
                    if math.isfinite(cd_h_f) and math.isfinite(cd_p_f)
                    else None
                ),
                "net_pnl_pct_total_iso_minus_platt": (
                    npp_iso - npp_platt
                    if math.isfinite(npp_iso) and math.isfinite(npp_platt)
                    else None
                ),
            },
        }

    # No REJECT gate fired AND PASS not satisfied → check PARTIAL.
    if (
        cal_improved and cal_above_ceiling and financial_ok and ranking_ok
    ):
        gap = cd_h_f - GATE_MAX_CAL_DEV_HOLDOUT
        improvement = cd_p_f - cd_h_f
        return {
            "verdict": "PARTIAL",
            "binding_criterion": "cal_dev_holdout>0.20",
            "pass_check_results": pass_check_results,
            "reject_check_results": reject_check_results,
            "partial_explanation": (
                f"cal_dev={cd_h_f:.4f}, {gap:.4f} above ceiling "
                f"{GATE_MAX_CAL_DEV_HOLDOUT}; isotonic improved cal_dev_holdout "
                f"by {improvement:.4f} vs Platt ({cd_p_f:.4f} → "
                f"{cd_h_f:.4f}); financial gates and ranking integrity "
                "preserved; no REJECT criterion tripped"
            ),
            "deltas": {
                "cal_dev_holdout_iso_minus_platt": cd_h_f - cd_p_f,
                "net_pnl_pct_total_iso_minus_platt": npp_iso - npp_platt,
            },
        }

    # Fall-through: PASS missed on a non-cal gate AND no REJECT
    # triggered (e.g. distinct count below floor without a worsening
    # cal_dev). The spec leaves only PASS / PARTIAL / REJECT as legal
    # verdicts; map this to REJECT and surface the missing PASS gates
    # as the binding criterion since they are the gates the PASS
    # ladder failed on.
    binding = next(
        (n for (n, ok, _) in pass_checks if not ok),
        "unknown_pass_gate",
    )
    return {
        "verdict": "REJECT",
        "binding_criterion": (
            f"PASS gate '{binding}' failed and PARTIAL preconditions "
            "not met"
        ),
        "pass_check_results": pass_check_results,
        "reject_check_results": reject_check_results,
        "partial_explanation": "",
        "deltas": {
            "cal_dev_holdout_iso_minus_platt": (
                cd_h_f - cd_p_f
                if math.isfinite(cd_h_f) and math.isfinite(cd_p_f)
                else None
            ),
            "net_pnl_pct_total_iso_minus_platt": (
                npp_iso - npp_platt
                if math.isfinite(npp_iso) and math.isfinite(npp_platt)
                else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_b2(
    coins: list[str], timeframes: list[str], *,
    seed: int = 643, lookback_ms_per_tf: dict[str, int],
    holdout_days: int = HOLDOUT_DAYS,
) -> dict:
    started_utc = datetime.now(timezone.utc)
    run_id = started_utc.strftime("%Y%m%dT%H%M%SZ")
    holdout_start_dt = started_utc - timedelta(days=holdout_days)
    holdout_start_ms = int(holdout_start_dt.timestamp() * 1000)

    summary: dict = {
        "task": "task-657-paper-trading-B2-isotonic-recalibration",
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
                    "b2_build_frame coin=%s tf=%s lookback_ms=%d",
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

                # Strict leakage gate: same as Task B.
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

                # SINGLE-TRAIN both heads in memory.
                logger.info(
                    "b2_train_shared_boosters coin=%s tf=%s n_train=%d",
                    coin, tf, len(cand.train_idx),
                )
                shared = _build_shared_fit(cand, seed=seed)
                cand_summary["shared_fit"] = {
                    "n_train_inner": shared.n_train_inner,
                    "n_val": shared.n_val,
                    "base_rate_train_inner": shared.base_rate_inner,
                    "long_head_present": shared.long_booster is not None,
                    "short_head_present": shared.short_booster is not None,
                    "train_inner_notes": shared.train_inner_notes,
                }

                # Calibrate + persist + score for BOTH methods on the
                # SAME boosters / val raw probs.
                platt_run = _calibrate_persist_score(
                    cand, shared, run_id=run_id, method="platt",
                )
                iso_run = _calibrate_persist_score(
                    cand, shared, run_id=run_id, method="isotonic",
                )

                # Booster-equality check via reloaded model_to_string.
                booster_eq = _verify_booster_equality(
                    platt_run.candidate_dir, iso_run.candidate_dir,
                )
                cand_summary["booster_equality"] = booster_eq
                if not booster_eq.get("all_heads_equal", False):
                    cand_summary["error"] = (
                        "booster_equality_violation: persisted heads "
                        "diverged across Platt and isotonic variants. "
                        "This must NEVER happen given single-train-in-"
                        "memory; the run is aborted before publishing "
                        "a comparison."
                    )
                    summary["candidates"].append(cand_summary)
                    continue

                # Comparison stats per spec — all evaluated on the
                # 14-day FORWARD HOLDOUT (not val):
                #   * Spearman(raw, iso_cal) per head (ranking integrity)
                #   * distinct iso_cal probability count per head
                #     (non-degeneracy)
                #   * trade-selection diff with the four spec-mandated
                #     fields
                # The val-side counterparts are also reported (under
                # `_validation` keys) for completeness because the
                # spec's comparison table includes both calibration
                # deviations on val and holdout.
                hm_iso_int = iso_run.holdout_metrics
                hm_platt_int = platt_run.holdout_metrics
                p_long_iso_holdout = np.asarray(
                    hm_iso_int["_p_long_cal"], dtype=float,
                )
                p_short_iso_holdout = np.asarray(
                    hm_iso_int["_p_short_cal"], dtype=float,
                )
                p_long_raw_holdout = np.asarray(
                    hm_iso_int["_p_long_raw"], dtype=float,
                )
                p_short_raw_holdout = np.asarray(
                    hm_iso_int["_p_short_raw"], dtype=float,
                )
                spear_long_holdout = _spearman_rho(
                    p_long_raw_holdout, p_long_iso_holdout,
                )
                spear_short_holdout = _spearman_rho(
                    p_short_raw_holdout, p_short_iso_holdout,
                )
                distinct_long_holdout = _distinct_count(p_long_iso_holdout)
                distinct_short_holdout = _distinct_count(p_short_iso_holdout)

                # Val-side ranking / non-degeneracy reported as
                # informational context (the comparison table uses
                # holdout probabilities for the gate decision).
                spear_long_val = _spearman_rho(
                    shared.p_long_val_raw, iso_run.p_long_val_cal,
                )
                spear_short_val = _spearman_rho(
                    shared.p_short_val_raw, iso_run.p_short_val_cal,
                )
                distinct_long_val = _distinct_count(iso_run.p_long_val_cal)
                distinct_short_val = _distinct_count(iso_run.p_short_val_cal)

                trade_diff = _trade_selection_diff(
                    hm_platt_int["_pred_side_per_bar"],
                    hm_iso_int["_pred_side_per_bar"],
                )

                # Strip the per-bar arrays from the metric blobs we
                # publish in the report — they balloon the JSON and
                # were only kept around for the overlap diff.
                def _strip_internal(d: dict) -> dict:
                    return {
                        k: v for k, v in d.items()
                        if not k.startswith("_")
                    }
                platt_hm_pub = _strip_internal(platt_run.holdout_metrics)
                iso_hm_pub = _strip_internal(iso_run.holdout_metrics)

                # Three-way verdict — gates evaluated on holdout
                # probabilities and metrics.
                verdict_block = _judge_b2(
                    platt_holdout=platt_hm_pub,
                    iso_holdout=iso_hm_pub,
                    spearman_long=spear_long_holdout,
                    spearman_short=spear_short_holdout,
                    distinct_long=distinct_long_holdout,
                    distinct_short=distinct_short_holdout,
                    leakage_passed=bool(leakage_passed),
                )

                cand_summary["platt"] = {
                    "candidate_dir": str(
                        platt_run.candidate_dir.relative_to(ML_ROOT)
                    ),
                    "tau": (
                        None if not np.isfinite(platt_run.tau)
                        else float(platt_run.tau)
                    ),
                    "calibration_block": platt_run.calibration_block,
                    "val_metrics": platt_run.val_metrics,
                    "holdout_metrics": platt_hm_pub,
                    "fit_notes": platt_run.notes,
                }
                cand_summary["isotonic"] = {
                    "candidate_dir": str(
                        iso_run.candidate_dir.relative_to(ML_ROOT)
                    ),
                    "tau": (
                        None if not np.isfinite(iso_run.tau)
                        else float(iso_run.tau)
                    ),
                    "calibration_block": iso_run.calibration_block,
                    "val_metrics": iso_run.val_metrics,
                    "holdout_metrics": iso_hm_pub,
                    "fit_notes": iso_run.notes,
                }
                cand_summary["comparison"] = {
                    # Holdout-side ranking integrity / non-degeneracy
                    # — the gate-binding numbers per spec.
                    "spearman_raw_iso_holdout": {
                        "long": spear_long_holdout,
                        "short": spear_short_holdout,
                    },
                    "distinct_iso_cal_count_holdout": {
                        "long": distinct_long_holdout,
                        "short": distinct_short_holdout,
                    },
                    # Val-side counterparts (informational).
                    "spearman_raw_iso_validation": {
                        "long": spear_long_val,
                        "short": spear_short_val,
                    },
                    "distinct_iso_cal_count_validation": {
                        "long": distinct_long_val,
                        "short": distinct_short_val,
                    },
                    # Spec-mandated trade-selection diff.
                    "trade_selection_diff_holdout": trade_diff,
                    # Headline deltas surfaced for the side-by-side
                    # report table.
                    "cal_dev_holdout_delta_iso_minus_platt": (
                        float(
                            iso_hm_pub["cal_dev_post_calibration"]
                            - platt_hm_pub["cal_dev_post_calibration"]
                        )
                        if (
                            iso_hm_pub.get("cal_dev_post_calibration") is not None
                            and platt_hm_pub.get("cal_dev_post_calibration") is not None
                            and math.isfinite(iso_hm_pub["cal_dev_post_calibration"])
                            and math.isfinite(platt_hm_pub["cal_dev_post_calibration"])
                        )
                        else None
                    ),
                    "net_pnl_pct_total_delta_iso_minus_platt": (
                        float(
                            (iso_hm_pub.get("net_pnl_pct_total") or 0.0)
                            - (platt_hm_pub.get("net_pnl_pct_total") or 0.0)
                        )
                    ),
                }
                cand_summary["b2_verdict"] = dict(verdict_block)
                cand_summary["b2_verdict"]["criteria"] = {
                    "n_trades_min": GATE_MIN_TRADES,
                    "net_pnl_pct_total_min": GATE_MIN_NET_PNL_PCT,
                    "profit_factor_min": GATE_MIN_PROFIT_FACTOR,
                    "cal_dev_post_calibration_holdout_max":
                        GATE_MAX_CAL_DEV_HOLDOUT,
                    "spearman_raw_iso_min_per_head_on_holdout":
                        GATE_MIN_SPEARMAN,
                    "distinct_iso_cal_min_per_head_on_holdout":
                        GATE_MIN_DISTINCT,
                    "reject_pnl_drop_pct_points_max":
                        GATE_REJECT_PNL_DROP_PCT_POINTS,
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "b2_failed coin=%s tf=%s", coin, tf,
                )
                cand_summary["error"] = f"b2_failed: {exc}"
            summary["candidates"].append(cand_summary)

    finished_utc = datetime.now(timezone.utc)
    summary["finished_utc"] = finished_utc.strftime("%Y%m%dT%H%M%SZ")

    # Aggregate verdict counts for the headline.
    verdicts = [
        (c.get("b2_verdict") or {}).get("verdict")
        for c in summary["candidates"]
    ]
    summary["verdict_counts"] = {
        "PASS": int(sum(1 for v in verdicts if v == "PASS")),
        "PARTIAL": int(sum(1 for v in verdicts if v == "PARTIAL")),
        "REJECT": int(sum(1 for v in verdicts if v == "REJECT")),
        "ERROR": int(sum(
            1 for c in summary["candidates"]
            if "error" in c
        )),
    }
    summary["any_pass"] = summary["verdict_counts"]["PASS"] > 0
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


def _delta_str(
    a, b, *, p: int = 4, pct: bool = False,
) -> str:
    """Render `iso − platt` delta with explicit sign."""
    if a is None or b is None:
        return "—"
    try:
        af = float(a); bf = float(b)
    except Exception:
        return "—"
    if not (math.isfinite(af) and math.isfinite(bf)):
        return "—"
    d = af - bf
    suf = "%" if pct else ""
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{p}f}{suf}"


def _aggregate_recommendation(verdicts: list[str]) -> str:
    """Render the spec-mandated aggregate recommendation block.

    The exact text variants are taken verbatim from the task spec:
      * at least one PASS (with or without PARTIAL companions)
      * all PARTIAL
      * all REJECT
    Mixed PARTIAL+REJECT (no PASS) is not enumerated by the spec; we
    report it as the strictest lower-bound, the all-REJECT termination
    text, with a one-liner identifying the PARTIAL candidate(s) so the
    user can act on the partial finding without misreading the
    headline.
    """
    pass_n = sum(1 for v in verdicts if v == "PASS")
    partial_n = sum(1 for v in verdicts if v == "PARTIAL")
    reject_n = sum(1 for v in verdicts if v == "REJECT")
    if pass_n >= 1:
        return (
            "Recommend creating Task C re-instantiation "
            "(paper-trading-C-go-live) with isotonic-calibrated "
            "candidate(s) [<list>] as the promotion target. Do not "
            "auto-create."
        )
    if partial_n >= 1 and reject_n == 0:
        return (
            "Calibration improved by [X] but did not reach 0.20 "
            "ceiling. User decision required: accept partial-cal "
            "promotion with documented sizing caveat, or attempt a "
            "different calibrator (Beta calibration, temperature "
            "scaling)."
        )
    if reject_n >= 1 and partial_n == 0:
        return (
            "Current app did not produce a trustworthy quant trading "
            "loop under tested designs."
        )
    # Mixed PARTIAL + REJECT (no PASS) — the spec does not enumerate
    # this case; report the all-REJECT termination text and surface
    # the PARTIAL candidate(s) so the user retains full information.
    return (
        "Current app did not produce a trustworthy quant trading "
        "loop under tested designs. (Note: mixed verdicts — at least "
        "one candidate was PARTIAL and at least one REJECT; see the "
        "per-candidate verdict table for the breakdown.)"
    )


def render_b2_markdown(summary: dict, *, ts: str) -> str:
    L: list[str] = []
    L.append(
        f"# Task #657 — Paper trading B2: Platt vs isotonic "
        f"recalibration ({ts})"
    )
    L.append("")
    vc = summary.get("verdict_counts", {})
    L.append(
        f"**Per-candidate verdicts**: PASS={vc.get('PASS', 0)}, "
        f"PARTIAL={vc.get('PARTIAL', 0)}, "
        f"REJECT={vc.get('REJECT', 0)}, "
        f"ERROR={vc.get('ERROR', 0)}"
    )
    L.append("")
    L.append("> Per the \"no rescue\" rule the spec encodes, "
             "this report writes verdicts truthfully and does NOT "
             "promote a champion or queue any follow-up tasks.")
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
        f"- frictions source: `{summary.get('frictions_source_file')}`\n"
    )

    # --- aggregate recommendation block (spec lines 232–242) ---
    candidates = summary.get("candidates", [])
    verdicts_for_summary: list[str] = []
    for c in candidates:
        if "error" in c:
            verdicts_for_summary.append("REJECT")
            continue
        v = (c.get("b2_verdict") or {}).get("verdict")
        if v in ("PASS", "PARTIAL", "REJECT"):
            verdicts_for_summary.append(v)
        else:
            verdicts_for_summary.append("REJECT")
    L.append("## Aggregate recommendation")
    L.append("")
    L.append(_aggregate_recommendation(verdicts_for_summary))
    L.append("")

    L.append("## Acceptance criteria (per candidate)")
    L.append("")
    L.append(
        "**PASS** iff the isotonic-calibrated variant satisfies ALL of:")
    L.append(
        f"- `cal_dev_post_calibration <= {GATE_MAX_CAL_DEV_HOLDOUT}` "
        "on the 14-day forward holdout"
    )
    L.append(f"- `n_trades >= {GATE_MIN_TRADES}` on holdout")
    L.append(
        f"- `net_pnl_pct_total > {GATE_MIN_NET_PNL_PCT}` on holdout "
        "(post-fee)"
    )
    L.append(f"- `profit_factor >= {GATE_MIN_PROFIT_FACTOR}` on holdout")
    L.append(
        f"- ranking integrity: Spearman(raw, iso_cal) "
        f"`>= {GATE_MIN_SPEARMAN}` per head on holdout"
    )
    L.append(
        f"- non-degeneracy: post-isotonic distribution has "
        f"`>= {GATE_MIN_DISTINCT}` distinct values per head on holdout"
    )
    L.append("")
    L.append(
        "**PARTIAL** iff calibration improves vs Platt "
        "(`cal_dev_holdout_iso < cal_dev_holdout_platt`) AND financial "
        "metrics remain positive AND ranking integrity holds, but "
        f"`cal_dev_post_calibration > {GATE_MAX_CAL_DEV_HOLDOUT}` on "
        "holdout. STOPS without proposing any follow-up."
    )
    L.append("")
    L.append("**REJECT** iff ANY of:")
    L.append(
        "- `cal_dev_holdout_iso > cal_dev_holdout_platt` "
        "(isotonic made calibration worse)"
    )
    L.append(
        f"- `net_pnl_pct_total_iso < net_pnl_pct_total_platt` by more "
        f"than {GATE_REJECT_PNL_DROP_PCT_POINTS}pp absolute "
        "(e.g. 75% → 65% rejects; 75% → 73% is acceptable noise)"
    )
    L.append(
        f"- `profit_factor_iso < {GATE_MIN_PROFIT_FACTOR}`"
    )
    L.append(
        f"- ranking integrity broken (Spearman per head < "
        f"{GATE_MIN_SPEARMAN})"
    )
    L.append("- any leakage detected (isotonic fit included holdout rows)")
    L.append("")

    L.append("## Side-by-side holdout metrics")
    L.append("")
    L.append(
        "| candidate | method | n_trades | precision | win_rate | "
        "avg_ret/trade | net_pnl_total | profit_factor | "
        "cal_dev | τ |"
    )
    L.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |"
    )
    for c in summary.get("candidates", []):
        coin = c.get("coin"); tf = c.get("timeframe")
        if "error" in c:
            L.append(
                f"| {coin}@{tf} / C | — | — | — | — | — | — | — | "
                f"— | — |"
            )
            continue
        for method_key, label in (("platt", "Platt"), ("isotonic", "Isotonic")):
            blk = c.get(method_key) or {}
            hm = blk.get("holdout_metrics") or {}
            L.append(
                f"| {coin}@{tf} / C | {label} | "
                f"{hm.get('n_trades')} | "
                f"{_fmt(hm.get('precision'), p=4)} | "
                f"{_fmt(hm.get('win_rate'), p=4)} | "
                f"{_fmt(hm.get('avg_return_per_trade_pct'), p=4, pct=True)} | "
                f"{_fmt(hm.get('net_pnl_pct_total'), p=4, pct=True)} | "
                f"{_fmt(hm.get('profit_factor'), p=4)} | "
                f"{_fmt(hm.get('cal_dev_post_calibration'), p=4)} | "
                f"{_fmt(blk.get('tau'), p=4)} |"
            )
    L.append("")

    L.append("## Per-candidate verdict")
    L.append("")
    L.append("| candidate | verdict | binding criterion | partial detail |")
    L.append("| --- | :---: | --- | --- |")
    for c in summary.get("candidates", []):
        coin = c.get("coin"); tf = c.get("timeframe")
        if "error" in c:
            L.append(
                f"| {coin}@{tf} / C | ERROR | error: {c['error']} | — |"
            )
            continue
        v = c.get("b2_verdict") or {}
        bc = v.get("binding_criterion") or "—"
        pe = v.get("partial_explanation") or "—"
        L.append(
            f"| {coin}@{tf} / C | {v.get('verdict', '—')} | "
            f"`{bc}` | {pe} |"
        )
    L.append("")

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
            f"horizon_bars={f.get('horizon_bars')})"
        )
        iq = f.get("ingestion_quality") or {}
        L.append(
            f"- ingestion: span_days={iq.get('span_days')}, "
            f"bar_gap_rate={iq.get('bar_gap_rate')}, "
            f"core_feature_nan_share={iq.get('core_feature_nan_share')}"
        )
        L.append(
            f"- training subset: n={f.get('n_train_subset')}; "
            f"holdout: n={f.get('n_holdout')}; "
            f"post-cost label threshold = "
            f"{f.get('post_cost_label_threshold_pct'):.4f}%"
        )
        lc = c.get("leakage_check") or {}
        L.append(
            f"- leakage check ({lc.get('rule')}): "
            f"{lc.get('passed')}  "
            f"(last_train_ts={lc.get('last_train_ts_ms')}, "
            f"first_holdout_ts={lc.get('first_holdout_ts_ms')}, "
            f"tf_ms={lc.get('tf_ms')})"
        )
        sf = c.get("shared_fit") or {}
        L.append(
            f"- shared boosters: n_train_inner={sf.get('n_train_inner')}, "
            f"n_val={sf.get('n_val')}, "
            f"base_rate_train_inner={_fmt(sf.get('base_rate_train_inner'), p=6)}, "
            f"long_head_present={sf.get('long_head_present')}, "
            f"short_head_present={sf.get('short_head_present')}"
        )
        be = c.get("booster_equality") or {}
        L.append(
            f"- booster equality (model_to_string md5): "
            f"long={be.get('long', {}).get('platt_checksum')} "
            f"== {be.get('long', {}).get('isotonic_checksum')} "
            f"({be.get('long', {}).get('equal')}); "
            f"short={be.get('short', {}).get('platt_checksum')} "
            f"== {be.get('short', {}).get('isotonic_checksum')} "
            f"({be.get('short', {}).get('equal')})"
        )

        comp = c.get("comparison") or {}
        sp_h = comp.get("spearman_raw_iso_holdout") or {}
        di_h = comp.get("distinct_iso_cal_count_holdout") or {}
        sp_v = comp.get("spearman_raw_iso_validation") or {}
        di_v = comp.get("distinct_iso_cal_count_validation") or {}
        td = comp.get("trade_selection_diff_holdout") or {}

        # Extract Platt and isotonic holdout/val metric blobs once for
        # the spec-format comparison table below.
        platt_blk = c.get("platt") or {}
        iso_blk = c.get("isotonic") or {}
        platt_hm = platt_blk.get("holdout_metrics") or {}
        iso_hm = iso_blk.get("holdout_metrics") or {}
        platt_vm = platt_blk.get("val_metrics") or {}
        iso_vm = iso_blk.get("val_metrics") or {}

        L.append("")
        L.append("**Step 5 comparison table (Platt vs isotonic, holdout)**")
        L.append("")
        L.append(
            "| metric | Platt | isotonic | delta (iso − platt) | direction |"
        )
        L.append("| --- | ---: | ---: | ---: | --- |")
        # Helper to emit one row.
        def _row(metric_label: str, platt_val, iso_val, *, p: int = 4,
                 pct: bool = False, direction: str = "") -> str:
            return (
                f"| {metric_label} | "
                f"{_fmt(platt_val, p=p, pct=pct)} | "
                f"{_fmt(iso_val, p=p, pct=pct)} | "
                f"{_delta_str(iso_val, platt_val, p=p, pct=pct)} | "
                f"{direction} |"
            )
        L.append(_row(
            "cal_dev_holdout",
            platt_hm.get("cal_dev_post_calibration"),
            iso_hm.get("cal_dev_post_calibration"),
            direction="lower=better",
        ))
        L.append(_row(
            "cal_dev_validation",
            platt_vm.get("cal_dev_post_calibration"),
            iso_vm.get("cal_dev_post_calibration"),
            direction="lower=better",
        ))
        L.append(_row(
            "n_trades",
            platt_hm.get("n_trades"),
            iso_hm.get("n_trades"),
            p=0, direction="informational",
        ))
        L.append(_row(
            "net_pnl_pct_total",
            platt_hm.get("net_pnl_pct_total"),
            iso_hm.get("net_pnl_pct_total"),
            pct=True, direction="higher=better",
        ))
        L.append(_row(
            "profit_factor",
            platt_hm.get("profit_factor"),
            iso_hm.get("profit_factor"),
            direction="higher=better",
        ))
        L.append(_row(
            "win_rate",
            platt_hm.get("win_rate"),
            iso_hm.get("win_rate"),
            direction="higher=better",
        ))
        L.append(_row(
            "max_drawdown_pct",
            platt_hm.get("max_drawdown_pct"),
            iso_hm.get("max_drawdown_pct"),
            pct=True, direction="smaller-magnitude=better",
        ))
        L.append(_row(
            "avg_return_per_trade_pct",
            platt_hm.get("avg_return_per_trade_pct"),
            iso_hm.get("avg_return_per_trade_pct"),
            pct=True, direction="higher=better",
        ))
        L.append(_row(
            "abstain_rate",
            platt_hm.get("abstain_rate"),
            iso_hm.get("abstain_rate"),
            direction="informational",
        ))
        L.append(_row(
            "tau",
            platt_blk.get("tau"),
            iso_blk.get("tau"),
            direction="informational",
        ))
        # Spearman row — Platt is 1.00 by construction (sigmoid is monotone).
        L.append(
            f"| spearman_raw_vs_cal_long | "
            f"1.0000* | "
            f"{_fmt(sp_h.get('long'), p=4)} | "
            f"{_delta_str(sp_h.get('long'), 1.0, p=4)} | "
            f"ranking integrity (≥{GATE_MIN_SPEARMAN}) |"
        )
        L.append(
            f"| spearman_raw_vs_cal_short | "
            f"1.0000* | "
            f"{_fmt(sp_h.get('short'), p=4)} | "
            f"{_delta_str(sp_h.get('short'), 1.0, p=4)} | "
            f"ranking integrity (≥{GATE_MIN_SPEARMAN}) |"
        )
        # Distinct counts on holdout (Platt is also a continuous
        # sigmoid, so distinct count is effectively len(unique raw)).
        L.append(
            f"| n_distinct_cal_probs_long (holdout) | — | "
            f"{di_h.get('long')} | — | "
            f"non-degeneracy (≥{GATE_MIN_DISTINCT}) |"
        )
        L.append(
            f"| n_distinct_cal_probs_short (holdout) | — | "
            f"{di_h.get('short')} | — | "
            f"non-degeneracy (≥{GATE_MIN_DISTINCT}) |"
        )
        L.append("")
        L.append(
            "\\* Platt's Spearman vs raw is always 1.00 by "
            "construction (a sigmoid is monotone)."
        )
        L.append("")
        L.append("**Trade-selection diff (holdout)**")
        L.append("")
        L.append(
            f"- `n_trades_only_in_platt`: "
            f"{td.get('n_trades_only_in_platt')}"
        )
        L.append(
            f"- `n_trades_only_in_isotonic`: "
            f"{td.get('n_trades_only_in_isotonic')}"
        )
        L.append(
            f"- `n_trades_in_both`: {td.get('n_trades_in_both')}"
        )
        L.append(
            f"- `n_trades_disagreed_on_side`: "
            f"{td.get('n_trades_disagreed_on_side')} "
            "(should be near-zero; sanity check)"
        )
        if td.get("warning"):
            L.append(f"- diff warning: `{td['warning']}`")
        L.append("")
        L.append(
            "**Validation-side ranking / non-degeneracy "
            "(informational, not gate-binding)**"
        )
        L.append("")
        L.append(
            f"- Spearman(raw, iso_cal) on val: "
            f"long={_fmt(sp_v.get('long'), p=4)}, "
            f"short={_fmt(sp_v.get('short'), p=4)}"
        )
        L.append(
            f"- Distinct iso_cal probabilities on val: "
            f"long={di_v.get('long')}, short={di_v.get('short')}"
        )

        for method_key, label in (("platt", "Platt"), ("isotonic", "Isotonic")):
            blk = c.get(method_key) or {}
            vm = blk.get("val_metrics") or {}
            hm = blk.get("holdout_metrics") or {}
            L.append("")
            L.append(f"**{label} — persisted to `{blk.get('candidate_dir')}`, "
                     f"τ = {_fmt(blk.get('tau'), p=6)}**")
            L.append("")
            cb = blk.get("calibration_block") or {}
            if method_key == "platt":
                pl = cb.get("long") or {}
                ps = cb.get("short") or {}
                L.append(
                    f"- Platt long: slope={_fmt(pl.get('slope'), p=4)}, "
                    f"intercept={_fmt(pl.get('intercept'), p=4)} | "
                    f"Platt short: slope={_fmt(ps.get('slope'), p=4)}, "
                    f"intercept={_fmt(ps.get('intercept'), p=4)}"
                )
            else:
                il = cb.get("long") or {}
                is_ = cb.get("short") or {}
                L.append(
                    f"- Isotonic long: knot count="
                    f"{len(il.get('x_thresholds') or [])}; "
                    f"x range=[{_fmt(min(il.get('x_thresholds') or [0.0]), p=4)}, "
                    f"{_fmt(max(il.get('x_thresholds') or [1.0]), p=4)}]; "
                    f"y range=[{_fmt(min(il.get('y_values') or [0.0]), p=4)}, "
                    f"{_fmt(max(il.get('y_values') or [1.0]), p=4)}]"
                )
                L.append(
                    f"- Isotonic short: knot count="
                    f"{len(is_.get('x_thresholds') or [])}; "
                    f"x range=[{_fmt(min(is_.get('x_thresholds') or [0.0]), p=4)}, "
                    f"{_fmt(max(is_.get('x_thresholds') or [1.0]), p=4)}]; "
                    f"y range=[{_fmt(min(is_.get('y_values') or [0.0]), p=4)}, "
                    f"{_fmt(max(is_.get('y_values') or [1.0]), p=4)}]"
                )
            L.append(
                f"- Validation: n={vm.get('n_total_holdout')}, "
                f"n_trades={vm.get('n_trades')}, "
                f"abstain_rate={_fmt(vm.get('abstain_rate'), p=4)}, "
                f"precision={_fmt(vm.get('precision'), p=4)}, "
                f"win_rate={_fmt(vm.get('win_rate'), p=4)}"
            )
            L.append(
                f"- Validation: avg_ret/trade="
                f"{_fmt(vm.get('avg_return_per_trade_pct'), p=4, pct=True)}, "
                f"net_pnl_total="
                f"{_fmt(vm.get('net_pnl_pct_total'), p=4, pct=True)}, "
                f"profit_factor={_fmt(vm.get('profit_factor'), p=4)}, "
                f"cal_dev={_fmt(vm.get('cal_dev_post_calibration'), p=4)}"
            )
            L.append(
                f"- Holdout: n={hm.get('n_total_holdout')}, "
                f"n_trades={hm.get('n_trades')}, "
                f"abstain_rate={_fmt(hm.get('abstain_rate'), p=4)}, "
                f"precision={_fmt(hm.get('precision'), p=4)}, "
                f"win_rate={_fmt(hm.get('win_rate'), p=4)}"
            )
            L.append(
                f"- Holdout: avg_ret/trade="
                f"{_fmt(hm.get('avg_return_per_trade_pct'), p=4, pct=True)}, "
                f"net_pnl_per_trade="
                f"{_fmt(hm.get('net_pnl_pct_per_trade'), p=4, pct=True)}, "
                f"net_pnl_total="
                f"{_fmt(hm.get('net_pnl_pct_total'), p=4, pct=True)}, "
                f"profit_factor={_fmt(hm.get('profit_factor'), p=4)}"
            )
            L.append(
                f"- Holdout: max_dd="
                f"{_fmt(hm.get('max_drawdown_pct'), p=4, pct=True)}, "
                f"cal_dev={_fmt(hm.get('cal_dev_post_calibration'), p=4)}, "
                f"share_long={_fmt(hm.get('share_long'), p=4)}, "
                f"share_short={_fmt(hm.get('share_short'), p=4)}"
            )
            bins = hm.get("calibration_bins") or []
            if bins:
                L.append("")
                L.append(
                    f"Calibration bins ({label}, holdout trades):"
                )
                L.append("")
                L.append(
                    "| bin | n | mean_predicted | "
                    "empirical_correct_rate | abs_dev |"
                )
                L.append("| --- | ---: | ---: | ---: | ---: |")
                for b in bins:
                    L.append(
                        f"| [{b['bin_lo']:.1f}, {b['bin_hi']:.1f}) | "
                        f"{b['n']} | {b['mean_predicted']:.4f} | "
                        f"{b['empirical_correct_rate']:.4f} | "
                        f"{b['abs_dev']:.4f} |"
                    )
            if blk.get("fit_notes"):
                L.append("")
                L.append(f"{label} fit notes:")
                for n in blk["fit_notes"]:
                    L.append(f"- `{n}`")
        v = c.get("b2_verdict") or {}
        L.append("")
        L.append(
            f"**B2 verdict: {v.get('verdict', '—')}** — binding "
            f"criterion: `{v.get('binding_criterion', '—')}`"
        )
        if v.get("partial_explanation"):
            L.append("")
            L.append(f"PARTIAL detail: {v['partial_explanation']}")
        # Per-gate PASS evaluation (every entry, including the ones
        # that passed, so the user can see the full gate panel).
        passes = v.get("pass_check_results") or []
        if passes:
            L.append("")
            L.append("PASS gate evaluation:")
            for entry in passes:
                tick = "PASS" if entry.get("passed") else "FAIL"
                L.append(
                    f"- [{tick}] `{entry.get('name')}` — "
                    f"{entry.get('detail')}"
                )
        # Per-gate REJECT evaluation.
        rejects = v.get("reject_check_results") or []
        if rejects:
            L.append("")
            L.append("REJECT gate evaluation:")
            for entry in rejects:
                tag = "TRIGGERED" if entry.get("triggered") else "ok"
                L.append(
                    f"- [{tag}] `{entry.get('name')}` — "
                    f"{entry.get('detail')}"
                )
        deltas = v.get("deltas") or {}
        cd_d = deltas.get("cal_dev_holdout_iso_minus_platt")
        pn_d = deltas.get("net_pnl_pct_total_iso_minus_platt")
        L.append("")
        L.append(
            "Deltas (iso − platt): "
            f"cal_dev_holdout={_fmt(cd_d, p=4)} "
            f"(negative = isotonic improved), "
            f"net_pnl_pct_total={_fmt(pn_d, p=4)}pp "
            f"(reject if drop > "
            f"{GATE_REJECT_PNL_DROP_PCT_POINTS}pp)"
        )
        L.append("")

    L.append("## Holdout horizon decision")
    L.append("")
    L.append(
        "The forward holdout PnL uses the **training-horizon (12 bars "
        "/ 1h)** forward return for parity with Task B (#655) — the "
        "trained heads predict `P(|fwd_return_12bar| > round_trip + "
        "margin)`, so a 1-bar evaluation horizon would not match the "
        "model's prediction target. The 0.30% round-trip cost is "
        "charged once per trade, sourced from "
        "`shared/trading-frictions.json` (NOT edited)."
    )
    L.append("")
    L.append("## What this report does NOT do")
    L.append("")
    L.append(
        "- No champion promotion. The B2 task is comparison-only.\n"
        "- No threshold / margin / cost edits.\n"
        "- No holdout swap.\n"
        "- No automatic follow-up tasks (\"no rescue\" rule).\n"
        "- No re-fit of either calibrator on the holdout itself."
    )
    L.append("")
    return "\n".join(L)


def write_b2_report(summary: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = summary.get("run_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ",
    )
    stem = f"task-B2-isotonic-recalibration-{ts}"
    md_path = REPORTS_DIR / f"{stem}.md"
    json_path = REPORTS_DIR / f"{stem}.json"
    md_path.write_text(render_b2_markdown(summary, ts=ts))
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    return md_path, json_path


# ---------------------------------------------------------------------------
# Standalone CLI entry point. The labels_research package CLI also
# exposes this via `--b2-isotonic`.
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
            "Task #657 paper-trading B2: side-by-side Platt vs "
            "isotonic recalibration of the dual-binary-head family-C "
            "models. PASS/PARTIAL/REJECT verdict per candidate; no "
            "champion promotion, no follow-up tasks."
        ),
    )
    p.add_argument("--coins", nargs="*", default=["bitcoin", "ethereum"])
    p.add_argument("--timeframes", nargs="*", default=["5m"])
    p.add_argument("--seed", type=int, default=643)
    p.add_argument(
        "--holdout-days", type=int, default=HOLDOUT_DAYS,
        help=(
            "Forward holdout window in calendar days (default 14, "
            "matching Task B). Exposed for diagnostics only — DO NOT "
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
        run_b2(
            coins=args.coins, timeframes=args.timeframes,
            seed=args.seed,
            lookback_ms_per_tf=lookback_ms_per_tf,
            holdout_days=args.holdout_days,
        )
    )
    md_path, json_path = write_b2_report(summary)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    # Exit 0 — B2 is a comparison report; the verdict counts are the
    # signal the caller cares about, not the exit code.


if __name__ == "__main__":
    main()
