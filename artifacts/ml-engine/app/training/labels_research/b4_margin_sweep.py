"""Task #667 — B4 margin sweep for BTC/5m calibration repair.

Background: Task #660 (B3) found that on BTC/5m every calibration method
(beta / temp / shrink / platt) produced a forward-holdout drawdown more
negative than the diagnostic-sandbox -5% auto-disable floor (beta DD =
-5.474%, all methods PARTIAL_OPERATOR_DECISION). The under-confident
direction of the miscalibration plus the BTC/5m volatility regime
suggested the post-cost label threshold (round-trip 0.30% + safety
margin 0.10% = 0.40%) is too loose: too many marginal bars get labeled
as +1 / -1, the boosters under-confidently call them, and the abstain
gate then selects the tail of the calibrated distribution which is
exactly where the calibration deviation lives.

This study sweeps the post-cost safety margin for BTC/5m only:

    margins_fraction = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030,
                        0.0040, 0.0050]

For each margin we:

  1. Build a CandidateFrame with `label_post_cost(margin_fraction=m)` —
     the label threshold becomes (round_trip + m). The training horizon
     (12 bars), feature contract, leakage gate, train_idx / holdout_idx
     splitting and 80/20 chronological inner split are inherited
     unchanged from `persist_truth_gate._prepare_candidate`.
  2. Single-train both binary heads on train_inner with the SAME
     hyperparameters / seed as B/B2/B3.
  3. Fit beta calibration (a, b, c) per head on val raw probs by
     scipy L-BFGS-B NLL minimisation — verbatim
     `b3_calibration_compare._fit_beta`.
  4. Recompute τ on val post-calibration max-prob at the
     (1 - base_rate_inner) quantile.
  5. Score the 14-day forward holdout post-calibration.

Selection rule (smallest passing margin wins):

  * holdout `max_drawdown_pct` STRICTLY > -5.0%
    (matches the diagnostic-sandbox floor verbatim — drawdownPct
    is negative for losses, the lane trips on `trough <= floor`).
  * holdout `n_trades` >= 10 (enough samples for the 10 paper-proof
    rollout the task demands and a meaningful equity walk).
  * 10-paper-proof equity walk (first 10 fired bars on the holdout,
    sized at the DS 0.5% pin) does NOT trip the -5% drawdown floor.

The B3 (task-660) study established that BTC/5m booster probabilities
are systematically under-confident across every calibration method
(beta/temp/shrink/platt) — `direction=under, n_overconfident_bins=0`,
holdout cal_dev ~0.41-0.48 across the board. The diagnostic-sandbox
lane is the operator-vetted home for exactly this regime: the DS drift
evaluator (`evaluateDiagnosticSandboxDrift`) does NOT enforce a
holdout cal_dev ceiling — it ONLY enforces `scope_constraint
.allowed_universe == ['bitcoin:5m']` for the BTC/5m champion AND
forbids `calibration_status="under_confident_documented"` from leaking
into non-DS slots. Accordingly the persisted manifest is tagged
`calibration_status="under_confident_documented"` (the registry's
operator-vetted enum value for this exact case) so it can serve the
DS lane and ONLY the DS lane.

If a winner is found we persist into the PRODUCTION registry layout
(NOT the `C_post_cost/<run-id>/` shadow tree the truth-gate uses):

    artifacts/ml-engine/models/bitcoin/5m/<version>/
        long_model.txt
        short_model.txt
        manifest.json     -- served_predictor_kind="dual_binary_head",
                             calibration_method="beta",
                             beta_calibration={"long":{a,b,c},
                                              "short":{a,b,c}},
                             scope_constraint.allowed_universe=
                                 ["bitcoin:5m"],
                             friction_threshold_pct = (rt + m_winner)*100,
                             label_family="C_post_cost".

The version uses `registry.make_version()` so the latest pointer flips
to the new dir automatically.

A 10-paper-proof rollout is computed FROM THE WINNING MARGIN'S HOLDOUT
(first 10 fired bars, equity walk weighted by the diagnostic-sandbox
0.5% sizing pin) so the report can show explicitly that those 10 trades
would NOT trip the -5% drawdown floor live. This is a paper-trading
projection — the actual live wiring (POST btc-version + mode + evaluate)
is handled by the sibling `b4_margin_sweep_promote.py` driver after
`save_model` lands the slice on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import producers
from .data import build_research_frame
from . import persist_truth_gate as ptg
from . import b3_calibration_compare as b3
from ..registry import (
    ModelManifest,
    REGISTRY_ROOT,
    make_version,
    save_model,
)

logger = logging.getLogger("labels_research.b4_margin_sweep")

# Acceptance criteria — fixed by Task #667 + the diagnostic-sandbox
# auto-disable contract in `artifacts/api-server/src/lib/mttm.ts`.
# `evaluateDiagnosticSandboxAutoDisable` trips on running drawdown
# `<= -5.0%`. The BTC/5m DS lane intentionally does NOT enforce a
# holdout cal_dev ceiling (the lane is the operator-vetted home for
# under-confident calibrators per Task #660 / B3) — see module
# docstring for the full rationale.
DRAWDOWN_FLOOR_PCT = -5.0     # holdout max_drawdown_pct must be > -5.0%
GATE_MIN_N_TRADES = 10        # enough for the 10-paper-proof rollout
DS_FIXED_SIZING_PCT = 0.005   # mttm.ts:MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT
TASK_ID = 667
LABEL_FAMILY = "C_post_cost"
CALIBRATION_STATUS = "under_confident_documented"

# Default margins to sweep — smallest-margin-first order matters for
# tie-break (we pick the smallest passing margin so the registry slice
# stays as close to the historical 0.10% safety margin as possible while
# clearing the floor).
DEFAULT_MARGINS = (
    0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050,
)

REPORTS_DIR = ptg.REPORTS_DIR
ML_ROOT = ptg.ML_ROOT


# ---------------------------------------------------------------------------
# Margin-aware candidate prep (override the hard-coded
# POST_COST_SAFETY_MARGIN_FRACTION inside `_prepare_candidate`)
# ---------------------------------------------------------------------------


def _prepare_candidate_for_margin(
    frame, holdout_start_ms: int, *, margin_fraction: float,
) -> ptg.CandidateFrame:
    """Mirror of `ptg._prepare_candidate` but with a configurable margin.

    The only difference vs. the truth-gate prep is that
    `producers.label_post_cost(...)` is called with `margin_fraction=m`
    and the resulting `threshold_fraction` is recorded as
    `(round_trip + m)`. Every other detail (forward-return horizon,
    feature columns, leakage-safe train/holdout split) is inherited
    EXACTLY from the truth-gate so the metrics we compare across the
    sweep are produced by the same protocol the diagnostic sandbox
    will see live.
    """
    import pandas as pd  # local import to keep top-level free of pandas
    df = frame.df.copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            f"empty research frame for {frame.coin_id}/{frame.timeframe}"
        )

    horizon = producers.horizon_bars(frame.timeframe)
    fr_1bar = df["forward_return"].astype(float).to_numpy()
    close_implied = np.zeros(len(df), dtype=float)
    close_implied[0] = 1.0
    for i in range(len(df) - 1):
        close_implied[i + 1] = close_implied[i] * (1.0 + fr_1bar[i])

    fwd = producers.compute_forward_returns(close_implied, horizon)
    side = producers.label_post_cost(
        fwd, margin_fraction=margin_fraction,
    ).trade_side
    threshold_fraction = (
        producers.round_trip_cost_fraction() + float(margin_fraction)
    )
    feature_cols = ptg._select_feature_columns(df)

    ts_ms = df["timestamp_ms"].astype("int64").to_numpy()
    tf_ms = ptg._TF_TO_MS[frame.timeframe]
    label_horizon_end_ms = ts_ms + horizon * tf_ms
    is_holdout = ts_ms >= holdout_start_ms
    is_finite_fwd = np.isfinite(fwd)
    is_train_label_safe = label_horizon_end_ms <= holdout_start_ms
    train_idx = np.where(
        ~is_holdout & is_finite_fwd & is_train_label_safe
    )[0]
    holdout_idx = np.where(is_holdout & is_finite_fwd)[0]
    _ = pd  # silence unused-import lint
    return ptg.CandidateFrame(
        coin=frame.coin_id, tf=frame.timeframe,
        df=df, feature_cols=feature_cols,
        fwd_full=fwd, side_labels=side, horizon=horizon,
        threshold_fraction=threshold_fraction,
        train_idx=train_idx, holdout_idx=holdout_idx,
        holdout_start_ms=int(holdout_start_ms),
    )


# ---------------------------------------------------------------------------
# Single-margin fit + score (in-memory, no persistence yet)
# ---------------------------------------------------------------------------


@dataclass
class _MarginRunResult:
    margin_fraction: float
    threshold_fraction: float
    base_rate_inner: float
    n_train_inner: int
    n_val: int
    n_holdout: int
    pos_long_inner: int
    pos_short_inner: int
    long_beta: dict
    short_beta: dict
    tau: float
    val_metrics: dict
    holdout_metrics: dict
    paper_proofs: list[dict]            # first 10 fired bars on holdout
    paper_proof_summary: dict
    notes: list[str]
    # Carried so the winner can be persisted without re-fitting:
    cand: Optional[ptg.CandidateFrame] = None
    long_booster: object = None
    short_booster: object = None
    feature_cols: Optional[list[str]] = None
    p_long_holdout_cal: Optional[np.ndarray] = None
    p_short_holdout_cal: Optional[np.ndarray] = None


def _run_one_margin(
    frame, holdout_start_ms: int, *, margin_fraction: float, seed: int,
) -> _MarginRunResult:
    cand = _prepare_candidate_for_margin(
        frame, holdout_start_ms, margin_fraction=margin_fraction,
    )

    if len(cand.train_idx) < 200:
        raise RuntimeError(
            f"train_subset_too_small n={len(cand.train_idx)} "
            f"(need >=200) for margin={margin_fraction}"
        )
    if len(cand.holdout_idx) < 50:
        raise RuntimeError(
            f"holdout_too_small n={len(cand.holdout_idx)} "
            f"(need >=50) for margin={margin_fraction}"
        )

    # Reuse the b2 single-fit context: trains long+short heads ONCE,
    # exposes val raw probs + labels for the calibrator fit.
    from . import b2_isotonic_compare as b2
    shared = b2._build_shared_fit(cand, seed=seed)

    long_beta = b3._fit_beta(shared.p_long_val_raw, shared.y_long_val)
    short_beta = b3._fit_beta(shared.p_short_val_raw, shared.y_short_val)

    p_long_val_cal = (
        b3._apply_beta(shared.p_long_val_raw, long_beta)
        if shared.long_booster is not None
        else np.zeros_like(shared.p_long_val_raw)
    )
    p_short_val_cal = (
        b3._apply_beta(shared.p_short_val_raw, short_beta)
        if shared.short_booster is not None
        else np.zeros_like(shared.p_short_val_raw)
    )

    base_rate_inner = float(shared.base_rate_inner)
    p_max_val_cal = np.maximum(p_long_val_cal, p_short_val_cal)
    finite = np.isfinite(p_max_val_cal)
    notes: list[str] = list(shared.train_inner_notes)
    if finite.sum() == 0 or base_rate_inner <= 0.0:
        tau = 1.1
        notes.append("tau_undefined_no_finite_val_probs_using_no_fire_sentinel")
    else:
        target_q = max(0.0, min(1.0, 1.0 - base_rate_inner))
        tau = float(np.quantile(p_max_val_cal[finite], target_q))
        notes.append(
            f"tau_from_val_post_beta q={target_q:.4f} tau={tau:.6f} "
            f"base_rate_inner={base_rate_inner:.6f}"
        )

    val_metrics = ptg._compute_metrics_post_calibration(
        p_long_val_cal, p_short_val_cal,
        fwd=shared.fwd_val, side_labels=shared.side_val,
        tau=tau, cost_fraction=producers.round_trip_cost_fraction(),
    )

    # Holdout scoring — same boosters in memory, no disk round-trip
    # (we only round-trip through disk AFTER selecting a winner).
    holdout_idx = cand.holdout_idx
    X_holdout = cand.df[shared.feature_cols].iloc[holdout_idx].reset_index(drop=True)
    fwd_holdout = cand.fwd_full[holdout_idx]
    side_holdout = cand.side_labels[holdout_idx]

    p_long_raw_h = ptg._booster_predict(shared.long_booster, X_holdout)
    p_short_raw_h = ptg._booster_predict(shared.short_booster, X_holdout)
    p_long_cal_h = (
        b3._apply_beta(p_long_raw_h, long_beta)
        if shared.long_booster is not None
        else np.zeros_like(p_long_raw_h)
    )
    p_short_cal_h = (
        b3._apply_beta(p_short_raw_h, short_beta)
        if shared.short_booster is not None
        else np.zeros_like(p_short_raw_h)
    )
    holdout_metrics = ptg._compute_metrics_post_calibration(
        p_long_cal_h, p_short_cal_h,
        fwd=fwd_holdout, side_labels=side_holdout,
        tau=tau, cost_fraction=producers.round_trip_cost_fraction(),
    )

    # 10-paper-proof rollout — first 10 fired bars on the holdout,
    # equity walk weighted by the DS 0.5% sizing pin, EXACTLY the
    # math `evaluateDiagnosticSandboxAutoDisable` runs over closed
    # paper trades. The point is to confirm those 10 BTC/5m trades
    # would not trip the floor live.
    paper_proofs, paper_summary = _build_paper_proofs(
        cand=cand, p_long_cal=p_long_cal_h, p_short_cal=p_short_cal_h,
        fwd=fwd_holdout, tau=tau,
        cost_fraction=producers.round_trip_cost_fraction(),
        n_proofs=10,
    )

    return _MarginRunResult(
        margin_fraction=float(margin_fraction),
        threshold_fraction=float(cand.threshold_fraction),
        base_rate_inner=base_rate_inner,
        n_train_inner=int(shared.n_train_inner),
        n_val=int(shared.n_val),
        n_holdout=int(len(holdout_idx)),
        pos_long_inner=int(shared.y_long_val.sum()),  # informational
        pos_short_inner=int(shared.y_short_val.sum()),
        long_beta=long_beta,
        short_beta=short_beta,
        tau=float(tau),
        val_metrics=val_metrics,
        holdout_metrics=holdout_metrics,
        paper_proofs=paper_proofs,
        paper_proof_summary=paper_summary,
        notes=notes,
        cand=cand,
        long_booster=shared.long_booster,
        short_booster=shared.short_booster,
        feature_cols=list(shared.feature_cols),
        p_long_holdout_cal=p_long_cal_h,
        p_short_holdout_cal=p_short_cal_h,
    )


# ---------------------------------------------------------------------------
# 10 paper-proof rollout — DS auto-disable equity walk
# ---------------------------------------------------------------------------


def _build_paper_proofs(
    *, cand: ptg.CandidateFrame,
    p_long_cal: np.ndarray, p_short_cal: np.ndarray,
    fwd: np.ndarray, tau: float, cost_fraction: float, n_proofs: int,
) -> tuple[list[dict], dict]:
    """Replay the holdout in chronological order and capture the first
    `n_proofs` fired bars. Each proof carries enough state to verify
    the DS auto-disable rule against the raw `evaluateDiagnosticSandbox
    AutoDisable` math: per-trade pnl_pct, account_return (pnl_pct *
    sizing_pct), running cumulative_pnl_pct, running equity, peak,
    trough.
    """
    holdout_idx = cand.holdout_idx
    n_h = len(holdout_idx)
    if n_h == 0:
        return [], {
            "n_proofs_requested": n_proofs,
            "n_proofs_emitted": 0,
            "trough_pct": 0.0,
            "cum_pnl_pct": 0.0,
            "would_trip_drawdown": False,
            "drawdown_floor_pct": DRAWDOWN_FLOOR_PCT,
            "sizing_pct": DS_FIXED_SIZING_PCT,
            "note": "empty_holdout",
        }

    p_max = np.maximum(p_long_cal, p_short_cal)
    long_winner = p_long_cal >= p_short_cal
    fire = (p_max >= tau) & np.isfinite(fwd)

    proofs: list[dict] = []
    equity = 1.0
    peak = 1.0
    trough = 0.0
    cum_pnl_pct = 0.0
    sizing_pct = DS_FIXED_SIZING_PCT
    ts_ms_arr = cand.df["timestamp_ms"].iloc[holdout_idx].to_numpy()
    for i in range(n_h):
        if not fire[i]:
            continue
        side = "long" if long_winner[i] else "short"
        signed_ret = (1.0 if long_winner[i] else -1.0) * float(fwd[i])
        pnl_pct = float((signed_ret - cost_fraction) * 100.0)
        per_trade_return_account = (signed_ret - cost_fraction) * sizing_pct
        cum_pnl_pct += per_trade_return_account * 100.0
        equity *= 1.0 + per_trade_return_account
        if equity > peak:
            peak = equity
        dd_now = equity / peak - 1.0
        if dd_now < trough:
            trough = dd_now

        proofs.append({
            "proof_idx": len(proofs) + 1,
            "holdout_bar_idx": int(holdout_idx[i]),
            "timestamp_ms": int(ts_ms_arr[i]),
            "p_long_cal": float(p_long_cal[i]),
            "p_short_cal": float(p_short_cal[i]),
            "tau": float(tau),
            "side": side,
            "fwd_return_pct": float(fwd[i] * 100.0),
            "cost_fraction_pct": float(cost_fraction * 100.0),
            "pnl_pct_per_trade": pnl_pct,
            "account_return_pct": float(per_trade_return_account * 100.0),
            "cumulative_pnl_pct_running": float(cum_pnl_pct),
            "equity_running": float(equity),
            "peak_running": float(peak),
            "drawdown_pct_running": float(dd_now * 100.0),
        })
        if len(proofs) >= n_proofs:
            break

    summary = {
        "n_proofs_requested": n_proofs,
        "n_proofs_emitted": len(proofs),
        "trough_pct": float(trough * 100.0),
        "cum_pnl_pct": float(cum_pnl_pct),
        "would_trip_drawdown": bool(trough * 100.0 <= DRAWDOWN_FLOOR_PCT),
        "drawdown_floor_pct": DRAWDOWN_FLOOR_PCT,
        "sizing_pct": DS_FIXED_SIZING_PCT,
        "cost_fraction": float(cost_fraction),
        "tau": float(tau),
    }
    return proofs, summary


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def _is_passing_run(run: _MarginRunResult) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    h = run.holdout_metrics
    n_trades = int(h.get("n_trades") or 0)
    if n_trades < GATE_MIN_N_TRADES:
        reasons.append(
            f"holdout.n_trades={n_trades} below floor "
            f"{GATE_MIN_N_TRADES} (need >=10 to project the 10-proof rollout)"
        )
    dd = h.get("max_drawdown_pct")
    dd_f = float(dd) if dd is not None and math.isfinite(dd) else float("nan")
    if not (math.isfinite(dd_f) and dd_f > DRAWDOWN_FLOOR_PCT):
        reasons.append(
            f"holdout.max_drawdown_pct={dd_f:.4f}% not strictly > "
            f"{DRAWDOWN_FLOOR_PCT}% (would trip DS auto-disable)"
        )
    # Paper-proof rollout MUST not trip the floor by itself either.
    if run.paper_proof_summary.get("would_trip_drawdown"):
        reasons.append(
            f"paper_proof_rollout trough_pct="
            f"{run.paper_proof_summary['trough_pct']:.4f}% "
            f"trips the DS floor {DRAWDOWN_FLOOR_PCT}% over the first "
            f"{run.paper_proof_summary['n_proofs_emitted']} fired bars"
        )
    # NOTE: cal_dev IS NOT a gate here — see module docstring. The DS
    # lane is the operator-vetted home for under-confident BTC/5m
    # calibrators (Task #660 / B3). cal_dev is REPORTED for traceability
    # but does not block selection.
    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Persist the winning slice into the production registry layout
# ---------------------------------------------------------------------------


def persist_winner_to_registry(
    winner: _MarginRunResult, *, version: str,
) -> tuple[Path, ModelManifest]:
    """Persist the winner under `models/bitcoin/5m/<version>/` via
    `registry.save_model`. Manifest is built by hand so the
    dual_binary_head + beta + scope_constraint contract is enforced
    end-to-end (manifest.validate runs inside save_model).
    """
    if winner.long_booster is None or winner.short_booster is None:
        raise RuntimeError(
            "winning run has a degenerate head — refusing to persist "
            "(re-run the sweep with a stricter min-class guard)"
        )
    cand = winner.cand
    assert cand is not None, "winner without cand (fitting bug)"

    holdout = winner.holdout_metrics
    val = winner.val_metrics
    metrics_block: dict[str, float] = {}
    for k, v in holdout.items():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            metrics_block[f"holdout/{k}"] = float(v)
    for k, v in val.items():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            metrics_block[f"validation/{k}"] = float(v)
    metrics_block["selection/margin_fraction"] = float(winner.margin_fraction)
    metrics_block["selection/threshold_fraction"] = float(winner.threshold_fraction)
    metrics_block["selection/n_train_inner"] = float(winner.n_train_inner)
    metrics_block["selection/n_val"] = float(winner.n_val)
    metrics_block["selection/n_holdout"] = float(winner.n_holdout)

    scope_constraint = {
        "coin_id": cand.coin,
        "timeframe": cand.tf,
        "candidate": LABEL_FAMILY,
        "label_family": LABEL_FAMILY,
        "allowed_universe": [f"{cand.coin}:{cand.tf}"],
    }

    beta_payload = {
        "long": {
            "a": float(winner.long_beta["a"]),
            "b": float(winner.long_beta["b"]),
            "c": float(winner.long_beta["c"]),
        },
        "short": {
            "a": float(winner.short_beta["a"]),
            "b": float(winner.short_beta["b"]),
            "c": float(winner.short_beta["c"]),
        },
    }

    note = (
        f"Task #{TASK_ID} B4 margin sweep winner. "
        f"margin_fraction={winner.margin_fraction:.4f} "
        f"(threshold_fraction={winner.threshold_fraction:.4f}, "
        f"i.e. {winner.threshold_fraction * 100.0:.4f}%). "
        f"Calibration: beta (scipy L-BFGS-B). "
        f"Abstain τ chosen on val post-beta at "
        f"q=(1-base_rate_inner)={1 - winner.base_rate_inner:.4f}, "
        f"τ={winner.tau:.6f}. "
        f"Holdout: n_trades="
        f"{int(winner.holdout_metrics.get('n_trades') or 0)}, "
        f"max_drawdown_pct="
        f"{float(winner.holdout_metrics.get('max_drawdown_pct') or 0):.4f}% "
        f"(strictly > DS floor {DRAWDOWN_FLOOR_PCT}%), "
        f"cal_dev_post_calibration="
        f"{float(winner.holdout_metrics.get('cal_dev_post_calibration') or float('nan')):.4f} "
        "(under-confident, documented). "
        f"10-paper-proof rollout: trough_pct="
        f"{winner.paper_proof_summary['trough_pct']:.4f}%, "
        f"would_trip_drawdown="
        f"{winner.paper_proof_summary['would_trip_drawdown']}. "
        "scope_constraint pinned to bitcoin:5m so the diagnostic-sandbox "
        "drift evaluator (`evaluateDiagnosticSandboxDrift`) accepts this "
        "champion. calibration_status='under_confident_documented' "
        "constrains this slice to the BTC/5m DS lane only."
    )

    manifest = ModelManifest(
        coin_id=cand.coin,
        timeframe=cand.tf,
        version=version,
        feature_names=list(winner.feature_cols or []),
        coin_vocab=[cand.coin],
        n_train_rows=int(winner.n_train_inner),
        n_test_rows=int(winner.n_val),
        metrics=metrics_block,
        baseline_metrics={},
        threshold_pct=float(winner.threshold_fraction * 100.0),
        horizon_candles=int(cand.horizon),
        served_predictor_kind="dual_binary_head",
        long_model_path="long_model.txt",
        short_model_path="short_model.txt",
        abstain_tau=(
            float(winner.tau)
            if math.isfinite(winner.tau) else 1.1
        ),
        platt_calibration=None,
        isotonic_calibration=None,
        beta_calibration=beta_payload,
        calibration_method="beta",
        friction_threshold_pct=float(winner.threshold_fraction * 100.0),
        label_family=LABEL_FAMILY,
        calibration_status=CALIBRATION_STATUS,
        scope_constraint=scope_constraint,
        note=note,
    )
    out_dir = save_model(
        coin_id=cand.coin,
        timeframe=cand.tf,
        version=version,
        booster=winner.long_booster,
        regressor=winner.short_booster,
        calibrators=None,
        manifest=manifest,
    )
    return out_dir, manifest


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_DEFAULT_LOOKBACK_MS = 380 * 24 * 60 * 60 * 1000


async def run_b4_sweep(
    *, coin: str = "bitcoin", timeframe: str = "5m",
    lookback_ms: int = _DEFAULT_LOOKBACK_MS,
    holdout_days: int = 14,
    seed: int = 643,
    margins: tuple[float, ...] = DEFAULT_MARGINS,
) -> dict:
    started_utc = datetime.now(timezone.utc)
    run_id = started_utc.strftime("%Y%m%dT%H%M%SZ")

    holdout_start_dt = started_utc - timedelta(days=holdout_days)
    holdout_start_ms = int(holdout_start_dt.timestamp() * 1000)

    summary: dict = {
        "task": f"task-{TASK_ID}-b4-margin-sweep",
        "started_utc": run_id,
        "run_id": run_id,
        "coin": coin,
        "timeframe": timeframe,
        "lookback_ms": int(lookback_ms),
        "lookback_days": round(lookback_ms / 86_400_000, 2),
        "holdout_days": holdout_days,
        "holdout_start_iso": holdout_start_dt.isoformat().replace("+00:00", "Z"),
        "round_trip_cost_pct": producers.round_trip_cost_fraction() * 100.0,
        "default_post_cost_safety_margin_pct": (
            producers.POST_COST_SAFETY_MARGIN_FRACTION * 100.0
        ),
        "margins_swept": list(margins),
        "selection_rule": {
            "drawdown_floor_pct": DRAWDOWN_FLOOR_PCT,
            "min_n_trades_holdout": GATE_MIN_N_TRADES,
            "ten_paper_proof_must_not_trip_dd_floor": True,
            "cal_dev_is_not_a_gate": True,
            "tie_break": "smallest passing margin",
            "calibration_method": "beta",
            "calibration_status": CALIBRATION_STATUS,
        },
        "ds_fixed_sizing_pct": DS_FIXED_SIZING_PCT,
        "frictions_source_file": "shared/trading-frictions.json",
        "runs": [],
        "winner": None,
        "registry": None,
    }

    logger.info(
        "b4_sweep_build_frame coin=%s tf=%s lookback_ms=%d",
        coin, timeframe, lookback_ms,
    )
    frame = await build_research_frame(coin, timeframe, lookback_ms)
    if frame.df.empty:
        summary["error"] = "build_research_frame returned empty"
        return summary
    summary["frame"] = {
        "rows_total": int(len(frame.df)),
        "bars_source": frame.bars_source,
        "ingestion_quality": frame.ingestion_quality,
        "self_leak_columns_dropped": list(frame.self_leak_columns_dropped),
    }

    runs: list[_MarginRunResult] = []
    for m in margins:
        try:
            logger.info(
                "b4_sweep_run margin=%.4f coin=%s tf=%s seed=%d",
                m, coin, timeframe, seed,
            )
            run = _run_one_margin(
                frame, holdout_start_ms,
                margin_fraction=m, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("b4_sweep_run_failed margin=%.4f", m)
            summary["runs"].append({
                "margin_fraction": float(m),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        runs.append(run)
        passed, reasons = _is_passing_run(run)
        summary["runs"].append({
            "margin_fraction": run.margin_fraction,
            "threshold_fraction": run.threshold_fraction,
            "threshold_pct": run.threshold_fraction * 100.0,
            "n_train_inner": run.n_train_inner,
            "n_val": run.n_val,
            "n_holdout": run.n_holdout,
            "base_rate_inner": run.base_rate_inner,
            "tau_post_beta": run.tau,
            "long_beta": run.long_beta,
            "short_beta": run.short_beta,
            "val_metrics": run.val_metrics,
            "holdout_metrics": run.holdout_metrics,
            "paper_proof_summary": run.paper_proof_summary,
            "passed_selection": bool(passed),
            "fail_reasons": reasons,
            "notes": run.notes,
        })

    # Pick the smallest passing margin.
    passing = [
        (r, _is_passing_run(r)) for r in runs
    ]
    passing_runs = [r for (r, (ok, _)) in passing if ok]
    passing_runs.sort(key=lambda r: r.margin_fraction)
    winner = passing_runs[0] if passing_runs else None

    if winner is None:
        summary["winner"] = None
        summary["winner_status"] = "no_margin_passed_selection"
        return summary

    version = make_version()
    out_dir, manifest = persist_winner_to_registry(winner, version=version)

    summary["winner"] = {
        "margin_fraction": winner.margin_fraction,
        "threshold_fraction": winner.threshold_fraction,
        "threshold_pct": winner.threshold_fraction * 100.0,
        "tau_post_beta": winner.tau,
        "n_train_inner": winner.n_train_inner,
        "n_val": winner.n_val,
        "n_holdout": winner.n_holdout,
        "base_rate_inner": winner.base_rate_inner,
        "long_beta": winner.long_beta,
        "short_beta": winner.short_beta,
        "val_metrics": winner.val_metrics,
        "holdout_metrics": winner.holdout_metrics,
        "paper_proofs": winner.paper_proofs,
        "paper_proof_summary": winner.paper_proof_summary,
        "notes": winner.notes,
    }
    summary["winner_status"] = "persisted"
    summary["registry"] = {
        "model_dir": str(out_dir.relative_to(ML_ROOT)),
        "version": version,
        "coin_id": manifest.coin_id,
        "timeframe": manifest.timeframe,
        "served_predictor_kind": manifest.served_predictor_kind,
        "calibration_method": manifest.calibration_method,
        "label_family": manifest.label_family,
        "scope_constraint": manifest.scope_constraint,
        "abstain_tau": manifest.abstain_tau,
        "friction_threshold_pct": manifest.friction_threshold_pct,
    }
    return summary


def _scrub(d):
    """Strip non-JSON-serialisable / huge internals before writing
    the report — mirrors the b3 helper."""
    if isinstance(d, dict):
        return {k: _scrub(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_scrub(x) for x in d]
    if isinstance(d, np.floating):
        return float(d)
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def write_report(summary: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = summary.get("started_utc") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ",
    )
    stem = f"task-{TASK_ID}-b4-margin-sweep-{ts}"
    json_path = REPORTS_DIR / f"{stem}.json"
    md_path = REPORTS_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(_scrub(summary), indent=2, default=str))
    md_path.write_text(_render_markdown(summary))
    return md_path, json_path


def _render_markdown(s: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Task #{TASK_ID} — BTC/5m B4 margin sweep")
    lines.append("")
    lines.append(f"- run_id: `{s.get('run_id')}`")
    lines.append(f"- coin/timeframe: `{s.get('coin')}/{s.get('timeframe')}`")
    lines.append(
        f"- lookback: `{s.get('lookback_days')} days` "
        f"(holdout `{s.get('holdout_days')}` days starting "
        f"`{s.get('holdout_start_iso')}`)"
    )
    lines.append(
        f"- round_trip_cost_pct=`{s.get('round_trip_cost_pct')}` "
        f"default_safety_margin_pct=`{s.get('default_post_cost_safety_margin_pct')}`"
    )
    lines.append(
        f"- selection rule: holdout `max_drawdown_pct > "
        f"{DRAWDOWN_FLOOR_PCT}%` AND "
        f"`n_trades >= {GATE_MIN_N_TRADES}` AND 10-paper-proof rollout "
        f"does not trip the DS floor; smallest-passing-margin tie-break. "
        f"`cal_dev_post_calibration` is REPORTED but not a gate (the "
        f"DS lane is the operator-vetted home for under-confident BTC/5m "
        f"calibrators per Task #660)."
    )
    lines.append("")
    lines.append("## Per-margin holdout metrics")
    lines.append("")
    lines.append(
        "| margin | threshold | n_trades | net_pnl% | max_dd% | cal_dev | tau | passed |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|:--|")
    for r in s.get("runs", []):
        if "error" in r:
            lines.append(
                f"| {r['margin_fraction']:.4f} | n/a | n/a | n/a | n/a | n/a | n/a | "
                f"FAIL ({r['error']}) |"
            )
            continue
        h = r.get("holdout_metrics", {})
        nt = int(h.get("n_trades") or 0)
        net = float(h.get("net_pnl_pct_total") or 0.0)
        dd = float(h.get("max_drawdown_pct") or 0.0)
        cd = h.get("cal_dev_post_calibration")
        cd_str = f"{float(cd):.4f}" if cd is not None and math.isfinite(cd) else "nan"
        tau = float(r.get("tau_post_beta") or 0.0)
        passed = "✓" if r.get("passed_selection") else "✗"
        lines.append(
            f"| {r['margin_fraction']:.4f} | "
            f"{r['threshold_pct']:.4f}% | {nt} | "
            f"{net:.4f}% | {dd:.4f}% | {cd_str} | {tau:.4f} | {passed} |"
        )
    lines.append("")

    winner = s.get("winner")
    if winner is None:
        lines.append("## Winner: NONE")
        lines.append("")
        lines.append(
            f"- status: `{s.get('winner_status', 'no_winner')}`"
        )
        for r in s.get("runs", []):
            if "fail_reasons" in r:
                lines.append(
                    f"  - margin={r['margin_fraction']:.4f}: "
                    f"{'; '.join(r['fail_reasons']) or '(unspecified)'}"
                )
    else:
        lines.append("## Winner")
        lines.append("")
        lines.append(
            f"- margin_fraction=`{winner['margin_fraction']:.4f}` "
            f"threshold=`{winner['threshold_pct']:.4f}%` "
            f"τ=`{winner['tau_post_beta']:.6f}`"
        )
        h = winner["holdout_metrics"]
        lines.append(
            f"- holdout: n_trades={int(h.get('n_trades') or 0)} "
            f"net_pnl={float(h.get('net_pnl_pct_total') or 0):.4f}% "
            f"max_dd={float(h.get('max_drawdown_pct') or 0):.4f}% "
            f"cal_dev={float(h.get('cal_dev_post_calibration') or float('nan')):.4f}"
        )
        lines.append(
            f"- registry slot: `{s['registry']['model_dir']}` "
            f"(version `{s['registry']['version']}`)"
        )
        lines.append("")
        lines.append("### 10 paper-proof rollout (DS auto-disable simulation)")
        lines.append("")
        lines.append(
            "Replays the first 10 fired bars of the BTC/5m holdout, weighted "
            f"by the diagnostic-sandbox sizing pin ({DS_FIXED_SIZING_PCT*100:.2f}%), "
            "applying `evaluateDiagnosticSandboxAutoDisable` math verbatim."
        )
        lines.append("")
        lines.append(
            "| # | side | p_long | p_short | fwd% | pnl% | acct% | cum_pnl% | dd% |"
        )
        lines.append("|---:|:--|---:|---:|---:|---:|---:|---:|---:|")
        for p in winner["paper_proofs"]:
            lines.append(
                f"| {p['proof_idx']} | {p['side']} | "
                f"{p['p_long_cal']:.4f} | {p['p_short_cal']:.4f} | "
                f"{p['fwd_return_pct']:+.4f} | {p['pnl_pct_per_trade']:+.4f} | "
                f"{p['account_return_pct']:+.4f} | "
                f"{p['cumulative_pnl_pct_running']:+.4f} | "
                f"{p['drawdown_pct_running']:+.4f} |"
            )
        ps = winner["paper_proof_summary"]
        lines.append("")
        lines.append(
            f"- proof_rollout: trough_pct=`{ps['trough_pct']:.4f}%`, "
            f"cum_pnl_pct=`{ps['cum_pnl_pct']:.4f}`, "
            f"would_trip_drawdown=`{ps['would_trip_drawdown']}` "
            f"(floor `{ps['drawdown_floor_pct']}%`)."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(
        description=(
            f"Task #{TASK_ID} — BTC/5m B4 margin sweep + beta calibration "
            "+ persist winner under models/bitcoin/5m/<version>/."
        ),
    )
    p.add_argument("--coin", default="bitcoin")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--seed", type=int, default=643)
    p.add_argument("--holdout-days", type=int, default=14)
    p.add_argument(
        "--lookback-days", type=int, default=380,
        help=(
            "Lookback in days for build_research_frame. Default 380 to "
            "satisfy the >=300d task requirement with margin."
        ),
    )
    p.add_argument(
        "--margins", nargs="*", type=float, default=list(DEFAULT_MARGINS),
        help="Post-cost safety margins (fraction) to sweep.",
    )
    args = p.parse_args()
    summary = asyncio.run(
        run_b4_sweep(
            coin=args.coin, timeframe=args.timeframe,
            lookback_ms=int(args.lookback_days) * 24 * 60 * 60 * 1000,
            holdout_days=int(args.holdout_days),
            seed=int(args.seed),
            margins=tuple(args.margins),
        )
    )
    md_path, json_path = write_report(summary)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    if summary.get("registry"):
        print(
            f"persisted winner -> "
            f"models/{summary['registry']['coin_id']}/"
            f"{summary['registry']['timeframe']}/"
            f"{summary['registry']['version']}/"
        )
    else:
        print("no winning margin — registry NOT updated")


if __name__ == "__main__":
    main()
