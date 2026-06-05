"""Auto-recalibration of the live quant-brain decision thresholds.

Wires the existing manual sweep
(`artifacts/ml-engine/scripts/calibrate_directional_gate.py`) into the
training pipeline so a retrained model that shifts the regression-head
distribution can't silently re-introduce the "0 trades" failure that
prompted task #130.

The flow is:

1. After training finishes, take the freshly-built dataset for the
   timeframe of interest (default: 5m — that's the slot the original
   sweep was tuned against and the only timeframe with enough OOS rows
   today) and run walk-forward OOS prediction once. If a cached OOS
   parquet from a recent sweep exists and the dataset hasn't changed,
   we reuse it (the sweep itself is cheap; the OOS predict is the
   expensive step).
2. Sweep candidate (mdp, mde, factor) triples through the live-mirror
   simulator, recording n_trades / pnl / sharpe per combination.
3. Choose a recommendation: the combination that maximises realised
   PnL among combinations that emit at least `min_trades`. If nothing
   clears the floor we still surface the most-active combination so the
   operator can see what the model would do.
4. Build a "proposal" dict that diffs the chosen triple against the
   current `shared/trading-frictions.json` value and bumps
   `policy_version` to ``v4-auto-{model_hash}`` where `model_hash` is
   derived deterministically from the trained model registry pointers
   (so two retrains that produce identical models keep the same
   policy_version).
5. Always write the proposal to ``models/calibration_recommendation.json``
   so a reviewer can inspect it before it lands.
6. If `ML_RECALIBRATE_THRESHOLDS_APPLY=1` is set, also rewrite
   ``shared/trading-frictions.json`` with the new triple + bumped
   policy_version. The contract loader cache is reset so subsequent
   calls in the same process pick up the change.

Behind the `ML_RECALIBRATE_THRESHOLDS` env flag (off by default — the
test suite never hits this path; production CLI runs with it on).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from ..backtest import contract as contract_module
from ..backtest.contract import Frictions, get_frictions
from ..backtest.metrics import compute_metrics
from ..backtest.run import (
    build_basket_change_lookup,
    build_tick_streams,
    latest_snapshot_for,
    regime_lookup_for,
)
from ..backtest.simulator import simulate
from ..backtest.walk_forward_oos import predict_oos_for_dataset

logger = logging.getLogger("ml-engine.threshold_calibration")

DEFAULT_TIMEFRAME = "5m"
# Combinations the sweep evaluates. Mirrors the grid in
# scripts/calibrate_directional_gate.py so the auto path produces the
# same recommendation a human running the script by hand would land on.
DEFAULT_GRID: list[tuple[float, float, float]] = [
    (mdp, mde, factor)
    for mdp in (0.05, 0.08, 0.12, 0.18, 0.25, 0.35)
    for mde in (0.005, 0.01, 0.02, 0.03, 0.05)
    for factor in (3.0, 1.5, 1.0, 0.5, 0.25, 0.0)
]
# A recommendation that emits fewer trades than this is considered too
# thin to bless — we'd rather flag "no candidate" than auto-tighten the
# brain into another silent zero. Configurable via env so an operator
# can lower it for an emergency unblock without redeploying.
DEFAULT_MIN_TRADES = int(os.environ.get("ML_RECALIBRATE_MIN_TRADES", "8"))

REPO_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
RECOMMENDATION_PATH = REPO_MODELS_DIR / "calibration_recommendation.json"
# Rolling JSONL of every recalibration run so an operator can see how the
# recommended (mdp, mde, factor) triple drifts retrain-over-retrain (task
# #142). Lives next to the directional-call-share history JSONL so the two
# training-history files share one directory + retention story.
HISTORY_DIR = REPO_MODELS_DIR / "training_history"
CALIBRATION_HISTORY_PATH = HISTORY_DIR / "calibration_recommendation.jsonl"
# Retention policy mirrors the directional-call-share history: cap N
# records per slice (here a "slice" is one timeframe — the recalibration
# only sweeps one tf per run today) and drop anything older than M days.
# Both are env-tunable so an operator can keep more history without
# redeploying.
CALIBRATION_HISTORY_MAX_PER_SLICE = int(
    os.environ.get("ML_RECALIBRATE_HISTORY_MAX_PER_SLICE", "500")
)
CALIBRATION_HISTORY_MAX_AGE_DAYS = int(
    os.environ.get("ML_RECALIBRATE_HISTORY_MAX_AGE_DAYS", "90")
)


# ── Sweep ──────────────────────────────────────────────────────────────────
def _shim_frictions(base_fr: Frictions, mdp: float, mde: float, factor: float) -> Frictions:
    raw = json.loads(json.dumps(base_fr.raw))  # deep copy
    raw["quant_brain"]["decision_thresholds"]["min_directional_prob"] = mdp
    raw["quant_brain"]["decision_thresholds"]["min_directional_edge"] = mde
    raw["quant_brain"]["decision_thresholds"]["min_expected_return_pct_factor"] = factor
    return Frictions(raw)


@dataclass
class SweepRow:
    mdp: float
    mde: float
    factor: float
    min_expected_return_pct: float
    n_trades: int
    n_skips: int
    final_pnl_usd: float
    expectancy_usd: float
    win_rate: float
    sharpe_per_trade: float

    def to_dict(self) -> dict:
        return {
            "min_directional_prob": self.mdp,
            "min_directional_edge": self.mde,
            "min_expected_return_pct_factor": self.factor,
            "min_expected_return_pct": self.min_expected_return_pct,
            "n_trades": self.n_trades,
            "n_skips": self.n_skips,
            "final_pnl_usd": self.final_pnl_usd,
            "expectancy_usd": self.expectancy_usd,
            "win_rate": self.win_rate,
            "sharpe_per_trade": self.sharpe_per_trade,
        }


def run_sweep(
    *,
    oos: pd.DataFrame,
    streams,
    regime_fn,
    base_fr: Frictions,
    timeframe: str,
    grid: Iterable[tuple[float, float, float]] = DEFAULT_GRID,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for mdp, mde, factor in grid:
        fr2 = _shim_frictions(base_fr, mdp, mde, factor)
        sim = simulate(
            timeframe=timeframe, oos_predictions=oos, tick_streams=streams,
            fr=fr2, regime_lookup=regime_fn,
        )
        m = compute_metrics(sim.trades, sim.initial_equity).to_dict()
        rows.append(SweepRow(
            mdp=mdp, mde=mde, factor=factor,
            min_expected_return_pct=fr2.min_expected_return_pct,
            n_trades=int(m["n_trades"]),
            n_skips=len(sim.skips),
            final_pnl_usd=float(m["final_pnl_usd"]),
            expectancy_usd=float(m["expectancy_usd"]),
            win_rate=float(m["win_rate"]),
            sharpe_per_trade=float(m["sharpe_per_trade"]),
        ))
    return rows


# ── Recommendation policy ──────────────────────────────────────────────────
def recommend_thresholds(
    rows: list[SweepRow], *, min_trades: int = DEFAULT_MIN_TRADES,
) -> tuple[Optional[SweepRow], str]:
    """Pick the best combination from a sweep.

    Policy:
      * Prefer combinations with at least ``min_trades`` realised trades —
        anything thinner is statistical noise and risks blessing a triple
        that happens to win on 1-2 lucky bars.
      * Among qualifying rows, pick the one with the highest realised
        PnL; tie-break on sharpe, then on n_trades.
      * If nothing clears the floor, fall back to the row with the most
        trades (so the report still surfaces what the model is actually
        capable of) and return reason ``"below_min_trades"``.
      * If the sweep produced 0 rows or every row had 0 trades, return
        ``(None, "no_signal")``.
    """
    if not rows:
        return None, "no_signal"
    qualifying = [r for r in rows if r.n_trades >= min_trades]
    if qualifying:
        qualifying.sort(
            key=lambda r: (-r.final_pnl_usd, -r.sharpe_per_trade, -r.n_trades)
        )
        return qualifying[0], "ok"
    # Nothing cleared the floor; fall back to the most-active row so the
    # proposal still surfaces a candidate to look at.
    rows_sorted = sorted(rows, key=lambda r: (-r.n_trades, -r.final_pnl_usd))
    head = rows_sorted[0]
    if head.n_trades == 0:
        return None, "no_signal"
    return head, "below_min_trades"


# ── Model hash ─────────────────────────────────────────────────────────────
def model_hash_from_report(report: dict) -> str:
    """Deterministic short hash of the trained model registry pointers in
    the training report. Two retrains over identical data produce
    identical hashes (so the policy_version stays stable when nothing
    actually changed).
    """
    parts: list[tuple[str, str, str]] = []
    for tf, tf_report in sorted((report.get("timeframes") or {}).items()):
        if not isinstance(tf_report, dict):
            continue
        for coin, slc in sorted((tf_report.get("per_coin") or {}).items()):
            if isinstance(slc, dict) and slc.get("version"):
                parts.append((coin, tf, str(slc["version"])))
        pooled = tf_report.get("pooled")
        if isinstance(pooled, dict) and pooled.get("version"):
            parts.append(("__pooled__", tf, str(pooled["version"])))
    if not parts:
        # No trained models — fall back to a marker that's still stable
        # over an empty report. Keeps the proposal writable for tests.
        payload = "no-models"
    else:
        payload = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def policy_version_from_hash(model_hash: str) -> str:
    """Naming scheme: ``v4-auto-<hash>``. The ``v4`` prefix is one bump
    past the manually-set ``v3-calibrated-2026-04-22`` so a stale report
    pinned to v3 surfaces as out-of-date.
    """
    return f"v4-auto-{model_hash}"


# ── Proposal builder + applier ─────────────────────────────────────────────
def build_proposal(
    *,
    timeframe: str,
    snapshot_name: Optional[str],
    rec: Optional[SweepRow],
    rec_status: str,
    n_oos_rows: int,
    base_fr: Frictions,
    model_hash: str,
    sweep_rows: list[SweepRow],
    min_trades: int = DEFAULT_MIN_TRADES,
) -> dict:
    """Assemble the human-reviewable proposal payload."""
    current = base_fr.quant_decision_thresholds
    new_policy_version = policy_version_from_hash(model_hash)
    if rec is None:
        proposed: Optional[dict] = None
        change = {"would_apply": False, "reason": rec_status}
    else:
        proposed = {
            "min_directional_prob": rec.mdp,
            "min_directional_edge": rec.mde,
            "min_expected_return_pct_factor": rec.factor,
            "policy_version": new_policy_version,
        }
        diff: dict[str, dict[str, Any]] = {}
        for k in ("min_directional_prob", "min_directional_edge",
                  "min_expected_return_pct_factor", "policy_version"):
            if proposed[k] != current.get(k):
                diff[k] = {"current": current.get(k), "proposed": proposed[k]}
        change = {
            "would_apply": bool(diff),
            "reason": rec_status,
            "diff": diff,
        }
    # Trim sweep_rows to the top 20 by trades-then-pnl for the report —
    # the full grid is ~180 rows and inflates the JSON for no benefit.
    top = sorted(
        sweep_rows, key=lambda r: (-r.n_trades, -r.final_pnl_usd),
    )[:20]
    return {
        "timeframe": timeframe,
        "snapshot": snapshot_name,
        "n_oos_rows": int(n_oos_rows),
        "model_hash": model_hash,
        "current": dict(current),
        "proposed": proposed,
        "change": change,
        "recommendation_status": rec_status,
        "min_trades_floor": min_trades,
        "top_combinations": [r.to_dict() for r in top],
    }


def apply_proposal(proposal: dict, *, contract_path: Optional[Path] = None) -> bool:
    """Rewrite shared/trading-frictions.json with the proposed triple +
    bumped policy_version. Returns True if a write actually happened
    (i.e. the diff was non-empty).
    """
    if not proposal.get("change", {}).get("would_apply"):
        return False
    proposed = proposal.get("proposed")
    if not proposed:
        return False
    path = contract_path or contract_module._contract_path()
    raw = json.loads(path.read_text())
    dt = raw.setdefault("quant_brain", {}).setdefault("decision_thresholds", {})
    dt["min_directional_prob"] = proposed["min_directional_prob"]
    dt["min_directional_edge"] = proposed["min_directional_edge"]
    dt["min_expected_return_pct_factor"] = proposed["min_expected_return_pct_factor"]
    dt["policy_version"] = proposed["policy_version"]
    # Atomic-ish: write to sibling tmp then rename.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(raw, indent=2) + "\n")
    os.replace(tmp, path)
    # Reset the load_contract LRU cache so subsequent get_frictions()
    # calls in the same process see the new values.
    contract_module._reset_cache()
    return True


# ── Rolling history (task #142) ────────────────────────────────────────────
def _history_record_from_proposal(
    proposal: dict, rec: Optional[SweepRow] = None,
    *, generated_at: Optional[str] = None,
) -> dict:
    """Build the JSONL row we persist for a single recalibration run.

    Captures enough state for the dashboard to render the (mdp, mde,
    factor, n_trades, pnl) timeseries plus the policy_version / status so
    an operator can see *why* a particular run didn't apply (e.g.
    `recommendation_status="below_min_trades"`).
    """
    proposed = proposal.get("proposed") or None
    current = proposal.get("current") or {}
    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "timeframe": proposal.get("timeframe"),
        "snapshot": proposal.get("snapshot"),
        "model_hash": proposal.get("model_hash"),
        "n_oos_rows": int(proposal.get("n_oos_rows") or 0),
        "status": proposal.get("status"),
        "recommendation_status": proposal.get("recommendation_status"),
        "would_apply": bool((proposal.get("change") or {}).get("would_apply", False)),
        "applied": bool(proposal.get("applied", False)),
        "current": {
            "min_directional_prob": current.get("min_directional_prob"),
            "min_directional_edge": current.get("min_directional_edge"),
            "min_expected_return_pct_factor": current.get("min_expected_return_pct_factor"),
            "policy_version": current.get("policy_version"),
        },
        "proposed": (
            {
                "min_directional_prob": proposed.get("min_directional_prob"),
                "min_directional_edge": proposed.get("min_directional_edge"),
                "min_expected_return_pct_factor": proposed.get("min_expected_return_pct_factor"),
                "policy_version": proposed.get("policy_version"),
            }
            if proposed
            else None
        ),
        "recommendation": (
            {
                "n_trades": int(rec.n_trades),
                "n_skips": int(rec.n_skips),
                "final_pnl_usd": float(rec.final_pnl_usd),
                "expectancy_usd": float(rec.expectancy_usd),
                "win_rate": float(rec.win_rate),
                "sharpe_per_trade": float(rec.sharpe_per_trade),
            }
            if rec is not None
            else None
        ),
    }


def _append_calibration_history(
    record: dict, *, path: Optional[Path] = None,
) -> None:
    """Append one recalibration record to the rolling history file.

    Best-effort: a write failure is logged and swallowed so a full disk
    can never break the training contract.
    """
    p = path or CALIBRATION_HISTORY_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(record, default=float) + "\n")
    except Exception as exc:  # noqa: BLE001 - history is non-essential
        logger.warning(
            "calibration_history_append_failed", extra={"error": str(exc)},
        )


def _trim_calibration_history(
    *,
    max_per_slice: int = CALIBRATION_HISTORY_MAX_PER_SLICE,
    max_age_days: int = CALIBRATION_HISTORY_MAX_AGE_DAYS,
    path: Optional[Path] = None,
) -> dict:
    """Apply the retention policy. Mirrors the directional-call-share trim
    in `train.py`: keeps the newest `max_per_slice` rows per timeframe,
    drops anything older than `max_age_days`, drops malformed lines, and
    rewrites the file atomically via a sibling temp file. Best-effort.
    """
    p = path or CALIBRATION_HISTORY_PATH
    summary: dict = {"kept": 0, "dropped": 0, "skipped": False}
    try:
        if not p.exists():
            summary["skipped"] = True
            return summary
        cutoff = None
        if max_age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        rows: list[tuple[datetime, str, str]] = []
        dropped_malformed = 0
        dropped_age = 0
        with p.open("r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    dropped_malformed += 1
                    continue
                ts_raw = rec.get("generated_at")
                try:
                    ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    ts = datetime.fromtimestamp(0, tz=timezone.utc)
                if cutoff is not None and ts < cutoff:
                    dropped_age += 1
                    continue
                slice_key = str(rec.get("timeframe"))
                rows.append((ts, slice_key, line))
        by_slice: dict[str, list[tuple[datetime, str]]] = {}
        for ts, key, line in rows:
            by_slice.setdefault(key, []).append((ts, line))
        kept: list[tuple[datetime, str]] = []
        dropped_capped = 0
        for items in by_slice.values():
            items.sort(key=lambda t: t[0])
            if max_per_slice > 0 and len(items) > max_per_slice:
                dropped_capped += len(items) - max_per_slice
                items = items[-max_per_slice:]
            for ts, line in items:
                kept.append((ts, line))
        kept.sort(key=lambda t: t[0])
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        with tmp_path.open("w") as f:
            for _ts, line in kept:
                f.write(line + "\n")
        os.replace(tmp_path, p)
        summary["kept"] = len(kept)
        summary["dropped"] = dropped_malformed + dropped_capped + dropped_age
        summary["dropped_malformed"] = dropped_malformed
        summary["dropped_capped"] = dropped_capped
        summary["dropped_age"] = dropped_age
        return summary
    except Exception as exc:  # noqa: BLE001 - history trim is non-essential
        logger.warning(
            "calibration_history_trim_failed", extra={"error": str(exc)},
        )
        summary["skipped"] = True
        return summary


def _record_history(proposal: dict, rec: Optional[SweepRow] = None) -> None:
    """Append + trim in one shot. Swallows every error — the rolling
    history is observability, not a correctness requirement.
    """
    try:
        record = _history_record_from_proposal(proposal, rec)
        _append_calibration_history(record)
        _trim_calibration_history()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "calibration_history_record_failed", extra={"error": str(exc)},
        )


# ── Orchestration (called from train.run_training) ─────────────────────────
def recalibrate_after_training(
    report: dict,
    *,
    timeframe: str = DEFAULT_TIMEFRAME,
    apply: bool = False,
    grid: Iterable[tuple[float, float, float]] = DEFAULT_GRID,
    min_trades: int = DEFAULT_MIN_TRADES,
    recommendation_path: Optional[Path] = None,
) -> dict:
    """Run the sweep against the freshly-trained model registry, write a
    proposal report, optionally apply it. Returns the proposal dict so the
    caller can stamp it into the training report for visibility.

    Always best-effort: any failure is caught and surfaced as a status
    field on the returned dict so a sweep crash never breaks the
    training run's contract.
    """
    out_path = recommendation_path or RECOMMENDATION_PATH
    proposal: dict = {
        "timeframe": timeframe,
        "applied": False,
        "status": "ok",
    }
    try:
        snap = latest_snapshot_for(timeframe)
        if snap is None:
            proposal["status"] = "no_dataset"
            proposal["recommendation_status"] = "no_dataset"
            _write_proposal(proposal, out_path)
            _record_history(proposal)
            return proposal
        df = pd.read_parquet(snap)
        if df.empty:
            proposal["status"] = "empty_dataset"
            proposal["recommendation_status"] = "empty_dataset"
            proposal["snapshot"] = snap.name
            _write_proposal(proposal, out_path)
            _record_history(proposal)
            return proposal
        oos = predict_oos_for_dataset(df)
        if oos.empty:
            proposal["status"] = "no_oos"
            proposal["recommendation_status"] = "no_oos"
            proposal["snapshot"] = snap.name
            _write_proposal(proposal, out_path)
            _record_history(proposal)
            return proposal

        fr = get_frictions()
        streams = build_tick_streams(df)
        basket_changes = build_basket_change_lookup(streams, fr.timeframe_ms(timeframe))
        regime_fn = regime_lookup_for(basket_changes)
        sweep_rows = run_sweep(
            oos=oos, streams=streams, regime_fn=regime_fn,
            base_fr=fr, timeframe=timeframe, grid=grid,
        )
        rec, rec_status = recommend_thresholds(sweep_rows, min_trades=min_trades)
        model_hash = model_hash_from_report(report)
        proposal = build_proposal(
            timeframe=timeframe, snapshot_name=snap.name,
            rec=rec, rec_status=rec_status, n_oos_rows=len(oos),
            base_fr=fr, model_hash=model_hash, sweep_rows=sweep_rows,
            min_trades=min_trades,
        )
        proposal["status"] = "ok"
        if apply:
            applied = apply_proposal(proposal)
            proposal["applied"] = applied
            if applied:
                logger.info(
                    "threshold_recalibration_applied",
                    extra={
                        "timeframe": timeframe,
                        "policy_version": proposal["proposed"]["policy_version"],
                        "diff": proposal["change"]["diff"],
                    },
                )
        else:
            proposal["applied"] = False
        _write_proposal(proposal, out_path)
        _record_history(proposal, rec)
        return proposal
    except Exception as exc:  # noqa: BLE001 — best-effort; never fail training
        logger.warning("threshold_recalibration_failed", extra={"error": str(exc)})
        proposal["status"] = "error"
        proposal["error"] = str(exc)
        try:
            _write_proposal(proposal, out_path)
        except Exception:  # pragma: no cover
            pass
        try:
            _record_history(proposal)
        except Exception:  # pragma: no cover
            pass
        return proposal


def _write_proposal(proposal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposal, indent=2, sort_keys=True, default=float))
