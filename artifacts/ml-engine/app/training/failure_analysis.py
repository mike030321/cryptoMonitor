"""Auto-generated failure-analysis report (Task #327).

Consumes the in-memory `report` dict produced by `run_training` (the same
dict that gets written to `models/report.json`) and emits a fresh
`reports/<UTC_TS>-failure-analysis.{json,md}` pair on every retrain.

This is the report-driven sibling of
`scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py`:
since Task #316 landed every diagnostic field directly on each per-slice
record (`per_class_*`, `prediction_collapse`, `regime_bucketed_da`,
`pnl_after_fees`, etc.), we no longer need to re-run inference over the
holdout to rebuild the analysis. The hand-run scripts stay in place for
recomputing from persisted artifacts; this module supplements them.

Hard rules:
- Never invents metrics. Reads only what the trainer already emitted.
- Never aborts training. The `generate_for_report` call is wrapped in a
  best-effort try/except in `run_training`.
- Same bucket rules as the hand-run script (`compute_failure_metrics.assign_bucket`)
  so the bucket counts can be tracked over time without a step change.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ml-engine.failure_analysis")

MIN_HOLDOUT_ROWS = 200
MIN_DIRECTIONAL_ACCURACY = 0.50
DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES = {"5m", "1h", "2h", "6h", "1d"}

# Task #401 — keep the failure-analysis bucket assignment in lock-step
# with the verification gate's per-timeframe directional-accuracy
# floor. If the gate retires a 1d slice with `DA = 0.520` as
# `below_coinflip`, the failure-analysis dashboard MUST also call that
# slice unpromoted; otherwise the bucket count and verification block
# disagree and the operator sees a 1d slice in the `promoted` bucket
# while the verification block reports it as `below_coinflip`.
from .verification import (  # noqa: E402  (avoid circular at module init)
    MIN_DIRECTIONAL_ACCURACY_PER_TF,
    min_directional_accuracy_for,
)

CALIBRATION_BROKEN_RELIABILITY_DEV = 0.10
PREDICTION_COLLAPSE_GAP = 0.15
PREDICTION_COLLAPSE_TOP_SHARE = 0.85
NEAR_PRIOR_DOMINANT_SHARE = 0.60  # >60% of holdout rows within ε of class prior

BUCKET_PROMOTED = "promoted"
# Task #400 — distinct cohort for slices whose served predictor is the
# multinomial-logistic baseline (because the booster lost head-to-head
# on directional accuracy AND the baseline cleared the verification
# gate on its own). Operationally these slices ARE serving — they
# bump the unified `slices_promoted` total — but the failure-analysis
# report carves them out so an operator can tell at a glance which
# slices are riding on the baseline rather than the booster, and can
# prioritise their booster-side investigation accordingly.
BUCKET_PROMOTED_BASELINE = "promoted_baseline_served"
BUCKET_SCHEMA_FIX = "salvageable_with_schema_fix"
BUCKET_INSUFFICIENT_SAMPLE = "insufficient_sample"
BUCKET_RETIRE = "structurally_noisy_retire"
BUCKET_FEATURES_OR_LABELS = "salvageable_with_better_features_or_labels"
BUCKET_UNKNOWN = "unknown"

ALL_BUCKETS = (
    BUCKET_PROMOTED,
    BUCKET_PROMOTED_BASELINE,
    BUCKET_SCHEMA_FIX,
    BUCKET_INSUFFICIENT_SAMPLE,
    BUCKET_RETIRE,
    BUCKET_FEATURES_OR_LABELS,
    BUCKET_UNKNOWN,
)


def _holdout_rows(slice_rep: dict[str, Any]) -> int:
    """Sum of per_class_holdout_breakdown is the source-of-truth for
    the holdout row count emitted alongside the rest of the per-class
    surface in #316."""
    bd = slice_rep.get("per_class_holdout_breakdown") or {}
    if isinstance(bd, dict):
        try:
            return int(sum(int(v) for v in bd.values()))
        except (TypeError, ValueError):
            return 0
    return 0


def _directional_accuracy(slice_rep: dict[str, Any]) -> float | None:
    metrics = slice_rep.get("metrics") or {}
    da = metrics.get("directional_accuracy")
    if isinstance(da, (int, float)):
        try:
            f = float(da)
        except (TypeError, ValueError):
            return None
        # NaN sentinel from a failed slice
        if f != f:  # noqa: PLR0124
            return None
        return f
    return None


def _is_calibration_broken(slice_rep: dict[str, Any]) -> bool:
    rmd = slice_rep.get("reliability_max_dev_per_class") or {}
    if not isinstance(rmd, dict):
        return False
    for v in rmd.values():
        try:
            if float(v) >= CALIBRATION_BROKEN_RELIABILITY_DEV:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _is_prediction_collapsed(slice_rep: dict[str, Any]) -> bool:
    pc = slice_rep.get("prediction_collapse") or {}
    if isinstance(pc, dict):
        try:
            if float(pc.get("collapse_gap") or 0.0) >= PREDICTION_COLLAPSE_GAP:
                return True
        except (TypeError, ValueError):
            pass
        try:
            if float(pc.get("predicted_top_class_share") or 0.0) >= PREDICTION_COLLAPSE_TOP_SHARE:
                return True
        except (TypeError, ValueError):
            pass
    # Match `compute_failure_metrics.is_prediction_collapsed` — also fires
    # when the calibrated head parks on the training class prior. The
    # trainer doesn't emit `predictions_near_prior` today; defensive check
    # so parity holds the moment it lands.
    near = slice_rep.get("predictions_near_prior") or {}
    if isinstance(near, dict):
        try:
            if float(near.get("share_within_eps") or 0.0) >= NEAR_PRIOR_DOMINANT_SHARE:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _is_schema_fix_required(slice_rep: dict[str, Any]) -> bool:
    """Matches `compute_failure_metrics.assign_bucket` step 2:
    `contamination_flag=True OR cadence audit shows mixed source`. The
    trainer doesn't emit `contamination_flag` today; defensive check so
    parity holds the moment it lands.
    """
    if bool(slice_rep.get("contamination_flag")):
        return True
    return bool(slice_rep.get("cadence_mixed"))


def assign_bucket(
    slice_rep: dict[str, Any] | None,
    *,
    timeframe: str,
    is_pooled: bool = False,
) -> tuple[str, str]:
    """Strict, priority-ordered bucket assignment matching #315's rules.

    Returns `(bucket, reason)`. `slice_rep is None` means the trainer
    didn't emit a slice (untrained).
    """
    if slice_rep is None or not isinstance(slice_rep, dict):
        return (
            BUCKET_INSUFFICIENT_SAMPLE,
            "status=untrained (slice missing from report); not enough labeled rows to train",
        )

    status = str(slice_rep.get("status") or "unknown")
    n = _holdout_rows(slice_rep)
    da = _directional_accuracy(slice_rep)
    # Task #401 — resolve the per-tf floor (0.530 at 1d, 0.50 elsewhere).
    tf_min_da = min_directional_accuracy_for(timeframe)

    # Task #401 — strict-greater-than to match the verification gate.
    # `verification.classify_slice` uses `da_f <= min_da → below_coinflip`,
    # so a slice that ties the floor exactly is NOT promoted. The
    # failure-analysis bucket assignment must use the same boundary or
    # the dashboard will mark a slice `BUCKET_PROMOTED` while the
    # verification block marks it `below_coinflip` — a split-brain on
    # the exact-threshold edge case.
    promoted = (
        status == "trained"
        and n >= MIN_HOLDOUT_ROWS
        and da is not None
        and da > tf_min_da
        and timeframe in DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES
    )
    if promoted:
        # Task #400 — split promoted slices by served-predictor identity.
        # A slice whose served head IS the multinomial-logistic baseline
        # gets attributed to the dedicated cohort so the operator can
        # see the booster-vs-baseline breakdown of currently-served
        # slots without losing the underlying promotion.
        served_kind = str(
            slice_rep.get("served_predictor_kind")
            or (slice_rep.get("manifest") or {}).get("served_predictor_kind")
            or "lightgbm"
        )
        if served_kind == "baseline":
            booster_metrics = slice_rep.get("lightgbm_cv_metrics") or {}
            booster_da = booster_metrics.get("directional_accuracy")
            booster_da_str = (
                f"{float(booster_da):.3f}"
                if isinstance(booster_da, (int, float))
                and float(booster_da) == float(booster_da)
                else "—"
            )
            return (
                BUCKET_PROMOTED_BASELINE,
                (
                    f"served=baseline; baseline DA {da:.3f} >= "
                    f"{MIN_DIRECTIONAL_ACCURACY} AND holdout {n} >= "
                    f"{MIN_HOLDOUT_ROWS}; booster CV DA={booster_da_str} lost "
                    f"head-to-head"
                ),
            )
        return (
            BUCKET_PROMOTED,
            f"DA {da:.3f} > {tf_min_da} AND holdout {n} >= {MIN_HOLDOUT_ROWS}",
        )

    if _is_schema_fix_required(slice_rep):
        return (
            BUCKET_SCHEMA_FIX,
            "contamination_flag or cadence_mixed=True on slice; fix the upstream cadence/source before retraining",
        )

    if status != "trained" or n == 0:
        return (
            BUCKET_INSUFFICIENT_SAMPLE,
            f"status={status} (holdout={n}); not enough labeled rows to train",
        )
    if n < MIN_HOLDOUT_ROWS:
        return (
            BUCKET_INSUFFICIENT_SAMPLE,
            f"holdout {n} < MIN_HOLDOUT_ROWS={MIN_HOLDOUT_ROWS}",
        )

    cadence_clean = not _is_schema_fix_required(slice_rep)
    calibration_broken = _is_calibration_broken(slice_rep)
    collapse = _is_prediction_collapsed(slice_rep)

    if cadence_clean and n >= MIN_HOLDOUT_ROWS and calibration_broken and collapse:
        rmd = slice_rep.get("reliability_max_dev_per_class") or {}
        try:
            rmd_max = max(float(v) for v in rmd.values())
        except (TypeError, ValueError):
            rmd_max = float("nan")
        pc = slice_rep.get("prediction_collapse") or {}
        evidence = (
            f"calibration_broken=True (max reliability deviation {rmd_max:.3f} "
            f">= {CALIBRATION_BROKEN_RELIABILITY_DEV}); prediction_collapse=True "
            f"(collapse_gap={pc.get('collapse_gap')}, "
            f"predicted_top_class_share={pc.get('predicted_top_class_share')})"
        )
        return (BUCKET_RETIRE, evidence)

    return (
        BUCKET_FEATURES_OR_LABELS,
        (
            f"red gate but signal remaining — calibration_broken={calibration_broken}, "
            f"prediction_collapse={collapse}; rebalance labels and/or extend feature set"
        ),
    )


def _fmt(x: Any, digits: int = 3) -> str:
    if isinstance(x, (int, float)):
        try:
            f = float(x)
        except (TypeError, ValueError):
            return "—"
        if f != f:  # NaN
            return "—"
        return f"{f:.{digits}f}"
    return "—"


def _coin_label(coin: str) -> str:
    return "(pooled)" if coin == "__pooled__" else coin


def build_analysis(report: dict[str, Any]) -> dict[str, Any]:
    """Walk every per-slice record on `report["timeframes"]` and produce
    the failure-analysis envelope (no IO).
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    timeframes = report.get("timeframes") or {}
    bucket_counts: dict[str, int] = {b: 0 for b in ALL_BUCKETS}
    slices: list[dict[str, Any]] = []

    for tf, tf_report in timeframes.items():
        if not isinstance(tf_report, dict):
            continue
        per_coin = tf_report.get("per_coin") or {}
        pooled = tf_report.get("pooled")

        for coin, slice_rep in per_coin.items():
            bucket, reason = assign_bucket(
                slice_rep if isinstance(slice_rep, dict) else None,
                timeframe=tf,
                is_pooled=False,
            )
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            slices.append(
                _slice_summary(coin, tf, slice_rep, bucket, reason, is_pooled=False)
            )
        if pooled is not None:
            bucket, reason = assign_bucket(
                pooled if isinstance(pooled, dict) else None,
                timeframe=tf,
                is_pooled=True,
            )
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            slices.append(
                _slice_summary("__pooled__", tf, pooled, bucket, reason, is_pooled=True)
            )

    return {
        "generated_at": generated_at,
        "source_report_generated_at": report.get("generated_at"),
        "gate_constants": {
            "MIN_HOLDOUT_ROWS": MIN_HOLDOUT_ROWS,
            "MIN_DIRECTIONAL_ACCURACY": MIN_DIRECTIONAL_ACCURACY,
            # Task #401 — surface the per-tf override so the dashboard's
            # "gate constants" line reflects the same per-timeframe floor
            # the verification block uses.
            "MIN_DIRECTIONAL_ACCURACY_PER_TF": dict(
                MIN_DIRECTIONAL_ACCURACY_PER_TF
            ),
            "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": sorted(
                DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES
            ),
            "CALIBRATION_BROKEN_RELIABILITY_DEV": CALIBRATION_BROKEN_RELIABILITY_DEV,
            "PREDICTION_COLLAPSE_GAP": PREDICTION_COLLAPSE_GAP,
            "PREDICTION_COLLAPSE_TOP_SHARE": PREDICTION_COLLAPSE_TOP_SHARE,
        },
        "bucket_counts": bucket_counts,
        "slices": slices,
    }


def _slice_summary(
    coin: str,
    timeframe: str,
    slice_rep: dict[str, Any] | None,
    bucket: str,
    reason: str,
    *,
    is_pooled: bool,
) -> dict[str, Any]:
    if slice_rep is None or not isinstance(slice_rep, dict):
        slice_rep = {}
    pc = slice_rep.get("prediction_collapse") or {}
    pnl = slice_rep.get("pnl_after_fees") or {}
    metrics = slice_rep.get("metrics") or {}
    baseline_metrics = slice_rep.get("baseline_metrics") or {}
    near_prior = slice_rep.get("predictions_near_prior") or {}
    share_near_prior: float | None
    try:
        snp_raw = near_prior.get("share_within_eps") if isinstance(near_prior, dict) else None
        share_near_prior = float(snp_raw) if snp_raw is not None else None
        if share_near_prior is not None and share_near_prior != share_near_prior:  # noqa: PLR0124
            share_near_prior = None
    except (TypeError, ValueError):
        share_near_prior = None
    return {
        "coin_id": coin,
        "timeframe": timeframe,
        "kind": "pooled" if is_pooled else "per_coin",
        "status": slice_rep.get("status"),
        "bucket": bucket,
        "bucket_reason": reason,
        "n_holdout": _holdout_rows(slice_rep),
        "directional_accuracy": _directional_accuracy(slice_rep),
        "baseline_directional_accuracy": baseline_metrics.get("directional_accuracy"),
        "brier": metrics.get("brier"),
        "baseline_brier": baseline_metrics.get("brier"),
        "collapse_gap": pc.get("collapse_gap"),
        "predicted_top_class_share": pc.get("predicted_top_class_share"),
        "label_top_class_share": pc.get("label_top_class_share"),
        "reliability_max_dev_per_class": slice_rep.get("reliability_max_dev_per_class") or {},
        "per_class_accuracy": slice_rep.get("per_class_accuracy") or {},
        "pnl_after_fees": {
            "n_trades": pnl.get("n_trades"),
            "net_pct_mean": pnl.get("net_pct_mean"),
            "round_trip_cost_pct": pnl.get("round_trip_cost_pct"),
            "win_rate": pnl.get("win_rate"),
        } if pnl else {},
        "cadence_mixed": bool(slice_rep.get("cadence_mixed")),
        "contamination_flag": bool(slice_rep.get("contamination_flag")),
        "predictions_near_prior": {
            "share_within_eps": share_near_prior,
        },
    }


def render_md(analysis: dict[str, Any]) -> str:
    """Render the failure-analysis JSON envelope as a human-readable
    markdown report. Mirrors the section structure of the hand-run
    `render_failure_analysis_md.py` for the parts derivable from the
    report alone (skips the cadence-audit / smallest-first-set / repair
    plan sections, which depend on offline analysis).
    """
    out: list[str] = []
    out.append(f"# Auto failure-analysis — {analysis['generated_at']}")
    out.append("")
    out.append(
        f"- **Source report generated_at:** `{analysis.get('source_report_generated_at')}`"
    )
    out.append(f"- **Gate constants:** {json.dumps(analysis['gate_constants'])}")
    out.append("")
    out.append(
        "> Auto-generated from `models/report.json` after every retrain — no offline "
        "re-inference required. For the full hand-run analysis (cadence audit, "
        "smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + "
        "`scripts/render_failure_analysis_md.py` against the persisted artifacts."
    )
    out.append("")

    out.append("## 1. Bucket assignment summary")
    out.append("")
    out.append("| Bucket | Count |")
    out.append("|---|---|")
    for k, v in sorted(
        analysis["bucket_counts"].items(), key=lambda kv: -kv[1]
    ):
        if v == 0:
            continue
        out.append(f"| `{k}` | {v} |")
    out.append("")

    slices = analysis["slices"]
    tfs = sorted({s["timeframe"] for s in slices})
    buckets = sorted({s["bucket"] for s in slices})
    out.append("## 2. Bucket × timeframe matrix")
    out.append("")
    out.append("| Timeframe | " + " | ".join(buckets) + " | total |")
    out.append("|" + "|".join(["---"] * (len(buckets) + 2)) + "|")
    for tf in tfs:
        cs = [s for s in slices if s["timeframe"] == tf]
        counts = {b: 0 for b in buckets}
        for s in cs:
            counts[s["bucket"]] = counts.get(s["bucket"], 0) + 1
        out.append(
            "| " + tf + " | "
            + " | ".join(str(counts.get(b, 0)) for b in buckets)
            + f" | {len(cs)} |"
        )
    out.append("")

    out.append("## 3. Per-slice detail")
    out.append("")
    for tf in tfs:
        out.append(f"### {tf}")
        out.append("")
        out.append(
            "| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | "
            "collapse_gap | top pred share | share near prior | contam | "
            "net PnL %/trade | reason |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        cs = [s for s in slices if s["timeframe"] == tf]
        cs.sort(key=lambda s: (s["coin_id"] != "__pooled__", s["coin_id"]))
        for s in cs:
            da_str = (
                f"{_fmt(s['directional_accuracy'])} / "
                f"{_fmt(s['baseline_directional_accuracy'])}"
            )
            br_str = f"{_fmt(s['brier'])} / {_fmt(s['baseline_brier'])}"
            pnl = s.get("pnl_after_fees") or {}
            pnl_str = (
                f"{_fmt(pnl.get('net_pct_mean'))} (n={pnl.get('n_trades') or 0})"
                if pnl else "—"
            )
            near = s.get("predictions_near_prior") or {}
            share_str = _fmt(near.get("share_within_eps"))
            contam_str = "⚠" if s.get("contamination_flag") else "—"
            note = (s.get("bucket_reason") or "")
            if len(note) > 80:
                note = note[:77] + "…"
            out.append(
                f"| {_coin_label(s['coin_id'])} | `{s['bucket']}` | {s['n_holdout']} | "
                f"{da_str} | {br_str} | {_fmt(s.get('collapse_gap'))} | "
                f"{_fmt(s.get('predicted_top_class_share'))} | {share_str} | "
                f"{contam_str} | {pnl_str} | {note} |"
            )
        out.append("")

    return "\n".join(out)


def _reports_dir(registry_root: Path) -> Path:
    """`models/` and `reports/` are siblings under `artifacts/ml-engine/`."""
    return registry_root.parent / "reports"


MAX_AUTO_REPORTS = 100


def prune_auto_reports(reports_dir: Path, *, keep: int = MAX_AUTO_REPORTS) -> int:
    """Trim ``*-failure-analysis-auto.{json,md}`` pairs in ``reports_dir``
    down to the newest ``keep`` entries.

    Hand-run files (without the ``-auto`` suffix) are left alone. Pairing
    is keyed by the JSON file's stem so an orphaned ``.md`` without a
    matching ``.json`` is also removed once the pair falls outside the
    retention window. Never raises; returns the number of files deleted.
    """
    deleted = 0
    try:
        if not reports_dir.exists():
            return 0
        json_files = sorted(reports_dir.glob("*-failure-analysis-auto.json"))
        if len(json_files) <= keep:
            return 0
        to_drop = json_files[: len(json_files) - keep]
        for jp in to_drop:
            mp = jp.parent / (jp.stem + ".md")
            for p in (jp, mp):
                try:
                    if p.exists():
                        p.unlink()
                        deleted += 1
                except OSError as exc:
                    logger.warning(
                        "failure_analysis_prune_unlink_failed",
                        extra={"path": str(p), "error": str(exc)},
                    )
        return deleted
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failure_analysis_prune_failed", extra={"error": str(exc)},
        )
        return deleted


def generate_for_report(
    report: dict[str, Any],
    registry_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build, render, and persist a failure-analysis pair from `report`.

    Returns a small envelope describing what was written so the caller
    can stash it on the in-memory report. Never raises.
    """
    summary: dict[str, Any] = {
        "status": "ok",
        "generated_at": None,
        "json_path": None,
        "md_path": None,
        "bucket_counts": {},
    }
    try:
        analysis = build_analysis(report)
        md = render_md(analysis)
        ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        rd = _reports_dir(registry_root)
        rd.mkdir(parents=True, exist_ok=True)
        json_path = rd / f"{ts}-failure-analysis-auto.json"
        md_path = rd / f"{ts}-failure-analysis-auto.md"
        json_path.write_text(json.dumps(analysis, indent=2, default=str))
        md_path.write_text(md)
        prune_auto_reports(rd)
        summary.update(
            {
                "generated_at": analysis["generated_at"],
                "json_path": str(json_path),
                "md_path": str(md_path),
                "bucket_counts": analysis["bucket_counts"],
            }
        )
        logger.info(
            "failure_analysis_written",
            extra={
                "json_path": str(json_path),
                "bucket_counts": analysis["bucket_counts"],
            },
        )
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failure_analysis_generate_failed", extra={"error": str(exc)},
        )
        summary["status"] = "error"
        summary["error"] = str(exc)
        return summary


def latest_pair(registry_root: Path) -> dict[str, Any]:
    """Read the newest auto-generated failure-analysis pair from disk.

    Returns `{generated_at, bucket_counts, summary_md, json_path, md_path}`
    or an empty envelope when no pair exists. Never raises.
    """
    empty = {
        "generated_at": None,
        "bucket_counts": {},
        "summary_md": "",
        "json_path": None,
        "md_path": None,
    }
    try:
        rd = _reports_dir(registry_root)
        if not rd.exists():
            return empty
        candidates = sorted(rd.glob("*-failure-analysis-auto.json"))
        if not candidates:
            return empty
        json_path = candidates[-1]
        md_path = json_path.with_suffix("").with_suffix(".md")
        # The above replaces .json then drops, simpler:
        md_path = json_path.parent / (json_path.stem + ".md")
        analysis = json.loads(json_path.read_text())
        md = md_path.read_text() if md_path.exists() else ""
        return {
            "generated_at": analysis.get("generated_at"),
            "bucket_counts": analysis.get("bucket_counts") or {},
            "summary_md": md,
            "json_path": str(json_path),
            "md_path": str(md_path) if md_path.exists() else None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failure_analysis_read_failed", extra={"error": str(exc)},
        )
        return {**empty, "error": str(exc)}


def history(registry_root: Path, *, limit: int = 30) -> dict[str, Any]:
    """Read the newest N auto-generated failure-analysis JSONs from disk.

    Returns `{rows: [{generated_at, bucket_counts, json_path}], count}`,
    newest first. Used by `/ml/admin/failure-analysis/history` so the
    diagnostics page can sparkline how each bucket trends across
    consecutive retrains. Never raises.
    """
    out: dict[str, Any] = {"rows": [], "count": 0}
    try:
        if limit <= 0:
            return out
        rd = _reports_dir(registry_root)
        if not rd.exists():
            return out
        candidates = sorted(rd.glob("*-failure-analysis-auto.json"))
        if not candidates:
            return out
        rows: list[dict[str, Any]] = []
        # Newest first, capped at `limit`.
        for json_path in reversed(candidates[-limit:]):
            try:
                analysis = json.loads(json_path.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "failure_analysis_history_skip",
                    extra={"path": str(json_path), "error": str(exc)},
                )
                continue
            rows.append(
                {
                    "generated_at": analysis.get("generated_at"),
                    "bucket_counts": analysis.get("bucket_counts") or {},
                    "json_path": str(json_path),
                }
            )
        out["rows"] = rows
        out["count"] = len(rows)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failure_analysis_history_failed", extra={"error": str(exc)},
        )
        return {**out, "error": str(exc)}
