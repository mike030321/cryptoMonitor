"""Task #236 — auto-retire approved features that fail validation in the
next training run.

After every training run we compare the freshly-trained pooled models'
validation metrics (per timeframe) against the previous run's report.
If the pooled validation `log_loss` regressed beyond a threshold AND
that timeframe applied an approved feature *for the first time* (i.e.
the feature was not in the prior run's `approved_features_applied`
list), we treat the new feature as the likely cause and quarantine it:

  - Remove it from the `feature_lab.approved` app_settings bucket so the
    *next* training run drops it from the schema.
  - Append it to `feature_lab.quarantined` (a parallel bucket) with the
    reason and the metric snapshot that triggered the move.
  - Flip the corresponding `feature_lab_candidates` row's `state` to
    "quarantined" so the operator UI surfaces it.
  - Log an operator alert.

We deliberately key on the **pooled** model's log_loss because it is
the single per-timeframe metric every training run produces (per-coin
slices may be `insufficient_data` on small-history coins). A regression
threshold of `+0.05` log_loss is the same band the existing meta-model
guardrail uses (see `auto-quarantine.ts`); it's loud enough to ignore
fold-to-fold noise on the 200-300 row pooled holdouts.

The whole module is best-effort: a missing prior report, missing DB,
or a malformed `app_settings` row never aborts the training contract.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from .approved_features import APPROVED_FEATURES_SETTING_KEY

logger = logging.getLogger("ml-engine.train.auto_retire")

QUARANTINED_FEATURES_SETTING_KEY = "feature_lab.quarantined"

# Validation regression threshold. A pooled log_loss increase strictly
# greater than this (current - prior) is considered a regression. 0.05
# matches the meta-model auto-quarantine guardrail in auto-quarantine.ts.
DEFAULT_LOG_LOSS_REGRESSION_THRESHOLD = float(
    os.environ.get("ML_AUTO_RETIRE_LOG_LOSS_THRESHOLD", "0.05")
)


def _pooled_log_loss(tf_report: dict) -> Optional[float]:
    """Extract the pooled model's mean validation log_loss for a
    timeframe report, or None when the pooled slot did not produce a
    LightGBM fit (insufficient data / prior-only fallback / error).
    """
    if not isinstance(tf_report, dict):
        return None
    pooled = tf_report.get("pooled")
    if not isinstance(pooled, dict):
        return None
    if pooled.get("status") != "trained":
        return None
    metrics = pooled.get("metrics")
    if not isinstance(metrics, dict):
        return None
    val = metrics.get("log_loss")
    try:
        if val is None:
            return None
        v = float(val)
    except (TypeError, ValueError):
        return None
    # NaN propagates as != itself.
    if v != v:  # noqa: PLR0124
        return None
    return v


def _approved_applied_set(tf_report: dict) -> set[str]:
    """Names of approved features actually applied to this timeframe's
    slice. Pulled from the report row written by `run_training`.
    """
    if not isinstance(tf_report, dict):
        return set()
    applied = tf_report.get("approved_features_applied")
    if not isinstance(applied, list):
        return set()
    return {str(n) for n in applied if isinstance(n, str)}


def diagnose_regressions(
    current_report: dict,
    prior_report: Optional[dict],
    *,
    threshold: float = DEFAULT_LOG_LOSS_REGRESSION_THRESHOLD,
) -> list[dict]:
    """Pure helper — returns one decision row per (timeframe, newly
    applied feature) that failed validation. Decision rows have:

        {
          "timeframe": str,
          "feature_name": str,
          "current_log_loss": float,
          "prior_log_loss": float,
          "delta_log_loss": float,        # current - prior, > threshold
          "threshold": float,
          "first_appearance": True,       # always True in the returned set
        }

    No DB / no I/O — used by tests and by `auto_retire_after_training`.
    """
    out: list[dict] = []
    if not prior_report:
        return out
    cur_tfs = (current_report or {}).get("timeframes") or {}
    prev_tfs = prior_report.get("timeframes") or {}
    for tf, tf_report in cur_tfs.items():
        cur_ll = _pooled_log_loss(tf_report)
        prev_ll = _pooled_log_loss(prev_tfs.get(tf) or {})
        if cur_ll is None or prev_ll is None:
            continue
        delta = cur_ll - prev_ll
        if delta <= threshold:
            continue
        cur_applied = _approved_applied_set(tf_report)
        prev_applied = _approved_applied_set(prev_tfs.get(tf) or {})
        new_features = cur_applied - prev_applied
        for name in sorted(new_features):
            out.append({
                "timeframe": tf,
                "feature_name": name,
                "current_log_loss": cur_ll,
                "prior_log_loss": prev_ll,
                "delta_log_loss": delta,
                "threshold": threshold,
                "first_appearance": True,
            })
    return out


def _spec_for_feature(name: str, approved_specs: list[dict]) -> Optional[dict]:
    for spec in approved_specs:
        if isinstance(spec, dict) and spec.get("name") == name:
            return spec
    return None


async def _persist_quarantine(
    decisions: list[dict], approved_specs: list[dict],
) -> dict:
    """Apply decisions to `app_settings` (and the candidates table when
    available). Returns a summary dict suitable for embedding in the
    training report.

    Idempotent: features already in `feature_lab.quarantined` are not
    duplicated; the latest reason / detail is recorded.
    """
    summary: dict = {
        "status": "skipped",
        "reason": None,
        "decisions": decisions,
        "quarantined_names": [],
    }
    if not decisions:
        summary["status"] = "noop"
        return summary
    if not os.environ.get("DATABASE_URL"):
        summary["reason"] = "no_database_url"
        return summary

    # Group by feature name — a single name may be implicated across
    # multiple timeframes; we still only quarantine it once.
    by_name: dict[str, list[dict]] = {}
    for d in decisions:
        by_name.setdefault(d["feature_name"], []).append(d)

    try:
        from ..db import init_pool
        pool = await init_pool()
    except Exception as exc:  # noqa: BLE001 — best-effort; never break a run
        logger.warning("auto_retire_db_unavailable", extra={"error": str(exc)})
        summary["reason"] = f"db_init_failed: {exc}"
        return summary

    quarantined_names: list[str] = []
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Re-read the approved bucket inside the transaction so we
                # don't race with a concurrent operator approval.
                row = await conn.fetchrow(
                    "SELECT value FROM app_settings WHERE key = $1",
                    APPROVED_FEATURES_SETTING_KEY,
                )
                live_approved = _coerce_features_list(row["value"] if row else None)
                # Prefer specs from the live bucket; fall back to the
                # report-time copy so we still record kind/source even
                # when the operator just deleted it.
                lookup = list(live_approved) + list(approved_specs)
                remaining = [
                    spec for spec in live_approved
                    if not isinstance(spec, dict) or spec.get("name") not in by_name
                ]

                row_q = await conn.fetchrow(
                    "SELECT value FROM app_settings WHERE key = $1",
                    QUARANTINED_FEATURES_SETTING_KEY,
                )
                live_q = _coerce_features_list(row_q["value"] if row_q else None)
                # Drop any prior records for the names we're about to
                # re-record so we don't accumulate stale duplicates.
                live_q = [
                    rec for rec in live_q
                    if not isinstance(rec, dict) or rec.get("name") not in by_name
                ]

                now_iso = datetime.now(timezone.utc).isoformat()
                for name, decs in by_name.items():
                    spec = _spec_for_feature(name, lookup) or {}
                    worst = max(decs, key=lambda d: d["delta_log_loss"])
                    record = {
                        "name": name,
                        "transformKind": spec.get("transformKind")
                            or spec.get("transform_kind"),
                        "sourceColumn": spec.get("sourceColumn")
                            or spec.get("source_column"),
                        "quarantinedAt": now_iso,
                        "reason": "validation_regression",
                        "detail": {
                            "trigger": "auto_retire_after_training",
                            "timeframes": [
                                {
                                    "timeframe": d["timeframe"],
                                    "current_log_loss": d["current_log_loss"],
                                    "prior_log_loss": d["prior_log_loss"],
                                    "delta_log_loss": d["delta_log_loss"],
                                }
                                for d in decs
                            ],
                            "threshold": worst["threshold"],
                        },
                    }
                    live_q.append(record)
                    quarantined_names.append(name)

                approved_payload = json.dumps({"features": remaining})
                quarantined_payload = json.dumps({"features": live_q})
                await conn.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO UPDATE
                      SET value = EXCLUDED.value, updated_at = now()
                    """,
                    APPROVED_FEATURES_SETTING_KEY, approved_payload,
                )
                await conn.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO UPDATE
                      SET value = EXCLUDED.value, updated_at = now()
                    """,
                    QUARANTINED_FEATURES_SETTING_KEY, quarantined_payload,
                )
                # Best-effort candidate-state flip. Wrapped in a nested
                # asyncpg `conn.transaction()` which (because we are
                # already inside an outer transaction) opens a real
                # Postgres SAVEPOINT. If the UPDATE fails — e.g. on an
                # older deployment without the column, or because the
                # table is missing in a test DB — the savepoint rolls
                # back cleanly while the outer transaction (the two
                # app_settings writes above) commits. We deliberately
                # let any exception propagate out of the `async with`
                # so asyncpg performs the ROLLBACK TO SAVEPOINT, then
                # swallow it here so the bucket updates still commit.
                try:
                    async with conn.transaction():
                        await conn.execute(
                            """
                            UPDATE feature_lab_candidates
                               SET state = 'quarantined',
                                   updated_at = now(),
                                   approval_note = COALESCE(approval_note, '') ||
                                     CASE WHEN approval_note IS NULL OR approval_note = ''
                                          THEN ''
                                          ELSE E'\n'
                                     END ||
                                     'auto-retired: validation regression'
                             WHERE name = ANY($1::text[])
                            """,
                            list(by_name.keys()),
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "auto_retire_candidate_state_update_failed",
                        extra={"error": str(exc), "names": list(by_name.keys())},
                    )
    except Exception as exc:  # noqa: BLE001
        logger.warning("auto_retire_persist_failed", extra={"error": str(exc)})
        summary["status"] = "error"
        summary["reason"] = str(exc)
        return summary

    summary["status"] = "applied"
    summary["quarantined_names"] = quarantined_names
    return summary


def _coerce_features_list(raw: object) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(raw, dict):
        return []
    feats = raw.get("features")
    if not isinstance(feats, list):
        return []
    return [f for f in feats if isinstance(f, dict)]


async def auto_retire_after_training(
    current_report: dict,
    prior_report: Optional[dict],
    approved_specs: list[dict],
    *,
    threshold: float = DEFAULT_LOG_LOSS_REGRESSION_THRESHOLD,
) -> dict:
    """End-to-end entry point called by `run_training` once a fresh
    report has been built. Returns a summary dict that the trainer
    embeds in the report under `auto_retired_features`.

    Never raises: any internal failure is logged and surfaced as a
    `status="error"` summary so the training contract holds.
    """
    try:
        decisions = diagnose_regressions(
            current_report, prior_report, threshold=threshold,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("auto_retire_diagnose_failed", extra={"error": str(exc)})
        return {"status": "error", "reason": str(exc), "decisions": []}
    if not decisions:
        return {"status": "noop", "decisions": [], "quarantined_names": []}
    summary = await _persist_quarantine(decisions, approved_specs)
    if summary.get("status") == "applied":
        for d in decisions:
            logger.warning(
                "feature_auto_retired feature=%s timeframe=%s "
                "log_loss prior=%.4f current=%.4f delta=%.4f threshold=%.4f",
                d["feature_name"], d["timeframe"],
                d["prior_log_loss"], d["current_log_loss"],
                d["delta_log_loss"], d["threshold"],
            )
    return summary
