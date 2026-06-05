"""Auto-register newly trained models into the registry as `shadow` rows.

Task #220 — after every successful training run we insert one row into the
shared `model_registry` Postgres table per trained
`(model_id, model_version, coin_id, timeframe)` tuple, with `state='shadow'`.
That makes the lifecycle end-to-end: previously an operator had to POST
to `/crypto/model-registry` after every retrain before promote/rollback
could see the new version.

Design notes:
  - We write directly to the `model_registry` table via asyncpg (the same
    pool ml-engine already uses for read-only feature data). No HTTP call
    to api-server, so the registration succeeds even if api-server is
    momentarily down.
  - Idempotent: re-runs of the same training output (same version, same
    slot) skip via a SELECT-then-INSERT check. The slot key is
    `(model_id, model_version, coin_id, timeframe)`. There is no DB-level
    unique index on that tuple, so we don't rely on ON CONFLICT — and
    creating a new index in this task would be out of scope.
  - Best-effort: every insert is wrapped so a single failure (bad row,
    DB hiccup) doesn't tank the training contract. The summary returned
    is folded into the training report so operators can audit.
  - Covers: per-coin slices, the pooled fallback (incl. prior-only
    fallback emitted by `_train_prior_pooled`), Phase-3 specialists,
    and the Phase-4 meta models from `train_meta.run_meta_training`.
    All trained slices land as shadow — promotion stays a deliberate
    operator action.
"""
from __future__ import annotations

import json as _json
import logging
from typing import Optional

from ..db import init_pool

logger = logging.getLogger("ml-engine.train.register_shadow")

# `model_id` distinguishes the family of model that occupies a registry
# slot. The api-server defaults `getCurrentChampion`'s lookup to "lightgbm"
# so per-coin / pooled / specialist slots all share that family. Meta
# models live under "lightgbm-meta" — same convention as the comment in
# `lib/db/src/schema/model_registry.ts`.
BASE_MODEL_ID = "lightgbm"
META_MODEL_ID = "lightgbm-meta"


def _slice_metrics_snapshot(slc: dict) -> dict:
    """Compact per-slice snapshot stored alongside the registry row.

    Mirrors the shape the api-server's `registerModel` would have written
    so a downstream UI doesn't have to special-case auto-registered rows.
    Kept small on purpose — the full training report still lives at
    `models/report.json` for any deep dive. Meta-only fields
    (`meta_status`) are preserved verbatim when present so the registry
    UI can tell heuristic-deployed meta models apart from LightGBM-trained
    ones without re-loading the manifest.
    """
    snapshot = {
        "metrics": slc.get("metrics"),
        "baseline_metrics": slc.get("baseline_metrics"),
        "directional_call_share": slc.get("directional_call_share"),
        "directional_call_share_n": slc.get("directional_call_share_n"),
        "n_rows": slc.get("n_rows"),
        "model_kind": slc.get("model_kind") or "lightgbm",
        "specialist_kind": slc.get("specialist_kind"),
        "regime_subset": slc.get("regime_subset"),
        "feature_schema_hash": slc.get("feature_schema_hash"),
    }
    if "meta_status" in slc:
        snapshot["meta_status"] = slc["meta_status"]
    # Task #235 — surface which approved feature-lab features actually
    # baked into this trained slice so the Feature Lab + Model Registry
    # UIs can confirm an approved feature went live in a specific model
    # version. The list is the timeframe-level
    # `approved_features_applied` injected into each slice by
    # `_slices_from_report`. Empty list means "no approved features were
    # added on top of the base FEATURE_COLUMNS for this slice's TF."
    if "approved_features_applied" in slc:
        snapshot["approved_features_applied"] = list(
            slc.get("approved_features_applied") or []
        )
    return snapshot


def _slices_from_report(
    report: dict,
) -> list[tuple[str, str, str, str, dict]]:
    """Walk a finished training report and yield every trained registry
    slot as `(model_id, coin_id, timeframe, version, snapshot)`.

    Includes per-coin, pooled (real or prior-only), Phase-3 specialists,
    and Phase-4 meta models. Skips anything whose status isn't `trained`
    (insufficient_data / errors stay out of the registry).
    """
    out: list[tuple[str, str, str, str, dict]] = []
    # Imported lazily so the module can be imported without dragging the
    # registry/lightgbm import chain (the helper is also unit-testable
    # in isolation).
    from .registry import POOLED_COIN_ID

    for tf, tf_report in (report.get("timeframes") or {}).items():
        if not isinstance(tf_report, dict):
            continue
        # Task #235 — the timeframe-level `approved_features_applied`
        # tells us which feature-lab features were baked into every base
        # slice trained for this TF. Inject it into each slice so the
        # snapshot carries a per-version provenance record.
        approved_applied = list(tf_report.get("approved_features_applied") or [])
        for coin, slc in (tf_report.get("per_coin") or {}).items():
            if not isinstance(slc, dict) or slc.get("status") != "trained":
                continue
            v = slc.get("version")
            if not v:
                continue
            slc.setdefault("approved_features_applied", approved_applied)
            out.append((BASE_MODEL_ID, str(coin), str(tf), str(v), slc))
        pooled = tf_report.get("pooled")
        if isinstance(pooled, dict) and pooled.get("status") == "trained" and pooled.get("version"):
            pooled.setdefault("approved_features_applied", approved_applied)
            out.append((
                BASE_MODEL_ID, POOLED_COIN_ID, str(tf),
                str(pooled["version"]), pooled,
            ))
        for kind, slc in (tf_report.get("specialists") or {}).items():
            if not isinstance(slc, dict) or slc.get("status") != "trained":
                continue
            v = slc.get("version")
            coin_id = slc.get("coin_id")
            if not v or not coin_id:
                continue
            slc.setdefault("approved_features_applied", approved_applied)
            out.append((BASE_MODEL_ID, str(coin_id), str(tf), str(v), slc))

    # Phase 4 meta models. `run_meta_training` returns `{tf -> {status,
    # version, ...}}`. Both the LightGBM-trained and heuristic-deployed
    # branches produce a usable, deployed version that should appear in
    # the registry; an "error" entry has no version and is skipped.
    for tf, meta in (report.get("meta_models") or {}).items():
        if not isinstance(meta, dict):
            continue
        v = meta.get("version")
        status = meta.get("status")
        if not v or status not in ("trained", "heuristic"):
            continue
        # Tag the snapshot with the meta status so a UI can tell trained
        # vs. heuristic at a glance without re-loading the manifest.
        snapshot = {"meta_status": status, "metrics": meta.get("metrics")}
        out.append((META_MODEL_ID, "__meta__", str(tf), str(v), snapshot))

    return out


async def register_shadow_rows(report: dict) -> dict:
    """Insert shadow rows for every trained slice in `report`. Idempotent
    and best-effort. Returns a summary dict suitable for embedding back
    into the training report.
    """
    summary: dict = {
        "candidates": 0,
        "inserted": 0,
        "skipped_existing": 0,
        "errors": 0,
    }
    try:
        slices = _slices_from_report(report)
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry_shadow_walk_failed", extra={"error": str(exc)})
        summary["errors"] += 1
        summary["status"] = "error"
        return summary

    summary["candidates"] = len(slices)
    if not slices:
        summary["status"] = "noop"
        return summary

    try:
        pool = await init_pool()
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry_shadow_pool_failed", extra={"error": str(exc)})
        summary["errors"] += 1
        summary["status"] = "error"
        return summary

    generated_at = report.get("generated_at")
    note_template = (
        f"Auto-registered as shadow after training run "
        f"(generated_at={generated_at})."
    )

    for model_id, coin_id, timeframe, version, slc in slices:
        try:
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    """
                    SELECT id FROM model_registry
                    WHERE model_id = $1 AND model_version = $2
                      AND coin_id = $3 AND timeframe = $4
                    LIMIT 1
                    """,
                    model_id, version, coin_id, timeframe,
                )
                if existing is not None:
                    summary["skipped_existing"] += 1
                    continue
                metrics_snapshot = _slice_metrics_snapshot(slc)
                await conn.execute(
                    """
                    INSERT INTO model_registry
                        (model_id, model_version, coin_id, timeframe,
                         state, note, metrics_snapshot, is_active)
                    VALUES ($1, $2, $3, $4, 'shadow', $5, $6::jsonb, true)
                    """,
                    model_id, version, coin_id, timeframe,
                    note_template,
                    _json.dumps(metrics_snapshot, default=str),
                )
                summary["inserted"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "registry_shadow_insert_failed",
                extra={
                    "model_id": model_id, "coin_id": coin_id,
                    "timeframe": timeframe, "version": version,
                    "error": str(exc),
                },
            )
            summary["errors"] += 1

    summary["status"] = "ok" if summary["errors"] == 0 else "partial"
    logger.info("registry_shadow_registered", extra=dict(summary))
    return summary
