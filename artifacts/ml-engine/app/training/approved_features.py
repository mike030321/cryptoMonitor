"""Bridge from the api-server's Feature Lab approval flow into training.

When an operator approves a candidate feature in the UI, the api-server
writes the candidate spec into the `app_settings` table under key
``feature_lab.approved`` (see
``artifacts/api-server/src/lib/feature-lab.ts``). The training pipeline
reads that setting at the start of each run and applies the approved
transforms to the labeled dataset before fitting any models, so the
resulting model manifests carry an extended ``feature_schema_hash`` and
any model trained against the old schema must re-enter validation.

Transforms are intentionally narrow and reuse the same allow-listed
implementations the ablation runner uses (see
``app.training.feature_lab._apply_transform``) — no user-supplied code
is ever evaluated.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from .feature_lab import SUPPORTED_TRANSFORMS, _apply_transform
from .registry import FORBIDDEN_FEATURE_PREFIXES

logger = logging.getLogger("ml-engine.train.approved_features")

APPROVED_FEATURES_SETTING_KEY = "feature_lab.approved"


async def fetch_approved_features() -> list[dict]:
    """Read the approved-features list from `app_settings`.

    Returns a list of dicts with keys: ``name``, ``transform_kind``,
    ``source_column`` (which may be None). Returns an empty list when
    DATABASE_URL is unset (e.g. the unit-test suite without a DB),
    when the row is missing, or when the JSON payload is malformed.
    """
    if not os.environ.get("DATABASE_URL"):
        return []
    try:
        from ..db import init_pool
        pool = await init_pool()
        row = await pool.fetchrow(
            "SELECT value FROM app_settings WHERE key = $1",
            APPROVED_FEATURES_SETTING_KEY,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; never break a run
        logger.warning(
            "approved_features_fetch_failed", extra={"error": str(exc)},
        )
        return []
    if not row or row["value"] is None:
        return []
    raw = row["value"]
    # asyncpg returns jsonb as either str or already-decoded; normalize.
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(raw, dict):
        return []
    feats = raw.get("features")
    if not isinstance(feats, list):
        return []
    out: list[dict] = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        kind = f.get("transformKind") or f.get("transform_kind")
        source = f.get("sourceColumn") or f.get("source_column")
        if not isinstance(name, str) or not isinstance(kind, str):
            continue
        out.append(
            {"name": name, "transform_kind": kind,
             "source_column": source if isinstance(source, str) else None}
        )
    return out


def _is_forbidden_feature_name(name: object) -> bool:
    if not isinstance(name, str):
        return False
    return any(name.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)


def apply_approved_features(
    df: pd.DataFrame, approved: list[dict],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply each approved transform to `df`, returning the extended
    dataframe and the list of newly-added column names (in order).

    Skips a candidate (with a warning) if its transform is unsupported,
    its source column is missing, or applying it raises. The training
    run continues with whichever approved features did apply cleanly —
    one bad row in `app_settings` should not nuke the run.
    """
    added: list[str] = []
    if df.empty or not approved:
        return df, added
    out = df
    for spec in approved:
        name = spec.get("name")
        kind = spec.get("transform_kind")
        source = spec.get("source_column")
        if not name or not kind:
            continue
        if _is_forbidden_feature_name(name):
            # Task #387 — Quant-Only Enforcement: an approved-feature row
            # whose name matches a forbidden prefix would re-introduce
            # the LLM/news channel into the trained schema and the
            # runtime guard in `registry.load_model` would then reject
            # every freshly-refit manifest, blanking the fleet. Filter
            # these out at the source so the auto-retrain loop cannot
            # re-poison itself via a stale `app_settings` row.
            logger.warning(
                "approved_feature_forbidden_prefix_skipped",
                extra={"feature_name": name, "transform_kind": kind},
            )
            continue
        if kind not in SUPPORTED_TRANSFORMS:
            logger.warning(
                "approved_feature_unsupported_transform",
                extra={"feature_name": name, "transform_kind": kind},
            )
            continue
        if name in out.columns:
            # Already present (e.g. a previous run materialized it). Trust
            # the existing column rather than overwriting.
            added.append(name)
            continue
        try:
            out = _apply_transform(out, kind, source, name)
        except Exception as exc:  # noqa: BLE001 — one bad spec ≠ broken run
            logger.warning(
                "approved_feature_apply_failed",
                extra={"feature_name": name, "transform_kind": kind, "error": str(exc)},
            )
            continue
        added.append(name)
    return out, added


def extend_feature_columns(
    base: list[str], added: list[str], *, categorical: Optional[list[str]] = None,
) -> list[str]:
    """Splice approved feature names into `base` so they sit BEFORE the
    trailing categorical columns (LightGBM treats categoricals positionally
    via the registered list, but it's still cleaner to keep them at the end
    of the schema for human inspection of the manifest).

    Idempotent: a name already in `base` is not duplicated.
    """
    cats = set(categorical or [])
    # Task #387 — even if a forbidden-prefix name slips through to this
    # layer (caller bypass, custom pipeline, etc.), strip it from BOTH
    # the base schema and the added list. The training contract is
    # Quant-Only; `registry.load_model` will reject any manifest whose
    # `feature_names` carries a forbidden column, so we MUST never write
    # one to disk.
    safe_base = [c for c in base if not _is_forbidden_feature_name(c)]
    dropped_base = [c for c in base if _is_forbidden_feature_name(c)]
    if dropped_base:
        logger.warning(
            "extend_feature_columns_dropped_forbidden_from_base",
            extra={"dropped": dropped_base},
        )
    safe_added: list[str] = []
    for n in added or []:
        if _is_forbidden_feature_name(n):
            logger.warning(
                "extend_feature_columns_dropped_forbidden_from_added",
                extra={"feature_name": n},
            )
            continue
        safe_added.append(n)
    if not safe_added:
        return list(safe_base)
    head = [c for c in safe_base if c not in cats]
    tail = [c for c in safe_base if c in cats]
    seen = set(head) | set(tail)
    for n in safe_added:
        if n in seen:
            continue
        head.append(n)
        seen.add(n)
    return head + tail
