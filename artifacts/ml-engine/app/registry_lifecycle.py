"""Phase 5 — champion/challenger lifecycle gate logic.

The DB-backed registry rows live in Postgres (`model_registry` table, owned
by the api-server). This module provides the *pure* gate evaluator that
decides whether a `challenger` is fit for promotion to `champion`.

Inputs are aggregate metrics (samples, net edge, drawdown, per-regime net
edge breakdown) collected by the api-server from the prediction journal.
Output is a structured verdict with a `pass/fail` per gate so the UI can
show exactly which line a model fails.
"""
from __future__ import annotations

import json as _json
import logging as _logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.backtest.contract import _require, get_frictions

_logger = _logging.getLogger("ml-engine.registry_lifecycle")


@dataclass
class PromotionMetrics:
    samples: int
    net_edge_pct: float                 # challenger's realized net edge (after fees)
    champion_net_edge_pct: float        # current champion's realized net edge
    drawdown_pct: float                 # worst window equity drawdown over shadow stream
    per_regime_net_edge_pct: dict[str, float]  # regime label -> net_edge_pct


@dataclass
class PromotionVerdict:
    eligible: bool
    samples_ok: bool
    edge_lift_ok: bool
    drawdown_ok: bool
    regime_robustness_ok: bool
    reasons: list[str]
    thresholds: dict
    metrics_summary: dict


def evaluate_promotion(metrics: PromotionMetrics) -> PromotionVerdict:
    fr = get_frictions()
    # Task #349 — fail-fast loader. The previous version silently fell back
    # to (200, 0.5, 20.0, 0.5) when the `champion_challenger` block (or
    # individual keys) were absent. That meant a typo in the JSON would let
    # the registry promote challengers against thresholds that exist nowhere
    # in the contract — invisible to anyone reading shared/trading-frictions.json.
    cc = _require(fr.raw, "champion_challenger", "raw")
    min_samples = int(_require(cc, "min_shadow_samples", "champion_challenger"))
    min_lift = float(_require(cc, "min_net_edge_lift_vs_champion", "champion_challenger"))
    max_dd = float(_require(cc, "max_drawdown_pct", "champion_challenger"))
    min_passing_share = float(_require(cc, "min_passing_regimes_share", "champion_challenger"))

    reasons: list[str] = []

    samples_ok = metrics.samples >= min_samples
    if not samples_ok:
        reasons.append(f"need {min_samples} samples, have {metrics.samples}")

    lift = metrics.net_edge_pct - metrics.champion_net_edge_pct
    edge_lift_ok = lift >= min_lift
    if not edge_lift_ok:
        reasons.append(f"net-edge lift {lift:+.3f}pp < required {min_lift:.2f}pp")

    drawdown_ok = metrics.drawdown_pct <= max_dd
    if not drawdown_ok:
        reasons.append(f"drawdown {metrics.drawdown_pct:.2f}% > cap {max_dd:.2f}%")

    if metrics.per_regime_net_edge_pct:
        passing = sum(1 for v in metrics.per_regime_net_edge_pct.values() if v > 0)
        share = passing / len(metrics.per_regime_net_edge_pct)
        regime_robustness_ok = share >= min_passing_share
        if not regime_robustness_ok:
            reasons.append(
                f"only {passing}/{len(metrics.per_regime_net_edge_pct)} regimes net-positive"
                f" ({share:.0%} < required {min_passing_share:.0%})"
            )
    else:
        # No per-regime breakdown — degrade safely (treat as not-yet-evaluable).
        regime_robustness_ok = False
        reasons.append("no per-regime breakdown available")

    eligible = all([samples_ok, edge_lift_ok, drawdown_ok, regime_robustness_ok])
    return PromotionVerdict(
        eligible=eligible,
        samples_ok=samples_ok,
        edge_lift_ok=edge_lift_ok,
        drawdown_ok=drawdown_ok,
        regime_robustness_ok=regime_robustness_ok,
        reasons=reasons,
        thresholds={
            "min_shadow_samples": min_samples,
            "min_net_edge_lift_vs_champion": min_lift,
            "max_drawdown_pct": max_dd,
            "min_passing_regimes_share": min_passing_share,
        },
        metrics_summary={
            "samples": metrics.samples,
            "net_edge_pct": metrics.net_edge_pct,
            "champion_net_edge_pct": metrics.champion_net_edge_pct,
            "edge_lift_pct": lift,
            "drawdown_pct": metrics.drawdown_pct,
            "per_regime_net_edge_pct": dict(metrics.per_regime_net_edge_pct),
        },
    )


# ---------------------------------------------------------------------------
# Task #654 — paper-trading promotion (manual operator action).
# ---------------------------------------------------------------------------


class PromotionError(Exception):
    """Raised by `promote_shadow_to_serving` when the requested row can't
    be safely promoted (missing, wrong state, manifest fails to load,
    DB error). Distinct exception type so callers can surface a clean
    error message instead of a generic 500.
    """


@dataclass
class PromotionResult:
    """Outcome of `promote_shadow_to_serving`.

    `promoted_id`         the model_registry row id now in `champion` state
    `previous_champion_id`  the row that was demoted to `shadow` (None when
                          there was no incumbent for the slot)
    `model_id` / `model_version` / `coin_id` / `timeframe`
                          identifiers of the promoted slot — pass-through
                          from the row so the caller can log without
                          re-querying.
    `scope_constraint`    the scope payload stamped on the new champion.
    `promoted_at`         UTC timestamp the promotion landed.
    `promoted_by`         operator-supplied identifier (free-form string).
    """
    promoted_id: int
    previous_champion_id: Optional[int]
    model_id: str
    model_version: str
    coin_id: str
    timeframe: str
    scope_constraint: dict
    promoted_at: datetime
    promoted_by: str


async def promote_shadow_to_serving(
    model_registry_id: int,
    *,
    scope_constraint: dict,
    promoted_by: str,
    note: Optional[str] = None,
    pool=None,
    load_model_fn=None,
) -> PromotionResult:
    """Atomically promote a `shadow` row to `champion`.

    Workflow (single transaction):
      1. SELECT … FOR UPDATE the row by `model_registry_id`. Refuse if
         missing or `state != 'shadow'`.
      2. Validate the on-disk manifest LOADS (so we never hand live
         traffic a slice that's missing files / has an invalid shape).
         The check uses `app.training.registry.load_model(coin_id,
         timeframe, model_version)` and short-circuits when the load
         returns `None`.
      3. Demote any other active champion for the SAME (model_id,
         coin_id, timeframe) tuple to `shadow` (clears the
         active-champion unique index so the new row can take it).
         Records the demoted row id as `previous_champion_id` on the
         promoted row.
      4. UPDATE the target row to `state='champion'`, `is_active=true`,
         `promoted_at=now`, stamps `scope_constraint`, `note`, and a
         marker in `metrics_snapshot.promoted_by` so the audit trail
         survives without a separate column.

    Strictly NO model promotion happens unless ALL steps succeed; on
    any failure the transaction is rolled back. Returns a
    `PromotionResult` so the caller can log / surface the new id.

    `pool` and `load_model_fn` are dependency-injection hooks for unit
    tests so a fake asyncpg pool + an in-memory manifest validator can
    drive the function without touching real Postgres or disk. Callers
    in production omit them and the function uses
    `app.db.init_pool()` and `app.training.registry.load_model`.
    """
    if not isinstance(scope_constraint, dict):
        raise PromotionError(
            f"scope_constraint must be a dict, got {type(scope_constraint).__name__}"
        )
    if not isinstance(promoted_by, str) or not promoted_by.strip():
        raise PromotionError("promoted_by must be a non-empty string")

    if pool is None:
        from .db import init_pool  # local import to keep registry_lifecycle
        # importable without DATABASE_URL (the module is also used by the
        # pure `evaluate_promotion` path which has no DB dependency).

        pool = await init_pool()

    if load_model_fn is None:
        from .training.registry import load_model as _load_model

        load_model_fn = _load_model

    promoted_at = datetime.now(timezone.utc)
    note_value = note or (
        f"Promoted shadow -> champion by {promoted_by} at "
        f"{promoted_at.isoformat()} (paper-trading scope: "
        f"{_json.dumps(scope_constraint, sort_keys=True)})"
    )

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id, model_id, model_version, coin_id, timeframe,
                       state, metrics_snapshot
                FROM model_registry
                WHERE id = $1
                FOR UPDATE
                """,
                model_registry_id,
            )
            if row is None:
                raise PromotionError(
                    f"model_registry row id={model_registry_id} not found"
                )
            if row["state"] != "shadow":
                raise PromotionError(
                    f"refusing to promote model_registry row id={model_registry_id} "
                    f"in state={row['state']!r} (expected 'shadow')"
                )

            # Validate the manifest LOADS before flipping any state. If
            # the on-disk artefact is missing or invalid, abort the
            # transaction so the row stays as-is and live traffic never
            # sees a half-broken champion.
            try:
                loaded = load_model_fn(
                    row["coin_id"], row["timeframe"], row["model_version"],
                )
            except Exception as exc:  # noqa: BLE001
                raise PromotionError(
                    f"manifest load failed for "
                    f"{row['coin_id']}/{row['timeframe']}/{row['model_version']}: "
                    f"{exc}"
                ) from exc
            if loaded is None:
                raise PromotionError(
                    f"manifest could not be loaded for "
                    f"{row['coin_id']}/{row['timeframe']}/{row['model_version']} "
                    "(load_model returned None)"
                )

            # Demote the existing champion (if any) for the SAME
            # (model_id, coin_id, timeframe, label_family) slot. The
            # active-champion unique index forbids two such rows on
            # (model_id, coin_id, timeframe); we move the incumbent
            # back to 'shadow' so a future rollback can re-promote it
            # without forcing the operator to re-train.
            #
            # `label_family` lives INSIDE `scope_constraint` (per the
            # schema comment: "{ scope, coins, timeframes, label_family,
            # expires_at }"), NOT as its own column. We therefore:
            #   - require the new scope_constraint to declare a
            #     `label_family` (None means "legacy / unspecified")
            #   - inspect each candidate champion's stored
            #     `scope_constraint.label_family` and only demote when
            #     it MATCHES the new one
            # This stops a Family-A promotion from accidentally evicting
            # a Family-C champion that happens to share (model_id,
            # coin_id, timeframe). When neither side declares a family
            # (legacy 3-class champion + legacy 3-class promotion) we
            # treat the slot as a single legacy bucket and demote — that
            # preserves today's "one champion per slot" behaviour.
            new_label_family = scope_constraint.get("label_family")
            candidate_champions = await conn.fetch(
                """
                SELECT id, scope_constraint FROM model_registry
                WHERE model_id = $1 AND coin_id = $2 AND timeframe = $3
                  AND state = 'champion' AND is_active = true
                  AND id <> $4
                """,
                row["model_id"], row["coin_id"], row["timeframe"],
                model_registry_id,
            )
            previous_champion_id: Optional[int] = None
            for cand in candidate_champions:
                cand_scope = cand["scope_constraint"]
                if isinstance(cand_scope, str):
                    try:
                        cand_scope = _json.loads(cand_scope)
                    except Exception:  # noqa: BLE001
                        cand_scope = None
                cand_family = (
                    cand_scope.get("label_family")
                    if isinstance(cand_scope, dict) else None
                )
                if cand_family == new_label_family:
                    previous_champion_id = int(cand["id"])
                    break
            if previous_champion_id is not None:
                await conn.execute(
                    """
                    UPDATE model_registry
                    SET state = 'shadow',
                        demoted_at = $2,
                        updated_at = $2
                    WHERE id = $1
                    """,
                    previous_champion_id, promoted_at,
                )

            # Stamp `promoted_by` into metrics_snapshot so an audit can
            # answer "who promoted this slot?" without needing a new
            # column. Existing snapshot keys (the gate verdict written
            # by `register_shadow`) are preserved.
            existing_snapshot = row["metrics_snapshot"] or {}
            if isinstance(existing_snapshot, str):
                try:
                    existing_snapshot = _json.loads(existing_snapshot)
                except Exception:  # noqa: BLE001
                    existing_snapshot = {}
            if not isinstance(existing_snapshot, dict):
                existing_snapshot = {}
            existing_snapshot = {
                **existing_snapshot,
                "promoted_by": promoted_by,
                "promoted_at": promoted_at.isoformat(),
                "previous_champion_id": previous_champion_id,
            }

            await conn.execute(
                """
                UPDATE model_registry
                SET state = 'champion',
                    is_active = true,
                    promoted_at = $2,
                    previous_champion_id = $3,
                    note = $4,
                    metrics_snapshot = $5::jsonb,
                    scope_constraint = $6::jsonb,
                    updated_at = $2
                WHERE id = $1
                """,
                model_registry_id,
                promoted_at,
                previous_champion_id,
                note_value,
                _json.dumps(existing_snapshot, default=str),
                _json.dumps(scope_constraint, default=str),
            )

    _logger.info(
        "promote_shadow_to_serving",
        extra={
            "modelRegistryId": model_registry_id,
            "modelId": row["model_id"],
            "modelVersion": row["model_version"],
            "coinId": row["coin_id"],
            "timeframe": row["timeframe"],
            "previousChampionId": previous_champion_id,
            "promotedBy": promoted_by,
            "scopeConstraint": scope_constraint,
        },
    )
    return PromotionResult(
        promoted_id=int(model_registry_id),
        previous_champion_id=(
            int(previous_champion_id) if previous_champion_id is not None else None
        ),
        model_id=str(row["model_id"]),
        model_version=str(row["model_version"]),
        coin_id=str(row["coin_id"]),
        timeframe=str(row["timeframe"]),
        scope_constraint=dict(scope_constraint),
        promoted_at=promoted_at,
        promoted_by=promoted_by,
    )
