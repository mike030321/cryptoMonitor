"""Phase 4 — meta-model training dataset.

Builds one row per resolved prediction_journal entry:
  features = (per-specialist probUp/probDown/expectedReturnPct, regime
              one-hot, base prob_up/prob_down, base expected_return_pct,
              raw_confidence, prediction_std_pct, atr-derived volatility
              proxy from feature_vector)
  labels   = (action_label, realized_edge_after_costs)

`action_label` is one of {"long", "short", "no_trade"}:
  - "long"     when realized_return_pct > round_trip_cost_pct
  - "short"    when realized_return_pct < -round_trip_cost_pct
  - "no_trade" otherwise (the spec's first-class abstain target).

`realized_edge_after_costs` is the signed return minus the round-trip
cost in the predicted direction (so the regressor learns the magnitude
of the edge a long/short would have captured, net of fees+slippage).

Read-only: never mutates the journal. Falls back to an empty DataFrame
when the journal is empty so the trainer can still run on day-1 and
deploy a heuristic meta-model (see train_meta.deploy_heuristic).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .. import db as db_mod

logger = logging.getLogger(__name__)

# Round-trip cost — taker fee + slippage, both ways. Mirrors the
# api-server's ROUND_TRIP_COST_PCT (0.003). Hardcoded here because
# pulling shared/trading-frictions.json requires a path import dance
# the trainer doesn't need; if it ever drifts the live trader logs
# the discrepancy via /api/crypto/quant-coverage anyway.
ROUND_TRIP_COST_PCT = 0.003  # 0.3% round trip


META_FEATURE_COLUMNS: list[str] = [
    "base_prob_up",
    "base_prob_down",
    "base_prob_stable",
    "base_expected_return_pct",
    "base_raw_confidence",
    "base_prediction_std_pct",
    # Per-specialist features (aligned to SPECIALIST_KINDS order). When a
    # specialist doesn't apply for the live regime, its values are 0 and
    # the `*_applicable` indicator tells the model so.
    "spec_momentum_prob_up",
    "spec_momentum_prob_down",
    "spec_momentum_exp_ret",
    "spec_momentum_applicable",
    "spec_mean_reversion_prob_up",
    "spec_mean_reversion_prob_down",
    "spec_mean_reversion_exp_ret",
    "spec_mean_reversion_applicable",
    "spec_breakout_prob_up",
    "spec_breakout_prob_down",
    "spec_breakout_exp_ret",
    "spec_breakout_applicable",
    "spec_vol_forecaster_exp_ret",
    "spec_vol_forecaster_applicable",
    # Regime one-hot (Phase 2 vocabulary). Compact and stable.
    "regime_trending_up",
    "regime_trending_down",
    "regime_range_chop",
    "regime_high_vol_breakout",
    "regime_low_vol_compression",
    "regime_panic_liquidation",
    # Disagreement features — the meta-model's whole reason to exist.
    "specialist_dir_agreement",   # 1.0 if all applicable specialists agree
    "specialist_count_applicable",
    # Reliability features — trailing 30d hit-rate context per coin and
    # per regime (computed STRICTLY before each row's timestamp so there
    # is no leakage). These are what make the meta-model adaptive: the
    # same specialist scores can mean "trade" in a regime/coin where the
    # base model has been right lately and "abstain" in one where it
    # hasn't. See meta_reliability.attach_reliability_to_dataset.
    "reliability_coin_winrate_30d",
    "reliability_regime_winrate_30d",
    "reliability_coin_n_30d",
    "reliability_regime_n_30d",
]

REGIME_VOCAB = [
    "trending_up", "trending_down", "range_chop",
    "high_vol_breakout", "low_vol_compression", "panic_liquidation",
]


def _row_to_features(row: dict) -> Optional[dict]:
    """Project a prediction_journal + paper_trades JOIN row into the
    meta-model feature dict + label. Returns None when the row is too
    incomplete to learn from."""
    if row.get("realized_return_pct") is None:
        return None
    realized_pct = float(row["realized_return_pct"])
    cost_pct = ROUND_TRIP_COST_PCT * 100.0  # convert fraction → percent

    # Action label — what the meta-model should have output to maximize
    # net realized return on this bar.
    if realized_pct > cost_pct:
        action = "long"
    elif realized_pct < -cost_pct:
        action = "short"
    else:
        action = "no_trade"

    # Edge after costs (signed). For the regressor head: how much net
    # return would a perfectly-directional bet have captured.
    if realized_pct > 0:
        edge_after_costs = realized_pct - cost_pct
    elif realized_pct < 0:
        edge_after_costs = realized_pct + cost_pct  # short captures abs(realized) - cost
    else:
        edge_after_costs = 0.0

    gates = row.get("gates_applied") or {}
    if isinstance(gates, str):
        # asyncpg returns jsonb as already-decoded but be defensive
        try:
            import json as _json
            gates = _json.loads(gates)
        except Exception:
            gates = {}
    specialists = gates.get("specialists") if isinstance(gates, dict) else None
    if not isinstance(specialists, list):
        specialists = []

    by_kind: dict[str, dict] = {}
    for sp in specialists:
        if isinstance(sp, dict) and isinstance(sp.get("kind"), str):
            by_kind[sp["kind"]] = sp

    def _sp(kind: str, key: str, default: float = 0.0) -> float:
        sp = by_kind.get(kind)
        if sp is None:
            return default
        v = sp.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _sp_applicable(kind: str) -> float:
        sp = by_kind.get(kind)
        return 1.0 if sp and sp.get("applicable") else 0.0

    # Specialist directional agreement — fraction of applicable specialists
    # whose argmax(prob_up, prob_down) matches the base direction.
    base_dir = "up" if (row.get("prob_up") or 0) >= (row.get("prob_down") or 0) else "down"
    applicable = [sp for sp in specialists if isinstance(sp, dict) and sp.get("applicable")]
    if applicable:
        agree = 0
        for sp in applicable:
            sp_up = sp.get("probUp") or 0
            sp_dn = sp.get("probDown") or 0
            sp_dir = "up" if sp_up >= sp_dn else "down"
            if sp_dir == base_dir:
                agree += 1
        agreement = agree / len(applicable)
    else:
        agreement = 0.0

    regime = row.get("regime_label") or ""
    feat = {
        "base_prob_up": float(row.get("prob_up") or 0),
        "base_prob_down": float(row.get("prob_down") or 0),
        "base_prob_stable": float(row.get("prob_stable") or 0),
        "base_expected_return_pct": float(row.get("expected_return_pct") or 0),
        "base_raw_confidence": float(row.get("raw_confidence") or 0),
        "base_prediction_std_pct": float(row.get("prediction_std_pct") or 0),
        "spec_momentum_prob_up": _sp("momentum", "probUp"),
        "spec_momentum_prob_down": _sp("momentum", "probDown"),
        "spec_momentum_exp_ret": _sp("momentum", "expectedReturnPct"),
        "spec_momentum_applicable": _sp_applicable("momentum"),
        "spec_mean_reversion_prob_up": _sp("mean_reversion", "probUp"),
        "spec_mean_reversion_prob_down": _sp("mean_reversion", "probDown"),
        "spec_mean_reversion_exp_ret": _sp("mean_reversion", "expectedReturnPct"),
        "spec_mean_reversion_applicable": _sp_applicable("mean_reversion"),
        "spec_breakout_prob_up": _sp("breakout", "probUp"),
        "spec_breakout_prob_down": _sp("breakout", "probDown"),
        "spec_breakout_exp_ret": _sp("breakout", "expectedReturnPct"),
        "spec_breakout_applicable": _sp_applicable("breakout"),
        "spec_vol_forecaster_exp_ret": _sp("volatility_forecaster", "expectedReturnPct"),
        "spec_vol_forecaster_applicable": _sp_applicable("volatility_forecaster"),
        "specialist_dir_agreement": float(agreement),
        "specialist_count_applicable": float(len(applicable)),
    }
    for r in REGIME_VOCAB:
        feat[f"regime_{r}"] = 1.0 if regime == r else 0.0

    feat["__action__"] = action
    feat["__edge_after_costs__"] = float(edge_after_costs)
    feat["__timeframe__"] = row.get("timeframe")
    feat["__coin_id__"] = row.get("coin_id")
    feat["__created_at__"] = row.get("created_at")
    return feat


_QUERY = """
SELECT
    pj.id,
    pj.created_at,
    pj.coin_id,
    pj.timeframe,
    pj.prob_up,
    pj.prob_down,
    pj.prob_stable,
    pj.expected_return_pct,
    pj.prediction_std_pct,
    pj.raw_confidence,
    pj.regime_label,
    pj.gates_applied,
    pj.realized_return_pct
FROM prediction_journal pj
WHERE pj.brain = 'QUANT'
  AND pj.realized_return_pct IS NOT NULL
  AND pj.gates_applied IS NOT NULL
ORDER BY pj.created_at ASC
LIMIT $1
"""


async def build_meta_dataset(timeframe: Optional[str] = None, limit: int = 50_000) -> pd.DataFrame:
    """Build the meta-model training frame from prediction_journal.

    Returns an empty DataFrame (with the canonical columns) when the
    journal has no resolved QUANT rows yet — the trainer special-cases
    that and deploys a heuristic meta-model.
    """
    pool = await db_mod.init_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(_QUERY, limit)
    feats: list[dict] = []
    for r in rows:
        d = dict(r)
        if timeframe and d.get("timeframe") != timeframe:
            continue
        feat = _row_to_features(d)
        if feat is None:
            continue
        feats.append(feat)
    if not feats:
        cols = META_FEATURE_COLUMNS + [
            "__action__", "__edge_after_costs__",
            "__timeframe__", "__coin_id__", "__created_at__",
        ]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(feats)
    # Phase 4 — attach trailing-30d reliability features (no leakage).
    from .meta_reliability import attach_reliability_to_dataset
    df = attach_reliability_to_dataset(df)
    return df
