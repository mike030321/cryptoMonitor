"""ML Engine — Phase 2 (multiclass LightGBM, per-coin registry, /report).

Wires:
  GET  /ml/health   liveness
  POST /ml/features Phase-1 feature vector for a (coin, timeframe)
  POST /ml/predict  Phase-2 LightGBM 3-class probability + EV from latest model
                    (resolves per-coin model first, falls back to pooled)
  GET  /ml/report   HTML view of the latest training run (no auth, read-only)
  POST /ml/admin/reload  cache invalidation, X-Admin-Token gated

Trading path is still untouched: no production Node code calls /ml/predict
yet. Wiring happens in Phase 4 (shadow mode) and Phase 5 (cutover).
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import asyncio
import threading
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .db import (
    close_pool,
    fetch_quarantined_versions,
    fetch_real_ticks,
    fetch_scope_for_active_champion,
    init_pool,
)
from .features import (
    MIN_CANDLES_FOR_FEATURES,
    TIMEFRAME_MS,
    build_feature_vector,
    feature_hash,
    resample_to_candles,
)
from .regime import (
    REGIME_LABELS,
    classify_regime_from_basket,
    classify_regime_from_features,
)
from .logging_config import configure_logging, logger
from .training.registry import (
    SPECIALIST_KINDS,
    SPECIALIST_REGIME_MAP,
    specialist_coin_id,
)
from .training.registry import (
    FEATURE_COLUMNS,
    LoadedDualHeadModel,
    LoadedModel,
    POOLED_COIN_ID,
    REGISTRY_ROOT,
    latest_version,
    list_versions,
    load_model,
    read_verification_verdict,
    resolve_model,
)
from .training.report import render_html as render_training_html
from . import meta_brain

LOOKBACK_MULTIPLE = {
    "1m": 60,
    "5m": 60,
    "1h": 60,
    "2h": 60,
    "6h": 40,
    "1d": 35,
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    try:
        await init_pool()
    except Exception as exc:  # noqa: BLE001
        logger.warning("db_pool_init_failed", extra={"error": str(exc)})
    _start_auto_retrain_scheduler()
    _start_fast_loop_scheduler()
    # Task #410 — daily 5m head top-up so the 305-day contiguous gate
    # doesn't silently erode between operator-launched campaigns.
    try:
        from . import scheduled_5m_topup
        scheduled_5m_topup.start_scheduler()
    except Exception as exc:  # noqa: BLE001
        logger.warning("topup_5m_start_failed", extra={"error": str(exc)})
    # Task #603 — weekly deep-tail 5m backfill so the historical bar
    # gets re-extended on a 7-day cadence (not just the daily HEAD
    # top-up). The script holds an advisory lock distinct from the
    # daily top-up's, so the two daemons never argue. Lives in-process
    # because the project sits at the 10-workflow ceiling — see the
    # module docstring for the rationale.
    try:
        from . import weekly_5m_tail_backfill
        weekly_5m_tail_backfill.start_scheduler()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "weekly_5m_tail_backfill_start_failed",
            extra={"error": str(exc)},
        )
    # Task #373 — Market Meta-Brain supervisory layer. Pure-stdlib
    # vendored package; never predicts price, never places trades.
    try:
        meta_brain.init()
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta_brain_init_failed", extra={"error": str(exc)})
    # Task #490 — nightly auto-warm of the supervisory brain from real
    # post-#444 trading data. Mirrors the 5m-topup daemon's lifecycle:
    # daemon thread, multi-replica advisory lock, structured failure
    # log, never blocks the trading path.
    try:
        from . import scheduled_meta_brain_replay
        scheduled_meta_brain_replay.start_scheduler()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "meta_brain_replay_start_failed", extra={"error": str(exc)}
        )
    yield
    try:
        meta_brain.shutdown()
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta_brain_shutdown_failed", extra={"error": str(exc)})
    _stop_fast_loop_scheduler()
    _stop_auto_retrain_scheduler()
    try:
        from . import scheduled_5m_topup
        scheduled_5m_topup.stop_scheduler()
    except Exception as exc:  # noqa: BLE001
        logger.warning("topup_5m_stop_failed", extra={"error": str(exc)})
    try:
        from . import scheduled_meta_brain_replay
        scheduled_meta_brain_replay.stop_scheduler()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "meta_brain_replay_stop_failed", extra={"error": str(exc)}
        )
    await close_pool()


app = FastAPI(title="ML Engine", lifespan=lifespan)

# Mount the Meta-Brain router under /ml/meta-brain/* so it lives next to
# the other /ml endpoints the api-server talks to.
app.include_router(meta_brain.router, prefix="/ml")


class FeatureRequest(BaseModel):
    coinId: str
    timeframe: str = Field(..., description="One of: 1m, 5m, 1h, 2h, 6h, 1d")


class FeatureResponse(BaseModel):
    coinId: str
    timeframe: str
    candleCount: int
    features: Optional[dict[str, float]]
    insufficientData: bool
    featureHash: Optional[str]
    durationMs: float


class FeatureImportance(BaseModel):
    feature: str
    gainPct: float


class SpecialistPrediction(BaseModel):
    """Phase 3 — single specialist head's view of the current bar.

    Persisted into the prediction journal (in `gates_applied` JSON) so the
    diagnostics page can compute per-specialist per-regime accuracy
    without re-running inference. `applicable` is true when the live
    regime falls in this specialist's `regime_subset` (or the specialist
    is the volatility forecaster, which spans all regimes).
    """
    kind: str                              # momentum | mean_reversion | breakout | volatility_forecaster
    modelVersion: str
    modelCoinId: str
    regimeSubset: list[str]
    applicable: bool
    probUp: Optional[float] = None
    probDown: Optional[float] = None
    probStable: Optional[float] = None
    expectedReturnPct: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    coinId: str
    timeframe: str
    probUp: float
    probDown: float
    probStable: float
    expectedReturnPct: float
    predictionStdPct: float
    confidence: float
    modelVersion: str
    modelCoinId: str               # which registry slot served this prediction
    featureImportanceTop5: list[FeatureImportance]
    source: str
    # Task #460 — stable short hash of the feature vector the model
    # actually saw at inference. The api-server's prediction journal
    # writer requires every QUANT row to carry one so downstream replay
    # can reconstruct exact inputs and de-duplicate "same bar, same
    # features" rows. Without this field on the response, the Node
    # client had no value to forward and journal-writer was silently
    # refusing every QUANT row from the freshly retrained 1d/6h
    # LightGBM models. lightgbm path: hash of the live `feats` dict.
    # prior path: deterministic `prior:{modelVersion}` since no feature
    # vector exists — still gives the journal a stable, traceable id.
    featureHash: Optional[str] = None
    # Phase 3 — optional specialist + regime context. The legacy 3-class
    # head above is still authoritative; these fields are additive so an
    # older client can ignore them and a Phase-4 meta-model can consume
    # them. `regime` is the live 6-class regime label for this bar.
    regime: Optional[str] = None
    specialists: list[SpecialistPrediction] = Field(default_factory=list)
    # Task #418 — attribution flag set when this response was served by
    # the pooled fallback for a coin whose per-coin slot was missing,
    # quarantined, or failed the verification gate (see
    # `_resolve_for_predict_with_verification` below). The dashboard
    # surfaces this as a "Pooled fallback" badge so operators don't
    # mistake the prediction for the per-coin head's view. None when
    # the response was served by the per-coin slot or the request
    # explicitly targeted the pooled slot.
    fallback: Optional[str] = None


def _validate_timeframe(tf: str) -> int:
    if tf not in TIMEFRAME_MS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown timeframe '{tf}', expected one of {list(TIMEFRAME_MS)}",
        )
    return TIMEFRAME_MS[tf]


@app.get("/ml/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "ml-engine", "phase": "2-lightgbm"}


@app.get("/ml/threshold-fallback-status")
async def threshold_fallback_status() -> dict:
    """Surface `LABEL_THRESHOLDS_FALLBACK_STATUS` so the dashboard can warn
    operators when a training run silently fell back to the hardcoded label
    threshold mirror because `shared/trading-frictions.json` was unreachable.
    See task #357 for the underlying flag and task #358 for the dashboard
    promotion.
    """
    from .training.labels import LABEL_THRESHOLDS_FALLBACK_STATUS

    return {
        "used_fallback": bool(LABEL_THRESHOLDS_FALLBACK_STATUS.get("used_fallback")),
        "reason": LABEL_THRESHOLDS_FALLBACK_STATUS.get("reason"),
        "path_tried": LABEL_THRESHOLDS_FALLBACK_STATUS.get("path_tried"),
    }


# --- Phase 4: meta-model serving --------------------------------------------
# The meta-model is a separate registry slot (`__meta__/{tf}/{version}/`).
# When the LightGBM-trained variant exists we use it; otherwise the manifest
# is `meta_kind == "heuristic"` and we apply a deterministic specialist-
# agreement + cost-cushion rule. Either way the response shape is identical
# so api-server's quant-brain can wire the meta-model from day-1, before the
# journal has accumulated enough resolved rows to fit a real classifier.
class MetaPredictRequest(BaseModel):
    coinId: str
    timeframe: str
    # Caller passes the raw base + specialist outputs. We do NOT re-run
    # /ml/predict here because the live trader already has them and the
    # round-trip would double the inference budget. A field that's missing
    # is treated as 0 — the heuristic and the trained model both tolerate it.
    base: dict
    regime: Optional[str] = None
    specialists: list[dict] = Field(default_factory=list)


class MetaPredictResponse(BaseModel):
    timeframe: str
    action: str            # "long" | "short" | "no_trade"
    sizeMultiplier: float  # 0.0 when action == no_trade; otherwise (0, 1.5]
    expectedEdgePct: float # net of round-trip cost, signed by action
    abstainReason: Optional[str] = None
    metaKind: str          # "lightgbm" | "heuristic"
    metaVersion: str
    actionProbs: dict      # {"no_trade": p, "long": p, "short": p}


def _meta_features_from_request(req: MetaPredictRequest) -> dict:
    """Project a /ml/meta/predict request body into the canonical feature
    dict used by `meta_dataset.META_FEATURE_COLUMNS`. Missing fields → 0
    so the response shape is stable even on day-1 with no data."""
    from .training.meta_dataset import META_FEATURE_COLUMNS, REGIME_VOCAB

    base = req.base or {}
    by_kind: dict[str, dict] = {}
    for sp in req.specialists or []:
        if isinstance(sp, dict) and isinstance(sp.get("kind"), str):
            by_kind[sp["kind"]] = sp

    def _b(k: str) -> float:
        v = base.get(k)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _sp(kind: str, key: str) -> float:
        sp = by_kind.get(kind) or {}
        v = sp.get(key)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _ap(kind: str) -> float:
        sp = by_kind.get(kind)
        return 1.0 if sp and sp.get("applicable") else 0.0

    base_dir = "up" if _b("probUp") >= _b("probDown") else "down"
    applicable = [sp for sp in (req.specialists or []) if isinstance(sp, dict) and sp.get("applicable")]
    if applicable:
        agree = 0
        for sp in applicable:
            sp_dir = "up" if (sp.get("probUp") or 0) >= (sp.get("probDown") or 0) else "down"
            if sp_dir == base_dir:
                agree += 1
        agreement = agree / len(applicable)
    else:
        agreement = 0.0

    feat = {
        "base_prob_up": _b("probUp"),
        "base_prob_down": _b("probDown"),
        "base_prob_stable": _b("probStable"),
        "base_expected_return_pct": _b("expectedReturnPct"),
        "base_raw_confidence": _b("rawConfidence"),
        "base_prediction_std_pct": _b("predictionStdPct"),
        "spec_momentum_prob_up": _sp("momentum", "probUp"),
        "spec_momentum_prob_down": _sp("momentum", "probDown"),
        "spec_momentum_exp_ret": _sp("momentum", "expectedReturnPct"),
        "spec_momentum_applicable": _ap("momentum"),
        "spec_mean_reversion_prob_up": _sp("mean_reversion", "probUp"),
        "spec_mean_reversion_prob_down": _sp("mean_reversion", "probDown"),
        "spec_mean_reversion_exp_ret": _sp("mean_reversion", "expectedReturnPct"),
        "spec_mean_reversion_applicable": _ap("mean_reversion"),
        "spec_breakout_prob_up": _sp("breakout", "probUp"),
        "spec_breakout_prob_down": _sp("breakout", "probDown"),
        "spec_breakout_exp_ret": _sp("breakout", "expectedReturnPct"),
        "spec_breakout_applicable": _ap("breakout"),
        "spec_vol_forecaster_exp_ret": _sp("volatility_forecaster", "expectedReturnPct"),
        "spec_vol_forecaster_applicable": _ap("volatility_forecaster"),
        "specialist_dir_agreement": float(agreement),
        "specialist_count_applicable": float(len(applicable)),
    }
    for r in REGIME_VOCAB:
        feat[f"regime_{r}"] = 1.0 if (req.regime or "") == r else 0.0
    # Reliability features default to neutral (0.5 / n=0) here; the
    # /ml/meta/predict route fills them in with a journal lookup before
    # passing the dict to the model. Including them here keeps the
    # column ordering stable and makes the function safe to unit-test.
    from .training.meta_reliability import DEFAULT_RELIABILITY
    for k, v in DEFAULT_RELIABILITY.items():
        feat[k] = v
    # Re-order to match META_FEATURE_COLUMNS for the LightGBM call.
    return {k: feat.get(k, 0.0) for k in META_FEATURE_COLUMNS}


def _meta_heuristic(feat: dict) -> tuple[str, float, float, Optional[str], dict]:
    """Day-1 deterministic meta-model. Mirrors the safety-net in
    quant-brain.ts so the live system has consistent abstain behaviour
    before any meta training data exists.

    Rules (in order):
      1. specialist_count_applicable < 1 → no_trade (low_calibration)
      2. specialist_dir_agreement < 0.5 → no_trade (specialist_disagreement)
      3. |base_expected_return_pct| < 2 * cost_pct (= 0.6%) → no_trade (low_edge)
      4. base_prob of chosen side < 0.40 AND raw_confidence < 0.55 → no_trade (bad_regime_fit)
      5. otherwise long/short with sizeMultiplier scaled by edge cushion.
    """
    cost_pct = 0.3  # round-trip cost in percent
    base_dir_up = feat["base_prob_up"] >= feat["base_prob_down"]
    side = "long" if base_dir_up else "short"
    chosen_prob = feat["base_prob_up"] if base_dir_up else feat["base_prob_down"]
    edge = feat["base_expected_return_pct"]
    abs_edge = abs(edge)
    raw_conf = feat["base_raw_confidence"]
    agreement = feat["specialist_dir_agreement"]
    n_app = feat["specialist_count_applicable"]
    sign_aligned = (edge >= 0 and base_dir_up) or (edge <= 0 and not base_dir_up)
    expected_edge_signed = (abs_edge - cost_pct) if abs_edge > cost_pct else 0.0
    if not base_dir_up:
        expected_edge_signed = -expected_edge_signed

    # Reliability — when we have a meaningful sample (n>=20) and the
    # base model has been wrong more than 55% of the time on this
    # coin lately, treat it as low_calibration even if the specialists
    # agree right now.
    coin_wr = feat.get("reliability_coin_winrate_30d", 0.5)
    coin_n = feat.get("reliability_coin_n_30d", 0.0)
    if n_app < 1:
        return "no_trade", 0.0, 0.0, "low_calibration", {"no_trade": 0.7, "long": 0.15, "short": 0.15}
    if coin_n >= 20 and coin_wr < 0.45:
        return "no_trade", 0.0, 0.0, "low_calibration", {"no_trade": 0.7, "long": 0.15, "short": 0.15}
    if agreement < 0.5:
        return "no_trade", 0.0, 0.0, "specialist_disagreement", {"no_trade": 0.7, "long": 0.15, "short": 0.15}
    if abs_edge < 2 * cost_pct or not sign_aligned:
        return "no_trade", 0.0, expected_edge_signed, "low_edge", {"no_trade": 0.6, "long": 0.2, "short": 0.2}
    if chosen_prob < 0.40 and raw_conf < 0.55:
        return "no_trade", 0.0, expected_edge_signed, "bad_regime_fit", {"no_trade": 0.6, "long": 0.2, "short": 0.2}
    # Size multiplier: scales linearly with cushion above 2*cost up to 4*cost.
    cushion = max(0.0, abs_edge - 2 * cost_pct)
    size_mult = max(0.5, min(1.5, 0.5 + cushion / (2 * cost_pct)))
    probs = ({"no_trade": 0.2, "long": 0.7, "short": 0.1} if base_dir_up
             else {"no_trade": 0.2, "long": 0.1, "short": 0.7})
    return side, size_mult, expected_edge_signed, None, probs


_ABSTAIN_REASON_MAP = {
    "low_edge": "low_edge",
    "specialist_disagreement": "specialist_disagreement",
    "low_calibration": "low_calibration",
    "bad_regime_fit": "bad_regime_fit",
}


@app.post("/ml/meta/predict", response_model=MetaPredictResponse)
async def meta_predict(req: MetaPredictRequest) -> MetaPredictResponse:
    from .training import train_meta as meta_mod
    from .training.meta_dataset import META_FEATURE_COLUMNS

    from .training.meta_reliability import compute_reliability_features

    manifest, clf, reg = meta_mod.load_meta_models(req.timeframe)
    feat = _meta_features_from_request(req)
    # Phase 4 — overwrite the neutral defaults with the live trailing-30d
    # reliability lookup so the meta-model (or the heuristic) sees the
    # same per-(coin,regime) signal it was trained on. Failures are
    # swallowed by `compute_reliability_features` itself.
    reliability = await compute_reliability_features(
        timeframe=req.timeframe, coin_id=req.coinId, regime=req.regime,
    )
    for k, v in reliability.items():
        feat[k] = v

    if manifest is None:
        # No /ml/meta has been deployed yet — apply heuristic so the live
        # quant-brain still gets a sensible answer. This matches the
        # day-1 contract the trainer eventually replaces with `lightgbm`.
        action, size_mult, edge, reason, probs = _meta_heuristic(feat)
        return MetaPredictResponse(
            timeframe=req.timeframe,
            action=action,
            sizeMultiplier=size_mult,
            expectedEdgePct=edge,
            abstainReason=reason,
            metaKind="heuristic",
            metaVersion="bootstrap",
            actionProbs=probs,
        )

    if manifest.get("meta_kind") == "lightgbm" and clf is not None and reg is not None:
        x = np.asarray([list(feat.values())], dtype=float)
        try:
            proba = np.asarray(clf.predict(x)).flatten()
            edge_pred = float(np.asarray(reg.predict(x)).flatten()[0])
        except Exception as exc:  # noqa: BLE001 - fall back to heuristic on any inference fault
            logger.warning("meta_predict_lgb_failed", extra={"timeframe": req.timeframe, "error": str(exc)})
            action, size_mult, edge, reason, probs = _meta_heuristic(feat)
            return MetaPredictResponse(
                timeframe=req.timeframe, action=action, sizeMultiplier=size_mult,
                expectedEdgePct=edge, abstainReason=reason,
                metaKind="heuristic_fallback", metaVersion=str(manifest.get("version", "")),
                actionProbs=probs,
            )
        action_idx = int(proba.argmax())
        action = meta_mod.ACTION_LABELS[action_idx]
        action_prob = float(proba[action_idx])
        probs = {meta_mod.ACTION_LABELS[i]: float(proba[i]) for i in range(len(proba))}
        if action == "no_trade":
            # When the model abstains, derive the most-likely reason from
            # the same heuristic rules so operators can still see which
            # gate dominated. This preserves the enumerated-reason contract.
            _, _, _, fallback_reason, _ = _meta_heuristic(feat)
            reason = fallback_reason or "low_edge"
            return MetaPredictResponse(
                timeframe=req.timeframe, action="no_trade", sizeMultiplier=0.0,
                expectedEdgePct=edge_pred, abstainReason=reason,
                metaKind="lightgbm", metaVersion=str(manifest.get("version", "")),
                actionProbs=probs,
            )
        # size_multiplier is the action probability scaled into [0.5, 1.5]
        size_mult = max(0.5, min(1.5, 0.5 + action_prob))
        return MetaPredictResponse(
            timeframe=req.timeframe, action=action, sizeMultiplier=size_mult,
            expectedEdgePct=edge_pred, abstainReason=None,
            metaKind="lightgbm", metaVersion=str(manifest.get("version", "")),
            actionProbs=probs,
        )

    # Heuristic manifest deployed (insufficient training rows).
    action, size_mult, edge, reason, probs = _meta_heuristic(feat)
    return MetaPredictResponse(
        timeframe=req.timeframe, action=action, sizeMultiplier=size_mult,
        expectedEdgePct=edge, abstainReason=reason,
        metaKind="heuristic", metaVersion=str(manifest.get("version", "")),
        actionProbs=probs,
    )


class RegimeRequest(BaseModel):
    coinId: str
    timeframe: str = Field(..., description="One of: 1m, 5m, 1h, 2h, 6h, 1d")


class RegimeResponse(BaseModel):
    coinId: str
    timeframe: str
    label: str
    confidence: float
    inputs: dict
    insufficientData: bool
    candleCount: int
    durationMs: float


class BasketRegimeRequest(BaseModel):
    changes24h: list[float]
    crossCoinVol: Optional[float] = None


class BasketRegimeResponse(BaseModel):
    label: str
    confidence: float
    inputs: dict


@app.post("/ml/regime", response_model=RegimeResponse)
async def regime(req: RegimeRequest) -> RegimeResponse:
    """Per-coin per-timeframe 6-class regime label.

    Computed from the same shared feature vector built by /ml/features
    so live, training, and backtest all read the regime from the same
    code path.
    """
    started = time.perf_counter()
    bucket_ms = _validate_timeframe(req.timeframe)
    multiple = LOOKBACK_MULTIPLE.get(req.timeframe, 60)
    lookback_ms = bucket_ms * multiple

    ticks = await fetch_real_ticks(req.coinId, lookback_ms)
    candles = resample_to_candles(ticks, bucket_ms)
    vec = build_feature_vector(candles)
    duration_ms = (time.perf_counter() - started) * 1000

    if vec is None:
        return RegimeResponse(
            coinId=req.coinId, timeframe=req.timeframe,
            label="range_chop", confidence=0.0, inputs={},
            insufficientData=True, candleCount=len(candles),
            durationMs=round(duration_ms, 2),
        )
    decision = classify_regime_from_features(vec)
    return RegimeResponse(
        coinId=req.coinId, timeframe=req.timeframe,
        label=decision.label, confidence=decision.confidence,
        inputs=decision.inputs, insufficientData=False,
        candleCount=len(candles), durationMs=round(duration_ms, 2),
    )


@app.post("/ml/regime/basket", response_model=BasketRegimeResponse)
async def regime_basket(req: BasketRegimeRequest) -> BasketRegimeResponse:
    """Market-wide 6-class regime from a basket of per-coin 24h % changes."""
    decision = classify_regime_from_basket(req.changes24h, req.crossCoinVol)
    return BasketRegimeResponse(
        label=decision.label,
        confidence=decision.confidence,
        inputs=decision.inputs,
    )


@app.get("/ml/regime/labels")
async def regime_labels() -> dict:
    """Static list of valid regime labels — UI uses this to render legends."""
    return {"labels": list(REGIME_LABELS)}


@app.post("/ml/features", response_model=FeatureResponse)
async def features(req: FeatureRequest) -> FeatureResponse:
    started = time.perf_counter()
    bucket_ms = _validate_timeframe(req.timeframe)
    multiple = LOOKBACK_MULTIPLE.get(req.timeframe, 60)
    lookback_ms = bucket_ms * multiple

    ticks = await fetch_real_ticks(req.coinId, lookback_ms)
    candles = resample_to_candles(ticks, bucket_ms)
    vec = build_feature_vector(candles)
    duration_ms = (time.perf_counter() - started) * 1000

    fhash = feature_hash(vec) if vec is not None else None
    insufficient = vec is None
    logger.info(
        "features_computed",
        extra={
            "coinId": req.coinId, "timeframe": req.timeframe,
            "candleCount": len(candles), "tickCount": len(ticks),
            "insufficientData": insufficient, "featureHash": fhash,
            "durationMs": round(duration_ms, 2),
            "minCandles": MIN_CANDLES_FOR_FEATURES,
        },
    )
    return FeatureResponse(
        coinId=req.coinId, timeframe=req.timeframe,
        candleCount=len(candles), features=vec,
        insufficientData=insufficient, featureHash=fhash,
        durationMs=round(duration_ms, 2),
    )


# --- Predict ---------------------------------------------------------------
@lru_cache(maxsize=64)
def _cached_load(coin_id: str, timeframe: str, version: str) -> LoadedModel:
    """Cache by (coin, timeframe, version) so a re-train invalidates by version bump."""
    m = load_model(coin_id, timeframe, version)
    if m is None:
        raise RuntimeError(f"failed to load model {coin_id}/{timeframe}/{version}")
    return m


def _resolve_specialist(kind: str, timeframe: str) -> Optional[LoadedModel]:
    """Phase 3 — load a per-regime specialist by kind. Returns None if
    that specialist hasn't been trained yet for this timeframe (e.g. the
    very first run before `train_specialists` ran). Cached by version so
    a re-train invalidates automatically.
    """
    slot = specialist_coin_id(kind)
    v = latest_version(slot, timeframe)
    if not v:
        return None
    try:
        return _cached_load(slot, timeframe, v)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "specialist_load_failed",
            extra={"kind": kind, "timeframe": timeframe, "version": v, "error": str(exc)},
        )
        return None


def _score_specialists(
    feats: dict, coin_id: str, timeframe: str, regime_label: str,
) -> list[SpecialistPrediction]:
    """Run every available specialist on the same feature vector. Returns
    one entry per specialist kind, even if the model isn't loaded yet —
    that way a downstream consumer always sees the full list shape and
    can spot "specialist not trained yet" without a separate catalog
    call. Failures are captured in the `error` field; they never bubble
    up and break /ml/predict.
    """
    out: list[SpecialistPrediction] = []
    for kind in SPECIALIST_KINDS:
        regime_subset = list(SPECIALIST_REGIME_MAP[kind])
        applicable = (kind == "volatility_forecaster") or (regime_label in regime_subset)
        m = _resolve_specialist(kind, timeframe)
        if m is None:
            out.append(SpecialistPrediction(
                kind=kind, modelVersion="", modelCoinId=specialist_coin_id(kind),
                regimeSubset=regime_subset, applicable=applicable,
                error="not_trained",
            ))
            continue
        try:
            X = _to_model_row(
                feats, coin_id, m.manifest.coin_vocab,
                feature_names=m.manifest.feature_names,
            )
            if m.manifest.model_kind == "prior":
                # Specialist fell back to a prior-only fit — emit it as an
                # honest no-opinion entry so the journal records the slot
                # exists without claiming a directional view.
                out.append(SpecialistPrediction(
                    kind=kind, modelVersion=m.manifest.version,
                    modelCoinId=m.manifest.coin_id,
                    regimeSubset=regime_subset, applicable=applicable,
                    probUp=0.0, probDown=0.0, probStable=1.0,
                    expectedReturnPct=0.0, confidence=0.0,
                ))
                continue
            probs = _calibrated_3class_probs(m, X)
            p_down, p_stable, p_up = float(probs[0]), float(probs[1]), float(probs[2])
            means_pct = list(m.manifest.class_return_means_pct)
            if len(means_pct) != 3:
                threshold_pct = m.manifest.threshold_pct
                means_pct = [-threshold_pct, 0.0, threshold_pct]
            class_mean_exp = float(
                p_down * means_pct[0] + p_stable * means_pct[1] + p_up * means_pct[2]
            )
            if m.regressor is not None:
                try:
                    reg_pred = m.regressor.predict(
                        X, num_iteration=m.regressor.best_iteration,
                    )
                    magnitude = abs(float(np.asarray(reg_pred).flatten()[0]))
                    sign = 1.0 if p_up >= p_down else -1.0
                    expected_return_pct = sign * magnitude
                except Exception:
                    expected_return_pct = class_mean_exp
            else:
                expected_return_pct = class_mean_exp
            confidence = float(max(0.0, (max(p_down, p_stable, p_up) - 1.0 / 3.0) * 1.5))
            out.append(SpecialistPrediction(
                kind=kind, modelVersion=m.manifest.version,
                modelCoinId=m.manifest.coin_id,
                regimeSubset=regime_subset, applicable=applicable,
                probUp=p_up, probDown=p_down, probStable=p_stable,
                expectedReturnPct=expected_return_pct,
                confidence=confidence,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "specialist_predict_failed",
                extra={"kind": kind, "timeframe": timeframe, "error": str(exc)},
            )
            out.append(SpecialistPrediction(
                kind=kind, modelVersion=m.manifest.version,
                modelCoinId=m.manifest.coin_id,
                regimeSubset=regime_subset, applicable=applicable,
                error=str(exc),
            ))
    return out


async def _resolve_for_predict(
    coin_id: str, timeframe: str,
) -> Optional[LoadedModel]:
    """Try per-coin first, then the pooled fallback. Cached by version.

    Task #232 — before picking the per-coin (or pooled) `latest` pointer
    we check the api-server's `model_registry` table for any row whose
    state == 'quarantined'. Those (coin, timeframe, version) slots are
    excluded from selection so the safety net (probability collapse,
    calibration drift, feature drift) actually keeps quarantined models
    out of live trade decisions. Falls back to the previous trained
    version (== previous champion) and then to the pooled slot, also
    skipping any quarantined pooled versions.
    """
    quarantined = await fetch_quarantined_versions()

    def _pick(slot: str) -> Optional[LoadedModel]:
        # Task #405 / B-PRED-500: the fast-path used to call
        # `_cached_load(...)` *unguarded*. When a manifest pointed to by
        # `latest` carries a forbidden-prefix feature (e.g. legacy
        # `news_*` columns), `load_model` returns `None`, `_cached_load`
        # raises `RuntimeError`, and the exception escapes past every
        # downstream fallback. Every coin/timeframe then 500s on
        # `/ml/predict`. Wrap the fast-path in try/except so a rejected
        # `latest` falls through to the descending-version walk and then
        # to the pooled fallback in `_resolve_for_predict`.
        latest = latest_version(slot, timeframe)
        if latest and (slot, timeframe, latest) not in quarantined:
            try:
                return _cached_load(slot, timeframe, latest)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "latest_load_failed",
                    extra={
                        "slot": slot, "timeframe": timeframe,
                        "version": latest, "error": str(exc),
                    },
                )
                # fall through to the descending walk below
        # Walk older versions newest-first; pick the first non-quarantined
        # one that still loads. This is the "fall back to the previous
        # champion" branch.
        for v in reversed(list_versions(slot, timeframe)):
            if (slot, timeframe, v) in quarantined:
                continue
            if v == latest:
                # already tried above; skip to avoid re-emitting the
                # same warning and re-raising the cached exception.
                continue
            try:
                return _cached_load(slot, timeframe, v)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "fallback_load_failed",
                    extra={
                        "slot": slot, "timeframe": timeframe,
                        "version": v, "error": str(exc),
                    },
                )
                continue
        return None

    m = _pick(coin_id)
    if m is not None:
        return m
    if coin_id != POOLED_COIN_ID:
        return _pick(POOLED_COIN_ID)
    return None


async def _resolve_for_predict_with_verification(
    coin_id: str, timeframe: str,
) -> tuple[Optional[LoadedModel], Optional[str]]:
    """Verification-aware wrapper around `_resolve_for_predict`.

    Task #418 — three structural promotion outcomes can land a per-coin
    slice on disk that the resolver still happily serves:

      1. Per-coin slice TRAINED but failed the verification gate (e.g.
         per-coin 1d slot whose holdout DA sits in the [0.500, 0.530]
         noise band raised by the tighter task #401 floor). The on-disk
         model loads fine, so `_resolve_for_predict` returns it even
         though the verification gate explicitly refused to promote
         that slice. Without this wrapper, the live trader keeps
         serving a slice we know to be in the noise band; the brief's
         third structural option is to fall back to the pooled head
         for that timeframe instead, IFF pooled was promoted.
      2. Per-coin slice missing or quarantined — `_resolve_for_predict`
         already falls back to pooled here; this wrapper just stamps
         `fallback="pooled"` so the response surfaces attribution.
      3. Pooled head also failed — keep serving the per-coin slice
         (current behaviour). Falling back to a pooled slice that
         itself sits in the noise band would gain us nothing and would
         silently change which model decides; we leave the noise-band
         per-coin slice in place rather than swap it for an unverified
         pool. The verification block already records both verdicts,
         so the operator can see exactly what's happening.

    Returns `(model, fallback)` where `fallback` is `"pooled"` when the
    response was served by the pooled slot for a non-pooled request
    (covers cases 1 and 2 above), and `None` otherwise.
    """
    primary = await _resolve_for_predict(coin_id, timeframe)
    if primary is None:
        return None, None
    # Caller asked for the pooled slot directly — never stamp `fallback`.
    if coin_id == POOLED_COIN_ID:
        return primary, None
    served_coin = primary.manifest.coin_id
    # Case 2: existing pooled fallback (per-coin missing / quarantined /
    # rejected for forbidden features). Mark as fallback so the dashboard
    # can render the same attribution the verification-driven fallback
    # below produces.
    if served_coin == POOLED_COIN_ID:
        return primary, "pooled"
    # Case 1: per-coin slice loaded; check its verification verdict.
    primary_verdict = read_verification_verdict(
        served_coin, timeframe, primary.manifest.version,
    )
    # No verdict on file — slice predates task #418's stamping or is a
    # specialist / meta slot the verification gate doesn't classify.
    # Preserve current behaviour: keep serving the per-coin model.
    if primary_verdict is None:
        return primary, None
    if primary_verdict.get("promoted"):
        return primary, None
    # Per-coin verdict is `promoted=False`. Try to fall back to a
    # promoted pooled head for the same timeframe. We deliberately do
    # NOT walk older per-coin versions here: if the *latest* per-coin
    # slice failed verification, an older one is presumed worse (it
    # was already replaced by this run). The pooled head is the
    # explicit safety net for this case.
    pooled = await _resolve_for_predict(POOLED_COIN_ID, timeframe)
    if pooled is None:
        return primary, None
    pooled_verdict = read_verification_verdict(
        POOLED_COIN_ID, timeframe, pooled.manifest.version,
    )
    if pooled_verdict is None or not pooled_verdict.get("promoted"):
        # Case 3: pooled head also unverified — keep the per-coin slice
        # rather than swap it for an unverified pool. The
        # verification block records both verdicts so an operator can
        # see exactly which slot is degraded.
        return primary, None
    logger.info(
        "predict_pooled_fallback_for_failed_verification",
        extra={
            "coinId": coin_id, "timeframe": timeframe,
            "perCoinReason": primary_verdict.get("reason"),
            "perCoinVersion": primary.manifest.version,
            "pooledVersion": pooled.manifest.version,
        },
    )
    return pooled, "pooled"


def _to_model_row(
    features_vec: dict[str, float],
    coin_id: str,
    vocab: list[str],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build the inference row.

    `feature_names` is the model's training-time schema (from manifest.json).
    When provided we project the live features onto that exact column list
    so a model trained before Phase 5 (no `news_*` columns) keeps working
    even after FEATURE_COLUMNS was extended with the news one-hot block.
    Missing columns default to 0.0; extra live columns are dropped.
    """
    cols = feature_names or FEATURE_COLUMNS
    coin_idx = vocab.index(coin_id) if coin_id in vocab else -1
    row = {col: features_vec.get(col, 0.0) for col in cols}
    if "coin_idx" in cols:
        row["coin_idx"] = coin_idx
    df = pd.DataFrame([row], columns=cols)
    if "coin_idx" in cols:
        df["coin_idx"] = df["coin_idx"].astype("int32")
    return df


def _top5_importance(booster, feature_names: list[str]) -> list[dict[str, float]]:
    try:
        gains = booster.feature_importance(importance_type="gain")
    except Exception:  # noqa: BLE001
        return []
    pairs = list(zip(feature_names, gains.tolist()))
    pairs.sort(key=lambda p: p[1], reverse=True)
    total = sum(g for _, g in pairs) or 1.0
    return [{"feature": n, "gainPct": float(g) / total * 100.0} for n, g in pairs[:5]]


def _calibrated_3class_probs(model: LoadedModel, X: pd.DataFrame) -> np.ndarray:
    """Run the booster and apply per-class isotonic calibration. Always
    returns a (3,) probability vector that sums to 1.
    """
    # Task #400 — when the slice's served predictor is the multinomial-
    # logistic baseline, there is no booster on disk; the served raw
    # prediction comes from the persisted (encoder, lr, priors) triple.
    # Per-class isotonic calibration was fit on baseline holdout
    # predictions during training (see `train_one_slice`) so the
    # downstream calibration step is identical.
    if (
        model.manifest.served_predictor_kind == "baseline"
        and model.baseline_artifact is not None
    ):
        from .training.train import _baseline_predict

        enc, lr, priors = model.baseline_artifact
        raw = _baseline_predict(enc, lr, priors, X)
    else:
        raw = model.booster.predict(X, num_iteration=model.booster.best_iteration)
    raw = np.atleast_2d(raw).astype(float)
    if raw.shape[1] != 3:
        # Defensive: shouldn't happen with a multiclass model, but if a stale
        # binary model is loaded we synthesize a 3-class vector and note the
        # source as `binary_legacy`.
        p = float(raw.flatten()[0])
        return np.array([1.0 - p, 0.0, p])
    if model.calibrators is not None:
        calibrated = np.zeros_like(raw)
        for k in range(3):
            cal = model.calibrators[k]
            if cal is None:
                calibrated[:, k] = raw[:, k]
            else:
                calibrated[:, k] = cal.predict(raw[:, k])
        s = calibrated.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        calibrated = calibrated / s
        return calibrated[0]
    s = raw.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (raw / s)[0]


def _scope_matches(scope: dict, coin_id: str, timeframe: str) -> bool:
    """Task #654 — return True iff `(coin_id, timeframe)` is permitted
    by `scope`. A scope key set to None / missing imposes no restriction
    along that axis. An empty list means "no values match" and the
    scope is treated as exclusionary (so an operator can't accidentally
    promote a model with `coins=[]` and have it serve everything).
    """
    coins = scope.get("coins")
    if isinstance(coins, list):
        if coin_id not in coins:
            return False
    timeframes = scope.get("timeframes")
    if isinstance(timeframes, list):
        if timeframe not in timeframes:
            return False
    return True


@app.post("/ml/predict", response_model=None)
async def predict(req: FeatureRequest):
    started = time.perf_counter()
    bucket_ms = _validate_timeframe(req.timeframe)

    model, fallback_kind = await _resolve_for_predict_with_verification(
        req.coinId, req.timeframe,
    )

    # Task #654 — paper-trading scope guard, evaluated PER RESOLVED
    # CHAMPION (not as a global allowlist). Once we know which row owns
    # the served prediction we look up its `scope_constraint`; if the
    # row has one and it doesn't admit (req.coinId, req.timeframe) we
    # short-circuit to `{"status": "out_of_scope", ...}` (HTTP 200) so
    # the caller can render a clean fence instead of mis-reading a 503
    # as a missing model. Champions with no scope (legacy 3-class) impose
    # no restriction. The check is best-effort — a DB blip returns
    # `None` from the helper and the request falls through to normal
    # serving — but the scoped row's scope IS authoritative when it
    # loads cleanly.
    if model is not None:
        resolved_scope = await fetch_scope_for_active_champion(
            model.manifest.coin_id, req.timeframe, model.manifest.version,
        )
        if resolved_scope is not None and not _scope_matches(
            resolved_scope, req.coinId, req.timeframe,
        ):
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "predict_out_of_scope",
                extra={
                    "coinId": req.coinId, "timeframe": req.timeframe,
                    "resolvedChampionCoinId": model.manifest.coin_id,
                    "resolvedChampionVersion": model.manifest.version,
                    "scopeConstraint": resolved_scope,
                    "durationMs": round(duration_ms, 2),
                },
            )
            return {
                "status": "out_of_scope",
                "coinId": req.coinId,
                "timeframe": req.timeframe,
                "scope_constraint": resolved_scope,
            }
    if model is None:
        # Phase 2 explicitly does not fabricate predictions for timeframes
        # that don't have a trained per-coin OR pooled model yet (e.g.
        # 6h/1d on current data).
        raise HTTPException(
            status_code=503,
            detail=(
                f"no model registered for coin '{req.coinId}' timeframe '{req.timeframe}' "
                "(per-coin and pooled both missing). "
                "Run `pnpm --filter @workspace/ml-engine train` first."
            ),
        )

    # Prior-only models bypass the feature pipeline — they were trained
    # precisely BECAUSE the timeframe lacks enough per-coin candles for the
    # MACD/RSI feature stack to compute. Honest fallback: emit a flat
    # STABLE call with zero confidence so the quant brain knows "we have
    # no opinion here" instead of speaking with the empirical class
    # frequencies as a directional call. The Laplace-smoothed empirical
    # priors stamped at training time are biased by the (short, currently
    # bullish-only) backfill window — argmax-ing them produced a stream
    # of "Quant: UP 4%" rows on 1d/6h/most-of-1h+2h that never traded
    # (downstream EV/cost gates filtered them) but actively misled the
    # dashboard. Source stays "prior" so analytics can still distinguish
    # trained-model emissions from fallback emissions; the raw smoothed
    # probs remain on the manifest for offline inspection.
    if model.manifest.model_kind == "prior":
        priors = list(model.manifest.prior_probs)
        if len(priors) != 3:
            raise HTTPException(
                status_code=500,
                detail=f"prior model {model.manifest.coin_id}/{model.manifest.timeframe}/"
                f"{model.manifest.version} has malformed prior_probs={priors}",
            )
        raw_p_down, raw_p_stable, raw_p_up = (
            float(priors[0]), float(priors[1]), float(priors[2]),
        )
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "predict_prior_stable_emission",
            extra={
                "coinId": req.coinId, "timeframe": req.timeframe,
                "modelCoinId": model.manifest.coin_id,
                "rawProbUp": raw_p_up,
                "rawProbDown": raw_p_down,
                "rawProbStable": raw_p_stable,
                "modelVersion": model.manifest.version,
                "durationMs": round(duration_ms, 2),
            },
        )
        return PredictResponse(
            coinId=req.coinId, timeframe=req.timeframe,
            probUp=0.0, probDown=0.0, probStable=1.0,
            expectedReturnPct=0.0,
            predictionStdPct=0.0,
            confidence=0.0,
            modelVersion=model.manifest.version,
            modelCoinId=model.manifest.coin_id,
            featureImportanceTop5=[],
            source="prior",
            # Task #460 — prior models bypass the feature pipeline entirely
            # (that's literally why they exist). There is no live feature
            # vector to hash, so we synthesize a deterministic, traceable
            # identifier from the model version. This still satisfies the
            # journal-writer's "every QUANT row has a feature_hash" contract
            # so prior-served abstain rows land in the journal instead of
            # being silently refused.
            featureHash=f"prior:{model.manifest.version}",
            regime=None,
            specialists=[],
            fallback=fallback_kind,
        )

    multiple = LOOKBACK_MULTIPLE.get(req.timeframe, 60)
    lookback_ms = bucket_ms * multiple
    ticks = await fetch_real_ticks(req.coinId, lookback_ms)
    candles = resample_to_candles(ticks, bucket_ms)
    # LLM isolation contract (Tasks #91/#255/#344): the LLM must NOT
    # influence trade decisions in any way, even when the sidecar is ON.
    # The Phase 5 news-tag one-hot block was an LLM input fed straight
    # into LightGBM at inference — that channel is now permanently shut.
    # We pass an empty list so the feature builder zero-fills the tag
    # columns; the column schema stays stable so legacy boosters trained
    # with these features still load (their tag splits become inert).
    # The `news_tags` table is still WRITTEN by the sidecar-gated news
    # classifier for dashboard display; it is just no longer READ here.
    news_tags: list[str] = []
    feats = build_feature_vector(candles, news_tags=news_tags)
    if feats is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"insufficient candles for {req.coinId}/{req.timeframe} "
                f"(have {len(candles)}, need >= {MIN_CANDLES_FOR_FEATURES})"
            ),
        )

    X = _to_model_row(
        feats,
        req.coinId,
        model.manifest.coin_vocab,
        feature_names=model.manifest.feature_names,
    )

    # Task #654 — paper-trading family C ("dual_binary_head"). Two binary
    # boosters + Platt sigmoid + abstain τ. Returns a distinct response
    # shape (`predictor_kind == "dual_binary_head"`) so the api-server's
    # quant brain can route abstains and side calls through dedicated
    # paper-trading code paths instead of bending the legacy 3-class
    # contract. Lives BEFORE the legacy 3-class block so a dual-head
    # model never falls into the booster.predict / per-class isotonic
    # path that doesn't exist for it.
    if isinstance(model, LoadedDualHeadModel):
        # Task #659 (C-BTC) — pass coin/timeframe so a scope-pinned
        # manifest (allowed_universe) can refuse out-of-scope calls.
        head_out = model.predict_one(
            X, coin_id=req.coinId, timeframe=req.timeframe,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        fhash_dh = feature_hash(feats)
        logger.info(
            "predict_dual_binary_head",
            extra={
                "coinId": req.coinId, "timeframe": req.timeframe,
                "modelCoinId": model.manifest.coin_id,
                "modelVersion": model.manifest.version,
                "pLong": head_out["p_long"],
                "pShort": head_out["p_short"],
                "abstain": head_out["abstain"],
                "side": head_out["side"],
                "abstainTau": float(model.manifest.abstain_tau),
                "candleCount": len(candles),
                "durationMs": round(duration_ms, 2),
            },
        )
        return {
            "status": "ok",
            "predictor_kind": "dual_binary_head",
            "coinId": req.coinId,
            "timeframe": req.timeframe,
            "p_long": head_out["p_long"],
            "p_short": head_out["p_short"],
            "abstain": head_out["abstain"],
            "side": head_out["side"],
            "confidence": head_out["confidence"],
            "label_family": model.manifest.label_family or "C_post_cost",
            "abstain_tau": float(model.manifest.abstain_tau),
            "friction_threshold_pct": (
                float(model.manifest.friction_threshold_pct)
                if model.manifest.friction_threshold_pct is not None
                else None
            ),
            "modelVersion": model.manifest.version,
            "modelCoinId": model.manifest.coin_id,
            "featureHash": fhash_dh,
            "fallback": fallback_kind,
        }

    probs = _calibrated_3class_probs(model, X)  # [DOWN, STABLE, UP]
    p_down, p_stable, p_up = float(probs[0]), float(probs[1]), float(probs[2])

    # expectedReturnPct: probability-weighted mean of the per-class mean returns
    # observed during training. predictionStdPct: probability-weighted std.
    # Both are now real model-derived quantities, not heuristics.
    means_pct = list(model.manifest.class_return_means_pct)
    if len(means_pct) != 3:
        # Legacy manifest fallback — keep the response sane.
        threshold_pct = model.manifest.threshold_pct
        means_pct = [-threshold_pct, 0.0, threshold_pct]
    # predictionStdPct still comes from the class-mean variance — it's a
    # cheap proxy for prediction dispersion that doesn't need a second
    # regressor head.
    class_mean_exp = float(
        p_down * means_pct[0] + p_stable * means_pct[1] + p_up * means_pct[2]
    )
    variance = (
        p_down * (means_pct[0] - class_mean_exp) ** 2
        + p_stable * (means_pct[1] - class_mean_exp) ** 2
        + p_up * (means_pct[2] - class_mean_exp) ** 2
    )
    prediction_std_pct = float(math.sqrt(max(0.0, variance)))
    # Task #135 — when the model has a regression head, use it for the
    # magnitude estimate (always positive, predicted from features), and
    # take the sign from `p_up - p_down`. Falls back to the diluted
    # class-mean expectation only for legacy models trained before the
    # regressor existed.
    if model.regressor is not None:
        try:
            reg_pred = model.regressor.predict(
                X, num_iteration=model.regressor.best_iteration,
            )
            magnitude = abs(float(np.asarray(reg_pred).flatten()[0]))
            sign = 1.0 if p_up >= p_down else -1.0
            expected_return_pct = sign * magnitude
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "regressor_predict_failed",
                extra={"coinId": req.coinId, "error": str(exc)},
            )
            expected_return_pct = class_mean_exp
    else:
        expected_return_pct = class_mean_exp

    # Confidence = max class prob normalized so a uniform [1/3,1/3,1/3] is 0
    # and a one-hot is 1. Easier for the trader UI than the raw max prob.
    confidence = float(max(0.0, (max(p_down, p_stable, p_up) - 1.0 / 3.0) * 1.5))

    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "predict_lgb",
        extra={
            "coinId": req.coinId, "timeframe": req.timeframe,
            "modelCoinId": model.manifest.coin_id,
            "probUp": p_up, "probDown": p_down, "probStable": p_stable,
            "expectedReturnPct": expected_return_pct,
            "modelVersion": model.manifest.version,
            "candleCount": len(candles), "durationMs": round(duration_ms, 2),
        },
    )

    # Phase 3 — annotate the response with the live regime label and
    # every specialist's view of the same bar. Regime classification is
    # cheap (already computed during feature build) and the specialist
    # heads use the SAME `feats` dict so live and journal stay in lock
    # step. Failures degrade silently — the legacy 3-class head above
    # is still authoritative until Phase 4.
    try:
        regime_decision = classify_regime_from_features(feats)
        regime_label: Optional[str] = regime_decision.label
    except Exception:
        regime_label = None
    try:
        specialists = _score_specialists(
            feats, req.coinId, req.timeframe, regime_label or "",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "specialists_block_failed",
            extra={"coinId": req.coinId, "timeframe": req.timeframe, "error": str(exc)},
        )
        specialists = []

    # Task #460 — every QUANT row in the api-server's prediction journal
    # MUST carry a feature_hash. The /ml/features endpoint already returns
    # one but /ml/predict didn't, so the live trading path was forwarding
    # null to journal-writer and every freshly-retrained 1d/6h LightGBM
    # row was being silently refused. Compute it here from the same
    # `feats` dict that fed the booster so live and journal stay in lock-
    # step.
    fhash = feature_hash(feats)

    return PredictResponse(
        coinId=req.coinId,
        timeframe=req.timeframe,
        probUp=p_up,
        probDown=p_down,
        probStable=p_stable,
        expectedReturnPct=expected_return_pct,
        predictionStdPct=prediction_std_pct,
        confidence=confidence,
        modelVersion=model.manifest.version,
        modelCoinId=model.manifest.coin_id,
        featureImportanceTop5=_top5_importance(model.booster, FEATURE_COLUMNS),
        source="lightgbm",
        featureHash=fhash,
        regime=regime_label,
        specialists=specialists,
        fallback=fallback_kind,
    )


@app.get("/ml/models")
async def list_available_models() -> dict:
    """Enumerate (coin, timeframe) pairs that currently have a trained model.

    Read by the api-server's ml-availability cache so the quant-brain can
    skip /predict calls that would otherwise 503 with "no model registered".
    Without this catalog, the orchestrator probes every uncovered (coin, tf)
    combination every cycle (6 coins x 4 untrained TFs = ~24 per cycle),
    which floods ml-engine logs with 503s and silently undermines the
    quant-as-primary cutover.

    The pooled slot (`__pooled__`) is included verbatim; callers resolve
    per-coin first then fall back to pooled, mirroring `_resolve_for_predict`.
    """
    available: list[dict[str, str]] = []
    if REGISTRY_ROOT.exists():
        for coin_dir in sorted(REGISTRY_ROOT.iterdir()):
            if not coin_dir.is_dir() or coin_dir.name == "datasets":
                continue
            # Phase 3 — specialist slots live under `__specialist_*__` and
            # are not coins; the api-server's availability cache should
            # never round-robin to them as a per-coin fallback.
            if coin_dir.name.startswith("__specialist_") and coin_dir.name.endswith("__"):
                continue
            for tf_dir in sorted(coin_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue
                v = latest_version(coin_dir.name, tf_dir.name)
                if not v:
                    continue
                # Cheap manifest peek so callers can distinguish a real
                # LightGBM model from a prior-only fallback. We only read
                # manifest.json (no booster load), so this stays O(1).
                kind = "lightgbm"
                try:
                    import json as _json
                    mf = _json.loads((REGISTRY_ROOT / coin_dir.name / tf_dir.name / v / "manifest.json").read_text())
                    kind = str(mf.get("model_kind", "lightgbm"))
                except Exception:
                    pass
                available.append({"coinId": coin_dir.name, "timeframe": tf_dir.name, "kind": kind})
    return {"available": available, "count": len(available)}


@app.get("/ml/training/history/directional-call-share")
async def directional_call_share_history(timeframe: Optional[str] = None, limit: int = 500) -> dict:
    """Rolling per-(coin, timeframe) timeseries of the model's directional
    call share, persisted across training runs (task #101).

    Each entry is one (coin, timeframe, version) record appended at the end
    of a training run. The dashboard groups by timeframe and plots the
    series so a regression after a label/threshold change is visible
    immediately. `alerts` lists slices whose latest sample is below the
    configured threshold (default 15% on tradeable timeframes).
    """
    from .training.train import (
        DIRECTIONAL_SHARE_HISTORY_PATH,
        DIRECTIONAL_SHARE_MIN_PCT,
        DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES,
    )

    history: list[dict] = []
    if DIRECTIONAL_SHARE_HISTORY_PATH.exists():
        try:
            with DIRECTIONAL_SHARE_HISTORY_PATH.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        history.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("directional_share_history_read_failed", extra={"error": str(exc)})
    if timeframe:
        history = [r for r in history if r.get("timeframe") == timeframe]
    # Tail-cap so a very long file doesn't dump megabytes into the response.
    capped = history[-max(1, min(limit, 5000)):]

    # Group by (timeframe, coin_id) and find the latest sample per group.
    by_series: dict[tuple[str, str], list[dict]] = {}
    for r in capped:
        key = (r.get("timeframe", "?"), r.get("coin_id", "?"))
        by_series.setdefault(key, []).append(r)

    series_out: list[dict] = []
    alerts: list[dict] = []
    for (tf, coin), points in by_series.items():
        points_sorted = sorted(points, key=lambda p: p.get("generated_at") or "")
        latest = points_sorted[-1]
        series_out.append({
            "timeframe": tf,
            "coinId": coin,
            "points": points_sorted,
            "latestSharePct": latest.get("directional_call_share_pct"),
            "latestVersion": latest.get("version"),
            "latestAt": latest.get("generated_at"),
            "tradeableTimeframe": bool(latest.get("tradeable_timeframe")),
        })
        if latest.get("below_threshold"):
            alerts.append({
                "timeframe": tf,
                "coinId": coin,
                "sharePct": latest.get("directional_call_share_pct"),
                "version": latest.get("version"),
                "thresholdPct": latest.get("threshold_pct", DIRECTIONAL_SHARE_MIN_PCT),
                "generatedAt": latest.get("generated_at"),
            })
    series_out.sort(key=lambda s: (s["timeframe"], s["coinId"]))

    return {
        "thresholdPct": DIRECTIONAL_SHARE_MIN_PCT,
        "tradeableTimeframes": sorted(DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES),
        "totalRecords": len(history),
        "returnedRecords": len(capped),
        "series": series_out,
        "alerts": alerts,
    }


@app.get("/ml/training/history/meta-calibration")
async def meta_calibration_history(timeframe: Optional[str] = None, limit: int = 500) -> dict:
    """Phase 4 — per-bucket meta-model calibration timeseries.

    Each entry is appended at the end of `run_meta_training` and records
    the per-decile realized-vs-predicted edge buckets. The dashboard
    plots latest snapshot + drift over retrains. Returns empty when no
    meta-training has happened yet (heuristic-only deployments do not
    append history rows).
    """
    from .training.train_meta import META_CALIBRATION_HISTORY_PATH
    history: list[dict] = []
    if META_CALIBRATION_HISTORY_PATH.exists():
        try:
            with META_CALIBRATION_HISTORY_PATH.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        history.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("meta_calibration_history_read_failed", extra={"error": str(exc)})
    if timeframe:
        history = [r for r in history if r.get("timeframe") == timeframe]
    capped = history[-max(1, min(limit, 5000)):]
    by_tf: dict[str, list[dict]] = {}
    for r in capped:
        tf = r.get("timeframe") or "?"
        by_tf.setdefault(tf, []).append(r)
    series_out = []
    for tf, points in by_tf.items():
        points_sorted = sorted(points, key=lambda p: p.get("generated_at") or "")
        latest = points_sorted[-1] if points_sorted else {}
        series_out.append({
            "timeframe": tf,
            "points": points_sorted,
            "latestVersion": latest.get("version"),
            "latestAt": latest.get("generated_at"),
            "latestActionAccuracy": latest.get("action_accuracy_holdout"),
            "latestEdgeMaePct": latest.get("edge_mae_holdout_pct"),
            "latestCalibrationBuckets": latest.get("calibration_buckets") or [],
            "latestClassCounts": latest.get("class_counts") or {},
        })
    series_out.sort(key=lambda s: s["timeframe"])
    return {
        "totalRecords": len(history),
        "returnedRecords": len(capped),
        "series": series_out,
    }


@app.get("/ml/training/history/calibration")
async def calibration_history(timeframe: Optional[str] = None, limit: int = 500) -> dict:
    """Rolling per-timeframe timeseries of the auto-recalibration sweep
    output, persisted across training runs (task #142).

    Each entry is one recalibration proposal appended at the end of a
    `recalibrate_after_training` run. The dashboard groups by timeframe
    and plots how (mdp, mde, factor, n_trades, pnl) drift retrain-over-
    retrain so an operator can spot a model whose distribution is
    shifting before any apply lands.
    """
    from .training.threshold_calibration import (
        CALIBRATION_HISTORY_PATH,
        DEFAULT_MIN_TRADES,
    )

    history: list[dict] = []
    if CALIBRATION_HISTORY_PATH.exists():
        try:
            with CALIBRATION_HISTORY_PATH.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        history.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("calibration_history_read_failed", extra={"error": str(exc)})
    if timeframe:
        history = [r for r in history if r.get("timeframe") == timeframe]
    capped = history[-max(1, min(limit, 5000)):]

    by_tf: dict[str, list[dict]] = {}
    for r in capped:
        tf = r.get("timeframe") or "?"
        by_tf.setdefault(tf, []).append(r)

    series_out: list[dict] = []
    for tf, points in by_tf.items():
        points_sorted = sorted(points, key=lambda p: p.get("generated_at") or "")
        latest = points_sorted[-1]
        series_out.append({
            "timeframe": tf,
            "points": points_sorted,
            "latestAt": latest.get("generated_at"),
            "latestStatus": latest.get("status"),
            "latestRecommendationStatus": latest.get("recommendation_status"),
            "latestProposed": latest.get("proposed"),
            "latestCurrent": latest.get("current"),
            "latestRecommendation": latest.get("recommendation"),
            "latestModelHash": latest.get("model_hash"),
            "latestApplied": bool(latest.get("applied")),
            "latestWouldApply": bool(latest.get("would_apply")),
        })
    series_out.sort(key=lambda s: s["timeframe"])

    return {
        "minTradesFloor": DEFAULT_MIN_TRADES,
        "totalRecords": len(history),
        "returnedRecords": len(capped),
        "series": series_out,
    }


@app.get("/ml/report", response_class=HTMLResponse)
async def report() -> HTMLResponse:
    html = render_training_html()
    return HTMLResponse(content=html, status_code=200)


@app.get("/ml/training/report")
async def training_report_json() -> dict:
    """Latest training report (the same dict that backs /ml/report's HTML).

    Lets the dashboard surface per-slice diagnostics — currently the
    gates-alignment share from task #147 — without HTML scraping. Returns
    `{}` (with `status: "missing"`) if no training run has happened yet.
    """
    from .training.report import REPORT_PATH
    if not REPORT_PATH.exists():
        return {"status": "missing"}
    try:
        return json.loads(REPORT_PATH.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("training_report_read_failed", extra={"error": str(exc)})
        return {"status": "error", "error": str(exc)}


@app.get("/ml/backtest", response_class=HTMLResponse)
async def backtest_report() -> HTMLResponse:
    """Serve the latest backtest HTML report (Phase 3). 404 with a plain
    message if the CLI hasn't been run yet — never fabricate.
    """
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "models" / "backtest_report.html"
    if not p.exists():
        return HTMLResponse(
            content="<h1>No backtest report yet</h1>"
                    "<p>Run <code>pnpm --filter @workspace/ml-engine backtest</code>.</p>",
            status_code=404,
        )
    return HTMLResponse(content=p.read_text(), status_code=200)


# --- Phase 5 — unified decision engine (`/ml/decide`) -----------------------
# Wraps `app.decision_engine.engine.decide` so the live api-server can ask
# the ml-engine for the SAME action / size / abstain decision the offline
# backtester evaluates. Pure HTTP wrapper — all gating logic lives in the
# Python module.

class DecideOpenPosition(BaseModel):
    coinId: str
    direction: str
    notionalUsd: float
    regimeAtEntry: Optional[str] = None
    betaToBtc: Optional[float] = None


class DecidePortfolioState(BaseModel):
    equityUsd: float
    cashUsd: float
    openPositions: list[DecideOpenPosition] = Field(default_factory=list)


class DecideRequest(BaseModel):
    coinId: str
    timeframe: str
    lastPrice: float
    atrValue: float
    probUp: float
    probDown: float
    probStable: float
    expectedReturnPct: float
    regime: Optional[str] = None
    trendBias: Optional[str] = None
    portfolio: Optional[DecidePortfolioState] = None
    recentOutcomes: list[int] = Field(default_factory=list)
    gateMinConfidence: Optional[float] = None
    gateMinTpDistancePct: Optional[float] = None
    gateMinEvVsCost: Optional[float] = None
    gateCounterTrendMinConfidence: Optional[float] = None


class DecideResponse(BaseModel):
    action: str
    confidence: float
    sizeMultiplier: float
    positionSizeUsd: float
    direction: Optional[str]
    slPrice: Optional[float]
    tpPrice: Optional[float]
    expiresInMs: int
    skipReason: Optional[str]
    skipDetail: Optional[str]
    gatesApplied: dict
    portfolioCheck: dict
    raw: dict


@app.post("/ml/decide", response_model=DecideResponse)
async def decide_endpoint(req: DecideRequest) -> DecideResponse:
    from .decision_engine import (
        DecisionRequest, OpenPosition, PortfolioState, decide as _decide,
    )
    portfolio = None
    if req.portfolio is not None:
        portfolio = PortfolioState(
            equity_usd=req.portfolio.equityUsd,
            cash_usd=req.portfolio.cashUsd,
            open_positions=[
                OpenPosition(
                    coin_id=p.coinId, direction=p.direction,
                    notional_usd=p.notionalUsd,
                    regime_at_entry=p.regimeAtEntry,
                    beta_to_btc=p.betaToBtc,
                )
                for p in req.portfolio.openPositions
            ],
        )
    result = _decide(DecisionRequest(
        coin_id=req.coinId, timeframe=req.timeframe,
        last_price=req.lastPrice, atr_value=req.atrValue,
        prob_up=req.probUp, prob_down=req.probDown, prob_stable=req.probStable,
        expected_return_pct=req.expectedReturnPct,
        regime=req.regime, trend_bias=req.trendBias,
        portfolio=portfolio, recent_outcomes=list(req.recentOutcomes),
        gate_min_confidence=req.gateMinConfidence,
        gate_min_tp_distance_pct=req.gateMinTpDistancePct,
        gate_min_ev_vs_cost=req.gateMinEvVsCost,
        gate_counter_trend_min_confidence=req.gateCounterTrendMinConfidence,
    ))
    return DecideResponse(
        action=result.action,
        confidence=result.confidence,
        sizeMultiplier=result.size_multiplier,
        positionSizeUsd=result.position_size_usd,
        direction=result.direction,
        slPrice=result.sl_price,
        tpPrice=result.tp_price,
        expiresInMs=result.expires_in_ms,
        skipReason=result.skip_reason,
        skipDetail=result.skip_detail,
        gatesApplied=result.gates_applied,
        portfolioCheck=result.portfolio_check,
        raw=result.raw,
    )


# --- Phase 5 — promotion-gate evaluator endpoint ----------------------------
# The api-server collects aggregate metrics from the prediction journal and
# POSTs them here for a verdict. Keeps the gate threshold logic in ONE place
# (the Python `registry_lifecycle` module) so the live promote button and the
# diagnostics dashboard see the same answer.

class PromotionMetricsBody(BaseModel):
    samples: int
    netEdgePct: float
    championNetEdgePct: float
    drawdownPct: float
    perRegimeNetEdgePct: dict[str, float] = Field(default_factory=dict)


class PromotionVerdictResponse(BaseModel):
    eligible: bool
    samplesOk: bool
    edgeLiftOk: bool
    drawdownOk: bool
    regimeRobustnessOk: bool
    reasons: list[str]
    thresholds: dict
    metricsSummary: dict


@app.get("/ml/registry/effective-serving")
async def effective_serving() -> dict:
    """Task #233 — for every (coin, timeframe) slot present on disk,
    report which version is *actually* served by the resolver right now.

    The resolver (`_resolve_for_predict`) skips quarantined registry rows
    and silently falls back to the previous trained version (or the
    pooled slot). The dashboard's registry panel shows the latest pointer
    by default, which means a quarantined champion looks "active" even
    though live trades route to an older version. This endpoint exposes
    the effective decision per slot — without loading models — so the UI
    can surface a "serving v_old (champion v_new quarantined)" badge.

    Output shape per slot:
      coinId, timeframe              — the requested slot
      latestVersion                  — file-system `latest` pointer
      servedCoinId, servedVersion    — what the resolver would actually pick
      fallback                       — true when servedVersion != latestVersion
                                       OR served via pooled fallback
      fallbackReason                 — "quarantined-skip" | "pooled-fallback"
                                       | "quarantined-skip+pooled-fallback"
                                       | null
      quarantinedVersions            — versions skipped within this slot
    """
    quarantined = await fetch_quarantined_versions()

    def _resolve(coin: str, timeframe: str) -> tuple[Optional[str], Optional[str], list[str]]:
        """Return (served_coin, served_version, skipped_versions) without
        loading any model — mirrors `_resolve_for_predict`'s pick logic."""
        skipped: list[str] = []

        def _pick(slot: str) -> Optional[str]:
            latest = latest_version(slot, timeframe)
            if latest is not None:
                if (slot, timeframe, latest) not in quarantined:
                    return latest
                skipped.append(latest)
            for v in reversed(list_versions(slot, timeframe)):
                if (slot, timeframe, v) in quarantined:
                    if v not in skipped:
                        skipped.append(v)
                    continue
                return v
            return None

        v = _pick(coin)
        if v is not None:
            return coin, v, skipped
        if coin != POOLED_COIN_ID:
            v2 = _pick(POOLED_COIN_ID)
            if v2 is not None:
                return POOLED_COIN_ID, v2, skipped
        return None, None, skipped

    out: list[dict] = []
    if REGISTRY_ROOT.exists():
        for coin_dir in sorted(REGISTRY_ROOT.iterdir()):
            if not coin_dir.is_dir() or coin_dir.name == "datasets":
                continue
            if coin_dir.name.startswith("__specialist_") and coin_dir.name.endswith("__"):
                continue
            for tf_dir in sorted(coin_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue
                coin = coin_dir.name
                tf = tf_dir.name
                latest = latest_version(coin, tf)
                served_coin, served_version, skipped = _resolve(coin, tf)
                pooled_fallback = served_coin is not None and served_coin != coin
                quarantined_skip = bool(skipped) and (
                    served_version is None
                    or served_version != latest
                    or pooled_fallback
                )
                fallback = served_version is None or served_version != latest or pooled_fallback
                if pooled_fallback and quarantined_skip:
                    reason = "quarantined-skip+pooled-fallback"
                elif pooled_fallback:
                    reason = "pooled-fallback"
                elif quarantined_skip:
                    reason = "quarantined-skip"
                else:
                    reason = None
                out.append({
                    "coinId": coin,
                    "timeframe": tf,
                    "latestVersion": latest,
                    "servedCoinId": served_coin,
                    "servedVersion": served_version,
                    "fallback": fallback,
                    "fallbackReason": reason,
                    "quarantinedVersions": skipped,
                })
    return {"slots": out, "count": len(out)}


@app.post("/ml/registry/evaluate-promotion", response_model=PromotionVerdictResponse)
async def evaluate_promotion_endpoint(req: PromotionMetricsBody) -> PromotionVerdictResponse:
    from .registry_lifecycle import PromotionMetrics, evaluate_promotion
    v = evaluate_promotion(PromotionMetrics(
        samples=req.samples,
        net_edge_pct=req.netEdgePct,
        champion_net_edge_pct=req.championNetEdgePct,
        drawdown_pct=req.drawdownPct,
        per_regime_net_edge_pct=dict(req.perRegimeNetEdgePct),
    ))
    return PromotionVerdictResponse(
        eligible=v.eligible,
        samplesOk=v.samples_ok,
        edgeLiftOk=v.edge_lift_ok,
        drawdownOk=v.drawdown_ok,
        regimeRobustnessOk=v.regime_robustness_ok,
        reasons=v.reasons,
        thresholds=v.thresholds,
        metricsSummary=v.metrics_summary,
    )


# --- Phase 6 — Feature Lab ablation runner ---------------------------------
class FeatureLabAblateBody(BaseModel):
    name: str
    transformKind: str
    sourceColumn: Optional[str] = None
    timeframe: str
    coinId: str = "__pooled__"
    nFolds: int = Field(default=3, ge=2, le=8)


@app.post("/ml/feature-lab/ablate")
async def feature_lab_ablate(body: FeatureLabAblateBody) -> dict:
    """Walk-forward ablate a candidate feature on the latest dataset
    snapshot for `timeframe`. Returns the metric deltas the api-server
    persists to `feature_lab_reports`. Pure compute; no registry
    mutation. See `app.training.feature_lab.run_ablation` for the
    transform allow-list and metric definitions.
    """
    from .training.feature_lab import run_ablation
    try:
        result = run_ablation(
            name=body.name,
            transform_kind=body.transformKind,
            source_column=body.sourceColumn,
            timeframe=body.timeframe,
            coin_id=body.coinId,
            n_folds=body.nFolds,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/ml/admin/reload")
async def reload_models(request: Request) -> dict[str, str]:
    """Manual cache invalidation hook so a re-train without service restart
    can be picked up.

    Auth: requires header `X-Admin-Token` matching `ML_ADMIN_TOKEN` env. If
    `ML_ADMIN_TOKEN` is unset the endpoint returns 404 (effectively disabled),
    so an accidental public deploy doesn't expose unauthenticated cache
    invalidation.
    """
    expected = os.environ.get("ML_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=404, detail="not enabled")
    provided = request.headers.get("x-admin-token")
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="bad token")
    _cached_load.cache_clear()
    return {"status": "ok", "message": "model cache cleared"}


_retrain_lock = threading.Lock()
_retrain_state: dict = {"running": False, "last_started_at": None, "last_finished_at": None, "last_status": None, "last_report": None, "last_error": None}


def _append_admin_retrain_progress(record: dict) -> None:
    """Append a structured progress event to `models/progress_updates.jsonl`.

    Mirrors the writer used by the scheduled slow-loop entry point
    (`scripts/run_full_training_campaign.py::_append_progress`) so the
    dashboard's `/ml/training/progress` endpoint sees admin-triggered
    retrains the same way it sees scheduled ones (Task #636). Failures
    are swallowed — a broken progress log must never abort a retrain.
    """
    try:
        from datetime import datetime, timezone
        progress_path = REGISTRY_ROOT / "progress_updates.jsonl"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"emitted_at": datetime.now(timezone.utc).isoformat(), **record}
        with progress_path.open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 — progress log is best-effort
        logger.warning("admin_retrain_progress_append_failed", extra={"error": str(exc)})


def _run_retrain_blocking(coins: Optional[list[str]], timeframes: Optional[list[str]]) -> None:
    from . import db as db_mod
    from .training.train import DEFAULT_COINS, DEFAULT_TIMEFRAMES, run_training
    from .training.registry import prune_contaminated_versions
    cs = list(coins) if coins else list(DEFAULT_COINS)
    tfs = list(timeframes) if timeframes else list(DEFAULT_TIMEFRAMES)
    # The FastAPI lifespan owns a pool bound to its event loop. Detach it
    # for the duration of training so run_training (in a fresh asyncio loop
    # on this background thread) creates+closes its own loop-local pool.
    saved_pool = db_mod._pool
    db_mod._pool = None
    try:
        # Task #636 — wire the campaign-style progress callback so the
        # dashboard's `/ml/training/progress` endpoint streams live events
        # for admin-triggered retrains, not just scheduled ones. Without
        # this, operators triggering a 2–3 hour retrain via the admin
        # endpoint had to fall back to filesystem polling and CPU watching.
        report = asyncio.run(
            run_training(cs, tfs, progress_callback=_append_admin_retrain_progress)
        )
        _retrain_state["last_report"] = {tf: (v.get("status") if isinstance(v, dict) else None) for tf, v in (report.get("timeframes") or {}).items()}
        _retrain_state["last_status"] = "ok"
        _retrain_state["last_error"] = None
        _cached_load.cache_clear()
    except Exception as e:  # pragma: no cover - background error path
        _retrain_state["last_status"] = "error"
        _retrain_state["last_error"] = str(e)
    finally:
        # run_training closed its loop-local pool already; restore the
        # FastAPI pool reference so subsequent /predict calls reuse it.
        db_mod._pool = saved_pool
        # Task #451 — auto-prune any contaminated model artifacts after
        # every training cycle. Runs in `finally` so it executes even
        # when the training run errored out (the slot it was trying to
        # promote may itself have been the contaminated dir we want to
        # remove). Best-effort: a janitor failure must not flip the
        # retrain state — we already have `last_status="ok"` set above
        # in the success path and the janitor's own error log will tell
        # an operator that disk cleanup was skipped.
        try:
            pruned = prune_contaminated_versions()
            _retrain_state["last_pruned_count"] = int(pruned.get("deleted") or 0)
            _retrain_state["last_pruned_bytes"] = int(pruned.get("freed_bytes") or 0)
            # Clear any stale error from a previous failed janitor run so
            # the dashboard doesn't keep showing it after recovery.
            _retrain_state["last_pruned_error"] = None
            if pruned.get("deleted"):
                logger.info(
                    "auto_prune_contaminated_models",
                    extra={
                        "deleted": pruned["deleted"],
                        "freed_bytes": pruned["freed_bytes"],
                    },
                )
                # A pruned dir may still be cached by the LRU; clear so
                # the next /predict re-resolves to a clean version.
                _cached_load.cache_clear()
        except Exception as exc:  # noqa: BLE001 — janitor must never break training
            logger.warning(
                "auto_prune_contaminated_models_failed",
                extra={"error": str(exc)},
            )
            _retrain_state["last_pruned_error"] = str(exc)
        _retrain_state["last_finished_at"] = time.time()
        _retrain_state["running"] = False
        _retrain_lock.release()


@app.post("/ml/admin/retrain")
async def admin_retrain(request: Request) -> dict:
    """Kick off a (re)training run in a background thread.

    Auth: same `X-Admin-Token` / `ML_ADMIN_TOKEN` env as /admin/reload.
    Body (optional JSON): {"coins": [...], "timeframes": [...]}. Defaults to
    DEFAULT_COINS / DEFAULT_TIMEFRAMES from training.train.

    Returns 202 on enqueue, 409 if a run is already in progress.
    """
    expected = os.environ.get("ML_ADMIN_TOKEN") or os.environ.get("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(status_code=404, detail="not enabled")
    provided = request.headers.get("x-admin-token") or request.headers.get("x-admin-key")
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="bad token")
    try:
        body = await request.json()
    except Exception:
        body = {}
    coins = body.get("coins") if isinstance(body, dict) else None
    timeframes = body.get("timeframes") if isinstance(body, dict) else None
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="retrain already in progress")
    _retrain_state["running"] = True
    _retrain_state["last_started_at"] = time.time()
    t = threading.Thread(target=_run_retrain_blocking, args=(coins, timeframes), daemon=True)
    t.start()
    return {"status": "accepted", "started_at": _retrain_state["last_started_at"]}


@app.get("/ml/admin/retrain/status")
async def admin_retrain_status() -> dict:
    return dict(_retrain_state)


@app.get("/ml/admin/verification-history")
async def admin_verification_history(limit: int = 50) -> dict:
    """Return the training watchdog history (newest first).

    Each row is a snapshot of `verification` from one retrain plus the
    `diff` envelope (status: converged | improving | regressed | flat |
    stalled | first_run). Read-only; no auth, mirrors `/admin/retrain/status`.
    """
    try:
        from app.training import registry as registry_module
        from app.training.watchdog import read_history
        rows = read_history(registry_module.REGISTRY_ROOT, limit=limit)
        return {"rows": rows, "count": len(rows)}
    except Exception as exc:  # noqa: BLE001
        return {"rows": [], "count": 0, "error": str(exc)}


@app.get("/ml/admin/failure-analysis/history")
async def admin_failure_analysis_history(limit: int = 30) -> dict:
    """Return the newest N auto-generated failure-analysis envelopes (Task #336).

    Each row is `{generated_at, bucket_counts, json_path}` from the most
    recent `<UTC_TS>-failure-analysis-auto.json` files in
    `artifacts/ml-engine/reports/`. Newest first. Read-only; mirrors
    `/ml/admin/verification-history`.
    """
    try:
        from app.training import registry as registry_module
        from app.training.failure_analysis import history
        capped = max(1, min(int(limit), 200))
        return history(registry_module.REGISTRY_ROOT, limit=capped)
    except Exception as exc:  # noqa: BLE001
        return {"rows": [], "count": 0, "error": str(exc)}


@app.get("/ml/admin/failure-analysis/latest")
async def admin_failure_analysis_latest() -> dict:
    """Return the newest auto-generated failure-analysis pair (Task #327).

    `generated_at`, `bucket_counts`, and the rendered `summary_md` are
    pulled from the most recent `<UTC_TS>-failure-analysis-auto.{json,md}`
    in `artifacts/ml-engine/reports/`. Read-only; no auth.
    """
    try:
        from app.training import registry as registry_module
        from app.training.failure_analysis import latest_pair
        return latest_pair(registry_module.REGISTRY_ROOT)
    except Exception as exc:  # noqa: BLE001
        return {
            "generated_at": None,
            "bucket_counts": {},
            "summary_md": "",
            "json_path": None,
            "md_path": None,
            "error": str(exc),
        }


# --- Auto-retrain scheduler (task #105) ---------------------------------------
# When prior-only pooled fallbacks exist for any advertised timeframe, this
# scheduler periodically retries `run_training` so a TF gets promoted from
# `prior` to `lightgbm` the moment it has enough per-coin candle history.
# Each promotion is logged AND persisted to a JSONL transitions file so the
# dashboard can show "1h promoted prior -> lightgbm at <timestamp>" after
# the fact, instead of requiring a manual `pnpm --filter ... train`.
#
# Ops note: the scheduler is process-local. If ml-engine is ever deployed
# with multiple workers/replicas, set `ML_AUTO_RETRAIN_ENABLED=0` on all
# but one instance — otherwise each process will tick independently and
# duplicate the retrain work (file-level coordination via `_retrain_lock`
# only exists within a single process).
AUTO_RETRAIN_ENABLED = os.environ.get("ML_AUTO_RETRAIN_ENABLED", "1") == "1"
AUTO_RETRAIN_INTERVAL_SECONDS = int(
    os.environ.get("ML_AUTO_RETRAIN_INTERVAL_SECONDS", "1800")
)  # 30 min default — fast enough that 1h gets promoted within an hour of crossing.
AUTO_RETRAIN_MIN_GAP_SECONDS = int(
    os.environ.get("ML_AUTO_RETRAIN_MIN_GAP_SECONDS", "1800")
)  # Don't kick off another auto-retrain within this window of the last attempt.
AUTO_RETRAIN_TRANSITIONS_FILENAME = "auto_retrain_transitions.jsonl"


def _auto_retrain_transitions_path():
    # Read REGISTRY_ROOT through the module so tests that monkeypatch it
    # (see test_app.py) point this file at their tmp dir.
    return REGISTRY_ROOT / AUTO_RETRAIN_TRANSITIONS_FILENAME

_auto_retrain_state: dict = {
    "enabled": AUTO_RETRAIN_ENABLED,
    "interval_seconds": AUTO_RETRAIN_INTERVAL_SECONDS,
    "min_gap_seconds": AUTO_RETRAIN_MIN_GAP_SECONDS,
    "last_check_at": None,
    "last_attempt_at": None,
    "last_attempt_outcome": None,  # "kicked_off" | "skipped_no_priors" | "skipped_busy" | "skipped_too_soon" | "error"
    "last_error": None,
    "last_promoted_count": 0,
    "promotions_total": 0,
}
_auto_retrain_stop_event = threading.Event()
_auto_retrain_thread: Optional[threading.Thread] = None


def _snapshot_pooled_kinds() -> dict[str, str]:
    """Return {timeframe: model_kind} for pooled models currently in the
    registry. Missing pooled timeframes are reported as kind 'missing' so the
    scheduler treats them as still-needing-promotion.
    """
    from .training.train import DEFAULT_TIMEFRAMES

    kinds: dict[str, str] = {}
    pooled_dir = REGISTRY_ROOT / POOLED_COIN_ID
    for tf in DEFAULT_TIMEFRAMES:
        v = latest_version(POOLED_COIN_ID, tf)
        if not v:
            kinds[tf] = "missing"
            continue
        try:
            mf = json.loads((pooled_dir / tf / v / "manifest.json").read_text())
            kinds[tf] = str(mf.get("model_kind", "lightgbm"))
        except Exception:
            kinds[tf] = "unknown"
    return kinds


def _needs_auto_retrain(kinds: dict[str, str]) -> list[str]:
    """Timeframes whose pooled slot is either missing or still serving a
    prior-only fallback. These are the candidates for promotion."""
    return [tf for tf, kind in kinds.items() if kind in ("prior", "missing")]


def _append_transition_record(record: dict) -> None:
    try:
        path = _auto_retrain_transitions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 - history is non-essential
        logger.warning("auto_retrain_transition_append_failed", extra={"error": str(exc)})


def _record_promotions(before: dict[str, str], after: dict[str, str]) -> int:
    """Compare pre/post pooled-kind snapshots, emit a structured log line and
    a JSONL record for every TF whose kind moved from prior/missing to a real
    lightgbm model. Returns the number of promotions detected.
    """
    promoted = 0
    ts = time.time()
    iso = pd.Timestamp(ts, unit="s", tz="UTC").isoformat()
    for tf, post_kind in after.items():
        pre_kind = before.get(tf, "missing")
        if pre_kind in ("prior", "missing") and post_kind == "lightgbm":
            promoted += 1
            logger.info(
                "auto_retrain_promotion",
                extra={
                    "timeframe": tf,
                    "from_kind": pre_kind,
                    "to_kind": post_kind,
                    "version": latest_version(POOLED_COIN_ID, tf),
                    "promoted_at": iso,
                },
            )
            _append_transition_record({
                "timeframe": tf,
                "from_kind": pre_kind,
                "to_kind": post_kind,
                "version": latest_version(POOLED_COIN_ID, tf),
                "promoted_at": iso,
                "trigger": "auto_retrain",
            })
    return promoted


def _auto_retrain_tick(now: Optional[float] = None) -> str:
    """One scheduler iteration. Returns the outcome string also written to
    `_auto_retrain_state['last_attempt_outcome']` so it's exercisable by tests
    without the polling loop.
    """
    now = now if now is not None else time.time()
    _auto_retrain_state["last_check_at"] = now
    if not AUTO_RETRAIN_ENABLED:
        _auto_retrain_state["last_attempt_outcome"] = "disabled"
        return "disabled"

    kinds_before = _snapshot_pooled_kinds()
    needs = _needs_auto_retrain(kinds_before)
    if not needs:
        # Nothing to do — every advertised TF already has a real LightGBM
        # pooled model. Cheap no-op until a new TF is added.
        _auto_retrain_state["last_attempt_outcome"] = "skipped_no_priors"
        return "skipped_no_priors"

    last_attempt = _auto_retrain_state.get("last_attempt_at")
    if last_attempt is not None and (now - float(last_attempt)) < AUTO_RETRAIN_MIN_GAP_SECONDS:
        _auto_retrain_state["last_attempt_outcome"] = "skipped_too_soon"
        return "skipped_too_soon"

    if not _retrain_lock.acquire(blocking=False):
        # A manual /admin/retrain (or a previous auto run) is already
        # underway. Try again next tick.
        _auto_retrain_state["last_attempt_outcome"] = "skipped_busy"
        return "skipped_busy"

    _retrain_state["running"] = True
    _retrain_state["last_started_at"] = now
    _auto_retrain_state["last_attempt_at"] = now
    _auto_retrain_state["last_attempt_outcome"] = "kicked_off"
    logger.info(
        "auto_retrain_kickoff",
        extra={
            "needs_promotion": needs,
            "kinds_before": kinds_before,
            "interval_seconds": AUTO_RETRAIN_INTERVAL_SECONDS,
        },
    )

    def _runner():
        try:
            _run_retrain_blocking(None, None)
        finally:
            try:
                kinds_after = _snapshot_pooled_kinds()
                promoted = _record_promotions(kinds_before, kinds_after)
                _auto_retrain_state["last_promoted_count"] = promoted
                _auto_retrain_state["promotions_total"] = (
                    int(_auto_retrain_state.get("promotions_total") or 0) + promoted
                )
                # Clear stale error on a clean post-check so the dashboard
                # doesn't keep showing a previous failure after recovery.
                _auto_retrain_state["last_error"] = None
            except Exception as exc:  # noqa: BLE001
                _auto_retrain_state["last_error"] = str(exc)
                logger.warning("auto_retrain_post_check_failed", extra={"error": str(exc)})

    threading.Thread(target=_runner, daemon=True).start()
    return "kicked_off"


def _auto_retrain_loop() -> None:
    # Small initial delay so the lifespan startup finishes (and DB pool warms
    # up) before the first tick.
    if _auto_retrain_stop_event.wait(5.0):
        return
    while not _auto_retrain_stop_event.is_set():
        try:
            _auto_retrain_tick()
        except Exception as exc:  # noqa: BLE001 - scheduler must never die
            _auto_retrain_state["last_error"] = str(exc)
            logger.warning("auto_retrain_tick_failed", extra={"error": str(exc)})
        if _auto_retrain_stop_event.wait(AUTO_RETRAIN_INTERVAL_SECONDS):
            return


def _start_auto_retrain_scheduler() -> None:
    global _auto_retrain_thread
    if not AUTO_RETRAIN_ENABLED:
        logger.info("auto_retrain_disabled")
        return
    # Tests use TestClient which spins lifespan up/down many times; don't
    # spawn a background thread there. Guarded by an explicit env var so a
    # production container can never accidentally inherit it.
    if os.environ.get("ML_AUTO_RETRAIN_TEST_DISABLE") == "1":
        return
    if _auto_retrain_thread is not None and _auto_retrain_thread.is_alive():
        return
    _auto_retrain_stop_event.clear()
    _auto_retrain_thread = threading.Thread(
        target=_auto_retrain_loop, daemon=True, name="ml-auto-retrain",
    )
    _auto_retrain_thread.start()
    logger.info(
        "auto_retrain_started",
        extra={
            "interval_seconds": AUTO_RETRAIN_INTERVAL_SECONDS,
            "min_gap_seconds": AUTO_RETRAIN_MIN_GAP_SECONDS,
        },
    )


def _stop_auto_retrain_scheduler() -> None:
    global _auto_retrain_thread
    _auto_retrain_stop_event.set()
    t = _auto_retrain_thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _auto_retrain_thread = None


@app.get("/ml/admin/auto-retrain/status")
async def auto_retrain_status(limit: int = 50) -> dict:
    """Scheduler state + the most recent prior->lightgbm transitions so the
    dashboard can render "1h promoted at <ts>" without re-reading the whole
    registry. Read-only; no auth (mirrors /ml/admin/retrain/status).
    """
    transitions: list[dict] = []
    path = _auto_retrain_transitions_path()
    if path.exists():
        try:
            with path.open() as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        transitions.append(json.loads(raw))
                    except Exception:
                        continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto_retrain_transitions_read_failed", extra={"error": str(exc)})
    transitions = transitions[-max(1, min(limit, 1000)):]
    return {
        **_auto_retrain_state,
        # Surface whether the training engine is currently busy (either from
        # a manual /admin/retrain or an in-flight auto tick) so the dashboard
        # can show a spinner rather than inferring from `last_attempt_at`.
        "in_progress": bool(_retrain_state.get("running")),
        "current_pooled_kinds": _snapshot_pooled_kinds(),
        "recent_transitions": transitions,
        "transitions_count": len(transitions),
    }


@app.get("/ml/admin/5m-topup/status")
async def five_m_topup_status() -> dict:
    """Task #410 — surface the daily 5m head top-up state so the dashboard
    can show "last topped up at <ts>, alerts: []" without scraping logs.
    Mirrors the shape of /ml/admin/auto-retrain/status (read-only, no auth).

    Task #424 — also overlay the cross-replica winner record from the
    shared `app_settings` row so any replica's status response reflects
    the actual fleet-wide last-winner. Without this overlay, a replica
    that has only ever skipped (because another box held the advisory
    lock) would report ``last_winning_replica=None`` even though some
    other replica did the work.
    """
    from . import scheduled_5m_topup

    body = dict(scheduled_5m_topup.state)
    winner = await scheduled_5m_topup._load_last_winner()
    if winner is not None:
        body["last_winning_replica"] = winner["replica"]
        body["last_winning_at"] = winner["tick_at"]
    # Task #442 — overlay the cross-replica stuck-replica streak from
    # the shared `app_settings` recent_winners row so any replica's
    # status response reflects the actual fleet-wide streak. A replica
    # that has only ever skipped (because another box held the
    # advisory lock) would otherwise report `stuck_replica_streak=0`
    # even when some other host has been winning every day for a
    # fortnight.
    try:
        overlay = await scheduled_5m_topup._load_stuck_replica_overlay()
    except Exception:  # noqa: BLE001 — best-effort; never break the endpoint
        overlay = None
    if overlay is not None:
        body["stuck_replica_threshold"] = overlay["stuck_replica_threshold"]
        body["stuck_replica_streak"] = overlay["stuck_replica_streak"]
        body["stuck_replica"] = overlay["stuck_replica"]
    return body


@app.get("/ml/admin/5m-topup/recent-winners")
async def five_m_topup_recent_winners(limit: str = "14") -> dict:
    """Task #435 — return the last N successful-tick winners
    (replica + ISO timestamp, newest-first) so the dashboard can render
    a recent-history list. The single ``/status`` endpoint already
    surfaces the very latest winner; this endpoint adds the "is the
    lock evenly distributed across boxes, or is the same replica
    winning every day?" view on top.

    The limit is clamped to ``[1, _RECENT_WINNERS_MAX]`` so a typo'd
    query string can never (a) request 0 rows and look broken nor
    (b) demand more rows than the writer ever stores. We accept the
    raw string and parse defensively (falling back to the default
    of 14) instead of letting FastAPI bounce non-numeric input with
    a 422 — the dashboard never wants to render an error here, it
    just wants the recent list.
    """
    from . import scheduled_5m_topup

    cap = scheduled_5m_topup._RECENT_WINNERS_MAX
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        parsed = scheduled_5m_topup._RECENT_WINNERS_DEFAULT_LIMIT
    safe_limit = max(1, min(parsed, cap))
    winners = await scheduled_5m_topup._load_recent_winners(limit=safe_limit)
    return {"winners": winners, "limit": safe_limit, "max": cap}


# --- Fast-loop scheduler (task #267 / #272) -----------------------------------
# The slow loop above retrains base + meta on a 30-min cadence. The fast loop
# runs only the meta refit (`train.run_meta_only`) on a much shorter cadence
# so the meta head can adapt within minutes once new resolved trades land,
# without paying the full base-retrain cost every time.
#
# Coordination: the two loops share `_retrain_lock` is too coarse (slow loop
# releases it only after its long base retrain). Instead the fast loop uses
# its own `_fast_loop_lock` and skips a tick whenever the slow runner reports
# `running` — that keeps the two loops mutually exclusive on a given tick
# (per TRAINING_CONTRACT.md rule 6) without the fast tick ever blocking the
# slow tick (it returns immediately on conflict).
FAST_LOOP_ENABLED = os.environ.get("ML_FAST_LOOP_ENABLED", "1") == "1"
FAST_LOOP_INTERVAL_SECONDS = int(
    os.environ.get("ML_FAST_LOOP_INTERVAL_SECONDS", "60")
)  # default 1 min — fine-grained polling, real fire gated by row-count threshold

_fast_loop_lock = threading.Lock()
_fast_loop_state: dict = {
    "enabled": FAST_LOOP_ENABLED,
    "interval_seconds": FAST_LOOP_INTERVAL_SECONDS,
    "min_new_rows": None,            # filled in on first tick from train.FAST_LOOP_MIN_NEW_ROWS
    "last_check_at": None,
    "last_attempt_at": None,
    "last_attempt_outcome": None,    # disabled | skipped_slow_in_progress | skipped_busy | below_threshold | count_failed | kicked_off
    "last_decision_reason": None,    # from should_run_fast_loop
    "last_finished_at": None,
    "last_resolved_count": 0,        # snapshot of resolved-meta-row count taken at last kickoff
    "last_new_rows": 0,              # delta vs. last_resolved_count seen on most recent tick
    "last_error": None,
    "ticks_total": 0,
    "runs_total": 0,
    "last_envelope": None,           # slim {loop, generated_at} from the last fast retrain
}
_fast_loop_stop_event = threading.Event()
_fast_loop_thread: Optional[threading.Thread] = None


async def _count_resolved_meta_rows() -> int:
    """Cheap COUNT(*) of QUANT prediction-journal rows that satisfy the same
    filter `meta_dataset.build_meta_dataset` uses. We compare this number
    against the last snapshot to derive `new_meta_rows_since_last` for
    `should_run_fast_loop` — no need to materialise the full frame.
    """
    from . import db as db_mod
    pool = await db_mod.init_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT count(*) AS n FROM prediction_journal "
            "WHERE brain = 'QUANT' "
            "  AND realized_return_pct IS NOT NULL "
            "  AND gates_applied IS NOT NULL"
        )
    return int(row["n"]) if row else 0


def _run_fast_loop_blocking() -> None:
    """Background runner for a single fast-loop kickoff. Runs in its own
    asyncio loop on a fresh thread (mirrors `_run_retrain_blocking`) so the
    FastAPI lifespan pool is left untouched — `run_meta_only` opens and
    closes a loop-local pool of its own.
    """
    from . import db as db_mod
    from .training.train import DEFAULT_TIMEFRAMES, run_meta_only

    saved_pool = db_mod._pool
    db_mod._pool = None
    try:
        envelope = asyncio.run(run_meta_only(list(DEFAULT_TIMEFRAMES)))
        _fast_loop_state["last_envelope"] = {
            "loop": envelope.get("loop") if isinstance(envelope, dict) else None,
            "generated_at": envelope.get("generated_at") if isinstance(envelope, dict) else None,
        }
        _fast_loop_state["last_error"] = None
        _cached_load.cache_clear()
    except Exception as exc:  # noqa: BLE001 - background error path
        _fast_loop_state["last_error"] = str(exc)
        logger.warning("fast_loop_run_failed", extra={"error": str(exc)})
    finally:
        db_mod._pool = saved_pool
        _fast_loop_state["last_finished_at"] = time.time()
        _fast_loop_state["runs_total"] = int(_fast_loop_state.get("runs_total") or 0) + 1
        _fast_loop_lock.release()


def _fast_loop_tick(now: Optional[float] = None) -> str:
    """One fast-loop scheduler iteration. Returns the outcome string also
    written to `_fast_loop_state['last_attempt_outcome']` so tests can drive
    it without the polling loop. NEVER blocks the caller for more than a
    cheap COUNT(*) — the actual meta retrain runs on a daemon thread.
    """
    from .training.train import FAST_LOOP_MIN_NEW_ROWS, should_run_fast_loop

    now = now if now is not None else time.time()
    _fast_loop_state["last_check_at"] = now
    _fast_loop_state["ticks_total"] = int(_fast_loop_state.get("ticks_total") or 0) + 1
    _fast_loop_state["min_new_rows"] = FAST_LOOP_MIN_NEW_ROWS

    if not FAST_LOOP_ENABLED:
        _fast_loop_state["last_attempt_outcome"] = "disabled"
        return "disabled"

    if _retrain_state.get("running"):
        # Slow loop (or a manual /admin/retrain) is in flight — skip and
        # try again next tick. The slow loop already retrains the meta
        # head at the end of its run, so there's nothing to gain from
        # firing the fast loop concurrently and we'd risk overlapping
        # writes to the meta registry slot.
        _fast_loop_state["last_attempt_outcome"] = "skipped_slow_in_progress"
        return "skipped_slow_in_progress"

    try:
        from . import db as db_mod
        saved_pool = db_mod._pool
        db_mod._pool = None
        try:
            count = asyncio.run(_count_resolved_meta_rows())
        finally:
            db_mod._pool = saved_pool
    except Exception as exc:  # noqa: BLE001
        _fast_loop_state["last_error"] = str(exc)
        _fast_loop_state["last_attempt_outcome"] = "count_failed"
        logger.warning("fast_loop_count_failed", extra={"error": str(exc)})
        return "count_failed"

    new_rows = max(0, count - int(_fast_loop_state.get("last_resolved_count") or 0))
    _fast_loop_state["last_new_rows"] = new_rows
    should, reason = should_run_fast_loop(
        new_meta_rows_since_last=new_rows, threshold=FAST_LOOP_MIN_NEW_ROWS,
    )
    _fast_loop_state["last_decision_reason"] = reason
    if not should:
        _fast_loop_state["last_attempt_outcome"] = "below_threshold"
        return "below_threshold"

    if not _fast_loop_lock.acquire(blocking=False):
        # Previous kickoff still in flight — wait for the next tick. We
        # don't bump `last_resolved_count` here so the next tick still
        # sees the accumulated rows.
        _fast_loop_state["last_attempt_outcome"] = "skipped_busy"
        return "skipped_busy"

    _fast_loop_state["last_attempt_at"] = now
    _fast_loop_state["last_attempt_outcome"] = "kicked_off"
    # Snapshot the row count BEFORE kickoff so the next tick measures
    # rows resolved while this run is in flight (or after).
    _fast_loop_state["last_resolved_count"] = count
    threading.Thread(target=_run_fast_loop_blocking, daemon=True, name="ml-fast-loop-run").start()
    return "kicked_off"


def _fast_loop_scheduler() -> None:
    # Small startup delay so the lifespan + DB pool finish initialising
    # before the first count query.
    if _fast_loop_stop_event.wait(7.0):
        return
    while not _fast_loop_stop_event.is_set():
        try:
            _fast_loop_tick()
        except Exception as exc:  # noqa: BLE001 - scheduler must never die
            _fast_loop_state["last_error"] = str(exc)
            logger.warning("fast_loop_tick_failed", extra={"error": str(exc)})
        if _fast_loop_stop_event.wait(FAST_LOOP_INTERVAL_SECONDS):
            return


def _start_fast_loop_scheduler() -> None:
    global _fast_loop_thread
    if not FAST_LOOP_ENABLED:
        logger.info("fast_loop_disabled")
        return
    # Tests use TestClient which spins lifespan up/down many times; reuse
    # the same env switch the slow scheduler honours so a single env var
    # disables both background threads in unit tests.
    if os.environ.get("ML_AUTO_RETRAIN_TEST_DISABLE") == "1":
        return
    if _fast_loop_thread is not None and _fast_loop_thread.is_alive():
        return
    _fast_loop_stop_event.clear()
    _fast_loop_thread = threading.Thread(
        target=_fast_loop_scheduler, daemon=True, name="ml-fast-loop",
    )
    _fast_loop_thread.start()
    logger.info(
        "fast_loop_started",
        extra={"interval_seconds": FAST_LOOP_INTERVAL_SECONDS},
    )


def _stop_fast_loop_scheduler() -> None:
    global _fast_loop_thread
    _fast_loop_stop_event.set()
    t = _fast_loop_thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _fast_loop_thread = None


@app.get("/ml/training/progress")
async def training_progress() -> dict:
    """Live training-run heartbeat for the dashboard.

    Reads the tail of `models/progress_updates.jsonl` (written by
    `_append_progress` in the training campaign and the per-coin
    callbacks in `train.py`) and surfaces:

      * `latest_emitted_at` — UTC ISO timestamp of the newest row
      * `stale` / `stale_seconds` — true when no row arrived in 5 min
      * `currently_fitting` — `{coin, timeframe, index, total}` of the
        per-coin slice now being trained (parsed from the headline of
        the most recent `per_coin_start` that has not yet been matched
        by a `per_coin_done`/`per_coin_skipped`)
      * `current_timeframe` — most recent timeframe seen via
        `build_dataset_*` / `train_*` events
      * `recent` — last ~20 events, newest first
      * `status: "missing"` when the file does not exist yet
    """
    from collections import deque
    import re
    from datetime import datetime, timezone

    progress_path = REGISTRY_ROOT / "progress_updates.jsonl"
    if not progress_path.exists():
        return {"status": "missing"}

    rows: list[dict] = []
    try:
        with progress_path.open("rb") as fh:
            try:
                fh.seek(0, 2)
                size = fh.tell()
                # 512 KiB tail comfortably covers a full ~60-coin × 5-timeframe
                # campaign (~300 slices × ~3 events ≈ 900 lines) so the
                # timeline chart on the dashboard can render the whole run.
                read_bytes = min(size, 512 * 1024)
                fh.seek(size - read_bytes)
                chunk = fh.read().decode("utf-8", errors="replace")
                truncated = read_bytes < size
            except OSError:
                fh.seek(0)
                chunk = fh.read().decode("utf-8", errors="replace")
                truncated = False
        lines = chunk.splitlines()
        if truncated and len(lines) > 1:
            lines = lines[1:]
        tail = deque(lines, maxlen=3000)
        for ln in tail:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    except Exception as exc:  # noqa: BLE001
        logger.warning("training_progress_read_failed", extra={"error": str(exc)})
        return {"status": "error", "error": str(exc)}

    if not rows:
        return {"status": "empty"}

    latest = rows[-1]
    latest_at = latest.get("emitted_at")
    stale_seconds: Optional[float] = None
    stale = False
    if isinstance(latest_at, str):
        try:
            ts = datetime.fromisoformat(latest_at.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            stale_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
            stale = stale_seconds > 5 * 60
        except ValueError:
            stale_seconds = None

    current_timeframe: Optional[str] = None
    for r in reversed(rows):
        tf = r.get("timeframe")
        if tf and r.get("phase") in {
            "build_dataset_start", "build_dataset_done",
            "train_start", "train_done",
        }:
            current_timeframe = tf
            break

    headline_re = re.compile(
        r"fitting\s+([^/\s]+)/([^/\s]+)\s+\((\d+)/(\d+)\)"
    )
    currently_fitting: Optional[dict] = None
    seen_done_coins: set[str] = set()
    for r in reversed(rows):
        phase = r.get("phase")
        coin = r.get("coin")
        if phase in {"per_coin_done", "per_coin_skipped"} and coin:
            seen_done_coins.add(coin)
            continue
        if phase == "per_coin_start" and coin and coin not in seen_done_coins:
            headline = r.get("headline") or ""
            m = headline_re.search(headline)
            tf_from_headline = m.group(2) if m else None
            try:
                idx = int(m.group(3)) if m else None
                total = int(m.group(4)) if m else None
            except (TypeError, ValueError):
                idx = total = None
            currently_fitting = {
                "coin": coin,
                "timeframe": tf_from_headline or current_timeframe,
                "index": idx,
                "total": total,
                "started_at": r.get("emitted_at"),
                "headline": headline or None,
            }
            break

    recent = list(reversed(rows[-20:]))

    # Build a per-slice timeline (start/end/elapsed) so the dashboard can
    # render a Gantt-style chart instead of a plain text list. Walk the
    # rows chronologically pairing each `per_coin_start` with the next
    # matching `per_coin_done` / `per_coin_skipped`. Unfinished slices
    # are still emitted (with status="running") so the live row shows up.
    def _parse_iso(value: object) -> Optional[datetime]:
        if not isinstance(value, str):
            return None
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    slices: list[dict] = []
    open_by_coin: dict[str, dict] = {}
    for r in rows:
        phase = r.get("phase")
        coin = r.get("coin")
        if not coin:
            continue
        emitted = r.get("emitted_at")
        if phase == "per_coin_start":
            headline = r.get("headline") or ""
            m = headline_re.search(headline)
            tf_from_headline = m.group(2) if m else None
            try:
                idx = int(m.group(3)) if m else None
                total = int(m.group(4)) if m else None
            except (TypeError, ValueError):
                idx = total = None
            slc = {
                "coin": coin,
                "timeframe": tf_from_headline or r.get("timeframe"),
                "index": idx,
                "total": total,
                "started_at": emitted,
                "ended_at": None,
                "elapsed_sec": None,
                "status": "running",
            }
            # If a previous start for this coin never closed (rare but
            # possible after a crash), flush it as an unfinished slice
            # so the timeline still shows it.
            prev = open_by_coin.pop(coin, None)
            if prev is not None:
                slices.append(prev)
            open_by_coin[coin] = slc
        elif phase in {"per_coin_done", "per_coin_skipped"}:
            slc = open_by_coin.pop(coin, None)
            if slc is None:
                continue
            slc["ended_at"] = emitted
            slc["status"] = "done" if phase == "per_coin_done" else "skipped"
            start_ts = _parse_iso(slc["started_at"])
            end_ts = _parse_iso(emitted)
            if start_ts and end_ts:
                slc["elapsed_sec"] = max(
                    0.0, (end_ts - start_ts).total_seconds()
                )
            slices.append(slc)
    # Append any still-running slice so the live "fitting" bar renders.
    for slc in open_by_coin.values():
        slices.append(slc)
    # Stable sort by start time so the chart timeline is monotonic.
    slices.sort(key=lambda s: s.get("started_at") or "")
    # Cap to the most recent ~200 slices to keep the payload bounded.
    slices = slices[-200:]

    # Detect idle gaps > 60s between consecutive slices (trainer was slow
    # or stuck between coins). These are highlighted in the chart so it's
    # obvious where the run lost wall-clock time vs. fitting.
    idle_gaps: list[dict] = []
    for prev_slc, next_slc in zip(slices, slices[1:]):
        prev_end = _parse_iso(prev_slc.get("ended_at"))
        next_start = _parse_iso(next_slc.get("started_at"))
        if prev_end is None or next_start is None:
            continue
        gap = (next_start - prev_end).total_seconds()
        if gap > 60.0:
            idle_gaps.append({
                "from": prev_slc.get("ended_at"),
                "to": next_slc.get("started_at"),
                "duration_sec": gap,
                "after_coin": prev_slc.get("coin"),
                "before_coin": next_slc.get("coin"),
            })

    last_phase = latest.get("phase")
    run_finished = last_phase in {
        # Terminal phases actually emitted by the campaign orchestrator
        # (`run_full_training_campaign.py`) and by `run_training` in
        # `app/training/train.py`. Anything outside this set keeps the
        # currently-fitting / stale view live.
        "campaign_done", "campaign_failed",
        "final_summary", "training_done",
    }
    if run_finished:
        currently_fitting = None

    return {
        "status": "ok",
        "latest_emitted_at": latest_at,
        "latest_phase": last_phase,
        "latest_status": latest.get("status"),
        "latest_headline": latest.get("headline"),
        "stale": stale,
        "stale_seconds": stale_seconds,
        "current_timeframe": current_timeframe,
        "currently_fitting": currently_fitting,
        "run_finished": run_finished,
        "recent": recent,
        "slices": slices,
        "idle_gaps": idle_gaps,
    }


@app.get("/ml/dataset-freshness")
async def dataset_freshness() -> dict:
    """Read-only proxy of ``models/datasets/_freshness_status.json``
    for the freshness dashboard panel (Task #559 — adds per-tf
    ``bytes_on_disk``, ``total_bytes_on_disk``, and a
    ``cache_size_history`` sparkline series so operators can confirm
    the cache is staying capped at the documented ~14-day footprint
    and spot per-tf bloat without `du -sh`'ing the parquet dir).

    Returns:
      ``{"status": "missing"}`` if the scheduler has never written
      the file (e.g. dataset-refresher workflow hasn't started yet).
      ``{"status": "error", "error": ...}`` if the file exists but
      cannot be parsed.
      Otherwise the parsed status dict, augmented with a top-level
      ``status: "ok"`` so the dashboard can branch cleanly.
    """
    status_path = REGISTRY_ROOT / "datasets" / "_freshness_status.json"
    if not status_path.exists():
        return {"status": "missing"}
    try:
        payload = json.loads(status_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "dataset_freshness_read_failed",
            extra={"error": str(exc)},
        )
        return {"status": "error", "error": str(exc)}
    payload.setdefault("status", "ok")
    return payload


@app.get("/ml/admin/fast-loop/status")
async def fast_loop_status() -> dict:
    """Read-only snapshot of the fast-loop scheduler so operators can see
    when the meta head was last refreshed independently of the slow loop.
    Mirrors /ml/admin/auto-retrain/status; no auth (read-only state).
    """
    fast_report_path = REGISTRY_ROOT / "fast_loop_report.json"
    fast_report: Optional[dict] = None
    if fast_report_path.exists():
        try:
            fast_report = json.loads(fast_report_path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("fast_loop_report_read_failed", extra={"error": str(exc)})
    return {
        **_fast_loop_state,
        "in_progress": _fast_loop_lock.locked(),
        "fast_loop_report": fast_report,
    }


# Task #615 — surface the per-slice live-gated replay block from the
# most recent campaign's `phase7_summary.json` so the trading dashboard
# can render the four-way verdict pill (bleeding/dormant/tradeable/
# inconclusive), loose-vs-live PnL, and the dominant rejection reason
# without an operator opening the run folder by hand. The block itself
# is produced by `scripts/run_full_training_campaign.py` (Task #613).
@app.get("/ml/training/live-gated-replay")
async def training_live_gated_replay() -> dict:
    """Return the `live_gated_replay` block from the most recent
    `models/training_run_<TS>/phase7_summary.json`, plus a small
    pointer to the source folder so the operator can audit the
    underlying file.

    Status codes (always HTTP 200, surfaced via `status` for the UI):
      * `ok`               — populated `per_slice` block returned
      * `empty`            — newest summary has no `per_slice` entries
                             (campaign ran before #613 added the block,
                             or `phase7` has not been emitted yet)
      * `missing`          — no `training_run_<TS>/phase7_summary.json`
                             on disk yet
      * `error`            — file existed but couldn't be parsed
    """
    if not REGISTRY_ROOT.exists():
        return {"status": "missing"}
    candidates: list[tuple[str, Path]] = []
    try:
        for child in REGISTRY_ROOT.iterdir():
            if not child.is_dir():
                continue
            if not child.name.startswith("training_run_"):
                continue
            summary = child / "phase7_summary.json"
            if summary.exists():
                candidates.append((child.name, summary))
    except OSError as exc:
        logger.warning(
            "training_live_gated_replay_scan_failed",
            extra={"error": str(exc)},
        )
        return {"status": "error", "error": str(exc)}
    if not candidates:
        return {"status": "missing"}
    # `training_run_<UTC ISO timestamp>` sorts lexicographically the same
    # as chronologically, so the lexicographic max is the newest run.
    candidates.sort(key=lambda item: item[0])
    run_name, summary_path = candidates[-1]
    try:
        payload = json.loads(summary_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "training_live_gated_replay_read_failed",
            extra={"error": str(exc), "path": str(summary_path)},
        )
        return {
            "status": "error",
            "error": str(exc),
            "run_dir": run_name,
        }
    block = payload.get("live_gated_replay") or {}
    per_slice = block.get("per_slice") or {}
    status = "ok" if per_slice else "empty"
    return {
        "status": status,
        "run_dir": run_name,
        "generated_at": payload.get("generated_at"),
        "run_started_iso": block.get("run_started_iso"),
        "per_slice": per_slice,
        "verdict_counts": block.get("verdict_counts") or {},
        "bleeding_slices": block.get("bleeding_slices") or [],
        "dormant_slices": block.get("dormant_slices") or [],
        "tradeable_slices": block.get("tradeable_slices") or [],
    }
