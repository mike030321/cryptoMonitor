"""Market Meta-Brain supervisory layer mounted on ml-engine.

The brain is a pure-stdlib Python package vendored under
`artifacts/ml-engine/vendor/market_meta_brain/`. It is deterministic and
bounded: it does NOT predict price and does NOT place trades. It sits
above the quant engine and emits supervisory directives (trust,
allocation, caution, suppression, defensive mode, exploration budget).

This module exposes two HTTP endpoints on the ml-engine FastAPI app:

  POST /ml/meta-brain/evaluate        — per-tick: accepts one telemetry
                                        batch across all coin/timeframe
                                        slices; returns directive + tick_id
  POST /ml/meta-brain/record-outcome  — per-close: accepts a realized
                                        outcome + the tick_id that
                                        authorized the entry; feeds
                                        bounded learning

Backed by a module-level singleton `MarketMetaBrainService` with disk
checkpointing (trust model, regime prototypes, episodic memory). The
singleton survives hot reloads (the state is rehydrated at startup from
`artifacts/ml-engine/models/meta_brain_state/`).

If the checkpoint is missing or malformed, the singleton starts with
neutral state and logs the reason — the trading path is never blocked.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Deque

from fastapi import APIRouter, HTTPException

# The package is vendored under ml-engine/vendor/market_meta_brain/src. We
# prepend it to sys.path so the pure-stdlib package imports cleanly
# without requiring an editable pip install.
_VENDOR_SRC = (
    Path(__file__).resolve().parent.parent
    / "vendor"
    / "market_meta_brain"
    / "src"
)
if str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))

from market_meta_brain.domain.types import GovernanceOutcome  # noqa: E402
from market_meta_brain.integration.quant_bridge import QuantBridge  # noqa: E402
from market_meta_brain.learning.trust_model import (  # noqa: E402
    FamilyTrustState,
)
from market_meta_brain.utils.math_utils import clamp  # noqa: E402
from market_meta_brain.runtime.checkpointing import Checkpointer  # noqa: E402
from market_meta_brain.runtime.logging import JsonlLogger  # noqa: E402
from market_meta_brain.runtime.service import MarketMetaBrainService  # noqa: E402

from .logging_config import logger

_STATE_ROOT = (
    Path(__file__).resolve().parent.parent / "models" / "meta_brain_state"
)
_STATE_ROOT.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _STATE_ROOT / "directives.jsonl"
# Capped tick cache so record_outcome can resolve the directive that
# authorized a given entry. LRU eviction at TICK_CACHE_SIZE. Lost on
# process restart, which is acceptable for bounded learning — any trades
# opened before a restart simply don't feed the learning loop.
TICK_CACHE_SIZE = 2048

_lock = threading.RLock()
_bridge = QuantBridge()
_service: MarketMetaBrainService | None = None
_tick_cache: "OrderedDict[str, Any]" = OrderedDict()

# Task #567 — per-timeframe role partitioning. The TS adapter (#550)
# stamps every slice payload with `slice_role` (one of
# `trade | shadow | context | disabled`) sourced from
# `shared/timeframe-roles.json`. The api-server is expected to pass
# the same field through on `/record-outcome` so the trust model only
# learns from outcomes the operator has authorized as trade-roled.
#
#   - `trade`    → existing trust update logic.
#   - `shadow`   → store outcome for shadow-mode analysis; never
#                  touch trust scores.
#   - `context`  → store outcome for governance/context analysis;
#                  never touch trust scores.
#   - `disabled` → reject + emit a structured warn. The upstream
#                  is broken if a `disabled` role ever arrives here.
#
# A back-compat default of `trade` is applied when the field is
# absent (a one-time per-process `[ROLE_BACKCOMPAT_DEFAULT]` warn
# fires the first time so stale callers are visible). The hard rule
# is: never silently accept missing roles.
VALID_SLICE_ROLES = ("trade", "shadow", "context", "disabled")
_DEFAULT_BACKCOMPAT_ROLE = "trade"
OUTCOMES_HISTORY_CAP = 2048
INPUTS_BY_ROLE_PERSIST_INTERVAL_S = 1.0

# Task #578 — short trend buffer so the dashboard can render a
# 24h-by-hour sparkline of role arrivals next to the cumulative
# totals. Each entry is keyed by epoch_hour (UTC, integer hours
# since the unix epoch) → Counter[role -> n_arrivals_in_that_hour].
# Buckets older than ROLE_HOURLY_RETENTION_HOURS are pruned on
# every read/write. Pure in-memory; lost on restart by design
# (the cumulative `_inputs_by_role` is what we persist).
ROLE_HOURLY_RETENTION_HOURS = 25
ROLE_TREND_WINDOW_HOURS = 24

_inputs_by_role: "Counter[str]" = Counter({r: 0 for r in VALID_SLICE_ROLES})
_role_hourly_buckets: "OrderedDict[int, Counter[str]]" = OrderedDict()
_shadow_outcomes: "Deque[dict]" = deque(maxlen=OUTCOMES_HISTORY_CAP)
_context_outcomes: "Deque[dict]" = deque(maxlen=OUTCOMES_HISTORY_CAP)
_inputs_by_role_dirty = False
_inputs_by_role_last_write_ts = 0.0
# Task #583 — debounced persistence for the hourly bucket map so the
# 24h sparkline survives an ml-engine restart instead of dropping to
# zeros for up to a day. Mirrors the cumulative-counter pattern above.
_role_hourly_buckets_dirty = False
_role_hourly_buckets_last_write_ts = 0.0
_role_backcompat_warned = False


_NUMERIC_SLICE_FIELDS = (
    "edge",
    "confidence",
    "calibrated_confidence",
    "risk_score",
    "recent_accuracy",
    "pnl_state",
    "drawdown_state",
    "disagreement",
    "prediction_error",
    "volatility",
    "correlation_shift",
    "exposure",
    "turnover",
    "slippage_bps",
)
# Task #390 — Strategy Lab benchmark telemetry. Governance-only.
# Spec invariant: benchmark alone must never push `defensive_mode`
# past "soft". Enforced by re-evaluating without the benchmark slice
# and demoting "hard" → "soft" when the no-benchmark baseline was
# below "hard". Never reaches the predictor or /ml/decide.
_BENCHMARK_NUMERIC_FIELDS = (
    "aiReturn7d",
    "bestBaselineReturn7d",
    "relativeAlpha7d",
    "relativeAlpha14d",
    "drawdownRatioVsBest",
)
_BENCHMARK_FAMILY = "benchmark"


_NUMERIC_PORTFOLIO_FIELDS = (
    "total_drawdown",
    "realized_vol",
    "concentration",
    "leverage",
    "liquidity_stress",
    "correlation_shift",
    "active_risk_budget",
    "kill_switch_distance",
)


def _normalize_payload_in_place(payload: dict) -> None:
    """Task #381 step 7 — replace `null` numeric values in slices and
    portfolio with 0.0 and append a `missing:<field>` flag. Vendored
    `QuantSliceTelemetry` / `PortfolioTelemetry` dataclasses require
    `float` (no `Optional`), and we do not want to fork the package.
    The brain's bounded plasticity is robust to sparse signal — what
    matters is that the missing flag is recorded so trust updates can
    be down-weighted. Operates in place; never raises.

    Task #567 — additionally strips the `slice_role` pass-through
    field (added by the TS adapter in #550) from each slice before
    handing the payload to QuantBridge. The vendored
    `QuantSliceTelemetry` is a strict dataclass that 400s on unknown
    kwargs; rather than fork the package to model a TS-only
    pass-through field, we strip it here. The role is consumed on
    `/record-outcome` instead, where it actually gates trust updates.
    """
    if not isinstance(payload, dict):
        return
    slices = payload.get("slices")
    if isinstance(slices, list):
        for s in slices:
            if not isinstance(s, dict):
                continue
            # Strip the TS-only pass-through. QuantSliceTelemetry has
            # no slot for it; record-outcome carries the role for
            # trust gating instead.
            s.pop("slice_role", None)
            flags = s.setdefault("anomaly_flags", [])
            if not isinstance(flags, list):
                flags = []
                s["anomaly_flags"] = flags
            for f in _NUMERIC_SLICE_FIELDS:
                v = s.get(f, None)
                if v is None:
                    s[f] = 0.0
                    flags.append(f"missing:{f}")
    p = payload.get("portfolio")
    if isinstance(p, dict):
        flags = p.setdefault("anomaly_flags", [])
        if not isinstance(flags, list):
            flags = []
            p["anomaly_flags"] = flags
        for f in _NUMERIC_PORTFOLIO_FIELDS:
            v = p.get(f, None)
            if v is None:
                p[f] = 0.0
                flags.append(f"missing:{f}")


def _coerce_benchmark(raw: Any) -> dict | None:
    """Defensive read of the optional benchmark block. Returns a dict
    with the canonical keys cast to floats / bools, or None if the
    payload is missing / malformed / explicitly stale. Never raises.
    """
    if not isinstance(raw, dict):
        return None
    try:
        out: dict = {f: float(raw.get(f, 0.0) or 0.0) for f in _BENCHMARK_NUMERIC_FIELDS}
        out["sustainedUnderperformance"] = bool(raw.get("sustainedUnderperformance", False))
        out["sampleCount"] = int(raw.get("sampleCount", 0) or 0)
        out["stale"] = bool(raw.get("stale", False))
    except (TypeError, ValueError):
        return None
    if out["stale"]:
        return None
    return out


def _benchmark_slice_from(bm: dict) -> dict:
    """Synthetic QuantSliceTelemetry mapping the benchmark block onto a
    real `benchmark` family in `state.family_states`. The planner's
    weighted aggregator handles it like any other family.

    Risk fields are intentionally tame: `risk_pressure` (the only path
    to `defensive_mode == "hard"` via SafetyGuardrails) is computed
    purely from `portfolio` fields in the vendor planner and is
    therefore unaffected by this slice — the soft-cap requirement is
    satisfied by construction.
    """
    alpha14 = clamp(bm["relativeAlpha14d"], -1.0, 1.0)
    alpha7 = clamp(bm["relativeAlpha7d"], -1.0, 1.0)
    excess_dd = max(0.0, min(2.0, bm["drawdownRatioVsBest"] - 1.0))
    conf = clamp(abs(alpha14) / 0.05, 0.0, 1.0)
    flags = (
        ["benchmark_sustained_underperformance"]
        if bm["sustainedUnderperformance"]
        else []
    )
    return {
        "coin": "__benchmark__",
        "timeframe": "benchmark",
        "strategy_family": _BENCHMARK_FAMILY,
        "edge": float(alpha14),
        "confidence": float(conf),
        "calibrated_confidence": float(conf),
        "risk_score": float(clamp(excess_dd * 0.5, 0.0, 1.0)),
        "recent_accuracy": float(clamp(0.5 + alpha14, 0.0, 1.0)),
        "pnl_state": float(alpha14),
        "drawdown_state": float(min(1.0, excess_dd)),
        "disagreement": float(clamp(abs(alpha14 - alpha7), 0.0, 1.0)),
        "prediction_error": float(max(0.0, -alpha7)),
        "regime": "benchmark",
        "volatility": float(clamp(abs(alpha14) * 2.0, 0.0, 1.0)),
        "correlation_shift": 0.0,
        "exposure": 0.0,
        "turnover": 0.0,
        "slippage_bps": 0.0,
        "anomaly_flags": flags,
    }


def _benchmark_outcome_from(bm: dict) -> GovernanceOutcome:
    """Map the benchmark block to a `GovernanceOutcome` for the
    standard `record_outcome` learning loop."""
    alpha14 = clamp(bm["relativeAlpha14d"], -1.0, 1.0)
    alpha7 = clamp(bm["relativeAlpha7d"], -1.0, 1.0)
    excess_dd = max(0.0, bm["drawdownRatioVsBest"] - 1.0)
    stability = clamp(1.0 - min(2.0, excess_dd) * 0.5, 0.0, 1.0)
    return GovernanceOutcome(
        realized_pnl=float(alpha14),
        realized_drawdown=float(min(1.0, excess_dd)),
        realized_stability=float(stability),
        turnover_cost=0.0,
        action_churn=0.0,
        correct_defense=1.0 if bm["sustainedUnderperformance"] else 0.0,
        correct_suppression=0.0,
        missed_edge_cost=float(max(0.0, -alpha7)),
    )


def _apply_benchmark_governance(directive: Any, bm: dict) -> None:
    """Spec Step 3: feed the benchmark outcome through the SAME
    `record_outcome` loop the api-server uses on paper-trade close.
    `record_outcome` pushes a GovernanceEpisode, updates regime memory,
    and broadcasts `learn_from_outcome` across `state.family_states`
    — exactly what the spec calls "trims trust on losing families".

    Soft cap is satisfied by construction (the synthetic family slice
    cannot push `risk_pressure` past the guardrail's hard threshold,
    because `risk_pressure` is portfolio-only). We additionally nudge
    `defensive_mode` `off → soft` on sustained negative 14d alpha so
    the dashboard reflects the governance bias.
    """
    if getattr(directive, "meta_state", None) is None:
        return
    outcome = _benchmark_outcome_from(bm)
    _service.record_outcome(directive, outcome)

    if (
        getattr(directive, "defensive_mode", "off") == "off"
        and bm["sustainedUnderperformance"]
        and bm["relativeAlpha14d"] < 0
    ):
        directive.defensive_mode = "soft"
        codes = list(getattr(directive, "reason_codes", []))
        codes.append("benchmark_alpha_negative")
        directive.reason_codes = codes


def _inputs_by_role_path() -> Path:
    """Resolve the on-disk counter file lazily so tests that swap
    ``_STATE_ROOT`` via ``monkeypatch`` get the patched value."""
    return _STATE_ROOT / "inputs_by_role.json"


def _persist_inputs_by_role(force: bool = False) -> None:
    """Persist the per-role arrival counters to disk with a 1Hz max
    write rate. Called from `_record_role_input` and forced from
    ``shutdown()``. Best-effort — never raises into the request path.
    """
    global _inputs_by_role_dirty, _inputs_by_role_last_write_ts
    with _lock:
        if not _inputs_by_role_dirty and not force:
            return
        now = time.monotonic()
        if not force and (now - _inputs_by_role_last_write_ts) < INPUTS_BY_ROLE_PERSIST_INTERVAL_S:
            return
        snapshot = {role: int(_inputs_by_role.get(role, 0)) for role in VALID_SLICE_ROLES}
    try:
        path = _inputs_by_role_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "meta_brain_inputs_by_role_persist_failed",
            extra={"error": str(exc)},
        )
        return
    with _lock:
        _inputs_by_role_dirty = False
        _inputs_by_role_last_write_ts = time.monotonic()


def _load_inputs_by_role() -> dict:
    """Restore the per-role counters at boot. Returns a small summary
    dict for the boot log. Missing or malformed files reset to zero
    silently — this is observability, not authoritative state."""
    path = _inputs_by_role_path()
    if not path.exists():
        return {"restored": False, "reason": "no_checkpoint"}
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"restored": False, "reason": f"load_failed: {exc}"}
    if not isinstance(snapshot, dict):
        return {"restored": False, "reason": "bad_snapshot"}
    with _lock:
        _inputs_by_role.clear()
        for role in VALID_SLICE_ROLES:
            try:
                _inputs_by_role[role] = int(snapshot.get(role, 0) or 0)
            except (TypeError, ValueError):
                _inputs_by_role[role] = 0
    return {"restored": True, "totals": dict(snapshot)}


def _role_hourly_buckets_path() -> Path:
    """Resolve the on-disk hourly-bucket file lazily so tests that swap
    ``_STATE_ROOT`` via ``monkeypatch`` get the patched value."""
    return _STATE_ROOT / "role_hourly_buckets.json"


def _persist_role_hourly_buckets(force: bool = False) -> None:
    """Persist the per-role hourly bucket map to disk with a 1Hz max
    write rate. Called from `_record_role_input` and forced from
    ``shutdown()``. Best-effort — never raises into the request path.

    Task #583 — without this, the dashboard's 24h sparkline drops to
    zero for up to a day after every ml-engine restart, which defeats
    the "spot a sudden spike" purpose of the panel.
    """
    global _role_hourly_buckets_dirty, _role_hourly_buckets_last_write_ts
    with _lock:
        if not _role_hourly_buckets_dirty and not force:
            return
        now = time.monotonic()
        if not force and (
            now - _role_hourly_buckets_last_write_ts
        ) < INPUTS_BY_ROLE_PERSIST_INTERVAL_S:
            return
        snapshot = {
            "version": 1,
            "buckets": [
                {
                    "hour": int(h),
                    "counts": {
                        role: int(bucket.get(role, 0))
                        for role in VALID_SLICE_ROLES
                    },
                }
                for h, bucket in _role_hourly_buckets.items()
            ],
        }
    try:
        path = _role_hourly_buckets_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "meta_brain_role_hourly_buckets_persist_failed",
            extra={"error": str(exc)},
        )
        return
    with _lock:
        _role_hourly_buckets_dirty = False
        _role_hourly_buckets_last_write_ts = time.monotonic()


def _load_role_hourly_buckets() -> dict:
    """Restore the per-role hourly bucket map at boot. Buckets older
    than ``ROLE_HOURLY_RETENTION_HOURS`` are dropped on load so a long
    downtime does not resurface stale data as "fresh" sparkline bars.
    Missing or malformed files reset to empty silently — this is
    observability, not authoritative state.
    """
    path = _role_hourly_buckets_path()
    if not path.exists():
        return {"restored": False, "reason": "no_checkpoint"}
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"restored": False, "reason": f"load_failed: {exc}"}
    if not isinstance(snapshot, dict):
        return {"restored": False, "reason": "bad_snapshot"}
    raw_buckets = snapshot.get("buckets")
    if not isinstance(raw_buckets, list):
        return {"restored": False, "reason": "bad_buckets"}
    now_hour = int(time.time()) // 3600
    cutoff = now_hour - ROLE_HOURLY_RETENTION_HOURS + 1
    parsed: list[tuple[int, Counter]] = []
    for entry in raw_buckets:
        if not isinstance(entry, dict):
            continue
        try:
            hour = int(entry.get("hour"))
        except (TypeError, ValueError):
            continue
        if hour < cutoff or hour > now_hour:
            # Drop buckets outside the retention window (long downtime)
            # and any future-dated buckets from a clock skew.
            continue
        counts_raw = entry.get("counts") or {}
        if not isinstance(counts_raw, dict):
            continue
        bucket: Counter = Counter()
        for role in VALID_SLICE_ROLES:
            try:
                bucket[role] = int(counts_raw.get(role, 0) or 0)
            except (TypeError, ValueError):
                bucket[role] = 0
        parsed.append((hour, bucket))
    parsed.sort(key=lambda kv: kv[0])
    with _lock:
        _role_hourly_buckets.clear()
        for hour, bucket in parsed:
            _role_hourly_buckets[hour] = bucket
    return {
        "restored": True,
        "buckets_loaded": len(parsed),
        "oldest_hour": parsed[0][0] if parsed else None,
        "newest_hour": parsed[-1][0] if parsed else None,
    }


def _normalize_slice_role(raw: Any, *, source: str) -> str | None:
    """Resolve the incoming `slice_role` to one of the four valid
    roles. Returns ``None`` when the role is invalid (the caller
    rejects the request); a missing role triggers the back-compat
    default of ``trade`` with a one-time per-process warn.
    """
    global _role_backcompat_warned
    if raw is None or raw == "":
        with _lock:
            already_warned = _role_backcompat_warned
            _role_backcompat_warned = True
        if not already_warned:
            logger.warning(
                "[ROLE_BACKCOMPAT_DEFAULT] meta-brain record-outcome "
                "received without slice_role; defaulting to 'trade'. "
                "Upstream callers should be updated to send slice_role "
                "from shared/timeframe-roles.json.",
                extra={
                    "event": "role_backcompat_default",
                    "source": source,
                    "default_role": _DEFAULT_BACKCOMPAT_ROLE,
                },
            )
        return _DEFAULT_BACKCOMPAT_ROLE
    if isinstance(raw, str) and raw in VALID_SLICE_ROLES:
        return raw
    return None


def _record_role_input(role: str) -> None:
    """Increment the per-role arrival counter and schedule a debounced
    persist. Safe to call from concurrent request handlers.

    Task #578 — also bump the matching hourly bucket so the dashboard
    can render a 24h sparkline of arrivals per role. The bucket map is
    pruned to the most recent ``ROLE_HOURLY_RETENTION_HOURS`` entries
    on each write, so its memory footprint is bounded and it cannot
    leak even under sustained load.
    """
    global _inputs_by_role_dirty, _role_hourly_buckets_dirty
    epoch_hour = int(time.time()) // 3600
    with _lock:
        _inputs_by_role[role] += 1
        _inputs_by_role_dirty = True
        bucket = _role_hourly_buckets.get(epoch_hour)
        if bucket is None:
            bucket = Counter()
            _role_hourly_buckets[epoch_hour] = bucket
        bucket[role] += 1
        # Prune: keep only the most recent retention window so the
        # dict never grows past a fixed bound. We also drop anything
        # explicitly older than the cutoff in case the process clock
        # jumps forward.
        cutoff = epoch_hour - ROLE_HOURLY_RETENTION_HOURS + 1
        while _role_hourly_buckets and next(iter(_role_hourly_buckets)) < cutoff:
            _role_hourly_buckets.popitem(last=False)
        _role_hourly_buckets_dirty = True
    _persist_inputs_by_role(force=False)
    _persist_role_hourly_buckets(force=False)


def _summarize_role_trend() -> dict:
    """Build the dashboard-facing 24h trend payload from the in-memory
    hourly bucket map. Returns:

    - ``inputs_by_role_24h``: ``{role: count_in_last_24h}``
    - ``inputs_by_role_hourly``: ordered list (oldest → newest, length
      ``ROLE_TREND_WINDOW_HOURS``) of
      ``{hour_start: ISO-8601, counts: {role: n}}`` so the dashboard
      can render a per-role sparkline without backfilling missing
      hours itself. Empty hours are zero-filled.
    - ``window_hours``: the trend window the counts cover.
    """
    now_hour = int(time.time()) // 3600
    cutoff = now_hour - ROLE_TREND_WINDOW_HOURS + 1
    totals: Counter = Counter({r: 0 for r in VALID_SLICE_ROLES})
    hourly: list[dict] = []
    with _lock:
        for h_offset in range(ROLE_TREND_WINDOW_HOURS):
            h = cutoff + h_offset
            bucket = _role_hourly_buckets.get(h)
            counts = {r: int(bucket.get(r, 0)) if bucket else 0 for r in VALID_SLICE_ROLES}
            for r, n in counts.items():
                totals[r] += n
            hourly.append(
                {
                    "hour_start": time.strftime(
                        "%Y-%m-%dT%H:00:00Z", time.gmtime(h * 3600)
                    ),
                    "counts": counts,
                }
            )
    return {
        "inputs_by_role_24h": {r: int(totals.get(r, 0)) for r in VALID_SLICE_ROLES},
        "inputs_by_role_hourly": hourly,
        "window_hours": ROLE_TREND_WINDOW_HOURS,
    }


def _hydrate_episodic_memory(service: MarketMetaBrainService) -> dict:
    """Task #467 step 7 — restore the full episodic-memory buffer from
    `episodic_memory.json` if a previous shutdown wrote one. The pre-
    Task #467 checkpoint format only persisted a 50-row reward summary
    which the buffer cannot reload from. Treat the legacy summary
    layout as "no buffer to restore" so a one-time migration is silent
    and lossless rather than fatal.
    """
    path = _STATE_ROOT / "episodic_memory.json"
    if not path.exists():
        return {"restored_episodes": 0, "reason": "no_checkpoint"}
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"restored_episodes": 0, "reason": f"load_failed: {exc}"}
    if not isinstance(snapshot, dict):
        return {"restored_episodes": 0, "reason": "bad_snapshot"}
    if "episodes" not in snapshot:
        # Pre-Task #467 summary-only checkpoint. Nothing to restore.
        return {"restored_episodes": 0, "reason": "legacy_summary_only"}
    try:
        restored = service.episodic_memory.load_state_dict(snapshot)
    except Exception as exc:  # noqa: BLE001
        return {"restored_episodes": 0, "reason": f"hydrate_failed: {exc}"}
    return {"restored_episodes": int(restored), "reason": "ok"}


def _hydrate_trust_model(service: MarketMetaBrainService) -> dict:
    """Task #381 step 10 — restore `trust_by_family` from the
    `trust_model.json` checkpoint. Other components (regime memory,
    episodic memory) deliberately start with neutral state on each
    boot — re-deriving them from one cycle of telemetry is cheap and
    avoids correctness drift from format changes. Returns a small
    summary dict for the boot log.
    """
    path = _STATE_ROOT / "trust_model.json"
    if not path.exists():
        return {"restored_families": 0, "reason": "no_checkpoint"}
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "restored_families": 0,
            "reason": f"load_failed: {exc}",
        }
    if not isinstance(snapshot, dict):
        return {"restored_families": 0, "reason": "bad_snapshot"}
    restored = 0
    for fam, raw in snapshot.items():
        if not isinstance(raw, dict):
            continue
        try:
            service.trust_model.trust_by_family[fam] = FamilyTrustState(
                trust=float(raw.get("trust", 1.0)),
                stability=float(raw.get("stability", 0.5)),
                exploration_eligibility=float(
                    raw.get("exploration_eligibility", 0.2)
                ),
                failure_streak=int(raw.get("failure_streak", 0)),
                recovery_score=float(raw.get("recovery_score", 0.5)),
                last_regime=str(raw.get("last_regime", "unknown")),
            )
            restored += 1
        except (TypeError, ValueError):
            continue
    return {"restored_families": restored, "reason": "ok"}


def _load_service() -> MarketMetaBrainService:
    """Hydrate the brain service from the on-disk checkpoint if any."""
    service = MarketMetaBrainService(logger=JsonlLogger(_LOG_PATH))
    trust_summary = _hydrate_trust_model(service)
    episodic_summary = _hydrate_episodic_memory(service)
    inputs_by_role_summary = _load_inputs_by_role()
    role_hourly_buckets_summary = _load_role_hourly_buckets()
    logger.info(
        "meta_brain_hydrate",
        extra={
            "trust": trust_summary,
            "regime_memory": "starts_neutral",
            "episodic_memory": episodic_summary,
            "inputs_by_role": inputs_by_role_summary,
            "role_hourly_buckets": role_hourly_buckets_summary,
        },
    )
    return service


def checkpoint() -> None:
    """Persist the brain's current learned state to disk. Best-effort."""
    with _lock:
        if _service is None:
            return
        try:
            cp = Checkpointer(_STATE_ROOT)
            cp.save_json(
                "trust_model",
                {
                    fam: {
                        "trust": fts.trust,
                        "stability": fts.stability,
                        "exploration_eligibility": fts.exploration_eligibility,
                        "failure_streak": fts.failure_streak,
                        "recovery_score": fts.recovery_score,
                        "last_regime": fts.last_regime,
                    }
                    for fam, fts in _service.trust_model.trust_by_family.items()
                },
            )
            cp.save_json(
                "regime_memory", _service.regime_memory.state_dict()
            )
            # Task #467: persist the full episodic-memory buffer (not
            # just a 50-row reward summary) so a restart actually
            # reloads what evaluate() has learned.
            cp.save_json(
                "episodic_memory", _service.episodic_memory.state_dict()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "meta_brain_checkpoint_failed", extra={"error": str(exc)}
            )


def init() -> None:
    """Lifespan-time init. Safe to call multiple times."""
    global _service
    with _lock:
        if _service is None:
            try:
                _service = _load_service()
                logger.info(
                    "meta_brain_initialized",
                    extra={"state_root": str(_STATE_ROOT)},
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "meta_brain_init_failed", extra={"error": str(exc)}
                )
                raise


def shutdown() -> None:
    checkpoint()
    # Force-flush the per-role arrival counters so any increments
    # since the last debounced write are durable across restarts.
    _persist_inputs_by_role(force=True)
    # Task #583 — also force-flush the hourly bucket map so the
    # dashboard's 24h sparkline picks up where it left off after a
    # restart instead of phantoming a 24h drop to zero.
    _persist_role_hourly_buckets(force=True)


router = APIRouter(prefix="/meta-brain", tags=["meta-brain"])


@router.get("/health")
def health() -> dict:
    with _lock:
        return {
            "ok": _service is not None,
            "tick_cache_size": len(_tick_cache),
            "state_root": str(_STATE_ROOT),
        }


@router.post("/evaluate")
def evaluate(payload: dict) -> dict:
    """Accept a telemetry batch (matching the QuantBridge contract) and
    return the directive + a tick_id the caller will send back at
    record_outcome time.
    """
    if _service is None:
        raise HTTPException(status_code=503, detail="meta_brain_not_initialized")
    # Task #381 step 7 — accept null numeric fields for honest
    # telemetry. Mutates in place; safe because FastAPI gives us a
    # fresh dict per request.
    _normalize_payload_in_place(payload)
    # Task #390 — pop the optional `benchmark` block, then inject a
    # synthetic slice for it BEFORE QuantBridge.from_payload so the
    # planner sees `benchmark` as a real family in the weighted
    # aggregator (spec: "treat it like any other family"). The raw
    # benchmark struct never reaches predictor / `/ml/decide`. Trust
    # learning happens via the existing `record_outcome` loop. The
    # soft-cap requirement is met by construction: `risk_pressure`
    # (the only path to `defensive_mode == "hard"` via guardrails)
    # is portfolio-only; the synthetic family slice cannot push it.
    benchmark = payload.pop("benchmark", None) if isinstance(payload, dict) else None
    bm = _coerce_benchmark(benchmark)
    if bm is not None and isinstance(payload, dict):
        slices = payload.get("slices")
        if isinstance(slices, list):
            slices.append(_benchmark_slice_from(bm))
    try:
        batch = _bridge.from_payload(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"invalid_payload: {exc}"
        ) from exc

    tick_id = str(uuid.uuid4())
    with _lock:
        directive = _service.evaluate(batch)
        if bm is not None:
            _apply_benchmark_governance(directive, bm)
        # Keep the directive around so record_outcome can resolve the
        # same object the package expects (record_outcome needs
        # directive.meta_state to be the one produced by evaluate()).
        _tick_cache[tick_id] = directive
        while len(_tick_cache) > TICK_CACHE_SIZE:
            _tick_cache.popitem(last=False)

    out = directive.to_dict() if hasattr(directive, "to_dict") else {
        "trust_multiplier": dict(directive.trust_multiplier),
        "allocation_weight": dict(directive.allocation_weight),
        "caution_level": float(directive.caution_level),
        "exploration_budget": float(directive.exploration_budget),
        "suppress_signal": bool(directive.suppress_signal),
        "defensive_mode": str(directive.defensive_mode),
        "suppressed_families": list(directive.suppressed_families),
        "paused_slices": list(directive.paused_slices),
        "reason_codes": list(directive.reason_codes),
    }
    # Never leak meta_state — it's a domain object, not JSON-friendly
    # for the api-server caller. The tick_id is the handle.
    out.pop("meta_state", None)
    out["tick_id"] = tick_id
    return out


@router.post("/record-outcome")
def record_outcome(payload: dict) -> dict:
    """Feed a realized outcome back into bounded learning.

    Payload shape:
      {
        "tick_id": str (required — the id returned by /evaluate),
        "timestamp": str (ISO-8601, optional),
        "slice_role": "trade" | "shadow" | "context" | "disabled"
            (Task #567 — gates whether trust updates fire. Missing
            triggers a one-time per-process warn and back-compat
            default of "trade"; invalid values are rejected.),
        "slice_id": str (optional — for `disabled` warn surfacing),
        "outcome": {
            "realized_pnl": float,
            "realized_drawdown": float,
            "realized_stability": float,
            "turnover_cost": float,
            "action_churn": float,
            "correct_defense": float,
            "correct_suppression": float,
            "missed_edge_cost": float
        }
      }
    """
    if _service is None:
        raise HTTPException(status_code=503, detail="meta_brain_not_initialized")
    tick_id = payload.get("tick_id")
    if not tick_id:
        raise HTTPException(status_code=400, detail="missing_tick_id")

    # Task #567 — resolve `slice_role` BEFORE touching any state. An
    # invalid role is a contract violation and we 400 the caller; a
    # missing role gets the back-compat default with a one-time warn
    # so the operator can find and patch stale upstreams.
    role = _normalize_slice_role(payload.get("slice_role"), source="record_outcome")
    if role is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "invalid_slice_role: must be one of "
                f"{list(VALID_SLICE_ROLES)}"
            ),
        )
    _record_role_input(role)

    out = payload.get("outcome") or {}
    try:
        outcome = GovernanceOutcome(
            realized_pnl=float(out.get("realized_pnl", 0.0)),
            realized_drawdown=float(out.get("realized_drawdown", 0.0)),
            realized_stability=float(out.get("realized_stability", 0.5)),
            turnover_cost=float(out.get("turnover_cost", 0.0)),
            action_churn=float(out.get("action_churn", 0.0)),
            correct_defense=float(out.get("correct_defense", 0.0)),
            correct_suppression=float(out.get("correct_suppression", 0.0)),
            missed_edge_cost=float(out.get("missed_edge_cost", 0.0)),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid_outcome: {exc}"
        ) from exc

    # Task #567 — `disabled` should never arrive here. If it does,
    # the upstream is leaking outcomes from a timeframe the operator
    # has explicitly turned off. Emit a structured warn and reject;
    # the trust model and the shadow/context stores stay untouched.
    if role == "disabled":
        logger.warning(
            "[disabled_outcome_received] meta-brain received an outcome "
            "for a disabled-role slice; upstream is leaking",
            extra={
                "event": "disabled_outcome_received",
                "tick_id": tick_id,
                "slice_id": payload.get("slice_id"),
                "source": "record_outcome",
            },
        )
        return {"ok": False, "reason": "disabled_role_rejected"}

    with _lock:
        directive = _tick_cache.get(tick_id)
    if directive is None:
        # Trade was opened before the brain booted (or older than
        # TICK_CACHE_SIZE). Silently no-op — bounded learning can't
        # learn from trades whose authorizing directive has evicted.
        return {"ok": False, "reason": "tick_id_not_in_cache"}

    # Task #567 — `shadow` and `context` outcomes are stored for
    # downstream analysis but MUST NOT touch trust scores. The trust
    # model only learns from outcomes the operator has authorized as
    # trade-roled (per shared/timeframe-roles.json).
    if role in ("shadow", "context"):
        snapshot = {
            "tick_id": tick_id,
            "timestamp": payload.get("timestamp"),
            "slice_id": payload.get("slice_id"),
            "outcome": asdict(outcome),
        }
        with _lock:
            store = _shadow_outcomes if role == "shadow" else _context_outcomes
            store.append(snapshot)
        return {"ok": True, "role": role, "trust_updated": False}

    # Default path: role == "trade" → existing trust update logic.
    with _lock:
        try:
            reward = _service.record_outcome(
                directive, outcome, payload.get("timestamp")
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "meta_brain_record_outcome_failed",
                extra={"tick_id": tick_id, "error": str(exc)},
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"ok": True, "role": role, "trust_updated": True, "reward": float(reward)}


@router.get("/last_replay")
def last_replay() -> dict:
    """Task #490 — surface the most recent scheduled replay run so ops
    can answer "did the supervisory brain warm up tonight, and on what
    evidence?" without scraping logs or walking the sandbox tree.

    Returns BOTH the latest run (committed or below-threshold) AND
    the latest committed run as separate pointers:

      * ``last_run`` — the most recent tick's manifest summary,
        regardless of whether the replay's three-gate was satisfied.
        Lets ops see "the daemon ran tonight, the gate said not yet
        ready, and here's the evidence" during the early days after
        the post-#444 cutover when the gates aren't yet satisfiable.
      * ``last_committed_run`` — the most recent tick whose manifest
        was actually promoted into the canonical state dir. This is
        the "latest manifest" in the strictest sense — by definition
        it always corresponds to the contents of
        ``models/meta_brain_state/`` that the live supervisory brain
        is loading from.

    Either pointer can be ``None`` independently:
      * ``last_run is None``  → daemon has never run.
      * ``last_committed_run is None``  → daemon has run but no tick
        has yet satisfied the (≥2000 trades, ≥30 days, ≥3 regimes)
        gate; the live supervisory brain is running on whatever state
        was already in canonical (or its hand-warmed defaults).

    When the daemon has never run we return
    ``{"ok": False, "reason": "no_replay_yet"}`` instead of 404 so a
    polling dashboard never has to special-case the empty state.

    The legacy ``summary`` key is preserved as an alias of
    ``last_run`` for any caller wired up before the dual-pointer
    contract landed.
    """
    from . import scheduled_meta_brain_replay

    last_run = scheduled_meta_brain_replay.load_last_replay()
    last_committed = (
        scheduled_meta_brain_replay.load_last_committed_replay()
    )
    if last_run is None and last_committed is None:
        return {"ok": False, "reason": "no_replay_yet"}
    scheduler_state = scheduled_meta_brain_replay.state
    return {
        "ok": True,
        "last_run": last_run,
        "last_committed_run": last_committed,
        # Backwards-compatible alias — same payload as `last_run`.
        "summary": last_run,
        "scheduler": {
            "enabled": scheduler_state.get("enabled"),
            "interval_seconds": scheduler_state.get("interval_seconds"),
            "window_days": scheduler_state.get("window_days"),
            "last_attempt_outcome": scheduler_state.get("last_attempt_outcome"),
            "last_check_at": scheduler_state.get("last_check_at"),
            "last_finished_at": scheduler_state.get("last_finished_at"),
            "last_error": scheduler_state.get("last_error"),
            "ticks_total": scheduler_state.get("ticks_total"),
            "runs_total": scheduler_state.get("runs_total"),
            "commits_total": scheduler_state.get("commits_total"),
            "skips_locked_total": scheduler_state.get("skips_locked_total"),
            "errors_total": scheduler_state.get("errors_total"),
        },
    }


@router.get("/stats")
def stats() -> dict:
    """Admin diagnostics. Never user-facing; logs + shadow-mode audit."""
    with _lock:
        if _service is None:
            return {"ok": False, "reason": "not_initialized"}
        trust = {
            fam: {"trust": fts.trust, "stability": fts.stability}
            for fam, fts in _service.trust_model.trust_by_family.items()
        }
        episodes = _service.episodic_memory.summarize_rewards(50)
        # Task #390 — surface the synthetic benchmark slot separately
        # so dashboards don't have to guess which key carries the
        # Strategy Lab telemetry.
        bench = _service.trust_model.trust_by_family.get("benchmark")
        # Task #567 — surface per-timeframe-role arrival counts and
        # the bounded shadow/context outcome stores so dashboards can
        # show "the brain is seeing N shadow records but learning
        # only from M trade records" without guessing.
        inputs_by_role = {role: int(_inputs_by_role.get(role, 0)) for role in VALID_SLICE_ROLES}
        shadow_count = len(_shadow_outcomes)
        context_count = len(_context_outcomes)
    # Task #578 — fold the 24h-by-hour trend into the /stats payload
    # so the dashboard can render a per-role sparkline beside the
    # cumulative totals without a second round-trip. Helper takes the
    # lock internally so we call it after dropping the outer one.
    trend = _summarize_role_trend()
    return {
        "ok": True,
        "tick_cache_size": len(_tick_cache),
        "trust_by_family": trust,
        "recent_rewards": episodes,
        "benchmark_trust": (
            {"trust": bench.trust, "stability": bench.stability}
            if bench is not None
            else None
        ),
        "inputs_by_role": inputs_by_role,
        "inputs_by_role_24h": trend["inputs_by_role_24h"],
        "inputs_by_role_hourly": trend["inputs_by_role_hourly"],
        "inputs_by_role_window_hours": trend["window_hours"],
        "shadow_outcomes_buffered": shadow_count,
        "context_outcomes_buffered": context_count,
        "state_root": str(_STATE_ROOT),
    }
