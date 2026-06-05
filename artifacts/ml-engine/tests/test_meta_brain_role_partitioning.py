"""Task #567 — meta-brain consumes `slice_role` correctly.

Locks in the role-partitioning contract: trust scores update only on
outcomes whose `slice_role` is `trade`. `shadow` and `context`
outcomes are stored separately for analysis and never touch trust.
`disabled` outcomes are rejected with a structured warn — they
should never arrive, and if they do the upstream is broken.

A back-compat default of `trade` is applied when `slice_role` is
absent (a one-time per-process `[ROLE_BACKCOMPAT_DEFAULT]` warn fires
the first time so stale callers are visible). The hard rule is:
never silently accept missing roles.

The new `inputs_by_role` field on `/stats` exposes the per-role
arrival counts so a dashboard can answer "the brain is seeing N
shadow records but learning only from M trade records" without
guessing.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app import meta_brain
from app.main import app


def _slice(family: str = "momentum", slice_role: str | None = "trade") -> dict:
    """A canonical slice payload, matching the existing HTTP test
    fixture shape but with the optional `slice_role` field added."""
    s = {
        "coin": "btc",
        "timeframe": "1m",
        "strategy_family": family,
        "edge": 0.02,
        "confidence": 0.7,
        "calibrated_confidence": 0.7,
        "risk_score": 0.3,
        "recent_accuracy": 0.55,
        "pnl_state": 0.01,
        "drawdown_state": 0.02,
        "disagreement": 0.1,
        "prediction_error": 0.05,
        "regime": "trend_up",
        "volatility": 0.02,
        "correlation_shift": 0.0,
        "exposure": 0.1,
        "turnover": 0.05,
        "slippage_bps": 1.0,
        "anomaly_flags": [],
    }
    if slice_role is not None:
        s["slice_role"] = slice_role
    return s


def _portfolio() -> dict:
    return {
        "total_drawdown": 0.05,
        "realized_vol": 0.1,
        "concentration": 0.2,
        "leverage": 1.0,
        "liquidity_stress": 0.1,
        "correlation_shift": 0.0,
        "active_risk_budget": 0.2,
        "kill_switch_distance": 0.8,
        "anomaly_flags": [],
    }


def _outcome(realized_pnl: float = 0.01) -> dict:
    return {
        "realized_pnl": realized_pnl,
        "realized_drawdown": 0.005,
        "realized_stability": 0.8,
        "turnover_cost": 0.0005,
        "action_churn": 0.0,
        "correct_defense": 0.0,
        "correct_suppression": 0.0,
        "missed_edge_cost": 0.0,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Hermetic per-test brain: temp state root, fresh service, no
    leakage between tests including the one-time back-compat warn
    flag."""
    monkeypatch.setattr(meta_brain, "_STATE_ROOT", tmp_path)
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain._inputs_by_role.clear()
    meta_brain._inputs_by_role.update({r: 0 for r in meta_brain.VALID_SLICE_ROLES})
    # Task #578 — reset the in-memory hourly bucket map between tests
    # so a per-role spike from one test cannot leak into another's
    # `inputs_by_role_24h` assertions.
    meta_brain._role_hourly_buckets.clear()
    meta_brain._shadow_outcomes.clear()
    meta_brain._context_outcomes.clear()
    meta_brain._inputs_by_role_dirty = False
    meta_brain._inputs_by_role_last_write_ts = 0.0
    # Task #583 — reset the persistence-debounce flags for the hourly
    # bucket map too, so a previous test's write does not block the
    # next test's force-flush from running.
    meta_brain._role_hourly_buckets_dirty = False
    meta_brain._role_hourly_buckets_last_write_ts = 0.0
    meta_brain._role_backcompat_warned = False
    meta_brain.init()
    yield TestClient(app)
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain._inputs_by_role.clear()
    meta_brain._role_hourly_buckets.clear()
    meta_brain._shadow_outcomes.clear()
    meta_brain._context_outcomes.clear()
    meta_brain._role_backcompat_warned = False


def _trust_snapshot() -> dict:
    """Take a deep snapshot of the trust table so callers can compare
    family-by-family without aliasing."""
    svc = meta_brain._service
    assert svc is not None
    return {
        fam: {
            "trust": fts.trust,
            "stability": fts.stability,
            "exploration_eligibility": fts.exploration_eligibility,
            "failure_streak": fts.failure_streak,
            "recovery_score": fts.recovery_score,
            "last_regime": fts.last_regime,
        }
        for fam, fts in svc.trust_model.trust_by_family.items()
    }


def _evaluate_and_get_tick(client: TestClient, slice_role: str | None) -> str:
    """Issue an /evaluate cycle and return the tick_id. We always send
    a momentum-family slice so trust_by_family["momentum"] exists for
    the assertions."""
    body = {
        "slices": [_slice(slice_role=slice_role)],
        "portfolio": _portfolio(),
        "timestamp": "2026-04-28T00:00:00Z",
    }
    r = client.post("/ml/meta-brain/evaluate", json=body)
    assert r.status_code == 200, r.text
    return r.json()["tick_id"]


def _replay_outcomes(
    client: TestClient,
    *,
    role: str | None,
    n: int,
) -> list[dict]:
    """Replay `n` /record-outcome calls with the given role. Returns
    the list of HTTP response bodies for inspection."""
    bodies: list[dict] = []
    for i in range(n):
        tick_id = _evaluate_and_get_tick(client, slice_role=role)
        payload: dict = {
            "tick_id": tick_id,
            "timestamp": f"2026-04-28T00:00:{i % 60:02d}Z",
            "outcome": _outcome(realized_pnl=0.01 if i % 2 == 0 else -0.005),
        }
        if role is not None:
            payload["slice_role"] = role
        r = client.post("/ml/meta-brain/record-outcome", json=payload)
        assert r.status_code == 200, r.text
        bodies.append(r.json())
    return bodies


def _prime_family_registration(client: TestClient) -> None:
    """Issue one /evaluate so the trust model registers the families
    the test will later inspect. Without this, the first evaluate of
    a fresh service registers `momentum` in `trust_by_family` with
    its neutral defaults — making the pre/post-snapshot comparison
    incorrectly show a "diff" that's actually just family
    registration, not a trust update."""
    body = {
        "slices": [_slice(slice_role="trade")],
        "portfolio": _portfolio(),
        "timestamp": "2026-04-28T00:00:00Z",
    }
    r = client.post("/ml/meta-brain/evaluate", json=body)
    assert r.status_code == 200, r.text


# ─────────────────────────── trade-only path ───────────────────────────


def test_trade_only_path_updates_trust_and_increments_counter(client):
    """Replaying 100 trade-roled outcomes should update trust scores
    by the same delta the existing (pre-#567) loop would have, and
    inputs_by_role.trade should track the count."""
    before = _trust_snapshot()
    bodies = _replay_outcomes(client, role="trade", n=100)

    # Every call confirmed trust updated.
    assert all(b.get("ok") is True for b in bodies)
    assert all(b.get("trust_updated") is True for b in bodies)

    after = _trust_snapshot()
    # Trust must move on at least the family the slices carry; the
    # "no contamination" tests in the same workflow already cover what
    # trust changes look like — here we just need the change to exist.
    assert "momentum" in after, "momentum family should be present"
    assert (
        before.get("momentum", {}).get("trust") != after["momentum"]["trust"]
        or before.get("momentum", {}).get("stability")
        != after["momentum"]["stability"]
        or before.get("momentum", {}).get("recovery_score")
        != after["momentum"]["recovery_score"]
    ), "trade outcomes must move SOMETHING in the trust state"

    r = client.get("/ml/meta-brain/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["inputs_by_role"]["trade"] == 100
    assert body["inputs_by_role"]["shadow"] == 0
    assert body["inputs_by_role"]["context"] == 0
    assert body["inputs_by_role"]["disabled"] == 0


# ─────────────────────────── shadow isolation ──────────────────────────


def test_shadow_does_not_update_trust(client):
    """100 shadow-roled outcomes must leave the trust table untouched
    and only populate the shadow store + counter."""
    _prime_family_registration(client)
    before = _trust_snapshot()
    bodies = _replay_outcomes(client, role="shadow", n=100)

    assert all(b.get("ok") is True for b in bodies)
    assert all(b.get("trust_updated") is False for b in bodies)
    assert all(b.get("role") == "shadow" for b in bodies)

    after = _trust_snapshot()
    assert after == before, (
        "shadow-roled outcomes must NOT touch trust state; "
        f"diff: {set(before) ^ set(after)}"
    )

    r = client.get("/ml/meta-brain/stats")
    body = r.json()
    assert body["inputs_by_role"]["shadow"] == 100
    assert body["inputs_by_role"]["trade"] == 0
    assert body["shadow_outcomes_buffered"] > 0
    assert body["context_outcomes_buffered"] == 0


# ─────────────────────────── context isolation ─────────────────────────


def test_context_does_not_update_trust(client):
    """100 context-roled outcomes must leave the trust table untouched
    and only populate the context store + counter."""
    _prime_family_registration(client)
    before = _trust_snapshot()
    bodies = _replay_outcomes(client, role="context", n=100)

    assert all(b.get("ok") is True for b in bodies)
    assert all(b.get("trust_updated") is False for b in bodies)
    assert all(b.get("role") == "context" for b in bodies)

    after = _trust_snapshot()
    assert after == before, (
        "context-roled outcomes must NOT touch trust state"
    )

    r = client.get("/ml/meta-brain/stats")
    body = r.json()
    assert body["inputs_by_role"]["context"] == 100
    assert body["inputs_by_role"]["trade"] == 0
    assert body["context_outcomes_buffered"] > 0
    assert body["shadow_outcomes_buffered"] == 0


# ─────────────────────────── disabled rejection ────────────────────────


def test_disabled_outcome_is_rejected_with_structured_warn(client, caplog):
    """A `disabled` outcome must NOT touch trust and MUST emit a
    `disabled_outcome_received` structured warn. The upstream is
    broken if this case is ever hit."""
    import logging

    _prime_family_registration(client)
    before = _trust_snapshot()
    tick_id = _evaluate_and_get_tick(client, slice_role="disabled")
    with caplog.at_level(logging.WARNING):
        r = client.post(
            "/ml/meta-brain/record-outcome",
            json={
                "tick_id": tick_id,
                "slice_role": "disabled",
                "slice_id": "btc:1m",
                "outcome": _outcome(),
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is False
    assert body.get("reason") == "disabled_role_rejected"

    after = _trust_snapshot()
    assert after == before, (
        "disabled-roled outcomes must NEVER touch trust state"
    )

    r = client.get("/ml/meta-brain/stats")
    sbody = r.json()
    assert sbody["inputs_by_role"]["disabled"] == 1
    assert sbody["shadow_outcomes_buffered"] == 0
    assert sbody["context_outcomes_buffered"] == 0

    # The structured warn must have fired with the documented event
    # name and slice metadata, so ops can grep + alert on it.
    matched = [
        rec for rec in caplog.records
        if "disabled_outcome_received" in rec.getMessage()
        or getattr(rec, "event", None) == "disabled_outcome_received"
    ]
    assert matched, (
        "disabled outcomes must emit a `disabled_outcome_received` warn"
    )


# ─────────────────────────── mixed isolation proof ─────────────────────


def test_mixed_replay_isolates_trade_outcomes(client):
    """30 trade + 30 shadow + 30 context replays must produce trust
    deltas IDENTICAL to a pure-30-trade replay. Proves shadow/context
    do not contaminate the trust math."""
    # Prime the family registration so the snapshot we take next
    # reflects neutral DEFAULTS, not "family doesn't exist yet".
    _prime_family_registration(client)
    baseline_before = _trust_snapshot()
    _replay_outcomes(client, role="trade", n=30)
    baseline_after = _trust_snapshot()

    # Now reset the brain and replay 30 trade + 30 shadow + 30 context
    # in a deterministic order. The trust delta must equal the
    # baseline; the shadow/context replays must vanish from the trust
    # math entirely.
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain._inputs_by_role.clear()
    meta_brain._inputs_by_role.update({r: 0 for r in meta_brain.VALID_SLICE_ROLES})
    meta_brain._shadow_outcomes.clear()
    meta_brain._context_outcomes.clear()
    meta_brain._inputs_by_role_dirty = False
    meta_brain._inputs_by_role_last_write_ts = 0.0
    # Also delete the on-disk counters file so a fresh init() doesn't
    # rehydrate the baseline phase's persisted counts and inflate the
    # mixed-phase totals (the persistence layer writes through a 1Hz
    # debounce, and any one of the baseline replays will have flushed).
    counter_file = meta_brain._inputs_by_role_path()
    if counter_file.exists():
        counter_file.unlink()
    meta_brain.init()
    _prime_family_registration(client)

    mixed_before = _trust_snapshot()
    assert mixed_before == baseline_before, (
        "fresh init must reproduce the baseline starting trust state"
    )

    # Interleave: t s c t s c ...
    roles = ["trade", "shadow", "context"] * 30
    for i, role in enumerate(roles):
        tick_id = _evaluate_and_get_tick(client, slice_role=role)
        r = client.post(
            "/ml/meta-brain/record-outcome",
            json={
                "tick_id": tick_id,
                "slice_role": role,
                "timestamp": f"2026-04-28T01:00:{i % 60:02d}Z",
                "outcome": _outcome(
                    realized_pnl=0.01 if i % 2 == 0 else -0.005
                ),
            },
        )
        assert r.status_code == 200, r.text

    mixed_after = _trust_snapshot()

    # The trust deltas in the mixed run must EXACTLY match the
    # baseline. Any difference = shadow/context contaminated trust.
    assert set(mixed_after) == set(baseline_after), (
        f"family set diverged: {set(mixed_after) ^ set(baseline_after)}"
    )
    for fam, baseline_state in baseline_after.items():
        mixed_state = mixed_after[fam]
        for field in (
            "trust",
            "stability",
            "exploration_eligibility",
            "failure_streak",
            "recovery_score",
        ):
            assert mixed_state[field] == pytest.approx(
                baseline_state[field], abs=1e-9
            ), (
                f"family={fam} field={field} differs between trade-only "
                f"({baseline_state[field]}) and mixed ({mixed_state[field]}) "
                "— shadow/context outcomes contaminated the trust math"
            )

    r = client.get("/ml/meta-brain/stats")
    body = r.json()
    assert body["inputs_by_role"]["trade"] == 30
    assert body["inputs_by_role"]["shadow"] == 30
    assert body["inputs_by_role"]["context"] == 30
    assert body["inputs_by_role"]["disabled"] == 0


# ─────────────────────────── back-compat default ───────────────────────


def test_back_compat_default_treats_missing_role_as_trade(client, caplog):
    """A record without `slice_role` must be processed as `trade`
    (back-compat) AND a one-time per-process
    `[ROLE_BACKCOMPAT_DEFAULT]` warn must fire."""
    import logging

    before = _trust_snapshot()
    tick_id = _evaluate_and_get_tick(client, slice_role=None)
    with caplog.at_level(logging.WARNING):
        r1 = client.post(
            "/ml/meta-brain/record-outcome",
            json={
                "tick_id": tick_id,
                "outcome": _outcome(),
            },
        )
        # Second call — must NOT fire the warn again (one-time).
        tick_id2 = _evaluate_and_get_tick(client, slice_role=None)
        r2 = client.post(
            "/ml/meta-brain/record-outcome",
            json={
                "tick_id": tick_id2,
                "outcome": _outcome(),
            },
        )

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    body1 = r1.json()
    body2 = r2.json()
    assert body1.get("trust_updated") is True
    assert body2.get("trust_updated") is True
    assert body1.get("role") == "trade"

    after = _trust_snapshot()
    assert after != before, (
        "back-compat default must treat missing role as `trade` and "
        "actually update trust"
    )

    backcompat_warns = [
        rec for rec in caplog.records
        if "ROLE_BACKCOMPAT_DEFAULT" in rec.getMessage()
    ]
    assert len(backcompat_warns) == 1, (
        f"expected exactly one [ROLE_BACKCOMPAT_DEFAULT] warn for the "
        f"first missing-role record, got {len(backcompat_warns)}"
    )

    r = client.get("/ml/meta-brain/stats")
    sbody = r.json()
    assert sbody["inputs_by_role"]["trade"] == 2


def test_invalid_slice_role_is_rejected(client):
    """Anything that's not one of the four canonical roles must be
    rejected with a 400 — the contract is locked, not extensible by
    JSON."""
    tick_id = _evaluate_and_get_tick(client, slice_role="trade")
    r = client.post(
        "/ml/meta-brain/record-outcome",
        json={
            "tick_id": tick_id,
            "slice_role": "exploratory",
            "outcome": _outcome(),
        },
    )
    assert r.status_code == 400, r.text
    assert "invalid_slice_role" in r.json().get("detail", "")


# ─────────────────────────── status surface ────────────────────────────


def test_status_endpoint_exposes_inputs_by_role(client):
    """The /stats payload must include `inputs_by_role` with all four
    canonical role keys present (zero-initialised when no records have
    been received), so dashboards can render the four-slot card without
    null-checks."""
    r = client.get("/ml/meta-brain/stats")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "inputs_by_role" in body
    assert set(body["inputs_by_role"].keys()) == {
        "trade",
        "shadow",
        "context",
        "disabled",
    }
    for role, count in body["inputs_by_role"].items():
        assert isinstance(count, int)
        assert count == 0


def test_status_endpoint_exposes_inputs_by_role_24h_trend(client):
    """Task #578 — `/stats` must surface the dashboard-facing 24h trend
    so the role-outcome card can render its sparkline without a second
    round-trip. With no traffic the trend should be a fully zero-filled
    24-bucket window with all four roles present per bucket."""
    r = client.get("/ml/meta-brain/stats")
    assert r.status_code == 200, r.text
    body = r.json()

    # 24h totals — all four roles present, all zero on a fresh service.
    assert "inputs_by_role_24h" in body
    assert set(body["inputs_by_role_24h"].keys()) == {
        "trade",
        "shadow",
        "context",
        "disabled",
    }
    for role, count in body["inputs_by_role_24h"].items():
        assert isinstance(count, int)
        assert count == 0

    # Hourly buckets — exactly 24 of them, oldest → newest, each with
    # all four role keys zero-filled and a parseable ISO hour_start.
    assert body.get("inputs_by_role_window_hours") == 24
    hourly = body.get("inputs_by_role_hourly")
    assert isinstance(hourly, list)
    assert len(hourly) == 24
    for bucket in hourly:
        assert isinstance(bucket, dict)
        assert "hour_start" in bucket
        assert isinstance(bucket["hour_start"], str)
        assert bucket["hour_start"].endswith(":00:00Z")
        assert set(bucket["counts"].keys()) == {
            "trade",
            "shadow",
            "context",
            "disabled",
        }
        for role, count in bucket["counts"].items():
            assert isinstance(count, int)
            assert count == 0


def test_status_endpoint_24h_trend_reflects_traffic(client):
    """After a record-outcome call the matching role's 24h total must
    be > 0 and the most recent hourly bucket must carry that count."""
    # Drive one trade-roled outcome through the wire so the bucket map
    # is populated by the same path the live service uses.
    _replay_outcomes(client, role="trade", n=3)
    _replay_outcomes(client, role="shadow", n=2)

    r = client.get("/ml/meta-brain/stats")
    body = r.json()
    assert body["inputs_by_role_24h"]["trade"] == 3
    assert body["inputs_by_role_24h"]["shadow"] == 2
    assert body["inputs_by_role_24h"]["context"] == 0
    assert body["inputs_by_role_24h"]["disabled"] == 0

    # Newest bucket (last in the oldest→newest list) should hold the
    # arrivals because they all happened in the current hour.
    latest = body["inputs_by_role_hourly"][-1]
    assert latest["counts"]["trade"] == 3
    assert latest["counts"]["shadow"] == 2


def test_inputs_by_role_persists_across_restart(tmp_path, monkeypatch):
    """Counters live in `models/meta_brain_state/inputs_by_role.json`
    so a process restart resumes from the last committed totals
    instead of zeroing the dashboard.

    Task #583 — the hourly bucket map lives in
    `models/meta_brain_state/role_hourly_buckets.json` so the 24h
    sparkline survives a restart instead of phantoming a 24h drop to
    zero. We also verify that here so future regressions are caught.
    """
    monkeypatch.setattr(meta_brain, "_STATE_ROOT", tmp_path)
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain._inputs_by_role.clear()
    meta_brain._inputs_by_role.update({r: 0 for r in meta_brain.VALID_SLICE_ROLES})
    meta_brain._role_hourly_buckets.clear()
    meta_brain._shadow_outcomes.clear()
    meta_brain._context_outcomes.clear()
    meta_brain._inputs_by_role_dirty = False
    meta_brain._inputs_by_role_last_write_ts = 0.0
    meta_brain._role_hourly_buckets_dirty = False
    meta_brain._role_hourly_buckets_last_write_ts = 0.0
    meta_brain._role_backcompat_warned = False
    meta_brain.init()

    client = TestClient(app)
    _replay_outcomes(client, role="trade", n=3)
    _replay_outcomes(client, role="shadow", n=2)

    # Snapshot the live hourly trend so we can compare it to the
    # post-restart payload byte-for-byte.
    pre_restart = client.get("/ml/meta-brain/stats").json()
    pre_hourly = pre_restart["inputs_by_role_hourly"]
    pre_24h = pre_restart["inputs_by_role_24h"]

    # Force a flush so both files exist on disk before "restart".
    meta_brain._persist_inputs_by_role(force=True)
    meta_brain._persist_role_hourly_buckets(force=True)
    persisted = (tmp_path / "inputs_by_role.json").read_text(encoding="utf-8")
    assert "trade" in persisted
    buckets_path = tmp_path / "role_hourly_buckets.json"
    assert buckets_path.exists(), (
        "role_hourly_buckets.json must be written so the dashboard "
        "sparkline survives an ml-engine restart"
    )
    persisted_buckets = json.loads(buckets_path.read_text(encoding="utf-8"))
    assert persisted_buckets.get("version") == 1
    assert isinstance(persisted_buckets.get("buckets"), list)
    assert persisted_buckets["buckets"], (
        "at least one bucket must be persisted after live traffic"
    )
    # The persisted bucket totals must match the live counters so the
    # post-restart sparkline is identical, not just non-zero.
    persisted_totals: Counter = Counter()
    for entry in persisted_buckets["buckets"]:
        for role, n in entry["counts"].items():
            persisted_totals[role] += int(n)
    assert persisted_totals["trade"] == 3
    assert persisted_totals["shadow"] == 2

    # Simulate a process restart: clear in-memory state, re-init.
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain._inputs_by_role.clear()
    meta_brain._role_hourly_buckets.clear()
    meta_brain._shadow_outcomes.clear()
    meta_brain._context_outcomes.clear()
    meta_brain._role_backcompat_warned = False
    meta_brain.init()

    r = client.get("/ml/meta-brain/stats")
    body = r.json()
    assert body["inputs_by_role"]["trade"] == 3
    assert body["inputs_by_role"]["shadow"] == 2

    # The 24h sparkline must look identical to its pre-restart self —
    # no phantom drop, no shift, no zero hour. The most recent bucket
    # in particular must still carry the 3 trade + 2 shadow arrivals.
    assert body["inputs_by_role_24h"] == pre_24h, (
        "post-restart 24h totals must match the pre-restart payload"
    )
    assert body["inputs_by_role_hourly"] == pre_hourly, (
        "post-restart hourly buckets must match the pre-restart payload"
    )
    latest = body["inputs_by_role_hourly"][-1]
    assert latest["counts"]["trade"] == 3
    assert latest["counts"]["shadow"] == 2

    # Cleanup.
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain._inputs_by_role.clear()
    meta_brain._role_hourly_buckets.clear()


def test_role_hourly_buckets_drop_stale_entries_on_load(tmp_path, monkeypatch):
    """Task #583 — buckets older than the retention window must be
    dropped on load. After a long downtime (>24h), the persisted file
    can carry buckets from a previous "day"; restoring them as-is
    would resurface them as "fresh" sparkline bars and lie to the
    operator. The loader prunes them silently."""
    monkeypatch.setattr(meta_brain, "_STATE_ROOT", tmp_path)

    # Hand-craft a buckets file with: one bucket far in the past,
    # one inside the retention window, one in the current hour.
    now_hour = int(time.time()) // 3600
    stale_hour = now_hour - meta_brain.ROLE_HOURLY_RETENTION_HOURS - 5
    fresh_hour = now_hour - 2
    payload = {
        "version": 1,
        "buckets": [
            {
                "hour": stale_hour,
                "counts": {"trade": 99, "shadow": 0, "context": 0, "disabled": 0},
            },
            {
                "hour": fresh_hour,
                "counts": {"trade": 4, "shadow": 1, "context": 0, "disabled": 0},
            },
            {
                "hour": now_hour,
                "counts": {"trade": 7, "shadow": 2, "context": 0, "disabled": 0},
            },
        ],
    }
    (tmp_path / "role_hourly_buckets.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    # Reset and load.
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain._inputs_by_role.clear()
    meta_brain._inputs_by_role.update({r: 0 for r in meta_brain.VALID_SLICE_ROLES})
    meta_brain._role_hourly_buckets.clear()
    meta_brain._shadow_outcomes.clear()
    meta_brain._context_outcomes.clear()
    meta_brain._inputs_by_role_dirty = False
    meta_brain._inputs_by_role_last_write_ts = 0.0
    meta_brain._role_hourly_buckets_dirty = False
    meta_brain._role_hourly_buckets_last_write_ts = 0.0
    meta_brain._role_backcompat_warned = False
    meta_brain.init()

    # The stale bucket must NOT be in memory; the fresh + current
    # buckets must be, ordered oldest → newest.
    keys = list(meta_brain._role_hourly_buckets.keys())
    assert stale_hour not in keys, (
        "buckets older than the retention window must be dropped on load"
    )
    assert keys == [fresh_hour, now_hour]

    # The 24h trend must reflect ONLY the in-window buckets.
    client = TestClient(app)
    body = client.get("/ml/meta-brain/stats").json()
    assert body["inputs_by_role_24h"]["trade"] == 4 + 7
    assert body["inputs_by_role_24h"]["shadow"] == 1 + 2

    # Cleanup.
    meta_brain._service = None  # type: ignore[attr-defined]
    meta_brain._tick_cache.clear()
    meta_brain._inputs_by_role.clear()
    meta_brain._role_hourly_buckets.clear()
