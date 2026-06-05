"""Integration tests for the FastAPI app.

`/health` and `/predict` are pure. `/features` is exercised here with a
monkey-patched `fetch_real_ticks` so the assertion stays hermetic but still
covers the full HTTP -> resampling -> feature-vector code path.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import main as main_module
from app.main import app


def test_health_returns_ok():
    with TestClient(app) as client:
        r = client.get("/ml/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "ml-engine"


def test_predict_never_silently_stubs(monkeypatch, tmp_path):
    """Phase-2 contract: predict must NOT silently fall back to a stub when
    no model is registered for the requested timeframe. Detailed model-loaded
    behavior lives in test_predict.py.

    Phase-6 update: every advertised TF in TIMEFRAMES now ships at least a
    `model_kind="prior"` pooled fallback (so the api-server's quant brain
    isn't 100%-LLM on 1h/2h/6h/1d). To keep this test exercising the
    "registry empty" path we point the registry at an empty tmp_path and
    expect 503 — confirming /predict still refuses to invent a probability
    when zero models exist for the requested coin/timeframe.
    """
    from app.training import registry as registry_module

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "REGISTRY_ROOT", tmp_path)
    main_module._cached_load.cache_clear()
    with TestClient(app) as client:
        r = client.post(
            "/ml/predict", json={"coinId": "bitcoin", "timeframe": "1d"}
        )
        # Either 503 (no model registered for 1d) or 409 (model exists but
        # too few candles); never 200 with a fabricated probability.
        assert r.status_code in (503, 409), r.text


def test_predict_uses_prior_pooled_fallback_when_only_a_prior_exists(
    monkeypatch, tmp_path,
):
    """Phase-6 / 2026-04-22 honesty fix: when only a `model_kind="prior"`
    pooled model is registered, /ml/predict no longer leaks the stored
    Laplace-smoothed empirical class frequencies as a directional call.
    Instead it emits a flat STABLE response with zero confidence so the
    downstream quant brain treats the slot as "no opinion" rather than
    speaking the recent backfill window's class skew (which on a short
    bullish-only window collapses to perpetual UP/STABLE and never DOWN).
    The `source` field stays `"prior"` so analytics can still distinguish
    fallback emissions from trained-model emissions.
    """
    from app.training import registry as registry_module
    from app.training.registry import ModelManifest, POOLED_COIN_ID, save_model

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "REGISTRY_ROOT", tmp_path)
    main_module._cached_load.cache_clear()
    manifest = ModelManifest(
        coin_id=POOLED_COIN_ID, timeframe="1d", version="vtest",
        feature_names=[], coin_vocab=["bitcoin"],
        n_train_rows=42, n_test_rows=0,
        metrics={"auc": float("nan"), "log_loss": float("nan"),
                 "brier": float("nan"), "directional_accuracy": float("nan")},
        baseline_metrics={"auc": float("nan"), "log_loss": float("nan"),
                          "brier": float("nan"), "directional_accuracy": float("nan")},
        threshold_pct=1.5, horizon_candles=1,
        class_return_means_pct=[-1.5, 0.0, 1.5],
        model_kind="prior",
        prior_probs=[0.2, 0.5, 0.3],
    )
    save_model(POOLED_COIN_ID, "1d", "vtest", booster=None, calibrators=None, manifest=manifest)

    with TestClient(app) as client:
        r = client.post("/ml/predict", json={"coinId": "bitcoin", "timeframe": "1d"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "prior"
        # Honest fallback: flat STABLE, zero confidence, zero expected return.
        # Raw stored priors (0.2/0.5/0.3) are intentionally NOT echoed here.
        assert math.isclose(body["probDown"], 0.0)
        assert math.isclose(body["probStable"], 1.0)
        assert math.isclose(body["probUp"], 0.0)
        assert math.isclose(body["expectedReturnPct"], 0.0)
        assert math.isclose(body["predictionStdPct"], 0.0)
        assert math.isclose(body["confidence"], 0.0)
        assert body["modelCoinId"] == POOLED_COIN_ID
        assert body["featureImportanceTop5"] == []
        # Task #460 — every QUANT prediction (including the prior fallback)
        # must carry a feature_hash so the api-server's journal-writer
        # accepts it. Prior models bypass the feature pipeline so we
        # synthesize a deterministic, traceable identifier from the
        # version. Without this, every prior-served abstain row was
        # silently refused at the journal layer.
        assert body["featureHash"] == "prior:vtest"


def test_predict_rejects_unknown_timeframe():
    with TestClient(app) as client:
        r = client.post(
            "/ml/predict", json={"coinId": "bitcoin", "timeframe": "bogus"}
        )
        assert r.status_code == 400


def _fake_ticks(n: int, start: datetime, step_s: int = 60) -> list[tuple[datetime, float]]:
    return [
        (start + timedelta(seconds=i * step_s), 100.0 + i * 0.5 + math.sin(i / 3.0) * 2.0)
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_features_with_sufficient_data_returns_full_vector(monkeypatch):
    """End-to-end: HTTP /ml/features with enough ticks returns a populated
    feature vector and a stable hash. We patch the DB layer so the test is
    hermetic while still exercising the resampling + feature pipeline.
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _fake_ticks(200, base, step_s=300)  # 200 ticks, 5min apart -> ample 1m candles

    async def fake_fetch(coin_id: str, lookback_ms: int, now=None):
        # Synthetic exclusion is enforced at SQL level in production; here we
        # just return real ticks to exercise the rest of the pipeline.
        assert coin_id == "pepe"
        assert lookback_ms > 0
        return ticks

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch)

    with TestClient(app) as client:
        r = client.post("/ml/features", json={"coinId": "pepe", "timeframe": "1m"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["coinId"] == "pepe"
        assert body["timeframe"] == "1m"
        assert body["candleCount"] >= 35
        assert body["insufficientData"] is False
        assert body["featureHash"] is not None and len(body["featureHash"]) == 12
        feats = body["features"]
        assert feats is not None
        assert 0.0 <= feats["rsi14"] <= 100.0
        assert math.isclose(feats["macdHist"], feats["macdLine"] - feats["macdSignal"], abs_tol=1e-9)


@pytest.mark.asyncio
async def test_features_insufficient_data_returns_null_vector(monkeypatch):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _fake_ticks(5, base, step_s=300)  # only 5 ticks -> can't make 35 candles

    async def fake_fetch(coin_id: str, lookback_ms: int, now=None):
        return ticks

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch)

    with TestClient(app) as client:
        r = client.post("/ml/features", json={"coinId": "pepe", "timeframe": "1m"})
        assert r.status_code == 200
        body = r.json()
        assert body["insufficientData"] is True
        assert body["features"] is None
        assert body["featureHash"] is None


def test_models_endpoint_lists_trained_pairs():
    """The api-server polls /ml/models to skip /predict calls that would 503
    with 'no model registered'. Contract: every entry must have a coinId and
    timeframe and the count must match the array length.
    """
    with TestClient(app) as client:
        r = client.get("/ml/models")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["available"], list)
        assert body["count"] == len(body["available"])
        for entry in body["available"]:
            assert "coinId" in entry and "timeframe" in entry
            assert isinstance(entry["coinId"], str) and isinstance(entry["timeframe"], str)


def test_features_rejects_unknown_timeframe():
    with TestClient(app) as client:
        r = client.post("/ml/features", json={"coinId": "bitcoin", "timeframe": "bogus"})
        assert r.status_code == 400


def _write_pooled_manifest(root, timeframe: str, kind: str, version: str = "vtest") -> None:
    """Stub a pooled manifest of the given kind into a tmp registry."""
    from app.training.registry import POOLED_COIN_ID

    d = root / POOLED_COIN_ID / timeframe / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        '{"model_kind": "' + kind + '", "coin_id": "__pooled__", '
        '"timeframe": "' + timeframe + '", "version": "' + version + '"}'
    )
    (root / POOLED_COIN_ID / timeframe / "latest").write_text(version)


def test_auto_retrain_skipped_when_no_priors(monkeypatch, tmp_path):
    """If every advertised pooled timeframe is already a real lightgbm
    model, the scheduler must be a cheap no-op — no retrain kickoff."""
    from app.training import registry as registry_module
    from app.training.train import DEFAULT_TIMEFRAMES

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "REGISTRY_ROOT", tmp_path)
    for tf in DEFAULT_TIMEFRAMES:
        _write_pooled_manifest(tmp_path, tf, "lightgbm")

    outcome = main_module._auto_retrain_tick(now=1_000_000.0)
    assert outcome == "skipped_no_priors"
    assert main_module._auto_retrain_state["last_attempt_outcome"] == "skipped_no_priors"


def test_auto_retrain_promotes_prior_to_lightgbm(monkeypatch, tmp_path):
    """End-to-end: a pooled `prior` model on 1d gets replaced by a real
    `lightgbm` model after one tick. The scheduler must log+persist the
    transition and bump promotions_total.
    """
    from app.training import registry as registry_module
    from app.training.train import DEFAULT_TIMEFRAMES

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "REGISTRY_ROOT", tmp_path)
    main_module._auto_retrain_state["last_attempt_at"] = None
    main_module._auto_retrain_state["promotions_total"] = 0
    main_module._auto_retrain_state["last_promoted_count"] = 0

    # Pre-state: 1d is a prior; everything else is a real lightgbm.
    for tf in DEFAULT_TIMEFRAMES:
        _write_pooled_manifest(tmp_path, tf, "prior" if tf == "1d" else "lightgbm")

    # Stub the heavy training run — promote 1d to lightgbm by overwriting
    # the manifest and bumping the version pointer (mirrors what a real
    # `run_training` would do once data finally suffices).
    def fake_run_blocking(coins, tfs):
        _write_pooled_manifest(tmp_path, "1d", "lightgbm", version="vnew")

    monkeypatch.setattr(main_module, "_run_retrain_blocking", fake_run_blocking)
    # Force enabled even if env disables in another test process.
    monkeypatch.setattr(main_module, "AUTO_RETRAIN_ENABLED", True)

    outcome = main_module._auto_retrain_tick(now=2_000_000.0)
    assert outcome == "kicked_off"

    # The runner is started in a daemon thread; wait briefly for completion.
    import time as _time
    deadline = _time.time() + 5
    while _time.time() < deadline and main_module._auto_retrain_state["last_promoted_count"] == 0:
        _time.sleep(0.05)

    assert main_module._auto_retrain_state["last_promoted_count"] == 1
    assert main_module._auto_retrain_state["promotions_total"] >= 1

    transitions_path = main_module._auto_retrain_transitions_path()
    assert transitions_path.exists()
    lines = [json.loads(l) for l in transitions_path.read_text().splitlines() if l.strip()]
    one_d = [r for r in lines if r["timeframe"] == "1d"]
    assert one_d, "expected a transition record for 1d"
    assert one_d[-1]["from_kind"] == "prior"
    assert one_d[-1]["to_kind"] == "lightgbm"
    assert one_d[-1]["trigger"] == "auto_retrain"

    # Status endpoint surfaces the transitions for the dashboard.
    main_module._retrain_lock = main_module.threading.Lock()  # release just in case
    with TestClient(app) as client:
        r = client.get("/ml/admin/auto-retrain/status")
        assert r.status_code == 200
        body = r.json()
        assert body["transitions_count"] >= 1
        assert any(t["timeframe"] == "1d" for t in body["recent_transitions"])
        assert body["current_pooled_kinds"]["1d"] == "lightgbm"


def test_auto_retrain_respects_min_gap(monkeypatch, tmp_path):
    """Two ticks back-to-back should only trigger one retrain attempt."""
    from app.training import registry as registry_module
    from app.training.train import DEFAULT_TIMEFRAMES

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "AUTO_RETRAIN_ENABLED", True)
    monkeypatch.setattr(main_module, "AUTO_RETRAIN_MIN_GAP_SECONDS", 600)
    main_module._auto_retrain_state["last_attempt_at"] = None

    for tf in DEFAULT_TIMEFRAMES:
        _write_pooled_manifest(tmp_path, tf, "prior" if tf == "1d" else "lightgbm")

    calls = {"n": 0}

    def fake_run_blocking(coins, tfs):
        calls["n"] += 1
        # Don't promote — we want both ticks to still see a prior.

    monkeypatch.setattr(main_module, "_run_retrain_blocking", fake_run_blocking)
    main_module._retrain_lock = main_module.threading.Lock()

    assert main_module._auto_retrain_tick(now=10_000.0) == "kicked_off"
    # Wait briefly for the daemon thread to release the lock.
    import time as _time
    _time.sleep(0.2)
    assert main_module._auto_retrain_tick(now=10_001.0) == "skipped_too_soon"
    assert calls["n"] == 1


import json  # noqa: E402  (used by the test above)


def _reset_fast_loop_state():
    main_module._fast_loop_state.update({
        "last_check_at": None,
        "last_attempt_at": None,
        "last_attempt_outcome": None,
        "last_decision_reason": None,
        "last_finished_at": None,
        "last_resolved_count": 0,
        "last_new_rows": 0,
        "last_error": None,
        "ticks_total": 0,
        "runs_total": 0,
        "last_envelope": None,
    })
    # Make sure the lock is free between tests.
    if main_module._fast_loop_lock.locked():
        try:
            main_module._fast_loop_lock.release()
        except RuntimeError:
            pass


def test_fast_loop_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(main_module, "FAST_LOOP_ENABLED", False)
    _reset_fast_loop_state()
    assert main_module._fast_loop_tick(now=1.0) == "disabled"
    assert main_module._fast_loop_state["last_attempt_outcome"] == "disabled"


def test_fast_loop_skipped_when_slow_in_progress(monkeypatch):
    monkeypatch.setattr(main_module, "FAST_LOOP_ENABLED", True)
    _reset_fast_loop_state()
    main_module._retrain_state["running"] = True
    try:
        assert main_module._fast_loop_tick(now=2.0) == "skipped_slow_in_progress"
    finally:
        main_module._retrain_state["running"] = False


def test_fast_loop_below_threshold(monkeypatch):
    monkeypatch.setattr(main_module, "FAST_LOOP_ENABLED", True)
    _reset_fast_loop_state()
    main_module._retrain_state["running"] = False

    async def fake_count():
        return 5

    monkeypatch.setattr(main_module, "_count_resolved_meta_rows", fake_count)
    monkeypatch.setattr(
        "app.training.train.FAST_LOOP_MIN_NEW_ROWS", 100, raising=False,
    )
    outcome = main_module._fast_loop_tick(now=3.0)
    assert outcome == "below_threshold"
    assert main_module._fast_loop_state["last_new_rows"] == 5
    assert main_module._fast_loop_state["last_decision_reason"] == "below_threshold"


def test_fast_loop_kicks_off_when_threshold_met(monkeypatch, tmp_path):
    """When enough new resolved rows accrue, the fast tick must kick off
    a meta-only retrain in a background thread, write fast_loop_report.json
    and never block the slow loop's lock."""
    from app.training import registry as registry_module

    monkeypatch.setattr(main_module, "FAST_LOOP_ENABLED", True)
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "REGISTRY_ROOT", tmp_path)
    _reset_fast_loop_state()
    main_module._retrain_state["running"] = False
    main_module._fast_loop_state["last_resolved_count"] = 10

    async def fake_count():
        return 250  # 240 new rows, well over the 100 threshold

    monkeypatch.setattr(main_module, "_count_resolved_meta_rows", fake_count)
    monkeypatch.setattr(
        "app.training.train.FAST_LOOP_MIN_NEW_ROWS", 100, raising=False,
    )

    ran = {"n": 0}

    async def fake_run_meta_only(timeframes):
        ran["n"] += 1
        envelope = {
            "generated_at": "2026-04-22T00:00:00+00:00",
            "loop": {"kind": "fast", "reason": "meta_only", "min_new_rows": 100},
            "meta_models": {},
        }
        # The real `run_meta_only` writes the report itself; mirror that
        # so the test can assert on the file's existence.
        (registry_module.REGISTRY_ROOT / "fast_loop_report.json").write_text(
            json.dumps(envelope)
        )
        return envelope

    monkeypatch.setattr("app.training.train.run_meta_only", fake_run_meta_only)
    # Slow loop must NOT be blocked by the fast tick — verify by holding
    # the slow lock across the call: the fast tick uses its own lock and
    # spawns a daemon thread, so the call returns immediately.
    main_module._retrain_lock = main_module.threading.Lock()
    assert main_module._retrain_lock.acquire(blocking=False)
    try:
        outcome = main_module._fast_loop_tick(now=4.0)
    finally:
        main_module._retrain_lock.release()

    # Slow lock was held but fast tick refused to skip on slow lock alone
    # (only `_retrain_state['running']` gates it), so we get kicked_off.
    assert outcome == "kicked_off"
    assert main_module._fast_loop_state["last_resolved_count"] == 250
    assert main_module._fast_loop_state["last_new_rows"] == 240

    # Wait briefly for the daemon thread.
    import time as _time
    deadline = _time.time() + 5
    while _time.time() < deadline and ran["n"] == 0:
        _time.sleep(0.05)
    assert ran["n"] == 1

    fast_path = tmp_path / "fast_loop_report.json"
    assert fast_path.exists()
    body = json.loads(fast_path.read_text())
    assert body["loop"]["kind"] == "fast"

    # Status endpoint surfaces it for operators.
    with TestClient(app) as client:
        r = client.get("/ml/admin/fast-loop/status")
        assert r.status_code == 200
        payload = r.json()
        assert payload["last_attempt_outcome"] == "kicked_off"
        assert payload["fast_loop_report"]["loop"]["kind"] == "fast"


def test_fast_loop_busy_lock_skips(monkeypatch):
    monkeypatch.setattr(main_module, "FAST_LOOP_ENABLED", True)
    _reset_fast_loop_state()
    main_module._retrain_state["running"] = False
    main_module._fast_loop_state["last_resolved_count"] = 0

    async def fake_count():
        return 500

    monkeypatch.setattr(main_module, "_count_resolved_meta_rows", fake_count)
    monkeypatch.setattr(
        "app.training.train.FAST_LOOP_MIN_NEW_ROWS", 100, raising=False,
    )
    # Pretend a previous fast run is still in flight by holding the lock.
    assert main_module._fast_loop_lock.acquire(blocking=False)
    try:
        assert main_module._fast_loop_tick(now=5.0) == "skipped_busy"
        # last_resolved_count must NOT advance while busy so the next tick
        # still sees the accumulated rows.
        assert main_module._fast_loop_state["last_resolved_count"] == 0
    finally:
        main_module._fast_loop_lock.release()


def test_admin_retrain_emits_progress_events(monkeypatch, tmp_path):
    """Task #636 — `/ml/admin/retrain` must wire `run_training`'s
    `progress_callback` to `models/progress_updates.jsonl` so an admin-
    triggered retrain shows up live on the dashboard, exactly like a
    scheduled run.

    Contract:
      - Hitting the endpoint with a valid token kicks off a thread that
        invokes `run_training(..., progress_callback=cb)` with a non-None
        callback.
      - When the callback fires, a JSONL row lands in
        `REGISTRY_ROOT/progress_updates.jsonl` containing the event keys
        plus an `emitted_at` timestamp added by the writer.
    """
    import time as _time
    from app.training import registry as registry_module

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "REGISTRY_ROOT", tmp_path)
    monkeypatch.setenv("ML_ADMIN_TOKEN", "secret-token")

    # Stub run_training to invoke its progress_callback once and return a
    # minimal, well-formed report — keeps the test hermetic (no DB, no
    # real model fits) while still exercising the wiring.
    captured: dict = {}

    async def fake_run_training(coins, timeframes, progress_callback=None):
        captured["progress_callback"] = progress_callback
        if progress_callback is not None:
            progress_callback({
                "phase": "build_dataset_start",
                "status": "running",
                "headline": "admin-triggered retrain warming up",
                "timeframe": (timeframes or ["1d"])[0],
            })
        return {"timeframes": {(timeframes or ["1d"])[0]: {"status": "ok"}}}

    monkeypatch.setattr(
        "app.training.train.run_training", fake_run_training, raising=True,
    )
    # Janitor pokes the real models dir on shutdown; stub it so the test
    # stays hermetic against the empty tmp_path registry.
    monkeypatch.setattr(
        "app.training.registry.prune_contaminated_versions",
        lambda: {"deleted": 0, "freed_bytes": 0},
        raising=True,
    )

    saved = dict(main_module._retrain_state)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/ml/admin/retrain",
                headers={"X-Admin-Token": "secret-token"},
                json={"coins": ["bitcoin"], "timeframes": ["1d"]},
            )
            assert r.status_code == 200, r.text
            # Wait for the background thread to drain.
            deadline = _time.time() + 5
            while _time.time() < deadline and main_module._retrain_state.get("running"):
                _time.sleep(0.05)
            assert main_module._retrain_state.get("running") is False
            assert main_module._retrain_state.get("last_status") == "ok"

        # Wiring: the callback we passed in is the admin progress writer.
        assert captured.get("progress_callback") is not None
        assert (
            captured["progress_callback"]
            is main_module._append_admin_retrain_progress
        )

        # End-to-end: at least one row landed in the JSONL file with the
        # event payload + an `emitted_at` stamp injected by the writer.
        progress_path = tmp_path / "progress_updates.jsonl"
        assert progress_path.exists(), "progress_updates.jsonl was never written"
        rows = [
            json.loads(ln)
            for ln in progress_path.read_text().splitlines()
            if ln.strip()
        ]
        assert rows, "no progress rows recorded"
        first = rows[0]
        assert first.get("phase") == "build_dataset_start"
        assert first.get("timeframe") == "1d"
        assert "emitted_at" in first
    finally:
        main_module._retrain_state.clear()
        main_module._retrain_state.update(saved)


def test_retrain_status_exposes_cleanup_fields():
    """Task #457 — the retrain status endpoint must surface the auto-prune
    janitor's bookkeeping so the dashboard chip can render
    "Last cleanup: N versions, X MB freed".

    Contract:
      - Whatever keys live on `_retrain_state` (including the post-#451
        `last_pruned_count` / `last_pruned_bytes` / `last_pruned_error`)
        round-trip verbatim through `/ml/admin/retrain/status`.
      - The endpoint returns a copy, not a live view, so a follow-up
        mutation of `_retrain_state` must NOT retro-edit a response that
        an operator's dashboard has already received.
    """
    saved = dict(main_module._retrain_state)
    try:
        main_module._retrain_state["last_pruned_count"] = 3
        main_module._retrain_state["last_pruned_bytes"] = 1_234_567
        main_module._retrain_state["last_pruned_error"] = None
        with TestClient(app) as client:
            r = client.get("/ml/admin/retrain/status")
            assert r.status_code == 200
            payload = r.json()
            assert payload["last_pruned_count"] == 3
            assert payload["last_pruned_bytes"] == 1_234_567
            assert payload["last_pruned_error"] is None
            # Mutate after the response was built; the returned dict must
            # be detached so the dashboard's snapshot stays stable.
            main_module._retrain_state["last_pruned_count"] = 999
            r2 = client.get("/ml/admin/retrain/status")
            assert r2.json()["last_pruned_count"] == 999
            assert payload["last_pruned_count"] == 3

        # Error-path contract: when the janitor fails its message is
        # surfaced verbatim so an operator can see *why* cleanup is broken
        # instead of a silent stale chip.
        main_module._retrain_state["last_pruned_error"] = "disk full"
        with TestClient(app) as client:
            r = client.get("/ml/admin/retrain/status")
            assert r.json()["last_pruned_error"] == "disk full"
    finally:
        main_module._retrain_state.clear()
        main_module._retrain_state.update(saved)
