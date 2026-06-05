"""Task #220 — auto-register newly trained models as shadow rows.

Covers the report walker (pure, no DB) and the end-to-end insert path
against a fake asyncpg pool that records the SQL it sees so the test
asserts both shape and idempotency without needing a live Postgres.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.training import register_shadow


def _sample_report() -> dict:
    return {
        "generated_at": "2026-04-22T00:00:00+00:00",
        "timeframes": {
            "5m": {
                "per_coin": {
                    "pepe": {
                        "status": "trained", "version": "v1",
                        "metrics": {"auc": 0.6}, "n_rows": 200,
                    },
                    "bonk": {
                        "status": "insufficient_data_per_coin",
                        "n_rows": 10,
                    },
                },
                "pooled": {
                    "status": "trained", "version": "v1",
                    "coin_id": "__pooled__", "n_rows": 800,
                },
                "specialists": {
                    "momentum": {
                        "status": "trained", "version": "v1",
                        "coin_id": "__specialist_momentum__",
                        "specialist_kind": "momentum",
                        "regime_subset": ["trending_up"],
                    },
                    "mean_reversion": {
                        "status": "insufficient_data",
                        "specialist_kind": "mean_reversion",
                    },
                },
            },
            "1h": {
                "per_coin": {},
                "pooled": {
                    "status": "trained", "version": "v1",
                    "model_kind": "prior",
                },
            },
        },
        "meta_models": {
            "5m": {"status": "trained", "version": "vmeta1"},
            "1h": {"status": "heuristic", "version": "vmeta1"},
            "1d": {"status": "error", "error": "boom"},  # no version → skipped
        },
    }


def test_slices_from_report_collects_only_trained() -> None:
    slices = register_shadow._slices_from_report(_sample_report())
    keys = sorted((m, c, t, v) for m, c, t, v, _ in slices)
    assert keys == [
        ("lightgbm", "__pooled__", "1h", "v1"),
        ("lightgbm", "__pooled__", "5m", "v1"),
        ("lightgbm", "__specialist_momentum__", "5m", "v1"),
        ("lightgbm", "pepe", "5m", "v1"),
        ("lightgbm-meta", "__meta__", "1h", "vmeta1"),
        ("lightgbm-meta", "__meta__", "5m", "vmeta1"),
    ]


class _FakeConn:
    def __init__(self, store: list[tuple[str, str, str, str]]) -> None:
        self._store = store
        self.executed: list[tuple[str, tuple]] = []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        # Match the SELECT in register_shadow_rows: lookup by tuple.
        model_id, version, coin_id, timeframe = args
        for row in self._store:
            if row == (model_id, version, coin_id, timeframe):
                return 1
        return None

    async def execute(self, sql: str, *args: Any) -> None:
        self.executed.append((sql, args))
        model_id, version, coin_id, timeframe = args[:4]
        self._store.append((model_id, version, coin_id, timeframe))


class _AcquireCtx:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakePool:
    def __init__(self) -> None:
        self.store: list[tuple[str, str, str, str]] = []
        self.conn = _FakeConn(self.store)

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self.conn)


def _patch_pool(monkeypatch: pytest.MonkeyPatch, pool: _FakePool) -> None:
    async def _init() -> _FakePool:
        return pool

    monkeypatch.setattr(register_shadow, "init_pool", _init)


def test_register_shadow_rows_inserts_then_skips_on_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    _patch_pool(monkeypatch, pool)
    report = _sample_report()

    summary1 = asyncio.run(register_shadow.register_shadow_rows(report))
    assert summary1["candidates"] == 6
    assert summary1["inserted"] == 6
    assert summary1["skipped_existing"] == 0
    assert summary1["errors"] == 0
    assert summary1["status"] == "ok"

    # Re-run with the same report → every row should be skipped (idempotent),
    # nothing new inserted.
    summary2 = asyncio.run(register_shadow.register_shadow_rows(report))
    assert summary2["inserted"] == 0
    assert summary2["skipped_existing"] == 6
    assert summary2["errors"] == 0


def test_register_shadow_rows_handles_per_row_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    original_execute = pool.conn.execute
    calls = {"n": 0}

    async def flaky_execute(sql: str, *args: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated DB hiccup")
        await original_execute(sql, *args)

    pool.conn.execute = flaky_execute  # type: ignore[assignment]
    _patch_pool(monkeypatch, pool)

    summary = asyncio.run(register_shadow.register_shadow_rows(_sample_report()))
    assert summary["candidates"] == 6
    assert summary["errors"] == 1
    assert summary["inserted"] == 5
    assert summary["status"] == "partial"


def test_register_shadow_rows_noop_on_empty_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    _patch_pool(monkeypatch, pool)
    summary = asyncio.run(register_shadow.register_shadow_rows({"timeframes": {}}))
    assert summary["candidates"] == 0
    assert summary["status"] == "noop"
    assert pool.store == []

def test_slices_carry_approved_features_applied_into_snapshot() -> None:
    """Task #235 — the timeframe-level `approved_features_applied` list
    must be threaded into every base slice's metrics_snapshot so the
    Feature Lab and Model Registry UIs can confirm an approved feature
    actually went live in a specific model version. Meta models are
    excluded — they operate on top of base predictions and don't carry
    the feature schema.
    """
    report = {
        "generated_at": "2026-04-22T00:00:00+00:00",
        "timeframes": {
            "5m": {
                "approved_features_applied": ["rsi_sq_band", "log_rv"],
                "per_coin": {
                    "pepe": {"status": "trained", "version": "v1"},
                },
                "pooled": {"status": "trained", "version": "v1"},
                "specialists": {
                    "momentum": {
                        "status": "trained", "version": "v1",
                        "coin_id": "__specialist_momentum__",
                    },
                },
            },
            "1h": {
                # No approved features applied this TF.
                "per_coin": {},
                "pooled": {"status": "trained", "version": "v1"},
            },
        },
        "meta_models": {
            "5m": {"status": "trained", "version": "vmeta1"},
        },
    }
    slices = register_shadow._slices_from_report(report)
    by_key = {(m, c, t, v): slc for m, c, t, v, slc in slices}
    snap_pepe = register_shadow._slice_metrics_snapshot(
        by_key[("lightgbm", "pepe", "5m", "v1")]
    )
    assert snap_pepe["approved_features_applied"] == ["rsi_sq_band", "log_rv"]
    snap_pooled_5m = register_shadow._slice_metrics_snapshot(
        by_key[("lightgbm", "__pooled__", "5m", "v1")]
    )
    assert snap_pooled_5m["approved_features_applied"] == ["rsi_sq_band", "log_rv"]
    snap_specialist = register_shadow._slice_metrics_snapshot(
        by_key[("lightgbm", "__specialist_momentum__", "5m", "v1")]
    )
    assert snap_specialist["approved_features_applied"] == ["rsi_sq_band", "log_rv"]
    snap_pooled_1h = register_shadow._slice_metrics_snapshot(
        by_key[("lightgbm", "__pooled__", "1h", "v1")]
    )
    assert snap_pooled_1h["approved_features_applied"] == []
    # Meta models do not carry the field.
    snap_meta = register_shadow._slice_metrics_snapshot(
        by_key[("lightgbm-meta", "__meta__", "5m", "vmeta1")]
    )
    assert "approved_features_applied" not in snap_meta

