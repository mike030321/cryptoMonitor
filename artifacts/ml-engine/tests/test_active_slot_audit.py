"""Task #376 — `audit_active_slots` contract.

After every retrain, the registry must hold a fresh, non-archived
manifest for every (coin, timeframe) pair the api-server lists as
active. The Quant-Only archive sweep
(scripts/archive_contaminated_models.py) renames `latest` pointers out
of the way; a future regression where `run_training` fails to write a
replacement would silently pause the agent for that slot. This test
pins the audit helper that catches that regression.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.training import registry as registry_module
from app.training.registry import (
    POOLED_COIN_ID,
    ModelManifest,
    audit_active_slots,
    save_model,
)


def _seed_prior_slot(root: Path, coin_id: str, tf: str) -> None:
    """Plant a real prior-only model so `resolve_model` returns it.
    Prior models need no booster on disk, so the test stays cheap."""
    manifest = ModelManifest(
        coin_id=coin_id,
        timeframe=tf,
        version="20260423T000000Z",
        feature_names=[],
        coin_vocab=[coin_id],
        n_train_rows=100,
        n_test_rows=20,
        metrics={"auc": 0.55},
        baseline_metrics={"auc": 0.50},
        threshold_pct=0.5,
        horizon_candles=4,
        class_return_means_pct=[-0.5, 0.0, 0.5],
        model_kind="prior",
        prior_probs=[0.33, 0.34, 0.33],
    )
    # save_model writes the version dir + the `latest` pointer.
    # Patching REGISTRY_ROOT happens in the test via monkeypatch.
    save_model(
        coin_id=coin_id,
        timeframe=tf,
        version="20260423T000000Z",
        booster=None,
        calibrators=None,
        manifest=manifest,
    )
    assert (root / coin_id / tf / "latest").exists()


def test_audit_passes_when_every_slot_resolves(monkeypatch, tmp_path):
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    coins = ["bitcoin", "ethereum"]
    tfs = ["1h", "1d"]
    for c in coins:
        for tf in tfs:
            _seed_prior_slot(tmp_path, c, tf)

    result = audit_active_slots(coins, tfs)
    assert result["ok"] is True
    assert result["n_checked"] == 4
    assert result["missing"] == []


def test_audit_falls_through_to_pooled_fallback(monkeypatch, tmp_path):
    """Per-coin slot missing is fine *if* the pooled fallback covers it
    — the same fallback `resolve_model` uses at inference. The audit
    must agree with the resolver: a slot is only "missing" when both
    levels fail."""
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    # Only seed the pooled fallback for 1h, NOT the per-coin slot.
    _seed_prior_slot(tmp_path, POOLED_COIN_ID, "1h")

    result = audit_active_slots(["bitcoin"], ["1h"])
    assert result["ok"] is True, result
    assert result["missing"] == []


def test_audit_flags_archived_slot(monkeypatch, tmp_path):
    """Simulate the `archive_contaminated_models.py` sweep: rename the
    `latest` pointer out of the way. The audit MUST flag the slot as
    missing — that is exactly the regression task #367's verification
    pass surfaced and task #376 guards against."""
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    _seed_prior_slot(tmp_path, "bitcoin", "1h")

    # Mimic the archive sweep — rename version dir + latest pointer.
    version_dir = tmp_path / "bitcoin" / "1h" / "20260423T000000Z"
    version_dir.rename(version_dir.with_name(
        version_dir.name + ".archived_pre_quantonly"
    ))
    latest = tmp_path / "bitcoin" / "1h" / "latest"
    latest.rename(latest.with_name("latest.archived"))

    result = audit_active_slots(["bitcoin"], ["1h"])
    assert result["ok"] is False
    assert result["n_checked"] == 1
    assert len(result["missing"]) == 1
    row = result["missing"][0]
    assert row["coin_id"] == "bitcoin"
    assert row["timeframe"] == "1h"
    # Per-coin gone AND no pooled fallback either — both diagnostics
    # must be present so an operator knows the slot is fully empty.
    assert "pooled_fallback_missing" in row["slot_state"]


def test_audit_flags_dangling_latest_pointer(monkeypatch, tmp_path):
    """`latest` exists but points at a version dir that has been
    deleted/archived. The resolver returns None; the audit must flag."""
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    _seed_prior_slot(tmp_path, "bitcoin", "1h")
    # Delete the version dir but leave `latest` pointing at it.
    version_dir = tmp_path / "bitcoin" / "1h" / "20260423T000000Z"
    for f in version_dir.iterdir():
        f.unlink()
    version_dir.rmdir()

    result = audit_active_slots(["bitcoin"], ["1h"])
    assert result["ok"] is False
    assert result["missing"][0]["slot_state"].startswith("latest_dangling")


def test_audit_flags_missing_per_coin_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    # Nothing seeded at all.
    result = audit_active_slots(["bitcoin"], ["1h"])
    assert result["ok"] is False
    assert result["missing"][0]["slot_state"].startswith("no_per_coin_dir")
