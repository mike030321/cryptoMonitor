"""Task #451 — auto-prune contaminated model artifacts.

Pins three contracts the post-training janitor relies on:

  1. `prune_contaminated_versions()` deletes any version directory whose
     manifest's `feature_names` includes a `FORBIDDEN_FEATURE_PREFIXES`
     column, frees disk space, and clears the `latest` pointer that
     names that version. Clean versions are left untouched.

  2. The deleted entries are merged into `audit/archived_models.json`
     (preserving any existing inventory rows from the original Task #365
     one-shot sweep) so an auditor can reconstruct what was removed.

  3. `save_model()` refuses to promote a manifest with forbidden
     feature names — the trainer cannot keep producing contaminated
     dirs that the janitor then has to clean up every cycle.
"""
from __future__ import annotations

import json

import pytest

from app.training import registry as registry_module
from app.training.registry import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_PREFIXES,
    ModelManifest,
    latest_version,
    list_versions,
    prune_contaminated_versions,
    save_model,
)


def _plant_version(
    root, coin: str, tf: str, version: str, *, contaminated: bool,
) -> None:
    """Write a minimal manifest + a few bytes of payload so the version
    dir is non-empty and the size accounting in the janitor has
    something to count."""
    feats = list(FEATURE_COLUMNS)
    if contaminated:
        feats = feats + ["news_tag_pump"]
    manifest = {
        "coin_id": coin,
        "timeframe": tf,
        "version": version,
        "feature_names": feats,
        "coin_vocab": [coin],
        "n_train_rows": 100,
        "n_test_rows": 20,
        "metrics": {"auc": 0.55},
        "baseline_metrics": {"auc": 0.50},
        "threshold_pct": 0.5,
        "horizon_candles": 4,
        "class_return_means_pct": [-0.5, 0.0, 0.5],
        "model_kind": "prior",
        "prior_probs": [0.33, 0.34, 0.33],
        "fold_metrics": [],
        "note": "",
        "directional_call_share": None,
        "directional_call_share_n": None,
        "directional_call_share_source": None,
        "served_predictor_kind": "prior",
        "has_regression_head": False,
        "regression_head_stats": None,
        "gates_alignment": None,
        "specialist_kind": None,
        "regime_subset": [],
        "feature_schema_hash": None,
        "training_window": None,
        "bars_source": None,
        "bars_native_cadence_ms": None,
        "bars_by_native_cadence": {},
        "cadence_mixed": False,
        "cadence_mitigation": None,
    }
    d = root / coin / tf / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest))
    # Add a non-trivial filler so freed_bytes is > 0 and we exercise the
    # rmtree path on a multi-file dir, not just an empty one.
    (d / "prior.json").write_text(json.dumps({"prior_probs": [0.33, 0.34, 0.33]}))
    (d / "padding.bin").write_bytes(b"\0" * 4096)


def test_prune_deletes_contaminated_and_keeps_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    _plant_version(tmp_path, "bitcoin", "1h", "20260101T000000Z", contaminated=False)
    _plant_version(tmp_path, "bitcoin", "1h", "20260201T000000Z", contaminated=True)
    # `latest` names the contaminated version — the janitor must clear it.
    (tmp_path / "bitcoin" / "1h" / "latest").write_text("20260201T000000Z")

    audit_dir = tmp_path / "_audit"
    result = prune_contaminated_versions(audit_dir=audit_dir)

    assert result["deleted"] == 1, result
    assert result["freed_bytes"] > 0
    [rec] = result["models"]
    assert rec["coin_id"] == "bitcoin"
    assert rec["timeframe"] == "1h"
    assert rec["version"] == "20260201T000000Z"
    assert any(c.startswith(p) for c in rec["forbidden_features"]
               for p in FORBIDDEN_FEATURE_PREFIXES)

    # Clean version is still on disk and selectable; contaminated is gone.
    assert "20260101T000000Z" in list_versions("bitcoin", "1h")
    assert "20260201T000000Z" not in list_versions("bitcoin", "1h")
    # `latest` was cleared because it pointed at the deleted version.
    assert latest_version("bitcoin", "1h") is None

    # Audit inventory was created and contains the deleted entry.
    inv = audit_dir / "archived_models.json"
    assert inv.exists()
    data = json.loads(inv.read_text())
    assert data["count"] == 1
    assert data["models"][0]["version"] == "20260201T000000Z"
    assert set(data["forbidden_prefixes"]) == set(FORBIDDEN_FEATURE_PREFIXES)


def test_prune_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    _plant_version(tmp_path, "ethereum", "1h", "20260201T000000Z", contaminated=True)

    audit_dir = tmp_path / "_audit"
    first = prune_contaminated_versions(audit_dir=audit_dir)
    second = prune_contaminated_versions(audit_dir=audit_dir)

    assert first["deleted"] == 1
    assert second["deleted"] == 0, "second pass should be a no-op"
    # Inventory keeps the single record (no duplicate row appended on
    # the no-op pass).
    data = json.loads((audit_dir / "archived_models.json").read_text())
    assert data["count"] == 1


def test_prune_preserves_existing_inventory(tmp_path, monkeypatch):
    """Janitor must MERGE into an existing audit/archived_models.json
    (e.g. the one written by the original Task #365 one-shot script),
    never overwrite it."""
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    audit_dir = tmp_path / "_audit"
    audit_dir.mkdir(parents=True)
    pre_existing = {
        "task": 365,
        "rule": "legacy",
        "forbidden_prefixes": list(FORBIDDEN_FEATURE_PREFIXES),
        "count": 1,
        "models": [{
            "coin_id": "litecoin",
            "timeframe": "1m",
            "version": "20251201T000000Z",
            "forbidden_features": ["news_tag_pump"],
        }],
    }
    (audit_dir / "archived_models.json").write_text(json.dumps(pre_existing))

    _plant_version(tmp_path, "solana", "1h", "20260301T000000Z", contaminated=True)
    prune_contaminated_versions(audit_dir=audit_dir)

    data = json.loads((audit_dir / "archived_models.json").read_text())
    assert data["count"] == 2
    versions = {m["version"] for m in data["models"]}
    assert "20251201T000000Z" in versions  # legacy preserved
    assert "20260301T000000Z" in versions  # new entry appended
    # Legacy task field is preserved (the test in test_quantonly_enforcement.py
    # reads it from the production file).
    assert data["task"] == 365


def test_save_model_refuses_forbidden_feature_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    bad_features = list(FEATURE_COLUMNS) + ["news_tag_pump"]
    manifest = ModelManifest(
        coin_id="bitcoin",
        timeframe="1h",
        version="20260301T000000Z",
        feature_names=bad_features,
        coin_vocab=["bitcoin"],
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
    with pytest.raises(ValueError, match="forbidden feature columns"):
        save_model(
            "bitcoin", "1h", "20260301T000000Z",
            booster=None, calibrators=None, manifest=manifest,
        )
    # Nothing was created on disk — the gate ran BEFORE mkdir.
    assert not (tmp_path / "bitcoin" / "1h" / "20260301T000000Z").exists()


def test_prune_handles_empty_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path / "does_not_exist")
    result = prune_contaminated_versions(audit_dir=tmp_path / "_audit")
    assert result == {"deleted": 0, "freed_bytes": 0, "models": []}


# --- Task #458 — row cap + rotation on archived_models.json -----------------

def _make_record(version: str, deleted_at: str | None) -> dict:
    rec = {
        "coin_id": "bitcoin",
        "timeframe": "1h",
        "version": version,
        "forbidden_features": ["news_tag_pump"],
    }
    if deleted_at is not None:
        rec["deleted_at"] = deleted_at
    return rec


def test_inventory_under_cap_does_not_rotate(tmp_path):
    audit_dir = tmp_path / "_audit"
    rows = [_make_record(f"v{i:05d}", "2026-04-24T11:38:00+00:00") for i in range(50)]
    registry_module._append_archived_inventory(audit_dir, rows)

    live = json.loads((audit_dir / "archived_models.json").read_text())
    assert live["count"] == 50
    assert len(live["models"]) == 50
    assert live["max_rows"] == registry_module.ARCHIVED_MODELS_MAX_ROWS
    # No sibling rotation files were created.
    siblings = list(audit_dir.glob("archived_models.*.json"))
    assert siblings == []


def test_inventory_over_cap_rolls_oldest_into_dated_sibling(
    tmp_path, monkeypatch,
):
    """When the merged list grows beyond the cap, the OLDEST overflow
    rows are spilled into `archived_models.<yyyymm>.json` and the live
    file keeps only the most recent ARCHIVED_MODELS_MAX_ROWS entries.
    The audit-report scripts that read `count` / `models` from the
    live file still see a consistent payload.
    """
    monkeypatch.setattr(registry_module, "ARCHIVED_MODELS_MAX_ROWS", 100)
    audit_dir = tmp_path / "_audit"

    # Seed with 95 already-rolled rows from Feb 2026 (under cap).
    initial = [
        _make_record(f"v{i:05d}", "2026-02-15T00:00:00+00:00") for i in range(95)
    ]
    registry_module._append_archived_inventory(audit_dir, initial)
    assert json.loads((audit_dir / "archived_models.json").read_text())["count"] == 95
    assert not list(audit_dir.glob("archived_models.*.json"))

    # Now push 30 more rows from April 2026 — total 125, cap 100, so
    # 25 oldest rows (all Feb 2026) must spill out.
    new = [_make_record(f"w{i:05d}", "2026-04-01T00:00:00+00:00") for i in range(30)]
    registry_module._append_archived_inventory(audit_dir, new)

    live = json.loads((audit_dir / "archived_models.json").read_text())
    assert live["count"] == 100
    assert len(live["models"]) == 100
    assert live["last_rolled_over_count"] == 25
    assert "last_rolled_over_at" in live
    # The 100 retained rows are the most recent ones: tail of Feb (70)
    # + all 30 April rows. The first row in the live file should be
    # the 26th seeded Feb row (`v00025`).
    assert live["models"][0]["version"] == "v00025"
    assert live["models"][-1]["version"] == "w00029"

    # Spillover landed in the Feb sibling.
    sibling = audit_dir / "archived_models.202602.json"
    assert sibling.exists()
    spilled = json.loads(sibling.read_text())
    assert spilled["bucket"] == "202602"
    assert spilled["count"] == 25
    assert {m["version"] for m in spilled["models"]} == {
        f"v{i:05d}" for i in range(25)
    }
    assert set(spilled["forbidden_prefixes"]) == set(
        registry_module.FORBIDDEN_FEATURE_PREFIXES
    )


def test_inventory_overflow_buckets_by_yyyymm(tmp_path, monkeypatch):
    """Mixed-timestamp overflow is bucketed per `yyyymm` so a year of
    spilled rows doesn't pile into a single file. Legacy rows without
    `deleted_at` land in the `legacy` bucket so they stay grouped."""
    monkeypatch.setattr(registry_module, "ARCHIVED_MODELS_MAX_ROWS", 5)
    audit_dir = tmp_path / "_audit"

    # 4 legacy + 4 Jan 2026 + 4 Mar 2026 rows = 12 total, cap 5.
    # Overflow = 7 oldest rows (4 legacy + 3 Jan).
    legacy = [_make_record(f"L{i}", None) for i in range(4)]
    jan = [_make_record(f"J{i}", "2026-01-10T00:00:00+00:00") for i in range(4)]
    mar = [_make_record(f"M{i}", "2026-03-20T00:00:00+00:00") for i in range(4)]
    registry_module._append_archived_inventory(audit_dir, legacy + jan + mar)

    live = json.loads((audit_dir / "archived_models.json").read_text())
    assert live["count"] == 5
    # Live file keeps the 5 newest: J3 + all 4 March rows.
    assert [m["version"] for m in live["models"]] == ["J3", "M0", "M1", "M2", "M3"]

    # 4 legacy rows -> archived_models.legacy.json
    legacy_file = audit_dir / "archived_models.legacy.json"
    assert legacy_file.exists()
    legacy_payload = json.loads(legacy_file.read_text())
    assert legacy_payload["bucket"] == "legacy"
    assert legacy_payload["count"] == 4
    assert {m["version"] for m in legacy_payload["models"]} == {"L0", "L1", "L2", "L3"}

    # 3 Jan rows -> archived_models.202601.json
    jan_file = audit_dir / "archived_models.202601.json"
    assert jan_file.exists()
    jan_payload = json.loads(jan_file.read_text())
    assert jan_payload["count"] == 3
    assert {m["version"] for m in jan_payload["models"]} == {"J0", "J1", "J2"}


def test_inventory_rotation_appends_to_existing_sibling(tmp_path, monkeypatch):
    """A second rotation pass into the same yyyymm bucket must MERGE
    into the existing sibling file, never overwrite it."""
    monkeypatch.setattr(registry_module, "ARCHIVED_MODELS_MAX_ROWS", 3)
    audit_dir = tmp_path / "_audit"

    # First pass: 5 rows, 2 spill into 202604.
    first = [_make_record(f"a{i}", "2026-04-01T00:00:00+00:00") for i in range(5)]
    registry_module._append_archived_inventory(audit_dir, first)
    sibling = audit_dir / "archived_models.202604.json"
    assert json.loads(sibling.read_text())["count"] == 2

    # Second pass: 3 more rows in the same bucket -> 6 total in live,
    # 3 spill out and merge into the existing sibling (now 5 total).
    second = [_make_record(f"b{i}", "2026-04-15T00:00:00+00:00") for i in range(3)]
    registry_module._append_archived_inventory(audit_dir, second)

    live = json.loads((audit_dir / "archived_models.json").read_text())
    assert live["count"] == 3
    assert [m["version"] for m in live["models"]] == ["b0", "b1", "b2"]

    spilled = json.loads(sibling.read_text())
    assert spilled["count"] == 5
    assert [m["version"] for m in spilled["models"]] == [
        "a0", "a1", "a2", "a3", "a4",
    ]


def test_existing_inventory_test_invariants_still_hold(tmp_path, monkeypatch):
    """Mirrors `test_archived_inventory_exists_and_is_consistent` from
    test_quantonly_enforcement.py against a rotated file: the live
    inventory must still satisfy every assertion that test makes."""
    monkeypatch.setattr(registry_module, "ARCHIVED_MODELS_MAX_ROWS", 50)
    audit_dir = tmp_path / "_audit"
    # Pre-seed the legacy `task: 365` field that the production audit
    # report relies on — `_append_archived_inventory` must preserve it
    # even after rotation.
    audit_dir.mkdir(parents=True)
    (audit_dir / "archived_models.json").write_text(json.dumps({
        "task": 365,
        "rule": "legacy",
        "forbidden_prefixes": list(registry_module.FORBIDDEN_FEATURE_PREFIXES),
        "count": 0,
        "models": [],
    }))

    # Push 80 records — cap 50 -> 30 spill, 50 retained.
    rows = [_make_record(f"v{i:05d}", "2026-04-24T11:38:00+00:00") for i in range(80)]
    registry_module._append_archived_inventory(audit_dir, rows)

    live = json.loads((audit_dir / "archived_models.json").read_text())
    # The four invariants the existing audit test pins:
    assert live["task"] == 365  # legacy field preserved
    assert set(live["forbidden_prefixes"]) == set(
        registry_module.FORBIDDEN_FEATURE_PREFIXES
    )
    assert live["count"] == len(live["models"])
    assert live["count"] > 0
    for m in live["models"][:25]:
        assert m["forbidden_features"]
