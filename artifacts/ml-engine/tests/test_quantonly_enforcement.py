"""Task #365 — Quant-Only Enforcement contract tests.

These three asserts are the live invariants the rest of the system relies
on. If any of them regresses, the audit guarantees in
`audit/enforcement-report.md` no longer hold.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.features import build_feature_vector
from app.training.registry import (
    FEATURE_COLUMNS,
    FEATURE_LINEAGE,
    FORBIDDEN_FEATURE_PREFIXES,
    ModelManifest,
    load_model,
)


def _is_forbidden(name: str) -> bool:
    return any(name.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)


def test_feature_columns_have_no_llm_derived_columns():
    bad = [c for c in FEATURE_COLUMNS if _is_forbidden(c)]
    assert not bad, (
        f"FEATURE_COLUMNS leaks LLM-derived feature(s): {bad}. "
        "Quant-only contract requires zero news_/llm_/gpt_/sentiment_/ai_ columns."
    )
    bad_lineage = [c for c in FEATURE_LINEAGE if _is_forbidden(c)]
    assert not bad_lineage, (
        f"FEATURE_LINEAGE still documents LLM features: {bad_lineage}"
    )


def test_load_model_rejects_manifest_with_forbidden_features(tmp_path, monkeypatch):
    # Point the registry at a scratch dir so we can plant a contaminated
    # manifest without touching production state.
    import app.training.registry as registry
    monkeypatch.setattr(registry, "REGISTRY_ROOT", tmp_path)

    bad_features = list(FEATURE_COLUMNS) + ["news_tag_pump"]
    manifest = ModelManifest(
        coin_id="bitcoin",
        timeframe="1h",
        version="20260423T000000Z",
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
    # Plant the manifest by hand (skipping booster files since model_kind=prior).
    d = tmp_path / "bitcoin" / "1h" / manifest.version
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        **manifest.__dict__,
    }))
    (tmp_path / "bitcoin" / "1h" / "latest").write_text(manifest.version)

    loaded = load_model("bitcoin", "1h")
    assert loaded is None, (
        "load_model() must refuse a manifest whose feature_names include any "
        "forbidden prefix; got a loaded model instead — the runtime guard is "
        "broken."
    )


def test_build_feature_vector_emits_no_forbidden_columns_even_with_news_tags():
    # 60 closes is comfortably above MIN_CANDLES_FOR_FEATURES (35), so the
    # vector is fully populated. We deliberately pass a non-empty
    # `news_tags` list — the legacy back-compat argument that USED to
    # inject `news_*` one-hots into the output. Post-#365 the function
    # must IGNORE that argument and emit only quant columns.
    closes = [100.0 + i * 0.1 for i in range(60)]
    vec = build_feature_vector(closes, news_tags=["pump", "exploit_or_hack", "etf_flow"])
    assert vec is not None, "expected a feature vector for 60 candles"
    bad_keys = [k for k in vec.keys() if _is_forbidden(k)]
    assert not bad_keys, (
        f"build_feature_vector emitted forbidden keys {bad_keys} — "
        "live /ml/predict feature builder is leaking LLM-derived columns."
    )
    # Sanity: the canonical quant columns ARE present.
    for required in ("ret1", "rsi14", "macdLine", "atr14", "bbWidth"):
        assert required in vec, f"missing canonical quant feature {required}"


def test_archived_inventory_exists_and_is_consistent():
    repo_root = Path(__file__).resolve().parents[3]
    inv = repo_root / "audit" / "archived_models.json"
    assert inv.exists(), "audit/archived_models.json missing — archive script never ran"
    data = json.loads(inv.read_text())
    assert data["task"] == 365
    assert set(data["forbidden_prefixes"]) == set(FORBIDDEN_FEATURE_PREFIXES)
    assert data["count"] == len(data["models"])
    assert data["count"] > 0, "expected to have archived contaminated models"
    # Spot-check: every archived entry must list at least one forbidden col.
    for m in data["models"][:25]:
        assert m["forbidden_features"], f"empty forbidden_features for {m}"
