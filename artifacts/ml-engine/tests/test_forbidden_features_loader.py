"""Task #408 — regression test for forbidden-feature manifest rejection.

Pins the three contracts the remediation in
`artifacts/ml-engine/app/training/registry.py:255-275` (and the
matching `_pick` fall-through in `artifacts/ml-engine/app/main.py:691-735`)
relies on:

  1. `load_model()` returns None when a manifest's `feature_names` carry
     ANY of the FORBIDDEN_FEATURE_PREFIXES (news_/llm_/gpt_/sentiment_/ai_).
     Without this, a historic poisoned manifest could load as if nothing
     happened and ship LLM-derived columns into a "quant-only" decision.
  2. The descending-version walk in `main.py:_pick` reaches an older,
     clean version without raising, even when the `latest` pointer
     names a poisoned version. Without this, the cached `RuntimeError`
     from `_cached_load(latest)` would escape past every fallback and
     500 every /ml/predict for that (coin, tf).
  3. `/ml/predict` returns 200 (not 500) when the latest version is
     poisoned, sourcing predictions from the older clean version.
     This is the runtime proof captured manually in
     `.local/remediation/01-quant-runtime/predict_200.txt` — pinning
     it as a pytest prevents a future agent from accidentally
     re-allowing news_* features without knowing why the guard exists.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import main as main_module
from app.main import app
from app.training import registry as registry_module
from app.training.labels import build_labeled_frame_for_coin
from app.training.registry import (
    FORBIDDEN_FEATURE_PREFIXES,
    list_versions,
    latest_version,
    load_model,
)
from app.training.train import train_timeframe


def _signal_ticks(n: int, start: datetime, drift: float, step_s: int = 60):
    """Trending series with vol noise so the labeler produces all 3
    classes — required for the per-class isotonic calibration to fit."""
    rng = np.random.default_rng(7)
    out = []
    p = 100.0
    for i in range(n):
        p *= (1.0 + drift + rng.normal(0, 0.005))
        out.append((start + timedelta(seconds=i * step_s), p))
    return out


def _build_dataset(coins_drift: list[tuple[str, float, int]]) -> pd.DataFrame:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    frames = []
    for coin, drift, n in coins_drift:
        frames.append(
            build_labeled_frame_for_coin(coin, "1m", _signal_ticks(n, base, drift))
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )


def _plant_poisoned_latest(
    registry_root, coin_id: str, timeframe: str, clean_version: str,
) -> str:
    """Drop a poisoned manifest into a NEWER (lex-greater) version dir
    for `(coin_id, timeframe)` and bump the `latest` pointer to it.

    The poisoned manifest is `model_kind="prior"` so we don't need a
    real LightGBM booster on disk — `load_model` rejects the manifest
    BEFORE it tries to read any booster file (the forbidden-prefix
    check runs immediately after `ModelManifest(**manifest_dict)`).

    Returns the planted poisoned version id.
    """
    poisoned_version = "29991231T235959Z-poisoned"
    assert poisoned_version > clean_version, (
        "poisoned version must sort AFTER the clean one so list_versions "
        "presents it as 'newer' and `_pick` walks BACK through it to the "
        "clean version. got poisoned=%r clean=%r"
        % (poisoned_version, clean_version)
    )
    d = registry_root / coin_id / timeframe / poisoned_version
    d.mkdir(parents=True, exist_ok=True)
    poisoned_manifest = {
        "coin_id": coin_id,
        "timeframe": timeframe,
        "version": poisoned_version,
        # The smoking gun — at least one column under a forbidden prefix.
        # We include every prefix family for belt-and-braces: any one of
        # them must be enough to trip the guard.
        "feature_names": [
            "ret1",
            "rsi14",
            "news_tag_pump",
            "llm_summary_score",
            "gpt_topic_id",
            "sentiment_compound",
            "ai_macro_regime",
        ],
        "coin_vocab": [coin_id],
        "n_train_rows": 100,
        "n_test_rows": 20,
        "metrics": {"auc": 0.55},
        "baseline_metrics": {"auc": 0.50},
        "threshold_pct": 0.5,
        "horizon_candles": 4,
        "class_return_means_pct": [-0.5, 0.0, 0.5],
        "model_kind": "prior",
        "prior_probs": [0.33, 0.34, 0.33],
    }
    (d / "manifest.json").write_text(json.dumps(poisoned_manifest))
    # Bump the `latest` pointer so production resolution paths see the
    # poisoned slot first — exactly the on-disk shape that produced the
    # B-PRED-500 outage before the remediation landed.
    (registry_root / coin_id / timeframe / "latest").write_text(poisoned_version)
    return poisoned_version


@pytest.fixture(scope="module")
def poisoned_latest_clean_fallback(tmp_path_factory):
    """Train a real (clean) per-coin model into a tmp registry, then
    plant a NEWER poisoned manifest as the `latest` pointer for one
    of the coins. The fixture returns:
        {"coin": str, "tf": str, "clean_version": str, "poisoned_version": str,
         "registry_root": Path}

    Module-scoped so the (~30s) training only runs once across all
    three contract tests in this file. We use a module-scoped
    `MonkeyPatch` because pytest's built-in `monkeypatch` fixture is
    function-scoped and can't outlive a single test.
    """
    mp = pytest.MonkeyPatch()
    tmp_path = tmp_path_factory.mktemp("forbidden_features_registry")
    mp.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    # Mirror the predict_test fixture so we get real LightGBM models
    # for "pepe" and "bonk" on timeframe "1m".
    df = _build_dataset([("pepe", 0.0010, 200), ("bonk", -0.0005, 200)])
    res = train_timeframe(df, "1m", coin_ids=["pepe", "bonk"])
    assert res["status"] == "trained", res

    coin, tf = "pepe", "1m"
    clean_version = latest_version(coin, tf)
    assert clean_version is not None, (
        "fixture didn't persist a clean per-coin model for pepe/1m; the "
        "rest of this test would be vacuous"
    )

    poisoned_version = _plant_poisoned_latest(tmp_path, coin, tf, clean_version)

    # Drop the lru_cache so the new on-disk shape is re-resolved.
    main_module._cached_load.cache_clear()

    yield {
        "coin": coin,
        "tf": tf,
        "clean_version": clean_version,
        "poisoned_version": poisoned_version,
        "registry_root": tmp_path,
    }

    main_module._cached_load.cache_clear()
    mp.undo()


# ---------------------------------------------------------------------------
# Contract 1 — load_model() returns None for a poisoned manifest.
# ---------------------------------------------------------------------------
def test_load_model_returns_none_for_poisoned_latest(poisoned_latest_clean_fallback):
    f = poisoned_latest_clean_fallback
    # `latest` MUST be the poisoned version we just planted — otherwise
    # the rest of the asserts wouldn't be testing the guard.
    assert latest_version(f["coin"], f["tf"]) == f["poisoned_version"]

    # Explicit version load: the guard rejects.
    assert load_model(f["coin"], f["tf"], f["poisoned_version"]) is None, (
        "load_model() must return None for a manifest whose feature_names "
        "include any FORBIDDEN_FEATURE_PREFIXES; got a loaded model — "
        "the runtime guard in registry.py:load_model is broken."
    )

    # Default-version load (no version arg → resolves `latest`): same
    # contract — must return None because `latest` IS the poisoned
    # version.
    assert load_model(f["coin"], f["tf"]) is None, (
        "load_model() with no version arg must also return None when "
        "the `latest` pointer names a poisoned manifest."
    )


# ---------------------------------------------------------------------------
# Contract 2 — the descending-version walk reaches the older clean version.
# ---------------------------------------------------------------------------
def test_descending_walk_reaches_clean_older_version(poisoned_latest_clean_fallback):
    f = poisoned_latest_clean_fallback

    # Sanity: list_versions returns BOTH versions (sorted lex-asc), and
    # the newest (== last) is the poisoned one we just planted.
    versions = list_versions(f["coin"], f["tf"])
    assert f["clean_version"] in versions
    assert f["poisoned_version"] in versions
    assert versions[-1] == f["poisoned_version"], (
        "list_versions must sort lex-asc so reversed() walks newest-first"
    )

    # The clean older version itself MUST still load — otherwise the
    # fallback has nothing to fall back TO and contract #3 would pass
    # vacuously via the 503 path instead of the 200 path.
    clean = load_model(f["coin"], f["tf"], f["clean_version"])
    assert clean is not None, (
        f"clean version {f['clean_version']} stopped loading — the "
        "fixture itself is broken (or the on-disk training output drifted)."
    )
    # Belt-and-braces: that clean manifest carries ZERO forbidden columns.
    leaked = [
        c for c in clean.manifest.feature_names
        if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)
    ]
    assert not leaked, (
        f"clean fixture model leaks forbidden columns {leaked}; the "
        "training pipeline regressed on the quant-only contract."
    )

    # Walk the way `_pick` in main.py walks: skip the poisoned `latest`,
    # pick the first older version that loads. This must NOT raise and
    # must return the clean version — proving the fall-through wired
    # by the remediation actually wires through end-to-end.
    picked = None
    for v in reversed(versions):
        m = load_model(f["coin"], f["tf"], v)
        if m is not None:
            picked = m
            break
    assert picked is not None, (
        "descending-version walk found NO loadable version — the "
        "_pick fall-through in main.py would 500 every /predict."
    )
    assert picked.manifest.version == f["clean_version"], (
        f"fall-through picked {picked.manifest.version!r}; expected "
        f"{f['clean_version']!r} (the clean older version)."
    )


# ---------------------------------------------------------------------------
# Contract 3 — /ml/predict returns 200 when the latest version is poisoned.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_predict_200_when_latest_version_is_poisoned(
    poisoned_latest_clean_fallback, monkeypatch,
):
    f = poisoned_latest_clean_fallback

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks = _signal_ticks(220, base, 0.0010)

    async def fake_fetch(coin_id: str, lookback_ms: int, now=None):
        return ticks

    monkeypatch.setattr(db_module, "fetch_real_ticks", fake_fetch)
    monkeypatch.setattr(main_module, "fetch_real_ticks", fake_fetch)

    with TestClient(app) as client:
        r = client.post(
            "/ml/predict",
            json={"coinId": f["coin"], "timeframe": f["tf"]},
        )
        # The whole point: a poisoned `latest` must NOT 500. This is
        # the runtime contract proven manually in
        # .local/remediation/01-quant-runtime/predict_200.txt.
        assert r.status_code == 200, (
            f"/ml/predict returned {r.status_code} when latest is poisoned; "
            f"expected 200 via fall-through to clean older version. body={r.text}"
        )
        body = r.json()
        # Predictions must come from the CLEAN older version, not the
        # poisoned latest. modelVersion is the public proof.
        assert body.get("modelVersion") == f["clean_version"], (
            f"predict served from {body.get('modelVersion')!r}; expected "
            f"clean older version {f['clean_version']!r}. "
            "Fall-through is silently picking something else."
        )
        # And the served model is a real LightGBM booster, not a prior
        # fallback or a partial response.
        assert body.get("source") == "lightgbm", (
            f"expected source=lightgbm from clean older model; got "
            f"{body.get('source')!r}"
        )
        # Probabilities form a valid distribution — sanity that the
        # prediction is real, not a stub.
        assert math.isclose(
            body["probUp"] + body["probDown"] + body["probStable"],
            1.0,
            abs_tol=1e-6,
        )
