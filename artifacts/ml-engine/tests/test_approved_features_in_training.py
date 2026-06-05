"""Task #231 — verify operator-approved Feature Lab features actually
flow into the next training run.

Covers:
  - `apply_approved_features` materializes approved transforms onto the
    labeled dataframe.
  - `extend_feature_columns` splices the new names into the schema
    BEFORE the trailing `coin_idx` categorical so LightGBM positional
    contracts stay intact.
  - `train_one_slice` honors a custom `feature_columns` list — the
    resulting manifest's `feature_names` and `feature_schema_hash`
    reflect the approved feature, so any model trained against the
    old schema must re-enter validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.training.approved_features import (
    apply_approved_features,
    extend_feature_columns,
)
from app.training.registry import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from app.training import train as train_mod


def _toy_labeled_frame(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    base_ts = 1_700_000_000_000
    rows = []
    for i in range(n):
        rsi = float(rng.uniform(20, 80))
        rows.append({
            "coin_id": "pepe",
            "timestamp_ms": base_ts + i * 60_000,
            "label_3class": int(rng.integers(0, 3)),
            "forward_return": float(rng.normal(0, 0.005)),
            **{c: float(rng.normal(0, 1.0)) for c in FEATURE_COLUMNS if c != "coin_idx"},
            "rsi14": rsi,
            "realizedVol": float(abs(rng.normal(0, 0.01))),
            "macdLine": float(rng.normal()),
            "atrPct": float(abs(rng.normal(0, 0.5))),
            "ret5": float(rng.normal(0, 0.01)),
            "ret10": float(rng.normal(0, 0.01)),
            "bbPctB": float(rng.uniform(0, 1)),
        })
    return pd.DataFrame(rows)


def test_apply_approved_features_extends_dataframe_and_schema():
    df = _toy_labeled_frame(50)
    approved = [
        {"name": "rsi_sq_band", "transform_kind": "rsi_squared",
         "source_column": None},
        {"name": "log_rv", "transform_kind": "log_realized_vol",
         "source_column": None},
    ]
    out, added = apply_approved_features(df, approved)
    assert added == ["rsi_sq_band", "log_rv"]
    assert "rsi_sq_band" in out.columns
    assert "log_rv" in out.columns
    # Sanity-check the math on rsi_squared: (rsi - 50)^2.
    expected = (df["rsi14"].to_numpy() - 50.0) ** 2
    np.testing.assert_allclose(out["rsi_sq_band"].to_numpy(), expected)


def test_apply_approved_features_skips_unsupported_kind_without_raising():
    df = _toy_labeled_frame(20)
    approved = [
        {"name": "ok", "transform_kind": "rsi_squared", "source_column": None},
        {"name": "bogus", "transform_kind": "definitely_not_real",
         "source_column": None},
    ]
    out, added = apply_approved_features(df, approved)
    # The bogus one is silently dropped; the valid one still applies.
    assert added == ["ok"]
    assert "ok" in out.columns
    assert "bogus" not in out.columns


def test_extend_feature_columns_keeps_categoricals_at_tail():
    extended = extend_feature_columns(
        FEATURE_COLUMNS, ["rsi_sq_band", "log_rv"],
        categorical=CATEGORICAL_FEATURES,
    )
    # All categorical features land at the end, in their original order.
    assert extended[-len(CATEGORICAL_FEATURES):] == CATEGORICAL_FEATURES
    # The new names are present, BEFORE the categoricals.
    assert "rsi_sq_band" in extended
    assert "log_rv" in extended
    rsi_idx = extended.index("rsi_sq_band")
    cat_idx = extended.index(CATEGORICAL_FEATURES[0])
    assert rsi_idx < cat_idx
    # Idempotency: re-adding a name already present doesn't duplicate.
    extended2 = extend_feature_columns(extended, ["rsi_sq_band"],
                                       categorical=CATEGORICAL_FEATURES)
    assert extended2.count("rsi_sq_band") == 1


def test_prepare_xy_consumes_approved_feature_columns():
    """Without spinning a full LightGBM run we still prove the wiring:
    the approved feature flows through `_prepare_xy` into the model's
    input matrix, and the schema hash that gets stamped on the manifest
    moves — which is what forces models trained on the old schema to
    re-enter validation.
    """
    df = _toy_labeled_frame(50)
    df = train_mod._encode_coin_idx(df, ["pepe"])
    approved = [{"name": "rsi_sq_band", "transform_kind": "rsi_squared",
                 "source_column": None}]
    df_aug, added = apply_approved_features(df, approved)
    assert added == ["rsi_sq_band"]
    extended = extend_feature_columns(
        FEATURE_COLUMNS, added, categorical=CATEGORICAL_FEATURES,
    )

    # Baseline projection has no approved column.
    X_baseline, _ = train_mod._prepare_xy(df_aug)
    assert "rsi_sq_band" not in X_baseline.columns
    # Extended projection does — and carries the right values.
    X_extended, _ = train_mod._prepare_xy(df_aug, feature_columns=extended)
    assert "rsi_sq_band" in X_extended.columns
    np.testing.assert_allclose(
        X_extended["rsi_sq_band"].to_numpy(),
        (df["rsi14"].to_numpy() - 50.0) ** 2,
    )

    h_baseline = train_mod._feature_schema_hash(list(FEATURE_COLUMNS))
    h_extended = train_mod._feature_schema_hash(extended)
    assert h_baseline != h_extended


def test_training_signatures_accept_feature_columns():
    """Guards the public contract: every public training entry-point has
    a `feature_columns` parameter, so `run_training` can thread the
    approved schema all the way down to per-coin / pooled / specialist
    fits. Without this the close-the-loop wiring is silently broken.
    """
    import inspect
    for fn in (train_mod.train_one_slice,
               train_mod.train_timeframe,
               train_mod.train_specialists):
        assert "feature_columns" in inspect.signature(fn).parameters, (
            f"{fn.__name__} must accept feature_columns="
        )


def test_train_one_slice_end_to_end_with_approved_feature(tmp_path, monkeypatch):
    """Task #237 — full end-to-end coverage of the approved-features
    training path. Earlier tests in this file stop at the wiring layer
    (`_prepare_xy`, signatures, schema-hash) because they intentionally
    skip the LightGBM fit. This test goes the rest of the way:

      1. Build a tiny in-memory labeled frame with all canonical features.
      2. Apply an approved transform (`rsi_squared` -> `rsi_sq_band`).
      3. Splice the new column into the feature schema, keeping the
         `coin_idx` categorical at the tail.
      4. Run a real `train_one_slice` against an isolated registry root
         (`monkeypatch` of `REGISTRY_ROOT` -> `tmp_path`) so it doesn't
         collide with the always-on ml-engine workflow.
      5. Reload the persisted model from disk and verify the manifest
         actually carries the approved column AND the schema hash rolled
         forward vs. the canonical baseline. This is the regression
         surface the unit tests can't cover: a serialization or manifest
         persistence bug would silently leave us serving a model whose
         on-disk schema disagrees with the in-memory schema, and the
         scheduler/registry would never notice.
    """
    from app.training import registry as registry_module
    from app.training.train import _feature_schema_hash, train_one_slice

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    df = _toy_labeled_frame(240)
    approved = [
        {"name": "rsi_sq_band", "transform_kind": "rsi_squared",
         "source_column": None},
    ]
    df_aug, added = apply_approved_features(df, approved)
    assert added == ["rsi_sq_band"]
    extended = extend_feature_columns(
        FEATURE_COLUMNS, added, categorical=CATEGORICAL_FEATURES,
    )

    res = train_one_slice(
        df_aug, coin_id="pepe", timeframe="1m", vocab=["pepe"],
        feature_columns=extended,
    )
    assert res["status"] == "trained", res

    # Manifest reported by the trainer reflects the extended schema.
    expected_hash = _feature_schema_hash(extended)
    baseline_hash = _feature_schema_hash(list(FEATURE_COLUMNS))
    assert res["feature_schema_hash"] == expected_hash
    assert res["feature_schema_hash"] != baseline_hash, (
        "schema hash must roll forward when an approved feature is added"
    )

    # And critically, the manifest persisted to disk matches — this is
    # what prevents a serialization regression from silently shipping a
    # model whose on-disk schema disagrees with what the trainer thinks
    # it just produced.
    loaded = registry_module.load_model("pepe", "1m", res["version"])
    assert loaded is not None
    assert "rsi_sq_band" in loaded.manifest.feature_names
    assert loaded.manifest.feature_names == extended
    # `coin_idx` (categorical) stays at the tail of the on-disk schema.
    assert loaded.manifest.feature_names[-len(CATEGORICAL_FEATURES):] == CATEGORICAL_FEATURES
    assert loaded.manifest.feature_schema_hash == expected_hash
    assert loaded.manifest.feature_schema_hash != baseline_hash

    # The deployed booster's feature contract also reflects the extended
    # schema — `_prepare_xy` would happily drop the extra column at
    # inference if the booster were trained against the old list, so
    # this is the second half of the on-disk parity check.
    assert loaded.booster is not None
    assert "rsi_sq_band" in loaded.booster.feature_name()


def _toy_orchestrator_frame(
    coins: list[str], rows_per_coin: int = 320,
) -> pd.DataFrame:
    """Multi-coin labeled frame rich enough to drive `train_timeframe`
    end-to-end:

      - per-coin row count >= MIN_PER_COIN_ROWS (80) so each coin gets a
        per-coin slot.
      - all canonical FEATURE_COLUMNS populated (so `_prepare_xy` works
        for both the baseline and approved-feature schema).
      - the trade-aware label columns (`tp_before_sl_long/short`,
        `opportunity_score`, `forward_window_return_pct`) so each
        directional + volatility specialist resolves a target.
      - `regime` cycles through every key referenced by
        SPECIALIST_REGIME_MAP so each non-volatility specialist's regime
        subset clears MIN_TRAIN_ROWS.
    """
    rng = np.random.default_rng(seed=7)
    base_ts = 1_700_000_000_000
    regimes = [
        "trending_up", "trending_down",
        "range_chop", "low_vol_compression",
        "high_vol_breakout", "panic_liquidation",
    ]
    rows: list[dict] = []
    for ci, coin in enumerate(coins):
        for i in range(rows_per_coin):
            rsi = float(rng.uniform(20, 80))
            label = int(rng.integers(0, 3))
            # Trade-aware barrier flags: line them up with `label` so
            # specialists have a non-degenerate target. Use 0/1 ints
            # (NaN where unresolved) — matches what labels.py emits.
            tp_long = 1 if label == 2 else (0 if label == 0 else 0)
            tp_short = 1 if label == 0 else (0 if label == 2 else 0)
            opp = float(rng.normal(loc=(label - 1) * 0.5, scale=1.0))
            fwd_ret_pct = float(rng.normal(loc=(label - 1) * 0.4, scale=0.8))
            row = {
                "coin_id": coin,
                "timestamp_ms": base_ts + (ci * rows_per_coin + i) * 60_000,
                "label_3class": label,
                "forward_return": float(rng.normal(0, 0.005)),
                "forward_window_return_pct": fwd_ret_pct,
                "tp_before_sl_long": tp_long,
                "tp_before_sl_short": tp_short,
                "opportunity_score": opp,
                "regime": regimes[i % len(regimes)],
                **{c: float(rng.normal(0, 1.0))
                   for c in FEATURE_COLUMNS if c != "coin_idx"},
                "rsi14": rsi,
                "realizedVol": float(abs(rng.normal(0, 0.01))),
                "macdLine": float(rng.normal()),
                "atrPct": float(abs(rng.normal(0, 0.5))),
                "ret5": float(rng.normal(0, 0.01)),
                "ret10": float(rng.normal(0, 0.01)),
                "bbPctB": float(rng.uniform(0, 1)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_run_training_orchestrator_threads_approved_feature_to_every_slot(
    tmp_path, monkeypatch,
):
    """Task #244 — single-slice coverage (task #237) doesn't catch a
    regression in the orchestrator that forgets to thread
    `feature_columns=` to one of its branches (per-coin, pooled, or any
    of the four specialists). A bug there would silently ship a
    partial-schema fleet of models — the per-coin slot would carry the
    approved feature, but pooled or a specialist might be trained on
    the canonical FEATURE_COLUMNS only, and we'd find out at predict
    time when the on-disk schema disagrees with the live feature
    pipeline.

    This test runs `train_timeframe` (which calls `train_specialists`
    internally — together they cover every branch `run_training`
    dispatches into) against a small in-memory multi-coin frame with
    one approved feature, then asserts that EVERY persisted manifest
    in the resulting registry tree carries the approved feature in
    `feature_names` and the same rolled-forward `feature_schema_hash`.
    """
    from app.training import registry as registry_module
    from app.training.train import _feature_schema_hash, train_timeframe
    from app.training.registry import (
        POOLED_COIN_ID, SPECIALIST_KINDS, specialist_coin_id,
    )

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    coins = ["pepe", "bonk"]
    timeframe = "1m"
    df = _toy_orchestrator_frame(coins, rows_per_coin=320)

    approved = [
        {"name": "rsi_sq_band", "transform_kind": "rsi_squared",
         "source_column": None},
    ]
    df_aug, added = apply_approved_features(df, approved)
    assert added == ["rsi_sq_band"]
    extended = extend_feature_columns(
        FEATURE_COLUMNS, added, categorical=CATEGORICAL_FEATURES,
    )
    expected_hash = _feature_schema_hash(extended)
    baseline_hash = _feature_schema_hash(list(FEATURE_COLUMNS))
    assert expected_hash != baseline_hash

    report = train_timeframe(
        df_aug, timeframe, coins, feature_columns=extended,
    )
    assert report["status"] == "trained", report

    # Every per-coin slot must report `trained` — if one fell back to
    # `insufficient_data_per_coin` the orchestrator would've routed it
    # through the pooled slot only and we'd silently lose coverage of
    # the per-coin branch.
    for coin in coins:
        slot = report["per_coin"].get(coin)
        assert slot and slot.get("status") == "trained", (coin, slot)

    # Pooled refit always happens (task #121) — verify it landed.
    assert report["pooled"] and report["pooled"].get("status") == "trained"

    # All four specialists should have trained too — the row counts and
    # `regime` distribution above are sized to clear MIN_TRAIN_ROWS for
    # every regime subset.
    specialists = report.get("specialists") or {}
    for kind in SPECIALIST_KINDS:
        spec = specialists.get(kind)
        assert spec and spec.get("status") == "trained", (kind, spec)

    # Now the actual contract: walk the on-disk registry tree and
    # assert EVERY persisted manifest carries the approved feature and
    # the rolled-forward schema hash. Any branch that forgot to thread
    # `feature_columns=` would show up here as a manifest stamped with
    # the canonical `baseline_hash`.
    expected_slots = (
        [(c, timeframe) for c in coins]
        + [(POOLED_COIN_ID, timeframe)]
        + [(specialist_coin_id(k), timeframe) for k in SPECIALIST_KINDS]
    )
    for coin_id, tf in expected_slots:
        versions = registry_module.list_versions(coin_id, tf)
        assert versions, f"no versions persisted for {coin_id}/{tf}"
        loaded = registry_module.load_model(coin_id, tf, versions[-1])
        assert loaded is not None, (coin_id, tf, versions[-1])
        manifest = loaded.manifest
        assert "rsi_sq_band" in manifest.feature_names, (
            f"{coin_id}/{tf} manifest missing approved feature: "
            f"{manifest.feature_names!r}"
        )
        assert manifest.feature_names == extended, (
            f"{coin_id}/{tf} manifest schema diverges from extended: "
            f"{manifest.feature_names!r}"
        )
        assert manifest.feature_schema_hash == expected_hash, (
            f"{coin_id}/{tf} manifest schema hash mismatch — got "
            f"{manifest.feature_schema_hash!r}, expected {expected_hash!r}"
        )
        assert manifest.feature_schema_hash != baseline_hash, (
            f"{coin_id}/{tf} manifest still on baseline schema hash"
        )
        # Booster's feature contract must match — `_prepare_xy` would
        # silently drop the approved column at inference if the booster
        # were trained against the old list, so this is the on-disk
        # parity check the manifest alone can't make.
        assert loaded.booster is not None, (coin_id, tf)
        assert "rsi_sq_band" in loaded.booster.feature_name(), (
            f"{coin_id}/{tf} booster trained without approved feature"
        )


def test_apply_approved_features_filters_forbidden_prefixes():
    """Task #387 — even if an approved-features row carries a name with a
    forbidden prefix (`news_*`, `llm_*`, `gpt_*`, `sentiment_*`, `ai_*`),
    `apply_approved_features` must drop it. Otherwise the next training
    cycle bakes the column into every manifest's `feature_names` and
    `registry.load_model` rejects the entire fleet at load time.
    """
    df = _toy_labeled_frame(20)
    approved = [
        {"name": "news_pump", "transform_kind": "rsi_squared",
         "source_column": None},
        {"name": "llm_sentiment_burst", "transform_kind": "rsi_squared",
         "source_column": None},
        {"name": "rsi_sq_band", "transform_kind": "rsi_squared",
         "source_column": None},
    ]
    out, added = apply_approved_features(df, approved)
    assert added == ["rsi_sq_band"]
    assert "news_pump" not in out.columns
    assert "llm_sentiment_burst" not in out.columns
    assert "rsi_sq_band" in out.columns


def test_extend_feature_columns_strips_forbidden_names():
    """Defense in depth: even if a caller hands a forbidden name to
    `extend_feature_columns` directly, it must not appear in the
    returned schema. The same rule applies to forbidden columns that
    might somehow already be in the base list.
    """
    extended = extend_feature_columns(
        FEATURE_COLUMNS,
        ["news_pump", "rsi_sq_band", "gpt_signal"],
        categorical=CATEGORICAL_FEATURES,
    )
    assert "news_pump" not in extended
    assert "gpt_signal" not in extended
    assert "rsi_sq_band" in extended

    polluted_base = list(FEATURE_COLUMNS) + ["news_legacy"]
    cleaned = extend_feature_columns(
        polluted_base, [], categorical=CATEGORICAL_FEATURES,
    )
    assert "news_legacy" not in cleaned


def test_run_training_path_never_writes_forbidden_features_to_manifest(
    tmp_path, monkeypatch,
):
    """Task #387 regression pin: simulate the exact bug from the
    enforcement-report (an `app_settings` row reinstating `news_pump`)
    and run a real `train_one_slice` against an isolated registry.
    The persisted manifest must contain ZERO forbidden-prefix columns
    so `registry.load_model` can load it cleanly.
    """
    from app.training import registry as registry_module
    from app.training.train import train_one_slice
    from app.training.registry import FORBIDDEN_FEATURE_PREFIXES

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    df = _toy_labeled_frame(240)
    approved = [
        {"name": "news_pump", "transform_kind": "rsi_squared",
         "source_column": None},
        {"name": "rsi_sq_band", "transform_kind": "rsi_squared",
         "source_column": None},
    ]
    df_aug, added = apply_approved_features(df, approved)
    assert "news_pump" not in added
    extended = extend_feature_columns(
        FEATURE_COLUMNS, added, categorical=CATEGORICAL_FEATURES,
    )
    assert not any(
        any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)
        for c in extended
    )

    res = train_one_slice(
        df_aug, coin_id="pepe", timeframe="1m", vocab=["pepe"],
        feature_columns=extended,
    )
    assert res["status"] == "trained", res

    loaded = registry_module.load_model("pepe", "1m", res["version"])
    assert loaded is not None, (
        "load_model returned None — manifest still carries a forbidden "
        "feature, which is exactly the regression this test pins"
    )
    forbidden = [
        c for c in loaded.manifest.feature_names
        if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)
    ]
    assert forbidden == [], forbidden


def test_apply_approved_features_handles_empty_inputs():
    # Empty approved list → no-op.
    df = _toy_labeled_frame(5)
    out, added = apply_approved_features(df, [])
    assert added == []
    assert list(out.columns) == list(df.columns)
    # Empty dataframe → no-op even if approved is non-empty.
    out2, added2 = apply_approved_features(df.iloc[:0].copy(),
                                            [{"name": "x", "transform_kind": "rsi_squared",
                                              "source_column": None}])
    assert added2 == []
    assert out2.empty
