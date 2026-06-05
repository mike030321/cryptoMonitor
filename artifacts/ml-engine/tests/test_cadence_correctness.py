"""Behavior-level cadence-correctness tests for the price store -> trainer
pipeline.

These tests are the executable contract for the schema fix proposed in
``artifacts/ml-engine/reports/20260423T000000Z-schema-audit.md`` (task #315).
They are EXPECTED TO FAIL on `main` today — the schema fix has not landed —
and to pass the moment task #317 ships its migration + trainer changes.

Each test exercises an actual code path with crafted mixed-cadence inputs
and asserts the post-fix behavior (rejection / quarantine / no merge),
rather than just probing schema or signatures.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_SCHEMA_DIR = REPO_ROOT / "lib" / "db" / "src" / "schema"
ML_MODELS = REPO_ROOT / "artifacts" / "ml-engine" / "models"


@pytest.fixture(autouse=True)
def _add_ml_engine_to_sys_path(monkeypatch):
    """Make `app.features` importable from any cwd, and ensure each test
    re-imports the module fresh so a post-fix module is picked up."""
    import sys

    ml_engine_root = REPO_ROOT / "artifacts" / "ml-engine"
    monkeypatch.syspath_prepend(str(ml_engine_root))
    sys.modules.pop("app.features", None)
    yield
    sys.modules.pop("app.features", None)


def _import_features():
    import importlib

    return importlib.import_module("app.features")


def test_daily_rows_are_not_silently_merged_into_5m_bars():
    """Behavior contract: a stream of *daily-cadence* rows must not silently
    produce 5-minute candles.

    Today's `resample_to_candles` happily buckets any tick into any
    `bucket_ms`; with two rows 24h apart and `bucket_ms=300_000` it returns
    two 5m closes — that is the contamination this fix targets. After the
    fix, the function must accept a `min_input_cadence_ms` cap and raise
    ``CadenceMismatchError`` when the inter-arrival of the input rows is
    coarser than the requested bucket, so coarser bars cannot feed finer
    buckets.
    """
    features = _import_features()
    resample = features.resample_to_candles
    base = datetime(2026, 4, 22, 0, 0, 0, tzinfo=timezone.utc)
    daily_ticks = [
        (base, 100.0),
        (base + timedelta(days=1), 110.0),
    ]

    err_cls = getattr(features, "CadenceMismatchError", None)
    assert err_cls is not None, (
        "app.features.CadenceMismatchError must be exported and raised when "
        "input row cadence is coarser than the requested bucket — see "
        "20260423T000000Z-schema-audit.md §3"
    )

    with pytest.raises(err_cls):
        resample(daily_ticks, bucket_ms=5 * 60 * 1000, min_input_cadence_ms=5 * 60 * 1000)


def test_resample_quarantines_coarser_rows_within_a_bucket():
    """Behavior contract: when a fine-cadence stream contains a single
    coarser-cadence row landing inside a 5m bucket, that row must NOT be
    used as the bucket close.

    We feed five rows: four 30-second ticks at $100 plus one daily row at
    $999 timestamped two minutes after the last tick (so it falls in the
    same 5m bucket). Today's last-tick-wins behavior returns ``[999.0]`` —
    the daily contaminant becomes the 5m close. After the fix, the daily
    row is quarantined and the bucket close is the last fine-cadence tick
    ($100.0).
    """
    features = _import_features()
    resample = features.resample_to_candles
    base = datetime(2026, 4, 22, 0, 0, 0, tzinfo=timezone.utc)
    ticks = [
        (base + timedelta(seconds=0), 100.0),
        (base + timedelta(seconds=30), 100.0),
        (base + timedelta(seconds=60), 100.0),
        (base + timedelta(seconds=90), 100.0),
        # The coarser-cadence row that today's code wrongly accepts as close:
        (base + timedelta(seconds=180), 999.0),
    ]
    closes = resample(ticks, bucket_ms=5 * 60 * 1000, min_input_cadence_ms=60 * 1000)
    assert isinstance(closes, list) and len(closes) == 1, (
        f"expected exactly one 5m close from a single 5m bucket, got {closes!r}"
    )
    assert math.isclose(closes[0], 100.0), (
        "the daily-cadence contaminant ($999) must be quarantined; the bucket "
        f"close must be the last fine-cadence tick ($100), not {closes[0]!r}"
    )


def test_trainer_provenance_records_native_cadence_and_refuses_mixed():
    """Behavior contract: every persisted slice manifest must carry a
    ``bars_by_native_cadence`` map AND a ``cadence_mixed`` boolean. A
    verification gate that loads such a manifest with ``cadence_mixed=true``
    and an unmitigated mixture of native cadences must NOT promote the
    slice (this is the verification-side enforcement of the schema fix).

    Resilience guard (task #450): pre-#317 manifests don't carry the
    cadence-provenance fields and never will — backfilling them onto disk
    is a one-shot maintenance op, not part of the trainer contract. So we
    enforce the on-disk schema check ONLY against manifests written AFTER
    task #317 landed (versions >= ``CADENCE_PROVENANCE_CUTOFF_VERSION``).
    To make sure the test never silently passes when nothing modern is on
    disk (e.g. after a prune leaves only legacy manifests), we ALSO build
    a fresh ``ModelManifest`` dataclass instance and assert its serialized
    form carries both fields — that way the contract still has teeth even
    on an empty / fully-pruned model registry.
    """
    # Versions are produced by `make_version()` as
    # ``datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")`` — lexicographic
    # comparison matches chronological order, so we can pick a cutoff
    # string and use plain ``>=``. The first lightgbm manifest written
    # by the post-#317 trainer was ``20260423T084003Z``; the cutoff is
    # set to the round hour just before that so any retrain landing
    # within the same training campaign is included.
    CADENCE_PROVENANCE_CUTOFF_VERSION = "20260423T080000Z"
    # The meta-brain (``__meta__`` slot, ``model_kind == "meta"``) is
    # written by a different code path (`train_meta.py`) that does NOT
    # use the `ModelManifest` dataclass and is intentionally exempt from
    # the cadence-provenance contract — meta inputs are upstream-model
    # probabilities, not bars. Same exemption for the prior-only
    # fallback (``model_kind == "prior"``): it carries a fixed prob
    # vector with no input bars at all.
    CADENCE_PROVENANCE_EXEMPT_MODEL_KINDS = {"meta", "prior"}

    # --- Backstop A: the dataclass itself MUST emit the cadence fields. ---
    # This catches a regression where someone drops the fields from
    # `ModelManifest`, regardless of what's on disk.
    import importlib

    registry = importlib.import_module("app.training.registry")
    fresh_manifest = registry.ModelManifest(
        coin_id="__fixture__",
        timeframe="1m",
        version=registry.make_version(),
        feature_names=[],
        coin_vocab=["__fixture__"],
        n_train_rows=0,
        n_test_rows=0,
        metrics={},
        baseline_metrics={},
        threshold_pct=0.0,
        horizon_candles=1,
        class_return_means_pct=[0.0, 0.0, 0.0],
    )
    fresh_dict = fresh_manifest.to_dict()
    assert "bars_by_native_cadence" in fresh_dict and "cadence_mixed" in fresh_dict, (
        "ModelManifest.to_dict() must carry both 'bars_by_native_cadence' "
        "and 'cadence_mixed' — the trainer contract from task #317 was "
        f"silently dropped from the dataclass (got keys: {sorted(fresh_dict)!r})"
    )

    # --- Backstop B: every MODERN on-disk manifest must carry the fields. ---
    # We deliberately don't fail when zero modern manifests exist (e.g. on
    # a freshly pruned registry) — Backstop A already guards the trainer
    # contract. We DO fail loudly if a modern manifest is missing the
    # fields, which is the real regression: the trainer ran post-#317 and
    # forgot to emit cadence provenance.
    manifests = list(ML_MODELS.glob("*/*/*/manifest.json"))
    missing: list[Path] = []
    for m in manifests:
        if m.parent.name < CADENCE_PROVENANCE_CUTOFF_VERSION:
            continue
        try:
            obj = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        if obj.get("model_kind") in CADENCE_PROVENANCE_EXEMPT_MODEL_KINDS:
            continue
        prov = obj.get("provenance") if isinstance(obj.get("provenance"), dict) else obj
        if "bars_by_native_cadence" not in prov or "cadence_mixed" not in prov:
            missing.append(m)
    assert not missing, (
        f"{len(missing)} post-#317 manifest(s) are missing cadence-provenance "
        f"fields (cutoff version >= {CADENCE_PROVENANCE_CUTOFF_VERSION}, "
        f"exempt model_kinds = {sorted(CADENCE_PROVENANCE_EXEMPT_MODEL_KINDS)!r}). "
        f"Examples: {[str(p) for p in missing[:3]]!r}. "
        "The trainer must record 'bars_by_native_cadence' and 'cadence_mixed' "
        "on every manifest it writes — see app/training/registry.py."
    )

    # Verification-gate behavior: the gate must refuse a manifest that is
    # both mixed and unmitigated. We import the gate's helper if available
    # (post-fix); pre-fix it does not exist, so this assertion fails loudly.
    import importlib

    try:
        gate = importlib.import_module("app.training.verification")
    except ModuleNotFoundError:
        pytest.fail("app.training.verification module missing — cannot validate gate behavior")
    refuser = getattr(gate, "manifest_blocks_promotion_for_cadence_mix", None)
    assert callable(refuser), (
        "app.training.verification.manifest_blocks_promotion_for_cadence_mix "
        "must exist and return True when a slice manifest is cadence-mixed "
        "and unmitigated (canonical helper location alongside MIN_DIRECTIONAL_ACCURACY)"
    )
    mixed_manifest = {
        "provenance": {
            "bars_by_native_cadence": {"daily": 100, "minute": 200},
            "cadence_mixed": True,
            "cadence_mitigation": None,
        }
    }
    assert refuser(mixed_manifest) is True, (
        "the gate must refuse to promote a manifest with cadence_mixed=true "
        "and no mitigation"
    )
