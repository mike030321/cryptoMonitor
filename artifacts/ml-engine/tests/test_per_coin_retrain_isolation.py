"""Task #343 — per-coin retrain isolation.

Pins the contract that calling `train_one_slice(coin, tf, ...)` only
mutates the registry directory for that exact (coin, tf) slice. Any
future change that:
  * accidentally rewrites the pooled fallback,
  * touches a *different* coin's per-coin model,
  * touches the same coin's *other* timeframes,
  * regresses a specialist / meta artifact,
must trip this test.

The test snapshots `(path, mtime_ns, sha256)` for every file under a
freshly-isolated `REGISTRY_ROOT` *before* the per-coin retrain, then
diffs the same snapshot afterwards. The only legal delta is files
under `<coin_id>/<timeframe>/...`.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.training import registry as registry_module
from app.training.labels import build_labeled_frame_for_coin
from app.training.train import train_one_slice


def _signal_ticks(n: int, start: datetime, drift: float, seed: int = 17,
                  step_s: int = 60):
    """Tick series with enough realised vol that the labeler produces
    rows in all three classes — required so per-class calibration can
    fit and the slice ends up `status=trained`."""
    rng = np.random.default_rng(seed)
    out = []
    p = 100.0
    for i in range(n):
        p *= (1.0 + drift + rng.normal(0, 0.005))
        out.append((start + timedelta(seconds=i * step_s), p))
    return out


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    snap: dict[str, tuple[int, str]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        st = p.stat()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        snap[rel] = (st.st_mtime_ns, h)
    return snap


def _seed_fake_slice(root: Path, coin: str, tf: str) -> None:
    """Drop a fake per-(coin, tf) artifact tree under the registry root
    so the snapshot diff has untouched siblings to assert against. We
    deliberately do NOT call `train_one_slice` here — a real LightGBM
    walk-forward fit takes ~50s per slice and would blow the 120s
    pytest cap. The isolation contract is about which file PATHS get
    written, not whether the bytes inside the seed are a valid booster.
    A future regression that erroneously rewrites e.g. `bonk/1m/...`
    will still flip the sha256 of these placeholder files and trip the
    test exactly the same as it would on real boosters."""
    version = "20250101T000000Z-seed"
    slice_dir = root / coin / tf / version
    slice_dir.mkdir(parents=True, exist_ok=True)
    (slice_dir / "model.txt").write_bytes(
        f"fake-booster-{coin}-{tf}-do-not-touch".encode()
    )
    (slice_dir / "manifest.json").write_text(
        f'{{"coin_id":"{coin}","timeframe":"{tf}","fake":true}}'
    )
    (slice_dir / "calibrators.joblib").write_bytes(
        b"\x00\x01" + coin.encode() + b"\x00" + tf.encode()
    )
    (root / coin / tf / "latest").write_text(version)


def test_per_coin_retrain_does_not_perturb_other_slices(monkeypatch, tmp_path):
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    vocab = ["pepe", "bonk"]
    tf = "1m"

    # ── 1. Seed the registry with placeholder artifacts for the slices
    # we expect to be UNTOUCHED by the per-coin retrain of (pepe, 1m):
    #   * bonk/1m         — different coin, same tf
    #   * pepe/5m         — same coin, different tf
    #   * __pooled__/1m   — pooled fallback at same tf (the regression
    #                       the task most cares about — a per-coin
    #                       retrain that silently rewrites pooled would
    #                       invalidate every other coin's fallback)
    #   * __specialist_volatility_forecaster__/1m — Phase-3 specialist
    #   * meta/1m         — Phase-4 meta-model directory
    _seed_fake_slice(tmp_path, "bonk", "1m")
    _seed_fake_slice(tmp_path, "pepe", "5m")
    _seed_fake_slice(tmp_path, "__pooled__", "1m")
    _seed_fake_slice(tmp_path, "__specialist_volatility_forecaster__", "1m")
    _seed_fake_slice(tmp_path, "meta", "1m")

    # ── 1b. Seed a multi-slice `report.json` so we can also assert
    # report-record isolation: a per-coin retrain of (pepe, 1m) must
    # not rewrite, drop, or mutate the report records belonging to the
    # other slices. `train.py` writes this file at the end of every
    # full-fat retrain pass; per-coin retrains via `train_one_slice`
    # must leave it untouched, since the per-slice metadata already
    # lives in each slice's own `manifest.json`.
    seeded_report = {
        "generated_at": "2025-01-01T00:00:00Z",
        "slices": {
            "bonk/1m": {"version": "20250101T000000Z-seed", "auc": 0.61},
            "pepe/5m": {"version": "20250101T000000Z-seed", "auc": 0.58},
            "__pooled__/1m": {"version": "20250101T000000Z-seed", "auc": 0.55},
            "pepe/1m": {"version": "20250101T000000Z-seed", "auc": 0.59},
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(seeded_report, indent=2))

    # ── 2. Snapshot every file under the registry BEFORE the per-coin
    # retrain.
    before = _snapshot(tmp_path)
    assert before, "registry should not be empty after the seed step"

    # ── 3. Per-coin retrain for (pepe, 1m). The slice's own files are
    # *expected* to change; everything else must not.
    pepe_df = build_labeled_frame_for_coin(
        "pepe", tf, _signal_ticks(180, base, 0.0008, seed=99),
    )
    res = train_one_slice(
        pepe_df, coin_id="pepe", timeframe=tf, vocab=vocab,
        note="per-coin retrain isolation test",
    )
    assert res["status"] == "trained", res

    after = _snapshot(tmp_path)

    # ── 4. Diff the snapshots and partition into legal vs illegal.
    legal_prefix_pepe = f"pepe/{tf}/"
    legal_prefix_dataset = "datasets/"  # train_one_slice writes a parquet snapshot

    def _is_legal(rel: str) -> bool:
        return rel.startswith(legal_prefix_pepe) or rel.startswith(legal_prefix_dataset)

    illegal_modified: list[str] = []
    illegal_added: list[str] = []
    illegal_removed: list[str] = []

    for rel, (mtime, sha) in after.items():
        prev = before.get(rel)
        if prev is None:
            if not _is_legal(rel):
                illegal_added.append(rel)
            continue
        if prev[1] != sha:
            if not _is_legal(rel):
                illegal_modified.append(rel)
    for rel in before:
        if rel not in after and not _is_legal(rel):
            illegal_removed.append(rel)

    assert not illegal_modified, (
        f"per-coin retrain of (pepe, {tf}) modified files outside its own "
        f"slice — task #343 forbids this. Touched: {illegal_modified[:10]}"
    )
    assert not illegal_added, (
        f"per-coin retrain of (pepe, {tf}) created files outside its own "
        f"slice — task #343 forbids this. Added: {illegal_added[:10]}"
    )
    assert not illegal_removed, (
        f"per-coin retrain of (pepe, {tf}) removed files outside its own "
        f"slice — task #343 forbids this. Removed: {illegal_removed[:10]}"
    )

    # ── 5. And positively assert that the target slice DID change, so a
    # future change that turns train_one_slice into a no-op cannot pass
    # the isolation half of the test by accident.
    pepe_files = [k for k in after if k.startswith(legal_prefix_pepe)]
    assert pepe_files, f"per-coin retrain wrote nothing to pepe/{tf}/"
    changed_in_slice = [
        k for k in pepe_files
        if k not in before or before[k][1] != after[k][1]
    ]
    assert changed_in_slice, (
        f"per-coin retrain produced no detectable change inside pepe/{tf}/ — "
        "the test would pass vacuously; check that train_one_slice still "
        "writes the model + manifest + latest pointer."
    )

    # ── 6. Explicit report-record isolation check. The seeded report
    # must come back with every non-target slice record byte-identical;
    # the per-coin retrain has no mandate to touch report.json at all,
    # but if a future change starts updating the (pepe, 1m) record in
    # place, the OTHER records must still be untouched. Asserting on
    # parsed JSON gives a much clearer failure message than the raw
    # sha256 diff above when this regresses.
    after_report = json.loads((tmp_path / "report.json").read_text())
    assert "slices" in after_report, "report.json lost its `slices` key"
    for sk in ("bonk/1m", "pepe/5m", "__pooled__/1m"):
        assert after_report["slices"].get(sk) == seeded_report["slices"][sk], (
            f"per-coin retrain of (pepe, {tf}) mutated the report.json "
            f"record for {sk!r} — task #343 forbids this."
        )
