"""Tests for ``scripts.diagnose_post_fee``.

Pins three contracts:
  * Per-trade row schema (column names + order)
  * All-trades aggregate (loose entry rule) matches the trainer's
    ``_holdout_pnl_after_fees`` byte-for-byte on the same inputs — this
    is the figure the manifest's ``slice.pnl_after_fees`` captures, so
    the diagnostic's "trainer-side reference" can be triangulated
    against the manifest payload.
  * Refuses to reconstruct the holdout when the dataset's row count for
    the slice no longer matches the manifest's ``n_train_rows``.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from app.training import registry as registry_module
from app.training.registry import FEATURE_COLUMNS, ModelManifest
from app.training.train import _holdout_pnl_after_fees

from scripts import diagnose_post_fee as diag


# ── Per-trade schema is part of the public contract ──────────────────────
def test_per_trade_columns_pin() -> None:
    assert diag.PER_TRADE_COLUMNS == [
        "row_index",
        "timestamp_ms",
        "timestamp_iso",
        "regime",
        "p_down",
        "p_stable",
        "p_up",
        "edge",
        "magnitude_pct",
        "expected_return_pct",
        "last_price",
        "atr14",
        "label_3class",
        "forward_return_pct",
        "action",
        "direction",
        "confidence",
        "size_multiplier",
        "position_size_usd",
        "sl_price",
        "tp_price",
        "skip_reason",
        "gross_pct",
        "net_pct",
    ]


# ── Loose aggregate matches the trainer's reference exactly ──────────────
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1337])
def test_loose_aggregate_matches_trainer(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 400
    raw = rng.dirichlet([1.0, 4.0, 1.0], size=n)
    forward_returns = rng.normal(loc=0.0, scale=0.01, size=n)
    magnitudes_pct = np.abs(rng.normal(loc=0.4, scale=0.2, size=n))

    fr = diag.get_frictions()
    mde = float(fr.min_directional_edge)
    mer = float(fr.min_expected_return_pct)
    rtc = float(fr.round_trip_cost_pct) * 100.0

    diag_out = diag.loose_post_fee_aggregate(
        probs=raw, forward_returns=forward_returns,
        fr=fr, magnitudes_pct=magnitudes_pct,
    )
    train_out = _holdout_pnl_after_fees(
        raw, forward_returns,
        min_directional_edge=mde,
        min_expected_return_pct=mer,
        round_trip_cost_pct=rtc,
        magnitudes_pct=magnitudes_pct,
    )
    # The trainer's contract is the source of truth; the diagnostic must
    # reproduce it field-for-field on the same inputs.
    assert diag_out["n_trades"] == train_out["n_trades"]
    assert diag_out["trade_share"] == train_out["trade_share"]
    assert diag_out["round_trip_cost_pct"] == train_out["round_trip_cost_pct"]
    assert diag_out["gross_pct_mean"] == train_out["gross_pct_mean"]
    assert diag_out["net_pct_mean"] == train_out["net_pct_mean"]
    assert diag_out["net_pct_total"] == train_out["net_pct_total"]
    assert diag_out["win_rate"] == train_out["win_rate"]


# ── Refusal on dataset / manifest holdout-row drift ──────────────────────
def _write_synthetic_slot(
    tmp_root: Path, *, n_dataset_rows: int, manifest_n_train_rows: int,
) -> tuple[str, str, str]:
    """Build a tiny per-coin booster slot at ``tmp_root`` with the given
    dataset row count and a manifest that claims a (potentially
    different) ``n_train_rows``. Returns (coin, timeframe, version).
    """
    coin, timeframe, version = "synthcoin", "5m", "v0"
    rng = np.random.default_rng(0)
    n = n_dataset_rows
    feature_cols = list(FEATURE_COLUMNS)
    df = pd.DataFrame(
        {c: rng.normal(size=n).astype(float) for c in feature_cols},
    )
    # Required non-feature columns the diagnostic reads downstream.
    df["coin_id"] = coin
    df["timeframe"] = timeframe
    df["timestamp_ms"] = (
        np.arange(n, dtype=np.int64) * 5 * 60 * 1000
        + 1_700_000_000_000
    )
    df["lastPrice"] = 1.0 + rng.normal(scale=0.001, size=n)
    df["forward_return"] = rng.normal(scale=0.005, size=n)
    df["label_3class"] = rng.integers(0, 3, size=n)
    df["regime"] = "neutral"
    df["coin_idx"] = 0
    # Persist dataset to the tmp registry's datasets/ dir.
    datasets_dir = tmp_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    ds_path = datasets_dir / f"{timeframe}_synthetic.parquet"
    df.to_parquet(ds_path, index=False)

    # Train a tiny booster on the same frame so load_model can recover it.
    X = df[feature_cols]
    y = df["label_3class"].to_numpy(dtype=int)
    train_set = lgb.Dataset(X, label=y, categorical_feature=["coin_idx"])
    booster = lgb.train(
        {
            "objective": "multiclass",
            "num_class": 3,
            "verbose": -1,
            "num_leaves": 7,
            "learning_rate": 0.1,
        },
        train_set,
        num_boost_round=5,
    )

    out_dir = tmp_root / coin / timeframe / version
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"))
    # No calibrators: load_model handles a missing calibrators.joblib.
    manifest = ModelManifest(
        coin_id=coin,
        timeframe=timeframe,
        version=version,
        feature_names=feature_cols,
        coin_vocab=[coin],
        n_train_rows=manifest_n_train_rows,
        n_test_rows=manifest_n_train_rows // 5,
        metrics={"auc": 0.5, "log_loss": 1.0, "brier": 0.2,
                 "directional_accuracy": 0.5},
        baseline_metrics={"auc": 0.5, "directional_accuracy": 0.5},
        threshold_pct=0.22,
        horizon_candles=1,
        class_return_means_pct=[-0.4, 0.0, 0.4],
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2),
    )
    (tmp_root / coin / timeframe / "latest").write_text(version)
    return coin, timeframe, version


def test_refuses_on_holdout_row_count_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    coin, timeframe, version = _write_synthetic_slot(
        tmp_path,
        n_dataset_rows=80,
        manifest_n_train_rows=200,  # deliberate mismatch
    )
    # Find dataset would fail to find a matching one (since 80 != 200);
    # we feed the explicit path to force the row-count check.
    ds_path = tmp_path / "datasets" / f"{timeframe}_synthetic.parquet"
    with pytest.raises(diag.HoldoutDriftError):
        diag.run_diagnostic(
            coin=coin, timeframe=timeframe, version=version,
            dataset_path=ds_path,
            out_root=tmp_path / "diagnostics",
        )


def test_runs_when_row_count_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    n = 200
    coin, timeframe, version = _write_synthetic_slot(
        tmp_path, n_dataset_rows=n, manifest_n_train_rows=n,
    )
    summary = diag.run_diagnostic(
        coin=coin, timeframe=timeframe, version=version,
        out_root=tmp_path / "diagnostics",
    )
    expected_holdout = n - max(1, int(n * 0.8))
    assert summary["holdout"]["n_rows"] == expected_holdout
    out_dir = Path(summary["output_dir"])
    assert (out_dir / "REPORT.md").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "per_trade.json").exists()
    assert (out_dir / "holdout_scored.csv").exists()
    # Per-trade schema sanity-check on the persisted CSV.
    scored_csv = pd.read_csv(out_dir / "holdout_scored.csv")
    assert list(scored_csv.columns) == diag.PER_TRADE_COLUMNS


def test_dataset_auto_discovery_picks_matching_row_count(
    tmp_path: Path,
) -> None:
    """``find_dataset_for_manifest`` walks ``datasets/<tf>_*.parquet``
    newest-first and returns the first parquet whose coin slice has the
    expected row count. This pins that row-count check, not mtime, is
    the selection criterion.
    """
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True)
    coin, timeframe = "synthcoin", "5m"

    def _make(path: Path, n: int) -> None:
        pd.DataFrame({
            "coin_id": [coin] * n,
            "timestamp_ms": np.arange(n, dtype=np.int64),
        }).to_parquet(path, index=False)

    decoy = datasets_dir / f"{timeframe}_decoy.parquet"
    real = datasets_dir / f"{timeframe}_real.parquet"
    _make(decoy, 17)
    _make(real, 42)
    # Touch decoy AFTER real so it is the "newest"; the resolver must
    # still pick `real` because `decoy`'s row count doesn't match.
    import os
    later = real.stat().st_mtime + 1
    os.utime(decoy, (later, later))

    picked = diag.find_dataset_for_manifest(
        timeframe=timeframe, coin=coin, n_expected=42,
        datasets_dir=datasets_dir,
    )
    assert picked == real

    with pytest.raises(diag.DatasetNotFoundError):
        diag.find_dataset_for_manifest(
            timeframe=timeframe, coin=coin, n_expected=999,
            datasets_dir=datasets_dir,
        )
