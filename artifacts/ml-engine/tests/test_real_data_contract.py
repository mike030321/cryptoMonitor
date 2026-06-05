"""Task #267 — enforcement tests for the quant training contract.

Covers:
  (a) provenance guard rejects a synthetic row and marks the report;
  (b) leakage audit catches a planted future-leak;
  (c) walk-forward CI guard fails when a banned random splitter is imported;
  (d) new feature columns survive `_prepare_xy` and end up in the labeled frame;
  (e) `realized_vol_next_horizon` and `net_pnl_after_costs_pct` are present,
      non-trivially populated and bounded;
  (f) fast loop runs without triggering a slow loop and vice versa.

The full contract document lives at `artifacts/ml-engine/TRAINING_CONTRACT.md`.
Updates to the contract MUST update both the document and the matching
assertion below.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.training import labels as labels_module
from app.training.labels import (
    EXTERNAL_STREAM_DEFAULTS,
    FORWARD_HORIZON_CANDLES,
    audit_leakage,
    build_labeled_frame_for_coin,
)
from app.training.registry import (
    CONTRACT_NEW_FEATURE_COLUMNS,
    EXTERNAL_STREAM_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_LINEAGE,
    FORWARD_TARGET_COLUMNS,
    SESSION_FEATURE_COLUMNS,
)
from app.training.train import (
    FAST_LOOP_MIN_NEW_ROWS,
    SLOW_LOOP_NEW_TICKS_THRESHOLD,
    _detect_unwired_external_streams,
    _prepare_xy,
    _provenance_summary,
    _summarize_feature_coverage,
    _summarize_feature_density,
    _summarize_target_row_counts,
    should_run_fast_loop,
    should_run_slow_loop,
)


def _trending_ticks(n: int, start: datetime, step_s: int = 60) -> list[tuple[datetime, float]]:
    return [
        (start + timedelta(seconds=i * step_s), 100.0 + i * 0.5 + math.sin(i / 3.0) * 2.0)
        for i in range(n)
    ]


# --- (a) Provenance guard ----------------------------------------------------
class _FakePool:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    async def fetch(self, _query: str, *args):
        return list(self._rows)


@pytest.mark.asyncio
async def test_provenance_guard_rejects_synthetic_rows(monkeypatch):
    """A lookback window that contains ANY synthetic row must be reported
    with `rejected_synthetic=true`. The trainer surfaces the rejection in
    `report.json::timeframes.<tf>.provenance` and excludes the slice from
    the labeled frame even though the SQL filter already drops the rows."""
    from app import db as db_mod

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"timestamp": base + timedelta(seconds=i * 60), "price": 100.0 + i,
         "is_synthetic": (i == 5)}
        for i in range(40)
    ]
    fake_pool = _FakePool(rows)

    async def _fake_init_pool():
        return fake_pool

    monkeypatch.setattr(db_mod, "init_pool", _fake_init_pool)
    res = await db_mod.fetch_real_ticks_with_provenance("pepe", lookback_ms=24 * 3600 * 1000)
    assert res["rows_real"] == 39
    assert res["rows_synthetic"] == 1
    assert res["rejected_synthetic"] is True
    assert all(t for t in res["ticks"])  # only real rows came back

    summary = _provenance_summary({
        "pepe": {"rows_real": 39, "rows_synthetic": 1, "rejected_synthetic": True},
        "bonk": {"rows_real": 100, "rows_synthetic": 0, "rejected_synthetic": False},
    })
    assert summary["rejected_synthetic"] is True
    assert summary["coins_rejected"] == ["pepe"]
    assert summary["rows_synthetic"] == 1


@pytest.mark.asyncio
async def test_provenance_guard_passes_when_all_real(monkeypatch):
    """A clean window leaves `rejected_synthetic=False` so the trainer
    proceeds normally — the guard only fires on actual contamination."""
    from app import db as db_mod

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"timestamp": base + timedelta(seconds=i * 60), "price": 100.0 + i,
         "is_synthetic": False}
        for i in range(20)
    ]

    async def _fake_init_pool():
        return _FakePool(rows)

    monkeypatch.setattr(db_mod, "init_pool", _fake_init_pool)
    res = await db_mod.fetch_real_ticks_with_provenance("pepe", lookback_ms=24 * 3600 * 1000)
    assert res["rejected_synthetic"] is False
    assert res["rows_synthetic"] == 0
    assert res["rows_real"] == 20


# --- (b) Leakage audit -------------------------------------------------------
def test_leakage_audit_catches_target_in_features():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    # Plant a future-leak: declare `forward_window_return_pct` as a feature.
    res = audit_leakage(
        df,
        feature_columns=["ret1", "forward_window_return_pct"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES,
    )
    assert res["passed"] is False
    assert any("forward_window_return_pct" in v for v in res["violations"])


def test_leakage_audit_catches_wrong_horizon():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    res = audit_leakage(
        df,
        feature_columns=["ret1"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES + 1,
    )
    assert res["passed"] is False
    assert any("forward_horizon_candles" in v for v in res["violations"])


def test_leakage_audit_passes_clean_frame():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    res = audit_leakage(
        df,
        feature_columns=[c for c in FEATURE_COLUMNS if c != "coin_idx"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES,
    )
    assert res["passed"] is True, res["violations"]
    assert res["lineage_unregistered"] == []
    assert res["future_corr_hits"] == []


def test_leakage_audit_lineage_gate_rejects_unregistered_feature():
    """Layer 2 — declaring an unknown column as a feature MUST fail the
    audit even when the column is benign and not target-named. This
    prevents a developer from silently adding a new feature without
    declaring its provenance in `FEATURE_LINEAGE`."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    df = df.copy()
    df["mystery_signal"] = 0.0
    res = audit_leakage(
        df,
        feature_columns=["ret1", "mystery_signal"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES,
    )
    assert res["passed"] is False
    assert "mystery_signal" in res["lineage_unregistered"]
    assert any("mystery_signal" in v and "lineage" in v for v in res["violations"])


def test_leakage_audit_lineage_gate_rejects_future_lookforward():
    """Layer 2 — even a registered column fails when the lineage entry
    declares `max_lookforward > 0`."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    bad_lineage = dict(FEATURE_LINEAGE)
    bad_lineage["ret1"] = {"max_lookforward": 1, "max_lookback": 1}
    res = audit_leakage(
        df,
        feature_columns=["ret1"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES,
        feature_lineage=bad_lineage,
    )
    assert res["passed"] is False
    assert any("max_lookforward=1" in v for v in res["violations"])


def test_leakage_audit_numerical_rejects_planted_future_leak():
    """Layer 3 — the killer case: a column that is NOT named like a
    target and is registered in lineage with `max_lookforward=0` but
    whose values are a near-copy of a forward target (a renamed
    `forward_return`). Layer 1 (schema) won't catch it because the
    column name doesn't match a target; Layer 2 (lineage gate) won't
    catch it because the developer registered it. Layer 3 catches it
    via correlation against the actual target columns.
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(160, base))
    assert not df.empty
    df = df.copy()
    # Plant the leak: a renamed forward_return column with tiny noise so
    # correlation is ~1.0 but not exactly 1.0 (real-world refactor case).
    rng = [math.sin(i * 7.13) * 1e-9 for i in range(len(df))]
    df["secret_signal"] = df["forward_return"].to_numpy() + rng
    # Register it so the lineage gate doesn't catch it first.
    lineage = dict(FEATURE_LINEAGE)
    lineage["secret_signal"] = {"max_lookforward": 0, "max_lookback": None}
    res = audit_leakage(
        df,
        feature_columns=["ret1", "secret_signal"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES,
        feature_lineage=lineage,
    )
    assert res["passed"] is False, res
    leaks = {h["feature"] for h in res["future_corr_hits"]}
    assert "secret_signal" in leaks
    assert any("secret_signal" in v and "future-leak" in v
               for v in res["violations"])


def test_leakage_audit_accepts_auto_registered_approved_features():
    """Regression — approved features (from Feature Lab) are added to
    `active_feature_columns` AFTER the labeled frame is built, so they
    are not in the static `FEATURE_LINEAGE` table. `run_training` must
    auto-register them as safe (max_lookforward=0) when calling the
    audit, otherwise every approved feature would deterministically
    fail the lineage gate and halt the retrain. This mirrors the
    `audit_lineage = {**FEATURE_LINEAGE, **approved}` merge that
    `run_training` performs."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(160, base))
    assert not df.empty
    df = df.copy()
    # Simulate an approved feature derived from base columns. Feature Lab
    # only emits algebraic transforms of registered base features, so
    # the result is point-in-time by construction.
    df["approved_ret1_x_rsi14"] = df["ret1"] * df["rsi14"]
    added = ["approved_ret1_x_rsi14"]
    audit_lineage = {
        **FEATURE_LINEAGE,
        **{n: {"max_lookforward": 0, "max_lookback": None,
               "auto_registered_via": "apply_approved_features"}
           for n in added},
    }
    res = audit_leakage(
        df,
        feature_columns=["ret1", "rsi14", "approved_ret1_x_rsi14"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES,
        feature_lineage=audit_lineage,
    )
    assert res["passed"] is True, res["violations"]
    # And without the merge, the same audit MUST fail — proving the
    # lineage gate still bites unregistered columns.
    res_strict = audit_leakage(
        df,
        feature_columns=["ret1", "rsi14", "approved_ret1_x_rsi14"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES,
        feature_lineage=FEATURE_LINEAGE,
    )
    assert res_strict["passed"] is False
    assert "approved_ret1_x_rsi14" in res_strict["lineage_unregistered"]


def test_external_stream_defaults_emit_nan_when_no_asof_match():
    """Task #633 — when a coin has zero `market_signals` rows (or no
    snapshot at-or-before the candle bucket), every external-stream
    feature column emits `NaN`, NOT `0.0`. LightGBM handles NaN
    natively via `use_missing=true`; 0-fill silently teaches the model
    "funding was exactly zero on this bar" for the ~75 % of historical
    6h rows where funding data didn't yet exist (OKX caps history at
    ~92 days). This is the contamination root cause documented in the
    MTTM 6h verdict report.

    The asof-join site is `build_labeled_frame_for_coin`. We pass an
    EMPTY `market_signals` sequence so every column falls through to
    the registered safe default.
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin(
        "pepe", "1m", _trending_ticks(60, base),
        market_signals=[],
    )
    assert not df.empty, "fixture must produce at least one labeled row"
    for col in EXTERNAL_STREAM_FEATURE_COLUMNS:
        assert col in df.columns, f"contract column missing: {col}"
        # No row may carry the legacy 0.0 default — NaN is the only
        # acceptable fingerprint when no provider data is available.
        zero_rows = (df[col] == 0.0).sum()
        assert zero_rows == 0, (
            f"{col} has {zero_rows} rows with the legacy 0.0 default "
            f"(post-#633 the default must be NaN)"
        )
        assert df[col].isna().all(), (
            f"{col} should be entirely NaN when market_signals is empty, "
            f"got non-NaN values: {df[col].dropna().head().to_list()}"
        )

    # And every entry in EXTERNAL_STREAM_DEFAULTS itself must be NaN.
    for col, default in EXTERNAL_STREAM_DEFAULTS.items():
        assert math.isnan(default), (
            f"EXTERNAL_STREAM_DEFAULTS[{col!r}] must be NaN, got {default!r}"
        )


def test_feature_density_distinguishes_unwired_from_missing():
    """Density is `% non-null AND non-zero`. After task #633 the
    unwired-default for external streams is `NaN` (not `0.0`), so an
    unwired stream column reports `coverage = 0.0` (no row has a
    non-null value) AND `density = 0.0` (no row has a non-null AND
    non-zero value). The operator signal called out in rule 7 stays
    intact: any external-stream column whose coverage is below 1.0 is
    receiving partial-or-no real provider data, and density still
    reports the share of rows carrying real non-zero signal.
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    df["coin_idx"] = 0
    coverage = _summarize_feature_coverage(df, FEATURE_COLUMNS)
    density = _summarize_feature_density(df, FEATURE_COLUMNS)
    for col in EXTERNAL_STREAM_FEATURE_COLUMNS:
        # No `market_signals` rows were passed to the builder, so the
        # asof-join falls through to the registered safe default for
        # every row. After task #633 that default is NaN: coverage 0.0
        # and density 0.0.
        assert coverage[col] == 0.0, (col, coverage[col])
        assert density[col] == 0.0, (col, density[col])


def test_leakage_audit_catches_unsorted_frame():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    shuffled = df.iloc[::-1].reset_index(drop=True)
    res = audit_leakage(
        shuffled,
        feature_columns=["ret1"],
        target_columns=FORWARD_TARGET_COLUMNS,
        expected_horizon=FORWARD_HORIZON_CANDLES,
    )
    assert res["passed"] is False
    assert any("monotonic" in v for v in res["violations"])


# --- (c) Walk-forward CI guard ---------------------------------------------
_BANNED_SPLITTERS = re.compile(
    r"\b(train_test_split|KFold|StratifiedKFold|ShuffleSplit)\b"
    r"|shuffle\s*=\s*True"
)
# Allow this enforcement test itself to mention the banned names.
_GUARD_ALLOWLIST = {"test_real_data_contract.py"}


def _scan_dir_for_banned_splitters(root: Path) -> list[tuple[Path, str]]:
    hits: list[tuple[Path, str]] = []
    for path in root.rglob("*.py"):
        if path.name in _GUARD_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in _BANNED_SPLITTERS.finditer(text):
            hits.append((path, m.group(0)))
    return hits


def test_no_random_split_imports_in_training_modules():
    """The training contract requires walk-forward as the only validator.
    Any module that imports `train_test_split`, `KFold`, `StratifiedKFold`,
    `ShuffleSplit`, or sets `shuffle=True` would silently break the
    chronological invariant — this test fails CI before the code ships.
    """
    repo_root = Path(__file__).resolve().parents[1]
    hits = []
    for sub in ("app/training", "app/backtest"):
        hits.extend(_scan_dir_for_banned_splitters(repo_root / sub))
    assert not hits, (
        "Banned random splitters found in training/backtest modules — "
        "walk-forward is the only allowed validator. Hits:\n"
        + "\n".join(f"  {p}: {tok}" for p, tok in hits)
    )


# --- (d) New feature columns survive _prepare_xy ----------------------------
def test_contract_feature_columns_registered():
    """Every new contract feature column must be in FEATURE_COLUMNS so
    `_prepare_xy` slices them out of the labeled frame at training time."""
    for col in CONTRACT_NEW_FEATURE_COLUMNS:
        assert col in FEATURE_COLUMNS, f"missing contract feature column: {col}"
    for col in EXTERNAL_STREAM_FEATURE_COLUMNS:
        assert col in EXTERNAL_STREAM_DEFAULTS, (
            f"external stream column {col} has no null-safe default"
        )


def test_prepare_xy_includes_contract_columns():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    # _encode_coin_idx + _prepare_xy together; we mimic the trainer.
    df = df.copy()
    df["coin_idx"] = 0
    X, _ = _prepare_xy(df)
    for col in CONTRACT_NEW_FEATURE_COLUMNS:
        assert col in X.columns, f"_prepare_xy dropped contract column: {col}"
    # Session columns must always be populated; one of the three exclusive
    # one-hots is hot on every row (sum == 1 across the three).
    one_hot_sum = X[["session_asia", "session_eu", "session_us"]].sum(axis=1)
    assert (one_hot_sum == 1.0).all(), "session one-hots are not exclusive"
    # Hour-of-day sin/cos must lie on the unit circle.
    radii = (X["hour_of_day_sin"] ** 2 + X["hour_of_day_cos"] ** 2)
    assert (radii.between(0.99, 1.01)).all()
    # Task #633 — external-stream columns default to NaN (not 0.0) when
    # no provider is wired up. LightGBM handles NaN natively via
    # `use_missing=true`, so the booster sees "this row had no funding
    # data" instead of the spurious "funding was exactly zero" signal.
    for col in EXTERNAL_STREAM_FEATURE_COLUMNS:
        assert X[col].isna().all(), (
            f"{col} should default to NaN when unwired, got "
            f"{X[col].dropna().head().to_list()}"
        )


def test_feature_coverage_summary_uses_non_null_definition():
    """Per rule 7 of the contract, `feature_coverage` is the NON-NULL
    share. After task #633 the unwired-default for external streams is
    NaN (not 0.0), so an unwired stream column reports `coverage = 0.0`
    — every row IS the (NaN) default, so no row is non-null. The
    `feature_density` view (`% non-null AND non-zero`) is also 0.0
    because there is no real provider data; both summaries agree.
    Always-populated columns like the session one-hots remain at 1.0.
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    df["coin_idx"] = 0
    coverage = _summarize_feature_coverage(df, FEATURE_COLUMNS)
    for col in EXTERNAL_STREAM_FEATURE_COLUMNS:
        assert coverage[col] == 0.0, (
            f"unwired stream {col} should report 0.0 coverage post-#633 "
            f"(NaN default), got {coverage[col]}"
        )
    # Three exclusive session one-hots: each is non-null on every row
    # because they're emitted from `_session_features_for_bucket`, so
    # coverage is 1.0 for each.
    for col in ("session_asia", "session_eu", "session_us"):
        assert coverage[col] == 1.0


# --- (e) realized_vol_next_horizon + net_pnl_after_costs_pct ----------------
def test_next_horizon_targets_present_and_bounded():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # Use a noisier series so realized_vol_next_horizon is non-trivial.
    rng_ticks = []
    p = 100.0
    for i in range(200):
        # deterministic oscillation so the test is reproducible
        p *= 1.0 + 0.002 * math.sin(i / 4.0)
        rng_ticks.append((base + timedelta(seconds=i * 60), p))
    df = build_labeled_frame_for_coin("pepe", "1m", rng_ticks)
    assert not df.empty
    assert "realized_vol_next_horizon" in df.columns
    assert "net_pnl_after_costs_pct" in df.columns

    rv = df["realized_vol_next_horizon"].dropna()
    assert len(rv) > 0, "realized_vol_next_horizon never populated"
    # Standard deviation in PERCENT — must be non-negative and far under
    # 100% on any sane series.
    assert (rv >= 0).all()
    assert (rv < 100.0).all()
    assert rv.max() > 0.0  # non-trivially populated

    npnl = df["net_pnl_after_costs_pct"].dropna()
    assert len(npnl) > 0
    # Net PnL must always be smaller magnitude than the gross window
    # return (cost subtracts magnitude from a directional move and
    # zero-floors a sub-cost move).
    fwr = df["forward_window_return_pct"].dropna()
    aligned = pd.concat([npnl, fwr], axis=1, join="inner").dropna()
    assert (aligned["net_pnl_after_costs_pct"].abs()
            <= aligned["forward_window_return_pct"].abs() + 1e-9).all()


def test_target_row_counts_summary_includes_new_targets():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _trending_ticks(120, base))
    assert not df.empty
    counts = _summarize_target_row_counts(df)
    for tgt in (
        "realized_vol_next_horizon", "net_pnl_after_costs_pct",
        "forward_window_return_pct", "label_3class",
    ):
        assert tgt in counts, f"target {tgt} missing from target_row_counts"
    assert counts["label_3class"] == len(df)
    assert counts["net_pnl_after_costs_pct"] > 0


# --- (f) Slow / fast loop trigger semantics ----------------------------------
def test_slow_loop_triggers_on_drift_and_cadence():
    fired, reason = should_run_slow_loop(
        seconds_since_last_slow=10.0, cadence_seconds=1800,
        drift_detected=True,
    )
    assert fired is True and reason == "drift_detected"

    fired, reason = should_run_slow_loop(
        seconds_since_last_slow=10.0, cadence_seconds=1800,
        regime_shift_detected=True,
    )
    assert fired is True and reason == "regime_shift"

    fired, reason = should_run_slow_loop(
        seconds_since_last_slow=10.0, cadence_seconds=1800,
        new_ticks_since_last=SLOW_LOOP_NEW_TICKS_THRESHOLD + 1,
    )
    assert fired is True and reason == "new_data_threshold"

    fired, reason = should_run_slow_loop(
        seconds_since_last_slow=2000.0, cadence_seconds=1800,
    )
    assert fired is True and reason == "cadence"

    fired, reason = should_run_slow_loop(
        seconds_since_last_slow=10.0, cadence_seconds=1800,
    )
    assert fired is False and reason == "cooldown"

    fired, reason = should_run_slow_loop(seconds_since_last_slow=None)
    assert fired is True and reason == "cold_start"


def test_fast_loop_only_triggers_on_threshold():
    fired, _ = should_run_fast_loop(
        new_meta_rows_since_last=FAST_LOOP_MIN_NEW_ROWS - 1,
    )
    assert fired is False
    fired, reason = should_run_fast_loop(
        new_meta_rows_since_last=FAST_LOOP_MIN_NEW_ROWS,
    )
    assert fired is True
    assert "new_meta_rows" in reason


def test_fast_and_slow_loops_are_independent_decisions():
    """Triggering the fast loop must not implicitly request a slow loop,
    and a slow loop in cooldown must not block a fast loop tick."""
    slow_fired, _ = should_run_slow_loop(
        seconds_since_last_slow=60.0, cadence_seconds=1800,
    )
    fast_fired, _ = should_run_fast_loop(
        new_meta_rows_since_last=FAST_LOOP_MIN_NEW_ROWS,
    )
    assert slow_fired is False
    assert fast_fired is True
    # And vice versa.
    slow_fired, _ = should_run_slow_loop(
        seconds_since_last_slow=2000.0, cadence_seconds=1800,
    )
    fast_fired, _ = should_run_fast_loop(new_meta_rows_since_last=0)
    assert slow_fired is True
    assert fast_fired is False


# --- (g) Existing live/backtest parity fixtures still pass unchanged --------
# The decision_engine parity fixture runs as its own workflow
# (`decision-engine-parity`) and as part of the broader pytest suite via
# `tests/test_journal_client_parity.py`. Re-importing it here would
# duplicate the assertion; instead we sanity-check that the fixture file
# is still where the parity workflow expects it.
# --- (h) External-stream pickup ---------------------------------------------
def test_detect_unwired_external_streams_flags_zero_density_columns():
    """Pure helper sanity-check: a column that is fully covered (every
    row has a value) but carries zero non-zero values is the fingerprint
    of an unwired external-stream provider, and the helper MUST flag it.
    A column with even a single non-zero row MUST NOT be flagged — the
    helper uses raw counts (not the rounded `feature_density` summary)
    so a sparsely-populated provider is still classified as wired."""
    n = 1000
    df = pd.DataFrame({c: [0.0] * n for c in EXTERNAL_STREAM_FEATURE_COLUMNS})
    # One non-zero row in 1000 — would round to density 0.0 but is wired.
    df.loc[0, "funding_rate"] = 0.0001
    flagged = _detect_unwired_external_streams(df)
    assert "funding_rate" not in flagged
    for c in EXTERNAL_STREAM_FEATURE_COLUMNS:
        if c == "funding_rate":
            continue
        assert c in flagged, f"{c} should be flagged as unwired"
    # An empty frame yields an empty list (no run, no signal).
    assert _detect_unwired_external_streams(pd.DataFrame()) == []


@pytest.mark.asyncio
async def test_build_labeled_dataset_picks_up_seeded_market_signals(monkeypatch):
    """Task #287 — end-to-end check that real `market_signals` rows land
    in the trainer's labeled frame as non-zero values for every one of
    the six contract external-stream feature columns. We seed the four
    direct market-signal columns plus a BTC and ETH lead-price series,
    monkey-patch the four DB readers `build_labeled_dataset` consumes,
    and assert that `_summarize_feature_density` reports > 0 for every
    registered column.
    """
    from app import db as db_mod
    from app.training import labels as labels_module
    from app.training.train import _summarize_feature_density

    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    n_ticks = 240  # 4h of 1m bars — plenty for MIN_CANDLES_FOR_FEATURES
    ticks = [
        (base + timedelta(seconds=i * 60), 100.0 + 0.3 * i + math.sin(i / 5.0) * 1.5)
        for i in range(n_ticks)
    ]

    async def _fake_provenance(coin_id, lookback_ms):
        return {
            "ticks": list(ticks),
            "rows_real": len(ticks),
            "rows_synthetic": 0,
            "rejected_synthetic": False,
        }

    async def _fake_news_tags(coin_id, limit=20):
        return []

    # Seed 60 minutes of market_signals snapshots — non-zero on every
    # column the trainer reads. Open-interest values walk so the asof
    # z-score has a real spread to normalize against.
    signals: list[dict] = []
    for i in range(60):
        signals.append({
            "timestamp_ms": int((base + timedelta(seconds=i * 60)).timestamp() * 1000),
            "funding_rate": 0.0001 * (1 + (i % 5)),
            "open_interest_usd": 1_000_000.0 + i * 25_000.0,
            "liquidations_1h_usd": 50_000.0 + (i % 7) * 1_000.0,
            "bid_ask_spread_bps": 4.5 + (i % 3) * 0.25,
        })

    async def _fake_market_signals(coin_id, lookback_ms):
        return list(signals)

    # BTC / ETH lead price series — needs > 5m of history per sample so
    # the 5m lead-return lookup actually emits values.
    btc_series = [
        (int((base + timedelta(seconds=i * 60)).timestamp() * 1000),
         60_000.0 + i * 25.0)
        for i in range(60)
    ]
    eth_series = [
        (int((base + timedelta(seconds=i * 60)).timestamp() * 1000),
         3_000.0 + i * 1.5)
        for i in range(60)
    ]

    async def _fake_lead_series(reference_coin_id, lookback_ms):
        return list(btc_series) if reference_coin_id == "btc" else list(eth_series)

    monkeypatch.setattr(db_mod, "fetch_real_ticks_with_provenance", _fake_provenance)
    monkeypatch.setattr(db_mod, "fetch_recent_news_tags", _fake_news_tags)
    monkeypatch.setattr(db_mod, "fetch_market_signals", _fake_market_signals)
    monkeypatch.setattr(db_mod, "fetch_lead_price_series", _fake_lead_series)

    df = await labels_module.build_labeled_dataset(
        ["pepe"], "1m", 6 * 3600 * 1000,
    )
    assert not df.empty, "seeded dataset should produce labeled rows"

    density = _summarize_feature_density(df, EXTERNAL_STREAM_FEATURE_COLUMNS)
    coverage = _summarize_feature_coverage(df, EXTERNAL_STREAM_FEATURE_COLUMNS)
    for col in EXTERNAL_STREAM_FEATURE_COLUMNS:
        assert coverage[col] == 1.0, (col, coverage[col])
        assert density[col] > 0.0, (
            f"{col} stayed zero across every labeled row even though "
            f"market_signals were seeded — density={density[col]}"
        )

    # And the helper MUST NOT flag any of them as unwired.
    flagged = _detect_unwired_external_streams(df)
    assert flagged == [], f"unexpectedly flagged as unwired: {flagged}"


def test_cross_market_liquidations_asof_join_populates_columns():
    """Task #295 — the three cross-market liquidation columns
    (`btc_liquidations_1h_usd`, `eth_liquidations_1h_usd`,
    `sol_liquidations_1h_usd`) must asof-join from the BTC/ETH/SOL
    pseudo-coin rows in `market_signals` and broadcast onto every
    per-coin training row. When no source rows are supplied the
    columns must fall through to the registered safe default (0.0)."""
    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    ticks = _trending_ticks(120, base)

    # Distinct values per source coin so we can verify each column
    # picks up the RIGHT source — no accidental cross-wiring.
    def _signals(value: float) -> list[dict]:
        return [
            {
                "timestamp_ms": int((base + timedelta(seconds=i * 60)).timestamp() * 1000),
                "liquidations_1h_usd": value + i * 100.0,
            }
            for i in range(120)
        ]

    cross = {
        "btc": _signals(1_000_000.0),
        "eth": _signals(500_000.0),
        "sol": _signals(250_000.0),
    }
    df = build_labeled_frame_for_coin(
        "pepe", "1m", ticks, cross_liq_signals=cross,
    )
    assert not df.empty
    # Every row must carry a value strictly greater than the seed floor
    # for that source — proves the asof join actually populated the
    # column rather than the registered zero default winning.
    assert (df["btc_liquidations_1h_usd"] >= 1_000_000.0).all()
    assert (df["eth_liquidations_1h_usd"] >= 500_000.0).all()
    assert (df["sol_liquidations_1h_usd"] >= 250_000.0).all()
    # And the last row must see the latest snapshot value (asof picks
    # the most recent row at-or-before bucket time).
    assert df["btc_liquidations_1h_usd"].iloc[-1] >= 1_000_000.0 + 50 * 100.0

    # Without sources, the columns degrade to the registered default.
    # Task #633 — that default is now NaN (was 0.0): LightGBM handles
    # NaN natively via `use_missing=true`, so a missing cross-market
    # liquidation pulse is treated as missing rather than as a literal
    # "zero liquidations" signal.
    df_default = build_labeled_frame_for_coin("pepe", "1m", ticks)
    assert not df_default.empty
    for col in (
        "btc_liquidations_1h_usd",
        "eth_liquidations_1h_usd",
        "sol_liquidations_1h_usd",
    ):
        assert df_default[col].isna().all(), (
            f"{col} should default to NaN when no cross-market signals "
            f"are supplied (post-#633), got "
            f"{df_default[col].dropna().head().to_list()}"
        )


@pytest.mark.asyncio
async def test_build_labeled_dataset_fetches_cross_market_liquidations(monkeypatch):
    """Task #295 — `build_labeled_dataset` must call `fetch_market_signals`
    once per cross-market source coin (`btc`, `eth`, `sol`) and surface
    the joined values on every coin's row of the labeled frame."""
    from app import db as db_mod
    from app.training import labels as labels_module

    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    ticks = _trending_ticks(240, base)

    async def _fake_provenance(coin_id, lookback_ms):
        return {
            "ticks": list(ticks),
            "rows_real": len(ticks),
            "rows_synthetic": 0,
            "rejected_synthetic": False,
        }

    async def _fake_news_tags(coin_id, limit=20):
        return []

    seen_signal_calls: list[str] = []

    async def _fake_market_signals(coin_id, lookback_ms):
        seen_signal_calls.append(coin_id)
        # Distinct value per source so we can assert it landed in the
        # right column with no cross-wiring.
        floor = {"btc": 9_000_000.0, "eth": 4_500_000.0, "sol": 1_750_000.0}.get(
            coin_id, 0.0,
        )
        return [
            {
                "timestamp_ms": int((base + timedelta(seconds=i * 60)).timestamp() * 1000),
                "liquidations_1h_usd": floor + i * 50.0,
                "funding_rate": None,
                "open_interest_usd": None,
                "bid_ask_spread_bps": None,
                "mid_price": None,
            }
            for i in range(60)
        ]

    async def _fake_lead_series(reference_coin_id, lookback_ms):
        return []

    monkeypatch.setattr(db_mod, "fetch_real_ticks_with_provenance", _fake_provenance)
    monkeypatch.setattr(db_mod, "fetch_recent_news_tags", _fake_news_tags)
    monkeypatch.setattr(db_mod, "fetch_market_signals", _fake_market_signals)
    monkeypatch.setattr(db_mod, "fetch_lead_price_series", _fake_lead_series)

    df = await labels_module.build_labeled_dataset(
        ["pepe"], "1m", 6 * 3600 * 1000,
    )
    assert not df.empty

    # The trainer must have queried each cross-market pseudo-coin id at
    # least once (BTC, ETH, SOL) plus the per-coin call for `pepe`.
    for src in ("btc", "eth", "sol"):
        assert src in seen_signal_calls, (
            f"build_labeled_dataset never queried market_signals for {src!r}"
        )

    # Every per-coin row must carry the seeded floor for each source.
    assert (df["btc_liquidations_1h_usd"] >= 9_000_000.0).all()
    assert (df["eth_liquidations_1h_usd"] >= 4_500_000.0).all()
    assert (df["sol_liquidations_1h_usd"] >= 1_750_000.0).all()


def test_parity_fixture_still_present():
    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests" / "fixtures" / "quant_brain_parity.json"
    )
    assert fixture.exists(), (
        "quant_brain_parity.json fixture is missing — the parity workflow "
        "would silently no-op"
    )
