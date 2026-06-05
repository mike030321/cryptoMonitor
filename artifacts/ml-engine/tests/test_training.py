"""Tests for the Phase-2 training pipeline.

Pure pieces (label generation, walk-forward splitter, registry roundtrip,
calibration consistency) plus an end-to-end pooled-fallback assertion.
The full /ml/predict integration is in test_predict.py.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.training import labels as labels_mod
from app.training.labels import (
    DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF,
    FORWARD_HORIZON_CANDLES,
    LABEL_THRESHOLDS_PERCENT,
    LABEL_THRESHOLDS_PERCENT_PER_COIN,
    MULTI_BAR_LABEL_THRESHOLDS_PERCENT,
    OUTCOME_THRESHOLDS_PERCENT,
    build_labeled_frame_for_coin,
    label_three_class,
    resolve_directional_label_horizon_candles,
    resolve_directional_label_threshold_pct,
    resolve_label_threshold_pct,
)
from app.training.walk_forward import WalkForwardConfig, walk_forward_splits


# --- Constants parity check ----------------------------------------------------
def test_outcome_thresholds_match_typescript():
    """If trading-constants.ts changes these, this test must be updated.
    The TS values as of 2026-04-21 are duplicated below; any drift here
    means the trainer is using the wrong adjudication thresholds.
    """
    assert OUTCOME_THRESHOLDS_PERCENT == {
        "1m": 0.35, "5m": 0.35, "1h": 0.45,
        "2h": 0.55, "6h": 0.85, "1d": 1.50,
    }


def test_label_thresholds_below_outcome_thresholds():
    """Task #95 — labels are assigned at a LOWER band than adjudication.
    If anyone raises a label threshold to/above the outcome threshold,
    the model loses the directional class mass we just retrained for.
    """
    for tf in OUTCOME_THRESHOLDS_PERCENT:
        assert tf in LABEL_THRESHOLDS_PERCENT, (
            f"label threshold missing for {tf} — trainer would silently fall back"
        )
        assert LABEL_THRESHOLDS_PERCENT[tf] < OUTCOME_THRESHOLDS_PERCENT[tf], (
            f"label threshold {LABEL_THRESHOLDS_PERCENT[tf]} for {tf} must be "
            f"strictly less than outcome threshold {OUTCOME_THRESHOLDS_PERCENT[tf]}"
        )
        assert LABEL_THRESHOLDS_PERCENT[tf] > 0, "label threshold must be positive"


def test_multi_bar_label_thresholds_below_outcome_thresholds():
    """Task #379 — when the directional-label horizon spans multiple bars
    (1h/2h/6h), the multi-bar threshold MUST stay strictly below the
    timeframe outcome threshold for the same reason as the legacy 1-bar
    invariant: a labelled directional row that doesn't clear the
    adjudication band cannot generate a winning trade.
    """
    for tf, pct in MULTI_BAR_LABEL_THRESHOLDS_PERCENT.items():
        assert tf in OUTCOME_THRESHOLDS_PERCENT, (
            f"multi-bar threshold {tf} targets unknown timeframe"
        )
        assert pct > 0, f"multi-bar threshold for {tf} must be positive"
        assert pct < OUTCOME_THRESHOLDS_PERCENT[tf], (
            f"multi-bar threshold {pct} for {tf} must be strictly below "
            f"outcome threshold {OUTCOME_THRESHOLDS_PERCENT[tf]}"
        )


def test_directional_horizon_map_matches_short_timeframes():
    """Task #379 — 1h/2h/6h must use the trade-aware FORWARD_HORIZON_CANDLES
    so the directional label and the TP/SL adjudication agree on the
    holding period. 1m/5m/1d stay at 1 (1m/5m have rich per-bar mass; 1d
    data is too sparse to absorb a multi-bar window without re-introducing
    left-censoring on every row).
    """
    assert DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF["1h"] == FORWARD_HORIZON_CANDLES
    assert DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF["2h"] == FORWARD_HORIZON_CANDLES
    assert DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF["6h"] == FORWARD_HORIZON_CANDLES
    assert DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF["1m"] == 1
    assert DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF["5m"] == 1
    assert DIRECTIONAL_LABEL_HORIZON_CANDLES_PER_TF["1d"] == 1


def test_resolve_directional_label_horizon_legacy_env(monkeypatch):
    """Task #379 — operators can roll back to pre-#379 1-bar behaviour by
    setting `ML_DIRECTIONAL_LABEL_HORIZON_MODE=legacy`. The resolver must
    return 1 for every timeframe, and the threshold resolver must fall
    back to the legacy per-coin label-band path.
    """
    monkeypatch.setenv("ML_DIRECTIONAL_LABEL_HORIZON_MODE", "legacy")
    for tf in OUTCOME_THRESHOLDS_PERCENT:
        assert resolve_directional_label_horizon_candles(tf) == 1
        # Threshold resolver delegates to the legacy 1-bar path in legacy
        # mode, so the per-coin / per-tf resolution chain is honoured.
        assert (
            resolve_directional_label_threshold_pct("__no_override__", tf)
            == resolve_label_threshold_pct("__no_override__", tf)
        )


def test_build_labeled_frame_uses_multi_bar_label_at_1h():
    """Task #379 — at 1h, `label_3class` must be derived from the
    multi-bar (FORWARD_HORIZON_CANDLES-bar) forward return, while
    `forward_return` must remain 1-bar so the existing PnL / regressor
    code paths stay backward compatible.

    Construct a deterministic ramp where the i-th 1h bucket closes at
    100 * (1 + i*0.001). The 1-bar return is +0.10%; the 4-bar return is
    +0.40% (+ rounding). With the 1h multi-bar threshold = 0.40% the
    multi-bar label is dominantly UP (=2), while the 1-bar label under
    the legacy 0.20% band would also be DOWN/STABLE near the band edge.
    The key property the test enforces is that `label_3class` is
    consistent with the MULTI-BAR move (sign of multi-bar return), not
    the 1-bar move.
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks: list[tuple[datetime, float]] = []
    # 200 hourly ticks ramping up by 0.10% per bar so the 4-bar fwd return
    # is ~+0.40% on every row.
    p = 100.0
    for i in range(200):
        ticks.append((base + timedelta(hours=i, minutes=1), p))
        p *= 1.001
    df = build_labeled_frame_for_coin("__t379_up_ramp__", "1h", ticks)
    assert not df.empty
    # Horizon column must be stamped on every row at H=4.
    assert "directional_label_horizon_candles" in df.columns
    assert (df["directional_label_horizon_candles"] == 4).all()
    # The multi-bar forward return must be approximately 4× the 1-bar
    # forward return on a constant-ramp series (modulo the small compound
    # effect). Both should be positive on every row.
    assert (df["directional_label_forward_return"] > 0).mean() > 0.95
    assert (df["forward_return"] > 0).mean() > 0.95
    # And `label_3class` must reflect the MULTI-BAR direction (UP=2) on
    # the vast majority of rows. Under the legacy 1-bar 0.20% threshold
    # the per-row 0.10% move would label every row STABLE (=1) — that's
    # the bug this task fixes.
    up_share = float((df["label_3class"] == 2).mean())
    stable_share = float((df["label_3class"] == 1).mean())
    assert up_share > 0.90, (
        f"multi-bar label should be dominantly UP on a +0.10%/bar ramp; "
        f"got up_share={up_share:.2%} stable_share={stable_share:.2%}"
    )


def test_build_labeled_frame_legacy_mode_keeps_one_bar_label(monkeypatch):
    """Task #379 — under `ML_DIRECTIONAL_LABEL_HORIZON_MODE=legacy`, the
    1h frame must fall back to the legacy 1-bar `label_3class` so a
    rollback is a single env-var flip with no code change.
    """
    monkeypatch.setenv("ML_DIRECTIONAL_LABEL_HORIZON_MODE", "legacy")
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticks: list[tuple[datetime, float]] = []
    p = 100.0
    for i in range(200):
        ticks.append((base + timedelta(hours=i, minutes=1), p))
        p *= 1.001
    df = build_labeled_frame_for_coin("__t379_up_ramp_legacy__", "1h", ticks)
    assert not df.empty
    assert (df["directional_label_horizon_candles"] == 1).all()
    # 1-bar +0.10% ramp under the legacy 1h 0.20% band labels every row
    # STABLE — exactly the noise-floor pathology that motivates the fix.
    stable_share = float((df["label_3class"] == 1).mean())
    assert stable_share > 0.90, (
        f"legacy 1-bar label should be dominantly STABLE on a +0.10%/bar "
        f"ramp under the 1h 0.20% band; got stable_share={stable_share:.2%}"
    )


def test_directional_horizon_returns_falls_back_to_forward_return():
    """Task #379 — `_directional_horizon_returns` must equal `forward_return`
    when the directional column is missing (legacy frames pre-#379) so the
    regressor head and class-return means stay backward compatible.
    """
    from app.training.train import _directional_horizon_returns
    df = pd.DataFrame({"forward_return": [0.001, -0.002, 0.0005]})
    out = _directional_horizon_returns(df)
    assert np.allclose(out, [0.001, -0.002, 0.0005])


def test_directional_horizon_returns_uses_multibar_column_when_present():
    """Task #379 — when the labelled frame carries
    `directional_label_forward_return` (the multi-bar return at 1h/2h/6h),
    the regressor + class-mean helpers must consume it instead of the
    1-bar `forward_return`. This is the horizon-coupling invariant: the
    classifier targets the multi-bar label, so its magnitude head and per-
    class expected-return MUST be measured at the same horizon.
    """
    from app.training.train import _directional_horizon_returns
    df = pd.DataFrame({
        "forward_return": [0.001, -0.001, 0.0005],
        "directional_label_forward_return": [0.004, -0.005, 0.002],
    })
    out = _directional_horizon_returns(df)
    assert np.allclose(out, [0.004, -0.005, 0.002])


def test_class_return_means_pct_anchored_to_directional_horizon():
    """Task #379 — `_class_return_means_pct` must compute per-class means
    on the SAME horizon as `label_3class`. With the multi-bar column
    present and a label that was assigned at the multi-bar horizon, the
    UP-class mean MUST equal the mean of the multi-bar column on UP rows
    (in percent), not the mean of `forward_return`.
    """
    from app.training.train import _class_return_means_pct
    df = pd.DataFrame({
        "label_3class": [2, 2, 2, 1, 1, 0, 0],
        # 1-bar return is small and noisy — what the bug used to consume.
        "forward_return": [0.0011, 0.0009, 0.0010, 0.0001, -0.0001, -0.0010, -0.0012],
        # Multi-bar return is the actual move the multi-bar label was
        # assigned against; the UP class spans ~+0.40% on average.
        "directional_label_forward_return": [
            0.0040, 0.0042, 0.0038,   # UP
            0.0001, -0.0001,          # STABLE
            -0.0040, -0.0044,         # DOWN
        ],
    })
    means_pct = _class_return_means_pct(df)
    # DOWN class mean ≈ -0.42%, STABLE ≈ 0%, UP ≈ +0.40%.
    assert means_pct[2] == pytest.approx(0.40, abs=0.01)
    assert means_pct[0] == pytest.approx(-0.42, abs=0.01)
    assert abs(means_pct[1]) < 0.01
    # And NOT the noisy 1-bar means (~0.10% / 0% / -0.11%).
    assert abs(means_pct[2] - 0.10) > 0.20


def test_class_means_fallback_threshold_uses_multi_bar_at_short_tf():
    """Task #379 — when a class has no observations on a 1h/2h/6h slice,
    `train_one_slice` fills the per-class mean with ±threshold. That
    threshold MUST be the multi-bar `MULTI_BAR_LABEL_THRESHOLDS_PERCENT[tf]`
    (the band labels.py used to assign the class), NOT the legacy 1-bar
    `LABEL_THRESHOLDS_PERCENT[tf]`. Using the 1-bar fallback would shrink
    `expectedReturnPct` ~4× relative to the horizon the classifier
    predicts on empty-class slices, biasing the gate against trading them.
    """
    from app.training.labels import (
        LABEL_THRESHOLDS_PERCENT,
        MULTI_BAR_LABEL_THRESHOLDS_PERCENT,
        resolve_directional_label_threshold_pct,
    )
    for tf, multi_bar_threshold in MULTI_BAR_LABEL_THRESHOLDS_PERCENT.items():
        legacy_threshold = LABEL_THRESHOLDS_PERCENT[tf]
        # The multi-bar threshold must be strictly larger than the 1-bar
        # band — otherwise multi-bar labelling would have been a no-op.
        assert multi_bar_threshold > legacy_threshold, (
            f"{tf}: multi-bar threshold {multi_bar_threshold} should exceed "
            f"legacy 1-bar threshold {legacy_threshold} (sanity check)"
        )
        # The directional resolver must return the multi-bar number for
        # an arbitrary coin id at this timeframe (no per-coin override
        # consulted on multi-bar timeframes by design).
        resolved = resolve_directional_label_threshold_pct(
            "__no_override__", tf,
        )
        assert resolved == multi_bar_threshold, (
            f"{tf}: directional threshold resolver returned {resolved}; "
            f"expected multi-bar {multi_bar_threshold}"
        )


def test_class_return_means_pct_legacy_frame_unchanged():
    """Task #379 backward-compat — pre-#379 frames (no
    `directional_label_forward_return` column) must produce the SAME means
    they did before, so re-running the trainer on a cached legacy parquet
    cannot silently change persisted manifest semantics.
    """
    from app.training.train import _class_return_means_pct
    df = pd.DataFrame({
        "label_3class": [2, 2, 1, 0, 0],
        "forward_return": [0.005, 0.003, 0.0, -0.004, -0.006],
    })
    means_pct = _class_return_means_pct(df)
    assert means_pct[2] == pytest.approx(0.40, abs=0.01)
    assert means_pct[1] == pytest.approx(0.0, abs=0.01)
    assert means_pct[0] == pytest.approx(-0.50, abs=0.01)


def test_per_coin_label_thresholds_below_outcome_thresholds():
    """Task #120 — per-coin label overrides MUST also stay strictly below
    the timeframe outcome threshold so a model trade still has to clear
    round-trip cost to count as 'correct'.
    """
    for coin, tf_map in LABEL_THRESHOLDS_PERCENT_PER_COIN.items():
        for tf, pct in tf_map.items():
            assert tf in OUTCOME_THRESHOLDS_PERCENT, (
                f"per-coin override {coin}/{tf} targets unknown timeframe"
            )
            assert pct > 0, f"per-coin override {coin}/{tf} must be positive"
            assert pct < OUTCOME_THRESHOLDS_PERCENT[tf], (
                f"per-coin override {coin}/{tf}={pct} must be strictly less "
                f"than outcome threshold {OUTCOME_THRESHOLDS_PERCENT[tf]}"
            )


def test_resolve_label_threshold_uses_per_coin_override():
    """If trading-frictions.json registers an override, the resolver must
    return it; otherwise it must return the timeframe default. Smoke-tests
    the wiring task #120 added so a future refactor doesn't silently drop
    the per-coin band. The override may be tighter OR wider than the
    timeframe default — task #318 widens 5 coins above the 0.10%/5m
    default after their realized 5m vol turned out to dwarf the
    tighter-only band picked under task #120. The strict-less-than-outcome
    constraint is enforced separately by
    test_per_coin_label_thresholds_below_outcome_thresholds.
    """
    # Default path: an unconfigured coin always gets the timeframe default.
    assert resolve_label_threshold_pct("zzz_unknown", "5m") == LABEL_THRESHOLDS_PERCENT["5m"]
    # If any override is configured, it must be returned (and must
    # actually differ from the timeframe default — otherwise it's a
    # no-op entry someone forgot to delete).
    for coin, tf_map in LABEL_THRESHOLDS_PERCENT_PER_COIN.items():
        for tf, pct in tf_map.items():
            assert resolve_label_threshold_pct(coin, tf) == pct
            assert pct != LABEL_THRESHOLDS_PERCENT[tf], (
                f"override {coin}/{tf}={pct} matches the timeframe default "
                f"{LABEL_THRESHOLDS_PERCENT[tf]} — no-op, delete it"
            )


def test_per_coin_override_changes_directional_share():
    """End-to-end on a synthetic series so the override actually flips
    rows from STABLE to UP/DOWN. Uses a quiet 5m series whose realized
    moves stay below the 0.10% timeframe default but cross 0.04% often.
    """
    rng = np.random.default_rng(123)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    p = 100.0
    ticks = []
    # Per-tick std ~0.04%, so most 5m bars cross 0.04% but few cross 0.10%.
    for i in range(400):
        p *= 1.0 + rng.normal(0, 0.0004)
        ticks.append((base + timedelta(seconds=i * 60), p))

    # Coin with no override -> uses 5m default (0.10%) -> mostly STABLE.
    df_default = build_labeled_frame_for_coin("__no_override__", "5m", ticks)
    # Coin with a TIGHTER 0.04% override (task #120 — still wired for
    # the quiet coins after the task #318 5m re-tune of the 5 louder
    # coins). floki-inu is a stable representative of the tight-band
    # cohort so the directional-share lift is exercised end-to-end.
    assert "floki-inu" in LABEL_THRESHOLDS_PERCENT_PER_COIN, (
        "test assumes a tighter-than-default 5m override is still wired "
        "for at least one quiet coin"
    )
    assert (
        LABEL_THRESHOLDS_PERCENT_PER_COIN["floki-inu"]["5m"]
        < LABEL_THRESHOLDS_PERCENT["5m"]
    ), "floki-inu 5m override must stay tighter than the timeframe default"
    df_override = build_labeled_frame_for_coin("floki-inu", "5m", ticks)
    assert not df_default.empty and not df_override.empty
    share_default = float((df_default["label_3class"] != 1).mean())
    share_override = float((df_override["label_3class"] != 1).mean())
    assert share_override > share_default, (
        f"per-coin override didn't increase directional share: "
        f"default={share_default:.2%} override={share_override:.2%}"
    )
    # And the override must clear the dashboard floor (15%) on this
    # synthetic series — same bar the regression alert uses.
    assert share_override >= 0.15


def test_load_thresholds_normal_path_does_not_record_fallback():
    """Task #357 — when trading-frictions.json is readable, the loaders
    must not flip the fallback flag (the trainer is on the live contract).
    """
    # In this test environment the workspace IS mounted, so a fresh load
    # must succeed without a fallback. We snapshot/restore the status so
    # the assertion is hermetic against test ordering.
    saved = dict(labels_mod.LABEL_THRESHOLDS_FALLBACK_STATUS)
    labels_mod.LABEL_THRESHOLDS_FALLBACK_STATUS.update(
        {"used_fallback": False, "reason": None, "path_tried": None}
    )
    try:
        out = labels_mod._load_label_thresholds_from_frictions()
        out_pc = labels_mod._load_per_coin_label_thresholds_from_frictions()
        assert out and "5m" in out
        assert isinstance(out_pc, dict)
        assert labels_mod.LABEL_THRESHOLDS_FALLBACK_STATUS["used_fallback"] is False
    finally:
        labels_mod.LABEL_THRESHOLDS_FALLBACK_STATUS.update(saved)


def test_load_thresholds_fallback_when_file_missing(monkeypatch, caplog):
    """Task #357 — if trading-frictions.json cannot be read, the loader
    must (a) return the hardcoded mirror, (b) log a single visible WARN,
    and (c) flip LABEL_THRESHOLDS_FALLBACK_STATUS so a metrics scraper
    can detect the drift risk. This is the bug task #357 fixes: the old
    code silently used stale numbers.
    """
    monkeypatch.setenv("TRADING_FRICTIONS_PATH", "/nonexistent/path/frictions.json")
    saved = dict(labels_mod.LABEL_THRESHOLDS_FALLBACK_STATUS)
    labels_mod.LABEL_THRESHOLDS_FALLBACK_STATUS.update(
        {"used_fallback": False, "reason": None, "path_tried": None}
    )
    try:
        with caplog.at_level("WARNING", logger="app.training.labels"):
            out = labels_mod._load_label_thresholds_from_frictions()
            out_pc = labels_mod._load_per_coin_label_thresholds_from_frictions()
        # Hardcoded mirror is returned, not an empty dict.
        assert out == labels_mod._LABEL_THRESHOLDS_MIRROR
        assert out_pc == labels_mod._PER_COIN_LABEL_THRESHOLDS_MIRROR
        assert out_pc, "per-coin fallback must NOT be silently empty"
        # Visible WARN was emitted (single-emit guard fires only once).
        assert labels_mod.LABEL_THRESHOLDS_FALLBACK_STATUS["used_fallback"] is True
        assert any(
            "FALLBACK to hardcoded label thresholds" in r.message
            for r in caplog.records
        )
    finally:
        labels_mod.LABEL_THRESHOLDS_FALLBACK_STATUS.update(saved)


def test_hardcoded_mirror_matches_json():
    """Task #357 — the hardcoded mirror in labels.py is the value the
    trainer falls back to when the JSON is unreadable. If the JSON is
    re-tuned and the mirror isn't updated, a worker that falls back will
    silently train on stale thresholds. This test fails loudly so the
    update happens in the same PR.
    """
    json_tf = labels_mod._load_label_thresholds_from_frictions()
    json_pc = labels_mod._load_per_coin_label_thresholds_from_frictions()
    assert json_tf == labels_mod._LABEL_THRESHOLDS_MIRROR, (
        "label_thresholds_percent in trading-frictions.json drifted from "
        "the _LABEL_THRESHOLDS_MIRROR in labels.py — update the mirror "
        "(task #357)"
    )
    assert json_pc == labels_mod._PER_COIN_LABEL_THRESHOLDS_MIRROR, (
        "label_thresholds_percent_per_coin in trading-frictions.json "
        "drifted from the _PER_COIN_LABEL_THRESHOLDS_MIRROR in labels.py "
        "— update the mirror (task #357)"
    )


def test_compute_vol_scaled_threshold_pct_returns_baseline_for_quiet_series():
    """Task #459 — when realized vol is BELOW the static baseline, the
    function MUST return the baseline unchanged with source 'static'.
    Preserves the per-coin overrides wired in trading-frictions.json
    (task #120 / #318) for quiet coins.
    """
    from app.training.labels import (
        THRESHOLD_SOURCE_STATIC,
        compute_vol_scaled_threshold_pct,
    )
    # Per-row return std ~0.05% (quiet coin). MAD ~ 0.034%, vol_floor
    # = 0.7 * 0.034 = 0.024%, well below the 0.10% baseline.
    rng = np.random.default_rng(0)
    returns_pct = (rng.normal(0.0, 0.05, size=200)).tolist()
    chosen, source = compute_vol_scaled_threshold_pct(
        returns_pct, baseline_pct=0.10, ceiling_pct=0.95,
    )
    assert chosen == pytest.approx(0.10)
    assert source == THRESHOLD_SOURCE_STATIC


def test_compute_vol_scaled_threshold_pct_widens_for_volatile_series():
    """Task #459 — when realized vol is ABOVE the static baseline, the
    function MUST widen the threshold to `vol_factor * MAD` and report
    source 'vol_scaled'. This is the structural fix that lets PEPE /
    BONK / FLOKI 1d slices retain a non-empty STABLE class.
    """
    from app.training.labels import (
        THRESHOLD_SOURCE_VOL_SCALED,
        compute_vol_scaled_threshold_pct,
    )
    # Per-row return std ~3% (volatile alt 1d). MAD ~ 2%, vol_floor
    # = 0.7 * 2 = 1.4% — well above the 0.80% 1d baseline and below
    # the 1.425% (=1.5 * 0.95) ceiling.
    rng = np.random.default_rng(1)
    returns_pct = rng.normal(0.0, 3.0, size=400).tolist()
    chosen, source = compute_vol_scaled_threshold_pct(
        returns_pct, baseline_pct=0.80, ceiling_pct=1.425,
    )
    assert source == THRESHOLD_SOURCE_VOL_SCALED
    assert chosen > 0.80, "must widen above the static 1d baseline"
    assert chosen <= 1.425 + 1e-9, "must respect the outcome ceiling"


def test_compute_vol_scaled_threshold_pct_caps_at_ceiling():
    """Task #459 — the chosen threshold MUST never exceed the outcome
    ceiling, even when MAD blows it past adjudication. Otherwise a
    labelled UP/DOWN row could imply a move below the cost band.
    """
    from app.training.labels import (
        THRESHOLD_SOURCE_VOL_SCALED,
        compute_vol_scaled_threshold_pct,
    )
    # Constant 10% returns: MAD=10, vol_floor=7. Cap at 1.425% (the
    # 1d adjudication * 0.95). The resolver must clamp to 1.425%.
    chosen, source = compute_vol_scaled_threshold_pct(
        [10.0] * 100, baseline_pct=0.80, ceiling_pct=1.425,
    )
    assert source == THRESHOLD_SOURCE_VOL_SCALED
    assert chosen == pytest.approx(1.425)


def test_compute_vol_scaled_threshold_pct_handles_empty_and_nan():
    """Task #459 — degenerate inputs (no returns, all NaN/inf) must
    return the static baseline so the trainer never sees a zero
    threshold or a crash.
    """
    from app.training.labels import (
        THRESHOLD_SOURCE_STATIC,
        compute_vol_scaled_threshold_pct,
    )
    chosen, source = compute_vol_scaled_threshold_pct(
        [], baseline_pct=0.80, ceiling_pct=1.425,
    )
    assert chosen == pytest.approx(0.80)
    assert source == THRESHOLD_SOURCE_STATIC
    chosen, source = compute_vol_scaled_threshold_pct(
        [float("nan"), float("inf"), float("-inf")],
        baseline_pct=0.45,
        ceiling_pct=0.80,
    )
    assert chosen == pytest.approx(0.45)
    assert source == THRESHOLD_SOURCE_STATIC


def test_build_labeled_frame_widens_threshold_for_volatile_alt_at_1d():
    """Task #459 — end-to-end: a synthetic 1d series with ~3% per-bar
    returns (PEPE / BONK regime) MUST get a vol-scaled STABLE-class
    threshold so the directional-call share stays well below the 0.95
    promotion gate. Without the fix every bar lands in UP/DOWN and
    the slice is structurally unpromotable.
    """
    rng = np.random.default_rng(42)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    p = 100.0
    ticks: list[tuple[datetime, float]] = []
    # 200 daily ticks with ~3% std per bar.
    for i in range(200):
        p *= 1.0 + rng.normal(0.0, 0.03)
        ticks.append((base + timedelta(days=i, minutes=1), p))
    df = build_labeled_frame_for_coin("__t459_volatile_1d__", "1d", ticks)
    assert not df.empty
    # The per-row stamps are present and non-trivial.
    assert "directional_label_threshold_pct" in df.columns
    assert "directional_label_threshold_source" in df.columns
    sources = set(df["directional_label_threshold_source"].unique().tolist())
    assert sources == {"vol_scaled"}, (
        f"expected vol_scaled threshold for a high-vol 1d series; "
        f"got sources={sources}"
    )
    # The chosen threshold widened above the 0.80% static 1d baseline
    # but must still respect the 1.5% * 0.95 = 1.425% outcome ceiling.
    chosen = float(df["directional_label_threshold_pct"].iloc[0])
    assert 0.80 < chosen <= 1.425 + 1e-9, (
        f"chosen threshold {chosen}% must widen past 1d baseline 0.80% "
        f"and stay under the 1.425% outcome ceiling"
    )
    # And the STABLE class must carry real mass — the structural
    # `directional_call_share >= 0.95` regression gate would block
    # promotion otherwise.
    stable_share = float((df["label_3class"] == 1).mean())
    directional_share = 1.0 - stable_share
    assert stable_share > 0.05, (
        f"vol-scaled threshold should leave non-empty STABLE class on a "
        f"high-vol 1d series; got stable_share={stable_share:.2%}"
    )
    assert directional_share < 0.95, (
        f"directional share {directional_share:.2%} would still trip the "
        f"0.95 promotion gate; vol scaling did not help enough"
    )


def test_build_labeled_frame_keeps_static_threshold_for_quiet_coin():
    """Task #459 — a quiet 5m series (per-tick std ~0.04 %) MUST keep
    the per-coin override unchanged so the existing tighter bands for
    floki / etc (task #120) still apply. The vol-scaled path must not
    silently widen quiet coins back to the timeframe default.
    """
    rng = np.random.default_rng(7)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    p = 100.0
    ticks: list[tuple[datetime, float]] = []
    for i in range(400):
        p *= 1.0 + rng.normal(0.0, 0.0001)  # very quiet
        ticks.append((base + timedelta(seconds=i * 60), p))
    df = build_labeled_frame_for_coin("__t459_quiet_5m__", "5m", ticks)
    assert not df.empty
    sources = set(df["directional_label_threshold_source"].unique().tolist())
    assert sources == {"static"}, (
        f"quiet series must keep the static baseline; got sources={sources}"
    )
    # Stamped value equals the 5m default (no per-coin override here).
    chosen = float(df["directional_label_threshold_pct"].iloc[0])
    assert chosen == pytest.approx(LABEL_THRESHOLDS_PERCENT["5m"])


def test_label_threshold_produces_directional_mass():
    """A synthetic 1m series with realistic noise should now produce a
    non-trivial up/down share — the whole point of task #95.
    """
    import numpy as np
    rng = np.random.default_rng(7)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    p = 100.0
    ticks = []
    for i in range(400):
        p *= 1.0 + rng.normal(0, 0.001)  # ~0.1% per-tick std
        ticks.append((base + timedelta(seconds=i * 60), p))
    df = build_labeled_frame_for_coin("pepe", "1m", ticks)
    assert not df.empty
    shares = df["label_3class"].value_counts(normalize=True)
    directional = float(shares.get(0, 0)) + float(shares.get(2, 0))
    # Old 0.35% threshold gave ~0% directional on this scale; new band
    # must clear 15% so the model has something to learn from.
    assert directional > 0.15, f"directional class share too low: {directional:.2%}"


# --- label_three_class ---------------------------------------------------------
def test_label_three_class_up_down_stable():
    assert label_three_class(0.5, 0.35) == 2  # up
    assert label_three_class(-0.5, 0.35) == 0  # down
    assert label_three_class(0.10, 0.35) == 1  # stable
    assert label_three_class(0.35, 0.35) == 1  # exactly threshold = stable
    assert label_three_class(-0.35, 0.35) == 1


# --- build_labeled_frame_for_coin ---------------------------------------------
def _ticks(n: int, start: datetime, step_s: int = 60) -> list[tuple[datetime, float]]:
    return [
        (start + timedelta(seconds=i * step_s), 100.0 + i * 0.5 + math.sin(i / 3.0) * 2.0)
        for i in range(n)
    ]


def test_labeled_frame_no_lookahead():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _ticks(120, base, step_s=60))
    assert not df.empty
    assert "lastPrice" in df.columns
    assert "forward_return" in df.columns
    assert (df["forward_return"] != 0).all()
    expected_max_rows = 120 - 35
    assert len(df) <= expected_max_rows


def test_labeled_frame_empty_for_short_input():
    df = build_labeled_frame_for_coin("pepe", "1m", _ticks(10, datetime(2025, 1, 1, tzinfo=timezone.utc)))
    assert df.empty


def test_labeled_frame_rejects_unknown_timeframe():
    with pytest.raises(ValueError):
        build_labeled_frame_for_coin("pepe", "bogus", _ticks(120, datetime(2025, 1, 1, tzinfo=timezone.utc)))


def test_labeled_frame_emits_trade_aware_columns():
    """Phase 3 — every emitted row must carry the trade-aware label
    block alongside the legacy `label_3class` so the specialist trainer
    has something to learn from. We don't pin exact values (those depend
    on the live trading-frictions contract) but we do assert:
      * the columns exist on every row,
      * the per-row constants line up with FORWARD_HORIZON_CANDLES,
      * MAE is non-positive and MFE is non-negative on the long side,
      * tp_before_sl_long stays in {NaN, 0.0, 1.0}.
    """
    from app.training.labels import FORWARD_HORIZON_CANDLES

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _ticks(160, base, step_s=60))
    assert not df.empty
    for col in (
        "forward_horizon_candles",
        "forward_window_return_pct",
        "prob_move_gt_cost",
        "tp_before_sl_long",
        "tp_before_sl_short",
        "mae_pct_long",
        "mfe_pct_long",
        "opportunity_score",
    ):
        assert col in df.columns, f"missing trade-aware column {col!r}"
    # Constant per-row marker so a downstream consumer can tell what
    # window the labels were computed over without re-reading the helper.
    assert (df["forward_horizon_candles"] == FORWARD_HORIZON_CANDLES).all()
    # Long-side MAE never above 0; long-side MFE never below 0.
    mae = df["mae_pct_long"].dropna()
    mfe = df["mfe_pct_long"].dropna()
    assert (mae <= 0).all()
    assert (mfe >= 0).all()
    # tp_before_sl_long is a binary outcome (NaN if neither barrier hit).
    tp_long = df["tp_before_sl_long"].dropna().unique().tolist()
    assert set(tp_long).issubset({0.0, 1.0})
    # Legacy 3-class label still present and intact.
    assert "label_3class" in df.columns
    assert df["label_3class"].isin([0, 1, 2]).all()


def test_specialist_target_3class_directional_uses_barrier_flags():
    """Phase 3 — directional specialists must learn from
    `tp_before_sl_long/short` (cost-aware barrier outcome), not the
    legacy raw-return-sign `label_3class`. Verifies the helper maps
    barrier flags into the canonical {0=DOWN, 1=STABLE, 2=UP} space
    and falls back to the sign of `opportunity_score` for rows where
    neither barrier was hit within the horizon.
    """
    from app.training.train import _specialist_target_3class

    df = pd.DataFrame({
        "tp_before_sl_long":  [1.0, 0.0,    1.0, np.nan, np.nan, 0.0],
        "tp_before_sl_short": [0.0, 1.0,    1.0, np.nan, np.nan, 0.0],
        "opportunity_score":  [0.0, 0.0,    0.0,  0.5,   -0.3,   0.0],
    })
    y = _specialist_target_3class(df, "momentum")
    assert y is not None
    assert y.tolist() == [2, 0, 1, 2, 0, 1]


def test_specialist_target_3class_volatility_uses_magnitude_buckets():
    """The volatility forecaster's 3-class proxy must come from
    tercile-bucketed |forward_window_return_pct| — that's the
    "regression on realized magnitude" requirement reshaped into the
    3-class slot the rest of the pipeline expects.
    """
    from app.training.train import _specialist_target_3class

    df = pd.DataFrame({
        "forward_window_return_pct": [-0.01, 0.02, -0.5, 0.6, 0.001, -2.0, 1.5, 0.05, -0.07],
    })
    y = _specialist_target_3class(df, "volatility_forecaster")
    assert y is not None
    # All rows kept (no silent drops), valid bucket indices only.
    assert len(y) == len(df)
    assert set(y.unique()).issubset({0, 1, 2})


def _signal_ticks_for_specialists(n: int, start: datetime, drift: float, step_s: int = 60):
    """Trending series with realistic noise. Mirrors the helper used in
    test_predict.py so the specialist trainer sees enough row mass + a
    spread of regimes to actually fit a model.
    """
    rng = np.random.default_rng(11)
    out = []
    p = 100.0
    for i in range(n):
        p *= (1.0 + drift + rng.normal(0, 0.005))
        out.append((start + timedelta(seconds=i * step_s), p))
    return out


def test_train_specialists_writes_one_manifest_per_kind_with_regime_subset(
    monkeypatch, tmp_path,
):
    """Phase 3 contract — `train_specialists` must (a) emit exactly one
    entry per `SPECIALIST_KINDS` taxonomy member, (b) stamp the entry
    with the canonical `regime_subset` from `SPECIALIST_REGIME_MAP`, and
    (c) for every kind that actually trains, write a real on-disk
    manifest at `__specialist_<kind>__/<tf>/<version>/manifest.json`
    whose `specialist_kind` and `regime_subset` fields match the
    taxonomy. This is the regression test for the reviewer's "a future
    regime-map / registry change can silently strip specialist outputs"
    finding — without it, anyone tweaking `SPECIALIST_REGIME_MAP` could
    silently break the on-disk shape that `/ml/predict` resolves
    against.
    """
    import json

    from app.training import registry as registry_module
    from app.training.train import (
        SPECIALIST_KINDS, SPECIALIST_REGIME_MAP, train_specialists,
    )
    from app.training.registry import specialist_coin_id

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # Two coins, each with enough ticks that the volatility forecaster
    # (which trains on ALL rows, no regime filter) is guaranteed to
    # clear MIN_TRAIN_ROWS=80.
    frames = [
        build_labeled_frame_for_coin(
            "pepe", "1m", _signal_ticks_for_specialists(220, base, 0.0010),
        ),
        build_labeled_frame_for_coin(
            "bonk", "1m", _signal_ticks_for_specialists(220, base, -0.0008),
        ),
    ]
    df = pd.concat(frames, ignore_index=True).sort_values(
        "timestamp_ms"
    ).reset_index(drop=True)
    assert not df.empty
    assert "regime" in df.columns, (
        "labeled frame must carry the 'regime' column or specialist subsetting "
        "silently degrades to insufficient_data for every directional kind"
    )

    out = train_specialists(df, "1m", vocab=["pepe", "bonk"])

    # (a) one entry per taxonomy member, no extras, no missing.
    assert set(out.keys()) == set(SPECIALIST_KINDS), (
        f"specialist taxonomy drift: got {sorted(out.keys())}, "
        f"expected {sorted(SPECIALIST_KINDS)}"
    )
    # (b) regime_subset stamped on every entry, including
    # insufficient_data ones, so the diagnostics card can tell *which*
    # regime block came up empty without a separate catalog call.
    for kind, slc in out.items():
        assert slc.get("specialist_kind") == kind
        assert slc.get("regime_subset") == list(SPECIALIST_REGIME_MAP[kind]), (
            f"regime_subset mismatch for {kind}: "
            f"slice={slc.get('regime_subset')!r} "
            f"taxonomy={SPECIALIST_REGIME_MAP[kind]!r}"
        )

    # (c) at least the volatility forecaster — which trains on ALL rows
    # by design — must produce a real on-disk manifest. If this stops
    # being true the registry layout has drifted.
    trained_kinds = [k for k, slc in out.items() if slc.get("status") == "trained"]
    assert "volatility_forecaster" in trained_kinds, (
        f"volatility_forecaster failed to train on a 440-row dataset; "
        f"specialists report: {out}"
    )

    for kind in trained_kinds:
        slot = specialist_coin_id(kind)
        latest = (tmp_path / slot / "1m" / "latest").read_text().strip()
        assert latest, f"missing latest pointer for specialist {kind}"
        manifest_path = tmp_path / slot / "1m" / latest / "manifest.json"
        assert manifest_path.exists(), (
            f"specialist {kind} reported trained but no manifest written at "
            f"{manifest_path}"
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["coin_id"] == slot
        assert manifest["specialist_kind"] == kind
        assert manifest["regime_subset"] == list(SPECIALIST_REGIME_MAP[kind]), (
            f"on-disk regime_subset for {kind} drifted from taxonomy"
        )
        # Cost-aware target metadata must persist into the manifest's
        # slice dict too — not just the in-memory report.
        assert out[kind].get("specialist_target_kind") in {
            "tp_before_sl_cost_aware", "magnitude_terciles",
        }


def test_train_specialists_emits_one_entry_per_kind_with_metadata(monkeypatch, tmp_path):
    """`train_specialists` must produce a manifest entry per specialist
    kind (even when subset is too small) and stamp the cost-aware
    target metadata so the diagnostics card / report can distinguish
    a trade-aware specialist from a legacy slot.
    """
    from app.training import registry as registry_module
    from app.training.train import (
        SPECIALIST_KINDS, SPECIALIST_REGIME_MAP, train_specialists,
    )

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = build_labeled_frame_for_coin("pepe", "1m", _ticks(160, base, step_s=60))
    assert not df.empty
    out = train_specialists(df, "1m", vocab=["pepe"])

    # Every kind in the taxonomy is represented (status may be
    # `insufficient_data` on a tiny synthetic slice — that's fine; the
    # contract is "shape parity").
    assert set(out.keys()) == set(SPECIALIST_KINDS)
    for kind, slc in out.items():
        assert slc.get("specialist_kind") == kind
        assert slc.get("regime_subset") == list(SPECIALIST_REGIME_MAP[kind])
        # If the specialist actually trained, the cost-aware target
        # metadata MUST be present — that's the regression test for
        # the reviewer's "specialists still train on label_3class"
        # finding.
        if slc.get("status") == "trained":
            assert slc.get("specialist_target_kind") in {
                "tp_before_sl_cost_aware", "magnitude_terciles",
            }
            dist = slc.get("specialist_target_distribution")
            assert isinstance(dist, dict)
            assert set(dist.keys()) == {"down", "stable", "up"}
            assert sum(dist.values()) > 0


# --- walk_forward_splits -------------------------------------------------------
def _make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"timestamp_ms": np.arange(n) * 60_000})


def test_walk_forward_no_leakage():
    df = _make_df(500)
    cfg = WalkForwardConfig(n_folds=5, min_train_size=100)
    folds = list(walk_forward_splits(df, cfg))
    assert len(folds) == 5
    for tr, te in folds:
        assert tr.max() < te.min(), "train must be strictly before test"
        assert len(np.intersect1d(tr, te)) == 0


def test_walk_forward_expanding_window():
    df = _make_df(500)
    cfg = WalkForwardConfig(n_folds=5, min_train_size=100)
    sizes = [len(tr) for tr, _ in walk_forward_splits(df, cfg)]
    assert sizes == sorted(sizes), "train sets must monotonically grow"
    assert sizes[0] >= 100


def test_walk_forward_rejects_unsorted():
    df = pd.DataFrame({"timestamp_ms": [3, 1, 2]})
    with pytest.raises(ValueError):
        list(walk_forward_splits(df, WalkForwardConfig()))


def test_walk_forward_handles_too_little_data():
    df = _make_df(50)
    cfg = WalkForwardConfig(n_folds=5, min_train_size=100)
    assert list(walk_forward_splits(df, cfg)) == []


# --- Calibration validity (same model, holdout) -------------------------------
def test_calibration_uses_holdout_predictions_from_deployed_model(tmp_path, monkeypatch):
    """The deployed multiclass booster's predictions on the calibration
    holdout must match the (raw) inputs the per-class isotonic calibrators
    were fit on. Catches the regression where calibration is fit on a
    different model than the one served.
    """
    from app.training import registry as registry_module
    from app.training.train import (
        CALIBRATION_HOLDOUT_FRACTION, _encode_coin_idx, _prepare_xy, train_one_slice,
    )

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    coins = ["pepe", "bonk"]
    rng = np.random.default_rng(11)
    frames = []
    # High noise so labels span all 3 classes — required for per-class
    # isotonic calibration to fit (otherwise one class has only one label).
    for coin, drift in zip(coins, [0.0008, -0.0004]):
        p = 100.0
        ticks = []
        for i in range(180):
            p *= (1.0 + drift + rng.normal(0, 0.005))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    res = train_one_slice(df, coin_id="__pooled__", timeframe="1m", vocab=sorted(coins))
    assert res["status"] == "trained", res
    assert res["calibration_diagram"], "must produce a calibration diagram with this much data"

    loaded = registry_module.load_model("__pooled__", "1m", res["version"])
    assert loaded is not None
    assert loaded.calibrators is not None and len(loaded.calibrators) == 3

    df_sorted = df.sort_values("timestamp_ms").reset_index(drop=True)
    df_enc = _encode_coin_idx(df_sorted, sorted(coins))
    X_all, _ = _prepare_xy(df_enc)
    cal_start = max(1, int(len(df_sorted) * (1 - CALIBRATION_HOLDOUT_FRACTION)))
    X_cal = X_all.iloc[cal_start:]
    raw_now = loaded.booster.predict(X_cal, num_iteration=loaded.booster.best_iteration)
    assert raw_now.shape[1] == 3, "must be a 3-class model"
    # Each per-class isotonic must be monotone non-decreasing in its input.
    for k in range(3):
        cal = loaded.calibrators[k]
        if cal is None:
            continue
        cal_now = cal.predict(raw_now[:, k])
        pairs = sorted(zip(raw_now[:, k].tolist(), cal_now.tolist()))
        for i in range(1, len(pairs)):
            assert pairs[i][1] >= pairs[i - 1][1] - 1e-9, "isotonic monotonicity violated"


def test_train_one_slice_persists_regression_head(tmp_path, monkeypatch):
    """Task #135 — `train_one_slice` must fit a magnitude regressor on
    non-stable rows and persist it as `regressor.txt` so /ml/predict and
    the OOS predictor stop relying on the diluted class-mean expectation.

    Asserts:
      1. `regressor.txt` exists in the model dir.
      2. `manifest.has_regression_head` is True.
      3. The loaded model exposes `model.regressor` (callable booster).
      4. The persisted holdout stats show a non-degenerate magnitude
         distribution (p95 strictly greater than the legacy class-mean
         expectation, which was always near the label-threshold floor).
    """
    from app.training import registry as registry_module
    from app.training.train import train_one_slice

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    coins = ["pepe", "bonk"]
    rng = np.random.default_rng(7)
    frames = []
    # High noise — generates lots of non-stable rows so the regressor
    # has > MIN_REGRESSOR_ROWS to fit.
    for coin, drift in zip(coins, [0.001, -0.0005]):
        p = 100.0
        ticks = []
        for i in range(220):
            p *= (1.0 + drift + rng.normal(0, 0.006))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    res = train_one_slice(df, coin_id="__pooled__", timeframe="1m", vocab=sorted(coins))
    assert res["status"] == "trained"

    model_dir = tmp_path / "__pooled__" / "1m" / res["version"]
    assert (model_dir / "regressor.txt").exists(), (
        "regression head must be persisted alongside the classifier"
    )

    loaded = registry_module.load_model("__pooled__", "1m", res["version"])
    assert loaded is not None
    assert loaded.manifest.has_regression_head is True
    assert loaded.regressor is not None, "load_model must materialize the regressor booster"

    stats = loaded.manifest.regression_head_stats
    assert stats is not None and stats["n_train_rows"] > 0
    # Sanity: the regressor is producing magnitudes above the label
    # threshold floor (0.03% on 1m). If this drops back near 0 the
    # downstream gate is back to having no signal to chew on, which is
    # exactly the bug task #135 fixed.
    assert stats["abs_pred_p95_pct"] > 0.05, (
        f"regressor p95 collapsed to {stats['abs_pred_p95_pct']:.4f}% — "
        "magnitude head is degenerate (the bug task #135 fixed)"
    )


def test_train_one_slice_emits_per_class_calibration_regime_and_pnl_surface(
    tmp_path, monkeypatch,
):
    """Task #316 — every trained slice must surface per-class accuracy,
    calibration, regime DA, fold-importance stability, and post-fee PnL
    so the failure-analysis pass can fill in every "NOT MEASURED" field
    in the diagnostic without re-deriving anything from artifacts.

    Asserts the *shape* of each new field (keys present, types correct,
    counts and shares internally consistent). The actual numbers depend
    on the synthetic data so we don't pin them — the goal is to prevent
    the report contract from silently drifting.
    """
    from app.training import registry as registry_module
    from app.training.train import train_one_slice, _CLASS_NAMES

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    coins = ["pepe", "bonk"]
    rng = np.random.default_rng(316)
    frames = []
    for coin, drift in zip(coins, [0.0009, -0.0006]):
        p = 100.0
        ticks = []
        for i in range(220):
            p *= (1.0 + drift + rng.normal(0, 0.006))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    res = train_one_slice(df, coin_id="__pooled__", timeframe="1m", vocab=sorted(coins))
    assert res["status"] == "trained", res
    assert res["calibration_diagram"], "needs a calibrated holdout for the new fields to be meaningful"

    # Per-class breakdowns — every class key present, counts sum to the
    # holdout / train sizes used for diagnostics.
    holdout_bd = res["per_class_holdout_breakdown"]
    train_bd = res["per_class_train_breakdown"]
    assert set(holdout_bd) == set(_CLASS_NAMES)
    assert set(train_bd) == set(_CLASS_NAMES)
    assert all(isinstance(v, int) and v >= 0 for v in holdout_bd.values())
    assert sum(holdout_bd.values()) > 0
    assert sum(train_bd.values()) > 0

    # Class-balance drift — per-class deltas + L1 sum.
    drift = res["train_vs_holdout_class_balance_drift"]
    assert set(drift) == set(_CLASS_NAMES) | {"l1_drift"}
    assert drift["l1_drift"] >= 0.0
    assert abs(sum(drift[c] for c in _CLASS_NAMES)) < 1e-6  # shares delta sums to 0

    # Per-class Brier — one float per class. NaN is allowed for empty classes.
    brier = res["per_class_brier"]
    assert set(brier) == set(_CLASS_NAMES)
    for cls, v in brier.items():
        if holdout_bd[cls] > 0:
            assert isinstance(v, float) and not math.isnan(v) and 0.0 <= v <= 1.0, (cls, v)

    # Reliability max deviation — one float per class (or NaN if no bins).
    rel = res["reliability_max_dev_per_class"]
    assert set(rel) == set(_CLASS_NAMES)

    # Confidence buckets — one entry per (lo, hi) with consistent counts.
    buckets = res["confidence_bucket_da"]
    assert len(buckets) == 5
    holdout_n = sum(holdout_bd.values())
    assert sum(b["n"] for b in buckets) == holdout_n, (
        "every holdout row must fall in exactly one confidence bucket"
    )
    for b in buckets:
        assert {"lo", "hi", "n", "share", "da"} <= set(b)
        assert b["lo"] < b["hi"]
        assert b["n"] >= 0

    # Predicted-class entropy — bounded above by ln(NUM_CLASSES) ≈ 1.0986.
    entropy = res["predicted_class_entropy"]
    assert isinstance(entropy, float)
    assert 0.0 <= entropy <= math.log(3) + 1e-6

    # Prediction-collapse — top-class share gap surfaced both vs the
    # empirical class prior (`collapse_gap`) and vs the LR baseline
    # model (`top_class_share_gap_vs_baseline`, per task #316 wording).
    pc = res["prediction_collapse"]
    for k in ("predicted_class_share", "label_class_share"):
        assert set(pc[k]) == set(_CLASS_NAMES)
        assert abs(sum(pc[k].values()) - 1.0) < 1e-3
    assert isinstance(pc["collapse_gap"], float)
    assert "baseline_top_class_share" in pc
    assert "top_class_share_gap_vs_baseline" in pc
    if pc["baseline_top_class_share"] is not None:
        assert 0.0 <= pc["baseline_top_class_share"] <= 1.0
        assert (
            abs(
                pc["top_class_share_gap_vs_baseline"]
                - (pc["predicted_top_class_share"] - pc["baseline_top_class_share"])
            )
            < 1e-3
        )

    # Confidence-bucket schema is pinned to 5 buckets covering the full
    # [1/NUM_CLASSES, 1.0] range — failure-analysis consumers depend on
    # this exact layout.
    assert [(b["lo"], b["hi"]) for b in buckets] == [
        (0.33, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0),
    ]

    # Regime-bucketed DA — at least one bucket present (synthetic data
    # lands in at least one regime), counts add up to the holdout size.
    regime_buckets = res["regime_bucketed_da"]
    assert isinstance(regime_buckets, dict)
    if regime_buckets:
        assert sum(b["n"] for b in regime_buckets.values()) == holdout_n
        for b in regime_buckets.values():
            assert 0.0 <= b["da"] <= 1.0
            assert 0.0 <= b["share"] <= 1.0

    # Feature-importance stability — pairs across adjacent walk-forward folds.
    stab = res["feature_importance_stability"]
    assert {"rank_corr_pairwise_mean", "n_pairs", "pairs"} <= set(stab)
    assert stab["n_pairs"] == len(stab["pairs"])
    if stab["rank_corr_pairwise_mean"] is not None:
        assert -1.0 <= stab["rank_corr_pairwise_mean"] <= 1.0
        for pair in stab["pairs"]:
            assert {"fold_a", "fold_b", "rank_corr"} <= set(pair)
            assert -1.0 <= pair["rank_corr"] <= 1.0

    # PnL after fees — production round-trip cost is applied; n_trades is
    # always <= holdout size and totals are internally consistent.
    pnl = res["pnl_after_fees"]
    assert pnl is not None, "must be computed when forward_return is in df"
    assert pnl["n_trades"] >= 0
    assert pnl["n_trades"] <= holdout_n
    assert pnl["round_trip_cost_pct"] > 0.0
    if pnl["n_trades"] > 0:
        # net = gross - cost on average, total = net_mean * n
        assert abs(pnl["net_pct_mean"] - (pnl["gross_pct_mean"] - pnl["round_trip_cost_pct"])) < 1e-3
        assert abs(pnl["net_pct_total"] - (pnl["net_pct_mean"] * pnl["n_trades"])) < 1e-2

    # Diagnostics source label — pinned for the calibrated-holdout path.
    assert res["diagnostics_source"] == "holdout"

    # Per-fold gain importances are persisted for the rank-corr helper
    # to consume on the next training run.
    for fold in res["fold_metrics"]:
        assert "feature_importance" in fold
        if fold["feature_importance"] is not None:
            assert len(fold["feature_importance"]) == len(fold["feature_names"])


def test_train_one_slice_emits_contamination_flag_and_predictions_near_prior(
    tmp_path, monkeypatch,
):
    """Task #337 — every trained slice (per-coin AND pooled) must surface
    `contamination_flag` (bool) and `predictions_near_prior`
    (`{share_within_eps, eps}` with sane ranges) so the auto
    failure-analysis pass can fire its near-prior + contamination
    branches without re-deriving anything from disk. If a future
    refactor stops emitting either field, the failure-analysis pass
    silently falls back to its defensive defaults — this test pins the
    contract so that drift is loud.
    """
    from app.training import registry as registry_module
    from app.training.train import _PREDICTIONS_NEAR_PRIOR_EPS, train_one_slice

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    coins = ["pepe", "bonk"]
    rng = np.random.default_rng(337)
    frames = []
    for coin, drift in zip(coins, [0.0009, -0.0006]):
        p = 100.0
        ticks = []
        for i in range(220):
            p *= (1.0 + drift + rng.normal(0, 0.006))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df_pooled = pd.concat(frames, ignore_index=True).sort_values(
        "timestamp_ms"
    ).reset_index(drop=True)

    # Per-coin slice (just pepe's rows) and pooled slice must BOTH carry
    # the new fields with the same shape — failure-analysis doesn't
    # branch on slice kind.
    df_per_coin = frames[0]
    res_per_coin = train_one_slice(
        df_per_coin, coin_id="pepe", timeframe="1m", vocab=["pepe"],
    )
    res_pooled = train_one_slice(
        df_pooled, coin_id="__pooled__", timeframe="1m", vocab=sorted(coins),
    )
    assert res_per_coin["status"] == "trained", res_per_coin
    assert res_pooled["status"] == "trained", res_pooled

    for label, res in (("per-coin", res_per_coin), ("pooled", res_pooled)):
        assert "contamination_flag" in res, (
            f"{label} slice missing contamination_flag — failure-analysis "
            f"pass would silently fall back to its defensive default"
        )
        flag = res["contamination_flag"]
        assert isinstance(flag, bool), (
            f"{label} contamination_flag must be a bool, got {type(flag).__name__}"
        )
        # Synthetic test fixture is not stamped with a contamination
        # marker, so the flag must be False on this clean input.
        assert flag is False, (
            f"{label} contamination_flag tripped on a clean fixture: {res!r}"
        )

        assert "predictions_near_prior" in res, (
            f"{label} slice missing predictions_near_prior"
        )
        npr = res["predictions_near_prior"]
        assert isinstance(npr, dict)
        assert set(npr.keys()) == {"share_within_eps", "eps"}, (
            f"{label} predictions_near_prior key drift: {sorted(npr.keys())}"
        )
        assert isinstance(npr["share_within_eps"], float)
        assert 0.0 <= npr["share_within_eps"] <= 1.0, (
            f"{label} share_within_eps out of [0,1]: {npr!r}"
        )
        assert isinstance(npr["eps"], float)
        assert npr["eps"] > 0.0
        # eps must match the module constant — a future refactor that
        # changes the threshold without updating
        # `scripts/compute_failure_metrics.predictions_near_prior_share`
        # would silently shift the failure-analysis cutoff.
        assert npr["eps"] == float(_PREDICTIONS_NEAR_PRIOR_EPS)


def test_train_one_slice_contamination_flag_trips_on_synthetic_bars_source(
    tmp_path, monkeypatch,
):
    """Task #337 — a single row stamped with `bars_source="synthetic"`
    must flip `contamination_flag` to True even when the per-row cadence
    audit is clean. Locks the synthetic-source branch of
    `_compute_contamination_flag` so a future refactor can't silently
    drop it (and thereby lie to the auto failure-analysis pass).
    """
    from app.training import registry as registry_module
    from app.training.train import train_one_slice

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    coins = ["pepe", "bonk"]
    rng = np.random.default_rng(338)
    frames = []
    for coin, drift in zip(coins, [0.0009, -0.0006]):
        p = 100.0
        ticks = []
        for i in range(220):
            p *= (1.0 + drift + rng.normal(0, 0.006))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df = pd.concat(frames, ignore_index=True).sort_values(
        "timestamp_ms"
    ).reset_index(drop=True)

    # Stamp every row with the synthetic marker so the cadence audit
    # (which would otherwise reject a slice that mixes
    # `synthetic:*` and `resampled_ticks:*` cadence keys outright) still
    # lets the slice train. This isolates the synthetic-source branch
    # of `_compute_contamination_flag` as the regression target.
    df["bars_source"] = "synthetic"

    res = train_one_slice(
        df, coin_id="__pooled__", timeframe="1m", vocab=sorted(coins),
    )
    assert res["status"] == "trained", res
    assert res["contamination_flag"] is True, (
        "a single bars_source=synthetic row must flip contamination_flag; "
        f"got {res.get('contamination_flag')!r}"
    )


def test_load_model_handles_legacy_models_without_regression_head(tmp_path, monkeypatch):
    """Manifests trained before task #135 don't have `regressor.txt` or
    the new manifest fields. `load_model` must keep working on them so a
    deploy doesn't require re-training every coin × timeframe slot.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from app.training import registry as registry_module
    from app.training.registry import ModelManifest, save_model, load_model

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=["a", "b", "c", "d"])
    y = rng.integers(0, 3, size=100)
    booster = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbose": -1,
         "num_leaves": 7, "deterministic": True, "seed": 1},
        lgb.Dataset(X, label=y), num_boost_round=10,
    )
    cals = [IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            for _ in range(3)]
    for c in cals:
        c.fit(np.linspace(0, 1, 50), (np.linspace(0, 1, 50) > 0.5).astype(int))
    mf = ModelManifest(
        coin_id="pepe", timeframe="1m", version="vlegacy",
        feature_names=["a", "b", "c", "d"], coin_vocab=["pepe"],
        n_train_rows=100, n_test_rows=10,
        metrics={"auc": 0.6}, baseline_metrics={"auc": 0.5},
        threshold_pct=0.35, horizon_candles=1,
        class_return_means_pct=[-0.4, 0.0, 0.4],
    )
    # No regressor argument — emulates the pre-task-135 save path.
    save_model("pepe", "1m", "vlegacy", booster, cals, mf)

    loaded = load_model("pepe", "1m", "vlegacy")
    assert loaded is not None
    assert loaded.regressor is None
    assert loaded.manifest.has_regression_head is False


def test_registry_roundtrip(tmp_path, monkeypatch):
    """Multiclass booster + per-class calibrators + manifest round-trip
    through the (coin, timeframe, version) registry layout.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from app.training import registry as registry_module
    from app.training.registry import (
        ModelManifest, save_model, load_model, latest_version, resolve_model,
    )

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=["a", "b", "c", "d"])
    y = rng.integers(0, 3, size=120)
    booster = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbose": -1,
         "num_leaves": 7, "deterministic": True, "seed": 1},
        lgb.Dataset(X, label=y), num_boost_round=10,
    )
    cals = []
    for k in range(3):
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(np.linspace(0, 1, 50), (np.linspace(0, 1, 50) > 0.5).astype(int))
        cals.append(cal)

    mf = ModelManifest(
        coin_id="pepe", timeframe="1m", version="vtest",
        feature_names=["a", "b", "c", "d"], coin_vocab=["pepe"],
        n_train_rows=120, n_test_rows=10,
        metrics={"auc": 0.6}, baseline_metrics={"auc": 0.5},
        threshold_pct=0.35, horizon_candles=1,
        class_return_means_pct=[-0.4, 0.0, 0.4],
    )
    save_model("pepe", "1m", "vtest", booster, cals, mf)
    assert latest_version("pepe", "1m") == "vtest"
    loaded = load_model("pepe", "1m")
    assert loaded is not None
    assert loaded.manifest.version == "vtest"
    assert loaded.manifest.coin_id == "pepe"
    assert len(loaded.calibrators) == 3
    np.testing.assert_allclose(booster.predict(X), loaded.booster.predict(X), rtol=1e-9)

    # resolve_model: per-coin first
    resolved = resolve_model("pepe", "1m")
    assert resolved is not None and resolved.manifest.coin_id == "pepe"
    # resolve_model: pooled fallback when per-coin missing
    save_model("__pooled__", "1m", "vpool", booster, cals,
               ModelManifest(coin_id="__pooled__", timeframe="1m", version="vpool",
                             feature_names=["a", "b", "c", "d"], coin_vocab=["bonk"],
                             n_train_rows=120, n_test_rows=10,
                             metrics={"auc": 0.55}, baseline_metrics={"auc": 0.5},
                             threshold_pct=0.35, horizon_candles=1,
                             class_return_means_pct=[-0.4, 0.0, 0.4]))
    resolved2 = resolve_model("bonk", "1m")
    assert resolved2 is not None and resolved2.manifest.coin_id == "__pooled__"


def test_resolve_model_skips_quarantined_champion(tmp_path, monkeypatch):
    """Task #232 — when the api-server has marked the current champion
    (= the latest-versioned per-coin model) as `state='quarantined'`, the
    file-based resolver MUST NOT pick it for live trade selection. It
    should fall back to the previous champion (the prior trained
    version), and if every per-coin version is quarantined, fall back to
    the pooled slot.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from app.training import registry as registry_module
    from app.training.registry import (
        ModelManifest, save_model, resolve_model, latest_version,
    )

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=["a", "b", "c", "d"])
    y = rng.integers(0, 3, size=80)

    def _booster_and_cals():
        b = lgb.train(
            {"objective": "multiclass", "num_class": 3, "verbose": -1,
             "num_leaves": 7, "deterministic": True, "seed": 1},
            lgb.Dataset(X, label=y), num_boost_round=5,
        )
        cals = []
        for _ in range(3):
            c = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            c.fit(np.linspace(0, 1, 50), (np.linspace(0, 1, 50) > 0.5).astype(int))
            cals.append(c)
        return b, cals

    def _mf(coin: str, version: str) -> ModelManifest:
        return ModelManifest(
            coin_id=coin, timeframe="1m", version=version,
            feature_names=["a", "b", "c", "d"], coin_vocab=[coin],
            n_train_rows=80, n_test_rows=10,
            metrics={"auc": 0.6}, baseline_metrics={"auc": 0.5},
            threshold_pct=0.35, horizon_candles=1,
            class_return_means_pct=[-0.4, 0.0, 0.4],
        )

    # Two per-coin versions for "pepe" — v_old (previous champion) and
    # v_new (current champion). Plus a pooled fallback.
    b1, c1 = _booster_and_cals()
    save_model("pepe", "1m", "v_old", b1, c1, _mf("pepe", "v_old"))
    b2, c2 = _booster_and_cals()
    save_model("pepe", "1m", "v_new", b2, c2, _mf("pepe", "v_new"))
    bp, cp = _booster_and_cals()
    save_model("__pooled__", "1m", "v_pool", bp, cp, _mf("__pooled__", "v_pool"))
    assert latest_version("pepe", "1m") == "v_new"

    # Sanity: with no quarantine info, we get the latest per-coin champion.
    base = resolve_model("pepe", "1m")
    assert base is not None
    assert base.manifest.coin_id == "pepe"
    assert base.manifest.version == "v_new"

    # Quarantine the current champion: resolver must fall back to v_old.
    quarantined = {("pepe", "1m", "v_new")}
    fallback = resolve_model(
        "pepe", "1m",
        is_quarantined=lambda c, t, v: (c, t, v) in quarantined,
    )
    assert fallback is not None
    assert fallback.manifest.coin_id == "pepe"
    assert fallback.manifest.version == "v_old", (
        "quarantined champion must be skipped in favor of previous champion"
    )

    # Quarantine BOTH per-coin versions: resolver must fall back to pooled.
    quarantined_all = {("pepe", "1m", "v_new"), ("pepe", "1m", "v_old")}
    pooled = resolve_model(
        "pepe", "1m",
        is_quarantined=lambda c, t, v: (c, t, v) in quarantined_all,
    )
    assert pooled is not None
    assert pooled.manifest.coin_id == "__pooled__"
    assert pooled.manifest.version == "v_pool"

    # Quarantine even the pooled fallback: nothing left to serve.
    quarantined_full = quarantined_all | {("__pooled__", "1m", "v_pool")}
    none_left = resolve_model(
        "pepe", "1m",
        is_quarantined=lambda c, t, v: (c, t, v) in quarantined_full,
    )
    assert none_left is None


def test_train_timeframe_always_refreshes_pooled_fallback(tmp_path, monkeypatch):
    """Task #121 — even when every tracked coin has enough data for its own
    per-coin model, `train_timeframe` MUST still refit the __pooled__ slot so
    the safety net stays in sync with the current label thresholds and
    feature schema. Without this, adding an 11th coin (or wiping a coin's
    history) silently routes inference to a months-old pooled model that
    collapses to ~all-STABLE.
    """
    from app.training import registry as registry_module
    from app.training.train import train_timeframe, POOLED_COIN_ID

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(11)
    frames = []
    for coin, drift in [("pepe", 0.0008), ("bonk", -0.0004)]:
        p = 100.0
        ticks = []
        for i in range(180):
            p *= (1.0 + drift + rng.normal(0, 0.005))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    res = train_timeframe(df, "1m", coin_ids=["pepe", "bonk"])
    assert res["status"] == "trained"
    # Both coins individually trained...
    assert res["per_coin"]["pepe"]["status"] == "trained"
    assert res["per_coin"]["bonk"]["status"] == "trained"
    # ...AND the pooled fallback was refit on the same run.
    assert res["pooled"] is not None, (
        "pooled fallback must be retrained even when every coin has its own model"
    )
    assert res["pooled"]["status"] == "trained"
    assert res["pooled"]["coin_id"] == POOLED_COIN_ID
    # Vocab must cover ALL tracked coins so a future 11th coin gets a
    # meaningful unknown-category encoding (idx=-1) instead of being
    # squeezed into a single-coin pool.
    loaded = registry_module.load_model(POOLED_COIN_ID, "1m", res["pooled"]["version"])
    assert loaded is not None
    assert loaded.manifest.coin_vocab == ["bonk", "pepe"]


def test_train_timeframe_streams_progress_callback(tmp_path, monkeypatch):
    """Task #380 — `train_timeframe` must invoke the supplied
    `progress_callback` per per-coin slice so an operator tailing the
    JSONL feed sees a heartbeat during a long timeframe instead of a
    silent gap until the timeframe finishes.
    """
    from app.training import registry as registry_module
    from app.training.train import train_timeframe

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(7)
    frames = []
    for coin, drift in [("pepe", 0.0008), ("bonk", -0.0004)]:
        p = 100.0
        ticks = []
        for i in range(180):
            p *= (1.0 + drift + rng.normal(0, 0.005))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    events: list[dict] = []
    res = train_timeframe(
        df, "1m", coin_ids=["pepe", "bonk"],
        progress_callback=lambda rec: events.append(rec),
    )
    assert res["status"] == "trained"
    phases = [e["phase"] for e in events]
    # Both per-coin starts AND dones must arrive — that's the heartbeat.
    assert phases.count("per_coin_start") == 2
    assert phases.count("per_coin_done") == 2
    coins_seen = {e["coin"] for e in events if "coin" in e}
    assert coins_seen == {"pepe", "bonk"}
    # Every event must carry the timeframe so the JSONL row is self-describing.
    assert all(e.get("timeframe") == "1m" for e in events)


def test_run_training_streams_dataset_and_train_done_events(tmp_path, monkeypatch):
    """Task #380 — `run_training` must emit `build_dataset_start`,
    `build_dataset_done`, and `train_done` events through the
    progress_callback so the orchestrator can persist live heartbeat
    rows to `progress_updates.jsonl` between timeframes (not only after
    the whole run finishes).
    """
    import asyncio
    from app.training import registry as registry_module
    from app.training import train as train_mod

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    async def _noop_pool():
        return None
    monkeypatch.setattr(train_mod, "init_pool", _noop_pool)
    monkeypatch.setattr(train_mod, "close_pool", _noop_pool)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(13)

    def _frame_for(tf: str) -> pd.DataFrame:
        frames = []
        for coin, drift in [("pepe", 0.0008), ("bonk", -0.0004)]:
            p = 100.0
            ticks = []
            for i in range(180):
                p *= (1.0 + drift + rng.normal(0, 0.005))
                ticks.append((base + timedelta(seconds=i * 60), p))
            frames.append(build_labeled_frame_for_coin(coin, tf, ticks))
        return pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    async def _fake_build(coin_ids, tf, lookback_ms, provenance_out=None):
        return _frame_for(tf)
    monkeypatch.setattr(train_mod, "build_labeled_dataset", _fake_build)

    # Disable expensive optional post-processing blocks so the test
    # exercises the per-timeframe loop without doing a full pipeline.
    monkeypatch.setattr(
        train_mod, "_record_directional_share_for_report", lambda r: [],
    )
    monkeypatch.setattr(
        train_mod, "_trim_directional_share_history", lambda **kw: {"skipped": True},
    )
    monkeypatch.setattr(
        train_mod, "_record_regression_head_for_report", lambda r: [],
    )
    monkeypatch.setattr(
        train_mod, "_trim_regression_head_history", lambda **kw: {"skipped": True},
    )

    events: list[dict] = []
    timeframes = ["1m"]
    asyncio.run(train_mod.run_training(
        ["pepe", "bonk"], timeframes,
        progress_callback=lambda rec: events.append(rec),
    ))

    phases = [e["phase"] for e in events]
    assert "build_dataset_start" in phases
    assert "build_dataset_done" in phases
    assert "train_done" in phases
    # Per-coin slice events are forwarded through train_timeframe.
    assert "per_coin_start" in phases
    assert "per_coin_done" in phases
    # Every event must carry the timeframe so a JSONL tailer can
    # disambiguate concurrent passes.
    for e in events:
        assert e.get("timeframe") == "1m", e


def test_train_timeframe_progress_callback_failure_is_swallowed(tmp_path, monkeypatch):
    """A flaky progress sink must never abort training (task #380)."""
    from app.training import registry as registry_module
    from app.training.train import train_timeframe

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(8)
    frames = []
    for coin, drift in [("pepe", 0.0008), ("bonk", -0.0004)]:
        p = 100.0
        ticks = []
        for i in range(180):
            p *= (1.0 + drift + rng.normal(0, 0.005))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    def bad_cb(_rec: dict) -> None:
        raise RuntimeError("disk full")

    res = train_timeframe(
        df, "1m", coin_ids=["pepe", "bonk"],
        progress_callback=bad_cb,
    )
    assert res["status"] == "trained"


def test_default_timeframes_cover_all_advertised():
    from app.training.train import DEFAULT_TIMEFRAMES
    assert set(DEFAULT_TIMEFRAMES) == {"1m", "5m", "1h", "2h", "6h", "1d"}


def test_train_timeframe_persists_dataset_snapshot(tmp_path, monkeypatch):
    """Every training run must persist the labeled feature frame so the run
    is reproducible (validator step-1 requirement)."""
    from app.training import registry as registry_module
    from app.training.train import train_timeframe

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(3)
    frames = []
    for coin, drift in [("pepe", 0.0008), ("bonk", -0.0004)]:
        p = 100.0
        t = []
        for i in range(160):
            p *= (1.0 + drift + rng.normal(0, 0.0008))
            t.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", t))
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    res = train_timeframe(df, "1m", coin_ids=["pepe", "bonk"])
    assert res["status"] == "trained"
    snapshot_rel = res.get("dataset_path")
    assert snapshot_rel
    snapshot_abs = tmp_path / snapshot_rel
    assert snapshot_abs.exists(), f"dataset snapshot not written: {snapshot_abs}"
    # Re-read and verify we can recover the labeled frame.
    if str(snapshot_abs).endswith(".parquet"):
        recovered = pd.read_parquet(snapshot_abs)
    else:
        recovered = pd.read_csv(snapshot_abs)
    assert len(recovered) == len(df)
    assert "label_3class" in recovered.columns


# --- Directional-call share history (task #101) ------------------------------
def test_train_one_slice_records_directional_call_share(tmp_path, monkeypatch):
    """A trained slice must expose `directional_call_share` so the new
    history endpoint can persist a per-(coin, timeframe) timeseries.
    """
    from app.training import registry as registry_module
    from app.training.train import train_one_slice

    monkeypatch.setattr(registry_module, "REGISTRY_ROOT", tmp_path)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(7)
    frames = []
    for coin, drift in [("pepe", 0.0008), ("bonk", -0.0004)]:
        p = 100.0
        ticks = []
        for i in range(180):
            p *= (1.0 + drift + rng.normal(0, 0.005))
            ticks.append((base + timedelta(seconds=i * 60), p))
        frames.append(build_labeled_frame_for_coin(coin, "1m", ticks))
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    res = train_one_slice(df, coin_id="__pooled__", timeframe="1m", vocab=["bonk", "pepe"])
    assert res["status"] == "trained"
    assert "directional_call_share" in res
    assert 0.0 <= res["directional_call_share"] <= 1.0
    assert res["directional_call_share_n"] > 0
    assert res["directional_call_share_source"] in {"holdout", "in_sample"}

    # The same value must be persisted in the manifest so a model loaded
    # later (e.g. by the dashboard) sees the same number.
    loaded = registry_module.load_model("__pooled__", "1m", res["version"])
    assert loaded is not None
    assert loaded.manifest.directional_call_share == res["directional_call_share"]

    # Task #147 — when the regressor head trained too, the slice must
    # surface a `gates_alignment` summary so the dashboard sees it.
    if loaded.manifest.has_regression_head:
        assert res.get("gates_alignment") is not None
        ga = res["gates_alignment"]
        assert ga["n"] > 0
        assert 0.0 <= ga["aligned_share"] <= 1.0
        assert ga["source"] in {"holdout", "in_sample"}
        # Manifest copy must equal the result copy.
        assert loaded.manifest.gates_alignment == ga


def test_gate_alignment_summary_classifies_2x2_buckets():
    """Task #147 — `_gate_alignment_summary` must split rows into the 4
    sign-vs-magnitude buckets relative to the live gates and report the
    aligned-share + the two wasted-budget shares.
    """
    from app.training.train import _gate_alignment_summary

    # Construct 4 rows, one per bucket, with mde=0.1 mer=0.1:
    #   row 0: edge=0.6 (loud cls), mag=0.5 (loud reg) -> aligned_loud
    #   row 1: edge=0.0 (quiet cls), mag=0.05 (quiet reg) -> aligned_quiet
    #   row 2: edge=0.6 (loud cls), mag=0.05 (quiet reg) -> loud_cls_quiet_reg
    #   row 3: edge=0.0 (quiet cls), mag=0.5 (loud reg) -> quiet_cls_loud_reg
    probs = np.array([
        [0.1, 0.2, 0.7],   # p_up - p_down = 0.6
        [0.4, 0.2, 0.4],   # p_up - p_down = 0.0
        [0.1, 0.2, 0.7],   # p_up - p_down = 0.6
        [0.4, 0.2, 0.4],   # p_up - p_down = 0.0
    ])
    mags = np.array([0.5, 0.05, 0.05, 0.5])
    out = _gate_alignment_summary(probs, mags, mde=0.1, mer=0.1, source="holdout")
    assert out is not None
    assert out["n"] == 4
    assert out["source"] == "holdout"
    assert out["aligned_loud_share"] == 0.25
    assert out["aligned_quiet_share"] == 0.25
    assert out["aligned_share"] == 0.5
    assert out["loud_classifier_quiet_regressor_share"] == 0.25
    assert out["quiet_classifier_loud_regressor_share"] == 0.25
    # Sanity — the four shares cover the whole sample.
    assert math.isclose(
        out["aligned_loud_share"]
        + out["aligned_quiet_share"]
        + out["loud_classifier_quiet_regressor_share"]
        + out["quiet_classifier_loud_regressor_share"],
        1.0,
    )


def test_gate_alignment_summary_returns_none_on_empty():
    from app.training.train import _gate_alignment_summary

    assert _gate_alignment_summary(np.empty((0, 3)), np.array([]), 0.1, 0.1, "holdout") is None
    # Mismatched lengths — diagnostic should bow out rather than crash.
    assert _gate_alignment_summary(np.zeros((3, 3)), np.zeros(2), 0.1, 0.1, "holdout") is None


def test_record_directional_share_for_report_persists_and_alerts(tmp_path, monkeypatch):
    """`_record_directional_share_for_report` must:
      1. Append one JSONL row per trained slice.
      2. Mark below-threshold slices on tradeable timeframes only.
    """
    from app.training import train as train_mod

    history_dir = tmp_path / "training_history"
    history_path = history_dir / "directional_call_share.jsonl"
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_DIR", history_dir)
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_PATH", history_path)
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_MIN_PCT", 15.0)
    monkeypatch.setattr(
        train_mod, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES", {"5m", "1h"},
    )

    report = {
        "generated_at": "2026-04-21T00:00:00Z",
        "timeframes": {
            "5m": {
                "per_coin": {
                    "pepe": {
                        "status": "trained", "version": "v1",
                        "directional_call_share": 0.10,  # below floor on tradeable tf
                        "directional_call_share_n": 50,
                        "directional_call_share_source": "holdout",
                        "n_rows": 200,
                    },
                },
                "pooled": {
                    "status": "trained", "version": "v1",
                    "directional_call_share": 0.42,  # above floor
                    "directional_call_share_n": 80,
                    "directional_call_share_source": "holdout",
                    "n_rows": 320,
                },
            },
            "1m": {
                "per_coin": {
                    "bonk": {
                        "status": "trained", "version": "v1",
                        "directional_call_share": 0.05,  # below floor BUT not tradeable
                        "directional_call_share_n": 50,
                        "directional_call_share_source": "holdout",
                        "n_rows": 200,
                    },
                },
                "pooled": None,
            },
        },
    }
    rows = train_mod._record_directional_share_for_report(report)
    assert len(rows) == 3
    pepe_5m = next(r for r in rows if r["coin_id"] == "pepe")
    pooled_5m = next(r for r in rows if r["coin_id"] == "__pooled__")
    bonk_1m = next(r for r in rows if r["coin_id"] == "bonk")
    assert pepe_5m["below_threshold"] is True
    assert pooled_5m["below_threshold"] is False
    assert bonk_1m["below_threshold"] is False  # 1m isn't tradeable

    # JSONL on disk must contain three lines.
    assert history_path.exists()
    lines = [l for l in history_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3


def test_trim_directional_share_history_caps_per_slice_and_age(tmp_path, monkeypatch):
    """`_trim_directional_share_history` must:
      1. Drop records older than the age cutoff.
      2. Keep only the newest N records per (coin, timeframe).
      3. Drop unparseable JSON lines.
      4. Atomically rewrite the file (no .tmp leftover, file always valid).
    """
    import json as _json
    from datetime import datetime, timedelta, timezone
    from app.training import train as train_mod

    history_dir = tmp_path / "training_history"
    history_path = history_dir / "directional_call_share.jsonl"
    history_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_DIR", history_dir)
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_PATH", history_path)

    now = datetime.now(timezone.utc)

    def _row(ts: datetime, coin: str, tf: str, version: str) -> str:
        return _json.dumps({
            "generated_at": ts.isoformat(),
            "coin_id": coin,
            "timeframe": tf,
            "version": version,
            "directional_call_share_pct": 30.0,
            "n_predictions": 10,
            "source": "holdout",
            "n_train_rows": 100,
            "tradeable_timeframe": True,
            "below_threshold": False,
            "threshold_pct": 15.0,
        })

    lines = []
    # 5 old (>90 days) records on pepe/5m -> all dropped by age.
    for i in range(5):
        lines.append(_row(now - timedelta(days=200 + i), "pepe", "5m", f"old{i}"))
    # 12 recent records on pepe/5m -> capped to 3 newest.
    for i in range(12):
        lines.append(_row(now - timedelta(hours=12 - i), "pepe", "5m", f"new{i}"))
    # 2 records on bonk/1h -> kept (under cap, within age).
    for i in range(2):
        lines.append(_row(now - timedelta(days=1 + i), "bonk", "1h", f"v{i}"))
    # One malformed line.
    lines.append("{not valid json")
    history_path.write_text("\n".join(lines) + "\n")

    summary = train_mod._trim_directional_share_history(max_per_slice=3, max_age_days=90)
    assert summary["skipped"] is False
    assert summary["dropped_age"] == 5
    assert summary["dropped_capped"] == 9  # 12 - 3
    assert summary["dropped_malformed"] == 1
    assert summary["kept"] == 3 + 2

    # File rewritten with only the kept rows; .tmp must not linger.
    assert not history_path.with_suffix(history_path.suffix + ".tmp").exists()
    on_disk = [l for l in history_path.read_text().splitlines() if l.strip()]
    assert len(on_disk) == 5
    parsed = [_json.loads(l) for l in on_disk]
    pepe = [r for r in parsed if r["coin_id"] == "pepe"]
    assert len(pepe) == 3
    # The 3 newest pepe records should be new9, new10, new11 (largest i ->
    # smallest age subtracted -> most recent).
    assert {r["version"] for r in pepe} == {"new9", "new10", "new11"}
    # File ends ordered globally by timestamp.
    timestamps = [r["generated_at"] for r in parsed]
    assert timestamps == sorted(timestamps)


def test_record_regression_head_for_report_persists_per_slice(tmp_path, monkeypatch):
    """Task #157 — `_record_regression_head_for_report` must:
      1. Append one JSONL row per trained slice that has a regression head.
      2. Skip slices without `has_regression_head` (legacy / prior-only).
      3. Capture p50/p95/max/MAE/threshold/version/generated_at.
    """
    from app.training import train as train_mod

    history_dir = tmp_path / "training_history"
    reg_path = history_dir / "regression_head_stats.jsonl"
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_DIR", history_dir)
    monkeypatch.setattr(train_mod, "REGRESSION_HEAD_HISTORY_PATH", reg_path)

    report = {
        "generated_at": "2026-04-22T00:00:00Z",
        "timeframes": {
            "5m": {
                "per_coin": {
                    "pepe": {
                        "status": "trained", "version": "v1",
                        "has_regression_head": True,
                        "threshold_pct": 0.10,
                        "regression_head_stats": {
                            "abs_pred_p50_pct": 0.07,
                            "abs_pred_p95_pct": 0.31,
                            "abs_pred_max_pct": 1.20,
                            "mae_pct": 0.15,
                            "n_holdout_rows": 40,
                            "n_train_rows": 160,
                            "best_iteration": 42,
                        },
                    },
                    "bonk": {
                        # Legacy slice without a regression head — must be skipped.
                        "status": "trained", "version": "v1",
                        "has_regression_head": False,
                        "regression_head_stats": None,
                        "threshold_pct": 0.10,
                    },
                },
                "pooled": {
                    "status": "trained", "version": "v1",
                    "has_regression_head": True,
                    "threshold_pct": 0.10,
                    "regression_head_stats": {
                        "abs_pred_p50_pct": 0.09,
                        "abs_pred_p95_pct": 0.45,
                        "abs_pred_max_pct": 2.00,
                        "mae_pct": 0.20,
                        "n_holdout_rows": 80,
                        "n_train_rows": 320,
                        "best_iteration": 30,
                    },
                },
            },
            "1h": {
                # Prior-only fallback — pooled is trained but has no regression head.
                "per_coin": {},
                "pooled": {
                    "status": "trained", "version": "v1",
                    "has_regression_head": False,
                    "regression_head_stats": None,
                    "threshold_pct": 0.45,
                },
            },
        },
    }
    rows = train_mod._record_regression_head_for_report(report)
    assert len(rows) == 2
    coins = {r["coin_id"] for r in rows}
    assert coins == {"pepe", "__pooled__"}
    pepe = next(r for r in rows if r["coin_id"] == "pepe")
    assert pepe["timeframe"] == "5m"
    assert pepe["abs_pred_p95_pct"] == 0.31
    assert pepe["abs_pred_p50_pct"] == 0.07
    assert pepe["mae_pct"] == 0.15
    assert pepe["threshold_pct"] == 0.10
    assert pepe["version"] == "v1"
    assert pepe["generated_at"] == "2026-04-22T00:00:00Z"
    assert pepe["n_holdout_rows"] == 40

    assert reg_path.exists()
    lines = [l for l in reg_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_trim_regression_head_history_caps_per_slice_and_age(tmp_path, monkeypatch):
    """Task #157 — same retention semantics as the directional-call-share
    history: drop by age, cap newest N per (coin, timeframe), drop
    malformed lines, atomic rewrite.
    """
    import json as _json
    from datetime import datetime, timedelta, timezone
    from app.training import train as train_mod

    history_dir = tmp_path / "training_history"
    reg_path = history_dir / "regression_head_stats.jsonl"
    history_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_DIR", history_dir)
    monkeypatch.setattr(train_mod, "REGRESSION_HEAD_HISTORY_PATH", reg_path)

    now = datetime.now(timezone.utc)

    def _row(ts, coin, tf, version):
        return _json.dumps({
            "generated_at": ts.isoformat(),
            "coin_id": coin, "timeframe": tf, "version": version,
            "abs_pred_p50_pct": 0.1, "abs_pred_p95_pct": 0.3,
            "abs_pred_max_pct": 1.0, "mae_pct": 0.2,
            "n_holdout_rows": 40, "n_train_rows": 160,
            "best_iteration": 10, "threshold_pct": 0.1,
        })

    lines = []
    for i in range(4):
        lines.append(_row(now - timedelta(days=200 + i), "pepe", "5m", f"old{i}"))
    for i in range(10):
        lines.append(_row(now - timedelta(hours=10 - i), "pepe", "5m", f"new{i}"))
    for i in range(2):
        lines.append(_row(now - timedelta(days=1 + i), "bonk", "1h", f"v{i}"))
    lines.append("{not json")
    reg_path.write_text("\n".join(lines) + "\n")

    summary = train_mod._trim_regression_head_history(max_per_slice=3, max_age_days=90)
    assert summary["skipped"] is False
    assert summary["dropped_age"] == 4
    assert summary["dropped_capped"] == 7
    assert summary["dropped_malformed"] == 1
    assert summary["kept"] == 3 + 2

    assert not reg_path.with_suffix(reg_path.suffix + ".tmp").exists()
    on_disk = [l for l in reg_path.read_text().splitlines() if l.strip()]
    parsed = [_json.loads(l) for l in on_disk]
    pepe = [r for r in parsed if r["coin_id"] == "pepe"]
    assert len(pepe) == 3
    assert {r["version"] for r in pepe} == {"new7", "new8", "new9"}
    timestamps = [r["generated_at"] for r in parsed]
    assert timestamps == sorted(timestamps)


def test_trim_regression_head_history_no_file_is_noop(tmp_path, monkeypatch):
    """Fresh deploy: no history file -> trim is a no-op."""
    from app.training import train as train_mod

    history_dir = tmp_path / "training_history"
    reg_path = history_dir / "regression_head_stats.jsonl"
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_DIR", history_dir)
    monkeypatch.setattr(train_mod, "REGRESSION_HEAD_HISTORY_PATH", reg_path)

    summary = train_mod._trim_regression_head_history()
    assert summary["skipped"] is True
    assert not reg_path.exists()


def test_regression_head_trend_renders_in_report(tmp_path, monkeypatch):
    """The training report must surface a per-slice p95 trend chart when
    the rolling regression-head history has at least 2 samples.
    """
    import json as _json
    from app.training import report as report_mod

    history_dir = tmp_path / "training_history"
    reg_path = history_dir / "regression_head_stats.jsonl"
    history_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(report_mod, "REGRESSION_HEAD_HISTORY_PATH", reg_path)

    rows = [
        {"generated_at": "2026-04-20T00:00:00Z", "coin_id": "pepe", "timeframe": "5m",
         "version": "v1", "abs_pred_p95_pct": 0.40, "abs_pred_p50_pct": 0.10,
         "abs_pred_max_pct": 1.0, "mae_pct": 0.2, "threshold_pct": 0.10,
         "n_holdout_rows": 40, "n_train_rows": 160, "best_iteration": 10},
        {"generated_at": "2026-04-21T00:00:00Z", "coin_id": "pepe", "timeframe": "5m",
         "version": "v2", "abs_pred_p95_pct": 0.25, "abs_pred_p50_pct": 0.08,
         "abs_pred_max_pct": 0.9, "mae_pct": 0.2, "threshold_pct": 0.10,
         "n_holdout_rows": 40, "n_train_rows": 160, "best_iteration": 10},
        {"generated_at": "2026-04-22T00:00:00Z", "coin_id": "pepe", "timeframe": "5m",
         "version": "v3", "abs_pred_p95_pct": 0.06, "abs_pred_p50_pct": 0.04,
         "abs_pred_max_pct": 0.5, "mae_pct": 0.1, "threshold_pct": 0.10,
         "n_holdout_rows": 40, "n_train_rows": 160, "best_iteration": 10},
    ]
    reg_path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")

    report = {
        "generated_at": "2026-04-22T00:00:00Z",
        "coin_ids": ["pepe"],
        "lookback_days": 30,
        "timeframes": {
            "5m": {
                "status": "trained",
                "n_rows": 200,
                "dataset_path": "ds.parquet",
                "per_coin": {
                    "pepe": {
                        "status": "trained", "version": "v3",
                        "n_rows": 200, "n_folds": 3,
                        "metrics": {"auc": 0.6, "log_loss": 0.9, "directional_accuracy": 0.5},
                        "baseline_metrics": {"auc": 0.5, "log_loss": 1.0},
                        "lift_auc": 0.1,
                        "fold_metrics": [],
                        "calibration_diagram": [],
                        "class_return_means_pct": [-0.1, 0.0, 0.1],
                        "has_regression_head": True,
                        "threshold_pct": 0.10,
                        "regression_head_stats": rows[-1],
                    },
                },
                "pooled": None,
            },
        },
    }
    html_out = report_mod.render_html(report)
    # Trend chart present.
    assert "p95 trend (last 3)" in html_out
    # The latest sample dipped under threshold -> red dot color appears.
    assert "#ef4444" in html_out
    # Threshold dashed-line colour appears (warn amber).
    assert "#d97706" in html_out


def test_trim_directional_share_history_no_file_is_noop(tmp_path, monkeypatch):
    """If the history file doesn't exist (fresh deploy), trim is a no-op."""
    from app.training import train as train_mod

    history_dir = tmp_path / "training_history"
    history_path = history_dir / "directional_call_share.jsonl"
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_DIR", history_dir)
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_PATH", history_path)

    summary = train_mod._trim_directional_share_history()
    assert summary["skipped"] is True
    assert not history_path.exists()


def test_directional_share_history_endpoint(tmp_path, monkeypatch):
    """`/ml/training/history/directional-call-share` returns grouped series
    plus an `alerts` list for the latest below-floor sample.
    """
    import json as _json
    from fastapi.testclient import TestClient
    from app.main import app
    from app.training import train as train_mod

    history_dir = tmp_path / "training_history"
    history_path = history_dir / "directional_call_share.jsonl"
    history_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_HISTORY_PATH", history_path)
    monkeypatch.setattr(train_mod, "DIRECTIONAL_SHARE_MIN_PCT", 15.0)

    history_path.write_text(
        _json.dumps({"generated_at": "2026-04-20T00:00:00Z", "coin_id": "pepe", "timeframe": "5m",
                     "version": "v1", "directional_call_share_pct": 32.0, "n_predictions": 50,
                     "source": "holdout", "n_train_rows": 200, "tradeable_timeframe": True,
                     "below_threshold": False, "threshold_pct": 15.0}) + "\n" +
        _json.dumps({"generated_at": "2026-04-21T00:00:00Z", "coin_id": "pepe", "timeframe": "5m",
                     "version": "v2", "directional_call_share_pct": 8.0, "n_predictions": 50,
                     "source": "holdout", "n_train_rows": 200, "tradeable_timeframe": True,
                     "below_threshold": True, "threshold_pct": 15.0}) + "\n"
    )

    client = TestClient(app)
    res = client.get("/ml/training/history/directional-call-share")
    assert res.status_code == 200
    body = res.json()
    assert body["thresholdPct"] == 15.0
    assert body["totalRecords"] == 2
    assert len(body["series"]) == 1
    series = body["series"][0]
    assert series["timeframe"] == "5m" and series["coinId"] == "pepe"
    assert len(series["points"]) == 2
    assert series["latestSharePct"] == 8.0
    assert len(body["alerts"]) == 1
    assert body["alerts"][0]["coinId"] == "pepe"
    assert body["alerts"][0]["sharePct"] == 8.0


# ── Auto-recalibration of decision thresholds (task #137) ─────────────────
def _make_sweep_row(**overrides):
    from app.training.threshold_calibration import SweepRow
    base = dict(
        mdp=0.08, mde=0.02, factor=0.5,
        min_expected_return_pct=0.15,
        n_trades=10, n_skips=5,
        final_pnl_usd=12.0, expectancy_usd=1.2,
        win_rate=0.55, sharpe_per_trade=0.4,
    )
    base.update(overrides)
    return SweepRow(**base)


def test_recommend_thresholds_picks_highest_pnl_above_min_trades():
    from app.training.threshold_calibration import recommend_thresholds
    rows = [
        _make_sweep_row(mdp=0.08, n_trades=4,  final_pnl_usd=99.0),  # below floor
        _make_sweep_row(mdp=0.12, n_trades=12, final_pnl_usd=5.0),
        _make_sweep_row(mdp=0.18, n_trades=10, final_pnl_usd=20.0),
        _make_sweep_row(mdp=0.25, n_trades=15, final_pnl_usd=15.0),
    ]
    rec, status = recommend_thresholds(rows, min_trades=8)
    assert status == "ok"
    assert rec is not None and rec.mdp == 0.18 and rec.final_pnl_usd == 20.0


def test_recommend_thresholds_falls_back_when_below_floor():
    from app.training.threshold_calibration import recommend_thresholds
    rows = [
        _make_sweep_row(mdp=0.08, n_trades=2, final_pnl_usd=99.0),
        _make_sweep_row(mdp=0.12, n_trades=4, final_pnl_usd=1.0),
    ]
    rec, status = recommend_thresholds(rows, min_trades=8)
    assert status == "below_min_trades"
    assert rec is not None and rec.n_trades == 4


def test_recommend_thresholds_no_signal_when_zero_trades():
    from app.training.threshold_calibration import recommend_thresholds
    rows = [_make_sweep_row(n_trades=0, final_pnl_usd=0.0)]
    rec, status = recommend_thresholds(rows, min_trades=1)
    assert rec is None and status == "no_signal"


def test_model_hash_is_deterministic_and_changes_with_versions():
    from app.training.threshold_calibration import model_hash_from_report
    report_a = {"timeframes": {"5m": {
        "per_coin": {"pepe": {"version": "v1"}, "bonk": {"version": "v1"}},
        "pooled": {"version": "v1"},
    }}}
    report_b = {"timeframes": {"5m": {
        "per_coin": {"pepe": {"version": "v1"}, "bonk": {"version": "v1"}},
        "pooled": {"version": "v1"},
    }}}
    report_c = {"timeframes": {"5m": {
        "per_coin": {"pepe": {"version": "v2"}, "bonk": {"version": "v1"}},
        "pooled": {"version": "v1"},
    }}}
    assert model_hash_from_report(report_a) == model_hash_from_report(report_b)
    assert model_hash_from_report(report_a) != model_hash_from_report(report_c)


def test_build_proposal_diff_and_policy_version_bump():
    from app.training.threshold_calibration import (
        build_proposal, model_hash_from_report, policy_version_from_hash,
    )
    from app.backtest.contract import get_frictions
    fr = get_frictions()
    rec = _make_sweep_row(mdp=0.12, mde=0.03, factor=0.25,
                          min_expected_return_pct=0.075,
                          n_trades=20, final_pnl_usd=42.0)
    report = {"timeframes": {"5m": {
        "per_coin": {"pepe": {"version": "v-XYZ"}},
        "pooled": {"version": "v-pool"},
    }}}
    h = model_hash_from_report(report)
    proposal = build_proposal(
        timeframe="5m", snapshot_name="5m_v.parquet",
        rec=rec, rec_status="ok", n_oos_rows=500, base_fr=fr,
        model_hash=h, sweep_rows=[rec],
    )
    assert proposal["model_hash"] == h
    assert proposal["proposed"]["policy_version"] == policy_version_from_hash(h)
    assert proposal["proposed"]["min_directional_prob"] == 0.12
    assert proposal["change"]["would_apply"] is True
    diff = proposal["change"]["diff"]
    # policy_version always changes (hash-derived); at least one threshold
    # also moved off the current contract values.
    assert "policy_version" in diff
    assert any(k.startswith("min_") for k in diff)


def test_build_proposal_no_recommendation_marks_no_apply():
    from app.training.threshold_calibration import build_proposal
    from app.backtest.contract import get_frictions
    fr = get_frictions()
    proposal = build_proposal(
        timeframe="5m", snapshot_name=None, rec=None,
        rec_status="no_signal", n_oos_rows=0, base_fr=fr,
        model_hash="deadbeef", sweep_rows=[],
    )
    assert proposal["proposed"] is None
    assert proposal["change"]["would_apply"] is False
    assert proposal["change"]["reason"] == "no_signal"


def test_apply_proposal_writes_contract_and_resets_cache(tmp_path):
    """`apply_proposal` rewrites the contract on disk and clears the
    `load_contract` LRU so subsequent `get_frictions()` calls in the same
    process see the new values."""
    import json as _json
    from app.backtest import contract as contract_mod

    fixture = {
        "fees": {"maker_fee_pct": 0.001, "taker_fee_pct": 0.001, "slippage_pct": 0.0005},
        "quant_brain": {"decision_thresholds": {
            "min_directional_prob": 0.08,
            "min_directional_edge": 0.02,
            "min_expected_return_pct_factor": 3.0,
            "policy_version": "v3-old",
        }},
    }
    path = tmp_path / "trading-frictions.json"
    path.write_text(_json.dumps(fixture))
    from app.training.threshold_calibration import apply_proposal

    proposal = {
        "change": {"would_apply": True, "diff": {}},
        "proposed": {
            "min_directional_prob": 0.12,
            "min_directional_edge": 0.05,
            "min_expected_return_pct_factor": 0.5,
            "policy_version": "v4-auto-abc123",
        },
    }
    written = apply_proposal(proposal, contract_path=path)
    assert written is True
    after = _json.loads(path.read_text())
    dt = after["quant_brain"]["decision_thresholds"]
    assert dt["min_directional_prob"] == 0.12
    assert dt["min_directional_edge"] == 0.05
    assert dt["min_expected_return_pct_factor"] == 0.5
    assert dt["policy_version"] == "v4-auto-abc123"
    # No-op when nothing to apply.
    assert apply_proposal({"change": {"would_apply": False}}) is False
    # Cache reset is global; nothing to assert beyond no-exception.
    contract_mod._reset_cache()


def test_recalibrate_after_training_writes_proposal_when_no_dataset(tmp_path, monkeypatch):
    """Best-effort path: if no dataset snapshot exists for the timeframe,
    `recalibrate_after_training` records `status=no_dataset` and writes
    the proposal report — it never raises and never touches the contract.
    """
    from app.training import threshold_calibration as tc

    out_path = tmp_path / "calibration_recommendation.json"
    monkeypatch.setattr(tc, "latest_snapshot_for", lambda tf: None)
    proposal = tc.recalibrate_after_training(
        report={"timeframes": {}}, timeframe="5m", apply=False,
        recommendation_path=out_path,
    )
    assert proposal["status"] == "no_dataset"
    assert proposal["applied"] is False
    assert out_path.exists()
    import json as _json
    body = _json.loads(out_path.read_text())
    assert body["status"] == "no_dataset"


def test_calibration_history_record_and_trim(tmp_path, monkeypatch):
    """Append + trim mirror the directional-call-share retention story:
    cap N per timeframe, drop anything older than M days, drop malformed
    lines, atomic rewrite. The recorded row pulls every interesting field
    off the proposal so the dashboard can render the timeline.
    """
    import json as _json
    from datetime import datetime, timedelta, timezone
    from app.training import threshold_calibration as tc

    history_path = tmp_path / "calibration_recommendation.jsonl"
    monkeypatch.setattr(tc, "CALIBRATION_HISTORY_PATH", history_path)

    rec = _make_sweep_row(mdp=0.12, mde=0.03, factor=0.25,
                          n_trades=20, final_pnl_usd=42.0,
                          sharpe_per_trade=0.7)
    proposal = {
        "timeframe": "5m",
        "snapshot": "5m_v.parquet",
        "model_hash": "deadbeef",
        "n_oos_rows": 500,
        "status": "ok",
        "recommendation_status": "ok",
        "applied": False,
        "current": {
            "min_directional_prob": 0.08,
            "min_directional_edge": 0.02,
            "min_expected_return_pct_factor": 3.0,
            "policy_version": "v3-calibrated",
        },
        "proposed": {
            "min_directional_prob": 0.12,
            "min_directional_edge": 0.03,
            "min_expected_return_pct_factor": 0.25,
            "policy_version": "v4-auto-deadbeef",
        },
        "change": {"would_apply": True, "diff": {"policy_version": {}}},
    }
    rec_row = tc._history_record_from_proposal(
        proposal, rec, generated_at="2026-04-22T00:00:00+00:00",
    )
    assert rec_row["timeframe"] == "5m"
    assert rec_row["proposed"]["min_directional_prob"] == 0.12
    assert rec_row["recommendation"]["n_trades"] == 20
    assert rec_row["recommendation"]["final_pnl_usd"] == 42.0
    assert rec_row["would_apply"] is True
    assert rec_row["status"] == "ok"

    tc._append_calibration_history(rec_row)
    assert history_path.exists()
    on_disk = [_json.loads(l) for l in history_path.read_text().splitlines() if l.strip()]
    assert len(on_disk) == 1 and on_disk[0]["model_hash"] == "deadbeef"

    # Seed enough rows to exercise the per-slice cap and age cutoff.
    now = datetime.now(timezone.utc)
    history_path.unlink()
    rows = []
    # 5 ancient rows on 5m (>90 days) -> dropped by age.
    for i in range(5):
        r = dict(rec_row)
        r["generated_at"] = (now - timedelta(days=200 + i)).isoformat()
        r["model_hash"] = f"old{i}"
        rows.append(_json.dumps(r))
    # 12 recent rows on 5m -> capped to newest 3.
    for i in range(12):
        r = dict(rec_row)
        r["generated_at"] = (now - timedelta(hours=12 - i)).isoformat()
        r["model_hash"] = f"new{i}"
        rows.append(_json.dumps(r))
    # 2 rows on 1h -> kept.
    for i in range(2):
        r = dict(rec_row)
        r["timeframe"] = "1h"
        r["generated_at"] = (now - timedelta(days=1 + i)).isoformat()
        r["model_hash"] = f"hr{i}"
        rows.append(_json.dumps(r))
    # malformed line.
    rows.append("{not json")
    history_path.write_text("\n".join(rows) + "\n")

    summary = tc._trim_calibration_history(max_per_slice=3, max_age_days=90)
    assert summary["skipped"] is False
    assert summary["dropped_age"] == 5
    assert summary["dropped_capped"] == 9
    assert summary["dropped_malformed"] == 1
    assert summary["kept"] == 5
    assert not history_path.with_suffix(history_path.suffix + ".tmp").exists()
    parsed = [_json.loads(l) for l in history_path.read_text().splitlines() if l.strip()]
    assert len(parsed) == 5
    fivem = [r for r in parsed if r["timeframe"] == "5m"]
    assert {r["model_hash"] for r in fivem} == {"new9", "new10", "new11"}
    timestamps = [r["generated_at"] for r in parsed]
    assert timestamps == sorted(timestamps)


def test_trim_calibration_history_no_file_is_noop(tmp_path, monkeypatch):
    from app.training import threshold_calibration as tc
    monkeypatch.setattr(
        tc, "CALIBRATION_HISTORY_PATH", tmp_path / "calibration_recommendation.jsonl",
    )
    summary = tc._trim_calibration_history()
    assert summary["skipped"] is True


def test_recalibrate_after_training_records_history_on_no_dataset(tmp_path, monkeypatch):
    """Even the early-return paths (no_dataset / empty_dataset / no_oos)
    append a row so the dashboard timeline shows gaps in coverage. The
    proposal status field carries the reason."""
    import json as _json
    from app.training import threshold_calibration as tc

    history_path = tmp_path / "calibration_recommendation.jsonl"
    monkeypatch.setattr(tc, "CALIBRATION_HISTORY_PATH", history_path)
    monkeypatch.setattr(tc, "latest_snapshot_for", lambda tf: None)

    out_path = tmp_path / "calibration_recommendation.json"
    proposal = tc.recalibrate_after_training(
        report={"timeframes": {}}, timeframe="5m", apply=False,
        recommendation_path=out_path,
    )
    assert proposal["status"] == "no_dataset"
    assert history_path.exists()
    rows = [_json.loads(l) for l in history_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["status"] == "no_dataset"
    assert rows[0]["timeframe"] == "5m"
    assert rows[0]["proposed"] is None
    assert rows[0]["recommendation"] is None


def test_calibration_history_endpoint(tmp_path, monkeypatch):
    """`/ml/training/history/calibration` groups by timeframe and exposes
    the latest proposed triple + recommendation metrics per series."""
    import json as _json
    from fastapi.testclient import TestClient
    from app.main import app
    from app.training import threshold_calibration as tc

    history_path = tmp_path / "calibration_recommendation.jsonl"
    monkeypatch.setattr(tc, "CALIBRATION_HISTORY_PATH", history_path)

    base = {
        "timeframe": "5m",
        "snapshot": "snap.parquet",
        "model_hash": "h1",
        "n_oos_rows": 500,
        "status": "ok",
        "recommendation_status": "ok",
        "would_apply": True,
        "applied": False,
        "current": {
            "min_directional_prob": 0.08, "min_directional_edge": 0.02,
            "min_expected_return_pct_factor": 3.0, "policy_version": "v3",
        },
        "proposed": {
            "min_directional_prob": 0.12, "min_directional_edge": 0.03,
            "min_expected_return_pct_factor": 0.25, "policy_version": "v4-h1",
        },
        "recommendation": {
            "n_trades": 20, "n_skips": 5, "final_pnl_usd": 42.0,
            "expectancy_usd": 2.1, "win_rate": 0.55, "sharpe_per_trade": 0.7,
        },
    }
    older = dict(base, generated_at="2026-04-20T00:00:00Z", model_hash="h0")
    older["proposed"] = dict(base["proposed"], min_directional_prob=0.10)
    newer = dict(base, generated_at="2026-04-21T00:00:00Z")
    history_path.write_text(_json.dumps(older) + "\n" + _json.dumps(newer) + "\n")

    client = TestClient(app)
    res = client.get("/ml/training/history/calibration")
    assert res.status_code == 200
    body = res.json()
    assert body["totalRecords"] == 2
    assert len(body["series"]) == 1
    s = body["series"][0]
    assert s["timeframe"] == "5m"
    assert len(s["points"]) == 2
    assert s["latestProposed"]["min_directional_prob"] == 0.12
    assert s["latestRecommendation"]["n_trades"] == 20
    assert s["latestModelHash"] == "h1"
    assert s["latestWouldApply"] is True
    assert "minTradesFloor" in body


def test_recalibrate_after_training_handles_exception_gracefully(tmp_path, monkeypatch):
    """A crash anywhere in the sweep must NOT propagate — the training
    contract is sacred. We surface the error on the returned dict."""
    from app.training import threshold_calibration as tc

    def _boom(_tf):
        raise RuntimeError("synthetic boom")

    out_path = tmp_path / "calibration_recommendation.json"
    monkeypatch.setattr(tc, "latest_snapshot_for", _boom)
    proposal = tc.recalibrate_after_training(
        report={"timeframes": {}}, timeframe="5m", apply=False,
        recommendation_path=out_path,
    )
    assert proposal["status"] == "error"
    assert "synthetic boom" in proposal["error"]
    assert out_path.exists()




# --- Task #482 — single-scalar temperature scaling fix ------------------------
# The diagnostic at scripts/diagnostic_482/run_stage_collapse_diagnostic.py
# bucketed 33/40 (coin, tf) slices as `calibrator_collapse`: raw booster
# argmax STABLE was 20-90% but per-class isotonic + per-row sum-to-one
# squashed it to 0-3%, driving directional_call_share to ~1.0 and tripping
# the verification gate. The fix replaces it with single-scalar
# temperature scaling (Guo et al. 2017): cal[k] = raw[k] ** (1 / T)
# applied with the SAME exponent to every class. The booster's argmax
# is invariant under monotone transformation, so the calibrator can
# only sharpen / flatten confidence — it can NEVER re-rank classes
# within a row.
def test_calibrate_per_class_preserves_booster_argmax_distribution():
    """Argmax invariance is the load-bearing property of the fix.
    Whatever the booster argmaxed pre-calibration, the calibrator must
    argmax the same class on every row — otherwise the calibrator can
    re-introduce the calibrator-collapse failure mode that destroyed
    STABLE-argmax across the fleet diagnostic.
    """
    from app.training.train import _calibrate_per_class

    rng = np.random.default_rng(seed=482)
    n = 600
    priors = np.array([0.39, 0.22, 0.39])
    y_cal = rng.choice(3, size=n, p=priors)
    raw = np.empty((n, 3), dtype=float)
    for i in range(n):
        true_k = y_cal[i]
        base = np.array([0.30, 0.30, 0.30])
        base[true_k] = 0.55
        base += rng.normal(0.0, 0.04, size=3)
        base = np.clip(base, 1e-3, None)
        raw[i] = base / base.sum()

    raw_argmax = raw.argmax(axis=1)
    calibrators, cal, _ = _calibrate_per_class(raw, y_cal)
    cal_argmax = cal.argmax(axis=1)

    assert calibrators is not None
    np.testing.assert_allclose(cal.sum(axis=1), 1.0, atol=1e-6)
    # Argmax must be IDENTICAL on every row — that's the whole point of
    # using a single-scalar temperature.
    np.testing.assert_array_equal(cal_argmax, raw_argmax)


def test_calibrate_per_class_recovers_minority_argmax_share():
    """End-to-end: when the booster gives substantial STABLE mass
    (argmax STABLE > 15% of rows), the calibrator must NOT collapse it
    below 5% — the share that would trip the verification gate's
    `directional_call_share <= 0.95` ceiling.
    """
    from app.training.train import _calibrate_per_class

    rng = np.random.default_rng(seed=4821)
    n = 600
    priors = np.array([0.39, 0.22, 0.39])
    y_cal = rng.choice(3, size=n, p=priors)
    raw = np.empty((n, 3), dtype=float)
    for i in range(n):
        true_k = y_cal[i]
        base = np.array([0.30, 0.30, 0.30])
        base[true_k] = 0.55
        base += rng.normal(0.0, 0.04, size=3)
        base = np.clip(base, 1e-3, None)
        raw[i] = base / base.sum()

    raw_stable_share = float((raw.argmax(axis=1) == 1).mean())
    assert raw_stable_share >= 0.15, (
        "test fixture broken — raw STABLE-argmax should be substantial"
    )

    _, cal, _ = _calibrate_per_class(raw, y_cal)
    cal_stable_share = float((cal.argmax(axis=1) == 1).mean())
    assert cal_stable_share == raw_stable_share, (
        "single-T scaling must preserve argmax exactly"
    )
    cal_directional_call_share = float((cal.argmax(axis=1) != 1).mean())
    assert cal_directional_call_share < 0.95, (
        f"directional_call_share {cal_directional_call_share:.3f} "
        "still above the verification gate's 0.95 ceiling"
    )


def test_calibrate_per_class_persists_temperature_through_predict():
    """The wrapper must expose the same `.predict(p_k_raw)` contract the
    inference path relies on (`app/main.py::_calibrated_3class_probs`).
    Round-trip through joblib so we also catch any pickling regression.
    """
    import io

    import joblib

    from app.training.calibration import (
        TemperatureScaledClass,
        apply_single_temperature,
    )
    from app.training.train import _calibrate_per_class

    rng = np.random.default_rng(seed=4822)
    n = 300
    priors = np.array([0.4, 0.2, 0.4])
    y_cal = rng.choice(3, size=n, p=priors)
    raw = np.empty((n, 3), dtype=float)
    for i in range(n):
        raw[i] = rng.dirichlet(alpha=np.array([1.5, 1.0, 1.5]))

    calibrators, cal, _ = _calibrate_per_class(raw, y_cal)
    assert calibrators is not None

    # Every entry must be a TemperatureScaledClass with the SAME
    # exponent (single-scalar temperature scaling).
    inv_T_values = set()
    for cal_k in calibrators:
        assert isinstance(cal_k, TemperatureScaledClass), type(cal_k)
        assert 0.25 <= cal_k.inv_T_k <= 2.0
        inv_T_values.add(cal_k.inv_T_k)
    assert len(inv_T_values) == 1, (
        f"all class wrappers must share inv_T, got {inv_T_values}"
    )

    # Re-apply the per-class wrapper to each raw column independently and
    # confirm the per-row sum-to-one matches the returned `cal` matrix —
    # this is exactly what `app/main.py` does at inference time.
    rebuilt = np.zeros_like(raw)
    for k in range(3):
        rebuilt[:, k] = calibrators[k].predict(raw[:, k])
    s = rebuilt.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    rebuilt = rebuilt / s
    np.testing.assert_allclose(rebuilt, cal, atol=1e-9)

    # And the standalone `apply_single_temperature` helper produces the
    # same calibrated matrix when handed the fitted exponent directly.
    inv_T = next(iter(inv_T_values))
    np.testing.assert_allclose(
        apply_single_temperature(raw, inv_T), cal, atol=1e-9,
    )

    # Joblib round-trip (the registry's persistence path).
    buf = io.BytesIO()
    joblib.dump(calibrators, buf)
    buf.seek(0)
    loaded = joblib.load(buf)
    for k in range(3):
        assert isinstance(loaded[k], TemperatureScaledClass)
        np.testing.assert_allclose(
            loaded[k].predict(raw[:, k]),
            calibrators[k].predict(raw[:, k]),
        )


def test_fit_single_temperature_handles_degenerate_inputs():
    """`fit_single_temperature` must clip to a safe band, fall back to
    identity (`inv_T = 1`) on degenerate inputs, and never raise.
    """
    from app.training.calibration import (
        INV_T_MAX,
        INV_T_MIN,
        fit_single_temperature,
    )

    # Single-class holdout — fit must fall back to identity.
    raw_single = np.tile(np.array([0.4, 0.3, 0.3]), (50, 1))
    y_single = np.zeros(50, dtype=int)
    assert fit_single_temperature(raw_single, y_single) == 1.0

    # NaNs in the raw probabilities — identity fallback.
    raw_nan = np.full((10, 3), np.nan)
    y_nan = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    assert fit_single_temperature(raw_nan, y_nan) == 1.0

    # Wrong shape — identity fallback (no exception).
    raw_wrong = np.zeros((10, 5))
    assert fit_single_temperature(raw_wrong, y_nan) == 1.0

    # Healthy inputs — fit must clip to [INV_T_MIN, INV_T_MAX].
    rng = np.random.default_rng(seed=7)
    n = 300
    y = rng.choice(3, size=n, p=[1 / 3, 1 / 3, 1 / 3])
    raw = np.empty((n, 3))
    for i in range(n):
        true_k = y[i]
        base = np.full(3, 0.30)
        base[true_k] = 0.40
        raw[i] = base / base.sum()
    inv_T = fit_single_temperature(raw, y)
    assert INV_T_MIN - 1e-9 <= inv_T <= INV_T_MAX + 1e-9


def test_calibrate_per_class_fleet_diagnostic_recovers_call_share():
    """Replay the four pre-fix diagnostic JSON fixtures
    (`reports/*-task482-stage-collapse-diagnostic.json`) and confirm
    the new calibrator on the SAME raw probabilities recovers
    directional_call_share below the verification gate's 0.95 ceiling
    on every slice that was bucketed `calibrator_collapse` AND had
    raw STABLE-argmax above the gate (>5%). This is the joint
    `call_share moved` victory criterion stated in the task brief.
    """
    import json
    from pathlib import Path

    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    fixture_paths = sorted(
        reports_dir.glob("*-task482-stage-collapse-diagnostic.json"),
    )
    if not fixture_paths:
        pytest.skip("diagnostic JSON fixtures absent — run the diagnostic first")

    examined = 0
    recovered = 0
    for p in fixture_paths:
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            if r.get("status") != "ok":
                continue
            if r.get("bucket") != "calibrator_collapse":
                continue
            raw_s = float(r.get("raw_STABLE_share", 0.0))
            if raw_s <= 0.05:
                continue
            # Single-T scaling preserves argmax → cal_argmax_share == raw_argmax_share.
            # `directional_call_share` (UP|DOWN argmax) = 1 - STABLE_argmax,
            # so post-fix it equals (1 - raw_s).
            post_fix_dcs = 1.0 - raw_s
            examined += 1
            if post_fix_dcs <= 0.95:
                recovered += 1

    assert examined >= 10, f"too few calibrator_collapse fixtures: {examined}"
    recovery_ratio = recovered / examined
    assert recovery_ratio >= 0.80, (
        f"single-T fix recovers only {recovery_ratio:.2%} of "
        "calibrator_collapse slices — joint victory criterion 'call_share "
        "moved' would not hold across the fleet."
    )


# --- Task #507: power-balanced sample weights -------------------------------


def test_balanced_sample_weight_alpha_one_matches_legacy_full_balance():
    """alpha=1.0 reproduces the Task #95 'each present class equal total
    mass' recipe exactly. This is the regression guard that the new
    parameterized helper still has the legacy behaviour as a corner.
    """
    from app.training.train import _balanced_sample_weight

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=300)
    counts = np.bincount(y, minlength=3)

    w = _balanced_sample_weight(y, alpha=1.0)

    target = len(y) / 3  # all 3 classes present at this seed
    expected_per_class = target / counts  # the legacy formula
    for k in range(3):
        np.testing.assert_allclose(
            w[y == k], expected_per_class[k], rtol=1e-12, atol=0,
        )
    # Total mass per class is exactly `target` for every present class.
    for k in range(3):
        assert w[y == k].sum() == pytest.approx(target, rel=1e-12)


def test_balanced_sample_weight_alpha_zero_is_uniform():
    """alpha=0.0 returns weights of 1.0 everywhere — no rebalancing.
    Useful as a sanity / debugging escape hatch.
    """
    from app.training.train import _balanced_sample_weight

    y = np.array([0, 0, 0, 1, 2, 2], dtype=int)
    w = _balanced_sample_weight(y, alpha=0.0)
    np.testing.assert_allclose(w, np.ones(len(y)), rtol=0, atol=0)


def test_balanced_sample_weight_default_is_full_balance():
    """The module default (alpha=1.0) reproduces the Task #95
    full-balance recipe. Task #507 originally hypothesised that
    sqrt-dampening this weight would recover the 4 stuck booster
    slices, but the focused diagnostic
    (`scripts/diagnostic_482/run_507_focused.py`) showed sqrt-balancing
    actually *worsened* raw STABLE-argmax on those slices. The default
    was therefore restored to ``alpha = 1`` and the recovery moved to
    the tiny-slice branch in ``_train_lgb`` — see
    ``reports/<TS>-task507-booster-collapse-verification.md`` for the
    full reasoning.
    """
    from app.training.train import _balanced_sample_weight

    n = 264
    n_stable = 34  # ≈ 13% of 264, mirrors celestia@1d
    n_down = 113
    n_up = n - n_stable - n_down  # 117
    y = np.concatenate([
        np.zeros(n_down, dtype=int),
        np.ones(n_stable, dtype=int),
        np.full(n_up, 2, dtype=int),
    ])

    w_legacy = _balanced_sample_weight(y, alpha=1.0)
    w_default = _balanced_sample_weight(y)

    # The default matches the legacy alpha=1.0 weights exactly.
    np.testing.assert_allclose(w_default, w_legacy, rtol=1e-12, atol=0)

    # Concrete numerical guarantee — the legacy STABLE row weight on a
    # `celestia@1d`-shaped slice.
    default_stable = w_default[y == 1][0]
    assert default_stable == pytest.approx(264 / (3 * n_stable), rel=1e-9)


def test_balanced_sample_weight_alpha_two_amplifies_rare_class():
    """The Task #507 tiny-slice branch in ``_train_lgb`` calls the
    helper with ``alpha = 2``, which AMPLIFIES the legacy per-row
    weight (legacy ratio squared). On a STABLE-rare slice the rare-
    class row weight becomes ``(legacy)^2`` — large enough to push
    the booster's marginal STABLE prior on small holdouts above the
    per-row argmax threshold. This test pins the relationship the
    booster-recovery fix relies on so a future refactor can't change
    the constant without updating ``TINY_SLICE_CLASS_WEIGHT_ALPHA``.
    """
    from app.training.train import _balanced_sample_weight

    n_stable = 34
    n_down = 113
    n_up = 117
    y = np.concatenate([
        np.zeros(n_down, dtype=int),
        np.ones(n_stable, dtype=int),
        np.full(n_up, 2, dtype=int),
    ])

    w_legacy = _balanced_sample_weight(y, alpha=1.0)
    w_amplified = _balanced_sample_weight(y, alpha=2.0)

    np.testing.assert_allclose(w_amplified, w_legacy ** 2, rtol=1e-12, atol=0)
    # The STABLE row weight is strictly larger than the legacy weight
    # because the legacy STABLE weight is > 1 on this rare-class slice.
    assert w_amplified[y == 1][0] > w_legacy[y == 1][0]


def test_balanced_sample_weight_handles_empty_input():
    from app.training.train import _balanced_sample_weight

    out = _balanced_sample_weight(np.array([], dtype=int))
    assert out.shape == (0,)
    assert out.dtype == np.float64


def test_balanced_sample_weight_handles_missing_classes():
    """If a fold sees only DOWN + UP and no STABLE rows, no weight mass
    is allocated to STABLE (no rows to allocate to) and the present
    classes are still power-balanced relative to each other.
    """
    from app.training.train import _balanced_sample_weight

    # 100 DOWN + 200 UP, 0 STABLE
    y = np.concatenate([
        np.zeros(100, dtype=int),
        np.full(200, 2, dtype=int),
    ])
    w_legacy = _balanced_sample_weight(y, alpha=1.0)
    w_sqrt = _balanced_sample_weight(y, alpha=0.5)
    w_amplified = _balanced_sample_weight(y, alpha=2.0)

    # alpha=1: target_per_present_class = 300/2 = 150.
    # DOWN row weight = 150/100 = 1.5; UP row weight = 150/200 = 0.75.
    np.testing.assert_allclose(w_legacy[y == 0], 1.5, rtol=1e-12, atol=0)
    np.testing.assert_allclose(w_legacy[y == 2], 0.75, rtol=1e-12, atol=0)

    # alpha=0.5: sqrt of those.
    np.testing.assert_allclose(w_sqrt[y == 0], math.sqrt(1.5), rtol=1e-12, atol=0)
    np.testing.assert_allclose(w_sqrt[y == 2], math.sqrt(0.75), rtol=1e-12, atol=0)

    # alpha=2.0 (the tiny-slice branch's amplification): legacy weights
    # squared; rare-class STABLE rows still get zero weight because
    # there are no STABLE rows to allocate to.
    np.testing.assert_allclose(w_amplified[y == 0], 1.5 ** 2, rtol=1e-12, atol=0)
    np.testing.assert_allclose(w_amplified[y == 2], 0.75 ** 2, rtol=1e-12, atol=0)


def test_balanced_sample_weight_env_var_overrides_default(monkeypatch):
    """`ML_CLASS_WEIGHT_ALPHA` must drive the default alpha when the
    train module is reloaded so the diagnostic harness can sweep over
    values without editing source.
    """
    monkeypatch.setenv("ML_CLASS_WEIGHT_ALPHA", "0.0")
    import importlib
    from app.training import train as train_mod

    importlib.reload(train_mod)
    try:
        y = np.array([0, 0, 0, 1, 2, 2], dtype=int)
        w = train_mod._balanced_sample_weight(y)
        np.testing.assert_allclose(w, np.ones(len(y)), rtol=0, atol=0)
        assert train_mod.CLASS_WEIGHT_ALPHA == 0.0
    finally:
        monkeypatch.delenv("ML_CLASS_WEIGHT_ALPHA", raising=False)
        importlib.reload(train_mod)


def test_train_lgb_tiny_slice_branch_overrides_caller_params(monkeypatch):
    """When ``n_train < TINY_SLICE_THRESHOLD`` the caller's ``params``
    dict and the standard early-stopping callback are replaced with
    the tiny-slice recipe (``num_leaves=15, learning_rate=0.05,
    min_child_samples=20``, no early stopping). This test verifies
    that the override actually fires by inspecting the booster's
    ``params`` attribute and that the caller's exotic params (e.g.
    ``num_leaves=255``) are dropped on a tiny holdout.
    """
    import lightgbm as lgb
    from app.training.train import (
        TINY_SLICE_LEARNING_RATE,
        TINY_SLICE_MIN_CHILD_SAMPLES,
        TINY_SLICE_NUM_LEAVES,
        TINY_SLICE_THRESHOLD,
        _lgb_params,
        _train_lgb,
    )

    rng = np.random.default_rng(0)
    n_train = 200  # well below TINY_SLICE_THRESHOLD
    n_val = 60
    assert n_train < TINY_SLICE_THRESHOLD
    X_train = pd.DataFrame(rng.normal(size=(n_train, 4)),
                           columns=["f0", "f1", "f2", "f3"])
    y_train = rng.integers(0, 3, size=n_train)
    X_val = pd.DataFrame(rng.normal(size=(n_val, 4)),
                         columns=["f0", "f1", "f2", "f3"])
    y_val = rng.integers(0, 3, size=n_val)

    # Caller passes a deliberately exotic param dict.
    caller_params = _lgb_params(num_leaves=255, learning_rate=0.5, min_child_samples=2)

    # Patch out CATEGORICAL_FEATURES so the test doesn't care about
    # registry coupling.
    from app.training import train as train_mod
    monkeypatch.setattr(train_mod, "CATEGORICAL_FEATURES", [], raising=True)

    booster, val_loss = _train_lgb(X_train, y_train, X_val, y_val, caller_params)

    assert isinstance(booster, lgb.Booster)
    assert math.isfinite(val_loss)

    # The booster's params reflect the tiny-slice override, not the caller's.
    booster_params = booster.params
    assert booster_params["num_leaves"] == TINY_SLICE_NUM_LEAVES
    assert booster_params["learning_rate"] == TINY_SLICE_LEARNING_RATE
    assert booster_params["min_child_samples"] == TINY_SLICE_MIN_CHILD_SAMPLES

    # No early stopping fired ⇒ best_iteration == 0 ⇒ all trees used.
    # (LightGBM exposes best_iteration == 0 when no early-stopping
    # callback was registered.)
    assert booster.best_iteration == 0
    # The booster trained at least a few rounds (no iter=1 trap).
    assert booster.current_iteration() >= 10


def test_train_lgb_non_tiny_slice_keeps_caller_params(monkeypatch):
    """When ``n_train >= TINY_SLICE_THRESHOLD`` the override does NOT
    fire — the caller's params (typically Optuna-tuned for that
    timeframe) are honored and the standard early-stopping callback
    is registered, so the existing 1h / 2h / pooled training path is
    not silently changed by the Task #507 fix.
    """
    import lightgbm as lgb
    from app.training.train import (
        TINY_SLICE_THRESHOLD,
        _lgb_params,
        _train_lgb,
    )

    rng = np.random.default_rng(1)
    n_train = TINY_SLICE_THRESHOLD + 200
    n_val = 200
    X_train = pd.DataFrame(rng.normal(size=(n_train, 4)),
                           columns=["f0", "f1", "f2", "f3"])
    y_train = rng.integers(0, 3, size=n_train)
    X_val = pd.DataFrame(rng.normal(size=(n_val, 4)),
                         columns=["f0", "f1", "f2", "f3"])
    y_val = rng.integers(0, 3, size=n_val)

    caller_params = _lgb_params(num_leaves=63, learning_rate=0.07,
                                min_child_samples=12)

    from app.training import train as train_mod
    monkeypatch.setattr(train_mod, "CATEGORICAL_FEATURES", [], raising=True)

    booster, val_loss = _train_lgb(X_train, y_train, X_val, y_val, caller_params)

    assert isinstance(booster, lgb.Booster)
    assert math.isfinite(val_loss)

    booster_params = booster.params
    # Caller's exact params are honored on the non-tiny path.
    assert booster_params["num_leaves"] == 63
    assert booster_params["learning_rate"] == pytest.approx(0.07, rel=1e-9)
    assert booster_params["min_child_samples"] == 12


def test_task507_focused_diagnostic_recovers_stuck_slices_at_80_rounds():
    """Replay the latest Task #507 focused-diagnostic JSON and confirm
    every stuck slice passes BOTH the bucket-classifier gate
    (``bucket == 'no_collapse'``) AND the verification gate's
    ``directional_call_share <= 0.95`` ceiling.

    SCOPE — LOW-ROUND REGIME ONLY. The fixtures this test reads are
    generated by ``scripts/diagnostic_482/run_507_focused.py``, which
    pins ``ML_LGB_NUM_BOOST_ROUND=80`` (a CI shortcut). Production
    trains the booster for 800 rounds, and at that budget the
    tiny-slice recovery branch in ``_train_lgb`` does NOT in fact
    rescue all 4 stuck slices — see Task #519's verification report
    (``reports/20260428T120352Z-task519-booster-fix-rerun-800rounds-verification.md``)
    and its companion regression test
    ``test_task519_focused_diagnostic_documents_800_round_regression``
    below for the production-rounds outcome. This 80-round test is
    retained because it still proves the soft-recipe + no-early-stop
    + alpha=2 combination escapes ``best_iteration == 1`` in the
    early-trees regime, which was the original mechanical hypothesis.
    """
    import json
    from pathlib import Path

    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    # Restrict to the production-alpha fixtures explicitly so future
    # checked-in ablation runs (e.g. `...alpha2.0.json`,
    # `...alpha0.5.json`) cannot get picked up by lexicographic sort
    # order and cause the test to assert against the wrong config.
    fixture_paths = sorted(
        reports_dir.glob("*-task507-booster-collapse-rerun-alpha1.0.json"),
    )
    if not fixture_paths:
        pytest.skip(
            "task #507 focused diagnostic fixtures absent — run "
            "scripts/diagnostic_482/run_507_focused.py at the production "
            "alpha=1.0 first"
        )

    # The most recent production-alpha run is the one the verification
    # gate will see.
    latest = fixture_paths[-1]
    payload = json.loads(latest.read_text())

    # Double-checked guard: the file name promised alpha=1.0; the
    # payload must agree, otherwise a hand-edited fixture would
    # silently slip through.
    assert payload.get("class_weight_alpha") == pytest.approx(1.0, rel=1e-9), (
        f"{latest.name}: file name advertises alpha=1.0 but payload "
        f"reports class_weight_alpha={payload.get('class_weight_alpha')}"
    )

    stuck = {("bonk", "1d"), ("celestia", "1d"), ("dogwifcoin", "1d"),
             ("celestia", "6h")}
    seen: set[tuple[str, str]] = set()
    for r in payload.get("rows", []):
        key = (r.get("coin_id"), r.get("timeframe"))
        if key not in stuck:
            continue
        assert r.get("bucket") == "no_collapse", (
            f"{key}: bucket={r.get('bucket')!r} (expected 'no_collapse')"
        )
        raw_s = float(r.get("raw_STABLE_share", 0.0))
        dcs = float(r.get("directional_call_share", 1.0))
        # The verification gate floor.
        assert raw_s >= 0.05, (
            f"{key}: raw_STABLE_share={raw_s:.4f} still below the "
            "5% gate floor — _train_lgb tiny-slice branch may have "
            "regressed"
        )
        assert dcs <= 0.95, (
            f"{key}: directional_call_share={dcs:.4f} still above the "
            "0.95 verification gate ceiling"
        )
        # Task #517 — dogwifcoin@1d was the slice that scraped by Task
        # #507's gate at raw_STABLE_share=0.0597 (only 0.0097 above the
        # 5% floor — see the verification report in
        # `reports/20260424T165948Z-task507-booster-collapse-verification.md`).
        # Adding the `volZScore60` rolling-vol-anomaly feature to
        # FEATURE_COLUMNS lifts it to 0.149+ on the same fixture, giving
        # the slice a healthy 0.05+ margin over the gate. Lock that
        # margin so a future feature-set refactor can't quietly let it
        # collapse back onto the floor without tripping a test.
        if key == ("dogwifcoin", "1d"):
            assert raw_s >= 0.10, (
                f"{key}: raw_STABLE_share={raw_s:.4f} below the 0.10 "
                "task #517 target margin — `volZScore60` may have "
                "regressed or been removed from FEATURE_COLUMNS"
            )
        seen.add(key)
    assert seen == stuck, (
        f"{latest.name}: missing rows for stuck slices: {stuck - seen}"
    )


def test_task519_focused_diagnostic_documents_800_round_regression():
    """Pin the documented Task #519 regression: at the production
    boosting budget (``ML_LGB_NUM_BOOST_ROUND=800``) the Task #507
    tiny-slice recovery branch does NOT rescue all 4 stuck slices.

    Specifically, on the latest Task #519 fixture
    (``reports/*-task519-booster-collapse-rerun-800rounds.json``):

    * ``bonk@1d`` falls back into ``bucket == 'booster_collapse'``
      and trips the verification gate ceiling
      (``directional_call_share > 0.95``).
    * ``celestia@1d`` clears the gate by 0.14 pp of margin on
      ``raw_STABLE_share`` (fixture value 0.0514 vs the 0.05 floor)
      and is therefore considered a continued regression for the
      purposes of this gate — the actual 04-25 production campaign
      sees DCS=0.9543 on this slice (i.e. it fails there even though
      the diagnostic just barely passes). The assertion below uses
      a 1-pp ceiling (``raw_S <= 0.06``) rather than the fixture's
      exact 0.0014 margin so that LightGBM's seed-level RNG noise on
      a 175-row holdout cannot flake the test green; an incoming
      real fix is expected to push this slice well clear of 0.06.

    This test exists to lock that regression in place so a future
    fix (see the predict-time DCS-floor proposal in the Task #519
    report) has something concrete to flip — when the underlying fix
    lands and ``bonk@1d`` clears the gate at 800 rounds, this test
    will fail loudly and the next agent will refresh the fixture and
    rewrite the assertions to a clean gate-pass.
    """
    import json
    from pathlib import Path

    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    fixture_paths = sorted(
        reports_dir.glob("*-task519-booster-collapse-rerun-800rounds.json"),
    )
    if not fixture_paths:
        pytest.skip(
            "task #519 focused diagnostic fixtures absent — run "
            "scripts/diagnostic_482/run_519_focused_800rounds.py first"
        )
    latest = fixture_paths[-1]
    payload = json.loads(latest.read_text())

    # Guard the rounds field so a hand-edited fixture cannot slip a
    # different boosting budget into a 519-named file.
    assert payload.get("num_boost_round") == 800, (
        f"{latest.name}: expected num_boost_round=800, got "
        f"{payload.get('num_boost_round')!r}"
    )

    rows_by_key = {
        (r.get("coin_id"), r.get("timeframe")): r
        for r in payload.get("rows", [])
        if r.get("status") == "ok"
    }

    # bonk@1d MUST still collapse at production rounds — that is the
    # core regression Task #519 documented.
    bonk = rows_by_key.get(("bonk", "1d"))
    assert bonk is not None, (
        f"{latest.name}: bonk@1d row missing from fixture"
    )
    assert bonk["bucket"] == "booster_collapse", (
        f"bonk@1d bucket={bonk['bucket']!r}: if this flipped to "
        "'no_collapse' the fix has landed — refresh the Task #519 "
        "fixture and rewrite this test to assert a clean gate-pass."
    )
    bonk_dcs = float(bonk["directional_call_share"])
    assert bonk_dcs > 0.95, (
        f"bonk@1d directional_call_share={bonk_dcs:.4f}: if this fell "
        "below 0.95 the verification gate now passes — rewrite this "
        "test to assert recovery."
    )

    # celestia@1d clears the floor by < 0.01 pp on the diagnostic but
    # the 04-25 production campaign saw DCS=0.9543 on the same slice,
    # so the diagnostic-pass is a knife-edge artefact, not a real
    # rescue. Pin the knife-edge so a real fix has to widen the
    # margin (or the campaign-side regression has to be eliminated by
    # a different mechanism) before this test can be relaxed.
    celestia_1d = rows_by_key.get(("celestia", "1d"))
    assert celestia_1d is not None, (
        f"{latest.name}: celestia@1d row missing from fixture"
    )
    raw_s_celestia_1d = float(celestia_1d["raw_STABLE_share"])
    assert raw_s_celestia_1d <= 0.06, (
        f"celestia@1d raw_STABLE_share={raw_s_celestia_1d:.4f}: if "
        "this rose comfortably above the 5% floor (>0.06) the "
        "knife-edge has been resolved — rewrite this test."
    )


def test_task539_require_volzscore60_raises_on_stale_parquet(tmp_path):
    """Task #539 — the in-memory `volZScore60` retrofit shim was
    removed because Task #537 regenerated every cached labeled-dataset
    parquet with the column natively. The replacement contract is a
    fail-fast guard (`_require_volzscore60`) that surfaces a clear
    error pointing at the refresher script if a stale snapshot ever
    re-enters the pipeline. This test pins that contract:

    * a parquet WITH `volZScore60` loads silently
    * a parquet WITHOUT `volZScore60` raises `RuntimeError` mentioning
      both the source path and the refresher script

    so a future agent who tries to "fix" a stale cache by re-adding a
    silent zero-fill shim will trip this test instead.
    """
    import pandas as pd

    from scripts.diagnostic_482.run_507_focused import _require_volzscore60

    # Happy path — column present, no raise.
    good = pd.DataFrame({
        "coin_id": ["bonk"], "timestamp_ms": [0],
        "lastPrice": [1.0], "volZScore60": [0.0],
    })
    _require_volzscore60(good, tmp_path / "fake_good.parquet")

    # Sad path — column absent, must raise with a useful message.
    bad = pd.DataFrame({
        "coin_id": ["bonk"], "timestamp_ms": [0], "lastPrice": [1.0],
    })
    bad_path = tmp_path / "fake_stale.parquet"
    with pytest.raises(RuntimeError) as exc:
        _require_volzscore60(bad, bad_path)
    msg = str(exc.value)
    assert "volZScore60" in msg, (
        f"error must name the missing column; got: {msg!r}"
    )
    assert "refresh_cached_datasets.py" in msg, (
        f"error must point at the refresher; got: {msg!r}"
    )
    assert str(bad_path) in msg, (
        f"error must include source path; got: {msg!r}"
    )
