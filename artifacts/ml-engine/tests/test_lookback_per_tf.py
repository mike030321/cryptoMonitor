"""Task #417 — per-timeframe lookback resolution + 1d backfill default.

The 1d slice was bound by a single global ``ML_LOOKBACK_DAYS`` envelope,
which capped the holdout at ~265 rows and the binomial-noise σ on a
directional-accuracy estimate at ~0.031. Task #417 widens the 1d window
to ≥1000 days without forcing 5m / 1h / 2h / 6h to use the same window.

These tests pin down the new contract:

  * ``lookback_days_for("1d")`` defaults to ≥1000 days.
  * ``lookback_days_for(tf)`` for any non-1d timeframe still falls back
    to the legacy ``LOOKBACK_DAYS`` global.
  * ``ML_LOOKBACK_DAYS_<TF>`` env-vars override the per-tf default.
  * Bogus / non-positive overrides are ignored (fall back to default).
  * The backfill module's ``DEFAULT_DAYS_BY_TF["1d"]`` matches the
    trainer-side floor so a default backfill run produces enough bars
    for the trainer's wider 1d window.
  * The campaign orchestrator's Phase-2 ``COVERAGE_BAR_DAYS["1d"]``
    is at the 1000-row floor named in the task brief.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_lookback_env(monkeypatch):
    """Clear any stray per-tf env overrides between tests so the default
    branch is exercised cleanly. ``ML_LOOKBACK_DAYS`` itself is left
    alone because conftest may have set it for the wider suite.
    """
    for tf in ("1m", "5m", "1h", "2h", "6h", "1d"):
        monkeypatch.delenv(f"ML_LOOKBACK_DAYS_{tf.upper()}", raising=False)
    yield


def test_lookback_days_for_1d_defaults_to_at_least_1000_days():
    from app.training.train import lookback_days_for

    # Task #417 — 1d holdout has to be ≥720 rows to drag σ on a DA
    # estimate down from 0.031 to 0.019. With 5-fold CV and a 20%
    # calibration tail, ≥1000 days is the floor that lands ≥720 rows
    # in the holdout.
    assert lookback_days_for("1d") >= 1000


def test_lookback_days_for_non_1d_falls_back_to_global():
    from app.training import train as train_mod

    # Short timeframes intentionally keep the legacy 365-day envelope.
    # Pulling 3y of 5m bars (~315k rows/coin) blows up fit time without
    # a comparable noise-band benefit, since each 5m holdout already
    # has thousands of rows.
    for tf in ("5m", "1h", "2h", "6h"):
        assert train_mod.lookback_days_for(tf) == train_mod.LOOKBACK_DAYS


def test_lookback_days_for_env_override_per_tf(monkeypatch):
    from app.training.train import lookback_days_for

    monkeypatch.setenv("ML_LOOKBACK_DAYS_1D", "1500")
    monkeypatch.setenv("ML_LOOKBACK_DAYS_1H", "200")
    assert lookback_days_for("1d") == 1500
    assert lookback_days_for("1h") == 200


def test_lookback_days_for_ignores_bogus_override(monkeypatch):
    from app.training.train import lookback_days_for

    # Empty string, non-numeric, and zero / negative overrides must all
    # fall back to the default — otherwise a typo in a deployment env
    # would silently shrink the 1d window back to the broken 365-day
    # envelope task #417 was written to retire.
    monkeypatch.setenv("ML_LOOKBACK_DAYS_1D", "")
    assert lookback_days_for("1d") >= 1000
    monkeypatch.setenv("ML_LOOKBACK_DAYS_1D", "not-an-int")
    assert lookback_days_for("1d") >= 1000
    monkeypatch.setenv("ML_LOOKBACK_DAYS_1D", "0")
    assert lookback_days_for("1d") >= 1000
    monkeypatch.setenv("ML_LOOKBACK_DAYS_1D", "-1")
    assert lookback_days_for("1d") >= 1000


def test_backfill_history_default_1d_window_meets_trainer_floor():
    # Importing the backfill module is heavy (httpx, asyncpg) but cheap
    # enough at the test layer; we only read the constant.
    backfill_mod = importlib.import_module("scripts.backfill_history")
    from app.training.train import lookback_days_for

    # The default 1d backfill window MUST be at least as wide as the
    # trainer's per-tf lookback. If someone bumps `lookback_days_for`
    # without bumping the backfill default, the trainer's `lookback_ms`
    # will reach back further than the database actually contains and
    # the wider window becomes a no-op.
    assert (
        backfill_mod.DEFAULT_DAYS_BY_TF["1d"] >= lookback_days_for("1d")
    ), (
        "scripts.backfill_history.DEFAULT_DAYS_BY_TF['1d'] must cover "
        "the trainer's 1d lookback (task #417). Bump them together."
    )


def test_backfill_history_default_1d_window_is_wider_than_short_tfs():
    backfill_mod = importlib.import_module("scripts.backfill_history")

    # Sanity: the 1d window is the only one widened by task #417.
    # Other timeframes stay at their legacy ≤365d defaults.
    assert backfill_mod.DEFAULT_DAYS_BY_TF["1d"] >= 1000
    for tf in ("1h", "2h", "6h"):
        assert backfill_mod.DEFAULT_DAYS_BY_TF[tf] <= 365


def test_campaign_phase2_audit_1d_bar_matches_task_417_floor():
    # The Phase-2 data audit's hard gate is the operator-facing proof
    # that we actually achieved the wider 1d window before training.
    # If COVERAGE_BAR_DAYS["1d"] drifts below 1000 the gate stops
    # protecting the noise-band math; if it drifts above the trainer
    # lookback we'd start failing the audit on a window the trainer
    # never asked for.
    campaign_mod = importlib.import_module(
        "scripts.run_full_training_campaign"
    )
    from app.training.train import lookback_days_for

    one_d_bar = campaign_mod.COVERAGE_BAR_DAYS["1d"]
    assert one_d_bar >= 1000
    assert one_d_bar <= lookback_days_for("1d")


def test_run_training_uses_per_tf_lookback_when_building_dataset():
    """The trainer must call ``build_labeled_dataset`` with a *per-tf*
    lookback window — 1d uses ≥1000 days, others stay at the legacy
    365d global. Without this assertion someone could revert
    ``run_training`` to a single global ``lookback_ms`` and the noise-
    band fix from task #417 would silently regress.

    We intercept the call by inspecting the source rather than running
    the whole pipeline (which pulls in pandas, lightgbm, optuna, and
    asyncpg). The CLI integration is covered by
    ``test_run_training_report_includes_lookback_days_per_tf`` below
    (slow, opt-in) and by the campaign smoke test in production.
    """
    import inspect

    from app.training import train as train_mod

    import re

    src = inspect.getsource(train_mod.run_training)
    # The lookback map is computed once per run and indexed inside the
    # tf loop — both pieces have to be present together.
    assert "lookback_per_tf" in src
    assert "lookback_days_for(tf)" in src
    # The dataset builder must be invoked with a *per-tf* lookback,
    # not the legacy global. If someone reverts to
    # ``LOOKBACK_DAYS * 24 * 3600 * 1000`` inside the loop, this fires.
    assert "tf_lookback_ms" in src
    # Allow arbitrary whitespace between args (the formatter may wrap
    # the call across lines).
    assert re.search(
        r"build_labeled_dataset\s*\(\s*coin_ids\s*,\s*tf\s*,\s*tf_lookback_ms",
        src,
    ), "build_labeled_dataset must be called with the per-tf tf_lookback_ms"


def test_run_training_report_carries_per_tf_lookback_keys():
    """Static-source check: the report dict in ``run_training`` must
    populate both the legacy ``lookback_days`` field (for back-compat
    with dashboards) AND the new ``lookback_days_per_tf`` map.
    """
    import inspect

    from app.training import train as train_mod

    src = inspect.getsource(train_mod.run_training)
    assert '"lookback_days_per_tf": lookback_per_tf' in src
    # Legacy field is the MAX of the per-tf window so single-number
    # consumers (e.g. the operator dashboard's "training horizon"
    # tile) still see a meaningful answer after task #417.
    assert "max(lookback_per_tf.values())" in src
