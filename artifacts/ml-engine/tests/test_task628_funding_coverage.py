"""Regression guard for Task #628 — Funding-rate backfill coverage.

The 8 MTTM coins each need ~84 days of historical OKX funding-rate
rows in `market_signals` so the slow-loop trainer's asof-join
populates `funding_rate` (and the derived `open_interest_z` z-score
denominator) for the 6h / 1d slices.

We've already lost those rows once (pre-Task #628 the table only
spanned a few days of "now" snapshots; Task #628 added
`scripts/backfill_funding_history.py` to fill the gap idempotently).
After that fix, *most* of the 6h cached parquet should have a real
funding value — the OKX cap is ~92 days, so at minimum ~84 / 365 =
~23 % of the year-long 6h window should be covered.

When the backfilled rows go missing again (e.g. an environment was
reset, the table was dropped, the post-merge setup did not re-run
the backfiller, …) every 6h slice ends up with `funding_rate = NaN`
across 100 % of training rows — exactly the dead-feature pattern
that costs the booster ~1 of 38 features and tanks log-loss /
calibration on the resulting model.

This test reads the latest cached `models/datasets/6h_<TS>.parquet`
and asserts:

  * the file exists (the dataset refresher has run at least once),
  * the file contains > 0 rows for the 8 MTTM coins,
  * the `funding_rate` column has at least
    `MIN_FUNDING_COVERAGE_PCT` non-null share (default 10 %).

The threshold is intentionally loose so a fresh OKX-only env (with
just the most recent ~7 days of funding before the next backfill
cron tick) still passes — but a wholesale 0 % coverage regression
fails loudly the next time CI runs.

In environments without the cached parquet or without populated 6h
slices for the MTTM universe (fresh CI, unit-test-only repos), the
test SKIPs rather than failing — the safety net is meant for live
ml-engine envs, not for installs that never built a dataset.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "models" / "datasets"

MTTM_COINS = (
    "bonk",
    "celestia",
    "dogwifcoin",
    "floki-inu",
    "injective-protocol",
    "jupiter-exchange-solana",
    "pepe",
    "render-token",
)

MIN_FUNDING_COVERAGE_PCT = 10.0


def _latest_6h_parquet() -> Path | None:
    if not DATASETS_DIR.exists():
        return None
    files = sorted(
        DATASETS_DIR.glob("6h_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def test_latest_6h_dataset_has_funding_rate_coverage():
    """The latest cached 6h dataset must have real funding values for
    at least `MIN_FUNDING_COVERAGE_PCT` of the MTTM-coin rows.

    See module docstring for the failure mode this guards against.
    """
    parquet = _latest_6h_parquet()
    if parquet is None:
        pytest.skip(
            "no cached 6h parquet on disk — dataset refresher has not "
            "produced a snapshot yet (fresh env / unit-test-only repo)"
        )
    df = pd.read_parquet(parquet)
    if "funding_rate" not in df.columns:
        pytest.fail(
            f"cached 6h dataset {parquet.name} is missing the "
            "`funding_rate` column — dataset schema regression"
        )
    if "coin_id" not in df.columns:
        pytest.skip(f"cached 6h dataset {parquet.name} has no coin_id column")
    sub = df[df["coin_id"].isin(MTTM_COINS)]
    if sub.empty:
        pytest.skip(
            f"cached 6h dataset {parquet.name} has no rows for the 8 "
            "MTTM coins — dataset built for a different universe"
        )
    coverage_pct = 100.0 * sub["funding_rate"].notna().sum() / len(sub)
    assert coverage_pct >= MIN_FUNDING_COVERAGE_PCT, (
        f"latest 6h dataset {parquet.name}: funding_rate coverage "
        f"{coverage_pct:.1f}% < {MIN_FUNDING_COVERAGE_PCT:.1f}% across "
        f"{len(sub)} MTTM-coin rows. The OKX funding-history backfill "
        "(scripts/backfill_funding_history.py, Task #628) likely needs "
        "to be re-run — without it the slow-loop trainer asof-joins "
        "100% NaN funding for the year-long 6h training window, "
        "killing one of LightGBM's 38 input features and tanking "
        "model calibration."
    )


def test_funding_coverage_consistent_across_mttm_coins():
    """All 8 MTTM coins should have similar funding coverage — if one
    coin is at 0% while the others are at 24%, the OKX SWAP listing
    map (`OKX_SWAP_BASE` in `backfill_funding_history.py`) is missing
    that coin's instrument id.
    """
    parquet = _latest_6h_parquet()
    if parquet is None:
        pytest.skip("no cached 6h parquet on disk")
    df = pd.read_parquet(parquet)
    if "funding_rate" not in df.columns or "coin_id" not in df.columns:
        pytest.skip("dataset missing funding_rate or coin_id column")
    per_coin: dict[str, float] = {}
    for coin in MTTM_COINS:
        sub = df[df["coin_id"] == coin]
        if sub.empty:
            continue
        per_coin[coin] = 100.0 * sub["funding_rate"].notna().sum() / len(sub)
    if not per_coin:
        pytest.skip("no MTTM coin rows in the latest 6h dataset")
    coverages = list(per_coin.values())
    spread = max(coverages) - min(coverages)
    # 25-percentage-point spread is generous: OKX returns 4h cycles for
    # some pairs and 8h cycles for others, so coin-to-coin coverage can
    # vary 12-25%. A larger spread implies one coin is unmapped.
    assert spread <= 25.0, (
        f"funding coverage spread {spread:.1f}pp across MTTM coins "
        f"is too wide — at least one coin's OKX SWAP listing is "
        f"missing from `OKX_SWAP_BASE`. Per-coin pct: {per_coin}"
    )
