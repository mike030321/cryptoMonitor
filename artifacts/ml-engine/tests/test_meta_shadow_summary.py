"""Task #455 — Unit tests for the shadow-24h summarizer.

These tests pin the math behind the two pass/fail checks in
`scripts/meta_shadow_24h.py::summarize_timeframe`:

  * `lightgbm_share` and its 60% threshold check.
  * `abstain_rate` (computed exactly over known-kind rows from a
    row-level kind×action cross-tab) and its 30%-80% band check.

They use synthetic accumulator buckets so the function can be
exercised without touching the database.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.meta_shadow_24h import (  # noqa: E402
    ABSTAIN_BAND_HI,
    ABSTAIN_BAND_LO,
    LIGHTGBM_SHARE_MIN,
    summarize_timeframe,
)


def _bucket(rows: list[tuple[str, str]], versions: dict[str, int] | None = None) -> dict:
    """Build an accumulator bucket from a list of (kind, action) row
    tuples — matching exactly what `_amain` accumulates per row from
    the journal."""
    by_kind = Counter(k for k, _ in rows)
    by_action = Counter(a for _, a in rows)
    by_kind_action = Counter(rows)
    total = len(rows)
    return {
        "total": total,
        "by_meta_kind": by_kind,
        "by_meta_action": by_action,
        "by_kind_action": by_kind_action,
        "by_meta_version": Counter(versions or {"unversioned": total}),
    }


def test_lightgbm_share_passes_when_majority_lightgbm():
    rows = (
        [("lightgbm", "long")] * 30
        + [("lightgbm", "short")] * 30
        + [("lightgbm", "no_trade")] * 20
        + [("heuristic", "no_trade")] * 10
        + [("unknown", "no_trade")] * 10
    )
    out = summarize_timeframe(_bucket(rows))
    assert out["lightgbm_share"] == 0.8
    assert out["lightgbm_share_check_60pct"] == "pass"


def test_lightgbm_share_fails_below_threshold():
    rows = [("lightgbm", "long")] * 50 + [("heuristic", "short")] * 50
    out = summarize_timeframe(_bucket(rows))
    assert out["lightgbm_share"] == 0.5
    # 50/50 lightgbm/heuristic is below the 60% bar but unknown is not
    # the majority, so the failure mode should be plain 'fail'.
    assert out["lightgbm_share_check_60pct"] == "fail"


def test_lightgbm_share_check_flags_pre_wiring_majority():
    rows = [("lightgbm", "long")] * 10 + [("unknown", "no_trade")] * 90
    out = summarize_timeframe(_bucket(rows))
    assert out["lightgbm_share"] == 0.10
    # unknown_n (90) > known_n (10) — the operator should see why the
    # share is low, so we want the explicit pre-wiring marker.
    assert out["lightgbm_share_check_60pct"] == "fail (mostly unknown — pre-wiring rows present)"


def test_lightgbm_share_check_returns_n_a_when_no_lightgbm_rows():
    rows = [("heuristic", "long")] * 10 + [("heuristic", "no_trade")] * 90
    out = summarize_timeframe(_bucket(rows))
    assert out["lightgbm_share"] == 0.0
    assert out["lightgbm_share_check_60pct"] == "n/a (no lightgbm rows on this tf)"


def test_abstain_rate_uses_exact_known_kind_cross_tab():
    """The abstain rate must be computed exactly from the
    (kind, action) cross-tab — NOT proportionally attributed.
    This is the case that fails proportional attribution: half the
    no_trade rows live on unknown rows and so must NOT count toward
    the known-kind abstain rate."""
    rows = (
        [("lightgbm", "long")] * 30
        + [("lightgbm", "short")] * 20
        # Only 5 of the lightgbm rows are no_trade.
        + [("lightgbm", "no_trade")] * 5
        # The other 45 no_trade rows live on unknown (pre-wiring) rows
        # and must NOT count toward the known-kind abstain rate.
        + [("unknown", "no_trade")] * 45
    )
    out = summarize_timeframe(_bucket(rows))
    # known_no_trade = 5; known_n = 55 → abstain_rate = 5/55 ≈ 0.0909.
    # A proportional approximation would have given
    # 50 * (55/100) / 55 = 0.5 — which is wildly wrong for the band check.
    assert out["known_no_trade_count"] == 5
    assert out["known_kind_count"] == 55
    assert out["abstain_rate"] == round(5 / 55, 4)
    assert out["abstain_rate_in_band_30_to_80pct"] == "fail (out of band)"


def test_abstain_rate_band_pass_at_lower_edge():
    rows = (
        [("lightgbm", "long")] * 35
        + [("lightgbm", "short")] * 35
        + [("lightgbm", "no_trade")] * 30
    )
    out = summarize_timeframe(_bucket(rows))
    assert out["abstain_rate"] == 0.30
    assert out["abstain_rate_in_band_30_to_80pct"] == "pass"


def test_abstain_rate_band_fail_below():
    rows = (
        [("lightgbm", "long")] * 50
        + [("lightgbm", "short")] * 40
        + [("lightgbm", "no_trade")] * 10
    )
    out = summarize_timeframe(_bucket(rows))
    assert out["abstain_rate"] == 0.10
    assert out["abstain_rate_in_band_30_to_80pct"] == "fail (out of band)"


def test_abstain_rate_band_fail_above():
    rows = (
        [("lightgbm", "long")] * 5
        + [("lightgbm", "short")] * 5
        + [("lightgbm", "no_trade")] * 90
    )
    out = summarize_timeframe(_bucket(rows))
    assert out["abstain_rate"] == 0.90
    assert out["abstain_rate_in_band_30_to_80pct"] == "fail (out of band)"


def test_empty_bucket_is_handled_gracefully():
    out = summarize_timeframe({})
    assert out["total_predictions_24h"] == 0
    assert out["lightgbm_share"] == 0.0
    assert out["abstain_rate"] == 0.0
    assert out["lightgbm_share_check_60pct"] == "n/a (no lightgbm rows on this tf)"
    assert out["abstain_rate_in_band_30_to_80pct"] == "n/a (no known-kind rows)"


def test_thresholds_recorded_alongside_summary():
    out = summarize_timeframe(_bucket([("lightgbm", "long")]))
    assert out["thresholds_used"]["lightgbm_share_minimum"] == LIGHTGBM_SHARE_MIN
    assert out["thresholds_used"]["abstain_band"] == [ABSTAIN_BAND_LO, ABSTAIN_BAND_HI]


def test_cross_tab_is_serialised_into_summary():
    """The summary must expose the kind×action cross-tab so the
    operator can audit the exact counts that produced the abstain
    rate."""
    rows = (
        [("lightgbm", "long")] * 2
        + [("lightgbm", "no_trade")] * 1
        + [("heuristic", "short")] * 3
        + [("unknown", "no_trade")] * 4
    )
    out = summarize_timeframe(_bucket(rows))
    assert out["by_kind_action"] == {
        "lightgbm": {"long": 2, "no_trade": 1},
        "heuristic": {"short": 3},
        "unknown": {"no_trade": 4},
    }
    # known_no_trade = 1 (only the lightgbm/no_trade row); known_n = 2+1+3 = 6.
    assert out["known_no_trade_count"] == 1
    assert out["known_kind_count"] == 6
    assert out["abstain_rate"] == round(1 / 6, 4)
