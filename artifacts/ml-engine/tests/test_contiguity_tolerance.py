"""Task #604 — gap-tolerant longest-contiguous-run measurement.

Pins the behaviour of `app.contiguity.compute_longest_contiguous_run`
and the gate's reaction to it. The motivation is documented in
`app/contiguity.py`'s module docstring: Coinbase Exchange only prints
a 5m candle when at least one trade occurred, so low-volume periods on
meme coins legitimately have missing buckets that the strict-consecutive
walker treats as the same data-loss signal as a real outage.

The tolerance only loosens the LONGEST-RUN measurement. The gate's
`gap_rate <= 0.01` and `density >= 0.80` clauses are unchanged, so
genuinely sparse coins (e.g. floki at 75% density) still fail the gate
despite the tolerance — these tests pin that invariant explicitly.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.contiguity import (
    CONTIGUITY_TOLERANCE_SECONDS,
    compute_longest_contiguous_run,
)
from scripts.run_full_training_campaign import (
    COVERAGE_BAR_DAYS,
    _evaluate_5m_gate,
    _evaluate_higher_tf_gate,
)


TF_5M = 300
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _bs(*offsets_in_buckets: int) -> list[datetime]:
    """Build a list of bucket_start datetimes from integer bucket
    offsets (one bucket = TF_5M seconds). E.g. _bs(0, 1, 3) → bucket
    indices 0, 1, 3 (gap of 1 missing bucket between 1 and 3)."""
    return [BASE + timedelta(seconds=o * TF_5M) for o in offsets_in_buckets]


# ── compute_longest_contiguous_run ───────────────────────────────────────

def test_tolerance_constant_pins_5m_7h_only():
    """The tolerance dict must only entry-cover 5m at 7h. If a future
    change adds another timeframe or shifts the value this test surfaces
    it loudly so the operator explicitly acknowledges the new ground
    truth (instead of silently broadening tolerance to OKX-served
    timeframes that don't need it)."""
    assert CONTIGUITY_TOLERANCE_SECONDS == {"5m": 25200}


def test_empty_input_returns_zero_zero():
    """Defensive: a coin with no rows must not crash the helper or the
    gate downstream. Returning (0.0, 0.0) lets the gate produce a
    `coverage_below_bar` verdict instead of raising."""
    assert compute_longest_contiguous_run([], TF_5M, 7200) == (0.0, 0.0)


def test_single_bucket_returns_one_bucket_width():
    """A single bucket spans exactly one tf_sec wall-clock. Both the
    tolerated and strict measures must agree on this — there's no
    multi-bucket run to disagree about."""
    tol, strict = compute_longest_contiguous_run(_bs(0), TF_5M, 7200)
    assert tol == TF_5M / 86400.0
    assert strict == TF_5M / 86400.0


def test_perfectly_consecutive_run_tolerated_equals_strict():
    """When there are NO gaps, tolerance must not inflate the answer.
    A 100-bucket consecutive run reads as 100 * tf_sec days under both
    measures."""
    tol, strict = compute_longest_contiguous_run(
        _bs(*range(100)), TF_5M, 7200,
    )
    expected_days = (100 * TF_5M) / 86400.0
    assert abs(tol - expected_days) < 1e-9
    assert abs(strict - expected_days) < 1e-9


def test_zero_tolerance_preserves_strict_semantic():
    """With tolerance_sec=0 the helper must produce the legacy
    strict-consecutive answer even for inputs with gaps. This is the
    contract higher timeframes (1h/2h/6h/1d) rely on — they have no
    entry in `CONTIGUITY_TOLERANCE_SECONDS` and must NOT silently get
    a loosened metric."""
    rows = _bs(0, 1, 2, 5, 6, 7, 8)  # 3-run, gap of 2, 4-run
    tol, strict = compute_longest_contiguous_run(rows, TF_5M, 0)
    expected = (4 * TF_5M) / 86400.0
    assert abs(tol - expected) < 1e-9
    assert abs(strict - expected) < 1e-9


def test_single_bucket_gap_within_tolerance_extends_run():
    """The motivating case: one missing 5m bucket in the middle of a
    long run. Strict semantic resets the counter; tolerance extends.
    Wall-clock span includes the gap minutes (honest framing)."""
    # buckets 0..49, gap of 1 (bucket 50 missing), buckets 51..99
    rows = _bs(*[i for i in range(100) if i != 50])
    tol, strict = compute_longest_contiguous_run(rows, TF_5M, 7200)
    # Tolerated: spans bucket 0 → bucket 99 inclusive
    # = (99 * 300) seconds + 300 (final bucket width) = 100 * 300 / 86400
    assert abs(tol - (100 * TF_5M) / 86400.0) < 1e-9
    # Strict: longest unbroken stretch is 50 buckets (0..49 inclusive)
    # or 49 buckets (51..99 inclusive — bucket 51 then 52..99 = 49). The
    # walker picks the longer of the two. `50 * 300 / 86400`.
    assert abs(strict - (50 * TF_5M) / 86400.0) < 1e-9


def test_24_bucket_gap_at_boundary_is_tolerated():
    """At a tolerance of 7200s (= 24 missing 5m buckets), a gap of
    exactly 24 missing buckets must be ABSORBED (delta = 25 * 300 =
    7500s, equal to tf_sec + tolerance_sec). Pin the inclusive boundary
    so a future edit to the comparison operator (`<` vs `<=`) is caught
    immediately. This test pins helper behavior at an arbitrary
    tolerance value, NOT the dict default — that's covered by
    `test_tolerance_constant_pins_5m_7h_only`."""
    # 50 consecutive, 24-bucket gap (skip 50..73), 50 consecutive
    rows = _bs(*range(50), *range(74, 124))
    tol, _ = compute_longest_contiguous_run(rows, TF_5M, 7200)
    # Tolerated wall-clock span: bucket 0 → bucket 123
    # = 124 * tf_sec / 86400
    assert abs(tol - (124 * TF_5M) / 86400.0) < 1e-9


def test_25_bucket_gap_breaks_run():
    """One bucket beyond the supplied tolerance (delta = 26 * 300 =
    7800s versus tf_sec + tolerance_sec = 7500s) must BREAK the run.
    Otherwise the chosen tolerance silently slides upward and a real
    data outage is masked. Pinned at 7200s here as a helper invariant
    irrespective of the dict default."""
    rows = _bs(*range(50), *range(75, 125))  # 25-bucket gap (skip 50..74)
    tol, _ = compute_longest_contiguous_run(rows, TF_5M, 7200)
    # Each segment is 50 buckets * tf_sec / 86400 wall-clock.
    assert abs(tol - (50 * TF_5M) / 86400.0) < 1e-9


def test_duplicate_buckets_neither_break_nor_extend_run():
    """OKX + Coinbase can both serve the same bucket_start in the
    overlap window. A duplicate must NOT break the run (delta=0 looks
    like a backwards step) and must NOT inflate it (a duplicated
    bucket isn't extra coverage). Pin the dedupe behaviour."""
    rows = _bs(0, 1, 1, 2, 3, 3, 4)  # 5 unique buckets, 2 dupes
    tol, strict = compute_longest_contiguous_run(rows, TF_5M, 7200)
    expected = (5 * TF_5M) / 86400.0
    assert abs(tol - expected) < 1e-9
    assert abs(strict - expected) < 1e-9


def test_multi_segment_picks_the_longest_tolerated_run():
    """Two stretches separated by a >2h gap. The helper must return
    the longer stretch's tolerated span, not the sum, and not the
    first one encountered."""
    # 30-bucket short stretch (0..29), 3h gap (skip 30..65),
    # 200-bucket long stretch (66..265)
    rows = _bs(*range(30), *range(66, 266))
    tol, _ = compute_longest_contiguous_run(rows, TF_5M, 7200)
    assert abs(tol - (200 * TF_5M) / 86400.0) < 1e-9


# ── End-to-end gate behaviour with realistic post-backfill numbers ───────

def _slice(coin: str, *, days: float, days_strict: float | None = None,
           density: float = 0.95, gap_rate: float = 0.0,
           synth: int = 0, tol_sec: int = 25200) -> dict:
    """Build a slice dict matching the post-Task-#604 shape — the gate
    now reads `contiguous_days` (tolerated) AND surfaces
    `contiguous_days_strict` + `contiguity_tolerance_seconds` in the
    reason string."""
    return {
        "coin": coin,
        "timeframe": "5m",
        "contiguous_days": days,
        "contiguous_days_strict": (
            days_strict if days_strict is not None else days
        ),
        "contiguity_tolerance_seconds": tol_sec,
        "density": density,
        "gap_rate": gap_rate,
        "synthetic_rows": synth,
    }


def test_gate_passes_bonk_like_profile_post_tolerance():
    """BONK after the 2026-04-29 backfill: tolerated_days≈320,
    strict_days≈128, density=0.997, gap_rate=0.003. Pre-tolerance the
    gate failed on contiguous_days; post-tolerance it must pass."""
    out = _evaluate_5m_gate([_slice(
        "bonk", days=320.0, days_strict=128.0,
        density=0.997, gap_rate=0.003,
    )])
    assert out["bonk"]["passed"] is True
    # Reason must surface BOTH numbers so an operator can audit which
    # value drove the verdict.
    assert "320" in out["bonk"]["reason"]
    assert "strict=128" in out["bonk"]["reason"]
    assert "tolerance=420m" in out["bonk"]["reason"]


def test_gate_fails_floki_like_profile_despite_tolerance():
    """FLOKI after the same backfill: tolerated_days≈320 (would clear
    the 305d bar), but density=0.749 and gap_rate=0.251. The tolerance
    must NOT override density/gap_rate — these clauses are the gate's
    last line of defense against genuinely sparse data."""
    out = _evaluate_5m_gate([_slice(
        "floki-inu", days=320.0, days_strict=66.0,
        density=0.749, gap_rate=0.251,
    )])
    assert out["floki-inu"]["passed"] is False
    # The reason must mention the failing clause so the operator
    # knows WHY a coin with apparently-fine coverage didn't clear.
    assert "gap_rate=0.251" in out["floki-inu"]["reason"]
    assert "density=0.75" in out["floki-inu"]["reason"]


def test_gate_fails_high_gap_rate_at_boundary():
    """A coin with a contiguous_days that clears the bar but
    gap_rate=0.013 (just over the 0.01 ceiling) must still fail.
    Pinned because injective/render/sei sit very close to this
    boundary post-backfill — a sloppy `>=` vs `>` swap would silently
    flip them to passing."""
    out = _evaluate_5m_gate([_slice(
        "injective-protocol", days=320.0, days_strict=66.0,
        density=0.987, gap_rate=0.013,
    )])
    assert out["injective-protocol"]["passed"] is False
    assert "gap_rate=0.013" in out["injective-protocol"]["reason"]


def test_gate_passing_reason_includes_tolerance_fragment():
    """Even on a clean pass the reason must mention the strict
    counterpart and tolerance — otherwise a future audit reader can't
    tell whether the pass came from a clean strict run or a tolerated
    one."""
    bar = COVERAGE_BAR_DAYS["5m"]
    out = _evaluate_5m_gate([_slice(
        "pepe", days=bar + 5, days_strict=bar + 5,
        density=0.99, gap_rate=0.005,
    )])
    assert out["pepe"]["passed"] is True
    assert "strict=" in out["pepe"]["reason"]
    assert "tolerance=420m" in out["pepe"]["reason"]


def test_failure_reason_omits_passing_clauses():
    """TRUTH-ONLY contract: when a coin fails the gate on ONE clause
    (e.g. gap_rate), the failure reason must NOT also list passing
    clauses as if they failed. Pre-fix the reason template was an
    "or"-template that surfaced every clause regardless of its real
    value (e.g. days=321 ... <305 even though 321>305) — that
    misled operators about which threshold drove the rejection
    (Task #604 architect review)."""
    # DOGWIF-shape: tolerated days clears the bar, density clears
    # the floor, only gap_rate fails. The reason must mention
    # gap_rate ONLY — NOT a phony "<305" or "<0.80".
    out = _evaluate_5m_gate([_slice(
        "dogwifcoin", days=320.0, days_strict=66.0,
        density=0.914, gap_rate=0.086,
    )])
    r = out["dogwifcoin"]["reason"]
    assert out["dogwifcoin"]["passed"] is False
    assert "coverage_below_bar" in r
    assert "gap_rate=0.086" in r
    # Honesty pins — these clauses are PASSING, must not appear:
    assert "<305" not in r, f"days=320 should not be flagged <305: {r}"
    assert "<0.80" not in r, f"density=0.91 should not be flagged <0.80: {r}"
    assert "synth=" not in r, f"synth=0 is passing, must not appear: {r}"
    # The full day numbers must still be present somewhere so audits
    # don't have to re-derive them.
    assert "days=320" in r
    assert "strict=66" in r


def test_higher_tf_gate_does_not_apply_tolerance():
    """1h/2h/6h/1d slices have no entry in the tolerance dict, so
    `contiguity_tolerance_seconds` is 0 and the reason must NOT mention
    a tolerance window. Pin so a future edit doesn't silently start
    reporting a tolerance string for OKX-only timeframes."""
    s = {
        "coin": "bonk",
        "timeframe": "1h",
        "contiguous_days": 360.0,
        "contiguous_days_strict": 360.0,
        "contiguity_tolerance_seconds": 0,
        "density": 0.99,
        "gap_rate": 0.001,
        "synthetic_rows": 0,
    }
    out = _evaluate_higher_tf_gate([s])
    verdict = out["bonk/1h"]
    assert verdict["passed"] is True
    assert "tolerance=" not in verdict["reason"]
    assert "strict=360" in verdict["reason"]
