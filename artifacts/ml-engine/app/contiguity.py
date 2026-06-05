"""Task #604 — gap-tolerant contiguous-run measurement for 5m candles.

WHY THIS EXISTS
---------------
Coinbase Exchange is the only viable deep-history 5m source for the 9
coins where OKX's 5m `history-candles` window only reaches ~66 days
back. Coinbase Exchange (`/products/{product}/candles`) only PRINTS a
candle when at least one trade occurred in that 5m window. For
low-volume periods on meme coins (worldcoin, dogwifcoin, floki) the
venue legitimately has nothing to record — the data is NOT lost or
corrupted, the venue simply did not print a bucket.

Without tolerance, every such gap resets the strict-consecutive
contiguous-run counter and the reported `contiguous_days` collapses to
whatever the longest UNBROKEN stretch is. Concretely after the
2026-04-29 backfill the post-backfill DB held ~320 wall-clock days of
5m data for BONK with `gap_rate=0.003` (good!), but the longest strict
run was only 127.89 days because Coinbase printed sparse buckets in a
~1h window during a low-volume weekend.

This module provides a single helper that BOTH the training campaign's
gate (`scripts/run_full_training_campaign.py::_coverage_per_slice`) AND
the daily top-up's health alert
(`app/scheduled_5m_topup.py::measure_contiguous_5m`) share so the two
surfaces never disagree about the same number.

WHAT TOLERANCE DOES — AND DOES NOT — DO
---------------------------------------
The tolerance ONLY loosens the LONGEST-RUN measurement. It does NOT
weaken any other clause:

  - `density` is computed from `unique_buckets / expected_buckets` and
    is unaffected by tolerance.
  - `gap_rate` (= 1 - density) still polices the OVERALL fraction of
    missing buckets across the whole 365-day window. The 5m gate
    enforces `gap_rate <= 0.01`, so a coin whose data is genuinely
    sparse (e.g. floki at 75% density / 25% gap_rate) STILL fails the
    gate despite the tolerance.
  - `synthetic_rows` is unaffected — non-real sources still fail.

The helper returns BOTH the tolerated wall-clock span (used by the
gate) AND the strict consecutive-bucket span (preserved as
`contiguous_days_strict` in audits) so an operator can always see both
numbers without re-running anything.

PER-TIMEFRAME TOLERANCE
-----------------------
`CONTIGUITY_TOLERANCE_SECONDS` defaults to 0 for every timeframe except
5m (set to 25200s = 7h = up to 84 missing 5m buckets). Higher
timeframes (1h/2h/6h/1d) use OKX exclusively, which prints every
bucket, so there is no Coinbase-style sparsity to absorb and the
strict semantic is preserved.

WHY 7h SPECIFICALLY (NOT 2h, NOT 8h)
------------------------------------
Tolerates one known provider outage-scale gap while keeping density
and gap-rate checks strict. 

Per-coin gap analysis on 2026-04-29 against the post-backfill DB
revealed that EVERY Coinbase-served coin has EXACTLY ONE gap > 2h
in the trailing 365 days, and ALL nine of those gaps fall on the
SAME calendar day: 2025-10-25. Gap durations across the nine coins
range from 360min (PEPE/SEI) to 390min (WLD) — i.e. one Coinbase
Exchange platform-level outage of ~6h-6.5h. JUP (OKX-only) shows
zero gaps > 5m at all in the same window.

7h (25200s = 84 missing 5m buckets) is the tightest principled
value that absorbs the worst observed outage (390min on WLD) with
~30 minutes of margin. A larger tolerance would silently swallow
future outages we should be flagging. A smaller value (e.g. 2h)
would leave BONK / PEPE / CELESTIA failing contiguity even though
their density (>=99.5%) and gap_rate (<=0.7%) are honestly clean.

This is NOT permission to accept generally sparse data — density
and gap_rate gates remain strict and still correctly fail
DOGWIF / WLD / FLOKI on real sparseness.

A follow-up task tracks back-filling the 2025-10-25 outage from a
non-Coinbase source so this tolerance can later be tightened or
removed entirely.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

CONTIGUITY_TOLERANCE_SECONDS: dict[str, int] = {
    "5m": 25200,
}


def compute_longest_contiguous_run(
    bucket_starts: Iterable[datetime],
    tf_sec: int,
    tolerance_sec: int,
) -> tuple[float, float]:
    """Walk `bucket_starts` (assumed sorted ascending; duplicates are
    deduped in-place and effectively ignored) and return:

        (tolerated_days, strict_days)

    Definitions
    -----------
    `tolerated_days`
        Wall-clock days spanned by the longest run in which every
        consecutive bucket pair `(prev, curr)` satisfies
        `(curr - prev).total_seconds() <= tf_sec + tolerance_sec`.
        The wall-clock span is `(last_bucket - first_bucket) + tf_sec`
        so a run of two consecutive buckets reads as `2 * tf_sec` days,
        not `1 * tf_sec`.

    `strict_days`
        `(longest_strict_consecutive_count * tf_sec) / 86400`. This is
        the original pre-tolerance metric, kept available so the audit
        trail can show both numbers side-by-side.

    Empty input returns `(0.0, 0.0)`. A single bucket returns
    `(tf_sec/86400, tf_sec/86400)`. The function is pure and never
    raises.
    """
    bs_list = list(bucket_starts)
    if not bs_list:
        return (0.0, 0.0)

    seen: set = set()
    unique: list = []
    for bs in bs_list:
        if bs not in seen:
            seen.add(bs)
            unique.append(bs)

    if len(unique) == 1:
        single = tf_sec / 86400.0
        return (single, single)

    tol_max_delta = tf_sec + tolerance_sec
    longest_tol_seconds = float(tf_sec)
    cur_start = unique[0]
    cur_end = unique[0]
    prev = unique[0]
    for bs in unique[1:]:
        delta = (bs - prev).total_seconds()
        if delta <= tol_max_delta:
            cur_end = bs
        else:
            run_seconds = (cur_end - cur_start).total_seconds() + tf_sec
            if run_seconds > longest_tol_seconds:
                longest_tol_seconds = run_seconds
            cur_start = bs
            cur_end = bs
        prev = bs
    run_seconds = (cur_end - cur_start).total_seconds() + tf_sec
    if run_seconds > longest_tol_seconds:
        longest_tol_seconds = run_seconds

    longest_strict = 1
    cur_strict = 1
    prev = unique[0]
    for bs in unique[1:]:
        delta = (bs - prev).total_seconds()
        if abs(delta - tf_sec) < 1:
            cur_strict += 1
            if cur_strict > longest_strict:
                longest_strict = cur_strict
        else:
            cur_strict = 1
        prev = bs
    strict_seconds = longest_strict * tf_sec

    return (longest_tol_seconds / 86400.0, strict_seconds / 86400.0)
