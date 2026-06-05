"""Task #603 — JUP-style partial-history exemption for `_evaluate_5m_gate`.

The 305-day contiguous-bucket bar comes from task #366 and is the right
default for coins that list on Coinbase Exchange (where 5m history goes
back ~10 years). `jupiter-exchange-solana` does NOT list on Coinbase
and OKX's 5m history for it only reaches 2025-02, so the truthful
ceiling is ~66 days as of 2026-04. Failing the gate for JUP would
either:

  1. Block the entire training campaign (gate is hard) — wrong, the
     other 9 coins should still train, OR
  2. Force the operator to silently lower the bar for everyone — also
     wrong, that masks regressions on the 9 coins that CAN clear 305d.

The fix is the explicit `KNOWN_5M_PARTIAL_COINS` set: members are
exempt from the contiguous_days clause but still must clear density,
gap_rate, and synthetic_rows checks. The verdict carries a
`partial_history_exempt=True` flag so the audit summary is honest about
which coins passed because they cleared the bar vs. which ones passed
because the bar was waived.
"""
from __future__ import annotations

from scripts.run_full_training_campaign import (
    COVERAGE_BAR_DAYS,
    KNOWN_5M_PARTIAL_COINS,
    _evaluate_5m_gate,
)


def _slice(coin: str, *, days: float, density: float = 0.95,
           gap_rate: float = 0.0, synth: int = 0) -> dict:
    """Return a minimal coverage-slice dict matching the shape
    `_coverage_per_slice` produces. Only the keys the gate inspects
    are populated."""
    return {
        "coin": coin,
        "timeframe": "5m",
        "contiguous_days": days,
        "density": density,
        "gap_rate": gap_rate,
        "synthetic_rows": synth,
    }


def test_jupiter_is_in_partial_history_set():
    """Pin JUP as the canonical exempt coin. If a future change adds
    or removes coins this test surfaces it loudly so the operator
    explicitly acknowledges the new ground truth (instead of silently
    accepting a lower bar)."""
    assert "jupiter-exchange-solana" in KNOWN_5M_PARTIAL_COINS


def test_passing_coverage_clears_gate_for_non_exempt_coin():
    """Sanity: a coin not in the exempt set passes when its
    contiguous_days + density + gap_rate + synth all clear the bar.
    Guards against the exemption logic accidentally inverting the
    branch for the common case."""
    bar = COVERAGE_BAR_DAYS["5m"]
    out = _evaluate_5m_gate([_slice("pepe", days=bar + 5)])
    assert out["pepe"]["passed"] is True
    assert out["pepe"]["partial_history_exempt"] is False
    # Task #604 — `ok` reasons now carry the days + strict counterpart
    # so audit logs show BOTH numbers without re-running the gate.
    # Pin the prefix instead of the full string so we don't lock the
    # exact strict / tolerance fragment (those are owned by
    # `tests/test_contiguity_tolerance.py`).
    assert out["pepe"]["reason"].startswith("ok (days=")


def test_below_bar_fails_for_non_exempt_coin():
    """A coin with only 100 contiguous days fails the gate and the
    reason mentions both `coverage_below_bar` and the actual day count
    so operators can grep the audit summary for the smoking gun."""
    bar = COVERAGE_BAR_DAYS["5m"]
    out = _evaluate_5m_gate([_slice("pepe", days=100)])
    assert out["pepe"]["passed"] is False
    assert out["pepe"]["partial_history_exempt"] is False
    assert "coverage_below_bar" in out["pepe"]["reason"]
    assert f"<{bar}" in out["pepe"]["reason"]


def test_jupiter_passes_at_partial_history_with_clean_quality():
    """The point of the exemption: JUP at 66 contiguous days passes
    because it's in `KNOWN_5M_PARTIAL_COINS`, and the verdict is
    explicitly tagged `partial_history_exempt=True` plus the reason
    says 'partial_history_exempt' so the audit summary makes the
    waiver visible."""
    out = _evaluate_5m_gate([_slice("jupiter-exchange-solana", days=66)])
    verdict = out["jupiter-exchange-solana"]
    assert verdict["passed"] is True
    assert verdict["partial_history_exempt"] is True
    assert "partial_history_exempt" in verdict["reason"]
    # The day count must still appear in the reason so an operator can
    # spot a sudden drop (e.g. 66d → 5d) without reading the slice
    # dict directly.
    assert "66" in verdict["reason"]


def test_exempt_coin_still_fails_on_density():
    """The exemption only waives the contiguous_days clause. A density
    breach on JUP must STILL fail the gate — otherwise a quiet data-
    quality regression (e.g. half the 5m buckets vanish) would slip
    through unnoticed."""
    out = _evaluate_5m_gate([_slice(
        "jupiter-exchange-solana", days=66, density=0.50,
    )])
    verdict = out["jupiter-exchange-solana"]
    assert verdict["passed"] is False
    assert verdict["partial_history_exempt"] is True
    assert "coverage_partial_exempt" in verdict["reason"]
    assert "density=0.50" in verdict["reason"]


def test_exempt_coin_still_fails_on_synthetic_rows():
    """Same principle for synthetic_rows — the exemption is purely
    about coverage breadth, not about quality. A coin that's been
    backfilled with fabricated rows must fail no matter what."""
    out = _evaluate_5m_gate([_slice(
        "jupiter-exchange-solana", days=66, synth=42,
    )])
    verdict = out["jupiter-exchange-solana"]
    assert verdict["passed"] is False
    assert verdict["partial_history_exempt"] is True
    assert "synth=42" in verdict["reason"]


def test_mixed_set_handles_each_coin_independently():
    """A realistic post-backfill snapshot: 9 coins clear the 305-day
    bar, JUP sits at 66 with a clean quality profile, and a hypothetical
    fresh-listed coin sits below the bar without an exemption. The
    gate must verdict each one according to its own rules."""
    bar = COVERAGE_BAR_DAYS["5m"]
    slices = [
        _slice("pepe", days=bar + 1),
        _slice("bonk", days=bar + 1),
        _slice("floki-inu", days=bar + 1),
        _slice("dogwifcoin", days=bar + 1),
        _slice("sei-network", days=bar + 1),
        _slice("render-token", days=bar + 1),
        _slice("injective-protocol", days=bar + 1),
        _slice("celestia", days=bar + 1),
        _slice("worldcoin-wld", days=bar + 1),
        _slice("jupiter-exchange-solana", days=66),  # exempt
        _slice("hypothetical-new", days=10),         # not exempt → fails
    ]
    out = _evaluate_5m_gate(slices)

    for c in (
        "pepe", "bonk", "floki-inu", "dogwifcoin", "sei-network",
        "render-token", "injective-protocol", "celestia",
        "worldcoin-wld",
    ):
        assert out[c]["passed"] is True, f"{c} should clear 305d bar"
        assert out[c]["partial_history_exempt"] is False

    assert out["jupiter-exchange-solana"]["passed"] is True
    assert out["jupiter-exchange-solana"]["partial_history_exempt"] is True

    assert out["hypothetical-new"]["passed"] is False
    assert out["hypothetical-new"]["partial_history_exempt"] is False
    assert "coverage_below_bar" in out["hypothetical-new"]["reason"]
