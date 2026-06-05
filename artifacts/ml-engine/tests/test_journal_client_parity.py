"""Phase 1 parity test — backtest journal rows must match the live
journal contract.

Locks down three invariants the api-server's adaptive engine relies on:

1. realizedReturnPct is the SIGNED price movement (not direction-adjusted
   PnL). A short trade that profited still records a NEGATIVE
   realizedReturnPct; the win/loss judgement lives in `outcome`.
2. outcome uses the live taxonomy (`correct` | `wrong` | `neutral`),
   never `win`/`loss`.
3. Skipped predictions ARE serialised, with becameTrade=false implied
   (no `simulatedTrade` payload) and a structured reason in
   `gatesApplied`.
"""
from __future__ import annotations

from app.backtest.journal_client import (
    _classify_outcome,
    _mae_pct,
    _mfe_pct,
    _skip_to_row,
    _trade_to_row,
)
from app.backtest.simulator import SkipRow, TradeRow


def _make_trade(direction: str, entry: float, exit_: float, pnl_usd: float) -> TradeRow:
    return TradeRow(
        coin_id="bitcoin", timeframe="1h", direction=direction,
        entry_ts_ms=1_700_000_000_000, exit_ts_ms=1_700_003_600_000,
        entry_price=entry, exit_price=exit_,
        position_size_usd=1000.0, pnl_usd=pnl_usd,
        pnl_pct=(pnl_usd / 1000.0) * 100.0,
        exit_reason="tp", regime_at_entry="bullish", confidence=0.7,
        peak_price=max(entry, exit_), mae_price=min(entry, exit_),
        entry_fee=0.5, exit_fee=0.5, slippage_pct=0.0001,
        raw_entry_price=entry, raw_exit_price=exit_,
    )


# Invariant 1: realizedReturnPct is signed price movement, not PnL.
def test_short_trade_pnl_positive_but_price_return_negative():
    # Down trade, price fell: pnl > 0 but price return < 0.
    t = _make_trade("down", entry=50_000.0, exit_=49_500.0, pnl_usd=10.0)
    row = _trade_to_row(t, model_version="test")
    assert row["realizedReturnPct"] < 0
    assert abs(row["realizedReturnPct"] - (-1.0)) < 1e-9
    # Live taxonomy: short + price-down => correct.
    assert row["outcome"] == "correct"


def test_long_trade_uses_signed_price_return():
    t = _make_trade("up", entry=50_000.0, exit_=50_500.0, pnl_usd=10.0)
    row = _trade_to_row(t, model_version="test")
    assert row["realizedReturnPct"] > 0
    assert abs(row["realizedReturnPct"] - 1.0) < 1e-9
    assert row["outcome"] == "correct"


# Invariant 2: outcome must be live taxonomy.
def test_outcome_uses_live_taxonomy():
    assert _classify_outcome("up", 100.0, 101.0) == "correct"
    assert _classify_outcome("up", 100.0, 99.0) == "wrong"
    assert _classify_outcome("down", 100.0, 99.0) == "correct"
    assert _classify_outcome("down", 100.0, 101.0) == "wrong"
    # 0.05% deadband for neutral.
    assert _classify_outcome("up", 100.0, 100.0001) == "neutral"
    # Never returns 'win'/'loss'.
    for d in ("up", "down", "stable"):
        for entry, exit_ in [(100.0, 101.0), (100.0, 99.0), (100.0, 100.0)]:
            assert _classify_outcome(d, entry, exit_) in {"correct", "wrong", "neutral"}


# Invariant: MAE matches live convention (POSITIVE magnitude, never < 0).
# Live formula in paper-trader.ts:817/820 wraps the value in `Math.max(0, ...)`.
# Backtest must mirror that exactly so trade_journal.maePct has one
# semantic across both sources.
def test_mae_is_positive_magnitude_for_long_trade():
    # Long trade, price dipped to 49,000 before rallying to 50,500.
    t = _make_trade("up", entry=50_000.0, exit_=50_500.0, pnl_usd=10.0)
    t.mae_price = 49_000.0
    t.peak_price = 50_500.0
    mae = _mae_pct(t)
    mfe = _mfe_pct(t)
    assert mae >= 0.0, "live convention: MAE is positive magnitude"
    assert abs(mae - 2.0) < 1e-9
    assert mfe >= 0.0
    assert abs(mfe - 1.0) < 1e-9


def test_mae_is_positive_magnitude_for_short_trade():
    # Short trade, price spiked to 51,000 before settling at 49,500.
    t = _make_trade("down", entry=50_000.0, exit_=49_500.0, pnl_usd=10.0)
    t.mae_price = 51_000.0  # adverse for a short = price WENT UP
    t.peak_price = 49_500.0
    mae = _mae_pct(t)
    mfe = _mfe_pct(t)
    assert mae >= 0.0
    assert abs(mae - 2.0) < 1e-9
    assert mfe >= 0.0
    assert abs(mfe - 1.0) < 1e-9


def test_mae_clamps_to_zero_when_no_adverse_excursion():
    # Long trade that only ever went up — MAE must clamp to 0 (live
    # behaviour via Math.max(0, …)).
    t = _make_trade("up", entry=50_000.0, exit_=50_500.0, pnl_usd=10.0)
    t.mae_price = 50_100.0  # never went below entry
    assert _mae_pct(t) == 0.0


# Invariant 3: skips serialise as no-trade rows with structured reason.
def test_skip_row_has_no_simulated_trade_and_carries_reason():
    s = SkipRow(coin_id="ethereum", timeframe="4h",
                timestamp_ms=1_700_000_000_000,
                reason="counter_trend_regime", detail="bullish vs down 0.55")
    row = _skip_to_row(s, model_version="test")
    # No simulatedTrade → api-server records becameTrade=false.
    assert "simulatedTrade" not in row
    # Structured reason is preserved in gatesApplied.
    assert row["gatesApplied"]["counter_trend_regime"] is True
    assert row["gatesApplied"]["backtest"] is True
    assert row["gatesApplied"]["detail"] == "bullish vs down 0.55"
    # Direction & outcome match live abstain conventions.
    assert row["direction"] == "stable"
    assert row["outcome"] == "neutral"
    assert row["realizedReturnPct"] is None
