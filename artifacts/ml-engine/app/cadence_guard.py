"""Cadence-correctness assertion for price-store writers (task #343).

Mirror of `artifacts/api-server/src/lib/cadence-guard.ts`. Every writer
to `price_history` or `price_candles` MUST call `assert_native_cadence`
before its INSERT — see the TS doc for the failure mode this prevents.
"""
from __future__ import annotations

PRICE_HISTORY_NATIVE_TIMEFRAME = "1m"

_VALID_PRICE_CANDLES_TIMEFRAMES = frozenset({
    "1m", "5m", "1h", "2h", "6h", "1d",
})


class CadenceGuardError(ValueError):
    """Raised when a writer tries to insert a row at the wrong cadence
    for the target table. See `assert_native_cadence` for the rules."""


def assert_native_cadence(timeframe: str, source: str, table: str) -> None:
    """Reject any non-1m write to `price_history` and any unknown
    timeframe write to `price_candles`."""
    if table == "price_history":
        if timeframe != PRICE_HISTORY_NATIVE_TIMEFRAME:
            raise CadenceGuardError(
                f"[cadence-guard] refusing {timeframe} write to price_history "
                f"from source='{source}': price_history is the 1m-tick store. "
                "Aggregated bars must go to price_candles "
                "(see schema-audit.md / task #343)."
            )
        return
    if table == "price_candles":
        if timeframe not in _VALID_PRICE_CANDLES_TIMEFRAMES:
            raise CadenceGuardError(
                f"[cadence-guard] refusing {timeframe} write to price_candles "
                f"from source='{source}': not a recognised native cadence."
            )
        return
    raise CadenceGuardError(
        f"[cadence-guard] unknown table {table!r} (source={source!r})."
    )
