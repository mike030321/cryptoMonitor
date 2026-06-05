from __future__ import annotations

from market_meta_brain.domain.types import PortfolioTelemetry, QuantSliceTelemetry, TelemetryBatch


class QuantBridge:
    """Thin adapter for merging into an existing quant platform."""

    def from_payload(self, payload: dict) -> TelemetryBatch:
        slices = [QuantSliceTelemetry(**item) for item in payload.get("slices", [])]
        portfolio = PortfolioTelemetry(**payload["portfolio"])
        return TelemetryBatch(slices=slices, portfolio=portfolio, timestamp=payload.get("timestamp"))
