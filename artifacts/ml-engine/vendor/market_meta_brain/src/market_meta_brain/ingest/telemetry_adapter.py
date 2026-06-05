from __future__ import annotations

from collections import defaultdict

from market_meta_brain.domain.types import FamilyState, QuantSliceTelemetry, TelemetryBatch
from market_meta_brain.utils.math_utils import safe_div


class TelemetryAdapter:
    """Aggregates slice telemetry into stable family-level features."""

    def aggregate_families(self, batch: TelemetryBatch) -> dict[str, FamilyState]:
        grouped: dict[str, list[QuantSliceTelemetry]] = defaultdict(list)
        for item in batch.slices:
            grouped[item.strategy_family].append(item)

        families: dict[str, FamilyState] = {}
        for family, items in grouped.items():
            n = len(items)
            families[family] = FamilyState(
                family=family,
                avg_edge=safe_div(sum(x.edge for x in items), n),
                avg_confidence=safe_div(sum(x.confidence for x in items), n),
                avg_calibrated_confidence=safe_div(sum(x.calibrated_confidence for x in items), n),
                avg_disagreement=safe_div(sum(x.disagreement for x in items), n),
                avg_prediction_error=safe_div(sum(x.prediction_error for x in items), n),
                avg_accuracy=safe_div(sum(x.recent_accuracy for x in items), n),
                avg_pnl_state=safe_div(sum(x.pnl_state for x in items), n),
                avg_drawdown_state=safe_div(sum(x.drawdown_state for x in items), n),
                avg_volatility=safe_div(sum(x.volatility for x in items), n),
                avg_correlation_shift=safe_div(sum(x.correlation_shift for x in items), n),
                total_exposure=sum(x.exposure for x in items),
                total_turnover=sum(x.turnover for x in items),
                avg_risk_score=safe_div(sum(x.risk_score for x in items), n),
                avg_slippage_bps=safe_div(sum(x.slippage_bps for x in items), n),
                anomaly_count=sum(len(x.anomaly_flags) for x in items),
                slice_count=n,
            )
        return families

    def dominant_regime(self, batch: TelemetryBatch) -> str:
        counts: dict[str, int] = defaultdict(int)
        for item in batch.slices:
            counts[item.regime] += 1
        if not counts:
            return "unknown"
        return max(counts.items(), key=lambda kv: kv[1])[0]
