from market_meta_brain.domain.types import GovernanceOutcome, PortfolioTelemetry, QuantSliceTelemetry, TelemetryBatch
from market_meta_brain.runtime.service import MarketMetaBrainService


def make_batch():
    return TelemetryBatch(
        slices=[
            QuantSliceTelemetry(
                coin="BTC",
                timeframe="1h",
                strategy_family="momentum",
                edge=0.12,
                confidence=0.72,
                calibrated_confidence=0.69,
                risk_score=0.30,
                recent_accuracy=0.57,
                pnl_state=0.03,
                drawdown_state=0.01,
                disagreement=0.12,
                prediction_error=0.10,
                regime="trend",
                volatility=0.40,
                correlation_shift=0.08,
                exposure=0.20,
                turnover=0.11,
                slippage_bps=2.0,
                anomaly_flags=[],
            ),
            QuantSliceTelemetry(
                coin="ETH",
                timeframe="4h",
                strategy_family="mean_reversion",
                edge=0.03,
                confidence=0.58,
                calibrated_confidence=0.55,
                risk_score=0.36,
                recent_accuracy=0.52,
                pnl_state=0.01,
                drawdown_state=0.02,
                disagreement=0.20,
                prediction_error=0.18,
                regime="trend",
                volatility=0.45,
                correlation_shift=0.11,
                exposure=0.10,
                turnover=0.15,
                slippage_bps=3.0,
                anomaly_flags=[],
            ),
        ],
        portfolio=PortfolioTelemetry(
            total_drawdown=0.05,
            realized_vol=0.30,
            concentration=0.24,
            leverage=0.12,
            liquidity_stress=0.05,
            correlation_shift=0.10,
            active_risk_budget=0.66,
            kill_switch_distance=0.85,
            anomaly_flags=[],
        ),
        timestamp="2026-04-23T11:00:00Z",
    )


def test_evaluate_returns_directive():
    service = MarketMetaBrainService()
    directive = service.evaluate(make_batch())
    assert directive.caution_level >= 0.0
    assert "momentum" in directive.trust_multiplier
    assert abs(sum(directive.allocation_weight.values()) - 1.0) < 1e-6


def test_record_outcome_updates_memory():
    service = MarketMetaBrainService()
    directive = service.evaluate(make_batch())
    reward = service.record_outcome(
        directive,
        GovernanceOutcome(
            realized_pnl=0.08,
            realized_drawdown=0.01,
            realized_stability=0.82,
            turnover_cost=0.01,
            action_churn=0.04,
            correct_defense=0.3,
            correct_suppression=0.2,
            missed_edge_cost=0.0,
        ),
        timestamp="2026-04-23T12:00:00Z",
    )
    assert reward > -1.0
    assert service.episodic_memory.summarize_rewards()["count"] == 1.0
    assert len(service.regime_memory.prototypes) == 1
