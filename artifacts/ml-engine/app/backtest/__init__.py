"""Phase-3 backtester.

Mirrors the live paper-trader (artifacts/api-server/src/lib/paper-trader.ts)
fee/slippage/risk/execution rules using the SHARED contract at
shared/trading-frictions.json. Outputs per-(coin, timeframe) metrics, a
Monte Carlo trade-order reshuffle, a per-regime breakdown, and a deploy
verdict.

Entry points:
* `app.backtest.run` — CLI: `pnpm --filter @workspace/ml-engine backtest`
* `app.backtest.simulator.simulate` — programmatic (used by tests)
* `/ml/backtest` — HTTP route returning the rendered HTML report.
"""
