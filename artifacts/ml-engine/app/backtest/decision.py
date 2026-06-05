"""Deploy/no-deploy verdict from a backtest report.

Rule lives in shared/trading-frictions.json under `backtest_deploy_gate`
so it is auditable and matches whatever the live trader expects.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from .contract import Frictions
from .regime_breakdown import REGIMES


@dataclass
class Verdict:
    deploy: bool
    reasons: list[str]
    passing_regimes: list[str]

    def to_dict(self) -> dict: return asdict(self)


def decide(
    overall_metrics: dict,
    regime_metrics: dict[str, dict],
    *,
    fr: Frictions,
    mc_p05_drawdown_pct: Optional[float] = None,
) -> Verdict:
    gate = fr.deploy_gate
    expectancy_min = float(gate["expectancy_usd_min"])
    sharpe_min = float(gate["sharpe_min"])
    require_one_regime = bool(gate.get("require_at_least_one_passing_regime", True))
    min_trades = int(gate.get("min_total_trades", 30))
    max_mc_dd = float(gate.get("max_monte_carlo_p05_drawdown_pct", 25.0))

    reasons: list[str] = []
    deploy = True

    n_trades = int(overall_metrics.get("n_trades") or 0)
    if n_trades < min_trades:
        deploy = False
        reasons.append(f"insufficient trades: {n_trades} < {min_trades}")

    expectancy = overall_metrics.get("expectancy_usd")
    if expectancy is None or expectancy < expectancy_min:
        deploy = False
        reasons.append(f"expectancy_usd {expectancy} < {expectancy_min}")

    sharpe = overall_metrics.get("sharpe_per_trade")
    if sharpe is None or sharpe < sharpe_min:
        deploy = False
        reasons.append(f"sharpe_per_trade {sharpe} < {sharpe_min}")

    if mc_p05_drawdown_pct is not None and mc_p05_drawdown_pct > max_mc_dd:
        deploy = False
        reasons.append(
            f"MC p05 drawdown {mc_p05_drawdown_pct:.2f}% > {max_mc_dd:.2f}%"
        )

    passing: list[str] = []
    regime_min_trades = max(1, min_trades // 2)
    for r in REGIMES:
        m = regime_metrics.get(r) or {}
        if (m.get("n_trades") or 0) >= regime_min_trades and (m.get("expectancy_usd") or 0) > 0:
            passing.append(r)

    if require_one_regime and not passing:
        deploy = False
        reasons.append("no regime passes (need ≥1 regime with positive expectancy)")

    if deploy and not reasons:
        reasons.append("all gates passed")
    return Verdict(deploy=deploy, reasons=reasons, passing_regimes=passing)
