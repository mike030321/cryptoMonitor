"""Pure-function unified decision engine.

Inputs (`DecisionRequest`):
  • base ML payload (probUp/probDown/probStable/expectedReturnPct)
  • optional regime + per-regime specialist views (Phase 3/4 meta gate)
  • timeframe + last price + ATR (for SL/TP geometry)
  • current open book (`PortfolioState`) — open positions, equity, cash
  • optional gate overrides (so the live tuner can loosen/tighten without
    redeploying the engine)

Output (`DecisionResult`):
  • action: "long" | "short" | "no_trade"
  • confidence: probability of the side actually called (NOT max-class)
  • size_multiplier: tier × portfolio scaler
  • position_size_usd: equity × tier × multiplier (post entry-fee deduction)
  • sl_price / tp_price (post-slippage)
  • skip_reason / skip_detail when action == "no_trade"
  • portfolio_check: {sector, beta, regime_budget, correlation} pass/fail map
  • gates_applied: jsonb-able bag of every threshold value used (mirrors
    the live `gates_applied` jsonb on the prediction journal)

The function is deterministic, side-effect free, and never raises on bad
input — every failure path returns `action="no_trade"` with a typed
`skip_reason`. This is load-bearing: a thrown exception in the live path
is invisible to operators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.backtest.contract import Frictions, _require, get_frictions

# ── Constants (mirror of trade-math.ts) ────────────────────────────────────
ANOMALY_PRICE_RATIO_HIGH = 3.0
ANOMALY_PRICE_RATIO_LOW = 0.33


# ── Inputs ─────────────────────────────────────────────────────────────────
@dataclass
class SpecialistView:
    """Per-regime specialist's view of the bar (Phase 3/4 meta gate input)."""
    kind: str
    applicable: bool = False
    prob_up: Optional[float] = None
    prob_down: Optional[float] = None
    prob_stable: Optional[float] = None
    error: Optional[str] = None


@dataclass
class OpenPosition:
    """One open position from the paper-portfolio."""
    coin_id: str
    direction: str            # "up" | "down" | "long" | "short"
    notional_usd: float
    regime_at_entry: Optional[str] = None
    beta_to_btc: Optional[float] = None  # rolling btc-correlation; None=use 1.0


@dataclass
class PortfolioState:
    """Current portfolio snapshot (live: from paper_positions; backtest:
    accumulated by the simulator). All numbers in USD."""
    equity_usd: float
    cash_usd: float
    open_positions: list[OpenPosition] = field(default_factory=list)
    # Date-bucketed running PnL — used by the backtest simulator for the
    # daily-loss + drawdown halts. Live code reads from
    # paper-trader's existing `checkRiskLimits` instead.
    day_start_equity: Optional[float] = None
    peak_equity: Optional[float] = None


@dataclass
class DecisionRequest:
    coin_id: str
    timeframe: str
    last_price: float
    atr_value: float          # raw ATR in price units; engine clamps to floor
    prob_up: float
    prob_down: float
    prob_stable: float
    expected_return_pct: float
    regime: Optional[str] = None
    trend_bias: Optional[str] = None    # "bullish" | "bearish" | None
    specialists: list[SpecialistView] = field(default_factory=list)
    portfolio: Optional[PortfolioState] = None
    # Per-coin recent loss ledger (1=win, 0=loss) to enforce the
    # consecutive-loss block. Backtest passes its accumulated deque;
    # live trader passes the per-coin history at call time.
    recent_outcomes: list[int] = field(default_factory=list)
    # Optional gate overrides (live tuner). All optional.
    gate_min_confidence: Optional[float] = None
    gate_min_tp_distance_pct: Optional[float] = None
    gate_min_ev_vs_cost: Optional[float] = None
    gate_counter_trend_min_confidence: Optional[float] = None
    # Optional per-call portfolio-constraint overrides. Mirrors the
    # per-call gate_* overrides above for the fleet-level caps. Keys are
    # the same as `portfolio_constraints` in shared/trading-frictions.json
    # (max_sector_exposure_pct, max_correlated_exposure_pct,
    # max_beta_to_btc, regime_budget_pct). Used by the parity test to
    # exercise the correlated-exposure branch deterministically without
    # mutating the shared json; can also be used by the live tuner to
    # tighten/loosen fleet caps without redeploying.
    portfolio_constraints_override: Optional[dict] = None


# ── Outputs ────────────────────────────────────────────────────────────────
@dataclass
class DecisionResult:
    action: str               # "long" | "short" | "no_trade"
    confidence: float
    size_multiplier: float
    position_size_usd: float
    direction: Optional[str]  # "up" | "down" | None
    sl_price: Optional[float]
    tp_price: Optional[float]
    expires_in_ms: int
    skip_reason: Optional[str]
    skip_detail: Optional[str]
    gates_applied: dict
    portfolio_check: dict     # {sector_ok, correlation_ok, beta_ok, regime_ok, ...}
    raw: dict                 # debug payload — probs, regime, etc.


# ── Helpers ────────────────────────────────────────────────────────────────
def _apply_entry_slippage(raw: float, direction: str, slip: float) -> float:
    return raw * (1 + slip if direction == "up" else 1 - slip)


def decide_direction(
    p_down: float, p_stable: float, p_up: float,
    expected_return_pct: float,
    *,
    fr: Frictions,
    min_directional_prob: Optional[float] = None,
    min_directional_edge: Optional[float] = None,
    min_expected_return_pct: Optional[float] = None,
) -> tuple[Optional[str], float, Optional[str]]:
    """Mirror of artifacts/api-server/src/lib/quant-brain.ts:safetyNet logic.

    Returns (direction|None, confidence, skip_reason).
    confidence is the probability of the side actually called
    (probUp/probDown/probStable, never the diluted max-class).
    """
    mdp = fr.min_directional_prob if min_directional_prob is None else min_directional_prob
    mde = fr.min_directional_edge if min_directional_edge is None else min_directional_edge
    mer = fr.min_expected_return_pct if min_expected_return_pct is None else min_expected_return_pct

    dir_side = "up" if p_up >= p_down else "down"
    dir_prob = max(p_up, p_down)
    dir_edge = abs(p_up - p_down)
    if expected_return_pct > 0:
        exp_ret_sign = "up"
    elif expected_return_pct < 0:
        exp_ret_sign = "down"
    else:
        exp_ret_sign = "stable"

    if dir_prob < mdp:
        return None, p_stable, "abstain_low_directional_prob"
    if dir_edge < mde:
        return None, p_stable, "abstain_no_directional_edge"
    if abs(expected_return_pct) < mer:
        return None, p_stable, "abstain_exp_ret_below_cost"
    if exp_ret_sign != dir_side:
        return None, p_stable, "abstain_exp_ret_disagrees"

    confidence = p_up if dir_side == "up" else p_down
    return dir_side, confidence, None


def _check_portfolio(
    *,
    coin_id: str,
    new_notional: float,
    regime: Optional[str],
    portfolio: PortfolioState,
    fr: Frictions,
    overrides: Optional[dict] = None,
) -> tuple[bool, Optional[str], dict]:
    """Apply portfolio-level constraints.

    Returns (ok, skip_reason_if_blocked, breakdown_dict).
    """
    # Task #349 — fail-fast loader. Previously `fr.raw.get("portfolio_constraints") or {}`
    # silently substituted an empty dict when the block was missing; the
    # downstream `pc.get("enabled", False)` then made every fleet gate pass
    # without warning. Now we require the block to exist; an absent block
    # is a config bug, while `enabled: false` remains the explicit kill switch.
    pc = dict(_require(fr.raw, "portfolio_constraints", "raw"))
    if overrides:
        # Only the four numeric fleet thresholds are overridable per-call,
        # matching the TS surface area (artifacts/api-server/src/lib/
        # portfolio-constraints.ts:checkPortfolioConstraints). Unknown
        # keys (including `enabled` and `sector_map`) are intentionally
        # ignored so the two engines cannot drift via override semantics.
        _ALLOWED_OVERRIDE_KEYS = (
            "max_sector_exposure_pct",
            "max_correlated_exposure_pct",
            "max_beta_to_btc",
            "regime_budget_pct",
        )
        for k in _ALLOWED_OVERRIDE_KEYS:
            if k in overrides and overrides[k] is not None:
                pc[k] = overrides[k]
    if not _require(pc, "enabled", "portfolio_constraints"):
        return True, None, {"enabled": False}

    # Once enabled, every numeric cap and the sector_map (with its _default)
    # must be present — falling back to a Python-side literal here would
    # silently diverge from the TS live trader (artifacts/api-server/src/
    # lib/portfolio-constraints.ts) which now hard-requires the same keys.
    sector_map: dict[str, str] = _require(pc, "sector_map", "portfolio_constraints")
    default_sector = _require(sector_map, "_default", "portfolio_constraints.sector_map")
    new_sector = sector_map.get(coin_id, default_sector)

    equity = max(portfolio.equity_usd, 1e-9)
    open_notional = sum(p.notional_usd for p in portfolio.open_positions)
    proposed_total = open_notional + new_notional

    breakdown: dict = {
        "enabled": True,
        "new_sector": new_sector,
        "regime": regime,
        "open_notional_usd": open_notional,
        "new_notional_usd": new_notional,
    }

    # Sector cap
    sector_caps = _require(pc, "max_sector_exposure_pct", "portfolio_constraints")
    by_sector: dict[str, float] = {}
    for p in portfolio.open_positions:
        s = sector_map.get(p.coin_id, default_sector)
        by_sector[s] = by_sector.get(s, 0.0) + p.notional_usd
    sector_after = by_sector.get(new_sector, 0.0) + new_notional
    sector_share = sector_after / equity
    breakdown["sector_share_after"] = sector_share
    breakdown["sector_cap"] = sector_caps
    if sector_share > sector_caps:
        return False, "portfolio_sector_cap", breakdown

    # Correlated-exposure cap (any sector with >=2 distinct coins)
    coins_in_sector = {p.coin_id for p in portfolio.open_positions
                       if sector_map.get(p.coin_id, default_sector) == new_sector}
    coins_in_sector.add(coin_id)
    correlated_cap = _require(pc, "max_correlated_exposure_pct", "portfolio_constraints")
    if len(coins_in_sector) >= 2:
        if sector_share > correlated_cap:
            breakdown["correlated_cap"] = correlated_cap
            return False, "portfolio_correlated_exposure", breakdown

    # Beta-to-BTC (equally weighted by notional). New coins default beta=1.0
    # so an unconfigured coin is treated as one-for-one with BTC — a safe
    # over-estimate. BTC itself counts as 1.0.
    beta_cap = _require(pc, "max_beta_to_btc", "portfolio_constraints")
    total = proposed_total if proposed_total > 0 else 1e-9
    beta_sum = 0.0
    for p in portfolio.open_positions:
        b = p.beta_to_btc if p.beta_to_btc is not None else 1.0
        beta_sum += b * p.notional_usd
    beta_sum += 1.0 * new_notional
    book_beta = beta_sum / total
    breakdown["book_beta"] = book_beta
    breakdown["beta_cap"] = beta_cap
    if book_beta > beta_cap:
        return False, "portfolio_beta_cap", breakdown

    # Regime budget
    regime_budget = _require(pc, "regime_budget_pct", "portfolio_constraints")
    if regime:
        by_regime: dict[str, float] = {}
        for p in portfolio.open_positions:
            r = p.regime_at_entry or "unknown"
            by_regime[r] = by_regime.get(r, 0.0) + p.notional_usd
        regime_after = by_regime.get(regime, 0.0) + new_notional
        regime_share = regime_after / equity
        breakdown["regime_share_after"] = regime_share
        breakdown["regime_budget"] = regime_budget
        if regime_share > regime_budget:
            return False, "portfolio_regime_budget", breakdown

    return True, None, breakdown


# ── Main entry point ──────────────────────────────────────────────────────
def decide(req: DecisionRequest, fr: Optional[Frictions] = None) -> DecisionResult:
    fr = fr or get_frictions()

    g_min_conf = req.gate_min_confidence if req.gate_min_confidence is not None \
        else fr.gate("MIN_CONFIDENCE_TO_TRADE")["value"]
    g_min_tpd = req.gate_min_tp_distance_pct if req.gate_min_tp_distance_pct is not None \
        else fr.gate("MIN_TP_DISTANCE_PCT")["value"]
    g_min_ev = req.gate_min_ev_vs_cost if req.gate_min_ev_vs_cost is not None \
        else fr.gate("MIN_EV_VS_COST")["value"]
    g_ct_min = req.gate_counter_trend_min_confidence if req.gate_counter_trend_min_confidence is not None \
        else fr.gate("COUNTER_TREND_MIN_CONFIDENCE")["value"]

    asym_long_min = fr.asymmetric_long_min_confidence
    sl_mult = fr.sl_mult(req.timeframe)
    tp_mult = fr.tp_mult(req.timeframe)
    atr_floor_pct = fr.atr_floor_pct(req.timeframe)
    tf_ms = fr.timeframe_ms(req.timeframe)
    round_trip_cost = fr.round_trip_cost_pct
    consec_loss_n = int(fr.raw["recent_loss_block"]["consecutive_losses"])

    gates_applied: dict = {
        "min_confidence": g_min_conf,
        "min_tp_distance_pct": g_min_tpd,
        "min_ev_vs_cost": g_min_ev,
        "counter_trend_min_confidence": g_ct_min,
        "asymmetric_long_min_confidence": asym_long_min,
        "policy_version": fr.quant_policy_version,
    }
    raw = {
        "prob_up": req.prob_up, "prob_down": req.prob_down, "prob_stable": req.prob_stable,
        "expected_return_pct": req.expected_return_pct, "regime": req.regime,
    }
    portfolio_check: dict = {}

    def _skip(reason: str, detail: str = "") -> DecisionResult:
        return DecisionResult(
            action="no_trade", confidence=0.0, size_multiplier=0.0,
            position_size_usd=0.0, direction=None,
            sl_price=None, tp_price=None,
            expires_in_ms=tf_ms, skip_reason=reason, skip_detail=detail,
            gates_applied=gates_applied, portfolio_check=portfolio_check, raw=raw,
        )

    # 1. Direction emit (mirror of quant-brain safetyNet floors)
    direction, confidence, skip = decide_direction(
        req.prob_down, req.prob_stable, req.prob_up,
        req.expected_return_pct, fr=fr,
    )
    if direction is None:
        return _skip(skip or "abstain_unknown")

    # 2. Min-confidence gate (asymmetric for longs)
    min_conf = max(g_min_conf, asym_long_min) if direction == "up" else g_min_conf
    if confidence < min_conf:
        return _skip("confidence_below_threshold", f"{confidence:.3f}<{min_conf:.3f}")

    # 3. Counter-trend gate
    if req.trend_bias == "bullish" and direction == "down" and confidence < g_ct_min:
        return _skip("counter_trend_regime")
    if req.trend_bias == "bearish" and direction == "up" and confidence < g_ct_min:
        return _skip("counter_trend_regime")

    # 4. Recent-loss block
    if len(req.recent_outcomes) >= consec_loss_n and all(w == 0 for w in req.recent_outcomes[-consec_loss_n:]):
        return _skip("consecutive_losses", f"{consec_loss_n} losses in a row on {req.coin_id}")

    # 5. SL/TP geometry & EV gate (post-slippage)
    if req.last_price <= 0:
        return _skip("no_entry_price")
    adj_entry = _apply_entry_slippage(req.last_price, direction, fr.slippage_pct)
    fallback_floor = adj_entry * atr_floor_pct
    effective_atr = max(req.atr_value, fallback_floor)
    sl_distance = effective_atr * sl_mult
    tp_distance = effective_atr * tp_mult
    tp_distance_pct = tp_distance / adj_entry

    if tp_distance_pct < g_min_tpd:
        return _skip("fee_gate_tp_floor", f"{tp_distance_pct:.4f}<{g_min_tpd:.4f}")
    ev_score = confidence * tp_distance_pct
    ev_required = g_min_ev * round_trip_cost
    if ev_score < ev_required:
        return _skip("fee_gate_ev", f"{ev_score:.4f}<{ev_required:.4f}")

    sl_price = adj_entry - sl_distance if direction == "up" else adj_entry + sl_distance
    tp_price = adj_entry + tp_distance if direction == "up" else adj_entry - tp_distance

    # 6. Sizing
    portfolio = req.portfolio
    if portfolio is None:
        portfolio = PortfolioState(equity_usd=fr.initial_balance_usd,
                                   cash_usd=fr.initial_balance_usd)
    equity = portfolio.equity_usd
    cash = portfolio.cash_usd
    invested = sum(p.notional_usd for p in portfolio.open_positions)
    if len(portfolio.open_positions) >= fr.max_open_positions_per_agent:
        return _skip("max_open_positions")

    tier_pct = fr.tiered_position_pct(confidence)
    position_size = equity * tier_pct
    position_size = min(position_size, cash * 0.80, equity * fr.max_position_pct)
    if (invested + position_size) / max(equity, 1e-9) >= fr.max_portfolio_at_risk:
        position_size = max(0.0, equity * fr.max_portfolio_at_risk - invested)
    entry_fee = position_size * fr.taker_fee_pct
    position_size_post_fee = position_size - entry_fee
    if position_size_post_fee < 1.0:
        return _skip("sizing_too_small")

    # 7. Portfolio constraints (Phase 5 — fleet-level)
    ok, p_skip, breakdown = _check_portfolio(
        coin_id=req.coin_id, new_notional=position_size_post_fee,
        regime=req.regime, portfolio=portfolio, fr=fr,
        overrides=req.portfolio_constraints_override,
    )
    portfolio_check.update(breakdown)
    if not ok:
        return _skip(p_skip or "portfolio_constraint", "")

    return DecisionResult(
        action="long" if direction == "up" else "short",
        confidence=confidence,
        size_multiplier=tier_pct,
        position_size_usd=position_size_post_fee,
        direction=direction,
        sl_price=sl_price,
        tp_price=tp_price,
        expires_in_ms=tf_ms,
        skip_reason=None,
        skip_detail=None,
        gates_applied=gates_applied,
        portfolio_check=portfolio_check,
        raw=raw,
    )
