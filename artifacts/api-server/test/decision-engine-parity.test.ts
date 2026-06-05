/**
 * Phase 5 unified decision engine — live ↔ backtest parity test.
 *
 * Phase 5 collapsed the live trader and the offline backtester onto a
 * single source of truth: the Python `decide()` function in
 * `artifacts/ml-engine/app/decision_engine/engine.py`. The TypeScript
 * paper-trader (`artifacts/api-server/src/lib/paper-trader.ts`) reaches
 * that function over HTTP via `getMlDecision()` / `/ml/decide`, and the
 * Python backtester (`artifacts/ml-engine/app/backtest/simulator.py`)
 * calls it in-process. Both paths must therefore return the same answer
 * on the same inputs — but until now nothing automated caught a
 * regression that broke that promise.
 *
 * This test feeds a handful of fixtures through BOTH paths:
 *   (a) Python decide() — invoked via a tiny stdin/stdout helper script
 *       at `artifacts/ml-engine/scripts/decide_for_parity.py`.
 *   (b) The TypeScript live-trader gate stack — re-implemented in this
 *       file using the same exported helpers the live trader uses
 *       (tieredPositionPct, getQuantEvGateRequiredPct, getSlMultiplier,
 *       getTpMultiplier, getAtrFloorPct, applyEntrySlippage,
 *       checkPortfolioConstraints, …) so any drift on either side
 *       is caught the moment it lands.
 *
 * The assertion: action, skipReason, and positionSizeUsd (within a
 * small epsilon) must match for every fixture. Failures print the
 * diverged field together with the fixture name so the bug is obvious.
 *
 * If you change `decide()` on either side, add a fixture here that
 * exercises the new branch — and update both implementations or the
 * test will fire.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  TAKER_FEE_PCT,
  ROUND_TRIP_COST_PCT,
  MAX_POSITION_PCT,
  MAX_PORTFOLIO_AT_RISK,
  MAX_OPEN_POSITIONS_PER_AGENT,
  ASYMMETRIC_LONG_MIN_CONFIDENCE,
  RECENT_LOSS_BLOCK_COUNT,
  QUANT_MIN_DIRECTIONAL_PROB,
  QUANT_MIN_DIRECTIONAL_EDGE,
  KELLY_MIN_TRADES,
  KELLY_RAMP_END,
  KELLY_FRACTION,
  FLEET_BRAKE_MIN_OPEN,
  FLEET_BRAKE_DOMINANCE,
  MAX_CASH_PER_POSITION_PCT,
  tieredPositionPct,
  getSlMultiplier,
  getTpMultiplier,
  getAtrFloorPct,
} from "../src/lib/trading-constants";
import { applyEntrySlippage } from "../src/lib/trade-math";
import {
  checkPortfolioConstraints,
  type PortfolioOpenPosition,
} from "../src/lib/portfolio-constraints";
import {
  getMinConfidenceToTrade,
  getMinTpDistancePct,
  getMinEvVsCost,
  getCounterTrendMinConfidence,
} from "../src/lib/tuning-tracker";
import { getMinExpectedReturnPct } from "../src/lib/quant-brain";

// ───────────────────────────────────────────────────────────────────────────
// Fixtures — each one targets a specific gate path through `decide()`.
// ───────────────────────────────────────────────────────────────────────────
interface OpenPosFx {
  coinId: string;
  direction: "up" | "down";
  notionalUsd: number;
  regimeAtEntry?: string | null;
  betaToBtc?: number | null;
}
interface PortfolioFx {
  equityUsd: number;
  cashUsd: number;
  openPositions: OpenPosFx[];
}
interface Fixture {
  name: string;
  coinId: string;
  timeframe: string;
  lastPrice: number;
  atrValue: number;
  probUp: number;
  probDown: number;
  probStable: number;
  expectedReturnPct: number;
  regime?: string | null;
  trendBias?: "bullish" | "bearish" | null;
  portfolio?: PortfolioFx | null;
  recentOutcomes?: number[];
  // Optional gate overrides (mirror of DecisionRequest.gate_*). When set,
  // both Python decide() and the TS mirror below use them in place of the
  // baseline gate value — exercises the engine's per-call override path
  // the live tuner uses to loosen / tighten gates without redeploying.
  gateMinConfidence?: number;
  gateMinTpDistancePct?: number;
  gateMinEvVsCost?: number;
  gateCounterTrendMinConfidence?: number;
  // Optional per-call portfolio-constraint overrides — mirror of
  // DecisionRequest.portfolio_constraints_override on the Python side.
  // Only the four numeric fleet thresholds are overridable; sector_map
  // and the kill switch stay sourced from shared/trading-frictions.json.
  portfolioConstraintsOverride?: {
    max_sector_exposure_pct?: number;
    max_correlated_exposure_pct?: number;
    max_beta_to_btc?: number;
    regime_budget_pct?: number;
  };
  // ── Live-trader-only state (task #345 sizing-parity wrapper) ─────────
  // The Python `decide()` engine doesn't model the fleet correlation
  // brake or Kelly sizing — those live exclusively in paper-trader.ts.
  // To keep the offline backtester from silently disagreeing with the
  // live trader on positionSizeUsd, BOTH sides apply the same wrapper
  // (apply_live_extras in decide_for_parity.py, applyLiveExtras below)
  // and the test asserts equality. Drift in either implementation
  // immediately breaks parity. Out of scope (per task-345.md): porting
  // these into the in-process simulator's quantity-emit path.
  fleetState?: {
    /** open same-side ("up") fleet positions across the whole fleet. */
    up: number;
    /** open opposite-side ("down") fleet positions across the whole fleet. */
    down: number;
  };
  kellyState?: {
    totalTrades: number;
    winningTrades: number;
    /** mean win as percent of position size, e.g. 2.0 = 2% */
    avgWinPct: number;
    /** mean loss as percent of position size (positive value). */
    avgLossPct: number;
  };
  // The skip_reason both engines MUST produce for this fixture (or null
  // for an approve_*). Asserted in addition to the cross-engine parity
  // check so a fixture cannot silently degrade into "both paths skip
  // for the wrong reason but still agree".
  expectedSkipReason: string | null;
  /**
   * Optional override of the expected positionSizeUsd. When set, the
   * test asserts the parity-mirrored output equals this value (within
   * the standard epsilon) on top of the cross-engine parity check —
   * the wrapper is what owns Kelly's positionSize, so this field
   * pins down the value both implementations are supposed to land on.
   */
  expectedPositionSizeUsd?: number;
}

const EMPTY_PORTFOLIO: PortfolioFx = {
  equityUsd: 1_000,
  cashUsd: 1_000,
  openPositions: [],
};

const FIXTURES: Fixture[] = [
  {
    name: "approve_long",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    regime: "trending_up",
    trendBias: "bullish",
    portfolio: EMPTY_PORTFOLIO,
    expectedSkipReason: null,
  },
  {
    name: "approve_short",
    coinId: "ethereum",
    timeframe: "1h",
    lastPrice: 3_000,
    atrValue: 30,
    probUp: 0.18,
    probDown: 0.62,
    probStable: 0.20,
    expectedReturnPct: -0.5,
    regime: "trending_down",
    trendBias: "bearish",
    portfolio: EMPTY_PORTFOLIO,
    expectedSkipReason: null,
  },
  {
    name: "abstain_low_directional_prob",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    // Both up & down below QUANT_MIN_DIRECTIONAL_PROB (0.08).
    probUp: 0.04,
    probDown: 0.03,
    probStable: 0.93,
    expectedReturnPct: 0.0,
    portfolio: EMPTY_PORTFOLIO,
    expectedSkipReason: "abstain_low_directional_prob",
  },
  {
    name: "abstain_no_directional_edge",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    // Both > min_dir_prob, but |diff| < QUANT_MIN_DIRECTIONAL_EDGE (0.05).
    probUp: 0.30,
    probDown: 0.29,
    probStable: 0.41,
    expectedReturnPct: 0.5,
    portfolio: EMPTY_PORTFOLIO,
    expectedSkipReason: "abstain_no_directional_edge",
  },
  {
    name: "confidence_below_threshold",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    // Direction emitted (probUp wins, edge ≥ 0.05, |expRet| ≥ floor) but
    // the confidence (0.40) is below MIN_CONFIDENCE_TO_TRADE (0.45).
    probUp: 0.40,
    probDown: 0.32,
    probStable: 0.28,
    expectedReturnPct: 0.5,
    portfolio: EMPTY_PORTFOLIO,
    expectedSkipReason: "confidence_below_threshold",
  },
  {
    name: "counter_trend_regime",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    // Bullish regime but model wants to short with confidence below the
    // counter-trend override (0.65 default).
    probUp: 0.20,
    probDown: 0.55,
    probStable: 0.25,
    expectedReturnPct: -0.5,
    regime: "trending_up",
    trendBias: "bullish",
    portfolio: EMPTY_PORTFOLIO,
    expectedSkipReason: "counter_trend_regime",
  },
  {
    name: "consecutive_losses",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    recentOutcomes: Array(RECENT_LOSS_BLOCK_COUNT).fill(0),
    portfolio: EMPTY_PORTFOLIO,
    expectedSkipReason: "consecutive_losses",
  },
  {
    name: "portfolio_sector_cap",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    regime: "trending_up",
    trendBias: "bullish",
    // Pre-load the sector with another majors coin so the new sizing
    // pushes the share past the 0.50 cap. Numbers picked so sizing
    // actually completes (cash + risk budget both fit) and the trip is
    // the sector cap, not sizing_too_small or the risk-budget clamp:
    //   tier=0.22 of 1000 = 220; min(220, 600*0.8=480, 300)=220.
    //   invested+220 = 620, /1000 = 0.62 < 0.75 → no risk-budget clamp.
    //   post-fee = 219.78. ETH(majors) 400 + BTC(majors) 219.78 = 619.78
    //   → share 0.6198 > 0.50 sector cap → portfolio_sector_cap.
    portfolio: {
      equityUsd: 1_000,
      cashUsd: 600,
      openPositions: [
        {
          coinId: "ethereum",
          direction: "up",
          notionalUsd: 400,
          regimeAtEntry: "trending_up",
        },
      ],
    },
    expectedSkipReason: "portfolio_sector_cap",
  },
  // ── fee_gate_tp_floor + gate_min_tp_distance_pct override ──────────────
  // ATR-derived tp_distance_pct (~3.0%) is well above the baseline 0.6%
  // floor on every timeframe, so the only way to exercise this branch is
  // through the live-tuner override. Doubles as coverage of the
  // gate_min_tp_distance_pct override path.
  {
    name: "fee_gate_tp_floor_via_override",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: EMPTY_PORTFOLIO,
    gateMinTpDistancePct: 0.05,
    expectedSkipReason: "fee_gate_tp_floor",
  },
  // ── fee_gate_ev + gate_min_ev_vs_cost override ─────────────────────────
  // Confidence (0.65) × tp_distance_pct (~3.0%) ≈ 0.0195. Round-trip cost
  // is 0.0025, so the baseline 3.0× requirement (0.0075) clears easily.
  // Push the EV multiplier high enough to fail and we exercise both the
  // fee_gate_ev branch and the gate_min_ev_vs_cost override path.
  {
    name: "fee_gate_ev_via_override",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: EMPTY_PORTFOLIO,
    gateMinEvVsCost: 10.0,
    expectedSkipReason: "fee_gate_ev",
  },
  // ── gate_min_confidence override ───────────────────────────────────────
  // Direction emits with confidence 0.65 — fine on the baseline (0.50/
  // 0.55 asymmetric). Push the per-call min above 0.65 and the gate
  // override path is what trips confidence_below_threshold.
  {
    name: "confidence_below_threshold_via_override",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: EMPTY_PORTFOLIO,
    gateMinConfidence: 0.99,
    expectedSkipReason: "confidence_below_threshold",
  },
  // ── gate_counter_trend_min_confidence override ─────────────────────────
  // Confidence (0.80) is ABOVE the baseline counter-trend min (0.65),
  // so on the baseline this fixture would APPROVE the short. The
  // override pushes the bar to 0.90 and the engine must skip. If
  // either side ignored the override, this fixture would diverge —
  // the TS mirror would approve, Python would skip → parity fails.
  {
    name: "counter_trend_via_override",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.10,
    probDown: 0.80,
    probStable: 0.10,
    expectedReturnPct: -0.5,
    regime: "trending_up",
    trendBias: "bullish",
    portfolio: EMPTY_PORTFOLIO,
    gateCounterTrendMinConfidence: 0.90,
    expectedSkipReason: "counter_trend_regime",
  },
  // ── portfolio_beta_cap ─────────────────────────────────────────────────
  // Existing ETH position carries an outsized β=5.0; adding a new BTC
  // (default β=1.0) still lifts the notional-weighted book β well above
  // the 1.50 cap. Sizing fits under sector + correlated caps so beta is
  // the gate that fires.
  {
    name: "portfolio_beta_cap",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: {
      equityUsd: 10_000,
      cashUsd: 8_000,
      openPositions: [
        {
          coinId: "ethereum",
          direction: "up",
          notionalUsd: 2_000,
          betaToBtc: 5.0,
        },
      ],
    },
    expectedSkipReason: "portfolio_beta_cap",
  },
  // ── portfolio_correlated_exposure (via portfolioConstraintsOverride) ──
  // Live config has max_sector_exposure_pct=0.50 < max_correlated_cap=
  // 0.60, so the sector cap always fires before the correlated cap can
  // be reached. Override flips that ordering: sector cap effectively
  // disabled (0.99), correlated cap tightened to 0.30, so two-coins-
  // in-majors with combined share ~0.42 trips the correlated branch.
  {
    name: "portfolio_correlated_exposure",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: {
      equityUsd: 10_000,
      cashUsd: 8_000,
      openPositions: [
        {
          coinId: "ethereum",
          direction: "up",
          notionalUsd: 2_000,
        },
      ],
    },
    portfolioConstraintsOverride: {
      max_sector_exposure_pct: 0.99,
      max_correlated_exposure_pct: 0.30,
    },
    expectedSkipReason: "portfolio_correlated_exposure",
  },
  // ── portfolio_regime_budget ────────────────────────────────────────────
  // Sector and beta clear (existing position is in `payments`, not
  // `majors`, with default β=1.0). The existing 5500 + the sized add
  // pushes the share of `trending_up` notional past the 0.70 budget.
  {
    name: "portfolio_regime_budget",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    regime: "trending_up",
    portfolio: {
      equityUsd: 10_000,
      cashUsd: 4_500,
      openPositions: [
        {
          coinId: "ripple",
          direction: "up",
          notionalUsd: 5_500,
          regimeAtEntry: "trending_up",
        },
      ],
    },
    expectedSkipReason: "portfolio_regime_budget",
  },
  // ── max_open_positions ─────────────────────────────────────────────────
  // Book is already at the per-agent cap (4) — sizing cannot proceed.
  {
    name: "max_open_positions",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: {
      equityUsd: 10_000,
      cashUsd: 8_000,
      openPositions: [
        { coinId: "solana",                  direction: "up", notionalUsd: 500 },
        { coinId: "uniswap",                 direction: "up", notionalUsd: 500 },
        { coinId: "ripple",                  direction: "up", notionalUsd: 500 },
        { coinId: "worldcoin-wld",           direction: "up", notionalUsd: 500 },
      ],
    },
    expectedSkipReason: "max_open_positions",
  },
  // ── sizing_too_small ───────────────────────────────────────────────────
  // Equity so small that even at the highest tier (22%) the post-fee
  // size lands below $1.00 → sizing_too_small.
  {
    name: "sizing_too_small",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: { equityUsd: 2, cashUsd: 2, openPositions: [] },
    expectedSkipReason: "sizing_too_small",
  },
  // ── task #345 — sizing parity (live ↔ backtest) ────────────────────────
  // The fixtures below exercise the post-decide wrapper that mirrors
  // paper-trader.ts's two live-only safeguards (fleet brake + Kelly).
  // Both Python (apply_live_extras) and TS (applyLiveExtras) implement
  // the SAME math, and the test asserts equality. If either side
  // drifts — or the live trader changes the math without updating the
  // wrapper — these fixtures fire immediately.
  //
  // Coverage: kelly_cap_hit, kelly_active_blended, fleet_brake_hit,
  // tuning_loosens_gate (gate dynamically loosened by the tuner so a
  // sub-baseline confidence still approves), and a normal-path fixture
  // that carries inert fleet/kelly state to prove the wrapper is a
  // no-op when neither safeguard fires.

  // Kelly fires and saturates at MAX_POSITION_PCT (0.30 of equity).
  // 1d timeframe: tfMult=1.7. With winRate=0.9, R=5, kellyPct=0.88,
  // fractional=0.308, confMult=0.8+0.65*0.6=1.19, raw=0.308*1.19*1.7=
  // 0.6228 → clamped to MAX_POSITION_PCT=0.30, kellySize=300. caps:
  // min(300, 800, 300)=300, no risk-budget clamp (0.30<0.75), entry
  // fee=300*0.001=0.30, post-fee≈299.70. Without the wrapper the
  // engine would emit tier-based 220 instead, so this fixture proves
  // the Kelly wrapper is ACTIVE on both sides.
  {
    name: "kelly_cap_hit_via_max_position_pct",
    coinId: "bitcoin",
    timeframe: "1d",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 1.5,
    portfolio: { equityUsd: 1_000, cashUsd: 1_000, openPositions: [] },
    kellyState: {
      totalTrades: 200,        // ≥ KELLY_RAMP_END (150) → pure Kelly path
      winningTrades: 180,
      avgWinPct: 5.0,
      avgLossPct: 1.0,
    },
    expectedSkipReason: null,
    expectedPositionSizeUsd: 299.70,
  },
  // Kelly active in the [KELLY_MIN_TRADES, KELLY_RAMP_END) blend window.
  // 1h timeframe: tfMult=0.8, winRate=0.6 (120/200), R=2, kellyPct=
  // (0.6*2 - 0.4)/2 = 0.4, fractional=0.4*0.35=0.14, confMult=
  // 0.8+0.65*0.6=1.19, raw=0.14*1.19*0.8=0.1333. Below MAX_POSITION_PCT
  // so kept at 0.1333. Wait — totalTrades=125 sits inside the ramp:
  // ramp=(125-100)/(150-100)=0.5. fixedSize=1000*0.22=220, kellySize=
  // 1000*0.1333=133.28. positionSize=220*0.5+133.28*0.5=176.64.
  // caps: min(176.64, 800, 300)=176.64; no risk-budget clamp; post-fee=
  // 176.64*(1-0.001)=176.46336.
  {
    name: "kelly_active_ramp_blend",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: { equityUsd: 1_000, cashUsd: 1_000, openPositions: [] },
    kellyState: {
      totalTrades: 125,
      winningTrades: 75,        // winRate=0.6
      avgWinPct: 2.0,
      avgLossPct: 1.0,
    },
    expectedSkipReason: null,
    expectedPositionSizeUsd: 176.46336,
  },
  // Fleet correlation brake fires: 6 open positions, 4 on the same
  // ("up") side as the candidate → sameSideShare=0.667 ≥ 0.6 → brake.
  // The engine itself would have emitted a long; the wrapper turns
  // that into the live-trader's `fleet_direction_imbalance` skip.
  {
    name: "fleet_brake_hit",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    portfolio: EMPTY_PORTFOLIO,
    fleetState: { up: 4, down: 2 },
    expectedSkipReason: "fleet_direction_imbalance",
  },
  // Gate dynamically loosened by the tuning-tracker: baseline min-conf
  // for shorts is 0.50, so a 0.45 short would skip with
  // `confidence_below_threshold`. The override drops the bar to 0.40
  // so the same input now approves on BOTH sides. If either engine
  // ignored the override the fixture would diverge (one approves,
  // one skips) and parity would fail.
  {
    name: "tuning_loosens_min_confidence_short_approves",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.20,
    probDown: 0.45,
    probStable: 0.35,
    expectedReturnPct: -0.5,
    portfolio: EMPTY_PORTFOLIO,
    gateMinConfidence: 0.40,
    expectedSkipReason: null,
  },
  // Kelly raw size would be 300 (cap), but cash=200 forces the
  // MAX_CASH_PER_POSITION_PCT (0.80) cap to bind first: cash*0.80=160.
  // No risk-budget clamp (160/1000=0.16 < 0.75). Fee=0.16, post-fee=
  // 159.84. Guards against a future change that drops the cash cap.
  {
    name: "kelly_cap_hit_then_cash_cap_binds",
    coinId: "bitcoin",
    timeframe: "1d",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 1.5,
    portfolio: { equityUsd: 1_000, cashUsd: 200, openPositions: [] },
    kellyState: {
      totalTrades: 200,
      winningTrades: 180,
      avgWinPct: 5.0,
      avgLossPct: 1.0,
    },
    expectedSkipReason: null,
    expectedPositionSizeUsd: 159.84,
  },
  // Normal path with INERT live-extras state: kellyState below
  // KELLY_MIN_TRADES → wrapper takes the no-op branch; fleetState
  // below FLEET_BRAKE_MIN_OPEN → wrapper takes the no-op branch. The
  // result must match the unwrapped `approve_long` size exactly.
  {
    name: "normal_path_with_inert_kelly_and_fleet_state",
    coinId: "bitcoin",
    timeframe: "1h",
    lastPrice: 50_000,
    atrValue: 500,
    probUp: 0.65,
    probDown: 0.20,
    probStable: 0.15,
    expectedReturnPct: 0.5,
    regime: "trending_up",
    trendBias: "bullish",
    portfolio: EMPTY_PORTFOLIO,
    kellyState: {
      totalTrades: 50,         // < KELLY_MIN_TRADES (100)
      winningTrades: 30,
      avgWinPct: 1.0,
      avgLossPct: 1.0,
    },
    fleetState: { up: 2, down: 1 },  // total=3 < FLEET_BRAKE_MIN_OPEN (6)
    expectedSkipReason: null,
  },
];

// ───────────────────────────────────────────────────────────────────────────
// Side (a): Python decide() via subprocess.
// ───────────────────────────────────────────────────────────────────────────
interface DecisionOut {
  action: "long" | "short" | "no_trade";
  confidence: number;
  sizeMultiplier: number;
  positionSizeUsd: number;
  direction: "up" | "down" | null;
  slPrice: number | null;
  tpPrice: number | null;
  skipReason: string | null;
  skipDetail: string | null;
}

function findRepoRoot(): string {
  let cur = path.dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 8; i++) {
    try {
      readFileSync(path.join(cur, "pnpm-workspace.yaml"));
      return cur;
    } catch {
      const parent = path.dirname(cur);
      if (parent === cur) break;
      cur = parent;
    }
  }
  throw new Error("could not locate workspace root from " + import.meta.url);
}

function pythonDecide(fixtures: Fixture[]): DecisionOut[] {
  const root = findRepoRoot();
  const py =
    process.env.PYTHON_BIN ??
    path.join(root, ".pythonlibs", "bin", "python3");
  const script = path.join(
    root,
    "artifacts",
    "ml-engine",
    "scripts",
    "decide_for_parity.py",
  );
  const mlRoot = path.join(root, "artifacts", "ml-engine");
  const res = spawnSync(py, [script], {
    input: JSON.stringify(fixtures),
    cwd: mlRoot,
    env: {
      ...process.env,
      PYTHONPATH: mlRoot + (process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ""),
    },
    encoding: "utf8",
    timeout: 30_000,
  });
  if (res.status !== 0) {
    throw new Error(
      `python decide_for_parity.py failed (status=${res.status}): ${res.stderr}`,
    );
  }
  return JSON.parse(res.stdout) as DecisionOut[];
}

// ───────────────────────────────────────────────────────────────────────────
// Side (b): TypeScript live-trader gate stack.
//
// Mirrors `decide()` step-by-step using the SAME exported helpers the
// real paper-trader/quant-brain code paths use, so the test acts as a
// machine-checked assertion that the two ports stay in lockstep. Order
// is identical to engine.py so a divergence on any single gate fires
// before the others can mask it.
// ───────────────────────────────────────────────────────────────────────────
function tsDecide(fx: Fixture): DecisionOut {
  const skip = (reason: string, detail = ""): DecisionOut => ({
    action: "no_trade",
    confidence: 0,
    sizeMultiplier: 0,
    positionSizeUsd: 0,
    direction: null,
    slPrice: null,
    tpPrice: null,
    skipReason: reason,
    skipDetail: detail,
  });

  // 1. Direction emit (mirror of engine.decide_direction).
  const mdp = QUANT_MIN_DIRECTIONAL_PROB;
  const mde = QUANT_MIN_DIRECTIONAL_EDGE;
  const mer = getMinExpectedReturnPct();
  const dirSide: "up" | "down" = fx.probUp >= fx.probDown ? "up" : "down";
  const dirProb = Math.max(fx.probUp, fx.probDown);
  const dirEdge = Math.abs(fx.probUp - fx.probDown);
  const expSign =
    fx.expectedReturnPct > 0 ? "up" : fx.expectedReturnPct < 0 ? "down" : "stable";

  if (dirProb < mdp) return skip("abstain_low_directional_prob");
  if (dirEdge < mde) return skip("abstain_no_directional_edge");
  if (Math.abs(fx.expectedReturnPct) < mer) return skip("abstain_exp_ret_below_cost");
  if (expSign !== dirSide) return skip("abstain_exp_ret_disagrees");

  const direction = dirSide;
  const confidence = direction === "up" ? fx.probUp : fx.probDown;

  // 2. Min-confidence gate (asymmetric for longs).
  const gMinConf = fx.gateMinConfidence ?? getMinConfidenceToTrade();
  const minConf =
    direction === "up"
      ? Math.max(gMinConf, ASYMMETRIC_LONG_MIN_CONFIDENCE)
      : gMinConf;
  if (confidence < minConf) {
    return skip(
      "confidence_below_threshold",
      `${confidence.toFixed(3)}<${minConf.toFixed(3)}`,
    );
  }

  // 3. Counter-trend gate.
  const ctMin = fx.gateCounterTrendMinConfidence ?? getCounterTrendMinConfidence();
  if (fx.trendBias === "bullish" && direction === "down" && confidence < ctMin) {
    return skip("counter_trend_regime");
  }
  if (fx.trendBias === "bearish" && direction === "up" && confidence < ctMin) {
    return skip("counter_trend_regime");
  }

  // 4. Recent-loss block.
  const consecLossN = RECENT_LOSS_BLOCK_COUNT;
  const recent = fx.recentOutcomes ?? [];
  if (
    recent.length >= consecLossN &&
    recent.slice(-consecLossN).every((w) => w === 0)
  ) {
    return skip(
      "consecutive_losses",
      `${consecLossN} losses in a row on ${fx.coinId}`,
    );
  }

  // 5. SL/TP geometry & EV gate (post-slippage).
  if (fx.lastPrice <= 0) return skip("no_entry_price");
  const adjEntry = applyEntrySlippage(fx.lastPrice, direction);
  const atrFloorPct = getAtrFloorPct(fx.timeframe);
  const fallbackFloor = adjEntry * atrFloorPct;
  const effectiveAtr = Math.max(fx.atrValue, fallbackFloor);
  const slMult = getSlMultiplier(fx.timeframe);
  const tpMult = getTpMultiplier(fx.timeframe);
  const slDistance = effectiveAtr * slMult;
  const tpDistance = effectiveAtr * tpMult;
  const tpDistancePct = tpDistance / adjEntry;
  const minTp = fx.gateMinTpDistancePct ?? getMinTpDistancePct();
  if (tpDistancePct < minTp) {
    return skip(
      "fee_gate_tp_floor",
      `${tpDistancePct.toFixed(4)}<${minTp.toFixed(4)}`,
    );
  }
  const evScore = confidence * tpDistancePct;
  const evRequired = (fx.gateMinEvVsCost ?? getMinEvVsCost()) * ROUND_TRIP_COST_PCT;
  if (evScore < evRequired) {
    return skip(
      "fee_gate_ev",
      `${evScore.toFixed(4)}<${evRequired.toFixed(4)}`,
    );
  }
  const slPrice =
    direction === "up" ? adjEntry - slDistance : adjEntry + slDistance;
  const tpPrice =
    direction === "up" ? adjEntry + tpDistance : adjEntry - tpDistance;

  // 6. Sizing.
  const portfolio = fx.portfolio ?? EMPTY_PORTFOLIO;
  const equity = portfolio.equityUsd;
  const cash = portfolio.cashUsd;
  const invested = portfolio.openPositions.reduce((s, p) => s + p.notionalUsd, 0);
  if (portfolio.openPositions.length >= MAX_OPEN_POSITIONS_PER_AGENT) {
    return skip("max_open_positions");
  }
  const tierPct = tieredPositionPct(confidence);
  let positionSize = equity * tierPct;
  positionSize = Math.min(positionSize, cash * 0.80, equity * MAX_POSITION_PCT);
  if ((invested + positionSize) / Math.max(equity, 1e-9) >= MAX_PORTFOLIO_AT_RISK) {
    positionSize = Math.max(0, equity * MAX_PORTFOLIO_AT_RISK - invested);
  }
  const entryFee = positionSize * TAKER_FEE_PCT;
  const positionSizePostFee = positionSize - entryFee;
  if (positionSizePostFee < 1.0) return skip("sizing_too_small");

  // 7. Portfolio constraints (sector / correlated / beta / regime).
  const openPositions: PortfolioOpenPosition[] = portfolio.openPositions.map((p) => ({
    coinId: p.coinId,
    direction: p.direction,
    notionalUsd: p.notionalUsd,
    regimeAtEntry: p.regimeAtEntry ?? null,
    betaToBtc: p.betaToBtc ?? null,
  }));
  const pc = checkPortfolioConstraints({
    coinId: fx.coinId,
    newNotionalUsd: positionSizePostFee,
    equityUsd: equity,
    regime: fx.regime ?? null,
    openPositions,
    overrides: fx.portfolioConstraintsOverride,
  });
  if (!pc.ok) return skip(pc.skipReason ?? "portfolio_constraint", "");

  return {
    action: direction === "up" ? "long" : "short",
    confidence,
    sizeMultiplier: tierPct,
    positionSizeUsd: positionSizePostFee,
    direction,
    slPrice,
    tpPrice,
    skipReason: null,
    skipDetail: null,
  };
}

// ───────────────────────────────────────────────────────────────────────────
// Side (b) wrapper: live-only safeguards (task #345).
//
// `decide()` is the SHARED engine — it intentionally doesn't model the
// fleet correlation brake or Kelly sizing because those are paper-trader
// concerns the offline backtester would otherwise re-implement and
// drift on. The wrapper below mirrors paper-trader.ts exactly, and the
// Python helper (`scripts/decide_for_parity.py`) carries the same code.
// Both sides MUST agree on positionSizeUsd for the same `(decide gate,
// fleet state, kelly state)` input. If they disagree, the test fires.
// ───────────────────────────────────────────────────────────────────────────
const TF_KELLY_MULT: Record<string, number> = {
  "5m": 0.7, "1h": 0.8, "2h": 1.2, "6h": 1.5, "1d": 1.7,
};
function calculateKellySize(
  winRate: number, avgWin: number, avgLoss: number,
  portfolioValue: number, confidence: number, timeframe: string,
): number {
  // Mirror of paper-trader.ts:calculateKellySize.
  if (avgLoss <= 0) avgLoss = 0.1;
  if (avgWin <= 0) avgWin = 0.1;
  const R = avgWin / avgLoss;
  const kellyPct = (winRate * R - (1 - winRate)) / R;
  if (kellyPct <= 0) return 0;
  const fractional = kellyPct * KELLY_FRACTION;
  const confMult = 0.8 + confidence * 0.6;
  const tfMult = TF_KELLY_MULT[timeframe] ?? 1.0;
  let pct = fractional * confMult * tfMult;
  pct = Math.min(pct, MAX_POSITION_PCT);
  if (pct < 0.03) return 0;
  return portfolioValue * pct;
}
function applyLiveExtras(fx: Fixture, base: DecisionOut): DecisionOut {
  if (base.action === "no_trade") return base;
  const direction = base.direction;
  if (direction === null) return base;

  // Fleet brake (paper-trader.ts:384-412) — fires BEFORE sizing; we
  // override the approve to a typed skip if it would fire.
  if (fx.fleetState) {
    const total = fx.fleetState.up + fx.fleetState.down;
    if (total >= FLEET_BRAKE_MIN_OPEN) {
      const sameSide = direction === "up" ? fx.fleetState.up : fx.fleetState.down;
      const share = total > 0 ? sameSide / total : 0;
      if (share >= FLEET_BRAKE_DOMINANCE) {
        return {
          action: "no_trade",
          confidence: 0,
          sizeMultiplier: 0,
          positionSizeUsd: 0,
          direction: null,
          slPrice: null,
          tpPrice: null,
          skipReason: "fleet_direction_imbalance",
          skipDetail: `sameSideShare=${share.toFixed(3)}>=${FLEET_BRAKE_DOMINANCE}`,
        };
      }
    }
  }

  // Kelly sizing override (paper-trader.ts:491-522).
  if (
    fx.kellyState &&
    fx.kellyState.totalTrades >= KELLY_MIN_TRADES &&
    fx.kellyState.winningTrades > 0
  ) {
    const portfolio = fx.portfolio ?? EMPTY_PORTFOLIO;
    const equity = portfolio.equityUsd;
    const cash = portfolio.cashUsd;
    const invested = portfolio.openPositions.reduce(
      (s, p) => s + p.notionalUsd, 0,
    );
    const winRate = fx.kellyState.winningTrades / fx.kellyState.totalTrades;
    const kellySize = calculateKellySize(
      winRate, fx.kellyState.avgWinPct, fx.kellyState.avgLossPct,
      equity, base.confidence, fx.timeframe,
    );
    let pos: number;
    if (fx.kellyState.totalTrades < KELLY_RAMP_END) {
      const ramp =
        (fx.kellyState.totalTrades - KELLY_MIN_TRADES) /
        (KELLY_RAMP_END - KELLY_MIN_TRADES);
      const fixedSize = equity * tieredPositionPct(base.confidence);
      pos = fixedSize * (1 - ramp) + kellySize * ramp;
    } else {
      pos = kellySize;
    }
    pos = Math.min(
      pos, cash * MAX_CASH_PER_POSITION_PCT, equity * MAX_POSITION_PCT,
    );
    if (equity > 0 && (invested + pos) / equity >= MAX_PORTFOLIO_AT_RISK) {
      pos = Math.max(0, equity * MAX_PORTFOLIO_AT_RISK - invested);
    }
    const fee = pos * TAKER_FEE_PCT;
    return { ...base, positionSizeUsd: pos - fee };
  }

  return base;
}

// ───────────────────────────────────────────────────────────────────────────
// Suite.
// ───────────────────────────────────────────────────────────────────────────
describe("decision-engine parity — Python decide() ⇔ TS live-trader gate stack", () => {
  const SIZE_EPS_USD = 0.01;
  const PRICE_EPS = 1e-6;

  // Compute Python answers once for the whole batch — keeps the
  // subprocess overhead at a single spawn for the suite.
  const pyResults = pythonDecide(FIXTURES);

  for (let i = 0; i < FIXTURES.length; i++) {
    const fx = FIXTURES[i];
    const py = pyResults[i];
    it(`fixture ${fx.name} produces the same action / skip / size`, () => {
      const ts = applyLiveExtras(fx, tsDecide(fx));
      const ctx = `fixture=${fx.name} python=${JSON.stringify(py)} ts=${JSON.stringify(ts)}`;
      if (fx.expectedPositionSizeUsd !== undefined) {
        // Pin the wrapper-owned positionSizeUsd so a refactor that
        // accidentally turns Kelly into a no-op (or doubles its
        // effect) is caught even before the cross-engine compare.
        assert.ok(
          Math.abs(ts.positionSizeUsd - fx.expectedPositionSizeUsd) <= 0.01,
          `fixture ${fx.name} positionSizeUsd ${ts.positionSizeUsd} != expected ${fx.expectedPositionSizeUsd}`,
        );
      }
      assert.equal(ts.action, py.action, `action diverged — ${ctx}`);
      assert.equal(
        ts.skipReason,
        py.skipReason,
        `skipReason diverged — ${ctx}`,
      );
      // Belt-and-suspenders: assert each fixture lands on the SPECIFIC
      // skip reason it was designed to exercise. Without this a fixture
      // that silently degrades (e.g. tipping into sizing_too_small
      // before reaching the gate it claims to test) would still pass
      // parity — both engines would agree on the wrong reason. The
      // explicit check fires the moment the targeted branch stops being
      // the one that runs.
      assert.equal(
        py.skipReason,
        fx.expectedSkipReason,
        `fixture ${fx.name} produced skipReason=${py.skipReason} but was authored to exercise ${fx.expectedSkipReason}`,
      );
      assert.ok(
        Math.abs(ts.positionSizeUsd - py.positionSizeUsd) <= SIZE_EPS_USD,
        `positionSizeUsd diverged (|Δ|=${Math.abs(ts.positionSizeUsd - py.positionSizeUsd)}) — ${ctx}`,
      );
      if (py.action !== "no_trade") {
        assert.equal(ts.direction, py.direction, `direction diverged — ${ctx}`);
        assert.ok(
          Math.abs((ts.slPrice ?? 0) - (py.slPrice ?? 0)) <=
            Math.max(PRICE_EPS, Math.abs(py.slPrice ?? 1) * 1e-9),
          `slPrice diverged — ${ctx}`,
        );
        assert.ok(
          Math.abs((ts.tpPrice ?? 0) - (py.tpPrice ?? 0)) <=
            Math.max(PRICE_EPS, Math.abs(py.tpPrice ?? 1) * 1e-9),
          `tpPrice diverged — ${ctx}`,
        );
      }
    });
  }
});
