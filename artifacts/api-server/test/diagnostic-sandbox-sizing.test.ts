import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import {
  MTTM_DIAGNOSTIC_SANDBOX_COIN,
  MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
  MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
  MTTM_DIAGNOSTIC_SANDBOX_MAX_OPEN_POSITIONS,
  MTTM_DIAGNOSTIC_SANDBOX_ROLLING_WINDOW_TRADES,
} from "../src/lib/mttm";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PAPER_TRADER_SRC = readFileSync(
  resolve(__dirname, "../src/lib/paper-trader.ts"),
  "utf8",
);
const MTTM_SRC = readFileSync(
  resolve(__dirname, "../src/lib/mttm.ts"),
  "utf8",
);

describe("Task #659 — DS lane sizing constants", () => {
  test("fixed position pct is 0.5%", () => {
    assert.equal(MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT, 0.005);
  });
  test("universe pin is bitcoin/5m", () => {
    assert.equal(MTTM_DIAGNOSTIC_SANDBOX_COIN, "bitcoin");
    assert.equal(MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME, "5m");
  });
  test("loss-limit defaults: -5% drawdown, n=50", () => {
    assert.equal(MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT, -0.05);
    assert.equal(MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL, 50);
  });
  test("max-open + rolling review window are operator-visible constants", () => {
    assert.equal(MTTM_DIAGNOSTIC_SANDBOX_MAX_OPEN_POSITIONS, 1);
    assert.equal(MTTM_DIAGNOSTIC_SANDBOX_ROLLING_WINDOW_TRADES, 100);
  });
});

describe("Task #659 — paper-trader DS branches (source assertions)", () => {
  test("_isDsLane is set from cfg.enabled && cfg.mode === 'diagnostic_sandbox'", () => {
    assert.match(
      PAPER_TRADER_SRC,
      /const\s+_isDsLane\s*=\s*_mttmActiveCfg\.enabled[\s\S]{0,80}_mttmActiveCfg\.mode\s*===\s*"diagnostic_sandbox"/,
    );
  });

  test("DS branch sets positionSize = totalValue * fixedPositionPct (no Kelly)", () => {
    assert.match(
      PAPER_TRADER_SRC,
      /if\s*\(_isDsLane\)\s*\{\s*positionSize\s*=\s*p\.totalValue\s*\*\s*_mttmActiveCfg\.diagnosticSandbox\.fixedPositionPct/,
    );
  });

  test("composite multiplier chain is gated on `!_isDsLane`", () => {
    // Production code at paper-trader.ts:875 reads:
    //   if (!_isDsLane && executionSizeMultiplier !== 1.0) { ... }
    assert.match(
      PAPER_TRADER_SRC,
      /if\s*\(\s*!_isDsLane\s*&&\s*executionSizeMultiplier\s*!==\s*1\.0\s*\)/,
    );
  });

  test("portfolio-at-risk cap is gated on `!_isDsLane`", () => {
    assert.match(
      PAPER_TRADER_SRC,
      /if\s*\(\s*!_isDsLane\s*&&[\s\S]{0,120}MAX_PORTFOLIO_AT_RISK/,
    );
  });

  test("MTTM/global per-position cap path is gated on `!_isDsLane` (DS bypasses Math.min cap)", () => {
    // Production code at paper-trader.ts:909-916: the `else` branch of
    // `if (_isDsLane)` is the default-lane Math.min cap that uses
    // `_maxPositionPct`. DS lane never hits this branch.
    assert.match(
      PAPER_TRADER_SRC,
      /\}\s*else\s*\{[\s\S]{0,300}positionSize\s*=\s*Math\.min\([\s\S]{0,200}_maxPositionPct/,
    );
  });

  test("DS final pin re-assignment is `positionSize = dsPin`", () => {
    assert.match(PAPER_TRADER_SRC, /positionSize\s*=\s*dsPin\s*;/);
  });

  test("DS off-universe skip uses reason `diagnostic_universe_locked`", () => {
    assert.match(
      PAPER_TRADER_SRC,
      /isDsLane\s*\?\s*"diagnostic_universe_locked"\s*:\s*"mttm_outside_universe"/,
    );
  });

  test("DS cash-sufficiency skip uses reason `ds_insufficient_cash` (no silent shrink)", () => {
    const matches = PAPER_TRADER_SRC.match(/"ds_insufficient_cash"/g) ?? [];
    assert.ok(
      matches.length >= 2,
      `expected at least 2 ds_insufficient_cash recordSkip sites, got ${matches.length}`,
    );
  });

  test("DS cash-debit branch only runs when `_isDsLane`; default lane debits fee from notional", () => {
    assert.match(
      PAPER_TRADER_SRC,
      /if\s*\(\s*!_isDsLane\s*\)\s*\{[\s\S]{0,200}positionSize\s*=\s*positionSize\s*-\s*entryFee/,
    );
    assert.match(
      PAPER_TRADER_SRC,
      /\}\s*else\s*\{\s*\/\/\s*Task #659[\s\S]{0,400}p\.cashBalance\s*<\s*positionSize\s*\+\s*entryFee/,
    );
  });

  test("DS invariant guard: cap-stage with off-pin (coin,timeframe) throws", () => {
    assert.match(PAPER_TRADER_SRC, /DS invariant: cap-stage reached with/);
  });
});

describe("Task #659 — v1 row contract emits all required operator fields", () => {
  test("buildFullDiagnosticSandboxRow returns label/universe/fixed_position_pct", () => {
    assert.match(MTTM_SRC, /label:\s*getDiagnosticSandboxLabel\(\)/);
    assert.match(MTTM_SRC, /coin_id:\s*MTTM_DIAGNOSTIC_SANDBOX_COIN/);
    assert.match(
      MTTM_SRC,
      /fixed_position_pct:\s*MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT/,
    );
  });
  test("v1 row exposes loss_limits {drawdown_floor_pct, n_neg_pnl_threshold}", () => {
    assert.match(
      MTTM_SRC,
      /loss_limits:\s*\{\s*drawdown_floor_pct:\s*dd,\s*n_neg_pnl_threshold:\s*nNeg\s*\}/,
    );
  });
  test("v1 row exposes review_windows {initial_review_n_trades, rolling_window_trades}", () => {
    assert.match(MTTM_SRC, /initial_review_n_trades:\s*Math\.max/);
    assert.match(MTTM_SRC, /rolling_window_trades:\s*Math\.max/);
  });
  test("v1 row exposes max_open_positions", () => {
    assert.match(
      MTTM_SRC,
      /max_open_positions:\s*MTTM_DIAGNOSTIC_SANDBOX_MAX_OPEN_POSITIONS/,
    );
  });
  test("v1 row exposes review {enabled_at, disable_reason, auto_disabled}", () => {
    assert.match(MTTM_SRC, /enabled_at:\s*partial\.review\?\.enabled_at/);
    assert.match(MTTM_SRC, /disable_reason:\s*partial\.review\?\.disable_reason/);
    assert.match(MTTM_SRC, /auto_disabled:\s*partial\.review\?\.auto_disabled/);
  });
});

describe("Task #659 — getMttmConfig regression guards", () => {
  test("default mode does NOT honour ds.enabled (legacy mttm_enabled remains authoritative)", () => {
    assert.doesNotMatch(
      MTTM_SRC,
      /else\s+if\s*\(\s*ds\.enabled\s*===\s*false\s*\)\s*\{[\s\S]{0,80}cfg\.enabled\s*=\s*false/,
    );
  });
  test("diagnostic_sandbox mode DOES honour ds.enabled (v1 row authoritative for DS)", () => {
    assert.match(
      MTTM_SRC,
      /if\s*\(\s*cfg\.mode\s*===\s*"diagnostic_sandbox"\s*\)\s*\{\s*cfg\.enabled\s*=\s*ds\.enabled\s*\?\?\s*true/,
    );
  });
});
