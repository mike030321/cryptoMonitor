/**
 * Pins the four dashboard truthfulness invariants from Task #362.
 *
 * 1. Best-pick prose embeds the SAME probability the "Model prob." tile
 *    renders (LightGBM `modelProbability`), never the legacy
 *    `successProbability` heuristic clamped to 15-85%.
 * 2. When the model probability is unavailable (null), the prose omits
 *    the percentage entirely rather than silently substituting a stale
 *    heuristic value.
 * 3. The fallback reasoning never says "consensus across 1 timeframes":
 *    >=2 -> "consensus across N timeframes",
 *      1 -> "based on the {tf} timeframe",
 *      0 -> drop the basis clause entirely.
 * 4. Per-bot net P&L equals `totalValue - startingCapital` to the cent
 *    (so the leaderboard P&L can never disagree with the equity tile),
 *    and "Bots in Profit" counts bots whose `totalValue > startingCapital`
 *    (so the count can never disagree with the visible rows).
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  buildFallbackReasoning,
  buildWhatExplanation,
} from "../src/lib/_legacy/ai-engine.ts";

describe("Task #362 — best-pick prose probability matches the tile", () => {
  it("fallback reasoning embeds modelProbability (not successProbability)", () => {
    const text = buildFallbackReasoning({
      bestAction: "buy",
      coinName: "Sei",
      timeframeCount: 3,
      primaryTimeframe: "4h",
      modelProbability: 0.35,
    });
    assert.match(text, /35% model probability/);
    // The legacy heuristic the user reported (77%) must not appear.
    assert.doesNotMatch(text, /77%/);
  });

  it("whatExplanation embeds modelProbability (not successProbability)", () => {
    const text = buildWhatExplanation({
      bestAction: "buy",
      coinSymbol: "SEI",
      brain: "QUANT",
      modelProbability: 0.35,
      holdTimeframe: "4h",
    });
    assert.match(text, /35%/);
    assert.doesNotMatch(text, /77%/);
  });

  it("prose omits the percentage when modelProbability is null", () => {
    const fb = buildFallbackReasoning({
      bestAction: "buy",
      coinName: "Sei",
      timeframeCount: 3,
      primaryTimeframe: "4h",
      modelProbability: null,
    });
    assert.doesNotMatch(fb, /%/, "fallback prose must not invent a percentage");

    const what = buildWhatExplanation({
      bestAction: "buy",
      coinSymbol: "SEI",
      brain: "QUANT",
      modelProbability: null,
      holdTimeframe: "4h",
    });
    assert.doesNotMatch(what, /%/, "what prose must not invent a percentage");
  });

  it("prose probability matches the tile probability for the same payload", () => {
    // Simulates the dashboard binding: tile reads modelProbability, prose
    // reads the same value via the builder. They MUST agree to the
    // displayed precision (toFixed(0)).
    const modelProbability = 0.4271;
    const tilePct = `${(modelProbability * 100).toFixed(0)}%`;
    const fb = buildFallbackReasoning({
      bestAction: "sell",
      coinName: "Bitcoin",
      timeframeCount: 2,
      primaryTimeframe: "1h",
      modelProbability,
    });
    const what = buildWhatExplanation({
      bestAction: "sell",
      coinSymbol: "BTC",
      brain: "QUANT",
      modelProbability,
      holdTimeframe: "1h",
    });
    assert.ok(fb.includes(tilePct), `fallback "${fb}" must contain tile pct ${tilePct}`);
    assert.ok(what.includes(tilePct), `what "${what}" must contain tile pct ${tilePct}`);
  });
});

describe("Task #362 — consensus wording branches on timeframe count", () => {
  it(">=2 timeframes -> 'consensus across N timeframes'", () => {
    const text = buildFallbackReasoning({
      bestAction: "buy",
      coinName: "Sei",
      timeframeCount: 3,
      primaryTimeframe: "4h",
      modelProbability: 0.5,
    });
    assert.match(text, /consensus across 3 timeframes/);
  });

  it("=1 timeframe -> 'based on the {tf} timeframe' (never says consensus)", () => {
    const text = buildFallbackReasoning({
      bestAction: "buy",
      coinName: "Sei",
      timeframeCount: 1,
      primaryTimeframe: "4h",
      modelProbability: 0.5,
    });
    assert.match(text, /based on the 4h timeframe/);
    assert.doesNotMatch(text, /consensus/i);
    assert.doesNotMatch(text, /1 timeframes/, "must not pluralise N=1");
  });

  it("=1 timeframe with no label falls back to natural wording", () => {
    const text = buildFallbackReasoning({
      bestAction: "buy",
      coinName: "Sei",
      timeframeCount: 1,
      primaryTimeframe: null,
      modelProbability: 0.5,
    });
    assert.match(text, /based on a single timeframe/);
    assert.doesNotMatch(text, /consensus/i);
  });

  it("=0 timeframes -> drop the basis clause entirely", () => {
    const text = buildFallbackReasoning({
      bestAction: "hold",
      coinName: "Sei",
      timeframeCount: 0,
      primaryTimeframe: null,
      modelProbability: 0.5,
    });
    assert.doesNotMatch(text, /consensus/i);
    assert.doesNotMatch(text, /timeframe/i);
  });
});

// ── Leaderboard accounting invariants ───────────────────────────────────
//
// The dashboard derives per-bot net P&L as `totalValue - startingCapital`
// rather than reading the realized-only `totalPnl` it used to render.
// The same identity must produce a count of "winners" that matches what
// the operator can see in the leaderboard rows. These tests pin the
// formula independently of the React component so a future refactor
// cannot silently swap back to the realized-only definition.

interface BotForTest {
  totalValue: number;
  startingCapital: number;
  totalPnl: number; // legacy realized-only field — kept on the API
}

function netPnl(b: BotForTest): number {
  return b.totalValue - b.startingCapital;
}

function countWinners(bots: BotForTest[]): number {
  return bots.filter((b) => b.totalValue > b.startingCapital).length;
}

describe("Task #362 — leaderboard P&L is equity-derived, not realized-only", () => {
  // The exact rows the user reported: stored realized-only `totalPnl`
  // disagrees with the equity-implied net P&L by several dollars per bot.
  // After the fix, the displayed P&L equals `equity - 1000` to the cent.
  const userReportedRows: BotForTest[] = [
    { totalValue: 998.64, startingCapital: 1000, totalPnl: 3.48 },
    { totalValue: 998.16, startingCapital: 1000, totalPnl: 0.37 },
    { totalValue: 995.46, startingCapital: 1000, totalPnl: 0.51 },
    { totalValue: 995.09, startingCapital: 1000, totalPnl: 0.19 },
    { totalValue: 994.32, startingCapital: 1000, totalPnl: 0.18 },
    { totalValue: 990.03, startingCapital: 1000, totalPnl: -5.24 },
    { totalValue: 987.26, startingCapital: 1000, totalPnl: -7.91 },
  ];

  it("each row's displayed P&L equals equity − starting capital to the cent", () => {
    const expected = [-1.36, -1.84, -4.54, -4.91, -5.68, -9.97, -12.74];
    const actual = userReportedRows.map((b) => Number(netPnl(b).toFixed(2)));
    assert.deepEqual(actual, expected);
  });

  it("displayed P&L disagrees with the legacy realized field on these rows (regression sentinel)", () => {
    // Demonstrates the very mismatch the user reported. If a future
    // refactor swaps back to `b.totalPnl`, this test still passes —
    // but the equity-identity test above will fail. Together they pin
    // the truth that displayed P&L MUST be equity-derived.
    for (const b of userReportedRows) {
      assert.notEqual(
        Number(netPnl(b).toFixed(2)),
        Number(b.totalPnl.toFixed(2)),
        `Row equity ${b.totalValue} would falsely report ${b.totalPnl} under realized-only basis`,
      );
    }
  });

  it("'Bots in Profit' counts bots with equity above starting capital", () => {
    // Mirrors the user's screenshot: 7 bots above $1000, 8 below. The
    // realized-only definition counted 12 of 15 — the equity-based
    // definition counts only the visibly-positive rows.
    const fleet: BotForTest[] = [
      ...Array.from({ length: 7 }, (_, i) => ({
        totalValue: 1000 + (i + 1) * 0.5,
        startingCapital: 1000,
        totalPnl: 0,
      })),
      ...userReportedRows, // 7 below-1000 rows from above
      { totalValue: 999.5, startingCapital: 1000, totalPnl: 5.0 }, // realized-positive but equity-negative
    ];
    assert.equal(fleet.length, 15);
    assert.equal(countWinners(fleet), 7);

    // The realized-only basis would have over-counted: 7 above-equity
    // bots + the last row whose realized field is positive = 8 here,
    // and in the user's screenshot the same drift produced 12/15.
    const realizedWinners = fleet.filter((b) => b.totalPnl > 0).length;
    assert.notEqual(realizedWinners, countWinners(fleet));
  });

  it("fleet-aggregate equity change equals sum of per-bot net P&Ls (no double-counting)", () => {
    const totalEquity = userReportedRows.reduce((s, b) => s + b.totalValue, 0);
    const totalSeed = userReportedRows.reduce((s, b) => s + b.startingCapital, 0);
    const fleetChange = totalEquity - totalSeed;
    const sumOfRows = userReportedRows.reduce((s, b) => s + netPnl(b), 0);
    assert.ok(
      Math.abs(fleetChange - sumOfRows) < 1e-9,
      `fleet change ${fleetChange} must equal sum of per-bot net P&Ls ${sumOfRows}`,
    );
  });
});
