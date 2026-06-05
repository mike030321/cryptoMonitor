// Task #505 — parity tests for the dashboard's MAE / stability helpers.
//
// `getPortfolioSummaries` augments each row in `recentTrades` with a
// `maePct` (true intra-trade max-adverse-excursion) and a `stability`
// score derived from the per-tick `paper_position_marks` stream. The
// authoritative math lives in `replay_meta_brain.py:_mae_from_marks`
// and `_stability_from_marks` — these tests pin the TS implementation
// to the same numerical outputs so the dashboard can never disagree
// with the meta-brain replay's `realized_drawdown` / `realized_stability`.
import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { maePctFromMarks, stabilityFromMarks } from "../src/lib/paper-trader.ts";

describe("Task #505 — maePctFromMarks", () => {
  test("LONG (buy) MAE is (entry - min) / entry, never negative", () => {
    // entry 100, min 92 → 8% drawdown
    const marks = [{ markPrice: 99 }, { markPrice: 95 }, { markPrice: 92 }, { markPrice: 105 }];
    const mae = maePctFromMarks(100, "buy", marks);
    assert.ok(mae !== null);
    assert.ok(Math.abs((mae as number) - 0.08) < 1e-9, `expected 0.08, got ${mae}`);
  });

  test("LONG MAE clamps to 0 when price never went below entry", () => {
    // Pure rip — every mark above entry. Replay returns max(0, ...) so
    // we mirror that floor instead of letting the percentage go negative.
    const marks = [{ markPrice: 101 }, { markPrice: 105 }, { markPrice: 110 }];
    assert.equal(maePctFromMarks(100, "buy", marks), 0);
  });

  test("SHORT (sell) MAE is (max - entry) / entry, never negative", () => {
    // entry 100, max 108 → 8% adverse for a short
    const marks = [{ markPrice: 101 }, { markPrice: 108 }, { markPrice: 95 }];
    const mae = maePctFromMarks(100, "sell", marks);
    assert.ok(mae !== null);
    assert.ok(Math.abs((mae as number) - 0.08) < 1e-9, `expected 0.08, got ${mae}`);
  });

  test("returns null when there are no marks (graceful '—' fallback)", () => {
    assert.equal(maePctFromMarks(100, "buy", []), null);
  });

  test("returns null on non-positive entry to avoid divide-by-zero", () => {
    assert.equal(maePctFromMarks(0, "buy", [{ markPrice: 1 }]), null);
    assert.equal(maePctFromMarks(-5, "sell", [{ markPrice: 1 }]), null);
  });

  test("returns null for actions other than buy/sell (e.g. 'close')", () => {
    // The reference Python impl only handles long/short; "close" rows
    // shouldn't synthesize a misleading MAE.
    assert.equal(maePctFromMarks(100, "close", [{ markPrice: 90 }]), null);
  });

  test("ignores non-finite mark prices (NaN / Infinity) instead of crashing", () => {
    const marks = [
      { markPrice: Number.NaN },
      { markPrice: 95 },
      { markPrice: Number.POSITIVE_INFINITY },
    ];
    const mae = maePctFromMarks(100, "buy", marks);
    assert.ok(mae !== null);
    assert.ok(Math.abs((mae as number) - 0.05) < 1e-9);
  });

  test("action match is case-insensitive (BUY / Sell)", () => {
    assert.ok((maePctFromMarks(100, "BUY", [{ markPrice: 90 }]) ?? 0) > 0);
    assert.ok((maePctFromMarks(100, "Sell", [{ markPrice: 110 }]) ?? 0) > 0);
  });
});

describe("Task #505 — stabilityFromMarks", () => {
  test("returns 1 for a perfectly flat hold (sigma = 0)", () => {
    const marks = [{ markPrice: 100 }, { markPrice: 100 }, { markPrice: 100 }, { markPrice: 100 }];
    assert.equal(stabilityFromMarks(marks), 1);
  });

  test("returns null when fewer than two valid returns exist", () => {
    // 0 marks → null
    assert.equal(stabilityFromMarks([]), null);
    // 1 mark → no returns → null
    assert.equal(stabilityFromMarks([{ markPrice: 100 }]), null);
    // 2 marks → 1 return → still null (Python requires >= 2 returns
    // for a meaningful stdev)
    assert.equal(stabilityFromMarks([{ markPrice: 100 }, { markPrice: 101 }]), null);
  });

  test("matches the Python reference 1 / (1 + 5*sigma) for a known sequence", () => {
    // Prices: 100, 102, 100, 102 → returns: 0.02, -0.0196..., 0.02
    // mean ≈ 0.00679, population stdev ≈ 0.018786 → expected ≈ 0.91399.
    const marks = [{ markPrice: 100 }, { markPrice: 102 }, { markPrice: 100 }, { markPrice: 102 }];
    const stability = stabilityFromMarks(marks);
    assert.ok(stability !== null);
    // Hand-computed expected value, recomputed inline to keep the
    // assertion self-checking even if the constants change.
    const rets = [
      (102 - 100) / 100,
      (100 - 102) / 102,
      (102 - 100) / 100,
    ];
    const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
    const variance =
      rets.reduce((a, b) => a + (b - mean) * (b - mean), 0) / rets.length;
    const sigma = Math.sqrt(variance);
    const expected = 1 / (1 + 5 * sigma);
    assert.ok(
      Math.abs((stability as number) - expected) < 1e-12,
      `expected ${expected}, got ${stability}`,
    );
  });

  test("clamps into [0, 1] for very volatile sequences", () => {
    // Wild swings → tiny stability, but never negative.
    const marks = [
      { markPrice: 100 },
      { markPrice: 200 },
      { markPrice: 50 },
      { markPrice: 300 },
      { markPrice: 25 },
    ];
    const stability = stabilityFromMarks(marks);
    assert.ok(stability !== null);
    assert.ok(stability! >= 0 && stability! <= 1);
    assert.ok(stability! < 0.2, `expected very low stability, got ${stability}`);
  });

  test("ignores non-positive prices in the return calc (no divide-by-zero)", () => {
    const marks = [
      { markPrice: 100 },
      { markPrice: 0 },
      { markPrice: 100 },
      { markPrice: 100 },
    ];
    // Should not throw; the 0 prev-price step is skipped.
    const stability = stabilityFromMarks(marks);
    assert.ok(stability === null || (stability >= 0 && stability <= 1));
  });
});
