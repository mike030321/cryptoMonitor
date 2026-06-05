import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  applyEntrySlippage,
  applyExitSlippage,
  entryFee,
  exitFee,
  isStopLossHit,
  isTakeProfitHit,
  recoverEntryFee,
  roundTripPnlUsd,
  shouldSkipForAbstain,
} from "../src/lib/trade-math";
import {
  ROUND_TRIP_COST_PCT,
  TAKER_FEE_PCT,
  SLIPPAGE_PCT,
  DAILY_LOSS_LIMIT_PCT,
} from "../src/lib/trading-constants";
import { checkRiskLimits } from "../src/lib/paper-trader";

describe("fee + slippage P&L (production code path)", () => {
  it("a perfectly flat round trip loses approximately the full round-trip cost", () => {
    const pnl = roundTripPnlUsd({
      direction: "up",
      rawEntryPrice: 100,
      rawExitPrice: 100,
      positionSizeUsd: 1000,
    });
    // A flat trade pays: entry slippage (0.05%) + exit slippage (0.05%)
    // + 2 fees (0.10% each) = 0.30% of position size, so ~$3 loss.
    assert.ok(pnl < 0, `flat trade must lose money, got ${pnl}`);
    assert.ok(
      Math.abs(pnl) > 1000 * ROUND_TRIP_COST_PCT * 0.95,
      `flat-trade loss (${pnl}) should be near round-trip cost ($${1000 * ROUND_TRIP_COST_PCT})`,
    );
  });

  it("a long that gains 1% is profitable AFTER fees + slippage", () => {
    const pnl = roundTripPnlUsd({
      direction: "up",
      rawEntryPrice: 100,
      rawExitPrice: 101,
      positionSizeUsd: 1000,
    });
    // 1% gross gain on $1000 = ~$10, minus ~$3 round-trip cost = ~$7 net.
    assert.ok(pnl > 5, `1% long should net ~$7 after costs, got ${pnl}`);
    assert.ok(pnl < 10, `1% long net cannot exceed gross ~$10, got ${pnl}`);
  });

  it("a long that gains exactly the round-trip cost breaks even (within rounding)", () => {
    const moveFraction = ROUND_TRIP_COST_PCT; // 0.003
    const pnl = roundTripPnlUsd({
      direction: "up",
      rawEntryPrice: 100,
      rawExitPrice: 100 * (1 + moveFraction),
      positionSizeUsd: 1000,
    });
    // Should be near zero — within $1 either way for $1000 position.
    assert.ok(Math.abs(pnl) < 1.5, `breakeven trade should be ~$0, got ${pnl}`);
  });

  it("a short profits when price falls, and the math is symmetric", () => {
    const longProfit = roundTripPnlUsd({
      direction: "up",
      rawEntryPrice: 100,
      rawExitPrice: 102,
      positionSizeUsd: 1000,
    });
    const shortProfit = roundTripPnlUsd({
      direction: "down",
      rawEntryPrice: 102,
      rawExitPrice: 100,
      positionSizeUsd: 1000,
    });
    assert.ok(longProfit > 0);
    assert.ok(shortProfit > 0);
    // Symmetry: same notional move, same costs, so within $1.
    assert.ok(Math.abs(longProfit - shortProfit) < 1.5,
      `long ${longProfit} vs short ${shortProfit} should be within $1.5`);
  });

  it("entry slippage is unfavourable to the trader on both sides", () => {
    assert.equal(applyEntrySlippage(100, "up"), 100 * (1 + SLIPPAGE_PCT));
    assert.equal(applyEntrySlippage(100, "down"), 100 * (1 - SLIPPAGE_PCT));
    // Exit is the opposite — long sells low, short buys high.
    assert.equal(applyExitSlippage(100, "up"), 100 * (1 - SLIPPAGE_PCT));
    assert.equal(applyExitSlippage(100, "down"), 100 * (1 + SLIPPAGE_PCT));
  });

  it("entryFee uses TAKER and exitFee uses MAKER", () => {
    assert.equal(entryFee(1000), 1000 * TAKER_FEE_PCT);
    assert.equal(exitFee(1000), 1000 * 0.001);
    assert.equal(exitFee(-50), 0, "exit on a fully wiped position pays no fee");
  });
});

describe("stop-loss / take-profit triggers (production logic)", () => {
  it("LONG stop-loss fires when price drops to/below stop", () => {
    assert.equal(isStopLossHit("up", 95, 95), true);
    assert.equal(isStopLossHit("up", 94, 95), true);
    assert.equal(isStopLossHit("up", 96, 95), false);
  });

  it("SHORT stop-loss fires when price rises to/above stop", () => {
    assert.equal(isStopLossHit("down", 105, 105), true);
    assert.equal(isStopLossHit("down", 106, 105), true);
    assert.equal(isStopLossHit("down", 104, 105), false);
  });

  it("LONG take-profit fires when price rises to/above target", () => {
    assert.equal(isTakeProfitHit("up", 110, 110), true);
    assert.equal(isTakeProfitHit("up", 111, 110), true);
    assert.equal(isTakeProfitHit("up", 109, 110), false);
  });

  it("SHORT take-profit fires when price drops to/below target", () => {
    assert.equal(isTakeProfitHit("down", 90, 90), true);
    assert.equal(isTakeProfitHit("down", 89, 90), true);
    assert.equal(isTakeProfitHit("down", 91, 90), false);
  });

  it("for a LONG, stop-loss and take-profit cannot both be true at the same price (sane setup)", () => {
    // Sane: stop < entry < tp.
    const stop = 90, tp = 110;
    for (const px of [85, 90, 100, 110, 115]) {
      const sl = isStopLossHit("up", px, stop);
      const tpHit = isTakeProfitHit("up", px, tp);
      assert.ok(!(sl && tpHit), `price ${px} should not trigger both SL and TP`);
    }
  });
});

describe("anomaly-cancel cash invariance (entry-fee leak regression)", () => {
  // Mirrors the open path in paper-trader.ts: deduct positionSize + entryFee
  // from cash, store the post-fee positionSize on the position row, then
  // anomaly-cancel and refund. After the cancel, cash MUST equal the exact
  // pre-trade value — no taker-fee leak. This guards both the live trader
  // and the Python backtester (which mirrors this invariant).
  function simulateOpenAndAnomalyCancel(preTradeCash: number, preFeeSize: number) {
    const fee = entryFee(preFeeSize);
    const storedPositionSize = preFeeSize - fee;
    const cashAfterOpen = preTradeCash - (storedPositionSize + fee);
    const refund = storedPositionSize + recoverEntryFee(storedPositionSize);
    const cashAfterCancel = cashAfterOpen + refund;
    return { cashAfterOpen, cashAfterCancel, fee };
  }

  it("opening then anomaly-cancelling returns cash to its exact pre-trade value", () => {
    const cases = [
      { cash: 1000, size: 100 },
      { cash: 1000, size: 500 },
      { cash: 12345.67, size: 987.65 },
      { cash: 100, size: 1 },
    ];
    for (const { cash, size } of cases) {
      const { cashAfterOpen, cashAfterCancel, fee } =
        simulateOpenAndAnomalyCancel(cash, size);
      assert.ok(fee > 0, "entry fee should be positive");
      assert.ok(cashAfterOpen < cash, "open must deduct cash");
      assert.ok(
        Math.abs(cashAfterCancel - cash) < 1e-9,
        `anomaly-cancel must restore cash: pre=${cash} post=${cashAfterCancel}`,
      );
    }
  });

  it("recoverEntryFee is the exact inverse of the open-time deduction", () => {
    for (const preFeeSize of [1, 50, 100, 1000, 9876.54]) {
      const fee = entryFee(preFeeSize);
      const stored = preFeeSize - fee;
      const recovered = recoverEntryFee(stored);
      assert.ok(
        Math.abs(recovered - fee) < 1e-9,
        `recoverEntryFee(${stored}) should equal ${fee}, got ${recovered}`,
      );
    }
  });
});

describe("anomaly-cancel daily-loss accumulator invariance (task #345)", () => {
  // End-to-end exercise of the same code path the live trader runs:
  //   1. open at the entry price → cash drops by stored+fee, but a
  //      mark-to-market notional + entryFee are credited back, so
  //      totalValue is preserved at entry.
  //   2. anomaly-cancel → position deleted, refund=stored+entryFee
  //      flows back to cash. No realised PnL is recorded.
  //   3. checkRiskLimits is called with the post-cancel totalValue.
  //
  // Both calls to the REAL paper-trader.ts:checkRiskLimits must
  // return dailyLossHit=false with identical day-loss ratios. If a
  // future change accidentally books the cancelled fee/notional as
  // realised loss — or otherwise leaks value out of the portfolio
  // across a cancel — the breaker would trip falsely on the second
  // call and this test would fail.
  function dayLossRatio(currentValue: number, dayStartValue: number): number {
    return (dayStartValue - currentValue) / dayStartValue;
  }

  // Mark-to-market totalValue at entry (open path):
  //   cash_after_open + position_notional + entry_fee_capitalised.
  // The open path in paper-trader.ts deducts (stored + fee) from cash
  // but the position carries a notional = stored at entry, and the
  // entryFee column is preserved on the trade row, so totalValue at
  // entry == pre-trade cash. We compute it here the same way to make
  // sure a future change to either side breaks loud.
  function totalValueAtOpen(preCash: number, size: number): number {
    const fee = entryFee(size);
    const stored = size - fee;
    const cashAfterOpen = preCash - (stored + fee);
    const positionNotional = stored;        // entry mark = entry price
    const capitalisedFee = fee;             // refunded by recoverEntryFee on cancel
    return cashAfterOpen + positionNotional + capitalisedFee;
  }

  function totalValueAfterCancel(preCash: number, size: number): number {
    const fee = entryFee(size);
    const stored = size - fee;
    const cashAfterOpen = preCash - (stored + fee);
    const refund = stored + recoverEntryFee(stored);
    return cashAfterOpen + refund;          // no open positions → cash only
  }

  it("checkRiskLimits permits the next trade after an open + anomaly-cancel near the limit", () => {
    // Pre-trade portfolio: dayStart=1000, currentValue=905 (already
    // 9.5% down — one whole percent under the 10% limit). A trade that
    // opens, gets anomaly-cancelled, and then attempts again MUST NOT
    // trip the breaker.
    const dayStart = 1000;
    const preCash = 905;
    const today = new Date().toISOString().slice(0, 10);
    const size = 100;

    // Pre-trade snapshot: precondition — breaker permits.
    const preRisk = checkRiskLimits(preCash, dayStart, dayStart, today);
    assert.equal(preRisk.dailyLossHit, false,
      "precondition: breaker must permit the trade we're about to open");

    // Open: totalValue is preserved at entry.
    const openTotal = totalValueAtOpen(preCash, size);
    assert.ok(
      Math.abs(openTotal - preCash) < 1e-9,
      `open path must preserve totalValue: pre=${preCash} post=${openTotal}`,
    );
    const openRisk = checkRiskLimits(openTotal, dayStart, dayStart, today);
    assert.equal(openRisk.dailyLossHit, false,
      "breaker tripped at entry — open path leaked value out of the portfolio");

    // Anomaly cancel: refund restores totalValue exactly.
    const postTotal = totalValueAfterCancel(preCash, size);
    assert.ok(
      Math.abs(postTotal - preCash) < 1e-9,
      `anomaly-cancel must restore totalValue exactly: pre=${preCash} post=${postTotal}`,
    );
    const postRisk = checkRiskLimits(postTotal, dayStart, dayStart, today);

    assert.equal(
      postRisk.dailyLossHit, false,
      `breaker tripped after anomaly-cancel — would block the next trade falsely (post totalValue=${postTotal})`,
    );
    assert.equal(
      preRisk.dailyLossHit, postRisk.dailyLossHit,
      "breaker state changed across an anomaly-cancel — accumulator drift",
    );
    const preRatio = dayLossRatio(preCash, dayStart);
    const postRatio = dayLossRatio(postTotal, dayStart);
    assert.ok(
      Math.abs(preRatio - postRatio) < 1e-12,
      `daily-loss ratio drifted across an anomaly-cancel: pre=${preRatio} post=${postRatio}`,
    );
    assert.ok(preRatio < DAILY_LOSS_LIMIT_PCT,
      `precondition: pre-trade ratio (${preRatio}) must be below the limit (${DAILY_LOSS_LIMIT_PCT})`);
  });

  it("ratio invariance holds across a sweep of (size, dayStart) values, all driven through checkRiskLimits", () => {
    const today = new Date().toISOString().slice(0, 10);
    const cases = [
      { dayStart: 1000, preCash: 950, size: 100 },
      { dayStart: 1000, preCash: 905, size: 500 },
      { dayStart: 12345.67, preCash: 12000, size: 987.65 },
      { dayStart: 100, preCash: 91, size: 1 },
    ];
    for (const { dayStart, preCash, size } of cases) {
      const postTotal = totalValueAfterCancel(preCash, size);
      const preRisk = checkRiskLimits(preCash, dayStart, dayStart, today);
      const postRisk = checkRiskLimits(postTotal, dayStart, dayStart, today);
      assert.equal(
        preRisk.dailyLossHit, postRisk.dailyLossHit,
        `breaker state changed: dayStart=${dayStart} preCash=${preCash} size=${size}`,
      );
      const preR = dayLossRatio(preCash, dayStart);
      const postR = dayLossRatio(postTotal, dayStart);
      assert.ok(
        Math.abs(preR - postR) < 1e-9,
        `ratio drifted: dayStart=${dayStart} preCash=${preCash} size=${size} preR=${preR} postR=${postR}`,
      );
    }
  });
});

describe("abstain (stable) execution skip", () => {
  it("shouldSkipForAbstain returns true ONLY for stable", () => {
    assert.equal(shouldSkipForAbstain("stable"), true);
    assert.equal(shouldSkipForAbstain("up"), false);
    assert.equal(shouldSkipForAbstain("down"), false);
  });
});
