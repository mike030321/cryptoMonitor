import {
  MAKER_FEE_PCT,
  TAKER_FEE_PCT,
  SLIPPAGE_PCT,
} from "./trading-constants";

/**
 * Pure-math helpers extracted from paper-trader.ts so the live trading code
 * and the test suite share ONE implementation. Anything that decides whether
 * a position is profitable, whether SL/TP is hit, or how much fee + slippage
 * costs a round trip MUST flow through these functions.
 */

export function applyEntrySlippage(rawPrice: number, direction: "up" | "down"): number {
  const adj = direction === "up" ? 1 + SLIPPAGE_PCT : 1 - SLIPPAGE_PCT;
  return rawPrice * adj;
}

export function applyExitSlippage(rawPrice: number, direction: "up" | "down"): number {
  // Exit is the opposite side of entry: a long sells (worse fill = lower),
  // a short buys to cover (worse fill = higher).
  const adj = direction === "up" ? 1 - SLIPPAGE_PCT : 1 + SLIPPAGE_PCT;
  return rawPrice * adj;
}

export function entryFee(positionSizeUsd: number): number {
  return positionSizeUsd * TAKER_FEE_PCT;
}

/**
 * Inverse of `entryFee` for the value stored in `paper_positions.position_size`.
 *
 * The open path computes `entryFee = entryFee(preFee)` and then stores the
 * net `positionSize = preFee - entryFee` (= preFee * (1 - TAKER_FEE_PCT)).
 * To reverse the open-time cash impact (`preFee + entryFee` was deducted),
 * the anomaly-cancel paths refund `positionSize + recoverEntryFee(positionSize)`.
 */
export function recoverEntryFee(storedPositionSizeUsd: number): number {
  return storedPositionSizeUsd * TAKER_FEE_PCT / (1 - TAKER_FEE_PCT);
}

export function exitFee(exitNotionalUsd: number): number {
  return Math.max(0, exitNotionalUsd) * MAKER_FEE_PCT;
}

/**
 * Net round-trip P&L in USD for a paper trade, including entry slippage,
 * exit slippage, and both fees. `direction` is the trade direction, NOT the
 * realised price move. Positive return = profit.
 */
export function roundTripPnlUsd(args: {
  direction: "up" | "down";
  rawEntryPrice: number;
  rawExitPrice: number;
  positionSizeUsd: number;
}): number {
  const { direction, rawEntryPrice, rawExitPrice, positionSizeUsd } = args;
  const adjEntry = applyEntrySlippage(rawEntryPrice, direction);
  const adjExit = applyExitSlippage(rawExitPrice, direction);
  const quantity = positionSizeUsd / adjEntry;
  const grossPnl = direction === "up"
    ? (adjExit - adjEntry) * quantity
    : (adjEntry - adjExit) * quantity;
  const exitNotional = positionSizeUsd + grossPnl;
  return grossPnl - entryFee(positionSizeUsd) - exitFee(exitNotional);
}

/**
 * Stop-loss trigger. For a LONG (direction "up") the stop fires when price
 * drops to or below stopLossPrice. For a SHORT (direction "down") it fires
 * when price rises to or above stopLossPrice. Mirrors paper-trader L573-577.
 */
export function isStopLossHit(
  direction: "up" | "down",
  currentPrice: number,
  stopLossPrice: number,
): boolean {
  return direction === "up"
    ? currentPrice <= stopLossPrice
    : currentPrice >= stopLossPrice;
}

/**
 * Take-profit trigger. Long fires at or above; short fires at or below.
 */
export function isTakeProfitHit(
  direction: "up" | "down",
  currentPrice: number,
  takeProfitPrice: number,
): boolean {
  return direction === "up"
    ? currentPrice >= takeProfitPrice
    : currentPrice <= takeProfitPrice;
}

/**
 * Should a trade be SKIPPED entirely because the agent abstained? Mirrors
 * the early-return at the top of executePaperTrade.
 */
export function shouldSkipForAbstain(direction: "up" | "down" | "stable"): boolean {
  return direction === "stable";
}
