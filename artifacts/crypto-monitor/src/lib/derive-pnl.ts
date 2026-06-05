// Task #368 — single source of truth for "what is this paper portfolio's
// current net P&L?". Every web page that surfaces a portfolio P&L number
// MUST go through `derivePnl(portfolio)` so they cannot drift from the
// equity-vs-seed identity the agent-detail page already uses.
//
// Net P&L is always derived as `totalValue − startingCapital`. The
// legacy realized-only `totalPnl` / `totalPnlPercent` fields were
// dropped from the API payload in Task #370 after every consumer was
// migrated through this helper; their fallback branch is gone.

export interface DerivablePortfolio {
  totalValue: number;
  startingCapital: number;
}

export interface DerivedPnl {
  /** USD net P&L = totalValue − startingCapital. */
  netPnl: number;
  /** Percent net return on starting capital. */
  netPnlPct: number;
  /** Seed used in the derivation. */
  seed: number;
}

export function derivePnl(p: DerivablePortfolio): DerivedPnl {
  const seed = p.startingCapital;
  const netPnl = p.totalValue - seed;
  const netPnlPct = seed > 0 ? (netPnl / seed) * 100 : 0;
  return { netPnl, netPnlPct, seed };
}
