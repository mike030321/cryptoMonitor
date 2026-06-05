/**
 * MTTM success-criteria report (task #614 step 6).
 *
 * Computes the 10-field truth report described in `task-614.md` for a
 * 24h / 48h / 72h trailing window:
 *
 *   1. decisions evaluated — candidate ticks for the MTTM lane only:
 *      MTTM trades opened in the window + skip events recorded in the
 *      window for any coin in the MTTM universe. Skips for coins
 *      outside the universe are intentionally excluded so the count is
 *      not inflated by unrelated brain skips on the wider monitored set.
 *   2. trades opened — MTTM trades opened in the window (any status).
 *   3. realised PnL (closed positions, post-fee) — restricted to MTTM
 *      trades whose `closedAt` falls inside the window. Honest cutoff:
 *      a trade opened earlier but closed in the window IS counted; a
 *      trade opened in the window but still open is NOT counted.
 *   4. unrealised PnL (open positions, mark-to-market) — for currently
 *      open MTTM positions.
 *   5. fees paid — sum of entry+exit fees on the closed-in-window slice.
 *   6. slippage estimate — SLIPPAGE_PCT × notional × 2 (round trip)
 *      for each closed-in-window trade. The spec asked for
 *      max(0, fill_price − mid_at_decision); this build does not
 *      journal mid_at_decision per fill, so the constant-rate estimate
 *      is the honest stand-in. Field name keeps the "Estimate" suffix
 *      to make this explicit on the wire.
 *   7. win rate (closed-in-window trades only).
 *   8. cost-aware directional accuracy — fraction of closed-in-window
 *      trades whose post-fee return on the intended direction is > 0.
 *   9. Δ vs Buy & Hold, DCA + Circuit Breaker, Trend Filter — measured
 *      from MTTM enable time, so the comparison is apples-to-apples.
 *  10. verdict — see VERDICT_RULES below.
 *
 * The report intentionally derives every number from the existing
 * `paper_trades` + `paper_positions` + `strategy_snapshots` +
 * `skip_events` tables — no new write paths, no new bookkeeping. If
 * the underlying data is missing for the window the corresponding
 * field is reported honestly (`null` / `0`) rather than padded.
 */
import { and, eq, gt, isNotNull, gte, inArray } from "drizzle-orm";
import {
  db,
  paperTradesTable,
  paperPositionsTable,
  strategySnapshotsTable,
  skipEventsTable,
} from "@workspace/db";
import { fetchCoinPrices } from "./coins";
import { ROUND_TRIP_COST_PCT, SLIPPAGE_PCT, TAKER_FEE_PCT } from "./trading-constants";
import { recoverEntryFee } from "./trade-math";
import { getMttmConfig, slotKey, type MttmConfig } from "./mttm";

export type MttmWindow = "24h" | "48h" | "72h";
export type MttmVerdict = "continue" | "expand" | "stop" | "insufficient_data";

export interface MttmBaselineDelta {
  strategyType: "buy-hold" | "dca-cb" | "trend-filter";
  label: string;
  startEquity: number | null;
  currentEquity: number | null;
  pnlPctAtStart: number | null;
  pnlPctNow: number | null;
  /**
   * Δ = (MTTM total return %) − (baseline total return %), both measured
   * from MTTM enable time to "now" or the end of the window. Positive
   * means MTTM is outperforming the baseline.
   */
  deltaPct: number | null;
}

export interface MttmReport {
  window: MttmWindow;
  generatedAt: string;
  enabled: boolean;
  enabledAt: string | null;
  windowStart: string;
  windowEnd: string;
  /** universe at report time — pinned slot list */
  universeSize: number;
  /** 1 — candidate ticks evaluated in the window */
  decisionsEvaluated: number;
  /** 2 — trades opened in the window (status open OR closed, opened in window) */
  tradesOpened: number;
  /** 3 — realised PnL post-fee, USD */
  realisedPnlUsd: number;
  /** 4 — unrealised PnL on currently-open MTTM positions, USD */
  unrealisedPnlUsd: number;
  /** 5 — fees paid in the window (sum of entry+exit fees), USD */
  feesPaidUsd: number;
  /** 6 — slippage estimate in the window, USD */
  slippageEstimateUsd: number;
  /** 7 — win rate on closed MTTM trades in the window, 0..1 */
  winRate: number | null;
  /** 8 — cost-aware directional accuracy, 0..1 */
  costAwareDirectionalAccuracy: number | null;
  /** 9 — Δ vs each baseline */
  baselines: MttmBaselineDelta[];
  /** 10 — verdict */
  verdict: MttmVerdict;
  /** Why the verdict came out that way (one-liner). */
  verdictDetail: string;
  /** Per-MTTM tally diagnostics. */
  consecutiveLosses: number;
  totalMttmTradesSinceEnable: number;
  postFeePnlPctSinceEnable: number | null;
  /** True when the auto-disable rule has tripped. */
  autoDisabled: boolean;
  disableReasonDetail: string | null;
}

const WINDOW_MS: Record<MttmWindow, number> = {
  "24h": 24 * 3600_000,
  "48h": 48 * 3600_000,
  "72h": 72 * 3600_000,
};

function pickBaselineEquityAt(
  rows: { equity: number; timestamp: Date | null }[],
  at: Date,
): number | null {
  // Pick the snapshot at-or-before `at`. Snapshots are written every
  // strategy-lab cycle, so the most recent <= `at` is the right anchor.
  let best: { equity: number; ts: number } | null = null;
  const target = at.getTime();
  for (const r of rows) {
    const ts = r.timestamp ? r.timestamp.getTime() : 0;
    if (ts > target) continue;
    if (!best || ts > best.ts) best = { equity: r.equity, ts };
  }
  return best ? best.equity : null;
}

function pickBaselineEquityNow(
  rows: { equity: number; timestamp: Date | null }[],
): number | null {
  if (rows.length === 0) return null;
  let best: { equity: number; ts: number } | null = null;
  for (const r of rows) {
    const ts = r.timestamp ? r.timestamp.getTime() : 0;
    if (!best || ts > best.ts) best = { equity: r.equity, ts };
  }
  return best ? best.equity : null;
}

interface BaselineSpec {
  strategyType: MttmBaselineDelta["strategyType"];
  label: string;
}
const BASELINES: BaselineSpec[] = [
  { strategyType: "buy-hold", label: "Buy & Hold" },
  { strategyType: "dca-cb", label: "DCA + Circuit Breaker" },
  { strategyType: "trend-filter", label: "Trend Filter" },
];

async function computeBaselines(
  enabledAt: Date,
  now: Date,
): Promise<MttmBaselineDelta[]> {
  const out: MttmBaselineDelta[] = [];
  const rows = await db
    .select({
      strategyType: strategySnapshotsTable.strategyType,
      equity: strategySnapshotsTable.equity,
      timestamp: strategySnapshotsTable.timestamp,
    })
    .from(strategySnapshotsTable)
    .where(
      inArray(
        strategySnapshotsTable.strategyType,
        BASELINES.map((b) => b.strategyType),
      ),
    );
  const byType = new Map<string, { equity: number; timestamp: Date | null }[]>();
  for (const r of rows) {
    const arr = byType.get(r.strategyType) ?? [];
    arr.push({ equity: r.equity, timestamp: r.timestamp });
    byType.set(r.strategyType, arr);
  }
  for (const b of BASELINES) {
    const list = byType.get(b.strategyType) ?? [];
    const startEquity = pickBaselineEquityAt(list, enabledAt);
    const currentEquity = pickBaselineEquityNow(list);
    out.push({
      strategyType: b.strategyType,
      label: b.label,
      startEquity,
      currentEquity,
      pnlPctAtStart: null,
      pnlPctNow:
        startEquity !== null && currentEquity !== null && startEquity > 0
          ? ((currentEquity - startEquity) / startEquity) * 100
          : null,
      deltaPct: null, // filled in by caller after MTTM total-return % is known
    });
  }
  return out;
}

function decideVerdict(
  rep: Omit<MttmReport, "verdict" | "verdictDetail">,
): { verdict: MttmVerdict; detail: string } {
  if (rep.autoDisabled) {
    return {
      verdict: "stop",
      detail: rep.disableReasonDetail ?? "auto-disabled",
    };
  }
  const usableBaselines = rep.baselines.filter((b) => b.deltaPct !== null);
  if (usableBaselines.length === 0) {
    return {
      verdict: "insufficient_data",
      detail: "no baseline equity snapshots available since MTTM enabled",
    };
  }
  const allBeatBaselines = usableBaselines.every((b) => (b.deltaPct ?? -Infinity) >= 0);
  const anyBeatsBaseline = usableBaselines.some((b) => (b.deltaPct ?? -Infinity) >= 0);
  // 72h-only "stop" rule: Δ vs ALL baselines < 0 after 72h window.
  if (rep.window === "72h" && !anyBeatsBaseline) {
    return {
      verdict: "stop",
      detail: `72h window: Δ vs all ${usableBaselines.length} baseline(s) < 0`,
    };
  }
  // Expand: Δ >= 0 vs all + realised PnL >= 0 + n_trades >= 10
  if (
    allBeatBaselines &&
    rep.realisedPnlUsd >= 0 &&
    rep.tradesOpened >= 10
  ) {
    return {
      verdict: "expand",
      detail: `Δ ≥ 0 vs all ${usableBaselines.length} baseline(s); realised PnL ≥ 0; n=${rep.tradesOpened} ≥ 10`,
    };
  }
  if (anyBeatsBaseline) {
    return {
      verdict: "continue",
      detail: `Δ ≥ 0 vs at least one baseline (${usableBaselines.filter((b) => (b.deltaPct ?? -Infinity) >= 0).length}/${usableBaselines.length})`,
    };
  }
  return {
    verdict: "continue",
    detail: "no auto-disable; insufficient outperformance for expand",
  };
}

export async function buildMttmReport(
  windowKey: MttmWindow,
  cfg?: MttmConfig,
): Promise<MttmReport> {
  const config = cfg ?? (await getMttmConfig());
  const now = new Date();
  const windowStart = new Date(now.getTime() - WINDOW_MS[windowKey]);
  // Anchor MTTM-since-enable measurements at max(enabledAt, windowStart).
  // This is the honest cutoff: "performance over the last N hours, but
  // never counting trades from before MTTM was actually on".
  const enabledAt = config.enabledAt ? new Date(config.enabledAt) : null;
  const measureFrom =
    enabledAt && enabledAt.getTime() > windowStart.getTime() ? enabledAt : windowStart;

  // ── Trades opened in window (any status) — drives field 2 and the
  // tradesOpened-since-enable denominators. ──────────────────────────
  const openedRows = await db
    .select({
      id: paperTradesTable.id,
      coinId: paperTradesTable.coinId,
      timeframe: paperTradesTable.timeframe,
      action: paperTradesTable.action,
      entryPrice: paperTradesTable.entryPrice,
      exitPrice: paperTradesTable.exitPrice,
      positionSize: paperTradesTable.positionSize,
      entryFee: paperTradesTable.entryFee,
      pnl: paperTradesTable.pnl,
      pnlPercent: paperTradesTable.pnlPercent,
      status: paperTradesTable.status,
      closedAt: paperTradesTable.closedAt,
      createdAt: paperTradesTable.createdAt,
    })
    .from(paperTradesTable)
    .where(gte(paperTradesTable.createdAt, measureFrom));
  const mttmOpenedTrades = openedRows.filter((r) =>
    config.universeKeys.has(slotKey(r.coinId, r.timeframe)),
  );

  // ── Closed-in-window MTTM trades (drives realised PnL, fees,
  // slippage, win rate, cost-aware DA — fields 3, 5, 6, 7, 8). A
  // trade opened weeks ago that closed inside the window MUST count
  // here; a trade opened in the window that has not yet closed must
  // NOT. The previous implementation filtered by createdAt and then
  // by status="closed", which got both halves of that wrong. ──────
  const closedRows = await db
    .select({
      id: paperTradesTable.id,
      coinId: paperTradesTable.coinId,
      timeframe: paperTradesTable.timeframe,
      action: paperTradesTable.action,
      entryPrice: paperTradesTable.entryPrice,
      exitPrice: paperTradesTable.exitPrice,
      positionSize: paperTradesTable.positionSize,
      entryFee: paperTradesTable.entryFee,
      pnl: paperTradesTable.pnl,
      pnlPercent: paperTradesTable.pnlPercent,
      status: paperTradesTable.status,
      closedAt: paperTradesTable.closedAt,
      createdAt: paperTradesTable.createdAt,
    })
    .from(paperTradesTable)
    .where(
      and(
        eq(paperTradesTable.status, "closed"),
        isNotNull(paperTradesTable.closedAt),
        gte(paperTradesTable.closedAt, measureFrom),
      ),
    );
  const closedMttm = closedRows.filter(
    (r) => r.closedAt && config.universeKeys.has(slotKey(r.coinId, r.timeframe)),
  );

  let realisedPnlUsd = 0;
  let feesPaidUsd = 0;
  let slippageEstimateUsd = 0;
  let wins = 0;
  let costAwareCorrect = 0;
  let costAwareTotal = 0;
  for (const t of closedMttm) {
    const pnl = t.pnl ?? 0;
    realisedPnlUsd += pnl;
    const entryFee = t.entryFee ?? recoverEntryFee(t.positionSize);
    // Exit fee back-derived from the post-pnl notional × taker fee,
    // matching paper-trader.ts.computeExitFee.
    const exitFee = (t.positionSize + pnl) * TAKER_FEE_PCT;
    feesPaidUsd += entryFee + exitFee;
    // Slippage estimate: round-trip slippage on the gross notional.
    slippageEstimateUsd += t.positionSize * SLIPPAGE_PCT * 2;
    if (pnl > 0) wins++;
    // Cost-aware directional accuracy: was the post-fee return on the
    // intended direction positive?
    const realisedReturnFrac = t.positionSize > 0 ? pnl / t.positionSize : 0;
    if (realisedReturnFrac > 0) costAwareCorrect++;
    costAwareTotal++;
  }
  const winRate = closedMttm.length > 0 ? wins / closedMttm.length : null;
  const costAwareDirectionalAccuracy =
    costAwareTotal > 0 ? costAwareCorrect / costAwareTotal : null;

  // ── Unrealised PnL on currently-open MTTM positions ────────────────
  const openPositions = await db
    .select({
      coinId: paperPositionsTable.coinId,
      timeframe: paperPositionsTable.timeframe,
      direction: paperPositionsTable.direction,
      entryPrice: paperPositionsTable.entryPrice,
      positionSize: paperPositionsTable.positionSize,
    })
    .from(paperPositionsTable);
  const mttmOpenPositions = openPositions.filter((p) =>
    config.universeKeys.has(slotKey(p.coinId, p.timeframe)),
  );
  let unrealisedPnlUsd = 0;
  if (mttmOpenPositions.length > 0) {
    let prices: { coinId: string; price: number }[] = [];
    try {
      prices = (await fetchCoinPrices()).map((p) => ({
        coinId: p.id,
        price: p.currentPrice,
      }));
    } catch {
      prices = [];
    }
    const priceMap = new Map(prices.map((p) => [p.coinId, p.price]));
    for (const p of mttmOpenPositions) {
      const px = priceMap.get(p.coinId);
      if (typeof px !== "number" || px <= 0) continue;
      const move =
        p.direction === "down"
          ? (p.entryPrice - px) / p.entryPrice
          : (px - p.entryPrice) / p.entryPrice;
      unrealisedPnlUsd += move * p.positionSize;
    }
  }

  // ── Decisions evaluated (MTTM-scoped candidate ticks) ───────────────
  // = MTTM trades opened in window + skip events in the window for any
  // coin that belongs to the MTTM universe (any timeframe — `skip_events`
  // does not record timeframe). Skips for unrelated coins are excluded
  // so this count reflects MTTM-lane activity, not the wider monitored
  // set. Skips with `coinId IS NULL` (universal skips, e.g. monitoring
  // off) are also excluded — they are not attributable to MTTM.
  const mttmCoinIds = Array.from(
    new Set(config.universe.map((s) => s.coinId)),
  );
  let mttmSkipCount = 0;
  if (mttmCoinIds.length > 0) {
    const skipsInWindow = await db
      .select({ id: skipEventsTable.id })
      .from(skipEventsTable)
      .where(
        and(
          gte(skipEventsTable.ts, measureFrom),
          inArray(skipEventsTable.coinId, mttmCoinIds),
        ),
      );
    mttmSkipCount = skipsInWindow.length;
  }
  const decisionsEvaluated = mttmOpenedTrades.length + mttmSkipCount;

  // ── Baseline Δ ──────────────────────────────────────────────────────
  // The baseline anchor MUST be the same `measureFrom` used for the
  // MTTM-side trade stats above (i.e. max(enabledAt, windowStart)):
  //   - When MTTM was enabled BEFORE windowStart: both MTTM and
  //     baselines measure from windowStart → trailing-window Δ.
  //   - When MTTM was enabled INSIDE the window (recently turned on):
  //     both MTTM and baselines measure from enabledAt → since-enable
  //     Δ, which is the only honest "is MTTM helping?" comparison
  //     because MTTM had no exposure before enabledAt.
  // Anchoring baselines at `windowStart` (when MTTM was enabled
  // mid-window) would give baselines a longer horizon and produce
  // unfair, non-apples-to-apples Δ values; anchoring at `enabledAt`
  // unconditionally would silently extend the verdict horizon past
  // the requested window. `measureFrom` is the unique anchor that
  // keeps numerator and denominator on the same horizon for both
  // sides of the comparison.
  const baselines = await computeBaselines(measureFrom, now);

  // MTTM total return % over the window: (realised on closed-in-window
  // trades + unrealised on currently-open positions) / matching gross
  // notional. The denominator must include the same trades as the
  // numerator: closed-in-window notional + currently-open notional.
  // Using opened-in-window notional here would mis-pair the realised
  // numerator (which is closed-in-window) with a different denominator.
  const closedNotional = closedMttm.reduce((s, t) => s + t.positionSize, 0);
  const openNotional = mttmOpenPositions.reduce((s, p) => s + p.positionSize, 0);
  const totalMttmNotional = closedNotional + openNotional;
  const mttmPnlPct =
    totalMttmNotional > 0
      ? ((realisedPnlUsd + unrealisedPnlUsd) / totalMttmNotional) * 100
      : null;
  for (const b of baselines) {
    if (b.pnlPctNow === null || mttmPnlPct === null) {
      b.deltaPct = null;
    } else {
      b.deltaPct = mttmPnlPct - b.pnlPctNow;
    }
  }

  // ── Per-MTTM-since-enable tally ────────────────────────────────────
  let consecutive = 0;
  let totalMttmTradesSinceEnable = 0;
  let postFeePnlPctSinceEnable: number | null = null;
  if (enabledAt) {
    const since = await db
      .select({
        coinId: paperTradesTable.coinId,
        timeframe: paperTradesTable.timeframe,
        pnl: paperTradesTable.pnl,
        positionSize: paperTradesTable.positionSize,
        closedAt: paperTradesTable.closedAt,
      })
      .from(paperTradesTable)
      .where(
        and(
          eq(paperTradesTable.status, "closed"),
          isNotNull(paperTradesTable.closedAt),
          gt(paperTradesTable.closedAt, enabledAt),
        ),
      );
    const sinceMttm = since
      .filter((r) => config.universeKeys.has(slotKey(r.coinId, r.timeframe)))
      .sort((a, b) => {
        const ta = a.closedAt ? a.closedAt.getTime() : 0;
        const tb = b.closedAt ? b.closedAt.getTime() : 0;
        return ta - tb;
      });
    totalMttmTradesSinceEnable = sinceMttm.length;
    for (let i = sinceMttm.length - 1; i >= 0; i--) {
      const pnl = sinceMttm[i].pnl ?? 0;
      if (pnl <= 0) consecutive++;
      else break;
    }
    if (sinceMttm.length > 0) {
      let totalPnl = 0;
      let totalNotional = 0;
      for (const t of sinceMttm) {
        totalPnl += t.pnl ?? 0;
        totalNotional += t.positionSize;
      }
      postFeePnlPctSinceEnable =
        totalNotional > 0 ? (totalPnl / totalNotional) * 100 : null;
    }
  }

  const draft: Omit<MttmReport, "verdict" | "verdictDetail"> = {
    window: windowKey,
    generatedAt: now.toISOString(),
    enabled: config.enabled,
    enabledAt: config.enabledAt,
    windowStart: windowStart.toISOString(),
    windowEnd: now.toISOString(),
    universeSize: config.universeKeys.size,
    decisionsEvaluated,
    tradesOpened: mttmOpenedTrades.length,
    realisedPnlUsd,
    unrealisedPnlUsd,
    feesPaidUsd,
    slippageEstimateUsd,
    winRate,
    costAwareDirectionalAccuracy,
    baselines,
    consecutiveLosses: consecutive,
    totalMttmTradesSinceEnable,
    postFeePnlPctSinceEnable,
    autoDisabled: !!config.disableReason,
    disableReasonDetail: config.disableReason?.detail ?? null,
  };
  const { verdict, detail } = decideVerdict(draft);
  return { ...draft, verdict, verdictDetail: detail };
}

// Re-export the round-trip cost so the dashboard can show "0.30% per
// round trip" without re-deriving it locally.
export const MTTM_ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_PCT;
