import { db, agentsTable, paperPortfoliosTable, paperPositionsTable, paperTradesTable, strategySnapshotsTable, strategyStateTable, strategySettingsTable, strategySettingsHistoryTable, priceCandlesTable } from "@workspace/db";
import { eq, and, desc, gte, sql, inArray } from "drizzle-orm";
import { logger } from "./logger";
import type { CoinPrice } from "./coins";
import { fetchCoinPrices, isPriceDataFresh, MONITORED_COINS } from "./coins";
// Task #365 — fees, slippage, and starting capital MUST be sourced from
// shared/trading-frictions.json via trading-constants. Local literals
// here would silently disagree with paper-trader on the next tuning change.
import {
  MAKER_FEE_PCT,
  SLIPPAGE_PCT,
  INITIAL_BALANCE_USD,
} from "./trading-constants";

const DCA_AGENT_NAME = "DCA + Circuit Breaker";
const HODL_AGENT_NAME = "Strategy Lab Buy & Hold";
const TREND_AGENT_NAME = "Trend Filter (30d basket)";

const TREND_LOOKBACK_MS = 30 * 24 * 60 * 60 * 1000;

const INITIAL_CAPITAL = INITIAL_BALANCE_USD;

export const DCA_DEFAULTS = {
  drawdownTriggerPct: 20,
  resumeLookbackDays: 14,
  cycleDeployUsd: 33.33,
  buyIntervalHours: 24,
} as const;

export interface DcaSettings {
  drawdownTriggerPct: number;
  resumeLookbackDays: number;
  cycleDeployUsd: number;
  buyIntervalHours: number;
}

async function getDcaSettings(): Promise<DcaSettings> {
  const rows = await db.select().from(strategySettingsTable).where(eq(strategySettingsTable.strategyType, "dca-cb"));
  if (rows.length === 0) {
    return { ...DCA_DEFAULTS };
  }
  const r = rows[0];
  return {
    drawdownTriggerPct: r.dcaDrawdownTriggerPct,
    resumeLookbackDays: r.dcaResumeLookbackDays,
    cycleDeployUsd: r.dcaCycleDeployUsd,
    buyIntervalHours: r.dcaBuyIntervalHours,
  };
}

export async function getStrategySettings(): Promise<DcaSettings> {
  return getDcaSettings();
}

export async function updateStrategySettings(input: Partial<DcaSettings>): Promise<DcaSettings> {
  const current = await getDcaSettings();
  const next: DcaSettings = {
    drawdownTriggerPct: clamp(input.drawdownTriggerPct ?? current.drawdownTriggerPct, 1, 90),
    resumeLookbackDays: Math.round(clamp(input.resumeLookbackDays ?? current.resumeLookbackDays, 1, 60)),
    cycleDeployUsd: clamp(input.cycleDeployUsd ?? current.cycleDeployUsd, 1, 500),
    buyIntervalHours: Math.round(clamp(input.buyIntervalHours ?? current.buyIntervalHours, 1, 168)),
  };
  await db.insert(strategySettingsTable).values({
    strategyType: "dca-cb",
    dcaDrawdownTriggerPct: next.drawdownTriggerPct,
    dcaResumeLookbackDays: next.resumeLookbackDays,
    dcaCycleDeployUsd: next.cycleDeployUsd,
    dcaBuyIntervalHours: next.buyIntervalHours,
  }).onConflictDoUpdate({
    target: strategySettingsTable.strategyType,
    set: {
      dcaDrawdownTriggerPct: next.drawdownTriggerPct,
      dcaResumeLookbackDays: next.resumeLookbackDays,
      dcaCycleDeployUsd: next.cycleDeployUsd,
      dcaBuyIntervalHours: next.buyIntervalHours,
      updatedAt: new Date(),
    },
  });

  const drawdownChanged = next.drawdownTriggerPct !== current.drawdownTriggerPct;
  const resumeChanged = next.resumeLookbackDays !== current.resumeLookbackDays;
  const deployChanged = Math.abs(next.cycleDeployUsd - current.cycleDeployUsd) > 1e-6;
  if (drawdownChanged || resumeChanged || deployChanged) {
    await db.insert(strategySettingsHistoryTable).values({
      strategyType: "dca-cb",
      drawdownTriggerBefore: drawdownChanged ? current.drawdownTriggerPct : null,
      drawdownTriggerAfter: drawdownChanged ? next.drawdownTriggerPct : null,
      resumeLookbackBefore: resumeChanged ? current.resumeLookbackDays : null,
      resumeLookbackAfter: resumeChanged ? next.resumeLookbackDays : null,
      cycleDeployBefore: deployChanged ? current.cycleDeployUsd : null,
      cycleDeployAfter: deployChanged ? next.cycleDeployUsd : null,
    });
  }

  logger.info({ next }, "Strategy lab DCA settings updated");
  return next;
}

function clamp(n: number, lo: number, hi: number): number {
  if (Number.isNaN(n) || !Number.isFinite(n)) return lo;
  return Math.min(hi, Math.max(lo, n));
}

const SNAPSHOT_INTERVAL_CYCLES = 1;
const FAR_FUTURE = new Date("2099-12-31T00:00:00Z");

async function ensureAgent(name: string, personality: string, strategyType: string): Promise<{ id: number; name: string }> {
  // Task #468 — every strategy-lab basket strategy maps to the
  // single `baseline_reference` registry profile. The strategyType
  // column keeps the variant (dca-cb / buy-hold / trend-filter)
  // visible to the dashboard; the registry policy itself is shared.
  const PROFILE_ID = "baseline_reference";
  const existing = await db.select().from(agentsTable).where(eq(agentsTable.name, name));
  if (existing.length > 0) {
    const needsUpdate =
      existing[0].strategyType !== strategyType ||
      existing[0].isActive !== false ||
      (existing[0] as { profileId?: string | null }).profileId !== PROFILE_ID;
    if (needsUpdate) {
      await db.update(agentsTable)
        .set({ strategyType, isActive: false, profileId: PROFILE_ID })
        .where(eq(agentsTable.id, existing[0].id));
    }
    return { id: existing[0].id, name: existing[0].name };
  }
  const [created] = await db.insert(agentsTable).values({
    name,
    personality,
    status: "active",
    isActive: false,
    strategyType,
    evolutionMethod: "baseline",
    preferredTimeframes: JSON.stringify(["1d"]),
    profileId: PROFILE_ID,
  }).returning({ id: agentsTable.id, name: agentsTable.name });

  await db.insert(paperPortfoliosTable).values({
    agentId: created.id,
    agentName: created.name,
    cashBalance: INITIAL_CAPITAL,
    totalValue: INITIAL_CAPITAL,
    totalTrades: 0,
    winningTrades: 0,
    losingTrades: 0,
    peakValue: INITIAL_CAPITAL,
    dayStartValue: INITIAL_CAPITAL,
  }).onConflictDoNothing();

  await db.insert(strategyStateTable).values({
    agentId: created.id,
    peakValue: INITIAL_CAPITAL,
  }).onConflictDoNothing();

  logger.info({ agentId: created.id, name }, "Strategy lab agent created");
  return created;
}

export async function ensureStrategyLabAgents(): Promise<{ dcaId: number; hodlId: number; trendId: number }> {
  const dca = await ensureAgent(DCA_AGENT_NAME, "Daily DCA across basket with 20% drawdown circuit breaker", "dca-cb");
  const hodl = await ensureAgent(HODL_AGENT_NAME, "Equal-weight buy-and-hold across the monitored basket", "buy-hold");
  const trend = await ensureAgent(TREND_AGENT_NAME, "Equal-weight basket buy when 30-day basket return > 0; full exit when negative", "trend-filter");

  for (const agentId of [dca.id, hodl.id, trend.id]) {
    const exists = await db.select().from(strategyStateTable).where(eq(strategyStateTable.agentId, agentId));
    if (exists.length === 0) {
      await db.insert(strategyStateTable).values({ agentId, peakValue: INITIAL_CAPITAL });
    }
  }

  return { dcaId: dca.id, hodlId: hodl.id, trendId: trend.id };
}

interface BasketCoin {
  id: string;
  name: string;
  symbol: string;
  currentPrice: number;
  marketCap: number;
}

function buildBasket(prices: CoinPrice[], size: number): BasketCoin[] {
  return [...prices]
    .filter(p => p.currentPrice > 0 && p.marketCap > 0)
    .sort((a, b) => b.marketCap - a.marketCap)
    .slice(0, size)
    .map(p => ({ id: p.id, name: p.name, symbol: p.symbol, currentPrice: p.currentPrice, marketCap: p.marketCap }));
}

async function applyBuy(agentId: number, agentName: string, coin: BasketCoin, usdAmount: number): Promise<void> {
  if (usdAmount <= 0 || coin.currentPrice <= 0) return;

  const slippageAdj = 1 + SLIPPAGE_PCT;
  const effectivePrice = coin.currentPrice * slippageAdj;
  const fee = usdAmount * MAKER_FEE_PCT;
  const netInvest = usdAmount - fee;
  if (netInvest <= 0) return;
  const quantity = netInvest / effectivePrice;

  await db.transaction(async (tx) => {
    const portfolios = await tx.select().from(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, agentId)).for("update");
    if (portfolios.length === 0) return;
    const portfolio = portfolios[0];
    if (portfolio.cashBalance < usdAmount) return;

    const [trade] = await tx.insert(paperTradesTable).values({
      agentId,
      agentName,
      coinId: coin.id,
      coinName: coin.name,
      action: "buy",
      entryPrice: effectivePrice,
      quantity,
      positionSize: netInvest,
      timeframe: "1d",
      predictionId: null,
      status: "open",
    }).returning({ id: paperTradesTable.id });

    const existing = await tx.select().from(paperPositionsTable)
      .where(and(eq(paperPositionsTable.agentId, agentId), eq(paperPositionsTable.coinId, coin.id)));

    if (existing.length > 0) {
      const pos = existing[0];
      const newQty = pos.quantity + quantity;
      const newSize = pos.positionSize + netInvest;
      const newAvg = newSize / newQty;
      await tx.update(paperPositionsTable).set({
        quantity: newQty,
        positionSize: newSize,
        entryPrice: newAvg,
      }).where(eq(paperPositionsTable.id, pos.id));
    } else {
      await tx.insert(paperPositionsTable).values({
        agentId,
        agentName,
        coinId: coin.id,
        coinName: coin.name,
        direction: "up",
        entryPrice: effectivePrice,
        quantity,
        positionSize: netInvest,
        timeframe: "1d",
        tradeId: trade.id,
        expiresAt: FAR_FUTURE,
        stopLossPrice: null,
        takeProfitPrice: null,
        peakPrice: coin.currentPrice,
      });
    }

    await tx.update(paperPortfoliosTable).set({
      cashBalance: portfolio.cashBalance - usdAmount,
      totalTrades: portfolio.totalTrades + 1,
      updatedAt: new Date(),
    }).where(eq(paperPortfoliosTable.agentId, agentId));

    await tx.update(strategyStateTable).set({
      lastBuyAt: new Date(),
      totalFees: sql`${strategyStateTable.totalFees} + ${fee}`,
      updatedAt: new Date(),
    }).where(eq(strategyStateTable.agentId, agentId));
  });
}

async function liquidateAll(agentId: number, agentName: string, prices: CoinPrice[]): Promise<void> {
  const positions = await db.select().from(paperPositionsTable).where(eq(paperPositionsTable.agentId, agentId));
  if (positions.length === 0) return;
  const priceMap = new Map(prices.map(p => [p.id, p.currentPrice]));

  for (const pos of positions) {
    const currentPrice = priceMap.get(pos.coinId);
    if (!currentPrice || currentPrice <= 0) continue;
    const exitPrice = currentPrice * (1 - SLIPPAGE_PCT);
    const grossProceeds = pos.quantity * exitPrice;
    const fee = grossProceeds * MAKER_FEE_PCT;
    const netProceeds = Math.max(0, grossProceeds - fee);
    const pnl = netProceeds - pos.positionSize;
    const pnlPct = pos.positionSize > 0 ? (pnl / pos.positionSize) * 100 : 0;

    await db.transaction(async (tx) => {
      await tx.insert(paperTradesTable).values({
        agentId,
        agentName,
        coinId: pos.coinId,
        coinName: pos.coinName,
        action: "close",
        entryPrice: pos.entryPrice,
        exitPrice,
        quantity: pos.quantity,
        positionSize: pos.positionSize,
        pnl,
        pnlPercent: pnlPct,
        timeframe: pos.timeframe,
        predictionId: null,
        status: "closed",
        closedAt: new Date(),
      });

      await tx.delete(paperPositionsTable).where(eq(paperPositionsTable.id, pos.id));

      const portfolios = await tx.select().from(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, agentId)).for("update");
      if (portfolios.length > 0) {
        const p = portfolios[0];
        const newCash = p.cashBalance + netProceeds;
        const newWin = pnl > 0 ? p.winningTrades + 1 : p.winningTrades;
        const newLoss = pnl <= 0 ? p.losingTrades + 1 : p.losingTrades;
        await tx.update(paperPortfoliosTable).set({
          cashBalance: newCash,
          totalTrades: p.totalTrades + 1,
          winningTrades: newWin,
          losingTrades: newLoss,
          updatedAt: new Date(),
        }).where(eq(paperPortfoliosTable.agentId, agentId));
      }

      await tx.update(strategyStateTable).set({
        totalFees: sql`${strategyStateTable.totalFees} + ${fee}`,
        updatedAt: new Date(),
      }).where(eq(strategyStateTable.agentId, agentId));
    });
  }

  logger.info({ agentId, agentName, positionsClosed: positions.length }, "Strategy lab: liquidated all positions");
}

async function computePortfolioEquity(agentId: number, prices: CoinPrice[]): Promise<{ equity: number; cash: number; invested: number }> {
  const portfolios = await db.select().from(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, agentId));
  if (portfolios.length === 0) return { equity: 0, cash: 0, invested: 0 };
  const portfolio = portfolios[0];
  const positions = await db.select().from(paperPositionsTable).where(eq(paperPositionsTable.agentId, agentId));
  const priceMap = new Map(prices.map(p => [p.id, p.currentPrice]));

  let invested = 0;
  for (const pos of positions) {
    const currentPrice = priceMap.get(pos.coinId);
    if (!currentPrice || currentPrice <= 0) {
      invested += pos.positionSize;
      continue;
    }
    const grossValue = pos.quantity * currentPrice;
    const exitFee = grossValue * MAKER_FEE_PCT;
    const slipCost = grossValue * SLIPPAGE_PCT;
    invested += Math.max(0, grossValue - exitFee - slipCost);
  }

  const equity = portfolio.cashBalance + invested;
  return { equity, cash: portfolio.cashBalance, invested };
}

async function basketAvgReturnSinceMs(prices: CoinPrice[], lookbackMs: number, agentId: number): Promise<number | null> {
  // Task #405 / B-TREND — first try the buy-hold snapshot proxy. This
  // remains the cheapest path when Buy & Hold has been running long
  // enough to have a snapshot at-or-after `cutoff`. When that proxy is
  // unavailable (e.g. Buy & Hold's `initialDeployDone` never set, or
  // its snapshot table got truncated), fall through to a price-candle
  // basket return so trend-filter exit can still fire on negative
  // 30-day basket returns. The audit's specific failure was agent 18
  // (trend-filter) staying frozen because the buy-hold proxy returned
  // null indefinitely.
  const cutoff = new Date(Date.now() - lookbackMs);
  const old = await db.select().from(strategySnapshotsTable)
    .where(and(eq(strategySnapshotsTable.strategyType, "buy-hold"), gte(strategySnapshotsTable.timestamp, cutoff)))
    .orderBy(strategySnapshotsTable.timestamp)
    .limit(1);
  const latest = await db.select().from(strategySnapshotsTable)
    .where(eq(strategySnapshotsTable.strategyType, "buy-hold"))
    .orderBy(desc(strategySnapshotsTable.timestamp))
    .limit(1);
  if (old.length > 0 && latest.length > 0 && old[0].equity > 0) {
    return (latest[0].equity - old[0].equity) / old[0].equity;
  }
  return basketAvgReturnFromCandles(prices, lookbackMs);
}

/**
 * Task #405 / B-TREND — fallback proxy for `basketAvgReturnSinceMs`.
 * Computes the equal-weight average return of the live `prices` basket
 * against the closest `1d` candle close at-or-before `Date.now() -
 * lookbackMs`. Returns null when not enough candles are available
 * across the basket (we require at least half the basket to have a
 * lookback close to avoid a single coin dominating the proxy).
 *
 * This makes the trend-filter strategy independent of the buy-hold
 * agent's equity history, which the audit identified as a cross-agent
 * coupling that froze the trend-filter strategy when buy-hold's
 * initialDeploy didn't run.
 */
async function basketAvgReturnFromCandles(prices: CoinPrice[], lookbackMs: number): Promise<number | null> {
  if (prices.length === 0) return null;
  const lookbackCutoff = new Date(Date.now() - lookbackMs);
  // Pull the most-recent 1d close at-or-before `lookbackCutoff` for
  // each monitored coin. We use 1d because the candles are reliably
  // backfilled from CMC for every monitored coin; finer cadences may
  // have gaps for newer listings.
  const coinIds = prices.map((p) => p.id);
  const rows = await db.select({
    coinId: priceCandlesTable.coinId,
    close: priceCandlesTable.close,
    bucketStart: priceCandlesTable.bucketStart,
  }).from(priceCandlesTable).where(and(
    inArray(priceCandlesTable.coinId, coinIds),
    eq(priceCandlesTable.timeframe, "1d"),
  ));
  // Pick the latest candle per coin that's at-or-before `lookbackCutoff`,
  // i.e. the closest historical close NOT into the future.
  const lookbackClose = new Map<string, number>();
  const bestTs = new Map<string, Date>();
  for (const r of rows) {
    if (r.bucketStart > lookbackCutoff) continue;
    const cur = bestTs.get(r.coinId);
    if (!cur || r.bucketStart > cur) {
      bestTs.set(r.coinId, r.bucketStart);
      lookbackClose.set(r.coinId, r.close);
    }
  }
  let acc = 0;
  let n = 0;
  for (const p of prices) {
    const past = lookbackClose.get(p.id);
    if (past === undefined || past <= 0) continue;
    acc += (p.currentPrice - past) / past;
    n++;
  }
  // Require at-least-half basket coverage — use ceil() so an odd-sized
  // basket of 3 still demands 2 (not 1) candle-backed coins. Floor() would
  // weaken the gate (1-of-3 was previously enough); fixed in remediation
  // 2026-04-24 per code-review feedback.
  if (n < Math.max(1, Math.ceil(prices.length / 2))) return null;
  return acc / n;
}

async function runDcaStrategy(agentId: number, prices: CoinPrice[]): Promise<void> {
  const states = await db.select().from(strategyStateTable).where(eq(strategyStateTable.agentId, agentId));
  if (states.length === 0) return;
  const state = states[0];

  const settings = await getDcaSettings();
  const drawdownTrigger = settings.drawdownTriggerPct / 100;
  const resumeLookbackMs = settings.resumeLookbackDays * 24 * 60 * 60 * 1000;
  const perCycleUsd = settings.cycleDeployUsd;

  const { equity } = await computePortfolioEquity(agentId, prices);
  const newPeak = Math.max(state.peakValue, equity);
  if (newPeak !== state.peakValue) {
    await db.update(strategyStateTable).set({ peakValue: newPeak, updatedAt: new Date() }).where(eq(strategyStateTable.agentId, agentId));
  }

  const drawdown = newPeak > 0 ? (newPeak - equity) / newPeak : 0;

  // Circuit breaker trigger
  if (!state.circuitBreakerActive && drawdown >= drawdownTrigger) {
    logger.warn({ agentId, drawdown: (drawdown * 100).toFixed(2) + "%" }, "DCA circuit breaker tripped — liquidating");
    await liquidateAll(agentId, DCA_AGENT_NAME, prices);
    // Re-anchor peak to post-liquidation cash so the breaker doesn't immediately re-trip
    const postLiq = await computePortfolioEquity(agentId, prices);
    await db.update(strategyStateTable).set({
      circuitBreakerActive: true,
      circuitBreakerActivatedAt: new Date(),
      peakValue: postLiq.equity,
      updatedAt: new Date(),
    }).where(eq(strategyStateTable.agentId, agentId));
    return;
  }

  // Circuit breaker resume check
  if (state.circuitBreakerActive) {
    const activatedAt = state.circuitBreakerActivatedAt?.getTime() ?? 0;
    const elapsed = Date.now() - activatedAt;
    if (elapsed >= resumeLookbackMs) {
      const ret = await basketAvgReturnSinceMs(prices, resumeLookbackMs, agentId);
      if (ret !== null && ret > 0) {
        logger.info({ agentId, ret14d: (ret * 100).toFixed(2) + "%" }, "DCA circuit breaker released — resuming");
        // Re-anchor peak to current equity on release so old peak doesn't re-trip
        const fresh = await computePortfolioEquity(agentId, prices);
        await db.update(strategyStateTable).set({
          circuitBreakerActive: false,
          circuitBreakerActivatedAt: null,
          peakValue: fresh.equity,
          updatedAt: new Date(),
        }).where(eq(strategyStateTable.agentId, agentId));
      }
    }
    return; // Don't buy while breaker active
  }

  // Time-gated buying
  const buyIntervalMs = settings.buyIntervalHours * 60 * 60 * 1000;
  const lastBuyMs = state.lastBuyAt?.getTime() ?? 0;
  if (Date.now() - lastBuyMs < buyIntervalMs) return;

  const portfolios = await db.select().from(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, agentId));
  if (portfolios.length === 0) return;
  const cash = portfolios[0].cashBalance;
  if (cash < 1) return;

  const basket = buildBasket(prices, MONITORED_COINS.length);
  if (basket.length === 0) return;

  const cycleAmount = Math.min(perCycleUsd, cash);
  const perCoin = cycleAmount / basket.length;
  if (perCoin < 0.5) return;

  for (const coin of basket) {
    await applyBuy(agentId, DCA_AGENT_NAME, coin, perCoin);
  }
  logger.info({ agentId, deployed: cycleAmount.toFixed(2), basketSize: basket.length }, "DCA cycle deployed");
}

async function runHodlStrategy(agentId: number, prices: CoinPrice[]): Promise<void> {
  const states = await db.select().from(strategyStateTable).where(eq(strategyStateTable.agentId, agentId));
  if (states.length === 0) return;
  const state = states[0];
  if (state.initialDeployDone) return;

  const portfolios = await db.select().from(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, agentId));
  if (portfolios.length === 0) return;
  const cash = portfolios[0].cashBalance;
  if (cash < 10) return;

  const basket = buildBasket(prices, MONITORED_COINS.length);
  if (basket.length === 0) return;

  const perCoin = (cash * 0.99) / basket.length;
  for (const coin of basket) {
    await applyBuy(agentId, HODL_AGENT_NAME, coin, perCoin);
  }
  await db.update(strategyStateTable).set({
    initialDeployDone: true,
    updatedAt: new Date(),
  }).where(eq(strategyStateTable.agentId, agentId));
  logger.info({ agentId, basketSize: basket.length, perCoin: perCoin.toFixed(2) }, "Buy & Hold initial deploy complete");
}

async function runTrendFilterStrategy(agentId: number, prices: CoinPrice[]): Promise<void> {
  const ret = await basketAvgReturnSinceMs(prices, TREND_LOOKBACK_MS, agentId);
  if (ret === null) return; // not enough basket history yet

  const positions = await db.select().from(paperPositionsTable).where(eq(paperPositionsTable.agentId, agentId));
  const invested = positions.length > 0;

  if (ret > 0 && !invested) {
    const portfolios = await db.select().from(paperPortfoliosTable).where(eq(paperPortfoliosTable.agentId, agentId));
    if (portfolios.length === 0) return;
    const cash = portfolios[0].cashBalance;
    if (cash < 10) return;
    const basket = buildBasket(prices, MONITORED_COINS.length);
    if (basket.length === 0) return;
    const perCoin = (cash * 0.99) / basket.length;
    for (const coin of basket) {
      await applyBuy(agentId, TREND_AGENT_NAME, coin, perCoin);
    }
    logger.info({ agentId, ret30d: (ret * 100).toFixed(2) + "%", basketSize: basket.length }, "Trend filter: entered basket");
  } else if (ret < 0 && invested) {
    await liquidateAll(agentId, TREND_AGENT_NAME, prices);
    logger.info({ agentId, ret30d: (ret * 100).toFixed(2) + "%" }, "Trend filter: exited to cash");
  }
}

async function recordSnapshot(strategyType: string, agentIds: number[], prices: CoinPrice[]): Promise<void> {
  let totalEquity = 0;
  let totalCash = 0;
  let totalInvested = 0;
  for (const id of agentIds) {
    const { equity, cash, invested } = await computePortfolioEquity(id, prices);
    totalEquity += equity;
    totalCash += cash;
    totalInvested += invested;
  }
  await db.insert(strategySnapshotsTable).values({
    strategyType,
    equity: totalEquity,
    cashBalance: totalCash,
    investedValue: totalInvested,
  });
}

let cyclesSinceSnapshot = 0;

export async function runStrategyLabCycle(prices: CoinPrice[]): Promise<void> {
  if (prices.length === 0) return;
  try {
    const { dcaId, hodlId, trendId } = await ensureStrategyLabAgents();
    await runHodlStrategy(hodlId, prices);
    await runDcaStrategy(dcaId, prices);
    await runTrendFilterStrategy(trendId, prices);

    cyclesSinceSnapshot++;
    if (cyclesSinceSnapshot >= SNAPSHOT_INTERVAL_CYCLES) {
      cyclesSinceSnapshot = 0;
      const aiAgents = await db.select({ id: agentsTable.id }).from(agentsTable)
        .where(eq(agentsTable.strategyType, "ai-bots"));
      const aiIds = aiAgents.map(a => a.id);
      if (aiIds.length > 0) await recordSnapshot("ai-bots", aiIds, prices);
      await recordSnapshot("dca-cb", [dcaId], prices);
      await recordSnapshot("buy-hold", [hodlId], prices);
      await recordSnapshot("trend-filter", [trendId], prices);

      // Time-based retention: keep ~60 days of snapshots
      const RETENTION_MS = 60 * 24 * 60 * 60 * 1000;
      const cutoffTs = new Date(Date.now() - RETENTION_MS);
      await db.delete(strategySnapshotsTable).where(sql`${strategySnapshotsTable.timestamp} < ${cutoffTs}`);
    }
  } catch (err) {
    logger.error({ err }, "Strategy lab cycle failed");
  }
}

export interface StrategyDecision {
  headline: string;
  detail?: string;
  tone: "good" | "warn" | "neutral";
  occurredAt?: string;
}

export interface StrategyBucketStats {
  strategyType: string;
  label: string;
  agentCount: number;
  startingCapital: number;
  currentEquity: number;
  cash: number;
  invested: number;
  totalPnl: number;
  totalPnlPct: number;
  totalTrades: number;
  totalFees: number;
  peakValue: number;
  maxDrawdownPct: number;
  circuitBreakerActive?: boolean;
  latestDecision?: StrategyDecision;
}

function formatRelative(ms: number): string {
  const abs = Math.abs(ms);
  const minutes = Math.round(abs / 60000);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.round(abs / 3600000);
  if (hours < 48) return `${hours}h`;
  const days = Math.round(abs / 86400000);
  return `${days}d`;
}

function formatDate(d: Date): string {
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

async function buildLatestDecision(
  strategyType: string,
  agentIds: number[],
  prices: CoinPrice[],
): Promise<StrategyDecision | undefined> {
  if (agentIds.length === 0) return undefined;

  if (strategyType === "dca-cb") {
    const states = await db.select().from(strategyStateTable).where(eq(strategyStateTable.agentId, agentIds[0]));
    if (states.length === 0) return undefined;
    const state = states[0];
    const settings = await getDcaSettings();
    if (state.circuitBreakerActive && state.circuitBreakerActivatedAt) {
      const activatedAt = state.circuitBreakerActivatedAt;
      return {
        headline: `Breaker tripped on ${formatDate(activatedAt)}`,
        detail: `Liquidated to cash after a ${settings.drawdownTriggerPct.toFixed(0)}% drawdown. Will resume when ${settings.resumeLookbackDays}d basket return turns positive.`,
        tone: "warn",
        occurredAt: activatedAt.toISOString(),
      };
    }
    const lastBuy = state.lastBuyAt;
    if (!lastBuy) {
      return { headline: "Awaiting first DCA buy", tone: "neutral" };
    }
    const buyIntervalMs = settings.buyIntervalHours * 60 * 60 * 1000;
    const nextBuyMs = lastBuy.getTime() + buyIntervalMs;
    const remaining = nextBuyMs - Date.now();
    if (remaining <= 0) {
      return {
        headline: `DCA buy due now (last ${formatRelative(Date.now() - lastBuy.getTime())} ago)`,
        detail: `Deploys ~$${settings.cycleDeployUsd.toFixed(2)} across the basket each cycle.`,
        tone: "good",
        occurredAt: lastBuy.toISOString(),
      };
    }
    return {
      headline: `Next DCA buy in ${formatRelative(remaining)}`,
      detail: `Last buy ${formatRelative(Date.now() - lastBuy.getTime())} ago. Deploys ~$${settings.cycleDeployUsd.toFixed(2)} every ${settings.buyIntervalHours}h.`,
      tone: "neutral",
      occurredAt: lastBuy.toISOString(),
    };
  }

  if (strategyType === "trend-filter") {
    const ret = await basketAvgReturnSinceMs(prices, TREND_LOOKBACK_MS, agentIds[0]);
    const positions = await db.select({ id: paperPositionsTable.id })
      .from(paperPositionsTable)
      .where(eq(paperPositionsTable.agentId, agentIds[0]));
    const invested = positions.length > 0;
    const lastTrade = await db.select().from(paperTradesTable)
      .where(eq(paperTradesTable.agentId, agentIds[0]))
      .orderBy(desc(paperTradesTable.createdAt))
      .limit(1);
    const retStr = ret === null ? "n/a" : `${(ret * 100).toFixed(2)}%`;
    if (ret === null) {
      return { headline: "Waiting for 30d basket history", tone: "neutral" };
    }
    if (invested) {
      const since = lastTrade.length > 0 ? lastTrade[0].createdAt : null;
      return {
        headline: `Holding basket — 30d return ${retStr}`,
        detail: since ? `Entered ${formatRelative(Date.now() - since.getTime())} ago. Will exit if 30d turns negative.` : undefined,
        tone: "good",
        occurredAt: since?.toISOString(),
      };
    }
    const since = lastTrade.length > 0 ? (lastTrade[0].closedAt ?? lastTrade[0].createdAt) : null;
    return {
      headline: `In cash — 30d return ${retStr}`,
      detail: since ? `Exited ${formatRelative(Date.now() - since.getTime())} ago. Will re-enter when 30d turns positive.` : "Will enter when 30d basket return turns positive.",
      tone: ret < 0 ? "warn" : "neutral",
      occurredAt: since?.toISOString(),
    };
  }

  if (strategyType === "buy-hold") {
    const states = await db.select().from(strategyStateTable).where(eq(strategyStateTable.agentId, agentIds[0]));
    if (states.length === 0) return undefined;
    const state = states[0];
    if (!state.initialDeployDone) {
      return { headline: "Awaiting initial deploy", tone: "neutral" };
    }
    const firstBuy = await db.select({ createdAt: paperTradesTable.createdAt })
      .from(paperTradesTable)
      .where(eq(paperTradesTable.agentId, agentIds[0]))
      .orderBy(paperTradesTable.createdAt)
      .limit(1);
    const at = firstBuy.length > 0 ? firstBuy[0].createdAt : null;
    return {
      headline: at ? `Deployed ${formatRelative(Date.now() - at.getTime())} ago` : "Deployed",
      detail: at ? `Equal-weight buy on ${formatDate(at)}. No further actions.` : "Equal-weight buy across the basket. No further actions.",
      tone: "neutral",
      occurredAt: at?.toISOString(),
    };
  }

  // ai-bots — last trade across the fleet
  const recent = await db.select().from(paperTradesTable)
    .where(inArray(paperTradesTable.agentId, agentIds))
    .orderBy(desc(paperTradesTable.createdAt))
    .limit(1);
  if (recent.length === 0) {
    return { headline: "No trades yet", tone: "neutral" };
  }
  const t = recent[0];
  const ts = t.closedAt ?? t.createdAt;
  const ago = formatRelative(Date.now() - ts.getTime());
  if (t.action === "close") {
    const pnlPct = t.pnlPercent ?? 0;
    const sign = pnlPct >= 0 ? "+" : "";
    return {
      headline: `Closed ${t.coinName} ${sign}${pnlPct.toFixed(2)}% (${ago} ago)`,
      detail: `${t.agentName} exited the position.`,
      tone: pnlPct >= 0 ? "good" : "warn",
      occurredAt: ts.toISOString(),
    };
  }
  return {
    headline: `Bought ${t.coinName} ${ago} ago`,
    detail: `${t.agentName} opened a $${t.positionSize.toFixed(2)} position.`,
    tone: "good",
    occurredAt: ts.toISOString(),
  };
}

export interface SettingsChangeEntry {
  field: "drawdownTriggerPct" | "resumeLookbackDays" | "cycleDeployUsd";
  label: string;
  before: number;
  after: number;
  formattedBefore: string;
  formattedAfter: string;
}

export interface SettingsHistoryEvent {
  id: number;
  timestamp: string;
  changes: SettingsChangeEntry[];
}

export interface StrategyComparison {
  buckets: StrategyBucketStats[];
  equityCurves: Record<string, { timestamp: string; equity: number }[]>;
  settingsHistory: Record<string, SettingsHistoryEvent[]>;
  generatedAt: string;
}

async function summarizeBucket(strategyType: string, label: string, agentIds: number[], prices: CoinPrice[]): Promise<StrategyBucketStats> {
  let cash = 0, invested = 0, equity = 0;
  for (const id of agentIds) {
    const r = await computePortfolioEquity(id, prices);
    cash += r.cash; invested += r.invested; equity += r.equity;
  }
  const startingCapital = agentIds.length * INITIAL_CAPITAL;

  let totalTrades = 0;
  let totalFees = 0;
  let circuitBreakerActive: boolean | undefined;

  if (agentIds.length > 0) {
    const portfolios = await db.select().from(paperPortfoliosTable);
    for (const p of portfolios) {
      if (agentIds.includes(p.agentId)) {
        totalTrades += p.totalTrades;
      }
    }
    const states = await db.select().from(strategyStateTable);
    for (const s of states) {
      if (agentIds.includes(s.agentId)) {
        totalFees += s.totalFees;
        if (strategyType === "dca-cb") circuitBreakerActive = s.circuitBreakerActive;
      }
    }
  }

  // True historical max drawdown from snapshot peak-to-trough walk.
  const history = await db.select({ equity: strategySnapshotsTable.equity })
    .from(strategySnapshotsTable)
    .where(eq(strategySnapshotsTable.strategyType, strategyType))
    .orderBy(strategySnapshotsTable.timestamp);
  let runningPeak = startingCapital;
  let maxDrawdown = 0;
  let allTimePeak = startingCapital;
  for (const row of history) {
    if (row.equity > runningPeak) runningPeak = row.equity;
    if (row.equity > allTimePeak) allTimePeak = row.equity;
    const dd = runningPeak > 0 ? (runningPeak - row.equity) / runningPeak : 0;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }
  // Also factor in the live (un-snapshotted) equity vs running peak
  if (equity < runningPeak) {
    const liveDd = (runningPeak - equity) / runningPeak;
    if (liveDd > maxDrawdown) maxDrawdown = liveDd;
  }
  const peakValue = Math.max(allTimePeak, equity);
  const drawdown = maxDrawdown * 100;
  const totalPnl = equity - startingCapital;
  const totalPnlPct = startingCapital > 0 ? (totalPnl / startingCapital) * 100 : 0;

  const latestDecision = await buildLatestDecision(strategyType, agentIds, prices);

  return {
    strategyType,
    label,
    agentCount: agentIds.length,
    startingCapital,
    currentEquity: equity,
    cash,
    invested,
    totalPnl,
    totalPnlPct,
    totalTrades,
    totalFees,
    peakValue,
    maxDrawdownPct: drawdown,
    circuitBreakerActive,
    latestDecision,
  };
}

export async function getStrategyComparison(): Promise<StrategyComparison> {
  const prices = isPriceDataFresh() ? await fetchCoinPrices() : await fetchCoinPrices().catch(() => []);

  const aiAgents = await db.select({ id: agentsTable.id }).from(agentsTable)
    .where(eq(agentsTable.strategyType, "ai-bots"));
  const dcaAgents = await db.select({ id: agentsTable.id }).from(agentsTable)
    .where(eq(agentsTable.strategyType, "dca-cb"));
  const hodlAgents = await db.select({ id: agentsTable.id }).from(agentsTable)
    .where(eq(agentsTable.strategyType, "buy-hold"));
  const trendAgents = await db.select({ id: agentsTable.id }).from(agentsTable)
    .where(eq(agentsTable.strategyType, "trend-filter"));

  const aiIds = aiAgents.map(a => a.id);
  const dcaIds = dcaAgents.map(a => a.id);
  const hodlIds = hodlAgents.map(a => a.id);
  const trendIds = trendAgents.map(a => a.id);

  const buckets: StrategyBucketStats[] = [
    await summarizeBucket("ai-bots", "Quant Fleet (aggregate)", aiIds, prices),
    await summarizeBucket("dca-cb", "DCA + Circuit Breaker", dcaIds, prices),
    await summarizeBucket("buy-hold", "Buy & Hold (basket)", hodlIds, prices),
    await summarizeBucket("trend-filter", "Trend Filter (30d basket)", trendIds, prices),
  ];

  const equityCurves: Record<string, { timestamp: string; equity: number }[]> = {};
  for (const bucket of ["ai-bots", "dca-cb", "buy-hold", "trend-filter"]) {
    const rows = await db.select({ timestamp: strategySnapshotsTable.timestamp, equity: strategySnapshotsTable.equity })
      .from(strategySnapshotsTable)
      .where(eq(strategySnapshotsTable.strategyType, bucket))
      .orderBy(strategySnapshotsTable.timestamp)
      .limit(500);
    equityCurves[bucket] = rows.map(r => ({ timestamp: r.timestamp.toISOString(), equity: r.equity }));
  }

  const settingsHistory: Record<string, SettingsHistoryEvent[]> = { "dca-cb": [] };
  const dcaHistoryRows = await db.select().from(strategySettingsHistoryTable)
    .where(eq(strategySettingsHistoryTable.strategyType, "dca-cb"))
    .orderBy(strategySettingsHistoryTable.timestamp);
  settingsHistory["dca-cb"] = dcaHistoryRows.map(row => {
    const changes: SettingsChangeEntry[] = [];
    if (row.drawdownTriggerBefore !== null && row.drawdownTriggerAfter !== null) {
      changes.push({
        field: "drawdownTriggerPct",
        label: "Drawdown trigger",
        before: row.drawdownTriggerBefore,
        after: row.drawdownTriggerAfter,
        formattedBefore: `${row.drawdownTriggerBefore.toFixed(0)}%`,
        formattedAfter: `${row.drawdownTriggerAfter.toFixed(0)}%`,
      });
    }
    if (row.resumeLookbackBefore !== null && row.resumeLookbackAfter !== null) {
      changes.push({
        field: "resumeLookbackDays",
        label: "Resume lookback",
        before: row.resumeLookbackBefore,
        after: row.resumeLookbackAfter,
        formattedBefore: `${row.resumeLookbackBefore}d`,
        formattedAfter: `${row.resumeLookbackAfter}d`,
      });
    }
    if (row.cycleDeployBefore !== null && row.cycleDeployAfter !== null) {
      changes.push({
        field: "cycleDeployUsd",
        label: "Per-cycle deploy",
        before: row.cycleDeployBefore,
        after: row.cycleDeployAfter,
        formattedBefore: `$${row.cycleDeployBefore.toFixed(2)}`,
        formattedAfter: `$${row.cycleDeployAfter.toFixed(2)}`,
      });
    }
    return {
      id: row.id,
      timestamp: row.timestamp.toISOString(),
      changes,
    };
  });

  return { buckets, equityCurves, settingsHistory, generatedAt: new Date().toISOString() };
}
