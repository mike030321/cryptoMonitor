import { db, marketSignalsTable } from "@workspace/db";
import { logger } from "./logger";
import { MONITORED_COINS } from "./coins";

/**
 * Task #271 — scheduled poller that writes per-coin snapshots of the
 * external exchange signals registered by the training contract (rule 5):
 *   funding_rate, open_interest_usd, liquidations_1h_usd, bid_ask_spread_bps
 * Plus a `mid_price` snapshot for the synthetic coin ids `btc` and `eth`
 * so the trainer can compute `btc_lead_ret_5m` / `eth_lead_ret_5m` for
 * every other coin.
 *
 * Source: OKX USDT-margined perpetual swap public REST endpoints (no API
 * key required and reachable from the Replit container, unlike Binance
 * and Bybit which are geo-blocked here).
 *
 * Task #286 — `liquidations_1h_usd` is now aggregated across multiple
 * exchanges instead of OKX-only. We add Gate.io USDT-margined futures as
 * a second source. The per-source USD totals are recorded in the
 * `source_breakdown` jsonb column so we can audit each exchange's
 * contribution. If the secondary source fails we silently fall back to
 * OKX-only for that coin (the row is still written, the breakdown just
 * omits the failing source). BTC / ETH / SOL are also polled for
 * liquidations now, since they're the dominant perp-volume names; their
 * rows are written under pseudo-coin ids `btc`, `eth`, `sol`.
 *
 * Task #294 — Coinglass joins as an optional third source. Coinglass is
 * the canonical multi-exchange aggregator (Binance + Bybit + Bitget etc.,
 * which are geo-blocked from the Replit container) so when a key is
 * provisioned via the `COINGLASS_API_KEY` secret we sum its 1h aggregated
 * liquidation USD into the breakdown alongside okx + gate. With no key,
 * or on outage, the poller silently falls back to whichever sources did
 * succeed (still OKX+Gate at minimum) — no key means nothing changes.
 *
 * Each coin's instId is `<okxBase>-USDT-SWAP`. Where a monitored coin
 * isn't listed on OKX SWAP the row is silently skipped (rather than
 * written as an artificial zero, per training contract rule 5).
 */

// Coin id -> OKX SWAP base symbol. Instrument id is `${base}-USDT-SWAP`.
// Coins not listed here are silently skipped by the poller.
const OKX_SWAP_BASE: Record<string, string> = {
  pepe: "PEPE",
  "floki-inu": "FLOKI",
  bonk: "BONK",
  dogwifcoin: "WIF",
  "render-token": "RENDER",
  "injective-protocol": "INJ",
  "sei-network": "SEI",
  celestia: "TIA",
  "jupiter-exchange-solana": "JUP",
  "worldcoin-wld": "WLD",
};

// Pseudo-coin ids used to record cross-market reference prices for the
// `btc_lead_ret_5m` / `eth_lead_ret_5m` features. These are also the
// top-3 perp-volume names so we additionally aggregate liquidations for
// them (task #286).
const LEAD_REFERENCES: Array<{ coinId: string; base: string }> = [
  { coinId: "btc", base: "BTC" },
  { coinId: "eth", base: "ETH" },
  { coinId: "sol", base: "SOL" },
];

// Coin ids whose `liquidations_1h_usd` should be aggregated across
// OKX + Gate.io. These are the highest-volume perp names — OKX alone
// captures only a small share of true liquidation activity for them.
const MULTI_SOURCE_LIQ_COIN_IDS: ReadonlySet<string> = new Set([
  "btc",
  "eth",
  "sol",
]);

// Gate.io USDT-margined futures contract symbol per coin id. Only coins
// listed here are queried on Gate.io for the secondary liquidations
// source — we keep the list explicit to avoid blasting Gate.io with
// guess-shaped requests for low-cap names that aren't listed there.
const GATE_USDT_CONTRACT: Record<string, string> = {
  btc: "BTC_USDT",
  eth: "ETH_USDT",
  sol: "SOL_USDT",
  pepe: "PEPE_USDT",
  "floki-inu": "FLOKI_USDT",
  bonk: "BONK_USDT",
  dogwifcoin: "WIF_USDT",
  "render-token": "RENDER_USDT",
  "injective-protocol": "INJ_USDT",
  "sei-network": "SEI_USDT",
  celestia: "TIA_USDT",
  "jupiter-exchange-solana": "JUP_USDT",
  "worldcoin-wld": "WLD_USDT",
};

// Coinglass base symbol per coin id. Coinglass uses bare base symbols
// (e.g. BTC, ETH, SOL, PEPE) for its aggregated futures endpoints. We
// keep the list explicit (mirroring Gate.io) so we don't make
// guess-shaped requests for low-cap names whose Coinglass coverage is
// thin or absent.
const COINGLASS_BASE_SYMBOL: Record<string, string> = {
  btc: "BTC",
  eth: "ETH",
  sol: "SOL",
  pepe: "PEPE",
  "floki-inu": "FLOKI",
  bonk: "BONK",
  dogwifcoin: "WIF",
  "render-token": "RENDER",
  "injective-protocol": "INJ",
  "sei-network": "SEI",
  celestia: "TIA",
  "jupiter-exchange-solana": "JUP",
  "worldcoin-wld": "WLD",
};

const OKX_BASE = "https://www.okx.com";
const GATE_BASE = "https://api.gateio.ws";
const COINGLASS_BASE = "https://open-api-v4.coinglass.com";
const POLL_INTERVAL_MS = 60_000; // 1 minute
const FETCH_TIMEOUT_MS = 8_000;

let pollerInterval: ReturnType<typeof setInterval> | null = null;
let lastPollAt: number | null = null;
let lastPollOk = false;
let lastPollError: string | null = null;
// Guard against overlapping polls: if an upstream call hangs near the
// FETCH_TIMEOUT_MS budget for many targets, the next 60s tick could fire
// before the previous run finished. We just skip the new tick rather than
// stack two parallel passes hammering the same endpoints.
let pollInFlight = false;

// Cached contract value (base-currency units per contract) per OKX
// instId. Required to convert liquidation `sz` (contracts) into USD
// notional.
const OKX_CT_VAL_CACHE = new Map<string, number>();

// Cached `quanto_multiplier` (base-currency units per contract) per
// Gate.io USDT-futures contract. Same role as OKX's ctVal.
const GATE_QUANTO_CACHE = new Map<string, number>();

interface SignalSnapshot {
  fundingRate: number | null;
  openInterestUsd: number | null;
  bidAskSpreadBps: number | null;
  midPrice: number | null;
  liquidations1hUsd: number | null;
  liquidationsBreakdown: Record<string, number> | null;
}

interface OkxResponse<T> {
  code?: string;
  msg?: string;
  data?: T;
}

async function fetchJson<T>(url: string): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, { signal: ctrl.signal });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} for ${url}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

async function fetchOkx<T>(path: string): Promise<T[]> {
  const data = await fetchJson<OkxResponse<T[]>>(`${OKX_BASE}${path}`);
  if (!data || data.code !== "0" || !Array.isArray(data.data)) {
    throw new Error(`OKX error code=${data?.code ?? "?"} msg=${data?.msg ?? "?"}`);
  }
  return data.data;
}

async function getOkxCtVal(instId: string): Promise<number | null> {
  const cached = OKX_CT_VAL_CACHE.get(instId);
  if (cached != null) return cached;
  try {
    const rows = await fetchOkx<{ ctVal?: string }>(
      `/api/v5/public/instruments?instType=SWAP&instId=${instId}`,
    );
    const v = Number(rows[0]?.ctVal);
    if (Number.isFinite(v) && v > 0) {
      OKX_CT_VAL_CACHE.set(instId, v);
      return v;
    }
    return null;
  } catch (err) {
    logger.debug({ err, instId }, "okx ctVal fetch failed");
    return null;
  }
}

async function fetchFunding(instId: string): Promise<number | null> {
  try {
    const rows = await fetchOkx<{ fundingRate?: string }>(
      `/api/v5/public/funding-rate?instId=${instId}`,
    );
    const v = Number(rows[0]?.fundingRate);
    return Number.isFinite(v) ? v : null;
  } catch (err) {
    logger.debug({ err, instId }, "funding fetch failed");
    return null;
  }
}

async function fetchOpenInterestUsd(instId: string): Promise<number | null> {
  try {
    const rows = await fetchOkx<{ oiUsd?: string }>(
      `/api/v5/public/open-interest?instType=SWAP&instId=${instId}`,
    );
    const v = Number(rows[0]?.oiUsd);
    return Number.isFinite(v) && v > 0 ? v : null;
  } catch (err) {
    logger.debug({ err, instId }, "OI fetch failed");
    return null;
  }
}

async function fetchTicker(instId: string): Promise<{ spreadBps: number | null; mid: number | null }> {
  try {
    const rows = await fetchOkx<{ bidPx?: string; askPx?: string }>(
      `/api/v5/market/ticker?instId=${instId}`,
    );
    const bid = Number(rows[0]?.bidPx);
    const ask = Number(rows[0]?.askPx);
    if (!Number.isFinite(bid) || !Number.isFinite(ask) || bid <= 0 || ask <= 0) {
      return { spreadBps: null, mid: null };
    }
    const mid = (bid + ask) / 2;
    const spreadBps = ((ask - bid) / mid) * 10_000;
    return { spreadBps, mid };
  } catch (err) {
    logger.debug({ err, instId }, "ticker fetch failed");
    return { spreadBps: null, mid: null };
  }
}

interface LiqDetail {
  bkPx?: string;
  sz?: string;
  ts?: string;
  time?: number;
}
interface LiqRow {
  details?: LiqDetail[];
  ts?: string;
}

async function fetchOkxLiquidations1hUsd(base: string, instId: string): Promise<number | null> {
  // OKX returns the latest liquidations for an underlying (e.g. BTC-USDT)
  // covering all expiries; we filter to the SWAP instId and the trailing
  // 1h window. Pagination uses `after=<ms>` to walk older pages until
  // we've covered the full hour. Hard cap at 5 pages.
  const since = Date.now() - 60 * 60 * 1000;
  const ctVal = await getOkxCtVal(instId);
  if (ctVal == null) return null;
  let total = 0;
  let cursor: string | null = null;
  let pages = 0;
  let foundAny = false;
  try {
    while (pages < 5) {
      const path = cursor == null
        ? `/api/v5/public/liquidation-orders?instType=SWAP&uly=${base}-USDT&state=filled&limit=100`
        : `/api/v5/public/liquidation-orders?instType=SWAP&uly=${base}-USDT&state=filled&limit=100&after=${cursor}`;
      const rows: LiqRow[] = await fetchOkx<LiqRow>(path);
      if (rows.length === 0) break;
      let oldestTs: number | null = null;
      let outOfWindow = false;
      let detailCount = 0;
      for (const row of rows) {
        const details = Array.isArray(row.details) ? row.details : [];
        for (const d of details) {
          detailCount += 1;
          const ts = Number(d.ts ?? d.time);
          if (!Number.isFinite(ts)) continue;
          if (ts < since) { outOfWindow = true; continue; }
          const px = Number(d.bkPx);
          const sz = Number(d.sz);
          if (!Number.isFinite(px) || !Number.isFinite(sz)) continue;
          total += px * sz * ctVal;
          foundAny = true;
          if (oldestTs == null || ts < oldestTs) oldestTs = ts;
        }
      }
      pages += 1;
      if (detailCount === 0 || outOfWindow || oldestTs == null) break;
      cursor = String(oldestTs);
    }
    return foundAny ? total : 0;
  } catch (err) {
    logger.debug({ err, base }, "okx liquidations fetch failed");
    return null;
  }
}

interface GateContract {
  name?: string;
  quanto_multiplier?: string;
}

async function getGateQuanto(contract: string): Promise<number | null> {
  const cached = GATE_QUANTO_CACHE.get(contract);
  if (cached != null) return cached;
  try {
    const row = await fetchJson<GateContract>(
      `${GATE_BASE}/api/v4/futures/usdt/contracts/${contract}`,
    );
    const v = Number(row?.quanto_multiplier);
    if (Number.isFinite(v) && v > 0) {
      GATE_QUANTO_CACHE.set(contract, v);
      return v;
    }
    return null;
  } catch (err) {
    logger.debug({ err, contract }, "gate quanto fetch failed");
    return null;
  }
}

interface GateLiqOrder {
  time?: number;
  time_ms?: number;
  size?: number;
  fill_price?: string;
  order_price?: string;
}

async function fetchGateLiquidations1hUsd(contract: string): Promise<number | null> {
  // Gate.io public USDT-futures liquidation history. Returns up to 1000
  // recent liquidations for the contract; we filter to the trailing 1h.
  // The endpoint is unauthenticated and does not require start/end
  // params — when omitted it returns the most recent records, which is
  // what we want for an aggregating poller running every minute.
  const since = Date.now() - 60 * 60 * 1000;
  const quanto = await getGateQuanto(contract);
  if (quanto == null) return null;
  let rows: GateLiqOrder[];
  try {
    rows = await fetchJson<GateLiqOrder[]>(
      `${GATE_BASE}/api/v4/futures/usdt/liq_orders?contract=${contract}&limit=1000`,
    );
  } catch (err) {
    logger.debug({ err, contract }, "gate liquidations fetch failed");
    return null;
  }
  if (!Array.isArray(rows)) return null;
  let total = 0;
  let foundAny = false;
  for (const r of rows) {
    const tsMs = Number(
      r.time_ms != null ? r.time_ms : (r.time != null ? r.time * 1000 : NaN),
    );
    if (!Number.isFinite(tsMs) || tsMs < since) continue;
    const px = Number(r.fill_price ?? r.order_price);
    const sz = Number(r.size);
    if (!Number.isFinite(px) || !Number.isFinite(sz)) continue;
    // size is signed (positive=long liq, negative=short liq); we want
    // absolute USD notional.
    total += Math.abs(sz) * quanto * px;
    foundAny = true;
  }
  return foundAny ? total : 0;
}

interface CoinglassResponse<T> {
  code?: string;
  msg?: string;
  data?: T;
}

interface CoinglassLiqBar {
  time?: number;
  longLiquidationUsd?: string | number;
  shortLiquidationUsd?: string | number;
  aggregatedLongLiquidationUsd?: string | number;
  aggregatedShortLiquidationUsd?: string | number;
}

function getCoinglassApiKey(): string | null {
  const k = process.env.COINGLASS_API_KEY;
  return k && k.trim().length > 0 ? k.trim() : null;
}

async function fetchCoinglassLiquidations1hUsd(
  symbol: string,
  apiKey: string,
): Promise<number | null> {
  // Coinglass v4 aggregated-history returns per-interval long/short
  // liquidation USD totals across all tracked exchanges. We pull the
  // most recent 1h bar — that already covers the trailing hour the
  // poller cares about, with no client-side bucketing needed.
  const url = `${COINGLASS_BASE}/api/futures/liquidation/aggregated-history?symbol=${symbol}&interval=1h&limit=1`;
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      signal: ctrl.signal,
      headers: { "CG-API-KEY": apiKey, accept: "application/json" },
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} for ${url}`);
    }
    const body = (await res.json()) as CoinglassResponse<CoinglassLiqBar[]>;
    if (!body || body.code !== "0" || !Array.isArray(body.data)) {
      throw new Error(
        `Coinglass error code=${body?.code ?? "?"} msg=${body?.msg ?? "?"}`,
      );
    }
    const bar = body.data[0];
    if (!bar) return 0;
    const longUsd = Number(
      bar.aggregatedLongLiquidationUsd ?? bar.longLiquidationUsd,
    );
    const shortUsd = Number(
      bar.aggregatedShortLiquidationUsd ?? bar.shortLiquidationUsd,
    );
    const lo = Number.isFinite(longUsd) ? longUsd : 0;
    const sh = Number.isFinite(shortUsd) ? shortUsd : 0;
    return lo + sh;
  } catch (err) {
    logger.debug({ err, symbol }, "coinglass liquidations fetch failed");
    return null;
  } finally {
    clearTimeout(t);
  }
}

interface AggregatedLiq {
  totalUsd: number | null;
  breakdown: Record<string, number> | null;
}

async function fetchAggregatedLiquidations(
  coinId: string,
  base: string,
  instId: string,
): Promise<AggregatedLiq> {
  const wantSecondary =
    MULTI_SOURCE_LIQ_COIN_IDS.has(coinId) || GATE_USDT_CONTRACT[coinId] != null;
  const gateContract = GATE_USDT_CONTRACT[coinId] ?? null;
  const coinglassSymbol = COINGLASS_BASE_SYMBOL[coinId] ?? null;
  const coinglassKey = getCoinglassApiKey();

  const [okx, gate, coinglass] = await Promise.all([
    fetchOkxLiquidations1hUsd(base, instId),
    wantSecondary && gateContract != null
      ? fetchGateLiquidations1hUsd(gateContract)
      : Promise.resolve<number | null>(null),
    coinglassKey != null && coinglassSymbol != null
      ? fetchCoinglassLiquidations1hUsd(coinglassSymbol, coinglassKey)
      : Promise.resolve<number | null>(null),
  ]);

  const breakdown: Record<string, number> = {};
  if (okx != null) breakdown.okx = okx;
  if (gate != null) breakdown.gate = gate;
  if (coinglass != null) breakdown.coinglass = coinglass;

  if (okx == null && gate == null && coinglass == null) {
    return { totalUsd: null, breakdown: null };
  }

  const total = (okx ?? 0) + (gate ?? 0) + (coinglass ?? 0);
  return {
    totalUsd: total,
    breakdown: Object.keys(breakdown).length > 0 ? breakdown : null,
  };
}

async function pollOne(
  coinId: string,
  base: string,
  includeLiquidations: boolean,
): Promise<SignalSnapshot> {
  const instId = `${base}-USDT-SWAP`;
  const [funding, ticker, oi, liq] = await Promise.all([
    fetchFunding(instId),
    fetchTicker(instId),
    fetchOpenInterestUsd(instId),
    includeLiquidations
      ? fetchAggregatedLiquidations(coinId, base, instId)
      : Promise.resolve<AggregatedLiq>({ totalUsd: null, breakdown: null }),
  ]);
  return {
    fundingRate: funding,
    openInterestUsd: oi,
    bidAskSpreadBps: ticker.spreadBps,
    midPrice: ticker.mid,
    liquidations1hUsd: liq.totalUsd,
    liquidationsBreakdown: liq.breakdown,
  };
}

function sourceLabelFor(breakdown: Record<string, number> | null): string {
  // Distinct label when no liquidation source succeeded so analytics
  // querying `source` alone can spot total-aggregator outages without
  // having to inspect `source_breakdown` and `liquidations_1h_usd`.
  if (!breakdown) return "no_liq_source";
  const parts: string[] = [];
  if ("okx" in breakdown) parts.push("okx_swap");
  if ("gate" in breakdown) parts.push("gate_swap");
  if ("coinglass" in breakdown) parts.push("coinglass");
  return parts.length > 0 ? parts.join("+") : "no_liq_source";
}

async function runPollOnce(): Promise<void> {
  if (pollInFlight) {
    logger.debug("market signals poll skipped (previous run still in flight)");
    return;
  }
  pollInFlight = true;
  try {
    await runPollOnceInner();
  } finally {
    pollInFlight = false;
  }
}

async function runPollOnceInner(): Promise<void> {
  const targets: Array<{ coinId: string; base: string; includeLiquidations: boolean }> = [];
  for (const coin of MONITORED_COINS) {
    const sym = OKX_SWAP_BASE[coin.id];
    if (!sym) continue;
    targets.push({ coinId: coin.id, base: sym, includeLiquidations: true });
  }
  for (const ref of LEAD_REFERENCES) {
    // Lead references carry the cross-coin mid-price feature. Task #286
    // also aggregates liquidations for these (BTC/ETH/SOL are the top
    // perp names) so the multi-exchange feed has somewhere to land.
    targets.push({ coinId: ref.coinId, base: ref.base, includeLiquidations: true });
  }

  const stamp = new Date();
  const rows: Array<typeof marketSignalsTable.$inferInsert> = [];
  let okCount = 0;
  let failCount = 0;
  for (const t of targets) {
    try {
      const snap = await pollOne(t.coinId, t.base, t.includeLiquidations);
      // Only persist if at least one signal field is populated; an
      // entirely-null row would just add noise to feature_density.
      if (
        snap.fundingRate == null && snap.openInterestUsd == null &&
        snap.bidAskSpreadBps == null && snap.midPrice == null &&
        snap.liquidations1hUsd == null
      ) {
        failCount += 1;
        continue;
      }
      rows.push({
        coinId: t.coinId,
        timestamp: stamp,
        fundingRate: snap.fundingRate,
        openInterestUsd: snap.openInterestUsd,
        liquidations1hUsd: snap.liquidations1hUsd,
        bidAskSpreadBps: snap.bidAskSpreadBps,
        midPrice: snap.midPrice,
        source: sourceLabelFor(snap.liquidationsBreakdown),
        sourceBreakdown: snap.liquidationsBreakdown,
      });
      okCount += 1;
    } catch (err) {
      failCount += 1;
      logger.debug({ err, coinId: t.coinId }, "market signal poll target failed");
    }
  }

  if (rows.length > 0) {
    try {
      await db.insert(marketSignalsTable).values(rows);
    } catch (err) {
      logger.error({ err }, "market_signals insert failed");
      throw err;
    }
  }

  lastPollAt = Date.now();
  lastPollOk = okCount > 0;
  lastPollError = okCount === 0 ? "all targets failed" : null;
  logger.info({ okCount, failCount, rows: rows.length }, "market signals poll complete");
}

export function startMarketSignalsPoller(): void {
  if (pollerInterval) return;
  // Kick off one immediate poll on startup so the model has a fresh row
  // before the next training tick, then settle into the normal cadence.
  void runPollOnce().catch((err) => {
    lastPollOk = false;
    lastPollError = String(err);
    logger.error({ err }, "initial market signals poll failed");
  });
  pollerInterval = setInterval(() => {
    void runPollOnce().catch((err) => {
      lastPollOk = false;
      lastPollError = String(err);
      logger.error({ err }, "market signals poll failed");
    });
  }, POLL_INTERVAL_MS);
  logger.info({ intervalMs: POLL_INTERVAL_MS }, "Market signals poller started");
}

export function stopMarketSignalsPoller(): void {
  if (pollerInterval) {
    clearInterval(pollerInterval);
    pollerInterval = null;
  }
}

/**
 * The canonical list of coin ids the poller is expected to write rows for
 * on every cycle. Used by the health endpoint so the dashboard can show a
 * red "0 rows" entry for a coin whose stream silently broke (otherwise
 * the missing coin would just be absent from the per-coin response and
 * the partial outage would be invisible).
 */
export function getMarketSignalsPollerTargets(): string[] {
  const ids = new Set<string>();
  for (const coin of MONITORED_COINS) {
    if (OKX_SWAP_BASE[coin.id]) ids.add(coin.id);
  }
  for (const ref of LEAD_REFERENCES) ids.add(ref.coinId);
  return Array.from(ids).sort();
}

export function getMarketSignalsPollerStatus(): {
  lastPollAt: number | null;
  lastPollOk: boolean;
  lastPollError: string | null;
  intervalMs: number;
} {
  return {
    lastPollAt,
    lastPollOk,
    lastPollError,
    intervalMs: POLL_INTERVAL_MS,
  };
}

// Task #286 — exported only for unit tests so they can validate the
// aggregation/fallback logic without spinning up the interval loop.
export const __testing = {
  fetchAggregatedLiquidations,
  sourceLabelFor,
  MULTI_SOURCE_LIQ_COIN_IDS,
  GATE_USDT_CONTRACT,
  COINGLASS_BASE_SYMBOL,
};
