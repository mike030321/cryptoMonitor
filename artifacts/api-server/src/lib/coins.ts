export interface CoinConfig {
  id: string;
  name: string;
  symbol: string;
  cmcSlug: string;
  cmcSymbol?: string;
}

export const MONITORED_COINS: CoinConfig[] = [
  { id: "pepe", name: "Pepe", symbol: "PEPE", cmcSlug: "pepe" },
  { id: "floki-inu", name: "Floki", symbol: "FLOKI", cmcSlug: "floki-inu" },
  { id: "bonk", name: "Bonk", symbol: "BONK", cmcSlug: "bonk1" },
  { id: "dogwifcoin", name: "dogwifhat", symbol: "WIF", cmcSlug: "dogwifhat" },
  { id: "render-token", name: "Render", symbol: "RNDR", cmcSlug: "render-token", cmcSymbol: "RENDER" },
  { id: "injective-protocol", name: "Injective", symbol: "INJ", cmcSlug: "injective" },
  { id: "sei-network", name: "Sei", symbol: "SEI", cmcSlug: "sei" },
  { id: "celestia", name: "Celestia", symbol: "TIA", cmcSlug: "celestia" },
  { id: "jupiter-exchange-solana", name: "Jupiter", symbol: "JUP", cmcSlug: "jupiter-ag" },
  { id: "worldcoin-wld", name: "Worldcoin", symbol: "WLD", cmcSlug: "worldcoin-org" },
];

export interface CoinPrice {
  id: string;
  name: string;
  symbol: string;
  currentPrice: number;
  priceChange24h: number;
  volume24h: number;
  marketCap: number;
  lastUpdated: string;
  isLiveData: boolean;
}

let cachedPrices: CoinPrice[] = [];
let lastFetchTime = 0;
let lastSuccessfulFetch = 0;

const CACHE_TTL = 30000;
const MAX_STALE_AGE = 90000;

export function isPriceDataFresh(): boolean {
  return lastSuccessfulFetch > 0 && Date.now() - lastSuccessfulFetch < MAX_STALE_AGE;
}

// Task #269 — test-only seam. Populates the in-memory price cache so
// `closeExpiredPositions()` can run deterministically inside an integration
// test without hitting CoinGecko / CoinMarketCap. Pass `null` to clear.
// Never called by production code.
export function __setCachedPricesForTest(prices: CoinPrice[] | null): void {
  if (prices === null) {
    cachedPrices = [];
    lastFetchTime = 0;
    lastSuccessfulFetch = 0;
    return;
  }
  cachedPrices = prices;
  lastFetchTime = Date.now();
  lastSuccessfulFetch = Date.now();
}

const CMC_API_KEY = process.env.COINMARKETCAP_API_KEY;
const COINGECKO_FALLBACK = !CMC_API_KEY;

async function fetchFromCoinMarketCap(): Promise<CoinPrice[]> {
  const cmcSymbols = new Set(MONITORED_COINS.map((c) => c.cmcSymbol || c.symbol));
  const symbols = [...cmcSymbols].join(",");
  const url = `https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest?symbol=${symbols}&convert=USD`;

  const response = await fetch(url, {
    headers: { "X-CMC_PRO_API_KEY": CMC_API_KEY! },
  });

  if (!response.ok) {
    throw new Error(`CMC API error: ${response.status}`);
  }

  const result = await response.json();
  if (result.status?.error_code) {
    throw new Error(`CMC API: ${result.status.error_message}`);
  }

  const prices: CoinPrice[] = [];

  for (const coin of MONITORED_COINS) {
    const lookupSymbol = coin.cmcSymbol || coin.symbol;
    const cmcData = result.data?.[lookupSymbol];
    const entry = Array.isArray(cmcData) ? cmcData[0] : cmcData;
    if (!entry) continue;

    const quote = entry.quote?.USD;
    if (!quote) continue;

    prices.push({
      id: coin.id,
      name: coin.name,
      symbol: coin.symbol,
      currentPrice: quote.price,
      priceChange24h: quote.percent_change_24h ?? 0,
      volume24h: quote.volume_24h ?? 0,
      marketCap: quote.market_cap ?? 0,
      lastUpdated: new Date().toISOString(),
      isLiveData: true,
    });
  }

  return prices;
}

async function fetchFromCoinGecko(): Promise<CoinPrice[]> {
  const ids = MONITORED_COINS.map((c) => c.id).join(",");
  const url = `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=${ids}&order=market_cap_desc&per_page=10&page=1&sparkline=false&price_change_percentage=24h`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`CoinGecko API error: ${response.status}`);

  const data = await response.json();
  if (!Array.isArray(data) || data.length === 0) throw new Error("CoinGecko: empty response");

  return data.map((coin: Record<string, unknown>) => ({
    id: coin.id as string,
    name: coin.name as string,
    symbol: (coin.symbol as string).toUpperCase(),
    currentPrice: coin.current_price as number,
    priceChange24h: (coin.price_change_percentage_24h as number) ?? 0,
    volume24h: coin.total_volume as number,
    marketCap: coin.market_cap as number,
    lastUpdated: new Date().toISOString(),
    isLiveData: true,
  }));
}

export interface FearGreedData {
  value: number;
  classification: string;
  updatedAt: string;
}

let cachedFearGreed: FearGreedData | null = null;
let lastFearGreedFetch = 0;
const FEAR_GREED_TTL = 300000;

export async function fetchFearGreedIndex(): Promise<FearGreedData | null> {
  const now = Date.now();
  if (now - lastFearGreedFetch < FEAR_GREED_TTL && cachedFearGreed) {
    return cachedFearGreed;
  }

  if (!CMC_API_KEY) return cachedFearGreed;

  try {
    const response = await fetch("https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest", {
      headers: { "X-CMC_PRO_API_KEY": CMC_API_KEY },
    });
    if (!response.ok) return cachedFearGreed;

    const result = await response.json();
    if (result.data) {
      cachedFearGreed = {
        value: result.data.value,
        classification: result.data.value_classification,
        updatedAt: result.data.update_time,
      };
      lastFearGreedFetch = now;
    }
    return cachedFearGreed;
  } catch {
    return cachedFearGreed;
  }
}

export interface BtcDominanceData {
  dominance: number;
  dominanceChange: number;
  btcPrice: number;
  btcChange24h: number;
}

let cachedBtcDominance: BtcDominanceData | null = null;
let previousBtcDominance: number | null = null;
let lastBtcDominanceFetch = 0;
const BTC_DOMINANCE_TTL = 300000;

export async function fetchBtcDominance(): Promise<BtcDominanceData | null> {
  const now = Date.now();
  if (now - lastBtcDominanceFetch < BTC_DOMINANCE_TTL && cachedBtcDominance) {
    return cachedBtcDominance;
  }

  try {
    const [globalRes, btcRes] = await Promise.all([
      fetch("https://api.coingecko.com/api/v3/global"),
      fetch("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=bitcoin&sparkline=false&price_change_percentage=24h"),
    ]);

    if (!globalRes.ok || !btcRes.ok) return cachedBtcDominance;

    const globalData = await globalRes.json();
    const btcData = await btcRes.json();
    const btcMarket = Array.isArray(btcData) && btcData[0] ? btcData[0] : null;

    const dominance = globalData?.data?.market_cap_percentage?.btc ?? null;
    if (dominance == null) return cachedBtcDominance;

    const domDelta = previousBtcDominance != null ? dominance - previousBtcDominance : 0;
    previousBtcDominance = dominance;

    cachedBtcDominance = {
      dominance,
      dominanceChange: domDelta,
      btcPrice: btcMarket?.current_price ?? 0,
      btcChange24h: btcMarket?.price_change_percentage_24h ?? 0,
    };
    lastBtcDominanceFetch = now;
    return cachedBtcDominance;
  } catch {
    return cachedBtcDominance;
  }
}

export interface HistoricalOHLCV {
  coinId: string;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export async function fetchHistoricalOHLCV(symbol: string, cmcSymbol?: string): Promise<HistoricalOHLCV[]> {
  if (!CMC_API_KEY) return [];

  try {
    const lookupSymbol = cmcSymbol || symbol;
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - 7 * 24 * 60 * 60 * 1000);
    const url = `https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical?symbol=${lookupSymbol}&time_start=${startDate.toISOString().split("T")[0]}&time_end=${endDate.toISOString().split("T")[0]}&interval=daily&convert=USD`;

    const response = await fetch(url, {
      headers: { "X-CMC_PRO_API_KEY": CMC_API_KEY },
    });

    if (!response.ok) return [];
    const result = await response.json();
    if (!result.data?.quotes) return [];

    const coinConfig = MONITORED_COINS.find((c) => (c.cmcSymbol || c.symbol) === lookupSymbol);
    const coinId = coinConfig?.id || symbol.toLowerCase();

    interface CMCQuoteEntry {
      time_open: string;
      quote: { USD: { open: number; high: number; low: number; close: number; volume: number } };
    }

    return result.data.quotes.map((q: CMCQuoteEntry) => {
      const usd = q.quote?.USD;
      return {
        coinId,
        timestamp: new Date(q.time_open),
        open: usd?.open ?? 0,
        high: usd?.high ?? 0,
        low: usd?.low ?? 0,
        close: usd?.close ?? 0,
        volume: usd?.volume ?? 0,
      };
    });
  } catch {
    return [];
  }
}

export async function fetchCoinPrices(force = false): Promise<CoinPrice[]> {
  const now = Date.now();
  if (!force && now - lastFetchTime < CACHE_TTL && cachedPrices.length > 0) {
    return cachedPrices;
  }

  try {
    const prices = COINGECKO_FALLBACK
      ? await fetchFromCoinGecko()
      : await fetchFromCoinMarketCap();

    if (prices.length > 0) {
      cachedPrices = prices;
      lastFetchTime = now;
      lastSuccessfulFetch = now;
      return cachedPrices;
    }
  } catch (err) {
    console.warn(`Primary data source failed: ${err instanceof Error ? err.message : err}`);

    if (!COINGECKO_FALLBACK) {
      try {
        const fallbackPrices = await fetchFromCoinGecko();
        if (fallbackPrices.length > 0) {
          cachedPrices = fallbackPrices;
          lastFetchTime = now;
          lastSuccessfulFetch = now;
          return cachedPrices;
        }
      } catch (fallbackErr) {
        console.warn(`CoinGecko fallback also failed: ${fallbackErr instanceof Error ? fallbackErr.message : fallbackErr}`);
      }
    }
  }

  lastFetchTime = now;
  if (cachedPrices.length > 0) return cachedPrices;
  return [];
}
