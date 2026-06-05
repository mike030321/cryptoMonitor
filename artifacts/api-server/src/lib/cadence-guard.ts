// Single source of truth for cadence-correctness assertions on the
// price store writers (task #343).
//
// `price_history` is the per-tick close-only table consumed by the live
// poller, the resampler fallback, and `pattern-analyzer`. It carries no
// `timeframe` column, so writing a coarser-cadence row (e.g. an hourly
// synthetic bar or an OKX 5m candle) silently mixes cadences inside the
// same `(coin_id, timestamp)` series. The trainer's resampler walks the
// stream and assigns rows to buckets by `floor(ts/bucket_ms)*bucket_ms`,
// so a daily contaminant becomes a 5m bucket close. This is the exact
// failure mode `reports/20260423T000000Z-schema-audit.md` documents.
//
// `price_candles` is the per-timeframe OHLCV table that owns aggregated
// bars. Native cadences land there, keyed by
// `(coin_id, timeframe, bucket_start)` so cadences cannot collide.
//
// Every writer to either table MUST call `assertNativeCadence` first.
// The guard throws an `Error` (no silent fallback, no swallowed warning)
// so the call site fails loudly and visibly in the workflow logs.

const PRICE_HISTORY_NATIVE_TIMEFRAME = "1m" as const;

const VALID_PRICE_CANDLES_TIMEFRAMES = new Set([
  "1m", "5m", "1h", "2h", "6h", "1d",
]);

export type CadenceTable = "price_history" | "price_candles";

export class CadenceGuardError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CadenceGuardError";
  }
}

/**
 * Reject any non-1m write to `price_history` and any unknown timeframe
 * write to `price_candles`. Returns void on success; throws
 * `CadenceGuardError` on rejection.
 */
export function assertNativeCadence(
  timeframe: string,
  source: string,
  table: CadenceTable,
): void {
  if (table === "price_history") {
    if (timeframe !== PRICE_HISTORY_NATIVE_TIMEFRAME) {
      throw new CadenceGuardError(
        `[cadence-guard] refusing ${timeframe} write to price_history from ` +
        `source='${source}': price_history is the 1m-tick store. Aggregated ` +
        `bars must go to price_candles (see schema-audit.md / task #343).`,
      );
    }
    return;
  }
  if (table === "price_candles") {
    if (!VALID_PRICE_CANDLES_TIMEFRAMES.has(timeframe)) {
      throw new CadenceGuardError(
        `[cadence-guard] refusing ${timeframe} write to price_candles from ` +
        `source='${source}': not a recognised native cadence.`,
      );
    }
    return;
  }
  throw new CadenceGuardError(
    `[cadence-guard] unknown table '${table}' (source='${source}').`,
  );
}
