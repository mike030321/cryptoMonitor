export * from "./generated/api";
export type * from "./generated/types";

// Both `./generated/api` (zod schemas) and `./generated/types` (TS types
// inferred from the OpenAPI spec) export `GetCoinDetailParams` and
// `GetPriceHistoryParams`, which makes the wildcard re-exports ambiguous and
// fails `tsc --build` with TS2308. Explicitly re-export the zod schemas from
// `./generated/api` so callers keep using them as runtime values; the matching
// TS shape is reachable via `z.infer<typeof GetCoinDetailParams>` /
// `z.infer<typeof GetPriceHistoryParams>` when needed.
export { GetCoinDetailParams, GetPriceHistoryParams } from "./generated/api";

// Hand-authored zod contracts that live alongside the generated ones.
// See ./timeframe-roles.ts for the rationale.
export * from "./timeframe-roles";
