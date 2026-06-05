/**
 * Dashboard ↔ trainer bar-source parity (task #345, blocked on P1-D).
 *
 * Today the customer-facing dashboard chart and the model trainer pull
 * bars from DIFFERENT stores:
 *
 *   - Dashboard `/crypto/price-history/:coinId` reads raw ticks from
 *     `price_history` (artifacts/api-server/src/routes/crypto/index.ts:511)
 *     and projects them as a flat (timestamp, price) stream.
 *   - The trainer's per-timeframe loader reads OHLCV bars from
 *     `price_candles` (artifacts/ml-engine/app/db.py:fetch_real_candles),
 *     which is populated by backfill_history.py from OKX/CMC.
 *
 * Those two stores aren't kept in lock-step, so the bars the user sees
 * on the dashboard for a (coin, timeframe) window are NOT necessarily
 * the same bars the model trained on. That's a real correctness hazard
 * — operator dashboards and model retraining should always agree on
 * "what happened on this coin in this window" — and is the work P1-D
 * is scoped to fix (unify both consumers behind `price_candles` /
 * `fetch_real_candles`, with a thin tick-aggregator for sub-1m views).
 *
 * Until P1-D ships there is no defensible way to make this test pass —
 * the two pipelines genuinely return different things. We land it
 * SKIPPED here so:
 *
 *   1. Anyone grepping for "dashboard parity" finds the documented
 *      gap and the path back to a fix.
 *   2. The moment P1-D unifies the source the skip can be flipped to
 *      `it(...)` and the contract becomes machine-checked.
 *
 * If you remove or relax the skip without shipping P1-D, the test will
 * fail loudly — that is the intended trip-wire.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

describe("dashboard ↔ trainer bar-source parity (skipped — blocked on P1-D)", () => {
  it.skip(
    "dashboard `/crypto/price-history` and trainer `fetch_real_candles` " +
      "return the same bars for the same (coin, timeframe) window — " +
      "implement after P1-D unifies both consumers behind price_candles",
    () => {
      // Intentionally left empty. See file header for the gap and the
      // P1-D follow-up. When P1-D lands:
      //   1. Replace `it.skip(...)` with `it(...)`.
      //   2. Hit /crypto/price-history/:coinId?hours=24 and the trainer's
      //      fetch_real_candles equivalent endpoint for the same window.
      //   3. Assert the (bucket_start, close) sequence matches within
      //      a small epsilon.
      assert.fail("placeholder — body added once P1-D unifies the source");
    },
  );
});
