import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";

import { __testing } from "../src/lib/market-signals-poller";

// Task #286 — verify the multi-source liquidations aggregation logic.
// We monkey-patch global `fetch` so the test never makes a real network
// call. Each scenario asserts that:
//   - both sources are summed when both succeed
//   - the row is still produced (with breakdown=okx-only) when the
//     secondary source (Gate.io) fails — i.e. graceful fallback
//   - the source label correctly reflects which feeds contributed
//   - non-top-perp coins (no Gate.io contract mapped) only call OKX.

const { fetchAggregatedLiquidations, sourceLabelFor } = __testing;

type FetchHandler = (url: string) => unknown | Promise<unknown>;

const realFetch = globalThis.fetch;
let handler: FetchHandler = () => {
  throw new Error("no handler installed");
};

before(() => {
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const body = await handler(url);
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
});

after(() => {
  globalThis.fetch = realFetch;
});

beforeEach(() => {
  handler = () => {
    throw new Error("unexpected fetch in test");
  };
  delete process.env.COINGLASS_API_KEY;
});

after(() => {
  delete process.env.COINGLASS_API_KEY;
});

function nowMs(): number {
  return Date.now();
}

function okxResp<T>(rows: T[]): { code: string; data: T[] } {
  return { code: "0", data: rows };
}

describe("multi-source liquidations aggregation", () => {
  it("sums OKX + Gate.io when both succeed and labels the source accordingly", async () => {
    const ts = nowMs() - 60_000; // within trailing 1h
    handler = (url) => {
      if (url.includes("/api/v5/public/instruments")) {
        // OKX ctVal — 0.01 BTC per contract.
        return okxResp([{ ctVal: "0.01" }]);
      }
      if (url.includes("/api/v5/public/liquidation-orders")) {
        if (url.includes("after=")) return okxResp([]);
        return okxResp([
          { details: [{ bkPx: "60000", sz: "5", ts: String(ts) }] },
        ]);
      }
      if (url.includes("/api/v4/futures/usdt/contracts/")) {
        // Gate quanto multiplier — 0.0001 BTC per contract.
        return { name: "BTC_USDT", quanto_multiplier: "0.0001" };
      }
      if (url.includes("/api/v4/futures/usdt/liq_orders")) {
        return [
          { time_ms: ts, fill_price: "60000", size: 100 },
          { time_ms: ts, fill_price: "60000", size: -50 },
        ];
      }
      throw new Error(`unexpected url ${url}`);
    };

    const result = await fetchAggregatedLiquidations("btc", "BTC", "BTC-USDT-SWAP");
    // OKX: 60000 * 5 * 0.01 = 3000. Gate: (100+50) * 0.0001 * 60000 = 900.
    assert.equal(result.totalUsd, 3900);
    assert.deepEqual(result.breakdown, { okx: 3000, gate: 900 });
    assert.equal(sourceLabelFor(result.breakdown), "okx_swap+gate_swap");
  });

  it("falls back to OKX-only when the Gate.io source fails", async () => {
    const ts = nowMs() - 30_000;
    handler = (url) => {
      if (url.includes("/api/v5/public/instruments")) {
        return okxResp([{ ctVal: "0.01" }]);
      }
      if (url.includes("/api/v5/public/liquidation-orders")) {
        if (url.includes("after=")) return okxResp([]);
        return okxResp([
          { details: [{ bkPx: "60000", sz: "2", ts: String(ts) }] },
        ]);
      }
      if (url.includes("api.gateio.ws")) {
        // Simulate aggregator outage by throwing inside the handler;
        // the production code must swallow this and still yield a row.
        throw new Error("gate.io 503");
      }
      throw new Error(`unexpected url ${url}`);
    };

    const result = await fetchAggregatedLiquidations("btc", "BTC", "BTC-USDT-SWAP");
    // OKX: 60000 * 2 * 0.01 = 1200. Gate fails → omitted from breakdown.
    assert.equal(result.totalUsd, 1200);
    assert.deepEqual(result.breakdown, { okx: 1200 });
    assert.equal(sourceLabelFor(result.breakdown), "okx_swap");
  });

  it("returns null totals when both sources fail (caller skips the row)", async () => {
    handler = (url) => {
      if (url.includes("/api/v5/public/instruments")) {
        return okxResp([{ ctVal: "0.01" }]);
      }
      if (url.includes("/api/v5/public/liquidation-orders")) {
        throw new Error("okx 502");
      }
      if (url.includes("api.gateio.ws")) {
        throw new Error("gate 502");
      }
      throw new Error(`unexpected url ${url}`);
    };

    const result = await fetchAggregatedLiquidations("btc", "BTC", "BTC-USDT-SWAP");
    assert.equal(result.totalUsd, null);
    assert.equal(result.breakdown, null);
    assert.equal(sourceLabelFor(result.breakdown), "no_liq_source");
  });

  it("includes Coinglass as a third source when COINGLASS_API_KEY is set", async () => {
    process.env.COINGLASS_API_KEY = "test-key";
    const ts = nowMs() - 60_000;
    let coinglassAuthHeader: string | null = null;
    handler = (url) => {
      if (url.includes("/api/v5/public/instruments")) {
        return okxResp([{ ctVal: "0.01" }]);
      }
      if (url.includes("/api/v5/public/liquidation-orders")) {
        if (url.includes("after=")) return okxResp([]);
        return okxResp([
          { details: [{ bkPx: "60000", sz: "5", ts: String(ts) }] },
        ]);
      }
      if (url.includes("/api/v4/futures/usdt/contracts/")) {
        return { name: "BTC_USDT", quanto_multiplier: "0.0001" };
      }
      if (url.includes("/api/v4/futures/usdt/liq_orders")) {
        return [{ time_ms: ts, fill_price: "60000", size: 100 }];
      }
      if (url.includes("open-api-v4.coinglass.com")) {
        return {
          code: "0",
          data: [
            {
              time: ts,
              aggregatedLongLiquidationUsd: "1500",
              aggregatedShortLiquidationUsd: "2500",
            },
          ],
        };
      }
      throw new Error(`unexpected url ${url}`);
    };

    // Capture the CG-API-KEY header by wrapping fetch — patched fetch in
    // the suite ignores headers, so we re-patch just for this test.
    const wrapped = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("open-api-v4.coinglass.com") && init?.headers) {
        const h = init.headers as Record<string, string>;
        coinglassAuthHeader = h["CG-API-KEY"] ?? null;
      }
      return wrapped(input as RequestInfo);
    }) as typeof fetch;

    try {
      const result = await fetchAggregatedLiquidations(
        "btc",
        "BTC",
        "BTC-USDT-SWAP",
      );
      // OKX 3000, Gate 600, Coinglass 4000 → 7600.
      assert.equal(result.totalUsd, 7600);
      assert.deepEqual(result.breakdown, {
        okx: 3000,
        gate: 600,
        coinglass: 4000,
      });
      assert.equal(
        sourceLabelFor(result.breakdown),
        "okx_swap+gate_swap+coinglass",
      );
      assert.equal(coinglassAuthHeader, "test-key");
    } finally {
      globalThis.fetch = wrapped;
    }
  });

  it("falls back to OKX+Gate when Coinglass fails (key set, outage)", async () => {
    process.env.COINGLASS_API_KEY = "test-key";
    const ts = nowMs() - 30_000;
    handler = (url) => {
      if (url.includes("/api/v5/public/instruments")) {
        return okxResp([{ ctVal: "0.01" }]);
      }
      if (url.includes("/api/v5/public/liquidation-orders")) {
        if (url.includes("after=")) return okxResp([]);
        return okxResp([
          { details: [{ bkPx: "60000", sz: "2", ts: String(ts) }] },
        ]);
      }
      if (url.includes("/api/v4/futures/usdt/contracts/")) {
        return { name: "BTC_USDT", quanto_multiplier: "0.0001" };
      }
      if (url.includes("/api/v4/futures/usdt/liq_orders")) {
        return [{ time_ms: ts, fill_price: "60000", size: 50 }];
      }
      if (url.includes("open-api-v4.coinglass.com")) {
        throw new Error("coinglass 503");
      }
      throw new Error(`unexpected url ${url}`);
    };

    const result = await fetchAggregatedLiquidations(
      "btc",
      "BTC",
      "BTC-USDT-SWAP",
    );
    // OKX 1200 + Gate 300 = 1500. Coinglass omitted on failure.
    assert.equal(result.totalUsd, 1500);
    assert.deepEqual(result.breakdown, { okx: 1200, gate: 300 });
    assert.equal(sourceLabelFor(result.breakdown), "okx_swap+gate_swap");
  });

  it("does not call Coinglass when no API key is configured", async () => {
    let coinglassCalled = false;
    const ts = nowMs() - 60_000;
    handler = (url) => {
      if (url.includes("open-api-v4.coinglass.com")) {
        coinglassCalled = true;
        throw new Error("should not be called");
      }
      if (url.includes("/api/v5/public/instruments")) {
        return okxResp([{ ctVal: "0.01" }]);
      }
      if (url.includes("/api/v5/public/liquidation-orders")) {
        if (url.includes("after=")) return okxResp([]);
        return okxResp([
          { details: [{ bkPx: "60000", sz: "1", ts: String(ts) }] },
        ]);
      }
      if (url.includes("/api/v4/futures/usdt/contracts/")) {
        return { name: "BTC_USDT", quanto_multiplier: "0.0001" };
      }
      if (url.includes("/api/v4/futures/usdt/liq_orders")) {
        return [];
      }
      throw new Error(`unexpected url ${url}`);
    };

    const result = await fetchAggregatedLiquidations(
      "btc",
      "BTC",
      "BTC-USDT-SWAP",
    );
    assert.equal(coinglassCalled, false);
    assert.equal(result.breakdown?.coinglass, undefined);
    assert.equal(sourceLabelFor(result.breakdown), "okx_swap+gate_swap");
  });

  it("only queries OKX for coins without a mapped Gate.io contract", async () => {
    const ts = nowMs() - 10_000;
    let gateCalled = false;
    handler = (url) => {
      if (url.includes("api.gateio.ws")) {
        gateCalled = true;
        throw new Error("should not be called");
      }
      if (url.includes("/api/v5/public/instruments")) {
        return okxResp([{ ctVal: "1" }]);
      }
      if (url.includes("/api/v5/public/liquidation-orders")) {
        if (url.includes("after=")) return okxResp([]);
        return okxResp([
          { details: [{ bkPx: "0.5", sz: "1000", ts: String(ts) }] },
        ]);
      }
      throw new Error(`unexpected url ${url}`);
    };

    const result = await fetchAggregatedLiquidations(
      "some-untracked-coin",
      "FOO",
      "FOO-USDT-SWAP",
    );
    assert.equal(gateCalled, false);
    assert.equal(result.totalUsd, 500);
    assert.deepEqual(result.breakdown, { okx: 500 });
  });
});
