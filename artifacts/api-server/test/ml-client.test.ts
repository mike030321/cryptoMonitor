/**
 * Integration test for the ML engine wiring (Phase 1).
 *
 * Spins up an in-process HTTP server that mimics the Python ml-engine's
 * stub /predict response, points the Node client at it, and asserts that
 * the response shape matches the contract documented in ml-client.ts.
 *
 * We do not start the real Python service here — that is exercised by the
 * pytest suite under artifacts/ml-engine/tests/.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import { getMlPrediction, mlHealth } from "../src/lib/ml-client";

function startMockMlEngine(): Promise<{ url: string; close: () => Promise<void> }> {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      if (req.method === "GET" && req.url === "/ml/health") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ status: "ok", service: "ml-engine" }));
        return;
      }
      if (req.method === "POST" && req.url === "/ml/predict") {
        let body = "";
        req.on("data", (c) => (body += c));
        req.on("end", () => {
          const parsed = JSON.parse(body) as { coinId: string; timeframe: string };
          res.writeHead(200, { "content-type": "application/json" });
          res.end(
            JSON.stringify({
              coinId: parsed.coinId,
              timeframe: parsed.timeframe,
              probUp: 0.5,
              probDown: 0.5,
              expectedReturnPct: 0,
              source: "stub",
            }),
          );
        });
        return;
      }
      res.writeHead(404).end();
    });
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as AddressInfo).port;
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise<void>((r) => server.close(() => r())),
      });
    });
  });
}

test("ml-client: health endpoint returns ok", async () => {
  const mock = await startMockMlEngine();
  process.env.ML_ENGINE_URL = mock.url;
  try {
    const h = await mlHealth();
    assert.equal(h.status, "ok");
    assert.equal(h.service, "ml-engine");
  } finally {
    await mock.close();
  }
});

test("ml-client: predict stub returns the documented shape", async () => {
  const mock = await startMockMlEngine();
  process.env.ML_ENGINE_URL = mock.url;
  try {
    const p = await getMlPrediction("bitcoin", "1h");
    assert.equal(p.coinId, "bitcoin");
    assert.equal(p.timeframe, "1h");
    assert.equal(p.probUp, 0.5);
    assert.equal(p.probDown, 0.5);
    assert.equal(p.expectedReturnPct, 0);
    assert.equal(p.source, "stub");
  } finally {
    await mock.close();
  }
});
