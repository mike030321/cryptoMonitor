/**
 * Task #412 — assert the new /crypto/brain/5m-topup/status proxy forwards
 * the ml-engine response verbatim and surfaces upstream failures with a
 * 502 instead of a generic 500.
 *
 * Strategy: spin up an in-process mock ml-engine on a random port, point
 * `process.env.ML_ENGINE_URL` at it, mount the api-server express app
 * (which uses `fetch` against `ML_ENGINE_URL` at request time), and drive
 * a request through `app(req, res)`-style invocation via supertest-free
 * lightweight HTTP. Mirrors the in-process pattern from `ml-client.test.ts`.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import app from "../src/app";

function startMockMlEngine(
  handler: (req: http.IncomingMessage, res: http.ServerResponse) => void,
): Promise<{ url: string; close: () => Promise<void> }> {
  return new Promise((resolve) => {
    const server = http.createServer(handler);
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as AddressInfo).port;
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise<void>((r) => server.close(() => r())),
      });
    });
  });
}

function startApiServer(): Promise<{ url: string; close: () => Promise<void> }> {
  return new Promise((resolve) => {
    const server = http.createServer(app);
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as AddressInfo).port;
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise<void>((r) => server.close(() => r())),
      });
    });
  });
}

test("Task #412: /crypto/brain/5m-topup/status forwards the ml-engine state shape", async () => {
  const fakeState = {
    enabled: true,
    interval_seconds: 86400,
    window_days: 7,
    alert_below_days: 310,
    last_check_at: 1_700_000_000,
    last_attempt_outcome: "ok",
    last_finished_at: 1_700_000_010,
    last_error: null,
    last_topup_inserted: 42,
    last_topup_per_coin: { bitcoin: 12, ethereum: 30 },
    last_health_per_coin: { bitcoin: 311.5, ethereum: 309.0 },
    last_alerts: ["ethereum"],
    ticks_total: 5,
    runs_total: 4,
    rows_inserted_total: 200,
    alerts_emitted_total: 1,
  };

  const ml = await startMockMlEngine((req, res) => {
    if (req.method === "GET" && req.url === "/ml/admin/5m-topup/status") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(fakeState));
      return;
    }
    res.writeHead(404).end();
  });
  const previousUrl = process.env.ML_ENGINE_URL;
  process.env.ML_ENGINE_URL = ml.url;
  const api = await startApiServer();
  try {
    const r = await fetch(`${api.url}/api/crypto/brain/5m-topup/status`);
    assert.equal(r.status, 200, "proxy must return 200 when ml-engine returns 200");
    const body = await r.json();
    assert.deepEqual(body, fakeState, "proxy must forward the ml-engine body verbatim");
  } finally {
    if (previousUrl === undefined) delete process.env.ML_ENGINE_URL;
    else process.env.ML_ENGINE_URL = previousUrl;
    await api.close();
    await ml.close();
  }
});

test("Task #412: /crypto/brain/5m-topup/status returns 502 when the ml-engine is unreachable", async () => {
  const previousUrl = process.env.ML_ENGINE_URL;
  // Point at a closed port — connection refusal must surface as 502
  // (bad gateway), not 500. Operators shouldn't have to grep logs to
  // tell "ml-engine is down" from "the proxy crashed".
  process.env.ML_ENGINE_URL = "http://127.0.0.1:1";
  const api = await startApiServer();
  try {
    const r = await fetch(`${api.url}/api/crypto/brain/5m-topup/status`);
    assert.equal(r.status, 502, "must surface upstream failure as 502 bad gateway");
    const body = (await r.json()) as { error?: string };
    assert.ok(typeof body.error === "string" && body.error.length > 0, "must include an error message");
  } finally {
    if (previousUrl === undefined) delete process.env.ML_ENGINE_URL;
    else process.env.ML_ENGINE_URL = previousUrl;
    await api.close();
  }
});
