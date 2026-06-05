/**
 * Task #435 — assert the new /crypto/brain/5m-topup-recent-winners
 * proxy forwards the ml-engine response verbatim, threads through the
 * optional `?limit` query, and surfaces upstream failures with a 502
 * (not a generic 500). Mirrors the in-process pattern from
 * `five-m-topup-status-proxy.test.ts` so a future refactor that drops
 * the route fails loud.
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

test("Task #435: /crypto/brain/5m-topup-recent-winners forwards the ml-engine list", async () => {
  const fakeBody = {
    winners: [
      { replica: "host-a/pid=1", tick_at: "2026-04-24T00:00:00+00:00" },
      { replica: "host-b/pid=2", tick_at: "2026-04-23T00:00:00+00:00" },
    ],
    limit: 14,
    max: 50,
  };

  let seenUrl: string | undefined;
  const ml = await startMockMlEngine((req, res) => {
    if (
      req.method === "GET" &&
      req.url &&
      req.url.startsWith("/ml/admin/5m-topup/recent-winners")
    ) {
      seenUrl = req.url;
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(fakeBody));
      return;
    }
    res.writeHead(404).end();
  });
  const previousUrl = process.env.ML_ENGINE_URL;
  process.env.ML_ENGINE_URL = ml.url;
  const api = await startApiServer();
  try {
    const r = await fetch(
      `${api.url}/api/crypto/brain/5m-topup-recent-winners?limit=14`,
    );
    assert.equal(r.status, 200, "proxy must return 200 when ml-engine returns 200");
    const body = await r.json();
    assert.deepEqual(body, fakeBody, "proxy must forward the ml-engine body verbatim");
    assert.ok(
      seenUrl && seenUrl.includes("limit=14"),
      `ml-engine must see the forwarded limit query, got ${seenUrl}`,
    );
  } finally {
    if (previousUrl === undefined) delete process.env.ML_ENGINE_URL;
    else process.env.ML_ENGINE_URL = previousUrl;
    await api.close();
    await ml.close();
  }
});

test("Task #435: /crypto/brain/5m-topup-recent-winners returns 502 when the ml-engine is unreachable", async () => {
  const previousUrl = process.env.ML_ENGINE_URL;
  // Closed port — connection refusal must surface as 502 (bad gateway).
  process.env.ML_ENGINE_URL = "http://127.0.0.1:1";
  const api = await startApiServer();
  try {
    const r = await fetch(`${api.url}/api/crypto/brain/5m-topup-recent-winners`);
    assert.equal(r.status, 502, "must surface upstream failure as 502 bad gateway");
    const body = (await r.json()) as { error?: string };
    assert.ok(
      typeof body.error === "string" && body.error.length > 0,
      "must include an error message",
    );
  } finally {
    if (previousUrl === undefined) delete process.env.ML_ENGINE_URL;
    else process.env.ML_ENGINE_URL = previousUrl;
    await api.close();
  }
});
