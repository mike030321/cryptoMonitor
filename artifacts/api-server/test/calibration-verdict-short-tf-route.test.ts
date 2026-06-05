import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import express from "express";
import type { AddressInfo } from "node:net";

import router from "../src/routes/crypto/index";

// Task #599 — schema contract for GET /crypto/calibration-verdict/short-tf.
//
// The dashboard relies on a stable response shape regardless of which
// branch the handler takes (no ml-engine dir → "unknown"; status file
// missing → "unknown" with statusFileExists:false; status file present
// but no calibration_verdict block → still all keys present). This test
// pins the keyset by spinning up a tiny express app, mounting the real
// production router, and asserting every documented key is present in
// the response payload — even when null. It complements the python
// loop tests (which already cover the writer side) by catching server
// schema drift before the dashboard breaks.

interface RunningServer {
  url: string;
  close: () => Promise<void>;
}

function listen(app: express.Express): Promise<RunningServer> {
  return new Promise((resolve) => {
    const server = http.createServer(app);
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as AddressInfo;
      resolve({
        url: `http://127.0.0.1:${addr.port}`,
        close: () =>
          new Promise<void>((r) => server.close(() => r())),
      });
    });
  });
}

function get(url: string): Promise<{ status: number; body: unknown }> {
  return new Promise((resolve, reject) => {
    http
      .get(url, (resp) => {
        const chunks: Buffer[] = [];
        resp.on("data", (c) => chunks.push(c));
        resp.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          let parsed: unknown = null;
          try {
            parsed = JSON.parse(text);
          } catch {
            parsed = { _raw: text };
          }
          resolve({ status: resp.statusCode ?? 0, body: parsed });
        });
      })
      .on("error", reject);
  });
}

let server: RunningServer | null = null;

before(async () => {
  const app = express();
  app.use("/api", router);
  server = await listen(app);
});

after(async () => {
  if (server) await server.close();
});

const TOP_LEVEL_KEYS = [
  "state",
  "statusFileExists",
  "statusReadError",
  "shortTf",
  "markdownPath",
  "jsonPath",
  "markdownTail",
  "markdownReadError",
  "fetchedAt",
] as const;

test("response shape: every documented top-level key is present", async () => {
  assert.ok(server, "server should be running");
  const { status, body } = await get(
    `${server!.url}/api/crypto/calibration-verdict/short-tf`,
  );
  assert.equal(status, 200);
  assert.ok(body && typeof body === "object", "body should be an object");
  const obj = body as Record<string, unknown>;
  for (const k of TOP_LEVEL_KEYS) {
    assert.ok(
      Object.prototype.hasOwnProperty.call(obj, k),
      `response is missing top-level key: ${k}`,
    );
  }
  // state pill is one of the four documented values.
  assert.ok(
    ["ok", "error", "timeout", "unknown"].includes(obj["state"] as string),
    `state pill is unexpected: ${String(obj["state"])}`,
  );
  // fetchedAt must parse as a valid ISO timestamp.
  assert.ok(
    typeof obj["fetchedAt"] === "string" &&
      !Number.isNaN(Date.parse(obj["fetchedAt"] as string)),
    "fetchedAt should be a parseable ISO timestamp",
  );
  // shortTf is either null or a structured block; if present, pin its
  // documented inner keys so the dashboard hook never breaks silently.
  const shortTf = obj["shortTf"];
  if (shortTf !== null) {
    assert.ok(
      shortTf && typeof shortTf === "object",
      "shortTf should be null or an object",
    );
    const s = shortTf as Record<string, unknown>;
    for (const k of [
      "lastStatus",
      "lastAttemptAt",
      "lastSuccessAt",
      "lastError",
      "lastElapsedSeconds",
      "triggerTimeframes",
      "timeoutSeconds",
      "command",
      "lastMdPath",
      "lastJsonPath",
      "summary",
    ]) {
      assert.ok(
        Object.prototype.hasOwnProperty.call(s, k),
        `shortTf is missing key: ${k}`,
      );
    }
  }
});
