import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import express from "express";
import { AddressInfo } from "node:net";

import { createBrainStatePostHandler } from "../src/lib/brain-state-route";
import type { PromotionGateVerdict } from "../src/lib/brain-promotion-gate";

// Task #406 — end-to-end integration test for POST /crypto/brain/state.
//
// Spins up a tiny Express app that mounts the production handler factory
// with controlled doubles for hasPromotedSlice / setBrainState. Exercises
// the full HTTP request → response cycle so the 409/200/400 wiring is
// validated against real http parsing, not a static source-shape match.
// This complements the unit tests in brain-promotion-gate.test.ts.

interface SetBrainCall {
  enabled: boolean;
  source: "manual";
}

function buildApp(opts: {
  verdict: PromotionGateVerdict;
  setBrainCalls: SetBrainCall[];
}) {
  const app = express();
  app.use(express.json());
  const handler = createBrainStatePostHandler({
    hasPromotedSlice: async () => opts.verdict,
    setBrainState: async (enabled, source) => {
      opts.setBrainCalls.push({ enabled, source });
      return {
        enabled,
        source,
        lastChangedAt: "2026-04-24T11:54:02.549Z",
      };
    },
    getBrainRevertLog: async () => [],
    getAutoRevertCounter: () => 0,
  });
  app.post("/api/crypto/brain/state", async (req, res) => {
    await handler(req, res);
  });
  return app;
}

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
        close: () => new Promise<void>((r) => server.close(() => r())),
      });
    });
  });
}

let setBrainCalls: SetBrainCall[];
let blockedServer: RunningServer;
let approvedServer: RunningServer;

before(async () => {
  setBrainCalls = [];
  // Server A: gate refuses (matches the live state captured in
  // .local/remediation/08-task-406/brain-state-enable-blocked-by-gate.json).
  blockedServer = await listen(
    buildApp({
      verdict: {
        ok: false,
        reason: "no_promoted_slices",
        evidence: {
          verification_status: "ok",
          passed: false,
          slices_promoted: 0,
          coins_with_promotion: [],
          recorded_at: 1777030969.6022975,
        },
      },
      setBrainCalls,
    }),
  );
  // Server B: gate approves (one promoted slice on bitcoin).
  approvedServer = await listen(
    buildApp({
      verdict: {
        ok: true,
        evidence: {
          verification_status: "ok",
          passed: true,
          slices_promoted: 1,
          coins_with_promotion: ["bitcoin"],
          recorded_at: 1777030999,
        },
      },
      setBrainCalls,
    }),
  );
});

after(async () => {
  await blockedServer.close();
  await approvedServer.close();
});

test("POST /api/crypto/brain/state {enabled:true} returns 409 with full gate verdict when no slice is promoted", async () => {
  const before = setBrainCalls.length;
  const resp = await fetch(`${blockedServer.url}/api/crypto/brain/state`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ enabled: true }),
  });
  assert.equal(resp.status, 409);
  const body = (await resp.json()) as {
    error: string;
    message: string;
    gate: PromotionGateVerdict;
  };
  assert.equal(body.error, "promotion_gate_blocked");
  assert.match(body.message, /promoted:true/);
  assert.equal(body.gate.ok, false);
  assert.equal(body.gate.reason, "no_promoted_slices");
  assert.equal(body.gate.evidence?.slices_promoted, 0);
  assert.deepEqual(body.gate.evidence?.coins_with_promotion, []);
  // setBrainState must not be invoked when the gate refuses.
  assert.equal(
    setBrainCalls.length,
    before,
    "setBrainState must not be called when the gate refuses",
  );
});

test("POST /api/crypto/brain/state {enabled:false} succeeds even when the gate would refuse (kill-switch never gated)", async () => {
  const before = setBrainCalls.length;
  const resp = await fetch(`${blockedServer.url}/api/crypto/brain/state`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ enabled: false }),
  });
  assert.equal(resp.status, 200);
  const body = (await resp.json()) as {
    enabled: boolean;
    source: string;
    autoRevert: { consecutiveDriftCycles: number; recentEvents: unknown[] };
  };
  assert.equal(body.enabled, false);
  assert.equal(body.source, "manual");
  assert.equal(body.autoRevert.consecutiveDriftCycles, 0);
  assert.deepEqual(body.autoRevert.recentEvents, []);
  // Disable must invoke setBrainState exactly once with (false, "manual").
  assert.equal(setBrainCalls.length, before + 1);
  const lastCall = setBrainCalls[setBrainCalls.length - 1];
  assert.equal(lastCall.enabled, false);
  assert.equal(lastCall.source, "manual");
});

test("POST /api/crypto/brain/state {enabled:true} returns 200 when the gate approves and persists (true,'manual')", async () => {
  const before = setBrainCalls.length;
  const resp = await fetch(`${approvedServer.url}/api/crypto/brain/state`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ enabled: true }),
  });
  assert.equal(resp.status, 200);
  const body = (await resp.json()) as { enabled: boolean; source: string };
  assert.equal(body.enabled, true);
  assert.equal(body.source, "manual");
  assert.equal(setBrainCalls.length, before + 1);
  const lastCall = setBrainCalls[setBrainCalls.length - 1];
  assert.equal(lastCall.enabled, true);
  assert.equal(lastCall.source, "manual");
});

test("POST /api/crypto/brain/state with non-boolean enabled returns 400 and never calls setBrainState", async () => {
  const before = setBrainCalls.length;
  const resp = await fetch(`${approvedServer.url}/api/crypto/brain/state`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ enabled: "yes" }),
  });
  assert.equal(resp.status, 400);
  const body = (await resp.json()) as { error: string };
  assert.match(body.error, /enabled: boolean/);
  assert.equal(setBrainCalls.length, before, "400 must not invoke setBrainState");
});

test("POST /api/crypto/brain/state with empty body returns 400 (default enabled=undefined fails the boolean check)", async () => {
  const before = setBrainCalls.length;
  const resp = await fetch(`${approvedServer.url}/api/crypto/brain/state`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  assert.equal(resp.status, 400);
  assert.equal(setBrainCalls.length, before);
});
