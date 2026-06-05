// Task #365 — route-level contract test (proof C support).
//
// Asserts that the live API responses from the two endpoints the
// agent-detail page consumes — `/api/crypto/paper-portfolios` and
// `/api/crypto/agents/:id` — actually emit a numeric `startingCapital`
// for every agent / portfolio. Without this, the new
// `totalValue − startingCapital` math on the frontend silently falls
// back to the legacy `totalPnl` snapshot.
//
// HERMETIC: this suite imports the Express app directly from
// `src/app.ts` and binds it to an ephemeral port for the duration of
// the test run. No external dev workflow is required — `pnpm test`
// works in CI without a side-channel server. The ephemeral listener
// is torn down in `after`.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import type { Server } from "node:http";
import app from "../src/app.ts";

let server: Server;
let BASE = "";

before(async () => {
  await new Promise<void>((resolve, reject) => {
    server = app.listen(0, (err?: Error) => (err ? reject(err) : resolve()));
  });
  const addr = server.address() as AddressInfo;
  BASE = `http://127.0.0.1:${addr.port}/api`;
  // Wait for the post-listen async restore (db migrations etc.) to
  // settle by polling /healthz.
  const deadline = Date.now() + 30_000;
  let lastErr: unknown = null;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${BASE}/healthz`, { signal: AbortSignal.timeout(1500) });
      if (r.ok) return;
      lastErr = new Error(`/healthz returned ${r.status}`);
    } catch (err) {
      lastErr = err;
    }
    await new Promise(res => setTimeout(res, 250));
  }
  throw new Error(`in-process api-server not ready after 30s: ${lastErr}`);
});

after(async () => {
  if (server) await new Promise<void>(r => server.close(() => r()));
});

test("route contract: /crypto/paper-portfolios emits numeric startingCapital", async () => {
  const r = await fetch(`${BASE}/crypto/paper-portfolios`);
  assert.equal(r.status, 200);
  const body = await r.json();
  assert.ok(Array.isArray(body), "expected array of paper portfolios");
  assert.ok(body.length > 0, "expected at least one paper portfolio");
  for (const p of body) {
    assert.equal(
      typeof p.startingCapital, "number",
      `paper-portfolios entry agentId=${p.agentId} missing numeric startingCapital`,
    );
    assert.ok(
      Number.isFinite(p.startingCapital) && p.startingCapital > 0,
      `paper-portfolios entry agentId=${p.agentId} has non-positive startingCapital=${p.startingCapital}`,
    );
  }
});

test("route contract: /crypto/agents/:id emits numeric startingCapital on the paperPortfolio", async () => {
  // Pull an agent id from the portfolios list so this test is
  // self-bootstrapping (no fixture id required).
  const list = await fetch(`${BASE}/crypto/paper-portfolios`).then(r => r.json());
  const id: number | undefined = list?.[0]?.agentId;
  assert.ok(typeof id === "number", "could not derive an agentId from /paper-portfolios");
  const r = await fetch(`${BASE}/crypto/agents/${id}`);
  assert.equal(r.status, 200);
  const body = await r.json();
  assert.ok(body.paperPortfolio, "expected paperPortfolio block on agent payload");
  assert.equal(
    typeof body.paperPortfolio.startingCapital, "number",
    `agents/${id} paperPortfolio missing numeric startingCapital`,
  );
  assert.ok(
    body.paperPortfolio.startingCapital > 0,
    `agents/${id} startingCapital non-positive: ${body.paperPortfolio.startingCapital}`,
  );
});
