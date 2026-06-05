/**
 * Task #574 — `slice_role` is sent on every `/record-outcome` HTTP
 * call from api-server, resolved at outcome-submission time from
 * `shared/timeframe-roles.json`.
 *
 * Two layers of coverage:
 *   1. Wire-shape unit tests — a `shadow`-roled timeframe sends
 *      `slice_role: "shadow"`; a `disabled`-roled timeframe sends
 *      `slice_role: "disabled"`. Resolution happens at submission
 *      time so a role flip mid-trade is honoured.
 *   2. End-to-end round-trip — open + close a `shadow`-roled trade
 *      against a Python-brain mock that mirrors the real Python
 *      gating logic (only `trade` outcomes mutate `trust_by_family`;
 *      every accepted role bumps `inputs_by_role`; `disabled` is
 *      rejected). Asserts that `inputs_by_role.shadow` increments on
 *      `/stats` and `trust_by_family` is NOT mutated.
 */
import { describe, it, beforeEach, afterEach, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { writeFileSync, readFileSync, existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  bindTradeToTick,
  sendRecordOutcome,
  __resetAdapterState,
} from "../src/lib/meta-brain/adapter";
import { postRecordOutcome } from "../src/lib/meta-brain/client";
import {
  _resetTimeframeRolesCacheForTests,
} from "../src/lib/timeframe-roles";

// ── workspace-root resolution (mirrors the lib) ───────────────────
function findWorkspaceRoot(): string {
  let cur = path.dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 8; i++) {
    if (existsSync(path.join(cur, "pnpm-workspace.yaml"))) return cur;
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  throw new Error("workspace root not found");
}

const ROLES_FILE = path.join(findWorkspaceRoot(), "shared", "timeframe-roles.json");
const ORIGINAL_CONTENT = readFileSync(ROLES_FILE, "utf8");

function writeRoles(doc: unknown): void {
  mkdirSync(path.dirname(ROLES_FILE), { recursive: true });
  writeFileSync(ROLES_FILE, JSON.stringify(doc, null, 2));
  _resetTimeframeRolesCacheForTests();
}

function restoreRoles(): void {
  writeFileSync(ROLES_FILE, ORIGINAL_CONTENT);
  _resetTimeframeRolesCacheForTests();
}

function rolesDoc(roles: Record<string, "trade" | "shadow" | "context" | "disabled">): unknown {
  const timeframes: Record<string, unknown> = {};
  for (const [tf, role] of Object.entries(roles)) {
    timeframes[tf] = {
      role,
      context_subkind: role === "context" ? "filter" : null,
      disabled_reason: role === "disabled" ? "by_safety" : null,
      reason: `task-574 fixture ${tf}=${role}`,
      evidence_ref: "test-fixture",
      last_reviewed_at: new Date().toISOString(),
      promoted_slices_in_tf: [],
    };
  }
  return {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    generated_by_task: "test-574",
    timeframes,
  };
}

function startServer(handler: http.RequestListener) {
  return new Promise<{ url: string; close: () => Promise<void> }>((resolve) => {
    const s = http.createServer(handler);
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as AddressInfo).port;
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise<void>((res) => s.close(() => res())),
      });
    });
  });
}

const baseOutcome = {
  realized_pnl: 0.01,
  realized_drawdown: 0.005,
  realized_stability: 0.7,
  turnover_cost: 0.001,
  action_churn: null,
  correct_defense: null,
  correct_suppression: null,
  missed_edge_cost: null,
};

describe("meta-brain record_outcome — slice_role on the wire (Task #574)", () => {
  before(() => {
    // Pin the roles fixture for the entire suite so every TF used
    // below has a known, deterministic role independent of whatever
    // the live shared/timeframe-roles.json says today.
    writeRoles(rolesDoc({
      "1h": "shadow",
      "1d": "trade",
      "5m": "disabled",
    }));
  });
  after(() => {
    restoreRoles();
  });
  beforeEach(() => {
    __resetAdapterState();
    process.env.META_BRAIN_ENABLED = "1";
  });
  afterEach(() => {
    delete process.env.META_BRAIN_ENABLED;
    delete process.env.ML_ENGINE_URL;
  });

  it("wire payload includes slice_role='shadow' for a shadow-roled timeframe", async () => {
    let received: any;
    const srv = await startServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        received = JSON.parse(raw);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;

    const ok = await postRecordOutcome({
      tick_id: "shadow:uuid-shadow-1",
      timeframe: "1h",
      outcome: baseOutcome,
    });
    assert.equal(ok, true);
    assert.equal(received.slice_role, "shadow");
    // Sanity: tick_id had its `shadow:` prefix stripped on the wire.
    assert.equal(received.tick_id, "uuid-shadow-1");

    await srv.close();
  });

  it("wire payload includes slice_role='trade' for a trade-roled timeframe", async () => {
    let received: any;
    const srv = await startServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        received = JSON.parse(raw);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;

    const ok = await postRecordOutcome({
      tick_id: "uuid-trade-1",
      timeframe: "1d",
      outcome: baseOutcome,
    });
    assert.equal(ok, true);
    assert.equal(received.slice_role, "trade");

    await srv.close();
  });

  it("wire payload includes slice_role='disabled' when the timeframe is disabled", async () => {
    // The brain rejects `disabled` outcomes (returns ok:false). We
    // verify the WIRE PAYLOAD carries the role faithfully — the
    // brain's rejection then triggers an audible audit trail rather
    // than silently mutating trust state.
    let received: any;
    const srv = await startServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        received = JSON.parse(raw);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: false, reason: "disabled_role_rejected" }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;

    const ok = await postRecordOutcome({
      tick_id: "uuid-disabled-1",
      timeframe: "5m",
      outcome: baseOutcome,
    });
    assert.equal(ok, false);
    assert.equal(received.slice_role, "disabled");

    await srv.close();
  });

  it("wire payload uses the CURRENT registry role (resolution happens at submission time, not at trade-open time)", async () => {
    // The trade is bound at "trade-open" time (we just bind directly
    // here). Then the operator flips the timeframe's role to shadow.
    // The next submission MUST send the new role, not the role that
    // was in effect when the trade opened.
    bindTradeToTick(101, "uuid-flip");

    // Roles are still pinned by the suite-level `before`. Flip 1d
    // from trade → shadow now (mid-"trade").
    writeRoles(rolesDoc({
      "1h": "shadow",
      "1d": "shadow", // <-- flipped
      "5m": "disabled",
    }));

    let received: any;
    const srv = await startServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        received = JSON.parse(raw);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;

    await sendRecordOutcome(
      { tick_id: "uuid-flip", timeframe: "1d", outcome: baseOutcome },
      { tradeId: 101 },
    );
    assert.equal(received.slice_role, "shadow");

    await srv.close();

    // Restore the suite fixture for any subsequent test in this file.
    writeRoles(rolesDoc({
      "1h": "shadow",
      "1d": "trade",
      "5m": "disabled",
    }));
  });

  it("e2e: shadow trade open+close → /stats shows inputs_by_role.shadow incremented and trust_by_family unchanged", async () => {
    // Mirrors the Python `record_outcome` gating in
    // artifacts/ml-engine/app/meta_brain.py: only `trade` outcomes
    // mutate trust_by_family; every accepted role bumps inputs_by_role;
    // `disabled` is rejected before any state change.
    const inputsByRole: Record<string, number> = {
      trade: 0,
      shadow: 0,
      context: 0,
      disabled: 0,
    };
    const trustByFamily: Record<string, { trust: number }> = {
      momentum: { trust: 1.0 },
      mean_reversion: { trust: 1.0 },
    };
    const initialTrustSnapshot = JSON.parse(JSON.stringify(trustByFamily));

    const srv = await startServer((req, res) => {
      if (req.method === "POST" && req.url === "/ml/meta-brain/record-outcome") {
        let raw = "";
        req.on("data", (c) => (raw += c));
        req.on("end", () => {
          const body = JSON.parse(raw);
          const role = body.slice_role;
          if (role === "disabled") {
            res.writeHead(200, { "content-type": "application/json" });
            res.end(JSON.stringify({ ok: false, reason: "disabled_role_rejected" }));
            return;
          }
          if (!role || !(role in inputsByRole)) {
            res.writeHead(400, { "content-type": "application/json" });
            res.end(JSON.stringify({ ok: false, reason: "invalid_slice_role" }));
            return;
          }
          inputsByRole[role] += 1;
          if (role === "trade") {
            // Only `trade` mutates trust state.
            trustByFamily.momentum.trust *= 1.01;
          }
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify({ ok: true }));
        });
        return;
      }
      if (req.method === "GET" && req.url === "/ml/meta-brain/stats") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({
          inputs_by_role: inputsByRole,
          trust_by_family: trustByFamily,
        }));
        return;
      }
      res.writeHead(404);
      res.end();
    });
    process.env.ML_ENGINE_URL = srv.url;

    // Open a trade bound to a shadow-roled tick (the `shadow:` prefix
    // models the brain ran but in shadow-only mode at evaluate time;
    // the record-outcome wire still carries the resolved role).
    bindTradeToTick(7, "shadow:uuid-e2e-shadow");

    // Close: realized outcome flows through sendRecordOutcome which
    // calls postRecordOutcome which resolves slice_role from the
    // (currently shadow-pinned) "1h" timeframe.
    await sendRecordOutcome(
      { tick_id: "shadow:uuid-e2e-shadow", timeframe: "1h", outcome: baseOutcome },
      { tradeId: 7 },
    );

    // Pull /stats from the mock brain.
    const statsRes = await fetch(`${srv.url}/ml/meta-brain/stats`);
    const stats = await statsRes.json() as {
      inputs_by_role: Record<string, number>;
      trust_by_family: Record<string, { trust: number }>;
    };

    assert.equal(stats.inputs_by_role.shadow, 1, "inputs_by_role.shadow must increment");
    assert.equal(stats.inputs_by_role.trade, 0, "shadow outcome must NOT count as a trade input");
    assert.deepEqual(
      stats.trust_by_family,
      initialTrustSnapshot,
      "trust_by_family must NOT be mutated by a shadow outcome",
    );

    await srv.close();
  });
});
