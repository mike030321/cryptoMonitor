import { test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync, rmSync, existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Task #550 — role-gate unit tests for the brain promotion gate
// (gate #2). These exercise hasPromotedSlice() with an injected
// fetcher AND temporary edits to shared/timeframe-roles.json so the
// per-timeframe role layer can be observed independently of the
// verification-history layer.
//
// The two layers are AND-ed: a non-trade role on any requested TF
// must short-circuit BEFORE the verification-history fetch (so a
// fail-closed roles file refuses everything even when ml-engine is
// down), and a trade-roled TF must still be subject to the
// verification-history check.

import {
  hasPromotedSlice,
} from "../src/lib/brain-promotion-gate";
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

function fakeFetcher(payload: unknown, init: { status?: number } = {}) {
  return (async (_url: string | URL | Request, _opts?: RequestInit) => {
    return new Response(JSON.stringify(payload), {
      status: init.status ?? 200,
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch;
}

const PASSING_HISTORY = {
  rows: [
    {
      recorded_at: 1777030969,
      verification_status: "ok",
      passed: true,
      coins_with_promotion: ["bitcoin"],
      counts: { slices_promoted: 1 },
    },
  ],
};

function rolesDoc(roles: Record<string, {
  role: "trade" | "shadow" | "context" | "disabled";
  context_subkind?: "filter" | "regime" | "risk_state" | null;
  disabled_reason?: "by_data" | "by_gate" | "by_operator" | "by_safety" | null;
}>): unknown {
  const timeframes: Record<string, unknown> = {};
  for (const [tf, spec] of Object.entries(roles)) {
    timeframes[tf] = {
      role: spec.role,
      context_subkind: spec.role === "context" ? (spec.context_subkind ?? "filter") : null,
      disabled_reason: spec.role === "disabled" ? (spec.disabled_reason ?? "by_safety") : null,
      reason: `test fixture for ${tf}=${spec.role}`,
      evidence_ref: "test-fixture",
      last_reviewed_at: new Date().toISOString(),
      promoted_slices_in_tf: [],
    };
  }
  return {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    generated_by_task: "test",
    timeframes,
  };
}

test("role gate: refuses requested TF when role is 'shadow' BEFORE fetching history", async () => {
  writeRoles(rolesDoc({
    "1h": { role: "shadow" },
    "1d": { role: "trade" },
  }));
  let fetcherCalled = false;
  const verdict = await hasPromotedSlice({
    fetcher: (async () => {
      fetcherCalled = true;
      throw new Error("should not be called");
    }) as unknown as typeof fetch,
    mlEngineBaseUrl: "http://fake",
    requestedTimeframes: ["1h"],
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "gate_pre_check_failed_by_role");
  assert.equal(fetcherCalled, false, "role gate must short-circuit before history fetch");
  assert.deepEqual(verdict.permitted_timeframes, ["1d"]);
  assert.equal(verdict.evidence?.refused_timeframes?.[0]?.timeframe, "1h");
  assert.equal(verdict.evidence?.refused_timeframes?.[0]?.role, "shadow");
  restoreRoles();
});

test("role gate: refuses 'context' and surfaces context_subkind in evidence", async () => {
  writeRoles(rolesDoc({
    "1m": { role: "context", context_subkind: "filter" },
    "1d": { role: "trade" },
  }));
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher(PASSING_HISTORY),
    mlEngineBaseUrl: "http://fake",
    requestedTimeframes: ["1m"],
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "gate_pre_check_failed_by_role");
  const refused = verdict.evidence?.refused_timeframes?.[0];
  assert.equal(refused?.role, "context");
  assert.equal(refused?.context_subkind, "filter");
  assert.equal(refused?.disabled_reason, null);
  restoreRoles();
});

test("role gate: refuses 'disabled' and surfaces disabled_reason in evidence", async () => {
  writeRoles(rolesDoc({
    "5m": { role: "disabled", disabled_reason: "by_data" },
    "1d": { role: "trade" },
  }));
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher(PASSING_HISTORY),
    mlEngineBaseUrl: "http://fake",
    requestedTimeframes: ["5m"],
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "gate_pre_check_failed_by_role");
  const refused = verdict.evidence?.refused_timeframes?.[0];
  assert.equal(refused?.role, "disabled");
  assert.equal(refused?.disabled_reason, "by_data");
  restoreRoles();
});

test("role gate: requestedTimeframes with multiple TFs reports ALL refused, not just the first", async () => {
  writeRoles(rolesDoc({
    "1m": { role: "context", context_subkind: "filter" },
    "5m": { role: "disabled", disabled_reason: "by_data" },
    "1d": { role: "trade" },
  }));
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher(PASSING_HISTORY),
    mlEngineBaseUrl: "http://fake",
    requestedTimeframes: ["1m", "5m", "1d"],
  });
  assert.equal(verdict.ok, false);
  const refused = verdict.evidence?.refused_timeframes ?? [];
  const tfs = refused.map((r) => r.timeframe).sort();
  assert.deepEqual(tfs, ["1m", "5m"]);
  restoreRoles();
});

test("role gate: 'trade' role passes role gate; downstream history gate still applies", async () => {
  writeRoles(rolesDoc({ "1d": { role: "trade" } }));
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher(PASSING_HISTORY),
    mlEngineBaseUrl: "http://fake",
    requestedTimeframes: ["1d"],
  });
  assert.equal(verdict.ok, true);
  assert.deepEqual(verdict.permitted_timeframes, ["1d"]);
  restoreRoles();
});

test("role gate: when no requestedTimeframes provided, role gate is a no-op (back-compat)", async () => {
  writeRoles(rolesDoc({ "1d": { role: "trade" } }));
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher(PASSING_HISTORY),
    mlEngineBaseUrl: "http://fake",
  });
  assert.equal(verdict.ok, true);
  assert.deepEqual(verdict.permitted_timeframes, ["1d"]);
  restoreRoles();
});

test("role gate: malformed roles file fails CLOSED (every TF disabled by_safety)", async () => {
  // Write garbage that does not parse as the documented schema.
  writeFileSync(ROLES_FILE, "{ this is not valid json");
  _resetTimeframeRolesCacheForTests();
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher(PASSING_HISTORY),
    mlEngineBaseUrl: "http://fake",
    requestedTimeframes: ["1d"],
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "gate_pre_check_failed_by_role");
  assert.deepEqual(verdict.permitted_timeframes, [], "fail-closed must permit nothing");
  const refused = verdict.evidence?.refused_timeframes?.[0];
  assert.equal(refused?.role, "disabled");
  assert.equal(refused?.disabled_reason, "by_safety");
  restoreRoles();
});

test("role gate: missing roles file fails CLOSED", async () => {
  rmSync(ROLES_FILE);
  _resetTimeframeRolesCacheForTests();
  const verdict = await hasPromotedSlice({
    fetcher: fakeFetcher(PASSING_HISTORY),
    mlEngineBaseUrl: "http://fake",
    requestedTimeframes: ["1d"],
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.reason, "gate_pre_check_failed_by_role");
  assert.deepEqual(verdict.permitted_timeframes, []);
  restoreRoles();
});
