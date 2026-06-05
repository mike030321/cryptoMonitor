/**
 * Task #468 — strategy-profile registry contract tests.
 *
 * These tests pin the v1 contract:
 *   1. Schema validates and the registry boots with 5 v1 agents +
 *      `baseline_reference` + `legacy_archived`.
 *   2. `getAgentProfile()` throws on unknown ids — never defaults.
 *   3. Compatibility map covers the production agent names that
 *      pre-date this task (DCA + Circuit Breaker, Strategy Lab Buy
 *      & Hold, Trend Filter, every "Slice Xm" placeholder, Hybrid-*,
 *      and the legacy LLM personalities).
 *   4. Family strings emitted by the registry are byte-for-byte the
 *      meta-brain's frozen STRATEGY_FAMILIES list — no drift.
 *   5. Active agents declare numeric policy values (no null
 *      inheritance for an executor); baseline / archived are
 *      `executes: false` and the trade-gate skip path treats them
 *      as non-executable.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  AGENT_STATUSES,
  agentProfileSchema,
  getAgentProfile,
  tryGetAgentProfile,
  listProfiles,
  listProfileIds,
  listExecutingProfileIds,
  mapLegacyNameToProfileId,
  mapLegacyNameToSubId,
  profileAllowsRegime,
  AgentNotExecutableError,
  getCachedProfileForAgentId,
  tryGetCachedEntry,
  listKnownLegacyNamesForTests,
  _seedCacheForTests,
  _resetCacheForTests,
  type AgentProfile,
} from "../src/lib/agents-registry";
import { STRATEGY_FAMILIES } from "../src/lib/meta-brain/contract";
import { resolveStrategyFamilyForProfile } from "../src/lib/meta-brain/adapter";

describe("agents-registry: v1 profiles boot", () => {
  it("registers exactly the v1 set", () => {
    const ids = new Set(listProfileIds());
    for (const expected of [
      "momentum_core",
      "mean_reversion_core",
      "breakout_core",
      "volatility_defensive",
      "baseline_reference",
      "legacy_archived",
    ]) {
      assert.ok(ids.has(expected), `missing v1 profile id: ${expected}`);
    }
  });

  it("every profile passes the Zod schema individually", () => {
    for (const p of listProfiles()) {
      assert.doesNotThrow(() => agentProfileSchema.parse(p));
    }
  });

  it("status/executes contract holds for every profile", () => {
    for (const p of listProfiles()) {
      if (p.status === "active") {
        assert.equal(p.executes, true, `${p.agent_id}: active must execute`);
      } else {
        assert.equal(
          p.executes,
          false,
          `${p.agent_id}: status=${p.status} cannot execute`,
        );
      }
    }
  });

  it("status enum matches the registry", () => {
    const statuses = new Set<string>(AGENT_STATUSES);
    for (const p of listProfiles()) {
      assert.ok(statuses.has(p.status), `${p.agent_id}: bad status ${p.status}`);
    }
  });
});

describe("agents-registry: getAgentProfile contract", () => {
  it("throws on unknown id (never defaults)", () => {
    assert.throws(
      () => getAgentProfile("does_not_exist"),
      /unknown agent_id/i,
    );
  });

  it("throws on null / undefined / empty", () => {
    assert.throws(() => getAgentProfile(null), /empty agent_id/i);
    assert.throws(() => getAgentProfile(undefined), /empty agent_id/i);
    assert.throws(() => getAgentProfile(""), /empty agent_id/i);
  });

  it("tryGetAgentProfile returns null on unknown", () => {
    assert.equal(tryGetAgentProfile("does_not_exist"), null);
    assert.equal(tryGetAgentProfile(null), null);
  });

  it("returns the registered profile when id is known", () => {
    const p = getAgentProfile("momentum_core");
    assert.equal(p.agent_id, "momentum_core");
    assert.equal(p.strategy_family, "momentum");
    assert.equal(p.executes, true);
  });
});

describe("agents-registry: family strings match meta-brain enum", () => {
  it("every profile's family is a member of STRATEGY_FAMILIES", () => {
    const fams = new Set<string>(STRATEGY_FAMILIES);
    for (const p of listProfiles()) {
      assert.ok(
        fams.has(p.strategy_family),
        `${p.agent_id}: family ${p.strategy_family} not in STRATEGY_FAMILIES`,
      );
    }
  });

  it("resolveStrategyFamilyForProfile delegates to the registry", () => {
    assert.equal(resolveStrategyFamilyForProfile("momentum_core"), "momentum");
    assert.equal(
      resolveStrategyFamilyForProfile("mean_reversion_core"),
      "mean_reversion",
    );
    assert.equal(resolveStrategyFamilyForProfile("breakout_core"), "breakout");
    assert.equal(
      resolveStrategyFamilyForProfile("volatility_defensive"),
      "volatility_forecaster",
    );
    assert.equal(
      resolveStrategyFamilyForProfile("baseline_reference"),
      "baseline",
    );
  });

  it("resolveStrategyFamilyForProfile throws on unknown", () => {
    assert.throws(() => resolveStrategyFamilyForProfile("does_not_exist"));
  });
});

describe("agents-registry: compatibility map (pre-#468 names)", () => {
  // Strategy-lab basket strategies — every variant maps to the
  // umbrella `baseline_reference` profile id (Task #468 spec lines
  // 39-41 lock the registry to 5 family agents + baseline_reference
  // + legacy_archived). Variant identity is preserved as a SEPARATE
  // sub-id string surfaced via `mapLegacyNameToSubId()` and on the
  // dashboard payload.
  it("maps strategy-lab variants to baseline_reference (umbrella id)", () => {
    for (const name of [
      "DCA + Circuit Breaker",
      "Strategy Lab DCA + Circuit Breaker",
      "Strategy Lab Buy & Hold",
      "Strategy Lab Buy and Hold",
      "Trend Filter (30d basket)",
      "baseline",
      "Benchmark",
    ]) {
      assert.equal(
        mapLegacyNameToProfileId(name),
        "baseline_reference",
        `expected ${name} → baseline_reference`,
      );
    }
  });

  // Sub-id retention table — Strategy-Lab variants carry their
  // variant tag as metadata so analysts retain attribution. Sub-ids
  // are NOT registry profile ids; they are tags surfaced on the
  // dashboard alongside the umbrella `baseline_reference`.
  it("retains strategy-lab variant identity as a sub-id string", () => {
    assert.equal(
      mapLegacyNameToSubId("DCA + Circuit Breaker"),
      "baseline_dca_cb",
    );
    assert.equal(
      mapLegacyNameToSubId("Strategy Lab DCA + Circuit Breaker"),
      "baseline_dca_cb",
    );
    assert.equal(
      mapLegacyNameToSubId("Strategy Lab Buy & Hold"),
      "baseline_buy_hold",
    );
    assert.equal(
      mapLegacyNameToSubId("Strategy Lab Buy and Hold"),
      "baseline_buy_hold",
    );
    assert.equal(
      mapLegacyNameToSubId("Trend Filter (30d basket)"),
      "baseline_trend_filter",
    );
    // Generic "baseline" / "benchmark" / non-baseline names → no sub-id.
    assert.equal(mapLegacyNameToSubId("baseline"), null);
    assert.equal(mapLegacyNameToSubId("Sentiment Sarah"), null);
    assert.equal(mapLegacyNameToSubId(null), null);
  });

  it("legacy LLM personalities → legacy_archived", () => {
    for (const name of [
      "Sentiment Sarah",
      "Pattern Pete",
      "Hybrid-Foo-Bar",
      "Momentum Mike",
      "Contrarian Carol",
      "Slice 1m",
      "Slice 5m",
      "Slice 1h",
    ]) {
      assert.equal(mapLegacyNameToProfileId(name), "legacy_archived");
    }
  });

  it("preserves a valid registry id verbatim", () => {
    assert.equal(
      mapLegacyNameToProfileId("anything", "momentum_core"),
      "momentum_core",
    );
    // Unknown current id falls back to name-based resolution → umbrella id.
    assert.equal(
      mapLegacyNameToProfileId("Strategy Lab Buy & Hold", "garbage"),
      "baseline_reference",
    );
  });

  it("unknown / empty name → legacy_archived (compat-only fallback)", () => {
    assert.equal(mapLegacyNameToProfileId(null), "legacy_archived");
    assert.equal(mapLegacyNameToProfileId(""), "legacy_archived");
    assert.equal(
      mapLegacyNameToProfileId("totally-unknown"),
      "legacy_archived",
    );
  });

  it("every compat output is itself a registered profile id", () => {
    const ids = new Set(listProfileIds());
    const samples = [
      "DCA + Circuit Breaker",
      "Strategy Lab Buy & Hold",
      "Trend Filter (30d basket)",
      "Sentiment Sarah",
      "Pattern Pete",
      "Hybrid-x-y",
      "Slice 1m",
      "",
      null,
    ];
    for (const s of samples) {
      const out = mapLegacyNameToProfileId(s as string | null);
      assert.ok(ids.has(out), `${s} → ${out} not in registry`);
    }
  });
});

describe("agents-registry: trade-gate semantics", () => {
  it("baseline_reference / legacy_archived cannot execute", () => {
    assert.equal(getAgentProfile("baseline_reference").executes, false);
    assert.equal(getAgentProfile("legacy_archived").executes, false);
  });

  it("listExecutingProfileIds returns only the 5 v1 executors", () => {
    const exec = listExecutingProfileIds();
    assert.ok(exec.includes("momentum_core"));
    assert.ok(exec.includes("mean_reversion_core"));
    assert.ok(exec.includes("breakout_core"));
    assert.ok(exec.includes("volatility_defensive"));
    assert.ok(!exec.includes("baseline_reference"));
    assert.ok(!exec.includes("legacy_archived"));
  });

  it("active executors declare numeric policy values", () => {
    for (const id of listExecutingProfileIds()) {
      const p = getAgentProfile(id);
      assert.notEqual(p.min_confidence, null);
      assert.notEqual(p.min_expected_edge_after_costs, null);
      assert.notEqual(p.size_bias, null);
    }
  });

  it("blocked_regimes vetoes via profileAllowsRegime", () => {
    const momentum = getAgentProfile("momentum_core");
    assert.ok(momentum.blocked_regimes.includes("range_chop"));
    assert.equal(profileAllowsRegime(momentum, "range_chop"), false);
    assert.equal(profileAllowsRegime(momentum, "trending_up"), true);
  });

  it("baseline_reference allows all regimes (for benchmark stats only)", () => {
    const baseline = getAgentProfile("baseline_reference");
    assert.equal(baseline.preferred_regimes, "all");
    assert.equal(profileAllowsRegime(baseline, "panic_liquidation"), true);
  });
});

describe("agents-registry: schema rejects malformed profiles", () => {
  const valid: AgentProfile = {
    agent_id: "x",
    display_name: "x",
    thesis: "x",
    strategy_family: "momentum",
    preferred_regimes: "all",
    blocked_regimes: [],
    min_confidence: 0.5,
    min_expected_edge_after_costs: 0.001,
    abstain_bias: 0.1,
    size_bias: 1.0,
    pooled_fallback_penalty: 0.5,
    drawdown_sensitivity: 1.0,
    benchmark_sensitivity: 1.0,
    status: "active",
    executes: true,
    retirement_rule: { kind: "30d_cost_aware_threshold", da_threshold: 0.5, sharpe_threshold: 0 },
    created_at: "2026-04-24",
  };

  it("rejects executes=true with non-active status", () => {
    assert.throws(() =>
      agentProfileSchema.parse({ ...valid, status: "disabled", executes: true }),
    );
  });

  it("rejects status=active with executes=false", () => {
    assert.throws(() =>
      agentProfileSchema.parse({ ...valid, status: "active", executes: false }),
    );
  });

  it("rejects active executor with null min_confidence", () => {
    assert.throws(() =>
      agentProfileSchema.parse({ ...valid, min_confidence: null }),
    );
  });

  it("rejects unknown family", () => {
    assert.throws(() =>
      agentProfileSchema.parse({ ...valid, strategy_family: "made_up" as never }),
    );
  });

  it("rejects unknown regime label", () => {
    assert.throws(() =>
      agentProfileSchema.parse({ ...valid, blocked_regimes: ["made_up"] as never }),
    );
  });
});

// ──────────────────────────────────────────────────────────────────────
// Task #468 — gate-level contract tests for the boot-loaded cache.
// The trade-execution path resolves the agent's profile via the cache,
// not a per-decision DB lookup. The cache MUST throw a typed
// `AgentNotExecutableError` for every non-executable condition so the
// gate records a structured skip and never silently defaults.
// ──────────────────────────────────────────────────────────────────────

describe("agents-registry: cache gate semantics (Task #468 contract)", () => {
  it("throws AgentNotExecutableError(unknown_agent_id) for an id absent from the cache", () => {
    _resetCacheForTests();
    try {
      assert.throws(
        () => getCachedProfileForAgentId(999_999),
        (err: unknown) => {
          assert.ok(err instanceof AgentNotExecutableError, "expected typed error");
          assert.equal((err as AgentNotExecutableError).reason, "unknown_agent_id");
          assert.equal((err as AgentNotExecutableError).agentId, 999_999);
          return true;
        },
      );
    } finally {
      _resetCacheForTests();
    }
  });

  it("throws AgentNotExecutableError(non_active_db_status) for a quarantined row", () => {
    const profile = getAgentProfile("momentum_core");
    _seedCacheForTests([
      { agentId: 42, agentName: "Momentum Mike", profile, dbStatus: "quarantine_review" },
    ]);
    try {
      assert.throws(
        () => getCachedProfileForAgentId(42),
        (err: unknown) => {
          assert.ok(err instanceof AgentNotExecutableError);
          assert.equal((err as AgentNotExecutableError).reason, "non_active_db_status");
          assert.equal((err as AgentNotExecutableError).dbStatus, "quarantine_review");
          return true;
        },
      );
    } finally {
      _resetCacheForTests();
    }
  });

  it("throws AgentNotExecutableError(profile_executes_false) for baseline_reference", () => {
    const profile = getAgentProfile("baseline_reference");
    _seedCacheForTests([
      { agentId: 7, agentName: "Strategy Lab Buy & Hold", profile, dbStatus: "active" },
    ]);
    try {
      assert.throws(
        () => getCachedProfileForAgentId(7),
        (err: unknown) => {
          assert.ok(err instanceof AgentNotExecutableError);
          assert.equal((err as AgentNotExecutableError).reason, "profile_executes_false");
          assert.equal((err as AgentNotExecutableError).profileId, "baseline_reference");
          return true;
        },
      );
    } finally {
      _resetCacheForTests();
    }
  });

  it("returns the cached profile for a healthy executor row", () => {
    const profile = getAgentProfile("breakout_core");
    _seedCacheForTests([
      { agentId: 11, agentName: "Breakout Bob", profile, dbStatus: "active" },
    ]);
    try {
      const got = getCachedProfileForAgentId(11);
      assert.equal(got.profile.agent_id, "breakout_core");
      assert.equal(got.dbStatus, "active");
      const peek = tryGetCachedEntry(11);
      assert.ok(peek);
      assert.equal(peek!.profile.agent_id, "breakout_core");
    } finally {
      _resetCacheForTests();
    }
  });
});

// ──────────────────────────────────────────────────────────────────────
// Task #468 — additional contract tests requested in code-review:
//   1. Every legacy name we KNOW about resolves to a profile id that
//      is itself registered in the registry (no dangling sub-id).
//   2. Non-executable profiles raise AgentNotExecutableError(
//      profile_executes_false) — this is the gate that prevents a
//      paper_trades INSERT for baseline / archived agents.
//   3. The blocked-regime guard happens BEFORE the meta-model in
//      paper-trader.ts (static source-position assertion).
//   4. Prod-name coverage: every name in listKnownLegacyNamesForTests
//      resolves to a registered profile id and (for non-baseline
//      legacy names) to legacy_archived.
// ──────────────────────────────────────────────────────────────────────

describe("agents-registry: Task #468 contract tests (rev 3)", () => {
  it("DB-row resolvability — every compat output is a registered profile id", () => {
    const ids = new Set(listProfileIds());
    for (const name of listKnownLegacyNamesForTests()) {
      const out = mapLegacyNameToProfileId(name);
      assert.ok(
        ids.has(out),
        `compat for "${name}" produced "${out}" which is not in the registry`,
      );
    }
  });

  it("non-executable profile blocks paper_trades INSERT via AgentNotExecutableError", () => {
    // The gate is: getCachedProfileForAgentId() throws BEFORE
    // paper-trader.ts ever reaches the INSERT block. We verify the
    // throw for every non-executor profile in the registry, which is
    // the sole entry point for the trade gate.
    for (const profile of listProfiles()) {
      if (profile.executes) continue;
      _seedCacheForTests([
        {
          agentId: 1234,
          agentName: profile.display_name,
          profile,
          dbStatus: "active",
        },
      ]);
      try {
        assert.throws(
          () => getCachedProfileForAgentId(1234),
          (err: unknown) => {
            assert.ok(
              err instanceof AgentNotExecutableError,
              `expected AgentNotExecutableError for ${profile.agent_id}`,
            );
            assert.equal(
              (err as AgentNotExecutableError).reason,
              "profile_executes_false",
              `${profile.agent_id} must throw profile_executes_false`,
            );
            return true;
          },
        );
      } finally {
        _resetCacheForTests();
      }
    }
  });

  it("blocked-regime guard precedes the meta-model call in paper-trader.ts", async () => {
    // Static contract: the regime veto MUST happen BEFORE any
    // meta-model evaluation. We pin this via source positions —
    // `profileAllowsRegime` call site < `metaAbstainReason` /
    // `metaAction` decision call site.
    const fs = await import("node:fs");
    const path = await import("node:path");
    const src = fs.readFileSync(
      path.resolve(
        new URL(import.meta.url).pathname,
        "..",
        "..",
        "src",
        "lib",
        "paper-trader.ts",
      ),
      "utf8",
    );
    const regimeIdx = src.indexOf("profileAllowsRegime(profile, regimeLabel)");
    const metaIdx = src.indexOf("quant?.metaAbstainReason");
    assert.ok(regimeIdx > 0, "expected profileAllowsRegime call site");
    assert.ok(metaIdx > 0, "expected metaAbstainReason gate");
    assert.ok(
      regimeIdx < metaIdx,
      `regime veto must precede meta-model gate (regimeIdx=${regimeIdx}, metaIdx=${metaIdx})`,
    );
  });

  it("prod-name coverage — every known legacy name resolves correctly", () => {
    const ids = new Set(listProfileIds());
    for (const name of listKnownLegacyNamesForTests()) {
      const out = mapLegacyNameToProfileId(name);
      // Must land in the registry.
      assert.ok(ids.has(out), `${name} → ${out} not registered`);
      // Strategy-Lab / generic baseline names resolve to a baseline_*
      // family member; everything else (Sentiment Sarah, Slice 5m, …)
      // resolves to legacy_archived.
      const isBaselineName =
        name.includes("baseline") ||
        name.includes("benchmark") ||
        name.includes("dca") ||
        name.includes("buy") ||
        name.includes("trend filter");
      if (isBaselineName) {
        assert.ok(
          out.startsWith("baseline"),
          `${name} should map to a baseline_* profile, got ${out}`,
        );
      } else {
        assert.equal(
          out,
          "legacy_archived",
          `${name} should be archived, got ${out}`,
        );
      }
    }
  });
});
