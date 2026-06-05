// Task #356 — fail-fast coverage for the shared/trading-frictions.json
// loaders. Tasks #343 and #349 hardened both modules so a missing
// required key throws at startup instead of silently substituting a
// hardcoded default. This file pins those throw paths so a future
// refactor that quietly re-introduces a silent fallback fails CI here
// instead of in production.
//
// Strategy: load the real contract, structurally clone it, delete one
// required key (or per-tf `_default`), and call the exported validator
// against the mutation. The validators mirror EXACTLY the inline
// `_requireConfig` calls in the module-level wiring of each loader.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  assertContractRequiredKeys,
  _requireTfLookup,
} from "../src/lib/trading-constants";
import { assertPortfolioConfigRequiredKeys } from "../src/lib/portfolio-constraints";

function loadContract(): any {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const root = path.resolve(here, "..", "..", "..");
  return JSON.parse(
    readFileSync(path.join(root, "shared", "trading-frictions.json"), "utf8"),
  );
}

function deepClone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x));
}

function withDeleted(p: string[], obj: any): any {
  const out = deepClone(obj);
  let cur: any = out;
  for (let i = 0; i < p.length - 1; i++) cur = cur[p[i]];
  delete cur[p[p.length - 1]];
  return out;
}

// ── trading-constants.ts (task #343) ──────────────────────────────────────
const CONTRACT_REQUIRED_KEYS: { path: string[]; key: string }[] = [
  { path: ["quant_brain"], key: "quant_brain" },
  { path: ["quant_brain", "enabled"], key: "quant_brain.enabled" },
  { path: ["quant_brain", "auto_revert"], key: "quant_brain.auto_revert" },
  {
    path: ["quant_brain", "decision_thresholds"],
    key: "quant_brain.decision_thresholds",
  },
  {
    path: ["quant_brain", "decision_thresholds", "min_directional_prob"],
    key: "quant_brain.decision_thresholds.min_directional_prob",
  },
  {
    path: ["quant_brain", "decision_thresholds", "min_directional_edge"],
    key: "quant_brain.decision_thresholds.min_directional_edge",
  },
  {
    path: ["quant_brain", "decision_thresholds", "min_expected_return_pct_factor"],
    key: "quant_brain.decision_thresholds.min_expected_return_pct_factor",
  },
  {
    path: ["quant_brain", "decision_thresholds", "policy_version"],
    key: "quant_brain.decision_thresholds.policy_version",
  },
];

for (const { path: p, key } of CONTRACT_REQUIRED_KEYS) {
  test(`trading-constants: missing '${key}' throws fail-fast (task #343)`, () => {
    const mutated = withDeleted(p, loadContract());
    assert.throws(
      () => assertContractRequiredKeys(mutated),
      (err: unknown) => {
        const msg = (err as Error).message ?? "";
        assert.match(msg, new RegExp(`required key '${key.replace(/\./g, "\\.")}'`));
        assert.match(msg, /task #343/);
        return true;
      },
    );
  });
}

test("trading-constants: unmodified contract passes the assertion", () => {
  // Sanity check — without this, every "missing X throws" test could pass
  // for the wrong reason (e.g. base contract already missing something).
  assert.doesNotThrow(() => assertContractRequiredKeys(loadContract()));
});

// ── per-tf `_default` cascade (task #349) ─────────────────────────────────
// Each per-tf map (`outcome_thresholds_percent`, `tf_sl_multiplier`,
// `tf_tp_multiplier`, `tf_atr_floor_pct`) is allowed to use the JSON
// `_default` cascade, but losing the cascade entirely must throw — never
// silently fall back to a TS-side literal.
const TF_LOOKUP_BLOCKS = [
  "outcome_thresholds_percent",
  "tf_sl_multiplier",
  "tf_tp_multiplier",
  "tf_atr_floor_pct",
];

for (const block of TF_LOOKUP_BLOCKS) {
  test(`trading-constants: '${block}' missing _default throws on unknown tf (task #349)`, () => {
    const c = loadContract();
    const map = deepClone(c[block]);
    delete map._default;
    assert.throws(
      () => _requireTfLookup(map, "__no_such_tf__", block),
      (err: unknown) => {
        const msg = (err as Error).message ?? "";
        assert.match(msg, new RegExp(`'${block}'`));
        assert.match(msg, /__no_such_tf__/);
        assert.match(msg, /_default/);
        return true;
      },
    );
  });
}

// ── portfolio-constraints.ts (task #349) ──────────────────────────────────
const PORTFOLIO_REQUIRED_KEYS: { path: string[]; key: string }[] = [
  { path: ["portfolio_constraints"], key: "portfolio_constraints" },
  { path: ["portfolio_constraints", "enabled"], key: "portfolio_constraints.enabled" },
  {
    path: ["portfolio_constraints", "max_sector_exposure_pct"],
    key: "portfolio_constraints.max_sector_exposure_pct",
  },
  {
    path: ["portfolio_constraints", "max_correlated_exposure_pct"],
    key: "portfolio_constraints.max_correlated_exposure_pct",
  },
  {
    path: ["portfolio_constraints", "max_beta_to_btc"],
    key: "portfolio_constraints.max_beta_to_btc",
  },
  {
    path: ["portfolio_constraints", "regime_budget_pct"],
    key: "portfolio_constraints.regime_budget_pct",
  },
  {
    path: ["portfolio_constraints", "sector_map"],
    key: "portfolio_constraints.sector_map",
  },
  {
    path: ["portfolio_constraints", "sector_map", "_default"],
    key: "portfolio_constraints.sector_map._default",
  },
];

for (const { path: p, key } of PORTFOLIO_REQUIRED_KEYS) {
  test(`portfolio-constraints: missing '${key}' throws fail-fast (task #349)`, () => {
    const mutated = withDeleted(p, loadContract());
    assert.throws(
      () => assertPortfolioConfigRequiredKeys(mutated),
      (err: unknown) => {
        const msg = (err as Error).message ?? "";
        assert.match(msg, new RegExp(`required key '${key.replace(/\./g, "\\.")}'`));
        assert.match(msg, /task #349/);
        return true;
      },
    );
  });
}

test("portfolio-constraints: unmodified contract passes the assertion", () => {
  assert.doesNotThrow(() => assertPortfolioConfigRequiredKeys(loadContract()));
});
