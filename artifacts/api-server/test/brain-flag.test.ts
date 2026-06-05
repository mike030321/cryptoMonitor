import { test } from "node:test";
import assert from "node:assert/strict";

// Phase 5 — brain flag safety properties.
//
// We test the two properties the cutover absolutely depends on:
//   1) The default brain is LLM (enabled === false) when nothing is set.
//   2) The QUANT_BRAIN_FORCE_OFF=1 env var wins over any persisted state.
//
// Both properties are evaluated against the real module talking to the real
// app_settings table — DATABASE_URL is provisioned in the test env. We reset
// the row to a known state before each assertion so the tests don't leak.

import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import {
  BRAIN_FLAG_KEY,
  getBrainState,
  setBrainState,
  isQuantBrainEnabled,
  invalidateBrainCache,
} from "../src/lib/brain-flag";

async function clearFlagRow() {
  await db.delete(appSettingsTable).where(eq(appSettingsTable.key, BRAIN_FLAG_KEY));
  invalidateBrainCache();
}

test("default brain state is OFF (no row in app_settings)", async () => {
  delete process.env.QUANT_BRAIN_FORCE_OFF;
  await clearFlagRow();
  const state = await getBrainState();
  assert.equal(state.enabled, false);
  assert.equal(state.source, "default");
  assert.equal(await isQuantBrainEnabled(), false);
});

test("setBrainState(true,'manual') persists across cache invalidation", async () => {
  delete process.env.QUANT_BRAIN_FORCE_OFF;
  await clearFlagRow();
  const after = await setBrainState(true, "manual");
  assert.equal(after.enabled, true);
  invalidateBrainCache();
  const reread = await getBrainState();
  assert.equal(reread.enabled, true);
  assert.equal(reread.source, "manual");
  await clearFlagRow();
});

test("QUANT_BRAIN_FORCE_OFF=1 overrides a persisted ON state", async () => {
  await clearFlagRow();
  await setBrainState(true, "manual");
  process.env.QUANT_BRAIN_FORCE_OFF = "1";
  invalidateBrainCache();
  try {
    const state = await getBrainState();
    assert.equal(state.enabled, false);
    assert.equal(state.source, "env");
    assert.equal(await isQuantBrainEnabled(), false);
  } finally {
    delete process.env.QUANT_BRAIN_FORCE_OFF;
    await clearFlagRow();
  }
});

test("env force-off branch must NOT route through defaultState() (JSON-default safety)", async () => {
  // Regression: defaultState() reads QUANT_BRAIN_ENABLED from
  // shared/trading-frictions.json. If the env-force-off branch ever spreads
  // defaultState() into its return value (the bug fixed in this change),
  // an operator who shipped JSON with `quant_brain.enabled: true` could no
  // longer disable the brain via env. ESM exports cannot be monkey-patched
  // at runtime, so instead we assert the code-shape: the env-force-off
  // branch must hard-return `enabled: false` as a literal, not a spread.
  const fs = await import("node:fs/promises");
  const src = await fs.readFile(
    new URL("../src/lib/brain-flag.ts", import.meta.url),
    "utf8",
  );
  const envBranchMatch = src.match(/if \(envForceOff\(\)\) \{[\s\S]*?\}/);
  assert.ok(envBranchMatch, "expected an `if (envForceOff()) { ... }` branch in getBrainState");
  const branch = envBranchMatch[0];
  assert.match(branch, /enabled:\s*false/, "env-force-off branch must hard-set enabled:false");
  assert.doesNotMatch(branch, /defaultState\s*\(/, "env-force-off branch must NOT call defaultState() (would inherit JSON value)");
  assert.doesNotMatch(branch, /\.\.\.\s*defaultState/, "env-force-off branch must NOT spread defaultState()");
});

test("QUANT_BRAIN_FORCE_OFF=1 also blocks setBrainState(true)", async () => {
  await clearFlagRow();
  process.env.QUANT_BRAIN_FORCE_OFF = "1";
  invalidateBrainCache();
  try {
    const result = await setBrainState(true, "manual");
    assert.equal(result.enabled, false);
  } finally {
    delete process.env.QUANT_BRAIN_FORCE_OFF;
    await clearFlagRow();
  }
});
