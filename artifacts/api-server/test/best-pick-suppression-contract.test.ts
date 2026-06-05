import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// Task #532 / Rev 2 — when /best-pick short-circuits because the
// brain runtime is anything other than `online`, the payload MUST be
// honest:
//   * coinId must be "" (so the dashboard cannot render a coin link)
//   * coinName must be "No live consensus"
//   * action must be "hold"
//   * brain must be null
//   * suppressedReason must be one of the documented values
//   * brainRuntimeState must be set
//
// The frontend (TopPickCard) keys its "no live consensus" card off
// the truthiness of suppressedReason, so any non-null value is safe.
// This contract test prevents a future change from emitting a
// half-populated `bestPick` while still claiming to be suppressed.

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROUTE_FILE = join(__dirname, "..", "src", "routes", "crypto", "index.ts");
const FRONTEND_HOOK = join(
  __dirname,
  "..",
  "..",
  "crypto-monitor",
  "src",
  "hooks",
  "use-news.ts",
);

const ALLOWED_REASONS = new Set([
  "brain_offline",
  "brain_offline_no_model",
  "brain_status_unknown",
]);

const ALLOWED_RUNTIME_STATES = new Set([
  "online",
  "offline_disabled",
  "offline_no_model",
  "unknown",
]);

describe("Task #532 — /best-pick suppression contract", () => {
  const src = readFileSync(ROUTE_FILE, "utf8");

  test("every emitted suppressedReason is in the documented set", () => {
    // Pull every literal that follows `suppressedReason: "..."` (and
    // ignore the dynamic lookup line). Then check each is allowed.
    const literalRe = /suppressedReason:\s*"([^"]+)"/g;
    const found = new Set<string>();
    for (const m of src.matchAll(literalRe)) {
      found.add(m[1]);
    }
    assert.ok(found.size > 0, "expected to find at least one suppressedReason literal");
    for (const value of found) {
      assert.ok(
        ALLOWED_REASONS.has(value),
        `suppressedReason "${value}" is emitted by /best-pick but not in the documented set: ${[
          ...ALLOWED_REASONS,
        ].join(", ")}`,
      );
    }
  });

  test("dynamic suppressedReason lookup table contains only documented keys", () => {
    // The route maps runtime.state -> suppressedReason via a Record<>
    // table. Walk the table literal and assert every value is allowed.
    const tableMatch = src.match(/const reasonByState[^{]*\{([\s\S]*?)\};/);
    assert.ok(tableMatch, "expected to find the reasonByState lookup table");
    const tableSrc = tableMatch[1];
    const valueRe = /"([^"]+)"\s*,?/g;
    const valuesInTable: string[] = [];
    for (const m of tableSrc.matchAll(valueRe)) {
      valuesInTable.push(m[1]);
    }
    // Pairs are key,value — odd-indexed entries (1,3,5,…) are values.
    const onlyValues = valuesInTable.filter((_v, i) => i % 2 === 1);
    assert.ok(
      onlyValues.length > 0,
      "expected the reasonByState table to map at least one runtime state",
    );
    for (const value of onlyValues) {
      assert.ok(
        ALLOWED_REASONS.has(value),
        `reasonByState maps to "${value}" which is not documented`,
      );
    }
  });

  test("every emitted brainRuntimeState is in the documented set", () => {
    const stateRe = /brainRuntimeState:\s*"([^"]+)"/g;
    const found = new Set<string>();
    for (const m of src.matchAll(stateRe)) {
      found.add(m[1]);
    }
    for (const value of found) {
      assert.ok(
        ALLOWED_RUNTIME_STATES.has(value),
        `brainRuntimeState "${value}" is emitted but not in the documented set`,
      );
    }
  });

  test("every suppressed-payload site sets coinId='' and coinName='No live consensus'", () => {
    // Locate every `suppressedReason:` occurrence and check that the
    // surrounding ~25 lines also contain coinId: "", coinName: "No
    // live consensus", action: "hold", and brain: null. This locks
    // the no-coin-link / no-recommendation-layout contract.
    const lines = src.split("\n");
    const indices: number[] = [];
    for (let i = 0; i < lines.length; i++) {
      // Catch both `suppressedReason: "literal"` and the object
      // shorthand `suppressedReason,` form (the dynamic path that
      // resolves through the reasonByState table).
      if (/^\s*suppressedReason\s*[:,]/.test(lines[i])) indices.push(i);
    }
    assert.ok(indices.length >= 2, `expected ≥2 suppressed-payload sites, got ${indices.length}`);
    for (const idx of indices) {
      const window = lines.slice(Math.max(0, idx - 25), idx + 5).join("\n");
      assert.match(
        window,
        /coinId\s*:\s*""/,
        `site near line ${idx + 1} must emit coinId: "" so no coin link is rendered`,
      );
      assert.match(
        window,
        /coinName\s*:\s*"No live consensus"/,
        `site near line ${idx + 1} must emit coinName: "No live consensus"`,
      );
      assert.match(
        window,
        /action\s*:\s*"hold"/,
        `site near line ${idx + 1} must emit action: "hold"`,
      );
      assert.match(
        window,
        /brain\s*:\s*null/,
        `site near line ${idx + 1} must emit brain: null`,
      );
    }
  });

  test("frontend BestPick.suppressedReason union accepts every backend value", () => {
    const hookSrc = readFileSync(FRONTEND_HOOK, "utf8");
    for (const reason of ALLOWED_REASONS) {
      assert.ok(
        hookSrc.includes(`"${reason}"`),
        `frontend BestPick.suppressedReason union must list "${reason}" so TS is exhaustive`,
      );
    }
  });
});
