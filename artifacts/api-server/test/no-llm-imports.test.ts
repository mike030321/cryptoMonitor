/**
 * Task #444 — contract test that the LLM/news/sidecar plane stays
 * removed.
 *
 * The existing `no-llm-fields-in-trade-decisions.test.ts` /
 * `no-llm-fields-runtime.test.ts` pair already proves no LLM-derived
 * KEY can reach a trade-decision payload. This file is the IMPORT
 * companion: walks every `.ts` file under `artifacts/api-server/src`
 * and asserts none of them import from any of the deleted modules
 * (the LLM SDKs, the news plane, the sidecar plane, agent-evolution,
 * llm-bias-*, the `_legacy/ai-engine` constant). A future regression
 * — re-adding `import { something } from "openai"` somewhere in the
 * server — fails this test before the code even runs.
 *
 * Wired into `decision-engine-parity` so it runs alongside the rest
 * of the parity / cadence checks.
 */
import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(__dirname, "..", "src");

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, out);
    else if (st.isFile() && (full.endsWith(".ts") || full.endsWith(".tsx"))) out.push(full);
  }
  return out;
}

// Match either:
//   import ... from "<spec>"
//   export ... from "<spec>"      (re-exports, e.g. `export * from "openai"`)
//   import("<spec>")
//   require("<spec>")
// We strip comments / string contents in importing-files-only fashion
// — a literal mention of "openai" inside a doc-block is fine; only an
// actual import binding fails.
function importSpecsIn(src: string): string[] {
  const out: string[] = [];
  const reStatic = /\bimport\b[^"']*?from\s*["']([^"']+)["']/g;
  const reBareSide = /\bimport\s*["']([^"']+)["']/g;
  const reDynamic = /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g;
  const reRequire = /\brequire\s*\(\s*["']([^"']+)["']\s*\)/g;
  const reExportFrom = /\bexport\b[^"']*?from\s*["']([^"']+)["']/g;
  let m: RegExpExecArray | null;
  for (const re of [reStatic, reBareSide, reDynamic, reRequire, reExportFrom]) {
    re.lastIndex = 0;
    while ((m = re.exec(src)) !== null) out.push(m[1]);
  }
  return out;
}

// Strip JS comments so a doc-block like `// see openai docs` doesn't
// trip the matcher.
function stripComments(src: string): string {
  let out = "";
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    const next = src[i + 1];
    if (c === "/" && next === "/") {
      while (i < src.length && src[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && next === "*") {
      i += 2;
      while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) i++;
      i += 2;
      continue;
    }
    out += c;
    i++;
  }
  return out;
}

const FORBIDDEN_SPECS: Array<{ pattern: RegExp; reason: string }> = [
  { pattern: /^@google\/generative-ai$/i,
    reason: "Gemini SDK — removed with the LLM plane in Task #444." },
  { pattern: /^openai$/i,
    reason: "OpenAI SDK — removed with the LLM plane in Task #444." },
  { pattern: /^@anthropic-ai\//i,
    reason: "Anthropic SDK — never used; banned alongside the LLM plane in Task #444." },
  { pattern: /(^|\/)news-fetcher(\.[jt]sx?)?$/i,
    reason: "news-fetcher — deleted in Task #444." },
  { pattern: /(^|\/)news-classifier(\.[jt]sx?)?$/i,
    reason: "news-classifier — deleted in Task #444." },
  { pattern: /(^|\/)agent-evolution(\.[jt]sx?)?$/i,
    reason: "agent-evolution — deleted in Task #444." },
  { pattern: /(^|\/)llm-bias-monitor(\.[jt]sx?)?$/i,
    reason: "llm-bias-monitor — deleted in Task #444." },
  { pattern: /(^|\/)llm-bias-demote-tracker(\.[jt]sx?)?$/i,
    reason: "llm-bias-demote-tracker — deleted in Task #444." },
  { pattern: /(^|\/)llm-sidecar(\/|$)/i,
    reason: "llm-sidecar/ directory — deleted in Task #444." },
  { pattern: /(^|\/)_legacy\/ai-engine(\.[jt]sx?)?$/i,
    reason: "_legacy/ai-engine personality fleet — deleted in Task #444." },
];

describe("Task #444 — no source file imports any deleted LLM/news module", () => {
  const files = walk(SRC_ROOT);

  test("import scan finds at least the routes/crypto/index.ts (sanity)", () => {
    assert.ok(
      files.some((f) => f.endsWith(path.join("routes", "crypto", "index.ts"))),
      `walker did not find the routes file under ${SRC_ROOT}`,
    );
  });

  test("no source file imports from a forbidden module", () => {
    const offenders: string[] = [];
    for (const file of files) {
      const raw = readFileSync(file, "utf8");
      const cleaned = stripComments(raw);
      const specs = importSpecsIn(cleaned);
      for (const spec of specs) {
        for (const { pattern, reason } of FORBIDDEN_SPECS) {
          if (pattern.test(spec)) {
            offenders.push(`${path.relative(SRC_ROOT, file)} → "${spec}" (${reason})`);
          }
        }
      }
    }
    assert.deepEqual(
      offenders,
      [],
      `Forbidden imports found:\n  - ${offenders.join("\n  - ")}`,
    );
  });

  test("the four sidecar route paths are NOT registered", () => {
    const routesFile = path.join(SRC_ROOT, "routes", "crypto", "index.ts");
    const cleaned = stripComments(readFileSync(routesFile, "utf8"));
    const sidecarRoutePatterns: Array<{ pattern: RegExp; label: string }> = [
      { pattern: /\brouter\.\w+\s*\(\s*["'][^"']*\/llm\/sidecar\/state["']/,
        label: "/crypto/llm/sidecar/state" },
      { pattern: /\brouter\.\w+\s*\(\s*["'][^"']*\/llm\/sidecar\/toggle["']/,
        label: "/crypto/llm/sidecar/toggle" },
      { pattern: /\brouter\.\w+\s*\(\s*["'][^"']*\/llm\/sidecar\/recent["']/,
        label: "/crypto/llm/sidecar/recent" },
      { pattern: /\brouter\.\w+\s*\(\s*["'][^"']*\/llm\/copilot["']/,
        label: "/crypto/llm/copilot" },
    ];
    const stillRegistered = sidecarRoutePatterns
      .filter((p) => p.pattern.test(cleaned))
      .map((p) => p.label);
    assert.deepEqual(
      stillRegistered,
      [],
      `Removed sidecar routes still found in routes/crypto/index.ts: ${stillRegistered.join(", ")}`,
    );
  });

  test("negative control: a synthetic line containing a forbidden spec is detected", () => {
    const synthetic = [
      `import OpenAI from "openai";`,
      `import { X } from "@google/generative-ai";`,
      `export * from "openai";`,
      `export { Foo } from "../lib/agent-evolution";`,
    ].join("\n");
    const cleaned = stripComments(synthetic);
    const specs = importSpecsIn(cleaned);
    const found = specs.filter((s) =>
      FORBIDDEN_SPECS.some(({ pattern }) => pattern.test(s)),
    );
    assert.ok(
      found.length >= 4,
      `expected 4 synthetic imports/re-exports to be detected; got ${JSON.stringify(specs)}`,
    );
  });
});
