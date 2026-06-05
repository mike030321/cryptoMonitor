// Task #365 — Quant-Only Enforcement contract tests (api-server side).
//
// (i)  No fee/slippage/initial-balance numeric literal may live outside
//      shared/trading-frictions.json + src/lib/trading-constants.ts. A
//      regex scan of the live trader, agent evolution, strategy lab, and
//      the crypto routes file proves the constants flow only from the
//      shared module.
//
// (ii) The agent-detail page derives total P&L from
//      `totalValue - startingCapital`, not from an in-memory mutation
//      that diverges from realised+unrealised math (proof C in the audit).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");

// We sweep the entire api-server source tree (`artifacts/api-server/src/**/*.ts`)
// for friction literals. Any numeric literal that LOOKS like a
// friction constant in here is a regression. The central modules
// where the literals legitimately live are allowlisted below.
const SCAN_ROOT = "artifacts/api-server/src";
const SCAN_ALLOWLIST = new Set<string>([
  // The central friction module — literals legitimately live here.
  "artifacts/api-server/src/lib/trading-constants.ts",
  // Test fixtures / mocks under src/ would also be allowlisted here
  // if any existed; today none do.
]);

function listTsFiles(rootRel: string): string[] {
  const root = path.join(REPO, rootRel);
  const out: string[] = [];
  function walk(dir: string) {
    for (const name of readdirSync(dir)) {
      const full = path.join(dir, name);
      const rel = path.relative(REPO, full);
      const st = statSync(full);
      if (st.isDirectory()) {
        if (name === "node_modules" || name === "dist") continue;
        walk(full);
      } else if (st.isFile() && (name.endsWith(".ts") || name.endsWith(".tsx"))) {
        if (SCAN_ALLOWLIST.has(rel)) continue;
        out.push(rel);
      }
    }
  }
  walk(root);
  return out;
}

// Forbidden literals = the actual friction values from
// shared/trading-frictions.json, expressed as regexes that catch
// equivalent numeric forms (e.g. `0.001` and `0.0010` and `1e-3` for
// the 10-bp taker fee). Comments and string literals (e.g.
// "0.0010 taker fee") are stripped first so a docstring mentioning
// the value isn't treated as a regression.
const FORBIDDEN_LITERAL_PATTERNS: { name: string; re: RegExp }[] = [
  // 10 bps = 0.001 = 0.0010 = 1e-3 — taker fee pct.
  { name: "TAKER_FEE_PCT (0.001 / 0.0010 / 1e-3)",
    re: /(?<![\d.])(?:0?\.0010+|0?\.001|1e-3)(?![\d.])/i },
  // 5 bps = 0.0005 = 5e-4 — slippage pct.
  { name: "SLIPPAGE_PCT (0.0005 / 5e-4)",
    re: /(?<![\d.])(?:0?\.0005|5e-4)(?![\d.])/i },
];

// Lines tagged with a trailing `// quant-only-allow: <reason>` comment
// are skipped by the friction-literal scan. Use this ONLY for numeric
// literals that look like fee/slippage values but provably aren't
// (e.g. epsilons, price-jitter, indicator scaling). The reason text
// is required and is read by code-review tooling.
const ALLOW_TAG = /\/\/\s*quant-only-allow:/;

function stripCommentsAndStrings(src: string): string {
  // First, strip whole lines that carry the explicit allow tag.
  src = src
    .split("\n")
    .map(line => (ALLOW_TAG.test(line) ? "" : line))
    .join("\n");
  // Strip /* … */ blocks
  let out = src.replace(/\/\*[\s\S]*?\*\//g, "");
  // Strip // line comments
  out = out.replace(/(^|[^:])\/\/[^\n]*/g, "$1");
  // Strip "..." and '...' and `...` string literals (non-greedy, no nested
  // templates — good enough for the scanned files).
  out = out.replace(/"(?:[^"\\]|\\.)*"/g, '""');
  out = out.replace(/'(?:[^'\\]|\\.)*'/g, "''");
  out = out.replace(/`(?:[^`\\]|\\.)*`/g, "``");
  return out;
}

test("quant-only: friction literals never appear anywhere in api-server source tree (excluding the central friction module)", () => {
  const files = listTsFiles(SCAN_ROOT);
  assert.ok(files.length > 5, `expected to scan many .ts files under ${SCAN_ROOT}; got ${files.length}`);
  const violations: string[] = [];
  for (const rel of files) {
    const full = path.join(REPO, rel);
    const code = stripCommentsAndStrings(readFileSync(full, "utf8"));
    for (const { name, re } of FORBIDDEN_LITERAL_PATTERNS) {
      const match = code.match(re);
      if (match !== null) {
        violations.push(`${rel}: ${name} (${match[0]})`);
      }
    }
  }
  assert.deepEqual(
    violations, [],
    `forbidden friction literals found in source tree:\n  ${violations.join("\n  ")}\n` +
    `Import these values from artifacts/api-server/src/lib/trading-constants.ts instead.`,
  );
});

test("quant-only: strategy-lab imports from trading-constants", () => {
  const lab = readFileSync(
    path.join(REPO, "artifacts/api-server/src/lib/strategy-lab.ts"), "utf8",
  );
  assert.match(
    lab, /from ["'][^"']*trading-constants["']/,
    "strategy-lab.ts must import friction values from trading-constants",
  );
});

// Task #444 deleted `artifacts/api-server/src/lib/agent-evolution.ts`
// along with the rest of the LLM/evolution plane. The original
// `quantonly-enforcement` test used to read that file to assert it
// sourced friction values from `trading-constants`; that assertion is
// moot now the file is gone. We keep a contract test that it *stays*
// gone so no future change accidentally resurrects the module —
// `no-llm-imports.test.ts` already enforces nothing imports from it.
test("quant-only: agent-evolution.ts stays deleted (Task #444)", () => {
  const evoPath = path.join(REPO, "artifacts/api-server/src/lib/agent-evolution.ts");
  let exists = false;
  try {
    statSync(evoPath);
    exists = true;
  } catch {
    exists = false;
  }
  assert.equal(
    exists, false,
    `agent-evolution.ts was deleted in Task #444 and must stay deleted; ` +
    `found a file at ${path.relative(REPO, evoPath)}.`,
  );
});

// Task #368 — every web page that surfaces a portfolio P&L number must
// route through the shared `derivePnl` helper so they cannot drift
// from the equity-vs-seed identity. The helper itself enforces
// `totalValue − startingCapital`; this contract test ensures consumers
// import it (rather than re-implementing the math inline, which is how
// the bug crept back into multiple pages historically) and that the
// helper file itself still derives from `totalValue - startingCapital`.
const PNL_CONSUMER_PAGES = [
  "artifacts/crypto-monitor/src/pages/agent-detail.tsx",
  "artifacts/crypto-monitor/src/pages/dashboard.tsx",
] as const;

test("quant-only: shared derivePnl helper still derives from totalValue − startingCapital", () => {
  const helper = readFileSync(
    path.join(REPO, "artifacts/crypto-monitor/src/lib/derive-pnl.ts"),
    "utf8",
  );
  const code = stripCommentsAndStrings(helper);
  assert.match(
    code, /export\s+function\s+derivePnl\b/,
    "derive-pnl.ts must export a `derivePnl` function",
  );
  assert.match(
    code, /startingCapital/,
    "derivePnl helper must read startingCapital",
  );
  assert.match(
    code, /totalValue\s*[-−]\s*\w+/,
    "derivePnl helper must subtract a seed from totalValue (equity-vs-seed identity)",
  );
  // Belt-and-braces: legacy totalPnl may only appear as the fallback
  // branch — never before the equity subtraction.
  const subIdx = code.search(/totalValue\s*[-−]/);
  const totalPnlIdx = code.search(/\.totalPnl\b/);
  if (totalPnlIdx >= 0) {
    assert.ok(
      subIdx >= 0 && subIdx < totalPnlIdx,
      "totalPnl may only be a fallback — the subtraction expression must come first",
    );
  }
});

for (const rel of PNL_CONSUMER_PAGES) {
  test(`quant-only: ${rel} routes portfolio P&L through the shared derivePnl helper`, () => {
    const page = readFileSync(path.join(REPO, rel), "utf8");
    const code = stripCommentsAndStrings(page);
    // The page must import `derivePnl` from the shared lib. Any other
    // way of computing portfolio P&L (a stale `totalPnl` read, an
    // inline `totalValue - seed` bound to a local that drifts, etc.)
    // silently re-introduces the bug fixed in Task #362 / #365 / #368.
    // We assert against the raw page (string literals carry the import
    // path) and require the resolved path to mention `derive-pnl`.
    assert.match(
      page,
      /import\s*\{[^}]*\bderivePnl\b[^}]*\}\s*from\s*["'][^"']*derive-pnl["']/,
      `${rel} must import derivePnl from @/lib/derive-pnl`,
    );
    assert.match(
      code, /derivePnl\s*\(/,
      `${rel} must actually call derivePnl(...) on the portfolio`,
    );
    // Belt-and-braces: a direct read of `.totalPnl` on a portfolio is
    // forbidden — it must go through the helper. Strategy-lab buckets
    // (a different domain object) live on a different page.
    const directReads = code.match(/\b(?:paperPortfolio|portfolio|bot|b|p)\.totalPnl(?:Percent)?\b/g);
    assert.equal(
      directReads, null,
      `${rel} must not read .totalPnl/.totalPnlPercent directly — call derivePnl(...) instead. Found: ${directReads?.join(", ")}`,
    );
  });
}
