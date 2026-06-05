import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { judgeDirection } from "../src/lib/trading-constants";

// ── Source-level parity guard ──────────────────────────────────────────
// The Phase 4 cutover gate is only valid if the LIVE LLM resolver, the
// SHADOW resolver, and the metrics aggregator all grade outcomes through
// the SAME judgeDirection helper. If a future change re-introduces an
// inline neutral-zone calculation in any of these three call sites, the
// shadow-vs-LLM lift number becomes meaningless. This test scans the
// source files and refuses to pass if any of them re-derive direction
// from the raw price-change sign or hand-roll a neutral-zone band.

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

function readSrc(rel: string): string {
  return readFileSync(path.join(root, rel), "utf8");
}

const FORBIDDEN_PATTERNS: RegExp[] = [
  /Math\.abs\([^)]*\)\s*<\s*0\.001/,        // raw-sign realized-class hack
  /change\s*>\s*0\s*\?\s*"up"\s*:\s*"down"/, // inline direction inference
  /neutralMultiplier\s*=/,                   // hand-rolled neutral band
];

test("live resolver, shadow resolver and metrics all route through judgeDirection", () => {
  const files = [
    "src/lib/monitor.ts",
    "src/lib/shadow-recorder.ts",
    "src/routes/crypto/index.ts",
  ];
  for (const f of files) {
    const src = readSrc(f);
    assert.ok(
      src.includes("judgeDirection"),
      `${f} must call judgeDirection — found no reference`,
    );
    for (const re of FORBIDDEN_PATTERNS) {
      assert.equal(
        re.test(src),
        false,
        `${f} contains a hand-rolled adjudication that bypasses judgeDirection: ${re}`,
      );
    }
  }
});

test("judgeDirection is deterministic and pure across timeframes/directions", () => {
  const tfs = ["5m", "1h", "2h", "6h", "1d"];
  const dirs = ["up", "down", "stable"] as const;
  const changes = [-3, -1.5, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 1.5, 3];
  for (const tf of tfs) {
    for (const dir of dirs) {
      for (const ch of changes) {
        const a = judgeDirection(dir, ch, tf);
        const b = judgeDirection(dir, ch, tf);
        assert.deepEqual(a, b, `${tf}/${dir}/${ch}`);
        // Mutually exclusive: never both correct AND neutral.
        assert.ok(!(a.correct && a.neutral), `${tf}/${dir}/${ch} both flags set`);
      }
    }
  }
});

test("neutral band is wider on 1h+ than on 5m for the same direction", () => {
  // Pick a tiny positive change. If 5m calls it neutral, 1h must too
  // (1h's band is 1.2× wider, so it's a strict superset of 5m's neutral
  // zone for non-correct outcomes).
  const a = judgeDirection("up", 0.05, "5m");
  const b = judgeDirection("up", 0.05, "1h");
  if (a.neutral) assert.equal(b.neutral, true);
});

test("stable direction is correct only when |change| < threshold", () => {
  assert.equal(judgeDirection("stable", 0.0, "5m").correct, true);
  assert.equal(judgeDirection("stable", 5.0, "5m").correct, false);
});
