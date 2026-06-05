/**
 * Task #405 / B-AVG-HIDES — regression test for the basket-coverage gate
 * inside `basketAvgReturnFromCandles`. Previously the gate used
 * `Math.floor(prices.length / 2)`, which let an odd basket of 3 pass
 * with only 1 candle-backed coin. The remediation switched to
 * `Math.ceil(...)` so an odd basket of 3 demands 2.
 *
 * We don't need a DB to test the threshold expression — the math is the
 * gate. This is a pure-arithmetic guard test that fails loudly if anyone
 * reverts the Math.ceil() back to Math.floor().
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

test("basketAvgReturnFromCandles uses Math.ceil(...) for the >=half-coverage gate", () => {
  const src = readFileSync(
    new URL("../src/lib/strategy-lab.ts", import.meta.url),
    "utf8",
  );
  const m = src.match(
    /if\s*\(\s*n\s*<\s*Math\.max\s*\(\s*1\s*,\s*Math\.(floor|ceil)\s*\(\s*prices\.length\s*\/\s*2\s*\)\s*\)\s*\)\s*return\s+null;/,
  );
  assert.ok(m, "basket-coverage gate not found in strategy-lab.ts");
  assert.equal(m[1], "ceil", "basket-coverage gate must use Math.ceil — Math.floor lets 1-of-3 pass");
});

test("ceil-based gate semantics: odd 3 → demands 2, even 4 → demands 2, 1 → demands 1", () => {
  const requires = (n: number) => Math.max(1, Math.ceil(n / 2));
  assert.equal(requires(0), 1);
  assert.equal(requires(1), 1);
  assert.equal(requires(2), 1);
  assert.equal(requires(3), 2);   // ← the bug case: floor said 1
  assert.equal(requires(4), 2);
  assert.equal(requires(5), 3);
  assert.equal(requires(7), 4);
});
