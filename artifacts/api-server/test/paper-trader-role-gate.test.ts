import { test } from "node:test";
import assert from "node:assert/strict";

// Task #550 — paper-trader role-gate code-shape regression.
//
// The runtime test for paper-trader is hard to mount in isolation
// (it imports the full DB layer), so this test enforces that the
// gate WIRING in src/lib/paper-trader.ts matches the design:
//
//   1. The role helper is imported from "./timeframe-roles".
//   2. After the static TRADEABLE_TIMEFRAMES check, the trade-execution
//      path computes the role for the request's timeframe.
//   3. If the role is not 'trade', the function returns and emits a
//      structured warn log including {timeframe, role,
//      reason: 'trade_blocked_by_role'} so production logs are greppable.
//   4. The gate sits in the same code path as the existing
//      TRADEABLE_TIMEFRAMES guard (i.e. it cannot be bypassed by
//      slipping a TF that's tradeable but role!='trade').
//
// Locking the source shape is intentional: a future refactor that
// removes the gate (or moves it to a place where a `return` cannot
// reach) will break this test even if the integration suite does
// not catch it.

import { readFile } from "node:fs/promises";

test("paper-trader: imports getRoleForTimeframe from timeframe-roles", async () => {
  const src = await readFile(
    new URL("../src/lib/paper-trader.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /import\s*\{\s*getRoleForTimeframe\s*\}\s*from\s*["']\.\/timeframe-roles["']/,
    "paper-trader must import getRoleForTimeframe from ./timeframe-roles",
  );
});

test("paper-trader: trade-blocked-by-role gate sits AFTER the TRADEABLE_TIMEFRAMES universe check, BEFORE the rest of execution", async () => {
  const src = await readFile(
    new URL("../src/lib/paper-trader.ts", import.meta.url),
    "utf8",
  );

  // Find the universe-check anchor and the role-gate anchor.
  const universeIdx = src.indexOf("if (!TRADEABLE_TIMEFRAMES.has(timeframe)) return;");
  assert.ok(universeIdx > 0, "expected the static TRADEABLE_TIMEFRAMES guard");

  const roleCallIdx = src.indexOf("getRoleForTimeframe(timeframe)");
  assert.ok(roleCallIdx > 0, "expected getRoleForTimeframe(timeframe) call");

  assert.ok(
    roleCallIdx > universeIdx,
    "role gate must run AFTER the static universe check (universe is the first filter)",
  );

  // The role gate must short-circuit (return) when role !== 'trade'.
  const between = src.slice(universeIdx, roleCallIdx + 4000);
  assert.match(
    between,
    /tfRole\s*!==\s*["']trade["']/,
    "role gate must explicitly compare against 'trade'",
  );
  // And emit a structured warn log with reason: 'trade_blocked_by_role'.
  assert.match(
    between,
    /reason:\s*["']trade_blocked_by_role["']/,
    "role-gate warn log must include reason: 'trade_blocked_by_role' for production filtering",
  );
  assert.match(
    between,
    /logger\.warn\s*\(/,
    "role gate must emit a structured warn log",
  );
  // Must short-circuit with `return;`.
  const guardBlock = between.slice(0, between.indexOf("getRoleForTimeframe(timeframe)") + 200);
  assert.match(
    src.slice(roleCallIdx, roleCallIdx + 800),
    /return;/,
    "role gate must short-circuit with `return;` when role !== 'trade'",
  );
  // Sanity: the captured guard block exists.
  assert.ok(guardBlock.length > 0);
});
