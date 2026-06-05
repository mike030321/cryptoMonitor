import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");

const paperTrader = readFileSync(
  path.join(REPO, "artifacts/api-server/src/lib/paper-trader.ts"),
  "utf8",
);

function functionBody(source: string, name: string): string {
  const start = source.indexOf(`export async function ${name}`);
  assert.ok(start >= 0, `${name} not found`);
  const open = source.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < source.length; i++) {
    if (source[i] === "{") depth++;
    if (source[i] === "}") depth--;
    if (depth === 0) return source.slice(open, i + 1);
  }
  throw new Error(`${name} body was not balanced`);
}

test("auto-deploy opens at fee/slippage-adjusted entry price", () => {
  const body = functionBody(paperTrader, "autoDeployIdleCash");
  assert.match(
    body,
    /const\s+adjustedEntryPrice\s*=\s*applyEntrySlippage\s*\(\s*coin\.currentPrice\s*,\s*direction\s*\)/,
    "auto-deploy must pay the same entry slippage as regular paper trades",
  );
  assert.match(
    body,
    /const\s+quantity\s*=\s*allocate\s*\/\s*adjustedEntryPrice/,
    "quantity must be derived from the adjusted entry price",
  );
  assert.match(
    body,
    /entryPrice:\s*adjustedEntryPrice/g,
    "trade and position rows must store adjusted entry price",
  );
  assert.match(
    body,
    /stopLossPrice:\s*adjustedEntryPrice\s*\*\s*stopLossMult/,
    "stop loss must be anchored to adjusted entry price",
  );
  assert.match(
    body,
    /takeProfitPrice:\s*adjustedEntryPrice\s*\*\s*takeProfitMult/,
    "take profit must be anchored to adjusted entry price",
  );
});
