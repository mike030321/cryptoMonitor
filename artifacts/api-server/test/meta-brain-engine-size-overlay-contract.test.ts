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

test("quant /ml/decide override preserves meta-brain and profile size overlays", () => {
  const body = functionBody(paperTrader, "executePaperTrade");
  assert.match(
    body,
    /const\s+executionSizeMultiplier\s*=\s*clampMetaSizeMultiplier\s*\(\s*compositeSizeMult\s*\)/,
    "meta-brain/profile/pooled-fallback multiplier must be captured once",
  );
  assert.match(
    body,
    /positionSize\s*=\s*engineDecision\.positionSizeUsd[\s\S]*positionSize\s*\*=\s*executionSizeMultiplier/,
    "the /ml/decide base size must still receive the api-server supervisory multiplier",
  );
  assert.match(
    body,
    /positionSize\s*=\s*Math\.min\s*\([\s\S]*p\.cashBalance\s*\*\s*MAX_CASH_PER_POSITION_PCT[\s\S]*p\.totalValue\s*\*\s*_maxPositionPct/,
    "post-engine size must still respect cash and MTTM/global max-position caps",
  );
  assert.match(
    body,
    /engineDecision\.positionSizeUsd[\s\S]*MAX_PORTFOLIO_AT_RISK[\s\S]*entryFee\s*=\s*positionSize\s*\*\s*TAKER_FEE_PCT\s*\/\s*Math\.max/,
    "post-engine size must re-apply fleet risk cap and recover entry fee from final post-fee notional",
  );
});
