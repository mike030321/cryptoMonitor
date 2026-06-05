import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");

const routes = readFileSync(
  path.join(REPO, "artifacts/api-server/src/routes/crypto/index.ts"),
  "utf8",
);

function routeBody(source: string, pathLiteral: string): string {
  const start = source.indexOf(`router.get("${pathLiteral}"`);
  assert.ok(start >= 0, `${pathLiteral} route not found`);
  const open = source.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < source.length; i++) {
    if (source[i] === "{") depth++;
    if (source[i] === "}") depth--;
    if (depth === 0) return source.slice(open, i + 1);
  }
  throw new Error(`${pathLiteral} route body was not balanced`);
}

test("/crypto/reality-check is scoped to live executor ai-bots only", () => {
  const body = routeBody(routes, "/crypto/reality-check");
  assert.match(body, /eq\s*\(\s*agentsTable\.strategyType\s*,\s*["']ai-bots["']\s*\)/);
  assert.match(body, /eq\s*\(\s*agentsTable\.isActive\s*,\s*true\s*\)/);
  assert.match(body, /isNull\s*\(\s*agentsTable\.archivedAt\s*\)/);
  assert.match(body, /agentsTable\.profileId[\s\S]*IS DISTINCT FROM\s*['"]legacy_archived['"]/);
  assert.match(
    body,
    /const\s+tradesAll\s*=\s*allTrades\.filter\s*\(\s*\(?t\)?\s*=>\s*aiAgentIds\.has\s*\(\s*t\.agentId\s*\)\s*\)/,
    "trade/prior-fallback counts must be live-fleet scoped",
  );
  assert.match(
    body,
    /const\s+openPositions\s*=\s*allOpenPositions\.filter\s*\(\s*\(?p\)?\s*=>\s*aiAgentIds\.has\s*\(\s*p\.agentId\s*\)\s*\)/,
    "open-position P&L must be live-fleet scoped",
  );
});
