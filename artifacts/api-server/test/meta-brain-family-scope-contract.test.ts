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
  const start = source.indexOf(`router.get(\n  "${pathLiteral}"`);
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

test("family coin drill-down is scoped to live non-legacy executor agents", () => {
  const body = routeBody(routes, "/crypto/agents/families/:profileId/coins");
  assert.match(body, /eq\s*\(\s*agentsTable\.profileId\s*,\s*profileId\s*\)/);
  assert.match(body, /eq\s*\(\s*agentsTable\.isActive\s*,\s*true\s*\)/);
  assert.match(body, /isNull\s*\(\s*agentsTable\.archivedAt\s*\)/);
  assert.match(
    body,
    /agentsTable\.profileId[\s\S]*IS DISTINCT FROM\s*'legacy_archived'/,
    "archived legacy agents must not leak into family per-coin rows",
  );
});
