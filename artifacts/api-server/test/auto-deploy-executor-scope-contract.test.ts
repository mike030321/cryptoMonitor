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
  assert.ok(open >= 0, `${name} body not found`);
  let depth = 0;
  for (let i = open; i < source.length; i++) {
    const ch = source[i];
    if (ch === "{") depth++;
    if (ch === "}") depth--;
    if (depth === 0) return source.slice(open, i + 1);
  }
  throw new Error(`${name} body was not balanced`);
}

test("auto-deploy opener uses the same live-executor scope as attribution", () => {
  const body = functionBody(paperTrader, "autoDeployIdleCash");

  assert.match(
    body,
    /eq\s*\(\s*agentsTable\.strategyType\s*,\s*["']ai-bots["']\s*\)/,
    "auto-deploy must target ai-bots only",
  );
  assert.match(
    body,
    /eq\s*\(\s*agentsTable\.isActive\s*,\s*true\s*\)/,
    "auto-deploy must exclude paused/inactive agents",
  );
  assert.match(
    body,
    /isNull\s*\(\s*agentsTable\.archivedAt\s*\)/,
    "auto-deploy must exclude archived agents",
  );
  assert.match(
    body,
    /agentsTable\.profileId[\s\S]*IS DISTINCT FROM\s*['"]legacy_archived['"]/,
    "auto-deploy must exclude legacy_archived rows",
  );
  assert.match(
    body,
    /new\s+Set\s*\(\s*liveAiBotAgents\.map[\s\S]*liveAiBotIds\.has\s*\(\s*portfolio\.agentId\s*\)/,
    "auto-deploy must gate portfolios through the live agent id set",
  );
});
