import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");

const monitor = readFileSync(
  path.join(REPO, "artifacts/api-server/src/lib/monitor.ts"),
  "utf8",
);

test("monitor cycle iterates only live executor ai-bots", () => {
  assert.match(
    monitor,
    /eq\s*\(\s*agentsTable\.isActive\s*,\s*true\s*\)[\s\S]*eq\s*\(\s*agentsTable\.strategyType\s*,\s*"ai-bots"\s*\)[\s\S]*isNull\s*\(\s*agentsTable\.archivedAt\s*\)[\s\S]*agentsTable\.profileId[\s\S]*IS DISTINCT FROM\s*'legacy_archived'/,
    "monitor must not generate predictions/trades for archived legacy ai-bots",
  );
  assert.match(
    monitor,
    /const\s+liveExecutorAgentIds\s*=\s*new\s+Set\s*\(\s*agents\.map\s*\(\s*\(a\)\s*=>\s*a\.id\s*\)\s*\)/,
    "the live executor roster must be captured once per cycle",
  );
});

test("monitor fleet telemetry excludes archived and legacy portfolios", () => {
  assert.match(
    monitor,
    /const\s+openPositionsForExposure\s*=\s*openPositionsAll\.filter\s*\(\s*\(p\)\s*=>[\s\S]*liveExecutorAgentIds\.has\s*\(\s*p\.agentId\s*\)/,
    "exposure telemetry must be scoped to live executor positions",
  );
  assert.match(
    monitor,
    /const\s+fleetPortfolios\s*=\s*fleetPortfoliosAll\.filter\s*\(\s*\(p\)\s*=>[\s\S]*liveExecutorAgentIds\.has\s*\(\s*p\.agentId\s*\)/,
    "fleet equity telemetry must be scoped to live executor portfolios",
  );
  assert.match(
    monitor,
    /const\s+portfolios\s*=\s*portfoliosAll\.filter\s*\(\s*\(p\)\s*=>[\s\S]*liveExecutorAgentIds\.has\s*\(\s*p\.agentId\s*\)/,
    "meta-brain portfolio telemetry must be scoped to live executors",
  );
  assert.match(
    monitor,
    /const\s+openPositions\s*=\s*openPositionsAll\.filter\s*\(\s*\(p\)\s*=>[\s\S]*liveExecutorAgentIds\.has\s*\(\s*p\.agentId\s*\)/,
    "meta-brain exposure telemetry must be scoped to live executor positions",
  );
});
