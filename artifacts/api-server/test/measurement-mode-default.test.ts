import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");

const measurementMode = readFileSync(
  path.join(REPO, "artifacts/api-server/src/lib/measurement-mode.ts"),
  "utf8",
);
const monitor = readFileSync(
  path.join(REPO, "artifacts/api-server/src/lib/monitor.ts"),
  "utf8",
);

test("measurement mode: env overrides resolve explicitly, default is ON, manual ON expires after 24h unless pinned", () => {
  assert.match(
    measurementMode,
    /MEASUREMENT_MODE\s*===\s*["']true["'][\s\S]*return\s*\{\s*enabled:\s*true,\s*source:\s*["']env["']\s*\}/,
    "MEASUREMENT_MODE=true must explicitly enable observation mode",
  );
  assert.match(
    measurementMode,
    /MEASUREMENT_MODE\s*===\s*["']false["'][\s\S]*return\s*\{\s*enabled:\s*false,\s*source:\s*["']env["']\s*\}/,
    "MEASUREMENT_MODE=false must explicitly disable observation mode",
  );
  assert.match(
    measurementMode,
    /return\s*\{\s*enabled:\s*true,\s*source:\s*["']default["']\s*\}/,
    "default state must be measurement ON (quant-only is the architecturally correct default)",
  );
  assert.match(
    measurementMode,
    /const\s+MANUAL_ON_TTL_MS\s*=\s*24\s*\*\s*60\s*\*\s*60\s*\*\s*1000/,
    "manual measurement-mode ON rows must have a 24h expiry",
  );
  assert.match(
    measurementMode,
    /value\.pinned\s*===\s*true/,
    "operators need an explicit pinned escape hatch for long measurement windows",
  );
  assert.match(
    measurementMode,
    /nowMs\s*-\s*updatedAtMs\s*>\s*MANUAL_ON_TTL_MS[\s\S]*envDefault\(\)/,
    "stale manual ON rows must collapse back to the env-default state",
  );
  assert.match(
    monitor,
    /if\s*\(\s*!MEASUREMENT_MODE\s*\)\s*\{[\s\S]*autoDeployIdleCash\s*\(\s*prices\s*\)/,
    "monitor must call autoDeployIdleCash when measurement mode is off",
  );
});
