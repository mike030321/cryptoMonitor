import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { PREDICTION_FLEET_RESET_AT } from "../src/lib/trading-constants";

const __dirname = dirname(fileURLToPath(import.meta.url));

describe("PREDICTION_FLEET_RESET_AT cutoff", () => {
  it("is the documented post-reset moment (2026-04-21T00:00:00Z)", () => {
    assert.equal(PREDICTION_FLEET_RESET_AT.toISOString(), "2026-04-21T00:00:00.000Z");
  });

  it("is in the past so the calibration filter actually excludes legacy rows", () => {
    assert.ok(
      PREDICTION_FLEET_RESET_AT.getTime() < Date.now(),
      "cutoff must be in the past or the calibrator would filter out everything",
    );
  });
});

describe("confidence calibrator queries are scoped to the cutoff", () => {
  // Regression guard: both calibration-input queries MUST gate on
  // predictions.createdAt >= PREDICTION_FLEET_RESET_AT. Removing either
  // filter would re-admit pre-reset poisoned outcomes into priors. We
  // assert via static text rather than a DB round-trip so the test stays
  // hermetic and fast.
  const source = readFileSync(
    join(__dirname, "..", "src", "lib", "confidence-calibrator.ts"),
    "utf8",
  );

  it("imports the cutoff and the gte helper", () => {
    assert.match(source, /PREDICTION_FLEET_RESET_AT/);
    assert.match(source, /\bgte\b/);
  });

  it("getAgentCalibration filters by PREDICTION_FLEET_RESET_AT", () => {
    const fnStart = source.indexOf("export async function getAgentCalibration");
    const fnEnd = source.indexOf("export ", fnStart + 1);
    assert.ok(fnStart >= 0, "getAgentCalibration must exist");
    const body = source.slice(fnStart, fnEnd);
    assert.match(
      body,
      /gte\(\s*predictionsTable\.createdAt\s*,\s*PREDICTION_FLEET_RESET_AT\s*\)/,
    );
  });

  it("getModelAccuracyForCoin filters by PREDICTION_FLEET_RESET_AT", () => {
    const fnStart = source.indexOf("export async function getModelAccuracyForCoin");
    const fnEnd = source.indexOf("export ", fnStart + 1);
    assert.ok(fnStart >= 0, "getModelAccuracyForCoin must exist");
    const body = source.slice(fnStart, fnEnd);
    assert.match(
      body,
      /gte\(\s*predictionsTable\.createdAt\s*,\s*PREDICTION_FLEET_RESET_AT\s*\)/,
    );
  });
});
