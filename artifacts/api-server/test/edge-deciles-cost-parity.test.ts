// Task #343 — Step 5 parity test.
//
// The /crypto/meta/edge-deciles handler used to declare a local
// `ROUND_TRIP_COST_PCT_LOCAL = 0.3` literal that visually mirrored
// trading-constants.ts but had no enforcement: any change in
// shared/trading-frictions.json (e.g. swapping the maker/taker schedule
// after an exchange tier change) would silently drift the realized-edge
// computation away from the live brain's cost model. We replaced the
// literal with `ROUND_TRIP_COST_PERCENT` from trading-constants — this
// test pins both the value and the *units* so a future "let's switch
// to the fractional ROUND_TRIP_COST_PCT" mistake is caught.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  ROUND_TRIP_COST_PCT,
  ROUND_TRIP_COST_PERCENT,
} from "../src/lib/trading-constants";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROUTE = path.resolve(HERE, "../src/routes/crypto/index.ts");

describe("edge-deciles cost-parity (task #343)", () => {
  it("pins ROUND_TRIP_COST_PERCENT == 100 * ROUND_TRIP_COST_PCT", () => {
    assert.ok(
      Math.abs(ROUND_TRIP_COST_PERCENT - 100 * ROUND_TRIP_COST_PCT) < 1e-9,
      `expected ROUND_TRIP_COST_PERCENT to be 100x the fractional ` +
        `value; got ${ROUND_TRIP_COST_PERCENT} vs 100*${ROUND_TRIP_COST_PCT}`,
    );
  });

  it("matches the legacy 0.30% literal at current friction tier", () => {
    // Sanity check that swapping to the shared constant did not silently
    // change the cost model. If shared/trading-frictions.json is
    // re-tuned, update this assertion in the same PR.
    assert.ok(
      Math.abs(ROUND_TRIP_COST_PERCENT - 0.3) < 1e-6,
      `ROUND_TRIP_COST_PERCENT drifted from the 0.30% baseline ` +
        `(now ${ROUND_TRIP_COST_PERCENT}). If this is intentional, ` +
        `update the test; otherwise audit shared/trading-frictions.json.`,
    );
  });

  it("edge-deciles handler uses the shared constant, not a local literal", () => {
    const src = readFileSync(ROUTE, "utf-8");
    // Window the assertion to the edge-deciles handler so we don't
    // false-positive on legitimate `0.3` literals elsewhere in the file.
    const start = src.indexOf("/crypto/meta/edge-deciles");
    assert.ok(start >= 0, "edge-deciles handler not found in route file");
    const slice = src.slice(start, start + 2500);

    assert.ok(
      slice.includes("ROUND_TRIP_COST_PERCENT"),
      "edge-deciles handler must subtract ROUND_TRIP_COST_PERCENT",
    );
    assert.ok(
      !slice.includes("ROUND_TRIP_COST_PCT_LOCAL"),
      "edge-deciles handler still references the deleted local " +
        "ROUND_TRIP_COST_PCT_LOCAL — task #343 forbids this re-introduction",
    );
    // Forbid the easy regression: subtracting the FRACTIONAL constant
    // would silently scale costs down by 100x.
    assert.ok(
      !/signed\s*-\s*ROUND_TRIP_COST_PCT\b/.test(slice),
      "edge-deciles must not subtract the fractional ROUND_TRIP_COST_PCT " +
        "(units mismatch — `signed` is in percent)",
    );
  });
});
