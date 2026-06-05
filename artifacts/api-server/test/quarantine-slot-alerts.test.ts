import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  recomputeSlotStarvationAlerts,
  getActiveSlotStarvationAlerts,
  POOLED_COIN_IDS,
} from "../src/lib/quarantine-slot-alerts.ts";

describe("quarantine slot starvation alerts", () => {
  it("POOLED_COIN_IDS recognises both __pooled__ and *", () => {
    assert.equal(POOLED_COIN_IDS.has("__pooled__"), true);
    assert.equal(POOLED_COIN_IDS.has("*"), true);
    assert.equal(POOLED_COIN_IDS.has("BTC"), false);
  });

  it("recomputeSlotStarvationAlerts returns an array and updates the cache", async () => {
    const alerts = await recomputeSlotStarvationAlerts();
    assert.ok(Array.isArray(alerts));
    for (const a of alerts) {
      assert.equal(typeof a.coinId, "string");
      assert.equal(typeof a.timeframe, "string");
      // Pooled rows are never the subject of an alert — only per-coin
      // slots starve (the pooled fallback is what they fall back to).
      assert.equal(POOLED_COIN_IDS.has(a.coinId), false);
      assert.ok(a.perCoinQuarantinedCount >= 1);
      assert.equal(typeof a.detectedAt, "string");
      assert.ok(
        a.lastQuarantineReason === null
          || typeof a.lastQuarantineReason === "string",
      );
    }
    const cached = getActiveSlotStarvationAlerts();
    assert.equal(cached.length, alerts.length);
  });
});
