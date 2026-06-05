import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  runQuarantineSweep,
  listRecentQuarantineEvents,
} from "../src/lib/auto-quarantine.ts";

describe("auto-quarantine", () => {
  it("listRecentQuarantineEvents returns an array", async () => {
    const events = await listRecentQuarantineEvents(5);
    assert.ok(Array.isArray(events));
    for (const e of events) {
      assert.equal(typeof e.id, "number");
      assert.equal(typeof e.reason, "string");
      assert.ok(e.triggeredAt instanceof Date || typeof e.triggeredAt === "string");
    }
  });

  it("runQuarantineSweep dry-run produces a decision summary without throwing", async () => {
    const summary = await runQuarantineSweep({ dryRun: true });
    assert.equal(summary.dryRun, true);
    assert.equal(typeof summary.windowHours, "number");
    assert.equal(typeof summary.generatedAt, "string");
    assert.ok(Array.isArray(summary.decisions));
    for (const d of summary.decisions) {
      assert.ok(["kept", "quarantined"].includes(d.decision));
      assert.equal(typeof d.registryId, "number");
      assert.ok(Array.isArray(d.reasonCodes));
    }
  });
});
