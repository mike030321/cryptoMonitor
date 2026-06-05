import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { db } from "@workspace/db";
import { recordSkip, getSkipPersistHealth } from "../src/lib/skip-tracker";

// Verify that a failed skip-event DB write surfaces on the health endpoint
// (`getSkipPersistHealth`) instead of being silently swallowed. Trade-loop
// callers intentionally do not await `recordSkip`, so the only signal an
// operator gets when persistence breaks is the module-level counter and
// last-error fields exposed via `/api/crypto/skip-tracker-health`.
//
// Strategy: monkey-patch `db.insert` to throw, call `recordSkip`, then
// confirm the counters moved. We restore the original insert in a finally
// block so this test cannot leak state into other tests in the suite.

describe("skip-tracker health surface", () => {
  it("a failed DB write increments the failure counter and records the error message", async () => {
    const originalInsert = db.insert.bind(db);
    const insertError = new Error("forced db failure for skip-tracker test");
    (db as unknown as { insert: () => never }).insert = () => {
      throw insertError;
    };

    try {
      const before = getSkipPersistHealth();

      await recordSkip(
        "confidence_below_threshold",
        "test-agent",
        "forced failure test",
        { foo: "bar" },
      );

      const after = getSkipPersistHealth();
      assert.equal(
        after.failures,
        before.failures + 1,
        "failure counter must increment when db.insert throws",
      );
      assert.equal(
        after.successes,
        before.successes,
        "success counter must not move when the write fails",
      );
      assert.match(
        after.lastError ?? "",
        /forced db failure/,
        "lastError must capture the underlying DB error message",
      );
      assert.ok(
        after.lastErrorAt !== null,
        "lastErrorAt must be populated after a failed write",
      );
    } finally {
      (db as unknown as { insert: typeof originalInsert }).insert = originalInsert;
    }
  });

  it("recordSkip resolves (does not reject) so fire-and-forget callers stay safe", async () => {
    const originalInsert = db.insert.bind(db);
    (db as unknown as { insert: () => never }).insert = () => {
      throw new Error("second forced failure");
    };
    try {
      await assert.doesNotReject(
        recordSkip("daily_loss_limit", "test-agent-2", "second failure"),
        "recordSkip must swallow the underlying error so void-callers don't crash",
      );
    } finally {
      (db as unknown as { insert: typeof originalInsert }).insert = originalInsert;
    }
  });
});
