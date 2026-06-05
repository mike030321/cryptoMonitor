/**
 * Task #381 — trade→tick binding restart survivability.
 *
 * Verifies that:
 *  - bindTradeToTick persists to disk
 *  - hydrateBindings restores the map after a fresh process start
 *  - peekTickForTrade returns the restored binding
 */
import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { promises as fsp } from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

let tmpRoot: string;

beforeEach(async () => {
  tmpRoot = await fsp.mkdtemp(path.join(os.tmpdir(), "mb-hydrate-"));
  process.env.META_BRAIN_STATE_DIR = tmpRoot;
});

afterEach(async () => {
  delete process.env.META_BRAIN_STATE_DIR;
  await fsp.rm(tmpRoot, { recursive: true, force: true });
});

describe("meta-brain trade→tick hydration (Task #381)", () => {
  it("hydrates persisted bindings on boot", async () => {
    // Re-import the adapter so it picks up the test's
    // META_BRAIN_STATE_DIR (the path is captured at module-eval time).
    const adapter = await import(
      `../src/lib/meta-brain/adapter?case=hydrate-roundtrip-${Date.now()}`
    );
    adapter.__resetAdapterState();
    adapter.bindTradeToTick(42, "tick-uuid-restored");
    adapter.bindTradeToTick(43, "shadow:tick-uuid-shadow");
    // Allow the coalesced single-flight write to drain.
    await new Promise((r) => setTimeout(r, 100));

    // Snapshot present?
    const onDisk = await fsp.readFile(
      path.join(tmpRoot, "trade_to_tick.json"),
      "utf-8",
    );
    assert.match(onDisk, /tick-uuid-restored/);

    // Simulate a process restart by clearing the in-memory map and
    // re-running hydrate.
    adapter.__resetAdapterState();
    assert.equal(
      adapter.peekTickForTrade(42),
      undefined,
      "must be empty before hydrate",
    );

    await adapter.hydrateBindings();
    assert.equal(adapter.peekTickForTrade(42), "tick-uuid-restored");
    assert.equal(adapter.peekTickForTrade(43), "shadow:tick-uuid-shadow");
  });

  it("hydrate is a no-op when no snapshot exists", async () => {
    const adapter = await import(
      `../src/lib/meta-brain/adapter?case=hydrate-noop-${Date.now()}`
    );
    adapter.__resetAdapterState();
    await adapter.hydrateBindings();
    assert.equal(adapter.peekTickForTrade(1), undefined);
  });
});
