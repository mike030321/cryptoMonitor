import { test } from "node:test";
import assert from "node:assert/strict";

// Task #550 — meta-brain adapter must pass `slice_role` through on
// every collected slice payload. Resolved at collection time so the
// Python side does NOT have to re-load the JSON; this is intentional
// pass-through (consumption is a separate downstream task).
//
// We exercise this at the source level instead of via collectSlice()
// directly because collectSlice() pushes onto an internal buffer and
// emits over HTTP — the in-process inspection of the buffered slice
// is private to the module. The source-shape test locks the field
// onto the contract type AND the assignment site in the adapter.

import { readFile } from "node:fs/promises";
import { SLICE_ROLES } from "../src/lib/meta-brain/contract";

test("contract.ts: SliceRole enum mirrors the timeframe-roles 4-role set", () => {
  // Lock the API surface — adding a 5th role must be a deliberate
  // edit to BOTH the loader enum and this contract enum (and any
  // downstream Python schema).
  assert.deepEqual([...SLICE_ROLES].sort(), ["context", "disabled", "shadow", "trade"]);
});

test("contract.ts: MetaBrainSlice declares a non-optional slice_role field", async () => {
  const src = await readFile(
    new URL("../src/lib/meta-brain/contract.ts", import.meta.url),
    "utf8",
  );
  // Non-optional: no `slice_role?:` form. We grep the interface body.
  const ifaceStart = src.indexOf("export interface MetaBrainSlice");
  assert.ok(ifaceStart > 0, "expected MetaBrainSlice interface");
  const ifaceEnd = src.indexOf("}", ifaceStart);
  const body = src.slice(ifaceStart, ifaceEnd);
  assert.match(body, /\bslice_role:\s*SliceRole\b/, "MetaBrainSlice must declare slice_role: SliceRole");
  assert.doesNotMatch(body, /slice_role\?:/, "slice_role must be REQUIRED on the wire payload");
});

test("adapter.ts: collectSlice resolves slice_role via getRoleForTimeframe", async () => {
  const src = await readFile(
    new URL("../src/lib/meta-brain/adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /import\s*\{\s*getRoleForTimeframe\s*\}\s*from\s*["']\.\.\/timeframe-roles["']/,
    "adapter must import getRoleForTimeframe from ../timeframe-roles",
  );
  // The slice_role field on the constructed slice must be filled by
  // calling getRoleForTimeframe on the slice's timeframe — not a
  // hard-coded default.
  assert.match(
    src,
    /slice_role:\s*getRoleForTimeframe\s*\(\s*args\.timeframe\s*\)/,
    "collectSlice must populate slice_role from getRoleForTimeframe(args.timeframe)",
  );
});
