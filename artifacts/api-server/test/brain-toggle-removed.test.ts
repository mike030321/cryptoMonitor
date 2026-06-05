/**
 * Task #255 — assert the legacy fleet brain-flip route is gone.
 *
 * The route still exists as an HTTP endpoint (so we can return a clear
 * 410 / `replacement` pointer to clients hitting the old URL), but it
 * MUST NOT call `setBrainState` and MUST return 410. We assert the route
 * file no longer imports `setBrainState` and the route body returns 410.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const routeSrc = readFileSync(resolve(here, "../src/routes/crypto/index.ts"), "utf8");

test("Task #255: /crypto/brain/toggle no longer flips the fleet", () => {
  // The legacy toggle block must not call setBrainState — calling it would
  // re-introduce the LLM-execution toggle. Task #406 added a separate
  // admin-gated `/crypto/brain/state` POST that legitimately calls
  // setBrainState to control the `quant_brain_enabled` kill-switch, so
  // this assertion is now scoped to the body of the toggle handler only.
  const block = routeSrc.split('router.post("/crypto/brain/toggle"')[1] ?? "";
  const toggleHandler = block.split("router.")[0];
  assert.ok(
    !/setBrainState\s*\(/.test(toggleHandler),
    "setBrainState must not be called from the legacy /crypto/brain/toggle handler (Task #255)",
  );
});

test("Task #255: /crypto/brain/toggle handler returns 410 Gone with no replacement", () => {
  // Cheap structural check — the route block contains a 410 status and an
  // explicit `replacement: null` because Task #444 removed the LLM sidecar
  // end-to-end. Old clients should stop calling the legacy toggle rather
  // than being pointed at another removed route.
  const block = routeSrc.split('router.post("/crypto/brain/toggle"')[1] ?? "";
  const handler = block.split("router.")[0]; // first router.* after this one
  assert.match(handler, /res\.status\(410\)/, "must respond 410 Gone");
  assert.match(handler, /replacement:\s*null/, "must not point to the removed LLM sidecar");
  assert.match(handler, /LLM\/news\/sidecar pipeline has been removed/, "must explain why there is no replacement");
});
