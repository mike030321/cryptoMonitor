import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");

const paperTrader = readFileSync(
  path.join(REPO, "artifacts/api-server/src/lib/paper-trader.ts"),
  "utf8",
);

test("anomaly-cancelled trades clear meta-brain bindings without recording an outcome", () => {
  const cancelIdx = paperTrader.indexOf("exitReason: \"anomaly-cancel\"");
  assert.ok(cancelIdx >= 0, "anomaly cancel journal block not found");
  const block = paperTrader.slice(cancelIdx, cancelIdx + 900);
  assert.match(
    block,
    /clearTickBinding\s*\(\s*pos\.tradeId\s*\)/,
    "data-glitch cancelled trades must not leave stale trade->tick bindings",
  );
  assert.doesNotMatch(
    block,
    /sendRecordOutcome\s*\(/,
    "anomaly cancels are full reversals and must not train the meta-brain",
  );
});
