import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");

const quantBrain = readFileSync(
  path.join(REPO, "artifacts/api-server/src/lib/quant-brain.ts"),
  "utf8",
);
const paperTrader = readFileSync(
  path.join(REPO, "artifacts/api-server/src/lib/paper-trader.ts"),
  "utf8",
);

test("learned meta-model no_trade is explicit zero-confidence abstain", () => {
  assert.match(
    quantBrain,
    /const\s+metaModelNoTrade\s*=\s*metaUsed\s*&&\s*metaAction\s*===\s*"no_trade"/,
  );
  assert.match(
    quantBrain,
    /const\s+confidence\s*=\s*metaGate\.abstained\s*\|\|\s*metaModelNoTrade[\s\S]*\?\s*0/,
    "meta no_trade must not inherit probStable as trade confidence",
  );
  assert.match(
    quantBrain,
    /\[ABSTAIN:\$\{metaAbstainReason\s*\?\?\s*"no_trade"\}\]/,
    "reasoning should make no_trade visible even when abstainReason is null",
  );
});

test("paper trader records meta no_trade before stable-direction early return", () => {
  const metaIdx = paperTrader.indexOf('quant?.metaAction === "no_trade"');
  const stableIdx = paperTrader.indexOf('if (direction === "stable") return;');
  assert.ok(metaIdx >= 0, "paper trader must check metaAction no_trade");
  assert.ok(stableIdx > metaIdx, "typed meta no_trade skip must happen before stable return");
  const block = paperTrader.slice(metaIdx, stableIdx);
  assert.match(block, /const\s+skipReason\s*=\s*mapped\s*\?\?\s*"quant_meta_abstain"/);
  assert.match(block, /await\s+recordSkip\s*\(/);
  assert.match(block, /return\s*;/);
});
