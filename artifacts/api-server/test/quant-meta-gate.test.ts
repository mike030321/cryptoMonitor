import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { decideMetaGate } from "../src/lib/quant-brain";
import type { QuantSpecialistView } from "../src/lib/_legacy/ai-engine";

function spec(
  kind: string,
  applicable: boolean,
  probUp: number | null,
  probDown: number | null,
  error: string | null = null,
): QuantSpecialistView {
  return {
    kind,
    modelVersion: "v1",
    modelCoinId: kind,
    regimeSubset: [],
    applicable,
    probUp,
    probDown,
    probStable:
      probUp !== null && probDown !== null
        ? Math.max(0, 1 - probUp - probDown)
        : null,
    expectedReturnPct: null,
    confidence: null,
    error,
  };
}

describe("decideMetaGate (Phase 4 specialist meta-gate)", () => {
  it("passes through when the main head abstained", () => {
    const out = decideMetaGate("stable", [spec("a", true, 0.6, 0.1)], "trending_up");
    assert.equal(out.abstained, false);
    assert.equal(out.reason, "main_head_abstain");
  });

  it("passes through when no specialists are applicable", () => {
    const out = decideMetaGate("up", [spec("a", false, 0.6, 0.1)], "trending_up");
    assert.equal(out.abstained, false);
    assert.equal(out.applicable, 0);
    assert.equal(out.reason, "no_applicable_specialists");
  });

  it("passes through below quorum (only 1 applicable)", () => {
    const out = decideMetaGate("up", [spec("a", true, 0.6, 0.1)], "trending_up");
    assert.equal(out.abstained, false);
    assert.equal(out.applicable, 1);
    assert.equal(out.reason, "below_quorum");
  });

  it("trades when specialists agree with the main direction", () => {
    const specs = [
      spec("momentum", true, 0.55, 0.20),
      spec("breakout", true, 0.60, 0.15),
    ];
    const out = decideMetaGate("up", specs, "trending_up");
    assert.equal(out.abstained, false);
    assert.equal(out.reason, "specialist_agree");
    assert.equal(out.agreementVotes, 2);
    assert.equal(out.dissentVotes, 0);
    assert.ok(out.meanDirectionalScore > 0);
  });

  it("abstains when specialists dissent (majority opposite)", () => {
    const specs = [
      spec("momentum", true, 0.20, 0.55), // calls down
      spec("breakout", true, 0.18, 0.60), // calls down
    ];
    const out = decideMetaGate("up", specs, "trending_up");
    assert.equal(out.abstained, true);
    assert.equal(out.reason, "specialist_dissent_high");
    assert.equal(out.dissentVotes, 2);
  });

  it("abstains when mean directional score points the wrong way (counter signal)", () => {
    // Even split votes (1 up, 1 down) but mean score net negative, main=up.
    // 0.5 agreement ratio passes the default 0.5 threshold (>=), so the
    // dissent gate doesn't fire — but mean = (0.05 + (-0.4))/2 = -0.175, so
    // score gate triggers instead.
    const specs = [
      spec("momentum", true, 0.40, 0.35),  // up, score +0.05
      spec("breakout", true, 0.20, 0.60),  // down, score -0.40
    ];
    const out = decideMetaGate("up", specs, "trending_up");
    assert.equal(out.abstained, true);
    assert.equal(out.reason, "specialist_meta_counter_signal");
  });

  it("ignores specialists with errors or null probs", () => {
    const specs = [
      spec("momentum", true, 0.55, 0.20),
      spec("breakout", true, null, null, "model load failed"),
      spec("vol", true, 0.50, 0.30),
    ];
    const out = decideMetaGate("up", specs, "trending_up");
    assert.equal(out.applicable, 2);
    assert.equal(out.abstained, false);
    assert.equal(out.reason, "specialist_agree");
  });
});
