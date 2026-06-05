import { test } from "node:test";
import assert from "node:assert/strict";

// Phase 5 — orchestrator cycle-cache + brain tagging.
//
// Rather than monkey-patching ESM modules (read-only), we drive the
// orchestrator through the real brain-flag with QUANT_BRAIN_FORCE_OFF=1
// pinned ON. That deterministically routes through the LLM brain. We then
// stub `getAgentPrediction` by injecting a function via the optional
// `__llmOverride` test seam exposed by the orchestrator. The test seam is
// gated on NODE_ENV === "test" so it is a no-op in production.
//
// The cache test uses two different (coin,timeframe) pairs to assert one
// cache entry per pair, and a repeat call to assert no extra LLM invocations.

import {
  getPrediction,
  resetCycleCache,
  getCycleCacheSize,
  __setLlmOverride,
  __setQuantOverride,
} from "../src/lib/prediction-orchestrator";
import { invalidateBrainCache } from "../src/lib/brain-flag";
import {
  __setMlAvailabilitySnapshot,
  __resetMlAvailabilityCache,
} from "../src/lib/ml-availability";

function makeArgs(coinId = "BTC", timeframe: "5m" | "15m" = "5m") {
  return {
    agent: { name: "TestAgent", personality: "p", systemPrompt: "s", temperature: 0.7, preferredTimeframes: ["5m"] },
    coin: { id: coinId, symbol: coinId, name: coinId, currentPrice: 100, priceChange24h: 0 } as never,
    timeframe,
    patternAnalysis: {
      atr: 1, signalStrength: 50, volumeTrend: "stable", trendStrength: 0, patternType: "none",
      rsi: 50, macd: { crossover: "none" }, bollingerBands: { percentB: 0.5 }, volatility: 1,
    } as never,
    pastPatterns: { historicalAccuracy: 0, totalPast: 0, commonOutcome: "none", avgPriceMove: 0 },
    agentAccuracy: null,
    coinNews: [],
    insightsContext: "",
    correlationsContext: "",
    fearGreed: null,
    signalStrength: 50,
    agentId: 1,
    temporalContextStr: "",
    regimeAdj: { slMultiplier: 1, tpMultiplier: 1, confidenceModifier: 0 },
    contagionContext: "",
    trendBias: "neutral" as const,
  };
}

test("Task #255: quant disabled + LLM_EXECUTION not allowed → ABSTAIN, LLM is NOT called", async () => {
  process.env.QUANT_BRAIN_FORCE_OFF = "1";
  delete process.env.LLM_EXECUTION_ENABLED;
  invalidateBrainCache();
  resetCycleCache();
  let llmCalls = 0;
  __setLlmOverride(async () => {
    llmCalls++;
    return { direction: "up", confidence: 0.7, reasoning: "raw llm reasoning",
      predictedPrice: 101, stopLoss: 0.01, takeProfit: 0.02 };
  });
  try {
    const r = await getPrediction(makeArgs());
    assert.equal(r.brain, "ABSTAIN");
    assert.equal(r.direction, "stable");
    assert.equal(r.confidence, 0);
    assert.equal(llmCalls, 0, "LLM must not author trades when LLM_EXECUTION_ENABLED is unset");
    assert.match(r.reasoning, /\[BRAIN=ABSTAIN\]/);
  } finally {
    __setLlmOverride(null);
    delete process.env.QUANT_BRAIN_FORCE_OFF;
    invalidateBrainCache();
  }
});

test("Task #255: even with LLM_EXECUTION_ENABLED=1 the orchestrator still ABSTAINS (no escape hatch)", async () => {
  // The reviewer for Task #255 specifically required that the LLM cannot
  // author executable trades — period. The previous developer "shadow lane"
  // env was removed; this test pins the new invariant so it can't regress.
  process.env.QUANT_BRAIN_FORCE_OFF = "1";
  process.env.LLM_EXECUTION_ENABLED = "1";
  invalidateBrainCache();
  resetCycleCache();
  let llmCalls = 0;
  __setLlmOverride(async () => {
    llmCalls++;
    return { direction: "up", confidence: 0.7, reasoning: "raw llm reasoning",
      predictedPrice: 101, stopLoss: 0.01, takeProfit: 0.02 };
  });
  try {
    const r = await getPrediction(makeArgs());
    assert.equal(r.brain, "ABSTAIN", "LLM_EXECUTION_ENABLED must NOT re-enable the LLM brain");
    assert.equal(llmCalls, 0, "LLM must not be invoked even with the legacy env set");
  } finally {
    __setLlmOverride(null);
    delete process.env.QUANT_BRAIN_FORCE_OFF;
    delete process.env.LLM_EXECUTION_ENABLED;
    invalidateBrainCache();
  }
});

test("Task #255: quant has no model AND no LLM-execution opt-in → ABSTAIN (LLM never called)", async () => {
  // Snapshot says BTC has a 5m model but nothing for 15m. The orchestrator
  // MUST NOT fall through to the LLM brain — it must abstain. This is the
  // central guard for Task #255: the LLM cannot author trades.
  __resetMlAvailabilityCache();
  __setMlAvailabilitySnapshot([{ coinId: "BTC", timeframe: "5m" }]);
  delete process.env.LLM_EXECUTION_ENABLED;

  let quantCalls = 0;
  let llmCalls = 0;
  __setQuantOverride(async () => {
    quantCalls++;
    throw new Error("quant brain should not be called when no model is available");
  });
  __setLlmOverride(async () => {
    llmCalls++;
    return {
      direction: "stable", confidence: 0.4, reasoning: "llm fallback",
      predictedPrice: 100, stopLoss: 0.01, takeProfit: 0.02,
    };
  });
  process.env.QUANT_BRAIN_TEST_FORCE_ON = "1";
  resetCycleCache();
  try {
    const r = await getPrediction(makeArgs("BTC", "15m"));
    assert.equal(quantCalls, 0, "ml-engine /predict must be skipped when no model is available");
    assert.equal(llmCalls, 0, "LLM brain MUST NOT be the fallback for trade decisions (Task #255)");
    assert.equal(r.brain, "ABSTAIN");
    assert.equal(r.abstainReason, "no_model");
  } finally {
    __setLlmOverride(null);
    __setQuantOverride(null);
    delete process.env.QUANT_BRAIN_TEST_FORCE_ON;
    __resetMlAvailabilityCache();
  }
});

test("QUANT brain calls /predict when availability snapshot has the model", async () => {
  __resetMlAvailabilityCache();
  __setMlAvailabilitySnapshot([{ coinId: "BTC", timeframe: "5m" }]);

  __setQuantOverride(async () => ({
    direction: "up", confidence: 0.6, rawConfidence: 0.6,
    reasoning: "[BRAIN=QUANT] up",
    predictedPrice: 101, stopLoss: 0.01, takeProfit: 0.02,
    modelVersion: "test", source: "stub", expectedReturnPct: 1,
    probUp: 0.6, probDown: 0.2, probStable: 0.2,
  }));
  process.env.QUANT_BRAIN_TEST_FORCE_ON = "1";
  resetCycleCache();
  try {
    const r = await getPrediction(makeArgs("BTC", "5m"));
    assert.equal(r.brain, "QUANT");
  } finally {
    __setQuantOverride(null);
    delete process.env.QUANT_BRAIN_TEST_FORCE_ON;
    __resetMlAvailabilityCache();
  }
});

test("Task #418: pooled-fallback prediction is treated as valid signal (not no-model)", async () => {
  // Regression for Task #418 — when ml-engine's per-coin slice fails the
  // verification gate, /ml/predict serves the pooled head and stamps
  // `fallback="pooled"` on the 200 response. The orchestrator MUST treat
  // that exactly the same as a regular per-coin prediction (route through
  // the QUANT brain, NOT abstain with `no_model`). The only "no model"
  // signal is the explicit `noModelAvailable=true` flag the quant-brain
  // adapter sets on a 503 from /ml/predict — a 200 pooled-fallback
  // response must never be confused with that.
  __resetMlAvailabilityCache();
  __setMlAvailabilitySnapshot([{ coinId: "BTC", timeframe: "5m" }]);
  __setQuantOverride(async () => ({
    direction: "up", confidence: 0.6, rawConfidence: 0.6,
    reasoning: "[BRAIN=QUANT] up (pooled fallback)",
    predictedPrice: 101, stopLoss: 0.01, takeProfit: 0.02,
    // The quant-brain adapter populates these from MlPredictResponse;
    // a pooled-fallback response carries `modelCoinId="__pooled__"`
    // and `fallback="pooled"`, but neither field is what gates the
    // orchestrator — `noModelAvailable` is. Leaving it absent here
    // mirrors the real adapter's behaviour for any 200 response.
    modelVersion: "pool-v1", source: "lightgbm", expectedReturnPct: 0.8,
    probUp: 0.6, probDown: 0.2, probStable: 0.2,
  }));
  __setLlmOverride(async () => { throw new Error("LLM must not be called for pooled-fallback predictions"); });
  process.env.QUANT_BRAIN_TEST_FORCE_ON = "1";
  resetCycleCache();
  try {
    const r = await getPrediction(makeArgs("BTC", "5m"));
    assert.equal(r.brain, "QUANT", "pooled-fallback must route through QUANT, not ABSTAIN");
    assert.notEqual(r.abstainReason, "no_model", "pooled response must NOT abstain with no_model");
    assert.equal(r.direction, "up");
  } finally {
    __setLlmOverride(null);
    __setQuantOverride(null);
    delete process.env.QUANT_BRAIN_TEST_FORCE_ON;
    __resetMlAvailabilityCache();
  }
});

test("QUANT brain path caches per coin×timeframe across agents", async () => {
  // Force flag ON via the test override (bypasses env force-off).
  __setQuantOverride(async () => ({
    direction: "up", confidence: 0.6, rawConfidence: 0.6,
    reasoning: "[BRAIN=QUANT] up",
    predictedPrice: 101, stopLoss: 0.01, takeProfit: 0.02,
    modelVersion: "test", source: "stub", expectedReturnPct: 1,
    probUp: 0.6, probDown: 0.2, probStable: 0.2,
  }));
  __setLlmOverride(async () => { throw new Error("LLM should not be called"); });
  // Force the orchestrator to think the flag is ON.
  process.env.QUANT_BRAIN_TEST_FORCE_ON = "1";
  resetCycleCache();
  try {
    const r1 = await getPrediction(makeArgs("BTC", "5m"));
    const r2 = await getPrediction({ ...makeArgs("BTC", "5m"), agent: { ...makeArgs().agent, name: "Other" } });
    const r3 = await getPrediction(makeArgs("ETH", "5m"));
    assert.equal(r1.brain, "QUANT");
    assert.equal(r2.brain, "QUANT");
    assert.equal(r3.brain, "QUANT");
    assert.match(r1.reasoning, /\[BRAIN=QUANT\]/);
    assert.equal(getCycleCacheSize(), 2, "cache holds BTC:5m and ETH:5m only");
    resetCycleCache();
    assert.equal(getCycleCacheSize(), 0);
  } finally {
    __setLlmOverride(null);
    __setQuantOverride(null);
    delete process.env.QUANT_BRAIN_TEST_FORCE_ON;
  }
});
