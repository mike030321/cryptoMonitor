/**
 * Task #369 — runtime companion to `no-llm-fields-in-trade-decisions.test.ts`.
 *
 * The sister file statically extracts object-literal keys from the live
 * production source. This file EXECUTES the live decision pipeline
 * (`executePaperTrade` from paper-trader.ts) against a fake market cycle,
 * captures every payload that the real code actually builds at runtime
 * (recordSkip contexts, /ml/decide request bodies, paper_trades /
 * paper_positions INSERTs, the "Paper trade opened" structured log),
 * and asserts:
 *
 *   1. every object key captured is in the quant-only ALLOWED_KEYS set
 *      (allowlist contract), AND
 *   2. no captured key matches an LLM-derived identifier pattern
 *      (belt-and-suspenders forbidden-prefix scan).
 *
 * All external dependencies (db, ml-client, skip-tracker, regime-detector,
 * horizon-gate, logger, journal-writer, coins) are replaced via
 * node:test `t.mock.module()` so the real executePaperTrade runs against
 * an in-memory fake and we observe the payloads it actually constructs.
 * Any future code change that adds a key to any of those payloads (e.g.
 * `newsBias`, `llmEdge`, `sentimentScore`) will be caught at runtime.
 *
 * This test requires `--experimental-test-module-mocks`. The
 * `decision-engine-parity` validation command passes that flag.
 */
import { describe, test, before, beforeEach, mock } from "node:test";
import assert from "node:assert/strict";

// --- Allowlist (duplicated from the static-analysis sister file so each
// test stands alone; a drift in allowlist between runtime and static
// surfaces as a bug because both tests must pass). ---------------------
const ALLOWED_KEYS = new Set<string>([
  "agentId", "agentName", "coinId", "coinName", "timeframe", "direction",
  "brain", "tradeId", "predictionId",
  "action", "entryPrice", "entryFee", "quantity", "positionSize",
  "stopLoss", "takeProfit", "stopLossPrice", "takeProfitPrice",
  "peakPrice", "expiresAt", "status", "slippage", "balance",
  "liveCash", "want", "closed", "trendBias", "minMomentum",
  "eligibleCount", "totalCoins", "regime", "cash",
  "confidence", "required", "requiredPct", "threshold",
  "tpDistancePct", "atrPct", "evScore", "evRequired",
  "sameSide", "totalOpen", "sameSideShare",
  "dailyLossLimit", "drawdownHalt", "dailyLossHit", "drawdownHit",
  "avgChange24h", "err",
  "probUp", "probDown", "probStable", "expectedReturnPct",
  "directionalReturnPct", "modelVersion", "source", "predictionStdPct",
  "rawConfidence", "featureHash",
  "metaAction", "metaKind", "metaVersion", "metaExpectedEdgePct",
  "metaSizeMultiplier", "metaAbstainReason", "metaGate",
  "lastPrice", "atrValue", "portfolio", "equityUsd", "cashUsd",
  "openPositions", "newNotionalUsd", "notionalUsd", "regimeAtEntry",
  "betaToBtc", "gates", "base",
  "reasoning", "predictedPrice", "noModelAvailable", "specialists",
  "quant", "coin",
  "enabled", "sector_share_after", "sector_cap", "correlated_cap",
  "book_beta", "beta_cap", "regime_share_after", "regime_budget",
  "sector", "correlatedCoins", "new_sector", "open_notional_usd",
  "new_notional_usd",
  "entryRegimeLabel",
  "skipReason", "skipDetail", "gatesApplied", "portfolioCheck",
]);

const FORBIDDEN = [
  /\b(news|llm|gpt|sentiment|ai|chatgpt|gemini|openai|anthropic|claude)_[a-z][\w]*/,
  /\b(news|llm|gpt|sentiment|ai|chatgpt|gemini|openai|anthropic|claude)[A-Z][\w]*/,
  /\b\w*?(News|Llm|Gpt|Sentiment|Chatgpt|Gemini|OpenAi|Anthropic|Claude)(?:[A-Z]\w*|Score|Bias|Tag|Vote|Edge|Signal|Rating|Call)\b/,
  // Task #390 — Strategy Lab benchmark telemetry must NEVER reach the
  // trade-decision payloads. Catches benchmark / alpha / baseline /
  // strategyLab identifiers in any case.
  /\b(benchmark|alpha|baseline|strategy_lab)_[a-z][\w]*/,
  /\b(benchmark|strategyLab)[A-Z][\w]*/,
  /\b\w*?(Benchmark|StrategyLab|Alpha|Baseline)(?:[A-Z]\w*|Score|Return|Ratio|Trust)\b/,
];
function scanForbiddenKey(k: string): boolean {
  for (const re of FORBIDDEN) {
    if (new RegExp(re.source).test(k)) return true;
  }
  return false;
}

function collectAllKeys(obj: unknown, keys: Set<string>, depth = 0, seen = new WeakSet<object>()): void {
  if (depth > 50 || obj === null || obj === undefined || typeof obj !== "object") return;
  if (seen.has(obj as object)) return;
  seen.add(obj as object);
  if (Array.isArray(obj)) {
    for (const v of obj) collectAllKeys(v, keys, depth + 1, seen);
    return;
  }
  for (const k of Object.keys(obj as object)) {
    keys.add(k);
    collectAllKeys((obj as Record<string, unknown>)[k], keys, depth + 1, seen);
  }
}

function assertPayloadClean(label: string, payload: unknown): void {
  const keys = new Set<string>();
  collectAllKeys(payload, keys);
  const forbidden: string[] = [];
  const notAllowlisted: string[] = [];
  for (const k of keys) {
    if (!ALLOWED_KEYS.has(k)) notAllowlisted.push(k);
    if (scanForbiddenKey(k)) forbidden.push(k);
  }
  assert.deepEqual(
    forbidden, [],
    `[${label}] runtime payload contains LLM-derived keys: ${forbidden.join(", ")}`,
  );
  assert.deepEqual(
    notAllowlisted, [],
    `[${label}] runtime payload contains non-allowlisted keys: ${notAllowlisted.join(", ")}\nAdd to ALLOWED_KEYS if legitimate quant-only, else this is a leak.`,
  );
}

// =============================================================================
// Single shared mock graph — installed once so paper-trader resolves all
// deps against the same mock objects across every scenario. Scenarios
// adjust runtime behavior by mutating module-level state, not by
// reinstalling mocks.
// =============================================================================
interface Captured {
  recordSkipContexts: Array<{ reason: string; ctx: unknown }>;
  mlDecideRequests: unknown[];
  paperTradesInserts: unknown[];
  paperPositionsInserts: unknown[];
  openedLogs: unknown[];
}
const captured: Captured = {
  recordSkipContexts: [],
  mlDecideRequests: [],
  paperTradesInserts: [],
  paperPositionsInserts: [],
  openedLogs: [],
};
function resetCaptured(): void {
  captured.recordSkipContexts.length = 0;
  captured.mlDecideRequests.length = 0;
  captured.paperTradesInserts.length = 0;
  captured.paperPositionsInserts.length = 0;
  captured.openedLogs.length = 0;
}

interface State {
  portfolio: {
    agentId: number; totalValue: number; cashBalance: number;
    peakValue: number; dayStartValue: number; dayStartDate: string;
    totalTrades: number; winningTrades: number;
  };
  positions: Array<{ agentId: number; coinId: string; direction: string; positionSize: number; entryRegimeLabel: string | null }>;
  fleetPositions: Array<{ direction: string }>;
  recentTrades: Array<{ pnl: number | null; positionSize: number }>;
  regime: { regime: string; regimeLabel: string; trendBias: string; avgChange24h: number } | null;
  mlDecide: {
    action: "trade" | "no_trade";
    direction?: "up" | "down";
    positionSizeUsd?: number;
    slPrice?: number | null;
    tpPrice?: number | null;
    gatesApplied?: unknown;
    portfolioCheck?: unknown;
    skipReason?: string | null;
    skipDetail?: string | null;
  };
}
const STATE: State = {
  portfolio: {
    agentId: 1, totalValue: 10_000, cashBalance: 10_000,
    peakValue: 10_000, dayStartValue: 10_000,
    dayStartDate: new Date().toISOString().slice(0, 10),
    totalTrades: 0, winningTrades: 0,
  },
  positions: [],
  fleetPositions: [],
  recentTrades: [],
  regime: { regime: "bull", regimeLabel: "bull_trend_high_vol", trendBias: "bullish", avgChange24h: 2.5 },
  mlDecide: { action: "trade", direction: "up", positionSizeUsd: 250, slPrice: 95, tpPrice: 110,
    gatesApplied: { gateConfidence: true, gateEv: true },
    portfolioCheck: { enabled: true, sector: "layer1", new_sector: "layer1", open_notional_usd: 0, new_notional_usd: 250, sector_share_after: 0.025, sector_cap: 0.35 },
  },
};
function resetState(): void {
  STATE.portfolio = {
    agentId: 1, totalValue: 10_000, cashBalance: 10_000,
    peakValue: 10_000, dayStartValue: 10_000,
    dayStartDate: new Date().toISOString().slice(0, 10),
    totalTrades: 0, winningTrades: 0,
  };
  STATE.positions = [];
  STATE.fleetPositions = [];
  STATE.recentTrades = [];
  STATE.regime = { regime: "bull", regimeLabel: "bull_trend_high_vol", trendBias: "bullish", avgChange24h: 2.5 };
  STATE.mlDecide = { action: "trade", direction: "up", positionSizeUsd: 250, slPrice: 95, tpPrice: 110,
    gatesApplied: { gateConfidence: true, gateEv: true },
    portfolioCheck: { enabled: true, sector: "layer1", new_sector: "layer1", open_notional_usd: 0, new_notional_usd: 250, sector_share_after: 0.025, sector_cap: 0.35 },
  };
}

type AnyFn = (...args: unknown[]) => unknown;
type ExecuteFn = (
  agentId: number, agentName: string, coinId: string, coinName: string,
  direction: "up" | "down" | "stable", confidence: number, entryPrice: number,
  timeframe: string, predictionId: number, atrValue: number, quant?: unknown,
) => Promise<void>;

let executePaperTrade: ExecuteFn;

before(async () => {
  // Marker tables (fake db routes by __table identity).
  const paperPortfoliosTable = { __table: "portfolios", agentId: "agentId" };
  const paperTradesTable = { __table: "trades", agentId: "agentId", coinId: "coinId", status: "status", closedAt: "closedAt", id: "id" };
  const paperPositionsTable = { __table: "positions", agentId: "agentId", coinId: "coinId", direction: "direction" };
  const inert = (name: string) => ({ __table: name });

  let projContext: "fleet" | "default" = "default";
  function chain() {
    let rows: unknown[] = [];
    const c: Record<string, AnyFn> & { then?: AnyFn } = {
      from: (tbl: { __table: string }) => {
        if (tbl.__table === "portfolios") rows = [STATE.portfolio];
        else if (tbl.__table === "positions") rows = projContext === "fleet" ? STATE.fleetPositions : STATE.positions;
        else if (tbl.__table === "trades") rows = STATE.recentTrades;
        else rows = [];
        return c;
      },
      where: () => c,
      for: () => c,
      orderBy: () => c,
      limit: () => c,
      then: (res: AnyFn, rej: AnyFn) => Promise.resolve(rows).then(res as never, rej as never),
    };
    return c;
  }
  const tx = {
    select: (proj?: unknown) => { projContext = proj ? "fleet" : "default"; return chain(); },
    update: () => ({ set: () => ({ where: () => Promise.resolve() }) }),
    insert: (tbl: { __table: string }) => ({
      values: (v: unknown) => {
        if (tbl.__table === "trades") captured.paperTradesInserts.push(v);
        else if (tbl.__table === "positions") captured.paperPositionsInserts.push(v);
        const rows = [{ id: 424242 }];
        const r: Record<string, AnyFn> & { then?: AnyFn } = {
          returning: () => Promise.resolve(rows),
          then: (res: AnyFn, rej: AnyFn) => Promise.resolve(rows).then(res as never, rej as never),
        };
        return r;
      },
    }),
  };
  const db = {
    select: tx.select,
    update: tx.update,
    insert: tx.insert,
    transaction: async (cb: (tx: typeof tx) => Promise<unknown>) => cb(tx),
  };

  mock.module("@workspace/db", {
    namedExports: {
      db, pool: {},
      paperPortfoliosTable, paperTradesTable, paperPositionsTable,
      paperPositionMarksTable: inert("paperPositionMarks"),
      agentsTable: inert("agents"),
      autoDeployAttributionSnapshotsTable: inert("autoDeployAttr"),
      agentFitnessSnapshotsTable: inert("agentFitnessSnapshots"),
      appSettingsTable: inert("appSettings"),
      coinCorrelationsTable: inert("coinCorrelations"),
      coinInsightsTable: inert("coinInsights"),
      evolutionHistoryTable: inert("evolutionHistory"),
      fingerprintBuffersTable: inert("fingerprintBuffers"),
      llmDriftExplanationsTable: inert("llmDriftExplanations"),
      llmEventSignalsTable: inert("llmEventSignals"),
      llmOperatorAnswersTable: inert("llmOperatorAnswers"),
      llmTradePostmortemsTable: inert("llmTradePostmortems"),
      marketSignalsTable: inert("marketSignals"),
      modelPredictionsTable: inert("modelPredictions"),
      monitoringStateTable: inert("monitoringState"),
      newsTagsTable: inert("newsTags"),
      predictionJournalTable: inert("predictionJournal"),
      predictionsTable: inert("predictions"),
      priceHistoryTable: inert("priceHistory"),
      skipEventsTable: inert("skipEvents"),
      strategySettingsHistoryTable: inert("strategySettingsHistory"),
      strategySettingsTable: inert("strategySettings"),
      strategySnapshotsTable: inert("strategySnapshots"),
      strategyStateTable: inert("strategyState"),
      tradeJournalTable: inert("tradeJournal"),
      tuningGateStateTable: inert("tuningGateState"),
      FEATURE_TRANSFORM_KINDS: ["identity", "log1p", "zscore"],
    },
  });

  const realSkipTracker = await import("../src/lib/skip-tracker.ts");
  mock.module("../src/lib/skip-tracker.ts", {
    namedExports: {
      ...realSkipTracker,
      recordSkip: (reason: string, _agent: string, _msg: string, ctx: unknown) => {
        captured.recordSkipContexts.push({ reason, ctx });
      },
    },
  });

  const realMlClient = await import("../src/lib/ml-client.ts");
  mock.module("../src/lib/ml-client.ts", {
    namedExports: {
      ...realMlClient,
      getMlDecision: async (req: unknown) => {
        captured.mlDecideRequests.push(req);
        const d = STATE.mlDecide;
        return {
          action: d.action,
          direction: d.direction ?? null,
          positionSizeUsd: d.positionSizeUsd ?? 0,
          slPrice: d.slPrice ?? null,
          tpPrice: d.tpPrice ?? null,
          gatesApplied: d.gatesApplied ?? {},
          portfolioCheck: d.portfolioCheck ?? {},
          skipReason: d.skipReason ?? null,
          skipDetail: d.skipDetail ?? null,
        };
      },
    },
  });

  const realRegime = await import("../src/lib/regime-detector.ts");
  mock.module("../src/lib/regime-detector.ts", {
    namedExports: { ...realRegime, getCurrentRegime: () => STATE.regime },
  });

  const realHorizon = await import("../src/lib/horizon-gate.ts");
  mock.module("../src/lib/horizon-gate.ts", {
    namedExports: { ...realHorizon, isHorizonDisabled: async () => false },
  });

  const realLogger = await import("../src/lib/logger.ts");
  const loggerStub: Record<string, AnyFn> = {
    info: (obj: unknown, msg?: unknown) => { if (msg === "Paper trade opened") captured.openedLogs.push(obj); },
    warn: () => {}, debug: () => {}, error: () => {}, trace: () => {}, fatal: () => {},
  };
  loggerStub.child = () => loggerStub;
  mock.module("../src/lib/logger.ts", {
    namedExports: { ...realLogger, logger: loggerStub },
  });

  const realJournal = await import("../src/lib/journal-writer.ts");
  mock.module("../src/lib/journal-writer.ts", {
    namedExports: { ...realJournal, writeTradeJournal: async () => {} },
  });

  const realCoins = await import("../src/lib/coins.ts");
  mock.module("../src/lib/coins.ts", {
    namedExports: {
      ...realCoins,
      fetchCoinPrices: async () => [],
      isPriceDataFresh: () => true,
    },
  });

  // Task #468 — paper-trader's first gate calls
  // getCachedProfileForAgentId() on the boot-loaded registry cache.
  // The mocked DB graph above never runs the real boot sweep, so the
  // cache is empty by default and every test would fail with
  // "agent_not_executable". Mock the cache module to return a fixed
  // momentum_core entry for agentId=1 (the id every fixture uses).
  // Tests that need to assert non-executable behaviour live in
  // agents-registry.test.ts where they exercise the real cache.
  const realRegistry = await import("../src/lib/agents-registry/index.ts");
  const realCache = await import("../src/lib/agents-registry/cache.ts");
  // Synthesise a maximally-permissive executor profile so the
  // registry gate never blocks any test fixture (the gates being
  // exercised here live AFTER the registry gate). Real per-profile
  // gate semantics are covered by agents-registry.test.ts.
  const fixedProfile = {
    ...realRegistry.getAgentProfile("momentum_core"),
    preferred_regimes: "all" as const,
    blocked_regimes: [],
    min_confidence: 0.0,
    min_expected_edge_after_costs: -1.0,
  };
  mock.module("../src/lib/agents-registry/cache.ts", {
    namedExports: {
      ...realCache,
      getCachedProfileForAgentId: (agentId: number) => {
        if (agentId === 1) {
          return { profile: fixedProfile, dbStatus: "active", agentName: "TestBot" };
        }
        throw new realCache.AgentNotExecutableError(
          "unknown_agent_id",
          agentId,
          null,
          null,
          `test-mock cache: unknown agent ${agentId}`,
        );
      },
      tryGetCachedEntry: (agentId: number) =>
        agentId === 1
          ? { agentId: 1, agentName: "TestBot", profile: fixedProfile, dbStatus: "active" }
          : null,
    },
  });
  // The barrel re-exports the cache symbols, so re-mock the barrel
  // too — paper-trader imports `getCachedProfileForAgentId` from the
  // barrel, not directly from cache.ts.
  mock.module("../src/lib/agents-registry/index.ts", {
    namedExports: {
      ...realRegistry,
      getCachedProfileForAgentId: (agentId: number) => {
        if (agentId === 1) {
          return { profile: fixedProfile, dbStatus: "active", agentName: "TestBot" };
        }
        throw new realCache.AgentNotExecutableError(
          "unknown_agent_id",
          agentId,
          null,
          null,
          `test-mock cache: unknown agent ${agentId}`,
        );
      },
      tryGetCachedEntry: (agentId: number) =>
        agentId === 1
          ? { agentId: 1, agentName: "TestBot", profile: fixedProfile, dbStatus: "active" }
          : null,
    },
  });

  // Now load paper-trader against the mocked graph.
  const mod = await import("../src/lib/paper-trader.ts");
  executePaperTrade = mod.executePaperTrade as ExecuteFn;
});

beforeEach(() => {
  resetCaptured();
  resetState();
});

// =============================================================================
// Scenarios
// =============================================================================
describe("Task #369 runtime — no LLM-derived field reaches live decision path", () => {
  test("meta-abstain on low_edge: typed skip context contains only allowlisted keys", async () => {
    await executePaperTrade(1, "TestBot", "bitcoin", "Bitcoin", "up", 0.70, 100, "5m", 1, 1.0, {
      probUp: 0.6, probDown: 0.2, probStable: 0.2, expectedReturnPct: 5.0, modelVersion: "v1",
      metaAction: "no_trade", metaAbstainReason: "low_edge", metaKind: "mlp", metaVersion: "v2",
      metaExpectedEdgePct: 0.05,
    });
    assert.equal(captured.recordSkipContexts.length, 1);
    assert.match(captured.recordSkipContexts[0].reason, /meta_no_trade_low_edge/);
    assertPayloadClean("meta_abstain skip ctx", captured.recordSkipContexts[0].ctx);
    assert.equal(captured.paperTradesInserts.length, 0);
  });

  test("confidence below floor: skip context contains only allowlisted keys", async () => {
    await executePaperTrade(1, "TestBot", "bitcoin", "Bitcoin", "up", 0.10, 100, "5m", 1, 1.0, {
      probUp: 0.55, probDown: 0.25, probStable: 0.20, expectedReturnPct: 5.0, modelVersion: "v1",
    });
    const skip = captured.recordSkipContexts.find((s) => s.reason === "confidence_below_threshold");
    assert.ok(skip, `expected confidence_below_threshold, got: ${captured.recordSkipContexts.map((s) => s.reason).join(",")}`);
    assertPayloadClean("confidence skip ctx", skip.ctx);
  });

  test("fleet correlation brake: skip context contains only allowlisted keys", async () => {
    STATE.fleetPositions = Array.from({ length: 8 }, () => ({ direction: "up" }));
    await executePaperTrade(1, "TestBot", "bitcoin", "Bitcoin", "up", 0.70, 100, "5m", 1, 1.5, {
      probUp: 0.65, probDown: 0.20, probStable: 0.15, expectedReturnPct: 5.0, modelVersion: "v1",
    });
    const skip = captured.recordSkipContexts.find((s) => s.reason === "fleet_direction_imbalance");
    assert.ok(skip, `expected fleet_direction_imbalance, got: ${captured.recordSkipContexts.map((s) => s.reason).join(",")}`);
    assertPayloadClean("fleet-brake skip ctx", skip.ctx);
  });

  test("counter-trend regime: skip context contains only allowlisted keys", async () => {
    STATE.regime = { regime: "bull", regimeLabel: "bull_trend_high_vol", trendBias: "bullish", avgChange24h: 3.0 };
    await executePaperTrade(1, "TestBot", "bitcoin", "Bitcoin", "down", 0.55, 100, "5m", 1, 1.5, {
      probUp: 0.20, probDown: 0.65, probStable: 0.15, expectedReturnPct: -5.0, modelVersion: "v1",
    });
    const skip = captured.recordSkipContexts.find((s) => s.reason === "counter_trend_regime");
    assert.ok(skip, `expected counter_trend_regime, got: ${captured.recordSkipContexts.map((s) => s.reason).join(",")}`);
    assertPayloadClean("counter-trend skip ctx", skip.ctx);
  });

  test("/ml/decide approve path: request + inserts + opened-log carry only allowlisted keys", async () => {
    await executePaperTrade(1, "TestBot", "bitcoin", "Bitcoin", "up", 0.70, 100, "5m", 1, 1.5, {
      probUp: 0.65, probDown: 0.20, probStable: 0.15, expectedReturnPct: 5.0, modelVersion: "v1",
      metaAction: "long", metaSizeMultiplier: 1.0, metaExpectedEdgePct: 0.4, metaKind: "mlp", metaVersion: "v2",
    });
    assert.equal(captured.mlDecideRequests.length, 1, `expected 1 /ml/decide req; skips: ${JSON.stringify(captured.recordSkipContexts.map((s) => s.reason))}`);
    assertPayloadClean("/ml/decide request", captured.mlDecideRequests[0]);
    assert.ok(captured.paperTradesInserts.length >= 1, `expected paper_trades INSERT; skips: ${JSON.stringify(captured.recordSkipContexts.map((s) => s.reason))}`);
    assertPayloadClean("paper_trades row", captured.paperTradesInserts[0]);
    assert.ok(captured.paperPositionsInserts.length >= 1);
    assertPayloadClean("paper_positions row", captured.paperPositionsInserts[0]);
    assert.ok(captured.openedLogs.length >= 1);
    assertPayloadClean("opened-log payload", captured.openedLogs[0]);
  });

  test("/ml/decide no_trade: skip context contains only allowlisted keys", async () => {
    STATE.mlDecide = {
      action: "no_trade", skipReason: "portfolio_sector_cap", skipDetail: "sector exposure too high",
      gatesApplied: { enabled: true },
      portfolioCheck: { enabled: true, sector: "alt", new_sector: "alt", sector_share_after: 0.4, sector_cap: 0.35 },
    };
    await executePaperTrade(1, "TestBot", "bitcoin", "Bitcoin", "up", 0.70, 100, "5m", 1, 1.5, {
      probUp: 0.65, probDown: 0.20, probStable: 0.15, expectedReturnPct: 5.0, modelVersion: "v1",
      metaAction: "long", metaSizeMultiplier: 1.0, metaKind: "mlp", metaVersion: "v2",
    });
    const skip = captured.recordSkipContexts.find((s) => s.reason === "portfolio_sector_cap");
    assert.ok(skip, `expected portfolio_sector_cap, got: ${captured.recordSkipContexts.map((s) => s.reason).join(",")}`);
    assertPayloadClean("engine no_trade skip ctx", skip.ctx);
    assert.equal(captured.paperTradesInserts.length, 0);
  });

  test("negative control: assertPayloadClean fires on a tainted payload", () => {
    const tainted = {
      agentName: "x",
      newsBias: 0.3,
      nested: { llmEdge: 0.5, deeper: { sentimentScore: 0.7 } },
      arr: [{ geminiVote: 1 }],
      chatgptCall: "yes",
      ai_signal: "buy",
      rawNewsScore: 42,
    };
    assert.throws(() => assertPayloadClean("tainted", tainted), /LLM-derived keys|non-allowlisted keys/);
  });

  test("negative control: assertPayloadClean fires on Strategy Lab benchmark fields (Task #390)", () => {
    const tainted = {
      agentName: "x",
      relativeAlpha14d: -0.02,
      bestBaselineReturn7d: 0.05,
      benchmarkTrust: 0.8,
      strategy_lab_signal: "soft",
      benchmarkScore: 1.0,
      nested: { drawdownRatio: 1.4 },
    };
    assert.throws(
      () => assertPayloadClean("tainted-bench", tainted),
      /LLM-derived keys|non-allowlisted keys/,
    );
  });

  test("negative control: assertPayloadClean does NOT fire on a clean quant-only payload", () => {
    const clean = {
      agentName: "x", coinName: "y", timeframe: "5m", direction: "up",
      confidence: 0.7, probUp: 0.6, probDown: 0.2, probStable: 0.2,
      expectedReturnPct: 5.0, modelVersion: "v1",
      portfolio: {
        equityUsd: 10_000, cashUsd: 10_000,
        openPositions: [{ coinId: "btc", direction: "up", notionalUsd: 100, regimeAtEntry: null, betaToBtc: null }],
      },
      gates: { enabled: true, sector: "alt" },
    };
    assert.doesNotThrow(() => assertPayloadClean("clean", clean));
  });
});
