/**
 * Task #620 — end-to-end paper-trader coverage of the MTTM whitelist guard.
 *
 * The sister file `mttm-universe-guard.test.ts` already proves the
 * primitive (`MttmConfig.universeKeys.has(slotKey(...))`) does the
 * right boolean lookup, and `mttm-position-size-override.test.ts`
 * proves the cap-pick math. Neither one calls `executePaperTrade`,
 * so neither one notices if a future refactor reorders the guards in
 * paper-trader.ts and accidentally lets an out-of-universe candidate
 * past the whitelist or runs the 30% global cap when MTTM is enabled.
 *
 * This file mounts paper-trader.ts against an in-memory mock graph
 * (same approach as `no-llm-fields-runtime.test.ts`) and asserts the
 * actual decision path produces:
 *
 *   1. Out-of-universe + MTTM enabled  → no `paper_trades` insert AND
 *      a recordSkip call with reason "mttm_outside_universe".
 *   2. In-universe + MTTM enabled  → a `paper_trades` insert whose
 *      positionSize is bound by the MTTM 5% cap, not the 30% global
 *      cap (i.e. ~5% of equity, minus the entry fee).
 *
 * Requires `--experimental-test-module-mocks` (the
 * `decision-engine-parity` workflow already passes this flag).
 */
import { before, beforeEach, describe, mock, test } from "node:test";
import assert from "node:assert/strict";

import {
  __setMttmCache,
  invalidateMttmCache,
  slotKey,
  DEFAULT_MTTM_UNIVERSE,
  MTTM_DEFAULT_MAX_POSITION_PCT,
  type MttmConfig,
} from "../src/lib/mttm";
import { MAX_POSITION_PCT } from "../src/lib/trading-constants";

interface Captured {
  recordSkipContexts: Array<{ reason: string; ctx: unknown }>;
  paperTradesInserts: unknown[];
  paperPositionsInserts: unknown[];
  openedLogs: Array<Record<string, unknown>>;
}
const captured: Captured = {
  recordSkipContexts: [],
  paperTradesInserts: [],
  paperPositionsInserts: [],
  openedLogs: [],
};
function resetCaptured(): void {
  captured.recordSkipContexts.length = 0;
  captured.paperTradesInserts.length = 0;
  captured.paperPositionsInserts.length = 0;
  captured.openedLogs.length = 0;
}

interface State {
  portfolio: {
    agentId: number;
    totalValue: number;
    cashBalance: number;
    peakValue: number;
    dayStartValue: number;
    dayStartDate: string;
    totalTrades: number;
    winningTrades: number;
  };
  positions: Array<{ agentId: number; coinId: string; direction: string; positionSize: number; entryRegimeLabel: string | null }>;
  fleetPositions: Array<{ direction: string }>;
  recentTrades: Array<{ pnl: number | null; positionSize: number }>;
  regime: { regime: string; regimeLabel: string; trendBias: string; avgChange24h: number } | null;
}
const STATE: State = {
  portfolio: {
    agentId: 1,
    totalValue: 10_000,
    cashBalance: 10_000,
    peakValue: 10_000,
    dayStartValue: 10_000,
    dayStartDate: new Date().toISOString().slice(0, 10),
    totalTrades: 0,
    winningTrades: 0,
  },
  positions: [],
  fleetPositions: [],
  recentTrades: [],
  regime: { regime: "bull", regimeLabel: "bull_trend_high_vol", trendBias: "bullish", avgChange24h: 2.5 },
};
function resetState(): void {
  STATE.portfolio = {
    agentId: 1,
    totalValue: 10_000,
    cashBalance: 10_000,
    peakValue: 10_000,
    dayStartValue: 10_000,
    dayStartDate: new Date().toISOString().slice(0, 10),
    totalTrades: 0,
    winningTrades: 0,
  };
  STATE.positions = [];
  STATE.fleetPositions = [];
  STATE.recentTrades = [];
  STATE.regime = { regime: "bull", regimeLabel: "bull_trend_high_vol", trendBias: "bullish", avgChange24h: 2.5 };
}

function mttmEnabledConfig(): MttmConfig {
  const keys = new Set<string>();
  for (const u of DEFAULT_MTTM_UNIVERSE) keys.add(slotKey(u.coinId, u.timeframe));
  return {
    enabled: true,
    enabledAt: new Date().toISOString(),
    universe: DEFAULT_MTTM_UNIVERSE,
    maxPositionPct: MTTM_DEFAULT_MAX_POSITION_PCT,
    consecutiveLossCap: 5,
    n10PostFeeCapPct: -0.02,
    disableReason: null,
    universeKeys: keys,
  };
}

type AnyFn = (...args: unknown[]) => unknown;
type ExecuteFn = (
  agentId: number,
  agentName: string,
  coinId: string,
  coinName: string,
  direction: "up" | "down" | "stable",
  confidence: number,
  entryPrice: number,
  timeframe: string,
  predictionId: number,
  atrValue: number,
  quant?: unknown,
) => Promise<void>;

let executePaperTrade: ExecuteFn;

before(async () => {
  const paperPortfoliosTable = { __table: "portfolios", agentId: "agentId" };
  const paperTradesTable = {
    __table: "trades",
    agentId: "agentId",
    coinId: "coinId",
    status: "status",
    closedAt: "closedAt",
    id: "id",
  };
  const paperPositionsTable = {
    __table: "positions",
    agentId: "agentId",
    coinId: "coinId",
    direction: "direction",
  };
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
    select: (proj?: unknown) => {
      projContext = proj ? "fleet" : "default";
      return chain();
    },
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
      db,
      pool: {},
      paperPortfoliosTable,
      paperTradesTable,
      paperPositionsTable,
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

  // recordSkip is captured so we can assert the typed reason and
  // (importantly) the absence of any `paper_trades` insert when the
  // whitelist guard fires.
  const realSkipTracker = await import("../src/lib/skip-tracker.ts");
  mock.module("../src/lib/skip-tracker.ts", {
    namedExports: {
      ...realSkipTracker,
      recordSkip: (reason: string, _agent: string, _msg: string, ctx: unknown) => {
        captured.recordSkipContexts.push({ reason, ctx });
      },
    },
  });

  // The MTTM whitelist guard sits BEFORE the /ml/decide call (we use
  // an LLM trade — no quant payload — so /ml/decide is bypassed
  // entirely and the locally-computed positionSize is what lands in
  // the paper_trades insert). Mock ml-client so any incidental import
  // resolves; the function should never be called for this test.
  const realMlClient = await import("../src/lib/ml-client.ts");
  mock.module("../src/lib/ml-client.ts", {
    namedExports: {
      ...realMlClient,
      getMlDecision: async () => {
        throw new Error("ml-client.getMlDecision should not be invoked for an LLM trade");
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

  // Capture the structured "Paper trade opened" log so we can
  // double-check the position size that the live trader would have
  // logged matches the size that ended up in the paper_trades row.
  const realLogger = await import("../src/lib/logger.ts");
  const loggerStub: Record<string, AnyFn> = {
    info: (obj: unknown, msg?: unknown) => {
      if (msg === "Paper trade opened") captured.openedLogs.push(obj as Record<string, unknown>);
    },
    warn: () => {},
    debug: () => {},
    error: () => {},
    trace: () => {},
    fatal: () => {},
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

  // The live `shared/timeframe-roles.json` currently marks every
  // timeframe non-trade (data-coverage gate) so the role gate fires
  // before our MTTM guard ever runs. Force-trade-promote the two
  // timeframes the MTTM universe pins (6h, 1d) so the gate we are
  // actually testing gets the chance to execute.
  const realRoles = await import("../src/lib/timeframe-roles.ts");
  mock.module("../src/lib/timeframe-roles.ts", {
    namedExports: {
      ...realRoles,
      getRoleForTimeframe: (tf: string) => (tf === "6h" || tf === "1d" ? "trade" : "disabled"),
    },
  });

  // Mirror no-llm-fields-runtime.test.ts: the registry cache is
  // empty in-process, so synthesise a maximally-permissive executor
  // profile for agentId=1 to clear the registry/regime/min-confidence
  // gates and let the trade flow reach the MTTM guard + the cap.
  const realRegistry = await import("../src/lib/agents-registry/index.ts");
  const realCache = await import("../src/lib/agents-registry/cache.ts");
  const fixedProfile = {
    ...realRegistry.getAgentProfile("momentum_core"),
    preferred_regimes: "all" as const,
    blocked_regimes: [],
    min_confidence: 0.0,
    min_expected_edge_after_costs: -1.0,
  };
  const cacheLookup = (agentId: number) => {
    if (agentId === 1) return { profile: fixedProfile, dbStatus: "active", agentName: "TestBot" };
    throw new realCache.AgentNotExecutableError(
      "unknown_agent_id",
      agentId,
      null,
      null,
      `test-mock cache: unknown agent ${agentId}`,
    );
  };
  const tryGet = (agentId: number) =>
    agentId === 1
      ? { agentId: 1, agentName: "TestBot", profile: fixedProfile, dbStatus: "active" }
      : null;
  mock.module("../src/lib/agents-registry/cache.ts", {
    namedExports: {
      ...realCache,
      getCachedProfileForAgentId: cacheLookup,
      tryGetCachedEntry: tryGet,
    },
  });
  mock.module("../src/lib/agents-registry/index.ts", {
    namedExports: {
      ...realRegistry,
      getCachedProfileForAgentId: cacheLookup,
      tryGetCachedEntry: tryGet,
    },
  });

  const mod = await import("../src/lib/paper-trader.ts");
  executePaperTrade = mod.executePaperTrade as ExecuteFn;
});

beforeEach(() => {
  resetCaptured();
  resetState();
  invalidateMttmCache();
});

describe("Task #620 — MTTM whitelist guard end-to-end via executePaperTrade", () => {
  test("MTTM enabled + out-of-universe (bitcoin/6h): trade is rejected with mttm_outside_universe and no paper_trades row is inserted", async () => {
    __setMttmCache(mttmEnabledConfig());

    // bitcoin is NOT one of the 8 promoted MTTM coins, so (bitcoin, 6h)
    // is out-of-universe — the whitelist guard at paper-trader.ts L518
    // must short-circuit before any trade insert.
    await executePaperTrade(
      1,
      "TestBot",
      "bitcoin",
      "Bitcoin",
      "up",
      0.7,
      100,
      "6h",
      1,
      1.0,
    );

    const skip = captured.recordSkipContexts.find((s) => s.reason === "mttm_outside_universe");
    assert.ok(
      skip,
      `expected an mttm_outside_universe skip event, got reasons: ${captured.recordSkipContexts
        .map((s) => s.reason)
        .join(",") || "(none)"}`,
    );
    assert.equal(
      captured.paperTradesInserts.length,
      0,
      "no paper_trades row may be inserted for an out-of-universe MTTM candidate",
    );
    assert.equal(
      captured.paperPositionsInserts.length,
      0,
      "no paper_positions row may be inserted for an out-of-universe MTTM candidate",
    );
  });

  test("MTTM enabled + in-universe (bonk/6h): trade is sized at the 5% MTTM cap, not the 30% global cap", async () => {
    __setMttmCache(mttmEnabledConfig());

    // bonk is in the MTTM universe at 6h, so the whitelist guard
    // passes. With confidence=0.70 the LLM tier (no quant payload, so
    // /ml/decide is skipped) starts the position at $10,000 × 0.22 =
    // $2,200 which is well above BOTH caps; the live trader must then
    // bind it down to the MTTM cap of 5% of equity ($500 minus the
    // entry fee), not the 30% global cap ($3,000).
    await executePaperTrade(
      1,
      "TestBot",
      "bonk",
      "BONK",
      "up",
      0.7,
      100,
      "6h",
      1,
      1.0,
    );

    assert.equal(
      captured.recordSkipContexts.length,
      0,
      `expected no skip events for an in-universe MTTM candidate; got: ${captured.recordSkipContexts
        .map((s) => s.reason)
        .join(",")}`,
    );
    assert.equal(
      captured.paperTradesInserts.length,
      1,
      "expected exactly one paper_trades insert for the in-universe candidate",
    );

    const tradeRow = captured.paperTradesInserts[0] as Record<string, unknown>;
    const positionSize = tradeRow.positionSize as number;
    const entryFee = tradeRow.entryFee as number;
    const mttmCapDollars = STATE.portfolio.totalValue * MTTM_DEFAULT_MAX_POSITION_PCT;
    const globalCapDollars = STATE.portfolio.totalValue * MAX_POSITION_PCT;

    assert.ok(
      positionSize > 0,
      `expected positive positionSize, got ${positionSize}`,
    );
    // The MTTM cap is the upper bound (the entry fee is then deducted
    // from positionSize before insert), so the recorded size must be
    // strictly <= the cap.
    assert.ok(
      positionSize <= mttmCapDollars,
      `positionSize ${positionSize} must be <= the 5% MTTM cap of ${mttmCapDollars}`,
    );
    // And it must clearly NOT be the 30% global cap — if a future
    // refactor swapped MTTM's `maxPositionPct` for `MAX_POSITION_PCT`
    // here, the recorded size would jump up around $2,200 (the pre-cap
    // tieredPositionPct contribution) and this assertion would fail.
    assert.ok(
      positionSize < globalCapDollars * 0.5,
      `positionSize ${positionSize} is suspiciously close to the 30% global cap (${globalCapDollars}); MTTM cap may not have been applied`,
    );
    // Tight expected band: cap = $500, fee = 500 * 0.001 = $0.50, so
    // positionSize lands at exactly $499.50. Allow a $1 envelope so
    // future fee-rounding tweaks don't flap this test.
    assert.ok(
      positionSize >= mttmCapDollars - 1 - entryFee && positionSize <= mttmCapDollars,
      `positionSize ${positionSize} should sit at the 5% MTTM cap (~${mttmCapDollars - entryFee}) within $1 tolerance`,
    );
    assert.equal(
      captured.paperPositionsInserts.length,
      1,
      "expected exactly one paper_positions insert for the in-universe candidate",
    );
  });
});
