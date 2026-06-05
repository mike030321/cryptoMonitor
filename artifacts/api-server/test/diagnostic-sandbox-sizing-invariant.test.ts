/**
 * Task #679 — DS-lane invariant property test (real-path edition).
 *
 * Defense-in-depth follow-up to Codex Audit Finding #3 (REJECTED, see
 * `reports/codex-audit-triage-20260501T083546Z.md` §2.3 / §5.4). The
 * audit misread the codebase as if the DS lane were not pinned; in
 * fact `paper-trader.ts` lines 754-921 + 1066-1093 hard-pin every DS
 * trade to `totalValue * 0.005`. This file proves that property by
 * driving the *real* `executePaperTrade` end-to-end with arbitrary
 * supervisory multipliers and asserting the recorded `paper_trades`
 * insert always sizes at exactly `totalValue * 0.005` — byte-for-byte.
 *
 * Sister files already cover slices of this surface:
 *   - `mttm-position-size-override.test.ts` proves the cap-pick math
 *     for the default (non-DS) MTTM 5% lane.
 *   - `diagnostic-sandbox-sizing.test.ts` pins the per-line gate
 *     ordering against the source.
 *   - `mttm-universe-guard-e2e.test.ts` is the harness this file
 *     mirrors for db / skip-tracker / ml-client / agents-registry
 *     mocks (same `--experimental-test-module-mocks` flag the
 *     `decision-engine-parity` workflow already passes).
 *
 * What this file uniquely covers (and `diagnostic-sandbox-sizing.ts`
 * does not):
 *   - Property-grid: a 3780-tuple cartesian over the six supervisory
 *     dimensions called out in the Task #679 spec (confidence,
 *     meta-brain mult, pool fallback penalty, profile size bias,
 *     family / timeframe mult, totalValue). Every accepted-trade
 *     tuple records `positionSize` byte-exactly equal to
 *     `totalValue * 0.005`.
 *   - Non-finite multipliers (NaN / +Inf / -Inf) cannot leak.
 *   - Off-pin (coinId,timeframe) under DS lane trips the runtime
 *     `DS invariant` throw at line 892 — exercised by widening the
 *     cached MTTM universe so the upstream universe-guard at line 547
 *     does not absorb the off-pin signal first.
 *
 * Hard guardrails honored (per task spec):
 *   - No production-code edits (paper-trader.ts / mttm.ts untouched).
 *   - No mocking of `MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT` —
 *     the real exported constant is used.
 *   - No mocking of `_isDsLane` or any internal that would bypass the
 *     gates. The test only sets the public `mttm_diagnostic_sandbox_v1`
 *     cache state and varies the multiplier inputs the production
 *     code already accepts.
 *   - The runtime invariant throw at line 892 is asserted via real
 *     `executePaperTrade` rejection, not via a mirror.
 */
import { before, beforeEach, describe, mock, test } from "node:test";
import assert from "node:assert/strict";

import {
  __setMttmCache,
  invalidateMttmCache,
  slotKey,
  MTTM_DEFAULT_MAX_POSITION_PCT,
  MTTM_DEFAULT_CONSECUTIVE_LOSS_CAP,
  MTTM_DEFAULT_N10_POST_FEE_CAP_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_COIN,
  MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
  MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
  MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
  type MttmConfig,
  type MttmSlot,
} from "../src/lib/mttm";

interface CapturedTrade {
  positionSize: number;
  entryFee: number;
  coinId: string;
  timeframe: string;
}
interface Captured {
  recordSkipContexts: Array<{ reason: string; ctx: unknown }>;
  paperTradesInserts: CapturedTrade[];
}
const captured: Captured = { recordSkipContexts: [], paperTradesInserts: [] };
function resetCaptured(): void {
  captured.recordSkipContexts.length = 0;
  captured.paperTradesInserts.length = 0;
}

interface PortfolioState {
  agentId: number;
  totalValue: number;
  cashBalance: number;
  peakValue: number;
  dayStartValue: number;
  dayStartDate: string;
  totalTrades: number;
  winningTrades: number;
}
interface State {
  portfolio: PortfolioState;
  positions: Array<{ agentId: number; coinId: string; direction: string; positionSize: number; entryRegimeLabel: string | null }>;
  fleetPositions: Array<{ direction: string }>;
  recentTrades: Array<{ pnl: number | null; positionSize: number }>;
  regime: { regime: string; regimeLabel: string; trendBias: string; avgChange24h: number } | null;
}
function makePortfolio(totalValue: number): PortfolioState {
  return {
    agentId: 1,
    totalValue,
    cashBalance: totalValue,
    peakValue: totalValue,
    dayStartValue: totalValue,
    dayStartDate: new Date().toISOString().slice(0, 10),
    totalTrades: 0,
    winningTrades: 0,
  };
}
const STATE: State = {
  portfolio: makePortfolio(10_000),
  positions: [],
  fleetPositions: [],
  recentTrades: [],
  regime: { regime: "bull", regimeLabel: "bull_trend_high_vol", trendBias: "bullish", avgChange24h: 2.5 },
};
function resetState(totalValue: number = 10_000): void {
  STATE.portfolio = makePortfolio(totalValue);
  STATE.positions = [];
  STATE.fleetPositions = [];
  STATE.recentTrades = [];
  STATE.regime = { regime: "bull", regimeLabel: "bull_trend_high_vol", trendBias: "bullish", avgChange24h: 2.5 };
}

/**
 * Mutable knobs that the mocked supervisory inputs read at call time.
 * Each test run sets these BEFORE invoking executePaperTrade so the
 * production code receives the per-tuple multiplier config.
 */
interface MultiplierKnobs {
  familyMult: number;
  sizeBias: number | null;
  pooledFallbackPenalty: number | null;
  benchmarkSensitivity: number | null;
}
const KNOBS: MultiplierKnobs = {
  familyMult: 1.0,
  sizeBias: null,
  pooledFallbackPenalty: null,
  benchmarkSensitivity: null,
};
function setKnobs(p: Partial<MultiplierKnobs>): void {
  Object.assign(KNOBS, p);
}
function resetKnobs(): void {
  KNOBS.familyMult = 1.0;
  KNOBS.sizeBias = null;
  KNOBS.pooledFallbackPenalty = null;
  KNOBS.benchmarkSensitivity = null;
}

function dsModeConfig(opts: {
  /** Extra (off-pin) slots to add to the universe so the upstream
   *  universe-guard does not absorb off-pin signals — required to
   *  reach the cap-stage `DS invariant` throw at paper-trader.ts:892. */
  extraUniverse?: MttmSlot[];
} = {}): MttmConfig {
  // Universe always includes the DS pin (bitcoin, 5m) so on-pin trades
  // pass the universe-guard. Off-pin tests opt into a wider universe
  // via extraUniverse.
  const universe: MttmSlot[] = [
    { coinId: MTTM_DIAGNOSTIC_SANDBOX_COIN, timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME },
    ...(opts.extraUniverse ?? []),
  ];
  const keys = new Set<string>();
  for (const u of universe) keys.add(slotKey(u.coinId, u.timeframe));
  return {
    enabled: true,
    enabledAt: new Date().toISOString(),
    universe,
    maxPositionPct: MTTM_DEFAULT_MAX_POSITION_PCT,
    consecutiveLossCap: MTTM_DEFAULT_CONSECUTIVE_LOSS_CAP,
    n10PostFeeCapPct: MTTM_DEFAULT_N10_POST_FEE_CAP_PCT,
    disableReason: null,
    universeKeys: keys,
    mode: "diagnostic_sandbox",
    diagnosticSandbox: {
      btcVersion: "test-v1",
      drawdownPct: MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      nNegPnl: MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
      coinId: MTTM_DIAGNOSTIC_SANDBOX_COIN,
      timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      fixedPositionPct: MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
    },
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
        if (tbl.__table === "trades") {
          captured.paperTradesInserts.push(v as CapturedTrade);
        }
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

  const realSkipTracker = await import("../src/lib/skip-tracker.ts");
  mock.module("../src/lib/skip-tracker.ts", {
    namedExports: {
      ...realSkipTracker,
      recordSkip: (reason: string, _agent: string, _msg: string, ctx: unknown) => {
        captured.recordSkipContexts.push({ reason, ctx });
      },
    },
  });

  // /ml/decide is the engine-override branch at paper-trader.ts:1009.
  // Every test in this file is structured so the engine call is NOT
  // reached: the property grid uses `atrValue=0` (line 1009 requires
  // `atrValue > 0`); the on-pin / off-pin / non-finite tests pass
  // `quant=undefined` (line 1009 requires a quant payload). Failing
  // hard if the mock is reached catches any future test addition that
  // accidentally crosses the engine boundary — at which point the
  // author must reason explicitly about whether their assertion is
  // about the LOCAL DS sizing path or the engine-override path.
  const realMlClient = await import("../src/lib/ml-client.ts");
  mock.module("../src/lib/ml-client.ts", {
    namedExports: {
      ...realMlClient,
      getMlDecision: async () => {
        throw new Error(
          "ml-client.getMlDecision should not be reached by this test "
          + "— either set atrValue=0 or omit the quant payload so the "
          + "engine-override branch at paper-trader.ts:1009 is skipped.",
        );
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
    info: () => {},
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

  // Promote 5m (DS pin) AND 6h/1h/1d so both on-pin and off-pin tests
  // get past the role gate; production timeframe-roles.json currently
  // marks every timeframe non-trade in this branch.
  const realRoles = await import("../src/lib/timeframe-roles.ts");
  mock.module("../src/lib/timeframe-roles.ts", {
    namedExports: {
      ...realRoles,
      getRoleForTimeframe: (tf: string) =>
        tf === "5m" || tf === "1h" || tf === "6h" || tf === "1d" ? "trade" : "disabled",
    },
  });

  // The supervisory meta-brain shapes sizing through (1) the per-family
  // multiplier and (2) the active directive's `defensive_mode` floor. We
  // mock both so per-test KNOBS.familyMult drives the production
  // multiplier path AND so the clampMetaSizeMultiplier floor stays at
  // its default 0.5 (no defensive directive). Suppression is forced
  // off so it never short-circuits before sizing.
  const realMetaBrain = await import("../src/lib/meta-brain/index.ts");
  mock.module("../src/lib/meta-brain/index.ts", {
    namedExports: {
      ...realMetaBrain,
      getFamilySizeMultiplier: () => KNOBS.familyMult,
      isFamilySuppressed: () => false,
      resolveStrategyFamily: () => "momentum",
      getActiveDirective: () => ({ tick_id: 0, defensive_mode: "off" }),
      isNeutralDirective: () => true,
      bindTradeToTick: () => {},
      peekTickForTrade: () => null,
      sendRecordOutcome: () => {},
    },
  });

  // Maximally-permissive executor profile so the registry/regime/min-
  // confidence gates are clear and the trade flow reaches the DS sizing
  // path. KNOBS.{sizeBias,pooledFallbackPenalty,benchmarkSensitivity}
  // are spliced into the profile at call time so production's
  // `compositeSizeMult` path sees them.
  const realRegistry = await import("../src/lib/agents-registry/index.ts");
  const realCache = await import("../src/lib/agents-registry/cache.ts");
  const baseProfile = realRegistry.getAgentProfile("momentum_core");
  const buildProfile = () => ({
    ...baseProfile,
    preferred_regimes: "all" as const,
    blocked_regimes: [],
    min_confidence: 0.0,
    min_expected_edge_after_costs: -1.0,
    size_bias: KNOBS.sizeBias,
    pooled_fallback_penalty: KNOBS.pooledFallbackPenalty,
    benchmark_sensitivity: KNOBS.benchmarkSensitivity,
    strategy_family: "momentum" as const,
  });
  const cacheLookup = (agentId: number) => {
    if (agentId === 1) return { profile: buildProfile(), dbStatus: "active", agentName: "TestBot" };
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
      ? { agentId: 1, agentName: "TestBot", profile: buildProfile(), dbStatus: "active" }
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
  resetKnobs();
  invalidateMttmCache();
});

/**
 * Drive a single on-pin DS trade through the real `executePaperTrade`.
 * `quant` and `atrValue` are wired so callers can exercise either the
 * pure-LLM path (quant=undefined, atrValue=1.0) OR the quant path
 * with `metaSizeMultiplier` shaping (quant=<payload>, atrValue=0 so
 * the `/ml/decide` engine-override at line 1009 is NOT called and the
 * recorded `positionSize` is the value the LOCAL DS sizing path
 * computed — i.e. the dsPin from the cap-stage at line 884).
 */
async function runOnPinDsTrade(opts: {
  totalValue: number;
  confidence: number;
  quant?: unknown;
  atrValue?: number;
} = { totalValue: 10_000, confidence: 0.7 }): Promise<void> {
  resetState(opts.totalValue);
  invalidateMttmCache();
  __setMttmCache(dsModeConfig());
  await executePaperTrade(
    1,
    "TestBot",
    MTTM_DIAGNOSTIC_SANDBOX_COIN,
    "Bitcoin",
    "up",
    opts.confidence,
    100,
    MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
    1,
    opts.atrValue ?? 1.0,
    opts.quant,
  );
}

describe("Task #679 — DS-lane positionSize invariant under arbitrary multipliers (real path)", () => {
  // Pre-flight sanity: a vanilla on-pin trade with no multiplier
  // shaping must produce exactly one paper_trades insert sized at the
  // pin. If this fails, the harness is broken and downstream property
  // assertions would be meaningless, so check it explicitly first.
  test("on-pin DS trade with neutral multipliers inserts exactly one paper_trades row at totalValue * 0.005", async () => {
    await runOnPinDsTrade();
    assert.equal(
      captured.paperTradesInserts.length,
      1,
      `expected exactly one paper_trades insert; skips=${captured.recordSkipContexts.map(s => s.reason).join(",") || "(none)"}`,
    );
    const t = captured.paperTradesInserts[0];
    assert.strictEqual(
      t.positionSize,
      10_000 * MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
      `expected positionSize === 10000 * 0.005 = 50, got ${t.positionSize}`,
    );
    assert.strictEqual(t.coinId, MTTM_DIAGNOSTIC_SANDBOX_COIN);
    assert.strictEqual(t.timeframe, MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME);
  });

  // ====================================================================
  // PROPERTY GRID — spec-exact cartesian over the six multiplier /
  // value dimensions called out in the Task #679 specification:
  //
  //   confidence:               {0.0, 0.5, 0.7, 0.9, 1.0}                 (5)
  //   meta-brain mult:          {0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0}      (7)
  //   pool fallback penalty:    {0.5, 0.8, 1.0}                           (3)
  //   profile size bias:        {0.5, 1.0, 1.5}                           (3)
  //   timeframe / family mult:  {0.5, 1.0, 1.5}                           (3)
  //   totalValue (USD):         {1_000, 10_000, 100_000, 1_000_000}       (4)
  //
  //   total tuples = 5 * 7 * 3 * 3 * 3 * 4 = 3780  (full cartesian)
  //
  // Each tuple drives `executePaperTrade` end-to-end through the REAL
  // production path. Every multiplier dimension is plumbed through the
  // production code — no mirror, no helper extraction, no source-regex
  // pins, no mocking of `MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT`.
  //
  // Path selected:
  //   - Quant payload is ALWAYS supplied (so `quant.metaSizeMultiplier`
  //     is exercised on the real `compositeSizeMult` line 808 path).
  //   - `quant.fallback = "pooled"` so `profile.pooled_fallback_penalty`
  //     is applied at line 821-828.
  //   - `atrValue = 0` so the `/ml/decide` engine call at line 1009 is
  //     NOT triggered (`atrValue > 0` is required). This keeps the
  //     recorded `positionSize` equal to the value computed by the
  //     LOCAL DS sizing path — which is what this invariant test is
  //     about: regardless of any multiplier the supervisory machinery
  //     hands in, the DS lane's local cap-stage at line 884 must
  //     hard-pin the recorded notional to exactly `totalValue * 0.005`.
  //
  // Assertion:
  //   - For tuples whose downstream gates accept the trade: a
  //     `paper_trades` insert was recorded and its `positionSize` is
  //     byte-exactly `totalValue * MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT`
  //     (assert.strictEqual — no epsilon).
  //   - For tuples whose downstream gates filter the trade (e.g.
  //     `confidence=0.0` fails the confidence floor; `confidence=0.5`
  //     fails the EV floor at the synthetic atrValue=0 fallback): the
  //     invariant trivially holds (no insert, nothing to leak). The
  //     test additionally verifies a recorded skip exists so a silent
  //     no-op cannot masquerade as an invariant pass.
  //
  // Sanity counters at the end of the test guarantee the cartesian
  // actually iterated and at least one tuple per totalValue bucket
  // produced an insert (otherwise the test would be a fancy assert
  // about nothing).
  // ====================================================================
  test("DS pin invariant byte-exact across the spec cartesian (3780 tuples)", async () => {
    const confidences = [0.0, 0.5, 0.7, 0.9, 1.0];
    const metaMults = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0];
    const poolPenalties = [0.5, 0.8, 1.0];
    const sizeBiases = [0.5, 1.0, 1.5];
    const familyMults = [0.5, 1.0, 1.5];
    const totalValues = [1_000, 10_000, 100_000, 1_000_000];

    let runs = 0;
    let inserts = 0;
    let skips = 0;
    const insertsByTv = new Map<number, number>();
    for (const tv of totalValues) insertsByTv.set(tv, 0);

    for (const conf of confidences) {
      for (const mm of metaMults) {
        for (const pp of poolPenalties) {
          for (const sb of sizeBiases) {
            for (const fm of familyMults) {
              for (const tv of totalValues) {
                resetCaptured();
                setKnobs({
                  familyMult: fm,
                  sizeBias: sb,
                  pooledFallbackPenalty: pp,
                  benchmarkSensitivity: null,
                });
                await runOnPinDsTrade({
                  totalValue: tv,
                  confidence: conf,
                  atrValue: 0,
                  quant: {
                    metaSizeMultiplier: mm,
                    probUp: 0.6,
                    probDown: 0.3,
                    probStable: 0.1,
                    expectedReturnPct: 5.0,
                    fallback: "pooled",
                    source: "model",
                  },
                });
                runs++;

                const expected = tv * MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT;
                if (captured.paperTradesInserts.length === 1) {
                  const recorded = captured.paperTradesInserts[0].positionSize;
                  assert.strictEqual(
                    recorded,
                    expected,
                    `tuple (conf=${conf},mm=${mm},pp=${pp},sb=${sb},fm=${fm},tv=${tv}): `
                    + `recorded positionSize ${recorded} !== expected dsPin ${expected}`,
                  );
                  inserts++;
                  insertsByTv.set(tv, (insertsByTv.get(tv) ?? 0) + 1);
                } else if (captured.paperTradesInserts.length === 0) {
                  // Trade was filtered upstream — invariant trivially
                  // holds (no positionSize was recorded so nothing
                  // could leak). Verify a SKIP was recorded so we
                  // know the no-insert is intentional, not a silent
                  // no-op.
                  assert.ok(
                    captured.recordSkipContexts.length >= 1,
                    `tuple (conf=${conf},mm=${mm},pp=${pp},sb=${sb},fm=${fm},tv=${tv}): `
                    + `no insert AND no skip — silent no-op masquerading as invariant pass`,
                  );
                  skips++;
                } else {
                  assert.fail(
                    `tuple (conf=${conf},mm=${mm},pp=${pp},sb=${sb},fm=${fm},tv=${tv}): `
                    + `unexpected ${captured.paperTradesInserts.length} inserts (must be 0 or 1)`,
                  );
                }
              }
            }
          }
        }
      }
    }

    // Cartesian sanity: 5 * 7 * 3 * 3 * 3 * 4 = 3780.
    assert.equal(runs, 3780, `expected 3780 grid runs, got ${runs}`);
    assert.equal(
      inserts + skips,
      runs,
      `every tuple must classify as insert or skip; got inserts=${inserts}, skips=${skips}, runs=${runs}`,
    );
    // Coverage sanity: at least one insert per totalValue bucket so
    // the byte-exact pin assertion was actually checked at every
    // notional scale (1k, 10k, 100k, 1M).
    for (const tv of totalValues) {
      assert.ok(
        (insertsByTv.get(tv) ?? 0) > 0,
        `totalValue=${tv} produced ZERO inserts — the byte-exact pin `
        + `assertion was never checked at this scale; tighten the EV-gate `
        + `or other downstream gate so at least one tuple inserts`,
      );
    }
    // Coverage sanity: the property test must actually exercise the
    // pin assertion on a non-trivial fraction of tuples — otherwise
    // most "passes" are trivial no-insert holds and the property is
    // vacuously satisfied. With confidences {0.9, 1.0} clearing the
    // synthetic-atr EV gate (2/5 = 40%) we expect ≥ 30% of tuples to
    // insert. If this floor regresses the harness needs widening.
    const insertShare = inserts / runs;
    assert.ok(
      insertShare >= 0.3,
      `only ${(insertShare * 100).toFixed(1)}% of tuples produced inserts `
      + `(${inserts}/${runs}); expected ≥ 30% so the byte-exact pin assertion `
      + `is actually checked on a non-trivial sample`,
    );
  });

  // Non-finite multiplier inputs would be a disaster if they leaked
  // into sizing (NaN poisons every downstream price calc). Production's
  // `clampMetaSizeMultiplier` defends with `Number.isFinite` (line 278)
  // and the DS gate at line 875 short-circuits the multiplication.
  // Verify both layers via the real path.
  test("non-finite multipliers (NaN, +Inf, -Inf) cannot leak into DS sizing", async () => {
    const poisons = [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
    for (const poison of poisons) {
      resetCaptured();
      setKnobs({ familyMult: poison, sizeBias: poison, pooledFallbackPenalty: poison, benchmarkSensitivity: null });
      await runOnPinDsTrade({ totalValue: 10_000, confidence: 0.7 });
      assert.equal(
        captured.paperTradesInserts.length,
        1,
        `poison ${poison}: expected one insert, got ${captured.paperTradesInserts.length}; skips=${captured.recordSkipContexts.map(s => s.reason).join(",")}`,
      );
      const recorded = captured.paperTradesInserts[0].positionSize;
      assert.strictEqual(
        recorded,
        50,
        `poison ${poison}: positionSize ${recorded} !== 50`,
      );
      assert.ok(
        Number.isFinite(recorded),
        `poison ${poison}: positionSize ${recorded} is non-finite — DS pin contaminated`,
      );
    }
  });
});

describe("Task #679 — DS-lane runtime invariant: off-pin (coinId,timeframe) throws", () => {
  // The cap-stage throw at paper-trader.ts:892 is reachable only when
  // the upstream universe-guard at line 539 lets an off-pin slot
  // through. In production today the guard always catches off-pin
  // signals first (the universe is locked to bitcoin/5m in DS mode).
  // To exercise the defensive throw we widen the cached universe to
  // include extra slots so the guard passes — then assert the cap-
  // stage rejects the call with the expected `DS invariant` message.
  const offPinCases: Array<{ coin: string; tf: string }> = [
    { coin: "ethereum", tf: "5m" },
    { coin: "bitcoin", tf: "1h" },
    { coin: "bitcoin", tf: "1d" },
    { coin: "bitcoin", tf: "6h" },
    { coin: "ethereum", tf: "1h" },
  ];

  for (const c of offPinCases) {
    test(`off-pin (${c.coin}, ${c.tf}) under DS lane rejects with /DS invariant/`, async () => {
      resetState();
      invalidateMttmCache();
      __setMttmCache(dsModeConfig({
        extraUniverse: [{ coinId: c.coin, timeframe: c.tf }],
      }));
      await assert.rejects(
        () => executePaperTrade(
          1,
          "TestBot",
          c.coin,
          c.coin.toUpperCase(),
          "up",
          0.7,
          100,
          c.tf,
          1,
          1.0,
        ),
        (err: unknown) => {
          assert.ok(err instanceof Error, `expected Error, got ${typeof err}: ${String(err)}`);
          const msg = (err as Error).message;
          assert.match(
            msg,
            /^DS invariant: cap-stage reached with /,
            `expected message starts with "DS invariant: cap-stage reached with ", got ${JSON.stringify(msg)}`,
          );
          assert.match(
            msg,
            new RegExp(`\\(${c.coin},${c.tf}\\)`),
            `expected message includes off-pin (coin,tf) "(${c.coin},${c.tf})", got ${JSON.stringify(msg)}`,
          );
          assert.match(
            msg,
            new RegExp(`pin is \\(${MTTM_DIAGNOSTIC_SANDBOX_COIN},${MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME}\\)`),
            `expected message includes the canonical pin, got ${JSON.stringify(msg)}`,
          );
          return true;
        },
      );
      // No paper_trades insert may have happened.
      assert.equal(
        captured.paperTradesInserts.length,
        0,
        `off-pin (${c.coin},${c.tf}) must NOT insert any paper_trades row`,
      );
    });
  }

  test("on-pin (bitcoin, 5m) under DS lane does NOT throw and DOES insert", async () => {
    resetState();
    invalidateMttmCache();
    __setMttmCache(dsModeConfig());
    await executePaperTrade(
      1,
      "TestBot",
      MTTM_DIAGNOSTIC_SANDBOX_COIN,
      "Bitcoin",
      "up",
      0.7,
      100,
      MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      1,
      1.0,
    );
    assert.equal(
      captured.paperTradesInserts.length,
      1,
      `on-pin must insert; skips=${captured.recordSkipContexts.map(s => s.reason).join(",")}`,
    );
    assert.strictEqual(
      captured.paperTradesInserts[0].positionSize,
      10_000 * MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
    );
  });
});

describe("Task #679 — fixedPositionPct constant pin", () => {
  // The byte-exact assertions above all reference
  // MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT directly. If the spec
  // value ever drifts, the entire suite recomputes — which would mask
  // a real change to the DS pin. Pin the constant explicitly so any
  // change to the spec-mandated 0.005 fails this single assertion
  // first and the task owner is forced to revisit Task #659.
  test("MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT === 0.005", () => {
    assert.strictEqual(
      MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
      0.005,
      "DS-lane fixed-position-pct constant has drifted from the Task #659 spec value (0.005). " +
      "If this is intentional, the property tests above and Task #659 docs must be updated together.",
    );
  });
});
