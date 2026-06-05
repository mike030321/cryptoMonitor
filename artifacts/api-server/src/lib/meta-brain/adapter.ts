/**
 * Telemetry adapter — per-tick collector + directive cache.
 *
 * Monitor pushes slice telemetry into this collector as predictions are
 * generated during a cycle. At cycle-end, `flushTick()` assembles the
 * 19-field-per-slice + 9-portfolio-field batch, posts to the brain, and
 * caches the returned directive for the *next* tick's sizing.
 *
 * Bounded staleness of one 30s cycle < shortest active timeframe (5m),
 * and the brain's bounded plasticity caps adaptation speed.
 *
 * Paper-trader looks up the active directive at sizing time via
 * `getActiveDirective()`. On trade open, the trade id is bound to the
 * current tick_id via `bindTradeToTick()`. On trade close,
 * `peekTickForTrade()` is used to resolve (without consuming) the
 * binding; the binding is removed only after a successful
 * `record_outcome` callback (or after a TTL-based sweep for stuck
 * bindings).
 *
 * Contract: the adapter never throws on the hot path. If the brain is
 * disabled or unreachable, the active directive is a neutral one and
 * sizing is unaffected.
 */

import { promises as fsp } from "node:fs";
import * as path from "node:path";
import { getAgentProfile as getAgentProfileForFamily } from "../agents-registry";
// Task #550 — pass-through of the per-timeframe role on every slice
// payload. The adapter does NOT change consumption logic — that is a
// separate task — so this is a single resolve call at slice ingest
// time. The Python side will treat a missing field as `trade` for
// back-compat (see SliceRole comment in ./contract.ts).
import { getRoleForTimeframe } from "../timeframe-roles";
import {
  postEvaluate,
  postRecordOutcome,
  type MetaBrainOutcome,
  consumeFallbackCounters,
  consumeLatencySamples,
} from "./client";
import {
  EXECUTING_DIRECTIVE_FIELDS,
  MetaBrainBatch,
  MetaBrainBenchmark,
  MetaBrainDirective,
  MetaBrainSlice,
  MetaBrainPortfolio,
  StrategyFamily,
  isNeutralDirective,
  neutralDirective,
} from "./contract";
import { logger } from "../logger";

// ────────────────────── per-tick collector state ─────────────────────

interface PendingSlice {
  slice: MetaBrainSlice;
}

let pendingSlices: PendingSlice[] = [];
let pendingPortfolio: MetaBrainPortfolio | null = null;
// Task #390 — optional per-cycle Strategy Lab benchmark telemetry.
// Set by monitor.ts immediately before flushTick(). Cleared at flush
// time (consumed exactly once per cycle).
let pendingBenchmark: MetaBrainBenchmark | null = null;
let activeDirective: MetaBrainDirective = neutralDirective("boot");

// Per-cycle counters used by `meta_brain_cycle_stats` log line. Reset
// when consumed at flush time. Suppressed-family count reflects the
// directive currently in effect for sizing.
let recordOutcomeCacheHits = 0;
let recordOutcomeCacheMisses = 0;
let suppressedFamilyTicks = 0;
let defensiveModeTicks: Record<"off" | "soft" | "hard", number> = {
  off: 0,
  soft: 0,
  hard: 0,
};

// trade_id → { tickId, ts }. Bindings live until record_outcome
// succeeds OR a TTL sweep reaps them. TTL is generous (24h) because
// the longest active timeframe is 1d. Bounded by a hard cap as a
// belt-and-suspenders against runaway growth.
interface TickBinding {
  tickId: string;
  createdAt: number;
}
const TRADE_TO_TICK: Map<number, TickBinding> = new Map();
const TRADE_TO_TICK_CAP = 4096;
const TRADE_TO_TICK_TTL_MS = 24 * 60 * 60 * 1000;

// ─────────────────── persistence (best-effort) ───────────────────────
// Tick-cache survivability (Task #381 step 11). The api-server side of
// the binding (a small int → string map) is the cheaper of the two
// caches to persist; the ml-engine side `_tick_cache` holds full
// directive objects and is left in-memory by design (any trades opened
// before an ml-engine restart simply skip the learning loop, which is
// what the brain's bounded plasticity is built for). Document choice.

// Task #381: live runtime state — never tracked in git. Defaults to
// `.cache/meta_brain_state/` at the workspace root (which is already
// gitignored) so a stale snapshot can never be committed by accident.
// Deployments that need a custom path can set META_BRAIN_STATE_DIR.
const STATE_DIR =
  process.env.META_BRAIN_STATE_DIR ||
  path.resolve(process.cwd(), "..", "..", ".cache", "meta_brain_state");
const TRADE_TO_TICK_PATH = path.join(STATE_DIR, "trade_to_tick.json");

let dirty = false;
let savePromise: Promise<void> = Promise.resolve();
function markDirty(): void {
  dirty = true;
}

async function persistBindings(): Promise<void> {
  if (!dirty) return;
  dirty = false;
  try {
    await fsp.mkdir(STATE_DIR, { recursive: true });
    const snapshot = Object.fromEntries(
      Array.from(TRADE_TO_TICK.entries()).map(([k, v]) => [String(k), v]),
    );
    await fsp.writeFile(
      TRADE_TO_TICK_PATH,
      JSON.stringify(snapshot),
      "utf-8",
    );
  } catch (err) {
    logger.debug(
      { err: String(err) },
      "meta-brain: tick binding snapshot failed (non-blocking)",
    );
  }
}

function schedulePersist(): void {
  // Coalesce writes — at most one in-flight at a time.
  savePromise = savePromise.then(persistBindings).catch(() => undefined);
}

let hydrated = false;
export async function hydrateBindings(): Promise<void> {
  if (hydrated) return;
  hydrated = true;
  try {
    const raw = await fsp.readFile(TRADE_TO_TICK_PATH, "utf-8");
    const snapshot = JSON.parse(raw) as Record<string, TickBinding>;
    const now = Date.now();
    let restored = 0;
    let expired = 0;
    for (const [k, v] of Object.entries(snapshot)) {
      const id = Number(k);
      if (!Number.isFinite(id) || !v?.tickId) continue;
      if (now - (v.createdAt ?? 0) > TRADE_TO_TICK_TTL_MS) {
        expired++;
        continue;
      }
      TRADE_TO_TICK.set(id, v);
      restored++;
    }
    logger.info(
      { restored, expired, path: TRADE_TO_TICK_PATH },
      "meta-brain: restored trade→tick bindings from disk",
    );
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code;
    if (code === "ENOENT") {
      logger.info(
        { path: TRADE_TO_TICK_PATH },
        "meta-brain: no trade→tick snapshot to restore (fresh start)",
      );
    } else {
      logger.warn(
        { err: String(err) },
        "meta-brain: trade→tick snapshot restore failed",
      );
    }
  }
}

function sweepExpiredBindings(): { reaped: number } {
  const now = Date.now();
  let reaped = 0;
  for (const [k, v] of TRADE_TO_TICK.entries()) {
    if (now - v.createdAt > TRADE_TO_TICK_TTL_MS) {
      TRADE_TO_TICK.delete(k);
      reaped++;
    }
  }
  if (reaped > 0) markDirty();
  return { reaped };
}

// ────────────────────────── family mapping ──────────────────────────

/**
 * LEGACY name-based family resolver. Retained ONLY for the contract
 * test in `meta-brain-family-mapping.test.ts`, which pins the
 * historical behaviour for any pre-#468 row that has not yet been
 * swept onto a registry profile_id.
 *
 * Live trade-path code (monitor.ts, paper-trader.ts) must call
 * `resolveStrategyFamilyForProfile(agent.profileId)` instead — that
 * function throws on unknown ids, which is the Task #468 contract.
 */
export function resolveStrategyFamily(
  agentPersonality: string | null | undefined,
  specialistKind?: string | null,
): StrategyFamily {
  const p = (agentPersonality ?? "").toLowerCase();
  const s = (specialistKind ?? "").toLowerCase();
  if (s.includes("momentum") || p.includes("momentum") || p.includes("trend")) {
    return "momentum";
  }
  if (
    s.includes("mean") ||
    p.includes("contrarian") ||
    p.includes("reversion") ||
    p.includes("revert")
  ) {
    return "mean_reversion";
  }
  if (s.includes("breakout") || p.includes("breakout")) {
    return "breakout";
  }
  if (s.includes("vol") || p.includes("scalper") || p.includes("vol")) {
    return "volatility_forecaster";
  }
  return "baseline";
}

/**
 * Task #468 — resolve strategy_family from a registry profile_id.
 * Throws on unknown ids (via getAgentProfile). The returned family is
 * guaranteed to be a member of `STRATEGY_FAMILIES` because the
 * registry's Zod schema enforces it at module load.
 *
 * The registry imports STRATEGY_FAMILIES from `meta-brain/contract`
 * (not from this `adapter.ts`), so this top-level import is safe and
 * does not introduce a cycle.
 */
export function resolveStrategyFamilyForProfile(
  profile_id: string | null | undefined,
): StrategyFamily {
  return getAgentProfileForFamily(profile_id).strategy_family;
}

// ─────────────────── slice + portfolio ingestion ─────────────────────

// Numeric fields accept either a real value or `null`. `null` means
// "we cannot truthfully compute this yet" — the adapter records a
// `missing:<field>` flag and substitutes 0.0 on the wire (since the
// vendored Python dataclass requires float). The brain's trust updater
// is expected to down-weight learning when missing flags are present.

export interface CollectSliceArgs {
  coin: string;
  timeframe: string;
  strategy_family: StrategyFamily;
  edge: number;
  confidence: number;
  calibrated_confidence: number;
  recent_accuracy: number | null;
  pnl_state: number | null;
  drawdown_state: number | null;
  disagreement: number;
  regime: string;
  volatility: number;
  correlation_shift: number | null;
  exposure: number | null;
  turnover: number | null;
  slippage_bps: number | null;
  prediction_error: number | null;
  risk_score: number;
  anomaly_flags?: string[];
}

function nullable(
  raw: number | null,
  field: string,
  flags: string[],
  clamp01 = false,
): number {
  if (raw === null || raw === undefined || !Number.isFinite(raw)) {
    flags.push(`missing:${field}`);
    return 0.0;
  }
  return clamp01 ? Math.max(0, Math.min(1, raw)) : raw;
}

export function collectSlice(args: CollectSliceArgs): void {
  const flags = [...(args.anomaly_flags ?? [])];

  const slice: MetaBrainSlice = {
    coin: args.coin,
    timeframe: args.timeframe,
    strategy_family: args.strategy_family,
    // Task #550 — pass-through. Resolved here so the field is always
    // present on the wire payload. If the JSON is missing/malformed
    // the loader returns `disabled` for every TF (fail-closed).
    slice_role: getRoleForTimeframe(args.timeframe),
    edge: safe(args.edge),
    confidence: safe01(args.confidence),
    calibrated_confidence: safe01(args.calibrated_confidence),
    risk_score: safe01(args.risk_score),
    recent_accuracy: nullable(args.recent_accuracy, "recent_accuracy", flags, true),
    pnl_state: nullable(args.pnl_state, "pnl_state", flags),
    drawdown_state: nullable(args.drawdown_state, "drawdown_state", flags),
    disagreement: safe01(args.disagreement),
    prediction_error: nullable(args.prediction_error, "prediction_error", flags),
    regime: args.regime ?? "unknown",
    volatility: safe(args.volatility),
    correlation_shift: nullable(args.correlation_shift, "correlation_shift", flags),
    exposure: nullable(args.exposure, "exposure", flags, true),
    turnover: nullable(args.turnover, "turnover", flags),
    slippage_bps: nullable(args.slippage_bps, "slippage_bps", flags),
    anomaly_flags: flags,
  };
  pendingSlices.push({ slice });
}

export interface CollectPortfolioArgs {
  total_drawdown: number | null;
  realized_vol: number | null;
  concentration: number | null;
  leverage: number | null;
  liquidity_stress: number | null;
  correlation_shift: number | null;
  active_risk_budget: number | null;
  kill_switch_distance: number | null;
  anomaly_flags?: string[];
}

/**
 * Task #390 — stash a Strategy Lab benchmark snapshot for inclusion in
 * the next `flushTick` batch. Pure governance signal: the brain may
 * use it to soft-clamp defensive_mode and update the synthetic
 * `benchmark` family in trust_by_family. Never reaches `/ml/decide`,
 * predictor input, or any trade-decision payload.
 */
export function setBenchmarkTelemetry(b: MetaBrainBenchmark | null): void {
  pendingBenchmark = b;
}

export function setPortfolioTelemetry(args: CollectPortfolioArgs): void {
  const flags = [...(args.anomaly_flags ?? [])];
  pendingPortfolio = {
    total_drawdown: nullable(args.total_drawdown, "total_drawdown", flags),
    realized_vol: nullable(args.realized_vol, "realized_vol", flags),
    concentration: nullable(args.concentration, "concentration", flags, true),
    leverage: nullable(args.leverage, "leverage", flags),
    liquidity_stress: nullable(args.liquidity_stress, "liquidity_stress", flags, true),
    correlation_shift: nullable(args.correlation_shift, "correlation_shift", flags),
    active_risk_budget: nullable(args.active_risk_budget, "active_risk_budget", flags, true),
    kill_switch_distance: nullable(args.kill_switch_distance, "kill_switch_distance", flags, true),
    anomaly_flags: flags,
  };
}

// ────────────────────── flush / evaluate ──────────────────────────

export async function flushTick(): Promise<void> {
  // Always sweep stale bindings; cheap.
  sweepExpiredBindings();

  if (pendingSlices.length === 0) {
    emitCycleStats(0);
    return;
  }
  const batch: MetaBrainBatch = {
    slices: pendingSlices.map((p) => p.slice),
    portfolio: pendingPortfolio ?? neutralPortfolio(),
    timestamp: new Date().toISOString(),
    // Task #390 — explicit benchmark contract:
    //   - non-stale block → attach the full struct
    //   - stale or absent → attach `null`
    // The ml-engine treats both `null` and a `stale: true` block as
    // "no signal" (the brain behaves identically to the pre-#390
    // path), but we send the field every cycle so the wire shape is
    // stable and the brain's payload schema is unambiguous.
    benchmark:
      pendingBenchmark && !pendingBenchmark.stale ? pendingBenchmark : null,
  };
  const sliceCount = batch.slices.length;
  // Reset pending state BEFORE the network call so a slow brain can
  // never double-count a tick's slices into a later batch.
  pendingSlices = [];
  pendingPortfolio = null;
  pendingBenchmark = null;

  try {
    const directive = await postEvaluate(batch);
    assertExecutingSurface(directive);
    activeDirective = directive;
    if (!directive.tick_id.startsWith("neutral:")) {
      logger.info(
        {
          tickId: directive.tick_id,
          sliceCount,
          trustFamilies: Object.keys(directive.trust_multiplier).length,
          defensiveMode: directive.defensive_mode,
          suppressedFamilies: directive.suppressed_families.length,
          shadow: directive.tick_id.startsWith("shadow:"),
        },
        "meta-brain: directive cached for next tick",
      );
    }
  } catch (err) {
    activeDirective = neutralDirective("flush_threw");
    logger.warn({ err: String(err) }, "meta-brain: flushTick threw");
  }

  // Bookkeeping for the next cycle's stats line.
  defensiveModeTicks[activeDirective.defensive_mode] += 1;
  if (
    activeDirective.suppress_signal ||
    activeDirective.suppressed_families.length > 0 ||
    activeDirective.paused_slices.length > 0
  ) {
    suppressedFamilyTicks += 1;
  }
  emitCycleStats(sliceCount);
}

// Task #381: pin the executing surface. The adapter / paper-trader /
// monitor are only allowed to read the fields named in
// EXECUTING_DIRECTIVE_FIELDS for sizing or gating; everything else
// (caution_level, exploration_budget, reason_codes, tick_id) is
// observability-only. This guard does not enforce the read pattern
// directly — it can't — but it asserts the contract surface is what
// we expect at runtime so any future schema drift surfaces here
// instead of silently changing executing behaviour. If the brain
// adds a new field we have to make an explicit decision: extend the
// executing surface (and update this list + tests) or document it
// as observability-only.
function assertExecutingSurface(d: MetaBrainDirective): void {
  for (const k of EXECUTING_DIRECTIVE_FIELDS) {
    if (!(k in d)) {
      logger.error(
        { missing: k, subsystem: "meta-brain" },
        "meta-brain directive missing executing-surface field; falling back",
      );
      throw new Error(`directive missing executing-surface field: ${k}`);
    }
  }
}

// Task #384 — keep last cycle's stats + a small ring buffer so the
// dashboard can show live trends without scraping logs. Bounded to
// 60 entries (~30 min at a 30s tick); pure observability, never
// read by the trading path.
export interface MetaBrainCycleStats {
  sliceCount: number;
  bindings: number;
  cacheHits: number;
  cacheMisses: number;
  suppressedTick: number;
  defensiveOff: number;
  defensiveSoft: number;
  defensiveHard: number;
  fallbacksByCause: Record<string, number>;
  evalLatencyP50Ms: number;
  evalLatencyP95Ms: number;
  evalCount: number;
  timeoutCount: number;
  activeTickId: string;
  activeDefensiveMode: "off" | "soft" | "hard";
  activeSuppressedFamilies: string[];
  activePausedSlices: string[];
  emittedAt: string;
}

const CYCLE_STATS_HISTORY_CAP = 60;
let lastCycleStats: MetaBrainCycleStats | null = null;
const cycleStatsHistory: MetaBrainCycleStats[] = [];

export function getCycleStats(): {
  last: MetaBrainCycleStats | null;
  history: MetaBrainCycleStats[];
} {
  return {
    last: lastCycleStats,
    history: cycleStatsHistory.slice(),
  };
}

function emitCycleStats(sliceCount: number): void {
  const fallbacks = consumeFallbackCounters();
  const latencies = consumeLatencySamples();
  const stats: MetaBrainCycleStats = {
    sliceCount,
    bindings: TRADE_TO_TICK.size,
    cacheHits: recordOutcomeCacheHits,
    cacheMisses: recordOutcomeCacheMisses,
    suppressedTick: suppressedFamilyTicks,
    defensiveOff: defensiveModeTicks.off,
    defensiveSoft: defensiveModeTicks.soft,
    defensiveHard: defensiveModeTicks.hard,
    fallbacksByCause: fallbacks,
    evalLatencyP50Ms: latencies.p50,
    evalLatencyP95Ms: latencies.p95,
    evalCount: latencies.count,
    timeoutCount: fallbacks.fetch_failed ?? 0,
    activeTickId: activeDirective.tick_id,
    activeDefensiveMode: activeDirective.defensive_mode,
    activeSuppressedFamilies: [...activeDirective.suppressed_families],
    activePausedSlices: [...activeDirective.paused_slices],
    emittedAt: new Date().toISOString(),
  };
  logger.info(
    { ...stats, subsystem: "meta-brain" },
    "meta_brain_cycle_stats",
  );
  lastCycleStats = stats;
  cycleStatsHistory.push(stats);
  while (cycleStatsHistory.length > CYCLE_STATS_HISTORY_CAP) {
    cycleStatsHistory.shift();
  }
  // Reset per-cycle counters.
  recordOutcomeCacheHits = 0;
  recordOutcomeCacheMisses = 0;
  suppressedFamilyTicks = 0;
  defensiveModeTicks = { off: 0, soft: 0, hard: 0 };
}

export function getActiveDirective(): MetaBrainDirective {
  return activeDirective;
}

/**
 * Per-family size multiplier derived from the active directive. Callers
 * combine this with the existing metaSizeMultiplier clamp at the
 * sizing hook in paper-trader.ts. Returns 1.0 (no effect) when the
 * directive is neutral / shadow / the family is not represented.
 *
 * Allocation scaling (Task #381 step 7): the previous formulation was
 * `trust × alloc × N_families`, which under any non-uniform softmax
 * collapsed losers to the 0.5 floor and saturated winners at the 1.5
 * cap. Replaced with `trust × (alloc / alloc_mean)` so a small
 * allocation shift produces a small sizing shift.
 */
export function getFamilySizeMultiplier(family: StrategyFamily): number {
  const d = activeDirective;
  if (isNeutralDirective(d)) return 1.0;

  const trust = d.trust_multiplier[family];
  const alloc = d.allocation_weight[family];
  if (!Number.isFinite(trust) || !Number.isFinite(alloc)) return 1.0;

  const allocValues = Object.values(d.allocation_weight);
  const allocMean =
    allocValues.length > 0
      ? allocValues.reduce((a, b) => a + b, 0) / allocValues.length
      : 1;
  // Smooth allocation factor: 1.0 when family at mean weight, > 1
  // when over-allocated, < 1 when under-allocated. Bounded by the
  // softmax temperature in the brain so excursions are gentle.
  const allocFactor = allocMean > 0 ? alloc / allocMean : 1;

  let mult = trust * allocFactor;
  // Defensive-mode shaping. "soft" damps; "hard" zeroes the
  // multiplier so the downstream clampMetaSizeMultiplier (with
  // defensive_mode === "hard" branch) lowers the floor to 0 and the
  // trade is suppressed via composition. See clampMetaSizeMultiplier
  // in paper-trader.ts and Task #381 step 6 doc.
  if (d.defensive_mode === "soft") mult *= 0.7;
  else if (d.defensive_mode === "hard") mult = 0.0;
  return mult;
}

/**
 * True iff the active directive suppresses this family (or the brain
 * emitted a global suppress_signal or a paused_slices entry covering
 * this slice). The paper-trader gate routes this through the existing
 * skip path — no new execution branch is introduced.
 *
 * Returns false for neutral / shadow directives so suppression can
 * never fire while the brain is disabled or in shadow mode.
 */
export function isFamilySuppressed(
  family: StrategyFamily,
  coin?: string,
  timeframe?: string,
): boolean {
  const d = activeDirective;
  if (isNeutralDirective(d)) return false;
  if (d.suppress_signal) return true;
  if (d.suppressed_families.includes(family)) return true;
  if (coin && timeframe) {
    const slice = `${coin}/${timeframe}`;
    if (d.paused_slices.includes(slice)) return true;
  }
  return false;
}

// ───────────────────────── open/close bindings ───────────────────────

export function bindTradeToTick(tradeId: number, tickId: string): void {
  // Only `neutral:*` bindings are skipped. `shadow:*` IS preserved so
  // that record_outcome can still feed the brain's learning loop while
  // sizing remains untouched.
  if (tickId.startsWith("neutral:")) return;
  TRADE_TO_TICK.set(tradeId, { tickId, createdAt: Date.now() });
  while (TRADE_TO_TICK.size > TRADE_TO_TICK_CAP) {
    const firstKey = TRADE_TO_TICK.keys().next().value;
    if (firstKey === undefined) break;
    TRADE_TO_TICK.delete(firstKey);
  }
  markDirty();
  schedulePersist();
}

/**
 * Read the tick_id WITHOUT consuming the binding. The caller is
 * expected to call `clearTickBinding(tradeId)` only after a successful
 * `record_outcome` round-trip (or to leave the binding for the TTL
 * sweep on permanent failure).
 *
 * The previous `resolveTickForTrade` consumed on read, making
 * record_outcome retries impossible after the first failure. Task #381
 * step 5 fixes that.
 */
export function peekTickForTrade(tradeId: number): string | undefined {
  const binding = TRADE_TO_TICK.get(tradeId);
  return binding?.tickId;
}

export function clearTickBinding(tradeId: number): void {
  if (TRADE_TO_TICK.delete(tradeId)) {
    markDirty();
    schedulePersist();
  }
}

/**
 * Backwards-compatible alias for `peekTickForTrade` to keep external
 * callers (and re-exports from index.ts) working during the refactor.
 * NOTE: this no longer consumes the binding; callers must explicitly
 * call `clearTickBinding` after `record_outcome` succeeds.
 */
export function resolveTickForTrade(tradeId: number): string | undefined {
  return peekTickForTrade(tradeId);
}

export async function sendRecordOutcome(
  payload: MetaBrainOutcome,
  opts: { tradeId?: number } = {},
): Promise<void> {
  const ok = await postRecordOutcome(payload);
  if (ok) {
    recordOutcomeCacheHits += 1;
    if (opts.tradeId !== undefined) clearTickBinding(opts.tradeId);
  } else {
    recordOutcomeCacheMisses += 1;
  }
}

// ───────────────────────── utility helpers ───────────────────────────

function safe(x: number): number {
  return Number.isFinite(x) ? x : 0.0;
}
function safe01(x: number): number {
  if (!Number.isFinite(x)) return 0.0;
  return Math.max(0, Math.min(1, x));
}

function neutralPortfolio(): MetaBrainPortfolio {
  return {
    total_drawdown: 0,
    realized_vol: 0,
    concentration: 0,
    leverage: 0,
    liquidity_stress: 0,
    correlation_shift: 0,
    active_risk_budget: 1,
    kill_switch_distance: 1,
    anomaly_flags: ["missing:portfolio_snapshot"],
  };
}

// Test-only export. Resets collector + cache between tests.
// NOTE: any future per-cycle field added to the adapter (set during a cycle
// and cleared on flush) must also be reset here to prevent state from one
// test bleeding into the next.
export function __resetAdapterState(): void {
  pendingSlices = [];
  pendingPortfolio = null;
  pendingBenchmark = null;
  activeDirective = neutralDirective("reset");
  TRADE_TO_TICK.clear();
  recordOutcomeCacheHits = 0;
  recordOutcomeCacheMisses = 0;
  suppressedFamilyTicks = 0;
  defensiveModeTicks = { off: 0, soft: 0, hard: 0 };
  hydrated = false;
  dirty = false;
}

export function __setActiveDirectiveForTest(d: MetaBrainDirective): void {
  activeDirective = d;
}
