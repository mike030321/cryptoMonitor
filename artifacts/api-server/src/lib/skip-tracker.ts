import { gte, lt, desc } from "drizzle-orm";
import { db, skipEventsTable } from "@workspace/db";
import { logger } from "./logger";

export type SkipReason =
  | "confidence_below_threshold"
  | "counter_trend_regime"
  | "daily_loss_limit"
  | "drawdown_halt"
  | "risk_recheck_halt"
  | "consecutive_losses"
  | "fee_gate_tp_floor"
  | "fee_gate_ev"
  | "both_models_down"
  | "no_trade_zone"
  | "single_model_penalty"
  | "fleet_direction_imbalance"
  | "quant_ev_below_costs"
  | "quant_meta_abstain"
  | "horizon_disabled_weak_signal"
  // Phase 4 — meta-model first-class abstain reasons. The meta-model is
  // the new gate; these enumerations let the dashboard show WHY the gate
  // demoted a candidate to no-trade so operators can validate the call.
  | "meta_no_trade_low_edge"
  | "meta_no_trade_bad_regime_fit"
  | "meta_no_trade_specialist_disagreement"
  | "meta_no_trade_low_calibration"
  // Phase 5 — fleet-level portfolio constraints (sector cap, correlated
  // exposure, beta-to-BTC, regime budget). Mirror of the gates the unified
  // Python decision engine applies in the offline backtester.
  | "portfolio_sector_cap"
  | "portfolio_correlated_exposure"
  | "portfolio_beta_cap"
  | "portfolio_regime_budget"
  | "portfolio_constraint"
  // Task #381 — meta-brain supervisory suppression. Fires when the
  // brain emits suppress_signal, family suppression for this slice's
  // family, or a paused_slices entry covering this (coin, timeframe).
  // Active only when META_BRAIN_ENABLED=1 (shadow mode never fires
  // suppression because the active directive is neutral).
  | "meta_brain_suppress"
  // Task #468 — registry-driven gates. `agent_not_executable` fires
  // when the agent's profile has executes=false (baseline / disabled
  // / quarantine_review / legacy_archived). `agent_blocked_regime`
  // fires when the live regime is in the profile's blocked_regimes
  // list or absent from a non-"all" preferred_regimes list.
  | "agent_not_executable"
  | "agent_blocked_regime"
  // Task #614 — Minimum Truthful Trading Mode (MTTM). Fires when MTTM is
  // enabled and the (coin, timeframe) is not on the pinned 16-slot
  // whitelist. Self-contained gate; turning MTTM off restores prior
  // behaviour 1:1.
  | "mttm_outside_universe"
  // Task #659 — DS lane requires exact 0.5% sizing. If cash can't cover
  // the pin we skip rather than silently shrink.
  | "ds_insufficient_cash"
  // Task #659 — DS lane locks the universe to bitcoin/5m. Any signal
  // for a different (coin,timeframe) is recorded under this strict
  // bucket so the operator audit trail distinguishes DS-lane lockdown
  // skips from generic 16-slot universe skips.
  | "diagnostic_universe_locked";

export interface SkipEvent {
  ts: number;
  reason: SkipReason;
  agentName: string;
  agentId: number | null;
  coinId: string | null;
  message: string;
  details: Record<string, unknown>;
}

const REASON_LABELS: Record<SkipReason, string> = {
  confidence_below_threshold: "Confidence below 0.40",
  counter_trend_regime: "Counter-trend in confirmed regime",
  daily_loss_limit: "Daily loss limit hit",
  drawdown_halt: "Max drawdown halt",
  risk_recheck_halt: "Risk re-check halt",
  consecutive_losses: "3 consecutive losses on coin",
  fee_gate_tp_floor: "TP distance below fee floor",
  fee_gate_ev: "Expected value below fee threshold",
  both_models_down: "Both AI models unavailable",
  no_trade_zone: "No-trade zone (model disagreement)",
  single_model_penalty: "Single-model penalty (no consensus)",
  fleet_direction_imbalance: "Fleet correlation brake (too many on same side)",
  quant_ev_below_costs: "Quant EV below round-trip cost",
  quant_meta_abstain: "Specialist meta-gate abstained",
  horizon_disabled_weak_signal: "Horizon disabled (weak directional accuracy)",
  meta_no_trade_low_edge: "Meta-model abstain — edge below cost cushion",
  meta_no_trade_bad_regime_fit: "Meta-model abstain — bad regime fit",
  meta_no_trade_specialist_disagreement: "Meta-model abstain — specialists disagree",
  meta_no_trade_low_calibration: "Meta-model abstain — low calibration",
  portfolio_sector_cap: "Portfolio sector exposure cap hit",
  portfolio_correlated_exposure: "Correlated exposure cap hit",
  portfolio_beta_cap: "Portfolio beta-to-BTC cap hit",
  portfolio_regime_budget: "Regime exposure budget hit",
  portfolio_constraint: "Portfolio constraint hit",
  meta_brain_suppress: "Meta-brain supervisory suppression",
  agent_not_executable: "Agent profile cannot trade (baseline/disabled/archived)",
  agent_blocked_regime: "Agent profile blocked / non-preferred for current regime",
  mttm_outside_universe: "MTTM on — slot not in 16-slot universe",
  ds_insufficient_cash: "DS lane — cash below 0.5% sizing pin",
  diagnostic_universe_locked:
    "Diagnostic Sandbox — universe locked to bitcoin/5m, signal off-scope",
};

const DEFAULT_RETENTION_DAYS = 7;
function getRetentionMs(): number {
  const raw = process.env["SKIP_EVENTS_RETENTION_DAYS"];
  const parsed = raw !== undefined ? Number(raw) : NaN;
  const days = Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_RETENTION_DAYS;
  return days * 24 * 60 * 60 * 1000;
}
const PRUNE_INTERVAL_MS = 60 * 60 * 1000;
let lastPruneAt = 0;

async function pruneIfDue(): Promise<void> {
  const now = Date.now();
  if (now - lastPruneAt < PRUNE_INTERVAL_MS) return;
  lastPruneAt = now;
  try {
    const cutoff = new Date(now - getRetentionMs());
    await db.delete(skipEventsTable).where(lt(skipEventsTable.ts, cutoff));
  } catch (err) {
    logger.error({ err }, "skip-tracker: prune failed");
  }
}

export interface RecordSkipOptions {
  agentId?: number | null;
  coinId?: string | null;
}

// Module-level health surface so silent DB failures from `recordSkip` are
// visible to operators via the monitoring-status endpoint. The trade-loop
// callers still use fire-and-forget semantics (skip-event persistence is not
// on the critical path of executing a trade), but a sustained failure now
// shows up as a non-zero `skipPersistFailures` on the dashboard instead of
// being swallowed by a single log line per occurrence.
let _skipPersistFailures = 0;
let _skipPersistSuccesses = 0;
let _lastSkipPersistError: string | null = null;
let _lastSkipPersistErrorAt: Date | null = null;

export interface SkipPersistHealth {
  failures: number;
  successes: number;
  lastError: string | null;
  lastErrorAt: string | null;
}

export function getSkipPersistHealth(): SkipPersistHealth {
  return {
    failures: _skipPersistFailures,
    successes: _skipPersistSuccesses,
    lastError: _lastSkipPersistError,
    lastErrorAt: _lastSkipPersistErrorAt?.toISOString() ?? null,
  };
}

/**
 * Persist a skip event. Returns a Promise so callers that care about the
 * write can `await` it (trading-path callers typically do not — they treat
 * the skip as already recorded once `logger.info` fires). Failures bump a
 * module-level counter exposed by `getSkipPersistHealth()` so silent DB
 * drops surface on the monitoring-status endpoint instead of vanishing.
 */
export async function recordSkip(
  reason: SkipReason,
  agentName: string,
  message: string,
  details: Record<string, unknown> = {},
  options: RecordSkipOptions = {},
): Promise<void> {
  const mergedDetails = { ...details, skipReason: reason };
  const ts = new Date();
  const agentId = options.agentId ?? null;
  const coinId = options.coinId ?? null;
  logger.info({ ...mergedDetails, agentName, agentId, coinId }, message);

  try {
    await db.insert(skipEventsTable).values({
      ts,
      reason,
      agentName,
      agentId,
      coinId,
      message,
      details: mergedDetails,
    });
    await pruneIfDue();
    _skipPersistSuccesses++;
  } catch (err) {
    _skipPersistFailures++;
    _lastSkipPersistError = err instanceof Error ? err.message : String(err);
    _lastSkipPersistErrorAt = new Date();
    logger.error({ err, reason, agentName }, "skip-tracker: failed to persist skip event");
    // Intentionally swallow: trade-path callers do not (and should not) await
    // this write. The failure is now visible via getSkipPersistHealth().
  }
}

export interface SkipReasonsSummary {
  windowMs: number;
  generatedAt: string;
  totalSkips: number;
  byReason: Array<{
    reason: SkipReason;
    label: string;
    count: number;
    byAgent: Array<{ agentName: string; count: number }>;
    recent: SkipEvent[];
  }>;
}

function rowToEvent(row: typeof skipEventsTable.$inferSelect): SkipEvent {
  return {
    ts: row.ts.getTime(),
    reason: row.reason as SkipReason,
    agentName: row.agentName,
    agentId: row.agentId ?? null,
    coinId: row.coinId ?? null,
    message: row.message,
    details: (row.details ?? {}) as Record<string, unknown>,
  };
}

export async function getSkipsForReason(
  reason: SkipReason,
  windowMs: number = 24 * 60 * 60 * 1000,
): Promise<SkipEvent[]> {
  const cutoff = new Date(Date.now() - windowMs);
  try {
    const rows = await db
      .select()
      .from(skipEventsTable)
      .where(gte(skipEventsTable.ts, cutoff));
    return rows.filter((r) => r.reason === reason).map(rowToEvent);
  } catch (err) {
    logger.error({ err, reason }, "skip-tracker: failed to read skip events for reason");
    return [];
  }
}

export async function getSkipsInBucket(
  reason: SkipReason,
  bucketTs: number,
  bucketMs: number,
): Promise<SkipEvent[]> {
  const start = new Date(bucketTs);
  const end = new Date(bucketTs + bucketMs);
  try {
    const rows = await db
      .select()
      .from(skipEventsTable)
      .where(gte(skipEventsTable.ts, start))
      .orderBy(skipEventsTable.ts);
    return rows
      .filter((r) => r.reason === reason && r.ts < end)
      .map(rowToEvent);
  } catch (err) {
    logger.error({ err, reason, bucketTs, bucketMs }, "skip-tracker: failed to read skip events for bucket");
    return [];
  }
}

export async function getSkipReasonsSummary(
  windowMs: number = 24 * 60 * 60 * 1000,
): Promise<SkipReasonsSummary> {
  const cutoff = new Date(Date.now() - windowMs);
  let rows: Array<typeof skipEventsTable.$inferSelect> = [];
  try {
    rows = await db
      .select()
      .from(skipEventsTable)
      .where(gte(skipEventsTable.ts, cutoff))
      .orderBy(desc(skipEventsTable.ts));
  } catch (err) {
    logger.error({ err }, "skip-tracker: failed to read skip events");
  }

  const recent = rows.map(rowToEvent);

  const grouped = new Map<SkipReason, SkipEvent[]>();
  for (const ev of recent) {
    const list = grouped.get(ev.reason) ?? [];
    list.push(ev);
    grouped.set(ev.reason, list);
  }

  const byReason: SkipReasonsSummary["byReason"] = [];
  for (const [reason, events] of grouped.entries()) {
    const agentCounts = new Map<string, number>();
    for (const e of events) {
      agentCounts.set(e.agentName, (agentCounts.get(e.agentName) ?? 0) + 1);
    }
    const sortedAsc = [...events].sort((a, b) => a.ts - b.ts);
    byReason.push({
      reason,
      label: REASON_LABELS[reason] ?? reason,
      count: events.length,
      byAgent: Array.from(agentCounts.entries())
        .map(([agentName, count]) => ({ agentName, count }))
        .sort((a, b) => b.count - a.count),
      recent: sortedAsc.slice(-5).reverse(),
    });
  }

  byReason.sort((a, b) => b.count - a.count);

  return {
    windowMs,
    generatedAt: new Date().toISOString(),
    totalSkips: recent.length,
    byReason,
  };
}

export interface SkipTimelineBucket {
  ts: number;
  total: number;
  byReason: Record<string, number>;
  spikeReasons: SkipReason[];
}

export interface SkipSpike {
  ts: number;
  reason: SkipReason;
  label: string;
  count: number;
  mean: number;
  stdDev: number;
  zScore: number;
}

export interface SkipTimeline {
  windowMs: number;
  bucketMs: number;
  generatedAt: string;
  totalSkips: number;
  reasons: Array<{ reason: SkipReason; label: string; total: number }>;
  buckets: SkipTimelineBucket[];
  spikes: SkipSpike[];
  spikeThreshold: { zScore: number; minCount: number };
}

const SPIKE_Z_THRESHOLD = 2;
const SPIKE_MIN_COUNT = 3;

export async function getSkipTimeline(
  windowMs: number = 24 * 60 * 60 * 1000,
  bucketMs?: number,
): Promise<SkipTimeline> {
  const now = Date.now();
  const cutoff = new Date(now - windowMs);

  // Default bucket size aims for ~24-30 buckets across the window
  const hours = windowMs / (60 * 60 * 1000);
  const defaultBucketHours = hours <= 24 ? 1 : hours <= 72 ? 3 : 6;
  const bucketMillis = bucketMs ?? defaultBucketHours * 60 * 60 * 1000;

  let rows: Array<typeof skipEventsTable.$inferSelect> = [];
  try {
    rows = await db
      .select()
      .from(skipEventsTable)
      .where(gte(skipEventsTable.ts, cutoff))
      .orderBy(skipEventsTable.ts);
  } catch (err) {
    logger.error({ err }, "skip-tracker: failed to read skip events for timeline");
  }

  const startTs = Math.floor((now - windowMs) / bucketMillis) * bucketMillis;
  const bucketCount = Math.max(1, Math.ceil((now - startTs) / bucketMillis) + 1);

  const buckets: SkipTimelineBucket[] = [];
  for (let i = 0; i < bucketCount; i++) {
    buckets.push({ ts: startTs + i * bucketMillis, total: 0, byReason: {}, spikeReasons: [] });
  }

  const reasonTotals = new Map<SkipReason, number>();

  for (const row of rows) {
    const t = row.ts.getTime();
    const idx = Math.floor((t - startTs) / bucketMillis);
    if (idx < 0 || idx >= bucketCount) continue;
    const bucket = buckets[idx];
    if (!bucket) continue;
    const reason = row.reason as SkipReason;
    bucket.total++;
    bucket.byReason[reason] = (bucket.byReason[reason] ?? 0) + 1;
    reasonTotals.set(reason, (reasonTotals.get(reason) ?? 0) + 1);
  }

  const reasons = Array.from(reasonTotals.entries())
    .map(([reason, total]) => ({ reason, label: REASON_LABELS[reason] ?? reason, total }))
    .sort((a, b) => b.total - a.total);

  const spikes: SkipSpike[] = [];
  if (buckets.length >= 3) {
    for (const { reason, label } of reasons) {
      const series = buckets.map((b) => b.byReason[reason] ?? 0);
      const n = series.length;
      const mean = series.reduce((a, b) => a + b, 0) / n;
      const variance = series.reduce((sum, v) => sum + (v - mean) * (v - mean), 0) / n;
      const stdDev = Math.sqrt(variance);
      if (stdDev <= 0) continue;
      for (let i = 0; i < n; i++) {
        const count = series[i] ?? 0;
        if (count < SPIKE_MIN_COUNT) continue;
        const z = (count - mean) / stdDev;
        if (z >= SPIKE_Z_THRESHOLD) {
          const bucket = buckets[i];
          if (!bucket) continue;
          bucket.spikeReasons.push(reason);
          spikes.push({
            ts: bucket.ts,
            reason,
            label,
            count,
            mean: Number(mean.toFixed(2)),
            stdDev: Number(stdDev.toFixed(2)),
            zScore: Number(z.toFixed(2)),
          });
        }
      }
    }
    spikes.sort((a, b) => b.zScore - a.zScore);
  }

  return {
    windowMs,
    bucketMs: bucketMillis,
    generatedAt: new Date().toISOString(),
    totalSkips: rows.length,
    reasons,
    buckets,
    spikes,
    spikeThreshold: { zScore: SPIKE_Z_THRESHOLD, minCount: SPIKE_MIN_COUNT },
  };
}

