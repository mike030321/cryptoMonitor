/**
 * Phase 5 — Auto-revert safety net.
 *
 * After every analysis cycle, if the quant brain is currently driving trades
 * AND the live-vs-backtest tracker reports `drift_low` on the majority of
 * coins for N consecutive cycles, we automatically disable the quant brain.
 * That returns the live decision path to explicit ABSTAIN/no-trade behavior;
 * there is no LLM fallback after Task #444. This is the safety net for the
 * case where the live model silently underperforms its offline backtest
 * baseline (overfit, regime shift, bad features, ...).
 *
 * Persistence:
 *   - revert events: `app_settings.brain_revert_log` (last 50, jsonb array)
 *   - in-memory consecutive-drift counter (cycle-level, not persisted —
 *     a process restart resets the gate, which is the safe default).
 */
import { db, appSettingsTable, modelPredictionsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";
import { setBrainState, isQuantBrainEnabled, BRAIN_REVERT_LOG_KEY } from "./brain-flag";
import { judgeDirection, QUANT_BRAIN_AUTO_REVERT } from "./trading-constants";

// All thresholds come from shared/trading-frictions.json so the Python
// backtester sees the same window when it simulates rollback decisions.
// Tune the JSON, never the constants here.
const CONSECUTIVE_DRIFT_CYCLES_TO_REVERT = QUANT_BRAIN_AUTO_REVERT.consecutive_drift_cycles;
const DRIFT_LOW_COIN_SHARE = QUANT_BRAIN_AUTO_REVERT.drift_share_threshold;
const MIN_RESOLVED_PER_COIN = QUANT_BRAIN_AUTO_REVERT.min_evaluable_coins;
const DRIFT_BAND_PCT = 0.10;
const MAX_REVERT_LOG_ENTRIES = 50;

let consecutiveDriftCycles = 0;

export interface BrainRevertEvent {
  ts: string;
  reason: string;
  driftedCoins: string[];
  totalCoinsEvaluated: number;
}

interface BacktestBaseline { winRate: number; nTrades: number }
interface BacktestPerCoinPayload { metrics?: { win_rate?: unknown; n_trades?: unknown } }
interface BacktestRun { per_coin?: Record<string, BacktestPerCoinPayload> }
interface BacktestReport { runs?: BacktestRun[] }

function asNumber(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

async function loadBaselines(): Promise<Record<string, BacktestBaseline>> {
  // Lazy-load from disk to avoid bundling node:fs at import time for tests
  // that don't need this module.
  try {
    const { readFileSync, existsSync } = await import("node:fs");
    const path = await import("node:path");
    const { fileURLToPath } = await import("node:url");
    let dir = path.dirname(fileURLToPath(import.meta.url));
    for (let i = 0; i < 8; i++) {
      const candidate = path.join(dir, "artifacts", "ml-engine", "models", "backtest_report.json");
      if (existsSync(candidate)) {
        const raw = JSON.parse(readFileSync(candidate, "utf8")) as BacktestReport;
        const out: Record<string, BacktestBaseline> = {};
        for (const run of raw.runs ?? []) {
          for (const [coinId, payload] of Object.entries(run.per_coin ?? {})) {
            const m = payload?.metrics ?? {};
            out[coinId] = { winRate: asNumber(m.win_rate), nTrades: asNumber(m.n_trades) };
          }
        }
        return out;
      }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  } catch {
    // Missing baselines file is fine — the gate just won't trigger.
  }
  return {};
}

async function appendRevertLog(event: BrainRevertEvent): Promise<void> {
  try {
    const rows = await db.select().from(appSettingsTable).where(eq(appSettingsTable.key, BRAIN_REVERT_LOG_KEY)).limit(1);
    const existing: BrainRevertEvent[] = rows.length > 0 && Array.isArray(rows[0].value)
      ? (rows[0].value as BrainRevertEvent[])
      : [];
    const next = [event, ...existing].slice(0, MAX_REVERT_LOG_ENTRIES);
    await db
      .insert(appSettingsTable)
      .values({ key: BRAIN_REVERT_LOG_KEY, value: next })
      .onConflictDoUpdate({
        target: appSettingsTable.key,
        set: { value: next, updatedAt: new Date() },
      });
  } catch (err) {
    logger.warn({ err }, "brain-auto-revert: failed to persist revert event");
  }
}

export async function getBrainRevertLog(): Promise<BrainRevertEvent[]> {
  try {
    const rows = await db.select().from(appSettingsTable).where(eq(appSettingsTable.key, BRAIN_REVERT_LOG_KEY)).limit(1);
    if (rows.length === 0) return [];
    return Array.isArray(rows[0].value) ? (rows[0].value as BrainRevertEvent[]) : [];
  } catch {
    return [];
  }
}

export function getAutoRevertCounter(): number {
  return consecutiveDriftCycles;
}

/**
 * Called once per analysis cycle. No-op when the quant brain is OFF.
 * Returns `true` if this call disabled the quant brain.
 */
export async function evaluateBrainAutoRevert(): Promise<boolean> {
  if (!(await isQuantBrainEnabled())) {
    consecutiveDriftCycles = 0;
    return false;
  }
  const baselines = await loadBaselines();
  if (Object.keys(baselines).length === 0) return false;

  const rows = await db.select().from(modelPredictionsTable);
  const byCoin = new Map<string, typeof rows>();
  for (const r of rows) {
    if (r.outcome === "pending") continue;
    const arr = byCoin.get(r.coinId) ?? [];
    arr.push(r);
    byCoin.set(r.coinId, arr);
  }

  let evaluated = 0;
  const drifted: string[] = [];
  for (const [coinId, baseline] of Object.entries(baselines)) {
    if (baseline.nTrades <= 0) continue;
    const arr = byCoin.get(coinId) ?? [];
    if (arr.length < MIN_RESOLVED_PER_COIN) continue;
    let wins = 0;
    for (const r of arr) {
      const judge = judgeDirection(r.modelDirection as "up" | "down" | "stable", r.resolvedOutcomePct ?? 0, r.timeframe);
      if (judge.correct) wins++;
    }
    const liveWinRate = wins / arr.length;
    const delta = liveWinRate - baseline.winRate;
    evaluated++;
    if (delta < -DRIFT_BAND_PCT) drifted.push(coinId);
  }

  if (evaluated === 0) {
    consecutiveDriftCycles = 0;
    return false;
  }

  const driftShare = drifted.length / evaluated;
  if (driftShare >= DRIFT_LOW_COIN_SHARE) {
    consecutiveDriftCycles++;
    logger.warn({ driftShare, drifted, evaluated, consecutiveDriftCycles },
      "brain-auto-revert: drift_low majority detected");
    if (consecutiveDriftCycles >= CONSECUTIVE_DRIFT_CYCLES_TO_REVERT) {
      const event: BrainRevertEvent = {
        ts: new Date().toISOString(),
        reason: `drift_low on ${drifted.length}/${evaluated} coins for ${consecutiveDriftCycles} consecutive cycles`,
        driftedCoins: drifted,
        totalCoinsEvaluated: evaluated,
      };
      await setBrainState(false, "auto_revert");
      await appendRevertLog(event);
      consecutiveDriftCycles = 0;
      logger.error({ event }, "brain-auto-revert: disabled quant brain after drift_low majority");
      return true;
    }
  } else {
    consecutiveDriftCycles = 0;
  }
  return false;
}
