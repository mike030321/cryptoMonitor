import { eq } from "drizzle-orm";
import { db, tuningGateStateTable, appSettingsTable } from "@workspace/db";
import { logger } from "./logger";
import { getSkipReasonsSummary, getSkipsForReason, type SkipEvent, type SkipReason } from "./skip-tracker";
import { GATES_BASELINE } from "./trading-constants";

const AUTO_APPLY_TIGHTEN_SETTING_KEY = "autoApplyTightenOverride";

export type GateKey =
  | "MIN_CONFIDENCE_TO_TRADE"
  | "COUNTER_TREND_MIN_CONFIDENCE"
  | "MIN_TP_DISTANCE_PCT"
  | "MIN_EV_VS_COST";

interface GateMeta {
  key: GateKey;
  label: string;
  baseline: number;
  unit: "ratio" | "pct" | "multiple";
  skipReasons: SkipReason[];
  minFloor: number;
}

const GATES: Record<GateKey, GateMeta> = {
  MIN_CONFIDENCE_TO_TRADE: {
    key: "MIN_CONFIDENCE_TO_TRADE",
    label: "Min confidence to trade",
    baseline: GATES_BASELINE.MIN_CONFIDENCE_TO_TRADE.value,
    unit: "ratio",
    skipReasons: ["confidence_below_threshold"],
    minFloor: GATES_BASELINE.MIN_CONFIDENCE_TO_TRADE.floor,
  },
  COUNTER_TREND_MIN_CONFIDENCE: {
    key: "COUNTER_TREND_MIN_CONFIDENCE",
    label: "Counter-trend min confidence",
    baseline: GATES_BASELINE.COUNTER_TREND_MIN_CONFIDENCE.value,
    unit: "ratio",
    skipReasons: ["counter_trend_regime"],
    minFloor: GATES_BASELINE.COUNTER_TREND_MIN_CONFIDENCE.floor,
  },
  MIN_TP_DISTANCE_PCT: {
    key: "MIN_TP_DISTANCE_PCT",
    label: "Min TP distance (fee floor)",
    baseline: GATES_BASELINE.MIN_TP_DISTANCE_PCT.value,
    unit: "pct",
    skipReasons: ["fee_gate_tp_floor"],
    minFloor: GATES_BASELINE.MIN_TP_DISTANCE_PCT.floor,
  },
  MIN_EV_VS_COST: {
    key: "MIN_EV_VS_COST",
    label: "Min EV vs round-trip cost",
    baseline: GATES_BASELINE.MIN_EV_VS_COST.value,
    unit: "multiple",
    skipReasons: ["fee_gate_ev"],
    minFloor: GATES_BASELINE.MIN_EV_VS_COST.floor,
  },
};

const current: Record<GateKey, number> = {
  MIN_CONFIDENCE_TO_TRADE: GATES.MIN_CONFIDENCE_TO_TRADE.baseline,
  COUNTER_TREND_MIN_CONFIDENCE: GATES.COUNTER_TREND_MIN_CONFIDENCE.baseline,
  MIN_TP_DISTANCE_PCT: GATES.MIN_TP_DISTANCE_PCT.baseline,
  MIN_EV_VS_COST: GATES.MIN_EV_VS_COST.baseline,
};

const belowBaselineSince: Record<GateKey, number | null> = {
  MIN_CONFIDENCE_TO_TRADE: null,
  COUNTER_TREND_MIN_CONFIDENCE: null,
  MIN_TP_DISTANCE_PCT: null,
  MIN_EV_VS_COST: null,
};

function updateBelowBaselineTracking(gate: GateKey, ts: number = Date.now()): void {
  const meta = GATES[gate];
  const cur = current[gate];
  // Use a tiny epsilon so floating-point drift doesn't flip the flag.
  const isBelow = cur < meta.baseline * (1 - 1e-9);
  if (isBelow) {
    if (belowBaselineSince[gate] === null) {
      belowBaselineSince[gate] = ts;
    }
  } else {
    belowBaselineSince[gate] = null;
  }
}

async function persistGateState(gate: GateKey): Promise<void> {
  const cur = current[gate];
  const since = belowBaselineSince[gate];
  const sinceDate = since === null ? null : new Date(since);
  try {
    await db
      .insert(tuningGateStateTable)
      .values({
        gate,
        currentValue: cur,
        belowBaselineSince: sinceDate,
        updatedAt: new Date(),
      })
      .onConflictDoUpdate({
        target: tuningGateStateTable.gate,
        set: {
          currentValue: cur,
          belowBaselineSince: sinceDate,
          updatedAt: new Date(),
        },
      });
  } catch (err) {
    logger.error({ err, gate }, "Tuning: failed to persist gate state");
  }
}

function persistGateStateAsync(gate: GateKey): void {
  void persistGateState(gate);
}

let loadPromise: Promise<void> | null = null;

export function loadTuningStateFromDb(): Promise<void> {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    try {
      const rows = await db.select().from(tuningGateStateTable);
      for (const row of rows) {
        if (!(row.gate in GATES)) continue;
        const gate = row.gate as GateKey;
        const meta = GATES[gate];
        // Clamp to safety floor in case minFloor was raised since last write.
        const cur = Math.max(meta.minFloor, Math.min(row.currentValue, meta.baseline));
        current[gate] = cur;
        const isBelow = cur < meta.baseline * (1 - 1e-9);
        if (isBelow) {
          belowBaselineSince[gate] = row.belowBaselineSince
            ? row.belowBaselineSince.getTime()
            : Date.now();
        } else {
          belowBaselineSince[gate] = null;
        }
      }
      logger.info({ loaded: rows.length }, "Tuning: loaded persisted gate state");
    } catch (err) {
      logger.error({ err }, "Tuning: failed to load persisted gate state");
    }
  })();
  return loadPromise;
}

export type TuningSource = "auto-suggest" | "auto-tighten" | "manual" | "env";

export type TuningChangeKind = "gate" | "auto-tighten-toggle";

export interface TuningChange {
  id: string;
  ts: number;
  kind: TuningChangeKind;
  /** Gate key for gate-kind entries; omitted for toggle entries. */
  gate?: GateKey;
  label: string;
  /** Numeric fields populated for gate-kind entries only. */
  oldValue?: number;
  newValue?: number;
  pctChange?: number;
  source: TuningSource;
  reverted: boolean;
  revertedAt: number | null;
  /** Populated for auto-tighten-toggle entries: the new effective enabled state. */
  enabled?: boolean;
}

const history: TuningChange[] = [];
const MAX_HISTORY = 50;

export function getMinConfidenceToTrade(): number {
  return current.MIN_CONFIDENCE_TO_TRADE;
}
/**
 * Quant-driven trades use the explicit `floor` from gates_baseline (0.40),
 * not the live `value` (0.50). Rationale: the 0.50 baseline + asymmetric +5pt
 * UP bump in paper-trader was tuned against the LLM fleet's per-call
 * confidence distribution (LLM BUY winrate 24% vs SELL 56%, see paper-trader
 * line 228-233). The quant model emits its own calibrated p_up / p_down with
 * the 3-class softmax → isotonic pipeline, so the LLM-era asymmetry doesn't
 * apply. Using the floor here is *not* a loosening: 0.40 is the value the
 * auto-tuner has always been allowed to descend to, so by construction the
 * fleet has already endorsed it as a safe lower bound.
 */
export function getQuantMinConfidenceToTrade(): number {
  return GATES.MIN_CONFIDENCE_TO_TRADE.minFloor;
}
export function getCounterTrendMinConfidence(): number {
  return current.COUNTER_TREND_MIN_CONFIDENCE;
}
export function getMinTpDistancePct(): number {
  return current.MIN_TP_DISTANCE_PCT;
}
export function getMinEvVsCost(): number {
  return current.MIN_EV_VS_COST;
}

export interface GateState {
  key: GateKey;
  label: string;
  baseline: number;
  current: number;
  minFloor: number;
  unit: "ratio" | "pct" | "multiple";
  pctFromBaseline: number;
  canLoosenMore: boolean;
  belowBaselineSince: number | null;
}

export function getTuningState(): { gates: GateState[]; history: TuningChange[] } {
  const gates: GateState[] = (Object.keys(GATES) as GateKey[]).map((k) => {
    const meta = GATES[k];
    const cur = current[k];
    return {
      key: k,
      label: meta.label,
      baseline: meta.baseline,
      current: cur,
      minFloor: meta.minFloor,
      unit: meta.unit,
      pctFromBaseline: ((cur - meta.baseline) / meta.baseline) * 100,
      canLoosenMore: cur * 0.9 >= meta.minFloor,
      belowBaselineSince: belowBaselineSince[k],
    };
  });
  return { gates, history: [...history].reverse() };
}

const LOOSEN_STEP_PCT = 0.10;
const TIGHTEN_STEP_PCT = 0.10;
const SUGGEST_DOMINANT_PCT = 0.60;
const SUGGEST_MIN_TOTAL_SKIPS = 30;
// Number of consecutive healthy observations required before we propose
// tightening a previously loosened gate back toward baseline. The monitor
// records an observation every 5 minutes, so 3 ticks ≈ 15 minutes of
// sustained healthy open-position count.
const SUSTAINED_HEALTHY_TICKS = 3;

let consecutiveHealthyTicks = 0;

export function recordHealthyObservation(isHealthy: boolean): void {
  if (isHealthy) {
    consecutiveHealthyTicks += 1;
  } else {
    consecutiveHealthyTicks = 0;
  }
}

// Number of consecutive refresh ticks a tighten suggestion must remain valid
// for the same gate before the engine auto-applies it. The monitor refreshes
// the tuning suggestion every 5 minutes, so 6 ticks ≈ 30 minutes of sustained
// validity.
const AUTO_APPLY_TIGHTEN_TICKS = 6;

let pendingTighten: { gate: GateKey; ticks: number } | null = null;

export function recordTightenSuggestionTick(gate: GateKey | null): {
  gate: GateKey;
  ticks: number;
} | null {
  if (gate === null) {
    pendingTighten = null;
    return null;
  }
  if (pendingTighten && pendingTighten.gate === gate) {
    pendingTighten.ticks += 1;
  } else {
    pendingTighten = { gate, ticks: 1 };
  }
  return { ...pendingTighten };
}

export function getPendingTighten(): { gate: GateKey; ticks: number } | null {
  return pendingTighten ? { ...pendingTighten } : null;
}

export function getAutoApplyTightenTicks(): number {
  return AUTO_APPLY_TIGHTEN_TICKS;
}

function envAutoApplyTightenEnabled(): boolean {
  const v = process.env.AUTO_APPLY_TIGHTEN_ENABLED;
  if (!v) return false;
  const s = v.trim().toLowerCase();
  return s === "1" || s === "true" || s === "yes" || s === "on";
}

let autoApplyTightenOverride: boolean | null = null;

export function isAutoApplyTightenEnabled(): boolean {
  if (autoApplyTightenOverride !== null) return autoApplyTightenOverride;
  return envAutoApplyTightenEnabled();
}

export interface AutoApplyTightenStatus {
  enabled: boolean;
  envDefault: boolean;
  override: boolean | null;
  source: "override" | "env";
}

export function getAutoApplyTightenStatus(): AutoApplyTightenStatus {
  const envDefault = envAutoApplyTightenEnabled();
  const override = autoApplyTightenOverride;
  return {
    enabled: override !== null ? override : envDefault,
    envDefault,
    override,
    source: override !== null ? "override" : "env",
  };
}

export async function setAutoApplyTightenOverride(value: boolean | null): Promise<AutoApplyTightenStatus> {
  const prevEnabled = isAutoApplyTightenEnabled();
  // Persist first so a successful API response guarantees the override
  // will survive a restart. If persistence fails we leave the in-memory
  // value untouched and surface the error to the caller.
  await persistAutoApplyTightenOverride(value);
  autoApplyTightenOverride = value;
  const status = getAutoApplyTightenStatus();
  logger.info(
    { override: value, effective: status.enabled, envDefault: status.envDefault },
    `Tuning: auto-apply tighten override set to ${value === null ? "(cleared)" : String(value)}`,
  );
  if (status.enabled !== prevEnabled) {
    // The effective on/off state changed — record it in tuning history so
    // operators reviewing why a gate moved (or didn't) can see when the
    // safety switch was flipped.
    const toggleSource: TuningSource = value === null ? "env" : "manual";
    const entry: TuningChange = {
      id: makeId(),
      ts: Date.now(),
      kind: "auto-tighten-toggle",
      label: `Auto-tighten turned ${status.enabled ? "ON" : "OFF"}`,
      source: toggleSource,
      reverted: false,
      revertedAt: null,
      enabled: status.enabled,
    };
    pushHistory(entry);
  }
  return status;
}

async function persistAutoApplyTightenOverride(value: boolean | null): Promise<void> {
  if (value === null) {
    await db.delete(appSettingsTable).where(eq(appSettingsTable.key, AUTO_APPLY_TIGHTEN_SETTING_KEY));
    return;
  }
  await db
    .insert(appSettingsTable)
    .values({ key: AUTO_APPLY_TIGHTEN_SETTING_KEY, value })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value, updatedAt: new Date() },
    });
}

/**
 * Load the persisted auto-apply tighten override from the database into
 * memory. Should be called once during server startup so the toggle's last
 * value (including a "use env default" choice) survives restarts.
 */
export async function loadAutoApplyTightenOverride(): Promise<void> {
  try {
    const rows = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, AUTO_APPLY_TIGHTEN_SETTING_KEY))
      .limit(1);
    const row = rows[0];
    if (!row) {
      autoApplyTightenOverride = null;
      return;
    }
    const v = row.value;
    if (typeof v === "boolean") {
      autoApplyTightenOverride = v;
      logger.info(
        { override: v, envDefault: envAutoApplyTightenEnabled() },
        "Tuning: restored persisted auto-apply tighten override",
      );
    } else {
      autoApplyTightenOverride = null;
    }
  } catch (err) {
    logger.error({ err }, "Tuning: failed to load persisted auto-apply tighten override");
  }
}

export interface TuningSuggestion {
  direction: "loosen" | "tighten";
  gate: GateKey;
  label: string;
  reason: SkipReason | null;
  reasonLabel: string;
  shareOfSkips: number;
  totalSkips: number;
  openPositionCount: number;
  healthyOpenFloor: number;
  currentValue: number;
  proposedValue: number;
  /** Magnitude of the change as a percent (positive number). */
  loosenPct: number;
  unit: "ratio" | "pct" | "multiple";
  projectedAdditionalTrades: number;
  projectedAdditionalTradesCapped: boolean;
  projectedSampleSize: number;
  /** How many consecutive healthy ticks have been observed (tighten only). */
  sustainedHealthyTicks?: number;
  requiredHealthyTicks?: number;
}

const PROJECTION_CAP = 50;

function parsePctString(value: unknown): number | null {
  if (typeof value !== "string") return null;
  const n = parseFloat(value.replace("%", "").trim());
  if (!Number.isFinite(n)) return null;
  return n / 100;
}

function countWouldPass(
  gateKey: GateKey,
  proposed: number,
  events: SkipEvent[],
  currentValue: number,
): number {
  let pass = 0;
  for (const ev of events) {
    const d = ev.details;
    switch (gateKey) {
      case "MIN_CONFIDENCE_TO_TRADE":
      case "COUNTER_TREND_MIN_CONFIDENCE": {
        const c = parsePctString(d["confidence"]);
        if (c !== null && c >= proposed) pass++;
        break;
      }
      case "MIN_TP_DISTANCE_PCT": {
        const tp = parsePctString(d["tpDistancePct"]);
        if (tp !== null && tp >= proposed) pass++;
        break;
      }
      case "MIN_EV_VS_COST": {
        const evScore = parsePctString(d["evScore"]);
        const evRequired = parsePctString(d["evRequired"]);
        if (evScore !== null && evRequired !== null && currentValue > 0) {
          const proposedRequirement = evRequired * (proposed / currentValue);
          if (evScore >= proposedRequirement) pass++;
        }
        break;
      }
    }
  }
  return pass;
}

function reasonToGate(reason: SkipReason): GateKey | null {
  for (const k of Object.keys(GATES) as GateKey[]) {
    if (GATES[k].skipReasons.includes(reason)) return k;
  }
  return null;
}

export async function computeSuggestion(
  openPositionCount: number,
  agentCount: number,
  windowMs: number = 24 * 60 * 60 * 1000,
): Promise<TuningSuggestion | null> {
  const summary = await getSkipReasonsSummary(windowMs);
  if (summary.totalSkips < SUGGEST_MIN_TOTAL_SKIPS) return null;
  const healthyOpenFloor = Math.max(1, agentCount); // expect at least ~1 open position per bot
  if (openPositionCount >= healthyOpenFloor) return null;

  const top = summary.byReason[0];
  if (!top) return null;
  const share = top.count / summary.totalSkips;
  if (share < SUGGEST_DOMINANT_PCT) return null;

  const gateKey = reasonToGate(top.reason as SkipReason);
  if (!gateKey) return null;

  const meta = GATES[gateKey];
  const cur = current[gateKey];
  const proposed = cur * (1 - LOOSEN_STEP_PCT);
  if (proposed < meta.minFloor) return null;

  const events = await getSkipsForReason(top.reason as SkipReason, windowMs);
  const wouldPass = countWouldPass(gateKey, proposed, events, cur);
  const capped = wouldPass > PROJECTION_CAP;
  const projectedAdditionalTrades = capped ? PROJECTION_CAP : wouldPass;

  return {
    direction: "loosen",
    gate: gateKey,
    label: meta.label,
    reason: top.reason as SkipReason,
    reasonLabel: top.label,
    shareOfSkips: share,
    totalSkips: summary.totalSkips,
    openPositionCount,
    healthyOpenFloor,
    currentValue: cur,
    proposedValue: proposed,
    loosenPct: LOOSEN_STEP_PCT * 100,
    unit: meta.unit,
    projectedAdditionalTrades,
    projectedAdditionalTradesCapped: capped,
    projectedSampleSize: events.length,
  };
}

/**
 * Symmetric counterpart to `computeSuggestion`: once trade volume has
 * recovered (open positions sustained at/above the healthy floor) and a
 * previously loosened gate is no longer the dominant skip reason, propose
 * tightening that gate by one step back toward baseline.
 */
export async function computeTightenSuggestion(
  openPositionCount: number,
  agentCount: number,
  windowMs: number = 24 * 60 * 60 * 1000,
): Promise<TuningSuggestion | null> {
  const healthyOpenFloor = Math.max(1, agentCount);
  if (openPositionCount < healthyOpenFloor) return null;
  if (consecutiveHealthyTicks < SUSTAINED_HEALTHY_TICKS) return null;

  const summary = await getSkipReasonsSummary(windowMs);
  const shareByReason = new Map<SkipReason, number>();
  if (summary.totalSkips > 0) {
    for (const row of summary.byReason) {
      shareByReason.set(row.reason as SkipReason, row.count / summary.totalSkips);
    }
  }

  // Among gates currently below baseline (i.e. previously loosened) whose
  // skip reason is no longer dominant, pick the one furthest below baseline
  // so we walk back to baseline systematically.
  let best: { gateKey: GateKey; pctBelow: number; reasonShare: number; reasonLabel: string; reason: SkipReason | null } | null = null;
  for (const k of Object.keys(GATES) as GateKey[]) {
    const meta = GATES[k];
    const cur = current[k];
    if (cur >= meta.baseline) continue; // not loosened
    let maxShare = 0;
    let dominantReason: SkipReason | null = null;
    let dominantLabel = "";
    for (const r of meta.skipReasons) {
      const s = shareByReason.get(r) ?? 0;
      if (s > maxShare) {
        maxShare = s;
        dominantReason = r;
        const row = summary.byReason.find((b) => b.reason === r);
        dominantLabel = row?.label ?? r;
      }
    }
    if (maxShare >= SUGGEST_DOMINANT_PCT) continue; // still the bottleneck
    const pctBelow = (meta.baseline - cur) / meta.baseline;
    if (!best || pctBelow > best.pctBelow) {
      best = { gateKey: k, pctBelow, reasonShare: maxShare, reasonLabel: dominantLabel, reason: dominantReason };
    }
  }

  if (!best) return null;

  const meta = GATES[best.gateKey];
  const cur = current[best.gateKey];
  // Step toward baseline by TIGHTEN_STEP_PCT of the current value, capped so
  // we never overshoot baseline.
  const proposed = Math.min(meta.baseline, cur * (1 + TIGHTEN_STEP_PCT));
  if (proposed <= cur) return null;

  return {
    direction: "tighten",
    gate: best.gateKey,
    label: meta.label,
    reason: best.reason,
    reasonLabel: best.reasonLabel || (best.reason ?? ""),
    shareOfSkips: best.reasonShare,
    totalSkips: summary.totalSkips,
    openPositionCount,
    healthyOpenFloor,
    currentValue: cur,
    proposedValue: proposed,
    loosenPct: ((proposed - cur) / cur) * 100,
    unit: meta.unit,
    // Tightening doesn't unlock new trades — it removes some borderline ones.
    // Projection fields are kept zeroed for shape compatibility with loosen.
    projectedAdditionalTrades: 0,
    projectedAdditionalTradesCapped: false,
    projectedSampleSize: 0,
    sustainedHealthyTicks: consecutiveHealthyTicks,
    requiredHealthyTicks: SUSTAINED_HEALTHY_TICKS,
  };
}

function pushHistory(change: TuningChange): void {
  history.push(change);
  if (history.length > MAX_HISTORY) history.splice(0, history.length - MAX_HISTORY);
}

function makeId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function applyLoosen(
  gate: GateKey,
  source: TuningSource = "manual",
): TuningChange {
  const meta = GATES[gate];
  if (!meta) throw new Error(`Unknown gate: ${gate}`);
  const oldValue = current[gate];
  const newValue = oldValue * (1 - LOOSEN_STEP_PCT);
  if (newValue < meta.minFloor) {
    throw new Error(
      `${gate} cannot be loosened further (would fall below safety floor ${meta.minFloor})`,
    );
  }
  current[gate] = newValue;
  const ts = Date.now();
  updateBelowBaselineTracking(gate, ts);
  persistGateStateAsync(gate);
  const change: TuningChange = {
    id: makeId(),
    ts,
    kind: "gate",
    gate,
    label: meta.label,
    oldValue,
    newValue,
    pctChange: -LOOSEN_STEP_PCT * 100,
    source,
    reverted: false,
    revertedAt: null,
  };
  pushHistory(change);
  logger.info(
    { gate, oldValue, newValue, source },
    `Tuning: loosened ${gate} from ${oldValue} to ${newValue}`,
  );
  return change;
}

export function applyTighten(
  gate: GateKey,
  source: TuningSource = "manual",
): TuningChange {
  const meta = GATES[gate];
  if (!meta) throw new Error(`Unknown gate: ${gate}`);
  const oldValue = current[gate];
  if (oldValue >= meta.baseline) {
    throw new Error(
      `${gate} is already at or above baseline (${meta.baseline}); nothing to tighten`,
    );
  }
  const newValue = Math.min(meta.baseline, oldValue * (1 + TIGHTEN_STEP_PCT));
  if (newValue <= oldValue) {
    throw new Error(`${gate} cannot be tightened further`);
  }
  current[gate] = newValue;
  const ts = Date.now();
  updateBelowBaselineTracking(gate, ts);
  persistGateStateAsync(gate);
  const pctChange = ((newValue - oldValue) / oldValue) * 100;
  const change: TuningChange = {
    id: makeId(),
    ts,
    kind: "gate",
    gate,
    label: meta.label,
    oldValue,
    newValue,
    pctChange,
    source,
    reverted: false,
    revertedAt: null,
  };
  pushHistory(change);
  if (pendingTighten && pendingTighten.gate === gate) {
    pendingTighten = null;
  }
  logger.info(
    { gate, oldValue, newValue, source },
    `Tuning: tightened ${gate} from ${oldValue} to ${newValue}`,
  );
  return change;
}

export function revertChange(changeId: string): TuningChange {
  const idx = history.findIndex((h) => h.id === changeId);
  if (idx === -1) throw new Error(`Change not found: ${changeId}`);
  const change = history[idx];
  if (change.reverted) throw new Error("Change already reverted");
  if (change.kind !== "gate" || !change.gate || change.oldValue === undefined) {
    throw new Error("Only gate changes can be reverted");
  }
  // Re-apply the old value
  current[change.gate] = change.oldValue;
  change.reverted = true;
  change.revertedAt = Date.now();
  updateBelowBaselineTracking(change.gate, change.revertedAt);
  persistGateStateAsync(change.gate);
  logger.info(
    { gate: change.gate, restoredTo: change.oldValue },
    `Tuning: reverted change ${changeId}`,
  );
  return change;
}

let cachedSuggestion: TuningSuggestion | null = null;

export function getCachedSuggestion(): TuningSuggestion | null {
  return cachedSuggestion;
}

export function setCachedSuggestion(s: TuningSuggestion | null): void {
  cachedSuggestion = s;
}
