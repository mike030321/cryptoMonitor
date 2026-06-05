/**
 * Task #234 — Slot starvation alerts.
 *
 * After every quarantine sweep we re-check whether any (coin, timeframe)
 * slot has been left with zero servable model versions. The ml-engine
 * resolver tries the per-coin model first, then falls back to the pooled
 * model — if BOTH have only quarantined active rows the resolver returns
 * null and `/ml/predict` 503s.
 *
 * Today that only shows up as request errors. This module turns it into a
 * proactive notification the dashboard / operator can see, naming the
 * starved slot AND the most recent quarantine reason that put it there.
 *
 * Notifications are kept in-memory (re-derived from the registry on every
 * sweep) and exposed both through `getActiveSlotStarvationAlerts()` and
 * the `/crypto/diagnostics/quarantine` endpoint payload.
 */
import {
  db,
  modelRegistryTable,
  quarantineEventsTable,
} from "@workspace/db";
import { and, eq, inArray, desc } from "drizzle-orm";
import { logger } from "./logger";

/** Coin ids that represent the pooled fallback in the registry. The
 *  ml-engine writes pooled rows as `__pooled__`; legacy / TS-side code
 *  defaults pooled to `*`. Treat both as "the pooled slot". */
export const POOLED_COIN_IDS: ReadonlySet<string> = new Set(["__pooled__", "*"]);

export interface SlotStarvationAlert {
  coinId: string;
  timeframe: string;
  /** Most recent reason that drove a model in this slot (per-coin OR
   *  pooled) into the quarantined state. null when no event row was
   *  found (shouldn't happen in practice but we don't want to crash). */
  lastQuarantineReason: string | null;
  lastQuarantineAt: string | null;
  /** How many active rows exist in the per-coin slot — all are quarantined
   *  when this alert fires. */
  perCoinQuarantinedCount: number;
  /** Same, for the pooled fallback. */
  pooledQuarantinedCount: number;
  detectedAt: string;
}

let activeAlerts: SlotStarvationAlert[] = [];

function isPooled(coinId: string): boolean {
  return POOLED_COIN_IDS.has(coinId);
}

interface SlotEventLookup {
  reason: string | null;
  at: string | null;
}

/**
 * Build a per-(coinId, timeframe) map of the most recent quarantine event
 * across all relevant registry rows. Done as TWO queries (one for the
 * candidate registry rows, one for their events) regardless of how many
 * starved slots there are, so sweep latency stays flat.
 */
async function buildLastEventLookup(
  starvedKeys: ReadonlyArray<{ coinId: string; timeframe: string }>,
): Promise<Map<string, SlotEventLookup>> {
  const out = new Map<string, SlotEventLookup>();
  if (starvedKeys.length === 0) return out;

  const timeframes = Array.from(new Set(starvedKeys.map((k) => k.timeframe)));
  const coinIds = Array.from(
    new Set([
      ...starvedKeys.map((k) => k.coinId),
      ...POOLED_COIN_IDS,
    ]),
  );

  // Pull all candidate registry rows in one shot, then index them in JS
  // by (coinId|timeframe). This is bounded by registry size and avoids
  // an N+1 over slots.
  const registryRows = await db
    .select({
      id: modelRegistryTable.id,
      coinId: modelRegistryTable.coinId,
      timeframe: modelRegistryTable.timeframe,
    })
    .from(modelRegistryTable)
    .where(
      and(
        eq(modelRegistryTable.state, "quarantined"),
        eq(modelRegistryTable.isActive, true),
        inArray(modelRegistryTable.timeframe, timeframes),
        inArray(modelRegistryTable.coinId, coinIds),
      ),
    );
  if (registryRows.length === 0) {
    for (const k of starvedKeys) {
      out.set(`${k.coinId}\u0000${k.timeframe}`, { reason: null, at: null });
    }
    return out;
  }

  // Map registryId -> "coinId\u0000timeframe" for both per-coin and pooled
  // rows; pooled rows are mapped against EVERY starved coin id at the
  // same timeframe so we can attribute the latest pooled event to each.
  const starvedTfs = new Map<string, Set<string>>();
  for (const k of starvedKeys) {
    let s = starvedTfs.get(k.timeframe);
    if (!s) {
      s = new Set();
      starvedTfs.set(k.timeframe, s);
    }
    s.add(k.coinId);
  }
  const idToSlots = new Map<number, string[]>();
  for (const r of registryRows) {
    const slots: string[] = [];
    if (isPooled(r.coinId)) {
      const targets = starvedTfs.get(r.timeframe);
      if (targets) {
        for (const c of targets) slots.push(`${c}\u0000${r.timeframe}`);
      }
    } else {
      const targets = starvedTfs.get(r.timeframe);
      if (targets && targets.has(r.coinId)) {
        slots.push(`${r.coinId}\u0000${r.timeframe}`);
      }
    }
    if (slots.length > 0) idToSlots.set(r.id, slots);
  }

  const ids = Array.from(idToSlots.keys());
  if (ids.length === 0) {
    for (const k of starvedKeys) {
      out.set(`${k.coinId}\u0000${k.timeframe}`, { reason: null, at: null });
    }
    return out;
  }

  const events = await db
    .select({
      registryId: quarantineEventsTable.registryId,
      reasonCode: quarantineEventsTable.reasonCode,
      createdAt: quarantineEventsTable.createdAt,
    })
    .from(quarantineEventsTable)
    .where(inArray(quarantineEventsTable.registryId, ids))
    .orderBy(desc(quarantineEventsTable.createdAt));

  for (const ev of events) {
    const slots = idToSlots.get(ev.registryId);
    if (!slots) continue;
    const at =
      ev.createdAt instanceof Date
        ? ev.createdAt.toISOString()
        : String(ev.createdAt);
    for (const slot of slots) {
      const existing = out.get(slot);
      // events come ordered DESC; first-write wins.
      if (!existing) out.set(slot, { reason: ev.reasonCode, at });
    }
  }

  for (const k of starvedKeys) {
    const slot = `${k.coinId}\u0000${k.timeframe}`;
    if (!out.has(slot)) out.set(slot, { reason: null, at: null });
  }
  return out;
}

/**
 * Re-derive the set of slot-starvation alerts from `model_registry`.
 * Returns the freshly-computed list AND replaces the in-memory cache.
 *
 * A slot starves when, at the same `timeframe`:
 *   - the per-coin slot has >=1 active row but ALL active rows are
 *     `state == 'quarantined'`, AND
 *   - the pooled fallback has zero active non-quarantined rows.
 * If the per-coin slot has no active rows at all (model never trained
 * yet for that coin) we don't alert — the pooled fallback is the
 * advertised behaviour, and `ml-availability` already surfaces "dark"
 * timeframes separately.
 */
export async function recomputeSlotStarvationAlerts(): Promise<SlotStarvationAlert[]> {
  const rows = await db
    .select({
      coinId: modelRegistryTable.coinId,
      timeframe: modelRegistryTable.timeframe,
      state: modelRegistryTable.state,
    })
    .from(modelRegistryTable)
    .where(eq(modelRegistryTable.isActive, true));

  // (coinId, tf) -> { quarantined, available }
  const counts = new Map<
    string,
    { coinId: string; timeframe: string; quarantined: number; available: number }
  >();
  for (const r of rows) {
    if (r.state === "retired") continue;
    const key = `${r.coinId}\u0000${r.timeframe}`;
    let c = counts.get(key);
    if (!c) {
      c = {
        coinId: r.coinId,
        timeframe: r.timeframe,
        quarantined: 0,
        available: 0,
      };
      counts.set(key, c);
    }
    if (r.state === "quarantined") c.quarantined += 1;
    else c.available += 1;
  }

  // Pooled availability per timeframe (sum across all pooled coin ids).
  const pooledByTf = new Map<string, { quarantined: number; available: number }>();
  for (const c of counts.values()) {
    if (!isPooled(c.coinId)) continue;
    const p = pooledByTf.get(c.timeframe) ?? { quarantined: 0, available: 0 };
    p.quarantined += c.quarantined;
    p.available += c.available;
    pooledByTf.set(c.timeframe, p);
  }

  const starved: Array<{
    coinId: string;
    timeframe: string;
    perCoinQuarantinedCount: number;
    pooledQuarantinedCount: number;
  }> = [];
  for (const c of counts.values()) {
    if (isPooled(c.coinId)) continue;
    if (c.available > 0) continue; // per-coin still has a serving version
    if (c.quarantined === 0) continue; // never trained — not a starvation case
    const pooled = pooledByTf.get(c.timeframe) ?? { quarantined: 0, available: 0 };
    if (pooled.available > 0) continue; // pooled fallback still works
    starved.push({
      coinId: c.coinId,
      timeframe: c.timeframe,
      perCoinQuarantinedCount: c.quarantined,
      pooledQuarantinedCount: pooled.quarantined,
    });
  }

  const eventLookup = await buildLastEventLookup(starved);
  const detectedAt = new Date().toISOString();
  const next: SlotStarvationAlert[] = starved.map((s) => {
    const ev = eventLookup.get(`${s.coinId}\u0000${s.timeframe}`)
      ?? { reason: null, at: null };
    return {
      coinId: s.coinId,
      timeframe: s.timeframe,
      lastQuarantineReason: ev.reason,
      lastQuarantineAt: ev.at,
      perCoinQuarantinedCount: s.perCoinQuarantinedCount,
      pooledQuarantinedCount: s.pooledQuarantinedCount,
      detectedAt,
    };
  });

  // Log only the slots that are NEWLY starved since the previous sweep so
  // we don't spam the same alert every cycle.
  const prevKeys = new Set(
    activeAlerts.map((a) => `${a.coinId}\u0000${a.timeframe}`),
  );
  for (const a of next) {
    const key = `${a.coinId}\u0000${a.timeframe}`;
    if (!prevKeys.has(key)) {
      logger.warn(
        {
          coinId: a.coinId,
          timeframe: a.timeframe,
          lastQuarantineReason: a.lastQuarantineReason,
          perCoinQuarantinedCount: a.perCoinQuarantinedCount,
          pooledQuarantinedCount: a.pooledQuarantinedCount,
        },
        `Quarantine: slot ${a.coinId}/${a.timeframe} has no servable model — every per-coin AND pooled version is quarantined (last reason: ${a.lastQuarantineReason ?? "unknown"})`,
      );
    }
  }
  activeAlerts = next;
  return next;
}

/** Snapshot of currently-active starvation alerts (cached from the last
 *  sweep). Cheap — does not hit the DB. */
export function getActiveSlotStarvationAlerts(): SlotStarvationAlert[] {
  return activeAlerts.slice();
}

/** Test seam — clear in-memory alert cache. */
export function __resetSlotStarvationAlerts(): void {
  activeAlerts = [];
}
