// Disables (brain, timeframe) horizons whose recent directional
// accuracy is below the coin-flip floor. Same thresholds as the
// /crypto/brain/accuracy endpoint so dashboard "WEAK" rows == trader
// gate. Lazy refresh, soft-fail keeps last snapshot on DB error.

import { sql } from "drizzle-orm";
import { db } from "@workspace/db";
import {
  HORIZON_WEAK_DIRECTIONAL_FLOOR_PCT,
  HORIZON_WEAK_MIN_RESOLVED,
} from "./trading-constants";

const REFRESH_MS = 60 * 1000;
const WINDOW_HOURS = 24;

let disabledSet: Set<string> = new Set();
let lastRefreshedAt = 0;
let refreshInFlight: Promise<void> | null = null;

async function refreshDisabledHorizons(): Promise<void> {
  try {
    const rows = await db.execute(sql`
      SELECT
        CASE
          WHEN reasoning LIKE '%[BRAIN=QUANT]%' THEN 'QUANT'
          WHEN reasoning LIKE '%[BRAIN=LLM]%'   THEN 'LLM'
          ELSE 'untagged'
        END AS brain,
        timeframe,
        COUNT(*) FILTER (WHERE direction != 'stable' AND outcome IS NOT NULL AND outcome::text != 'pending') AS directional_resolved,
        COUNT(*) FILTER (WHERE direction != 'stable' AND outcome::text = 'correct') AS directional_correct
      FROM predictions
      WHERE created_at > NOW() - (${WINDOW_HOURS}::int * INTERVAL '1 hour')
        AND source IS DISTINCT FROM 'prior'
      GROUP BY 1, 2
    `);
    const next = new Set<string>();
    for (const raw of rows.rows as Array<Record<string, unknown>>) {
      const brain = String(raw.brain ?? "");
      const timeframe = String(raw.timeframe ?? "");
      const dirResolved = Number(raw.directional_resolved ?? 0);
      const dirCorrect = Number(raw.directional_correct ?? 0);
      if (!brain || !timeframe) continue;
      if (dirResolved < HORIZON_WEAK_MIN_RESOLVED) continue;
      const dirAccPct = (100 * dirCorrect) / dirResolved;
      if (dirAccPct < HORIZON_WEAK_DIRECTIONAL_FLOOR_PCT) {
        next.add(`${brain}:${timeframe}`);
      }
    }
    disabledSet = next;
    lastRefreshedAt = Date.now();
  } catch {
    lastRefreshedAt = Date.now();
  }
}

async function ensureFresh(): Promise<void> {
  if (Date.now() - lastRefreshedAt < REFRESH_MS) return;
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = refreshDisabledHorizons().finally(() => {
    refreshInFlight = null;
  });
  await refreshInFlight;
}

// True when the (brain, timeframe) horizon is disabled by weak signal.
export async function isHorizonDisabled(
  brain: "QUANT" | "LLM",
  timeframe: string,
): Promise<boolean> {
  await ensureFresh();
  return disabledSet.has(`${brain}:${timeframe}`);
}

export function getDisabledHorizonsSnapshot(): {
  disabled: string[];
  lastRefreshedAt: number;
  windowHours: number;
  floorPct: number;
  minResolved: number;
} {
  return {
    disabled: Array.from(disabledSet).sort(),
    lastRefreshedAt,
    windowHours: WINDOW_HOURS,
    floorPct: HORIZON_WEAK_DIRECTIONAL_FLOOR_PCT,
    minResolved: HORIZON_WEAK_MIN_RESOLVED,
  };
}
