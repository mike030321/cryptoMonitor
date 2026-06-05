/**
 * Minimum Truthful Trading Mode (MTTM) — task #614.
 * Paper-only lane restricted to the 16 promoted lightgbm slots with
 * tighter risk + auto-disable. Read-only on global frictions/gates.
 * Settings in app_settings, cached 5s, sync accessor for paper-trader.
 */
import {
  db,
  appSettingsTable,
  paperTradesTable,
  modelRegistryTable,
} from "@workspace/db";
import { and, eq, gt, inArray, isNotNull } from "drizzle-orm";
import { logger } from "./logger";
import { notifyMttmAutoDisabled } from "./mttm-disable-notifier";

export const MTTM_ENABLED_KEY = "mttm_enabled";
export const MTTM_UNIVERSE_KEY = "mttm_universe";
export const MTTM_MAX_POSITION_PCT_KEY = "mttm_max_position_pct";
export const MTTM_CONSECUTIVE_LOSS_CAP_KEY = "mttm_consecutive_loss_cap";
export const MTTM_N10_POST_FEE_CAP_PCT_KEY = "mttm_n10_post_fee_cap_pct";
export const MTTM_DISABLE_REASON_KEY = "mttm_disable_reason";
export const MTTM_ENABLED_AT_KEY = "mttm_enabled_at";
// Task #659 — single keyed JSON object for the diagnostic-sandbox lane.
// All DS state (mode, btc_version, dd_pct, n_neg_pnl) round-trips
// through this one app_settings row. Versioned so we can evolve the
// shape without colliding with any historical layout.
export const MTTM_DIAGNOSTIC_SANDBOX_KEY = "mttm_diagnostic_sandbox_v1";

export const MTTM_KEYS = [
  MTTM_ENABLED_KEY,
  MTTM_UNIVERSE_KEY,
  MTTM_MAX_POSITION_PCT_KEY,
  MTTM_CONSECUTIVE_LOSS_CAP_KEY,
  MTTM_N10_POST_FEE_CAP_PCT_KEY,
  MTTM_DISABLE_REASON_KEY,
  MTTM_ENABLED_AT_KEY,
  MTTM_DIAGNOSTIC_SANDBOX_KEY,
] as const;

/** Diagnostic-sandbox mode constants — hard-pinned, NOT operator-overridable. */
export const MTTM_DIAGNOSTIC_SANDBOX_COIN = "bitcoin";
export const MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME = "5m";
/** Fixed 0.5% sizing — under-confident calibration is unsafe at full Kelly. */
export const MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT = 0.005;
/** Drawdown floor (–5%): peak-to-trough on the equity curve since enable. */
export const MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT = -0.05;
/** n≥50 + cumulative PnL<0 ⇒ trip auto-disable (calibration is not paying). */
export const MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL = 50;
/** Single-position lane: BTC/5m is the only slot, only one open at a time. */
export const MTTM_DIAGNOSTIC_SANDBOX_MAX_OPEN_POSITIONS = 1;
/** Rolling review window in trades: stats reset over this horizon. */
export const MTTM_DIAGNOSTIC_SANDBOX_ROLLING_WINDOW_TRADES = 100;

export type MttmMode = "default" | "diagnostic_sandbox";

/** Stable identity-string for `mttm_disable_reason.reason` from
 * diagnostic-sandbox-only triggers. */
export type MttmDiagnosticSandboxBreach =
  | "diagnostic_drawdown_exceeded"
  | "diagnostic_negative_pnl_at_review"
  | "diagnostic_universe_drift_detected"
  | "diagnostic_scope_drift_detected"
  | "diagnostic_unauthorized_under_confident_serving";

export interface MttmSlot {
  coinId: string;
  timeframe: string;
  version: string;
}

/**
 * Default MTTM universe — the 16 slots audited as promoted+lightgbm in
 * the task spec (task-614.md). Used as the seed value when no
 * `mttm_universe` row exists; an operator can override via the admin
 * route. The audit script is the gate that decides whether the live
 * registry actually matches this list before MTTM may enable.
 */
export const DEFAULT_MTTM_UNIVERSE: MttmSlot[] = [
  { coinId: "bonk",                    timeframe: "6h", version: "20260425T102133Z" },
  { coinId: "bonk",                    timeframe: "1d", version: "20260425T104026Z" },
  { coinId: "celestia",                timeframe: "6h", version: "20260425T102205Z" },
  { coinId: "celestia",                timeframe: "1d", version: "20260425T103335Z" },
  { coinId: "dogwifcoin",              timeframe: "6h", version: "20260425T102246Z" },
  { coinId: "dogwifcoin",              timeframe: "1d", version: "20260425T103353Z" },
  { coinId: "floki-inu",               timeframe: "6h", version: "20260425T102320Z" },
  { coinId: "floki-inu",               timeframe: "1d", version: "20260425T104054Z" },
  { coinId: "injective-protocol",      timeframe: "6h", version: "20260425T102353Z" },
  { coinId: "injective-protocol",      timeframe: "1d", version: "20260425T103444Z" },
  { coinId: "jupiter-exchange-solana", timeframe: "6h", version: "20260425T102430Z" },
  { coinId: "jupiter-exchange-solana", timeframe: "1d", version: "20260425T103507Z" },
  { coinId: "pepe",                    timeframe: "6h", version: "20260425T102509Z" },
  { coinId: "pepe",                    timeframe: "1d", version: "20260425T104124Z" },
  { coinId: "render-token",            timeframe: "6h", version: "20260425T102545Z" },
  { coinId: "render-token",            timeframe: "1d", version: "20260425T103552Z" },
];

export const MTTM_DEFAULT_MAX_POSITION_PCT = 0.05;
export const MTTM_DEFAULT_CONSECUTIVE_LOSS_CAP = 5;
export const MTTM_DEFAULT_N10_POST_FEE_CAP_PCT = -0.02;

export interface MttmDisableReason {
  reason:
    | "consecutive_losses"
    | "n10_post_fee"
    | "manual"
    // Task #659 — diagnostic-sandbox-only triggers. Listed inline so a
    // disable reason persisted under this enum still type-checks.
    | "diagnostic_drawdown_exceeded"
    | "diagnostic_negative_pnl_at_review"
    | "diagnostic_universe_drift_detected"
    | "diagnostic_scope_drift_detected"
    | "diagnostic_unauthorized_under_confident_serving";
  detail: string;
  trippedAt: string;
  consecutiveLosses?: number;
  nTrades?: number;
  postFeePnlPct?: number;
  /** Diagnostic-sandbox-only metrics, populated by the DS evaluator. */
  drawdownPct?: number;
  cumulativePnlPct?: number;
}

export interface MttmConfig {
  enabled: boolean;
  enabledAt: string | null;
  universe: MttmSlot[];
  maxPositionPct: number;
  consecutiveLossCap: number;
  n10PostFeeCapPct: number;
  disableReason: MttmDisableReason | null;
  /** Lookup set "coinId|timeframe" for O(1) whitelist check. */
  universeKeys: Set<string>;
  /** Task #659 — active MTTM lane (default 16-slot or BTC/5m diagnostic). */
  mode: MttmMode;
  /** Diagnostic-sandbox-only thresholds, populated whether or not the
   * lane is active so the dashboard can render the policy panel. */
  diagnosticSandbox: {
    /** Current calibrated BTC/5m version pinned to the DS lane.
     * `null` until `promote_shadow_to_serving` stamps a version. */
    btcVersion: string | null;
    /** Drawdown floor (peak-to-trough) since enable. Negative number. */
    drawdownPct: number;
    /** Trade count threshold for the n≥N + PnL<0 rule. */
    nNegPnl: number;
    /** Constants echoed for clients (banner / API). */
    coinId: string;
    timeframe: string;
    fixedPositionPct: number;
  };
}

const CACHE_TTL_MS = 5_000;

interface CacheEntry { config: MttmConfig; expiresAt: number }
let cache: CacheEntry | null = null;

export function slotKey(coinId: string, timeframe: string): string {
  return `${coinId}|${timeframe}`;
}

function buildUniverseKeys(universe: MttmSlot[]): Set<string> {
  const s = new Set<string>();
  for (const u of universe) s.add(slotKey(u.coinId, u.timeframe));
  return s;
}

function defaultConfig(): MttmConfig {
  return {
    enabled: false,
    enabledAt: null,
    universe: DEFAULT_MTTM_UNIVERSE,
    maxPositionPct: MTTM_DEFAULT_MAX_POSITION_PCT,
    consecutiveLossCap: MTTM_DEFAULT_CONSECUTIVE_LOSS_CAP,
    n10PostFeeCapPct: MTTM_DEFAULT_N10_POST_FEE_CAP_PCT,
    disableReason: null,
    universeKeys: buildUniverseKeys(DEFAULT_MTTM_UNIVERSE),
    mode: "default",
    diagnosticSandbox: {
      btcVersion: null,
      drawdownPct: MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT,
      nNegPnl: MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL,
      coinId: MTTM_DIAGNOSTIC_SANDBOX_COIN,
      timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      fixedPositionPct: MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
    },
  };
}

/**
 * Full typed shape of `mttm_diagnostic_sandbox_v1`. Operator-mutable
 * fields: `mode`, `btc_version`, `dd_pct`, `n_neg_pnl`. The remaining
 * fields are the pinned constants + last-known-good runtime snapshot
 * (label, universe, fixed_position_pct, enabled, review). Constants
 * are re-stamped on every write so the row alone is a complete audit
 * record of the lane state.
 */
interface DiagnosticSandboxRow {
  mode?: MttmMode;
  enabled?: boolean;
  label?: string;
  universe?: { coin_id: string; timeframe: string }[];
  fixed_position_pct?: number;
  max_open_positions?: number;
  btc_version?: string | null;
  loss_limits?: {
    drawdown_floor_pct?: number;
    n_neg_pnl_threshold?: number;
  };
  review_windows?: {
    initial_review_n_trades?: number;
    rolling_window_trades?: number;
  };
  review?: {
    enabled_at?: string | null;
    disable_reason?: string | null;
    auto_disabled?: boolean;
  };
  // Backwards-compat: pre-rename `limits` and pre-v1 flat fields.
  limits?: {
    drawdown_floor_pct?: number;
    n_neg_pnl_threshold?: number;
  };
  dd_pct?: number;
  n_neg_pnl?: number;
}

function parseDiagnosticSandboxRow(v: unknown): DiagnosticSandboxRow | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const out: DiagnosticSandboxRow = {};
  if (o.mode === "default" || o.mode === "diagnostic_sandbox") out.mode = o.mode;
  if (typeof o.enabled === "boolean") out.enabled = o.enabled;
  if (typeof o.label === "string") out.label = o.label;
  if (Array.isArray(o.universe)) {
    const u: { coin_id: string; timeframe: string }[] = [];
    for (const e of o.universe) {
      if (
        e &&
        typeof e === "object" &&
        typeof (e as { coin_id?: unknown }).coin_id === "string" &&
        typeof (e as { timeframe?: unknown }).timeframe === "string"
      ) {
        u.push({
          coin_id: (e as { coin_id: string }).coin_id,
          timeframe: (e as { timeframe: string }).timeframe,
        });
      }
    }
    out.universe = u;
  }
  if (typeof o.fixed_position_pct === "number" && Number.isFinite(o.fixed_position_pct)) {
    out.fixed_position_pct = o.fixed_position_pct;
  }
  if (typeof o.max_open_positions === "number" && Number.isFinite(o.max_open_positions)) {
    out.max_open_positions = Math.max(1, Math.floor(o.max_open_positions));
  }
  if (o.btc_version === null || typeof o.btc_version === "string") {
    out.btc_version = o.btc_version;
  }
  const parseLossLimits = (raw: unknown): DiagnosticSandboxRow["loss_limits"] | null => {
    if (!raw || typeof raw !== "object") return null;
    const lim = raw as Record<string, unknown>;
    const ll: NonNullable<DiagnosticSandboxRow["loss_limits"]> = {};
    if (typeof lim.drawdown_floor_pct === "number" && Number.isFinite(lim.drawdown_floor_pct)) {
      ll.drawdown_floor_pct = lim.drawdown_floor_pct;
    }
    if (typeof lim.n_neg_pnl_threshold === "number" && Number.isFinite(lim.n_neg_pnl_threshold)) {
      ll.n_neg_pnl_threshold = lim.n_neg_pnl_threshold;
    }
    return ll;
  };
  const ll = parseLossLimits(o.loss_limits);
  if (ll) out.loss_limits = ll;
  const llLegacy = parseLossLimits(o.limits);
  if (llLegacy) out.limits = llLegacy;
  if (o.review_windows && typeof o.review_windows === "object") {
    const rw = o.review_windows as Record<string, unknown>;
    out.review_windows = {};
    if (typeof rw.initial_review_n_trades === "number" && Number.isFinite(rw.initial_review_n_trades)) {
      out.review_windows.initial_review_n_trades = Math.max(1, Math.floor(rw.initial_review_n_trades));
    }
    if (typeof rw.rolling_window_trades === "number" && Number.isFinite(rw.rolling_window_trades)) {
      out.review_windows.rolling_window_trades = Math.max(1, Math.floor(rw.rolling_window_trades));
    }
  }
  if (o.review && typeof o.review === "object") {
    const rev = o.review as Record<string, unknown>;
    out.review = {};
    if (rev.enabled_at === null || typeof rev.enabled_at === "string") {
      out.review.enabled_at = rev.enabled_at;
    }
    if (rev.disable_reason === null || typeof rev.disable_reason === "string") {
      out.review.disable_reason = rev.disable_reason;
    }
    if (typeof rev.auto_disabled === "boolean") out.review.auto_disabled = rev.auto_disabled;
  }
  // Legacy flat fields (still accepted on read).
  if (typeof o.dd_pct === "number" && Number.isFinite(o.dd_pct)) out.dd_pct = o.dd_pct;
  if (typeof o.n_neg_pnl === "number" && Number.isFinite(o.n_neg_pnl)) {
    out.n_neg_pnl = o.n_neg_pnl;
  }
  return out;
}

/**
 * Build the canonical full v1 row payload. Pinned constants come from
 * code; operator-mutable fields fall back to defaults. The result is
 * what every write persists — the row is always self-describing.
 */
function buildFullDiagnosticSandboxRow(partial: DiagnosticSandboxRow): {
  mode: MttmMode;
  enabled: boolean;
  label: string;
  universe: { coin_id: string; timeframe: string }[];
  fixed_position_pct: number;
  max_open_positions: number;
  btc_version: string | null;
  loss_limits: { drawdown_floor_pct: number; n_neg_pnl_threshold: number };
  review_windows: { initial_review_n_trades: number; rolling_window_trades: number };
  review: { enabled_at: string | null; disable_reason: string | null; auto_disabled: boolean };
} {
  const mode: MttmMode = partial.mode ?? "default";
  const ddFromLoss = partial.loss_limits?.drawdown_floor_pct;
  const ddFromLegacy = partial.limits?.drawdown_floor_pct;
  const nFromLoss = partial.loss_limits?.n_neg_pnl_threshold;
  const nFromLegacy = partial.limits?.n_neg_pnl_threshold;
  const dd = ddFromLoss !== undefined && ddFromLoss < 0
    ? ddFromLoss
    : ddFromLegacy !== undefined && ddFromLegacy < 0
    ? ddFromLegacy
    : (partial.dd_pct !== undefined && partial.dd_pct < 0
      ? partial.dd_pct
      : MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_DD_PCT);
  const nNeg = nFromLoss !== undefined
    ? Math.max(1, Math.floor(nFromLoss))
    : nFromLegacy !== undefined
    ? Math.max(1, Math.floor(nFromLegacy))
    : (partial.n_neg_pnl !== undefined
      ? Math.max(1, Math.floor(partial.n_neg_pnl))
      : MTTM_DIAGNOSTIC_SANDBOX_DEFAULT_N_NEG_PNL);
  const initialReviewN = partial.review_windows?.initial_review_n_trades ?? nNeg;
  const rolling = partial.review_windows?.rolling_window_trades
    ?? MTTM_DIAGNOSTIC_SANDBOX_ROLLING_WINDOW_TRADES;
  return {
    mode,
    enabled: partial.enabled ?? (mode === "diagnostic_sandbox"),
    label: getDiagnosticSandboxLabel(),
    universe: [
      {
        coin_id: MTTM_DIAGNOSTIC_SANDBOX_COIN,
        timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
      },
    ],
    fixed_position_pct: MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT,
    max_open_positions: MTTM_DIAGNOSTIC_SANDBOX_MAX_OPEN_POSITIONS,
    btc_version: partial.btc_version ?? null,
    loss_limits: { drawdown_floor_pct: dd, n_neg_pnl_threshold: nNeg },
    review_windows: {
      initial_review_n_trades: Math.max(1, Math.floor(initialReviewN)),
      rolling_window_trades: Math.max(1, Math.floor(rolling)),
    },
    review: {
      enabled_at: partial.review?.enabled_at ?? null,
      disable_reason: partial.review?.disable_reason ?? null,
      auto_disabled: partial.review?.auto_disabled ?? false,
    },
  };
}

function parseUniverse(v: unknown): MttmSlot[] | null {
  if (!Array.isArray(v)) return null;
  const out: MttmSlot[] = [];
  for (const e of v) {
    if (
      e &&
      typeof e === "object" &&
      typeof (e as MttmSlot).coinId === "string" &&
      typeof (e as MttmSlot).timeframe === "string" &&
      typeof (e as MttmSlot).version === "string"
    ) {
      out.push({
        coinId: (e as MttmSlot).coinId,
        timeframe: (e as MttmSlot).timeframe,
        version: (e as MttmSlot).version,
      });
    }
  }
  return out;
}

function parseDisableReason(v: unknown): MttmDisableReason | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  if (typeof o.reason !== "string" || typeof o.detail !== "string") return null;
  if (typeof o.trippedAt !== "string") return null;
  const r = o.reason as MttmDisableReason["reason"];
  // Task #659 — accept the diagnostic-sandbox reason codes alongside
  // the legacy default-lane codes so a tripped DS lane round-trips
  // through app_settings.
  const ALLOWED: ReadonlySet<MttmDisableReason["reason"]> = new Set([
    "consecutive_losses",
    "n10_post_fee",
    "manual",
    "diagnostic_drawdown_exceeded",
    "diagnostic_negative_pnl_at_review",
    "diagnostic_universe_drift_detected",
    "diagnostic_scope_drift_detected",
    "diagnostic_unauthorized_under_confident_serving",
  ]);
  if (!ALLOWED.has(r)) return null;
  const out: MttmDisableReason = {
    reason: r,
    detail: o.detail,
    trippedAt: o.trippedAt,
  };
  if (typeof o.consecutiveLosses === "number") out.consecutiveLosses = o.consecutiveLosses;
  if (typeof o.nTrades === "number") out.nTrades = o.nTrades;
  if (typeof o.postFeePnlPct === "number") out.postFeePnlPct = o.postFeePnlPct;
  if (typeof o.drawdownPct === "number") out.drawdownPct = o.drawdownPct;
  if (typeof o.cumulativePnlPct === "number") {
    out.cumulativePnlPct = o.cumulativePnlPct;
  }
  return out;
}

function unwrapEnabled(v: unknown): boolean {
  if (typeof v === "boolean") return v;
  if (v && typeof v === "object" && typeof (v as { enabled?: unknown }).enabled === "boolean") {
    return (v as { enabled: boolean }).enabled;
  }
  return false;
}

function unwrapNumber(v: unknown, fallback: number): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (v && typeof v === "object" && typeof (v as { value?: unknown }).value === "number") {
    const n = (v as { value: number }).value;
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function unwrapString(v: unknown): string | null {
  if (typeof v === "string") return v;
  if (v && typeof v === "object" && typeof (v as { value?: unknown }).value === "string") {
    return (v as { value: string }).value;
  }
  return null;
}

export async function getMttmConfig(): Promise<MttmConfig> {
  if (cache && cache.expiresAt > Date.now()) return cache.config;
  let cfg = defaultConfig();
  try {
    const rows = await db
      .select()
      .from(appSettingsTable)
      .where(inArray(appSettingsTable.key, [...MTTM_KEYS]));
    const map = new Map<string, unknown>();
    for (const r of rows) map.set(r.key, r.value);

    if (map.has(MTTM_ENABLED_KEY)) cfg.enabled = unwrapEnabled(map.get(MTTM_ENABLED_KEY));
    if (map.has(MTTM_ENABLED_AT_KEY)) cfg.enabledAt = unwrapString(map.get(MTTM_ENABLED_AT_KEY));
    if (map.has(MTTM_UNIVERSE_KEY)) {
      const u = parseUniverse(map.get(MTTM_UNIVERSE_KEY));
      if (u && u.length > 0) {
        cfg.universe = u;
        cfg.universeKeys = buildUniverseKeys(u);
      }
    }
    if (map.has(MTTM_MAX_POSITION_PCT_KEY)) {
      cfg.maxPositionPct = unwrapNumber(
        map.get(MTTM_MAX_POSITION_PCT_KEY),
        MTTM_DEFAULT_MAX_POSITION_PCT,
      );
    }
    if (map.has(MTTM_CONSECUTIVE_LOSS_CAP_KEY)) {
      cfg.consecutiveLossCap = Math.max(
        1,
        Math.floor(
          unwrapNumber(
            map.get(MTTM_CONSECUTIVE_LOSS_CAP_KEY),
            MTTM_DEFAULT_CONSECUTIVE_LOSS_CAP,
          ),
        ),
      );
    }
    if (map.has(MTTM_N10_POST_FEE_CAP_PCT_KEY)) {
      cfg.n10PostFeeCapPct = unwrapNumber(
        map.get(MTTM_N10_POST_FEE_CAP_PCT_KEY),
        MTTM_DEFAULT_N10_POST_FEE_CAP_PCT,
      );
    }
    if (map.has(MTTM_DISABLE_REASON_KEY)) {
      cfg.disableReason = parseDisableReason(map.get(MTTM_DISABLE_REASON_KEY));
    }
    // Task #659 — single-key DS state. Reject malformed dd_pct (must
    // be negative; a positive floor would self-trip on first loss).
    if (map.has(MTTM_DIAGNOSTIC_SANDBOX_KEY)) {
      const ds = parseDiagnosticSandboxRow(map.get(MTTM_DIAGNOSTIC_SANDBOX_KEY));
      if (ds) {
        if (ds.mode) cfg.mode = ds.mode;
        // The DS row's `enabled` bit is only meaningful while
        // `mode === diagnostic_sandbox`; in default mode the legacy
        // `mttm_enabled` key (already applied above) remains the sole
        // source of truth, so flipping the DS row off cannot disable
        // the default lane.
        if (cfg.mode === "diagnostic_sandbox") {
          cfg.enabled = ds.enabled ?? true;
          if (cfg.enabled && cfg.enabledAt === null) {
            cfg.enabledAt = ds.review?.enabled_at ?? new Date().toISOString();
          }
        }
        if (ds.btc_version !== undefined && ds.btc_version !== null) {
          cfg.diagnosticSandbox.btcVersion = ds.btc_version;
        }
        // Prefer the v1 `loss_limits.*` block; accept legacy `limits.*`
        // and pre-v1 flat fields for backwards-compat.
        const ddCandidate = ds.loss_limits?.drawdown_floor_pct
          ?? ds.limits?.drawdown_floor_pct
          ?? ds.dd_pct;
        if (ddCandidate !== undefined && ddCandidate < 0) {
          cfg.diagnosticSandbox.drawdownPct = ddCandidate;
        }
        const nCandidate = ds.loss_limits?.n_neg_pnl_threshold
          ?? ds.limits?.n_neg_pnl_threshold
          ?? ds.n_neg_pnl;
        if (nCandidate !== undefined) {
          cfg.diagnosticSandbox.nNegPnl = Math.max(1, Math.floor(nCandidate));
        }
      }
    }
    // Apply DS hard pins (universe + sizing) AFTER every other read.
    if (cfg.mode === "diagnostic_sandbox") {
      const pinned: MttmSlot[] = [
        {
          coinId: MTTM_DIAGNOSTIC_SANDBOX_COIN,
          timeframe: MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
          // Use the stamped version when present, else a stable
          // sentinel so the universe still type-checks. Trade
          // execution must additionally check `btcVersion !== null`
          // before placing an order.
          version: cfg.diagnosticSandbox.btcVersion ?? "PENDING_PROMOTION",
        },
      ];
      cfg.universe = pinned;
      cfg.universeKeys = buildUniverseKeys(pinned);
      cfg.maxPositionPct = MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT;
    }
  } catch (err) {
    logger.warn({ err }, "mttm: failed to read app_settings — falling back to defaults");
    cfg = defaultConfig();
  }
  cache = { config: cfg, expiresAt: Date.now() + CACHE_TTL_MS };
  return cfg;
}

/**
 * Task #659 — convenience predicate. Returns true iff the diagnostic
 * sandbox lane is active AND has a real BTC version stamped (i.e. ready
 * to drive trades, not the "PENDING_PROMOTION" sentinel).
 */
export function isDiagnosticSandboxReady(cfg: MttmConfig): boolean {
  return (
    cfg.enabled &&
    cfg.mode === "diagnostic_sandbox" &&
    cfg.diagnosticSandbox.btcVersion !== null
  );
}

export async function isMttmEnabled(): Promise<boolean> {
  return (await getMttmConfig()).enabled;
}

/**
 * Synchronous accessor for the per-decision hot path. Returns the most
 * recently fetched config, or `null` if the cache has not been warmed
 * yet. Callers MUST treat `null` as "MTTM unknown — proceed as before"
 * (fail-open) so a cold start cannot accidentally block trading.
 *
 * The cache is warmed by `getMttmConfig()` which the periodic monitor
 * loop calls at the start of every cycle.
 */
export function getMttmConfigCached(): MttmConfig | null {
  if (cache && cache.expiresAt > Date.now()) return cache.config;
  return null;
}

export async function setMttmEnabled(
  enabled: boolean,
  opts?: { clearDisableReason?: boolean; enabledAt?: Date },
): Promise<MttmConfig> {
  const at = (opts?.enabledAt ?? new Date()).toISOString();
  await db.transaction(async (tx) => {
    await tx
      .insert(appSettingsTable)
      .values({ key: MTTM_ENABLED_KEY, value: { enabled } })
      .onConflictDoUpdate({
        target: appSettingsTable.key,
        set: { value: { enabled }, updatedAt: new Date() },
      });
    if (enabled) {
      await tx
        .insert(appSettingsTable)
        .values({ key: MTTM_ENABLED_AT_KEY, value: { value: at } })
        .onConflictDoUpdate({
          target: appSettingsTable.key,
          set: { value: { value: at }, updatedAt: new Date() },
        });
    }
    if (opts?.clearDisableReason) {
      await tx
        .delete(appSettingsTable)
        .where(eq(appSettingsTable.key, MTTM_DISABLE_REASON_KEY));
    }
  });
  cache = null;
  const cfg = await getMttmConfig();
  logger.info({ enabled, enabledAt: cfg.enabledAt }, "mttm: enabled flag updated");
  return cfg;
}

export async function setMttmUniverse(universe: MttmSlot[]): Promise<MttmConfig> {
  await db
    .insert(appSettingsTable)
    .values({ key: MTTM_UNIVERSE_KEY, value: universe })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: universe, updatedAt: new Date() },
    });
  cache = null;
  return getMttmConfig();
}

export async function setMttmDisableReason(reason: MttmDisableReason): Promise<void> {
  await db
    .insert(appSettingsTable)
    .values({ key: MTTM_DISABLE_REASON_KEY, value: reason })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: reason, updatedAt: new Date() },
    });
  cache = null;
}

export async function clearMttmDisableReason(): Promise<void> {
  await db.delete(appSettingsTable).where(eq(appSettingsTable.key, MTTM_DISABLE_REASON_KEY));
  cache = null;
}

/** Read the persisted DS row (or {} if absent) for an atomic mutate-then-write. */
async function readDiagnosticSandboxRow(): Promise<DiagnosticSandboxRow> {
  const rows = await db
    .select()
    .from(appSettingsTable)
    .where(eq(appSettingsTable.key, MTTM_DIAGNOSTIC_SANDBOX_KEY));
  if (rows.length === 0) return {};
  return parseDiagnosticSandboxRow(rows[0].value) ?? {};
}

async function writeDiagnosticSandboxRow(next: DiagnosticSandboxRow): Promise<void> {
  // Re-stamp the full canonical shape so the persisted row is always
  // self-describing (constants + last-known mutable state in one blob).
  const full = buildFullDiagnosticSandboxRow(next);
  await db
    .insert(appSettingsTable)
    .values({ key: MTTM_DIAGNOSTIC_SANDBOX_KEY, value: full })
    .onConflictDoUpdate({
      target: appSettingsTable.key,
      set: { value: full, updatedAt: new Date() },
    });
}

/**
 * Task #666 — flip the v1 DS row to the auto-disabled state.
 *
 * `setMttmEnabled(false)` only writes the legacy `mttm_enabled` row,
 * but in DS mode `getMttmConfig()` overrides `cfg.enabled` from the v1
 * row's `enabled` bit. Without also flipping the v1 row, the lane will
 * silently re-arm if `mttm_disable_reason` is ever cleared (operator
 * action or a bug). This helper persists the disable on both
 * authoritative reads and stamps `review.auto_disabled = true` so the
 * row alone is a complete audit trail of the trip.
 */
async function markDiagnosticSandboxAutoDisabled(
  reason: MttmDisableReason,
): Promise<void> {
  const cur = await readDiagnosticSandboxRow();
  const reviewBase = cur.review ?? {};
  const next: DiagnosticSandboxRow = {
    ...cur,
    enabled: false,
    review: {
      enabled_at: reviewBase.enabled_at ?? null,
      disable_reason: reason.reason,
      auto_disabled: true,
    },
  };
  await writeDiagnosticSandboxRow(next);
  cache = null;
}

/** Task #659 — flip DS lane mode by merging into mttm_diagnostic_sandbox_v1.
 *
 * The v1 row is the single source of truth: switching mode also
 * (un)sets the authoritative `enabled` flag and stamps
 * `review.enabled_at`. Operator no longer has to write two keys.
 */
export async function setMttmMode(mode: MttmMode): Promise<MttmConfig> {
  const cur = await readDiagnosticSandboxRow();
  const enabled = mode === "diagnostic_sandbox";
  const reviewBase = cur.review ?? {};
  const next: DiagnosticSandboxRow = {
    ...cur,
    mode,
    enabled,
    review: {
      enabled_at: enabled
        ? (reviewBase.enabled_at ?? new Date().toISOString())
        : null,
      disable_reason: enabled ? null : (reviewBase.disable_reason ?? null),
      auto_disabled: enabled ? false : (reviewBase.auto_disabled ?? false),
    },
  };
  await writeDiagnosticSandboxRow(next);
  cache = null;
  const cfg = await getMttmConfig();
  logger.info(
    { mode, effectiveMode: cfg.mode, enabled: cfg.enabled },
    "mttm: mode updated",
  );
  return cfg;
}

/** Task #659 — stamp the calibrated BTC/5m version into the DS row. */
export async function setDiagnosticSandboxBtcVersion(
  version: string | null,
): Promise<MttmConfig> {
  const cur = await readDiagnosticSandboxRow();
  await writeDiagnosticSandboxRow({ ...cur, btc_version: version });
  cache = null;
  const cfg = await getMttmConfig();
  logger.info(
    { version, ready: isDiagnosticSandboxReady(cfg) },
    "mttm: DS btc_version updated",
  );
  return cfg;
}

/**
 * Task #659 — DS auto-disable. Two rules over closed trades since
 * enabledAt: peak-to-trough DD ≤ floor (default -5%), or n ≥ threshold
 * (default 50) with cumulative PnL < 0. Idempotent.
 */
export async function evaluateDiagnosticSandboxAutoDisable(
  config?: MttmConfig,
): Promise<MttmDisableReason | null> {
  const cfg = config ?? (await getMttmConfig());
  if (!cfg.enabled) return null;
  if (cfg.mode !== "diagnostic_sandbox") return null;
  if (cfg.disableReason) return cfg.disableReason; // already tripped
  if (cfg.universeKeys.size === 0) return null;
  if (!cfg.enabledAt) return null;

  const enabledAt = new Date(cfg.enabledAt);
  if (Number.isNaN(enabledAt.getTime())) return null;

  const rows = await db
    .select({
      coinId: paperTradesTable.coinId,
      timeframe: paperTradesTable.timeframe,
      pnl: paperTradesTable.pnl,
      pnlPercent: paperTradesTable.pnlPercent,
      positionSize: paperTradesTable.positionSize,
      closedAt: paperTradesTable.closedAt,
    })
    .from(paperTradesTable)
    .where(
      and(
        eq(paperTradesTable.status, "closed"),
        isNotNull(paperTradesTable.closedAt),
        gt(paperTradesTable.closedAt, enabledAt),
      ),
    );

  const dsTrades = rows
    .filter((r) => cfg.universeKeys.has(slotKey(r.coinId, r.timeframe)))
    .sort((a, b) => {
      const ta = a.closedAt ? a.closedAt.getTime() : 0;
      const tb = b.closedAt ? b.closedAt.getTime() : 0;
      return ta - tb;
    });

  if (dsTrades.length === 0) return null;

  // Equity walk in percent space; row pnlPercent is per-position so we
  // weight by the 0.5% sizing pin to get the account return.
  let equity = 1.0;
  let peak = 1.0;
  let trough = 0.0; // worst peak-to-trough seen
  let cumulativePnlPct = 0.0;
  const sizingPct = MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT;
  for (const t of dsTrades) {
    const r = t.pnlPercent ?? 0;
    const accountReturn = r * sizingPct;
    cumulativePnlPct += accountReturn;
    equity *= 1 + accountReturn;
    if (equity > peak) peak = equity;
    const dd = equity / peak - 1; // ≤ 0
    if (dd < trough) trough = dd;
  }

  if (trough <= cfg.diagnosticSandbox.drawdownPct) {
    const reason: MttmDisableReason = {
      reason: "diagnostic_drawdown_exceeded",
      detail:
        `Diagnostic sandbox auto-disabled — peak-to-trough drawdown ` +
        `${(trough * 100).toFixed(2)}% breached floor ` +
        `${(cfg.diagnosticSandbox.drawdownPct * 100).toFixed(2)}% ` +
        `over ${dsTrades.length} BTC/5m trades.`,
      trippedAt: new Date().toISOString(),
      nTrades: dsTrades.length,
      drawdownPct: trough,
      cumulativePnlPct,
    };
    await setMttmDisableReason(reason);
    await markDiagnosticSandboxAutoDisabled(reason);
    await setMttmEnabled(false);
    logger.warn(
      { drawdownPct: trough, nTrades: dsTrades.length },
      "mttm: diagnostic-sandbox auto-disabled (drawdown floor)",
    );
    void notifyMttmAutoDisabled(reason).catch((err) =>
      logger.warn(
        { err: err instanceof Error ? err.message : String(err) },
        "mttm: diagnostic-sandbox notifier failed (non-fatal)",
      ),
    );
    return reason;
  }

  if (
    dsTrades.length >= cfg.diagnosticSandbox.nNegPnl &&
    cumulativePnlPct < 0
  ) {
    const reason: MttmDisableReason = {
      reason: "diagnostic_negative_pnl_at_review",
      detail:
        `Diagnostic sandbox auto-disabled — ${dsTrades.length} BTC/5m trades ` +
        `with cumulative PnL ${(cumulativePnlPct * 100).toFixed(2)}% < 0 ` +
        `(threshold n=${cfg.diagnosticSandbox.nNegPnl}).`,
      trippedAt: new Date().toISOString(),
      nTrades: dsTrades.length,
      drawdownPct: trough,
      cumulativePnlPct,
    };
    await setMttmDisableReason(reason);
    await markDiagnosticSandboxAutoDisabled(reason);
    await setMttmEnabled(false);
    logger.warn(
      { nTrades: dsTrades.length, cumulativePnlPct },
      "mttm: diagnostic-sandbox auto-disabled (n + neg PnL)",
    );
    void notifyMttmAutoDisabled(reason).catch((err) =>
      logger.warn(
        { err: err instanceof Error ? err.message : String(err) },
        "mttm: diagnostic-sandbox notifier failed (non-fatal)",
      ),
    );
    return reason;
  }

  return null;
}

/**
 * Task #659 — drift-trip helper. Called by the periodic monitor when a
 * universe / scope / promotion-state mismatch is detected against the
 * pinned BTC/5m row. Forcibly disables the lane and stamps a reason
 * the dashboard banner can render. Idempotent — a repeat call after
 * the lane is already disabled is a no-op.
 */
export async function tripDiagnosticSandboxDrift(
  reason: MttmDiagnosticSandboxBreach,
  detail: string,
  config?: MttmConfig,
): Promise<MttmDisableReason | null> {
  const cfg = config ?? (await getMttmConfig());
  if (!cfg.enabled) return null;
  if (cfg.mode !== "diagnostic_sandbox") return null;
  if (cfg.disableReason) return cfg.disableReason;
  if (
    reason !== "diagnostic_universe_drift_detected" &&
    reason !== "diagnostic_scope_drift_detected" &&
    reason !== "diagnostic_unauthorized_under_confident_serving"
  ) {
    throw new Error(
      `tripDiagnosticSandboxDrift: invalid reason ${reason} (must be a drift code)`,
    );
  }
  const out: MttmDisableReason = {
    reason,
    detail,
    trippedAt: new Date().toISOString(),
  };
  await setMttmDisableReason(out);
  await markDiagnosticSandboxAutoDisabled(out);
  await setMttmEnabled(false);
  logger.warn({ reason, detail }, "mttm: diagnostic-sandbox drift trip");
  void notifyMttmAutoDisabled(out).catch((err) =>
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "mttm: diagnostic-sandbox drift notifier failed (non-fatal)",
    ),
  );
  return out;
}

/**
 * Task #659 — read-only metrics snapshot for the DS status endpoint.
 * Equity curve / cumulative PnL / trade count, no DB writes.
 */
export interface DiagnosticSandboxMetrics {
  nTrades: number;
  cumulativePnlPct: number;
  drawdownPct: number;
  reviewsRemaining: number;
  drawdownFloorPct: number;
  nNegPnlThreshold: number;
}

export async function getDiagnosticSandboxMetrics(
  config?: MttmConfig,
): Promise<DiagnosticSandboxMetrics | null> {
  const cfg = config ?? (await getMttmConfig());
  if (cfg.mode !== "diagnostic_sandbox") return null;
  if (!cfg.enabledAt) {
    // Lane staged but never enabled — return zeros so the banner
    // can still render the threshold preview.
    return {
      nTrades: 0,
      cumulativePnlPct: 0,
      drawdownPct: 0,
      reviewsRemaining: cfg.diagnosticSandbox.nNegPnl,
      drawdownFloorPct: cfg.diagnosticSandbox.drawdownPct,
      nNegPnlThreshold: cfg.diagnosticSandbox.nNegPnl,
    };
  }
  const enabledAt = new Date(cfg.enabledAt);
  if (Number.isNaN(enabledAt.getTime())) return null;

  let rows: Array<{ coinId: string; timeframe: string;
    pnlPercent: number | null; closedAt: Date | null }>;
  try {
    rows = await db
      .select({
        coinId: paperTradesTable.coinId,
        timeframe: paperTradesTable.timeframe,
        pnlPercent: paperTradesTable.pnlPercent,
        closedAt: paperTradesTable.closedAt,
      })
      .from(paperTradesTable)
      .where(
        and(
          eq(paperTradesTable.status, "closed"),
          isNotNull(paperTradesTable.closedAt),
          gt(paperTradesTable.closedAt, enabledAt),
        ),
      );
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "mttm: diagnostic-sandbox metrics DB read failed (non-fatal)",
    );
    return null;
  }

  const dsTrades = rows
    .filter((r) => cfg.universeKeys.has(slotKey(r.coinId, r.timeframe)))
    .sort((a, b) => {
      const ta = a.closedAt ? a.closedAt.getTime() : 0;
      const tb = b.closedAt ? b.closedAt.getTime() : 0;
      return ta - tb;
    });

  let equity = 1.0;
  let peak = 1.0;
  let trough = 0.0;
  let cumulativePnlPct = 0.0;
  const sizingPct = MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT;
  for (const t of dsTrades) {
    const r = t.pnlPercent ?? 0;
    const accountReturn = r * sizingPct;
    cumulativePnlPct += accountReturn;
    equity *= 1 + accountReturn;
    if (equity > peak) peak = equity;
    const dd = equity / peak - 1;
    if (dd < trough) trough = dd;
  }

  return {
    nTrades: dsTrades.length,
    cumulativePnlPct,
    drawdownPct: trough,
    reviewsRemaining: Math.max(
      0,
      cfg.diagnosticSandbox.nNegPnl - dsTrades.length,
    ),
    drawdownFloorPct: cfg.diagnosticSandbox.drawdownPct,
    nNegPnlThreshold: cfg.diagnosticSandbox.nNegPnl,
  };
}

/**
 * Task #670 — DS health probe.
 *
 * The auto-disable evaluator computes peak-to-trough drawdown over the
 * full DS lane (every closed BTC/5m trade since `enabledAt`) and trips
 * once the floor is breached. By design that is an after-the-fact
 * surface: by the time it fires the lane is already off and operators
 * have only the disable reason to explain it.
 *
 * The B4 sweep showed the same model can move from -4.52% to -5.55%
 * holdout drawdown with a 12-minute window shift, so once the lane is
 * running live a small drift in the trailing window can silently push
 * the champion over the floor. This helper exposes a cheap "DS health"
 * surface that operators can watch BEFORE the floor trips: the
 * peak-to-trough drawdown computed only over the most recent
 * `windowTrades` closed DS trades, the headroom remaining vs. the floor,
 * and a `needsRefit` boolean that flips to `true` once the trailing
 * drawdown has eaten through the configured warning fraction of the
 * floor (default 80%, so warn at -4% if the floor is -5%).
 *
 * No model fit, no inference, no Python round-trip — same trades the
 * sandbox already realised, just sliced to the trailing window. Cheap
 * enough to recompute on every dashboard poll / cycle without touching
 * the holdout pipeline.
 */
export const MTTM_DIAGNOSTIC_SANDBOX_HEALTH_WARN_FRACTION = 0.8;

export interface DiagnosticSandboxHealth {
  /** True iff DS mode is enabled and we have at least one closed trade. */
  evaluable: boolean;
  /** Stable operator label (mirrors `getDiagnosticSandboxLabel()`). */
  label: string;
  /** Pinned slot (always BTC/5m today, but echoed for clients). */
  coinId: string;
  timeframe: string;
  /** Active calibrated BTC/5m version, or null if none staged. */
  btcVersion: string | null;
  /** Configured trailing window size (most-recent N closed trades). */
  windowTrades: number;
  /** Number of trades actually counted (may be < windowTrades early on). */
  nTradesObserved: number;
  /** Peak-to-trough drawdown over the trailing window (≤ 0). */
  trailingDrawdownPct: number;
  /** Cumulative PnL over the trailing window (account-weighted). */
  trailingPnlPct: number;
  /** Configured drawdown floor (negative, e.g. -0.05). */
  drawdownFloorPct: number;
  /** Warning trip line: `floor * warnFraction` (negative, less negative
   *  than the floor — e.g. -0.04 when floor is -0.05). */
  warnThresholdPct: number;
  /** Fraction of |floor| used to compute `warnThresholdPct`. */
  warnFraction: number;
  /** Distance from trailing drawdown to the floor: `trailing - floor`.
   *  Positive ⇒ headroom remaining; ≤ 0 ⇒ floor breached. */
  headroomPct: number;
  /** True iff trailing drawdown has eaten through the warn threshold or
   *  the floor itself. Operators should stage a refit. */
  needsRefit: boolean;
  /** True iff trailing drawdown is at or past the floor. */
  floorBreached: boolean;
  /** Wall-clock when this snapshot was computed. */
  computedAt: string;
}

interface DiagnosticSandboxHealthCacheEntry {
  health: DiagnosticSandboxHealth;
  expiresAt: number;
  /** Identity hash of the inputs we computed against — invalidate if mode,
   *  floor, window or BTC version change between calls. */
  inputsKey: string;
}

const DS_HEALTH_CACHE_TTL_MS = 30_000;
let dsHealthCache: DiagnosticSandboxHealthCacheEntry | null = null;

function emptyDsHealth(cfg: MttmConfig): DiagnosticSandboxHealth {
  const floor = cfg.diagnosticSandbox.drawdownPct;
  const warnFraction = MTTM_DIAGNOSTIC_SANDBOX_HEALTH_WARN_FRACTION;
  // floor is negative; warn line is closer to zero (less negative).
  const warnThresholdPct = floor * warnFraction;
  return {
    evaluable: false,
    label: getDiagnosticSandboxLabel(),
    coinId: cfg.diagnosticSandbox.coinId,
    timeframe: cfg.diagnosticSandbox.timeframe,
    btcVersion: cfg.diagnosticSandbox.btcVersion,
    windowTrades: cfg.diagnosticSandbox.nNegPnl,
    nTradesObserved: 0,
    trailingDrawdownPct: 0,
    trailingPnlPct: 0,
    drawdownFloorPct: floor,
    warnThresholdPct,
    warnFraction,
    headroomPct: 0 - floor, // floor is negative ⇒ headroom is positive
    needsRefit: false,
    floorBreached: false,
    computedAt: new Date().toISOString(),
  };
}

export async function getDiagnosticSandboxHealth(
  config?: MttmConfig,
  options?: { force?: boolean },
): Promise<DiagnosticSandboxHealth> {
  const cfg = config ?? (await getMttmConfig());
  const window = Math.max(1, Math.floor(cfg.diagnosticSandbox.nNegPnl));
  const inputsKey = [
    cfg.mode,
    cfg.enabled ? "1" : "0",
    cfg.enabledAt ?? "-",
    cfg.diagnosticSandbox.btcVersion ?? "-",
    cfg.diagnosticSandbox.drawdownPct.toString(),
    window.toString(),
  ].join("|");

  const now = Date.now();
  if (
    !options?.force &&
    dsHealthCache !== null &&
    dsHealthCache.expiresAt > now &&
    dsHealthCache.inputsKey === inputsKey
  ) {
    return dsHealthCache.health;
  }

  let health: DiagnosticSandboxHealth;
  if (cfg.mode !== "diagnostic_sandbox" || !cfg.enabled || !cfg.enabledAt) {
    health = emptyDsHealth(cfg);
  } else {
    const enabledAt = new Date(cfg.enabledAt);
    if (Number.isNaN(enabledAt.getTime())) {
      health = emptyDsHealth(cfg);
    } else {
      let rows: Array<{ coinId: string; timeframe: string;
        pnlPercent: number | null; closedAt: Date | null }>;
      try {
        rows = await db
          .select({
            coinId: paperTradesTable.coinId,
            timeframe: paperTradesTable.timeframe,
            pnlPercent: paperTradesTable.pnlPercent,
            closedAt: paperTradesTable.closedAt,
          })
          .from(paperTradesTable)
          .where(
            and(
              eq(paperTradesTable.status, "closed"),
              isNotNull(paperTradesTable.closedAt),
              gt(paperTradesTable.closedAt, enabledAt),
            ),
          );
      } catch (err) {
        logger.warn(
          { err: err instanceof Error ? err.message : String(err) },
          "mttm: diagnostic-sandbox health DB read failed (non-fatal)",
        );
        // Don't poison the cache on a transient DB blip — surface
        // an empty (non-evaluable) health snapshot for this call
        // and let the next poll retry.
        return emptyDsHealth(cfg);
      }

      const dsTrades = rows
        .filter((r) => cfg.universeKeys.has(slotKey(r.coinId, r.timeframe)))
        .sort((a, b) => {
          const ta = a.closedAt ? a.closedAt.getTime() : 0;
          const tb = b.closedAt ? b.closedAt.getTime() : 0;
          return ta - tb;
        });

      // Trailing window: the most recent `window` closed DS trades.
      const trailing = dsTrades.slice(-window);

      const floor = cfg.diagnosticSandbox.drawdownPct;
      const warnFraction = MTTM_DIAGNOSTIC_SANDBOX_HEALTH_WARN_FRACTION;
      const warnThresholdPct = floor * warnFraction;
      const sizingPct = MTTM_DIAGNOSTIC_SANDBOX_FIXED_POSITION_PCT;

      let equity = 1.0;
      let peak = 1.0;
      let trough = 0.0;
      let cumulativePnlPct = 0.0;
      for (const t of trailing) {
        const r = t.pnlPercent ?? 0;
        const accountReturn = r * sizingPct;
        cumulativePnlPct += accountReturn;
        equity *= 1 + accountReturn;
        if (equity > peak) peak = equity;
        const dd = equity / peak - 1;
        if (dd < trough) trough = dd;
      }

      const evaluable = trailing.length > 0;
      const floorBreached = evaluable && trough <= floor;
      // `trough <= warnThresholdPct` (both negative) means the trailing
      // drawdown has eaten past the warn line. Always true when the
      // floor is breached, so this is the union of the two states.
      const needsRefit = evaluable && trough <= warnThresholdPct;

      health = {
        evaluable,
        label: getDiagnosticSandboxLabel(),
        coinId: cfg.diagnosticSandbox.coinId,
        timeframe: cfg.diagnosticSandbox.timeframe,
        btcVersion: cfg.diagnosticSandbox.btcVersion,
        windowTrades: window,
        nTradesObserved: trailing.length,
        trailingDrawdownPct: trough,
        trailingPnlPct: cumulativePnlPct,
        drawdownFloorPct: floor,
        warnThresholdPct,
        warnFraction,
        headroomPct: trough - floor,
        needsRefit,
        floorBreached,
        computedAt: new Date().toISOString(),
      };
    }
  }

  dsHealthCache = {
    health,
    expiresAt: now + DS_HEALTH_CACHE_TTL_MS,
    inputsKey,
  };
  return health;
}

/** Test seam — wipe the DS health cache so the next call re-reads. */
export function invalidateDiagnosticSandboxHealthCache(): void {
  dsHealthCache = null;
}

/**
 * Task #659 — verbatim operator label (server-owned so copy edits
 * ship without redeploying the FE).
 */
export function getDiagnosticSandboxLabel(): string {
  return "BTC/5m diagnostic paper sandbox — probabilities under-confident/untrusted, fixed-size only";
}

/**
 * Task #659 — DS drift evaluator. Reads the RAW persisted state
 * (bypassing the DS hard-pin in `getMttmConfig`) and trips
 * `diagnostic_universe_drift_detected` (universe shape mismatch)
 * or `diagnostic_unauthorized_under_confident_serving` (no real
 * BTC version stamped). Scope-drift is reserved for a registry
 * callback. Idempotent.
 */
export async function evaluateDiagnosticSandboxDrift(
  config?: MttmConfig,
): Promise<MttmDisableReason | null> {
  const cfg = config ?? (await getMttmConfig());
  if (!cfg.enabled) return null;
  if (cfg.mode !== "diagnostic_sandbox") return null;
  if (cfg.disableReason) return cfg.disableReason;

  // Read the RAW universe row (cfg.universe was collapsed by the pin).
  let rawUniverse: MttmSlot[] | null = null;
  try {
    const row = await db
      .select()
      .from(appSettingsTable)
      .where(eq(appSettingsTable.key, MTTM_UNIVERSE_KEY))
      .limit(1);
    if (row.length > 0) {
      rawUniverse = parseUniverse(row[0].value);
    }
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "mttm: drift-evaluator failed to read mttm_universe (non-fatal)",
    );
    return null; // Don't trip on a read failure — that's its own bug class.
  }

  // Universe drift: persisted universe must be absent or exactly the pin.
  if (rawUniverse !== null) {
    const pinnedKey = slotKey(
      MTTM_DIAGNOSTIC_SANDBOX_COIN,
      MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME,
    );
    const rawKeys = new Set(rawUniverse.map((s) => slotKey(s.coinId, s.timeframe)));
    const isExactPin = rawKeys.size === 1 && rawKeys.has(pinnedKey);
    if (!isExactPin) {
      return tripDiagnosticSandboxDrift(
        "diagnostic_universe_drift_detected",
        `mttm_universe persisted shape (${rawUniverse.length} slots: ${[...rawKeys].join(",")}) does not match the DS pin (${pinnedKey}).`,
        cfg,
      );
    }
  }

  // Promotion / under-confident serving: lane enabled without a real BTC version.
  if (cfg.diagnosticSandbox.btcVersion === null) {
    return tripDiagnosticSandboxDrift(
      "diagnostic_unauthorized_under_confident_serving",
      "DS lane is enabled but mttm_diagnostic_sandbox_btc_version is unset; the re-fit + promote_shadow_to_serving flow has not run.",
      cfg,
    );
  }

  // Manifest-scope drift + promotion drift across all active champions.
  // Read once, branch twice. A read failure is non-fatal (return null) so
  // a transient DB hiccup doesn't auto-disable the lane.
  let champions: { coinId: string; timeframe: string; scopeConstraint: unknown; metricsSnapshot: unknown }[] = [];
  try {
    champions = await db
      .select({
        coinId: modelRegistryTable.coinId,
        timeframe: modelRegistryTable.timeframe,
        scopeConstraint: modelRegistryTable.scopeConstraint,
        metricsSnapshot: modelRegistryTable.metricsSnapshot,
      })
      .from(modelRegistryTable)
      .where(
        and(
          eq(modelRegistryTable.state, "champion"),
          eq(modelRegistryTable.isActive, true),
        ),
      );
  } catch (err) {
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "mttm: drift-evaluator failed to read model_registry champions (non-fatal)",
    );
    return null;
  }

  const expectedAllowedUniverse = `${MTTM_DIAGNOSTIC_SANDBOX_COIN}:${MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME}`;

  for (const row of champions) {
    const sc = (row.scopeConstraint ?? null) as
      | { allowed_universe?: unknown; coins?: unknown; timeframes?: unknown; calibration_status?: unknown }
      | null;
    const ms = (row.metricsSnapshot ?? null) as { calibration_status?: unknown } | null;
    const calibStatus =
      (sc && typeof sc.calibration_status === "string" ? sc.calibration_status : null) ??
      (ms && typeof ms.calibration_status === "string" ? ms.calibration_status : null);
    const isBtc5m =
      row.coinId === MTTM_DIAGNOSTIC_SANDBOX_COIN &&
      row.timeframe === MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME;

    if (isBtc5m) {
      // Manifest drift: allowed_universe (preferred) OR (coins,timeframes)
      // must equal exactly the BTC/5m pin. Anything else trips drift.
      let manifestOk = false;
      if (Array.isArray(sc?.allowed_universe)) {
        const au = (sc!.allowed_universe as unknown[]).map((x) => String(x));
        manifestOk = au.length === 1 && au[0] === expectedAllowedUniverse;
      } else if (sc && Array.isArray(sc.coins) && Array.isArray(sc.timeframes)) {
        const coins = (sc.coins as unknown[]).map((x) => String(x));
        const tfs = (sc.timeframes as unknown[]).map((x) => String(x));
        manifestOk =
          coins.length === 1 &&
          coins[0] === MTTM_DIAGNOSTIC_SANDBOX_COIN &&
          tfs.length === 1 &&
          tfs[0] === MTTM_DIAGNOSTIC_SANDBOX_TIMEFRAME;
      }
      if (!manifestOk) {
        return tripDiagnosticSandboxDrift(
          "diagnostic_scope_drift_detected",
          `BTC/5m champion scope_constraint does not match the DS pin (expected allowed_universe=[${expectedAllowedUniverse}], got ${JSON.stringify(sc)}).`,
          cfg,
        );
      }
      continue;
    }

    // Promotion drift: any non-BTC/5m active champion that advertises
    // the under-confident calibration label (which is DS-only by
    // contract) means a DS-grade model is serving a non-DS slot.
    if (calibStatus === "under_confident_documented") {
      return tripDiagnosticSandboxDrift(
        "diagnostic_unauthorized_under_confident_serving",
        `non-DS champion ${row.coinId}/${row.timeframe} advertises calibration_status="under_confident_documented" — under-confident calibrators may serve only the BTC/5m DS lane.`,
        cfg,
      );
    }
  }

  return null;
}

/**
 * Per-MTTM tally evaluator (task-614 step 5).
 *
 * Reads the closed paper trades whose (coin, timeframe) sit in the MTTM
 * universe and were closed AFTER the current MTTM enable timestamp, then
 * applies the two breach rules:
 *   - consecutiveLosses ≥ consecutiveLossCap (default 5)
 *   - nTrades ≥ 10  AND  post-fee PnL% on those trades < n10PostFeeCapPct (default -2%)
 *
 * On breach, flips `mttm_enabled=false` and persists `mttm_disable_reason`
 * so the dashboard banner can announce why the lane shut itself off.
 *
 * Returns the disable reason if a breach fired, else `null`. Idempotent —
 * calling with MTTM already disabled is a no-op.
 *
 * The tally intentionally lives in this module (not in paper-trader.ts)
 * so the report endpoint and any future operator script can re-evaluate
 * without duplicating the SQL.
 */
export async function evaluateMttmAutoDisable(
  config?: MttmConfig,
): Promise<MttmDisableReason | null> {
  const cfg = config ?? (await getMttmConfig());
  if (!cfg.enabled) return null;
  if (cfg.disableReason) return cfg.disableReason; // already tripped
  if (cfg.universeKeys.size === 0) return null;
  if (!cfg.enabledAt) return null; // never enabled — nothing to evaluate

  const enabledAt = new Date(cfg.enabledAt);
  if (Number.isNaN(enabledAt.getTime())) return null;

  // Pull all closed MTTM trades since enable time, ordered by close time.
  const rows = await db
    .select({
      coinId: paperTradesTable.coinId,
      timeframe: paperTradesTable.timeframe,
      pnl: paperTradesTable.pnl,
      pnlPercent: paperTradesTable.pnlPercent,
      positionSize: paperTradesTable.positionSize,
      closedAt: paperTradesTable.closedAt,
    })
    .from(paperTradesTable)
    .where(
      and(
        eq(paperTradesTable.status, "closed"),
        isNotNull(paperTradesTable.closedAt),
        gt(paperTradesTable.closedAt, enabledAt),
      ),
    );

  // Filter to MTTM-universe trades only, then sort chronologically by
  // closedAt to compute the `consecutive_losses` tail accurately.
  const mttmTrades = rows
    .filter((r) => cfg.universeKeys.has(slotKey(r.coinId, r.timeframe)))
    .sort((a, b) => {
      const ta = a.closedAt ? a.closedAt.getTime() : 0;
      const tb = b.closedAt ? b.closedAt.getTime() : 0;
      return ta - tb;
    });

  if (mttmTrades.length === 0) return null;

  // Consecutive-loss tail (counts from the end of the chronological list).
  let consecutive = 0;
  for (let i = mttmTrades.length - 1; i >= 0; i--) {
    const pnl = mttmTrades[i].pnl ?? 0;
    if (pnl <= 0) consecutive++;
    else break;
  }

  if (consecutive >= cfg.consecutiveLossCap) {
    const reason: MttmDisableReason = {
      reason: "consecutive_losses",
      detail:
        `MTTM auto-disabled — ${consecutive} consecutive MTTM losses ` +
        `(cap = ${cfg.consecutiveLossCap}).`,
      trippedAt: new Date().toISOString(),
      consecutiveLosses: consecutive,
      nTrades: mttmTrades.length,
    };
    await setMttmDisableReason(reason);
    await setMttmEnabled(false);
    logger.warn(
      { consecutive, nTrades: mttmTrades.length },
      "mttm: auto-disabled (consecutive losses)",
    );
    // Task #619 — fire off-dashboard alert on the rising edge of an
    // auto-disable. Fire-and-forget so a slow webhook never blocks
    // the trade-close sweep; the notifier dedups on `trippedAt` so a
    // restart cannot re-page on the same trip.
    void notifyMttmAutoDisabled(reason).catch((err) =>
      logger.warn(
        { err: err instanceof Error ? err.message : String(err) },
        "mttm: auto-disable notifier failed (non-fatal)",
      ),
    );
    return reason;
  }

  // n≥10 + post-fee PnL% rule. PnL% on the paper_trades row is already
  // post-fee (entry/exit fee deducted in paper-trader.ts close path), so
  // we sum the position-weighted return.
  if (mttmTrades.length >= 10) {
    let totalPnl = 0;
    let totalNotional = 0;
    for (const t of mttmTrades) {
      totalPnl += t.pnl ?? 0;
      totalNotional += t.positionSize;
    }
    const postFeePnlPct = totalNotional > 0 ? totalPnl / totalNotional : 0;
    if (postFeePnlPct < cfg.n10PostFeeCapPct) {
      const reason: MttmDisableReason = {
        reason: "n10_post_fee",
        detail:
          `MTTM auto-disabled — ${mttmTrades.length} trades, post-fee PnL ` +
          `${(postFeePnlPct * 100).toFixed(2)}% < cap ` +
          `${(cfg.n10PostFeeCapPct * 100).toFixed(2)}%.`,
        trippedAt: new Date().toISOString(),
        consecutiveLosses: consecutive,
        nTrades: mttmTrades.length,
        postFeePnlPct,
      };
      await setMttmDisableReason(reason);
      await setMttmEnabled(false);
      logger.warn(
        { nTrades: mttmTrades.length, postFeePnlPct },
        "mttm: auto-disabled (n10 post-fee floor)",
      );
      // Task #619 — fire off-dashboard alert on the rising edge.
      // See the consecutive-loss branch above for rationale.
      void notifyMttmAutoDisabled(reason).catch((err) =>
        logger.warn(
          { err: err instanceof Error ? err.message : String(err) },
          "mttm: auto-disable notifier failed (non-fatal)",
        ),
      );
      return reason;
    }
  }

  return null;
}

/** Test seam — wipe the in-process cache so the next call re-reads. */
export function invalidateMttmCache(): void {
  cache = null;
}

/** Test seam — populate the cache deterministically (no DB hit). */
export function __setMttmCache(config: MttmConfig): void {
  cache = { config, expiresAt: Date.now() + CACHE_TTL_MS };
}
