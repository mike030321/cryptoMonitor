/**
 * Phase 5 — fleet-level portfolio constraint gate.
 *
 * Mirrors `app/decision_engine/engine.py:_check_portfolio` so the live trader
 * and the Python backtester apply identical sector / correlated-exposure /
 * beta / regime-budget caps. All thresholds live in
 * `shared/trading-frictions.json` under `portfolio_constraints` — change
 * the JSON, both paths update.
 */
import frictions from "../../../../shared/trading-frictions.json" with { type: "json" };

type Json = typeof frictions;

interface PortfolioConfig {
  enabled: boolean;
  max_sector_exposure_pct: number;
  max_correlated_exposure_pct: number;
  max_beta_to_btc: number;
  regime_budget_pct: number;
  sector_map: Record<string, string>;
}

// Task #349 — fail-fast loader. The previous version read `portfolio_constraints`
// via an unchecked cast, so a missing block (typo, deleted key, wrong file
// path) silently degraded to `PC.enabled === undefined` → kill-switch off
// while every numeric cap below evaluated to `undefined > x === false` →
// every fleet gate silently passed. Now we require the block and every
// field it owns, and we require `sector_map._default` so an unmapped coin
// falls into a JSON-defined sector instead of a TS-side literal "other".
export function _requireConfig<T>(value: T | undefined | null, key: string): T {
  if (value === undefined || value === null) {
    throw new Error(
      `[portfolio-constraints] required key '${key}' missing from ` +
      `shared/trading-frictions.json. Refusing to start with a silent ` +
      `default — see task #349.`,
    );
  }
  return value;
}

// Task #356 — exposed for unit tests so the missing-key throws can be
// exercised against mutated copies of the contract without spawning a
// subprocess to reload the module. Mirrors EXACTLY the inline
// `_requireConfig` calls in the module-level wiring below.
export function assertPortfolioConfigRequiredKeys(raw: unknown): void {
  const r =
    raw && typeof raw === "object"
      ? (raw as Record<string, unknown>)
      : {};
  const pc = _requireConfig(
    r.portfolio_constraints as Record<string, unknown> | undefined,
    "portfolio_constraints",
  );
  _requireConfig(pc.enabled as boolean | undefined, "portfolio_constraints.enabled");
  _requireConfig(
    pc.max_sector_exposure_pct as number | undefined,
    "portfolio_constraints.max_sector_exposure_pct",
  );
  _requireConfig(
    pc.max_correlated_exposure_pct as number | undefined,
    "portfolio_constraints.max_correlated_exposure_pct",
  );
  _requireConfig(
    pc.max_beta_to_btc as number | undefined,
    "portfolio_constraints.max_beta_to_btc",
  );
  _requireConfig(
    pc.regime_budget_pct as number | undefined,
    "portfolio_constraints.regime_budget_pct",
  );
  const sm = _requireConfig(
    pc.sector_map as Record<string, string> | undefined,
    "portfolio_constraints.sector_map",
  );
  _requireConfig(sm._default, "portfolio_constraints.sector_map._default");
}
const _PC_RAW = _requireConfig(
  (frictions as unknown as Record<string, unknown>).portfolio_constraints as
    | PortfolioConfig
    | undefined,
  "portfolio_constraints",
);
const PC: PortfolioConfig = {
  enabled: _requireConfig(_PC_RAW.enabled, "portfolio_constraints.enabled"),
  max_sector_exposure_pct: _requireConfig(
    _PC_RAW.max_sector_exposure_pct,
    "portfolio_constraints.max_sector_exposure_pct",
  ),
  max_correlated_exposure_pct: _requireConfig(
    _PC_RAW.max_correlated_exposure_pct,
    "portfolio_constraints.max_correlated_exposure_pct",
  ),
  max_beta_to_btc: _requireConfig(
    _PC_RAW.max_beta_to_btc,
    "portfolio_constraints.max_beta_to_btc",
  ),
  regime_budget_pct: _requireConfig(
    _PC_RAW.regime_budget_pct,
    "portfolio_constraints.regime_budget_pct",
  ),
  sector_map: _requireConfig(
    _PC_RAW.sector_map,
    "portfolio_constraints.sector_map",
  ),
};
_requireConfig(PC.sector_map._default, "portfolio_constraints.sector_map._default");

export interface PortfolioOpenPosition {
  coinId: string;
  direction: "up" | "down" | "long" | "short";
  notionalUsd: number;
  regimeAtEntry?: string | null;
  betaToBtc?: number | null;
}

export interface PortfolioCheckResult {
  ok: boolean;
  skipReason?:
    | "portfolio_sector_cap"
    | "portfolio_correlated_exposure"
    | "portfolio_beta_cap"
    | "portfolio_regime_budget";
  detail?: string;
  breakdown: Record<string, number | string | boolean | null>;
}

export function getSectorForCoin(coinId: string): string {
  // PC.sector_map._default is required at module load (see _requireConfig
  // above), so the cascade always resolves to a JSON-defined sector.
  return PC.sector_map[coinId] ?? PC.sector_map._default;
}

export function checkPortfolioConstraints(args: {
  coinId: string;
  newNotionalUsd: number;
  equityUsd: number;
  regime?: string | null;
  openPositions: PortfolioOpenPosition[];
  // Optional per-call overrides — mirror of the Python
  // DecisionRequest.portfolio_constraints_override field. Lets the
  // parity test exercise branches that the live config cannot reach
  // (e.g. correlated-exposure when sector_cap < correlated_cap), and
  // gives the live tuner a knob to tighten/loosen fleet caps without
  // redeploying. Only the four numeric thresholds are overridable; the
  // sector_map and the enabled flag stay sourced from the JSON.
  overrides?: {
    max_sector_exposure_pct?: number;
    max_correlated_exposure_pct?: number;
    max_beta_to_btc?: number;
    regime_budget_pct?: number;
  };
}): PortfolioCheckResult {
  if (!PC.enabled) {
    return { ok: true, breakdown: { enabled: false } };
  }
  const ov = args.overrides ?? {};
  const cfg = {
    max_sector_exposure_pct: ov.max_sector_exposure_pct ?? PC.max_sector_exposure_pct,
    max_correlated_exposure_pct: ov.max_correlated_exposure_pct ?? PC.max_correlated_exposure_pct,
    max_beta_to_btc: ov.max_beta_to_btc ?? PC.max_beta_to_btc,
    regime_budget_pct: ov.regime_budget_pct ?? PC.regime_budget_pct,
  };
  const equity = Math.max(args.equityUsd, 1e-9);
  // sector_map._default is required at module load — see _requireConfig.
  const sectorOf = (cid: string) =>
    PC.sector_map[cid] ?? PC.sector_map._default;
  const newSector = sectorOf(args.coinId);
  const breakdown: Record<string, number | string | boolean | null> = {
    enabled: true,
    new_sector: newSector,
    regime: args.regime ?? null,
    open_notional_usd: args.openPositions.reduce((s, p) => s + p.notionalUsd, 0),
    new_notional_usd: args.newNotionalUsd,
  };

  // Sector cap
  const bySector = new Map<string, number>();
  for (const p of args.openPositions) {
    const s = sectorOf(p.coinId);
    bySector.set(s, (bySector.get(s) ?? 0) + p.notionalUsd);
  }
  const sectorAfter = (bySector.get(newSector) ?? 0) + args.newNotionalUsd;
  const sectorShare = sectorAfter / equity;
  breakdown.sector_share_after = sectorShare;
  breakdown.sector_cap = cfg.max_sector_exposure_pct;
  if (sectorShare > cfg.max_sector_exposure_pct) {
    return {
      ok: false,
      skipReason: "portfolio_sector_cap",
      detail: `${newSector} share=${(sectorShare * 100).toFixed(1)}% > cap=${(cfg.max_sector_exposure_pct * 100).toFixed(1)}%`,
      breakdown,
    };
  }

  // Correlated exposure
  const coinsInSector = new Set(
    args.openPositions
      .filter((p) => sectorOf(p.coinId) === newSector)
      .map((p) => p.coinId),
  );
  coinsInSector.add(args.coinId);
  if (coinsInSector.size >= 2) {
    breakdown.correlated_cap = cfg.max_correlated_exposure_pct;
    if (sectorShare > cfg.max_correlated_exposure_pct) {
      return {
        ok: false,
        skipReason: "portfolio_correlated_exposure",
        detail: `${coinsInSector.size} coins in ${newSector}, total share=${(sectorShare * 100).toFixed(1)}% > cap=${(cfg.max_correlated_exposure_pct * 100).toFixed(1)}%`,
        breakdown,
      };
    }
  }

  // Beta to BTC (notional-weighted; default beta=1.0 for unknown)
  const proposedTotal =
    args.openPositions.reduce((s, p) => s + p.notionalUsd, 0) +
    args.newNotionalUsd;
  if (proposedTotal > 0) {
    let betaSum = 0;
    for (const p of args.openPositions) {
      const b = p.betaToBtc ?? 1.0;
      betaSum += b * p.notionalUsd;
    }
    betaSum += 1.0 * args.newNotionalUsd;
    const bookBeta = betaSum / proposedTotal;
    breakdown.book_beta = bookBeta;
    breakdown.beta_cap = cfg.max_beta_to_btc;
    if (bookBeta > cfg.max_beta_to_btc) {
      return {
        ok: false,
        skipReason: "portfolio_beta_cap",
        detail: `book β=${bookBeta.toFixed(2)} > cap=${cfg.max_beta_to_btc.toFixed(2)}`,
        breakdown,
      };
    }
  }

  // Regime budget
  if (args.regime) {
    const byRegime = new Map<string, number>();
    for (const p of args.openPositions) {
      const r = p.regimeAtEntry ?? "unknown";
      byRegime.set(r, (byRegime.get(r) ?? 0) + p.notionalUsd);
    }
    const regimeAfter = (byRegime.get(args.regime) ?? 0) + args.newNotionalUsd;
    const regimeShare = regimeAfter / equity;
    breakdown.regime_share_after = regimeShare;
    breakdown.regime_budget = cfg.regime_budget_pct;
    if (regimeShare > cfg.regime_budget_pct) {
      return {
        ok: false,
        skipReason: "portfolio_regime_budget",
        detail: `regime ${args.regime} share=${(regimeShare * 100).toFixed(1)}% > cap=${(cfg.regime_budget_pct * 100).toFixed(1)}%`,
        breakdown,
      };
    }
  }

  return { ok: true, breakdown };
}

export const PORTFOLIO_CONSTRAINTS_CONFIG = PC;
