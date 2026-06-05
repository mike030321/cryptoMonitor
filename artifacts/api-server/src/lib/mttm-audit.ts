/**
 * MTTM universe audit — pure library form.
 *
 * Walks `artifacts/ml-engine/models/{coin}/{tf}/latest` for every slot
 * in the configured (or supplied) MTTM universe and asserts:
 *   - the `latest` pointer resolves to a model directory that exists
 *   - the manifest reports `served_predictor_kind = "lightgbm"`
 *   - the verification.json reports `promoted = true`
 *   - the latest pointer matches the version pinned in `mttm_universe`
 *     (drift between the registry and the configured universe means
 *     the operator must explicitly re-pin before MTTM may enable)
 *
 * This module is import-safe — it does NO work at import time. The CLI
 * wrapper at `src/scripts/mttm-audit.ts` calls `runMttmAudit()` and
 * formats the result; the `/mttm/state/toggle` route imports
 * `runMttmAudit` directly to gate "enable" on a green audit. Splitting
 * library from CLI here is critical because the api-server bundle
 * (esbuild) inlines every imported file — if the CLI's `main()` lived
 * in this module, the bundled server would auto-execute the audit on
 * boot and `process.exit(1)` if any slot was ineligible.
 */
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { DEFAULT_MTTM_UNIVERSE, getMttmConfig, type MttmSlot } from "./mttm";

export interface AuditRow {
  coinId: string;
  timeframe: string;
  expectedVersion: string;
  latestVersion: string | null;
  servedPredictorKind: string | null;
  promoted: boolean | null;
  reason: string | null;
  ok: boolean;
  problems: string[];
}

export interface AuditResult {
  ok: boolean;
  modelsRoot: string;
  source: "app_settings" | "default" | "explicit";
  rows: AuditRow[];
  failingSlots: AuditRow[];
}

export function findRepoRoot(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  let cur = here;
  for (let i = 0; i < 10; i++) {
    if (existsSync(path.join(cur, "pnpm-workspace.yaml"))) return cur;
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  throw new Error(`mttm-audit: could not locate workspace root (started ${here})`);
}

export function findModelsRoot(): string {
  return path.join(findRepoRoot(), "artifacts", "ml-engine", "models");
}

function readJson<T = unknown>(p: string): T | null {
  try {
    return JSON.parse(readFileSync(p, "utf8")) as T;
  } catch {
    return null;
  }
}

function readLatest(coinDir: string, tf: string): string | null {
  const latestPath = path.join(coinDir, tf, "latest");
  try {
    const raw = readFileSync(latestPath, "utf8").trim();
    return raw || null;
  } catch {
    return null;
  }
}

export function auditSlot(modelsRoot: string, slot: MttmSlot): AuditRow {
  const row: AuditRow = {
    coinId: slot.coinId,
    timeframe: slot.timeframe,
    expectedVersion: slot.version,
    latestVersion: null,
    servedPredictorKind: null,
    promoted: null,
    reason: null,
    ok: false,
    problems: [],
  };
  const coinDir = path.join(modelsRoot, slot.coinId);
  const latest = readLatest(coinDir, slot.timeframe);
  if (!latest) {
    row.problems.push(`no latest pointer at ${slot.coinId}/${slot.timeframe}`);
    return row;
  }
  row.latestVersion = latest;
  const sliceDir = path.join(coinDir, slot.timeframe, latest);
  if (!existsSync(sliceDir)) {
    row.problems.push(`latest version ${latest} dir missing`);
    return row;
  }
  const manifest = readJson<{ served_predictor_kind?: string; model_kind?: string }>(
    path.join(sliceDir, "manifest.json"),
  );
  if (!manifest) {
    row.problems.push("manifest.json unreadable");
    return row;
  }
  row.servedPredictorKind = manifest.served_predictor_kind ?? manifest.model_kind ?? null;
  if (row.servedPredictorKind !== "lightgbm") {
    row.problems.push(`served_predictor_kind=${row.servedPredictorKind ?? "null"} (expected lightgbm)`);
  }
  const verification = readJson<{ promoted?: boolean; reason?: string }>(
    path.join(sliceDir, "verification.json"),
  );
  if (!verification) {
    row.problems.push("verification.json unreadable");
  } else {
    row.promoted = verification.promoted ?? null;
    row.reason = verification.reason ?? null;
    if (verification.promoted !== true) {
      row.problems.push(`not promoted (reason=${verification.reason ?? "n/a"})`);
    }
  }
  if (latest !== slot.version) {
    row.problems.push(
      `latest ${latest} ≠ pinned ${slot.version} (re-pin universe before enabling)`,
    );
  }
  row.ok = row.problems.length === 0;
  return row;
}

/**
 * Programmatic entry point. Pass `slots` to audit a specific universe;
 * omit to load the currently-pinned `mttm_universe` from app_settings,
 * falling back to the compiled default if the DB cannot be reached so
 * the audit is still meaningful in offline contexts (e.g. CI).
 */
export async function runMttmAudit(opts?: {
  slots?: MttmSlot[];
}): Promise<AuditResult> {
  const modelsRoot = findModelsRoot();
  if (!existsSync(modelsRoot)) {
    return {
      ok: false,
      modelsRoot,
      source: "default",
      rows: [],
      failingSlots: [
        {
          coinId: "*",
          timeframe: "*",
          expectedVersion: "*",
          latestVersion: null,
          servedPredictorKind: null,
          promoted: null,
          reason: null,
          ok: false,
          problems: [`model registry not found at ${modelsRoot}`],
        },
      ],
    };
  }

  let universe: MttmSlot[];
  let source: AuditResult["source"];
  if (opts?.slots) {
    universe = opts.slots;
    source = "explicit";
  } else {
    universe = DEFAULT_MTTM_UNIVERSE;
    source = "default";
    try {
      const cfg = await getMttmConfig();
      if (cfg.universe.length > 0) {
        universe = cfg.universe;
        source = "app_settings";
      }
    } catch {
      // Keep the compiled default; the audit is still meaningful.
    }
  }

  const rows: AuditRow[] = universe.map((slot) => auditSlot(modelsRoot, slot));
  const failingSlots = rows.filter((r) => !r.ok);
  return { ok: failingSlots.length === 0, modelsRoot, source, rows, failingSlots };
}
