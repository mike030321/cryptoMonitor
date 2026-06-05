#!/usr/bin/env node
/**
 * MTTM universe audit (task #614 step 1) — CLI wrapper.
 *
 * Thin formatter around `runMttmAudit()` from `../lib/mttm-audit`.
 * Prints the audit table and exits non-zero on any miss. The library
 * deliberately holds all logic so that `routes/crypto/index.ts` can
 * gate the `/mttm/state/toggle` "enable" path on the same audit
 * without re-running this CLI as a subprocess and without importing
 * a module that has top-level side effects (the bundled api-server
 * inlines every imported file).
 *
 * Usage:
 *   pnpm --filter @workspace/api-server exec tsx src/scripts/mttm-audit.ts
 */
import { runMttmAudit, type AuditRow } from "../lib/mttm-audit";

function pad(s: string, n: number): string {
  if (s.length >= n) return s;
  return s + " ".repeat(n - s.length);
}

function printRow(r: AuditRow): void {
  const verdict = r.ok ? "OK" : `FAIL — ${r.problems.join("; ")}`;
  console.log(
    [
      pad(r.coinId, 26),
      pad(r.timeframe, 4),
      pad(r.latestVersion ?? "—", 18),
      pad(r.servedPredictorKind ?? "—", 10),
      pad(r.promoted === null ? "—" : r.promoted ? "yes" : "no", 9),
      verdict,
    ].join("  "),
  );
}

async function main(): Promise<void> {
  const result = await runMttmAudit();

  console.log(
    `MTTM universe audit — ${result.rows.length} slots (source: ${result.source})`,
  );
  console.log(`registry: ${result.modelsRoot}`);
  console.log("");

  if (result.rows.length === 0) {
    console.error(result.failingSlots[0]?.problems.join("; ") ?? "no rows");
    process.exit(2);
  }

  const header = [
    pad("coin", 26),
    pad("tf", 4),
    pad("latest", 18),
    pad("kind", 10),
    pad("promoted", 9),
    "verdict",
  ].join("  ");
  console.log(header);
  console.log("-".repeat(header.length + 8));
  for (const r of result.rows) printRow(r);
  console.log("");
  if (result.ok) {
    console.log(`mttm-audit: PASS — ${result.rows.length}/${result.rows.length} slots green`);
    process.exit(0);
  } else {
    console.log(
      `mttm-audit: FAIL — ${result.failingSlots.length}/${result.rows.length} slot(s) ineligible. ` +
      `MTTM cannot enable until every row is OK.`,
    );
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("mttm-audit: unexpected error", err);
  process.exit(2);
});
