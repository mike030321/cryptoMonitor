/**
 * Task #680 — pagination tests for /crypto/brain/runtime-status.
 *
 * The route used to issue:
 *
 *   db.select(...).from(predictionJournalTable)
 *     .where(gte(predictionJournalTable.createdAt, since))
 *
 * with `since = now − 30 min` and **no `.limit(...)`**. Under any
 * prediction-journal surge (busy meta-brain epoch, backfill,
 * multi-coin update burst) this pulled an unbounded row count
 * into Node memory. The fix paginates the SELECT internally with
 * a bounded page size and streams the aggregation; the
 * `BrainRuntimeStatePayload` shape is byte-identical for the same
 * input rows (counters add, max-of-monotonic-Date is still max).
 *
 * The two invariants this test pins:
 *
 * 1. Correctness: with ≥ 50 000 synthetic rows the paginated
 *    aggregation produces exactly the same `recentAbstainReasons`
 *    rollup, `recentNonAbstainCount`, `lastSuccessfulAt`, and
 *    `state` discriminant as the unpaginated reference would on
 *    the same input.
 *
 * 2. No N+1: the route does not call `fetchJournalPage` more than
 *    `ceil(total / pageSize)` times. The pager fetches `pageSize + 1`
 *    rows per round-trip and uses the extra row as a "has more"
 *    sentinel, so even when `total` is an exact multiple of
 *    `pageSize` no trailing empty-page probe is needed.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  BRAIN_RUNTIME_PAGE_SIZE,
  computeBrainRuntimeState,
  type BrainRuntimeJournalRow,
  type BrainRuntimeStateDataSource,
} from "../src/lib/brain-runtime-state";

interface SyntheticRow extends BrainRuntimeJournalRow {}

interface DataSourceCallLog {
  pages: number;
  rowsServed: number;
  fetchLastSuccessfulAtCalls: number;
  getBrainStateCalls: number;
}

function buildStubDataSource(
  rows: SyntheticRow[],
  brain: { enabled: boolean; source: string },
): { ds: BrainRuntimeStateDataSource; log: DataSourceCallLog } {
  // Rows must be pre-sorted by (createdAt, id) ascending — the production
  // SELECT does this via `.orderBy(createdAt, id)` and the cursor relies
  // on lexicographic row comparison, so the stub mirrors that contract.
  const sorted = [...rows].sort((a, b) => {
    const ta = a.createdAt ? a.createdAt.getTime() : 0;
    const tb = b.createdAt ? b.createdAt.getTime() : 0;
    if (ta !== tb) return ta - tb;
    return a.id - b.id;
  });
  const log: DataSourceCallLog = {
    pages: 0,
    rowsServed: 0,
    fetchLastSuccessfulAtCalls: 0,
    getBrainStateCalls: 0,
  };
  const ds: BrainRuntimeStateDataSource = {
    async fetchJournalPage({ since, afterCursor, limit }) {
      log.pages += 1;
      const sinceMs = since.getTime();
      const filtered = sorted.filter((r) => {
        if (!r.createdAt) return false;
        if (r.createdAt.getTime() < sinceMs) return false;
        if (!afterCursor) return true;
        const rt = r.createdAt.getTime();
        const ct = afterCursor.createdAt.getTime();
        if (rt !== ct) return rt > ct;
        return r.id > afterCursor.id;
      });
      const page = filtered.slice(0, limit);
      log.rowsServed += page.length;
      return page;
    },
    async fetchLastSuccessfulAt() {
      log.fetchLastSuccessfulAtCalls += 1;
      let best: Date | null = null;
      for (const r of sorted) {
        if (r.becameTrade && r.createdAt) {
          if (!best || r.createdAt > best) best = r.createdAt;
        }
      }
      return best;
    },
    async getBrainState() {
      log.getBrainStateCalls += 1;
      return brain;
    },
  };
  return { ds, log };
}

/**
 * Reference, unpaginated implementation — identical body to the
 * pre-Task #680 route. We compute its output and compare to the
 * paginated implementation to assert byte-identical rollups.
 */
const NO_MODEL_REASONS = new Set([
  "no_model",
  "model_load_failed",
  "model_unavailable",
  "missing_model",
]);
function referenceCompute(
  rows: SyntheticRow[],
  since: Date,
  brain: { enabled: boolean; source: string },
): {
  recentAbstainReasons: Record<string, number>;
  recentNonAbstainCount: number;
  lastSuccessfulAt: Date | null;
  state: "online" | "offline_no_model" | "offline_disabled";
} {
  const sinceMs = since.getTime();
  const recent = rows.filter(
    (r) => r.createdAt && r.createdAt.getTime() >= sinceMs,
  );
  const recentAbstainReasons: Record<string, number> = {};
  let recentNonAbstain = 0;
  let lastSuccessfulAt: Date | null = null;
  for (const r of recent) {
    const reason = (r.skipReason ?? "").startsWith("quant_abstain_")
      ? r.skipReason!.replace(/^quant_abstain_/, "") || "no_model"
      : null;
    if (reason) {
      recentAbstainReasons[reason] = (recentAbstainReasons[reason] ?? 0) + 1;
    } else if (
      r.becameTrade ||
      (r.skipReason && !r.skipReason.startsWith("quant_abstain_"))
    ) {
      recentNonAbstain += 1;
      if (
        !lastSuccessfulAt ||
        (r.createdAt && r.createdAt > lastSuccessfulAt)
      ) {
        lastSuccessfulAt = r.createdAt ?? null;
      }
    }
  }
  if (!lastSuccessfulAt) {
    let best: Date | null = null;
    for (const r of rows) {
      if (r.becameTrade && r.createdAt) {
        if (!best || r.createdAt > best) best = r.createdAt;
      }
    }
    lastSuccessfulAt = best;
  }
  const noModelAbstains = Object.entries(recentAbstainReasons)
    .filter(([reason]) => NO_MODEL_REASONS.has(reason))
    .reduce((sum, [, count]) => sum + count, 0);
  const totalAbstains = Object.values(recentAbstainReasons).reduce(
    (sum, count) => sum + count,
    0,
  );
  let state: "online" | "offline_no_model" | "offline_disabled";
  if (!brain.enabled) {
    state = "offline_disabled";
  } else if (
    recentNonAbstain === 0 &&
    totalAbstains > 0 &&
    noModelAbstains > 0 &&
    noModelAbstains >= totalAbstains / 2
  ) {
    state = "offline_no_model";
  } else {
    state = "online";
  }
  return {
    recentAbstainReasons,
    recentNonAbstainCount: recentNonAbstain,
    lastSuccessfulAt,
    state,
  };
}

function makeSyntheticRows(total: number, now: number): SyntheticRow[] {
  // Spread rows across the inner 28 minutes of the 30-minute window so
  // a millisecond-scale clock drift between the test capturing `now`
  // and `computeBrainRuntimeState` recomputing `Date.now()` cannot
  // shift any row across the `since` boundary. Multiple rows share a
  // `createdAt` to stress the (createdAt, id) lexicographic cursor:
  //   - quant_abstain_no_model
  //   - quant_abstain_low_confidence
  //   - quant_abstain_<empty>  (becomes "no_model" via the empty-suffix branch)
  //   - non-abstain skip_reason ("post_cost_negative", etc.)
  //   - becameTrade=true rows (advance lastSuccessfulAt)
  const innerStartMs = now - 29 * 60 * 1000; // 1 min inside the window
  const innerEndMs = now - 60 * 1000; // 1 min before now
  const rows: SyntheticRow[] = [];
  for (let i = 0; i < total; i++) {
    // Same createdAt for batches of 7 rows -> exercises (createdAt,id)
    // tie-breaking in the cursor.
    const bucket = Math.floor(i / 7);
    const totalBuckets = Math.ceil(total / 7);
    const createdAt = new Date(
      innerStartMs +
        (bucket * (innerEndMs - innerStartMs)) / Math.max(1, totalBuckets - 1),
    );
    const mod = i % 11;
    let skipReason: string | null = null;
    let becameTrade: boolean | null = null;
    if (mod === 0) {
      skipReason = "quant_abstain_no_model";
    } else if (mod === 1 || mod === 2) {
      skipReason = "quant_abstain_low_confidence";
    } else if (mod === 3) {
      skipReason = "quant_abstain_"; // empty suffix -> "no_model"
    } else if (mod === 4) {
      skipReason = "quant_abstain_model_load_failed";
    } else if (mod === 5) {
      skipReason = "quant_abstain_regime_filter";
    } else if (mod === 6) {
      skipReason = "post_cost_negative";
      becameTrade = false;
    } else if (mod === 7) {
      skipReason = "fee_gate_ev";
      becameTrade = false;
    } else if (mod === 8) {
      becameTrade = true;
    } else {
      // mod 9, 10 -> undecided (becameTrade null, skipReason null)
      skipReason = null;
      becameTrade = null;
    }
    rows.push({ id: i + 1, createdAt, skipReason, becameTrade });
  }
  // Add a handful of pre-window rows that MUST be excluded by the
  // `gte(createdAt, since)` filter — proves the cursor scan honors the
  // since-bound on every page.
  for (let i = 0; i < 50; i++) {
    rows.push({
      id: total + i + 1,
      createdAt: new Date(now - 30 * 60 * 1000 - 60_000 - i * 1000),
      skipReason: "quant_abstain_no_model",
      becameTrade: null,
    });
  }
  return rows;
}

describe("Task #680 — paginated /crypto/brain/runtime-status journal scan", () => {
  it("with ≥ 50 000 synthetic rows produces the same rollup as the unpaginated reference", async () => {
    const now = Date.now();
    const TOTAL = 50_000;
    const rows = makeSyntheticRows(TOTAL, now);
    const brain = { enabled: true, source: "manual" };

    const { ds } = buildStubDataSource(rows, brain);
    const paginated = await computeBrainRuntimeState(ds);

    const since = new Date(now - 30 * 60 * 1000);
    const reference = referenceCompute(rows, since, brain);

    assert.deepEqual(
      paginated.recentAbstainReasons,
      reference.recentAbstainReasons,
      "recentAbstainReasons rollup must be byte-identical to the unpaginated reference",
    );
    assert.equal(paginated.recentNonAbstainCount, reference.recentNonAbstainCount);
    assert.equal(
      paginated.lastSuccessfulAt?.getTime() ?? null,
      reference.lastSuccessfulAt?.getTime() ?? null,
    );
    assert.equal(paginated.state, reference.state);
    assert.equal(paginated.brainEnabled, true);
    assert.equal(paginated.brainSource, "manual");
  });

  it("does not call fetchJournalPage more than ceil(total / pageSize) times (no N+1)", async () => {
    const now = Date.now();
    // Use an exact multiple of pageSize so the +1-sentinel pager has
    // to prove it does NOT issue a trailing empty-page probe — this
    // is the failure mode the bound `ceil(total/pageSize)` rules out.
    const TOTAL = 50_000;
    const rows = makeSyntheticRows(TOTAL, now);
    const brain = { enabled: true, source: "manual" };

    const pageSize = 5_000;
    const { ds, log } = buildStubDataSource(rows, brain);
    await computeBrainRuntimeState(ds, pageSize);

    // The 30-min window contains all TOTAL rows; the pre-window rows
    // are filtered by `gte(createdAt, since)` and never enter the
    // aggregation. The pager fetches `pageSize + 1` rows per round-
    // trip and uses the extra row as a "has more" sentinel, so the
    // call count is capped at exactly ceil(TOTAL/pageSize) — no
    // trailing empty-page probe even when TOTAL is an exact multiple
    // of pageSize.
    const expectedMax = Math.ceil(TOTAL / pageSize);
    assert.ok(
      log.pages <= expectedMax,
      `fetchJournalPage was invoked ${log.pages} times, expected ≤ ${expectedMax} (no N+1)`,
    );
    // Sanity: the loop did make multiple page fetches (i.e. it's
    // actually paginating, not silently dumping the whole table in
    // one go).
    assert.ok(log.pages >= 2, "must have paginated across multiple pages");
    // The +1-sentinel re-reads the boundary row on every non-final
    // page, so served rows = TOTAL + (pages - 1). This is the
    // upper bound on data-source work done by the route — distinct
    // from the row-count served to the aggregator (which is exactly
    // TOTAL, asserted by the byte-identical correctness test above).
    assert.equal(
      log.rowsServed,
      TOTAL + (log.pages - 1),
      "served-row count must match the +1-sentinel pager invariant",
    );
  });

  it("emits state='offline_disabled' regardless of journal contents when brain is disabled", async () => {
    const now = Date.now();
    const rows = makeSyntheticRows(2_000, now);
    const brain = { enabled: false, source: "default" };
    const { ds } = buildStubDataSource(rows, brain);
    const result = await computeBrainRuntimeState(ds);
    assert.equal(result.state, "offline_disabled");
    assert.equal(result.brainEnabled, false);
    assert.equal(result.brainSource, "default");
  });

  it("emits state='offline_no_model' when ≥half of recent abstains are no-model and there are zero non-abstain rows", async () => {
    const now = Date.now();
    const within = (offsetMin: number) =>
      new Date(now - offsetMin * 60 * 1000);
    // 6 no_model abstains, 4 low_confidence abstains, no non-abstain rows.
    // 6 / 10 = 60% >= 50% -> offline_no_model.
    const rows: SyntheticRow[] = [];
    for (let i = 0; i < 6; i++) {
      rows.push({
        id: i + 1,
        createdAt: within(5),
        skipReason: "quant_abstain_no_model",
        becameTrade: null,
      });
    }
    for (let i = 0; i < 4; i++) {
      rows.push({
        id: i + 100,
        createdAt: within(5),
        skipReason: "quant_abstain_low_confidence",
        becameTrade: null,
      });
    }
    const { ds } = buildStubDataSource(rows, { enabled: true, source: "manual" });
    const result = await computeBrainRuntimeState(ds);
    assert.equal(result.state, "offline_no_model");
    assert.equal(result.recentAbstainReasons["no_model"], 6);
    assert.equal(result.recentAbstainReasons["low_confidence"], 4);
  });

  it("uses the fallback fetchLastSuccessfulAt only when the in-window scan finds no successful row", async () => {
    const now = Date.now();
    const within = (offsetMin: number) =>
      new Date(now - offsetMin * 60 * 1000);

    // Scenario A: in-window has becameTrade=true -> fallback NOT called.
    {
      const rows: SyntheticRow[] = [
        {
          id: 1,
          createdAt: within(5),
          skipReason: null,
          becameTrade: true,
        },
      ];
      const { ds, log } = buildStubDataSource(rows, {
        enabled: true,
        source: "manual",
      });
      await computeBrainRuntimeState(ds);
      assert.equal(
        log.fetchLastSuccessfulAtCalls,
        0,
        "fetchLastSuccessfulAt must NOT be called when in-window scan found a successful row",
      );
    }
    // Scenario B: in-window has only abstains -> fallback IS called.
    {
      const rows: SyntheticRow[] = [
        {
          id: 1,
          createdAt: within(5),
          skipReason: "quant_abstain_low_confidence",
          becameTrade: null,
        },
      ];
      const { ds, log } = buildStubDataSource(rows, {
        enabled: true,
        source: "manual",
      });
      await computeBrainRuntimeState(ds);
      assert.equal(
        log.fetchLastSuccessfulAtCalls,
        1,
        "fetchLastSuccessfulAt must be called exactly once when in-window scan found no successful row",
      );
    }
  });

  it("default page size is bounded (≤ 5000) per the task contract", () => {
    assert.ok(
      BRAIN_RUNTIME_PAGE_SIZE > 0 && BRAIN_RUNTIME_PAGE_SIZE <= 5000,
      `BRAIN_RUNTIME_PAGE_SIZE must be 1..5000, got ${BRAIN_RUNTIME_PAGE_SIZE}`,
    );
  });
});
