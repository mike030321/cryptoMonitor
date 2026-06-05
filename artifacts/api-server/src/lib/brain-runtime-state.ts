import { and, desc, eq, gte, sql } from "drizzle-orm";

import { db, predictionJournalTable } from "@workspace/db";

import { getBrainState } from "./brain-flag";

/**
 * Task #680 — extracted from `routes/crypto/index.ts` so the
 * prediction-journal scan that backs `/crypto/brain/runtime-status`
 * can be unit-tested with a stubbed db.
 *
 * The previous implementation issued a single
 * `db.select(...).from(predictionJournalTable).where(gte(createdAt, since))`
 * with `since = now − 30 min` and **no `.limit(...)`** — under any
 * prediction-journal surge (a busy meta-brain epoch, a backfill,
 * a multi-coin update burst) this pulled an unbounded row count
 * into Node memory. The route is hit at the dashboard's polling
 * cadence, so the load was recurring.
 *
 * The fix: paginate the SELECT with a bounded page size and
 * stream the aggregation. The aggregation output
 * (`BrainRuntimeStatePayload`) is byte-identical for the same
 * input rows — counters add, the `lastSuccessfulAt` update is a
 * monotonic max, and the `state` discriminant logic is unchanged.
 */

export const NO_MODEL_REASONS: ReadonlySet<string> = new Set([
  "no_model",
  "model_load_failed",
  "model_unavailable",
  "missing_model",
]);

export type BrainRuntimeStateName =
  | "online"
  | "offline_no_model"
  | "offline_disabled";

export interface BrainRuntimeStatePayload {
  state: BrainRuntimeStateName;
  brainEnabled: boolean;
  brainSource: string;
  recentAbstainReasons: Record<string, number>;
  recentNonAbstainCount: number;
  lastSuccessfulAt: Date | null;
}

/**
 * Page size for the prediction-journal scan. 5000 rows ≈ a few MB
 * at the journal's column shape and is well below the upstream
 * audit's "≥ 50 000 synthetic rows" stress target. Each page is
 * consumed and made garbage-collectable before the next fetch.
 *
 * The pager fetches `BRAIN_RUNTIME_PAGE_SIZE + 1` rows per round-trip
 * so it can decide "no more data" without an extra empty-page probe:
 * if the result has `pageSize + 1` rows the loop processes `pageSize`
 * of them and advances the cursor to the last processed row; if it
 * has `<= pageSize` rows the loop processes all of them and stops.
 * This caps the call count at exactly `ceil(total / pageSize)` even
 * when `total` is an exact multiple of `pageSize`.
 */
export const BRAIN_RUNTIME_PAGE_SIZE = 5000;

export interface BrainRuntimeJournalRow {
  id: number;
  skipReason: string | null;
  becameTrade: boolean | null;
  createdAt: Date | null;
}

export interface BrainRuntimeJournalCursor {
  createdAt: Date;
  id: number;
}

export interface BrainRuntimeStateDataSource {
  /**
   * Fetch one page of journal rows whose `createdAt >= since`,
   * ordered by `(createdAt, id)` ascending, strictly after
   * `afterCursor` (lexicographic row comparison) when supplied,
   * limited to `limit` rows. Returning fewer than `limit` rows
   * signals "no more pages".
   */
  fetchJournalPage(opts: {
    since: Date;
    afterCursor: BrainRuntimeJournalCursor | null;
    limit: number;
  }): Promise<BrainRuntimeJournalRow[]>;
  /**
   * Fallback used only when no `becameTrade=true` row was found
   * inside the 30-minute window. Returns the most recent
   * `becameTrade=true` `createdAt` across all of history, or null.
   */
  fetchLastSuccessfulAt(): Promise<Date | null>;
  getBrainState(): Promise<{ enabled: boolean; source: string }>;
}

export const productionBrainRuntimeDataSource: BrainRuntimeStateDataSource = {
  async fetchJournalPage({ since, afterCursor, limit }) {
    const conds = [gte(predictionJournalTable.createdAt, since)];
    if (afterCursor) {
      conds.push(
        sql`(${predictionJournalTable.createdAt}, ${predictionJournalTable.id}) > (${afterCursor.createdAt}, ${afterCursor.id})`,
      );
    }
    return db
      .select({
        id: predictionJournalTable.id,
        skipReason: predictionJournalTable.skipReason,
        becameTrade: predictionJournalTable.becameTrade,
        createdAt: predictionJournalTable.createdAt,
      })
      .from(predictionJournalTable)
      .where(and(...conds))
      .orderBy(predictionJournalTable.createdAt, predictionJournalTable.id)
      .limit(limit);
  },
  async fetchLastSuccessfulAt() {
    const fallback = await db
      .select({ createdAt: predictionJournalTable.createdAt })
      .from(predictionJournalTable)
      .where(and(eq(predictionJournalTable.becameTrade, true)))
      .orderBy(desc(predictionJournalTable.createdAt))
      .limit(1);
    return fallback[0]?.createdAt ?? null;
  },
  async getBrainState() {
    return getBrainState();
  },
};

export async function computeBrainRuntimeState(
  ds: BrainRuntimeStateDataSource = productionBrainRuntimeDataSource,
  pageSize: number = BRAIN_RUNTIME_PAGE_SIZE,
): Promise<BrainRuntimeStatePayload> {
  const brainState = await ds.getBrainState();
  const since = new Date(Date.now() - 30 * 60 * 1000);

  const recentAbstainReasons: Record<string, number> = {};
  let recentNonAbstain = 0;
  let lastSuccessfulAt: Date | null = null;

  let cursor: BrainRuntimeJournalCursor | null = null;
  while (true) {
    // Fetch one extra row so we can decide "no more data" without a
    // trailing empty-page probe when `total` is an exact multiple of
    // `pageSize`. See BRAIN_RUNTIME_PAGE_SIZE comment for the call-
    // count proof.
    const fetched = await ds.fetchJournalPage({
      since,
      afterCursor: cursor,
      limit: pageSize + 1,
    });
    const hasMore = fetched.length > pageSize;
    const page = hasMore ? fetched.slice(0, pageSize) : fetched;
    for (const r of page) {
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
    if (!hasMore) break;
    const last = page[page.length - 1];
    if (!last.createdAt) {
      // The schema declares `created_at NOT NULL` (see
      // lib/db/src/schema/prediction_journal.ts) and the SELECT
      // filters by `gte(createdAt, since)` which already excludes
      // null rows. A null here means the data source contract was
      // violated — fail fast instead of silently truncating the scan.
      throw new Error(
        `brain-runtime-state: journal row ${last.id} has null createdAt; cannot advance cursor`,
      );
    }
    cursor = { createdAt: last.createdAt, id: last.id };
  }

  if (!lastSuccessfulAt) {
    lastSuccessfulAt = await ds.fetchLastSuccessfulAt();
  }

  const noModelAbstains = Object.entries(recentAbstainReasons)
    .filter(([reason]) => NO_MODEL_REASONS.has(reason))
    .reduce((sum, [, count]) => sum + count, 0);
  const totalAbstains = Object.values(recentAbstainReasons).reduce(
    (sum, count) => sum + count,
    0,
  );

  let state: BrainRuntimeStateName;
  if (!brainState.enabled) {
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
    state,
    brainEnabled: brainState.enabled,
    brainSource: brainState.source,
    recentAbstainReasons,
    recentNonAbstainCount: recentNonAbstain,
    lastSuccessfulAt,
  };
}
