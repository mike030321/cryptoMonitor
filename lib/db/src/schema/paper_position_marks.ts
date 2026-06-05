import { pgTable, serial, integer, text, real, timestamp, index } from "drizzle-orm/pg-core";

/**
 * Task #491 — per-tick mark-to-market history for an open paper position.
 *
 * One row per open position per `updatePortfolioValues()` call (every
 * 15s in the live monitor). The row is intentionally tiny — just enough
 * to reconstruct the intra-trade price path so the meta-brain replay
 * (`scripts/replay_meta_brain.py:derive_outcome`) can compute true
 * max-adverse-excursion (`realized_drawdown`) and rolling-stdev
 * (`realized_stability`) instead of falling back to `max(0, -pnl_pct)`
 * and a neutral 0.5.
 *
 * Why a separate table (rather than extending `paper_positions`):
 *   * `paper_positions` is one-row-per-open-position and is deleted on
 *     close — extending it would either lose the history at close or
 *     bloat the row.
 *   * The replay needs the per-tick stream after the trade is closed,
 *     so the marks must outlive the open position. Joining via
 *     `trade_id` (which lives on `paper_trades`) keeps the link stable.
 *
 * Retention: NOT enforced here — these are append-only marks and the
 * table can grow ~1 row per open position per 15s. Trim policy is
 * tracked as a follow-up so this task stays scoped to "richer feedback
 * for the replay".
 */
export const paperPositionMarksTable = pgTable(
  "paper_position_marks",
  {
    id: serial("id").primaryKey(),
    // Foreign-key-by-convention. The position row is deleted on close,
    // so we cannot enforce a real FK without losing marks; `tradeId`
    // below is the durable join key for the replay.
    positionId: integer("position_id").notNull(),
    tradeId: integer("trade_id").notNull(),
    agentId: integer("agent_id").notNull(),
    coinId: text("coin_id").notNull(),
    // Mark-to-market price at sample time (raw, not slippage-adjusted —
    // the replay's MAE math compares against `paper_positions.entry_price`
    // which is also raw at open time).
    markPrice: real("mark_price").notNull(),
    // Convenience: signed unrealized pnl as a fraction of position size
    // (post-fee/slippage estimate, mirroring the value used in
    // `updatePortfolioValues`). The replay does not require this — it
    // recomputes from `markPrice` against entry — but it makes the table
    // human-readable for ad-hoc inspection.
    pnlPct: real("pnl_pct"),
    markedAt: timestamp("marked_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => ({
    byTradeTs: index("paper_position_marks_trade_ts_idx").on(t.tradeId, t.markedAt),
    byTs: index("paper_position_marks_ts_idx").on(t.markedAt),
  }),
);

export type PaperPositionMark = typeof paperPositionMarksTable.$inferSelect;
