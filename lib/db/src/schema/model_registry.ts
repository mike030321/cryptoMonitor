import {
  pgTable,
  serial,
  text,
  integer,
  boolean,
  timestamp,
  jsonb,
  index,
  uniqueIndex,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

/**
 * Phase 5 — Model registry lifecycle.
 *
 * Tracks every (model_id, model_version, coin_id, timeframe) slot through
 * its lifecycle states:
 *   • shadow      — receiving inputs, predictions journaled with shadow=true,
 *                   never participates in live trading.
 *   • challenger  — eligible for promotion gate evaluation; still does not
 *                   trade live (live trader still routes to current champion).
 *   • champion    — the model whose decisions actually open live trades.
 *                   At most one champion per (coin_id|"*", timeframe|"*") slot.
 *   • quarantined — previously a champion / challenger but failed a runtime
 *                   guard (drift, exception storm, etc.). Stays out of trading
 *                   until manually un-quarantined.
 *   • retired     — previously a champion that has been rolled back; kept on
 *                   record for audit / rollback target lookup.
 *
 * `previousChampionId` is the registry row id of the model this row REPLACED
 * when it was promoted. One-click rollback simply demotes this row to
 * "retired" and re-promotes `previousChampionId` to "champion".
 *
 * `metricsSnapshot` is the gate-evaluator output captured at the moment of
 * the last state transition — preserves "why we promoted" / "why we rolled
 * back" without having to recompute from the journal.
 */
export const modelRegistryTable = pgTable(
  "model_registry",
  {
    id: serial("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),

    modelId: text("model_id").notNull(),         // e.g. "lightgbm" / "lightgbm-meta"
    modelVersion: text("model_version").notNull(),
    coinId: text("coin_id").notNull().default("*"),       // "*" = all coins
    timeframe: text("timeframe").notNull().default("*"),  // "*" = all timeframes

    state: text("state").notNull(),              // shadow|challenger|champion|quarantined|retired

    promotedAt: timestamp("promoted_at", { withTimezone: true }),
    demotedAt: timestamp("demoted_at", { withTimezone: true }),
    previousChampionId: integer("previous_champion_id"),

    note: text("note"),
    metricsSnapshot: jsonb("metrics_snapshot"),  // PromotionMetrics + verdict
    isActive: boolean("is_active").notNull().default(true),

    // Task #654 — Paper trading scope constraint. When non-null on an
    // active champion row, /ml/predict refuses any (coin, timeframe)
    // request that does not match the recorded scope and returns
    // `{"status": "out_of_scope", "scope_constraint": {...}}` instead of
    // a model-backed prediction. Shape is intentionally schema-light so
    // future expansions (regimes, sessions, feature_hash buckets) can be
    // added without a migration:
    //   { "coins": ["bitcoin", ...] | null,
    //     "timeframes": ["5m", ...] | null,
    //     ...arbitrary additional keys an operator stamps at promotion }
    // A `null` (or missing) coins/timeframes list means "no restriction
    // along that axis". Set ONLY by `promote_shadow_to_serving`; never
    // mutated for shadow / quarantined rows.
    scopeConstraint: jsonb("scope_constraint"),
  },
  (t) => ({
    stateIdx: index("model_registry_state_idx").on(t.state),
    slotIdx: index("model_registry_slot_idx").on(t.modelId, t.coinId, t.timeframe),
    activeChampionUnique: uniqueIndex("model_registry_active_champion_unique")
      .on(t.modelId, t.coinId, t.timeframe)
      .where(sql`state = 'champion' AND is_active = true`),
  }),
);

export const insertModelRegistrySchema = createInsertSchema(modelRegistryTable).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});
export type InsertModelRegistry = z.infer<typeof insertModelRegistrySchema>;
export type ModelRegistryRow = typeof modelRegistryTable.$inferSelect;

export const MODEL_REGISTRY_STATES = [
  "shadow",
  "challenger",
  "champion",
  "quarantined",
  "retired",
] as const;
export type ModelRegistryState = (typeof MODEL_REGISTRY_STATES)[number];
