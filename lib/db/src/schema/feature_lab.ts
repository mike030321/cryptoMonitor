import {
  pgTable,
  serial,
  text,
  integer,
  real,
  boolean,
  timestamp,
  jsonb,
  index,
  uniqueIndex,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

/**
 * Phase 6 — Feature Lab.
 *
 * `feature_lab_candidates` — every proposed new feature waiting for a
 * walk-forward ablation report. A candidate references a deterministic
 * transform kind (one of FEATURE_TRANSFORM_KINDS) so the ml-engine
 * ablation runner can compute its values from the existing feature
 * column set without ever evaluating user-supplied code.
 *
 * `feature_lab_reports` — one row per ablation run for a candidate.
 * Stores baseline-vs-augmented walk-forward metrics so the dashboard can
 * render a side-by-side delta and gate the "Promote feature" button on
 * positive delta + minimum sample count. Approving a candidate bumps the
 * `approved_at` column on `feature_lab_candidates` and writes the
 * promoted feature spec into `app_settings` for the next training run
 * to pick up.
 */
export const FEATURE_TRANSFORM_KINDS = [
  "passthrough_existing",
  "log_realized_vol",
  "rsi_squared",
  "macd_x_atr",
  "ret5_minus_ret10",
  "bb_pctb_squared",
] as const;
export type FeatureTransformKind = (typeof FEATURE_TRANSFORM_KINDS)[number];

export const FEATURE_LAB_STATES = [
  "draft",
  "ablated",
  "approved",
  "rejected",
  "quarantined",
] as const;
export type FeatureLabState = (typeof FEATURE_LAB_STATES)[number];

export const featureLabCandidatesTable = pgTable(
  "feature_lab_candidates",
  {
    id: serial("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),

    name: text("name").notNull(),                    // unique candidate name
    description: text("description"),
    transformKind: text("transform_kind").notNull(), // one of FEATURE_TRANSFORM_KINDS
    sourceColumn: text("source_column"),             // for passthrough_existing
    state: text("state").notNull().default("draft"), // draft|ablated|approved|rejected

    proposedBy: text("proposed_by"),                 // free-form ("ops"/"auto-search"/etc)
    approvedAt: timestamp("approved_at", { withTimezone: true }),
    approvedBy: text("approved_by"),
    approvalNote: text("approval_note"),
  },
  (t) => ({
    nameUnique: uniqueIndex("feature_lab_candidates_name_unique").on(t.name),
    stateIdx: index("feature_lab_candidates_state_idx").on(t.state),
  }),
);

export const featureLabReportsTable = pgTable(
  "feature_lab_reports",
  {
    id: serial("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),

    candidateId: integer("candidate_id").notNull(),
    timeframe: text("timeframe").notNull(),
    coinId: text("coin_id").notNull().default("__pooled__"),

    // Walk-forward fold count actually run (the ablation runner cap).
    nFolds: integer("n_folds").notNull(),
    nSamples: integer("n_samples").notNull(),

    // Baseline (without candidate) and augmented (with candidate) metrics.
    baselineLogLoss: real("baseline_log_loss"),
    augmentedLogLoss: real("augmented_log_loss"),
    deltaLogLoss: real("delta_log_loss"),               // baseline - augmented (positive = better)
    baselineAccuracy: real("baseline_accuracy"),
    augmentedAccuracy: real("augmented_accuracy"),
    deltaAccuracy: real("delta_accuracy"),

    // Free-form per-fold detail / extra notes from the ml-engine runner.
    extra: jsonb("extra"),

    runnerStatus: text("runner_status").notNull(),     // "ok" | "error"
    runnerError: text("runner_error"),
  },
  (t) => ({
    candidateIdx: index("feature_lab_reports_candidate_idx").on(t.candidateId),
    createdAtIdx: index("feature_lab_reports_created_at_idx").on(t.createdAt),
  }),
);

export const insertFeatureLabCandidateSchema = createInsertSchema(featureLabCandidatesTable).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});
export type InsertFeatureLabCandidate = z.infer<typeof insertFeatureLabCandidateSchema>;
export type FeatureLabCandidateRow = typeof featureLabCandidatesTable.$inferSelect;

export const insertFeatureLabReportSchema = createInsertSchema(featureLabReportsTable).omit({
  id: true,
  createdAt: true,
});
export type InsertFeatureLabReport = z.infer<typeof insertFeatureLabReportSchema>;
export type FeatureLabReportRow = typeof featureLabReportsTable.$inferSelect;

/**
 * Phase 6 — drift snapshots & quarantine events.
 *
 * The drift trackers compute calibration error, prediction-distribution
 * drift, and per-feature drift on demand from `prediction_journal`. This
 * table persists a periodic snapshot per (registryId, kind) so the UI
 * can render a time series without re-aggregating the full journal on
 * every refresh, and so the auto-quarantine routine has stable historical
 * baselines.
 *
 * `quarantine_events` records each automatic (or manual) state transition
 * to `quarantined`, with the reason code and the metric snapshot that
 * triggered it. Joined back to `model_registry.id`.
 */
export const DRIFT_KINDS = [
  "calibration",
  "prediction_distribution",
  "feature",
] as const;
export type DriftKind = (typeof DRIFT_KINDS)[number];

export const driftSnapshotsTable = pgTable(
  "drift_snapshots",
  {
    id: serial("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),

    registryId: integer("registry_id"),                  // null = global / cross-model
    coinId: text("coin_id").notNull().default("*"),
    timeframe: text("timeframe").notNull().default("*"),

    kind: text("kind").notNull(),                        // DRIFT_KINDS

    nSamples: integer("n_samples").notNull(),
    score: real("score").notNull(),                      // unitless drift magnitude (kind-specific)
    threshold: real("threshold").notNull(),
    breached: boolean("breached").notNull().default(false),

    detail: jsonb("detail"),                             // per-bucket / per-feature breakdown
  },
  (t) => ({
    createdAtIdx: index("drift_snapshots_created_at_idx").on(t.createdAt),
    kindRegistryIdx: index("drift_snapshots_kind_registry_idx").on(t.kind, t.registryId),
  }),
);

export const quarantineEventsTable = pgTable(
  "quarantine_events",
  {
    id: serial("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),

    registryId: integer("registry_id").notNull(),
    fromState: text("from_state").notNull(),
    toState: text("to_state").notNull(),                 // typically "quarantined"
    reasonCode: text("reason_code").notNull(),           // calibration_drift | feature_drift | prob_collapse | manual
    triggeredBy: text("triggered_by").notNull(),         // "auto" | "operator"
    detail: jsonb("detail"),
  },
  (t) => ({
    createdAtIdx: index("quarantine_events_created_at_idx").on(t.createdAt),
    registryIdx: index("quarantine_events_registry_idx").on(t.registryId),
  }),
);

export const insertDriftSnapshotSchema = createInsertSchema(driftSnapshotsTable).omit({
  id: true,
  createdAt: true,
});
export type InsertDriftSnapshot = z.infer<typeof insertDriftSnapshotSchema>;
export type DriftSnapshotRow = typeof driftSnapshotsTable.$inferSelect;

export const insertQuarantineEventSchema = createInsertSchema(quarantineEventsTable).omit({
  id: true,
  createdAt: true,
});
export type InsertQuarantineEvent = z.infer<typeof insertQuarantineEventSchema>;
export type QuarantineEventRow = typeof quarantineEventsTable.$inferSelect;

/**
 * Task #247 — audit log for every operator un-quarantine override on a
 * Feature Lab candidate. Each row captures a point-in-time snapshot of
 * the prior quarantine reason so reviewers can later spot patterns
 * (e.g. the same operator repeatedly un-quarantines features that go
 * on to regress again). The candidate row's `approvalNote` only keeps
 * the most recent override; this table is the time-ordered history.
 */
export const featureLabUnquarantineEventsTable = pgTable(
  "feature_lab_unquarantine_events",
  {
    id: serial("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),

    candidateId: integer("candidate_id").notNull(),
    candidateName: text("candidate_name").notNull(),
    operator: text("operator").notNull(),
    note: text("note"),
    priorReason: text("prior_reason"),
    priorReasonDetail: jsonb("prior_reason_detail"),
    priorQuarantinedAt: timestamp("prior_quarantined_at", { withTimezone: true }),
  },
  (t) => ({
    createdAtIdx: index("feature_lab_unquarantine_events_created_at_idx").on(t.createdAt),
    candidateIdx: index("feature_lab_unquarantine_events_candidate_idx").on(t.candidateId),
  }),
);

export const insertFeatureLabUnquarantineEventSchema = createInsertSchema(
  featureLabUnquarantineEventsTable,
).omit({ id: true, createdAt: true });
export type InsertFeatureLabUnquarantineEvent = z.infer<
  typeof insertFeatureLabUnquarantineEventSchema
>;
export type FeatureLabUnquarantineEventRow =
  typeof featureLabUnquarantineEventsTable.$inferSelect;
