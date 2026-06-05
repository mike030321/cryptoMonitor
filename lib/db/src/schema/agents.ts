import { pgTable, text, serial, timestamp, integer, real, pgEnum, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

// Task #468 — added `quarantine_review` and `disabled` so the nightly
// retirement evaluator can flip a row out of `active` when its profile
// rule triggers, and so the trade gate has a typed status to refuse
// against (in addition to the registry's own `executes=false`).
export const agentStatusEnum = pgEnum("agent_status", [
  "active",
  "resting",
  "degraded",
  "quarantine_review",
  "disabled",
]);
export const streakTypeEnum = pgEnum("streak_type", ["win", "loss", "none"]);

export const agentsTable = pgTable("agents", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  personality: text("personality").notNull(),
  score: real("score").notNull().default(100),
  totalPredictions: integer("total_predictions").notNull().default(0),
  correctPredictions: integer("correct_predictions").notNull().default(0),
  wrongPredictions: integer("wrong_predictions").notNull().default(0),
  streak: integer("streak").notNull().default(0),
  streakType: streakTypeEnum("streak_type").notNull().default("none"),
  status: agentStatusEnum("status").notNull().default("active"),
  generation: integer("generation").notNull().default(1),
  parentIds: text("parent_ids"),
  evolutionMethod: text("evolution_method").notNull().default("original"),
  isActive: boolean("is_active").notNull().default(true),
  systemPrompt: text("system_prompt"),
  temperature: real("temperature"),
  preferredTimeframes: text("preferred_timeframes"),
  strategyType: text("strategy_type").notNull().default("ai-bots"),
  // Task #468 — registry profile id. Boot sweeps populate this from
  // the legacy `name`/`personality` via the compatibility map; every
  // live consumer reads behaviour through `getAgentProfile(profile_id)`
  // and unknown ids throw (never default).
  profileId: text("profile_id"),
  // Task #512 — set to NOW() at boot when profile_id='legacy_archived'.
  // Rows with a non-null archivedAt are excluded from the live executor
  // surfaces (paper-portfolios, family summary) but their history stays
  // queryable on the Archived Agents page. Live executor + baseline
  // reference rows leave this null.
  archivedAt: timestamp("archived_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow().$onUpdate(() => new Date()),
});

export const insertAgentSchema = createInsertSchema(agentsTable).omit({ id: true, createdAt: true, updatedAt: true });
export type InsertAgent = z.infer<typeof insertAgentSchema>;
export type Agent = typeof agentsTable.$inferSelect;
