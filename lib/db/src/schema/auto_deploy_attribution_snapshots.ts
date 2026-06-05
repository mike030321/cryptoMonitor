import { pgTable, serial, timestamp, doublePrecision, integer, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const autoDeployAttributionSnapshotsTable = pgTable(
  "auto_deploy_attribution_snapshots",
  {
    id: serial("id").primaryKey(),
    capturedAt: timestamp("captured_at", { withTimezone: true }).notNull().defaultNow(),
    longRealizedPnlUsd: doublePrecision("long_realized_pnl_usd").notNull().default(0),
    longUnrealizedPnlUsd: doublePrecision("long_unrealized_pnl_usd").notNull().default(0),
    shortRealizedPnlUsd: doublePrecision("short_realized_pnl_usd").notNull().default(0),
    shortUnrealizedPnlUsd: doublePrecision("short_unrealized_pnl_usd").notNull().default(0),
    totalNetPnlUsd: doublePrecision("total_net_pnl_usd").notNull().default(0),
    deployedUsd: doublePrecision("deployed_usd").notNull().default(0),
    closedTrades: integer("closed_trades").notNull().default(0),
    openPositions: integer("open_positions").notNull().default(0),
  },
  (t) => ({
    capturedAtIdx: index("auto_deploy_attribution_snapshots_captured_at_idx").on(t.capturedAt),
  }),
);

export const insertAutoDeployAttributionSnapshotSchema = createInsertSchema(autoDeployAttributionSnapshotsTable).omit({ id: true });
export type InsertAutoDeployAttributionSnapshot = z.infer<typeof insertAutoDeployAttributionSnapshotSchema>;
export type AutoDeployAttributionSnapshot = typeof autoDeployAttributionSnapshotsTable.$inferSelect;
