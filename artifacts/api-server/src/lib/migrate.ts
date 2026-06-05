// Lightweight, idempotent SQL migration runner (task #347).
//
// The project does not yet have a general-purpose migration tool wired
// into deploys — `pnpm --filter @workspace/db push` exists for dev, but
// nothing runs it automatically on boot or as part of the deploy
// pipeline. Task #347's database-level safety net for `price_history`
// needs to apply before the API server starts accepting writes, in BOTH
// dev and production, without hand-running a script.
//
// Rather than introduce drizzle-kit `migrate` and a generated SQL
// folder, this runner ships a small ordered list of inline, explicitly
// idempotent SQL statements. Each statement uses `IF NOT EXISTS` /
// `pg_constraint` guards so re-running on every boot is a no-op once
// applied. New migrations append to the list; old ones stay in place.
// When the list grows large enough to justify a real tool, swap this
// runner out — until then this is the smallest safe thing.

import { pool } from "@workspace/db";
import { logger } from "./logger";

interface Migration {
  readonly id: string;
  readonly sql: string;
}

const MIGRATIONS: readonly Migration[] = [
  {
    id: "001_price_history_cadence_guard",
    // Adds a `cadence` column to `price_history` and a CHECK constraint
    // that rejects any row whose cadence is not '1m'. Existing rows
    // backfill to '1m' via the column default. See
    // `lib/db/src/schema/price_history.ts` for the full contract.
    sql: `
      ALTER TABLE price_history
        ADD COLUMN IF NOT EXISTS cadence text NOT NULL DEFAULT '1m';

      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint
          WHERE conname = 'price_history_cadence_is_1m'
            AND conrelid = 'public.price_history'::regclass
        ) THEN
          ALTER TABLE price_history
            ADD CONSTRAINT price_history_cadence_is_1m
            CHECK (cadence = '1m');
        END IF;
      END
      $$;
    `,
  },
  {
    id: "002_drop_paper_portfolios_realized_pnl_columns",
    // Task #372: the legacy `total_pnl` / `total_pnl_percent` columns on
    // `paper_portfolios` are no longer written or read by any code path
    // (Task #370 routed all P&L through the equity-vs-seed `derivePnl`
    // helper). Drop them so a future code path can't accidentally
    // re-introduce a stale realized-only read. Idempotent via
    // `IF EXISTS`.
    sql: `
      ALTER TABLE paper_portfolios
        DROP COLUMN IF EXISTS total_pnl,
        DROP COLUMN IF EXISTS total_pnl_percent;
    `,
  },
  {
    id: "003_paper_position_marks_table",
    // Task #491: lightweight per-tick mark history for open paper
    // positions. Created here so the live `updatePortfolioValues` loop
    // and the meta-brain replay can rely on the table existing in both
    // dev and prod without a drizzle-kit push step. Idempotent via
    // `IF NOT EXISTS`. See `lib/db/src/schema/paper_position_marks.ts`
    // for the column contract.
    sql: `
      CREATE TABLE IF NOT EXISTS paper_position_marks (
        id           serial PRIMARY KEY,
        position_id  integer NOT NULL,
        trade_id     integer NOT NULL,
        agent_id     integer NOT NULL,
        coin_id      text NOT NULL,
        mark_price   real NOT NULL,
        pnl_pct      real,
        marked_at    timestamptz NOT NULL DEFAULT now()
      );

      CREATE INDEX IF NOT EXISTS paper_position_marks_trade_ts_idx
        ON paper_position_marks (trade_id, marked_at);

      CREATE INDEX IF NOT EXISTS paper_position_marks_ts_idx
        ON paper_position_marks (marked_at);
    `,
  },
  {
    id: "004_drop_predictions_model_contributions_column",
    // Task #501: the per-LLM `predictions.model_contributions` jsonb
    // column was the last surface still carrying gpt/gemini per-prediction
    // attribution. After Tasks #444 / #485 / #500 ripped out the LLM
    // ensemble, nothing on the live decision path or the dashboards reads
    // this column any more (it was only ever written by the deleted
    // GPT+Gemini producer). Drop it so old jsonb blobs stop sitting in
    // the journal forever and so the schema stops advertising a dead
    // attribution surface to contributors. Idempotent via `IF EXISTS` —
    // existing rows' jsonb payloads are dropped along with the column,
    // which is the intended backfill (the data has no remaining reader).
    sql: `
      ALTER TABLE predictions
        DROP COLUMN IF EXISTS model_contributions;
    `,
  },
  {
    id: "005_agents_add_profile_id_column",
    // Task #468 — DB alignment for the deterministic strategy-profile
    // registry. Add the nullable `profile_id text` column on `agents`
    // so `syncAgentProfileIds()` can populate it via the compatibility
    // map at boot. Stays nullable forever — `null` means "not yet
    // swept", and the trade gate's
    // `AgentNotExecutableError(unknown_agent_id)` keeps the row from
    // trading until the next boot completes the sweep. Idempotent.
    sql: `
      ALTER TABLE agents
        ADD COLUMN IF NOT EXISTS profile_id text;
    `,
  },
  {
    id: "006_agent_status_enum_add_quarantine_review",
    // Task #468 — extend `agent_status` so the nightly retirement
    // evaluator can flip rows from `active` to `quarantine_review`
    // without `enum_invalid_input`. `ALTER TYPE … ADD VALUE
    // IF NOT EXISTS` is idempotent (Postgres 12+) and must run
    // outside a transaction block, so it lives in its own migration
    // (the runner's `client.query()` simple-query protocol sends
    // each migration as its own implicit transaction). The matching
    // schema enum is in lib/db/src/schema/agents.ts.
    sql: `
      ALTER TYPE agent_status ADD VALUE IF NOT EXISTS 'quarantine_review';
    `,
  },
  {
    id: "007_agent_status_enum_add_disabled",
    // Task #468 — companion to migration 006. Operators can hard-
    // disable an agent by writing `status='disabled'`; the trade
    // gate refuses both `quarantine_review` and `disabled` rows
    // (see `cache.ts:AgentNotExecutableError(non_active_db_status)`).
    sql: `
      ALTER TYPE agent_status ADD VALUE IF NOT EXISTS 'disabled';
    `,
  },
  {
    id: "008_agents_add_archived_at",
    // Task #512 — add a nullable `archived_at` timestamp on `agents`
    // so the boot-time legacy archive sweep (migration 009) and any
    // future operator-driven archive can mark a row as historical
    // without deleting it. Live executor and baseline rows leave this
    // null. The dashboard's executor surfaces (paper-portfolios,
    // /crypto/agents/families) exclude rows where `archived_at` is
    // not null OR `profile_id='legacy_archived'`.
    sql: `
      ALTER TABLE agents
        ADD COLUMN IF NOT EXISTS archived_at timestamptz;
    `,
  },
  {
    id: "009_agents_archive_legacy_rows",
    // Task #512 — one-shot boot archive of every legacy personality
    // row. Sets `archived_at = now()` and `is_active = false` for
    // every row whose `profile_id='legacy_archived'`. Idempotent: the
    // `archived_at IS NULL` guard means re-running the migration on a
    // db that already archived a row is a no-op (the timestamp stays
    // pinned to the original archive moment). Trade history and the
    // rows themselves stay intact — the rows only stop trading and
    // stop appearing on the live leaderboard.
    sql: `
      UPDATE agents
      SET archived_at = NOW(),
          is_active   = false
      WHERE profile_id = 'legacy_archived'
        AND archived_at IS NULL;
    `,
  },
];

/**
 * Apply every migration in order. Each statement is idempotent so this
 * is safe to call on every boot. Throws on first failure so the server
 * refuses to start with a half-applied schema.
 */
export async function runMigrations(): Promise<void> {
  // When using MSSQL the schema is already provisioned via dump restore.
  // PostgreSQL-specific DDL (DO $$ ... $$, ADD COLUMN IF NOT EXISTS, etc.)
  // cannot run against MSSQL so we skip the migration runner entirely.
  if (process.env.MSSQL_HOST) {
    logger.info("MSSQL mode: skipping PostgreSQL migrations (schema already provisioned)");
    return;
  }

  const client = await pool.connect();
  try {
    for (const migration of MIGRATIONS) {
      const startedAt = Date.now();
      try {
        await client.query(migration.sql);
        logger.info(
          { migrationId: migration.id, durationMs: Date.now() - startedAt },
          "Applied migration (idempotent)",
        );
      } catch (err) {
        logger.error(
          { err, migrationId: migration.id },
          "Migration failed; aborting boot",
        );
        throw err;
      }
    }
  } finally {
    client.release();
  }
}
