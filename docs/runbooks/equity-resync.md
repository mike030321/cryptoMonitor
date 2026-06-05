# Equity Re-sync Runbook

**Audit reference:** `docs/audits/2026-04-23-full-system-audit.md` § B-EQUITY-PROD-STALE
**Source:** `artifacts/api-server/src/lib/paper-trader.ts:1311-1370` (`updatePortfolioValues`)
**When to run:** any time the paper-portfolio cache (`paper_portfolios.total_value`) drifts from the live mark-to-market value of `paper_positions` for an agent. Symptoms: stale `total_value` while open positions are clearly off-water, or staleness > 15 min on the dashboard "Updated" badge.

## Background

`updatePortfolioValues` reads every open `paper_positions` row, multiplies by the current price from `coins`, and writes:
- `paper_portfolios.total_value = cash_balance + Σ (qty × price)`
- `paper_portfolios.peak_value = MAX(peak_value, total_value)`
- `paper_portfolios.day_start_value` (rolled at UTC midnight)
- `paper_portfolios.updated_at = NOW()`

The auto-loop calls this every 30s. The function CAN crash silently if a coin row is missing — the bad-coin path was the root cause of the stale agents 31/32/39.

## Dry-run

```sh
# Capture the baseline so the runbook is auditable
psql "$DATABASE_URL" -c "
  SELECT a.id, a.name, p.cash_balance, p.total_value, p.updated_at,
         NOW() - p.updated_at AS staleness
  FROM agents a JOIN paper_portfolios p ON p.agent_id=a.id
  WHERE a.id IN (31,32,39) ORDER BY a.id;" > /tmp/equity_before.tsv

# Trigger an out-of-band recompute via the api-server admin endpoint.
# (Internal endpoint; requires API_ADMIN_TOKEN.)
curl -sS -X POST -H "x-admin-token: $API_ADMIN_TOKEN" \
  http://localhost:5000/api/crypto/admin/equity-resync \
  -d '{"agentIds":[31,32,39]}'

# Re-sample
psql "$DATABASE_URL" -c "
  SELECT a.id, a.name, p.total_value, p.updated_at, NOW() - p.updated_at AS staleness
  FROM agents a JOIN paper_portfolios p ON p.agent_id=a.id
  WHERE a.id IN (31,32,39) ORDER BY a.id;" > /tmp/equity_after.tsv

diff /tmp/equity_before.tsv /tmp/equity_after.tsv
```

## Acceptance criteria

- All three agents' `updated_at` is within 60s of `NOW()` after the call.
- `total_value` matches `cash_balance + Σ (qty × current_price)` (re-derive from `paper_positions` ⨝ `coins`).
- The next 30s heartbeat keeps `updated_at` rolling forward (i.e. the auto-loop is not stuck).

## Rollback

There is no rollback — the function only writes derived values. If a bad recompute is committed, simply allow the next auto-loop tick to overwrite it. The previous values are also captured in `/tmp/equity_before.tsv` for comparison.
