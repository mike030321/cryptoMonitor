# Task #581 — 5m Historical Backfill: Gap Audit

**Generated:** 2026-04-28T21:10:00Z
**Script:** `artifacts/ml-engine/scripts/backfill_5m_extend.py`
**Run window:** 2026-04-28T20:23:13Z → 2026-04-28T21:08:31Z (≈45m wall-clock, well under 2h ceiling)
**Lock label:** `ml_engine.scheduled_5m_topup.historical_backfill` (distinct from `_TOPUP_LOCK_LABEL`)
**Source contract:** OKX `history-candles` only (Coinbase fallback path was not exercised — none of the 9 affected coins are in the Coinbase symbol map for 5m)

## Pre-run baseline (from gap audit at 2026-04-28T19:40Z)

All 9 affected coins had:
- ~18,883 rows each
- oldest_bucket = 2026-02-22 (≈65.5 days deep)
- 100% `source='okx'`
- 0 internal gaps > 1h
- Verdict: real data, just shallow — regression cause was that nothing was inserted before 2026-02-22, not corruption.

## Post-run audit

| coin | rows | oldest_bucket (UTC) | newest_bucket (UTC) | coverage_days | gaps>1h | max_gap_min |
|---|---|---|---|---|---|---|
| bonk | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| celestia | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| dogwifcoin | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| floki-inu | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| injective-protocol | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| jupiter-exchange-solana | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| pepe | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| render-token | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| worldcoin-wld | 92,200 | 2025-06-12 17:05 | 2026-04-28 20:20 | 320.17 | 0 | 0 |
| sei-network† | 47,681 | 2025-11-14 07:00 | 2026-04-28 20:20 | 165.59 | 0 | 0 |

† sei-network is included in `OKX_SYMBOLS` so the script processed it for free. It is **not** one of the 9 affected coins from the original goal. OKX returned only ~165 days of depth for SEI-USDT before the page reached its end-of-history sentinel; this is an upstream-data limitation, not a backfill defect. SEI's 5m bar coverage is tracked separately.

## Source breakdown (post-run)

```
bonk                    : okx=92,200   synthetic=0   coinbase=0   other=0
celestia                : okx=92,200   synthetic=0   coinbase=0   other=0
dogwifcoin              : okx=92,200   synthetic=0   coinbase=0   other=0
floki-inu               : okx=92,200   synthetic=0   coinbase=0   other=0
injective-protocol      : okx=92,200   synthetic=0   coinbase=0   other=0
jupiter-exchange-solana : okx=92,200   synthetic=0   coinbase=0   other=0
pepe                    : okx=92,200   synthetic=0   coinbase=0   other=0
render-token            : okx=92,200   synthetic=0   coinbase=0   other=0
worldcoin-wld           : okx=92,200   synthetic=0   coinbase=0   other=0
sei-network             : okx=47,681   synthetic=0   coinbase=0   other=0
```

**100% `source='okx'`. Zero `source='synthetic'` rows.** Real data only.

## Gap distribution

For all 9 affected coins: zero internal gaps strictly greater than 1 hour between consecutive 5m bucket starts. (Buckets are spaced exactly 5 minutes apart end-to-end from 2025-06-12T17:05Z to 2026-04-28T20:20Z, modulo the boundary at the most-recent bucket which the dataset-refresher will continue to top up cadence-driven.)

## Per-coin run telemetry (from `models/progress_updates.jsonl`, phase=`5m_historical_backfill`)

| coin | pulled | inserted (new) | elapsed_s | deadline_reached |
|---|---|---|---|---|
| bonk | 92,200 | 73,317 | 2,703.2 | false |
| celestia | 92,200 | 73,317 | 2,718.4 | false |
| dogwifcoin | 92,200 | 73,317 | 2,717.2 | false |
| floki-inu | 92,200 | 73,316 | 2,718.4 | false |
| injective-protocol | 92,200 | 73,317 | 2,717.0 | false |
| jupiter-exchange-solana | 92,200 | 73,318 | 2,718.4 | false |
| pepe | 92,200 | 73,313 | 2,707.4 | false |
| render-token | 92,200 | 73,318 | 2,705.6 | false |
| worldcoin-wld | 92,200 | 73,316 | 2,712.5 | false |
| sei-network | 47,681 | 30,092 | 1,484.7 | false |

`pulled - inserted ≈ 18,883` per affected coin = the pre-existing rows the ON CONFLICT DO NOTHING dedup correctly skipped. Idempotency confirmed.

## Operational verification

- Postgres advisory lock acquired and released cleanly via `db_mod.try_advisory_lock` keyed on `ml_engine.scheduled_5m_topup.historical_backfill`. (One earlier attempt correctly emitted `skipped_locked` because an orphaned backend from a SIGTERM'd predecessor still held the lock; that backend was terminated and the next attempt acquired and finished.)
- `shared/timeframe-roles.json` was **not** touched. The 5m role remains `disabled`; this backfill prepares the data for a future role flip but does not perform one.
- `app/cadence_guard.py` `COVERAGE_BAR_DAYS["5m"]` (305) is unchanged.
- `app/scheduled_5m_topup.py` is unchanged.
- 2h wall-clock ceiling honored (`BACKFILL_5M_DEADLINE_SECONDS=7200`); actual run wrapped at ≈45 min, no `deadline_reached=true` for any coin.
