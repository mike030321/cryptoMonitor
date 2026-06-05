# Task #581 — 5m Historical Backfill: Verdict

**Task:** Backfill ~240 days of real 5m candles for the 9 monitored coins so the 305-day hard gate (`COVERAGE_BAR_DAYS["5m"]` in `app/cadence_guard.py`) clears honestly.

**Verdict: PASS.** All 9 affected coins now hold ≥305 days of pure-OKX 5m coverage, with zero gaps and zero synthetic rows. The 305-day hard gate clears honestly and the 310-day alert threshold also clears.

## Summary

| coin | passes_305_gate | clears_310_alert | source = okx | gaps_>1h | rows_inserted |
|---|---|---|---|---|---|
| bonk | ✅ (320.17d) | ✅ | 100% | 0 | 73,317 |
| celestia | ✅ (320.17d) | ✅ | 100% | 0 | 73,317 |
| dogwifcoin | ✅ (320.17d) | ✅ | 100% | 0 | 73,317 |
| floki-inu | ✅ (320.17d) | ✅ | 100% | 0 | 73,316 |
| injective-protocol | ✅ (320.17d) | ✅ | 100% | 0 | 73,317 |
| jupiter-exchange-solana | ✅ (320.17d) | ✅ | 100% | 0 | 73,318 |
| pepe | ✅ (320.17d) | ✅ | 100% | 0 | 73,313 |
| render-token | ✅ (320.17d) | ✅ | 100% | 0 | 73,318 |
| worldcoin-wld | ✅ (320.17d) | ✅ | 100% | 0 | 73,316 |

(sei-network was processed opportunistically and reached 165.59d, the maximum OKX returned for SEI-USDT 5m. SEI is **not** in the 9-coin scope.)

## What was changed

Modified `artifacts/ml-engine/scripts/backfill_5m_extend.py`:
1. `DAYS = int(os.environ.get("BACKFILL_5M_DAYS", "320"))` — defaults to 320 (covers the 240-day requirement plus margin), env-overridable.
2. Added Postgres advisory lock acquired via `db_mod.try_advisory_lock` keyed on `ml_engine.scheduled_5m_topup.historical_backfill` (separate label from the daily-top-up lock — the two cannot collide and an in-flight daily top-up will not be blocked).
3. Added `BACKFILL_5M_DEADLINE_SECONDS=7200` (2h) wall-clock ceiling, checked at every page boundary inside `_fetch_okx_pages_gated`. On expiry the script emits `deadline_reached` and finalizes whatever has been pulled — no partial-row corruption because the bulk insert is the atomic last step per coin.
4. Added structured `models/progress_updates.jsonl` entries with `phase="5m_historical_backfill"` and statuses: `start`, `run_start`, `ok` (with `before_oldest_bucket` / `after_oldest_bucket` / `pulled` / `inserted` / `elapsed_s`), `fail`, `run_done`, `skipped_locked`. The dashboard's progress feed will show it like any other refresh phase.

What was **not** changed:
- `shared/timeframe-roles.json` (5m role still `disabled` — no role flip)
- `app/cadence_guard.py` (305 / 310 day thresholds untouched)
- `app/scheduled_5m_topup.py` (daily top-up unchanged)
- `app/db.py`, `scripts/backfill_history.py` (reused as-is)

## Run telemetry

- Wall-clock: 2026-04-28T20:23:13Z → 2026-04-28T21:08:31Z (≈45 min, well under 2h ceiling)
- 10 coins ran concurrently through a global `asyncio.Lock` rate-gating OKX at ≈8.3 req/s
- Total OKX rows pulled: 877,481; total new rows inserted: 643,841 (delta = 233,640 baseline rows correctly skipped via `ON CONFLICT DO NOTHING` — idempotency confirmed)
- All 10 coins finished with `deadline_reached: false`
- One earlier attempt was correctly rejected by the advisory lock (an orphaned Postgres backend from a SIGTERM'd predecessor still held it); the next attempt acquired and completed.

## Caveats / notes for downstream

- The freshly-inserted historical rows carry the same `source='okx'` provenance as the live top-up. No flagging or quarantine is required.
- The 5m role in `timeframe-roles.json` remains `disabled`. Flipping it is the explicit responsibility of a follow-up task (already queued downstream as the training-campaign re-run); this task only ensures the data underneath would honestly justify such a flip.
- sei-network's coverage is bounded by upstream OKX history depth; it cannot be lifted to 305 days from OKX alone. If SEI 5m is ever required to clear the gate, a separate Coinbase or alternative-source backfill is needed. Out of scope here.

## Reports

- Gap audit: `artifacts/ml-engine/reports/20260428T211000Z-task581-5m-historical-backfill-gap-audit.md`
- Verdict (this file): `artifacts/ml-engine/reports/20260428T211100Z-task581-5m-historical-backfill-verdict.md`

## Rerun procedure

The one-off workflow was removed (it should not be part of any regular run path). To rerun the historical backfill (idempotent — `ON CONFLICT DO NOTHING` will skip everything already in DB):

```bash
cd artifacts/ml-engine && ../../.pythonlibs/bin/python -u -m scripts.backfill_5m_extend
```

Optional environment overrides:
- `BACKFILL_5M_DAYS=320` — depth in days (default 320)
- `BACKFILL_5M_DEADLINE_SECONDS=7200` — wall-clock ceiling in seconds (default 7200 = 2h)

The script is lock-isolated from the daily top-up and is safe to run while `dataset-refresher` is active.
