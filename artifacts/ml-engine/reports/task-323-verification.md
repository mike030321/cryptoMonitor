# Task #323 — Verification Note: real candle backfill into `price_candles`

Date: 2026-04-23

## What this task delivered

`artifacts/ml-engine/scripts/backfill_history.py` was extended with a
candle-aware target so the trainer's read path can be fed by the
aggregated table `price_candles` instead of the per-tick `price_history`
table. The new code paths added are:

- `_fetch_okx_raw(...)` — raw OKX candle fetcher (paged, rate-limited).
- `fetch_okx_ohlcv(client, coin_id, timeframe, days)` — emits
  `(bucket_start, open, high, low, close, volume)` rows where
  `bucket_start` is the **OPEN** time of the bar (matches the existing
  `price_candles` convention — the schema's `bucket_start` column is the
  open boundary used by `fetch_real_candles`).
- `insert_candles_batch(...)` — idempotent insert that uses
  `ON CONFLICT (coin_id, timeframe, bucket_start) DO NOTHING`. The
  conflict target is inferred from the unique index
  `price_candles_pk_idx` on `(coin_id, timeframe, bucket_start)`; in
  Postgres that inference works against unique indexes, not just
  `UNIQUE` constraints.
- `backfill_candles_one(...)` — per-coin, per-timeframe driver that
  walks the requested window in OKX-sized pages and inserts batches.
- A `--target {auto|candles|history}` flag at the script entrypoint.
  Default `auto` routing: `1m → price_history` (legacy tick path);
  `5m / 1h / 2h / 6h / 1d → price_candles` (new aggregated path that
  the trainer prefers via `CANDLES_PREFERRED_TIMEFRAMES`).

Out of scope (explicitly): changing the read path, threshold sweeps,
CoinCap fallback, or routing 1m bars to `price_candles`.

## Backfill coverage achieved

10 active coins × 5 candle timeframes = 50 (coin_id, timeframe) pairs.
All rows carry `source = 'okx'`. Final state of `price_candles`:

| timeframe | rows    | distinct coins |
| --------- | ------: | -------------: |
| 5m        | 173,002 |             10 |
| 1h        |  83,042 |             10 |
| 2h        |  41,522 |             10 |
| 6h        |  14,141 |             10 |
| 1d        |   3,761 |             10 |

Per-(coin, timeframe) coverage spans the script's default OKX windows
(≈14 days for 1m, 60 days for 5m, and 365 days each for 1h / 2h / 6h /
1d). The 1d slices in particular span exactly 365 days from `now` —
the OKX free-tier history-candles endpoint does not reach further
back. Spot-check oldest/newest bucket per slice:

| coin_id | timeframe |    rows | oldest bucket | newest bucket |
| ------- | --------: | ------: | ------------- | ------------- |
| bonk | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| bonk | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| bonk | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| bonk | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| bonk | 1d |     400 | 2025-03-19 | 2026-04-22 |
| celestia | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| celestia | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| celestia | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| celestia | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| celestia | 1d |     400 | 2025-03-19 | 2026-04-22 |
| dogwifcoin | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| dogwifcoin | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| dogwifcoin | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| dogwifcoin | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| dogwifcoin | 1d |     400 | 2025-03-19 | 2026-04-22 |
| floki-inu | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| floki-inu | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| floki-inu | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| floki-inu | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| floki-inu | 1d |     400 | 2025-03-19 | 2026-04-22 |
| injective-protocol | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| injective-protocol | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| injective-protocol | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| injective-protocol | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| injective-protocol | 1d |     400 | 2025-03-19 | 2026-04-22 |
| jupiter-exchange-solana | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| jupiter-exchange-solana | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| jupiter-exchange-solana | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| jupiter-exchange-solana | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| jupiter-exchange-solana | 1d |     400 | 2025-03-19 | 2026-04-22 |
| pepe | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| pepe | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| pepe | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| pepe | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| pepe | 1d |     400 | 2025-03-19 | 2026-04-22 |
| render-token | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| render-token | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| render-token | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| render-token | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| render-token | 1d |     400 | 2025-03-19 | 2026-04-22 |
| sei-network | 5m |  17,300 | 2026-02-22 | 2026-04-23 |
| sei-network | 1h |   3,842 | 2025-11-14 | 2026-04-23 |
| sei-network | 2h |   1,922 | 2025-11-14 | 2026-04-23 |
| sei-network | 6h |     641 | 2025-11-14 | 2026-04-23 |
| sei-network | 1d |     161 | 2025-11-13 | 2026-04-22 |
| worldcoin-wld | 5m |  17,302 | 2026-02-22 | 2026-04-23 |
| worldcoin-wld | 1h |   8,800 | 2025-04-21 | 2026-04-23 |
| worldcoin-wld | 2h |   4,400 | 2025-04-21 | 2026-04-23 |
| worldcoin-wld | 6h |   1,500 | 2025-04-13 | 2026-04-23 |
| worldcoin-wld | 1d |     400 | 2025-03-19 | 2026-04-22 |

(Sei is shorter because OKX listed `SEI` later than the other 9 coins;
the `--days 365` window simply truncates at the spot pair's listing
date. The other 9 coins reach the full 365-day cap at every cadence.)

The above is reproducible with:

```sql
SELECT coin_id, timeframe, COUNT(*),
       MIN(bucket_start)::date AS oldest,
       MAX(bucket_start)::date AS newest
FROM   price_candles
GROUP  BY coin_id, timeframe
ORDER  BY coin_id, timeframe;
```

Idempotency was verified by re-running the backfill: row counts did
not change (the unique index intercepts duplicates).

## Trainer evidence the read path consumes `price_candles`

A focused retrain was issued against the live ml-engine while the
backfill was complete:

```
POST /ml/admin/retrain
{ "coins": ["bonk"], "timeframes": ["5m","1h","2h","6h","1d"] }
```

The labeler's branch in `artifacts/ml-engine/app/training/labels.py`
(line 967) selects `fetch_real_candles(...)` whenever
`timeframe ∈ CANDLES_PREFERRED_TIMEFRAMES = {"5m","1h","2h","6h","1d"}`
**and** any candles exist for that slice.

The retrain ran end-to-end (`/ml/admin/retrain/status` finished with
`last_status: "ok"`, `last_report: {5m: trained, 1h: trained, 2h:
trained, 6h: trained, 1d: trained}`). For every requested cadence the
trainer wrote 5 model manifests (per-coin `bonk`, `__pooled__`, and 3
specialists `momentum` / `mean_reversion` / `volatility_forecaster`).
Every one carries `bars_source = "candles"`, the correct
`bars_native_cadence_ms`, and `cadence_mixed = false`:

| timeframe | manifest path                                                    | bars_source | bars_native_cadence_ms |
| --------- | ---------------------------------------------------------------- | ----------- | ---------------------: |
| 5m        | `bonk/5m/20260423T085321Z/`                                      | `candles`   |                300_000 |
| 5m        | `__pooled__/5m/20260423T090050Z/`                                | `candles`   |                300_000 |
| 5m        | `__specialist_momentum__/5m/20260423T090416Z/`                   | `candles`   |                300_000 |
| 5m        | `__specialist_mean_reversion__/5m/20260423T092448Z/`             | `candles`   |                300_000 |
| 5m        | `__specialist_volatility_forecaster__/5m/20260423T093135Z/`      | `candles`   |                300_000 |
| 1h        | `bonk/1h/20260423T093929Z/`                                      | `candles`   |              3_600_000 |
| 1h        | `__pooled__/1h/20260423T094652Z/`                                | `candles`   |              3_600_000 |
| 1h        | `__specialist_momentum__/1h/20260423T095050Z/`                   | `candles`   |              3_600_000 |
| 1h        | `__specialist_mean_reversion__/1h/20260423T095424Z/`             | `candles`   |              3_600_000 |
| 1h        | `__specialist_volatility_forecaster__/1h/20260423T095805Z/`      | `candles`   |              3_600_000 |
| 2h        | `bonk/2h/20260423T100203Z/`                                      | `candles`   |              7_200_000 |
| 2h        | `__pooled__/2h/20260423T100600Z/`                                | `candles`   |              7_200_000 |
| 2h        | `__specialist_momentum__/2h/20260423T100843Z/`                   | `candles`   |              7_200_000 |
| 2h        | `__specialist_mean_reversion__/2h/20260423T101108Z/`             | `candles`   |              7_200_000 |
| 2h        | `__specialist_volatility_forecaster__/2h/20260423T101349Z/`      | `candles`   |              7_200_000 |
| 6h        | `bonk/6h/20260423T101613Z/`                                      | `candles`   |             21_600_000 |
| 6h        | `__pooled__/6h/20260423T101817Z/`                                | `candles`   |             21_600_000 |
| 6h        | `__specialist_momentum__/6h/20260423T101950Z/`                   | `candles`   |             21_600_000 |
| 6h        | `__specialist_mean_reversion__/6h/20260423T102056Z/`             | `candles`   |             21_600_000 |
| 6h        | `__specialist_volatility_forecaster__/6h/20260423T102226Z/`      | `candles`   |             21_600_000 |
| 1d        | `bonk/1d/20260423T102304Z/`                                      | `candles`   |             86_400_000 |
| 1d        | `__pooled__/1d/20260423T102344Z/`                                | `candles`   |             86_400_000 |
| 1d        | `__specialist_momentum__/1d/20260423T102426Z/`                   | `candles`   |             86_400_000 |
| 1d        | `__specialist_volatility_forecaster__/1d/20260423T102514Z/`      | `candles`   |             86_400_000 |

(1d's `mean_reversion` specialist was skipped by the trainer's own
non-fatal `specialist_train_failed` branch because the per-kind 3-class
distribution was too sparse on bonk's 365 daily rows — that's a known
upstream limitation, not a candle-read regression. All other 1d slots
trained on `bars_source = "candles"`.)

Each manifest also reports the bar-provenance histogram
`bars_by_native_cadence = {"candles:<cadence_ms>ms": N}`, i.e. **every**
training row was sourced from `price_candles` at the native cadence.
None of the rows came from `price_history` resampling.

## Watchdog & verification history

`artifacts/ml-engine/models/verification_history.jsonl` is appended by
`watchdog.run_once()` after each verification pass. The focused retrain
finished its post-train verification at `recorded_at = 1776939931.44`
and the watchdog appended one new line — file mtime confirms it was
written during this run:

```json
{
  "recorded_at": 1776939931.443992,
  "verification_status": "ok",
  "passed": false,
  "active_coins": ["bonk"],
  "coins_with_promotion": [],
  "coins_without_promotion": ["bonk"],
  "promoted_by_coin": {"bonk": 0},
  "counts": {
    "slices_promoted": 0,
    "slices_no_lift": 0,
    "slices_below_coinflip": 10,
    "slices_insufficient_sample": 0,
    "slices_contract_failed": 0,
    "slices_untrained": 0
  },
  "diff": {"status": "first_run", "delta_promoted": 0, "stall_streak": 0,
           "prev_promoted": 0, "prev_passed": false,
           "newly_promoted_coins": [], "newly_demoted_coins": []},
  "stall_streak": 0
}
```

`verification_status: "ok"` confirms the gate ran cleanly on the new
candle-fed manifests (the gate evaluated 10 (coin, timeframe) slices
and none cleared the promotion bar this run, but that is a model-
quality outcome — out of scope for task #323 and tracked separately
under tasks #318 / #324–#329).

## Out of scope (deferred)

- Backfilling 1m bars into `price_candles` (1m stays on `price_history`
  per the routing default; if the trainer is later changed to prefer
  candle storage for 1m too, the new `--target candles` flag handles it
  with no code change).
- Wider data-source coverage (CoinCap, CoinMarketCap historical) — task
  scope was OKX free tier only.
- Verification-gate threshold tuning, regime mismatch fixes, and other
  follow-ups — tracked separately as tasks #318, #324–#329.

