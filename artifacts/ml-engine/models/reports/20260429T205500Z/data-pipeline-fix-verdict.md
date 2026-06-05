# Data Pipeline Fix — BTC/ETH/SOL Lead Mid-Price Backfill

**Date:** 2026-04-29 ~20:55 UTC
**Scope:** MTTM 6h universe (8 coins)
**Trigger:** User directive "fix data pipeline"

## Root Cause

The `market_signals` table held BTC, ETH and SOL rows from the live poller only, covering
just the last ~7 days (2026-04-22 → 2026-04-29). The Task #586 historical backfill
script (`scripts/backfill_market_signals.py` with `ML_BACKFILL_STREAMS=mid`) had
**never been executed in this environment**, so no rows existed under the
`okx_backfill_mid_v1` source label.

Consequence: the `build_labeled_dataset` lead-lag asof-join in
`app/training/labels.py` (lines 594-595, 1223-1227, 1284) had nothing to align
against for any 6h bar older than ~7 days. The `btc_lead_ret_5m` and
`eth_lead_ret_5m` feature columns therefore came back **100 % NaN** across the
entire ~365-day training window — the LightGBM brain was training on dead inputs
for two of its key cross-asset features.

### Pre-fix evidence

Cache: `models/datasets/6h_20260429T195137Z.parquet` (built 19:51 UTC, 7 coins, 9 772 rows)

| feature             | NaN %    | non-null rows |
|---------------------|----------|---------------|
| `btc_lead_ret_5m`   | 100.00 % | 0 / 9 772     |
| `eth_lead_ret_5m`   | 100.00 % | 0 / 9 772     |

Per-coin: 0 / 1 396 non-null for every coin (bonk, celestia, dogwifcoin,
floki-inu, injective-protocol, jupiter-exchange-solana, render-token).

DB state for BTC/ETH/SOL `market_signals`:

```
btc | okx_swap+gate_swap | 5237 | 2026-04-22 → 2026-04-29
btc | okx_swap           |   24 | 2026-04-22
eth | okx_swap+gate_swap | 5237 | 2026-04-22 → 2026-04-29
eth | okx_swap           |   24 | 2026-04-22
sol | okx_swap+gate_swap | 5237 | 2026-04-22 → 2026-04-29
*** NO `okx_backfill_mid_v1` rows ***
```

## Fix Applied

Re-ran `scripts/backfill_market_signals.py` with `ML_BACKFILL_STREAMS=mid` for
BTC, ETH and SOL across the full 365-day window. The script's idempotent
`DELETE+INSERT` per source-tagged window let the work be split into chunks
without stomping prior results:

| coin | invocations | rows landed | window covered |
|------|-------------|-------------|----------------|
| BTC  | 3 chunks (0-30, 30-90, 90-210, 210-365) | 105 117 | 2025-04-29 → 2026-04-29 |
| ETH  | 3 chunks (0-60, 60-120, 120-275, 275-365) | 105 117 | 2025-04-29 → 2026-04-29 |
| SOL  | 3 chunks (0-60, 60-120, 120-275, 275-365) | 105 118 | 2025-04-29 → 2026-04-29 |

Source label: `okx_backfill_mid_v1` (does not collide with the existing
`okx_swap+gate_swap` live-poller rows nor with the `okx_funding_history_backfill`
funding rows from the prior Task #628 fix).

No code changes — script behaviour is intact, only data was missing.

## Post-fix Evidence

Caches rebuilt for all 8 MTTM coins via `scripts/refresh_cached_datasets.py`
(`ML_REFRESH_TIMEFRAMES=6h`). Two batches of 4 coins each were merged into a
single 8-coin parquet:

`models/datasets/6h_20260429T205500Z.parquet` (8 coins, 11 168 rows)

| feature             | NaN %  | non-null rows  |
|---------------------|--------|----------------|
| `btc_lead_ret_5m`   | 0.00 % | 11 168 / 11 168 |
| `eth_lead_ret_5m`   | 0.00 % | 11 168 / 11 168 |

Per-coin populations (all 100 %):

```
bonk                       100.0% (n=1396)
celestia                   100.0% (n=1396)
dogwifcoin                 100.0% (n=1396)
floki-inu                  100.0% (n=1396)
injective-protocol         100.0% (n=1396)
jupiter-exchange-solana    100.0% (n=1396)
pepe                       100.0% (n=1396)
render-token               100.0% (n=1396)
```

Distribution stats look reasonable (e.g. pepe slice):
- `btc_lead_ret_5m`: mean -0.000495, std 0.151661, min -1.86, max +1.16
- `eth_lead_ret_5m`: mean -0.003089, std 0.231548, min -1.87, max +1.54

Per-month coverage is solid 95-124 bars/month from 2025-05 → 2026-04 — the
entire training window has live signal in both lead features for the first
time.

## Retrain

8-coin / 6h retrain triggered via `POST /ml/admin/retrain` at
1777496151.585559 (2026-04-29 20:55:51 UTC). Status `running:true` at submission
time. The retrain rebuilds its own dataset from the same DB rows used here, so
the produced models are guaranteed to consume the populated lead-lag features.

The downstream verdict (LL / AUC / DA / call-share deltas vs the prior dead-
feature retrain) is left to the next operator review.

## What This Does NOT Fix

The following 6h schema columns remain absent or fully NaN because no public
data source provides them and policy forbids synthetic fills:

- `liquidations_1h_usd`, `bid_ask_spread_bps` (own-coin)
- `btc_liquidations_1h_usd`, `eth_liquidations_1h_usd`, `sol_liquidations_1h_usd`
- `open_interest_z` (needs paid Coinglass / future-pro feed)

These remain on the followups list and do not block the lead-lag fix above.
