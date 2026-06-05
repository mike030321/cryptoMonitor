# Task #409 — SEI 5m hard gate: rationale for removing SEI from `DEFAULT_COINS`

**Date:** 2026-04-24
**Decision:** Remove `sei-network` from `DEFAULT_COINS` (the "OR removed with rationale documented" branch of the task's done condition).
**Reversal cost:** one-line edit in `app/training/train.py`; backfilled Coinbase data is left in place.

## Goal

Get long-term 5m history for SEI so it clears the 5m hard gate
(`contiguous_days ≥ 305 AND density ≥ 0.80 AND gap_rate ≤ 0.01 AND synthetic_rows = 0`)
and trains alongside the other 9 monitored coins.

## Venues evaluated

| Venue | SEI symbol | Result | Notes |
|---|---|---|---|
| **OKX free `history-candles`** | `SEI-USDT` | ❌ truncates ~161d back | Returns 0 rows for any `after` cursor older than 2025-11-14. Already the primary source for the other 9 coins. |
| **Binance** | `SEIUSDT` | ❌ geo-blocked | TCP-reachable but every klines request returns HTTP 451 from this Replit container. |
| **Bybit** | `SEIUSDT` | ❌ geo-blocked | Same — HTTP 403 / region-restricted. |
| **Coinbase Exchange** | `SEI-USD` | ⚠️ partial | Serves 5m back to ~2025-06-10 (318-day span), but does not emit zero-volume bars. Has a real ~6h venue outage on 2025-10-25. |
| **Kraken** | `SEIUSD` | ❌ too short | 5m OHLC only goes back ~5 days from the API. Cannot patch the Coinbase outage. |
| **OKX paid history-candles tier** | `SEI-USDT` | not pursued | Out of scope for this task — needs commercial subscription. |

## What we built (and kept)

A working Coinbase 5m fetcher in `scripts/backfill_history.py`:
- `COINBASE_BASE`, `COINBASE_PRODUCTS`, `COINBASE_GRANULARITY`
- `_fetch_coinbase_raw` paginates 290 candles × granularity per request, walks `end` backward, sleeps 0.18s between calls
- `fetch_coinbase_ohlcv` reorders Coinbase's `[time, low, high, open, close, vol]` to canonical `(bucket_start, o, h, l, c, v)`
- `backfill_candles_one(..., source=...)` and CLI `--source {okx,coinbase}` dispatch
- Coinbase is in `real_source_aliases` in `run_full_training_campaign.py` so its rows pass the gate's source check
- A Coinbase pre-pass in `phase2_data_audit` runs `--source coinbase --days 320` for any 5m-skipped coin in `COINBASE_PRODUCTS` before the OKX iterative loop

This infrastructure is left in place for two reasons:
1. It's a sound secondary venue and may be useful for other coins later.
2. Reversing the SEI removal is a one-line change; the data is already there.

After backfill, `price_candles` for `sei-network @ 5m` holds:
- `coinbase` rows: 73,598 (2025-06-10 → 2026-02-22)
- `okx` rows: 17,300 (2026-02-22 → 2026-04-24)
- Combined span: 318 days, density 0.9919, gap_rate 0.0081

## Why the gate still does not clear

`contiguous_days` is the **strictest** field of the gate: the longest run of consecutive 5-minute buckets with no gap at all. Two things break it:

1. **Coinbase thinness gaps (643 of them, mostly 2 buckets = 10 min).** Coinbase Exchange does not emit a candle when zero trades occur in a 5m window. OKX does. So the same asset on OKX is gap-free, on Coinbase is sparse. Bridging these with carry candles (O=H=L=C=prev_close, vol=0) is industry-standard, but there is one larger gap that breaks the analogy.

2. **Real ~6h Coinbase outage on 2025-10-25 15:10→21:10 UTC.** Direct API check: only 14 of the expected 84 bars exist in this window. SEI did trade on other venues during this period; the close went from 0.19719 (15:10) to 0.20002 (21:10) — a real ~1.5% move. Bridging this gap with carry candles would smear that move into the resume bar. No other free venue has SEI 5m for that date (Kraken's history is too short).

So the longest contiguous run we can build from free sources, even with the Coinbase pre-pass, is ~181 days (the post-outage Coinbase block extended forward into the OKX block). 181 < 305.

## Path not taken

| Option | Why rejected |
|---|---|
| Bridge thinness gaps **and** the 6h outage with carry candles | Smears a real ~1.5% price move into one resume bar; conflicts with the spirit of `synthetic_rows = 0` and the "real OKX bars" contract in `app/db.py`. |
| Bridge only thinness gaps | Still capped at ~181d by the 6h outage. |
| Loosen `contiguous_days` to bridge gaps ≤ N buckets | Changes gate semantics for **all 10 coins**, not just SEI — too much blast radius for one coin. |
| OKX paid tier | Requires a paid subscription that's out of scope for this task. |
| Subscribe to Binance/Bybit via VPN/proxy | Out of scope and adds operational complexity. |

## Effect on the campaign

- `DEFAULT_COINS` shrinks from 10 → 9.
- 5m hard gate now passes on all 9 retained coins (after the orchestrator's normal OKX iterative loop runs).
- SEI 1h / 2h / 6h / 1d still have OKX coverage and would be trainable, but those slices are no longer evaluated since SEI is dropped from the monitored set.
- All ML pipelines, `/ml/predict`, registry paths, etc. simply iterate over `DEFAULT_COINS` and need no other change.

## How to re-enable

If OKX's free history grows organically beyond 305 days for SEI, or a deeper-history source is added:
1. Add `"sei-network"` back to `DEFAULT_COINS` in `app/training/train.py`.
2. Run `python -m scripts.backfill_history --coins sei-network --timeframes 5m --source okx --days 320` (or `--source coinbase` if Coinbase is still the deepest source).
3. Re-run the full training campaign — the gate will re-evaluate automatically.
