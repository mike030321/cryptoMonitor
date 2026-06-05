# MTTM 6h — Bug-fix verdict (2026-04-29)

User directive: *"yeah work only on 6h"*, *"fix this bugs"*.
Constraints (unchanged): paper-only, narrow universe, **no synthetic
data, no gate weakening, no friction edits, no flag flips, no
meta-brain authority**.

---

## TL;DR

| Item                                                            | Status |
|-----------------------------------------------------------------|--------|
| Root-cause identified for 6h dataset feature collapse           | ✅ |
| Funding-history backfill re-run; `funding_rate` revived in 6h   | ✅ |
| pepe/6h retrained — calibration improved (LL **3.90 → 1.75**)   | ✅ |
| Regression test added so this loss can't recur silently         | ✅ |
| 7-coin retrain confirmation triggered (running in background)   | ⏳ |
| MTTM 6h gate-pass for any of the 8 coins                        | ❌ |

The honest reading: **the bug is fixed, calibration is materially
better, but 6h still does not pass MTTM gates for pepe** because
8 microstructure features (open interest, liquidations, BTC/ETH/SOL
lead-lag, bid/ask spread) remain 100 % NaN — there is no public
historical API to backfill them, so 6h's *structural* edge ceiling
is what remains.

---

## The bug

`market_signals` had lost the rows that Task #628's funding-history
backfill produced. The slow-loop trainer's asof-join therefore
populated **9 of 38** features with 100 % NaN across the year-long
6h training window:

```
funding_rate                 100 % NaN     ← Task #628 backfill rows missing
open_interest_z              100 % NaN     ← derived from OI snapshots
liquidations_1h_usd          100 % NaN     ← OKX caps history at 24 h
bid_ask_spread_bps           100 % NaN     ← live-only
btc_lead_ret_5m              100 % NaN     ← BTC absent from price_candles
eth_lead_ret_5m              100 % NaN     ← ETH absent from price_candles
btc_liquidations_1h_usd      100 % NaN
eth_liquidations_1h_usd      100 % NaN
sol_liquidations_1h_usd      100 % NaN
```

The 5m / 1h / 2h / 1d slices had `funding_rate` populated (live polls
fill new rows continuously), so the regression was *6h-only* and
silent — training ran, models published, MTTM kept refusing to
trade, and no log line said "your funding column is 0 %".

## The fix

```bash
python3 -m artifacts.ml_engine.scripts.backfill_funding_history
# inserted 3 288 rows across 8 MTTM coins
# source = 'okx_funding_history_backfill'
# span = 2026-01-28 → 2026-04-29 (~91 days, OKX's hard cap)
```

Then refreshed the 6h cache and confirmed:

```
6h_20260429T194127Z.parquet   funding_rate coverage 24 % (was 0 %)
```

The 24 % is OKX's structural ceiling for SWAP funding history (~92
days out of the year-long training window). Per-coin spread is
±2 pp — uniform across the 8 MTTM coins, so no coin was left behind
by a missing OKX SWAP listing in `OKX_SWAP_BASE`.

## The result on pepe/6h (only retrained coin so far)

| Metric        | Before fix | After fix | Baseline | Direction |
|---------------|-----------:|----------:|---------:|-----------|
| AUC           | 0.5299     | 0.5304    | 0.5228   | tiny ↑    |
| log loss      | 3.899      | **1.750** | 1.048    | **major ↓** |
| Brier         | 0.350      | 0.299     | n/a      | ↓         |
| DA            | 0.4190     | 0.4043    | 0.4345   | ↓ (worse than baseline) |
| call_share    | 0.65       | 0.40      | n/a      | model now abstains more |
| funding_rate rank | n/a    | #24 / 38, importance 222 | n/a | feature is alive |

What this says:

1. **Calibration is dramatically better** — log loss more than halved.
   The pre-fix model was wildly overconfident on noise; the post-fix
   model knows when to shrug.
2. **Directional accuracy fell** — the model now *abstains more* (60 %
   "stable" calls vs 35 % before) because it has more honest
   uncertainty signal.
3. **AUC barely moved** — funding alone is not enough lift; the eight
   still-dead features (OI, liquidations, cross-market) are where
   the real 6h edge would live.
4. **Still does not clear the MTTM directional gate** for pepe (need
   DA > baseline; we're 3 pp below).

## Why we are not over-fitting the fix

The 8 coins all need:

* OI history → only via paid Coinglass API (no `COINGLASS_API_KEY` set,
  user has not asked to provision one).
* Liquidations history → OKX exposes only the last 24 h, and even
  the live poller writes empty rows for these MTTM coins.
* Cross-market lead-lag → BTC / ETH / SOL prices are *not* in
  `price_candles` at all (the universe excludes the majors).
* Bid / ask spread → an order-book snapshot, no historical replay.

None of these can be honestly backfilled today, so working on 6h
edge from this side is blocked on either:

* (a) Adding a BTC/ETH/SOL price source to revive the lead-lag pair
  (a separate, documented task — `COINCAP_API_KEY` is available),
  or
* (b) Provisioning a Coinglass API key for OI/liquidations history,
  or
* (c) Accepting that 6h is structurally weaker than the slices that
  have richer features and recording that in the MTTM universe
  decision (the constraint-honoring fallback).

We did **not** weaken any MTTM gate, edit friction, flip flags, or
hand the meta-brain authority. The bug was in the data pipeline,
not in the gate logic, and the fix is in the data pipeline.

## Regression guard

`tests/test_task628_funding_coverage.py` (added this session)
asserts the latest cached `6h_*.parquet` has ≥ 10 % `funding_rate`
coverage across the 8 MTTM coins, and that per-coin coverage spread
stays within 25 pp. If Task #628's backfill rows ever vanish again
the test fails loudly with the exact remediation step (re-run
`scripts.backfill_funding_history`).

The threshold is loose by design — a fresh env that only has the
last week of OKX live-poll funding rows still passes; only a
wholesale 0 % regression fires the alarm.

## What is currently running

Admin-triggered background retrain for the remaining 7 MTTM coins
on the 6h timeframe (started 19:51 UTC). Status surface:
`GET /ml/admin/retrain/status`. Expected wall-clock: ~30–60 min for
the dataset rebuild + ~5 min × 7 coins for training. Results land
under `models/<coin>/6h/<TS>/manifest.json` and feed into the
slow-loop summary the dashboard already shows.

## Files touched this session

* `artifacts/ml-engine/tests/test_task628_funding_coverage.py` — new
  regression guard for the funding-coverage failure mode.
* `artifacts/ml-engine/models/datasets/6h_20260429T194127Z.parquet`
  — fresh 6h cache with funding revived.
* `artifacts/ml-engine/models/pepe/6h/20260429T194608Z/` — pepe
  retrain artifacts (booster, manifest, calibration plots).
* `market_signals` table — 3 288 new rows tagged
  `okx_funding_history_backfill` covering 8 MTTM coins × 91 days.
