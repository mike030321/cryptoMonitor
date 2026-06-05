# Task #517 — `volZScore60` margin-improvement report

## Problem statement

After Task #507's tiny-slice booster fix shipped, the four originally-stuck
verification-gate slices all cleared `bucket=='no_collapse'`, but
**dogwifcoin@1d** scraped through with `raw_STABLE_share=0.0597` — only
**0.0097** above the 5% gate floor. Any incremental noise in the next data
snapshot would push it back below the floor and re-block promotion. The
task is to lift that slice's margin to ≥ 0.10 (a healthy 5%-point cushion)
without regressing any of the other 39 (coin × tf) slices.

## Root cause

The four stuck slices were not stuck because of a calibrator quirk — they
were stuck because the **raw booster** could not separate the STABLE
class from the directional classes on the available features. For
dogwifcoin@1d in particular, the existing 22-column price/momentum/EMA/
MACD/Bollinger feature set has near-zero correlation with the STABLE
indicator (verified during exploration), so the booster argmax-collapses
toward the directional classes regardless of how the booster's
hyperparameters or class weights are tuned. The fix has to add a
**new feature** that carries information about whether the next bar is
likely to land in the STABLE band.

## Feature added: `volZScore60`

Defined as the rolling 60-bar z-score of the absolute 1-bar return at
the current bar:

```
abs_ret[k]  = |close[k]/close[k-1] - 1|
window      = abs_ret[max(0, k-59) : k+1]      # last ≤ 60 absolute returns
volZScore60 = (abs_ret[k] - mean(window)) / std(window, ddof=1)
```

with `min_periods=10` (returns 0 when the window is shorter), and a
**0-fallback when `std` ≤ 1e-12** so a flat-price segment never produces
NaN/inf. The window only ever contains absolute returns observed
**at-or-before** the current bar, so `max_lookforward=0` in
`FEATURE_LINEAGE` and the leakage audit accepts it.

### Why this works

- It captures whether the latest move is **anomalous relative to the
  recent volatility regime**, which is exactly the discriminator the
  booster needs for the STABLE class — STABLE bars sit near the centre
  of the local volatility distribution (small `volZScore60`), while
  directional bars sit in the tails (large positive `volZScore60`).
- It is **normalized per-coin and per-timeframe** by construction
  (the rolling window adapts), so the same feature works on bonk@2h
  and dogwifcoin@1d without any coin-specific scaling.
- It is **price-only** (no exogenous data, no LLM, no news), so it
  passes the Task #365 quant-only enforcement contract and does not
  require any new ingestion plumbing.

### Why a single feature, not several

During exploration multi-feature additions (e.g. `rv60 + absRetEma10`,
`bbSqueezeRatio + rangePct20`) repeatedly destabilized the larger
non-stuck slices — for example, bonk@2h dropped from
`raw_STABLE_share=0.264` to `0.000` under one such combination. The
single-feature `volZScore60` solution is the only one that lifted
dogwifcoin@1d above the target margin **without** producing any
regression below the 5% floor on the other slices.

## Margin proof — focused diagnostic (dogwifcoin@1d on Task #507's parquet)

Re-ran `scripts/diagnostic_482/run_507_focused.py` against the same
1d parquet snapshot Task #507 verified on
(`models/datasets/1d_20260423T043348Z.parquet`, `ML_TASK507_1D_DATASET`
override) so the margin lift is a strictly feature-level comparison.
Fixture:
`reports/20260428T114426Z-task507-booster-collapse-rerun-alpha1.0.json`.

| slice           | base (Task #507) | with `volZScore60` | Δ        | margin above 5% floor |
|-----------------|-----------------:|-------------------:|---------:|----------------------:|
| bonk@1d         |          0.1045  |             0.1343 |  +0.0299 |               0.0843  |
| celestia@1d     |          0.2985  |             0.2239 |  −0.0746 |               0.1739  |
| **dogwifcoin@1d** |        0.0597  |           **0.1493** | **+0.0896** |        **0.0993**  |
| celestia@6h     |          0.2133  |             0.1643 |  −0.0490 |               0.1143  |

dogwifcoin@1d's margin grew from **0.0097 → 0.0993**, a **10×** lift,
and its raw_STABLE_share is now **49% above** the new 0.10 task
target. All four originally-stuck slices still clear the 5% gate by a
comfortable cushion; celestia@1d and celestia@6h give back some of
their already-large margin (still ≥ 0.10) but neither approaches the
floor.

## Full-fleet regression check — 38 (coin × tf) slices

Companion artifact:
`reports/20260428T120024Z-task517-volZScore60-full-fleet-regression.json`.
Same harness, same hyperparameters, every available (coin × tf) slice
under the same 4 timeframes. Datasets are listed in the JSON.

**Result: `n_regressions_below_floor = 0`**. No slice that started at
`raw_STABLE_share ≥ 0.05` in BASE drops below 0.05 once `volZScore60`
is added.

Notable per-tf shape:

- **1d** (10 coins, Task #507's parquet): every slice stays above 0.13;
  the largest two give back margin (celestia 0.30→0.22, pepe 0.54→0.40)
  but neither approaches the floor.
- **6h** (10 coins, latest pooled): every slice stays above 0.10; the
  five-figure-margin slices give back 0.05–0.12 but stay deep inside
  the gate.
- **2h** (9 coins, latest pooled): **every slice except one improves**.
  bonk@2h jumps 0.264 → 0.466 (+0.20). The one slice already below the
  floor in BASE — jupiter-exchange-solana@2h at 0.0127 — improves to
  0.0440, still below the floor but **closer to it**. This is not a
  feature-induced regression: BASE was already below the floor.
- **1h** (9 coins, latest pooled): every slice except one improves;
  dogwifcoin@1h itself jumps 0.639 → 0.801 (+0.16). worldcoin-wld@1h
  gives back margin (0.292 → 0.135) but stays well above the floor.

Across all 38 verified slices the median Δ is **+0.0143** with a
strong positive tail.

## Code surface

- `app/features.py` — added `volZScore60` to both
  `build_feature_vector` (per-call, exact reference math) and
  `build_feature_vectors_for_series` (batch, equivalent rolling
  computation in one pass over the close series).
- `app/training/registry.py` — registered `volZScore60` in
  `FEATURE_COLUMNS` (between `bbWidthPct` and the contract-new
  external-stream block, ahead of the `coin_idx` categorical that must
  stay last) and in `FEATURE_LINEAGE` with
  `{max_lookforward: 0, max_lookback: 60}`.
- `scripts/diagnostic_482/run_507_focused.py` — added an in-process
  `_augment_with_missing_feature_columns` that reconstructs
  `volZScore60` from `lastPrice` per-coin so the focused diagnostic can
  replay against parquet snapshots written before the feature existed,
  plus a `ML_TASK507_<TF>_DATASET` env override so Task #507's
  pre-existing 1d snapshot can be pinned for like-for-like comparison.

## Test surface

- `tests/test_features.py` — four new unit tests:
  - `test_vol_zscore_60_matches_manual_rolling_zscore` — locks the
    numeric contract against an independent re-derivation
  - `test_vol_zscore_60_is_zero_when_window_has_no_dispersion` —
    flat-price segment never emits NaN/inf
  - `test_vol_zscore_60_is_zero_when_history_below_min_periods` —
    early-history fallback is finite
  - `test_vol_zscore_60_batch_matches_per_call` — batch builder is
    bit-identical to per-call at every k
- `tests/test_training.py::test_task507_focused_diagnostic_recovers_all_stuck_slices`
  — kept the pre-existing `raw_S ≥ 0.05` floor for all 4 stuck slices
  AND added a `raw_S ≥ 0.10` lock specifically for dogwifcoin@1d so a
  future feature-set refactor cannot quietly regress this slice back
  onto the floor.

## Follow-ups

Filed as project tasks:

- **#524** — Retrain all (coin × tf) slices so production actually serves
  predictions with `volZScore60` (existing trained models on disk were
  trained against the old feature schema and will not see the new
  column at inference until each slice is re-promoted).
- **#525** — Fix the unrelated pre-existing
  `test_build_feature_vector_full_shape_when_enough_data` failure
  (asserts on `news_*` keys removed by Task #365's quant-only
  enforcement).
