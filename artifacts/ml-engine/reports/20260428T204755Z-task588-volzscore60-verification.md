# Task #588 — volZScore60 restoration on 2h snapshot (verification)

## Bug
The lex-newest 2h snapshot before this fix —
`models/datasets/2h_20260428T154707Z.parquet` — had only 61 columns and was
missing `volZScore60`. It also covered 9 coins (no `sei-network`) instead of
the canonical 10. As a result, task #580's feature edge search produced two
`error: 'volZScore60'` cells for the 2h timeframe
(`vol_zscore60_squared`, `ret1_x_volZ60`); see
`reports/20260428T195627Z-task580-feature-edge-verdict.md`.

## Fix
Regenerated the 2h snapshot via the canonical refresh script:

```
ML_REFRESH_TIMEFRAMES=2h python -m scripts.refresh_cached_datasets
```

This wrote `2h_20260428T202925Z.parquet` (62 cols incl. `volZScore60`,
10 coins incl. `sei-network`, 40368 rows) — schema-aligned with the
1h/6h/1d snapshots.

Also updated a stale code comment in
`scripts/feature_edge_search/run_search.py` that pointed at this gap; the
defensive `FEATURE_COLUMNS`-intersection guard is retained as a fallback
for legacy snapshots.

## Verification — schema parity across timeframes

| timeframe | latest snapshot                        | n_cols | volZScore60 |
|-----------|----------------------------------------|-------:|:-----------:|
| 1h        | `1h_20260428T144544Z.parquet`          |     62 | yes         |
| 2h        | `2h_20260428T202925Z.parquet` (new)    |     62 | yes         |
| 6h        | `6h_20260428T133439Z.parquet`          |     62 | yes         |
| 1d        | `1d_20260428T133304Z.parquet`          |     62 | yes         |

`volZScore60` on the new 2h snapshot: count=40368, mean≈0.001, std≈1.007,
range [-1.61, 7.46] — consistent with the rolling z-score the other
timeframes show.

## Verification — task #580 search re-run on 2h (the previously-broken slice)

Re-ran the stage-1 ablation with `TIMEFRAMES = ['2h']` against the new
snapshot. Result: **0 error cells**. All 16 candidates evaluated; the two
that previously errored are now admitted to stage 2:

| candidate              | tf | folds+ DA | folds+ PnL | folds+ both | admitted | error |
|------------------------|----|-----------|------------|-------------|----------|-------|
| vol_zscore60_squared   | 2h | 2/3       | 2/3        | 2/3         | yes      | —     |
| ret1_x_volZ60          | 2h | 3/3       | 2/3        | 2/3         | yes      | —     |

The 1h / 6h / 1d cells already worked in the prior verdict and were not
re-run.

## Done-criteria check

- [x] 2h dataset snapshot includes `volZScore60` computed identically to
      the 1h/6h/1d snapshots (same window, same point-in-time guarantees —
      same builder code path: `build_feature_vectors_for_series`).
- [x] A re-run of task #580's feature search produces no
      `error: 'volZScore60'` cells.
