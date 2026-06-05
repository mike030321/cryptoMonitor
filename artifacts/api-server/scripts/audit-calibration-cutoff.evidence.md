# Calibration cutoff audit evidence (Task #79)

Run on the live database immediately after the `PREDICTION_FLEET_RESET_AT`
filter was added to `confidence-calibrator.ts`. See
`artifacts/api-server/scripts/audit-calibration-cutoff.sql` for the query
source. Re-run any time you suspect pre-reset rows have leaked back in.

## Top-level counts

```
total_predictions                = 447
pre_reset_rows                   = 0
post_reset_rows                  = 447
pre_reset_resolved_rows          = 0
pre_reset_with_model_attribution = 0
earliest_row                     = 2026-04-21 08:27:14.529777+00
latest_row                       = 2026-04-21 08:50:03.463436+00
```

The earliest live row (`2026-04-21 08:27:14Z`) is **after** the cutoff
constant (`2026-04-21T00:00:00Z`), so the calibrator's
`gte(predictions.created_at, PREDICTION_FLEET_RESET_AT)` filter is
correctly admitting all live data while excluding any pre-reset row that
might appear later (e.g. via backup restore).

## Per-confidence-bucket breakdown (resolved predictions only)

```
confidence_bucket | bucket_total | pre_reset_in_bucket
------------------+--------------+--------------------
                2 |            9 |                  0
                3 |            1 |                  0
```

Every confidence bucket that currently has resolved data contains zero
pre-reset prediction IDs. As more predictions resolve and additional
buckets fill in, this audit should continue to report `0` in every
`pre_reset_in_bucket` column.

## Cutoff selection

`PREDICTION_FLEET_RESET_AT = 2026-04-21T00:00:00Z` was chosen to sit
strictly between any pre-reset rows (none currently in the table; all
were purged before this task) and the earliest known good row
(`2026-04-21T08:27:14Z`). This gives a ~8 hour safety margin so that
slight clock drift around the reset moment cannot accidentally re-admit
a polluted row.
