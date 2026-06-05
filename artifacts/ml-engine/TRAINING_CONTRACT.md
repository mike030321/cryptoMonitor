# Quant Training Contract

This document is the source of truth for what the ml-engine trainer is and is
not allowed to do. Any change that violates a rule below MUST update this
document and the matching enforcement test in `tests/test_real_data_contract.py`
in the same commit.

## 1. Real data only

- Every training row originates from `price_history` rows where
  `is_synthetic IS NULL OR is_synthetic = false`. The SQL filter lives in
  `app/db.py::fetch_real_ticks` and is double-checked by
  `fetch_real_ticks_with_provenance`, which also returns the count of
  synthetic rows that fell inside the lookback window.
- If any synthetic row would have been included, the trainer rejects the
  affected (coin, timeframe) slice and stamps `report.json` with
  `provenance.rejected_synthetic = true` instead of silently training.
- Synthetic seed scripts (`scripts/backfill_history.py`) write rows with
  `is_synthetic=false` only when the operator explicitly opts in; the
  default path is `is_synthetic=true` and is invisible to the trainer.

## 2. Point-in-time features

- For a candle at index `i` every feature must be computable from
  `closes[: i + 1]` (and exogenous columns whose timestamp is `<= ts[i]`).
  No future-bar information is allowed in any feature column.
- The leakage audit in `labels.py::audit_leakage` runs once per slice
  inside `run_training` BEFORE any LightGBM cycles and enforces three
  independent layers — any one of which can reject the slice:
  1. **Schema**: no feature shares a name with a target,
     `forward_horizon_candles` matches the declared horizon, the frame
     `timestamp_ms` is monotonic non-decreasing, and every declared
     target column is present.
  2. **Lineage gate**: every feature column MUST be present in
     `registry.FEATURE_LINEAGE` with `max_lookforward == 0`. An
     unregistered feature fails the audit so a developer cannot sneak
     in a new column without declaring its provenance.
  3. **Numerical future-leak detection**: for each numeric feature the
     audit computes the Pearson correlation against
     `lastPrice.shift(-k)` for `k = 1..expected_horizon`; any column
     with `|corr| >= 0.99` is flagged as a probable leak. This catches
     a registered column whose declared lineage is `max_lookforward=0`
     but whose actual values track the future close (e.g. a renamed
     `forward_return`).
- Trainer report stamps `leakage_audit` with `passed`, `violations`,
  `lineage_unregistered`, and `future_corr_hits` so operators can see
  exactly which layer fired.

## 3. Walk-forward validation only

- The single allowed validator path is
  `app/training/walk_forward.py::walk_forward_splits`, an expanding-window
  splitter that yields strictly chronological folds.
- `tests/test_real_data_contract.py::test_no_random_split_imports`
  greps every module under `app/training/` and `app/backtest/` for
  banned splitter names (`train_test_split`, `KFold`, `StratifiedKFold`,
  `ShuffleSplit`, `shuffle=True`). A new entrypoint that imports any of
  them fails CI.

## 4. Cost-aware targets

The labeled frame carries the existing trade-aware label set
(`tp_before_sl_long`, `tp_before_sl_short`, `mae_pct_long`, `mfe_pct_long`,
`opportunity_score`, `prob_move_gt_cost`, `forward_window_return_pct`)
plus two contract-locked additions:

- `realized_vol_next_horizon` — standard deviation of forward per-bar
  returns over `forward_horizon_candles`, in percent. Captures
  next-window volatility for sizing/abstain heads.
- `net_pnl_after_costs_pct` — `forward_window_return_pct` minus the
  shared `round_trip_cost_pct`, signed by the move's direction. Equal
  to `0` when `|forward_window_return_pct| < round_trip_cost_pct` so a
  trade that doesn't clear cost contributes no PnL.

## 5. Input streams (graceful when missing)

The trainer registers null-safe columns for every external stream the
contract cares about. When a provider isn't wired up yet the column is
populated with the default (zero) and the per-feature coverage in the
training report is `0%` so operators can see take-up at a glance:

| Column                   | Source (when wired)                             |
| ------------------------ | ----------------------------------------------- |
| `funding_rate`           | perp funding-rate provider                      |
| `open_interest_z`        | OI provider, z-scored over rolling window       |
| `liquidations_1h_usd`    | liquidations feed                               |
| `bid_ask_spread_bps`     | top-of-book ticker                              |
| `btc_lead_ret_5m`        | BTC 5m return shifted into the coin frame       |
| `eth_lead_ret_5m`        | ETH 5m return shifted into the coin frame       |
| `btc_liquidations_1h_usd`| BTC 1h liquidations (cross-market regime cue)   |
| `eth_liquidations_1h_usd`| ETH 1h liquidations (cross-market regime cue)   |
| `sol_liquidations_1h_usd`| SOL 1h liquidations (cross-market regime cue)   |
| `session_asia`           | derived from `timestamp_ms` (always populated)  |
| `session_eu`             | derived from `timestamp_ms` (always populated)  |
| `session_us`             | derived from `timestamp_ms` (always populated)  |
| `hour_of_day_sin/cos`    | derived from `timestamp_ms` (always populated)  |

Adding a real provider for any of the above is a strict superset change:
the column exists and is null-safe; new code only needs to populate it
before the labeled frame is built.

## 6. Two retrain loops

- **Slow loop** — full base-model retrain over the rolling lookback
  window. Cadence (default 30 min) plus event triggers from
  `_should_run_slow_loop()`: drift detected, regime shift, or a
  `new-data threshold` reached. Entry point:
  `train.run_training(coin_ids, timeframes)`.
- **Fast loop** — meta-model retrain over recent resolved
  prediction/trade outcomes. Cadence: every `ML_FAST_LOOP_MIN_NEW_ROWS`
  new resolved meta rows (default `100`, target ≈ hourly). Entry point:
  `train.run_meta_only(timeframes)`. The fast loop NEVER refits the
  base classifier or regressor; it only refreshes the meta head.

The two loops are mutually exclusive on a given tick: if
`should_run_slow_loop()` returns true the slow loop runs and the fast
loop is skipped (the slow loop already retrains the meta model at the
end). The slow loop writes the canonical `models/report.json`. The
fast loop writes a slim `models/fast_loop_report.json` envelope so it
never overwrites the most recent slow-loop report — operators can read
both files to see when each loop last produced a meta model. The
api-server's ML status endpoint surfaces both timestamps.

Approved features (from Feature Lab) are accepted by the leakage gate
through an in-memory merge: `run_training` extends `FEATURE_LINEAGE`
with `{name: max_lookforward=0}` for every column added by
`apply_approved_features` before calling `audit_leakage`. Approved
features are algebraic transforms of registered base features and so
inherit the point-in-time property; persisting their lineage with the
approval record is a separate hardening task and does not loosen the
contract for unapproved code paths.

## 7. Report transparency

`models/report.json` carries:

- `provenance` — per-coin, per-timeframe `{rows_real, rows_synthetic,
  rejected_synthetic}` so an operator can see whether dirty data leaked.
- `feature_coverage` — `{column: pct_non_null}` for every feature column.
  A coverage of `1.0` means every row carries a non-null value
  (regardless of magnitude); `0.0` means the column is missing entirely.
- `feature_density` — `{column: pct_non_null_and_non_zero}` for the same
  columns. A column with `coverage=1.0, density=0.0` is the canonical
  signature of "registered but unwired": every row defaults to 0.
- `target_row_counts` — `{target: non_null_row_count}` for every
  trade-aware label and the two contract additions.
- `leakage_audit` — `{passed: bool, violations: [...]}` produced by
  `audit_leakage`. A failed audit aborts the slice and the failure
  is recorded here.
- `loop` — which loop produced the run (`slow` or `fast`), the trigger
  reason, and the row count that gated the fast loop.

## 8. Out of scope (kept here so reviewers don't ask)

- Order-book / tick-level execution models.
- Replacing the LightGBM stack.
- Backfilling historical funding/OI beyond what providers already
  surface (the contract handles missing data; ingestion expansion is a
  separate task).
- Any change to `paper-trader.ts` or `/ml/decide` semantics.
