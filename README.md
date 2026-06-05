# Crypto AI Monitor

This monorepo hosts the Crypto AI Monitor stack: the React dashboard
(`artifacts/crypto-monitor`), the API server (`artifacts/api-server`), and the
Python ML engine (`artifacts/ml-engine`).

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull
request:

- **api-server tests** — `pnpm --filter @workspace/api-server exec node --import tsx --test test/*.test.ts`
- **ml-engine tests** — `pnpm --filter @workspace/ml-engine exec pytest`

A failing job blocks the PR from being merged once the matching jobs are
marked as required checks in the repository's branch-protection settings
(`Settings → Branches → Branch protection rules`). Both `api-server tests
(Node)` and `ml-engine tests (pytest)` should be added to the required-checks
list for the default branch.

## Load-bearing data-integrity gates (task #343)

Three tests added by task #343 are the project's hard contracts for price-data
cadence correctness and per-coin retrain isolation. **Do not skip, mark as
expected-fail, or weaken these without an explicit RFC** — a regression in any
of them silently corrupts training data or pollutes other coins' models:

1. **`artifacts/api-server/test/price-candles-uniqueness.test.ts`** — pins the
   `price_candles` schema contract: a separate cadence-aware bar table keyed by
   `(coin_id, timeframe, bucket_start, source)`, so daily/hourly/live-poll bars
   cannot collide inside `price_history`.
2. **`artifacts/ml-engine/tests/test_cadence_correctness.py`** — pins that the
   trainer's resampler reads bars from `price_candles` at the requested
   timeframe and never silently mixes cadences. (This was originally drafted
   as `cadence-correctness.test.ts` in the task but lives on the Python side
   where the resampler does.)
3. **`artifacts/ml-engine/tests/test_per_coin_retrain_isolation.py`** — proves
   that `train_one_slice(coin, tf, …)` only writes inside its own
   `<coin>/<timeframe>/` directory: it must not rewrite the pooled fallback,
   another coin's model, the same coin's other timeframes, the Phase-3
   specialists, the Phase-4 meta model, or other slices' records in
   `report.json`.

These three are also wired as Replit validation workflows
(`cadence-tests`, `per-coin-isolation`) so they can be run on demand from the
workspace, in addition to running on every push/PR through GitHub Actions.
