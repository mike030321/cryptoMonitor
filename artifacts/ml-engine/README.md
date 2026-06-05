# ML Engine (Phase 2)

Python FastAPI sidecar that owns numeric feature engineering, model training,
and model inference for the crypto monitor. Phase 2 ships real LightGBM
multiclass models trained per (coin, timeframe) with a pooled cross-coin
fallback for coins without enough history.

## Endpoints

- `GET  /ml/health` — liveness check.
- `POST /ml/features` — compute a numeric feature vector from `price_history`
  for a `(coinId, timeframe)` pair. Synthetic rows are filtered at the SQL
  layer.
- `POST /ml/predict` — returns the calibrated 3-class probability vector
  (`probDown`, `probStable`, `probUp`), `confidence`, `expectedReturnPct`,
  `predictionStdPct`, top-5 feature importances, the model `version`, and
  `modelCoinId` (the actual coin or `__pooled__` indicating which registry
  slot served the prediction). Returns **503** when no model is registered
  for the requested timeframe (per-coin AND pooled both missing) — the
  service never fabricates a 50/50 fallback.
- `GET  /ml/report` — HTML training report with per-coin sections, pooled
  fallback, fold-level metrics, baseline-vs-LightGBM lift, and a per-class
  reliability diagram (DOWN/STABLE/UP).
- `POST /ml/admin/reload` — invalidate the model cache. Token-gated by the
  `ML_ADMIN_TOKEN` env var (404 when unset, 401 when wrong).

All routes are mounted at `/ml/*` because the artifact's preview path is `/ml`.

## Run locally

```
pnpm --filter @workspace/ml-engine run dev      # FastAPI server
pnpm --filter @workspace/ml-engine run test     # pytest suite
pnpm --filter @workspace/ml-engine run train    # retrain models
```

## Deployment env vars

The trainer reads the live trading contract (label thresholds, per-coin
overrides) from `shared/trading-frictions.json` so it stays aligned with
the live trader and backtester. The path is resolved in this order:

1. `TRADING_FRICTIONS_PATH` if set — absolute or `~`-expanded path to the
   JSON file.
2. `<repo_root>/shared/trading-frictions.json` derived from the
   `app/training/labels.py` module location (works when the trainer runs
   from a workspace checkout).

**`TRADING_FRICTIONS_PATH` (recommended for any worker that runs outside
the workspace checkout).** Required when the trainer process is launched
from a deployment image, container, or worker that does NOT mount the
monorepo at the expected layout. If the JSON cannot be read, the loader
silently falls back to a hardcoded mirror inside `labels.py` and stamps
`LABEL_THRESHOLDS_FALLBACK_STATUS["used_fallback"] = True` (surfaced via
the metrics scraper) — a worker provisioned without this env var WILL
drift the moment an operator tunes the thresholds in
`trading-frictions.json` and the mirror in `labels.py` is not bumped in
the same commit. Set the variable explicitly so the worker fails loud
on a missing JSON instead of training on stale mirrored values. See
task #357.

Recommended deployment value: an absolute path to the JSON inside the
deployed bundle (e.g. `/app/shared/trading-frictions.json`). The
in-process trainer that runs inside the ML Engine FastAPI service uses
`shared/trading-frictions.json` resolved relative to the artifact's
working directory; the env var is preset in `artifact.toml` so the
service has a known value even if the file moves.

The trainer accepts `--coins` and `--timeframes` arguments:

```
../../.pythonlibs/bin/python -m app.training.train --timeframes 1m 5m 1h
```

Uses the workspace-root `.pythonlibs` virtualenv created by `uv` from the
top-level `pyproject.toml`.

## Model registry layout

```
artifacts/ml-engine/models/
├── datasets/{tf}_{version}.parquet       # labeled feature snapshot per run
├── report.json                           # latest training report
├── __pooled__/{tf}/{version}/            # cross-coin fallback model
│   ├── model.txt                         # LightGBM booster
│   ├── calibrators.joblib                # one isotonic per class
│   └── manifest.json                     # ModelManifest (incl. class_return_means_pct)
├── __pooled__/{tf}/latest                # text file containing the latest version id
└── {coin_id}/{tf}/{version}/             # per-coin models, same structure
```

`/ml/predict` resolves a request by trying `(coin, tf)` first, then falling
back to `(__pooled__, tf)` if no per-coin model exists. Coins with fewer
than `MIN_PER_COIN_ROWS=80` labeled rows in the lookback window are not
trained individually and are served by the pooled model.

## Training pipeline

- **Labels**: 3-class (DOWN/STABLE/UP) using the same per-timeframe
  thresholds as `trading-constants.ts` (locked by parity test).
- **Walk-forward CV**: chronological-only splits, expanding window. Fold
  count is **adaptive** in the range `[2, 5]` — capped at `len(df) // 30`
  so each fold has enough rows. We deliberately allow 2 folds for sparse
  per-coin slices (≥80 rows) instead of skipping training entirely; pooled
  models always have enough data for the full 5 folds.
- **Hyperparameter search**: Optuna TPE sampler, 6 trials / 30s timeout
  per fold + final fit. The test suite shrinks this via `ML_SKIP_OPTUNA=1`
  and `ML_LGB_NUM_BOOST_ROUND` (see `tests/conftest.py`) — production
  runs leave them unset.
- **Calibration**: per-class isotonic regression fit on the last 20% of
  the labeled frame, using predictions from the *same* booster that gets
  saved (no train/serve skew). Calibrated probabilities are renormalized
  to the simplex.
- **Inference math**: `expectedReturnPct = Σ p_k · class_return_means[k]`
  and `predictionStdPct = sqrt(Σ p_k · (means[k] − E)²)`. Both are
  derived from real per-class mean returns persisted in the manifest, not
  synthesized from a heuristic.
- **Persistence**: every run writes the labeled feature frame to
  `models/datasets/{tf}_{version}.parquet` *before* training so runs are
  reproducible.

## Cached training datasets stay fresh automatically (Task #540)

The `<tf>_{version}.parquet` snapshots under `models/datasets/` are
re-generated on a documented cadence by the `dataset-refresher`
workflow (`scripts/scheduled_refresh_loop.py`). It ticks every 30 min
(env: `ML_REFRESH_TICK_SECONDS`) and re-refreshes any timeframe whose
newest snapshot is older than the per-tf cadence:

| Timeframe | Cadence |
|-----------|---------|
| `1d`      | 24 h    |
| `6h`      | 24 h    |
| `2h`      | 6 h     |
| `1h`      | 6 h     |
| `5m`      | 6 h     |
| `1m`      | 6 h     |

Per-tf cadence overrides via `ML_REFRESH_CADENCE_<TF>_HOURS`. The
`dataset-refresher` workflow is part of the parallel `Project`
runButton so it auto-starts with every project boot — the whole point
of #540 is that the cache stays fresh without operator intervention.
Each tick hits the production DB, so anyone running the project
locally on a developer DB should either point `DATABASE_URL` at a dev
copy or set `ML_REFRESH_RUN_ONCE=1` to make the loop exit after one
tick. To force-disable the refresher entirely for a debugging session,
remove the `dataset-refresher` task from the `Project` workflow.

Each tick writes:

* `models/datasets/_freshness_status.json` — per-tf `last_success_at`,
  `last_attempt_at`, `last_error`, `next_due_at`, and the mtime of the
  newest snapshot on disk. Each per-tf `retention` block also carries
  `bytes_on_disk` (post-trim sum of `<tf>_*.parquet` sizes) and the
  top-level status carries `total_bytes_on_disk`, `cache_size_warning`
  (true when the total exceeds 5 GB), and a rolling
  `cache_size_history` list (last 60 samples by default; override with
  `ML_REFRESH_SIZE_HISTORY_LEN`) so the freshness dashboard can render
  per-tf cache sizes plus a sparkline of growth over time (Task #559).
  Inspect this file to confirm the refresher is healthy. The same
  payload is served by `GET /ml/dataset-freshness` (proxied to the
  dashboard as `/api/crypto/dataset-freshness`).
* `models/datasets/_freshness_alerts.jsonl` — append-only log of every
  failed refresh (DB unreachable, schema drift, OOM, etc.). The
  scheduler also emits a loud `[ALERT]` line on stderr so the failure
  shows up in the workflow log.

The retrain harness now reads the **freshest** snapshot by mtime
(`_latest_pooled_dataset` in `scripts/diagnostic_482/run_stage_collapse_diagnostic.py`),
not the biggest one by size. `scripts/retrain_task524.py` passes
`max_age_hours=36` (env: `ML_TASK524_MAX_AGE_HOURS`) so a stale cache
fails the retrain loud rather than silently producing a model trained
on week-old data. `ML_DATASET_MAX_AGE_HOURS=0` is the operator escape
hatch for one-off reruns against a deliberately-pinned snapshot.

## Training contract

See [`TRAINING_CONTRACT.md`](TRAINING_CONTRACT.md) for the full rules. Short
version: real `price_history` rows only (synthetic guard double-checks the
SQL filter), point-in-time features verified by `audit_leakage`,
`walk_forward_splits` is the only validator (CI test rejects
`train_test_split`/`KFold`/`shuffle=True`), targets include
`net_pnl_after_costs_pct` and `realized_vol_next_horizon`, and retrain runs
in a slow loop (`run_training`, full base + meta) and a fast loop
(`run_meta_only`, meta-only on every `ML_FAST_LOOP_MIN_NEW_ROWS` new
resolved meta rows).

## Phase 1 → 2 migration notes

The Phase 1 stub `predict` (`{probUp: 0.5, ...}`) has been removed. Any
caller that previously depended on the stub will now receive a real
prediction (when a model exists) or a 503 (when it doesn't). The Node
ml-client typings already reflect the new response shape.
