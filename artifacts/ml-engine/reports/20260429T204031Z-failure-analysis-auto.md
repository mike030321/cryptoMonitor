# Auto failure-analysis — 2026-04-29T20:40:31.559125+00:00

- **Source report generated_at:** `2026-04-29T17:14:02.101073+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "MIN_DIRECTIONAL_ACCURACY_PER_TF": {"1d": 0.53}, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `insufficient_sample` | 6 |
| `salvageable_with_better_features_or_labels` | 2 |
| `structurally_noisy_retire` | 1 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | salvageable_with_better_features_or_labels | structurally_noisy_retire | total |
|---|---|---|---|---|
| 1d | 6 | 2 | 1 | 9 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 1354 | 0.374 / 0.396 | 0.295 / 0.218 | 0.129 | 0.545 | 0.049 | — | -0.368 (n=886) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `insufficient_sample` | 162 | 0.421 / 0.409 | 0.316 / 0.218 | 0.414 | 0.840 | 0.006 | — | -0.466 (n=148) | holdout 162 < MIN_HOLDOUT_ROWS=200 |
| celestia | `insufficient_sample` | 176 | 0.384 / 0.400 | 0.345 / 0.231 | 0.381 | 0.824 | 0.028 | — | 0.172 (n=154) | holdout 176 < MIN_HOLDOUT_ROWS=200 |
| dogwifcoin | `insufficient_sample` | 142 | 0.349 / 0.371 | 0.356 / 0.241 | 0.366 | 0.796 | 0.007 | — | -0.401 (n=121) | holdout 142 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `salvageable_with_better_features_or_labels` | 213 | 0.357 / 0.391 | 0.344 / 0.220 | 0.136 | 0.535 | 0.005 | — | -0.280 (n=150) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| injective-protocol | `insufficient_sample` | 170 | 0.340 / 0.374 | 0.364 / 0.243 | 0.176 | 0.559 | 0.000 | — | -0.468 (n=103) | holdout 170 < MIN_HOLDOUT_ROWS=200 |
| jupiter-exchange-solana | `insufficient_sample` | 157 | 0.344 / 0.356 | 0.359 / 0.230 | 0.287 | 0.688 | 0.025 | — | -0.341 (n=109) | holdout 157 < MIN_HOLDOUT_ROWS=200 |
| pepe | `structurally_noisy_retire` | 212 | 0.370 / 0.409 | 0.336 / 0.223 | 0.269 | 0.684 | 0.019 | — | -0.099 (n=157) | calibration_broken=True (max reliability deviation 0.673 >= 0.1); prediction_… |
| render-token | `insufficient_sample` | 124 | 0.322 / 0.334 | 0.376 / 0.250 | 0.290 | 0.685 | 0.008 | — | -0.382 (n=91) | holdout 124 < MIN_HOLDOUT_ROWS=200 |
