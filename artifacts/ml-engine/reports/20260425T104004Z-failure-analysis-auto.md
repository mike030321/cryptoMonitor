# Auto failure-analysis — 2026-04-25T10:40:04.801989+00:00

- **Source report generated_at:** `2026-04-25T07:24:14.751707+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "MIN_DIRECTIONAL_ACCURACY_PER_TF": {"1d": 0.53}, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 23 |
| `salvageable_with_better_features_or_labels` | 10 |
| `insufficient_sample` | 7 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | salvageable_with_better_features_or_labels | structurally_noisy_retire | total |
|---|---|---|---|---|
| 1d | 7 | 1 | 2 | 10 |
| 1h | 0 | 4 | 6 | 10 |
| 2h | 0 | 2 | 8 | 10 |
| 6h | 0 | 3 | 7 | 10 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 1543 | 0.374 / 0.391 | 0.301 / 0.218 | 0.128 | 0.542 | 0.047 | — | -0.259 (n=936) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `insufficient_sample` | 161 | 0.431 / 0.393 | 0.315 / 0.218 | 0.410 | 0.845 | 0.000 | — | -0.756 (n=153) | holdout 161 < MIN_HOLDOUT_ROWS=200 |
| celestia | `insufficient_sample` | 175 | 0.385 / 0.404 | 0.351 / 0.231 | 0.051 | 0.503 | 0.006 | — | 0.069 (n=158) | holdout 175 < MIN_HOLDOUT_ROWS=200 |
| dogwifcoin | `insufficient_sample` | 142 | 0.361 / 0.349 | 0.366 / 0.234 | 0.437 | 0.866 | 0.000 | — | -0.208 (n=130) | holdout 142 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `structurally_noisy_retire` | 213 | 0.365 / 0.375 | 0.356 / 0.220 | 0.183 | 0.587 | 0.000 | — | -0.736 (n=163) | calibration_broken=True (max reliability deviation 0.851 >= 0.1); prediction_… |
| injective-protocol | `insufficient_sample` | 169 | 0.349 / 0.364 | 0.366 / 0.246 | 0.308 | 0.686 | 0.000 | — | -0.412 (n=145) | holdout 169 < MIN_HOLDOUT_ROWS=200 |
| jupiter-exchange-solana | `insufficient_sample` | 157 | 0.349 / 0.368 | 0.381 / 0.230 | 0.280 | 0.681 | 0.000 | — | -0.432 (n=118) | holdout 157 < MIN_HOLDOUT_ROWS=200 |
| pepe | `structurally_noisy_retire` | 212 | 0.377 / 0.425 | 0.351 / 0.222 | 0.269 | 0.689 | 0.005 | — | -0.346 (n=171) | calibration_broken=True (max reliability deviation 0.597 >= 0.1); prediction_… |
| render-token | `insufficient_sample` | 123 | 0.341 / 0.363 | 0.382 / 0.250 | 0.366 | 0.764 | 0.016 | — | -0.596 (n=101) | holdout 123 < MIN_HOLDOUT_ROWS=200 |
| worldcoin-wld | `insufficient_sample` | 195 | 0.349 / 0.386 | 0.363 / 0.241 | 0.426 | 0.851 | 0.000 | — | -0.438 (n=179) | holdout 195 < MIN_HOLDOUT_ROWS=200 |

### 1h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 15615 | 0.364 / 0.398 | 0.252 / 0.216 | 0.104 | 0.502 | 0.000 | — | -0.322 (n=1775) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `salvageable_with_better_features_or_labels` | 1735 | 0.384 / 0.404 | 0.325 / 0.216 | 0.137 | 0.534 | 0.010 | — | -0.309 (n=1241) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| celestia | `salvageable_with_better_features_or_labels` | 1735 | 0.385 / 0.395 | 0.335 / 0.216 | 0.044 | 0.432 | 0.018 | — | -0.313 (n=1315) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| dogwifcoin | `structurally_noisy_retire` | 1735 | 0.392 / 0.395 | 0.338 / 0.217 | 0.237 | 0.653 | 0.013 | — | -0.315 (n=1240) | calibration_broken=True (max reliability deviation 0.400 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 1735 | 0.380 / 0.411 | 0.347 / 0.215 | 0.162 | 0.565 | 0.017 | — | -0.310 (n=1307) | calibration_broken=True (max reliability deviation 0.343 >= 0.1); prediction_… |
| injective-protocol | `salvageable_with_better_features_or_labels` | 1735 | 0.375 / 0.370 | 0.344 / 0.220 | 0.146 | 0.518 | 0.005 | — | -0.323 (n=1061) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 1735 | 0.371 / 0.399 | 0.334 / 0.220 | 0.230 | 0.617 | 0.023 | — | -0.327 (n=1191) | calibration_broken=True (max reliability deviation 0.618 >= 0.1); prediction_… |
| pepe | `structurally_noisy_retire` | 1735 | 0.350 / 0.374 | 0.374 / 0.219 | 0.390 | 0.789 | 0.032 | — | -0.273 (n=1466) | calibration_broken=True (max reliability deviation 0.419 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 1735 | 0.365 / 0.387 | 0.317 / 0.218 | 0.212 | 0.613 | 0.034 | — | -0.295 (n=1196) | calibration_broken=True (max reliability deviation 0.410 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 1735 | 0.366 / 0.400 | 0.326 / 0.218 | 0.261 | 0.684 | 0.008 | — | -0.295 (n=1213) | calibration_broken=True (max reliability deviation 0.216 >= 0.1); prediction_… |

### 2h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 7773 | 0.386 / 0.423 | 0.262 / 0.213 | 0.023 | 0.445 | 0.000 | — | -0.316 (n=2561) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `structurally_noisy_retire` | 864 | 0.408 / 0.447 | 0.337 / 0.210 | 0.211 | 0.638 | 0.036 | — | -0.329 (n=748) | calibration_broken=True (max reliability deviation 0.564 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 864 | 0.394 / 0.404 | 0.328 / 0.214 | 0.233 | 0.664 | 0.005 | — | -0.315 (n=801) | calibration_broken=True (max reliability deviation 0.487 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 864 | 0.408 / 0.426 | 0.351 / 0.213 | 0.220 | 0.667 | 0.006 | — | -0.357 (n=608) | calibration_broken=True (max reliability deviation 0.286 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 864 | 0.392 / 0.441 | 0.357 / 0.213 | 0.241 | 0.660 | 0.028 | — | -0.275 (n=731) | calibration_broken=True (max reliability deviation 0.830 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 864 | 0.391 / 0.403 | 0.349 / 0.217 | 0.449 | 0.839 | 0.009 | — | -0.275 (n=788) | calibration_broken=True (max reliability deviation 0.847 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 864 | 0.408 / 0.414 | 0.321 / 0.219 | 0.222 | 0.624 | 0.010 | — | -0.270 (n=780) | calibration_broken=True (max reliability deviation 0.664 >= 0.1); prediction_… |
| pepe | `structurally_noisy_retire` | 864 | 0.381 / 0.414 | 0.350 / 0.214 | 0.230 | 0.665 | 0.008 | — | -0.300 (n=704) | calibration_broken=True (max reliability deviation 0.400 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 864 | 0.375 / 0.410 | 0.340 / 0.217 | 0.321 | 0.732 | 0.024 | — | -0.370 (n=725) | calibration_broken=True (max reliability deviation 0.605 >= 0.1); prediction_… |
| worldcoin-wld | `salvageable_with_better_features_or_labels` | 864 | 0.392 / 0.409 | 0.346 / 0.217 | 0.149 | 0.589 | 0.007 | — | -0.305 (n=737) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |

### 6h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 2544 | 0.420 / 0.434 | 0.299 / 0.210 | 0.134 | 0.571 | 0.002 | — | -0.268 (n=1640) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `structurally_noisy_retire` | 283 | 0.417 / 0.453 | 0.339 / 0.212 | 0.357 | 0.795 | 0.000 | — | -0.099 (n=261) | calibration_broken=True (max reliability deviation 0.404 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 283 | 0.420 / 0.416 | 0.354 / 0.222 | 0.534 | 0.965 | 0.004 | — | -0.434 (n=281) | calibration_broken=True (max reliability deviation 0.493 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 283 | 0.423 / 0.443 | 0.353 / 0.221 | 0.159 | 0.632 | 0.018 | — | -0.528 (n=220) | calibration_broken=True (max reliability deviation 0.695 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 283 | 0.395 / 0.449 | 0.366 / 0.210 | 0.198 | 0.611 | 0.007 | — | -0.340 (n=260) | calibration_broken=True (max reliability deviation 0.468 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 283 | 0.432 / 0.412 | 0.342 / 0.226 | 0.544 | 0.968 | 0.000 | — | -0.372 (n=278) | calibration_broken=True (max reliability deviation 0.499 >= 0.1); prediction_… |
| jupiter-exchange-solana | `salvageable_with_better_features_or_labels` | 283 | 0.404 / 0.421 | 0.358 / 0.221 | 0.053 | 0.481 | 0.014 | — | -0.315 (n=246) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| pepe | `salvageable_with_better_features_or_labels` | 283 | 0.426 / 0.437 | 0.348 / 0.212 | -0.110 | 0.346 | 0.011 | — | -0.116 (n=178) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| render-token | `structurally_noisy_retire` | 283 | 0.441 / 0.409 | 0.328 / 0.217 | 0.399 | 0.813 | 0.014 | — | -0.393 (n=269) | calibration_broken=True (max reliability deviation 0.608 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 283 | 0.441 / 0.434 | 0.343 / 0.219 | 0.286 | 0.767 | 0.000 | — | -0.094 (n=274) | calibration_broken=True (max reliability deviation 0.447 >= 0.1); prediction_… |
