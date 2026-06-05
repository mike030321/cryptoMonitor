# Auto failure-analysis — 2026-04-29T18:27:56.746218+00:00

- **Source report generated_at:** `2026-04-29T13:59:26.059494+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "MIN_DIRECTIONAL_ACCURACY_PER_TF": {"1d": 0.53}, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 22 |
| `salvageable_with_better_features_or_labels` | 11 |
| `insufficient_sample` | 7 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | salvageable_with_better_features_or_labels | structurally_noisy_retire | total |
|---|---|---|---|---|
| 1d | 7 | 1 | 2 | 10 |
| 1h | 0 | 3 | 7 | 10 |
| 2h | 0 | 4 | 6 | 10 |
| 6h | 0 | 3 | 7 | 10 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 1549 | 0.376 / 0.392 | 0.286 / 0.219 | 0.151 | 0.564 | 0.032 | — | -0.316 (n=810) | calibration_broken=True (max reliability deviation 0.618 >= 0.1); prediction_… |
| bonk | `insufficient_sample` | 162 | 0.421 / 0.409 | 0.316 / 0.218 | 0.414 | 0.840 | 0.006 | — | -0.466 (n=148) | holdout 162 < MIN_HOLDOUT_ROWS=200 |
| celestia | `insufficient_sample` | 176 | 0.384 / 0.400 | 0.345 / 0.231 | 0.381 | 0.824 | 0.028 | — | 0.172 (n=154) | holdout 176 < MIN_HOLDOUT_ROWS=200 |
| dogwifcoin | `insufficient_sample` | 142 | 0.349 / 0.371 | 0.356 / 0.241 | 0.366 | 0.796 | 0.007 | — | -0.401 (n=121) | holdout 142 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `salvageable_with_better_features_or_labels` | 213 | 0.357 / 0.391 | 0.344 / 0.220 | 0.136 | 0.535 | 0.005 | — | -0.280 (n=150) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| injective-protocol | `insufficient_sample` | 170 | 0.340 / 0.374 | 0.364 / 0.243 | 0.176 | 0.559 | 0.000 | — | -0.468 (n=103) | holdout 170 < MIN_HOLDOUT_ROWS=200 |
| jupiter-exchange-solana | `insufficient_sample` | 157 | 0.344 / 0.356 | 0.359 / 0.230 | 0.287 | 0.688 | 0.025 | — | -0.341 (n=109) | holdout 157 < MIN_HOLDOUT_ROWS=200 |
| pepe | `structurally_noisy_retire` | 212 | 0.370 / 0.409 | 0.336 / 0.223 | 0.269 | 0.684 | 0.019 | — | -0.099 (n=157) | calibration_broken=True (max reliability deviation 0.673 >= 0.1); prediction_… |
| render-token | `insufficient_sample` | 124 | 0.322 / 0.334 | 0.376 / 0.250 | 0.290 | 0.685 | 0.008 | — | -0.382 (n=91) | holdout 124 < MIN_HOLDOUT_ROWS=200 |
| worldcoin-wld | `insufficient_sample` | 196 | 0.331 / 0.370 | 0.369 / 0.241 | 0.418 | 0.837 | 0.000 | — | -0.383 (n=176) | holdout 196 < MIN_HOLDOUT_ROWS=200 |

### 1h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 15432 | 0.365 / 0.395 | 0.247 / 0.216 | 0.079 | 0.474 | 0.000 | — | -0.342 (n=2185) | red gate but signal remaining — calibration_broken=False, prediction_collapse… |
| bonk | `structurally_noisy_retire` | 1715 | 0.388 / 0.401 | 0.323 / 0.216 | 0.192 | 0.587 | 0.006 | — | -0.351 (n=1098) | calibration_broken=True (max reliability deviation 0.528 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 1715 | 0.376 / 0.390 | 0.317 / 0.216 | 0.361 | 0.744 | 0.018 | — | -0.335 (n=1508) | calibration_broken=True (max reliability deviation 0.672 >= 0.1); prediction_… |
| dogwifcoin | `salvageable_with_better_features_or_labels` | 1715 | 0.390 / 0.395 | 0.346 / 0.217 | 0.131 | 0.543 | 0.007 | — | -0.285 (n=1147) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| floki-inu | `structurally_noisy_retire` | 1715 | 0.373 / 0.409 | 0.330 / 0.215 | 0.289 | 0.689 | 0.026 | — | -0.321 (n=1331) | calibration_broken=True (max reliability deviation 0.415 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 1715 | 0.370 / 0.370 | 0.325 / 0.220 | 0.251 | 0.620 | 0.006 | — | -0.300 (n=1171) | calibration_broken=True (max reliability deviation 0.313 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 1715 | 0.375 / 0.401 | 0.304 / 0.220 | 0.173 | 0.556 | 0.033 | — | -0.336 (n=1432) | calibration_broken=True (max reliability deviation 0.422 >= 0.1); prediction_… |
| pepe | `structurally_noisy_retire` | 1715 | 0.353 / 0.374 | 0.329 / 0.219 | 0.230 | 0.626 | 0.037 | — | -0.275 (n=1303) | calibration_broken=True (max reliability deviation 0.607 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 1715 | 0.374 / 0.391 | 0.328 / 0.217 | 0.258 | 0.663 | 0.040 | — | -0.326 (n=1172) | calibration_broken=True (max reliability deviation 0.651 >= 0.1); prediction_… |
| worldcoin-wld | `salvageable_with_better_features_or_labels` | 1715 | 0.369 / 0.402 | 0.338 / 0.217 | 0.136 | 0.558 | 0.020 | — | -0.302 (n=1251) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |

### 2h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 7681 | 0.387 / 0.424 | 0.268 / 0.212 | 0.051 | 0.472 | 0.000 | — | -0.318 (n=3047) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `structurally_noisy_retire` | 854 | 0.390 / 0.448 | 0.318 / 0.210 | 0.417 | 0.842 | 0.004 | — | -0.340 (n=734) | calibration_broken=True (max reliability deviation 0.815 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 854 | 0.394 / 0.406 | 0.346 / 0.214 | 0.337 | 0.766 | 0.006 | — | -0.300 (n=773) | calibration_broken=True (max reliability deviation 0.836 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 854 | 0.398 / 0.421 | 0.341 / 0.214 | 0.301 | 0.746 | 0.006 | — | -0.294 (n=776) | calibration_broken=True (max reliability deviation 0.647 >= 0.1); prediction_… |
| floki-inu | `salvageable_with_better_features_or_labels` | 854 | 0.386 / 0.439 | 0.357 / 0.212 | 0.095 | 0.515 | 0.014 | — | -0.300 (n=584) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| injective-protocol | `structurally_noisy_retire` | 854 | 0.382 / 0.401 | 0.351 / 0.218 | 0.428 | 0.817 | 0.015 | — | -0.322 (n=758) | calibration_broken=True (max reliability deviation 0.572 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 854 | 0.405 / 0.415 | 0.315 / 0.219 | 0.276 | 0.674 | 0.016 | — | -0.303 (n=764) | calibration_broken=True (max reliability deviation 0.476 >= 0.1); prediction_… |
| pepe | `salvageable_with_better_features_or_labels` | 854 | 0.374 / 0.414 | 0.356 / 0.214 | 0.000 | 0.436 | 0.001 | — | -0.291 (n=491) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| render-token | `structurally_noisy_retire` | 854 | 0.387 / 0.402 | 0.342 / 0.217 | 0.310 | 0.723 | 0.007 | — | -0.371 (n=759) | calibration_broken=True (max reliability deviation 0.728 >= 0.1); prediction_… |
| worldcoin-wld | `salvageable_with_better_features_or_labels` | 854 | 0.380 / 0.408 | 0.358 / 0.218 | -0.019 | 0.420 | 0.000 | — | -0.331 (n=555) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |

### 6h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 2513 | 0.415 / 0.442 | 0.307 / 0.209 | 0.177 | 0.617 | 0.021 | — | -0.324 (n=1956) | calibration_broken=True (max reliability deviation 0.318 >= 0.1); prediction_… |
| bonk | `salvageable_with_better_features_or_labels` | 280 | 0.410 / 0.458 | 0.332 / 0.210 | 0.143 | 0.586 | 0.011 | — | 0.053 (n=220) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| celestia | `structurally_noisy_retire` | 280 | 0.394 / 0.426 | 0.349 / 0.219 | 0.400 | 0.832 | 0.032 | — | -0.264 (n=251) | calibration_broken=True (max reliability deviation 0.849 >= 0.1); prediction_… |
| dogwifcoin | `salvageable_with_better_features_or_labels` | 280 | 0.399 / 0.435 | 0.348 / 0.219 | 0.143 | 0.614 | 0.032 | — | -0.173 (n=215) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| floki-inu | `structurally_noisy_retire` | 280 | 0.392 / 0.452 | 0.359 / 0.210 | 0.486 | 0.896 | 0.011 | — | -0.352 (n=259) | calibration_broken=True (max reliability deviation 0.402 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 280 | 0.426 / 0.412 | 0.330 / 0.221 | 0.400 | 0.821 | 0.007 | — | -0.218 (n=239) | calibration_broken=True (max reliability deviation 0.812 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 280 | 0.385 / 0.415 | 0.351 / 0.223 | 0.221 | 0.646 | 0.014 | — | -0.310 (n=236) | calibration_broken=True (max reliability deviation 0.573 >= 0.1); prediction_… |
| pepe | `salvageable_with_better_features_or_labels` | 280 | 0.414 / 0.435 | 0.349 / 0.212 | -0.111 | 0.346 | 0.014 | — | -0.288 (n=174) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| render-token | `structurally_noisy_retire` | 280 | 0.436 / 0.418 | 0.317 / 0.215 | 0.300 | 0.714 | 0.011 | — | -0.411 (n=236) | calibration_broken=True (max reliability deviation 0.325 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 280 | 0.401 / 0.437 | 0.351 / 0.218 | 0.172 | 0.654 | 0.029 | — | -0.168 (n=254) | calibration_broken=True (max reliability deviation 0.311 >= 0.1); prediction_… |
