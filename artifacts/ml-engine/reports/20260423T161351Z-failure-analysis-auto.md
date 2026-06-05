# Auto failure-analysis — 2026-04-23T16:13:51.809912+00:00

- **Source report generated_at:** `2026-04-23T15:35:33.912238+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 32 |
| `insufficient_sample` | 11 |
| `salvageable_with_better_features_or_labels` | 1 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | salvageable_with_better_features_or_labels | structurally_noisy_retire | total |
|---|---|---|---|---|
| 1d | 10 | 0 | 1 | 11 |
| 1h | 0 | 0 | 11 | 11 |
| 2h | 0 | 0 | 11 | 11 |
| 6h | 1 | 1 | 9 | 11 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 618 | 0.421 / 0.452 | 0.282 / 0.212 | 0.393 | 0.874 | 0.024 | — | -0.023 (n=596) | calibration_broken=True (max reliability deviation 0.546 >= 0.1); prediction_… |
| bonk | `insufficient_sample` | 66 | 0.371 / 0.476 | 0.278 / 0.225 | 0.364 | 0.818 | 0.151 | — | 0.092 (n=57) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| celestia | `insufficient_sample` | 66 | 0.476 / 0.429 | 0.285 / 0.232 | 0.258 | 0.742 | 0.000 | — | 0.552 (n=53) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| dogwifcoin | `insufficient_sample` | 66 | 0.516 / 0.509 | 0.271 / 0.220 | 0.394 | 0.909 | 0.333 | — | 1.057 (n=43) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `insufficient_sample` | 66 | 0.411 / 0.418 | 0.341 / 0.233 | 0.348 | 0.788 | 0.258 | — | 1.424 (n=46) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| injective-protocol | `insufficient_sample` | 66 | 0.415 / 0.385 | 0.285 / 0.259 | 0.152 | 0.606 | 0.000 | — | 0.835 (n=54) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| jupiter-exchange-solana | `insufficient_sample` | 66 | 0.415 / 0.425 | 0.308 / 0.241 | 0.379 | 0.833 | 0.045 | — | 0.134 (n=50) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| pepe | `insufficient_sample` | 66 | 0.436 / 0.451 | 0.298 / 0.226 | 0.030 | 0.561 | 0.000 | — | 0.767 (n=30) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| render-token | `insufficient_sample` | 66 | 0.433 / 0.433 | 0.264 / 0.232 | 0.333 | 0.742 | 0.000 | — | 0.683 (n=66) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| sei-network | `insufficient_sample` | 26 | 0.289 / 0.408 | 0.297 / 0.224 | 0.308 | 0.846 | 0.000 | — | 1.419 (n=26) | holdout 26 < MIN_HOLDOUT_ROWS=200 |
| worldcoin-wld | `insufficient_sample` | 66 | 0.396 / 0.433 | 0.318 / 0.227 | 0.136 | 0.651 | 0.000 | — | 1.313 (n=65) | holdout 66 < MIN_HOLDOUT_ROWS=200 |

### 1h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 16454 | 0.363 / 0.405 | 0.227 / 0.214 | 0.340 | 0.728 | 0.106 | — | -0.268 (n=3164) | calibration_broken=True (max reliability deviation 0.310 >= 0.1); prediction_… |
| bonk | `structurally_noisy_retire` | 1744 | 0.393 / 0.405 | 0.253 / 0.213 | 0.578 | 0.966 | 0.006 | — | -0.244 (n=132) | calibration_broken=True (max reliability deviation 0.248 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 1744 | 0.378 / 0.398 | 0.257 / 0.214 | 0.526 | 0.915 | 0.137 | — | -0.220 (n=460) | calibration_broken=True (max reliability deviation 0.177 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 1744 | 0.395 / 0.405 | 0.251 / 0.214 | 0.347 | 0.745 | 0.062 | — | -0.274 (n=720) | calibration_broken=True (max reliability deviation 0.202 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 1744 | 0.384 / 0.411 | 0.255 / 0.214 | 0.248 | 0.627 | 0.005 | — | -0.264 (n=187) | calibration_broken=True (max reliability deviation 0.588 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 1744 | 0.373 / 0.397 | 0.258 / 0.215 | 0.243 | 0.627 | 0.000 | — | -0.182 (n=418) | calibration_broken=True (max reliability deviation 0.409 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 1744 | 0.389 / 0.409 | 0.254 / 0.213 | 0.421 | 0.809 | 0.064 | — | -0.133 (n=164) | calibration_broken=True (max reliability deviation 0.902 >= 0.1); prediction_… |
| pepe | `structurally_noisy_retire` | 1744 | 0.373 / 0.398 | 0.264 / 0.216 | 0.425 | 0.812 | 0.060 | — | -0.284 (n=1039) | calibration_broken=True (max reliability deviation 0.609 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 1744 | 0.384 / 0.399 | 0.261 / 0.214 | 0.339 | 0.735 | 0.359 | — | -0.155 (n=178) | calibration_broken=True (max reliability deviation 0.872 >= 0.1); prediction_… |
| sei-network | `structurally_noisy_retire` | 762 | 0.337 / 0.358 | 0.310 / 0.223 | 0.295 | 0.660 | 0.489 | — | -0.269 (n=149) | calibration_broken=True (max reliability deviation 0.341 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 1744 | 0.384 / 0.402 | 0.262 / 0.215 | 0.195 | 0.603 | 0.060 | — | -0.253 (n=1125) | calibration_broken=True (max reliability deviation 0.389 >= 0.1); prediction_… |

### 2h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 8192 | 0.370 / 0.401 | 0.231 / 0.215 | 0.586 | 0.988 | 0.094 | — | -0.278 (n=2819) | calibration_broken=True (max reliability deviation 0.654 >= 0.1); prediction_… |
| bonk | `structurally_noisy_retire` | 869 | 0.391 / 0.414 | 0.265 / 0.214 | 0.496 | 0.893 | 0.004 | — | -0.285 (n=453) | calibration_broken=True (max reliability deviation 0.391 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 869 | 0.372 / 0.401 | 0.279 / 0.216 | 0.445 | 0.846 | 0.007 | — | -0.180 (n=260) | calibration_broken=True (max reliability deviation 0.469 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 869 | 0.389 / 0.403 | 0.287 / 0.215 | 0.540 | 0.967 | 0.090 | — | -0.200 (n=633) | calibration_broken=True (max reliability deviation 0.409 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 869 | 0.387 / 0.409 | 0.277 / 0.215 | 0.475 | 0.857 | 0.075 | — | -0.188 (n=498) | calibration_broken=True (max reliability deviation 0.164 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 869 | 0.393 / 0.397 | 0.280 / 0.216 | 0.556 | 0.945 | 0.008 | — | -0.191 (n=217) | calibration_broken=True (max reliability deviation 0.213 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 869 | 0.398 / 0.408 | 0.266 / 0.215 | 0.161 | 0.569 | 0.066 | — | -0.198 (n=724) | calibration_broken=True (max reliability deviation 0.862 >= 0.1); prediction_… |
| pepe | `structurally_noisy_retire` | 869 | 0.377 / 0.391 | 0.285 / 0.218 | 0.433 | 0.837 | 0.127 | — | -0.279 (n=715) | calibration_broken=True (max reliability deviation 0.306 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 869 | 0.384 / 0.397 | 0.261 / 0.215 | 0.153 | 0.569 | 0.534 | — | -0.179 (n=230) | calibration_broken=True (max reliability deviation 0.277 >= 0.1); prediction_… |
| sei-network | `structurally_noisy_retire` | 378 | 0.332 / 0.367 | 0.294 / 0.227 | 0.257 | 0.630 | 0.455 | — | -0.208 (n=207) | calibration_broken=True (max reliability deviation 0.700 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 869 | 0.386 / 0.395 | 0.264 / 0.218 | 0.346 | 0.742 | 0.071 | — | -0.106 (n=217) | calibration_broken=True (max reliability deviation 0.427 >= 0.1); prediction_… |

### 6h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 2683 | 0.382 / 0.416 | 0.254 / 0.214 | 0.546 | 0.982 | 0.077 | — | -0.268 (n=2483) | calibration_broken=True (max reliability deviation 0.440 >= 0.1); prediction_… |
| bonk | `structurally_noisy_retire` | 285 | 0.387 / 0.424 | 0.309 / 0.213 | 0.523 | 0.937 | 0.768 | — | 0.204 (n=24) | calibration_broken=True (max reliability deviation 0.509 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 285 | 0.376 / 0.384 | 0.304 / 0.222 | 0.572 | 0.993 | 0.232 | — | -0.078 (n=94) | calibration_broken=True (max reliability deviation 0.709 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 285 | 0.396 / 0.420 | 0.314 / 0.216 | 0.365 | 0.810 | 0.144 | — | -0.111 (n=235) | calibration_broken=True (max reliability deviation 0.254 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 285 | 0.386 / 0.439 | 0.304 / 0.214 | 0.488 | 0.919 | 0.775 | — | -0.174 (n=105) | calibration_broken=True (max reliability deviation 0.346 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 285 | 0.381 / 0.401 | 0.290 / 0.223 | 0.565 | 0.979 | 0.000 | — | -0.465 (n=72) | calibration_broken=True (max reliability deviation 0.294 >= 0.1); prediction_… |
| jupiter-exchange-solana | `salvageable_with_better_features_or_labels` | 285 | 0.376 / 0.405 | 0.311 / 0.221 | 0.123 | 0.544 | 0.365 | — | -0.310 (n=70) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| pepe | `structurally_noisy_retire` | 285 | 0.376 / 0.419 | 0.306 / 0.215 | 0.249 | 0.656 | 0.000 | — | -0.111 (n=228) | calibration_broken=True (max reliability deviation 0.490 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 285 | 0.386 / 0.399 | 0.306 / 0.219 | 0.263 | 0.709 | 0.000 | — | 0.050 (n=230) | calibration_broken=True (max reliability deviation 0.270 >= 0.1); prediction_… |
| sei-network | `insufficient_sample` | 122 | 0.386 / 0.406 | 0.307 / 0.222 | 0.533 | 1.000 | 0.000 | — | -0.328 (n=122) | holdout 122 < MIN_HOLDOUT_ROWS=200 |
| worldcoin-wld | `structurally_noisy_retire` | 285 | 0.380 / 0.421 | 0.305 / 0.217 | 0.467 | 0.923 | 0.653 | — | -0.023 (n=233) | calibration_broken=True (max reliability deviation 0.150 >= 0.1); prediction_… |
