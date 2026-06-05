# Auto failure-analysis — 2026-04-25T10:33:07.290340+00:00

- **Source report generated_at:** `2026-04-25T07:07:20.232728+00:00`
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
| 1d | 7 | 2 | 1 | 10 |
| 1h | 0 | 4 | 6 | 10 |
| 2h | 0 | 2 | 8 | 10 |
| 6h | 0 | 2 | 8 | 10 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 1543 | 0.375 / 0.391 | 0.299 / 0.218 | 0.130 | 0.542 | 0.047 | — | -0.258 (n=936) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `insufficient_sample` | 161 | 0.416 / 0.393 | 0.317 / 0.218 | 0.398 | 0.832 | 0.000 | — | -0.319 (n=148) | holdout 161 < MIN_HOLDOUT_ROWS=200 |
| celestia | `insufficient_sample` | 175 | 0.379 / 0.404 | 0.344 / 0.231 | 0.051 | 0.497 | 0.006 | — | -0.173 (n=133) | holdout 175 < MIN_HOLDOUT_ROWS=200 |
| dogwifcoin | `insufficient_sample` | 142 | 0.354 / 0.349 | 0.351 / 0.234 | 0.387 | 0.810 | 0.007 | — | -0.402 (n=122) | holdout 142 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `salvageable_with_better_features_or_labels` | 213 | 0.353 / 0.375 | 0.353 / 0.220 | 0.113 | 0.512 | 0.019 | — | -0.788 (n=142) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| injective-protocol | `insufficient_sample` | 169 | 0.340 / 0.364 | 0.366 / 0.246 | 0.172 | 0.550 | 0.006 | — | -0.568 (n=114) | holdout 169 < MIN_HOLDOUT_ROWS=200 |
| jupiter-exchange-solana | `insufficient_sample` | 157 | 0.349 / 0.368 | 0.360 / 0.230 | 0.223 | 0.624 | 0.013 | — | -0.443 (n=107) | holdout 157 < MIN_HOLDOUT_ROWS=200 |
| pepe | `structurally_noisy_retire` | 212 | 0.374 / 0.425 | 0.336 / 0.222 | 0.259 | 0.679 | 0.005 | — | -0.310 (n=156) | calibration_broken=True (max reliability deviation 0.680 >= 0.1); prediction_… |
| render-token | `insufficient_sample` | 123 | 0.339 / 0.363 | 0.373 / 0.250 | 0.236 | 0.634 | 0.016 | — | -0.114 (n=90) | holdout 123 < MIN_HOLDOUT_ROWS=200 |
| worldcoin-wld | `insufficient_sample` | 195 | 0.349 / 0.386 | 0.361 / 0.241 | 0.400 | 0.826 | 0.000 | — | -0.439 (n=171) | holdout 195 < MIN_HOLDOUT_ROWS=200 |

### 1h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 15615 | 0.364 / 0.398 | 0.252 / 0.216 | 0.104 | 0.502 | 0.000 | — | -0.322 (n=1775) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `salvageable_with_better_features_or_labels` | 1735 | 0.383 / 0.404 | 0.314 / 0.216 | 0.137 | 0.534 | 0.010 | — | -0.309 (n=1241) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| celestia | `salvageable_with_better_features_or_labels` | 1735 | 0.384 / 0.395 | 0.327 / 0.216 | 0.044 | 0.432 | 0.018 | — | -0.313 (n=1315) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| dogwifcoin | `structurally_noisy_retire` | 1735 | 0.390 / 0.395 | 0.331 / 0.217 | 0.237 | 0.653 | 0.013 | — | -0.315 (n=1240) | calibration_broken=True (max reliability deviation 0.400 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 1735 | 0.377 / 0.411 | 0.340 / 0.215 | 0.162 | 0.565 | 0.017 | — | -0.310 (n=1307) | calibration_broken=True (max reliability deviation 0.343 >= 0.1); prediction_… |
| injective-protocol | `salvageable_with_better_features_or_labels` | 1735 | 0.374 / 0.370 | 0.346 / 0.220 | 0.146 | 0.518 | 0.005 | — | -0.323 (n=1061) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 1735 | 0.369 / 0.399 | 0.331 / 0.220 | 0.230 | 0.617 | 0.023 | — | -0.327 (n=1191) | calibration_broken=True (max reliability deviation 0.618 >= 0.1); prediction_… |
| pepe | `structurally_noisy_retire` | 1735 | 0.352 / 0.374 | 0.362 / 0.219 | 0.390 | 0.789 | 0.032 | — | -0.273 (n=1466) | calibration_broken=True (max reliability deviation 0.419 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 1735 | 0.361 / 0.387 | 0.307 / 0.218 | 0.212 | 0.613 | 0.034 | — | -0.295 (n=1196) | calibration_broken=True (max reliability deviation 0.410 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 1735 | 0.366 / 0.400 | 0.313 / 0.218 | 0.261 | 0.684 | 0.008 | — | -0.295 (n=1213) | calibration_broken=True (max reliability deviation 0.216 >= 0.1); prediction_… |

### 2h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 7775 | 0.384 / 0.422 | 0.272 / 0.213 | 0.068 | 0.490 | 0.001 | — | -0.327 (n=3726) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `structurally_noisy_retire` | 864 | 0.401 / 0.446 | 0.335 / 0.210 | 0.228 | 0.655 | 0.020 | — | -0.322 (n=750) | calibration_broken=True (max reliability deviation 0.623 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 864 | 0.395 / 0.407 | 0.346 / 0.214 | 0.177 | 0.609 | 0.020 | — | -0.301 (n=750) | calibration_broken=True (max reliability deviation 0.832 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 864 | 0.410 / 0.426 | 0.340 / 0.213 | 0.188 | 0.634 | 0.029 | — | -0.296 (n=730) | calibration_broken=True (max reliability deviation 0.308 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 864 | 0.386 / 0.440 | 0.350 / 0.212 | 0.260 | 0.679 | 0.004 | — | -0.308 (n=787) | calibration_broken=True (max reliability deviation 0.687 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 864 | 0.392 / 0.403 | 0.352 / 0.217 | 0.419 | 0.809 | 0.014 | — | -0.354 (n=773) | calibration_broken=True (max reliability deviation 0.813 >= 0.1); prediction_… |
| jupiter-exchange-solana | `salvageable_with_better_features_or_labels` | 864 | 0.405 / 0.413 | 0.320 / 0.219 | 0.147 | 0.549 | 0.057 | — | -0.251 (n=712) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| pepe | `structurally_noisy_retire` | 864 | 0.381 / 0.414 | 0.348 / 0.214 | 0.164 | 0.600 | 0.021 | — | -0.324 (n=715) | calibration_broken=True (max reliability deviation 0.391 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 864 | 0.377 / 0.407 | 0.346 / 0.217 | 0.309 | 0.720 | 0.025 | — | -0.350 (n=710) | calibration_broken=True (max reliability deviation 0.623 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 864 | 0.402 / 0.409 | 0.344 / 0.217 | 0.225 | 0.664 | 0.007 | — | -0.282 (n=786) | calibration_broken=True (max reliability deviation 0.510 >= 0.1); prediction_… |

### 6h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 2546 | 0.418 / 0.433 | 0.287 / 0.211 | 0.138 | 0.575 | 0.000 | — | -0.300 (n=1385) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `structurally_noisy_retire` | 283 | 0.417 / 0.452 | 0.329 / 0.211 | 0.392 | 0.830 | 0.042 | — | -0.078 (n=257) | calibration_broken=True (max reliability deviation 0.346 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 283 | 0.414 / 0.418 | 0.348 / 0.222 | 0.523 | 0.954 | 0.011 | — | -0.395 (n=275) | calibration_broken=True (max reliability deviation 0.634 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 283 | 0.409 / 0.445 | 0.348 / 0.221 | 0.198 | 0.671 | 0.007 | — | -0.238 (n=205) | calibration_broken=True (max reliability deviation 0.802 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 283 | 0.412 / 0.452 | 0.353 / 0.210 | 0.467 | 0.880 | 0.028 | — | -0.286 (n=255) | calibration_broken=True (max reliability deviation 0.420 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 283 | 0.415 / 0.409 | 0.333 / 0.227 | 0.537 | 0.961 | 0.000 | — | -0.360 (n=277) | calibration_broken=True (max reliability deviation 0.836 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 283 | 0.397 / 0.417 | 0.344 / 0.222 | 0.177 | 0.604 | 0.011 | — | -0.335 (n=219) | calibration_broken=True (max reliability deviation 0.446 >= 0.1); prediction_… |
| pepe | `salvageable_with_better_features_or_labels` | 283 | 0.416 / 0.439 | 0.344 / 0.212 | -0.074 | 0.382 | 0.011 | — | -0.181 (n=183) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| render-token | `structurally_noisy_retire` | 283 | 0.422 / 0.414 | 0.330 / 0.216 | 0.329 | 0.742 | 0.004 | — | -0.378 (n=245) | calibration_broken=True (max reliability deviation 0.814 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 283 | 0.414 / 0.431 | 0.346 / 0.219 | 0.254 | 0.735 | 0.028 | — | -0.141 (n=260) | calibration_broken=True (max reliability deviation 0.335 >= 0.1); prediction_… |
