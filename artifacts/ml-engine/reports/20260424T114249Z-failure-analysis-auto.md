# Auto failure-analysis — 2026-04-24T11:42:49.716300+00:00

- **Source report generated_at:** `2026-04-24T11:06:49.802730+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 11 |
| `insufficient_sample` | 9 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | structurally_noisy_retire | total |
|---|---|---|---|
| 1d | 9 | 1 | 10 |
| 6h | 0 | 10 | 10 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 593 | 0.430 / 0.442 | 0.325 / 0.212 | 0.410 | 0.877 | 0.440 | — | -0.087 (n=574) | calibration_broken=True (max reliability deviation 0.832 >= 0.1); prediction_… |
| bonk | `insufficient_sample` | 66 | 0.375 / 0.476 | 0.320 / 0.225 | 0.303 | 0.758 | 0.212 | — | -0.009 (n=62) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| celestia | `insufficient_sample` | 66 | 0.462 / 0.429 | 0.314 / 0.232 | 0.136 | 0.621 | 0.015 | — | 0.893 (n=58) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| dogwifcoin | `insufficient_sample` | 66 | 0.498 / 0.509 | 0.291 / 0.220 | 0.455 | 0.970 | 0.167 | — | 0.588 (n=54) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `insufficient_sample` | 66 | 0.407 / 0.418 | 0.353 / 0.233 | 0.121 | 0.561 | 0.000 | — | 1.041 (n=59) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| injective-protocol | `insufficient_sample` | 66 | 0.422 / 0.385 | 0.351 / 0.259 | 0.167 | 0.621 | 0.273 | — | 0.575 (n=64) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| jupiter-exchange-solana | `insufficient_sample` | 66 | 0.404 / 0.425 | 0.335 / 0.241 | 0.227 | 0.682 | 0.000 | — | 0.395 (n=64) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| pepe | `insufficient_sample` | 66 | 0.447 / 0.451 | 0.328 / 0.226 | 0.394 | 0.924 | 0.000 | — | 0.042 (n=44) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| render-token | `insufficient_sample` | 66 | 0.411 / 0.433 | 0.345 / 0.232 | 0.379 | 0.788 | 0.000 | — | 0.446 (n=65) | holdout 66 < MIN_HOLDOUT_ROWS=200 |
| worldcoin-wld | `insufficient_sample` | 66 | 0.436 / 0.433 | 0.338 / 0.227 | 0.182 | 0.697 | 0.000 | — | 1.197 (n=66) | holdout 66 < MIN_HOLDOUT_ROWS=200 |

### 6h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 2556 | 0.374 / 0.414 | 0.293 / 0.214 | 0.560 | 0.984 | 0.865 | — | 0.247 (n=67) | calibration_broken=True (max reliability deviation 0.479 >= 0.1); prediction_… |
| bonk | `structurally_noisy_retire` | 284 | 0.390 / 0.429 | 0.358 / 0.213 | 0.201 | 0.616 | 0.563 | — | 0.718 (n=1) | calibration_broken=True (max reliability deviation 0.145 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 284 | 0.387 / 0.386 | 0.355 / 0.221 | 0.172 | 0.592 | 0.074 | — | -0.079 (n=120) | calibration_broken=True (max reliability deviation 0.186 >= 0.1); prediction_… |
| dogwifcoin | `structurally_noisy_retire` | 284 | 0.395 / 0.422 | 0.347 / 0.217 | 0.370 | 0.813 | 0.000 | — | 0.079 (n=219) | calibration_broken=True (max reliability deviation 0.738 >= 0.1); prediction_… |
| floki-inu | `structurally_noisy_retire` | 284 | 0.384 / 0.436 | 0.367 / 0.214 | 0.510 | 0.940 | 0.014 | — | -0.228 (n=178) | calibration_broken=True (max reliability deviation 0.444 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 284 | 0.372 / 0.392 | 0.358 / 0.225 | 0.549 | 0.961 | 0.028 | — | -0.365 (n=224) | calibration_broken=True (max reliability deviation 0.243 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 284 | 0.380 / 0.405 | 0.361 / 0.222 | 0.444 | 0.863 | 0.852 | — | 0.412 (n=40) | calibration_broken=True (max reliability deviation 0.223 >= 0.1); prediction_… |
| pepe | `structurally_noisy_retire` | 284 | 0.388 / 0.417 | 0.366 / 0.215 | 0.518 | 0.922 | 0.085 | — | -0.198 (n=92) | calibration_broken=True (max reliability deviation 0.426 >= 0.1); prediction_… |
| render-token | `structurally_noisy_retire` | 284 | 0.394 / 0.399 | 0.346 / 0.219 | 0.246 | 0.694 | 0.011 | — | -0.030 (n=268) | calibration_broken=True (max reliability deviation 0.639 >= 0.1); prediction_… |
| worldcoin-wld | `structurally_noisy_retire` | 284 | 0.376 / 0.414 | 0.366 / 0.218 | 0.366 | 0.820 | 0.000 | — | -0.011 (n=260) | calibration_broken=True (max reliability deviation 0.257 >= 0.1); prediction_… |
