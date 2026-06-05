# Quant Verification Gate Failure Analysis — 2026-04-23

- **Source verification report:** `artifacts/ml-engine/reports/20260422T223431Z-baseline-verification.json`
- **Source counts:** {"slices_promoted": 0, "slices_no_lift": 4, "slices_below_coinflip": 41, "slices_insufficient_sample": 10, "slices_contract_failed": 0, "slices_untrained": 11}
- **Enrichment source:** `artifacts/ml-engine/models/report.json` (generated 2026-04-22T22:51:29.978473+00:00)
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_HOLDOUT_FRACTION": 0.2, "TAKER_FEE_BPS_PER_TRADE": 0.15}

> **Caveat:** The 22:34:31Z report's full per-slice surface was overwritten by the next training cycle (~17 min later, 2026-04-23T00:26:32Z). All Brier/calibration/fold/regime/PnL fields below are from that next cycle, which exhibits the same failure pattern (2/66 promoted vs 0/66). Source DA / baseline_DA / n_test / status fields come from the original 22:34:31Z file.

## 1. Bucket assignment summary

Strict assignment rules (priority order, exactly as implemented in `compute_failure_metrics.assign_bucket`):

1. `promoted` — DA ≥ 0.50 AND holdout ≥ 200 AND timeframe in tradeable set.
2. `salvageable_with_schema_fix` — `contamination_flag=True` OR cadence audit shows mixed source.
3. `insufficient_sample` — `status=untrained` OR `n_test < 200`.
4. `structurally_noisy_retire` — CONJUNCTION of all four:
   - cadence-clean (no contamination, no mixed cadence)
   - sufficient sample (`n_test ≥ 200`)
   - calibration broken (max per-class reliability deviation `≥ 0.10`)
   - prediction collapse (`collapse_gap ≥ 0.15` OR `predicted_top_class_share ≥ 0.85` OR `share_within_eps_of_prior ≥ 0.60`)
   Importance instability (rank-corr < 0.5 across folds) is corroborating evidence; not required because per-fold importances are not persisted today (see follow-up #316).
5. `salvageable_with_better_features_or_labels` — anything else (red gate but signal remaining).

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 24 |
| `salvageable_with_better_features_or_labels` | 22 |
| `insufficient_sample` | 20 |

**Why `salvageable_with_schema_fix = 0` today:** `price_history` has no native-cadence column, so `contamination_flag` cannot be set by the trainer and is `false` everywhere. The cadence-audit proxy (inter-arrival-gap analysis on the labeled dataset) finds no mixed cadence in any per-coin slice. The schema-fix risk is preventive — see `20260423T000000Z-schema-audit.md` — and binds the moment task #306's CMC-daily and OKX-hourly backfill modules land.

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | salvageable_with_better_features_or_labels | structurally_noisy_retire | total |
|---|---|---|---|---|
| 1m | 0 | 4 | 7 | 11 |
| 5m | 0 | 6 | 5 | 11 |
| 1h | 0 | 5 | 6 | 11 |
| 2h | 0 | 5 | 6 | 11 |
| 6h | 10 | 1 | 0 | 11 |
| 1d | 10 | 1 | 0 | 11 |
| (pooled) | 0 | 6 | 0 | 6 |

## 3. Per-slice diagnostic detail

All metrics below come from re-running the persisted `(model.txt + calibrators.joblib)` over the chronological 20% calibration holdout from the corresponding dataset parquet (matching `CALIBRATION_HOLDOUT_FRACTION=0.2` in `train.py`). Source DA / baseline_DA / n_test come from the original 22:34:31Z `baseline-verification.json`.

### 1m

| Coin | Bucket | n_test | DA / baseline | Brier vs base | Pred-class share (D/S/U) | Confidence DA ≥0.6 | Net PnL %/trade | Top regime DA | Reliability max-dev | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| __pooled__ | `salvageable_with_better_features_or_labels` | 7160 | 0.482 / 0.468 | 0.239 / 0.207 | — | — | — | — | — | red gate but signal remaining — calibration_broken=False, prediction_collapse=False; re… |
| bonk | `structurally_noisy_retire` | 1800 | 0.442 / 0.484 | 0.314 / 0.218 | 0.00/0.99/0.01 | 0.657 (n=505) | 0.008 (n=5) | trending_down 0.684 | 0.149 | calibration_broken=True (max reliability deviation 0.149 >= 0.1); prediction_collapse=T… |
| celestia | `structurally_noisy_retire` | 1400 | 0.485 / 0.548 | 0.269 / 0.203 | 0.00/1.00/0.00 | 0.720 (n=614) | 0.000 (n=0) | low_vol_compression 0.702 | 0.105 | calibration_broken=True (max reliability deviation 0.105 >= 0.1); prediction_collapse=T… |
| dogwifcoin | `salvageable_with_better_features_or_labels` | 1400 | 0.518 / 0.563 | 0.306 / 0.201 | 0.00/1.00/0.00 | 0.700 (n=540) | 0.140 (n=1) | low_vol_compression 0.673 | 0.027 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| floki-inu | `structurally_noisy_retire` | 1400 | 0.454 / 0.556 | 0.275 / 0.218 | 0.00/1.00/0.00 | 0.702 (n=419) | 0.000 (n=0) | trending_down 0.720 | 0.104 | calibration_broken=True (max reliability deviation 0.104 >= 0.1); prediction_collapse=T… |
| injective-protocol | `salvageable_with_better_features_or_labels` | 1400 | 0.526 / 0.583 | 0.256 / 0.202 | 0.00/1.00/0.00 | 0.749 (n=704) | 0.000 (n=0) | low_vol_compression 0.753 | 0.032 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 1400 | 0.506 / 0.600 | 0.281 / 0.198 | 0.00/1.00/0.00 | 0.783 (n=700) | -0.117 (n=2) | low_vol_compression 0.783 | 0.220 | calibration_broken=True (max reliability deviation 0.220 >= 0.1); prediction_collapse=T… |
| pepe | `structurally_noisy_retire` | 1830 | 0.424 / 0.462 | 0.288 / 0.217 | 0.00/1.00/0.00 | 0.645 (n=740) | 0.013 (n=1) | trending_up 1.000 | 0.267 | calibration_broken=True (max reliability deviation 0.267 >= 0.1); prediction_collapse=T… |
| render-token | `salvageable_with_better_features_or_labels` | 1400 | 0.497 / 0.580 | 0.272 / 0.205 | 0.00/1.00/0.00 | 0.709 (n=697) | 0.000 (n=0) | low_vol_compression 0.709 | 0.077 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| sei-network | `structurally_noisy_retire` | 1400 | 0.425 / 0.548 | 0.354 / 0.223 | 0.01/0.99/0.00 | 0.582 (n=189) | 0.115 (n=6) | trending_up 0.600 | 0.264 | calibration_broken=True (max reliability deviation 0.264 >= 0.1); prediction_collapse=T… |
| worldcoin-wld | `structurally_noisy_retire` | 1400 | 0.457 / 0.562 | 0.279 / 0.202 | 0.02/0.97/0.01 | 0.688 (n=471) | -0.082 (n=19) | low_vol_compression 0.648 | 0.156 | calibration_broken=True (max reliability deviation 0.156 >= 0.1); prediction_collapse=T… |

### 5m

| Coin | Bucket | n_test | DA / baseline | Brier vs base | Pred-class share (D/S/U) | Confidence DA ≥0.6 | Net PnL %/trade | Top regime DA | Reliability max-dev | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| __pooled__ | `salvageable_with_better_features_or_labels` | 4712 | 0.461 / 0.466 | 0.227 / 0.190 | — | — | — | — | — | red gate but signal remaining — calibration_broken=False, prediction_collapse=False; re… |
| bonk | `structurally_noisy_retire` | 765 | 0.442 / 0.468 | 0.287 / 0.192 | 0.89/0.00/0.11 | 0.667 (n=3) | -0.151 (n=487) | trending_up 0.512 | 0.200 | calibration_broken=True (max reliability deviation 0.200 >= 0.1); prediction_collapse=T… |
| celestia | `structurally_noisy_retire` | 730 | 0.452 / 0.419 | 0.318 / 0.192 | 0.60/0.00/0.40 | 0.750 (n=8) | -0.202 (n=477) | trending_up 0.521 | 0.155 | calibration_broken=True (max reliability deviation 0.155 >= 0.1); prediction_collapse=T… |
| dogwifcoin | `structurally_noisy_retire` | 730 | 0.456 / 0.464 | 0.304 / 0.190 | 0.99/0.01/0.00 | — | -0.128 (n=475) | trending_down 0.494 | 0.106 | calibration_broken=True (max reliability deviation 0.106 >= 0.1); prediction_collapse=T… |
| floki-inu | `salvageable_with_better_features_or_labels` | 730 | 0.458 / 0.464 | 0.298 / 0.190 | 0.77/0.00/0.23 | 0.000 (n=2) | -0.079 (n=479) | trending_up 0.468 | 0.035 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| injective-protocol | `salvageable_with_better_features_or_labels` | 730 | 0.434 / 0.430 | 0.295 / 0.196 | 0.98/0.01/0.01 | 0.800 (n=5) | -0.218 (n=474) | trending_down 0.449 | 0.059 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| jupiter-exchange-solana | `salvageable_with_better_features_or_labels` | 730 | 0.423 / 0.452 | 0.291 / 0.193 | 0.46/0.00/0.54 | 0.500 (n=4) | -0.180 (n=479) | trending_up 0.500 | 0.081 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| pepe | `structurally_noisy_retire` | 765 | 0.438 / 0.472 | 0.302 / 0.190 | 0.60/0.00/0.40 | 0.583 (n=12) | -0.147 (n=487) | trending_up 0.593 | 0.178 | calibration_broken=True (max reliability deviation 0.178 >= 0.1); prediction_collapse=T… |
| render-token | `structurally_noisy_retire` | 730 | 0.445 / 0.442 | 0.278 / 0.192 | 0.90/0.00/0.10 | 0.875 (n=8) | -0.132 (n=478) | trending_down 0.521 | 0.126 | calibration_broken=True (max reliability deviation 0.126 >= 0.1); prediction_collapse=T… |
| sei-network | `salvageable_with_better_features_or_labels` | 730 | 0.448 / 0.434 | 0.325 / 0.199 | 0.87/0.00/0.13 | 0.750 (n=12) | -0.213 (n=388) | trending_down 0.500 | 0.100 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| worldcoin-wld | `salvageable_with_better_features_or_labels` | 730 | 0.464 / 0.467 | 0.277 / 0.187 | 0.55/0.02/0.43 | 0.714 (n=14) | 0.021 (n=469) | trending_down 0.562 | 0.160 | red gate but signal remaining — calibration_broken=True, prediction_collapse=False; reb… |

### 1h

| Coin | Bucket | n_test | DA / baseline | Brier vs base | Pred-class share (D/S/U) | Confidence DA ≥0.6 | Net PnL %/trade | Top regime DA | Reliability max-dev | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| __pooled__ | `salvageable_with_better_features_or_labels` | 4110 | 0.390 / 0.415 | 0.266 / 0.208 | — | — | — | — | — | red gate but signal remaining — calibration_broken=False, prediction_collapse=False; re… |
| bonk | `salvageable_with_better_features_or_labels` | 570 | 0.351 / 0.398 | 0.338 / 0.208 | 0.44/0.13/0.42 | 0.500 (n=6) | 0.157 (n=365) | trending_up 0.469 | 0.121 | red gate but signal remaining — calibration_broken=True, prediction_collapse=False; reb… |
| celestia | `salvageable_with_better_features_or_labels` | 570 | 0.344 / 0.344 | 0.327 / 0.209 | 0.45/0.05/0.50 | 0.833 (n=6) | -0.077 (n=399) | range_chop 0.488 | 0.057 | red gate but signal remaining — calibration_broken=False, prediction_collapse=False; re… |
| dogwifcoin | `salvageable_with_better_features_or_labels` | 570 | 0.368 / 0.398 | 0.327 / 0.211 | 0.89/0.00/0.11 | — | -0.086 (n=421) | trending_up 0.474 | 0.062 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| floki-inu | `structurally_noisy_retire` | 570 | 0.351 / 0.416 | 0.331 / 0.209 | 0.64/0.03/0.33 | 1.000 (n=3) | -0.193 (n=409) | trending_up 0.478 | 0.121 | calibration_broken=True (max reliability deviation 0.121 >= 0.1); prediction_collapse=T… |
| injective-protocol | `structurally_noisy_retire` | 570 | 0.370 / 0.340 | 0.334 / 0.214 | 0.24/0.13/0.63 | 0.500 (n=4) | 0.035 (n=367) | trending_up 0.447 | 0.131 | calibration_broken=True (max reliability deviation 0.131 >= 0.1); prediction_collapse=T… |
| jupiter-exchange-solana | `salvageable_with_better_features_or_labels` | 570 | 0.340 / 0.360 | 0.356 / 0.212 | 0.56/0.00/0.44 | 1.000 (n=4) | -0.121 (n=421) | range_chop 0.440 | 0.068 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |
| pepe | `structurally_noisy_retire` | 570 | 0.333 / 0.335 | 0.328 / 0.208 | 0.29/0.00/0.71 | 0.750 (n=8) | 0.098 (n=421) | trending_up 0.467 | 0.213 | calibration_broken=True (max reliability deviation 0.213 >= 0.1); prediction_collapse=T… |
| render-token | `structurally_noisy_retire` | 570 | 0.377 / 0.379 | 0.326 / 0.210 | 0.67/0.00/0.32 | 0.625 (n=8) | -0.073 (n=419) | trending_down 0.449 | 0.163 | calibration_broken=True (max reliability deviation 0.163 >= 0.1); prediction_collapse=T… |
| sei-network | `structurally_noisy_retire` | 570 | 0.289 / 0.335 | 0.375 / 0.222 | 0.12/0.77/0.11 | — | -0.243 (n=75) | trending_down 0.414 | 0.127 | calibration_broken=True (max reliability deviation 0.127 >= 0.1); prediction_collapse=T… |
| worldcoin-wld | `structurally_noisy_retire` | 570 | 0.409 / 0.404 | 0.337 / 0.206 | 0.73/0.00/0.27 | 0.765 (n=17) | -0.112 (n=420) | trending_down 0.495 | 0.309 | calibration_broken=True (max reliability deviation 0.309 >= 0.1); prediction_collapse=T… |

### 2h

| Coin | Bucket | n_test | DA / baseline | Brier vs base | Pred-class share (D/S/U) | Confidence DA ≥0.6 | Net PnL %/trade | Top regime DA | Reliability max-dev | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| __pooled__ | `salvageable_with_better_features_or_labels` | 3273 | 0.411 / 0.431 | 0.282 / 0.208 | — | — | — | — | — | red gate but signal remaining — calibration_broken=False, prediction_collapse=False; re… |
| bonk | `structurally_noisy_retire` | 270 | 0.344 / 0.381 | 0.327 / 0.206 | 0.93/0.02/0.05 | 0.864 (n=22) | -0.048 (n=334) | trending_up 0.489 | 0.234 | calibration_broken=True (max reliability deviation 0.234 >= 0.1); prediction_collapse=T… |
| celestia | `salvageable_with_better_features_or_labels` | 270 | 0.326 / 0.370 | 0.336 / 0.212 | 0.47/0.01/0.52 | — | 0.033 (n=336) | range_chop 0.441 | 0.042 | red gate but signal remaining — calibration_broken=False, prediction_collapse=False; re… |
| dogwifcoin | `structurally_noisy_retire` | 270 | 0.367 / 0.322 | 0.349 / 0.211 | 0.92/0.01/0.08 | 0.600 (n=5) | -0.090 (n=338) | trending_up 0.528 | 0.170 | calibration_broken=True (max reliability deviation 0.170 >= 0.1); prediction_collapse=T… |
| floki-inu | `structurally_noisy_retire` | 270 | 0.407 / 0.378 | 0.348 / 0.209 | 0.92/0.07/0.01 | 0.778 (n=27) | -0.159 (n=316) | trending_up 0.532 | 0.293 | calibration_broken=True (max reliability deviation 0.293 >= 0.1); prediction_collapse=T… |
| injective-protocol | `structurally_noisy_retire` | 270 | 0.333 / 0.326 | 0.365 / 0.215 | 0.24/0.64/0.13 | 0.000 (n=2) | 0.361 (n=124) | trending_up 0.432 | 0.132 | calibration_broken=True (max reliability deviation 0.132 >= 0.1); prediction_collapse=T… |
| jupiter-exchange-solana | `salvageable_with_better_features_or_labels` | 270 | 0.333 / 0.359 | 0.341 / 0.215 | 0.52/0.00/0.48 | 0.333 (n=3) | 0.085 (n=340) | trending_down 0.430 | 0.131 | red gate but signal remaining — calibration_broken=True, prediction_collapse=False; reb… |
| pepe | `salvageable_with_better_features_or_labels` | 270 | 0.363 / 0.348 | 0.338 / 0.208 | 0.53/0.00/0.47 | 0.714 (n=14) | 0.336 (n=339) | trending_up 0.495 | 0.159 | red gate but signal remaining — calibration_broken=True, prediction_collapse=False; reb… |
| render-token | `structurally_noisy_retire` | 270 | 0.341 / 0.437 | 0.324 / 0.210 | 0.81/0.00/0.18 | 0.833 (n=6) | -0.275 (n=339) | trending_up 0.519 | 0.147 | calibration_broken=True (max reliability deviation 0.147 >= 0.1); prediction_collapse=T… |
| sei-network | `structurally_noisy_retire` | 270 | 0.430 / 0.330 | 0.357 / 0.219 | 0.20/0.15/0.65 | 0.667 (n=3) | -0.182 (n=185) | trending_down 0.408 | 0.131 | calibration_broken=True (max reliability deviation 0.131 >= 0.1); prediction_collapse=T… |
| worldcoin-wld | `salvageable_with_better_features_or_labels` | 270 | 0.385 / 0.422 | 0.336 / 0.208 | 0.97/0.00/0.03 | 1.000 (n=1) | -0.052 (n=339) | trending_down 0.464 | 0.050 | red gate but signal remaining — calibration_broken=False, prediction_collapse=True; reb… |

### 6h

| Coin | Bucket | n_test | DA / baseline | Brier vs base | Pred-class share (D/S/U) | Confidence DA ≥0.6 | Net PnL %/trade | Top regime DA | Reliability max-dev | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| __pooled__ | `salvageable_with_better_features_or_labels` | 2729 | 0.388 / 0.421 | 0.295 / 0.214 | — | — | — | — | — | red gate but signal remaining — calibration_broken=False, prediction_collapse=False; re… |
| bonk | `insufficient_sample` | 36 | 0.444 / 0.500 | 0.350 / 0.214 | 0.51/0.01/0.48 | — | 0.167 (n=283) | trending_up 0.556 | 0.072 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| celestia | `insufficient_sample` | 36 | 0.444 / 0.333 | 0.355 / 0.220 | 1.00/0.00/0.00 | 0.375 (n=8) | -0.441 (n=286) | trending_up 0.516 | 0.276 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| dogwifcoin | `insufficient_sample` | 36 | 0.389 / 0.306 | 0.350 / 0.217 | 0.86/0.00/0.14 | — | -0.205 (n=286) | trending_up 0.563 | 0.155 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `insufficient_sample` | 36 | 0.444 / 0.417 | 0.364 / 0.214 | 0.95/0.00/0.05 | 0.667 (n=3) | -0.334 (n=286) | trending_up 0.595 | 0.149 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| injective-protocol | `insufficient_sample` | 36 | 0.444 / 0.361 | 0.348 / 0.222 | 0.92/0.01/0.07 | 0.800 (n=5) | -0.259 (n=283) | trending_up 0.544 | 0.160 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| jupiter-exchange-solana | `insufficient_sample` | 36 | 0.417 / 0.389 | 0.353 / 0.221 | 0.93/0.00/0.07 | 0.714 (n=7) | -0.381 (n=285) | trending_up 0.522 | 0.146 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| pepe | `insufficient_sample` | 36 | 0.389 / 0.444 | 0.375 / 0.215 | 0.71/0.00/0.29 | 0.750 (n=4) | 0.236 (n=286) | trending_up 0.591 | 0.124 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| render-token | `insufficient_sample` | 36 | 0.389 / 0.389 | 0.343 / 0.219 | 0.57/0.00/0.43 | 0.700 (n=10) | 0.021 (n=286) | range_chop 0.560 | 0.086 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| sei-network | `insufficient_sample` | 36 | 0.417 / 0.389 | 0.359 / 0.230 | 0.88/0.01/0.11 | 0.714 (n=7) | -0.177 (n=161) | trending_up 0.512 | 0.138 | holdout 36 < MIN_HOLDOUT_ROWS=200 |
| worldcoin-wld | `insufficient_sample` | 36 | 0.361 / 0.333 | 0.359 / 0.217 | 0.89/0.00/0.11 | 0.667 (n=27) | 0.498 (n=286) | trending_down 0.475 | 0.069 | holdout 36 < MIN_HOLDOUT_ROWS=200 |

### 1d

| Coin | Bucket | n_test | DA / baseline | Brier vs base | Pred-class share (D/S/U) | Confidence DA ≥0.6 | Net PnL %/trade | Top regime DA | Reliability max-dev | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| __pooled__ | `salvageable_with_better_features_or_labels` | 662 | 0.410 / 0.444 | 0.323 / 0.214 | — | — | — | — | — | red gate but signal remaining — calibration_broken=False, prediction_collapse=False; re… |
| bonk | `insufficient_sample` | 0 | — | 0.345 / 0.247 | 0.96/0.00/0.04 | — | 0.202 (n=67) | trending_up 0.571 | 0.102 | status=untrained (n_test=0); not enough labeled rows to train |
| celestia | `insufficient_sample` | 0 | — | 0.342 / 0.248 | 0.96/0.00/0.04 | 1.000 (n=1) | -0.864 (n=67) | trending_up 0.727 | 0.085 | status=untrained (n_test=0); not enough labeled rows to train |
| dogwifcoin | `insufficient_sample` | 0 | — | 0.336 / 0.242 | 0.73/0.00/0.27 | 0.700 (n=10) | 1.288 (n=67) | trending_up 0.667 | 0.071 | status=untrained (n_test=0); not enough labeled rows to train |
| floki-inu | `insufficient_sample` | 0 | — | 0.332 / 0.228 | 0.87/0.00/0.13 | 0.636 (n=11) | 0.608 (n=67) | trending_down 0.645 | 0.040 | status=untrained (n_test=0); not enough labeled rows to train |
| injective-protocol | `insufficient_sample` | 0 | — | 0.345 / 0.242 | 0.85/0.00/0.15 | 0.750 (n=4) | -1.308 (n=67) | trending_up 0.625 | 0.032 | status=untrained (n_test=0); not enough labeled rows to train |
| jupiter-exchange-solana | `insufficient_sample` | 0 | — | 0.351 / 0.246 | 0.69/0.00/0.31 | 0.778 (n=18) | 1.741 (n=67) | trending_down 0.647 | 0.117 | status=untrained (n_test=0); not enough labeled rows to train |
| pepe | `insufficient_sample` | 0 | — | 0.337 / 0.217 | 0.66/0.01/0.33 | 0.719 (n=32) | 1.405 (n=66) | trending_down 0.704 | 0.111 | status=untrained (n_test=0); not enough labeled rows to train |
| render-token | `insufficient_sample` | 0 | — | 0.361 / 0.251 | 0.67/0.03/0.30 | 0.857 (n=7) | 0.137 (n=65) | trending_down 0.583 | 0.243 | status=untrained (n_test=0); not enough labeled rows to train |
| sei-network | `insufficient_sample` | 0 | — | 0.336 / 0.225 | 0.85/0.15/0.00 | 0.632 (n=19) | 0.672 (n=57) | range_chop 0.667 | 0.199 | status=untrained (n_test=0); not enough labeled rows to train |
| worldcoin-wld | `insufficient_sample` | 0 | — | 0.329 / 0.221 | 0.91/0.03/0.06 | 0.700 (n=10) | 2.151 (n=65) | trending_up 1.000 | 0.085 | status=untrained (n_test=0); not enough labeled rows to train |

## 4. Smallest first set (per-coin 5m, ranked)

Ranking score = `model_DA + 2 × max(0, lift)`. We pursue per-coin 5m first because the next-step lever (per-coin realized-vol-driven label thresholds, follow-up #318) attaches at the per-coin × per-timeframe level and these slices are closest to baseline_DA = 0.50 with a real holdout.

| # | Coin | baseline_DA | model_DA | lift | n_test | rank_score | Repair |
|---|---|---|---|---|---|---|---|
| 1 | sei-network | 0.4342 | 0.4479 | +0.0137 | 730 | 0.4753 | iterate on per-coin label thresholds (#318) and feature set; rerun gate |
| 2 | worldcoin-wld | 0.4671 | 0.4644 | -0.0027 | 730 | 0.4644 | iterate on per-coin label thresholds (#318) and feature set; rerun gate |
| 3 | floki-inu | 0.4644 | 0.4575 | -0.0068 | 730 | 0.4575 | iterate on per-coin label thresholds (#318) and feature set; rerun gate |
| 4 | injective-protocol | 0.4301 | 0.4342 | +0.0041 | 730 | 0.4425 | iterate on per-coin label thresholds (#318) and feature set; rerun gate |
| 5 | jupiter-exchange-solana | 0.4521 | 0.4233 | -0.0288 | 730 | 0.4233 | iterate on per-coin label thresholds (#318) and feature set; rerun gate |

## 5. Repair plan (no actual repairs run)

Apply in priority order. The verification gate constants stay unchanged — `MIN_HOLDOUT_ROWS=200`, `MIN_DIRECTIONAL_ACCURACY=0.50` — and no synthetic fills are introduced.

1. **Ship preventive schema fix before #306 retries.** New `price_candles` table per `20260423T000000Z-schema-audit.md`; trainer reads candles directly for any timeframe ≥ 5m. Tracked as follow-up #317. Required to keep verification trustworthy once daily and hourly backfill modules land.
2. **Extend `report.py` to persist Brier / per-class breakdown / confidence-bucket DA / regime DA / PnL / per-fold feature importances during training.** Today these are partially derivable from persisted artifacts post-hoc (this report did so) but `feature_importance_stability` cannot be computed without per-fold importances, which fold_metrics does not store. Tracked as follow-up #316.
3. **Apply per-coin 5m label thresholds derived from realized 5m vol** to the smallest first set above (follow-up #318). Acceptance bar (gate unchanged): baseline DA ≥ 0.55 AND model DA > baseline DA + 0.01 AND model DA > 0.50 on holdout ≥ 200.
4. **Retire the slices in `structurally_noisy_retire`** (cadence-clean + sufficient sample but with both broken calibration and predictions parked near the class prior), OR redefine the label scheme entirely (e.g., binary up-vs-not, multi-horizon). The 3-class head on these slices has demonstrably nothing to learn under the current label thresholds — per-coin threshold tuning will not rescue a collapsed, miscalibrated head.
5. **Wait for more data on 6h and 1d.** 6h holdout is below the 200-row floor (median ≈ 36); 1d is untrained (live-poll ticks have <35 distinct daily closes per coin). No code change required — the slices will graduate as the data window grows.

## 6. Per-class collapse and PnL — at-a-glance for the 5m cohort

| Coin | label_top_share | pred_top_share | collapse_gap | high-conf rows ≥ 0.6 | high-conf DA | net PnL %/trade | round-trip cost % |
|---|---|---|---|---|---|---|---|
| bonk | 0.413 | 0.889 | 0.476 | 3 | 0.667 | -0.151 | 0.150 |
| celestia | 0.426 | 0.597 | 0.171 | 8 | 0.750 | -0.202 | 0.150 |
| dogwifcoin | 0.428 | 0.988 | 0.559 | 0 | — | -0.128 | 0.150 |
| floki-inu | 0.424 | 0.772 | 0.349 | 2 | 0.000 | -0.079 | 0.150 |
| injective-protocol | 0.397 | 0.983 | 0.587 | 5 | 0.800 | -0.218 | 0.150 |
| jupiter-exchange-solana | 0.380 | 0.541 | 0.161 | 4 | 0.500 | -0.180 | 0.150 |
| pepe | 0.423 | 0.596 | 0.172 | 12 | 0.583 | -0.147 | 0.150 |
| render-token | 0.411 | 0.902 | 0.491 | 8 | 0.875 | -0.132 | 0.150 |
| sei-network | 0.389 | 0.869 | 0.479 | 12 | 0.750 | -0.213 | 0.150 |
| worldcoin-wld | 0.438 | 0.547 | 0.109 | 14 | 0.714 | 0.021 | 0.150 |

## 6b. Train-vs-holdout class balance and near-prior diagnostics

`l1_drift` = sum |hold_share − train_share| over the three label classes; high drift means the holdout's regime differs from training. `share_within_eps_of_prior` is the fraction of holdout rows whose predicted distribution is within L1=0.05 of the training class prior — a high share means the calibrated head is essentially outputting the prior. `max_prob_std` is the std-dev of the predicted top-class probability across rows; near-zero values flag a model that has parked on the prior.

| Slice | TvH l1_drift | Δ DOWN | Δ STABLE | Δ UP | share within ε of prior | max_prob_mean | max_prob_std |
|---|---|---|---|---|---|---|---|
| bonk 1m | 0.861 | -0.251 | 0.431 | -0.180 | 0.000 | 0.630 | 0.077 |
| celestia 1m | 0.973 | -0.243 | 0.486 | -0.243 | 0.000 | 0.693 | 0.098 |
| dogwifcoin 1m | 0.901 | -0.203 | 0.450 | -0.247 | 0.000 | 0.666 | 0.069 |
| floki-inu 1m | 0.888 | -0.256 | 0.444 | -0.188 | 0.000 | 0.647 | 0.097 |
| injective-protocol 1m | 1.072 | -0.268 | 0.536 | -0.268 | 0.000 | 0.751 | 0.072 |
| jupiter-exchange-solana 1m | 1.131 | -0.259 | 0.566 | -0.306 | 0.000 | 0.784 | 0.080 |
| pepe 1m | 0.911 | -0.242 | 0.455 | -0.213 | 0.000 | 0.645 | 0.036 |
| render-token 1m | 0.984 | -0.247 | 0.492 | -0.245 | 0.000 | 0.708 | 0.052 |
| sei-network 1m | 0.614 | -0.175 | 0.307 | -0.132 | 0.000 | 0.582 | 0.052 |
| worldcoin-wld 1m | 0.857 | -0.221 | 0.429 | -0.208 | 0.000 | 0.626 | 0.112 |
| bonk 5m | 0.291 | -0.095 | 0.145 | -0.051 | 0.000 | 0.435 | 0.042 |
| celestia 5m | 0.295 | -0.081 | 0.148 | -0.066 | 0.000 | 0.431 | 0.037 |
| dogwifcoin 5m | 0.350 | -0.065 | 0.175 | -0.110 | 0.000 | 0.430 | 0.029 |
| floki-inu 5m | 0.268 | -0.102 | 0.134 | -0.032 | 0.000 | 0.431 | 0.036 |
| injective-protocol 5m | 0.357 | -0.097 | 0.178 | -0.081 | 0.000 | 0.401 | 0.050 |
| jupiter-exchange-solana 5m | 0.452 | -0.114 | 0.226 | -0.112 | 0.000 | 0.413 | 0.051 |
| pepe 5m | 0.261 | -0.093 | 0.130 | -0.037 | 0.000 | 0.446 | 0.051 |
| render-token 5m | 0.369 | -0.081 | 0.185 | -0.104 | 0.000 | 0.430 | 0.043 |
| sei-network 5m | 0.379 | -0.137 | 0.189 | -0.053 | 0.000 | 0.396 | 0.065 |
| worldcoin-wld 5m | 0.256 | -0.071 | 0.128 | -0.056 | 0.000 | 0.485 | 0.069 |
| bonk 1h | 0.265 | -0.085 | 0.132 | -0.048 | 0.000 | 0.430 | 0.067 |
| celestia 1h | 0.180 | -0.066 | 0.090 | -0.024 | 0.000 | 0.446 | 0.059 |
| dogwifcoin 1h | 0.246 | -0.049 | 0.123 | -0.074 | 0.000 | 0.399 | 0.026 |
| floki-inu 1h | 0.127 | -0.037 | 0.064 | -0.026 | 0.017 | 0.454 | 0.048 |
| injective-protocol 1h | 0.271 | -0.093 | 0.135 | -0.042 | 0.000 | 0.404 | 0.039 |
| jupiter-exchange-solana 1h | 0.167 | -0.049 | 0.084 | -0.035 | 0.014 | 0.408 | 0.035 |
| pepe 1h | 0.201 | -0.064 | 0.100 | -0.036 | 0.000 | 0.419 | 0.064 |
| render-token 1h | 0.265 | -0.061 | 0.132 | -0.071 | 0.002 | 0.412 | 0.054 |
| sei-network 1h | 0.281 | -0.109 | 0.141 | -0.032 | 0.046 | 0.382 | 0.053 |
| worldcoin-wld 1h | 0.203 | -0.070 | 0.102 | -0.032 | 0.005 | 0.442 | 0.081 |
| bonk 2h | 0.328 | -0.069 | 0.164 | -0.095 | 0.026 | 0.423 | 0.100 |
| celestia 2h | 0.316 | -0.072 | 0.158 | -0.086 | 0.000 | 0.400 | 0.049 |
| dogwifcoin 2h | 0.301 | -0.051 | 0.151 | -0.100 | 0.009 | 0.410 | 0.056 |
| floki-inu 2h | 0.248 | -0.039 | 0.124 | -0.085 | 0.000 | 0.426 | 0.088 |
| injective-protocol 2h | 0.407 | -0.112 | 0.203 | -0.092 | 0.000 | 0.399 | 0.052 |
| jupiter-exchange-solana 2h | 0.179 | -0.063 | 0.089 | -0.026 | 0.006 | 0.410 | 0.050 |
| pepe 2h | 0.241 | -0.045 | 0.120 | -0.076 | 0.000 | 0.434 | 0.089 |
| render-token 2h | 0.183 | -0.029 | 0.092 | -0.063 | 0.000 | 0.442 | 0.061 |
| sei-network 2h | 0.285 | -0.102 | 0.142 | -0.041 | 0.018 | 0.377 | 0.055 |
| worldcoin-wld 2h | 0.155 | -0.036 | 0.078 | -0.042 | 0.035 | 0.437 | 0.030 |
| bonk 6h | 0.086 | -0.015 | 0.043 | -0.029 | 0.000 | 0.439 | 0.034 |
| celestia 6h | 0.062 | 0.031 | -0.007 | -0.024 | 0.717 | 0.460 | 0.050 |
| dogwifcoin 6h | 0.092 | 0.040 | 0.006 | -0.046 | 0.000 | 0.465 | 0.021 |
| floki-inu 6h | 0.042 | 0.021 | -0.009 | -0.012 | 0.559 | 0.460 | 0.037 |
| injective-protocol 6h | 0.130 | -0.006 | 0.065 | -0.059 | 0.000 | 0.431 | 0.050 |
| jupiter-exchange-solana 6h | 0.055 | 0.028 | -0.012 | -0.015 | 0.678 | 0.433 | 0.052 |
| pepe 6h | 0.121 | -0.008 | 0.060 | -0.052 | 0.032 | 0.426 | 0.058 |
| render-token 6h | 0.110 | 0.024 | -0.055 | 0.031 | 0.000 | 0.461 | 0.059 |
| sei-network 6h | 0.068 | 0.034 | -0.015 | -0.018 | 0.000 | 0.501 | 0.064 |
| worldcoin-wld 6h | 0.043 | 0.016 | 0.005 | -0.021 | 0.000 | 0.469 | 0.085 |
| bonk 1d | 0.070 | -0.033 | 0.035 | -0.002 | 0.239 | 0.460 | 0.037 |
| celestia 1d | 0.071 | -0.026 | 0.035 | -0.009 | 0.000 | 0.468 | 0.056 |
| dogwifcoin 1d | 0.086 | 0.038 | 0.005 | -0.043 | 0.000 | 0.585 | 0.056 |
| floki-inu 1d | 0.049 | 0.001 | 0.024 | -0.025 | 0.000 | 0.524 | 0.088 |
| injective-protocol 1d | 0.048 | -0.018 | -0.006 | 0.024 | 0.000 | 0.461 | 0.098 |
| jupiter-exchange-solana 1d | 0.033 | -0.003 | -0.014 | 0.016 | 0.000 | 0.554 | 0.118 |
| pepe 1d | 0.020 | 0.008 | -0.010 | 0.002 | 0.000 | 0.609 | 0.103 |
| render-token 1d | 0.109 | 0.012 | -0.055 | 0.043 | 0.000 | 0.530 | 0.130 |
| sei-network 1d | 0.138 | 0.034 | 0.035 | -0.069 | 0.000 | 0.542 | 0.109 |
| worldcoin-wld 1d | 0.056 | 0.000 | 0.028 | -0.028 | 0.000 | 0.534 | 0.123 |

## 7. Reproducibility

This report is regenerated end-to-end by:

```bash
pnpm --filter @workspace/ml-engine exec python scripts/compute_failure_metrics.py
pnpm --filter @workspace/ml-engine exec python scripts/render_failure_analysis_md.py
```

The cadence-correctness contract for the schema fix is enforced by:

- `artifacts/ml-engine/tests/test_cadence_correctness.py` — three behavior tests:
  - `test_daily_rows_are_not_silently_merged_into_5m_bars` — feeds two daily rows into `resample_to_candles(bucket_ms=300_000)` and asserts `CadenceMismatchError` is raised. Fails today (function silently buckets).
  - `test_resample_quarantines_coarser_rows_within_a_bucket` — feeds four 30s ticks at $100 plus a daily $999 row in the same 5m bucket and asserts the close is $100. Fails today (returns $999).
  - `test_trainer_provenance_records_native_cadence_and_refuses_mixed` — asserts persisted manifests carry `bars_by_native_cadence` + `cadence_mixed` AND that the verification gate refuses to promote cadence-mixed unmitigated slices. Fails today (no field, no helper).
- `artifacts/api-server/test/price-candles-uniqueness.test.ts` — three node-test assertions over the Drizzle schema package: (a) `priceCandlesTable` exported, (b) carries `(coin_id,timeframe,bucket_start,source)` columns, (c) `priceHistoryTable` must NOT gain a `timeframe` column. (a)+(b) fail today; (c) is a passing regression guard.

## 8. What is NOT measurable from existing artifacts

- **Per-fold feature-importance stability** — `fold_metrics` does not persist the per-fold LightGBM importance arrays (only `best_params` and CV metrics). `feature_importance_stability.status` is therefore `deferred` for every slice. Closing this gap requires extending `train.py` to write each fold's importance vector into `fold_metrics`. This is captured in follow-up #316; until that lands, importance-stability cannot be computed retrospectively.
- **`regime_subset` block in source report** — the trainer wrote an empty list for every slice. Regime-bucketed DA in this report instead comes from re-running inference and grouping by the `regime` column on the dataset parquet, which is the same regime label the trainer would have used.
