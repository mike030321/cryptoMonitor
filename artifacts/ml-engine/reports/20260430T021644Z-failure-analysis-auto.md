# Auto failure-analysis — 2026-04-30T02:16:44.240990+00:00

- **Source report generated_at:** `2026-04-30T01:13:57.654936+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "MIN_DIRECTIONAL_ACCURACY_PER_TF": {"1d": 0.53}, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `insufficient_sample` | 7 |
| `structurally_noisy_retire` | 2 |
| `salvageable_with_better_features_or_labels` | 1 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | salvageable_with_better_features_or_labels | structurally_noisy_retire | total |
|---|---|---|---|---|
| 1d | 7 | 1 | 2 | 10 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 1549 | 0.373 / 0.385 | 0.293 / 0.221 | 0.033 | 0.446 | 0.023 | — | 0.098 (n=1081) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `insufficient_sample` | 162 | 0.431 / 0.410 | 0.315 / 0.219 | 0.296 | 0.722 | 0.018 | — | -0.080 (n=143) | holdout 162 < MIN_HOLDOUT_ROWS=200 |
| celestia | `insufficient_sample` | 176 | 0.388 / 0.401 | 0.346 / 0.232 | 0.375 | 0.818 | 0.017 | — | 0.369 (n=152) | holdout 176 < MIN_HOLDOUT_ROWS=200 |
| dogwifcoin | `insufficient_sample` | 142 | 0.363 / 0.351 | 0.354 / 0.242 | 0.261 | 0.690 | 0.007 | — | -0.258 (n=104) | holdout 142 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `structurally_noisy_retire` | 213 | 0.351 / 0.380 | 0.347 / 0.221 | 0.160 | 0.559 | 0.005 | — | -0.182 (n=155) | calibration_broken=True (max reliability deviation 0.625 >= 0.1); prediction_… |
| injective-protocol | `insufficient_sample` | 170 | 0.343 / 0.359 | 0.363 / 0.244 | 0.123 | 0.506 | 0.006 | — | -0.520 (n=101) | holdout 170 < MIN_HOLDOUT_ROWS=200 |
| jupiter-exchange-solana | `insufficient_sample` | 157 | 0.340 / 0.362 | 0.358 / 0.231 | 0.204 | 0.605 | 0.013 | — | 0.077 (n=109) | holdout 157 < MIN_HOLDOUT_ROWS=200 |
| pepe | `structurally_noisy_retire` | 212 | 0.374 / 0.407 | 0.335 / 0.223 | 0.212 | 0.627 | 0.014 | — | -0.012 (n=154) | calibration_broken=True (max reliability deviation 0.399 >= 0.1); prediction_… |
| render-token | `insufficient_sample` | 124 | 0.332 / 0.359 | 0.374 / 0.251 | 0.258 | 0.653 | 0.040 | — | -0.081 (n=92) | holdout 124 < MIN_HOLDOUT_ROWS=200 |
| worldcoin-wld | `insufficient_sample` | 196 | 0.333 / 0.373 | 0.368 / 0.241 | 0.393 | 0.811 | 0.000 | — | -0.477 (n=165) | holdout 196 < MIN_HOLDOUT_ROWS=200 |
