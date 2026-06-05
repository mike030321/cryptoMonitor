# Auto failure-analysis — 2026-04-29T15:41:34.080607+00:00

- **Source report generated_at:** `2026-04-29T12:52:13.050776+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "MIN_DIRECTIONAL_ACCURACY_PER_TF": {"1d": 0.53}, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 6 |
| `salvageable_with_better_features_or_labels` | 3 |

## 2. Bucket × timeframe matrix

| Timeframe | salvageable_with_better_features_or_labels | structurally_noisy_retire | total |
|---|---|---|---|
| 6h | 3 | 6 | 9 |

## 3. Per-slice detail

### 6h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `salvageable_with_better_features_or_labels` | 2236 | 0.418 / 0.437 | 0.316 / 0.210 | 0.147 | 0.581 | 0.012 | — | -0.332 (n=1570) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| bonk | `structurally_noisy_retire` | 280 | 0.411 / 0.451 | 0.334 / 0.210 | 0.175 | 0.618 | 0.011 | — | 0.034 (n=243) | calibration_broken=True (max reliability deviation 0.320 >= 0.1); prediction_… |
| celestia | `structurally_noisy_retire` | 280 | 0.397 / 0.426 | 0.349 / 0.218 | 0.464 | 0.896 | 0.014 | — | -0.401 (n=262) | calibration_broken=True (max reliability deviation 0.832 >= 0.1); prediction_… |
| dogwifcoin | `salvageable_with_better_features_or_labels` | 280 | 0.407 / 0.439 | 0.341 / 0.219 | 0.068 | 0.539 | 0.043 | — | -0.082 (n=210) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| floki-inu | `structurally_noisy_retire` | 280 | 0.373 / 0.447 | 0.353 / 0.210 | 0.500 | 0.911 | 0.029 | — | -0.312 (n=256) | calibration_broken=True (max reliability deviation 0.414 >= 0.1); prediction_… |
| injective-protocol | `structurally_noisy_retire` | 280 | 0.422 / 0.412 | 0.330 / 0.221 | 0.443 | 0.864 | 0.007 | — | -0.346 (n=250) | calibration_broken=True (max reliability deviation 0.832 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 280 | 0.386 / 0.409 | 0.351 / 0.224 | 0.232 | 0.657 | 0.014 | — | -0.298 (n=225) | calibration_broken=True (max reliability deviation 0.546 >= 0.1); prediction_… |
| pepe | `salvageable_with_better_features_or_labels` | 280 | 0.403 / 0.434 | 0.348 / 0.212 | -0.107 | 0.350 | 0.000 | — | -0.212 (n=164) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| render-token | `structurally_noisy_retire` | 280 | 0.431 / 0.423 | 0.317 / 0.214 | 0.286 | 0.700 | 0.011 | — | -0.377 (n=243) | calibration_broken=True (max reliability deviation 0.626 >= 0.1); prediction_… |
