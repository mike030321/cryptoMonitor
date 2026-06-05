# Auto failure-analysis — 2026-04-29T21:53:39.845677+00:00

- **Source report generated_at:** `2026-04-29T18:27:57.092389+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "MIN_DIRECTIONAL_ACCURACY_PER_TF": {"1d": 0.53}, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 2 |
| `insufficient_sample` | 1 |
| `salvageable_with_better_features_or_labels` | 1 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | salvageable_with_better_features_or_labels | structurally_noisy_retire | total |
|---|---|---|---|---|
| 1d | 1 | 1 | 2 | 4 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 587 | 0.373 / 0.395 | 0.316 / 0.220 | 0.256 | 0.668 | 0.003 | — | -0.395 (n=498) | calibration_broken=True (max reliability deviation 0.809 >= 0.1); prediction_… |
| bonk | `insufficient_sample` | 162 | 0.421 / 0.409 | 0.316 / 0.218 | 0.414 | 0.840 | 0.006 | — | -0.466 (n=148) | holdout 162 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `salvageable_with_better_features_or_labels` | 213 | 0.357 / 0.391 | 0.344 / 0.220 | 0.136 | 0.535 | 0.005 | — | -0.280 (n=150) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| pepe | `structurally_noisy_retire` | 212 | 0.370 / 0.409 | 0.336 / 0.223 | 0.269 | 0.684 | 0.019 | — | -0.099 (n=157) | calibration_broken=True (max reliability deviation 0.673 >= 0.1); prediction_… |
