# Auto failure-analysis — 2026-04-25T11:04:49.264695+00:00

- **Source report generated_at:** `2026-04-25T10:33:08.438200+00:00`
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
| (pooled) | `structurally_noisy_retire` | 585 | 0.378 / 0.408 | 0.313 / 0.219 | 0.248 | 0.665 | 0.026 | — | -0.349 (n=428) | calibration_broken=True (max reliability deviation 0.270 >= 0.1); prediction_… |
| bonk | `insufficient_sample` | 161 | 0.416 / 0.393 | 0.317 / 0.218 | 0.398 | 0.832 | 0.000 | — | -0.319 (n=148) | holdout 161 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `salvageable_with_better_features_or_labels` | 213 | 0.353 / 0.375 | 0.353 / 0.220 | 0.113 | 0.512 | 0.019 | — | -0.788 (n=142) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| pepe | `structurally_noisy_retire` | 212 | 0.374 / 0.425 | 0.336 / 0.222 | 0.259 | 0.679 | 0.005 | — | -0.310 (n=156) | calibration_broken=True (max reliability deviation 0.680 >= 0.1); prediction_… |
