# Auto failure-analysis — 2026-04-25T10:44:09.964969+00:00

- **Source report generated_at:** `2026-04-25T10:40:06.117403+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "MIN_DIRECTIONAL_ACCURACY_PER_TF": {"1d": 0.53}, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 3 |
| `insufficient_sample` | 1 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | structurally_noisy_retire | total |
|---|---|---|---|
| 1d | 1 | 3 | 4 |

## 3. Per-slice detail

### 1d

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 585 | 0.380 / 0.408 | 0.323 / 0.219 | 0.246 | 0.665 | 0.026 | — | -0.352 (n=428) | calibration_broken=True (max reliability deviation 0.270 >= 0.1); prediction_… |
| bonk | `insufficient_sample` | 161 | 0.431 / 0.393 | 0.315 / 0.218 | 0.410 | 0.845 | 0.000 | — | -0.756 (n=153) | holdout 161 < MIN_HOLDOUT_ROWS=200 |
| floki-inu | `structurally_noisy_retire` | 213 | 0.365 / 0.375 | 0.356 / 0.220 | 0.183 | 0.587 | 0.000 | — | -0.736 (n=163) | calibration_broken=True (max reliability deviation 0.851 >= 0.1); prediction_… |
| pepe | `structurally_noisy_retire` | 212 | 0.377 / 0.425 | 0.351 / 0.222 | 0.269 | 0.689 | 0.005 | — | -0.346 (n=171) | calibration_broken=True (max reliability deviation 0.597 >= 0.1); prediction_… |
