# Auto failure-analysis — 2026-04-29T13:59:25.328833+00:00

- **Source report generated_at:** `2026-04-29T08:16:06.865468+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "MIN_DIRECTIONAL_ACCURACY_PER_TF": {"1d": 0.53}, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `promoted_baseline_served` | 4 |
| `salvageable_with_better_features_or_labels` | 1 |

## 2. Bucket × timeframe matrix

| Timeframe | promoted_baseline_served | salvageable_with_better_features_or_labels | total |
|---|---|---|---|
| 5m | 4 | 1 | 5 |

## 3. Per-slice detail

### 5m

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `promoted_baseline_served` | 73477 | 0.587 / 0.587 | 0.179 / 0.179 | 0.272 | 0.929 | 0.042 | — | -0.280 (n=2282) | served=baseline; baseline DA 0.587 >= 0.5 AND holdout 73477 >= 200; booster C… |
| bonk | `promoted_baseline_served` | 18384 | 0.663 / 0.663 | 0.158 / 0.158 | 0.242 | 0.990 | 0.035 | — | -0.322 (n=106) | served=baseline; baseline DA 0.663 >= 0.5 AND holdout 18384 >= 200; booster C… |
| celestia | `promoted_baseline_served` | 18342 | 0.652 / 0.652 | 0.163 / 0.163 | 0.274 | 0.980 | 0.066 | — | -0.317 (n=149) | served=baseline; baseline DA 0.652 >= 0.5 AND holdout 18342 >= 200; booster C… |
| jupiter-exchange-solana | `salvageable_with_better_features_or_labels` | 18435 | 0.411 / 0.440 | 0.219 / 0.213 | 0.081 | 0.560 | 0.051 | — | -0.296 (n=5709) | red gate but signal remaining — calibration_broken=True, prediction_collapse=… |
| pepe | `promoted_baseline_served` | 18316 | 0.591 / 0.591 | 0.183 / 0.183 | 0.274 | 0.972 | 0.020 | — | -0.262 (n=330) | served=baseline; baseline DA 0.591 >= 0.5 AND holdout 18316 >= 200; booster C… |
