# Auto failure-analysis — 2026-04-24T08:29:02.639422+00:00

- **Source report generated_at:** `2026-04-24T08:28:54.893032+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `insufficient_sample` | 1 |

## 2. Bucket × timeframe matrix

| Timeframe | insufficient_sample | total |
|---|---|---|
| 1h | 1 | 1 |

## 3. Per-slice detail

### 1h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | share near prior | contam | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| (pooled) | `insufficient_sample` | 0 | — / — | — / — | — | — | — | — | — | status=trained (holdout=0); not enough labeled rows to train |
