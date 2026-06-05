# Auto failure-analysis — 2026-04-23T11:09:03.339373+00:00

- **Source report generated_at:** `2026-04-23T10:34:15.657816+00:00`
- **Gate constants:** {"MIN_HOLDOUT_ROWS": 200, "MIN_DIRECTIONAL_ACCURACY": 0.5, "DIRECTIONAL_SHARE_TRADEABLE_TIMEFRAMES": ["1d", "1h", "2h", "5m", "6h"], "CALIBRATION_BROKEN_RELIABILITY_DEV": 0.1, "PREDICTION_COLLAPSE_GAP": 0.15, "PREDICTION_COLLAPSE_TOP_SHARE": 0.85}

> Auto-generated from `models/report.json` after every retrain — no offline re-inference required. For the full hand-run analysis (cadence audit, smallest-first-set, repair plan) run `scripts/compute_failure_metrics.py` + `scripts/render_failure_analysis_md.py` against the persisted artifacts.

## 1. Bucket assignment summary

| Bucket | Count |
|---|---|
| `structurally_noisy_retire` | 2 |

## 2. Bucket × timeframe matrix

| Timeframe | structurally_noisy_retire | total |
|---|---|---|
| 1h | 2 | 2 |

## 3. Per-slice detail

### 1h

| Coin | Bucket | n_holdout | DA / baseline | Brier vs base | collapse_gap | top pred share | net PnL %/trade | reason |
|---|---|---|---|---|---|---|---|---|
| (pooled) | `structurally_noisy_retire` | 1745 | 0.393 / 0.409 | 0.309 / 0.213 | 0.253 | 0.640 | -0.278 (n=316) | calibration_broken=True (max reliability deviation 0.657 >= 0.1); prediction_… |
| jupiter-exchange-solana | `structurally_noisy_retire` | 1745 | 0.393 / 0.409 | 0.309 / 0.213 | 0.169 | 0.556 | -0.195 (n=177) | calibration_broken=True (max reliability deviation 0.874 >= 0.1); prediction_… |
