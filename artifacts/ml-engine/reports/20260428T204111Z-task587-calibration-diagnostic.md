# Task #587 — Calibration / decision-threshold diagnostic (20260428T204111Z)
Investigates why task #580 stage-2 evaluation produced `trade_share` of 0.83-1.00 across every (coin, timeframe) slice — far outside the unmodified gate band [0.40, 0.85]. The hypothesis: the search runner trains an unweighted LightGBM and uses raw `argmax(proba) != STABLE` as the trade rule, with no calibration applied to the booster's confidence. With STABLE-class label share in the 15-23% range across timeframes, the booster's argmax rarely lands on STABLE, so trade_share collapses to ~1.0 by construction.
## Method
- Per (coin, timeframe), single time-ordered split: first 75% train, last 25% test.
- Calibration tail: last 20% of train (point-in-time-safe, never overlaps test).
- Calibration: single-scalar temperature scaling fitted on the cal tail (Guo et al. 2017). `inv_T = 1.0` is identity (no calibration).
- Round-trip cost: 0.3000% from `shared/trading-frictions.json`.
- Trade rule swept across neutral-band widths `delta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]`: trade iff `max(P_UP, P_DOWN) > P_STABLE + delta`.

## Per-timeframe label distribution (full snapshot)
| timeframe | n_rows | n_features | DOWN | STABLE | UP | dataset |
|---|---|---|---|---|---|---|
| 6h | 13212 | 38 | 0.462 | 0.154 | 0.384 | `6h_20260428T133439Z.parquet` |
| 1d | 7864 | 38 | 0.413 | 0.234 | 0.354 | `1d_20260428T133304Z.parquet` |

## Per-(coin, timeframe) calibration summary
Raw trade_share is the share of test rows where `argmax(raw_proba) != STABLE` (delta=0). Calibrated trade_share applies single-T temperature scaling first.

| coin | tf | n_test | inv_T | T | raw trade_share | cal trade_share | raw DA | cal DA |
|---|---|---|---|---|---|---|---|---|
| bonk | 6h | 351 | 0.250 | 4.000 | 0.8632 | 0.8632 | 0.4021 | 0.4021 |
| celestia | 6h | 351 | 0.266 | 3.765 | 1.0000 | 1.0000 | 0.4793 | 0.4793 |
| dogwifcoin | 6h | 351 | 0.250 | 4.000 | 0.9943 | 0.9943 | 0.4054 | 0.4054 |
| floki-inu | 6h | 351 | 0.250 | 4.000 | 0.9174 | 0.9174 | 0.4000 | 0.4000 |
| injective-protocol | 6h | 351 | 0.250 | 4.000 | 1.0000 | 1.0000 | 0.4634 | 0.4634 |
| jupiter-exchange-solana | 6h | 351 | 0.250 | 4.000 | 0.9886 | 0.9886 | 0.4653 | 0.4653 |
| pepe | 6h | 351 | 0.254 | 3.942 | 0.9915 | 0.9915 | 0.4765 | 0.4765 |
| render-token | 6h | 351 | 0.271 | 3.690 | 0.8604 | 0.8604 | 0.4899 | 0.4899 |
| sei-network | 6h | 151 | 0.250 | 4.000 | 0.8874 | 0.8874 | 0.3388 | 0.3388 |
| worldcoin-wld | 6h | 351 | 0.250 | 4.000 | 0.8860 | 0.8860 | 0.4139 | 0.4139 |
| bonk | 1d | 202 | 0.280 | 3.571 | 0.9802 | 0.9802 | 0.4295 | 0.4295 |
| celestia | 1d | 219 | 0.250 | 4.000 | 0.8676 | 0.8676 | 0.4036 | 0.4036 |
| dogwifcoin | 1d | 178 | 0.250 | 4.000 | 0.9888 | 0.9888 | 0.4094 | 0.4094 |
| floki-inu | 1d | 267 | 0.250 | 4.000 | 0.8315 | 0.8315 | 0.4339 | 0.4339 |
| injective-protocol | 1d | 212 | 0.250 | 4.000 | 0.8443 | 0.8443 | 0.3882 | 0.3882 |
| jupiter-exchange-solana | 1d | 196 | 0.250 | 4.000 | 0.9949 | 0.9949 | 0.4759 | 0.4759 |
| pepe | 1d | 265 | 0.469 | 2.132 | 0.9623 | 0.9623 | 0.5126 | 0.5126 |
| render-token | 1d | 155 | 0.250 | 4.000 | 0.8387 | 0.8387 | 0.3913 | 0.3913 |
| worldcoin-wld | 1d | 244 | 0.277 | 3.612 | 0.6639 | 0.6639 | 0.3146 | 0.3146 |

## trade_share as a function of neutral-band width (pooled per timeframe)
Pooled across coins by averaging the per-slice trade_share at each delta. The gate band [0.40, 0.85] is the production target.

| timeframe | source | d=0.00 | d=0.05 | d=0.10 | d=0.15 | d=0.20 | d=0.25 |
|---|---|---|---|---|---|---|---|
| 6h | raw | 0.939 | 0.927 | 0.908 | 0.890 | 0.868 | 0.844 |
| 6h | calibrated | 0.939 | 0.857 | 0.728 | 0.555 | 0.352 | 0.180 |
| 1d | raw | 0.886 | 0.861 | 0.833 | 0.807 | 0.767 | 0.721 |
| 1d | calibrated | 0.886 | 0.769 | 0.612 | 0.447 | 0.302 | 0.179 |

## Predicted-prob distribution stats (pooled per timeframe)
Means and quantiles of per-class predicted probabilities, averaged over the per-slice stats. The key signal is `STABLE.mean` for raw vs calibrated: if calibration meaningfully sharpens or flattens P(STABLE), the decision-threshold band moves. The `MARGIN` row is the per-row `max(P_UP,P_DOWN) - P_STABLE` distribution (the quantity the trade rule cuts on).

| timeframe | source | class | mean | p10 | p25 | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| 6h | raw | DOWN | +0.1525 | +0.0640 | +0.0880 | +0.1268 | +0.2022 | +0.2652 | +0.3540 |
| 6h | raw | STABLE | +0.1640 | +0.0493 | +0.0811 | +0.1414 | +0.2171 | +0.3123 | +0.3734 |
| 6h | raw | UP | +0.6835 | +0.4985 | +0.6010 | +0.7022 | +0.7884 | +0.8449 | +0.8769 |
| 6h | raw | MARGIN_DIRECTIONAL_OVER_STABLE | +0.5420 | +0.2408 | +0.4182 | +0.5716 | +0.7102 | +0.8021 | +0.8396 |
| 6h | calibrated | DOWN | +0.2787 | +0.2396 | +0.2544 | +0.2732 | +0.3002 | +0.3246 | +0.3481 |
| 6h | calibrated | STABLE | +0.2836 | +0.2332 | +0.2540 | +0.2816 | +0.3092 | +0.3371 | +0.3537 |
| 6h | calibrated | UP | +0.4376 | +0.3851 | +0.4108 | +0.4381 | +0.4656 | +0.4911 | +0.5065 |
| 6h | calibrated | MARGIN_DIRECTIONAL_OVER_STABLE | +0.1599 | +0.0639 | +0.1124 | +0.1605 | +0.2116 | +0.2566 | +0.2808 |
| 1d | raw | DOWN | +0.2181 | +0.0819 | +0.1177 | +0.1858 | +0.2867 | +0.3916 | +0.4796 |
| 1d | raw | STABLE | +0.1974 | +0.0677 | +0.1063 | +0.1711 | +0.2621 | +0.3703 | +0.4435 |
| 1d | raw | UP | +0.5846 | +0.3598 | +0.4795 | +0.5984 | +0.7068 | +0.7923 | +0.8290 |
| 1d | raw | MARGIN_DIRECTIONAL_OVER_STABLE | +0.4282 | +0.0684 | +0.2762 | +0.4751 | +0.6142 | +0.7166 | +0.7684 |
| 1d | calibrated | DOWN | +0.3022 | +0.2449 | +0.2666 | +0.2975 | +0.3319 | +0.3669 | +0.3887 |
| 1d | calibrated | STABLE | +0.2858 | +0.2295 | +0.2529 | +0.2831 | +0.3139 | +0.3455 | +0.3655 |
| 1d | calibrated | UP | +0.4121 | +0.3433 | +0.3761 | +0.4127 | +0.4489 | +0.4820 | +0.5010 |
| 1d | calibrated | MARGIN_DIRECTIONAL_OVER_STABLE | +0.1398 | +0.0257 | +0.0826 | +0.1444 | +0.2016 | +0.2493 | +0.2733 |

## Headline conclusion
Best (delta, share-of-slices-in-band) per (timeframe, source):

| timeframe | source | best delta | share of slices in [0.40, 0.85] | mean trade_share at best delta |
|---|---|---|---|---|
| 6h | raw | 0.20 | 0.50 | 0.868 |
| 6h | calibrated | 0.05 | 0.50 | 0.857 |
| 1d | raw | 0.20 | 0.67 | 0.767 |
| 1d | calibrated | 0.05 | 0.56 | 0.769 |

## Interpretation
1. If raw trade_share at `delta=0` is uniformly ~1.0, the booster's argmax never picks STABLE under the natural label distribution; this is the task #580 symptom by construction.
2. If calibration alone (single-T) moves trade_share into [0.40, 0.85] for some timeframes at `delta=0`, the gate failure is purely a calibration miss — production already does this on the trainer side, but the search runner did not, so the search verdict was measuring a different surface than what production would deploy.
3. If even after calibration trade_share stays above 0.85 at `delta=0`, a per-slice neutral-band tuning is needed (sweep above), and the diagnostic table shows the band width that lands inside [0.40, 0.85].
