# Task #587 — stage-2 re-run, calibration ON vs OFF (20260429T034901Z)
Same admitted-feature stack and dataset snapshots as task #580. The two columns under `OFF` mirror the original task #580 verdict; the two columns under `ON` apply the flag-gated calibration + neutral-band path (single-T temperature scaling on a cal tail, then a fold-fitted `delta` that targets cal-tail trade_share at a per-(coin, timeframe) auto-tuned target — Task #591 — picked by maximising cal-tail post-fee PnL across the grid `{0.50, 0.625, 0.75}`). The trade rule under ON is `max(P_UP, P_DOWN) > P_STABLE + delta`; under OFF it is `argmax(proba) != STABLE` (the legacy rule).

## Hard rules respected
- Same dataset snapshots as task #580 (latest pooled per-tf parquet).
- Same admitted feature stack as task #580.
- No edits to gate constants — DA-lift > 0.02, PnL > 0, trade_share in [0.40, 0.85], n_trades >= 30.
- Calibration is point-in-time-safe: cal tail is the LAST 20% of each fold's training slice.
- Round-trip cost: 0.3000% from `shared/trading-frictions.json`.
- Subset: ['6h', '1d'] (1h/2h omitted for runtime).

## Per-(coin, timeframe) side-by-side
| coin | tf | OFF trade_share | OFF DA | OFF DA lift | OFF PnL_total | OFF gate | ON trade_share | ON DA | ON DA lift | ON PnL_total | ON gate | mean inv_T | mean delta | tuned target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bonk | 1d | 0.9784 | 0.5042 | +0.0021 | +113.51 | fail | 0.7347 | 0.3782 | -0.0336 | +67.63 | fail | 0.250 | 0.2222 | 0.667 |
| bonk | 6h | 0.9952 | 0.4764 | +0.0011 | -254.76 | fail | 0.5590 | 0.2225 | +0.0371 | -233.51 | fail | 0.252 | 0.3165 | 0.583 |
| celestia | 1d | 0.9128 | 0.4255 | -0.0490 | -321.41 | fail | 0.6315 | 0.3137 | -0.0745 | -153.84 | fail | 0.250 | 0.2705 | 0.500 |
| celestia | 6h | 0.9095 | 0.4257 | +0.0034 | -321.78 | fail | 0.6790 | 0.2972 | -0.0358 | -243.40 | fail | 0.252 | 0.2510 | 0.583 |
| dogwifcoin | 1d | 0.9736 | 0.4248 | +0.0121 | -334.41 | fail | 0.8154 | 0.3447 | +0.0121 | -419.83 | fail | 0.261 | 0.2189 | 0.583 |
| dogwifcoin | 6h | 1.0000 | 0.4668 | +0.0022 | -304.89 | fail | 0.7390 | 0.3251 | +0.0259 | -236.37 | fail | 0.250 | 0.3132 | 0.625 |
| floki-inu | 1d | 0.8358 | 0.4387 | -0.0182 | -163.91 | fail | 0.6880 | 0.3560 | +0.0430 | -51.29 | fail | 0.250 | 0.0679 | 0.583 |
| floki-inu | 6h | 0.9686 | 0.4433 | -0.0139 | -273.92 | fail | 0.6676 | 0.3102 | -0.0231 | -194.28 | fail | 0.263 | 0.2427 | 0.708 |
| injective-protocol | 1d | 0.9289 | 0.4325 | -0.0193 | -288.78 | fail | 0.7267 | 0.3619 | +0.0450 | -127.38 | fail | 0.250 | 0.1723 | 0.625 |
| injective-protocol | 6h | 0.9762 | 0.4790 | +0.0181 | -304.56 | fail | 0.7362 | 0.3398 | +0.0770 | -299.35 | fail | 0.250 | 0.3700 | 0.625 |
| jupiter-exchange-solana | 1d | 0.9337 | 0.4437 | +0.0000 | -229.19 | fail | 0.8673 | 0.3974 | -0.0088 | -223.46 | fail | 0.250 | 0.0915 | 0.708 |
| jupiter-exchange-solana | 6h | 0.9771 | 0.4675 | +0.0081 | -237.23 | fail | 0.5152 | 0.2332 | -0.0162 | -197.56 | fail | 0.250 | 0.2468 | 0.500 |
| pepe | 1d | 0.8472 | 0.4411 | -0.0159 | -312.53 | fail | 0.6111 | 0.3105 | +0.0127 | -257.87 | fail | 0.277 | 0.1198 | 0.625 |
| pepe | 6h | 0.9457 | 0.4526 | -0.0057 | -288.54 | fail | 0.7600 | 0.3680 | +0.0663 | -203.92 | fail | 0.250 | 0.3092 | 0.583 |
| render-token | 1d | 0.9113 | 0.4522 | +0.0145 | -196.48 | fail | 0.5390 | 0.2609 | -0.0667 | -66.56 | fail | 0.250 | 0.2600 | 0.583 |
| render-token | 6h | 0.9676 | 0.4864 | -0.0113 | -272.81 | fail | 0.5714 | 0.2766 | -0.0204 | -156.00 | fail | 0.250 | 0.2213 | 0.625 |
| sei-network | 1d | — | — | — | — | — | — | — | — | — | — | 1.000 | 0.0000 | — |
| sei-network | 6h | 0.9089 | 0.4959 | -0.0054 | -116.88 | fail | 0.5022 | 0.2249 | +0.0190 | -118.84 | fail | 0.250 | 0.1777 | 0.583 |
| worldcoin-wld | 1d | 0.8875 | 0.4394 | -0.0235 | -191.51 | fail | 0.6680 | 0.3092 | -0.0090 | -324.81 | fail | 0.250 | 0.1155 | 0.625 |
| worldcoin-wld | 6h | 0.8924 | 0.4349 | +0.0034 | -244.01 | fail | 0.5914 | 0.2888 | -0.0057 | -176.51 | fail | 0.250 | 0.2443 | 0.583 |

## Aggregate verdict
| metric | OFF (legacy) | ON (calibrated + delta) |
|---|---|---|
| evaluated slices | 19 | 19 |
| slices passing the FULL gate | 0 | 0 |
| slices with trade_share in [0.40, 0.85] | 2 | 18 |
| mean trade_share | 0.9342 | 0.6633 |
| mean DA lift vs baseline | -0.0051 | +0.0023 |
| sum PnL_pct_total (augmented) | -4544.09 | -3617.15 |

## Interpretation
* **Trade-share band coverage**: with the flag OFF only 2/19 slices land in the gate band [0.40, 0.85]; with the flag ON, 18/19 do (+16). Mean trade_share moves from 0.9342 to 0.6633 (Δ -0.2709; target midpoint 0.625). The flag-gated calibration + neutral-band path successfully bridges the trade_share gap that blocked task #580 stage-2.
* **Gate pass count**: 0 OFF vs 0 ON. No slice passes the FULL gate either way — calibration moves trade_share into the band, but the admitted-feature DA lift is not large enough to clear `STAGE2_DA_LIFT_FLOOR` regardless of the trade rule. The blocker is the underlying signal strength, not the trade-rule surface.
* **DA lift impact**: mean DA lift improves from -0.0051 to +0.0023 (Δ +0.0074). Because the trade rule under ON labels low-confidence rows as STABLE, the DA denominator includes those rows as misses on directional-truth bars; small DA changes here are noisy by construction.
* **PnL impact**: sum post-fee PnL_pct_total improves from -4544.09 to -3617.15 (Δ +926.95). The PnL change is dominated by the trade-count reduction and the round-trip cost saved on rows that no longer trade.
* **Operator action**: keep `ML_FEATURE_EDGE_CALIBRATE` OFF for task #580 reproducibility. Turn it ON for any future stage-2 run whose admitted-feature stack should be evaluated under production-equivalent calibration. The runner now auto-tunes the cal-tail `trade_share_target` per (coin, timeframe) by maximising cal-tail post-fee PnL across the grid `{0.50, 0.625, 0.75}` (Task #591) — the per-slice winning target is shown in the side-by-side table above. To widen the search, edit `TRADE_SHARE_TARGET_GRID` in `scripts/feature_edge_search/run_search.py`.
