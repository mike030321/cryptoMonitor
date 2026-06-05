# Task #580 — Feature edge search verdict (20260428T195627Z)
Successor to cancelled task #558. Searches deterministic, real-data-only candidate features for at least one (coin, timeframe) slice that earns `role=trade` under the **unmodified** promotion gate.
## Hard rules respected
- No edits to `app/training/registry.py`, gate constants, or role JSON.
- `shared/forbidden-features.json` cross-checked: prefixes match registry tuple ('news_', 'llm_', 'gpt_', 'sentiment_', 'ai_').
- Every candidate is point-in-time (`max_lookforward == 0`) and computed only from columns already in the persisted dataset snapshot.
- Round-trip cost used: 0.3000% (sourced from `shared/trading-frictions.json`).
- Stage-1 admit rule: >= 2 of 3 OOS folds positive in BOTH (directional_accuracy delta, post_fee_pnl_pct_total delta) vs same-fold baseline.
- Stage-2 gate: DA > baseline_DA + 0.02, post_fee_pnl_pct_total > 0.0, trade_share in [0.4, 0.85], n_trades >= 30.

## Headline answer — is there ≥1 trade-worthy slice?
**NO.** Every (coin, timeframe) slice failed at least one gate condition under the stacked stage-1-admitted feature set. See per-slice table below.

## Stage 1 — per-candidate ablation (pooled per timeframe)
Per-fold deltas vs the same-fold baseline (LightGBM trained on the pooled per-coin frame for each timeframe). 'admitted' = admitted to stage 2.

| candidate | bucket | tf | folds+ DA | folds+ PnL | folds+ both | admitted |
|---|---|---|---|---|---|---|
| vol_of_vol_30 | volatility_regime | 1h | 2/3 | 2/3 | 2/3 | **yes** |
| realizedVol_log | price_derived | 1h | 1/3 | 2/3 | 1/3 | no |
| ret1_squared | price_derived | 1h | 2/3 | 3/3 | 2/3 | **yes** |
| ret5_minus_ret10 | price_derived | 1h | 1/3 | 2/3 | 1/3 | no |
| rsi14_centered_squared | price_derived | 1h | 1/3 | 1/3 | 1/3 | no |
| bb_pctb_extreme | price_derived | 1h | 2/3 | 1/3 | 1/3 | no |
| macd_hist_norm_atr | price_derived | 1h | 1/3 | 2/3 | 1/3 | no |
| vol_zscore60_squared | volatility_regime | 1h | 1/3 | 3/3 | 1/3 | no |
| ema_spread_per_atr | price_derived | 1h | 2/3 | 2/3 | 1/3 | no |
| atr_pct_zscore_60 | volatility_regime | 1h | 2/3 | 2/3 | 2/3 | **yes** |
| drawdown_30 | price_derived | 1h | 3/3 | 3/3 | 3/3 | **yes** |
| btc_lead_x_self_ret | cross_coin_lead_lag | 1h | 0/3 | 0/3 | 0/3 | no |
| eth_lead_minus_btc_lead | cross_coin_lead_lag | 1h | 0/3 | 0/3 | 0/3 | no |
| macd_signal_cross_strength | price_derived | 1h | 2/3 | 2/3 | 1/3 | no |
| ret1_x_volZ60 | price_derived | 1h | 3/3 | 2/3 | 2/3 | **yes** |
| log_liquidations_self | liquidity_funding_oi_spread | 1h | 0/3 | 0/3 | 0/3 | no |
| vol_of_vol_30 | volatility_regime | 2h | 1/3 | 0/3 | 0/3 | no |
| realizedVol_log | price_derived | 2h | 2/3 | 2/3 | 2/3 | **yes** |
| ret1_squared | price_derived | 2h | 2/3 | 2/3 | 1/3 | no |
| ret5_minus_ret10 | price_derived | 2h | 1/3 | 1/3 | 1/3 | no |
| rsi14_centered_squared | price_derived | 2h | 2/3 | 1/3 | 1/3 | no |
| bb_pctb_extreme | price_derived | 2h | 2/3 | 1/3 | 1/3 | no |
| macd_hist_norm_atr | price_derived | 2h | 1/3 | 2/3 | 1/3 | no |
| vol_zscore60_squared | volatility_regime | 2h | — | — | — | error: 'volZScore60' |
| ema_spread_per_atr | price_derived | 2h | 3/3 | 0/3 | 0/3 | no |
| atr_pct_zscore_60 | volatility_regime | 2h | 2/3 | 2/3 | 1/3 | no |
| drawdown_30 | price_derived | 2h | 2/3 | 1/3 | 1/3 | no |
| btc_lead_x_self_ret | cross_coin_lead_lag | 2h | 0/3 | 0/3 | 0/3 | no |
| eth_lead_minus_btc_lead | cross_coin_lead_lag | 2h | 0/3 | 0/3 | 0/3 | no |
| macd_signal_cross_strength | price_derived | 2h | 2/3 | 1/3 | 1/3 | no |
| ret1_x_volZ60 | price_derived | 2h | — | — | — | error: 'volZScore60' |
| log_liquidations_self | liquidity_funding_oi_spread | 2h | 0/3 | 0/3 | 0/3 | no |
| vol_of_vol_30 | volatility_regime | 6h | 2/3 | 2/3 | 1/3 | no |
| realizedVol_log | price_derived | 6h | 1/3 | 3/3 | 1/3 | no |
| ret1_squared | price_derived | 6h | 2/3 | 3/3 | 2/3 | **yes** |
| ret5_minus_ret10 | price_derived | 6h | 1/3 | 2/3 | 0/3 | no |
| rsi14_centered_squared | price_derived | 6h | 3/3 | 2/3 | 2/3 | **yes** |
| bb_pctb_extreme | price_derived | 6h | 3/3 | 2/3 | 2/3 | **yes** |
| macd_hist_norm_atr | price_derived | 6h | 1/3 | 1/3 | 1/3 | no |
| vol_zscore60_squared | volatility_regime | 6h | 0/3 | 2/3 | 0/3 | no |
| ema_spread_per_atr | price_derived | 6h | 3/3 | 1/3 | 1/3 | no |
| atr_pct_zscore_60 | volatility_regime | 6h | 2/3 | 1/3 | 1/3 | no |
| drawdown_30 | price_derived | 6h | 2/3 | 1/3 | 1/3 | no |
| btc_lead_x_self_ret | cross_coin_lead_lag | 6h | 0/3 | 0/3 | 0/3 | no |
| eth_lead_minus_btc_lead | cross_coin_lead_lag | 6h | 0/3 | 0/3 | 0/3 | no |
| macd_signal_cross_strength | price_derived | 6h | 2/3 | 3/3 | 2/3 | **yes** |
| ret1_x_volZ60 | price_derived | 6h | 0/3 | 3/3 | 0/3 | no |
| log_liquidations_self | liquidity_funding_oi_spread | 6h | 0/3 | 0/3 | 0/3 | no |
| vol_of_vol_30 | volatility_regime | 1d | 2/3 | 1/3 | 1/3 | no |
| realizedVol_log | price_derived | 1d | 2/3 | 2/3 | 2/3 | **yes** |
| ret1_squared | price_derived | 1d | 2/3 | 1/3 | 1/3 | no |
| ret5_minus_ret10 | price_derived | 1d | 2/3 | 2/3 | 2/3 | **yes** |
| rsi14_centered_squared | price_derived | 1d | 3/3 | 3/3 | 3/3 | **yes** |
| bb_pctb_extreme | price_derived | 1d | 2/3 | 2/3 | 2/3 | **yes** |
| macd_hist_norm_atr | price_derived | 1d | 1/3 | 2/3 | 1/3 | no |
| vol_zscore60_squared | volatility_regime | 1d | 2/3 | 1/3 | 1/3 | no |
| ema_spread_per_atr | price_derived | 1d | 3/3 | 3/3 | 3/3 | **yes** |
| atr_pct_zscore_60 | volatility_regime | 1d | 2/3 | 2/3 | 2/3 | **yes** |
| drawdown_30 | price_derived | 1d | 3/3 | 2/3 | 2/3 | **yes** |
| btc_lead_x_self_ret | cross_coin_lead_lag | 1d | 0/3 | 0/3 | 0/3 | no |
| eth_lead_minus_btc_lead | cross_coin_lead_lag | 1d | 0/3 | 0/3 | 0/3 | no |
| macd_signal_cross_strength | price_derived | 1d | 2/3 | 2/3 | 2/3 | **yes** |
| ret1_x_volZ60 | price_derived | 1d | 1/3 | 1/3 | 1/3 | no |
| log_liquidations_self | liquidity_funding_oi_spread | 1d | 0/3 | 0/3 | 0/3 | no |

## Stage 1 — per-timeframe baseline reference
| timeframe | n_rows | base_features | mean fold DA | sum fold pnl_pct |
|---|---|---|---|---|
| 1h | 77364 | 38 | 0.4944 | -16775.40 |
| 2h | 38511 | 37 | 0.4993 | -7256.91 |
| 6h | 13212 | 38 | 0.5090 | -2075.00 |
| 1d | 7864 | 38 | 0.4771 | -1616.88 |

## Stage 2 — per-(coin, timeframe) gate evaluation (stacked admitted features)
Admitted-for-stacking set: `['atr_pct_zscore_60', 'bb_pctb_extreme', 'drawdown_30', 'ema_spread_per_atr', 'macd_signal_cross_strength', 'realizedVol_log', 'ret1_squared', 'ret1_x_volZ60', 'ret5_minus_ret10', 'rsi14_centered_squared', 'vol_of_vol_30']`.

| coin | tf | n_rows | base DA | aug DA | DA lift | base pnl_pct_total | aug pnl_pct_total | aug trade_share | aug n_trades | gate_pass |
|---|---|---|---|---|---|---|---|---|---|---|
| bonk | 1h | 8596 | 0.4776 | 0.4736 | -0.0041 | -1862.45 | -1901.83 | 0.9698 | 6252 | fail |
| celestia | 1h | 8596 | 0.4765 | 0.4783 | +0.0018 | -1845.13 | -1926.66 | 0.9750 | 6286 | fail |
| dogwifcoin | 1h | 8596 | 0.4755 | 0.4554 | -0.0201 | -1865.14 | -1855.67 | 0.9245 | 5960 | fail |
| floki-inu | 1h | 8596 | 0.4596 | 0.4525 | -0.0071 | -1797.65 | -1835.93 | 0.9316 | 6006 | fail |
| injective-protocol | 1h | 8596 | 0.4846 | 0.4855 | +0.0008 | -1899.26 | -1872.13 | 0.9536 | 6148 | fail |
| jupiter-exchange-solana | 1h | 8596 | 0.4647 | 0.4822 | +0.0175 | -1657.13 | -1789.90 | 0.9483 | 6114 | fail |
| pepe | 1h | 8596 | 0.4441 | 0.4562 | +0.0121 | -1708.38 | -1691.93 | 0.9060 | 5841 | fail |
| render-token | 1h | 8596 | 0.4753 | 0.4856 | +0.0103 | -1866.25 | -1839.50 | 0.9549 | 6156 | fail |
| worldcoin-wld | 1h | 8596 | 0.4450 | 0.4605 | +0.0155 | -1730.15 | -1742.21 | 0.9088 | 5859 | fail |
| bonk | 2h | 4279 | 0.4708 | 0.4716 | +0.0008 | -879.90 | -887.84 | 0.9838 | 3155 | fail |
| celestia | 2h | 4279 | 0.4618 | 0.4656 | +0.0038 | -999.89 | -1019.03 | 0.9903 | 3176 | fail |
| dogwifcoin | 2h | 4279 | 0.4865 | 0.4862 | -0.0004 | -913.60 | -889.36 | 0.9935 | 3186 | fail |
| floki-inu | 2h | 4279 | 0.4737 | 0.4582 | -0.0156 | -828.52 | -956.43 | 0.9800 | 3143 | fail |
| injective-protocol | 2h | 4279 | 0.4737 | 0.4816 | +0.0079 | -1040.19 | -982.94 | 0.9925 | 3183 | fail |
| jupiter-exchange-solana | 2h | 4279 | 0.5016 | 0.4786 | -0.0229 | -783.39 | -753.85 | 0.9370 | 3005 | fail |
| pepe | 2h | 4279 | 0.4530 | 0.4490 | -0.0039 | -877.66 | -898.79 | 0.9311 | 2986 | fail |
| render-token | 2h | 4279 | 0.4520 | 0.4589 | +0.0069 | -915.95 | -988.17 | 0.9747 | 3126 | fail |
| worldcoin-wld | 2h | 4279 | 0.5006 | 0.4920 | -0.0085 | -947.09 | -989.11 | 0.9872 | 3166 | fail |
| bonk | 6h | 1401 | 0.4528 | 0.4618 | +0.0090 | -307.03 | -332.11 | 0.9933 | 1043 | fail |
| celestia | 6h | 1401 | 0.4056 | 0.4223 | +0.0168 | -214.96 | -196.38 | 0.9248 | 971 | fail |
| dogwifcoin | 6h | 1401 | 0.4634 | 0.4657 | +0.0022 | -269.47 | -304.96 | 1.0000 | 1050 | fail |
| floki-inu | 6h | 1401 | 0.4532 | 0.4301 | -0.0231 | -208.10 | -256.56 | 0.9419 | 989 | fail |
| injective-protocol | 6h | 1401 | 0.4553 | 0.4689 | +0.0136 | -414.43 | -288.61 | 0.9895 | 1039 | fail |
| jupiter-exchange-solana | 6h | 1401 | 0.4385 | 0.4304 | -0.0081 | -278.70 | -363.68 | 0.9543 | 1002 | fail |
| pepe | 6h | 1401 | 0.4571 | 0.4606 | +0.0034 | -303.59 | -322.17 | 0.9705 | 1019 | fail |
| render-token | 6h | 1401 | 0.4955 | 0.5023 | +0.0068 | -314.75 | -215.97 | 0.9648 | 1013 | fail |
| sei-network | 6h | 603 | 0.5230 | 0.4770 | -0.0461 | -126.27 | -116.04 | 0.8956 | 403 | fail |
| worldcoin-wld | 6h | 1401 | 0.4066 | 0.4292 | +0.0227 | -251.25 | -337.74 | 0.8981 | 943 | fail |
| bonk | 1d | 807 | 0.5021 | 0.5042 | +0.0021 | -65.63 | +113.51 | 0.9784 | 590 | fail |
| celestia | 1d | 875 | 0.4745 | 0.4255 | -0.0490 | -250.62 | -321.41 | 0.9128 | 597 | fail |
| dogwifcoin | 1d | 709 | 0.4126 | 0.4248 | +0.0121 | -466.40 | -334.41 | 0.9736 | 517 | fail |
| floki-inu | 1d | 1065 | 0.4570 | 0.4387 | -0.0182 | -317.77 | -163.91 | 0.8358 | 667 | fail |
| injective-protocol | 1d | 846 | 0.4518 | 0.4325 | -0.0193 | -231.53 | -288.78 | 0.9289 | 588 | fail |
| jupiter-exchange-solana | 1d | 784 | 0.4437 | 0.4437 | +0.0000 | -184.16 | -229.19 | 0.9337 | 549 | fail |
| pepe | 1d | 1059 | 0.4570 | 0.4411 | -0.0159 | -389.84 | -312.53 | 0.8472 | 671 | fail |
| render-token | 1d | 618 | 0.4377 | 0.4522 | +0.0145 | -159.14 | -196.48 | 0.9113 | 421 | fail |
| sei-network | 1d | 126 | — | — | — | — | — | — | — | skipped: fewer than 200 rows |
| worldcoin-wld | 1d | 975 | 0.4629 | 0.4394 | -0.0235 | -169.31 | -191.51 | 0.8875 | 647 | fail |

## Per-candidate verdicts
| candidate | bucket | timeframes admitted (stage 1) | verdict |
|---|---|---|---|
| vol_of_vol_30 | volatility_regime | 1h | **keep_observing** |
| realizedVol_log | price_derived | 2h,1d | **keep_observing** |
| ret1_squared | price_derived | 1h,6h | **keep_observing** |
| ret5_minus_ret10 | price_derived | 1d | **keep_observing** |
| rsi14_centered_squared | price_derived | 6h,1d | **keep_observing** |
| bb_pctb_extreme | price_derived | 6h,1d | **keep_observing** |
| macd_hist_norm_atr | price_derived | — | **reject** |
| vol_zscore60_squared | volatility_regime | — | **reject** |
| ema_spread_per_atr | price_derived | 1d | **keep_observing** |
| atr_pct_zscore_60 | volatility_regime | 1h,1d | **keep_observing** |
| drawdown_30 | price_derived | 1h,1d | **keep_observing** |
| btc_lead_x_self_ret | cross_coin_lead_lag | — | **reject** |
| eth_lead_minus_btc_lead | cross_coin_lead_lag | — | **reject** |
| macd_signal_cross_strength | price_derived | 6h,1d | **keep_observing** |
| ret1_x_volZ60 | price_derived | 1h | **keep_observing** |
| log_liquidations_self | liquidity_funding_oi_spread | — | **reject** |

## Out of scope (reaffirmed)
- This task does NOT modify `FEATURE_COLUMNS`, gate floors, or role JSON.
- Promotion of any `keep_observing` candidate into the production schema is a separate downstream task.
- Re-running the full training campaign is owned by the existing rerun task.
