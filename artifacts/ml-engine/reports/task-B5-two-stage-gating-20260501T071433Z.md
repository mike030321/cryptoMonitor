# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T071433Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:14:33.820247Z`)
- frame rows: `92253` (span_days=`320.566`, bar_gap_rate=`3.3e-05`, bars_source=`candles`)
- candidate: n_train=`88346`, n_holdout=`3883`, threshold_fraction=`0.0080` (horizon=`12` bars, n_features=`50`)
- margin_fraction held constant at `0.005` (B4 winner — B5 isolates the calibration axis)

## Selection rule (trustworthy graduation gate)

- holdout `cal_dev_post_calibration <= 0.2` (HARD gate — required for `calibration_status='trustworthy'`)
- holdout `max_drawdown_pct > -5.0%` (matches DS auto-disable floor)
- holdout `n_trades >= 10`
- 10-paper-proof rollout (first 10 fired bars sized at the DS `0.50%` pin) does NOT trip the DS drawdown floor
- tie-break: lowest holdout cal_dev; ties break by highest profit_factor then highest n_trades

## Per-variant holdout metrics

| variant | method | n_trades | net_pnl% | max_dd% | cal_dev | profit_factor | tau | trustworthy? |
|:--|:--|---:|---:|---:|---:|---:|---:|:--|
| unconditional_beta_baseline (baseline) | beta | 103 | +41.9258% | -2.1691% | 0.6277 | 8.1321 | 0.2101 | n/a (reference) |
| platt_two_stage | platt | 104 | +42.8523% | -1.4697% | 0.2149 | 8.9971 | 0.7792 | no |
| isotonic_two_stage | isotonic | 3883 | -1068.9146% | -1092.7048% | 0.5052 | 0.1444 | 0.0000 | no |

### Variants that did NOT graduate

- `platt_two_stage`
  - holdout.cal_dev_post_calibration=0.2149 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
- `isotonic_two_stage`
  - holdout.cal_dev_post_calibration=0.5052 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
  - holdout.max_drawdown_pct=-1092.7048% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 33 | 0.2511 | 0.8788 | 0.6277 |
| [0.30, 0.40) | 29 | 0.3527 | 0.8621 | 0.5093 |
| [0.40, 0.50) | 20 | 0.4413 | 0.9500 | 0.5087 |
| [0.50, 0.60) | 14 | 0.5408 | 0.9286 | 0.3878 |
| [0.60, 0.70) | 6 | 0.6413 | 1.0000 | 0.3587 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.70, 0.80) | 7 | 0.7851 | 1.0000 | 0.2149 |
| [0.80, 0.90) | 40 | 0.8460 | 0.8750 | 0.0290 |
| [0.90, 1.00) | 57 | 0.9361 | 0.9298 | 0.0063 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.00, 0.10) | 3781 | 0.0000 | 0.5052 | 0.5052 |
| [0.70, 0.80) | 6 | 0.7836 | 0.6667 | 0.1169 |
| [0.80, 0.90) | 55 | 0.8725 | 0.9091 | 0.0366 |
| [0.90, 1.00) | 39 | 0.9608 | 0.9487 | 0.0121 |

## Winner: NONE

- status: `no_variant_passed_trustworthy_gate`
- `platt_two_stage`: holdout.cal_dev_post_calibration=0.2149 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
- `isotonic_two_stage`: holdout.cal_dev_post_calibration=0.5052 above ceiling 0.2 (HARD gate for calibration_status='trustworthy'); holdout.max_drawdown_pct=-1092.7048% not strictly > -5.0% (would trip DS auto-disable)
