# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T070527Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:05:27.296145Z`)
- frame rows: `92253` (span_days=`320.566`, bar_gap_rate=`3.3e-05`, bars_source=`candles`)
- candidate: n_train=`88345`, n_holdout=`3884`, threshold_fraction=`0.0080` (horizon=`12` bars, n_features=`50`)
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
| unconditional_beta_baseline (baseline) | beta | 105 | +38.7243% | -1.7266% | 0.5819 | 6.6981 | 0.2147 | n/a (reference) |
| platt_two_stage | platt | 107 | +39.8414% | -1.7266% | 0.8049 | 6.9812 | 0.1929 | no |
| isotonic_two_stage | isotonic | 3884 | -1118.2331% | -1117.9001% | 0.5789 | 0.1321 | 0.1500 | no |

### Variants that did NOT graduate

- `platt_two_stage`
  - holdout.cal_dev_post_calibration=0.8049 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
- `isotonic_two_stage`
  - holdout.cal_dev_post_calibration=0.5789 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
  - holdout.max_drawdown_pct=-1117.9001% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 38 | 0.2602 | 0.8421 | 0.5819 |
| [0.30, 0.40) | 35 | 0.3468 | 0.8857 | 0.5389 |
| [0.40, 0.50) | 16 | 0.4545 | 1.0000 | 0.5455 |
| [0.50, 0.60) | 8 | 0.5351 | 0.8750 | 0.3399 |
| [0.60, 0.70) | 7 | 0.6475 | 1.0000 | 0.3525 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.10, 0.20) | 5 | 0.1951 | 1.0000 | 0.8049 |
| [0.20, 0.30) | 41 | 0.2504 | 0.8293 | 0.5788 |
| [0.30, 0.40) | 28 | 0.3482 | 0.8929 | 0.5447 |
| [0.40, 0.50) | 17 | 0.4618 | 1.0000 | 0.5382 |
| [0.50, 0.60) | 6 | 0.5398 | 0.8333 | 0.2936 |
| [0.60, 0.70) | 7 | 0.6324 | 1.0000 | 0.3676 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.10, 0.20) | 3796 | 0.1501 | 0.4966 | 0.3464 |
| [0.20, 0.30) | 9 | 0.2036 | 0.7778 | 0.5742 |
| [0.30, 0.40) | 58 | 0.3728 | 0.8793 | 0.5065 |
| [0.40, 0.50) | 8 | 0.4211 | 1.0000 | 0.5789 |
| [0.50, 0.60) | 6 | 0.5469 | 1.0000 | 0.4531 |
| [0.60, 0.70) | 5 | 0.6360 | 1.0000 | 0.3640 |

## Winner: NONE

- status: `no_variant_passed_trustworthy_gate`
- `platt_two_stage`: holdout.cal_dev_post_calibration=0.8049 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
- `isotonic_two_stage`: holdout.cal_dev_post_calibration=0.5789 above ceiling 0.2 (HARD gate for calibration_status='trustworthy'); holdout.max_drawdown_pct=-1117.9001% not strictly > -5.0% (would trip DS auto-disable)
