# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T070042Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:00:42.634874Z`)
- frame rows: `92253` (span_days=`320.566`, bar_gap_rate=`3.3e-05`, bars_source=`candles`)
- candidate: n_train=`88344`, n_holdout=`3885`, threshold_fraction=`0.0080` (horizon=`12` bars, n_features=`50`)
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
| unconditional_beta_baseline (baseline) | beta | 106 | +38.9615% | -1.4562% | 0.6058 | 6.6737 | 0.2134 | n/a (reference) |
| platt_two_stage | platt | 111 | +39.0800% | -1.4562% | 0.6120 | 6.2101 | 0.2030 | no |
| isotonic_two_stage | isotonic | 118 | +38.0219% | -2.0795% | 0.5790 | 5.0375 | 0.2143 | no |

### Variants that did NOT graduate

- `platt_two_stage`
  - holdout.cal_dev_post_calibration=0.6120 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
- `isotonic_two_stage`
  - holdout.cal_dev_post_calibration=0.5790 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 37 | 0.2590 | 0.8649 | 0.6058 |
| [0.30, 0.40) | 37 | 0.3513 | 0.8919 | 0.5406 |
| [0.40, 0.50) | 16 | 0.4548 | 0.9375 | 0.4827 |
| [0.50, 0.60) | 10 | 0.5387 | 0.9000 | 0.3613 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 42 | 0.2451 | 0.8571 | 0.6120 |
| [0.30, 0.40) | 36 | 0.3518 | 0.8889 | 0.5371 |
| [0.40, 0.50) | 12 | 0.4528 | 0.9167 | 0.4638 |
| [0.50, 0.60) | 15 | 0.5406 | 0.9333 | 0.3928 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 41 | 0.2259 | 0.8049 | 0.5790 |
| [0.30, 0.40) | 54 | 0.3635 | 0.9074 | 0.5439 |
| [0.40, 0.50) | 14 | 0.4562 | 0.8571 | 0.4009 |

## Winner: NONE

- status: `no_variant_passed_trustworthy_gate`
- `platt_two_stage`: holdout.cal_dev_post_calibration=0.6120 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
- `isotonic_two_stage`: holdout.cal_dev_post_calibration=0.5790 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
