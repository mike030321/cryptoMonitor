# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T071822Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:18:22.836665Z`)
- frame rows: `92253` (span_days=`320.566`, bar_gap_rate=`3.3e-05`, bars_source=`candles`)
- candidate: n_train=`88347`, n_holdout=`3882`, threshold_fraction=`0.0080` (horizon=`12` bars, n_features=`50`)
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
| unconditional_beta_baseline (baseline) | beta | 123 | +41.0646% | -1.5301% | 0.5659 | 5.7724 | 0.2185 | n/a (reference) |
| platt_two_stage | platt | 125 | +42.6239% | -1.8756% | 0.0296 | 5.2960 | 0.8279 | TRUSTWORTHY |
| isotonic_two_stage | isotonic | 3882 | -1062.5913% | -1087.1566% | 0.0470 | 0.1459 | 0.5000 | no |

### Variants that did NOT graduate

- `isotonic_two_stage`
  - holdout.max_drawdown_pct=-1087.1566% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 37 | 0.2518 | 0.8108 | 0.5591 |
| [0.30, 0.40) | 40 | 0.3591 | 0.9250 | 0.5659 |
| [0.40, 0.50) | 25 | 0.4429 | 0.8800 | 0.4371 |
| [0.50, 0.60) | 15 | 0.5585 | 0.9333 | 0.3748 |
| [0.60, 0.70) | 6 | 0.6402 | 1.0000 | 0.3598 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.80, 0.90) | 54 | 0.8611 | 0.8704 | 0.0093 |
| [0.90, 1.00) | 71 | 0.9310 | 0.9014 | 0.0296 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.50, 0.60) | 3763 | 0.5000 | 0.5054 | 0.0054 |
| [0.80, 0.90) | 56 | 0.8637 | 0.9107 | 0.0470 |
| [0.90, 1.00) | 63 | 0.9355 | 0.8889 | 0.0466 |

## Winner

- variant: `platt_two_stage` (calibration_method=`platt`)
- abstain τ: `0.827919`
- holdout: n_trades=`125` net_pnl=`42.6239%` max_dd=`-1.8756%` cal_dev=`0.0296` profit_factor=`5.2960`

### 10 paper-proof rollout (DS auto-disable simulation)

Replays the first 10 fired bars of the BTC/5m holdout, weighted by the diagnostic-sandbox sizing pin (0.50%), applying `evaluateDiagnosticSandboxAutoDisable` math verbatim.

| # | side | p_long | p_short | fwd% | pnl% | acct% | cum_pnl% | dd% |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| 1 | long | 0.8473 | 0.3513 | +1.5753 | +1.2753 | +0.0064 | +0.0064 | +0.0000 |
| 2 | long | 0.9070 | 0.3738 | +0.5005 | +0.2005 | +0.0010 | +0.0074 | +0.0000 |
| 3 | long | 0.8720 | 0.3596 | +0.2743 | -0.0257 | -0.0001 | +0.0073 | -0.0001 |
| 4 | long | 0.9131 | 0.3598 | -0.0348 | -0.3348 | -0.0017 | +0.0056 | -0.0018 |
| 5 | short | 0.6924 | 0.9088 | -1.3730 | +1.0730 | +0.0054 | +0.0109 | +0.0000 |
| 6 | short | 0.6870 | 0.8452 | -0.4557 | +0.1557 | +0.0008 | +0.0117 | +0.0000 |
| 7 | short | 0.6837 | 0.8941 | -0.3350 | +0.0350 | +0.0002 | +0.0119 | +0.0000 |
| 8 | long | 0.9369 | 0.4433 | +1.2146 | +0.9146 | +0.0046 | +0.0165 | +0.0000 |
| 9 | long | 0.8411 | 0.3808 | +1.1892 | +0.8892 | +0.0044 | +0.0209 | +0.0000 |
| 10 | long | 0.9385 | 0.3709 | +1.4035 | +1.1035 | +0.0055 | +0.0264 | +0.0000 |

- proof_rollout: trough_pct=`-0.0018%`, cum_pnl_pct=`0.0264`, would_trip_drawdown=`False` (floor `-5.0%`).
