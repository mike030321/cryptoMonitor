# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T071806Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:18:06.723363Z`)
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
| unconditional_beta_baseline (baseline) | beta | 110 | +44.5818% | -2.7645% | 0.6338 | 8.1192 | 0.2317 | n/a (reference) |
| platt_two_stage | platt | 100 | +42.9353% | -1.9295% | 0.0285 | 9.9489 | 0.8567 | TRUSTWORTHY |
| isotonic_two_stage | isotonic | 3882 | -1102.9253% | -1102.4738% | 0.1702 | 0.1371 | 0.6667 | no |

### Variants that did NOT graduate

- `isotonic_two_stage`
  - holdout.max_drawdown_pct=-1102.4738% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 30 | 0.2662 | 0.9000 | 0.6338 |
| [0.30, 0.40) | 37 | 0.3520 | 0.8649 | 0.5129 |
| [0.40, 0.50) | 26 | 0.4565 | 0.9231 | 0.4666 |
| [0.50, 0.60) | 7 | 0.5361 | 1.0000 | 0.4639 |
| [0.60, 0.70) | 5 | 0.6337 | 1.0000 | 0.3663 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.80, 0.90) | 44 | 0.8806 | 0.9091 | 0.0285 |
| [0.90, 1.00) | 56 | 0.9332 | 0.9107 | 0.0225 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.60, 0.70) | 3771 | 0.6667 | 0.4964 | 0.1702 |
| [0.80, 0.90) | 37 | 0.8555 | 0.9189 | 0.0634 |
| [0.90, 1.00) | 70 | 0.9278 | 0.9143 | 0.0136 |

## Winner

- variant: `platt_two_stage` (calibration_method=`platt`)
- abstain τ: `0.856735`
- holdout: n_trades=`100` net_pnl=`42.9353%` max_dd=`-1.9295%` cal_dev=`0.0285` profit_factor=`9.9489`

### 10 paper-proof rollout (DS auto-disable simulation)

Replays the first 10 fired bars of the BTC/5m holdout, weighted by the diagnostic-sandbox sizing pin (0.50%), applying `evaluateDiagnosticSandboxAutoDisable` math verbatim.

| # | side | p_long | p_short | fwd% | pnl% | acct% | cum_pnl% | dd% |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| 1 | long | 0.8817 | 0.7384 | +0.5005 | +0.2005 | +0.0010 | +0.0010 | +0.0000 |
| 2 | long | 0.8767 | 0.7469 | -0.0348 | -0.3348 | -0.0017 | -0.0007 | -0.0017 |
| 3 | short | 0.5931 | 0.9346 | -1.3730 | +1.0730 | +0.0054 | +0.0047 | +0.0000 |
| 4 | short | 0.5653 | 0.8739 | -0.4557 | +0.1557 | +0.0008 | +0.0055 | +0.0000 |
| 5 | short | 0.6110 | 0.8739 | -0.3220 | +0.0220 | +0.0001 | +0.0056 | +0.0000 |
| 6 | short | 0.5806 | 0.9486 | -0.3350 | +0.0350 | +0.0002 | +0.0058 | +0.0000 |
| 7 | long | 0.8810 | 0.7542 | +0.9792 | +0.6792 | +0.0034 | +0.0092 | +0.0000 |
| 8 | long | 0.9168 | 0.7400 | +1.2146 | +0.9146 | +0.0046 | +0.0137 | +0.0000 |
| 9 | long | 0.8704 | 0.7462 | +1.5149 | +1.2149 | +0.0061 | +0.0198 | +0.0000 |
| 10 | long | 0.8602 | 0.7458 | +1.1892 | +0.8892 | +0.0044 | +0.0242 | +0.0000 |

- proof_rollout: trough_pct=`-0.0017%`, cum_pnl_pct=`0.0242`, would_trip_drawdown=`False` (floor `-5.0%`).
