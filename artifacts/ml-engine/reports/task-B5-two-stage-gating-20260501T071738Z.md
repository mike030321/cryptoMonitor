# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T071738Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:17:38.493011Z`)
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
| unconditional_beta_baseline (baseline) | beta | 113 | +46.7629% | -1.5154% | 0.6531 | 9.6372 | 0.2090 | n/a (reference) |
| platt_two_stage | platt | 115 | +46.5036% | -1.4562% | 0.0455 | 8.9681 | 0.8129 | TRUSTWORTHY |
| isotonic_two_stage | isotonic | 3882 | -1071.9275% | -1094.9374% | 0.2457 | 0.1433 | 0.7500 | no |

### Variants that did NOT graduate

- `isotonic_two_stage`
  - holdout.cal_dev_post_calibration=0.2457 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
  - holdout.max_drawdown_pct=-1094.9374% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 42 | 0.2517 | 0.9048 | 0.6531 |
| [0.30, 0.40) | 40 | 0.3494 | 0.9000 | 0.5506 |
| [0.40, 0.50) | 6 | 0.4435 | 1.0000 | 0.5565 |
| [0.50, 0.60) | 14 | 0.5582 | 0.9286 | 0.3704 |
| [0.60, 0.70) | 10 | 0.6458 | 1.0000 | 0.3542 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.80, 0.90) | 76 | 0.8624 | 0.9079 | 0.0455 |
| [0.90, 1.00) | 39 | 0.9369 | 0.9487 | 0.0118 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.70, 0.80) | 3777 | 0.7501 | 0.5044 | 0.2457 |
| [0.80, 0.90) | 30 | 0.8463 | 0.8667 | 0.0204 |
| [0.90, 1.00) | 75 | 0.9270 | 0.9333 | 0.0063 |

## Winner

- variant: `platt_two_stage` (calibration_method=`platt`)
- abstain τ: `0.812939`
- holdout: n_trades=`115` net_pnl=`46.5036%` max_dd=`-1.4562%` cal_dev=`0.0455` profit_factor=`8.9681`

### 10 paper-proof rollout (DS auto-disable simulation)

Replays the first 10 fired bars of the BTC/5m holdout, weighted by the diagnostic-sandbox sizing pin (0.50%), applying `evaluateDiagnosticSandboxAutoDisable` math verbatim.

| # | side | p_long | p_short | fwd% | pnl% | acct% | cum_pnl% | dd% |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| 1 | long | 0.8288 | 0.5683 | +1.5753 | +1.2753 | +0.0064 | +0.0064 | +0.0000 |
| 2 | long | 0.9272 | 0.5703 | +0.5005 | +0.2005 | +0.0010 | +0.0074 | +0.0000 |
| 3 | short | 0.6576 | 0.8732 | -1.3730 | +1.0730 | +0.0054 | +0.0127 | +0.0000 |
| 4 | short | 0.6440 | 0.8906 | -0.3350 | +0.0350 | +0.0002 | +0.0129 | +0.0000 |
| 5 | long | 0.9377 | 0.5760 | +1.2146 | +0.9146 | +0.0046 | +0.0175 | +0.0000 |
| 6 | long | 0.8284 | 0.5760 | +1.5149 | +1.2149 | +0.0061 | +0.0236 | +0.0000 |
| 7 | long | 0.8798 | 0.5820 | +1.1892 | +0.8892 | +0.0044 | +0.0280 | +0.0000 |
| 8 | long | 0.9505 | 0.5887 | +1.4035 | +1.1035 | +0.0055 | +0.0335 | +0.0000 |
| 9 | long | 0.9491 | 0.6021 | +1.4739 | +1.1739 | +0.0059 | +0.0394 | +0.0000 |
| 10 | long | 0.9582 | 0.5885 | +1.4877 | +1.1877 | +0.0059 | +0.0453 | +0.0000 |

- proof_rollout: trough_pct=`0.0000%`, cum_pnl_pct=`0.0453`, would_trip_drawdown=`False` (floor `-5.0%`).
