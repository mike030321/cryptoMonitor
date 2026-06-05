# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T071700Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:17:00.572018Z`)
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
| unconditional_beta_baseline (baseline) | beta | 119 | +40.3266% | -2.3354% | 0.5966 | 6.1990 | 0.2018 | n/a (reference) |
| platt_two_stage | platt | 122 | +41.0441% | -1.9295% | 0.0809 | 6.0836 | 0.7706 | TRUSTWORTHY |
| isotonic_two_stage | isotonic | 3882 | -1114.6554% | -1114.2038% | 0.0324 | 0.1329 | 0.5000 | no |

### Variants that did NOT graduate

- `isotonic_two_stage`
  - holdout.max_drawdown_pct=-1114.2038% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 31 | 0.2421 | 0.8387 | 0.5966 |
| [0.30, 0.40) | 23 | 0.3602 | 0.9130 | 0.5528 |
| [0.40, 0.50) | 32 | 0.4398 | 0.9062 | 0.4664 |
| [0.50, 0.60) | 19 | 0.5370 | 0.9474 | 0.4103 |
| [0.60, 0.70) | 9 | 0.6511 | 0.8889 | 0.2378 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.70, 0.80) | 15 | 0.7846 | 0.7333 | 0.0513 |
| [0.80, 0.90) | 45 | 0.8524 | 0.9333 | 0.0809 |
| [0.90, 1.00) | 62 | 0.9409 | 0.9032 | 0.0377 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.50, 0.60) | 3764 | 0.5000 | 0.4947 | 0.0053 |
| [0.70, 0.80) | 18 | 0.7454 | 0.7778 | 0.0324 |
| [0.80, 0.90) | 41 | 0.8767 | 0.9024 | 0.0258 |
| [0.90, 1.00) | 58 | 0.9460 | 0.9310 | 0.0149 |

## Winner

- variant: `platt_two_stage` (calibration_method=`platt`)
- abstain τ: `0.770610`
- holdout: n_trades=`122` net_pnl=`41.0441%` max_dd=`-1.9295%` cal_dev=`0.0809` profit_factor=`6.0836`

### 10 paper-proof rollout (DS auto-disable simulation)

Replays the first 10 fired bars of the BTC/5m holdout, weighted by the diagnostic-sandbox sizing pin (0.50%), applying `evaluateDiagnosticSandboxAutoDisable` math verbatim.

| # | side | p_long | p_short | fwd% | pnl% | acct% | cum_pnl% | dd% |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| 1 | long | 0.9292 | 0.4109 | +0.5005 | +0.2005 | +0.0010 | +0.0010 | +0.0000 |
| 2 | long | 0.9072 | 0.4162 | -0.0348 | -0.3348 | -0.0017 | -0.0007 | -0.0017 |
| 3 | short | 0.3100 | 0.9109 | -1.3730 | +1.0730 | +0.0054 | +0.0047 | +0.0000 |
| 4 | short | 0.3044 | 0.8328 | -0.4557 | +0.1557 | +0.0008 | +0.0055 | +0.0000 |
| 5 | short | 0.3044 | 0.8245 | -0.3220 | +0.0220 | +0.0001 | +0.0056 | +0.0000 |
| 6 | short | 0.3044 | 0.8907 | -0.3350 | +0.0350 | +0.0002 | +0.0058 | +0.0000 |
| 7 | long | 0.9292 | 0.4187 | +1.2146 | +0.9146 | +0.0046 | +0.0103 | +0.0000 |
| 8 | long | 0.9292 | 0.4253 | +1.4035 | +1.1035 | +0.0055 | +0.0158 | +0.0000 |
| 9 | long | 0.9292 | 0.4242 | +1.4739 | +1.1739 | +0.0059 | +0.0217 | +0.0000 |
| 10 | long | 0.9292 | 0.4355 | +1.4877 | +1.1877 | +0.0059 | +0.0277 | +0.0000 |

- proof_rollout: trough_pct=`-0.0017%`, cum_pnl_pct=`0.0277`, would_trip_drawdown=`False` (floor `-5.0%`).
