# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T071724Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:17:24.728221Z`)
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
| unconditional_beta_baseline (baseline) | beta | 123 | +40.2607% | -2.8283% | 0.5772 | 5.0209 | 0.2132 | n/a (reference) |
| platt_two_stage | platt | 123 | +40.4538% | -2.2795% | 0.0909 | 5.1348 | 0.8011 | TRUSTWORTHY |
| isotonic_two_stage | isotonic | 3882 | -1071.8804% | -1096.4457% | 0.2166 | 0.1433 | 0.7209 | no |

### Variants that did NOT graduate

- `isotonic_two_stage`
  - holdout.cal_dev_post_calibration=0.2166 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
  - holdout.max_drawdown_pct=-1096.4457% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 57 | 0.2474 | 0.8246 | 0.5772 |
| [0.30, 0.40) | 30 | 0.3432 | 0.8667 | 0.5235 |
| [0.40, 0.50) | 14 | 0.4387 | 0.9286 | 0.4898 |
| [0.50, 0.60) | 9 | 0.5331 | 0.8889 | 0.3557 |
| [0.60, 0.70) | 8 | 0.6351 | 0.8750 | 0.2399 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.80, 0.90) | 83 | 0.8450 | 0.8795 | 0.0346 |
| [0.90, 1.00) | 40 | 0.9409 | 0.8500 | 0.0909 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.70, 0.80) | 3767 | 0.7209 | 0.5044 | 0.2166 |
| [0.80, 0.90) | 69 | 0.8528 | 0.8551 | 0.0022 |
| [0.90, 1.00) | 46 | 0.9383 | 0.8913 | 0.0470 |

## Winner

- variant: `platt_two_stage` (calibration_method=`platt`)
- abstain τ: `0.801094`
- holdout: n_trades=`123` net_pnl=`40.4538%` max_dd=`-2.2795%` cal_dev=`0.0909` profit_factor=`5.1348`

### 10 paper-proof rollout (DS auto-disable simulation)

Replays the first 10 fired bars of the BTC/5m holdout, weighted by the diagnostic-sandbox sizing pin (0.50%), applying `evaluateDiagnosticSandboxAutoDisable` math verbatim.

| # | side | p_long | p_short | fwd% | pnl% | acct% | cum_pnl% | dd% |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| 1 | long | 0.8842 | 0.4438 | +0.5005 | +0.2005 | +0.0010 | +0.0010 | +0.0000 |
| 2 | long | 0.8448 | 0.4489 | +0.2743 | -0.0257 | -0.0001 | +0.0009 | -0.0001 |
| 3 | long | 0.8922 | 0.4506 | -0.0348 | -0.3348 | -0.0017 | -0.0008 | -0.0018 |
| 4 | short | 0.6139 | 0.8839 | -1.3730 | +1.0730 | +0.0054 | +0.0046 | +0.0000 |
| 5 | short | 0.5668 | 0.8413 | -0.4557 | +0.1557 | +0.0008 | +0.0053 | +0.0000 |
| 6 | short | 0.5736 | 0.8151 | -0.3220 | +0.0220 | +0.0001 | +0.0055 | +0.0000 |
| 7 | short | 0.5708 | 0.8803 | -0.3350 | +0.0350 | +0.0002 | +0.0056 | +0.0000 |
| 8 | long | 0.8620 | 0.4761 | +1.2146 | +0.9146 | +0.0046 | +0.0102 | +0.0000 |
| 9 | long | 0.8236 | 0.4731 | +1.1892 | +0.8892 | +0.0044 | +0.0146 | +0.0000 |
| 10 | long | 0.9203 | 0.4731 | +1.4035 | +1.1035 | +0.0055 | +0.0202 | +0.0000 |

- proof_rollout: trough_pct=`-0.0018%`, cum_pnl_pct=`0.0202`, would_trip_drawdown=`False` (floor `-5.0%`).
