# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T071752Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `14` days starting `2026-04-17T07:17:52.308771Z`)
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
| unconditional_beta_baseline (baseline) | beta | 109 | +41.8145% | -1.9159% | 0.6406 | 7.9145 | 0.2019 | n/a (reference) |
| platt_two_stage | platt | 110 | +41.3698% | -1.9295% | 0.0278 | 6.7460 | 0.8290 | TRUSTWORTHY |
| isotonic_two_stage | isotonic | 3882 | -1113.4217% | -1112.9702% | 0.0106 | 0.1333 | 0.5000 | no |

### Variants that did NOT graduate

- `isotonic_two_stage`
  - holdout.max_drawdown_pct=-1112.9702% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 27 | 0.2483 | 0.8889 | 0.6406 |
| [0.30, 0.40) | 46 | 0.3491 | 0.8696 | 0.5205 |
| [0.40, 0.50) | 18 | 0.4580 | 0.9444 | 0.4864 |
| [0.50, 0.60) | 10 | 0.5407 | 1.0000 | 0.4593 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.80, 0.90) | 61 | 0.8638 | 0.8361 | 0.0278 |
| [0.90, 1.00) | 49 | 0.9262 | 0.9388 | 0.0126 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.50, 0.60) | 3773 | 0.5000 | 0.4956 | 0.0044 |
| [0.80, 0.90) | 66 | 0.8809 | 0.8788 | 0.0021 |
| [0.90, 1.00) | 37 | 0.9353 | 0.9459 | 0.0106 |

## Winner

- variant: `platt_two_stage` (calibration_method=`platt`)
- abstain τ: `0.829040`
- holdout: n_trades=`110` net_pnl=`41.3698%` max_dd=`-1.9295%` cal_dev=`0.0278` profit_factor=`6.7460`

### 10 paper-proof rollout (DS auto-disable simulation)

Replays the first 10 fired bars of the BTC/5m holdout, weighted by the diagnostic-sandbox sizing pin (0.50%), applying `evaluateDiagnosticSandboxAutoDisable` math verbatim.

| # | side | p_long | p_short | fwd% | pnl% | acct% | cum_pnl% | dd% |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| 1 | long | 0.8923 | 0.6915 | +0.5005 | +0.2005 | +0.0010 | +0.0010 | +0.0000 |
| 2 | long | 0.8695 | 0.6949 | -0.0348 | -0.3348 | -0.0017 | -0.0007 | -0.0017 |
| 3 | short | 0.2575 | 0.8848 | -1.3730 | +1.0730 | +0.0054 | +0.0047 | +0.0000 |
| 4 | short | 0.4112 | 0.8773 | -0.4557 | +0.1557 | +0.0008 | +0.0055 | +0.0000 |
| 5 | short | 0.4190 | 0.9017 | -0.3220 | +0.0220 | +0.0001 | +0.0056 | +0.0000 |
| 6 | short | 0.4110 | 0.9117 | -0.3350 | +0.0350 | +0.0002 | +0.0058 | +0.0000 |
| 7 | long | 0.8305 | 0.7104 | +0.3712 | +0.0712 | +0.0004 | +0.0061 | +0.0000 |
| 8 | long | 0.9177 | 0.6942 | +1.2146 | +0.9146 | +0.0046 | +0.0107 | +0.0000 |
| 9 | long | 0.9198 | 0.6993 | +1.4035 | +1.1035 | +0.0055 | +0.0162 | +0.0000 |
| 10 | long | 0.9169 | 0.6980 | +1.4739 | +1.1739 | +0.0059 | +0.0221 | +0.0000 |

- proof_rollout: trough_pct=`-0.0017%`, cum_pnl_pct=`0.0221`, would_trip_drawdown=`False` (floor `-5.0%`).
