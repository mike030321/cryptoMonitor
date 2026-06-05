# Task #669 — BTC/5m B5 two-stage gating study

- run_id: `20260501T072130Z`
- coin/timeframe: `bitcoin/5m`
- lookback: `380.0 days` (holdout `15` days starting `2026-04-16T07:21:30.693696Z`)
- frame rows: `92253` (span_days=`320.566`, bar_gap_rate=`3.3e-05`, bars_source=`candles`)
- candidate: n_train=`88060`, n_holdout=`4169`, threshold_fraction=`0.0080` (horizon=`12` bars, n_features=`50`)
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
| unconditional_beta_baseline (baseline) | beta | 127 | +47.4541% | -2.2593% | 0.6182 | 7.0944 | 0.2127 | n/a (reference) |
| platt_two_stage | platt | 129 | +46.8992% | -2.2593% | 0.0811 | 6.7413 | 0.7890 | TRUSTWORTHY |
| isotonic_two_stage | isotonic | 4169 | -1204.3867% | -1204.3842% | 0.2711 | 0.1367 | 0.7625 | no |

### Variants that did NOT graduate

- `isotonic_two_stage`
  - holdout.cal_dev_post_calibration=0.2711 above ceiling 0.2 (HARD gate for calibration_status='trustworthy')
  - holdout.max_drawdown_pct=-1204.3842% not strictly > -5.0% (would trip DS auto-disable)

## Per-variant holdout calibration bins

### unconditional_beta_baseline (beta)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.20, 0.30) | 46 | 0.2513 | 0.8696 | 0.6182 |
| [0.30, 0.40) | 16 | 0.3450 | 0.9375 | 0.5925 |
| [0.40, 0.50) | 27 | 0.4381 | 0.8889 | 0.4508 |
| [0.50, 0.60) | 26 | 0.5491 | 0.9231 | 0.3740 |
| [0.60, 0.70) | 9 | 0.6543 | 1.0000 | 0.3457 |

### platt_two_stage (platt)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.70, 0.80) | 8 | 0.7939 | 0.8750 | 0.0811 |
| [0.80, 0.90) | 58 | 0.8390 | 0.8793 | 0.0404 |
| [0.90, 1.00) | 63 | 0.9390 | 0.9206 | 0.0184 |

### isotonic_two_stage (isotonic)

| bin | n | mean_predicted | empirical_correct | abs_dev |
|:--|---:|---:|---:|---:|
| [0.70, 0.80) | 4052 | 0.7625 | 0.4914 | 0.2711 |
| [0.80, 0.90) | 65 | 0.8674 | 0.8615 | 0.0058 |
| [0.90, 1.00) | 52 | 0.9448 | 0.9423 | 0.0025 |

## Winner

- variant: `platt_two_stage` (calibration_method=`platt`)
- abstain τ: `0.788967`
- holdout: n_trades=`129` net_pnl=`46.8992%` max_dd=`-2.2593%` cal_dev=`0.0811` profit_factor=`6.7413`
- registry slot: `models/bitcoin/5m/20260501T072142Z` (version `20260501T072142Z`)
- manifest tags: calibration_method=`platt`, calibration_status=`trustworthy`, label_family=`C_post_cost`, abstain_tau=`0.7889671952682393`, friction_threshold_pct=`0.8`

### 10 paper-proof rollout (DS auto-disable simulation)

Replays the first 10 fired bars of the BTC/5m holdout, weighted by the diagnostic-sandbox sizing pin (0.50%), applying `evaluateDiagnosticSandboxAutoDisable` math verbatim.

| # | side | p_long | p_short | fwd% | pnl% | acct% | cum_pnl% | dd% |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| 1 | short | 0.4350 | 0.8521 | -1.6136 | +1.3136 | +0.0066 | +0.0066 | +0.0000 |
| 2 | short | 0.4350 | 0.9818 | -1.2217 | +0.9217 | +0.0046 | +0.0112 | +0.0000 |
| 3 | short | 0.4410 | 0.9756 | -0.9912 | +0.6912 | +0.0035 | +0.0146 | +0.0000 |
| 4 | short | 0.4374 | 0.9756 | -0.8598 | +0.5598 | +0.0028 | +0.0174 | +0.0000 |
| 5 | short | 0.4391 | 0.9375 | -0.5599 | +0.2599 | +0.0013 | +0.0187 | +0.0000 |
| 6 | short | 0.6227 | 0.8033 | +0.4735 | -0.7735 | -0.0039 | +0.0149 | -0.0039 |
| 7 | short | 0.5457 | 0.8678 | -0.9769 | +0.6769 | +0.0034 | +0.0182 | -0.0005 |
| 8 | long | 0.8088 | 0.4882 | +0.8545 | +0.5545 | +0.0028 | +0.0210 | +0.0000 |
| 9 | long | 0.8149 | 0.4813 | +1.5753 | +1.2753 | +0.0064 | +0.0274 | +0.0000 |
| 10 | long | 0.8109 | 0.4836 | +0.5005 | +0.2005 | +0.0010 | +0.0284 | +0.0000 |

- proof_rollout: trough_pct=`-0.0039%`, cum_pnl_pct=`0.0284`, would_trip_drawdown=`False` (floor `-5.0%`).
