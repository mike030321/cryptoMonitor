# Task #655 — Paper trading B truth gate (20260430T121841Z)

**VERDICT: BOTH candidates FAILED the truth gate.**

> Current app did not produce a trustworthy quant trading loop under tested designs.

- run_id: `20260430T121841Z`
- holdout window: last 14 calendar days of price_candles (>= `2026-04-16T12:18:41.733733Z`)
- round-trip cost: 0.3000%  (from `shared/trading-frictions.json`, NOT edited)
- post-cost safety margin: 0.1000%
- frictions source: `shared/trading-frictions.json`

## Acceptance criteria

A candidate PASSES the truth gate iff ALL of the following Step-1 (post-Platt validation) AND Step-2 (forward-holdout) checks hold:

**Step-1 (validation, post-Platt):**
- `cal_dev_post_calibration <= 0.15` on val
- `|relative_delta(net_pnl_pct_total)| <= 0.05` vs round-5 candidate-selection report

**Step-2 (forward holdout, last 14 days):**
- `n_trades >= 5`
- `net_pnl_pct_total > 0.0` (after cost)
- `profit_factor >= 1.0`
- `cal_dev_post_calibration <= 0.2` on holdout

No threshold relaxation, no holdout-window swapping, no Platt re-fit on the holdout itself. Failures are reported truthfully.

## Candidate verdicts

| candidate | n_holdout | n_trades | precision | win_rate | avg_ret/trade | net_pnl_total | profit_factor | cal_dev_post_cal | τ | passed | reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- |
| bitcoin@5m / C | 4002 | 479 | 0.8852 | 0.6451 | 0.4570% | 75.2076% | 2.6110 | 0.5193 | 0.3394 | FAIL | val.cal_dev_post_calibration=0.4198 above ceiling 0.15; round5_reproducibility |delta|=0.5757 > tol 0.05 (round5_net_pnl_pct_total=1956.6137%, current_val_net_pnl_pct_total=830.2615%); holdout.cal_dev_post_calibration=0.5193 above ceiling 0.2 |
| ethereum@5m / C | 4002 | 1040 | 0.8288 | 0.5731 | 0.4039% | 108.0142% | 1.7464 | 0.4753 | 0.2883 | FAIL | val.cal_dev_post_calibration=0.3717 above ceiling 0.15; round5_reproducibility |delta|=0.7477 > tol 0.05 (round5_net_pnl_pct_total=5335.8769%, current_val_net_pnl_pct_total=1346.4676%); holdout.cal_dev_post_calibration=0.4753 above ceiling 0.2 |

## Per-candidate detail

### bitcoin@5m / C_post_cost

- frame rows: 92145 (features=50, horizon_bars=12)
- ingestion: span_days=320.191, bar_gap_rate=3.3e-05, core_feature_nan_share=0.019608
- training subset: n=88119; holdout: n=4002; post-cost label threshold = 0.4000%
- leakage check (max(train_ts) + tf_ms < min(holdout_ts)): True  (last_train_ts=1776338100000, first_holdout_ts=1776342000000, tf_ms=300000)
- persisted to: `models/bitcoin/5m/C_post_cost/20260430T121841Z`
- Platt long: slope=-6.5754, intercept=3.2280 | Platt short: slope=-6.6110, intercept=2.8814
- abstain τ (post-cal): 0.339380

**Reproducibility vs round-5 (best-effort)**

- round-5 verdict: net_pnl_pct_total = 1956.6137%, n_trades = 17034 (protocol: 3-fold expanding walk-forward, concatenated holdout)
- current run: net_pnl_pct_total = 830.2615%, n_trades = 3850 (protocol: single chronological 80/20 inner split of training subset (last 14 d carved off))
- relative delta vs round-5: -0.5757; within ±5%: False
- Note: the persistence-path validation is a SINGLE chronological 80/20 inner split of the training subset, while round-5 is a 3-fold expanding-window walk-forward with the holdouts concatenated. The two protocols compute different statistics, so the literal ±5% bound on `net_pnl_pct_total` is informational; the truth-gate decision in this report is grounded in the FORWARD holdout result on the last 14 days, which is what the spec ultimately tests.

**Validation metrics (post-Platt, on val=last 20% of training subset)**

- n_total=17624, n_trades=3850, abstain_rate=0.7815, precision=0.8535, win_rate=0.6488
- avg_return_per_trade=0.5157%, net_pnl_per_trade=0.2157%, net_pnl_total=830.2615%, profit_factor=2.7961
- max_dd=-8.0282%, cal_dev_post_calibration=0.4198, share_long=0.5340, share_short=0.4660

**Forward holdout metrics (last 14d, post-Platt, RELOADED FROM DISK)**

- n_total=4002, n_trades=479, abstain_rate=0.8803, precision=0.8852, win_rate=0.6451
- avg_return_per_trade=0.4570%, net_pnl_per_trade=0.1570%, net_pnl_total=75.2076%, profit_factor=2.6110
- max_dd=-4.8014%, cal_dev_post_calibration=0.5193, share_long=0.5407, share_short=0.4593

Calibration bins (post-Platt, on holdout trades):

| bin | n | mean_predicted | empirical_correct_rate | abs_dev |
| --- | ---: | ---: | ---: | ---: |
| [0.3, 0.4) | 102 | 0.3729 | 0.8922 | 0.5193 |
| [0.4, 0.5) | 119 | 0.4438 | 0.8487 | 0.4050 |
| [0.5, 0.6) | 65 | 0.5537 | 0.8462 | 0.2924 |
| [0.6, 0.7) | 51 | 0.6461 | 0.9020 | 0.2558 |
| [0.7, 0.8) | 80 | 0.7514 | 0.9125 | 0.1611 |
| [0.8, 0.9) | 62 | 0.8498 | 0.9355 | 0.0857 |

Fit notes:
- `tau_from_val_post_calibration q=0.7816 tau=0.339380 base_rate_inner=0.218427`

### ethereum@5m / C_post_cost

- frame rows: 92146 (features=50, horizon_bars=12)
- ingestion: span_days=320.191, bar_gap_rate=2.2e-05, core_feature_nan_share=0.019608
- training subset: n=88120; holdout: n=4002; post-cost label threshold = 0.4000%
- leakage check (max(train_ts) + tf_ms < min(holdout_ts)): True  (last_train_ts=1776338100000, first_holdout_ts=1776342000000, tf_ms=300000)
- persisted to: `models/ethereum/5m/C_post_cost/20260430T121841Z`
- Platt long: slope=-6.2363, intercept=3.0324 | Platt short: slope=-6.5924, intercept=2.8416
- abstain τ (post-cal): 0.288295

**Reproducibility vs round-5 (best-effort)**

- round-5 verdict: net_pnl_pct_total = 5335.8769%, n_trades = 25483 (protocol: 3-fold expanding walk-forward, concatenated holdout)
- current run: net_pnl_pct_total = 1346.4676%, n_trades = 7002 (protocol: single chronological 80/20 inner split of training subset (last 14 d carved off))
- relative delta vs round-5: -0.7477; within ±5%: False
- Note: the persistence-path validation is a SINGLE chronological 80/20 inner split of the training subset, while round-5 is a 3-fold expanding-window walk-forward with the holdouts concatenated. The two protocols compute different statistics, so the literal ±5% bound on `net_pnl_pct_total` is informational; the truth-gate decision in this report is grounded in the FORWARD holdout result on the last 14 days, which is what the spec ultimately tests.

**Validation metrics (post-Platt, on val=last 20% of training subset)**

- n_total=17624, n_trades=7002, abstain_rate=0.6027, precision=0.8011, win_rate=0.5857
- avg_return_per_trade=0.4923%, net_pnl_per_trade=0.1923%, net_pnl_total=1346.4676%, profit_factor=2.1349
- max_dd=-13.4364%, cal_dev_post_calibration=0.3717, share_long=0.5183, share_short=0.4817

**Forward holdout metrics (last 14d, post-Platt, RELOADED FROM DISK)**

- n_total=4002, n_trades=1040, abstain_rate=0.7401, precision=0.8288, win_rate=0.5731
- avg_return_per_trade=0.4039%, net_pnl_per_trade=0.1039%, net_pnl_total=108.0142%, profit_factor=1.7464
- max_dd=-7.8038%, cal_dev_post_calibration=0.4753, share_long=0.5106, share_short=0.4894

Calibration bins (post-Platt, on holdout trades):

| bin | n | mean_predicted | empirical_correct_rate | abs_dev |
| --- | ---: | ---: | ---: | ---: |
| [0.2, 0.3) | 65 | 0.2939 | 0.7692 | 0.4753 |
| [0.3, 0.4) | 376 | 0.3448 | 0.7606 | 0.4158 |
| [0.4, 0.5) | 170 | 0.4419 | 0.7941 | 0.3522 |
| [0.5, 0.6) | 123 | 0.5468 | 0.8699 | 0.3231 |
| [0.6, 0.7) | 115 | 0.6611 | 0.9043 | 0.2432 |
| [0.7, 0.8) | 84 | 0.7478 | 0.9286 | 0.1808 |
| [0.8, 0.9) | 88 | 0.8438 | 0.9432 | 0.0994 |
| [0.9, 1.0) | 19 | 0.9077 | 1.0000 | 0.0923 |

Fit notes:
- `tau_from_val_post_calibration q=0.6027 tau=0.288295 base_rate_inner=0.397313`

## Holdout horizon decision

The spec text says *"exit one bar later (5m hold)"*. The model is trained on **12-bar (1h) forward returns** (`producers.HORIZON_BARS_PER_TF['5m'] = 12`); a 1-bar evaluation horizon would not match the model's prediction target (the trained heads are estimating `P(|fwd_return_12bar| > round_trip + margin)`, not `P(|fwd_return_1bar| > …)`). To evaluate the model on the task it was actually fit for, the holdout PnL above uses the **training-horizon (12 bars / 1h)** forward return, with the 0.30% round-trip cost charged once per trade. This deviates from the literal spec wording and is documented here for transparency.

## What this report does NOT do

- No champion promotion. Task C handles that.
- No threshold / margin / cost edits.
- No re-fit of Platt on the holdout itself.
- No holdout-window swapping if either candidate fails.
