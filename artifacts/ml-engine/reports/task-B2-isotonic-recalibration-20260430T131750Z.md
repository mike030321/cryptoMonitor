# Task #657 — Paper trading B2: Platt vs isotonic recalibration (20260430T131750Z)

**Per-candidate verdicts**: PASS=0, PARTIAL=0, REJECT=2, ERROR=0

> Per the "no rescue" rule the spec encodes, this report writes verdicts truthfully and does NOT promote a champion or queue any follow-up tasks.

- run_id: `20260430T131750Z`
- holdout window: last 14 calendar days of price_candles (>= `2026-04-16T13:17:50.196739Z`)
- round-trip cost: 0.3000%  (from `shared/trading-frictions.json`, NOT edited)
- post-cost safety margin: 0.1000%
- frictions source: `shared/trading-frictions.json`

## Aggregate recommendation

Current app did not produce a trustworthy quant trading loop under tested designs.

## Acceptance criteria (per candidate)

**PASS** iff the isotonic-calibrated variant satisfies ALL of:
- `cal_dev_post_calibration <= 0.2` on the 14-day forward holdout
- `n_trades >= 5` on holdout
- `net_pnl_pct_total > 0.0` on holdout (post-fee)
- `profit_factor >= 1.0` on holdout
- ranking integrity: Spearman(raw, iso_cal) `>= 0.95` per head on holdout
- non-degeneracy: post-isotonic distribution has `>= 5` distinct values per head on holdout

**PARTIAL** iff calibration improves vs Platt (`cal_dev_holdout_iso < cal_dev_holdout_platt`) AND financial metrics remain positive AND ranking integrity holds, but `cal_dev_post_calibration > 0.2` on holdout. STOPS without proposing any follow-up.

**REJECT** iff ANY of:
- `cal_dev_holdout_iso > cal_dev_holdout_platt` (isotonic made calibration worse)
- `net_pnl_pct_total_iso < net_pnl_pct_total_platt` by more than 5.0pp absolute (e.g. 75% → 65% rejects; 75% → 73% is acceptable noise)
- `profit_factor_iso < 1.0`
- ranking integrity broken (Spearman per head < 0.95)
- any leakage detected (isotonic fit included holdout rows)

## Side-by-side holdout metrics

| candidate | method | n_trades | precision | win_rate | avg_ret/trade | net_pnl_total | profit_factor | cal_dev | τ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bitcoin@5m / C | Platt | 481 | 0.8857 | 0.6403 | 0.4574% | 75.7015% | 2.6910 | 0.5137 | 0.3284 |
| bitcoin@5m / C | Isotonic | 503 | 0.8748 | 0.6203 | 0.4377% | 69.2869% | 2.3416 | 0.4558 | 0.3667 |
| ethereum@5m / C | Platt | 1027 | 0.8277 | 0.5667 | 0.4018% | 104.5256% | 1.7288 | 0.4568 | 0.2718 |
| ethereum@5m / C | Isotonic | 1094 | 0.8181 | 0.5430 | 0.3907% | 99.1994% | 1.6423 | 0.4096 | 0.3143 |

## Per-candidate verdict

| candidate | verdict | binding criterion | partial detail |
| --- | :---: | --- | --- |
| bitcoin@5m / C | REJECT | `net_pnl_dropped_more_than_5pp_vs_platt` | — |
| ethereum@5m / C | REJECT | `net_pnl_dropped_more_than_5pp_vs_platt` | — |

## Per-candidate detail

### bitcoin@5m / C_post_cost

- frame rows: 92156 (features=50, horizon_bars=12)
- ingestion: span_days=320.229, bar_gap_rate=3.3e-05, core_feature_nan_share=0.019608
- training subset: n=88131; holdout: n=4001; post-cost label threshold = 0.4000%
- leakage check (max(train_ts) + tf_ms < min(holdout_ts)): True  (last_train_ts=1776341700000, first_holdout_ts=1776345600000, tf_ms=300000)
- shared boosters: n_train_inner=70505, n_val=17626, base_rate_train_inner=0.218524, long_head_present=True, short_head_present=True
- booster equality (model_to_string md5): long=9813c21867e513dc7244ebb4c815da69 == 9813c21867e513dc7244ebb4c815da69 (True); short=652fc55191edb188c4ebf1963ab35233 == 652fc55191edb188c4ebf1963ab35233 (True)

**Step 5 comparison table (Platt vs isotonic, holdout)**

| metric | Platt | isotonic | delta (iso − platt) | direction |
| --- | ---: | ---: | ---: | --- |
| cal_dev_holdout | 0.5137 | 0.4558 | -0.0579 | lower=better |
| cal_dev_validation | 0.4083 | 0.3731 | -0.0352 | lower=better |
| n_trades | 481 | 503 | +22 | informational |
| net_pnl_pct_total | 75.7015% | 69.2869% | -6.4146% | higher=better |
| profit_factor | 2.6910 | 2.3416 | -0.3494 | higher=better |
| win_rate | 0.6403 | 0.6203 | -0.0201 | higher=better |
| max_drawdown_pct | -6.1023% | -5.1191% | +0.9832% | smaller-magnitude=better |
| avg_return_per_trade_pct | 0.4574% | 0.4377% | -0.0196% | higher=better |
| abstain_rate | 0.8798 | 0.8743 | -0.0055 | informational |
| tau | 0.3284 | 0.3667 | +0.0383 | informational |
| spearman_raw_vs_cal_long | 1.0000* | 0.9970 | -0.0030 | ranking integrity (≥0.95) |
| spearman_raw_vs_cal_short | 1.0000* | 0.9971 | -0.0029 | ranking integrity (≥0.95) |
| n_distinct_cal_probs_long (holdout) | — | 47 | — | non-degeneracy (≥5) |
| n_distinct_cal_probs_short (holdout) | — | 50 | — | non-degeneracy (≥5) |

\* Platt's Spearman vs raw is always 1.00 by construction (a sigmoid is monotone).

**Trade-selection diff (holdout)**

- `n_trades_only_in_platt`: 10
- `n_trades_only_in_isotonic`: 32
- `n_trades_in_both`: 471
- `n_trades_disagreed_on_side`: 0 (should be near-zero; sanity check)

**Validation-side ranking / non-degeneracy (informational, not gate-binding)**

- Spearman(raw, iso_cal) on val: long=0.9982, short=0.9975
- Distinct iso_cal probabilities on val: long=41, short=48

**Platt — persisted to `models/bitcoin/5m/C_post_cost/20260430T131750Z-platt`, τ = 0.328384**

- Platt long: slope=-6.4356, intercept=3.2473 | Platt short: slope=-6.5765, intercept=2.8705
- Validation: n=17626, n_trades=3852, abstain_rate=0.7815, precision=0.8453, win_rate=0.6449
- Validation: avg_ret/trade=0.5064%, net_pnl_total=794.8786%, profit_factor=2.6274, cal_dev=0.4083
- Holdout: n=4001, n_trades=481, abstain_rate=0.8798, precision=0.8857, win_rate=0.6403
- Holdout: avg_ret/trade=0.4574%, net_pnl_per_trade=0.1574%, net_pnl_total=75.7015%, profit_factor=2.6910
- Holdout: max_dd=-6.1023%, cal_dev=0.5137, share_long=0.5301, share_short=0.4699

Calibration bins (Platt, holdout trades):

| bin | n | mean_predicted | empirical_correct_rate | abs_dev |
| --- | ---: | ---: | ---: | ---: |
| [0.3, 0.4) | 117 | 0.3667 | 0.8803 | 0.5137 |
| [0.4, 0.5) | 110 | 0.4444 | 0.8182 | 0.3737 |
| [0.5, 0.6) | 64 | 0.5530 | 0.9062 | 0.3533 |
| [0.6, 0.7) | 51 | 0.6526 | 0.9412 | 0.2886 |
| [0.7, 0.8) | 75 | 0.7469 | 0.8933 | 0.1465 |
| [0.8, 0.9) | 64 | 0.8504 | 0.9375 | 0.0871 |

Platt fit notes:
- `platt_tau_from_val_post_calibration q=0.7815 tau=0.328384 base_rate_inner=0.218524`

**Isotonic — persisted to `models/bitcoin/5m/C_post_cost/20260430T131750Z-iso`, τ = 0.366667**

- Isotonic long: knot count=82; x range=[0.0143, 0.8456]; y range=[0.0000, 1.0000]
- Isotonic short: knot count=95; x range=[0.0101, 0.7911]; y range=[0.0000, 1.0000]
- Validation: n=17626, n_trades=3871, abstain_rate=0.7804, precision=0.8463, win_rate=0.6448
- Validation: avg_ret/trade=0.5076%, net_pnl_total=803.7112%, profit_factor=2.6519, cal_dev=0.3731
- Holdout: n=4001, n_trades=503, abstain_rate=0.8743, precision=0.8748, win_rate=0.6203
- Holdout: avg_ret/trade=0.4377%, net_pnl_per_trade=0.1377%, net_pnl_total=69.2869%, profit_factor=2.3416
- Holdout: max_dd=-5.1191%, cal_dev=0.4558, share_long=0.4871, share_short=0.5129

Calibration bins (Isotonic, holdout trades):

| bin | n | mean_predicted | empirical_correct_rate | abs_dev |
| --- | ---: | ---: | ---: | ---: |
| [0.3, 0.4) | 89 | 0.3869 | 0.8427 | 0.4558 |
| [0.4, 0.5) | 205 | 0.4390 | 0.8390 | 0.4001 |
| [0.5, 0.6) | 56 | 0.5616 | 0.9107 | 0.3491 |
| [0.6, 0.7) | 86 | 0.6617 | 0.9070 | 0.2452 |
| [0.7, 0.8) | 29 | 0.7574 | 0.8966 | 0.1391 |
| [0.8, 0.9) | 15 | 0.8632 | 1.0000 | 0.1368 |
| [0.9, 1.0) | 23 | 0.9387 | 1.0000 | 0.0613 |

Isotonic fit notes:
- `isotonic_tau_from_val_post_calibration q=0.7815 tau=0.366667 base_rate_inner=0.218524`

**B2 verdict: REJECT** — binding criterion: `net_pnl_dropped_more_than_5pp_vs_platt`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — isotonic.holdout.cal_dev_post_calibration=0.4558 vs ceiling 0.2
- [PASS] `n_trades>=5` — isotonic.holdout.n_trades=503 vs floor 5
- [PASS] `net_pnl_pct_total>0` — isotonic.holdout.net_pnl_pct_total=69.2869% vs floor >0.0
- [PASS] `profit_factor>=1.0` — isotonic.holdout.profit_factor=2.3416 vs floor 1.0
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, iso_cal) long=0.9970 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, iso_cal) short=0.9971 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — holdout distinct iso_cal probabilities long=47 vs floor 5
- [PASS] `distinct_holdout_short>=5` — holdout distinct iso_cal probabilities short=50 vs floor 5

REJECT gate evaluation:
- [ok] `cal_dev_holdout_iso_worse_than_platt` — iso_cal_dev_holdout=0.4558 > platt_cal_dev_holdout=0.5137 (isotonic made calibration worse)
- [TRIGGERED] `net_pnl_dropped_more_than_5pp_vs_platt` — net_pnl_pct_total drop = 6.4146pp (platt=75.7015% → iso=69.2869%); threshold = 5.0pp
- [ok] `profit_factor_iso_below_1.0` — isotonic.holdout.profit_factor=2.3416 below 1.0
- [ok] `ranking_integrity_broken` — holdout spearman(raw, iso_cal) long=0.9970, short=0.9971; floor 0.95
- [ok] `leakage_detected` — leakage gate max(train_ts) + tf_ms < min(holdout_ts) failed

Deltas (iso − platt): cal_dev_holdout=-0.0579 (negative = isotonic improved), net_pnl_pct_total=-6.4146pp (reject if drop > 5.0pp)

### ethereum@5m / C_post_cost

- frame rows: 92157 (features=50, horizon_bars=12)
- ingestion: span_days=320.229, bar_gap_rate=2.2e-05, core_feature_nan_share=0.019608
- training subset: n=88132; holdout: n=4001; post-cost label threshold = 0.4000%
- leakage check (max(train_ts) + tf_ms < min(holdout_ts)): True  (last_train_ts=1776341700000, first_holdout_ts=1776345600000, tf_ms=300000)
- shared boosters: n_train_inner=70506, n_val=17626, base_rate_train_inner=0.397328, long_head_present=True, short_head_present=True
- booster equality (model_to_string md5): long=d14661dca38f01f547fdd5680016d820 == d14661dca38f01f547fdd5680016d820 (True); short=9b3bfd1c40a5be383a7304aabed4cb6d == 9b3bfd1c40a5be383a7304aabed4cb6d (True)

**Step 5 comparison table (Platt vs isotonic, holdout)**

| metric | Platt | isotonic | delta (iso − platt) | direction |
| --- | ---: | ---: | ---: | --- |
| cal_dev_holdout | 0.4568 | 0.4096 | -0.0473 | lower=better |
| cal_dev_validation | 0.3934 | 0.3801 | -0.0133 | lower=better |
| n_trades | 1027 | 1094 | +67 | informational |
| net_pnl_pct_total | 104.5256% | 99.1994% | -5.3262% | higher=better |
| profit_factor | 1.7288 | 1.6423 | -0.0865 | higher=better |
| win_rate | 0.5667 | 0.5430 | -0.0237 | higher=better |
| max_drawdown_pct | -9.7574% | -9.9846% | -0.2272% | smaller-magnitude=better |
| avg_return_per_trade_pct | 0.4018% | 0.3907% | -0.0111% | higher=better |
| abstain_rate | 0.7433 | 0.7266 | -0.0167 | informational |
| tau | 0.2718 | 0.3143 | +0.0426 | informational |
| spearman_raw_vs_cal_long | 1.0000* | 0.9976 | -0.0024 | ranking integrity (≥0.95) |
| spearman_raw_vs_cal_short | 1.0000* | 0.9968 | -0.0032 | ranking integrity (≥0.95) |
| n_distinct_cal_probs_long (holdout) | — | 43 | — | non-degeneracy (≥5) |
| n_distinct_cal_probs_short (holdout) | — | 66 | — | non-degeneracy (≥5) |

\* Platt's Spearman vs raw is always 1.00 by construction (a sigmoid is monotone).

**Trade-selection diff (holdout)**

- `n_trades_only_in_platt`: 94
- `n_trades_only_in_isotonic`: 161
- `n_trades_in_both`: 933
- `n_trades_disagreed_on_side`: 1 (should be near-zero; sanity check)

**Validation-side ranking / non-degeneracy (informational, not gate-binding)**

- Spearman(raw, iso_cal) on val: long=0.9979, short=0.9967
- Distinct iso_cal probabilities on val: long=42, short=57

**Platt — persisted to `models/ethereum/5m/C_post_cost/20260430T131750Z-platt`, τ = 0.271760**

- Platt long: slope=-6.0032, intercept=2.9541 | Platt short: slope=-6.3510, intercept=2.7592
- Validation: n=17626, n_trades=7003, abstain_rate=0.6027, precision=0.8057, win_rate=0.5886
- Validation: avg_ret/trade=0.4967%, net_pnl_total=1377.4749%, profit_factor=2.1853, cal_dev=0.3934
- Holdout: n=4001, n_trades=1027, abstain_rate=0.7433, precision=0.8277, win_rate=0.5667
- Holdout: avg_ret/trade=0.4018%, net_pnl_per_trade=0.1018%, net_pnl_total=104.5256%, profit_factor=1.7288
- Holdout: max_dd=-9.7574%, cal_dev=0.4568, share_long=0.5141, share_short=0.4859

Calibration bins (Platt, holdout trades):

| bin | n | mean_predicted | empirical_correct_rate | abs_dev |
| --- | ---: | ---: | ---: | ---: |
| [0.2, 0.3) | 128 | 0.2853 | 0.7422 | 0.4568 |
| [0.3, 0.4) | 307 | 0.3435 | 0.7655 | 0.4219 |
| [0.4, 0.5) | 169 | 0.4469 | 0.8343 | 0.3874 |
| [0.5, 0.6) | 126 | 0.5421 | 0.8333 | 0.2912 |
| [0.6, 0.7) | 123 | 0.6519 | 0.8862 | 0.2343 |
| [0.7, 0.8) | 75 | 0.7537 | 0.9067 | 0.1529 |
| [0.8, 0.9) | 87 | 0.8437 | 0.9770 | 0.1333 |
| [0.9, 1.0) | 12 | 0.9086 | 1.0000 | 0.0914 |

Platt fit notes:
- `platt_tau_from_val_post_calibration q=0.6027 tau=0.271760 base_rate_inner=0.397328`

**Isotonic — persisted to `models/ethereum/5m/C_post_cost/20260430T131750Z-iso`, τ = 0.314342**

- Isotonic long: knot count=84; x range=[0.0258, 0.9020]; y range=[0.0000, 0.9886]
- Isotonic short: knot count=114; x range=[0.0372, 0.7741]; y range=[0.0000, 1.0000]
- Validation: n=17626, n_trades=7244, abstain_rate=0.5890, precision=0.8026, win_rate=0.5826
- Validation: avg_ret/trade=0.4860%, net_pnl_total=1347.3576%, profit_factor=2.0890, cal_dev=0.3801
- Holdout: n=4001, n_trades=1094, abstain_rate=0.7266, precision=0.8181, win_rate=0.5430
- Holdout: avg_ret/trade=0.3907%, net_pnl_per_trade=0.0907%, net_pnl_total=99.1994%, profit_factor=1.6423
- Holdout: max_dd=-9.9846%, cal_dev=0.4096, share_long=0.3958, share_short=0.6042

Calibration bins (Isotonic, holdout trades):

| bin | n | mean_predicted | empirical_correct_rate | abs_dev |
| --- | ---: | ---: | ---: | ---: |
| [0.3, 0.4) | 502 | 0.3335 | 0.7430 | 0.4096 |
| [0.4, 0.5) | 233 | 0.4453 | 0.8240 | 0.3787 |
| [0.5, 0.6) | 98 | 0.5595 | 0.9184 | 0.3588 |
| [0.6, 0.7) | 140 | 0.6368 | 0.8929 | 0.2561 |
| [0.7, 0.8) | 71 | 0.7514 | 0.9437 | 0.1923 |
| [0.8, 0.9) | 15 | 0.8596 | 0.8667 | 0.0071 |
| [0.9, 1.0) | 35 | 0.9936 | 1.0000 | 0.0064 |

Isotonic fit notes:
- `isotonic_tau_from_val_post_calibration q=0.6027 tau=0.314342 base_rate_inner=0.397328`

**B2 verdict: REJECT** — binding criterion: `net_pnl_dropped_more_than_5pp_vs_platt`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — isotonic.holdout.cal_dev_post_calibration=0.4096 vs ceiling 0.2
- [PASS] `n_trades>=5` — isotonic.holdout.n_trades=1094 vs floor 5
- [PASS] `net_pnl_pct_total>0` — isotonic.holdout.net_pnl_pct_total=99.1994% vs floor >0.0
- [PASS] `profit_factor>=1.0` — isotonic.holdout.profit_factor=1.6423 vs floor 1.0
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, iso_cal) long=0.9976 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, iso_cal) short=0.9968 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — holdout distinct iso_cal probabilities long=43 vs floor 5
- [PASS] `distinct_holdout_short>=5` — holdout distinct iso_cal probabilities short=66 vs floor 5

REJECT gate evaluation:
- [ok] `cal_dev_holdout_iso_worse_than_platt` — iso_cal_dev_holdout=0.4096 > platt_cal_dev_holdout=0.4568 (isotonic made calibration worse)
- [TRIGGERED] `net_pnl_dropped_more_than_5pp_vs_platt` — net_pnl_pct_total drop = 5.3262pp (platt=104.5256% → iso=99.1994%); threshold = 5.0pp
- [ok] `profit_factor_iso_below_1.0` — isotonic.holdout.profit_factor=1.6423 below 1.0
- [ok] `ranking_integrity_broken` — holdout spearman(raw, iso_cal) long=0.9976, short=0.9968; floor 0.95
- [ok] `leakage_detected` — leakage gate max(train_ts) + tf_ms < min(holdout_ts) failed

Deltas (iso − platt): cal_dev_holdout=-0.0473 (negative = isotonic improved), net_pnl_pct_total=-5.3262pp (reject if drop > 5.0pp)

## Holdout horizon decision

The forward holdout PnL uses the **training-horizon (12 bars / 1h)** forward return for parity with Task B (#655) — the trained heads predict `P(|fwd_return_12bar| > round_trip + margin)`, so a 1-bar evaluation horizon would not match the model's prediction target. The 0.30% round-trip cost is charged once per trade, sourced from `shared/trading-frictions.json` (NOT edited).

## What this report does NOT do

- No champion promotion. The B2 task is comparison-only.
- No threshold / margin / cost edits.
- No holdout swap.
- No automatic follow-up tasks ("no rescue" rule).
- No re-fit of either calibrator on the holdout itself.
