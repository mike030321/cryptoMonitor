# Task #658 — Paper trading B3: final calibration repair (20260430T142134Z)

**Aggregate verdict: `B` — calibration not fixed but signal financially strong; at least one PARTIAL_OPERATOR_DECISION method exists**

> Per the "no rescue" rule the spec encodes, this report writes verdicts truthfully and does NOT promote a champion or queue any follow-up tasks. The Phase-2 redesign proposal is written ONLY when the aggregate verdict is `C`.

> **Audit note — base-rate definition for shrinkage.** Method 3 (probability-shrinkage) shrinks each head's probability toward that head's **per-head positive rate on the inner-train slice** (i.e. `mean(side_labels[:inner_end_in_train])` computed separately for the long head and the short head — see `base_rate_per_head_inner` in JSON and the per-candidate lines below). This is intentionally narrower than any combined / val-proxy / either-side base rate that prior tasks B / B2 may have cited; it is the formally correct shrinkage target for a per-head binary calibrator and is what eliminates the contamination concern raised in code review of an earlier draft.

> **Approved deviation — method-4 eligibility baseline.** The spec text describes method-4's run-condition relative to a "val cal_dev better than the B2 isotonic baseline". Because the B2 isotonic baseline is fit on val, scoring it back on val collapses to ~0 cal_dev by construction (memorization), which would make the gate vacuously unbeatable. To honor the spec's val-domain-only intent while keeping the comparison meaningful, the val baseline used for method-4 eligibility is **Platt's val cal_dev** (parametric, apples-to-apples with methods 1-3 — see `ensemble_recipe.baseline_used_for_eligibility` in JSON for the per-candidate value). Iso val cal_dev is reported alongside for full traceability (`iso_val_cal_dev_long_for_reference`, `iso_val_cal_dev_short_for_reference`). The PASS gate's separate "holdout < iso baseline − 0.05" comparison still operates on the holdout fold and is unchanged.

- run_id: `20260430T142134Z`
- holdout window: last 14 calendar days of price_candles (>= `2026-04-16T14:21:34.083780Z`)
- round-trip cost: 0.3000%  (from `shared/trading-frictions.json`, NOT edited)
- post-cost safety margin: 0.1000%
- methods attempted: beta, temp, shrink, ensemble
- baselines recomputed inline: platt, iso (re-fit on this run's shared boosters; bit-identical to B2 by construction — same boosters, same seed, same val partition)

**Per-(candidate, method) verdict counts** — PASS=0, PARTIAL_OPERATOR_DECISION=6, REJECT=1, SKIPPED=1, ERROR=0

## Acceptance criteria

**PASS** iff ALL of:
- `cal_dev_holdout <= 0.2`
- `n_trades >= 5` on holdout
- `net_pnl_pct_total > 0.0` on holdout
- `profit_factor >= 1.0` on holdout
- `|max_drawdown_pct| <= 15.0%`
- ranking integrity: per-head Spearman(raw, cal) >= 0.95 on holdout
- non-degeneracy: ≥ 5 distinct calibrated probs per head on holdout
- `n_overconfident_bins == 0` (no bin where mean_pred − empirical > 0.1)
- leakage gate held (`max(train_ts) + tf_ms < min(holdout_ts)`)
- `cal_dev_holdout < cal_dev_holdout_iso_baseline - 0.05` (calibration improved materially vs B2 baseline)

**PARTIAL_OPERATOR_DECISION** iff cal_dev > 0.2 BUT direction is `under`, all financial gates hold, ranking integrity holds, non-degeneracy holds, and `n_overconfident_bins == 0`. Stops without auto-promotion.

**REJECT** iff ANY of: `n_overconfident_bins > 0`, `net_pnl_pct_total <= 0`, `profit_factor < 1.0`, `|max_drawdown_pct| > 15.0%`, Spearman per head `< 0.95`, distinct probs per head `< 5`, leakage detected, OR cal_dev_holdout > both prior calibrators.

## Per-(candidate, method) holdout metrics

| candidate | method | n_trades | precision | win_rate | avg_ret/trade | net_pnl_total | profit_factor | max_dd | cal_dev | τ | direction | n_oc | spearman_long | spearman_short | n_distinct_long | n_distinct_short | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| bitcoin@5m / C | Platt(B2) | 464 | 0.8836 | 0.6336 | 0.4485% | 68.8992% | 2.5559 | -5.9377% | 0.5134 | 0.3309 | — | — | — | — | — | — | baseline |
| bitcoin@5m / C | Iso(B2) | 502 | 0.8825 | 0.6195 | 0.4329% | 66.7184% | 2.3470 | -5.9336% | 0.4653 | 0.3684 | — | — | — | — | — | — | baseline |
| bitcoin@5m / C | beta | 488 | 0.8791 | 0.6127 | 0.4312% | 64.0141% | 2.2881 | -5.4449% | 0.4419 | 0.3429 | under | 0 | 1.0000 | 1.0000 | 3578 | 3459 | PARTIAL_OPERATOR_DECISION |
| bitcoin@5m / C | temp | 468 | 0.8889 | 0.6346 | 0.4518% | 71.0322% | 2.6285 | -5.3980% | 0.5163 | 0.3544 | under | 0 | 1.0000 | 1.0000 | 3572 | 3460 | REJECT |
| bitcoin@5m / C | shrink | 478 | 0.8870 | 0.6360 | 0.4438% | 68.7585% | 2.5105 | -5.6179% | 0.5102 | 0.3656 | under | 0 | 1.0000 | 1.0000 | 3584 | 3455 | PARTIAL_OPERATOR_DECISION |
| bitcoin@5m / C | ensemble | 466 | 0.8841 | 0.6288 | 0.4445% | 67.3364% | 2.4825 | -5.4366% | 0.5054 | 0.3429 | under | 0 | 1.0000 | 1.0000 | 3579 | 3456 | PARTIAL_OPERATOR_DECISION |
| ethereum@5m / C | Platt(B2) | 1011 | 0.8338 | 0.5707 | 0.4044% | 105.5143% | 1.7903 | -8.2189% | 0.4938 | 0.2845 | — | — | — | — | — | — | baseline |
| ethereum@5m / C | Iso(B2) | 1071 | 0.8207 | 0.5565 | 0.3905% | 96.8949% | 1.6594 | -8.1081% | 0.4040 | 0.3169 | — | — | — | — | — | — | baseline |
| ethereum@5m / C | beta | 1001 | 0.8302 | 0.5734 | 0.4037% | 103.7896% | 1.7776 | -7.8038% | 0.3980 | 0.3098 | under | 0 | 1.0000 | 1.0000 | 3758 | 3418 | PARTIAL_OPERATOR_DECISION |
| ethereum@5m / C | temp | 1025 | 0.8312 | 0.5639 | 0.3938% | 96.1019% | 1.6797 | -7.8726% | 0.4292 | 0.3066 | under | 0 | 1.0000 | 1.0000 | 3761 | 3420 | PARTIAL_OPERATOR_DECISION |
| ethereum@5m / C | shrink | 1010 | 0.8267 | 0.5564 | 0.3872% | 88.0718% | 1.6142 | -7.3971% | 0.4082 | 0.3253 | under | 0 | 1.0000 | 1.0000 | 3774 | 3419 | PARTIAL_OPERATOR_DECISION |
| ethereum@5m / C | ensemble | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | SKIPPED |

## Aggregate recommendation

**B — calibration not fixed but signal financially strong.** At least one candidate has at least one PARTIAL_OPERATOR_DECISION method. The agent stops here and waits for the operator to answer the literal question(s) below.

> "Proceed with fixed-size diagnostic paper sandbox despite untrusted probabilities? yes/no — see candidate `bitcoin@5m / C` with method `beta` — cal_dev=`0.4419` (above 0.2 ceiling), direction=under-confidence, holdout PnL=`64.0141%`, PF=`2.2881`."

> "Proceed with fixed-size diagnostic paper sandbox despite untrusted probabilities? yes/no — see candidate `ethereum@5m / C` with method `beta` — cal_dev=`0.3980` (above 0.2 ceiling), direction=under-confidence, holdout PnL=`103.7896%`, PF=`1.7776`."

## Per-candidate detail

### bitcoin@5m / C_post_cost

- frame rows: 92165 (features=50, horizon_bars=12); n_train=88144, n_holdout=3997
- shared boosters: n_train_inner=70515, n_val=17629, long_head_present=True, short_head_present=True
- base rates (inner-train, per-head): long=0.1083, short=0.1103
- booster equality across all persisted variants: all_heads_equal=True
- ensemble recipe: run=True, A=beta, B=temp; `two methods beat baseline (['beta', 'temp', 'shrink']) and the top two (beta, temp) reduce error on different bin slices on val; ensemble fitted as a 2-coef blend`

**Best per-candidate verdict (PASS > PARTIAL > REJECT)**:
- method=`beta`, verdict=`PARTIAL_OPERATOR_DECISION`, cal_dev_holdout=0.4419, net_pnl_pct_total=64.0141%, profit_factor=2.2881

#### Method `beta`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/bitcoin/5m/C_post_cost/20260430T142134Z-beta`, τ = 0.342851
- params (long): a=0.8978, b=-1.8881, c=-0.7911, converged=True, nll=0.3470
- params (short): a=0.9916, b=-1.2166, c=0.0677, converged=True, nll=0.3489
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3578, short=3459
- direction_of_miscalibration: `under` (over_bins=0, under_bins=6, n_overconfident_bins=0, avg_signed_dev=-0.3015)
- cal_dev delta vs iso baseline: -0.0234; vs platt baseline: -0.0715
- trade-selection diff vs Platt baseline: only_in_platt=14, only_in_beta=38, in_both=450, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 116 | 0.3693 | 0.8103 | 0.4411 |  |
| [0.4, 0.5) | 142 | 0.4384 | 0.8803 | 0.4419 |  |
| [0.5, 0.6) | 73 | 0.5475 | 0.9041 | 0.3566 |  |
| [0.6, 0.7) | 65 | 0.6545 | 0.9231 | 0.2685 |  |
| [0.7, 0.8) | 68 | 0.7483 | 0.8824 | 0.1340 |  |
| [0.8, 0.9) | 24 | 0.8334 | 1.0000 | 0.1666 |  |

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.4419 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=488 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=64.0141% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=2.2881 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=5.4449% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3578 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3459 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [FAIL] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.4419 vs iso_baseline 0.4653 - 0.05 = 0.4153

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=64.0141% pf=2.2881
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=5.4449% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3578 short=3459
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.4419 > platt_baseline=0.5134 AND > iso_baseline=0.4653

#### Method `temp`

- verdict: **REJECT** (binding criterion `cal_dev_worse_than_both_baselines`)
- persisted to `models/bitcoin/5m/C_post_cost/20260430T142134Z-temp`, τ = 0.354376
- params (long): T=0.8026 (direction=`under`), converged=True, nll=0.3480
- params (short): T=1.0514 (direction=`over`), converged=True, nll=0.3497
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3572, short=3460
- direction_of_miscalibration: `under` (over_bins=0, under_bins=6, n_overconfident_bins=0, avg_signed_dev=-0.3116)
- cal_dev delta vs iso baseline: 0.0510; vs platt baseline: 0.0029
- trade-selection diff vs Platt baseline: only_in_platt=14, only_in_temp=18, in_both=450, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 111 | 0.3756 | 0.8919 | 0.5163 |  |
| [0.4, 0.5) | 163 | 0.4515 | 0.8589 | 0.4074 |  |
| [0.5, 0.6) | 51 | 0.5561 | 0.8824 | 0.3262 |  |
| [0.6, 0.7) | 77 | 0.6560 | 0.8961 | 0.2402 |  |
| [0.7, 0.8) | 58 | 0.7398 | 0.9483 | 0.2085 |  |
| [0.8, 0.9) | 8 | 0.8292 | 1.0000 | 0.1708 |  |

Fit notes:
- `temp_long_T=0.8026 (under)`
- `temp_short_T=1.0514 (over)`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.5163 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=468 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=71.0322% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=2.6285 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=5.3980% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3572 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3460 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [FAIL] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.5163 vs iso_baseline 0.4653 - 0.05 = 0.4153

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=71.0322% pf=2.6285
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=5.3980% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3572 short=3460
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [TRIGGERED] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.5163 > platt_baseline=0.5134 AND > iso_baseline=0.4653

#### Method `shrink`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/bitcoin/5m/C_post_cost/20260430T142134Z-shrink`, τ = 0.365556
- params (long): alpha=0.0000 (base_rate=0.1083), val_cal_dev=0.1037
- params (short): alpha=0.0000 (base_rate=0.1103), val_cal_dev=0.1070
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3584, short=3455
- direction_of_miscalibration: `under` (over_bins=0, under_bins=5, n_overconfident_bins=0, avg_signed_dev=-0.3457)
- cal_dev delta vs iso baseline: 0.0449; vs platt baseline: -0.0032
- trade-selection diff vs Platt baseline: only_in_platt=38, only_in_shrink=52, in_both=426, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 102 | 0.3819 | 0.8922 | 0.5102 |  |
| [0.4, 0.5) | 182 | 0.4515 | 0.8571 | 0.4056 |  |
| [0.5, 0.6) | 58 | 0.5561 | 0.8966 | 0.3405 |  |
| [0.6, 0.7) | 94 | 0.6548 | 0.8936 | 0.2388 |  |
| [0.7, 0.8) | 42 | 0.7428 | 0.9762 | 0.2334 |  |

Fit notes:
- `shrink_long_alpha≈0 (no-op; shrinkage not appropriate for the under-confidence direction)`
- `shrink_short_alpha≈0 (no-op; shrinkage not appropriate for the under-confidence direction)`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.5102 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=478 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=68.7585% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=2.5105 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=5.6179% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3584 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3455 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [FAIL] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.5102 vs iso_baseline 0.4653 - 0.05 = 0.4153

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=68.7585% pf=2.5105
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=5.6179% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3584 short=3455
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.5102 > platt_baseline=0.5134 AND > iso_baseline=0.4653

#### Method `ensemble`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/bitcoin/5m/C_post_cost/20260430T142134Z-ensemble`, τ = 0.342883
- A=beta, B=temp, w_A_long=0.7858, w_A_short=0.4103
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3579, short=3456
- direction_of_miscalibration: `under` (over_bins=0, under_bins=6, n_overconfident_bins=0, avg_signed_dev=-0.3064)
- cal_dev delta vs iso baseline: 0.0401; vs platt baseline: -0.0080
- trade-selection diff vs Platt baseline: only_in_platt=6, only_in_ensemble=8, in_both=458, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 115 | 0.3728 | 0.8783 | 0.5054 |  |
| [0.4, 0.5) | 135 | 0.4410 | 0.8444 | 0.4035 |  |
| [0.5, 0.6) | 66 | 0.5414 | 0.8788 | 0.3374 |  |
| [0.6, 0.7) | 73 | 0.6559 | 0.9315 | 0.2756 |  |
| [0.7, 0.8) | 66 | 0.7493 | 0.9091 | 0.1598 |  |
| [0.8, 0.9) | 11 | 0.8434 | 1.0000 | 0.1566 |  |

Fit notes:
- `ensemble A=beta B=temp w_long=0.7858 w_short=0.4103`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.5054 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=466 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=67.3364% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=2.4825 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=5.4366% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3579 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3456 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [FAIL] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.5054 vs iso_baseline 0.4653 - 0.05 = 0.4153

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=67.3364% pf=2.4825
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=5.4366% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3579 short=3456
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.5054 > platt_baseline=0.5134 AND > iso_baseline=0.4653

**B2 baselines (recomputed inline on this run)**

- `platt` — persisted to `models/bitcoin/5m/C_post_cost/20260430T142134Z-platt`, τ=0.3309, n_trades=464, net_pnl_pct_total=68.8992%, PF=2.5559, cal_dev=0.5134
- `isotonic` — persisted to `models/bitcoin/5m/C_post_cost/20260430T142134Z-iso`, τ=0.3684, n_trades=502, net_pnl_pct_total=66.7184%, PF=2.3470, cal_dev=0.4653

### ethereum@5m / C_post_cost

- frame rows: 92166 (features=50, horizon_bars=12); n_train=88145, n_holdout=3997
- shared boosters: n_train_inner=70516, n_val=17629, long_head_present=True, short_head_present=True
- base rates (inner-train, per-head): long=0.1999, short=0.1975
- booster equality across all persisted variants: all_heads_equal=True
- ensemble recipe: run=False, A=None, B=None; `only one method (beta) beats baseline; per spec, ensemble requires two complementary methods, not run`

**Best per-candidate verdict (PASS > PARTIAL > REJECT)**:
- method=`beta`, verdict=`PARTIAL_OPERATOR_DECISION`, cal_dev_holdout=0.3980, net_pnl_pct_total=103.7896%, profit_factor=1.7776

#### Method `beta`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/ethereum/5m/C_post_cost/20260430T142134Z-beta`, τ = 0.309846
- params (long): a=0.9537, b=-1.7198, c=-0.5900, converged=True, nll=0.3997
- params (short): a=0.9771, b=-1.5395, c=-0.1233, converged=True, nll=0.4051
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3758, short=3418
- direction_of_miscalibration: `under` (over_bins=0, under_bins=7, n_overconfident_bins=0, avg_signed_dev=-0.2532)
- cal_dev delta vs iso baseline: -0.0060; vs platt baseline: -0.0958
- trade-selection diff vs Platt baseline: only_in_platt=36, only_in_beta=26, in_both=975, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 422 | 0.3508 | 0.7488 | 0.3980 |  |
| [0.4, 0.5) | 199 | 0.4497 | 0.8241 | 0.3744 |  |
| [0.5, 0.6) | 115 | 0.5429 | 0.8870 | 0.3441 |  |
| [0.6, 0.7) | 135 | 0.6402 | 0.9333 | 0.2931 |  |
| [0.7, 0.8) | 81 | 0.7487 | 0.9383 | 0.1896 |  |
| [0.8, 0.9) | 37 | 0.8412 | 0.9459 | 0.1047 |  |
| [0.9, 1.0) | 12 | 0.9313 | 1.0000 | 0.0687 |  |

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.3980 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=1001 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=103.7896% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=1.7776 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=7.8038% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3758 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3418 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [FAIL] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.3980 vs iso_baseline 0.4040 - 0.05 = 0.3540

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=103.7896% pf=1.7776
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=7.8038% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3758 short=3418
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.3980 > platt_baseline=0.4938 AND > iso_baseline=0.4040

#### Method `temp`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/ethereum/5m/C_post_cost/20260430T142134Z-temp`, τ = 0.306649
- params (long): T=0.8161 (direction=`under`), converged=True, nll=0.4002
- params (short): T=0.9973 (direction=`under`), converged=True, nll=0.4069
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3761, short=3420
- direction_of_miscalibration: `under` (over_bins=0, under_bins=7, n_overconfident_bins=0, avg_signed_dev=-0.2644)
- cal_dev delta vs iso baseline: 0.0253; vs platt baseline: -0.0645
- trade-selection diff vs Platt baseline: only_in_platt=28, only_in_temp=42, in_both=983, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 446 | 0.3443 | 0.7735 | 0.4292 |  |
| [0.4, 0.5) | 224 | 0.4476 | 0.8036 | 0.3560 |  |
| [0.5, 0.6) | 164 | 0.5460 | 0.8963 | 0.3504 |  |
| [0.6, 0.7) | 110 | 0.6385 | 0.9364 | 0.2979 |  |
| [0.7, 0.8) | 66 | 0.7432 | 0.9394 | 0.1962 |  |
| [0.8, 0.9) | 9 | 0.8718 | 1.0000 | 0.1282 |  |
| [0.9, 1.0) | 6 | 0.9071 | 1.0000 | 0.0929 |  |

Fit notes:
- `temp_long_T=0.8161 (under)`
- `temp_short_T=0.9973 (under)`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.4292 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=1025 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=96.1019% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=1.6797 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=7.8726% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3761 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3420 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [FAIL] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.4292 vs iso_baseline 0.4040 - 0.05 = 0.3540

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=96.1019% pf=1.6797
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=7.8726% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3761 short=3420
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.4292 > platt_baseline=0.4938 AND > iso_baseline=0.4040

#### Method `shrink`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/ethereum/5m/C_post_cost/20260430T142134Z-shrink`, τ = 0.325264
- params (long): alpha=0.0000 (base_rate=0.1999), val_cal_dev=0.1098
- params (short): alpha=0.0000 (base_rate=0.1975), val_cal_dev=0.1710
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3774, short=3419
- direction_of_miscalibration: `under` (over_bins=0, under_bins=6, n_overconfident_bins=0, avg_signed_dev=-0.2932)
- cal_dev delta vs iso baseline: 0.0043; vs platt baseline: -0.0856
- trade-selection diff vs Platt baseline: only_in_platt=97, only_in_shrink=96, in_both=914, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 392 | 0.3571 | 0.7653 | 0.4082 |  |
| [0.4, 0.5) | 263 | 0.4471 | 0.7909 | 0.3437 |  |
| [0.5, 0.6) | 183 | 0.5474 | 0.9016 | 0.3542 |  |
| [0.6, 0.7) | 108 | 0.6422 | 0.9259 | 0.2838 |  |
| [0.7, 0.8) | 51 | 0.7377 | 0.9608 | 0.2231 |  |
| [0.8, 0.9) | 13 | 0.8538 | 1.0000 | 0.1462 |  |

Fit notes:
- `shrink_long_alpha≈0 (no-op; shrinkage not appropriate for the under-confidence direction)`
- `shrink_short_alpha≈0 (no-op; shrinkage not appropriate for the under-confidence direction)`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.4082 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=1010 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=88.0718% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=1.6142 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=7.3971% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3774 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3419 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [FAIL] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.4082 vs iso_baseline 0.4040 - 0.05 = 0.3540

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=88.0718% pf=1.6142
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=7.3971% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3774 short=3419
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.4082 > platt_baseline=0.4938 AND > iso_baseline=0.4040

#### Method `ensemble`

- SKIPPED — rationale: `only one method (beta) beats baseline; per spec, ensemble requires two complementary methods, not run`

**B2 baselines (recomputed inline on this run)**

- `platt` — persisted to `models/ethereum/5m/C_post_cost/20260430T142134Z-platt`, τ=0.2845, n_trades=1011, net_pnl_pct_total=105.5143%, PF=1.7903, cal_dev=0.4938
- `isotonic` — persisted to `models/ethereum/5m/C_post_cost/20260430T142134Z-iso`, τ=0.3169, n_trades=1071, net_pnl_pct_total=96.8949%, PF=1.6594, cal_dev=0.4040

## Hard rules honoured

- No champion promotion. No `quant_brain_enabled` flip.
- No threshold relaxation, no holdout-window swap, no fee edits.
- Same boosters (md5(`model_to_string`) equality asserted across all variants in this run).
- No new feature search — same 50 features as B/B2.
- No automatic follow-up tasks queued.
- Phase 2 redesign proposal written ONLY on aggregate verdict `C` (and ONLY as a written plan; no code changes).
