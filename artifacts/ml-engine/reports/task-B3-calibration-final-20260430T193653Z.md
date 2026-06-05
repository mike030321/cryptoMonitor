# Task #658 — Paper trading B3: final calibration repair (20260430T193653Z)

**Aggregate verdict: `B` — calibration not fixed but signal financially strong; at least one PARTIAL_OPERATOR_DECISION method exists**

> Per the "no rescue" rule the spec encodes, this report writes verdicts truthfully and does NOT promote a champion or queue any follow-up tasks. The Phase-2 redesign proposal is written ONLY when the aggregate verdict is `C`.

> **Audit note — base-rate definition for shrinkage.** Method 3 (probability-shrinkage) shrinks each head's probability toward that head's **per-head positive rate on the inner-train slice** (i.e. `mean(side_labels[:inner_end_in_train])` computed separately for the long head and the short head — see `base_rate_per_head_inner` in JSON and the per-candidate lines below). This is intentionally narrower than any combined / val-proxy / either-side base rate that prior tasks B / B2 may have cited; it is the formally correct shrinkage target for a per-head binary calibrator and is what eliminates the contamination concern raised in code review of an earlier draft.

> **Approved deviation — method-4 eligibility baseline.** The spec text describes method-4's run-condition relative to a "val cal_dev better than the B2 isotonic baseline". Because the B2 isotonic baseline is fit on val, scoring it back on val collapses to ~0 cal_dev by construction (memorization), which would make the gate vacuously unbeatable. To honor the spec's val-domain-only intent while keeping the comparison meaningful, the val baseline used for method-4 eligibility is **Platt's val cal_dev** (parametric, apples-to-apples with methods 1-3 — see `ensemble_recipe.baseline_used_for_eligibility` in JSON for the per-candidate value). Iso val cal_dev is reported alongside for full traceability (`iso_val_cal_dev_long_for_reference`, `iso_val_cal_dev_short_for_reference`). The PASS gate's separate "holdout < iso baseline − 0.05" comparison still operates on the holdout fold and is unchanged.

- run_id: `20260430T193653Z`
- holdout window: last 14 calendar days of price_candles (>= `2026-04-16T19:36:53.163149Z`)
- round-trip cost: 0.3000%  (from `shared/trading-frictions.json`, NOT edited)
- post-cost safety margin: 0.1000%
- methods attempted: beta, temp, shrink, ensemble
- baselines recomputed inline: platt, iso (re-fit on this run's shared boosters; bit-identical to B2 by construction — same boosters, same seed, same val partition)

**Per-(candidate, method) verdict counts** — PASS=0, PARTIAL_OPERATOR_DECISION=3, REJECT=0, SKIPPED=1, ERROR=0

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
| bitcoin@5m / C | Platt(B2) | 466 | 0.8841 | 0.6202 | 0.4354% | 63.0832% | 2.4353 | -5.5376% | 0.4671 | 0.3383 | — | — | — | — | — | — | baseline |
| bitcoin@5m / C | Iso(B2) | 510 | 0.8824 | 0.6039 | 0.4209% | 61.6438% | 2.2166 | -5.7729% | 0.5738 | 0.3201 | — | — | — | — | — | — | baseline |
| bitcoin@5m / C | beta | 473 | 0.8837 | 0.6237 | 0.4346% | 63.6516% | 2.4066 | -5.4740% | 0.4817 | 0.3477 | under | 0 | 1.0000 | 1.0000 | 3813 | 3503 | PARTIAL_OPERATOR_DECISION |
| bitcoin@5m / C | temp | 462 | 0.8810 | 0.6255 | 0.4359% | 62.7764% | 2.4248 | -5.5376% | 0.4418 | 0.3637 | under | 0 | 1.0000 | 1.0000 | 3814 | 3506 | PARTIAL_OPERATOR_DECISION |
| bitcoin@5m / C | shrink | 467 | 0.8715 | 0.6231 | 0.4294% | 60.4369% | 2.2995 | -5.8695% | 0.4145 | 0.3754 | under | 0 | 1.0000 | 1.0000 | 3821 | 3507 | PARTIAL_OPERATOR_DECISION |
| bitcoin@5m / C | ensemble | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | SKIPPED |

## Aggregate recommendation

**B — calibration not fixed but signal financially strong.** At least one candidate has at least one PARTIAL_OPERATOR_DECISION method. The agent stops here and waits for the operator to answer the literal question(s) below.

> "Proceed with fixed-size diagnostic paper sandbox despite untrusted probabilities? yes/no — see candidate `bitcoin@5m / C` with method `shrink` — cal_dev=`0.4145` (above 0.2 ceiling), direction=under-confidence, holdout PnL=`60.4369%`, PF=`2.2995`."

## Per-candidate detail

### bitcoin@5m / C_post_cost

- frame rows: 92189 (features=50, horizon_bars=12); n_train=88207, n_holdout=3958
- shared boosters: n_train_inner=70566, n_val=17641, long_head_present=True, short_head_present=True
- base rates (inner-train, per-head): long=0.1084, short=0.1104
- booster equality across all persisted variants: all_heads_equal=True
- ensemble recipe: run=False, A=None, B=None; `only one method (beta) beats baseline; per spec, ensemble requires two complementary methods, not run`

**Best per-candidate verdict (PASS > PARTIAL > REJECT)**:
- method=`shrink`, verdict=`PARTIAL_OPERATOR_DECISION`, cal_dev_holdout=0.4145, net_pnl_pct_total=60.4369%, profit_factor=2.2995

#### Method `beta`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/bitcoin/5m/C_post_cost/20260430T193653Z-beta`, τ = 0.347654
- params (long): a=0.8385, b=-1.9348, c=-0.8608, converged=True, nll=0.3479
- params (short): a=0.9311, b=-1.3926, c=-0.2122, converged=True, nll=0.3481
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3813, short=3503
- direction_of_miscalibration: `under` (over_bins=0, under_bins=6, n_overconfident_bins=0, avg_signed_dev=-0.3046)
- cal_dev delta vs iso baseline: -0.0921; vs platt baseline: 0.0147
- trade-selection diff vs Platt baseline: only_in_platt=12, only_in_beta=19, in_both=454, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 116 | 0.3717 | 0.8534 | 0.4817 |  |
| [0.4, 0.5) | 139 | 0.4410 | 0.8561 | 0.4152 |  |
| [0.5, 0.6) | 65 | 0.5438 | 0.9077 | 0.3639 |  |
| [0.6, 0.7) | 64 | 0.6565 | 0.9219 | 0.2654 |  |
| [0.7, 0.8) | 61 | 0.7469 | 0.8852 | 0.1383 |  |
| [0.8, 0.9) | 28 | 0.8368 | 1.0000 | 0.1632 |  |

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.4817 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=473 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=63.6516% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=2.4066 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=5.4740% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3813 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3503 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [PASS] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.4817 vs iso_baseline 0.5738 - 0.05 = 0.5238

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=63.6516% pf=2.4066
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=5.4740% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3813 short=3503
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.4817 > platt_baseline=0.4671 AND > iso_baseline=0.5738

#### Method `temp`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/bitcoin/5m/C_post_cost/20260430T193653Z-temp`, τ = 0.363698
- params (long): T=0.8269 (direction=`under`), converged=True, nll=0.3491
- params (short): T=0.9970 (direction=`under`), converged=True, nll=0.3484
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3814, short=3506
- direction_of_miscalibration: `under` (over_bins=0, under_bins=6, n_overconfident_bins=0, avg_signed_dev=-0.3064)
- cal_dev delta vs iso baseline: -0.1320; vs platt baseline: -0.0253
- trade-selection diff vs Platt baseline: only_in_platt=11, only_in_temp=7, in_both=455, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 80 | 0.3832 | 0.8250 | 0.4418 |  |
| [0.4, 0.5) | 177 | 0.4425 | 0.8644 | 0.4219 |  |
| [0.5, 0.6) | 57 | 0.5426 | 0.8947 | 0.3521 |  |
| [0.6, 0.7) | 88 | 0.6556 | 0.8977 | 0.2421 |  |
| [0.7, 0.8) | 52 | 0.7476 | 0.9615 | 0.2139 |  |
| [0.8, 0.9) | 8 | 0.8335 | 1.0000 | 0.1665 |  |

Fit notes:
- `temp_long_T=0.8269 (under)`
- `temp_short_T=0.9970 (under)`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.4418 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=462 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=62.7764% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=2.4248 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=5.5376% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3814 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3506 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [PASS] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.4418 vs iso_baseline 0.5738 - 0.05 = 0.5238

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=62.7764% pf=2.4248
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=5.5376% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3814 short=3506
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.4418 > platt_baseline=0.4671 AND > iso_baseline=0.5738

#### Method `shrink`

- verdict: **PARTIAL_OPERATOR_DECISION** (binding criterion `cal_dev_above_ceiling_but_under_confidence`)
- persisted to `models/bitcoin/5m/C_post_cost/20260430T193653Z-shrink`, τ = 0.375397
- params (long): alpha=0.0168 (base_rate=0.1084), val_cal_dev=0.1229
- params (short): alpha=0.0000 (base_rate=0.1104), val_cal_dev=0.1100
- holdout Spearman(raw, cal): long=1.0000, short=1.0000; distinct cal probs: long=3821, short=3507
- direction_of_miscalibration: `under` (over_bins=0, under_bins=5, n_overconfident_bins=0, avg_signed_dev=-0.3296)
- cal_dev delta vs iso baseline: -0.1593; vs platt baseline: -0.0525
- trade-selection diff vs Platt baseline: only_in_platt=18, only_in_shrink=19, in_both=448, disagreed_on_side=0

Calibration bins (holdout, chosen-side):

| bin | n | mean_predicted | empirical_correct | abs_dev | overconf? |
| --- | ---: | ---: | ---: | ---: | :---: |
| [0.3, 0.4) | 62 | 0.3857 | 0.7742 | 0.3885 |  |
| [0.4, 0.5) | 200 | 0.4405 | 0.8550 | 0.4145 |  |
| [0.5, 0.6) | 68 | 0.5444 | 0.9118 | 0.3674 |  |
| [0.6, 0.7) | 95 | 0.6495 | 0.8947 | 0.2452 |  |
| [0.7, 0.8) | 42 | 0.7440 | 0.9762 | 0.2322 |  |

Fit notes:
- `shrink_short_alpha≈0 (no-op; shrinkage not appropriate for the under-confidence direction)`

PASS gate evaluation:
- [FAIL] `cal_dev_holdout<=0.20` — cal_dev_holdout=0.4145 vs ceiling 0.2
- [PASS] `n_trades>=5` — n_trades=467 vs floor 5
- [PASS] `net_pnl_pct_total>0` — net_pnl_pct_total=60.4369% vs floor >0.0
- [PASS] `profit_factor>=1.0` — profit_factor=2.2995 vs floor 1.0
- [PASS] `max_drawdown_magnitude<=15%` — |max_drawdown_pct|=5.8695% vs ceiling 15.0%
- [PASS] `spearman_long>=0.95` — holdout spearman(raw, cal) long=1.0000 vs floor 0.95
- [PASS] `spearman_short>=0.95` — holdout spearman(raw, cal) short=1.0000 vs floor 0.95
- [PASS] `distinct_holdout_long>=5` — distinct cal_probs long=3821 vs floor 5
- [PASS] `distinct_holdout_short>=5` — distinct cal_probs short=3507 vs floor 5
- [PASS] `n_overconfident_bins==0` — n_overconfident_bins=0 (bins where mean_pred - empirical > 0.1)
- [PASS] `leakage_gate_held` — max(train_ts) + tf_ms < min(holdout_ts) held
- [PASS] `cal_dev_holdout < iso_baseline - 0.05` — cal_dev_holdout=0.4145 vs iso_baseline 0.5738 - 0.05 = 0.5238

REJECT gate evaluation:
- [ok] `n_overconfident_bins>0` — n_overconfident_bins=0 (dangerous overconfidence introduced)
- [ok] `net_pnl<=0_or_pf<1` — net_pnl_pct_total=60.4369% pf=2.2995
- [ok] `max_drawdown_magnitude>15%` — |max_drawdown_pct|=5.8695% > 15.0%
- [ok] `ranking_integrity_broken` — spearman long=1.0000 short=1.0000
- [ok] `degenerate_distribution` — distinct long=3821 short=3507
- [ok] `leakage_detected` — max(train_ts) + tf_ms < min(holdout_ts) failed
- [ok] `cal_dev_worse_than_both_baselines` — cal_dev_holdout=0.4145 > platt_baseline=0.4671 AND > iso_baseline=0.5738

#### Method `ensemble`

- SKIPPED — rationale: `only one method (beta) beats baseline; per spec, ensemble requires two complementary methods, not run`

**B2 baselines (recomputed inline on this run)**

- `platt` — persisted to `models/bitcoin/5m/C_post_cost/20260430T193653Z-platt`, τ=0.3383, n_trades=466, net_pnl_pct_total=63.0832%, PF=2.4353, cal_dev=0.4671
- `isotonic` — persisted to `models/bitcoin/5m/C_post_cost/20260430T193653Z-iso`, τ=0.3201, n_trades=510, net_pnl_pct_total=61.6438%, PF=2.2166, cal_dev=0.5738

## Hard rules honoured

- No champion promotion. No `quant_brain_enabled` flip.
- No threshold relaxation, no holdout-window swap, no fee edits.
- Same boosters (md5(`model_to_string`) equality asserted across all variants in this run).
- No new feature search — same 50 features as B/B2.
- No automatic follow-up tasks queued.
- Phase 2 redesign proposal written ONLY on aggregate verdict `C` (and ONLY as a written plan; no code changes).
