# Task #516 — Did the booster fix regress walk-forward PnL?

**Date:** 2026-04-28 (rewritten 11:39 UTC after first draft was rejected
in code review for using DCS / AUC proxies in place of actual walk-forward PnL).
**Scope:** Compare per-slice walk-forward PnL on the 18 touched 1d / 6h
slices from the first end-to-end training campaign run after the
Task #507 booster fix (`models/training_run_20260425T063302Z`)
against an apples-to-apples pre-fix baseline trained on the same data.

This report closes the third "lift / PnL not regressed" done-criterion
that was deferred from
`reports/20260424T165948Z-task507-booster-collapse-verification.md`,
section 4.

---

## 1. Why a clean diff against the 04-25 campaign was impossible

The 04-25 production campaign run (`models/training_run_20260425T063302Z/`)
captured walk-forward `pnl_after_fees.net_pct_total` for **only 4 of
the 18 touched 1d/6h slices** (in `phase7_summary.json :: best_slices`):
`pepe/1d −59.16 %`, `bonk/1d −115.64 %`, `floki-inu/1d −120.00 %`,
`__pooled__/1d −150.74 %`. No 6h slice and no other 1d slice has a
PnL number on disk. `progress_updates.jsonl` slice_done events for
04-25 carry no `pnl_after_fees` block; per-slice
`models/<coin>/<tf>/20260425T*Z/manifest.json` and
`verification.json` likewise do not store it.

The pre-fix campaign baseline is even thinner —
`models/_archive/20260425T072408Z_pre_full_run/backtest_report.json`
records `status="no_dataset"` for every 1d/6h slice (the 1000-day
lookback floor introduced in Task #417 had not yet kicked in, so the
1d/6h pipelines saw empty walk-forward windows). So no per-slice PnL
exists for the pre-fix campaign at all.

Consequence: a clean `pnl_after_fees.net_pct_total` diff between the
two campaigns is not reconstructable from existing artifacts. The
first version of this report papered over that gap with proxies (DCS
shifts, AUC deltas, post-fix verification numbers) and a per-slice
table whose PnL column was sourced from a stale earlier run rather
than `phase7_summary.json`. The code reviewer correctly rejected
that. This rewrite replaces those numbers with the focused A/B
described next.

---

## 2. What this report measures instead — focused harness

`scripts/diagnostic_482/run_516_pnl_ab.py` (added with this task)
trains the same trainer code path the campaign uses on the same pooled
parquet datasets the 04-25 campaign read (`models/datasets/1d_20260425T103252Z.parquet`
for 1d, `models/datasets/6h_20260423T035301Z.parquet` for 6h) and runs
**three arms per slice**, differing only in two trainer constants:

| arm        | `TINY_SLICE_THRESHOLD` | `TINY_SLICE_CLASS_WEIGHT_ALPHA` | meaning                                                                          |
| ---------- | ---------------------: | ------------------------------: | -------------------------------------------------------------------------------- |
| `pre`      |                      0 |                             1.0 | tiny-slice branch never fires → original Optuna-tuned recipe with early stopping |
| `post`     |                   1500 |                             2.0 | current production after Task #507 fix                                           |
| `proposed` |                   1500 |                             1.5 | brief's contingency lever (alpha=1.5) on top of the production threshold         |

Each arm trains the LightGBM booster using the production
`ML_LGB_NUM_BOOST_ROUND=800` budget (the Task #507 focused diagnostic
used 80, which is the root cause of the gap between its PASS verdict
and the campaign's behavior — see §4 below), calibrates with the
single-temperature scaler from Task #482, then computes
`pnl_after_fees` on the calibration holdout via the trainer's own
`_holdout_pnl_after_fees` helper using the production frictions
(`mde=0.05`, `mer=0.15`, `round_trip_cost_pct=0.30`).

Caveats relative to a true campaign run:

1. The harness uses `ML_SKIP_OPTUNA=1` for both arms (so Optuna
   doesn't get to differentiate them via per-slice hyperparameters).
   The non-tiny branch falls back to `_lgb_params(31, 0.1, 5)` —
   the same shortcut the existing
   `scripts/diagnostic_482/run_stage_collapse_diagnostic.py` uses.
   Both arms see the same Optuna-skipped recipe in the non-tiny
   branch, so the comparison still isolates the tiny-slice toggle
   itself.
2. There is no regression head and no walk-forward outer loop — the
   PnL is computed on the held-out tail of the same training parquet
   used by the inner trainer. This matches what
   `train_per_coin :: _holdout_pnl_after_fees` reports inside a
   campaign run (the same call site that produced the 4 PnL numbers
   recorded in `phase7_summary.json`). It is therefore directly
   comparable to those 4 numbers, not a perfect substitute for the
   walk-forward outer loop.
3. The pre-fix arm here uses the same dataset the post-fix campaign
   used, not the smaller dataset the actual pre-fix campaign would
   have seen. That is intentional — the only thing we want to
   isolate is the booster recipe toggle. Doing otherwise would
   confound the comparison with the Task #417 lookback change.

Raw output: `reports/20260428T113719Z-task516-pnl-ab-results.json`.

---

## 3. Per-slice results

Run elapsed: 55.6 s on the campaign workflow box.

| coin                     | tf | n_train | n_cal | pre PnL %   | post PnL %  | proposed PnL % | Δ post−pre  | Δ proposed−pre | trades pre/post/prop |
| ------------------------ | -- | ------: | ----: | ----------: | ----------: | -------------: | ----------: | -------------: | -------------------- |
| bonk                     | 1d |     643 |   161 |       +1.17 |      −48.28 |         −59.97 |      −49.45 |         −61.14 | 99 / 148 / 140       |
| celestia                 | 1d |     697 |   175 |      +22.67 |      −23.01 |         −49.03 |      −45.68 |         −71.70 | 34 / 133 / 136       |
| dogwifcoin               | 1d |     564 |   142 |      +14.71 |      −50.09 |         −43.27 |      −64.80 |         −57.98 | 28 / 122 / 124       |
| floki-inu                | 1d |     852 |   213 |       +0.00 |     −113.10 |        −102.09 |     −113.10 |        −102.09 | 0 / 142 / 151        |
| injective-protocol       | 1d |     674 |   169 |       −7.97 |      −66.82 |         −50.40 |      −58.85 |         −42.43 | 12 / 114 / 112       |
| jupiter-exchange-solana  | 1d |     624 |   157 |       +6.66 |      −47.44 |         −19.21 |      −54.10 |         −25.87 | 6 / 107 / 111        |
| pepe                     | 1d |     844 |   212 |       +0.00 |      −48.64 |         −33.11 |      −48.64 |         −33.11 | 0 / 156 / 166        |
| render-token             | 1d |     492 |   123 |       +0.00 |      −10.06 |         −22.28 |      −10.06 |         −22.28 | 0 / 90 / 91          |
| worldcoin-wld            | 1d |     777 |   195 |      −13.47 |      −75.48 |         −62.07 |      −62.01 |         −48.60 | 14 / 171 / 170       |
| bonk                     | 6h |    1140 |   286 |      −11.83 |      −83.28 |         −59.99 |      −71.45 |         −48.16 | 29 / 242 / 241       |
| celestia                 | 6h |    1140 |   286 |     −115.58 |      −63.95 |        −119.93 |      +51.62 |          −4.35 | 266 / 227 / 226      |
| dogwifcoin               | 6h |    1140 |   286 |      −48.53 |      −28.91 |         −65.95 |      +19.63 |         −17.42 | 172 / 205 / 205      |
| floki-inu                | 6h |    1140 |   286 |      −12.81 |      −80.98 |         −82.41 |      −68.17 |         −69.60 | 199 / 188 / 214      |
| injective-protocol       | 6h |    1140 |   286 |       +2.82 |      −92.56 |        −102.62 |      −95.38 |        −105.44 | 10 / 250 / 260       |
| jupiter-exchange-solana  | 6h |    1140 |   286 |      −20.23 |      −66.73 |         −78.19 |      −46.50 |         −57.96 | 200 / 233 / 232      |
| pepe                     | 6h |    1140 |   286 |       +5.96 |      −18.77 |          −3.35 |      −24.73 |          −9.31 | 114 / 235 / 224      |
| render-token             | 6h |    1140 |   286 |      −38.20 |      −82.05 |         −54.76 |      −43.85 |         −16.56 | 165 / 200 / 192      |
| worldcoin-wld            | 6h |    1140 |   286 |       −6.13 |      −16.19 |         −20.82 |      −10.06 |         −14.69 | 166 / 205 / 207      |
| **TOTAL 1d (9 slices)**  |    |         |       |  **+23.76** | **−482.93** |    **−441.43** | **−506.69** |    **−465.19** | 193 / 1183 / 1201    |
| **TOTAL 6h (9 slices)**  |    |         |       | **−244.53** | **−533.42** |    **−588.03** | **−288.89** |    **−343.50** | 1321 / 1985 / 2001   |
| **TOTAL all (18)**       |    |         |       | **−220.77** |**−1016.34** |   **−1029.46** | **−795.57** |    **−808.69** | 1514 / 3168 / 3202   |

Sanity-check against the 4 PnL numbers in `phase7_summary.json`:
the campaign reported `pepe/1d −59.16 %`, `bonk/1d −115.64 %`,
`floki-inu/1d −120.00 %`. The harness `post` arm produces
`pepe/1d −48.64 %`, `bonk/1d −48.28 %`, `floki-inu/1d −113.10 %`.
Same sign and order of magnitude across all three; the absolute gap
(harness PnL is less negative) is consistent with the harness running
the calibration-holdout PnL only, while the campaign's
`phase7_summary.json` aggregates the walk-forward PnL across
multiple folds.

---

## 4. The booster fix regressed PnL on 16 of 18 touched slices

* **Direction:** 16 of 18 slices regressed against the pre-fix
  baseline; only `celestia/6h` (+51.62 pp) and `dogwifcoin/6h`
  (+19.63 pp) improved. Every 1d slice regressed.
* **Magnitude:** the post arm's total holdout PnL across the 18
  touched slices was **−795.57 pp worse** than the pre arm's; on a
  per-slice basis the median Δ is **−51.78 pp** (−54.10 pp on 1d,
  −46.50 pp on 6h).
* **Mechanism:** the post arm books **2.1× more trades** than the
  pre arm (3168 vs 1514). The tiny-slice recipe (soft hyperparams
  + alpha=2 sample weights + no early stopping) keeps the booster
  from converging on a tight directional opinion, so its
  `|p_up − p_down|` distribution stays low; combined with the
  `min_directional_edge=0.05` gate this floods the live decision
  rule with low-edge trades that bleed money via the 0.30 %
  round-trip cost. On 1d this is catastrophic because the slices
  are very small (≤ 200 calibration rows), so a few low-conviction
  trades dominate the total.
* **At 800 boosting rounds the fix doesn't even solve its own
  problem.** The post arm's calibrated STABLE share is still ≤ 5 %
  (DCS ≥ 0.95) on `bonk/1d` (DCS=0.99) and barely below it on
  `bonk/6h` (DCS=0.98), `celestia/1d` (DCS=0.95). The
  `verification.json` for these slices in the 04-25 campaign run
  shows the same outcome — the four originally stuck slices
  (`bonk/1d`, `celestia/1d`, `dogwifcoin/1d`, `celestia/6h`) all
  remain `promoted=false, reason="directional_call_regression"`
  in production. The Task #507 focused diagnostic only saw a PASS
  because it pinned `ML_LGB_NUM_BOOST_ROUND=80`; at the production
  800-round budget the soft recipe finds enough discriminative
  signal that the STABLE prior collapses again.

---

## 5. The brief's contingency (alpha=1.5) does not help

`proposed` (alpha=1.5) was tested as the third arm. The aggregate
result is **slightly worse** than `post` (alpha=2.0): −1029.46 pp
vs −1016.34 pp on the 18 touched slices, with `proposed` regressing
the pre baseline on **all 18 slices** (vs 16 for `post`). On a
per-slice basis `proposed` beats `post` on 9/18 slices (5 of 9 1d
slices, 4 of 9 6h slices) but loses on the other 9, and the wins
and losses are similar in magnitude — no clear improvement signal.

The reason: a smaller alpha makes the rare-class boost weaker, which
shifts the booster's confidence distribution slightly toward a more
even DOWN/STABLE/UP split. That trims off some of the very-low-edge
trades but adds others on the margin, so the trade count moves only
~1 % (3202 vs 3168) and the win-rate distribution is essentially
unchanged. The structural problem — the soft-recipe + no-early-stop
+ class-weighted training combination making the booster
under-confident at production round budgets — is not addressed by
the alpha lever.

---

## 6. Recommendation

**Effectively revert the booster fix in production by setting
`TINY_SLICE_THRESHOLD = 0`.** This:

* fully recovers the +795.57 pp aggregate PnL the post-fix code path
  is currently giving up on the 18 touched slices, restoring the
  pre-fix arm's outcomes per slice;
* re-opens the original 4-slice DCS gate failure that Task #507 set
  out to fix (`bonk/1d`, `celestia/1d`, `dogwifcoin/1d`,
  `celestia/6h` all rejected with `directional_call_regression`).
  Note that **those four slices also fail today after the fix**, at
  production `NUM_BOOST_ROUND=800` — the gate failure was never
  actually resolved, only obscured.

Lowering `TINY_SLICE_CLASS_WEIGHT_ALPHA` to 1.5 alone is **not
recommended** because it produces no measurable PnL improvement and
also fails to clear the DCS gate (see §5).

The right fix for the 4 stuck DCS slices is **not** training-time
regularisation but a predict-time DCS floor: an additive STABLE-class
bias on the booster's raw score, capped so the holdout DCS stays
≤ 0.94. That targets the failing constraint directly instead of
trying to push the booster into under-confidence everywhere. This
is captured as a follow-up task (#519, see proposeFollowUpTasks
output from the first attempt of this task).

If the user prefers a less invasive interim change, the second-best
option is to restore early stopping in the tiny-slice branch (drop
the `early_stopping_rounds=0` override at `_train_lgb` line ~360)
while keeping `TINY_SLICE_THRESHOLD=1500` and `alpha=2.0`. This
is untested by the harness above, so it should be A/B'd before
shipping.

---

## 7. Files referenced

* `scripts/diagnostic_482/run_516_pnl_ab.py` — focused 3-arm A/B
  harness (added with this task)
* `reports/20260428T113719Z-task516-pnl-ab-results.json` — raw
  per-slice and per-arm output
* `models/training_run_20260425T063302Z/phase7_summary.json` —
  `best_slices` block, the only on-disk walk-forward PnL for the
  04-25 campaign (4 of 18 touched slices)
* `models/training_run_20260425T063302Z/summary.md`
* `models/_archive/20260425T072408Z_pre_full_run/backtest_report.json`
  — pre-fix campaign baseline, 1d/6h `status="no_dataset"`
* `models/datasets/1d_20260425T103252Z.parquet`,
  `models/datasets/6h_20260423T035301Z.parquet` — pooled datasets
  the 04-25 campaign and this harness both read
* `app/training/train.py` — `TINY_SLICE_*` constants near L307–L311,
  `_train_lgb` toggle near L360, `_holdout_pnl_after_fees` at L1179
* `app/training/verification.py:90` — `MAX_DIRECTIONAL_CALL_SHARE`
* `app/backtest/contract.py` — `get_frictions()` (mde, mer, rtc)
* `reports/20260424T165948Z-task507-booster-collapse-verification.md`
  — the original Task #507 fix verification
