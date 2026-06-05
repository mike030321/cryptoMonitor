# Task #598 — Why 1h and 2h still fail the stage-2 gate (20260429T040031Z)

Follow-up to task #592 (`20260429T033030Z-task592-1h2h-stage2-verdict.md`). The
parallel re-run confirmed that turning the calibration flag ON moves
trade_share into the gate band [0.40, 0.85] for 19/19 1h+2h slices, but **0/19
clear the FULL gate** — the admitted-feature DA lift is too small. The same
admitted stack passes the trade_share check on 6h/1d (task #587 verdict) and
hits a similar DA-lift problem there, but the structural gap is widest on
1h/2h. This diagnostic explains why and proposes one concrete next iteration.

## Inputs
- Latest snapshots task #592 used:
  - 1h: `models/datasets/1h_20260429T032448Z.parquet` (n=77,247)
  - 2h: `models/datasets/2h_20260428T215813Z.parquet` (n=40,368)
  - 6h: `models/datasets/6h_20260428T212141Z.parquet` (n=13,203)
  - 1d: `models/datasets/1d_20260428T133304Z.parquet`  (n=7,864)
- Admitted stack (11 features) from task #580 / unchanged for #587 + #592:
  `atr_pct_zscore_60`, `bb_pctb_extreme`, `drawdown_30`, `ema_spread_per_atr`,
  `macd_signal_cross_strength`, `realizedVol_log`, `ret1_squared`,
  `ret1_x_volZ60`, `ret5_minus_ret10`, `rsi14_centered_squared`, `vol_of_vol_30`.
- Round-trip cost: 0.30% (`shared/trading-frictions.json` → 2*(taker+slippage)).

## Finding 1 — The PnL gate scores 1-bar return but the model is trained against multi-bar return; on 1h/2h, 30% of "correct" calls become losers
Per task #379, short-tf labels (1h/2h/6h) span `FORWARD_HORIZON_CANDLES = 4`
forward bars (`MULTI_BAR_LABEL_THRESHOLDS_PERCENT` in
`app/training/labels.py` = `{1h: 0.40 %, 2h: 0.50 %, 6h: 0.80 %}`). The
persisted snapshot carries BOTH columns:

* `forward_return` — 1-bar return,  `(close[i+1] − close[i]) / close[i]`.
* `directional_label_forward_return` — multi-bar return at the directional
  horizon, `(close[i+H] − close[i]) / close[i]`.

`label_3class` is the SIGN of `directional_label_forward_return` (vs the
multi-bar threshold), so the LightGBM is trained to predict the 4-bar
direction at 1h. But `score_fold` in `scripts/feature_edge_search/run_search.py`
uses `forward_return` (1-bar) for PnL — so the gate's `post_fee_pnl_pct_total`
and the gate's DA are evaluated on the **1-bar** outcome. On 1h/2h/6h the
two horizons disagree often:

| tf | label horizon (bars) | mean \|fwd_1bar\| (dir-lbl rows) | mean \|dir_multibar\| (dir-lbl rows) | share of dir-lbl rows where 1-bar sign == multi-bar sign |
|---|---|---|---|---|
| 1h | 4 | **0.83 %** | 1.93 % | **0.699** |
| 2h | 4 | 1.15 % | 2.66 % | 0.701 |
| 6h | 4 | 2.00 % | 4.68 % | 0.696 |
| 1d | **1** | 5.59 % | 5.59 % | **1.000** |

Translation: on 1h, **30 % of "correctly predicted by the model" rows are
actually money-losers when scored on 1 bar** — even before any DA improvement
from the admitted stack matters. On 1d the two horizons are identical (`H=1`)
so this mismatch is zero. The split timeframe data shows a single mismatched
contract on 1h/2h/6h but a coherent contract on 1d.

This alone explains the headline gap: the gate is asking the model "did you
predict the next 1 bar correctly" while the model was optimized to answer
"did you predict where price will be in 4 bars". Even a perfect 4-bar
classifier would lose ~30 % of its calls on the 1-bar PnL account, which —
combined with the 0.30 % round-trip cost — caps the achievable post-fee PnL.

## Finding 2 — Per-tf payout-vs-cost geometry makes the gate structurally hard at 1h/2h
Even after fixing Finding 1, the PnL gate requires net-positive trades
after a 0.30 % round-trip cost. The DA needed to break even at the mean
directional payout falls out of the EV identity
`p · (payout − cost) − (1 − p) · (payout + cost) = 0` →
`p = 0.5 + cost / (2 · payout)`:

| tf | mean \|payout\| (1-bar, dir-lbl rows) | round-trip cost | break-even DA on 1-bar PnL | mean \|payout\| (multi-bar) | break-even DA on multi-bar PnL |
|---|---|---|---|---|---|
| 1h | 0.83 % | 0.30 % | **0.680** | 1.93 % | 0.578 |
| 2h | 1.15 % | 0.30 % | 0.631 | 2.66 % | 0.556 |
| 6h | 2.00 % | 0.30 % | 0.575 | 4.68 % | 0.532 |
| 1d | 5.59 % | 0.30 % | **0.527** | 5.59 % | 0.527 |

The single-fold DA the booster actually achieves is in the 0.42–0.50 range
across all four timeframes. Under the **current 1-bar gate**, 1h needs a
~+18-point DA jump and 1d a ~+4-point jump. Under a **multi-bar gate**
(the horizon the booster is actually trained for), the gap on 1h shrinks to
~+8 points and the gap on 1d is unchanged. Same booster, same data — but
the contract change makes 1h a feasible target.

## Finding 3 — The admitted stack is selected for 6h/1d edge, then force-fed into 1h/2h
Per-feature, single-fold ablation against the registry baseline (75/25
time-ordered split, pooled across coins, same LightGBM hyperparameters as
the search runner). DA is the gate definition (correct call on
non-STABLE-truth rows, scored under the current 1-bar contract). `*` marks
the 11 admitted features.

| candidate | 1h dDA | 2h dDA | 6h dDA | 1d dDA |
|---|---|---|---|---|
| `vol_of_vol_30` * | −0.0037 | −0.0048 | −0.0029 | −0.0131 |
| `realizedVol_log` * | +0.0014 | +0.0053 | +0.0033 | −0.0104 |
| `ret1_squared` * | **−0.0069** | +0.0010 | −0.0054 | **−0.0201** |
| `ret5_minus_ret10` * | **−0.0065** | +0.0029 | −0.0014 | −0.0104 |
| `rsi14_centered_squared` * | +0.0032 | −0.0021 | +0.0007 | +0.0028 |
| `bb_pctb_extreme` * | +0.0030 | +0.0052 | −0.0004 | −0.0076 |
| `macd_hist_norm_atr` | −0.0053 | +0.0035 | +0.0011 | −0.0173 |
| `vol_zscore60_squared` | −0.0014 | +0.0047 | −0.0025 | −0.0007 |
| `ema_spread_per_atr` * | −0.0008 | −0.0029 | +0.0033 | +0.0007 |
| `atr_pct_zscore_60` * | **−0.0120** | **−0.0108** | −0.0018 | −0.0104 |
| `drawdown_30` * | −0.0003 | +0.0005 | −0.0018 | +0.0041 |
| `btc_lead_x_self_ret` | +0.0000 | +0.0000 | −0.0080 | +0.0000 |
| `eth_lead_minus_btc_lead` | +0.0000 | +0.0000 | −0.0022 | +0.0000 |
| `macd_signal_cross_strength` * | **−0.0054** | n/a | +0.0022 | −0.0035 |
| `ret1_x_volZ60` * | **−0.0044** | +0.0029 | −0.0072 | −0.0028 |
| `log_liquidations_self` | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

(`btc_lead_*` / `eth_lead_*` / `log_liquidations_self` collapse to "no
effect" because the source columns are mostly NaN/zero in the persisted
snapshots — already noted in `candidates.json`.)

Six of the eleven admitted features (`vol_of_vol_30`, `ret1_squared`,
`ret5_minus_ret10`, `atr_pct_zscore_60`, `macd_signal_cross_strength`,
`ret1_x_volZ60`) are **single-fold-negative on 1h**, three of them clearly
so (≤ −0.005). Stage-1 admission was applied per-tf with a "≥ 2 of 3 OOS
folds positive in BOTH metrics" rule (`run_search.py`
`STAGE1_MIN_FOLDS_POSITIVE = 2`), but the **stage-2 stack is global** —
every admitted feature is added to every tf. Result: features that earned
admission because they helped on 6h/1d are dragging 1h DA down, which
compounds the contract / signal-strength problems in Findings 1–2.

## Finding 4 — Calibration helps trade_share but cannot create DA on 1h/2h
The task #592 ON column moves trade_share from 0.96 → 0.65 (good) but
calibrates the per-row prob distribution to a near-uniform shape: per the
task #587 calibration diagnostic the fitted `inv_T` saturates at 0.25 (the
fitter floor) for almost every 6h/1d slice — and the same pattern is visible
in the per-slice `mean inv_T` column of the #592 verdict for 1h/2h
(every value is 0.25–0.37). A calibrated margin distribution that flat
means the cal-tail-fit `delta` cuts an essentially uniform slice out of the
trades, so the **post-rule DA on the rows that DO trade is basically the
same as the raw DA**. The DA-lift floor (+0.02) sees an unchanged surface;
PnL improves only because we trade fewer rows and pay fewer round-trip fees.

## Why this hits 1h/2h harder than 6h/1d
1. **Horizon mismatch is non-zero only on 1h/2h/6h, and the largest in
   relative terms on 1h** (Finding 1) — every "correct" multi-bar call has
   only a 70 % chance of being a 1-bar winner.
2. **Cost-vs-payout** is 6.7× tighter on 1h than 1d (Finding 2) — the gate
   is a physically harder bar even after the contract is fixed.
3. **The admitted stack was tuned for the longer timeframes' edge surface**
   (Finding 3) — multiple admitted features actively regress DA on 1h.
4. **Calibration moves trade_share but not DA** (Finding 4) — the
   trade-rule surface is no longer the bottleneck.

## Concrete next-iteration suggestion

Two complementary moves; either alone is testable in one search run, both
together is the recommended package because they attack independent failure
modes.

### A. Fix the 1-bar / multi-bar gate-contract mismatch in `score_fold`
In `artifacts/ml-engine/scripts/feature_edge_search/run_search.py`:

* In `score_fold`, prefer `directional_label_forward_return` when the column
  is present in `test_df`; fall back to `forward_return` only for legacy
  snapshots that lack it. (The column is already populated for 1h/2h/6h —
  the snapshot column listing in Finding 1 confirms it.)
* In `_fit_neutral_band_delta` / `fit_predict_with_optional_calibration` no
  change is needed — they operate on probabilities, not returns.
* Update the verdict markdown to label the PnL column "post_fee_pnl_pct_total
  on `directional_label_forward_return`" so future readers can tell which
  contract a given report ran under.

Effect on 1h: mean payout per call rises 0.83 % → 1.93 %, break-even DA
falls 0.680 → 0.578, and ~30 % of "model-correct, 1-bar-wrong" rows stop
counting as PnL losses. None of this changes the **gate constants** — the
gate is still "DA-lift > 0.02 AND PnL > 0" — but the surface the gate is
measured on now matches the surface the booster is trained on.

### B. Build per-tf stage-2 stacks instead of a global stack
Replace the current "admit if any-tf passes stage-1, then stack everywhere"
flow with **per-tf admission lists** that survive into stage-2. Concretely
in `run_search.py`:

* Stage-1 already records per-tf fold positivity per candidate. Persist the
  per-tf admission set instead of the global union.
* `stage2_evaluation` uses the per-tf admitted list when fitting the
  augmented booster on that tf's data.
* Verdict reports per-tf gate pass independently.

This removes the six 1h-negative admitted features from the 1h booster's
input (Finding 3) and makes the 1h/2h DA surface the **best-of-tf** surface,
not the union-across-tfs. It costs zero new candidate engineering and one
bounded edit to the stage-2 path + the verdict renderer.

### Why both, not just A
A reduces the structural difficulty (the gate becomes feasible), but keeps
a 1h-hostile feature stack inside the booster. B picks the right features
for each tf, but if A is not also done the gate stays at break-even DA
0.680 on 1h (essentially unreachable for tree boosters on this feature
universe). Together: the booster is trained AND scored on the multi-bar
contract it was designed for, AND it only carries the features that
demonstrably lift on its own tf.

## What this does NOT recommend
* **Adding new candidate features** (e.g. order-flow imbalance,
  microstructure features). The diagnostic does not have evidence that
  those would lift; the existing `btc_lead_*` / `eth_lead_*` /
  `log_liquidations_self` columns are mostly NaN/zero in the snapshots,
  suggesting an upstream ingestion gap is the real blocker for cross-coin /
  liquidity features. Re-feeding those columns is itself a separate task.
* **Lowering the gate constants** (`STAGE2_DA_LIFT_FLOOR`, the trade_share
  band). Tasks #580/#587/#592 all explicitly held these constant; that
  contract should not be silently broken from a diagnostic.
* **Turning off calibration.** The flag-gated calibration in #587 is the
  right trade-share lever; it is not the DA lever and was never claimed to
  be.
* **Raising the multi-bar label thresholds** (`MULTI_BAR_LABEL_THRESHOLDS_PERCENT`).
  The current values (1h 0.40 %, 2h 0.50 %, 6h 0.80 %) already sit
  comfortably above the 0.30 % round-trip cost on the multi-bar return
  scale and below the matching outcome thresholds (1h 0.45 %, 2h 0.55 %,
  6h 0.85 %). Changing them is unrelated to the diagnostic findings.

## Reproduction
The numbers in this report come from a single-fold (75/25 time-ordered
split) ablation per candidate, pooled across coins, with the same LightGBM
hyperparameters and registry feature columns the production search runner
uses. The full single-fold-ablation tables and the cost-vs-payout statistics
were computed inline against the same dataset snapshots task #592 used.
There is no new committed script — this report is read-only diagnostic on
existing artifacts.
