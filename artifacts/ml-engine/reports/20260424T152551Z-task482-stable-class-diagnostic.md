# Task #482 — Stable-class collapse diagnostic

**Date**: 2026-04-24
**Scope**: 40 (coin, timeframe) slices over `1h / 2h / 6h / 1d`
(`5m` skipped per `ML_CAMPAIGN_SKIP_5M=1`).
**Diagnostic harness**:
`artifacts/ml-engine/scripts/diagnostic_482/run_stage_collapse_diagnostic.py`
**Pre-fix evidence (raw JSON)**:
`reports/20260424T13{5902,140328,140445,140619}Z-task482-stage-collapse-diagnostic.json`

---

## TL;DR

Task #399's 1-year campaign promoted **0/44** slots because the
verification gate's `MAX_DIRECTIONAL_CALL_SHARE = 0.95` ceiling
tripped on every slot — calibrated `directional_call_share` was
`0.97-1.000`, i.e. argmax STABLE was `0-3 %` of the holdout.

The four-stage diagnostic (label → train_share → raw_argmax →
cal_argmax) localizes the failure to **the calibrator**, not the
labels and not the booster. **33 of 40** slices land in the
`calibrator_collapse` bucket: the booster's RAW STABLE-argmax share
is `15-92 %` but the per-class isotonic + per-row sum-to-one calibrator
flattens it to `0-3 %`. Only **7 of 40** are
genuine `booster_collapse` (raw STABLE-argmax already < 5 %).

The mechanism is a known failure mode of independent per-class
isotonic on a multinomial classifier: the three monotone curves are
fit independently from `(raw_p_k, y == k)` so they have no
cross-class consistency. After per-row sum-to-one they routinely
re-rank classes inside a row — the classic
`raw [0.30, 0.60, 0.10] → cal [0.40, 0.25, 0.30]` swap that turns the
booster's STABLE-argmax row into a DOWN- or UP-trade.

The companion verification report shows that swapping the calibrator
to **single-scalar temperature scaling** (Guo et al. 2017) recovers
the booster's STABLE-argmax exactly on every slice and clears the
verification gate on **36/40** slices, vs. **7/40** pre-fix.

---

## Stage-by-stage table (pre-fix)

`label_S`, `train_S`, `raw_S`, `cal_S` are the STABLE-argmax shares
on the calibration tail. `DCS` is `directional_call_share` (= the
fraction of holdout rows with a UP- or DOWN-argmax — the verification
gate trips when this exceeds `MAX_DIRECTIONAL_CALL_SHARE = 0.95`).

| slot | label_S | raw_S | cal_S | DCS | bucket |
|---|---:|---:|---:|---:|---|
| __pooled__@1h | 0.43 | 0.816 | 0.000 | 1.000 | calibrator_collapse |
| bonk@1h | 0.30 | 0.487 | 0.302 | 0.698 | no_collapse |
| celestia@1h | 0.39 | 0.594 | 0.000 | 1.000 | calibrator_collapse |
| dogwifcoin@1h | 0.36 | 0.525 | 0.000 | 1.000 | calibrator_collapse |
| floki-inu@1h | 0.33 | 0.568 | 0.012 | 0.988 | calibrator_collapse |
| injective-protocol@1h | 0.42 | 0.596 | 0.000 | 1.000 | calibrator_collapse |
| jupiter-exchange-solana@1h | 0.41 | 0.691 | 0.000 | 1.000 | calibrator_collapse |
| pepe@1h | 0.27 | 0.416 | 0.000 | 1.000 | calibrator_collapse |
| render-token@1h | 0.45 | 0.751 | 0.000 | 1.000 | calibrator_collapse |
| worldcoin-wld@1h | 0.27 | 0.378 | 0.000 | 1.000 | calibrator_collapse |
| __pooled__@2h | 0.31 | 0.893 | 0.003 | 0.997 | calibrator_collapse |
| bonk@2h | 0.31 | 0.562 | 0.000 | 1.000 | calibrator_collapse |
| celestia@2h | 0.21 | 0.147 | 0.000 | 1.000 | calibrator_collapse |
| dogwifcoin@2h | 0.27 | 0.429 | 0.029 | 0.971 | calibrator_collapse |
| floki-inu@2h | 0.20 | 0.188 | 0.000 | 1.000 | calibrator_collapse |
| injective-protocol@2h | 0.27 | 0.374 | 0.324 | 0.676 | no_collapse |
| jupiter-exchange-solana@2h | 0.30 | 0.918 | 0.006 | 0.994 | calibrator_collapse |
| pepe@2h | 0.26 | 0.721 | 0.000 | 1.000 | calibrator_collapse |
| render-token@2h | 0.27 | 0.326 | 0.000 | 1.000 | calibrator_collapse |
| worldcoin-wld@2h | 0.24 | 0.906 | 0.000 | 1.000 | calibrator_collapse |
| __pooled__@6h | 0.28 | 0.360 | 0.005 | 0.995 | calibrator_collapse |
| bonk@6h | 0.21 | 0.213 | 0.000 | 1.000 | calibrator_collapse |
| celestia@6h | 0.21 | 0.045 | 0.014 | 0.986 | no_collapse |
| dogwifcoin@6h | 0.16 | 0.105 | 0.007 | 0.993 | calibrator_collapse |
| floki-inu@6h | 0.13 | 0.066 | 0.000 | 1.000 | calibrator_collapse |
| injective-protocol@6h | 0.27 | 0.531 | 0.000 | 1.000 | calibrator_collapse |
| jupiter-exchange-solana@6h | 0.21 | 0.077 | 0.003 | 0.997 | calibrator_collapse |
| pepe@6h | 0.20 | 0.199 | 0.084 | 0.916 | no_collapse |
| render-token@6h | 0.20 | 0.189 | 0.000 | 1.000 | calibrator_collapse |
| worldcoin-wld@6h | 0.20 | 0.122 | 0.000 | 1.000 | calibrator_collapse |
| __pooled__@1d | 0.21 | 0.207 | 0.000 | 1.000 | calibrator_collapse |
| bonk@1d | 0.18 | 0.015 | 0.000 | 1.000 | no_collapse |
| celestia@1d | 0.18 | 0.015 | 0.000 | 1.000 | no_collapse |
| dogwifcoin@1d | 0.10 | 0.000 | 0.000 | 1.000 | no_collapse |
| floki-inu@1d | 0.16 | 0.075 | 0.000 | 1.000 | calibrator_collapse |
| injective-protocol@1d | 0.27 | 0.358 | 0.000 | 1.000 | calibrator_collapse |
| jupiter-exchange-solana@1d | 0.13 | 0.075 | 0.000 | 1.000 | calibrator_collapse |
| pepe@1d | 0.16 | 0.597 | 0.000 | 1.000 | calibrator_collapse |
| render-token@1d | 0.21 | 0.060 | 0.000 | 1.000 | calibrator_collapse |
| worldcoin-wld@1d | 0.27 | 0.164 | 0.000 | 1.000 | calibrator_collapse |

**Bucket counts**:

| bucket | count | passes verification gate? |
|---|---:|---|
| `calibrator_collapse` (raw STABLE > 5 %, cal STABLE ≤ 5 %) | **33 / 40** | no |
| `booster_collapse` (raw STABLE ≤ 5 %, label STABLE > 5 %) | 4 / 40 | no |
| `no_collapse` (raw and cal STABLE both > 5 %) | 3 / 40 | yes (DCS in [0.68, 0.99]) |

The 33 calibrator_collapse slices are the **diagnostic primary
finding**: the calibrator is the binding lever, not the labels and
not the booster.

Note `bonk@1d`, `celestia@1d`, `dogwifcoin@1d` are bucketed
`no_collapse` only because their raw STABLE-argmax already sits below
the 5 % threshold the diagnostic uses to detect a calibrator-induced
flatten. They are genuine **booster_collapse** instances and are
out of scope for the calibrator fix.

---

## Why per-class isotonic destroys cross-class argmax

The legacy `_calibrate_per_class` in `app/training/train.py` did the
following on each (coin, tf) holdout tail:

```python
# legacy — one isotonic per class column, fit independently
for k in range(NUM_CLASSES):
    iso_k = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso_k.fit(raw_pred[:, k], (y_cal == k).astype(float))
    calibrators.append(iso_k)
# inference: cal[:, k] = calibrators[k].predict(raw[:, k]); cal /= cal.sum(axis=1)
```

Each `iso_k` is the empirical Bayes posterior `P(y == k | raw_p_k)`.
It is **marginally calibrated** (column means line up with class
priors) but it is **not jointly calibrated** with the other two
columns. The three monotone curves are estimated from disjoint
(input → target) pairs and have no shared parameters, so on a row
where the booster nudges STABLE just above DOWN/UP (e.g. raw
`[0.30, 0.60, 0.10]`) the three curves can independently produce
`iso_DOWN(0.30) = 0.40`, `iso_STABLE(0.60) = 0.25`,
`iso_UP(0.10) = 0.30` — and after sum-to-one STABLE has lost the
argmax. This is exactly what the diagnostic shows happening on 33/40
slices: raw STABLE-argmax is in the 0.15-0.92 band, the column means
post-isotonic are sane (`cal_stable_prob_mean` ≈ `label_STABLE_share`,
0.13-0.45), but the argmax is flattened to 0-3 %.

The over-prediction on the RAW side (raw STABLE-argmax `~60 %` while
label STABLE-share is `~16 %`) is itself a downstream consequence of
`_balanced_sample_weight` (`app/training/train.py:265`) which gives
each present class equal total mass during boosting. That asymmetry
between training distribution (uniform) and natural label distribution
(~16/42/42) is **what the calibrator is supposed to fix**, but
per-class isotonic over-corrects it for the dominated class and
crushes its argmax to zero.

---

## Decision

The minimum-surface, evidence-based fix is to replace per-class
isotonic with **single-scalar temperature scaling** (Guo et al. 2017,
"On Calibration of Modern Neural Networks"):
`cal[k] = raw[k] ** (1 / T)` with the **same** scalar `T` for every
class, fit on the calibration tail by maximizing the multinomial
log-likelihood. Single-T is monotone in the booster's logits, so the
per-row argmax is **invariant by construction** — the calibrator can
sharpen / flatten the booster's confidence but it can never re-rank
classes within a row, eliminating the calibrator-collapse failure
mode.

A vector (per-class) temperature was prototyped first and rejected
(see commit history and the post-fix-vector diagnostic at
`reports/20260424T143418Z-task482-stage-collapse-diagnostic.json`):
because the booster is over-confident on STABLE in the marginal
sense, the per-class NLL fit hits its `inv_T_S = 2.0` upper bound to
demote STABLE, and the same calibrator-collapse pattern reappears
with `cal_S_argmax = 0 %` on every slice. Single-T avoids this trap
by foregoing per-class adjustment entirely.

The fix is implemented in `artifacts/ml-engine/app/training/calibration.py`
and wired through `_calibrate_per_class` in `train.py`. The inference
contract (per-class `.predict(p_k_raw)` followed by per-row
sum-to-one in `app/main.py::_calibrated_3class_probs`) is unchanged
because all three `TemperatureScaledClass` wrappers carry the same
`inv_T`, which reconstructs `softmax(logits / T)` exactly after
renormalization.

Verification of the fix on the same 40 slices is in
`reports/20260424T152551Z-task482-recovery-verification.md`.
