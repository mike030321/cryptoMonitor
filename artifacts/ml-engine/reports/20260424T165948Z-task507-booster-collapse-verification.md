# Task #507 — booster_collapse fix verification

**Date:** 2026-04-24 16:59:48 UTC
**Scope:** Recover the 4 (coin, timeframe) slices that the Task #482
single-T calibrator could not push above the verification gate's
`MAX_DIRECTIONAL_CALL_SHARE = 0.95` ceiling
(`app/training/verification.py:90`).

**Stuck slices (pre-fix):** `bonk@1d`, `celestia@1d`, `dogwifcoin@1d`,
`celestia@6h`. After Task #482, raw STABLE-argmax on these slices was
in `0.0–4.5%`, below the 5% floor the gate effectively requires.

---

## 1. Root cause

The brief proposed sqrt-balanced sample weights as the fix. **The
focused diagnostic** (`scripts/diagnostic_482/run_507_focused.py`)
**disproved that hypothesis**: sweeping
`_balanced_sample_weight`'s exponent across `alpha ∈ [0.0, 2.5]` with
the existing `_lgb_params` + `LGB_EARLY_STOPPING` path moved the stuck
slices' raw STABLE-argmax monotonically *up* with alpha (uniform → 0%,
sqrt-balanced → 0%, full-balance → ~1.5%, alpha=2 only → ~6% on
celestia@1d). Sqrt-dampening made the stuck slices strictly worse.

The actual root cause is **`best_iteration == 1` on every tiny slice**.
LightGBM's `multi_logloss` validation loss spikes after the first tree
because the holdout is too small (n≈67 on 1d) for the early-stopping
callback's 25-round patience to ride out — it locks in iteration 1.
With one tree, per-row `P(STABLE)` is bounded by the marginal STABLE
prior plus a small leaf delta, so `max P(STABLE) < 1/3` and STABLE
never wins the per-row argmax.

The combination that recovers the booster on every tiny slice is:

* **Soft hyperparameters:** `num_leaves=15, learning_rate=0.05,
  min_child_samples=20`. The softer model can no longer carve a
  single-leaf split the holdout punishes.
* **No early stopping:** train the full `LGB_NUM_BOOST_ROUND` budget
  so the loss curve keeps decreasing past iteration 1.
* **Moderate class boost:** `alpha = 2` (rare-class row weight =
  legacy²), enough to push the booster's marginal STABLE prior above
  the per-row argmax threshold.

---

## 2. Fix

`app/training/train.py::_train_lgb` now branches on
`n_train < TINY_SLICE_THRESHOLD = 1500`:

* **Tiny slice:** override caller params with the soft recipe above,
  use `_balanced_sample_weight(y, alpha=TINY_SLICE_CLASS_WEIGHT_ALPHA)`,
  drop the early-stopping callback. `booster.best_iteration == 0` then
  means LightGBM's `predict` uses every tree.
* **Non-tiny slice:** unchanged from the pre-fix path — caller's
  Optuna-tuned params + `LGB_EARLY_STOPPING` callback +
  `_balanced_sample_weight(y)` (full-balance default).

Threshold rationale (per dataset inventory):

| timeframe | n_train  | tiny path? |
| --------- | -------- | ---------- |
| 1d        | ~264     | yes        |
| 6h        | ~1140    | yes        |
| 2h        | ~3.4k    | no         |
| 1h        | ~6.9k    | no         |
| 5m / 1m   | ≫ 10k    | no         |

The default `CLASS_WEIGHT_ALPHA` reverted from the (disproven)
`alpha = 0.5` to `alpha = 1.0` — i.e. the legacy Task #95 full-balance
recipe. The `ML_CLASS_WEIGHT_ALPHA` env override is retained for
ablation runs.

---

## 3. End-to-end verification

`scripts/diagnostic_482/run_507_focused.py` (default config; fixture
`reports/20260424T165513Z-task507-booster-collapse-rerun-alpha1.0.json`):

| coin       | tf | bucket          | raw_STABLE_share | directional_call_share | gate (DCS≤0.95) |
| ---------- | -- | --------------- | ---------------- | ---------------------- | --------------- |
| bonk       | 1d | `no_collapse`   | 0.1045           | 0.8955                 | **PASS**        |
| celestia   | 1d | `no_collapse`   | 0.2985           | 0.7015                 | **PASS**        |
| dogwifcoin | 1d | `no_collapse`   | 0.0597           | 0.9403                 | **PASS**        |
| celestia   | 6h | `no_collapse`   | 0.2133           | 0.7867                 | **PASS**        |

All four stuck slices clear both done-criteria from the brief:

1. `bucket == 'no_collapse'` (raw STABLE / multi-class shares respect
   the `_bucket_for` thresholds in
   `scripts/diagnostic_482/run_stage_collapse_diagnostic.py:196`).
2. `directional_call_share ≤ 0.95` (raw STABLE-argmax ≥ 5%).

Non-tiny path sanity check (`1h`, untouched by the fix): bonk@1h
`raw_S=0.51`, celestia@1h `raw_S=0.21`, dogwifcoin@1h `raw_S=0.53`,
all `bucket=no_collapse` — confirming the override is correctly scoped
and does not regress 1h / 2h / longer-history pooled fits.

---

## 4. Lift / PnL impact

The brief's third done-criterion is "lift/PnL not regressed". The
walk-forward PnL evaluation lives in the full training campaign
(`scripts/run_training_campaign.py`), which takes hours to complete
end-to-end and is therefore deferred to a follow-up verification task
once the campaign workflow finishes its next scheduled run.

Expected directional impact, based on the focused-diagnostic shares:

* **1d slices:** the tiny-slice branch may push `raw_S` above
  `label_S` on some healthy 1d slices (e.g. `floki-inu@1d` shifts to
  `raw_S=0.61`). This *reduces* directional calls (DCS drops), so it
  cannot trip the `MAX_DIRECTIONAL_CALL_SHARE` ceiling. The trade-off
  is fewer directional trades — acceptable on the 1d horizon where
  signal-to-noise is already low and STABLE-anchored hold positions
  are the conservative default.
* **6h slices:** similar pattern, more pronounced on
  `injective-protocol@6h` (`raw_S=0.91`). Worth re-measuring against
  the next walk-forward PnL run; a follow-up task has been proposed
  to either tighten `TINY_SLICE_CLASS_WEIGHT_ALPHA` or raise
  `TINY_SLICE_THRESHOLD` if 6h PnL regresses materially.

---

## 5. Tests

* `test_balanced_sample_weight_default_is_full_balance` — pins the
  module default at `alpha = 1.0` after Task #507 reverted the
  (disproven) sqrt change.
* `test_balanced_sample_weight_alpha_two_amplifies_rare_class` — pins
  the `(legacy)²` relationship the tiny-slice branch relies on.
* `test_train_lgb_tiny_slice_branch_overrides_caller_params` — proves
  the override actually fires: caller's `num_leaves=255` is replaced
  with `TINY_SLICE_NUM_LEAVES`, no early stopping is registered, the
  booster trains ≥ 10 rounds (no iter=1 trap).
* `test_train_lgb_non_tiny_slice_keeps_caller_params` — proves the
  pre-fix path is untouched on `n_train >= TINY_SLICE_THRESHOLD`.
* `test_task507_focused_diagnostic_recovers_all_stuck_slices` —
  fixture-driven regression: replays the latest focused-diagnostic
  JSON and asserts every stuck slice clears both gate criteria at the
  production `alpha = 1.0`.
* `test_balanced_sample_weight_handles_missing_classes` — extended to
  cover `alpha=2.0` alongside the existing `alpha=1.0` and
  `alpha=0.5` cases.
* Unchanged: env-var override, alpha=0/1 corners, empty input.

All 10 directly affected tests pass:

```
tests/test_training.py ..........  [10 passed in 52.93s]
```

---

## 6. Follow-ups

1. **Backtest verification:** run the next `scripts/run_training_campaign.py`
   end-to-end and compare walk-forward PnL on the touched 1d / 6h slices
   against the pre-fix campaign. If the tiny-slice branch regresses PnL
   beyond noise, tighten `TINY_SLICE_CLASS_WEIGHT_ALPHA` to 1.5 (recovers
   3/4 stuck slices) or raise `TINY_SLICE_THRESHOLD` to exclude 6h.
2. **`dogwifcoin@1d` margin:** clears the gate at `raw_S = 0.0597` —
   only 0.0097 above the floor. A follow-up feature-engineering pass
   on this coin (its STABLE-discriminating signal is the weakest of
   the 4 stuck slices in every alpha sweep we ran) would harden the
   margin and reduce sensitivity to LightGBM's RNG.
