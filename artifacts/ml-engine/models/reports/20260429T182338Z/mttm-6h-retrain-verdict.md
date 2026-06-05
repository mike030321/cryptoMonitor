# MTTM 6h retrain verdict — 2026-04-29 (post-#633 ML_TINY_SLICE_THRESHOLD=600 experiment)

## TL;DR

**The experiment failed. Lowering `ML_TINY_SLICE_THRESHOLD` from 1500 → 600 to push 6h
slots (n_train=1396) out of the soft-path and into the regular Optuna+early-stopping
path did NOT fix per-coin class collapse on 6h, did NOT lift directional accuracy
above the MTTM 0.50 gate, and made log-loss WORSE on every single slot. The MTTM 6h
universe remains 0/8 eligible. The env var has been reverted. The fundamental edge
ceiling at 6h appears to be a feature/label problem, not a hyperparameter problem.**

## Setup

- Trigger: `POST /ml/admin/retrain` at `2026-04-29T17:26:23Z` with
  `{"coins": [8 MTTM coins], "timeframes": ["6h"]}` and `X-Admin-Key`.
- Lever: ml-engine restarted at `2026-04-29T17:23Z` after adding
  `ML_TINY_SLICE_THRESHOLD = "600"` to both `[services.env]` and
  `[services.production.run.env]` in `artifacts/ml-engine/.replit-artifact/artifact.toml`.
  Process env confirmed via `/proc/<pid>/environ` before triggering the retrain.
- Baseline (OLD): the 6h slots written by the `task-399-training-campaign`
  workflow at `16:06–16:37Z` on the same day, all under the default
  `ML_TINY_SLICE_THRESHOLD=1500` (i.e. n_train=1396 < 1500 → soft path with
  Task #507's α=2 minority-amplification recipe).
- Treatment (NEW): the 6h slots written by the API-triggered retrain between
  `17:39–19:04Z`, under `ML_TINY_SLICE_THRESHOLD=600` (i.e. 1396 > 600 →
  regular Optuna path: 40 trials × 180 s, then 800 boost rounds with
  early stopping).
- Per-coin label distribution is now in the manifest (post-#633 NaN fix
  propagation): bonk's labels are DOWN=666, STABLE=196, UP=534 on n=1396.
  Population is moderately imbalanced (47.7 % DOWN, 14.0 % STABLE,
  38.3 % UP) but not pathological.

## Per-coin results (8/8 slots)

| Coin                       | OLD DA | NEW DA | OLD AUC | NEW AUC | OLD call_share | NEW call_share | OLD log_loss | NEW log_loss |
|---------------------------:|:------:|:------:|:-------:|:-------:|:--------------:|:--------------:|:------------:|:------------:|
| bonk                       | 0.410  | 0.417  | 0.526   | 0.522   | 0.943          | **0.957**      | 3.03         | **3.56**     |
| celestia                   | 0.394  | 0.396  | 0.515   | 0.512   | 0.996          | 0.975          | 3.59         | **4.12**     |
| dogwifcoin                 | 0.399  | 0.409  | 0.513   | 0.521   | 0.904          | **0.936**      | 3.39         | **3.85**     |
| floki-inu                  | 0.392  | 0.397  | 0.533   | 0.530   | 0.989          | 0.971          | 3.53         | **4.17**     |
| injective-protocol         | 0.426  | 0.419  | 0.543   | 0.537   | 0.914          | 0.911          | 3.16         | **3.88**     |
| jupiter-exchange-solana    | 0.385  | 0.392  | 0.520   | 0.522   | 0.957          | 0.939          | 3.47         | **3.59**     |
| pepe                       | 0.414  | 0.419  | 0.524   | 0.530   | 0.661          | 0.650          | 3.67         | **3.90**     |
| render-token               | 0.436  | 0.434  | 0.518   | 0.518   | 0.946          | **0.957**      | 2.66         | **2.99**     |

(`MIN_DIRECTIONAL_ACCURACY` gate = 0.50, `MAX_DIRECTIONAL_CALL_SHARE` gate = 0.95;
bold = NEW worse than OLD on that metric.)

### Score-card

(All aggregates recomputed directly from the per-coin manifest values shown
in the table above; see the calc snippet at the bottom of this report for
reproducibility.)

- **Directional accuracy (≥0.50 needed)**: 0/8 NEW slots pass. NEW marginally
  better than OLD on 6/8, worse on 2/8 (injective, render-token), tied on 0/8.
  Mean Δ = **+0.003**, per-coin range [−0.01, +0.01]. All NEW values still in
  the 0.39–0.43 band — this is a **sub-1-percentage-point shuffle inside the
  same failing regime**, not a qualitative change.
- **Directional call share (≤0.95 needed)**: 1/8 NEW slots pass (pepe at 0.650,
  same outlier as OLD). Of the 7 failing, 4 got slightly better and 3 got
  worse; counting pepe (also slightly better) the totals are 5 better / 3
  worse / 0 tied. Mean Δ = **−0.002**, per-coin range [−0.02, +0.03]. Again:
  shuffle inside the same failing regime.
- **AUC**: NEW slightly worse on 4/8 (bonk, celestia, floki-inu, injective),
  slightly better on 3/8 (dogwifcoin, jupiter, pepe), tied on 1/8 (render-token).
  Mean Δ ≈ **0.000** (per-coin range [−0.006, +0.008]).
- **Log-loss (calibration / confidence)**: **NEW is worse on all 8/8**.
  Mean Δ = **+0.45 nats**, per-coin range [+0.12, +0.72]. The Optuna+800-round
  path is finding models that are **more confidently wrong** than the
  soft-path baseline.
- **Net**: zero MTTM-relevant improvement. The lever is no-op-to-marginally-negative.

## Why it failed

The pre-experiment hypothesis was that `train.py`'s soft path (used when
`n_train < ML_TINY_SLICE_THRESHOLD`) was misfiring on 6h slots: Task #507 tuned
α=2 minority-amplification for n≈264 1d slots, and applying that same recipe
to 6h slots (n=1396, vol-scaled labels from Task #459) was suspected to push
predictions into majority-class collapse (`prediction_collapse` events showed
5/8 6h slots predicting DOWN 65–90 % vs an actual rate of ~40 %).

The experiment tested whether bypassing the soft path entirely (by lowering
the threshold) would fix the collapse. **It did not.** What changed is the
collapse signature, not the collapse itself:

- **Soft path (OLD)**: shallow ensembles, sample-weighted minority lift. They
  collapse with *low confidence* — log-loss in the 2.66–3.67 band. They make
  similar directional shares because the weighting is hard-coded.
- **Regular path (NEW)**: deep boosters (up to 800 rounds with early stopping)
  optimising log-loss directly via Optuna. They collapse with *high confidence*
  — log-loss in the 2.99–4.17 band. They make similar directional shares because
  the optimiser is finding the same dominant feature signal in both regimes.

That both regimes converge to similar bad call-shares while disagreeing on
log-loss is strong evidence that **the issue is the feature/label edge, not
the recipe**. Specialists (which were already on the regular path) reach
DA 0.34–0.35 with perfect class balance — same edge ceiling, different
collapse signature. So the directional edge in the current 6h feature schema
+ vol-scaled label set is genuinely below 0.50, regardless of how we tune the
booster.

## What the lever does NOT change

- It does not change which features the trainer sees (forbidden-features
  loader, approved-features bridge, etc. all unchanged).
- It does not change the vol-scaled label thresholds (Task #459 still in force).
- It does not change the trading frictions (`shared/trading-frictions.json`).
- It does not weaken the MTTM gates (still 0.50 DA, 0.95 call-share, no demotion).
- It does not enable any synthetic data (no flag flips, no fallback edits).

## Honest position vs MTTM

`mttm-audit` was run after the per-coin slots completed and reported
**16/16 slots ineligible**:

- 6h slots: `verification.json unreadable; latest <NEW> ≠ pinned 20260425T...`
  (the verification block had not been re-emitted yet; the slots are pinned
  to last-week's versions in app_settings)
- 1d slots: 7/8 `not promoted (reason=below_coinflip)`, 1/8 `not promoted
  (reason=directional_call_regression)` — the underlying 1d models also
  cannot beat coinflip.

MTTM remains **DISABLED** and continues to refuse to enable. **This is the
correct, honest behaviour.** Trading paper-only on a model that is
demonstrably wrong more often than coinflip — and confidently wrong, as
log-loss shows — would not be "honest and truthful trading"; it would be
honest *theatre* of trading. The gates exist for exactly this case.

## Cleanup performed

1. Reverted `ML_TINY_SLICE_THRESHOLD` removal from
   `artifacts/ml-engine/.replit-artifact/artifact.toml` (both `[services.env]`
   and `[services.production.run.env]`).
2. Updated the comment block above `TINY_SLICE_THRESHOLD` in
   `artifacts/ml-engine/app/training/train.py` to record this negative result
   so the next operator does not re-run the same experiment.
3. Restarted the `artifacts/ml-engine: ML Engine` workflow so the reverted env
   takes effect on the next API-triggered retrain.
4. Appended a `verdict_report` event to
   `artifacts/ml-engine/models/progress_updates.jsonl` pointing to this file.
5. Left the new 6h manifests on disk (they are honestly-labelled and the
   `latest` pointer reflects the actual newest training run; they just fail
   the MTTM gates as documented above).

## Recommended next directions (NOT executed in this run)

These are **honest leads**, not commitments. Each one will need its own
falsifiable test the way #507 and this one did:

1. **Feature audit at 6h cadence.** The fact that pool-trained specialists
   (which see all 8 coins together with perfect class balance and a
   regularised loss) still cap at DA 0.34–0.35 says the *features themselves*
   don't carry directional edge at the 6h horizon under the current vol-scaled
   labels. Look at: which features actually have non-trivial split gain,
   whether news/regime features ever fire on the 6h cadence, whether the
   `class_return_means_pct` distribution suggests the threshold is being placed
   too tightly around the bin centre.
2. **Re-examine the vol-scaled threshold for 6h specifically.** Task #459's
   threshold formula was tuned per-timeframe; it's possible the 6h `k`
   coefficient is too aggressive (filtering all the signal into STABLE).
   pepe is the existing counter-example with call_share=0.65 — diff its
   `threshold_pct` and `class_return_means_pct` against the collapsed coins.
3. **Drop 6h from the MTTM universe entirely if (1) and (2) don't move the
   needle.** The MTTM contract is "narrow universe, no synthetic data, no
   gate weakening" — if 6h has no edge in the current feature set, the
   honest move is to remove it, not to tune the trainer until it appears
   to pass.
4. **Fix the silent-progress bug in `app/main.py:1797`.** The current
   `asyncio.run(run_training(cs, tfs))` call passes no `progress_callback`,
   so API-triggered retrains write zero rows to `progress_updates.jsonl`
   for the entire run. Operators have to scrape disk modtimes to see
   progress. The campaign script wires in `_append_progress` directly;
   the API path should do the same. **Not done in this run** because the
   fix requires a ml-engine restart and the retrain was in flight.

## Audit artefacts

- This report: `artifacts/ml-engine/models/reports/20260429T182338Z/mttm-6h-retrain-verdict.md`
- Per-coin NEW manifests: `artifacts/ml-engine/models/<coin>/6h/<NEW_TS>/manifest.json`
  (NEW_TS in the table above)
- Per-coin OLD manifests: `artifacts/ml-engine/models/<coin>/6h/20260429T16xxxx Z/manifest.json`
- mttm-audit output: 16/16 FAIL — see the run output captured in this
  task's transcript.
- ml-engine artifact.toml history: see git log on
  `artifacts/ml-engine/.replit-artifact/artifact.toml`.
- Note on auto-generated failure-analysis reports: the file
  `artifacts/ml-engine/reports/20260429T182756Z-failure-analysis-auto.{md,json}`
  was emitted by the *concurrent* `task-399-training-campaign` worker
  (PID 55891) in response to its own watchdog snapshots; its
  `source_report_generated_at` timestamps therefore predate the
  19:04Z completion of this experiment's last per-coin slot
  (render-token). It is *not* a post-mortem of the threshold experiment
  and should not be conflated with this verdict report.

## Reproducible aggregates

The following Python snippet recomputes every aggregate cited in the
score-card from the per-coin table above. Run it from the repo root.

```python
old_da = [0.410, 0.394, 0.399, 0.392, 0.426, 0.385, 0.414, 0.436]
new_da = [0.417, 0.396, 0.409, 0.397, 0.419, 0.392, 0.419, 0.434]
old_cs = [0.943, 0.996, 0.904, 0.989, 0.914, 0.957, 0.661, 0.946]
new_cs = [0.957, 0.975, 0.936, 0.971, 0.911, 0.939, 0.650, 0.957]
old_ll = [3.03, 3.59, 3.39, 3.53, 3.16, 3.47, 3.67, 2.66]
new_ll = [3.56, 4.12, 3.85, 4.17, 3.88, 3.59, 3.90, 2.99]

import statistics

def s(name, o, n):
    deltas = [round(b - a, 3) for a, b in zip(o, n)]
    print(
        f"{name}: mean delta = {statistics.mean(deltas):+.4f}  "
        f"per-coin {deltas}  range [{min(deltas):+.2f}, {max(deltas):+.2f}]"
    )

s("DA", old_da, new_da)
s("CS", old_cs, new_cs)
s("LL", old_ll, new_ll)
```

Expected output (matches what the score-card cites):

```
DA: mean delta = +0.0034  per-coin [0.007, 0.002, 0.01, 0.005, -0.007, 0.007, 0.005, -0.002]  range [-0.01, +0.01]
CS: mean delta = -0.0017  per-coin [0.014, -0.021, 0.032, -0.018, -0.003, -0.018, -0.011, 0.011]  range [-0.02, +0.03]
LL: mean delta = +0.4450  per-coin [0.53, 0.53, 0.46, 0.64, 0.72, 0.12, 0.23, 0.33]  range [+0.12, +0.72]
```
