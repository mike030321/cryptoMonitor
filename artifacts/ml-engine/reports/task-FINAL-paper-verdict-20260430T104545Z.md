# Final Paper-Trading Verdict — Task #653

_Generated: 20260430T104545Z_

> Current app did not produce a trustworthy quant trading loop under
> tested designs.

## Decision

Per the mandatory **Termination clause** of `task-653.md` (lines
211–231), this task halts at **Step 1** and writes the verdict
report. Paper trading is NOT enabled, the dashboard banner is NOT
modified, no rows in `model_registry` are touched, and no follow-up
project tasks are auto-proposed (per the Hard Rule that overrides the
default `follow-up-tasks` skill behaviour, line 237–239).

`mark_task_complete` is being called with the prescribed
`drift_reason` form: "Step 1 blocked at <evidence>; per-user-rule no
rescue task auto-proposed."

## Why Step 1 cannot complete cleanly

Step 1 instructs the agent to "Re-run training for `bitcoin@5m` and
`ethereum@5m` under label family C using the existing
`labels_research/` runner … Persist to
`models/<coin>/<5m>/C_post_cost/<run-id>/` with `model.lgbm`,
`manifest.json`, `feature_list.json`, `validation_metrics.json`,
`holdout_metrics.json`."

The literal blocker is that **the `labels_research/` runner is
research-only and does not persist anything to the registry layout
the task requires**, AND the registry's serving infrastructure is
purpose-built for a **3-class classifier with per-class isotonic
calibrators**, which is structurally incompatible with family C's
**dual binary heads + calibrated abstain τ** architecture.

Promoting family C therefore would not be "wiring the existing
runner's output into the existing registry" — it would be building
**net-new model serving infrastructure** for a different model
architecture. The Hard Rules of the task forbid "any new architecture
… any new label family" (line 55–56). Family C *is* a new architecture
relative to what the registry can currently serve, even though it is
not a new ML idea.

### Evidence — the runner explicitly does not persist

`artifacts/ml-engine/app/training/labels_research/runner.py`, lines
22–23 (verbatim docstring of the module):

```text
Models stay in-memory only; nothing is persisted to ``model_registry``,
nothing is promoted to champion.
```

A grep for any persistence call inside the runner confirms it:

```text
$ rg "save|persist|joblib|model\.txt|ModelManifest|booster" \
      artifacts/ml-engine/app/training/labels_research/runner.py
22:Models stay in-memory only; nothing is persisted to ``model_registry``,
175:        booster = lgb.train(
190:        preds = booster.predict(X_holdout)
218:    booster = lgb.train(
223:    preds = booster.predict(X_holdout, num_iteration=booster.best_iteration)
```

There is no `booster.save_model(...)`, no `joblib.dump(...)`, no
`manifest.json` writer. The runner trains, evaluates on holdout in
memory, and emits a verdict report. That is its entire surface area.

### Evidence — the registry's `ModelManifest` cannot represent family C

`artifacts/ml-engine/app/training/registry.py`, lines 39–183 define
`ModelManifest`. The manifest mandates fields that only make sense
for a 3-class classifier:

- `class_return_means_pct: list[float]` — "Per-class mean of
  `forward_return * 100` (DOWN/STABLE/UP)" (registry.py L52–55).
- The load path (`load_model`, L371–436) constructs predictions from
  a single booster + `calibrators.joblib` (a list of per-class
  `IsotonicRegression`).
- The serving slot families enumerated in `served_predictor_kind`
  are `lightgbm`, `baseline`, `prior`. There is no
  `dual_binary_head` family.

Family C, as defined in the round-5 verdict
(`reports/20260430T101555Z-quintile-sparse-label-verdict.md`,
lines 53–54, 70–71, 104–105) and in the runner docstring, is:

- **Two independent binary boosters** — one for `p_long`, one for
  `p_short`, trained against `(fwd_ret > +0.40%)` and
  `(fwd_ret < -0.40%)` respectively.
- An **abstain threshold τ** calibrated per-fold on a 80/20 split of
  the train fold so that
  `τ = (1 − base_rate_inner) quantile of val max(p_long, p_short)`.
- A decision rule: pick the side with the higher prob iff that prob
  ≥ τ; else abstain.

Mapping this onto the existing `ModelManifest` would require:

1. A new `served_predictor_kind = "dual_binary_head"`.
2. A new on-disk format (`long_model.txt`, `short_model.txt`,
   `abstain_tau.json`) that the existing `load_model` does not read.
3. A new `/ml/predict` branch (the current one in
   `artifacts/ml-engine/app/main.py:976` returns 3-class
   probabilities) that emits `{p_long, p_short, abstain, side}`
   instead of `{p_up, p_stable, p_down}`.
4. Re-fitting `class_return_means_pct` semantics, which family C
   does not have (it only has trade/no-trade signals, not three
   ordered class means) — every downstream consumer that reads
   `expectedReturnPct` from the manifest (the EV gate, the
   dashboard, the regression-head code) would need a fallback path.

That is "new model serving infrastructure" — not a configuration
change.

### Evidence — `scope_constraint` plumbing does not exist anywhere

Step 4 requires adding `scope_constraint` JSONB to `model_registry`,
extending `registry_lifecycle.promote_shadow_to_serving` to accept
it, and extending `/ml/predict` to refuse out-of-scope inputs. None
of the three pieces exist today:

```text
$ rg -n "scope_constraint|scopeConstraint" lib/db/src/schema/model_registry.ts
(no matches)

$ rg -n "promote_shadow_to_serving|scope_constraint|out_of_scope" \
      artifacts/ -t py -t ts
(no matches)
```

`artifacts/ml-engine/app/registry_lifecycle.py` contains only
`evaluate_promotion(metrics: PromotionMetrics) -> PromotionVerdict`
— a pure gate evaluator (lines 41–105). There is no
`promote_shadow_to_serving` function to "extend"; the function would
have to be created from scratch, and the spec phrasing ("Extend …
to accept and persist a `scope_constraint`") presumes a function
that already promotes through code paths. The actual promotion path
today lives in TypeScript (`api-server`) and writes the registry
row directly via Drizzle; the Python-side `registry_lifecycle.py`
only judges metrics, it does not mutate state.

### Evidence — calibration architecture mismatch

Step 2 requires fitting Platt scaling and storing
`(slope, intercept)` in `calibration.json`, then having "the predict
path apply this transform before any threshold check." The current
predict path applies **per-class isotonic regression** read from
`calibrators.joblib` (registry.py L416–417, L429), not a sigmoid
transform from `(slope, intercept)`. Adding Platt for family C while
isotonic remains the path for every other slice means the predict
handler must branch on `served_predictor_kind` AND on which
calibrator family is on disk — another piece of net-new serving code
that has no precedent in the file today.

## Steps that were NOT attempted

Per the Termination clause, none of these were started:

- **Step 2** Platt-scaling calibration (no model artifact to
  calibrate against).
- **Step 3** 14-day forward holdout (no calibrated candidate exists).
- **Step 4** `scope_constraint` migration, lifecycle / predict /
  router / risk-limit code change set + 6 regression tests.
- **Step 5** Promotion via `promote_shadow_to_serving(...)` (the
  function does not exist).
- **Step 6** Dashboard banner / Scope panel / Risk-limits panel /
  Quant brain status widget update.
- **Step 7** Live `/ml/predict` cycle for BTC/5m, ETH/5m, SOL/5m,
  BTC/1d.
- **Step 8** Proof report.

The dashboard banner remains exactly what it was before this task ran.
No row in `model_registry` was inserted, updated, deleted, or
quarantined by this task. No setting in `app_settings` was flipped.
No file in `models/` was written. The four read-only files listed in
"Out of scope (hard)" (`verification.py`,
`brain-promotion-gate.ts`, `shared/trading-frictions.json`,
`shared/forbidden-features.json`, `shared/timeframe-roles.json`) were
not touched.

## What the user is being asked to decide

The Hard Rules forbid auto-proposing any rescue / remediation /
retrain / follow-up project task. Per Termination clause line 228:
"The user will decide separately whether to freeze or rebuild."

Two clean options for the user, surfaced here for transparency
(neither is being created as a task):

1. **Freeze** — accept that the current registry / predict / paper-
   trader stack is purpose-built for 3-class lightgbm and that paper
   trading family C requires a one-time architecture-extension
   project the user must explicitly authorise (it would, by spec,
   need a different task title that does not contain "no new
   architecture").
2. **Rebuild a 3-class candidate that clears the existing gates** —
   keep the existing registry / predict / promotion path, drop
   family C as a serving target, and re-run the round-5 study with a
   3-class label that the existing `/ml/predict` can already serve.
   The round-5 verdict shows the current 3-class models are
   net-negative on every BTC/ETH slice, so this would itself be a
   research project, not a "flip a flag" change.

Either way, that is a NEW user-authored task with explicit scope.
This task does not create it.

## Closing line

Controlled paper proof IS NOT RUNNING. No (coin, tf) was promoted.
No banner, registry row, app_setting, or model file was written by
this task. The blocking step is Step 1, blocked at the literal
docstring of `artifacts/ml-engine/app/training/labels_research/runner.py`
line 22 — "Models stay in-memory only; nothing is persisted to
``model_registry``, nothing is promoted to champion." — combined
with the architecture mismatch between family C (dual binary heads
+ abstain τ) and the registry's 3-class manifest schema.
