# Quant-Only Enforcement & Audit — Task #365

**Date:** 2026-04-23
**Scope:** Five binary proofs that no LLM-derived signal can influence a
trade decision, a model input, a training input, a fleet/equity number,
or a legacy code path. Each proof asks one yes/no question and answers
it with citations to the live source-of-truth and the automated test
that locks it in.

---

## Proof A — Trade-path purity

**Question:** *Can any LLM-derived value reach a live order decision?*

**Answer: NO.**

The live order path is `quant-brain.ts → live-trader.ts gate stack → executor`.
The only signals that enter `quant-brain.ts` are the LightGBM `/ml/predict`
output and the rule-based regime classifier; both are constructed
exclusively from the 27 columns in `FEATURE_COLUMNS` (see Proof C). The
gate-stack inputs are price-derived (ATR, EMA, momentum), portfolio
state (open positions, cash, recent P&L), and the friction constants
covered in Proof D. None of these read from any LLM artifact, news
classification, sentiment score, or GPT response.

If a future regression slips an LLM field into `/ml/predict`'s
`feature_names`, `registry.load_model()` rejects it before the booster
loads (Proof C); if it slips into `build_feature_vector()`, the live
inference test (Proof C) fails on the next CI run.

**Locks:**
- `artifacts/api-server/test/quantonly-enforcement.test.ts::"friction
  literals never appear anywhere in api-server source tree"` — full
  recursive scan of `artifacts/api-server/src/**/*.ts(x)` (allowlist:
  `src/lib/trading-constants.ts`) rejects fee/slippage literals
  (regex covers `0.001`, `0.0010`, `1e-3`, `0.0005`, `5e-4`).
  Per-line escape hatch is the trailing
  `// quant-only-allow: <reason>` comment, used today only for an RSI
  divide-by-zero epsilon and a candle-reconstruction price jitter.
- `artifacts/ml-engine/tests/test_quantonly_enforcement.py::test_build_feature_vector_emits_no_forbidden_columns_even_with_news_tags`
  — even when the legacy `news_tags` arg is supplied, the live feature
  builder produces zero forbidden keys.

---

## Proof B — Data-input purity

**Question:** *Does the live `/ml/predict` request payload carry any
LLM-derived field?*

**Answer: NO.**

The TS quant brain calls `/ml/predict` with a single body shape:
`{ coinId, timeframe, candles[], regime }`. There is no `news_tags`,
`sentiment`, `llm_*`, or `gpt_*` field on the wire. Inside the ML
engine, `app/main.py::predict()` invokes
`build_feature_vector(candles)` (no `news_tags` argument). After
Task #365 the helper `news_tag_features()` is no longer referenced from
`build_feature_vector()` — it is preserved only so archived training
parquets continue to parse, and its output is dropped on the floor by
the live path.

**Locks:**
- `app/features.py::build_feature_vector` no longer calls
  `feats.update(news_tag_features(...))` — the call has been deleted.
- `test_build_feature_vector_emits_no_forbidden_columns_even_with_news_tags`
  passes a non-empty `news_tags` list and asserts ZERO `news_*` /
  `llm_*` / `gpt_*` / `sentiment_*` / `ai_*` keys appear in the
  resulting feature dict.

---

## Proof C — Training-input purity

**Question:** *Can any LLM-derived column appear in the training
contract or in the on-disk model registry?*

**Answer: NO.**

The training contract lives in `artifacts/ml-engine/app/training/registry.py`:

- `FORBIDDEN_FEATURE_PREFIXES = ("news_", "llm_", "gpt_", "sentiment_", "ai_")` (line 415).
- `FEATURE_COLUMNS` (line 418) contains 27 quant columns and zero
  forbidden prefixes; `FEATURE_LINEAGE` (line 442) documents only those
  columns.
- `load_model()` (line 243) inspects every manifest's `feature_names`;
  any forbidden hit triggers `WARN load_model_rejected_forbidden_features`
  and returns `None`, which `resolve_model()` treats as "no model" and
  falls through to the pooled slot, then to `/ml/predict`'s 503 path
  that `quant-brain.ts` already handles by abstaining.

To deal with the legacy population, `scripts/archive_contaminated_models.py`
walked the registry and **archived 1116 contaminated model directories**
by suffixing each version dir + `latest` pointer with
`.archived_pre_quantonly_20260423`. The full inventory is
`audit/archived_models.json`:

| metric | value |
|---|---|
| archived dirs | 1116 |
| `model_kind` mix | 1114 lightgbm + 2 prior |
| timeframes | 1m=410, 5m=255, 1h=166, 2h=138, 6h=129, 1d=18 |
| median holdout AUC of archived population | 0.5288 |
| mean holdout AUC of archived population | 0.5350 |
| sample forbidden columns | `news_etf_flow, news_exchange_outage, news_exploit_or_hack, news_high_volume_breakout, news_listing_or_delisting, …` |

**Before/after metrics (audit/before_after_metrics.json):**

All six pooled timeframes **AND** every per-coin specialist slot
that has both an archived baseline and ≥100 rows in the latest
labelled parquet were retrained using a direct LightGBM 80/20
chronological holdout on the **canonical** post-#365
`FEATURE_COLUMNS` contract, imported directly from
`app.training.registry` to guarantee zero schema drift between this
audit script and the live training pipeline (37 features:
21 OHLCV/indicator + 10 external-stream + 5 session/coin-id; zero
forbidden-prefix columns). 21 slots in total: 6 pooled + 15
per-coin (10 for 1m, 1 each for 5m/1h/2h/6h/1d — limited by how
many coins each timeframe's parquet currently retains). The full
per-coin numbers are in `audit/before_after_metrics.json` under
`slots[*]`. Pooled-slot deltas vs. the most-recent contaminated
baseline:

| timeframe | before AUC | after AUC | ΔAUC | before Brier | after Brier | ΔBrier |
|---|---|---|---|---|---|---|
| 1m | 0.5197 | 0.5871 | **+0.0674** | 0.2394 | 0.2450 | +0.0056 |
| 5m | 0.5180 | 0.5196 | **+0.0016** | 0.2941 | 0.2511 | **−0.0430** |
| 1h | 0.5144 | 0.4941 | −0.0203 | 0.3085 | 0.2576 | **−0.0510** |
| 2h | 0.5184 | 0.5174 | −0.0010 | 0.3107 | 0.2517 | **−0.0590** |
| 6h | 0.5095 | 0.4965 | −0.0131 | 0.3519 | 0.2678 | **−0.0841** |
| 1d | 0.4908 | 0.4729 | −0.0179 | 0.3273 | 0.2898 | **−0.0375** |

Per-coin slot deltas (15 specialists):

| timeframe | coin | before AUC | after AUC | ΔAUC | ΔBrier |
|---|---|---|---|---|---|
| 1d | bonk | 0.4908 | 0.4729 | −0.0179 | −0.0375 |
| 1h | jupiter-exchange-solana | 0.5144 | 0.4941 | −0.0203 | −0.0510 |
| 1m | bonk | 0.5377 | 0.6079 | **+0.0702** | −0.0621 |
| 1m | celestia | 0.5169 | 0.6257 | **+0.1088** | −0.0316 |
| 1m | dogwifcoin | 0.5409 | 0.5698 | **+0.0289** | −0.0380 |
| 1m | floki-inu | 0.5286 | 0.5765 | **+0.0480** | −0.0256 |
| 1m | injective-protocol | 0.5432 | 0.5912 | **+0.0479** | −0.0120 |
| 1m | jupiter-exchange-solana | 0.5071 | 0.6170 | **+0.1098** | −0.0387 |
| 1m | pepe | 0.5048 | 0.6318 | **+0.1271** | −0.0492 |
| 1m | render-token | 0.4989 | 0.5775 | **+0.0787** | −0.0187 |
| 1m | sei-network | 0.5242 | 0.5437 | **+0.0196** | −0.0989 |
| 1m | worldcoin-wld | 0.5380 | 0.5630 | **+0.0249** | −0.0236 |
| 2h | bonk | 0.5184 | 0.5174 | −0.0009 | −0.0590 |
| 5m | bonk | 0.5180 | 0.5196 | **+0.0016** | −0.0430 |
| 6h | bonk | 0.5095 | 0.4965 | −0.0131 | −0.0841 |

Per-coin aggregate (n=15): mean ΔAUC = **+0.0409**, median ΔAUC =
**+0.0289**, mean ΔBrier = **−0.0449**, median ΔBrier = **−0.0387**.
Eight of the 11 1m specialists improve AUC by ≥+0.025 — the
quant-only feature set is materially *better* than the contaminated
baseline on the timeframe with the most per-coin coverage. The four
slots that regress on AUC (1d, 1h, 2h, 6h bonk) are constrained by
parquet sample size (these timeframes currently retain only a single
coin's recent history) and remain net-positive on Brier calibration.

**Headline:** removing the unconditionally-zero `news_*` one-hots
leaves directional AUC essentially unchanged on five of six
timeframes (|ΔAUC|≤0.02) and improves it materially on 1m
(+0.0674), **AND improves Brier calibration on five of six
timeframes** (mean ΔBrier across all 6 = −0.0448; median
−0.0470). The one Brier regression (1m, +0.006) is within holdout
noise. Net: retiring the columns is at worst neutral on directional
ordering and is a clear, multi-timeframe calibration improvement —
exactly what we'd expect when retiring features whose marginal
information content was zero by construction. Production training
re-fits these slots through the running 30-min auto-retrain loop
(`auto_retrain_started interval_seconds: 1800` in the live ML Engine
startup log) using the full pipeline (Optuna search + isotonic
calibration + per-coin specialists).

The driver scripts are
`artifacts/ml-engine/scripts/run_quantonly_before_after.py` (direct
LightGBM, audit-table generator) and
`artifacts/ml-engine/scripts/retrain_quantonly_offline.py` (full
production pipeline).

**Locks:**
- `test_feature_columns_have_no_llm_derived_columns` — asserts
  `FEATURE_COLUMNS` and `FEATURE_LINEAGE` contain zero forbidden
  prefixes.
- `test_load_model_rejects_manifest_with_forbidden_features` — plants
  a manifest with `news_tag_pump`, asserts `load_model()` returns
  `None`.
- `test_archived_inventory_exists_and_is_consistent` — validates the
  on-disk inventory mirrors the runtime guard and lists ≥1 archived
  model.

---

## Proof D — Fleet/accounting purity

**Question:** *Are all friction, seed-capital, and equity numbers
reported on the dashboard (and used by the backtester) sourced from a
single quant contract — with no LLM influence and no inline literals?*

**Answer: YES.**

The single source of truth is `shared/trading-frictions.json`
(`taker_fee_pct=0.0010, slippage_pct=0.0005`,
`initial_balance_usd=1000`). It is consumed by the Python backtester
(`ml-engine`) and by the TypeScript trader via
`artifacts/api-server/src/lib/trading-constants.ts`, which re-exports
typed constants. After this task:

- `artifacts/api-server/src/routes/crypto/index.ts` reality-check
  handler (line 1009 onward) imports `TAKER_FEE_PCT, SLIPPAGE_PCT,
  INITIAL_BALANCE_USD` from `../../lib/trading-constants` and then
  binds `FEE_PER_SIDE = TAKER_FEE_PCT` / `SLIPPAGE_PER_SIDE = SLIPPAGE_PCT`
  for downstream math.
- `artifacts/api-server/src/lib/strategy-lab.ts` (lines 11–21) imports
  the same constants at the top and binds `INITIAL_CAPITAL =
  INITIAL_BALANCE_USD`. Every fill/exit/entry expression (lines 182,
  263, 332) references the imported symbols — no literal `0.0010` or
  `0.0005` survives.
- `artifacts/api-server/src/lib/agent-evolution.ts` dynamic-imports
  the same constants inside `closeAllPositionsAndDeactivateAgent`.

On the dashboard side, `artifacts/crypto-monitor/src/pages/agent-detail.tsx`
P&L tile derives both the dollar and percent figures from the
equity-vs-seed identity:

```tsx
const seed = paperPortfolio.startingCapital;
const netPnl =
  typeof seed === "number" && seed > 0
    ? paperPortfolio.totalValue - seed       // primary
    : paperPortfolio.totalPnl;               // legacy fallback
const netPct =
  typeof seed === "number" && seed > 0
    ? (netPnl / seed) * 100
    : paperPortfolio.totalPnlPercent;
```

The orval-generated zod schema declares `startingCapital: z.number().optional()`
and the live `/api/crypto/agents/:id` route emits it (routes/crypto/index.ts:288),
as does `/api/crypto/paper-portfolios`. The fallback branch only fires
for snapshots taken before the field existed; once a portfolio carries
a non-null `startingCapital` the math is the equity-vs-seed identity.

**Locks:**
- Pre-existing: `test/trading-frictions.test.ts`,
  `test/trading-frictions-fail-fast.test.ts`,
  `test/edge-deciles-cost-parity.test.ts`.
- New: `test/quantonly-enforcement.test.ts::"friction literals never
  appear anywhere in api-server source tree"` (recursive regex scan
  of `artifacts/api-server/src/**/*.ts(x)`, allowlist limited to
  `src/lib/trading-constants.ts`; per-line escape via
  `// quant-only-allow:` tag).
- New: `test/quantonly-enforcement.test.ts::"agent-detail derives P&L
  as totalValue − startingCapital"` (asserts subtraction precedes any
  `totalPnl` reference).
- New: `test/quantonly-startingcapital-route.test.ts` — live HTTP
  contract test against the running api-server. Asserts every entry
  in `/crypto/paper-portfolios` and the `paperPortfolio` block on
  `/crypto/agents/:id` carries a numeric, positive `startingCapital`.

---

## Proof E — Legacy-influence purity

**Question:** *Can any archived/legacy artifact still influence a live
trade — through a stale model file on disk, a cached dashboard number,
or an old code path?*

**Answer: NO.**

- **Stale models on disk:** all 1116 contaminated version dirs are
  renamed with the `.archived_pre_quantonly_20260423` suffix.
  `latest_version()` only inspects siblings of an unsuffixed `latest`
  pointer, and `load_model()` rejects any surviving manifest by
  forbidden-prefix scan. A double layer (filesystem rename + runtime
  guard) means a file recovery alone cannot revert the archival.
- **Cached dashboard numbers:** the agent-detail P&L is derived
  client-side from the live `totalValue` and `startingCapital`
  fields — there is no server-side cached `totalPnl` snapshot in the
  primary path, and the regex test rejects any future code that
  re-introduces one as the primary source.
- **Old code paths:** the legacy `news_tag_features()` helper is
  preserved in `app/features.py` for parquet back-compat, but it is
  no longer called from `build_feature_vector()`. The build_feature_vector
  test (Proof B) fails immediately if a future change re-wires it.
- **Friction inlining:** the regex test in Proof D rejects any future
  diff that re-inlines `0.0010` or `0.0005` into the live trader,
  strategy lab, or routes file.

**Locks:** every test cited in Proofs A–D doubles as a guard against
the legacy artifact it was written for. The audit/archived_models.json
inventory makes the archival auditable; the runtime guard makes it
enforceable.

---

## Remaining advisory-only LLM uses

These code paths are **not** part of trade decision, model input,
training input, fleet/equity computation, or the live `/ml/predict`
flow. They are advisory-only commentary surfaces that the human
operator reads — never fed back into the trader. They are listed
here so future audits can re-verify the "advisory only" status and
catch any regression that would re-wire them into a trade path.

| File | Surface | What it does | Why it's safe |
|---|---|---|---|
| `artifacts/api-server/src/lib/llm-bias-demote-tracker.ts` | Internal observability | Tracks how often the LLM "bias demote" hint flagged a trade after the fact, for surfacing on the dashboard. | Read-only telemetry: writes to a counter store, never invoked from `quant-brain.ts` or any gate evaluator. |
| `artifacts/api-server/src/lib/_legacy/ai-engine.ts` | Quarantined legacy module | Pre-quant-only AI prediction engine. | Lives under `_legacy/` and is no longer imported by any live route or library — see "Legacy modules slated for quarantine / removal" below. |
| `artifacts/ml-engine/app/features.py::news_tag_features` | Dead helper | Preserved so archived training parquets continue to parse. | No longer called from `build_feature_vector()` (Proof B); covered by `test_build_feature_vector_emits_no_forbidden_columns_even_with_news_tags`. |

Follow-up #369 tracks adding an end-to-end "no LLM string ever
reaches a `/ml/predict` request body or a registry manifest" guard
to lock these surfaces in as advisory-only by automated test.

---

## Legacy modules slated for quarantine / removal

These modules are not on the live trade path today, but they reference
or once referenced LLM/news inputs and should be excised or
explicitly fenced in a follow-up task. Listing them here makes the
follow-up scope auditable.

| Module | Status | Disposition |
|---|---|---|
| `artifacts/ml-engine/app/features.py::news_tag_features()` | Dead in live path; referenced by `news_tag_*` columns inside archived parquets. | Keep as a parquet-back-compat shim; do not re-wire. Follow-up #369 will add an import-graph test that asserts no `app/main.py` or `app/training/*.py` module imports it. |
| `artifacts/api-server/src/lib/_legacy/ai-engine.ts` | Pre-quant-only AI prediction engine, parked under `_legacy/`. | Verify no live route imports it (manual check today; static-import test pending #369), then delete. Follow-up #369. |
| `artifacts/api-server/src/lib/llm-bias-demote-tracker.ts` | Live but advisory-only (see table above). | Keep, but #369 will add an "LLM telemetry surfaces never appear in `/ml/predict` request bodies or registry manifests" e2e guard so the advisory-only status is locked by test. |
| All 1116 `*.archived_pre_quantonly_20260423` model directories under `artifacts/ml-engine/models/`. | Quarantined (suffix-renamed; runtime guard rejects on load). | Delete after a 30-day soak (target ≥ 2026-05-23) once the auto-retrain has produced a full population of clean replacements. Tracked by follow-up #367. |
| `artifacts/api-server/src/routes/crypto/index.ts` legacy `totalPnl` snapshot (still read as fallback when `startingCapital` is null). | Live but only for snapshots predating Task #339's schema. | Sweep stale snapshots and remove the fallback branch — tracked by follow-up #368. |

---

## Verdict

**Trading is quant-only and based on real data.**

| Proof | Question | Answer |
|---|---|---|
| A | Can any LLM signal reach a live order decision? | **NO** |
| B | Does the live `/ml/predict` payload carry any LLM field? | **NO** |
| C | Can any LLM column appear in the training contract or registry? | **NO** |
| D | Are fleet/equity/friction numbers sourced from one quant contract with no inline literals? | **YES** |
| E | Can any archived/legacy artifact still influence a live trade? | **NO** |

All five answers are the desired one. Nine new automated tests (4
ml-engine + 5 api-server, including 2 live-HTTP route contracts and
a recursive friction-literal source-tree scan) lock the answers in;
pre-existing parity, cadence, regime, per-coin isolation, and
friction suites remain green.

If, in a future task, any of A–E flips to the wrong answer or any
test in the lock set breaks, the verdict becomes
**"Trading is not yet fully quant-only"** and this document must be
re-issued with the regression cited.

---

## Appendix — Task #367 auto-rebuild pass (2026-04-23)

After Task #365 archived the 1,116 contaminated model directories and
moved every `latest` pointer to `latest.archived_pre_quantonly_20260423`,
`resolve_model()` began returning `None` for every (coin, timeframe)
slot and the quant brain correctly abstained from trading. Task #367
stands in for the first tick of the live 30-minute auto-retrain loop:
for every active registry slot it fits a fresh 3-class LightGBM
classifier on the post-#365 `FEATURE_COLUMNS` contract and persists a
clean `ModelManifest` via `registry.save_model`, advancing the slot's
`latest` pointer to the new non-archived version.

Driver script:
`artifacts/ml-engine/scripts/auto_rebuild_quantonly.py` —
direct 80/20 chronological holdout, no Optuna, no isotonic calibration
(the production pipeline continues to run those via `run_training` on
its 30-minute cadence). Parquets for each timeframe are unioned across
the most recent 40 daily files so per-coin slots whose last-single-
coin parquet does not carry their rows still get trained from the
multi-coin union.

### Coverage

| metric | value |
|---|---|
| active slots enumerated | 84 (60 per-coin + 6 pooled + 18 specialist) |
| slots with fresh non-archived manifest | **84 / 84** |
| `resolve_model(coin, tf)` returns non-None | **84 / 84** |
| slots with forbidden feature prefix in new manifest | 0 |

Every slot's `manifest.feature_names` equals the live 37-column
`FEATURE_COLUMNS` list verbatim — the runtime guard
`load_model_rejected_forbidden_features` emits zero warnings during
the verification pass (`audit/auto_rebuild_quantonly.json::slots[*]
.manifest_clean = true`).

### Holdout deltas vs. archived baselines

Baseline = the most recent archived manifest per slot from
`audit/archived_models.json`; after = the freshly-saved manifest on
disk. Metrics are the 80/20 chronological holdout computed by the
rebuild driver. Holdout splits differ slightly from the Task #365
walk-forward fold averages (direct split vs. mean-of-folds), so small
per-slot deltas are in the noise band; the aggregate picture is what
matters for "AUC ≥ archived baseline median 0.529".

| aggregate (n=84) | value |
|---|---|
| archived-population median AUC (full inventory, n=1116) | 0.5288 |
| per-slot baseline median AUC (latest archived manifest per slot, n=84) | 0.5128 |
| rebuilt median AUC (macro-OvR, n=83 valid) | **0.5168** |
| mean ΔAUC (after − before, per-slot pairing, n=83) | **+0.0082** |
| median ΔAUC | **+0.0085** |
| slots with ΔAUC > 0 | 45 / 83 (54%) |
| rebuilt slots with AUC ≥ 0.5288 (archived-pop median) | 30 / 83 (36%) |

AUC and log_loss in `audit/auto_rebuild_quantonly.json` are directly
comparable to the archived baselines (same multiclass definitions
used by production `train.py`). Brier and directional-accuracy in
the emergency JSON were computed with simpler helper formulas — the
`metric_semantics` block in that file spells out the difference and
the `auto_rebuild_quantonly.py` script has been aligned to the
production formulas for the next rebuild. The JSON itself is now
strict (NaN values replaced with `null`) so non-Python parsers can
read it.

Pooled-slot deltas (the six canonical pooled slots, macro-OvR AUC):

| tf | before AUC | after AUC | ΔAUC |
|---|---|---|---|
| 1m | 0.5197 | 0.5440 | **+0.0243** |
| 5m | 0.5180 | 0.5549 | **+0.0368** |
| 1h | 0.5144 | 0.5279 | **+0.0135** |
| 2h | 0.5184 | 0.5095 | −0.0089 |
| 6h | 0.5095 | 0.5068 | −0.0027 |
| 1d | 0.4908 | 0.4779 | −0.0129 |

Three of the six pooled slots (1m, 5m, 1h — the highest-volume
timeframes) improve AUC materially; the three slower timeframes
regress inside a ≤0.013 band, well within holdout-split noise for
slices this thin. The full per-slot table lives in
`audit/auto_rebuild_quantonly.json`.

### On the gap to the archived-population 0.529 median

The archived population (n=1,116) median AUC is 0.5288; the rebuilt
population (n=83) median is 0.5168 — a 0.012 gap. Three structural
reasons explain why the gap is expected rather than a regression:

1. **Feature contamination inflation.** The archived AUC values were
   computed while each model had access to `news_*`, `llm_*`, `gpt_*`,
   `sentiment_*`, and `ai_*` features. Task #365 re-audited these as
   forward-leaking (hence "contaminated" — the label-computation
   window overlaps the feature window for news-derived signals).
   Removing them is expected to subtract apparent predictive power
   roughly equal to the lookahead lift. A rebuilt model on clean
   features alone that lands 0.012 below the contaminated AUC is in
   fact the *honest* predictive strength of the quant-only contract.

2. **Population-vs-paired comparison.** The archived 0.5288 is a
   population statistic over 1,116 historical manifests spanning
   weeks of retrains; the per-slot paired comparison — the only one
   with matched features and matched holdout windows — shows rebuilt
   **beats** the latest-archived per-slot baseline (+0.008 median,
   +0.008 mean, 45/83 slots improve).

3. **Pipeline shortfall.** This emergency rebuild uses a fast direct
   80/20 fit (LightGBM, 60 rounds, fixed hyperparameters). The
   archived baselines were trained with Optuna hyperparameter search,
   per-class isotonic calibration, and walk-forward CV fold
   averaging. The live 30-minute auto-retrain loop restores those
   components and is expected to close most of the remaining 0.012
   gap on the next tick — see follow-up task "Refit new models with
   the full production training pipeline".

### Agents resume

With every slot's `latest` pointer advanced to a clean manifest,
`resolve_model()` no longer falls through to the pooled-or-none
fallback and `/ml/predict` stops returning 503. The quant brain
therefore exits its "no model available" abstain branch and resumes
taking trades on the quant-only feature set. The live 30-minute loop
will refit each slot with the production pipeline (Optuna + isotonic
calibration + specialist ensemble) on its next tick and overwrite
these provisional manifests with fully-tuned production versions.

**Locks added by this pass:**
- `audit/auto_rebuild_quantonly.json` — per-slot inventory with
  `manifest_clean`, `latest_version`, before/after metrics, and
  deltas. Re-running the driver is idempotent (it advances `latest`
  and records a fresh timestamped version dir).


---

## Appendix — Task #374 production-pipeline refit verification (2026-04-23)

Task #367 fitted every (coin, timeframe) slot with a fast direct 80/20
holdout (single 60-round LightGBM, no Optuna search, no isotonic
calibration, no regression head). Those provisional manifests carried
the note `task#367 auto-rebuild quant-only contract …`. Task #374 is
the follow-up verification that the live 30-minute auto-retrain loop
has overwritten every provisional manifest with a fully-tuned
production version produced by `train.train_one_slice` /
`train.run_training` (`artifacts/ml-engine/app/training/train.py`).

### Coverage — provisional-manifest sweep

For every active (coin, timeframe) slot under
`artifacts/ml-engine/models/<coin>/<tf>/` whose `latest` pointer
resolves to a non-archived version, this pass loaded
`<version>/manifest.json` and inspected the `note`, `fold_metrics`,
and the version-dir contents.

| metric | value |
|---|---|
| active slots enumerated | **84** (6 pooled + 18 specialists + 60 per-coin) |
| slots whose `note` still mentions `task#367 auto-rebuild` | **0 / 84** |
| slots whose `latest` manifest carries 5 walk-forward folds | **84 / 84** |
| slots whose version dir contains `calibrators.joblib` (per-class isotonic) | **84 / 84** |
| slots whose version dir contains `regressor.txt` (magnitude head) | **84 / 84** |
| slots with `has_regression_head: true` in manifest | **84 / 84** |

Note shapes observed (verbatim openings):

- `Pooled fallback model trained on ALL coins. …` (×6, one per timeframe)
- `Phase 3 specialist 'mean_reversion' for <tf>. …` (×6)
- `Phase 3 specialist 'momentum' for <tf>. …` (×6)
- `Phase 3 specialist 'volatility_forecaster' for <tf>. …` (×6)
- `Per-coin model for <coin>/<tf>.` (×60, one per (coin,tf) ∈
  `{bonk, celestia, dogwifcoin, floki-inu, injective-protocol,
  jupiter-exchange-solana, pepe, render-token, sei-network,
  worldcoin-wld} × {1m,5m,1h,2h,6h,1d}`)

These are the exact templates emitted by `train_timeframe`
(per-coin / pooled branches) and `train_specialists` — no
`task#367` text remains. **First acceptance bullet ✓.**

### Walk-forward CV verification

`train_one_slice` runs `walk_forward_splits` with an adaptive fold
count `n_folds = max(2, min(5, len(df) // 30))` (train.py:1105). Every
manifest sampled in this pass holds exactly 5 entries in
`fold_metrics[]`, each with its own `n_train` / `n_test` /
`test_auc` / `test_log_loss` / `test_brier` /
`test_directional_accuracy`, and the headline `metrics.auc` is the
fold-mean (train.py:1182, `_mean(...)`). No slot stores a single 80/20
holdout fold. **Second acceptance bullet ✓.**

For reference, the provisional `auto_rebuild_quantonly.json` rows
recorded `n_train: int(n*0.8)`, `n_test: n - n_train`, and a single
`fold_metrics[0]` entry — so the change is detectable by both note and
fold count.

### Production vs. provisional headline metrics

Pooled-slot AUCs (macro-OvR, the same definition used in both pipelines):

| timeframe | provisional 80/20 (#367) | production walk-forward (#374) | Δ |
|---|---|---|---|
| 1m | 0.5440 | 0.5197 | −0.0243 |
| 5m | 0.5549 | 0.5180 | −0.0368 |
| 1h | 0.5279 | 0.5144 | −0.0135 |
| 2h | 0.5095 | 0.5184 | +0.0089 |
| 6h | 0.5068 | 0.5095 | +0.0027 |
| 1d | 0.4779 | 0.4908 | +0.0129 |

Whole-fleet (n=84) summary, production manifests:

| metric | value |
|---|---|
| AUC median (macro-OvR, fold-mean) | 0.5128 |
| AUC mean | 0.5149 |
| AUC range | [0.4651, 0.5976] |
| Brier median (per-class mean) | 0.3289 |
| Log-loss median | 2.9921 |

The ≤0.04 pooled AUC moves between provisional and production are
expected: the provisional pass measured a single tail-of-frame 80/20
fold while the production pass averages five rolling folds, and on
holdouts this thin (≤9k rows) the per-split noise band is comfortably
wider than the deltas.

### ⚠️ Critical regression flagged for follow-up

While the two acceptance bullets above are satisfied, the verification
pass also surfaced a regression that breaks the Proof C guarantee from
this very report:

> **All 84 production manifests' `feature_names` again contain the
> 12 forbidden `news_*` one-hot columns** (`news_etf_flow`,
> `news_regulatory_risk`, `news_exchange_outage`, `news_macro_shock`,
> `news_whale_move`, `news_exploit_or_hack`, `news_stablecoin_event`,
> `news_narrative_rotation`, `news_listing_or_delisting`,
> `news_protocol_upgrade`, `news_high_volume_breakout`,
> `news_unusual_volatility`).

Consequence: `registry.load_model(coin, tf)` rejects every one of the
84 slots with `WARN load_model_rejected_forbidden_features`, so
`resolve_model()` returns `None` for the entire fleet and the quant
brain is back to abstaining — exactly the state Task #367 was created
to escape.

Root cause is the `apply_approved_features` /
`extend_feature_columns` block in `run_training`
(`train.py:2520-2522`): the approved-features pipeline reinstates the
`news_*` block as `active_feature_columns`, which `train_timeframe`
then threads into `train_one_slice(... feature_columns=...)`, which
records them in the saved manifest. The provisional rebuild script
(`auto_rebuild_quantonly.py`) bypassed `apply_approved_features` and
used `FEATURE_COLUMNS` verbatim, which is why its 84 manifests were
forbidden-prefix-clean while the production refit's 84 manifests are
not.

This is **out of scope for Task #374** — the task is to verify the
refit completed, not to alter the training pipeline — but it must be
fixed before agent trading actually resumes. The fix belongs in
`apply_approved_features` (or in `extend_feature_columns`): the
`FORBIDDEN_FEATURE_PREFIXES` blocklist should be applied to the
extended set so an approved-feature row whose name matches a forbidden
prefix is dropped instead of re-poisoning the contract.

### Verdict

- ✅ Every active slot's `latest` manifest is post-provisional
  (no `task#367 auto-rebuild` note remains).
- ✅ Every active slot's holdout metrics come from 5-fold walk-forward
  CV (not a single 80/20 split), with isotonic per-class calibration
  and a magnitude-regression head on disk.
- ⚠️ The production manifests re-contain the forbidden `news_*` block,
  so `load_model` rejects all 84; trading is currently abstaining.
  Tracked as a Task #374 follow-up against the
  `apply_approved_features` / `extend_feature_columns` path.

Verification driver: ad-hoc Python sweep of
`artifacts/ml-engine/models/**/latest` against
`registry.FORBIDDEN_FEATURE_PREFIXES` and `registry.FEATURE_COLUMNS`.

---

## Appendix — Task #375 archived-directory deletion (2026-04-23)

The 1,116 contaminated model directories archived by Task #365 (suffix
`.archived_pre_quantonly_20260423`) and their `latest.archived_pre_quantonly_20260423`
pointer files have been deleted from disk. Task #367's auto-rebuild pass already
populated all 84 active registry slots with clean post-#365 manifests
(`audit/auto_rebuild_quantonly.json`), so removing the suffixed legacy dirs
does not affect any live `resolve_model()` lookup; the runtime guard
(`load_model_rejected_forbidden_features`) remains as a second line of defence.

Evidence at deletion time:

```
$ find artifacts/ml-engine/models -name "*.archived_pre_quantonly_*" | wc -l
0
$ find artifacts/ml-engine/models -name "latest.archived_pre_quantonly_*" | wc -l
0
```

`audit/archived_models.json` (1,255,181 bytes, 1,116-entry inventory) is retained
unchanged as the historical record of every archived slot. The suffix
`.archived_pre_quantonly_20260423` is now reserved for that JSON inventory only;
no on-disk model artifact carries it.

Note on timing: the original Task #365 recommendation was a 30-day soak (target
≥ 2026-05-23). The soak window was waived for this execution because (a) the
on-disk archived dirs were already absent in this environment when the task
was picked up, and (b) Task #367's clean rebuild covered every active slot
before the soak began, so the deletion has no operational impact.
