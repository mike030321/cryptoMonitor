# Full-System Audit — 2026-04-23

Read-only proof-based audit of the quant platform (api-server, ml-engine,
crypto-monitor, Postgres). No code, schema, or model writes were made.
Every claim cites a file:line, a captured log, or a SQL/HTTP capture under
`.local/audit/<group>/`. Per-group raw sections live in
`.local/audit/0{1..9}-*/section.md` and `.local/audit/10-*/section.md`.

> **Brief deviation.** The brief references
> `artifacts/ml-engine/models/training_run_20260423T181613Z/`. **That path
> does not exist on disk.** No `training_run_*` directory exists anywhere
> in the repo. The actual training artifacts live directly under
> `artifacts/ml-engine/models/{report.json, fast_loop_report.json,
> auto_retrain_transitions.jsonl, verification_history.jsonl, datasets/,
> __meta__/, __pooled__/, <coin>/}`. The latest slow-loop run is
> `report.json @ 2026-04-23T10:34:15Z`; the latest fast-loop tick is
> `2026-04-23T20:23:14Z`; the latest failure-analysis report is sourced
> from a `2026-04-23T18:27:53Z` slow-loop run that itself has no on-disk
> artifact. Group 8 walks what is on disk and treats `18:16:13Z` as
> phantom.

---

## 1. Master report by 10 audit groups

| # | Group | Health | Headline |
|---|-------|--------|----------|
| 1 | Runtime architecture | 🟡 | Three-process topology + health probes proven; one boot race + several env-var fragility flags. |
| 2 | Quant brain (LightGBM) | 🔴 | `/ml/predict` returns **HTTP 500 on 36/36 probes**; quant brain has produced no journal row since 15:18 UTC. |
| 3 | Meta-Brain (governance only) | 🟢 (code) / 🟡 (live, unverified) | Code is clean — directive consumed only at sizing, evaluate + record_outcome round-trip works on demand. The audit shell did not observe `META_BRAIN_ENABLED` / `META_BRAIN_SHADOW`; if the live api-server worker is also missing them, `bindTradeToTick` is a no-op for every trade and there is no production learning loop. Confirming the worker env requires a follow-up with log access (see §C). |
| 4 | Strategy Lab / benchmark | 🟡 | DCA snapshot reconciles to $0.03; **Buy & Hold (agent 18) is a fixed-point lie at $1000** which biases the meta-brain benchmark and freezes the trend-filter exit signal. |
| 5 | Monitor cycle | 🟡 | Cycle ordering matches code; one-tick-lag wiring of meta-brain proven; **but 100 % of recent cycles abstain (`quant_disabled`)** and no paper trade has executed since 10:05 UTC. |
| 6 | Paper trading / hard PnL truth | 🟡 | Fee/slippage formulas single-sourced; recompute matches stored PnL bit-for-bit (|Δ| ≤ 8e-6 USD); **all 262 auto-deploy basket buys have NULL `entry_fee`**; 3 stale equity rows on prod replica. |
| 7 | Risk / safety boundaries | 🟢 | Every guard cited end-to-end. EV-vs-cost gate fired 165× / 7d; sector cap fired; fleet-correlation brake fired 130×; auto-quarantine wired but 0 fires (consistent with no model collapse in window). |
| 8 | Training campaign | 🔴 | Phase-4 promotion gate has **promoted 0 slices**. Every active manifest still ships `news_*` columns the loader is supposed to reject. Pooled fallback `coin_vocab=["jupiter-exchange-solana"]` only — pooled = jupiter for every other coin. |
| 9 | DB / schema integrity | 🟢 | All 5 invariants pass: 0 orphan positions, 0 duplicate closes, 0 journal orphans, monotone snapshots, 0 market-signal gaps > 5 min in 24 h. Caveat: FKs are application-level only. |
| 10 | Dashboard / observability | 🟡 | Most cards reproduce from DB; **`/quant-calibration-history` silently returns empty**; headline `overallAccuracy=56.5%` and "Best Agent 83.19 %" hide the live brain abstaining and 24 h directional accuracy of 19.5 %. |

Legend: 🟢 proven correct • 🟡 mostly correct, real issues • 🔴 contract broken in production today.

---

## 2. Proof ledger

Every claim in this report is backed by a captured artifact. Index:

| Group | Section | Key evidence files |
|-------|---------|--------------------|
| 1 | `.local/audit/01-runtime/section.md` | `health-probes.json`, captured workflow boot logs |
| 2 | `.local/audit/02-quant-brain/section.md` | `predict_live_500.txt`, `journal_5_quant_rows.csv`, `brain_hourly_24h.txt`, `feature_hash_population.txt`, `api_server_quant_failures.log` |
| 3 | `.local/audit/03-meta-brain/section.md` | `evaluate_request.json`, `evaluate_sample.json`, `record_outcome_sample.json`, `stats_after_*.json`, `last_directives.jsonl` |
| 4 | `.local/audit/04-strategy-lab/section.md` | `snapshots_recent.csv`, `dca_reconciliation.txt`, `no_llm_fields_test.txt` |
| 5 | `.local/audit/05-monitor-cycle/section.md` | `monitoring_state.csv`, `cycle7_predictions.csv`, `cycle7_journal.csv`, `last_paper_trades.csv`, `recent_meta_directives.csv` |
| 6 | `.local/audit/06-pnl/section.md` | `paper_trades_7d_dev.csv`, `recompute_sample_v2.csv`, `equity_reconcile_dev.csv`, `phantom_pnl_dev.csv`, `duplicate_closes_dev.csv`, `fees_null_zero_dev.csv`, `api_vs_db_portfolios.csv` |
| 7 | `.local/audit/07-risk/section.md` | `skip_events_by_reason.csv`, `skip_events_samples.csv`, `portfolios_snapshot.csv`, `quarantine_state.csv` |
| 8 | `.local/audit/08-training/section.md` | `predict_sample.json` (booster round-trip), `latest_pointers.txt`, `auto_retrain_status.json`, `failure_analysis_latest.json`, `verification_history.jsonl` |
| 9 | `.local/audit/09-db/section.md` | 5 invariant CSVs (orphans/duplicates/journal/snapshots/market_signals) |
| 10 | `.local/audit/10-dashboard/section.md` | 13 raw API JSON captures + per-card DB cross-checks |

Every Section A/B/C bullet cites an explicit `path/to/file.ts:LINE-LINE` or
a CSV row. No claim in the master report is unanchored.

---

## 3. Hard PnL truth

**Per-trade PnL accounting is correct.** Stored `paper_trades.pnl` was
recomputed from raw inputs for 20 closed trades using the formulas at
`paper-trader.ts:1075-1083`; per-trade `|Δ| ≤ 8e-6 USD` (float rounding).
See `.local/audit/06-pnl/recompute_sample_v2.csv`. Fee + slippage
constants flow from `shared/trading-frictions.json:3-7` through
`trade-math.ts:14-44` with no parallel definitions
(`maker_fee=0.001, taker_fee=0.001, slippage=0.0005`, round-trip 0.30 %).

**Per-agent equity reconciles.** For the 24 dev-DB agents, 22 / 24 satisfy
`|total_value - (cash + open_notional)| ≤ $5.32`, with the residual fully
explained by unrealized MTM net of est. exit fee + slippage
(`paper-trader.ts:1311-1370`). Worked example — agent 23 (Hybrid-4-3):
realized $-10.41, equity drift $-12.10, residual $-1.69 = unrealized loss
on $367 of open notional (-0.46 % MTM). Same shape for agent 19. No
phantom equity on dev.

**Adversarial scans are clean on dev.** No phantom profits
(`pnl > position_size`), no duplicate closes by `(prediction_id, agent_id)`,
no open positions older than 24 h that aren't intentional 1d basket
positions held by the auto-deploy bots (agents 17 / 20).

**EV-vs-cost gate works.** 165 `quant_ev_below_costs` fires in the last 7 d
(`.local/audit/07-risk/skip_events_by_reason.csv`). Latest:
`requiredPct=0.900 % vs directionalReturnPct=0.830 %` ⇒ gated, no trade
written.

**Trading is currently muted.** No paper trade since 2026-04-23 10:05:55Z
(`.local/audit/05-monitor-cycle/last_paper_trades.csv`, latest id 3063).
The system has been emitting `quant_abstain_quant_disabled` skips for
~10 hours because `app_settings.quant_brain_enabled` row is missing
(`brain-flag.ts:18,86` defaults off). Combined with Group 2's finding that
`/ml/predict` is 500-ing on every coin/timeframe, the quant brain is
non-operational end-to-end and the journal has been writing
`brain='LLM'` rows with NULL specialist fields — a direct contradiction
of the "LLM cannot author trades" contract.

**Production-replica anomalies (NOT in dev).** Three agents on the
production replica show stale equity not present in dev:
agent 31 (+$197 phantom equity), agent 32 (-$101 unrealized vs $179 open
notional ≈ -57 % MTM), agent 39 (cash $1000.25 / open $0 / equity
$801.48 ⇒ -$198.77 stale). Most likely `updatePortfolioValues` short-
circuited on stale prices (`paper-trader.ts:1312`). The running api-server
trades against dev DB and is unaffected, but a deploy would inherit this
state.

---

## 4. A/B/C summary across all groups

### A. Proven correct (selected, full evidence in per-group sections)

- Three-process topology (api-server :8080, ml-engine :8000 with
  `--root-path /ml`, crypto-monitor :21714); 4/4 health endpoints respond
  200 with bodies (`.local/audit/01-runtime/health-probes.json`).
- The `/ml/predict` request body is strictly `{coinId, timeframe}` —
  zero LLM/news/sentiment/benchmark fields. Verified at
  `ml-client.ts:166-168`, OpenAPI schema, and ml-engine handler
  `main.py:861-871` (zeroes news_tags channel).
- Meta-brain directive is consumed only at sizing in paper-trader
  (`paper-trader.ts:571-603`, `:888`); `quant-brain.ts` and `ml-client.ts`
  contain ZERO references to `getActiveDirective` or trust/allocation
  fields. Defensive_mode === "hard" floor relaxation lives only in
  `clampMetaSizeMultiplier` (`paper-trader.ts:256-271`).
- Fee/slippage/starting-capital formulas single-sourced from
  `shared/trading-frictions.json` through `trade-math.ts` and
  `trading-constants.ts`. DCA snapshot reconciles to ±$0.03 over 10 open
  positions; 20 closed trades recompute to ±8e-6 USD.
- Cycle ordering matches `monitor.ts:541-1536`. One-tick-lag wiring proven:
  trades on cycle N consume the directive cached at end of cycle N-1.
- All 5 DB invariants pass: 0 orphan positions, 0 duplicate closes, 0
  journal orphans, snapshots monotone, 0 market-signal gaps > 5 min.
- Phase-4 promotion gate is implemented and disciplined
  (`verification.py:62-146`): `holdout_rows ≥ 200`, `da > 0.50`,
  `da > baseline_da`, cadence-mix block, every active coin must have ≥1
  promoted slice for the run to pass. Auto-shadow registration on every
  trained slice (`register_shadow.py:154-233`).
- 13 dashboard cards cross-checked against DB: `dashboard`, `predictions`,
  `paper-portfolios`, `strategy-lab`, `brain/accuracy`, `liquidations`,
  `market-signals-health`, `failure-analysis`, `regime`. All reproduce
  within single-digit row drift (live cycle ticked between captures).

### B. Proven broken (every item linked to fixable code or state)

- **B-PRED-500.** `/ml/predict` returns HTTP 500 on every coin/timeframe.
  Root cause: the fast-path branch in `_pick`
  (`ml-engine/app/main.py:691-694`) calls `_cached_load(...)` without a
  try/except, and `_cached_load` raises `RuntimeError` when the loader
  rejects a manifest for forbidden `news_*` features. The descending
  fallback only runs in the *quarantined* branch (`:698-707`), so a
  forbidden-feature reject on the `latest` pointer escapes uncaught and
  the pooled fallback (`_resolve_for_predict:707-712`) is never tried.
  Pooled `latest` carries the same `news_*` columns and would also
  reject. (`02-quant-brain/section.md` §B; reproduction in
  `predict_live_500.txt`.)

- **B-LLM-AUTHORSHIP.** Journal rows for the last ~5 hours are stamped
  `brain='LLM', source=NULL, model_version=NULL, prob_*=NULL,
  specialist_scores=NULL` while still carrying a `direction`. This
  contradicts the documented "LLM cannot author trades" contract — an
  LLM-side fallback path is writing decisions while quant is down.
  (`02-quant-brain/section.md` §B; `brain_hourly_24h.txt`.)

- **B-FEATHASH.** ~37 % of QUANT journal rows have NULL `feature_hash`
  (e.g. jupiter/1h: 187/415 null = 45 %). The "every QUANT row is
  reproducible offline" contract is not being met.
  (`02-quant-brain/section.md` §B; `feature_hash_population.txt`.)

- **B-QUANT-OFF.** `app_settings` has no `quant_brain_enabled` row. Default
  is OFF (`brain-flag.ts:18,86`; `shared/trading-frictions.json:68`). 100 %
  of recent cycles abstain with `quant_abstain_quant_disabled`; no paper
  trade since 2026-04-23 10:05:55Z. (`05-monitor-cycle/section.md` §B;
  `last_paper_trades.csv`.)

- **B-BUYHOLD-LIE.** Strategy Lab agent 18 (Buy & Hold) is pinned at
  $1000 equity with **zero positions and zero `paper_trades` rows**
  despite `strategy_state.initial_deploy_done=TRUE` and a
  `last_buy_at=2026-04-20T07:13Z`. Consequences:
  (a) meta-brain `bestBaselineReturn7d/14d` are biased low,
  `relativeAlpha7d/14d` for the AI fleet biased high,
  `sustainedUnderperformance` flag will never trip;
  (b) the trend-filter exit signal uses Buy & Hold as basket-return proxy
  (`strategy-lab.ts:341-352`), so Trend Filter (agent 20) cannot exit
  until Buy & Hold is repaired. (`04-strategy-lab/section.md` §B.)

- **B-MOCK-TESTS.** `mock.module is not a function` on Node 24.13 cancels
  `test/no-llm-fields-runtime.test.ts` (10 tests) and
  `test/meta-brain-benchmark.test.ts` (1 fail). The static parity guard
  still passes (22/22) but the runtime cross-check is silently skipped —
  a regression that adds an LLM- or benchmark-derived key to a runtime
  payload would NOT be caught. (`04-strategy-lab/section.md` §B;
  `no_llm_fields_test.txt`.)

- **B-AUTODEPLOY-NOFEE.** All 262 auto-deploy basket `action='buy'` rows
  have `entry_fee IS NULL`. Cause: insert at `paper-trader.ts:1687-1699`
  omits the column (the "real" trader insert at `:843-856` includes it).
  Cash debit at `:1722` also omits the entry-fee deduction. Effect:
  `recordOutcome` (`:1196-1198`) reads `entryFee ?? 0` and reports an
  understated `turnover_cost` to the meta-brain for these strategies.
  Realized PnL on the row itself is unaffected (no cash mismatch — the fee
  simply wasn't charged), but the brain's bounded learning loop sees these
  bots as cheaper than they would be in live trading. Same NULL pattern on
  the production replica (255/255 buys). (`06-pnl/section.md` §B;
  `fees_null_zero_dev.csv`.)

- **B-PROMOTE-ZERO.** Phase-4 promotion gate has promoted **zero slices**
  this campaign. Every per-coin LightGBM slice is in
  `structurally_noisy_retire` (calibration max_dev ≥ 0.10) or
  `insufficient_sample`. Every 1h/2h/6h slice's net PnL after costs is
  negative (`failure_analysis_latest.json`). Verification stall_streak=1.
  (`08-training/section.md` §B.)

- **B-MANIFEST-NEWS.** Trainer is still emitting `news_*` columns into
  manifests on every today-stamped version, even though the loader
  enforcement list (`registry.py:262-273`,
  `FORBIDDEN_FEATURE_PREFIXES`) explicitly rejects them. This is the
  upstream cause of B-PRED-500. (`08-training/section.md` §B.)

- **B-POOLED-VOCAB.** `__pooled__` model has
  `coin_vocab=["jupiter-exchange-solana"]` only — `n_train_rows=8723`
  matches the per-coin jupiter slice. Routing any other coin to pooled
  defaults `coin_idx=0=jupiter` ⇒ pooled inference is effectively
  jupiter-specific for every other coin. (`08-training/section.md` §B.)

- **B-DIR-CALL.** Pooled 1h `directional_call_share=0.9988` — the
  calibrated head emits a directional call on >99.8 % of holdout rows.
  This is exactly the regression the dashboard card was added to catch.
  (`08-training/section.md` §B.)

- **B-CALIB-CARD.** `/api/crypto/quant-calibration-history` always returns
  empty `series`; no source table exists in the DB. Card is decorative
  and silently empty (no 5xx so monitoring won't catch it).
  (`10-dashboard/section.md` §B.)

- **B-AVG-HIDES.** Headline `overallAccuracy=56.5 %` and "Best Agent
  83.19 %" hide the fact that 24 h QUANT 1h directional accuracy is
  19.5 % (n=1 119, below the 45 % coin-flip floor) and the brain is
  abstaining 100 % right now. Classic averaging-conceals-truth.
  (`10-dashboard/section.md` §B.)

- **B-EQUITY-PROD-STALE.** Production-replica equity rows for agents 31
  (+$197), 32 (-$101 vs -$0.81 expected), and 39 (-$198.77 with no open
  positions) are stale. (`06-pnl/section.md` §B.)

### C. Uncertain (could not confirm read-only; needs follow-up evidence)

- `META_BRAIN_ENABLED` / `META_BRAIN_SHADOW` not visible in this shell;
  if the live api-server worker also has them unset, `bindTradeToTick`
  is a no-op for every live trade and there is no learning loop in
  production today. (Group 3.)
- `trust_model.json` on disk is `{}` (2 B). Either the api-server has
  not been driving evaluate calls or the periodic checkpoint
  (`meta_brain.py:328-362`) has not run since `_service` last accepted
  outcomes. `directives.jsonl` has 141 entries. (Group 3.)
- `ML_ENGINE_URL` not set — calls happen to work because both layers add
  `/ml`; setting `ML_ENGINE_URL` to include `/ml` would 404 every call.
  ml-engine has no `ADMIN_API_KEY`/`ML_ADMIN_TOKEN`, so
  `triggerScheduledRetrain` may be unreachable. `COINGLASS_API_KEY` not
  set ⇒ market-signals poller silently degrades. (Group 1.)
- Auto-quarantine has 0 events / 0 quarantined rows out of 125 in
  `model_registry`. Path is wired but cannot be exercised in read-only
  mode. (Group 7.)
- Slippage column does not exist on `paper_trades`; cross-check via
  `trade_journal` (`paper-trader.ts:1241-1268, slippagePct: SLIPPAGE_PCT`)
  was not run in this scope. (Group 6.)
- `paper_trades.action` enum has no `'open'`/`'close'` values — closes
  are UPDATEs in place (`paper-trader.ts:1108-1117`). Brief's wording
  for "fees on action='open'/'close'" was scoped to all buys/sells
  instead. (Group 6.)
- Boot race at `training-contract-notifier` — fetch fails ~2 s before
  ml-engine ready; first poll succeeds on next interval. Self-heals;
  flagged. (Group 1.)

---

## 5. Risk-ranked bug list

| # | ID | Severity | Title | One-line fix |
|---|----|----------|-------|--------------|
| 1 | B-PRED-500 + B-MANIFEST-NEWS | **P0** | `/ml/predict` 500 on every coin/timeframe; trainer still emits `news_*` columns the loader rejects | Wrap `_cached_load` in `_pick` fast-path with try/except so pooled fallback is reached; **and** strip `news_*` (and other forbidden prefixes) from `FEATURE_COLUMNS` before training, then re-train. |
| 2 | B-LLM-AUTHORSHIP | **P0** | Journal rows stamped `brain='LLM'` with NULL specialist fields are still carrying a `direction` while quant is down — violates "LLM cannot author trades" contract | Make the LLM fallback path write `direction='stable'` + `outcome='abstained'` (matches the abstain contract), or fail loud and refuse to write any direction without a quant prediction. |
| 3 | B-QUANT-OFF | **P0** | `app_settings.quant_brain_enabled` missing; 100 % abstain since 10:05 UTC | Insert the row OR fix the Phase-4 gate so quant has a champion to promote. (Will only matter once #1 is fixed.) |
| 4 | B-PROMOTE-ZERO + B-DIR-CALL | **P0** | Promotion gate has promoted zero slices; calibration is broken (max_dev ≥ 0.10), directional_call_share = 0.9988 | Investigate label / feature drift; tighten or rebalance class targets; consider raising `MIN_HOLDOUT_ROWS` or adding a min-stable-share floor. |
| 5 | B-BUYHOLD-LIE | **P1** | Buy & Hold (agent 18) stuck at $1000, distorting meta-brain benchmark and freezing trend-filter exit | Re-run `initialDeploy` for agent 18 with the breaker reset, OR change the trend-filter basket-return proxy off Buy & Hold. |
| 6 | B-AUTODEPLOY-NOFEE | **P1** | 262/262 auto-deploy basket buys have NULL `entry_fee`; cash also not debited for the entry fee on these paths | Add `entryFee` to the insert at `paper-trader.ts:1687-1699` and debit `cashBalance - (allocate + entryFee)` at `:1722`. |
| 7 | B-POOLED-VOCAB | **P1** | Pooled fallback was trained on jupiter-exchange-solana only; routes every other coin through a jupiter-specific model | Re-train pooled with the full coin vocab; assert `len(coin_vocab) > 1` in `verification.py`. |
| 8 | B-FEATHASH | **P2** | ~37 % of QUANT journal rows have NULL `feature_hash` — reproducibility contract broken | Make `feature_hash` non-null in journal write path; add a not-null DB constraint after backfill. |
| 9 | B-CALIB-CARD | **P2** | `/quant-calibration-history` silently empty; card decorative | Either delete the card and the route, or wire it to `quant_calibration_history` (table doesn't exist — needs schema). |
| 10 | B-AVG-HIDES | **P2** | Headline `overallAccuracy=56.5 %` and Best Agent 83 % hide the live abstain + 19.5 % 24h directional accuracy | Add a 24h rolling card next to the headline; demote "Best Agent" when its 24h directional sample size > 0 and accuracy < coin-flip floor. |
| 11 | B-MOCK-TESTS | **P2** | `mock.module` regression on Node 24 silently skips the runtime no-llm-fields + meta-brain-benchmark suites | Pin Node version OR replace `mock.module` with `vi.mock`/`tap`-compatible patch. |
| 12 | B-EQUITY-PROD-STALE | **P3** | Production-replica equity for agents 31/32/39 is stale | Backfill `paper_portfolios.totalValue` from `cash + Σopen MTM` once price feed is fresh; flag as runbook step. |
| 13 | Boot race (Group 1 §B) | **P3** | `training-contract-notifier` fetch fails ~2 s before ml-engine is ready | Retry with backoff in `training-contract-notifier.ts:230-234`, or wait on `/ml/health` before first poll. |
| 14 | Env-var fragility (Group 1 §C) | **P3** | `ML_ENGINE_URL` unset; ml-engine missing `ML_ADMIN_TOKEN`; `COINGLASS_API_KEY` missing | Pin `ML_ENGINE_URL`, drop the redundant `/ml` from one of the two layers; populate the missing keys in env-secrets. |

(P0 = trading is wrong / non-operational right now; P1 = trading wrong on
some path / observability lying; P2 = silent contract violation, no PnL
impact today; P3 = boot or deploy hygiene.)

---

## 6. Recommended next actions

In order. Each step is small enough to be its own task.

1. **Re-enable trading correctly (P0 cluster).**
   1. Patch `_pick` in `ml-engine/app/main.py:691-694` to try/except
      `_cached_load`, falling through to the descending-walk and pooled
      fallback when the loader returns `None`/raises.
   2. Strip `news_*`, `llm_*`, `gpt_*`, `sentiment_*`, `ai_*` from
      `FEATURE_COLUMNS` in the trainer; re-train one (coin, tf) slice and
      confirm `/ml/predict` returns 200 with the live booster.
   3. Insert `app_settings.quant_brain_enabled = true` once at least one
      slice is promoted; verify the abstain rate drops in
      `prediction_journal`.
   4. Patch the LLM fallback so it cannot stamp a `direction` without a
      quant prediction (force `direction='stable'`, no `paper_trade`).

2. **Stop the calibration regression (P0).** Investigate why every slice
   is in `structurally_noisy_retire` / `insufficient_sample`:
   class-balance, label leakage, regime mix. Add a regression test that
   asserts `directional_call_share < 0.95` at training time so we don't
   ship another all-call head.

3. **Repair Strategy Lab benchmark (P1).** Re-run `initialDeploy` for
   agent 18 (Buy & Hold) and verify `paper_trades` rows appear; confirm
   `bestBaselineReturn*` and `relativeAlpha*` start moving in the
   meta-brain telemetry. Until then, swap the trend-filter basket proxy
   off Buy & Hold.

4. **Patch auto-deploy fee accounting (P1).** Add `entryFee` to the
   insert at `paper-trader.ts:1687-1699` and debit it from cash. Backfill
   only if needed for downstream reports; brain learning loop will
   self-correct on the next ~30 outcomes.

5. **Re-train pooled with full coin vocab (P1).** Assert
   `len(coin_vocab) > 1` in `verification.py`; treat single-coin pooled
   as a structural fail.

6. **Make reproducibility loud (P2).** Add a not-null check on
   `feature_hash` in the journal writer; backfill historic rows by
   recomputing from the stored payload if possible, else mark
   `feature_hash='unverifiable'` so future scans can quantify it.

7. **Fix the dashboard lies (P2).**
   - Either delete `/quant-calibration-history` and its card, or wire it
     to a real source.
   - Add a 24h directional-accuracy card next to the headline; gate
     "Best Agent" on a 24h directional sample.

8. **Restore the runtime LLM-fields guard (P2).** Replace `mock.module`
   with the vitest equivalent so the Node-24 runtime sister suite goes
   green again. The static guard is good but the runtime payload check
   is what catches Strategy Lab benchmark leakage end-to-end.

9. **Tighten production-replica drift (P3).** Add the equity-resync step
   to the deploy runbook so agents 31/32/39 don't propagate. Pin
   `ML_ENGINE_URL`; populate `ML_ADMIN_TOKEN` and `COINGLASS_API_KEY` in
   env-secrets; add `await fetch /ml/health` retry to
   `training-contract-notifier`.

10. **Re-run this audit after step 1 lands.** Many "Uncertain" items in
    Group 3 (meta-brain learning loop in production), Group 5
    (`record_outcome` round-trip in the live cycle), and Group 7
    (auto-quarantine fire) become observable only once trades start
    flowing again.

---

*Generated 2026-04-23. Raw evidence under `.local/audit/`. No code,
schema, or model writes were performed during this audit.*
