# Runtime Truth + Dashboard Repair Audit

**Task:** #532
**Date:** 2026-04-28
**Scope:** crypto-monitor (web) + api-server + ml-engine
**Author:** task-agent
**Status:** Phase 0–7 complete. Brain remains OFF (out of scope per #522).

---

## TL;DR

The dashboard was lying about three things at once: it claimed the brain
was "kill-switched OFF" when in fact `app_settings.quant_brain_enabled`
**has never had a row written** (default OFF, source `default`); it ranked
"HOLD SEI 37 %" as a consensus pick while every agent was emitting
`direction=stable, confidence=0` quant abstains; and it attributed
**$3.24 of 7-day P&L to the autoDeploy fleet** despite all 4 active
executors having zero closed trades — the P&L came from 232 `legacy_archived`
rows that the live aggregation filter forgot to exclude. None of these
panels showed staleness, source, or "n/a" anywhere.

This audit landed:

- a contradiction map (12 items, all root-caused),
- a bottom-up P&L reconcile against `paper_trades`,
- the new `/api/crypto/meta-brain/status` proxy and a `freshness` block on
  `/api/crypto/shadow/metrics`,
- 7 dashboard surfaces re-wired to render honest "n/a" / "no live
  consensus" / "calibration is stale" copy when the underlying signal is
  absent,
- a contract test that locks the autoDeploy filter to non-archived rows,
- before/after screenshots and the full payload corpus.

The brain was **not** brought online — that work belongs to #522.

---

## Six operator questions, answered in 5 seconds on the new dashboard

| # | Question                              | Where on the dashboard                                                                            |
|---|---------------------------------------|---------------------------------------------------------------------------------------------------|
| 1 | Is the brain online?                  | Top status banner: "BRAIN OFFLINE — quant brain has never been enabled" + `source: default` badge |
| 2 | When was the last *real* quant call?  | Activity strip pill: `last quant 5d ago` (distinct from `last exec —`)                            |
| 3 | What did it decide?                   | TopPickCard: "No live consensus" + `suppressedReason: brain_offline` (no fake HOLD)               |
| 4 | What is the live P&L?                 | Quant Fleet card: `$0.00 realized · $0.00 unrealized · 0/4 in profit` (matches DB)                |
| 5 | Are the bots taking abstains?         | Family card: `Abstain rate 100 %` (was 0 %); Live Bot Predictions: "85 abstain hidden" badge      |
| 6 | Is calibration current?               | Quant Live Health: red banner "Calibration is stale. Last live sample 2.9 d ago"                  |

---

## Phase 0 — contradiction map

Full evidence: `.local/truth-audit/00-contradiction-map.md` (12 items
C-1..C-12; each cross-references UI / API / DB / Runtime and assigns a
single source of truth). Headline contradictions:

| Id   | Contradiction                                                                       | Source of truth |
|------|-------------------------------------------------------------------------------------|-----------------|
| C-1  | Activity strip "0 executor / 10 baseline" vs Quant Fleet "4"                        | DB (4 active executors, 0 trades)        |
| C-1b | "last exec 4 h ago" vs no real quant decision in 5d                                 | predictions.reasoning rollup             |
| C-2  | TopPickCard "HOLD SEI 37 %" vs every agent emitting quant_disabled abstains          | brain-flag (offline → suppress)          |
| C-3  | "Kill-switch OFF" copy vs `app_settings.quant_brain_enabled` row never created       | brain-flag.source = default              |
| C-4  | Live Bot Predictions feed showed quant_disabled rows as "ABSTAIN" trades             | predictions.reasoning                    |
| C-5  | autoDeploy 7d P&L $3.24 / 232 trades vs active executors 0/0                         | paper_trades joined to live agents       |
| C-6  | Family abstain rate 0 % vs 100 % of last-hour predictions are abstains              | predictions table (not skip-tracker)     |
| C-7  | best/worst agent ranks dominated by quant_disabled abstains (88 % "accuracy")        | exclude abstains + ≥30 sample floor      |
| C-8  | Quant Live Health rows treated as live; max(model_predictions.created_at) 3 d stale  | freshness.lastSampleAt                   |
| C-9  | "1d 31 %" directional accuracy reported as if current                                | freshness.isStale gate                   |
| C-10 | Diagnostics MetaBrain card empty (read wrong endpoint)                               | new /meta-brain/status proxy             |
| C-11 | No surface for ml-engine /meta-brain/health · /stats · /last_replay                  | proxy + MetaBrainStatusCard              |
| C-12 | brainSource ∈ {default, manual, auto_revert, env} not surfaced anywhere              | BrainStateBanner copy switched on source |

---

## Phase 4 — P&L bottom-up reconcile

`.local/truth-audit/final/pnl_reconcile.csv` (computed straight from
`paper_trades` + `paper_position_marks` + `paper_portfolios`):

```
class                                , closed_7d, realized_7d_usd, closed_24h, realized_24h_usd, open, deployed_open_usd
autodeploy_buggy_filter (current code),       232,          3.243 ,          0,             0   ,    0,          0
autodeploy_fixed_filter (proposed)    ,         0,          0     ,          0,             0   ,    0,          0
baseline_active                       ,         0,          0     ,          0,             0   ,   60,       1155.495
executor_active                       ,         0,          0     ,          0,             0   ,    0,          0
legacy_archived                       ,       264,        -43.664 ,          0,             0   ,    0,          0
```

**Root cause.** `getAutoDeployAttribution` filtered only on
`strategy_type='ai-bots'`, which silently included 20 `legacy_archived`
agents whose `strategy_type` was never reset during the registry sweep.
The buggy filter happened to surface only the *positive* tail of the
legacy P&L (selection bias from a date filter further down the function)
which is why operators saw a misleadingly *good* number. The fixed
filter adds `archived_at IS NULL AND profile_id <> 'legacy_archived'` and
correctly returns 0/0 — matching the truth that the 4 live executors
have no closed trades.

Fix: `artifacts/api-server/src/lib/paper-trader.ts:getAutoDeployAttribution`.
Locked in by `artifacts/api-server/test/auto-deploy-attribution-contract.test.ts`
(3 assertions, all passing).

---

## Phase 7 — dashboard repairs (A–H)

| Id | Surface                          | Change                                                                                                                          |
|----|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| A  | `BrainStateBanner`               | Source-aware copy switched on `brainSource`; "default" now reads "quant brain has never been enabled" (not "kill-switch OFF")  |
| B  | `ActivityBanner`                 | New `last quant Nd ago` pill from `lastValidQuantDecisionAt` (predictions where `reasoning NOT ILIKE '%quant_disabled%'`)       |
| C  | `RecentPredictions`              | Hides quant_disabled / quant_abstain rows; shows `N abstain hidden` badge so the count is still visible                         |
| D  | `TopPickCard`                    | Renders "No live consensus" when `bestPick.suppressedReason === "brain_offline"`; backend short-circuits before the cache       |
| E  | `/agents/families` abstainRate   | Now `MAX(skip-tracker, predictions-table)` per family — skip-tracker alone reports 0 % when the orchestrator abstains upstream  |
| F  | `/dashboard` best/worst agent    | Excludes `quant_disabled='true'` predictions and requires ≥30 sample; falls back to `id:0/N:"N/A"` when insufficient signal     |
| G  | Quant Live Health page           | Red staleness banner from new `freshness` block (`lastSampleAt`, `ageMinutes`, `staleAfterMinutes=60`, `isStale`)                |
| H  | New `/meta-brain/status` proxy   | Fans out to ml-engine `/health` `/stats` `/last_replay` in parallel; `MetaBrainStatusCard` renders learning state on /diagnostics |

---

## Backend changes

| File                                                  | Change                                                                                                  |
|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `src/lib/paper-trader.ts`                             | autoDeploy filter excludes archived/legacy rows; added missing `sql` import                              |
| `src/routes/crypto/index.ts` `/best-pick`             | Brain-offline short-circuit with `suppressedReason: "brain_offline"` instead of stale cached pick        |
| `src/routes/crypto/index.ts` `/dashboard-activity`    | Adds `lastValidQuantDecisionAt` from predictions table                                                   |
| `src/routes/crypto/index.ts` `/dashboard` best/worst  | Excludes quant_disabled rows + ≥30 sample floor + `id:0/N:"N/A"` fallback when insufficient_signal       |
| `src/routes/crypto/index.ts` `/agents/families`       | Family abstainRate now `MAX(skip-based, predictions-based)` — see C-6                                   |
| `src/routes/crypto/index.ts` `/shadow/metrics`        | Adds `freshness {lastSampleAt, ageMinutes, staleAfterMinutes, isStale}`                                  |
| `src/routes/crypto/index.ts` `/meta-brain/status`     | New endpoint proxying ml-engine learning-state for the diagnostics card                                  |

## Frontend changes

| File                                                  | Change                                                                                          |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `components/brain-state-banner.tsx`                   | Source-aware copy (`default`/`manual`/`auto_revert`/`env`)                                       |
| `components/activity-banner.tsx`                      | Adds `last quant Nd ago` pill                                                                    |
| `components/meta-brain-status-card.tsx` (NEW)         | Reads `/meta-brain/status`; renders pill + learning state + last replay                          |
| `pages/dashboard.tsx`                                 | TopPickCard suppression card; RecentPredictions abstain filter                                   |
| `pages/quant-shadow.tsx`                              | Staleness banner when `freshness.isStale`                                                        |
| `pages/diagnostics.tsx`                               | Wires `<MetaBrainStatusCard/>` into the Meta-gate section                                        |
| `hooks/use-news.ts`                                   | Type updates: `BestPick.suppressedReason`, `DashboardActivity.lastValidQuantDecisionAt`          |

---

## Tests

`artifacts/api-server/test/auto-deploy-attribution-contract.test.ts` —
**new**, 3 assertions, all green:

```
▶ Task #532 — autoDeploy attribution contract
  ✔ live ai-bots filter excludes legacy_archived rows
  ✔ live ai-bots filter is a strict subset of strategy_type='ai-bots'
  ✔ any legacy_archived row that is also strategy_type='ai-bots' is excluded by the live filter
✔ Task #532 — autoDeploy attribution contract
ℹ pass 3 / fail 0
```

This test runs against the live workspace DB and will fail loudly if a
future filter change re-introduces archived rows into the autoDeploy
aggregation.

---

## Live verification (curl against `localhost:8080`)

| Endpoint                              | Status | Notable field                                                                                       |
|---------------------------------------|--------|-----------------------------------------------------------------------------------------------------|
| `GET /api/crypto/best-pick`           | 200    | `suppressedReason: "brain_offline"`, `coinName: "No live consensus"`                                |
| `GET /api/crypto/meta-brain/status`   | 200    | `available: true`, `reachable: true`, `stats.trust_by_family` populated                              |
| `GET /api/crypto/shadow/metrics`      | 200    | `freshness.lastSampleAt: "2026-04-25T15:25:02.316Z"`, `ageMinutes: 4155`, `isStale: true`            |
| `GET /api/crypto/dashboard-activity`  | 200    | `lastValidQuantDecisionAt: "2026-04-23T15:18:12.065Z"`, `lastExecutorTradeAt: null`                  |
| `GET /api/crypto/agents/families`     | 200    | OK                                                                                                  |
| `GET /api/crypto/reality-check`       | 200    | OK (was 500 in the regressed state — unblocked by the `sql` import fix)                              |
| `GET /api/crypto/dashboard`           | 200    | OK                                                                                                  |

---

## Proof pack inventory (`.local/truth-audit/`)

```
00-contradiction-map.md                                # Phase 0
final/pnl_reconcile.csv                                # Phase 4
final/dashboard_payloads/                              # raw "before" capture (15 endpoints)
final/dashboard_payloads/after/                        # raw "after" capture (10 endpoints)
final/screenshots/dashboard_after.jpg                  # Command Center
final/screenshots/quant_shadow_after.jpg               # Calibration is stale banner
final/screenshots/diagnostics_after.jpg                # Brain Diagnostics (Meta gate is below the fold)
```

---

## What was deliberately left undone

- **Brain not brought online.** Per the constraint, #522 owns the
  decision to flip `quant_brain_enabled=true`. This audit only repairs
  honesty about the current state.
- **Legacy `legacy_archived` rows not deleted.** Per the constraint,
  data is preserved; the fix is at the *aggregation filter* layer.
- **Phase-3-backtest baselines** on `/shadow/vs-backtest` were not
  re-touched — they already had the correct staleness story.
- **Brain-runtime banner copy** for the rare `auto_revert` source is
  written but cannot be visually verified on this workspace because the
  current source is `default`.

---

## Follow-ups (proposed for the agent inbox)

1. Bring the brain online once the verification gate has promoted at
   least one slice (#522, blocked on this audit landing first).
2. Schedule a weekly job that asserts the contract test is still green
   in CI so a future schema sweep cannot silently regress.
3. Once #522 is in, snapshot the dashboard again and compare against the
   "after" pack here — the staleness banner, abstain badge, and
   "No live consensus" card should all flip off without any code change.

---

## Rev 2 — Code-review rejection follow-ups (same day)

The first review rejected the audit on four concrete blockers. Each has
been closed; the corresponding source pointers are listed below so a
reviewer can re-check without re-reading the whole index file.

| # | Blocker (review wording)                                                                                          | Resolution                                                                                                                                                                                                                                                                                                                                                                  |
|---|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `/best-pick` only consulted `brainState.enabled`; ignored ml-engine reachability + model presence                 | Extracted `computeBrainRuntimeState()` helper in `src/routes/crypto/index.ts`. `/best-pick` and `/brain/runtime-status` both call it. Best-pick now emits one of `suppressedReason: "brain_offline" \| "brain_offline_no_model" \| "brain_status_unknown"` and surfaces `brainRuntimeState` (`offline_disabled`, `offline_no_model`, `online`, `unknown`). Verified by curl. |
| 2 | Dashboard returned a fake `bestAgent`/`worstAgent` placeholder when there were no signals                          | `/dashboard` now returns `bestAgent: null`, `worstAgent: null`, plus a new `signal: "ok" \| "insufficient_signal"` discriminator. `lib/api-zod` schema updated (`.nullable()` + new optional `signal` enum). Frontend ignores them when `null`; no UI consumer was relying on the legacy fake-zero shape (verified via ripgrep).                                              |
| 3 | Meta-Brain card omitted last-evaluate, last-record-outcome, 24 h trust-state changes, last directive, zero-activity note | `/api/crypto/meta-brain/status` proxy adds a `learningTruth` block with `lastEvaluateAt`, `lastRecordOutcomeAt`, `trustStateChanges24h`, `closedTrades24h`, `lastDirective`, and an explicit `activityNote` ("Zero meta-brain learning events because no live trades closed…"). `MetaBrainStatusCard` renders all five fields with `n/a` fallback when the proxy returned `null`.       |
| 4 | Proof pack missing the four operator artefacts                                                                    | All four written to `.local/truth-audit/final/`: `runtime_state.txt` (concatenated probe transcript), `executor_counts.csv` (baseline / executor / legacy_archived counts), `family_metrics.csv` (per-profile abstain rate over the last 1 h), `meta_brain_learning_state.json` (ml-engine `stats` + `last_replay` + the api-server proxy response).                          |

### Verification (post-Rev-2)

```text
$ curl -s /api/crypto/best-pick           # suppressedReason="brain_offline", brainRuntimeState="offline_disabled"
$ curl -s /api/crypto/dashboard           # bestAgent=null, worstAgent=null, signal="insufficient_signal"
$ curl -s /api/crypto/meta-brain/status   # learningTruth.{lastEvaluateAt,lastRecordOutcomeAt}=null,
                                          #               trustStateChanges24h=0, closedTrades24h=0,
                                          #               activityNote="Zero meta-brain learning events…"
$ pnpm --filter @workspace/crypto-monitor exec tsc --noEmit   # clean
```

The brain remains OFF by design — none of the Rev 2 changes alter
runtime behaviour, they only make the existing zero state legible.

### Rev 2.1 — second-round review follow-ups

The Rev 2 review correctly flagged that the frontend was still keying
the suppression card on the literal `"brain_offline"`, so the two
other suppression states (`brain_offline_no_model`,
`brain_status_unknown`) would have fallen through to the recommendation
layout (rendering `HOLD ` with an empty coin link). Closed by:

- **`TopPickCard` (artifacts/crypto-monitor/src/pages/dashboard.tsx)** —
  the suppression branch now triggers on the truthiness of
  `pick.suppressedReason`, with per-reason copy for each documented
  value and a safe fallback for any future backend-side addition.
  Also surfaces `brainRuntimeState` in the diagnostic footer line.
- **`BestPick.suppressedReason` (artifacts/crypto-monitor/src/hooks/use-news.ts)** —
  union widened to `"brain_offline" | "brain_offline_no_model" |
  "brain_status_unknown" | (string & {}) | null`. `brainRuntimeState`
  added with the same `(string & {})` escape hatch so a future enum
  value cannot silently break TS exhaustiveness or the suppression
  check.
- **New contract test
  (`artifacts/api-server/test/best-pick-suppression-contract.test.ts`)**
  — 5 assertions:
  1. every literal `suppressedReason` emitted by the route is in the
     documented set,
  2. the dynamic `reasonByState` lookup table only maps to documented
     values,
  3. every `brainRuntimeState` literal is in the documented set,
  4. **every suppressed-payload site emits `coinId: ""`,
     `coinName: "No live consensus"`, `action: "hold"`, and
     `brain: null`** — locking the no-coin-link / no-recommendation
     contract,
  5. the frontend `BestPick.suppressedReason` union accepts every
     backend value (TS exhaustiveness guard).

  All 5 pass; the existing 3 autoDeploy attribution assertions still
  pass — combined `pnpm exec node --test` exits 0 with 8/8 green.

Pre-existing tsc errors in `routes/crypto/index.ts` near lines 2627 /
2689 / 2706 (drizzle schema `profileId` typing) are untouched by this
audit — they predate Task #532 and the api-server runs correctly. They
are tracked separately and out of scope for the truth audit.

### Rev 2.2 — APPROVED-WITH-COMMENTS follow-ups

The Rev 2.1 review approved the diff with three non-blocking comments.
All three are now closed:

1. **Banner copy normalised to "QUANT DISABLED — …" variants.**
   `BrainStateBanner` (`artifacts/crypto-monitor/src/components/brain-state-banner.tsx`)
   was leading with the generic phrase "BRAIN OFFLINE" for every
   disabled state. Per the explicit task contract, this is now
   "QUANT DISABLED — …" across all five branches (`default`, `env`,
   `auto_revert`, `manual`, fallback) plus the `NO_MODEL_COPY`
   constant. The phrase is more specific to what is actually OFF
   (the quant brain) and removes the operator ambiguity around
   "offline = unreachable".
2. **`getAutoDeployAttribution()` filter now also enforces `isActive=true`.**
   `artifacts/api-server/src/lib/paper-trader.ts` was filtering on
   `strategyType='ai-bots'`, `archivedAt IS NULL`, and
   `profileId <> 'legacy_archived'` — but a paused-but-not-archived
   executor would still leak its closed trades into the live
   attribution surface. The new `eq(agentsTable.isActive, true)`
   clause closes that drift path. The contract test
   (`auto-deploy-attribution-contract.test.ts`) was updated in
   lock-step so the filter and the test assert the same set; all
   3/3 assertions still pass.
3. **`MetaBrainStatusCard` replay extraction reads the nested
   ml-engine shape.** ml-engine `/ml/meta-brain/last_replay` returns
   `{ ok, last_run: {...}, last_committed_run: {...} }`. The card
   was reading `lastReplay.ts/outcome/reason` at root, which would
   render `n/a` even when data existed. Now prefers
   `last_committed_run` (most recent promotion), falls back to
   `last_run` (most recent attempt — may have been gated by
   `thresholds_not_met`), and only then to the legacy flat shape
   for forward/backward compat. The "outcome" is derived from the
   explicit `commit_details.promoted` / `last_run.commit` boolean
   flags so the badge correctly shows `promoted` vs
   `thresholds_not_met` instead of "n/a".

Combined contract suite still 8/8 green; `tsc --noEmit` on
`crypto-monitor` clean. Brain remains OFF.

## Revision 2.3 — leaderboard streak/score honesty + meta-brain
nullable-on-failure + dashboard suppression integration test

The Rev 2.2 architect review approved the work but flagged three
remaining honesty leaks:

1. **best/worst leaderboard payloads still carried abstain-era
   `streak` / `streakType` / `score`.** The route was correctly
   recomputing `accuracy` from the non-abstain dataset for the
   ranking, but the agents-row badges (`streak`, `score`) were
   passing through unmodified — a "12-day streak" inflated by the
   pre-Task-#522 era when every quant-disabled tick was scored as a
   "correct stable" call. Rev 2.3 introduces
   `cleanRankingPayload()` in `routes/crypto/index.ts` (lines 633-
   638) that null-strips those three fields before emitting
   `bestAgent` / `worstAgent`. The full `agentRankings` list still
   carries the raw values for the per-agent detail view, which is
   honest about its all-time scope.

2. **`/crypto/meta-brain/status` numeric counters were 0 on probe
   failure instead of `null`.** `closedTrades24h` defaulted to `0`
   in the DB-query catch path, and `trustStateChanges24h` started
   at `0` even when the upstream `/ml/meta-brain/stats` probe
   failed. Both now flow `number | null` so the dashboard's
   "n/a" branch fires instead of falsely claiming "0 in 24h"
   when the source is genuinely unknown:
   - `closedTrades24h` (line 3342): typed `number | null`,
     `catch { closedTrades24h = null; }`.
   - `trustStateChanges24h` (line 3403): initialised to `null`
     when `stats === null`, otherwise `0` (then incremented as
     normal).
   The frontend `MetaBrainStatusCard` already renders `n/a` for
   nullable numeric fields, so no UI work was required.

3. **Added `/dashboard` insufficient-signal integration test.**
   `artifacts/api-server/test/dashboard-leaderboard-suppression-contract.test.ts`
   covers the full contract:
   - **Source-static**: the route declares the `dashboardSignal`
     discriminator with the documented union, and pipes
     `bestAgent` / `worstAgent` through `cleanRankingPayload`
     (which null-strips streak/streakType/score).
   - **Runtime**: hits the live `/api/crypto/dashboard` and
     asserts `bestAgent === null`, `worstAgent === null`,
     `signal === "insufficient_signal"` in the current
     brain-offline state — the literal end-to-end guarantee.
   - **Meta-brain nullable**: source-asserts that
     `closedTrades24h` and `trustStateChanges24h` are typed
     `number | null` and follow the no-fake-zero contract.

Combined contract suite is now 13/13 green
(3 autoDeploy + 5 best-pick + 5 dashboard/meta-brain). Brain
remains OFF by design (Task #522 separation preserved).
