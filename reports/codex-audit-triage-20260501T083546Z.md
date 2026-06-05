# Replit firewall triage of Codex audit `reports/codex-audit-20260501T080738Z.md`

> **Task:** #674 — Replit firewall triage of the Codex audit report.
> **Status:** Triage only. No application code, no DB, no project tasks created. Output is this single new file.

## 0. Inputs and ground rules

- **Upstream audit:** `reports/codex-audit-20260501T080738Z.md` (produced by Task #673 / `.local/tasks/codex-external-audit.md`).
- **Codex CLI:** `codex-cli 0.125.0` against `gpt-5.1-codex-mini`, repo HEAD `4881bb2cd6a78aaf9a8bc0c7a2e5aacffcddd68e`.
- **Coverage caveat carried over from upstream §5:** the upstream run was capped at ~125 s of wall time (Replit shell limit). Codex inspected the brain-promotion gate, runtime-status route, paper-trader DS lane, meta-brain adapter, and brain-state banner. It did NOT cover ml-engine training/registry, calibration code paths, OpenAPI contract surface, dashboard pages outside the banner, or the dataset-refresher path. This triage applies to what Codex actually returned; an additional sweep covering the un-audited surface is recommended (see §11.5).
- **No re-running of Codex.** Source-of-truth for findings is the upstream report's verbatim "Finding 1..5" plus its "Final Summary" sub-bullets.
- **No second AI opinion.** Cross-checks were done by reading the actual source files referenced.
- **No code changes, no deletions, no DB access, no project tasks.** Verified before commit (see §13).

## 1. Triage prompt applied (verbatim, for traceability)

```
Review the Codex audit report against our project hard rules.

Do **not** apply Codex suggestions automatically.

Your job is to act as the firewall between Codex and the production repo.

## Classify every Codex finding as one of:

* `ACCEPT`
* `REJECT`
* `DEFER`
* `NEEDS HUMAN DECISION`

## For every finding, explain:

* whether Codex is technically correct
* whether it violates any hard rule
* whether it risks breaking BTC/5m diagnostic sandbox
* whether it risks deleting experiment evidence
* whether it touches model promotion / gates / quant enablement
* whether it should become a follow-up task
* exact tests/proofs required before implementation

## Hard rules

Reject or escalate anything that violates:

* no synthetic data
* no LLM/news/sentiment/GPT features in trade path
* no gate weakening
* no manual SQL champion promotion
* no manual `quant_brain_enabled` flip
* no real-money path
* no legacy agent resurrection
* no deletion of experiment evidence
* no fee/friction edits
* no dynamic sizing in BTC/5m diagnostic sandbox
* meta-brain shadow only unless explicitly approved
* no global quant activation
* no ETH activation unless separately approved

## Current live diagnostic scope

The only allowed diagnostic trading scope is:

* `bitcoin`
* `5m`
* `C_post_cost`
* beta calibration
* fixed-size paper only
* no dynamic sizing
* meta-brain shadow only
* global quant disabled
* no ETH
* no other coins
* no other timeframes

## Special protection

Do not approve deletion of:

* training reports
* failed verdicts
* calibration manifests
* JSONL histories
* proof packs
* model artifacts
* task reports
* old experiment evidence

unless you can prove they are not referenced and are not needed for audit history.

## Output format

Return a triage table with:

* Codex finding title
* Replit verdict: ACCEPT / REJECT / DEFER / NEEDS HUMAN DECISION
* reason
* hard-rule impact
* BTC/5m sandbox impact
* required tests/proofs
* recommended follow-up task name, if any

## Final output

At the end, provide:

1. accepted safe fixes
2. rejected findings and why
3. deferred findings
4. items requiring my decision
5. proposed follow-up tasks
6. anything that must be checked before touching code

Do not implement anything until I approve the triage.
```

## 2. Per-finding triage

### 2.1 Codex Finding 1 — "Promotion gate fetch timeouts leave `quant_brain_enabled` disabled by default"

| Field | Value |
|---|---|
| **Codex severity** | high |
| **Codex's suggested fix** | "Add retry/backoff or cached 'last known good' history and surface that to the operator so temporary outages don't block manual enable." |
| **Verdict** | **NEEDS HUMAN DECISION** (split: the *retry/backoff* sub-suggestion is potentially ACCEPT-able; the *cached promotion-history* sub-suggestion is REJECT) |
| **Codex technically correct?** | Partially. The single 8 s `AbortController` / no-retry shape on the `/ml/admin/verification-history?limit=1` fetch is real (`artifacts/api-server/src/lib/brain-promotion-gate.ts` lines 125–131, 182–188). But Codex's framing — "kill-switch remains off … delaying diagnostics and confusing operators" — misreads the *intent* of the gate. The gate is **deliberately fail-closed**: a `history_unreachable` response causes the route to refuse the manual enable, which is the entire point of Task #406's remediation in `docs/remediation/2026-04-24-full-system-remediation.md`. Codex treats fail-closed as a bug; the design treats it as the contract. |
| **Hard-rule impact?** | The "cached last known good history" half of the fix would directly violate the **"no manual `quant_brain_enabled` flip without recent passing verification"** rule. A cached promotion record stale by minutes/hours could let an operator enable the brain after a slice was un-promoted, after the verification batch flipped to `verification_status_not_ok`, or after the role file flipped a TF out of `trade`. That is a concrete gate-weakening. The "retry/backoff" half does NOT touch the rule — it only changes how patient the gate is about a transient ML-engine outage. |
| **BTC/5m sandbox impact?** | The promotion gate guards the *global* `quant_brain_enabled` flip, not the BTC/5m diagnostic sandbox specifically. So sandbox scope is not directly leaked. However, Codex's separate Final-Summary item "Promotion gate / role checks do not explicitly constrain timeframe scope to BTC/5m" implies extending the scope check at the route — the gate already exposes `requestedTimeframes` for that (line 89), the route just doesn't pass it. Tightening the route to scope to BTC/5m on enable is a separate decision (see §5 of this triage). |
| **Evidence-deletion impact?** | None. |
| **Promotion / gate / quant-enable touch?** | YES — directly. This is the gate. Any change here touches the most sensitive surface in the repo. |
| **Required tests / proofs before implementing the retry-only sub-suggestion** | (a) Existing `artifacts/api-server/test/brain-promotion-gate.test.ts` and `brain-state-route-integration.test.ts` MUST continue to pass byte-identically. (b) New test: gate returns `history_unreachable` after N retries when fetch keeps failing (i.e., the retry does NOT mask a real outage as "ok"). (c) New test: gate observes the abort signal and does not exceed `timeoutMs * (retries + 1)` total wall time. (d) Verify `brain-flag.test.ts` (operator manual enable / disable / auto-revert paths) still passes. (e) Verify `brain-toggle-removed.test.ts` still passes. |
| **Required tests / proofs to ALLOW the cached-history sub-suggestion** | None acceptable. The cached-history variant fails the gate-weakening rule by construction. |
| **Recommended follow-up task name (if user accepts the retry-only variant)** | "Promotion gate: add bounded retry/backoff to verification-history fetch (no caching)" |

### 2.2 Codex Finding 2 — "`brain/runtime-status` uses prediction journal scan without pagination causing unbounded memory"

| Field | Value |
|---|---|
| **Codex severity** | medium |
| **Codex's suggested fix** | "Add pagination/streaming (e.g., `for await` from cursor) or limit batch size plus early exit once totals are satisfied." |
| **Verdict** | **ACCEPT** |
| **Codex technically correct?** | Yes. `artifacts/api-server/src/routes/crypto/index.ts` lines 3611–3619: `db.select(...).from(predictionJournalTable).where(gte(predictionJournalTable.createdAt, since))` with `since = now − 30 min` and no `.limit(...)`. The aggregation that follows is a single in-memory `for (const r of recent)` loop. Under any prediction surge this can pull an unbounded row count into Node memory. The route is hit at the dashboard's polling cadence, so the load is recurring rather than one-shot. |
| **Hard-rule impact?** | None. The route is a read-only health surface; aggregation correctness is unchanged when paginated. |
| **BTC/5m sandbox impact?** | None. Aggregation is over the prediction journal, not the trade path. |
| **Evidence-deletion impact?** | None. |
| **Promotion / gate / quant-enable touch?** | None. |
| **Required tests / proofs** | (a) New unit test with a stubbed `db` returning ≥ 50 000 synthetic rows, asserting the route returns the same `recentAbstainReasons` rollup as the current implementation (correctness). (b) New test that the route does not call `db.select(...)` more than `ceil(total / pageSize)` times (no accidental N+1). (c) Existing `dashboard-truthfulness.test.ts` and any test referencing `computeBrainRuntimeState` MUST continue to pass (same numeric output for the BrainRuntimeStatePayload). (d) The `state` discriminant logic at lines 3655–3667 — `offline_disabled` / `offline_no_model` / `online` — must produce byte-identical output before and after. |
| **Recommended follow-up task name** | "Paginate `/crypto/brain/runtime-status` prediction-journal scan to bound memory" |

### 2.3 Codex Finding 3 — "Fixed 0.5% sizing in paper-trader is globally applied before family meta filters"

| Field | Value |
|---|---|
| **Codex severity** | medium |
| **Codex's suggested fix** | "Reorder so DS lane sets position size after all multipliers or bypasses the meta/pool/profile clamps entirely when `_isDsLane` is true." |
| **Verdict** | **REJECT** |
| **Codex technically correct?** | No — Codex misread the file. The DS-lane is **already** bypassed for multiplier application AND hard-pinned at the cap stage. Concretely, `artifacts/api-server/src/lib/paper-trader.ts`: |
| | • Line 759–762: `_isDsLane` set; `positionSize = totalValue * fixedPositionPct`. |
| | • Lines 807–869: `compositeSizeMult` is *computed* (meta-brain mult, profile bias, pooled penalty, family mult) but only as a value. |
| | • Line 875: `if (!_isDsLane && executionSizeMultiplier !== 1.0) { positionSize = positionSize * executionSizeMultiplier; }` — the multiplier is **explicitly NOT applied** when `_isDsLane`. |
| | • Lines 884–908: in DS lane, `positionSize = dsPin` (a fresh re-assignment to `totalValue * fixedPositionPct`), bypassing the cash/portfolio caps. The DS lane also throws a hard `DS invariant` error at line 892 if the (coin, timeframe) does not equal `(bitcoin, 5m)`. |
| | • Line 919: `if (!_isDsLane && (totalInvested + positionSize) / p.totalValue >= MAX_PORTFOLIO_AT_RISK)` — DS-lane skipped. |
| | • Line 924: `if (!_isDsLane)` — another DS-lane bypass for risk gating below. |
| | The protection Codex says is missing is in fact present and proven by `artifacts/api-server/test/diagnostic-sandbox-sizing.test.ts` (and the surrounding `mttm-position-size-override.test.ts`, `mttm-diagnostic-sandbox-e2e.test.ts`, `mttm-diagnostic-sandbox-health.test.ts`). |
| **Hard-rule impact** | If we *acted* on Codex's recommendation and "reordered", we would risk reintroducing dynamic sizing into the BTC/5m diagnostic sandbox lane — a direct **violation of "no dynamic sizing in BTC/5m diagnostic sandbox"**. That is the exact rule Task #659 was created to enforce. |
| **BTC/5m sandbox impact** | Acting on this recommendation IS itself the sandbox scope leak. The current code is correct; Codex's suggested patch is the leak. |
| **Evidence-deletion impact** | None. |
| **Promotion / gate / quant-enable touch?** | None. |
| **Required tests / proofs** | None to "implement" — there is nothing to implement. The existing proofs that demonstrate Codex misread:
- `artifacts/api-server/test/diagnostic-sandbox-sizing.test.ts`
- `artifacts/api-server/test/mttm-position-size-override.test.ts`
- `artifacts/api-server/test/mttm-diagnostic-sandbox-e2e.test.ts`
- `artifacts/api-server/test/mttm-diagnostic-sandbox-health.test.ts`
- The runtime invariant throw at `paper-trader.ts:892–896` (`DS invariant: cap-stage reached with (coinId,timeframe); pin is (bitcoin,5m)`).
If we wanted defense-in-depth (NOT the same as the Codex fix), we could ADD a property-based test asserting "for every meta/profile/pooled/family multiplier triple, `_isDsLane === true` ⇒ final positionSize === totalValue * fixedPositionPct", but that is a *strengthening* of the existing pin and is not what Codex asked for. |
| **Recommended follow-up task name** | None (REJECT). If the user wants defense-in-depth, see §5.4. |

### 2.4 Codex Finding 4 — "Meta-brain adapter persists trade→tick bindings without locking, risking concurrent writes"

| Field | Value |
|---|---|
| **Codex severity** | low |
| **Codex's suggested fix** | "Acquire a simple flock/lock file before writing, or persist to temp file and atomically rename while ensuring `hydrateBindings` waits until write completes." |
| **Verdict** | **DEFER** |
| **Codex technically correct?** | Partially. Codex correctly identified the file-write path (`artifacts/api-server/src/lib/meta-brain/adapter.ts:115–139`). However, intra-process concurrent writes are already serialized by the `savePromise = savePromise.then(persistBindings).catch(...)` chain at line 138 — only one `persistBindings` is in flight at a time within a process. The remaining concern is *inter-process* (a second api-server instance writing the same `.cache/meta_brain_state/trade_to_tick.json`), which is not the current deployment shape. Codex did not differentiate. |
| **Hard-rule impact** | None. The file is gitignored runtime state, not experiment evidence. The meta-brain remains shadow-only either way; this affects only the persistence of `trade→tick` bindings used to attribute paper trades to meta-brain ticks for telemetry (no execution authority). |
| **BTC/5m sandbox impact** | None. The bindings are advisory telemetry; sandbox sizing is unaffected. |
| **Evidence-deletion impact** | None. |
| **Promotion / gate / quant-enable touch?** | None. |
| **Required tests / proofs (if user accepts later)** | (a) Existing `artifacts/api-server/test/meta-brain-hydration.test.ts` MUST continue to pass byte-identically (especially the TTL-expiry behavior at adapter lines 153–157 and the `restored` / `expired` log invariant at lines 161–164). (b) New test: simulated concurrent `markDirty()` + `persistBindings()` rapid bursts produce a final on-disk snapshot whose `JSON.parse(...)` round-trips to the in-memory `TRADE_TO_TICK` map (no torn writes). (c) New test: writing to temp + `rename` does not race with `hydrateBindings()` reading the live path (i.e., `hydrateBindings` either sees the old content or the new content, never a half-written file). |
| **Why DEFER, not ACCEPT** | The reported failure mode (corrupted snapshot from inter-process write race) is not currently reachable: there is one api-server process, and intra-process writes are already serialized. The fix (atomic-rename) is benign and trivially safe, but acting on it now is solving a hypothetical. **Precondition to upgrade to ACCEPT:** either (i) a second api-server instance is planned, or (ii) we have observed at least one corrupted `trade_to_tick.json` in the wild. Until then, leave as DEFER. |
| **Recommended follow-up task name (if user wants to upgrade now)** | "Meta-brain adapter: write trade→tick snapshot via temp + atomic rename" |

### 2.5 Codex Finding 5 — "Brain state banner relies on `brainSource` copy that's not updated when banner hides"

| Field | Value |
|---|---|
| **Codex severity** | low |
| **Codex's suggested fix** | "Add explicit fallback that surfaces the raw `brainSource` value along with a note 'source unknown'." |
| **Verdict** | **ACCEPT** |
| **Codex technically correct?** | Partially. The fixed switch with cases `default`/`env`/`auto_revert`/`manual` at `artifacts/crypto-monitor/src/components/brain-state-banner.tsx:43–95` does fall through to a generic "quant_brain_enabled is false" copy if a new source value is added server-side (e.g., `"shadow"` or any future enum). However, Codex understated the existing mitigation: the meta-line at lines 172–174 *already* prints `source: {data.brainSource}` regardless, so the operator does see the raw value. The title/subtitle, though, would not name the new state. So: factually a small UX gap, but not the "operator misled into wrong action" framing Codex implies. |
| **Hard-rule impact** | None. UI-only. |
| **BTC/5m sandbox impact** | None. |
| **Evidence-deletion impact** | None. |
| **Promotion / gate / quant-enable touch?** | None directly — but if a new `brainSource` enum is added server-side AT THE SAME TIME the banner is updated, both must agree. Coordination only. |
| **Required tests / proofs** | (a) Existing snapshots/tests for the banner (search `brain-state-banner` under `artifacts/crypto-monitor/test/` and `artifacts/api-server/test/dashboard-truthfulness.test.ts`) MUST continue to pass for the four known sources `default` / `env` / `auto_revert` / `manual`. (b) New test: render with `brainSource = "some_unrecognized_value"` asserting the title contains "source unknown" and the subtitle contains the literal raw source string. (c) Type-level: tighten `BrainSource` typedef so the OpenAPI schema and the banner agree on the enum, OR explicitly accept `string` and document that fallback. (d) The OpenAPI contract for `/crypto/brain/runtime-status` must NOT silently widen `brainSource` without a contract bump — verify `lib/api-spec/openapi.yaml` is the source of truth. |
| **Recommended follow-up task name** | "Brain-state banner: explicit 'unknown source' fallback that surfaces raw `brainSource`" |

## 3. Final-Summary sub-bullets from Codex (handled separately because they are not standalone "findings" but are surfaced in §7 of the upstream report)

### 3.1 "Legacy `resolveStrategyFamily` name-based fallback in `meta-brain/adapter.ts` now only used for contract tests — candidate for pruning"

| Field | Value |
|---|---|
| **Verdict** | **REJECT** |
| **Codex technically correct?** | No. Codex itself notes that the contract test (`artifacts/api-server/test/meta-brain-family-mapping.test.ts`) still pins this behaviour. The comment at `artifacts/api-server/src/lib/meta-brain/adapter.ts:194–199` (read in §2.4 verification) explicitly says it is "Retained ONLY for the contract test … pins the historical behaviour for any pre-#468 row that has not yet been swept." Pre-#468 rows existing in the prediction journal are real; the fallback IS load-bearing for those. |
| **Hard-rule impact** | Removing it would (a) break a passing contract test and (b) potentially break replay/audit of pre-#468 prediction-journal rows — touching **"no legacy agent resurrection"** indirectly (we'd be DE-resurrecting historical attribution, not reviving a legacy agent, but the audit trail would lose the family attribution for any unswept row). |
| **BTC/5m sandbox impact** | None directly. |
| **Evidence-deletion impact** | Indirect — the family attribution for old prediction-journal rows would no longer resolve. |
| **Recommended follow-up** | None. |

### 3.2 "`brain/runtime-status` returning 'online' despite `quant_brain_enabled=false` once it fails to fetch recent journal rows (false positives)"

| Field | Value |
|---|---|
| **Verdict** | **REJECT** |
| **Codex technically correct?** | No. The route's state machine at `artifacts/api-server/src/routes/crypto/index.ts:3655–3667` is `if (!brainState.enabled) state = "offline_disabled"` BEFORE any consideration of journal contents. A false-positive `online` state when `quant_brain_enabled=false` is not reachable from the current code. Codex misread. |
| **Hard-rule impact** | None — but had we acted on a fix without checking, we'd risk altering the `state` discriminant the dashboard banner relies on. |
| **BTC/5m sandbox impact** | None. |
| **Recommended follow-up** | None. |

### 3.3 "Diagnostic lane sizing logic in `paper-trader.ts` still runs meta-brain/family multipliers after DS branch, risking sizing drift outside fixed sandbox"

This is the same claim as Finding 3. **REJECT** for the same reason (see §2.3). The `!_isDsLane` guard at line 875 plus the dsPin re-assignment at line 908 already enforces the invariant, and the runtime throw at line 892 hard-fails if the (coin, timeframe) is not `(bitcoin, 5m)`.

### 3.4 "Promotion gate / role checks do not explicitly constrain timeframe scope to BTC/5m; if requests include other TFs, gate might inadvertently approve wider live market coverage"

| Field | Value |
|---|---|
| **Verdict** | **NEEDS HUMAN DECISION** |
| **Codex technically correct?** | Partially. The gate **already supports** scoped checks via `options.requestedTimeframes` (`brain-promotion-gate.ts:67–73, 89–119`) — it rejects with `gate_pre_check_failed_by_role` if any requested TF's role is not `trade`. But the route handler (`POST /api/crypto/brain/state`) does not currently pass a `requestedTimeframes` argument because the manual enable is intentionally global (per Task #406's design). So Codex's framing ("might inadvertently approve wider live market coverage") is correct in that the gate is global, but the *design* is global by choice. Constraining the route to BTC/5m on enable is a deliberate scope-narrowing decision, not a bug fix. |
| **Hard-rule impact** | If accepted: tightening to `requestedTimeframes=["5m"]` at the route would re-shape what "enable the brain" means on this workspace from "global enable, role file decides which TFs trade" to "BTC/5m only". That is a policy change that touches the very meaning of the kill switch. Should not be unilateral. |
| **BTC/5m sandbox impact** | This *is* a scope question, but it cuts both ways — narrowing could make the sandbox more invariant; or it could prevent a future TF from being enabled without an explicit route change. |
| **Recommended follow-up** | None until the user answers §5.5. |

## 4. Aggregate verdict tally

| Verdict | Count | Findings |
|---|---|---|
| ACCEPT | 2 | F2 (runtime-status pagination), F5 (banner unknown-source fallback) |
| REJECT | 3 | F3 (DS lane sizing — Codex misread), Final-§3.1 (legacy resolver — load-bearing), Final-§3.2 (false-positive online — Codex misread) |
| DEFER | 1 | F4 (meta-brain adapter atomic write — premature) |
| NEEDS HUMAN DECISION | 2 | F1-split (retry-vs-cache for promotion gate), Final-§3.4 (route-level TF scoping of manual enable) |

## 5. Final summary sections (per task spec ordering)

### 5.1 Accepted safe fixes (with proposed follow-up task names)

1. **F2** — "Paginate `/crypto/brain/runtime-status` prediction-journal scan to bound memory"
2. **F5** — "Brain-state banner: explicit 'unknown source' fallback that surfaces raw `brainSource`"

Both have zero hard-rule impact, zero BTC/5m sandbox impact, zero promotion/gate/quant-enable touch, and zero evidence-deletion impact. Both have well-scoped tests listed in §2.

### 5.2 Rejected findings (one paragraph each)

- **F3 — "DS lane fixed 0.5% sizing globally applied before family meta filters."** Codex misread the file. The DS lane is already (a) the only branch that sets positionSize to the fixed pin at lines 761–762, (b) explicitly skipped by the `!_isDsLane` guard at line 875 when applying `executionSizeMultiplier`, (c) hard-reset to `dsPin` at line 908 after meta computation, (d) protected by a runtime invariant throw at line 892 if the (coin, timeframe) is not `(bitcoin, 5m)`, and (e) covered by `diagnostic-sandbox-sizing.test.ts`, `mttm-diagnostic-sandbox-e2e.test.ts`, `mttm-position-size-override.test.ts`, `mttm-diagnostic-sandbox-health.test.ts`. Acting on Codex's "reorder" recommendation would be the actual sandbox scope leak — it would risk reintroducing dynamic sizing into the BTC/5m DS lane, violating the **"no dynamic sizing in BTC/5m diagnostic sandbox"** hard rule. The current code is correct; the fix is the bug.

- **Final §3.1 — "Legacy `resolveStrategyFamily` name-based fallback is dead code."** The fallback at `meta-brain/adapter.ts:194–...` is documented in-source as load-bearing for any pre-#468 row in the prediction journal that has not been swept, and pinned by `meta-brain-family-mapping.test.ts`. Removing it would (a) break a passing contract test (a REJECT criterion in classification rules), (b) lose family attribution for unswept historical journal rows (audit-trail erosion). **REJECT.**

- **Final §3.2 — "`brain/runtime-status` returning 'online' despite `quant_brain_enabled=false`."** The state machine at `routes/crypto/index.ts:3655–3667` checks `!brainState.enabled` first and unconditionally returns `state="offline_disabled"`, so the false-positive Codex describes is not reachable in the current code. **REJECT** as a misread.

### 5.3 Deferred findings (and what would need to be true to accept later)

- **F4 — "Meta-brain adapter persists `trade→tick` bindings without locking."** Intra-process writes are already serialized via the `savePromise.then(...)` chain. The reported failure mode (inter-process race + half-written file) is not currently reachable in the single-instance api-server deployment. **Precondition to upgrade to ACCEPT:** either (i) a second concurrent api-server instance is planned (multi-replica deployment), OR (ii) at least one corrupted `trade_to_tick.json` is observed in the wild. Until then, atomic-rename is a benign-but-unjustified change.

### 5.4 Items requiring user decision (phrased as concrete questions)

1. **Promotion-gate retry/backoff (F1, retry-only sub-suggestion).** The current behaviour is fail-closed: a transient `/ml/admin/verification-history` outage refuses the manual enable. Codex's suggested fix has two halves; the cached-history half is REJECTED by this triage as a gate-weakening hard-rule violation. The retry/backoff half is a question for you:
   > **Question 1.** "Do you want the promotion gate to retry the verification-history fetch on transient failure (e.g., 3 attempts with exponential backoff, total wall-time ≤ 24 s), keeping the same fail-closed semantics if all retries fail? Or do you prefer to keep the current single-shot behaviour to make 'gate refused' clearly mean 'ML engine is down right now'?"

2. **Route-level TF scoping of manual enable (Final §3.4).** The promotion gate already supports `requestedTimeframes`-scoped role checks; the route deliberately doesn't pass them today, making manual enable a global flip. The current diagnostic scope is BTC/5m only.
   > **Question 2.** "Do you want `POST /api/crypto/brain/state` to pass `requestedTimeframes=["5m"]` (or some explicit per-TF set) to the gate, so that an enable request can ONLY succeed when the requested TF's role is `trade`? This would tighten the kill-switch to the BTC/5m sandbox at the cost of requiring a route change before any future TF can ever be enabled."

3. **F4 deferral re-evaluation (meta-brain adapter atomic write).** Currently DEFER per §5.3.
   > **Question 3.** "Are we planning to run more than one concurrent api-server instance in any environment in the next quarter? If yes, F4 should be upgraded to ACCEPT now and an atomic-rename follow-up created. If no, leave deferred."

4. **Defense-in-depth for DS-lane invariant (related to F3 REJECT, but a strengthening, not Codex's fix).** This is purely operator preference, not Codex-derived.
   > **Question 4.** "Do you want a property-based test added that asserts, for arbitrary (meta multiplier, profile bias, pooled penalty, family multiplier) tuples, the final DS-lane positionSize equals `totalValue * fixedPositionPct` byte-exactly? The runtime invariant throw at `paper-trader.ts:892` plus the existing tests already enforce this; the property test would be belt-and-suspenders."

### 5.5 Proposed follow-up task list (titles only — creation is the user's decision)

1. *(if Q1 = yes)* "Promotion gate: add bounded retry/backoff to verification-history fetch (no caching)"
2. "Paginate `/crypto/brain/runtime-status` prediction-journal scan to bound memory"
3. "Brain-state banner: explicit 'unknown source' fallback that surfaces raw `brainSource`"
4. *(if Q2 = yes)* "Scope `POST /api/crypto/brain/state` enable to BTC/5m via `requestedTimeframes`"
5. *(if Q3 = yes)* "Meta-brain adapter: write `trade_to_tick.json` via temp + atomic rename"
6. *(if Q4 = yes)* "Property test: DS-lane positionSize is invariant under arbitrary meta/profile/pooled/family multipliers"

These titles match the style of existing project tasks (verb-first, scope-named, single concern, no implementation detail leakage). Creating any of them is out of scope for this task per `.local/tasks/task-674.md` Done-looks-like and Out-of-scope clauses.

### 5.6 Pre-flight checklist for any task that touches code touched by accepted findings

Before opening a PR for any follow-up above, the implementing agent MUST verify the following are still passing on `main` AND will still pass after the change:

**For #1 ("Promotion gate retry/backoff"):**
- `artifacts/api-server/test/brain-promotion-gate.test.ts` (every existing assertion, byte-identical reasons)
- `artifacts/api-server/test/brain-state-route-integration.test.ts`
- `artifacts/api-server/test/brain-flag.test.ts`
- `artifacts/api-server/test/brain-toggle-removed.test.ts`
- New required: assert `history_unreachable` IS still returned after retries are exhausted (fail-closed preserved)
- New required: total wall time bounded by `timeoutMs * (retries + 1)`
- The Task #406 remediation note in `docs/remediation/2026-04-24-full-system-remediation.md` MUST be re-read before editing the gate; do NOT add a "cached promotion record" code path under any circumstance — that is the specific gate weakening this triage rejected.

**For #2 ("Paginate runtime-status"):**
- `artifacts/api-server/test/dashboard-truthfulness.test.ts`
- The `BrainRuntimeStatePayload` shape at `routes/crypto/index.ts:3601–3608` MUST NOT change
- The `state` discriminant logic at lines 3655–3667 MUST produce byte-identical output for the same input rows
- New required: ≥ 50 000-row stub returns the same `recentAbstainReasons` rollup as the unpaginated implementation
- New required: page count is `ceil(total / pageSize)`, no N+1
- Verify the `lib/api-spec/openapi.yaml` schema for `BrainRuntimeStatusResponse` is unchanged

**For #3 ("Banner unknown-source fallback"):**
- All existing `brain-state-banner` tests under `artifacts/crypto-monitor/test/` (search before editing)
- `artifacts/api-server/test/dashboard-truthfulness.test.ts`
- The four known sources `default` / `env` / `auto_revert` / `manual` MUST render byte-identical title/subtitle to today
- The OpenAPI schema for `brainSource` in `lib/api-spec/openapi.yaml` MUST be inspected; if widening the enum is required, update the contract first and regen via `pnpm --filter @workspace/api-spec run codegen`

**For #4 ("TF-scoped manual enable") — only if user answers Q2 = yes:**
- `artifacts/api-server/test/brain-promotion-gate.test.ts` (the `gate_pre_check_failed_by_role` path is already covered for direct gate calls; new tests must cover the route-level integration)
- `artifacts/api-server/test/brain-state-route-integration.test.ts`
- The route's existing 409 / 200 / `verification_status_not_ok` / `no_promoted_slices` paths MUST continue to behave identically when `requestedTimeframes=["5m"]` is the only request
- A regression test that the route REFUSES enable when `requestedTimeframes` is omitted by an old client (decide: is omission "global" or "default to ['5m']"? — also a sub-question for the user)

**For #5 ("Meta-brain atomic rename") — only if user answers Q3 = yes:**
- `artifacts/api-server/test/meta-brain-hydration.test.ts` (TTL-expiry behavior at adapter:153–157 and the `restored`/`expired` log invariant)
- The serialization invariant of `savePromise.then(...)` at adapter:138 MUST be preserved
- New required: `hydrateBindings()` reading concurrently with a temp+rename writer either sees the old content or the new content, never a half-written file

**For #6 ("DS-lane property test") — only if user answers Q4 = yes:**
- `artifacts/api-server/test/diagnostic-sandbox-sizing.test.ts`
- `artifacts/api-server/test/mttm-diagnostic-sandbox-e2e.test.ts`
- `artifacts/api-server/test/mttm-position-size-override.test.ts`
- `artifacts/api-server/test/mttm-diagnostic-sandbox-health.test.ts`
- The runtime invariant throw at `paper-trader.ts:892–896` MUST remain in place
- The `_isDsLane` guards at lines 875, 884, 919, 924 (and any others added since) MUST remain in place

**Across all six (sandbox-scope cross-cuts):**
- `artifacts/api-server/test/mttm-universe-guard.test.ts` MUST pass — this is the universe scope guard
- The `diagnosticSandbox.coinId === "bitcoin"` and `diagnosticSandbox.timeframe === "5m"` invariants in `mttm.ts` MUST NOT be touched
- `quant_brain_enabled` MUST NOT be flipped on as part of any of these changes
- Meta-brain MUST remain shadow-only — verify the `clampMetaSizeMultiplier` at `paper-trader.ts:874` and the `if (!_isDsLane && executionSizeMultiplier !== 1.0)` guard at line 875 are unchanged

## 6. Cross-checks performed during this triage (read-only, source-only, no Codex re-run)

For each finding, this triage cross-read the actual file and surrounding context against Codex's claim:

| Codex claim | File checked | Lines read | Misread? |
|---|---|---|---|
| F1 — single-shot fetch, fail-closed | `artifacts/api-server/src/lib/brain-promotion-gate.ts` | 1–192 (full file) | No — Codex correct on shape, but framed fail-closed as a bug |
| F2 — unbounded SELECT over 30 min | `artifacts/api-server/src/routes/crypto/index.ts` | 3592–3699 | No — Codex correct |
| F3 — DS sizing applied before multipliers | `artifacts/api-server/src/lib/paper-trader.ts` | 750–924 | **Yes** — explicit `!_isDsLane` guards and dsPin re-assignment present |
| F4 — concurrent persist races | `artifacts/api-server/src/lib/meta-brain/adapter.ts` | 100–199 | Partial — intra-process serialization missed |
| F5 — banner default copy | `artifacts/crypto-monitor/src/components/brain-state-banner.tsx` | 1–180 (full file) | Partial — meta-line `source: …` already shown |
| Final §3.1 — legacy resolver dead | `artifacts/api-server/src/lib/meta-brain/adapter.ts` | 194–199 (in-source comment) | **Yes** — explicitly retained for contract test + pre-#468 rows |
| Final §3.2 — runtime-status false-positive online | `artifacts/api-server/src/routes/crypto/index.ts` | 3655–3667 | **Yes** — `!brainState.enabled` short-circuit prevents this |
| Final §3.4 — global gate scope | `artifacts/api-server/src/lib/brain-promotion-gate.ts` | 67–73, 89–119 | No — gate supports per-TF, route is intentionally global |

## 7. Hard-rule audit of every finding (compact matrix)

| Finding | Hard rule possibly touched | Could the fix violate it? |
|---|---|---|
| F1 (cache half) | "no manual `quant_brain_enabled` flip without recent passing verification" | **Yes** — REJECT this half |
| F1 (retry half) | none | No |
| F2 | none | No |
| F3 | "no dynamic sizing in BTC/5m diagnostic sandbox" | **Yes if implemented** — REJECT entire finding |
| F4 | none | No |
| F5 | none | No |
| Final §3.1 | "no legacy agent resurrection" (indirectly: removing means de-resurrecting historical attribution); "no deletion of experiment evidence" (indirectly: family attribution lost for old journal rows) | Yes if implemented — REJECT |
| Final §3.2 | none (Codex misread; nothing to implement) | n/a |
| Final §3.4 | scope question — not a hard-rule violation either way, but a deliberate design decision | Decide first |

## 8. Sandbox-scope-leak audit of every finding

| Finding | Could the fix let the sandbox scope leak (other coin / TF / dynamic sizing / meta authority / global quant flip / ETH activation)? |
|---|---|
| F1 | No (route is global today; only narrowing was suggested in §3.4) |
| F2 | No |
| F3 | **YES** — the suggested fix would re-introduce multiplier application to the DS lane, which is dynamic sizing. REJECTED. |
| F4 | No (telemetry-only path) |
| F5 | No |
| Final §3.1 | No |
| Final §3.2 | No (Codex misread) |
| Final §3.4 | No — the proposed change would NARROW scope, not widen it |

## 9. Evidence-deletion audit of every finding

No Codex finding in this audit recommends deleting any of:
- `reports/**`
- `artifacts/ml-engine/reports/**`
- `artifacts/ml-engine/models/**` (gitignored anyway)
- `lib/**/manifests/**`, any `manifest.json`
- Any `*.jsonl` training/calibration/journal/proof file
- Any file under `.local/tasks/**`
- Any file containing "verdict", "proof", "calibration", or "ratchet" in the path

Codex's "Safe cleanup candidates" section explicitly says "None identified within reviewed files; audit rule forbids touching promotion/history evidence." — Codex respected the evidence-protection rule. Final §3.1 (legacy resolver) is a *code* deletion candidate that is REJECTED in §3.1 above for separate reasons (load-bearing for contract test + journal attribution), not for evidence-deletion reasons.

**No deletion of any protected evidence is being approved or recommended by this triage.**

## 10. Items expressly out of scope per `.local/tasks/task-674.md`

This triage did NOT:
- apply any Codex suggestion (no code edits)
- delete any file
- access the database or run any SQL
- create any project task (recommended follow-ups in §5.5 are *names only*, creation is the user's call)
- promote any model or flip `quant_brain_enabled`
- re-run Codex
- consult a second AI opinion
- look outside the upstream Codex report + the repo source for evidence

## 11. Coverage note (carried from upstream §5)

Codex itself flagged in its Top-10: *"Five additional findings need review beyond inspected sections; no new data means not listed."* Files Codex did NOT inspect (per upstream §5):
- `artifacts/ml-engine/app/training/labels_research/` (B/B2/B3/B4 calibration code)
- `artifacts/ml-engine/app/registry/` (full model registry path)
- `lib/api-spec/openapi.yaml` (contract truth)
- The dataset-refresher path (`dataset-refresher` workflow + `_freshness_status.json` / `_freshness_alerts.jsonl` consumers)
- Most of `artifacts/crypto-monitor/src/pages/` outside the brain-state banner
- `.gitignore` and the gitignored model-artifact directory

A repeat audit run in an environment without the 120 s wall-clock cap would close this gap. That is a separate task decision and is NOT created here.

## 12. Header summary for downstream consumers

For agents picking up follow-ups from this triage, the one-screen summary is:

```
Codex audit         : reports/codex-audit-20260501T080738Z.md
Triage              : reports/codex-audit-triage-20260501T083546Z.md (this file)
Findings reviewed   : 5 numbered + 4 final-summary sub-bullets
ACCEPT              : 2 (F2 runtime-status pagination; F5 banner unknown-source)
REJECT              : 3 (F3 DS-lane misread; legacy resolver load-bearing; runtime-status false-positive misread)
DEFER               : 1 (F4 atomic rename — premature)
NEEDS HUMAN DECISION: 2 (F1 retry/backoff; route-level TF scoping)
Hard-rule violations
  approved          : 0
  rejected          : 2 (F1 cache half; F3 entirety)
Evidence deletions
  approved          : 0
  recommended       : 0
DB writes           : 0
Code edits          : 0
quant_brain_enabled : not touched
Sandbox scope       : unchanged (BTC/5m, C_post_cost, beta cal, fixed-size paper, meta shadow)
```

## 13. Pre-commit guardrail verification (recorded for the wrap-up commit)

Per the task's hard guardrails (rules 1–6 of "Hard guardrails"), this triage produces ONLY:

- `reports/codex-audit-triage-20260501T083546Z.md` (this file, new)

Verification will be performed at commit time:
- `git diff --diff-filter=D --name-only` — must be empty (no deletions)
- `git diff --name-only` — must list only this one new file
- No DB script invoked, no SQL executed, no `psql` run, no model-promotion script run
- No project task created via the project_tasks skill

If any of those fail, the task does NOT mark complete.

---

*End of triage. This file recommends; the user decides. Implementation of any ACCEPT or HUMAN-DECISION outcome is the subject of separately-created follow-up tasks, not this one.*
