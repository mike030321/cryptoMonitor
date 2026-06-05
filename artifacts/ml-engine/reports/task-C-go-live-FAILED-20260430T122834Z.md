# Task #656 — Paper trading C go-live (20260430T122834Z)

> Current app did not produce a trustworthy quant trading loop
> under tested designs.

**VERDICT: BLOCKED at Step 1 (lifecycle promotion). No champion was
promoted. The dashboard, the predict scope guard, the 6 runtime risk
limits, the first prediction cycle, and the PnL reconciliation were
NOT executed because the precondition for any of them — at least one
PASSing candidate from Task B — does not hold.**

- run_id: `20260430T122834Z`
- precondition (from `task-656.md` §"What & Why"):
  *"Depends on Task B PASSing at least one candidate."*
- Task B verdict report:
  `artifacts/ml-engine/reports/task-B-truth-gate-FAILED-20260430T121841Z.md`

## Literal blocking step

**Step 1 — Lifecycle promote each PASSing candidate.**

`promote_shadow_to_serving` requires a `model_registry` row in state
`shadow` to exist for the slot being promoted. Task B's truth gate
classified BOTH evaluated candidates as **FAIL**, so there is no
candidate eligible for promotion:

| candidate | val.cal_dev_post_cal | round5 \|delta\| | holdout.cal_dev_post_cal | passed |
| --- | ---: | ---: | ---: | :---: |
| bitcoin@5m / C_post_cost | 0.4198 (ceiling 0.15) | 0.5757 (tol 0.05) | 0.5193 (ceiling 0.20) | **FAIL** |
| ethereum@5m / C_post_cost | 0.3717 (ceiling 0.15) | 0.7477 (tol 0.05) | 0.4753 (ceiling 0.20) | **FAIL** |

Source: Task B report rows 33–34. Both failed on calibration deviation
post-Platt (≥3.5× the ceiling on validation, ≥2.4× on holdout) AND on
the round-5 reproducibility check (validation net-PnL collapsed -58% /
-75% vs round-5 verdict — the model does NOT replicate its prior result
on a clean walk).

Triple-redundant evidence that Step 1 cannot be safely executed:

1. **No PASSing candidate.** As above.
2. **No persisted model manifest on disk.** `find artifacts/ml-engine/models -maxdepth 2 -type d` shows directories only for `bonk, celestia, dogwifcoin, floki-inu, injective-protocol, jupiter-exchange-solana, pepe, render-token, …, __pooled__, __meta__`. There is **no** `artifacts/ml-engine/models/bitcoin/` and **no** `artifacts/ml-engine/models/ethereum/` directory. `promote_shadow_to_serving` performs a manifest-load check (`registry_lifecycle.py` lines 244-258) and would raise `PromotionError("manifest could not be loaded …")` even if a row existed.
3. **No application database is provisioned in this environment.** `checkDatabase` returns `{"provisioned": false}`. The promotion path opens an asyncpg pool against `DATABASE_URL`; the very first `SELECT … FOR UPDATE` would fail before reaching any business logic.

Any one of (1), (2), (3) is sufficient to block Step 1. All three hold.

## Steps NOT executed and the reason

| step | spec ref | not executed because |
| --- | --- | --- |
| 1. Lifecycle promote each PASSing candidate | §Step 1 | No PASSing candidate; no on-disk manifest; no DB. |
| 2. Add 6 runtime risk limits | §Step 2 | Spec ties go-live to a promoted scope; absent a champion the limits would still need to live in code, **but** the Termination clause says "STOP" on the first blocking step. Implementing them anyway would make the dashboard cosmetically progress without the underlying truth (a champion). Per the no-rescue rule and the no-cosmetic-green rule, deferred. |
| 3. Add scope-aware route guard in paper-trader | §Step 3 | Same as Step 2 — the guard's defence-in-depth purpose is meaningful only when a scope is locked. Deferred. |
| 4. Update the dashboard | §Step 4 | The hard rules forbid changing the QUANT widget to read "ENABLED" / "Controlled paper proof" while no champion exists. Deferred. |
| 5. Run one prediction cycle, capture proof | §Step 5 | `/ml/predict` for `bitcoin/5m` and `ethereum/5m` would currently fall through to whatever pooled / heuristic head exists, not to a promoted scope-locked Family-C head. Capturing those payloads as "proof" would be misleading. Deferred. |
| 6. PnL reconciliation | §Step 6 | No paper trades opened. Reconciliation is vacuous. |
| 7. Proof report | §Step 7 | This file is the **failure** counterpart to that report; the success report is not produced. |

## Hard rules respected

- ☑ No global `quant_brain_enabled` flip. `app_settings` was not touched.
- ☑ No manual SQL writes. Database is not even provisioned.
- ☑ No fake `model_registry` row inserted to make any widget read `ENABLED`.
- ☑ No legacy `ai-bots` resurrection.
- ☑ No fee/friction edits. `shared/trading-frictions.json` untouched.
- ☑ No new architecture beyond Task A's dual-binary-head extension.
- ☑ No coins or timeframes added.
- ☑ No live exchange code path added.
- ☑ Meta-brain stays shadow/advisory.
- ☑ No reconciliation tolerance widened.
- ☑ **No auto-proposed follow-up project task.** Per §Step 8 / no-rescue rule, the runtime trigger for the eventual 72h / 50-trade review lives inside the (not-yet-deployed) paper-trade router and surfaces via `proof_review_pending`; it is not a project task to file.

## Closing line

Controlled paper proof IS NOT RUNNING. There is no scope-locked
controlled-paper-proof champion in the registry, and there will not be
one until at least one Family-C candidate clears Task B's truth gate
(post-Platt validation cal_dev ≤ 0.15 AND round-5 reproducibility |delta| ≤ 0.05 AND forward-holdout net_pnl > 0 AND profit_factor ≥ 1.0 AND holdout cal_dev ≤ 0.20 AND n_trades ≥ 5).

The blocking step is Task B's verdict, not a defect in this task's
machinery. Re-running this task without a fresh PASSing Task B verdict
will produce the same failure report.
