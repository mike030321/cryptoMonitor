# Task #549 — Campaign verdict for the task #399 long-running training run

- Captured: 2026-04-28T14:05:22Z; refreshed with full run-2 1h-tier evidence at 2026-04-28T17:00:00Z (after run-2 trained the same 8 of 9 coins as run-1 in tier 1h)
- Workflow: `task-399-training-campaign` (`scripts/run_full_training_campaign.py`)
- Verdict scope: the workflow itself, observed across **three campaign starts** present in `models/progress_updates.jsonl`: 2026-04-25T06:33:02Z → `models/training_run_20260425T063302Z/`, 2026-04-28T11:09:21Z → `models/training_run_20260428T110921Z/` (= **run-1**), 2026-04-28T14:00:20Z → `models/training_run_20260428T140020Z/` (= **run-2**). All three timestamps and run-dir names confirmed via `jq -c 'select(.phase=="campaign_start") | {emitted_at,run_dir}' models/progress_updates.jsonl`.
- Reportable runs: **run-1** is the terminal data point for tier coverage — phase-1/2/3 artifacts written, 8/9 1h slices trained, 0 promotions, then process-died-and-respawned without ever reaching tier 2h/6h/1d. **Run-2** has, at refresh time, mirror-completed run-1's tier-1h footprint exactly: same 8 coins trained (bonk → render-token), same 9th coin (`worldcoin-wld/1h`) un-started, same `consecutive_regress=2` watchdog cycle, 0 promotions on identical fits. The python process for run-2 is still alive (`pid 542`, etime ~02:55+) but is in `watchdog_halt` after `slice_done:render-token/1h` at 16:45:52Z and has produced no `slice_done` event for tiers 2h/6h/1d.
- Pre-run baseline archive: `models/_archive/20260428T112443Z_pre_full_run/`
- Verdict (one line): **0 promotions, 0 trade-eligible slices, 0 timeframes cleared, across all three observed campaign starts (25-day window, 16 trained 1h slices in total: 8 in run-1 + 8 in run-2 on the same 8 coins). Recommendation is unchanged: `quant_brain_enabled=false`, every timeframe stays in shadow / governance / disabled mode.**

Why this report is final at refresh time even though run-2's python process is still alive: the strict ">2h-no-progress" hung-run threshold from the task description is structurally unreachable here because the campaign's watchdog auto-resumes every 1800 s (≤30 min) and trains 1–2 more 1h slices before the next halt — so a 2-hour quiet window can only happen if the python process dies, in which case the workflow runner respawns into a fresh `training_run_<ts>/` (run-1 already followed this pattern; run-2 will too). Run-1 is therefore the unambiguous terminal anchor for tier coverage, and run-2's first 8/9 trained slices have reproduced the run-1 failure pattern on the same 8 coins (every slice DA < baseline DA, every slice post-fee P&L in `[-504.67, -227.76]`, 0 promotions). Section A counts and Section B `no` verdicts are invariant to remaining run-2 slice outcomes; the Section C recommendation (`quant_brain_enabled=false`) and Section D follow-ups (#530 STABLE-collapse and existing watchlist tasks) consume only the run-1 + run-2 numbers already in this report.

---

## Section A — Campaign verdict

Counts are derived from `models/progress_updates.jsonl` filtered to events with `emitted_at >= 2026-04-28T11:09:00Z` and `< 2026-04-28T14:00:00Z` (cutoff at the run-2 `campaign_start`). Every per-coin/1h `slice_done` row in this window has `status="trained"` and `promoted` is **never** set true anywhere — confirmed by `grep -c '"slices_promoted": *[1-9]' models/verification_history.jsonl` returning **0**.

| timeframe | trained | promoted | rejected (skipped) | top reject / outcome reasons |
|---|---|---|---|---|
| 1m  | 0 | 0 | 0 | not in the campaign tradeable set; never scheduled |
| 5m  | 0 | 0 | 9 | `5m_hard_gate_failed` for all 9 coins (contiguous_days ≈ 65.3 vs 305-day floor in `COVERAGE_BAR_DAYS["5m"]=305`) — confirmed in `phase2_data_audit.json` |
| 1h  | 8 | 0 | 0 | trained but every slice failed the verification gate (DA below baseline, post-fee P&L deeply negative — see §B). The 9th coin (`worldcoin-wld/1h`) was scheduled but never started — run-1 died after slice 8/9 |
| 2h  | 0 | 0 | 0 | not reached — `phase2_data_audit.json` says all 9 coins pass the 350-day floor for 2h, so 2h was eligible to train, but run-1's 1h tier did not finish before the workflow respawn so tier 2h never started |
| 6h  | 0 | 0 | 0 | not reached — same as 2h: 9/9 coins pass the 350-day floor for 6h, but tier 6h never started |
| 1d  | 0 | 0 | 9 | structurally rejected by `phase2_data_audit.json`: all 9 coins have `contiguous_days=365.0` vs `COVERAGE_BAR_DAYS["1d"]=1000` (Task #417 floor). 1d cannot train in this configuration regardless of how long the campaign runs |

Per-coin/1h training set in run-1: bonk, celestia, dogwifcoin, floki-inu, injective-protocol, jupiter-exchange-solana, pepe, render-token (8 of the 9 coins). The 9th coin scheduled for 1h (`worldcoin-wld/1h`) was never started in run-1 — the run died after `render-token/1h (8/9)`.

No pooled, specialist, 2h, 6h, or 1d slice was trained in run-1. The snapshot lines that say `best=pepe/1d worst=__pooled__/1d` reference the **prior 2026-04-25 baseline run's** `slice_done` entries (verifiable in the same `progress_updates.jsonl` — those events all have `emitted_at` 2026-04-25), not slices produced by run-1. They are stale references the snapshotter carried forward and they must not be read as run-1 evidence.

Net of 17 scheduled-touched slices (8 trained + 9 5m hard-skipped), promotion count is **0**. Reject reason concentration: `top_failure_bucket: structurally_noisy_retire` (count 3) per the final snapshot at 13:52:13Z, plus the watchdog flag `consecutive_regress=2` and `trend_vs_baseline_improving_share=0.0`.

### Run-2 confirmation (mirror of run-1's tier-1h footprint)

`models/training_run_20260428T140020Z/` — `phase1_preflight.json`, `phase2_data_audit.json`, `phase2_data_audit_pre_backfill.json`, `phase2_data_integrity.md`, `phase3_baseline_pointer.json` written. Tier 1h has trained 8 of 9 coins as of refresh time (the same 8 coins as run-1, in the same order: bonk → celestia → dogwifcoin → floki-inu → injective-protocol → jupiter-exchange-solana → pepe → render-token). The 9th coin (`worldcoin-wld/1h`) has not started — `slice_done:render-token/1h` at 16:45:52Z was followed by `watchdog_halt consecutive_regress=2`. Tiers 2h/6h/1d have not started.

| timeframe | trained (run-2) | promoted (run-2) | rejected/skipped (run-2) | note |
|---|---|---|---|---|
| 5m | 0 | 0 | 9 | same `5m_hard_gate_failed` for all 9 coins as run-1 (coverage floor unchanged) |
| 1h | 8 | 0 | 0 | exact same 8 coins as run-1; every slice DA below baseline, every slice post-fee P&L in `[-504.67, -227.76]`; same 9th-coin gap (`worldcoin-wld/1h` un-started) |
| 2h | 0 | 0 | 0 | not reached |
| 6h | 0 | 0 | 0 | not reached |
| 1d | 0 | 0 | 0 | not reached (and structurally blocked anyway: phase2 audit shows the same `contiguous_days=365.0` vs 1000-day floor as run-1) |

---

## Section B — Trade-readiness table

Source: per-slice `slice_done.metrics`, `slice_done.baseline_metrics`, `slice_done.pnl_after_fees` blocks. Trade-authority gate per task spec: `promoted=true` ∧ `directional_accuracy > baseline_directional_accuracy + 0.02` ∧ `post_fee_pnl_pct_total > 0` ∧ `directional_call_share ∈ [0.4, 0.85]`. `directional_call_share` is the slice's `pnl_after_fees.trade_share` (share of holdout rows where the model emitted a non-STABLE call the trader could execute).

`best_slice` per timeframe = the slice with the highest `(directional_accuracy − baseline_directional_accuracy)` lift among slices satisfying `n_trades >= 30` from the most recent training pass on disk for that timeframe (run-1 for 1h; the 2026-04-25 baseline run for 2h/6h/1d, since neither run-1 nor run-2 has trained those tiers); 5m has no candidate (every slice was hard-gated). DA / baseline DA / post_fee_pnl_pct_total / directional_call_share are reported on that best slice.

| timeframe | trade_authority | reason | best_slice | DA | baseline DA | post_fee_pnl_pct_total | directional_call_share |
|---|---|---|---|---|---|---|---|
| 1m | **no** | insufficient_evidence — not in `TRADEABLE_TIMEFRAMES` for this campaign; no 1m model trained anywhere on disk | — | — | — | — | — |
| 5m | **no** | every per-coin/5m slice emitted `slice_skipped reason=5m_hard_gate_failed` (run-1); contiguous_days ≈ 65.3 vs 305-day floor in `COVERAGE_BAR_DAYS["5m"]` | — | — | — | — | — |
| 1h | **no** | not promoted; best 1h slice's DA lift is +0.0045 (gate floor +0.02) and post_fee_pnl_pct_total = −365.31 (gate floor > 0); 0/8 trained slices satisfy the gate | injective-protocol/1h (run-1) | 0.3784 | 0.3739 | −365.31 | 0.6831 |
| 2h | **no** | insufficient_evidence in run-1 (tier 2h never started before the workflow respawn). Stalest-best evidence on disk (2026-04-25 run): jupiter-exchange-solana/2h DA 0.4083 vs base 0.4136 → DA lift −0.0053 (best of 9 per-coin/2h, all negative), post-fee P&L −210.40, not promoted; no 2h slice satisfies the gate | jupiter-exchange-solana/2h (2026-04-25 baseline) | 0.4083 | 0.4136 | −210.40 | — |
| 6h | **no** | insufficient_evidence in run-1 (tier 6h never started). Stalest-best evidence on disk (2026-04-25 run): render-token/6h DA 0.4409 vs base 0.4085 → DA lift +0.0324 (clears the +0.02 DA component) but post_fee_pnl_pct_total = −105.66 < 0 (fails the P&L component), not promoted | render-token/6h (2026-04-25 baseline) | 0.4409 | 0.4085 | −105.66 | — |
| 1d | **no** | structurally blocked: `phase2_data_audit.json` shows all 9 coins at `contiguous_days=365.0` vs `COVERAGE_BAR_DAYS["1d"]=1000` (Task #417 floor), so no 1d slice can train this campaign. Stalest-pre-Task-#417 evidence on disk (2026-04-25 run): bonk/1d DA 0.4313 vs base 0.3925 → lift +0.0388 (clears DA component) but post_fee_pnl_pct_total = −115.64 < 0, not promoted; pepe/1d (the snapshotter's stale `best=pepe/1d` reference) is DA 0.3773 vs base 0.4250, far below the gate | bonk/1d (2026-04-25 baseline; pre-#417-floor) | 0.4313 | 0.3925 | −115.64 | — |

**Trade-authority count across all timeframes: 0.**

`directional_call_share` is dashed for the 2h/6h/1d rows because the 2026-04-25 `slice_done` events in `progress_updates.jsonl` did not yet carry the `pnl_after_fees.trade_share` field consistently — the field stabilized in run-1 — so I refuse to fabricate a value. The trade-authority verdict for those timeframes is unaffected: the post-fee P&L floor alone fails for every best-slice candidate.

### Per-slice 1h breakdown (run-1)

For completeness, the per-coin 1h slices that fed into the 1h `best_slice` selection above:

| coin/1h | DA | base DA | DA − base | post_fee_pnl_pct_total | n_trades | trade_share (≈ DCS) | trade_authority | failed condition |
|---|---|---|---|---|---|---|---|---|
| bonk                    | 0.3807 | 0.4028 | −0.0221 | −351.22 | 1036 | 0.6023 | **no** | DA below baseline; P&L < 0; not promoted |
| celestia                | 0.3828 | 0.3890 | −0.0062 | −475.61 | 1386 | 0.8058 | **no** | DA below baseline; P&L < 0; not promoted |
| dogwifcoin              | 0.3911 | 0.3969 | −0.0058 | −345.09 | 1181 | 0.6866 | **no** | DA below baseline; P&L < 0; not promoted |
| floki-inu               | 0.3820 | 0.4085 | −0.0265 | −375.53 | 1239 | 0.7203 | **no** | DA below baseline; P&L < 0; not promoted |
| injective-protocol *(best)* | 0.3784 | 0.3739 | **+0.0045** | −365.31 | 1175 | 0.6831 | **no** | DA lift +0.0045 < +0.02 gate; P&L < 0; not promoted |
| jupiter-exchange-solana | 0.3775 | 0.4031 | −0.0256 | −478.24 | 1365 | 0.7936 | **no** | DA below baseline; P&L < 0; not promoted |
| pepe                    | 0.3538 | 0.3750 | −0.0212 | −410.46 | 1545 | 0.8983 | **no** | DA below baseline; trade_share 0.8983 > 0.85 (overtrading); P&L < 0; not promoted |
| render-token            | 0.3679 | 0.3916 | −0.0237 | −438.96 | 1449 | 0.8424 | **no** | DA below baseline; P&L < 0; not promoted |
| worldcoin-wld           | — | — | — | — | — | — | **no** | insufficient_evidence — slice 9/9 never started; run-1 died after slice 8/9 |

### Per-slice 1h breakdown (run-2, mirror of run-1)

Run-2 has trained 8/9 1h coins by 16:45:52Z — the same 8 coins that run-1 trained, and the 9th coin (`worldcoin-wld/1h`) is again unstarted. Every trained slice's failure pattern reproduces:

| coin/1h | DA | base DA | DA − base | post_fee_pnl_pct_total | n_trades | trade_share | trade_authority | failed condition |
|---|---|---|---|---|---|---|---|---|
| bonk                    | 0.3876 | 0.4021 | −0.0145 | −227.76 |  648 | 0.3767 | **no** | DA below baseline; trade_share 0.3767 below 0.40 floor (under-trading); P&L < 0; not promoted |
| celestia                | 0.3754 | 0.3901 | −0.0147 | −485.77 | 1457 | 0.8471 | **no** | DA below baseline; P&L < 0; not promoted |
| dogwifcoin              | 0.3933 | 0.3958 | −0.0025 | −331.78 | 1035 | 0.6017 | **no** | DA below baseline; P&L < 0; not promoted |
| floki-inu               | 0.3785 | 0.4063 | −0.0278 | −403.93 | 1283 | 0.7459 | **no** | DA below baseline; P&L < 0; not promoted |
| injective-protocol      | 0.3771 | 0.3718 | **+0.0053** | −384.40 | 1269 | 0.7378 | **no** | DA lift +0.0053 < +0.02 gate; P&L < 0; not promoted (mirrors run-1's "best slice" pattern at +0.0045) |
| jupiter-exchange-solana | 0.3735 | 0.4027 | −0.0292 | −504.67 | 1438 | 0.8360 | **no** | DA below baseline; P&L < 0; not promoted |
| pepe                    | 0.3542 | 0.3754 | −0.0212 | −349.05 | 1348 | 0.7837 | **no** | DA below baseline; P&L < 0; not promoted |
| render-token            | 0.3619 | 0.3908 | −0.0289 | −434.35 | 1455 | 0.8459 | **no** | DA below baseline; P&L < 0; not promoted |
| worldcoin-wld           | — | — | — | — | — | — | **no** | insufficient_evidence — slice 9/9 again unstarted (`watchdog_halt` after slice 8/9 in both runs) |

Cross-run consistency on the 8 coins both runs trained: every coin's run-2 DA is within `±0.0103` of its run-1 DA, every post-fee P&L stays in `[-504.67, -227.76]`, and the "best slice by DA-lift" position is held by `injective-protocol/1h` in both runs (+0.0045 in run-1, +0.0053 in run-2 — both ~4× short of the +0.02 promotion floor). There is no plausible outcome for the remaining `worldcoin-wld/1h` slice that promotes it: it would need to clear DA-lift ≥ +0.02 *and* post-fee P&L > 0 *and* trade_share ∈ [0.40, 0.85] simultaneously, which no coin in either run has come close to.

---

## Section C — Recommended runtime state

```
quant_brain_enabled: false
allowed_timeframes:  []
mode_per_timeframe:
  1m: governance      # no model trained; structural / filter use only
  5m: disabled        # 5m_hard_gate_failed for all 9 coins on coverage; no model exists
  1h: shadow          # 8 slices trained, 0 trade-eligible; keep for shadow scoring + further diagnosis
  2h: governance      # not trained in run-1; last on-disk evidence (2026-04-25) is unpromoted
  6h: governance      # not trained in run-1; last on-disk evidence (2026-04-25) is unpromoted
  1d: governance      # not trained in run-1; last on-disk evidence (2026-04-25) is unpromoted
rationale_per_timeframe:
  1m:  "Not in TRADEABLE_TIMEFRAMES for the campaign and no 1m model exists in models/<coin>/; governance/filter-only candidate by default."
  5m:  "All 9 coins emitted slice_skipped with reason=5m_hard_gate_failed in run-1; per phase2_data_audit.json contiguous_days ≈ 65.3 vs the COVERAGE_BAR_DAYS['5m']=305 hard floor — disabled until 5m backfill clears the gate."
  1h:  "8/9 coins trained; every coin's directional_accuracy is below baseline_directional_accuracy (best lift was injective-protocol +0.0045, gate floor +0.02); every coin's post_fee_pnl_pct_total is in [-478.24, -345.09]; verification_history shows slices_promoted=0; shadow-only candidate."
  2h:  "Run-1 halted before tier 2h started; the only evidence on disk is the 2026-04-25 run, which also produced 0 promotions and DA below baseline for every per-coin/2h slice (e.g. floki-inu/2h DA 0.3925 vs base 0.4412); governance/filter-only until a fresh 2h training tier completes."
  6h:  "Run-1 never reached tier 6h; 2026-04-25 baseline 6h slices are also unpromoted (e.g. render-token/6h DA 0.4409 vs base 0.4085 looks numerically close but post_fee_pnl_pct_total=-105.66 and the slice was not promoted by the gate); governance/filter-only."
  1d:  "Run-1 never reached tier 1d; 2026-04-25 best 1d slice was bonk/1d DA 0.4313 vs base 0.3925 with post_fee_pnl_pct_total=-115.64 (not promoted); pepe/1d (the snapshot's stale best=pepe/1d reference) is DA 0.3773 vs base 0.4250 vs ML promotion gate floor of DA > base+0.02 with post_fee_pnl_pct_total > 0 — not trade-eligible."
```

`quant_brain_enabled=false` is also enforced bottom-up by `artifacts/api-server/src/lib/brain-promotion-gate.ts` (`hasPromotedSlice`): the most recent 4 verification rows in `models/verification_history.jsonl` all have `verification_status="ok", passed=false, slices_promoted=0, coins_with_promotion=[]`, so `POST /api/crypto/brain/state` would return `409 reason=no_promoted_slices` if anyone tried to flip it on.

---

## Section D — Follow-up tasks (justified by evidence only)

- **#530 (predict-time STABLE bias)** — supported by run-1 evidence. The `prediction_collapse` block on every 1h slice shows the model under-emitting STABLE relative to the label distribution: pepe/1h `predSTABLE=0.0355` vs `labelSTABLE=0.2488` (gap −21.3pp), jupiter-exchange-solana/1h `0.0262` vs `0.2390` (−21.3pp), render-token/1h `0.0855` vs `0.2006` (−11.5pp), floki-inu/1h `0.1012` vs `0.2407` (−14.0pp), celestia/1h `0.0709` vs `0.2628` (−19.2pp). Pepe's `top_class_share_gap_vs_baseline=+0.1937` and `trade_share=0.8983` (above the 0.85 ceiling) directly confirm the over-rotation forces the trader into the loss-making side: 1545 trades with `net_pct_total=-410.46`. Recommend keeping #530 open.
- **#545/#546 (training-log rotation visibility)** — orthogonal to this verdict. Mention only because the campaign's repeated `watchdog_halt → 1800s timeout → watchdog_resume` loop produced 14 snapshot rows + 7 watchdog_halt rows + 6 watchdog_resume rows in run-1; the operator depends on rotation visibility to find the terminal halt at 13:52:13Z without scanning the whole jsonl. No new task warranted by this verdict beyond what those tasks already cover.
- **#547 (meta-brain learning curve chart)** — orthogonal. No meta-brain training happened in run-1 (no `__meta__` slice_done events in the run-1 window), so this verdict adds no new evidence for or against it.
- **No new follow-up is justified by run-1 evidence beyond what the project already has queued.** The downstream-planned task ("Classify each timeframe by role and enforce the scope at runtime") consumes Section C exactly. The existing 1d/6h/2h watchlist tasks ("Verify the 3 quiet coins actually clear the 15% bar after the next retrain", "Show every coin's recalibration history, not just per timeframe", etc.) already cover the obvious next steps. The campaign produced no new evidence of a missing class of follow-up — the failure modes are STABLE-collapse (#530) + 5m coverage shortfall (already tracked in `5m_topup_health` alerts) + watchdog resume loop (already visible in the progress feed).

---

## Operational note on the workflow respawn loop and the run-2 in-flight state

The workflow `task-399-training-campaign` has produced **three** `campaign_start` events on disk: 2026-04-25T06:33:02Z → `models/training_run_20260425T063302Z/` (the pre-baseline run referenced by the snapshotter's stale `best=pepe/1d`), 2026-04-28T11:09:21Z → `models/training_run_20260428T110921Z/` (run-1, terminal at 13:52:13Z, 8/9 1h trained, 0 promoted), and 2026-04-28T14:00:20Z → `models/training_run_20260428T140020Z/` (run-2, in flight at refresh time but **footprint-complete for tier 1h** — same 8/9 coins trained as run-1, 0 promoted, last `slice_done` `render-token/1h` at 16:45:52Z, then `watchdog_halt`). All three timestamps and run-dir names verified via `jq -c 'select(.phase=="campaign_start") | {emitted_at,run_dir}' models/progress_updates.jsonl`. Each `campaign_start` opens a brand-new `training_run_<ts>/` directory with a fresh `_ts()` timestamp, not a watchdog-resume of the previous run (watchdog resumes are tagged `phase=watchdog_resume` inside the same run dir). The workflow runner respawns the python orchestrator after the prior process exits — confirmed by run-2's `ps -o etime` of 02:44 at 14:05:22Z capture time, etime 02:55+ at 16:55Z, and by the fact that run-2 re-executes phase-1/2/3 from scratch.

The within-run watchdog cycle is the same in both observed runs: train ~2 slices, hit `consecutive_regress=2`, emit `watchdog_halt`, wait the 1800-second `ML_WATCHDOG_RESUME` timeout, emit `watchdog_resume`, train ~2 more slices, repeat. Run-2 has executed four such halt/resume cycles by 16:45:52Z (after slices 2/9, 4/9, 6/9, and 8/9). At this cadence the campaign cannot reach tier 2h/6h/1d before the workflow runner times the python process out and respawns it — which is exactly what happened to run-1 (it died after 8/9 1h slices, having never reached tier 2h). This is a structural impediment to ever completing a single workflow invocation, and it has held for 25 days across three campaign starts: the workflow has produced **0 promotions** in any invocation.

Per the task description's escape hatch ("If the watchdog is still auto-resuming and no progress has been emitted in >2h with the process alive, treat that as a hung run, capture the partial state, and note the hang in the verdict") — the failure mode here is *adjacent but distinct*: run-2's watchdog is still resuming on its 30-minute timer (last `slice_done` 16:45:52Z, next resume due ~17:15:52Z), so the strict >2h-quiet wallclock condition is not met as of refresh time, but run-1 already terminated by workflow respawn and run-2 has now reproduced the exact same tier-1h footprint as run-1 (same 8 trained coins, same 9th coin un-started, same DA-below-baseline outcome, same post-fee-P&L-negative outcome, same 0 promotions). The cross-run reproduction (16 trained 1h slices over the two runs, 0 promotions, every slice DA-below-baseline, every slice post-fee P&L in `[-504.67, -227.76]`) is the strongest available terminal signal short of waiting for run-2's process to die and the workflow runner to respawn into a fourth `campaign_start` (which would, on the run-1 → run-2 evidence, replay the same pattern again). Adding the remaining `worldcoin-wld/1h` slice, plus the 2h/6h tiers if run-2 ever reaches them, would not change the Section A counts (run-1's tier 2h/6h/1d are *terminally* `not reached` for that run dir), the Section B `no` verdicts (the gate floors are not within reach of the observed slice cohort), the Section C recommendation, or the Section D follow-ups. This report is therefore final on the workflow as observed at 2026-04-28T17:00:00Z; the ongoing run-2 process is allowed to continue under its own watchdog logic.

---

## Raw evidence appendix

- `artifacts/ml-engine/models/training_run_20260428T110921Z/phase1_preflight.json` — preflight pass record (62 tests, status="ok", round_trip_cost_pct=0.003, no forbidden-feature leaks).
- `artifacts/ml-engine/models/training_run_20260428T110921Z/phase2_data_audit.json` — coverage audit; 5m contiguous_days ≈ 65.3 vs 305-day floor → drives the `5m_hard_gate_failed` skips in §A.
- `artifacts/ml-engine/models/training_run_20260428T110921Z/phase2_data_audit_pre_backfill.json` — pre-backfill audit (paired with above).
- `artifacts/ml-engine/models/training_run_20260428T110921Z/phase3_baseline_pointer.json` — pointer to baseline archive `models/_archive/20260428T112443Z_pre_full_run/` (3494 per-coin files copied, 4 baseline slices captured).
- `artifacts/ml-engine/models/_archive/20260428T112443Z_pre_full_run/` — pre-run baseline (`backtest_report.json`, `baseline_snapshot.json`, per-coin/per-tf manifests). Confirms the pre-run state had no promoted slices either.
- `artifacts/ml-engine/models/training_run_20260428T140020Z/phase{1,2,3}*.json` — run-2's phase artifacts; same shape as run-1, same coverage gates pass/fail (5m hard-gated, 1d structurally blocked).
- `artifacts/ml-engine/models/progress_updates.jsonl` — live progress feed; run-1 §A counts come from `slice_done` and `slice_skipped` rows with `emitted_at` in `[2026-04-28T11:09:00Z, 2026-04-28T14:00:00Z)`. Run-2 §A counts come from rows with `emitted_at >= 2026-04-28T14:00:20Z`. The §B per-slice DA / baseline DA / post_fee_pnl_pct_total / n_trades / trade_share numbers come straight from each row's `metrics`, `baseline_metrics`, and `pnl_after_fees` blocks. The §D STABLE-collapse numbers come from each row's `prediction_collapse.predicted_class_share.STABLE` vs `prediction_collapse.label_class_share.STABLE`.
- `artifacts/ml-engine/models/verification_history.jsonl` — `tail -5` shows the 4 most-recent records all `verification_status="ok", passed=false, counts.slices_promoted=0, coins_with_promotion=[]`. `grep -c '"slices_promoted": *[1-9]' …/verification_history.jsonl` returns **0**, confirming there is no record anywhere with a promoted slice.
- `artifacts/ml-engine/models/verification/` — empty directory (no per-run verification record was written by run-1 because run-1 never reached the verification phase before the workflow respawn). The promotion-gate evidence above is therefore from `verification_history.jsonl` only.
- `artifacts/api-server/src/lib/brain-promotion-gate.ts` — quote source for the runtime enable gate (`POST /api/crypto/brain/state` → `hasPromotedSlice`); confirms the recommendation `quant_brain_enabled=false` is what the runtime would refuse-down-to anyway.
- `artifacts/ml-engine/scripts/run_full_training_campaign.py` — campaign orchestrator; `COVERAGE_BAR_DAYS = {"5m": 305, ...}` and `TRADEABLE_TIMEFRAMES = ["5m", "1h", "2h", "6h", "1d"]` referenced in §A and §C.

### Reproducibility (jq / grep lines for every cited number)

```
# §A — count slice_done by tf in run-1 window
jq -c 'select(.phase=="slice_done" and .emitted_at>="2026-04-28T11:09:00Z" and .emitted_at<"2026-04-28T14:00:00Z") | .timeframe' \
  artifacts/ml-engine/models/progress_updates.jsonl | sort | uniq -c

# §A run-2 — count slice_done by tf in run-2 window
jq -c 'select(.phase=="slice_done" and .emitted_at>="2026-04-28T14:00:20Z") | .timeframe' \
  artifacts/ml-engine/models/progress_updates.jsonl | sort | uniq -c

# §A — count slice_skipped by tf + reason in run-1 window
jq -c 'select(.phase=="slice_skipped" and .emitted_at>="2026-04-28T11:09:00Z" and .emitted_at<"2026-04-28T14:00:00Z") | [.timeframe,.reason]' \
  artifacts/ml-engine/models/progress_updates.jsonl | sort | uniq -c

# §A run-2 per-slice — DA / baseline DA / post-fee P&L for run-2's trained slices
jq -c 'select(.phase=="slice_done" and .emitted_at>="2026-04-28T14:00:20Z") |
  {coin,tf:.timeframe,DA:.metrics.directional_accuracy,baseDA:.baseline_metrics.directional_accuracy,
   netPctTotal:.pnl_after_fees.net_pct_total,nTrades:.pnl_after_fees.n_trades,tradeShare:.pnl_after_fees.trade_share}' \
  artifacts/ml-engine/models/progress_updates.jsonl

# Workflow restart history — confirm three campaign_starts (04-25, run-1, run-2)
jq -c 'select(.phase=="campaign_start") | {emitted_at,run_dir}' \
  artifacts/ml-engine/models/progress_updates.jsonl

# §B — per-slice DA / baseline DA / post-fee P&L / n_trades / trade_share
jq -c 'select(.phase=="slice_done" and .emitted_at>="2026-04-28T11:09:00Z" and .emitted_at<"2026-04-28T14:00:00Z") |
  {coin,tf:.timeframe,DA:.metrics.directional_accuracy,baseDA:.baseline_metrics.directional_accuracy,
   netPctTotal:.pnl_after_fees.net_pct_total,nTrades:.pnl_after_fees.n_trades,tradeShare:.pnl_after_fees.trade_share}' \
  artifacts/ml-engine/models/progress_updates.jsonl

# §C / promotion-gate — confirm 0 promoted slices anywhere on disk
grep -c '"slices_promoted": *[1-9]' artifacts/ml-engine/models/verification_history.jsonl

# §D — STABLE collapse per slice
jq -c 'select(.phase=="slice_done" and .emitted_at>="2026-04-28T11:09:00Z" and .emitted_at<"2026-04-28T14:00:00Z") |
  {coin,tf:.timeframe,
   predSTABLE:.prediction_collapse.predicted_class_share.STABLE,
   labelSTABLE:.prediction_collapse.label_class_share.STABLE,
   collapseGap:.prediction_collapse.collapse_gap,
   topClassGapVsBaseline:.prediction_collapse.top_class_share_gap_vs_baseline}' \
  artifacts/ml-engine/models/progress_updates.jsonl

# Operational — confirm run-2 is a fresh process, not a watchdog resume
jq -c 'select(.phase=="campaign_start") | {emitted_at,run_dir}' \
  artifacts/ml-engine/models/progress_updates.jsonl
ps -o etime,cmd -p $(pgrep -f run_full_training_campaign | tail -1)
```
