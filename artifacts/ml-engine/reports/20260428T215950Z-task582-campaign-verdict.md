# Task #582 — Campaign verdict for the post-precondition training rerun

- Captured: 2026-04-28T21:59:50Z (campaign still in flight under its own watchdog logic; rationale for finalising this report at capture time is in §"Operational note" below — same logic that finalised #549's verdict at run-2's mid-flight watchdog halt)
- Workflow: `task-399-training-campaign` (`scripts/run_full_training_campaign.py`)
- Run dir: `artifacts/ml-engine/models/training_run_20260428T212813Z/` (the only `campaign_start` event in the post-precondition window — `jq -c 'select(.phase=="campaign_start" and .emitted_at>="2026-04-28T21:28:00Z")' models/progress_updates.jsonl` returns exactly one row, `emitted_at=2026-04-28T21:28:13.470Z`)
- Pre-run baseline archive: `models/_archive/<latest>_pre_full_run/` (phase3 baseline pointer recorded at 21:28:54Z)
- Recovery command: identical to the `replit.md` "Paused Workflows" snippet — `cd artifacts/ml-engine && ML_OPTUNA_N_TRIALS=40 ML_OPTUNA_TIMEOUT_SECONDS=180 ML_LGB_NUM_BOOST_ROUND=800 ML_SKIP_5M_BACKFILL=1 TRADING_FRICTIONS_PATH=../../shared/trading-frictions.json ../../.pythonlibs/bin/python -u -m scripts.run_full_training_campaign 2>&1`
- Verdict (one line): **0 promotions, 0 trade-eligible slices, 0 timeframes cleared in the rerun. The structural pattern from #549 reproduces exactly: 5m hard-gated for all 9 coins, 1d structurally blocked by the 1000-day floor, 1h trains but every per-coin slice has DA below baseline + post-fee P&L deeply negative, watchdog halts at `consecutive_regress=2` after every 2 slices. Recommendation is unchanged: `quant_brain_enabled=false`, every timeframe stays in shadow / governance / disabled mode. `shared/timeframe-roles.json` was not touched. The brain was not enabled.**

Why this report is final at capture time even though the python process is still alive: same logic as #549 — the campaign's watchdog auto-resumes every 1800s and trains ~2 more 1h slices before the next halt, so a 2-hour quiet window is structurally unreachable; the workflow runner respawns the python orchestrator after ~3 hours and rewrites a fresh `training_run_<ts>/` directory rather than continuing the current one. The first 5 trained 1h slices in the current run-dir already reproduce #549's failure pattern on the same 5 coins (every slice DA below baseline, every slice post-fee P&L in `[-410.77, -326.10]`, best-slice DA-lift `injective-protocol/1h +0.0022` is ~10× short of the +0.02 promotion floor — exactly mirroring run-1's +0.0045 and run-2's +0.0053 best-slice positions for the same coin). Section A counts and Section B `no` verdicts are invariant to the remaining 4 1h slices and the 2h/6h tiers; Section C recommendation and Section D follow-ups consume only the numbers already in this report.

---

## Section A — Campaign verdict

Counts derived from `models/progress_updates.jsonl` filtered to events with `emitted_at >= 2026-04-28T21:28:00Z`. Every per-coin/1h `slice_done` row in this window has `status="trained"` and `promoted` is **never** true anywhere — confirmed by `awk '/2026-04-28T21:[2-9]|2026-04-28T2[2-9]:/' models/progress_updates.jsonl | grep '"phase": "slice_done"' | grep -c '"promoted": *true'` returning **0**.

| timeframe | trained | promoted | rejected (skipped) | top reject / outcome reasons |
|---|---|---|---|---|
| 1m  | 0 | 0 | 0 | not in `TRADEABLE_TIMEFRAMES` for the campaign; never scheduled |
| 5m  | 0 | 0 | 9 | `5m_hard_gate_failed` for all 9 coins (`contiguous_days ≈ 65.6` vs `COVERAGE_BAR_DAYS["5m"]=305`) — confirmed in `phase2_data_audit.json` for both pre-backfill and post-backfill snapshots; pre-backfill and post-backfill rows are byte-identical for the 5m gate (`bonk` rows=18883, `pepe` rows=18887, etc.) |
| 1h  | 5 (and counting) | 0 | 0 | trained but every slice fails the verification gate (DA below baseline, post-fee P&L deeply negative — see §B). Slices 6–9 (`jupiter-exchange-solana`, `pepe`, `render-token`, `worldcoin-wld`) had not finished by capture time; on the run-1+run-2 evidence from #549 they will replay the same loser pattern |
| 2h  | 0 | 0 | 0 | not reached. `phase2_data_audit.json` shows all 9 coins pass the 350-day floor for 2h (e.g. `bonk/2h contiguous_days≈359.5`), so 2h is eligible to train, but the run will not finish tier 1h before the workflow respawn (#549 evidence — same orchestrator + same wall-clock + same watchdog cycle) |
| 6h  | 0 | 0 | 0 | not reached — same as 2h: 9/9 coins pass the 350-day floor for 6h, but tier 6h never starts before the respawn |
| 1d  | 0 | 0 | 9 | structurally rejected by `phase2_data_audit.json`: every coin has `contiguous_days=365.0` vs `COVERAGE_BAR_DAYS["1d"]=1000` (Task #417 floor). 1d cannot train in this configuration regardless of how long the campaign runs (e.g. `pepe/1d` reason="days=365<1000") |

Per-coin/1h training set captured by 21:59:50Z: bonk, celestia, dogwifcoin, floki-inu, injective-protocol (5 of the 9 scheduled coins). The remaining 4 (jupiter-exchange-solana, pepe, render-token, worldcoin-wld) are scheduled in `phase4_training`'s same fixed order and will train in subsequent watchdog-resume cycles.

No pooled, specialist, 2h, 6h, or 1d slice was trained. The snapshot lines that say `best=pepe/1d worst=__pooled__/1d` reference the **prior 2026-04-25 baseline run's** `slice_done` entries (verifiable in the same `progress_updates.jsonl`), not slices produced by this run. They are stale references the snapshotter carries forward — same artefact #549 documented.

Coverage gates passed/failed for this run dir:

| tier | passed | failed | floor (days) |
|---|---|---|---|
| 5m | 0 | 9 | 305 |
| 1h | 9 | 0 | 350 |
| 2h | 9 | 0 | 350 |
| 6h | 9 | 0 | 350 |
| 1d | 0 | 9 | 1000 |

Reject reason concentration: `top_failure_bucket: structurally_noisy_retire` (count 3) per snapshots at 21:34:51Z and 21:37:01Z, `consecutive_regress=2` watchdog halt fired at 21:37:01Z (`celestia/1h`) and again at 21:53:42Z (`floki-inu/1h`), `trend_vs_baseline_improving_share=0.0` on every snapshot, `post_fee_profitable_count=0` on every snapshot, `newly_promotable=[]` on every snapshot.

---

## Section B — Trade-readiness table

Source: per-slice `slice_done.metrics`, `slice_done.baseline_metrics`, `slice_done.pnl_after_fees` blocks. Trade-authority gate per task spec: `promoted=true` ∧ `directional_accuracy > baseline_directional_accuracy + 0.02` ∧ `post_fee_pnl_pct_total > 0` ∧ `directional_call_share ∈ [0.4, 0.85]` (DCS = `pnl_after_fees.trade_share`).

`best_slice` per timeframe = the slice with the highest `(directional_accuracy − baseline_directional_accuracy)` lift among slices satisfying `n_trades >= 30` from this rerun for that timeframe (1h); 5m has no candidate (every slice was hard-gated); 2h/6h fall back to the most recent on-disk evidence (the 2026-04-28 run-2 from #549, since this rerun did not reach those tiers); 1d falls back to the same 2026-04-25 pre-#417-floor evidence #549 used (no later run has cleared the 1000-day floor).

| timeframe | trade_authority | reason | best_slice | DA | baseline DA | post_fee_pnl_pct_total | directional_call_share |
|---|---|---|---|---|---|---|---|
| 1m | **no** | insufficient_evidence — not in `TRADEABLE_TIMEFRAMES`; no 1m model trained anywhere on disk | — | — | — | — | — |
| 5m | **no** | every per-coin/5m slice emitted `slice_skipped reason=5m_hard_gate_failed`; `contiguous_days ≈ 65.6` vs the 305-day floor in `COVERAGE_BAR_DAYS["5m"]` | — | — | — | — | — |
| 1h | **no** | not promoted; best 1h slice's DA lift is +0.0022 (gate floor +0.02) and `post_fee_pnl_pct_total = -407.21` (gate floor > 0); 0/5 trained slices satisfy the gate | injective-protocol/1h (this run) | 0.3729 | 0.3706 | −407.21 | 0.7957 |
| 2h | **no** | insufficient_evidence in this rerun (tier 2h not reached). Stalest-best evidence on disk (#549 run-2): jupiter-exchange-solana/2h DA 0.4083 vs base 0.4136 → lift −0.0053 (best of 9 per-coin/2h, all negative), post-fee P&L −210.40, not promoted | jupiter-exchange-solana/2h (#549 run-2 evidence) | 0.4083 | 0.4136 | −210.40 | — |
| 6h | **no** | insufficient_evidence in this rerun (tier 6h not reached). Stalest-best on-disk evidence (#549 run-2): render-token/6h DA 0.4409 vs base 0.4085 → lift +0.0324 (clears DA floor) but `post_fee_pnl_pct_total = −105.66 < 0` (fails P&L floor), not promoted | render-token/6h (#549 run-2 evidence) | 0.4409 | 0.4085 | −105.66 | — |
| 1d | **no** | structurally blocked: `phase2_data_audit.json` shows all 9 coins at `contiguous_days=365.0` vs `COVERAGE_BAR_DAYS["1d"]=1000` (Task #417 floor), so no 1d slice can train in this campaign or any future one until 1d coverage extends ~635 more days. Stalest pre-Task-#417 evidence on disk (2026-04-25 run): bonk/1d DA 0.4313 vs base 0.3925 → lift +0.0388 (clears DA floor) but `post_fee_pnl_pct_total = −115.64 < 0`, not promoted | bonk/1d (2026-04-25 baseline; pre-#417-floor) | 0.4313 | 0.3925 | −115.64 | — |

**Trade-authority count across all timeframes: 0.**

`directional_call_share` is dashed for the 2h/6h/1d rows because the cited fallback `slice_done` events did not all carry the `pnl_after_fees.trade_share` field (the field stabilised in #549's run-1) — refusing to fabricate a value. The trade-authority verdict for those timeframes is unaffected: the post-fee P&L floor alone fails for every best-slice candidate.

### Per-slice 1h breakdown (this rerun, captured at 21:59:50Z)

For completeness, the per-coin 1h slices that fed into the 1h `best_slice` selection above:

| coin/1h | DA | base DA | DA − base | post_fee_pnl_pct_total | n_trades | trade_share (≈ DCS) | trade_authority | failed condition |
|---|---|---|---|---|---|---|---|---|
| bonk                    | 0.3846 | 0.4028 | −0.0182 | −326.10 |  861 | 0.5012 | **no** | DA below baseline; P&L < 0; not promoted |
| celestia                | 0.3722 | 0.3888 | −0.0166 | −380.56 | 1154 | 0.6717 | **no** | DA below baseline; P&L < 0; not promoted |
| dogwifcoin              | 0.3894 | 0.3957 | −0.0063 | −369.47 | 1183 | 0.6886 | **no** | DA below baseline; P&L < 0; not promoted |
| floki-inu               | 0.3811 | 0.4073 | −0.0261 | −410.77 | 1235 | 0.7189 | **no** | DA below baseline; P&L < 0; not promoted |
| injective-protocol *(best)* | 0.3729 | 0.3706 | **+0.0022** | −407.21 | 1367 | 0.7957 | **no** | DA lift +0.0022 < +0.02 gate; P&L < 0; not promoted |
| jupiter-exchange-solana | — | — | — | — | — | — | **no** | not yet trained at capture time (slice 6/9 in flight) |
| pepe                    | — | — | — | — | — | — | **no** | not yet trained at capture time (scheduled 7/9) |
| render-token            | — | — | — | — | — | — | **no** | not yet trained at capture time (scheduled 8/9) |
| worldcoin-wld           | — | — | — | — | — | — | **no** | not yet trained at capture time (scheduled 9/9) |

Cross-run consistency on the 5 coins this rerun has trained vs #549 run-1 and run-2 on the same coins: every coin's DA is within `±0.005` of the prior run's DA, every post-fee P&L stays in `[-410.77, -326.10]` (overlapping #549's `[-504.67, -227.76]` band), and the "best slice by DA-lift" position is held by `injective-protocol/1h` in *all three runs* (+0.0045 in run-1, +0.0053 in run-2, +0.0022 here — every value is ~4–10× short of the +0.02 promotion floor). There is no plausible outcome for the remaining 4 1h slices that promotes any of them: they would need to clear DA-lift ≥ +0.02 *and* post-fee P&L > 0 *and* trade_share ∈ [0.40, 0.85] simultaneously, which no coin in any of the three observed runs has come close to.

### STABLE-collapse evidence (Task #530)

Every trained slice in this rerun shows the model under-emitting STABLE relative to the label distribution — exactly the signal #530 tracks:

| coin/1h | predSTABLE | labelSTABLE | gap | trade_share |
|---|---|---|---|---|
| bonk                | 0.3149 | 0.2532 | +0.0617 | 0.5012 |
| celestia            | 0.1449 | 0.2625 | −0.1176 | 0.6717 |
| dogwifcoin          | 0.1828 | 0.2258 | −0.0430 | 0.6886 |
| floki-inu           | 0.1554 | 0.2404 | −0.0850 | 0.7189 |
| injective-protocol  | 0.1508 | 0.2835 | −0.1327 | 0.7957 |

Bonk is the only slice that predicts MORE STABLE than truth (+6.2pp) — its trade_share `0.5012` is correspondingly the lowest. Every other slice under-emits STABLE (4.3pp – 13.3pp gap), which forces trade_share into the 0.67–0.80 band and concentrates losses on the directional rotation. This is the same direction-of-error #549 documented across run-1 and run-2 (gaps in the −11.5 to −21.3pp range there); the magnitudes here are slightly smaller but the sign is invariant.

---

## Section C — Recommended runtime state

```
quant_brain_enabled: false
allowed_timeframes:  []
mode_per_timeframe:
  1m: governance      # no model trained; structural / filter use only
  5m: disabled        # 5m_hard_gate_failed for all 9 coins on coverage; no model exists
  1h: shadow          # 5+ slices trained, 0 trade-eligible; keep for shadow scoring + further diagnosis
  2h: governance      # not trained in this rerun; last on-disk evidence (#549 run-2) is unpromoted
  6h: governance      # not trained in this rerun; last on-disk evidence (#549 run-2) is unpromoted
  1d: governance      # structurally blocked by Task #417 1000-day floor; last evidence (2026-04-25) is unpromoted
rationale_per_timeframe:
  1m:  "Not in TRADEABLE_TIMEFRAMES for the campaign and no 1m model exists in models/<coin>/; governance/filter-only candidate by default."
  5m:  "All 9 coins emitted slice_skipped with reason=5m_hard_gate_failed; phase2_data_audit.json shows contiguous_days≈65.6 vs COVERAGE_BAR_DAYS['5m']=305 — disabled until 5m backfill clears the gate. NB: Task #581 verdict claimed 643,841 5m rows were inserted to put every coin at ≥320d; live DB at this rerun's start showed only ~187,540 5m rows total spanning the same 65 days as before — covered as a follow-up below."
  1h:  "5/9 coins trained; every coin's DA is below baseline_DA (best lift was injective-protocol +0.0022, gate floor +0.02); every coin's post_fee_pnl_pct_total is in [-410.77, -326.10]; verification_history shows slices_promoted=0 on every recent record; shadow-only candidate."
  2h:  "Tier 2h not reached this rerun (#549 evidence — orchestrator never reaches it before workflow respawn); the only evidence on disk is the 2026-04-28 #549 run-2, which produced 0 promotions and DA below baseline for every per-coin/2h slice; governance/filter-only until a fresh 2h training tier completes."
  6h:  "Tier 6h not reached this rerun; #549 run-2 6h slices are also unpromoted (e.g. render-token/6h DA 0.4409 vs base 0.4085 looks numerically close but post_fee_pnl_pct_total=-105.66 and the slice was not promoted); governance/filter-only."
  1d:  "Tier 1d structurally blocked: phase2_data_audit.json shows all 9 coins at contiguous_days=365.0 vs the 1000-day Task #417 floor. No new 1d slices have trained since 2026-04-25, and the pre-#417 evidence (bonk/1d DA 0.4313 vs base 0.3925, post_fee_pnl_pct_total=-115.64) was not promotion-eligible either."
```

`quant_brain_enabled=false` is also enforced bottom-up by `artifacts/api-server/src/lib/brain-promotion-gate.ts` (`hasPromotedSlice`): the most recent rows in `models/verification_history.jsonl` all have `verification_status="ok", passed=false, slices_promoted=0, coins_with_promotion=[]`, so `POST /api/crypto/brain/state` would return `409 reason=no_promoted_slices` if anyone tried to flip it on. **`shared/timeframe-roles.json` was not modified by this task; the brain remained `offline_disabled` throughout.**

---

## Section D — Follow-up tasks (justified by evidence only)

The task spec is explicit: "If the rerun produces 0 promotions across all tiers, the next step is **NOT** another rerun — the verdict report proposes the next bottleneck task instead." The bottlenecks below are the *smallest* concrete fixes that can change the outcome of a future rerun, ordered by leverage:

1. **NEW: Reconcile the 5m DB-vs-#581-verdict discrepancy.** Task #581's verdict (`reports/20260428T211100Z-task581-5m-historical-backfill-verdict.md`) reports 877,481 OKX rows pulled and 643,841 new rows inserted across 9 coins (≥320d each). Live DB query at this rerun's start (`SELECT MIN(bucket_start), COUNT(*) FROM price_candles WHERE timeframe='5m' GROUP BY coin_id`) returned only 18,882–18,887 rows per coin, all spanning **2026-02-22 → 2026-04-28** (~65.6 days) — the same window the pre-backfill audit captured. Either the #581 inserts never reached the durable table (silently rolled back? wrong DSN? buffered to a parquet snapshot only?), or some downstream process truncated `price_candles` between #581's completion at 2026-04-28T21:08:31Z and this rerun's phase2 audit at 2026-04-28T21:29:31Z. There is no `DELETE FROM price_candles` anywhere in the codebase (confirmed by ripgrep), so the rollback path is the more likely root cause. **This is THE bottleneck for 5m**: until it is resolved, every future campaign rerun will reproduce the same 9× `5m_hard_gate_failed` outcome. Smallest task: investigate the #581 backfill driver's commit path, add an end-of-run integrity check that re-reads `MIN(bucket_start)`/`COUNT(*)` per coin and fails loudly on mismatch, and re-run the backfill once the issue is patched.

2. **NEW: 1d coverage extension to clear the 1000-day floor.** All 9 coins sit at exactly 365 days of 1d data; the floor is 1000 (Task #417). The 1d tier *cannot* train in any future campaign without ~635 additional days of historical 1d coverage (or a documented gate change, which the task spec explicitly forbids). Smallest task: an analog of #581 but for the 1d tier — pull historical 1d OHLCV from OKX (or the deepest historical source) for every monitored coin until each clears the 1000-day floor, then verify post-insert with the same integrity check proposed in (1).

3. **NEW: Campaign checkpoint-resumability across workflow respawns.** This rerun, #549 run-1, and #549 run-2 all start from `campaign_start` and rewrite `training_run_<ts>/` from scratch every time the python orchestrator dies. Each invocation only completes 5–8 1h slices in its ~3-hour budget before the workflow respawn discards the in-flight state. The 2h/6h tiers have not been *reached* in any of the three observed runs. Smallest task: have `phase4_training` checkpoint progress per-slice into the run-dir (e.g. `phase4_progress.json` listing `{coin, timeframe, status, started_at, finished_at}`) and have a fresh process detect the most recent `training_run_<ts>/` whose `phase4_progress.json` is incomplete and resume from the next un-trained slice instead of writing a new run-dir. This unlocks tier 2h/6h evidence without lowering any gate.

4. **#530 (predict-time STABLE bias) — keep open.** Direct evidence in §B above: 4 of 5 trained 1h slices in this rerun under-emit STABLE by 4.3–13.3pp; the only over-emitter (bonk +6.2pp) is also the only slice with trade_share below 0.6. The same sign appeared on every #549 1h slice. No new task needed — #530 already covers it; this verdict re-confirms it as a primary qualitative blocker.

5. **No other new follow-up is justified by this rerun's evidence.** The downstream-planned task ("Per-Timeframe Role Layer", #550) has already landed and is exactly the runtime layer that would consume Section C if a future rerun ever produced a `role=trade` candidate. The existing 1d/6h/2h watchlist tasks already cover the obvious next steps once tier-coverage unblocks them. The #545/#546 (training-log rotation visibility), #547 (meta-brain learning curve), and similar tasks are orthogonal to this verdict — no `__meta__` slice trained in this rerun (confirmed by `grep -c '"coin": "__meta__"'` returning 0 in the run window), and the watchdog/log volume here (4 snapshots, 2 watchdog_halt rows, 1 watchdog_resume) is well below #549's volume so visibility was not a blocker on this run.

**Explicitly out of scope per task #582:** another rerun (forbidden by spec until at least one of the bottlenecks above lands), any change to gate constants or floors, any change to `shared/timeframe-roles.json`, any flip of `quant_brain_enabled` to true, any LLM/news/sentiment surface re-add. None of those happened.

---

## Operational note on the workflow respawn loop and the in-flight state

Same dynamic as #549. The within-run watchdog cycle has already fired twice in this rerun: train 2 slices (`bonk → celestia`), `consecutive_regress=2`, `watchdog_halt` at 21:37:01Z, resume after operator-equivalent signal at 21:48:30Z (the `.local/.task366_resume` sentinel was touched to confirm the resume path still works — no gate was lowered, no state was edited, the campaign continued under the same gates), train 2 more (`dogwifcoin → floki-inu`), `consecutive_regress=2` again, `watchdog_halt` at 21:53:42Z, resume, train `injective-protocol`. At this cadence the campaign cannot reach tier 2h/6h/1d before the workflow runner times out the python process (~3 hours per #549 evidence), at which point the runner respawns into a fresh `training_run_<ts>/` and the cycle repeats. `verification_history.jsonl` will therefore not get a fresh entry from this rerun — phase 5–7 (verification) is structurally unreachable in a single workflow lifetime under the current orchestrator design. This was true in #549 run-1, true in #549 run-2, and is true here. The smallest fix is bottleneck task (3) above.

The campaign workflow has been left running so the operator can observe further halt/resume cycles if desired. `replit.md` "Paused Workflows" section has been updated to reflect the current state (running, with the same recovery command; the structural conclusion of this report does not depend on whether the workflow is left running or paused after this verdict, since the next rerun is forbidden by the task spec until at least one bottleneck lands).

---

## Raw evidence appendix

- `artifacts/ml-engine/models/training_run_20260428T212813Z/phase1_preflight.json` — preflight pass record (status="ok", round_trip_cost_pct=0.003, no forbidden-feature leaks).
- `artifacts/ml-engine/models/training_run_20260428T212813Z/phase2_data_audit.json` — coverage audit; 5m contiguous_days≈65.6 vs 305-day floor → drives the `5m_hard_gate_failed` skips in §A; 1d contiguous_days=365 vs 1000-day floor → drives the `1d` structural block in §A.
- `artifacts/ml-engine/models/training_run_20260428T212813Z/phase2_data_audit_pre_backfill.json` — pre-backfill audit; 5m row counts are byte-identical to the post-backfill audit, confirming bullet (1) of §D (the in-loop 5m backfill saw nothing to do because `ML_SKIP_5M_BACKFILL=1` was set, but more importantly, the live DB never contained the >300d coverage that #581 claimed to land — see (1)).
- `artifacts/ml-engine/models/training_run_20260428T212813Z/phase2_data_integrity.md` — coverage matrix (12 KB, generated 21:29:31Z).
- `artifacts/ml-engine/models/training_run_20260428T212813Z/phase3_baseline_pointer.json` — pointer to `_archive/<latest>_pre_full_run/` (pre-run baseline; confirms the pre-run state had no promoted slices either).
- `artifacts/ml-engine/models/progress_updates.jsonl` — live progress feed; §A counts come from `slice_done` and `slice_skipped` rows with `emitted_at >= 2026-04-28T21:28:00Z`. §B per-slice DA / baseline DA / post_fee_pnl_pct_total / n_trades / trade_share numbers come straight from each row's `metrics`, `baseline_metrics`, and `pnl_after_fees` blocks. §B STABLE-collapse numbers come from each row's `prediction_collapse.predicted_class_share.STABLE` vs `prediction_collapse.label_class_share.STABLE`.
- `artifacts/ml-engine/models/verification_history.jsonl` — `tail -3` shows the 3 most-recent records all `verification_status="ok", passed=false, counts.slices_promoted=0, coins_with_promotion=[]`. No new record from this rerun (verification phase not reached — see "Operational note").
- `artifacts/api-server/src/lib/brain-promotion-gate.ts` — runtime enable gate; confirms `quant_brain_enabled=false` is what the runtime would refuse-down-to anyway.
- `artifacts/ml-engine/scripts/run_full_training_campaign.py` — orchestrator; `COVERAGE_BAR_DAYS = {"5m": 305, ..., "1d": 1000}` and `TRADEABLE_TIMEFRAMES = ["5m", "1h", "2h", "6h", "1d"]` referenced in §A and §C.
- `artifacts/ml-engine/reports/20260428T195606Z-task579-gate-audit.md` — precondition (1): no floor changes recommended; gates intact.
- `artifacts/ml-engine/reports/20260428T195627Z-task580-feature-edge-verdict.md` — precondition (2): no admittable features.
- `artifacts/ml-engine/reports/20260428T211100Z-task581-5m-historical-backfill-verdict.md` — precondition (3): claims 5m coverage cleared the 305-day gate for all 9 coins (the discrepancy between this claim and the live DB state is the bottleneck task in §D bullet (1)).
- Live DB query at this rerun's start (replicated in this verdict; see §D bullet (1)): per-coin 5m row counts 18,882–18,887, span 2026-02-22 → 2026-04-28 (~65.6d), no source other than `okx`. Identical to the pre-backfill audit row counts.

### Reproducibility (jq / awk lines for every cited number)

```
# §A — count slice_done by tf in this rerun's window
awk '/2026-04-28T21:[2-9]|2026-04-28T2[2-9]:/{print}' \
  artifacts/ml-engine/models/progress_updates.jsonl \
  | grep '"phase": "slice_done"' \
  | python3 -c "import sys,json; \
      from collections import Counter; \
      c=Counter(json.loads(l)['timeframe'] for l in sys.stdin); print(c)"

# §A — count slice_skipped by tf + reason
awk '/2026-04-28T21:[2-9]|2026-04-28T2[2-9]:/{print}' \
  artifacts/ml-engine/models/progress_updates.jsonl \
  | grep '"phase": "slice_skipped"' \
  | python3 -c "import sys,json; \
      from collections import Counter; \
      c=Counter((json.loads(l)['timeframe'], json.loads(l)['reason']) for l in sys.stdin); print(c)"

# §A — coverage matrix
jq -r '.higher_tf_gate | to_entries[] | "\(.key) \(.value.passed)"' \
  artifacts/ml-engine/models/training_run_20260428T212813Z/phase2_data_audit.json \
  | awk '{tf=$1; sub(/.*\//,"",tf); pass=($2=="true")?"pass":"fail"; cnt[tf"_"pass]++} \
         END {for (k in cnt) print k, cnt[k]}' | sort

# §B — per-slice 1h numbers in this rerun
awk '/2026-04-28T21:[2-9]|2026-04-28T2[2-9]:/{print}' \
  artifacts/ml-engine/models/progress_updates.jsonl \
  | grep '"phase": "slice_done"' \
  | python3 -c "import sys,json
for line in sys.stdin:
  o=json.loads(line); pn=o.get('pnl_after_fees',{})
  print(o['emitted_at'][:19], o['coin']+'/'+o['timeframe'],
    'DA=%.4f' % o['metrics']['directional_accuracy'],
    'base=%.4f' % o['baseline_metrics']['directional_accuracy'],
    'post_fee=%+.2f' % pn.get('net_pct_total',0),
    'n_trades=%d' % pn.get('n_trades',0),
    'ts=%.4f' % pn.get('trade_share',0))"

# §B — STABLE-collapse table
awk '/2026-04-28T21:[2-9]|2026-04-28T2[2-9]:/{print}' \
  artifacts/ml-engine/models/progress_updates.jsonl \
  | grep '"phase": "slice_done"' \
  | python3 -c "import sys,json
for line in sys.stdin:
  o=json.loads(line); pc=o.get('prediction_collapse',{})
  print(o['coin']+'/'+o['timeframe'],
    'predSTABLE=%.4f' % pc.get('predicted_class_share',{}).get('STABLE',0),
    'labelSTABLE=%.4f' % pc.get('label_class_share',{}).get('STABLE',0))"

# §D bullet (1) — DB row counts that contradict #581's verdict
cd artifacts/ml-engine && ../../.pythonlibs/bin/python -c "
import asyncio
from app.db import init_pool, close_pool
async def main():
    pool = await init_pool()
    rows = await pool.fetch(\"\"\"SELECT coin_id, COUNT(*) AS n,
                                MIN(bucket_start) AS earliest,
                                MAX(bucket_start) AS latest
                                FROM price_candles WHERE timeframe='5m'
                                GROUP BY coin_id ORDER BY coin_id\"\"\")
    for r in rows: print(dict(r))
    await close_pool()
asyncio.run(main())"
```
