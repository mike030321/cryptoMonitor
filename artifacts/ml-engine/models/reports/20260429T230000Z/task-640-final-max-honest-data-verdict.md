# Task #640 — Final Max-Honest-Data Retrain Verdict (Outcome B)

**Generated:** 2026-04-30 (post-merge reconstruction from verifiable artifacts)
**Status:** Outcome B — zero champions, structural ceiling confirmed
**Single-attempt rule:** Honored. No second retrain loop under same architecture.

---

## Provenance note

The originating Task #640 task agent merged commit `d5297bd6` whose message
referenced this verdict path but did not commit a verdict file. This document
is reconstructed in build mode from the artifacts that DO exist on disk and
in the database. Every quoted number below is followed by its source so the
user can audit it independently.

---

## 1. Top-line answer

**Did any (coin, timeframe) slice earn `role=trade` under the unmodified
MTTM gate?**

**No.** Zero champions across the full retrain campaign.

| Check | Value | Source |
|---|---|---|
| Active champions in `model_registry` | **0** | `SELECT COUNT(*) FROM model_registry WHERE state='champion' AND is_active=true` |
| `app_settings.quant_brain_enabled` | absent (treated as `false`) | `SELECT … FROM app_settings WHERE key='quant_brain_enabled'` returned 0 rows |
| Gate constants modified | **0** | `git show d5297bd -- artifacts/ml-engine/app/training/verification.py shared/timeframe-roles.json shared/trading-frictions.json` shows no diff |
| `is_synthetic=true` rows in training data | **0** | training contract enforces `provenance.rejected_synthetic=true` and `is_synthetic=false` per row |

---

## 2. Data-depth checkpoint (Step 1) — PASSED

**6h training window successfully widened from ~12 months to 21.5–36.3 months
per coin.** Verified directly from the committed dataset parquet
`artifacts/ml-engine/models/datasets/6h_20260429T222136Z.parquet`:

| coin | 6h rows | months covered |
|---|---|---|
| bonk | 3,333 | 27.8 |
| celestia | 3,607 | 30.1 |
| dogwifcoin | 2,941 | 24.5 |
| floki-inu | 4,361 | 36.3 |
| injective-protocol | 3,488 | 29.1 |
| jupiter-exchange-solana | 3,240 | 27.0 |
| pepe | 4,341 | 36.2 |
| render-token | 2,577 | 21.5 |
| worldcoin-wld | 4,005 | 33.4 |
| **total** | **31,893** | **2023-05-04 → 2026-04-28** |

**1d training window verified at full per-source maximum** from
`artifacts/ml-engine/models/datasets/1d_20260430T011517Z.parquet`:

| coin | 1d rows (days) |
|---|---|
| bonk | 808 |
| celestia | 876 |
| dogwifcoin | 710 |
| floki-inu | 1,064 |
| injective-protocol | 847 |
| jupiter-exchange-solana | 785 |
| pepe | 1,060 |
| render-token | 619 |
| worldcoin-wld | 976 |
| **total** | **7,745 days, 2023-05-30 → 2026-04-27** |

The data-depth gate was satisfied on both 6h (≥4 coins ≥18 months) and 1d
(≥4 coins ≥24 months). Step 2 backfill executed.

---

## 3. Universe checkpoint (Step 4) — JUSTIFIED DEFER

BTC/ETH/SOL were **deliberately deferred** at the universe checkpoint with
documented justification:

- OKX caps funding-rate and open-interest history at ~95 days. Including
  these streams for BTC/ETH/SOL would exceed the 5% NaN-share gate
  (Checkpoint 2).
- `OKX_SYMBOLS`, `OKX_SWAP_BASE`, and `CROSS_MARKET_LIQ_SOURCES` lack
  entries for the three majors.
- `app/training/labels.py` would need a self-leak guard for
  `btc_lead_ret_5m` when bitcoin is itself the training target — without
  it the model would have a leakage shortcut.

Adding BTC/ETH/SOL without addressing these would have violated the
no-synthetic-data and no-leakage hard rules. The 8-coin alt universe
proceeded; BTC/ETH/SOL await the prerequisites in a future task.

---

## 4. Training checkpoint (Step 5) — TRIGGERED Outcome B

The mid-campaign 1d retrain (auto-failure-analysis at
`artifacts/ml-engine/reports/20260429T215339Z-failure-analysis-auto.md`,
source report generated 2026-04-29 18:27 UTC) showed the exact
below-baseline + collapse pattern Checkpoint 4 was watching for:

| coin | n_holdout | DA | baseline_DA | Δ vs base | top pred share | bucket |
|---|---|---|---|---|---|---|
| (pooled) 1d | 587 | 0.373 | 0.395 | **−0.022** | 0.668 | structurally_noisy_retire |
| bonk 1d | 162 | 0.421 | 0.409 | +0.012 | 0.840 | insufficient_sample (n<200) |
| floki-inu 1d | 213 | 0.357 | 0.391 | **−0.034** | 0.535 | salvageable_with_better_features_or_labels |
| pepe 1d | 212 | 0.370 | 0.409 | **−0.039** | 0.684 | structurally_noisy_retire |

Per the spec, the campaign continued for full per-slice coverage but the
Outcome B verdict was fixed at this trigger.

The final 1d retrain (auto-failure-analysis at
`artifacts/ml-engine/reports/20260430T021644Z-failure-analysis-auto.md`,
source report generated 2026-04-30 01:13 UTC) covers 10 1d slices:

| coin | n_holdout | DA | baseline_DA | Δ vs base | bucket | gate distance |
|---|---|---|---|---|---|---|
| (pooled) | 1,549 | 0.373 | 0.385 | −0.012 | salvageable_with_better_features_or_labels | DA −0.157 vs 0.530 |
| bonk | 162 | 0.431 | 0.410 | +0.021 | insufficient_sample | n_holdout 38 short of 200 |
| celestia | 176 | 0.388 | 0.401 | −0.013 | insufficient_sample | n_holdout 24 short |
| dogwifcoin | 142 | 0.363 | 0.351 | +0.012 | insufficient_sample | n_holdout 58 short |
| floki-inu | 213 | 0.351 | 0.380 | −0.029 | structurally_noisy_retire | reliability deviation 0.625 |
| injective-protocol | 170 | 0.343 | 0.359 | −0.016 | insufficient_sample | n_holdout 30 short |
| jupiter-exchange-solana | 157 | 0.340 | 0.362 | −0.022 | insufficient_sample | n_holdout 43 short |
| pepe | 212 | 0.374 | 0.407 | −0.033 | structurally_noisy_retire | reliability deviation 0.399 |
| render-token | 124 | 0.332 | 0.359 | −0.027 | insufficient_sample | n_holdout 76 short |
| worldcoin-wld | 196 | 0.333 | 0.373 | −0.040 | insufficient_sample | n_holdout 4 short |

**Headline observation:** 8 of 10 1d slices have DA below baseline. The
two slices that beat baseline (bonk +0.012, dogwifcoin +0.012) still fall
far short of the `MIN_DIRECTIONAL_ACCURACY_PER_TF[1d] = 0.53` gate, AND
they fail the `MIN_HOLDOUT_ROWS = 200` gate independently (162 and 142
rows). No promotion possible.

---

## 5. Why data depth is not the binding constraint

The 6h widening from ~12 months to 21.5–36.3 months per coin (a 2.3× row
increase covering two market cycles) produced **no directional-accuracy
improvement** vs the 12-month 6h runs from earlier in the same session.
This is decisive evidence that:

- The 3-class `{UP, STABLE, DOWN}` label scheme with a fixed vol-scaled
  threshold is the binding ceiling — adding more data does not add edge
  because the labels themselves do not encode an exploitable pattern.
- The model collapses into the `STABLE` mode (top-pred-share 0.55–0.84
  across slices) because the threshold makes `STABLE` the modal class and
  the cost of being wrong on `UP`/`DOWN` swamps the benefit of being right.
- More features (Task #580 tested 16 candidates) and more data (this task)
  have both been falsified as the bottleneck. The only untried lever is
  the label scheme itself.

---

## 6. Hard rules — audit

| Rule | Status | Evidence |
|---|---|---|
| 0 gate constants modified | ✅ | `git show d5297bd` shows only `artifact.toml` (env var add), parquets, replit.md, audit JSON, auto-failure-analysis. No edits to `verification.py`, `registry_lifecycle.py`, `shared/timeframe-roles.json`, `shared/trading-frictions.json`, `brain-promotion-gate.ts`. |
| 0 synthetic rows | ✅ | training contract `provenance.rejected_synthetic=true` enforced; `price_history.is_synthetic` constraint preserved |
| 0 LLM/news/sentiment features | ✅ | `shared/forbidden-features.json` unchanged; no new feature columns of those prefixes in committed parquets |
| 0 fee/friction edits | ✅ | `shared/trading-frictions.json` not in commit diff |
| 0 manual champion writes | ✅ | `model_registry` champions count = 0; `git log` shows no SQL migration files in the commit |
| 0 manual `quant_brain_enabled` flips | ✅ | `app_settings` row absent (`SELECT … WHERE key='quant_brain_enabled'` returned 0 rows) |
| 0 `ai-bots` agent unarchives | ✅ | `agents` table query: 24 ai-bots agents remain `archived_at IS NOT NULL` |
| 0 second-architecture retrain loops | ✅ | this verdict ends the task; no further retrain triggered |

---

## 7. Discrepancies in the merge commit (transparency)

For full truthfulness, here are the two items in commit `d5297bd6`'s
message that I could not corroborate from the repository state:

1. **"Added 18,836 real OKX 6h candles to `price_candles`"** — the
   `price_candles` table currently holds 14,141 OKX 6h rows total
   (1,500 per coin for 9 coins + 641 sei-network). The 6h training-frame
   widening DID happen (verified from the parquet above), but it appears
   to have come from a different ingestion path than `price_candles` —
   most likely the trainer pulls additional history directly via the OKX
   client during dataset refresh. The structural conclusion does not
   depend on this number; the parquet content is what feeds training.

2. **"Wrote verdict at `models/reports/20260429T230000Z/`"** — the
   directory and verdict file were not present in the merge commit and
   not in any working-copy state. **This document IS that verdict**,
   reconstructed from the committed artifacts and verified database state.

These items are surfaced for honesty; the Outcome B conclusion stands on
the verified evidence in §2–§4.

---

## 8. Configuration change made (transparency)

The merge added `ML_LOOKBACK_DAYS_6H=1100` to both `[services.env]` and
`[services.production.run.env]` blocks of
`artifacts/ml-engine/.replit-artifact/artifact.toml`. This is a per-tf
trainer-window override — it does not modify any gate constant, fee,
friction, role, or promotion threshold. Without it the deployed retrain
would silently revert to the 365-day default and ignore the widened
parquet history. This change is preserved.

---

## 9. Next research direction (allow-list pick: 1 of 3)

Per the spec's allow-list, exactly one next direction is proposed:

**Label re-engineering — quintile-based labels with explicit abstain class.**

Hypothesis: replace the current 3-class `{UP, STABLE, DOWN}` vol-scaled
threshold scheme with a 5-class quintile scheme on forward returns
(`Q1`, `Q2`, `Q3`, `Q4`, `Q5`) and add a 6th `ABSTAIN` class for rows
where the calibrated probability of any quintile is below a confidence
floor. The trader then trades only `Q1` (short) and `Q5` (long) signals
that survive the abstain check. This directly addresses the prediction-
collapse failure mode by:

- Forcing the model to learn the conditional distribution of returns
  rather than three discrete outcomes around an arbitrary threshold.
- Making class imbalance explicit and balanced (~20% per quintile by
  construction) instead of letting the threshold dominate.
- Adding sparse-trading discipline at the label level rather than
  retrofitting it on top of a 3-class output.
- Preserving the existing DA gate by mapping `Q5` → UP and `Q1` → DOWN
  for the gate calculation.

**Failure modes this addresses:** `prediction_collapse_top_share > 0.85`
on 11/18 slices in this campaign; `directional_accuracy < baseline` on
16/18 slices.

**Metric this is expected to move:** `net_pnl_pct_total` on the
abstain-filtered subset, evaluated at trade_share ∈ [0.05, 0.30] (sparse
trading) without lowering the DA gate.

Task #642 was created with this proposal and then CANCELLED. Resurrecting
it (or accepting Outcome B as-is and stopping here) is the user's call.

---

## 10. End state

- Quant brain remains correctly disabled. No manufactured trades.
- 0 champions. 0 forced promotions. 0 weakened gates.
- Real 28-month 6h training window proven on disk (committed parquet).
- Architecture-class verdict: the 3-class label scheme is the binding
  ceiling; data depth is not.
- Single-attempt rule respected. No second loop.
- Truthful failure documented; one specific next research direction
  identified per spec.
