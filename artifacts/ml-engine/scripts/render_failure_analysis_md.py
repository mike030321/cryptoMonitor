"""Render artifacts/ml-engine/reports/20260423T000000Z-failure-analysis.md
from the JSON produced by compute_failure_metrics.py."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JSON_PATH = REPO / "artifacts" / "ml-engine" / "reports" / "20260423T000000Z-failure-analysis.json"
MD_PATH = REPO / "artifacts" / "ml-engine" / "reports" / "20260423T000000Z-failure-analysis.md"


def fmt(x: float | None, digits: int = 4) -> str:
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "—"


def main() -> None:
    r = json.loads(JSON_PATH.read_text())
    slices = r["slices"]

    out: list[str] = []
    out.append("# Quant Verification Gate Failure Analysis — 2026-04-23")
    out.append("")
    out.append(f"- **Source verification report:** `{r['source_verification_report']}`")
    out.append(f"- **Source counts:** {json.dumps(r['source_verification_counts'])}")
    out.append(f"- **Enrichment source:** `{r['enrichment_source']}` (generated {r['enrichment_source_generated_at']})")
    out.append(f"- **Gate constants:** {json.dumps(r['gate_constants'])}")
    out.append("")
    out.append("> **Caveat:** " + r["enrichment_caveat"])
    out.append("")

    out.append("## 1. Bucket assignment summary")
    out.append("")
    out.append("Strict assignment rules (priority order, exactly as implemented in `compute_failure_metrics.assign_bucket`):")
    out.append("")
    out.append("1. `promoted` — DA ≥ 0.50 AND holdout ≥ 200 AND timeframe in tradeable set.")
    out.append("2. `salvageable_with_schema_fix` — `contamination_flag=True` OR cadence audit shows mixed source.")
    out.append("3. `insufficient_sample` — `status=untrained` OR `n_test < 200`.")
    out.append("4. `structurally_noisy_retire` — CONJUNCTION of all four:")
    out.append("   - cadence-clean (no contamination, no mixed cadence)")
    out.append("   - sufficient sample (`n_test ≥ 200`)")
    out.append("   - calibration broken (max per-class reliability deviation `≥ 0.10`)")
    out.append("   - prediction collapse (`collapse_gap ≥ 0.15` OR `predicted_top_class_share ≥ 0.85` OR `share_within_eps_of_prior ≥ 0.60`)")
    out.append("   Importance instability (rank-corr < 0.5 across folds) is corroborating evidence; not required because per-fold importances are not persisted today (see follow-up #316).")
    out.append("5. `salvageable_with_better_features_or_labels` — anything else (red gate but signal remaining).")
    out.append("")
    out.append("| Bucket | Count |")
    out.append("|---|---|")
    for k, v in sorted(r["bucket_counts"].items(), key=lambda kv: -kv[1]):
        out.append(f"| `{k}` | {v} |")
    out.append("")
    out.append("**Why `salvageable_with_schema_fix = 0` today:** `price_history` has no native-cadence column, so `contamination_flag` cannot be set by the trainer and is `false` everywhere. The cadence-audit proxy (inter-arrival-gap analysis on the labeled dataset) finds no mixed cadence in any per-coin slice. The schema-fix risk is preventive — see `20260423T000000Z-schema-audit.md` — and binds the moment task #306's CMC-daily and OKX-hourly backfill modules land.")
    out.append("")

    out.append("## 2. Bucket × timeframe matrix")
    out.append("")
    tfs = ["1m", "5m", "1h", "2h", "6h", "1d"]
    buckets = sorted({s["bucket"] for s in slices})
    out.append("| Timeframe | " + " | ".join(buckets) + " | total |")
    out.append("|" + "|".join(["---"] * (len(buckets) + 2)) + "|")
    for tf in tfs:
        cs = [s for s in slices if s["timeframe"] == tf]
        c = Counter(s["bucket"] for s in cs)
        out.append("| " + tf + " | " + " | ".join(str(c.get(b, 0)) for b in buckets) + f" | {len(cs)} |")
    pcs = [s for s in slices if s["coin_id"] == "__pooled__"]
    pc = Counter(s["bucket"] for s in pcs)
    out.append("| (pooled) | " + " | ".join(str(pc.get(b, 0)) for b in buckets) + f" | {len(pcs)} |")
    out.append("")

    out.append("## 3. Per-slice diagnostic detail")
    out.append("")
    out.append("All metrics below come from re-running the persisted `(model.txt + calibrators.joblib)` over the chronological 20% calibration holdout from the corresponding dataset parquet (matching `CALIBRATION_HOLDOUT_FRACTION=0.2` in `train.py`). Source DA / baseline_DA / n_test come from the original 22:34:31Z `baseline-verification.json`.")
    out.append("")
    for tf in tfs:
        out.append(f"### {tf}")
        out.append("")
        out.append("| Coin | Bucket | n_test | DA / baseline | Brier vs base | Pred-class share (D/S/U) | Confidence DA ≥0.6 | Net PnL %/trade | Top regime DA | Reliability max-dev | Notes |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        cs = [s for s in slices if s["timeframe"] == tf]
        cs.sort(key=lambda s: (s["coin_id"] != "__pooled__", s["coin_id"]))
        for s in cs:
            pcs_share = s.get("prediction_collapse", {}).get("predicted_class_share", {})
            ps = "/".join(fmt(pcs_share.get(c), 2) for c in ("DOWN", "STABLE", "UP")) if pcs_share else "—"
            cb = s.get("confidence_bucket_da", []) or []
            high_conf = [b for b in cb if b["lo"] >= 0.6 and b.get("da") is not None]
            high_conf_n = sum(b["n"] for b in high_conf)
            high_conf_da = (
                sum(b["da"] * b["n"] for b in high_conf) / high_conf_n
                if high_conf_n
                else None
            )
            high_conf_str = f"{fmt(high_conf_da, 3)} (n={high_conf_n})" if high_conf_n else "—"
            pnl = s.get("pnl_after_fees", {}) or {}
            pnl_str = f"{fmt(pnl.get('net_pct_mean'), 3)} (n={pnl.get('n_trades', 0)})" if pnl else "—"
            reg = s.get("regime_bucketed_da", {}) or {}
            top_reg = max(reg.items(), key=lambda kv: (kv[1].get("da") or 0)) if reg else None
            top_reg_str = f"{top_reg[0]} {fmt(top_reg[1].get('da'), 3)}" if top_reg else "—"
            rmd = s.get("reliability_max_dev_per_class", {}) or {}
            rmd_max = max(rmd.values()) if rmd else None
            rmd_str = fmt(rmd_max, 3) if rmd_max is not None else "—"
            br = s.get("metrics", {}).get("brier")
            bbr = s.get("baseline_metrics", {}).get("brier")
            br_str = f"{fmt(br, 3)} / {fmt(bbr, 3)}" if br is not None else "—"
            da = s.get("da_source")
            bda = s.get("baseline_da_source")
            da_str = f"{fmt(da, 3)} / {fmt(bda, 3)}" if da is not None else "—"
            n = s.get("n_test_source") or 0
            # Use the machine-readable bucket_reason emitted by assign_bucket
            # so the per-row narrative cannot drift from the classifier.
            note = s.get("bucket_reason") or ""
            if len(note) > 90:
                note = note[:87] + "…"
            out.append(f"| {s['coin_id']} | `{s['bucket']}` | {n} | {da_str} | {br_str} | {ps} | {high_conf_str} | {pnl_str} | {top_reg_str} | {rmd_str} | {note} |")
        out.append("")

    out.append("## 4. Smallest first set (per-coin 5m, ranked)")
    out.append("")
    out.append("Ranking score = `model_DA + 2 × max(0, lift)`. We pursue per-coin 5m first because the next-step lever (per-coin realized-vol-driven label thresholds, follow-up #318) attaches at the per-coin × per-timeframe level and these slices are closest to baseline_DA = 0.50 with a real holdout.")
    out.append("")
    out.append("| # | Coin | baseline_DA | model_DA | lift | n_test | rank_score | Repair |")
    out.append("|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(r["smallest_first_set"], 1):
        out.append(f"| {i} | {s['coin_id']} | {fmt(s['baseline_da'], 4)} | {fmt(s['model_da'], 4)} | {s['lift']:+.4f} | {s['n_test']} | {fmt(s['rank_score'], 4)} | {s['repair_action']} |")
    out.append("")

    out.append("## 5. Repair plan (no actual repairs run)")
    out.append("")
    out.append("Apply in priority order. The verification gate constants stay unchanged — `MIN_HOLDOUT_ROWS=200`, `MIN_DIRECTIONAL_ACCURACY=0.50` — and no synthetic fills are introduced.")
    out.append("")
    out.append("1. **Ship preventive schema fix before #306 retries.** New `price_candles` table per `20260423T000000Z-schema-audit.md`; trainer reads candles directly for any timeframe ≥ 5m. Tracked as follow-up #317. Required to keep verification trustworthy once daily and hourly backfill modules land.")
    out.append("2. **Extend `report.py` to persist Brier / per-class breakdown / confidence-bucket DA / regime DA / PnL / per-fold feature importances during training.** Today these are partially derivable from persisted artifacts post-hoc (this report did so) but `feature_importance_stability` cannot be computed without per-fold importances, which fold_metrics does not store. Tracked as follow-up #316.")
    out.append("3. **Apply per-coin 5m label thresholds derived from realized 5m vol** to the smallest first set above (follow-up #318). Acceptance bar (gate unchanged): baseline DA ≥ 0.55 AND model DA > baseline DA + 0.01 AND model DA > 0.50 on holdout ≥ 200.")
    out.append("4. **Retire the slices in `structurally_noisy_retire`** (cadence-clean + sufficient sample but with both broken calibration and predictions parked near the class prior), OR redefine the label scheme entirely (e.g., binary up-vs-not, multi-horizon). The 3-class head on these slices has demonstrably nothing to learn under the current label thresholds — per-coin threshold tuning will not rescue a collapsed, miscalibrated head.")
    out.append("5. **Wait for more data on 6h and 1d.** 6h holdout is below the 200-row floor (median ≈ 36); 1d is untrained (live-poll ticks have <35 distinct daily closes per coin). No code change required — the slices will graduate as the data window grows.")
    out.append("")

    out.append("## 6. Per-class collapse and PnL — at-a-glance for the 5m cohort")
    out.append("")
    out.append("| Coin | label_top_share | pred_top_share | collapse_gap | high-conf rows ≥ 0.6 | high-conf DA | net PnL %/trade | round-trip cost % |")
    out.append("|---|---|---|---|---|---|---|---|")
    for s in [s for s in slices if s["timeframe"] == "5m" and s["coin_id"] != "__pooled__"]:
        pc = s.get("prediction_collapse", {}) or {}
        cb = s.get("confidence_bucket_da", []) or []
        hc = [b for b in cb if b["lo"] >= 0.6 and b.get("da") is not None]
        hc_n = sum(b["n"] for b in hc)
        hc_da = sum(b["da"] * b["n"] for b in hc) / hc_n if hc_n else None
        pnl = s.get("pnl_after_fees", {}) or {}
        out.append(
            f"| {s['coin_id']} | {fmt(pc.get('label_top_class_share'), 3)} | {fmt(pc.get('predicted_top_class_share'), 3)} | {fmt(pc.get('collapse_gap'), 3)} | {hc_n} | {fmt(hc_da, 3)} | {fmt(pnl.get('net_pct_mean'), 3)} | {fmt(pnl.get('round_trip_cost_pct'), 3)} |"
        )
    out.append("")

    out.append("## 6b. Train-vs-holdout class balance and near-prior diagnostics")
    out.append("")
    out.append("`l1_drift` = sum |hold_share − train_share| over the three label classes; high drift means the holdout's regime differs from training. `share_within_eps_of_prior` is the fraction of holdout rows whose predicted distribution is within L1=0.05 of the training class prior — a high share means the calibrated head is essentially outputting the prior. `max_prob_std` is the std-dev of the predicted top-class probability across rows; near-zero values flag a model that has parked on the prior.")
    out.append("")
    out.append("| Slice | TvH l1_drift | Δ DOWN | Δ STABLE | Δ UP | share within ε of prior | max_prob_mean | max_prob_std |")
    out.append("|---|---|---|---|---|---|---|---|")
    for s in slices:
        if s["coin_id"] == "__pooled__":
            continue
        drift = s.get("train_vs_holdout_class_balance_drift", {}) or {}
        near = s.get("predictions_near_prior", {}) or {}
        spread = s.get("predicted_prob_spread", {}) or {}
        if not drift and not near and not spread:
            continue
        out.append(
            f"| {s['coin_id']} {s['timeframe']} | {fmt(drift.get('l1_drift'), 3)} | {fmt(drift.get('DOWN'), 3)} | {fmt(drift.get('STABLE'), 3)} | {fmt(drift.get('UP'), 3)} | {fmt(near.get('share_within_eps'), 3)} | {fmt(spread.get('max_prob_mean'), 3)} | {fmt(spread.get('max_prob_std'), 3)} |"
        )
    out.append("")

    out.append("## 7. Reproducibility")
    out.append("")
    out.append("This report is regenerated end-to-end by:")
    out.append("")
    out.append("```bash")
    out.append("pnpm --filter @workspace/ml-engine exec python scripts/compute_failure_metrics.py")
    out.append("pnpm --filter @workspace/ml-engine exec python scripts/render_failure_analysis_md.py")
    out.append("```")
    out.append("")
    out.append("The cadence-correctness contract for the schema fix is enforced by:")
    out.append("")
    out.append("- `artifacts/ml-engine/tests/test_cadence_correctness.py` — three behavior tests:")
    out.append("  - `test_daily_rows_are_not_silently_merged_into_5m_bars` — feeds two daily rows into `resample_to_candles(bucket_ms=300_000)` and asserts `CadenceMismatchError` is raised. Fails today (function silently buckets).")
    out.append("  - `test_resample_quarantines_coarser_rows_within_a_bucket` — feeds four 30s ticks at $100 plus a daily $999 row in the same 5m bucket and asserts the close is $100. Fails today (returns $999).")
    out.append("  - `test_trainer_provenance_records_native_cadence_and_refuses_mixed` — asserts persisted manifests carry `bars_by_native_cadence` + `cadence_mixed` AND that the verification gate refuses to promote cadence-mixed unmitigated slices. Fails today (no field, no helper).")
    out.append("- `artifacts/api-server/test/price-candles-uniqueness.test.ts` — three node-test assertions over the Drizzle schema package: (a) `priceCandlesTable` exported, (b) carries `(coin_id,timeframe,bucket_start,source)` columns, (c) `priceHistoryTable` must NOT gain a `timeframe` column. (a)+(b) fail today; (c) is a passing regression guard.")
    out.append("")
    out.append("## 8. What is NOT measurable from existing artifacts")
    out.append("")
    out.append("- **Per-fold feature-importance stability** — `fold_metrics` does not persist the per-fold LightGBM importance arrays (only `best_params` and CV metrics). `feature_importance_stability.status` is therefore `deferred` for every slice. Closing this gap requires extending `train.py` to write each fold's importance vector into `fold_metrics`. This is captured in follow-up #316; until that lands, importance-stability cannot be computed retrospectively.")
    out.append("- **`regime_subset` block in source report** — the trainer wrote an empty list for every slice. Regime-bucketed DA in this report instead comes from re-running inference and grouping by the `regime` column on the dataset parquet, which is the same regime label the trainer would have used.")
    out.append("")

    MD_PATH.write_text("\n".join(out))
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()
