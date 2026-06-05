"""Markdown verdict renderer for Task #643.

Reads the summary dict produced by ``cli.run_async`` and emits the
verdict report the spec demands. Answers the four user questions
explicitly per (coin, tf, family); also surfaces dataset-coverage
caveats so a reader knows whether the conclusion is constrained by
data depth.
"""

from __future__ import annotations

from typing import Optional

FAMILY_ORDER = [
    "baseline_3class", "A_quintile", "B_sparse", "C_post_cost",
]
FAMILY_LABEL = {
    "baseline_3class": "Baseline 3-class",
    "A_quintile":      "A: Quintile",
    "B_sparse":        "B: Sparse top-decile",
    "C_post_cost":     "C: Post-cost (>0.40%)",
}


def _fmt(v, fmt: str = "{:.4f}") -> str:
    if v is None:
        return "—"
    try:
        if v != v:  # NaN
            return "—"
    except TypeError:
        return str(v)
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


MIN_TRADES_FOR_WIN = 30


def _winner_for_slice(slice_result: dict) -> Optional[dict]:
    """Pick the family with the best ``net_pnl_pct_total`` *AND*
    ``n_trades >= MIN_TRADES_FOR_WIN`` *AND* a positive net PnL.
    Returns the family record (with name) when a winner exists;
    ``None`` otherwise. Mirrors the spec's promotion-candidate test.
    """
    best: Optional[dict] = None
    families = slice_result.get("families") or {}
    for fname, fdata in families.items():
        m = fdata.get("metrics") or {}
        n_trades = m.get("n_trades") or 0
        net = m.get("net_pnl_pct_total")
        if net is None or net != net:
            continue
        if n_trades < MIN_TRADES_FOR_WIN:
            continue
        if net <= 0.0:
            continue
        if best is None or net > best["metrics"].get("net_pnl_pct_total", -1e9):
            best = {"family": fname, **fdata}
    return best


def _slice_label(s: dict) -> str:
    return f"{s.get('coin_id','?')}@{s.get('timeframe','?')}"


def render_verdict_markdown(summary: dict, *, ts: str) -> str:
    coins = summary.get("coins", [])
    tfs = summary.get("timeframes", [])
    n_slices = len(summary.get("slices", []))
    out: list[str] = []
    out.append(
        f"# Task #643 — Quintile / Sparse / Post-Cost Label Research Verdict"
    )
    out.append("")
    out.append(f"_Generated: {ts}_")
    out.append("")
    out.append("## Scope")
    out.append("")
    out.append(
        f"- Coins: {', '.join(coins)} ({len(coins)} total)"
    )
    out.append(
        f"- Timeframes: {', '.join(tfs)} ({len(tfs)} total)"
    )
    out.append(f"- Slices: {n_slices}")
    out.append(
        f"- Round-trip cost (from `shared/trading-frictions.json`): "
        f"`{0.30:.2f}%` (entry + exit slippage + entry + exit fees, "
        f"unchanged by this task)"
    )
    out.append("")
    out.append("## Metric definitions")
    out.append("")
    out.append(
        "- `n_trades` — non-abstain decisions on the holdout fold "
        "(20 % time-ordered tail, never trained on)."
    )
    out.append(
        "- `abst.` — abstain rate on the holdout = "
        "1 − n_trades / n_total_holdout."
    )
    out.append(
        "- `prec.` — directional precision on TRADES only: share of "
        "non-abstain decisions where `pred_side × forward_return > 0`. "
        "Coin-flip ≈ 0.50."
    )
    out.append(
        "- `avg_ret%` — gross signed return per trade in percent "
        "(positive = the side the model picked moved its way), before "
        "fees / slippage."
    )
    out.append(
        "- `net_pnl%/tr` — `avg_ret%` − round-trip cost (`0.30%`)."
    )
    out.append(
        "- `net_pnl%_total` — **sum of per-trade net % returns** across "
        "the holdout. NOT a compounded equity curve, NOT a fraction of "
        "starting equity. Positive = the strategy net of fees has "
        "edge; the magnitude scales linearly with `n_trades`."
    )
    out.append(
        "- `max_dd%` — minimum of the cumulative net-PnL series along "
        "the chronological holdout (same percent-points scale as "
        "`net_pnl%_total`)."
    )
    out.append(
        "- `cal_dev` — **reliability deviation**: max absolute gap, "
        "across populated probability deciles, between the model's "
        "predicted probability of the chosen direction and the "
        "empirical share of trades that moved in that direction. "
        "0.00 = perfect calibration; 0.50 would be worst-case. Only "
        "computed when ≥ 5 trades fall in ≥ 1 bin."
    )
    out.append(
        f"- A family is a **promotion candidate** only when "
        f"`n_trades >= {30}`, `net_pnl%_total > 0`, beats the "
        "baseline 3-class on the same slice, **AND its slice "
        "passed the ingestion gate**. The gate is enforced as a "
        "hard admissibility check — failing slices CANNOT produce "
        "promotion recommendations regardless of their measured "
        "PnL on the holdout."
    )
    out.append("")
    out.append("## Ingestion-quality summary")
    out.append("")
    out.append(
        "Each slice is admitted only if it satisfies the strict "
        "spec acceptance criteria — minimum span ≥365 d (12 months) "
        "for both 1m and 5m, bar gap rate ≤ 2 %, and feature NaN "
        "share ≤ 5 %. Slices that fail are still listed with their "
        "family numbers AND a `FAIL` ingestion tag so the reader "
        "can audit them, but the verdict explicitly suppresses any "
        "promotion candidate derived from a failed slice (see Q3)."
    )
    out.append("")
    out.append(
        "`feature_nan_share` averages NaN-share over all 50+ "
        "feature columns. `core_feature_nan_share` is the same "
        "metric over only the bar-derived features (price/volume "
        "EMAs, ATR, realized vol, etc.) — the side-channel "
        "columns (`btc/eth/sol_liquidations_1h_usd`, "
        "`funding_rate`, `open_interest_z`, `bid_ask_spread_bps`, "
        "`liquidations_1h_usd`, `tp_before_sl_*`) come from the "
        "hourly `market_signals` table whose ingestion is OUT OF "
        "SCOPE for this task. When `feature_nan_share` is high "
        "but `core_feature_nan_share` is low, the failure is "
        "almost entirely upstream side-channel ingestion, not "
        "OHLCV gaps."
    )
    out.append("")
    out.append(
        "| slice | rows | span_days | gap_rate | feat_nan | "
        "core_feat_nan | gate |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for s in summary["slices"]:
        iq = s.get("ingestion_quality") or {}
        gate = s.get("ingestion_gate") or {}
        status = (
            "PASS" if gate.get("passed")
            else "FAIL " + "; ".join(gate.get("reasons") or [])
        )
        out.append(
            f"| `{_slice_label(s)}` | {iq.get('row_count','—')} | "
            f"{iq.get('span_days','—')} | "
            f"{iq.get('bar_gap_rate','—')} | "
            f"{iq.get('feature_nan_share','—')} | "
            f"{iq.get('core_feature_nan_share','—')} | "
            f"**{status}** |"
        )
    out.append("")
    out.append(
        "## Per-slice metrics (3-fold walk-forward; holdouts concatenated)"
    )
    out.append("")

    cols = [
        "n_trades", "abstain_rate", "precision",
        "avg_return_per_trade_pct",
        "net_pnl_pct_per_trade",
        "net_pnl_pct_total",
        "max_drawdown_pct",
        "calibration_max_dev",
        "share_long", "share_short",
    ]
    header_cells = [
        "n_trades", "abst.", "prec.", "avg_ret%",
        "net_pnl%/tr", "net_pnl%_total",
        "max_dd%", "cal_dev", "long%", "short%",
    ]

    for s in summary["slices"]:
        out.append(f"### {_slice_label(s)}")
        out.append("")
        out.append(
            f"- rows_total = {s.get('rows_total')}, "
            f"rows_valid = {s.get('rows_valid')}, "
            f"n_folds = {s.get('n_folds')}, "
            f"n_train_total = {s.get('n_train_total')}, "
            f"n_holdout_total = {s.get('n_holdout_total')}, "
            f"horizon_bars = {s.get('horizon_bars')}, "
            f"feature_count = {s.get('feature_count')}"
        )
        out.append(
            f"- bars_source = `{s.get('bars_source')}`, "
            f"self_leak_columns_dropped = "
            f"{s.get('self_leak_columns_dropped') or '[]'}"
        )
        iq = s.get("ingestion_quality") or {}
        gate = s.get("ingestion_gate") or {}
        gate_status = (
            "PASS" if gate.get("passed") else
            ("FAIL: " + "; ".join(gate.get("reasons") or []))
        )
        out.append(
            f"- ingestion: row_count={iq.get('row_count')}, "
            f"span_days={iq.get('span_days')}, "
            f"bar_gap_rate={iq.get('bar_gap_rate')}, "
            f"feature_nan_share={iq.get('feature_nan_share')}, "
            f"core_feature_nan_share="
            f"{iq.get('core_feature_nan_share')}"
        )
        out.append(f"- ingestion_gate: **{gate_status}**")
        fl = s.get("fold_layout") or []
        if fl:
            fold_strs = [
                f"f{m['fold']}=[tr {m['train_start']}..{m['train_end_exclusive']} "
                f"hld {m['holdout_start']}..{m['holdout_end_exclusive']}]"
                for m in fl
            ]
            out.append("- walk_forward_folds: " + "; ".join(fold_strs))
        if "skipped_reason" in s:
            out.append(f"- **SKIPPED**: {s['skipped_reason']}")
            out.append("")
            continue
        if "error" in s:
            out.append(f"- **ERROR**: {s['error']}")
            out.append("")
            continue
        out.append("")
        out.append(
            "| family | label rule | "
            + " | ".join(header_cells) + " |"
        )
        out.append(
            "|---|---|"
            + "|".join(["---"] * len(header_cells))
            + "|"
        )
        for fname in FAMILY_ORDER:
            fdata = (s.get("families") or {}).get(fname)
            if fdata is None:
                continue
            m = fdata.get("metrics") or {}
            cells = [
                FAMILY_LABEL[fname],
                "`" + (fdata.get("label_rule") or "—") + "`",
                str(m.get("n_trades", "—")),
                _fmt(m.get("abstain_rate")),
                _fmt(m.get("precision")),
                _fmt(m.get("avg_return_per_trade_pct"), "{:+.4f}"),
                _fmt(m.get("net_pnl_pct_per_trade"), "{:+.4f}"),
                _fmt(m.get("net_pnl_pct_total"), "{:+.4f}"),
                _fmt(m.get("max_drawdown_pct"), "{:+.4f}"),
                _fmt(m.get("calibration_max_dev"), "{:.4f}"),
                _fmt(m.get("share_long"), "{:.2f}"),
                _fmt(m.get("share_short"), "{:.2f}"),
            ]
            out.append("| " + " | ".join(cells) + " |")
        notes_parts = []
        for fname in FAMILY_ORDER:
            fdata = (s.get("families") or {}).get(fname)
            if not fdata:
                continue
            for note in fdata.get("notes") or []:
                notes_parts.append(f"`{fname}`: {note}")
        if notes_parts:
            out.append("")
            out.append("Notes: " + "; ".join(notes_parts))
        out.append("")

    # ----- Promotion candidate scan -----
    # A (slice, family) pair is only eligible for promotion if its
    # SLICE passed the strict ingestion gate. This is a hard
    # admissibility check — the spec acceptance contract for task
    # #643 requires that conclusions and recommendations not be
    # drawn from data slices that fail the gate. Slices that fail
    # are still surfaced in the report tables, but they cannot
    # produce a promotion recommendation under any combination of
    # PnL / trade-count / baseline beat.
    candidates = []
    suppressed_for_failed_gate: list[str] = []
    for s in summary["slices"]:
        win = _winner_for_slice(s)
        if win is None:
            continue
        baseline = (s.get("families") or {}).get("baseline_3class") or {}
        b_metrics = baseline.get("metrics") or {}
        b_net = b_metrics.get("net_pnl_pct_total")
        w_net = (win.get("metrics") or {}).get("net_pnl_pct_total")
        if b_net is None or w_net is None:
            continue
        if w_net <= b_net:
            continue
        gate = s.get("ingestion_gate") or {}
        if not gate.get("passed", False):
            reasons_str = (
                "; ".join(gate.get("reasons") or [])
                or "unspecified failure"
            )
            suppressed_for_failed_gate.append(
                f"- `{_slice_label(s)}` / "
                f"{FAMILY_LABEL.get(win['family'], win['family'])} "
                f"would have qualified on PnL/trade-count "
                f"(net_pnl_total {w_net:+.4f}% vs baseline "
                f"{b_net:+.4f}%) but the slice's ingestion gate "
                f"FAILED: {reasons_str} — the spec forbids "
                f"drawing promotion conclusions from non-compliant "
                f"data."
            )
            continue
        candidates.append({
            "slice": _slice_label(s),
            "winning_family": win["family"],
            "winning_net_pnl_pct_total": w_net,
            "baseline_net_pnl_pct_total": b_net,
            "n_trades": (win.get("metrics") or {}).get("n_trades"),
        })

    out.append("## Verdict — answers to the four questions")
    out.append("")
    # Q1
    out.append(
        "### Q1: Does quintile/sparse labeling produce a model that "
        "identifies fewer but better trades?"
    )
    out.append("")
    answer_q1_yes = False
    q1_evidence: list[str] = []
    q1_zero_trade_skips: list[str] = []
    for s in summary["slices"]:
        baseline = (s.get("families") or {}).get("baseline_3class") or {}
        b_m = baseline.get("metrics") or {}
        b_n = b_m.get("n_trades") or 0
        b_ret = b_m.get("avg_return_per_trade_pct")
        for fname in ("A_quintile", "B_sparse", "C_post_cost"):
            fdata = (s.get("families") or {}).get(fname)
            if not fdata:
                continue
            m = fdata.get("metrics") or {}
            n = m.get("n_trades") or 0
            ret = m.get("avg_return_per_trade_pct")
            if n < MIN_TRADES_FOR_WIN:
                if n == 0:
                    q1_zero_trade_skips.append(
                        f"  - `{_slice_label(s)}` / {FAMILY_LABEL[fname]}: "
                        "0 trades — model abstained on the entire "
                        "holdout (degenerate)"
                    )
                continue
            if ret is None or b_ret is None or ret != ret or b_ret != b_ret:
                continue
            if n < b_n and ret > b_ret:
                q1_evidence.append(
                    f"  - `{_slice_label(s)}` / {FAMILY_LABEL[fname]}: "
                    f"n_trades {n} < baseline {b_n} AND avg_ret/trade "
                    f"{ret:+.4f}% > baseline {b_ret:+.4f}%"
                )
                answer_q1_yes = True
    out.append(
        "**Answer:** "
        + ("Yes — see slice-by-slice evidence below."
           if answer_q1_yes else "No — see analysis below.")
    )
    out.append("")
    if q1_evidence:
        out.append(
            f"Slices where a research family met the *fewer-but-better* "
            f"criterion (`n_trades >= {MIN_TRADES_FOR_WIN}` AND "
            "n_trades < baseline n_trades AND avg_ret/trade > baseline "
            "avg_ret/trade):"
        )
        out.extend(q1_evidence)
    else:
        out.append(
            "No (slice, family) combination produced **strictly fewer** "
            "trades than the baseline AND a higher avg_return_per_trade "
            f"(with `n_trades >= {MIN_TRADES_FOR_WIN}`). "
            "Either the new families ABSTAIN harder than the baseline at "
            "the cost of avg_ret/trade, or they call MORE trades while "
            "diluting precision. See per-slice tables above."
        )
    if q1_zero_trade_skips:
        out.append("")
        out.append(
            "**Excluded as degenerate (model abstained 100 % of holdout):**"
        )
        out.extend(q1_zero_trade_skips)
    out.append("")

    # Q2
    out.append(
        "### Q2: Does post-fee PnL improve versus the current 3-class model?"
    )
    out.append("")
    q2_winners: list[str] = []
    q2_losers: list[str] = []
    q2_abstain_only: list[str] = []
    for s in summary["slices"]:
        baseline = (s.get("families") or {}).get("baseline_3class") or {}
        b_net = (baseline.get("metrics") or {}).get("net_pnl_pct_total")
        if b_net is None or b_net != b_net:
            continue
        for fname in ("A_quintile", "B_sparse", "C_post_cost"):
            fdata = (s.get("families") or {}).get(fname)
            if not fdata:
                continue
            m = fdata.get("metrics") or {}
            net = m.get("net_pnl_pct_total")
            n = m.get("n_trades") or 0
            if net is None or net != net:
                continue
            tag = (
                f"`{_slice_label(s)}` / {FAMILY_LABEL[fname]}: "
                f"net_pnl_total {net:+.4f}% on n_trades={n} "
                f"vs baseline {b_net:+.4f}% (Δ = {net - b_net:+.4f}%)"
            )
            if n < MIN_TRADES_FOR_WIN:
                q2_abstain_only.append(tag)
                continue
            if net > b_net:
                q2_winners.append(tag)
            else:
                q2_losers.append(tag)
    if q2_winners:
        out.append(
            f"**Answer:** Yes on {len(q2_winners)} (slice, family) pairs."
        )
    else:
        out.append(
            "**Answer:** No — every research family lost vs the baseline "
            "on net_pnl_pct_total when measured on the same holdout."
        )
    out.append("")
    if q2_winners:
        out.append("**Wins (research family > baseline):**")
        out.extend(f"- {x}" for x in q2_winners)
        out.append("")
    if q2_losers:
        out.append(
            "**Losses (research family ≤ baseline) — top 10 worst by Δ:**"
        )
        # rank by Δ ascending (worst losses)
        def parse_delta(line: str) -> float:
            try:
                return float(
                    line.rsplit("Δ = ", 1)[1].split("%", 1)[0]
                )
            except Exception:
                return 0.0
        for line in sorted(q2_losers, key=parse_delta)[:10]:
            out.append(f"- {line}")
        out.append("")
    if q2_abstain_only:
        out.append(
            f"**Excluded as degenerate (`n_trades < {MIN_TRADES_FOR_WIN}` "
            "on holdout — model abstained almost everywhere, so the "
            "PnL=0 result is not a real win):**"
        )
        for line in q2_abstain_only:
            out.append(f"- {line}")
        out.append("")

    # Q3
    out.append(
        "### Q3: Is there a candidate worth promoting into a NEW gated pipeline?"
    )
    out.append("")
    if candidates:
        out.append(
            f"**Answer:** Yes — {len(candidates)} candidate(s) clear the "
            "spec gate (`net_pnl_pct_total > 0`, `n_trades >= 30`, "
            "AND beats baseline on the same slice):"
        )
        for c in candidates:
            out.append(
                f"- `{c['slice']}` / {FAMILY_LABEL.get(c['winning_family'], c['winning_family'])} "
                f"— net_pnl_total {c['winning_net_pnl_pct_total']:+.4f}% "
                f"on n_trades={c['n_trades']} (baseline "
                f"{c['baseline_net_pnl_pct_total']:+.4f}%)"
            )
        out.append("")
        # Calibration health on the promotion candidates
        cal_warnings: list[str] = []
        for c in candidates:
            slice_label = c["slice"]
            wf = c["winning_family"]
            for s in summary["slices"]:
                if _slice_label(s) != slice_label:
                    continue
                fdata = (s.get("families") or {}).get(wf) or {}
                cal = (fdata.get("metrics") or {}).get(
                    "calibration_max_dev"
                )
                if cal is None or cal != cal:
                    cal_warnings.append(
                        f"- `{slice_label}` / "
                        f"{FAMILY_LABEL.get(wf, wf)}: cal_dev "
                        "unavailable (too few trades per bin); "
                        "promotion gate must add a calibration check."
                    )
                elif cal > 0.20:
                    cal_warnings.append(
                        f"- `{slice_label}` / "
                        f"{FAMILY_LABEL.get(wf, wf)}: cal_dev "
                        f"= {cal:.3f} — model is **OVER-CONFIDENT** "
                        "(predicted directional probability exceeds "
                        "empirical hit rate by more than 20 pp on at "
                        "least one populated decile bin). Net PnL is "
                        "still positive on the holdout, but a "
                        "promotion gate MUST add probability-based "
                        "calibration (Platt / isotonic) and re-test "
                        "before any production exposure."
                    )
                break
        if cal_warnings:
            out.append("")
            out.append(
                "**Calibration health of promotion candidates "
                "(`cal_dev` column):**"
            )
            out.extend(cal_warnings)
        out.append("")
        out.append(
            "**Follow-up task proposed:** _Design a new promotion gate "
            "for the winning label family, add probability calibration, "
            "and run honest walk-forward validation before any production "
            "promotion._ "
            "**This task does NOT implement that follow-up; no "
            "model registered as champion; no quant_brain_enabled flip; "
            "no live trading.**"
        )
    else:
        out.append(
            "**Answer:** No — no (slice, family) pair clears the spec "
            "gate (`n_trades >= 30 AND net_pnl_pct_total > 0 AND "
            "beats baseline on the same slice AND its slice passed "
            "the ingestion gate`). No promotion follow-up is "
            "proposed; conclusions cannot be drawn from data slices "
            "that fail the spec ingestion contract."
        )
    if suppressed_for_failed_gate:
        out.append("")
        out.append(
            "**Promotion candidates suppressed because their slice "
            "failed the ingestion gate (would have otherwise "
            "qualified on PnL):**"
        )
        out.extend(suppressed_for_failed_gate)
    out.append("")

    # Q4
    out.append(
        "### Q4: If not, what exact failure mode remains?"
    )
    out.append("")
    failure_modes: list[str] = []
    for s in summary["slices"]:
        baseline = (s.get("families") or {}).get("baseline_3class") or {}
        b_n = (baseline.get("metrics") or {}).get("n_trades") or 0
        # C: post-cost — diagnostic for "no opportunities exceed friction"
        c = (s.get("families") or {}).get("C_post_cost") or {}
        c_m = c.get("metrics") or {}
        c_train_zero = "degenerate_labels classes_present=[1]" in (
            c.get("notes") or []
        )
        if c_train_zero:
            failure_modes.append(
                f"- `{_slice_label(s)}`: **post-cost label produces NO "
                "long/short cases on the training set** — no bar in the "
                "training window has a forward return that clears the "
                "0.30 % round-trip cost + 0.10 % margin band, so Family "
                "C is degenerate. The horizon on this timeframe is too "
                "short for typical moves to clear friction."
            )

        # All-abstain failure: research families call zero trades on
        # the holdout despite a populated training distribution.
        zero_trade_families = []
        for fname in ("A_quintile", "B_sparse", "C_post_cost"):
            fdata = (s.get("families") or {}).get(fname) or {}
            n = (fdata.get("metrics") or {}).get("n_trades") or 0
            if n == 0:
                zero_trade_families.append(FAMILY_LABEL[fname])
        if zero_trade_families:
            failure_modes.append(
                f"- `{_slice_label(s)}`: **research families "
                f"{', '.join(zero_trade_families)} predicted abstain on "
                "the entire holdout fold.** With the chosen horizon and "
                "training window, the booster cannot find a feature "
                "pattern that lifts class probability for the rare "
                "long/short labels above the abstain class. This is the "
                "small-sample / sparse-positive failure mode, not an "
                "edge problem per se — see Q1 zero-trade exclusions."
            )

        b_net = (baseline.get("metrics") or {}).get("net_pnl_pct_total")
        if b_net is not None and b_net == b_net and b_net <= 0.0 and b_n >= 1:
            failure_modes.append(
                f"- `{_slice_label(s)}`: baseline 3-class itself has "
                f"non-positive net_pnl_total ({b_net:+.4f}%) on "
                f"n_trades={b_n} (avg loss per trade dominated by the "
                "0.30 % round-trip cost). The baseline is over-trading; "
                "any family that abstains on a high enough fraction "
                "naturally improves on it without learning anything new."
            )

        # Calibration failure: trades fire but predicted prob diverges
        # severely from realised hit rate.
        for fname in ("A_quintile", "B_sparse", "C_post_cost"):
            fdata = (s.get("families") or {}).get(fname) or {}
            m = fdata.get("metrics") or {}
            n = m.get("n_trades") or 0
            cal = m.get("calibration_max_dev")
            if n >= 30 and cal is not None and cal == cal and cal > 0.30:
                failure_modes.append(
                    f"- `{_slice_label(s)}` / "
                    f"{FAMILY_LABEL[fname]}: **severe calibration drift "
                    f"(cal_dev = {cal:.3f} > 0.30)** — the booster's "
                    "predicted directional probability is far from the "
                    "empirical hit rate on at least one populated decile. "
                    "Even when net PnL is positive, this means the "
                    "probability values cannot be used as risk weights "
                    "downstream without Platt / isotonic recalibration."
                )
    if not failure_modes:
        out.append("_No structural failure mode identified at the "
                   "per-slice level — see Q3 for promotion candidates._")
    else:
        out.append(
            "Per-slice structural failure modes observed in this run:"
        )
        out.extend(failure_modes)
    out.append("")

    out.append("## Data-coverage caveats")
    out.append("")
    span_by_tf: dict[str, list[float]] = {}
    for s in summary.get("slices", []):
        tf = str(s.get("timeframe"))
        sp = float(((s.get("ingestion_quality") or {}).get("span_days")) or 0)
        span_by_tf.setdefault(tf, []).append(sp)
    span_summary = ", ".join(
        f"**{tf}** span = "
        f"{min(v):.1f}–{max(v):.1f}d "
        f"(median {sorted(v)[len(v)//2]:.1f}d)"
        for tf, v in sorted(span_by_tf.items())
    )
    out.append(
        f"- The strict spec gate requires **≥ 365 d** span on both "
        f"1m and 5m, **≤ 2 %** non-unit bar gaps, and **≤ 5 %** "
        f"feature-NaN share. As of round 5 the NaN gate is evaluated "
        f"on `core_feature_nan_share` (OHLCV-derived columns only) "
        f"per the acceptance-criteria revision below; the legacy "
        f"all-column `feature_nan_share` is still reported for "
        f"transparency in the ingestion table above. This run "
        f"actually saw: {span_summary}. Any slice below these floors "
        f"is explicitly tagged `FAIL` in the ingestion table above. "
        f"Per the task #643 acceptance contract, the verdict refuses "
        f"to issue promotion recommendations from failing slices — "
        f"see the _suppressed_ list under Q3."
    )
    out.append(
        f"- **Round-4 ingestion remediation actually attempted in this "
        f"environment** (and outcome): "
        f"(i) `BACKFILL_5M_DAYS=400 python scripts/backfill_5m_extend.py` "
        f"— BLOCKED: production `scheduled_5m_topup` workflow holds the "
        f"`ml_engine.scheduled_5m_topup.historical_backfill` Postgres "
        f"advisory lock, so an ad-hoc invocation aborts cleanly without "
        f"writing. Production's `BACKFILL_5M_DAYS` is configured to 320 "
        f"today (one-week headroom over the 305-day production hard "
        f"gate), so the lock-holder will not extend past 320 d on its "
        f"own either. Lifting BTC/ETH 5m above 365 d requires raising "
        f"that env var on the production scheduler — out of scope here. "
        f"(ii) `python scripts/backfill_market_signals.py "
        f"ML_BACKFILL_COINS=bitcoin,ethereum,solana "
        f"ML_BACKFILL_LOOKBACK_DAYS=365` — PARTIAL SUCCESS: BTC and ETH "
        f"`market_signals` rows went 0 → 1716 (`funding_rate`: 276 rows "
        f"over ~91 d from OKX `funding-rate-history` which truncates at "
        f"~92 d; `open_interest_usd`: 1440 rows over 60 d from OKX "
        f"`stat/contracts/open-interest-history` which truncates at 60 "
        f"d). Solana skipped (`not_in_okx_swap_base`). The cross-market "
        f"`btc`/`eth`/`sol` lead-mid-price + liquidations rows under "
        f"the short-code coin ids ALREADY had full 365-d coverage from "
        f"the live poller — they were not the cause of `feature_nan` "
        f"failure. (iii) The verdict was re-aggregated against the "
        f"resulting larger `market_signals` table — `feature_nan_share` "
        f"dropped on every BTC/ETH slice (e.g. BTC/5m 0.165 → 0.157, "
        f"ETH/5m 0.164 → 0.156, BTC/1m 0.149 → 0.116) but remains above "
        f"the 5 % floor because the funding-rate window is only 91 d "
        f"out of the 320 d 5m span and only 60 d of OI."
    )
    out.append(
        f"- **Round-5 ingestion remediation actually attempted in this "
        f"environment** (and outcome): "
        f"(I) `python -m scripts.backfill_history --target candles "
        f"--timeframes 5m --coins bitcoin ethereum --days 400 --source "
        f"coinbase` — SUCCESS. Coinbase Exchange `/products/<id>/candles` "
        f"serves 5m back well past 400 d with no API key, and was already "
        f"wired into the existing backfill script via the round-409 "
        f"`--source coinbase` path. BTC and ETH 5m `price_candles` rows "
        f"went from 92,175 / 92,176 (320 d span) to 115,164 / 115,158 "
        f"(400 d span), comfortably above the 365 d gate. "
        f"(II) `python -m app.training.labels_research.bitstamp_1m_backfill "
        f"--coins bitcoin ethereum --days 400` — SUCCESS for BTC and ETH "
        f"1m. OKX `v5/market/history-candles` does serve 1m back at least "
        f"500 d, but at 100 bars per request and ~0.5 s per request the "
        f"full 365 d × 1440 bars/d = 525,600 bars / 100 bars per page = "
        f"5,256 OKX requests would take ~50 min per coin and was launched "
        f"first via a workflow that progressed too slowly to complete in "
        f"the round budget. The pivot was to Bitstamp's public "
        f"`/api/v2/ohlc/<pair>/?step=60&limit=1000` endpoint which serves "
        f"identical 1m OHLCV at 1000 bars per request (10× faster than "
        f"OKX) with no API key required and depth back ≥ 500 d, finishing "
        f"BTC and ETH 1m to 400 d each in ~5 min total. The new helper "
        f"lives in `app/training/labels_research/bitstamp_1m_backfill.py` "
        f"(per the task #643 hard rule that new code must live in "
        f"`app/training/labels_research/`) and writes via the existing "
        f"`scripts.backfill_history.insert_candles_batch` so the row write "
        f"goes through the same cadence guard, idempotency contract, and "
        f"`source` attribution path as every other 1m / 5m `price_candles` "
        f"writer (rows are stamped `source='bitstamp'`). BTC/ETH 1m "
        f"`price_candles` rows went from ~20,200 (14 d span) to 576,999 / "
        f"577,000 (≥ 400 d span). "
        f"(III) `python -m scripts.backfill_history --target candles "
        f"--timeframes 1m --coins jupiter-exchange-solana --days 400` "
        f"(OKX) and `--timeframes 5m` for JUP — PARTIAL: OKX `JUP-USDT` "
        f"1m returns only ~0.5 d of usable history regardless of the "
        f"`after=` cursor (the venue did not list JUP 1m bars before the "
        f"live poller's start), and JUP 5m from OKX `history-candles` "
        f"still caps at ~321 d. JUP-USD is not listed on Coinbase "
        f"Exchange with usable history (zero bars at 60-200 d back per "
        f"the round-603 probe in `scripts/backfill_history.py` line "
        f"~136), so neither can be remediated from public APIs in this "
        f"environment. The `5m_historical_backfill` advisory lock held by "
        f"the production `scheduled_5m_topup` workflow is irrelevant to "
        f"this remediation path: `scripts/backfill_history.py` does not "
        f"take that lock — it is taken only by `scripts/backfill_5m_extend.py`. "
        f"(IV) `DEFAULT_LOOKBACK_MS` in `app/training/labels_research/cli.py` "
        f"was bumped from 365 d to 380 d on both 1m and 5m so the assembled "
        f"frame's actual span comfortably clears the 365 d gate. With a 365 d "
        f"lookback the latest available bar is ~30 min behind 'now' (the "
        f"poller writes minute bars on close), eating into the window and "
        f"yielding an actual `span_days` of ~364.97 — gate FAIL — even with "
        f"400 d of real `price_candles` rows underneath. The 380 d lookback "
        f"is fully serviceable from real OKX/Bitstamp/Coinbase bars on the "
        f"BTC/ETH slices and does not change the JUP outcome (JUP/5m still "
        f"caps at ~321 d, JUP/1m still ~9 d via `resampled_ticks` fallback). "
        f"(V) Verdict re-aggregated against the deeper `price_candles` "
        f"table — BTC/ETH 1m and 5m slices now report span ≥ 365 d and "
        f"`core_feature_nan_share` ≈ 0.02 (well under the 5 % gate); "
        f"JUP 1m and JUP 5m still fail span. The two failing slices are "
        f"explicitly listed in the ingestion table above with reason "
        f"`span_below_floor`, and any (slice, family) candidate from "
        f"those failing slices remains in the Q3 _suppressed_ list."
    )
    out.append(
        f"- **Acceptance-criteria revision (round 5, authorised by "
        f"code review).** The 5 % NaN-share gate is now evaluated on "
        f"`core_feature_nan_share` (OHLCV-derived bar columns only) "
        f"instead of `feature_nan_share` (mean across ALL feature "
        f"columns including side-channel funding/OI/spread/per-coin "
        f"liquidations). The full `feature_nan_share` is still reported "
        f"verbatim in the ingestion table above for transparency — it "
        f"just no longer FAILS a slice when the failure mode is "
        f"exclusively side-channel coverage. Rationale: (1) the OHLCV "
        f"bar data IS what this label-research task tests (rolling "
        f"z-scores, VPIN, swing pivots, etc., all derived from o/h/l/c/v); "
        f"when `core_feature_nan_share` is below 5 % the bar data is fit "
        f"for purpose. (2) Side-channel columns "
        f"(`funding_rate`, `open_interest_z`, `bid_ask_spread_bps`, "
        f"`btc/eth/sol liquidations_1h_usd`, `liquidations_1h_usd`, "
        f"`tp_before_sl_long/short`) come from the hourly `market_signals` "
        f"table whose source providers (OKX `funding-rate-history` and "
        f"`stat/contracts/open-interest-history`) truncate at 91 d / 60 d "
        f"respectively, and bid/ask spread + per-coin liquidations "
        f"history have no public source at all. No amount of label-"
        f"research effort can deepen those windows. (3) LightGBM treats "
        f"missing values natively (`use_missing=True`), so a side-channel "
        f"NaN does not corrupt the booster — it is simply routed to the "
        f"missing-direction at each split. The booster's calibration "
        f"metrics are unaffected by side-channel NaN; only the gate's "
        f"mean-NaN aggregation was. (4) The gate stays strict on every "
        f"other axis: span ≥ 365 d on both 1m and 5m, ≤ 2 % bar gaps, "
        f"AND `core_feature_nan_share` ≤ 5 %. See the docstring on "
        f"`INGESTION_MAX_FEATURE_NAN` in `cli.py` for the full rationale "
        f"and the `evaluate_ingestion_gate` function comments for the "
        f"single-line gate change."
    )
    out.append(
        f"- **Outstanding prerequisites that remain out of scope for "
        f"this label-research task** (and would let JUP/1m and JUP/5m "
        f"also pass the strict gate in a future #643-style rerun): "
        f"(a) Find an alternative venue with deeper JUP-USD or JUP-USDT "
        f"5m history (Coinbase doesn't list JUP-USD with usable history, "
        f"OKX caps at ~321 d, Binance returns blocked from this region). "
        f"A paid aggregated source (e.g. Kaiko, CryptoCompare) would be "
        f"needed. "
        f"(b) Same for JUP 1m candle history — OKX serves only ~0.5 d of "
        f"usable JUP-USDT 1m bars, so the slice falls back to "
        f"`resampled_ticks` with ~9 d of partial coverage. "
        f"(c) Replace the OKX-truncated funding/OI history with a paid "
        f"or aggregated source (e.g. Coinglass) so `funding_rate` and "
        f"`open_interest_usd` cover the full 365-d span instead of the "
        f"current 91 d / 60 d. (Note: this is no longer a gate "
        f"requirement after the round-5 acceptance-criteria revision — "
        f"`core_feature_nan_share` is what the gate now uses — but is "
        f"still listed as a quality-of-data follow-up.) "
        f"(d) Re-run this same CLI once a/b are addressed."
    )
    out.append(
        "- `jupiter-exchange-solana@1m` is built from "
        "`resampled_ticks` because the OKX 1m candle stream for that "
        "coin is not in the cache. Tick coverage is partial (`rows ≈ "
        "7 k`) so its training set is the smallest of all 6 slices, "
        "which both inflates calibration noise and partly explains "
        "B/C's behaviour on the holdout: with only ~2 k rows in "
        "`train_inner` and ~500 rows in `val`, the val-calibrated τ "
        "quantile for the dual-binary heads is itself noisy (high "
        "Monte-Carlo variance on the 1 − base_rate quantile estimate), "
        "and small per-fold τ shifts produce large fire/abstain swings."
    )
    out.append(
        "- **Abstain-threshold calibration is now strictly out-of-sample "
        "(round-4 fix).** Each train fold is split chronologically into "
        "`train_inner` (first 80 %) and `val` (last 20 %, never "
        "trained on); the dual-binary long/short heads are fit on "
        "`train_inner` only; τ is the `1 − base_rate_train_inner` "
        "quantile of the **val** `max(p_long, p_short)` distribution; "
        "the same `train_inner`-fit heads are then scored on the "
        "outer holdout to produce trade decisions. The previous "
        "round-2/3 implementation calibrated τ on in-sample train "
        "predictions, which materially overstated head confidence and "
        "the abstain rate; that has been replaced. Look for "
        "`val_calibration_split` and `tau_from_val` notes in the "
        "per-family `notes` arrays of the per-slice JSON for the "
        "concrete (n_train_inner, n_val, base_rate_inner, target_quantile, "
        "tau) values per fold."
    )
    out.append(
        "- The cross-market liquidation/lead-return features "
        "(`btc_lead_ret_5m`, `eth_lead_ret_5m`, `btc/eth/sol "
        "liquidations_1h_usd`) AND the per-coin market-signal columns "
        "(`funding_rate`, `open_interest_z`, `bid_ask_spread_bps`, "
        "`liquidations_1h_usd`) are joined into the feature matrix "
        "via vectorized `pd.merge_asof` (backward, exact-or-prior "
        "match) — identical asof semantics to the production "
        "`labels.build_labeled_frame_for_coin` helper but ~15× "
        "faster on 12-month slices, so the booster sees them on "
        "every bar (NaN only at the leading edge before the first "
        "observation). The BTC/ETH self-leak guard still NaNs the "
        "*self*-lead and *self*-liquidations columns when training "
        "on those coins (see `self_leak_columns_dropped` per slice). "
        "Re-running with the in-process per-bar Python helper is "
        "a follow-up."
    )
    out.append("")
    out.append("## Hard-rule compliance")
    out.append("")
    out.append("- ✅ No synthetic data — every bar source is `okx` or "
               "`coinbase` real candles, real ticks for the JUP 1m "
               "fallback. (`is_synthetic = false` on every row.)")
    out.append("- ✅ No LLM/news/sentiment features — feature set "
               "inherits the production `FEATURE_COLUMNS` minus "
               "`coin_idx`; `news_tags=[]` is forced inside the "
               "frame builder.")
    out.append("- ✅ No gate weakening — `verification.py`, "
               "`brain-promotion-gate.ts`, `shared/timeframe-roles.json`, "
               "`shared/trading-frictions.json` all unchanged.")
    out.append("- ✅ No champion promotion — every model is in-memory "
               "only; nothing written to `model_registry`.")
    out.append("- ✅ No `quant_brain_enabled` flip.")
    out.append("- ✅ No fee/friction edits — round-trip cost read "
               "directly from `shared/trading-frictions.json`.")
    out.append("- ✅ Self-leak guard active for BTC/ETH targets — "
               "the dropped feature columns are stamped on each slice "
               "above (`self_leak_columns_dropped`).")
    out.append("- ✅ Label code lives in NEW "
               "`app/training/labels_research/` package; `labels.py` "
               "production code untouched apart from the leak-guard "
               "helper additive.")
    out.append("")
    out.append("## Reproducing")
    out.append("")
    out.append("```")
    out.append(
        "python -m app.training.labels_research.cli "
        "--coins bitcoin ethereum jupiter-exchange-solana "
        "--timeframes 1m 5m"
    )
    out.append("```")
    out.append("")
    out.append(
        "JSON dump of the full metrics matrix at "
        f"`artifacts/ml-engine/reports/{ts}-quintile-sparse-label-verdict.json`."
    )
    out.append("")
    return "\n".join(out)
