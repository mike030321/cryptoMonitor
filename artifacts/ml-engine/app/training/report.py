"""HTML report renderer for the Phase-2 training results.

No external templating dep — just an f-string. The report intentionally
shows fold-level metrics, per-coin/pooled split, a real per-class
reliability diagram, and a data-sufficiency banner so a non-technical
user can see whether the model has any edge before Phase 3 backtesting.
"""
from __future__ import annotations

import html
import json
import math
from typing import Optional

from .registry import REGISTRY_ROOT

REPORT_PATH = REGISTRY_ROOT / "report.json"
# Task #157 — rolling per-slice history of the magnitude head's holdout
# stats. Kept next to the report so the renderer can show a p95 trend
# even when only the latest training run is in `report.json`.
REGRESSION_HEAD_HISTORY_PATH = (
    REGISTRY_ROOT / "training_history" / "regression_head_stats.jsonl"
)
# How many of the most-recent samples per slice to render in the trend.
REGRESSION_HEAD_TREND_POINTS = 8


def _load_regression_head_history() -> dict[tuple[str, str], list[dict]]:
    """Load the rolling regression-head history grouped by (coin, timeframe).
    Each value is sorted oldest -> newest. Returns {} if the file is
    missing or unreadable — the report is best-effort and never fails on
    history rendering.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    if not REGRESSION_HEAD_HISTORY_PATH.exists():
        return grouped
    try:
        with REGRESSION_HEAD_HISTORY_PATH.open() as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = (str(rec.get("coin_id", "?")), str(rec.get("timeframe", "?")))
                grouped.setdefault(key, []).append(rec)
    except Exception:
        return grouped
    for key, rows in grouped.items():
        rows.sort(key=lambda r: str(r.get("generated_at") or ""))
    return grouped


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.4f}"
    return html.escape(str(v))


def _reliability_svg(diagram: list[dict]) -> str:
    """Per-class reliability diagram: x = mean predicted prob in bin, y =
    empirical fraction of positives. One colored series per class. Diagonal
    means perfectly calibrated.
    """
    if not diagram:
        return "<p><em>No calibration data (insufficient holdout).</em></p>"
    w, h = 320, 220
    pad = 32
    plot_w, plot_h = w - 2 * pad, h - 2 * pad

    def x(p): return pad + p * plot_w
    def y(p): return (h - pad) - p * plot_h

    diag = f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#9ca3af" stroke-dasharray="3,3"/>'
    color_for = {"DOWN": "#ef4444", "STABLE": "#9ca3af", "UP": "#10b981"}

    pts = []
    legend = []
    for cls, color in color_for.items():
        entries = sorted(
            (d for d in diagram if d.get("class") == cls),
            key=lambda d: d["mean_predicted"],
        )
        if not entries:
            continue
        line_pts = []
        for entry in entries:
            cx, cy = x(entry["mean_predicted"]), y(entry["fraction_positive"])
            r = max(2, min(7, int(math.sqrt(entry["n"]))))
            pts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" opacity="0.85">'
                f'<title>{cls} pred={entry["mean_predicted"]:.2f} actual={entry["fraction_positive"]:.2f} n={entry["n"]}</title>'
                f'</circle>'
            )
            line_pts.append(f"{cx:.1f},{cy:.1f}")
        if len(line_pts) > 1:
            pts.append(
                f'<polyline points="{" ".join(line_pts)}" stroke="{color}" fill="none" stroke-width="1.5" opacity="0.6"/>'
            )
        legend.append(f'<tspan fill="{color}">■ {cls}</tspan>')

    legend_text = " ".join(legend)
    axes = (
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#374151"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#374151"/>'
        f'<text x="{pad}" y="14" font-size="10" fill="#374151">Reliability — calibrated probability vs actual frequency</text>'
        f'<text x="{pad}" y="26" font-size="9">{legend_text}</text>'
        f'<text x="{w-pad}" y="{h-6}" font-size="9" fill="#6b7280" text-anchor="end">predicted prob.</text>'
        f'<text x="4" y="{pad+8}" font-size="9" fill="#6b7280">actual freq.</text>'
    )
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="border:1px solid #e5e7eb;">{axes}{diag}{"".join(pts)}</svg>'


def _regression_head_trend_svg(history: list[dict], threshold: Optional[float]) -> str:
    """Tiny inline sparkline of holdout p95(|pred|) over the last
    REGRESSION_HEAD_TREND_POINTS training runs for one slice. The
    horizontal dashed line is the slice's label-threshold floor — points
    that dip under it are colored red so a slow drift toward degeneracy
    (the bug task #135 fixed) is visible at a glance.
    """
    pts = [
        h for h in history
        if isinstance(h.get("abs_pred_p95_pct"), (int, float))
        and not math.isnan(float(h["abs_pred_p95_pct"]))
    ]
    pts = pts[-REGRESSION_HEAD_TREND_POINTS:]
    if len(pts) < 2:
        # Single sample (or none) — not a trend yet, skip the chart.
        return ""
    values = [float(p["abs_pred_p95_pct"]) for p in pts]
    thr = float(threshold) if isinstance(threshold, (int, float)) else None
    y_lo = min(values + ([thr] if thr is not None else []))
    y_hi = max(values + ([thr] if thr is not None else []))
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-6
    pad = 1
    w, h = 200, 48
    plot_w, plot_h = w - 2 * pad - 4, h - 2 * pad - 12

    def x(i): return pad + (i / max(1, len(values) - 1)) * plot_w
    def y(v): return (h - pad - 4) - ((v - y_lo) / (y_hi - y_lo)) * plot_h

    line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    dots = []
    for i, (v, p) in enumerate(zip(values, pts)):
        below = thr is not None and v < thr
        color = "#ef4444" if below else "#2563eb"
        dots.append(
            f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.5" fill="{color}">'
            f'<title>{p.get("generated_at","?")}: p95={v:.4f}%'
            f'{" (below threshold)" if below else ""}</title>'
            f'</circle>'
        )
    threshold_line = ""
    if thr is not None:
        threshold_line = (
            f'<line x1="{pad}" y1="{y(thr):.1f}" x2="{w - pad}" y2="{y(thr):.1f}"'
            f' stroke="#d97706" stroke-dasharray="3,3" stroke-width="1"/>'
        )
    label = (
        f'<text x="{pad}" y="10" font-size="9" fill="#6b7280">'
        f'p95 trend (last {len(values)}) — first {values[0]:.3f}% → latest {values[-1]:.3f}%'
        f'</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="margin-top:6px;">'
        f'{label}{threshold_line}'
        f'<polyline points="{line}" fill="none" stroke="#2563eb" stroke-width="1.5"/>'
        f'{"".join(dots)}'
        f'</svg>'
    )


def _regression_head_html(r: dict, history: Optional[list[dict]] = None) -> str:
    """Task #146 — render the magnitude regressor's holdout stats so a
    future retrain that collapses the regressor back near 0 (the bug
    task #135 fixed) is visible on the report. We compare p95(|pred|)
    against the timeframe's label-threshold floor and flag the row when
    it falls below — that's the canonical "magnitude head is degenerate"
    signal.
    """
    if not r.get("has_regression_head"):
        return (
            "<p class='meta'><em>No regression head on this slice "
            "(legacy or prior-only model).</em></p>"
        )
    stats = r.get("regression_head_stats") or {}
    if not stats:
        return (
            "<p class='meta'><em>Regression head trained but no holdout "
            "stats persisted (legacy manifest).</em></p>"
        )
    p50 = stats.get("abs_pred_p50_pct")
    p95 = stats.get("abs_pred_p95_pct")
    pmax = stats.get("abs_pred_max_pct")
    mae = stats.get("mae_pct")
    n_train = stats.get("n_train_rows")
    n_hold = stats.get("n_holdout_rows")
    best_iter = stats.get("best_iteration")
    threshold = r.get("threshold_pct")
    degenerate = (
        isinstance(p95, (int, float))
        and isinstance(threshold, (int, float))
        and not math.isnan(float(p95))
        and float(p95) < float(threshold)
    )
    flag = ""
    if degenerate:
        flag = (
            f"<p class='warn'>Magnitude head looks degenerate: holdout "
            f"p95(|pred|) = {p95:.4f}% &lt; label threshold "
            f"{threshold:.4f}%. /ml/predict's <code>expectedReturnPct</code> "
            f"will collapse toward 0 (the bug task #135 fixed). Inspect "
            f"the regressor before trusting downstream gates.</p>"
        )
    trend_svg = _regression_head_trend_svg(history or [], threshold) if history else ""
    return (
        "<h4 class='subhead'>Regression head (magnitude)</h4>"
        f"{flag}"
        "<div class='kv'>"
        f"<div><span>holdout p50 |pred|</span><b>{_fmt(p50)}%</b></div>"
        f"<div><span>holdout p95 |pred|</span><b>{_fmt(p95)}%</b></div>"
        f"<div><span>holdout max |pred|</span><b>{_fmt(pmax)}%</b></div>"
        f"<div><span>holdout MAE</span><b>{_fmt(mae)}%</b></div>"
        f"<div><span>label threshold</span><b>{_fmt(threshold)}%</b></div>"
        f"<div><span>n_train / n_holdout</span><b>{_fmt(n_train)} / {_fmt(n_hold)}</b></div>"
        f"<div><span>best iteration</span><b>{_fmt(best_iter)}</b></div>"
        "</div>"
        f"{trend_svg}"
    )


def _slice_html(label: str, r: dict, regression_history: Optional[list[dict]] = None) -> str:
    """Render one trained slice (per-coin or pooled)."""
    metrics = r.get("metrics", {})
    base = r.get("baseline_metrics", {})
    lift = r.get("lift_auc")
    rows_html = "".join(
        f"<tr><td>{f['fold']}</td><td>{f['n_train']}</td><td>{f['n_test']}</td>"
        f"<td>{_fmt(f['auc'])}</td><td>{_fmt(f['baseline_auc'])}</td>"
        f"<td>{_fmt(f['log_loss'])}</td><td>{_fmt(f['baseline_log_loss'])}</td>"
        f"<td>{_fmt(f['directional_accuracy'])}</td>"
        f"<td>{_fmt(f['baseline_directional_accuracy'])}</td></tr>"
        for f in r.get("fold_metrics", [])
    )
    means_pct = r.get("class_return_means_pct", [])
    means_str = (
        f"DOWN={_fmt(means_pct[0])}% STABLE={_fmt(means_pct[1])}% UP={_fmt(means_pct[2])}%"
        if len(means_pct) == 3 else "—"
    )
    gates = r.get("gates_alignment") or {}
    if gates:
        aligned_pct = gates["aligned_share"] * 100.0
        loud_quiet = gates["loud_classifier_quiet_regressor_share"] * 100.0
        quiet_loud = gates["quiet_classifier_loud_regressor_share"] * 100.0
        gates_html = (
            "<div class='gates'><b>Gates aligned:</b> "
            f"<span class='aligned'>{aligned_pct:.1f}%</span> of {gates['n']} "
            f"{html.escape(gates['source'])} predictions agree on trade/skip. "
            f"<span title='Classifier confident, regressor below cost floor — wasted classifier budget'>"
            f"loud-cls/quiet-reg: {loud_quiet:.1f}%</span> &middot; "
            f"<span title='Regressor screams, classifier near 50/50 — wasted regressor budget'>"
            f"quiet-cls/loud-reg: {quiet_loud:.1f}%</span> "
            f"<span class='meta'>(mde={_fmt(gates.get('min_directional_edge'))}, "
            f"mer={_fmt(gates.get('min_expected_return_pct'))}%)</span></div>"
        )
    else:
        gates_html = (
            "<div class='gates meta'>Gates alignment: <em>n/a</em> "
            "(no regressor head or insufficient holdout)</div>"
        )
    return f"""
<div class='slice'>
  <h3>{html.escape(label)} <span class='version'>v{html.escape(r.get('version', ''))}</span></h3>
  <p class='meta'>Rows: <b>{_fmt(r.get('n_rows'))}</b> &middot; Folds: <b>{_fmt(r.get('n_folds'))}</b> &middot; Class mean returns: <code>{means_str}</code></p>
  <div class='kv'>
    <div><span>LightGBM macro-AUC</span><b>{_fmt(metrics.get('auc'))}</b></div>
    <div><span>Baseline macro-AUC</span><b>{_fmt(base.get('auc'))}</b></div>
    <div><span>Lift</span><b>{_fmt(lift)}</b></div>
    <div><span>LightGBM log-loss</span><b>{_fmt(metrics.get('log_loss'))}</b></div>
    <div><span>Baseline log-loss</span><b>{_fmt(base.get('log_loss'))}</b></div>
    <div><span>Directional acc.</span><b>{_fmt(metrics.get('directional_accuracy'))}</b></div>
  </div>
  {gates_html}
  {_reliability_svg(r.get('calibration_diagram', []))}
  {_regression_head_html(r, regression_history)}
  <table>
    <thead><tr><th>fold</th><th>n_train</th><th>n_test</th>
      <th>AUC (lgb)</th><th>AUC (lr)</th>
      <th>logloss (lgb)</th><th>logloss (lr)</th>
      <th>dir.acc (lgb)</th><th>dir.acc (lr)</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""


def _section(
    tf: str,
    r: dict,
    regression_history: Optional[dict[tuple[str, str], list[dict]]] = None,
) -> str:
    status = r.get("status", "unknown")
    if status == "insufficient_data":
        return (
            f"<section><h2>{html.escape(tf)}</h2>"
            f"<p class='warn'>Insufficient data: only {_fmt(r.get('n_rows'))} rows; "
            f"need {_fmt(r.get('min_rows_required'))}. No model saved.</p></section>"
        )

    parts = [f"<section><h2>{html.escape(tf)}</h2>"]
    parts.append(
        f"<p class='meta'>Total rows: <b>{_fmt(r.get('n_rows'))}</b>"
        f" &middot; Dataset snapshot: <code>{html.escape(str(r.get('dataset_path', '')))}</code></p>"
    )
    pooled = r.get("pooled")
    per_coin = r.get("per_coin", {}) or {}
    trained_coins = [(c, slc) for c, slc in per_coin.items() if slc.get("status") == "trained"]
    fallback_coins = [c for c, slc in per_coin.items() if slc.get("status") != "trained"]

    history = regression_history or {}
    if trained_coins:
        parts.append("<h3 class='subhead'>Per-coin models</h3>")
        for coin, slc in trained_coins:
            parts.append(_slice_html(coin, slc, history.get((coin, tf))))
    else:
        parts.append("<p class='meta'>No per-coin model has enough rows yet — all coins served by pooled fallback.</p>")

    if pooled is not None and pooled.get("status") == "trained":
        parts.append(f"<h3 class='subhead'>Pooled fallback (serves: {html.escape(', '.join(fallback_coins) or 'none')})</h3>")
        parts.append(_slice_html("__pooled__", pooled, history.get(("__pooled__", tf))))
    elif pooled is not None and pooled.get("status") == "insufficient_data":
        parts.append(
            f"<p class='warn'>Pooled fallback also insufficient: "
            f"{_fmt(pooled.get('n_rows'))} rows. Coins without per-coin model "
            f"will receive 503 from /ml/predict.</p>"
        )

    # Phase 3 — per-specialist + per-regime block. We render a compact
    # comparison table so a human can eyeball, at a glance, whether any
    # specialist beats the pooled baseline on its own regime block.
    specialists = r.get("specialists") or {}
    if specialists:
        parts.append("<h3 class='subhead'>Specialists (Phase 3, observability-only)</h3>")
        parts.append(
            "<p class='meta'>Specialists are scored on a cost-aware target derived "
            "from the trade-aware label block (TP-before-SL barrier flags / opportunity "
            "score for directional kinds; tercile-bucketed |forward return| for the "
            "volatility forecaster). The directional accuracy reported here is the "
            "training-time 3-class metric. The diagnostics page's per-specialist "
            "accuracy uses a 2-way realized-return-sign comparison (UP vs DOWN) on "
            "the prediction journal — these two definitions are intentionally "
            "different and are not directly comparable.</p>"
        )
        pooled_auc = (pooled or {}).get("metrics", {}).get("auc") if isinstance(pooled, dict) else None
        rows = []
        for kind, slc in specialists.items():
            status = slc.get("status", "?")
            regimes = ", ".join(slc.get("regime_subset") or []) or "ALL"
            metrics = slc.get("metrics") or {}
            auc = metrics.get("auc")
            dacc = metrics.get("directional_accuracy")
            target_kind = slc.get("specialist_target_kind", "—")
            lift = (
                (auc - pooled_auc)
                if (auc is not None and pooled_auc is not None
                    and not math.isnan(auc) and not math.isnan(pooled_auc))
                else None
            )
            rows.append(
                "<tr>"
                f"<td>{html.escape(kind)}</td>"
                f"<td>{html.escape(regimes)}</td>"
                f"<td>{html.escape(target_kind)}</td>"
                f"<td>{html.escape(status)}</td>"
                f"<td>{_fmt(slc.get('n_rows'))}</td>"
                f"<td>{_fmt(auc)}</td>"
                f"<td>{_fmt(dacc)}</td>"
                f"<td>{_fmt(lift)}</td>"
                "</tr>"
            )
        parts.append(
            "<table><thead><tr>"
            "<th>Specialist</th><th>Regimes</th><th>Target</th><th>Status</th>"
            "<th>Rows</th><th>AUC</th><th>Dir. acc.</th><th>AUC lift vs pooled</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    parts.append("</section>")
    return "\n".join(parts)


def render_html(report: Optional[dict] = None) -> str:
    if report is None:
        if not REPORT_PATH.exists():
            return (
                "<!doctype html><html><body><h1>ML report</h1>"
                "<p>No training report yet. Run <code>pnpm --filter @workspace/ml-engine train</code>.</p>"
                "</body></html>"
            )
        report = json.loads(REPORT_PATH.read_text())

    regression_history = _load_regression_head_history()
    sections = "".join(
        _section(tf, r, regression_history)
        for tf, r in report.get("timeframes", {}).items()
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>ML Engine — Training Report</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 960px; margin: 24px auto; color:#111827; padding: 0 16px; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 32px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }}
  h3 {{ margin-top: 18px; }}
  .subhead {{ color:#374151; font-size: 1em; margin-top: 24px; border-bottom: 1px dashed #e5e7eb; padding-bottom: 4px; }}
  .slice {{ background:#fafafa; border:1px solid #e5e7eb; border-radius:8px; padding: 12px 16px; margin: 8px 0; }}
  .version {{ font-size: 0.65em; color: #6b7280; font-weight: normal; }}
  .meta {{ color:#6b7280; font-size: 0.9em; }}
  .warn {{ color: #b45309; background: #fffbeb; padding: 8px 12px; border-left: 3px solid #d97706; }}
  .kv {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 8px 16px; margin: 12px 0; }}
  .kv > div {{ background:#fff; padding:8px 10px; border-radius:6px; border:1px solid #e5e7eb; }}
  .kv span {{ display:block; font-size: 0.75em; color:#6b7280; }}
  .kv b {{ font-size: 1.1em; }}
  .gates {{ margin: 8px 0 4px; padding: 6px 10px; border-radius: 6px; background: #f0f9ff; border-left: 3px solid #0284c7; font-size: 0.85em; color: #0c4a6e; }}
  .gates .aligned {{ font-weight: 700; color: #047857; }}
  .gates .meta {{ color: #6b7280; }}
  table {{ width:100%; border-collapse: collapse; margin-top: 12px; font-size: 0.85em; }}
  th, td {{ border-bottom: 1px solid #e5e7eb; padding: 4px 6px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background:#f3f4f6; padding: 1px 5px; border-radius:4px; }}
</style></head>
<body>
<h1>ML Engine — Training Report</h1>
<p class="meta">Generated: {html.escape(report.get('generated_at',''))} &middot;
Coins: {html.escape(', '.join(report.get('coin_ids', [])))} &middot;
Lookback: {report.get('lookback_days','?')} days</p>
<p class="meta">Multiclass (DOWN / STABLE / UP) LightGBM with per-class isotonic calibration on a held-out tail. Per-coin models are trained whenever a coin has at least 80 rows; coins below that threshold are served by a pooled fallback. See task #88 plan for the data-reality rationale.</p>
{sections}
</body></html>"""
