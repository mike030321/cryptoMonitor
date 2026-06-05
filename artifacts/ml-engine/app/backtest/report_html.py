"""Self-contained HTML report renderer (no external assets, no JS deps)."""
from __future__ import annotations

import html
import json
from typing import Iterable


def _fmt(v, kind="num"):
    if v is None: return "—"
    if isinstance(v, bool): return "yes" if v else "no"
    if kind == "pct": return f"{v*100:.2f}%" if isinstance(v, (int, float)) else "—"
    if kind == "usd": return f"${v:,.2f}" if isinstance(v, (int, float)) else "—"
    if isinstance(v, float): return f"{v:.4f}"
    return html.escape(str(v))


def _equity_svg(p05: list[float], p50: list[float], p95: list[float],
                width=700, height=240, pad=30) -> str:
    if not p50:
        return f'<svg viewBox="0 0 {width} {height}" />'
    xs = list(range(len(p50)))
    y_min = min(min(p05), min(p50), min(p95))
    y_max = max(max(p05), max(p50), max(p95))
    if y_max == y_min:
        y_max = y_min + 1
    def sx(i): return pad + (i / max(1, len(xs) - 1)) * (width - 2 * pad)
    def sy(v): return height - pad - ((v - y_min) / (y_max - y_min)) * (height - 2 * pad)

    band_pts = ([f"{sx(i)},{sy(p95[i])}" for i in xs]
              + [f"{sx(i)},{sy(p05[i])}" for i in reversed(xs)])
    band_poly = " ".join(band_pts)
    median_pts = " ".join(f"{sx(i)},{sy(p50[i])}" for i in xs)

    return f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="{width}" height="{height}" fill="#0f172a"/>
      <polygon points="{band_poly}" fill="#3b82f6" fill-opacity="0.25"/>
      <polyline points="{median_pts}" fill="none" stroke="#60a5fa" stroke-width="2"/>
      <line x1="{pad}" y1="{sy(p50[0])}" x2="{width-pad}" y2="{sy(p50[0])}"
            stroke="#475569" stroke-dasharray="4 4"/>
      <text x="{pad}" y="20" fill="#cbd5e1" font-family="monospace" font-size="12">
        equity: ${y_min:,.0f} – ${y_max:,.0f} (5/50/95th MC)
      </text>
    </svg>'''


def _per_coin_table(per_coin: dict[str, dict]) -> str:
    rows = []
    for c, info in sorted(per_coin.items(), key=lambda kv: kv[0]):
        m = info.get("metrics", {}) or {}
        rows.append(f"""<tr>
          <td>{html.escape(c)}</td>
          <td class="num">{info.get('n_oos_rows', 0)}</td>
          <td class="num">{m.get('n_trades', 0)}</td>
          <td class="num">{info.get('n_skips', 0)}</td>
          <td class="num">{_fmt(m.get('win_rate'), 'pct')}</td>
          <td class="num">{_fmt(m.get('expectancy_usd'), 'usd')}</td>
          <td class="num">{_fmt(m.get('final_pnl_usd'), 'usd')}</td>
        </tr>""")
    return "\n".join(rows) or '<tr><td colspan="7" class="muted">no coins</td></tr>'


def _regime_table(regime_metrics: dict[str, dict]) -> str:
    rows = []
    for r, m in regime_metrics.items():
        rows.append(f"""<tr>
          <td>{html.escape(r)}</td>
          <td class="num">{m.get('n_trades', 0)}</td>
          <td class="num">{_fmt(m.get('win_rate'), 'pct')}</td>
          <td class="num">{_fmt(m.get('expectancy_usd'), 'usd')}</td>
          <td class="num">{_fmt(m.get('sharpe_per_trade'))}</td>
          <td class="num">{_fmt(m.get('max_drawdown_pct'))}%</td>
        </tr>""")
    return "\n".join(rows)


def render_report(report: dict) -> str:
    runs = report.get("runs", [])
    summary = report.get("summary", {})
    cards: list[str] = []
    for run in runs:
        m = run.get("metrics", {}) or {}
        mc = run.get("monte_carlo", {}) or {}
        v = run.get("verdict", {}) or {}
        verdict_class = "verdict-pass" if v.get("deploy") else "verdict-fail"
        verdict_text = "DEPLOY" if v.get("deploy") else "DO NOT DEPLOY"
        reasons_html = "".join(f"<li>{html.escape(str(r))}</li>" for r in v.get("reasons", []))
        passing = ", ".join(v.get("passing_regimes", [])) or "none"
        svg = _equity_svg(
            mc.get("equity_band_p05", []),
            mc.get("equity_band_p50", []),
            mc.get("equity_band_p95", []),
        )
        cards.append(f"""
        <section class="card">
          <header>
            <h2>{html.escape(run.get('timeframe', '?'))}</h2>
            <span class="badge {verdict_class}">{verdict_text}</span>
          </header>
          <div class="grid">
            <div><label>Trades</label><b>{m.get('n_trades', 0)}</b></div>
            <div><label>Win rate</label><b>{_fmt(m.get('win_rate'), 'pct')}</b></div>
            <div><label>Expectancy</label><b>{_fmt(m.get('expectancy_usd'), 'usd')}</b></div>
            <div><label>Profit factor</label><b>{_fmt(m.get('profit_factor'))}</b></div>
            <div><label>Sharpe / trade</label><b>{_fmt(m.get('sharpe_per_trade'))}</b></div>
            <div><label>Max DD</label><b>{_fmt(m.get('max_drawdown_pct'))}%</b></div>
            <div><label>Final P&amp;L</label><b>{_fmt(m.get('final_pnl_usd'), 'usd')}</b></div>
            <div><label>Time in market</label><b>{_fmt(m.get('time_in_market_pct'))}%</b></div>
            <div><label>Avg hold</label><b>{(m.get('avg_hold_ms') or 0)/60000:.1f} min</b></div>
            <div><label>MC final p05/p50/p95</label>
              <b>{_fmt(mc.get('final_pnl_p05'), 'usd')} /
                 {_fmt(mc.get('final_pnl_p50'), 'usd')} /
                 {_fmt(mc.get('final_pnl_p95'), 'usd')}</b></div>
          </div>
          <div class="chart">{svg}</div>
          <h3>Per-regime</h3>
          <table>
            <thead><tr><th>regime</th><th>trades</th><th>win</th>
              <th>expectancy</th><th>sharpe</th><th>max DD</th></tr></thead>
            <tbody>{_regime_table(run.get('regime_breakdown', {}))}</tbody>
          </table>
          <h3>Per-coin</h3>
          <table>
            <thead><tr><th>coin</th><th>OOS rows</th><th>trades</th><th>skips</th>
              <th>win</th><th>expectancy</th><th>final P&amp;L</th></tr></thead>
            <tbody>{_per_coin_table(run.get('per_coin', {}))}</tbody>
          </table>
          <h3>Verdict reasons</h3>
          <ul>{reasons_html}</ul>
          <p class="muted">passing regimes: {html.escape(passing)}</p>
        </section>""")

    overall_deploy = bool(summary.get("deploy"))
    overall_class = "verdict-pass" if overall_deploy else "verdict-fail"
    overall_text = "DEPLOY" if overall_deploy else "DO NOT DEPLOY"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Backtest report</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif;
          background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1 {{ font-size: 22px; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 0; }}
  h3 {{ font-size: 14px; margin: 16px 0 8px; color: #94a3b8; }}
  .muted {{ color: #94a3b8; font-size: 12px; }}
  .card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px;
           padding: 18px; margin-bottom: 18px; }}
  .card header {{ display: flex; align-items: center; justify-content: space-between; }}
  .badge {{ padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 12px; }}
  .verdict-pass {{ background: #052e16; color: #86efac; border: 1px solid #14532d; }}
  .verdict-fail {{ background: #450a0a; color: #fca5a5; border: 1px solid #7f1d1d; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
           margin-top: 12px; }}
  .grid > div {{ background: #1e293b; padding: 10px 12px; border-radius: 8px; }}
  .grid label {{ display: block; font-size: 11px; color: #94a3b8;
                 text-transform: uppercase; letter-spacing: 0.05em; }}
  .grid b {{ font-size: 16px; }}
  .chart {{ margin-top: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #1e293b; }}
  th {{ color: #94a3b8; font-weight: 500; font-size: 11px; text-transform: uppercase; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  ul {{ margin: 4px 0 0 0; padding-left: 20px; font-size: 13px; }}
</style></head>
<body>
  <h1>Backtest report
    <span class="badge {overall_class}" style="margin-left: 12px;">{overall_text}</span>
  </h1>
  <p class="muted">Generated {html.escape(str(report.get('generated_at', '')))}.
     Runs: {len(runs)}. Source: shared/trading-frictions.json.</p>
  {''.join(cards) if cards else '<p>No runs.</p>'}
</body></html>"""
