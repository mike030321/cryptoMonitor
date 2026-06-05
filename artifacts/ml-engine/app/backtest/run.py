"""End-to-end backtest CLI.

Iterates the tradeable timeframes from shared/trading-frictions.json,
locates the latest dataset snapshot per timeframe at
`models/datasets/{tf}_*.parquet`, runs walk-forward OOS prediction,
simulates with the live trader's frictions, computes metrics + Monte
Carlo + per-regime breakdown + verdict, then writes
`models/backtest_report.json` and `models/backtest_report.html`.

If a snapshot is missing the timeframe is skipped (with a recorded reason)
— never fabricate data the live system would not have had.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Determinism: backtests must reproduce bit-for-bit. Skip Optuna's
# timeout-driven hyperparameter search by default — fixed-config LightGBM
# trains in milliseconds and is reproducible. Callers can override by
# setting ML_BACKTEST_SKIP_OPTUNA=0 BEFORE invoking the CLI (not
# recommended for the deploy-gate report).
os.environ.setdefault("ML_SKIP_OPTUNA", os.environ.get("ML_BACKTEST_SKIP_OPTUNA", "1"))

import pandas as pd

from .contract import get_frictions
from .decision import decide
from .metrics import compute_metrics
from .monte_carlo import run_monte_carlo
from .regime import RegimeState, classify_regime
from .regime_breakdown import regime_breakdown
from .report_html import render_report
from .simulator import CoinTickStream, simulate
from .walk_forward_oos import predict_oos_for_dataset
from .journal_client import emit_backtest_journal

logger = logging.getLogger("ml-engine.backtest.run")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DATASETS_DIR = REPO_MODELS_DIR / "datasets"
REPORT_JSON = REPO_MODELS_DIR / "backtest_report.json"
REPORT_HTML = REPO_MODELS_DIR / "backtest_report.html"


def latest_snapshot_for(tf: str) -> Optional[Path]:
    if not DATASETS_DIR.exists():
        return None
    candidates = sorted(DATASETS_DIR.glob(f"{tf}_*.parquet"))
    return candidates[-1] if candidates else None


def build_tick_streams(df: pd.DataFrame) -> dict[str, CoinTickStream]:
    """The simulator needs a chronological close-price tape per coin so it
    can fire SL/TP/expiry between decision events. We reuse the per-row
    derived `entry_price` (= atr14 / atrPct * 100) as the close at that
    bucket — the same series the labeler used.
    """
    streams: dict[str, CoinTickStream] = {}
    for coin_id, group in df.groupby("coin_id"):
        g = group.sort_values("timestamp_ms")
        ts = g["timestamp_ms"].astype("int64").tolist()
        # entry_price is None for rows where atrPct=0; drop them from the tape.
        prices = (g["atr14"] / g["atrPct"] * 100.0).tolist()
        ts_clean, px_clean = [], []
        for t, p in zip(ts, prices):
            if p == p and p > 0:  # not NaN, positive
                ts_clean.append(int(t)); px_clean.append(float(p))
        if ts_clean:
            streams[coin_id] = CoinTickStream(coin_id, ts_clean, px_clean)
    return streams


def build_basket_change_lookup(streams: dict[str, CoinTickStream], tf_ms: int) -> dict[int, list[float]]:
    """At each bucket timestamp, compute a 24h percent change for every coin
    that has both a `now` and `now - 24h` price in its tape. The regime
    classifier consumes one such vector per evaluation timestamp.
    """
    one_day_ms = 24 * 60 * 60 * 1000
    by_ts: dict[int, list[float]] = defaultdict(list)
    for coin_id, s in streams.items():
        ts_to_px = dict(zip(s.timestamps_ms, s.prices))
        ts_sorted = s.timestamps_ms
        for i, ts in enumerate(ts_sorted):
            target = ts - one_day_ms
            # Find latest tick with timestamp <= target.
            j = i
            while j > 0 and ts_sorted[j] > target:
                j -= 1
            if j >= 0 and ts_sorted[j] <= target:
                base = s.prices[j]
                cur = s.prices[i]
                if base > 0:
                    pct = (cur - base) / base * 100.0
                    by_ts[ts].append(pct)
    return dict(by_ts)


def regime_lookup_for(basket_changes: dict[int, list[float]]):
    """Curry a fast lookup: for an evaluation ts, find the latest precomputed
    bucket and classify."""
    keys = sorted(basket_changes.keys())

    def _lookup(ts_ms: int) -> RegimeState:
        if not keys:
            return classify_regime([])
        # Binary search for last key <= ts_ms.
        lo, hi = 0, len(keys) - 1
        chosen = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if keys[mid] <= ts_ms:
                chosen = mid; lo = mid + 1
            else:
                hi = mid - 1
        if chosen < 0:
            return classify_regime([])
        return classify_regime(basket_changes[keys[chosen]])

    return _lookup


def run_one_timeframe(tf: str) -> dict:
    snap = latest_snapshot_for(tf)
    if snap is None:
        return {"timeframe": tf, "status": "no_dataset",
                "metrics": {}, "monte_carlo": {}, "regime_breakdown": {},
                "verdict": {"deploy": False, "reasons": ["no dataset snapshot"], "passing_regimes": []}}

    logger.info("backtest tf=%s snapshot=%s", tf, snap.name)
    df = pd.read_parquet(snap)
    if df.empty:
        return {"timeframe": tf, "status": "empty_dataset",
                "metrics": {}, "monte_carlo": {}, "regime_breakdown": {},
                "verdict": {"deploy": False, "reasons": ["empty snapshot"], "passing_regimes": []}}

    fr = get_frictions()
    oos = predict_oos_for_dataset(df)
    if oos.empty:
        return {"timeframe": tf, "status": "no_oos_predictions",
                "metrics": {}, "monte_carlo": {}, "regime_breakdown": {},
                "verdict": {"deploy": False, "reasons": ["walk-forward produced no OOS rows"],
                            "passing_regimes": []}}

    streams = build_tick_streams(df)
    basket_changes = build_basket_change_lookup(streams, fr.timeframe_ms(tf))
    regime_fn = regime_lookup_for(basket_changes)

    sim = simulate(
        timeframe=tf, oos_predictions=oos, tick_streams=streams,
        fr=fr, regime_lookup=regime_fn,
    )
    # Phase 1 — emit BACKTEST rows into prediction_journal so simulated
    # decisions sit alongside live ones for replay and counterfactuals.
    # No-op when ADMIN_API_KEY is unset; failures are swallowed and never
    # affect the simulation or its reports.
    try:
        n_journaled = emit_backtest_journal(sim, model_version=fr.quant_policy_version)
        if n_journaled:
            logger.info("emitted %d backtest journal rows for tf=%s", n_journaled, tf)
    except Exception as err:  # noqa: BLE001 — journal is fire-and-forget
        logger.warning("backtest journal emission failed for tf=%s: %s", tf, err)

    metrics = compute_metrics(sim.trades, sim.initial_equity).to_dict()
    mc = run_monte_carlo(sim.trades, sim.initial_equity, n_runs=2000).to_dict()
    rb = regime_breakdown(sim.trades, sim.initial_equity)
    # Per-coin breakdown (T007 acceptance: per-(coin, tf) cards). Coins
    # never see each other's PnL, so initial equity passed through unchanged.
    coins = sorted({t.coin_id for t in sim.trades} |
                   set(oos["coin_id"].unique().tolist()))
    per_coin: dict[str, dict] = {}
    for c in coins:
        c_trades = [t for t in sim.trades if t.coin_id == c]
        c_skips = [s for s in sim.skips if s.coin_id == c]
        per_coin[c] = {
            "metrics": compute_metrics(c_trades, sim.initial_equity).to_dict(),
            "n_oos_rows": int((oos["coin_id"] == c).sum()),
            "n_skips": len(c_skips),
        }
    verdict = decide(metrics, rb, fr=fr,
                     mc_p05_drawdown_pct=mc.get("max_drawdown_p05"))
    return {
        "timeframe": tf, "status": "ok",
        "n_oos_rows": len(oos),
        "n_skips": len(sim.skips),
        "metrics": metrics,
        "monte_carlo": mc,
        "regime_breakdown": rb,
        "per_coin": per_coin,
        "verdict": verdict.to_dict(),
    }


def _data_as_of(runs: list[dict]) -> Optional[str]:
    """Deterministic timestamp = max snapshot mtime found, formatted UTC.
    Falls back to None when no snapshots existed."""
    snaps = []
    if DATASETS_DIR.exists():
        snaps = list(DATASETS_DIR.glob("*.parquet"))
    if not snaps:
        return None
    latest = max(s.stat().st_mtime for s in snaps)
    return dt.datetime.fromtimestamp(latest, tz=dt.timezone.utc).isoformat()


def _input_fingerprint() -> str:
    """SHA256 of (contract JSON + sorted snapshot filenames + sizes). Two
    runs over the same inputs share the same fingerprint, so the report
    is byte-stable when the underlying data hasn't changed.
    """
    h = hashlib.sha256()
    h.update(json.dumps(get_frictions().raw, sort_keys=True).encode())
    if DATASETS_DIR.exists():
        for s in sorted(DATASETS_DIR.glob("*.parquet")):
            h.update(s.name.encode()); h.update(str(s.stat().st_size).encode())
    return h.hexdigest()[:16]


def run_backtest() -> dict:
    fr = get_frictions()
    runs = [run_one_timeframe(tf) for tf in fr.tradeable_timeframes()]
    deploy_overall = bool(runs) and any(r["verdict"].get("deploy") for r in runs)
    report = {
        # Deterministic: derived from inputs, NOT wall-clock. Same inputs
        # ⇒ identical report bytes (architect requirement).
        "data_as_of": _data_as_of(runs),
        "input_fingerprint": _input_fingerprint(),
        # Stamps the live quant-brain decision-rule version this report was
        # produced against. If the live rule is changed (e.g. new abstain
        # branch added in artifacts/api-server/src/lib/quant-brain.ts), bump
        # quant_brain.decision_thresholds.policy_version in
        # shared/trading-frictions.json so old reports loaded by the
        # dashboard surface as stale.
        "quant_policy_version": fr.quant_policy_version,
        "runs": runs,
        "summary": {"deploy": deploy_overall, "n_runs": len(runs)},
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    REPORT_HTML.write_text(render_report(report))
    logger.info("wrote %s and %s", REPORT_JSON, REPORT_HTML)
    return report


if __name__ == "__main__":
    run_backtest()
