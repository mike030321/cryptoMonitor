"""Read-only post-fee economic diagnostic for a saved per-coin model.

Reconstructs the calibration-tail holdout the trainer used when scoring
``directional_call_share`` and ``pnl_after_fees`` for a (coin, timeframe,
version) slice, scores it with the **saved** model + calibrators +
regression head, and walks every row through the live decision engine
(``app.decision_engine.decide``). Emits a markdown report and per-trade
JSON under ``artifacts/ml-engine/diagnostics/<slug>_<timestamp>/``.

Strictly read-only: never touches the registry, the database, the
frictions config, the gates, the watchdog, or the live campaign. Refuses
to run when the dataset row count drifts from the manifest's
``n_train_rows``.

Usage:
    python -m scripts.diagnose_post_fee \\
        --coin bonk --timeframe 5m --version 20260429T083323Z

Outputs (under ``artifacts/ml-engine/diagnostics/<coin>_<tf>_post_fee_<TS>/``):
    REPORT.md           Markdown narrative
    summary.json        Aggregate metrics keyed for downstream tooling
    per_trade.json      Per-trade rows that survived the live decision rule
    holdout_scored.csv  Every holdout row's calibrated probs + decision
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backtest.contract import Frictions, get_frictions  # noqa: E402
from app.decision_engine.engine import DecisionRequest, decide  # noqa: E402
from app.training import registry as registry_module  # noqa: E402
from app.training.registry import (  # noqa: E402
    FEATURE_COLUMNS,
    LoadedModel,
    load_model,
)

CALIBRATION_HOLDOUT_FRACTION = 0.2  # mirrors app.training.train


# ── Schema for per-trade rows. Pinned by the test suite. ───────────────────
PER_TRADE_COLUMNS: list[str] = [
    "row_index",
    "timestamp_ms",
    "timestamp_iso",
    "regime",
    "p_down",
    "p_stable",
    "p_up",
    "edge",
    "magnitude_pct",
    "expected_return_pct",
    "last_price",
    "atr14",
    "label_3class",
    "forward_return_pct",
    "action",
    "direction",
    "confidence",
    "size_multiplier",
    "position_size_usd",
    "sl_price",
    "tp_price",
    "skip_reason",
    "gross_pct",
    "net_pct",
]


# ── Errors ─────────────────────────────────────────────────────────────────
class HoldoutDriftError(RuntimeError):
    """Raised when the dataset row count for ``coin`` no longer matches
    the manifest's ``n_train_rows`` — the saved holdout cannot be
    reconstructed faithfully and the diagnostic refuses to proceed.
    """


class DatasetNotFoundError(RuntimeError):
    """Raised when no labeled-dataset parquet matches the manifest."""


# ── Dataset discovery ─────────────────────────────────────────────────────
def find_dataset_for_manifest(
    timeframe: str, coin: str, n_expected: int,
    explicit_path: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
) -> Path:
    """Locate the labeled dataset whose ``coin`` slice matches the
    manifest's ``n_train_rows``. If ``explicit_path`` is provided it is
    used as-is (and validated by the caller). Otherwise we walk
    ``datasets/<tf>_*.parquet`` newest-first and return the first parquet
    whose coin row count equals ``n_expected``.
    """
    if explicit_path is not None:
        return explicit_path
    root = datasets_dir or (registry_module.REGISTRY_ROOT / "datasets")
    candidates = sorted(
        root.glob(f"{timeframe}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            head = pd.read_parquet(path, columns=["coin_id"])
        except Exception:
            continue
        n_rows = int((head["coin_id"] == coin).sum())
        if n_rows == n_expected:
            return path
    raise DatasetNotFoundError(
        f"No dataset under {root} has a {coin}/{timeframe} slice with "
        f"{n_expected} rows. Cannot reconstruct the saved holdout."
    )


# ── Vectorized scoring ────────────────────────────────────────────────────
def calibrated_3class_probs_batch(
    model: LoadedModel, X: pd.DataFrame,
) -> np.ndarray:
    """Vectorized mirror of ``app.main._calibrated_3class_probs``.

    Returns a ``(n, 3)`` array in DOWN/STABLE/UP order with rows summing
    to one. Handles both booster-served slots (LightGBM model on disk)
    and baseline-served slots (multinomial-logistic ``(encoder, lr,
    priors)`` triple). Applies whichever calibrator family the manifest
    persisted (``IsotonicRegression`` or ``TemperatureScaledClass``);
    both expose the same ``.predict`` interface.
    """
    served = getattr(model.manifest, "served_predictor_kind", None)
    if served == "baseline" and model.baseline_artifact is not None:
        # Mirror app.main._calibrated_3class_probs's baseline branch.
        from app.training.train import _baseline_predict

        enc, lr, priors = model.baseline_artifact
        raw = _baseline_predict(enc, lr, priors, X)
    elif model.booster is not None:
        raw = model.booster.predict(
            X, num_iteration=model.booster.best_iteration,
        )
    else:
        raise RuntimeError(
            "diagnose_post_fee requires either a booster or a baseline "
            f"artifact; got served_predictor_kind={served!r} with both "
            "missing.",
        )
    raw = np.atleast_2d(raw).astype(float)
    if raw.shape[1] != 3:
        raise RuntimeError(
            f"Booster returned shape {raw.shape}; expected (n, 3).",
        )
    if model.calibrators is None:
        s = raw.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return raw / s
    cal = np.zeros_like(raw)
    for k in range(3):
        c = model.calibrators[k]
        cal[:, k] = raw[:, k] if c is None else c.predict(raw[:, k])
    s = cal.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return cal / s


def regressor_magnitudes_pct(
    model: LoadedModel, X: pd.DataFrame,
) -> Optional[np.ndarray]:
    if model.regressor is None:
        return None
    pred = model.regressor.predict(
        X, num_iteration=model.regressor.best_iteration,
    )
    return np.abs(np.asarray(pred, dtype=float))


# ── Decision-engine sweep ─────────────────────────────────────────────────
def run_decision_sweep(
    df_holdout: pd.DataFrame,
    probs: np.ndarray,
    magnitudes_pct: Optional[np.ndarray],
    fr: Frictions,
    coin: str,
    timeframe: str,
    means_pct: list[float],
) -> pd.DataFrame:
    """Apply ``decide()`` to every holdout row. Mirrors
    ``app.main.predict``'s composition of probabilities, regressor
    magnitude, and ``expectedReturnPct``.
    """
    n = len(df_holdout)
    p_down = probs[:, 0]
    p_stable = probs[:, 1]
    p_up = probs[:, 2]
    edge = p_up - p_down

    if magnitudes_pct is not None:
        sign = np.where(p_up >= p_down, 1.0, -1.0)
        expected = sign * magnitudes_pct
    else:
        expected = (
            p_down * means_pct[0]
            + p_stable * means_pct[1]
            + p_up * means_pct[2]
        )

    forward_return_arr = df_holdout["forward_return"].to_numpy(dtype=float)
    last_price_arr = df_holdout["lastPrice"].to_numpy(dtype=float)
    atr_arr = df_holdout["atr14"].to_numpy(dtype=float)
    regime_arr = (
        df_holdout["regime"].astype(object).to_numpy()
        if "regime" in df_holdout.columns
        else np.array([None] * n, dtype=object)
    )
    label_arr = df_holdout["label_3class"].to_numpy(dtype=int)
    ts_arr = df_holdout["timestamp_ms"].to_numpy(dtype=np.int64)
    iso_arr = pd.to_datetime(ts_arr, unit="ms", utc=True).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00",
    )

    rtc_pct = float(fr.round_trip_cost_pct) * 100.0
    rows: list[dict] = []
    for i in range(n):
        regime = regime_arr[i] if isinstance(regime_arr[i], str) else None
        req = DecisionRequest(
            coin_id=coin,
            timeframe=timeframe,
            last_price=float(last_price_arr[i]),
            atr_value=float(atr_arr[i]),
            prob_up=float(p_up[i]),
            prob_down=float(p_down[i]),
            prob_stable=float(p_stable[i]),
            expected_return_pct=float(expected[i]),
            regime=regime,
        )
        try:
            res = decide(req, fr=fr)
        except Exception as exc:  # noqa: BLE001 - log + skip
            rows.append({
                "row_index": int(i),
                "timestamp_ms": int(ts_arr[i]),
                "timestamp_iso": str(iso_arr[i]),
                "regime": regime,
                "p_down": float(p_down[i]),
                "p_stable": float(p_stable[i]),
                "p_up": float(p_up[i]),
                "edge": float(edge[i]),
                "magnitude_pct": (
                    float(magnitudes_pct[i])
                    if magnitudes_pct is not None else float("nan")
                ),
                "expected_return_pct": float(expected[i]),
                "last_price": float(last_price_arr[i]),
                "atr14": float(atr_arr[i]),
                "label_3class": int(label_arr[i]),
                "forward_return_pct": float(forward_return_arr[i]) * 100.0,
                "action": "no_trade",
                "direction": None,
                "confidence": 0.0,
                "size_multiplier": 0.0,
                "position_size_usd": 0.0,
                "sl_price": None,
                "tp_price": None,
                "skip_reason": f"engine_exception:{type(exc).__name__}",
                "gross_pct": float("nan"),
                "net_pct": float("nan"),
            })
            continue

        if res.action == "no_trade":
            gross_pct = float("nan")
            net_pct = float("nan")
        else:
            direction_sign = 1.0 if res.direction == "up" else -1.0
            gross_pct = direction_sign * float(forward_return_arr[i]) * 100.0
            net_pct = gross_pct - rtc_pct

        rows.append({
            "row_index": int(i),
            "timestamp_ms": int(ts_arr[i]),
            "timestamp_iso": str(iso_arr[i]),
            "regime": regime,
            "p_down": float(p_down[i]),
            "p_stable": float(p_stable[i]),
            "p_up": float(p_up[i]),
            "edge": float(edge[i]),
            "magnitude_pct": (
                float(magnitudes_pct[i])
                if magnitudes_pct is not None else float("nan")
            ),
            "expected_return_pct": float(expected[i]),
            "last_price": float(last_price_arr[i]),
            "atr14": float(atr_arr[i]),
            "label_3class": int(label_arr[i]),
            "forward_return_pct": float(forward_return_arr[i]) * 100.0,
            "action": res.action,
            "direction": res.direction,
            "confidence": float(res.confidence),
            "size_multiplier": float(res.size_multiplier),
            "position_size_usd": float(res.position_size_usd),
            "sl_price": (
                None if res.sl_price is None else float(res.sl_price)
            ),
            "tp_price": (
                None if res.tp_price is None else float(res.tp_price)
            ),
            "skip_reason": res.skip_reason,
            "gross_pct": gross_pct,
            "net_pct": net_pct,
        })

    out = pd.DataFrame(rows, columns=PER_TRADE_COLUMNS)
    return out


# ── Aggregations ──────────────────────────────────────────────────────────
def loose_post_fee_aggregate(
    probs: np.ndarray,
    forward_returns: np.ndarray,
    fr: Frictions,
    magnitudes_pct: Optional[np.ndarray],
) -> dict:
    """Mirror of ``app.training.train._holdout_pnl_after_fees``.

    Same MDE / MER gates the trainer applied when persisting
    ``slice.pnl_after_fees`` — i.e. the "loose" entry rule that
    determines the manifest-side post-fee figure. Used as a triangulation
    reference so the diagnostic's "no-engine" total matches the trainer's
    own number on the same holdout.
    """
    n = len(forward_returns)
    rtc_pct = float(fr.round_trip_cost_pct) * 100.0
    if n == 0:
        return {
            "n_trades": 0, "trade_share": 0.0, "gross_pct_mean": 0.0,
            "round_trip_cost_pct": round(rtc_pct, 4),
            "net_pct_mean": 0.0, "net_pct_total": 0.0,
            "win_rate": None,
        }
    p_down = probs[:, 0]
    p_up = probs[:, 2]
    edge = p_up - p_down
    argmax = probs.argmax(axis=1)
    take = (argmax != 1) & (np.abs(edge) >= float(fr.min_directional_edge))
    if magnitudes_pct is not None and len(magnitudes_pct) == n:
        take = take & (
            np.abs(magnitudes_pct) >= float(fr.min_expected_return_pct)
        )
    n_trades = int(take.sum())
    if n_trades == 0:
        return {
            "n_trades": 0, "trade_share": 0.0, "gross_pct_mean": 0.0,
            "round_trip_cost_pct": round(rtc_pct, 4),
            "net_pct_mean": 0.0, "net_pct_total": 0.0,
            "win_rate": None,
        }
    direction = np.sign(edge[take])
    direction[direction == 0] = 1.0
    gross_pct = direction * np.asarray(forward_returns, dtype=float)[take] * 100.0
    net_pct = gross_pct - rtc_pct
    return {
        "n_trades": n_trades,
        "trade_share": round(n_trades / n, 4),
        "gross_pct_mean": round(float(gross_pct.mean()), 4),
        "round_trip_cost_pct": round(rtc_pct, 4),
        "net_pct_mean": round(float(net_pct.mean()), 4),
        "net_pct_total": round(float(net_pct.sum()), 4),
        "win_rate": round(float((net_pct > 0).mean()), 4),
    }


def confidence_buckets(scored: pd.DataFrame) -> list[dict]:
    bins = [0.0, 0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.001]
    labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins) - 1)]
    confidence = np.maximum(scored["p_up"], scored["p_down"])
    out = []
    for i, lab in enumerate(labels):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        sub = scored[mask]
        traded = sub[sub["action"].isin(["long", "short"])]
        out.append({
            "bucket": lab,
            "n_holdout_rows": int(mask.sum()),
            "n_trades": int(len(traded)),
            "trade_share": (
                float(len(traded) / max(int(mask.sum()), 1))
            ),
            "net_pct_mean": (
                float(traded["net_pct"].mean()) if len(traded) else None
            ),
            "net_pct_total": (
                float(traded["net_pct"].sum()) if len(traded) else 0.0
            ),
            "win_rate": (
                float((traded["net_pct"] > 0).mean())
                if len(traded) else None
            ),
        })
    return out


def edge_filter_table(scored: pd.DataFrame) -> list[dict]:
    edges = [0.05, 0.07, 0.10, 0.15, 0.20]
    rows = []
    for thr in edges:
        mask = scored["edge"].abs() >= thr
        sub = scored[mask]
        # Hypothetical pnl if we forced an entry on every row that
        # cleared this edge (sign from edge), using the same round-trip
        # cost as the live engine. Useful to disentangle "engine vetoed"
        # vs "model edge wasn't profitable".
        forced_dir = np.sign(sub["edge"].to_numpy())
        forced_dir[forced_dir == 0] = 1.0
        gross = forced_dir * sub["forward_return_pct"].to_numpy()
        rtc = scored.attrs.get("round_trip_cost_pct", 0.3)
        net = gross - rtc
        # And the realized post-engine view: trades the engine actually
        # took at this edge floor.
        traded = sub[sub["action"].isin(["long", "short"])]
        rows.append({
            "edge_floor": thr,
            "n_holdout_rows_above": int(mask.sum()),
            "n_engine_trades_above": int(len(traded)),
            "engine_net_pct_total": (
                float(traded["net_pct"].sum()) if len(traded) else 0.0
            ),
            "engine_win_rate": (
                float((traded["net_pct"] > 0).mean())
                if len(traded) else None
            ),
            "forced_net_pct_mean": (
                float(net.mean()) if len(net) else None
            ),
            "forced_net_pct_total": (
                float(net.sum()) if len(net) else 0.0
            ),
            "forced_win_rate": (
                float((net > 0).mean()) if len(net) else None
            ),
        })
    return rows


def calibration_check(scored: pd.DataFrame) -> dict:
    """For each class, compute mean predicted prob vs realized class
    frequency on the holdout.
    """
    labels = scored["label_3class"].to_numpy(dtype=int)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    out: dict = {"by_class": {}, "reliability_curve_up": []}
    for k, name in enumerate(("DOWN", "STABLE", "UP")):
        col = scored[["p_down", "p_stable", "p_up"][k]].to_numpy(dtype=float)
        out["by_class"][name] = {
            "predicted_mean": float(col.mean()),
            "realized_share": float((labels == k).mean()),
            "brier": float(np.mean(((labels == k).astype(float) - col) ** 2)),
        }
    p_up = scored["p_up"].to_numpy(dtype=float)
    y_up = (labels == 2).astype(float)
    for i in range(len(bins) - 1):
        mask = (p_up >= bins[i]) & (p_up < bins[i + 1])
        if mask.sum() == 0:
            continue
        out["reliability_curve_up"].append({
            "bin": f"[{bins[i]:.2f},{bins[i + 1]:.2f})",
            "n": int(mask.sum()),
            "predicted_mean": float(p_up[mask].mean()),
            "realized": float(y_up[mask].mean()),
        })
    return out


def trade_distribution(scored: pd.DataFrame) -> dict:
    traded = scored[scored["action"].isin(["long", "short"])]
    skips = (
        scored[scored["action"] == "no_trade"]["skip_reason"]
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )
    return {
        "n_holdout_rows": int(len(scored)),
        "n_no_trade": int((scored["action"] == "no_trade").sum()),
        "n_long": int((scored["action"] == "long").sum()),
        "n_short": int((scored["action"] == "short").sum()),
        "n_trades": int(len(traded)),
        "trade_share": float(len(traded) / max(len(scored), 1)),
        "skip_reason_counts": {str(k): int(v) for k, v in skips.items()},
        "regime_breakdown": {
            str(reg): int((traded["regime"] == reg).sum())
            for reg in sorted({str(r) for r in traded["regime"].unique()})
        } if len(traded) else {},
    }


def hold_horizon_sensitivity(
    scored: pd.DataFrame, df_holdout: pd.DataFrame, fr: Frictions,
    horizons: tuple[int, ...] = (1, 2, 3, 6),
) -> list[dict]:
    """For every accepted trade, compute the realized pnl if we held for
    ``h`` bars instead of ``horizon_candles=1``. The n-bar return is the
    chained product of one-bar ``forward_return``s starting at the entry
    bar, so a 1-bar horizon equals the engine's default exit.
    """
    fr_arr = df_holdout["forward_return"].to_numpy(dtype=float)
    n = len(fr_arr)
    rtc_pct = float(fr.round_trip_cost_pct) * 100.0
    traded_idx = scored[
        scored["action"].isin(["long", "short"])
    ]["row_index"].to_numpy(dtype=int)
    direction_sign = np.where(
        scored.loc[
            scored["action"].isin(["long", "short"]), "direction"
        ].to_numpy() == "up",
        1.0,
        -1.0,
    )
    rows = []
    for h in horizons:
        if h <= 0:
            continue
        # Chained n-bar return ending at entry+h. Drop trades where the
        # holdout ends before we see h bars.
        keep_mask = traded_idx + h <= n
        if not keep_mask.any():
            rows.append({
                "horizon_bars": int(h),
                "n_eligible_trades": 0,
                "gross_pct_mean": None,
                "net_pct_mean": None,
                "net_pct_total": 0.0,
                "win_rate": None,
            })
            continue
        eligible_idx = traded_idx[keep_mask]
        eligible_dir = direction_sign[keep_mask]
        # Vectorized chained product: for each trade idx, prod((1+fr)[i:i+h]) - 1
        chained = np.empty(len(eligible_idx), dtype=float)
        for j, i0 in enumerate(eligible_idx):
            chained[j] = float(np.prod(1.0 + fr_arr[i0:i0 + h]) - 1.0)
        gross_pct = eligible_dir * chained * 100.0
        net_pct = gross_pct - rtc_pct
        rows.append({
            "horizon_bars": int(h),
            "n_eligible_trades": int(len(eligible_idx)),
            "gross_pct_mean": float(gross_pct.mean()),
            "net_pct_mean": float(net_pct.mean()),
            "net_pct_total": float(net_pct.sum()),
            "win_rate": float((net_pct > 0).mean()),
        })
    return rows


# ── Markdown rendering ────────────────────────────────────────────────────
def _fmt(v, fmt: str = ".4f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return format(float(v), fmt)


def render_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"# Post-fee economic diagnostic — "
        f"{summary['coin']} @ {summary['timeframe']} "
        f"(version `{summary['version']}`)",
    )
    lines.append("")
    lines.append(f"_Generated {summary['generated_at_iso']}_")
    lines.append("")
    lines.append("Read-only diagnostic. No registry, gates, or campaign state")
    lines.append("was modified. Holdout reconstructed from the labeled dataset")
    lines.append(
        f"`{summary['dataset_path']}` (last 20% / "
        f"{summary['holdout']['n_rows']} rows).",
    )
    lines.append("")

    lines.append("## Manifest header")
    lines.append("")
    mh = summary["manifest"]
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k in (
        "n_train_rows", "n_test_rows", "threshold_pct",
        "horizon_candles", "feature_schema_hash",
        "directional_call_share", "directional_call_share_n",
        "directional_call_share_source",
        "served_predictor_kind",
    ):
        lines.append(f"| `{k}` | {mh.get(k)} |")
    tw = mh.get("training_window") or {}
    lines.append(
        f"| `training_window` | {tw.get('start', '—')} → "
        f"{tw.get('end', '—')} |",
    )
    lines.append(f"| `metrics.auc` | {_fmt(mh['metrics']['auc'])} |")
    lines.append(
        f"| `metrics.directional_accuracy` | "
        f"{_fmt(mh['metrics']['directional_accuracy'])} |",
    )
    lines.append("")

    lines.append("## Aggregate post-fee outcome")
    lines.append("")
    a = summary["aggregate"]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| holdout rows | {a['n_holdout_rows']} |")
    lines.append(f"| trades emitted | {a['n_trades']} |")
    lines.append(f"| trade share | {_fmt(a['trade_share'])} |")
    lines.append(f"| longs | {a['n_long']} |")
    lines.append(f"| shorts | {a['n_short']} |")
    lines.append(
        f"| round-trip cost (pct) | {_fmt(a['round_trip_cost_pct'])} |",
    )
    lines.append(f"| gross pct mean | {_fmt(a['gross_pct_mean'])} |")
    lines.append(f"| net pct mean | {_fmt(a['net_pct_mean'])} |")
    lines.append(f"| net pct total | {_fmt(a['net_pct_total'])} |")
    lines.append(f"| win rate | {_fmt(a['win_rate'])} |")
    lines.append("")

    lines.append("## Trainer-side reference (loose entry rule)")
    lines.append("")
    lines.append(
        "Mirrors `app.training.train._holdout_pnl_after_fees` — the figure"
        " the trainer persisted in `slice.pnl_after_fees`. Only the "
        "MDE / MER quant-brain floors apply (no MIN_CONFIDENCE_TO_TRADE / "
        "MIN_TP_DISTANCE_PCT / MIN_EV_VS_COST gates). This is the "
        "all-trades baseline the test pins against.",
    )
    lines.append("")
    lp = summary["loose_pnl_after_fees"]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| n_trades | {lp['n_trades']} |")
    lines.append(f"| trade share | {_fmt(lp['trade_share'])} |")
    lines.append(f"| gross pct mean | {_fmt(lp['gross_pct_mean'])} |")
    lines.append(f"| net pct mean | {_fmt(lp['net_pct_mean'])} |")
    lines.append(f"| net pct total | {_fmt(lp['net_pct_total'])} |")
    lines.append(f"| win rate | {_fmt(lp['win_rate'])} |")
    lines.append("")

    lines.append("## Confidence buckets")
    lines.append("")
    lines.append(
        "| bucket | rows | trades | share | net mean | net total | win |",
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in summary["confidence_buckets"]:
        lines.append(
            f"| `{b['bucket']}` | {b['n_holdout_rows']} | "
            f"{b['n_trades']} | {_fmt(b['trade_share'])} | "
            f"{_fmt(b['net_pct_mean'])} | {_fmt(b['net_pct_total'])} | "
            f"{_fmt(b['win_rate'])} |",
        )
    lines.append("")

    lines.append("## Edge-floor sensitivity")
    lines.append("")
    lines.append("Each row counts holdout signals where `|p_up - p_down| >=` floor.")
    lines.append("`forced_*` ignores the live decision engine and assumes every such")
    lines.append("row was traded; `engine_*` is what `decide()` actually accepted.")
    lines.append("")
    lines.append(
        "| floor | rows | engine trades | engine net total | "
        "forced trades net mean | forced net total | forced win |",
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary["edge_filters"]:
        lines.append(
            f"| {_fmt(r['edge_floor'])} | {r['n_holdout_rows_above']} | "
            f"{r['n_engine_trades_above']} | "
            f"{_fmt(r['engine_net_pct_total'])} | "
            f"{_fmt(r['forced_net_pct_mean'])} | "
            f"{_fmt(r['forced_net_pct_total'])} | "
            f"{_fmt(r['forced_win_rate'])} |",
        )
    lines.append("")

    lines.append("## Calibration check (holdout)")
    lines.append("")
    lines.append("| class | predicted_mean | realized_share | brier |")
    lines.append("|---|---:|---:|---:|")
    for cls, payload in summary["calibration"]["by_class"].items():
        lines.append(
            f"| {cls} | {_fmt(payload['predicted_mean'])} | "
            f"{_fmt(payload['realized_share'])} | "
            f"{_fmt(payload['brier'])} |",
        )
    lines.append("")
    lines.append("Reliability curve for `p_up`:")
    lines.append("")
    lines.append("| bin | n | predicted_mean | realized |")
    lines.append("|---|---:|---:|---:|")
    for r in summary["calibration"]["reliability_curve_up"]:
        lines.append(
            f"| `{r['bin']}` | {r['n']} | {_fmt(r['predicted_mean'])} | "
            f"{_fmt(r['realized'])} |",
        )
    lines.append("")

    lines.append("## Trade distribution")
    lines.append("")
    td = summary["trade_distribution"]
    lines.append(f"- holdout rows: **{td['n_holdout_rows']}**")
    lines.append(f"- no_trade: **{td['n_no_trade']}**")
    lines.append(f"- long / short: **{td['n_long']} / {td['n_short']}**")
    lines.append("")
    lines.append("Top engine skip reasons:")
    lines.append("")
    lines.append("| reason | n |")
    lines.append("|---|---:|")
    skip_items = sorted(
        td["skip_reason_counts"].items(),
        key=lambda kv: kv[1], reverse=True,
    )
    for name, n in skip_items[:12]:
        lines.append(f"| `{name}` | {n} |")
    lines.append("")
    if td["regime_breakdown"]:
        lines.append("Trades per regime:")
        lines.append("")
        lines.append("| regime | n |")
        lines.append("|---|---:|")
        for reg, n in td["regime_breakdown"].items():
            lines.append(f"| `{reg}` | {n} |")
        lines.append("")

    lines.append("## Hold-horizon sensitivity")
    lines.append("")
    lines.append("For each accepted trade, what would PnL look like if we")
    lines.append(
        f"held the position for `h` bars instead of "
        f"`horizon_candles={summary['manifest']['horizon_candles']}`? "
        "(chained 1-bar forward returns.)",
    )
    lines.append("")
    lines.append(
        "| horizon | eligible | gross mean | net mean | net total | win |",
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for h in summary["hold_horizon_sensitivity"]:
        lines.append(
            f"| {h['horizon_bars']} | {h['n_eligible_trades']} | "
            f"{_fmt(h['gross_pct_mean'])} | {_fmt(h['net_pct_mean'])} | "
            f"{_fmt(h['net_pct_total'])} | {_fmt(h['win_rate'])} |",
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Generated by `scripts/diagnose_post_fee.py` (read-only).")
    return "\n".join(lines) + "\n"


# ── Top-level orchestration ───────────────────────────────────────────────
def run_diagnostic(
    coin: str,
    timeframe: str,
    version: str,
    dataset_path: Optional[Path] = None,
    out_root: Optional[Path] = None,
    horizons: tuple[int, ...] = (1, 2, 3, 6),
) -> dict:
    model = load_model(coin, timeframe, version)
    if model is None:
        raise FileNotFoundError(
            f"no model registered for coin={coin!r} timeframe={timeframe!r} "
            f"version={version!r}",
        )
    manifest = model.manifest

    ds_path = find_dataset_for_manifest(
        timeframe=timeframe,
        coin=coin,
        n_expected=int(manifest.n_train_rows),
        explicit_path=dataset_path,
    )
    df = pd.read_parquet(ds_path)
    df = df[df["coin_id"] == coin].copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    if len(df) != int(manifest.n_train_rows):
        raise HoldoutDriftError(
            f"dataset {ds_path} has {len(df)} rows for coin={coin!r}, "
            f"manifest declares n_train_rows={manifest.n_train_rows}. "
            "Refusing to reconstruct the holdout — re-train or pass "
            "--dataset to the original snapshot.",
        )
    cal_start = max(1, int(len(df) * (1.0 - CALIBRATION_HOLDOUT_FRACTION)))
    n_holdout_expected = len(df) - cal_start
    df_holdout = df.iloc[cal_start:].reset_index(drop=True)
    if len(df_holdout) != n_holdout_expected:
        raise HoldoutDriftError(
            f"holdout slice produced {len(df_holdout)} rows; expected "
            f"{n_holdout_expected}.",
        )

    # Encode coin_idx exactly like the trainer.
    vocab = list(manifest.coin_vocab)
    idx = {c: i for i, c in enumerate(vocab)}
    df_holdout["coin_idx"] = (
        df_holdout["coin_id"].map(lambda c: idx.get(c, -1)).astype("int32")
    )

    feature_cols = list(manifest.feature_names) or list(FEATURE_COLUMNS)
    missing = [c for c in feature_cols if c not in df_holdout.columns]
    if missing:
        raise RuntimeError(
            "labeled dataset is missing feature columns the manifest "
            f"declares: {missing}",
        )
    X_holdout = df_holdout[feature_cols].copy()

    probs = calibrated_3class_probs_batch(model, X_holdout)
    magnitudes = regressor_magnitudes_pct(model, X_holdout)

    means_pct = list(manifest.class_return_means_pct) or [
        -float(manifest.threshold_pct),
        0.0,
        float(manifest.threshold_pct),
    ]

    fr = get_frictions()
    scored = run_decision_sweep(
        df_holdout=df_holdout,
        probs=probs,
        magnitudes_pct=magnitudes,
        fr=fr,
        coin=coin,
        timeframe=timeframe,
        means_pct=means_pct,
    )
    rtc_pct = float(fr.round_trip_cost_pct) * 100.0
    scored.attrs["round_trip_cost_pct"] = rtc_pct

    forward_returns_arr = df_holdout["forward_return"].to_numpy(dtype=float)
    loose_pnl = loose_post_fee_aggregate(
        probs=probs,
        forward_returns=forward_returns_arr,
        fr=fr,
        magnitudes_pct=magnitudes,
    )

    traded = scored[scored["action"].isin(["long", "short"])]
    aggregate = {
        "n_holdout_rows": int(len(scored)),
        "n_trades": int(len(traded)),
        "n_long": int((scored["action"] == "long").sum()),
        "n_short": int((scored["action"] == "short").sum()),
        "trade_share": float(len(traded) / max(len(scored), 1)),
        "round_trip_cost_pct": round(rtc_pct, 4),
        "gross_pct_mean": (
            float(traded["gross_pct"].mean()) if len(traded) else 0.0
        ),
        "gross_pct_total": (
            float(traded["gross_pct"].sum()) if len(traded) else 0.0
        ),
        "net_pct_mean": (
            float(traded["net_pct"].mean()) if len(traded) else 0.0
        ),
        "net_pct_total": (
            float(traded["net_pct"].sum()) if len(traded) else 0.0
        ),
        "win_rate": (
            float((traded["net_pct"] > 0).mean()) if len(traded) else None
        ),
    }

    summary = {
        "coin": coin,
        "timeframe": timeframe,
        "version": version,
        "generated_at_iso": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        "dataset_path": str(ds_path),
        "manifest": {
            "n_train_rows": manifest.n_train_rows,
            "n_test_rows": manifest.n_test_rows,
            "threshold_pct": manifest.threshold_pct,
            "horizon_candles": manifest.horizon_candles,
            "feature_schema_hash": getattr(
                manifest, "feature_schema_hash", None,
            ),
            "directional_call_share": manifest.directional_call_share,
            "directional_call_share_n": manifest.directional_call_share_n,
            "directional_call_share_source": (
                manifest.directional_call_share_source
            ),
            "served_predictor_kind": getattr(
                manifest, "served_predictor_kind", None,
            ),
            "training_window": getattr(manifest, "training_window", None),
            "metrics": dict(manifest.metrics),
        },
        "frictions": {
            "policy_version": fr.quant_policy_version,
            "round_trip_cost_pct": rtc_pct,
            "min_confidence_to_trade": fr.gate("MIN_CONFIDENCE_TO_TRADE")[
                "value"
            ],
            "min_tp_distance_pct": fr.gate("MIN_TP_DISTANCE_PCT")["value"],
            "min_ev_vs_cost": fr.gate("MIN_EV_VS_COST")["value"],
            "min_directional_prob": fr.min_directional_prob,
            "min_directional_edge": fr.min_directional_edge,
            "min_expected_return_pct": fr.min_expected_return_pct,
            "asymmetric_long_min_confidence": (
                fr.asymmetric_long_min_confidence
            ),
        },
        "holdout": {
            "n_rows": int(len(df_holdout)),
            "cal_start_index": int(cal_start),
            "first_timestamp_iso": pd.to_datetime(
                int(df_holdout["timestamp_ms"].iloc[0]), unit="ms", utc=True,
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_timestamp_iso": pd.to_datetime(
                int(df_holdout["timestamp_ms"].iloc[-1]), unit="ms", utc=True,
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "aggregate": aggregate,
        "loose_pnl_after_fees": loose_pnl,
        "confidence_buckets": confidence_buckets(scored),
        "edge_filters": edge_filter_table(scored),
        "calibration": calibration_check(scored),
        "trade_distribution": trade_distribution(scored),
        "hold_horizon_sensitivity": hold_horizon_sensitivity(
            scored, df_holdout, fr, horizons=horizons,
        ),
    }

    # Persist outputs.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_root is None:
        out_root = ROOT / "diagnostics"
    out_dir = out_root / f"{coin}_{timeframe}_post_fee_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
    )
    traded_records = scored[
        scored["action"].isin(["long", "short"])
    ].to_dict(orient="records")
    (out_dir / "per_trade.json").write_text(
        json.dumps(traded_records, indent=2, default=_json_default),
    )
    scored.to_csv(out_dir / "holdout_scored.csv", index=False)
    (out_dir / "REPORT.md").write_text(render_report(summary))
    summary["output_dir"] = str(out_dir)
    return summary


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is None or isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"unserializable: {type(obj)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coin", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--version", required=True)
    p.add_argument(
        "--dataset", default=None,
        help="Optional explicit path to the labeled dataset parquet.",
    )
    p.add_argument(
        "--out-root", default=None,
        help="Output root (defaults to artifacts/ml-engine/diagnostics).",
    )
    p.add_argument(
        "--horizons", default="1,2,3,6",
        help="Comma-separated hold horizons in bars (default: 1,2,3,6).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    summary = run_diagnostic(
        coin=args.coin,
        timeframe=args.timeframe,
        version=args.version,
        dataset_path=Path(args.dataset) if args.dataset else None,
        out_root=Path(args.out_root) if args.out_root else None,
        horizons=horizons,
    )
    print(json.dumps({
        "output_dir": summary["output_dir"],
        "n_trades": summary["aggregate"]["n_trades"],
        "net_pct_total": summary["aggregate"]["net_pct_total"],
        "win_rate": summary["aggregate"]["win_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
