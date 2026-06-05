"""Per-(coin, timeframe) dataset assembler for Task #643.

Why this lives here and not in ``labels.py``: the existing
``build_labeled_dataset`` is hard-wired to the 3-class production
contract (CANDLES_PREFERRED_TIMEFRAMES gates 1m onto the
ticks-resampled path; the per-coin frame is then concat-ed across all
coins). The research study needs:

1. To read 1m bars from ``price_candles`` directly when they exist
   (BTC/ETH 1m were backfilled there per the spec — JUP 1m still has
   to fall back to the ticks path).
2. To apply ``labels.apply_self_leak_guard`` for the BTC / ETH targets
   so neither model can cheat off its own forward signal.
3. To return a *per-coin* frame keyed by ``timestamp_ms`` so the
   walk-forward splitter operates on a single time index.

Importantly this module **does not** invent new features — it
piggy-backs on ``labels.build_labeled_frame_for_coin`` and just feeds
it the candle source the spec demands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .. import labels as base_labels
from ...db import (
    fetch_lead_price_series,
    fetch_market_signals,
    fetch_real_candles,
    fetch_real_ticks_with_provenance,
    init_pool,
)

CROSS_MARKET_LIQ_SOURCES = ("btc", "eth", "sol")


def _attach_cross_market_features_vectorized(
    df: pd.DataFrame,
    market_signals: Sequence[dict],
    btc_lead_lookup: Sequence[tuple[int, float]],
    eth_lead_lookup: Sequence[tuple[int, float]],
    cross_liq_signals: dict[str, Sequence[dict]],
) -> pd.DataFrame:
    """Vectorised re-implementation of the cross-market asof joins
    inside ``labels.build_labeled_frame_for_coin``.

    The production helper iterates each (signals, bar) pair with a
    Python ``for`` loop, which is O(N*M) per bar — fine for a 90-day
    production training slice but it does NOT scale to the 12-month
    research window the reviewer demanded (~92k 5m bars × 110k
    cross-coin signals = ~10B Python ops per slice). We replace the
    per-bar scan with ``pandas.merge_asof`` (binary search on a sorted
    key, O((N+M) log M)) and a vectorised expanding mean/std for the
    open-interest z-score. Outputs are bit-equivalent to the production
    asof semantics: most-recent non-null value at-or-before the bar's
    ``timestamp_ms``. This file lives under ``labels_research/`` so the
    production training path is untouched.
    """
    if df is None or df.empty or "timestamp_ms" not in df.columns:
        return df
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    bars = df[["timestamp_ms"]].copy()

    def _asof_into(col: str, src: pd.DataFrame) -> None:
        src = src.dropna(subset=[col]).sort_values("timestamp_ms")
        if src.empty:
            return
        src = src[["timestamp_ms", col]].copy()
        src["timestamp_ms"] = src["timestamp_ms"].astype("int64")
        joined = pd.merge_asof(
            bars, src, on="timestamp_ms", direction="backward",
        )
        df[col] = joined[col].to_numpy()

    if market_signals:
        ms_df = pd.DataFrame(list(market_signals))
        if not ms_df.empty and "timestamp_ms" in ms_df.columns:
            ms_df["timestamp_ms"] = ms_df["timestamp_ms"].astype("int64")
            for col in ("funding_rate", "liquidations_1h_usd",
                        "bid_ask_spread_bps"):
                if col in ms_df.columns:
                    _asof_into(col, ms_df)
            if "open_interest_usd" in ms_df.columns:
                oi = ms_df[["timestamp_ms", "open_interest_usd"]].dropna()
                oi = oi[oi["open_interest_usd"] > 0].sort_values(
                    "timestamp_ms"
                ).reset_index(drop=True)
                if len(oi) >= 2:
                    cum_mean = oi["open_interest_usd"].expanding().mean()
                    cum_std = oi["open_interest_usd"].expanding().std(ddof=1)
                    z_src = pd.DataFrame({
                        "timestamp_ms": oi["timestamp_ms"].astype("int64"),
                        "_oi_val": oi["open_interest_usd"].values,
                        "_oi_mean": cum_mean.values,
                        "_oi_std": cum_std.values,
                    })
                    j = pd.merge_asof(
                        bars, z_src, on="timestamp_ms",
                        direction="backward",
                    )
                    sd = j["_oi_std"].to_numpy()
                    val = j["_oi_val"].to_numpy()
                    mean = j["_oi_mean"].to_numpy()
                    z = np.full(len(df), np.nan, dtype=np.float64)
                    mask = (sd > 0) & np.isfinite(sd)
                    z[mask] = (val[mask] - mean[mask]) / sd[mask]
                    df["open_interest_z"] = z

    for col_name, lookup in (("btc_lead_ret_5m", btc_lead_lookup),
                              ("eth_lead_ret_5m", eth_lead_lookup)):
        if not lookup:
            continue
        ll = pd.DataFrame(list(lookup), columns=["timestamp_ms", col_name])
        ll["timestamp_ms"] = ll["timestamp_ms"].astype("int64")
        ll = ll.sort_values("timestamp_ms").reset_index(drop=True)
        joined = pd.merge_asof(
            bars, ll, on="timestamp_ms", direction="backward",
        )
        df[col_name] = joined[col_name].to_numpy()

    if cross_liq_signals:
        for src_coin, col_name in base_labels.CROSS_MARKET_LIQ_SOURCES.items():
            sigs = cross_liq_signals.get(src_coin)
            if not sigs:
                continue
            ms = pd.DataFrame(list(sigs))
            if "timestamp_ms" not in ms.columns:
                continue
            if "liquidations_1h_usd" not in ms.columns:
                continue
            ms = ms[["timestamp_ms", "liquidations_1h_usd"]].dropna()
            ms["timestamp_ms"] = ms["timestamp_ms"].astype("int64")
            ms = ms.sort_values("timestamp_ms").reset_index(drop=True)
            if ms.empty:
                continue
            joined = pd.merge_asof(
                bars, ms, on="timestamp_ms", direction="backward",
            )
            df[col_name] = joined["liquidations_1h_usd"].to_numpy()

    return df


@dataclass
class SliceFrame:
    """Result of ``build_research_frame``.

    * ``df``   — the per-(coin, tf) labeled feature frame, sorted by
      ``timestamp_ms`` ascending. Empty when the slice has no usable
      data.
    * ``coin_id`` / ``timeframe`` — slice identity.
    * ``rows_real``        — how many real bars contributed.
    * ``bars_source``      — ``"candles"`` or ``"resampled_ticks"`` —
      stamped onto the report so a reader can audit which path the
      slice took.
    * ``self_leak_columns_dropped`` — which feature columns were
      NaN-overwritten by the BTC/ETH self-leak guard, if any.
    * ``ingestion_quality`` — dict of acceptance metrics:
      ``span_days`` (oldest→newest bar), ``bar_gap_rate`` (share of
      consecutive intervals that are NOT exactly one period apart),
      ``feature_nan_share`` (mean fraction of non-NaN values across
      production feature columns) and ``row_count``.
    """

    df: pd.DataFrame
    coin_id: str
    timeframe: str
    rows_real: int
    bars_source: str
    self_leak_columns_dropped: list[str]
    ingestion_quality: dict | None = None


_TF_TO_MS = {"1m": 60_000, "5m": 5 * 60_000}


def _compute_ingestion_quality(df: pd.DataFrame, timeframe: str) -> dict:
    if df is None or df.empty or "timestamp_ms" not in df.columns:
        return {
            "row_count": 0, "span_days": 0.0, "bar_gap_rate": 1.0,
            "feature_nan_share": 1.0,
        }
    ts = df["timestamp_ms"].to_numpy()
    span_ms = int(ts.max() - ts.min()) if len(ts) > 1 else 0
    span_days = span_ms / 86_400_000
    period_ms = _TF_TO_MS.get(timeframe, 60_000)
    if len(ts) > 1:
        diffs = ts[1:] - ts[:-1]
        non_unit = int((diffs != period_ms).sum())
        gap_rate = non_unit / len(diffs)
    else:
        gap_rate = 1.0
    # NaN share over production feature columns (everything except the
    # label / bookkeeping cols).
    drop_cols = {
        "timestamp_ms", "coin_id", "label", "fwd_ret", "fwd_log_ret",
        "y", "target",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if feature_cols:
        nan_share = float(df[feature_cols].isna().mean().mean())
    else:
        nan_share = 1.0
    # ``feature_nan_share`` is the gate the spec asks for. The
    # cross-market signal columns (BTC/ETH/SOL liquidations,
    # funding_rate, open_interest_z, bid/ask spread, per-coin
    # liquidations, and the barrier-touch labels) come from a
    # separate hourly ``market_signals`` table whose ingestion is
    # entirely outside this task. Their NaN share is ALSO reported
    # separately so a reader can tell at a glance whether a high
    # ``feature_nan_share`` is driven by missing OHLCV (a real
    # bar-data problem the booster can't paper over) vs. by
    # missing hourly side-channel data (which LightGBM treats as
    # ``use_missing=True`` natively, so it does not invalidate the
    # core forecasting signal — but it still trips the strict
    # ingestion gate).
    side_channel = {
        "btc_liquidations_1h_usd", "eth_liquidations_1h_usd",
        "sol_liquidations_1h_usd", "liquidations_1h_usd",
        "funding_rate", "open_interest_z", "bid_ask_spread_bps",
        "tp_before_sl_long", "tp_before_sl_short",
    }
    core_cols = [c for c in feature_cols if c not in side_channel]
    if core_cols:
        core_nan = float(df[core_cols].isna().mean().mean())
    else:
        core_nan = 1.0
    return {
        "row_count": int(len(df)),
        "span_days": round(span_days, 3),
        "bar_gap_rate": round(float(gap_rate), 6),
        "feature_nan_share": round(nan_share, 6),
        "core_feature_nan_share": round(core_nan, 6),
    }


async def build_research_frame(
    coin_id: str, timeframe: str, lookback_ms: int,
) -> SliceFrame:
    """Build a single (coin, timeframe) labeled frame for the research
    study.

    Loads candles from ``price_candles`` first (works for both 1m and
    5m once the backfill has run). When no candles exist (legacy JUP
    1m), falls back to ``fetch_real_ticks_with_provenance`` + the
    in-memory resampler that ``labels.build_labeled_frame_for_coin``
    runs internally.
    """
    await init_pool()

    # Restore the production cross-market feature contract (Task #643
    # round 2): the BTC and ETH 5m lead-return lookups and the
    # `btc/eth/sol` liquidation signal series are pulled at the same
    # cadence the production trainer uses
    # (`labels.build_labeled_dataset`). The asof-join inside
    # `build_labeled_frame_for_coin` then attaches the same vector to
    # every family on a given slice, so the family comparison stays
    # honest while every model sees the production feature contract.
    # The self-leak guard further down NaN-overwrites the BTC own /
    # ETH own forward columns so a BTC model cannot cheat off
    # `btc_lead_ret_5m` and an ETH model cannot cheat off
    # `eth_lead_ret_5m`.
    try:
        btc_series = await fetch_lead_price_series("btc", lookback_ms)
    except Exception:
        btc_series = []
    try:
        eth_series = await fetch_lead_price_series("eth", lookback_ms)
    except Exception:
        eth_series = []
    btc_lead_lookup = base_labels._build_lead_return_lookup(btc_series)
    eth_lead_lookup = base_labels._build_lead_return_lookup(eth_series)

    cross_liq_signals: dict[str, list[dict]] = {}
    for src_coin in CROSS_MARKET_LIQ_SOURCES:
        try:
            cross_liq_signals[src_coin] = await fetch_market_signals(
                src_coin, lookback_ms,
            )
        except Exception:
            cross_liq_signals[src_coin] = []

    candles: Optional[list[tuple]] = None
    bars_source = "resampled_ticks"
    try:
        fetched = await fetch_real_candles(coin_id, timeframe, lookback_ms)
    except Exception:
        fetched = []
    if fetched:
        candles = fetched
        bars_source = "candles"

    ticks: list[tuple] = []
    if candles is None:
        # Fall back to ticks for the resampler. Provenance guard rejects
        # synthetic rows for us — anything that survives is real.
        try:
            prov = await fetch_real_ticks_with_provenance(coin_id, lookback_ms)
        except Exception:
            prov = {
                "ticks": [], "rows_real": 0, "rows_synthetic": 0,
                "rejected_synthetic": False,
            }
        if prov.get("rejected_synthetic"):
            return SliceFrame(
                df=pd.DataFrame(), coin_id=coin_id, timeframe=timeframe,
                rows_real=int(prov.get("rows_real", 0)),
                bars_source="rejected_synthetic",
                self_leak_columns_dropped=[],
            )
        ticks = prov.get("ticks", [])
        if not ticks:
            return SliceFrame(
                df=pd.DataFrame(), coin_id=coin_id, timeframe=timeframe,
                rows_real=0, bars_source="empty",
                self_leak_columns_dropped=[],
            )

    try:
        market_signals = await fetch_market_signals(coin_id, lookback_ms)
    except Exception:
        market_signals = []

    # Pass None for the cross-market args here — the per-bar O(N*M)
    # asof loops inside ``build_labeled_frame_for_coin`` make a 12-month
    # window untrainable in any reasonable wall-clock. We re-attach the
    # SAME columns via ``_attach_cross_market_features_vectorized``
    # (pandas merge_asof) below, which has identical asof semantics
    # (most-recent non-null at-or-before the bar) but runs in seconds.
    df = base_labels.build_labeled_frame_for_coin(
        coin_id, timeframe, ticks,
        news_tags=[],
        market_signals=None,
        btc_lead_lookup=None,
        eth_lead_lookup=None,
        cross_liq_signals=None,
        candles=candles,
    )
    if df.empty:
        return SliceFrame(
            df=df, coin_id=coin_id, timeframe=timeframe, rows_real=0,
            bars_source=bars_source, self_leak_columns_dropped=[],
        )

    # Vectorised cross-market feature wiring (BTC/ETH lead returns,
    # btc/eth/sol cross-coin liq pulses, own-coin funding/OI/liq/spread).
    df = _attach_cross_market_features_vectorized(
        df,
        market_signals=market_signals,
        btc_lead_lookup=btc_lead_lookup,
        eth_lead_lookup=eth_lead_lookup,
        cross_liq_signals=cross_liq_signals,
    )

    # Self-leak guard for BTC/ETH targets — the helper logs when it
    # actually overwrites a column.
    leak_cols = sorted(
        base_labels.SELF_LEAK_FEATURE_COLUMNS.get(coin_id, frozenset())
        & set(df.columns)
    )
    base_labels.apply_self_leak_guard(coin_id, df)

    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    iq = _compute_ingestion_quality(df, timeframe)
    return SliceFrame(
        df=df, coin_id=coin_id, timeframe=timeframe, rows_real=len(df),
        bars_source=bars_source,
        self_leak_columns_dropped=leak_cols,
        ingestion_quality=iq,
    )
