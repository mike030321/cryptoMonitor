"""Phase 4 — meta-model reliability features.

The meta-model is supposed to learn an *adaptive* gate, not just a
pattern over specialist outputs. To do that, training rows and live
inference both need to know how reliable the base model has been
recently for the same (coin, regime) bucket. This module computes
those rolling reliability stats once, in one place, so dataset
construction and the live `/ml/meta/predict` route stay in lockstep.

Output feature schema (added to META_FEATURE_COLUMNS):
  - reliability_coin_winrate_30d:    trailing 30d hit rate per coin
  - reliability_regime_winrate_30d:  trailing 30d hit rate per regime
  - reliability_coin_n_30d:          trailing 30d sample count per coin
  - reliability_regime_n_30d:        trailing 30d sample count per regime

`win` = base predicted direction (argmax(prob_up, prob_down)) matched
the realized return sign. Default 0.5 with n=0 when the trailing
window is empty so the model can learn that no-data == low-trust.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .. import db as db_mod

logger = logging.getLogger(__name__)

RELIABILITY_FEATURE_COLUMNS: list[str] = [
    "reliability_coin_winrate_30d",
    "reliability_regime_winrate_30d",
    "reliability_coin_n_30d",
    "reliability_regime_n_30d",
]

DEFAULT_RELIABILITY: dict[str, float] = {
    "reliability_coin_winrate_30d": 0.5,
    "reliability_regime_winrate_30d": 0.5,
    "reliability_coin_n_30d": 0.0,
    "reliability_regime_n_30d": 0.0,
}

WINDOW_DAYS = 30

_RELIABILITY_QUERY = """
SELECT
    pj.created_at,
    pj.coin_id,
    pj.timeframe,
    pj.regime_label,
    pj.prob_up,
    pj.prob_down,
    pj.realized_return_pct
FROM prediction_journal pj
WHERE pj.brain = 'QUANT'
  AND pj.timeframe = $1
  AND pj.realized_return_pct IS NOT NULL
  AND pj.created_at >= $2
  AND pj.created_at < $3
"""


def _row_was_win(prob_up: float, prob_down: float, realized: float) -> Optional[bool]:
    if realized is None:
        return None
    pred_up = (prob_up or 0) >= (prob_down or 0)
    real_up = realized > 0
    real_dn = realized < 0
    if not real_up and not real_dn:
        return None  # zero-return rows can't score the directional call
    return (pred_up and real_up) or ((not pred_up) and real_dn)


async def compute_reliability_features(
    timeframe: str,
    coin_id: str,
    regime: Optional[str],
    ref_time: Optional[datetime] = None,
) -> dict[str, float]:
    """Live-inference path. Single short SQL pulls the trailing-30d
    journal slice for this timeframe and computes the four features.
    Returns DEFAULT_RELIABILITY on any failure so the meta-model never
    crashes the trade path.
    """
    try:
        if ref_time is None:
            ref_time = datetime.now(timezone.utc)
        since = ref_time - timedelta(days=WINDOW_DAYS)
        pool = await db_mod.init_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(_RELIABILITY_QUERY, timeframe, since, ref_time)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.debug("reliability_lookup_failed", extra={"err": str(exc)})
        return dict(DEFAULT_RELIABILITY)

    coin_wins = coin_n = 0
    regime_wins = regime_n = 0
    for r in rows:
        win = _row_was_win(r["prob_up"], r["prob_down"], r["realized_return_pct"])
        if win is None:
            continue
        if r["coin_id"] == coin_id:
            coin_n += 1
            if win:
                coin_wins += 1
        if regime and r["regime_label"] == regime:
            regime_n += 1
            if win:
                regime_wins += 1
    return {
        "reliability_coin_winrate_30d": (coin_wins / coin_n) if coin_n > 0 else 0.5,
        "reliability_regime_winrate_30d": (regime_wins / regime_n) if regime_n > 0 else 0.5,
        "reliability_coin_n_30d": float(coin_n),
        "reliability_regime_n_30d": float(regime_n),
    }


def attach_reliability_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Training-path equivalent of `compute_reliability_features`. Walks
    rows in time order and computes trailing-30d per-(coin) and
    per-(regime) win rates STRICTLY before each row's `__created_at__`,
    so there is no leakage. Mutates and returns `df` with the four
    reliability columns populated.
    """
    if df.empty:
        for col, default in DEFAULT_RELIABILITY.items():
            df[col] = default
        return df

    df = df.sort_values("__created_at__", kind="stable").reset_index(drop=True)

    # Per-key rolling deque-style state. Keep (created_at, was_win) so
    # we can drop entries that fall out of the 30d window.
    coin_buf: dict[str, list[tuple[datetime, bool]]] = {}
    regime_buf: dict[str, list[tuple[datetime, bool]]] = {}
    coin_wr: list[float] = []
    coin_n: list[float] = []
    regime_wr: list[float] = []
    regime_n: list[float] = []

    # We need the underlying source data (prob_up/down/realized) to know
    # if past rows were wins. The dataset already filters realized!=None.
    # We piggy-back on base_prob_up/base_prob_down + the action label
    # (which encodes the realized direction modulo cost): action=long
    # means realized>cost (so realized>0), action=short means realized<0.
    for i, row in df.iterrows():
        ts = row["__created_at__"]
        if not isinstance(ts, datetime):
            try:
                ts = pd.to_datetime(ts).to_pydatetime()
            except Exception:
                ts = datetime.now(timezone.utc)
        cutoff = ts - timedelta(days=WINDOW_DAYS)
        coin = row.get("__coin_id__") or ""
        regime = next(
            (r for r in [
                "trending_up", "trending_down", "range_chop",
                "high_vol_breakout", "low_vol_compression", "panic_liquidation",
            ] if row.get(f"regime_{r}") == 1.0),
            "",
        )

        # Drop expired entries from each key's buffer (cheap: short lists).
        for buf in (coin_buf.get(coin, []), regime_buf.get(regime, [])):
            while buf and buf[0][0] < cutoff:
                buf.pop(0)

        cb = coin_buf.get(coin, [])
        rb = regime_buf.get(regime, [])
        cw = sum(1 for _, w in cb if w)
        rw = sum(1 for _, w in rb if w)
        coin_wr.append(cw / len(cb) if cb else 0.5)
        coin_n.append(float(len(cb)))
        regime_wr.append(rw / len(rb) if rb else 0.5)
        regime_n.append(float(len(rb)))

        # Now record THIS row's outcome into the buffers for FUTURE rows
        # (strictly forward — never visible to the row that just emitted).
        action = row.get("__action__")
        pred_up = row["base_prob_up"] >= row["base_prob_down"]
        if action in ("long", "short"):
            real_up = action == "long"
            was_win = (pred_up and real_up) or ((not pred_up) and (not real_up))
            coin_buf.setdefault(coin, []).append((ts, was_win))
            if regime:
                regime_buf.setdefault(regime, []).append((ts, was_win))
        # action == "no_trade" — realized was within ±cost; we don't
        # score the directional call (treat as undecidable).

    df["reliability_coin_winrate_30d"] = coin_wr
    df["reliability_regime_winrate_30d"] = regime_wr
    df["reliability_coin_n_30d"] = coin_n
    df["reliability_regime_n_30d"] = regime_n
    return df
