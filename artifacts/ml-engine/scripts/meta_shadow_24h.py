"""Task #455 — Compute the 24h shadow / serving snapshot for the
Meta-model from the live `prediction_journal`.

Writes `.local/cleanup/meta-rebuild/shadow-24h.json` with:

  - Per-timeframe distribution of `gates_applied.meta_kind` over the
    last 24 hours (lightgbm vs heuristic vs unknown).
  - Per-timeframe distribution of `gates_applied.meta_action` (long /
    short / no_trade) — this gives both the lightgbm-share check
    (>= 60% on slices that have a promoted trained head) AND the
    abstain-rate check (in the 30%–80% band).
  - Top served `meta_version`s per timeframe so an operator can confirm
    which `__meta__/{tf}/{version}` is actually live.

This script is read-only: it never moves the `latest` pointer and
never trains. Run after a deploy / promotion to take a fresh snapshot:

    cd artifacts/ml-engine
    ../../.pythonlibs/bin/python -m scripts.meta_shadow_24h

"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from app import db as db_mod  # noqa: E402

OUT_DIR = REPO_ROOT / ".local" / "cleanup" / "meta-rebuild"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Pure helpers (kept module-level so tests can call them without DB).
# ----------------------------------------------------------------------
LIGHTGBM_SHARE_MIN = 0.60
ABSTAIN_BAND_LO = 0.30
ABSTAIN_BAND_HI = 0.80


def summarize_timeframe(bucket: dict) -> dict:
    """Convert a per-tf accumulator into the JSON-serialisable summary
    block written to shadow-24h.json.

    Definitions (single source of truth):

      * `lightgbm_share` = lightgbm rows / TOTAL rows in the 24h
        window (including unknown / pre-wiring rows). This is the share
        the operator actually sees served by the trained head.

      * `abstain_rate` = exact count of (kind ∈ {lightgbm, heuristic}
        AND action == "no_trade") rows divided by the count of
        known-kind rows (kind ∈ {lightgbm, heuristic}). Pre-wiring
        rows whose meta_kind is "unknown" are excluded from BOTH
        numerator and denominator so they do not deflate the rate.
        The exact cross-tab is taken from the row-level
        `by_kind_action` accumulator — no proportional approximation.

      * `lightgbm_share_check_60pct`:
          - "n/a (no lightgbm rows on this tf)" if lgb_n == 0
          - "pass" if lightgbm_share >= 0.60
          - "fail (mostly unknown — pre-wiring rows present)" if
            unknown_n > lgb_n + heur_n
          - "fail" otherwise

      * `abstain_rate_in_band_30_to_80pct`:
          - "n/a (no known-kind rows)" if known_n == 0
          - "pass" if 0.30 <= abstain_rate <= 0.80
          - "fail (out of band)" otherwise
    """
    total = int(bucket.get("total", 0))
    by_kind = bucket.get("by_meta_kind", {})
    by_action = bucket.get("by_meta_action", {})
    by_version = bucket.get("by_meta_version", {})
    by_kind_action: dict[tuple[str, str], int] = bucket.get("by_kind_action", {})

    lgb_n = int(by_kind.get("lightgbm", 0))
    heur_n = int(by_kind.get("heuristic", 0))
    unknown_n = int(by_kind.get("unknown", 0))
    known_n = lgb_n + heur_n

    no_trade_n = int(by_action.get("no_trade", 0))
    long_n = int(by_action.get("long", 0))
    short_n = int(by_action.get("short", 0))

    # Exact known × no_trade count from the (kind, action) cross-tab.
    no_trade_known = sum(
        int(n) for (k, a), n in by_kind_action.items()
        if k in ("lightgbm", "heuristic") and a == "no_trade"
    )

    lgb_share = (lgb_n / total) if total > 0 else 0.0
    abstain_rate = (no_trade_known / known_n) if known_n > 0 else 0.0

    if lgb_n == 0:
        lgb_check = "n/a (no lightgbm rows on this tf)"
    elif lgb_share >= LIGHTGBM_SHARE_MIN:
        lgb_check = "pass"
    elif unknown_n > known_n:
        lgb_check = "fail (mostly unknown — pre-wiring rows present)"
    else:
        lgb_check = "fail"

    if known_n == 0:
        ab_check = "n/a (no known-kind rows)"
    elif ABSTAIN_BAND_LO <= abstain_rate <= ABSTAIN_BAND_HI:
        ab_check = "pass"
    else:
        ab_check = "fail (out of band)"

    # Serialise the cross-tab in a JSON-friendly shape so the operator
    # can audit the exact counts behind the abstain rate.
    cross_tab: dict[str, dict[str, int]] = {}
    for (k, a), n in by_kind_action.items():
        cross_tab.setdefault(k, {})[a] = int(n)

    return {
        "total_predictions_24h": total,
        "by_meta_kind": dict(by_kind),
        "by_meta_action": {"long": long_n, "short": short_n, "no_trade": no_trade_n},
        "by_kind_action": cross_tab,
        "top_meta_versions": dict(Counter(by_version).most_common(8)),
        "lightgbm_share": round(lgb_share, 4),
        "lightgbm_share_check_60pct": lgb_check,
        "known_no_trade_count": no_trade_known,
        "known_kind_count": known_n,
        "abstain_rate": round(abstain_rate, 4),
        "abstain_rate_in_band_30_to_80pct": ab_check,
        "thresholds_used": {
            "lightgbm_share_minimum": LIGHTGBM_SHARE_MIN,
            "abstain_band": [ABSTAIN_BAND_LO, ABSTAIN_BAND_HI],
        },
    }


async def _amain() -> int:
    pool = await db_mod.init_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
              timeframe,
              gates_applied,
              direction,
              became_trade
            FROM prediction_journal
            WHERE brain = 'QUANT'
              AND created_at >= now() - interval '24 hours'
              AND timeframe IS NOT NULL
            """
        )

    per_tf: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "by_meta_kind": Counter(),
        "by_meta_version": Counter(),
        "by_meta_action": Counter(),
        "by_kind_action": Counter(),
    })

    for r in rows:
        tf = r["timeframe"]
        gates = r["gates_applied"] or {}
        if isinstance(gates, str):
            try:
                gates = json.loads(gates)
            except Exception:
                gates = {}
        kind = (gates.get("meta_kind") or "unknown") or "unknown"
        version = (gates.get("meta_version") or "unversioned") or "unversioned"
        action = gates.get("meta_action")
        if action not in ("long", "short", "no_trade"):
            # Pre-/ml/meta/predict rows: fall back to direction so the
            # abstain-rate band is computed on a consistent basis.
            action = "long" if r["direction"] == "up" else (
                "short" if r["direction"] == "down" else "no_trade"
            )
        bucket = per_tf[tf]
        bucket["total"] += 1
        bucket["by_meta_kind"][kind] += 1
        bucket["by_meta_version"][version] += 1
        bucket["by_meta_action"][action] += 1
        bucket["by_kind_action"][(kind, action)] += 1

    summary: dict[str, dict[str, Any]] = {tf: summarize_timeframe(b) for tf, b in per_tf.items()}

    payload = {
        "task": "455 — meta-model 24h shadow / serving snapshot",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": 24,
        "source": (
            "select timeframe, gates_applied->>'meta_kind', gates_applied->>'meta_version', "
            "gates_applied->>'meta_action' from prediction_journal where brain='QUANT' "
            "and created_at >= now() - interval '24 hours'"
        ),
        "thresholds": {
            "lightgbm_share_minimum": LIGHTGBM_SHARE_MIN,
            "abstain_rate_band": [ABSTAIN_BAND_LO, ABSTAIN_BAND_HI],
        },
        "definitions": {
            "lightgbm_share": "lightgbm rows / total 24h rows (includes pre-/ml/meta/predict 'unknown' rows in the denominator)",
            "abstain_rate": "no_trade rows attributed to known-kind rows / known-kind rows. Pre-wiring 'unknown' rows are excluded from BOTH numerator and denominator so they do not deflate the rate.",
        },
        "per_timeframe": summary,
        "notes": [
            "This snapshot is the live-runtime evidence companion to the offline holdout proxy in holdout-{tf}.json.",
            "lightgbm_share_check_60pct returns 'n/a' for tfs with zero lightgbm rows (rebuild left them heuristic — e.g. 1d below the 200-row floor) and 'fail (mostly unknown — pre-wiring rows present)' when more than half the rows pre-date the /ml/meta/predict wiring.",
            "abstain_rate_in_band check is the operational sanity bound (the meta-model should neither trade everything nor abstain on everything).",
        ],
    }

    (OUT_DIR / "shadow-24h.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"per_tf_summary": {tf: {
        "n": s["total_predictions_24h"],
        "lgb_share": s["lightgbm_share"],
        "abstain_rate": s["abstain_rate"],
    } for tf, s in summary.items()}}, indent=2))
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
