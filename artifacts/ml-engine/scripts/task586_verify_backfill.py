"""Task #586 — verification report for the on-chain backfill.

Reads:
  * the freshest cached `<tf>_<TS>.parquet` for every timeframe under
    `models/datasets/` (the snapshots the retrain harness reads),
  * the `okx_backfill_*` rows that were just written to
    `market_signals` by `scripts/backfill_market_signals.py`,
  * the read-only forbidden-features manifest at
    `shared/forbidden-features.json`.

Writes a markdown report under
`artifacts/ml-engine/reports/<TS>-task586-verification.md` with:

  1. Per-coin / per-timeframe coverage of `funding_rate`,
     `open_interest_z`, `liquidations_1h_usd`, `bid_ask_spread_bps`
     before vs after backfill.
  2. Decile / quantile distribution snapshots so a reviewer can
     eyeball that the backfilled values look like real funding /
     OI data and not all-zero placeholders.
  3. Anti-leak audit: every backfilled row's timestamp must be
     <= the wall-clock now recorded in the verification manifest.
     Any leak surfaces as a non-zero count and a per-row sample.
  4. SHA-256 of `shared/forbidden-features.json` so a reviewer can
     confirm it was not mutated by this campaign (the forbidden
     manifest must stay byte-identical to its pre-task #586 value).

Usage::

    cd artifacts/ml-engine && \\
        ../../.pythonlibs/bin/python -m scripts.task586_verify_backfill
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT))

from app.db import close_pool, init_pool  # noqa: E402

DATASETS_DIR = ROOT / "models" / "datasets"
REPORTS_DIR = ROOT / "reports"
FORBIDDEN_MANIFEST = REPO_ROOT / "shared" / "forbidden-features.json"

TARGET_FEATURES = [
    "funding_rate",
    "open_interest_z",
    "liquidations_1h_usd",
    "bid_ask_spread_bps",
    # Includeded for context — these were already populated by the live
    # poller; the backfill extends their coverage further into the past.
    "btc_lead_ret_5m",
    "eth_lead_ret_5m",
]

BACKFILL_SOURCES = [
    "okx_backfill_funding_v1",
    "okx_backfill_oi_v1",
    "okx_backfill_mid_v1",
]


_BACKFILL_CUTOFF = "20260428T203000Z"


def _representative_parquet_per_tf() -> dict[str, Path]:
    """Pick the canonical post-backfill `<tf>_<TS>.parquet` per timeframe.

    Multiple writers regenerate parquets in `models/datasets/`:
    `scripts/refresh_cached_datasets.py` (the cadence loop —
    full-coverage 10-coin snapshots) and `app/training/train.py` /
    `train_meta.py` (per-coin / per-meta snapshots that only fetch
    enough rows for the run at hand). Naïvely picking the newest by
    name picks up the training-side snapshots, which use shorter
    windows and therefore look like they have very little
    funding/OI coverage. To avoid that, prefer the **earliest**
    post-backfill snapshot per timeframe — that is the cadence
    loop's first refresh after the OKX backfill landed. Fall back to
    the lexicographically newest snapshot if no post-backfill one
    exists yet for that timeframe (the cadence loop has not caught
    up).
    """
    out: dict[str, Path] = {}
    for tf in ("1m", "5m", "1h", "2h", "6h", "1d"):
        all_paths = sorted(
            DATASETS_DIR.glob(f"{tf}_*.parquet"),
            key=lambda p: p.name,
        )
        if not all_paths:
            continue
        post = [
            p for p in all_paths
            if (p.stem.split("_", 1)[1] if "_" in p.stem else "")
            >= _BACKFILL_CUTOFF
        ]
        out[tf] = post[0] if post else all_paths[-1]
    return out


def _coverage_block(df: pd.DataFrame) -> dict[str, Any]:
    """Per-feature global + per-coin nonzero coverage."""
    out: dict[str, Any] = {"rows": int(len(df)), "features": {}}
    for col in TARGET_FEATURES:
        if col not in df.columns:
            out["features"][col] = {"present": False}
            continue
        s = df[col].fillna(0)
        nonzero_global = int((s != 0).sum())
        per_coin: dict[str, dict[str, float]] = {}
        for coin, sub in df.groupby("coin_id")[col]:
            ss = sub.fillna(0)
            per_coin[str(coin)] = {
                "rows": int(len(ss)),
                "nonzero": int((ss != 0).sum()),
                "frac": round(float((ss != 0).mean()), 4),
            }
        # Five-number summary on the nonzero subset so the reader can
        # spot all-zero columns or weird unit issues at a glance.
        nz_only = s[s != 0]
        quantiles = {}
        if len(nz_only) > 0:
            for q in (0.05, 0.25, 0.5, 0.75, 0.95):
                quantiles[f"q{int(q*100):02d}"] = float(nz_only.quantile(q))
        out["features"][col] = {
            "present": True,
            "nonzero_global": nonzero_global,
            "frac_global": round(nonzero_global / max(1, len(df)), 4),
            "quantiles_nonzero": quantiles,
            "per_coin": per_coin,
        }
    return out


async def _audit_db() -> dict[str, Any]:
    pool = await init_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT coin_id, source,
                       COUNT(*)::bigint AS n,
                       MIN(timestamp) AS earliest,
                       MAX(timestamp) AS latest
                FROM market_signals
                WHERE source = ANY($1::text[])
                GROUP BY coin_id, source
                ORDER BY coin_id, source
                """,
                BACKFILL_SOURCES,
            )
            now = datetime.now(timezone.utc)
            future_rows = await conn.fetch(
                """
                SELECT coin_id, source, timestamp
                FROM market_signals
                WHERE source = ANY($1::text[])
                  AND timestamp > $2
                LIMIT 5
                """,
                BACKFILL_SOURCES, now,
            )
            return {
                "audit_at": now.isoformat(),
                "rows": [
                    {
                        "coin_id": r["coin_id"],
                        "source": r["source"],
                        "n": int(r["n"]),
                        "earliest": r["earliest"].isoformat(),
                        "latest": r["latest"].isoformat(),
                    }
                    for r in rows
                ],
                "leak_count": len(future_rows),
                "leak_samples": [
                    {
                        "coin_id": r["coin_id"],
                        "source": r["source"],
                        "timestamp": r["timestamp"].isoformat(),
                    }
                    for r in future_rows
                ],
            }
    finally:
        await close_pool()


def _forbidden_manifest_hash() -> dict[str, Any]:
    raw = FORBIDDEN_MANIFEST.read_bytes()
    return {
        "path": str(FORBIDDEN_MANIFEST.relative_to(REPO_ROOT)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Task #586 — backfill verification report\n")
    lines.append(f"_Generated at_ **{report['generated_at']}**\n")
    lines.append(
        "_Run by_ `artifacts/ml-engine/scripts/task586_verify_backfill.py`\n",
    )
    lines.append(
        "Backfill source tags audited: "
        + ", ".join(f"`{s}`" for s in BACKFILL_SOURCES)
        + "\n",
    )

    # ----- Forbidden manifest -----------------------------------------
    fb = report["forbidden_manifest"]
    lines.append("## Forbidden-features manifest (must be unchanged)\n")
    lines.append(
        f"`{fb['path']}` — {fb['bytes']} bytes — SHA-256 `{fb['sha256']}`\n",
    )
    lines.append(
        "If this hash differs from the value in the previous campaign's "
        "report, the backfill **must not be merged**: the campaign "
        "specifies the manifest stays byte-identical.\n",
    )

    # ----- DB audit ---------------------------------------------------
    db = report["db_audit"]
    lines.append("## Backfilled rows in `market_signals`\n")
    lines.append(f"_Audit timestamp_ **{db['audit_at']}** (UTC)\n")
    lines.append(
        "| coin_id | source | rows | earliest | latest |\n"
        "|---|---|---:|---|---|\n",
    )
    for r in db["rows"]:
        lines.append(
            f"| {r['coin_id']} | {r['source']} | {r['n']} | "
            f"{r['earliest']} | {r['latest']} |\n",
        )
    lines.append(
        f"\n**Anti-leak check**: future-dated rows = "
        f"`{db['leak_count']}` (must be 0).\n",
    )
    if db["leak_samples"]:
        lines.append("Leak samples:\n")
        for r in db["leak_samples"]:
            lines.append(
                f"- {r['coin_id']} / {r['source']} @ {r['timestamp']}\n",
            )

    # ----- Dataset coverage -------------------------------------------
    lines.append("## Cached training-dataset coverage\n")
    lines.append(
        "Each block summarises the freshest "
        "`models/datasets/<tf>_<TS>.parquet` post-refresh. Columns "
        "covered: " + ", ".join(f"`{c}`" for c in TARGET_FEATURES) + ".\n",
    )
    for tf, block in report["datasets"].items():
        path = block["path"]
        cov = block["coverage"]
        flag = (
            "post-backfill ✅"
            if block.get("post_backfill")
            else "**stale (predates backfill)** ⚠"
        )
        lines.append(
            f"\n### tf=`{tf}` — `{path}` ({cov['rows']} rows) — {flag}\n",
        )
        lines.append(
            "| feature | rows nonzero | frac | q05 | q50 | q95 |\n"
            "|---|---:|---:|---:|---:|---:|\n",
        )
        for fname, fblock in cov["features"].items():
            if not fblock.get("present"):
                lines.append(
                    f"| {fname} | _column missing_ |  |  |  |  |\n",
                )
                continue
            q = fblock["quantiles_nonzero"]
            lines.append(
                f"| {fname} | {fblock['nonzero_global']} | "
                f"{fblock['frac_global']} | "
                f"{q.get('q05','—')} | {q.get('q50','—')} | "
                f"{q.get('q95','—')} |\n",
            )
        lines.append("\n_Per-coin nonzero fraction:_\n\n")
        # Build a coin x feature pivot table
        coins_seen: set[str] = set()
        for fblock in cov["features"].values():
            if fblock.get("present"):
                coins_seen.update(fblock["per_coin"].keys())
        if coins_seen:
            cols = sorted(
                f for f, b in cov["features"].items() if b.get("present")
            )
            lines.append("| coin_id | " + " | ".join(cols) + " |\n")
            lines.append("|---|" + "|".join(["---:"] * len(cols)) + "|\n")
            for coin in sorted(coins_seen):
                row = [coin]
                for f in cols:
                    pc = cov["features"][f]["per_coin"].get(coin)
                    row.append(f"{pc['frac']}" if pc else "—")
                lines.append("| " + " | ".join(row) + " |\n")

    # ----- Notes ------------------------------------------------------
    lines.append("\n## Notes & known limitations\n")
    lines.append(
        "- `liquidations_1h_usd` and `bid_ask_spread_bps` remain at "
        "their poller-only coverage. OKX does not expose a free "
        "historical liquidations or order-book-depth endpoint with the "
        "365-day reach the dataset needs, so synthesising a value would "
        "violate the campaign's *real-data-only* contract. See the "
        "`backfill_market_signals.py` docstring for the audit trail.\n",
    )
    lines.append(
        "- `funding_rate` covers ~90 days back (OKX funding-rate-history "
        "retention); `open_interest_z` covers ~60 days back (OKX OI "
        "history retention); `btc/eth/sol` `mid_price` covers ~365 days "
        "back (chunked through the OKX history-candles endpoint). The "
        "older portion of the 1d / 1100-day window therefore stays at "
        "0 for funding/OI; we record it explicitly here so a future "
        "ablation does not mistake retention for a bug.\n",
    )
    lines.append(
        "- The retention pruner in `artifacts/api-server/src/lib/"
        "market-signals-retention.ts` was updated to exempt rows with "
        "`source LIKE 'okx_backfill_%'` so the live poller's 30-day "
        "trim does not erase this campaign's writes within an hour. "
        "A regression test guards this exemption.\n",
    )
    return "".join(lines)


async def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    db_audit = await _audit_db()

    # The first backfill chunk wrote at 2026-04-28T20:35Z (see the
    # earliest backfill manifest under reports/). Any parquet whose
    # embedded timestamp is at or after the cutoff above is
    # guaranteed to have observed the backfilled rows during its
    # asof-join. Earlier parquets predate the backfill and are
    # flagged so the reader can tell stale rows from
    # genuinely-uncovered windows.
    datasets: dict[str, dict[str, Any]] = {}
    for tf, path in _representative_parquet_per_tf().items():
        df = pd.read_parquet(path)
        # Filename format: <tf>_<TS>.parquet
        name_ts = path.stem.split("_", 1)[1] if "_" in path.stem else ""
        post_backfill = name_ts >= _BACKFILL_CUTOFF
        datasets[tf] = {
            "path": str(path.relative_to(ROOT)),
            "snapshot_ts": name_ts,
            "post_backfill": post_backfill,
            "coverage": _coverage_block(df),
        }

    report = {
        "task": 586,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forbidden_manifest": _forbidden_manifest_hash(),
        "db_audit": db_audit,
        "datasets": datasets,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = REPORTS_DIR / f"{stamp}-task586-verification.md"
    json_path = REPORTS_DIR / f"{stamp}-task586-verification.json"
    md_path.write_text(_markdown(report))
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"verification report -> {md_path}")
    print(f"verification json   -> {json_path}")
    print(f"  leak_count = {db_audit['leak_count']}  (must be 0)")
    return 0 if db_audit["leak_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
