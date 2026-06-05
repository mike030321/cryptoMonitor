"""Task #366 — Full 1-Year Quant Training Run orchestrator.

Drives the multi-phase campaign described in `.local/tasks/task-366.md`:

  Phase 1 — preflight audit (test suite + quant-only invariants)
  Phase 2 — data-integrity report + 5m hard skip gate
  Phase 3 — pre-training baseline snapshot (archive of models/)
  Phase 4 — full training campaign (per-coin / pooled / specialist heads)
  Phase 5 — periodic structured proof updates -> models/progress_updates.jsonl
  Phase 6 — diagnostic checks the operator named
  Phase 7 — final summary -> models/training_run_<timestamp>/summary.md

Designed to be invoked once and run to completion. Phases write their
outputs to the run folder as they complete so the operator never loses
state if the long Phase-4 step is interrupted.

Run:
    python -m scripts.run_full_training_campaign
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.contiguity import (  # noqa: E402
    CONTIGUITY_TOLERANCE_SECONDS,
    compute_longest_contiguous_run,
)
from app.db import close_pool, init_pool  # noqa: E402
from app.training.train import (  # noqa: E402
    DEFAULT_COINS,
    LOOKBACK_DAYS,
    lookback_days_for,
    run_training,
)

logger = logging.getLogger("ml-engine.campaign")

REGISTRY_ROOT = ROOT / "models"
PROGRESS_PATH = REGISTRY_ROOT / "progress_updates.jsonl"
ARCHIVE_ROOT = REGISTRY_ROOT / "_archive"
TRADEABLE_TIMEFRAMES = ["5m", "1h", "2h", "6h", "1d"]
COVERAGE_BAR_DAYS = {
    "1h": 350, "2h": 350, "6h": 350,
    "5m": 305,  # HARD GATE per task #366
    # Task #417 — 1d floor raised from 350 → 1000 contiguous days. The
    # 1d slice's holdout has to grow from ~265 rows to ~720+ rows for
    # the noise band σ on the directional-accuracy gate to shrink from
    # ~0.031 to ~0.019. The Phase-2 audit now refuses to start training
    # until 1d coverage clears the 1000-row floor for every coin.
    "1d": 1000,
}

# Task #603 — coins with KNOWN partial 5m history that the Coinbase +
# OKX combined fetch cannot lift past `COVERAGE_BAR_DAYS["5m"]`. JUP
# (`jupiter-exchange-solana`) is NOT listed on Coinbase Exchange and the
# OKX 5m history only goes back to 2025-02-25, so the truthful ceiling
# is ~66 contiguous days as of 2026-04. The 5m gate exempts these coins
# from the contiguous_days check but STILL enforces the density / gap-
# rate / synthetic-rows clauses so a quiet data quality regression is
# still caught. The exemption is per-coin and explicit on purpose — we
# never want a coin to silently slip below 305d without an operator
# noticing the addition to this set.
KNOWN_5M_PARTIAL_COINS: frozenset[str] = frozenset({
    "jupiter-exchange-solana",
})
EXPECTED_BARS_PER_DAY = {
    "1m": 1440, "5m": 288, "1h": 24, "2h": 12, "6h": 4, "1d": 1,
}

# Task #521 — booster-fix watchlist. The Task #507 booster fix
# (TINY_SLICE_THRESHOLD=1500, alpha=2.0) shifted predicted STABLE share
# up substantially on these 4 healthy slices that did not need rescuing,
# cutting their directional-call share by 15-32pp and roughly halving
# their realized trade count. We have no clean pre-fix PnL on disk
# (see reports/20260428T111719Z-task516-pnl-impact-verification.md §1)
# so we cannot retroactively prove a regression — instead, every campaign
# run flags these slices' current STABLE share, n_trades, and post-fee
# net_pct_total so the operator can spot a material regression as soon
# as Task #516 follow-up #2 (per-slice PnL snapshot) lands. The
# `pre_fix_directional_call_share_pct_drop` column records the original
# pre-fix → post-fix DCS shift in pp from
# `reports/20260428T111719Z-task516-pnl-impact-verification.md` so the
# watchlist row is self-explanatory without leaving summary.md.
TASK521_BOOSTER_FIX_WATCHLIST: tuple[dict[str, Any], ...] = (
    {"coin": "pepe",                    "tf": "6h", "pre_fix_dcs_drop_pp": 32},
    {"coin": "jupiter-exchange-solana", "tf": "1d", "pre_fix_dcs_drop_pp": 22},
    {"coin": "floki-inu",               "tf": "1d", "pre_fix_dcs_drop_pp": 20},
    {"coin": "dogwifcoin",              "tf": "6h", "pre_fix_dcs_drop_pp": 15},
)


# ── Utilities ─────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_progress(record: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"emitted_at": _utcnow_iso(), **record}
    with PROGRESS_PATH.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    print(f"[progress] {record.get('phase','?')} :: {record.get('status','?')} :: {record.get('headline','')}", flush=True)


# ── Phase 1 — preflight ───────────────────────────────────────────────────
PREFLIGHT_TESTS = [
    "tests/test_quantonly_enforcement.py",
    "tests/test_real_data_contract.py",
    "tests/test_cadence_correctness.py",
    "tests/test_per_coin_retrain_isolation.py",
    "tests/test_contract_fail_fast.py",
]


def phase1_preflight(run_dir: Path) -> dict:
    started = time.time()
    cmd = ["../../.pythonlibs/bin/pytest", *PREFLIGHT_TESTS, "-q"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    passed = proc.returncode == 0
    summary = {
        "phase": "preflight",
        "status": "ok" if passed else "fail",
        "tests": PREFLIGHT_TESTS,
        "elapsed_sec": round(time.time() - started, 1),
        "tail": "\n".join(out.strip().splitlines()[-25:]),
    }
    (run_dir / "phase1_preflight.json").write_text(json.dumps(summary, indent=2))
    # Quant-only invariant — assert no FORBIDDEN_FEATURE_PREFIXES leak
    # into the registry's active feature columns.
    from app.training.registry import (
        FEATURE_COLUMNS, FORBIDDEN_FEATURE_PREFIXES, load_model,
    )
    leaks = [c for c in FEATURE_COLUMNS
             if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)]
    summary["forbidden_feature_leaks"] = leaks
    if leaks:
        summary["status"] = "fail"
    # Friction contract loads cleanly?
    try:
        from app.backtest.contract import get_frictions
        fr = get_frictions()
        summary["round_trip_cost_pct"] = float(fr.round_trip_cost_pct)
        summary["frictions_ok"] = True
    except Exception as exc:  # noqa: BLE001
        summary["frictions_ok"] = False
        summary["frictions_error"] = str(exc)
        summary["status"] = "fail"
    # Archived-model loadability gate (round-2 review fix). Walk every
    # already-trained on-disk model. Any manifest carrying a forbidden
    # feature prefix, or that load_model() refuses to load while the
    # manifest claims model_kind != "prior", is a hard preflight fail —
    # we will NOT start a new campaign on top of poisoned baselines.
    archived_audit: dict = {
        "manifests_scanned": 0,
        "forbidden_feature_leaks": [],
        "load_failures": [],
    }
    skip_dirs = {"_archive", "datasets", "training_history"}
    for manifest_path in REGISTRY_ROOT.glob("*/*/v*/manifest.json"):
        # Path shape: models/<coin>/<tf>/<version>/manifest.json
        parts = manifest_path.relative_to(REGISTRY_ROOT).parts
        if len(parts) < 4 or parts[0] in skip_dirs:
            continue
        coin, tf, version = parts[0], parts[1], parts[2]
        archived_audit["manifests_scanned"] += 1
        try:
            payload = json.loads(manifest_path.read_text())
        except Exception as exc:  # noqa: BLE001
            archived_audit["load_failures"].append({
                "coin": coin, "tf": tf, "version": version,
                "reason": f"manifest_unreadable:{exc}",
            })
            continue
        feats = payload.get("feature_names") or []
        bad = [f for f in feats
               if isinstance(f, str)
               and any(f.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)]
        if bad:
            archived_audit["forbidden_feature_leaks"].append({
                "coin": coin, "tf": tf, "version": version,
                "manifest": str(manifest_path.relative_to(ROOT)),
                "forbidden_features": sorted(bad),
            })
            continue
        # load_model() applies the same forbidden-prefix gate AND
        # actually deserializes the booster from disk. If the manifest
        # is clean but load_model returns None for a non-prior model,
        # the baseline is unusable and we must abort.
        try:
            loaded = load_model(coin, tf, version=version)
        except Exception as exc:  # noqa: BLE001
            archived_audit["load_failures"].append({
                "coin": coin, "tf": tf, "version": version,
                "reason": f"load_model_raised:{type(exc).__name__}:{exc}",
            })
            continue
        if loaded is None and payload.get("model_kind") != "prior":
            archived_audit["load_failures"].append({
                "coin": coin, "tf": tf, "version": version,
                "reason": "load_model_returned_none_for_non_prior",
            })
    summary["archived_model_audit"] = archived_audit
    if archived_audit["forbidden_feature_leaks"] or archived_audit["load_failures"]:
        summary["status"] = "fail"
    (run_dir / "phase1_preflight.json").write_text(json.dumps(summary, indent=2))
    _append_progress({
        "phase": "preflight",
        "status": summary["status"],
        "headline": (
            f"{len(PREFLIGHT_TESTS)} test files, forbidden_leaks="
            f"{len(leaks)}, frictions_ok={summary['frictions_ok']}, "
            f"archived_manifests_scanned={archived_audit['manifests_scanned']}, "
            f"archived_forbidden_leaks={len(archived_audit['forbidden_feature_leaks'])}, "
            f"archived_load_failures={len(archived_audit['load_failures'])}"
        ),
    })
    if summary["status"] != "ok":
        raise RuntimeError(f"preflight failed: {summary}")
    return summary


# ── Phase 2 — data-integrity audit ────────────────────────────────────────
async def _query(sql: str, *args) -> list:
    pool = await init_pool()
    return list(await pool.fetch(sql, *args))


TF_SECONDS = {"5m": 300, "1h": 3600, "2h": 7200, "6h": 21600, "1d": 86400}


def _extract_snapshot_from_report(c_rep: dict) -> dict:
    """Extract the per-slice fields we diff against post-run. Must match
    the post-#365 report schema: `pnl_after_fees.net_pct_total` is the
    canonical PnL metric (previously `backtest.final_pnl_usd`).
    """
    return {
        "status": c_rep.get("status"),
        "metrics": c_rep.get("metrics"),
        "baseline_metrics": c_rep.get("baseline_metrics"),
        "lift_auc": c_rep.get("lift_auc"),
        "calibration": c_rep.get("calibration"),
        "pnl_after_fees": c_rep.get("pnl_after_fees"),
        "n_rows": c_rep.get("n_rows"),
        "version": c_rep.get("version"),
    }


def _harvest_prior_campaign_slice_pnl(
    progress_path: Path = PROGRESS_PATH,
) -> dict:
    """Walk `progress_updates.jsonl` and harvest a per-slice PnL row for
    every `slice_done` event in the prior full campaign. Used by the
    pre-run baseline archiver so the next "did PnL regress?" check is a
    one-line diff instead of a multi-hour reconstruction (task #520).

    The "prior campaign" is the most recent fully-bracketed
    `campaign_start` … `campaign_done` window strictly before the
    current `campaign_start` (which the orchestrator emits before this
    archiver runs). If no completed prior campaign exists (bootstrap
    case), we fall back to the second-most-recent `campaign_start` and
    collect everything up to the next campaign event.

    Returned shape:
        {
          "prior_run_dir": "models/training_run_<TS>" | None,
          "prior_campaign_started_at": iso8601 | None,
          "prior_campaign_finished_at": iso8601 | None,
          "completed": bool,
          "slice_count": int,
          "per_slice": {
            "<coin>/<tf>": {
              "coin": str, "timeframe": str, "status": str,
              "n_trades": int|None, "net_pct_total": float|None,
              "net_pct_mean": float|None, "gross_pct_mean": float|None,
              "win_rate": float|None, "trade_share": float|None,
              "round_trip_cost_pct": float|None,
              "auc": float|None, "baseline_auc": float|None,
              "lift_auc": float|None, "directional_accuracy": float|None,
              "n_rows": int|None, "emitted_at": iso8601,
            },
            ...
          },
        }

    If the same `(coin, timeframe)` slice fires `slice_done` twice in
    one campaign window (e.g. retried on a transient error) the LAST
    event wins so the row matches the model that was actually written.
    """
    out: dict[str, Any] = {
        "prior_run_dir": None,
        "prior_campaign_started_at": None,
        "prior_campaign_finished_at": None,
        "completed": False,
        "slice_count": 0,
        "per_slice": {},
    }
    if not progress_path.exists():
        return out

    events: list[dict] = []
    try:
        with progress_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:  # noqa: BLE001 — best-effort
                    continue
    except Exception:  # noqa: BLE001
        return out

    # Indices of every `campaign_start` and `campaign_done` event.
    starts = [i for i, e in enumerate(events) if e.get("phase") == "campaign_start"]
    dones = [i for i, e in enumerate(events) if e.get("phase") == "campaign_done"]
    if not starts:
        return out

    # The most recent `campaign_start` is normally the *current* one
    # (the orchestrator emits it before phase3). The prior campaign's
    # window is everything before that index.
    cur_start_idx = starts[-1]

    # Prefer a fully-bracketed prior campaign: latest `campaign_done`
    # with index < cur_start_idx, then walk back to its matching
    # `campaign_start`.
    completed_dones = [i for i in dones if i < cur_start_idx]
    prior_start_idx: Optional[int] = None
    prior_done_idx: Optional[int] = None
    completed = False
    if completed_dones:
        prior_done_idx = completed_dones[-1]
        # Matching start = greatest start index < prior_done_idx.
        prior_starts = [i for i in starts if i < prior_done_idx]
        if prior_starts:
            prior_start_idx = prior_starts[-1]
            completed = True
    if prior_start_idx is None:
        # Bootstrap / interrupted prior run: take the second-most-recent
        # `campaign_start` and collect events up to the current start.
        if len(starts) >= 2:
            prior_start_idx = starts[-2]
            prior_done_idx = cur_start_idx  # exclusive upper bound
            completed = False
        else:
            return out

    start_evt = events[prior_start_idx]
    done_evt = events[prior_done_idx] if (
        completed and prior_done_idx is not None
    ) else None
    out["prior_run_dir"] = start_evt.get("run_dir")
    out["prior_campaign_started_at"] = start_evt.get("emitted_at")
    out["prior_campaign_finished_at"] = (
        done_evt.get("emitted_at") if done_evt else None
    )
    out["completed"] = completed

    upper = prior_done_idx if prior_done_idx is not None else cur_start_idx
    per_slice: dict[str, dict] = {}
    for evt in events[prior_start_idx + 1 : upper + 1]:
        if evt.get("phase") != "slice_done":
            continue
        coin = evt.get("coin")
        tf = evt.get("timeframe")
        if not coin or not tf:
            continue
        pnl = evt.get("pnl_after_fees") or {}
        metrics = evt.get("metrics") or {}
        base_metrics = evt.get("baseline_metrics") or {}
        per_slice[f"{coin}/{tf}"] = {
            "coin": coin,
            "timeframe": tf,
            "status": evt.get("status"),
            "n_trades": pnl.get("n_trades"),
            "net_pct_total": pnl.get("net_pct_total"),
            "net_pct_mean": pnl.get("net_pct_mean"),
            "gross_pct_mean": pnl.get("gross_pct_mean"),
            "win_rate": pnl.get("win_rate"),
            "trade_share": pnl.get("trade_share"),
            "round_trip_cost_pct": pnl.get("round_trip_cost_pct"),
            "auc": metrics.get("auc"),
            "baseline_auc": base_metrics.get("auc"),
            "lift_auc": evt.get("lift_auc"),
            "directional_accuracy": metrics.get("directional_accuracy"),
            "n_rows": evt.get("n_rows"),
            "emitted_at": evt.get("emitted_at"),
        }
    out["per_slice"] = per_slice
    out["slice_count"] = len(per_slice)
    return out


async def _coverage_per_slice() -> list[dict]:
    """For every monitored coin × tradeable timeframe, measure:
       row count, earliest/latest bucket, TRUE contiguous-day window
       (longest run of buckets with no gap), actual missing-bucket gap
       list, source distribution, duplicate count, and synthetic mix.
    """
    out: list[dict] = []
    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)
    for coin in DEFAULT_COINS:
        for tf in TRADEABLE_TIMEFRAMES + ["1m"]:
            if tf == "1m":
                tick_rows = await _query(
                    """
                    SELECT COUNT(*) AS n_real,
                           SUM(CASE WHEN COALESCE(is_synthetic,false) THEN 1 ELSE 0 END) AS n_synth,
                           MIN(timestamp) AS earliest, MAX(timestamp) AS latest
                    FROM price_history
                    WHERE coin_id=$1 AND timestamp >= $2
                    """, coin, one_year_ago,
                )
                n = int(tick_rows[0]["n_real"]) if tick_rows else 0
                synth = int(tick_rows[0]["n_synth"] or 0) if tick_rows else 0
                earliest = tick_rows[0]["earliest"] if tick_rows else None
                latest = tick_rows[0]["latest"] if tick_rows else None
                days = (latest - earliest).total_seconds() / 86400.0 if (earliest and latest) else 0.0
                out.append({
                    "coin": coin, "timeframe": tf, "rows": n,
                    "earliest": earliest.isoformat() if earliest else None,
                    "latest": latest.isoformat() if latest else None,
                    "span_days": round(days, 2),
                    "contiguous_days": round(days, 2),  # 1m not gate-evaluated
                    "contiguous_days_strict": round(days, 2),
                    "contiguity_tolerance_seconds": 0,
                    "density": None, "gap_rate": None,
                    "duplicate_buckets": 0, "synthetic_rows": synth,
                    "source_distribution": {}, "gap_count": 0,
                    "largest_gap_buckets": 0, "n_gaps_over_2h": 0,
                    "deepest_source": None,
                })
                continue

            # price_candles: pull bucket_starts + source in one pass so
            # we can compute a real contiguous-day run and a real gap list
            # without round-tripping the DB per bucket.
            rows = await _query(
                """
                SELECT bucket_start, source
                FROM price_candles
                WHERE coin_id=$1 AND timeframe=$2 AND bucket_start >= $3
                ORDER BY bucket_start, source
                """, coin, tf, one_year_ago,
            )
            n = len(rows)
            earliest = rows[0]["bucket_start"] if rows else None
            latest = rows[-1]["bucket_start"] if rows else None
            # Source distribution — proves the candles came from OKX
            # (not a synthetic or internal fallback source).
            src_dist: dict[str, int] = {}
            for r in rows:
                s = r["source"] or "unknown"
                src_dist[s] = src_dist.get(s, 0) + 1
            # STRICT real-data gate: any candle whose source is not OKX
            # is treated as synthetic-or-non-real for the gate. The whole
            # pipeline assumes real OKX bars; if anything else slips into
            # `price_candles` the gate must catch it instead of silently
            # training on it.
            # Real-source allow-list. OKX is the primary venue; Coinbase
            # was added in task #409 as the secondary 5m source for coins
            # whose OKX `history-candles` window truncates short of the
            # 305-day hard gate. SEI was the original target coin; even
            # with Coinbase it could not clear the strict-contiguous
            # gate (see reports/task-409-sei-5m-rationale.md) and was
            # removed from DEFAULT_COINS, but the Coinbase pre-pass
            # below stays in place for any future short-OKX-history coin.
            # Any source outside this set is counted as non-real for the gate.
            real_source_aliases = {
                "okx", "okx-history", "okx_history",
                "okx:5m", "okx:1h", "okx:2h", "okx:6h", "okx:1d",
                "coinbase",
            }
            non_real_rows = sum(c for s, c in src_dist.items() if s not in real_source_aliases)
            # Duplicate detection
            seen: dict[Any, int] = {}
            for r in rows:
                seen[r["bucket_start"]] = seen.get(r["bucket_start"], 0) + 1
            dups = sum(c - 1 for c in seen.values() if c > 1)
            # True gap list: walk consecutive buckets; count how many
            # expected buckets are missing between each pair.
            tf_sec = TF_SECONDS[tf]
            gaps: list[int] = []
            prev = None
            for r in rows:
                bs = r["bucket_start"]
                if prev is not None:
                    delta_s = (bs - prev).total_seconds()
                    missing = int(round(delta_s / tf_sec)) - 1
                    if missing > 0:
                        gaps.append(missing)
                prev = bs
            gap_count = len(gaps)
            largest_gap = max(gaps) if gaps else 0
            # Gaps that cross a 2-hour real-time threshold — the task's
            # "material gap" definition. A 5m slice with 24 missing
            # buckets == 2h of missing data, a 1h slice with 2 missing
            # buckets == 2h, etc.
            two_hour_threshold = max(1, int(7200 / tf_sec))
            n_material_gaps = sum(1 for g in gaps if g >= two_hour_threshold)
            # Task #604 — gap-tolerant longest contiguous run. The
            # tolerance comes from `app.contiguity` (5m=7h, others=0)
            # so the strict semantic is preserved for OKX-served TFs
            # and only the Coinbase-served 5m slice gets to absorb
            # the venue's low-volume sparsity. The strict counterpart
            # is also returned so the audit trail can show both.
            tolerance_sec = CONTIGUITY_TOLERANCE_SECONDS.get(tf, 0)
            contig_days, contig_days_strict = compute_longest_contiguous_run(
                [r["bucket_start"] for r in rows], tf_sec, tolerance_sec,
            )
            span_days = (latest - earliest).total_seconds() / 86400.0 if (earliest and latest) else 0.0
            expected = int(EXPECTED_BARS_PER_DAY[tf] * max(span_days, 1))
            density = (n / expected) if expected > 0 else 0.0
            gap_rate = max(0.0, 1.0 - density)
            # The source attached to the EARLIEST bucket — i.e. the
            # venue/source that reaches deepest into history. Used by
            # the markdown summary's source-depth shortfall section so
            # an operator can immediately see which feed needs to be
            # extended (or replaced) when `span_days` < the gate bar.
            deepest_source = (rows[0]["source"] or "unknown") if rows else None
            out.append({
                "coin": coin, "timeframe": tf,
                "rows": n,
                "earliest": earliest.isoformat() if earliest else None,
                "latest": latest.isoformat() if latest else None,
                "span_days": round(span_days, 2),
                "contiguous_days": round(contig_days, 2),
                "contiguous_days_strict": round(contig_days_strict, 2),
                "contiguity_tolerance_seconds": tolerance_sec,
                "expected_bars": expected,
                "density": round(density, 4),
                "gap_rate": round(gap_rate, 4),
                "duplicate_buckets": dups,
                "synthetic_rows": non_real_rows,
                "source_distribution": src_dist,
                "gap_count": gap_count,
                "largest_gap_buckets": largest_gap,
                "n_gaps_over_2h": n_material_gaps,
                "deepest_source": deepest_source,
            })
    await close_pool()
    return out


def _evaluate_5m_gate(slices: list[dict]) -> dict[str, dict]:
    """Per-coin 5m verdict: pass iff
        contiguous_days >= 305 AND density >= 0.80 AND gap_rate <= 0.01
        AND synthetic_rows == 0.

    Coins listed in `KNOWN_5M_PARTIAL_COINS` are EXEMPT from the
    contiguous_days clause (they have no Coinbase product and OKX's 5m
    history is too short to clear 305d), but the density / gap_rate /
    synthetic_rows clauses still apply so a quiet data-quality
    regression is still caught. The verdict carries
    `partial_history_exempt=True` for these coins so the audit summary
    can clearly distinguish "passed because waived" from "passed because
    we cleared the bar".

    Task #604 — `contiguous_days` is now the GAP-TOLERANT measure
    produced by `app.contiguity.compute_longest_contiguous_run` (5m
    tolerance = 7h / 84 missing buckets). The strict pre-tolerance
    counterpart is in `s["contiguous_days_strict"]` and is included
    in the reason string so the audit trail shows BOTH numbers without
    requiring an operator to re-run anything. The tolerance only
    loosens the longest-run measure — `gap_rate <= 0.01` and
    `density >= 0.80` are unchanged, so a coin with genuinely sparse
    data (e.g. floki at 75% density) still fails the gate.
    """
    bar = COVERAGE_BAR_DAYS["5m"]
    verdicts: dict[str, dict] = {}
    for s in slices:
        if s["timeframe"] != "5m":
            continue
        exempt = s["coin"] in KNOWN_5M_PARTIAL_COINS
        coverage_ok = exempt or s["contiguous_days"] >= bar
        passed = (
            coverage_ok
            and s["density"] >= 0.80
            and s["gap_rate"] <= 0.01
            and s["synthetic_rows"] == 0
        )
        # Tolerance + strict-counterpart fragment so every reason
        # surfaces BOTH numbers — operators can grep an audit summary
        # and immediately see "tolerated 320 / strict 128 / 7h tol"
        # without re-running the gate.
        tol_sec = s.get("contiguity_tolerance_seconds", 0) or 0
        strict_d = s.get("contiguous_days_strict", s["contiguous_days"])
        tol_frag = (
            f"strict={strict_d:.0f}, tolerance={tol_sec // 60}m"
            if tol_sec > 0 else f"strict={strict_d:.0f}"
        )
        if passed and exempt:
            reason = (
                f"ok (partial_history_exempt; days={s['contiguous_days']:.0f}"
                f" [{tol_frag}] — see KNOWN_5M_PARTIAL_COINS)"
            )
        elif passed:
            reason = f"ok (days={s['contiguous_days']:.0f} [{tol_frag}])"
        else:
            # Truth-only failure reason: enumerate ONLY the clauses
            # that actually fail, joined with " AND ". An "or"-template
            # that lists every clause regardless of its real value
            # misleads the operator about which threshold drove the
            # rejection (Task #604 architect review).
            fail_clauses: list[str] = []
            if not exempt and s["contiguous_days"] < bar:
                fail_clauses.append(
                    f"days={s['contiguous_days']:.0f} [{tol_frag}] <{bar}"
                )
            if s["density"] < 0.80:
                fail_clauses.append(f"density={s['density']:.2f}<0.80")
            if s["gap_rate"] > 0.01:
                fail_clauses.append(f"gap_rate={s['gap_rate']:.3f}>0.01")
            if s["synthetic_rows"] != 0:
                fail_clauses.append(f"synth={s['synthetic_rows']}")
            # Always show both day numbers in the parenthetical even
            # when only density/gap_rate/synth fail, so audits never
            # have to re-derive the run length.
            if not any(c.startswith("days=") for c in fail_clauses):
                fail_clauses.append(
                    f"days={s['contiguous_days']:.0f} [{tol_frag}]"
                )
            prefix = "coverage_partial_exempt" if exempt else "coverage_below_bar"
            reason = f"{prefix} ({' AND '.join(fail_clauses)})"
        verdicts[s["coin"]] = {
            "passed": passed,
            "reason": reason,
            "partial_history_exempt": exempt,
            **s,
        }
    return verdicts


def _evaluate_higher_tf_gate(slices: list[dict]) -> dict[str, dict]:
    """Per (coin, tf in {1h,2h,6h,1d}) verdict: pass iff
        contiguous_days >= COVERAGE_BAR_DAYS[tf]
        AND gap_rate <= 0.01 AND synthetic_rows == 0.

    The 1d bar is 1000 days (task #417) so the trainer's 1d holdout
    grows from ~265 rows to ~720+ rows; 1h/2h/6h keep the 350-day bar.
    """
    out: dict[str, dict] = {}
    for s in slices:
        if s["timeframe"] not in ("1h", "2h", "6h", "1d"):
            continue
        bar = COVERAGE_BAR_DAYS[s["timeframe"]]
        passed = (
            s["contiguous_days"] >= bar
            and s["gap_rate"] <= 0.01
            and s["synthetic_rows"] == 0
        )
        # Higher TFs use OKX exclusively and the tolerance dict has no
        # entry for them, so `contiguity_tolerance_seconds == 0` and the
        # tolerated value equals the strict value. We still surface the
        # strict counterpart so audit lines have the same shape as the
        # 5m gate's reasons.
        tol_sec = s.get("contiguity_tolerance_seconds", 0) or 0
        strict_d = s.get("contiguous_days_strict", s["contiguous_days"])
        tol_frag = (
            f"strict={strict_d:.0f}, tolerance={tol_sec // 60}m"
            if tol_sec > 0 else f"strict={strict_d:.0f}"
        )
        if passed:
            reason = f"ok (days={s['contiguous_days']:.0f} [{tol_frag}])"
        else:
            # Truth-only: enumerate ONLY the failed clauses.
            fail_clauses: list[str] = []
            if s["contiguous_days"] < bar:
                fail_clauses.append(
                    f"days={s['contiguous_days']:.0f} [{tol_frag}] <{bar}"
                )
            if s["gap_rate"] > 0.01:
                fail_clauses.append(f"gap_rate={s['gap_rate']:.3f}>0.01")
            if s["synthetic_rows"] != 0:
                fail_clauses.append(f"synth={s['synthetic_rows']}")
            if not any(c.startswith("days=") for c in fail_clauses):
                fail_clauses.append(
                    f"days={s['contiguous_days']:.0f} [{tol_frag}]"
                )
            reason = " AND ".join(fail_clauses)
        out[f"{s['coin']}/{s['timeframe']}"] = {
            "passed": passed,
            "reason": reason,
            **s,
        }
    return out


def phase2_data_audit(run_dir: Path) -> dict:
    started = time.time()
    slices = asyncio.run(_coverage_per_slice())
    pre_5m_gate = _evaluate_5m_gate(slices)
    pre_high_gate = _evaluate_higher_tf_gate(slices)

    skipped_5m = sorted([c for c, v in pre_5m_gate.items() if not v["passed"]])
    passed_5m = sorted([c for c, v in pre_5m_gate.items() if v["passed"]])

    pre_report = {
        "phase": "data_audit_pre_backfill",
        "generated_at": _utcnow_iso(),
        "slices": slices,
        "five_m_gate": pre_5m_gate,
        "higher_tf_gate": pre_high_gate,
        "skipped_5m_coins": skipped_5m,
        "passed_5m_coins": passed_5m,
    }
    (run_dir / "phase2_data_audit_pre_backfill.json").write_text(
        json.dumps(pre_report, indent=2, default=str)
    )

    # ── Task #409 — Coinbase 5m fallback pre-pass. For coins listed in
    # `scripts.backfill_history.COINBASE_PRODUCTS` whose 5m gate is
    # currently failing, run a one-shot Coinbase backfill before the
    # OKX iterative loop. Coinbase's `/products/<id>/candles` endpoint
    # serves SEI 5m back to early 2024 with no key required (unlike
    # OKX which truncates SEI 5m at ~161 days). Rows land with
    # `source="coinbase"`, which is whitelisted by `real_source_aliases`
    # above.
    coinbase_log: list[dict] = []
    if skipped_5m and os.environ.get("ML_SKIP_5M_BACKFILL") != "1":
        try:
            from scripts.backfill_history import COINBASE_PRODUCTS
        except Exception:  # noqa: BLE001
            COINBASE_PRODUCTS = {}
        coinbase_eligible = [c for c in skipped_5m if c in COINBASE_PRODUCTS]
        if coinbase_eligible:
            cb_timeout = int(os.environ.get("ML_COINBASE_BACKFILL_TIMEOUT_SEC", "900"))
            cb_days = int(os.environ.get("ML_COINBASE_5M_DAYS", "320"))
            for coin in coinbase_eligible:
                t0 = time.time()
                # Earliest bucket BEFORE Coinbase pull — proves whether
                # the venue actually pushed the window earlier.
                earliest_before_q = asyncio.run(_query(
                    "SELECT MIN(bucket_start) AS e FROM price_candles "
                    "WHERE coin_id=$1 AND timeframe='5m'", coin,
                ))
                earliest_before = (
                    earliest_before_q[0]["e"] if earliest_before_q else None
                )
                cmd = [
                    "../../.pythonlibs/bin/python",
                    "-m", "scripts.backfill_history",
                    "--coins", coin,
                    "--timeframes", "5m",
                    "--target", "candles",
                    "--source", "coinbase",
                    "--days", str(cb_days),
                ]
                try:
                    proc = subprocess.run(
                        cmd, cwd=ROOT, capture_output=True, text=True,
                        timeout=cb_timeout,
                    )
                    ok = proc.returncode == 0
                    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-8:]
                except subprocess.TimeoutExpired:
                    ok = False
                    tail = ["TIMEOUT — coinbase backfill exceeded ML_COINBASE_BACKFILL_TIMEOUT_SEC"]
                earliest_after_q = asyncio.run(_query(
                    "SELECT MIN(bucket_start) AS e FROM price_candles "
                    "WHERE coin_id=$1 AND timeframe='5m'", coin,
                ))
                earliest_after = (
                    earliest_after_q[0]["e"] if earliest_after_q else None
                )
                advanced = bool(
                    earliest_after is not None
                    and (earliest_before is None or earliest_after < earliest_before)
                )
                coinbase_log.append({
                    "coin": coin, "ok": ok, "days": cb_days,
                    "earliest_before": earliest_before.isoformat() if earliest_before else None,
                    "earliest_after": earliest_after.isoformat() if earliest_after else None,
                    "advanced": advanced,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "tail": "\n".join(tail),
                })
            # Re-evaluate the 5m gate so the OKX iterative loop below
            # only retries coins that Coinbase did NOT already lift past
            # the gate.
            slices_after_cb = asyncio.run(_coverage_per_slice())
            gate_after_cb = _evaluate_5m_gate(slices_after_cb)
            skipped_5m = sorted([
                c for c, v in gate_after_cb.items() if not v["passed"]
            ])

    # ── Iterative 5m extension loop — calls scripts.backfill_history
    # repeatedly with a progressively-older `--end-ts-ms` cursor so we
    # walk the OKX history past the venue's 200-page single-invocation
    # cap (~70 days of 5m per call). The loop exits when (a) the gate
    # clears for every still-skipped coin, (b) an iteration adds zero
    # new rows for every coin (venue exhausted = listing date reached or
    # OKX history truncated), or (c) the iteration cap fires. Per-iter
    # row counts are recorded so the operator can see exactly how far
    # back OKX would actually serve.
    backfill_log: list[dict] = []
    if skipped_5m and os.environ.get("ML_SKIP_5M_BACKFILL") != "1":
        max_iters = int(os.environ.get("ML_5M_BACKFILL_MAX_ITERS", "8"))
        per_iter_timeout = int(os.environ.get("ML_BACKFILL_TIMEOUT_SEC", "420"))
        # Window per iteration — OKX caps each invocation at 200 pages ×
        # 100 5m bars = 20 000 bars ≈ 69 days, so step ~65 days per iter
        # and overlap a bit to avoid skipped windows when the cursor
        # rounds awkwardly.
        step_ms = 65 * 24 * 60 * 60 * 1000
        # Coins still failing the gate this iteration. Re-evaluated each
        # loop so coins that pass mid-loop drop out immediately.
        still_skipped = list(skipped_5m)
        # Per-coin earliest bucket reached so far (drives the next end_ts).
        async def _earliest_5m(coin: str) -> Optional[datetime]:
            r = await _query(
                "SELECT MIN(bucket_start) AS e FROM price_candles "
                "WHERE coin_id=$1 AND timeframe='5m'", coin,
            )
            return r[0]["e"] if r and r[0]["e"] else None
        for iter_idx in range(max_iters):
            if not still_skipped:
                break
            # Compute next `end_ts_ms` per coin (now-step on iter 0; older
            # of (earliest_seen - 1ms) and (prev_end - step) thereafter).
            iter_log: dict = {"iter": iter_idx, "coins": list(still_skipped), "per_coin": {}}
            for coin in still_skipped:
                earliest = asyncio.run(_earliest_5m(coin))
                if iter_idx == 0 or earliest is None:
                    end_ts_ms = int(time.time() * 1000)
                else:
                    end_ts_ms = int(earliest.timestamp() * 1000) - 1
                t0 = time.time()
                cmd = [
                    "../../.pythonlibs/bin/python",
                    "-m", "scripts.backfill_history",
                    "--coins", coin,
                    "--timeframes", "5m",
                    "--target", "candles",
                    "--end-ts-ms", str(end_ts_ms),
                ]
                try:
                    proc = subprocess.run(
                        cmd, cwd=ROOT, capture_output=True, text=True,
                        timeout=per_iter_timeout,
                    )
                    ok = proc.returncode == 0
                    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-6:]
                except subprocess.TimeoutExpired:
                    ok = False
                    tail = ["TIMEOUT — backfill exceeded ML_BACKFILL_TIMEOUT_SEC"]
                # Pull the new earliest bucket — proves whether the
                # venue actually returned older rows in this iteration.
                new_earliest = asyncio.run(_earliest_5m(coin))
                advanced = bool(
                    new_earliest is not None
                    and (earliest is None or new_earliest < earliest)
                )
                iter_log["per_coin"][coin] = {
                    "ok": ok,
                    "end_ts_ms": end_ts_ms,
                    "earliest_before": earliest.isoformat() if earliest else None,
                    "earliest_after": new_earliest.isoformat() if new_earliest else None,
                    "advanced": advanced,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "tail": "\n".join(tail),
                }
            backfill_log.append(iter_log)
            # Prune coins whose 5m window did not advance — the venue
            # has nothing older to give. Re-check the gate to drop coins
            # that now pass.
            slices_chk = asyncio.run(_coverage_per_slice())
            gate_chk = _evaluate_5m_gate(slices_chk)
            still_skipped = [
                c for c in still_skipped
                if (not gate_chk.get(c, {}).get("passed", False))
                and iter_log["per_coin"][c]["advanced"]
            ]

    # ── Task #417 — 1d extension loop. Mirrors the 5m loop but is much
    # cheaper because OKX 1d candles paginate at 100 bars/page, so 1100
    # days fits in ~11 pages — well under the 200-page single-invocation
    # cap. Iteration here is mostly defensive: a single backfill call
    # should clear the 1000-day gate, but we re-audit and retry once if
    # the venue trickles rows in over multiple invocations. Skipped
    # entirely when ML_SKIP_1D_BACKFILL=1 (operator escape hatch for
    # smoke tests).
    one_d_backfill_log: list[dict] = []
    bar_1d = COVERAGE_BAR_DAYS["1d"]
    coins_needing_1d = [
        s["coin"] for s in slices
        if s["timeframe"] == "1d" and s["contiguous_days"] < bar_1d
    ]
    if coins_needing_1d and os.environ.get("ML_SKIP_1D_BACKFILL") != "1":
        max_iters_1d = int(os.environ.get("ML_1D_BACKFILL_MAX_ITERS", "3"))
        per_iter_timeout = int(os.environ.get("ML_BACKFILL_TIMEOUT_SEC", "420"))
        # Per-coin earliest 1d bucket — drives the next end_ts for the
        # iterative walk-back. Same cursor-seed pattern as the 5m loop.
        async def _earliest_1d(coin: str) -> Optional[datetime]:
            r = await _query(
                "SELECT MIN(bucket_start) AS e FROM price_candles "
                "WHERE coin_id=$1 AND timeframe='1d'", coin,
            )
            return r[0]["e"] if r and r[0]["e"] else None
        # Pull the 1d default once so we send `--days` aligned with the
        # backfill module's own default (which task #417 set to 1100).
        # Importing inside the function keeps the campaign module's
        # top-level imports light; backfill_history.py is heavy.
        from scripts.backfill_history import DEFAULT_DAYS_BY_TF as _BF_DEFAULTS
        days_arg_1d = str(_BF_DEFAULTS.get("1d", 1100))
        still_short = list(coins_needing_1d)
        for iter_idx in range(max_iters_1d):
            if not still_short:
                break
            iter_log: dict = {"iter": iter_idx, "coins": list(still_short), "per_coin": {}}
            for coin in still_short:
                earliest = asyncio.run(_earliest_1d(coin))
                if iter_idx == 0 or earliest is None:
                    end_ts_ms = int(time.time() * 1000)
                else:
                    # Walk strictly older than what we already have.
                    end_ts_ms = int(earliest.timestamp() * 1000) - 1
                t0 = time.time()
                cmd = [
                    "../../.pythonlibs/bin/python",
                    "-m", "scripts.backfill_history",
                    "--coins", coin,
                    "--timeframes", "1d",
                    "--target", "candles",
                    "--days", days_arg_1d,
                    "--end-ts-ms", str(end_ts_ms),
                ]
                try:
                    proc = subprocess.run(
                        cmd, cwd=ROOT, capture_output=True, text=True,
                        timeout=per_iter_timeout,
                    )
                    ok = proc.returncode == 0
                    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-6:]
                except subprocess.TimeoutExpired:
                    ok = False
                    tail = ["TIMEOUT — backfill exceeded ML_BACKFILL_TIMEOUT_SEC"]
                new_earliest = asyncio.run(_earliest_1d(coin))
                advanced = bool(
                    new_earliest is not None
                    and (earliest is None or new_earliest < earliest)
                )
                iter_log["per_coin"][coin] = {
                    "ok": ok,
                    "end_ts_ms": end_ts_ms,
                    "earliest_before": earliest.isoformat() if earliest else None,
                    "earliest_after": new_earliest.isoformat() if new_earliest else None,
                    "advanced": advanced,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "tail": "\n".join(tail),
                }
            one_d_backfill_log.append(iter_log)
            # Re-evaluate after this iter — coins that cleared the 1000d
            # floor drop out, coins whose 1d window did not advance also
            # drop out (venue has nothing older).
            slices_chk = asyncio.run(_coverage_per_slice())
            still_short = [
                c for c in still_short
                if any(
                    s["coin"] == c and s["timeframe"] == "1d"
                    and s["contiguous_days"] < bar_1d
                    for s in slices_chk
                )
                and iter_log["per_coin"][c]["advanced"]
            ]

    # Re-audit
    slices_post = asyncio.run(_coverage_per_slice())
    post_5m_gate = _evaluate_5m_gate(slices_post)
    post_high_gate = _evaluate_higher_tf_gate(slices_post)
    skipped_5m_post = sorted([c for c, v in post_5m_gate.items() if not v["passed"]])
    passed_5m_post = sorted([c for c, v in post_5m_gate.items() if v["passed"]])

    post_report = {
        "phase": "data_audit_post_backfill",
        "generated_at": _utcnow_iso(),
        "slices": slices_post,
        "five_m_gate": post_5m_gate,
        "higher_tf_gate": post_high_gate,
        "skipped_5m_coins": skipped_5m_post,
        "passed_5m_coins": passed_5m_post,
        "backfill_log": backfill_log,
        "coinbase_backfill_log": coinbase_log,
        "one_d_backfill_log": one_d_backfill_log,
        "coverage_bar_days": COVERAGE_BAR_DAYS,
    }
    (run_dir / "phase2_data_audit.json").write_text(
        json.dumps(post_report, indent=2, default=str)
    )

    # Markdown summary
    md = ["# Phase 2 — Data Integrity Report", ""]
    md.append(f"_generated {_utcnow_iso()}_")
    md.append("")
    md.append("## Coverage matrix (post-backfill)")
    md.append("")
    md.append(
        "| coin | tf | rows | span_d | contig_d | density | gap_rate | "
        "gaps | largest_gap | ≥2h_gaps | dups | synth | source_mix |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for s in slices_post:
        src = s.get("source_distribution") or {}
        src_txt = ", ".join(f"{k}={v}" for k, v in sorted(src.items(), key=lambda x: -x[1])) or "-"
        dens = s.get("density")
        gr = s.get("gap_rate")
        md.append(
            f"| {s['coin']} | {s['timeframe']} | {s['rows']} "
            f"| {s.get('span_days', 0):.1f} | {s['contiguous_days']:.1f} "
            f"| {f'{dens:.3f}' if dens is not None else '-'} "
            f"| {f'{gr:.3f}' if gr is not None else '-'} "
            f"| {s.get('gap_count', 0)} | {s.get('largest_gap_buckets', 0)} "
            f"| {s.get('n_gaps_over_2h', 0)} "
            f"| {s['duplicate_buckets']} | {s['synthetic_rows']} | {src_txt} |"
        )
    md.append("")
    md.append("## 5m HARD GATE (≥305 contiguous days, ≥80% density, ≤1% gap, 0 synth)")
    md.append("")
    md.append(f"- **Passed (will train 5m):** {passed_5m_post or 'none'}")
    md.append(f"- **Skipped (insufficient 5m history):** {skipped_5m_post}")
    md.append("")
    md.append("Skip detail (per coin):")
    md.append("")
    try:
        from scripts.backfill_history import COINBASE_PRODUCTS as _CB_PROD
    except Exception:  # noqa: BLE001
        _CB_PROD = {}
    for coin in skipped_5m_post:
        v = post_5m_gate[coin]
        if coin in _CB_PROD:
            cause = (
                "Both OKX (`history-candles` ~161-day truncation) and "
                "Coinbase (`/products/<id>/candles` fallback added in "
                "task #409) failed to reach the gate after the backfill "
                "passes above. Inspect the per-iter `coinbase_backfill_log` "
                "+ `backfill_log` blocks to see exactly how far each venue "
                "served before refusing more rows."
            )
        else:
            cause = (
                "OKX `history-candles` paginates 100 bars/request and the "
                "`scripts/backfill_history.py` worker caps at 200 pages "
                "(~70 days of 5m). Reaching 305 days needs either the OKX "
                "iterative loop or the Coinbase fallback (task #409 — add "
                "an entry to COINBASE_PRODUCTS)."
            )
        md.append(
            f"- `{coin}` — days={v['contiguous_days']:.0f}, density={v['density']:.2f}, "
            f"gap_rate={v['gap_rate']:.3f}, synth={v['synthetic_rows']}. "
            f"Venue cause: {cause}"
        )
    md.append("")
    md.append(
        "## 1h / 2h / 6h gate (≥350 contiguous days, ≤1% gap, 0 synth) "
        "and 1d gate (≥1000 contiguous days, ≤1% gap, 0 synth — task #417)"
    )
    md.append("")
    failed_high = [k for k, v in post_high_gate.items() if not v["passed"]]
    md.append(f"- **Failing slices:** {failed_high or 'none'}")
    if failed_high:
        for k in failed_high:
            v = post_high_gate[k]
            bar_for_tf = COVERAGE_BAR_DAYS.get(v["timeframe"], 350)
            md.append(
                f"  - `{k}` — days={v['contiguous_days']:.0f}/{bar_for_tf}, "
                f"gap_rate={v['gap_rate']:.3f}, synth={v['synthetic_rows']}"
            )
    md.append("")
    # ── Source-depth shortfall section (task #422) ────────────────────
    # Flag every (coin, tf) whose deepest-available data span is itself
    # shorter than the gate bar — i.e. even with zero gaps the source
    # CAN'T reach the gate. This is the operator's early-warning for
    # the next "SEI" (task #409) where OKX's `history-candles` window
    # truncates a coin below 305 days for 5m. Use a 5-day buffer so a
    # coin sitting right on the bar isn't flagged for normal clock drift.
    SHORTFALL_BUFFER_DAYS = 5
    shortfalls = []
    for s in slices_post:
        tf = s["timeframe"]
        if tf not in COVERAGE_BAR_DAYS:
            continue  # 1m is informational only — not gated.
        bar = COVERAGE_BAR_DAYS[tf]
        span = float(s.get("span_days") or 0.0)
        if span + SHORTFALL_BUFFER_DAYS < bar:
            shortfalls.append({
                "coin": s["coin"], "timeframe": tf, "bar": bar,
                "span_days": span, "shortfall": bar - span,
                "deepest_source": s.get("deepest_source"),
                "rows": s.get("rows", 0),
            })
    md.append(
        "## Source-depth shortfalls "
        "(`span_days` < gate bar — source can't reach the gate)"
    )
    md.append("")
    md.append(
        f"_Listed when `span_days + {SHORTFALL_BUFFER_DAYS}` < `COVERAGE_BAR_DAYS[tf]`. "
        "These coins will fail the gate no matter how many backfill "
        "iterations run, because the deepest source itself doesn't "
        "reach back far enough. Add a deeper venue (e.g. extend "
        "`COINBASE_PRODUCTS` for 5m) or accept the skip._"
    )
    md.append("")
    if not shortfalls:
        md.append("_No coin has a source shorter than its gate bar._")
    else:
        md.append(
            "| coin | tf | gate bar (d) | span (d) | shortfall (d) "
            "| deepest source | rows |"
        )
        md.append("|---|---|---:|---:|---:|---|---:|")
        for sh in sorted(shortfalls, key=lambda x: (-x["shortfall"], x["coin"], x["timeframe"])):
            md.append(
                f"| {sh['coin']} | {sh['timeframe']} | {sh['bar']} "
                f"| {sh['span_days']:.1f} | {sh['shortfall']:.1f} "
                f"| {sh['deepest_source'] or 'no data'} | {sh['rows']} |"
            )
    md.append("")
    md.append("## Coinbase 5m fallback pre-pass (task #409)")
    md.append("")
    if not coinbase_log:
        md.append("_No coin needed/qualified for the Coinbase fallback._")
    for cb in coinbase_log:
        md.append(
            f"- `{cb['coin']}` ok={cb['ok']} advanced={cb['advanced']} days={cb['days']} "
            f"earliest_before={cb['earliest_before']} → earliest_after={cb['earliest_after']} "
            f"({cb['elapsed_sec']}s)"
        )
    md.append("")
    md.append("## Iterative 5m backfill log (per OKX-cap-bound iteration)")
    md.append("")
    if not backfill_log:
        md.append("_No 5m coin needed/qualified for backfill._")
    for b in backfill_log:
        iter_idx = b.get("iter", "?")
        coins_lbl = b.get("coins") or []
        md.append(f"- iter `{iter_idx}` over {len(coins_lbl)} coin(s):")
        for coin, pc in (b.get("per_coin") or {}).items():
            md.append(
                f"  - `{coin}` ok={pc.get('ok')} advanced={pc.get('advanced')} "
                f"earliest_before={pc.get('earliest_before')} → earliest_after={pc.get('earliest_after')} "
                f"({pc.get('elapsed_sec')}s)"
            )
    md.append("")
    md.append(
        "## Iterative 1d backfill log (task #417 — widen 1d window to ≥1000 days)"
    )
    md.append("")
    if not one_d_backfill_log:
        md.append("_No 1d coin needed/qualified for backfill._")
    for b in one_d_backfill_log:
        iter_idx = b.get("iter", "?")
        coins_lbl = b.get("coins") or []
        md.append(f"- iter `{iter_idx}` over {len(coins_lbl)} coin(s):")
        for coin, pc in (b.get("per_coin") or {}).items():
            md.append(
                f"  - `{coin}` ok={pc.get('ok')} advanced={pc.get('advanced')} "
                f"earliest_before={pc.get('earliest_before')} → earliest_after={pc.get('earliest_after')} "
                f"({pc.get('elapsed_sec')}s)"
            )
    md.append("")
    (run_dir / "phase2_data_integrity.md").write_text("\n".join(md))

    # Task #417 — surface the 1d-specific failures separately so the
    # operator can see at a glance whether the wider 1d window was
    # actually achieved. A 1d gate failure means the trainer will run
    # with fewer than 1000 1d rows for that coin; this is a soft warning
    # (the trainer will still attempt training on whatever exists), but
    # the noise-band math motivating task #417 will not hold.
    one_d_failures = [
        k for k, v in post_high_gate.items()
        if v.get("timeframe") == "1d" and not v["passed"]
    ]
    summary = {
        "phase": "data_audit",
        "status": "ok",
        "elapsed_sec": round(time.time() - started, 1),
        "skipped_5m_coins": skipped_5m_post,
        "passed_5m_coins": passed_5m_post,
        "higher_tf_failures": failed_high,
        "one_d_floor_failures": one_d_failures,
        "one_d_floor_days": COVERAGE_BAR_DAYS["1d"],
    }
    _append_progress({
        "phase": "data_audit",
        "status": "ok",
        "headline": (
            f"5m skipped={len(skipped_5m_post)}/{len(DEFAULT_COINS)} "
            f"coins, higher-tf failures={len(failed_high)} "
            f"(1d_under_{COVERAGE_BAR_DAYS['1d']}d={len(one_d_failures)})"
        ),
        "skipped_5m_coins": skipped_5m_post,
        "one_d_floor_failures": one_d_failures,
    })
    return summary


# ── Phase 3 — pre-training baseline snapshot (archive) ───────────────────
def phase3_archive_baseline(run_dir: Path) -> dict:
    started = time.time()
    archive_dir = ARCHIVE_ROOT / f"{_ts()}_pre_full_run"
    archive_dir.mkdir(parents=True, exist_ok=True)
    # Copy report.json + verification_history.jsonl + per-coin folders so
    # the post-run diff has somewhere to reach back to. We DON'T move —
    # the live `models/` directory must stay readable for the running
    # ml-engine workflow.
    keep_files = ["report.json", "backtest_report.json",
                  "verification_history.jsonl", "calibration_recommendation.json"]
    copied: list[str] = []
    for f in keep_files:
        src = REGISTRY_ROOT / f
        if src.exists():
            shutil.copy2(src, archive_dir / f)
            copied.append(f)
    # Per-coin model folders — copy joblib + manifest only (small).
    per_coin_files = 0
    for coin_dir in REGISTRY_ROOT.iterdir():
        if not coin_dir.is_dir():
            continue
        if coin_dir.name in {"_archive", "datasets", "training_history"}:
            continue
        target = archive_dir / coin_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for inner in coin_dir.rglob("*"):
            if inner.is_file() and inner.suffix in {".json", ".joblib", ".pkl"}:
                rel = inner.relative_to(coin_dir)
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(inner, dst)
                    per_coin_files += 1
                except Exception:  # noqa: BLE001 — best-effort
                    pass

    baseline: dict[str, Any] = {"per_slice": {}}
    rep_path = REGISTRY_ROOT / "report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        for tf, tf_rep in (rep.get("timeframes") or {}).items():
            if not isinstance(tf_rep, dict):
                continue
            for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
                if isinstance(c_rep, dict):
                    baseline["per_slice"][f"{coin}/{tf}"] = _extract_snapshot_from_report(c_rep)
            pooled = tf_rep.get("pooled") or {}
            if isinstance(pooled, dict):
                baseline["per_slice"][f"__pooled__/{tf}"] = _extract_snapshot_from_report(pooled)
        baseline["verification"] = rep.get("verification")
        baseline["verification_diff"] = rep.get("verification_diff")
        baseline["generated_at"] = rep.get("generated_at")
    # Archived-model quant-only audit: scan every copied manifest /
    # feature_names payload for a FORBIDDEN_FEATURE_PREFIXES leak. The
    # registry's `load_model()` already rejects these at load time, but
    # the audit gives the operator a one-glance proof that the baseline
    # snapshot itself is clean before the new campaign overwrites it.
    from app.training.registry import FORBIDDEN_FEATURE_PREFIXES
    leaks: list[dict] = []
    scanned = 0
    for manifest in archive_dir.rglob("*.json"):
        try:
            payload = json.loads(manifest.read_text())
        except Exception:  # noqa: BLE001
            continue
        scanned += 1
        feats = payload.get("feature_names") or payload.get("features")
        if not isinstance(feats, list):
            continue
        for f in feats:
            if isinstance(f, str) and any(f.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES):
                leaks.append({"manifest": str(manifest.relative_to(archive_dir)), "feature": f})
    baseline["archived_model_audit"] = {
        "manifests_scanned": scanned,
        "forbidden_feature_leaks": leaks,
    }
    # Task #520 — durably capture every slice's walk-forward PnL from
    # the prior campaign by harvesting `slice_done` events from
    # `progress_updates.jsonl`. The existing `per_slice` block above is
    # built from `report.json`, which only carries the slices that the
    # *most recent* (possibly partial) run touched — historically as
    # few as 2. Without this harvest, the next "did PnL regress vs the
    # prior campaign?" check has nothing durable to diff against.
    prior_campaign_pnl = _harvest_prior_campaign_slice_pnl()
    baseline["prior_campaign_per_slice_pnl"] = prior_campaign_pnl
    baseline["schema_version"] = "task520_v1"
    (archive_dir / "baseline_snapshot.json").write_text(
        json.dumps(baseline, indent=2, default=str)
    )
    (run_dir / "phase3_baseline_pointer.json").write_text(json.dumps({
        "archive_dir": str(archive_dir.relative_to(ROOT)),
        "files_copied": copied,
        "per_coin_files_copied": per_coin_files,
        "baseline_slice_count": len(baseline.get("per_slice", {})),
        "prior_campaign_slice_count": prior_campaign_pnl.get("slice_count", 0),
        "prior_campaign_run_dir": prior_campaign_pnl.get("prior_run_dir"),
        "prior_campaign_completed": prior_campaign_pnl.get("completed", False),
    }, indent=2))
    _append_progress({
        "phase": "baseline_archive",
        "status": "ok",
        "headline": (
            f"baseline archived to {archive_dir.relative_to(ROOT)} "
            f"({len(baseline.get('per_slice', {}))} report slices, "
            f"{prior_campaign_pnl.get('slice_count', 0)} prior-campaign "
            f"slice_done rows)"
        ),
        "archive_dir": str(archive_dir.relative_to(ROOT)),
        "prior_campaign_slice_count": prior_campaign_pnl.get("slice_count", 0),
        "prior_campaign_run_dir": prior_campaign_pnl.get("prior_run_dir"),
        "prior_campaign_completed": prior_campaign_pnl.get("completed", False),
    })
    return {
        "phase": "baseline_archive",
        "status": "ok",
        "archive_dir": str(archive_dir.relative_to(ROOT)),
        "elapsed_sec": round(time.time() - started, 1),
        "baseline_slice_count": len(baseline.get("per_slice", {})),
        "prior_campaign_slice_count": prior_campaign_pnl.get("slice_count", 0),
    }


# ── Phase 4 — full training campaign ─────────────────────────────────────
_SNAPSHOT_STATE: dict = {
    "slices_done": [],          # list[dict]: rows accumulated as slices finish
    "consecutive_regress": 0,   # for watchdog-pause logic
    "last_snapshot_at": 0.0,
    "skip_5m_coins": [],
    "baseline_per_slice": {},   # populated by phase3 archive
    # Task #613 — count of operator-driven (or auto-timeout) resumes the
    # watchdog has consumed during this process. Used by the
    # `ML_WATCHDOG_SINGLE_RESUME=1` opt-in mode: under that mode, the
    # second halt does NOT wait — it writes a halt-and-report and exits
    # cleanly so the operator can investigate.
    "total_resumes": 0,
    # Task #613 — small ring buffer of the last few `_emit_snapshot`
    # rows so the halt-and-report can show the operator the
    # consecutive-regress trail that triggered the halt without
    # forcing them to grep `progress_updates.jsonl`.
    "recent_snapshots": [],
    # Task #613 — universe of (coin, tf) slices the campaign expects to
    # train, set by phase4. Used to derive a "pending slices" list in
    # the halt-and-report.
    "planned_slices": [],
}


# ── Task #613 — single-resume watchdog + per-slice live-gated replay ─────
def _economic_verdict(
    loose_pct: Optional[float],
    live_trades: Optional[int],
    live_pct: Optional[float],
) -> str:
    """Per-slice economic verdict from the loose post-fee PnL on the
    holdout (`pnl_after_fees.net_pct_total` from the trainer) and the
    live-gated replay returned by `scripts/diagnose_post_fee.py`:

      * `bleeding`     — loose<0 AND live n>=5 AND live<0
      * `dormant`      — loose<0 AND live n<5 (gates abstained, no edge realised)
      * `tradeable`    — live>0 AND live n>=5
      * `inconclusive` — neither set is observable, or signals disagree
    """
    n = int(live_trades or 0)
    if loose_pct is not None and loose_pct < 0:
        if n >= 5 and live_pct is not None and live_pct < 0:
            return "bleeding"
        if n < 5:
            return "dormant"
    if live_pct is not None and live_pct > 0 and n >= 5:
        return "tradeable"
    return "inconclusive"


def _run_live_replay(coin: str, timeframe: str) -> dict:
    """Best-effort: invoke `scripts/diagnose_post_fee.py` for the freshly
    trained (coin, timeframe) slot and return the live-gated PnL fields.

    NEVER raises — every failure path returns a dict with a populated
    `live_replay_status` (`skipped` / `error` / `timeout` / `ok`) and
    `live_replay_error` so the campaign cannot be blocked by a
    diagnostic regression.

    Pooled and specialist heads (`__pooled__`, `__specialist_*__`) are
    skipped because the diagnostic tool is per-coin.
    """
    if not coin or coin.startswith("__"):
        return {
            "live_replay_status": "skipped",
            "live_replay_error": "non-real coin (pooled or specialist head)",
            "live_trade_count": None,
            "live_net_pnl_pct": None,
            "dominant_rejection_reason": None,
        }
    latest_path = REGISTRY_ROOT / coin / timeframe / "latest"
    if not latest_path.exists():
        return {
            "live_replay_status": "skipped",
            "live_replay_error": "no `latest` pointer for slot on disk",
            "live_trade_count": None,
            "live_net_pnl_pct": None,
            "dominant_rejection_reason": None,
        }
    try:
        version = latest_path.read_text().strip()
    except Exception as exc:  # noqa: BLE001
        return {
            "live_replay_status": "skipped",
            "live_replay_error": f"read latest: {type(exc).__name__}: {exc}",
            "live_trade_count": None,
            "live_net_pnl_pct": None,
            "dominant_rejection_reason": None,
        }
    if not version:
        return {
            "live_replay_status": "skipped",
            "live_replay_error": "`latest` pointer is empty",
            "live_trade_count": None,
            "live_net_pnl_pct": None,
            "dominant_rejection_reason": None,
        }
    diag_root = ROOT / "diagnostics"
    try:
        diag_root.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    cmd = [
        sys.executable, "-m", "scripts.diagnose_post_fee",
        "--coin", coin, "--timeframe", timeframe, "--version", version,
        "--out-root", str(diag_root),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "live_replay_status": "timeout",
            "live_replay_error": "diagnose_post_fee subprocess exceeded 120s",
            "live_trade_count": None,
            "live_net_pnl_pct": None,
            "dominant_rejection_reason": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "live_replay_status": "error",
            "live_replay_error": f"{type(exc).__name__}: {exc}",
            "live_trade_count": None,
            "live_net_pnl_pct": None,
            "dominant_rejection_reason": None,
        }
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        return {
            "live_replay_status": "error",
            "live_replay_error": (
                f"non-zero exit {proc.returncode}: {tail[-512:]}"
            ),
            "live_trade_count": None,
            "live_net_pnl_pct": None,
            "dominant_rejection_reason": None,
        }
    try:
        out_envelope = json.loads(proc.stdout)
        out_dir = Path(out_envelope["output_dir"])
        summary = json.loads((out_dir / "summary.json").read_text())
    except Exception as exc:  # noqa: BLE001
        return {
            "live_replay_status": "error",
            "live_replay_error": (
                f"parse subprocess output: {type(exc).__name__}: {exc}"
            ),
            "live_trade_count": None,
            "live_net_pnl_pct": None,
            "dominant_rejection_reason": None,
        }
    aggregate = summary.get("aggregate") or {}
    skip_counts = (
        (summary.get("trade_distribution") or {}).get("skip_reason_counts")
        or {}
    )
    dominant = (
        max(skip_counts.items(), key=lambda x: x[1])[0] if skip_counts else None
    )
    try:
        rel_dir = str(out_dir.relative_to(ROOT))
    except ValueError:
        rel_dir = str(out_dir)
    return {
        "live_replay_status": "ok",
        "live_replay_error": None,
        "live_trade_count": int(aggregate.get("n_trades") or 0),
        "live_net_pnl_pct": aggregate.get("net_pct_total"),
        "dominant_rejection_reason": dominant,
        "live_replay_output_dir": rel_dir,
    }


def _parse_triggering_slice(reason: str) -> Optional[str]:
    """Extract the `<coin>/<tf>` slug from a watchdog reason string of
    the form `slice_done:<coin>/<tf>` or `heartbeat:...`. Returns None
    when the reason isn't slice-keyed (e.g. heartbeat halts)."""
    if not reason or not reason.startswith("slice_done:"):
        return None
    slug = reason.split(":", 1)[1].strip()
    return slug or None


def _write_halt_report(
    reason: str,
    state: dict,
    run_dir: Optional[Path] = None,
) -> Path:
    """Write a markdown halt-and-report under
    `<ml-engine>/diagnostics/campaign_halt_<TS>/REPORT.md` so the
    operator has a single artifact to open after a single-resume halt.

    The report concentrates everything the operator needs in one
    place per Task #613 acceptance: triggering slice identity, loose
    trainer PnL + n_trades for that slice, live-gated replay verdict
    and details, the consecutive-regress trail of the last few
    snapshots, the trained-so-far list, and the still-pending list.
    """
    ts = _ts()
    halt_dir = ROOT / "diagnostics" / f"campaign_halt_{ts}"
    halt_dir.mkdir(parents=True, exist_ok=True)
    report_path = halt_dir / "REPORT.md"

    slices_done = list(state.get("slices_done") or [])
    done_slugs = {row["slice"] for row in slices_done if row.get("slice")}
    planned = list(state.get("planned_slices") or [])
    pending = sorted(s for s in planned if s not in done_slugs)

    triggering_slug = _parse_triggering_slice(reason)
    triggering_row: Optional[dict] = None
    if triggering_slug:
        for row in reversed(slices_done):
            if row.get("slice") == triggering_slug:
                triggering_row = row
                break

    def _fmt(v: Any, kind: str = "") -> str:
        if v is None:
            return "n/a"
        if kind == "pct":
            try:
                return f"{float(v):+.4f}"
            except (TypeError, ValueError):
                return str(v)
        return str(v)

    md = ["# Task #613 — campaign halt report", ""]
    md.append(f"_Generated {_utcnow_iso()}_")
    md.append("")
    md.append("## Halt header")
    md.append("")
    md.append(f"- **Halt reason:** `{reason}`")
    md.append(
        f"- **Total operator/auto-timeout resumes consumed:** "
        f"{int(state.get('total_resumes', 0))}"
    )
    md.append(
        f"- **Consecutive regress/stall snapshots:** "
        f"{int(state.get('consecutive_regress', 0))}"
    )
    md.append(
        "- **Mode:** `ML_WATCHDOG_SINGLE_RESUME=1` (operator opted into "
        "one-shot resume — exiting cleanly after the second halt)."
    )
    if run_dir is not None:
        try:
            md.append(f"- **Run folder:** `{run_dir.relative_to(ROOT)}`")
        except ValueError:
            md.append(f"- **Run folder:** `{run_dir}`")
    md.append(
        f"- **Slices trained this run:** {len(slices_done)} of "
        f"{len(planned)} planned ({len(pending)} pending)."
    )
    md.append("")

    md.append("## Triggering slice")
    md.append("")
    if triggering_row is not None:
        loose_pct = triggering_row.get("post_fee_pct_total")
        loose_n = triggering_row.get("post_fee_n_trades")
        live_n = triggering_row.get("live_trade_count")
        live_pct = triggering_row.get("live_net_pnl_pct")
        verdict = _economic_verdict(loose_pct, live_n, live_pct)
        verdict_phrase = {
            "bleeding": "bleeding / negative under production gates",
            "dormant": "dormant / no-edge under production gates",
            "tradeable": "tradeable / positive under production gates",
            "inconclusive": "inconclusive / signals do not agree or live diagnostic missing",
        }.get(verdict, verdict)
        md.append(f"- **Slice:** `{triggering_row['slice']}`")
        md.append(f"- **Trainer status:** `{triggering_row.get('status')}`")
        md.append(
            "- **Loose trainer post-fee:** "
            f"`pct_total={_fmt(loose_pct, 'pct')}`, "
            f"`n_trades={_fmt(loose_n)}`"
        )
        md.append(
            "- **Live-gated replay:** "
            f"`pct_total={_fmt(live_pct, 'pct')}`, "
            f"`n_trades={_fmt(live_n)}`, "
            f"`status={triggering_row.get('live_replay_status') or 'missing'}`"
        )
        md.append(
            "- **Dominant rejection reason (live):** "
            f"`{triggering_row.get('dominant_rejection_reason') or 'n/a'}`"
        )
        if triggering_row.get("live_replay_error"):
            md.append(
                "- **Live replay error:** "
                f"`{triggering_row.get('live_replay_error')}`"
            )
        md.append(f"- **Economic verdict:** **{verdict_phrase}**")
    elif triggering_slug:
        md.append(
            f"- Triggering slug `{triggering_slug}` parsed from reason but "
            "no matching `slice_done` row recorded in this process. "
            "Inspect `models/progress_updates.jsonl` directly."
        )
    else:
        md.append(
            "- Halt reason is not slice-keyed (e.g. heartbeat halt). "
            "See the snapshot trail below for the failing watchdog verdict."
        )
    md.append("")

    md.append("## Last 3 snapshot trail (consecutive-regress evidence)")
    md.append("")
    md.append("| at | reason | watchdog verdict | consec regress | best | worst | halted? |")
    md.append("|---|---|---|---:|---|---|:---:|")
    trail = list(state.get("recent_snapshots") or [])[-3:]
    if not trail:
        md.append("| — | (no snapshots emitted in this process) | — | — | — | — | — |")
    for snap in trail:
        md.append(
            f"| {snap.get('at') or 'n/a'} "
            f"| `{snap.get('reason') or 'n/a'}` "
            f"| {snap.get('watchdog_verdict') or 'n/a'} "
            f"| {int(snap.get('consecutive_regress') or 0)} "
            f"| {snap.get('best_slice') or 'n/a'} "
            f"| {snap.get('worst_slice') or 'n/a'} "
            f"| {'YES' if snap.get('halted') else 'no'} |"
        )
    md.append("")

    md.append(f"## Trained slices so far ({len(slices_done)})")
    md.append("")
    md.append("| slice | status | loose pct_total | loose n | live pct_total | live n | verdict |")
    md.append("|---|---|---:|---:|---:|---:|---|")
    if not slices_done:
        md.append("| — | — | — | — | — | — | — |")
    for row in slices_done:
        v = _economic_verdict(
            row.get("post_fee_pct_total"),
            row.get("live_trade_count"),
            row.get("live_net_pnl_pct"),
        )
        md.append(
            f"| `{row.get('slice')}` "
            f"| {row.get('status') or 'n/a'} "
            f"| {_fmt(row.get('post_fee_pct_total'), 'pct')} "
            f"| {_fmt(row.get('post_fee_n_trades'))} "
            f"| {_fmt(row.get('live_net_pnl_pct'), 'pct')} "
            f"| {_fmt(row.get('live_trade_count'))} "
            f"| {v} |"
        )
    md.append("")

    md.append(f"## Pending slices ({len(pending)})")
    md.append("")
    if pending:
        md.append(", ".join(f"`{s}`" for s in pending))
    else:
        md.append("_None — every planned slice was attempted this run._")
    md.append("")

    md.append("## Operator next-action checklist")
    md.append("")
    md.append(
        "- [ ] Open `models/progress_updates.jsonl` and grep for "
        "`phase=snapshot` to see the most recent best/worst slice + "
        "watchdog verdict."
    )
    md.append(
        "- [ ] Inspect per-slice live-gated replay output in "
        "`diagnostics/<coin>_<tf>_post_fee_<TS>/summary.json` (latest "
        "write per slice)."
    )
    md.append(
        "- [ ] Decide whether to (a) raise a fix task, (b) re-run the "
        "campaign with `ML_WATCHDOG_SINGLE_RESUME=1` so the next halt "
        "cycle is bounded to one resume, or (c) leave the registry "
        "as-is and re-train later."
    )
    md.append(
        "- [ ] If you want a manual resume before restart, set "
        "`ML_WATCHDOG_RESUME=1` in the env or "
        "`touch .local/.task366_resume` BEFORE restarting the workflow."
    )
    md.append("")
    report_path.write_text("\n".join(md))
    return report_path


def _handle_watchdog_halt(
    reason: str,
    snapshot_state: dict,
    run_dir: Optional[Path] = None,
) -> None:
    """Implement the watchdog-pause policy. Two modes:

      * Default (no env): wait up to `ML_WATCHDOG_MAX_HALT_SEC` for
        `ML_WATCHDOG_RESUME=1` or the sentinel file, then continue.
        Resume counter is incremented but never gates exit.
      * `ML_WATCHDOG_SINGLE_RESUME=1`: the FIRST halt waits exactly
        like the default mode and increments the resume counter. Any
        subsequent halt skips the wait, writes a halt-and-report
        markdown, and `sys.exit(0)` so the orchestrator finishes
        cleanly with halt artifacts the operator can read.

    Extracted from `_emit_snapshot` so it can be unit-tested without
    spinning up the full Phase-4 driver.
    """
    single_resume = os.environ.get("ML_WATCHDOG_SINGLE_RESUME") == "1"
    consumed = int(snapshot_state.get("total_resumes", 0))
    max_halt_sec = int(os.environ.get("ML_WATCHDOG_MAX_HALT_SEC", "1800"))
    sentinel_path = Path(
        os.environ.get("ML_WATCHDOG_RESUME_FILE", ".local/.task366_resume")
    )
    if not sentinel_path.is_absolute():
        sentinel_path = ROOT.parents[1] / sentinel_path

    if single_resume and consumed >= 1:
        report_path = _write_halt_report(reason, snapshot_state, run_dir)
        try:
            rel_report = str(report_path.relative_to(ROOT))
        except ValueError:
            rel_report = str(report_path)
        _append_progress({
            "phase": "watchdog_halt",
            "status": "halted_final",
            "reason": reason,
            "headline": (
                f"single-resume mode: second halt reached after "
                f"{consumed} resume(s); writing halt-and-report and "
                f"exiting cleanly. report={rel_report}"
            ),
            "consecutive_regress": int(
                snapshot_state.get("consecutive_regress", 0)
            ),
            "total_resumes_consumed": consumed,
            "single_resume_mode": True,
            "halt_report_path": rel_report,
        })
        sys.exit(0)

    _append_progress({
        "phase": "watchdog_halt",
        "status": "halted",
        "reason": reason,
        "headline": (
            f"campaign halted after "
            f"{int(snapshot_state.get('consecutive_regress', 0))} "
            f"consecutive regress/stall snapshots. Resume by setting "
            f"ML_WATCHDOG_RESUME=1 or `touch {sentinel_path}`. "
            f"Bounded auto-resume after {max_halt_sec}s."
            + (
                " Single-resume mode is ON: the next halt will exit cleanly."
                if single_resume else ""
            )
        ),
        "consecutive_regress": int(
            snapshot_state.get("consecutive_regress", 0)
        ),
        "resume_env_var": "ML_WATCHDOG_RESUME",
        "resume_sentinel_file": str(sentinel_path),
        "max_halt_sec": max_halt_sec,
        "single_resume_mode": single_resume,
        "total_resumes_consumed": consumed,
    })
    halt_started = time.time()
    resume_cause = "timeout"
    while time.time() - halt_started < max_halt_sec:
        if os.environ.get("ML_WATCHDOG_RESUME") == "1":
            resume_cause = "env_var"
            break
        if sentinel_path.exists():
            resume_cause = "sentinel_file"
            try:
                sentinel_path.unlink()
            except Exception:  # noqa: BLE001
                pass
            break
        time.sleep(5)
    elapsed_halt = round(time.time() - halt_started, 1)
    snapshot_state["total_resumes"] = consumed + 1
    _append_progress({
        "phase": "watchdog_resume",
        "status": "ok" if resume_cause != "timeout" else "warn",
        "headline": (
            f"resumed after {elapsed_halt}s via {resume_cause}"
            + (
                " (single-resume mode ON — a subsequent halt will exit cleanly)"
                if single_resume else ""
            )
        ),
        "resume_cause": resume_cause,
        "halted_seconds": elapsed_halt,
        "total_resumes": consumed + 1,
        "single_resume_mode": single_resume,
    })
    snapshot_state["consecutive_regress"] = 0


_POST_FEE_DIAGNOSTIC_GLOB = "*_post_fee_*"


def _parse_post_fee_diagnostic_dirname(
    name: str,
) -> Optional[tuple[str, str]]:
    """Parse `<coin>_<tf>_post_fee_<TS>` into `(coin, tf)`.

    Returns None when the name does not match the expected layout
    produced by `scripts/diagnose_post_fee.py`. The coin token may
    itself contain underscores; the timeframe is taken as the segment
    immediately preceding `_post_fee_`.
    """
    sep = "_post_fee_"
    idx = name.find(sep)
    if idx <= 0 or idx + len(sep) >= len(name):
        return None
    prefix = name[:idx]
    if "_" not in prefix:
        return None
    coin, tf = prefix.rsplit("_", 1)
    if not coin or not tf:
        return None
    return coin, tf


def _backfill_post_fee_diagnostics() -> list[dict]:
    """Scan `diagnostics/*_post_fee_*` and emit at most one idempotent
    `slice_live_replay_backfill` row per (coin, tf) per run into
    `models/progress_updates.jsonl`.

    Each emitted row carries the same shape that the live-replay hook
    would have produced inside the campaign, so the per-slice economic
    verdict pipeline (which reads `progress_updates.jsonl`) has a
    live-gated row for any slice the operator manually post-fee'd —
    even when the original training campaign halted before the
    live-replay hook ran.

    Behaviour:
    - Within a single run: if multiple
      `diagnostics/<coin>_<tf>_post_fee_<TS>` directories exist for
      the same `(coin, tf)`, the lexicographically-latest directory
      name (i.e. the newest timestamp suffix) is selected and at most
      one row is appended for that slice.
    - Across runs: idempotency is keyed off the chosen
      `source_diagnostic` per slice. If the latest diagnostic for a
      slice was already back-filled in a prior run, the slice is a
      no-op. If a newer diagnostic appears later, the next run picks
      it up.

    Returns the list of newly appended records (may be empty).
    """
    diag_root = ROOT / "diagnostics"
    if not diag_root.is_dir():
        return []

    seen_markers: set[str] = set()
    if PROGRESS_PATH.exists():
        try:
            with PROGRESS_PATH.open() as fh:
                for line in fh:
                    if "slice_live_replay_backfill" not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if rec.get("phase") != "slice_live_replay_backfill":
                        continue
                    marker = rec.get("source_diagnostic")
                    if isinstance(marker, str) and marker:
                        seen_markers.add(marker)
        except Exception:  # noqa: BLE001
            pass

    # Group candidate diagnostic dirs by (coin, tf), keeping the
    # lexicographically-latest dir name per slice (TS suffix sorts
    # chronologically thanks to the `YYYYMMDDTHHMMSSZ` format).
    latest_by_slice: dict[tuple[str, str], str] = {}
    for diag_dir in diag_root.glob(_POST_FEE_DIAGNOSTIC_GLOB):
        if not diag_dir.is_dir():
            continue
        parsed = _parse_post_fee_diagnostic_dirname(diag_dir.name)
        if parsed is None:
            continue
        if not (diag_dir / "summary.json").exists():
            continue
        prior = latest_by_slice.get(parsed)
        if prior is None or diag_dir.name > prior:
            latest_by_slice[parsed] = diag_dir.name

    appended: list[dict] = []
    for (coin, timeframe), marker in sorted(latest_by_slice.items()):
        if marker in seen_markers:
            continue
        summary_path = diag_root / marker / "summary.json"
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:  # noqa: BLE001
            err_record = {
                "phase": "slice_live_replay_backfill",
                "status": "error",
                "headline": (
                    f"{coin}/{timeframe} backfill: parse summary failed: "
                    f"{exc}"
                ),
                "coin": coin,
                "timeframe": timeframe,
                "source_diagnostic": marker,
                "live_replay_status": "error",
                "live_replay_error": str(exc),
                "live_trade_count": None,
                "live_net_pnl_pct": None,
                "dominant_rejection_reason": None,
            }
            _append_progress(err_record)
            seen_markers.add(marker)
            appended.append(err_record)
            continue
        aggregate = summary.get("aggregate") or {}
        skip_counts = (
            (summary.get("trade_distribution") or {}).get(
                "skip_reason_counts"
            )
            or {}
        )
        dominant = (
            max(skip_counts.items(), key=lambda x: x[1])[0]
            if skip_counts
            else None
        )
        record = {
            "phase": "slice_live_replay_backfill",
            "status": "ok",
            "coin": coin,
            "timeframe": timeframe,
            "source_diagnostic": marker,
            "headline": (
                f"{coin}/{timeframe} live-replay back-fill from {marker}: "
                f"n={aggregate.get('n_trades')} "
                f"net_pct_total={aggregate.get('net_pct_total')} "
                f"dominant_rejection={dominant}"
            ),
            "live_trade_count": int(aggregate.get("n_trades") or 0),
            "live_net_pnl_pct": aggregate.get("net_pct_total"),
            "dominant_rejection_reason": dominant,
            "live_replay_status": "ok",
            "live_replay_error": None,
            "live_replay_output_dir": f"diagnostics/{marker}",
        }
        _append_progress(record)
        seen_markers.add(marker)
        appended.append(record)
    return appended


def _aggregate_live_replay_per_slice(
    run_started_at_iso: Optional[str] = None,
) -> dict[str, dict]:
    """Read `models/progress_updates.jsonl` and return the latest
    live-replay snapshot per `coin/tf` slice.

    `slice_live_replay_backfill` rows are always honoured (they are
    operator back-fills). Regular `slice_done` rows are filtered to the
    current campaign window if `run_started_at_iso` is provided.
    """
    out: dict[str, dict] = {}
    if not PROGRESS_PATH.exists():
        return out
    try:
        with PROGRESS_PATH.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                phase = rec.get("phase")
                if phase not in (
                    "slice_done",
                    "slice_live_replay_backfill",
                ):
                    continue
                if "live_replay_status" not in rec:
                    continue
                emitted = rec.get("emitted_at")
                if (
                    run_started_at_iso
                    and phase == "slice_done"
                    and emitted
                    and emitted < run_started_at_iso
                ):
                    continue
                slug = f"{rec.get('coin')}/{rec.get('timeframe')}"
                out[slug] = {
                    "live_trade_count": rec.get("live_trade_count"),
                    "live_net_pnl_pct": rec.get("live_net_pnl_pct"),
                    "dominant_rejection_reason": rec.get(
                        "dominant_rejection_reason"
                    ),
                    "live_replay_status": rec.get("live_replay_status"),
                    "live_replay_error": rec.get("live_replay_error"),
                    "live_replay_output_dir": rec.get(
                        "live_replay_output_dir"
                    ),
                    "source_phase": phase,
                    "source_emitted_at": emitted,
                }
    except Exception:  # noqa: BLE001
        pass
    return out


def _emit_snapshot(reason: str, baseline_dir: Optional[Path] = None) -> None:
    """Compute and append a rich snapshot record covering every Phase 5
    contract item: best/worst by post-fee, watchdog verdict, post-fee
    trend vs the archived baseline, regime-mismatch dominance, newly
    promotable slices, class-collapse warnings, persistent 5m skip list.

    Reads the freshly-written `models/report.json` so every number is
    citable on disk. Safe to call between slices (the hook does this
    after every slice_done) and on a 15-minute heartbeat.
    """
    rep_path = REGISTRY_ROOT / "report.json"
    if not rep_path.exists():
        return
    try:
        report = json.loads(rep_path.read_text())
    except Exception:
        return
    best, worst = _rank_slices(report)
    baseline_per_slice = _SNAPSHOT_STATE.get("baseline_per_slice") or {}
    diff = _post_fee_baseline_diff({"per_slice": baseline_per_slice}, report)
    fb = _failure_buckets(report)
    verification = report.get("verification") or {}
    per_slice = verification.get("per_slice") or []
    promote = [f"{s['coin']}/{s['timeframe']}" for s in per_slice if s.get("promoted")]
    watchdog = report.get("verification_diff") or {}
    watchdog_verdict = watchdog.get("verdict") or watchdog.get("status") or "unknown"
    # Regime-mismatch dominance: which failure bucket has the most
    # slices, and is `regime_mismatch` (or close variant) the top?
    top_bucket = None
    top_count = 0
    if fb:
        top_bucket, top_count = max(fb.items(), key=lambda x: x[1])
    regime_dominant = bool(top_bucket and "regime" in str(top_bucket).lower())
    # Class collapse — surfaced by train_one_slice in the slice records
    collapsed: list[str] = []
    for tf, tf_rep in (report.get("timeframes") or {}).items():
        if not isinstance(tf_rep, dict):
            continue
        for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
            if isinstance(c_rep, dict) and (c_rep.get("prediction_collapse") or {}).get("collapsed"):
                collapsed.append(f"{coin}/{tf}")
    # Trend vs baseline — share of slices with positive post-fee delta
    deltas = [d["delta_post_fee_pct_total"] for d in diff.values()
              if d.get("delta_post_fee_pct_total") is not None]
    trend_share_improving = (sum(1 for d in deltas if d > 0) / len(deltas)) if deltas else None
    # Watchdog pause logic: if the verdict is regressed/stalled twice in
    # a row, halt with an explicit `halted` status and wait for an
    # operator-driven resume signal — either ML_WATCHDOG_RESUME=1 in
    # the env, or a sentinel file at $ML_WATCHDOG_RESUME_FILE (default
    # `.local/.task366_resume`). Bounded by ML_WATCHDOG_MAX_HALT_SEC
    # (default 1800s) so a forgotten halt cannot wedge the campaign
    # forever; on timeout we emit `pause_timeout` and continue. The
    # halt is consumed (counter reset, sentinel deleted) so a subsequent
    # regression triggers a fresh halt.
    if watchdog_verdict in ("regressed", "stalled"):
        _SNAPSHOT_STATE["consecutive_regress"] += 1
    else:
        _SNAPSHOT_STATE["consecutive_regress"] = 0
    halted = _SNAPSHOT_STATE["consecutive_regress"] >= 2

    _append_progress({
        "phase": "snapshot",
        "status": "halted" if halted else "ok",
        "reason": reason,
        "headline": (
            f"snapshot reason={reason} watchdog={watchdog_verdict} "
            f"best={(best[0]['slice'] if best else 'n/a')} "
            f"worst={(worst[0]['slice'] if worst else 'n/a')} "
            f"promoted={len(promote)} regime_dominant={regime_dominant} "
            f"trend_improving_share={trend_share_improving}"
            + (" HALTED-AWAITING-RESUME" if halted else "")
        ),
        "watchdog_verdict": watchdog_verdict,
        "watchdog_halted": halted,
        "consecutive_regress": _SNAPSHOT_STATE["consecutive_regress"],
        "best_slice": best[0] if best else None,
        "worst_slice": worst[0] if worst else None,
        "newly_promotable": promote,
        "regime_mismatch_dominant": regime_dominant,
        "top_failure_bucket": {"name": top_bucket, "count": top_count},
        "class_collapsed_slices": collapsed,
        "trend_vs_baseline_improving_share": trend_share_improving,
        "post_fee_profitable_count": sum(
            1 for d in diff.values()
            if d.get("new_post_fee_pct_total") is not None
            and d["new_post_fee_pct_total"] > 0
        ),
        "persistent_5m_skip_list": list(_SNAPSHOT_STATE.get("skip_5m_coins") or []),
    })
    _SNAPSHOT_STATE["last_snapshot_at"] = time.time()
    # Task #613 — keep the last 5 snapshot rows so the halt-and-report
    # can show the operator the regress trail without grepping the
    # progress log.
    recent: list = _SNAPSHOT_STATE.setdefault("recent_snapshots", [])
    recent.append({
        "at": _utcnow_iso(),
        "reason": reason,
        "watchdog_verdict": watchdog_verdict,
        "consecutive_regress": int(_SNAPSHOT_STATE["consecutive_regress"]),
        "best_slice": (best[0]["slice"] if best else None),
        "worst_slice": (worst[0]["slice"] if worst else None),
        "halted": halted,
    })
    if len(recent) > 5:
        del recent[: len(recent) - 5]
    if halted:
        # Task #613 — wait/exit policy lives in `_handle_watchdog_halt`
        # so the single-resume opt-in path is unit-testable without
        # spinning up the rest of Phase 4.
        _handle_watchdog_halt(reason, _SNAPSHOT_STATE)


def _install_progress_hook() -> None:
    """Monkey-patch `app.training.train.train_one_slice` so that every
    (coin, timeframe) model fit emits a structured progress record. This
    satisfies the task contract for "periodic structured proof updates
    after each slice". The hook runs in-process; it does not touch the
    model fitting itself.
    """
    from app.training import train as train_mod
    original = train_mod.train_one_slice
    if getattr(original, "__wrapped_by_task366__", False):
        return

    def wrapped(df, *, coin_id, timeframe, vocab, **kwargs):
        t0 = time.time()
        _append_progress({
            "phase": "slice_start",
            "status": "running",
            "headline": f"fitting {coin_id}/{timeframe}",
            "coin": coin_id, "timeframe": timeframe,
            "n_rows_input": int(len(df)),
        })
        try:
            res = original(df, coin_id=coin_id, timeframe=timeframe, vocab=vocab, **kwargs)
        except Exception as exc:  # noqa: BLE001
            _append_progress({
                "phase": "slice_done", "status": "error",
                "headline": f"{coin_id}/{timeframe} raised {type(exc).__name__}: {exc}",
                "coin": coin_id, "timeframe": timeframe,
                "elapsed_sec": round(time.time() - t0, 1),
            })
            raise
        pnl = (res or {}).get("pnl_after_fees") or {}
        metrics = (res or {}).get("metrics") or {}
        base = (res or {}).get("baseline_metrics") or {}
        # Task #613 — best-effort live-gated replay via the
        # post-fee diagnostic. Bounded to 120s and never raises so the
        # campaign cannot be blocked by a diagnostic regression. The
        # five live-gated fields below carry the verdict-ready inputs
        # for the per-slice economic verdict (bleeding/dormant/
        # tradeable/inconclusive) computed in Phase 7.
        try:
            live_replay = _run_live_replay(coin_id, timeframe)
        except Exception as exc:  # noqa: BLE001 — defence in depth
            live_replay = {
                "live_replay_status": "error",
                "live_replay_error": (
                    f"unexpected: {type(exc).__name__}: {exc}"
                ),
                "live_trade_count": None,
                "live_net_pnl_pct": None,
                "dominant_rejection_reason": None,
            }
        _append_progress({
            "phase": "slice_done",
            "status": (res or {}).get("status") or "unknown",
            "headline": (
                f"{coin_id}/{timeframe} status={(res or {}).get('status')} "
                f"auc={metrics.get('auc')} base_auc={base.get('auc')} "
                f"post_fee_pct_total={pnl.get('net_pct_total')} "
                f"n_trades={pnl.get('n_trades')} "
                f"live_n={live_replay.get('live_trade_count')} "
                f"live_pct={live_replay.get('live_net_pnl_pct')} "
                f"live_status={live_replay.get('live_replay_status')}"
            ),
            "coin": coin_id, "timeframe": timeframe,
            "elapsed_sec": round(time.time() - t0, 1),
            "metrics": metrics,
            "baseline_metrics": base,
            "lift_auc": (res or {}).get("lift_auc"),
            "pnl_after_fees": pnl,
            "n_rows": (res or {}).get("n_rows"),
            "prediction_collapse": (res or {}).get("prediction_collapse"),
            "live_trade_count": live_replay.get("live_trade_count"),
            "live_net_pnl_pct": live_replay.get("live_net_pnl_pct"),
            "dominant_rejection_reason": live_replay.get(
                "dominant_rejection_reason"
            ),
            "live_replay_status": live_replay.get("live_replay_status"),
            "live_replay_error": live_replay.get("live_replay_error"),
            "live_replay_output_dir": live_replay.get(
                "live_replay_output_dir"
            ),
        })
        # Task #613 — record a compact slice-done row in process state
        # so `_write_halt_report` can show the operator the trained
        # slice list and look up the triggering slice's loose/live PnL
        # without re-reading `progress_updates.jsonl`.
        _SNAPSHOT_STATE.setdefault("slices_done", []).append({
            "slice": f"{coin_id}/{timeframe}",
            "coin": coin_id,
            "timeframe": timeframe,
            "status": (res or {}).get("status") or "unknown",
            "post_fee_pct_total": pnl.get("net_pct_total"),
            "post_fee_n_trades": pnl.get("n_trades"),
            "live_trade_count": live_replay.get("live_trade_count"),
            "live_net_pnl_pct": live_replay.get("live_net_pnl_pct"),
            "dominant_rejection_reason": live_replay.get(
                "dominant_rejection_reason"
            ),
            "live_replay_status": live_replay.get("live_replay_status"),
            "live_replay_error": live_replay.get("live_replay_error"),
        })
        # Per-Phase-5 contract: a structured snapshot covering best/
        # worst, watchdog verdict, trend vs baseline, regime-mismatch
        # dominance, newly promotable, class-collapse warnings, and the
        # persistent 5m skip list — emitted after every slice and at
        # most every 15 minutes (heartbeat is enforced inside the
        # tf-loop wrapper). Safe-noop until report.json exists.
        try:
            _emit_snapshot(reason=f"slice_done:{coin_id}/{timeframe}")
        except Exception as e:  # noqa: BLE001
            _append_progress({
                "phase": "snapshot_error",
                "status": "warn",
                "headline": f"snapshot failed after {coin_id}/{timeframe}: {e}",
            })
        return res

    wrapped.__wrapped_by_task366__ = True  # type: ignore[attr-defined]
    train_mod.train_one_slice = wrapped


def phase4_training(run_dir: Path, skip_5m_coins: list[str]) -> dict:
    """Drive the existing `run_training` entry point with a one-year
    lookback for every monitored coin × every tradeable timeframe that
    survived the data-integrity gate.

    5m gating is per-coin: coins that clear the hard gate are trained on
    5m via a dedicated `run_training(passing_5m_coins, ['5m'])` pass,
    while coins that failed are explicitly skipped with a progress
    record. The higher timeframes (1h/2h/6h/1d) are always trained on
    the full universe.
    """
    started = time.time()
    coins = list(DEFAULT_COINS)
    _install_progress_hook()
    higher_tfs = ["1h", "2h", "6h", "1d"]
    train_5m = sorted(set(coins) - set(skip_5m_coins))
    # Populate snapshot state used by `_emit_snapshot` after every slice
    _SNAPSHOT_STATE["skip_5m_coins"] = sorted(skip_5m_coins)
    _SNAPSHOT_STATE["consecutive_regress"] = 0
    _SNAPSHOT_STATE["last_snapshot_at"] = time.time()
    # Task #613 — record the universe of (coin, tf) slices the
    # campaign expects to train so `_write_halt_report` can derive a
    # "pending slices" list. Skipped 5m slices are deliberately
    # excluded since they will never be attempted this run.
    planned: list[str] = []
    for coin in coins:
        if coin in train_5m:
            planned.append(f"{coin}/5m")
        for tf in higher_tfs:
            planned.append(f"{coin}/{tf}")
    _SNAPSHOT_STATE["planned_slices"] = sorted(set(planned))
    # Reset trail/done lists per run so reports are scoped to this run.
    _SNAPSHOT_STATE["slices_done"] = []
    _SNAPSHOT_STATE["recent_snapshots"] = []
    _SNAPSHOT_STATE["total_resumes"] = 0
    # Pull the most recent baseline_snapshot.json the orchestrator just
    # archived, so per-slice deltas vs baseline are computed in-flight.
    try:
        latest = sorted(ARCHIVE_ROOT.glob("*_pre_full_run/baseline_snapshot.json"))
        if latest:
            _SNAPSHOT_STATE["baseline_per_slice"] = (
                json.loads(latest[-1].read_text()).get("per_slice") or {}
            )
    except Exception:  # noqa: BLE001
        _SNAPSHOT_STATE["baseline_per_slice"] = {}
    # Task #417 — surface the per-tf lookback in the training_start
    # record so the operator can confirm 1d is using the wider window.
    lookback_per_tf = {
        tf: lookback_days_for(tf)
        for tf in (["5m"] if train_5m else []) + higher_tfs
    }
    _append_progress({
        "phase": "training_start",
        "status": "running",
        "headline": (
            f"higher_tfs={higher_tfs}, coins={coins}, "
            f"5m_train_list={train_5m}, 5m_skip_list={sorted(skip_5m_coins)}, "
            f"lookback_per_tf={lookback_per_tf}"
        ),
        "lookback_days": LOOKBACK_DAYS,
        "lookback_days_per_tf": lookback_per_tf,
        "five_m_pass_list": train_5m,
        "five_m_skip_list": sorted(skip_5m_coins),
    })
    # Explicit per-coin 5m skip record so the jsonl audit trail names
    # every coin that did not train at 5m and why.
    for coin in sorted(skip_5m_coins):
        _append_progress({
            "phase": "slice_skipped", "status": "skipped",
            "headline": f"{coin}/5m skipped — 5m hard gate failed",
            "coin": coin, "timeframe": "5m",
            "reason": "5m_hard_gate_failed",
        })

    report: dict[str, Any] = {"timeframes": {}}

    # 15-minute heartbeat thread — emits a snapshot if the in-process
    # slice hook hasn't fired one in `heartbeat_sec`. Some specialist /
    # pooled fits run >15 min on the slowest tf, and the task contract
    # explicitly requires "every fifteen minutes if a slice is taking
    # longer".
    import threading
    heartbeat_sec = int(os.environ.get("ML_HEARTBEAT_SEC", "900"))
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(60):
            quiet_for = time.time() - (_SNAPSHOT_STATE.get("last_snapshot_at") or 0)
            if quiet_for >= heartbeat_sec:
                try:
                    _emit_snapshot(reason=f"heartbeat:{int(quiet_for)}s_quiet")
                except Exception as e:  # noqa: BLE001
                    _append_progress({
                        "phase": "snapshot_error", "status": "warn",
                        "headline": f"heartbeat snapshot failed: {e}",
                    })
    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    # Per-coin 5m training pass (only coins that cleared the hard gate).
    if train_5m:
        _append_progress({
            "phase": "tf_start", "status": "running",
            "headline": f"5m training on {len(train_5m)} gated coins",
            "timeframe": "5m", "coins": train_5m,
        })
        r5 = asyncio.run(run_training(
            train_5m, ["5m"], progress_callback=_append_progress,
        ))
        for k, v in (r5.get("timeframes") or {}).items():
            report["timeframes"][k] = v
        # Carry forward non-timeframe top-level keys on first pass.
        for k, v in r5.items():
            if k not in ("timeframes",):
                report[k] = v

    # Higher-timeframe pass on the full universe.
    _append_progress({
        "phase": "tf_start", "status": "running",
        "headline": f"higher-tf training on {len(coins)} coins, tfs={higher_tfs}",
        "timeframes": higher_tfs, "coins": coins,
    })
    r_high = asyncio.run(run_training(
        coins, higher_tfs, progress_callback=_append_progress,
    ))
    for k, v in (r_high.get("timeframes") or {}).items():
        report["timeframes"][k] = v
    for k, v in r_high.items():
        if k != "timeframes":
            report[k] = v
    heartbeat_stop.set()
    elapsed = round(time.time() - started, 1)
    (run_dir / "phase4_training_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )

    # ── Phase 4b — in-band recency-weighting A/B (round-2 review fix) ──
    # Run a deterministic, bounded paired evaluation with the flipped
    # ML_RECENCY_WEIGHTED env on a fixed subset of (coin × tf) slices.
    # The trainer may or may not honor this knob today — running both
    # modes and diffing post-fee net_pct_total per slice is the only
    # honest way to answer Phase 6(d). The result is written to
    # `phase4b_recency_ab.json`. The subset is intentionally small
    # (3 coins × ['1d']) so total wall time grows by a few minutes
    # rather than doubling. Operator can override via
    # ML_RECENCY_AB_COINS / ML_RECENCY_AB_TFS / ML_RECENCY_AB_DISABLE=1.
    ab_payload: dict = {
        "skipped": False,
        "this_run_mode": os.environ.get("ML_RECENCY_WEIGHTED", "1"),
        "flipped_mode": None,
        "subset_coins": [],
        "subset_tfs": [],
        "per_slice_diff": {},
        "summary": {},
    }
    if os.environ.get("ML_RECENCY_AB_DISABLE") == "1":
        ab_payload["skipped"] = True
        ab_payload["skip_reason"] = "ML_RECENCY_AB_DISABLE=1"
    else:
        ab_coins_env = os.environ.get("ML_RECENCY_AB_COINS")
        ab_tfs_env = os.environ.get("ML_RECENCY_AB_TFS")
        ab_coins = [c.strip() for c in ab_coins_env.split(",") if c.strip()] \
            if ab_coins_env else list(coins[:3])
        ab_tfs = [t.strip() for t in ab_tfs_env.split(",") if t.strip()] \
            if ab_tfs_env else ["1d"]
        ab_payload["subset_coins"] = ab_coins
        ab_payload["subset_tfs"] = ab_tfs
        original_mode = os.environ.get("ML_RECENCY_WEIGHTED", "1")
        flipped_mode = "0" if original_mode == "1" else "1"
        ab_payload["flipped_mode"] = flipped_mode
        _append_progress({
            "phase": "recency_ab_start", "status": "running",
            "headline": (
                f"Phase 4b recency A/B: re-training {len(ab_coins)} coin(s) "
                f"× {ab_tfs} with ML_RECENCY_WEIGHTED={flipped_mode} "
                f"(this run was {original_mode})"
            ),
            "subset_coins": ab_coins, "subset_tfs": ab_tfs,
            "this_run_mode": original_mode, "flipped_mode": flipped_mode,
        })
        ab_started = time.time()
        try:
            os.environ["ML_RECENCY_WEIGHTED"] = flipped_mode
            r_ab = asyncio.run(run_training(ab_coins, ab_tfs))
        except Exception as exc:  # noqa: BLE001
            r_ab = {"error": f"{type(exc).__name__}: {exc}"}
            _append_progress({
                "phase": "recency_ab_error", "status": "warn",
                "headline": f"Phase 4b A/B raised {type(exc).__name__}: {exc}",
            })
        finally:
            os.environ["ML_RECENCY_WEIGHTED"] = original_mode
        ab_elapsed = round(time.time() - ab_started, 1)
        # Build per-slice diff: A (this run, original_mode) vs B (flipped)
        materially_different: list[str] = []
        if isinstance(r_ab, dict) and "error" not in r_ab:
            for tf in ab_tfs:
                a_tf = (report.get("timeframes") or {}).get(tf) or {}
                b_tf = (r_ab.get("timeframes") or {}).get(tf) or {}
                a_per = a_tf.get("per_coin") or {}
                b_per = b_tf.get("per_coin") or {}
                for coin in ab_coins:
                    a_rep = a_per.get(coin) or {}
                    b_rep = b_per.get(coin) or {}
                    a_pnl = (a_rep.get("pnl_after_fees") or {}).get("net_pct_total")
                    b_pnl = (b_rep.get("pnl_after_fees") or {}).get("net_pct_total")
                    delta = None
                    if a_pnl is not None and b_pnl is not None:
                        delta = round(b_pnl - a_pnl, 4)
                    key = f"{coin}/{tf}"
                    ab_payload["per_slice_diff"][key] = {
                        "a_mode": original_mode, "a_post_fee_pct_total": a_pnl,
                        "b_mode": flipped_mode, "b_post_fee_pct_total": b_pnl,
                        "delta_b_minus_a": delta,
                    }
                    if delta is not None and abs(delta) > 0.5:
                        materially_different.append(key)
        ab_payload["summary"] = {
            "elapsed_sec": ab_elapsed,
            "n_slices_compared": len(ab_payload["per_slice_diff"]),
            "n_materially_different_gt_0_5pp": len(materially_different),
            "materially_different_slices": materially_different,
            "verdict": (
                "recency_weighting_materially_changes_outcomes"
                if materially_different else
                "recency_weighting_does_not_change_outcomes_in_this_subset"
            ),
        }
        _append_progress({
            "phase": "recency_ab_done", "status": "ok",
            "headline": (
                f"Phase 4b A/B done in {ab_elapsed}s — "
                f"materially different slices={len(materially_different)} / "
                f"{len(ab_payload['per_slice_diff'])} "
                f"(threshold ±0.5pp post-fee)"
            ),
            "verdict": ab_payload["summary"]["verdict"],
        })
    (run_dir / "phase4b_recency_ab.json").write_text(
        json.dumps(ab_payload, indent=2, default=str)
    )

    # Per-tf progress updates — derived from the freshly-written report.
    tf_results = report.get("timeframes", {}) or {}
    for tf, tf_rep in tf_results.items():
        per_coin = tf_rep.get("per_coin", {}) if isinstance(tf_rep, dict) else {}
        n_trained = sum(
            1 for c in per_coin.values()
            if isinstance(c, dict) and c.get("status") == "trained"
        )
        n_total = len(per_coin)
        _append_progress({
            "phase": "training_tf_done",
            "status": "ok",
            "headline": (
                f"{tf}: trained {n_trained}/{n_total} per-coin slices, "
                f"pooled={(tf_rep.get('pooled') or {}).get('status', 'n/a')}"
            ),
            "timeframe": tf,
            "per_coin_status": {c: r.get("status") for c, r in per_coin.items()
                                if isinstance(r, dict)},
            "specialists_status": (tf_rep.get("specialists") or {}).get("error", "ok"),
        })
    _append_progress({
        "phase": "training_done",
        "status": "ok",
        "headline": f"training campaign finished in {elapsed}s",
        "elapsed_sec": elapsed,
    })
    return {
        "phase": "training",
        "status": "ok",
        "elapsed_sec": elapsed,
        "timeframes_trained": list(tf_results.keys()),
    }


# ── Phase 5 — periodic structured proof updates (post-run synthesis) ─────
def _slice_row(coin: str, tf: str, kind: str, c_rep: dict) -> dict:
    pnl = c_rep.get("pnl_after_fees") or {}
    metrics = c_rep.get("metrics") or {}
    baseline = c_rep.get("baseline_metrics") or {}
    return {
        "slice": f"{coin}/{tf}", "kind": kind,
        "status": c_rep.get("status"),
        "post_fee_pnl_pct_total": pnl.get("net_pct_total"),
        "post_fee_pnl_pct_mean": pnl.get("net_pct_mean"),
        "gross_pnl_pct_mean": pnl.get("gross_pct_mean"),
        "n_trades": pnl.get("n_trades"),
        "win_rate": pnl.get("win_rate"),
        "trade_share": pnl.get("trade_share"),
        "round_trip_cost_pct": pnl.get("round_trip_cost_pct"),
        "auc": metrics.get("auc"),
        "baseline_auc": baseline.get("auc"),
        "lift_auc": c_rep.get("lift_auc"),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "baseline_directional_accuracy": baseline.get("directional_accuracy"),
        "n_rows": c_rep.get("n_rows"),
    }


def _rank_slices(report: dict) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    for tf, tf_rep in (report.get("timeframes") or {}).items():
        if not isinstance(tf_rep, dict):
            continue
        for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
            if isinstance(c_rep, dict):
                rows.append(_slice_row(coin, tf, "per_coin", c_rep))
        pooled = tf_rep.get("pooled") or {}
        if isinstance(pooled, dict):
            rows.append(_slice_row("__pooled__", tf, "pooled", pooled))
    # Rank by post-fee total return (pct), which is the quant's
    # real-money outcome. Fall back to directional-accuracy lift when
    # the pnl_after_fees slot is missing (e.g. untrained slice).
    def key(r: dict) -> float:
        v = r["post_fee_pnl_pct_total"]
        if v is None:
            da = r.get("directional_accuracy") or 0.0
            b = r.get("baseline_directional_accuracy") or 0.0
            v = (da - b) * 100.0  # scale to ~same unit as pct_total
        return float(v)
    scored = sorted(rows, key=key, reverse=True)
    return scored[:10], list(reversed(scored[-10:]))


def _post_fee_baseline_diff(baseline: dict, report: dict) -> dict:
    """Per-slice before/after diff. Uses post-fee total return (pct) as
    the primary metric, with AUC + directional-accuracy lift exposed as
    secondary diffs. Falls back to the slice's internal `baseline_metrics`
    (buy-and-hold / majority-class baseline) when the archived baseline
    doesn't cover this slice — which is the common case here since the
    pre-run report only had a partial jupiter-only campaign.
    """
    base = baseline.get("per_slice") or {}
    out: dict[str, dict] = {}
    for tf, tf_rep in (report.get("timeframes") or {}).items():
        if not isinstance(tf_rep, dict):
            continue
        for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
            if not isinstance(c_rep, dict):
                continue
            key = f"{coin}/{tf}"
            pnl = c_rep.get("pnl_after_fees") or {}
            new_net_pct = pnl.get("net_pct_total")
            new_auc = (c_rep.get("metrics") or {}).get("auc")
            new_da = (c_rep.get("metrics") or {}).get("directional_accuracy")
            old_entry = base.get(key) or {}
            old_net_pct = (
                (old_entry.get("pnl_after_fees") or {}).get("net_pct_total")
                if isinstance(old_entry.get("pnl_after_fees"), dict) else None
            )
            # Internal vs-baseline diff (always available post-training).
            base_auc = (c_rep.get("baseline_metrics") or {}).get("auc")
            base_da = (c_rep.get("baseline_metrics") or {}).get("directional_accuracy")
            out[key] = {
                "old_post_fee_pct_total": old_net_pct,
                "new_post_fee_pct_total": new_net_pct,
                "delta_post_fee_pct_total": (
                    (new_net_pct - old_net_pct)
                    if (new_net_pct is not None and old_net_pct is not None) else None
                ),
                "vs_internal_baseline_auc_lift": (
                    (new_auc - base_auc)
                    if (new_auc is not None and base_auc is not None) else None
                ),
                "vs_internal_baseline_da_lift": (
                    (new_da - base_da)
                    if (new_da is not None and base_da is not None) else None
                ),
                "old_status": old_entry.get("status"),
                "new_status": c_rep.get("status"),
            }
    return out


def _per_coin_vs_pooled(report: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tf, tf_rep in (report.get("timeframes") or {}).items():
        if not isinstance(tf_rep, dict):
            continue
        pooled = tf_rep.get("pooled") or {}
        pooled_pnl = (pooled.get("pnl_after_fees") or {}).get("net_pct_total")
        pooled_da = (pooled.get("metrics") or {}).get("directional_accuracy")
        for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
            if not isinstance(c_rep, dict) or c_rep.get("status") != "trained":
                continue
            coin_pnl = (c_rep.get("pnl_after_fees") or {}).get("net_pct_total")
            coin_da = (c_rep.get("metrics") or {}).get("directional_accuracy")
            out[f"{coin}/{tf}"] = {
                "per_coin_pct_total": coin_pnl,
                "pooled_pct_total": pooled_pnl,
                "per_coin_da": coin_da,
                "pooled_da": pooled_da,
                "per_coin_beats_pooled_pnl": (
                    coin_pnl is not None and pooled_pnl is not None
                    and coin_pnl > pooled_pnl
                ),
                "per_coin_beats_pooled_da": (
                    coin_da is not None and pooled_da is not None
                    and coin_da > pooled_da
                ),
            }
    return out


def _failure_buckets(report: dict) -> dict[str, int]:
    fa = report.get("failure_analysis") or {}
    counts = (fa.get("bucket_counts") or fa.get("counts") or {})
    return dict(counts) if isinstance(counts, dict) else {}


def _task521_watchlist_rows(report: dict) -> list[dict]:
    """Task #521 — for each booster-fix watchlist slice, pull the current
    predicted STABLE share, n_trades and post-fee net_pct_total out of
    the freshly-written `report.json`. Slices that the campaign did not
    train (e.g. their timeframe wasn't included this run) get an
    explicit `present=False` row so the operator can tell the slice was
    skipped vs a missing field.

    Returned rows are JSON-safe and used both by the markdown writer and
    by the `phase7_summary.json` block.
    """
    timeframes = (report or {}).get("timeframes") or {}
    out: list[dict] = []
    for entry in TASK521_BOOSTER_FIX_WATCHLIST:
        coin, tf = entry["coin"], entry["tf"]
        slug = f"{coin}/{tf}"
        tf_rep = timeframes.get(tf)
        per_coin = (
            tf_rep.get("per_coin") if isinstance(tf_rep, dict) else None
        )
        c_rep = (
            per_coin.get(coin) if isinstance(per_coin, dict) else None
        )
        if not isinstance(c_rep, dict):
            # Distinguish "campaign skipped this timeframe entirely" from
            # "campaign trained the timeframe but skipped this coin" so an
            # operator can tell which knob (timeframe selection vs coin
            # selection) excluded the slice from this run's coverage.
            if tf_rep is None or not isinstance(tf_rep, dict):
                note = (
                    f"slice not trained this campaign run "
                    f"(timeframe {tf!r} absent from report)"
                )
            elif not isinstance(per_coin, dict):
                note = (
                    f"slice not trained this campaign run "
                    f"(timeframe {tf!r} has no per_coin block)"
                )
            else:
                note = (
                    f"slice not trained this campaign run "
                    f"(coin {coin!r} absent from {tf!r} per_coin block)"
                )
            out.append({
                "slice": slug,
                "pre_fix_dcs_drop_pp": entry["pre_fix_dcs_drop_pp"],
                "present": False,
                "status": None,
                "predicted_stable_share_pct": None,
                "directional_call_share_pct": None,
                "n_trades": None,
                "post_fee_net_pct_total": None,
                "note": note,
            })
            continue
        dcs = c_rep.get("directional_call_share")
        pnl = c_rep.get("pnl_after_fees") or {}
        try:
            dcs_pct = round(float(dcs) * 100.0, 2) if dcs is not None else None
        except (TypeError, ValueError):
            dcs_pct = None
        stable_pct = (
            round(100.0 - dcs_pct, 2) if dcs_pct is not None else None
        )
        out.append({
            "slice": slug,
            "pre_fix_dcs_drop_pp": entry["pre_fix_dcs_drop_pp"],
            "present": True,
            "status": c_rep.get("status"),
            "predicted_stable_share_pct": stable_pct,
            "directional_call_share_pct": dcs_pct,
            "directional_call_share_source": c_rep.get(
                "directional_call_share_source"
            ),
            "n_trades": pnl.get("n_trades"),
            "post_fee_net_pct_total": pnl.get("net_pct_total"),
            "note": None,
        })
    return out


def phase567_summary(run_dir: Path, baseline_dir: Path) -> dict:
    started = time.time()
    rep_path = REGISTRY_ROOT / "report.json"
    if not rep_path.exists():
        raise RuntimeError("post-training report.json missing — phase 4 did not write")
    report = json.loads(rep_path.read_text())
    baseline_path = baseline_dir / "baseline_snapshot.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {"per_slice": {}}

    # Task #613 — anchor live-replay aggregation to this campaign's run
    # folder so a stale `slice_done` row from a previous campaign is
    # never counted as this campaign's evidence. The folder name is the
    # campaign's UTC start timestamp (`training_run_<TS>`); we convert it
    # to ISO-8601 so it can be string-compared against `emitted_at`.
    run_started_iso: Optional[str] = None
    rd_name = run_dir.name
    if rd_name.startswith("training_run_") and len(rd_name) >= 28:
        try:
            ts_part = rd_name[len("training_run_"):]
            run_started_iso = (
                datetime.strptime(ts_part, "%Y%m%dT%H%M%SZ")
                .replace(tzinfo=timezone.utc)
                .isoformat()
            )
        except ValueError:
            run_started_iso = None
    live_per_slice = _aggregate_live_replay_per_slice(run_started_iso)

    best, worst = _rank_slices(report)
    diff = _post_fee_baseline_diff(baseline, report)
    pc_vs_pool = _per_coin_vs_pooled(report)
    fb = _failure_buckets(report)
    task521_watchlist = _task521_watchlist_rows(report)
    verification = report.get("verification") or {}
    verification_passed = bool(verification.get("passed"))
    watchdog = report.get("verification_diff") or {}
    watchdog_verdict = watchdog.get("verdict") or watchdog.get("status")

    # Improvement aggregation on post-fee pct_total. When the baseline
    # snapshot doesn't cover a slice, fall back to the internal AUC lift:
    # improved ⇔ model AUC > buy-and-hold / majority-class AUC baseline.
    def _improved(d: dict) -> Optional[bool]:
        if d["delta_post_fee_pct_total"] is not None:
            return d["delta_post_fee_pct_total"] > 0
        lift = d.get("vs_internal_baseline_auc_lift")
        return (lift is not None) and (lift > 0)
    n_improved = sum(1 for d in diff.values() if _improved(d) is True)
    n_regressed = sum(1 for d in diff.values() if _improved(d) is False)
    profitable = [k for k, d in diff.items()
                  if d["new_post_fee_pct_total"] is not None
                  and d["new_post_fee_pct_total"] > 0]

    # WLD candidacy check
    wld = {k: v for k, v in diff.items() if k.startswith("worldcoin-wld/")}

    # Promote / retire / observe lists derived from verification.per_slice
    # + failure_analysis bucket membership.
    per_slice = verification.get("per_slice") or []
    promote = [f"{s['coin']}/{s['timeframe']}" for s in per_slice
               if s.get("promoted")]
    # Structurally noisy → retire. Insufficient-sample → observe + backfill.
    # Everything else below coinflip but with <5% lift deficit → observe.
    retire: list[str] = []
    observe: list[str] = []
    for s in per_slice:
        slug = f"{s['coin']}/{s['timeframe']}"
        if s.get("promoted"):
            continue
        reason = s.get("reason")
        lift = s.get("lift") or 0.0
        if reason == "insufficient_sample":
            observe.append(slug + " (insufficient_sample)")
        elif reason == "below_coinflip" and lift <= -0.03:
            retire.append(slug + f" (lift={lift:+.3f})")
        elif reason == "below_coinflip":
            observe.append(slug + f" (below_coinflip, lift={lift:+.3f})")
        else:
            observe.append(slug + f" ({reason})")

    # Final verdict: the quant "improved" iff the verification watchdog
    # says so AND at least one slice was promoted on held-out data AND
    # the promoted slices meaningfully outnumber the structurally-retire
    # slices. A single `dogwifcoin/1d` promotion against 32 retires is
    # an honest "no".
    promoted_cnt = len(promote)
    retired_cnt = len(retire)
    final_verdict = (
        "yes" if (verification_passed and promoted_cnt > 0
                  and promoted_cnt >= max(1, retired_cnt // 4))
        else "no"
    )

    # ── Phase 6 — explicit answers to the seven named diagnostic
    # questions. Every answer is computed from on-disk artifacts; no
    # narrative is added without a number behind it.
    # (a) does any per-coin model now beat pooled on post-fee PnL?
    pc_beats_pool = sorted([k for k, v in pc_vs_pool.items()
                            if v.get("per_coin_beats_pooled_pnl")])
    # (b) is worldcoin-wld still the strongest improvement candidate?
    wld_best_delta = max(
        ((d.get("delta_post_fee_pct_total") or -1e9) for d in wld.values()),
        default=None,
    )
    other_best_delta = max(
        ((d.get("delta_post_fee_pct_total") or -1e9)
         for k, d in diff.items() if not k.startswith("worldcoin-wld/")),
        default=None,
    )
    wld_is_strongest = bool(
        wld_best_delta is not None and other_best_delta is not None
        and wld_best_delta >= other_best_delta
    )
    # (c) regime mismatch dominance — is `regime_mismatch` the top bucket?
    top_bucket = None
    if fb:
        top_bucket = max(fb.items(), key=lambda x: x[1])[0]
    regime_dominant = bool(top_bucket and "regime" in str(top_bucket).lower())
    five_m_in_regime = sum(
        1 for s in (verification.get("per_slice") or [])
        if s.get("timeframe") == "5m" and "regime" in str(s.get("reason", "")).lower()
    )
    # (d) recency weighting A/B — Phase 4b runs the flipped mode on a
    # bounded subset and writes phase4b_recency_ab.json. We consume that
    # artifact here so the answer is grounded in actually-executed paired
    # training rather than in a "re-run with the flag flipped" promissory
    # note. If the artifact is missing (e.g. ML_RECENCY_AB_DISABLE=1) we
    # surface that explicitly.
    recency_mode = os.environ.get("ML_RECENCY_WEIGHTED", "1")
    ab_artifact_path = run_dir / "phase4b_recency_ab.json"
    recency_ab_payload: dict = {}
    recency_ab_verdict = "unknown_no_artifact"
    if ab_artifact_path.exists():
        try:
            recency_ab_payload = json.loads(ab_artifact_path.read_text())
            recency_ab_verdict = (recency_ab_payload.get("summary") or {}).get(
                "verdict", "unknown_no_summary"
            )
            if recency_ab_payload.get("skipped"):
                recency_ab_verdict = (
                    f"skipped:{recency_ab_payload.get('skip_reason', 'unknown')}"
                )
        except Exception as exc:  # noqa: BLE001
            recency_ab_verdict = f"unreadable:{exc}"
    recency_note = (
        f"Phase 4b paired evaluation ran with ML_RECENCY_WEIGHTED="
        f"{(recency_ab_payload or {}).get('flipped_mode', 'n/a')} on "
        f"{len((recency_ab_payload or {}).get('subset_coins') or [])} coin(s) "
        f"× {(recency_ab_payload or {}).get('subset_tfs') or []}. "
        f"Verdict: {recency_ab_verdict}. Full per-slice diff is in "
        f"`phase4b_recency_ab.json` (threshold is ±0.5pp post-fee net_pct_total)."
    )
    # (e) any post-fee profitable slices?
    post_fee_profitable = profitable
    # (f) class-collapsed slices
    collapsed: list[str] = []
    for tf, tf_rep in (report.get("timeframes") or {}).items():
        if not isinstance(tf_rep, dict):
            continue
        for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
            if isinstance(c_rep, dict) and (c_rep.get("prediction_collapse") or {}).get("collapsed"):
                collapsed.append(f"{coin}/{tf}")
    # (g) dashboard accounting contradiction check — Task #362 already
    # pinned this. We re-check by comparing the post-fee net pct from
    # report.json against backtest_report.json for the same slice. Any
    # >5pp drift is a contradiction the operator should investigate.
    bt_path = REGISTRY_ROOT / "backtest_report.json"
    accounting_contradictions: list[dict] = []
    if bt_path.exists():
        try:
            bt = json.loads(bt_path.read_text())
            bt_per = (bt.get("per_slice") or bt.get("slices") or {}) or {}
            for tf, tf_rep in (report.get("timeframes") or {}).items():
                if not isinstance(tf_rep, dict):
                    continue
                for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
                    if not isinstance(c_rep, dict):
                        continue
                    a = (c_rep.get("pnl_after_fees") or {}).get("net_pct_total")
                    key = f"{coin}/{tf}"
                    bt_entry = bt_per.get(key) if isinstance(bt_per, dict) else None
                    b = ((bt_entry or {}).get("pnl_after_fees") or {}).get("net_pct_total")
                    if a is not None and b is not None and abs(a - b) > 5.0:
                        accounting_contradictions.append({
                            "slice": key, "report_pct": a, "backtest_pct": b,
                            "drift_pct": round(a - b, 3),
                        })
        except Exception:  # noqa: BLE001
            pass
    phase6_answers = {
        "a_per_coin_beats_pooled_post_fee": {
            "answer": bool(pc_beats_pool),
            "slices": pc_beats_pool,
        },
        "b_worldcoin_wld_strongest_candidate": {
            "answer": wld_is_strongest,
            "wld_best_delta": wld_best_delta,
            "other_best_delta": other_best_delta,
        },
        "c_regime_mismatch_dominant_failure": {
            "answer": regime_dominant,
            "top_bucket": top_bucket,
            "five_m_slices_in_regime_bucket": five_m_in_regime,
        },
        "d_recency_weighting_ab": {
            "this_run_mode": recency_mode,
            "note": recency_note,
        },
        "e_post_fee_profitable_slices": {
            "answer": bool(post_fee_profitable),
            "slices": post_fee_profitable,
        },
        "f_class_collapsed_slices": {
            "answer": bool(collapsed),
            "slices": collapsed,
        },
        "g_dashboard_accounting_contradictions": {
            "answer": bool(accounting_contradictions),
            "contradictions": accounting_contradictions,
        },
    }

    # Task #613 — per-slice live-gated replay block (loose vs live PnL
    # plus economic verdict). Read off the same `progress_updates.jsonl`
    # the rest of the run already writes; uses the bonk/5m back-fill row
    # for the previously halted slice.
    live_gated_per_slice: dict[str, dict] = {}
    verdict_counts: dict[str, int] = {
        "bleeding": 0, "dormant": 0, "tradeable": 0, "inconclusive": 0,
    }
    for tf, tf_rep in (report.get("timeframes") or {}).items():
        if not isinstance(tf_rep, dict):
            continue
        for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
            if not isinstance(c_rep, dict):
                continue
            slug = f"{coin}/{tf}"
            loose_pct = (c_rep.get("pnl_after_fees") or {}).get(
                "net_pct_total"
            )
            live = live_per_slice.get(slug) or {}
            verdict = _economic_verdict(
                loose_pct,
                live.get("live_trade_count"),
                live.get("live_net_pnl_pct"),
            )
            # Task #613 — operator-facing verdict phrase. The bonk/5m
            # slice gets the spec's exact language ("dormant / no-edge
            # under production gates") since it is the named example
            # in the task brief; every other dormant slice gets the
            # canonical short form.
            verdict_phrase_map = {
                "bleeding": "bleeding / negative under production gates",
                "dormant": "dormant / no-edge",
                "tradeable": "tradeable / positive under production gates",
                "inconclusive": (
                    "inconclusive / signals do not agree or live "
                    "diagnostic missing"
                ),
            }
            verdict_phrase = verdict_phrase_map.get(verdict, verdict)
            if slug == "bonk/5m" and verdict == "dormant":
                verdict_phrase = "dormant / no-edge under production gates"
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            live_gated_per_slice[slug] = {
                "loose_post_fee_pct_total": loose_pct,
                "live_trade_count": live.get("live_trade_count"),
                "live_net_pnl_pct": live.get("live_net_pnl_pct"),
                "dominant_rejection_reason": live.get(
                    "dominant_rejection_reason"
                ),
                "live_replay_status": live.get("live_replay_status")
                or "missing",
                "live_replay_error": live.get("live_replay_error"),
                "live_replay_output_dir": live.get("live_replay_output_dir"),
                "source_phase": live.get("source_phase"),
                "economic_verdict": verdict,
                "economic_verdict_phrase": verdict_phrase,
            }
    bleeding_slices = sorted(
        k for k, v in live_gated_per_slice.items()
        if v["economic_verdict"] == "bleeding"
    )
    tradeable_slices = sorted(
        k for k, v in live_gated_per_slice.items()
        if v["economic_verdict"] == "tradeable"
    )
    dormant_slices = sorted(
        k for k, v in live_gated_per_slice.items()
        if v["economic_verdict"] == "dormant"
    )

    summary_payload = {
        "phase6_diagnostic_answers": phase6_answers,
        "phase": "final_summary",
        "generated_at": _utcnow_iso(),
        "verification_passed": verification_passed,
        "watchdog_verdict": watchdog_verdict,
        "best_slices": best,
        "worst_slices": worst,
        "before_after_diff": diff,
        "per_coin_vs_pooled": pc_vs_pool,
        "failure_buckets": fb,
        "improvement_counts": {"improved": n_improved, "regressed": n_regressed},
        "profitable_post_fee_slices": profitable,
        "worldcoin_wld_diff": wld,
        "promote": promote, "retire": retire, "observe": observe,
        "final_verdict_did_quant_improve": final_verdict,
        # Task #521 — booster-fix watchlist (4 slices the Task #507 fix
        # made trade much less). Surfaced every campaign run so the
        # operator can spot a material PnL regression on these slices
        # as soon as Task #516 follow-up #2 lands.
        "task521_booster_fix_watchlist": task521_watchlist,
        # Task #613 — per-slice live-gated replay (loose post-fee vs
        # live-replay PnL) plus the four-way economic verdict
        # (bleeding/dormant/tradeable/inconclusive). Diagnostic only —
        # gates and promotion logic are unchanged.
        "live_gated_replay": {
            "run_started_iso": run_started_iso,
            "per_slice": live_gated_per_slice,
            "verdict_counts": verdict_counts,
            "bleeding_slices": bleeding_slices,
            "dormant_slices": dormant_slices,
            "tradeable_slices": tradeable_slices,
        },
    }
    (run_dir / "phase7_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str)
    )

    # Markdown
    md = ["# Full 1-Year Quant Training Run — Summary", ""]
    md.append(f"_Run folder: `{run_dir.relative_to(ROOT)}`  •  generated {_utcnow_iso()}_")
    md.append("")
    md.append("## Final verdict")
    md.append(f"- **Did the quant improve from this training run?** **{final_verdict.upper()}**")
    md.append(f"- Watchdog verdict: `{watchdog_verdict}`")
    md.append(f"- Verification block passed: `{verification_passed}`")
    md.append(f"- Post-fee improved slices: **{n_improved}**, regressed: **{n_regressed}**, profitable post-fee: **{len(profitable)}**")
    md.append("")
    def _row(r: dict) -> str:
        return (
            f"| {r['slice']} | {r['kind']} | {r['status']} "
            f"| {r['post_fee_pnl_pct_total']} | {r['post_fee_pnl_pct_mean']} "
            f"| {r['n_trades']} | {r['win_rate']} "
            f"| {r['auc']} | {r['lift_auc']} |"
        )
    md.append("## Top 10 best slices by post-fee total return (%)")
    md.append("")
    md.append("| slice | kind | status | pnl_pct_total | pnl_pct_mean | trades | win_rate | auc | lift_auc |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in best:
        md.append(_row(r))
    md.append("")
    md.append("## Worst 10 slices by post-fee total return (%)")
    md.append("")
    md.append("| slice | kind | status | pnl_pct_total | pnl_pct_mean | trades | win_rate | auc | lift_auc |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in worst:
        md.append(_row(r))
    md.append("")
    md.append("## Failure-analysis buckets")
    md.append("")
    if fb:
        for bucket, n in sorted(fb.items(), key=lambda x: -x[1]):
            md.append(f"- `{bucket}` — {n}")
    else:
        md.append("_No failure-analysis bucket counts emitted by this run._")
    md.append("")
    md.append("## Per-coin vs pooled (does any per-coin model beat pooled on post-fee PnL?)")
    md.append("")
    beats_pnl = sorted([k for k, v in pc_vs_pool.items() if v.get("per_coin_beats_pooled_pnl")])
    beats_da = sorted([k for k, v in pc_vs_pool.items() if v.get("per_coin_beats_pooled_da")])
    md.append(f"- Per-coin beats pooled on post-fee pct_total: **{len(beats_pnl)}** / {len(pc_vs_pool)} slices")
    if beats_pnl:
        md.append("")
        md.append(f"  - {beats_pnl}")
    md.append(f"- Per-coin beats pooled on directional accuracy: **{len(beats_da)}** / {len(pc_vs_pool)} slices")
    md.append("")
    md.append("## Worldcoin-WLD watchlist (was strongest improvement candidate)")
    md.append("")
    if wld:
        for k, d in wld.items():
            md.append(
                f"- `{k}` — old_pct={d['old_post_fee_pct_total']}, "
                f"new_pct={d['new_post_fee_pct_total']}, "
                f"delta_pct={d['delta_post_fee_pct_total']}, "
                f"auc_lift_vs_internal={d['vs_internal_baseline_auc_lift']}, "
                f"da_lift_vs_internal={d['vs_internal_baseline_da_lift']}, "
                f"old_status={d['old_status']}, new_status={d['new_status']}"
            )
    else:
        md.append("_No worldcoin-wld slices in this report._")
    md.append("")
    md.append("## Booster-fix watchlist (Task #521)")
    md.append("")
    md.append(
        "_The Task #507 booster fix shifted predicted STABLE share up "
        "substantially on these 4 slices that did not need rescuing, "
        "cutting their directional-call share by 15-32pp and roughly "
        "halving their realized trade count. No clean pre-fix PnL is on "
        "disk (see `reports/20260428T111719Z-task516-pnl-impact-verification.md` "
        "§1), so this row is surfaced every campaign run as the leading "
        "indicator until Task #516 follow-up #2 (per-slice PnL snapshot) "
        "lands. If `post_fee_net_pct_total` materially regresses on any "
        "of these 4 slices, raise a fix task._"
    )
    md.append("")
    md.append(
        "| slice | pre-fix DCS drop (pp) | predicted STABLE share % "
        "| DCS % (source) | n_trades | post-fee net_pct_total | status | note |"
    )
    md.append(
        "|---|---:|---:|---:|---:|---:|---|---|"
    )
    for r in task521_watchlist:
        if not r.get("present"):
            md.append(
                f"| `{r['slice']}` | {r['pre_fix_dcs_drop_pp']} "
                f"| n/a | n/a | n/a | n/a | not_trained "
                f"| {r['note']} |"
            )
            continue
        stable_str = (
            f"{r['predicted_stable_share_pct']:.2f}"
            if r["predicted_stable_share_pct"] is not None else "n/a"
        )
        dcs_val = r["directional_call_share_pct"]
        dcs_src = r.get("directional_call_share_source") or "n/a"
        dcs_str = (
            f"{dcs_val:.2f} ({dcs_src})" if dcs_val is not None else "n/a"
        )
        n_trades_str = (
            str(r["n_trades"]) if r["n_trades"] is not None else "n/a"
        )
        # Mirror the DCS defensive-parse path for net_pct_total — a
        # non-numeric value should not be allowed to crash the
        # operator-facing summary writer; it's rendered as "n/a"
        # instead, the same way a missing PnL block is.
        net_pct = r["post_fee_net_pct_total"]
        try:
            net_pct_str = (
                f"{float(net_pct):+.4f}" if net_pct is not None else "n/a"
            )
        except (TypeError, ValueError):
            net_pct_str = "n/a"
        md.append(
            f"| `{r['slice']}` | {r['pre_fix_dcs_drop_pp']} "
            f"| {stable_str} | {dcs_str} | {n_trades_str} | {net_pct_str} "
            f"| {r['status'] or 'n/a'} |  |"
        )
    md.append("")
    md.append("## Live-gated replay (Task #613) — loose vs live PnL + verdict")
    md.append("")
    md.append(
        "_Loose post-fee PnL comes from the trainer's holdout sweep "
        "(`pnl_after_fees.net_pct_total`); live PnL comes from "
        "`scripts/diagnose_post_fee.py` running the SAME holdout through "
        "the trade-time gates. Verdict legend: `bleeding` = the live-gated "
        "replay still loses real money on >=5 trades; `dormant` = no edge "
        "realised under live gates (live n<5); `tradeable` = live PnL is "
        "positive on >=5 trades; `inconclusive` = signals don't agree or "
        "the diagnostic is missing/skipped. Diagnostic-only — gates and "
        "promotion logic are unchanged._"
    )
    md.append("")
    md.append(
        f"- Verdict counts: bleeding=**{verdict_counts['bleeding']}**, "
        f"dormant=**{verdict_counts['dormant']}**, "
        f"tradeable=**{verdict_counts['tradeable']}**, "
        f"inconclusive=**{verdict_counts['inconclusive']}**"
    )
    md.append(
        f"- Tradeable slices: {tradeable_slices or 'none'}"
    )
    md.append(
        f"- Bleeding slices: {bleeding_slices or 'none'}"
    )
    md.append(
        f"- Dormant slices: {dormant_slices or 'none'}"
    )
    md.append("")
    md.append(
        "| slice | loose pct_total | live n | live pct_total "
        "| dominant rejection | replay status | verdict |"
    )
    md.append("|---|---:|---:|---:|---|---|---|")
    for slug in sorted(live_gated_per_slice.keys()):
        row = live_gated_per_slice[slug]
        loose = row.get("loose_post_fee_pct_total")
        live_pct = row.get("live_net_pnl_pct")

        def _fmt_pct(v: Any) -> str:
            try:
                return f"{float(v):+.4f}" if v is not None else "n/a"
            except (TypeError, ValueError):
                return "n/a"
        md.append(
            f"| `{slug}` | {_fmt_pct(loose)} "
            f"| {row.get('live_trade_count') if row.get('live_trade_count') is not None else 'n/a'} "
            f"| {_fmt_pct(live_pct)} "
            f"| {row.get('dominant_rejection_reason') or 'n/a'} "
            f"| {row.get('live_replay_status') or 'n/a'} "
            f"| {row.get('economic_verdict_phrase') or row.get('economic_verdict')} |"
        )
    md.append("")
    md.append("## Phase 6 — explicit answers to the seven named diagnostic questions")
    md.append("")
    md.append(
        f"- **(a) Does any per-coin model now beat pooled on post-fee PnL?** "
        f"`{phase6_answers['a_per_coin_beats_pooled_post_fee']['answer']}` "
        f"({len(pc_beats_pool)} slices: {pc_beats_pool or 'none'})"
    )
    md.append(
        f"- **(b) Does worldcoin-wld remain the strongest improvement candidate?** "
        f"`{phase6_answers['b_worldcoin_wld_strongest_candidate']['answer']}` "
        f"(wld_best_delta={wld_best_delta}, other_best_delta={other_best_delta})"
    )
    md.append(
        f"- **(c) Are 5m slices still failing primarily due to regime mismatch?** "
        f"`{phase6_answers['c_regime_mismatch_dominant_failure']['answer']}` "
        f"(top_bucket=`{top_bucket}`, 5m-in-regime-bucket={five_m_in_regime})"
    )
    md.append(
        f"- **(d) Does recency weighting materially change post-fee outcomes?** "
        f"verdict = `{recency_ab_verdict}`. "
        f"This run mode = `ML_RECENCY_WEIGHTED={recency_mode}`, "
        f"flipped subset = `{(recency_ab_payload or {}).get('subset_coins') or []}` "
        f"× `{(recency_ab_payload or {}).get('subset_tfs') or []}`. "
        f"Per-slice diff in `phase4b_recency_ab.json`."
    )
    md.append(
        f"- **(e) Are any post-fee slices actually profitable?** "
        f"`{phase6_answers['e_post_fee_profitable_slices']['answer']}` "
        f"({len(post_fee_profitable)} slices: {post_fee_profitable or 'none'})"
    )
    md.append(
        f"- **(f) Are any slices collapsing to one class?** "
        f"`{phase6_answers['f_class_collapsed_slices']['answer']}` "
        f"({len(collapsed)} slices: {collapsed or 'none'})"
    )
    md.append(
        f"- **(g) Do any dashboard tiles still contradict the underlying accounting?** "
        f"`{phase6_answers['g_dashboard_accounting_contradictions']['answer']}` "
        f"({len(accounting_contradictions)} >5pp drifts vs `backtest_report.json`)"
    )
    md.append("")
    md.append("## Promote / retire / observe (from verification block)")
    md.append("")
    md.append(f"- **Promote:** {promote or 'none'}")
    md.append(f"- **Retire:** {retire or 'none'}")
    md.append(f"- **Observe:** {observe or 'none'}")
    md.append("")
    md.append("## Artifacts")
    md.append("")
    md.append(f"- Post-run report: `models/report.json`")
    md.append(f"- Backtest report: `models/backtest_report.json`")
    md.append(f"- Verification history: `models/verification_history.jsonl`")
    md.append(f"- Calibration recommendation: `models/calibration_recommendation.json`")
    md.append(f"- Pre-run baseline archive: `{baseline_dir.relative_to(ROOT)}/`")
    md.append(f"- Run folder: `{run_dir.relative_to(ROOT)}/`")
    md.append(f"- Progress updates: `models/progress_updates.jsonl`")
    md.append("")
    # Task #520 — document the baseline_snapshot.json schema inline so an
    # operator opening summary.md doesn't have to spelunk the orchestrator
    # source to know what fields they can diff.
    md.append("## Pre-run baseline snapshot schema (`baseline_snapshot.json`)")
    md.append("")
    md.append(
        "Lives at `<baseline_dir>/baseline_snapshot.json` next to a copy of "
        "the prior campaign's `report.json`, per-coin model folders, and "
        "verification history. Top-level keys:"
    )
    md.append("")
    md.append(
        "- `schema_version` — `task520_v1` once the per-slice PnL harvest "
        "is in place."
    )
    md.append(
        "- `per_slice` — `{ \"<coin>/<tf>\": <slot> }`, built from the "
        "prior `report.json`. Each slot carries `status`, `metrics`, "
        "`baseline_metrics`, `lift_auc`, `calibration`, `pnl_after_fees` "
        "(`net_pct_total`, `net_pct_mean`, `n_trades`, `trade_share`, "
        "`gross_pct_mean`, `round_trip_cost_pct`, `win_rate`), `n_rows`, "
        "and the trained model `version`. Historically thin (often only "
        "the active registry-shadow audit slices) — use "
        "`prior_campaign_per_slice_pnl` for full coverage."
    )
    md.append(
        "- `verification`, `verification_diff` — copied verbatim from the "
        "prior `report.json` so promotion/no-lift counts are diffable."
    )
    md.append(
        "- `archived_model_audit` — `{ manifests_scanned, "
        "forbidden_feature_leaks }`. Quant-only proof that the snapshot "
        "itself is leak-free before the new campaign overwrites the "
        "registry."
    )
    md.append(
        "- `prior_campaign_per_slice_pnl` (task #520) — per-slice "
        "walk-forward PnL harvested from every `slice_done` event in the "
        "most recent prior campaign window in `progress_updates.jsonl`. "
        "Shape: `{ prior_run_dir, prior_campaign_started_at, "
        "prior_campaign_finished_at, completed, slice_count, per_slice }`. "
        "Each `per_slice[\"<coin>/<tf>\"]` row: `coin`, `timeframe`, "
        "`status`, `n_trades`, `net_pct_total`, `net_pct_mean`, "
        "`gross_pct_mean`, `win_rate`, `trade_share`, "
        "`round_trip_cost_pct`, `auc`, `baseline_auc`, `lift_auc`, "
        "`directional_accuracy`, `n_rows`, `emitted_at`. The canonical "
        "PnL field is `net_pct_total` (post-fee % return over the "
        "walk-forward holdout)."
    )
    md.append(
        "- `generated_at` — iso8601 timestamp the prior `report.json` was "
        "written."
    )
    md.append("")
    md.append(
        "Diff two campaigns' per-slice PnL with "
        "`python -m scripts.diff_campaign_pnl <run_a> <run_b>` (each arg "
        "may be a `models/_archive/<TS>_pre_full_run/` dir, its "
        "`baseline_snapshot.json`, or the matching `training_run_<TS>/` "
        "folder)."
    )
    md.append("")
    (run_dir / "summary.md").write_text("\n".join(md))

    _append_progress({
        "phase": "final_summary",
        "status": "ok",
        "headline": (
            f"verdict={final_verdict}, improved={n_improved}, regressed={n_regressed}, "
            f"profitable={len(profitable)}, watchdog={watchdog_verdict}"
        ),
        "summary_path": str((run_dir / "summary.md").relative_to(ROOT)),
    })
    return {
        "phase": "summary", "status": "ok",
        "elapsed_sec": round(time.time() - started, 1),
        "verdict": final_verdict,
    }


# ── Driver ────────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_ts = _ts()
    run_dir = REGISTRY_ROOT / f"training_run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[campaign] run_dir={run_dir.relative_to(ROOT)}", flush=True)

    _append_progress({
        "phase": "campaign_start", "status": "running",
        "headline": f"task #366 full 1-year quant training run, dir={run_dir.relative_to(ROOT)}",
        "run_dir": str(run_dir.relative_to(ROOT)),
    })

    # Tasks #613, #617 — opportunistic, idempotent back-fill of every
    # `diagnostics/<coin>_<tf>_post_fee_<TS>` directory the operator
    # has produced into the live-replay row of
    # `progress_updates.jsonl`. Cheap, never blocks startup, and a
    # no-op on subsequent runs (already-back-filled diagnostics are
    # skipped).
    try:
        _backfill_post_fee_diagnostics()
    except Exception as exc:  # noqa: BLE001
        _append_progress({
            "phase": "slice_live_replay_backfill",
            "status": "warn",
            "headline": (
                f"post-fee diagnostic backfill swallowed exception: {exc}"
            ),
            "live_replay_status": "error",
            "live_replay_error": str(exc),
        })

    try:
        phase1_preflight(run_dir)
        audit = phase2_data_audit(run_dir)
        skip_5m = audit["skipped_5m_coins"]
        baseline_info = phase3_archive_baseline(run_dir)
        baseline_dir = ROOT / baseline_info["archive_dir"]
        phase4_training(run_dir, skip_5m_coins=skip_5m)
        phase567_summary(run_dir, baseline_dir)
    except Exception as exc:  # noqa: BLE001
        _append_progress({
            "phase": "campaign_failed", "status": "fail",
            "headline": f"{type(exc).__name__}: {exc}",
        })
        raise

    _append_progress({
        "phase": "campaign_done", "status": "ok",
        "headline": f"task #366 finished, summary at {run_dir.relative_to(ROOT)}/summary.md",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
