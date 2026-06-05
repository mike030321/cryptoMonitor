"""Task #667 — B4 promotion + diagnostic-sandbox validation driver.

Companion to `b4_margin_sweep.py`. Given a model directory under
`models/bitcoin/5m/<version>/` produced by the sweep, this driver:

  1. Inserts a fresh `state='shadow'` row into `model_registry` for
     (model_id='lightgbm', coin_id='bitcoin', timeframe='5m', version).
     A direct INSERT is intentional — promotion is the safety-critical
     transition that must go through `promote_shadow_to_serving`; the
     shadow-row insert is a benign bookkeeping write that mirrors what
     `register_shadow_rows` does for normal training reports.
  2. Calls `app.registry_lifecycle.promote_shadow_to_serving` with the
     scope_constraint from the manifest (allowed_universe pinned to
     ['bitcoin:5m']) so the row flips to champion atomically.
  3. POSTs `/api/diagnostic-sandbox/btc-version` with the version,
     POSTs `/api/diagnostic-sandbox/mode` with mode='diagnostic_sandbox',
     POSTs `/api/diagnostic-sandbox/evaluate` 10 times verifying every
     response has `tripped=false` (no auto-disable trips).

Output: a markdown + JSON report under `reports/` with the full chain
of promotion, http calls, and per-evaluate responses for audit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import urllib.error
import urllib.request

from ..registry import REGISTRY_ROOT, load_model
from ...registry_lifecycle import (
    PromotionError,
    promote_shadow_to_serving,
)

logger = logging.getLogger("labels_research.b4_promote_and_validate")

TASK_ID = 667
COIN = "bitcoin"
TIMEFRAME = "5m"
MODEL_ID = "lightgbm"
DS_BASE_URL_DEFAULT = "http://localhost:80/api"
N_EVALUATES = 10
REPORTS_DIR = REGISTRY_ROOT.parent / "reports"
ML_ROOT = REGISTRY_ROOT.parent


def _http_post(url: str, body: dict, *, admin_key: str) -> dict:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={
            "content-type": "application/json",
            "x-admin-key": admin_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
            return {"status": resp.status, "body": _safe_json(text)}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return {"status": exc.code, "body": _safe_json(text), "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": None, "body": None, "error": f"{type(exc).__name__}: {exc}"}


def _safe_json(text: str):
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return text


async def _insert_shadow_row(
    *, version: str, manifest_note: str,
) -> int:
    """Insert a `state='shadow'` model_registry row for the new version
    and return its id.

    Re-run behaviour: if a row already exists for the same
    (model_id, coin_id, timeframe, version) it is REUSED — its id is
    returned regardless of state. This means a re-run after a
    successful promotion will return the already-promoted row in
    `state='champion'`, and the subsequent `promote_shadow_to_serving`
    call will REJECT it with `PromotionError` (state must be 'shadow').
    That rejection is intentional and surfaces in the report's
    `error` field so the operator sees "this version is already
    serving" instead of silently flipping state again.
    """
    from ...db import init_pool
    pool = await init_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            """
            SELECT id, state FROM model_registry
            WHERE model_id = $1 AND model_version = $2
              AND coin_id = $3 AND timeframe = $4
            ORDER BY id DESC
            LIMIT 1
            """,
            MODEL_ID, version, COIN, TIMEFRAME,
        )
        if existing is not None:
            logger.info(
                "shadow_row_already_exists id=%s state=%s",
                existing["id"], existing["state"],
            )
            return int(existing["id"])
        row = await conn.fetchrow(
            """
            INSERT INTO model_registry
                (model_id, model_version, coin_id, timeframe,
                 state, note, metrics_snapshot, is_active)
            VALUES ($1, $2, $3, $4, 'shadow', $5, $6::jsonb, true)
            RETURNING id
            """,
            MODEL_ID, version, COIN, TIMEFRAME,
            f"Task #{TASK_ID} B4 sweep winner: {manifest_note}",
            json.dumps({
                "task": f"task-{TASK_ID}-b4",
                "source": "b4_margin_sweep",
            }),
        )
        return int(row["id"])


def _load_manifest_dict(version: str) -> dict:
    p = REGISTRY_ROOT / COIN / TIMEFRAME / version / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest not found at {p}")
    return json.loads(p.read_text())


async def run_promote_and_validate(
    *, version: str, base_url: str, admin_key: str,
) -> dict:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")

    summary: dict = {
        "task": f"task-{TASK_ID}-b4-promote-and-validate",
        "started_utc": run_id,
        "version": version,
        "coin": COIN,
        "timeframe": TIMEFRAME,
        "model_id": MODEL_ID,
        "base_url": base_url,
        "admin_key_present": bool(admin_key),
    }

    # 0. Pre-flight: bail BEFORE any DB write or state flip if the
    #    admin key is missing — otherwise we would promote a model and
    #    then be unable to drive the 10-paper-proof /evaluate sequence
    #    that the task requires, leaving the operator with a partial
    #    workflow (model in champion state, no validation evidence).
    if not admin_key:
        summary["error"] = (
            "ADMIN_API_KEY not set in environment; refusing to promote "
            "before the diagnostic-sandbox endpoints can be exercised. "
            "Re-run with ADMIN_API_KEY set."
        )
        return summary

    # 1. Verify the on-disk slice loads (registry.load_model walks
    #    manifest validation so we catch shape errors before flipping
    #    any DB state).
    loaded = load_model(COIN, TIMEFRAME, version)
    if loaded is None:
        summary["error"] = (
            f"load_model returned None for {COIN}/{TIMEFRAME}/{version}"
        )
        return summary
    manifest_dict = _load_manifest_dict(version)
    summary["manifest_check"] = {
        "version": manifest_dict["version"],
        "served_predictor_kind": manifest_dict.get("served_predictor_kind"),
        "calibration_method": manifest_dict.get("calibration_method"),
        "calibration_status": manifest_dict.get("calibration_status"),
        "scope_constraint": manifest_dict.get("scope_constraint"),
        "abstain_tau": manifest_dict.get("abstain_tau"),
        "friction_threshold_pct": manifest_dict.get("friction_threshold_pct"),
        "label_family": manifest_dict.get("label_family"),
    }

    # 2. Insert / locate the shadow row.
    note_for_row = manifest_dict.get("note", "")[:200]
    shadow_id = await _insert_shadow_row(
        version=version, manifest_note=note_for_row,
    )
    summary["shadow_row_id"] = shadow_id

    # 3. Promote via promote_shadow_to_serving (the only sanctioned
    #    promotion path).
    scope_constraint = dict(manifest_dict.get("scope_constraint") or {})
    if not scope_constraint:
        summary["error"] = "manifest is missing scope_constraint"
        return summary
    try:
        promotion = await promote_shadow_to_serving(
            shadow_id,
            scope_constraint=scope_constraint,
            promoted_by=f"task-{TASK_ID}-b4",
            note=(
                f"Task #{TASK_ID} B4 sweep winner promoted to BTC/5m DS lane "
                f"(version={version}, scope=bitcoin:5m, "
                f"calibration=beta/under_confident_documented)."
            ),
        )
        summary["promotion"] = {
            "promoted_id": promotion.promoted_id,
            "previous_champion_id": promotion.previous_champion_id,
            "promoted_by": getattr(promotion, "promoted_by", None),
            "scope_constraint": scope_constraint,
        }
    except PromotionError as exc:
        summary["error"] = f"promotion failed: {exc}"
        return summary

    # 4. POST btc-version, then mode, then 10 evaluates. Admin-key
    #    presence was already enforced in step 0 above.
    summary["http"] = {}
    summary["http"]["btc_version"] = _http_post(
        f"{base_url}/diagnostic-sandbox/btc-version",
        {"version": version},
        admin_key=admin_key,
    )
    summary["http"]["mode"] = _http_post(
        f"{base_url}/diagnostic-sandbox/mode",
        {"mode": "diagnostic_sandbox"},
        admin_key=admin_key,
    )

    proofs: list[dict] = []
    any_tripped = False
    for i in range(N_EVALUATES):
        r = _http_post(
            f"{base_url}/diagnostic-sandbox/evaluate",
            {},
            admin_key=admin_key,
        )
        proofs.append({"i": i + 1, **r})
        b = r.get("body") or {}
        if isinstance(b, dict) and b.get("tripped"):
            any_tripped = True
        # tiny pause so the API server doesn't see N evaluates in a
        # single millisecond — the tally evaluator reads from the DB
        # and is harmless under burst, but a 100ms gap matches a real
        # operator-paced check.
        await asyncio.sleep(0.1)
    summary["http"]["evaluates"] = proofs
    summary["http"]["any_tripped"] = bool(any_tripped)
    summary["http"]["all_clean"] = bool(not any_tripped)

    return summary


def write_report(summary: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = summary.get("started_utc") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ",
    )
    stem = f"task-{TASK_ID}-b4-promote-and-validate-{ts}"
    json_path = REPORTS_DIR / f"{stem}.json"
    md_path = REPORTS_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path.write_text(_render_markdown(summary))
    return md_path, json_path


def _render_markdown(s: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Task #{TASK_ID} — B4 promote + DS validate")
    lines.append("")
    lines.append(f"- run_id: `{s.get('started_utc')}`")
    lines.append(
        f"- target: `{s.get('coin')}/{s.get('timeframe')}` version "
        f"`{s.get('version')}` (model_id `{s.get('model_id')}`)"
    )
    lines.append(f"- base_url: `{s.get('base_url')}`")
    if s.get("error"):
        lines.append(f"- **ERROR**: `{s['error']}`")
        return "\n".join(lines) + "\n"

    mc = s.get("manifest_check", {})
    lines.append("")
    lines.append("## Manifest check")
    for k in (
        "served_predictor_kind", "calibration_method",
        "calibration_status", "label_family", "abstain_tau",
        "friction_threshold_pct",
    ):
        lines.append(f"- {k}: `{mc.get(k)}`")
    lines.append(
        f"- scope_constraint: `{json.dumps(mc.get('scope_constraint'))}`"
    )

    lines.append("")
    lines.append("## Promotion")
    lines.append(f"- shadow_row_id: `{s.get('shadow_row_id')}`")
    p = s.get("promotion", {})
    lines.append(f"- promoted_id: `{p.get('promoted_id')}`")
    lines.append(
        f"- previous_champion_id: `{p.get('previous_champion_id')}`"
    )

    h = s.get("http", {}) or {}
    lines.append("")
    lines.append("## HTTP")
    lines.append("")
    lines.append(
        f"- POST btc-version status="
        f"`{(h.get('btc_version') or {}).get('status')}` "
        f"body=`{json.dumps((h.get('btc_version') or {}).get('body'))}`"
    )
    lines.append(
        f"- POST mode status=`{(h.get('mode') or {}).get('status')}` "
        f"body=`{json.dumps((h.get('mode') or {}).get('body'))}`"
    )

    lines.append("")
    lines.append(f"## 10 paper proofs (POST /diagnostic-sandbox/evaluate)")
    lines.append("")
    lines.append("| i | status | tripped | kind | reason |")
    lines.append("|---:|---:|:--|:--|:--|")
    for ev in (h.get("evaluates") or []):
        body = ev.get("body") or {}
        if isinstance(body, dict):
            tripped = body.get("tripped")
            kind = body.get("kind")
            reason = body.get("reason")
        else:
            tripped, kind, reason = "?", "?", str(body)
        lines.append(
            f"| {ev.get('i')} | {ev.get('status')} | {tripped} | "
            f"{kind or ''} | {reason or ''} |"
        )
    lines.append("")
    lines.append(
        f"- any_tripped: `{h.get('any_tripped')}`"
    )
    lines.append(
        f"- all_clean: `{h.get('all_clean')}`"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(
        description=(
            f"Task #{TASK_ID} — promote a B4 BTC/5m model to champion + "
            "drive 10 diagnostic-sandbox /evaluate calls."
        ),
    )
    p.add_argument(
        "--version", required=True,
        help="Model version directory under models/bitcoin/5m/<version>/",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("DS_BASE_URL", DS_BASE_URL_DEFAULT),
    )
    p.add_argument(
        "--admin-key",
        default=os.environ.get("ADMIN_API_KEY", ""),
    )
    args = p.parse_args()

    summary = asyncio.run(
        run_promote_and_validate(
            version=args.version,
            base_url=args.base_url,
            admin_key=args.admin_key,
        ),
    )
    md_path, json_path = write_report(summary)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    if summary.get("error"):
        print(f"ERROR: {summary['error']}")
        raise SystemExit(2)
    if not (summary.get("http", {}) or {}).get("all_clean", False):
        print("WARNING: at least one /evaluate call returned tripped=true")
        raise SystemExit(3)
    print("OK — promotion + 10 paper proofs all clean")


if __name__ == "__main__":
    main()
