"""Regenerate Task #366 Phase 2 (data-integrity), Phase 3 (baseline
snapshot) and Phase 7 (summary) artifacts for the existing run folder
using the corrected schema and evidence fields. Phase 4 (training) is
NOT re-run — the trained models and `report.json` on disk are reused.

Usage:
    python -m scripts.regenerate_phases_236 <run_dir_name>
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_training_campaign import (  # noqa: E402
    REGISTRY_ROOT,
    ARCHIVE_ROOT,
    phase2_data_audit,
    phase567_summary,
    _append_progress,
    _extract_snapshot_from_report,
)

def main(run_dir_name: str) -> None:
    run_dir = REGISTRY_ROOT / run_dir_name
    assert run_dir.exists(), f"{run_dir} missing"

    # Regenerate Phase 2 data audit (live DB).
    print("=== Phase 2 (data audit) ===", flush=True)
    phase2_data_audit(run_dir)

    # Regenerate Phase 3 baseline snapshot against the EARLIEST archived
    # pre-run snapshot (truly pre-run — before any Task #366 training).
    print("=== Phase 3 (baseline snapshot re-extract) ===", flush=True)
    archives = sorted([d for d in ARCHIVE_ROOT.iterdir() if d.is_dir() and d.name.endswith("_pre_full_run")])
    assert archives, "no _pre_full_run archive found"
    earliest = archives[0]
    rep_path = earliest / "report.json"
    assert rep_path.exists(), f"archived report.json missing in {earliest}"
    rep = json.loads(rep_path.read_text())
    baseline = {"per_slice": {}, "source_archive": str(earliest.name)}
    for tf, tf_rep in (rep.get("timeframes") or {}).items():
        if not isinstance(tf_rep, dict):
            continue
        for coin, c_rep in (tf_rep.get("per_coin") or {}).items():
            if isinstance(c_rep, dict):
                baseline["per_slice"][f"{coin}/{tf}"] = _extract_snapshot_from_report(c_rep)
        pooled = tf_rep.get("pooled") or {}
        if isinstance(pooled, dict):
            baseline["per_slice"][f"__pooled__/{tf}"] = _extract_snapshot_from_report(pooled)
    baseline["generated_at"] = rep.get("generated_at")

    # Archived model audit for forbidden feature prefixes.
    from app.training.registry import FORBIDDEN_FEATURE_PREFIXES
    leaks = []
    scanned = 0
    for manifest in earliest.rglob("*.json"):
        try:
            payload = json.loads(manifest.read_text())
        except Exception:
            continue
        scanned += 1
        feats = payload.get("feature_names") or payload.get("features")
        if not isinstance(feats, list):
            continue
        for f in feats:
            if isinstance(f, str) and any(f.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES):
                leaks.append({"manifest": str(manifest.relative_to(earliest)), "feature": f})
    baseline["archived_model_audit"] = {
        "manifests_scanned": scanned,
        "forbidden_feature_leaks": leaks,
    }
    (earliest / "baseline_snapshot.json").write_text(json.dumps(baseline, indent=2, default=str))
    (run_dir / "phase3_baseline_pointer.json").write_text(json.dumps({
        "archive_dir": str(earliest),
        "regenerated_via": "regenerate_phases_236",
        "forbidden_feature_leaks": leaks,
        "manifests_scanned": scanned,
        "baseline_slices_captured": len(baseline["per_slice"]),
    }, indent=2))
    print(f"  archive={earliest.name} slices={len(baseline['per_slice'])} leaks={len(leaks)} scanned={scanned}", flush=True)
    _append_progress({
        "phase": "baseline_archive_regen",
        "status": "ok",
        "headline": f"slices={len(baseline['per_slice'])} leaks={len(leaks)} scanned={scanned}",
    })

    # Regenerate Phase 7.
    print("=== Phase 7 (summary regen) ===", flush=True)
    phase567_summary(run_dir, earliest)


if __name__ == "__main__":
    main(sys.argv[1])
