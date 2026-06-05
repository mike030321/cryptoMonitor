"""Task #365 — Quant-Only Enforcement.

Walk the on-disk model registry and archive every model whose
manifest.feature_names still advertises an LLM-derived feature
(news_*, llm_*, gpt_*, sentiment_*, ai_*).

Strategy:
  - The model directory itself is RENAMED to add a `.archived_pre_quantonly`
    suffix. registry.list_versions skips dirs whose `manifest.json` is
    missing under the original version name, so renaming is sufficient
    to retire a slot from `latest_version()` selection.
  - The latest pointer file (`latest`) is renamed to `latest.archived`.
  - A JSON inventory of every archived dir is written to
    `audit/archived_models.json` so the audit report (proof B) can cite
    exact counts and example paths.

This script is idempotent: re-running it skips already-archived dirs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS = REPO_ROOT / "artifacts" / "ml-engine" / "models"
AUDIT = REPO_ROOT / "audit"
FORBIDDEN = ("news_", "llm_", "gpt_", "sentiment_", "ai_")
SUFFIX = ".archived_pre_quantonly_20260423"


def main() -> int:
    archived: list[dict] = []
    # Materialize the list BEFORE iterating: rglob is a generator and we
    # mutate the directory tree as we go (renames). Without this we hit
    # FileNotFoundError partway through the walk.
    all_manifests = list(MODELS.rglob("manifest.json"))
    for manifest in all_manifests:
        s = str(manifest)
        if "_archive_" in s or SUFFIX in s:
            continue
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            continue
        feats = data.get("feature_names") or []
        bad = sorted({c for c in feats if any(c.startswith(p) for p in FORBIDDEN)})
        if not bad:
            continue
        version_dir = manifest.parent  # …/<coin>/<tf>/<version>/
        new_name = version_dir.with_name(version_dir.name + SUFFIX)
        if not new_name.exists():
            version_dir.rename(new_name)
        # Move the `latest` pointer out of the way too — it may still
        # name a non-archived sibling, but we want the registry to
        # re-resolve from scratch on next training cycle.
        latest = version_dir.parent / "latest"
        if latest.exists() and not latest.name.endswith(SUFFIX):
            latest.rename(latest.with_name("latest" + SUFFIX))
        archived.append({
            "coin_id": data.get("coin_id"),
            "timeframe": data.get("timeframe"),
            "version": data.get("version"),
            "model_kind": data.get("model_kind"),
            "metrics": data.get("metrics"),
            "baseline_metrics": data.get("baseline_metrics"),
            "n_train_rows": data.get("n_train_rows"),
            "n_test_rows": data.get("n_test_rows"),
            "forbidden_features": bad,
            "archived_path": str(new_name.relative_to(REPO_ROOT)),
        })
    AUDIT.mkdir(exist_ok=True)
    out = AUDIT / "archived_models.json"
    out.write_text(json.dumps({
        "task": 365,
        "rule": "Quant-Only Enforcement — archive any model with LLM-derived feature columns",
        "forbidden_prefixes": list(FORBIDDEN),
        "archive_suffix": SUFFIX,
        "count": len(archived),
        "models": archived,
    }, indent=2))
    print(f"archived={len(archived)} inventory={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
