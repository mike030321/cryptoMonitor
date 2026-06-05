"""Tests for failure_analysis.history (task #336)."""
from __future__ import annotations

import json
from pathlib import Path

from app.training.failure_analysis import history


def _write(reports_dir: Path, ts: str, buckets: dict[str, int]) -> Path:
    p = reports_dir / f"{ts}-failure-analysis-auto.json"
    p.write_text(
        json.dumps(
            {
                "generated_at": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}+00:00",
                "bucket_counts": buckets,
            }
        )
    )
    return p


def test_history_empty_when_no_reports_dir(tmp_path: Path):
    registry_root = tmp_path / "models"
    registry_root.mkdir()
    out = history(registry_root, limit=10)
    assert out == {"rows": [], "count": 0}


def test_history_returns_newest_first_capped_by_limit(tmp_path: Path):
    registry_root = tmp_path / "models"
    registry_root.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    _write(reports, "20260101T000000Z", {"promoted": 1})
    _write(reports, "20260102T000000Z", {"promoted": 2})
    _write(reports, "20260103T000000Z", {"promoted": 3})

    out = history(registry_root, limit=2)
    assert out["count"] == 2
    counts = [r["bucket_counts"]["promoted"] for r in out["rows"]]
    assert counts == [3, 2]


def test_history_skips_corrupt_files(tmp_path: Path):
    registry_root = tmp_path / "models"
    registry_root.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    _write(reports, "20260101T000000Z", {"promoted": 1})
    bad = reports / "20260102T000000Z-failure-analysis-auto.json"
    bad.write_text("{not json")
    _write(reports, "20260103T000000Z", {"promoted": 3})

    out = history(registry_root, limit=10)
    assert out["count"] == 2
    assert [r["bucket_counts"]["promoted"] for r in out["rows"]] == [3, 1]


def test_history_limit_zero_returns_empty(tmp_path: Path):
    registry_root = tmp_path / "models"
    registry_root.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    _write(reports, "20260101T000000Z", {"promoted": 1})
    out = history(registry_root, limit=0)
    assert out == {"rows": [], "count": 0}
