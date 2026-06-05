"""Task #338: rotation helper for auto failure-analysis reports."""
from __future__ import annotations

from pathlib import Path

from app.training.failure_analysis import MAX_AUTO_REPORTS, prune_auto_reports


def _make_pair(reports_dir: Path, idx: int) -> tuple[Path, Path]:
    ts = f"20260101T{idx:06d}Z"
    jp = reports_dir / f"{ts}-failure-analysis-auto.json"
    mp = reports_dir / f"{ts}-failure-analysis-auto.md"
    jp.write_text("{}")
    mp.write_text("# stub")
    return jp, mp


def test_prune_keeps_newest_n_pairs(tmp_path: Path) -> None:
    rd = tmp_path / "reports"
    rd.mkdir()

    pairs: list[tuple[Path, Path]] = []
    for i in range(105):
        pairs.append(_make_pair(rd, i))

    # Hand-run files (no `-auto` suffix) and unrelated files must survive.
    hand_json = rd / "20260101T999999Z-failure-analysis.json"
    hand_md = rd / "20260101T999999Z-failure-analysis.md"
    hand_json.write_text("{}")
    hand_md.write_text("# hand")
    other = rd / "20260101T000000Z-schema-audit.md"
    other.write_text("# other")

    deleted = prune_auto_reports(rd, keep=MAX_AUTO_REPORTS)

    # 5 pairs * 2 files each = 10 deletions.
    assert deleted == 10

    surviving_auto = sorted(rd.glob("*-failure-analysis-auto.json"))
    assert len(surviving_auto) == MAX_AUTO_REPORTS
    # Newest 100 (indices 5..104) should remain.
    expected = {pairs[i][0].name for i in range(5, 105)}
    assert {p.name for p in surviving_auto} == expected
    # Matching .md files survive too.
    for i in range(5, 105):
        assert pairs[i][1].exists()
    # Oldest 5 pairs are gone.
    for i in range(5):
        assert not pairs[i][0].exists()
        assert not pairs[i][1].exists()

    # Hand-run + unrelated files untouched.
    assert hand_json.exists()
    assert hand_md.exists()
    assert other.exists()


def test_prune_noop_when_under_limit(tmp_path: Path) -> None:
    rd = tmp_path / "reports"
    rd.mkdir()
    for i in range(10):
        _make_pair(rd, i)
    assert prune_auto_reports(rd, keep=MAX_AUTO_REPORTS) == 0
    assert len(list(rd.glob("*-failure-analysis-auto.json"))) == 10


def test_prune_missing_dir_is_safe(tmp_path: Path) -> None:
    assert prune_auto_reports(tmp_path / "nope") == 0
