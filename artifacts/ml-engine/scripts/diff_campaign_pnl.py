"""Task #520 — diff per-slice walk-forward PnL between two campaigns.

Prints a markdown table (sorted by absolute delta, descending) of every
slice that appears in either snapshot's `prior_campaign_per_slice_pnl`
block. Reads `baseline_snapshot.json` written by
`scripts/run_full_training_campaign.py`'s phase 3.

Usage:
    python -m scripts.diff_campaign_pnl <run_a> <run_b>

`<run_a>` / `<run_b>` may each be:
  - a path to a `baseline_snapshot.json` file, or
  - a path to a `models/_archive/<TS>_pre_full_run/` directory, or
  - a path to a `models/training_run_<TS>/` directory (resolved via
    `phase3_baseline_pointer.json`).

Exit status:
  0 — diff printed (regardless of regressions).
  2 — could not resolve a snapshot for one of the inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "models" / "_archive"


def _resolve_snapshot(arg: str) -> Path:
    """Best-effort resolve a CLI arg to a `baseline_snapshot.json` path."""
    p = Path(arg)
    if not p.is_absolute():
        # Try several common roots before giving up.
        candidates = [Path.cwd() / arg, ROOT / arg, ARCHIVE_ROOT / arg]
        for cand in candidates:
            if cand.exists():
                p = cand
                break
        else:
            p = candidates[0]
    if p.is_file() and p.name == "baseline_snapshot.json":
        return p
    if p.is_dir():
        direct = p / "baseline_snapshot.json"
        if direct.exists():
            return direct
        # training_run_<TS>/ — follow phase3_baseline_pointer.json.
        pointer = p / "phase3_baseline_pointer.json"
        if pointer.exists():
            try:
                rel = json.loads(pointer.read_text()).get("archive_dir")
                if rel:
                    snap = (ROOT / rel / "baseline_snapshot.json")
                    if snap.exists():
                        return snap
            except Exception:  # noqa: BLE001
                pass
    raise FileNotFoundError(
        f"Could not resolve a baseline_snapshot.json from '{arg}'. "
        f"Tried: {p}"
    )


def _load_per_slice_pnl(snapshot_path: Path) -> dict[str, dict]:
    payload = json.loads(snapshot_path.read_text())
    block = payload.get("prior_campaign_per_slice_pnl") or {}
    return dict(block.get("per_slice") or {})


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta(a, b):
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_a",
        help="Older snapshot (baseline_snapshot.json, archive dir, or run dir).",
    )
    parser.add_argument(
        "run_b",
        help="Newer snapshot (baseline_snapshot.json, archive dir, or run dir).",
    )
    parser.add_argument(
        "--metric",
        default="net_pct_total",
        choices=["net_pct_total", "net_pct_mean", "n_trades", "win_rate", "auc"],
        help="Per-slice field to diff (default: net_pct_total).",
    )
    args = parser.parse_args(argv)

    try:
        snap_a = _resolve_snapshot(args.run_a)
        snap_b = _resolve_snapshot(args.run_b)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    a_slices = _load_per_slice_pnl(snap_a)
    b_slices = _load_per_slice_pnl(snap_b)
    all_keys = sorted(set(a_slices) | set(b_slices))

    print(f"# Per-slice PnL diff: {args.metric}")
    print()
    print(f"- A (older): `{snap_a.relative_to(ROOT) if snap_a.is_relative_to(ROOT) else snap_a}` ({len(a_slices)} slices)")
    print(f"- B (newer): `{snap_b.relative_to(ROOT) if snap_b.is_relative_to(ROOT) else snap_b}` ({len(b_slices)} slices)")
    print(f"- Slices in either: {len(all_keys)}")
    print()

    if not all_keys:
        print("_Both snapshots' `prior_campaign_per_slice_pnl.per_slice` are empty._")
        print()
        print(
            "Tip: this block is populated by archives written after task #520. "
            "Older archives only carry the (smaller) `per_slice` block sourced "
            "from `report.json`."
        )
        return 0

    rows: list[tuple] = []
    for k in all_keys:
        a_row = a_slices.get(k) or {}
        b_row = b_slices.get(k) or {}
        a_v = a_row.get(args.metric)
        b_v = b_row.get(args.metric)
        d = _delta(a_v, b_v)
        rows.append((
            k,
            a_row.get("status"),
            b_row.get("status"),
            a_v,
            b_v,
            d,
            a_row.get("n_trades"),
            b_row.get("n_trades"),
        ))
    # Sort by absolute delta descending; None deltas sink to the bottom.
    rows.sort(key=lambda r: (r[5] is None, -abs(r[5]) if r[5] is not None else 0))

    print(f"| slice | A.status | B.status | A.{args.metric} | B.{args.metric} | delta | A.n_trades | B.n_trades |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for k, a_st, b_st, a_v, b_v, d, a_n, b_n in rows:
        print(
            f"| {k} | {_fmt(a_st)} | {_fmt(b_st)} | {_fmt(a_v)} | {_fmt(b_v)} "
            f"| {_fmt(d)} | {_fmt(a_n)} | {_fmt(b_n)} |"
        )

    deltas = [r[5] for r in rows if r[5] is not None]
    if deltas:
        n_reg = sum(1 for d in deltas if d < 0)
        n_imp = sum(1 for d in deltas if d > 0)
        print()
        print(
            f"_Summary on `{args.metric}`: improved={n_imp}, regressed={n_reg}, "
            f"flat={len(deltas) - n_imp - n_reg}, undiffable={len(rows) - len(deltas)}._"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
