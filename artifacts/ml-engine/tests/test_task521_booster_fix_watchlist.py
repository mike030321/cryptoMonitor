"""Task #521 — booster-fix watchlist surfaced in every campaign run.

The Task #507 booster fix (`TINY_SLICE_THRESHOLD=1500`,
`TINY_SLICE_CLASS_WEIGHT_ALPHA=2.0`) shifted predicted STABLE share up
substantially on these 4 healthy slices that did not need rescuing,
roughly halving their realized trade count. We have no clean pre-fix
PnL on disk (see
`reports/20260428T111719Z-task516-pnl-impact-verification.md` §1) so
we cannot prove a regression today; instead the campaign's
`summary.md` flags these slices' current STABLE share, n_trades, and
post-fee `net_pct_total` every run so an operator can spot a material
regression as soon as Task #516 follow-up #2 (per-slice PnL snapshot)
lands.

These tests pin both the membership of the watchlist (so an accidental
edit surfaces in PR review) and the data shape returned by the helper
(so an accidental rename in `report.json` consumers also surfaces).
"""
from __future__ import annotations

import importlib


def _load_campaign_module():
    return importlib.import_module("scripts.run_full_training_campaign")


def test_task521_watchlist_membership_is_pinned():
    """The four watched slices are explicit, not inferred. Any edit to
    this list must be deliberate (it ships into every campaign's
    summary.md row), so we pin both the slugs and the recorded pre-fix
    DCS-drop magnitudes from the source report.
    """
    mod = _load_campaign_module()

    slugs = [
        (entry["coin"], entry["tf"], entry["pre_fix_dcs_drop_pp"])
        for entry in mod.TASK521_BOOSTER_FIX_WATCHLIST
    ]
    # Order matches the brief's table order (largest pre-fix DCS shift
    # first), and the magnitudes match
    # `reports/20260428T111719Z-task516-pnl-impact-verification.md` §4.
    assert slugs == [
        ("pepe",                    "6h", 32),
        ("jupiter-exchange-solana", "1d", 22),
        ("floki-inu",               "1d", 20),
        ("dogwifcoin",              "6h", 15),
    ]


def test_task521_watchlist_emits_row_per_slice_with_full_coverage():
    """When the report covers all four slices, the helper emits one row
    per watchlist entry with `present=True`, the predicted STABLE share
    derived from `directional_call_share`, and the post-fee PnL pulled
    from `pnl_after_fees`.
    """
    mod = _load_campaign_module()
    report = {
        "timeframes": {
            "1d": {"per_coin": {
                "floki-inu": {
                    "status": "trained",
                    "directional_call_share": 0.8028,
                    "directional_call_share_source": "holdout",
                    "pnl_after_fees": {"n_trades": 163, "net_pct_total": -119.995},
                },
                "jupiter-exchange-solana": {
                    "status": "trained",
                    "directional_call_share": 0.7834,
                    "directional_call_share_source": "holdout",
                    "pnl_after_fees": {"n_trades": 157, "net_pct_total": -47.44},
                },
            }},
            "6h": {"per_coin": {
                "pepe": {
                    "status": "trained",
                    "directional_call_share": 0.6784,
                    "directional_call_share_source": "holdout",
                    "pnl_after_fees": {"n_trades": 235, "net_pct_total": -18.77},
                },
                "dogwifcoin": {
                    "status": "trained",
                    "directional_call_share": 0.8516,
                    "directional_call_share_source": "holdout",
                    "pnl_after_fees": {"n_trades": 205, "net_pct_total": -28.91},
                },
            }},
        },
    }

    rows = mod._task521_watchlist_rows(report)
    by_slug = {r["slice"]: r for r in rows}
    assert set(by_slug) == {
        "pepe/6h",
        "jupiter-exchange-solana/1d",
        "floki-inu/1d",
        "dogwifcoin/6h",
    }

    floki = by_slug["floki-inu/1d"]
    assert floki["present"] is True
    assert floki["status"] == "trained"
    assert floki["directional_call_share_pct"] == 80.28
    # STABLE share = 100 - DCS%, rounded to 2dp. The brief's done
    # criterion explicitly names "predicted STABLE share" as a column.
    assert floki["predicted_stable_share_pct"] == 19.72
    assert floki["directional_call_share_source"] == "holdout"
    assert floki["n_trades"] == 163
    assert floki["post_fee_net_pct_total"] == -119.995
    assert floki["pre_fix_dcs_drop_pp"] == 20
    assert floki["note"] is None

    # pepe/6h had the largest pre-fix shift (32pp); confirm both the
    # pre-fix annotation and the live STABLE share survive the
    # round-trip.
    pepe = by_slug["pepe/6h"]
    assert pepe["pre_fix_dcs_drop_pp"] == 32
    assert pepe["predicted_stable_share_pct"] == 32.16


def test_task521_watchlist_marks_absent_slices_when_timeframe_missing():
    """A campaign that did not (re)train one of the watchlist
    timeframes must still emit a row for every watched slice — with
    `present=False` and an explanatory `note` — so the operator can
    distinguish "we trained it and it looks fine" from "we never looked
    at it this run". Otherwise a regression in an un-retrained slice
    would silently drop off the summary.
    """
    mod = _load_campaign_module()

    # 1d covered for floki only; 6h not trained at all.
    report = {
        "timeframes": {
            "1d": {"per_coin": {
                "floki-inu": {
                    "status": "trained",
                    "directional_call_share": 0.5,
                    "directional_call_share_source": "holdout",
                    "pnl_after_fees": {"n_trades": 10, "net_pct_total": 1.0},
                },
            }},
        },
    }
    rows = mod._task521_watchlist_rows(report)
    by_slug = {r["slice"]: r for r in rows}

    # Always 4 rows, one per watchlist entry, regardless of coverage.
    assert len(rows) == 4

    for slug in ("pepe/6h", "dogwifcoin/6h"):
        r = by_slug[slug]
        assert r["present"] is False
        assert r["status"] is None
        assert r["predicted_stable_share_pct"] is None
        assert r["n_trades"] is None
        assert r["post_fee_net_pct_total"] is None
        assert "not trained this campaign run" in r["note"]
        # 6h timeframe entirely absent from this report → the note
        # specifically calls out the missing timeframe so an operator
        # can tell which knob (timeframe selection) excluded the slice.
        assert "timeframe '6h' absent from report" in r["note"]

    # 1d trained but the watched coin isn't in its per_coin block →
    # the note must distinguish this from "timeframe missing" so a
    # campaign that explicitly excluded the coin doesn't read the same
    # as one that simply skipped 1d entirely.
    juju = by_slug["jupiter-exchange-solana/1d"]
    assert juju["present"] is False
    assert (
        "coin 'jupiter-exchange-solana' absent from '1d' per_coin block"
        in juju["note"]
    )

    # The one slice that was retrained still gets the live numbers.
    floki = by_slug["floki-inu/1d"]
    assert floki["present"] is True
    assert floki["predicted_stable_share_pct"] == 50.0
    assert floki["n_trades"] == 10


def test_task521_watchlist_handles_missing_dcs_or_pnl_gracefully():
    """A trained slice whose `directional_call_share` or
    `pnl_after_fees` is missing (older trainer build, partial write,
    untrained pooled fallback path) must produce an explicit n/a row
    rather than crashing the summary writer. The Phase 7 markdown is
    the campaign's only operator-facing artifact, so this code path
    cannot fail open.
    """
    mod = _load_campaign_module()
    report = {
        "timeframes": {
            "1d": {"per_coin": {
                "floki-inu": {"status": "trained"},  # no DCS, no PnL
                "jupiter-exchange-solana": {
                    "status": "trained",
                    "directional_call_share": "not-a-number",
                    "pnl_after_fees": {},  # PnL block exists but empty
                },
            }},
        },
    }
    rows = mod._task521_watchlist_rows(report)
    by_slug = {r["slice"]: r for r in rows}

    floki = by_slug["floki-inu/1d"]
    assert floki["present"] is True
    assert floki["directional_call_share_pct"] is None
    assert floki["predicted_stable_share_pct"] is None
    assert floki["n_trades"] is None
    assert floki["post_fee_net_pct_total"] is None

    juju = by_slug["jupiter-exchange-solana/1d"]
    assert juju["present"] is True
    # Garbage DCS is coerced to None, not raised.
    assert juju["directional_call_share_pct"] is None
    assert juju["predicted_stable_share_pct"] is None
    assert juju["n_trades"] is None
    assert juju["post_fee_net_pct_total"] is None
