"""Unit tests for the on-real-data verification gate (Task #306)."""
from __future__ import annotations

from app.training.verification import (
    MIN_DIRECTIONAL_ACCURACY,
    MIN_DIRECTIONAL_ACCURACY_PER_TF,
    MIN_HOLDOUT_ROWS,
    REASON_BELOW_COINFLIP,
    REASON_CONTRACT_FAILED,
    REASON_INSUFFICIENT_SAMPLE,
    REASON_NO_LIFT,
    REASON_PROMOTED,
    REASON_UNTRAINED,
    build_verification_block,
    classify_slice,
    min_directional_accuracy_for,
)


def _slice(da: float, base_da: float, holdout_rows: int) -> dict:
    """Mint a fake `train_one_slice` result with a single fold whose
    `n_test` matches `holdout_rows`."""
    return {
        "status": "trained",
        "metrics": {"directional_accuracy": da},
        "baseline_metrics": {"directional_accuracy": base_da},
        "fold_metrics": [{"fold": 0, "n_train": 1000, "n_test": holdout_rows}],
    }


def test_classify_promoted_slice():
    res = classify_slice(_slice(0.58, 0.50, 250))
    assert res["promoted"] is True
    assert res["reason"] == REASON_PROMOTED
    assert res["lift"] == 0.58 - 0.50


def test_classify_no_lift_when_tying_baseline():
    res = classify_slice(_slice(0.55, 0.55, 250))
    assert res["promoted"] is False
    assert res["reason"] == REASON_NO_LIFT


def test_classify_below_coinflip():
    res = classify_slice(_slice(0.45, 0.40, 250))
    assert res["promoted"] is False
    assert res["reason"] == REASON_BELOW_COINFLIP


def test_classify_insufficient_sample():
    res = classify_slice(_slice(0.60, 0.50, MIN_HOLDOUT_ROWS - 1))
    assert res["promoted"] is False
    assert res["reason"] == REASON_INSUFFICIENT_SAMPLE


def test_classify_untrained_slice():
    res = classify_slice({"status": "insufficient_data", "n_rows": 10})
    assert res["promoted"] is False
    assert res["reason"] == REASON_UNTRAINED


def test_block_passed_requires_every_coin_promoted():
    report = {
        "timeframes": {
            "1h": {
                "status": "trained",
                "per_coin": {
                    "pepe": _slice(0.58, 0.50, 250),
                    "bonk": _slice(0.45, 0.40, 250),  # below coinflip
                },
                "pooled": None,
            },
        },
    }
    v = build_verification_block(report, ["pepe", "bonk"])
    assert v["passed"] is False
    assert "bonk" in v["coins_without_promotion"]
    assert "pepe" in v["coins_with_promotion"]
    assert v["counts"]["slices_promoted"] == 1
    assert v["counts"]["slices_below_coinflip"] == 1


def test_block_passed_when_pooled_promotes():
    """A promoted pooled slice serves every active coin."""
    report = {
        "timeframes": {
            "1h": {
                "status": "trained",
                "per_coin": {
                    "pepe": {"status": "insufficient_data", "n_rows": 10},
                    "bonk": {"status": "insufficient_data", "n_rows": 10},
                },
                "pooled": _slice(0.58, 0.50, 250),
            },
        },
    }
    v = build_verification_block(report, ["pepe", "bonk"])
    assert v["passed"] is True
    assert v["coins_without_promotion"] == []
    # Pool counts toward both coins
    assert v["promoted_by_coin"]["pepe"] >= 1
    assert v["promoted_by_coin"]["bonk"] >= 1


def test_contract_failed_propagates():
    report = {
        "timeframes": {
            "1h": {
                "status": "leakage_audit_failed",
                "per_coin": {},
                "pooled": None,
            },
        },
    }
    v = build_verification_block(report, ["pepe", "bonk"])
    assert v["passed"] is False
    # Both coins recorded as contract_failed for the timeframe.
    assert v["counts"]["slices_contract_failed"] >= 2


def test_provenance_rejected_marks_contract_failed():
    report = {
        "timeframes": {
            "1h": {
                "status": "trained",
                "provenance": {"rejected_synthetic": True},
                "per_coin": {"pepe": _slice(0.58, 0.50, 250)},
                "pooled": None,
            },
        },
    }
    v = build_verification_block(report, ["pepe"])
    assert v["counts"]["slices_contract_failed"] >= 1


def test_min_thresholds_documented():
    """Sanity: the constants used in TRAINING_CONTRACT.md match the gate."""
    assert MIN_HOLDOUT_ROWS == 200
    assert MIN_DIRECTIONAL_ACCURACY == 0.50


# ── Task #401 — per-tf directional-accuracy floor ────────────────────────


def test_min_directional_accuracy_per_tf_documents_1d_override():
    """The 1d floor is the 1σ-above-coin-flip value documented in the
    module-level comment (sqrt(0.25/265) ≈ 0.0307 → 0.530). Nothing
    else gets an override; default is unchanged."""
    assert MIN_DIRECTIONAL_ACCURACY_PER_TF["1d"] == 0.530
    # Default unchanged.
    assert MIN_DIRECTIONAL_ACCURACY == 0.50
    # 1d is the only override.
    assert set(MIN_DIRECTIONAL_ACCURACY_PER_TF.keys()) == {"1d"}


def test_min_directional_accuracy_for_resolves_default_for_other_tf():
    assert min_directional_accuracy_for("1h") == 0.50
    assert min_directional_accuracy_for("2h") == 0.50
    assert min_directional_accuracy_for("6h") == 0.50
    assert min_directional_accuracy_for("5m") == 0.50
    assert min_directional_accuracy_for(None) == 0.50
    assert min_directional_accuracy_for("unknown_tf") == 0.50
    # 1d is the documented override.
    assert min_directional_accuracy_for("1d") == 0.530


def test_classify_1d_below_new_floor_is_below_coinflip():
    """A 1d slice with DA in (0.500, 0.530] used to clear the gate
    under the legacy `> 0.50` floor. Under the per-tf floor it is
    correctly retired as `below_coinflip`. This is the regression case
    the task is designed to land — `dogwifcoin/1d` cleared at DA 0.516
    in the latest campaign and is statistically indistinguishable from
    coin-flip on n=265 holdout rows."""
    res = classify_slice(_slice(0.516, 0.500, 265), timeframe="1d")
    assert res["promoted"] is False
    assert res["reason"] == REASON_BELOW_COINFLIP
    assert res["min_directional_accuracy_applied"] == 0.530


def test_classify_1d_at_new_floor_is_below_coinflip():
    """The gate is strict-greater-than: equality with the floor still
    fails, mirroring the default `0.50` behaviour."""
    res = classify_slice(_slice(0.530, 0.500, 265), timeframe="1d")
    assert res["promoted"] is False
    assert res["reason"] == REASON_BELOW_COINFLIP


def test_classify_1d_above_new_floor_with_lift_promotes():
    """A 1d slice that beats the 0.530 floor AND beats its baseline
    promotes. Sanity-check the upper happy-path."""
    res = classify_slice(_slice(0.555, 0.500, 265), timeframe="1d")
    assert res["promoted"] is True
    assert res["reason"] == REASON_PROMOTED
    assert res["min_directional_accuracy_applied"] == 0.530


def test_classify_1d_above_new_floor_without_lift_is_no_lift():
    """A 1d slice that beats the floor but ties / loses to baseline
    falls into `no_lift` — same precedence as at the default floor."""
    res = classify_slice(_slice(0.555, 0.560, 265), timeframe="1d")
    assert res["promoted"] is False
    assert res["reason"] == REASON_NO_LIFT


def test_classify_1h_at_legacy_floor_unchanged():
    """The 1h floor must NOT shift — only 1d gets the override. A 1h
    slice with DA = 0.510 over baseline 0.500 keeps promoting."""
    res = classify_slice(_slice(0.510, 0.500, 250), timeframe="1h")
    assert res["promoted"] is True
    assert res["reason"] == REASON_PROMOTED
    assert res["min_directional_accuracy_applied"] == 0.50


def test_classify_omitted_timeframe_falls_back_to_default():
    """Backward compat: callers that don't pass `timeframe` see the
    legacy `0.50` floor. This is what the pre-#401 callers and the
    existing happy-path tests rely on."""
    res = classify_slice(_slice(0.510, 0.500, 250))
    assert res["promoted"] is True
    assert res["min_directional_accuracy_applied"] == 0.50


def test_block_routes_per_tf_floor_to_classifier():
    """`build_verification_block` must hand each slice the timeframe-
    specific floor. Concrete data point: a 1d slice at DA 0.516 (the
    `dogwifcoin/1d` regression case) and a 1h slice at DA 0.516 are
    both passed in. The 1h slice promotes, the 1d slice does not, and
    the per-tf override map is surfaced on the block."""
    report = {
        "timeframes": {
            "1d": {
                "status": "trained",
                "per_coin": {"dogwifcoin": _slice(0.516, 0.500, 265)},
                "pooled": None,
            },
            "1h": {
                "status": "trained",
                "per_coin": {"dogwifcoin": _slice(0.516, 0.500, 250)},
                "pooled": None,
            },
        },
    }
    v = build_verification_block(report, ["dogwifcoin"])
    # 1h promoted, 1d retired.
    by_tf = {(s["timeframe"], s["coin"]): s for s in v["per_slice"]
             if s["kind"] == "per_coin"}
    assert by_tf[("1h", "dogwifcoin")]["promoted"] is True
    assert by_tf[("1h", "dogwifcoin")]["reason"] == REASON_PROMOTED
    assert by_tf[("1h", "dogwifcoin")]["min_directional_accuracy_applied"] == 0.50
    assert by_tf[("1d", "dogwifcoin")]["promoted"] is False
    assert by_tf[("1d", "dogwifcoin")]["reason"] == REASON_BELOW_COINFLIP
    assert by_tf[("1d", "dogwifcoin")]["min_directional_accuracy_applied"] == 0.530
    # Block surfaces the per-tf map.
    assert v["min_directional_accuracy_per_tf"] == {"1d": 0.530}
    assert v["min_directional_accuracy"] == 0.50
