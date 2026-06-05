"""On-real-data verification gate for the slow-loop trainer.

For every per-coin and pooled slice produced by `train_one_slice`, this
module decides whether the slice is **promotable** — i.e. the LightGBM
fit beat the multinomial-logistic baseline on the walk-forward holdout
AND beat coin-flip AND saw enough holdout rows for the comparison to
mean anything.

The output is written to the top-level `verification` block of
`report.json` so downstream consumers (the dashboard card, the
notifier) can surface a single pass/fail signal without re-deriving it.

Per the training contract: NO falsified metrics. Every number here is
read straight from the per-slice report assembled by `train_one_slice`,
which sourced it from the walk-forward folds. We never substitute,
smooth, or interpolate.
"""
from __future__ import annotations

from typing import Optional

# Promotion gate constants. Documented in TRAINING_CONTRACT.md §6.
MIN_HOLDOUT_ROWS = 200
MIN_DIRECTIONAL_ACCURACY = 0.50

# Task #401 — per-timeframe override for `MIN_DIRECTIONAL_ACCURACY`.
#
# The default `0.50` floor is exactly coin-flip. For tradeable timeframes
# whose holdout sample is small enough that a 0.50 DA estimate has 1σ
# noise > the lift we can plausibly observe, the strict-`> 0.50` gate
# turns into a coin-flip on whether a real-edge slice gets promoted at
# all. The task #379 root cause analysis quantified this:
#
#   - 1d per-coin slices have ~365 rows of 1-year history. The 5-fold
#     expanding-window split yields ~265 holdout rows per slice. At
#     `p = 0.5, n = 265` the 1σ noise on a directional-accuracy estimate
#     is `sqrt(0.5 * 0.5 / 265) ≈ 0.0307`. So a slice with a real
#     `+0.03 DA` edge over coin-flip lands inside the noise band of
#     the threshold ~50 % of the time.
#   - In the latest campaign only `dogwifcoin/1d` cleared the gate, by
#     `0.516 - 0.500 = 0.016 DA` — i.e. 0.5σ above coin-flip. Whether a
#     given 1d slice with a real edge clears the gate is essentially
#     coin-flip on the inner-Optuna seed.
#
# The fix: raise the 1d floor to 1σ above coin-flip (0.530). A slice
# that promotes under this floor is, with ~84 % posterior confidence
# under the binomial null, doing better than coin-flip. The number is
# strictly TIGHTER than 0.50 — it does not loosen the bar. Any 1d slice
# whose DA is in [0.500, 0.530] (the region where the old gate would
# have promoted noise) is now correctly retired as `below_coinflip`.
#
# 1d is the only timeframe that gets an override:
#   - 1h/2h/6h holdouts are an order of magnitude larger after the
#     #379 multi-bar label fix; the default 0.50 floor's 1σ noise is
#     well below the lift those slices can realistically show.
#   - 5m has even more rows.
#   - 1m is not tradeable.
#
# Operators reading the verification block see the per-tf map under
# `min_directional_accuracy_per_tf`; the per-slice verdict carries
# `min_directional_accuracy_applied` so the floor that decided the
# verdict is auditable from the report alone.
MIN_DIRECTIONAL_ACCURACY_PER_TF: dict[str, float] = {
    "1d": 0.530,
}


def min_directional_accuracy_for(timeframe: Optional[str]) -> float:
    """Resolve the directional-accuracy floor for `timeframe`.

    Falls back to the default `MIN_DIRECTIONAL_ACCURACY` when the
    timeframe is unknown or has no override. The returned value is the
    STRICT-greater-than threshold the gate enforces: a slice whose DA
    is exactly equal to the floor still fails as `below_coinflip`.
    """
    if isinstance(timeframe, str):
        override = MIN_DIRECTIONAL_ACCURACY_PER_TF.get(timeframe)
        if override is not None:
            return float(override)
    return MIN_DIRECTIONAL_ACCURACY

# Task #405 / B-DIR-CALL — directional-call regression guard. A slice
# whose calibrated head emits a directional ("up" or "down") call on
# more than this share of holdout rows is structurally suspect: it
# cannot be honestly distinguishing a "stable" outcome from a
# directional one. Per the brief, this MUST be a structural fail at
# training time, not a soft warning. 0.95 is well above the empirical
# stable-share floor in the active training set; the regression case
# (`pooled 1h directional_call_share=0.9988`) trips it cleanly.
MAX_DIRECTIONAL_CALL_SHARE = 0.95

# Task #405 / B-POOLED-VOCAB — pooled fallback must cover more than one
# coin. A pooled model with `len(coin_vocab) == 1` is structurally a
# per-coin model in disguise: every other coin routed through it gets
# `coin_idx = -1` (or worse, defaults to the single trained coin),
# producing predictions that are not meaningfully shared across the
# fleet. This gate refuses to promote such a pool.
MIN_POOLED_COIN_VOCAB = 2

# Reasons emitted per slice. The dashboard / notifier branches on these,
# so don't change the strings without updating the consumer.
REASON_PROMOTED = "lift"
REASON_NO_LIFT = "no_lift"
REASON_BELOW_COINFLIP = "below_coinflip"
REASON_INSUFFICIENT_SAMPLE = "insufficient_sample"
REASON_CONTRACT_FAILED = "contract_failed"
REASON_UNTRAINED = "untrained"
REASON_CADENCE_MIXED = "cadence_mixed"
# Task #405 — new structural-fail reasons.
REASON_DIRECTIONAL_CALL_REGRESSION = "directional_call_regression"
REASON_POOLED_VOCAB_TOO_SMALL = "pooled_vocab_too_small"
# Task #400 — baseline-served slices land here when promoted. Distinct
# from REASON_PROMOTED so dashboards can attribute baseline-served
# promotions separately from booster-served promotions; both still bump
# the `slices_promoted` counter so the top-level pass/fail signal
# (`coins_with_promotion`) treats them equivalently.
REASON_PROMOTED_BASELINE = "lift_baseline_served"


def manifest_blocks_promotion_for_cadence_mix(manifest: dict) -> bool:
    """Task #317 — return True iff the slice manifest was assembled from
    rows of more than one native cadence AND no operator-approved
    mitigation is recorded. The verification gate must refuse to promote
    such a slice: a daily-cadence row silently feeding a 5m bucket close
    is the contamination this fix targets, and once `cadence_mixed=True`
    the slice's metrics are no longer trustworthy as a directional signal.
    """
    if not isinstance(manifest, dict):
        return False
    prov = manifest.get("provenance")
    base = prov if isinstance(prov, dict) else manifest
    if not base.get("cadence_mixed"):
        return False
    mitigation = base.get("cadence_mitigation")
    return mitigation is None or mitigation in ("", "none")


def _is_finite(x) -> bool:
    try:
        return x is not None and float("-inf") < float(x) < float("inf")
    except (TypeError, ValueError):
        return False


def _slice_directional_call_share(slice_report: dict) -> Optional[float]:
    """Task #405 / B-DIR-CALL — pull `directional_call_share` from a
    slice report. The training pipeline writes this number into both the
    slice's metrics block and into the per-slice manifest. We accept
    either; nullable to handle older reports that pre-date the field.
    """
    if not isinstance(slice_report, dict):
        return None
    metrics = slice_report.get("metrics") or {}
    if isinstance(metrics, dict) and "directional_call_share" in metrics:
        try:
            return float(metrics["directional_call_share"])
        except (TypeError, ValueError):
            return None
    manifest = slice_report.get("manifest") or {}
    if isinstance(manifest, dict) and "directional_call_share" in manifest:
        try:
            return float(manifest["directional_call_share"])
        except (TypeError, ValueError):
            return None
    if "directional_call_share" in slice_report:
        try:
            return float(slice_report["directional_call_share"])
        except (TypeError, ValueError):
            return None
    return None


def _slice_coin_vocab_size(slice_report: dict) -> Optional[int]:
    """Task #405 / B-POOLED-VOCAB — pull `len(coin_vocab)` from a slice
    report's manifest. Nullable when the report doesn't carry the
    manifest snapshot.
    """
    if not isinstance(slice_report, dict):
        return None
    manifest = slice_report.get("manifest") or {}
    if isinstance(manifest, dict):
        vocab = manifest.get("coin_vocab")
        if isinstance(vocab, (list, tuple)):
            return len(vocab)
    vocab = slice_report.get("coin_vocab")
    if isinstance(vocab, (list, tuple)):
        return len(vocab)
    return None


def classify_slice(
    slice_report: dict,
    kind: str = "per_coin",
    timeframe: Optional[str] = None,
) -> dict:
    """Decide the promotion verdict for one (coin, timeframe) slice.

    `slice_report` is a single per-coin (or pooled) entry from a
    timeframe report — the dict produced by `train_one_slice`.
    `timeframe` is consumed to resolve the per-tf directional-accuracy
    floor (see `MIN_DIRECTIONAL_ACCURACY_PER_TF`). When omitted, the
    default `MIN_DIRECTIONAL_ACCURACY` is used. Returns:

        {
            "promoted": bool,
            "reason": str,                  # one of REASON_* above
            "directional_accuracy": float,
            "baseline_directional_accuracy": float,
            "lift": float,                  # da - baseline
            "holdout_rows": int,            # sum of fold n_test
            "min_directional_accuracy_applied": float,  # per-tf floor
        }
    """
    min_da = min_directional_accuracy_for(timeframe)

    status = slice_report.get("status")
    if status not in ("trained", "trained_per_coin"):
        return {
            "promoted": False, "reason": REASON_UNTRAINED,
            "directional_accuracy": None,
            "baseline_directional_accuracy": None,
            "lift": None, "holdout_rows": 0,
            "min_directional_accuracy_applied": min_da,
        }

    # Task #317 — cadence-mix gate. Even a slice with strong lift gets
    # blocked if its rows came from more than one native cadence and
    # the operator did not record an explicit mitigation. This is the
    # second line of defence behind the resampler's `min_input_cadence_ms`
    # cap; it catches any slice whose row stream slipped past the
    # per-row quarantine.
    if manifest_blocks_promotion_for_cadence_mix(slice_report):
        return {
            "promoted": False, "reason": REASON_CADENCE_MIXED,
            "directional_accuracy": slice_report.get("metrics", {}).get(
                "directional_accuracy"
            ),
            "baseline_directional_accuracy": slice_report.get(
                "baseline_metrics", {}
            ).get("directional_accuracy"),
            "lift": None,
            "holdout_rows": sum(
                int(f.get("n_test", 0) or 0)
                for f in (slice_report.get("fold_metrics") or [])
            ),
            "min_directional_accuracy_applied": min_da,
        }

    metrics = slice_report.get("metrics") or {}
    baseline = slice_report.get("baseline_metrics") or {}
    da = metrics.get("directional_accuracy")
    base_da = baseline.get("directional_accuracy")

    folds = slice_report.get("fold_metrics") or []
    holdout_rows = sum(int(f.get("n_test", 0) or 0) for f in folds)

    if not _is_finite(da) or not _is_finite(base_da):
        return {
            "promoted": False, "reason": REASON_UNTRAINED,
            "directional_accuracy": da,
            "baseline_directional_accuracy": base_da,
            "lift": None, "holdout_rows": holdout_rows,
            "min_directional_accuracy_applied": min_da,
        }

    da_f = float(da)
    base_f = float(base_da)
    lift = da_f - base_f

    # Task #405 / B-DIR-CALL — structural-fail check BEFORE the lift /
    # coin-flip checks. A slice that emits a directional call on >95 %
    # of holdout rows cannot be honestly distinguishing classes; refuse
    # to promote it regardless of headline accuracy. The audit's
    # specific regression — pooled 1h with directional_call_share=0.9988
    # — used to slip through because the lift check passed.
    dir_call_share = _slice_directional_call_share(slice_report)
    if dir_call_share is not None and dir_call_share >= MAX_DIRECTIONAL_CALL_SHARE:
        return {
            "promoted": False, "reason": REASON_DIRECTIONAL_CALL_REGRESSION,
            "directional_accuracy": da_f,
            "baseline_directional_accuracy": base_f,
            "lift": lift, "holdout_rows": holdout_rows,
            "directional_call_share": dir_call_share,
            "min_directional_accuracy_applied": min_da,
        }

    # Task #405 / B-POOLED-VOCAB — refuse to promote a pooled slice that
    # was trained on a single coin. Per-coin slices are exempt: their
    # vocab is intentionally [their coin].
    if kind == "pooled":
        vocab_size = _slice_coin_vocab_size(slice_report)
        if vocab_size is not None and vocab_size < MIN_POOLED_COIN_VOCAB:
            return {
                "promoted": False, "reason": REASON_POOLED_VOCAB_TOO_SMALL,
                "directional_accuracy": da_f,
                "baseline_directional_accuracy": base_f,
                "lift": lift, "holdout_rows": holdout_rows,
                "coin_vocab_size": vocab_size,
                "min_directional_accuracy_applied": min_da,
            }

    # Task #400 — when the slice's served predictor IS the baseline,
    # the lift check (`da > baseline_da`) is meaningless: the trainer
    # copies baseline metrics into `metrics`, so the served-head DA
    # *equals* the baseline DA by construction. Skip the lift check
    # for these slices and promote on the same coin-flip + sample-size
    # floor a booster-served slice has to clear. The verdict carries
    # REASON_PROMOTED_BASELINE so dashboards / failure-analysis can
    # bucket baseline-served promotions separately from booster-served
    # promotions.
    served_kind = (
        (slice_report.get("manifest") or {}).get("served_predictor_kind")
        or slice_report.get("served_predictor_kind")
        or "lightgbm"
    )

    if holdout_rows < MIN_HOLDOUT_ROWS:
        reason = REASON_INSUFFICIENT_SAMPLE
        promoted = False
    elif da_f <= min_da:
        reason = REASON_BELOW_COINFLIP
        promoted = False
    elif served_kind == "baseline":
        # Baseline IS the served head — no lift comparison to perform.
        # Sample size and coin-flip floors above already passed.
        reason = REASON_PROMOTED_BASELINE
        promoted = True
    elif da_f <= base_f:
        reason = REASON_NO_LIFT
        promoted = False
    else:
        reason = REASON_PROMOTED
        promoted = True

    return {
        "promoted": promoted, "reason": reason,
        "directional_accuracy": da_f,
        "baseline_directional_accuracy": base_f,
        "lift": lift, "holdout_rows": holdout_rows,
        "min_directional_accuracy_applied": min_da,
        "served_predictor_kind": served_kind,
    }


def build_verification_block(report: dict, active_coins: list[str]) -> dict:
    """Walk every (coin, timeframe) slice in `report["timeframes"]` and
    produce the top-level `verification` block.

    `passed` is true only when EVERY active coin has at least one
    promoted slice. A timeframe whose contract failed (leakage or
    provenance) contributes `contract_failed` slices for every active
    coin so the operator sees the full attribution.
    """
    timeframes = report.get("timeframes") or {}

    counts = {
        "slices_promoted": 0,
        "slices_no_lift": 0,
        "slices_below_coinflip": 0,
        "slices_insufficient_sample": 0,
        "slices_contract_failed": 0,
        "slices_untrained": 0,
        "slices_cadence_mixed": 0,
        # Task #405 — surface the new structural-fail buckets so
        # operators see attribution at a glance.
        "slices_directional_call_regression": 0,
        "slices_pooled_vocab_too_small": 0,
        # Task #400 — baseline-served promotions are counted both into
        # `slices_promoted` (so the top-level pass signal is unchanged)
        # and into this dedicated bucket, so the dashboard can show
        # how many of the promotions were on the baseline head.
        "slices_promoted_baseline": 0,
    }
    per_slice: list[dict] = []
    promoted_by_coin: dict[str, int] = {c: 0 for c in active_coins}

    for tf, tf_report in timeframes.items():
        if not isinstance(tf_report, dict):
            continue
        contract_failed = tf_report.get("status") == "leakage_audit_failed" or (
            (tf_report.get("provenance") or {}).get("rejected_synthetic") is True
        )
        per_coin = tf_report.get("per_coin") or {}
        pooled = tf_report.get("pooled")

        # Per-coin slices. Treat coins missing from per_coin as untrained.
        # Task #401 — pre-resolve the per-tf floor so every verdict for
        # this timeframe (including contract-failed and untrained shells)
        # carries the same `min_directional_accuracy_applied` value.
        tf_min_da = min_directional_accuracy_for(tf)

        for coin in active_coins:
            slice_rep = per_coin.get(coin)
            if contract_failed:
                verdict = {
                    "promoted": False, "reason": REASON_CONTRACT_FAILED,
                    "directional_accuracy": None,
                    "baseline_directional_accuracy": None,
                    "lift": None, "holdout_rows": 0,
                    "min_directional_accuracy_applied": tf_min_da,
                }
            elif slice_rep is None:
                verdict = {
                    "promoted": False, "reason": REASON_UNTRAINED,
                    "directional_accuracy": None,
                    "baseline_directional_accuracy": None,
                    "lift": None, "holdout_rows": 0,
                    "min_directional_accuracy_applied": tf_min_da,
                }
            else:
                verdict = classify_slice(slice_rep, timeframe=tf)
            verdict.update({
                "coin": coin, "timeframe": tf, "kind": "per_coin",
            })
            per_slice.append(verdict)
            _bump(counts, verdict["reason"])
            if verdict["promoted"]:
                promoted_by_coin[coin] = promoted_by_coin.get(coin, 0) + 1

        # Pooled slice (one per timeframe). Pooled promotion gives the
        # coin a fallback even if its per-coin slice was untrainable —
        # mirror that here so a healthy pool counts toward each coin.
        if pooled is not None:
            if contract_failed:
                pooled_verdict = {
                    "promoted": False, "reason": REASON_CONTRACT_FAILED,
                    "directional_accuracy": None,
                    "baseline_directional_accuracy": None,
                    "lift": None, "holdout_rows": 0,
                    "min_directional_accuracy_applied": tf_min_da,
                }
            else:
                pooled_verdict = classify_slice(pooled, kind="pooled", timeframe=tf)
            pooled_verdict.update({
                "coin": "__pooled__", "timeframe": tf, "kind": "pooled",
            })
            per_slice.append(pooled_verdict)
            _bump(counts, pooled_verdict["reason"])
            if pooled_verdict["promoted"]:
                # A promoted pool serves every active coin.
                for c in active_coins:
                    promoted_by_coin[c] = promoted_by_coin.get(c, 0) + 1

    coins_with_promotion = [c for c, n in promoted_by_coin.items() if n > 0]
    coins_without_promotion = [c for c in active_coins if promoted_by_coin.get(c, 0) == 0]
    passed = len(coins_without_promotion) == 0 and counts["slices_promoted"] > 0

    return {
        "passed": passed,
        "min_holdout_rows": MIN_HOLDOUT_ROWS,
        "min_directional_accuracy": MIN_DIRECTIONAL_ACCURACY,
        # Task #401 — surface the per-tf override map so dashboards /
        # downstream consumers can render the correct floor next to a
        # 1d slice without having to re-import the constant.
        "min_directional_accuracy_per_tf": dict(MIN_DIRECTIONAL_ACCURACY_PER_TF),
        "active_coins": list(active_coins),
        "coins_with_promotion": coins_with_promotion,
        "coins_without_promotion": coins_without_promotion,
        "promoted_by_coin": promoted_by_coin,
        "counts": counts,
        "per_slice": per_slice,
    }


def _bump(counts: dict, reason: str) -> None:
    key = {
        REASON_PROMOTED: "slices_promoted",
        REASON_NO_LIFT: "slices_no_lift",
        REASON_BELOW_COINFLIP: "slices_below_coinflip",
        REASON_INSUFFICIENT_SAMPLE: "slices_insufficient_sample",
        REASON_CONTRACT_FAILED: "slices_contract_failed",
        REASON_UNTRAINED: "slices_untrained",
        REASON_CADENCE_MIXED: "slices_cadence_mixed",
        REASON_DIRECTIONAL_CALL_REGRESSION: "slices_directional_call_regression",
        REASON_POOLED_VOCAB_TOO_SMALL: "slices_pooled_vocab_too_small",
    }.get(reason)
    if key:
        counts[key] = counts.get(key, 0) + 1
    # Task #400 — baseline-served promotions ALSO bump
    # `slices_promoted` so existing dashboards / downstream consumers
    # see the unified promotion total. The `slices_promoted_baseline`
    # bucket is the attribution view; the union (`slices_promoted`)
    # remains the operational signal.
    if reason == REASON_PROMOTED_BASELINE:
        counts["slices_promoted"] = counts.get("slices_promoted", 0) + 1
        counts["slices_promoted_baseline"] = (
            counts.get("slices_promoted_baseline", 0) + 1
        )
