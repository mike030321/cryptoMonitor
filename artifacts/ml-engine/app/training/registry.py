"""On-disk model registry with (coin, timeframe, version) granularity.

Layout:
    artifacts/ml-engine/models/
        {coin}/
            {timeframe}/
                {version}/
                    model.txt           (LightGBM booster, native format)
                    calibrators.joblib  (list of per-class IsotonicRegression)
                    manifest.json
                latest -> {version}    (text file containing latest version id)
        __pooled__/                     (special "coin" id; trained on all coins)
            {timeframe}/...
        datasets/
            {timeframe}_{version}.parquet   (labeled training frame, persisted
                                             so a training run is reproducible)
        report.json

Resolution order at inference: try (coin, tf), then fall back to
(__pooled__, tf). This satisfies the (coin, timeframe, version) granularity
requirement while letting the trainer fall back to a pooled model for any
coin that doesn't have enough history yet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import joblib
import lightgbm as lgb

REGISTRY_ROOT = Path(__file__).resolve().parents[2] / "models"
POOLED_COIN_ID = "__pooled__"


@dataclass
class ModelManifest:
    coin_id: str                       # actual coin id, or POOLED_COIN_ID
    timeframe: str
    version: str
    feature_names: list[str]
    coin_vocab: list[str]              # categorical mapping; for per-coin models this is just [coin_id]
    n_train_rows: int
    n_test_rows: int
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    threshold_pct: float
    horizon_candles: int
    # Per-class mean of `forward_return * 100` (DOWN/STABLE/UP). Used at
    # inference to compute a *real* expectedReturnPct and predictionStdPct
    # from the calibrated 3-class probability vector. Default empty list
    # so non-3-class manifests (e.g. Task #654 dual-binary-head family C)
    # don't have to fabricate this field — they ignore it.
    class_return_means_pct: list[float] = field(default_factory=list)
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    note: str = ""
    # Share of holdout predictions where argmax(prob) != STABLE — i.e. how
    # often the model emits a directional (UP/DOWN) call. Tracked so a
    # retrain that collapses this back toward 0 (e.g. by reverting label
    # thresholds) is visible in the dashboard. Optional for backwards-compat
    # with manifests trained before task #101.
    directional_call_share: Optional[float] = None
    directional_call_share_n: Optional[int] = None
    directional_call_share_source: Optional[str] = None  # "holdout" | "in_sample"
    # Phase 6 — `model_kind` lets a registry slot hold either a real LightGBM
    # booster ("lightgbm") or a feature-free empirical prior ("prior"). The
    # prior variant exists so every advertised timeframe in TIMEFRAMES has at
    # least a pooled fallback, even when there isn't enough per-coin history
    # (>= MIN_CANDLES_FOR_FEATURES candles per coin) to fit an indicator-based
    # model yet. Without it, 1h/2h/6h/1d would silently route 100% of traffic
    # to the LLM brain. The prior is replaced automatically by a real model
    # once data accumulates (latest_version picks the newest).
    model_kind: str = "lightgbm"
    # For model_kind == "prior", the deployed [DOWN, STABLE, UP] probability
    # vector. /ml/predict returns this verbatim (no feature pipeline). Empty
    # for lightgbm models — they derive probs from the booster + calibrators.
    prior_probs: list[float] = field(default_factory=list)
    # Task #400 — which trained predictor is *served* at inference for this
    # slice. Distinct from `model_kind` (which describes the registry slot
    # family): a slot can hold a real LightGBM family ("lightgbm") and yet
    # serve the multinomial-logistic baseline at inference because the
    # booster lost head-to-head on directional accuracy. Values:
    #   "lightgbm" — booster.predict + per-class isotonic (default, legacy
    #                manifests stay on this path).
    #   "baseline" — baseline.joblib (encoder + LR + priors) drives
    #                predictions; the booster file is intentionally NOT
    #                written to disk for this slot, so a future load must
    #                use the baseline payload.
    #   "prior"    — feature-free fallback (manifest.prior_probs); only
    #                emitted by `_train_prior_pooled` when no per-coin
    #                history is available yet.
    # The verification gate special-cases `served_predictor_kind == "baseline"`
    # (skips the booster-vs-baseline lift check; the baseline IS the served
    # predictor), and the auto failure-analysis pass attributes baseline-
    # served slices to a separate cohort so dashboards can tell them apart
    # from booster-served promotions.
    served_predictor_kind: str = "lightgbm"
    # Task #135 — when True, the model dir also contains `regressor.txt`,
    # a separate LightGBM regressor trained on `forward_return * 100` over
    # non-stable rows only. /ml/predict and the OOS predictor use it as the
    # source of `expectedReturnPct`, replacing the old probability-weighted
    # mean of class returns (which collapsed toward 0 because p_stable
    # dominates and the per-class means are clipped near the label
    # threshold). The classifier still drives p_up/p_down/p_stable; only
    # the magnitude estimate moves to the regressor. Optional for
    # backwards-compat with manifests trained before task #135.
    has_regression_head: bool = False
    # Diagnostic stats from the regression-head holdout (last
    # CALIBRATION_HOLDOUT_FRACTION of train rows). Kept so the dashboard
    # can show p95(|expRet|) without re-running the OOS predictor.
    regression_head_stats: Optional[dict] = None
    # Task #147 — joint diagnostic on the calibration-tail (holdout) sample
    # of the SIGN head (classifier) vs the MAGNITUDE head (regressor). Tells
    # us how often the two heads "disagree" relative to the live decision
    # gates: e.g. classifier confident but regressor below the cost floor,
    # or regressor screams but classifier near 50/50. A drop in the
    # `aligned_share` between retrains is a sign the heads are arguing and
    # the gate budget may be misallocated.
    # Schema: see `_gate_alignment_summary` in training/train.py. Optional
    # for backwards compat with older manifests.
    gates_alignment: Optional[dict] = None
    # Phase 3 — specialist ensemble. When `specialist_kind` is set, this
    # registry slot is one of the per-regime specialists (momentum,
    # mean_reversion, breakout, volatility_forecaster) instead of the
    # generic per-coin / pooled model. `regime_subset` lists the regime
    # labels this specialist was trained on (empty for the volatility
    # forecaster, which spans all regimes). `feature_schema_hash` is a
    # short content hash of `feature_names` so a downstream consumer can
    # tell at a glance whether two specialists were trained against the
    # same feature contract. `training_window` records the wall-clock
    # interval the data covered (ISO8601 strings) so a registry browser
    # can show the temporal shape of the slice without re-loading the
    # parquet snapshot. All four are optional for backwards compat with
    # pre-Phase-3 manifests.
    specialist_kind: Optional[str] = None       # momentum | mean_reversion | breakout | volatility_forecaster
    regime_subset: list[str] = field(default_factory=list)
    feature_schema_hash: Optional[str] = None
    training_window: Optional[dict] = None      # {"start": iso, "end": iso}
    # Task #317 — cadence provenance. `bars_source` is the dominant input
    # source for this slice ("candles" when read from price_candles,
    # "resampled_ticks" when synthesized from raw price_history ticks,
    # "mixed" when a slice was assembled from rows of more than one
    # source — which the trainer must refuse to promote).
    # `bars_native_cadence_ms` is the dominant native cadence in ms.
    # `bars_by_native_cadence` is a {label -> row_count} breakdown so an
    # operator can see exactly which cadences fed the slice.
    # `cadence_mixed` is True iff `bars_by_native_cadence` has more than
    # one bucket. `cadence_mitigation` records any explicit operator-
    # approved mitigation ("none" / null = not mitigated). The
    # verification gate refuses promotion when `cadence_mixed=True` and
    # no mitigation is recorded — see
    # `app.training.verification.manifest_blocks_promotion_for_cadence_mix`.
    bars_source: Optional[str] = None
    bars_native_cadence_ms: Optional[int] = None
    bars_by_native_cadence: dict = field(default_factory=dict)
    cadence_mixed: bool = False
    cadence_mitigation: Optional[str] = None
    # Task #459 — source of `threshold_pct`. "static" means the trainer
    # used the per-(coin, tf) constant from trading-frictions.json (the
    # legacy behaviour). "vol_scaled" means the constant was widened to
    # the realized-volatility floor (the static value was structurally
    # too tight for the timeframe — every bar moved more than the
    # constant — so the STABLE class was near-empty and the
    # `directional_call_share` regression gate blocked promotion). The
    # actual chosen value is in `threshold_pct`. Optional for backwards-
    # compat with manifests trained before task #459.
    threshold_pct_source: str = "static"
    # Task #633 — alias of `threshold_pct_source` exposed under the
    # field name the post-NaN-cutover MTTM verdict tooling reads. Kept
    # in sync by the manifest writer; consumers may read either field.
    threshold_source: str = "static"
    # Task #633 — UP/STABLE/DOWN row counts of `label_3class` over the
    # full training frame for this slice. Surfaces the realized class
    # distribution so future audits can read the shape of the labels
    # the slice was trained on without reverse-engineering the
    # threshold + parquet snapshot. Schema:
    #   {"DOWN": int, "STABLE": int, "UP": int, "n": int}
    # Optional for backwards-compat with manifests trained before #633.
    label_distribution: Optional[dict] = None
    # ------------------------------------------------------------------
    # Task #654 — dual-binary-head family ("C_post_cost").
    #
    # When `served_predictor_kind == "dual_binary_head"` the slice is
    # served by TWO LightGBM binary boosters instead of a single 3-class
    # multinomial booster:
    #   • long head  — P(forward_return >= +friction_threshold_pct)
    #   • short head — P(forward_return <= -friction_threshold_pct)
    # plus a Platt sigmoid per head (`platt_calibration`) and an
    # abstain threshold τ on max(p_long_cal, p_short_cal). When neither
    # head clears τ the slice abstains. The label family is `C_post_cost`
    # — the friction threshold is set to round-trip cost + a small safety
    # margin (see labels_research/producers.label_post_cost). The serving
    # path lives in `app/main.py::predict`; training (saving) is the
    # responsibility of the family-C trainer added by Task B.
    #
    # All fields are Optional and default to None so legacy 3-class
    # manifests round-trip unchanged through `ModelManifest(**dict)`.
    # `friction_threshold_pct` is the percent (e.g. 0.30 for 0.30%)
    # used to define the binary labels.
    long_model_path: Optional[str] = None       # filename inside version dir, e.g. "long_model.txt"
    short_model_path: Optional[str] = None      # filename inside version dir, e.g. "short_model.txt"
    abstain_tau: Optional[float] = None         # threshold on max(p_long, p_short) after Platt
    platt_calibration: Optional[dict] = None    # {"long": {"slope": s, "intercept": b}, "short": {...}}
    friction_threshold_pct: Optional[float] = None  # absolute % used to define +/- binary labels
    label_family: Optional[str] = None          # e.g. "C_post_cost" — informational, drives /predict label_family field
    # ------------------------------------------------------------------
    # Task #657 — paper-trading B2 isotonic recalibration.
    #
    # When `calibration_method == "isotonic"`, the dual-binary-head
    # serving path applies a per-head sklearn IsotonicRegression
    # transform instead of the Platt sigmoid. The threshold arrays
    # come straight from the fitted estimator's `X_thresholds_` and
    # `y_thresholds_` attributes (already sorted ascending in x), so
    # serving needs nothing more than `numpy.interp(clip(raw, x[0],
    # x[-1]), x, y)` to reproduce `IsotonicRegression(out_of_bounds=
    # "clip").transform(raw)` exactly. Storing the threshold arrays in
    # the manifest itself (pure JSON) means the serving path never
    # needs joblib to load a calibrator. Optional joblib copies may be
    # written alongside the booster files for fast paper-trading load
    # but the manifest is the source of truth.
    #
    # `calibration_method` defaults to "platt" so the four B-era
    # manifests (and every legacy dual-head slice) round-trip
    # unchanged through `ModelManifest(**dict)`.
    isotonic_calibration: Optional[dict] = None  # {"long": {"x_thresholds": [...], "y_values": [...]}, "short": {...}}
    calibration_method: str = "platt"            # "platt" | "isotonic" | "beta"
    # Task #659 — beta calibration: per-head sigmoid
    # p_cal = 1/(1+exp(-(a*log(p)+b*log(1-p)+c))) with eps=1e-6 clip.
    # Mirrors b3_calibration_compare._apply_beta. Mutually exclusive
    # with platt/isotonic blocks (enforced by validate()).
    # Schema: {"long":{"a","b","c"}, "short":{"a","b","c"}}
    beta_calibration: Optional[dict] = None
    # calibration_status: "trustworthy" | "under_confident_documented"
    # | "over_confident_blocked". Optional for legacy manifests.
    calibration_status: Optional[str] = None
    # Task #659 — on-disk mirror of the DB scope_constraint so
    # predict_one can refuse off-universe requests without a DB hit.
    # Schema: {"coin_id","timeframe","candidate","allowed_universe":[...]}
    scope_constraint: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        """Raise ValueError when the manifest is internally inconsistent
        with its `served_predictor_kind`. Called by `save_model` and
        `load_model` so a misconfigured slice is rejected at the
        boundary instead of producing a silent NaN at /ml/predict.
        """
        kind = self.served_predictor_kind
        if kind == "dual_binary_head":
            missing: list[str] = []
            if not self.long_model_path:
                missing.append("long_model_path")
            if not self.short_model_path:
                missing.append("short_model_path")
            if self.abstain_tau is None:
                missing.append("abstain_tau")
            if self.friction_threshold_pct is None:
                missing.append("friction_threshold_pct")
            # Task #657 — calibration contract is method-specific. The
            # legacy default is "platt" so manifests written before
            # B2 keep validating exactly as before; "isotonic"
            # requires the per-head threshold arrays AND forbids a
            # stale Platt block from sneaking through (a re-run that
            # forgot to clear the old field would otherwise silently
            # ship two calibrators).
            method = self.calibration_method or "platt"
            if method not in ("platt", "isotonic", "beta"):
                missing.append(
                    f"calibration_method (got {method!r}; "
                    "expected 'platt', 'isotonic' or 'beta')"
                )
            elif method == "platt":
                if self.isotonic_calibration is not None:
                    missing.append(
                        "isotonic_calibration (must be None when "
                        "calibration_method='platt')"
                    )
                if self.beta_calibration is not None:
                    missing.append(
                        "beta_calibration (must be None when "
                        "calibration_method='platt')"
                    )
                if not isinstance(self.platt_calibration, dict):
                    missing.append("platt_calibration")
                else:
                    for side in ("long", "short"):
                        p = self.platt_calibration.get(side)
                        if (
                            not isinstance(p, dict)
                            or "slope" not in p
                            or "intercept" not in p
                        ):
                            missing.append(
                                f"platt_calibration[{side!r}]."
                                f"(slope|intercept)"
                            )
            elif method == "isotonic":
                if self.platt_calibration is not None:
                    missing.append(
                        "platt_calibration (must be None when "
                        "calibration_method='isotonic')"
                    )
                if self.beta_calibration is not None:
                    missing.append(
                        "beta_calibration (must be None when "
                        "calibration_method='isotonic')"
                    )
                if not isinstance(self.isotonic_calibration, dict):
                    missing.append("isotonic_calibration")
                else:
                    for side in ("long", "short"):
                        iso = self.isotonic_calibration.get(side)
                        if (
                            not isinstance(iso, dict)
                            or "x_thresholds" not in iso
                            or "y_values" not in iso
                        ):
                            missing.append(
                                f"isotonic_calibration[{side!r}]."
                                f"(x_thresholds|y_values)"
                            )
                            continue
                        x = iso["x_thresholds"]
                        y = iso["y_values"]
                        if (
                            not isinstance(x, list)
                            or not isinstance(y, list)
                            or len(x) < 2
                            or len(x) != len(y)
                        ):
                            missing.append(
                                f"isotonic_calibration[{side!r}]: "
                                "x_thresholds and y_values must be "
                                "lists of equal length >= 2 (got "
                                f"len(x)={len(x) if isinstance(x, list) else 'n/a'}, "
                                f"len(y)={len(y) if isinstance(y, list) else 'n/a'})"
                            )
            else:  # beta — Task #659
                # Mutually exclusive with platt/isotonic for unambiguous dispatch.
                if self.platt_calibration is not None:
                    missing.append(
                        "platt_calibration (must be None when "
                        "calibration_method='beta')"
                    )
                if self.isotonic_calibration is not None:
                    missing.append(
                        "isotonic_calibration (must be None when "
                        "calibration_method='beta')"
                    )
                if not isinstance(self.beta_calibration, dict):
                    missing.append("beta_calibration")
                else:
                    import math as _math

                    for side in ("long", "short"):
                        b = self.beta_calibration.get(side)
                        if (
                            not isinstance(b, dict)
                            or "a" not in b
                            or "b" not in b
                            or "c" not in b
                        ):
                            missing.append(
                                f"beta_calibration[{side!r}].(a|b|c)"
                            )
                            continue
                        for coef in ("a", "b", "c"):
                            v = b.get(coef)
                            if (
                                not isinstance(v, (int, float))
                                or not _math.isfinite(float(v))
                            ):
                                missing.append(
                                    f"beta_calibration[{side!r}][{coef!r}]: "
                                    f"must be a finite float (got {v!r})"
                                )
            if missing:
                raise ValueError(
                    "dual_binary_head manifest missing required fields: "
                    f"{missing}"
                )
            # Calibration status — when present must be one of the
            # operator-vetted enum values. Unknown values are rejected
            # at the boundary so a typo can't slip through and silently
            # drive serving decisions.
            if self.calibration_status is not None and (
                self.calibration_status
                not in {
                    "trustworthy",
                    "under_confident_documented",
                    "over_confident_blocked",
                }
            ):
                raise ValueError(
                    "dual_binary_head manifest has invalid "
                    f"calibration_status={self.calibration_status!r} "
                    "(expected one of 'trustworthy', "
                    "'under_confident_documented', "
                    "'over_confident_blocked')"
                )
        # scope_constraint shape gate (manifest-level, all kinds). When
        # populated the `allowed_universe` list MUST exist and contain
        # at least one "coin:tf" string. Validated here so a malformed
        # scope can't reach the predict-one defence-in-depth check.
        if self.scope_constraint is not None:
            if not isinstance(self.scope_constraint, dict):
                raise ValueError(
                    "manifest.scope_constraint must be a dict "
                    f"(got {type(self.scope_constraint).__name__})"
                )
            allowed = self.scope_constraint.get("allowed_universe")
            if not isinstance(allowed, list) or not allowed:
                raise ValueError(
                    "manifest.scope_constraint.allowed_universe must be "
                    f"a non-empty list (got {allowed!r})"
                )
            for entry in allowed:
                if (
                    not isinstance(entry, str)
                    or ":" not in entry
                    or not entry.split(":", 1)[0]
                    or not entry.split(":", 1)[1]
                ):
                    raise ValueError(
                        "manifest.scope_constraint.allowed_universe "
                        "entries must be 'coin:timeframe' strings "
                        f"(got {entry!r})"
                    )
        elif kind == "lightgbm":
            # Legacy 3-class path requires class_return_means_pct so the
            # diluted EV path in /ml/predict has real numbers. The field
            # was relaxed to default-empty for the dual-head family;
            # enforce it here so a 3-class slice can't slip through
            # without it.
            if len(self.class_return_means_pct) != 3:
                raise ValueError(
                    "lightgbm (3-class) manifest requires class_return_means_pct "
                    f"of length 3 (got {self.class_return_means_pct!r})"
                )
        # "baseline" and "prior" have their own contracts checked
        # elsewhere (`save_model` already asserts shape) so we don't
        # double-validate here.


# Phase 3 — specialist taxonomy. The ensemble has three regime-conditioned
# directional specialists plus a single magnitude/volatility forecaster
# that the others can lean on.
SPECIALIST_REGIME_MAP: dict[str, list[str]] = {
    "momentum":              ["trending_up", "trending_down"],
    "mean_reversion":        ["range_chop", "low_vol_compression"],
    "breakout":              ["high_vol_breakout", "panic_liquidation"],
    "volatility_forecaster": [],   # special-cased: trained on ALL rows
}
SPECIALIST_KINDS: list[str] = list(SPECIALIST_REGIME_MAP.keys())


def specialist_coin_id(kind: str) -> str:
    """Registry slot id for a specialist. Mirrors POOLED_COIN_ID's
    convention so `list_coins()` callers can filter the directory tree
    without parsing manifests.
    """
    return f"__specialist_{kind}__"


def is_specialist_coin_id(coin_id: str) -> bool:
    return coin_id.startswith("__specialist_") and coin_id.endswith("__")


def _model_dir(coin_id: str, timeframe: str, version: str) -> Path:
    return REGISTRY_ROOT / coin_id / timeframe / version


def make_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def dataset_path(timeframe: str, version: str) -> Path:
    p = REGISTRY_ROOT / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{timeframe}_{version}.parquet"


def save_model(
    coin_id: str,
    timeframe: str,
    version: str,
    booster: Optional[lgb.Booster],   # None for prior-only models
    calibrators,             # list[IsotonicRegression] | None (one per class)
    manifest: ModelManifest,
    regressor: Optional[lgb.Booster] = None,
    baseline_artifact: Optional[tuple] = None,   # (encoder, lr, priors)
) -> Path:
    # Task #451 — refuse to PROMOTE a version whose feature schema still
    # advertises a forbidden (LLM-derived) column. The load-time gate
    # below already protects inference, but without a write-time gate a
    # buggy training run could still keep producing contaminated dirs
    # that the auto-prune janitor then has to clean up every cycle.
    # Failing fast here keeps `models/` bounded by construction.
    forbidden_in_save = sorted({
        c for c in (manifest.feature_names or [])
        if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)
    })
    if forbidden_in_save:
        raise ValueError(
            "refusing to promote model with forbidden feature columns "
            f"coin={coin_id} tf={timeframe} version={version} "
            f"forbidden={forbidden_in_save}"
        )
    out_dir = _model_dir(coin_id, timeframe, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Task #400 — when the served predictor is the multinomial-logistic
    # baseline (because the booster lost head-to-head on directional
    # accuracy AND the baseline cleared the verification gate on its
    # own), we persist the baseline pipeline to disk and intentionally
    # SKIP writing the booster file. The slot's served prediction at
    # inference comes straight from `baseline.joblib`; keeping a stale
    # booster on disk would only confuse later loaders. The booster
    # *was* trained — its CV metrics still live in the manifest and
    # report — we just don't ship it as the served head.
    if manifest.served_predictor_kind == "dual_binary_head":
        # Task #654 — paper-trading family C. Persists TWO LightGBM
        # binary boosters (long + short heads) under filenames recorded
        # on the manifest, plus the Platt sigmoid params + abstain τ
        # which already live on the manifest itself. We accept the two
        # boosters via the existing kwargs:
        #   - `booster`   → long head
        #   - `regressor` → short head
        # so the function signature stays backwards-compatible. The
        # caller MUST set `manifest.long_model_path` /
        # `manifest.short_model_path` to the filenames to write here
        # (validate() checks both); we don't invent paths so the
        # manifest is the single source of truth.
        manifest.validate()  # raises ValueError on missing fields
        if booster is None or regressor is None:
            raise ValueError(
                "save_model: dual_binary_head requires `booster` (long head) "
                "and `regressor` (short head); got long="
                f"{booster!r} short={regressor!r}"
            )
        booster.save_model(str(out_dir / manifest.long_model_path))
        regressor.save_model(str(out_dir / manifest.short_model_path))
        # Calibrators (per-class isotonic) are not used by this family;
        # Platt params on the manifest are the only calibration applied.
        # Skip persisting calibrators / regressor.txt / baseline.joblib —
        # we early-return BEFORE the legacy 3-class trailing writes
        # below, otherwise `regressor` (which is the SHORT head, not a
        # magnitude regressor) would also get written to `regressor.txt`
        # and load_model's `served_predictor_kind` branch would silently
        # see a stray file.
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2)
        )
        latest_pointer = REGISTRY_ROOT / coin_id / timeframe / "latest"
        latest_pointer.parent.mkdir(parents=True, exist_ok=True)
        latest_pointer.write_text(version)
        return out_dir
    elif manifest.served_predictor_kind == "baseline":
        assert baseline_artifact is not None, (
            "served_predictor_kind='baseline' requires a baseline_artifact "
            "(encoder, lr, priors) to persist"
        )
        joblib.dump(baseline_artifact, out_dir / "baseline.joblib")
    elif booster is None:
        # Prior-only models carry no booster — the manifest's `prior_probs`
        # is the deployed prediction. We still write a marker file so directory
        # walkers can tell at a glance.
        assert manifest.model_kind == "prior", (
            f"booster=None requires model_kind='prior' (got {manifest.model_kind!r})"
        )
        (out_dir / "prior.json").write_text(json.dumps({
            "prior_probs": list(manifest.prior_probs),
            "class_return_means_pct": list(manifest.class_return_means_pct),
        }, indent=2))
    else:
        booster.save_model(str(out_dir / "model.txt"))
    if calibrators is not None:
        joblib.dump(calibrators, out_dir / "calibrators.joblib")
    # Task #135 — persist the regression-head booster alongside the
    # classifier so /ml/predict and the OOS predictor can derive
    # `expectedReturnPct` directly from features instead of from the
    # diluted probability-weighted class-mean expectation.
    if regressor is not None:
        regressor.save_model(str(out_dir / "regressor.txt"))
        manifest.has_regression_head = True
    (out_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))

    latest_pointer = REGISTRY_ROOT / coin_id / timeframe / "latest"
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(version)
    return out_dir


def latest_version(coin_id: str, timeframe: str) -> Optional[str]:
    p = REGISTRY_ROOT / coin_id / timeframe / "latest"
    if not p.exists():
        return None
    v = p.read_text().strip()
    if not v:
        return None
    if not _model_dir(coin_id, timeframe, v).exists():
        return None
    return v


# Task #418 — verification verdicts are computed once per training run
# (see `app.training.verification.build_verification_block`) but live
# inference needs to read them per-slice without parsing `report.json`.
# We persist a small `verification.json` file inside each slice's version
# directory so `_resolve_for_predict` can decide whether to keep serving
# a per-coin model whose holdout DA fell into the noise band, or to
# defer to a promoted pooled fallback for the same timeframe. The file
# carries the full verdict dict so downstream consumers (e.g. the
# dashboard) can render the same reason / DA / lift the verification
# block surfaces, without round-tripping through report.json.
VERIFICATION_FILENAME = "verification.json"


def write_verification_verdict(
    coin_id: str, timeframe: str, version: str, verdict: dict,
) -> Optional[Path]:
    """Persist a single per-slice verification verdict next to the slice's
    manifest. Best-effort — returns None when the slice directory doesn't
    exist (e.g. an `untrained` verdict for a coin that never produced a
    version). Idempotent: subsequent training runs overwrite the file.
    """
    if not isinstance(verdict, dict):
        return None
    d = _model_dir(coin_id, timeframe, version)
    if not d.exists():
        return None
    p = d / VERIFICATION_FILENAME
    p.write_text(json.dumps(verdict, indent=2, default=str))
    return p


def read_verification_verdict(
    coin_id: str, timeframe: str, version: str,
) -> Optional[dict]:
    """Return the persisted verdict dict for `(coin_id, timeframe, version)`
    or None when no verdict file exists yet. Slices trained before
    task #418 (or specialist / meta slots that the verification gate
    doesn't classify) will return None — callers MUST treat that as
    "verdict unknown" and fall back to current behaviour.
    """
    p = _model_dir(coin_id, timeframe, version) / VERIFICATION_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@dataclass
class LoadedModel:
    booster: Optional[lgb.Booster]      # None for prior-only and baseline-served models
    calibrators: object | None  # list[IsotonicRegression] or None
    manifest: ModelManifest
    regressor: Optional[lgb.Booster] = None   # Task #135 — magnitude head
    # Task #400 — `(encoder, lr, priors)` triple for baseline-served slices.
    # None for booster-served (the legacy default) and prior-only slots.
    # `app/main.py` branches on `manifest.served_predictor_kind == "baseline"`
    # and falls through to this payload at inference.
    baseline_artifact: Optional[tuple] = None


class ScopeViolationError(RuntimeError):
    """Task #659 — predict_one called outside allowed_universe.
    FastAPI maps this to 422; defence-in-depth on top of the DB guard.
    """


@dataclass
class LoadedDualHeadModel:
    """Task #654 — paper-trading family C (dual_binary_head). Holds
    long+short LightGBM boosters plus the manifest's calibration and
    abstain τ. Returned by load_model; distinct from LoadedModel so
    call-sites touching .booster fail fast on the wrong kind.

    predict_one(X) → {p_long, p_short, abstain, side, confidence};
    abstain returns side="none" per Task #654 wire contract.
    """
    booster_long: lgb.Booster
    booster_short: lgb.Booster
    manifest: ModelManifest

    @staticmethod
    def _platt(raw: float, slope: float, intercept: float) -> float:
        # Standard Platt sigmoid: P = 1 / (1 + exp(slope*raw + intercept))
        # `slope` is signed (typically negative on a probability input
        # so a higher raw score → higher calibrated prob). We compute
        # in float64 and clip to a safe exponent range so NaNs from the
        # boundaries can't poison the response.
        import math as _m

        z = slope * float(raw) + intercept
        if z > 60.0:
            return 0.0
        if z < -60.0:
            return 1.0
        return 1.0 / (1.0 + _m.exp(z))

    @staticmethod
    def _apply_isotonic(
        raw: float, x_thresholds: list, y_values: list,
    ) -> float:
        """Task #657 — scalar serving form of
        `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")
        .transform([raw])[0]`.

        sklearn stores the fitted estimator's monotone interpolation
        knots in `X_thresholds_` (sorted ascending) and corresponding
        `y_thresholds_`; `out_of_bounds="clip"` then linearly
        interpolates between knots and clips raw inputs outside the
        `[X_thresholds_[0], X_thresholds_[-1]]` range to the boundary
        y values. `numpy.interp` already implements that exact
        contract: x < xp[0] returns fp[0], x > xp[-1] returns fp[-1],
        and the in-range branch is piecewise-linear over the knot
        grid. We import numpy lazily so the legacy `_platt` path
        doesn't acquire an import dependency at the call boundary.
        """
        import numpy as _np

        if not x_thresholds or not y_values:
            # Defensive — `validate()` should have already caught this,
            # but a runtime fall-through must not silently NaN-poison
            # the response.
            raise RuntimeError(
                "dual_binary_head isotonic calibration has empty "
                "threshold arrays (manifest.validate() should have "
                "caught this earlier)"
            )
        x = _np.asarray(x_thresholds, dtype=float)
        y = _np.asarray(y_values, dtype=float)
        out = float(_np.interp(float(raw), x, y))
        # Defend against numerical drift outside [0,1] (sklearn
        # already constrains the fitted y values, but a re-serialised
        # manifest could in principle have been hand-edited).
        if out < 0.0:
            return 0.0
        if out > 1.0:
            return 1.0
        return out

    # Task #659 — safe-log clip eps for _apply_beta. Research helper
    # uses 1e-7; production uses 1e-6 (documented drift; round-trip
    # test asserts agreement when called with the SAME eps).
    BETA_EPS: float = 1e-6

    @staticmethod
    def _apply_beta(p: float, a: float, b: float, c: float) -> float:
        """Beta sigmoid: p_cal = 1/(1+exp(-(a*log(p)+b*log(1-p)+c))),
        p clipped to [eps, 1-eps]. Mirrors b3_calibration_compare.
        """
        import math as _m

        eps = LoadedDualHeadModel.BETA_EPS
        p_c = max(eps, min(1.0 - eps, float(p)))
        log_p = _m.log(p_c)
        log_1_p = _m.log(1.0 - p_c)
        z = float(a) * log_p + float(b) * log_1_p + float(c)
        # Sigmoid form: 1/(1+exp(-z)). Mirror the bounded-z guard from
        # `_platt` so an extreme `z` returns the boundary directly.
        if z > 60.0:
            return 1.0
        if z < -60.0:
            return 0.0
        return 1.0 / (1.0 + _m.exp(-z))

    def _enforce_scope(
        self,
        coin_id: Optional[str],
        timeframe: Optional[str],
    ) -> None:
        """Task #659 — refuse off-universe predict_one calls. Legacy
        manifests (scope_constraint=None or empty) are unrestricted.
        """
        sc = self.manifest.scope_constraint
        if sc is None:
            return
        allowed = sc.get("allowed_universe")
        if not isinstance(allowed, list) or not allowed:
            return
        if coin_id is None or timeframe is None:
            raise ScopeViolationError(
                f"predict_one refused: scope-pinned manifest but caller "
                f"did not pass coin_id/timeframe (allowed={allowed!r}, "
                f"model={self.manifest.coin_id}/"
                f"{self.manifest.timeframe}/{self.manifest.version})"
            )
        key = f"{coin_id}:{timeframe}"
        if key not in allowed:
            raise ScopeViolationError(
                f"predict_one refused: ({coin_id!r},{timeframe!r}) "
                f"outside allowed_universe={allowed!r}"
            )

    def predict_one(
        self,
        X,
        *,
        coin_id: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> dict:
        # Decision rule (τ on max(p_long_cal, p_short_cal), side=argmax,
        # confidence scaled by headroom above τ) is identical across
        # calibrators; only the raw→calibrated transform changes.
        self._enforce_scope(coin_id, timeframe)

        method = self.manifest.calibration_method or "platt"
        if self.manifest.abstain_tau is None:
            raise RuntimeError(
                "dual_binary_head model loaded without abstain_tau "
                "(manifest.validate() should have caught this earlier)"
            )
        if method == "platt" and self.manifest.platt_calibration is None:
            raise RuntimeError(
                "dual_binary_head model loaded with calibration_method="
                "'platt' but platt_calibration is None "
                "(manifest.validate() should have caught this earlier)"
            )
        if method == "isotonic" and self.manifest.isotonic_calibration is None:
            raise RuntimeError(
                "dual_binary_head model loaded with calibration_method="
                "'isotonic' but isotonic_calibration is None "
                "(manifest.validate() should have caught this earlier)"
            )
        if method == "beta" and self.manifest.beta_calibration is None:
            raise RuntimeError(
                "dual_binary_head model loaded with calibration_method="
                "'beta' but beta_calibration is None "
                "(manifest.validate() should have caught this earlier)"
            )
        raw_long = float(
            self.booster_long.predict(X, num_iteration=self.booster_long.best_iteration).flatten()[0]
        )
        raw_short = float(
            self.booster_short.predict(X, num_iteration=self.booster_short.best_iteration).flatten()[0]
        )
        if method == "isotonic":
            il = self.manifest.isotonic_calibration["long"]
            is_ = self.manifest.isotonic_calibration["short"]
            p_long = self._apply_isotonic(
                raw_long, il["x_thresholds"], il["y_values"],
            )
            p_short = self._apply_isotonic(
                raw_short, is_["x_thresholds"], is_["y_values"],
            )
        elif method == "beta":
            bl = self.manifest.beta_calibration["long"]
            bs = self.manifest.beta_calibration["short"]
            p_long = self._apply_beta(
                raw_long, float(bl["a"]), float(bl["b"]), float(bl["c"]),
            )
            p_short = self._apply_beta(
                raw_short, float(bs["a"]), float(bs["b"]), float(bs["c"]),
            )
        else:
            pl = self.manifest.platt_calibration["long"]
            ps = self.manifest.platt_calibration["short"]
            p_long = self._platt(raw_long, float(pl["slope"]), float(pl["intercept"]))
            p_short = self._platt(raw_short, float(ps["slope"]), float(ps["intercept"]))
        tau = float(self.manifest.abstain_tau)
        p_max = max(p_long, p_short)
        if p_max < tau:
            return {
                "p_long": p_long, "p_short": p_short,
                "abstain": True, "side": "none",
                "confidence": 0.0,
            }
        side = "long" if p_long >= p_short else "short"
        # Confidence = how far above τ the winning head sits, scaled to
        # [0,1] by the headroom available (1.0 − τ). At p == τ the
        # response is 0; at p == 1 it is 1.
        headroom = max(1e-9, 1.0 - tau)
        confidence = max(0.0, min(1.0, (p_max - tau) / headroom))
        return {
            "p_long": p_long, "p_short": p_short,
            "abstain": False, "side": side,
            "confidence": confidence,
        }


def load_model(
    coin_id: str,
    timeframe: str,
    version: Optional[str] = None,
    *,
    requested_for: Optional[tuple] = None,
):
    """Load the model at (coin_id, timeframe, version).

    Task #659 — optional `requested_for=(serving_coin, serving_tf)`
    triggers scope refusal: returns None when the manifest's
    allowed_universe excludes that pair (same surface as missing).

    Returns LoadedModel | LoadedDualHeadModel | None. Callers must
    branch on isinstance / served_predictor_kind before touching
    kind-specific attrs.
    """
    v = version or latest_version(coin_id, timeframe)
    if not v:
        return None
    d = _model_dir(coin_id, timeframe, v)
    if not d.exists():
        return None
    manifest_dict = json.loads((d / "manifest.json").read_text())
    manifest = ModelManifest(**manifest_dict)
    # Task #365 — Quant-Only: refuse models with LLM-derived features.
    # Treat as missing (return None) so resolve_model falls through.
    forbidden = [
        c for c in manifest.feature_names
        if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)
    ]
    if forbidden:
        import logging
        logging.getLogger(__name__).warning(
            "load_model_rejected_forbidden_features "
            "coin=%s tf=%s version=%s forbidden=%s",
            coin_id, timeframe, v, sorted(forbidden),
        )
        return None

    # Task #659 — load-time scope refusal: return None when the
    # requested (coin,tf) isn't in allowed_universe.
    if requested_for is not None and manifest.scope_constraint is not None:
        allowed_uni = manifest.scope_constraint.get("allowed_universe")
        if isinstance(allowed_uni, list) and allowed_uni:
            req_coin, req_tf = requested_for
            req_key = f"{req_coin}:{req_tf}"
            if req_key not in allowed_uni:
                import logging
                logging.getLogger(__name__).warning(
                    "load_model_refused_out_of_scope "
                    "model=%s/%s/%s requested_for=%s allowed=%s",
                    coin_id, timeframe, v, req_key, allowed_uni,
                )
                return None

    if manifest.served_predictor_kind == "dual_binary_head":
        # Task #654 — paper-trading family C. Validate manifest before
        # disk reads so a partial slice fails clean (not NaN downstream).
        try:
            manifest.validate()
        except ValueError as exc:
            import logging

            logging.getLogger(__name__).warning(
                "load_model_rejected_invalid_dual_head "
                "coin=%s tf=%s version=%s error=%s",
                coin_id, timeframe, v, str(exc),
            )
            return None
        long_path = d / manifest.long_model_path
        short_path = d / manifest.short_model_path
        if not long_path.exists() or not short_path.exists():
            return None
        return LoadedDualHeadModel(
            booster_long=lgb.Booster(model_file=str(long_path)),
            booster_short=lgb.Booster(model_file=str(short_path)),
            manifest=manifest,
        )

    booster: Optional[lgb.Booster]
    regressor: Optional[lgb.Booster] = None
    baseline_artifact: Optional[tuple] = None
    if manifest.served_predictor_kind == "baseline":
        # Task #400 — baseline-served slot. The booster file was
        # intentionally not written by `save_model`; load the persisted
        # baseline pipeline (encoder, lr, priors) instead. Calibrators
        # were fit on the BASELINE's holdout predictions and are still
        # applied at inference (see `app/main.py`).
        booster = None
        baseline_path = d / "baseline.joblib"
        if not baseline_path.exists():
            return None
        baseline_artifact = joblib.load(baseline_path)
        calib_path = d / "calibrators.joblib"
        calibrators = joblib.load(calib_path) if calib_path.exists() else None
        reg_path = d / "regressor.txt"
        if reg_path.exists():
            regressor = lgb.Booster(model_file=str(reg_path))
    elif manifest.model_kind == "prior":
        # Prior-only: no booster on disk. Predictions come straight from
        # manifest.prior_probs in main.py.
        booster = None
        calibrators = None
    else:
        booster = lgb.Booster(model_file=str(d / "model.txt"))
        calib_path = d / "calibrators.joblib"
        calibrators = joblib.load(calib_path) if calib_path.exists() else None
        reg_path = d / "regressor.txt"
        if reg_path.exists():
            regressor = lgb.Booster(model_file=str(reg_path))
    return LoadedModel(
        booster=booster, calibrators=calibrators, manifest=manifest,
        regressor=regressor, baseline_artifact=baseline_artifact,
    )


def resolve_model(
    coin_id: str,
    timeframe: str,
    is_quarantined: Optional[Callable[[str, str, str], bool]] = None,
) -> Optional[LoadedModel]:
    """Try the per-coin model first, then fall back to the pooled model.

    `(coin, timeframe, version)` is the canonical granularity; pooled is the
    explicit fallback when a coin doesn't have enough history yet.

    Task #232 — when `is_quarantined(coin_id, timeframe, version)` is
    provided, ANY version it returns True for is skipped during selection.
    For the per-coin slot we walk versions newest-first and pick the first
    non-quarantined one (so we naturally fall back to the previous champion
    when the current `latest` pointer is quarantined). If every per-coin
    version is quarantined we then walk pooled versions the same way.
    A registry row whose state == "quarantined" must never be selected for
    a live trade decision.
    """
    is_q = is_quarantined or (lambda _c, _t, _v: False)

    def _pick(slot: str) -> Optional[LoadedModel]:
        # Fast path: the `latest` pointer is fine in the common case where
        # nothing is quarantined. Falling through to the descending walk
        # only happens after a quarantine event.
        #
        # Task #405 / B-PRED-500 mirror fix: if the `latest` pointer
        # itself fails to load (e.g. forbidden-prefix feature reject),
        # `load_model` returns None — DO NOT bail; walk older versions.
        latest = latest_version(slot, timeframe)
        if latest and not is_q(slot, timeframe, latest):
            m = load_model(slot, timeframe, latest)
            if m is not None:
                return m
        for v in reversed(list_versions(slot, timeframe)):
            if is_q(slot, timeframe, v):
                continue
            if v == latest:
                continue  # already tried above
            m = load_model(slot, timeframe, v)
            if m is not None:
                return m
        return None

    m = _pick(coin_id)
    if m is not None:
        return m
    if coin_id != POOLED_COIN_ID:
        return _pick(POOLED_COIN_ID)
    return None


def list_versions(coin_id: str, timeframe: str) -> list[str]:
    tf_dir = REGISTRY_ROOT / coin_id / timeframe
    if not tf_dir.exists():
        return []
    versions = [
        p.name for p in tf_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists()
    ]
    versions.sort()
    return versions


def audit_active_slots(
    coin_ids: list[str],
    timeframes: list[str],
    is_quarantined: Optional[Callable[[str, str, str], bool]] = None,
) -> dict:
    """Task #376 — assert every active (coin, tf) slot resolves to a
    fresh, non-archived manifest.

    For every (coin, tf) pair the api-server lists as active, attempt
    `resolve_model(coin, tf, is_quarantined=...)`. A slot that resolves
    to None means the per-coin model is missing AND the pooled fallback
    is missing too — i.e. the agent for that coin/timeframe is silently
    paused (every prediction will 503 and route to the LLM brain, or be
    skipped entirely).
    `archive_contaminated_models.py`-style archive sweeps rename the
    `latest` pointer out of the way, so the only way the slot stays
    "live" after a retrain is for `run_training` to have written a new
    manifest. This audit catches the regression where the archival
    happened but the retrain did NOT replace the slot.

    Returns a dict with a stable schema so the training report can carry
    it forward unchanged:
        {
            "ok": bool,                        # True iff every slot resolves
            "n_checked": int,
            "missing": [
                {"coin_id": str, "timeframe": str, "slot_state": "..."},
                ...
            ],
        }
    `slot_state` is a short diagnostic string drawn from the on-disk
    layout so an operator can tell at a glance WHY the slot is empty:
      * "no_per_coin_dir"     — `<root>/<coin>/<tf>/` doesn't exist
      * "no_latest_pointer"   — dir exists but no `latest` file
      * "latest_dangling"     — `latest` points at a missing version
      * "all_versions_archived" — every version dir was archived but
                                  pooled fallback also missing
      * "pooled_fallback_missing" — per-coin missing AND pooled missing
    """
    is_q = is_quarantined or (lambda _c, _t, _v: False)

    def _slot_fresh(slot: str, tf: str) -> bool:
        """Slot is considered fresh iff the `latest` pointer exists,
        names an extant version dir whose manifest loads, and that
        version is not quarantined. We deliberately do NOT walk
        `list_versions` here the way `resolve_model` does: an archive
        sweep typically renames the `latest` pointer out of the way but
        leaves the version dirs in place, so `list_versions` would still
        find a stale archived dir and let the resolver "pass". The
        whole point of this audit is to fail in exactly that case."""
        v = latest_version(slot, tf)
        if not v:
            return False
        if is_q(slot, tf, v):
            return False
        return load_model(slot, tf, v) is not None

    missing: list[dict] = []
    for coin_id in coin_ids:
        for tf in timeframes:
            if _slot_fresh(coin_id, tf):
                continue
            if coin_id != POOLED_COIN_ID and _slot_fresh(POOLED_COIN_ID, tf):
                continue
            # Diagnose why the slot is empty.
            tf_dir = REGISTRY_ROOT / coin_id / tf
            pooled_present = _slot_fresh(POOLED_COIN_ID, tf)
            if not tf_dir.exists():
                state = "no_per_coin_dir"
            else:
                latest_ptr = tf_dir / "latest"
                if not latest_ptr.exists():
                    if list_versions(coin_id, tf):
                        state = "all_versions_archived"
                    else:
                        state = "no_latest_pointer"
                else:
                    pointed = latest_ptr.read_text().strip()
                    if pointed and not _model_dir(coin_id, tf, pointed).exists():
                        state = "latest_dangling"
                    else:
                        state = "all_versions_archived"
            if not pooled_present:
                state = f"{state}+pooled_fallback_missing"
            missing.append({
                "coin_id": coin_id,
                "timeframe": tf,
                "slot_state": state,
            })
    return {
        "ok": not missing,
        "n_checked": len(coin_ids) * len(timeframes),
        "missing": missing,
    }


def list_coins() -> list[str]:
    if not REGISTRY_ROOT.exists():
        return []
    return sorted([
        p.name for p in REGISTRY_ROOT.iterdir()
        if p.is_dir() and p.name not in {"datasets"}
    ])


# Task #451 — auto-prune of contaminated model artifacts.
#
# A one-shot remediation script (`scripts/archive_contaminated_models.py`)
# previously cleaned up 1117 versions whose feature schemas advertised
# LLM-derived columns. Without an automated janitor, every fresh
# contaminated promotion (or any legacy dir that wasn't caught) would
# silently re-accumulate on disk and re-trigger
# `load_model_rejected_forbidden_features` warnings on every resolver
# walk. The janitor below is meant to run after every training cycle
# (see `app.main._run_retrain_blocking`) so the registry stays bounded
# without operator intervention.
ARCHIVED_MODELS_INVENTORY = "archived_models.json"
# Task #458 — hard row cap on the live audit inventory. The janitor
# merges every deletion into `audit/archived_models.json`; without a
# cap the file grows without bound and slows the audit/dashboard
# scripts that read it. When the merged list exceeds the cap, the
# OLDEST overflow rows are spilled into a sibling
# `audit/archived_models.<yyyymm>.json` (bucketed by `deleted_at`)
# and the live file keeps the most recent `ARCHIVED_MODELS_MAX_ROWS`.
# The audit-report script and the existing
# `test_archived_inventory_exists_and_is_consistent` test still read
# the live file unchanged — `count` always matches `len(models)`.
ARCHIVED_MODELS_MAX_ROWS = 5000


def _default_audit_dir() -> Path:
    """Where to record pruned-model metadata.

    Production: `<repo_root>/audit/` (sibling of `artifacts/`). Anchored
    on `REGISTRY_ROOT` rather than `__file__` so that tests monkeypatching
    `REGISTRY_ROOT` onto a tmp dir get an audit dir inside the same
    scratch tree (and don't need to create one explicitly to exercise the
    inventory write path). This mirrors the one-shot
    `scripts/archive_contaminated_models.py` which writes to
    `REPO_ROOT / "audit"` for the same reason.
    """
    # REGISTRY_ROOT = <repo>/artifacts/ml-engine/models in prod →
    # parents[2] is the repo root. In tests it is whatever tmp_path
    # the fixture pinned, and parents[2] is still inside the temp tree
    # (pytest tmp_path is several levels deep).
    return REGISTRY_ROOT.parents[2] / "audit"


def _dir_size_bytes(p: Path) -> int:
    total = 0
    try:
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _archived_inventory_bucket(record: dict) -> str:
    """Return the `yyyymm` bucket id for a spilled-over inventory row.

    Rows produced by the post-training janitor carry an ISO-8601
    `deleted_at` timestamp (`2026-04-24T11:38:...`) — we use the year
    and month for bucketing so a year of overflow doesn't all land in
    a single file. Legacy rows from the original Task #365 one-shot
    sweep have no `deleted_at` field; those land in a single
    `legacy` bucket so they stay grouped together when they roll out.
    """
    ts = record.get("deleted_at")
    if isinstance(ts, str) and len(ts) >= 7 and ts[:4].isdigit() and ts[5:7].isdigit():
        return f"{ts[:4]}{ts[5:7]}"
    return "legacy"


def _spill_archived_overflow(audit_dir: Path, overflow: list[dict]) -> None:
    """Append `overflow` rows into per-month sibling files
    (`archived_models.<yyyymm>.json`). Existing sibling files are
    merged into, never overwritten, so the same bucket can keep growing
    across multiple training cycles.
    """
    if not overflow:
        return
    buckets: dict[str, list[dict]] = {}
    for r in overflow:
        buckets.setdefault(_archived_inventory_bucket(r), []).append(r)
    for bucket, rows in buckets.items():
        sibling = audit_dir / f"archived_models.{bucket}.json"
        existing: dict = {}
        if sibling.exists():
            try:
                e = json.loads(sibling.read_text())
                if isinstance(e, dict):
                    existing = e
            except Exception:  # noqa: BLE001 — corrupt sibling is tolerable, we rewrite it
                existing = {}
        existing_models = existing.get("models")
        if not isinstance(existing_models, list):
            existing_models = []
        existing_models.extend(rows)
        existing["rule"] = (
            "Quant-Only Enforcement — archive any model with LLM-derived "
            "feature columns (rolled over from archived_models.json)"
        )
        existing["forbidden_prefixes"] = list(FORBIDDEN_FEATURE_PREFIXES)
        existing["bucket"] = bucket
        existing["count"] = len(existing_models)
        existing["models"] = existing_models
        sibling.write_text(json.dumps(existing, indent=2, default=str))


def _append_archived_inventory(audit_dir: Path, new_records: list[dict]) -> None:
    """Merge `new_records` into `<audit_dir>/archived_models.json`,
    preserving any existing entries (e.g. from the original Task #365
    one-shot sweep). Idempotent on re-runs because the janitor only ever
    feeds it manifests it just deleted from disk.

    Task #458 — caps the live file at `ARCHIVED_MODELS_MAX_ROWS`. When
    the merged list exceeds the cap, the oldest overflow is spilled
    into per-month sibling files (`archived_models.<yyyymm>.json`)
    and the live file keeps only the most recent rows. The cap keeps
    the audit/dashboard scripts that read the live file fast even
    after many training cycles.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    out = audit_dir / ARCHIVED_MODELS_INVENTORY
    payload: dict = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
            if isinstance(existing, dict):
                payload = existing
        except Exception:  # noqa: BLE001 — corrupt file is tolerable, we rewrite it
            payload = {}
    models = payload.get("models")
    if not isinstance(models, list):
        models = []
    models.extend(new_records)
    # Task #458 — enforce the row cap. The janitor appends in
    # chronological order (newest at the tail) so the OLDEST overflow
    # is at the head of the list. Spill those into a dated sibling
    # file before truncating the live list.
    if len(models) > ARCHIVED_MODELS_MAX_ROWS:
        overflow_n = len(models) - ARCHIVED_MODELS_MAX_ROWS
        overflow = models[:overflow_n]
        kept = models[overflow_n:]
        try:
            _spill_archived_overflow(audit_dir, overflow)
            models = kept
            payload["last_rolled_over_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
            payload["last_rolled_over_count"] = overflow_n
        except OSError as exc:
            # Spill failed — keep the un-truncated list rather than
            # silently losing rows. The cap will be re-attempted on
            # the next janitor pass. Log loudly so an operator can
            # tell that the cap is currently NOT being enforced (a
            # repeated failure here means the live file will keep
            # growing past `ARCHIVED_MODELS_MAX_ROWS`).
            import logging
            logging.getLogger(__name__).warning(
                "archived_inventory_rotation_failed "
                "overflow=%d retained=%d error=%s",
                overflow_n, len(models), exc,
            )
    payload["rule"] = (
        "Quant-Only Enforcement — archive any model with LLM-derived "
        "feature columns"
    )
    payload["forbidden_prefixes"] = list(FORBIDDEN_FEATURE_PREFIXES)
    payload["max_rows"] = ARCHIVED_MODELS_MAX_ROWS
    payload["count"] = len(models)
    payload["models"] = models
    if new_records:
        payload["last_pruned_at"] = new_records[-1].get("deleted_at")
    out.write_text(json.dumps(payload, indent=2, default=str))


def prune_contaminated_versions(audit_dir: Optional[Path] = None) -> dict:
    """Walk every `manifest.json` under `REGISTRY_ROOT`, archive metadata
    for any version whose `feature_names` still advertises a forbidden
    (LLM-derived) column, and DELETE that version directory from disk.

    Designed as the post-training janitor: combined with the save-time
    gate in `save_model`, this keeps `artifacts/ml-engine/models/`
    bounded and the `load_model_rejected_forbidden_features` warning
    channel quiet without manual intervention.

    Returns a summary dict::

        {"deleted": int, "freed_bytes": int, "models": [<record>, ...]}

    The function is best-effort and idempotent: a per-version rmtree
    failure is logged and counted as a skip, never re-raised, so a
    janitor failure cannot mark a training cycle as failed.
    """
    import logging
    import shutil

    if not REGISTRY_ROOT.exists():
        return {"deleted": 0, "freed_bytes": 0, "models": []}
    audit_dir = audit_dir if audit_dir is not None else _default_audit_dir()
    deleted: list[dict] = []
    freed_bytes = 0
    ts = datetime.now(timezone.utc).isoformat()
    log = logging.getLogger(__name__)
    # Materialize the iterator BEFORE walking — `shutil.rmtree` mutates
    # the directory tree as we go, and `rglob` is a generator. Without
    # this, deleting a per-coin/<tf>/<version>/ dir part-way through can
    # raise FileNotFoundError on the next stat.
    all_manifests = list(REGISTRY_ROOT.rglob("manifest.json"))
    for manifest_path in all_manifests:
        version_dir = manifest_path.parent
        if not version_dir.exists():
            # A previous iteration in the same loop may have already
            # rmtree'd a parent (shouldn't happen at the version
            # granularity, but safe to skip).
            continue
        try:
            data = json.loads(manifest_path.read_text())
        except Exception:  # noqa: BLE001 — malformed manifest is unrelated to this gate
            continue
        feats = data.get("feature_names") or []
        bad = sorted({
            c for c in feats
            if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)
        })
        if not bad:
            continue
        size = _dir_size_bytes(version_dir)
        record = {
            "coin_id": data.get("coin_id"),
            "timeframe": data.get("timeframe"),
            "version": data.get("version") or version_dir.name,
            "model_kind": data.get("model_kind"),
            "metrics": data.get("metrics"),
            "n_train_rows": data.get("n_train_rows"),
            "n_test_rows": data.get("n_test_rows"),
            "forbidden_features": bad,
            "deleted_path": str(version_dir),
            "deleted_bytes": size,
            "deleted_at": ts,
        }
        # If `latest` points at the dir we're about to delete, clear the
        # pointer so `latest_version` returns None and `_pick` falls
        # through to an older clean version (or the pooled fallback).
        slot_dir = version_dir.parent
        latest_ptr = slot_dir / "latest"
        if latest_ptr.exists():
            try:
                if latest_ptr.read_text().strip() == version_dir.name:
                    latest_ptr.unlink()
            except OSError:
                pass
        try:
            shutil.rmtree(version_dir)
        except OSError as exc:
            log.warning(
                "prune_contaminated_versions_rmtree_failed "
                "coin=%s tf=%s version=%s error=%s",
                record["coin_id"], record["timeframe"],
                record["version"], exc,
            )
            continue
        freed_bytes += size
        deleted.append(record)
    if deleted:
        try:
            _append_archived_inventory(audit_dir, deleted)
        except OSError as exc:
            log.warning(
                "prune_contaminated_versions_inventory_write_failed "
                "deleted=%d error=%s",
                len(deleted), exc,
            )
        log.warning(
            "prune_contaminated_versions deleted=%d freed_bytes=%d",
            len(deleted), freed_bytes,
        )
    return {
        "deleted": len(deleted),
        "freed_bytes": freed_bytes,
        "models": deleted,
    }


# Set of feature column names we feed the model. Defined here so /predict can
# enforce schema parity with training. Order matters for LightGBM input.
# Phase 5 — news-derived one-hot columns are appended to the canonical
# feature schema. The `news_*` block is sourced from
# `features.NEWS_TAG_VOCABULARY` so adding/removing a tag updates this
# list in one place and the training pipeline picks it up automatically.
# Order matters for LightGBM input.
from app.features import NEWS_TAG_VOCABULARY  # noqa: E402

# Task #267 / #633 — null-safe input streams from the training contract.
# Every column is registered up-front so a future ingestion task only has
# to populate the value: the trainer already pipes them through
# `_prepare_xy`, the manifest already records them, and the per-feature
# coverage in `report.json` already shows their take-up. When a stream is
# missing the column is `NaN` (Task #633 cutover; was `0.0` pre-#633) so
# LightGBM's native `use_missing` path handles it without learning the
# spurious "value was exactly zero" signal that 0-fill baked in.
EXTERNAL_STREAM_FEATURE_COLUMNS: list[str] = [
    "funding_rate",
    "open_interest_z",
    "liquidations_1h_usd",
    "bid_ask_spread_bps",
    "btc_lead_ret_5m",
    "eth_lead_ret_5m",
    # Task #295 — cross-market liquidation pulses, asof-joined from
    # `market_signals` rows whose coin_id is the BTC/ETH/SOL pseudo-coin
    # written by the api-server's poller (task #286). Same value is
    # broadcast onto every per-coin training row because regime stress
    # in the dominant perps usually leads moves in the alts. Defaults
    # to NaN (post-Task #633) when the snapshot table is unavailable.
    "btc_liquidations_1h_usd",
    "eth_liquidations_1h_usd",
    "sol_liquidations_1h_usd",
]
# Session / time-of-day features. Always populated from `timestamp_ms`,
# so coverage is 100% out of the box. Three exclusive session one-hots
# (asia/eu/us) plus a sin/cos encoding of hour-of-day so the booster can
# learn intraday seasonality without an arbitrary cut.
SESSION_FEATURE_COLUMNS: list[str] = [
    "session_asia",
    "session_eu",
    "session_us",
    "hour_of_day_sin",
    "hour_of_day_cos",
]
CONTRACT_NEW_FEATURE_COLUMNS: list[str] = (
    EXTERNAL_STREAM_FEATURE_COLUMNS + SESSION_FEATURE_COLUMNS
)

# Task #365 — Quant-Only Enforcement.
# The Phase 5 `news_*` one-hot block is REMOVED from the canonical feature
# contract. The LLM must not influence trade decisions, even via a frozen
# at-zero one-hot vector — the columns themselves carried the LLM brand on
# the schema, and a future regression that re-enabled the channel would
# have looked like a tuning change. Any model whose `feature_names` still
# contains a `news_*` (or `llm_*`, `gpt_*`, `sentiment_*`, `ai_*`) column
# is REJECTED at load time by `load_model()` below — see the prefix
# blocklist + audit/enforcement-report.md proof B.
FORBIDDEN_FEATURE_PREFIXES: tuple[str, ...] = (
    "news_", "llm_", "gpt_", "sentiment_", "ai_",
)
FEATURE_COLUMNS: list[str] = [
    "ret1", "ret5", "ret10", "momentum", "realizedVol",
    "rsi14", "macdLine", "macdSignal", "macdHist",
    "atr14", "atrPct",
    "ema9", "ema21", "emaSpreadPct", "distFromEma9Pct", "distFromEma21Pct",
    "bbUpper", "bbMiddle", "bbLower", "bbWidth", "bbPctB", "bbWidthPct",
    # Task #517 — rolling 60-bar z-score of |ret1|. Added to lift the
    # dogwifcoin@1d STABLE-class signal off the 5% verification floor;
    # see `reports/<TS>-task507-booster-collapse-rerun-alpha1.0.json`
    # for the per-slice raw_STABLE_share evidence.
    "volZScore60",
    *CONTRACT_NEW_FEATURE_COLUMNS,
    "coin_idx",  # categorical (LightGBM-native), kept LAST so stream columns are stable
]
CATEGORICAL_FEATURES = ["coin_idx"]

# Feature lineage table — for every registered feature column, declare
# the maximum FORWARD index offset its computation is allowed to read
# from. The training contract requires every feature to be strictly
# point-in-time, so `max_lookforward == 0` for ALL entries. The leakage
# audit refuses to train a slice whose feature set contains any column
# with `max_lookforward > 0`, OR any column missing from this table —
# that prevents a developer from quietly adding a new column without
# declaring its lineage.
#
# `max_lookback` is informational (used by the audit's diagnostic logs);
# leave it at None to mean "unspecified". The audit does NOT enforce a
# minimum history requirement here — the per-timeframe MIN_CANDLES guard
# already does that.
FEATURE_LINEAGE: dict[str, dict] = {
    # Price / momentum features computed inside `build_feature_vector`
    # consume only past candles (see `app/features.py`). Each entry is
    # checked at audit time.
    "ret1": {"max_lookforward": 0, "max_lookback": 1},
    "ret5": {"max_lookforward": 0, "max_lookback": 5},
    "ret10": {"max_lookforward": 0, "max_lookback": 10},
    "momentum": {"max_lookforward": 0, "max_lookback": 10},
    "realizedVol": {"max_lookforward": 0, "max_lookback": 30},
    "rsi14": {"max_lookforward": 0, "max_lookback": 14},
    "macdLine": {"max_lookforward": 0, "max_lookback": 26},
    "macdSignal": {"max_lookforward": 0, "max_lookback": 35},
    "macdHist": {"max_lookforward": 0, "max_lookback": 35},
    "atr14": {"max_lookforward": 0, "max_lookback": 14},
    "atrPct": {"max_lookforward": 0, "max_lookback": 14},
    "ema9": {"max_lookforward": 0, "max_lookback": 9},
    "ema21": {"max_lookforward": 0, "max_lookback": 21},
    "emaSpreadPct": {"max_lookforward": 0, "max_lookback": 21},
    "distFromEma9Pct": {"max_lookforward": 0, "max_lookback": 9},
    "distFromEma21Pct": {"max_lookforward": 0, "max_lookback": 21},
    "bbUpper": {"max_lookforward": 0, "max_lookback": 20},
    "bbMiddle": {"max_lookforward": 0, "max_lookback": 20},
    "bbLower": {"max_lookforward": 0, "max_lookback": 20},
    "bbWidth": {"max_lookforward": 0, "max_lookback": 20},
    "bbPctB": {"max_lookforward": 0, "max_lookback": 20},
    "bbWidthPct": {"max_lookforward": 0, "max_lookback": 20},
    # Task #517 — z-score of the last absolute 1-bar return against the
    # trailing 60-bar window of |ret1|. Strictly point-in-time (window
    # is `abs_returns[max(0, k-60):k]`), so `max_lookforward == 0`.
    "volZScore60": {"max_lookforward": 0, "max_lookback": 60},
    "coin_idx": {"max_lookforward": 0, "max_lookback": 0},
    # Task #365 — News tag one-hots have been REMOVED from the feature
    # contract. The `news_tag_features` builder function is preserved in
    # app/features.py for backwards compat with archived parquets, but
    # its output is no longer wired into FEATURE_COLUMNS or this lineage
    # table — and the load-time guard refuses any manifest that still
    # advertises a `news_*` column.
    # External market signals — defaults are NaN (post-Task #633);
    # once wired up they read the latest snapshot at or before bar time.
    **{c: {"max_lookforward": 0, "max_lookback": None}
       for c in EXTERNAL_STREAM_FEATURE_COLUMNS},
    # Session features are deterministic functions of `timestamp_ms`.
    **{c: {"max_lookforward": 0, "max_lookback": 0}
       for c in SESSION_FEATURE_COLUMNS},
}

# Targets that the trainer writes into the labeled frame but MUST never
# appear in any feature column. Used by `audit_leakage` to fail the build
# if a refactor accidentally copies a forward-looking target into the
# feature set.
FORWARD_TARGET_COLUMNS: list[str] = [
    "forward_return",
    "label_binary_up",
    "label_3class",
    "forward_window_return_pct",
    "prob_move_gt_cost",
    "tp_before_sl_long",
    "tp_before_sl_short",
    "mae_pct_long",
    "mfe_pct_long",
    "opportunity_score",
    "realized_vol_next_horizon",
    "net_pnl_after_costs_pct",
    # Task #379 — multi-bar directional label transparency fields. The
    # forward window stamped here can be H bars ahead, so they are
    # categorically forward-looking and must never feed the feature set.
    "directional_label_forward_return",
    "directional_label_horizon_candles",
]
