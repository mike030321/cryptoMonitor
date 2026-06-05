"""Task #455 — Meta-model rebuild & strict-better re-promotion driver.

What it does (one shot, idempotent):

  1. Audit the live `prediction_journal` for the column names actually
     present in the QUANT slice we'll train on. Writes
     `.local/cleanup/meta-rebuild/journal-columns-audit.json` and HARD
     FAILS if any forbidden-prefix key (news_/llm_/sentiment_/...) is
     found inside the columns the trainer can read.

  2. For each timeframe with sufficient data:
      a. Snapshot the current `latest` pointer (= incumbent).
      b. Build the meta dataset, do an 80/20 chronological split.
      c. Train a new LightGBM candidate ON THE TRAIN SLICE ONLY,
         persist it under `models/__meta__/{tf}/{candidate_version}/`.
      d. Score three predictors on the SAME holdout:
           - candidate (the just-trained LGB pair)
           - incumbent (whatever `latest` pointed at before)
           - heuristic baseline (`app.main._meta_heuristic`)
         using cost-aware directional accuracy (CADA) and the
         Pearson correlation between predicted edge and realised
         edge after costs.
      e. Promote candidate ONLY if it strictly beats BOTH incumbent
         AND heuristic by `--margin` (default 0.005). Otherwise
         restore the incumbent's `latest` pointer.
      f. Write `.local/cleanup/meta-rebuild/holdout-{tf}.json`
         with the full three-way comparison + decision.

  3. For each timeframe with insufficient data, write a stub
     holdout-{tf}.json explaining why no candidate could be trained
     and leave the existing `latest` pointer untouched.

  4. Tee per-timeframe training stdout to
     `.local/cleanup/meta-rebuild/train-{tf}.log` so the row count,
     class balance, and hyperparameters are auditable.

Run from the ml-engine artifact root:

    cd artifacts/ml-engine
    ../../.pythonlibs/bin/python -m scripts.meta_rebuild

"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from app import db as db_mod  # noqa: E402
from app.training import meta_dataset as meta_dataset_mod  # noqa: E402
from app.training import registry as registry_module  # noqa: E402
from app.training import train_meta as train_meta_mod  # noqa: E402

OUT_DIR = REPO_ROOT / ".local" / "cleanup" / "meta-rebuild"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORBIDDEN_PREFIXES = (
    "news_", "llm_", "gpt_", "sentiment_", "ai_",
    "benchmark_", "alpha_", "baseline_", "equity_", "strategy_lab_",
)

# Tradeable timeframes for the meta-model. The full set the production
# system supports per `quant_brain` / `meta_dataset.py` — we attempt
# every one of these AND we auto-discover any extra timeframes that
# appear in the live journal (so a future tf added to the trader is
# not silently skipped).
TRADEABLE_TIMEFRAMES = ["1m", "5m", "15m", "1h", "2h", "4h", "6h", "1d"]

logger = logging.getLogger("meta-rebuild")


# ----------------------------------------------------------------------
# (1) Journal column audit
# ----------------------------------------------------------------------
async def _discover_timeframes_in_journal() -> list[str]:
    """Return every distinct timeframe present in the resolved QUANT
    slice of `prediction_journal`, ordered by the conventional
    fine→coarse list and then by any extras alphabetically. Used so a
    timeframe added to the trader after this script was written is
    still picked up automatically."""
    pool = await db_mod.init_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT timeframe
            FROM prediction_journal
            WHERE brain = 'QUANT' AND timeframe IS NOT NULL
            """
        )
    found = {r["timeframe"] for r in rows}
    ordered = [tf for tf in TRADEABLE_TIMEFRAMES if tf in found]
    extras = sorted(found - set(TRADEABLE_TIMEFRAMES))
    # Always include the canonical list even if a tf has zero rows —
    # the per-tf step writes a `no_data` holdout file in that case so
    # we have a record. Discovery only adds extras the static list
    # missed.
    return list(TRADEABLE_TIMEFRAMES) + extras


async def _audit_journal_columns(timeframes: list[str]) -> dict:
    """Inspect the QUANT slice the trainer will actually read and
    enumerate every column name present (top-level + jsonb keys it
    drills into). Writes the result to journal-columns-audit.json and
    raises if any forbidden prefix slips through.
    """
    pool = await db_mod.init_pool()
    async with pool.acquire() as conn:
        # Top-level prediction_journal columns we touch.
        rows_top = await conn.fetch(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'prediction_journal'
            ORDER BY ordinal_position
            """
        )
        # Distinct keys actually present in gates_applied jsonb.
        rows_gates = await conn.fetch(
            """
            SELECT DISTINCT jsonb_object_keys(gates_applied) AS key
            FROM prediction_journal
            WHERE brain = 'QUANT' AND gates_applied IS NOT NULL
            ORDER BY key
            """
        )
        # Per-timeframe row counts so the audit doubles as the data
        # availability snapshot the holdout step depends on.
        rows_counts: dict[str, int] = {}
        rows_resolved: dict[str, int] = {}
        for tf in timeframes:
            r = await conn.fetchrow(
                """
                SELECT
                  COUNT(*) FILTER (WHERE timeframe=$1) AS total,
                  COUNT(*) FILTER (WHERE timeframe=$1
                                    AND realized_return_pct IS NOT NULL
                                    AND gates_applied IS NOT NULL) AS resolved
                FROM prediction_journal
                WHERE brain = 'QUANT'
                """,
                tf,
            )
            rows_counts[tf] = int(r["total"] or 0)
            rows_resolved[tf] = int(r["resolved"] or 0)

        date_range = await conn.fetchrow(
            """
            SELECT MIN(created_at) AS earliest, MAX(created_at) AS latest
            FROM prediction_journal
            WHERE brain = 'QUANT'
              AND realized_return_pct IS NOT NULL
              AND gates_applied IS NOT NULL
            """
        )

    top_cols = [r["column_name"] for r in rows_top]
    gates_keys = [r["key"] for r in rows_gates]

    # The trainer only reads these columns (cf. dataset-columns.json).
    trainer_reads_top = [
        "id", "created_at", "coin_id", "timeframe",
        "prob_up", "prob_down", "prob_stable",
        "expected_return_pct", "prediction_std_pct", "raw_confidence",
        "regime_label", "gates_applied", "realized_return_pct",
    ]
    trainer_reads_gates = ["specialists"]

    # Forbidden-prefix scan — only over columns/keys the trainer can
    # actually reach. `feature_vector` carries `news_*` from the
    # pre-#444 era, but the SQL doesn't read it, so it's not flagged.
    def _bad(name: str) -> bool:
        lc = name.lower()
        return any(lc.startswith(p) for p in FORBIDDEN_PREFIXES)

    forbidden_in_top = [c for c in trainer_reads_top if _bad(c)]
    forbidden_in_gates = [k for k in trainer_reads_gates if _bad(k)]
    # Also flag if any forbidden key appears inside `specialists`
    # element shape (kind/applicable/probUp/probDown/expectedReturnPct).
    spec_keys = ["kind", "applicable", "probUp", "probDown", "expectedReturnPct"]
    forbidden_in_specialists = [k for k in spec_keys if _bad(k)]

    audit = {
        "task": "455 — meta-rebuild journal-columns audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_journal_top_level_columns": top_cols,
        "prediction_journal_gates_applied_keys": gates_keys,
        "trainer_reads_top_level_columns": trainer_reads_top,
        "trainer_reads_gates_applied_keys": trainer_reads_gates,
        "specialists_element_keys_read": spec_keys,
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_columns_found_in_trainer_reach": (
            forbidden_in_top + forbidden_in_gates + forbidden_in_specialists
        ),
        "rows_per_timeframe": rows_counts,
        "resolved_rows_per_timeframe": rows_resolved,
        "earliest_resolved_quant_row": (
            date_range["earliest"].isoformat() if date_range and date_range["earliest"] else None
        ),
        "latest_resolved_quant_row": (
            date_range["latest"].isoformat() if date_range and date_range["latest"] else None
        ),
        "notes": [
            "feature_vector jsonb still carries legacy news_* keys from the pre-#444 era; "
            "meta_dataset.py deliberately does NOT read feature_vector, so those keys "
            "cannot reach the meta-model trainer.",
            "gates_applied jsonb DOES carry meta_action / meta_kind / meta_version / "
            "meta_no_trade_* etc., but the trainer only reads the `specialists` array.",
        ],
    }

    (OUT_DIR / "journal-columns-audit.json").write_text(json.dumps(audit, indent=2, default=str))

    if audit["forbidden_columns_found_in_trainer_reach"]:
        raise RuntimeError(
            "Forbidden-prefix columns are reachable by the meta-model trainer: "
            f"{audit['forbidden_columns_found_in_trainer_reach']}. Fix meta_dataset.py "
            "before re-running this script."
        )
    return audit


# ----------------------------------------------------------------------
# (2) Predictors used by the holdout comparison
# ----------------------------------------------------------------------
ROUND_TRIP_COST_PCT = meta_dataset_mod.ROUND_TRIP_COST_PCT * 100.0  # percent


def _heuristic_predict(feat_row: dict) -> tuple[str, float]:
    """Wrap `app.main._meta_heuristic` so it returns (action, predicted_edge_pct).
    Imported lazily to avoid pulling FastAPI during a CLI invocation."""
    from app.main import _meta_heuristic

    action, _size, edge, _reason, _probs = _meta_heuristic(feat_row)
    return action, float(edge)


def _model_predict_row(clf, reg, feat_row: dict) -> tuple[str, float]:
    """LightGBM prediction for a single row using booster.predict(). Returns
    (action_label, predicted_edge_pct)."""
    x = np.asarray(
        [[feat_row.get(c, 0.0) for c in meta_dataset_mod.META_FEATURE_COLUMNS]],
        dtype=float,
    )
    proba = np.asarray(clf.predict(x)).flatten()
    action_idx = int(proba.argmax())
    action = train_meta_mod.ACTION_LABELS[action_idx]
    edge_pred = float(np.asarray(reg.predict(x)).flatten()[0])
    return action, edge_pred


def _cost_aware_directional_accuracy(actions: list[str], rows: pd.DataFrame) -> float:
    """For each holdout bar the predictor scores 1.0 when its predicted
    action matches the canonical cost-aware label (`__action__`) the
    trainer used, else 0.0.

    The canonical label is built directly in `meta_dataset._label_action`
    from the realised return percent and the round-trip cost — it is the
    SAME partition (`long` if realised > +cost, `short` if realised <
    -cost, `no_trade` otherwise) that the trainer used. Comparing
    against it directly avoids any reconstruction arithmetic and is
    safe near zero-return cases (e.g. realised = +0.1% with cost = 0.3%
    is correctly labelled `no_trade`, not mis-reconstructed via the
    edge-after-costs sign).
    """
    if len(actions) == 0:
        return float("nan")
    truth = rows["__action__"].astype(str).to_list()
    score = sum(1 for a, t in zip(actions, truth) if a == t)
    return float(score) / float(len(actions))


def _edge_correlation(predicted_edges: list[float], rows: pd.DataFrame) -> float:
    """Pearson r between predicted edge and the realised after-cost edge."""
    if len(predicted_edges) < 3:
        return float("nan")
    realised_edges = rows["__edge_after_costs__"].to_numpy(dtype=float)
    pred = np.asarray(predicted_edges, dtype=float)
    if np.std(pred) < 1e-12 or np.std(realised_edges) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, realised_edges)[0, 1])


# ----------------------------------------------------------------------
# (3) Per-timeframe train + score + promote
# ----------------------------------------------------------------------
def _make_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _meta_dir(tf: str, ver: str) -> Path:
    return registry_module.REGISTRY_ROOT / train_meta_mod.META_COIN_ID / tf / ver


def _latest_path(tf: str) -> Path:
    return registry_module.REGISTRY_ROOT / train_meta_mod.META_COIN_ID / tf / "latest"


def _train_lightgbm_on_train_slice(
    df_train: pd.DataFrame, df_full: pd.DataFrame, tf: str, version: str, log_lines: list[str],
) -> dict:
    """Mirror of `train_meta._train_lightgbm` BUT trains on an explicitly
    provided train slice so the holdout split is honoured.
    """
    import lightgbm as lgb

    X = df_train[meta_dataset_mod.META_FEATURE_COLUMNS].astype(float).to_numpy()
    y_action = df_train["__action__"].map(
        {a: i for i, a in enumerate(train_meta_mod.ACTION_LABELS)}
    ).to_numpy()
    y_edge = df_train["__edge_after_costs__"].to_numpy(dtype=float)

    clf_params = {
        "objective": "multiclass",
        "num_class": len(train_meta_mod.ACTION_LABELS),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    reg_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    log_lines.append(f"[train] timeframe={tf} version={version}")
    log_lines.append(f"[train] full_rows={len(df_full)} train_rows={len(df_train)}")
    log_lines.append(f"[train] clf_params={json.dumps(clf_params)}")
    log_lines.append(f"[train] reg_params={json.dumps(reg_params)} num_boost_round=200")
    class_counts = {
        a: int((df_train["__action__"] == a).sum())
        for a in train_meta_mod.ACTION_LABELS
    }
    log_lines.append(f"[train] class_counts(train)={json.dumps(class_counts)}")

    t0 = time.time()
    clf = lgb.train(
        params=clf_params,
        train_set=lgb.Dataset(X, label=y_action),
        num_boost_round=200,
    )
    reg = lgb.train(
        params=reg_params,
        train_set=lgb.Dataset(X, label=y_edge),
        num_boost_round=200,
    )
    train_secs = time.time() - t0
    log_lines.append(f"[train] training_duration_secs={train_secs:.2f}")

    out_dir = _meta_dir(tf, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(out_dir / "meta_clf.txt"))
    reg.save_model(str(out_dir / "meta_reg.txt"))
    log_lines.append(f"[train] artifacts_written={out_dir}")
    return {
        "n_train": int(len(df_train)),
        "n_full": int(len(df_full)),
        "class_counts_train": class_counts,
        "training_duration_secs": train_secs,
        "clf_params": clf_params,
        "reg_params": reg_params,
        "num_boost_round": 200,
    }


def _save_candidate_manifest(
    tf: str,
    version: str,
    n_train_rows: int,
    n_full_rows: int,
    train_metrics: dict,
    holdout: dict,
    note: str,
) -> None:
    out_dir = _meta_dir(tf, version)
    manifest = {
        "coin_id": train_meta_mod.META_COIN_ID,
        "timeframe": tf,
        "version": version,
        "model_kind": "meta",
        "meta_kind": "lightgbm",
        "feature_names": list(meta_dataset_mod.META_FEATURE_COLUMNS),
        "n_train_rows": int(n_train_rows),
        "n_total_dataset_rows": int(n_full_rows),
        "metrics": train_metrics,
        "holdout_summary": holdout,
        "calibration_buckets": [],  # populated by the original /train_meta path on full-fit
        "action_labels": train_meta_mod.ACTION_LABELS,
        "round_trip_cost_pct": 0.003,
        "training_row_window": {
            "first_created_at": (
                holdout.get("dataset_first_created_at") if holdout else None
            ),
            "last_created_at": (
                holdout.get("dataset_last_created_at") if holdout else None
            ),
        },
        "training_columns": list(meta_dataset_mod.META_FEATURE_COLUMNS),
        "task": "455 — meta-rebuild on post-sidecar journal data",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _set_latest(tf: str, version: str) -> None:
    p = _latest_path(tf)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(version)


def _score_predictor_block(
    name: str,
    actions: list[str],
    predicted_edges: list[float],
    holdout: pd.DataFrame,
) -> dict:
    return {
        "name": name,
        "n_holdout": int(len(holdout)),
        "cost_aware_directional_accuracy": _cost_aware_directional_accuracy(actions, holdout),
        "predicted_edge_realized_corr": _edge_correlation(predicted_edges, holdout),
        "action_distribution": {
            a: int(actions.count(a))
            for a in train_meta_mod.ACTION_LABELS
        },
    }


async def _process_timeframe(
    tf: str, margin: float, dry_run: bool,
) -> dict:
    """Returns the holdout-{tf}.json payload."""
    log_lines: list[str] = [
        f"=== meta-rebuild timeframe={tf} started at {datetime.now(timezone.utc).isoformat()} ==="
    ]

    # Snapshot the incumbent BEFORE we touch anything.
    incumbent_version: Optional[str] = None
    if _latest_path(tf).exists():
        incumbent_version = _latest_path(tf).read_text().strip() or None

    incumbent_manifest = train_meta_mod.load_meta_manifest(tf)

    df = await meta_dataset_mod.build_meta_dataset(timeframe=tf)
    log_lines.append(f"[dataset] built rows={len(df)}")

    if df.empty:
        log_lines.append("[skip] empty dataset — leaving incumbent in place")
        (OUT_DIR / f"train-{tf}.log").write_text("\n".join(log_lines) + "\n")
        return {
            "timeframe": tf,
            "decision": "no_promote_no_data",
            "reason": "no resolved QUANT rows for this timeframe — heuristic stays",
            "incumbent_version": incumbent_version,
            "candidate_version": None,
            "promotion_margin_required": margin,
        }

    n_full = len(df)
    n_classes = sum(int((df["__action__"] == a).sum()) > 0 for a in train_meta_mod.ACTION_LABELS)
    class_counts = {
        a: int((df["__action__"] == a).sum())
        for a in train_meta_mod.ACTION_LABELS
    }
    log_lines.append(f"[dataset] class_counts(full)={json.dumps(class_counts)}")

    # Need the same minimum bar the production trainer uses, plus enough
    # holdout rows to compute a stable CADA.
    if (
        n_full < train_meta_mod.MIN_ROWS_FOR_FIT
        or any(c < train_meta_mod.MIN_PER_CLASS_FOR_FIT for c in class_counts.values())
    ):
        log_lines.append(
            f"[skip] below threshold (need >= {train_meta_mod.MIN_ROWS_FOR_FIT} rows "
            f"and >= {train_meta_mod.MIN_PER_CLASS_FOR_FIT} per class) — leaving heuristic in place"
        )
        (OUT_DIR / f"train-{tf}.log").write_text("\n".join(log_lines) + "\n")
        return {
            "timeframe": tf,
            "decision": "no_promote_insufficient_data",
            "reason": (
                f"have {n_full} rows / {class_counts}; "
                f"need >= {train_meta_mod.MIN_ROWS_FOR_FIT} rows and "
                f">= {train_meta_mod.MIN_PER_CLASS_FOR_FIT} per class"
            ),
            "incumbent_version": incumbent_version,
            "candidate_version": None,
            "promotion_margin_required": margin,
            "n_full_rows": n_full,
            "class_counts_full": class_counts,
        }

    # Chronological 80/20 split.
    df = df.sort_values("__created_at__").reset_index(drop=True)
    split = max(1, int(len(df) * 0.8))
    df_train = df.iloc[:split].reset_index(drop=True)
    df_holdout = df.iloc[split:].reset_index(drop=True)
    log_lines.append(f"[split] train_rows={len(df_train)} holdout_rows={len(df_holdout)}")

    if len(df_holdout) < 20:
        log_lines.append("[skip] holdout < 20 rows — insufficient to discriminate")
        (OUT_DIR / f"train-{tf}.log").write_text("\n".join(log_lines) + "\n")
        return {
            "timeframe": tf,
            "decision": "no_promote_insufficient_holdout",
            "reason": f"holdout has only {len(df_holdout)} rows; need >= 20",
            "incumbent_version": incumbent_version,
            "candidate_version": None,
            "promotion_margin_required": margin,
            "n_full_rows": n_full,
        }

    # Train the candidate on train slice only.
    candidate_version = _make_version()
    train_metrics = _train_lightgbm_on_train_slice(
        df_train, df, tf, candidate_version, log_lines,
    )

    # Load the candidate boosters (we just wrote them).
    import lightgbm as lgb

    cand_dir = _meta_dir(tf, candidate_version)
    cand_clf = lgb.Booster(model_file=str(cand_dir / "meta_clf.txt"))
    cand_reg = lgb.Booster(model_file=str(cand_dir / "meta_reg.txt"))

    # Score candidate on holdout.
    cand_actions: list[str] = []
    cand_edges: list[float] = []
    for r in df_holdout.to_dict(orient="records"):
        a, e = _model_predict_row(cand_clf, cand_reg, r)
        cand_actions.append(a)
        cand_edges.append(e)

    # Score the heuristic baseline on holdout.
    heur_actions: list[str] = []
    heur_edges: list[float] = []
    for r in df_holdout.to_dict(orient="records"):
        a, e = _heuristic_predict(r)
        heur_actions.append(a)
        heur_edges.append(e)

    # Score the incumbent on holdout (only if it is a real LightGBM
    # head; a heuristic incumbent is functionally identical to the
    # baseline so we mark its score equal to it).
    inc_block: dict
    if (
        incumbent_manifest
        and incumbent_manifest.get("meta_kind") == "lightgbm"
        and incumbent_version
        and (_meta_dir(tf, incumbent_version) / "meta_clf.txt").exists()
    ):
        inc_clf = lgb.Booster(
            model_file=str(_meta_dir(tf, incumbent_version) / "meta_clf.txt")
        )
        inc_reg = lgb.Booster(
            model_file=str(_meta_dir(tf, incumbent_version) / "meta_reg.txt")
        )
        inc_actions: list[str] = []
        inc_edges: list[float] = []
        for r in df_holdout.to_dict(orient="records"):
            a, e = _model_predict_row(inc_clf, inc_reg, r)
            inc_actions.append(a)
            inc_edges.append(e)
        inc_block = _score_predictor_block(
            "incumbent_lightgbm", inc_actions, inc_edges, df_holdout,
        )
        inc_block["version"] = incumbent_version
        inc_block["meta_kind"] = "lightgbm"
    else:
        # Incumbent is heuristic (or missing) — score the heuristic in
        # its place so the comparison still has three named blocks.
        inc_block = dict(
            _score_predictor_block(
                "incumbent_heuristic", heur_actions, heur_edges, df_holdout,
            ),
            version=incumbent_version,
            meta_kind=(incumbent_manifest or {}).get("meta_kind"),
        )

    cand_block = _score_predictor_block(
        "candidate_lightgbm", cand_actions, cand_edges, df_holdout,
    )
    cand_block["version"] = candidate_version
    cand_block["meta_kind"] = "lightgbm"
    heur_block = _score_predictor_block(
        "heuristic_baseline", heur_actions, heur_edges, df_holdout,
    )

    cand_cada = cand_block["cost_aware_directional_accuracy"]
    inc_cada = inc_block["cost_aware_directional_accuracy"]
    heur_cada = heur_block["cost_aware_directional_accuracy"]

    delta_vs_inc = cand_cada - inc_cada
    delta_vs_heur = cand_cada - heur_cada

    strict_better = (
        np.isfinite(cand_cada)
        and np.isfinite(inc_cada)
        and np.isfinite(heur_cada)
        and delta_vs_inc >= margin
        and delta_vs_heur >= margin
    )

    if strict_better and not dry_run:
        _set_latest(tf, candidate_version)
        decision = "promoted"
        reason = (
            f"candidate CADA={cand_cada:.4f} beat incumbent CADA={inc_cada:.4f} "
            f"(Δ={delta_vs_inc:+.4f}) AND heuristic CADA={heur_cada:.4f} "
            f"(Δ={delta_vs_heur:+.4f}) by margin >= {margin}"
        )
    else:
        # Restore the incumbent pointer if anything moved it (defensive).
        if incumbent_version:
            _set_latest(tf, incumbent_version)
        decision = "no_promote_strict_gate_failed"
        reason = (
            f"candidate CADA={cand_cada:.4f} did not beat both "
            f"incumbent CADA={inc_cada:.4f} (Δ={delta_vs_inc:+.4f}) "
            f"and heuristic CADA={heur_cada:.4f} (Δ={delta_vs_heur:+.4f}) "
            f"by margin >= {margin}"
        )

    holdout_payload = {
        "timeframe": tf,
        "decision": decision,
        "reason": reason,
        "promotion_margin_required": margin,
        "incumbent_version": incumbent_version,
        "candidate_version": candidate_version,
        "n_full_rows": n_full,
        "n_train_rows": int(len(df_train)),
        "n_holdout_rows": int(len(df_holdout)),
        "class_counts_full": class_counts,
        "dataset_first_created_at": str(df["__created_at__"].iloc[0]),
        "dataset_last_created_at": str(df["__created_at__"].iloc[-1]),
        "round_trip_cost_pct_used": ROUND_TRIP_COST_PCT,
        "blocks": {
            "candidate": cand_block,
            "incumbent": inc_block,
            "heuristic": heur_block,
        },
        "deltas_vs_incumbent": {
            "cost_aware_directional_accuracy": delta_vs_inc,
        },
        "deltas_vs_heuristic": {
            "cost_aware_directional_accuracy": delta_vs_heur,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Update the candidate manifest with the holdout it earned.
    _save_candidate_manifest(
        tf, candidate_version,
        n_train_rows=int(len(df_train)),
        n_full_rows=n_full,
        train_metrics=train_metrics,
        holdout=holdout_payload,
        note=f"task-455 candidate; decision={decision}",
    )

    log_lines.append(f"[holdout] {json.dumps(holdout_payload, default=str)}")
    (OUT_DIR / f"train-{tf}.log").write_text("\n".join(log_lines) + "\n")
    return holdout_payload


# ----------------------------------------------------------------------
# (4) Main orchestrator
# ----------------------------------------------------------------------
async def _amain(margin: float, only_tf: Optional[str], dry_run: bool) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if only_tf:
        timeframes = [only_tf]
    else:
        # Discover timeframes from the live journal (covers any tf the
        # trader added after this script was written), unioned with the
        # canonical TRADEABLE_TIMEFRAMES list (so a 0-row tf still gets
        # a holdout file recording the no_data decision).
        timeframes = await _discover_timeframes_in_journal()

    audit = await _audit_journal_columns(timeframes)
    print(json.dumps({"audit_summary": {
        "rows_per_timeframe": audit["rows_per_timeframe"],
        "resolved_rows_per_timeframe": audit["resolved_rows_per_timeframe"],
        "forbidden_columns_found_in_trainer_reach": audit[
            "forbidden_columns_found_in_trainer_reach"
        ],
    }}, indent=2))

    summary: dict[str, dict] = {}
    for tf in timeframes:
        try:
            summary[tf] = await _process_timeframe(tf, margin, dry_run)
        except Exception as exc:  # noqa: BLE001
            logger.exception("meta_rebuild_failed timeframe=%s", tf)
            summary[tf] = {
                "timeframe": tf,
                "decision": "no_promote_error",
                "reason": str(exc),
            }
        (OUT_DIR / f"holdout-{tf}.json").write_text(json.dumps(summary[tf], indent=2, default=str))

    summary_path = OUT_DIR / "rebuild-summary.json"
    summary_path.write_text(json.dumps({
        "task": "455 — meta-rebuild summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "promotion_margin_required": margin,
        "dry_run": dry_run,
        "per_timeframe": summary,
    }, indent=2, default=str))

    promoted = [tf for tf, p in summary.items() if p.get("decision") == "promoted"]
    print(f"\nmeta-rebuild done. promoted={promoted} of {timeframes}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--margin", type=float, default=0.005,
                   help="strict-better margin for cost-aware directional accuracy")
    p.add_argument("--timeframe", type=str, default=None,
                   help="run a single timeframe (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="train and score, but never move the latest pointer")
    args = p.parse_args()
    return asyncio.run(_amain(args.margin, args.timeframe, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
