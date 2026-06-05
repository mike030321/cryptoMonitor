"""Contract test for the Task #467 Meta-Brain replay.

Asserts the pinned dataset allow-list, the SQL allow-list, the
replay's feature filter, the outcome derivation, and the manifest
schema collectively guarantee:

  1. No column name ever fed to the supervisory layer matches a
     forbidden prefix (news_/llm_/gpt_/sentiment_/ai_).
  2. Every per-table SQL_COLUMNS entry is a SUBSET of the
     corresponding bucket in dataset-columns.json — and every column
     name appearing in a SQL string in scripts/replay_meta_brain.py
     is in SQL_COLUMNS for its table. A future query that selects
     `news_*` (or any unlisted column) will turn this guard red.
  3. Real journal feature_vector payloads — which DO contain
     news_*-prefixed keys today — are stripped before the slice
     reaches `evaluate()`, and every dropped key is recorded under
     `forbidden_columns_seen` so the manifest tells the truth.
  4. Honest outcome derivation never fabricates the three
     counterfactual fields, and intra-trade MAE / stability come
     from real price_candles when present (no neutral placeholders
     when the data is there).
  5. The five strategy families resolve to the bounded set the
     Meta-model expects (matches resolveStrategyFamily in adapter).
  6. Manifest schema carries every Task #467 §manifest field
     (data_window, row_counts, regimes_observed, families_observed,
     final_trust_state, regime_prototype_count, episode_buffer_size,
     commit semantics, state ∈ {production_ready,
     pipeline_validation_only}). Verified by exercising the writer
     end-to-end against an in-memory engine.
  7. Episodic memory survives a state_dict / load_state_dict round
     trip with `None` counterfactual fields (the post-#467 schema).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ML_ENGINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ML_ENGINE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import replay_meta_brain  # noqa: E402
from replay_meta_brain import (  # noqa: E402
    ALLOWED_FAMILIES,
    ALLOW_LIST_PATH,
    DEFAULT_HOLDOUT_PCT,
    DEFAULT_MIN_DAYS,
    DEFAULT_MIN_REGIMES,
    DEFAULT_MIN_TRADES,
    FORBIDDEN_PREFIXES,
    SQL_COLUMNS,
    _filter_features,
    build_manifest,
    build_slice,
    cluster_into_cycles,
    compute_holdout_metrics,
    derive_outcome,
    is_forbidden,
    load_allow_list,
    resolve_strategy_family,
)


# ────────────────────────── helpers ────────────────────────────────


def _make_row(created_at, **overrides) -> dict:
    base = {
        "id": 1,
        "created_at": created_at,
        "agent_id": 1,
        "agent_name": "Momentum Max",
        "coin_id": "bitcoin",
        "timeframe": "1h",
        "regime_label": "bull",
        "direction": "up",
        "confidence": 0.62,
        "raw_confidence": 0.55,
        "prob_up": 0.62,
        "prob_down": 0.30,
        "prob_stable": 0.08,
        "expected_return_pct": 0.4,
        "prediction_std_pct": 0.1,
        "price_at_prediction": 100.0,
        "predicted_price": 100.4,
        "feature_vector": {
            "ema9": 100.1,
            "ema21": 99.8,
            "rsi14": 58.0,
            "realizedVol": 0.012,
            "macdHist": 0.001,
            "news_etf_flow": 1,
            "news_whale_move": 0,
            "sentiment_score": 0.3,
            "llm_signal": 0.7,
            "gpt_summary_tokens": 42,
            "ai_meta_flag": 1,
        },
        "became_trade": False,
        "trade_id": None,
        "resolved_at": None,
        "actual_price": None,
        "realized_return_pct": None,
        "outcome": None,
        "shadow": False,
    }
    base.update(overrides)
    return base


# ────────────────── 1. allow-list integrity ────────────────────────


def test_allow_list_has_no_forbidden_keys() -> None:
    allow = load_allow_list()
    seen: list[str] = []
    for bucket in (
        "prediction_journal_columns",
        "feature_vector_keys",
        "paper_trades_columns",
        "paper_positions_columns",
        "paper_position_marks_columns",
        "paper_portfolios_columns",
        "agents_columns",
        "strategy_snapshots_columns",
        "market_signals_columns",
        "price_candles_columns",
    ):
        for key in allow[bucket]:
            if is_forbidden(key):
                seen.append(f"{bucket}.{key}")
    assert seen == [], (
        "dataset-columns.json contains forbidden-prefix keys: "
        + ", ".join(seen)
    )


def test_forbidden_prefixes_match_registry() -> None:
    """Mirrors FORBIDDEN_FEATURE_PREFIXES in app/training/registry.py.
    If the registry adds a new banned prefix, this guard tells us so
    instead of silently allowing it through replay."""
    from app.training import registry

    assert tuple(registry.FORBIDDEN_FEATURE_PREFIXES) == FORBIDDEN_PREFIXES


# ──────────────── 2. SQL allow-list enforcement ────────────────────


def test_sql_columns_subset_of_dataset_columns() -> None:
    """Every column the script names in a SELECT must also live in
    the pinned dataset-columns.json bucket for the same table."""
    allow = load_allow_list()
    bucket_for = {
        "prediction_journal": "prediction_journal_columns",
        "paper_trades": "paper_trades_columns",
        "paper_positions": "paper_positions_columns",
        "paper_position_marks": "paper_position_marks_columns",
        "paper_portfolios": "paper_portfolios_columns",
        "agents": "agents_columns",
        "strategy_snapshots": "strategy_snapshots_columns",
        "market_signals": "market_signals_columns",
        "price_candles": "price_candles_columns",
    }
    for table, cols in SQL_COLUMNS.items():
        bucket = bucket_for[table]
        diff = set(cols) - set(allow[bucket])
        assert not diff, (
            f"SQL_COLUMNS[{table}] references columns missing from "
            f"dataset-columns.json[{bucket}]: {sorted(diff)}"
        )
        for col in cols:
            assert not is_forbidden(col), (
                f"SQL_COLUMNS[{table}] contains forbidden column {col}"
            )


def test_sql_strings_only_reference_allowed_columns() -> None:
    """Walk the AST of replay_meta_brain.py, isolate every string
    literal that begins with `SELECT`, and assert every column name
    in its SELECT clause appears in SQL_COLUMNS for that table. This
    is the guard against a future diff that adds a `news_*` (or any
    unlisted) column to a query string but forgets to add it to
    SQL_COLUMNS — the manifest would still claim the columns_used
    list was honoured."""
    import ast

    src = (SCRIPTS_ROOT / "replay_meta_brain.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    sql_strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            stripped = node.value.strip()
            if stripped.upper().startswith("SELECT"):
                sql_strings.append(stripped)
    pattern = re.compile(
        r"^SELECT\s+(.+?)\s+FROM\s+(\w+)",
        re.IGNORECASE | re.DOTALL,
    )
    fingerprints: list[tuple[str, str]] = []
    for sql in sql_strings:
        m = pattern.match(sql)
        if m is None:
            continue
        fingerprints.append((m.group(1), m.group(2)))
    seen_tables: set[str] = set()
    for select_clause, table in fingerprints:
        if table not in SQL_COLUMNS:
            continue
        seen_tables.add(table)
        # Strip aggregates / aliases / casts; collect bare identifiers.
        cleaned = re.sub(r"AS\s+\w+", "", select_clause, flags=re.IGNORECASE)
        cleaned = re.sub(r"::\w+", "", cleaned)
        # Treat the COUNT/MIN/MAX wrappers as transparent to inner col.
        cleaned = re.sub(
            r"\b(COUNT|MIN|MAX|SUM|AVG|COALESCE)\s*\(", "(", cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.replace("(", " ").replace(")", " ")
        idents = re.findall(r"\b[a-z_][a-z_0-9]*\b", cleaned)
        skip = {
            "true", "false", "null", "and", "or", "not", "as",
            "distinct", "from", "where", "group", "by", "order",
            "limit", "asc", "desc", "in", "any", "all", "case",
            "when", "then", "else", "end", "is", "on", "between",
        }
        for ident in idents:
            if ident in skip:
                continue
            if ident.isdigit():
                continue
            assert ident in SQL_COLUMNS[table], (
                f"SQL string for `{table}` references unlisted "
                f"identifier `{ident}` — add it to SQL_COLUMNS or "
                f"remove the column from the SELECT."
            )
    # Sanity: at least the core tables must have been fingerprinted.
    for required in (
        "prediction_journal", "paper_trades", "strategy_snapshots",
        "market_signals", "price_candles",
    ):
        assert required in seen_tables, (
            f"contract test never fingerprinted SELECT FROM {required}"
        )


# ──────────────────── 3. feature filtering ─────────────────────────


def test_filter_features_strips_forbidden_keys() -> None:
    allowed = set(load_allow_list()["feature_vector_keys"])
    forbidden_seen: set[str] = set()
    raw = {
        "ema9": 1.0,
        "rsi14": 50.0,
        "news_etf_flow": 1,
        "sentiment_score": 0.5,
        "llm_signal": 0.7,
        "gpt_token_count": 10,
        "ai_directive": 0.1,
    }
    out = _filter_features(
        raw, allowed_keys=allowed, forbidden_seen=forbidden_seen
    )
    assert "ema9" in out and "rsi14" in out
    for forbidden_key in (
        "news_etf_flow",
        "sentiment_score",
        "llm_signal",
        "gpt_token_count",
        "ai_directive",
    ):
        assert forbidden_key not in out
        assert forbidden_key in forbidden_seen


def test_build_slice_records_forbidden_keys_and_drops_them() -> None:
    allowed = set(load_allow_list()["feature_vector_keys"])
    forbidden_seen: set[str] = set()
    row = _make_row(datetime(2026, 4, 22, tzinfo=timezone.utc))
    portfolio = {
        "total_value": 1000.0,
        "peak_value": 1100.0,
        "day_start_value": 1050.0,
        "cash_balance": 600.0,
    }
    market_signal = {
        "funding_rate": 0.0001,
        "bid_ask_spread_bps": 8.0,
    }
    slc = build_slice(
        row,
        family="momentum",
        portfolio_snap=portfolio,
        market_signal=market_signal,
        recent_accuracy=0.55,
        allowed_feature_keys=allowed,
        forbidden_seen=forbidden_seen,
    )
    for flag in slc.anomaly_flags:
        assert not is_forbidden(flag)
    assert slc.regime == "bull"
    assert slc.strategy_family == "momentum"
    assert slc.slippage_bps == pytest.approx(8.0)
    assert slc.correlation_shift == pytest.approx(0.0001)
    for k in (
        "news_etf_flow",
        "news_whale_move",
        "sentiment_score",
        "llm_signal",
        "gpt_summary_tokens",
        "ai_meta_flag",
    ):
        assert k in forbidden_seen


def test_cluster_into_cycles_respects_window() -> None:
    base = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        {"created_at": base},
        {"created_at": base + timedelta(seconds=5)},
        {"created_at": base + timedelta(seconds=20)},
        {"created_at": base + timedelta(seconds=120)},
        {"created_at": base + timedelta(seconds=125)},
    ]
    cycles = cluster_into_cycles(rows, window_s=30)
    assert len(cycles) == 2
    assert len(cycles[0]) == 3
    assert len(cycles[1]) == 2


# ────────────── 4. honest outcome derivation ───────────────────────


def test_derive_outcome_never_fabricates_counterfactuals() -> None:
    losing = {
        "pnl_percent": -1.5,
        "position_size": 200.0,
        "entry_fee": 0.1,
        "entry_price": 100.0,
        "action": "buy",
    }
    out, _ = derive_outcome(
        losing, coin_candles=None, journal_during_hold=None
    )
    assert out.correct_defense is None
    assert out.correct_suppression is None
    assert out.missed_edge_cost is None
    # No journal → action_churn honestly reports zero flips, not 1.0.
    assert out.action_churn == 0.0
    # MAE falls back to |pnl_pct| when no candles are available.
    assert out.realized_drawdown == pytest.approx(0.015)


def test_derive_outcome_uses_real_price_candles_when_available() -> None:
    """MAE and stability MUST come from real price_candles when the
    fetcher returned rows for the trade window — no neutral
    placeholder values allowed."""
    trade = {
        "pnl_percent": 0.5,
        "position_size": 200.0,
        "entry_fee": 0.05,
        "entry_price": 100.0,
        "action": "buy",
    }
    candles = [
        {"high": 101.0, "low": 99.5, "close": 100.0},
        {"high": 100.8, "low": 96.0, "close": 99.0},  # adverse swing low
        {"high": 102.0, "low": 99.0, "close": 100.5},
        {"high": 101.5, "low": 100.0, "close": 100.5},
    ]
    out, deriv = derive_outcome(
        trade, coin_candles=candles, journal_during_hold=None
    )
    assert deriv["mae_source"] == "price_candles_low"
    assert out.realized_drawdown == pytest.approx((100.0 - 96.0) / 100.0)
    assert deriv["stability_source"] == "price_candles_close_stdev"
    assert 0.0 <= out.realized_stability <= 1.0
    # action_churn from the journal stream, not invented.
    journal = [
        {"direction": "up", "created_at": datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)},
        {"direction": "down", "created_at": datetime(2026, 4, 22, 12, 5, tzinfo=timezone.utc)},
        {"direction": "up", "created_at": datetime(2026, 4, 22, 12, 10, tzinfo=timezone.utc)},
    ]
    out2, deriv2 = derive_outcome(
        trade, coin_candles=candles, journal_during_hold=journal
    )
    assert deriv2["churn_source"] == "journal_direction_flips"
    assert out2.action_churn == pytest.approx(2.0)


def test_derive_outcome_prefers_position_marks_over_candles() -> None:
    """Task #491 — when paper_position_marks rows are available the
    intra-trade MAE / stability MUST come from the higher-fidelity
    mark stream (15s cadence) instead of the 5m candles.  This is
    the whole point of the new table; if the script silently kept
    using candles it would leak `realized_drawdown` <-> live truth
    and the meta-brain would never learn intra-trade defense."""
    trade = {
        "pnl_percent": 0.5,
        "position_size": 200.0,
        "entry_fee": 0.05,
        "entry_price": 100.0,
        "action": "buy",
    }
    # Candles claim a -4% adverse swing low; marks reveal the true
    # intra-trade trough was only -1.5% — the marks number must win.
    candles = [
        {"high": 101.0, "low": 99.5, "close": 100.0},
        {"high": 100.8, "low": 96.0, "close": 99.0},
        {"high": 102.0, "low": 99.0, "close": 100.5},
    ]
    base = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
    marks = [
        {"mark_price": 100.0, "marked_at": base},
        {"mark_price": 99.5,  "marked_at": base + timedelta(seconds=15)},
        {"mark_price": 98.5,  "marked_at": base + timedelta(seconds=30)},
        {"mark_price": 99.0,  "marked_at": base + timedelta(seconds=45)},
        {"mark_price": 100.5, "marked_at": base + timedelta(seconds=60)},
    ]
    out, deriv = derive_outcome(
        trade,
        coin_candles=candles,
        journal_during_hold=None,
        position_marks=marks,
    )
    assert deriv["mae_source"] == "position_marks"
    assert deriv["stability_source"] == "position_marks_returns_stdev"
    # Mark trough = 98.5 → MAE = (100 - 98.5)/100 = 0.015, NOT 0.04
    # (which is what the candle-low fallback would have produced).
    assert out.realized_drawdown == pytest.approx(0.015)
    assert 0.0 <= out.realized_stability <= 1.0
    # And: if marks are absent or empty, the candle path still runs.
    out2, deriv2 = derive_outcome(
        trade,
        coin_candles=candles,
        journal_during_hold=None,
        position_marks=[],
    )
    assert deriv2["mae_source"] == "price_candles_low"
    assert out2.realized_drawdown == pytest.approx((100.0 - 96.0) / 100.0)


def test_partial_outcome_skips_trust_terms_instead_of_zeroing() -> None:
    """If reward and trust treated `None` as zero we'd silently bias
    every replay toward "no defensive value, no missed edge"."""
    from market_meta_brain.domain.types import GovernanceOutcome
    from market_meta_brain.learning.reward_model import compute_meta_reward
    from market_meta_brain.learning.trust_model import StrategyTrustModel

    partial = GovernanceOutcome(
        realized_pnl=0.01,
        realized_drawdown=0.0,
        realized_stability=0.5,
        turnover_cost=0.0,
        action_churn=1.0,
        correct_defense=None,
        correct_suppression=None,
        missed_edge_cost=None,
    )
    zeros = GovernanceOutcome(
        realized_pnl=0.01,
        realized_drawdown=0.0,
        realized_stability=0.5,
        turnover_cost=0.0,
        action_churn=1.0,
        correct_defense=0.0,
        correct_suppression=0.0,
        missed_edge_cost=0.0,
    )
    r_partial = compute_meta_reward(partial)
    r_zeros = compute_meta_reward(zeros)
    assert r_partial == pytest.approx(r_zeros)
    model = StrategyTrustModel()
    model.ensure_family("momentum")
    model.learn_from_outcome({"momentum": partial}, regime="bull")


# ───────── 5. family taxonomy mirrors api-server adapter ───────────


def test_resolve_strategy_family_matches_adapter_taxonomy() -> None:
    cases = [
        ("Aggressive momentum trader, follows trends hard", "momentum"),
        ("Pure trend-following specialist using multi-EMA alignment", "momentum"),
        ("Contrarian who bets against the crowd", "mean_reversion"),
        ("Mean-reversion specialist", "mean_reversion"),
        ("Breakout and breakdown detection using Bollinger squeeze", "breakout"),
        ("Short-term momentum scalper targeting quick moves", "momentum"),
        ("Volume-focused analyst, tracks money flow", "volatility_forecaster"),
        ("LLM-evolved specialist (refined from Sentiment Sarah)", "volatility_forecaster"),
        ("Pattern recognition specialist", "baseline"),
        ("", "baseline"),
        (None, "baseline"),
    ]
    for personality, expected in cases:
        assert resolve_strategy_family(personality) == expected
    for _, expected in cases:
        assert expected in ALLOWED_FAMILIES


# ─────────── 6. manifest schema carries §467 fields ────────────────


def test_manifest_schema_carries_required_fields() -> None:
    """Drive build_manifest with a stub engine + service-like state
    and assert the schema the operator sees on disk includes every
    field the spec calls out: data_window, row_counts, regimes_observed,
    families_observed, final_trust_state, regime_prototype_count,
    episode_buffer_size, commit semantics, holdout_metrics, and the
    state label restricted to the documented vocabulary."""
    started = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=42)
    args = SimpleNamespace(
        window_s=30,
        holdout_pct=DEFAULT_HOLDOUT_PCT,
        min_trades=DEFAULT_MIN_TRADES,
        min_days=DEFAULT_MIN_DAYS,
        min_regimes=DEFAULT_MIN_REGIMES,
    )
    engine = SimpleNamespace(
        regimes_observed={"bull": 20, "panic_liquidation": 14, "range_chop": 8},
        families_observed={"momentum": 18, "mean_reversion": 12, "baseline": 12},
        cycles_replayed=42,
        holdout_cycles=8,
        trades_attributed=12,
        trades_unmatched=1,
        trades_skipped_holdout=3,
        forbidden_seen=set(),
    )
    final_trust = {"momentum": {"trust": 0.5, "stability": 0.5,
                                 "exploration_eligibility": 0.5,
                                 "failure_streak": 0,
                                 "recovery_score": 0.0,
                                 "last_regime": "bull"}}
    full_allow = load_allow_list()
    manifest = build_manifest(
        run_id="replay-test",
        started_at=started,
        finished_at=finished,
        args=args,
        columns_used=replay_meta_brain.load_allow_list_full(full_allow),
        forbidden_seen=["news_etf_flow", "sentiment_score"],
        source_counts={
            "prediction_journal_rows": 100,
            "paper_trades_closed": 12,
            "paper_positions_open": 0,
            "agents": 13,
            "paper_portfolios": 13,
            "strategy_snapshots": 200,
            "market_signals": 250,
            "price_candles_5m": 800,
            "cycles_total": 50,
            "train_cycles": 42,
            "holdout_cycles": 8,
        },
        engine=engine,
        pre_hashes={},
        post_hashes={"trust_model.json": "abc"},
        warmed_hashes={"trust_model.json": "warm"},
        sandbox_state=Path("/tmp/x"),
        commit_details={"requested": False, "promoted": False,
                        "thresholds_satisfied": False,
                        "thresholds": {"trades_attributed": 12,
                                       "min_trades": 2000,
                                       "span_days": 0.5,
                                       "min_days": 30,
                                       "distinct_regimes": 3,
                                       "min_regimes": 3}},
        state_label="pipeline_validation_only",
        final_trust_state=final_trust,
        final_trust_state_per_regime={"momentum": {"trust": 0.5, "regime": "bull"}},
        regime_prototype_count=4,
        episode_buffer_size=42,
        data_window={"min_ts": started.isoformat(),
                     "max_ts": finished.isoformat(), "days_covered": 0.0005},
        holdout_metrics={"cycle_count": 8, "avg_reward_proxy": 0.6,
                         "defensive_mode_hit_rate": 0.125,
                         "family_trust_calibration_error": {}},
    )
    for required in (
        "run_id", "task", "started_at", "finished_at", "cutoff_utc",
        "data_window", "window_seconds", "holdout_pct", "replay_mode",
        "state", "columns_used", "forbidden_columns_seen", "row_counts",
        "regimes_observed", "families_observed", "cycles_replayed",
        "holdout_cycle_count", "trades_attributed", "trades_unmatched",
        "trades_skipped_holdout",
        "final_trust_state", "final_trust_state_per_regime",
        "regime_prototype_count", "episode_buffer_size",
        "holdout_metrics", "brain_state", "commit", "commit_details",
        "constants",
    ):
        assert required in manifest, f"manifest missing field {required}"
    # State vocabulary
    assert manifest["state"] in ("production_ready", "pipeline_validation_only")
    assert manifest["replay_mode"] == "no_counterfactuals"
    # data_window contract uses the canonical keys from the spec.
    assert manifest["data_window"].keys() >= {"min_ts", "max_ts", "days_covered"}
    # row_counts canonical counters.
    assert manifest["row_counts"].keys() >= {
        "journal_rows_consumed", "cycles_replayed",
        "outcomes_recorded", "outcomes_skipped_no_match",
        "outcomes_skipped_holdout",
    }
    # regimes/families MUST be {name: count} dicts, not bare lists.
    assert isinstance(manifest["regimes_observed"], dict)
    assert isinstance(manifest["families_observed"], dict)
    assert all(isinstance(v, int) for v in manifest["regimes_observed"].values())
    assert all(isinstance(v, int) for v in manifest["families_observed"].values())
    # Top-level commit MUST be a boolean; details live in commit_details.
    assert isinstance(manifest["commit"], bool)
    assert "thresholds" in manifest["commit_details"]
    assert manifest["commit_details"]["thresholds"].keys() >= {
        "trades_attributed", "min_trades",
        "span_days", "min_days",
        "distinct_regimes", "min_regimes",
    }
    # columns_used MUST equal the full pinned allow-list (every bucket).
    for bucket in (
        "prediction_journal_columns",
        "feature_vector_keys",
        "paper_trades_columns",
        "paper_positions_columns",
        "paper_position_marks_columns",
        "paper_portfolios_columns",
        "agents_columns",
        "strategy_snapshots_columns",
        "market_signals_columns",
        "price_candles_columns",
    ):
        assert manifest["columns_used"][bucket] == full_allow[bucket], (
            f"columns_used[{bucket}] must echo dataset-columns.json verbatim"
        )


def test_state_label_vocabulary_is_bounded() -> None:
    """The two state labels the spec sanctions are
    `production_ready` and `pipeline_validation_only`. Older drafts
    used `warm`; assert it's gone from the script."""
    src = (SCRIPTS_ROOT / "replay_meta_brain.py").read_text(encoding="utf-8")
    assert '"warm"' not in src
    assert "production_ready" in src
    assert "pipeline_validation_only" in src


def test_holdout_metrics_writer_emits_required_fields(tmp_path) -> None:
    holdout_records = [
        {"ts": "2026-04-24T12:00:00+00:00",
         "dominant_regime": "bull",
         "caution_level": 0.2,
         "defensive_mode": "off",
         "suppressed_count": 0,
         "trust_map": {"momentum": 0.95, "baseline": 1.0}},
        {"ts": "2026-04-24T12:00:30+00:00",
         "dominant_regime": "panic_liquidation",
         "caution_level": 0.85,
         "defensive_mode": "hard",
         "suppressed_count": 1,
         "trust_map": {"momentum": 0.4, "baseline": 0.9}},
    ]
    final_trust = {
        "momentum": {"trust": 0.6}, "baseline": {"trust": 0.95},
    }
    out = compute_holdout_metrics(
        holdout_records=holdout_records,
        final_trust_state=final_trust,
        holdout_dir=tmp_path,
    )
    assert (tmp_path / "metrics.json").exists()
    assert out["cycle_count"] == 2
    assert 0.0 <= out["avg_reward_proxy"] <= 1.0
    assert out["defensive_mode_hit_rate"] == pytest.approx(0.5)
    assert "momentum" in out["family_trust_calibration_error"]
    cal = out["family_trust_calibration_error"]["momentum"]
    assert cal["samples"] == 2
    assert "absolute_error" in cal


# ───────────────── 7. episodic round-trip ──────────────────────────


def test_episodic_memory_round_trip() -> None:
    from market_meta_brain.domain.types import GovernanceEpisode, GovernanceOutcome
    from market_meta_brain.memory.episodic_market import EpisodicMarketMemory

    mem = EpisodicMarketMemory(capacity=8)
    for i in range(3):
        mem.push(
            GovernanceEpisode(
                timestamp=f"2026-04-22T12:00:0{i}+00:00",
                meta_state_vector=[0.1 * i, 0.2 * i, 0.3 * i],
                dominant_regime="bull",
                family_snapshot={"momentum": 0.05},
                action_summary={"caution_level": 0.4},
                reward=0.1 * i,
                outcome=GovernanceOutcome(
                    realized_pnl=0.01 * i,
                    realized_drawdown=0.0,
                    realized_stability=0.5,
                    turnover_cost=0.0,
                    action_churn=1.0,
                    correct_defense=None,
                    correct_suppression=None,
                    missed_edge_cost=None,
                ),
            )
        )
    snap = mem.state_dict()
    blob = json.loads(json.dumps(snap))
    fresh = EpisodicMarketMemory(capacity=2)
    restored = fresh.load_state_dict(blob)
    assert restored == 3
    assert fresh.capacity == 8
    assert len(fresh) == 3
    assert fresh.recent(1)[0].outcome.correct_defense is None
