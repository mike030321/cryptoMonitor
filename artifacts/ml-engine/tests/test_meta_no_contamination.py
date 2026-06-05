"""Task #455 — Meta-model dataset no-contamination contract.

Locks in the rule that the meta-model training frame may NEVER pull a
column whose name carries a forbidden prefix from the pre-#444
LLM/news/sidecar era. The allow-list is the union of:

  - Top-level prediction_journal columns the SQL in
    `meta_dataset.build_meta_dataset` selects.
  - The single jsonb key (`specialists`) it reads out of
    `gates_applied`.
  - Every column the dataset emits as a feature
    (`META_FEATURE_COLUMNS`) or as a label (`__action__`,
    `__edge_after_costs__`, `__timeframe__`, `__coin_id__`,
    `__created_at__`).

Three independent assertions:

  (a) `meta_dataset._QUERY` literally selects only the columns the
      pinned allow-list says it does. A future schema mistake that
      pulls e.g. `feature_vector` (which still carries `news_*`
      keys) fails this assertion immediately.

  (b) Every `META_FEATURE_COLUMNS` entry passes the deny-prefix
      check.

  (c) Every column on the actual built DataFrame (when one can be
      built — empty journal in CI is fine, the column list is
      still canonical) passes the deny-prefix check.

This test is wired into the `per-coin-isolation` validation
workflow alongside the other contract tests so the rebuild stays
locked across future refactors.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from app.training import meta_dataset

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
COLUMNS_FILE = FIXTURES_DIR / "meta-dataset-columns.json"

FORBIDDEN_PREFIXES = (
    "news_",
    "llm_",
    "gpt_",
    "sentiment_",
    "ai_",
    "benchmark_",
    "alpha_",
    "baseline_",
    "equity_",
    "strategy_lab_",
)


def _is_forbidden(col: str) -> bool:
    lc = col.lower()
    return any(lc.startswith(p) for p in FORBIDDEN_PREFIXES)


def _pinned() -> dict:
    """Load the canonical column allow-list. Lives in
    `tests/fixtures/meta-dataset-columns.json` so the contract test is
    self-contained on a clean CI checkout (no dependency on the
    `.local/...` evidence bundle, which is ephemeral)."""
    assert COLUMNS_FILE.exists(), (
        f"Pinned dataset column allow-list missing: {COLUMNS_FILE}. "
        "It should be tracked under tests/fixtures/."
    )
    return json.loads(COLUMNS_FILE.read_text())


def test_dataset_query_selects_only_pinned_columns():
    """The literal SELECT clause in meta_dataset._QUERY must touch only
    the columns the pinned allow-list declares. A regression that adds
    `pj.feature_vector` (which still carries `news_*` keys) is caught
    here before it can poison a training run.
    """
    pinned = _pinned()
    allowed = set(pinned["prediction_journal_columns_read"])
    # Strip comments / parens / aliases, keep only identifiers after `pj.`.
    pj_refs = set(re.findall(r"pj\.([a-z_][a-z0-9_]*)", meta_dataset._QUERY))
    extra = pj_refs - allowed
    assert not extra, (
        f"meta_dataset._QUERY references unpinned columns: {sorted(extra)}. "
        f"Either add them to dataset-columns.json (after auditing for "
        "contamination) or remove the SELECT."
    )


def test_meta_feature_columns_have_no_forbidden_prefix():
    """The hard list `META_FEATURE_COLUMNS` is the canonical column
    set the trainer fits on. None may carry a forbidden prefix."""
    bad = [c for c in meta_dataset.META_FEATURE_COLUMNS if _is_forbidden(c)]
    assert not bad, (
        f"META_FEATURE_COLUMNS contains forbidden-prefix columns: {bad}. "
        "Remove the column or update FORBIDDEN_PREFIXES in this test "
        "with a written justification."
    )


def test_pinned_column_set_matches_meta_feature_columns():
    """The pinned allow-list must stay in lockstep with
    META_FEATURE_COLUMNS so a column added in code without a doc
    update fails the test."""
    pinned = _pinned()
    pinned_features = set(pinned["feature_columns_emitted"])
    code_features = set(meta_dataset.META_FEATURE_COLUMNS)
    missing = code_features - pinned_features
    extra = pinned_features - code_features
    assert not missing and not extra, (
        f"Pinned allow-list drift. missing_in_pin={sorted(missing)} "
        f"extra_in_pin={sorted(extra)}. Update "
        ".local/cleanup/meta-rebuild/dataset-columns.json."
    )


@pytest.mark.asyncio
async def test_built_dataset_columns_have_no_forbidden_prefix(monkeypatch):
    """Build the dataset against an empty pool (no journal rows) so the
    test is hermetic, and assert the canonical column list the function
    returns has no forbidden-prefix entries."""

    class _EmptyConn:
        async def fetch(self, *_a, **_kw):
            return []

    class _EmptyPool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self_):
                    return _EmptyConn()

                async def __aexit__(self_, *_a):
                    return False

            return _Ctx()

    async def _fake_init_pool():
        return _EmptyPool()

    from app import db as db_mod

    monkeypatch.setattr(db_mod, "init_pool", _fake_init_pool)

    df = await meta_dataset.build_meta_dataset(timeframe="1h")
    assert isinstance(df, pd.DataFrame)
    bad = [c for c in df.columns if _is_forbidden(c)]
    assert not bad, (
        f"Built meta-dataset DataFrame has forbidden-prefix columns: {bad}."
    )
