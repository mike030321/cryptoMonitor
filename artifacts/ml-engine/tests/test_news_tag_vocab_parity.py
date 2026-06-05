"""Canonical-column guard for the Python news tag feature pipeline.

Historical context: this file used to also contain
``test_news_tag_vocab_parity_with_ts``, which read
``artifacts/api-server/src/lib/news-classifier.ts`` and asserted that the
TS ``TAG_VOCABULARY`` matched the Python ``NEWS_TAG_VOCABULARY`` exactly.

That parity test was removed (Task #543) because the TS-side news
classifier was deleted as part of the Quant-Only Enforcement work
(Task #365): ``news-classifier.ts`` no longer exists, ``build_feature_vector``
no longer emits the ``news_*`` block, and the test was failing on every CI
run with a bare ``FileNotFoundError``.

Do not re-add a TS parity check here unless a new TS source-of-truth for
the news tag vocabulary is reintroduced. The Python ``news_tag_features``
helper is still used by parquet backfills, so the canonical-column
contract below is still worth guarding.
"""
from __future__ import annotations

from app.features import NEWS_TAG_VOCABULARY


def test_news_tag_features_emits_canonical_columns() -> None:
    from app.features import news_tag_features

    out = news_tag_features(["etf_flow", "macro_shock"])
    expected_keys = {f"news_{t}" for t in NEWS_TAG_VOCABULARY}
    assert set(out.keys()) == expected_keys
    assert out["news_etf_flow"] == 1.0
    assert out["news_macro_shock"] == 1.0
    assert out["news_regulatory_risk"] == 0.0
