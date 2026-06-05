"""Task #643 — quintile / sparse / post-cost label research module.

Parallel to ``app/training/labels.py``. Produces three NEW label
families on top of the same pre-feature dataset so a side-by-side
comparison versus the existing 3-class baseline is honest.

Modules:

* ``producers`` — pure functions that turn a forward-return series into
  label-family vectors (Q1..Q5 quintile, sparse top-decile, post-cost).
* ``data``     — dataset assembler that asof-joins the existing feature
  set, applies the BTC/ETH self-leak guard, and returns a per-(coin,
  timeframe) frame ready for training.
* ``runner``   — walk-forward trainer that fits one LightGBM model per
  (slice, family) and writes the holdout metrics blob the verdict
  report renders from. Models stay shadow — never promoted.
* ``cli``      — operator entry point: ``python -m
  app.training.labels_research.cli`` runs the full 6-slice × 4-family
  matrix and stamps the verdict-input JSON.

NONE of the code in this package modifies the existing 3-class
training pipeline, the verification gates, the promotion gate, or the
trading-frictions config. The new label families are research artefacts
only.
"""
