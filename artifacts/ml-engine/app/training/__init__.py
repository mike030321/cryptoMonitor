"""Phase-2 training pipeline.

Builds labeled feature frames from real Postgres OHLCV, runs walk-forward
validation, trains a logistic-regression baseline + a LightGBM classifier
with isotonic calibration, and persists models to the on-disk registry.
"""
