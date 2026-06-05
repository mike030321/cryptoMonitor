"""Test-suite hyperparameter overrides.

Optuna defaults take ~30s per slice; with 4+ slices per fixture the suite
exceeds the 120s shell cap. We shrink the budget aggressively for tests —
correctness checks (calibration monotonicity, fallback resolution, dataset
persistence) don't need a tuned model.

These env vars must be set BEFORE `app.training.train` is imported, so this
conftest is loaded by pytest before any test module.
"""
from __future__ import annotations

import os

os.environ.setdefault("ML_OPTUNA_N_TRIALS", "2")
os.environ.setdefault("ML_OPTUNA_TIMEOUT_SECONDS", "5")
os.environ.setdefault("ML_LGB_NUM_BOOST_ROUND", "30")
# Skip the Optuna search entirely in tests — we just need a trained model,
# not a tuned one. Production training (train.py CLI) leaves this unset.
os.environ.setdefault("ML_SKIP_OPTUNA", "1")

# Auto-retrain scheduler kicks off a background training thread on app
# startup (task #105). Tests use the FastAPI lifespan many times via
# TestClient — we never want that thread to actually fire during the
# suite, so disable it explicitly. Production containers don't set this.
os.environ.setdefault("ML_AUTO_RETRAIN_TEST_DISABLE", "1")

# Silence Optuna's per-trial INFO chatter — keeps test output readable.
import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
