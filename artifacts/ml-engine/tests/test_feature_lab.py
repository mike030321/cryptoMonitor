import numpy as np
import pandas as pd
import pytest

from app.training.feature_lab import (
    SUPPORTED_TRANSFORMS,
    _apply_transform,
)


def test_supported_transforms_are_allowlisted():
    expected = {
        "passthrough_existing",
        "log_realized_vol",
        "rsi_squared",
        "macd_x_atr",
        "ret5_minus_ret10",
        "bb_pctb_squared",
    }
    assert set(SUPPORTED_TRANSFORMS) == expected


def test_apply_transform_rsi_squared():
    df = pd.DataFrame({"rsi14": [40.0, 50.0, 60.0, 80.0]})
    out = _apply_transform(df, "rsi_squared", None, "rsi_sq_50")
    assert "rsi_sq_50" in out.columns
    np.testing.assert_allclose(
        out["rsi_sq_50"].to_numpy(), np.array([100.0, 0.0, 100.0, 900.0]),
    )


def test_apply_transform_passthrough_requires_source():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    out = _apply_transform(df, "passthrough_existing", "x", "x_copy")
    assert "x_copy" in out.columns
    np.testing.assert_allclose(out["x_copy"].to_numpy(), [1.0, 2.0])
    with pytest.raises(ValueError):
        _apply_transform(df, "passthrough_existing", None, "x_copy")
    with pytest.raises(ValueError):
        _apply_transform(df, "passthrough_existing", "missing_col", "x_copy")


def test_apply_transform_macd_x_atr():
    df = pd.DataFrame({"macdLine": [1.0, -2.0, 3.0], "atrPct": [0.5, 0.5, 2.0]})
    out = _apply_transform(df, "macd_x_atr", None, "feat")
    np.testing.assert_allclose(out["feat"].to_numpy(), [0.5, -1.0, 6.0])


def test_apply_transform_ret5_minus_ret10():
    df = pd.DataFrame({"ret5": [0.05, 0.10], "ret10": [0.02, 0.08]})
    out = _apply_transform(df, "ret5_minus_ret10", None, "diff")
    np.testing.assert_allclose(out["diff"].to_numpy(), [0.03, 0.02])


def test_apply_transform_bb_pctb_squared():
    df = pd.DataFrame({"bbPctB": [0.5, 0.0, 1.0]})
    out = _apply_transform(df, "bb_pctb_squared", None, "bb_sq")
    np.testing.assert_allclose(out["bb_sq"].to_numpy(), [0.0, 0.25, 0.25])


def test_apply_transform_log_realized_vol():
    df = pd.DataFrame({"realizedVol": [0.0, np.expm1(1.0)]})
    out = _apply_transform(df, "log_realized_vol", None, "lv")
    np.testing.assert_allclose(out["lv"].to_numpy(), [0.0, 1.0])


def test_apply_transform_unknown_kind_raises():
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError):
        _apply_transform(df, "definitely_not_real", None, "y")
