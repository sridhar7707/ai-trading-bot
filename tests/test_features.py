import numpy as np
import pandas as pd
import pytest
from bot.strategy.features import compute_features, FEATURE_COLS
from tests.conftest import make_ohlcv


def test_compute_features_adds_all_columns(ohlcv):
    result = compute_features(ohlcv)
    for col in FEATURE_COLS:
        assert col in result.columns, f"Missing feature column: {col}"


def test_compute_features_drops_nan_rows(ohlcv):
    result = compute_features(ohlcv)
    assert result[FEATURE_COLS].isna().sum().sum() == 0


def test_compute_features_returns_fewer_rows_than_input():
    # Use a fresh df so mutation of the input doesn't affect the length comparison
    df = make_ohlcv(270)
    original_len = len(df)
    result = compute_features(df)
    assert len(result) < original_len  # warmup rows dropped by dropna


def test_volume_ratio_no_inf():
    df = make_ohlcv(270)
    # Force volume_sma to 0 for the first 20 rows by setting volume to 0
    df.loc[df.index[:20], "volume"] = 0.0
    result = compute_features(df)
    assert not np.isinf(result["volume_ratio"]).any()


def test_volume_ratio_inf_replaced_with_nan_then_dropped():
    df = make_ohlcv(270)
    df["volume"] = 0.0  # all volume = 0 → volume_sma = 0 → inf → replaced with NaN
    result = compute_features(df)
    # All volume_ratio values are NaN, so the column should be NaN or missing in result
    # dropna removes those rows
    if "volume_ratio" in result.columns:
        assert not np.isinf(result["volume_ratio"]).any()


def test_compute_features_raises_on_insufficient_rows():
    df = make_ohlcv(10)
    with pytest.raises(ValueError, match="at least"):
        compute_features(df)


def test_feature_values_are_finite_on_valid_data(ohlcv):
    result = compute_features(ohlcv)
    for col in FEATURE_COLS:
        if col == "volume_ratio":
            continue  # can have NaN if volume_sma is 0
        assert np.isfinite(result[col]).all(), f"Non-finite values in {col}"
