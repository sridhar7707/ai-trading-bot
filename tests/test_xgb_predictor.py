import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from tests.conftest import make_ohlcv
from bot.strategy.features import compute_features


@pytest.fixture()
def feature_df():
    raw = make_ohlcv(270)
    df = compute_features(raw)
    df["regime"] = 0
    return df.reset_index(drop=True)


def test_xgb_predictor_predict_returns_float_without_model():
    from bot.strategy.xgb_predictor import XGBPredictor
    predictor = XGBPredictor()
    predictor.model = None
    result = predictor.predict_proba(pd.Series({col: 0.0 for col in ["rsi", "macd_diff_pct"]}))
    assert isinstance(result, float)


def test_xgb_predictor_predict_returns_probability_range(feature_df):
    from bot.strategy.xgb_predictor import XGBPredictor
    predictor = XGBPredictor()
    if predictor.model is None:
        pytest.skip("No XGBoost model on disk")
    row = feature_df.iloc[-1]
    prob = predictor.predict_proba(row)
    assert 0.0 <= prob <= 1.0


def test_xgb_predictor_train_skips_without_xgboost(feature_df):
    from bot.strategy.xgb_predictor import XGBPredictor
    predictor = XGBPredictor()
    with patch.dict("sys.modules", {"xgboost": None}):
        predictor.train(feature_df)
