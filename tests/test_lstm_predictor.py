import pytest
import pandas as pd
import numpy as np
from tests.conftest import make_ohlcv
from bot.strategy.features import compute_features


@pytest.fixture()
def feature_df():
    raw = make_ohlcv(270)
    df = compute_features(raw)
    df["regime"] = 0
    return df.reset_index(drop=True)


def test_lstm_predictor_predict_returns_float_without_model():
    from bot.strategy.lstm_predictor import LSTMPredictor
    predictor = LSTMPredictor()
    predictor.model = None
    row = pd.Series(np.zeros(20))
    result = predictor.predict_proba(row)
    assert isinstance(result, float)


def test_lstm_predictor_predict_no_model_returns_zero():
    from bot.strategy.lstm_predictor import LSTMPredictor
    predictor = LSTMPredictor()
    predictor.model = None
    row = pd.Series({"rsi": 50.0, "macd_diff_pct": 0.01})
    assert predictor.predict_proba(row) == 0.5


def test_lstm_predictor_predict_range_with_model(feature_df):
    from bot.strategy.lstm_predictor import LSTMPredictor
    predictor = LSTMPredictor()
    if predictor.model is None:
        pytest.skip("No LSTM model on disk")
    row = feature_df.iloc[-1]
    prob = predictor.predict_proba(row)
    assert 0.0 <= prob <= 1.0
