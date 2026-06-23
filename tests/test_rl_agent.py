import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv
from bot.strategy.features import compute_features, FEATURE_COLS


@pytest.fixture()
def feature_df():
    raw = make_ohlcv(270)
    df = compute_features(raw)
    df["regime"] = 0
    return df.reset_index(drop=True)


def test_trading_env_reset_returns_correct_shape(feature_df):
    from bot.strategy.rl_agent import TradingEnv
    env = TradingEnv(feature_df)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_trading_env_step_hold(feature_df):
    from bot.strategy.rl_agent import TradingEnv
    env = TradingEnv(feature_df)
    env.reset()
    obs, reward, done, truncated, info = env.step(0)  # Hold
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_trading_env_step_buy_then_sell(feature_df):
    from bot.strategy.rl_agent import TradingEnv
    env = TradingEnv(feature_df, initial_balance=1000.0)
    env.reset()
    env.step(1)  # Buy
    assert env.shares_held > 0
    env.step(2)  # Sell
    assert env.shares_held == 0.0


def test_rl_agent_predict_returns_int_when_no_model():
    from bot.strategy.rl_agent import RLAgent
    agent = RLAgent()
    agent.model = None
    obs = np.zeros(19, dtype=np.float32)
    action = agent.predict(obs)
    assert action == 0  # Hold fallback
