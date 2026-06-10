from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from loguru import logger
from config import (
    MODEL_SAVE_PATH, RL_TIMESTEPS, RL_LEARNING_RATE,
    RL_N_STEPS, RL_BATCH_SIZE, RL_N_EPOCHS,
)
from bot.strategy.features import FEATURE_COLS


class TradingEnv(gym.Env):
    """Custom Gym environment for single-symbol paper trading."""

    def __init__(self, df: pd.DataFrame, initial_balance: float = 1000.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        # Observation: 16 indicators + balance_ratio + shares_held + regime = 19
        obs_dim = len(FEATURE_COLS) + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # Actions: 0=Hold, 1=Buy 20% of cash, 2=Sell all
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.returns_history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        indicators = row[FEATURE_COLS].values.astype(np.float32)
        balance_ratio = self.balance / self.initial_balance
        shares_norm = self.shares_held / 100.0
        regime = float(row.get("regime", 0))
        return np.concatenate([indicators, [balance_ratio, shares_norm, regime]]).astype(np.float32)

    def step(self, action: int):
        row = self.df.iloc[self.current_step]
        price = float(row["close"])

        if action == 1 and self.balance > 1:  # Buy
            spend = self.balance * 0.20
            self.shares_held += spend / price
            self.balance -= spend
        elif action == 2 and self.shares_held > 0:  # Sell
            self.balance += self.shares_held * price
            self.shares_held = 0.0

        portfolio_value = self.balance + self.shares_held * price
        ret = (portfolio_value - self.initial_balance) / self.initial_balance
        self.returns_history.append(ret)

        # Sharpe proxy reward
        if len(self.returns_history) > 1:
            r = np.array(self.returns_history)
            reward = float(np.mean(r) / (np.std(r) + 1e-8))
        else:
            reward = 0.0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_obs(), reward, done, False, {}


class RLAgent:
    def __init__(self):
        self.model: PPO | None = None
        self._load()

    def _load(self):
        try:
            self.model = PPO.load(MODEL_SAVE_PATH)
            logger.info("PPO model loaded from disk.")
        except Exception as e:
            logger.warning(f"PPO model not loaded: {e}")

    def train(self, df: pd.DataFrame, initial_balance: float = 1000.0):
        env = TradingEnv(df, initial_balance)
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=RL_LEARNING_RATE,
            n_steps=RL_N_STEPS,
            batch_size=RL_BATCH_SIZE,
            n_epochs=RL_N_EPOCHS,
            verbose=1,
        )
        self.model.learn(total_timesteps=RL_TIMESTEPS)
        self.model.save(MODEL_SAVE_PATH)
        logger.info(f"PPO model trained and saved to {MODEL_SAVE_PATH}")

    def predict(self, obs: np.ndarray) -> int:
        if self.model is None:
            return 0  # Hold if no model
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
