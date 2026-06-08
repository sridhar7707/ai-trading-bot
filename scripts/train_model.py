"""Offline training script — trains all models. Run locally or on HuggingFace ZeroGPU."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger
from bot.strategy.features import compute_features
from bot.strategy.regime_classifier import RegimeClassifier, label_regime
from bot.strategy.rl_agent import RLAgent
from bot.strategy.xgb_predictor import XGBPredictor
from bot.strategy.lstm_predictor import LSTMPredictor
from config import SYMBOLS, INITIAL_CAPITAL

DATA_DIR = "data/raw"


def load_combined() -> pd.DataFrame:
    frames = []
    for sym in SYMBOLS:
        path = f"{DATA_DIR}/{sym}.csv"
        if not os.path.exists(path):
            logger.warning(f"Missing data for {sym} — skipping.")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = compute_features(df)
        df["regime"] = label_regime(df)
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No data found. Run scripts/download_data.py first.")
    return pd.concat(frames).sort_index()


def main():
    logger.info("Loading training data...")
    df = load_combined()

    logger.info("Training regime classifier...")
    regime_clf = RegimeClassifier()
    regime_clf.train(df)

    # Use combined data for XGBoost (more samples = better generalisation)
    logger.info("Training XGBoost predictor...")
    xgb = XGBPredictor()
    xgb.train(df.drop(columns=["symbol", "regime"], errors="ignore"))

    # Use SPY as primary single-asset series for sequence models
    primary = df[df["symbol"] == "SPY"].drop(columns=["symbol", "regime"], errors="ignore")

    logger.info("Training LSTM predictor...")
    lstm = LSTMPredictor()
    lstm.train(primary)

    logger.info("Training PPO RL agent...")
    agent = RLAgent()
    agent.train(primary, initial_balance=INITIAL_CAPITAL)

    logger.info("All models trained successfully.")


if __name__ == "__main__":
    main()
