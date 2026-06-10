"""Offline training script — trains all models. Run locally or on HuggingFace ZeroGPU."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger
from bot.strategy.features import compute_features
from bot.strategy.regime_classifier import RegimeClassifier, label_regime
from bot.strategy.rl_agent import RLAgent
from bot.strategy.xgb_predictor import XGBPredictor
from bot.strategy.lstm_predictor import LSTMPredictor
from config import TRAINING_SYMBOLS, INITIAL_CAPITAL

DATA_DIR = "data/raw"

# Walk-forward split: train on 2015-2022, validate/test on 2023-present
TRAIN_CUTOFF = "2023-01-01"


def load_combined(cutoff: str | None = None) -> pd.DataFrame:
    frames = []
    for sym in TRAINING_SYMBOLS:
        path = f"{DATA_DIR}/{sym}.csv"
        if not os.path.exists(path):
            logger.warning(f"Missing data for {sym} — skipping.")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if cutoff:
            df = df[df.index < cutoff]
        df = compute_features(df)
        df["regime"] = label_regime(df)
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No data found. Run scripts/download_data.py first.")
    return pd.concat(frames).sort_index()


def main():
    logger.info(f"Loading training data (cutoff={TRAIN_CUTOFF}) — {len(TRAINING_SYMBOLS)} symbols...")
    df = load_combined(cutoff=TRAIN_CUTOFF)
    logger.info(f"Training rows: {len(df):,} | date range: {df.index.min().date()} → {df.index.max().date()}")

    logger.info("Training regime classifier...")
    regime_clf = RegimeClassifier()
    regime_clf.train(df)

    # Use combined data for XGBoost (more samples = better generalisation)
    logger.info("Training XGBoost predictor...")
    xgb = XGBPredictor()
    xgb.train(df.drop(columns=["symbol", "regime"], errors="ignore"))

    # Train LSTM on all symbols combined (normalized features remove price-unit cross-symbol bias).
    # Using full dataset instead of SPY-only exposes the model to different volatility regimes.
    logger.info("Training LSTM predictor on full multi-symbol dataset...")
    lstm = LSTMPredictor()
    lstm.train(df.drop(columns=["symbol", "regime"], errors="ignore"))

    # RL agent environment is single-asset — use SPY as the representative series
    primary = df[df["symbol"] == "SPY"].drop(columns=["symbol", "regime"], errors="ignore")
    logger.info("Training PPO RL agent...")
    agent = RLAgent()
    agent.train(primary, initial_balance=INITIAL_CAPITAL)

    logger.info("All models trained successfully.")


if __name__ == "__main__":
    main()
