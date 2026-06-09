import os
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from bot.strategy.features import FEATURE_COLS

XGB_MODEL_PATH = "models/saved/xgb_predictor.pkl"
FORWARD_PERIODS = 5


class XGBPredictor:
    def __init__(self):
        self.model = None
        self._load()

    def _load(self):
        if os.path.exists(XGB_MODEL_PATH):
            try:
                self.model = joblib.load(XGB_MODEL_PATH)
                logger.info("XGBoost model loaded.")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost model: {e}")
        else:
            logger.warning("No XGBoost model found — will need training.")

    def train(self, df: pd.DataFrame):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("xgboost not installed. Run: pip install xgboost")
            return

        df = df.copy()
        df["target"] = (df["close"].shift(-FORWARD_PERIODS) > df["close"]).astype(int)
        df.dropna(inplace=True)

        # Temporal split — train only on the first 80% so the model never sees
        # future prices during training (prevents lookahead bias / overfitting).
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        mask = train_df[FEATURE_COLS].notna().all(axis=1)
        X = train_df.loc[mask, FEATURE_COLS]
        y = train_df.loc[mask, "target"]

        self.model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X, y)

        val_mask = val_df[FEATURE_COLS].notna().all(axis=1)
        X_val = val_df.loc[val_mask, FEATURE_COLS]
        y_val = val_df.loc[val_mask, "target"]
        if len(X_val) > 0:
            val_acc = float((self.model.predict(X_val) == y_val).mean())
            if val_acc < 0.50:
                logger.warning(
                    f"XGBoost val accuracy {val_acc:.3f} is below chance — model may have degraded. "
                    "Review training data before deploying."
                )
            else:
                logger.info(f"XGBoost val accuracy (holdout 20%): {val_acc:.3f}")

        os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, XGB_MODEL_PATH)
        logger.info(f"XGBoost trained and saved to {XGB_MODEL_PATH}")

    def predict_proba(self, row: pd.Series) -> float:
        """Return probability (0-1) that price will be higher in {FORWARD_PERIODS} candles."""
        if self.model is None:
            return 0.5
        try:
            features = row[FEATURE_COLS].values.reshape(1, -1)
            return float(self.model.predict_proba(features)[0, 1])
        except Exception as e:
            logger.error(f"XGBoost predict failed: {e}")
            return 0.5
