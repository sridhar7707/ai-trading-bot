from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
from config import REGIME_MODEL_PATH

REGIMES = {0: "TRENDING_UP", 1: "TRENDING_DOWN", 2: "RANGING", 3: "HIGH_VOLATILITY"}

ATR_VOLATILITY_THRESHOLD = 0.03  # atr/close ratio above which market is HIGH_VOLATILITY
REGIME_LOOKAHEAD = 20            # bars to look FORWARD when generating training labels


def label_regime(df: pd.DataFrame) -> pd.Series:
    """
    Generate regime labels based on what price DOES over the next REGIME_LOOKAHEAD bars.
    The model then learns to predict future regime from current features — not to
    replicate the current RSI/MACD rules, which was the old circular labeling approach.

    Labels are training-only; live prediction uses the trained RandomForest on current features.
    """
    closes = df["close"].values
    atrs   = df["atr"].values if "atr" in df.columns else np.zeros(len(df))
    labels = []

    for i in range(len(df)):
        atr_ratio = atrs[i] / closes[i] if closes[i] > 0 else 0.0
        if atr_ratio > ATR_VOLATILITY_THRESHOLD:
            labels.append(3)  # HIGH_VOLATILITY — detected from current bar, not future
            continue

        end = min(i + REGIME_LOOKAHEAD, len(df) - 1)
        if end <= i:
            labels.append(2)  # RANGING — not enough future data at tail of series
            continue

        window      = closes[i: end + 1]
        fwd_return  = (window[-1] - window[0]) / (window[0] + 1e-8)
        fwd_rets    = np.diff(window) / (window[:-1] + 1e-8)
        fwd_sharpe  = float(np.mean(fwd_rets) / (np.std(fwd_rets) + 1e-8)) if len(fwd_rets) > 1 else 0.0

        if fwd_sharpe > 0.5 and fwd_return > 0.005:
            labels.append(0)  # TRENDING_UP — consistent positive move over next 20 bars
        elif fwd_sharpe < -0.5 and fwd_return < -0.005:
            labels.append(1)  # TRENDING_DOWN
        else:
            labels.append(2)  # RANGING

    return pd.Series(labels, index=df.index)


REGIME_FEATURES = [
    "rsi", "macd_diff_pct", "bb_width", "atr_pct",
    "volume_ratio", "norm_close", "returns",
]


class RegimeClassifier:
    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self._load()

    def _load(self):
        try:
            self.model = joblib.load(REGIME_MODEL_PATH)
            logger.info("Regime classifier loaded from disk.")
        except FileNotFoundError:
            logger.warning("No regime classifier found — using rule-based fallback.")

    def train(self, df: pd.DataFrame):
        df = df.copy().reset_index(drop=True)
        df["regime"] = label_regime(df)
        mask = df[REGIME_FEATURES].notna().all(axis=1)
        X = df.loc[mask, REGIME_FEATURES]
        y = df.loc[mask, "regime"]
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        joblib.dump(self.model, REGIME_MODEL_PATH)
        logger.info("Regime classifier trained and saved.")

    def predict(self, row: pd.Series) -> int:
        if self.model is not None:
            features = pd.DataFrame([row[REGIME_FEATURES]])
            return int(self.model.predict(features)[0])
        return self._rule_based(row)

    def _rule_based(self, row: pd.Series) -> int:
        close = row.get("close", 1) or 1  # guard against 0
        atr_ratio = row.get("atr", 0) / close
        if atr_ratio > ATR_VOLATILITY_THRESHOLD:
            return 3
        rsi = row.get("rsi", 50)
        macd_diff = row.get("macd_diff", 0)
        if rsi > 55 and macd_diff > 0:
            return 0
        if rsi < 45 and macd_diff < 0:
            return 1
        return 2

    def regime_name(self, code: int) -> str:
        return REGIMES.get(code, "UNKNOWN")
