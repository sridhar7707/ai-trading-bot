import pandas as pd
import numpy as np
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to an OHLCV DataFrame."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Momentum
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Trend
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["sma_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()

    # Volatility
    bb = ta.volatility.BollingerBands(close)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    # Volume
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["volume_sma"] = volume.rolling(20).mean()
    df["volume_ratio"] = (volume / df["volume_sma"]).replace([np.inf, -np.inf], np.nan)

    # Price action
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["hl_ratio"] = (high - low) / close
    df["norm_close"] = (close - close.rolling(20).mean()) / close.rolling(20).std()

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "rsi", "stoch_k", "stoch_d",
    "macd", "macd_signal", "macd_diff",
    "ema_20", "ema_50", "sma_20",
    "bb_high", "bb_low", "bb_width", "atr",
    "obv", "volume_sma", "volume_ratio",
    "returns", "log_returns", "hl_ratio", "norm_close",
]
