import pandas as pd
import numpy as np
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to an OHLCV DataFrame."""
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    eps    = 1e-8  # guard against division by zero in normalizations

    # Momentum
    df["rsi"]     = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch         = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Trend (raw — kept for risk management use; FEATURE_COLS uses normalized versions)
    macd            = ta.trend.MACD(close)
    df["macd"]      = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["ema_20"]    = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"]    = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["sma_20"]    = ta.trend.SMAIndicator(close, window=20).sma_indicator()

    # Volatility (raw ATR kept for stop-loss calculations in risk_manager)
    bb             = ta.volatility.BollingerBands(close)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()   # already a ratio — no normalization needed
    df["atr"]      = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    # Volume
    df["obv"]         = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["volume_sma"]  = volume.rolling(20).mean()
    df["volume_ratio"] = (volume / df["volume_sma"]).replace([np.inf, -np.inf], np.nan)

    # Price action (already unitless — no normalization needed)
    df["returns"]   = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["hl_ratio"]  = (high - low) / close
    df["norm_close"] = (close - close.rolling(20).mean()) / close.rolling(20).std()

    # ── Normalized features (price-unit features ÷ close price) ──────────────
    # Raw MACD/EMA/OBV values are in dollar/volume units that differ wildly
    # across symbols (NVDA≈$900 vs SPY≈$520). A single shared model cannot
    # generalize without price-relative features.
    df["macd_pct"]      = df["macd"]        / (close + eps)
    df["macd_sig_pct"]  = df["macd_signal"] / (close + eps)
    df["macd_diff_pct"] = df["macd_diff"]   / (close + eps)
    df["ema20_pct"]     = df["ema_20"]      / (close + eps) - 1  # % above/below close
    df["ema50_pct"]     = df["ema_50"]      / (close + eps) - 1
    df["sma20_pct"]     = df["sma_20"]      / (close + eps) - 1
    df["bb_high_pct"]   = df["bb_high"]     / (close + eps) - 1
    df["bb_low_pct"]    = df["bb_low"]      / (close + eps) - 1
    df["atr_pct"]       = df["atr"]         / (close + eps)       # ATR as fraction of price
    vol_sma_safe        = df["volume_sma"].replace(0, np.nan)
    df["obv_chg_pct"]   = df["obv"].diff()  / (vol_sma_safe + eps)  # OBV change per avg-volume unit

    df.dropna(inplace=True)
    return df


# NOTE: FEATURE_COLS changed from raw-dollar to normalized features.
# Saved XGBoost and LSTM models are INVALID — re-run scripts/train_model.py.
FEATURE_COLS = [
    # Already unitless
    "rsi", "stoch_k", "stoch_d",
    "bb_width", "volume_ratio",
    "returns", "log_returns", "hl_ratio", "norm_close",
    # Normalized to price (comparable across any symbol/price level)
    "macd_pct", "macd_sig_pct", "macd_diff_pct",
    "ema20_pct", "ema50_pct", "sma20_pct",
    "bb_high_pct", "bb_low_pct",
    "atr_pct",
    "obv_chg_pct",
]
