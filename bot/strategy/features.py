import pandas as pd
import numpy as np
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to an OHLCV DataFrame."""
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    eps    = 1e-8

    # Momentum
    df["rsi"]     = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch         = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Trend (raw — kept for risk management; FEATURE_COLS uses normalized versions)
    macd              = ta.trend.MACD(close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    df["ema_20"]      = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"]      = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["sma_20"]      = ta.trend.SMAIndicator(close, window=20).sma_indicator()

    # Volatility (raw ATR kept for stop-loss calculations in risk_manager)
    bb             = ta.volatility.BollingerBands(close)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr"]      = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    # Volume
    df["obv"]          = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["volume_sma"]   = volume.rolling(20).mean()
    df["volume_ratio"] = (volume / df["volume_sma"]).replace([np.inf, -np.inf], np.nan)

    # Money Flow Index — volume-weighted RSI, captures buying/selling pressure
    df["mfi"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()

    # Price action (already unitless)
    df["returns"]    = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["hl_ratio"]   = (high - low) / (close + eps)
    df["norm_close"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + eps)

    # VWAP deviation — where price sits relative to intraday fair value
    typical_price  = (high + low + close) / 3
    vwap_rolling   = (typical_price * volume).rolling(20).sum() / (df["volume_sma"] + eps)
    df["vwap_dev"] = (close - vwap_rolling) / (vwap_rolling + eps)

    # 15-minute RSI (multi-timeframe momentum) — resampled to 5-min frequency
    # Requires a DatetimeIndex; falls back to standard RSI when unavailable.
    if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
        try:
            bars_15m       = df[["close"]].resample("15min").last().dropna()
            rsi_15m_series = ta.momentum.RSIIndicator(bars_15m["close"], window=14).rsi()
            df["rsi_15m"]  = rsi_15m_series.reindex(df.index, method="ffill")
        except Exception:
            df["rsi_15m"] = df["rsi"]
    else:
        df["rsi_15m"] = df["rsi"]

    # ── Normalized features (price-unit → ratio) ──────────────────────────────
    df["macd_pct"]      = df["macd"]        / (close + eps)
    df["macd_sig_pct"]  = df["macd_signal"] / (close + eps)
    df["macd_diff_pct"] = df["macd_diff"]   / (close + eps)
    df["ema20_pct"]     = df["ema_20"]      / (close + eps) - 1
    df["ema50_pct"]     = df["ema_50"]      / (close + eps) - 1
    df["sma20_pct"]     = df["sma_20"]      / (close + eps) - 1
    df["bb_high_pct"]   = df["bb_high"]     / (close + eps) - 1
    df["bb_low_pct"]    = df["bb_low"]      / (close + eps) - 1
    df["atr_pct"]       = df["atr"]         / (close + eps)
    vol_sma_safe        = df["volume_sma"].replace(0, np.nan)
    df["obv_chg_pct"]   = df["obv"].diff()  / (vol_sma_safe + eps)

    df.dropna(inplace=True)
    return df


# NOTE: FEATURE_COLS changed — saved XGBoost and LSTM models must be retrained.
FEATURE_COLS = [
    # Momentum (unitless 0-100 oscillators)
    "rsi", "stoch_k", "stoch_d", "mfi",
    # Volatility / structure (already ratios)
    "bb_width", "volume_ratio",
    # Price action (unitless)
    "returns", "log_returns", "hl_ratio", "norm_close",
    # VWAP & multi-timeframe
    "vwap_dev", "rsi_15m",
    # Trend (normalized to price — cross-symbol comparable)
    "macd_pct", "macd_sig_pct", "macd_diff_pct",
    "ema20_pct", "ema50_pct", "sma20_pct",
    "bb_high_pct", "bb_low_pct",
    "atr_pct",
    "obv_chg_pct",
]
