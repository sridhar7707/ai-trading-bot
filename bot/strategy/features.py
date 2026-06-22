import pandas as pd
import numpy as np
import ta


# Multi-period momentum features (ret_126d, mom_12_1, high_52w_pct) need 252+ bars.
# Live bot fetches period="2y" (~504 bars); training CSVs go back to 2007 (~4500 bars).
MIN_BARS = 260

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to an OHLCV DataFrame."""
    if len(df) < MIN_BARS:
        raise ValueError(f"compute_features requires at least {MIN_BARS} bars, got {len(df)}")
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    eps    = 1e-8

    # Momentum oscillators
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

    # VWAP deviation — 20-bar volume-weighted average price vs current close.
    # Denominator must be rolling SUM of volume (not mean) for true VWAP.
    typical_price  = (high + low + close) / 3
    vol_sum_20     = volume.rolling(20).sum()
    vwap_rolling   = (typical_price * volume).rolling(20).sum() / (vol_sum_20 + eps)
    df["vwap_dev"] = (close - vwap_rolling) / (vwap_rolling + eps)

    # 15-minute RSI (multi-timeframe) — kept for compatibility; on daily bars equals rsi
    if isinstance(df.index, pd.DatetimeIndex):
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

    # ── Consolidated trend features (replace 3 correlated EMA/SMA features) ──
    # ema_spread > 0 means EMA-20 above EMA-50 (uptrend); < 0 means downtrend
    df["ema_spread"]  = df["ema20_pct"] - df["ema50_pct"]

    # Bollinger Band position: 0 = at lower band, 1 = at upper band, 0.5 = midpoint
    bb_range = (df["bb_high"] - df["bb_low"]).replace(0, np.nan)
    df["bb_position"] = (close - df["bb_low"]) / (bb_range + eps)

    # Volume momentum: short-term volume trend vs longer-term baseline
    df["vol_ratio_trend"] = (
        df["volume_ratio"].rolling(5).mean()
        / (df["volume_ratio"].rolling(20).mean() + eps)
    )

    # ── Multi-period momentum (academically validated factors) ─────────────────
    # Jegadeesh-Titman (1993): intermediate momentum strongly predicts next-month returns
    df["ret_5d"]   = close.pct_change(5)    # 1-week momentum
    df["ret_21d"]  = close.pct_change(21)   # 1-month momentum
    df["ret_63d"]  = close.pct_change(63)   # 3-month momentum
    df["ret_126d"] = close.pct_change(126)  # 6-month momentum
    # AQR 12-1 month momentum: skip the most recent month to remove the 1-month reversal
    df["mom_12_1"] = close.pct_change(252) - close.pct_change(21)

    # Distance from 52-week high (George-Hwang 2004): stocks near their high
    # exhibit continuation; far-from-high stocks show anchoring/reversal
    df["high_52w_pct"] = close / (close.rolling(252).max() + eps) - 1

    df.dropna(inplace=True)
    return df


# NOTE: FEATURE_COLS changed — saved XGBoost and LSTM models must be retrained
# before the bot trades again. Trigger: scripts/train_model.py or the train_models
# GitHub Actions workflow.
#
# Changes from v1 (22 features) → v2 (19 features):
#   Removed (r > 0.85 with kept features): stoch_k, stoch_d, log_returns, rsi_15m,
#     macd_pct, macd_sig_pct, ema20_pct, ema50_pct, sma20_pct, bb_high_pct, bb_low_pct
#   Added (evidence-based, independent): ema_spread, bb_position, vol_ratio_trend,
#     ret_5d, ret_21d, ret_63d, ret_126d, mom_12_1, high_52w_pct
FEATURE_COLS = [
    # Momentum oscillators (independent: RSI=price range, MFI=volume-weighted)
    "rsi",
    "mfi",
    # Volume signals
    "volume_ratio",       # current vs 20-bar average
    "obv_chg_pct",        # OBV directional flow
    "vol_ratio_trend",    # is volume accelerating or fading?
    # Volatility & band structure
    "bb_width",           # Bollinger Band width (regime volatility)
    "atr_pct",            # ATR normalized to price
    "bb_position",        # price location within Bollinger Bands (0–1)
    # Price action
    "returns",            # 1-bar return
    "hl_ratio",           # intraday high-low range / close
    "vwap_dev",           # price vs 20-bar volume-weighted avg price
    # Trend
    "macd_diff_pct",      # MACD histogram (crossover signal, normalized)
    "ema_spread",         # EMA-20 minus EMA-50 (trend direction & strength)
    # Multi-period momentum (Jegadeesh-Titman / AQR factors)
    "ret_5d",             # 1-week return
    "ret_21d",            # 1-month return
    "ret_63d",            # 3-month return
    "ret_126d",           # 6-month return
    "mom_12_1",           # 12-1 month momentum (AQR style, skips reversal month)
    "high_52w_pct",       # distance from 52-week high (George-Hwang 2004)
]
