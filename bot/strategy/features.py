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
    df["ret_63d"]  = close.pct_change(63)   # 3-month momentum (kept for compatibility)
    df["ret_126d"] = close.pct_change(126)  # 6-month momentum (kept for compatibility)
    # AQR 12-1 month momentum: skip the most recent month to remove the 1-month reversal
    df["mom_12_1"] = close.pct_change(252) - close.pct_change(21)

    # Distance from 52-week high (George-Hwang 2004): stocks near their high
    # exhibit continuation; far-from-high stocks show anchoring/reversal
    df["high_52w_pct"] = close / (close.rolling(252).max() + eps) - 1

    # ── Short-term signals for 1-week prediction horizon (FEATURE_COLS_V3) ──────
    # Overnight gap: open vs prior close. Large gaps show mean-reversion within 1-3 days.
    if "open" in df.columns:
        df["gap_overnight"] = (df["open"] - close.shift(1)) / (close.shift(1) + eps)
    else:
        df["gap_overnight"] = 0.0

    # RSI divergence: today's RSI minus yesterday's — captures momentum acceleration
    df["rsi_divergence"] = df["rsi"].diff(1)

    # MACD crossover boolean: 1 when histogram flips from negative to positive (buy signal)
    df["macd_cross_up"] = (
        (df["macd_diff"] > 0) & (df["macd_diff"].shift(1) <= 0)
    ).astype(float)

    df.dropna(inplace=True)
    return df


FEATURE_COLS_V3 = [
    "rsi",
    "mfi",
    "volume_ratio",
    "obv_chg_pct",
    "vol_ratio_trend",
    "bb_width",
    "atr_pct",
    "bb_position",
    "returns",
    "hl_ratio",
    "vwap_dev",
    "macd_diff_pct",
    "ema_spread",
    "ret_5d",             # 1-week return — still relevant at 1-week target
    "ret_21d",            # 1-month return — useful for mean-reversion context
    "high_52w_pct",       # proximity to yearly high (breakout continuation)
    "gap_overnight",      # overnight gap size — mean-reversion signal (new)
    "rsi_divergence",     # RSI momentum acceleration (new)
    "macd_cross_up",      # MACD histogram flipped positive (new)
]

# Activated 2026-06-30: short-term signals aligned with FORWARD_PERIODS=5 target.
# Drops long-horizon features (ret_63d, ret_126d, mom_12_1) that noise the 1-week signal.
FEATURE_COLS = list(FEATURE_COLS_V3)
