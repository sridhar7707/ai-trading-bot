"""Data loading utilities for the evaluation framework.

All public functions return DataFrames with consistent column names
that metrics.py and ablation.py expect.
"""
from __future__ import annotations
import sqlite3

import pandas as pd

from config import TRADE_DB_PATH


def _con(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


def load_completed_trades(
    db_path: str = TRADE_DB_PATH,
    days: int = 365,
    con: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    """Load BUY→SELL round-trip pairs with component scores captured at entry.

    The Nth SELL for a symbol is matched to the Nth BUY (in timestamp order).
    This is the only reliable link without a foreign-key trade_id column.

    Returned columns:
      symbol, buy_ts, sell_ts, entry_price, exit_price,
      pnl_pct, holding_days, realized_pnl, regime,
      xgb_prob, lstm_prob, sentiment_score, macro_score, ensemble_score,
      stop_loss, take_profit, risk_reward_ratio, notional, portfolio_value
    """
    cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).isoformat()
    _own_con = con is None
    if _own_con:
        con = _con(db_path)
    try:
        buys = pd.read_sql_query(
            """
            SELECT symbol,
                   timestamp        AS buy_ts,
                   price            AS entry_price,
                   notional,
                   portfolio_value,
                   xgb_prob,
                   lstm_prob,
                   sentiment_score,
                   macro_score,
                   ensemble_score,
                   regime,
                   stop_loss,
                   take_profit,
                   risk_reward_ratio
            FROM trades
            WHERE action = 'BUY' AND timestamp >= ?
            ORDER BY symbol, timestamp
            """,
            con, params=[cutoff],
        )
        sells = pd.read_sql_query(
            """
            SELECT symbol,
                   timestamp  AS sell_ts,
                   price      AS exit_price,
                   pnl_pct,
                   holding_days,
                   realized_pnl
            FROM trades
            WHERE action LIKE 'SELL%' AND timestamp >= ?
            ORDER BY symbol, timestamp
            """,
            con, params=[cutoff],
        )
    finally:
        if _own_con:
            con.close()

    if buys.empty or sells.empty:
        return pd.DataFrame()

    buys["_rank"]  = buys.groupby("symbol").cumcount()
    sells["_rank"] = sells.groupby("symbol").cumcount()

    merged = pd.merge(buys, sells, on=["symbol", "_rank"], how="inner").drop(columns=["_rank"])
    merged["buy_ts"]       = pd.to_datetime(merged["buy_ts"],  utc=True, errors="coerce")
    merged["sell_ts"]      = pd.to_datetime(merged["sell_ts"], utc=True, errors="coerce")
    merged["pnl_pct"]      = pd.to_numeric(merged["pnl_pct"],      errors="coerce").fillna(0.0)
    merged["holding_days"] = pd.to_numeric(merged["holding_days"], errors="coerce").fillna(0.0)
    merged["realized_pnl"] = pd.to_numeric(merged["realized_pnl"], errors="coerce").fillna(0.0)
    return merged.reset_index(drop=True)


def load_equity_curve(
    db_path: str = TRADE_DB_PATH,
    days: int = 365,
    con: sqlite3.Connection | None = None,
) -> pd.Series:
    """Daily equity curve from portfolio_snapshots.

    Returns Series[datetime → portfolio_value], last snapshot per day.
    Empty Series when no data exists.
    """
    cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).isoformat()
    _own_con = con is None
    if _own_con:
        con = _con(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, portfolio_value FROM portfolio_snapshots
            WHERE timestamp >= ? AND portfolio_value > 0
            ORDER BY timestamp ASC
            """,
            con, params=[cutoff],
        )
    finally:
        if _own_con:
            con.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    return df["portfolio_value"].resample("1D").last().dropna()


def load_signal_log(
    db_path: str = TRADE_DB_PATH,
    days: int = 90,
    con: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    """All per-cycle signal evaluations, including cycles where no trade fired.

    Columns: timestamp, symbol, xgb_prob, lstm_prob, sentiment_score,
             macro_score, ensemble_score, ensemble_action, regime
    """
    cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).isoformat()
    _own_con = con is None
    if _own_con:
        con = _con(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, symbol, xgb_prob, lstm_prob, sentiment_score,
                   macro_score, ensemble_score, ensemble_action, regime
            FROM signal_log
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """,
            con, params=[cutoff],
        )
    finally:
        if _own_con:
            con.close()

    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.reset_index(drop=True)


def fetch_spy_daily(days: int = 365) -> pd.Series:
    """Fetch SPY daily returns from yfinance.

    Returns Series[datetime.date → daily_pct_return], empty on failure.
    """
    try:
        import yfinance as yf
        spy = yf.download("SPY", period=f"{days}d", progress=False, auto_adjust=True)
        if spy.empty:
            return pd.Series(dtype=float)
        closes = spy["Close"].squeeze()
        rets = closes.pct_change().dropna()
        rets.index = pd.to_datetime(rets.index).date
        return rets
    except Exception:
        return pd.Series(dtype=float)
