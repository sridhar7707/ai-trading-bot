"""Signal and screener data functions extracted from dashboard_data.py."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

_MUTED  = "#6b7280"
_POS    = "#15803d"
_NEG    = "#dc2626"

_ACTION_BADGE: dict[str, str] = {
    "STRONG_BUY":  _POS,
    "BUY":         "#15803d",
    "HOLD":        _MUTED,
    "SELL":        _NEG,
    "STRONG_SELL": "#7f1d1d",
}

_SIGNAL_COLS = ["Symbol", "As Of", "Signal", "Score", "XGB", "LSTM", "Sentiment", "Macro", "Regime"]
_SCREENER_COLS = ["Rank", "Symbol", "Sector", "Score", "Analyst", "ETF Mom", "Regime", "Screened At"]


def _con():
    import bot.monitor.dashboard_data as _dd
    db = _dd._DB
    if not Path(db).exists():
        return None
    return sqlite3.connect(db, check_same_thread=False)


def _empty_signals_df() -> pd.DataFrame:
    """Typed empty DataFrame so Gradio renders column headers instead of 'undefined'."""
    return pd.DataFrame(columns=_SIGNAL_COLS)


def get_latest_signals_df() -> pd.DataFrame:
    """Return the most recent signal log row per symbol (latest cycle).

    Pulls from signal_log table written every bot cycle for all evaluated symbols,
    so this stays current even when no trade fires.
    """
    con = _con()
    if con is None:
        return _empty_signals_df()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                xgb_prob REAL, lstm_prob REAL, sentiment_score REAL,
                macro_score REAL, ensemble_score REAL,
                ensemble_action TEXT, regime TEXT
            )
        """)
        con.commit()
    except Exception:
        pass
    try:
        df = pd.read_sql_query(
            """
            SELECT s.symbol, s.timestamp, s.ensemble_action, s.ensemble_score,
                   s.xgb_prob, s.lstm_prob, s.sentiment_score, s.macro_score, s.regime
            FROM signal_log s
            INNER JOIN (
                SELECT symbol, MAX(timestamp) AS ts FROM signal_log GROUP BY symbol
            ) latest ON s.symbol = latest.symbol AND s.timestamp = latest.ts
            ORDER BY s.ensemble_score DESC
            """,
            con,
        )
    except Exception:
        con.close()
        return _empty_signals_df()
    con.close()
    if df.empty:
        return _empty_signals_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m-%d %H:%M")
    df = df.rename(columns={
        "symbol":          "Symbol",
        "timestamp":       "As Of",
        "ensemble_action": "Signal",
        "ensemble_score":  "Score",
        "xgb_prob":        "XGB",
        "lstm_prob":       "LSTM",
        "sentiment_score": "Sentiment",
        "macro_score":     "Macro",
        "regime":          "Regime",
    })
    for col in ("Score", "XGB", "LSTM", "Sentiment", "Macro"):
        df[col] = df[col].round(3)
    return df


def _empty_screener_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_SCREENER_COLS)


def get_screener_df() -> pd.DataFrame:
    """Return today's screener picks with factor scores, sorted by rank."""
    con = _con()
    if con is None:
        return _empty_screener_df()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS screener_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screened_at TEXT NOT NULL, symbol TEXT NOT NULL,
                rank INTEGER, composite_score REAL, analyst_signal REAL,
                etf_momentum REAL, regime TEXT, sector TEXT
            )
        """)
        con.commit()
    except Exception:
        pass
    try:
        df = pd.read_sql_query("""
            SELECT s.rank, s.symbol, s.sector, s.composite_score,
                   s.analyst_signal, s.etf_momentum, s.regime, s.screened_at
            FROM screener_log s
            INNER JOIN (
                SELECT MAX(screened_at) AS latest FROM screener_log
            ) r ON s.screened_at = r.latest
            ORDER BY s.rank
        """, con)
    except Exception:
        con.close()
        return _empty_screener_df()
    con.close()
    if df.empty:
        return _empty_screener_df()
    df["screened_at"] = pd.to_datetime(df["screened_at"]).dt.strftime("%m-%d %H:%M")

    def _fmt_analyst(v):
        if v is None or v != v:
            return "—"
        if v > 0.1:
            return f"+{v:.2f} ▲"
        if v < -0.1:
            return f"{v:.2f} ▼"
        return f"{v:.2f}"

    df["analyst_signal"] = df["analyst_signal"].apply(_fmt_analyst)
    df["etf_momentum"] = df["etf_momentum"].apply(
        lambda v: "↑ Above SMA" if v is not None and v >= 0.5 else ("↓ Below SMA" if v is not None else "—")
    )
    df["composite_score"] = df["composite_score"].round(3)
    df = df.rename(columns={
        "rank":            "Rank",
        "symbol":          "Symbol",
        "sector":          "Sector",
        "composite_score": "Score",
        "analyst_signal":  "Analyst",
        "etf_momentum":    "ETF Mom",
        "regime":          "Regime",
        "screened_at":     "Screened At",
    })
    return df
