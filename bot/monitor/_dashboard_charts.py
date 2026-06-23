"""Matplotlib chart functions extracted from dashboard_data.py."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import pandas as pd

_BG     = "#ffffff"
_CARD   = "#f7f8fa"
_TEXT   = "#111827"
_MUTED  = "#6b7280"
_GRID   = "#e5e7eb"
_ACCENT = "#0891b2"
_POS    = "#15803d"
_NEG    = "#dc2626"
_EMPTY_HINT = ("The bot trades at market open (9:30 AM ET, Mon–Fri). "
               "Data appears here after the first cycle of the day.")


def _con():
    import bot.monitor.dashboard_data as _dd
    db = _dd._DB
    if not Path(db).exists():
        return None
    return sqlite3.connect(db, check_same_thread=False)


def _ax_style(ax) -> None:
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(axis="y", color=_GRID, linestyle="--", alpha=0.8)


def portfolio_chart(days: int = 60) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(_BG); _ax_style(ax)
    con = _con()
    if con is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=_MUTED); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? "
            "UNION ALL "
            "SELECT timestamp, portfolio_value FROM portfolio_snapshots WHERE timestamp >= ? "
            "ORDER BY timestamp",
            con, params=(since, since),
        )
    except Exception:
        df = pd.read_sql_query(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
            con, params=(since,),
        )
    con.close()
    if df.empty:
        ax.text(0.5, 0.5, f"No data yet\n{_EMPTY_HINT}", ha="center", va="center",
                color=_MUTED, fontsize=9, wrap=True); return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    ax.plot(df["timestamp"], df["portfolio_value"], color=_ACCENT, lw=1.8, label="Portfolio")
    ax.fill_between(df["timestamp"], df["portfolio_value"], df["portfolio_value"].min() * 0.999,
                    alpha=0.12, color=_ACCENT)
    ax.set_title("Portfolio Value", color=_TEXT, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate(); fig.tight_layout(); return fig


def signals_chart(days: int = 30) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    fig.patch.set_facecolor(_BG)
    for ax in axes.flatten(): _ax_style(ax)
    con = _con()
    if con is None:
        axes[0][0].text(0.5, 0.5, "No data", ha="center", va="center", color=_MUTED); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT xgb_prob, lstm_prob, sentiment_score, ensemble_score FROM trades "
        "WHERE action='BUY' AND timestamp >= ?", con, params=(since,),
    )
    con.close()
    pairs = [(axes[0][0], "xgb_prob", "XGB Probability", "#2563eb"),
             (axes[0][1], "lstm_prob", "LSTM Probability", _POS),
             (axes[1][0], "sentiment_score", "Sentiment Score", "#d97706"),
             (axes[1][1], "ensemble_score", "Ensemble Score", "#7c3aed")]
    for ax, col, title, color in pairs:
        data = df[col].dropna() if not df.empty and col in df.columns else pd.Series(dtype=float)
        if data.empty:
            ax.text(0.5, 0.5, "No BUY data yet", ha="center", va="center", color=_MUTED, fontsize=9)
        else:
            ax.hist(data, bins=20, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(title, color=_TEXT, fontsize=10, fontweight="bold")
    fig.suptitle("Signal Score Distributions  (BUY entries)", color=_TEXT, fontsize=12, fontweight="bold")
    fig.tight_layout(); return fig


def monthly_chart(days: int = 180) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(_BG); _ax_style(ax)
    con = _con()
    if con is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=_MUTED); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? "
            "UNION ALL "
            "SELECT timestamp, portfolio_value FROM portfolio_snapshots WHERE timestamp >= ? "
            "ORDER BY timestamp",
            con, params=(since, since),
        )
    except Exception:
        df = pd.read_sql_query(
            "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
            con, params=(since,),
        )
    con.close()
    if df.empty:
        ax.text(0.5, 0.5, f"No trades yet\n{_EMPTY_HINT}", ha="center", va="center",
                color=_MUTED, fontsize=9, wrap=True); return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["month"]     = df["timestamp"].dt.to_period("M")
    m = df.groupby("month")["portfolio_value"].agg(["first", "last"])
    m["ret"] = (m["last"] - m["first"]) / (m["first"] + 1e-8) * 100
    colors = [_POS if r >= 0 else _NEG for r in m["ret"]]
    ax.bar([str(p) for p in m.index], m["ret"], color=colors, edgecolor="none")
    ax.axhline(0, color=_MUTED, lw=0.8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_title("Monthly Returns", color=_TEXT, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=25, colors=_MUTED)
    fig.tight_layout(); return fig
