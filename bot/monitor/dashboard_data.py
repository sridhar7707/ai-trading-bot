"""Read-only data layer for the trading dashboard."""
from __future__ import annotations
import json
import sqlite3
import sys
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TRADE_DB_PATH, DAILY_LOSS_LIMIT_PCT, WEEKLY_LOSS_LIMIT_PCT

_DB   = TRADE_DB_PATH
_DARK = "#0f0f0f"
_CARD = "#1a1a2e"


def _con() -> sqlite3.Connection | None:
    if not Path(_DB).exists():
        return None
    return sqlite3.connect(_DB, check_same_thread=False)


def _ax_style(ax):
    ax.set_facecolor(_CARD)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(axis="y", color="#333", linestyle="--", alpha=0.4)


# ── Overview (Subscriber+) ────────────────────────────────────────────────────

def get_overview() -> dict:
    con = _con()
    if con is None:
        return {"_error": "trades.db not found — no trading data yet."}
    today = date.today().isoformat()
    wk    = date.today().strftime("%G-W%V")
    try:
        rs = {r[0]: r[1] for r in con.execute("SELECT key, value FROM risk_state")}
    except Exception:
        rs = {}
    try:
        mc = {r[0]: float(r[1]) for r in con.execute("SELECT key, value FROM macro_cache")}
    except Exception:
        mc = {}

    try:
        row = con.execute(
            "SELECT portfolio_value FROM trades ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        portfolio = float(row[0]) if row else 0.0
    except Exception:
        portfolio = 0.0

    daily_start  = float(rs.get("daily_start_value",  0) or 0)
    weekly_start = float(rs.get("weekly_start_value", 0) or 0)
    is_today     = rs.get("daily_start_date")  == today
    is_this_week = rs.get("weekly_start_week") == wk

    day_pnl  = (portfolio - daily_start)  / daily_start  if is_today  and daily_start  else 0.0
    week_pnl = (portfolio - weekly_start) / weekly_start if is_this_week and weekly_start else 0.0

    try:
        day_trade_dates = json.loads(rs.get("day_trade_dates", "[]"))
    except Exception:
        day_trade_dates = []

    try:
        trades_today = con.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (today + "%",)
        ).fetchone()[0]
    except Exception:
        trades_today = 0
    try:
        open_positions = con.execute("SELECT COUNT(*) FROM position_state").fetchone()[0]
    except Exception:
        open_positions = 0
    con.close()

    return {
        "portfolio":        portfolio,
        "day_pnl":          day_pnl,
        "week_pnl":         week_pnl,
        "trades_today":     trades_today,
        "open_positions":   open_positions,
        "day_trades_used":  day_trade_dates.count(today),
        "macro_score":      mc.get("score", 0.5),
        "macro_halt":       bool(mc.get("halt", 0)),
        "emergency_halt":   Path("data/HALT_TRADING").exists(),
        "daily_limit_hit":  is_today  and daily_start  and day_pnl  <= -DAILY_LOSS_LIMIT_PCT,
        "weekly_limit_hit": is_this_week and weekly_start and week_pnl <= -WEEKLY_LOSS_LIMIT_PCT,
    }


def overview_md(d: dict) -> str:
    if "_error" in d:
        return f"⚠️ {d['_error']}"

    status = "🔴 EMERGENCY HALT" if d["emergency_halt"] else (
             "🔴 VIX HALT"        if d["macro_halt"]      else (
             "🔴 DAILY LIMIT HIT" if d["daily_limit_hit"] else (
             "🟡 WEEKLY LIMIT"    if d["weekly_limit_hit"] else "🟢 ACTIVE")))

    return (
        f"### Bot Status: {status}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Portfolio Value | **${d['portfolio']:,.2f}** |\n"
        f"| Day P&L | **{d['day_pnl']:+.2%}** |\n"
        f"| Week P&L | **{d['week_pnl']:+.2%}** |\n"
        f"| Trades Today | {d['trades_today']} |\n"
        f"| Open Positions | {d['open_positions']} |\n"
        f"| Day Trades Used | {d['day_trades_used']}/3 (PDT) |\n"
        f"| Macro Score | {d['macro_score']:.2f} |\n"
    )


# ── Positions (Subscriber+) ───────────────────────────────────────────────────

def get_positions_df() -> pd.DataFrame:
    _empty = pd.DataFrame(columns=["Symbol", "Entry $", "HWM $", "ATR", "Opened At", "Days"])
    con = _con()
    if con is None:
        return _empty
    try:
        df = pd.read_sql_query(
            "SELECT symbol, entry_price, high_water_mark, atr_at_entry, opened_at FROM position_state", con
        )
    except Exception:
        con.close()
        return _empty
    con.close()
    if df.empty:
        return df.rename(columns={"symbol":"Symbol","entry_price":"Entry $",
                                   "high_water_mark":"HWM $","atr_at_entry":"ATR","opened_at":"Opened At"})
    now = datetime.now(timezone.utc)
    df["opened_at"] = pd.to_datetime(df["opened_at"], utc=True, errors="coerce")
    df["days"]      = df["opened_at"].apply(
        lambda t: f"{(now - t).total_seconds() / 86400:.1f}d" if pd.notna(t) else "–"
    )
    df["opened_at"] = df["opened_at"].dt.strftime("%m-%d %H:%M")
    df.columns = ["Symbol", "Entry $", "HWM $", "ATR", "Opened At", "Days"]
    return df


# ── Trade Log (Subscriber+) ───────────────────────────────────────────────────

def get_trades_df(days: int = 30) -> pd.DataFrame:
    con = _con()
    if con is None:
        return pd.DataFrame()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, symbol, action, shares, price, notional, pnl_pct "
        "FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 200",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m-%d %H:%M")
    df["pnl_pct"]   = df["pnl_pct"].apply(lambda x: f"{x:+.2%}" if x else "–")
    df["notional"]  = df["notional"].apply(lambda x: f"${x:,.2f}" if x else "–")
    df["price"]     = df["price"].apply(lambda x: f"${x:.2f}" if x else "–")
    df.columns = ["Time", "Symbol", "Action", "Shares", "Price", "Notional", "P&L"]
    return df


# ── Performance metrics (Pro+) ────────────────────────────────────────────────

def get_performance_metrics(days: int = 60) -> dict:
    con = _con()
    if con is None:
        return {}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = con.execute(
        "SELECT action, pnl_pct, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        (since,),
    ).fetchall()
    con.close()
    if not rows:
        return {"sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "total_return": 0.0}
    vals    = np.array([r[2] for r in rows])
    rets    = np.diff(vals) / (vals[:-1] + 1e-8)
    sharpe  = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252 * 78)) if len(rets) > 1 else 0.0
    peak = vals[0]; max_dd = 0.0
    for v in vals:
        peak  = max(peak, v)
        max_dd = max(max_dd, (peak - v) / (peak + 1e-8))
    sells    = [r for r in rows if r[0].startswith("SELL")]
    win_rate = sum(1 for r in sells if r[1] > 0) / len(sells) if sells else 0.0
    return {
        "sharpe":       round(sharpe, 2),
        "win_rate":     round(win_rate, 4),
        "max_drawdown": round(max_dd, 4),
        "total_return": round((vals[-1] - vals[0]) / (vals[0] + 1e-8), 4),
        "trade_count":  len(rows),
    }


def performance_md(m: dict) -> str:
    if not m:
        return "No data."
    return (
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Sharpe Ratio | **{m['sharpe']:.2f}** |\n"
        f"| Win Rate | **{m['win_rate']:.1%}** |\n"
        f"| Max Drawdown | **{m['max_drawdown']:.1%}** |\n"
        f"| Total Return | **{m['total_return']:+.2%}** |\n"
        f"| Trades Analysed | {m['trade_count']} |"
    )


# ── Charts (Pro+) ─────────────────────────────────────────────────────────────

def portfolio_chart(days: int = 60) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(_DARK); _ax_style(ax)
    con = _con()
    if con is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white"); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        ax.text(0.5, 0.5, "No trades yet", ha="center", va="center", color="white"); return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ax.plot(df["timestamp"], df["portfolio_value"], color="#00d4ff", lw=1.5, label="Portfolio")
    ax.fill_between(df["timestamp"], df["portfolio_value"], df["portfolio_value"].min() * 0.999,
                    alpha=0.12, color="#00d4ff")
    ax.set_title("Portfolio Value", color="white", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate(); fig.tight_layout(); return fig


def signals_chart(days: int = 30) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    fig.patch.set_facecolor(_DARK)
    for ax in axes.flatten(): _ax_style(ax)
    con = _con()
    if con is None:
        axes[0][0].text(0.5, 0.5, "No data", ha="center", va="center", color="white"); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT xgb_prob, lstm_prob, sentiment_score, ensemble_score FROM trades "
        "WHERE action='BUY' AND timestamp >= ?", con, params=(since,),
    )
    con.close()
    pairs = [(axes[0][0], "xgb_prob", "XGB Probability", "#4fc3f7"),
             (axes[0][1], "lstm_prob", "LSTM Probability", "#81c784"),
             (axes[1][0], "sentiment_score", "Sentiment Score", "#ffb74d"),
             (axes[1][1], "ensemble_score", "Ensemble Score", "#ce93d8")]
    for ax, col, title, color in pairs:
        data = df[col].dropna() if not df.empty and col in df.columns else pd.Series(dtype=float)
        if data.empty:
            ax.text(0.5, 0.5, "No BUY data yet", ha="center", va="center", color="grey", fontsize=9)
        else:
            ax.hist(data, bins=20, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(title, color="white", fontsize=10)
    fig.suptitle("Signal Score Distributions  (BUY entries)", color="white", fontsize=12)
    fig.tight_layout(); return fig


def monthly_chart(days: int = 180) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(_DARK); _ax_style(ax)
    con = _con()
    if con is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white"); return fig
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, portfolio_value FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        ax.text(0.5, 0.5, "No trades yet", ha="center", va="center", color="white"); return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"]     = df["timestamp"].dt.to_period("M")
    m = df.groupby("month")["portfolio_value"].agg(["first", "last"])
    m["ret"] = (m["last"] - m["first"]) / (m["first"] + 1e-8) * 100
    colors = ["#4caf50" if r >= 0 else "#ef5350" for r in m["ret"]]
    ax.bar([str(p) for p in m.index], m["ret"], color=colors, edgecolor="none")
    ax.axhline(0, color="#555", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_title("Monthly Returns", color="white", fontsize=13)
    ax.tick_params(axis="x", rotation=25, colors="white")
    fig.tight_layout(); return fig


# ── Audit Trail (Institutional) ───────────────────────────────────────────────

def get_audit_df(days: int = 60) -> pd.DataFrame:
    con = _con()
    if con is None:
        return pd.DataFrame()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, symbol, action, price, notional, pnl_pct, "
        "xgb_prob, lstm_prob, sentiment_score, macro_score, ensemble_score, regime "
        "FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC",
        con, params=(since,),
    )
    con.close()
    return df


# ── Compliance (Institutional) ────────────────────────────────────────────────

def get_compliance_state() -> dict:
    con = _con()
    if con is None:
        return {}
    today = date.today().isoformat()
    wk    = date.today().strftime("%G-W%V")
    try:
        rs = {r[0]: r[1] for r in con.execute("SELECT key, value FROM risk_state")}
    except Exception:
        rs = {}
    try:
        row = con.execute(
            "SELECT portfolio_value FROM trades ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        portfolio = float(row[0]) if row else 0.0
    except Exception:
        portfolio = 0.0
    daily_start  = float(rs.get("daily_start_value",  0) or 0)
    weekly_start = float(rs.get("weekly_start_value", 0) or 0)
    try:
        dtd = json.loads(rs.get("day_trade_dates", "[]"))
    except Exception:
        dtd = []
    con.close()
    return {
        "portfolio":           portfolio,
        "day_pnl_pct":         (portfolio - daily_start)  / daily_start  if daily_start  else 0.0,
        "week_pnl_pct":        (portfolio - weekly_start) / weekly_start if weekly_start else 0.0,
        "daily_limit_pct":     DAILY_LOSS_LIMIT_PCT,
        "weekly_limit_pct":    WEEKLY_LOSS_LIMIT_PCT,
        "day_trades_used":     dtd.count(today),
        "day_trades_limit":    3,
        "daily_warning_sent":  rs.get("daily_warning_sent_date") == today,
        "weekly_halt_alerted": rs.get("weekly_halt_alerted_week") == wk,
    }


def compliance_md(c: dict) -> str:
    if not c:
        return "No data."
    daily_used  = abs(c["day_pnl_pct"])  / c["daily_limit_pct"]  * 100
    weekly_used = abs(c["week_pnl_pct"]) / c["weekly_limit_pct"] * 100
    pdt_used    = c["day_trades_used"]   / c["day_trades_limit"]  * 100
    return (
        f"### Risk Limits\n\n"
        f"| Limit | Used | Threshold | Status |\n|-------|------|-----------|--------|\n"
        f"| Daily Loss | {c['day_pnl_pct']:+.2%} | -{c['daily_limit_pct']:.0%} | "
        f"{'🔴 HIT' if daily_used >= 100 else f'🟡 {daily_used:.0f}%' if daily_used >= 50 else f'🟢 {daily_used:.0f}%'} |\n"
        f"| Weekly Loss | {c['week_pnl_pct']:+.2%} | -{c['weekly_limit_pct']:.0%} | "
        f"{'🔴 HIT' if weekly_used >= 100 else f'🟡 {weekly_used:.0f}%' if weekly_used >= 50 else f'🟢 {weekly_used:.0f}%'} |\n"
        f"| PDT Day Trades | {c['day_trades_used']}/{c['day_trades_limit']} | 3/rolling 5d | "
        f"{'🔴 FULL' if pdt_used >= 100 else f'🟡 {pdt_used:.0f}%' if pdt_used >= 66 else f'🟢 {pdt_used:.0f}%'} |\n"
        f"\n### Flags\n\n"
        f"- Daily warning sent: {'✅ Yes' if c['daily_warning_sent'] else '— No'}\n"
        f"- Weekly halt alerted: {'✅ Yes' if c['weekly_halt_alerted'] else '— No'}\n"
    )


# ── Visual risk gauges — HTML progress bars (Institutional) ──────────────────

def compliance_gauges_html(c: dict) -> str:
    if not c:
        return "<p style='color:#888'>No data.</p>"

    def _bar(label: str, value_str: str, limit_str: str, pct: float) -> str:
        pct     = min(pct * 100, 100)
        color   = "#ef5350" if pct >= 100 else "#ff9800" if pct >= 50 else "#4caf50"
        return (
            f"<div style='margin:14px 0'>"
            f"<div style='display:flex;justify-content:space-between;color:#ccc;font-size:13px;margin-bottom:4px'>"
            f"<span>{label}</span><span>{value_str} &nbsp;/&nbsp; limit {limit_str}</span></div>"
            f"<div style='background:#2a2a3e;border-radius:6px;height:18px'>"
            f"<div style='background:{color};width:{pct:.1f}%;height:100%;border-radius:6px;"
            f"transition:width .3s;display:flex;align-items:center;padding-left:6px;"
            f"font-size:11px;color:#000;font-weight:bold'>"
            f"{'&nbsp;' + f'{pct:.0f}%' if pct > 10 else ''}"
            f"</div></div></div>"
        )

    daily_pct  = abs(c["day_pnl_pct"])  / (c["daily_limit_pct"]  or 1)
    weekly_pct = abs(c["week_pnl_pct"]) / (c["weekly_limit_pct"] or 1)
    pdt_pct    = c["day_trades_used"]   / (c["day_trades_limit"]  or 1)

    flags = ""
    if c["daily_warning_sent"]:
        flags += "<span style='background:#ff9800;color:#000;padding:2px 8px;border-radius:4px;font-size:12px;margin-right:6px'>⚠ Daily warning sent</span>"
    if c["weekly_halt_alerted"]:
        flags += "<span style='background:#ef5350;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px'>🔴 Weekly halt alerted</span>"

    return (
        f"<div style='background:#1a1a2e;padding:18px;border-radius:10px;font-family:sans-serif'>"
        f"<h3 style='color:#fff;margin-top:0'>Risk Limit Gauges</h3>"
        + _bar("Daily Loss",   f"{c['day_pnl_pct']:+.2%}",  f"-{c['daily_limit_pct']:.0%}",  daily_pct)
        + _bar("Weekly Loss",  f"{c['week_pnl_pct']:+.2%}", f"-{c['weekly_limit_pct']:.0%}", weekly_pct)
        + _bar("PDT Trades",   f"{c['day_trades_used']}/{c['day_trades_limit']}", "3 / 5-day window", pdt_pct)
        + (f"<div style='margin-top:12px'>{flags}</div>" if flags else "")
        + "</div>"
    )


# ── Color-coded Trade Log HTML (Subscriber+) ──────────────────────────────────

_ACTION_COLOR = {
    "BUY":                "#388e3c",   # green
    "SELL":               "#1565c0",   # blue — planned signal exit
    "SELL_TAKE_PROFIT":   "#00838f",   # teal
    "SELL_TRAILING_STOP": "#e65100",   # deep orange
    "SELL_TIME_EXIT":     "#6a1b9a",   # purple
    "SELL_STOP":          "#b71c1c",   # red
    "SELL_GAP_DOWN":      "#7f0000",   # dark red
}


def trades_html_table(days: int = 30) -> str:
    con = _con()
    if con is None:
        return "<p style='color:#888'>No trades data.</p>"
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, symbol, action, shares, price, notional, pnl_pct "
        "FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 200",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        return "<p style='color:#888'>No trades in the selected window.</p>"

    rows_html = ""
    for _, row in df.iterrows():
        action  = str(row["action"])
        color   = _ACTION_COLOR.get(action, "#555")
        ts      = pd.to_datetime(row["timestamp"]).strftime("%m-%d %H:%M")
        pnl_str = f"{row['pnl_pct']:+.2%}" if row["pnl_pct"] else "–"
        pnl_col = "#4caf50" if row["pnl_pct"] and row["pnl_pct"] > 0 else ("#ef5350" if row["pnl_pct"] and row["pnl_pct"] < 0 else "#888")
        notional_str = f"${row['notional']:,.2f}" if row["notional"] else "–"
        rows_html += (
            f"<tr style='border-bottom:1px solid #2a2a3e'>"
            f"<td style='color:#aaa'>{ts}</td>"
            f"<td style='color:#fff;font-weight:bold'>{row['symbol']}</td>"
            f"<td><span style='background:{color};color:#fff;padding:2px 7px;border-radius:4px;"
            f"font-size:11px;white-space:nowrap'>{action}</span></td>"
            f"<td style='color:#ccc'>{row['shares']:.4f}</td>"
            f"<td style='color:#ccc'>${row['price']:.2f}</td>"
            f"<td style='color:#ccc'>{notional_str}</td>"
            f"<td style='color:{pnl_col};font-weight:bold'>{pnl_str}</td>"
            f"</tr>"
        )

    return (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:13px'>"
        "<thead><tr style='color:#888;border-bottom:2px solid #333'>"
        "<th style='text-align:left;padding:6px'>Time</th>"
        "<th style='text-align:left;padding:6px'>Symbol</th>"
        "<th style='text-align:left;padding:6px'>Action</th>"
        "<th style='text-align:left;padding:6px'>Shares</th>"
        "<th style='text-align:left;padding:6px'>Price</th>"
        "<th style='text-align:left;padding:6px'>Notional</th>"
        "<th style='text-align:left;padding:6px'>P&amp;L</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )


# ── Emergency halt toggle ─────────────────────────────────────────────────────

_HALT_FILE = Path("data/HALT_TRADING")


def halt_status_html() -> str:
    active = _HALT_FILE.exists()
    color  = "#b71c1c" if active else "#1b5e20"
    text   = "🔴 HALT ACTIVE — bot is paused" if active else "🟢 BOT RUNNING — no halt file"
    btn_label = "▶ Remove Halt &amp; Resume" if active else "⏹ Activate Emergency Halt"
    return (
        f"<div style='background:{color};padding:12px 16px;border-radius:8px;"
        f"color:#fff;font-size:14px;font-weight:bold'>{text}</div>"
    ), btn_label


def toggle_halt() -> tuple[str, str]:
    _HALT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _HALT_FILE.exists():
        _HALT_FILE.unlink()
        msg = "✅ Halt removed — bot will resume on next cycle."
    else:
        _HALT_FILE.touch()
        msg = "🔴 HALT FILE CREATED — bot will skip trading until removed."
    status_html, btn_label = halt_status_html()
    return status_html, btn_label, msg
