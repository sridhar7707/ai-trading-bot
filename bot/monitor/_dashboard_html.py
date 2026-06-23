"""HTML rendering functions (trade log, compliance gauges) extracted from dashboard_data.py."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_BG     = "#ffffff"
_TEXT   = "#111827"
_MUTED  = "#6b7280"
_GRID   = "#e5e7eb"
_POS    = "#15803d"
_NEG    = "#dc2626"
_FONT   = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
_EMPTY_HINT = ("The bot trades at market open (9:30 AM ET, Mon–Fri). "
               "Data appears here after the first cycle of the day.")

_FEATURE_LABELS: dict[str, str] = {
    "rsi":           "RSI",        "rsi_15m":       "RSI(15m)",
    "stoch_k":       "Stoch %K",   "stoch_d":       "Stoch %D",
    "macd_diff_pct": "MACD cross", "macd_pct":      "MACD",
    "macd_sig_pct":  "MACD sig",   "volume_ratio":  "Volume",
    "obv":           "OBV",        "mfi":           "Money Flow",
    "bb_width":      "BB width",   "bb_high_pct":   "BB high",
    "bb_low_pct":    "BB low",     "atr_pct":       "Volatility",
    "norm_close":    "Price pos",  "ema20_pct":     "EMA20",
    "ema50_pct":     "EMA50",      "sma20_pct":     "SMA20",
    "returns":       "Returns",    "vwap_dev":      "VWAP dev",
    "hl_ratio":      "H/L range",
}

_ACTION_COLOR = {
    "BUY":                "#388e3c",
    "SELL":               "#1565c0",
    "SELL_TAKE_PROFIT":   "#00838f",
    "SELL_TRAILING_STOP": "#e65100",
    "SELL_TIME_EXIT":     "#6a1b9a",
    "SELL_STOP":          "#b71c1c",
    "SELL_GAP_DOWN":      "#7f0000",
    "SELL_TRIM":          "#0277bd",
    "SELL_RECONCILE":     "#546e7a",
}

_SELL_REASON = {
    "SELL":               "Signal exit",
    "SELL_TAKE_PROFIT":   "Took profit",
    "SELL_TRAILING_STOP": "Trailing stop",
    "SELL_TIME_EXIT":     "Max hold reached",
    "SELL_STOP":          "Stop-loss hit",
    "SELL_GAP_DOWN":      "Gap-down protection",
    "SELL_TRIM":          "Position trimmed",
    "SELL_RECONCILE":     "Account reset",
}


def _con():
    import bot.monitor.dashboard_data as _dd
    db = _dd._DB
    if not Path(db).exists():
        return None
    return sqlite3.connect(db, check_same_thread=False)


def compliance_gauges_html(c: dict) -> str:
    if not c:
        return f"<p style='color:{_MUTED};font-family:{_FONT}'>No compliance data yet. {_EMPTY_HINT}</p>"

    def _bar(label: str, value_str: str, limit_str: str, pct: float) -> str:
        pct     = min(pct * 100, 100)
        color   = _NEG if pct >= 100 else "#d97706" if pct >= 50 else _POS
        return (
            f"<div style='margin:14px 0'>"
            f"<div style='display:flex;justify-content:space-between;color:{_MUTED};font-size:13px;margin-bottom:4px'>"
            f"<span>{label}</span><span>{value_str} &nbsp;/&nbsp; limit {limit_str}</span></div>"
            f"<div style='background:{_GRID};border-radius:6px;height:18px'>"
            f"<div style='background:{color};width:{pct:.1f}%;height:100%;border-radius:6px;"
            f"transition:width .3s;display:flex;align-items:center;padding-left:6px;"
            f"font-size:11px;color:#fff;font-weight:bold'>"
            f"{'&nbsp;' + f'{pct:.0f}%' if pct > 10 else ''}"
            f"</div></div></div>"
        )

    daily_pct  = max(0.0, -c["day_pnl_pct"])  / (c["daily_limit_pct"]  or 1)
    weekly_pct = max(0.0, -c["week_pnl_pct"]) / (c["weekly_limit_pct"] or 1)
    pdt_pct    = c["day_trades_used"]          / (c["day_trades_limit"]  or 1)

    flags = ""
    if c["daily_warning_sent"]:
        flags += f"<span style='background:#d97706;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;margin-right:6px'>⚠ Daily warning sent</span>"
    if c["weekly_halt_alerted"]:
        flags += f"<span style='background:{_NEG};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px'>🔴 Weekly halt alerted</span>"

    return (
        f"<div style='background:{_BG};border:1px solid {_GRID};padding:18px;border-radius:10px;font-family:{_FONT}'>"
        f"<h3 style='color:{_TEXT};margin-top:0'>Risk Limit Gauges</h3>"
        + _bar("Daily Loss",   f"{c['day_pnl_pct']:+.2%}",  f"-{c['daily_limit_pct']:.0%}",  daily_pct)
        + _bar("Weekly Loss",  f"{c['week_pnl_pct']:+.2%}", f"-{c['weekly_limit_pct']:.0%}", weekly_pct)
        + _bar("Day Trades",   f"{c['day_trades_used']}/{c['day_trades_limit']}", "3 max / rolling 5 business days", pdt_pct)
        + (f"<div style='margin-top:12px'>{flags}</div>" if flags else "")
        + "</div>"
    )


def _trade_rationale(row) -> str:
    """One-line 'why' for a trade.

    Exit rows: plain-English reason (already stored as the action type).
    Entry rows: subscriber-friendly signal summary — avoids raw model names
    (XGB/LSTM) that mean nothing to a non-technical user.
    """
    action = str(row["action"])
    if action.startswith("SELL"):
        return _SELL_REASON.get(action, "Exit")
    xgb    = float(row.get("xgb_prob")        or 0)
    lstm   = float(row.get("lstm_prob")       or 0)
    sent   = float(row.get("sentiment_score") or 0)
    regime = str(row.get("regime") or "").strip()
    avg_model = (xgb + lstm) / 2 if (xgb or lstm) else 0
    if avg_model >= 0.75:
        strength = "Strong buy signal"
    elif avg_model >= 0.55:
        strength = "Buy signal"
    else:
        strength = "AI signal"
    if sent > 0.15:
        strength += " · positive news"
    elif sent < -0.15:
        strength += " · negative news"
    if regime and regime not in ("", "Unknown"):
        strength += f" · {regime.replace('_', ' ').title()}"
    drivers_json = row.get("feature_drivers")
    if drivers_json:
        try:
            import json as _json
            drivers = _json.loads(drivers_json)
            labels = []
            for name, shap_val in drivers[:2]:
                label = _FEATURE_LABELS.get(name, name)
                labels.append(f"{label}{'↑' if shap_val > 0 else '↓'}")
            if labels:
                strength += " · " + ", ".join(labels)
        except Exception:
            pass
    return strength


def trades_html_table(days: int = 30) -> str:
    con = _con()
    if con is None:
        return f"<p style='color:{_MUTED};font-family:{_FONT}'>No trades yet. {_EMPTY_HINT}</p>"
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, symbol, action, shares, price, notional, pnl_pct, "
        "realized_pnl, xgb_prob, lstm_prob, sentiment_score, regime, feature_drivers "
        "FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 200",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        return (f"<p style='color:{_MUTED};font-family:{_FONT}'>No trades in the selected window. "
                f"Try a longer range, or check back after the next market session.</p>")

    _ACTION_DISPLAY = {
        "BUY":                "BUY",
        "SELL":               "SELL",
        "SELL_TAKE_PROFIT":   "SELL",
        "SELL_TRAILING_STOP": "SELL",
        "SELL_TIME_EXIT":     "SELL",
        "SELL_STOP":          "SELL",
        "SELL_GAP_DOWN":      "SELL",
    }

    rows_html = ""
    for _, row in df.iterrows():
        action  = str(row["action"])
        color   = _ACTION_COLOR.get(action, _MUTED)
        display_action = _ACTION_DISPLAY.get(action, action)
        glyph   = "▲" if action == "BUY" else "▼"
        ts      = pd.to_datetime(row["timestamp"]).strftime("%m-%d %H:%M ET")
        rlz_pct = row.get("pnl_pct")
        rlz_usd = row.get("realized_pnl")
        if action.startswith("SELL") and (rlz_pct or rlz_usd):
            pnl_usd_str = f"${rlz_usd:+,.2f}" if rlz_usd else ""
            pnl_pct_str = f"{rlz_pct:+.2%}"   if rlz_pct else ""
            pnl_str = f"{pnl_usd_str} ({pnl_pct_str})" if pnl_usd_str and pnl_pct_str else pnl_usd_str or pnl_pct_str
            pnl_col = _POS if (rlz_usd or rlz_pct or 0) > 0 else _NEG
        else:
            pnl_str = "–"
            pnl_col = _MUTED
        notional_str = f"${row['notional']:,.2f}" if row["notional"] else "–"
        why = _trade_rationale(row)
        rows_html += (
            f"<tr style='border-bottom:1px solid {_GRID}'>"
            f"<td style='color:{_MUTED};padding:6px'>{ts}</td>"
            f"<td style='color:{_TEXT};font-weight:bold;padding:6px'>{row['symbol']}</td>"
            f"<td style='padding:6px'><span style='background:{color};color:#fff;padding:2px 7px;border-radius:4px;"
            f"font-size:11px;white-space:nowrap'>{glyph} {display_action}</span></td>"
            f"<td style='color:{_TEXT};padding:6px'>{row['shares']:.3f}</td>"
            f"<td style='color:{_TEXT};padding:6px'>${row['price']:.2f}</td>"
            f"<td style='color:{_TEXT};padding:6px'>{notional_str}</td>"
            f"<td style='color:{pnl_col};font-weight:bold;padding:6px'>{pnl_str}</td>"
            f"<td style='color:{_MUTED};padding:6px;font-size:12px'>{why}</td>"
            f"</tr>"
        )

    return (
        "<div class='cf-table' style='overflow-x:auto'>"
        f"<table style='width:100%;border-collapse:collapse;font-family:{_FONT};font-size:13px'>"
        f"<thead><tr style='color:{_MUTED};border-bottom:2px solid {_GRID}'>"
        "<th style='text-align:left;padding:6px'>Time</th>"
        "<th style='text-align:left;padding:6px'>Symbol</th>"
        "<th style='text-align:left;padding:6px'>Action</th>"
        "<th style='text-align:left;padding:6px'>Shares</th>"
        "<th style='text-align:left;padding:6px'>Price</th>"
        "<th style='text-align:left;padding:6px'>Amount $</th>"
        "<th style='text-align:left;padding:6px'>Realized P&amp;L</th>"
        "<th style='text-align:left;padding:6px'>Why</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )
