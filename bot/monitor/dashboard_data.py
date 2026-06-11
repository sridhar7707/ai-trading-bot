"""Read-only data layer for the trading dashboard."""
from __future__ import annotations
import json
import sqlite3
import sys
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TRADE_DB_PATH, DAILY_LOSS_LIMIT_PCT, WEEKLY_LOSS_LIMIT_PCT

_DB   = TRADE_DB_PATH
_DARK = "#0f0f0f"
_CARD = "#1a1a2e"

# Tracks last pull outcome for display in overview
_last_sync: dict = {"ok": None, "ts": None, "err": ""}
# Serializes HF pulls so concurrent tab loads don't race the download/copy.
_pull_lock = threading.Lock()


def _con() -> sqlite3.Connection | None:
    # Every read goes through here, so pull the latest DB first. This is the one
    # place that guarantees ALL tabs (not just Overview) see fresh data — without
    # it, tabs that render before Overview's async pull read the empty bundled DB.
    # Cheap after the first call: pull_db respects a 5-min cache, and off-Space
    # (tests/local) refresh_db_from_hf() is a no-op.
    refresh_db_from_hf()
    if not Path(_DB).exists():
        logger.warning(f"dashboard_data: {_DB} not found — no data to show")
        return None
    return sqlite3.connect(_DB, check_same_thread=False)


def _latest_portfolio_value(con) -> float:
    """Most recent portfolio value from either a trade row or a heartbeat snapshot.

    Snapshots are written every cycle (even with no trade), so this stays live
    instead of reading $0.00 until the first fill.
    """
    candidates: list[tuple[str, float]] = []
    try:
        t = con.execute(
            "SELECT timestamp, portfolio_value FROM trades ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if t and t[1] is not None:
            candidates.append((t[0], float(t[1])))
    except Exception:
        pass
    try:
        s = con.execute(
            "SELECT timestamp, portfolio_value FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if s and s[1] is not None:
            candidates.append((s[0], float(s[1])))
    except Exception:
        pass
    return max(candidates, key=lambda r: r[0])[1] if candidates else 0.0


def refresh_db_from_hf(force: bool = False) -> None:
    """Pull a fresh trades.db from HF dataset (Space-only).

    Called on every _con() (cheap — pull_db respects a 5-min cache) and with
    force=True by the explicit Refresh button. The lock serializes concurrent
    pulls so two tab loads can't download/copy the file at the same time.
    pull_db() already logs the meaningful events (download, cache-skip, failure),
    so this only records the outcome for the overview status line.
    """
    global _last_sync
    if not os.environ.get("SPACE_ID"):
        return
    try:
        from bot.monitor.sync_db import pull_db
        with _pull_lock:
            ok = pull_db(force=force)
        _last_sync = {
            "ok": ok,
            "ts": datetime.now(timezone.utc),
            "err": "" if ok else "pull_db returned False — check HF_TOKEN and HF_DB_REPO_ID",
        }
        if not ok:
            logger.error("refresh_db_from_hf: DB pull returned False — dashboard may show stale data")
    except Exception as exc:
        _last_sync = {"ok": False, "ts": datetime.now(timezone.utc), "err": str(exc)}
        logger.error(f"refresh_db_from_hf: exception — {exc}")


def diagnostics() -> dict:
    """Log a full snapshot of the dashboard's environment and DB state.

    Call this once at Space startup. The log output makes the common failure
    modes (missing HF_TOKEN, empty DB, wrong repo, never-synced) obvious at a
    glance instead of silently rendering $0.00.
    """
    import sys as _sys
    from config import HF_DB_REPO_ID, HF_TOKEN

    in_space = bool(os.environ.get("SPACE_ID"))
    token_present = bool(HF_TOKEN or os.environ.get("HF_TOKEN"))
    db_path = Path(_DB)
    db_exists = db_path.exists()

    info: dict = {
        "in_hf_space":   in_space,
        "space_id":      os.environ.get("SPACE_ID", "—"),
        "hf_token_set":  token_present,
        "hf_db_repo":    HF_DB_REPO_ID or "—",
        "db_path":       str(db_path.resolve()),
        "db_exists":     db_exists,
        "python":        _sys.version.split()[0],
    }

    logger.info("─── DASHBOARD DIAGNOSTICS ───────────────────────────────")
    logger.info(f"  Running in HF Space : {in_space}  (SPACE_ID={info['space_id']})")
    logger.info(f"  HF_TOKEN present    : {token_present}")
    logger.info(f"  HF_DB_REPO_ID       : {info['hf_db_repo']}")
    logger.info(f"  DB path             : {info['db_path']}")
    logger.info(f"  DB exists           : {db_exists}")

    if not in_space:
        logger.warning("  ⚠ SPACE_ID not set — refresh_db_from_hf() will SKIP pulling from HF.")
    if in_space and not token_present:
        logger.error("  ⚠ In a Space but HF_TOKEN is MISSING — add it as a Space secret, "
                     "otherwise the private dataset cannot be pulled and the dashboard stays at $0.00.")

    if db_exists:
        try:
            size_kb = db_path.stat().st_size / 1024
            age_s   = (datetime.now(timezone.utc) -
                       datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc)).total_seconds()
            info["db_size_kb"] = round(size_kb, 1)
            info["db_age_s"]   = round(age_s)
            logger.info(f"  DB size             : {size_kb:.1f} KB")
            logger.info(f"  DB file age         : {_fmt_age(age_s)}")
        except Exception as exc:
            logger.warning(f"  Could not stat DB file: {exc}")

        con = _con()
        if con is not None:
            for table in ("trades", "position_state", "risk_state", "macro_cache", "portfolio_snapshots"):
                try:
                    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    info[f"rows_{table}"] = n
                    logger.info(f"  rows[{table:<14}]: {n}")
                except Exception as exc:
                    info[f"rows_{table}"] = f"ERROR: {exc}"
                    logger.warning(f"  rows[{table:<14}]: table missing/unreadable — {exc}")
            try:
                row = con.execute(
                    "SELECT timestamp, symbol, action, portfolio_value "
                    "FROM trades ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row:
                    info["latest_trade"] = row[0]
                    logger.info(f"  latest trade        : {row[0]} {row[1]} {row[2]} "
                                f"portfolio=${row[3]:,.2f}")
                else:
                    logger.warning("  latest trade        : NONE — trades table is empty, "
                                   "dashboard will show $0.00 portfolio")
            except Exception as exc:
                logger.warning(f"  latest trade        : query failed — {exc}")
            con.close()
    else:
        logger.error("  ⚠ trades.db does NOT exist locally — nothing to display. "
                     "Either the bot has never run, or the HF pull failed.")

    logger.info("──────────────────────────────────────────────────────────")
    return info


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

    # Live portfolio value — from latest trade or heartbeat snapshot, whichever newer
    portfolio = _latest_portfolio_value(con)

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

    # DB sync status for display
    sync_age_s: float | None = None
    sync_ok: bool | None = _last_sync.get("ok")
    if _last_sync.get("ts"):
        sync_age_s = (datetime.now(timezone.utc) - _last_sync["ts"]).total_seconds()

    # Also check file mtime as a fallback indicator
    db_mtime_s: float | None = None
    try:
        db_mtime_s = (datetime.now(timezone.utc) - datetime.fromtimestamp(
            Path(_DB).stat().st_mtime, tz=timezone.utc
        )).total_seconds()
    except Exception:
        pass

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
        "sync_ok":          sync_ok,
        "sync_age_s":       sync_age_s,
        "sync_err":         _last_sync.get("err", ""),
        "db_age_s":         db_mtime_s,
    }


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    return f"{int(seconds / 3600)}h ago"


def overview_md(d: dict) -> str:
    if "_error" in d:
        return f"⚠️ {d['_error']}"

    status = "🔴 EMERGENCY HALT" if d["emergency_halt"] else (
             "🔴 VIX HALT"        if d["macro_halt"]      else (
             "🔴 DAILY LIMIT HIT" if d["daily_limit_hit"] else (
             "🟡 WEEKLY LIMIT"    if d["weekly_limit_hit"] else "🟢 ACTIVE")))

    # DB sync status line
    sync_ok  = d.get("sync_ok")
    db_age   = d.get("db_age_s")
    sync_err = d.get("sync_err", "")
    if sync_ok is True:
        sync_line = f"🟢 DB synced {_fmt_age(d.get('sync_age_s'))} · file updated {_fmt_age(db_age)}"
    elif sync_ok is False:
        err_hint = f" — {sync_err}" if sync_err else ""
        sync_line = f"🔴 DB sync FAILED{err_hint} · file last updated {_fmt_age(db_age)}"
    else:
        sync_line = f"⚪ DB not yet synced · file last updated {_fmt_age(db_age)}"

    return (
        f"### Bot Status: {status}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Portfolio Value | **${d['portfolio']:,.2f}** |\n"
        f"| Day P&L | **{d['day_pnl']:+.2%}** |\n"
        f"| Week P&L | **{d['week_pnl']:+.2%}** |\n"
        f"| Trades Today | {d['trades_today']} |\n"
        f"| Open Positions | {d['open_positions']} |\n"
        f"| Day Trades Used | {d['day_trades_used']}/3 (PDT) |\n"
        f"| Macro Score | {d['macro_score']:.2f} |\n\n"
        f"*{sync_line}*\n"
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
    # Union trade rows and heartbeat snapshots so the curve is populated even on
    # no-trade days; falls back to trades-only on older DBs lacking the snapshot table.
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
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", color="white"); return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
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
    portfolio = _latest_portfolio_value(con)
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
