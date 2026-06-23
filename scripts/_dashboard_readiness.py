"""Go-live readiness check, extracted from dashboard.py."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_readiness_check() -> str:
    try:
        import sqlite3, numpy as np
        from config import TRADE_DB_PATH
        from datetime import datetime, timezone
        from bot.monitor.dashboard_data import _spy_return_alpaca

        con = sqlite3.connect(TRADE_DB_PATH)

        # ── Trade rows (for win rate + first-trade date) ──────────────────────
        trade_rows = con.execute(
            "SELECT timestamp, action, pnl_pct FROM trades ORDER BY timestamp"
        ).fetchall()

        # ── Portfolio value series: union trades + heartbeat snapshots ────────
        # Using trades alone gives 0 history when no SELL has fired (portfolio
        # value is only recorded in trade rows, which haven't changed since the
        # last BUY). portfolio_snapshots updates every 5-min cycle — essential
        # for accurate Sharpe, drawdown, and total-return calculations.
        try:
            pv_rows = con.execute(
                "SELECT timestamp, portfolio_value FROM trades "
                "WHERE portfolio_value IS NOT NULL "
                "UNION ALL "
                "SELECT timestamp, portfolio_value FROM portfolio_snapshots "
                "WHERE portfolio_value IS NOT NULL "
                "ORDER BY timestamp"
            ).fetchall()
        except Exception:
            pv_rows = con.execute(
                "SELECT timestamp, portfolio_value FROM trades "
                "WHERE portfolio_value IS NOT NULL ORDER BY timestamp"
            ).fetchall()
        con.close()

        if not trade_rows and not pv_rows:
            return "⚠️ No trades in database yet — run paper trading first."

        # ── Days trading: first-trade to TODAY (not last-trade) ──────────────
        # First-trade → last-trade would be 0d if all opens happened on the same
        # day; measuring to now correctly reflects how long the bot has been live.
        ts_first  = datetime.fromisoformat(trade_rows[0][0]) if trade_rows else \
                    datetime.fromisoformat(pv_rows[0][0])
        now_utc   = datetime.now(timezone.utc)
        ts_first_aware = ts_first.replace(tzinfo=timezone.utc) if ts_first.tzinfo is None else ts_first
        days = (now_utc - ts_first_aware).days

        # ── Win rate: closed trades only ──────────────────────────────────────
        sells    = [r for r in trade_rows if r[1].startswith("SELL")]
        n_sells  = len(sells)
        win_rate = sum(1 for r in sells if r[2] and r[2] > 0) / n_sells if n_sells else None

        # ── Sharpe, drawdown, consecutive losses from the full PV series ──────
        vals = np.array([r[1] for r in pv_rows if r[1] is not None], dtype=float)
        if len(vals) >= 2:
            rets   = np.diff(vals) / (vals[:-1] + 1e-8)
            std_r  = float(np.std(rets))
            sharpe = float(np.mean(rets) / std_r * np.sqrt(252 * 78)) if std_r > 0 else 0.0
            peak = vals[0]; max_dd = 0.0
            for v in vals:
                peak   = max(peak, v)
                max_dd = max(max_dd, (peak - v) / (peak + 1e-8))
            # Consecutive losing CALENDAR DAYS (one return per day)
            daily_close: dict[str, float] = {}
            for ts, pv in pv_rows:
                if pv is not None:
                    daily_close[ts[:10]] = float(pv)
            day_vals  = [daily_close[d] for d in sorted(daily_close)]
            day_rets  = np.diff(np.array(day_vals)) / (np.array(day_vals[:-1]) + 1e-8) if len(day_vals) >= 2 else []
            streak = cur = 0
            for r in day_rets:
                if r < 0: cur += 1; streak = max(streak, cur)
                else: cur = 0
            bot_ret  = float((vals[-1] - vals[0]) / (vals[0] + 1e-8))
        else:
            sharpe = max_dd = streak = bot_ret = 0.0

        spy_ret = _spy_return_alpaca(ts_first.date().isoformat()) or 0.0

        T = {"min_days": 60, "min_win_rate": 0.52, "min_sharpe": 1.0,
             "max_drawdown": 0.15, "max_consec_loss": 4}

        win_str  = f"{win_rate:.1%}" if win_rate is not None else f"n/a ({n_sells} closed trades)"
        win_pass = win_rate >= T["min_win_rate"] if win_rate is not None else False

        gates = [
            ("Days trading",      f"{days}d",        f"≥ {T['min_days']}d",         days >= T["min_days"]),
            ("Win rate",          win_str,            f"≥ {T['min_win_rate']:.0%}",  win_pass),
            ("Sharpe ratio",      f"{sharpe:.2f}",   f"≥ {T['min_sharpe']:.1f}",    sharpe >= T["min_sharpe"]),
            ("Max drawdown",      f"{max_dd:.1%}",   f"≤ {T['max_drawdown']:.0%}",  max_dd <= T["max_drawdown"]),
            ("Max consec losses", str(streak),        f"≤ {T['max_consec_loss']}d", streak <= T["max_consec_loss"]),
            ("vs S&P 500",        f"{bot_ret:.2%}",  f"> SPY {spy_ret:.2%}",        bot_ret > spy_ret),
        ]
        all_pass = all(g[3] for g in gates)
        header   = "## ✅ ALL CHECKS PASSED — Ready for live trading!\n\n" if all_pass else \
                   "## ❌ NOT READY — Keep paper trading.\n\n"
        note     = (f"> 📊 Based on {days}d of trading history · "
                    f"{len(trade_rows)} trade rows · {len(pv_rows)} portfolio snapshots\n\n")
        rows_md  = "\n".join(
            f"| {'✅' if ok else '❌'} | {name} | {val} | {thr} |"
            for name, val, thr, ok in gates
        )
        return header + note + f"| | Gate | Value | Threshold |\n|--|------|-------|----------|\n{rows_md}"
    except Exception as e:
        return f"⚠️ Check failed: {e}"
