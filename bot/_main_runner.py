"""End-of-day summary, run_loop, and CLI helpers extracted from bot/main.py."""
from __future__ import annotations

import socket
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Module-level singleton socket — holds the port for the lifetime of this process.
# Declared at module scope so it is never garbage-collected.
_singleton_socket: socket.socket | None = None
_SINGLETON_PORT = 47219  # arbitrary unprivileged port, unique to this bot

import pandas as pd
import yfinance as yf
from loguru import logger

import bot.monitor.telegram_bot as tg
from bot.core.error_logger import log_exception
from bot.execution.alpaca_client import AlpacaClient
from bot.strategy.features import FEATURE_COLS
from bot.strategy.regime_classifier import RegimeClassifier
from bot.strategy.xgb_predictor import XGBPredictor
from bot.strategy.lstm_predictor import LSTMPredictor
from bot._main_db import (
    _get_macro_from_db, _load_risk_state, _week_key, init_db,
)
from bot._main_market import _is_market_hours


def _check_ml_versions() -> None:
    """Compare runtime ML library versions against what trained the models.

    Called once at run_loop startup. Sends a Telegram alert if versions differ
    so the operator knows predictions may be unreliable before trading begins.
    """
    _vfile = Path("models/runtime_versions.json")
    if not _vfile.exists():
        return
    try:
        import json as _j
        _trained = _j.loads(_vfile.read_text())
        _mismatches: list[str] = []
        try:
            import sklearn as _sk
            t = _trained.get("scikit-learn", "")
            if t and t != _sk.__version__:
                _mismatches.append(f"sklearn  trained={t}  runtime={_sk.__version__}")
        except ImportError:
            pass
        try:
            import torch as _torch
            t = _trained.get("torch", "")
            if t and t != _torch.__version__:
                _mismatches.append(f"torch    trained={t}  runtime={_torch.__version__}")
        except ImportError:
            pass
        try:
            import xgboost as _xgb_v
            t = _trained.get("xgboost", "")
            if t and t != _xgb_v.__version__:
                _mismatches.append(f"xgboost  trained={t}  runtime={_xgb_v.__version__}")
        except ImportError:
            pass
        if _mismatches:
            _msg = "\n".join(_mismatches)
            logger.error(f"ML version mismatch — model predictions may be wrong:\n{_msg}\n"
                         f"Trigger retrain.yml to rebuild with current libraries.")
            tg._send(
                f"⚠️ <b>ML version mismatch detected</b>\n"
                f"<code>{_msg}</code>\n"
                f"Regime/XGB/LSTM predictions may be silently wrong. "
                f"Trigger the weekly retrain workflow immediately."
            )
        else:
            logger.info(
                f"ML versions OK — sklearn={_trained.get('scikit-learn')} "
                f"torch={_trained.get('torch')} xgb={_trained.get('xgboost')}"
            )
    except Exception as _ve:
        logger.debug(f"ML version check skipped: {_ve}")


def end_of_day_summary() -> None:
    today_str = date.today().isoformat()
    _eod_sentinel     = Path(f"data/.eod_sent_{today_str}")
    _started_sentinel = Path(f"data/.trading_started_{today_str}")

    if _eod_sentinel.exists():
        logger.info("EOD summary already sent today — skipping duplicate (run_loop already called it).")
        return
    if not _started_sentinel.exists():
        logger.info("Trading loop never entered the market window today — suppressing false EOD summary.")
        return

    _eod_sentinel.parent.mkdir(parents=True, exist_ok=True)
    _eod_sentinel.touch()

    import zoneinfo
    con    = init_db()
    client = AlpacaClient()
    today  = today_str

    trades_count = con.execute(
        "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (today + "%",)
    ).fetchone()[0]
    daily_start, day_trade_dates, weekly_start, _, __, ___ = _load_risk_state(con)
    day_trade_count = day_trade_dates.count(today)
    portfolio_value, available_cash = client.get_account_summary()
    positions = client.get_positions()
    day_return = ((portfolio_value - daily_start) / daily_start) if daily_start else 0.0

    vs_spy = 0.0
    try:
        _spy = yf.download("SPY", period="5d", interval="1d", progress=False, auto_adjust=True)
        if _spy is not None and len(_spy) > 1:
            if isinstance(_spy.columns, pd.MultiIndex):
                _spy.columns = [c[0].lower() for c in _spy.columns]
            else:
                _spy.columns = [c.lower() for c in _spy.columns]
            vs_spy = float(_spy["close"].pct_change().iloc[-1])
    except Exception as exc:
        logger.debug(f"spy_yf_fetch: {exc}")

    # ── Best / worst trade today ───────────────────────────────────────────────
    best_trade: tuple | None  = None
    worst_trade: tuple | None = None
    try:
        sells_today = con.execute(
            "SELECT symbol, pnl_pct, notional FROM trades "
            "WHERE timestamp LIKE ? AND action LIKE 'SELL%' AND action != 'SELL_RECONCILE'",
            (today + "%",),
        ).fetchall()
        if sells_today:
            best  = max(sells_today, key=lambda r: float(r[1] or 0))
            worst = min(sells_today, key=lambda r: float(r[1] or 0))
            best_trade  = (best[0],  float(best[1]  or 0), float(best[1]  or 0) * float(best[2]  or 0))
            worst_trade = (worst[0], float(worst[1] or 0), float(worst[1] or 0) * float(worst[2] or 0))
    except Exception as _e:
        logger.warning(f"best/worst trade fetch failed: {_e}")

    # ── Portfolio Health Score (same formula as dashboard) ────────────────────
    health_score = 0
    try:
        macro_score_val, _, macro_halt = _get_macro_from_db(con)
        _vix_pts  = 5 if macro_halt else (25 if macro_score_val > 0.65 else (15 if macro_score_val > 0.40 else 5))
        cash_pct  = available_cash / portfolio_value * 100 if portfolio_value > 0 else 100.0
        _cash_pts = 25 if cash_pct > 30 else (15 if cash_pct > 15 else 5)
        max_conc  = 0.0
        if positions and portfolio_value > 0:
            max_conc = max(
                float(getattr(p, "market_value", 0) or 0) / portfolio_value * 100
                for p in positions.values()
            )
        _conc_pts = 25 if max_conc < 15 else (15 if max_conc < 25 else 5)
        max_dd    = 0.0
        pv_rows   = con.execute(
            "SELECT portfolio_value FROM trades WHERE portfolio_value > 0 ORDER BY id"
        ).fetchall()
        if len(pv_rows) > 1:
            vals = [float(r[0]) for r in pv_rows]
            pk   = vals[0]
            for v in vals:
                pk     = max(pk, v)
                max_dd = max(max_dd, (pk - v) / pk * 100 if pk > 0 else 0)
        _dd_pts   = 25 if max_dd < 3 else (15 if max_dd < 8 else 5)
        health_score = _vix_pts + _cash_pts + _conc_pts + _dd_pts
    except Exception as _e:
        logger.warning(f"health score calc failed: {_e}")
        cash_pct = 0.0

    tg.alert_daily_summary(
        day_return=day_return,
        vs_spy=vs_spy,
        positions=list(positions.keys()),
        cash=available_cash,
        trades=trades_count,
        day_trades=day_trade_count,
        portfolio_value=portfolio_value,
        best_trade=best_trade,
        worst_trade=worst_trade,
        cash_pct=cash_pct,
        health_score=health_score,
    )
    logger.info(f"End-of-day summary sent: return={day_return:.2%}, trades={trades_count}, health={health_score}")

    try:
        from database.services.analytics_service import analytics_service as _as
        _as.save_daily_snapshot({"portfolio_value": portfolio_value, "cash": available_cash})
    except Exception as _ae:
        log_exception(logger, "end_of_day.save_snapshot", _ae)

    # Friday weekly report — Portfolio Manager visibility into week performance
    et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    if et.weekday() == 4:  # Friday
        try:
            monday = date.today() - timedelta(days=date.today().weekday())
            week_rows = con.execute(
                "SELECT action, pnl_pct, portfolio_value FROM trades WHERE timestamp >= ?",
                (monday.isoformat(),)
            ).fetchall()
            week_sells  = [r for r in week_rows if r[0].startswith("SELL")]
            week_wins   = [r for r in week_sells if r[1] > 0]
            week_wr     = len(week_wins) / len(week_sells) if week_sells else 0.0
            week_return = ((portfolio_value - weekly_start) / weekly_start) if weekly_start else 0.0
            pv_week     = [r[2] for r in week_rows if r[2] > 0]
            week_dd     = 0.0
            if pv_week:
                pk = pv_week[0]
                for v in pv_week:
                    pk = max(pk, v)
                    week_dd = max(week_dd, (pk - v) / (pk + 1e-8))
            spy_wk = 0.0
            try:
                _spy_wk = yf.download("SPY", period="15d", interval="1d", progress=False, auto_adjust=True)
                if _spy_wk is not None and len(_spy_wk) >= 6:
                    if isinstance(_spy_wk.columns, pd.MultiIndex):
                        _spy_wk.columns = [c[0].lower() for c in _spy_wk.columns]
                    else:
                        _spy_wk.columns = [c.lower() for c in _spy_wk.columns]
                    spy_wk = float(_spy_wk["close"].iloc[-1] / _spy_wk["close"].iloc[-6] - 1)
            except Exception as exc:
                logger.debug(f"spy_wk_yf_fetch: {exc}")
            tg.alert_weekly_report(
                week_return=week_return,
                vs_spy=spy_wk,
                win_rate=week_wr,
                sharpe=0.0,   # requires full intraday history; omitted for simplicity
                drawdown=week_dd,
            )
            logger.info(f"Weekly report sent: return={week_return:.2%}, win_rate={week_wr:.1%}")
        except Exception as e:
            logger.warning(f"Weekly report failed: {e}")

    # ── Sync signal_archive to DuckDB BEFORE pruning so history is preserved ──
    try:
        from database.sync.sync_jobs import run_nightly_sync
        sync_results = run_nightly_sync()
        logger.info(f"DuckDB sync: {sync_results}")
    except Exception as _se:
        log_exception(logger, "end_of_day.duckdb_sync", _se)

    # ── Prune signal_log (keep 30 days — ~84K rows max vs unbounded growth) ───
    try:
        pruned = con.execute(
            "DELETE FROM signal_log WHERE timestamp < datetime('now', '-30 days')"
        ).rowcount
        con.commit()
        if pruned:
            logger.info(f"Pruned {pruned} signal_log rows older than 30 days")
    except Exception as _pe:
        log_exception(logger, "end_of_day.prune_signal_log", _pe)

    # ── Flush query timing metrics accumulated during the trading session ──────
    try:
        from database.query_metrics import flush_to_db as _flush_metrics
        _flush_metrics(con)
    except Exception as _me:
        log_exception(logger, "end_of_day.flush_metrics", _me)

    con.close()


def _acquire_singleton() -> None:
    """Bind a local TCP port to guarantee only one bot process runs at a time.

    A second invocation (e.g. overlapping cron) cannot bind the same port and
    exits immediately with CRITICAL log + non-zero exit code. The OS reclaims
    the port automatically when the first process terminates — no cleanup file
    needed, survives crashes and kill signals.
    """
    global _singleton_socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        sock.bind(("127.0.0.1", _SINGLETON_PORT))
        _singleton_socket = sock  # keep alive for process lifetime
    except OSError:
        sock.close()
        logger.critical(
            f"Bot already running (port {_SINGLETON_PORT} in use) — "
            "this instance will exit to prevent duplicate trades and log noise."
        )
        sys.exit(1)


def run_loop(mode: str = "paper") -> None:
    """Load models once, cycle every 5 minutes until market close.

    Eliminates per-cycle Python startup + model-loading overhead (~10-30 s/cycle).
    Models (XGB, LSTM, scaler) stay in memory for the full trading session.
    """
    from bot.main import run  # lazy import — avoids circular at module level

    _acquire_singleton()
    logger.info("Long-running mode — loading models once for full session.")
    regime_clf = RegimeClassifier()
    xgb        = XGBPredictor()
    lstm       = LSTMPredictor()
    _check_ml_versions()
    _missing_models = [m for m, p in [("XGBoost", xgb), ("LSTM", lstm)] if p.model is None]
    if _missing_models:
        tg.send(
            f"⚠️ <b>Model(s) missing: {', '.join(_missing_models)}</b>\n"
            "All signals default to 0.5 (neutral) — no trades will fire.\n"
            "Check HuggingFace model sync or run retraining."
        )

    # Stale-model detection: check validation report matches current feature set.
    _report_path = Path("models/validation_report.json")
    if _report_path.exists():
        try:
            import json as _json
            _report = _json.loads(_report_path.read_text())
            _trained_features = _report.get("feature_count", 0)
            _current_features = len(FEATURE_COLS)
            if _trained_features != _current_features:
                tg.send(
                    f"⚠️ <b>Stale models detected</b>\n"
                    f"Models trained on {_trained_features} features; "
                    f"current code expects {_current_features}.\n"
                    "Signals will fail or be random. Run retraining immediately."
                )
                logger.error(
                    f"STALE MODEL: trained_features={_trained_features} "
                    f"!= current={_current_features}. Retrain required."
                )
        except Exception as _ve:
            logger.warning(f"Could not validate model report: {_ve}")

    client = AlpacaClient()
    if not _is_market_hours(client.api):
        logger.info("Market is closed at session start — nothing to trade.")
        tg._send("⚠️ <b>Trading Bot fired but market is closed</b>. Check cron schedule or holiday calendar.")
        return

    try:
        from database.services.analytics_service import analytics_service as _as
        _health = _as.check_health()
        if _health.get("overall") != "ok":
            logger.warning(f"Analytics service degraded at startup: {_health}")
    except Exception as _ahe:
        log_exception(logger, "run_loop.analytics_health", _ahe)

    # Mark that trading actually entered the market window today.
    # end_of_day_summary() checks for this sentinel so it can suppress
    # false EOD messages sent when the bot fires outside market hours.
    _started_sentinel = Path(f"data/.trading_started_{date.today().isoformat()}")
    _started_sentinel.parent.mkdir(parents=True, exist_ok=True)
    _started_sentinel.touch()

    cycle = 0
    consecutive_failures = 0
    while _is_market_hours(client.api):
        cycle += 1
        logger.info(f"\n=== Loop cycle {cycle} ===")
        try:
            run(mode=mode, _regime_clf=regime_clf, _xgb=xgb, _lstm=lstm, _client=client)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Cycle {cycle} crashed: {e}")
            tg._send(f"⚠️ <b>CYCLE {cycle} CRASHED</b> — {e}. Continuing to next cycle.")
            if consecutive_failures == 3:
                tg._send(
                    f"🔴 <b>3 consecutive cycle crashes</b> — likely Alpaca API outage. "
                    f"Bot is retrying every 5 min. Check status.alpaca.markets"
                )
        for _ in range(10):
            time.sleep(30)
            if not _is_market_hours(client.api):
                break

    logger.info("Market closed — loop complete.")
    # Wait until 4:05pm ET so late fills settle before sending the summary
    try:
        import zoneinfo as _zi
        _et  = datetime.now(_zi.ZoneInfo("America/New_York"))
        _tgt = _et.replace(hour=16, minute=5, second=0, microsecond=0)
        _wait = (_tgt - _et).total_seconds()
        if 0 < _wait < 600:   # max 10 min — guard against clock drift
            logger.info(f"Waiting {_wait:.0f}s for 4:05pm ET before sending daily summary.")
            time.sleep(_wait)
    except Exception as exc:
        logger.debug(f"eod_sleep_timer: {exc}")
    try:
        end_of_day_summary()
    except Exception as e:
        logger.error(f"End-of-day summary failed: {e}")


def _do_reset_daily_start() -> None:
    """Clear the stale daily-start anchor and restore the correct weekly baseline.

    Daily: deletes daily_start_value/date so the next bot cycle calls
    _anchor_daily_start (returns None when no prior-business-day data exists),
    then reset_daily(current_portfolio_value) — Day P&L = $0 at session open.

    Weekly: restores weekly_start to the FIRST portfolio value seen this ISO week
    (e.g. Monday's $100,000) rather than today's restart value, so Week P&L
    correctly shows Mon-to-now gain instead of just today's intraday move.
    """
    con = init_db()
    # Clear stale daily anchor
    con.execute(
        "DELETE FROM risk_state WHERE key IN ('daily_start_value', 'daily_start_date')"
    )
    # Restore weekly_start to the first portfolio value of the current ISO week
    today = date.today()
    monday = today - timedelta(days=today.weekday())   # Monday of this ISO week
    mon_iso = monday.isoformat()
    wk = _week_key()
    row = con.execute(
        "SELECT portfolio_value FROM portfolio_snapshots "
        "WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
        (mon_iso,),
    ).fetchone()
    if row is None or row[0] is None:
        row = con.execute(
            "SELECT portfolio_value FROM trades "
            "WHERE timestamp >= ? AND portfolio_value IS NOT NULL "
            "ORDER BY timestamp ASC LIMIT 1",
            (mon_iso,),
        ).fetchone()
    if row and row[0] is not None:
        weekly_val = float(row[0])
        ts = datetime.now(timezone.utc).isoformat()
        for k, v in [("weekly_start_value", str(weekly_val)), ("weekly_start_week", wk)]:
            con.execute(
                "INSERT OR REPLACE INTO risk_state (key, value, updated_at) VALUES (?,?,?)",
                (k, v, ts),
            )
        logger.info(f"weekly_start restored to {weekly_val:.2f} (first value from {mon_iso})")
    else:
        logger.warning(f"No portfolio data found from {mon_iso} — weekly_start unchanged")
    con.commit()
    con.close()
    logger.info("daily_start cleared — next bot cycle will re-anchor Day P&L to today's open.")


def _do_clean_db() -> None:
    """Wipe all trading history and push a fresh empty DB to HuggingFace.

    Run AFTER resetting the Alpaca paper account to $100,000 via the Alpaca
    dashboard (https://app.alpaca.markets → Paper Trading → Reset account).
    The next bot cycle starts with clean books — all P&L, Holdings, and
    Signals metrics will be accurate from day one with no stale data.
    """
    con = init_db()
    tables = ("trades", "position_state", "risk_state",
              "portfolio_snapshots", "signal_log", "screener_log")
    for table in tables:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            con.execute(f"DELETE FROM {table}")
            logger.info(f"clean_db: cleared {n} rows from {table}")
        except Exception as exc:
            logger.warning(f"clean_db: could not clear {table} — {exc}")
    con.commit()
    con.close()
    logger.info("clean_db: all trading data wiped — pushing fresh DB to HuggingFace…")
    from bot.monitor.sync_db import push_db
    if push_db():
        logger.info("clean_db: empty DB pushed to HuggingFace — ready for clean start.")
    else:
        logger.warning("clean_db: DB cleared locally but HF push failed — "
                       "the dashboard will still show old data until the next successful push.")
