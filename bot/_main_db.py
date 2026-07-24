"""DB infrastructure helpers extracted from bot/main.py to keep it under 500 lines."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone

from loguru import logger

from bot.core.error_logger import log_exception
from bot.risk.risk_manager import RiskManager
from bot.strategy.ensemble import WEIGHTS
from bot.strategy.macro import _get_cached as _get_macro_cached
import bot.monitor.telegram_bot as tg
from config import TRADE_DB_PATH

_MACRO_DB_TTL = 4 * 3600


def _enable_wal_mode(db_path: str) -> None:
    """Enable WAL journal mode so dashboard readers don't block the bot writer."""
    try:
        con = sqlite3.connect(db_path)
        row = con.execute("PRAGMA journal_mode=WAL").fetchone()
        actual = row[0] if row else "unknown"
        if actual != "wal":
            logger.warning(
                f"WAL mode not confirmed on {db_path}: got {actual!r} — "
                "concurrent reads may block the bot writer"
            )
        else:
            logger.info(f"WAL mode verified: {db_path}")
        con.execute("PRAGMA synchronous=NORMAL")
        con.close()
    except Exception as exc:
        log_exception(logger, "_enable_wal_mode", exc, {"db_path": db_path})


def init_db(db_path: str = TRADE_DB_PATH) -> sqlite3.Connection:
    _enable_wal_mode(db_path)
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, action TEXT,
            shares REAL, price REAL, notional REAL,
            regime TEXT, portfolio_value REAL, pnl_pct REAL,
            xgb_prob REAL DEFAULT 0.0,
            lstm_prob REAL DEFAULT 0.0,
            sentiment_score REAL DEFAULT 0.0,
            macro_score REAL DEFAULT 0.0,
            ensemble_score REAL DEFAULT 0.0,
            realized_pnl REAL DEFAULT 0.0,
            order_id TEXT DEFAULT NULL,
            holding_days INTEGER DEFAULT 0,
            feature_drivers TEXT DEFAULT NULL
        )
    """)
    # Migration: add columns to existing DBs (safe no-op if already present)
    for _col in (
        "xgb_prob REAL DEFAULT 0.0",
        "lstm_prob REAL DEFAULT 0.0",
        "sentiment_score REAL DEFAULT 0.0",
        "macro_score REAL DEFAULT 0.0",
        "ensemble_score REAL DEFAULT 0.0",
        "realized_pnl REAL DEFAULT 0.0",
        "order_id TEXT DEFAULT NULL",
        "holding_days INTEGER DEFAULT 0",
        "feature_drivers TEXT DEFAULT NULL",
        "ai_reasoning TEXT DEFAULT NULL",
        "stop_loss REAL DEFAULT NULL",
        "take_profit REAL DEFAULT NULL",
        "risk_reward_ratio REAL DEFAULT NULL",
    ):
        try:
            con.execute(f"ALTER TABLE trades ADD COLUMN {_col}")
        except sqlite3.OperationalError:
            pass
    con.execute("""
        CREATE TABLE IF NOT EXISTS position_state (
            symbol TEXT PRIMARY KEY, entry_price REAL,
            high_water_mark REAL, atr_at_entry REAL, opened_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS risk_state (
            key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS earnings_cache (
            symbol TEXT PRIMARY KEY, near_earnings INTEGER, cached_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS macro_cache (
            key TEXT PRIMARY KEY, value REAL, cached_at TEXT
        )
    """)
    # Per-cycle account snapshot — lets the dashboard show live portfolio value
    # even on cycles where no trade executes (otherwise it reads $0.00 until the
    # first fill, because portfolio value was only ever recorded on trade rows).
    con.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            timestamp TEXT PRIMARY KEY,
            portfolio_value REAL,
            available_cash REAL,
            open_positions INTEGER
        )
    """)
    # Per-cycle signal log — records model output for every symbol evaluated,
    # including cycles where no trade fires. Lets the dashboard show live signals
    # rather than freezing at the last trade date.
    con.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            xgb_prob REAL,
            lstm_prob REAL,
            sentiment_score REAL,
            macro_score REAL,
            ensemble_score REAL,
            ensemble_action TEXT,
            regime TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS screener_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            screened_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            rank INTEGER,
            composite_score REAL,
            analyst_signal REAL,
            etf_momentum REAL,
            regime TEXT,
            sector TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            entry_price   REAL,
            stop_price    REAL,
            target_price  REAL,
            rr_ratio      REAL,
            setup_type    TEXT,
            xgb_prob      REAL,
            lstm_prob     REAL,
            ensemble_score REAL,
            macro_score   REAL,
            outcome       TEXT DEFAULT 'pending',
            outcome_price REAL,
            outcome_pct   REAL,
            outcome_ts    TEXT
        )
    """)
    # Per-symbol daily recommendation — synced to HF via trades.db so the
    # dashboard Rec History widget actually has data (DuckDB is never pushed).
    con.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol                  TEXT NOT NULL,
            prediction_date         TEXT NOT NULL,
            recommendation          TEXT,
            confidence              REAL,
            prev_recommendation     TEXT,
            price_at_recommendation REAL,
            created_at              TEXT,
            UNIQUE(symbol, prediction_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            symbol TEXT, fetch_date TEXT, headlines_json TEXT, cached_at TEXT,
            PRIMARY KEY (symbol, fetch_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    _init_v2_tables(con)

    # ── Performance indexes ────────────────────────────────────────────────────
    # signal_log grows at ~2,808 rows/day (36 symbols × 78 cycles).
    # Without these indexes the self-JOIN in get_latest_signals_df() and the
    # ROW_NUMBER() join in recommendation_history.py degrade as O(n).
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_signal_log_sym_ts "
        "ON signal_log (symbol, timestamp DESC)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_sym_ts "
        "ON trades (symbol, timestamp DESC)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_action "
        "ON trades (action, timestamp DESC)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_signal_log_action "
        "ON signal_log (ensemble_action, timestamp DESC)"
    )

    # ── Query timing metrics ───────────────────────────────────────────────────
    con.execute("""
        CREATE TABLE IF NOT EXISTS query_metrics (
            query_name TEXT PRIMARY KEY,
            avg_ms     REAL    NOT NULL DEFAULT 0.0,
            max_ms     REAL    NOT NULL DEFAULT 0.0,
            calls      INTEGER NOT NULL DEFAULT 0,
            last_run   TEXT
        )
    """)

    con.commit()
    return con


def _init_v2_tables(con: sqlite3.Connection) -> None:
    """Create v2 advanced-feature tables (safe no-ops if already present)."""
    con.execute("CREATE TABLE IF NOT EXISTS capital_accounts ("
                "id INTEGER PRIMARY KEY, initial_deposit REAL NOT NULL DEFAULT 1000.0,"
                "ai_generated_profit REAL NOT NULL DEFAULT 0.0,"
                "reinvest_profits_only INTEGER NOT NULL DEFAULT 0,"
                "updated_at TEXT DEFAULT (datetime('now')))")
    con.execute("CREATE TABLE IF NOT EXISTS investment_theses ("
                "thesis_id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,"
                "thesis_text TEXT, price_target REAL, invalidation_criteria TEXT,"
                "review_trigger TEXT DEFAULT 'quarterly', next_review_date TEXT,"
                "confidence_at_entry INTEGER DEFAULT 75, current_validity TEXT DEFAULT 'valid',"
                "last_evaluated_date TEXT, ai_evaluation_notes TEXT,"
                "created_at TEXT DEFAULT (datetime('now')))")
    con.execute("CREATE TABLE IF NOT EXISTS decision_log ("
                "decision_id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,"
                "decision_date TEXT NOT NULL, decision_type TEXT NOT NULL,"
                "price_at_decision REAL, quantity_changed REAL, reasoning TEXT,"
                "ai_confidence INTEGER, portfolio_value_at_time REAL,"
                "triggered_by TEXT DEFAULT 'ai', created_at TEXT DEFAULT (datetime('now')))")
    con.execute("CREATE TABLE IF NOT EXISTS daily_changes ("
                "change_id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,"
                "change_date TEXT NOT NULL, confidence_yesterday INTEGER,"
                "confidence_today INTEGER, action_yesterday TEXT, action_today TEXT,"
                "action_changed INTEGER DEFAULT 0, change_reason TEXT,"
                "significance TEXT DEFAULT 'minor', UNIQUE(symbol, change_date))")
    con.execute("CREATE TABLE IF NOT EXISTS investor_profile ("
                "id INTEGER PRIMARY KEY, last_updated TEXT,"
                "avg_hold_duration_days REAL DEFAULT 0.0, early_exit_rate REAL DEFAULT 0.0,"
                "trim_compliance_rate REAL DEFAULT 0.0,"
                "best_performing_sectors TEXT DEFAULT '[]',"
                "worst_performing_sectors TEXT DEFAULT '[]',"
                "best_market_condition TEXT, worst_market_condition TEXT,"
                "behavioral_insights TEXT DEFAULT '[]', ai_adaptations TEXT DEFAULT '[]')")
    con.execute("CREATE TABLE IF NOT EXISTS behavioral_observations ("
                "observation_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "observation_date TEXT NOT NULL, observation_type TEXT NOT NULL,"
                "symbol TEXT, trade_id INTEGER, outcome TEXT, notes TEXT)")


def _record_snapshot(con: sqlite3.Connection, portfolio_value: float, available_cash: float, open_positions: int) -> None:
    """Write a heartbeat snapshot of account value for the dashboard.

    Runs every cycle regardless of whether a trade happened, so the dashboard
    always has a fresh portfolio value to display.
    """
    con.execute(
        "INSERT OR REPLACE INTO portfolio_snapshots "
        "(timestamp, portfolio_value, available_cash, open_positions) VALUES (?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), portfolio_value, available_cash, open_positions),
    )
    con.commit()
    logger.info(
        f"Snapshot recorded — portfolio=${portfolio_value:,.2f}, cash=${available_cash:,.2f}, "
        f"open_positions={open_positions}"
    )


def _log_signal(
    con: sqlite3.Connection, symbol: str,
    xgb_prob: float, lstm_prob: float, sentiment_score: float,
    macro_score: float, regime: str, ensemble_action: str,
) -> None:
    """Record model output for every symbol evaluated each cycle.

    Runs regardless of whether a trade fires, so the dashboard Signals tab
    always shows live data even during hold-only cycles.
    """
    sent_norm     = (sentiment_score + 1.0) / 2.0
    ensemble_score = (
        WEIGHTS["xgb"]       * xgb_prob +
        WEIGHTS["lstm"]      * lstm_prob +
        WEIGHTS["sentiment"] * sent_norm +
        WEIGHTS["macro"]     * macro_score
    )
    con.execute(
        "INSERT INTO signal_log "
        "(timestamp, symbol, xgb_prob, lstm_prob, sentiment_score, macro_score, ensemble_score, ensemble_action, regime) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), symbol,
         round(xgb_prob, 4), round(lstm_prob, 4), round(sentiment_score, 4),
         round(macro_score, 4), round(ensemble_score, 4), ensemble_action, regime),
    )
    # Caller is responsible for committing — batch all 25 signal rows in one fsync


def _log_recommendation(
    con: sqlite3.Connection,
    symbol: str,
    recommendation: str,
    confidence: float,
    price: float | None = None,
) -> None:
    """Upsert today's recommendation for symbol into trades.db.

    Writes to SQLite (not DuckDB) so push_db() picks it up and the dashboard
    Rec History widget has data on HuggingFace Spaces.
    """
    today = date.today().isoformat()
    now   = datetime.now(timezone.utc).isoformat()
    prev_row = con.execute(
        "SELECT recommendation FROM recommendations "
        "WHERE symbol = ? ORDER BY prediction_date DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    prev = prev_row[0] if prev_row else None
    con.execute(
        """INSERT INTO recommendations
               (symbol, prediction_date, recommendation, confidence,
                prev_recommendation, price_at_recommendation, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(symbol, prediction_date) DO UPDATE SET
               recommendation          = excluded.recommendation,
               confidence              = excluded.confidence,
               prev_recommendation     = excluded.prev_recommendation,
               price_at_recommendation = excluded.price_at_recommendation,
               created_at              = excluded.created_at""",
        (symbol, today, recommendation, round(confidence, 4), prev, price, now),
    )
    # Caller commits after batching all symbols



def _anchor_daily_start(con: sqlite3.Connection) -> tuple[float | None, str]:
    """Pick the start-of-day equity baseline for Day P&L.

    Looks for the last snapshot from the nearest prior BUSINESS day only.
    If the bot did not run that day (e.g., Mon run then Thu restart — no
    Tue/Wed data), returns None and reset_daily anchors to today's opening
    portfolio value (Day P&L = $0 at session start, correct for a fresh start).

    A 4-day window would wrongly accept Mon data on Thu, reporting "return
    since last run" instead of "today's gain".  Exact prior-business-day
    prevents that while still handling Fri→Mon (3-calendar-day) weekends.

    Returns (value_or_None, source_description).
    """
    today = date.today()
    prior = today - timedelta(days=1)
    while prior.weekday() >= 5:   # skip Saturday(5) and Sunday(6)
        prior -= timedelta(days=1)
    prior_start = prior.isoformat()
    prior_end   = today.isoformat()   # exclusive upper bound

    row = con.execute(
        "SELECT timestamp, portfolio_value FROM portfolio_snapshots "
        "WHERE timestamp >= ? AND timestamp < ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (prior_start, prior_end),
    ).fetchone()
    if row and row[1] is not None:
        return float(row[1]), f"snapshot from {row[0][:10]}"

    row = con.execute(
        "SELECT timestamp, portfolio_value FROM trades "
        "WHERE timestamp >= ? AND timestamp < ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (prior_start, prior_end),
    ).fetchone()
    if row and row[1] is not None:
        return float(row[1]), f"trade from {row[0][:10]}"

    return None, f"no data for {prior_start} — starting fresh"


def log_trade(
    con: sqlite3.Connection,
    symbol: str,
    action: str,
    shares: float,
    price: float,
    notional: float,
    regime: str,
    portfolio_value: float,
    pnl_pct: float,
    xgb_prob: float = 0.0,
    lstm_prob: float = 0.0,
    sentiment_score: float = 0.0,
    macro_score: float = 0.0,
    entry_price: float = 0.0,
    order_id: str | None = None,
    holding_days: int = 0,
    feature_drivers: str | None = None,
    ai_reasoning: str | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    risk_reward_ratio: float | None = None,
) -> None:
    sentiment_norm  = (sentiment_score + 1.0) / 2.0
    ensemble_score  = (WEIGHTS["xgb"]  * xgb_prob + WEIGHTS["lstm"] * lstm_prob
                       + WEIGHTS["sentiment"] * sentiment_norm + WEIGHTS["macro"] * macro_score)
    realized_pnl = shares * (price - entry_price) if "SELL" in action and entry_price > 0 else 0.0
    con.execute(
        """INSERT INTO trades
           (timestamp, symbol, action, shares, price, notional, regime, portfolio_value, pnl_pct,
            xgb_prob, lstm_prob, sentiment_score, macro_score, ensemble_score, realized_pnl,
            order_id, holding_days, feature_drivers, ai_reasoning, stop_loss, take_profit, risk_reward_ratio)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now(timezone.utc).isoformat(),
         symbol, action, shares, price, notional, regime, portfolio_value, pnl_pct,
         xgb_prob, lstm_prob, sentiment_score, macro_score, ensemble_score, realized_pnl,
         order_id, holding_days, feature_drivers, ai_reasoning, stop_loss, take_profit, risk_reward_ratio),
    )
    con.commit()


def _week_key() -> str:
    """Return current ISO year-week string, e.g. '2026-W24'."""
    return date.today().strftime("%G-W%V")


def _load_risk_state(con: sqlite3.Connection) -> tuple[float | None, list[str], float | None, bool, bool, float | None]:
    """Returns (daily_start, day_trade_dates, weekly_start, daily_warning_sent, weekly_halt_alerted, portfolio_high)."""
    today = date.today().isoformat()
    wk    = _week_key()
    rows  = {r[0]: r[1] for r in con.execute("SELECT key, value FROM risk_state")}

    daily_start: float | None = None
    if rows.get("daily_start_date") == today:
        try:
            daily_start = float(rows["daily_start_value"])
        except (KeyError, ValueError, TypeError):
            pass

    day_trade_dates: list[str] = []
    try:
        day_trade_dates = json.loads(rows.get("day_trade_dates", "[]"))
    except (json.JSONDecodeError, TypeError):
        pass

    weekly_start: float | None = None
    if rows.get("weekly_start_week") == wk:
        try:
            weekly_start = float(rows["weekly_start_value"])
        except (KeyError, ValueError, TypeError):
            pass

    daily_warning_sent  = rows.get("daily_warning_sent_date") == today
    weekly_halt_alerted = rows.get("weekly_halt_alerted_week") == wk

    portfolio_high: float | None = None
    try:
        portfolio_high = float(rows["portfolio_high"])
    except (KeyError, ValueError, TypeError):
        pass

    return daily_start, day_trade_dates, weekly_start, daily_warning_sent, weekly_halt_alerted, portfolio_high


def _save_risk_state(con: sqlite3.Connection, risk: RiskManager) -> None:
    today = date.today().isoformat()
    wk    = _week_key()
    trades  = json.dumps([d.isoformat() for d in risk.day_trade_log])
    start   = str(risk.daily_start_value)  if risk.daily_start_value  is not None else ""
    weekly  = str(risk.weekly_start_value) if risk.weekly_start_value is not None else ""
    entries = [
        ("daily_start_value",        start),
        ("daily_start_date",         today),
        ("day_trade_dates",          trades),
        ("weekly_start_value",       weekly),
        ("weekly_start_week",        wk),
        ("daily_warning_sent_date",  today if risk.daily_warning_sent else ""),
        ("weekly_halt_alerted_week", wk    if risk.weekly_halt_alerted else ""),
        ("portfolio_high",           str(risk.portfolio_high) if risk.portfolio_high is not None else ""),
        ("trading_halted_date",      today if risk.halted else ""),
    ]
    for key, val in entries:
        con.execute(
            "INSERT OR REPLACE INTO risk_state (key, value, updated_at) VALUES (?,?,?)",
            (key, val, datetime.now(timezone.utc).isoformat())
        )
    con.commit()


def _get_macro_from_db(con: sqlite3.Connection) -> tuple[float, float, bool]:
    """Returns (score, cap, halt). halt=True means VIX >= MACRO_HALT_VIX — block all new buys."""
    now  = time.time()
    rows = {r[0]: (float(r[1]), r[2])
            for r in con.execute("SELECT key, value, cached_at FROM macro_cache")}
    if "score" in rows and "cap" in rows:
        try:
            cached_ts = datetime.fromisoformat(rows["score"][1]).timestamp()
            if now - cached_ts < _MACRO_DB_TTL:
                halt = bool(rows["halt"][0]) if "halt" in rows else False
                return rows["score"][0], rows["cap"][0], halt
        except (ValueError, TypeError):
            pass
    try:
        result = _get_macro_cached()
    except Exception as e:
        logger.warning(f"Macro fetch failed — using neutral defaults: {e}")
        result = {"score": 0.5, "cap": 1.0, "halt": False}
        tg.send(
            f"⚠️ <b>FRED macro data unavailable</b> — {e}\n"
            "VIX/yield-curve circuit breaker is disabled. Market halt protection off."
        )
    ts = datetime.now(timezone.utc).isoformat()
    for key in ("score", "cap", "halt"):
        con.execute(
            "INSERT OR REPLACE INTO macro_cache (key, value, cached_at) VALUES (?,?,?)",
            (key, float(result[key]), ts)
        )
    con.commit()
    return result["score"], result["cap"], bool(result["halt"])
