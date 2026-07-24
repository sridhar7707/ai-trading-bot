"""
Gradio data-validation tests — Phase 1 + Phase 2 feature coverage.

These tests call the same render functions the Gradio UI calls, seeding a real
SQLite DB with known values and asserting the HTML output contains correct data.
No server needed → runs in CI automatically via pytest tests/.

To also hit a live server (localhost:7860), run:
    pytest tests/test_gradio_client.py -v -m live --live
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("_BOT_LOG_HANDLER_ADDED", "1")

import bot.monitor.dashboard_data as dd
import dashboard.data as ddata
from bot.main import init_db

# ── Seed constants ────────────────────────────────────────────────────────────
TODAY      = date.today().isoformat()
POOL_ALLOC = 5_000.0
POOL_CASH  = 3_200.0   # 5000 - 1800 invested in two buys
POOL_INVST = 1_800.0
POOL_RESRV = 200.0
POOL_PNL   = 50.0      # realized profit so far


def _ts(hour: int) -> str:
    return f"{TODAY}T{hour:02d}:00:00+00:00"


# ── Shared DB fixture ─────────────────────────────────────────────────────────
@pytest.fixture
def phase2_db(tmp_path, monkeypatch):
    """DB seeded with capital pool, daily actions, and attribution trade data."""
    db_path = str(tmp_path / "p2.db")
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()

    # ── Trades for attribution (Phase 2C) ─────────────────────────────────────
    def _trade(ts, sym, action, shares, price, notional, pv, pnl,
               xgb=0.0, lstm=0.0, sent=0.0, macro=0.0, ens=0.0):
        con.execute(
            "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,"
            "portfolio_value,pnl_pct,xgb_prob,lstm_prob,sentiment_score,macro_score,"
            "ensemble_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, sym, action, shares, price, notional, "TRENDING_UP",
             pv, pnl, xgb, lstm, sent, macro, ens),
        )

    _trade(_ts(9),  "AAPL", "BUY",  5.0, 180.0,  900.0, 10_000.0, 0.0,
           xgb=0.82, lstm=0.75, sent=0.60, macro=0.40, ens=0.70)
    _trade(_ts(10), "AAPL", "SELL", 5.0, 195.0,  975.0, 10_500.0, 0.083,
           xgb=0.82, lstm=0.75, sent=0.60, macro=0.40, ens=0.70)
    _trade(_ts(11), "MSFT", "BUY",  3.0, 300.0,  900.0, 10_500.0, 0.0,
           xgb=0.65, lstm=0.60, sent=0.55, macro=0.40, ens=0.62)
    _trade(_ts(12), "MSFT", "SELL", 3.0, 285.0,  855.0, 10_350.0, -0.05,
           xgb=0.65, lstm=0.60, sent=0.55, macro=0.40, ens=0.62)
    _trade(_ts(13), "NVDA", "BUY",  2.0, 450.0,  900.0, 10_350.0, 0.0,
           xgb=0.88, lstm=0.80, sent=0.70, macro=0.40, ens=0.80)
    _trade(_ts(14), "NVDA", "SELL", 2.0, 480.0,  960.0, 10_500.0, 0.067,
           xgb=0.88, lstm=0.80, sent=0.70, macro=0.40, ens=0.80)

    # snapshot → drives portfolio value
    con.execute("INSERT INTO portfolio_snapshots VALUES (?,?,?,?)",
                (_ts(15), 10_500.0, 7_000.0, 1))

    # risk_state (required by overview render)
    for k, v in [
        ("daily_start_value",  "10000.0"),
        ("daily_start_date",   TODAY),
        ("weekly_start_value", "9800.0"),
        ("weekly_start_week",  date.today().strftime("%G-W%V")),
        ("day_trade_dates",    "[]"),
        ("portfolio_high",     "10500.0"),
        ("weekly_halt_alerted_week", ""),
    ]:
        con.execute("INSERT INTO risk_state (key,value,updated_at) VALUES (?,?,?)",
                    (k, v, _ts(15)))

    # macro_cache
    for k, v in [("score", 0.40), ("cap", 1.0), ("halt", 0.0)]:
        con.execute("INSERT INTO macro_cache (key,value,cached_at) VALUES (?,?,?)",
                    (k, v, _ts(15)))

    # ── Capital pool (Phase 2A) ───────────────────────────────────────────────
    con.execute(
        "CREATE TABLE IF NOT EXISTS capital_pools ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, status TEXT, "
        "allocated_amount REAL, available_cash REAL, invested_amount REAL, "
        "reserve REAL DEFAULT 0, realized_profit REAL DEFAULT 0, "
        "created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))"
    )
    con.execute(
        "INSERT INTO capital_pools (name,status,allocated_amount,available_cash,"
        "invested_amount,reserve,realized_profit) VALUES (?,?,?,?,?,?,?)",
        ("default", "active", POOL_ALLOC, POOL_CASH, POOL_INVST, POOL_RESRV, POOL_PNL),
    )

    # ── Daily actions (Phase 2B) ──────────────────────────────────────────────
    con.execute(
        "CREATE TABLE IF NOT EXISTS daily_actions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_date TEXT, symbol TEXT, "
        "action_type TEXT, reasoning TEXT, confidence INTEGER DEFAULT 0, "
        "expected_impact TEXT, recommended_time TEXT DEFAULT 'Today', "
        "status TEXT DEFAULT 'pending', estimated_minutes INTEGER DEFAULT 2, "
        "created_at TEXT DEFAULT (datetime('now')))"
    )
    con.execute(
        "INSERT INTO daily_actions (session_date,symbol,action_type,reasoning,"
        "confidence,expected_impact,status) VALUES (?,?,?,?,?,?,?)",
        (TODAY, "AAPL", "buy", "XGB high confidence breakout", 82, "+$150", "pending"),
    )
    con.execute(
        "INSERT INTO daily_actions (session_date,symbol,action_type,reasoning,"
        "confidence,expected_impact,status) VALUES (?,?,?,?,?,?,?)",
        (TODAY, "NVDA", "sell", "Take-profit target reached", 75, "+$60", "executed"),
    )

    con.commit()
    con.close()

    # Patch both data layers so all render functions hit the test DB.
    monkeypatch.setattr(dd,    "_DB",      db_path)
    monkeypatch.setattr(dd,    "_HALT_FILE", Path(tmp_path / "NO_HALT"))
    monkeypatch.setattr(ddata, "DB_PATH",  db_path)
    # Clear any stale cache so tests see seeded data immediately.
    ddata._CACHE.clear()
    ddata._CACHE_TS = 0.0
    monkeypatch.delenv("SPACE_ID", raising=False)
    return db_path


# ── Phase 1: Capital tab (existing components) ────────────────────────────────

def test_capital_overview_shows_dollar_values(phase2_db):
    from dashboard.components.capital import render_capital_overview
    html = render_capital_overview()
    assert "$" in html, "Capital overview must show dollar amounts"
    assert "Deposit" in html or "Initial" in html


def test_capital_chart_returns_figure(phase2_db):
    from dashboard.components.capital import render_capital_chart
    import plotly.graph_objects as go
    fig = render_capital_chart()
    assert isinstance(fig, go.Figure)


def test_profit_breakdown_shows_realized_and_unrealized(phase2_db):
    from dashboard.components.capital import render_profit_breakdown
    html = render_profit_breakdown()
    assert "Realized" in html
    assert "Unrealized" in html
    assert "$" in html


# ── Phase 2A: Managed Capital Pool ───────────────────────────────────────────

def test_managed_capital_heading_visible(phase2_db):
    from dashboard.components.capital import render_managed_capital
    html = render_managed_capital()
    assert "Managed Capital Pool" in html or "Capital Pool" in html


def test_managed_capital_allocated_amount(phase2_db):
    from dashboard.components.capital import render_managed_capital
    html = render_managed_capital()
    # $5,000.00 allocated
    assert "5,000" in html, f"Expected allocated $5,000 in output:\n{html[:500]}"


def test_managed_capital_tradeable_cash(phase2_db):
    from dashboard.components.capital import render_managed_capital
    html = render_managed_capital()
    assert "Tradeable" in html
    # tradeable = 3200 - 200 reserve = 3000
    assert "3,000" in html, f"Expected tradeable cash $3,000:\n{html[:500]}"


def test_managed_capital_invested_amount(phase2_db):
    from dashboard.components.capital import render_managed_capital
    html = render_managed_capital()
    assert "Invested" in html
    assert "1,800" in html, f"Expected invested $1,800:\n{html[:500]}"


def test_managed_capital_total_value(phase2_db):
    from dashboard.components.capital import render_managed_capital
    html = render_managed_capital()
    # total = available_cash + invested = 3200 + 1800 = 5000
    assert "Total" in html


def test_managed_capital_shows_reserve(phase2_db):
    from dashboard.components.capital import render_managed_capital
    html = render_managed_capital()
    assert "Reserve" in html


# ── Phase 2B: Decision Workspace ─────────────────────────────────────────────

def test_decision_bar_renders_without_error(phase2_db):
    from dashboard.components.decision_bar import render_decision_bar
    html = render_decision_bar()
    assert html  # non-empty
    assert "Error" not in html[:100], f"render_decision_bar returned error HTML:\n{html[:200]}"


def test_decision_bar_shows_pending_action(phase2_db):
    from dashboard.components.decision_bar import render_decision_bar
    html = render_decision_bar()
    # Should show the AAPL pending buy recommendation
    assert "AAPL" in html or "pending" in html.lower() or "AI" in html


def test_daily_actions_get_pending_returns_list(phase2_db):
    import sqlite3 as _sq
    from bot.decision.daily_actions import get_pending
    con = _sq.connect(phase2_db)
    actions = get_pending(con)
    con.close()
    assert isinstance(actions, list)
    # Only the pending AAPL action; NVDA is executed
    assert len(actions) == 1
    assert actions[0]["symbol"] == "AAPL"
    assert actions[0]["confidence"] == 82


def test_daily_actions_mark_executed_returns_bool(phase2_db):
    import sqlite3 as _sq
    from bot.decision.daily_actions import mark_executed, get_pending
    con = _sq.connect(phase2_db)
    actions = get_pending(con)
    action_id = actions[0]["id"]
    result = mark_executed(con, action_id)
    assert result is True
    # Now get_pending should return empty
    assert get_pending(con) == []
    con.close()


def test_daily_actions_mark_executed_returns_false_on_missing_id(phase2_db):
    import sqlite3 as _sq
    from bot.decision.daily_actions import mark_executed
    con = _sq.connect(phase2_db)
    result = mark_executed(con, 99999)  # non-existent id
    assert result is False
    con.close()


def test_daily_actions_get_pending_safe_on_fresh_db(tmp_path):
    """get_pending returns [] on a DB without the daily_actions table."""
    import sqlite3 as _sq
    from bot.decision.daily_actions import get_pending
    con = _sq.connect(str(tmp_path / "fresh.db"))
    result = get_pending(con)
    con.close()
    assert result == []


# ── Phase 2C: Portfolio Attribution ──────────────────────────────────────────

def test_attribution_by_symbol_shows_traded_symbols(phase2_db):
    from dashboard.components.attribution import render_attribution_by_symbol
    html = render_attribution_by_symbol()
    assert "AAPL" in html
    assert "NVDA" in html


def test_attribution_by_symbol_shows_win_rate(phase2_db):
    from dashboard.components.attribution import render_attribution_by_symbol
    html = render_attribution_by_symbol()
    # AAPL: 1 win (pnl +8.3%), NVDA: 1 win (pnl +6.7%), MSFT: 1 loss
    assert "%" in html, "Attribution should show win rate percentages"


def test_attribution_by_sector_renders(phase2_db):
    from dashboard.components.attribution import render_attribution_by_sector
    html = render_attribution_by_sector()
    assert html  # non-empty
    assert "Error" not in html[:100]


def test_attribution_by_model_renders(phase2_db):
    from dashboard.components.attribution import render_attribution_by_model
    html = render_attribution_by_model()
    assert html
    # Should mention at least one model name
    assert any(m in html for m in ("XGB", "LSTM", "Sentiment", "Ensemble", "Macro"))


def test_attribution_by_trade_shows_top_and_bottom(phase2_db):
    from dashboard.components.attribution import render_attribution_by_trade
    html = render_attribution_by_trade()
    assert html
    # AAPL was the best (pnl=0.083, notional=975 → $75 profit)
    # MSFT was the worst (pnl=-0.05, notional=855 → -$45)
    assert "AAPL" in html or "NVDA" in html  # in top trades
    assert "MSFT" in html                    # in bottom trades


def test_attribution_by_trade_renders_two_sections(phase2_db):
    from dashboard.components.attribution import render_attribution_by_trade
    html = render_attribution_by_trade()
    assert "Top" in html and ("Bottom" in html or "Worst" in html)


# ── Gradio live-server tests (skip when server is not running) ────────────────

def _server_up() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:7860/info", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _server_up(), reason="Gradio server not running on :7860")
def test_live_server_info_endpoint():
    """Server responds with 200 on /info."""
    import requests
    r = requests.get("http://localhost:7860/info", timeout=5)
    assert r.status_code == 200


@pytest.mark.skipif(not _server_up(), reason="Gradio server not running on :7860")
def test_live_gradio_client_connects():
    """gradio_client.Client can connect and list the API."""
    from gradio_client import Client
    client = Client("http://localhost:7860", verbose=False)
    assert client is not None
