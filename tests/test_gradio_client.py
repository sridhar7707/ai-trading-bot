"""
Gradio data-validation tests — ALL dashboard components.

Calls every render function the Gradio UI calls, seeding a real SQLite DB with
known values. Two tiers of coverage:

  1. Smoke tests (parametrized) — every render function: must return a result
     and must NOT produce a safe_render error card (contains "unavailable").

  2. Precise tests — Phase 1 + Phase 2 components: assert correct values from
     the seeded DB (dollar amounts, symbol names, win rates, etc.).

No server needed → runs in CI automatically via pytest tests/.
When the live Gradio server is on :7860 the two @live tests also run.

Update this file whenever a dashboard component's output format changes.
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("_BOT_LOG_HANDLER_ADDED", "1")

import bot.monitor.dashboard_data as dd
import dashboard.data as ddata
from bot.main import init_db

TODAY      = date.today().isoformat()
POOL_ALLOC = 5_000.0
POOL_CASH  = 3_200.0
POOL_INVST = 1_800.0
POOL_RESRV = 200.0
POOL_PNL   = 50.0


def _ts(hour: int) -> str:
    return f"{TODAY}T{hour:02d}:00:00+00:00"


# ── Fixture: seeded DB ────────────────────────────────────────────────────────

@pytest.fixture
def phase2_db(tmp_path, monkeypatch):
    """DB seeded with capital pool, daily actions, trades, and portfolio data."""
    db_path = str(tmp_path / "p2.db")
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()

    def _trade(ts, sym, action, shares, price, notional, pv, pnl,
               xgb=0.0, lstm=0.0, sent=0.0, macro=0.0, ens=0.0):
        con.execute(
            "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,"
            "portfolio_value,pnl_pct,xgb_prob,lstm_prob,sentiment_score,macro_score,"
            "ensemble_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, sym, action, shares, price, notional, "TRENDING_UP",
             pv, pnl, xgb, lstm, sent, macro, ens),
        )

    _trade(_ts(9),  "AAPL", "BUY",  5.0, 180.0,  900.0, 10_000.0, 0.0,   xgb=0.82, lstm=0.75, sent=0.60, macro=0.40, ens=0.70)
    _trade(_ts(10), "AAPL", "SELL", 5.0, 195.0,  975.0, 10_500.0, 0.083, xgb=0.82, lstm=0.75, sent=0.60, macro=0.40, ens=0.70)
    _trade(_ts(11), "MSFT", "BUY",  3.0, 300.0,  900.0, 10_500.0, 0.0,   xgb=0.65, lstm=0.60, sent=0.55, macro=0.40, ens=0.62)
    _trade(_ts(12), "MSFT", "SELL", 3.0, 285.0,  855.0, 10_350.0, -0.05, xgb=0.65, lstm=0.60, sent=0.55, macro=0.40, ens=0.62)
    _trade(_ts(13), "NVDA", "BUY",  2.0, 450.0,  900.0, 10_350.0, 0.0,   xgb=0.88, lstm=0.80, sent=0.70, macro=0.40, ens=0.80)
    _trade(_ts(14), "NVDA", "SELL", 2.0, 480.0,  960.0, 10_500.0, 0.067, xgb=0.88, lstm=0.80, sent=0.70, macro=0.40, ens=0.80)

    con.execute("INSERT INTO portfolio_snapshots VALUES (?,?,?,?)",
                (_ts(15), 10_500.0, 7_000.0, 1))

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

    for k, v in [("score", 0.40), ("cap", 1.0), ("halt", 0.0)]:
        con.execute("INSERT INTO macro_cache (key,value,cached_at) VALUES (?,?,?)",
                    (k, v, _ts(15)))

    # Capital pool (Phase 2A)
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

    # Daily actions (Phase 2B)
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

    monkeypatch.setattr(dd,    "_DB",        db_path)
    monkeypatch.setattr(dd,    "_HALT_FILE", Path(tmp_path / "NO_HALT"))
    monkeypatch.setattr(ddata, "DB_PATH",    db_path)
    ddata._CACHE.clear()
    ddata._CACHE_TS = 0.0
    monkeypatch.delenv("SPACE_ID", raising=False)
    return db_path


# ── Tier 1: Smoke tests — every render function ───────────────────────────────
# Tuple: (test_id, module, function_name, kwargs)
# Excluded: render_news_feed — makes live yfinance HTTP calls.
_SMOKE: list[tuple[str, str, str, dict]] = [
    # Brief tab
    ("executive_summary",        "dashboard.components.executive_summary",      "render_executive_summary",        {}),
    ("three_question_summary",   "dashboard.components.brief",                  "render_three_question_summary",   {}),
    ("morning_brief",            "dashboard.components.brief",                  "render_morning_brief",            {}),
    ("scheduler_status",         "dashboard.components.brief",                  "render_scheduler_status",         {}),
    ("decision_bar",             "dashboard.components.decision_bar",           "render_decision_bar",             {}),
    ("whats_changed",            "dashboard.components.history",                "render_whats_changed",            {}),
    ("market_mood",              "dashboard.components.market_mood",            "render_market_mood",              {}),
    ("ai_recommendation",        "dashboard.components.ai_panel",               "render_ai_recommendation",        {}),
    ("ai_committee",             "dashboard.components.ai_panel",               "render_ai_committee",             {}),
    ("risk_panel",               "dashboard.components.risk",                   "render_risk_panel",               {}),
    ("market_intelligence",      "dashboard.components.risk",                   "render_market_intelligence",      {}),
    ("news_feed_initial",        "dashboard.components.news",                   "render_news_feed_initial",        {}),
    ("all_timelines",            "dashboard.components.timeline",               "render_all_timelines",            {}),
    ("decision_timeline",        "dashboard.components.timeline",               "render_decision_timeline",        {"symbol": None}),
    # Portfolio tab
    ("weekly_summary",           "dashboard.components.weekly_summary",         "render_weekly_summary",           {}),
    ("daily_headline",           "dashboard.components.overview",               "render_daily_headline",           {}),
    ("portfolio_health_hero",    "dashboard.components.overview",               "render_portfolio_health_hero",    {}),
    ("spy_banner",               "dashboard.components.overview",               "render_spy_banner",               {}),
    ("benchmark_comparison",     "dashboard.components.overview",               "render_benchmark_comparison",     {}),
    ("portfolio_performance",    "dashboard.components.history",                "render_portfolio_performance",    {"period": "1M  —"}),
    ("positions",                "dashboard.components.portfolio",              "render_positions",                {}),
    ("trades",                   "dashboard.components.portfolio",              "render_trades",                   {}),
    ("watchlist",                "dashboard.components.signals",                "render_watchlist",                {}),
    ("timeline",                 "dashboard.components.signals",                "render_timeline",                 {}),
    ("symbol_detail",            "dashboard.components.symbol_detail",          "render_symbol_detail",            {"symbol": "AAPL"}),
    ("decision_center",          "dashboard.components.decision",               "render_decision_center",          {}),
    ("rebalance",                "dashboard.components.rebalance",              "render_rebalance",                {}),
    ("rebalance_suggestions",    "dashboard.components.rebalance",              "render_rebalance_suggestions",    {}),
    ("thesis_tracker",           "dashboard.components.thesis",                 "render_thesis_tracker",           {}),
    ("todays_actions",           "dashboard.components.actions",                "render_todays_actions",           {}),
    ("portfolio_actions",        "dashboard.components.actions",                "render_portfolio_actions",        {}),
    ("portfolio_simulator",      "dashboard.components.simulator",              "render_portfolio_simulator",      {"symbol": None}),
    # Capital tab
    ("capital_overview",         "dashboard.components.capital",                "render_capital_overview",         {}),
    ("managed_capital",          "dashboard.components.capital",                "render_managed_capital",          {}),
    ("profit_breakdown",         "dashboard.components.capital",                "render_profit_breakdown",         {}),
    # Trades tab
    ("top_picks",                "dashboard.components.recommendation_history", "render_top_picks",                {}),
    ("trade_frequency",          "dashboard.components.overview",               "render_trade_frequency",          {}),
    ("buy_candidates",           "dashboard.components.recommendation_history", "render_buy_candidates",           {}),
    ("signal_history",           "dashboard.components.signal_history",         "render_signal_history",           {}),
    ("recommendation_history",   "dashboard.components.recommendation_history", "render_recommendation_history",   {}),
    # Performance tab
    ("institutional_metrics",    "dashboard.components.models",                 "render_institutional_metrics",    {}),
    ("paper_trading_scorecard",  "dashboard.components.models",                 "render_paper_trading_scorecard",  {}),
    ("investor_view",            "dashboard.components.models",                 "render_investor_view",            {}),
    ("validation_report",        "dashboard.components.models",                 "render_validation_report",        {}),
    ("attribution_by_symbol",    "dashboard.components.attribution",            "render_attribution_by_symbol",    {}),
    ("attribution_by_sector",    "dashboard.components.attribution",            "render_attribution_by_sector",    {}),
    ("attribution_by_model",     "dashboard.components.attribution",            "render_attribution_by_model",     {}),
    ("attribution_by_trade",     "dashboard.components.attribution",            "render_attribution_by_trade",     {}),
    # Settings tab
    ("settings_summary",         "dashboard.components.settings",               "render_settings_summary",         {}),
    ("investor_profile",         "dashboard.components.settings",               "render_investor_profile",         {}),
    # Analysis / misc
    ("sell_analysis",            "dashboard.components.analysis",               "render_sell_analysis",            {}),
    ("position_sizing",          "dashboard.components.analysis",               "render_position_sizing",          {}),
    ("position_sizing_panel",    "dashboard.components.analysis",               "render_position_sizing_panel",    {}),
]


@pytest.mark.parametrize("tid,module,fn,kwargs", _SMOKE, ids=[s[0] for s in _SMOKE])
def test_component_renders_without_crash(phase2_db, tid, module, fn, kwargs):
    """Every render function must return a result and must not show a safe_render error card."""
    import plotly.graph_objects as go
    mod    = importlib.import_module(module)
    result = getattr(mod, fn)(**kwargs)
    assert result is not None, f"{fn} returned None"
    if isinstance(result, str):
        assert "unavailable" not in result, (
            f"{fn} returned a safe_render error card:\n{result[:300]}"
        )
    elif not isinstance(result, go.Figure):
        # unexpected type — just confirm it's truthy
        assert result, f"{fn} returned falsy result: {result!r}"


# ── Tier 2: Capital tab — precise value assertions ────────────────────────────

def test_capital_overview_shows_dollar_values(phase2_db):
    from dashboard.components.capital import render_capital_overview
    html = render_capital_overview()
    assert "$" in html
    assert "Deposit" in html or "Initial" in html


def test_capital_chart_returns_figure(phase2_db):
    from dashboard.components.capital import render_capital_chart
    import plotly.graph_objects as go
    assert isinstance(render_capital_chart(), go.Figure)


def test_profit_breakdown_shows_realized_and_unrealized(phase2_db):
    from dashboard.components.capital import render_profit_breakdown
    html = render_profit_breakdown()
    assert "Realized" in html
    assert "Unrealized" in html
    assert "$" in html


def test_managed_capital_heading_visible(phase2_db):
    from dashboard.components.capital import render_managed_capital
    assert "Capital Pool" in render_managed_capital()


def test_managed_capital_allocated_amount(phase2_db):
    from dashboard.components.capital import render_managed_capital
    html = render_managed_capital()
    assert "5,000" in html, f"Expected allocated $5,000:\n{html[:400]}"


def test_managed_capital_tradeable_cash(phase2_db):
    from dashboard.components.capital import render_managed_capital
    html = render_managed_capital()
    assert "Tradeable" in html
    assert "3,000" in html, f"Expected tradeable $3,000:\n{html[:400]}"


def test_managed_capital_invested_amount(phase2_db):
    from dashboard.components.capital import render_managed_capital
    assert "1,800" in render_managed_capital()


def test_managed_capital_shows_reserve(phase2_db):
    from dashboard.components.capital import render_managed_capital
    assert "Reserve" in render_managed_capital()


# ── Tier 2: Decision Workspace — precise assertions ───────────────────────────

def test_decision_bar_shows_pending_action(phase2_db):
    from dashboard.components.decision_bar import render_decision_bar
    html = render_decision_bar()
    assert "AAPL" in html or "pending" in html.lower() or "AI" in html


def test_daily_actions_get_pending_returns_correct_row(phase2_db):
    con = sqlite3.connect(phase2_db)
    from bot.decision.daily_actions import get_pending
    actions = get_pending(con)
    con.close()
    assert len(actions) == 1
    assert actions[0]["symbol"] == "AAPL"
    assert actions[0]["confidence"] == 82


def test_daily_actions_mark_executed_returns_true(phase2_db):
    con = sqlite3.connect(phase2_db)
    from bot.decision.daily_actions import mark_executed, get_pending
    action_id = get_pending(con)[0]["id"]
    assert mark_executed(con, action_id) is True
    assert get_pending(con) == []
    con.close()


def test_daily_actions_mark_executed_returns_false_on_missing_id(phase2_db):
    con = sqlite3.connect(phase2_db)
    from bot.decision.daily_actions import mark_executed
    assert mark_executed(con, 99999) is False
    con.close()


def test_daily_actions_get_pending_safe_on_fresh_db(tmp_path):
    from bot.decision.daily_actions import get_pending
    con = sqlite3.connect(str(tmp_path / "fresh.db"))
    assert get_pending(con) == []
    con.close()


# ── Tier 2: Attribution — precise assertions ──────────────────────────────────

def test_attribution_by_symbol_shows_traded_symbols(phase2_db):
    from dashboard.components.attribution import render_attribution_by_symbol
    html = render_attribution_by_symbol()
    assert "AAPL" in html
    assert "NVDA" in html


def test_attribution_by_symbol_shows_win_rate(phase2_db):
    from dashboard.components.attribution import render_attribution_by_symbol
    assert "%" in render_attribution_by_symbol()


def test_attribution_by_model_mentions_a_model(phase2_db):
    from dashboard.components.attribution import render_attribution_by_model
    html = render_attribution_by_model()
    assert any(m in html for m in ("XGB", "LSTM", "Sentiment", "Ensemble", "Macro"))


def test_attribution_by_trade_shows_best_and_worst(phase2_db):
    from dashboard.components.attribution import render_attribution_by_trade
    html = render_attribution_by_trade()
    assert ("AAPL" in html or "NVDA" in html), "Winner not in attribution"
    assert "MSFT" in html, "Loser not in attribution"


def test_attribution_by_trade_has_two_sections(phase2_db):
    from dashboard.components.attribution import render_attribution_by_trade
    html = render_attribution_by_trade()
    assert "Top" in html and ("Bottom" in html or "Worst" in html)


# ── Live-server tests (skipped when :7860 is not running) ─────────────────────

def _server_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:7860/info", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _server_up(), reason="Gradio server not running on :7860")
def test_live_server_info_endpoint():
    import urllib.request
    r = urllib.request.urlopen("http://localhost:7860/info", timeout=5)
    assert r.status == 200


@pytest.mark.skipif(not _server_up(), reason="Gradio server not running on :7860")
def test_live_gradio_client_connects():
    from gradio_client import Client
    assert Client("http://localhost:7860", verbose=False) is not None
