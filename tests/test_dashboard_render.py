"""Accuracy tests for the dashboard data layer.

Seeds a SQLite DB with KNOWN values, points dashboard_data at it, and asserts
every rendered surface shows the mathematically correct result. If we ever
display inaccurate data by accident (wrong field, stale source, broken math,
invisible text), one of these fails.

All math here is recomputed by hand from the seed so the expected values are
independent of the implementation.
"""
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

import bot.monitor.dashboard_data as dd
from bot.main import init_db
from config import DAILY_LOSS_LIMIT_PCT, WEEKLY_LOSS_LIMIT_PCT

# ── Known seed values (every expected number is derived from these) ────────────
DAILY_START   = 100_000.0
WEEKLY_START  =  80_000.0
SNAP_PORTF    = 102_000.0   # latest snapshot → the live portfolio value
SNAP_CASH     =  50_000.0
SNAP_OPEN     = 2
PORTFOLIO_HIGH = 105_000.0
MACRO_SCORE   = 0.42

TODAY = date.today().isoformat()
WEEK  = date.today().strftime("%G-W%V")

# Trade trajectory portfolio_values (ordered by timestamp) + the snapshot:
#   BUY  ZZZA pv=100000  (T10:00)
#   SELL ZZZB pv=101000  (T11:00)  pnl +0.05  → win
#   SELL ZZZC pv=100500  (T12:00)  pnl -0.03  → loss
#   SNAPSHOT  pv=102000  (T13:00)
# total_return = (102000 - 100000) / 100000 = +0.02
# closed_trades = 2 ; win_rate = 1/2 = 0.5
EXP_TOTAL_RETURN = (SNAP_PORTF - 100_000.0) / 100_000.0   # +0.02
EXP_DAY_PNL      = (SNAP_PORTF - DAILY_START)  / DAILY_START    # +0.02
EXP_WEEK_PNL     = (SNAP_PORTF - WEEKLY_START) / WEEKLY_START   # +0.275


def _ts(hour: int) -> str:
    return f"{TODAY}T{hour:02d}:00:00+00:00"


@pytest.fixture
def dash_db(tmp_path, monkeypatch):
    """Build a seeded DB with the production schema and point the dashboard at it."""
    db_path = str(tmp_path / "dash.db")
    # Build the exact production schema via init_db().
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()

    def _trade(ts, symbol, action, shares, price, notional, pv, pnl,
               xgb=0.0, lstm=0.0, sent=0.0, macro=0.0, ens=0.0, realized=0.0):
        con.execute(
            "INSERT INTO trades (timestamp,symbol,action,shares,price,notional,regime,"
            "portfolio_value,pnl_pct,xgb_prob,lstm_prob,sentiment_score,macro_score,"
            "ensemble_score,realized_pnl) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, symbol, action, shares, price, notional, "TRENDING_UP",
             pv, pnl, xgb, lstm, sent, macro, ens, realized),
        )

    _trade(_ts(10), "ZZZA", "BUY",  1.0, 150.0, 150.0, 100_000.0, 0.0,
           xgb=0.80, lstm=0.70, sent=0.50, macro=MACRO_SCORE, ens=0.65)
    _trade(_ts(11), "ZZZB", "SELL", 1.0, 160.0, 160.0, 101_000.0, 0.05, realized=10.0)
    _trade(_ts(12), "ZZZC", "SELL", 1.0,  90.0,  90.0, 100_500.0, -0.03, realized=-10.0)

    # positions
    con.execute("INSERT INTO position_state VALUES (?,?,?,?,?)",
                ("ZZZA", 150.0, 155.0, 2.5, _ts(10)))
    con.execute("INSERT INTO position_state VALUES (?,?,?,?,?)",
                ("ZZZD", 200.0, 205.0, 3.0, _ts(10)))

    # risk_state
    for k, v in [
        ("daily_start_value",  str(DAILY_START)),
        ("daily_start_date",   TODAY),
        ("weekly_start_value", str(WEEKLY_START)),
        ("weekly_start_week",  WEEK),
        ("day_trade_dates",    f'["{TODAY}", "{TODAY}"]'),
        ("portfolio_high",     str(PORTFOLIO_HIGH)),
        ("weekly_halt_alerted_week", ""),
    ]:
        con.execute("INSERT INTO risk_state (key,value,updated_at) VALUES (?,?,?)",
                    (k, v, _ts(13)))

    # macro_cache
    for k, v in [("score", MACRO_SCORE), ("cap", 1.0), ("halt", 0.0)]:
        con.execute("INSERT INTO macro_cache (key,value,cached_at) VALUES (?,?,?)",
                    (k, v, _ts(13)))

    # snapshot (latest timestamp → drives live portfolio value)
    con.execute("INSERT INTO portfolio_snapshots VALUES (?,?,?,?)",
                (_ts(13), SNAP_PORTF, SNAP_CASH, SNAP_OPEN))

    con.commit()
    con.close()

    monkeypatch.setattr(dd, "_DB", db_path)
    # Deterministic halt state: point at a file that does not exist.
    monkeypatch.setattr(dd, "_HALT_FILE", Path(tmp_path / "NO_HALT"))
    monkeypatch.delenv("SPACE_ID", raising=False)  # refresh_db_from_hf is a no-op
    return db_path


# ── Overview ──────────────────────────────────────────────────────────────────

def test_overview_portfolio_is_latest_snapshot(dash_db):
    d = dd.get_overview()
    assert d["portfolio"] == pytest.approx(SNAP_PORTF)


def test_overview_day_and_week_pnl(dash_db):
    d = dd.get_overview()
    assert d["day_pnl"]  == pytest.approx(EXP_DAY_PNL)    # +0.02
    assert d["week_pnl"] == pytest.approx(EXP_WEEK_PNL)   # +0.275


def test_overview_counts(dash_db):
    d = dd.get_overview()
    assert d["trades_today"]    == 3      # 3 trades seeded today
    assert d["open_positions"]  == 2      # 2 position_state rows
    assert d["day_trades_used"] == 2      # 2 entries in day_trade_dates for today


def test_overview_macro_and_halt_flags(dash_db):
    d = dd.get_overview()
    assert d["macro_score"] == pytest.approx(MACRO_SCORE)
    assert d["macro_halt"] is False
    assert d["daily_limit_hit"] is False   # +2% is a gain, not a loss
    assert d["weekly_limit_hit"] is False


def test_overview_md_shows_correct_strings(dash_db):
    md = dd.overview_md(dd.get_overview())
    assert "$102,000.00" in md       # portfolio
    assert "+2.00%" in md            # day P&L
    assert "+27.50%" in md           # week P&L
    assert "2/3" in md               # day trades used
    assert "0.42" in md              # macro score


def test_overview_dollar_pnl(dash_db):
    d = dd.get_overview()
    # day: 102000 - 100000 = +2000 ; week: 102000 - 80000 = +22000
    assert d["day_pnl_dollars"]  == pytest.approx(SNAP_PORTF - DAILY_START)
    assert d["week_pnl_dollars"] == pytest.approx(SNAP_PORTF - WEEKLY_START)
    md = dd.overview_md(d)
    assert "+$2,000.00 (+2.00%)" in md
    assert "+$22,000.00 (+27.50%)" in md


def test_overview_total_return_and_inception(dash_db):
    d = dd.get_overview()
    # inception = earliest portfolio_value (the BUY at 100000)
    assert d["total_return"] == pytest.approx((SNAP_PORTF - 100_000.0) / 100_000.0)
    assert d["inception_date"] is not None


def test_overview_provenance_line(dash_db):
    d = dd.get_overview()
    # fixture seeds 3 trades (1 BUY + 2 SELL)
    assert d["total_trades"] == 3
    line = dd._provenance_line(d)
    assert "Paper trading" in line
    assert "3 trades executed" in line
    assert "since" in line   # inception date present


def test_overview_vs_spy_beating(dash_db):
    d = dd.get_overview()
    d["spy_return"] = 0.012   # bot +2% beats SPY +1.2%
    md = dd.overview_md(d)
    assert "+2.00% vs S&P +1.20%" in md
    assert "beating the market" in md


def test_overview_vs_spy_absent_shows_inception(dash_db):
    d = dd.get_overview()
    d["spy_return"] = None
    assert "since inception" in dd.overview_md(d)


def test_spy_return_skipped_off_space(dash_db, monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert dd.spy_return_since("2026-01-01") is None  # no network off-Space


def test_money_formatting():
    assert dd._money(2000.0)  == "+$2,000.00"
    assert dd._money(-120.5)  == "-$120.50"
    assert dd._money(0.0)     == "+$0.00"


# ── Positions ─────────────────────────────────────────────────────────────────

def test_positions_df_has_both_symbols_and_prices(dash_db):
    df = dd.get_positions_df(prices={}, portfolio=102_000.0)
    assert len(df) == 2
    syms = set(df["Symbol"])
    assert {"ZZZA", "ZZZD"} == syms
    entries = dict(zip(df["Symbol"], df["Entry $"]))
    assert entries["ZZZA"] == pytest.approx(150.0)
    assert entries["ZZZD"] == pytest.approx(200.0)


def test_positions_shares_derived_from_trades(dash_db):
    # ZZZA has one BUY of 1.0 share; ZZZD has no trades → 0 shares.
    df = dd.get_positions_df(prices={}, portfolio=102_000.0)
    shares = dict(zip(df["Symbol"], df["Shares"]))
    assert shares["ZZZA"] == pytest.approx(1.0)
    assert shares["ZZZD"] == pytest.approx(0.0)


def test_positions_unrealized_pnl_math(dash_db):
    # ZZZA entry 150, current 165 → +10.00% ; value = 1 share * 165 = 165
    df = dd.get_positions_df(prices={"ZZZA": 165.0, "ZZZD": 220.0}, portfolio=102_000.0)
    za = df[df["Symbol"] == "ZZZA"].iloc[0]
    assert za["Current $"] == pytest.approx(165.0)
    assert za["Unrealized %"] == "+10.00%"
    assert za["Value $"] == pytest.approx(165.0)
    # % portfolio = 165 / 102000 = 0.16% → "0.2%"
    assert za["% Port"] == "0.2%"


def test_returns_summary_open_position_math(dash_db):
    # ZZZA: BUY 1 @150 (invested 150), open, current 165 → unrealized +15 → +10%
    df = dd.get_returns_summary_df(prices={"ZZZA": 165.0})
    za = df[df["Symbol"] == "ZZZA"].iloc[0]
    assert za["Invested $"] == pytest.approx(150.0)
    assert za["Return $"]  == pytest.approx(15.0)     # 1 * (165 - 150)
    assert za["Value $"]   == pytest.approx(165.0)    # invested + return
    assert za["Return %"]  == "+10.00%"
    assert "Open" in za["Status"]


def test_returns_summary_sold_position_uses_realized(dash_db):
    # ZZZB: one SELL with realized_pnl +10, fully exited → Sold, return = +10
    df = dd.get_returns_summary_df(prices={})
    zb = df[df["Symbol"] == "ZZZB"].iloc[0]
    assert zb["Return $"] == pytest.approx(10.0)
    assert "Sold" in zb["Status"]
    # ZZZC: realized -10 (a loss)
    zc = df[df["Symbol"] == "ZZZC"].iloc[0]
    assert zc["Return $"] == pytest.approx(-10.0)
    assert "Sold" in zc["Status"]


def test_returns_summary_open_without_price_shows_dash(dash_db):
    # Open position but no live price → return unknown, shown as "—" (no fake number)
    df = dd.get_returns_summary_df(prices={})
    za = df[df["Symbol"] == "ZZZA"].iloc[0]
    assert za["Return %"] == "—"
    assert za["Invested $"] == pytest.approx(150.0)   # invested is always known


def test_returns_summary_columns_and_empty(empty_schema_db):
    df = dd.get_returns_summary_df(prices={})
    assert list(df.columns) == dd._RETURNS_COLS
    assert len(df) == 0


def test_positions_price_unavailable_shows_dash(dash_db):
    # No prices (off-Space) → current/unrealized/%port show em-dash, no crash.
    df = dd.get_positions_df(prices={}, portfolio=102_000.0)
    za = df[df["Symbol"] == "ZZZA"].iloc[0]
    assert za["Current $"] == "—"
    assert za["Unrealized %"] == "—"


def test_live_prices_skipped_off_space(dash_db, monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert dd._live_prices(["ZZZA"]) == {}   # no network off-Space


def test_live_prices_prefers_alpaca_over_yfinance(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "x")
    monkeypatch.setattr(dd, "_prices_alpaca", lambda syms: {"ZZZA": 111.0})
    monkeypatch.setattr(dd, "_prices_yfinance", lambda syms: {"ZZZA": 999.0})
    assert dd._live_prices(["ZZZA"]) == {"ZZZA": 111.0}   # Alpaca wins


def test_live_prices_falls_back_to_yfinance(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "x")
    monkeypatch.setattr(dd, "_prices_alpaca", lambda syms: {})      # Alpaca unavailable
    monkeypatch.setattr(dd, "_prices_yfinance", lambda syms: {"ZZZA": 222.0})
    assert dd._live_prices(["ZZZA"]) == {"ZZZA": 222.0}


def test_alpaca_headers_none_without_creds(monkeypatch):
    monkeypatch.setattr(dd, "_alpaca_headers", dd._alpaca_headers)  # ensure real fn
    monkeypatch.setattr("config.ALPACA_KEY", "", raising=False)
    monkeypatch.setattr("config.ALPACA_SECRET", "", raising=False)
    monkeypatch.delenv("ALPACA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    assert dd._alpaca_headers() is None
    # With creds present, headers carry the Alpaca auth fields.
    monkeypatch.setenv("ALPACA_KEY", "k123")
    monkeypatch.setenv("ALPACA_SECRET", "s456")
    monkeypatch.setattr("config.ALPACA_KEY", "", raising=False)
    monkeypatch.setattr("config.ALPACA_SECRET", "", raising=False)
    h = dd._alpaca_headers()
    assert h["APCA-API-KEY-ID"] == "k123"
    assert h["APCA-API-SECRET-KEY"] == "s456"


def test_spy_return_prefers_alpaca(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "x")
    dd._spy_cache = {"key": None, "ret": None}
    monkeypatch.setattr(dd, "_spy_return_alpaca", lambda d: 0.05)
    monkeypatch.setattr(dd, "_spy_return_yfinance", lambda d: 0.99)
    assert dd.spy_return_since("2026-01-01") == 0.05


def test_spy_return_falls_back_to_yfinance(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "x")
    dd._spy_cache = {"key": None, "ret": None}
    monkeypatch.setattr(dd, "_spy_return_alpaca", lambda d: None)
    monkeypatch.setattr(dd, "_spy_return_yfinance", lambda d: 0.03)
    assert dd.spy_return_since("2026-01-01") == 0.03


# ── Trade Log (HTML) ──────────────────────────────────────────────────────────

def test_trades_html_shows_every_symbol(dash_db):
    html = dd.trades_html_table(30)
    for sym in ("ZZZA", "ZZZB", "ZZZC"):
        assert sym in html, f"{sym} missing from trade log"


def test_trades_html_symbol_is_visible_dark_text(dash_db):
    """Regression: symbol was rendered white (#fff) on the light theme → invisible."""
    html = dd.trades_html_table(30)
    # Symbol cell uses the dark primary-text color, not white.
    assert f"color:{dd._TEXT};font-weight:bold" in html
    assert "color:#fff;font-weight:bold" not in html  # the old invisible style


def test_trades_html_uses_shared_font(dash_db):
    html = dd.trades_html_table(30)
    assert dd._FONT in html


def test_get_trades_df_accurate(dash_db):
    df = dd.get_trades_df(30)
    assert len(df) == 3
    assert "ZZZA" in set(df["Symbol"])


def test_trade_log_buy_rationale_shows_drivers(dash_db):
    """BUY row 'Why' shows the model/regime drivers."""
    html = dd.trades_html_table(30)
    assert "XGB 0.80" in html
    assert "LSTM 0.70" in html
    assert "Trending Up" in html   # regime humanized


def test_trade_log_sell_rationale_is_plain_language(dash_db):
    html = dd.trades_html_table(30)
    assert "Signal exit" in html   # plain SELL → "Signal exit"


def test_trade_log_has_colorblind_glyphs(dash_db):
    html = dd.trades_html_table(30)
    assert "▲ BUY" in html         # up-triangle for buys
    assert "▼ SELL" in html        # down-triangle for sells


def test_trade_rationale_helper_maps_exit_reasons():
    import pandas as pd
    def _row(action, **kw):
        return pd.Series({"action": action, **kw})
    assert dd._trade_rationale(_row("SELL_STOP")) == "Stop-loss hit"
    assert dd._trade_rationale(_row("SELL_TAKE_PROFIT")) == "Took profit"
    assert dd._trade_rationale(_row("SELL_TRAILING_STOP")) == "Trailing stop"
    assert dd._trade_rationale(_row("SELL_GAP_DOWN")) == "Gap-down protection"
    # all-zero BUY (legacy row) → no fabricated rationale
    assert dd._trade_rationale(_row("BUY", xgb_prob=0, lstm_prob=0,
                                    sentiment_score=0, regime="")) == "—"


# ── Performance ───────────────────────────────────────────────────────────────

def test_performance_total_return_uses_snapshots(dash_db):
    """Total Return must reflect real account growth (snapshots), not just trades."""
    m = dd.get_performance_metrics(60)
    assert m["total_return"] == pytest.approx(EXP_TOTAL_RETURN, abs=1e-6)   # +0.02


def test_performance_win_rate_and_closed_count(dash_db):
    m = dd.get_performance_metrics(60)
    assert m["closed_trades"] == 2          # two SELL rows
    assert m["win_rate"] == pytest.approx(0.5)   # 1 win of 2
    assert m["trade_count"] == 3


def test_performance_sharpe_gated_until_enough_history(dash_db):
    m = dd.get_performance_metrics(60)
    assert m["sharpe"] is None              # only 3 returns < _MIN_SHARPE_OBS


def test_performance_max_drawdown_positive(dash_db):
    m = dd.get_performance_metrics(60)
    # peak 101000 → dip 100500 → dd = (101000-100500)/101000 ≈ 0.00495
    assert m["max_drawdown"] == pytest.approx((101_000 - 100_500) / 101_000, abs=1e-4)


def test_performance_md_renders_na_and_return(dash_db):
    md = dd.performance_md(dd.get_performance_metrics(60))
    assert "+2.00%" in md                          # total return
    assert "n/a (need more history)" in md         # sharpe
    assert "n/a (no closed trades yet)" not in md  # we DO have closed trades
    assert "50.0%" in md                           # win rate


# ── Compliance ────────────────────────────────────────────────────────────────

def test_compliance_state_accurate(dash_db):
    c = dd.get_compliance_state()
    assert c["portfolio"]      == pytest.approx(SNAP_PORTF)
    assert c["day_pnl_pct"]    == pytest.approx(EXP_DAY_PNL)
    assert c["week_pnl_pct"]   == pytest.approx(EXP_WEEK_PNL)
    assert c["day_trades_used"] == 2
    assert c["daily_limit_pct"]  == pytest.approx(DAILY_LOSS_LIMIT_PCT)
    assert c["weekly_limit_pct"] == pytest.approx(WEEKLY_LOSS_LIMIT_PCT)


def test_compliance_gauges_html_light_theme(dash_db):
    html = dd.compliance_gauges_html(dd.get_compliance_state())
    assert dd._BG in html      # light card background
    assert dd._FONT in html
    assert "#1a1a2e" not in html  # the old dark card


# ── Audit Trail ───────────────────────────────────────────────────────────────

def test_audit_df_has_signal_columns_and_values(dash_db):
    df = dd.get_audit_df(60)
    assert len(df) == 3
    for col in ("xgb_prob", "lstm_prob", "sentiment_score", "ensemble_score"):
        assert col in df.columns
    row = df[df["symbol"] == "ZZZA"].iloc[0]
    assert row["xgb_prob"] == pytest.approx(0.80)
    assert row["ensemble_score"] == pytest.approx(0.65)


# ── Charts (theme + no-exception) ─────────────────────────────────────────────

@pytest.mark.parametrize("fn", ["portfolio_chart", "signals_chart", "monthly_chart"])
def test_charts_render_light_theme(dash_db, fn):
    fig = getattr(dd, fn)()
    # White figure background (unified light theme).
    assert fig.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))


# ── Halt banner ───────────────────────────────────────────────────────────────

def test_halt_status_enabled_when_no_file(dash_db):
    html, btn = dd.halt_status_html()
    assert "TRADING ENABLED" in html
    assert "Activate Emergency Halt" in btn
    assert dd._FONT in html


def test_halt_status_active_when_file_present(dash_db, tmp_path, monkeypatch):
    halt = tmp_path / "HALT_NOW"
    halt.touch()
    monkeypatch.setattr(dd, "_HALT_FILE", halt)
    html, btn = dd.halt_status_html()
    assert "HALT ACTIVE" in html


# ── Empty-state guidance ──────────────────────────────────────────────────────

_HINT_FRAGMENT = "market open (9:30 AM ET"


@pytest.fixture
def missing_db(tmp_path, monkeypatch):
    """Point the dashboard at a non-existent DB so every tab hits its empty state."""
    monkeypatch.setattr(dd, "_DB", str(tmp_path / "does_not_exist.db"))
    monkeypatch.delenv("SPACE_ID", raising=False)
    return None


def test_overview_empty_shows_guidance(missing_db):
    md = dd.overview_md(dd.get_overview())
    assert _HINT_FRAGMENT in md


def test_trade_log_empty_shows_guidance(missing_db):
    assert _HINT_FRAGMENT in dd.trades_html_table(30)


def test_performance_empty_shows_guidance(missing_db):
    assert _HINT_FRAGMENT in dd.performance_md(dd.get_performance_metrics(60))


def test_compliance_empty_shows_guidance(missing_db):
    c = dd.get_compliance_state()
    assert _HINT_FRAGMENT in dd.compliance_md(c)
    assert _HINT_FRAGMENT in dd.compliance_gauges_html(c)


@pytest.fixture
def empty_schema_db(tmp_path, monkeypatch):
    """Real DB with the full schema but zero rows → df-empty branches."""
    db_path = str(tmp_path / "empty.db")
    monkeypatch.setattr("bot.main.TRADE_DB_PATH", db_path)
    con = init_db()
    con.close()
    monkeypatch.setattr(dd, "_DB", db_path)
    monkeypatch.delenv("SPACE_ID", raising=False)
    return db_path


def test_empty_window_trade_log_message(empty_schema_db):
    html = dd.trades_html_table(30)
    assert "No trades in the selected window" in html


def test_positions_empty_returns_typed_columns(empty_schema_db):
    df = dd.get_positions_df(prices={}, portfolio=0.0)
    assert list(df.columns) == dd._POSITION_COLS
    assert len(df) == 0
