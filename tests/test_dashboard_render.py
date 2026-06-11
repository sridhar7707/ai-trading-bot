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
    df = dd.get_positions_df()
    assert len(df) == 2
    syms = set(df["Symbol"])
    assert {"ZZZA", "ZZZD"} == syms
    entries = dict(zip(df["Symbol"], df["Entry $"]))
    assert entries["ZZZA"] == pytest.approx(150.0)
    assert entries["ZZZD"] == pytest.approx(200.0)


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
