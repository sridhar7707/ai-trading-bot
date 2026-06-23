def test_get_overview_no_db():
    from bot.monitor._dashboard_overview import get_overview
    result = get_overview()
    assert isinstance(result, dict)


def test_overview_md_error_key():
    from bot.monitor._dashboard_overview import overview_md
    result = overview_md({"_error": "test error"})
    assert "test error" in result


def test_overview_md_full_dict():
    from bot.monitor._dashboard_overview import overview_md
    d = {
        "portfolio": 10000.0, "day_pnl": 0.01, "week_pnl": 0.02,
        "day_pnl_dollars": 100.0, "week_pnl_dollars": 200.0,
        "total_return": 0.05, "inception_date": None,
        "total_trades": 10, "spy_return": None,
        "trades_today": 2, "open_positions": 3, "day_trades_used": 1,
        "macro_score": 0.6, "macro_halt": False, "emergency_halt": False,
        "daily_limit_hit": False, "weekly_limit_hit": False,
        "sync_ok": True, "sync_age_s": 60.0, "sync_err": "", "db_age_s": 120.0,
    }
    result = overview_md(d)
    assert "Portfolio Value" in result
    assert "ACTIVE" in result


def test_fmt_age():
    from bot.monitor._dashboard_overview import _fmt_age
    assert "s ago" in _fmt_age(30)
    assert "m ago" in _fmt_age(90)
    assert "h ago" in _fmt_age(7200)
    assert _fmt_age(None) == "unknown"
