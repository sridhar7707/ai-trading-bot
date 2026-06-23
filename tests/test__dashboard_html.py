def test_compliance_gauges_html_empty_input():
    from bot.monitor._dashboard_html import compliance_gauges_html
    result = compliance_gauges_html({})
    assert isinstance(result, str)
    assert len(result) > 0


def test_compliance_gauges_html_with_data():
    from bot.monitor._dashboard_html import compliance_gauges_html
    c = {
        "day_pnl_pct": -0.01, "week_pnl_pct": -0.02,
        "daily_limit_pct": 0.05, "weekly_limit_pct": 0.10,
        "day_trades_used": 1, "day_trades_limit": 3,
        "daily_warning_sent": False, "weekly_halt_alerted": False,
    }
    html = compliance_gauges_html(c)
    assert "Risk Limit Gauges" in html
    assert "Daily Loss" in html


def test_trades_html_table_no_db():
    from bot.monitor._dashboard_html import trades_html_table
    result = trades_html_table(days=7)
    assert isinstance(result, str)


def test_trade_rationale_buy():
    from bot.monitor._dashboard_html import _trade_rationale
    import pandas as pd
    row = pd.Series({"action": "BUY", "xgb_prob": 0.8, "lstm_prob": 0.7,
                     "sentiment_score": 0.2, "regime": "TRENDING_UP", "feature_drivers": None})
    result = _trade_rationale(row)
    assert "buy" in result.lower() or "signal" in result.lower()


def test_trade_rationale_sell():
    from bot.monitor._dashboard_html import _trade_rationale
    import pandas as pd
    row = pd.Series({"action": "SELL_STOP", "xgb_prob": None, "lstm_prob": None,
                     "sentiment_score": None, "regime": None, "feature_drivers": None})
    assert _trade_rationale(row) == "Stop-loss hit"
