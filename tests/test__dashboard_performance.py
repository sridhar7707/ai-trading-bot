def test_get_performance_metrics_no_db():
    from bot.monitor._dashboard_performance import get_performance_metrics
    result = get_performance_metrics(days=30)
    assert isinstance(result, dict)


def test_performance_md_empty():
    from bot.monitor._dashboard_performance import performance_md
    result = performance_md({})
    assert isinstance(result, str)
    assert len(result) > 0


def test_performance_md_with_data():
    from bot.monitor._dashboard_performance import performance_md
    m = {
        "sharpe": 1.5, "sortino": 2.0, "win_rate": 0.60,
        "avg_win": 0.03, "avg_loss": -0.015, "max_drawdown": 0.08,
        "calmar": 1.2, "alpha": 0.05, "total_return": 0.12,
        "trade_count": 50, "closed_trades": 30,
    }
    result = performance_md(m)
    assert "Sharpe" in result
    assert "60.0%" in result
