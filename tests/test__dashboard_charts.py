import matplotlib.pyplot as plt
import pytest


def test_portfolio_chart_returns_figure_without_db():
    from bot.monitor._dashboard_charts import portfolio_chart
    fig = portfolio_chart(days=7)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_signals_chart_returns_figure_without_db():
    from bot.monitor._dashboard_charts import signals_chart
    fig = signals_chart(days=7)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_monthly_chart_returns_figure_without_db():
    from bot.monitor._dashboard_charts import monthly_chart
    fig = monthly_chart(days=30)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
