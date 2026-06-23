import pandas as pd


def test_get_positions_df_no_db():
    from bot.monitor._dashboard_positions import get_positions_df
    result = get_positions_df()
    assert isinstance(result, pd.DataFrame)


def test_get_returns_summary_df_no_db():
    from bot.monitor._dashboard_positions import get_returns_summary_df
    result = get_returns_summary_df()
    assert isinstance(result, pd.DataFrame)


def test_get_trades_df_no_db():
    from bot.monitor._dashboard_positions import get_trades_df
    result = get_trades_df(days=7)
    assert isinstance(result, pd.DataFrame)


def test_live_prices_off_space():
    from bot.monitor._dashboard_positions import _live_prices
    result = _live_prices(["AAPL", "MSFT"])
    assert isinstance(result, dict)
