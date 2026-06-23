import pandas as pd


def test_empty_signals_df_columns():
    from bot.monitor._dashboard_signals import _empty_signals_df, _SIGNAL_COLS
    df = _empty_signals_df()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == _SIGNAL_COLS


def test_get_latest_signals_df_no_db():
    from bot.monitor._dashboard_signals import get_latest_signals_df
    result = get_latest_signals_df()
    assert isinstance(result, pd.DataFrame)


def test_empty_screener_df_columns():
    from bot.monitor._dashboard_signals import _empty_screener_df, _SCREENER_COLS
    df = _empty_screener_df()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == _SCREENER_COLS


def test_get_screener_df_no_db():
    from bot.monitor._dashboard_signals import get_screener_df
    result = get_screener_df()
    assert isinstance(result, pd.DataFrame)
