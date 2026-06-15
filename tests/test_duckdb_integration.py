import datetime

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    """Redirect the DuckDB file to a temp directory so tests never touch the real DB."""
    import database.repositories.analytics_repository as repo_mod
    monkeypatch.setattr(repo_mod, "DB_PATH", tmp_path / "test_analytics.duckdb")


def _repo():
    from database.repositories.analytics_repository import AnalyticsRepository
    return AnalyticsRepository()


def _svc():
    from database.services.analytics_service import AnalyticsService
    return AnalyticsService()


def test_db_initializes():
    r = _repo()
    assert r is not None


def test_price_history_save_load():
    r = _repo()
    df = pd.DataFrame({
        "date": [datetime.date.today()],
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [103.0],
        "volume": [1_000_000],
    })
    assert r.save_price_history(df, "TEST") is True
    result = r.load_price_history("TEST", days=1)
    assert not result.empty
    assert float(result.iloc[0]["close"]) == 103.0


def test_portfolio_snapshot_save_load():
    r = _repo()
    snapshot = {
        "snapshot_date": datetime.date.today(),
        "portfolio_value": 50_000.0,
        "cash_balance": 5_000.0,
        "health_score": 75.0,
        "max_drawdown": -3.5,
        "sharpe_ratio": 1.2,
    }
    assert r.save_portfolio_snapshot(snapshot) is True
    result = r.load_snapshots(days=1)
    assert not result.empty
    assert float(result.iloc[0]["portfolio_value"]) == 50_000.0


def test_recommendation_save():
    r = _repo()
    assert r.save_recommendation("AAPL", "BUY", 0.85, price=150.0) is True


def test_sharpe_ratio():
    svc = _svc()
    returns = pd.Series([0.01, -0.005, 0.008, 0.012, -0.003])
    result = svc.get_sharpe_ratio(returns)
    assert isinstance(result, float)


def test_max_drawdown():
    svc = _svc()
    values = pd.Series([100.0, 110.0, 95.0, 105.0, 90.0])
    result = svc.get_max_drawdown(values)
    assert result < 0
    assert result > -100


def test_analytics_service_check_health():
    from database.services.analytics_service import AnalyticsService
    svc = AnalyticsService()
    health = svc.check_health()
    assert health["overall"] in ("ok", "degraded")
    assert health["duckdb_connection"] == "ok"
    assert health["sharpe_calculation"] == "ok"


def test_recommendation_history_change_detected():
    """save_recommendation() must record prev and detect a BUY→SELL change."""
    r = _repo()
    r.save_recommendation("TESTX", "BUY", 0.80, price=100.0)
    r.save_recommendation("TESTX", "SELL", 0.65, price=95.0)

    df = r.load_recommendation_history(symbol="TESTX", days=1, changes_only=True)
    assert len(df) >= 1
    row = df[df["symbol"] == "TESTX"].iloc[0]
    assert row["prev_recommendation"] == "BUY"
    assert row["recommendation"] == "SELL"


def test_benchmark_comparison_returns_dict():
    """get_benchmark_comparison() must return all required keys."""
    from database.services.analytics_service import AnalyticsService
    svc = AnalyticsService()
    result = svc.get_benchmark_comparison(10.0, "YTD")
    assert isinstance(result, dict)
    for key in ("portfolio_return", "spy_return", "qqq_return", "vs_spy", "vs_qqq"):
        assert key in result, f"Missing key: {key}"


def test_benchmark_spy_return_is_numeric():
    """Benchmark returns must be floats regardless of network availability."""
    from database.services.analytics_service import AnalyticsService
    svc = AnalyticsService()
    result = svc.get_benchmark_comparison(5.0, "1M")
    assert isinstance(result["spy_return"], float)
    assert isinstance(result["qqq_return"], float)
