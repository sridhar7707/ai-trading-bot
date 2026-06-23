from bot.core.recommendation_portfolio import get_portfolio_health, _portfolio_val, _cash_pct


def test_get_portfolio_health_empty_input():
    result = get_portfolio_health({})
    assert isinstance(result, dict)
    assert "total" in result
    assert "grade" in result


def test_get_portfolio_health_score_range():
    result = get_portfolio_health({"portfolio": "$10,000", "vix": 15.0})
    assert 0 <= result["total"] <= 100


def test_get_portfolio_health_grade_label():
    result = get_portfolio_health({})
    assert result["grade"] in ("A", "B+", "B", "C", "D")
    assert isinstance(result["grade_label"], str)


def test_portfolio_val_parsing():
    assert _portfolio_val({"portfolio": "$10,000.00"}) == 10000.0
    assert _portfolio_val({"portfolio": "—"}) == 0.0
    assert _portfolio_val({}) == 0.0


def test_cash_pct_no_positions():
    pct = _cash_pct({"open_pos": {}, "prices": {}}, 5000.0)
    assert pct == 100.0
