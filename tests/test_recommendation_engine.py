"""Comprehensive tests for the three critical recommendation engine functions.

These functions drive real money decisions and must be tested against known inputs
with known expected outputs.
"""
from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures import (
    HEALTHY_PORTFOLIO,
    OVERSIZED_POSITION,
    ALL_CASH,
    ALL_INVESTED,
    HIGH_VIX,
    SINGLE_STOCK,
    LARGE_GAIN,
    LARGE_LOSS,
    HIGH_CONCENTRATION,
    TINY_PORTFOLIO,
)
from bot.core.recommendation_engine import (
    get_portfolio_action,
    get_sell_analysis,
    get_position_sizing,
    get_portfolio_health,
)


# ══════════════════════════════════════════════
# get_portfolio_action() tests
# ══════════════════════════════════════════════

class TestGetPortfolioAction:

    def test_returns_dict(self):
        result = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        assert "action"     in result
        assert "confidence" in result
        assert "reason"     in result
        assert "urgency"    in result

    def test_action_is_valid_value(self):
        result = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        assert result["action"] in (
            "BUY", "ADD", "HOLD", "TRIM", "SELL", "EXIT", "WATCH"
        )

    def test_urgency_is_valid_value(self):
        result = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        assert result["urgency"] in ("high", "medium", "low")

    def test_hold_signal_balanced_portfolio(self):
        """MU at ~15% of portfolio with good signal should not be TRIM or EXIT."""
        result = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        assert result["action"] in ("HOLD", "ADD", "WATCH"), (
            f"Expected HOLD/ADD/WATCH for balanced MU position but got {result['action']}"
        )

    def test_trim_signal_oversized_position(self):
        """MU at ~73% of portfolio should trigger TRIM or EXIT — never HOLD or ADD."""
        result = get_portfolio_action("MU", OVERSIZED_POSITION)
        assert result["action"] in ("TRIM", "EXIT", "SELL"), (
            f"Expected TRIM/EXIT for oversized MU but got {result['action']}"
        )

    def test_exit_signal_large_loss_low_confidence(self):
        """APLD at -50% loss with ensemble 0.42 should trigger EXIT or SELL."""
        result = get_portfolio_action("APLD", LARGE_LOSS)
        assert result["action"] in ("EXIT", "SELL", "TRIM"), (
            f"Expected EXIT/SELL for APLD at -50% loss but got {result['action']}"
        )

    def test_trim_signal_large_gain(self):
        """MU at +225% gain with large position should suggest at least TRIM."""
        result = get_portfolio_action("MU", LARGE_GAIN)
        assert result["action"] in ("TRIM", "EXIT"), (
            f"Expected TRIM/EXIT for MU at +225% gain but got {result['action']}"
        )

    def test_exit_urgency_is_high(self):
        """EXIT and SELL actions must have urgency=high."""
        result = get_portfolio_action("APLD", LARGE_LOSS)
        if result["action"] in ("EXIT", "SELL"):
            assert result["urgency"] == "high", (
                f"EXIT/SELL must have urgency=high but got {result['urgency']}"
            )

    def test_hold_urgency_is_low(self):
        """HOLD actions must have urgency=low."""
        result = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        if result["action"] == "HOLD":
            assert result["urgency"] == "low", (
                f"HOLD must have urgency=low but got {result['urgency']}"
            )

    def test_single_stock_triggers_trim(self):
        """100% in one stock must trigger TRIM or EXIT regardless of signal quality."""
        result = get_portfolio_action("MU", SINGLE_STOCK)
        assert result["action"] in ("TRIM", "EXIT", "SELL"), (
            f"Single stock portfolio must trigger TRIM/EXIT but got {result['action']}"
        )

    def test_unknown_symbol_returns_safe_fallback(self):
        """Symbol not in open_pos should return safe fallback not crash."""
        result = get_portfolio_action("FAKESYM", HEALTHY_PORTFOLIO)
        assert isinstance(result, dict)
        assert "action" in result

    def test_confidence_is_numeric(self):
        result = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 100

    def test_reason_is_non_empty_string(self):
        result = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


# ══════════════════════════════════════════════
# get_sell_analysis() tests
# ══════════════════════════════════════════════

class TestGetSellAnalysis:

    def test_returns_dict(self):
        result = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
        assert "sell_score"      in result
        assert "recommendation"  in result
        assert "reasons_to_sell" in result
        assert "reasons_to_hold" in result
        assert "stop_loss_pct"   in result

    def test_sell_score_range(self):
        result = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
        assert 0 <= result["sell_score"] <= 100

    def test_recommendation_is_valid(self):
        result = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
        assert result["recommendation"] in ("HOLD", "WATCH", "TRIM", "SELL", "EXIT")

    def test_hold_score_balanced_position(self):
        """Healthy balanced MU position should have low sell score (< 40) and HOLD/WATCH."""
        result = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
        assert result["sell_score"] < 40, (
            f"Balanced position should have sell_score < 40 but got {result['sell_score']}"
        )
        assert result["recommendation"] in ("HOLD", "WATCH")

    def test_high_score_oversized_position(self):
        """73% position should have sell_score > 60."""
        result = get_sell_analysis("MU", OVERSIZED_POSITION)
        assert result["sell_score"] > 60, (
            f"Oversized position should have sell_score > 60 but got {result['sell_score']}"
        )

    def test_exit_score_large_loss(self):
        """-50% loss with low AI conviction should have sell_score > 75 and EXIT/SELL."""
        result = get_sell_analysis("APLD", LARGE_LOSS)
        assert result["sell_score"] > 75, (
            f"Large loss with low confidence should score > 75 but got {result['sell_score']}"
        )
        assert result["recommendation"] in ("EXIT", "SELL")

    def test_trim_score_large_gain(self):
        """+225% gain should score high enough for at least TRIM recommendation."""
        result = get_sell_analysis("MU", LARGE_GAIN)
        assert result["recommendation"] in ("TRIM", "SELL", "EXIT"), (
            f"Large gain should trigger TRIM/SELL but got {result['recommendation']}"
        )

    def test_reasons_are_lists(self):
        result = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
        assert isinstance(result["reasons_to_sell"], list)
        assert isinstance(result["reasons_to_hold"], list)

    def test_reasons_are_strings(self):
        result = get_sell_analysis("APLD", LARGE_LOSS)
        for reason in result["reasons_to_sell"]:
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_score_consistency_with_recommendation(self):
        """Sell score and recommendation must agree."""
        result = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
        score = result["sell_score"]
        rec   = result["recommendation"]

        if score <= 25:
            assert rec in ("HOLD",), (
                f"Score {score} should be HOLD but got {rec}"
            )
        elif score <= 45:
            assert rec in ("WATCH", "HOLD"), (
                f"Score {score} should be WATCH but got {rec}"
            )
        elif score <= 65:
            assert rec in ("TRIM", "WATCH"), (
                f"Score {score} should be TRIM but got {rec}"
            )

    def test_stop_loss_pct_is_positive(self):
        result = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
        assert result["stop_loss_pct"] > 0
        assert result["stop_loss_pct"] <= 20

    def test_unknown_symbol_safe_fallback(self):
        result = get_sell_analysis("FAKESYM", HEALTHY_PORTFOLIO)
        assert isinstance(result, dict)
        assert "sell_score" in result
        assert result["sell_score"] == 0


# ══════════════════════════════════════════════
# get_position_sizing() edge case tests
# ══════════════════════════════════════════════

class TestGetPositionSizing:

    def test_returns_dict(self):
        result = get_position_sizing("MU", HEALTHY_PORTFOLIO)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_position_sizing("MU", HEALTHY_PORTFOLIO)
        assert "current_weight"  in result
        assert "target_weight"   in result
        assert "delta_weight"    in result
        assert "delta_dollars"   in result
        assert "action"          in result
        assert "dollar_display"  in result

    def test_all_cash_no_crash(self):
        """Empty portfolio should not crash. Symbol not held returns current_weight=0."""
        result = get_position_sizing("MU", ALL_CASH)
        assert isinstance(result, dict)
        assert result["current_weight"] == 0.0

    def test_oversized_recommends_reduce(self):
        """73% position should recommend reduce not add or hold."""
        result = get_position_sizing("MU", OVERSIZED_POSITION)
        assert result["action"] == "reduce", (
            f"73% position should action=reduce but got {result['action']}"
        )
        assert result["delta_dollars"] < 0, (
            f"Oversized should have negative delta_dollars but got {result['delta_dollars']}"
        )

    def test_tiny_portfolio_no_crash(self):
        """$500 portfolio should not crash or return nonsensical values."""
        result = get_position_sizing("MU", TINY_PORTFOLIO)
        assert isinstance(result, dict)
        assert result["current_weight"] <= 100.0

    def test_no_negative_weights(self):
        """Weights must never be negative."""
        for fixture_name, fixture in [
            ("HEALTHY",      HEALTHY_PORTFOLIO),
            ("OVERSIZED",    OVERSIZED_POSITION),
            ("ALL_INVESTED", ALL_INVESTED),
            ("SINGLE_STOCK", SINGLE_STOCK),
            ("TINY",         TINY_PORTFOLIO),
        ]:
            for sym in fixture["open_pos"]:
                result = get_position_sizing(sym, fixture)
                assert result["current_weight"] >= 0, (
                    f"{fixture_name}/{sym}: current_weight cannot be negative"
                )
                assert result["target_weight"] >= 0, (
                    f"{fixture_name}/{sym}: target_weight cannot be negative"
                )

    def test_target_never_exceeds_25_pct(self):
        """No single position should be targeted above 25% regardless of conviction."""
        for sym in HEALTHY_PORTFOLIO["open_pos"]:
            result = get_position_sizing(sym, HEALTHY_PORTFOLIO)
            assert result["target_weight"] <= 25.0, (
                f"{sym}: target_weight {result['target_weight']} exceeds 25% maximum"
            )

    def test_dollar_display_is_string(self):
        result = get_position_sizing("MU", HEALTHY_PORTFOLIO)
        assert isinstance(result["dollar_display"], str)
        assert len(result["dollar_display"]) > 0

    def test_all_invested_cash_warning(self):
        """When fully invested, function should still return without crashing."""
        result = get_position_sizing("NVDA", ALL_INVESTED)
        assert isinstance(result, dict)


# ══════════════════════════════════════════════
# get_portfolio_health() tests
# ══════════════════════════════════════════════

class TestGetPortfolioHealth:

    def test_returns_dict(self):
        result = get_portfolio_health(HEALTHY_PORTFOLIO)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_portfolio_health(HEALTHY_PORTFOLIO)
        assert "total"        in result
        assert "grade"        in result
        assert "components"   in result
        assert "biggest_risk" in result
        assert "strengths"    in result
        assert "weaknesses"   in result

    def test_score_range(self):
        for fixture in [
            HEALTHY_PORTFOLIO,
            OVERSIZED_POSITION,
            ALL_CASH,
            HIGH_VIX,
            SINGLE_STOCK,
        ]:
            result = get_portfolio_health(fixture)
            assert 0 <= result["total"] <= 100, (
                f"Health score out of range: {result['total']}"
            )

    def test_grade_is_valid(self):
        for fixture in [HEALTHY_PORTFOLIO, OVERSIZED_POSITION]:
            result = get_portfolio_health(fixture)
            assert result["grade"] in ("A", "B+", "B", "C", "D", "—")

    def test_healthy_portfolio_scores_well(self):
        """Balanced portfolio with low VIX should score above 70."""
        result = get_portfolio_health(HEALTHY_PORTFOLIO)
        assert result["total"] >= 70, (
            f"Healthy portfolio should score >= 70 but got {result['total']}"
        )

    def test_all_cash_scores_high_on_cash_component(self):
        """100% cash — should score high on cash component."""
        result = get_portfolio_health(ALL_CASH)
        cash_comp = result["components"].get("cash", {})
        assert cash_comp.get("score", 0) >= 15, (
            f"All cash should score high on cash component but got {cash_comp.get('score', 0)}"
        )

    def test_high_concentration_scores_low(self):
        """4 semiconductor stocks = 100% one sector. Diversification score should be low."""
        result = get_portfolio_health(HIGH_CONCENTRATION)
        div_comp = result["components"].get("diversification", {})
        assert div_comp.get("score", 25) < 15, (
            f"High concentration should score low on diversification but got {div_comp.get('score')}"
        )

    def test_high_vix_scores_low_on_risk(self):
        """VIX 32 should lower the risk component."""
        result = get_portfolio_health(HIGH_VIX)
        risk_comp = result["components"].get("risk", {})
        assert risk_comp.get("score", 25) <= 10, (
            f"VIX 32 should score <= 10 on risk component but got {risk_comp.get('score')}"
        )

    def test_single_stock_scores_low(self):
        """100% in one stock should score poorly overall."""
        result = get_portfolio_health(SINGLE_STOCK)
        assert result["total"] < 60, (
            f"Single stock portfolio should score < 60 but got {result['total']}"
        )

    def test_biggest_risk_is_string(self):
        result = get_portfolio_health(HEALTHY_PORTFOLIO)
        assert isinstance(result["biggest_risk"], str)

    def test_components_scores_sum_to_total(self):
        """Sum of component scores must equal the total score."""
        result = get_portfolio_health(HEALTHY_PORTFOLIO)
        comp_sum = sum(c.get("score", 0) for c in result["components"].values())
        assert comp_sum == result["total"], (
            f"Component scores {comp_sum} don't sum to total {result['total']}"
        )

    def test_grade_matches_score(self):
        """Grade must be consistent with score."""
        result = get_portfolio_health(HEALTHY_PORTFOLIO)
        score = result["total"]
        grade = result["grade"]

        if score >= 90:
            assert grade == "A"
        elif score >= 80:
            assert grade == "B+"
        elif score >= 70:
            assert grade == "B"
        elif score >= 60:
            assert grade == "C"
        else:
            assert grade == "D"

    def test_all_cash_no_crash(self):
        """Empty portfolio must not crash."""
        result = get_portfolio_health(ALL_CASH)
        assert isinstance(result, dict)
        assert "total" in result

    def test_tiny_portfolio_no_crash(self):
        """$500 portfolio must not crash."""
        result = get_portfolio_health(TINY_PORTFOLIO)
        assert isinstance(result, dict)

    def test_none_inputs_safe_fallback(self):
        """Passing empty dict must return safe fallback not raise exception."""
        result = get_portfolio_health({})
        assert isinstance(result, dict)
        assert "total" in result


# ══════════════════════════════════════════════
# Cross-function consistency tests
# ══════════════════════════════════════════════

class TestCrossConsistency:

    def test_exit_action_has_high_sell_score(self):
        """If get_portfolio_action returns EXIT, get_sell_analysis must agree."""
        action = get_portfolio_action("APLD", LARGE_LOSS)
        if action["action"] in ("EXIT", "SELL"):
            sell = get_sell_analysis("APLD", LARGE_LOSS)
            assert sell["sell_score"] > 45, (
                f"EXIT action but sell_score only {sell['sell_score']} — inconsistent"
            )

    def test_hold_action_has_low_sell_score(self):
        """If get_portfolio_action returns HOLD, get_sell_analysis should score < 55."""
        action = get_portfolio_action("MU", HEALTHY_PORTFOLIO)
        if action["action"] == "HOLD":
            sell = get_sell_analysis("MU", HEALTHY_PORTFOLIO)
            assert sell["sell_score"] < 55, (
                f"HOLD action but sell_score is {sell['sell_score']} — inconsistent"
            )

    def test_reduce_sizing_matches_trim_action(self):
        """If get_portfolio_action returns TRIM/EXIT, get_position_sizing must show reduce."""
        action = get_portfolio_action("MU", OVERSIZED_POSITION)
        if action["action"] in ("TRIM", "EXIT"):
            sizing = get_position_sizing("MU", OVERSIZED_POSITION)
            assert sizing["action"] == "reduce", (
                f"TRIM portfolio action but sizing action is {sizing['action']}"
            )
            assert sizing["delta_dollars"] < 0, (
                f"TRIM should have negative delta_dollars"
            )

    def test_oversized_detected_by_all_functions(self):
        """An oversized position should be flagged by ALL three functions."""
        action = get_portfolio_action("MU", OVERSIZED_POSITION)
        sell   = get_sell_analysis("MU", OVERSIZED_POSITION)
        sizing = get_position_sizing("MU", OVERSIZED_POSITION)

        assert action["action"] in ("TRIM", "EXIT", "SELL"), (
            f"portfolio_action missed oversized: {action}"
        )
        assert sell["sell_score"] > 50, (
            f"sell_analysis missed oversized: score={sell['sell_score']}"
        )
        assert sizing["action"] == "reduce", (
            f"position_sizing missed oversized: {sizing['action']}"
        )


# ══════════════════════════════════════════════
# Parametrized crash-safety tests
# ══════════════════════════════════════════════

@pytest.mark.parametrize("fixture,name", [
    (HEALTHY_PORTFOLIO,  "healthy"),
    (OVERSIZED_POSITION, "oversized"),
    (ALL_CASH,           "all_cash"),
    (ALL_INVESTED,       "all_invested"),
    (HIGH_VIX,           "high_vix"),
    (SINGLE_STOCK,       "single_stock"),
    (LARGE_GAIN,         "large_gain"),
    (LARGE_LOSS,         "large_loss"),
    (HIGH_CONCENTRATION, "high_concentration"),
    (TINY_PORTFOLIO,     "tiny"),
])
def test_portfolio_health_never_crashes(fixture, name):
    """Every fixture must return valid dict."""
    result = get_portfolio_health(fixture)
    assert isinstance(result, dict), f"{name}: get_portfolio_health crashed"
    assert 0 <= result.get("total", -1) <= 100, f"{name}: score out of range"


@pytest.mark.parametrize("fixture,sym,name", [
    (HEALTHY_PORTFOLIO,  "MU",   "healthy_MU"),
    (OVERSIZED_POSITION, "MU",   "oversized_MU"),
    (LARGE_GAIN,         "MU",   "gain_MU"),
    (LARGE_LOSS,         "APLD", "loss_APLD"),
    (SINGLE_STOCK,       "MU",   "single_MU"),
    (TINY_PORTFOLIO,     "MU",   "tiny_MU"),
])
def test_portfolio_action_never_crashes(fixture, sym, name):
    result = get_portfolio_action(sym, fixture)
    assert isinstance(result, dict), f"{name}: get_portfolio_action crashed"
    assert result.get("action") in (
        "BUY", "ADD", "HOLD", "TRIM", "SELL", "EXIT", "WATCH"
    ), f"{name}: invalid action {result.get('action')}"


@pytest.mark.parametrize("fixture,sym,name", [
    (HEALTHY_PORTFOLIO,  "MU",   "healthy_MU"),
    (OVERSIZED_POSITION, "MU",   "oversized_MU"),
    (LARGE_GAIN,         "MU",   "gain_MU"),
    (LARGE_LOSS,         "APLD", "loss_APLD"),
    (ALL_CASH,           "MU",   "cash_MU"),
    (TINY_PORTFOLIO,     "MU",   "tiny_MU"),
])
def test_sell_analysis_never_crashes(fixture, sym, name):
    result = get_sell_analysis(sym, fixture)
    assert isinstance(result, dict), f"{name}: get_sell_analysis crashed"
    assert 0 <= result.get("sell_score", -1) <= 100, f"{name}: sell_score out of range"
