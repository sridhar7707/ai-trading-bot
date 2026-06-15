"""Tests for SPEC 53 view models and builder functions."""
from __future__ import annotations

import sys
import os
import types
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers to patch get_data ─────────────────────────────────────────────────

def _empty_data():
    return {
        "open_pos": {}, "prices": {}, "trades_df": pd.DataFrame(),
        "portfolio": "—", "regime_raw": "Unknown",
        "total_trades": 0, "buy_count": 0, "sell_count": 0, "win_count": 0,
        "recent_trades": [],
        "vix": 0.0, "avg_confidence": 0.0, "sentiment_avg": 0.0,
        "latest_buy_signal": {}, "today_buy_signals": [],
    }


def _pos_data(symbols=None):
    d = _empty_data()
    d["portfolio"] = "$10000.00"
    d["prices"]    = {s: 100.0 for s in (symbols or ["AAPL"])}
    d["open_pos"]  = {
        s: {"shares": 10.0, "invested": 900.0}
        for s in (symbols or ["AAPL"])
    }
    return d


def _signal_data(symbol="AAPL"):
    d = _pos_data([symbol])
    d["latest_buy_signal"] = {
        "symbol":         symbol,
        "ensemble_score": 0.78,
        "xgb_prob":       0.80,
        "lstm_prob":      0.75,
        "sentiment_score": 0.10,
        "price":          100.0,
        "regime":         "trending_up",
        "timestamp":      "2026-06-14T10:00:00",
        "feature_drivers": None,
    }
    d["vix"] = 14.0
    return d


def _trades_data():
    d = _pos_data()
    d["recent_trades"] = [
        ("2026-06-14T10:00:00", "AAPL", "BUY",  10.0, 95.0, 950.0, None,   "trending_up"),
        ("2026-06-14T14:00:00", "AAPL", "SELL", 10.0, 105.0, 1050.0, 0.105, "trending_up"),
    ]
    d["total_trades"] = 2
    return d


# ── Stub recommendation engine responses ──────────────────────────────────────

def _stub_pa(sym, d):
    return {"action": "HOLD", "confidence": 70, "reason": "Stable", "urgency": "low"}


def _stub_sz(sym, d):
    return {
        "target_weight": 10.0, "current_weight": 9.0, "delta_weight": 1.0,
        "dollar_display": "+$100", "action": "add", "delta_dollars": 100.0,
    }


def _stub_sa(sym, d):
    return {
        "sell_score": 30,
        "reasons_to_sell": [],
        "reasons_to_hold": ["Momentum positive"],
    }


def _stub_health(d):
    return {
        "total": 75, "grade": "B+", "grade_label": "Good",
        "biggest_risk": "VIX elevated",
        "components": {
            "risk": {"label": "Risk", "score": 20, "max": 25, "detail": "low"},
            "diversification": {"label": "Diversification", "score": 15, "max": 25, "detail": "ok"},
            "cash": {"label": "Cash", "score": 15, "max": 25, "detail": "ok"},
            "momentum": {"label": "Momentum", "score": 15, "max": 25, "detail": "ok"},
            "quality": {"label": "Quality", "score": 10, "max": 25, "detail": "ok"},
        },
        "strengths": ["Low VIX", "Good cash"],
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data",             lambda: _pos_data())
    monkeypatch.setattr(bld, "get_portfolio_action", _stub_pa)
    monkeypatch.setattr(bld, "get_position_sizing",  _stub_sz)
    monkeypatch.setattr(bld, "get_sell_analysis",    _stub_sa)
    monkeypatch.setattr(bld, "get_portfolio_health", _stub_health)


# ── PositionRow tests ─────────────────────────────────────────────────────────

def test_build_positions_returns_list():
    from dashboard.builders import build_positions_vm
    rows = build_positions_vm()
    assert isinstance(rows, list)


def test_build_positions_empty_when_no_positions(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _empty_data())
    from dashboard.builders import build_positions_vm
    assert build_positions_vm() == []


def test_build_positions_fields(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _pos_data(["AAPL"]))
    from dashboard.builders import build_positions_vm
    from dashboard.viewmodels import PositionRow
    rows = build_positions_vm()
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, PositionRow)
    assert r.symbol == "AAPL"
    assert r.action == "HOLD"
    assert r.confidence == 70
    assert isinstance(r.pnl_color, str) and r.pnl_color.startswith("#")


def test_build_positions_pnl_color_gain(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _pos_data(["AAPL"]))
    from dashboard.builders import build_positions_vm
    from dashboard.design_system import GAIN
    rows = build_positions_vm()
    assert rows[0].pnl_color == GAIN  # invested 900, value 1000 → positive


# ── TradeRow tests ────────────────────────────────────────────────────────────

def test_build_trades_empty_when_no_trades(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _empty_data())
    from dashboard.builders import build_trades_vm
    assert build_trades_vm() == []


def test_build_trades_pnl_pct_none_for_buy(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _trades_data())
    from dashboard.builders import build_trades_vm
    rows = build_trades_vm()
    buy_row = next((r for r in rows if r.action == "BUY"), None)
    assert buy_row is not None
    assert buy_row.pnl_pct is None


def test_build_trades_pnl_pct_float_for_sell(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _trades_data())
    from dashboard.builders import build_trades_vm
    rows = build_trades_vm()
    sell_row = next((r for r in rows if r.action == "SELL"), None)
    assert sell_row is not None
    assert isinstance(sell_row.pnl_pct, float)


# ── HealthViewModel tests ─────────────────────────────────────────────────────

def test_build_health_vm_returns_viewmodel():
    from dashboard.builders import build_health_vm
    from dashboard.viewmodels import HealthViewModel
    vm = build_health_vm()
    assert isinstance(vm, HealthViewModel)


def test_build_health_vm_fields():
    from dashboard.builders import build_health_vm
    vm = build_health_vm()
    assert vm.total == 75
    assert vm.grade == "B+"
    assert vm.grade_label == "Good"
    assert "VIX" in vm.biggest_risk


def test_build_health_vm_components():
    from dashboard.builders import build_health_vm
    from dashboard.viewmodels import HealthComponent
    vm = build_health_vm()
    assert len(vm.components) == 5
    assert all(isinstance(c, HealthComponent) for c in vm.components)
    labels = [c.label for c in vm.components]
    assert "Risk" in labels


def test_build_health_vm_strengths():
    from dashboard.builders import build_health_vm
    vm = build_health_vm()
    assert isinstance(vm.strengths, list)
    assert len(vm.strengths) <= 2


# ── ActionRow tests ───────────────────────────────────────────────────────────

def test_build_actions_empty_when_no_positions(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _empty_data())
    from dashboard.builders import build_actions_vm
    assert build_actions_vm() == []


def test_build_actions_fields(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _pos_data(["AAPL"]))
    from dashboard.builders import build_actions_vm
    from dashboard.viewmodels import ActionRow
    rows = build_actions_vm()
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, ActionRow)
    assert r.symbol == "AAPL"
    assert r.action == "HOLD"
    assert r.confidence == 70
    assert r.urgency == "low"


# ── DecisionRow tests ─────────────────────────────────────────────────────────

def test_build_decision_vm_fields(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _pos_data(["AAPL"]))
    from dashboard.builders import build_decision_vm
    from dashboard.viewmodels import DecisionRow
    rows = build_decision_vm()
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, DecisionRow)
    assert r.symbol == "AAPL"
    assert r.sell_score == 30
    assert r.pa_reason == "Stable"


# ── RebalanceRow tests ────────────────────────────────────────────────────────

def test_build_rebalance_vm_fields(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _pos_data(["AAPL"]))
    from dashboard.builders import build_rebalance_vm
    from dashboard.viewmodels import RebalanceRow
    rows = build_rebalance_vm()
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, RebalanceRow)
    assert r.symbol == "AAPL"
    assert isinstance(r.delta_dollars, float)


# ── CommitteeViewModel tests ──────────────────────────────────────────────────

def test_build_committee_vm_no_data_when_no_signal(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _pos_data(["AAPL"]))
    from dashboard.builders import build_committee_vm
    vm = build_committee_vm("AAPL")
    assert vm.no_data is True


def test_build_committee_vm_five_members(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _signal_data("AAPL"))
    from dashboard.builders import build_committee_vm
    vm = build_committee_vm("AAPL")
    assert vm.no_data is False
    assert len(vm.members) == 5


def test_build_committee_vm_member_names(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _signal_data("AAPL"))
    from dashboard.builders import build_committee_vm
    vm = build_committee_vm("AAPL")
    names = [m.name for m in vm.members]
    assert "XGBoost" in names
    assert "LSTM" in names
    assert "Sentiment" in names
    assert "Regime" in names
    assert "Macro" in names


def test_build_committees_vm_three_members(monkeypatch):
    """build_committees_vm returns 3-member committees from trades_df."""
    import dashboard.builders as bld

    df = pd.DataFrame([{
        "action":          "BUY",
        "symbol":          "AAPL",
        "xgb_prob":        0.80,
        "lstm_prob":       0.75,
        "sentiment_score": 0.10,
    }])
    d = _pos_data(["AAPL"])
    d["trades_df"] = df
    monkeypatch.setattr(bld, "get_data", lambda: d)
    from dashboard.builders import build_committees_vm
    vms = build_committees_vm()
    assert len(vms) == 1
    assert len(vms[0].members) == 3


def test_build_committees_vm_no_data_when_no_trades(monkeypatch):
    import dashboard.builders as bld
    monkeypatch.setattr(bld, "get_data", lambda: _pos_data(["AAPL"]))
    from dashboard.builders import build_committees_vm
    vms = build_committees_vm()
    assert len(vms) == 1
    assert vms[0].no_data is True
