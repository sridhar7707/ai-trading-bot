"""Centralized Gradio timer callback registration (req 7.1)."""
from __future__ import annotations

import gradio as gr

from dashboard.components.executive_summary import render_executive_summary
from dashboard.components.decision_bar import render_decision_bar
from dashboard.components.brief import (
    render_morning_brief, render_scheduler_status, render_three_question_summary,
)
from dashboard.components.overview import (
    render_daily_headline, render_portfolio_health_hero,
    render_trade_frequency, render_spy_banner,
)
from dashboard.components.market_mood import render_market_mood
from dashboard.components.ai_panel import render_ai_recommendation, render_ai_committee
from dashboard.components.risk import render_risk_panel, render_market_intelligence
from dashboard.components.portfolio import render_positions, render_trades
from dashboard.components.models import (
    render_validation_report, render_institutional_metrics,
    render_investor_view, render_paper_trading_scorecard,
)
from dashboard.components.signals import render_watchlist, render_timeline
from dashboard.components.history import render_whats_changed
from dashboard.components.recommendation_history import (
    render_recommendation_history, render_buy_candidates, render_top_picks,
)
from dashboard.components.news import render_news_feed
from dashboard.components.signal_history import render_signal_history
from dashboard.components.rebalance import render_rebalance
from dashboard.components.decision import render_decision_center
from dashboard.components.thesis import render_thesis_tracker
from dashboard.components.timeline import render_all_timelines
from dashboard.components.settings import render_settings_summary, render_investor_profile
from dashboard.components.capital import (
    render_capital_overview, render_capital_chart, render_profit_breakdown,
)
from dashboard.charts import (
    render_equity_chart, render_allocation_chart, render_pnl_chart,
    render_feature_importance_chart, render_returns_histogram, render_winloss_chart,
)


def register_all_timers(timer: gr.Timer, c: dict) -> None:
    """Register every timer.tick() callback. c maps name → gr.Component."""
    _brief(timer, c)
    _portfolio(timer, c)
    _capital(timer, c)
    _trades(timer, c)
    _performance(timer, c)
    _settings(timer, c)
    timer.tick(fn=render_executive_summary, outputs=c["exec_summary_out"])


def _brief(timer: gr.Timer, c: dict) -> None:
    timer.tick(fn=render_three_question_summary, outputs=c["three_q_out"])
    timer.tick(fn=render_decision_bar,           outputs=c["decision_bar_out"])
    timer.tick(fn=render_scheduler_status,       outputs=c["scheduler_status_out"])
    timer.tick(fn=render_morning_brief,          outputs=c["morning_brief_out"])
    timer.tick(fn=render_positions,              outputs=c["pos_brief_out"])
    timer.tick(fn=render_whats_changed,          outputs=c["whats_changed_out"])
    timer.tick(fn=render_market_mood,            outputs=c["market_mood_out"])
    timer.tick(fn=render_ai_recommendation,      outputs=c["ai_rec_brief_out"])
    timer.tick(fn=render_risk_panel,             outputs=c["risk_panel_out"])
    timer.tick(fn=render_market_intelligence,    outputs=c["mkt_intel_out"])
    timer.tick(fn=render_news_feed,              outputs=c["news_out"])
    timer.tick(fn=render_all_timelines,          outputs=c["timeline_brief_out"])


def _portfolio(timer: gr.Timer, c: dict) -> None:
    def _refresh_perf(current_key: str):
        from dashboard.components.history import _perf_choices, render_portfolio_performance
        if not isinstance(current_key, str):
            current_key = "1M"
        choices = _perf_choices()
        matched = next((ch for ch in choices if ch.split()[0] == current_key), None)
        val = matched or (choices[2] if len(choices) > 2 else choices[0] if choices else None)
        new_key = val.split()[0] if val else current_key
        return gr.update(choices=choices, value=val), new_key, render_portfolio_performance(val or "1M")

    def _refresh_sym(sel: str):
        from dashboard.components.symbol_detail import _get_symbol_choices
        choices = _get_symbol_choices()
        val = sel if sel in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=val), val

    def _sym_detail(sel):
        from dashboard.components.symbol_detail import render_symbol_detail
        return render_symbol_detail(sel)

    def _sim_choices():
        from dashboard.data import get_data as _gd
        choices = sorted(_gd().get("prices", {}).keys()) or []
        return gr.update(choices=choices)

    timer.tick(fn=_refresh_perf, inputs=[c["perf_key_state"]],
               outputs=[c["perf_tabs"], c["perf_key_state"], c["perf_out"]])
    timer.tick(fn=render_daily_headline,         outputs=c["daily_headline_out"])
    timer.tick(fn=render_portfolio_health_hero,  outputs=c["hero_out"])
    timer.tick(fn=render_spy_banner,             outputs=c["spy_banner_out"])
    timer.tick(fn=render_equity_chart,           outputs=c["eq_plot"])
    timer.tick(fn=render_allocation_chart,       outputs=c["alloc_plot"])
    timer.tick(fn=render_pnl_chart,              outputs=c["pnl_plot"])
    timer.tick(fn=render_ai_committee,           outputs=c["committee_out"])
    timer.tick(fn=render_decision_center,        outputs=c["decision_center_out"])
    timer.tick(fn=render_rebalance,              outputs=c["rebalance_out"])
    timer.tick(fn=render_watchlist,              outputs=c["watchlist_out"])
    timer.tick(fn=render_positions,              outputs=c["pos_out"])
    timer.tick(fn=render_trades,                 outputs=c["trades_out"])
    timer.tick(fn=render_thesis_tracker,         outputs=c["thesis_out"])
    timer.tick(fn=_refresh_sym, inputs=[c["_sym_state"]],
               outputs=[c["symbol_selector"], c["_sym_state"]])
    timer.tick(fn=_sym_detail,  inputs=[c["_sym_state"]], outputs=[c["symbol_detail_out"]])
    timer.tick(fn=_sim_choices,                  outputs=c["sim_sym_dd"])


def _capital(timer: gr.Timer, c: dict) -> None:
    timer.tick(fn=render_capital_overview,  outputs=c["capital_overview_out"])
    timer.tick(fn=render_capital_chart,     outputs=c["capital_chart_out"])
    timer.tick(fn=render_profit_breakdown,  outputs=c["profit_breakdown_out"])


def _trades(timer: gr.Timer, c: dict) -> None:
    timer.tick(fn=render_top_picks,              outputs=c["top_picks_out"])
    timer.tick(fn=render_trade_frequency,        outputs=c["trade_freq_out"])
    timer.tick(fn=render_buy_candidates,         outputs=c["buy_candidates_out"])
    timer.tick(fn=render_signal_history,         outputs=c["signal_history_out"])
    timer.tick(fn=render_recommendation_history, outputs=c["rec_history_out"])
    timer.tick(fn=render_timeline,               outputs=c["timeline_trades_out"])


def _performance(timer: gr.Timer, c: dict) -> None:
    timer.tick(fn=render_paper_trading_scorecard,  outputs=c["scorecard_out"])
    timer.tick(fn=render_institutional_metrics,    outputs=c["metrics_out"])
    timer.tick(fn=render_returns_histogram,        outputs=c["returns_hist_plot"])
    timer.tick(fn=render_winloss_chart,            outputs=c["winloss_plot"])
    timer.tick(fn=render_investor_view,            outputs=c["investor_out"])
    timer.tick(fn=render_feature_importance_chart, outputs=c["fi_plot"])
    timer.tick(fn=render_validation_report,        outputs=c["val_out"])


def _settings(timer: gr.Timer, c: dict) -> None:
    timer.tick(fn=render_settings_summary,  outputs=c["settings_summary_out"])
    timer.tick(fn=render_investor_profile,  outputs=c["investor_profile_out"])
