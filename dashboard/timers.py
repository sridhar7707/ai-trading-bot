"""Centralized Gradio timer callback registration (req 7.1).

Two timers keep the UI responsive without hammering external APIs:
  timer_ui   (60 s)  — lightweight DB reads only; no yfinance calls
  timer_data (300 s) — heavy: yfinance (15 s daemon-thread timeout), charts, AI; batched + stateful

Batching all callbacks into a single timer.tick() per timer prevents Gradio 5
from firing N separate sequential SSE events (one per registration), which
causes components to show loading indicators one-by-one and looks like a
continuous page refresh on slow servers.
"""
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
from dashboard.components.weekly_summary import render_weekly_summary
from dashboard.components.settings import render_settings_summary, render_investor_profile
from dashboard.components.capital import (
    render_capital_overview, render_capital_chart, render_profit_breakdown,
)
from dashboard.charts import (
    render_equity_chart, render_allocation_chart, render_pnl_chart,
    render_feature_importance_chart, render_returns_histogram, render_winloss_chart,
)


def register_all_timers(
    timer_ui: gr.Timer,
    timer_data: gr.Timer,
    c: dict,
) -> None:
    """Register batched timer.tick() callbacks.

    Each timer uses one batched callback (returning a tuple) instead of N
    separate timer.tick() registrations.  Gradio 5 fires each registration as
    its own sequential SSE event; batching collapses N events into 1, so the
    client sees one combined loading state instead of N rapid sequential ones.
    """
    _register_ui_tick(timer_ui, c)
    _register_data_tick(timer_data, c)


# ── Fast (60 s) — DB reads only, no external API calls ────────────────────────

def _register_ui_tick(timer: gr.Timer, c: dict) -> None:
    """One batched tick for all lightweight components (DB reads only)."""

    def _tick():
        return (
            render_executive_summary(),
            render_decision_bar(),
            render_scheduler_status(),
            render_morning_brief(),
            render_positions(),           # → pos_brief_out
            render_daily_headline(),
            render_positions(),           # → pos_out
            render_weekly_summary(),
            render_capital_overview(),
            render_profit_breakdown(),
            render_settings_summary(),
            render_investor_profile(),
        )

    timer.tick(fn=_tick, outputs=[
        c["exec_summary_out"],
        c["decision_bar_out"],
        c["scheduler_status_out"],
        c["morning_brief_out"],
        c["pos_brief_out"],
        c["daily_headline_out"],
        c["pos_out"],
        c["weekly_summary_out"],
        c["capital_overview_out"],
        c["profit_breakdown_out"],
        c["settings_summary_out"],
        c["investor_profile_out"],
    ])


# ── Slow (300 s) — yfinance (15 s timeout) + charts + AI ──────────────────────

def _register_data_tick(timer: gr.Timer, c: dict) -> None:
    """One batched tick for all heavy renders, plus stateful callbacks."""

    def _tick():
        return (
            render_three_question_summary(),
            render_whats_changed(),
            render_market_mood(),
            render_ai_recommendation(),
            render_risk_panel(),
            render_market_intelligence(),
            render_news_feed(),
            render_all_timelines(),
            render_equity_chart(),
            render_allocation_chart(),
            render_pnl_chart(),
            render_ai_committee(),
            render_decision_center(),
            render_rebalance(),
            render_watchlist(),
            render_trades(),
            render_thesis_tracker(),
            render_capital_chart(),
            render_top_picks(),
            render_trade_frequency(),
            render_buy_candidates(),
            render_signal_history(),
            render_recommendation_history(),
            render_timeline(),
            render_paper_trading_scorecard(),
            render_institutional_metrics(),
            render_returns_histogram(),
            render_winloss_chart(),
            render_investor_view(),
            render_feature_importance_chart(),
            render_validation_report(),
            render_portfolio_health_hero(),
            render_spy_banner(),
        )

    timer.tick(fn=_tick, outputs=[
        c["three_q_out"],
        c["whats_changed_out"],
        c["market_mood_out"],
        c["ai_rec_brief_out"],
        c["risk_panel_out"],
        c["mkt_intel_out"],
        c["news_out"],
        c["timeline_brief_out"],
        c["eq_plot"],
        c["alloc_plot"],
        c["pnl_plot"],
        c["committee_out"],
        c["decision_center_out"],
        c["rebalance_out"],
        c["watchlist_out"],
        c["trades_out"],
        c["thesis_out"],
        c["capital_chart_out"],
        c["top_picks_out"],
        c["trade_freq_out"],
        c["buy_candidates_out"],
        c["signal_history_out"],
        c["rec_history_out"],
        c["timeline_trades_out"],
        c["scorecard_out"],
        c["metrics_out"],
        c["returns_hist_plot"],
        c["winloss_plot"],
        c["investor_out"],
        c["fi_plot"],
        c["val_out"],
        c["hero_out"],
        c["spy_banner_out"],
    ])

    # Stateful callbacks need gr.State inputs — kept as separate ticks.
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
    timer.tick(fn=_refresh_sym,  inputs=[c["_sym_state"]],
               outputs=[c["symbol_selector"], c["_sym_state"]])
    timer.tick(fn=_sym_detail,   inputs=[c["_sym_state"]], outputs=[c["symbol_detail_out"]])
    timer.tick(fn=_sim_choices,  outputs=[c["sim_sym_dd"]])
