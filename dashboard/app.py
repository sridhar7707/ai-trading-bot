# ================================================================
# UI CHANGE TRACKING
# After making ANY change to this file run:
#   python tests/ui_changelog.py
# This updates docs/UI_CHANGELOG.md automatically.
# Do not skip this step.
# ================================================================
# ================================================================
# REQUIREMENTS TRACKING
# After making ANY change to this file run:
#   python tests/requirements_tracker.py
# This updates docs/REQUIREMENTS.md automatically.
# ================================================================
"""Gradio dashboard &mdash; TradeGenius AI, hosted on HuggingFace Spaces."""
from __future__ import annotations

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import gradio as gr
from loguru import logger

# ── Design system ─────────────────────────────────────────────────────────────
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    ACTION_BUY_BG, ACTION_SELL_BG, ACTION_TRIM_BG,
    ACTION_HOLD_BG, ACTION_WATCH_BG, ACTION_ADD_BG, ACTION_EXIT_BG,
    PRIMARY, GAIN, LOSS, NEURAL,
    PRIMARY_BG, GAIN_BG, LOSS_BG, NEURAL_BG,
    GAIN_BD, LOSS_BD, NEURAL_BD, ACTION_ADD, ACTION_EXIT,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, CARD_RADIUS, ROW_PADDING, SECTION_GAP, INNER_GAP,
    SYMBOL_STYLE, PLOTLY_LAYOUT,
    _pnl_color, _card, _label, _hero_value, _section_title, _action_badge,
    _symbol, _confidence_bar, _metric_row, _progress_bar, _divider,
    _empty_state, _action_row, _table,
    _sym, _badge, _num, _pnl, _section, _wrap, _stat_card,
    TH, TD, TD0,
)
from dashboard.layout import GRADIO_CSS, STYLES, LOGO, HEADER_HTML, FOOTER_HTML, TAB_FIX_JS
from dashboard.data import (
    get_data, DB_PATH, HF_TOKEN, HF_REPO_ID,
    _now_ct, _to_ct, _market_status,
)
from dashboard.charts import (
    render_equity_chart, render_allocation_chart,
    render_pnl_chart, render_feature_importance_chart,
    render_returns_histogram, render_winloss_chart,
    _get_sym_hist, _sym_perf, _sparkline, _FI_LABELS,
)
from dashboard.components.overview import (
    render_daily_headline, render_portfolio_health_hero,
    render_trade_frequency, render_spy_banner,
)
from dashboard.components.market_mood import render_market_mood
from dashboard.components.ai_panel import render_ai_recommendation, render_ai_committee, _WHY_MAP
from dashboard.components.risk import render_risk_panel, render_market_intelligence, _risk_level, _SECTOR_MAP
from dashboard.components.portfolio import render_positions, render_trades
from dashboard.components.models import (
    render_validation_report, render_institutional_metrics,
    render_investor_view, render_paper_trading_scorecard,
)
from dashboard.components.signals import render_watchlist, render_timeline
from dashboard.components.history import render_whats_changed, render_portfolio_performance, _perf_choices
from dashboard.components.recommendation_history import (
    render_recommendation_history, render_buy_candidates, render_top_picks,
)
from dashboard.components.news import render_news_feed
from dashboard.components.signal_history import render_signal_history
from dashboard.components.actions import render_todays_actions, render_portfolio_actions
from dashboard.components.analysis import render_sell_analysis, render_position_sizing_panel, render_position_sizing
from dashboard.components.decision import render_decision_center
from dashboard.components.rebalance import render_rebalance
from dashboard.components.symbol_detail import render_symbol_detail, _get_symbol_choices
from dashboard.components.settings import render_settings_summary, render_investor_profile
from dashboard.components.brief import render_morning_brief, render_scheduler_status, render_three_question_summary
from dashboard.components.thesis import render_thesis_tracker
from dashboard.components.weekly_summary import render_weekly_summary
from dashboard.components.simulator import render_portfolio_simulator
from dashboard.components.timeline import render_all_timelines
from dashboard.components.executive_summary import render_executive_summary
from dashboard.components.decision_bar import render_decision_bar
from dashboard.components.capital import (
    render_capital_overview, render_capital_chart,
    render_profit_breakdown, save_reinvestment_mode,
)
from dashboard.timers import register_all_timers
from database.user_settings import get_all_settings, save_setting, get_setting
from bot.core.error_logger import safe_render, timed
from bot.core.recommendation_engine import (
    get_portfolio_action, get_position_sizing,
    get_sell_analysis, get_recommendation_explanation, get_portfolio_health,
)

_logger = logger

_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#0f1115",
    body_text_color="#ffffff",
    block_background_fill="#171a21",
    block_border_color="#2d3445",
    block_radius="12px",
    button_primary_background_fill="#00c853",
    button_primary_text_color="#000000",
    button_secondary_background_fill="#222733",
    button_secondary_text_color="#ffffff",
    border_color_primary="#2d3445",
)

with gr.Blocks(title="TradeGenius AI", theme=_theme, css=GRADIO_CSS, js=TAB_FIX_JS) as _demo:
    gr.HTML(HEADER_HTML)
    with gr.Tabs():
        # ── Tab 1: Brief ──────────────────────────────────────────────────────
        with gr.TabItem("📋 Brief"):
            # Fast cards (DB-only): rendered immediately via value=callable.
            # Everything else starts empty and is populated by timers.
            exec_summary_out     = gr.HTML(value=render_executive_summary, show_label=False)
            three_q_out          = gr.HTML(value="", show_label=False, elem_id="three_q_out")
            decision_bar_out     = gr.HTML(value=render_decision_bar, show_label=False)
            scheduler_status_out = gr.HTML(value=render_scheduler_status, show_label=False)
            morning_brief_out    = gr.HTML(value=render_morning_brief, show_label=False)
            pos_brief_out        = gr.HTML(value=render_positions, show_label=False)
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("What Changed Today", open=False):
                        whats_changed_out = gr.HTML(value="")
                with gr.Column():
                    with gr.Accordion("Market Mood", open=False):
                        market_mood_out   = gr.HTML(value="")
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("AI Committee", open=False):
                        ai_rec_brief_out  = gr.HTML(value="")
                with gr.Column():
                    with gr.Accordion("Risk Panel", open=False):
                        risk_panel_out    = gr.HTML(value="")
                        mkt_intel_out     = gr.HTML(value="")
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("News", open=False):
                        news_out          = gr.HTML(value="")
                with gr.Column():
                    with gr.Accordion("Decision Timeline", open=False):
                        timeline_brief_out = gr.HTML(value="")

        # ── Tab 2: Portfolio ──────────────────────────────────────────────────
        with gr.TabItem("💼 Portfolio"):
            weekly_summary_out  = gr.HTML(value=render_weekly_summary)
            daily_headline_out  = gr.HTML(value=render_daily_headline)
            hero_out            = gr.HTML(value=render_portfolio_health_hero)
            spy_banner_out      = gr.HTML(value="")
            perf_tabs           = gr.Radio(
                choices=_perf_choices(), value=_perf_choices()[2],
                label="", container=False, elem_classes=["perf-tabs"],
            )
            perf_key_state = gr.State(value="1M")
            perf_out       = gr.HTML(value="")
            with gr.Row():
                with gr.Column(scale=65):
                    eq_plot    = gr.Plot(value=None, label="", show_label=False)
                with gr.Column(scale=35):
                    alloc_plot = gr.Plot(value=None, label="", show_label=False)
            pnl_plot            = gr.Plot(value=None, label="", show_label=False)
            committee_out       = gr.HTML(value="")
            decision_center_out = gr.HTML(value="")
            rebalance_out       = gr.HTML(value="")
            watchlist_out       = gr.HTML(value="")
            pos_out             = gr.HTML(value=render_positions)
            trades_out          = gr.HTML(value="")
            thesis_out          = gr.HTML(value="")
            _initial_choices = _get_symbol_choices()
            _initial_sym     = _initial_choices[0] if _initial_choices else None
            symbol_selector  = gr.Dropdown(
                choices=_initial_choices, label="🔍 Symbol Detail",
                value=_initial_sym, container=True, elem_classes=["sym-selector"],
            )
            symbol_detail_out = gr.HTML(value="")
            _sym_state        = gr.State(value=_initial_sym)
            _sim_syms  = sorted(get_data().get("prices", {}).keys()) or []
            sim_sym_dd = gr.Dropdown(choices=_sim_syms, label="🔬 Simulate: Symbol", container=True)
            sim_amt_sl = gr.Slider(minimum=100, maximum=10000, value=500, step=100,
                                   label="Amount ($)", container=True)
            simulator_out = gr.HTML(value="")

        # ── Tab 3: Capital ────────────────────────────────────────────────────
        with gr.TabItem("💰 Capital"):
            capital_overview_out  = gr.HTML(value=render_capital_overview)
            capital_chart_out     = gr.Plot(value=None, label="Capital Growth", show_label=False)
            profit_breakdown_out  = gr.HTML(value=render_profit_breakdown)
            _cur_reinvest = get_setting("reinvest_profits_only", "false")
            reinvest_radio = gr.Radio(
                choices=[
                    "Reinvest everything (profits + initial deposit)",
                    "Reinvest profits only (protect initial deposit)",
                ],
                value=(
                    "Reinvest profits only (protect initial deposit)"
                    if _cur_reinvest == "true"
                    else "Reinvest everything (profits + initial deposit)"
                ),
                label="Reinvestment Mode",
            )
            _reinvest_desc = (
                "Reinvest profits only &mdash; your initial deposit is always protected"
                if _cur_reinvest == "true"
                else "Reinvest everything &mdash; profits and initial deposit both grow the position"
            )
            reinvest_status = gr.HTML(value=(
                f'<span style="color:{GAIN};font-size:12px;">'
                f'&#10003; Active: {_reinvest_desc}</span>'
            ))

        # ── Tab 4: Trades ─────────────────────────────────────────────────────
        with gr.TabItem("📈 Trades"):
            top_picks_out       = gr.HTML(value="")
            trade_freq_out      = gr.HTML(value="")
            buy_candidates_out  = gr.HTML(value="")
            signal_history_out  = gr.HTML(value="")
            rec_history_out     = gr.HTML(value="")
            timeline_trades_out = gr.HTML(value="")

        # ── Tab 5: Performance ────────────────────────────────────────────────
        with gr.TabItem("📊 Performance"):
            scorecard_out = gr.HTML(value="")          # yfinance — populated by 300 s timer
            metrics_out   = gr.HTML(value=render_institutional_metrics)  # callable: spy path guarded by @safe_render; 300 s timer refreshes
            with gr.Row():
                returns_hist_plot = gr.Plot(value=None, label="", show_label=False)
                winloss_plot      = gr.Plot(value=None, label="", show_label=False)
            model_view = gr.Radio(
                choices=["📊 Investor View", "🔬 Developer View"],
                value="📊 Investor View", label="", container=False,
            )
            investor_out = gr.HTML(value=render_investor_view, visible=True)
            with gr.Column(visible=False) as dev_col:
                with gr.Row():
                    with gr.Column(scale=65):
                        fi_plot = gr.Plot(value=None, label="", show_label=False)
                    with gr.Column(scale=35):
                        val_out = gr.HTML(value="")

        # ── Tab 6: Settings ───────────────────────────────────────────────────
        with gr.TabItem("⚙️ Settings"):
            _s0 = get_all_settings()
            def _pct(key: str, default: str) -> float:
                try:
                    return round(float(_s0.get(key, default)) * 100, 1)
                except (ValueError, TypeError):
                    return float(default) * 100
            with gr.Row():
                with gr.Column(scale=1):
                    _risk_radio = gr.Radio(
                        choices=["Conservative", "Moderate", "Aggressive"],
                        value=_s0.get("risk_tolerance", "Moderate"), label="Risk Tolerance",
                    )
                    _bench_radio = gr.Radio(
                        choices=["SPY", "QQQ", "DIA"],
                        value=_s0.get("benchmark", "SPY"), label="Benchmark",
                    )
                    _max_pos_sl = gr.Slider(minimum=5, maximum=50, step=1,
                                            value=_pct("max_position_pct", "0.20"),
                                            label="Max Position Size %")
                    _max_dd_sl  = gr.Slider(minimum=5, maximum=30, step=1,
                                            value=_pct("max_drawdown_pct", "0.12"),
                                            label="Max Drawdown Threshold %")
                    _stop_sl    = gr.Slider(minimum=1, maximum=15, step=0.5,
                                            value=_pct("stop_loss_pct", "0.04"),
                                            label="Stop-Loss Default %")
                    _notif_check = gr.Checkbox(
                        value=_s0.get("notifications_enabled", "false") == "true",
                        label="Enable Notifications",
                    )
                    _save_btn    = gr.Button("💾 Save Settings", variant="primary")
                    _save_status = gr.HTML(value="")
                with gr.Column(scale=1):
                    settings_summary_out = gr.HTML(value=render_settings_summary)
            investor_profile_out = gr.HTML(value=render_investor_profile)

    gr.HTML(value=FOOTER_HTML)

    # ── Event handlers ────────────────────────────────────────────────────────
    model_view.change(
        fn=lambda v: (gr.update(visible=(v == "📊 Investor View")),
                      gr.update(visible=(v == "🔬 Developer View"))),
        inputs=[model_view], outputs=[investor_out, dev_col],
    )

    def _on_perf_change(period_label):
        key = (period_label or "1M").split()[0]
        return render_portfolio_performance(period_label), key
    perf_tabs.change(fn=_on_perf_change, inputs=[perf_tabs],
                     outputs=[perf_out, perf_key_state])

    symbol_selector.change(
        fn=lambda v: (render_symbol_detail(v), v),
        inputs=[symbol_selector], outputs=[symbol_detail_out, _sym_state],
    )

    def _run_sim(sym, amt):
        return render_portfolio_simulator(sym, float(amt) if amt else 500.0)
    sim_sym_dd.change(fn=_run_sim, inputs=[sim_sym_dd, sim_amt_sl], outputs=simulator_out)
    sim_amt_sl.change(fn=_run_sim, inputs=[sim_sym_dd, sim_amt_sl], outputs=simulator_out)

    reinvest_radio.change(fn=save_reinvestment_mode, inputs=[reinvest_radio],
                          outputs=[reinvest_status])

    def _save_settings(risk_tol, benchmark, max_pos, max_dd, stop_loss, notif):
        max_pos   = max(5.0,  min(50.0, max_pos))
        max_dd    = max(5.0,  min(30.0, max_dd))
        stop_loss = max(1.0,  min(15.0, stop_loss))
        results = [
            save_setting("risk_tolerance",        risk_tol),
            save_setting("benchmark",             benchmark),
            save_setting("max_position_pct",      str(round(max_pos   / 100, 4))),
            save_setting("max_drawdown_pct",      str(round(max_dd    / 100, 4))),
            save_setting("stop_loss_pct",         str(round(stop_loss / 100, 4))),
            save_setting("notifications_enabled", "true" if notif else "false"),
        ]
        ok = all(results)
        status = (
            '<p style="color:#00c853;font-weight:600;margin:8px 0 0">'
            '&#10003; Saved &mdash; active on next bot cycle</p>'
            if ok else
            '<p style="color:#ef4444;font-weight:600;margin:8px 0 0">'
            '&#9888; Save failed &mdash; check application logs</p>'
        )
        return render_settings_summary(), status
    _save_btn.click(
        fn=_save_settings,
        inputs=[_risk_radio, _bench_radio, _max_pos_sl, _max_dd_sl, _stop_sl, _notif_check],
        outputs=[settings_summary_out, _save_status],
    )

    # ── Timer registration ────────────────────────────────────────────────────
    timer_ui   = gr.Timer(value=60)    # 1 min — DB reads only, no yfinance; fast on HF free tier
    timer_data = gr.Timer(value=300)   # 5 min — yfinance (15 s timeout), charts, AI, news
    register_all_timers(timer_ui, timer_data, {
        "exec_summary_out":    exec_summary_out,
        # Brief tab
        "three_q_out":         three_q_out,
        "decision_bar_out":    decision_bar_out,
        "scheduler_status_out": scheduler_status_out,
        "morning_brief_out":   morning_brief_out,
        "pos_brief_out":       pos_brief_out,
        "whats_changed_out":   whats_changed_out,
        "market_mood_out":     market_mood_out,
        "ai_rec_brief_out":    ai_rec_brief_out,
        "risk_panel_out":      risk_panel_out,
        "mkt_intel_out":       mkt_intel_out,
        "news_out":            news_out,
        "timeline_brief_out":  timeline_brief_out,
        # Portfolio tab
        "weekly_summary_out":  weekly_summary_out,
        "daily_headline_out":  daily_headline_out,
        "hero_out":            hero_out,
        "spy_banner_out":      spy_banner_out,
        "perf_tabs":           perf_tabs,
        "perf_key_state":      perf_key_state,
        "perf_out":            perf_out,
        "eq_plot":             eq_plot,
        "alloc_plot":          alloc_plot,
        "pnl_plot":            pnl_plot,
        "committee_out":       committee_out,
        "decision_center_out": decision_center_out,
        "rebalance_out":       rebalance_out,
        "watchlist_out":       watchlist_out,
        "pos_out":             pos_out,
        "trades_out":          trades_out,
        "thesis_out":          thesis_out,
        "symbol_selector":     symbol_selector,
        "_sym_state":          _sym_state,
        "symbol_detail_out":   symbol_detail_out,
        "sim_sym_dd":          sim_sym_dd,
        # Capital tab
        "capital_overview_out":  capital_overview_out,
        "capital_chart_out":     capital_chart_out,
        "profit_breakdown_out":  profit_breakdown_out,
        # Trades tab
        "top_picks_out":       top_picks_out,
        "trade_freq_out":      trade_freq_out,
        "buy_candidates_out":  buy_candidates_out,
        "signal_history_out":  signal_history_out,
        "rec_history_out":     rec_history_out,
        "timeline_trades_out": timeline_trades_out,
        # Performance tab
        "scorecard_out":       scorecard_out,
        "metrics_out":         metrics_out,
        "returns_hist_plot":   returns_hist_plot,
        "winloss_plot":        winloss_plot,
        "investor_out":        investor_out,
        "fi_plot":             fi_plot,
        "val_out":             val_out,
        # Settings tab
        "settings_summary_out":  settings_summary_out,
        "investor_profile_out":  investor_profile_out,
    })


# ── Cron HTTP endpoint + static-file fix ─────────────────────────────────────
# Gradio 5.9.0's App.create_app builds the API routes but never registers a
# handler for /_app/immutable/* — those requests fall through to FastAPI's
# default 404.  Fix: insert a StaticFiles Mount at index 0 of the router
# (checked before every other route/mount) so all /_app/* requests are served
# from Gradio's compiled SvelteKit frontend package.
#
# NOTE: do NOT mount /static — Gradio registers APIRoute /static/{path:path}
# (static_resource) to serve its own assets.  An overlapping StaticFiles mount
# at index 0 would shadow that route and break Gradio's static resource serving.
# The /static/fonts/*.woff2 404s from Gradio 5's CSS are benign — those are
# OS-level system fonts (ui-sans-serif, system-ui) that browsers provide natively.
import threading
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount
from gradio.routes import App as _GradioApp

app = _GradioApp.create_app(_demo, app_kwargs={"docs_url": None, "redoc_url": None})

_GR_APP_DIR = Path(gr.__file__).parent / "templates" / "frontend" / "_app"
if _GR_APP_DIR.is_dir():
    app.router.routes.insert(0, Mount("/_app", app=StaticFiles(directory=str(_GR_APP_DIR))))


@app.get("/run/cron")
async def _cron_endpoint():
    from scheduler.dispatcher import main as _dispatch
    threading.Thread(target=_dispatch, daemon=True, name="cron-dispatcher").start()
    return JSONResponse({"status": "accepted"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
