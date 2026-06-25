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
"""Gradio dashboard — TradeGenius AI, hosted on HuggingFace Spaces."""
from __future__ import annotations

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import gradio as gr
from loguru import logger

# ── Design system constants ───────────────────────────────────────────────────
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER,
    TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    ACTION_ADD, ACTION_EXIT,
    ACTION_BUY_BG, ACTION_SELL_BG, ACTION_TRIM_BG,
    ACTION_HOLD_BG, ACTION_WATCH_BG, ACTION_ADD_BG, ACTION_EXIT_BG,
    PRIMARY, GAIN, LOSS, NEURAL,
    PRIMARY_BG, GAIN_BG, LOSS_BG, NEURAL_BG,
    GAIN_BD, LOSS_BD, NEURAL_BD,
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
# ── Layout constants (CSS + static HTML) ─────────────────────────────────────
from dashboard.layout import GRADIO_CSS, STYLES, LOGO, HEADER_HTML, FOOTER_HTML

# ── Data layer ────────────────────────────────────────────────────────────────
from dashboard.data import (
    get_data, DB_PATH, HF_TOKEN, HF_REPO_ID,
    _now_ct, _to_ct, _market_status,
)

# ── Chart render functions ────────────────────────────────────────────────────
from dashboard.charts import (
    render_equity_chart, render_allocation_chart,
    render_pnl_chart, render_feature_importance_chart,
    _get_sym_hist, _sym_perf, _sparkline, _FI_LABELS,
)

# ── Component render functions ────────────────────────────────────────────────
from dashboard.components.overview import (
    render_metrics, render_dashboard_hero, render_portfolio_health_hero,
    render_benchmark_comparison,
)
from dashboard.components.market_mood import render_market_mood
from dashboard.components.ai_panel import (
    render_ai_recommendation, render_ai_committee, _WHY_MAP,
)
from dashboard.components.risk import (
    render_risk_panel, render_market_intelligence,
    _risk_level, _SECTOR_MAP,
)
from dashboard.components.portfolio import (
    render_positions, render_trades, _SELL_REASON,
)
from dashboard.components.models import (
    render_validation_report, render_institutional_metrics, render_investor_view,
)
from dashboard.components.signals import (
    render_watchlist, render_signals_tab, render_timeline,
)
from dashboard.components.history import (
    render_whats_changed, render_portfolio_performance, _perf_choices,
    render_recommendation_history,
)
from dashboard.components.signal_history import render_signal_history
from dashboard.components.actions import (
    render_todays_actions, render_portfolio_actions,
)
from dashboard.components.analysis import (
    render_sell_analysis, render_position_sizing_panel, render_position_sizing,
)
from dashboard.components.decision import render_decision_center
from dashboard.components.rebalance import render_rebalance
from dashboard.components.symbol_detail import (
    render_symbol_detail, _get_symbol_choices,
)

# ── Recommendation engine (imported for components that need it at top-level) ─
from bot.core.error_logger import safe_render, timed
from bot.core.recommendation_engine import (
    get_portfolio_action,
    get_position_sizing,
    get_sell_analysis,
    get_recommendation_explanation,
    get_portfolio_health,
)

_logger = logger

# ── Gradio layout — 4-tab design ──────────────────────────────────────────────
# Gradio 5 removed every= from components. Use gr.Timer + .tick() instead.
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

with gr.Blocks(title="TradeGenius AI", theme=_theme, css=GRADIO_CSS) as demo:
    gr.HTML(HEADER_HTML)
    gr.HTML("""
    <script>
    function enforceTabStyles() {
      const buttons = document.querySelectorAll(
        '.tab-nav button, .tabs button[role="tab"], button[id*="tab"]'
      );
      buttons.forEach(btn => {
        btn.style.setProperty('color', '#ffffff', 'important');
        const isSelected = btn.classList.contains('selected');
        btn.style.setProperty('opacity', isSelected ? '1' : '0.6', 'important');
        if (!btn._tgListeners) {
          btn._tgListeners = true;
          btn.addEventListener('mouseenter', () => {
            btn.style.setProperty('opacity', '1', 'important');
          });
          btn.addEventListener('mouseleave', () => {
            if (!btn.classList.contains('selected')) {
              btn.style.setProperty('opacity', '0.6', 'important');
            }
          });
        }
      });
    }
    const observer = new MutationObserver(enforceTabStyles);
    observer.observe(document.body, {
      subtree: true, attributes: true, attributeFilter: ['class']
    });
    setTimeout(enforceTabStyles, 300);
    setTimeout(enforceTabStyles, 800);
    setTimeout(enforceTabStyles, 2000);
    </script>
    """)

    with gr.Tabs():
        with gr.TabItem("📊 Dashboard"):
            # Exactly 5 panels — open dashboard and within 3s: health, actions, risk
            hero_out           = gr.HTML(value=render_portfolio_health_hero)
            market_mood_out    = gr.HTML(value=render_market_mood)
            todays_actions_out = gr.HTML(value=render_todays_actions)
            ai_rec_out         = gr.HTML(value=render_ai_recommendation)
            risk_panel_out     = gr.HTML(value=render_risk_panel)
            benchmark_out      = gr.HTML(value=render_benchmark_comparison)
            whats_changed_out  = gr.HTML(value=render_whats_changed)
            # ── Symbol drilldown ──────────────────────────────────────────────
            symbol_selector = gr.Dropdown(
                choices=_get_symbol_choices(),
                label="🔍 Symbol Detail — select a ticker to drill down",
                value=None, container=True,
            )
            symbol_detail_out = gr.HTML(value=lambda: render_symbol_detail(None))

        with gr.TabItem("⚡ Signals"):
            timeline_out  = gr.HTML(value=render_timeline)
            signals_out   = gr.HTML(value=render_signals_tab)
            with gr.Row():
                with gr.Column(scale=55):
                    mkt_intel_out = gr.HTML(value=render_market_intelligence)
                with gr.Column(scale=45):
                    watchlist_out = gr.HTML(value=render_watchlist)

        with gr.TabItem("🎯 Signal History"):
            signal_history_out = gr.HTML(value=render_signal_history)

        with gr.TabItem("💼 Portfolio"):
            perf_tabs   = gr.Radio(
                choices=_perf_choices(),
                value=_perf_choices()[2],   # default: 1M
                label="", container=False,
                elem_classes=["perf-tabs"],
            )
            perf_out    = gr.HTML(value=render_portfolio_performance)
            with gr.Row():
                with gr.Column(scale=65):
                    eq_plot    = gr.Plot(value=render_equity_chart, label="")
                with gr.Column(scale=35):
                    alloc_plot = gr.Plot(value=render_allocation_chart, label="")
            pnl_plot          = gr.Plot(value=render_pnl_chart, label="")
            committee_out     = gr.HTML(value=render_ai_committee)
            decision_center_out = gr.HTML(value=render_decision_center)
            rebalance_out     = gr.HTML(value=render_rebalance)
            pos_out           = gr.HTML(value=render_positions)
            trades_out        = gr.HTML(value=render_trades)

        with gr.TabItem("🔬 Models"):
            model_view = gr.Radio(
                choices=["📊 Investor View", "🔬 Developer View"],
                value="📊 Investor View",
                label="", container=False,
            )
            rec_history_out = gr.HTML(value=render_recommendation_history)
            investor_out = gr.HTML(value=render_investor_view, visible=True)
            with gr.Column(visible=False) as dev_col:
                metrics_out = gr.HTML(value=render_institutional_metrics)
                with gr.Row():
                    with gr.Column(scale=65):
                        fi_plot = gr.Plot(value=render_feature_importance_chart, label="")
                    with gr.Column(scale=35):
                        val_out = gr.HTML(value=render_validation_report)

    gr.HTML(value=FOOTER_HTML)

    # Models tab toggle
    model_view.change(
        fn=lambda v: (gr.update(visible=(v == "📊 Investor View")),
                      gr.update(visible=(v == "🔬 Developer View"))),
        inputs=[model_view],
        outputs=[investor_out, dev_col],
    )

    # Portfolio performance period selection
    perf_tabs.change(
        fn=render_portfolio_performance,
        inputs=[perf_tabs],
        outputs=[perf_out],
    )

    # Symbol drilldown
    symbol_selector.change(
        fn=render_symbol_detail,
        inputs=[symbol_selector],
        outputs=[symbol_detail_out],
    )

    # One shared timer — cache layer ensures a single DB+API refresh per tick
    timer = gr.Timer(value=60)
    # Dashboard (5 panels)
    timer.tick(fn=render_portfolio_health_hero, outputs=hero_out)
    timer.tick(fn=render_market_mood,           outputs=market_mood_out)
    timer.tick(fn=render_todays_actions,        outputs=todays_actions_out)
    timer.tick(fn=render_ai_recommendation,     outputs=ai_rec_out)
    timer.tick(fn=render_risk_panel,            outputs=risk_panel_out)
    timer.tick(fn=render_benchmark_comparison,  outputs=benchmark_out)
    timer.tick(fn=render_whats_changed,         outputs=whats_changed_out)
    timer.tick(fn=lambda: gr.update(choices=_get_symbol_choices()), outputs=symbol_selector)
    timer.tick(fn=render_symbol_detail, inputs=[symbol_selector], outputs=[symbol_detail_out])
    # Signals tab
    timer.tick(fn=render_timeline,              outputs=timeline_out)
    timer.tick(fn=render_signals_tab,           outputs=signals_out)
    timer.tick(fn=render_market_intelligence,   outputs=mkt_intel_out)
    timer.tick(fn=render_watchlist,             outputs=watchlist_out)
    # Signal History tab
    timer.tick(fn=render_signal_history, outputs=signal_history_out)
    # Portfolio tab
    timer.tick(fn=lambda: gr.update(choices=_perf_choices()), outputs=perf_tabs)
    timer.tick(fn=render_portfolio_performance, outputs=perf_out)
    timer.tick(fn=render_equity_chart,          outputs=eq_plot)
    timer.tick(fn=render_allocation_chart,      outputs=alloc_plot)
    timer.tick(fn=render_pnl_chart,             outputs=pnl_plot)
    timer.tick(fn=render_ai_committee,          outputs=committee_out)
    timer.tick(fn=render_decision_center,       outputs=decision_center_out)
    timer.tick(fn=render_rebalance,             outputs=rebalance_out)
    timer.tick(fn=render_positions,             outputs=pos_out)
    timer.tick(fn=render_trades,                outputs=trades_out)
    # Models tab
    timer.tick(fn=render_recommendation_history,   outputs=rec_history_out)
    timer.tick(fn=render_investor_view,            outputs=investor_out)
    timer.tick(fn=render_institutional_metrics,    outputs=metrics_out)
    timer.tick(fn=render_feature_importance_chart, outputs=fi_plot)
    timer.tick(fn=render_validation_report,        outputs=val_out)

if __name__ == "__main__":
    demo.launch()
