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
    render_daily_headline, render_portfolio_health_hero,
    render_trade_frequency, render_spy_banner,
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
    render_positions, render_trades,
)
from dashboard.components.models import (
    render_validation_report, render_institutional_metrics, render_investor_view,
    render_paper_trading_scorecard,
)
from dashboard.components.signals import (
    render_watchlist, render_timeline,
)
from dashboard.components.history import (
    render_whats_changed, render_portfolio_performance, _perf_choices,
)
from dashboard.components.recommendation_history import render_recommendation_history, render_buy_candidates, render_top_picks
from dashboard.components.news import render_news_feed
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
from dashboard.components.settings import render_settings_summary
from database.user_settings import get_all_settings, save_setting

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

# ── Gradio layout &mdash; 4-tab design ──────────────────────────────────────────────
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
            daily_headline_out  = gr.HTML(value=render_daily_headline)
            hero_out            = gr.HTML(value=render_portfolio_health_hero)
            spy_banner_dash_out = gr.HTML(value=render_spy_banner)
            top_picks_out       = gr.HTML(value=render_top_picks)
            market_mood_out     = gr.HTML(value=render_market_mood)
            trade_freq_out      = gr.HTML(value=render_trade_frequency)
            todays_actions_out  = gr.HTML(value=render_todays_actions)
            ai_rec_out          = gr.HTML(value=render_ai_recommendation)
            with gr.Row():
                with gr.Column(scale=65):
                    risk_panel_out   = gr.HTML(value=render_risk_panel)
                with gr.Column(scale=35):
                    mkt_intel_out    = gr.HTML(value=render_market_intelligence)
            whats_changed_out   = gr.HTML(value=render_whats_changed)
            # ── Symbol drilldown ──────────────────────────────────────────────
            _initial_choices = _get_symbol_choices()
            _initial_sym     = _initial_choices[0] if _initial_choices else None
            symbol_selector = gr.Dropdown(
                choices=_initial_choices,
                label="🔍 Symbol Detail",
                value=_initial_sym, container=True,
                elem_classes=["sym-selector"],
            )
            symbol_detail_out = gr.HTML(value=lambda: render_symbol_detail(_initial_sym))
            _sym_state = gr.State(value=_initial_sym)   # tracks selection without self-reference

        with gr.TabItem("📰 News"):
            news_out = gr.HTML(value=render_news_feed)

        with gr.TabItem("⚡ Signals"):
            buy_candidates_out  = gr.HTML(value=render_buy_candidates)
            timeline_out        = gr.HTML(value=render_timeline)
            signal_history_out  = gr.HTML(value=render_signal_history)
            rec_history_sig_out = gr.HTML(value=render_recommendation_history)

        with gr.TabItem("💼 Portfolio"):
            perf_tabs   = gr.Radio(
                choices=_perf_choices(),
                value=_perf_choices()[2],   # default: 1M
                label="", container=False,
                elem_classes=["perf-tabs"],
            )
            perf_key_state = gr.State(value="1M")   # tracks period key independent of label
            perf_out    = gr.HTML(value=render_portfolio_performance)
            with gr.Row():
                with gr.Column(scale=65):
                    eq_plot    = gr.Plot(value=render_equity_chart, label="")
                with gr.Column(scale=35):
                    alloc_plot = gr.Plot(value=render_allocation_chart, label="")
            pnl_plot          = gr.Plot(value=render_pnl_chart, label="")
            committee_out     = gr.HTML(value=render_ai_committee)
            decision_center_out = gr.HTML(value=render_decision_center)
            rebalance_out       = gr.HTML(value=render_rebalance)
            watchlist_out       = gr.HTML(value=render_watchlist)
            pos_out             = gr.HTML(value=render_positions)
            trades_out          = gr.HTML(value=render_trades)

        with gr.TabItem("📈 Performance"):
            scorecard_out = gr.HTML(value=render_paper_trading_scorecard)
            model_view = gr.Radio(
                choices=["📊 Investor View", "🔬 Developer View"],
                value="📊 Investor View",
                label="", container=False,
            )
            investor_out = gr.HTML(value=render_investor_view, visible=True)
            with gr.Column(visible=False) as dev_col:
                metrics_out = gr.HTML(value=render_institutional_metrics)
                with gr.Row():
                    with gr.Column(scale=65):
                        fi_plot = gr.Plot(value=render_feature_importance_chart, label="")
                    with gr.Column(scale=35):
                        val_out = gr.HTML(value=render_validation_report)

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
                        value=_s0.get("risk_tolerance", "Moderate"),
                        label="Risk Tolerance",
                    )
                    _bench_radio = gr.Radio(
                        choices=["SPY", "QQQ", "DIA"],
                        value=_s0.get("benchmark", "SPY"),
                        label="Benchmark",
                    )
                    _max_pos_sl = gr.Slider(
                        minimum=5, maximum=50, step=1,
                        value=_pct("max_position_pct", "0.20"),
                        label="Max Position Size %",
                    )
                    _max_dd_sl = gr.Slider(
                        minimum=5, maximum=30, step=1,
                        value=_pct("max_drawdown_pct", "0.12"),
                        label="Max Drawdown Threshold %",
                    )
                    _stop_sl = gr.Slider(
                        minimum=1, maximum=15, step=0.5,
                        value=_pct("stop_loss_pct", "0.04"),
                        label="Stop-Loss Default %",
                    )
                    _notif_check = gr.Checkbox(
                        value=_s0.get("notifications_enabled", "false") == "true",
                        label="Enable Notifications",
                    )
                    _save_btn = gr.Button("💾 Save Settings", variant="primary")
                    _save_status = gr.HTML(value="")
                with gr.Column(scale=1):
                    settings_summary_out = gr.HTML(value=render_settings_summary)

    gr.HTML(value=FOOTER_HTML)

    # Models tab toggle
    model_view.change(
        fn=lambda v: (gr.update(visible=(v == "📊 Investor View")),
                      gr.update(visible=(v == "🔬 Developer View"))),
        inputs=[model_view],
        outputs=[investor_out, dev_col],
    )

    # Portfolio performance period selection &mdash; also store the bare key in state
    def _on_perf_change(period_label):
        key = (period_label or "1M").split()[0]
        return render_portfolio_performance(period_label), key
    perf_tabs.change(
        fn=_on_perf_change,
        inputs=[perf_tabs],
        outputs=[perf_out, perf_key_state],
    )

    # Symbol drilldown — single handler updates both the detail panel and the state
    symbol_selector.change(
        fn=lambda v: (render_symbol_detail(v), v),
        inputs=[symbol_selector],
        outputs=[symbol_detail_out, _sym_state],
    )

    # One shared timer &mdash; cache layer ensures a single DB+API refresh per tick
    timer = gr.Timer(value=60)
    # Dashboard tab
    timer.tick(fn=render_daily_headline,        outputs=daily_headline_out)
    timer.tick(fn=render_portfolio_health_hero, outputs=hero_out)
    timer.tick(fn=render_spy_banner,            outputs=spy_banner_dash_out)
    timer.tick(fn=render_top_picks,             outputs=top_picks_out)
    timer.tick(fn=render_market_mood,           outputs=market_mood_out)
    timer.tick(fn=render_trade_frequency,       outputs=trade_freq_out)
    timer.tick(fn=render_todays_actions,        outputs=todays_actions_out)
    timer.tick(fn=render_ai_recommendation,     outputs=ai_rec_out)
    timer.tick(fn=render_risk_panel,            outputs=risk_panel_out)
    timer.tick(fn=render_market_intelligence,   outputs=mkt_intel_out)
    timer.tick(fn=render_whats_changed,         outputs=whats_changed_out)
    def _refresh_symbol_choices(sel):
        choices = _get_symbol_choices()
        val = sel if sel in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=val), val   # val heals _sym_state on fallback
    # Use _sym_state as input so symbol_selector is never both input and output;
    # also write val back to _sym_state so it stays in sync when timer changes the selection
    timer.tick(fn=_refresh_symbol_choices, inputs=[_sym_state], outputs=[symbol_selector, _sym_state])
    timer.tick(fn=render_symbol_detail, inputs=[_sym_state], outputs=[symbol_detail_out])
    # News tab (30-min internal cache &mdash; refreshes on every timer tick but skips API if cached)
    timer.tick(fn=render_news_feed, outputs=news_out)
    # Signals tab
    timer.tick(fn=render_buy_candidates,         outputs=buy_candidates_out)
    timer.tick(fn=render_timeline,               outputs=timeline_out)
    timer.tick(fn=render_signal_history,         outputs=signal_history_out)
    timer.tick(fn=render_recommendation_history, outputs=rec_history_sig_out)
    # Portfolio tab &mdash; use key state (not Radio value) to avoid stale-label validation errors
    def _refresh_perf_tabs(current_key):
        if not isinstance(current_key, str):
            current_key = "1M"
        choices = _perf_choices()
        matched = next((c for c in choices if c.split()[0] == current_key), None)
        val     = matched or (choices[2] if len(choices) > 2 else choices[0] if choices else None)
        new_key = val.split()[0] if val else current_key
        html    = render_portfolio_performance(val or "1M  —")
        return gr.update(choices=choices, value=val), new_key, html
    timer.tick(fn=_refresh_perf_tabs, inputs=[perf_key_state],
               outputs=[perf_tabs, perf_key_state, perf_out])
    timer.tick(fn=render_equity_chart,          outputs=eq_plot)
    timer.tick(fn=render_allocation_chart,      outputs=alloc_plot)
    timer.tick(fn=render_pnl_chart,             outputs=pnl_plot)
    timer.tick(fn=render_ai_committee,          outputs=committee_out)
    timer.tick(fn=render_decision_center,       outputs=decision_center_out)
    timer.tick(fn=render_rebalance,             outputs=rebalance_out)
    timer.tick(fn=render_watchlist,             outputs=watchlist_out)
    timer.tick(fn=render_positions,             outputs=pos_out)
    timer.tick(fn=render_trades,                outputs=trades_out)
    # Performance tab
    timer.tick(fn=render_paper_trading_scorecard,  outputs=scorecard_out)
    timer.tick(fn=render_investor_view,            outputs=investor_out)
    timer.tick(fn=render_institutional_metrics,    outputs=metrics_out)
    timer.tick(fn=render_feature_importance_chart, outputs=fi_plot)
    timer.tick(fn=render_validation_report,        outputs=val_out)
    # Settings tab — timer keeps summary in sync if another session saved changes
    timer.tick(fn=render_settings_summary, outputs=settings_summary_out)

    def _save_settings(
        risk_tol: str, benchmark: str,
        max_pos: float, max_dd: float, stop_loss: float,
        notif: bool,
    ) -> tuple[str, str]:
        # Server-side bounds — UI sliders are client-only guards
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
        if all(results):
            status = (
                '<p style="color:#00c853;font-weight:600;margin:8px 0 0">'
                '&#10003; Saved &mdash; active on next bot cycle</p>'
            )
        else:
            status = (
                '<p style="color:#ef4444;font-weight:600;margin:8px 0 0">'
                '&#9888; Save failed &mdash; check application logs</p>'
            )
        return render_settings_summary(), status

    _save_btn.click(
        fn=_save_settings,
        inputs=[_risk_radio, _bench_radio, _max_pos_sl, _max_dd_sl, _stop_sl, _notif_check],
        outputs=[settings_summary_out, _save_status],
    )

if __name__ == "__main__":
    demo.launch()
