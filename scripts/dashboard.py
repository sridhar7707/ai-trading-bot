"""
Trading Bot Dashboard — two-screen Gradio UI.

Select tier at the top:
  Subscriber             — Overview (halt), Positions, Trade Log
  Institutional / Enterprise — all 8 tabs (superset of Subscriber)

Run:  python scripts/dashboard.py
Then open http://localhost:7860
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# HF Spaces pins Gradio 4.x via sdk_version, but huggingface_hub>=0.30 removed HfFolder.
# Patch it back before Gradio imports so gradio/oauth.py doesn't blow up.
import huggingface_hub as _hfhub
if not hasattr(_hfhub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
        @staticmethod
        def delete_token(): pass
    _hfhub.HfFolder = _HfFolder

import gradio as gr
from loguru import logger
from bot.monitor.dashboard_data import (
    get_overview, overview_md,
    get_positions_df, get_returns_summary_df,
    trades_html_table,
    get_performance_metrics, performance_md, portfolio_chart, signals_chart, monthly_chart,
    get_audit_df, get_latest_signals_df, get_screener_df,
    get_compliance_state, compliance_gauges_html,
    halt_status_html, toggle_halt,
    refresh_db_from_hf, diagnostics, spy_return_since,
)

# ── Startup diagnostics ───────────────────────────────────────────────────────
# Runs once when the Space boots. The log output reveals the common $0.00 causes
# (missing HF_TOKEN, never-synced DB, empty trades table) immediately.
logger.info("Dashboard starting — running startup diagnostics…")
diagnostics()
# Force an initial pull so the very first render has fresh data (bypasses 5-min cache)
refresh_db_from_hf(force=True)
diagnostics()  # second pass shows DB state AFTER the pull


# ── Readiness check (Institutional) ──────────────────────────────────────────
from scripts._dashboard_readiness import run_readiness_check  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(label: str) -> str:
    # Light section header bar — matches the light dashboard theme.
    return (f"<div style='background:#f1f5f9;color:#475569;padding:6px 12px;"
            f"border-left:3px solid #0891b2;border-radius:4px;font-size:12px;"
            f"font-family:system-ui,-apple-system,\"Segoe UI\",Roboto,sans-serif;"
            f"margin-bottom:6px'>{label}</div>")


def _toggle_view(view: str):
    # "Detailed" unlocks the full institutional tab set; "Simple" is the subscriber view.
    is_detailed = view == "Detailed"
    return gr.update(visible=not is_detailed), gr.update(visible=is_detailed)


# ── Build UI ──────────────────────────────────────────────────────────────────

# Explicit font: one real web font + a plain generic fallback. Avoids the woff2
# 404s Gradio emits when its default theme tries to self-host "system-ui" /
# "ui-sans-serif" as font files.
_theme = gr.themes.Base(
    primary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)
_gr_major = int(gr.__version__.split(".")[0])

# Mobile-friendly tweaks. IMPORTANT: scope custom table CSS to our own HTML
# tables only (.cf-table) — a global `table { display:block }` rule breaks
# Gradio's DataFrame layout and leaves empty gaps. Our custom HTML tables carry
# their own overflow wrapper, so no global override is needed.
_CSS = """
/* ── Layout ───────────────────────────────────────────────────────────────── */
.gradio-container { max-width: 1200px !important; margin: auto !important; }

/* ── Unified table look (markdown tables, Gradio DataFrames, custom HTML) ──── */
/* Only safe, non-layout properties so component grids aren't broken.           */
.gradio-container thead th {
  background: #f1f5f9 !important;
  color: #475569 !important;
  font-weight: 600 !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: .03em !important;
  padding: 8px 10px !important;
  border-bottom: 2px solid #e2e8f0 !important;
}
.gradio-container tbody td { padding: 7px 10px !important; border-bottom: 1px solid #eef2f6 !important; }
.gradio-container tbody tr:nth-child(even) { background: #fafbfc !important; }
.gradio-container tbody tr:hover { background: #eef6fb !important; }

/* Numeric data grids: right-align everything except the first (label) column. */
.num-table td:not(:first-child), .num-table th:not(:first-child) { text-align: right !important; }

/* Custom HTML tables carry their own horizontal-scroll wrapper. */
.cf-table { overflow-x: auto; }

@media (max-width: 640px) {
  .gradio-container { padding: 4px !important; }
  .gradio-container table { font-size: 12px !important; }
  h1 { font-size: 1.3rem !important; }
}
"""

_GLOSSARY = """
**Portfolio Value** — your total account value (cash + the market value of all positions).
**Day / Week P&L** — profit or loss today / this week, shown in dollars and percent.
**Return vs S&P 500** — how the bot is doing compared to simply buying the index.
**Unrealized P&L** — paper gain/loss on positions you still hold (not yet sold).
**Macro Score** — a 0–1 read on overall market conditions (higher = more favorable); the bot trades smaller when it's low.
**Regime** — the detected market state: Trending Up/Down, Ranging, or Volatile.
**Sharpe Ratio** — return earned per unit of risk; higher is better (above 1 is good). Shows "n/a" until there's enough history.
**Max Drawdown** — the largest peak-to-trough drop, i.e. how bad it got at the worst point.
**PDT (Pattern Day Trader)** — a FINRA rule: no more than 3 day trades in any rolling 5-business-day window for accounts under $25k. The "Day Trades Used" counter shows your total in the current 5-day window, not just today.
**Stop-loss / Trailing stop / Take-profit** — automatic exit rules that protect capital and lock in gains.
**ATR** — a measure of a stock's volatility, used to size the stop-loss distance.
"""

_REPO_URL = "https://github.com/sridhar7707/ai-trading-bot"
_ABOUT = f"""
An automated **paper-trading** system — simulated capital, no real money. Market
data and order execution use Alpaca's paper-trading API.

**📦 Source code:** [{_REPO_URL.replace('https://', '')}]({_REPO_URL})

**How a trade is made (every ~5 min during market hours):**
1. **Features & regime** — compute technical indicators per symbol and classify the
   market regime (Trending / Ranging / Volatile).
2. **Signal ensemble** — score price direction with **XGBoost + LSTM + FinBERT
   sentiment**, combined into one weighted ensemble score.
3. **Risk gate (hard-coded, the model can't bypass it)** — position sizing via the
   Kelly fraction, stop-loss / trailing-stop / take-profit, daily & weekly loss
   limits, portfolio drawdown cap, sector concentration, wash-sale and **PDT**
   compliance.
4. **Execute** — only orders that pass every gate are sent to Alpaca; each fill is
   logged with its full signal audit trail.

**Simulated vs. real:** trades are simulated (paper); the market data, prices, and
brokerage mechanics are real. An emergency-halt file can pause all trading instantly.
"""

with gr.Blocks(
    title="Trading Bot Dashboard",
    css=_CSS,
    **({} if _gr_major >= 6 else {"theme": _theme}),
) as demo:

    gr.Markdown("# 📊 AI Trading Bot Dashboard")
    gr.HTML(
        "<div style='background:#e65100;color:#fff;padding:5px 14px;border-radius:4px;"
        "display:inline-block;font-size:12px;font-weight:bold;margin-bottom:8px'>"
        "PAPER TRADING — Simulated capital only. No real money at risk.</div>"
    )

    with gr.Row():
        tier_radio = gr.Radio(
            choices=["Simple", "Detailed"],
            value="Simple",
            label="View",
            info="Simple = portfolio, positions, trades. Detailed = + performance, signals, audit, compliance.",
            interactive=True,
        )

    # About panel — the engineering story + repo link, one click from the top so a
    # reviewer sees what's simulated vs real and how signals flow to orders.
    with gr.Accordion("📖 About this project (architecture & source)", open=False):
        gr.Markdown(_ABOUT)

    # Plain-language glossary so the jargon (Sharpe, Macro, Regime, PDT, ATR…)
    # is one click away from anywhere on the dashboard.
    with gr.Accordion("ℹ️ What do these terms mean?", open=False):
        gr.Markdown(_GLOSSARY)

    # ═══════════════════════════════════════════════════════════════════════════
    # SCREEN 1 — Subscriber
    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Column(visible=True) as sub_screen:

        with gr.Tabs():

            # ── Overview + Emergency Halt ─────────────────────────────────────
            with gr.TabItem("🏠 Overview"):
                gr.HTML(_section("Portfolio snapshot · Bot status · Today's activity · Emergency halt"))
                with gr.Row():
                    with gr.Column(scale=3):
                        s_overview = gr.Markdown("*Loading...*")
                    with gr.Column(scale=1):
                        s_halt_status  = gr.HTML("*Loading...*")
                        s_halt_btn     = gr.Button("Loading...", variant="stop", size="sm")
                        s_halt_msg     = gr.Markdown("")
                        gr.Markdown(
                            "<span style='color:#666;font-size:11px'>"
                            "Creates/removes `data/HALT_TRADING`. Bot checks this file at each cycle.</span>"
                        )
                gr.HTML(_section("Safety limits — how much of the daily/weekly loss budget has been used (green = safe)"))
                s_risk_gauges = gr.HTML("*Loading...*")
                s_refresh_ov = gr.Button("🔄 Refresh", size="sm")

            # ── Positions ─────────────────────────────────────────────────────
            with gr.TabItem("📂 Positions"):
                gr.HTML(_section("Currently held positions — shares, entry, live price, unrealized $ and %"))
                s_positions = gr.DataFrame(interactive=False, elem_classes=["num-table"])
                gr.HTML(_section("Holdings & Returns — invested vs total return for every stock, open and sold"))
                s_returns = gr.DataFrame(interactive=False, elem_classes=["num-table"])
                s_refresh_pos = gr.Button("🔄 Refresh", size="sm")

            # ── Trade Log ─────────────────────────────────────────────────────
            with gr.TabItem("📋 Trade Log"):
                gr.HTML(_section(
                    "Last 200 trades · "
                    "<span style='background:#388e3c;color:#fff;padding:1px 5px;border-radius:3px'>BUY</span> &nbsp;"
                    "<span style='background:#1565c0;color:#fff;padding:1px 5px;border-radius:3px'>SELL signal</span> &nbsp;"
                    "<span style='background:#00838f;color:#fff;padding:1px 5px;border-radius:3px'>Take-profit</span> &nbsp;"
                    "<span style='background:#e65100;color:#fff;padding:1px 5px;border-radius:3px'>Trailing stop</span> &nbsp;"
                    "<span style='background:#b71c1c;color:#fff;padding:1px 5px;border-radius:3px'>Stop-loss</span>"
                ))
                s_days_slider  = gr.Slider(7, 90, value=30, step=7, label="Days back")
                s_trades_table = gr.HTML()
                s_refresh_tl   = gr.Button("🔄 Refresh", size="sm")

    # ═══════════════════════════════════════════════════════════════════════════
    # SCREEN 2 — Institutional / Enterprise
    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Column(visible=False) as inst_screen:

        with gr.Tabs():

            # ── Overview + Emergency Halt ─────────────────────────────────────
            with gr.TabItem("🏠 Overview"):
                gr.HTML(_section("Portfolio snapshot · Bot status · Today's activity · Emergency halt"))
                with gr.Row():
                    with gr.Column(scale=3):
                        i_overview = gr.Markdown("*Loading...*")
                    with gr.Column(scale=1):
                        i_halt_status  = gr.HTML("*Loading...*")
                        i_halt_btn     = gr.Button("Loading...", variant="stop", size="sm")
                        i_halt_msg     = gr.Markdown("")
                        gr.Markdown(
                            "<span style='color:#666;font-size:11px'>"
                            "Creates/removes `data/HALT_TRADING`. Bot checks this file at each cycle.</span>"
                        )
                i_refresh_ov = gr.Button("🔄 Refresh", size="sm")

            # ── Positions ─────────────────────────────────────────────────────
            with gr.TabItem("📂 Positions"):
                gr.HTML(_section("Currently held positions — shares, entry, live price, unrealized $ and %"))
                i_positions   = gr.DataFrame(interactive=False, elem_classes=["num-table"])
                gr.HTML(_section("Holdings & Returns — invested vs total return for every stock, open and sold"))
                i_returns     = gr.DataFrame(interactive=False, elem_classes=["num-table"])
                i_refresh_pos = gr.Button("🔄 Refresh", size="sm")

            # ── Trade Log ─────────────────────────────────────────────────────
            with gr.TabItem("📋 Trade Log"):
                gr.HTML(_section(
                    "Last 200 trades · "
                    "<span style='background:#388e3c;color:#fff;padding:1px 5px;border-radius:3px'>BUY</span> &nbsp;"
                    "<span style='background:#1565c0;color:#fff;padding:1px 5px;border-radius:3px'>SELL signal</span> &nbsp;"
                    "<span style='background:#00838f;color:#fff;padding:1px 5px;border-radius:3px'>Take-profit</span> &nbsp;"
                    "<span style='background:#e65100;color:#fff;padding:1px 5px;border-radius:3px'>Trailing stop</span> &nbsp;"
                    "<span style='background:#b71c1c;color:#fff;padding:1px 5px;border-radius:3px'>Stop-loss</span>"
                ))
                i_days_slider  = gr.Slider(7, 90, value=30, step=7, label="Days back")
                i_trades_table = gr.HTML()
                i_refresh_tl   = gr.Button("🔄 Refresh", size="sm")

            # ── Performance ───────────────────────────────────────────────────
            with gr.TabItem("📈 Performance"):
                gr.HTML(_section("Portfolio chart · Sharpe · Win rate · Drawdown · Monthly returns"))
                i_perf_days    = gr.Slider(14, 120, value=60, step=7, label="Analysis window (days)")
                i_perf_metrics = gr.Markdown("*Loading...*")
                i_perf_chart   = gr.Plot()
                i_monthly_plot = gr.Plot()
                i_refresh_pf   = gr.Button("🔄 Refresh", size="sm")

            # ── Signal Analysis ───────────────────────────────────────────────
            with gr.TabItem("🔬 Signals"):
                gr.HTML(_section(
                    "Pre-market screener picks · "
                    "<b>Score</b>: composite factor rank 0–1 (higher = better) · "
                    "<b>Analyst</b>: Finnhub upgrade/downgrade signal −1 to +1 · "
                    "<b>ETF Mom</b>: sector ETF vs 20-day SMA"
                ))
                i_screener = gr.DataFrame(
                    label="Today's screened universe (ranked by composite score)",
                    interactive=False, elem_classes=["num-table"],
                )
                gr.HTML(_section(
                    "Live model output · Updated every bot cycle · "
                    "<b>XGB / LSTM / Score</b>: 0–1 probability &nbsp;|&nbsp; "
                    "<b>Sentiment</b>: −1 (negative) to +1 (positive) · "
                    "<b>Macro</b>: 0–1 market-condition score"
                ))
                i_sig_live  = gr.DataFrame(
                    label="Latest signals (most recent cycle per symbol)",
                    interactive=False, elem_classes=["num-table"],
                )
                i_refresh_sg = gr.Button("🔄 Refresh", size="sm")
                gr.HTML(_section("XGB · LSTM · Sentiment · Ensemble score distributions for BUY entries"))
                i_sig_days  = gr.Slider(7, 90, value=30, step=7, label="Days back")
                i_sig_chart = gr.Plot()

            # ── Go-Live Check ─────────────────────────────────────────────────
            with gr.TabItem("🚀 Go-Live Check"):
                gr.HTML(_section("6-gate confidence check — same logic as confidence_check.py"))
                i_readiness_md  = gr.Markdown("Click **Run Check** to evaluate readiness.")
                i_run_check_btn = gr.Button("▶ Run Check", variant="primary")

            # ── Audit Trail ───────────────────────────────────────────────────
            with gr.TabItem("🗂 Audit Trail"):
                gr.HTML(_section("Full signal audit per trade — XGB · LSTM · Sentiment · Macro · Ensemble"))
                i_audit_days  = gr.Slider(7, 90, value=60, step=7, label="Days back")
                i_audit_table = gr.DataFrame(interactive=False, elem_classes=["num-table"])
                i_audit_dl    = gr.DownloadButton("⬇ Export CSV", visible=False)
                i_refresh_au  = gr.Button("🔄 Refresh", size="sm")

            # ── Compliance gauges ─────────────────────────────────────────────
            with gr.TabItem("⚖ Compliance"):
                gr.HTML(_section("Daily · Weekly · PDT limits — live visual gauges with traffic-light colours"))
                i_compliance  = gr.HTML("*Loading...*")
                i_refresh_cp  = gr.Button("🔄 Refresh", size="sm")

    # ── Named helpers (no lambdas — Gradio 5.x lambda API conflicts) ─────────────
    def _overview_with_benchmark():
        d = get_overview()
        # Best-effort S&P comparison (network, Space-only, cached daily)
        d["spy_return"] = spy_return_since(d.get("inception_date"))
        return overview_md(d)

    def _load_positions():
        # One handler returns both tables → avoids binding two handlers to the
        # same load/refresh (which triggers Gradio's "too many arguments" warning).
        return get_positions_df(), get_returns_summary_df()

    def _load_ov():
        refresh_db_from_hf()          # on auto-load: respect 5-min cache
        s, b = halt_status_html()
        return _overview_with_benchmark(), s, b

    def _force_refresh_ov():
        refresh_db_from_hf(force=True)  # on manual Refresh: bypass cache, always re-pull
        s, b = halt_status_html()
        return _overview_with_benchmark(), s, b

    def _do_halt():
        s, b, msg = toggle_halt()
        return s, b, msg

    def _s_trades_default():    return trades_html_table(30)
    def _s_trades_slider(d):    return trades_html_table(int(d))
    def _i_trades_default():    return trades_html_table(30)
    def _i_trades_slider(d):    return trades_html_table(int(d))

    def _perf(d):
        d = int(d)
        return performance_md(get_performance_metrics(d)), portfolio_chart(d), monthly_chart(d * 3)

    def _perf_default():        return _perf(60)
    def _sig_live():             return get_latest_signals_df()
    def _screener():            return get_screener_df()
    def _sig_default():         return signals_chart(30)
    def _sig_slider(d):         return signals_chart(int(d))

    def _audit(d):
        df   = get_audit_df(int(d))
        path = "/tmp/audit_export.csv"
        if not df.empty:
            df.to_csv(path, index=False)
        return df, gr.update(value=path if not df.empty else None, visible=not df.empty)

    def _audit_default():       return _audit(60)
    def _comp():                return compliance_gauges_html(get_compliance_state())

    _kw = {"api_name": False}   # suppress Gradio 5.x API-name conflicts

    # ── Tier toggle ───────────────────────────────────────────────────────────
    tier_radio.change(_toggle_view, inputs=tier_radio, outputs=[sub_screen, inst_screen], **_kw)

    # ── Subscriber wiring ─────────────────────────────────────────────────────
    demo.load(_load_ov,          outputs=[s_overview, s_halt_status, s_halt_btn], **_kw)
    demo.load(_comp,             outputs=s_risk_gauges, **_kw)
    demo.load(_load_positions,   outputs=[s_positions, s_returns], **_kw)
    demo.load(_s_trades_default, outputs=s_trades_table, **_kw)
    s_refresh_ov.click(_force_refresh_ov, outputs=[s_overview, s_halt_status, s_halt_btn], **_kw)
    s_refresh_ov.click(_comp,            outputs=s_risk_gauges, **_kw)
    s_halt_btn.click(_do_halt,           outputs=[s_halt_status, s_halt_btn, s_halt_msg], **_kw)
    s_refresh_pos.click(_load_positions, outputs=[s_positions, s_returns], **_kw)
    s_refresh_tl.click(_s_trades_slider,  inputs=s_days_slider, outputs=s_trades_table, **_kw)
    s_days_slider.change(_s_trades_slider, inputs=s_days_slider, outputs=s_trades_table, **_kw)

    # ── Institutional wiring ──────────────────────────────────────────────────
    demo.load(_load_ov,          outputs=[i_overview, i_halt_status, i_halt_btn], **_kw)
    demo.load(_load_positions,   outputs=[i_positions, i_returns], **_kw)
    demo.load(_i_trades_default, outputs=i_trades_table, **_kw)
    i_refresh_ov.click(_force_refresh_ov,  outputs=[i_overview, i_halt_status, i_halt_btn], **_kw)
    i_halt_btn.click(_do_halt,            outputs=[i_halt_status, i_halt_btn, i_halt_msg], **_kw)
    i_refresh_pos.click(_load_positions, outputs=[i_positions, i_returns], **_kw)
    i_refresh_tl.click(_i_trades_slider,   inputs=i_days_slider, outputs=i_trades_table, **_kw)
    i_days_slider.change(_i_trades_slider, inputs=i_days_slider, outputs=i_trades_table, **_kw)

    demo.load(_perf_default,  outputs=[i_perf_metrics, i_perf_chart, i_monthly_plot], **_kw)
    i_refresh_pf.click(_perf, inputs=i_perf_days, outputs=[i_perf_metrics, i_perf_chart, i_monthly_plot], **_kw)
    i_perf_days.change(_perf, inputs=i_perf_days, outputs=[i_perf_metrics, i_perf_chart, i_monthly_plot], **_kw)

    demo.load(_screener,     outputs=i_screener,  **_kw)
    demo.load(_sig_live,     outputs=i_sig_live,  **_kw)
    demo.load(_sig_default,  outputs=i_sig_chart, **_kw)
    i_refresh_sg.click(_screener,     outputs=i_screener,  **_kw)
    i_refresh_sg.click(_sig_live,    outputs=i_sig_live,  **_kw)
    i_refresh_sg.click(_sig_slider,  inputs=i_sig_days, outputs=i_sig_chart, **_kw)
    i_sig_days.change(_sig_slider,   inputs=i_sig_days, outputs=i_sig_chart, **_kw)

    i_run_check_btn.click(run_readiness_check, outputs=i_readiness_md, **_kw)

    demo.load(_audit_default, outputs=[i_audit_table, i_audit_dl], **_kw)
    i_refresh_au.click(_audit, inputs=i_audit_days, outputs=[i_audit_table, i_audit_dl], **_kw)
    i_audit_days.change(_audit, inputs=i_audit_days, outputs=[i_audit_table, i_audit_dl], **_kw)

    demo.load(_comp,           outputs=i_compliance, **_kw)
    i_refresh_cp.click(_comp,  outputs=i_compliance, **_kw)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    # Tick every 60s so the live "where am I" views update without a manual click.
    # Only the cheap DB-backed surfaces (overview, halt, risk gauges) are auto-
    # refreshed; positions intentionally aren't (each pull hits yfinance). The
    # DB pull inside _con() respects its 5-min cache, so a tick is cheap.
    if hasattr(gr, "Timer"):
        _auto = gr.Timer(60)
        _auto.tick(_load_ov, outputs=[s_overview, s_halt_status, s_halt_btn], **_kw)
        _auto.tick(_comp,    outputs=s_risk_gauges, **_kw)
        _auto.tick(_load_ov, outputs=[i_overview, i_halt_status, i_halt_btn], **_kw)
        _auto.tick(_comp,    outputs=i_compliance, **_kw)


if __name__ == "__main__":
    _launch_kw = {"theme": _theme} if _gr_major >= 6 else {}
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, **_launch_kw)
