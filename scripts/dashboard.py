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
    get_positions_df,
    trades_html_table,
    get_performance_metrics, performance_md, portfolio_chart, signals_chart, monthly_chart,
    get_audit_df,
    get_compliance_state, compliance_gauges_html,
    halt_status_html, toggle_halt,
    refresh_db_from_hf, diagnostics,
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

def run_readiness_check() -> str:
    try:
        import sqlite3, numpy as np, yfinance as yf
        from collections import defaultdict
        from config import TRADE_DB_PATH
        from datetime import datetime

        con  = sqlite3.connect(TRADE_DB_PATH)
        rows = con.execute(
            "SELECT timestamp, action, pnl_pct, portfolio_value FROM trades ORDER BY timestamp"
        ).fetchall()
        con.close()
        if not rows:
            return "⚠️ No trades in database yet."

        ts_start = datetime.fromisoformat(rows[0][0])
        ts_end   = datetime.fromisoformat(rows[-1][0])
        days     = (ts_end - ts_start).days
        sells    = [r for r in rows if r[1].startswith("SELL")]
        win_rate = sum(1 for r in sells if r[2] > 0) / len(sells) if sells else 0.0
        vals     = np.array([r[3] for r in rows])
        rets     = np.diff(vals) / (vals[:-1] + 1e-8)
        sharpe   = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252 * 78)) if len(rets) > 1 else 0.0
        peak = vals[0]; max_dd = 0.0
        for v in vals:
            peak   = max(peak, v)
            max_dd = max(max_dd, (peak - v) / (peak + 1e-8))
        daily    = defaultdict(list)
        for r in rows:
            daily[r[0][:10]].append(r[2])
        day_rets = [sum(v) for v in daily.values()]
        streak = cur = 0
        for r in day_rets:
            if r < 0: cur += 1; streak = max(streak, cur)
            else: cur = 0
        bot_ret = (vals[-1] - vals[0]) / vals[0] if vals[0] else 0.0
        spy_ret = 0.0
        try:
            spy = yf.download("SPY", start=ts_start.date(), end=ts_end.date(),
                              progress=False, auto_adjust=True)
            if len(spy) > 1:
                spy_ret = float((spy["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0])
        except Exception:
            pass

        T = {"min_days": 60, "min_win_rate": 0.52, "min_sharpe": 1.0,
             "max_drawdown": 0.15, "max_consec_loss": 4}
        gates = [
            ("Days trading",      f"{days}d",          f"≥ {T['min_days']}d",          days >= T["min_days"]),
            ("Win rate",          f"{win_rate:.1%}",   f"≥ {T['min_win_rate']:.0%}",   win_rate >= T["min_win_rate"]),
            ("Sharpe ratio",      f"{sharpe:.2f}",     f"≥ {T['min_sharpe']:.1f}",     sharpe >= T["min_sharpe"]),
            ("Max drawdown",      f"{max_dd:.1%}",     f"≤ {T['max_drawdown']:.0%}",   max_dd <= T["max_drawdown"]),
            ("Max consec losses", str(streak),         f"≤ {T['max_consec_loss']}d",   streak <= T["max_consec_loss"]),
            ("vs S&P 500",        f"{bot_ret:.2%}",    f"> SPY {spy_ret:.2%}",         bot_ret > spy_ret),
        ]
        all_pass = all(g[3] for g in gates)
        header   = "## ✅ ALL CHECKS PASSED — Ready for live trading!\n\n" if all_pass else \
                   "## ❌ NOT READY — Keep paper trading.\n\n"
        rows_md  = "\n".join(
            f"| {'✅' if ok else '❌'} | {name} | {val} | {thr} |"
            for name, val, thr, ok in gates
        )
        return header + f"| | Gate | Value | Threshold |\n|--|------|-------|----------|\n{rows_md}"
    except Exception as e:
        return f"⚠️ Check failed: {e}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(label: str) -> str:
    # Light section header bar — matches the light dashboard theme.
    return (f"<div style='background:#f1f5f9;color:#475569;padding:6px 12px;"
            f"border-left:3px solid #0891b2;border-radius:4px;font-size:12px;"
            f"font-family:system-ui,-apple-system,\"Segoe UI\",Roboto,sans-serif;"
            f"margin-bottom:6px'>{label}</div>")


def _toggle_view(tier: str):
    is_inst = tier == "Institutional"
    return gr.update(visible=not is_inst), gr.update(visible=is_inst)


# ── Build UI ──────────────────────────────────────────────────────────────────

_theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate")
_gr_major = int(gr.__version__.split(".")[0])
with gr.Blocks(
    title="Trading Bot Dashboard",
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
            choices=["Subscriber", "Institutional"],
            value="Subscriber",
            label="Access Tier",
            interactive=True,
        )

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
                s_refresh_ov = gr.Button("🔄 Refresh", size="sm")

            # ── Positions ─────────────────────────────────────────────────────
            with gr.TabItem("📂 Positions"):
                gr.HTML(_section("Currently held positions — entry price, high-water mark, hold duration"))
                s_positions = gr.DataFrame(interactive=False)
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
                gr.HTML(_section("Currently held positions — entry price, high-water mark, hold duration"))
                i_positions   = gr.DataFrame(interactive=False)
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
                gr.HTML(_section("XGB · LSTM · Sentiment · Ensemble score distributions for BUY entries"))
                i_sig_days  = gr.Slider(7, 90, value=30, step=7, label="Days back")
                i_sig_chart = gr.Plot()
                i_refresh_sg = gr.Button("🔄 Refresh", size="sm")

            # ── Go-Live Check ─────────────────────────────────────────────────
            with gr.TabItem("🚀 Go-Live Check"):
                gr.HTML(_section("6-gate confidence check — same logic as confidence_check.py"))
                i_readiness_md  = gr.Markdown("Click **Run Check** to evaluate readiness.")
                i_run_check_btn = gr.Button("▶ Run Check", variant="primary")

            # ── Audit Trail ───────────────────────────────────────────────────
            with gr.TabItem("🗂 Audit Trail"):
                gr.HTML(_section("Full signal audit per trade — XGB · LSTM · Sentiment · Macro · Ensemble"))
                i_audit_days  = gr.Slider(7, 90, value=60, step=7, label="Days back")
                i_audit_table = gr.DataFrame(interactive=False)
                i_audit_dl    = gr.DownloadButton("⬇ Export CSV", visible=False)
                i_refresh_au  = gr.Button("🔄 Refresh", size="sm")

            # ── Compliance gauges ─────────────────────────────────────────────
            with gr.TabItem("⚖ Compliance"):
                gr.HTML(_section("Daily · Weekly · PDT limits — live visual gauges with traffic-light colours"))
                i_compliance  = gr.HTML("*Loading...*")
                i_refresh_cp  = gr.Button("🔄 Refresh", size="sm")

    # ── Named helpers (no lambdas — Gradio 5.x lambda API conflicts) ─────────────
    def _load_ov():
        refresh_db_from_hf()          # on auto-load: respect 5-min cache
        s, b = halt_status_html()
        return overview_md(get_overview()), s, b

    def _force_refresh_ov():
        refresh_db_from_hf(force=True)  # on manual Refresh: bypass cache, always re-pull
        s, b = halt_status_html()
        return overview_md(get_overview()), s, b

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
    demo.load(get_positions_df,  outputs=s_positions, **_kw)
    demo.load(_s_trades_default, outputs=s_trades_table, **_kw)
    s_refresh_ov.click(_force_refresh_ov, outputs=[s_overview, s_halt_status, s_halt_btn], **_kw)
    s_halt_btn.click(_do_halt,           outputs=[s_halt_status, s_halt_btn, s_halt_msg], **_kw)
    s_refresh_pos.click(get_positions_df, outputs=s_positions, **_kw)
    s_refresh_tl.click(_s_trades_slider,  inputs=s_days_slider, outputs=s_trades_table, **_kw)
    s_days_slider.change(_s_trades_slider, inputs=s_days_slider, outputs=s_trades_table, **_kw)

    # ── Institutional wiring ──────────────────────────────────────────────────
    demo.load(_load_ov,          outputs=[i_overview, i_halt_status, i_halt_btn], **_kw)
    demo.load(get_positions_df,  outputs=i_positions, **_kw)
    demo.load(_i_trades_default, outputs=i_trades_table, **_kw)
    i_refresh_ov.click(_force_refresh_ov,  outputs=[i_overview, i_halt_status, i_halt_btn], **_kw)
    i_halt_btn.click(_do_halt,            outputs=[i_halt_status, i_halt_btn, i_halt_msg], **_kw)
    i_refresh_pos.click(get_positions_df,  outputs=i_positions, **_kw)
    i_refresh_tl.click(_i_trades_slider,   inputs=i_days_slider, outputs=i_trades_table, **_kw)
    i_days_slider.change(_i_trades_slider, inputs=i_days_slider, outputs=i_trades_table, **_kw)

    demo.load(_perf_default,  outputs=[i_perf_metrics, i_perf_chart, i_monthly_plot], **_kw)
    i_refresh_pf.click(_perf, inputs=i_perf_days, outputs=[i_perf_metrics, i_perf_chart, i_monthly_plot], **_kw)
    i_perf_days.change(_perf, inputs=i_perf_days, outputs=[i_perf_metrics, i_perf_chart, i_monthly_plot], **_kw)

    demo.load(_sig_default,  outputs=i_sig_chart, **_kw)
    i_refresh_sg.click(_sig_slider, inputs=i_sig_days, outputs=i_sig_chart, **_kw)
    i_sig_days.change(_sig_slider,  inputs=i_sig_days, outputs=i_sig_chart, **_kw)

    i_run_check_btn.click(run_readiness_check, outputs=i_readiness_md, **_kw)

    demo.load(_audit_default, outputs=[i_audit_table, i_audit_dl], **_kw)
    i_refresh_au.click(_audit, inputs=i_audit_days, outputs=[i_audit_table, i_audit_dl], **_kw)
    i_audit_days.change(_audit, inputs=i_audit_days, outputs=[i_audit_table, i_audit_dl], **_kw)

    demo.load(_comp,           outputs=i_compliance, **_kw)
    i_refresh_cp.click(_comp,  outputs=i_compliance, **_kw)


if __name__ == "__main__":
    _launch_kw = {"theme": _theme} if _gr_major >= 6 else {}
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, **_launch_kw)
