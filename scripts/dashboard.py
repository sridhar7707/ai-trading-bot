"""
Trading Bot Dashboard — multi-tier Gradio UI.

Run:  python scripts/dashboard.py
Then open http://localhost:7860

Tier access:
  Subscriber          — Overview (+ halt button), Positions, Trade Log
  Pro Subscriber      — + Performance, Signal Analysis, Readiness Check
  Institutional       — + Audit Trail, Compliance (visual gauges)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from bot.monitor.dashboard_data import (
    get_overview, overview_md,
    get_positions_df,
    trades_html_table,
    get_performance_metrics, performance_md, portfolio_chart, signals_chart, monthly_chart,
    get_audit_df,
    get_compliance_state, compliance_gauges_html,
    halt_status_html, toggle_halt,
)

# ── Confidence check (Pro+) ───────────────────────────────────────────────────

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


# ── Layout helper ─────────────────────────────────────────────────────────────

def _section(label: str) -> str:
    return (f"<div style='background:#111;color:#aaa;padding:4px 10px;"
            f"border-left:3px solid #0af;font-size:12px;margin-bottom:6px'>{label}</div>")


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Trading Bot Dashboard", theme=gr.themes.Base(
    primary_hue="cyan", neutral_hue="slate"
)) as demo:

    gr.Markdown("# 📊 AI Trading Bot Dashboard")

    with gr.Tabs():

        # ── Tab 1: Overview + Emergency Halt (Subscriber+) ────────────────────
        with gr.TabItem("🏠 Overview  [Subscriber+]"):
            gr.HTML(_section("Portfolio snapshot · Bot status · Today's activity · Emergency halt"))
            with gr.Row():
                with gr.Column(scale=3):
                    overview_display = gr.Markdown("*Loading...*")
                with gr.Column(scale=1):
                    halt_status  = gr.HTML("*Loading...*")
                    halt_btn     = gr.Button("Loading...", variant="stop", size="sm")
                    halt_message = gr.Markdown("")
                    gr.Markdown(
                        "<span style='color:#666;font-size:11px'>"
                        "Creates/removes `data/HALT_TRADING`. Bot checks this file at the start of each cycle.</span>"
                    )
            refresh_ov = gr.Button("🔄 Refresh", size="sm")

            def _ov():
                s, b = halt_status_html()
                return overview_md(get_overview()), s, b

            def _halt():
                s, b, msg = toggle_halt()
                return s, b, msg

            demo.load(_ov, outputs=[overview_display, halt_status, halt_btn])
            refresh_ov.click(_ov, outputs=[overview_display, halt_status, halt_btn])
            halt_btn.click(_halt, outputs=[halt_status, halt_btn, halt_message])

        # ── Tab 2: Positions (Subscriber+) ────────────────────────────────────
        with gr.TabItem("📂 Positions  [Subscriber+]"):
            gr.HTML(_section("Currently held positions — entry price, high-water mark, hold duration"))
            positions_table = gr.DataFrame(value=get_positions_df, interactive=False)
            refresh_pos = gr.Button("🔄 Refresh", size="sm")
            refresh_pos.click(get_positions_df, outputs=positions_table)

        # ── Tab 3: Trade Log — color-coded (Subscriber+) ──────────────────────
        with gr.TabItem("📋 Trade Log  [Subscriber+]"):
            gr.HTML(_section(
                "Last 200 trades · "
                "<span style='background:#388e3c;color:#fff;padding:1px 5px;border-radius:3px'>BUY</span> &nbsp;"
                "<span style='background:#1565c0;color:#fff;padding:1px 5px;border-radius:3px'>SELL signal</span> &nbsp;"
                "<span style='background:#00838f;color:#fff;padding:1px 5px;border-radius:3px'>Take-profit</span> &nbsp;"
                "<span style='background:#e65100;color:#fff;padding:1px 5px;border-radius:3px'>Trailing stop</span> &nbsp;"
                "<span style='background:#b71c1c;color:#fff;padding:1px 5px;border-radius:3px'>Stop-loss</span> &nbsp;"
                "<span style='background:#7f0000;color:#fff;padding:1px 5px;border-radius:3px'>Gap-down</span>"
            ))
            days_slider  = gr.Slider(7, 90, value=30, step=7, label="Days back")
            trades_table = gr.HTML()
            refresh_tl   = gr.Button("🔄 Refresh", size="sm")

            demo.load(lambda: trades_html_table(30), outputs=trades_table)
            refresh_tl.click(lambda d: trades_html_table(int(d)), inputs=days_slider, outputs=trades_table)
            days_slider.change(lambda d: trades_html_table(int(d)), inputs=days_slider, outputs=trades_table)

        # ── Tab 4: Performance (Pro+) ─────────────────────────────────────────
        with gr.TabItem("📈 Performance  [Pro+]"):
            gr.HTML(_section("Portfolio chart · Sharpe · Win rate · Drawdown · Monthly returns"))
            perf_days    = gr.Slider(14, 120, value=60, step=7, label="Analysis window (days)")
            perf_metrics = gr.Markdown("*Loading...*")
            perf_chart   = gr.Plot()
            monthly_plot = gr.Plot()
            refresh_pf   = gr.Button("🔄 Refresh", size="sm")

            def _perf(d):
                d = int(d)
                return performance_md(get_performance_metrics(d)), portfolio_chart(d), monthly_chart(d * 3)

            demo.load(lambda: _perf(60), outputs=[perf_metrics, perf_chart, monthly_plot])
            refresh_pf.click(_perf, inputs=perf_days, outputs=[perf_metrics, perf_chart, monthly_plot])
            perf_days.change(_perf, inputs=perf_days, outputs=[perf_metrics, perf_chart, monthly_plot])

        # ── Tab 5: Signal Analysis (Pro+) ─────────────────────────────────────
        with gr.TabItem("🔬 Signals  [Pro+]"):
            gr.HTML(_section("XGB · LSTM · Sentiment · Ensemble score distributions for BUY entries"))
            sig_days  = gr.Slider(7, 90, value=30, step=7, label="Days back")
            sig_chart = gr.Plot()
            refresh_sg = gr.Button("🔄 Refresh", size="sm")

            demo.load(lambda: signals_chart(30), outputs=sig_chart)
            refresh_sg.click(lambda d: signals_chart(int(d)), inputs=sig_days, outputs=sig_chart)
            sig_days.change(lambda d: signals_chart(int(d)), inputs=sig_days, outputs=sig_chart)

        # ── Tab 6: Readiness Check (Pro+) ─────────────────────────────────────
        with gr.TabItem("🚀 Go-Live Check  [Pro+]"):
            gr.HTML(_section("6-gate confidence check — same logic as confidence_check.py"))
            readiness_md  = gr.Markdown("Click **Run Check** to evaluate readiness.")
            run_check_btn = gr.Button("▶ Run Check", variant="primary")
            run_check_btn.click(run_readiness_check, outputs=readiness_md)

        # ── Tab 7: Audit Trail (Institutional) ────────────────────────────────
        with gr.TabItem("🗂 Audit Trail  [Institutional]"):
            gr.HTML(_section("Full signal audit per trade — XGB · LSTM · Sentiment · Macro · Ensemble"))
            audit_days  = gr.Slider(7, 90, value=60, step=7, label="Days back")
            audit_table = gr.DataFrame(interactive=False)
            audit_dl    = gr.DownloadButton("⬇ Export CSV", visible=False)
            refresh_au  = gr.Button("🔄 Refresh", size="sm")

            def _audit(d):
                df   = get_audit_df(int(d))
                path = "/tmp/audit_export.csv"
                if not df.empty:
                    df.to_csv(path, index=False)
                return df, gr.DownloadButton(value=path, visible=not df.empty)

            demo.load(lambda: _audit(60), outputs=[audit_table, audit_dl])
            refresh_au.click(_audit, inputs=audit_days, outputs=[audit_table, audit_dl])
            audit_days.change(_audit, inputs=audit_days, outputs=[audit_table, audit_dl])

        # ── Tab 8: Compliance — visual gauges (Institutional) ─────────────────
        with gr.TabItem("⚖ Compliance  [Institutional]"):
            gr.HTML(_section("Daily · Weekly · PDT limits — live visual gauges with traffic-light colours"))
            compliance_gauges = gr.HTML("*Loading...*")
            refresh_cp = gr.Button("🔄 Refresh", size="sm")

            def _comp():
                return compliance_gauges_html(get_compliance_state())

            demo.load(_comp, outputs=compliance_gauges)
            refresh_cp.click(_comp, outputs=compliance_gauges)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
