"""Model validation, investor view, and institutional metrics."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    GAIN, LOSS, NEURAL,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, SECTION_GAP,
    _card, _label, _section_title, _action_badge, _symbol,
    _metric_row, _divider, _empty_state, _section, _wrap,
    _sym, _badge, _num, _pnl, _stat_card, TH, TD, TD0,
)
import pandas as pd
from dashboard.data import get_data, _to_ct
from bot.core.error_logger import safe_render
import os
_logger = logger

@safe_render("Validation Report")
def render_validation_report() -> str:
    import json as _json
    vr_path = "models/validation_report.json"
    if not os.path.exists(vr_path):
        msg = (f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:{FONT_LABEL};">'
               f'No validation report yet.<br>Run: python scripts/train_model.py</div>')
        return f'<div class="nt nt-wrap">{_section("🔬", "Model Validation")}{_wrap(msg)}</div>'
    try:
        with open(vr_path) as fh:
            r = _json.load(fh)
    except Exception as exc:
        return (f'<div class="nt nt-wrap">{_section("🔬", "Model Validation")}'
                f'<span style="color:{LOSS}">{exc}</span></div>')

    auc      = r.get("xgb_val_auc",  0.0)
    val_loss = r.get("lstm_val_loss", 1.0)
    auc_c    = GAIN if auc >= 0.60 else (NEURAL if auc >= 0.55 else LOSS)
    loss_c   = GAIN if val_loss < 0.65 else (NEURAL if val_loss < 0.70 else LOSS)
    dr       = r.get("date_range", {})

    def _vr(label: str, val: str, color: str = TEXT1) -> str:
        return (
            f'<tr>'
            f'<td style="padding:9px 14px;border-bottom:1px solid {BORDER};background:{SURFACE};'
            f'color:{TEXT2};font-size:{FONT_LABEL};font-weight:600;">{label}</td>'
            f'<td style="padding:9px 14px;border-bottom:1px solid {BORDER};background:{SURFACE};'
            f'font-family:-apple-system,monospace;color:{color};font-weight:700;">{val}</td>'
            f'</tr>'
        )

    rows = (
        _vr("XGB Val AUC",    f"{auc:.3f}",                    auc_c)
        + _vr("LSTM Val Loss",f"{val_loss:.4f}",               loss_c)
        + _vr("Train Rows",   f"{r.get('training_rows',0):,}")
        + _vr("Symbols",      str(r.get("training_symbols","—")))
        + _vr("Cutoff",       r.get("train_cutoff", "—"))
        + _vr("Data From",    dr.get("from", "—"))
        + _vr("Data To",      dr.get("to",   "—"))
        + _vr("Generated",    r.get("generated_at","")[:10])
    )
    help_html = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:10px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.6;">'
        f'<strong style="color:{TEXT1};">How to read this:</strong><br>'
        f'<b>XGB Val AUC</b> — How well XGBoost predicts the right direction on data it '
        f'<em>never trained on</em>. 0.50 = random guessing. 0.60+ = meaningfully predictive. '
        f'1.0 = perfect (never achieved in practice).<br>'
        f'<b>LSTM Val Loss</b> — Prediction error on unseen data. Lower is better. '
        f'A random classifier scores ~0.69; a well-trained model scores below 0.65.<br>'
        f'<b>Train Cutoff</b> — All data <em>after</em> this date was held out during training '
        f'to test real-world performance.'
        f'</div>'
    )
    table = _wrap(f'<table class="nt-tbl" style="width:100%">{rows}</table>' + help_html)
    note = (f'<div style="font-size:{FONT_LABEL};color:{TEXT2};padding:2px 0 6px;">'
            f'AUC ≥ 0.60 = good · ≥ 0.55 = acceptable · &lt; 0.52 = near-random</div>')
    return f'<div class="nt nt-wrap">{_section("🔬", "Model Validation")}{note}{table}</div>'


# ── Render: institutional metrics ─────────────────────────────────────────────
@safe_render("Institutional Metrics")
def render_institutional_metrics() -> str:
    d  = get_data()
    df = d["trades_df"]

    if df.empty or "portfolio_value" not in df.columns:
        msg = f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:{FONT_LABEL};">No trade history yet.</div>'
        return f'<div class="nt nt-wrap">{_section("📐","Institutional Metrics")}{_wrap(msg)}</div>'

    daily = (df.dropna(subset=["portfolio_value"])
               .groupby("date")["portfolio_value"].last()
               .reset_index()
               .sort_values("date"))
    daily.columns = ["date", "value"]

    if len(daily) < 3:
        msg = f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:{FONT_LABEL};">Need ≥ 3 days of history.</div>'
        return f'<div class="nt nt-wrap">{_section("📐","Institutional Metrics")}{_wrap(msg)}</div>'

    rets   = daily["value"].pct_change().dropna()
    mean_r = float(rets.mean())
    std_r  = float(rets.std())

    # Sharpe (annualised, 252 trading days)
    sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0

    # Sortino (downside std only)
    neg_rets = rets[rets < 0]
    down_std = float(neg_rets.std()) if len(neg_rets) > 1 else std_r
    sortino  = (mean_r / down_std * (252 ** 0.5)) if down_std > 0 else 0.0

    # Max drawdown
    vals  = daily["value"]
    peak  = vals.cummax()
    max_dd = float(((peak - vals) / peak.replace(0, float("nan"))).max())

    # CAGR
    n_days  = (pd.to_datetime(daily["date"].iloc[-1]) - pd.to_datetime(daily["date"].iloc[0])).days
    start_v = float(daily["value"].iloc[0])
    end_v   = float(daily["value"].iloc[-1])
    cagr    = ((end_v / start_v) ** (365.0 / n_days) - 1) if n_days > 0 and start_v > 0 else 0.0

    # Calmar
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    # VaR 95% (1-day)
    var_95 = float(rets.quantile(0.05)) if len(rets) >= 5 else 0.0

    # Win rate
    sells    = df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")]
    win_rate = float((sells["pnl_pct"] > 0).sum() / len(sells)) if len(sells) > 0 else 0.0

    def _row(label, val_str, color, desc):
        return (
            f'<tr><td style="padding:10px 14px;border-bottom:1px solid {BORDER};'
            f'background:{SURFACE};color:{TEXT2};font-size:{FONT_LABEL};font-weight:600;">{label}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {BORDER};'
            f'background:{SURFACE};font-family:-apple-system,monospace;'
            f'color:{color};font-weight:700;">{val_str}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {BORDER};'
            f'background:{SURFACE};color:{TEXT2};font-size:{FONT_LABEL};">{desc}</td></tr>'
        )

    sh_c = GAIN if sharpe > 1 else (NEURAL if sharpe > 0.5 else LOSS)
    so_c = GAIN if sortino > 1.5 else (NEURAL if sortino > 0.8 else LOSS)
    dd_c = GAIN if max_dd < 0.05 else (NEURAL if max_dd < 0.12 else LOSS)
    ca_c = GAIN if calmar > 2 else (NEURAL if calmar > 1 else LOSS)
    vr_c = GAIN if var_95 > -0.02 else (NEURAL if var_95 > -0.04 else LOSS)
    wr_c = GAIN if win_rate > 0.55 else (NEURAL if win_rate > 0.45 else LOSS)

    rows = (
        _row("Sharpe Ratio",    f"{sharpe:.2f}",  sh_c, ">1.0 = good · >2.0 = excellent")
        + _row("Sortino Ratio", f"{sortino:.2f}", so_c, "Like Sharpe but penalises only downside vol")
        + _row("Max Drawdown",  f"{max_dd:.1%}",  dd_c, "Worst peak-to-trough in account history")
        + _row("CAGR",          f"{cagr:.1%}",    (GAIN if cagr > 0.15 else (NEURAL if cagr > 0 else LOSS)),
               "Compound Annual Growth Rate over tracked period")
        + _row("Calmar Ratio",  f"{calmar:.2f}",  ca_c, "CAGR ÷ max drawdown — higher is better")
        + _row("VaR (95%, 1d)", f"{var_95:.2%}",  vr_c, "Worst expected 1-day loss at 95% confidence")
        + _row("Win Rate",      f"{win_rate:.1%}", wr_c, "% of closed trades that returned a profit")
    )
    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.6;">'
        f'Metrics computed from all trade history since launch. '
        f'Short history (&lt;30 days) may produce unreliable Sharpe / Sortino estimates.'
        f'</div>'
    )
    n_str = f"{n_days} days of history" if n_days > 0 else "—"
    table = _wrap(f'<table class="nt-tbl" style="width:100%">{rows}</table>' + help_block)
    return (f'<div class="nt nt-wrap">'
            f'{_section("📐","Institutional Metrics", n_str)}{table}</div>')


# ── Render: AI decision feed (trade timeline) ────────────────────────────────
# ── Render: investor view (plain-language Models tab) ────────────────────────
@safe_render("Investor View")
def render_investor_view() -> str:
    d  = get_data()
    df = d["trades_df"]
    if df.empty:
        msg = (f'<div style="color:{TEXT2};text-align:center;padding:32px;font-size:{FONT_VALUE};">'
               f'No trade history yet.</div>')
        return f'<div class="nt nt-wrap">{_section("🤖","AI Performance","investor summary")}{_wrap(msg)}</div>'

    sells  = df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")]
    n_s    = len(sells)
    wins   = sells[sells["pnl_pct"] > 0]  if n_s > 0 else pd.DataFrame()
    losses = sells[sells["pnl_pct"] <= 0] if n_s > 0 else pd.DataFrame()
    wr     = len(wins) / n_s if n_s > 0 else 0.0
    avg_w  = float(wins["pnl_pct"].mean()   * 100) if len(wins)   > 0 else 0.0
    avg_l  = float(losses["pnl_pct"].mean() * 100) if len(losses) > 0 else 0.0
    rr     = abs(avg_w / avg_l) if avg_l != 0 else 0.0

    cards = (
        f'<div class="nt-cards">'
        + _stat_card("Win Rate",          f"{wr:.0%}" if n_s > 0 else "—",
                TEXT2, GAIN if wr >= 0.55 else (NEURAL if wr >= 0.45 else LOSS),
                f"AI correct {len(wins)} of {n_s} closed trades", 0.0)
        + _stat_card("Avg Winning Trade", f"+{avg_w:.1f}%" if avg_w > 0 else "—",
                TEXT2, GAIN, "Average gain per winning trade", 0.06)
        + _stat_card("Avg Losing Trade",  f"{avg_l:.1f}%"  if avg_l < 0 else "—",
                TEXT2, LOSS, "Average loss per losing trade",  0.12)
        + _stat_card("Risk / Reward",     f"{rr:.1f}×"     if rr > 0   else "—",
                TEXT2, GAIN if rr >= 1.5 else (NEURAL if rr >= 1.0 else LOSS),
                "Avg win ÷ avg loss — >1.5× is good", 0.18)
        + f'</div>'
    )

    # Top buy signals from recent SHAP
    signal_counts: dict[str, int] = {}
    for _, row in df[df["action"] == "BUY"].tail(20).iterrows():
        drv_raw = row.get("feature_drivers")
        if not drv_raw:
            continue
        try:
            import json as _j
            ds = _j.loads(drv_raw) if isinstance(drv_raw, str) else drv_raw
            for feat, val in (ds or []):
                if float(val) > 0:
                    name = _WHY_MAP.get(feat, (_FI_LABELS.get(feat, feat),))[0]
                    signal_counts[name] = signal_counts.get(name, 0) + 1
        except Exception as exc:
            logger.debug(f"parse_signal_counts: {exc}")
    top3 = sorted(signal_counts.items(), key=lambda x: -x[1])[:3]
    sig_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid {BORDER};">'
        f'<span style="font-size:{FONT_VALUE};">📡</span>'
        f'<span style="font-size:{FONT_VALUE};color:{TEXT1};">{name}</span>'
        f'<span style="margin-left:auto;font-size:{FONT_LABEL};color:{TEXT2};">fired {cnt}× recently</span>'
        f'</div>'
        for name, cnt in top3
    ) or f'<div style="color:{TEXT2};font-size:{FONT_LABEL};padding:10px 0;">Building signal history.</div>'

    signals_box = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;margin-top:8px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;">Most Common Buy Signals</div>'
        f'{sig_rows}</div>'
    )

    last6 = sells.tail(6).iloc[::-1]
    result_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid {BORDER};">'
        f'<span style="font-size:{FONT_VALUE};">{"✅" if float(row.get("pnl_pct",0) or 0) > 0 else "❌"}</span>'
        f'<span style="font-family:Courier New,monospace;font-weight:700;color:{PRIMARY};font-size:{FONT_VALUE};">{row.get("symbol","")}</span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{_SELL_REASON.get(str(row.get("action","")),"Exit")}</span>'
        f'<span style="margin-left:auto;font-weight:700;color:{"" + GAIN if float(row.get("pnl_pct",0) or 0) > 0 else LOSS};">'
        f'{float(row.get("pnl_pct",0) or 0):+.1%}</span>'
        f'</div>'
        for _, row in last6.iterrows()
    ) or f'<div style="color:{TEXT2};font-size:{FONT_LABEL};padding:10px 0;">No closed trades yet.</div>'

    results_box = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;margin-top:8px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;">Recent Trade Results</div>'
        f'{result_rows}</div>'
    )
    explain = (
        f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;margin-top:8px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};line-height:1.7;">'
        f'<strong style="color:{TEXT1};">How TradeGenius AI works:</strong><br>'
        f'Three AI models vote on every trade: an <b>XGBoost</b> pattern engine trained on price history, '
        f'an <b>LSTM</b> that reads momentum, and a <b>FinBERT</b> model that reads financial news. '
        f'The AI only buys when all three agree <em>and</em> risk limits, market regime, and '
        f'position sizing rules all pass. Every position has an automatic stop-loss.'
        f'</div></div>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("🤖","AI Performance","investor summary")}'
            f'{cards}{signals_box}{results_box}{explain}</div>')


