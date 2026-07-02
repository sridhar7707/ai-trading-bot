"""Model validation, investor view, and institutional metrics."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    PRIMARY,
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
from dashboard.charts import _FI_LABELS
from dashboard.components.ai_panel import _WHY_MAP
from bot.core.error_logger import safe_render
from dashboard.components.portfolio import _SELL_REASON
import os
import time as _time
from dashboard.data import get_db_conn
_logger = logger

# ── Paper Trading Scorecard ───────────────────────────────────────────────────
_BENCH_CACHE: dict = {}
_BENCH_CACHE_TS: float = 0.0
_BENCH_CACHE_TTL: float = 3600.0


@safe_render("Paper Trading Scorecard")
def render_paper_trading_scorecard() -> str:
    global _BENCH_CACHE, _BENCH_CACHE_TS
    import yfinance as yf

    d  = get_data()
    df = d.get("trades_df", pd.DataFrame())

    if df.empty:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("🏆", "Paper Trading Scorecard", "vs market")}'
            f'{_card(_empty_state("📊", "No history yet", "Scorecard activates after the first trade."))}'
            f'</div>'
        )

    daily = (df.dropna(subset=["portfolio_value"])
               .groupby("date")["portfolio_value"].last()
               .reset_index().sort_values("date"))
    daily.columns = ["date", "value"]

    if len(daily) < 2:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("🏆", "Paper Trading Scorecard", "vs market")}'
            f'{_card(_empty_state("📊", "Need ≥ 2 days", "Building history &mdash; check back tomorrow."))}'
            f'</div>'
        )

    start_date = str(daily["date"].iloc[0])
    bot_start  = float(daily["value"].iloc[0])
    bot_end    = float(daily["value"].iloc[-1])
    bot_ret    = (bot_end / bot_start - 1) if bot_start > 0 else 0.0
    n_days     = (pd.to_datetime(daily["date"].iloc[-1]) - pd.to_datetime(daily["date"].iloc[0])).days

    rets   = daily["value"].pct_change().dropna()
    std_r  = float(rets.std())
    mean_r = float(rets.mean())
    sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0
    peak   = daily["value"].cummax()
    max_dd = float(((peak - daily["value"]) / peak.replace(0, float("nan"))).max())

    sells    = df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")]
    win_rate = float((sells["pnl_pct"] > 0).sum() / len(sells)) if len(sells) > 0 else 0.0

    # AI follow rate: % of BUY recommendations executed as actual BUY trades (last 30 days)
    follow_rate: float | None = None
    try:
        with get_db_conn() as _con:
            _res = _con.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN EXISTS (
                            SELECT 1 FROM trades t
                            WHERE t.symbol = r.symbol
                              AND date(t.timestamp) = r.prediction_date
                              AND t.action = 'BUY'
                          ) THEN 1 ELSE 0 END) AS executed
                   FROM recommendations r
                   WHERE r.recommendation = 'BUY'
                     AND r.prediction_date >= date('now', '-30 days')"""
            ).fetchone()
            if _res and _res[0] > 0:
                follow_rate = float(_res[1]) / float(_res[0])
    except Exception as _fe:
        logger.debug(f"ai_follow_rate: {_fe}")

    # SPY/QQQ returns over the same period (1-hour cache)
    spy_ret = qqq_ret = None
    if _time.time() - _BENCH_CACHE_TS < _BENCH_CACHE_TTL and _BENCH_CACHE:
        spy_ret = _BENCH_CACHE.get("spy")
        qqq_ret = _BENCH_CACHE.get("qqq")
    else:
        try:
            _bd = yf.download(["SPY", "QQQ"], start=start_date, progress=False, auto_adjust=True)
            for sym, key in [("SPY", "spy"), ("QQQ", "qqq")]:
                try:
                    _col = (_bd["Close"][sym] if isinstance(_bd["Close"], pd.DataFrame)
                            else _bd["Close"]).dropna()
                    if len(_col) >= 2:
                        _BENCH_CACHE[key] = float(_col.iloc[-1] / _col.iloc[0] - 1)
                except Exception:
                    pass
            _BENCH_CACHE_TS = _time.time()
            spy_ret = _BENCH_CACHE.get("spy")
            qqq_ret = _BENCH_CACHE.get("qqq")
        except Exception as _be:
            logger.debug(f"benchmark_fetch: {_be}")

    def _vs(bench):
        if bench is None:
            return "&mdash;"
        diff = bot_ret - bench
        return f"{'+'if diff>=0 else ''}{diff:.1%}"

    def _vc(bench):
        return (GAIN if bot_ret >= bench else LOSS) if bench is not None else TEXT2

    bot_c = GAIN if bot_ret > 0 else LOSS
    sh_c  = GAIN if sharpe > 1 else (NEURAL if sharpe > 0.5 else LOSS)
    dd_c  = GAIN if max_dd < 0.05 else (NEURAL if max_dd < 0.12 else LOSS)
    wr_c  = GAIN if win_rate >= 0.55 else (NEURAL if win_rate >= 0.45 else LOSS)
    fr_c  = (GAIN if follow_rate is not None and follow_rate > 0.15 else TEXT2)

    spy_sub = f"SPY: {spy_ret:+.1%}" if spy_ret is not None else "loading…"
    qqq_sub = f"QQQ: {qqq_ret:+.1%}" if qqq_ret is not None else "loading…"

    row1 = (
        f'<div class="nt-cards">'
        + _stat_card("Bot Return",   f"{bot_ret:+.1%}", TEXT2, bot_c,
                     f"since {start_date}", 0.0)
        + _stat_card("vs SPY",       _vs(spy_ret),      TEXT2, _vc(spy_ret), spy_sub, 0.06)
        + _stat_card("vs QQQ",       _vs(qqq_ret),      TEXT2, _vc(qqq_ret), qqq_sub, 0.12)
        + _stat_card("Win Rate",     f"{win_rate:.0%}" if len(sells) > 0 else "&mdash;",
                     TEXT2, wr_c, f"{len(sells)} closed trades", 0.18)
        + f'</div>'
    )
    row2 = (
        f'<div class="nt-cards">'
        + _stat_card("Risk-Adjusted Return", f"{sharpe:.2f}", TEXT2, sh_c,
                     ">1.0 = good · >2.0 = excellent", 0.0)
        + _stat_card("Worst Portfolio Dip", f"{max_dd:.1%}", TEXT2, dd_c,
                     "Biggest drop from peak to bottom", 0.06)
        + _stat_card("Signals Executed",
                     f"{follow_rate:.0%}" if follow_rate is not None else "&mdash;",
                     TEXT2, fr_c, "% of buy signals the bot acted on (30d)", 0.12)
        + _stat_card("Days Running",    str(n_days),       TEXT2, TEXT2,
                     f"since {start_date}", 0.18)
        + f'</div>'
    )
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("🏆", "Paper Trading Scorecard", f"vs SPY · {n_days} days")}'
        f'{row1}{row2}'
        f'</div>'
    )

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
        + _vr("Symbols",      str(r.get("training_symbols","&mdash;")))
        + _vr("Cutoff",       r.get("train_cutoff", "&mdash;"))
        + _vr("Data From",    dr.get("from", "&mdash;"))
        + _vr("Data To",      dr.get("to",   "&mdash;"))
        + _vr("Generated",    r.get("generated_at","")[:10])
    )
    help_html = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:10px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.6;">'
        f'<strong style="color:{TEXT1};">How to read this:</strong><br>'
        f'<b>XGB Val AUC</b> &mdash; How well XGBoost predicts the right direction on data it '
        f'<em>never trained on</em>. 0.50 = random guessing. 0.60+ = meaningfully predictive. '
        f'1.0 = perfect (never achieved in practice).<br>'
        f'<b>LSTM Val Loss</b> &mdash; Prediction error on unseen data. Lower is better. '
        f'A random classifier scores ~0.69; a well-trained model scores below 0.65.<br>'
        f'<b>Train Cutoff</b> &mdash; All data <em>after</em> this date was held out during training '
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
        return f'<div class="nt nt-wrap">{_section("📐","Performance Deep Dive")}{_wrap(msg)}</div>'

    daily = (df.dropna(subset=["portfolio_value"])
               .groupby("date")["portfolio_value"].last()
               .reset_index()
               .sort_values("date"))
    daily.columns = ["date", "value"]

    if len(daily) < 3:
        msg = f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:{FONT_LABEL};">Need ≥ 3 days of history.</div>'
        return f'<div class="nt nt-wrap">{_section("📐","Performance Deep Dive")}{_wrap(msg)}</div>'

    rets   = daily["value"].pct_change().dropna()
    mean_r = float(rets.mean())
    std_r  = float(rets.std())

    sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0

    neg_rets = rets[rets < 0]
    down_std = float(neg_rets.std()) if len(neg_rets) > 1 else std_r
    sortino  = (mean_r / down_std * (252 ** 0.5)) if down_std > 0 else 0.0

    vals  = daily["value"]
    peak  = vals.cummax()
    max_dd = float(((peak - vals) / peak.replace(0, float("nan"))).max())

    n_days  = (pd.to_datetime(daily["date"].iloc[-1]) - pd.to_datetime(daily["date"].iloc[0])).days
    start_v = float(daily["value"].iloc[0])
    end_v   = float(daily["value"].iloc[-1])
    cagr    = ((end_v / start_v) ** (365.0 / n_days) - 1) if n_days > 0 and start_v > 0 else 0.0
    calmar  = (cagr / max_dd) if max_dd > 0 else 0.0

    sells    = df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")]
    n_s      = len(sells)
    win_rate = float((sells["pnl_pct"] > 0).sum() / n_s) if n_s > 0 else 0.0
    pnl_vals = [float(p) for p in sells["pnl_pct"].dropna()] if n_s > 0 else []
    gross_w  = sum(p for p in pnl_vals if p > 0)
    gross_l  = abs(sum(p for p in pnl_vals if p < 0))
    profit_factor: float | None = round(gross_w / gross_l, 2) if gross_l > 0 else None

    total_return = (end_v - start_v) / start_v if start_v > 0 else 0.0
    scratch_count = sum(1 for p in pnl_vals if p == 0)
    alpha_inc: float | None = None
    alpha_60d: float | None = None
    try:
        from bot.monitor.dashboard_data import spy_return_since as _spy_rs
        _spy = _spy_rs(str(daily["date"].iloc[0]))
        if _spy is not None:
            alpha_inc = round(total_return - _spy, 4)
        _s60 = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        _d60 = daily[daily["date"] >= _s60]
        if len(_d60) >= 2:
            _r60 = float(_d60["value"].iloc[-1]) / float(_d60["value"].iloc[0]) - 1
            _spy60 = _spy_rs(_s60)
            if _spy60 is not None:
                alpha_60d = round(_r60 - _spy60, 4)
    except Exception:
        pass

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
    wr_c = GAIN if win_rate > 0.55 else (NEURAL if win_rate > 0.45 else LOSS)
    pf_c     = GAIN if profit_factor is not None and profit_factor > 1.5 else (NEURAL if profit_factor is not None and profit_factor > 1.0 else LOSS)
    al_inc_c = GAIN if alpha_inc is not None and alpha_inc > 0 else (NEURAL if alpha_inc is not None and alpha_inc > -0.02 else LOSS)
    al60_c   = GAIN if alpha_60d is not None and alpha_60d > 0 else (NEURAL if alpha_60d is not None and alpha_60d > -0.02 else LOSS)

    pf_str     = f"{profit_factor:.2f}" if profit_factor is not None else "n/a"
    al_inc_str = f"{alpha_inc:+.2%}" if alpha_inc is not None else "n/a"
    al60_str   = f"{alpha_60d:+.2%}" if alpha_60d is not None else "n/a"

    rows = (
        _row("Return quality vs. risk taken", f"{sharpe:.2f}", sh_c,
             "Sharpe ratio &mdash; how much return per unit of total risk. &gt;1.0 = good, &gt;2.0 = excellent")
        + _row("How well losses are controlled", f"{sortino:.2f}", so_c,
               "Sortino ratio &mdash; like Sharpe but only penalises losing months, not winning volatility")
        + _row("Worst drop from peak", f"{max_dd:.1%}", dd_c,
               "Max drawdown &mdash; biggest drop from peak to trough")
        + _row("Return vs. worst-case loss", f"{calmar:.2f}", ca_c,
               "Calmar ratio &mdash; annualised return divided by max drawdown. Higher = better recovery")
        + _row("Trades that made money", f"{win_rate:.1%}", wr_c,
               "Win rate &mdash; % of closed trades that were profitable (target: &gt;55%)")
        + _row("Gross wins &divide; gross losses", pf_str, pf_c,
               f"Profit factor &mdash; total gains &divide; total losses. &gt;1.5 = good ({scratch_count} scratch)")
        + _row("Alpha vs. S&amp;P 500 (since launch)", al_inc_str, al_inc_c,
               "Inception-to-date outperformance vs. the S&amp;P 500 benchmark")
        + _row("Alpha vs. S&amp;P 500 (rolling 60d)", al60_str, al60_c,
               "Rolling 60-day alpha &mdash; validates whether win rate is beating the market now")
    )
    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.6;">'
        f'All metrics computed from trade history since launch. '
        f'Need at least 30 days of history for reliable estimates &mdash; early numbers will fluctuate.'
        f'</div>'
    )
    n_str = f"{n_days} days of history" if n_days > 0 else "&mdash;"
    table = _wrap(f'<table class="nt-tbl" style="width:100%">{rows}</table>' + help_block)
    return (f'<div class="nt nt-wrap">'
            f'{_section("📐","Performance Deep Dive", n_str)}{table}</div>')


# ── Render: investor view (plain-language Models tab) ────────────────────────
def _model_quality_fallback() -> str:
    """Return a model-quality card when no closed trades exist yet."""
    import json as _json
    vr_path = "models/validation_report.json"
    if not os.path.exists(vr_path):
        return (
            f'<div style="color:{TEXT2};text-align:center;padding:32px;font-size:{FONT_VALUE};">'
            f'Bot is live and scanning markets.<br>'
            f'<span style="font-size:{FONT_LABEL};">Win Rate and P&L will appear after the first completed trade.</span>'
            f'</div>'
        )
    try:
        with open(vr_path) as fh:
            r = _json.load(fh)
    except Exception:
        return (
            f'<div style="color:{TEXT2};text-align:center;padding:32px;font-size:{FONT_VALUE};">'
            f'Bot is live. Awaiting first trade to show performance metrics.</div>'
        )
    auc      = r.get("xgb_val_auc",  0.0)
    val_loss = r.get("lstm_val_loss", 1.0)
    auc_c    = GAIN if auc >= 0.60 else (NEURAL if auc >= 0.55 else LOSS)
    loss_c   = GAIN if val_loss < 0.65 else (NEURAL if val_loss < 0.70 else LOSS)
    generated = r.get("generated_at", "")[:10]

    cards = (
        f'<div class="nt-cards">'
        + _stat_card("Win Rate",          "&mdash;",
                TEXT2, TEXT2, "appears after first closed trade", 0.0)
        + _stat_card("Avg Winning Trade", "&mdash;",
                TEXT2, TEXT2, "appears after first closed trade", 0.06)
        + _stat_card("XGB Model AUC",     f"{auc:.3f}",
                TEXT2, auc_c, "≥0.60 = good · 0.50 = random", 0.12)
        + _stat_card("LSTM Val Loss",     f"{val_loss:.4f}",
                TEXT2, loss_c, "≤0.65 = well-trained · 0.69 = random", 0.18)
        + f'</div>'
    )
    explain = (
        f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;margin-top:8px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};line-height:1.7;">'
        f'<strong style="color:{TEXT1};">Models trained · awaiting first trade</strong><br>'
        f'Models last trained <b>{generated}</b>. The AI is actively scanning all symbols each cycle. '
        f'Win Rate and P&L metrics appear automatically after the first position closes.'
        f'</div></div>'
    )
    return cards + explain


@safe_render("Investor View")
def render_investor_view() -> str:
    d  = get_data()
    df = d["trades_df"]
    if df.empty:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("🤖","AI Performance","investor summary")}'
            f'{_model_quality_fallback()}'
            f'</div>'
        )

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
        + _stat_card("Win Rate",          f"{wr:.0%}" if n_s > 0 else "&mdash;",
                TEXT2, GAIN if wr >= 0.55 else (NEURAL if wr >= 0.45 else LOSS),
                f"AI correct {len(wins)} of {n_s} closed trades", 0.0)
        + _stat_card("Avg Winning Trade", f"+{avg_w:.1f}%" if avg_w > 0 else "&mdash;",
                TEXT2, GAIN, "Average gain per winning trade", 0.06)
        + _stat_card("Avg Losing Trade",  f"{avg_l:.1f}%"  if avg_l < 0 else "&mdash;",
                TEXT2, LOSS, "Average loss per losing trade",  0.12)
        + _stat_card("Risk / Reward",     f"{rr:.1f}×"     if rr > 0   else "&mdash;",
                TEXT2, GAIN if rr >= 1.5 else (NEURAL if rr >= 1.0 else LOSS),
                "Avg win ÷ avg loss &mdash; >1.5× is good", 0.18)
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


