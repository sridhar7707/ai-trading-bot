"""Symbol drilldown and why panel."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    PRIMARY,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    GAIN, LOSS, NEURAL, GAIN_BG, LOSS_BG, GAIN_BD,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, SECTION_GAP,
    _card, _label, _section_title, _action_badge, _symbol,
    _confidence_bar, _metric_row, _progress_bar, _divider, _empty_state,
    _action_row, _section, _wrap, _sym, _badge, _stat_card, TH, TD, TD0,
)
import pandas as pd
from dashboard.data import get_data, _to_ct
from dashboard.charts import _get_sym_hist, _sym_perf, _sparkline, _FI_LABELS
from dashboard.components.ai_panel import _WHY_MAP
from bot.core.error_logger import safe_render
from bot.core.recommendation_engine import (
    get_portfolio_action, get_position_sizing, get_sell_analysis,
    get_recommendation_explanation,
)
_logger = logger

# ── Symbol choices + detail drilldown ─────────────────────────────────────────
def _get_symbol_choices() -> list[str]:
    from config import SYMBOLS
    d = get_data()
    # Open positions first (actionable &mdash; AI card is live for these)
    syms = list(d["open_pos"].keys())
    # Then full watchlist for research; closed/sold positions are excluded
    # because the AI action card is suppressed for them anyway
    for s in SYMBOLS:
        if s not in syms:
            syms.append(s)
    return syms


@safe_render("Symbol Detail")
def render_symbol_detail(symbol: str) -> str:
    if not symbol:
        return (f'<div class="nt nt-wrap"><div style="color:{TEXT2};text-align:center;'
                f'padding:20px;font-size:{FONT_LABEL};">Select a symbol above to see its AI analysis.</div></div>')
    d       = get_data()
    df      = d["trades_df"]
    prices  = d["prices"]
    open_pos = d["open_pos"]

    sym_df   = df[df["symbol"] == symbol] if not df.empty else pd.DataFrame()
    buy_df   = sym_df[sym_df["action"] == "BUY"]
    lb       = buy_df.iloc[-1] if not buy_df.empty else None

    cur_price  = prices.get(symbol, 0.0)
    pos        = open_pos.get(symbol)
    has_bot_buy = lb is not None
    # Reconciled: bot sold the position but never recorded a BUY (seeded from Alpaca)
    has_sells   = not sym_df[sym_df["action"].str.startswith("SELL")].empty if not sym_df.empty else False
    is_external = (not has_bot_buy) and has_sells

    conf    = float(lb.get("ensemble_score",  0.0) or 0.0) if lb is not None else 0.0
    xgb_p   = float(lb.get("xgb_prob",        0.0) or 0.0) if lb is not None else 0.0
    lstm_p  = float(lb.get("lstm_prob",        0.0) or 0.0) if lb is not None else 0.0
    sent    = float(lb.get("sentiment_score",  0.0) or 0.0) if lb is not None else 0.0
    regime  = (str(lb.get("regime") or "&mdash;").replace("_", " ").title() if lb is not None
               else d["regime_raw"].title())
    entry   = float(lb.get("price", 0.0) or 0.0) if lb is not None else 0.0
    drv_raw = lb.get("feature_drivers") if lb is not None else None
    ts      = lb.get("timestamp", "") if lb is not None else ""

    conf_c  = GAIN if conf >= 0.75 else (NEURAL if conf >= 0.60 else TEXT2)
    sent_c  = GAIN if sent > 0.05 else (LOSS if sent < -0.05 else TEXT2)
    sent_l  = "Positive" if sent > 0.05 else ("Negative" if sent < -0.05 else "Neutral")
    r_lower = regime.lower()
    r_color = (GAIN if any(x in r_lower for x in ["bull","trending up"]) else
               LOSS if any(x in r_lower for x in ["bear","trending down"]) else NEURAL)

    pnl_str, pnl_c = "&mdash;", TEXT2
    if pos and entry > 0 and cur_price > 0:
        pnl_v   = (cur_price - entry) / entry * 100
        pnl_str = f"{pnl_v:+.1f}%"
        pnl_c   = GAIN if pnl_v >= 0 else LOSS

    if pos:
        status_lbl, status_c = "OPEN POSITION",      GAIN
    elif is_external:
        status_lbl, status_c = "EXTERNAL / RECONCILED", NEURAL
    elif not sym_df.empty:
        status_lbl, status_c = "RECENTLY TRADED",    TEXT2
    else:
        status_lbl, status_c = "NO HISTORY",         TEXT3
    conf_pct   = f"{conf*100:.0f}%" if conf > 0 else "&mdash;"

    # SHAP drivers
    why_html = ""
    try:
        import json as _j
        ds = _j.loads(drv_raw) if isinstance(drv_raw, str) else (drv_raw or [])
        pos_d = sorted([(f, float(v)) for f, v in (ds or []) if float(v) > 0], key=lambda x: -x[1])[:4]
        for feat, _ in pos_d:
            w    = _WHY_MAP.get(feat)
            name = w[0] if w else _FI_LABELS.get(feat, feat)
            desc = w[1] if w else ""
            why_html += (
                f'<div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px solid {BORDER};">'
                f'<span style="color:{GAIN};font-weight:700;width:14px;flex-shrink:0;">+</span>'
                f'<div><div style="font-size:{FONT_LABEL};color:{TEXT1};">{name}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{desc}</div></div></div>'
            )
    except Exception as exc:
        logger.debug(f"build_why_html: {exc}")
    if not why_html:
        if is_external:
            why_html = (
                f'<div style="color:{TEXT2};font-size:{FONT_LABEL};padding:8px 0;">'
                f'This position was <strong>seeded from your Alpaca account</strong> at bot startup '
                f'&mdash; it was not opened by this bot session. No BUY signal or SHAP drivers on record.</div>'
            )
        elif not has_bot_buy:
            why_html = f'<div style="color:{TEXT2};font-size:{FONT_LABEL};padding:8px 0;">No bot BUY on record for this symbol.</div>'
        else:
            why_html = f'<div style="color:{TEXT2};font-size:{FONT_LABEL};padding:8px 0;">SHAP breakdown available after next model retrain.</div>'

    # Mini model bars
    def _mbar(label, v, c):
        return (f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                f'<span style="font-size:{FONT_LABEL};color:{TEXT2};width:60px;">{label}</span>'
                f'<div style="background:{BORDER};border-radius:2px;height:4px;flex:1;">'
                f'<div style="background:{c};height:100%;width:{int(v*100)}%;"></div></div>'
                f'<span style="font-size:{FONT_LABEL};color:{c};width:32px;text-align:right;">{v*100:.0f}%</span></div>')

    model_html = ""
    if xgb_p > 0 or lstm_p > 0:
        xc = GAIN if xgb_p >= 0.70 else (NEURAL if xgb_p >= 0.55 else TEXT2)
        lc = GAIN if lstm_p >= 0.70 else (NEURAL if lstm_p >= 0.55 else TEXT2)
        model_html = (
            f'<div style="margin-top:10px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">Model Scores</div>'
            + _mbar("XGBoost", xgb_p, xc) + _mbar("LSTM", lstm_p, lc)
            + f'</div>'
        )

    # Recent trades for this symbol
    hist_rows = ""
    for _, r in sym_df.tail(5).iloc[::-1].iterrows():
        act = str(r.get("action", ""))
        px  = float(r.get("price", 0) or 0)
        p   = float(r.get("pnl_pct", 0) or 0)
        p_c = GAIN if p > 0 else (LOSS if p < 0 else TEXT2)
        hist_rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid {BORDER};">'
            f'{_badge(act)}'
            f'<span style="font-family:monospace;font-size:{FONT_LABEL};color:{TEXT1};">${px:.2f}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{p_c};margin-left:auto;">'
            f'{f"{p:+.1%}" if p != 0 else ""}</span></div>'
        )

    # Stats grid
    stat_g = (
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;">'
        f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">AI Score</div>'
        f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{conf_c};">{conf_pct}</div></div>'
        f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Sentiment</div>'
        f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{sent_c};">{sent_l}</div></div>'
        f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Regime</div>'
        f'<div style="font-size:{FONT_VALUE};font-weight:700;color:{r_color};">{regime}</div></div>'
        f'</div>'
    )
    pos_g = ""
    if pos:
        cur_str = f"${cur_price:.2f}" if cur_price > 0 else "&mdash;"
        pos_g = (
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;">'
            f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Entry</div>'
            f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{TEXT1};">${entry:.2f}</div></div>'
            f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Current</div>'
            f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{TEXT1};">{cur_str}</div></div>'
            f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Unrealized P&amp;L</div>'
            f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{pnl_c};">{pnl_str}</div></div>'
            f'</div>'
        )

    ts_note = f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:8px;">Signal: {_to_ct(ts)[:16]}</div>' if ts else ""

    # ── SPEC 31: action card ──────────────────────────────────────────────────────
    # ── Why Panel &mdash; recommendation engine signals ──────────────────────────────
    if pos:
        # Live open position &mdash; full recommendation engine output
        _pa  = get_portfolio_action(symbol, d)
        _exp = get_recommendation_explanation(symbol, d)
        _sz2 = get_position_sizing(symbol, d)
        _ac      = _pa.get("action", "HOLD")
        _pa_conf = _pa.get("confidence", 0)
        _pa_reason = _pa.get("reason", "&mdash;")
        _ac_colors = {
            "EXIT":  (LOSS,      "#2a0a0a"),
            "SELL":  (LOSS,      "#2a0a0a"),
            "TRIM":  ("#f59e0b", "#2a1f08"),
            "WATCH": (NEURAL,    "#1a1030"),
            "ADD":   (GAIN,      "#0a2010"),
            "BUY":   (GAIN,      "#0a2010"),
            "HOLD":  (TEXT2,     SURFACE2),
        }
        _ac_c, _ac_bg = _ac_colors.get(_ac, (TEXT2, SURFACE2))
        _bc = GAIN if _pa_conf >= 75 else (NEURAL if _pa_conf >= 60 else TEXT2)

        _bull_items = _exp.get("bullish", [])[:3]
        _bear_items = _exp.get("bearish", [])[:3]
        _bull_html  = "".join(
            f'<div style="font-size:{FONT_LABEL};color:{GAIN};margin-bottom:2px;">+ {b}</div>'
            for b in _bull_items
        ) or f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">No bullish signals</div>'
        _bear_html  = "".join(
            f'<div style="font-size:{FONT_LABEL};color:{LOSS};margin-bottom:2px;">- {b}</div>'
            for b in _bear_items
        ) or ""

        _dol_disp = _sz2.get("dollar_display", "&mdash;")
        _tgt_w    = _sz2.get("target_weight", 0.0)
        _sh = (f"Target {_tgt_w:.0f}% · {_dol_disp}" if _tgt_w > 0 else
               "Max 12% allocation" if _pa_conf >= 75 else
               "Max 8% allocation"  if _pa_conf >= 60 else "Max 5% allocation")

        action_card_html = (
            f'<div style="background:{BG};border-radius:6px;padding:12px 14px;margin-bottom:14px;">'
            f'<div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">'
            f'<div style="flex:0 0 auto;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:4px;">AI Action</div>'
            f'<span style="display:inline-block;background:{_ac_bg};border:1px solid {_ac_c};'
            f'color:{_ac_c};font-size:{FONT_VALUE};font-weight:700;letter-spacing:.5px;'
            f'padding:4px 14px;border-radius:4px;">{_ac}</span>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:4px;max-width:140px;">{_pa_reason}</div>'
            f'</div>'
            f'<div style="flex:1;min-width:160px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:4px;">AI Conviction</div>'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">'
            f'<div style="background:{BORDER};border-radius:2px;height:6px;flex:1;">'
            f'<div style="background:{_bc};height:100%;width:{_pa_conf}%;border-radius:2px;"></div>'
            f'</div><span style="font-size:{FONT_LABEL};font-weight:700;color:{_bc};">{_pa_conf}%</span>'
            f'</div>'
            f'{_bull_html}{_bear_html}'
            f'</div>'
            f'<div style="text-align:right;flex:0 0 auto;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:4px;">Sizing Guidance</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{_sh}</div>'
            f'</div></div></div>'
        )
    else:
        # Position is closed or was never held &mdash; no live AI signal to show
        _close_note = (
            "Position was seeded from Alpaca at startup (external entry). "
            "The bot closed it via stop-loss. No AI signal was generated at entry."
        ) if is_external else "Position is closed. AI signals are only generated for open positions."
        action_card_html = (
            f'<div style="background:{BG};border-radius:6px;padding:12px 14px;margin-bottom:14px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{_close_note}</div>'
            f'</div>'
        )

    card = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-top:3px solid {PRIMARY};border-radius:8px;padding:20px;">'
        f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px;">'
        f'<span style="font-family:Courier New,monospace;font-size:{FONT_HERO};font-weight:700;color:{PRIMARY};letter-spacing:-1px;">{symbol}</span>'
        f'<span style="background:{SURFACE2};border:1px solid {status_c};color:{status_c};'
        f'padding:2px 10px;border-radius:4px;font-size:{FONT_LABEL};font-weight:700;letter-spacing:.5px;">{status_lbl}</span>'
        f'</div>'
        f'{action_card_html}{stat_g}{pos_g}'
        f'<div class="nt-ai-split">'
        f'<div><div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;">Why the AI entered</div>'
        f'{why_html}{model_html}</div>'
        f'<div class="nt-ai-right">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;">Trade History</div>'
        f'{hist_rows}{ts_note}'
        f'</div></div></div>'
    )
    return f'<div class="nt nt-wrap">{_section("🔍", symbol, "AI analysis")}{card}</div>'


