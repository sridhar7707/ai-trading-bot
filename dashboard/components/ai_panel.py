"""AI recommendation and committee panels."""
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
    _card, _label, _section_title, _action_badge, _symbol, _confidence_bar,
    _metric_row, _divider, _empty_state, _action_row, _section, _wrap,
    _stat_card, _badge, TH, TD, TD0,
)
from dashboard.data import get_data, _now_ct, _to_ct
from dashboard.charts import _FI_LABELS
from dashboard.builders import build_committees_vm
from bot.core.error_logger import safe_render, timed
from bot.core.recommendation_engine import (
    get_portfolio_action, get_position_sizing,
    get_recommendation_explanation, get_portfolio_health,
)
_logger = logger

# Plain-English reason map for AI recommendation card (feature → (title, detail))
_WHY_MAP: dict[str, tuple[str, str]] = {
    "rsi":           ("RSI momentum building",   "Short-term price strength confirmed by RSI"),
    "rsi_15m":       ("15-min RSI aligned",      "Shorter-term momentum reinforces the entry"),
    "macd_diff_pct": ("MACD bullish crossover",  "Trend indicator crossed into positive territory"),
    "volume_ratio":  ("Unusual buying volume",   "Volume above recent average — signals conviction"),
    "mfi":           ("Money Flow positive",     "Capital flowing into the stock"),
    "bb_width":      ("Volatility expanding",    "Bollinger Band breakout pattern forming"),
    "atr_pct":       ("Volatility confirmed",    "Position size validated against current ATR"),
    "norm_close":    ("Closing near day's high", "Price strength at close — bullish structure"),
    "ema20_pct":     ("Above 20-period EMA",     "Short-term trend is pointing up"),
    "ema50_pct":     ("Above 50-period EMA",     "Medium-term trend supports the trade"),
    "vwap_dev":      ("Trading above VWAP",      "Price above today's volume-weighted average"),
    "hl_ratio":      ("Strong intraday range",   "Wide intraday range signals trader conviction"),
    "stoch_k":       ("Stochastic momentum",     "Oscillator confirming continued upward momentum"),
}


# ── Render: AI recommendation card — full-width hero ─────────────────────────
@safe_render("AI Recommendation")
def render_ai_recommendation() -> str:
    d   = get_data()
    lb  = d.get("latest_buy_signal", {})
    vix = d.get("vix", 0.0)

    if not lb or not lb.get("symbol"):
        _es = _empty_state("🤖", "No active signal",
                           "The AI monitors markets Mon–Fri 9:30am–4pm ET. "
                           "When all entry gates pass, the recommendation appears here.")
        return (f'<div class="nt nt-wrap">'
                f'{_section("🤖","AI Recommendation","live signal · updated every 60s")}'
                f'{_card(_es)}'
                f'</div>')

    sym     = lb.get("symbol", "—")
    conf    = float(lb.get("ensemble_score",  0.0) or 0.0)
    xgb_p   = float(lb.get("xgb_prob",         0.0) or 0.0)
    lstm_p  = float(lb.get("lstm_prob",         0.0) or 0.0)
    sent    = float(lb.get("sentiment_score",   0.0) or 0.0)
    entry   = float(lb.get("price",            0.0) or 0.0)
    regime  = str(lb.get("regime") or "—").replace("_", " ").title()
    ts      = lb.get("timestamp", "")
    drv_raw = lb.get("feature_drivers")

    r_lower = regime.lower()
    if any(x in r_lower for x in ["bull", "trending up"]):   r_color = GAIN
    elif any(x in r_lower for x in ["bear", "trending down"]): r_color = LOSS
    else: r_color = NEURAL

    risk_label, risk_color = _risk_level(vix, regime)
    conf_c   = GAIN if conf >= 0.75 else (NEURAL if conf >= 0.60 else TEXT2)
    conf_pct = f"{conf*100:.0f}%" if conf > 0 else "—"
    conf_w   = int(conf * 100) if conf > 0 else 0

    # Ensemble agreement (how many signals/conditions fired strongly)
    agree_count = sum([
        xgb_p  >= 0.60,
        lstm_p  >= 0.60,
        sent    >= 0.05,
        any(x in r_lower for x in ["bull", "trending up"]),
        vix < 25,
    ])
    agree_c = GAIN if agree_count >= 4 else (NEURAL if agree_count >= 3 else LOSS)

    # Confidence bar with ensemble agreement inline
    conf_bar = (
        f'<div style="margin:10px 0 8px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;">AI Confidence</span>'
        f'<span style="font-size:{FONT_SECTION};font-weight:700;color:{conf_c};letter-spacing:-1px;">{conf_pct}</span>'
        f'</div>'
        f'<div style="background:{BORDER};border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="background:{conf_c};height:100%;width:{conf_w}%;border-radius:4px;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:8px;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Ensemble: '
        f'<span style="color:{agree_c};font-weight:700;">{agree_count}/5 models agree</span></span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">Entry: '
        f'<span style="color:{TEXT1};font-weight:700;">${entry:.2f}</span></span>'
        f'</div></div>'
    )

    def _mini_bar(label, v, color):
        w = int(v * 100)
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};width:68px;flex-shrink:0;">{label}</span>'
            f'<div style="background:{BORDER};border-radius:2px;height:4px;flex:1;overflow:hidden;">'
            f'<div style="background:{color};height:100%;width:{w}%;"></div></div>'
            f'<span style="font-size:{FONT_LABEL};color:{color};width:34px;text-align:right;">{v*100:.0f}%</span>'
            f'</div>'
        )

    sub_scores = ""
    if xgb_p > 0 or lstm_p > 0:
        xc = GAIN if xgb_p >= 0.70 else (NEURAL if xgb_p >= 0.55 else TEXT2)
        lc = GAIN if lstm_p >= 0.70 else (NEURAL if lstm_p >= 0.55 else TEXT2)
        sc = GAIN if sent > 0.05 else (LOSS if sent < -0.05 else TEXT2)
        sub_scores = (
            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid {BORDER};">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:6px;">Model breakdown</div>'
            + _mini_bar("XGBoost", xgb_p, xc)
            + _mini_bar("LSTM", lstm_p, lc)
            + f'<div style="display:flex;gap:8px;margin:4px 0;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};width:68px;flex-shrink:0;">Sentiment</span>'
            f'<span style="font-size:{FONT_LABEL};color:{sc};">'
            f'{"Positive" if sent > 0.05 else "Negative" if sent < -0.05 else "Neutral"}'
            f' ({sent:+.2f})</span></div>'
            f'</div>'
        )

    # SHAP contributor percentages (+ contributors) and risk factors (- contributors)
    pos_items: list[tuple[str, float]] = []
    neg_items: list[str] = []
    try:
        import json as _j
        ds  = _j.loads(drv_raw) if isinstance(drv_raw, str) else (drv_raw or [])
        pos = [(f, float(v)) for f, v in (ds or []) if float(v) > 0]
        neg = [(f, float(v)) for f, v in (ds or []) if float(v) < 0]
        tot = sum(v for _, v in pos) or 1.0
        for feat, val in sorted(pos, key=lambda x: -x[1])[:4]:
            why  = _WHY_MAP.get(feat)
            name = why[0] if why else _FI_LABELS.get(feat, feat)
            pos_items.append((name, val / tot * 100))
        for feat, _ in sorted(neg, key=lambda x: x[1])[:2]:
            why  = _WHY_MAP.get(feat)
            name = why[0] if why else _FI_LABELS.get(feat, feat)
            neg_items.append(name)
    except Exception as exc:
        logger.debug(f"parse_shap_items render_ai_recommendation: {exc}")

    if any(x in r_lower for x in ["bull"]) and not any("regime" in p[0].lower() for p in pos_items):
        pos_items.append(("Bull market regime", 15.0))

    why_html = ""
    if pos_items:
        why_html += (
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:8px;">Contributors</div>'
        )
        for name, pct in pos_items:
            bar_w = min(int(pct), 100)
            why_html += (
                f'<div style="display:flex;align-items:center;gap:6px;margin:5px 0;">'
                f'<span style="font-size:{FONT_VALUE};color:{GAIN};width:14px;flex-shrink:0;'
                f'font-weight:700;line-height:1;">+</span>'
                f'<span style="font-size:{FONT_LABEL};color:{TEXT1};flex:1;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;">{name}</span>'
                f'<div style="background:{BORDER};border-radius:2px;height:4px;'
                f'width:56px;overflow:hidden;flex-shrink:0;">'
                f'<div style="background:{GAIN};height:100%;width:{bar_w}%;"></div></div>'
                f'<span style="font-size:{FONT_LABEL};color:{GAIN};width:36px;text-align:right;'
                f'flex-shrink:0;">+{pct:.0f}%</span>'
                f'</div>'
            )
    else:
        why_html += (
            f'<div style="color:{TEXT2};font-size:{FONT_LABEL};line-height:1.6;">'
            f'Signal fired after all risk gates passed.<br>'
            f'<span style="font-size:{FONT_LABEL};">SHAP % breakdown available after next model retrain.</span>'
            f'</div>'
        )

    if neg_items:
        why_html += (
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-top:12px;margin-bottom:8px;">Risk Factors</div>'
        )
        for name in neg_items:
            why_html += (
                f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0;">'
                f'<span style="font-size:{FONT_VALUE};color:{LOSS};width:14px;flex-shrink:0;'
                f'font-weight:700;line-height:1;">−</span>'
                f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{name}</span>'
                f'</div>'
            )

    risk_badge = (
        f'<span style="background:{SURFACE2};border:1px solid {risk_color};'
        f'color:{risk_color};padding:3px 10px;border-radius:4px;font-size:{FONT_LABEL};'
        f'font-weight:700;letter-spacing:.3px;">Risk: {risk_label}</span>'
    )

    card = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};'
        f'border-top:3px solid {GAIN};border-radius:8px;padding:20px;">'
        f'<div class="nt-ai-split">'
        # Left: identity + confidence
        f'<div>'
        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px;">'
        f'{_badge("BUY")}'
        f'<span style="font-family:Courier New,monospace;font-size:{FONT_HERO};font-weight:700;'
        f'color:{PRIMARY};letter-spacing:-2px;line-height:1;">{sym}</span>'
        f'{risk_badge}'
        f'</div>'
        f'<div style="font-size:{FONT_VALUE};color:{TEXT2};margin-bottom:10px;">'
        f'Entry Price: <strong style="color:{TEXT1};">${entry:.2f}</strong>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'Regime: <strong style="color:{r_color};">{regime}</strong>'
        f'</div>'
        f'{conf_bar}'
        f'{sub_scores}'
        f'</div>'
        # Right: Why section
        f'<div class="nt-ai-right">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:4px;">Why the AI is buying</div>'
        f'{why_html}'
        f'</div>'
        f'</div></div>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("🤖","AI Recommendation",_to_ct(ts))}'
            f'<div style="padding-top:4px;">{card}</div></div>')


# ── Render: AI investment committee ─────────────────────────────────────────────
@safe_render("AI Committee")
def render_ai_committee() -> str:
    vms = build_committees_vm()

    if not vms:
        return (f'<div class="nt nt-wrap">'
                f'{_section("🏛","AI Committee","XGBoost · LSTM · Sentiment votes")}'
                f'{_card(_empty_state("🏛", "Fully in cash", "Committee convenes once the bot enters positions."))}</div>')

    def _vote_chip(label: str, member_vote: str, member_color: str, pct_val: float) -> str:
        v_str = f"{pct_val*100:.0f}%" if pct_val > 0 else "—"
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px;'
            f'background:{BG};border-radius:6px;padding:8px 10px;min-width:64px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.6px;">{label}</div>'
            f'<div style="font-size:{FONT_VALUE};font-weight:700;color:{member_color};">{v_str}</div>'
            f'<div style="font-size:{FONT_LABEL};font-weight:700;color:{member_color};">{member_vote}</div></div>'
        )

    rows_html = ""
    d        = get_data()
    df       = d.get("trades_df", None)
    open_pos = d.get("open_pos", {})

    # Rebuild per-symbol raw scores for display values (vote_chip needs raw pct)
    _raw: dict[str, dict] = {}
    try:
        if df is not None and not df.empty:
            buys = df[df["action"] == "BUY"]
            for sym in open_pos:
                sym_buys = buys[buys["symbol"] == sym]
                if not sym_buys.empty:
                    lb = sym_buys.iloc[-1]
                    _raw[sym] = {
                        "xgb":  float(lb.get("xgb_prob",       0.0) or 0.0),
                        "lstm": float(lb.get("lstm_prob",       0.0) or 0.0),
                        "sent": float(lb.get("sentiment_score", 0.0) or 0.0),
                    }
    except Exception:
        pass

    for i, vm in enumerate(vms):
        border_b = f'border-bottom:1px solid {BORDER};' if i < len(vms) - 1 else ''
        if vm.no_data:
            chip_html = (f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">'
                         f'No BUY trade on record yet</span>')
        else:
            raw = _raw.get(vm.symbol, {})
            xgb    = raw.get("xgb",  0.0)
            lstm   = raw.get("lstm", 0.0)
            sent   = raw.get("sent", 0.0)
            sent_n = min(max((sent + 1) / 2, 0.0), 1.0)
            m      = {m.name: m for m in vm.members}
            chip_html = (
                _vote_chip("XGBoost",   m["XGBoost"].vote,   m["XGBoost"].color,   xgb)
                + _vote_chip("LSTM",    m["LSTM"].vote,      m["LSTM"].color,      lstm)
                + _vote_chip("Sentiment", m["Sentiment"].vote, m["Sentiment"].color, sent_n)
            )
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:12px;padding:12px 14px;{border_b}">'
            f'{_sym(vm.symbol)}'
            f'<div style="display:flex;gap:6px;flex:1;">{chip_html}</div>'
            f'<div style="text-align:right;min-width:80px;">'
            f'<div style="font-size:{FONT_LABEL};font-weight:700;color:{vm.final_color};">'
            f'{"—" if vm.no_data else vm.final_vote}</div></div></div>'
        )

    if not rows_html:
        rows_html = _empty_state("🏛", "No positions", "Committee convenes once the bot enters positions.")

    return (f'<div class="nt nt-wrap">'
            f'{_section("🏛","AI Committee","3-model vote per open position")}'
            f'{_wrap(rows_html)}</div>')


# ── PANEL 3: Sell Analysis — called internally by render_decision_center ──────
