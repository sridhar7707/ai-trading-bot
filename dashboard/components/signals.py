"""Watchlist, signals tab, and trade timeline."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER, TEXT1, TEXT2, TEXT3,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM,
    GAIN, LOSS, NEURAL,
    FONT_HERO, FONT_SECTION, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM, WEIGHT_NORMAL,
    CARD_PADDING, SECTION_GAP,
    _card, _label, _section_title, _action_badge, _symbol,
    _metric_row, _divider, _empty_state, _section, _wrap,
    _sym, _badge, _num, _pnl, _stat_card, TH, TD, TD0,
)
from dashboard.data import get_data, safe_query, _to_ct
from dashboard.charts import _FI_LABELS
from dashboard.components.portfolio import _SELL_REASON
from dashboard.components.ai_panel import _WHY_MAP
from bot.core.error_logger import safe_render
_logger = logger

# ── Render: watchlist (open positions with live return vs avg cost) ────────────
@safe_render("Watchlist")
def render_watchlist() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("👁","Watchlist","open positions · vs avg cost")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    rows  = ""
    items = list(open_pos.items())[:8]
    for i, (sym, pos) in enumerate(items):
        cur      = prices.get(sym, 0.0)
        shares   = pos["shares"]
        invested = pos["invested"]
        avg_cost = invested / shares if shares > 0 else 0
        chg_pct  = ((cur - avg_cost) / avg_cost * 100) if avg_cost > 0 and cur > 0 else 0.0
        arrow    = "↑" if chg_pct >= 0 else "↓"
        chg_c    = GAIN if chg_pct >= 0 else LOSS
        td = TD if i < len(items) - 1 else TD0
        rows += (
            f'<tr>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}><span style="font-family:Courier New,monospace;font-weight:600;'
            f'color:{TEXT1};">${cur:.2f}</span></td>'
            f'<td {td}><span style="font-weight:700;font-size:{FONT_VALUE};color:{chg_c};">'
            f'{arrow} {chg_pct:+.1f}%</span></td>'
            f'</tr>'
        )
    table = _wrap(
        f'<table class="nt-tbl" style="width:100%"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Price</th><th {TH}>vs Avg Cost</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("👁","Watchlist","vs avg cost · live")}{table}</div>')


# ── Render: AI decision feed (trade timeline) ────────────────────────────────
@safe_render("Timeline")
def render_timeline() -> str:
    d  = get_data()
    df = d["trades_df"]
    if df.empty:
        # Fall back to signal_log BUY evaluations so the timeline is never blank
        sig_rows = safe_query("""
            SELECT timestamp, symbol, ensemble_score, xgb_prob, lstm_prob, regime
            FROM signal_log
            WHERE ensemble_action IN ('BUY', 'STRONG_BUY')
            ORDER BY timestamp DESC
            LIMIT 20
        """, default=[])
        if not sig_rows:
            empty = (
                f'<div style="color:{TEXT2};text-align:center;padding:40px;font-size:{FONT_VALUE};">'
                f'Bot is live and scanning. Trade signals will appear here during market hours.</div>'
            )
            return f'<div class="nt nt-wrap">{_section("🕐","AI Decision Feed","live")}{_wrap(empty)}</div>'

        items = ""
        for i, (ts, sym, ens, xgb, lstm, regime) in enumerate(sig_rows):
            ens   = float(ens or 0)
            xgb   = float(xgb or 0)
            lstm  = float(lstm or 0)
            regime_lbl = str(regime or "").replace("_", " ").title()
            ts_full  = _to_ct(ts)
            time_lbl = ts_full[11:16] if len(ts_full) >= 16 else ts_full[:5]
            tz_lbl   = ts_full[17:20] if len(ts_full) >= 20 else ""
            date_lbl = ts_full[:10]
            is_last  = (i == len(sig_rows) - 1)
            c_c      = GAIN if ens >= 0.75 else (NEURAL if ens >= 0.60 else TEXT2)
            line     = f'border-bottom:1px solid {BORDER};' if not is_last else ''
            connector = (
                f'<div style="width:1px;flex:1;background:{BORDER};min-height:14px;"></div>'
                if not is_last else ''
            )
            detail = f"Confidence {ens*100:.0f}% · XGB {xgb*100:.0f}% · LSTM {lstm*100:.0f}%"
            if regime_lbl:
                detail += f" · {regime_lbl}"
            items += (
                f'<div style="display:flex;gap:14px;padding:10px 0;{line}">'
                f'<div style="flex-shrink:0;width:58px;text-align:right;">'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT1};font-family:monospace;font-weight:600;">{time_lbl}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{tz_lbl}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{date_lbl}</div>'
                f'</div>'
                f'<div style="display:flex;flex-direction:column;align-items:center;padding-top:4px;">'
                f'<div style="width:10px;height:10px;border-radius:50%;background:{GAIN};flex-shrink:0;'
                f'box-shadow:0 0 6px {GAIN}44;"></div>'
                f'{connector}'
                f'</div>'
                f'<div style="flex:1;min-width:0;">'
                f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:3px;">'
                f'<span style="background:#0d3320;border:1px solid {GAIN};color:{GAIN};'
                f'font-size:{FONT_LABEL};font-weight:700;padding:2px 7px;border-radius:4px;">MODEL SIGNAL</span>'
                f'<span style="font-family:Courier New,monospace;font-weight:700;color:#00c853;">{sym}</span>'
                f'<span style="font-size:{FONT_LABEL};color:{c_c};font-weight:700;">{ens*100:.0f}%</span>'
                f'</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{detail}</div>'
                f'</div></div>'
            )
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("🕐","AI Decision Feed",f"last {len(sig_rows)} model signals · no trades executed yet")}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:0 16px;">'
            f'{items}</div></div>'
        )

    recent = df.tail(30).iloc[::-1]
    items  = ""
    for i, (_, row) in enumerate(recent.iterrows()):
        action  = str(row.get("action", ""))
        sym     = str(row.get("symbol", "&mdash;"))
        ts      = row.get("timestamp", "")
        conf    = float(row.get("ensemble_score",  0.0) or 0.0)
        regime  = str(row.get("regime") or "").replace("_", " ").title()
        sent    = float(row.get("sentiment_score", 0.0) or 0.0)
        pnl     = float(row.get("pnl_pct",         0.0) or 0.0)
        drv_raw = row.get("feature_drivers")
        is_last = i == len(recent) - 1
        dot_c   = GAIN if action == "BUY" else LOSS

        ts_full  = _to_ct(ts)
        time_lbl = ts_full[11:16] if len(ts_full) >= 16 else ts_full[:5]
        tz_lbl   = ts_full[17:20] if len(ts_full) >= 20 else ""
        date_lbl = ts_full[:10]

        if action == "BUY":
            parts = []
            if conf > 0:        parts.append(f"Confidence {conf*100:.0f}%")
            if sent > 0.05:     parts.append("Positive sentiment")
            elif sent < -0.05:  parts.append("Negative sentiment")
            if regime:          parts.append(regime)
            try:
                import json as _j
                ds  = _j.loads(drv_raw) if isinstance(drv_raw, str) else (drv_raw or [])
                pos = [(f, float(v)) for f, v in (ds or []) if float(v) > 0]
                if pos:
                    best = max(pos, key=lambda x: x[1])
                    w = _WHY_MAP.get(best[0])
                    parts.append(w[0] if w else _FI_LABELS.get(best[0], best[0]))
            except Exception as exc:
                logger.debug(f"parse_detail_parts: {exc}")
            detail = " · ".join(parts)
        else:
            reason  = _SELL_REASON.get(action, "Exit")
            pnl_str = f"{pnl:+.1%}" if pnl != 0 else ""
            detail  = f"{reason} · {pnl_str}" if pnl_str else reason

        conf_badge = ""
        if action == "BUY" and conf > 0:
            c_c = GAIN if conf >= 0.75 else (NEURAL if conf >= 0.60 else TEXT2)
            conf_badge = (f'<span style="font-size:{FONT_LABEL};color:{c_c};font-weight:700;">'
                          f'{conf*100:.0f}%</span>')

        line = f'border-bottom:1px solid {BORDER};' if not is_last else ''
        connector = (f'<div style="width:1px;flex:1;background:{BORDER};min-height:14px;"></div>'
                     if not is_last else '')
        items += (
            f'<div style="display:flex;gap:14px;padding:10px 0;{line}">'
            f'<div style="flex-shrink:0;width:58px;text-align:right;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT1};font-family:monospace;font-weight:600;">{time_lbl}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{tz_lbl}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{date_lbl}</div>'
            f'</div>'
            f'<div style="display:flex;flex-direction:column;align-items:center;padding-top:4px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{dot_c};flex-shrink:0;'
            f'box-shadow:0 0 6px {dot_c}44;"></div>'
            f'{connector}'
            f'</div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:3px;">'
            f'{_badge(action)}{_sym(sym)}{conf_badge}'
            f'</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;">{detail}</div>'
            f'</div></div>'
        )
    return (f'<div class="nt nt-wrap">'
            f'{_section("🕐","AI Decision Feed",f"last {len(recent)} decisions · newest first")}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:0 16px;">'
            f'{items}</div></div>')


# ── Render: investor view (plain-language Models tab) ────────────────────────
