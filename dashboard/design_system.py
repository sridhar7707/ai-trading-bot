"""TradeGenius Design System v1.0 — constants and HTML builder helpers."""
from __future__ import annotations

# ══════════════════════════════════════════════════
# TRADEGENIUS DESIGN SYSTEM v1.0
# Bloomberg clarity + Robinhood simplicity + Apple spacing
# DO NOT change these values without updating
# docs/DESIGN_SYSTEM.md first.
# ══════════════════════════════════════════════════

# ── Backgrounds ───────────────────────────────────
BG       = "#0f1115"   # page background — not pure black
SURFACE  = "#171a21"   # card background
SURFACE2 = "#222733"   # elevated surface / hover
BORDER   = "#2d3445"   # card borders and dividers

# ── Text — exactly 3 levels, no more ─────────────
TEXT1 = "#ffffff"   # primary — all values, numbers, amounts
TEXT2 = "#b0b7c3"   # secondary — labels, captions, timestamps
TEXT3 = "#7f8896"   # tertiary — helper text, placeholders only

# ── Action Colors — consistent everywhere ─────────
ACTION_BUY   = "#00c853"   # green
ACTION_SELL  = "#ff5252"   # red
ACTION_TRIM  = "#ffb300"   # amber
ACTION_HOLD  = "#64b5f6"   # blue
ACTION_WATCH = "#ab47bc"   # purple
ACTION_ADD   = "#00c853"   # same as BUY
ACTION_EXIT  = "#ff5252"   # same as SELL

# Action background fills (dark tinted versions)
ACTION_BUY_BG   = "#00200d"
ACTION_SELL_BG  = "#200808"
ACTION_TRIM_BG  = "#1f1500"
ACTION_HOLD_BG  = "#081428"
ACTION_WATCH_BG = "#150820"
ACTION_ADD_BG   = "#00200d"
ACTION_EXIT_BG  = "#200808"

# ── Aliases for backward compatibility ────────────
PRIMARY    = ACTION_BUY
GAIN       = ACTION_BUY
LOSS       = ACTION_SELL
NEURAL     = ACTION_WATCH
PRIMARY_BG = ACTION_BUY_BG
GAIN_BG    = ACTION_BUY_BG
LOSS_BG    = ACTION_SELL_BG
NEURAL_BG  = ACTION_WATCH_BG
GAIN_BD    = "#00a005"
LOSS_BD    = "#cc3d00"
NEURAL_BD  = "#8b3aaa"

# ── Typography — exactly 4 sizes, no more ─────────
FONT_HERO    = "36px"   # portfolio value, health score
FONT_SECTION = "20px"   # card titles
FONT_VALUE   = "15px"   # data values, prices, percentages
FONT_LABEL   = "11px"   # labels, captions (uppercase only)

# Font weights
WEIGHT_BOLD   = "700"
WEIGHT_MEDIUM = "500"
WEIGHT_NORMAL = "400"

# ── Spacing ───────────────────────────────────────
CARD_PADDING = "20px"
CARD_RADIUS  = "12px"
ROW_PADDING  = "12px 0"
SECTION_GAP  = "16px"
INNER_GAP    = "8px"

# ── Symbol styling ────────────────────────────────
SYMBOL_STYLE = (
    "font-family:Courier New,monospace;"
    f"font-weight:{WEIGHT_BOLD};"
    "letter-spacing:0.5px;"
    f"color:{ACTION_BUY};"
    f"font-size:{FONT_VALUE};"
)

# Plotly shared theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT2, family="Inter,system-ui,sans-serif", size=11),
    margin=dict(l=50, r=20, t=40, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(color=TEXT2)),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
)

# ── GRADIO_CSS, STYLES, LOGO, HEADER_HTML, FOOTER_HTML moved to dashboard/layout.py ──


# ── HTML builders ─────────────────────────────────────────────────────────────
def _pnl_color(v: str) -> str:
    return GAIN if v.startswith("+") else (LOSS if v.startswith("-") else TEXT2)


# ══════════════════════════════════════════════════
# DESIGN SYSTEM COMPONENT HELPERS
# Every render function uses ONLY these. No inline
# styles for badges, symbols, cards, labels, bars.
# ══════════════════════════════════════════════════

def _card(content: str, accent_color: str = None,
          padding: str = CARD_PADDING) -> str:
    """Standard card container. accent_color adds 3px top border."""
    accent = f"border-top:3px solid {accent_color};" if accent_color else ""
    return (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};{accent}'
        f'border-radius:{CARD_RADIUS};padding:{padding};margin-bottom:{SECTION_GAP};">'
        f'{content}</div>'
    )

def _label(text: str) -> str:
    """Uppercase small label. Max 3 words. Always uppercase."""
    return (
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:{WEIGHT_MEDIUM};margin-bottom:4px;">{text}</div>'
    )

def _hero_value(value: str, color: str = TEXT1, subtext: str = "") -> str:
    """Large hero number — portfolio value, health score."""
    sub = (f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:4px;">'
           f'{subtext}</div>' if subtext else "")
    return (
        f'<div style="font-size:{FONT_HERO};font-weight:{WEIGHT_BOLD};color:{color};'
        f'line-height:1;letter-spacing:-1px;">{value}</div>{sub}'
    )

def _section_title(title: str, note: str = "") -> str:
    """Card section heading. Max 4 words."""
    note_html = (
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};font-weight:{WEIGHT_NORMAL};'
        f'margin-left:8px;">{note}</span>' if note else ""
    )
    return (
        f'<div style="font-size:{FONT_SECTION};font-weight:{WEIGHT_BOLD};color:{TEXT1};'
        f'margin-bottom:16px;">{title}{note_html}</div>'
    )

def _action_badge(action: str, size: str = "normal") -> str:
    """Colored action badge. Single source of truth. Colors FIXED — never override."""
    action = action.upper()
    _colors = {
        "BUY":   (ACTION_BUY,   ACTION_BUY_BG),
        "ADD":   (ACTION_ADD,   ACTION_ADD_BG),
        "HOLD":  (ACTION_HOLD,  ACTION_HOLD_BG),
        "TRIM":  (ACTION_TRIM,  ACTION_TRIM_BG),
        "SELL":  (ACTION_SELL,  ACTION_SELL_BG),
        "EXIT":  (ACTION_EXIT,  ACTION_EXIT_BG),
        "WATCH": (ACTION_WATCH, ACTION_WATCH_BG),
    }
    color, bg = _colors.get(action, (TEXT2, SURFACE2))
    _sizes = {
        "small":  ("9px",  "2px 7px",  "10px"),
        "normal": ("11px", "4px 10px", "11px"),
        "large":  ("15px", "8px 20px", "14px"),
    }
    ltr, pad, fsize = _sizes.get(size, _sizes["normal"])
    return (
        f'<span style="background:{bg};border:1px solid {color};color:{color};'
        f'padding:{pad};border-radius:6px;font-size:{fsize};font-weight:{WEIGHT_BOLD};'
        f'letter-spacing:{ltr};white-space:nowrap;display:inline-block;">{action}</span>'
    )

def _symbol(sym: str, size: str = FONT_VALUE) -> str:
    """Stock symbol. Always monospace ACTION_BUY green bold."""
    return f'<span style="{SYMBOL_STYLE}font-size:{size};">{sym}</span>'

def _confidence_bar(pct: float, show_label: bool = True) -> str:
    """Always show BOTH number and bar. pct: 0.0 to 1.0"""
    pct_int = int(pct * 100)
    color = (ACTION_BUY if pct >= 0.75 else
             ACTION_TRIM if pct >= 0.60 else ACTION_SELL)
    label = (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:baseline;margin-bottom:6px;">'
        f'{_label("Confidence")}'
        f'<span style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};'
        f'color:{color};">{pct_int}%</span></div>'
        if show_label else ""
    )
    bar = (
        f'<div style="background:{BORDER};border-radius:4px;height:6px;overflow:hidden;">'
        f'<div style="background:{color};height:100%;width:{pct_int}%;'
        f'border-radius:4px;"></div></div>'
    )
    return label + bar

def _metric_row(label: str, value: str, value_color: str = TEXT1, note: str = "") -> str:
    """Single label-value row with optional note."""
    note_html = (
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};margin-left:8px;">'
        f'{note}</span>' if note else ""
    )
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:{ROW_PADDING};border-bottom:1px solid {BORDER};">'
        f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">{label}</span>'
        f'<span style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};'
        f'color:{value_color};">{value}{note_html}</span></div>'
    )

def _progress_bar(label: str, score: int, max_score: int,
                  color: str = ACTION_BUY) -> str:
    """Labeled progress bar for health score breakdown."""
    pct = int(score / max_score * 100) if max_score else 0
    return (
        f'<div style="margin:8px 0;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:1px;">{label}</span>'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT1};font-weight:{WEIGHT_BOLD};">'
        f'{score}/{max_score}</span></div>'
        f'<div style="background:{BORDER};border-radius:4px;height:4px;overflow:hidden;">'
        f'<div style="background:{color};height:100%;width:{pct}%;border-radius:4px;">'
        f'</div></div></div>'
    )

def _divider() -> str:
    return f'<div style="border-top:1px solid {BORDER};margin:{SECTION_GAP} 0;"></div>'

def _empty_state(icon: str, title: str, subtitle: str) -> str:
    return (
        f'<div style="text-align:center;padding:48px 24px;">'
        f'<div style="font-size:{FONT_HERO};margin-bottom:12px;">{icon}</div>'
        f'<div style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{TEXT1};'
        f'margin-bottom:8px;">{title}</div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};line-height:1.8;'
        f'max-width:260px;margin:0 auto;">{subtitle}</div></div>'
    )

def _action_row(symbol: str, action: str, reason: str,
                detail: str = "", number: int = None) -> str:
    """Single action row with correct visual hierarchy."""
    action = action.upper()
    urgent = action in ("EXIT", "SELL")
    medium = action in ("TRIM", "BUY", "ADD")

    if urgent:
        row_bg, row_border = ACTION_SELL_BG, f"border-left:3px solid {ACTION_SELL};"
        row_pad, sym_color, rsn_color, badge_size = "14px 16px 14px 13px", TEXT1, TEXT1, "large"
    elif medium:
        c = ACTION_TRIM if action == "TRIM" else ACTION_BUY
        bg = ACTION_TRIM_BG if action == "TRIM" else ACTION_BUY_BG
        row_bg, row_border = bg, f"border-left:3px solid {c};"
        row_pad, sym_color, rsn_color, badge_size = "12px 16px 12px 13px", TEXT1, TEXT2, "large"
    else:
        row_bg, row_border = "transparent", f"border-left:3px solid transparent;"
        row_pad, sym_color, rsn_color, badge_size = "10px 16px", TEXT1, TEXT2, "small"

    num_html = (
        f'<span style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};'
        f'color:{ACTION_BUY};min-width:20px;margin-right:12px;">{number}</span>'
        if number else ""
    )
    detail_html = (
        f'<div style="font-size:{FONT_LABEL};color:{TEXT3};margin-top:3px;">{detail}</div>'
        if detail else ""
    )
    return (
        f'<div style="display:flex;align-items:center;gap:12px;padding:{row_pad};'
        f'background:{row_bg};{row_border}border-bottom:1px solid {BORDER};flex-wrap:wrap;">'
        f'{num_html}'
        f'{_symbol(symbol, FONT_VALUE)}'
        f'{_action_badge(action, badge_size)}'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:{FONT_VALUE};color:{rsn_color};white-space:nowrap;'
        f'overflow:hidden;text-overflow:ellipsis;">{reason}</div>'
        f'{detail_html}</div></div>'
    )

def _table(headers: list, rows: list) -> str:
    """Standard table. Pass header names and pre-built <tr> row strings."""
    th_cells = "".join(f'<th {TH}>{h}</th>' for h in headers)
    return (
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;'
        f'font-family:Inter,system-ui,sans-serif;">'
        f'<thead><tr>{th_cells}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        f'</table></div>'
    )

# ── Backward-compatible shims (existing code continues to work) ───────────────
def _sym(s: str) -> str:
    return _symbol(s)

def _badge(action: str) -> str:
    return _action_badge(action)

def _num(v: str, bold: bool = False) -> str:
    w = WEIGHT_BOLD if bold else "600"
    return (f'<span style="font-family:Courier New,monospace;font-weight:{w};'
            f'font-size:{FONT_VALUE};color:{TEXT1} !important;">{v}</span>')

def _pnl(v: str, big: bool = False) -> str:
    c = _pnl_color(v)
    sz = FONT_VALUE if big else "13px"
    return (f'<span style="font-family:-apple-system,monospace;font-weight:{WEIGHT_BOLD};'
            f'font-size:{sz};color:{c} !important;">{v}</span>')

def _section(icon: str, title: str, note: str = "") -> str:
    note_html = (f'<span style="font-size:{FONT_LABEL};color:{TEXT2} !important;'
                 f'font-weight:{WEIGHT_NORMAL};letter-spacing:0;margin-left:6px;">{note}</span>'
                 if note else "")
    return (f'<div class="nt-sec" style="animation:fadeInUp .4s ease both;">'
            f'<span style="font-size:{FONT_VALUE};">{icon}</span>'
            f'<span style="color:{ACTION_BUY} !important;font-size:{FONT_LABEL};'
            f'font-weight:{WEIGHT_BOLD};">{title}</span>{note_html}'
            f'<span class="nt-sec-line"></span></div>')

def _wrap(inner: str) -> str:
    return (f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;overflow-x:auto;-webkit-overflow-scrolling:touch;">{inner}</div>')

def _stat_card(label: str, value: str, accent: str = None,
               color: str = TEXT1, sub: str = "", delay: float = 0) -> str:
    sub_html = (f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:2px;">'
                f'{sub}</div>' if sub else "")
    return (
        f'<div class="nt-card" style="animation-delay:{delay:.2f}s;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;font-weight:{WEIGHT_MEDIUM};margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:{FONT_SECTION};font-weight:{WEIGHT_BOLD};letter-spacing:-0.3px;'
        f'color:{color};line-height:1;">{value}</div>'
        f'{sub_html}</div>'
    )

# ── Table cell style strings ──────────────────────────────────────────────────
TH  = (f'style="background:{BG};color:{TEXT2};font-size:{FONT_LABEL};'
       f'font-weight:{WEIGHT_MEDIUM};text-transform:uppercase;letter-spacing:1px;'
       f'padding:10px 14px;border-bottom:1px solid {BORDER};white-space:nowrap;"')
TD  = (f'style="font-size:{FONT_VALUE};color:{TEXT1};padding:12px 14px;'
       f'border-bottom:1px solid {BORDER};white-space:nowrap;"')
TD0 = (f'style="font-size:{FONT_VALUE};color:{TEXT1};padding:12px 14px;'
       f'white-space:nowrap;"')
