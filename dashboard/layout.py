"""Gradio CSS and static HTML layout constants &mdash; extracted from design_system.py."""
from __future__ import annotations

from dashboard.design_system import (
    BG, SURFACE, SURFACE2, BORDER,
    TEXT1, TEXT2,
    ACTION_BUY,
    PRIMARY, GAIN, NEURAL, PRIMARY_BG, GAIN_BG, GAIN_BD, NEURAL_BG, NEURAL_BD,
)

# ── Gradio CSS: dark page + strip Gradio chrome ───────────────────────────────
GRADIO_CSS = f"""
.gradio-container, .gradio-container > .main {{
  background-color: {BG} !important;
}}
.block, .form, .wrap {{ background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important; }}
.gap {{ gap: 8px !important; }}
.contain {{ padding: 8px 12px !important; }}
.plot-container, .plot-container > div {{ background: transparent !important; }}
footer {{ display: none !important; }}

/* ── Tab navigation &mdash; high contrast fix ─────── */
.tabs > .tab-nav,
div.tabs > div.tab-nav,
.gradio-container .tabs > .tab-nav {{
  background: {SURFACE2} !important;
  border-bottom: 2px solid {BORDER} !important;
  padding: 0 8px !important;
  display: flex !important;
  gap: 4px !important;
}}

.tabs > .tab-nav > button,
div.tabs > div.tab-nav > button,
.gradio-container .tabs > .tab-nav > button {{
  color: {TEXT1} !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 3px solid transparent !important;
  border-radius: 0 !important;
  padding: 12px 20px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  opacity: 0.6 !important;
  transition: opacity 0.15s, border-color 0.15s !important;
  white-space: nowrap !important;
  cursor: pointer !important;
  margin-bottom: -2px !important;
}}

.tabs > .tab-nav > button:hover,
div.tabs > div.tab-nav > button:hover {{
  opacity: 1 !important;
  background: rgba(255,255,255,0.05) !important;
  border-bottom-color: {TEXT2} !important;
}}

.tabs > .tab-nav > button.selected,
div.tabs > div.tab-nav > button.selected,
.gradio-container .tabs > .tab-nav > button.selected {{
  color: {TEXT1} !important;
  opacity: 1 !important;
  border-bottom: 3px solid {ACTION_BUY} !important;
  background: transparent !important;
  font-weight: 700 !important;
}}

.tabitem, div.tabitem {{
  background: transparent !important;
  border: none !important;
  padding: 16px 0 0 0 !important;
}}

/* ── Symbol selector dropdown ─────────────────────────────────────────────── */
/* Outer container label */
.sym-selector > label,
.sym-selector span.label-wrap,
.sym-selector .label-wrap span {{
  color: {TEXT2} !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.8px !important;
}}
/* The visible input box */
.sym-selector .wrap,
.sym-selector > .wrap {{
  background: {SURFACE2} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 6px !important;
  box-shadow: none !important;
  padding: 0 !important;
}}
.sym-selector .wrap:focus-within,
.sym-selector > .wrap:focus-within {{
  border-color: {PRIMARY} !important;
  outline: none !important;
}}
/* Text inside the input */
.sym-selector input,
.sym-selector input[type="text"],
.sym-selector .wrap input {{
  background: transparent !important;
  color: {TEXT1} !important;
  font-family: "Courier New", monospace !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  caret-color: {PRIMARY} !important;
  padding: 10px 12px !important;
}}
/* The currently-selected token chip shown when value is set */
.sym-selector .token,
.sym-selector .token span,
.sym-selector .secondary-wrap .token {{
  background: {SURFACE2} !important;
  color: {TEXT1} !important;
  font-family: "Courier New", monospace !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  border: none !important;
}}
/* Dropdown list popup */
.sym-selector ul.options,
.sym-selector .options {{
  background: {SURFACE2} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 6px !important;
  margin-top: 4px !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
}}
/* Each list item */
.sym-selector ul.options li,
.sym-selector .options li {{
  color: {TEXT1} !important;
  font-family: "Courier New", monospace !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 9px 14px !important;
  background: transparent !important;
  cursor: pointer !important;
}}
.sym-selector ul.options li:hover,
.sym-selector .options li:hover,
.sym-selector ul.options li.selected,
.sym-selector .options li.selected {{
  background: {BORDER} !important;
  color: {PRIMARY} !important;
}}

/* ── Portfolio performance period tabs ────────────────────────────────────── */
.perf-tabs > .wrap {{ flex-wrap:wrap !important; gap:6px !important; }}
.perf-tabs label {{
  padding:6px 16px !important; border-radius:6px !important;
  border:1px solid {BORDER} !important; background:{SURFACE} !important;
  color:{TEXT2} !important; font-size:12px !important; font-weight:700 !important;
  white-space:nowrap !important; cursor:pointer !important;
  transition:color .15s, border-color .15s !important;
}}
.perf-tabs label:has(input:checked) {{
  color:{PRIMARY} !important; border-color:{PRIMARY} !important;
  background:{BG} !important;
}}

/* ── All tables: scrollable, not clipped ─────────────────────────────────── */
.nt-wrap table {{ width: 100%; table-layout: fixed; }}
.nt-wrap td {{ overflow: hidden; text-overflow: ellipsis; max-width: 200px; }}

/* ── Mobile 768px ────────────────────────────────────────────────────────── */
@media (max-width: 768px) {{
  .perf-tabs label {{
    min-height: 44px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
  }}
  /* Sticky tab bar — stays visible while scrolling content */
  .tabs > .tab-nav,
  div.tabs > div.tab-nav,
  .gradio-container .tabs > .tab-nav {{
    position: sticky !important;
    top: 0 !important;
    z-index: 50 !important;
  }}
}}

/* ── Mobile 480px ────────────────────────────────────────────────────────── */
@media (max-width: 480px) {{
  .nt-wrap {{ padding: 8px !important; }}
  .nt-cards {{ grid-template-columns: repeat(2, 1fr) !important; gap: 6px !important; }}
  .nt-ai-split {{ flex-direction: column !important; gap: 16px !important; }}
  table {{ font-size: 13px !important; }}
  table td, table th {{ padding: 8px 10px !important; }}

  /* Tab nav: scroll horizontally — all tabs always reachable */
  .tabs > .tab-nav,
  div.tabs > div.tab-nav,
  .gradio-container .tabs > .tab-nav {{
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    flex-wrap: nowrap !important;
    scrollbar-width: none !important;
    padding-bottom: 3px !important;
  }}
  .tabs > .tab-nav::-webkit-scrollbar,
  div.tabs > div.tab-nav::-webkit-scrollbar {{ display: none !important; }}
  .tabs > .tab-nav > button,
  div.tabs > div.tab-nav > button {{
    padding: 10px 12px !important;
    font-size: 12px !important;
    min-height: 44px !important;
    flex-shrink: 0 !important;
  }}

  /* Header: wrap logo+title above badge; hide badge entirely to save height */
  .nt-header {{ padding: 10px 12px !important; gap: 8px !important; flex-wrap: wrap !important; }}
  .nt-badge {{ display: none !important; }}

  /* Status bar: stack vertically, hide countdown bar */
  .nt-status {{ flex-direction: column !important; align-items: flex-start !important;
    gap: 4px !important; padding: 6px 12px !important; }}
  .nt-countdown {{ display: none !important; }}
  .nt-refresh-label {{ display: none !important; }}
}}
"""

# ── Stylesheet (injected once via static HEADER_HTML) ────────────────────────
STYLES = f"""<style>
.nt {{ font-family:-apple-system,'Inter',BlinkMacSystemFont,'Segoe UI',sans-serif;
  color:{TEXT1};box-sizing:border-box; }}
.nt *, .nt *::before, .nt *::after {{ box-sizing:border-box; }}
.nt-wrap {{ padding:12px 16px 0; }}
.nt-header {{
  display:flex;align-items:center;gap:16px;padding:16px 24px;
  background:{SURFACE};border-radius:8px;border:1px solid {BORDER};
  position:relative;overflow:hidden;
}}
.nt-header::before {{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:{PRIMARY};
}}
.nt-status {{
  display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:6px 12px;
  padding:7px 14px;margin:10px 0 8px;
  background:{SURFACE};border:1px solid {BORDER};border-radius:6px;font-size:11px;
}}
.nt-hero {{
  text-align:center;padding:20px 16px 6px;
}}
.nt-hero-val {{
  font-size:44px;font-weight:700;letter-spacing:-1px;color:{TEXT1};line-height:1;
}}
.nt-hero-chg {{
  font-size:15px;font-weight:600;margin-top:6px;
}}
.nt-cards {{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-bottom:10px;
}}
.nt-card {{
  background:{SURFACE};border-radius:8px;padding:14px 16px;
  position:relative;overflow:hidden;transition:background .15s,box-shadow .15s;
  box-shadow:0 2px 8px rgba(0,0,0,0.35);
}}
.nt-card:hover {{ background:{SURFACE2};box-shadow:0 4px 16px rgba(0,0,0,0.5); }}
.nt-sec {{
  display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;
  text-transform:uppercase;letter-spacing:1.5px;margin:12px 0 8px;
}}
.nt-sec-line {{ flex:1;height:1px;background:{BORDER}; }}
.nt-tbl {{ width:100%;border-collapse:collapse; }}
.nt-tbl th {{
  background:{BG};color:{TEXT2};font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.8px;
  padding:10px 16px;border-bottom:1px solid {BORDER};text-align:left;white-space:nowrap;
}}
.nt-tbl td {{ padding:12px 16px;border-bottom:1px solid {BORDER};vertical-align:middle; }}
.nt-tbl tr:last-child td {{ border-bottom:none; }}
.nt-tbl tr:hover td {{ background:{SURFACE2}; }}
@keyframes shimmer    {{ 0%{{background-position:0%}} 100%{{background-position:200%}} }}
@keyframes pulse      {{ 0%,100%{{opacity:1}} 50%{{opacity:0.35}} }}
@keyframes fadeInUp   {{ from{{opacity:0;transform:translateY(6px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes slideInRow {{ from{{opacity:0;transform:translateX(-4px)}} to{{opacity:1;transform:translateX(0)}} }}
@keyframes countdown  {{ from{{width:120px}} to{{width:0px}} }}
.nt-card {{ animation:fadeInUp .3s ease both; }}
.nt-ai-split {{ display:grid;grid-template-columns:1fr 1fr;gap:20px; }}
.nt-ai-right {{ border-left:1px solid {BORDER};padding-left:20px; }}
@media(max-width:768px){{
  .nt-tbl   {{ display:block;overflow-x:auto;-webkit-overflow-scrolling:touch;white-space:nowrap; }}
  .nt-ai-split {{ grid-template-columns:1fr!important; }}
  .nt-ai-right {{ border-left:none!important;padding-left:0!important;
    border-top:1px solid {BORDER};padding-top:14px;margin-top:14px; }}
}}
@media(max-width:480px){{
  .nt-hero-val  {{ font-size:28px!important; }}
  .nt-wrap      {{ padding:8px 10px 0!important; }}
  .nt-tbl th    {{ font-size:11px!important; }}
  .nt-tbl td    {{ font-size:13px!important;padding:10px 12px!important; }}
  .nt-sec       {{ font-size:10px!important; }}
  .nt-card      {{ padding:11px 12px!important; }}
  /* Bump 11px labels to 12px on phones — inline style needs !important override */
  .nt-card div[style*="font-size:11px"],
  .nt-card span[style*="font-size:11px"] {{ font-size:12px!important; }}
  .nt-hero-chg  {{ font-size:13px!important; }}
  .nt-cards     {{ grid-template-columns:repeat(2,1fr)!important;gap:6px!important; }}
}}
</style>"""

# ── Logo ──────────────────────────────────────────────────────────────────────
LOGO = f"""<svg width="52" height="52" viewBox="0 0 56 56" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="ag" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="{GAIN}" stop-opacity="0.7"/>
      <stop offset="100%" stop-color="{GAIN}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="bg2" cx="50%" cy="30%" r="70%">
      <stop offset="0%" stop-color="{PRIMARY_BG}" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="{BG}" stop-opacity="1"/>
    </radialGradient>
    <filter id="hg" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="1.4" result="b"/>
      <feComposite in="SourceGraphic" in2="b" operator="over"/>
    </filter>
  </defs>
  <circle cx="28" cy="28" r="27" fill="url(#bg2)" stroke="{BORDER}" stroke-width="1.5"/>
  <line x1="14" y1="34" x2="21" y2="22" stroke="{PRIMARY}" stroke-width="0.8" opacity="0.35"/>
  <line x1="28" y1="34" x2="21" y2="22" stroke="{PRIMARY}" stroke-width="0.8" opacity="0.35"/>
  <line x1="28" y1="34" x2="35" y2="22" stroke="{PRIMARY}" stroke-width="0.8" opacity="0.35"/>
  <line x1="42" y1="34" x2="35" y2="22" stroke="{PRIMARY}" stroke-width="0.8" opacity="0.35"/>
  <line x1="21" y1="22" x2="28" y2="10" stroke="{GAIN}" stroke-width="1.3" opacity="0.8"/>
  <line x1="35" y1="22" x2="28" y2="10" stroke="{GAIN}" stroke-width="1.3" opacity="0.8"/>
  <polygon points="21,34 17.5,40 10.5,40 7,34 10.5,28 17.5,28" fill="{BG}" stroke="{BORDER}" stroke-width="1.2" opacity="0.8"/>
  <polygon points="35,34 31.5,40 24.5,40 21,34 24.5,28 31.5,28" fill="{BG}" stroke="{BORDER}" stroke-width="1.2" opacity="0.8"/>
  <polygon points="49,34 45.5,40 38.5,40 35,34 38.5,28 45.5,28" fill="{BG}" stroke="{BORDER}" stroke-width="1.2" opacity="0.8"/>
  <polygon points="28,22 24.5,28 17.5,28 14,22 17.5,16 24.5,16" fill="{PRIMARY_BG}" stroke="{PRIMARY}" stroke-width="1.4"/>
  <polygon points="42,22 38.5,28 31.5,28 28,22 31.5,16 38.5,16" fill="{PRIMARY_BG}" stroke="{PRIMARY}" stroke-width="1.4"/>
  <circle cx="28" cy="10" r="10" fill="url(#ag)"/>
  <polygon points="35,10 31.5,16 24.5,16 21,10 24.5,4 31.5,4" fill="{GAIN_BG}" stroke="{GAIN}" stroke-width="2" filter="url(#hg)"/>
  <circle cx="28" cy="10" r="3.5" fill="{GAIN}" opacity="0.95"/>
  <circle cx="21" cy="22" r="1.8" fill="{PRIMARY}" opacity="0.9"/>
  <circle cx="35" cy="22" r="1.8" fill="{PRIMARY}" opacity="0.9"/>
</svg>"""

HEADER_HTML = f"""{STYLES}
<div class="nt nt-wrap">
<div class="nt-header" style="flex-wrap:wrap;">
  {LOGO}
  <div style="flex:1;min-width:160px;">
    <div style="font-size:22px;font-weight:700;letter-spacing:-0.3px;color:{TEXT1};">
      TradeGenius AI</div>
    <div style="font-size:11px;color:{TEXT2};margin-top:2px;">
      XGBoost + SHAP &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; FinBERT &nbsp;·&nbsp; Walk-Forward Validated
    </div>
  </div>
  <div class="nt-badge" style="display:flex;gap:8px;align-items:center;">
    <div style="display:flex;align-items:center;gap:6px;background:{NEURAL_BG};
      border:1px solid {NEURAL_BD};color:{NEURAL};padding:5px 14px;
      border-radius:6px;font-size:11px;font-weight:700;letter-spacing:.3px;">
      <span style="width:6px;height:6px;background:{NEURAL};border-radius:50%;
        display:inline-block;animation:pulse 2s infinite;flex-shrink:0;"></span>PAPER TRADING
    </div>
  </div>
</div>
</div>"""

FOOTER_HTML = f"""<div class="nt nt-wrap">
<div style="text-align:center;color:{TEXT2} !important;font-size:11px;
  margin-top:8px;padding:14px;border-top:1px solid {BORDER};">
  Refreshes every 60 s &nbsp;·&nbsp; Paper trading only &nbsp;·&nbsp;
  Alpaca Markets &nbsp;·&nbsp; Stress-tested · Walk-forward validated &nbsp;·&nbsp; TradeGenius AI v2
</div></div>"""
