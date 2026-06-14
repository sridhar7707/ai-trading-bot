# ================================================================
# UI CHANGE TRACKING
# After making ANY change to this file run:
#   python tests/ui_changelog.py
# This updates docs/UI_CHANGELOG.md automatically.
# Do not skip this step.
# ================================================================
# ================================================================
# REQUIREMENTS TRACKING
# After making ANY change to this file run:
#   python tests/requirements_tracker.py
# This updates docs/REQUIREMENTS.md automatically.
# ================================================================
"""Gradio dashboard — TradeGenius AI, hosted on HuggingFace Spaces."""
from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
import datetime
import pandas as pd
import gradio as gr
from loguru import logger
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from bot.core.error_logger import safe_render, timed
_logger = logger  # alias used by safe_render and timed decorators
from bot.core.recommendation_engine import (
    get_portfolio_action,
    get_position_sizing,
    get_sell_analysis,
    get_recommendation_explanation,
    get_portfolio_health,
)

DB_PATH    = "trades.db"
HF_TOKEN   = os.getenv("HF_TOKEN",   "")
HF_REPO_ID = os.getenv("HF_DB_REPO_ID", os.getenv("HF_REPO_ID", "ksri77/ai-trading-bot-db"))

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

/* ── Tab navigation — high contrast fix ─────── */
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

/* ── Mobile: 390px support ───────────────────────────────────────────────── */
@media (max-width: 480px) {{
  .nt-wrap {{ padding: 8px !important; }}
  .nt-cards {{ grid-template-columns: 1fr 1fr !important; }}
  .nt-ai-split {{ flex-direction: column !important; gap: 16px !important; }}
  table {{ font-size: 13px !important; }}
  table td, table th {{ padding: 8px !important; }}
}}

/* ── All tables: prevent overflow ────────────────────────────────────────── */
.nt-wrap table {{ width: 100%; table-layout: fixed; }}

/* ── Long text: truncate not overflow ────────────────────────────────────── */
.nt-wrap td {{
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 120px;
}}

/* ── Cards: full width on small screens ──────────────────────────────────── */
.nt-cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
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
  display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px;
}}
.nt-card {{
  background:{SURFACE};border-radius:8px;padding:14px 16px;
  position:relative;overflow:hidden;transition:background .15s;
}}
.nt-card:hover {{ background:{SURFACE2}; }}
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
  .nt-cards {{ grid-template-columns:repeat(2,1fr)!important; }}
  .nt-tbl   {{ display:block;overflow-x:auto;white-space:nowrap; }}
  .nt-ai-split {{ grid-template-columns:1fr!important; }}
  .nt-ai-right {{ border-left:none!important;padding-left:0!important;
    border-top:1px solid {BORDER};padding-top:14px;margin-top:14px; }}
}}
@media(max-width:480px){{
  .nt-cards     {{ grid-template-columns:1fr!important; }}
  .nt-hero-val  {{ font-size:32px!important; }}
  .nt-wrap      {{ padding:8px 10px 0!important; }}
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
<div class="nt-header">
  {LOGO}
  <div style="flex:1;">
    <div style="font-size:22px;font-weight:700;letter-spacing:-0.3px;color:{TEXT1};">
      TradeGenius AI</div>
    <div style="font-size:11px;color:{TEXT2};margin-top:2px;">
      XGBoost + SHAP &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; FinBERT &nbsp;·&nbsp; Walk-Forward Validated
    </div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <div style="display:flex;align-items:center;gap:6px;background:{GAIN_BG};
      border:1px solid {GAIN_BD};color:{GAIN};padding:5px 14px;
      border-radius:6px;font-size:11px;font-weight:700;letter-spacing:.3px;">
      <span style="width:6px;height:6px;background:{GAIN};border-radius:50%;
        display:inline-block;animation:pulse 2s infinite;flex-shrink:0;"></span>LIVE
    </div>
    <div style="background:{SURFACE2};border:1px solid {BORDER};
      color:{TEXT2};padding:5px 14px;border-radius:6px;font-size:11px;font-weight:600;">
      PAPER</div>
  </div>
</div>
</div>"""

FOOTER_HTML = f"""<div class="nt nt-wrap">
<div style="text-align:center;color:{TEXT2} !important;font-size:11px;
  margin-top:8px;padding:14px;border-top:1px solid {BORDER};">
  Refreshes every 60 s &nbsp;·&nbsp; Paper trading only &nbsp;·&nbsp;
  Alpaca Markets &nbsp;·&nbsp; Stress-tested · Walk-forward validated &nbsp;·&nbsp; TradeGenius AI v2
</div></div>"""

# ── Shared data cache (55-second TTL) ─────────────────────────────────────────
# All render functions share one DB read + one yfinance call per cycle instead
# of each making their own, which previously caused 5× HF calls and 2× price
# fetches every minute.
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TS: float = 0.0
_CACHE_TTL: float = 55.0
_price_cache: dict = {}
_price_cache_time: dict = {}
_PRICE_CACHE_TTL: float = 3600.0

_EMPTY_CACHE: dict = {
    "open_pos": {}, "prices": {}, "trades_df": pd.DataFrame(),
    "portfolio": "—", "regime_raw": "Unknown",
    "total_trades": 0, "buy_count": 0, "sell_count": 0, "win_count": 0,
    "recent_trades": [],
    "vix": 0.0, "avg_confidence": 0.0, "sentiment_avg": 0.0,
    "latest_buy_signal": {}, "today_buy_signals": [],
}


def _sync_db() -> None:
    if not HF_TOKEN or not HF_REPO_ID:
        return
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename="trades.db",
                                  repo_type="dataset", token=HF_TOKEN, force_download=True)
        shutil.copy(cached, DB_PATH)
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("404", "not found", "entry", "does not exist")):
            # File was deleted from HF — remove local copy so dashboard shows empty state
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
                logger.info("DB sync: trades.db deleted from HF — local copy removed")
        else:
            logger.opt(exception=True).warning(f"DB sync: {e}")
    # Pull validation / explainability artefacts — non-fatal if absent
    for filename, dest in [
        ("validation_report.json",  "models/validation_report.json"),
        ("feature_importance.json", "models/feature_importance.json"),
    ]:
        try:
            from huggingface_hub import hf_hub_download
            cached = hf_hub_download(repo_id=HF_REPO_ID, filename=filename,
                                      repo_type="dataset", token=HF_TOKEN, force_download=True)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(cached, dest)
        except Exception as exc:
            logger.debug(f"hf_artifact_download: {exc}")  # non-critical file absent


def _current_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    try:
        import yfinance as yf
        df = yf.download(" ".join(symbols), period="2d", progress=False, auto_adjust=True)
        if df.empty:
            return {s: 0.0 for s in symbols}
        close = df["Close"]
        prices = {}
        for sym in symbols:
            try:
                col = close[sym] if isinstance(close, pd.DataFrame) else close
                prices[sym] = float(col.dropna().iloc[-1])
            except Exception:
                prices[sym] = 0.0
        return prices
    except Exception as e:
        logger.warning(f"Price fetch: {e}")
        return {s: 0.0 for s in symbols}


def _refresh_cache() -> dict:
    """One DB read + one yfinance call; derives everything all render fns need."""
    _sync_db()
    result = dict(_EMPTY_CACHE)
    if not os.path.exists(DB_PATH):
        return result
    try:
        con = sqlite3.connect(DB_PATH)
        # Migrate schema so the dashboard works with any DB version pulled from HF.
        # ALTER TABLE is a no-op when the column already exists (OperationalError is swallowed).
        for _col in (
            "xgb_prob REAL DEFAULT 0.0",
            "lstm_prob REAL DEFAULT 0.0",
            "sentiment_score REAL DEFAULT 0.0",
            "macro_score REAL DEFAULT 0.0",
            "ensemble_score REAL DEFAULT 0.0",
            "realized_pnl REAL DEFAULT 0.0",
            "order_id TEXT DEFAULT NULL",
            "holding_days INTEGER DEFAULT 0",
            "feature_drivers TEXT DEFAULT NULL",
        ):
            try:
                con.execute(f"ALTER TABLE trades ADD COLUMN {_col}")
                con.commit()
            except sqlite3.OperationalError:
                pass
        try:
            df = pd.read_sql(
                "SELECT id,timestamp,symbol,action,shares,price,notional,"
                "pnl_pct,portfolio_value,regime,"
                "COALESCE(ensemble_score,0.0) AS ensemble_score,"
                "COALESCE(sentiment_score,0.0) AS sentiment_score,"
                "COALESCE(xgb_prob,0.0)       AS xgb_prob,"
                "COALESCE(lstm_prob,0.0)       AS lstm_prob,"
                "feature_drivers "
                "FROM trades ORDER BY id", con)
        except Exception as _e:
            logger.opt(exception=True).warning(f"Extended trades query failed (missing columns?): {_e} — falling back to base schema")
            df = pd.read_sql(
                "SELECT id,timestamp,symbol,action,shares,price,notional,"
                "pnl_pct,portfolio_value,regime FROM trades ORDER BY id", con)
            df["ensemble_score"] = 0.0
            df["sentiment_score"] = 0.0
            df["xgb_prob"]        = 0.0
            df["lstm_prob"]       = 0.0
            df["feature_drivers"] = None
        con.close()
    except Exception as e:
        logger.opt(exception=True).warning(f"DB read: {e}")
        return result

    if df.empty:
        return result

    # Timestamp & date
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    result["trades_df"] = df

    # Summary stats — single pass over the DataFrame
    result["total_trades"] = len(df)
    result["buy_count"]    = int((df["action"] == "BUY").sum())
    sells_mask             = df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")
    result["sell_count"]   = int(sells_mask.sum())
    result["win_count"]    = int((sells_mask & (df["pnl_pct"] > 0)).sum())

    # Latest portfolio & regime
    last = df.iloc[-1]
    result["portfolio"]  = (f"${last['portfolio_value']:,.2f}"
                            if pd.notna(last["portfolio_value"]) else "—")
    result["regime_raw"] = (str(last["regime"] or "Unknown")).replace("_", " ")

    # Open positions — walk trades in order
    pos: dict = {}
    for _, row in df.iterrows():
        sym = row["symbol"]
        shares   = row["shares"]   or 0.0
        notional = row["notional"] or 0.0
        if row["action"] == "BUY":
            if sym not in pos:
                pos[sym] = {"shares": 0.0, "invested": 0.0}
            pos[sym]["shares"]   += shares
            pos[sym]["invested"] += notional
        elif row["action"].startswith("SELL") and sym in pos and pos[sym]["shares"] > 0:
            avg = pos[sym]["invested"] / pos[sym]["shares"]
            pos[sym]["shares"]   = max(0.0, pos[sym]["shares"] - shares)
            pos[sym]["invested"] = max(0.0, pos[sym]["invested"] - avg * shares)
    result["open_pos"] = {s: d for s, d in pos.items() if d["shares"] > 0.001}

    # Recent trades (last 15, newest first) — columns matching render_trades usage
    recent = df.tail(15).iloc[::-1][
        ["timestamp", "symbol", "action", "shares", "price", "notional", "pnl_pct", "regime"]
    ]
    result["recent_trades"] = list(recent.itertuples(index=False, name=None))

    # Current prices + VIX in one batch call (always runs, even with no open positions)
    fetch_syms = list(result["open_pos"].keys()) + ["^VIX"]
    all_prices = _current_prices(fetch_syms)
    result["prices"] = {k: v for k, v in all_prices.items() if k != "^VIX"}
    result["vix"]    = all_prices.get("^VIX", 0.0)

    # Signal intelligence — confidence, sentiment, latest BUY, today's signals
    buys_df = df[df["action"] == "BUY"]
    if not buys_df.empty:
        result["avg_confidence"] = float(buys_df.tail(5)["ensemble_score"].mean())
        result["sentiment_avg"]  = float(buys_df.tail(20)["sentiment_score"].mean())
        result["latest_buy_signal"] = buys_df.iloc[-1].to_dict()
        today_str  = str(datetime.date.today())
        today_buys = buys_df[buys_df["date"].astype(str) == today_str]
        if today_buys.empty:
            today_buys = buys_df.tail(10)  # fallback: last 10 signals
        result["today_buy_signals"] = today_buys.iloc[::-1].to_dict("records")

    return result


def get_data() -> dict:
    """Return cached data, refreshing if TTL has elapsed."""
    global _CACHE, _CACHE_TS
    now = time.time()
    with _CACHE_LOCK:
        if _CACHE and (now - _CACHE_TS) < _CACHE_TTL:
            return _CACHE
        _CACHE = _refresh_cache()
        _CACHE_TS = now
    return _CACHE


# ── Helpers ───────────────────────────────────────────────────────────────────
def _now_ct() -> str:
    try:
        from zoneinfo import ZoneInfo
        ct = datetime.datetime.now(datetime.timezone.utc).astimezone(ZoneInfo("America/Chicago"))
        label = "CDT" if ct.dst() and ct.dst().total_seconds() else "CST"
        return ct.strftime(f"%b %d, %Y &nbsp;%H:%M {label}")
    except Exception as exc:
        logger.debug(f"_ct_now timezone: {exc}")
        return datetime.datetime.utcnow().strftime("%H:%M UTC")

def _to_ct(ts) -> str:
    try:
        from zoneinfo import ZoneInfo
        if isinstance(ts, datetime.datetime):
            dt = ts
        else:
            dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        ct = dt.astimezone(ZoneInfo("America/Chicago"))
        label = "CDT" if ct.dst() and ct.dst().total_seconds() else "CST"
        return ct.strftime(f"%Y-%m-%d %H:%M {label}")
    except Exception as exc:
        logger.debug(f"_to_ct parse: {exc}")
        return str(ts)[:16].replace("T", " ")

def _market_status() -> tuple[str, str]:
    """Returns (label, color)."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.datetime.now(ZoneInfo("America/New_York"))
        if et.weekday() >= 5:
            return "Weekend", TEXT2
        open_t  = et.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_t = et.replace(hour=16, minute=0,  second=0, microsecond=0)
        if open_t <= et < close_t:
            return "Market Open", GAIN
        elif et < open_t:
            return "Pre-Market", NEURAL
        return "After Hours", TEXT2
    except Exception as exc:
        logger.debug(f"_market_status: {exc}")
        return "—", TEXT2


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

def _num(v: str, bold=False) -> str:
    w = WEIGHT_BOLD if bold else "600"
    return (f'<span style="font-family:Courier New,monospace;font-weight:{w};'
            f'font-size:{FONT_VALUE};color:{TEXT1} !important;">{v}</span>')

def _pnl(v: str, big=False) -> str:
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
            f'border-radius:8px;overflow:hidden;">{inner}</div>')

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

# ── Render: metrics ───────────────────────────────────────────────────────────
@safe_render("Metrics")
def render_metrics() -> str:
    d = get_data()
    open_syms      = d["open_pos"]
    prices         = d["prices"]
    total_invested = sum(v["invested"] for v in open_syms.values())
    total_cur      = sum(v["shares"] * prices.get(s, 0.0) for s, v in open_syms.items())
    total_pnl      = total_cur - total_invested
    pnl_pct_all    = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0

    pnl_str    = f"${total_pnl:+,.2f}" if total_invested > 0 else "—"
    pnl_sub    = f"{pnl_pct_all:+.2f}% on capital" if total_invested > 0 else "no open positions"
    pnl_color  = GAIN if total_pnl >= 0 else LOSS
    pnl_accent = GAIN_BD if total_pnl >= 0 else LOSS_BD

    invested_str = f"${total_invested:,.2f}" if total_invested > 0 else "—"

    r_lower = d["regime_raw"].lower()
    if any(x in r_lower for x in ["bull", "trending up"]):
        r_color, r_accent = GAIN, GAIN_BD
    elif any(x in r_lower for x in ["bear", "trending down"]):
        r_color, r_accent = LOSS, LOSS_BD
    else:
        r_color, r_accent = NEURAL, NEURAL_BD

    sell_count = d["sell_count"]
    win_count  = d["win_count"]
    win_rate   = (win_count / sell_count * 100) if sell_count > 0 else 0.0
    wr_str     = f"{win_rate:.1f}%" if sell_count > 0 else "—"
    wr_color   = GAIN if win_rate >= 50 else (LOSS if sell_count > 0 else TEXT2)
    wr_accent  = GAIN_BD if win_rate >= 50 else (LOSS_BD if sell_count > 0 else BORDER)

    open_count = len(open_syms)
    mkt_label, mkt_color = _market_status()

    # ── Hero: large portfolio value (Robinhood-style focal point) ────────────
    portfolio_val = d["portfolio"]
    pnl_sign      = "+" if total_pnl >= 0 else ""
    hero_chg      = (f'{pnl_sign}${total_pnl:,.2f} ({pnl_pct_all:+.2f}%)'
                     if total_invested > 0 else "No open positions")
    hero = (
        f'<div class="nt-hero">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:6px;">Alpaca Paper Account Balance</div>'
        f'<div class="nt-hero-val">{portfolio_val}</div>'
        f'<div class="nt-hero-chg" style="color:{pnl_color};">{hero_chg}</div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:4px;">'
        f'Unrealized gain / loss on open positions vs. what the bot paid</div>'
        f'</div>'
    )

    status = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">'
        f'Updated &nbsp;<strong style="color:{TEXT1};">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:{FONT_LABEL};">'
        f'{mkt_label}</span></span>'
        f'<div style="height:2px;width:100px;background:{BORDER};border-radius:1px;">'
        f'<div style="height:100%;width:100%;background:{PRIMARY};border-radius:1px;'
        f'animation:countdown 60s linear forwards;"></div></div>'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">60s refresh</span>'
        f'</div>'
    )

    legend = (
        f'<div style="display:flex;gap:18px;padding:4px 2px 8px;font-size:{FONT_LABEL};color:{TEXT2};">'
        f'<span><span style="color:{GAIN};">●</span> Gain / Bull regime</span>'
        f'<span><span style="color:{LOSS};">●</span> Loss / Bear regime</span>'
        f'<span><span style="color:{NEURAL};">●</span> Neutral / Ranging</span>'
        f'<span style="margin-left:auto;font-style:italic;">Paper money — no real funds at risk</span>'
        f'</div>'
    )

    row1 = (
        f'<div class="nt-cards">'
        + _stat_card("Unrealized P&amp;L",  pnl_str,                pnl_color, pnl_color,
                "Open trade gain/loss vs. cost basis",         0.00)
        + _stat_card("Total Invested",      invested_str,            TEXT2,     TEXT1,
                "Capital currently deployed in open trades",   0.06)
        + _stat_card("Market Regime",       d["regime_raw"].title(), TEXT2,     r_color,
                "AI-detected trend — drives position sizing",  0.12)
        + _stat_card("Market Session",      mkt_label,               TEXT2,     mkt_color,
                "NYSE/NASDAQ open 9:30am–4pm ET, Mon–Fri",    0.18)
        + f'</div>'
    )

    row2 = (
        f'<div class="nt-cards">'
        + _stat_card("Open Positions", str(open_count),
                TEXT2, TEXT1,
                f"Unique stocks held now (max 8 allowed)", 0.24)
        + _stat_card("Win Rate",       wr_str,
                TEXT2, wr_color,
                f"% of closed trades that made money · {win_count}/{sell_count}", 0.30)
        + _stat_card("Total Trades",   str(d["total_trades"]),
                TEXT2, TEXT1, "All BUY + SELL orders since launch", 0.36)
        + _stat_card("Buys / Sells",   f"{d['buy_count']} / {d['sell_count']}",
                TEXT2, TEXT1, "Entry orders vs. exit orders placed", 0.42)
        + f'</div>'
    )

    return f'<div class="nt nt-wrap">{hero}{status}{legend}{row1}{row2}</div>'


# ── Render: equity chart (65% width) ─────────────────────────────────────────
def render_equity_chart():
    try:
        import plotly.graph_objects as go
        df  = get_data()["trades_df"]
        fig = go.Figure()

        has_data = (not df.empty and "portfolio_value" in df.columns
                    and df["portfolio_value"].notna().any())
        if not has_data:
            fig.add_annotation(
                text="Building history — bot trades 9:30am–4pm ET, Mon–Fri. Chart appears after the first trading day.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=12))
        else:
            daily = df.groupby("date")["portfolio_value"].last().reset_index()
            daily.columns = ["date", "value"]
            daily["date"] = pd.to_datetime(daily["date"])

            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["value"],
                fill="tozeroy", fillcolor="rgba(0,200,5,0.08)",
                line=dict(color=PRIMARY, width=2),
                mode="lines",
                hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.2f}<extra></extra>",
                name="Portfolio Value",
            ))
            if len(daily) > 1:
                peak_idx = daily["value"].idxmax()
                fig.add_annotation(
                    x=daily.loc[peak_idx, "date"], y=daily.loc[peak_idx, "value"],
                    text=f"Peak ${daily.loc[peak_idx,'value']:,.0f}",
                    showarrow=True, arrowhead=2, arrowcolor=GAIN,
                    font=dict(color=GAIN, size=10), bgcolor=GAIN_BG, bordercolor=GAIN_BD)

        fig.update_layout(
            title=dict(text="Portfolio Value Over Time  <span style='font-size:11px;'>— end-of-day snapshots, includes cash + open positions</span>",
                       font=dict(color=TEXT1, size=13), x=0.01),
            xaxis=dict(title="", **PLOTLY_LAYOUT["xaxis"], tickfont=dict(color=TEXT2)),
            yaxis=dict(title="Account Value ($)", **PLOTLY_LAYOUT["yaxis"],
                       tickformat="$,.0f", tickfont=dict(color=TEXT2)),
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            height=300,
        )
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                             if k not in ("xaxis", "yaxis")}, height=300)
        fig.add_annotation(text=f"Chart error: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=LOSS))
        return fig


# ── Render: allocation chart (35% width) ─────────────────────────────────────
def render_allocation_chart():
    try:
        import plotly.graph_objects as go
        d = get_data()
        open_syms = d["open_pos"]
        fig = go.Figure()

        if not open_syms:
            fig.add_annotation(
                text="No open positions",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=13))
        else:
            syms      = list(open_syms.keys())
            invested  = [open_syms[s]["invested"] for s in syms]
            total_inv = sum(invested)
            palette   = [PRIMARY, GAIN, NEURAL, "#f7931a", "#e040fb", "#00bcd4", "#76ff03"]
            colors    = [palette[i % len(palette)] for i in range(len(syms))]

            fig.add_trace(go.Pie(
                labels=syms, values=invested, hole=0.58,
                marker=dict(colors=colors, line=dict(color=BG, width=2)),
                textfont=dict(color=TEXT1, size=11),
                texttemplate="<b>%{label}</b><br>%{percent:.0%}",
                hovertemplate="<b>%{label}</b><br>$%{value:,.2f} (%{percent:.1%})<extra></extra>",
            ))
            fig.add_annotation(
                text=f"<b>${total_inv:,.0f}</b><br>invested",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color=TEXT1, size=13),
                xref="paper", yref="paper")

        fig.update_layout(
            title=dict(text="Capital Allocation", font=dict(color=TEXT1, size=13), x=0.01),
            showlegend=True,
            legend=dict(orientation="v", x=1.02, y=0.5,
                        font=dict(color=TEXT2, size=10), bgcolor="rgba(0,0,0,0)"),
            **{k: v for k, v in PLOTLY_LAYOUT.items()
               if k not in ("xaxis", "yaxis", "legend")},
            height=300,
        )
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                             if k not in ("xaxis", "yaxis")}, height=300)
        fig.add_annotation(text=f"Chart error: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=LOSS))
        return fig


# ── Render: daily P&L bar chart ───────────────────────────────────────────────
def render_pnl_chart():
    try:
        import plotly.graph_objects as go
        df    = get_data()["trades_df"]
        fig   = go.Figure()
        sells = df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")].copy() if not df.empty else pd.DataFrame()

        if sells.empty or "pnl_pct" not in sells.columns or sells["pnl_pct"].isna().all():
            fig.add_annotation(
                text="Realized P&L appears here after the first sell. Each bar = sum of all closed trades on that day.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=12))
        else:
            sells["pnl_dollar"] = sells["pnl_pct"] * sells["notional"].fillna(0)
            daily_pnl = sells.groupby("date")["pnl_dollar"].sum().reset_index()
            daily_pnl["date"] = pd.to_datetime(daily_pnl["date"])
            colors = [GAIN if v >= 0 else LOSS for v in daily_pnl["pnl_dollar"]]

            fig.add_trace(go.Bar(
                x=daily_pnl["date"], y=daily_pnl["pnl_dollar"],
                marker_color=colors, marker_line=dict(width=0),
                hovertemplate="<b>%{x|%b %d}</b><br>P&L: $%{y:+,.2f}<extra></extra>",
                name="Daily P&L",
            ))

        fig.update_layout(
            title=dict(text="Daily Realized P&L  <span style='font-size:11px;'>— profit/loss from SELL trades only (unrealized not included)</span>",
                       font=dict(color=TEXT1, size=13), x=0.01),
            xaxis=dict(title="", **PLOTLY_LAYOUT["xaxis"], tickfont=dict(color=TEXT2)),
            yaxis=dict(title="P&L ($)", **PLOTLY_LAYOUT["yaxis"], tickformat="$,.0f",
                       tickfont=dict(color=TEXT2), zeroline=True, zerolinewidth=1),
            bargap=0.3,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            height=300,
        )
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                             if k not in ("xaxis", "yaxis")}, height=300)
        fig.add_annotation(text=f"Chart error: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=LOSS))
        return fig


def _get_sym_hist(symbol: str):
    """Fetch yfinance max-period history for symbol; cached for 1 h. Returns DataFrame or None."""
    now = time.time()
    if symbol in _price_cache and now - _price_cache_time.get(symbol, 0.0) < _PRICE_CACHE_TTL:
        return _price_cache[symbol]
    try:
        import yfinance as _yf
        hist = _yf.Ticker(symbol).history(period="max")
        if hist is not None and not hist.empty:
            _price_cache[symbol] = hist
            _price_cache_time[symbol] = now
            return hist
    except Exception as _exc:
        logger.warning(f"yfinance {symbol}: {_exc}")
    return None


def _sym_perf(hist, buy_date) -> dict:
    """Compute pct returns (1D/1W/1M/1Y/All) from a yfinance history DataFrame."""
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {}
    today = datetime.date.today()
    try:
        dates = [d.date() if hasattr(d, "date") else d for d in hist.index]
    except Exception as exc:
        logger.debug(f"_pct_changes date parse: {exc}")
        return {}
    closes = list(hist["Close"])
    if not closes:
        return {}
    cur = float(closes[-1])

    def _pct_at(cutoff: datetime.date):
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= cutoff:
                ref = float(closes[i])
                return (cur - ref) / ref * 100 if ref > 0 else None
        return None

    result: dict = {
        "1D": _pct_at(today - datetime.timedelta(days=1)),
        "1W": _pct_at(today - datetime.timedelta(days=7)),
        "1M": _pct_at(today - datetime.timedelta(days=30)),
        "1Y": _pct_at(today - datetime.timedelta(days=365)),
    }
    if buy_date:
        try:
            result["All"] = _pct_at(datetime.date.fromisoformat(str(buy_date)[:10]))
        except Exception as exc:
            logger.debug(f"_pct_changes buy_date: {exc}")
            result["All"] = None
    else:
        result["All"] = None
    return result


def _sparkline(symbol: str) -> str:
    """Return an 80×32 inline SVG sparkline from the last 30 days of cached price data."""
    hist = _price_cache.get(symbol)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return f'<span style="color:{TEXT2};">—</span>'
    try:
        prices = [float(p) for p in hist["Close"].iloc[-30:]]
    except Exception as exc:
        logger.debug(f"_sparkline prices: {exc}")
        return f'<span style="color:{TEXT2};">—</span>'
    if not prices:
        return f'<span style="color:{TEXT2};">—</span>'

    color = GAIN if prices[-1] >= prices[0] else LOSS

    if len(prices) == 1:
        pts       = "2,16 78,16"
        lx, ly    = 78.0, 16.0
    else:
        min_p   = min(prices)
        max_p   = max(prices)
        range_p = (max_p - min_p) or 1.0
        n       = len(prices) - 1
        xy      = [(2.0 + (i / n) * 76.0,
                    30.0 - ((p - min_p) / range_p) * 28.0)
                   for i, p in enumerate(prices)]
        pts     = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        lx, ly  = xy[-1]

    return (
        f'<svg width="80" height="32" style="display:block;overflow:visible;">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2" fill="{color}"/>'
        f'</svg>'
    )


# ── Render: positions table ───────────────────────────────────────────────────
@timed(_logger)
@safe_render("Positions")
def render_positions() -> str:
    """Columns: Symbol | Action | Weight | Target | Confidence | P&L"""
    d         = get_data()
    open_syms = d.get("open_pos", {})
    prices    = d.get("prices", {})

    if not open_syms:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("📊","Open Positions","no positions held")}'
            + _card(_empty_state("📊", "No open positions",
                                 "The bot enters trades during market hours "
                                 "(9:30am–4pm ET, Mon–Fri) when signals align."))
            + f'</div>'
        )

    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d.get("portfolio", "—") != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value: {exc}")

    row_htmls = []
    items = list(open_syms.items())
    n = len(items)
    for i, (sym, v) in enumerate(items):
        cur_price = prices.get(sym, 0.0)
        cur_val   = v["shares"] * cur_price if cur_price > 0 else v["invested"]
        invested  = v["invested"]
        pnl_pct   = ((cur_val - invested) / invested * 100) if invested > 0 else 0.0
        pos_pct   = (cur_val / _pv * 100) if _pv > 0 else 0.0

        # Use recommendation engine (single source of truth)
        pa  = get_portfolio_action(sym, d)
        sz  = get_position_sizing(sym, d)
        action  = pa.get("action", "HOLD")
        conf    = pa.get("confidence", 0) / 100.0
        tgt_w   = sz.get("target_weight", 0.0)
        reason  = pa.get("reason", "—")

        # P&L color
        pnl_c   = ACTION_BUY if pnl_pct >= 0 else ACTION_SELL
        pnl_str = f"{pnl_pct:+.1f}%"

        # Confidence bar (compact, no label)
        conf_int = int(conf * 100)
        conf_c   = (ACTION_BUY if conf >= 0.75 else
                    ACTION_TRIM if conf >= 0.60 else ACTION_SELL)
        conf_html = (
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="background:{BORDER};border-radius:3px;height:5px;width:50px;">'
            f'<div style="background:{conf_c};height:100%;width:{conf_int}%;border-radius:3px;">'
            f'</div></div>'
            f'<span style="font-size:{FONT_LABEL};font-weight:{WEIGHT_BOLD};color:{conf_c};">'
            f'{conf_int}%</span></div>'
        )

        # Row background from action hierarchy
        _urgent = action in ("EXIT", "SELL")
        _medium = action in ("TRIM", "BUY", "ADD")
        if _urgent:
            row_bg  = f'background:{ACTION_SELL_BG};border-left:3px solid {ACTION_SELL};'
        elif _medium:
            _c = ACTION_TRIM if action == "TRIM" else ACTION_BUY
            _b = ACTION_TRIM_BG if action == "TRIM" else ACTION_BUY_BG
            row_bg  = f'background:{_b};border-left:3px solid {_c};'
        else:
            row_bg  = ""

        badge_size = "large" if _urgent else ("large" if action in ("TRIM", "BUY") else
                     "normal" if action in ("ADD", "WATCH") else "small")
        td  = TD  if i < n - 1 else TD0
        sym_c = TEXT1 if (_urgent or _medium) else TEXT2

        row_htmls.append(
            f'<tr style="{row_bg}">'
            f'<td {td}>{_symbol(sym)}</td>'
            f'<td {td}>{_action_badge(action, badge_size)}</td>'
            f'<td {td}><span style="font-size:{FONT_VALUE};color:{TEXT1};">'
            f'{pos_pct:.1f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_VALUE};color:{TEXT2};">'
            f'{tgt_w:.1f}%</span></td>'
            f'<td {td}>{conf_html}</td>'
            f'<td {td}><span style="font-weight:{WEIGHT_BOLD};color:{pnl_c};">'
            f'{pnl_str}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT3};">'
            f'{reason[:50]}{"…" if len(reason) > 50 else ""}</span></td>'
            f'</tr>'
        )

    note = f"{n} position{'s' if n != 1 else ''} · live price · 60s refresh"
    table = _table(
        ["Symbol", "Action", "Weight", "Target", "Confidence", "P&L", "Reason"],
        row_htmls,
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("📊","Open Positions",note)}'
            + _card(table, padding="0")
            + f'</div>')


# ── Render: trades table ──────────────────────────────────────────────────────
@safe_render("Trades")
def render_trades() -> str:
    d             = get_data()
    raw           = d["recent_trades"]
    total_trades  = d["total_trades"]

    if not raw:
        return f'<div class="nt nt-wrap">{_section("⚡","Recent Trades")}{_card(_empty_state("⚡", "No trades yet", "The bot logs trades here as they execute during market hours."))}</div>'

    shown = len(raw)
    note  = f"last {shown} of {total_trades}" if total_trades > shown else f"{shown} total"

    rows = ""
    for i, row in enumerate(raw):
        ts, sym, action, shares, price, notional, pnl_pct, regime = row
        pnl_str = f"{pnl_pct:+.2%}" if pnl_pct else "—"
        val_str = f"${notional:.2f}" if notional else "—"
        qty_str = f"{shares:.4f}"   if shares   else "—"
        px_str  = f"${price:.2f}"   if price    else "—"
        reg_str = (regime or "—").replace("_", " ").title()
        td   = TD if i < len(raw) - 1 else TD0
        anim = f'style="animation:slideInRow .35s ease both;animation-delay:{i*0.05:.2f}s;"'
        rows += (
            f'<tr {anim}>'
            f'<td {td}><span style="font-family:Courier New,monospace;font-size:{FONT_LABEL};'
            f'color:{TEXT2} !important;">{_to_ct(ts)}</span></td>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_badge(action)}</td>'
            f'<td {td}>{_num(qty_str)}</td>'
            f'<td {td}>{_num(px_str,bold=True)}</td>'
            f'<td {td}>{_num(val_str,bold=True)}</td>'
            f'<td {td}>{_pnl(pnl_str,big=True)}</td>'
            f'<td {td}><span style="font-size:12px;color:{TEXT2} !important;'
            f'font-weight:600;">{reg_str}</span></td>'
            f'</tr>'
        )
    legend_row = (
        f'<tr><td colspan="8" style="padding:6px 16px 4px;background:{BG};'
        f'font-size:{FONT_LABEL};color:{TEXT2};border-bottom:1px solid {BORDER};">'
        f'BUY = bot entered a position &nbsp;·&nbsp; '
        f'SELL = normal exit (target hit or stop) &nbsp;·&nbsp; '
        f'SELL_STOP = stop-loss triggered &nbsp;·&nbsp; '
        f'P&amp;L shown on exit trades only &nbsp;·&nbsp; '
        f'Regime = market trend at time of trade'
        f'</td></tr>'
    )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Time (CT)</th><th {TH}>Symbol</th>'
        f'<th {TH}>Action</th><th {TH}>Qty</th>'
        f'<th {TH}>Price</th><th {TH}>Value</th>'
        f'<th {TH}>P&amp;L</th><th {TH}>Regime</th>'
        f'</tr>{legend_row}</thead><tbody>{rows}</tbody></table>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("⚡","Recent Trades", note)}{table}</div>')


# ── Feature-name display labels (XGB feature_importances_ key → readable text) ─
_FI_LABELS: dict[str, str] = {
    "rsi": "RSI", "rsi_15m": "RSI 15m", "stoch_k": "Stoch %K",
    "macd_diff_pct": "MACD Cross", "volume_ratio": "Volume Ratio",
    "mfi": "Money Flow", "bb_width": "BB Width", "atr_pct": "Volatility",
    "norm_close": "Price Pos", "ema20_pct": "EMA20 Dev",
    "ema50_pct": "EMA50 Dev", "vwap_dev": "VWAP Dev", "hl_ratio": "H/L Range",
}

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


_SECTOR_MAP: dict[str, str] = {
    "NVDA": "Tech",    "MSFT": "Tech",    "AAPL": "Tech",  "GOOGL": "Tech",
    "META": "Tech",    "AMZN": "Consumer","TSLA": "Auto",  "AMD":   "Tech",
    "INTC": "Tech",    "QCOM": "Tech",    "MU":   "Tech",  "AVGO":  "Tech",
    "CRM":  "Tech",    "NOW":  "Tech",    "SNOW": "Tech",  "PLTR":  "Tech",
    "JPM":  "Finance", "BAC":  "Finance", "GS":   "Finance","MS":   "Finance",
    "XOM":  "Energy",  "CVX":  "Energy",  "SPY":  "Index", "QQQ":   "Index",
}

_SELL_REASON: dict[str, str] = {
    "SELL":           "Target exit",
    "SELL_STOP":      "Stop-loss triggered",
    "SELL_TP":        "Take-profit hit",
    "SELL_TRAIL":     "Trailing stop hit",
    "SELL_TRIM":      "Oversize trim",
    "SELL_TIME":      "Time-based exit",
    "SELL_ENSEMBLE":  "Signal deteriorated",
    "SELL_RECONCILE": "Reconciled on startup",
}


def _risk_level(vix: float, regime: str) -> tuple[str, str]:
    r = regime.lower()
    if vix > 30 or "bear" in r:
        return "High", LOSS
    elif vix > 20 or any(x in r for x in ["ranging", "neutral"]):
        return "Medium", NEURAL
    return "Low", GAIN


# ── Render: XGBoost feature importance chart ──────────────────────────────────
def render_feature_importance_chart():
    try:
        import json as _json
        import plotly.graph_objects as go
        fig = go.Figure()
        fi_path = "models/feature_importance.json"
        if not os.path.exists(fi_path):
            fig.add_annotation(
                text="Feature importance not yet available — run scripts/train_model.py first, then push models to HF.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=12))
        else:
            with open(fi_path) as fh:
                importances = _json.load(fh)
            top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
            # Reverse so highest importance is at top of horizontal bar chart
            features = [_FI_LABELS.get(k, k) for k, _ in reversed(top)]
            values   = [v for _, v in reversed(top)]
            max_v    = top[0][1] if top else 1.0
            colors   = [GAIN if v >= max_v * 0.5 else
                        (PRIMARY if v >= max_v * 0.25 else TEXT2) for v in values]
            fig.add_trace(go.Bar(
                x=values, y=features, orientation="h",
                marker_color=colors, marker_line=dict(width=0),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                name="Importance",
            ))
        fig.update_layout(
            title=dict(
                text="Which signals drive the AI's BUY decisions  <span style='font-size:{FONT_LABEL};'>— longer bar = more influence on each trade</span>",
                font=dict(color=TEXT1, size=13), x=0.01),
            xaxis=dict(title="Importance (normalised gain)", **PLOTLY_LAYOUT["xaxis"],
                       tickfont=dict(color=TEXT2)),
            yaxis=dict(title="", **PLOTLY_LAYOUT["yaxis"], tickfont=dict(color=TEXT1)),
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            height=380,
        )
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                             if k not in ("xaxis", "yaxis")}, height=380)
        fig.add_annotation(text=f"Chart error: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=LOSS))
        return fig


# ── Render: model validation report ──────────────────────────────────────────
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


# ── Render: dashboard hero (Bloomberg-style 4-pack + status bar) ─────────────
@safe_render("Dashboard Hero")
def render_dashboard_hero() -> str:
    d = get_data()
    open_syms = d["open_pos"]
    prices    = d["prices"]
    total_invested = sum(v["invested"] for v in open_syms.values())
    total_cur      = sum(v["shares"] * prices.get(s, 0.0) for s, v in open_syms.items())
    total_pnl      = total_cur - total_invested
    pnl_pct_all    = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
    pnl_color      = GAIN if total_pnl >= 0 else LOSS
    portfolio_val  = d["portfolio"]
    pnl_sign       = "+" if total_pnl >= 0 else ""
    hero_chg       = (f'{pnl_sign}${total_pnl:,.2f} ({pnl_pct_all:+.2f}%)'
                      if total_invested > 0 else "No open positions")

    avg_conf   = d.get("avg_confidence", 0.0)
    conf_str   = f"{avg_conf*100:.0f}%" if avg_conf > 0 else "—"
    conf_color = GAIN if avg_conf >= 0.75 else (NEURAL if avg_conf >= 0.60 else TEXT2)

    vix = d.get("vix", 0.0)
    vix_str = f"{vix:.1f}" if vix > 0 else "—"
    if vix == 0: vix_color = TEXT2
    elif vix < 15: vix_color = GAIN
    elif vix < 25: vix_color = NEURAL
    else: vix_color = LOSS

    mkt_label, mkt_color = _market_status()

    def _big(label, value, sub, color):
        return (
            f'<div class="nt-card" style="padding:20px 18px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:10px;">{label}</div>'
            f'<div style="font-size:{FONT_HERO};font-weight:700;letter-spacing:-1.5px;'
            f'color:{color};line-height:1;">{value}</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:6px;">{sub}</div>'
            f'</div>'
        )

    # ── Portfolio Health Score ──────────────────────────────────────────────
    pv_float = 0.0
    try:
        pv_float = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_portfolio_health_hero: {exc}")

    cash_pct_h = ((pv_float - total_invested) / pv_float * 100) if pv_float > 0 else 100.0

    max_dd_h = 0.0
    df_h = d["trades_df"]
    if not df_h.empty and "portfolio_value" in df_h.columns:
        vals_h = df_h["portfolio_value"].dropna()
        if len(vals_h) > 1:
            peak_h  = vals_h.cummax()
            max_dd_h = float(((peak_h - vals_h) / peak_h.replace(0, float("nan"))).max()) * 100

    max_conc_h = 0.0
    if open_syms and pv_float > 0:
        for _s, _p in open_syms.items():
            _cur = d["prices"].get(_s, 0.0)
            _val = _p["shares"] * _cur if _cur > 0 else _p["invested"]
            max_conc_h = max(max_conc_h, _val / pv_float * 100)

    _vix_pts  = 25 if vix < 15  else (15 if vix < 25  else 5)
    _cash_pts = 25 if cash_pct_h > 30 else (15 if cash_pct_h > 15 else 5)
    _conc_pts = 25 if max_conc_h < 15 else (15 if max_conc_h < 25 else 5)
    _dd_pts   = 25 if max_dd_h < 3  else (15 if max_dd_h < 8   else 5)
    health    = _vix_pts + _cash_pts + _conc_pts + _dd_pts

    health_c = GAIN if health >= 75 else (NEURAL if health >= 50 else LOSS)

    _components = [
        ("VIX",          _vix_pts),
        ("Cash Reserve", _cash_pts),
        ("Concentration", _conc_pts),
        ("Drawdown",     _dd_pts),
    ]
    weakest_name, weakest_pts = min(_components, key=lambda x: x[1])
    if weakest_pts == 25:
        weak_sub = "All risk factors look healthy"
    else:
        weak_sub = f"⚠ {weakest_name} is your biggest risk"

    health_card = (
        f'<div class="nt-card" style="padding:20px 18px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:10px;">Portfolio Health</div>'
        f'<div style="font-size:{FONT_HERO};font-weight:700;letter-spacing:-1.5px;'
        f'color:{health_c};line-height:1;">{health}<span style="font-size:{FONT_VALUE};'
        f'color:{TEXT2};font-weight:400;">/100</span></div>'
        f'<div style="margin:8px 0 6px;background:{BORDER};border-radius:3px;height:4px;">'
        f'<div style="background:{health_c};height:100%;width:{health}%;'
        f'border-radius:3px;transition:width .4s;"></div></div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{weak_sub}</div>'
        f'</div>'
    )

    val_color = pnl_color if total_invested > 0 else TEXT1
    cards = (
        f'<div class="nt-cards">'
        + _big("Portfolio Value",  portfolio_val,     f"Unrealized: {hero_chg}", val_color)
        + _big("Open Positions",   str(len(open_syms)), "Stocks held now (max 8)",  TEXT1)
        + _big("AI Confidence",    conf_str,          "Avg signal strength · last 5 buys", conf_color)
        + _big("VIX",              vix_str,           "Fear gauge · <15 calm · >30 fear",  vix_color)
        + health_card
        + f'</div>'
    )

    status = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">'
        f'Updated &nbsp;<strong style="color:{TEXT1};">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:{FONT_LABEL};">'
        f'{mkt_label}</span></span>'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">60s refresh &nbsp;·&nbsp; Paper money only</span>'
        f'</div>'
    )
    return f'<div class="nt nt-wrap">{cards}{status}</div>'


# ── PANEL 1: Portfolio Health Hero ────────────────────────────────────────────
@safe_render("Portfolio Health")
def render_portfolio_health_hero() -> str:
    d    = get_data()
    h    = get_portfolio_health(d)
    mkt_label, mkt_color = _market_status()

    score    = h["total"]
    grade    = h["grade"]
    gl       = h["grade_label"]
    risk_txt = h["biggest_risk"]
    comps    = h.get("components", {})

    grade_c = (GAIN if score >= 80 else (NEURAL if score >= 60 else LOSS))

    # Grade pill
    grade_pill = (
        f'<div style="display:inline-flex;align-items:center;gap:8px;">'
        f'<span style="font-size:{FONT_HERO};font-weight:800;color:{grade_c};'
        f'letter-spacing:-2px;line-height:1;">{grade}</span>'
        f'<div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;">{gl}</div>'
        f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{TEXT1};line-height:1.1;">'
        f'{score}<span style="font-size:{FONT_LABEL};color:{TEXT2};font-weight:400;">/100</span></div>'
        f'</div>'
        f'</div>'
    )

    # Score bar
    bar = (
        f'<div style="margin:10px 0 6px;background:{BORDER};border-radius:3px;height:5px;">'
        f'<div style="background:{grade_c};height:100%;width:{score}%;border-radius:3px;'
        f'transition:width .4s;"></div></div>'
    )

    # Component bars
    comp_rows = []
    comp_order = ["risk", "diversification", "cash", "momentum", "quality"]
    for k in comp_order:
        c = comps.get(k, {})
        lbl    = c.get("label", k.title())
        pts    = c.get("score", 0)
        maxpts = c.get("max", 25)
        detail = c.get("detail", "")
        pct    = int(pts / maxpts * 100) if maxpts > 0 else 0
        bar_c  = GAIN if pct >= 70 else (NEURAL if pct >= 40 else LOSS)
        comp_rows.append(
            f'<div style="margin-bottom:7px;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin-bottom:2px;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{lbl}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{bar_c};font-weight:600;">'
            f'{pts}/{maxpts}&nbsp;<span style="color:{TEXT2};font-weight:400;">{detail}</span></span>'
            f'</div>'
            f'<div style="background:{BORDER};border-radius:2px;height:3px;">'
            f'<div style="background:{bar_c};height:100%;width:{pct}%;border-radius:2px;"></div>'
            f'</div>'
            f'</div>'
        )
    comp_html = "".join(comp_rows)

    # Biggest risk callout
    risk_icon = "⚠" if score < 80 else "✓"
    risk_color = LOSS if score < 60 else (NEURAL if score < 80 else GAIN)
    risk_callout = (
        f'<div style="margin-top:8px;padding:7px 10px;background:{SURFACE2};'
        f'border-left:3px solid {risk_color};border-radius:0 4px 4px 0;">'
        f'<span style="font-size:{FONT_LABEL};color:{risk_color};font-weight:600;">'
        f'{risk_icon} {risk_txt}</span>'
        f'</div>'
    )

    # Strengths (max 2)
    strengths = h.get("strengths", [])[:2]
    str_html = ""
    if strengths:
        items = "".join(
            f'<span style="font-size:{FONT_LABEL};color:{GAIN};background:#0a2010;'
            f'border-radius:3px;padding:2px 6px;">{s}</span> '
            for s in strengths
        )
        str_html = f'<div style="margin-top:6px;">{items}</div>'

    # Portfolio value from existing data
    open_pos = d.get("open_pos", {})
    prices   = d.get("prices", {})
    total_inv = sum(v["invested"] for v in open_pos.values())
    total_cur = sum(v["shares"] * prices.get(s, 0.0) for s, v in open_pos.items())
    total_pnl = total_cur - total_inv
    pnl_pct   = (total_pnl / total_inv * 100) if total_inv > 0 else 0.0
    pnl_c     = GAIN if total_pnl >= 0 else LOSS
    pnl_sign  = "+" if total_pnl >= 0 else ""
    hero_chg  = (f'{pnl_sign}${total_pnl:,.2f} ({pnl_pct:+.2f}%)'
                 if total_inv > 0 else "No open positions")

    def _stat(label, val, color=None):
        c = color or TEXT1
        return (
            f'<div style="text-align:center;padding:10px 14px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.7px;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:{FONT_SECTION};font-weight:700;color:{c};">{val}</div>'
            f'</div>'
        )

    avg_conf   = d.get("avg_confidence", 0.0)
    conf_str   = f"{avg_conf*100:.0f}%" if avg_conf > 0 else "—"
    conf_c     = GAIN if avg_conf >= 0.75 else (NEURAL if avg_conf >= 0.60 else TEXT2)
    vix        = d.get("vix", 0.0)
    vix_str    = f"{vix:.1f}" if vix > 0 else "—"
    vix_c      = GAIN if vix < 15 else (NEURAL if vix < 25 else LOSS)

    stats_row = (
        f'<div style="display:flex;flex-wrap:wrap;border-top:1px solid {BORDER};margin-top:10px;">'
        + _stat("Portfolio", d.get("portfolio", "—"), TEXT1)
        + _stat("P&L", hero_chg, pnl_c)
        + _stat("Positions", str(len(open_pos)), TEXT1)
        + _stat("AI Conf.", conf_str, conf_c)
        + _stat("VIX", vix_str, vix_c)
        + f'</div>'
    )

    body = (
        f'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
        f'<div style="flex:0 0 auto;min-width:160px;">{grade_pill}{bar}'
        f'{risk_callout}{str_html}</div>'
        f'<div style="flex:1;min-width:200px;">{comp_html}</div>'
        f'</div>'
        f'{stats_row}'
    )

    timestamp = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">'
        f'Updated &nbsp;<strong style="color:{TEXT1};">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:{FONT_LABEL};">'
        f'{mkt_label}</span></span>'
        f'<span style="color:{TEXT2};font-size:{FONT_LABEL};">60s refresh</span>'
        f'</div>'
    )
    inner = (
        f'<div class="nt-card" style="padding:20px 18px;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:12px;">Portfolio Health</div>'
        f'{body}'
        f'</div>'
        f'{timestamp}'
    )
    return f'<div class="nt nt-wrap">{inner}</div>'


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


# ── Render: market intelligence (VIX / regime / confidence / sentiment) ───────
@safe_render("Market Intelligence")
def render_market_intelligence() -> str:
    d        = get_data()
    vix      = d.get("vix", 0.0)
    regime   = d.get("regime_raw", "—")
    avg_conf = d.get("avg_confidence", 0.0)
    sent     = d.get("sentiment_avg", 0.0)

    if vix == 0: vix_label, vix_color = "N/A", TEXT2
    elif vix < 15: vix_label, vix_color = "Low Fear", GAIN
    elif vix < 25: vix_label, vix_color = "Moderate", NEURAL
    elif vix < 35: vix_label, vix_color = "High Fear", LOSS
    else: vix_label, vix_color = "Extreme Fear", LOSS

    r_lower = regime.lower()
    if any(x in r_lower for x in ["bull", "trending up"]):   r_color = GAIN
    elif any(x in r_lower for x in ["bear", "trending down"]): r_color = LOSS
    else: r_color = NEURAL

    conf_color = GAIN if avg_conf >= 0.75 else (NEURAL if avg_conf >= 0.60 else TEXT2)

    if sent == 0: sent_label, sent_color = "No data", TEXT2
    elif sent > 0.05: sent_label, sent_color = "Positive", GAIN
    elif sent < -0.05: sent_label, sent_color = "Negative", LOSS
    else: sent_label, sent_color = "Neutral", NEURAL

    cards = (
        f'<div class="nt-cards">'
        + _stat_card("VIX", f"{vix:.1f}" if vix > 0 else "—",
                TEXT2, vix_color, f"{vix_label} · <15=calm, >30=fear", 0.00)
        + _stat_card("Market Regime", regime.replace("_", " ").title(),
                TEXT2, r_color, "AI-detected trend · drives position size", 0.06)
        + _stat_card("Signal Strength", f"{avg_conf*100:.0f}%" if avg_conf > 0 else "—",
                TEXT2, conf_color, "Avg confidence · last 5 buy signals", 0.12)
        + _stat_card("News Sentiment", sent_label,
                TEXT2, sent_color, "FinBERT score · recent headlines", 0.18)
        + f'</div>'
    )
    return f'<div class="nt nt-wrap">{_section("📡","Market Intelligence","live")}{cards}</div>'


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


# ── Render: signals tab (recent BUY signals with confidence + SHAP) ───────────
@safe_render("Signals")
def render_signals_tab() -> str:
    d    = get_data()
    buys = d.get("today_buy_signals", [])

    if not buys:
        _es = _empty_state("⚡", "No signals yet",
                           "The AI generates signals Mon–Fri 9:30am–4pm ET "
                           "when all entry gates pass.")
        return (f'<div class="nt nt-wrap">'
                f'{_section("⚡","AI Buy Signals","recent")}'
                f'{_card(_es)}'
                f'</div>')

    rows  = ""
    shown = buys[:20]
    for i, sig in enumerate(shown):
        ts      = sig.get("timestamp", "")
        sym     = sig.get("symbol", "—")
        price   = float(sig.get("price",          0.0) or 0.0)
        conf    = float(sig.get("ensemble_score",  0.0) or 0.0)
        regime  = str(sig.get("regime") or "—").replace("_", " ").title()
        drv_raw = sig.get("feature_drivers")
        driver_text = "—"
        try:
            import json as _j
            ds = _j.loads(drv_raw) if isinstance(drv_raw, str) else (drv_raw or [])
            parts = [
                f"{_FI_LABELS.get(f, f)}{'↑' if float(v) > 0 else '↓'}"
                for f, v in (ds or [])[:2]
            ]
            driver_text = " · ".join(parts) if parts else "—"
        except Exception as exc:
            logger.debug(f"parse_driver_text: {exc}")
        conf_pct = f"{conf*100:.0f}%" if conf > 0 else "—"
        conf_c   = GAIN if conf >= 0.75 else (NEURAL if conf >= 0.60 else TEXT2)
        td   = TD if i < len(shown) - 1 else TD0
        anim = f'style="animation:slideInRow .3s ease both;animation-delay:{i*0.04:.2f}s;"'
        rows += (
            f'<tr {anim}>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{_to_ct(ts)}</span></td>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_badge("BUY")}</td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{TEXT1};">'
            f'${price:.2f}</span></td>'
            f'<td {td}><span style="font-weight:700;color:{conf_c};">{conf_pct}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{regime}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{driver_text}</span></td>'
            f'</tr>'
        )
    note = f"last {len(shown)} signals · confidence = XGBoost + LSTM + sentiment ensemble"
    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.7;">'
        f'<b>Confidence</b> ≥75% strong · 60–75% moderate · &lt;60% weak &nbsp;·&nbsp;'
        f'<b>Top Drivers</b> show which indicators pushed the AI to BUY &nbsp;·&nbsp;'
        f'<b>Regime</b> = macro trend when signal fired'
        f'</div>'
    )
    table_inner = (
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Time (CT)</th><th {TH}>Symbol</th>'
        f'<th {TH}>Signal</th><th {TH}>Entry</th>'
        f'<th {TH}>Confidence</th><th {TH}>Regime</th>'
        f'<th {TH}>Top Drivers</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>' + help_block
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("⚡","AI Buy Signals", note)}'
            f'{_wrap(table_inner)}</div>')


# ── Render: risk controls panel ──────────────────────────────────────────────
@safe_render("Risk Panel")
def render_risk_panel() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]
    vix      = d.get("vix", 0.0)
    df       = d["trades_df"]

    # Portfolio value as float
    pv = 0.0
    try:
        pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_risk_panel: {exc}")

    total_invested = sum(v["invested"] for v in open_pos.values())
    cash_pct = ((pv - total_invested) / pv * 100) if pv > 0 else 100.0

    # Max drawdown from portfolio history
    max_dd = 0.0
    if not df.empty and "portfolio_value" in df.columns:
        vals = df["portfolio_value"].dropna()
        if len(vals) > 1:
            peak  = vals.cummax()
            max_dd = float(((peak - vals) / peak.replace(0, float("nan"))).max()) * 100

    # Daily loss (today's sells, average pnl_pct)
    daily_pnl = 0.0
    if not df.empty:
        today_str  = str(datetime.date.today())
        sells_today = df[
            df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE") &
            (df["date"].astype(str) == today_str)
        ]
        if not sells_today.empty and "pnl_pct" in sells_today.columns:
            daily_pnl = float(sells_today["pnl_pct"].mean()) * 100

    # Sector exposure
    sector_exp: dict[str, float] = {}
    for sym, pos in open_pos.items():
        cur = prices.get(sym, 0.0)
        val = pos["shares"] * cur if cur > 0 else pos["invested"]
        sector = _SECTOR_MAP.get(sym.upper(), "Other")
        sector_exp[sector] = sector_exp.get(sector, 0.0) + val
    total_eq = sum(sector_exp.values()) or 1.0
    sector_pcts = {s: v / total_eq * 100 for s, v in sorted(sector_exp.items(), key=lambda x: -x[1])}

    # Largest position concentration
    max_conc = 0.0
    if open_pos and pv > 0:
        for sym, pos in open_pos.items():
            cur = prices.get(sym, 0.0)
            val = pos["shares"] * cur if cur > 0 else pos["invested"]
            max_conc = max(max_conc, val / pv * 100)

    # Overall risk
    risk_pts = sum([vix > 25, max_dd > 8, cash_pct < 15, max_conc > 20])
    if risk_pts >= 3: overall_risk, risk_c = "High",   LOSS
    elif risk_pts >= 1: overall_risk, risk_c = "Medium", NEURAL
    else: overall_risk, risk_c = "Low", GAIN

    dd_c  = GAIN if max_dd < 5 else (NEURAL if max_dd < 12 else LOSS)
    dl_c  = GAIN if daily_pnl >= 0 else (NEURAL if daily_pnl > -2 else LOSS)
    cc_c  = GAIN if max_conc < 15 else (NEURAL if max_conc < 20 else LOSS)
    ca_c  = GAIN if cash_pct > 30 else (NEURAL if cash_pct > 15 else LOSS)

    cards = (
        f'<div class="nt-cards">'
        + _stat_card("Portfolio Risk",  overall_risk,       TEXT2, risk_c,
                "VIX + drawdown + concentration", 0.00)
        + _stat_card("Max Drawdown",    f"{max_dd:.1f}%",   TEXT2, dd_c,
                "Peak-to-trough all-time",        0.06)
        + _stat_card("Today's P&L",     f"{daily_pnl:+.2f}%", TEXT2, dl_c,
                "Realised from closed trades",    0.12)
        + _stat_card("Cash Reserve",    f"{cash_pct:.1f}%", TEXT2, ca_c,
                "Uninvested capital buffer",      0.18)
        + f'</div>'
    )

    sector_rows = ""
    for sector, pct in list(sector_pcts.items())[:5]:
        bar_c = LOSS if pct > 50 else (NEURAL if pct > 30 else GAIN)
        sector_rows += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};width:70px;flex-shrink:0;">{sector}</span>'
            f'<div style="background:{BORDER};border-radius:2px;height:5px;flex:1;overflow:hidden;">'
            f'<div style="background:{bar_c};height:100%;width:{min(pct,100):.0f}%;"></div></div>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT1};width:36px;text-align:right;">{pct:.0f}%</span>'
            f'</div>'
        )
    if not sector_rows:
        sector_rows = f'<div style="color:{TEXT2};font-size:{FONT_LABEL};">No open positions — fully in cash</div>'

    note = (f'Concentration: <span style="color:{cc_c};font-weight:700;">{max_conc:.1f}%</span>'
            f' largest position')
    return (f'<div class="nt nt-wrap">'
            f'{_section("🛡","Risk Controls","real-time")}'
            f'{cards}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;padding:14px 16px;margin-top:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;">Sector Exposure</div>'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{note}</div>'
            f'</div>'
            f'{sector_rows}</div></div>')


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
@safe_render("Timeline")
def render_timeline() -> str:
    d  = get_data()
    df = d["trades_df"]
    if df.empty:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:40px;font-size:{FONT_VALUE};">'
                 f'No decisions yet. The AI trades Mon–Fri 9:30am–4pm ET.</div>')
        return f'<div class="nt nt-wrap">{_section("🕐","AI Decision Feed","live")}{_wrap(empty)}</div>'

    recent = df.tail(30).iloc[::-1]
    items  = ""
    for i, (_, row) in enumerate(recent.iterrows()):
        action  = str(row.get("action", ""))
        sym     = str(row.get("symbol", "—"))
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


# ── Symbol choices + detail drilldown ─────────────────────────────────────────
def _get_symbol_choices() -> list[str]:
    d = get_data()
    syms = list(d["open_pos"].keys())
    if not d["trades_df"].empty:
        for s in d["trades_df"]["symbol"].unique():
            if s not in syms:
                syms.append(s)
    return syms[:20]


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

    cur_price = prices.get(symbol, 0.0)
    pos       = open_pos.get(symbol)

    conf    = float(lb.get("ensemble_score",  0.0) or 0.0) if lb is not None else 0.0
    xgb_p   = float(lb.get("xgb_prob",        0.0) or 0.0) if lb is not None else 0.0
    lstm_p  = float(lb.get("lstm_prob",        0.0) or 0.0) if lb is not None else 0.0
    sent    = float(lb.get("sentiment_score",  0.0) or 0.0) if lb is not None else 0.0
    regime  = (str(lb.get("regime") or "—").replace("_", " ").title() if lb is not None
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

    pnl_str, pnl_c = "—", TEXT2
    if pos and entry > 0 and cur_price > 0:
        pnl_v   = (cur_price - entry) / entry * 100
        pnl_str = f"{pnl_v:+.1f}%"
        pnl_c   = GAIN if pnl_v >= 0 else LOSS

    status_lbl = "OPEN POSITION" if pos else "RECENTLY TRADED"
    status_c   = GAIN if pos else TEXT2
    conf_pct   = f"{conf*100:.0f}%" if conf > 0 else "—"

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
        cur_str = f"${cur_price:.2f}" if cur_price > 0 else "—"
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
    # ── Why Panel — recommendation engine signals ──────────────────────────────
    _pa  = get_portfolio_action(symbol, d)
    _exp = get_recommendation_explanation(symbol, d)
    _sz2 = get_position_sizing(symbol, d)
    _ac  = _pa.get("action", "HOLD")
    _pa_conf = _pa.get("confidence", 0)
    _pa_reason = _pa.get("reason", "—")
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

    _dol_disp = _sz2.get("dollar_display", "—")
    _tgt_w    = _sz2.get("target_weight", 0.0)
    _sh = (f"Target {_tgt_w:.0f}% · {_dol_disp}" if _tgt_w > 0 else
           "Max 12% allocation" if _pa_conf >= 75 else
           "Max 8% allocation"  if _pa_conf >= 60 else "Max 5% allocation")

    action_card_html = (
        f'<div style="background:{BG};border-radius:6px;padding:12px 14px;margin-bottom:14px;">'
        f'<div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">'
        # Action badge
        f'<div style="flex:0 0 auto;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:4px;">AI Action</div>'
        f'<span style="display:inline-block;background:{_ac_bg};border:1px solid {_ac_c};'
        f'color:{_ac_c};font-size:{FONT_VALUE};font-weight:700;letter-spacing:.5px;'
        f'padding:4px 14px;border-radius:4px;">{_ac}</span>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:4px;max-width:140px;">{_pa_reason}</div>'
        f'</div>'
        # Conviction bar + signals
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
        # Sizing
        f'<div style="text-align:right;flex:0 0 auto;">'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:4px;">Sizing Guidance</div>'
        f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{_sh}</div>'
        f'</div></div></div>'
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


# ── Render: since-yesterday comparison panel ─────────────────────────────────
@timed(_logger)
@safe_render("Since Yesterday")
def render_whats_changed() -> str:
    today     = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    d         = datetime.date.today()
    date_label = f"{d.strftime('%B')} {d.day}"

    def _empty(msg: str) -> str:
        inner = (f'<div style="color:{TEXT2};text-align:center;padding:24px;font-size:{FONT_LABEL};">{msg}</div>')
        return (f'<div class="nt nt-wrap">'
                f'{_section("📅", "Since Yesterday", date_label)}'
                f'{_wrap(inner)}</div>')

    if not os.path.exists(DB_PATH):
        return _empty("No trade data yet.")

    try:
        con = sqlite3.connect(DB_PATH)

        def _latest_per_symbol(date_str: str) -> list:
            return con.execute(
                "SELECT t.symbol, t.ensemble_score, t.regime, t.sentiment_score, t.portfolio_value "
                "FROM trades t "
                "INNER JOIN (SELECT symbol, MAX(id) AS mid FROM trades "
                "            WHERE date(timestamp) = ? GROUP BY symbol) m "
                "ON t.id = m.mid",
                (date_str,),
            ).fetchall()

        today_rows = _latest_per_symbol(today)
        yest_rows  = _latest_per_symbol(yesterday)

        # Portfolio bookends: yesterday's last overall value vs today's last
        yest_pv_row  = con.execute(
            "SELECT portfolio_value FROM trades WHERE date(timestamp) = ? "
            "AND portfolio_value > 0 ORDER BY id DESC LIMIT 1", (yesterday,)
        ).fetchone()
        today_pv_row = con.execute(
            "SELECT portfolio_value FROM trades WHERE portfolio_value > 0 "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
    except Exception as exc:
        logger.warning(f"render_whats_changed DB error: {exc}")
        return _empty("Could not load comparison data.")

    if not yest_rows:
        return _empty("First session — no comparison available yet.")

    yest_map  = {r[0]: {"score": float(r[1] or 0), "regime": r[2] or "",
                         "sent": float(r[3] or 0)} for r in yest_rows}
    today_map = {r[0]: {"score": float(r[1] or 0), "regime": r[2] or "",
                         "sent": float(r[3] or 0)} for r in today_rows}

    # ── Portfolio delta summary ────────────────────────────────────────────────
    pv_html = ""
    if yest_pv_row and today_pv_row:
        yv = float(yest_pv_row[0] or 0)
        tv = float(today_pv_row[0] or 0)
        if yv > 0:
            delta = tv - yv
            pct   = delta / yv * 100
            d_c   = GAIN if delta >= 0 else LOSS
            icon  = "📈" if delta >= 0 else "📉"
            word  = "up" if delta >= 0 else "down"
            pv_html = (
                f'<div style="background:{BG};border:1px solid {d_c}33;border-radius:6px;'
                f'padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:{FONT_VALUE};">{icon}</span>'
                f'<span style="font-size:{FONT_VALUE};color:{TEXT1};">Portfolio '
                f'<strong style="color:{d_c};">{word} ${abs(delta):,.2f} ({pct:+.2f}%)</strong>'
                f' since yesterday</span></div>'
            )

    # ── Per-symbol change rows ─────────────────────────────────────────────────
    def _sent_label(v: float) -> str:
        return "Positive" if v > 0.05 else ("Negative" if v < -0.05 else "Neutral")

    rows_html    = ""
    changes_seen = False

    for sym in sorted(set(yest_map) & set(today_map)):
        y = yest_map[sym]
        t = today_map[sym]

        score_delta = t["score"] - y["score"]
        regime_y    = y["regime"].replace("_", " ").title()
        regime_t    = t["regime"].replace("_", " ").title()
        sent_y      = _sent_label(y["sent"])
        sent_t      = _sent_label(t["sent"])

        changes: list[tuple[str, str, str]] = []

        if abs(score_delta) > 0.05:
            arrow = (f'<span style="color:{GAIN};font-weight:700;">↑</span>' if score_delta > 0
                     else f'<span style="color:{LOSS};font-weight:700;">↓</span>')
            changes.append(("Confidence", arrow,
                            f'{y["score"] * 100:.0f}% → {t["score"] * 100:.0f}%'))

        if regime_y and regime_t and regime_y != regime_t:
            changes.append(("Regime",
                            f'<span style="color:{TEXT2};font-weight:700;">→</span>',
                            f'{regime_y} → {regime_t}'))

        if sent_y != sent_t:
            if sent_t == "Positive":
                s_arrow = f'<span style="color:{GAIN};font-weight:700;">↑</span>'
            elif sent_t == "Negative":
                s_arrow = f'<span style="color:{LOSS};font-weight:700;">↓</span>'
            else:
                s_arrow = f'<span style="color:{TEXT2};font-weight:700;">→</span>'
            changes.append(("Sentiment", s_arrow, f'{sent_y} → {sent_t}'))

        if not changes:
            continue

        changes_seen = True
        for i, (metric, arrow_html, mag) in enumerate(changes):
            sym_cell = (f'<span style="font-family:Courier New,monospace;font-weight:700;'
                        f'color:{PRIMARY};font-size:{FONT_VALUE};">{sym}</span>') if i == 0 else ""
            rows_html += (
                f'<div style="display:grid;grid-template-columns:80px 100px 32px 1fr;'
                f'align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid {BORDER};">'
                f'<div>{sym_cell}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">{metric}</div>'
                f'<div style="text-align:center;font-size:{FONT_VALUE};">{arrow_html}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT1};">{mag}</div>'
                f'</div>'
            )

    if not changes_seen:
        rows_html = (
            f'<div style="color:{TEXT2};font-size:{FONT_LABEL};padding:12px 0;text-align:center;">'
            f'No significant changes since yesterday — AI signals are stable.</div>'
        )

    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📅", "Since Yesterday", date_label)}'
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;">'
        f'{pv_html}{rows_html}'
        f'</div></div>'
    )


# ── Portfolio performance period helpers ──────────────────────────────────────
_PERF_PERIODS = [
    ("1D",       1),
    ("1W",       7),
    ("1M",      30),
    ("3M",      90),
    ("YTD",     None),   # special: Jan 1 of current year
    ("1Y",     365),
    ("All Time", None),  # special: first DB record
]
_PERF_LABELS = {
    "1D":       "today",
    "1W":       "this week",
    "1M":       "this month",
    "3M":       "last 3 months",
    "YTD":      "year to date",
    "1Y":       "last year",
    "All Time": "since inception",
}


def _query_perf_stats() -> dict[str, tuple[float, float, str] | None]:
    """
    Returns {period_key: (start_val, end_val, start_date_label) | None}.
    None means insufficient data for that period.
    """
    if not os.path.exists(DB_PATH):
        return {k: None for k, _ in _PERF_PERIODS}
    try:
        con    = sqlite3.connect(DB_PATH)
        today  = datetime.date.today()
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Current (latest) portfolio value
        cur_row = con.execute(
            "SELECT portfolio_value FROM trades WHERE portfolio_value > 0 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not cur_row:
            con.close()
            return {k: None for k, _ in _PERF_PERIODS}
        cur_val = float(cur_row[0])

        # First ever portfolio value
        first_row = con.execute(
            "SELECT portfolio_value, timestamp FROM trades WHERE portfolio_value > 0 ORDER BY id ASC LIMIT 1"
        ).fetchone()
        first_val = float(first_row[0]) if first_row else None
        first_ts  = first_row[1][:10]  if first_row else None

        result: dict[str, tuple[float, float, str] | None] = {}

        for key, days in _PERF_PERIODS:
            if key == "All Time":
                if first_val is not None and first_val != cur_val:
                    result[key] = (first_val, cur_val, first_ts or "")
                else:
                    result[key] = None
                continue

            if key == "YTD":
                cutoff = datetime.date(today.year, 1, 1).isoformat()
            else:
                cutoff = (today - datetime.timedelta(days=days)).isoformat()

            # First DB record on or after the cutoff date (start-of-period proxy)
            row = con.execute(
                "SELECT portfolio_value, timestamp FROM trades "
                "WHERE portfolio_value > 0 AND date(timestamp) >= ? ORDER BY id ASC LIMIT 1",
                (cutoff,),
            ).fetchone()
            if row:
                start_val  = float(row[0])
                start_date = row[1][:10]
                result[key] = (start_val, cur_val, start_date)
            else:
                result[key] = None

        con.close()
        return result
    except Exception as exc:
        logger.warning(f"_query_perf_stats: {exc}")
        return {k: None for k, _ in _PERF_PERIODS}


def _perf_choices() -> list[str]:
    """Build Radio choices with inline % for display, e.g. '1M  +8.9%'."""
    stats = _query_perf_stats()
    choices = []
    for key, _ in _PERF_PERIODS:
        s = stats.get(key)
        if s and s[0] > 0:
            pct = (s[1] - s[0]) / s[0] * 100
            choices.append(f"{key}  {pct:+.1f}%")
        else:
            choices.append(f"{key}  —")
    return choices


@timed(_logger)
@safe_render("Portfolio Performance")
def render_portfolio_performance(period: str = "1M  —") -> str:
    # Strip the inline stat suffix so we always have a clean key
    key = period.split()[0] if period else "1M"

    stats = _query_perf_stats()
    cur_row_val = None
    first_any   = any(v is not None for v in stats.values())

    if not first_any:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:20px;font-size:{FONT_LABEL};">'
                 f'No portfolio history yet — data appears after the first trade.</div>')
        return f'<div class="nt nt-wrap">{empty}</div>'

    s = stats.get(key)

    # ── Strip: all period mini-badges (decorative, Radio handles selection) ────
    strip_items = ""
    for pk, _ in _PERF_PERIODS:
        ps = stats.get(pk)
        if ps and ps[0] > 0:
            pct = (ps[1] - ps[0]) / ps[0] * 100
            c   = GAIN if pct >= 0 else LOSS
            strip_items += (
                f'<div style="text-align:center;padding:8px 12px;background:{SURFACE};'
                f'border:1px solid {"" + PRIMARY if pk == key else BORDER};'
                f'border-radius:6px;min-width:60px;">'
                f'<div style="font-size:{FONT_LABEL};color:{"" + PRIMARY if pk == key else TEXT2};'
                f'font-weight:700;margin-bottom:4px;">{pk}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{c};font-weight:700;">{pct:+.1f}%</div>'
                f'</div>'
            )
        else:
            strip_items += (
                f'<div style="text-align:center;padding:8px 12px;background:{BG};'
                f'border:1px solid {BORDER};border-radius:6px;min-width:60px;opacity:0.4;">'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};font-weight:700;margin-bottom:4px;">{pk}</div>'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT2};">—</div>'
                f'</div>'
            )

    strip = (
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;">'
        f'{strip_items}</div>'
    )

    # ── Detail card for selected period ───────────────────────────────────────
    if not s:
        detail = (
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;'
            f'padding:20px;text-align:center;color:{TEXT2};font-size:{FONT_VALUE};">'
            f'Not enough data yet for <strong>{key}</strong> — the bot needs more trading history.</div>'
        )
    else:
        start_val, end_val, start_date = s
        delta     = end_val - start_val
        pct       = (delta / start_val * 100) if start_val > 0 else 0.0
        c         = GAIN if delta >= 0 else LOSS
        sign      = "+" if delta >= 0 else ""
        label_str = _PERF_LABELS.get(key, key.lower())

        detail = (
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-top:3px solid {c};border-radius:8px;padding:20px 24px;">'
            # Big headline
            f'<div style="font-size:{FONT_HERO};font-weight:700;color:{c};letter-spacing:-1px;'
            f'line-height:1;margin-bottom:8px;">'
            f'{sign}${abs(delta):,.2f}</div>'
            # Subline
            f'<div style="font-size:{FONT_VALUE};color:{TEXT2};margin-bottom:14px;">'
            f'{sign}{pct:.2f}% {label_str}</div>'
            # From → To
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">From</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{TEXT1};">${start_val:,.2f}</span>'
            f'<span style="font-size:{FONT_VALUE};color:{TEXT2};">→</span>'
            f'<span style="font-size:{FONT_VALUE};font-weight:700;color:{TEXT1};">${end_val:,.2f}</span>'
            f'</div>'
            # Start date
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};margin-top:10px;">'
            f'Period start: {start_date}</div>'
            f'</div>'
        )

    return f'<div class="nt nt-wrap">{strip}{detail}</div>'


# ── Render: today's trades timeline ─────────────────────────────────────────────
# ── PANEL 2: Today's Priority Actions — recommendation-based ─────────────────
@safe_render("Today's Actions")
def render_todays_actions() -> str:
    d        = get_data()
    open_pos = d.get("open_pos", {})

    if not open_pos:
        return f'<div class="nt nt-wrap">{_section("⚡","Priority Actions","What to do right now")}{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>'

    _ACTION_ORDER = {"EXIT": 0, "SELL": 1, "TRIM": 2, "WATCH": 3, "ADD": 4, "BUY": 5, "HOLD": 6}
    recommendations: list[dict] = []
    for sym in open_pos:
        rec    = get_portfolio_action(sym, d)
        sz     = get_position_sizing(sym, d)
        rec["symbol"]      = sym
        rec["sizing_hint"] = sz.get("dollar_display", "—")
        rec["shares_hint"] = sz.get("shares_display", "—")
        recommendations.append(rec)

    # Sort: urgent actions first
    recommendations.sort(key=lambda r: (_ACTION_ORDER.get(r.get("action", "HOLD"), 9),
                                         -r.get("confidence", 0)))

    _badge_color = {
        "EXIT":  (LOSS,      "#2a0a0a"),
        "SELL":  (LOSS,      "#2a0a0a"),
        "TRIM":  ("#f59e0b", "#2a1f08"),
        "WATCH": (NEURAL,    "#1a1030"),
        "ADD":   (GAIN,      "#0a2010"),
        "BUY":   (GAIN,      "#0a2010"),
        "HOLD":  (TEXT2,     SURFACE2),
    }

    n    = len(recommendations)
    rows = ""
    for i, rec in enumerate(recommendations):
        sym      = rec["symbol"]
        action   = rec.get("action", "HOLD")
        conf     = rec.get("confidence", 0)
        reason   = rec.get("reason", "—")
        sizing   = rec.get("sizing_hint", "—")
        urgency  = rec.get("urgency", "low")
        txt_c, bg_c = _badge_color.get(action, (TEXT2, SURFACE2))
        td = TD if i < n - 1 else TD0

        badge_html = (
            f'<span style="display:inline-block;padding:2px 8px;border-radius:3px;'
            f'background:{bg_c};color:{txt_c};font-size:{FONT_LABEL};font-weight:700;'
            f'letter-spacing:.5px;">{action}</span>'
        )
        conf_c = GAIN if conf >= 75 else (NEURAL if conf >= 60 else TEXT2)
        urg_c  = LOSS if urgency == "high" else (NEURAL if urgency == "medium" else TEXT2)
        urg_dot = (
            f'<span style="width:6px;height:6px;border-radius:50%;'
            f'background:{urg_c};display:inline-block;margin-left:4px;"></span>'
        )
        rows += (
            f'<tr>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}>{badge_html}{urg_dot}</td>'
            f'<td {td}><span style="font-weight:700;color:{conf_c};">{conf}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{reason}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT1};font-family:Courier New,monospace;">'
            f'{sizing}</span></td>'
            f'</tr>'
        )

    urgent_count = sum(1 for r in recommendations if r.get("urgency") == "high")
    note = (f"{urgent_count} urgent · {n} positions" if urgent_count else
            f"{n} position{'s' if n != 1 else ''} · sorted by priority")
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Action</th>'
        f'<th {TH}>Conf.</th><th {TH}>Reason</th><th {TH}>Sizing</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("⚡","Priority Actions",note)}{table}</div>'


# ── Render: portfolio actions — called internally by render_decision_center ───
@safe_render("Portfolio Actions")
def render_portfolio_actions() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]
    df       = d["trades_df"]

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("🎯","Portfolio Actions","AI recommendation per position")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_portfolio_actions: {exc}")

    # Latest ensemble score per open symbol from trades_df (no extra DB call)
    _ens: dict[str, float] = {}
    if not df.empty:
        buys = df[df["action"] == "BUY"]
        for sym in open_pos:
            sym_buys = buys[buys["symbol"] == sym]
            if not sym_buys.empty:
                _ens[sym] = float(sym_buys.iloc[-1].get("ensemble_score", 0.0) or 0.0)

    _AMBER = "#f59e0b"
    rows  = ""
    items = list(open_pos.items())
    for i, (sym, pos) in enumerate(items):
        cur      = prices.get(sym, 0.0)
        invested = pos["invested"]
        cur_val  = pos["shares"] * cur
        pnl_pct  = ((cur_val - invested) / invested * 100) if invested > 0 else 0.0
        pos_pct  = (cur_val / _pv * 100) if _pv > 0 else 0.0
        ens      = _ens.get(sym, 1.0)

        sz_pts   = 30 if pos_pct > 25 else (20 if pos_pct > 15 else (10 if pos_pct > 10 else 0))
        pr_pts   = (30 if pnl_pct > 50 else (20 if pnl_pct > 25 else (10 if pnl_pct > 10 else 0))) if pnl_pct > 0 else 0
        cf_pts   = 25 if ens < 0.55 else (15 if ens < 0.65 else 0)
        dd_pts   = (15 if pnl_pct < -8 else (10 if pnl_pct < -5 else 0)) if pnl_pct < 0 else 0
        total    = sz_pts + pr_pts + cf_pts + dd_pts

        scored: list[tuple[int, str]] = []
        if sz_pts: scored.append((sz_pts, "Position oversized"))
        if pr_pts:
            scored.append((pr_pts, "Profit > 50%" if pnl_pct > 50 else ("Profit > 25%" if pnl_pct > 25 else "Profit > 10%")))
        if cf_pts: scored.append((cf_pts, "AI confidence weakening"))
        if dd_pts: scored.append((dd_pts, "Drawdown risk"))
        scored.sort(key=lambda x: -x[0])
        reason = scored[0][1] if scored else "All metrics healthy"

        if total <= 30:   label, bc, bbg = "HOLD",  GAIN,   "#0a2010"
        elif total <= 59: label, bc, bbg = "WATCH", NEURAL, "#1a1030"
        elif total <= 79: label, bc, bbg = "TRIM",  _AMBER, "#2a1f08"
        else:             label, bc, bbg = "EXIT",  LOSS,   "#2a0a0a"

        pnl_c   = GAIN if pnl_pct >= 0 else LOSS
        ens_c   = GAIN if ens >= 0.75 else (NEURAL if ens >= 0.60 else TEXT2)
        ens_str = f"{ens*100:.0f}%" if ens > 0 else "—"
        td = TD if i < len(items) - 1 else TD0
        rows += (
            f'<tr><td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_action_badge(label)}</td>'
            f'<td {td}><span style="font-weight:700;color:{ens_c};">{ens_str}</span></td>'
            f'<td {td}><span style="font-weight:700;color:{pnl_c};">{pnl_pct:+.1f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{reason}</span></td>'
            f'</tr>'
        )

    note  = f"{len(items)} position{'s' if len(items) != 1 else ''}"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Action</th><th {TH}>AI Score</th>'
        f'<th {TH}>P&amp;L</th><th {TH}>Top Reason</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("🎯","Portfolio Actions","AI recommendation per position")}{table}</div>')


# ── Render: position sizing recommendations ────────────────────────────────────
@safe_render("Position Sizing")
def render_position_sizing() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]
    df       = d["trades_df"]

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("📐","Position Sizing","conviction-based target allocation")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_position_sizing: {exc}")

    _ens: dict[str, float] = {}
    if not df.empty:
        buys = df[df["action"] == "BUY"]
        for sym in open_pos:
            sym_buys = buys[buys["symbol"] == sym]
            if not sym_buys.empty:
                _ens[sym] = float(sym_buys.iloc[-1].get("ensemble_score", 0.65) or 0.65)

    rows  = ""
    items = list(open_pos.items())
    for i, (sym, pos) in enumerate(items):
        cur      = prices.get(sym, 0.0)
        cur_val  = pos["shares"] * cur if cur > 0 else pos["invested"]
        cur_pct  = (cur_val / _pv * 100) if _pv > 0 else 0.0
        ens      = _ens.get(sym, 0.65)

        if ens >= 0.75:   target_pct, rationale = 12.0, "High conviction"
        elif ens >= 0.65: target_pct, rationale = 8.0,  "Moderate conviction"
        elif ens >= 0.55: target_pct, rationale = 5.0,  "Low conviction"
        else:             target_pct, rationale = 3.0,  "Very low — consider exit"

        delta = target_pct - cur_pct
        if abs(delta) < 0.5:   adj_lbl, adj_c = "On target",           TEXT2
        elif delta > 0:        adj_lbl, adj_c = f"Add +{delta:.1f}%",  GAIN
        else:                  adj_lbl, adj_c = f"Reduce {delta:.1f}%", "#f59e0b"

        target_val = (_pv * target_pct / 100) if _pv > 0 else 0.0
        val_hint   = f"(~${target_val:,.0f})" if target_val > 0 else ""
        td = TD if i < len(items) - 1 else TD0
        rows += (
            f'<tr><td {td}>{_sym(sym)}</td>'
            f'<td {td}><span style="font-weight:700;color:{TEXT1};">{cur_pct:.1f}%</span></td>'
            f'<td {td}><span style="font-weight:700;color:{NEURAL};">{target_pct:.0f}%</span></td>'
            f'<td {td}><span style="font-weight:700;color:{adj_c};">{adj_lbl}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{rationale}</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};margin-left:6px;">{val_hint}</span></td>'
            f'</tr>'
        )

    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:{FONT_LABEL};color:{TEXT2};line-height:1.6;">'
        f'Target derived from AI ensemble score &nbsp;·&nbsp; '
        f'75%+ = 12% &nbsp;·&nbsp; 65%+ = 8% &nbsp;·&nbsp; 55%+ = 5% &nbsp;·&nbsp; &lt;55% = 3%'
        f'</div>'
    )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Current</th><th {TH}>Target</th>'
        f'<th {TH}>Adjustment</th><th {TH}>Rationale</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>' + help_block
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("📐","Position Sizing","conviction-based target allocation")}{table}</div>')


# ── Render: AI investment committee ─────────────────────────────────────────────
@safe_render("AI Committee")
def render_ai_committee() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    df       = d["trades_df"]

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("🏛","AI Committee","XGBoost · LSTM · Sentiment votes")}'
                f'{_card(_empty_state("🏛", "Fully in cash", "Committee convenes once the bot enters positions."))}</div>')

    # Extract latest BUY scores per symbol from trades_df
    _votes: dict[str, dict] = {}
    if not df.empty:
        buys = df[df["action"] == "BUY"]
        for sym in open_pos:
            sym_buys = buys[buys["symbol"] == sym]
            if not sym_buys.empty:
                lb = sym_buys.iloc[-1]
                _votes[sym] = {
                    "xgb":  float(lb.get("xgb_prob",        0.0) or 0.0),
                    "lstm": float(lb.get("lstm_prob",        0.0) or 0.0),
                    "sent": float(lb.get("sentiment_score",  0.0) or 0.0),
                }

    def _vote_chip(label: str, pct_val: float, threshold: float = 0.60) -> str:
        vote  = "BUY" if pct_val >= threshold else ("HOLD" if pct_val >= 0.45 else "SELL")
        c     = GAIN if vote == "BUY" else (TEXT2 if vote == "HOLD" else LOSS)
        v_str = f"{pct_val*100:.0f}%" if pct_val > 0 else "—"
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px;'
            f'background:{BG};border-radius:6px;padding:8px 10px;min-width:64px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT2};text-transform:uppercase;letter-spacing:.6px;">{label}</div>'
            f'<div style="font-size:{FONT_VALUE};font-weight:700;color:{c};">{v_str}</div>'
            f'<div style="font-size:{FONT_LABEL};font-weight:700;color:{c};">{vote}</div></div>'
        )

    rows_html = ""
    for i, (sym, _pos) in enumerate(list(open_pos.items())[:8]):
        v     = _votes.get(sym, {})
        xgb   = v.get("xgb",  0.0)
        lstm  = v.get("lstm", 0.0)
        sent  = v.get("sent", 0.0)
        sent_n = min(max((sent + 1) / 2, 0.0), 1.0)   # -1..1 → 0..1

        buy_votes = (1 if xgb >= 0.60 else 0) + (1 if lstm >= 0.60 else 0) + (1 if sent_n >= 0.55 else 0)
        verdict_c = GAIN if buy_votes >= 2 else (NEURAL if buy_votes == 1 else LOSS)
        verdict   = f"{buy_votes}/3 BUY" if buy_votes > 0 else "No BUY votes"
        no_data   = not v

        border_b = f'border-bottom:1px solid {BORDER};' if i < min(len(open_pos), 8) - 1 else ''
        if no_data:
            chip_html = (f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">'
                         f'No BUY trade on record yet</span>')
        else:
            chip_html = (
                _vote_chip("XGBoost", xgb)
                + _vote_chip("LSTM", lstm)
                + _vote_chip("Sentiment", sent_n, 0.55)
            )
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:12px;padding:12px 14px;{border_b}">'
            f'{_sym(sym)}'
            f'<div style="display:flex;gap:6px;flex:1;">{chip_html}</div>'
            f'<div style="text-align:right;min-width:80px;">'
            f'<div style="font-size:{FONT_LABEL};font-weight:700;color:{verdict_c};">'
            f'{"—" if no_data else verdict}</div></div></div>'
        )

    if not rows_html:
        rows_html = _empty_state("🏛", "No positions", "Committee convenes once the bot enters positions.")

    return (f'<div class="nt nt-wrap">'
            f'{_section("🏛","AI Committee","3-model vote per open position")}'
            f'{_wrap(rows_html)}</div>')


# ── PANEL 3: Sell Analysis — called internally by render_decision_center ──────
@safe_render("Sell Analysis")
def render_sell_analysis() -> str:
    d        = get_data()
    open_pos = d.get("open_pos", {})

    if not open_pos:
        return f'<div class="nt nt-wrap">{_section("📉","Sell Analysis","When should I sell?")}{_card(_empty_state("💰", "Fully in cash", "Sell analysis runs once the bot holds positions."))}</div>'

    _REC_ORDER = {"EXIT": 0, "SELL": 1, "TRIM": 2, "WATCH": 3, "HOLD": 4}

    analyses = []
    for sym in open_pos:
        sa = get_sell_analysis(sym, d)
        sa["symbol"] = sym
        analyses.append(sa)
    analyses.sort(key=lambda a: (_REC_ORDER.get(a["recommendation"], 9), -a["sell_score"]))

    n    = len(analyses)
    rows = ""
    for i, sa in enumerate(analyses):
        sym    = sa["symbol"]
        score  = sa["sell_score"]
        rec    = sa["recommendation"]
        unreal = sa.get("unrealised_pct", 0.0)
        pw     = sa.get("position_weight", 0.0)
        ens    = sa.get("ensemble_score", 0.0)
        reasons_sell = sa.get("reasons_to_sell", [])
        reasons_hold = sa.get("reasons_to_hold", [])
        trim_pct = sa.get("trim_amount_pct", 0)

        td = TD if i < n - 1 else TD0
        bar_c = LOSS if score > 65 else (NEURAL if score > 35 else GAIN)
        bar_html = (
            f'<div style="display:inline-flex;align-items:center;gap:6px;">'
            f'<div style="background:{BORDER};border-radius:2px;height:4px;width:60px;">'
            f'<div style="background:{bar_c};height:100%;width:{score}%;border-radius:2px;"></div>'
            f'</div>'
            f'<span style="font-size:{FONT_LABEL};color:{bar_c};font-weight:600;">{score}</span>'
            f'</div>'
        )
        unreal_c  = GAIN if unreal >= 0 else LOSS
        unreal_str = f"{unreal:+.1f}%"
        trim_note  = f"Trim {trim_pct}%" if trim_pct > 0 else ""

        # Primary sell reason or hold reason
        primary_reason = reasons_sell[0] if reasons_sell else (reasons_hold[0] if reasons_hold else "No signal")

        rows += (
            f'<tr>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_action_badge(rec)}</td>'
            f'<td {td}>{bar_html}</td>'
            f'<td {td}><span style="color:{unreal_c};font-weight:700;">{unreal_str}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{pw:.0f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{primary_reason}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{NEURAL};">{trim_note}</span></td>'
            f'</tr>'
        )

    act_count = sum(1 for a in analyses if a["recommendation"] != "HOLD")
    note = f"{act_count} need attention · stop-loss 8%" if act_count else f"{n} positions — all holding"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Signal</th><th {TH}>Score</th>'
        f'<th {TH}>P&amp;L</th><th {TH}>Weight</th><th {TH}>Top Reason</th><th {TH}>Action</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("📉","Sell Analysis",note)}{table}</div>'


# ── PANEL 5: Position Sizing — called internally by render_decision_center ────
@safe_render("Position Sizing")
def render_position_sizing_panel() -> str:
    d        = get_data()
    open_pos = d.get("open_pos", {})

    if not open_pos:
        return f'<div class="nt nt-wrap">{_section("📐","Position Sizing","Conviction-based target allocations")}{_card(_empty_state("💰", "Fully in cash", "Sizing guidance runs once the bot holds positions."))}</div>'

    sizings = []
    for sym in open_pos:
        sz = get_position_sizing(sym, d)
        sz["symbol"] = sym
        sizings.append(sz)
    sizings.sort(key=lambda s: abs(s["delta_weight"]), reverse=True)

    n    = len(sizings)
    rows = ""
    for i, sz in enumerate(sizings):
        sym     = sz["symbol"]
        cur_w   = sz["current_weight"]
        tgt_w   = sz["target_weight"]
        delta_w = sz["delta_weight"]
        dol_disp = sz["dollar_display"]
        reason  = sz["reason"]
        action  = sz["action"]
        td = TD if i < n - 1 else TD0

        act_c = GAIN if action == "add" else (LOSS if action == "reduce" else TEXT2)
        delta_str = f"{delta_w:+.1f}%"
        delta_c   = GAIN if delta_w > 0 else (LOSS if delta_w < 0 else TEXT2)

        # Weight bar showing current vs target
        bar_max = max(tgt_w, cur_w, 5.0)
        cur_bar_w = int(cur_w / bar_max * 100)
        tgt_bar_w = int(tgt_w / bar_max * 100)
        bar_html = (
            f'<div style="position:relative;width:80px;height:6px;background:{BORDER};border-radius:3px;">'
            f'<div style="position:absolute;left:0;top:0;height:100%;width:{cur_bar_w}%;'
            f'background:{TEXT2};border-radius:3px;"></div>'
            f'<div style="position:absolute;left:0;top:0;height:100%;width:{tgt_bar_w}%;'
            f'background:{act_c};opacity:.4;border-radius:3px;"></div>'
            f'</div>'
        )

        rows += (
            f'<tr>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{TEXT1};">{cur_w:.1f}%</span></td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{act_c};">{tgt_w:.1f}%</span></td>'
            f'<td {td}>{bar_html}</td>'
            f'<td {td}><span style="font-weight:700;color:{delta_c};">{delta_str}</span></td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{act_c};">{dol_disp}</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT2};">{reason}</span></td>'
            f'</tr>'
        )

    note = f"{n} positions · conviction-weighted · max 25% single stock"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Current</th><th {TH}>Target</th>'
        f'<th {TH}>Bar</th><th {TH}>Delta</th><th {TH}>Amount</th><th {TH}>Reason</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("📐","Position Sizing",note)}{table}</div>'


# ── PANEL: Decision Center — what to do with each position ────────────────────
# NOTE: render_portfolio_actions, render_sell_analysis, render_position_sizing_panel
#       are consolidated here. They remain functional but are not wired to layout.
@timed(_logger)
@safe_render("Decision Center")
def render_decision_center() -> str:
    d        = get_data()
    open_pos = d.get("open_pos", {})

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("🎯","Decision Center","What to do with each position")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Bot enters trades during market hours when signals align."))}</div>')

    _ORDER = {"EXIT": 0, "SELL": 1, "TRIM": 2, "BUY": 3, "ADD": 4, "WATCH": 5, "HOLD": 6}
    rows_data: list[dict] = []
    for sym in open_pos:
        pa = get_portfolio_action(sym, d)
        sa = get_sell_analysis(sym, d)
        sz = get_position_sizing(sym, d)
        rows_data.append({
            "symbol":       sym,
            "action":       pa.get("action", "HOLD"),
            "sell_score":   sa.get("sell_score", 0),
            "cur_w":        sz.get("current_weight", 0.0),
            "tgt_w":        sz.get("target_weight", 0.0),
            "delta_w":      sz.get("delta_weight", 0.0),
            "dol_disp":     sz.get("dollar_display", "—"),
            "reasons_sell": sa.get("reasons_to_sell", []),
            "reasons_hold": sa.get("reasons_to_hold", []),
            "pa_reason":    pa.get("reason", ""),
        })
    rows_data.sort(key=lambda r: (_ORDER.get(r["action"], 9), -r["sell_score"]))

    n    = len(rows_data)
    rows = ""
    for i, r in enumerate(rows_data):
        sym        = r["symbol"]
        action     = r["action"]
        score      = r["sell_score"]
        cur_w      = r["cur_w"]
        tgt_w      = r["tgt_w"]
        delta_w    = r["delta_w"]
        dol_disp   = r["dol_disp"]
        reasons_s  = r["reasons_sell"]
        reasons_h  = r["reasons_hold"]
        pa_reason  = r["pa_reason"]
        td = TD if i < n - 1 else TD0

        bar_c = ACTION_SELL if score > 65 else (ACTION_TRIM if score > 35 else ACTION_BUY)
        score_html = (
            f'<div style="display:inline-flex;align-items:center;gap:5px;">'
            f'<div style="background:{BORDER};border-radius:2px;height:4px;width:40px;overflow:hidden;">'
            f'<div style="background:{bar_c};height:100%;width:{score}%;border-radius:2px;"></div></div>'
            f'<span style="font-size:{FONT_LABEL};color:{bar_c};font-weight:{WEIGHT_BOLD};">{score}</span>'
            f'</div>'
        )

        delta_c = ACTION_BUY if delta_w > 1 else (ACTION_SELL if delta_w < -1 else TEXT2)
        weight_html = (
            f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{cur_w:.1f}%</span>'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};margin:0 3px;">→</span>'
            f'<span style="font-size:{FONT_LABEL};color:{delta_c};font-weight:{WEIGHT_BOLD};">{tgt_w:.1f}%</span>'
        )

        reason_parts = []
        if reasons_s:
            reason_parts.append(
                f'<span style="color:{ACTION_SELL};">✗</span>'
                f'<span style="font-size:{FONT_LABEL};color:{TEXT2};"> {reasons_s[0]}</span>'
            )
        if reasons_h:
            reason_parts.append(
                f'<span style="color:{ACTION_BUY};">✓</span>'
                f'<span style="font-size:{FONT_LABEL};color:{TEXT2};"> {reasons_h[0]}</span>'
            )
        if not reason_parts and pa_reason:
            reason_parts.append(f'<span style="font-size:{FONT_LABEL};color:{TEXT2};">{pa_reason}</span>')
        reasons_html = '<br>'.join(reason_parts) if reason_parts else (
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};">No signal</span>'
        )

        rows += (
            f'<tr>'
            f'<td {td}>{_symbol(sym)}</td>'
            f'<td {td}>{_action_badge(action)}</td>'
            f'<td {td}>{score_html}</td>'
            f'<td {td}>{weight_html}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT1};'
            f'font-family:Courier New,monospace;">{dol_disp}</span></td>'
            f'<td {td}><div style="line-height:1.7;">{reasons_html}</div></td>'
            f'</tr>'
        )

    act_count = sum(1 for r in rows_data if r["action"] not in ("HOLD", "WATCH"))
    note = f"{act_count} need action · {n} positions" if act_count else f"{n} positions · holding"
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Action</th><th {TH}>Score</th>'
        f'<th {TH}>Weight</th><th {TH}>Amount</th><th {TH}>Reasons</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("🎯","Decision Center",note)}{table}</div>'


# ── PANEL: Rebalance — current vs target allocation ───────────────────────────
@timed(_logger)
@safe_render("Rebalance")
def render_rebalance() -> str:
    d        = get_data()
    open_pos = d.get("open_pos", {})

    if not open_pos:
        return (f'<div class="nt nt-wrap">'
                f'{_section("⚖","Rebalance","Current vs target allocation")}'
                f'{_card(_empty_state("💰", "Fully in cash", "Rebalance panel activates once the bot holds positions."))}</div>')

    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$","").replace(",","")) if d["portfolio"] != "—" else 0.0
    except Exception as exc:
        logger.debug(f"parse_portfolio_value render_rebalance: {exc}")

    prices = d.get("prices", {})
    invested_total = sum(
        pos["shares"] * prices.get(sym, pos["invested"] / max(pos["shares"], 1))
        for sym, pos in open_pos.items()
    )
    cash_pct = max(0.0, (_pv - invested_total) / _pv * 100) if _pv > 0 else 0.0

    sizings = []
    for sym in open_pos:
        sz = get_position_sizing(sym, d)
        sz["symbol"] = sym
        sizings.append(sz)
    sizings.sort(key=lambda s: -s.get("current_weight", 0.0))

    health = get_portfolio_health(d)

    n    = len(sizings)
    rows = ""
    for i, sz in enumerate(sizings):
        sym     = sz["symbol"]
        cur_w   = sz.get("current_weight", 0.0)
        tgt_w   = sz.get("target_weight", 0.0)
        delta_w = sz.get("delta_weight", 0.0)
        dol     = sz.get("dollar_display", "—")
        sz_act  = sz.get("action", "hold").lower()
        badge_a = "ADD" if sz_act == "add" else ("TRIM" if sz_act == "reduce" else "HOLD")
        td = TD if i < n - 1 else TD0

        delta_c   = ACTION_BUY if delta_w > 1 else (ACTION_SELL if delta_w < -1 else TEXT2)
        delta_str = f"{delta_w:+.1f}%"
        rows += (
            f'<tr>'
            f'<td {td}>{_symbol(sym)}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT1};'
            f'font-family:Courier New,monospace;">{cur_w:.1f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{delta_c};'
            f'font-weight:{WEIGHT_BOLD};">{tgt_w:.1f}%</span></td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{delta_c};'
            f'font-weight:{WEIGHT_BOLD};">{delta_str}</span></td>'
            f'<td {td}>{_action_badge(badge_a, "small")}</td>'
            f'<td {td}><span style="font-size:{FONT_LABEL};color:{TEXT1};'
            f'font-family:Courier New,monospace;">{dol}</span></td>'
            f'</tr>'
        )

    tgt_sum     = sum(sz.get("target_weight", 0.0) for sz in sizings)
    target_cash = max(0.0, 100.0 - tgt_sum)
    cash_delta  = target_cash - cash_pct
    cash_c      = ACTION_BUY if cash_delta > 1 else (ACTION_SELL if cash_delta < -1 else TEXT2)
    cash_badge  = "ADD" if cash_delta > 1 else ("TRIM" if cash_delta < -1 else "HOLD")
    rows += (
        f'<tr>'
        f'<td {TD0}><span style="font-family:Courier New,monospace;font-weight:{WEIGHT_BOLD};'
        f'color:{TEXT3};font-size:{FONT_VALUE};">CASH</span></td>'
        f'<td {TD0}><span style="font-size:{FONT_LABEL};color:{TEXT1};'
        f'font-family:Courier New,monospace;">{cash_pct:.1f}%</span></td>'
        f'<td {TD0}><span style="font-size:{FONT_LABEL};color:{cash_c};'
        f'font-weight:{WEIGHT_BOLD};">{target_cash:.1f}%</span></td>'
        f'<td {TD0}><span style="font-size:{FONT_LABEL};color:{cash_c};'
        f'font-weight:{WEIGHT_BOLD};">{cash_delta:+.1f}%</span></td>'
        f'<td {TD0}>{_action_badge(cash_badge, "small")}</td>'
        f'<td {TD0}>—</td>'
        f'</tr>'
    )

    net_rebalance = sum(abs(sz.get("delta_dollars", 0.0)) for sz in sizings) / 2
    net_str = f"${net_rebalance:,.0f}" if net_rebalance > 0 else "—"
    health_score = health.get("total", 0)
    grade        = health.get("grade", "—")

    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th><th {TH}>Current</th><th {TH}>Target</th>'
        f'<th {TH}>Delta</th><th {TH}>Action</th><th {TH}>Amount</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    summary = (
        f'<div style="display:flex;gap:0;flex-direction:column;">'
        + _metric_row("Net to rebalance", net_str, TEXT1)
        + _metric_row("Health score", f"{health_score}/100", ACTION_BUY if health_score >= 75 else (ACTION_TRIM if health_score >= 50 else ACTION_SELL), grade)
        + f'</div>'
    )
    note = f"{n} positions · ~{net_str} to rebalance"
    return (
        f'<div class="nt nt-wrap">'
        f'{_section("⚖","Rebalance",note)}'
        f'{table}'
        f'{_card(summary)}'
        f'</div>'
    )


# ── Gradio layout — 4-tab design ──────────────────────────────────────────────
# Gradio 5 removed every= from components. Use gr.Timer + .tick() instead.
_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#0f1115",
    body_text_color="#ffffff",
    block_background_fill="#171a21",
    block_border_color="#2d3445",
    block_radius="12px",
    button_primary_background_fill="#00c853",
    button_primary_text_color="#000000",
    button_secondary_background_fill="#222733",
    button_secondary_text_color="#ffffff",
    border_color_primary="#2d3445",
)

with gr.Blocks(title="TradeGenius AI", theme=_theme, css=GRADIO_CSS) as demo:
    gr.HTML(HEADER_HTML)
    gr.HTML("""
    <script>
    function enforceTabStyles() {
      const buttons = document.querySelectorAll(
        '.tab-nav button, .tabs button[role="tab"], button[id*="tab"]'
      );
      buttons.forEach(btn => {
        btn.style.setProperty('color', '#ffffff', 'important');
        const isSelected = btn.classList.contains('selected');
        btn.style.setProperty('opacity', isSelected ? '1' : '0.6', 'important');
        if (!btn._tgListeners) {
          btn._tgListeners = true;
          btn.addEventListener('mouseenter', () => {
            btn.style.setProperty('opacity', '1', 'important');
          });
          btn.addEventListener('mouseleave', () => {
            if (!btn.classList.contains('selected')) {
              btn.style.setProperty('opacity', '0.6', 'important');
            }
          });
        }
      });
    }
    const observer = new MutationObserver(enforceTabStyles);
    observer.observe(document.body, {
      subtree: true, attributes: true, attributeFilter: ['class']
    });
    setTimeout(enforceTabStyles, 300);
    setTimeout(enforceTabStyles, 800);
    setTimeout(enforceTabStyles, 2000);
    </script>
    """)

    with gr.Tabs():
        with gr.TabItem("📊 Dashboard"):
            # Exactly 5 panels — open dashboard and within 3s: health, actions, risk
            hero_out           = gr.HTML(value=render_portfolio_health_hero)
            todays_actions_out = gr.HTML(value=render_todays_actions)
            ai_rec_out         = gr.HTML(value=render_ai_recommendation)
            risk_panel_out     = gr.HTML(value=render_risk_panel)
            whats_changed_out  = gr.HTML(value=render_whats_changed)
            # ── Symbol drilldown ──────────────────────────────────────────────
            symbol_selector = gr.Dropdown(
                choices=_get_symbol_choices(),
                label="🔍 Symbol Detail — select a ticker to drill down",
                value=None, container=True,
            )
            symbol_detail_out = gr.HTML(value="")

        with gr.TabItem("⚡ Signals"):
            timeline_out  = gr.HTML(value=render_timeline)
            signals_out   = gr.HTML(value=render_signals_tab)
            with gr.Row():
                with gr.Column(scale=55):
                    mkt_intel_out = gr.HTML(value=render_market_intelligence)
                with gr.Column(scale=45):
                    watchlist_out = gr.HTML(value=render_watchlist)

        with gr.TabItem("💼 Portfolio"):
            perf_tabs   = gr.Radio(
                choices=_perf_choices(),
                value=_perf_choices()[2],   # default: 1M
                label="", container=False,
                elem_classes=["perf-tabs"],
            )
            perf_out    = gr.HTML(value=render_portfolio_performance)
            with gr.Row():
                with gr.Column(scale=65):
                    eq_plot    = gr.Plot(value=render_equity_chart, label="")
                with gr.Column(scale=35):
                    alloc_plot = gr.Plot(value=render_allocation_chart, label="")
            pnl_plot          = gr.Plot(value=render_pnl_chart, label="")
            committee_out     = gr.HTML(value=render_ai_committee)
            decision_center_out = gr.HTML(value=render_decision_center)
            rebalance_out     = gr.HTML(value=render_rebalance)
            pos_out           = gr.HTML(value=render_positions)
            trades_out        = gr.HTML(value=render_trades)

        with gr.TabItem("🔬 Models"):
            model_view = gr.Radio(
                choices=["📊 Investor View", "🔬 Developer View"],
                value="📊 Investor View",
                label="", container=False,
            )
            investor_out = gr.HTML(value=render_investor_view, visible=True)
            with gr.Column(visible=False) as dev_col:
                metrics_out = gr.HTML(value=render_institutional_metrics)
                with gr.Row():
                    with gr.Column(scale=65):
                        fi_plot = gr.Plot(value=render_feature_importance_chart, label="")
                    with gr.Column(scale=35):
                        val_out = gr.HTML(value=render_validation_report)

    gr.HTML(value=FOOTER_HTML)

    # Models tab toggle
    model_view.change(
        fn=lambda v: (gr.update(visible=(v == "📊 Investor View")),
                      gr.update(visible=(v == "🔬 Developer View"))),
        inputs=[model_view],
        outputs=[investor_out, dev_col],
    )

    # Portfolio performance period selection
    perf_tabs.change(
        fn=render_portfolio_performance,
        inputs=[perf_tabs],
        outputs=[perf_out],
    )

    # Symbol drilldown
    symbol_selector.change(
        fn=render_symbol_detail,
        inputs=[symbol_selector],
        outputs=[symbol_detail_out],
    )

    # One shared timer — cache layer ensures a single DB+API refresh per tick
    timer = gr.Timer(value=60)
    # Dashboard (5 panels)
    timer.tick(fn=render_portfolio_health_hero, outputs=hero_out)
    timer.tick(fn=render_todays_actions,        outputs=todays_actions_out)
    timer.tick(fn=render_ai_recommendation,     outputs=ai_rec_out)
    timer.tick(fn=render_risk_panel,            outputs=risk_panel_out)
    timer.tick(fn=render_whats_changed,         outputs=whats_changed_out)
    timer.tick(fn=lambda: gr.update(choices=_get_symbol_choices()), outputs=symbol_selector)
    # Signals tab
    timer.tick(fn=render_timeline,              outputs=timeline_out)
    timer.tick(fn=render_signals_tab,           outputs=signals_out)
    timer.tick(fn=render_market_intelligence,   outputs=mkt_intel_out)
    timer.tick(fn=render_watchlist,             outputs=watchlist_out)
    # Portfolio tab
    timer.tick(fn=lambda: gr.update(choices=_perf_choices()), outputs=perf_tabs)
    timer.tick(fn=render_portfolio_performance, outputs=perf_out)
    timer.tick(fn=render_equity_chart,          outputs=eq_plot)
    timer.tick(fn=render_allocation_chart,      outputs=alloc_plot)
    timer.tick(fn=render_pnl_chart,             outputs=pnl_plot)
    timer.tick(fn=render_ai_committee,          outputs=committee_out)
    timer.tick(fn=render_decision_center,       outputs=decision_center_out)
    timer.tick(fn=render_rebalance,             outputs=rebalance_out)
    timer.tick(fn=render_positions,             outputs=pos_out)
    timer.tick(fn=render_trades,                outputs=trades_out)
    # Models tab
    timer.tick(fn=render_investor_view,            outputs=investor_out)
    timer.tick(fn=render_institutional_metrics,    outputs=metrics_out)
    timer.tick(fn=render_feature_importance_chart, outputs=fi_plot)
    timer.tick(fn=render_validation_report,        outputs=val_out)

if __name__ == "__main__":
    demo.launch()
