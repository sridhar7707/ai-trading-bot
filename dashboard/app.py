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

DB_PATH    = "trades.db"
HF_TOKEN   = os.getenv("HF_TOKEN",   "")
HF_REPO_ID = os.getenv("HF_DB_REPO_ID", os.getenv("HF_REPO_ID", "ksri77/ai-trading-bot-db"))

# ── Design tokens — Robinhood-inspired palette ────────────────────────────────
BG        = "#0e0e0e"   # true near-black (Robinhood app background)
SURFACE   = "#1b1b1b"   # elevated surface
SURFACE2  = "#252525"   # hover state
BORDER    = "#2a2a2a"   # very subtle separator
PRIMARY   = "#00c805"   # Robinhood signature green
GAIN      = "#00c805"   # profit green
LOSS      = "#ff5000"   # Robinhood loss red-orange
NEURAL    = "#9d4edd"   # regime purple (kept)
TEXT1     = "#ffffff"
TEXT2     = "#a0a0a0"
PRIMARY_BG = "#001602"
GAIN_BG    = "#001602"
LOSS_BG    = "#1e0800"
NEURAL_BG  = "#14003a"
GAIN_BD    = "#00a005"
LOSS_BD    = "#cc3d00"
NEURAL_BD  = "#7b2fc9"

# Plotly shared theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT2, family="Inter, monospace", size=11),
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

/* ── Tab navigation ───────────────────────────────────────────────────────── */
.tabs > .tab-nav {{
  background: {SURFACE} !important;
  border-bottom: 1px solid {BORDER} !important;
  padding: 0 12px !important;
}}
.tabs > .tab-nav > button {{
  color: {TEXT2} !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 10px 18px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  transition: color .15s, border-color .15s !important;
}}
.tabs > .tab-nav > button:hover {{
  color: {TEXT1} !important;
  border-bottom-color: {BORDER} !important;
}}
.tabs > .tab-nav > button.selected {{
  color: {PRIMARY} !important;
  border-bottom-color: {PRIMARY} !important;
  background: transparent !important;
}}
.tabitem {{ background: transparent !important; border: none !important; }}

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
            logger.warning(f"DB sync: {e}")
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
        except Exception:
            pass  # charts show placeholder until model has been retrained + pushed


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
            logger.warning(f"Extended trades query failed (missing columns?): {_e} — falling back to base schema")
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
        logger.warning(f"DB read: {e}")
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return "—", TEXT2


# ── HTML builders ─────────────────────────────────────────────────────────────
def _pnl_color(v: str) -> str:
    return GAIN if v.startswith("+") else (LOSS if v.startswith("-") else TEXT2)

def _sym(s: str) -> str:
    return (f'<span style="display:inline-block;background:{PRIMARY_BG} !important;'
            f'border:1px solid {BORDER};border-left:3px solid {PRIMARY};border-radius:4px;'
            f'padding:3px 9px;font-family:Courier New,monospace;font-weight:700;'
            f'font-size:13px;color:{PRIMARY} !important;letter-spacing:.5px;">{s}</span>')

def _num(v: str, bold=False) -> str:
    w = "800" if bold else "600"
    return (f'<span style="font-family:Courier New,monospace;font-weight:{w};'
            f'font-size:14px;color:{TEXT1} !important;">{v}</span>')

def _pnl(v: str, big=False) -> str:
    c  = _pnl_color(v)
    sz = "15px" if big else "13px"; fw = "700"
    return (f'<span style="font-family:-apple-system,monospace;font-weight:{fw};'
            f'font-size:{sz};color:{c} !important;">{v}</span>')

def _badge(action: str) -> str:
    if action == "BUY":
        bg, fg, bd = GAIN_BG, GAIN, GAIN_BD
    elif action.startswith("SELL"):
        bg, fg, bd = LOSS_BG, LOSS, LOSS_BD
    else:
        bg, fg, bd = NEURAL_BG, NEURAL, NEURAL_BD
    return (f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'border:1px solid {bd};padding:2px 10px;border-radius:4px;font-size:11px;'
            f'font-weight:700;letter-spacing:.3px;font-family:-apple-system,monospace;">{action}</span>')

def _section(icon: str, title: str, note: str = "") -> str:
    note_html = (f'<span style="font-size:10px;color:{TEXT2} !important;'
                 f'font-weight:400;letter-spacing:0;margin-left:6px;">{note}</span>'
                 if note else "")
    return (f'<div class="nt-sec" style="animation:fadeInUp .4s ease both;">'
            f'<span style="font-size:13px;">{icon}</span>'
            f'<span style="color:{PRIMARY} !important;font-size:11px;font-weight:700;">'
            f'{title}</span>{note_html}'
            f'<span class="nt-sec-line"></span></div>')

def _wrap(inner: str) -> str:
    return (f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;overflow:hidden;">'
            f'{inner}</div>')

def _card(label: str, value: str, accent: str = PRIMARY,
          color: str = TEXT1, sub: str = "", delay: float = 0) -> str:
    sub_html = (f'<div style="font-size:10px;color:{TEXT2};margin-top:2px;">{sub}</div>'
                if sub else "")
    return (
        f'<div class="nt-card" style="animation-delay:{delay:.2f}s;">'
        f'<div style="font-size:11px;color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;font-weight:600;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:22px;font-weight:700;letter-spacing:-0.3px;'
        f'color:{color};line-height:1;">{value}</div>'
        f'{sub_html}</div>'
    )

TH  = (f'style="background:{BG};color:{TEXT2};font-size:10px;font-weight:600;'
       f'text-transform:uppercase;letter-spacing:.8px;padding:10px 16px;'
       f'border-bottom:1px solid {BORDER};text-align:left;white-space:nowrap;"')
TD  = (f'style="padding:12px 16px;border-bottom:1px solid {BORDER};'
       f'vertical-align:middle;background:{SURFACE};color:{TEXT1};"')
TD0 = (f'style="padding:12px 16px;vertical-align:middle;'
       f'background:{SURFACE};color:{TEXT1};"')

# ── Render: metrics ───────────────────────────────────────────────────────────
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
        f'<div style="font-size:11px;color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:6px;">Alpaca Paper Account Balance</div>'
        f'<div class="nt-hero-val">{portfolio_val}</div>'
        f'<div class="nt-hero-chg" style="color:{pnl_color};">{hero_chg}</div>'
        f'<div style="font-size:10px;color:{TEXT2};margin-top:4px;">'
        f'Unrealized gain / loss on open positions vs. what the bot paid</div>'
        f'</div>'
    )

    status = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2};font-size:11px;">'
        f'Updated &nbsp;<strong style="color:{TEXT1};">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:11px;">'
        f'{mkt_label}</span></span>'
        f'<div style="height:2px;width:100px;background:{BORDER};border-radius:1px;">'
        f'<div style="height:100%;width:100%;background:{PRIMARY};border-radius:1px;'
        f'animation:countdown 60s linear forwards;"></div></div>'
        f'<span style="color:{TEXT2};font-size:11px;">60s refresh</span>'
        f'</div>'
    )

    legend = (
        f'<div style="display:flex;gap:18px;padding:4px 2px 8px;font-size:10px;color:{TEXT2};">'
        f'<span><span style="color:{GAIN};">●</span> Gain / Bull regime</span>'
        f'<span><span style="color:{LOSS};">●</span> Loss / Bear regime</span>'
        f'<span><span style="color:{NEURAL};">●</span> Neutral / Ranging</span>'
        f'<span style="margin-left:auto;font-style:italic;">Paper money — no real funds at risk</span>'
        f'</div>'
    )

    row1 = (
        f'<div class="nt-cards">'
        + _card("Unrealized P&amp;L",  pnl_str,                pnl_color, pnl_color,
                "Open trade gain/loss vs. cost basis",         0.00)
        + _card("Total Invested",      invested_str,            TEXT2,     TEXT1,
                "Capital currently deployed in open trades",   0.06)
        + _card("Market Regime",       d["regime_raw"].title(), TEXT2,     r_color,
                "AI-detected trend — drives position sizing",  0.12)
        + _card("Market Session",      mkt_label,               TEXT2,     mkt_color,
                "NYSE/NASDAQ open 9:30am–4pm ET, Mon–Fri",    0.18)
        + f'</div>'
    )

    row2 = (
        f'<div class="nt-cards">'
        + _card("Open Positions", str(open_count),
                TEXT2, TEXT1,
                f"Unique stocks held now (max 8 allowed)", 0.24)
        + _card("Win Rate",       wr_str,
                TEXT2, wr_color,
                f"% of closed trades that made money · {win_count}/{sell_count}", 0.30)
        + _card("Total Trades",   str(d["total_trades"]),
                TEXT2, TEXT1, "All BUY + SELL orders since launch", 0.36)
        + _card("Buys / Sells",   f"{d['buy_count']} / {d['sell_count']}",
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


def _sym_perf(hist, buy_date: str | None) -> dict:
    """Compute pct returns (1D/1W/1M/1Y/All) from a yfinance history DataFrame."""
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {}
    today = datetime.date.today()
    try:
        dates = [d.date() if hasattr(d, "date") else d for d in hist.index]
    except Exception:
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
        except Exception:
            result["All"] = None
    else:
        result["All"] = None
    return result


# ── Render: positions table ───────────────────────────────────────────────────
def render_positions() -> str:
    d         = get_data()
    open_syms = d["open_pos"]
    prices    = d["prices"]

    if not open_syms:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:40px;font-size:13px;">'
                 f'No open positions right now. The bot will enter trades during market hours '
                 f'(9:30am–4pm ET, Mon–Fri) when its signals align.</div>')
        return (f'<div class="nt nt-wrap">'
                f'{_section("📊","Open Positions","mark-to-market · price updated every 60s")}'
                f'{_wrap(empty)}</div>')

    _AMBER = "#f59e0b"

    # Portfolio value as float (needed for position-size %)
    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception:
        pass

    # Batch-fetch latest ensemble_score + earliest BUY date per open symbol
    _ens: dict[str, float] = {}
    _buy_dates: dict[str, str] = {}
    if os.path.exists(DB_PATH):
        try:
            sym_list = list(open_syms.keys())
            ph  = ",".join("?" * len(sym_list))
            con = sqlite3.connect(DB_PATH)
            for _r in con.execute(
                f"SELECT t.symbol, COALESCE(t.ensemble_score, 0.0) "
                f"FROM trades t "
                f"INNER JOIN (SELECT symbol, MAX(id) AS mid FROM trades "
                f"            WHERE symbol IN ({ph}) GROUP BY symbol) m "
                f"ON t.id = m.mid",
                sym_list,
            ).fetchall():
                _ens[_r[0]] = float(_r[1])
            for _r in con.execute(
                f"SELECT symbol, MIN(timestamp) FROM trades "
                f"WHERE symbol IN ({ph}) AND action = 'BUY' GROUP BY symbol",
                sym_list,
            ).fetchall():
                _buy_dates[_r[0]] = str(_r[1])
            con.close()
        except Exception as _exc:
            logger.warning(f"render_positions ens/buy_date fetch: {_exc}")

    # Pre-fetch yfinance history and compute perf dicts for all open symbols
    _perfs: dict[str, dict] = {}
    for _s in list(open_syms.keys()):
        _perfs[_s] = _sym_perf(_get_sym_hist(_s), _buy_dates.get(_s))

    def _ai_action(sym: str, pnl_pct: float, pos_pct: float):
        ens = _ens.get(sym, 1.0)  # default confident if not yet in DB

        # ── 4-component scoring ────────────────────────────────────────────────
        size_pts   = 30 if pos_pct > 25 else (20 if pos_pct > 15 else (10 if pos_pct > 10 else 0))
        profit_pts = (30 if pnl_pct > 50 else (20 if pnl_pct > 25 else (10 if pnl_pct > 10 else 0))
                      ) if pnl_pct > 0 else 0
        conf_pts   = 25 if ens < 0.55 else (15 if ens < 0.65 else 0)
        dd_pts     = (15 if pnl_pct < -8 else (10 if pnl_pct < -5 else 0)) if pnl_pct < 0 else 0
        total      = size_pts + profit_pts + conf_pts + dd_pts

        # ── Top reasons ────────────────────────────────────────────────────────
        scored: list[tuple[int, str]] = []
        if size_pts:   scored.append((size_pts,   "Position oversized"))
        if profit_pts:
            lbl = ("Profit > 50%" if pnl_pct > 50 else
                   "Profit > 25%" if pnl_pct > 25 else "Profit > 10%")
            scored.append((profit_pts, lbl))
        if conf_pts:   scored.append((conf_pts,   "AI confidence weakening"))
        if dd_pts:     scored.append((dd_pts,     "Drawdown risk"))
        scored.sort(key=lambda x: -x[0])
        reason = " · ".join(r for _, r in scored[:2]) if scored else "All metrics healthy"

        # ── Badge ──────────────────────────────────────────────────────────────
        if total <= 30:   label, bc, bbg = "HOLD",  GAIN,   "#0a2010"
        elif total <= 59: label, bc, bbg = "WATCH", NEURAL, "#1a1030"
        elif total <= 79: label, bc, bbg = "TRIM",  _AMBER, "#2a1f08"
        else:             label, bc, bbg = "EXIT",  LOSS,   "#2a0a0a"

        return total, label, bc, bbg, reason

    _td_perf = (f'style="padding:12px 10px 4px;vertical-align:middle;'
                f'background:{SURFACE};text-align:right;"')

    def _perf_cell(val) -> str:
        if val is None:
            return f'<td {_td_perf}><span style="color:{TEXT2} !important;">—</span></td>'
        clr  = GAIN if val >= 0 else LOSS
        sign = "+" if val >= 0 else ""
        return (f'<td {_td_perf}>'
                f'<span style="color:{clr} !important;font-weight:700;">'
                f'{sign}{val:.1f}%</span></td>')

    rows  = ""
    items = list(open_syms.items())
    last  = len(items) - 1

    for i, (sym, v) in enumerate(items):
        cur_price = prices.get(sym, 0.0)
        cur_val   = v["shares"] * cur_price
        invested  = v["invested"]
        pnl       = cur_val - invested
        pnl_pct   = (pnl / invested * 100) if invested > 0 else 0.0
        pos_pct   = (cur_val / _pv * 100)   if _pv > 0      else 0.0
        cv_str    = f"${cur_val:.2f}"   if cur_price else "—"
        p_str     = f"${pnl:+.2f}"     if cur_price else "—"
        pct_str   = f"{pnl_pct:+.2f}%" if cur_price else "—"

        score, label, bc, bbg, reason = _ai_action(sym, pnl_pct, pos_pct)

        badge = (
            f'<span style="display:inline-block;background:{bbg};border:1px solid {bc};'
            f'color:{bc};font-size:10px;font-weight:700;letter-spacing:.5px;'
            f'padding:2px 8px;border-radius:4px;white-space:nowrap;">'
            f'{label} ({score})</span>'
        )
        anim    = f'style="animation:slideInRow .35s ease both;animation-delay:{i*0.07:.2f}s;"'
        td_main = (f'style="padding:12px 16px 4px;vertical-align:middle;'
                   f'background:{SURFACE};color:{TEXT1};"')
        sub_sep = f'border-bottom:1px solid {BORDER};' if i < last else ''
        td_sub  = (f'style="padding:2px 16px 10px;background:{SURFACE};'
                   f'{sub_sep}color:{TEXT2};font-size:11px;"')

        pf         = _perfs.get(sym, {})
        perf_cells = "".join([
            _perf_cell(pf.get("1D")),
            _perf_cell(pf.get("1W")),
            _perf_cell(pf.get("1M")),
            _perf_cell(pf.get("1Y")),
            _perf_cell(pf.get("All")),
        ])

        rows += (
            f'<tr {anim}>'
            f'<td {td_main}>{_sym(sym)}</td>'
            f'<td {td_main}>{_num(str(round(v["shares"], 4)))}</td>'
            f'<td {td_main}>{_num(f"${invested:.2f}", bold=True)}</td>'
            f'<td {td_main}>{_num(cv_str, bold=True)}</td>'
            f'<td {td_main}>{_pnl(p_str)}</td>'
            f'<td {td_main}>{_pnl(pct_str, big=True)}</td>'
            f'{perf_cells}'
            f'<td {td_main}>{badge}</td>'
            f'</tr>'
            f'<tr><td colspan="12" {td_sub}>{reason}</td></tr>'
        )

    _th_perf = f'style="padding:10px 10px;text-align:right;font-size:10px;font-weight:700;letter-spacing:.5px;color:{TEXT2};text-transform:uppercase;white-space:nowrap;"'
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th>'
        f'<th {TH}>Shares  <span style="font-weight:400;text-transform:none;letter-spacing:0;">held</span></th>'
        f'<th {TH}>Invested  <span style="font-weight:400;text-transform:none;letter-spacing:0;">cost basis</span></th>'
        f'<th {TH}>Current Value  <span style="font-weight:400;text-transform:none;letter-spacing:0;">live price</span></th>'
        f'<th {TH}>P&amp;L $  <span style="font-weight:400;text-transform:none;letter-spacing:0;">unrealised</span></th>'
        f'<th {TH}>P&amp;L %</th>'
        f'<th {_th_perf}>1D</th>'
        f'<th {_th_perf}>1W</th>'
        f'<th {_th_perf}>1M</th>'
        f'<th {_th_perf}>1Y</th>'
        f'<th {_th_perf}>All Time</th>'
        f'<th {TH}>AI Action</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("📊","Open Positions","mark-to-market · price updated every 60s")}'
            f'{table}</div>')


# ── Render: trades table ──────────────────────────────────────────────────────
def render_trades() -> str:
    d             = get_data()
    raw           = d["recent_trades"]
    total_trades  = d["total_trades"]

    if not raw:
        empty = (f'<div style="color:{TEXT2} !important;text-align:center;'
                 f'padding:40px;font-size:14px;">No trades yet.</div>')
        return f'<div class="nt nt-wrap">{_section("⚡","Recent Trades")}{_wrap(empty)}</div>'

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
            f'<td {td}><span style="font-family:Courier New,monospace;font-size:12px;'
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
        f'font-size:10px;color:{TEXT2};border-bottom:1px solid {BORDER};">'
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
                text="Which signals drive the AI's BUY decisions  <span style='font-size:11px;'>— longer bar = more influence on each trade</span>",
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
def render_validation_report() -> str:
    import json as _json
    vr_path = "models/validation_report.json"
    if not os.path.exists(vr_path):
        msg = (f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:12px;">'
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
            f'color:{TEXT2};font-size:11px;font-weight:600;">{label}</td>'
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
        f'padding:10px 14px;font-size:10px;color:{TEXT2};line-height:1.6;">'
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
    note = (f'<div style="font-size:10px;color:{TEXT2};padding:2px 0 6px;">'
            f'AUC ≥ 0.60 = good · ≥ 0.55 = acceptable · &lt; 0.52 = near-random</div>')
    return f'<div class="nt nt-wrap">{_section("🔬", "Model Validation")}{note}{table}</div>'


# ── Render: dashboard hero (Bloomberg-style 4-pack + status bar) ─────────────
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
            f'<div style="font-size:11px;color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:10px;">{label}</div>'
            f'<div style="font-size:34px;font-weight:700;letter-spacing:-1.5px;'
            f'color:{color};line-height:1;">{value}</div>'
            f'<div style="font-size:11px;color:{TEXT2};margin-top:6px;">{sub}</div>'
            f'</div>'
        )

    # ── Portfolio Health Score ──────────────────────────────────────────────
    pv_float = 0.0
    try:
        pv_float = float(d["portfolio"].replace("$", "").replace(",", "")) if d["portfolio"] != "—" else 0.0
    except Exception:
        pass

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
        f'<div style="font-size:11px;color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:10px;">Portfolio Health</div>'
        f'<div style="font-size:34px;font-weight:700;letter-spacing:-1.5px;'
        f'color:{health_c};line-height:1;">{health}<span style="font-size:16px;'
        f'color:{TEXT2};font-weight:400;">/100</span></div>'
        f'<div style="margin:8px 0 6px;background:{BORDER};border-radius:3px;height:4px;">'
        f'<div style="background:{health_c};height:100%;width:{health}%;'
        f'border-radius:3px;transition:width .4s;"></div></div>'
        f'<div style="font-size:11px;color:{TEXT2};">{weak_sub}</div>'
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
        f'<span style="color:{TEXT2};font-size:11px;">'
        f'Updated &nbsp;<strong style="color:{TEXT1};">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color};font-weight:600;font-size:11px;">'
        f'{mkt_label}</span></span>'
        f'<span style="color:{TEXT2};font-size:11px;">60s refresh &nbsp;·&nbsp; Paper money only</span>'
        f'</div>'
    )
    return f'<div class="nt nt-wrap">{cards}{status}</div>'


# ── Render: AI recommendation card — full-width hero ─────────────────────────
def render_ai_recommendation() -> str:
    d   = get_data()
    lb  = d.get("latest_buy_signal", {})
    vix = d.get("vix", 0.0)

    if not lb or not lb.get("symbol"):
        empty = (
            f'<div style="text-align:center;padding:48px 24px;">'
            f'<div style="font-size:36px;margin-bottom:12px;">🤖</div>'
            f'<div style="font-size:18px;font-weight:700;color:{TEXT1};margin-bottom:8px;">'
            f'No active signal</div>'
            f'<div style="font-size:13px;color:{TEXT2};line-height:1.8;">'
            f'The AI monitors markets Mon–Fri 9:30am–4pm ET.<br>'
            f'When a trade meets all entry gates, the full recommendation with reasoning '
            f'will appear here.</div></div>'
        )
        return (f'<div class="nt nt-wrap">'
                f'{_section("🤖","AI Recommendation","live signal · updated every 60s")}'
                f'{_wrap(empty)}</div>')

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
        f'<span style="font-size:11px;color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;">AI Confidence</span>'
        f'<span style="font-size:28px;font-weight:700;color:{conf_c};letter-spacing:-1px;">{conf_pct}</span>'
        f'</div>'
        f'<div style="background:{BORDER};border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="background:{conf_c};height:100%;width:{conf_w}%;border-radius:4px;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:8px;">'
        f'<span style="font-size:11px;color:{TEXT2};">Ensemble: '
        f'<span style="color:{agree_c};font-weight:700;">{agree_count}/5 models agree</span></span>'
        f'<span style="font-size:11px;color:{TEXT2};">Entry: '
        f'<span style="color:{TEXT1};font-weight:700;">${entry:.2f}</span></span>'
        f'</div></div>'
    )

    def _mini_bar(label, v, color):
        w = int(v * 100)
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<span style="font-size:11px;color:{TEXT2};width:68px;flex-shrink:0;">{label}</span>'
            f'<div style="background:{BORDER};border-radius:2px;height:4px;flex:1;overflow:hidden;">'
            f'<div style="background:{color};height:100%;width:{w}%;"></div></div>'
            f'<span style="font-size:11px;color:{color};width:34px;text-align:right;">{v*100:.0f}%</span>'
            f'</div>'
        )

    sub_scores = ""
    if xgb_p > 0 or lstm_p > 0:
        xc = GAIN if xgb_p >= 0.70 else (NEURAL if xgb_p >= 0.55 else TEXT2)
        lc = GAIN if lstm_p >= 0.70 else (NEURAL if lstm_p >= 0.55 else TEXT2)
        sc = GAIN if sent > 0.05 else (LOSS if sent < -0.05 else TEXT2)
        sub_scores = (
            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid {BORDER};">'
            f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:6px;">Model breakdown</div>'
            + _mini_bar("XGBoost", xgb_p, xc)
            + _mini_bar("LSTM", lstm_p, lc)
            + f'<div style="display:flex;gap:8px;margin:4px 0;">'
            f'<span style="font-size:11px;color:{TEXT2};width:68px;flex-shrink:0;">Sentiment</span>'
            f'<span style="font-size:11px;color:{sc};">'
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
    except Exception:
        pass

    if any(x in r_lower for x in ["bull"]) and not any("regime" in p[0].lower() for p in pos_items):
        pos_items.append(("Bull market regime", 15.0))

    why_html = ""
    if pos_items:
        why_html += (
            f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-bottom:8px;">Contributors</div>'
        )
        for name, pct in pos_items:
            bar_w = min(int(pct), 100)
            why_html += (
                f'<div style="display:flex;align-items:center;gap:6px;margin:5px 0;">'
                f'<span style="font-size:14px;color:{GAIN};width:14px;flex-shrink:0;'
                f'font-weight:700;line-height:1;">+</span>'
                f'<span style="font-size:12px;color:{TEXT1};flex:1;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;">{name}</span>'
                f'<div style="background:{BORDER};border-radius:2px;height:4px;'
                f'width:56px;overflow:hidden;flex-shrink:0;">'
                f'<div style="background:{GAIN};height:100%;width:{bar_w}%;"></div></div>'
                f'<span style="font-size:11px;color:{GAIN};width:36px;text-align:right;'
                f'flex-shrink:0;">+{pct:.0f}%</span>'
                f'</div>'
            )
    else:
        why_html += (
            f'<div style="color:{TEXT2};font-size:12px;line-height:1.6;">'
            f'Signal fired after all risk gates passed.<br>'
            f'<span style="font-size:11px;">SHAP % breakdown available after next model retrain.</span>'
            f'</div>'
        )

    if neg_items:
        why_html += (
            f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;'
            f'letter-spacing:.8px;margin-top:12px;margin-bottom:8px;">Risk Factors</div>'
        )
        for name in neg_items:
            why_html += (
                f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0;">'
                f'<span style="font-size:14px;color:{LOSS};width:14px;flex-shrink:0;'
                f'font-weight:700;line-height:1;">−</span>'
                f'<span style="font-size:12px;color:{TEXT2};">{name}</span>'
                f'</div>'
            )

    risk_badge = (
        f'<span style="background:{SURFACE2};border:1px solid {risk_color};'
        f'color:{risk_color};padding:3px 10px;border-radius:4px;font-size:11px;'
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
        f'<span style="font-family:Courier New,monospace;font-size:36px;font-weight:700;'
        f'color:{PRIMARY};letter-spacing:-2px;line-height:1;">{sym}</span>'
        f'{risk_badge}'
        f'</div>'
        f'<div style="font-size:13px;color:{TEXT2};margin-bottom:10px;">'
        f'Entry Price: <strong style="color:{TEXT1};">${entry:.2f}</strong>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'Regime: <strong style="color:{r_color};">{regime}</strong>'
        f'</div>'
        f'{conf_bar}'
        f'{sub_scores}'
        f'</div>'
        # Right: Why section
        f'<div class="nt-ai-right">'
        f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;'
        f'letter-spacing:.8px;margin-bottom:4px;">Why the AI is buying</div>'
        f'{why_html}'
        f'</div>'
        f'</div></div>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("🤖","AI Recommendation",_to_ct(ts))}'
            f'<div style="padding-top:4px;">{card}</div></div>')


# ── Render: market intelligence (VIX / regime / confidence / sentiment) ───────
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
        + _card("VIX", f"{vix:.1f}" if vix > 0 else "—",
                TEXT2, vix_color, f"{vix_label} · <15=calm, >30=fear", 0.00)
        + _card("Market Regime", regime.replace("_", " ").title(),
                TEXT2, r_color, "AI-detected trend · drives position size", 0.06)
        + _card("Signal Strength", f"{avg_conf*100:.0f}%" if avg_conf > 0 else "—",
                TEXT2, conf_color, "Avg confidence · last 5 buy signals", 0.12)
        + _card("News Sentiment", sent_label,
                TEXT2, sent_color, "FinBERT score · recent headlines", 0.18)
        + f'</div>'
    )
    return f'<div class="nt nt-wrap">{_section("📡","Market Intelligence","live")}{cards}</div>'


# ── Render: watchlist (open positions with live return vs avg cost) ────────────
def render_watchlist() -> str:
    d        = get_data()
    open_pos = d["open_pos"]
    prices   = d["prices"]

    if not open_pos:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:24px;font-size:12px;">'
                 f'No open positions — bot is in cash</div>')
        return (f'<div class="nt nt-wrap">'
                f'{_section("👁","Watchlist","open positions · vs avg cost")}'
                f'{_wrap(empty)}</div>')

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
            f'<td {td}><span style="font-weight:700;font-size:14px;color:{chg_c};">'
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
def render_signals_tab() -> str:
    d    = get_data()
    buys = d.get("today_buy_signals", [])

    if not buys:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:40px;font-size:13px;">'
                 f'No buy signals yet.<br>The AI generates signals during market hours '
                 f'(9:30am–4pm ET, Mon–Fri) when all entry gates pass.</div>')
        return (f'<div class="nt nt-wrap">'
                f'{_section("⚡","AI Buy Signals","recent")}{_wrap(empty)}</div>')

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
        except Exception:
            pass
        conf_pct = f"{conf*100:.0f}%" if conf > 0 else "—"
        conf_c   = GAIN if conf >= 0.75 else (NEURAL if conf >= 0.60 else TEXT2)
        td   = TD if i < len(shown) - 1 else TD0
        anim = f'style="animation:slideInRow .3s ease both;animation-delay:{i*0.04:.2f}s;"'
        rows += (
            f'<tr {anim}>'
            f'<td {td}><span style="font-size:11px;color:{TEXT2};">{_to_ct(ts)}</span></td>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_badge("BUY")}</td>'
            f'<td {td}><span style="font-family:Courier New,monospace;color:{TEXT1};">'
            f'${price:.2f}</span></td>'
            f'<td {td}><span style="font-weight:700;color:{conf_c};">{conf_pct}</span></td>'
            f'<td {td}><span style="font-size:12px;color:{TEXT2};">{regime}</span></td>'
            f'<td {td}><span style="font-size:12px;color:{TEXT2};">{driver_text}</span></td>'
            f'</tr>'
        )
    note = f"last {len(shown)} signals · confidence = XGBoost + LSTM + sentiment ensemble"
    help_block = (
        f'<div style="background:{BG};border-top:1px solid {BORDER};'
        f'padding:8px 14px;font-size:10px;color:{TEXT2};line-height:1.7;">'
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
    except Exception:
        pass

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
        + _card("Portfolio Risk",  overall_risk,       TEXT2, risk_c,
                "VIX + drawdown + concentration", 0.00)
        + _card("Max Drawdown",    f"{max_dd:.1f}%",   TEXT2, dd_c,
                "Peak-to-trough all-time",        0.06)
        + _card("Today's P&L",     f"{daily_pnl:+.2f}%", TEXT2, dl_c,
                "Realised from closed trades",    0.12)
        + _card("Cash Reserve",    f"{cash_pct:.1f}%", TEXT2, ca_c,
                "Uninvested capital buffer",      0.18)
        + f'</div>'
    )

    sector_rows = ""
    for sector, pct in list(sector_pcts.items())[:5]:
        bar_c = LOSS if pct > 50 else (NEURAL if pct > 30 else GAIN)
        sector_rows += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
            f'<span style="font-size:11px;color:{TEXT2};width:70px;flex-shrink:0;">{sector}</span>'
            f'<div style="background:{BORDER};border-radius:2px;height:5px;flex:1;overflow:hidden;">'
            f'<div style="background:{bar_c};height:100%;width:{min(pct,100):.0f}%;"></div></div>'
            f'<span style="font-size:11px;color:{TEXT1};width:36px;text-align:right;">{pct:.0f}%</span>'
            f'</div>'
        )
    if not sector_rows:
        sector_rows = f'<div style="color:{TEXT2};font-size:12px;">No open positions — fully in cash</div>'

    note = (f'Concentration: <span style="color:{cc_c};font-weight:700;">{max_conc:.1f}%</span>'
            f' largest position')
    return (f'<div class="nt nt-wrap">'
            f'{_section("🛡","Risk Controls","real-time")}'
            f'{cards}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-radius:8px;padding:14px 16px;margin-top:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            f'<div style="font-size:11px;color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;">Sector Exposure</div>'
            f'<div style="font-size:11px;color:{TEXT2};">{note}</div>'
            f'</div>'
            f'{sector_rows}</div></div>')


# ── Render: institutional metrics ─────────────────────────────────────────────
def render_institutional_metrics() -> str:
    d  = get_data()
    df = d["trades_df"]

    if df.empty or "portfolio_value" not in df.columns:
        msg = f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:12px;">No trade history yet.</div>'
        return f'<div class="nt nt-wrap">{_section("📐","Institutional Metrics")}{_wrap(msg)}</div>'

    daily = (df.dropna(subset=["portfolio_value"])
               .groupby("date")["portfolio_value"].last()
               .reset_index()
               .sort_values("date"))
    daily.columns = ["date", "value"]

    if len(daily) < 3:
        msg = f'<div style="color:{TEXT2};text-align:center;padding:28px;font-size:12px;">Need ≥ 3 days of history.</div>'
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
            f'background:{SURFACE};color:{TEXT2};font-size:11px;font-weight:600;">{label}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {BORDER};'
            f'background:{SURFACE};font-family:-apple-system,monospace;'
            f'color:{color};font-weight:700;">{val_str}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {BORDER};'
            f'background:{SURFACE};color:{TEXT2};font-size:11px;">{desc}</td></tr>'
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
        f'padding:8px 14px;font-size:10px;color:{TEXT2};line-height:1.6;">'
        f'Metrics computed from all trade history since launch. '
        f'Short history (&lt;30 days) may produce unreliable Sharpe / Sortino estimates.'
        f'</div>'
    )
    n_str = f"{n_days} days of history" if n_days > 0 else "—"
    table = _wrap(f'<table class="nt-tbl" style="width:100%">{rows}</table>' + help_block)
    return (f'<div class="nt nt-wrap">'
            f'{_section("📐","Institutional Metrics", n_str)}{table}</div>')


# ── Render: AI decision feed (trade timeline) ────────────────────────────────
def render_timeline() -> str:
    d  = get_data()
    df = d["trades_df"]
    if df.empty:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:40px;font-size:13px;">'
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
            except Exception:
                pass
            detail = " · ".join(parts)
        else:
            reason  = _SELL_REASON.get(action, "Exit")
            pnl_str = f"{pnl:+.1%}" if pnl != 0 else ""
            detail  = f"{reason} · {pnl_str}" if pnl_str else reason

        conf_badge = ""
        if action == "BUY" and conf > 0:
            c_c = GAIN if conf >= 0.75 else (NEURAL if conf >= 0.60 else TEXT2)
            conf_badge = (f'<span style="font-size:11px;color:{c_c};font-weight:700;">'
                          f'{conf*100:.0f}%</span>')

        line = f'border-bottom:1px solid {BORDER};' if not is_last else ''
        connector = (f'<div style="width:1px;flex:1;background:{BORDER};min-height:14px;"></div>'
                     if not is_last else '')
        items += (
            f'<div style="display:flex;gap:14px;padding:10px 0;{line}">'
            f'<div style="flex-shrink:0;width:58px;text-align:right;">'
            f'<div style="font-size:12px;color:{TEXT1};font-family:monospace;font-weight:600;">{time_lbl}</div>'
            f'<div style="font-size:10px;color:{TEXT2};">{tz_lbl}</div>'
            f'<div style="font-size:10px;color:{TEXT2};">{date_lbl}</div>'
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
            f'<div style="font-size:11px;color:{TEXT2};white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;">{detail}</div>'
            f'</div></div>'
        )
    return (f'<div class="nt nt-wrap">'
            f'{_section("🕐","AI Decision Feed",f"last {len(recent)} decisions · newest first")}'
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:0 16px;">'
            f'{items}</div></div>')


# ── Render: investor view (plain-language Models tab) ────────────────────────
def render_investor_view() -> str:
    d  = get_data()
    df = d["trades_df"]
    if df.empty:
        msg = (f'<div style="color:{TEXT2};text-align:center;padding:32px;font-size:13px;">'
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
        + _card("Win Rate",          f"{wr:.0%}" if n_s > 0 else "—",
                TEXT2, GAIN if wr >= 0.55 else (NEURAL if wr >= 0.45 else LOSS),
                f"AI correct {len(wins)} of {n_s} closed trades", 0.0)
        + _card("Avg Winning Trade", f"+{avg_w:.1f}%" if avg_w > 0 else "—",
                TEXT2, GAIN, "Average gain per winning trade", 0.06)
        + _card("Avg Losing Trade",  f"{avg_l:.1f}%"  if avg_l < 0 else "—",
                TEXT2, LOSS, "Average loss per losing trade",  0.12)
        + _card("Risk / Reward",     f"{rr:.1f}×"     if rr > 0   else "—",
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
        except Exception:
            pass
    top3 = sorted(signal_counts.items(), key=lambda x: -x[1])[:3]
    sig_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid {BORDER};">'
        f'<span style="font-size:15px;">📡</span>'
        f'<span style="font-size:13px;color:{TEXT1};">{name}</span>'
        f'<span style="margin-left:auto;font-size:11px;color:{TEXT2};">fired {cnt}× recently</span>'
        f'</div>'
        for name, cnt in top3
    ) or f'<div style="color:{TEXT2};font-size:12px;padding:10px 0;">Building signal history.</div>'

    signals_box = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;margin-top:8px;">'
        f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;">Most Common Buy Signals</div>'
        f'{sig_rows}</div>'
    )

    last6 = sells.tail(6).iloc[::-1]
    result_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid {BORDER};">'
        f'<span style="font-size:15px;">{"✅" if float(row.get("pnl_pct",0) or 0) > 0 else "❌"}</span>'
        f'<span style="font-family:Courier New,monospace;font-weight:700;color:{PRIMARY};font-size:13px;">{row.get("symbol","")}</span>'
        f'<span style="font-size:11px;color:{TEXT2};">{_SELL_REASON.get(str(row.get("action","")),"Exit")}</span>'
        f'<span style="margin-left:auto;font-weight:700;color:{"" + GAIN if float(row.get("pnl_pct",0) or 0) > 0 else LOSS};">'
        f'{float(row.get("pnl_pct",0) or 0):+.1%}</span>'
        f'</div>'
        for _, row in last6.iterrows()
    ) or f'<div style="color:{TEXT2};font-size:12px;padding:10px 0;">No closed trades yet.</div>'

    results_box = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;margin-top:8px;">'
        f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;">Recent Trade Results</div>'
        f'{result_rows}</div>'
    )
    explain = (
        f'<div style="background:{BG};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;margin-top:8px;">'
        f'<div style="font-size:12px;color:{TEXT2};line-height:1.7;">'
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


def render_symbol_detail(symbol: str) -> str:
    if not symbol:
        return (f'<div class="nt nt-wrap"><div style="color:{TEXT2};text-align:center;'
                f'padding:20px;font-size:12px;">Select a symbol above to see its AI analysis.</div></div>')
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
                f'<div><div style="font-size:12px;color:{TEXT1};">{name}</div>'
                f'<div style="font-size:11px;color:{TEXT2};">{desc}</div></div></div>'
            )
    except Exception:
        pass
    if not why_html:
        why_html = f'<div style="color:{TEXT2};font-size:12px;padding:8px 0;">SHAP breakdown available after next model retrain.</div>'

    # Mini model bars
    def _mbar(label, v, c):
        return (f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                f'<span style="font-size:11px;color:{TEXT2};width:60px;">{label}</span>'
                f'<div style="background:{BORDER};border-radius:2px;height:4px;flex:1;">'
                f'<div style="background:{c};height:100%;width:{int(v*100)}%;"></div></div>'
                f'<span style="font-size:11px;color:{c};width:32px;text-align:right;">{v*100:.0f}%</span></div>')

    model_html = ""
    if xgb_p > 0 or lstm_p > 0:
        xc = GAIN if xgb_p >= 0.70 else (NEURAL if xgb_p >= 0.55 else TEXT2)
        lc = GAIN if lstm_p >= 0.70 else (NEURAL if lstm_p >= 0.55 else TEXT2)
        model_html = (
            f'<div style="margin-top:10px;">'
            f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">Model Scores</div>'
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
            f'<span style="font-family:monospace;font-size:12px;color:{TEXT1};">${px:.2f}</span>'
            f'<span style="font-size:11px;color:{p_c};margin-left:auto;">'
            f'{f"{p:+.1%}" if p != 0 else ""}</span></div>'
        )

    # Stats grid
    stat_g = (
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;">'
        f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">AI Score</div>'
        f'<div style="font-size:20px;font-weight:700;color:{conf_c};">{conf_pct}</div></div>'
        f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Sentiment</div>'
        f'<div style="font-size:18px;font-weight:700;color:{sent_c};">{sent_l}</div></div>'
        f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Regime</div>'
        f'<div style="font-size:14px;font-weight:700;color:{r_color};">{regime}</div></div>'
        f'</div>'
    )
    pos_g = ""
    if pos:
        cur_str = f"${cur_price:.2f}" if cur_price > 0 else "—"
        pos_g = (
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;">'
            f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
            f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Entry</div>'
            f'<div style="font-size:18px;font-weight:700;color:{TEXT1};">${entry:.2f}</div></div>'
            f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
            f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Current</div>'
            f'<div style="font-size:18px;font-weight:700;color:{TEXT1};">{cur_str}</div></div>'
            f'<div style="background:{BG};border-radius:6px;padding:10px 12px;">'
            f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">Unrealized P&amp;L</div>'
            f'<div style="font-size:18px;font-weight:700;color:{pnl_c};">{pnl_str}</div></div>'
            f'</div>'
        )

    ts_note = f'<div style="font-size:10px;color:{TEXT2};margin-top:8px;">Signal: {_to_ct(ts)[:16]}</div>' if ts else ""
    card = (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-top:3px solid {PRIMARY};border-radius:8px;padding:20px;">'
        f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px;">'
        f'<span style="font-family:Courier New,monospace;font-size:30px;font-weight:700;color:{PRIMARY};letter-spacing:-1px;">{symbol}</span>'
        f'<span style="background:{SURFACE2};border:1px solid {status_c};color:{status_c};'
        f'padding:2px 10px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:.5px;">{status_lbl}</span>'
        f'</div>'
        f'{stat_g}{pos_g}'
        f'<div class="nt-ai-split">'
        f'<div><div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;">Why the AI entered</div>'
        f'{why_html}{model_html}</div>'
        f'<div class="nt-ai-right">'
        f'<div style="font-size:10px;color:{TEXT2};text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;">Trade History</div>'
        f'{hist_rows}{ts_note}'
        f'</div></div></div>'
    )
    return f'<div class="nt nt-wrap">{_section("🔍", symbol, "AI analysis")}{card}</div>'


# ── Render: since-yesterday comparison panel ─────────────────────────────────
def render_whats_changed() -> str:
    today     = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    d         = datetime.date.today()
    date_label = f"{d.strftime('%B')} {d.day}"

    def _empty(msg: str) -> str:
        inner = (f'<div style="color:{TEXT2};text-align:center;padding:24px;font-size:12px;">{msg}</div>')
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
                f'<span style="font-size:16px;">{icon}</span>'
                f'<span style="font-size:13px;color:{TEXT1};">Portfolio '
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
                        f'color:{PRIMARY};font-size:13px;">{sym}</span>') if i == 0 else ""
            rows_html += (
                f'<div style="display:grid;grid-template-columns:80px 100px 32px 1fr;'
                f'align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid {BORDER};">'
                f'<div>{sym_cell}</div>'
                f'<div style="font-size:12px;color:{TEXT2};">{metric}</div>'
                f'<div style="text-align:center;font-size:15px;">{arrow_html}</div>'
                f'<div style="font-size:12px;color:{TEXT1};">{mag}</div>'
                f'</div>'
            )

    if not changes_seen:
        rows_html = (
            f'<div style="color:{TEXT2};font-size:12px;padding:12px 0;text-align:center;">'
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


def render_portfolio_performance(period: str = "1M  —") -> str:
    # Strip the inline stat suffix so we always have a clean key
    key = period.split()[0] if period else "1M"

    stats = _query_perf_stats()
    cur_row_val = None
    first_any   = any(v is not None for v in stats.values())

    if not first_any:
        empty = (f'<div style="color:{TEXT2};text-align:center;padding:20px;font-size:12px;">'
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
                f'<div style="font-size:10px;color:{"" + PRIMARY if pk == key else TEXT2};'
                f'font-weight:700;margin-bottom:4px;">{pk}</div>'
                f'<div style="font-size:12px;color:{c};font-weight:700;">{pct:+.1f}%</div>'
                f'</div>'
            )
        else:
            strip_items += (
                f'<div style="text-align:center;padding:8px 12px;background:{BG};'
                f'border:1px solid {BORDER};border-radius:6px;min-width:60px;opacity:0.4;">'
                f'<div style="font-size:10px;color:{TEXT2};font-weight:700;margin-bottom:4px;">{pk}</div>'
                f'<div style="font-size:11px;color:{TEXT2};">—</div>'
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
            f'padding:20px;text-align:center;color:{TEXT2};font-size:13px;">'
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
            f'<div style="font-size:32px;font-weight:700;color:{c};letter-spacing:-1px;'
            f'line-height:1;margin-bottom:8px;">'
            f'{sign}${abs(delta):,.2f}</div>'
            # Subline
            f'<div style="font-size:14px;color:{TEXT2};margin-bottom:14px;">'
            f'{sign}{pct:.2f}% {label_str}</div>'
            # From → To
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:13px;color:{TEXT2};">From</span>'
            f'<span style="font-size:15px;font-weight:700;color:{TEXT1};">${start_val:,.2f}</span>'
            f'<span style="font-size:13px;color:{TEXT2};">→</span>'
            f'<span style="font-size:15px;font-weight:700;color:{TEXT1};">${end_val:,.2f}</span>'
            f'</div>'
            # Start date
            f'<div style="font-size:11px;color:{TEXT2};margin-top:10px;">'
            f'Period start: {start_date}</div>'
            f'</div>'
        )

    return f'<div class="nt nt-wrap">{strip}{detail}</div>'


# ── Gradio layout — 4-tab design ──────────────────────────────────────────────
# Gradio 5 removed every= from components. Use gr.Timer + .tick() instead.
with gr.Blocks(title="TradeGenius AI", theme=gr.themes.Base(), css=GRADIO_CSS) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.TabItem("📊 Dashboard"):
            ai_rec_out        = gr.HTML(value=render_ai_recommendation)
            hero_out          = gr.HTML(value=render_dashboard_hero)
            whats_changed_out = gr.HTML(value=render_whats_changed)
            risk_panel_out    = gr.HTML(value=render_risk_panel)
            with gr.Row():
                with gr.Column(scale=50):
                    mkt_intel_out = gr.HTML(value=render_market_intelligence)
                with gr.Column(scale=50):
                    watchlist_out = gr.HTML(value=render_watchlist)
            # ── Symbol drilldown ──────────────────────────────────────────────
            symbol_selector = gr.Dropdown(
                choices=_get_symbol_choices(),
                label="🔍 Symbol Detail — select a ticker to drill down",
                value=None, container=True,
            )
            symbol_detail_out = gr.HTML(value="")

        with gr.TabItem("⚡ Signals"):
            timeline_out = gr.HTML(value=render_timeline)
            signals_out  = gr.HTML(value=render_signals_tab)

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
            pnl_plot   = gr.Plot(value=render_pnl_chart, label="")
            pos_out    = gr.HTML(value=render_positions)
            trades_out = gr.HTML(value=render_trades)

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
    timer.tick(fn=render_dashboard_hero,           outputs=hero_out)
    timer.tick(fn=render_whats_changed,            outputs=whats_changed_out)
    timer.tick(fn=render_ai_recommendation,        outputs=ai_rec_out)
    timer.tick(fn=render_risk_panel,               outputs=risk_panel_out)
    timer.tick(fn=render_market_intelligence,      outputs=mkt_intel_out)
    timer.tick(fn=render_watchlist,                outputs=watchlist_out)
    timer.tick(fn=render_timeline,                 outputs=timeline_out)
    timer.tick(fn=render_signals_tab,              outputs=signals_out)
    timer.tick(fn=lambda: gr.update(choices=_perf_choices()), outputs=perf_tabs)
    timer.tick(fn=render_portfolio_performance,        outputs=perf_out)
    timer.tick(fn=render_equity_chart,             outputs=eq_plot)
    timer.tick(fn=render_allocation_chart,         outputs=alloc_plot)
    timer.tick(fn=render_pnl_chart,                outputs=pnl_plot)
    timer.tick(fn=render_positions,                outputs=pos_out)
    timer.tick(fn=render_trades,                   outputs=trades_out)
    timer.tick(fn=render_investor_view,            outputs=investor_out)
    timer.tick(fn=render_institutional_metrics,    outputs=metrics_out)
    timer.tick(fn=render_feature_importance_chart, outputs=fi_plot)
    timer.tick(fn=render_validation_report,        outputs=val_out)
    timer.tick(fn=lambda: gr.update(choices=_get_symbol_choices()), outputs=symbol_selector)

if __name__ == "__main__":
    demo.launch()
