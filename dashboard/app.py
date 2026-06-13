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
HF_REPO_ID = os.getenv("HF_REPO_ID", "ksri77/ai-trading-bot")

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
                                  token=HF_TOKEN, force_download=True)
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
                                      token=HF_TOKEN, force_download=True)
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
    cfg = {"BUY": (GAIN_BG, GAIN, GAIN_BD), "SELL": (LOSS_BG, LOSS, LOSS_BD)}.get(
        action, (NEURAL_BG, NEURAL, NEURAL_BD))
    bg, fg, bd = cfg
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
               if k not in ("xaxis", "yaxis", "gridcolor", "zerolinecolor")},
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
            yaxis=dict(title="P&L ($)", **PLOTLY_LAYOUT["yaxis"], tickformat="+$,.0f",
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

    rows  = ""
    items = list(open_syms.items())
    for i, (sym, v) in enumerate(items):
        cur_price = prices.get(sym, 0.0)
        cur_val   = v["shares"] * cur_price
        invested  = v["invested"]
        pnl       = cur_val - invested
        pnl_pct   = (pnl / invested * 100) if invested > 0 else 0.0
        cv_str  = f"${cur_val:.2f}"   if cur_price else "—"
        p_str   = f"${pnl:+.2f}"     if cur_price else "—"
        pct_str = f"{pnl_pct:+.2f}%" if cur_price else "—"
        td   = TD if i < len(items) - 1 else TD0
        anim = f'style="animation:slideInRow .35s ease both;animation-delay:{i*0.07:.2f}s;"'
        rows += (
            f'<tr {anim}>'
            f'<td {td}>{_sym(sym)}</td>'
            f'<td {td}>{_num(str(round(v["shares"],4)))}</td>'
            f'<td {td}>{_num(f"${invested:.2f}",bold=True)}</td>'
            f'<td {td}>{_num(cv_str,bold=True)}</td>'
            f'<td {td}>{_pnl(p_str)}</td>'
            f'<td {td}>{_pnl(pct_str,big=True)}</td>'
            f'</tr>'
        )
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Symbol</th>'
        f'<th {TH}>Shares  <span style="font-weight:400;text-transform:none;letter-spacing:0;">held</span></th>'
        f'<th {TH}>Invested  <span style="font-weight:400;text-transform:none;letter-spacing:0;">cost basis</span></th>'
        f'<th {TH}>Current Value  <span style="font-weight:400;text-transform:none;letter-spacing:0;">live price</span></th>'
        f'<th {TH}>P&amp;L $  <span style="font-weight:400;text-transform:none;letter-spacing:0;">unrealised</span></th>'
        f'<th {TH}>P&amp;L %</th>'
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

    val_color = pnl_color if total_invested > 0 else TEXT1
    cards = (
        f'<div class="nt-cards">'
        + _big("Portfolio Value",  portfolio_val,     f"Unrealized: {hero_chg}", val_color)
        + _big("Open Positions",   str(len(open_syms)), "Stocks held now (max 8)",  TEXT1)
        + _big("AI Confidence",    conf_str,          "Avg signal strength · last 5 buys", conf_color)
        + _big("VIX",              vix_str,           "Fear gauge · <15 calm · >30 fear",  vix_color)
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


# ── Gradio layout — 4-tab design ──────────────────────────────────────────────
# Gradio 5 removed every= from components. Use gr.Timer + .tick() instead.
with gr.Blocks(title="TradeGenius AI", theme=gr.themes.Base(), css=GRADIO_CSS) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.TabItem("📊 Dashboard"):
            ai_rec_out    = gr.HTML(value=render_ai_recommendation)   # hero — first thing visible
            hero_out      = gr.HTML(value=render_dashboard_hero)
            risk_panel_out = gr.HTML(value=render_risk_panel)
            with gr.Row():
                with gr.Column(scale=50):
                    mkt_intel_out = gr.HTML(value=render_market_intelligence)
                with gr.Column(scale=50):
                    watchlist_out = gr.HTML(value=render_watchlist)

        with gr.TabItem("⚡ Signals"):
            signals_out = gr.HTML(value=render_signals_tab)

        with gr.TabItem("💼 Portfolio"):
            with gr.Row():
                with gr.Column(scale=65):
                    eq_plot    = gr.Plot(value=render_equity_chart, label="")
                with gr.Column(scale=35):
                    alloc_plot = gr.Plot(value=render_allocation_chart, label="")
            pnl_plot   = gr.Plot(value=render_pnl_chart, label="")
            pos_out    = gr.HTML(value=render_positions)
            trades_out = gr.HTML(value=render_trades)

        with gr.TabItem("🔬 Models"):
            metrics_out = gr.HTML(value=render_institutional_metrics)
            with gr.Row():
                with gr.Column(scale=65):
                    fi_plot = gr.Plot(value=render_feature_importance_chart, label="")
                with gr.Column(scale=35):
                    val_out = gr.HTML(value=render_validation_report)

    gr.HTML(value=FOOTER_HTML)

    # One shared timer — cache layer ensures a single DB+API refresh per tick
    timer = gr.Timer(value=60)
    timer.tick(fn=render_dashboard_hero,           outputs=hero_out)
    timer.tick(fn=render_ai_recommendation,        outputs=ai_rec_out)
    timer.tick(fn=render_risk_panel,               outputs=risk_panel_out)
    timer.tick(fn=render_market_intelligence,      outputs=mkt_intel_out)
    timer.tick(fn=render_watchlist,                outputs=watchlist_out)
    timer.tick(fn=render_signals_tab,              outputs=signals_out)
    timer.tick(fn=render_equity_chart,             outputs=eq_plot)
    timer.tick(fn=render_allocation_chart,         outputs=alloc_plot)
    timer.tick(fn=render_pnl_chart,                outputs=pnl_plot)
    timer.tick(fn=render_positions,                outputs=pos_out)
    timer.tick(fn=render_trades,                   outputs=trades_out)
    timer.tick(fn=render_institutional_metrics,    outputs=metrics_out)
    timer.tick(fn=render_feature_importance_chart, outputs=fi_plot)
    timer.tick(fn=render_validation_report,        outputs=val_out)

if __name__ == "__main__":
    demo.launch()
