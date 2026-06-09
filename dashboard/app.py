"""Gradio dashboard — TradeGenius AI, hosted on HuggingFace Spaces."""
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

# ── Design tokens ─────────────────────────────────────────────────────────────
BG        = "#060810"
SURFACE   = "#0e1420"
BORDER    = "#1a2236"
PRIMARY   = "#00c8ff"
GAIN      = "#00e676"
LOSS      = "#ff3d57"
NEURAL    = "#9d4edd"
TEXT1     = "#f0f6fc"
TEXT2     = "#8892a4"
PRIMARY_BG = "#001a2e"
GAIN_BG    = "#001a0d"
LOSS_BG    = "#1a0010"
NEURAL_BG  = "#14003a"
GAIN_BD    = "#00b854"
LOSS_BD    = "#cc1f2e"
NEURAL_BD  = "#7b2fc9"

# Plotly shared theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT2, family="Inter, monospace", size=11),
    margin=dict(l=50, r=20, t=40, b=50),
    gridcolor=BORDER,
    zerolinecolor=BORDER,
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(color=TEXT2)),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
)

# ── Gradio CSS: dark page + strip Gradio chrome ───────────────────────────────
GRADIO_CSS = f"""
.gradio-container, .gradio-container > .main {{
  background-color: {BG} !important;
  background-image: radial-gradient(rgba(0,200,255,0.05) 1px, transparent 1px) !important;
  background-size: 22px 22px !important;
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
.nt {{ font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;color:{TEXT1};
  box-sizing:border-box; }}
.nt *, .nt *::before, .nt *::after {{ box-sizing:border-box; }}
.nt-wrap {{ padding:12px 16px 0; }}
.nt-header {{
  display:flex;align-items:center;gap:16px;padding:18px 24px;
  background:{SURFACE};backdrop-filter:blur(16px);
  border-radius:12px;border:1px solid {BORDER};
  box-shadow:0 0 0 1px rgba(0,200,255,0.12),0 8px 32px rgba(0,0,0,0.7);
  position:relative;overflow:hidden;
}}
.nt-header::before {{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,{PRIMARY},{GAIN},{NEURAL},{PRIMARY});
  background-size:200% 100%;animation:shimmer 4s linear infinite;
}}
.nt-status {{
  display:flex;align-items:center;justify-content:space-between;
  padding:8px 14px;margin:10px 0 8px;
  background:{SURFACE};border:1px solid {BORDER};border-radius:8px;font-size:11px;
}}
.nt-cards {{
  display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px;
}}
.nt-card {{
  background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:16px;
  position:relative;overflow:hidden;transition:border-color .2s,box-shadow .2s;
}}
.nt-card:hover {{ border-color:rgba(0,200,255,.3);box-shadow:0 0 20px rgba(0,200,255,.08); }}
.nt-sec {{
  display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;
  text-transform:uppercase;letter-spacing:1.5px;margin:12px 0 8px;
}}
.nt-sec-line {{ flex:1;height:1px;background:linear-gradient(90deg,{BORDER},transparent); }}
.nt-tbl {{ width:100%;border-collapse:collapse; }}
.nt-tbl th {{
  background:{BG};color:{TEXT2};font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:1px;
  padding:12px 16px;border-bottom:1px solid {BORDER};text-align:left;white-space:nowrap;
}}
.nt-tbl td {{ padding:13px 16px;border-bottom:1px solid #0d1218;vertical-align:middle; }}
.nt-tbl tr:last-child td {{ border-bottom:none; }}
.nt-tbl tr:hover td {{ background:rgba(0,200,255,.025); }}
@keyframes shimmer    {{ 0%{{background-position:0%}} 100%{{background-position:200%}} }}
@keyframes pulse      {{ 0%,100%{{opacity:1}} 50%{{opacity:0.35}} }}
@keyframes fadeInUp   {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes slideInRow {{ from{{opacity:0;transform:translateX(-5px)}} to{{opacity:1;transform:translateX(0)}} }}
@keyframes countdown  {{ from{{width:120px}} to{{width:0px}} }}
.nt-card {{ animation:fadeInUp .4s ease both; }}
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
    <div style="font-size:26px;font-weight:800;letter-spacing:-0.5px;
      background:linear-gradient(135deg,{TEXT1} 0%,{PRIMARY} 50%,{GAIN} 100%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;color:{PRIMARY};">TradeGenius AI</div>
    <div style="font-size:12px;color:{TEXT2} !important;margin-top:3px;">
      Autonomous Paper Trading &nbsp;·&nbsp; PPO &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; FinBERT
    </div>
  </div>
  <div style="display:flex;gap:10px;align-items:center;">
    <div style="display:flex;align-items:center;gap:7px;background:{GAIN_BG} !important;
      border:1px solid {GAIN_BD};color:{GAIN} !important;padding:6px 16px;
      border-radius:20px;font-size:12px;font-weight:800;letter-spacing:.5px;">
      <span style="width:7px;height:7px;background:{GAIN};border-radius:50%;
        display:inline-block;animation:pulse 2s infinite;flex-shrink:0;"></span>LIVE
    </div>
    <div style="background:{SURFACE} !important;border:1px solid {BORDER};
      color:{TEXT2} !important;padding:6px 16px;border-radius:20px;
      font-size:12px;font-weight:700;">PAPER</div>
  </div>
</div>
</div>"""

FOOTER_HTML = f"""<div class="nt nt-wrap">
<div style="text-align:center;color:{TEXT2} !important;font-size:11px;
  margin-top:8px;padding:14px;border-top:1px solid {BORDER};">
  Refreshes every 60 s &nbsp;·&nbsp; Paper trading only &nbsp;·&nbsp;
  Alpaca Markets &nbsp;·&nbsp; TradeGenius AI v1
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
        logger.warning(f"DB sync: {e}")


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
        df = pd.read_sql(
            "SELECT id,timestamp,symbol,action,shares,price,notional,"
            "pnl_pct,portfolio_value,regime FROM trades ORDER BY id",
            con)
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
    sells_mask             = df["action"].isin(["SELL", "SELL_STOP"])
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
        elif row["action"] in ("SELL", "SELL_STOP") and sym in pos and pos[sym]["shares"] > 0:
            avg = pos[sym]["invested"] / pos[sym]["shares"]
            pos[sym]["shares"]   = max(0.0, pos[sym]["shares"] - shares)
            pos[sym]["invested"] = max(0.0, pos[sym]["invested"] - avg * shares)
    result["open_pos"] = {s: d for s, d in pos.items() if d["shares"] > 0.001}

    # Recent trades (last 15, newest first) — columns matching render_trades usage
    recent = df.tail(15).iloc[::-1][
        ["timestamp", "symbol", "action", "shares", "price", "notional", "pnl_pct", "regime"]
    ]
    result["recent_trades"] = list(recent.itertuples(index=False, name=None))

    # Current prices — single yfinance batch call
    if result["open_pos"]:
        result["prices"] = _current_prices(list(result["open_pos"].keys()))

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
            f'border:1px solid #1a2236;border-left:3px solid {PRIMARY};border-radius:4px;'
            f'padding:3px 9px;font-family:Courier New,monospace;font-weight:800;'
            f'font-size:13px;color:{PRIMARY} !important;letter-spacing:.5px;">{s}</span>')

def _num(v: str, bold=False) -> str:
    w = "800" if bold else "600"
    return (f'<span style="font-family:Courier New,monospace;font-weight:{w};'
            f'font-size:14px;color:{TEXT1} !important;">{v}</span>')

def _pnl(v: str, big=False) -> str:
    c  = _pnl_color(v)
    gl = {"#00e676":"rgba(0,230,118,.4)","#ff3d57":"rgba(255,61,87,.4)"}.get(c,"transparent")
    sz = "16px" if big else "14px"; fw = "800" if big else "700"
    return (f'<span style="font-family:Courier New,monospace;font-weight:{fw};'
            f'font-size:{sz};color:{c} !important;text-shadow:0 0 10px {gl};">{v}</span>')

def _badge(action: str) -> str:
    cfg = {"BUY": (GAIN_BG, GAIN, GAIN_BD), "SELL": (LOSS_BG, LOSS, LOSS_BD)}.get(
        action, (NEURAL_BG, NEURAL, NEURAL_BD))
    bg, fg, bd = cfg
    return (f'<span style="display:inline-block;background:{bg} !important;color:{fg} !important;'
            f'border:1px solid {bd};padding:3px 11px;border-radius:4px;font-size:11px;'
            f'font-weight:800;letter-spacing:.3px;font-family:Courier New,monospace;">{action}</span>')

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
    return (f'<div style="background:#000 !important;border:1px solid {BORDER};'
            f'border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.7);">'
            f'{inner}</div>')

def _card(label: str, value: str, accent: str = PRIMARY,
          color: str = TEXT1, sub: str = "", delay: float = 0) -> str:
    """Unified card. Pass color=GAIN/LOSS/NEURAL for colored values."""
    glow = {"#00e676":"rgba(0,230,118,.3)","#ff3d57":"rgba(255,61,87,.3)",
            "#9d4edd":"rgba(157,78,221,.25)"}.get(color, "transparent")
    shadow = f"text-shadow:0 0 16px {glow};" if glow != "transparent" else ""
    sub_html = (f'<div style="font-size:10px;color:{TEXT2} !important;margin-top:3px;">{sub}</div>'
                if sub else "")
    return (
        f'<div class="nt-card" style="animation-delay:{delay:.2f}s;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:{accent};'
        f'border-radius:12px 12px 0 0;"></div>'
        f'<div style="font-size:20px;font-weight:800;letter-spacing:-0.5px;'
        f'color:{color} !important;line-height:1.1;margin-top:4px;{shadow}">{value}</div>'
        f'<div style="font-size:10px;color:{TEXT2} !important;text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:700;margin-top:7px;">{label}</div>'
        f'{sub_html}</div>'
    )

TH  = (f'style="background:{BG} !important;color:{TEXT2} !important;font-size:10px;font-weight:700;'
       f'text-transform:uppercase;letter-spacing:1px;padding:12px 16px;'
       f'border-bottom:1px solid {BORDER};text-align:left;white-space:nowrap;"')
TD  = (f'style="padding:13px 16px;border-bottom:1px solid #0d1218 !important;'
       f'vertical-align:middle;background:#000 !important;color:{TEXT1} !important;"')
TD0 = (f'style="padding:13px 16px;vertical-align:middle;'
       f'background:#000 !important;color:{TEXT1} !important;"')

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

    status = (
        f'<div class="nt-status">'
        f'<span style="color:{TEXT2} !important;font-family:Courier New,monospace;">'
        f'Last updated &nbsp;<strong style="color:{PRIMARY} !important;">{_now_ct()}</strong></span>'
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:6px;height:6px;background:{mkt_color};border-radius:50%;'
        f'display:inline-block;"></span>'
        f'<span style="color:{mkt_color} !important;font-weight:700;font-size:11px;">'
        f'{mkt_label}</span></span>'
        f'<div style="height:2px;width:120px;background:linear-gradient(90deg,{PRIMARY},{GAIN});'
        f'border-radius:1px;animation:countdown 60s linear forwards;"></div>'
        f'<span style="color:{TEXT2} !important;font-size:11px;font-weight:600;">'
        f'&#x23F1; Next refresh in 60s</span>'
        f'</div>'
    )

    row1 = (
        f'<div class="nt-cards">'
        + _card("Unrealized P&amp;L",  pnl_str,             pnl_accent, pnl_color, pnl_sub, 0.00)
        + _card("Total Invested",      invested_str,         PRIMARY,   TEXT1,     "across open positions", 0.08)
        + _card("Portfolio Value",     d["portfolio"],       PRIMARY,   TEXT1,     "Alpaca paper account", 0.16)
        + _card("Market Regime",       d["regime_raw"].title(), r_accent, r_color, delay=0.24)
        + f'</div>'
    )

    row2 = (
        f'<div class="nt-cards">'
        + _card("Open Positions",  str(open_count),
                PRIMARY, TEXT1,
                f"{open_count} symbol(s) held" if open_count else "no active positions", 0.32)
        + _card("Win Rate",        wr_str,
                wr_accent, wr_color,
                f"{win_count} wins of {sell_count} closed", 0.40)
        + _card("Total Trades",    str(d["total_trades"]),
                NEURAL, TEXT1, "all-time executions", 0.48)
        + _card("Buys / Sells",    f"{d['buy_count']} / {d['sell_count']}",
                PRIMARY, TEXT1, "lifetime order split", 0.56)
        + f'</div>'
    )

    return f'<div class="nt nt-wrap">{status}{row1}{row2}</div>'


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
                text="Building history — bot runs 9:30am–4pm ET, Mon–Fri",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=13))
        else:
            daily = df.groupby("date")["portfolio_value"].last().reset_index()
            daily.columns = ["date", "value"]
            daily["date"] = pd.to_datetime(daily["date"])

            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["value"],
                fill="tozeroy", fillcolor="rgba(0,200,255,0.06)",
                line=dict(color=PRIMARY, width=2.5),
                mode="lines+markers",
                marker=dict(color=PRIMARY, size=6, line=dict(color=BG, width=1.5)),
                hovertemplate="<b>%{x|%b %d}</b><br>Portfolio: $%{y:,.2f}<extra></extra>",
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
            title=dict(text="Portfolio Equity Curve", font=dict(color=TEXT1, size=13), x=0.01),
            xaxis=dict(title="", **PLOTLY_LAYOUT["xaxis"], tickfont=dict(color=TEXT2)),
            yaxis=dict(title="Value ($)", **PLOTLY_LAYOUT["yaxis"],
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
        sells = df[df["action"].isin(["SELL", "SELL_STOP"])].copy() if not df.empty else pd.DataFrame()

        if sells.empty or "pnl_pct" not in sells.columns or sells["pnl_pct"].isna().all():
            fig.add_annotation(
                text="No closed trades yet — realized P&L appears here after first sell",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=13))
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
            title=dict(text="Daily Realized P&L", font=dict(color=TEXT1, size=13), x=0.01),
            xaxis=dict(title="", **PLOTLY_LAYOUT["xaxis"], tickfont=dict(color=TEXT2)),
            yaxis=dict(title="P&L ($)", **PLOTLY_LAYOUT["yaxis"], tickformat="$+,.0f",
                       tickfont=dict(color=TEXT2), zeroline=True,
                       zerolinecolor=BORDER, zerolinewidth=1),
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
        empty = (f'<div style="color:{TEXT2} !important;text-align:center;'
                 f'padding:40px;font-size:14px;">No open positions yet.</div>')
        return f'<div class="nt nt-wrap">{_section("📊","Open Positions")}{_wrap(empty)}</div>'

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
        f'<th {TH}>Symbol</th><th {TH}>Shares</th>'
        f'<th {TH}>Invested</th><th {TH}>Current Value</th>'
        f'<th {TH}>P&amp;L $</th><th {TH}>P&amp;L %</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return f'<div class="nt nt-wrap">{_section("📊","Open Positions")}{table}</div>'


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
    table = _wrap(
        f'<table class="nt-tbl"><thead><tr>'
        f'<th {TH}>Time (CT)</th><th {TH}>Symbol</th>'
        f'<th {TH}>Action</th><th {TH}>Qty</th>'
        f'<th {TH}>Price</th><th {TH}>Value</th>'
        f'<th {TH}>P&amp;L</th><th {TH}>Regime</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    return (f'<div class="nt nt-wrap">'
            f'{_section("⚡","Recent Trades", note)}{table}</div>')


# ── Gradio layout ─────────────────────────────────────────────────────────────
# Gradio 5 removed every= from components. Use gr.Timer + .tick() instead.
with gr.Blocks(title="TradeGenius AI", theme=gr.themes.Base(), css=GRADIO_CSS) as demo:
    gr.HTML(HEADER_HTML)

    metrics_out = gr.HTML(value=render_metrics)

    # 65 / 35 split: equity curve dominates, donut is supplemental
    with gr.Row():
        with gr.Column(scale=65):
            eq_plot = gr.Plot(value=render_equity_chart, label="")
        with gr.Column(scale=35):
            alloc_plot = gr.Plot(value=render_allocation_chart, label="")

    pnl_plot   = gr.Plot(value=render_pnl_chart, label="")
    pos_out    = gr.HTML(value=render_positions)
    trades_out = gr.HTML(value=render_trades)
    gr.HTML(value=FOOTER_HTML)

    # One shared timer — cache layer ensures a single DB+API refresh per tick
    timer = gr.Timer(value=60)
    timer.tick(fn=render_metrics,          outputs=metrics_out)
    timer.tick(fn=render_equity_chart,     outputs=eq_plot)
    timer.tick(fn=render_allocation_chart, outputs=alloc_plot)
    timer.tick(fn=render_pnl_chart,        outputs=pnl_plot)
    timer.tick(fn=render_positions,        outputs=pos_out)
    timer.tick(fn=render_trades,           outputs=trades_out)

if __name__ == "__main__":
    demo.launch()
