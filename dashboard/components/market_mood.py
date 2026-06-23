"""Market Mood widget — intraday broad market, sector, and stock snapshot."""
from __future__ import annotations

import datetime
import logging
import time
from typing import Optional

from bot.core.error_logger import log_exception
from dashboard.design_system import (
    _card, _section, _wrap,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM,
    TEXT1, TEXT2, TEXT3, BORDER,
    FONT_VALUE, FONT_LABEL, WEIGHT_BOLD, WEIGHT_MEDIUM,
)

_log = logging.getLogger("tradegenie.market_mood")

_mood_cache: Optional[dict] = None
_mood_cache_ts: float = 0.0
_MOOD_TTL: float = 300.0  # refresh every 5 minutes

SECTOR_ETFS: dict[str, str] = {
    "Technology":       "XLK",
    "Financials":       "XLF",
    "Healthcare":       "XLV",
    "Energy":           "XLE",
    "Industrials":      "XLI",
    "Consumer Disc":    "XLY",
    "Staples":          "XLP",
    "Comm. Services":   "XLC",
}


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_mood_data() -> dict:
    """Batch-download 2 days of OHLCV and compute today's % changes. Cached 5 min."""
    global _mood_cache, _mood_cache_ts

    now = time.time()
    if _mood_cache is not None and (now - _mood_cache_ts) < _MOOD_TTL:
        return _mood_cache

    try:
        import yfinance as yf
        from config import STOCKS

        broad = ["SPY", "QQQ", "IWM", "^VIX"]
        sectors = list(SECTOR_ETFS.values())
        all_tickers = broad + sectors + list(STOCKS)

        raw = yf.download(all_tickers, period="2d", progress=False, auto_adjust=True)
        close = raw["Close"]  # DataFrame with tickers as columns

        def _pct(sym: str) -> Optional[float]:
            try:
                col = close[sym].dropna()
                if len(col) < 2:
                    return None
                return float((col.iloc[-1] - col.iloc[-2]) / col.iloc[-2] * 100)
            except Exception:
                return None

        def _level(sym: str) -> Optional[float]:
            try:
                col = close[sym].dropna()
                return float(col.iloc[-1]) if not col.empty else None
            except Exception:
                return None

        result: dict = {
            "spy": _pct("SPY"),
            "qqq": _pct("QQQ"),
            "iwm": _pct("IWM"),
            "vix": _level("^VIX"),
            "sectors": {name: v for name, etf in SECTOR_ETFS.items()
                        if (v := _pct(etf)) is not None},
            "stocks": {s: v for s in STOCKS if (v := _pct(s)) is not None},
        }

        _mood_cache = result
        _mood_cache_ts = now
        return result

    except Exception as exc:
        log_exception(_log, "_fetch_mood_data", exc)
        _mood_cache = {}
        _mood_cache_ts = now
        return {}


# ── Classification helpers ─────────────────────────────────────────────────────

def _mood_label(spy: float) -> tuple[str, str]:
    """Return (label, hex_color) for the mood badge."""
    if spy >= 1.5:
        return "Strong Rally",   ACTION_BUY
    if spy >= 0.5:
        return "Mild Bull",      "#4caf50"
    if spy >= -0.5:
        return "Neutral",        TEXT2
    if spy >= -1.5:
        return "Mild Pullback",  ACTION_TRIM
    return "Broad Sell-off",     ACTION_SELL


def _sector_note(spy: float, sectors: dict[str, float]) -> str:
    """Return a short note when one or two sectors are clear outliers vs SPY."""
    if not sectors:
        return ""
    threshold = max(abs(spy) * 1.5, 1.5)
    outliers = [name for name, v in sectors.items() if abs(v - spy) >= threshold]
    if not outliers:
        return ""
    return "Led by: " + ", ".join(outliers[:2])


def _stock_outliers(
    spy: float, stocks: dict[str, float]
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return (laggards, leaders) relative to SPY move."""
    gap = max(abs(spy) * 1.5, 2.0)
    laggards = sorted(
        [(s, v) for s, v in stocks.items() if v < spy - gap], key=lambda x: x[1]
    )[:4]
    leaders = sorted(
        [(s, v) for s, v in stocks.items() if v > spy + gap],
        key=lambda x: x[1], reverse=True,
    )[:4]
    return laggards, leaders


# ── HTML builders ─────────────────────────────────────────────────────────────

def _pct_span(v: Optional[float], color: str) -> str:
    if v is None:
        return f'<span style="color:{TEXT3};">—</span>'
    return (f'<span style="color:{color};font-weight:{WEIGHT_BOLD};'
            f'font-size:{FONT_VALUE};">{v:+.2f}%</span>')


def _index_pill(label: str, v: Optional[float]) -> str:
    c = ACTION_BUY if (v or 0) >= 0 else ACTION_SELL
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'gap:2px;min-width:56px;">'
        f'<span style="font-size:{FONT_LABEL};color:{TEXT3};'
        f'text-transform:uppercase;letter-spacing:1px;">{label}</span>'
        f'{_pct_span(v, c)}'
        f'</div>'
    )


def _sector_bar(name: str, v: float, spy: float) -> str:
    c = ACTION_BUY if v >= 0 else ACTION_SELL
    bar_w = min(int(abs(v) * 12), 72)
    is_outlier = abs(v - spy) >= max(abs(spy) * 1.5, 1.5)
    name_style = (f"color:{TEXT1};font-weight:{WEIGHT_BOLD};"
                  if is_outlier else f"color:{TEXT2};")
    return (
        f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
        f'<span style="font-size:{FONT_LABEL};{name_style}min-width:115px;">{name}</span>'
        f'<div style="width:{bar_w}px;height:3px;background:{c}44;border-radius:2px;'
        f'border-right:2px solid {c};"></div>'
        f'<span style="color:{c};font-size:{FONT_LABEL};font-weight:{WEIGHT_MEDIUM};">'
        f'{v:+.1f}%</span>'
        f'</div>'
    )


def _chip(sym: str, v: float, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:3px 8px;margin:2px 2px;'
        f'border-radius:4px;background:{color}22;color:{color};'
        f'font-size:{FONT_LABEL};font-family:Courier New,monospace;'
        f'letter-spacing:0.5px;">{sym} {v:+.1f}%</span>'
    )


# ── Main render function ───────────────────────────────────────────────────────

def render_market_mood() -> str:
    try:
        data = _fetch_mood_data()
        if not data or data.get("spy") is None:
            return (
                f'<div class="nt nt-wrap">'
                f'{_section("🌡", "Market Mood", "unavailable")}'
                f'{_card(f"<span style=\'color:{TEXT3}\'>Market data unavailable</span>")}'
                f'</div>'
            )

        spy     = data["spy"]
        qqq     = data.get("qqq")
        iwm     = data.get("iwm")
        vix     = data.get("vix")
        sectors = data.get("sectors", {})
        stocks  = data.get("stocks", {})

        label, mood_color = _mood_label(spy)
        sector_note       = _sector_note(spy, sectors)
        laggards, leaders = _stock_outliers(spy, stocks)

        # ── Mood badge ──────────────────────────────────────────────
        badge = (
            f'<span style="display:inline-block;padding:5px 14px;border-radius:6px;'
            f'background:{mood_color}22;color:{mood_color};font-weight:{WEIGHT_BOLD};'
            f'font-size:13px;letter-spacing:0.5px;">{label}</span>'
        )
        note_span = (
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};'
            f'text-transform:uppercase;letter-spacing:1px;">{sector_note}</span>'
            if sector_note else ""
        )
        header_row = (
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">'
            f'{badge}{note_span}'
            f'</div>'
        )

        # ── Broad indices + VIX ─────────────────────────────────────
        vix_color = (ACTION_SELL if (vix or 0) > 28
                     else ACTION_TRIM if (vix or 0) > 20 else TEXT2)
        vix_pill = (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'gap:2px;min-width:56px;">'
            f'<span style="font-size:{FONT_LABEL};color:{TEXT3};'
            f'text-transform:uppercase;letter-spacing:1px;">VIX</span>'
            f'<span style="color:{vix_color};font-weight:{WEIGHT_BOLD};'
            f'font-size:{FONT_VALUE};">{vix:.1f}</span>'
            f'</div>'
            if vix else ""
        )
        indices_row = (
            f'<div style="display:flex;gap:20px;align-items:center;padding:10px 0;'
            f'border-bottom:1px solid {BORDER};margin-bottom:12px;">'
            f'{_index_pill("SPY", spy)}'
            f'{_index_pill("QQQ", qqq)}'
            f'{_index_pill("IWM", iwm)}'
            f'<div style="flex:1;"></div>'
            f'{vix_pill}'
            f'</div>'
        )

        # ── Sector bars (sorted worst → best) ──────────────────────
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1])
        sector_bars = "".join(_sector_bar(name, v, spy) for name, v in sorted_sectors)
        sector_block = (
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT3};text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:6px;">Sectors</div>'
            f'{sector_bars}'
            f'</div>'
        ) if sector_bars else ""

        # ── Stock outliers ──────────────────────────────────────────
        outlier_chips = "".join(
            _chip(s, v, ACTION_SELL) for s, v in laggards
        ) + "".join(
            _chip(s, v, ACTION_BUY) for s, v in leaders
        )
        outlier_block = (
            f'<div style="padding-top:8px;border-top:1px solid {BORDER};">'
            f'<div style="font-size:{FONT_LABEL};color:{TEXT3};text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:6px;">Notable movers vs market</div>'
            f'{outlier_chips}'
            f'</div>'
        ) if outlier_chips else ""

        content = header_row + indices_row + sector_block + outlier_block

        _now = datetime.datetime.now()
        now_str = _now.strftime("%I:%M %p").lstrip("0")
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("🌡", "Market Mood", f"Today · {now_str} CDT · refreshes 5 min")}'
            f'{_card(content)}'
            f'</div>'
        )

    except Exception as exc:
        log_exception(_log, "render_market_mood", exc)
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("🌡", "Market Mood", "error")}'
            f'<p style="color:{TEXT3};padding:12px;">Unable to load market mood</p>'
            f'</div>'
        )
