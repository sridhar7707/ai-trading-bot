"""Market Mood widget &mdash; intraday broad market, sector, and stock snapshot."""
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
    """Return a short note distinguishing sectors dragging down vs holding up."""
    if not sectors:
        return ""
    threshold = max(abs(spy) * 1.5, 1.5)
    dragging = [n for n, v in sectors.items() if v - spy <= -threshold]
    holding  = [n for n, v in sectors.items() if v - spy >= threshold]
    parts = []
    if dragging:
        parts.append("Dragging: " + ", ".join(dragging[:2]))
    if holding:
        parts.append("Holding up: " + ", ".join(holding[:2]))
    return " · ".join(parts)


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


def _generate_reasons(
    spy: float,
    qqq: Optional[float],
    iwm: Optional[float],
    vix: Optional[float],
    sectors: dict[str, float],
    stocks: dict[str, float],
) -> list[tuple[str, str]]:
    """Return list of (icon, text) reason strings derived from the data."""
    reasons: list[tuple[str, str]] = []

    # 1. QQQ vs SPY divergence &mdash; pinpoints tech-specific vs broad move
    if qqq is not None:
        gap = qqq - spy
        if gap < -1.5:
            reasons.append(("📉", f"Tech-specific selling &mdash; QQQ lagging SPY by {abs(gap):.1f}%. "
                            "Large-cap tech is driving the weakness, not the broader market."))
        elif gap > 1.5:
            reasons.append(("📈", f"Tech-led rally &mdash; QQQ ahead of SPY by {gap:.1f}%. "
                            "Growth stocks are pulling the market up."))

    # 2. IWM vs SPY &mdash; signals whether move is macro or large-cap specific
    if iwm is not None:
        gap = iwm - spy
        if gap > 1.0:
            reasons.append(("🏦", f"Small caps ({iwm:+.1f}%) outperforming large caps ({spy:+.1f}%) &mdash; "
                            "selloff is concentrated in mega-cap stocks, not economy-wide."))
        elif gap < -1.0:
            reasons.append(("⚠️", f"Small caps ({iwm:+.1f}%) lagging large caps ({spy:+.1f}%) &mdash; "
                            "broader risk-off; smaller companies more exposed to rate/credit pressure."))

    # 3. Sector rotation pattern
    if sectors:
        worst_name = min(sectors, key=sectors.get)
        best_name  = max(sectors, key=sectors.get)
        worst_v    = sectors[worst_name]
        best_v     = sectors[best_name]
        spread     = best_v - worst_v

        all_neg = all(v < 0 for v in sectors.values())
        all_pos = all(v > 0 for v in sectors.values())

        if all_neg:
            reasons.append(("🔻", "All 8 sectors negative &mdash; broad-based selling with no safe sector."))
        elif all_pos:
            reasons.append(("🟢", "All 8 sectors positive &mdash; broad participation in the rally."))
        elif spread > 3.0:
            neg_sectors = [n for n, v in sectors.items() if v < -0.5]
            pos_sectors = [n for n, v in sectors.items() if v > 0.5]
            if neg_sectors and pos_sectors:
                reasons.append(("🔄", f"Sector rotation &mdash; capital moving out of "
                                f"{', '.join(neg_sectors[:2])} "
                                f"into {', '.join(pos_sectors[:2])}. "
                                f"Spread between best ({best_name} {best_v:+.1f}%) "
                                f"and worst ({worst_name} {worst_v:+.1f}%) is {spread:.1f}pp."))

    # 4. VIX interpretation
    if vix is not None and vix > 0:
        if vix > 30:
            reasons.append(("🚨", f"VIX {vix:.1f} &mdash; extreme fear. Institutions are buying protection at elevated cost. "
                            "Historically precedes sharp reversals."))
        elif vix > 22:
            reasons.append(("⚠️", f"VIX {vix:.1f} &mdash; elevated fear. Market participants pricing in meaningful uncertainty. "
                            "Bot macro cap remains 1.0× until VIX exceeds 28."))
        elif vix > 16:
            reasons.append(("📊", f"VIX {vix:.1f} &mdash; moderate concern (calm baseline is <15). "
                            "Some uncertainty priced in but not alarming."))
        else:
            reasons.append(("✅", f"VIX {vix:.1f} &mdash; low volatility. Markets calm; complacency risk if sustained."))

    # 5. Semiconductor / AI cluster (AMD, NVDA, AVGO, CRM)
    semis = {s: v for s, v in stocks.items() if s in ("AMD", "NVDA", "AVGO")}
    semis_down = [(s, v) for s, v in semis.items() if v < spy - 2.0]
    if len(semis_down) >= 2:
        names = ", ".join(f"{s} ({v:+.1f}%)" for s, v in semis_down)
        reasons.append(("🖥️", f"Semiconductor weakness: {names}. AI/chip stocks often move together "
                        "on macro repricing, earnings expectations, or export news."))

    # 6. Defensive stocks leading (WMT, JNJ, PG, ABBV, COST)
    defensives = {s: v for s, v in stocks.items() if s in ("WMT", "JNJ", "PG", "ABBV", "COST")}
    def_up = [(s, v) for s, v in defensives.items() if v > spy + 1.5]
    if len(def_up) >= 2:
        names = ", ".join(f"{s} ({v:+.1f}%)" for s, v in sorted(def_up, key=lambda x: -x[1])[:3])
        reasons.append(("🛡️", f"Defensive stocks leading: {names}. "
                        "Investors rotating into lower-risk, dividend-paying names &mdash; "
                        "classic 'risk-off within equities' signal."))

    return reasons[:5]  # cap at 5 reasons for readability


# ── HTML builders ─────────────────────────────────────────────────────────────

def _pct_span(v: Optional[float], color: str) -> str:
    if v is None:
        return f'<span style="color:{TEXT3};">&mdash;</span>'
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
            _unavail = f'<span style="color:{TEXT3}">Market data unavailable</span>'
            return (
                f'<div class="nt nt-wrap">'
                f'{_section("🌡", "Market Mood", "unavailable")}'
                f'{_card(_unavail)}'
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
        reasons           = _generate_reasons(spy, qqq, iwm, vix, sectors, stocks)

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

        # ── Why section ─────────────────────────────────────────────
        if reasons:
            reason_rows = "".join(
                f'<div style="display:flex;gap:10px;padding:5px 0;'
                f'border-bottom:1px solid {BORDER}22;align-items:flex-start;">'
                f'<span style="font-size:14px;flex-shrink:0;">{icon}</span>'
                f'<span style="font-size:{FONT_LABEL};color:{TEXT2};line-height:1.5;">{text}</span>'
                f'</div>'
                for icon, text in reasons
            )
            why_block = (
                f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid {BORDER};">'
                f'<div style="font-size:{FONT_LABEL};color:{TEXT3};text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:8px;">Why</div>'
                f'{reason_rows}'
                f'</div>'
            )
        else:
            why_block = ""

        content = header_row + indices_row + sector_block + outlier_block + why_block

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
