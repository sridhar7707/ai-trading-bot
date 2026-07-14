"""News feed &mdash; Yahoo Finance headlines for held positions and BUY-signal watchlist."""
from __future__ import annotations
import datetime
from loguru import logger
from dashboard.design_system import (
    SURFACE, BORDER, TEXT1, TEXT2, TEXT3,
    PRIMARY, FONT_VALUE, FONT_LABEL,
    WEIGHT_BOLD, WEIGHT_MEDIUM,
    _section, _card, _empty_state, _symbol,
)
from dashboard.data import get_data, safe_query
from bot.core.error_logger import safe_render, timed

_logger = logger

_news_cache: dict[str, tuple[float, list]] = {}  # symbol → (fetched_at, items)
_NEWS_CACHE_TTL = 1800.0  # 30 minutes


def _time_ago(pub_date_str: str) -> str:
    try:
        import time as _t
        from datetime import timezone
        dt = datetime.datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        secs = _t.time() - dt.timestamp()
        if secs < 3600:
            return f"{int(secs // 60)}m ago"
        if secs < 86400:
            return f"{int(secs // 3600)}h ago"
        return f"{int(secs // 86400)}d ago"
    except Exception:
        return ""


def _fetch_symbol_news(symbol: str, max_items: int = 3) -> list:
    import time as _t
    now = _t.time()
    cached_ts, cached = _news_cache.get(symbol, (0.0, []))
    if cached and now - cached_ts < _NEWS_CACHE_TTL:
        return cached
    try:
        import yfinance as _yf
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

        def _get_news():
            return _yf.Ticker(symbol).news or []

        with ThreadPoolExecutor(max_workers=1) as _pool:
            _fut = _pool.submit(_get_news)
            try:
                raw = _fut.result(timeout=10)
            except _FutTimeout:
                logger.debug(f"news_fetch {symbol}: yfinance timed out — using DB cache")
                return _db_news_fallback(symbol, max_items, now)

        items = []
        for r in raw[:max_items]:
            c = r.get("content", {})
            url = (c.get("clickThroughUrl") or {}).get("url", "")
            items.append({
                "symbol":    symbol,
                "title":     c.get("title", ""),
                "publisher": (c.get("provider") or {}).get("displayName", ""),
                "pub_date":  c.get("pubDate", ""),
                "url":       url,
            })
        _news_cache[symbol] = (now, items)
        return items
    except Exception as exc:
        logger.debug(f"news_fetch {symbol}: {exc}")
        return _db_news_fallback(symbol, max_items, now)


def _db_news_fallback(symbol: str, max_items: int, now: float) -> list:
    """Fall back to the SQLite news_cache table when yfinance is unavailable."""
    try:
        rows = safe_query(
            "SELECT headlines_json, fetch_date FROM news_cache WHERE symbol = ? "
            "ORDER BY cached_at DESC LIMIT 1",
            (symbol,), default=[],
        )
        if not rows or not rows[0][0]:
            _news_cache[symbol] = (now, [])
            return []
        import json
        headlines = json.loads(rows[0][0])[:max_items]
        fetch_date = rows[0][1]
        items = [
            {
                "symbol":    symbol,
                "title":     h,
                "publisher": "NewsAPI",
                "pub_date":  f"{fetch_date}T12:00:00Z",
                "url":       "",
            }
            for h in headlines if h
        ]
        _news_cache[symbol] = (now, items)
        return items
    except Exception:
        _news_cache[symbol] = (now, [])
        return []


def _news_block(title: str, icon: str, items: list) -> str:
    """Render a labelled list of news items."""
    if not items:
        return ""
    rows_html = ""
    for item in items:
        sym       = item["symbol"]
        headline  = item["title"]
        publisher = item["publisher"]
        pub_date  = item["pub_date"]
        url       = item["url"]
        age       = _time_ago(pub_date)

        title_el = (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'style="color:{TEXT1};text-decoration:none;font-size:{FONT_VALUE};'
            f'line-height:1.4;font-weight:{WEIGHT_MEDIUM};"'
            f'onmouseover="this.style.color=\'{PRIMARY}\'" '
            f'onmouseout="this.style.color=\'{TEXT1}\'">'
            f'{headline}'
            f'</a>'
        ) if url else f'<span style="font-size:{FONT_VALUE};color:{TEXT1};">{headline}</span>'

        rows_html += (
            f'<div style="display:grid;grid-template-columns:52px 1fr;gap:10px;'
            f'padding:10px 0;border-bottom:1px solid {BORDER};">'
            f'<div style="padding-top:2px;">{_symbol(sym)}</div>'
            f'<div>'
            f'{title_el}'
            f'<div style="margin-top:4px;font-size:{FONT_LABEL};color:{TEXT3};">'
            f'{publisher}'
            + (f' &nbsp;&middot;&nbsp; {age}' if age else "")
            + f'</div>'
            f'</div>'
            f'</div>'
        )

    return (
        f'<div style="margin-bottom:20px;">'
        f'<div style="font-size:{FONT_VALUE};font-weight:{WEIGHT_BOLD};color:{TEXT2};'
        f'padding:10px 0 4px;letter-spacing:.5px;text-transform:uppercase;">'
        f'{icon} {title}</div>'
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:0 16px;">'
        f'{rows_html}'
        f'</div>'
        f'</div>'
    )


@timed(_logger)
@safe_render("News Feed")
def render_news_feed() -> str:
    d = get_data()
    open_syms = list(d.get("open_pos", {}).keys())

    today_str = datetime.date.today().isoformat()
    buy_signals = safe_query(
        "SELECT DISTINCT symbol FROM recommendations "
        "WHERE recommendation = 'BUY' AND prediction_date = ?",
        (today_str,), default=[]
    )
    executed_today = safe_query(
        "SELECT DISTINCT symbol FROM trades WHERE action='BUY' AND date(timestamp)=?",
        (today_str,), default=[]
    )
    executed_syms = {r[0] for r in (executed_today or [])}
    watchlist_syms = [r[0] for r in (buy_signals or []) if r[0] not in executed_syms and r[0] not in open_syms]

    held_news: list = []
    for sym in open_syms[:5]:
        held_news.extend(_fetch_symbol_news(sym, max_items=2))
    held_news.sort(key=lambda x: x["pub_date"], reverse=True)
    held_news = held_news[:5]

    radar_news: list = []
    for sym in watchlist_syms[:5]:
        radar_news.extend(_fetch_symbol_news(sym, max_items=2))
    radar_news.sort(key=lambda x: x["pub_date"], reverse=True)
    radar_news = radar_news[:5]

    if not held_news and not radar_news:
        return (
            f'<div class="nt nt-wrap">'
            f'{_section("📰", "Market News", "")}'
            f'{_card(_empty_state("📰", "No news available", "Yahoo Finance returned no recent articles for your positions."))}'
            f'</div>'
        )

    held_block   = _news_block("Stocks You Hold", "📌", held_news)
    radar_block  = _news_block("On the Radar (BUY signals not yet bought)", "🔭", radar_news)
    note = "via Yahoo Finance · free · updates every 30 min"

    return (
        f'<div class="nt nt-wrap">'
        f'{_section("📰", "Market News", note)}'
        f'{held_block}'
        f'{radar_block}'
        f'</div>'
    )
