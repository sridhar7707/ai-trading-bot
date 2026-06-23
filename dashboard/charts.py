"""Plotly chart render functions and chart-specific helpers."""
from __future__ import annotations

import datetime
import os
import time
from typing import Any

import pandas as pd
from loguru import logger

from dashboard.design_system import (
    BG, SURFACE, BORDER, TEXT1, TEXT2,
    PRIMARY, GAIN, LOSS, GAIN_BG, GAIN_BD, NEURAL,
    PLOTLY_LAYOUT, FONT_LABEL,
)
from dashboard.data import (
    get_data,
    _price_cache, _price_cache_time, _PRICE_CACHE_TTL,
)

# ── Feature-name display labels (XGB feature_importances_ key → readable text) ─
_FI_LABELS: dict = {
    "rsi": "RSI", "rsi_15m": "RSI 15m", "stoch_k": "Stoch %K",
    "macd_diff_pct": "MACD Cross", "volume_ratio": "Volume Ratio",
    "mfi": "Money Flow", "bb_width": "BB Width", "atr_pct": "Volatility",
    "norm_close": "Price Pos", "ema20_pct": "EMA20 Dev",
    "ema50_pct": "EMA50 Dev", "vwap_dev": "VWAP Dev", "hl_ratio": "H/L Range",
}


def render_equity_chart() -> Any:
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


def render_allocation_chart() -> Any:
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


def render_pnl_chart() -> Any:
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


def render_feature_importance_chart() -> Any:
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
