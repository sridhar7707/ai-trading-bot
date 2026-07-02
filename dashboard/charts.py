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
    # Momentum oscillators
    "rsi":            "RSI",
    "rsi_15m":        "RSI 15m",
    "stoch_k":        "Stoch %K",
    "macd_diff_pct":  "MACD Cross",
    "mfi":            "Money Flow",
    # Volume
    "volume_ratio":   "Volume Ratio",
    "obv_chg_pct":    "OBV Flow",
    "vol_ratio_trend":"Volume Trend",
    # Volatility / range
    "atr_pct":        "Volatility",
    "bb_width":       "BB Width",
    "bb_position":    "BB Position",
    "hl_ratio":       "H/L Range",
    # Price vs moving averages
    "norm_close":     "Price Position",
    "ema20_pct":      "EMA20 Dev",
    "ema50_pct":      "EMA50 Dev",
    "sma20_pct":      "SMA20 Dev",
    "ema_spread":     "EMA Trend Spread",
    "vwap_dev":       "VWAP Dev",
    # Multi-period momentum (Jegadeesh-Titman / AQR)
    "ret_5d":         "1-Week Return",
    "ret_21d":        "1-Month Return",
    "ret_63d":        "3-Month Return",
    "ret_126d":       "6-Month Return",
    "mom_12_1":       "12-1 Month Momentum",
    "high_52w_pct":   "vs 52-Week High",
    # Other
    "returns":        "1-Bar Return",
}


def render_equity_chart() -> Any:
    try:
        import plotly.graph_objects as go
        from dashboard.data import get_db_conn
        import os, sqlite3 as _sq3
        df  = get_data()["trades_df"]
        fig = go.Figure()

        has_data = (not df.empty and "portfolio_value" in df.columns
                    and df["portfolio_value"].notna().any()
                    and (df["portfolio_value"].fillna(0) > 0).any())

        # Fallback: use portfolio_snapshots if trades.portfolio_value is all-null/zero
        _snap_daily = None
        if not has_data:
            try:
                from dashboard.data import DB_PATH
                if os.path.exists(DB_PATH):
                    with get_db_conn() as _con:
                        _rows = _con.execute(
                            "SELECT date(timestamp) AS d, AVG(portfolio_value) AS v "
                            "FROM portfolio_snapshots WHERE portfolio_value > 0 "
                            "GROUP BY d ORDER BY d"
                        ).fetchall()
                    if _rows:
                        _snap_daily = pd.DataFrame(_rows, columns=["date", "value"])
                        _snap_daily["date"] = pd.to_datetime(_snap_daily["date"])
                        has_data = True
            except Exception:
                pass

        if not has_data:
            fig.add_annotation(
                text="Building history &mdash; bot trades 9:30am-4pm ET, Mon-Fri. Chart appears after the first trading day.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=12))
        else:
            if _snap_daily is not None:
                daily = _snap_daily
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
                _spy_hist = _get_sym_hist("SPY")
                if _spy_hist is not None and not _spy_hist.empty and "Close" in _spy_hist.columns:
                    _spy_c = _spy_hist["Close"].dropna()
                    _s0    = daily["date"].min().strftime("%Y-%m-%d")
                    _spy_c = _spy_c[[str(d)[:10] >= _s0 for d in _spy_c.index]]
                    if not _spy_c.empty:
                        _spy_norm = _spy_c / float(_spy_c.iloc[0]) * float(daily["value"].iloc[0])
                        fig.add_trace(go.Scatter(
                            x=_spy_c.index, y=_spy_norm,
                            line=dict(color=TEXT2, width=1.5, dash="dot"),
                            mode="lines", opacity=0.6,
                            hovertemplate="<b>SPY %{x|%b %d}</b><br>$%{y:,.2f}<extra></extra>",
                            name="SPY (scaled)",
                        ))

        fig.update_layout(
            title=dict(text="Portfolio Value Over Time  <span style='font-size:11px;'>&mdash; end-of-day snapshots, includes cash + open positions</span>",
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
            title=dict(text="Daily Realized P&L  <span style='font-size:11px;'>&mdash; profit/loss from SELL trades only (unrealized not included)</span>",
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
        return f'<span style="color:{TEXT2};">&mdash;</span>'
    try:
        prices = [float(p) for p in hist["Close"].iloc[-30:]]
    except Exception as exc:
        logger.debug(f"_sparkline prices: {exc}")
        return f'<span style="color:{TEXT2};">&mdash;</span>'
    if not prices:
        return f'<span style="color:{TEXT2};">&mdash;</span>'

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


def render_returns_histogram() -> Any:
    try:
        import plotly.graph_objects as go
        df    = get_data()["trades_df"]
        fig   = go.Figure()
        sells = (df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")].copy()
                 if not df.empty else pd.DataFrame())
        if sells.empty or "pnl_pct" not in sells.columns or sells["pnl_pct"].isna().all():
            fig.add_annotation(
                text="Return distribution appears here after the first closed trade.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=12))
        else:
            pnl = sells["pnl_pct"].dropna() * 100
            fig.add_trace(go.Histogram(
                x=pnl, nbinsx=20,
                marker_color=PRIMARY, marker_line=dict(width=0),
                opacity=0.8,
                hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
                name="Return %",
            ))
            fig.add_vline(x=0, line_width=1, line_color=TEXT2, opacity=0.5)
        fig.update_layout(
            title=dict(text="Return Distribution  <span style='font-size:11px;'>&mdash; each bar = a closed trade</span>",
                       font=dict(color=TEXT1, size=13), x=0.01),
            xaxis=dict(title="Return %", **PLOTLY_LAYOUT["xaxis"], tickfont=dict(color=TEXT2)),
            yaxis=dict(title="Count", **PLOTLY_LAYOUT["yaxis"], tickfont=dict(color=TEXT2)),
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            height=280,
        )
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                             if k not in ("xaxis", "yaxis")}, height=280)
        fig.add_annotation(text=f"Chart error: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=LOSS))
        return fig


def render_winloss_chart() -> Any:
    try:
        import plotly.graph_objects as go
        df    = get_data()["trades_df"]
        fig   = go.Figure()
        sells = (df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")].copy()
                 if not df.empty else pd.DataFrame())
        if sells.empty or "pnl_pct" not in sells.columns or sells["pnl_pct"].isna().all():
            fig.add_annotation(
                text="Win/loss breakdown appears here after the first closed trade.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=TEXT2, size=12))
        else:
            wins   = int((sells["pnl_pct"] > 0).sum())
            losses = int((sells["pnl_pct"] <= 0).sum())
            fig.add_trace(go.Bar(
                x=["Winning Trades", "Losing Trades"],
                y=[wins, losses],
                marker_color=[GAIN, LOSS], marker_line=dict(width=0),
                text=[str(wins), str(losses)],
                textposition="outside",
                textfont=dict(color=TEXT1, size=13),
                hovertemplate="%{x}: %{y}<extra></extra>",
                name="Trades",
            ))
        fig.update_layout(
            title=dict(text="Win / Loss Breakdown",
                       font=dict(color=TEXT1, size=13), x=0.01),
            xaxis=dict(title="", **PLOTLY_LAYOUT["xaxis"], tickfont=dict(color=TEXT1)),
            yaxis=dict(title="Trade Count", **PLOTLY_LAYOUT["yaxis"], tickfont=dict(color=TEXT2)),
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            height=280,
        )
        return fig
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                             if k not in ("xaxis", "yaxis")}, height=280)
        fig.add_annotation(text=f"Chart error: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=LOSS))
        return fig


def render_feature_importance_chart() -> Any:
    try:
        import json as _json
        import plotly.graph_objects as go
        fig = go.Figure()
        fi_path = "models/feature_importance.json"
        if not os.path.exists(fi_path):
            fig.add_annotation(
                text="Feature importance not yet available &mdash; run scripts/train_model.py first, then push models to HF.",
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
                text="Which signals drive the AI's BUY decisions  <span style='font-size:{FONT_LABEL};'>&mdash; longer bar = more influence on each trade</span>",
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
