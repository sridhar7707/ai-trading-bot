"""Portfolio health scoring and sector-analysis helpers, extracted from recommendation_engine."""
from __future__ import annotations

import logging

from bot.core.error_logger import log_exception

_log = logging.getLogger("tradegenie.recommendation")

# ── Sector map (matches dashboard/app.py) ────────────────────────────────────
_SECTOR_MAP: dict[str, str] = {
    "NVDA": "Tech",    "MSFT": "Tech",    "AAPL": "Tech",  "GOOGL": "Tech",
    "META": "Tech",    "AMZN": "Consumer","TSLA": "Auto",  "AMD":   "Tech",
    "INTC": "Tech",    "QCOM": "Tech",    "MU":   "Tech",  "AVGO":  "Tech",
    "CRM":  "Tech",    "NOW":  "Tech",    "SNOW": "Tech",  "PLTR":  "Tech",
    "JPM":  "Finance", "BAC":  "Finance", "GS":   "Finance","MS":   "Finance",
    "XOM":  "Energy",  "CVX":  "Energy",  "SPY":  "Index", "QQQ":   "Index",
    "GLD":  "Commodity", "SLV": "Commodity", "APLD": "Tech",
}

_FALLBACK_HEALTH = {"total": 0, "grade": "—", "grade_label": "Unknown",
                    "biggest_risk": "Data unavailable", "strengths": [], "weaknesses": [],
                    "components": {}}


def _portfolio_val(d: dict) -> float:
    try:
        pv = d.get("portfolio", "—")
        return float(str(pv).replace("$", "").replace(",", "")) if pv != "—" else 0.0
    except Exception:
        return 0.0


def _cash_pct(d: dict, pv: float) -> float:
    try:
        open_pos = d.get("open_pos", {})
        prices   = d.get("prices", {})
        total_inv = sum(
            pos["shares"] * prices.get(sym, 0.0) if prices.get(sym) else pos["invested"]
            for sym, pos in open_pos.items()
        )
        return ((pv - total_inv) / pv * 100) if pv > 0 else 100.0
    except Exception:
        return 100.0


def _max_sector_conc(open_pos: dict, prices: dict, pv: float) -> float:
    if not open_pos or pv <= 0:
        return 0.0
    sector_vals: dict[str, float] = {}
    for sym, pos in open_pos.items():
        cur = prices.get(sym, 0.0)
        val = pos["shares"] * cur if cur > 0 else pos["invested"]
        s   = _SECTOR_MAP.get(sym.upper(), "Other")
        sector_vals[s] = sector_vals.get(s, 0.0) + val
    # Denominator is total portfolio value so uninvested cash dilutes concentration
    return max(v / pv * 100 for v in sector_vals.values()) if sector_vals else 0.0


def _worst_sector(open_pos: dict, prices: dict, pv: float) -> str:
    if not open_pos:
        return ""
    sector_vals: dict[str, float] = {}
    for sym, pos in open_pos.items():
        cur = prices.get(sym, 0.0)
        val = pos["shares"] * cur if cur > 0 else pos["invested"]
        s   = _SECTOR_MAP.get(sym.upper(), "Other")
        sector_vals[s] = sector_vals.get(s, 0.0) + val
    total = sum(sector_vals.values()) or 1.0
    return max(sector_vals, key=lambda s: sector_vals[s] / total) if sector_vals else ""


def get_portfolio_health(d: dict) -> dict:
    """Returns portfolio health score 0-100 with grade and component breakdown."""
    try:
        open_pos = d.get("open_pos", {})
        prices   = d.get("prices", {})
        df       = d.get("trades_df")
        vix      = d.get("vix", 0.0)
        pv       = _portfolio_val(d)

        # ── Risk score (25 pts) — from VIX ────────────────────────────────────
        if vix < 15:   risk_s = 25
        elif vix < 20: risk_s = 20
        elif vix < 25: risk_s = 12
        else:          risk_s =  5
        vix_lbl = f"VIX {vix:.0f}" if vix > 0 else "VIX unavailable"

        # ── Diversification score (25 pts) — max sector concentration ─────────
        sector_conc = _max_sector_conc(open_pos, prices, pv)
        worst_sector = _worst_sector(open_pos, prices, pv)
        if sector_conc < 20:   div_s = 25
        elif sector_conc < 30: div_s = 18
        elif sector_conc < 40: div_s = 10
        else:                  div_s =  3

        # ── Cash score (20 pts) ───────────────────────────────────────────────
        cash_pct = _cash_pct(d, pv)
        if cash_pct > 30:   cash_s = 20
        elif cash_pct > 20: cash_s = 15
        elif cash_pct > 10: cash_s =  8
        else:               cash_s =  2

        # ── Momentum score (15 pts) — avg ensemble score ──────────────────────
        avg_ens = 0.0
        scores: list[float] = []
        if df is not None and not df.empty and open_pos:
            buys = df[df["action"] == "BUY"]
            for sym in open_pos:
                sb = buys[buys["symbol"] == sym]
                if not sb.empty:
                    scores.append(float(sb.iloc[-1].get("ensemble_score", 0.0) or 0.0))
            avg_ens = sum(scores) / len(scores) if scores else 0.0
        if not scores:
            lbs = d.get("latest_buy_signal", {})
            if isinstance(lbs, dict) and lbs.get("ensemble_score"):
                avg_ens = float(lbs["ensemble_score"])
        if avg_ens > 0.75:   mom_s = 15
        elif avg_ens > 0.65: mom_s = 10
        elif avg_ens > 0.55: mom_s =  5
        else:                mom_s =  0

        # ── Quality score (15 pts) — win rate ─────────────────────────────────
        qual_s = 0
        win_rate = 0.0
        if df is not None and not df.empty:
            sells = df[df["action"].str.startswith("SELL") & (df["action"] != "SELL_RECONCILE")]
            if len(sells) > 0 and "pnl_pct" in sells.columns:
                win_rate = float((sells["pnl_pct"] > 0).sum() / len(sells))
        if win_rate > 0.60:   qual_s = 15
        elif win_rate > 0.50: qual_s = 10
        elif win_rate > 0.40: qual_s =  5
        else:                 qual_s =  0

        total = risk_s + div_s + cash_s + mom_s + qual_s

        # ── Grade ──────────────────────────────────────────────────────────────
        if total >= 90:   grade, grade_lbl = "A",  "Excellent"
        elif total >= 80: grade, grade_lbl = "B+", "Healthy"
        elif total >= 70: grade, grade_lbl = "B",  "Fair"
        elif total >= 60: grade, grade_lbl = "C",  "At Risk"
        else:             grade, grade_lbl = "D",  "Critical"

        # ── Biggest risk ───────────────────────────────────────────────────────
        weaknesses: list[str] = []
        strengths:  list[str] = []
        if sector_conc >= 35 and worst_sector:
            weaknesses.append(f"{worst_sector} concentration at {sector_conc:.0f}%")
        if vix >= 25:      weaknesses.append(f"Market volatility elevated (VIX {vix:.0f})")
        if cash_pct <= 10: weaknesses.append(f"Low cash reserve ({cash_pct:.0f}%)")
        if avg_ens < 0.60: weaknesses.append("AI conviction below target across positions")
        if win_rate < 0.45 and win_rate > 0: weaknesses.append(f"Win rate below 45% ({win_rate:.0%})")

        if cash_pct > 20:    strengths.append(f"Strong cash position ({cash_pct:.0f}%)")
        if avg_ens > 0.70:   strengths.append("High AI conviction across positions")
        if win_rate > 0.55:  strengths.append(f"Good trade quality ({win_rate:.0%} win rate)")
        if vix < 20:         strengths.append(f"Low market volatility (VIX {vix:.0f})")
        if sector_conc < 25: strengths.append("Well-diversified across sectors")

        biggest_risk = weaknesses[0] if weaknesses else "Portfolio appears balanced"

        return {"total": total, "grade": grade, "grade_label": grade_lbl,
                "biggest_risk": biggest_risk,
                "strengths": strengths[:3], "weaknesses": weaknesses[:3],
                "components": {
                    "risk":            {"score": risk_s,  "max": 25, "label": "Market Risk",      "detail": vix_lbl},
                    "diversification": {"score": div_s,   "max": 25, "label": "Diversification",  "detail": f"Max sector {sector_conc:.0f}%"},
                    "cash":            {"score": cash_s,  "max": 20, "label": "Cash Reserve",     "detail": f"Cash {cash_pct:.0f}%"},
                    "momentum":        {"score": mom_s,   "max": 15, "label": "AI Momentum",      "detail": f"Avg conviction {avg_ens*100:.0f}%"},
                    "quality":         {"score": qual_s,  "max": 15, "label": "Trade Quality",    "detail": f"Win rate {win_rate:.0%}"},
                }}

    except Exception as exc:
        log_exception(_log, "get_portfolio_health", exc)
        return dict(_FALLBACK_HEALTH)
