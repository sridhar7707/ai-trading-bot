"""
recommendation_engine.py — Single source of truth for all dashboard recommendations.

All five helpers accept a pre-fetched data dict `d` (output of get_data()) so they
can be called in a tight loop without redundant DB/network reads.  Render functions
call get_data() once at the top and pass it here.
"""
from __future__ import annotations

import json as _json
from typing import Any

from bot.core.error_logger import log_exception

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

# ── Feature → plain-English map (matches _WHY_MAP in app.py) ─────────────────
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

# ── Fallback dicts (returned on any error) ────────────────────────────────────
_FALLBACK_ACTION   = {"action": "HOLD", "confidence": 0, "reason": "Data unavailable",
                      "secondary_reasons": [], "urgency": "low"}
_FALLBACK_SIZING   = {"current_weight": 0.0, "target_weight": 0.0, "delta_weight": 0.0,
                      "delta_dollars": 0.0, "delta_shares": 0.0, "action": "hold",
                      "dollar_display": "—", "shares_display": "—", "reason": "Data unavailable"}
_FALLBACK_SELL     = {"sell_score": 0, "recommendation": "HOLD", "trim_amount_pct": 0,
                      "reasons_to_sell": [], "reasons_to_hold": [], "stop_loss_pct": 8.0}
_FALLBACK_EXPLAIN  = {"symbol": "", "action": "HOLD", "confidence": 0,
                      "bullish": [], "bearish": [], "model_breakdown": {}, "summary": "Data unavailable"}
_FALLBACK_HEALTH   = {"total": 0, "grade": "—", "grade_label": "Unknown",
                      "biggest_risk": "Data unavailable", "strengths": [], "weaknesses": [],
                      "components": {}}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER 1 — get_portfolio_action
# Answers: What should I buy or sell?
# ─────────────────────────────────────────────────────────────────────────────
def get_portfolio_action(symbol: str, d: dict) -> dict:
    """
    Returns priority-ordered action recommendation for `symbol`.
    Priority: EXIT > TRIM > SELL > WATCH > HOLD > ADD > BUY
    """
    try:
        open_pos = d.get("open_pos", {})
        prices   = d.get("prices", {})
        df       = d.get("trades_df")

        # ── Gather position metrics ────────────────────────────────────────────
        pv = _portfolio_val(d)
        pos = open_pos.get(symbol)
        cur_price = prices.get(symbol, 0.0)

        # Ensemble score and sentiment from latest BUY trade for this symbol
        ens, sent, regime, xgb, lstm = 0.65, 0.0, "", 0.65, 0.65
        prev_ens: list[float] = []
        if df is not None and not df.empty:
            sym_buys = df[(df["symbol"] == symbol) & (df["action"] == "BUY")]
            if not sym_buys.empty:
                lb = sym_buys.iloc[-1]
                ens  = float(lb.get("ensemble_score",  0.65) or 0.65)
                sent = float(lb.get("sentiment_score", 0.0)  or 0.0)
                xgb  = float(lb.get("xgb_prob",        0.65) or 0.65)
                lstm = float(lb.get("lstm_prob",        0.65) or 0.65)
                regime = str(lb.get("regime") or "").lower()
                if len(sym_buys) > 1:
                    prev_ens = [float(v) for v in sym_buys.iloc[:-1]["ensemble_score"].dropna()]

        if pos is None:
            # Symbol not held — check for BUY opportunity
            cash_pct = _cash_pct(d, pv)
            sector_conc = _max_sector_conc(open_pos, prices, pv)
            if ens >= 0.75 and cash_pct > 15 and sector_conc < 35:
                return {"action": "BUY", "confidence": int(ens * 100),
                        "reason": f"AI conviction {ens*100:.0f}% — new entry opportunity",
                        "secondary_reasons": [f"Cash available: {cash_pct:.0f}%"],
                        "urgency": "medium"}
            return dict(_FALLBACK_ACTION)

        # ── Calculate position metrics ─────────────────────────────────────────
        invested  = pos["invested"]
        cur_val   = pos["shares"] * cur_price if cur_price > 0 else invested
        unreal_pct = ((cur_val - invested) / invested * 100) if invested > 0 else 0.0
        pos_weight = (cur_val / pv * 100) if pv > 0 else 0.0
        cash_pct   = _cash_pct(d, pv)
        was_high   = bool(prev_ens and (sum(prev_ens) / len(prev_ens)) > 0.70)
        is_bear    = any(x in regime for x in ("bear", "trending down", "volatile"))

        reasons:  list[str] = []
        sec:      list[str] = []

        # ── EXIT conditions (any one) ──────────────────────────────────────────
        exit_triggers = []
        if unreal_pct < -8:      exit_triggers.append(f"Unrealised loss {unreal_pct:.1f}%")
        if ens < 0.45:           exit_triggers.append(f"AI conviction collapsed ({ens*100:.0f}%)")
        if pos_weight > 30:      exit_triggers.append(f"Position {pos_weight:.0f}% exceeds 30% cap")
        if exit_triggers:
            return {"action": "EXIT", "confidence": int(ens * 100),
                    "reason": exit_triggers[0],
                    "secondary_reasons": exit_triggers[1:],
                    "urgency": "high"}

        # ── TRIM conditions (any one) ─────────────────────────────────────────
        trim_triggers = []
        if pos_weight > 20:      trim_triggers.append(f"Position {pos_weight:.0f}% exceeds 20% target")
        if unreal_pct > 50:      trim_triggers.append(f"Unrealised gain {unreal_pct:.0f}% — protect profits")
        if ens < 0.60 and was_high: trim_triggers.append("AI confidence declining from previous high")
        if trim_triggers:
            return {"action": "TRIM", "confidence": int(ens * 100),
                    "reason": trim_triggers[0],
                    "secondary_reasons": trim_triggers[1:],
                    "urgency": "medium"}

        # ── SELL conditions (ALL) ─────────────────────────────────────────────
        if ens < 0.55 and sent < -0.05 and is_bear:
            return {"action": "SELL", "confidence": int(ens * 100),
                    "reason": "All sell conditions met — conviction, sentiment, regime all negative",
                    "secondary_reasons": [f"Ensemble {ens*100:.0f}%", "Bearish regime", "Negative sentiment"],
                    "urgency": "high"}

        # ── WATCH ──────────────────────────────────────────────────────────────
        if 0.55 <= ens < 0.65:
            return {"action": "WATCH", "confidence": int(ens * 100),
                    "reason": f"AI signal mixed ({ens*100:.0f}%) — no clear direction",
                    "secondary_reasons": [],
                    "urgency": "low"}

        # ── ADD (more of current position) ────────────────────────────────────
        if ens >= 0.75 and pos_weight < 10 and cash_pct > 15:
            sector_conc = _max_sector_conc(open_pos, prices, pv)
            if sector_conc < 35:
                return {"action": "ADD", "confidence": int(ens * 100),
                        "reason": f"High conviction ({ens*100:.0f}%) and underweight at {pos_weight:.0f}%",
                        "secondary_reasons": [f"Cash: {cash_pct:.0f}%"],
                        "urgency": "medium"}

        # ── HOLD ──────────────────────────────────────────────────────────────
        return {"action": "HOLD", "confidence": int(ens * 100),
                "reason": f"Position healthy — conviction {ens*100:.0f}%, size {pos_weight:.0f}%",
                "secondary_reasons": [],
                "urgency": "low"}

    except Exception as exc:
        log_exception("get_portfolio_action", exc)
        return dict(_FALLBACK_ACTION)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER 2 — get_position_sizing
# Answers: How much should I buy?
# ─────────────────────────────────────────────────────────────────────────────
def get_position_sizing(symbol: str, d: dict) -> dict:
    """Returns target weight, delta $ and shares for symbol."""
    try:
        open_pos  = d.get("open_pos", {})
        prices    = d.get("prices", {})
        df        = d.get("trades_df")
        pv        = _portfolio_val(d)

        cur_price = prices.get(symbol, 0.0)
        pos       = open_pos.get(symbol)

        ens = 0.65
        if df is not None and not df.empty:
            sym_buys = df[(df["symbol"] == symbol) & (df["action"] == "BUY")]
            if not sym_buys.empty:
                ens = float(sym_buys.iloc[-1].get("ensemble_score", 0.65) or 0.65)

        # Target weight from conviction
        if ens >= 0.80:   target_w, ens_lbl = 15.0, "≥80% conviction"
        elif ens >= 0.70: target_w, ens_lbl = 12.0, "≥70% conviction"
        elif ens >= 0.65: target_w, ens_lbl = 10.0, "≥65% conviction"
        elif ens >= 0.55: target_w, ens_lbl =  5.0, "≥55% conviction"
        else:             target_w, ens_lbl =  0.0, "<55% — exit signal"

        # Cap at 25% single position
        target_w = min(target_w, 25.0)

        # Current position weight
        if pos and cur_price > 0 and pv > 0:
            cur_val = pos["shares"] * cur_price
            cur_w   = cur_val / pv * 100
        elif pos and pv > 0:
            cur_w   = pos["invested"] / pv * 100
        else:
            cur_w = 0.0

        delta_w     = target_w - cur_w
        delta_dol   = delta_w / 100 * pv if pv > 0 else 0.0
        delta_shares = abs(delta_dol) / cur_price if cur_price > 0 else 0.0

        if delta_w > 0.5:    action, reason = "add",    f"Underweight by {delta_w:.1f}% — {ens_lbl}"
        elif delta_w < -0.5: action, reason = "reduce",  f"Overweight by {abs(delta_w):.1f}%"
        else:                action, reason = "hold",    "Position is on target"

        sign = "+" if delta_dol >= 0 else "-"
        dol_disp = f"{sign}${abs(delta_dol):,.0f}" if delta_dol != 0 else "—"
        sh_disp  = f"~{delta_shares:.0f} shares" if delta_shares > 0.5 else "—"

        return {"current_weight": round(cur_w, 1),
                "target_weight":  round(target_w, 1),
                "delta_weight":   round(delta_w, 1),
                "delta_dollars":  round(delta_dol, 2),
                "delta_shares":   round(delta_shares, 1),
                "action":         action,
                "dollar_display": dol_disp,
                "shares_display": sh_disp,
                "current_price":  cur_price,
                "reason":         reason}

    except Exception as exc:
        log_exception("get_position_sizing", exc)
        return dict(_FALLBACK_SIZING)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER 3 — get_sell_analysis
# Answers: When should I sell?
# ─────────────────────────────────────────────────────────────────────────────
def get_sell_analysis(symbol: str, d: dict) -> dict:
    """Returns sell score 0-100 and recommendation for symbol."""
    try:
        open_pos  = d.get("open_pos", {})
        prices    = d.get("prices", {})
        df        = d.get("trades_df")
        pv        = _portfolio_val(d)

        pos       = open_pos.get(symbol)
        cur_price = prices.get(symbol, 0.0)

        ens, sent, regime = 0.65, 0.0, ""
        if df is not None and not df.empty:
            sym_buys = df[(df["symbol"] == symbol) & (df["action"] == "BUY")]
            if not sym_buys.empty:
                lb   = sym_buys.iloc[-1]
                ens  = float(lb.get("ensemble_score",  0.65) or 0.65)
                sent = float(lb.get("sentiment_score", 0.0)  or 0.0)
                regime = str(lb.get("regime") or "").lower()

        cur_val   = (pos["shares"] * cur_price) if (pos and cur_price > 0) else (pos["invested"] if pos else 0.0)
        invested  = pos["invested"] if pos else 0.0
        unreal_pct = ((cur_val - invested) / invested * 100) if invested > 0 else 0.0
        pos_w      = (cur_val / pv * 100) if pv > 0 else 0.0

        # ── Scoring ───────────────────────────────────────────────────────────
        size_pts, size_reasons = 0, []
        if pos_w > 25:   size_pts, size_reasons = 30, ["Position >25% is extreme concentration"]
        elif pos_w > 15: size_pts, size_reasons = 20, [f"Position {pos_w:.0f}% exceeds 15% guideline"]
        elif pos_w > 10: size_pts, size_reasons = 10, [f"Position {pos_w:.0f}% above 10% target"]

        prof_pts, prof_reasons = 0, []
        if unreal_pct > 50:   prof_pts, prof_reasons = 25, [f"Unrealised gain {unreal_pct:.0f}% — consider locking in"]
        elif unreal_pct > 25: prof_pts, prof_reasons = 15, [f"Unrealised gain {unreal_pct:.0f}%"]
        elif unreal_pct > 10: prof_pts, prof_reasons =  8, [f"Unrealised gain {unreal_pct:.0f}%"]

        conv_pts, conv_reasons = 0, []
        if ens < 0.50:   conv_pts, conv_reasons = 25, [f"AI conviction very low ({ens*100:.0f}%)"]
        elif ens < 0.60: conv_pts, conv_reasons = 15, [f"AI conviction weakening ({ens*100:.0f}%)"]
        elif ens < 0.65: conv_pts, conv_reasons =  8, [f"AI conviction below target ({ens*100:.0f}%)"]

        dd_pts, dd_reasons = 0, []
        if unreal_pct < -10:  dd_pts, dd_reasons = 20, [f"Loss {unreal_pct:.0f}% exceeds 10% stop"]
        elif unreal_pct < -7: dd_pts, dd_reasons = 15, [f"Loss {unreal_pct:.0f}% approaching 8% stop"]
        elif unreal_pct < -5: dd_pts, dd_reasons =  8, [f"Loss {unreal_pct:.0f}% — drawdown watch"]

        score = size_pts + prof_pts + conv_pts + dd_pts
        reasons_to_sell = size_reasons + prof_reasons + conv_reasons + dd_reasons

        if score <= 25:   rec = "HOLD"
        elif score <= 45: rec = "WATCH"
        elif score <= 65: rec = "TRIM"
        elif score <= 80: rec = "SELL"
        else:             rec = "EXIT"

        trim_pct = 0
        if rec == "TRIM":  trim_pct = 25 if pos_w > 20 else 15
        elif rec == "SELL": trim_pct = 75
        elif rec == "EXIT": trim_pct = 100

        # Reasons to hold
        reasons_to_hold = []
        if ens >= 0.70:      reasons_to_hold.append(f"Strong AI conviction ({ens*100:.0f}%)")
        if sent > 0.05:      reasons_to_hold.append("Positive news sentiment")
        if unreal_pct > 0:   reasons_to_hold.append(f"Position up {unreal_pct:.0f}%")
        if "bull" in regime or "trending up" in regime:
            reasons_to_hold.append("Bull market regime")

        return {"sell_score": score,
                "recommendation": rec,
                "trim_amount_pct": trim_pct,
                "reasons_to_sell": reasons_to_sell,
                "reasons_to_hold": reasons_to_hold,
                "stop_loss_pct": 8.0,
                "unrealised_pct": round(unreal_pct, 1),
                "position_weight": round(pos_w, 1),
                "ensemble_score": ens}

    except Exception as exc:
        log_exception("get_sell_analysis", exc)
        return dict(_FALLBACK_SELL)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER 4 — get_recommendation_explanation
# Answers: Why?
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendation_explanation(symbol: str, d: dict) -> dict:
    """Returns bullish/bearish factors and model breakdown for symbol."""
    try:
        df    = d.get("trades_df")
        vix   = d.get("vix", 0.0)

        action_data = get_portfolio_action(symbol, d)
        action      = action_data.get("action", "HOLD")
        confidence  = action_data.get("confidence", 0)

        ens, xgb, lstm, sent, regime, drv_raw = 0.65, 0.65, 0.65, 0.0, "", None
        if df is not None and not df.empty:
            sym_buys = df[(df["symbol"] == symbol) & (df["action"] == "BUY")]
            if not sym_buys.empty:
                lb      = sym_buys.iloc[-1]
                ens     = float(lb.get("ensemble_score",  0.65) or 0.65)
                xgb     = float(lb.get("xgb_prob",        0.65) or 0.65)
                lstm    = float(lb.get("lstm_prob",        0.65) or 0.65)
                sent    = float(lb.get("sentiment_score", 0.0)  or 0.0)
                regime  = str(lb.get("regime") or "").replace("_", " ").title()
                drv_raw = lb.get("feature_drivers")

        bullish:  list[str] = []
        bearish:  list[str] = []

        # SHAP-driven factors
        try:
            ds = _json.loads(drv_raw) if isinstance(drv_raw, str) else (drv_raw or [])
            for feat, val in (ds or []):
                w    = _WHY_MAP.get(feat)
                name = w[0] if w else feat.replace("_", " ").title()
                if float(val) > 0:
                    bullish.append(name)
                else:
                    bearish.append(name)
        except Exception:
            pass

        # Model-level factors
        if xgb >= 0.70:  bullish.append(f"XGBoost signal strong ({xgb*100:.0f}%)")
        elif xgb < 0.55: bearish.append(f"XGBoost signal weak ({xgb*100:.0f}%)")
        if lstm >= 0.70:  bullish.append(f"LSTM momentum confirmed ({lstm*100:.0f}%)")
        elif lstm < 0.55: bearish.append(f"LSTM momentum weak ({lstm*100:.0f}%)")
        if sent > 0.05:   bullish.append("Positive news sentiment")
        elif sent < -0.05: bearish.append("Negative news sentiment")

        # Regime
        r_lower = regime.lower()
        if "bull" in r_lower or "trending up" in r_lower:
            bullish.append(f"Bull market regime ({regime})")
        elif "bear" in r_lower or "trending down" in r_lower:
            bearish.append(f"Bear market regime ({regime})")

        # VIX
        if vix > 30: bearish.append(f"VIX elevated at {vix:.0f} (fear)")
        elif vix < 15: bullish.append(f"VIX low at {vix:.0f} (calm market)")

        # Sizing/position factors from action
        for r in action_data.get("secondary_reasons", []):
            if action in ("TRIM", "SELL", "EXIT"):
                bearish.append(r)

        # Deduplicate, cap at 5 each
        bullish = list(dict.fromkeys(bullish))[:5]
        bearish = list(dict.fromkeys(bearish))[:5]

        # Summary sentence
        if action in ("EXIT", "SELL"):
            summary = f"Multiple sell signals active — consider exiting {symbol}."
        elif action == "TRIM":
            summary = f"Strong AI conviction but position needs trimming for risk management."
        elif action in ("ADD", "BUY"):
            summary = f"High conviction entry — AI signals aligned for {symbol}."
        elif action == "WATCH":
            summary = f"Mixed signals on {symbol} — monitor closely before acting."
        else:
            summary = f"{symbol} on track — no action needed at current levels."

        return {"symbol": symbol, "action": action, "confidence": confidence,
                "bullish": bullish, "bearish": bearish,
                "model_breakdown": {"xgboost": round(xgb, 2), "lstm": round(lstm, 2),
                                    "sentiment": round(sent, 2), "regime": regime,
                                    "risk": "high" if vix > 30 else ("medium" if vix > 20 else "low")},
                "summary": summary}

    except Exception as exc:
        log_exception("get_recommendation_explanation", exc)
        fb = dict(_FALLBACK_EXPLAIN)
        fb["symbol"] = symbol
        return fb


# ─────────────────────────────────────────────────────────────────────────────
# HELPER 5 — get_portfolio_health
# Answers: What is my biggest risk?
# ─────────────────────────────────────────────────────────────────────────────
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
        if df is not None and not df.empty and open_pos:
            buys = df[df["action"] == "BUY"]
            scores = []
            for sym in open_pos:
                sb = buys[buys["symbol"] == sym]
                if not sb.empty:
                    scores.append(float(sb.iloc[-1].get("ensemble_score", 0.0) or 0.0))
            avg_ens = sum(scores) / len(scores) if scores else 0.0
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
        log_exception("get_portfolio_health", exc)
        return dict(_FALLBACK_HEALTH)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────
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
    total = sum(sector_vals.values()) or 1.0
    return max(v / total * 100 for v in sector_vals.values()) if sector_vals else 0.0


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
