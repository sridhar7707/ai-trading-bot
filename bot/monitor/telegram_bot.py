import requests
from datetime import datetime, timezone, timedelta
from loguru import logger
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

_CDT = timezone(timedelta(hours=-5))


def _now_cdt() -> str:
    """Current time as 'HH:MM AM/PM CDT' for Telegram messages."""
    return datetime.now(_CDT).strftime("%-I:%M %p CDT")


def _send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug(f"[Telegram disabled] {text}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=5)
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


# Human-readable labels for SHAP feature drivers (mirrors dashboard/_WHY_MAP)
_WHY_LABELS: dict[str, str] = {
    "rsi":           "RSI momentum building",
    "rsi_15m":       "15-min RSI aligned",
    "macd_diff_pct": "MACD bullish crossover",
    "volume_ratio":  "Unusual buying volume",
    "mfi":           "Money Flow positive",
    "bb_width":      "Volatility expanding",
    "atr_pct":       "Volatility confirmed",
    "norm_close":    "Closing near day's high",
    "ema20_pct":     "Above 20-period EMA",
    "ema50_pct":     "Above 50-period EMA",
    "vwap_dev":      "Trading above VWAP",
    "hl_ratio":      "Strong intraday range",
    "stoch_k":       "Stochastic momentum",
}

_SELL_REASON_LABELS: dict[str, str] = {
    "signal":        "Signal exit",
    "stop-loss":     "Stop-loss hit",
    "trailing-stop": "Trailing stop hit",
    "drift-trim":    "Position drift trim",
    "time-exit":     "Time-based exit",
    "gap-down":      "Gap-down floor",
    "risk":          "Risk limit triggered",
    "manual":        "Manual exit",
}


def alert_bot_started(mode: str, portfolio: float):
    _send(f"🟢 <b>Trading Day Started</b> — {mode.upper()}  · {_now_cdt()}\n   Portfolio: ${portfolio:,.2f}")


def alert_buy(symbol: str, shares: float, price: float, regime: str, portfolio: float,
              vs_spy: float, notional: float = 0.0,
              xgb_prob: float = 0.0, lstm_prob: float = 0.0,
              sentiment_score: float = 0.0, ensemble_score: float = 0.0,
              drivers: list | None = None,
              sector: str = "", sector_pct_after: float = 0.0,
              cash_pct_after: float = 0.0):
    regime_label = regime.replace("_", " ").title()
    sent_str     = f"+{sentiment_score:.2f}" if sentiment_score >= 0 else f"{sentiment_score:.2f}"

    # Top 2 SHAP drivers in plain English
    why_parts: list[str] = []
    if drivers:
        try:
            pos_d = sorted(
                [(f, float(v)) for f, v in drivers if float(v) > 0],
                key=lambda x: -x[1],
            )[:2]
            for feat, _ in pos_d:
                why_parts.append(_WHY_LABELS.get(feat, feat))
        except Exception:
            pass
    why_str = " · ".join(why_parts) if why_parts else "Ensemble consensus"

    lines = [
        f"🟢 <b>BUY EXECUTED — {symbol}</b>  · {_now_cdt()}",
        f"   Entry: ${price:.2f} | Confidence: {ensemble_score * 100:.0f}%",
        f"   Models: XGBoost {xgb_prob * 100:.0f}% · LSTM {lstm_prob * 100:.0f}% · Sentiment {sent_str}",
        f"   Regime: {regime_label}",
        f"   Why: {why_str}",
    ]
    if sector and sector_pct_after > 0:
        lines.append(f"   Risk: {sector} now {sector_pct_after:.0f}% of portfolio")
    if cash_pct_after > 0:
        lines.append(f"   Cash remaining: {cash_pct_after:.0f}%")

    _send("\n".join(lines))


def alert_sell(symbol: str, shares: float, price: float, pnl_pct: float,
               reason: str = "signal", notional: float = 0.0,
               cash_freed_pct: float = 0.0):
    emoji        = "🟢" if pnl_pct >= 0 else "🔴"
    pnl_dollars  = f" (${notional * pnl_pct:+,.2f})" if notional else ""
    reason_label = _SELL_REASON_LABELS.get(reason, reason.replace("-", " ").title())

    lines = [
        f"{emoji} <b>SELL — {symbol}</b>  · {_now_cdt()}",
        f"   Exit: ${price:.2f} | P&amp;L: {pnl_pct:+.2%}{pnl_dollars}",
        f"   Reason: {reason_label}",
    ]
    if cash_freed_pct > 0:
        lines.append(f"   Freed to cash: {cash_freed_pct:.0f}% of portfolio")

    _send("\n".join(lines))


def alert_hold(symbol: str, regime: str):
    _send(f"⚪ HOLD — {symbol} | Regime: {regime}")


def alert_stop_loss(symbol: str, pnl_pct: float, notional: float = 0.0):
    pnl_dollars = f" (${notional * pnl_pct:+,.2f})" if notional else ""
    _send(f"⚠️ <b>STOP-LOSS</b> triggered — {symbol}  · {_now_cdt()}  P&amp;L: {pnl_pct:+.2%}{pnl_dollars}")


def alert_sell_failed(symbol: str, reason: str = "stop-loss"):
    _send(f"🚨 <b>SELL FAILED</b> — {symbol} | Reason: {reason} | Will retry next cycle")


def alert_daily_loss_limit(portfolio: float, pnl_pct: float):
    _send(f"🚨 <b>DAILY LOSS LIMIT HIT</b> — Bot halted\n   Portfolio: ${portfolio:,.2f}\n   Day P&amp;L: {pnl_pct:+.2%}")


def alert_risk_warning(portfolio: float, pnl_pct: float):
    _send(f"⚠️ <b>RISK WARNING</b> — 50% of daily loss limit reached\n   Portfolio: ${portfolio:,.2f}\n   Day P&amp;L: {pnl_pct:+.2%}")


def alert_weekly_loss_limit(portfolio: float, pnl_pct: float):
    _send(f"🚨 <b>WEEKLY LOSS LIMIT HIT</b> — No new buys until next week\n   Portfolio: ${portfolio:,.2f}\n   Week P&amp;L: {pnl_pct:+.2%}")


def alert_vix_halt():
    _send("🔴 <b>VIX EMERGENCY HALT</b> — VIX ≥ 40\n   No new buys this cycle. Existing positions still managed.")


def alert_daily_summary(
    day_return: float, vs_spy: float, positions: list, cash: float,
    trades: int, day_trades: int,
    portfolio_value: float = 0.0,
    best_trade: tuple | None = None,
    worst_trade: tuple | None = None,
    cash_pct: float = 0.0,
    health_score: int = 0,
):
    now     = datetime.now(_CDT)
    today   = now.strftime(f"%B {now.day}, %Y")
    outperf = day_return - vs_spy
    trophy  = " 🏆" if outperf > 0 else ""

    pv_str = f"${portfolio_value:,.2f}  " if portfolio_value > 0 else ""

    lines = [
        f"📊 <b>Daily Summary — {today}</b>",
        f"   Portfolio:       {pv_str}{day_return:+.2%} vs S&amp;P {vs_spy:+.2%}{trophy}",
        f"   Trades Today:    {trades}  ·  Day Trades: {day_trades}/3 (PDT)",
    ]

    if best_trade:
        sym, pct, usd = best_trade
        lines.append(f"   Best Trade:      {sym}  {pct:+.1%}  (${usd:+,.2f})")
    if worst_trade:
        sym, pct, usd = worst_trade
        lines.append(f"   Worst Trade:     {sym}  {pct:+.1%}  (${usd:+,.2f})")

    lines.append(f"   Open Positions:  {', '.join(positions) or 'None'}")

    if cash_pct > 0:
        lines.append(f"   Cash:            ${cash:,.2f}  ({cash_pct:.0f}% of portfolio)")
    else:
        lines.append(f"   Cash:            ${cash:,.2f}")

    if health_score > 0:
        h_icon = "🟢" if health_score >= 75 else ("🟡" if health_score >= 50 else "🔴")
        lines.append(f"   Portfolio Health: {h_icon} {health_score}/100")

    _send("\n".join(lines))


def alert_bot_offline():
    _send("🔴 <b>BOT OFFLINE</b> — Health check missed 2 consecutive pings")


def alert_confidence_passed():
    _send("🚀 <b>CONFIDENCE CHECK PASSED</b> — Bot is ready for real money!")


def alert_weekly_report(week_return: float, vs_spy: float, win_rate: float, sharpe: float, drawdown: float):
    _send(
        f"📈 <b>Weekly Performance Report</b>\n"
        f"   Week Return:  {week_return:+.2%}\n"
        f"   vs S&amp;P 500:   {vs_spy:+.2%}\n"
        f"   Win Rate:     {win_rate:.1%}\n"
        f"   Sharpe Ratio: {sharpe:.2f}\n"
        f"   Max Drawdown: {drawdown:.2%}"
    )
