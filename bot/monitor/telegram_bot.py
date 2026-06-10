import requests
from datetime import datetime
from loguru import logger
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


def _send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug(f"[Telegram disabled] {text}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=5)
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


def alert_bot_started(mode: str, portfolio: float):
    _send(f"🟢 <b>Trading Day Started</b> — {mode.upper()}\n   Portfolio: ${portfolio:,.2f}")


def alert_buy(symbol: str, shares: float, price: float, regime: str, portfolio: float,
              vs_spy: float, notional: float = 0.0):
    pct          = f"+{vs_spy:.1f}%" if vs_spy >= 0 else f"{vs_spy:.1f}%"
    notional_str = f"   Notional: ${notional:,.2f}\n" if notional else ""
    _send(
        f"🟢 <b>BUY — {symbol}</b>\n"
        f"   Shares: {shares:.4f}  |  Price: ${price:.2f}\n"
        + notional_str +
        f"   Regime: {regime}\n"
        f"   Portfolio: ${portfolio:,.2f}\n"
        f"   vs S&amp;P 500 today: {pct}"
    )


def alert_sell(symbol: str, shares: float, price: float, pnl_pct: float,
               reason: str = "signal", notional: float = 0.0):
    emoji       = "🟢" if pnl_pct >= 0 else "🔴"
    pnl_dollars = f" (${notional * pnl_pct:+,.2f})" if notional else ""
    _send(
        f"{emoji} <b>SELL — {symbol}</b>\n"
        f"   Shares: {shares:.4f}  |  Price: ${price:.2f}\n"
        f"   P&amp;L: {pnl_pct:+.2%}{pnl_dollars}\n"
        f"   Reason: {reason}"
    )


def alert_hold(symbol: str, regime: str):
    _send(f"⚪ HOLD — {symbol} | Regime: {regime}")


def alert_stop_loss(symbol: str, pnl_pct: float, notional: float = 0.0):
    pnl_dollars = f" (${notional * pnl_pct:+,.2f})" if notional else ""
    _send(f"⚠️ <b>STOP-LOSS</b> triggered — {symbol}  P&amp;L: {pnl_pct:+.2%}{pnl_dollars}")


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


def alert_daily_summary(day_return: float, vs_spy: float, positions: list, cash: float, trades: int, day_trades: int):
    now = datetime.now()
    today = now.strftime(f"%B {now.day}, %Y")
    outperf = day_return - vs_spy
    trophy  = " 🏆" if outperf > 0 else ""
    _send(
        f"📊 <b>Daily P&amp;L Report — {today}</b>\n"
        f"   Day Return:       {day_return:+.2%}\n"
        f"   vs S&amp;P 500:       {vs_spy:+.2%}\n"
        f"   Outperformed:     {outperf:+.2%}{trophy}\n"
        f"   Open Positions:  {', '.join(positions) or 'None'}\n"
        f"   Cash Available:  ${cash:,.2f}\n"
        f"   Trades Today:    {trades}\n"
        f"   Day Trades Used: {day_trades}/3 (PDT)"
    )


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
