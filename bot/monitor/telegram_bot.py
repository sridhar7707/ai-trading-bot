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


def alert_buy(symbol: str, shares: float, price: float, regime: str, portfolio: float, vs_spy: float):
    pct = f"+{vs_spy:.1f}%" if vs_spy >= 0 else f"{vs_spy:.1f}%"
    _send(
        f"🟢 <b>BUY — {symbol}</b>\n"
        f"   Shares: {shares:.4f} (fractional)\n"
        f"   Price: ${price:.2f}\n"
        f"   Regime: {regime}\n"
        f"   Portfolio: ${portfolio:.2f}\n"
        f"   vs S&P 500 today: {pct}"
    )


def alert_sell(symbol: str, shares: float, price: float, pnl_pct: float, reason: str = "signal"):
    emoji = "🟢" if pnl_pct >= 0 else "🔴"
    _send(
        f"{emoji} <b>SELL — {symbol}</b>\n"
        f"   Shares: {shares:.4f}\n"
        f"   Price: ${price:.2f}\n"
        f"   P&L: {pnl_pct:+.2%}\n"
        f"   Reason: {reason}"
    )


def alert_hold(symbol: str, regime: str):
    _send(f"⚪ HOLD — {symbol} | Regime: {regime}")


def alert_stop_loss(symbol: str, pnl_pct: float):
    _send(f"⚠️ <b>STOP-LOSS</b> triggered — {symbol} ({pnl_pct:+.2%})")


def alert_daily_loss_limit(portfolio: float, pnl_pct: float):
    _send(f"🚨 <b>DAILY LOSS LIMIT HIT</b> — Bot halted\n   Portfolio: ${portfolio:.2f}\n   Day P&L: {pnl_pct:+.2%}")


def alert_daily_summary(day_return: float, vs_spy: float, positions: list, cash: float, trades: int, day_trades: int):
    now = datetime.now()
    today = now.strftime(f"%B {now.day}, %Y")  # platform-safe (no %-d)
    outperf = day_return - vs_spy
    trophy = " 🏆" if outperf > 0 else ""
    _send(
        f"📊 <b>Daily P&L Report — {today}</b>\n"
        f"   Day Return:      {day_return:+.2%} ✅\n"
        f"   vs S&P 500:      {vs_spy:+.2%}\n"
        f"   Outperformed:    {outperf:+.2%}{trophy}\n"
        f"   Open Positions: {', '.join(positions) or 'None'}\n"
        f"   Cash Available: ${cash:.2f}\n"
        f"   Trades Today:   {trades}\n"
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
        f"   vs S&P 500:   {vs_spy:+.2%}\n"
        f"   Win Rate:     {win_rate:.1%}\n"
        f"   Sharpe Ratio: {sharpe:.2f}\n"
        f"   Max Drawdown: {drawdown:.2%}"
    )
