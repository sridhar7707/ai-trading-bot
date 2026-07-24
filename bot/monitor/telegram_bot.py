"""Telegram alert gateway.

Message schedule (maximum):
  • 1 message per trading day  — alert_daily_summary (market close)
  • 1 message per week         — alert_weekly_report (Friday EOD)
  • Emergency only (rare)      — alert_daily_loss_limit, alert_weekly_loss_limit,
                                 alert_vix_halt

All per-trade and operational alerts are intentionally suppressed here and
logged locally at DEBUG level so callers need no changes.
"""
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

_CDT = timezone(timedelta(hours=-5))


def _now_cdt() -> str:
    t = datetime.now(_CDT)
    return t.strftime("%I:%M %p CDT").lstrip("0")


def _send(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug(f"[Telegram disabled] {text[:80]}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as exc:
        logger.error(f"Telegram send failed: {exc}")


# ── Suppressed — logged locally only ─────────────────────────────────────────
# Callers are unchanged; these just no longer send a Telegram message.

def alert_bot_started(mode: str, portfolio: float) -> None:
    logger.debug(f"[tg-suppressed] bot_started mode={mode} portfolio=${portfolio:,.2f}")


def alert_buy(symbol: str, shares: float, price: float, regime: str, portfolio: float,
              vs_spy: float, notional: float = 0.0,
              xgb_prob: float = 0.0, lstm_prob: float = 0.0,
              sentiment_score: float = 0.0, ensemble_score: float = 0.0,
              drivers: Optional[list] = None,
              sector: str = "", sector_pct_after: float = 0.0,
              cash_pct_after: float = 0.0) -> None:
    logger.debug(f"[tg-suppressed] buy {symbol} @${price:.2f} conf={ensemble_score*100:.0f}%")


def alert_sell(symbol: str, shares: float, price: float, pnl_pct: float,
               reason: str = "signal", notional: float = 0.0,
               cash_freed_pct: float = 0.0) -> None:
    logger.debug(f"[tg-suppressed] sell {symbol} @${price:.2f} pnl={pnl_pct:+.1%} reason={reason}")


def alert_hold(symbol: str, regime: str) -> None:
    pass


def alert_stop_loss(symbol: str, pnl_pct: float, notional: float = 0.0) -> None:
    logger.debug(f"[tg-suppressed] stop_loss {symbol} pnl={pnl_pct:+.1%}")


def alert_sell_failed(symbol: str, reason: str = "stop-loss") -> None:
    logger.debug(f"[tg-suppressed] sell_failed {symbol} reason={reason}")


def alert_risk_warning(portfolio: float, pnl_pct: float) -> None:
    logger.debug(f"[tg-suppressed] risk_warning portfolio=${portfolio:,.2f} pnl={pnl_pct:+.1%}")


def alert_bot_offline() -> None:
    logger.debug("[tg-suppressed] bot_offline")


def alert_confidence_passed() -> None:
    logger.debug("[tg-suppressed] confidence_passed")


# ── Emergency alerts — fire immediately ───────────────────────────────────────

def alert_daily_loss_limit(portfolio: float, pnl_pct: float) -> None:
    _send(
        f"🚨 <b>DAILY LOSS LIMIT HIT</b> — Trading halted\n"
        f"   Portfolio: ${portfolio:,.2f}  ·  Day P&amp;L: {pnl_pct:+.2%}"
    )


def alert_weekly_loss_limit(portfolio: float, pnl_pct: float) -> None:
    _send(
        f"🚨 <b>WEEKLY LOSS LIMIT HIT</b> — No new buys until next week\n"
        f"   Portfolio: ${portfolio:,.2f}  ·  Week P&amp;L: {pnl_pct:+.2%}"
    )


def alert_vix_halt() -> None:
    _send("🔴 <b>VIX HALT</b> — VIX ≥ 40 · No new buys this cycle")


# ── Scheduled summaries ───────────────────────────────────────────────────────

def alert_daily_summary(
    day_return: float, vs_spy: float, positions: list, cash: float,
    trades: int, day_trades: int,
    portfolio_value: float = 0.0,
    best_trade: Optional[tuple] = None,
    worst_trade: Optional[tuple] = None,
    cash_pct: float = 0.0,
    health_score: int = 0,
    cycles_run: int = 0,
    failed_steps: Optional[list] = None,
    errors_today: int = 0,
) -> None:
    now    = datetime.now(_CDT)
    today  = now.strftime(f"%b {now.day}")
    trophy = " 🏆" if day_return > vs_spy else ""
    pv_str = f"${portfolio_value:,.2f}  " if portfolio_value > 0 else ""

    lines = [
        f"📊 <b>{today} — End of Day</b>",
        f"   {pv_str}{day_return:+.2%}  vs S&amp;P {vs_spy:+.2%}{trophy}",
        f"   Trades: {trades}  ·  Day trades used: {day_trades}/3",
    ]
    if best_trade:
        sym, pct, usd = best_trade
        lines.append(f"   Best:  {sym} {pct:+.1%} (${usd:+,.0f})")
    if worst_trade:
        sym, pct, usd = worst_trade
        lines.append(f"   Worst: {sym} {pct:+.1%} (${usd:+,.0f})")

    pos_str = ", ".join(positions) if positions else "None"
    h_str   = ""
    if health_score > 0:
        h_icon = "🟢" if health_score >= 75 else ("🟡" if health_score >= 50 else "🔴")
        h_str  = f"  ·  Health {h_icon} {health_score}/100"

    cash_str = f"${cash:,.0f} ({cash_pct:.0f}%)" if cash_pct > 0 else f"${cash:,.0f}"
    lines.append(f"   Positions: {pos_str}")
    lines.append(f"   Cash: {cash_str}{h_str}")

    # System health section
    sys_ok = not failed_steps and errors_today == 0
    sys_icon = "🟢" if sys_ok else "🔴"
    sys_parts = []
    if cycles_run > 0:
        sys_parts.append(f"{cycles_run} cycles")
    if failed_steps:
        sys_parts.append(f"failed: {', '.join(failed_steps)}")
    if errors_today > 0:
        sys_parts.append(f"{errors_today} error(s)")
    sys_str = "  ·  ".join(sys_parts) if sys_parts else "all clear"
    lines.append(f"   System: {sys_icon} {sys_str}")

    _send("\n".join(lines))


def alert_weekly_report(
    week_return: float, vs_spy: float, win_rate: float,
    sharpe: float, drawdown: float, extra: str = "",
) -> None:
    trophy = " 🏆" if week_return > vs_spy else ""
    _send(
        f"📈 <b>Weekly Report</b>\n"
        f"   Return:    {week_return:+.2%}  vs S&amp;P {vs_spy:+.2%}{trophy}\n"
        f"   Win Rate:  {win_rate:.0%}  ·  Sharpe: {sharpe:.2f}  ·  Drawdown: {drawdown:.1%}"
        + extra
    )


def send(text: str) -> None:
    """Ad-hoc message for callers that need direct Telegram access."""
    _send(text)
