"""DB helpers for the user-facing high-confidence signal_history table."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import bot.monitor.telegram_bot as tg
from loguru import logger


def init_signal_history(con: sqlite3.Connection) -> None:
    """Create signal_history table. Called once from init_db()."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            entry_price   REAL,
            stop_price    REAL,
            target_price  REAL,
            rr_ratio      REAL,
            setup_type    TEXT,
            xgb_prob      REAL,
            lstm_prob     REAL,
            ensemble_score REAL,
            macro_score   REAL,
            outcome       TEXT DEFAULT 'pending',
            outcome_price REAL,
            outcome_pct   REAL,
            outcome_ts    TEXT
        )
    """)
    con.commit()


def record_signal(
    con: sqlite3.Connection,
    symbol: str,
    meta: dict,
    xgb_prob: float,
    lstm_prob: float,
    ensemble_score: float,
    macro_score: float,
) -> None:
    """Insert a new high-confidence signal and fire an enhanced Telegram alert."""
    ts = datetime.now(timezone.utc).isoformat()
    con.execute("""
        INSERT INTO signal_history
            (timestamp, symbol, entry_price, stop_price, target_price,
             rr_ratio, setup_type, xgb_prob, lstm_prob, ensemble_score, macro_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, symbol,
        meta["entry_price"], meta["stop_price"], meta["target_price"],
        meta["rr_ratio"], meta["setup_type"],
        xgb_prob, lstm_prob, ensemble_score, macro_score,
    ))
    con.commit()
    logger.info(
        f"[SIGNAL GATE] High-confidence signal recorded: {symbol} "
        f"entry={meta['entry_price']:.2f} stop={meta['stop_price']:.2f} "
        f"target={meta['target_price']:.2f} R:R={meta['rr_ratio']:.2f}"
    )
    _send_signal_alert(symbol, meta, xgb_prob, lstm_prob, macro_score)


def _send_signal_alert(
    symbol: str, meta: dict,
    xgb_prob: float, lstm_prob: float, macro_score: float,
) -> None:
    """Send a structured Telegram alert with entry / stop / target."""
    setup  = meta.get("setup_type", "").replace("_", " ").title()
    entry  = meta["entry_price"]
    stop   = meta["stop_price"]
    target = meta["target_price"]
    rr     = meta["rr_ratio"]
    vol    = meta.get("volume_ratio", 0.0)
    spy    = meta.get("spy_today_pct", 0.0)

    stop_pct   = (stop   - entry) / entry * 100
    target_pct = (target - entry) / entry * 100
    spy_emoji  = "📈" if spy > 0 else "📉"

    tg._send(
        f"🎯 <b>HIGH-CONFIDENCE SIGNAL — ${symbol}</b>  [{setup}]\n\n"
        f"<b>Entry:  </b>${entry:.2f}\n"
        f"<b>Stop:   </b>${stop:.2f}  ({stop_pct:+.1f}%)\n"
        f"<b>Target: </b>${target:.2f}  ({target_pct:+.1f}%)\n"
        f"<b>R:R:    </b>{rr:.1f} : 1\n\n"
        f"🤖 XGB {xgb_prob*100:.0f}%  LSTM {lstm_prob*100:.0f}%  Macro {macro_score:.2f}\n"
        f"📊 Volume {vol:.1f}× avg  {spy_emoji} SPY {spy:+.2%} today\n\n"
        f"<i>All 7 confidence gates passed. Exit at stop if thesis fails.</i>"
    )


def update_signal_outcomes(
    con: sqlite3.Connection,
    prices: dict[str, float],
) -> None:
    """Resolve pending signals against current prices.

    target_hit  — price reached the target (WIN)
    stop_hit    — price fell to the stop (LOSS)
    expired     — signal is >7 calendar days old with no resolution (NEUTRAL)
    """
    rows = con.execute("""
        SELECT id, symbol, entry_price, stop_price, target_price, timestamp
        FROM signal_history WHERE outcome = 'pending'
    """).fetchall()

    for row in rows:
        sid, symbol, entry, stop, target, ts_str = row
        cur = prices.get(symbol)
        if cur is None:
            continue

        outcome = None
        if cur >= target:
            outcome = "target_hit"
        elif cur <= stop:
            outcome = "stop_hit"
        else:
            try:
                ts  = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - ts).days
                if age >= 7:
                    outcome = "expired"
            except Exception:
                pass

        if outcome:
            pct = (cur - entry) / entry if entry > 0 else 0.0
            now = datetime.now(timezone.utc).isoformat()
            con.execute("""
                UPDATE signal_history
                SET outcome=?, outcome_price=?, outcome_pct=?, outcome_ts=?
                WHERE id=?
            """, (outcome, cur, pct, now, sid))
            emoji = "✅" if outcome == "target_hit" else ("❌" if outcome == "stop_hit" else "⏰")
            logger.info(
                f"[SIGNAL GATE] {symbol} → {outcome} {emoji} "
                f"at ${cur:.2f} ({pct:+.1%})"
            )

    con.commit()
