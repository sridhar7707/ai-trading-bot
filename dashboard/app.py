"""Gradio dashboard — read-only portfolio view, hosted on HuggingFace Spaces."""
import sqlite3
import pandas as pd
import gradio as gr
from loguru import logger

DB_PATH = "trades.db"


def load_recent_trades(n: int = 10) -> pd.DataFrame:
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT timestamp, symbol, action, price, pnl_pct, portfolio_value, regime FROM trades ORDER BY id DESC LIMIT ?",
            con, params=(n,),
        )
        con.close()
        df["pnl_pct"] = df["pnl_pct"].map(lambda x: f"{x:+.2%}" if x else "—")
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "symbol", "action", "price", "pnl_pct", "portfolio_value", "regime"])


def load_open_positions() -> pd.DataFrame:
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute("SELECT symbol, action FROM trades ORDER BY id").fetchall()
        con.close()
        holdings: dict[str, int] = {}
        for symbol, action in rows:
            if action == "BUY":
                holdings[symbol] = holdings.get(symbol, 0) + 1
            elif action in ("SELL", "SELL_STOP"):
                holdings[symbol] = max(0, holdings.get(symbol, 0) - 1)
        open_pos = [sym for sym, count in holdings.items() if count > 0]
        return pd.DataFrame({"Symbol": open_pos}) if open_pos else pd.DataFrame({"Symbol": ["None"]})
    except Exception:
        return pd.DataFrame({"Symbol": ["No data"]})


def load_summary() -> str:
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT portfolio_value, regime FROM trades ORDER BY id DESC LIMIT 1").fetchone()
        total_trades = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        con.close()
        if row:
            return f"Portfolio: ${row[0]:,.2f} | Regime: {row[1] or 'Unknown'} | Total trades: {total_trades}"
        return "No data yet."
    except Exception:
        return "Database not found."


with gr.Blocks(title="AI Trading Bot Dashboard") as demo:
    gr.Markdown("# AI Trading Bot — Live Dashboard")
    gr.Markdown("> Paper trading only. Read-only view.")

    with gr.Row():
        summary = gr.Textbox(label="Portfolio Summary", value=load_summary, every=60)

    with gr.Row():
        positions_table = gr.DataFrame(value=load_open_positions, label="Open Positions", every=60)
        trades_table = gr.DataFrame(value=load_recent_trades, label="Last 10 Trades", every=60)

    gr.Markdown("Refreshes every 60 seconds.")

if __name__ == "__main__":
    demo.launch()
