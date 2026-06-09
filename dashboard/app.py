"""Gradio dashboard — read-only portfolio view, hosted on HuggingFace Spaces."""
import os
import shutil
import sqlite3
import pandas as pd
import gradio as gr
from loguru import logger

DB_PATH = "trades.db"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "ksri77/ai-trading-bot")


def _sync_db():
    """Pull latest trades.db from HuggingFace model repo."""
    if not HF_TOKEN or not HF_REPO_ID:
        return
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename="trades.db", token=HF_TOKEN, force_download=True)
        shutil.copy(cached, DB_PATH)
    except Exception as e:
        logger.warning(f"Could not sync trades.db from HuggingFace: {e}")


def _current_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch latest prices via yfinance history (most reliable across environments)."""
    if not symbols:
        return {}
    import yfinance as yf
    prices = {}
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period="1d")
            prices[sym] = float(hist["Close"].iloc[-1]) if len(hist) > 0 else 0.0
        except Exception as e:
            logger.warning(f"Price fetch failed for {sym}: {e}")
            prices[sym] = 0.0
    return prices


def load_open_positions() -> pd.DataFrame:
    _sync_db()
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT symbol, action, shares, price, notional FROM trades ORDER BY id"
        ).fetchall()
        con.close()

        # Build per-symbol position: shares held, total invested (weighted avg cost basis)
        pos: dict[str, dict] = {}
        for symbol, action, shares, buy_price, notional in rows:
            shares = shares or 0.0
            notional = notional or 0.0
            if action == "BUY":
                if symbol not in pos:
                    pos[symbol] = {"shares": 0.0, "invested": 0.0}
                pos[symbol]["shares"] += shares
                pos[symbol]["invested"] += notional
            elif action in ("SELL", "SELL_STOP"):
                if symbol in pos and pos[symbol]["shares"] > 0:
                    avg = pos[symbol]["invested"] / pos[symbol]["shares"]
                    pos[symbol]["shares"] = max(0.0, pos[symbol]["shares"] - shares)
                    pos[symbol]["invested"] = max(0.0, pos[symbol]["invested"] - avg * shares)

        open_syms = {s: d for s, d in pos.items() if d["shares"] > 0.001}
        if not open_syms:
            return pd.DataFrame({"Symbol": ["None"], "Shares": ["—"], "Invested": ["—"],
                                  "Current Value": ["—"], "P&L $": ["—"], "P&L %": ["—"]})

        prices = _current_prices(list(open_syms.keys()))

        out = []
        for sym, d in open_syms.items():
            cur_price = prices.get(sym, 0.0)
            cur_value = d["shares"] * cur_price
            invested = d["invested"]
            pnl = cur_value - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0
            out.append({
                "Symbol": sym,
                "Shares": round(d["shares"], 4),
                "Invested": f"${invested:.2f}",
                "Current Value": f"${cur_value:.2f}" if cur_price else "—",
                "P&L $": f"${pnl:+.2f}" if cur_price else "—",
                "P&L %": f"{pnl_pct:+.2f}%" if cur_price else "—",
            })
        return pd.DataFrame(out)
    except Exception as e:
        logger.warning(f"load_open_positions failed: {e}")
        return pd.DataFrame({"Symbol": ["Error"], "Shares": ["—"], "Invested": ["—"],
                              "Current Value": ["—"], "P&L $": ["—"], "P&L %": ["—"]})


def load_recent_trades(n: int = 10) -> pd.DataFrame:
    _sync_db()
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT timestamp, symbol, action, shares, price, notional, pnl_pct, regime "
            "FROM trades ORDER BY id DESC LIMIT ?",
            con, params=(n,),
        )
        con.close()
        df["pnl_pct"] = df["pnl_pct"].map(lambda x: f"{x:+.2%}" if x else "—")
        df["notional"] = df["notional"].map(lambda x: f"${x:.2f}" if x else "—")
        df.rename(columns={"notional": "value", "shares": "qty"}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "symbol", "action", "qty", "price", "value", "pnl_pct", "regime"])


def load_summary() -> str:
    _sync_db()
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT portfolio_value, regime FROM trades ORDER BY id DESC LIMIT 1").fetchone()
        total_trades = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        buys = con.execute("SELECT COUNT(*) FROM trades WHERE action='BUY'").fetchone()[0]
        sells = con.execute("SELECT COUNT(*) FROM trades WHERE action IN ('SELL','SELL_STOP')").fetchone()[0]
        con.close()
        if row:
            return (f"Alpaca Portfolio: ${row[0]:,.2f} | Regime: {row[1] or 'Unknown'} | "
                    f"Total trades: {total_trades} (B:{buys} / S:{sells})")
        return "No trades yet — bot starts at market open (9:30am EST, Mon–Fri)."
    except Exception:
        return "Waiting for first trade cycle..."


with gr.Blocks(title="AI Trading Bot Dashboard") as demo:
    gr.Markdown("# AI Trading Bot — Live Dashboard")
    gr.Markdown("> Paper trading only. Syncs with live trade data every 60 seconds.")

    with gr.Row():
        summary = gr.Textbox(label="Portfolio Summary", value=load_summary, every=60)

    with gr.Row():
        positions_table = gr.DataFrame(value=load_open_positions, label="Open Positions", every=60)

    with gr.Row():
        trades_table = gr.DataFrame(value=load_recent_trades, label="Last 10 Trades", every=60)

    gr.Markdown("Refreshes every 60 seconds. Data sourced live from the trading bot.")

if __name__ == "__main__":
    demo.launch()
