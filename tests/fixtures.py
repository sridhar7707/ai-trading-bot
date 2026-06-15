"""
Mock portfolio states for testing.
All prices and values are fictional.
"""

# ── Healthy balanced portfolio ────────────────
HEALTHY_PORTFOLIO = {
    "portfolio": "$42,350.00",
    "open_pos": {
        "MU": {
            "shares":  50.0,
            "invested": 5000.0,
        },
        "NVDA": {
            "shares":  15.0,
            "invested": 3000.0,
        },
        "GLD": {
            "shares":  20.0,
            "invested": 2000.0,
        },
    },
    "prices": {
        "MU":   130.0,
        "NVDA": 280.0,
        "GLD":  185.0,
    },
    "vix": 15.0,
    "latest_buy_signal": {
        "symbol":          "NVDA",
        "ensemble_score":  0.82,
        "xgb_prob":        0.80,
        "lstm_prob":       0.78,
        "sentiment_score": 0.15,
        "regime":          "bull_trending",
        "price":           280.0,
    },
    "trades_df":            None,
    "recent_trades":        [],
    "total_trades":         42,
    "win_rate":             0.64,
}

# ── Oversized single position ─────────────────
# MU is 73% of portfolio — should trigger TRIM/EXIT
OVERSIZED_POSITION = {
    "portfolio": "$42,350.00",
    "open_pos": {
        "MU": {
            "shares":  240.0,
            "invested": 28000.0,
        },
        "NVDA": {
            "shares":  10.0,
            "invested": 2000.0,
        },
    },
    "prices": {
        "MU":   130.0,
        "NVDA": 280.0,
    },
    "vix": 16.0,
    "latest_buy_signal": {},
    "trades_df":         None,
    "recent_trades":     [],
    "total_trades":      20,
    "win_rate":          0.60,
}

# ── 100% cash ─────────────────────────────────
ALL_CASH = {
    "portfolio": "$42,350.00",
    "open_pos":  {},
    "prices":    {},
    "vix":       14.0,
    "latest_buy_signal": {},
    "trades_df":         None,
    "recent_trades":     [],
    "total_trades":      0,
    "win_rate":          0.0,
}

# ── 100% invested, no cash ────────────────────
ALL_INVESTED = {
    "portfolio": "$42,350.00",
    "open_pos": {
        "MU":   {"shares": 100.0, "invested": 13000.0},
        "NVDA": {"shares":  50.0, "invested": 14000.0},
        "APLD": {"shares": 500.0, "invested": 8000.0},
        "GLD":  {"shares":  40.0, "invested": 7350.0},
    },
    "prices": {
        "MU":   130.0,
        "NVDA": 280.0,
        "APLD":  16.0,
        "GLD":  185.0,
    },
    "vix": 18.0,
    "latest_buy_signal": {},
    "trades_df":         None,
    "recent_trades":     [],
    "total_trades":      55,
    "win_rate":          0.58,
}

# ── High VIX / stressed market ────────────────
HIGH_VIX = {
    "portfolio": "$42,350.00",
    "open_pos": {
        "MU":   {"shares": 50.0, "invested": 5000.0},
        "NVDA": {"shares": 15.0, "invested": 3000.0},
    },
    "prices": {
        "MU":   130.0,
        "NVDA": 280.0,
    },
    "vix": 32.0,
    "latest_buy_signal": {
        "symbol":         "MU",
        "ensemble_score": 0.51,
        "xgb_prob":       0.52,
        "lstm_prob":      0.50,
        "sentiment_score": -0.08,
        "regime":         "bear_trending",
        "price":          130.0,
    },
    "trades_df":     None,
    "recent_trades": [],
    "total_trades":  30,
    "win_rate":      0.48,
}

# ── Single stock portfolio ────────────────────
SINGLE_STOCK = {
    "portfolio": "$42,350.00",
    "open_pos": {
        "MU": {"shares": 325.0, "invested": 42350.0},
    },
    "prices": {"MU": 130.0},
    "vix": 17.0,
    "latest_buy_signal": {},
    "trades_df":         None,
    "recent_trades":     [],
    "total_trades":      5,
    "win_rate":          0.60,
}

# ── Large gain position (profit taking) ───────
LARGE_GAIN = {
    "portfolio": "$42,350.00",
    "open_pos": {
        "MU": {
            "shares":   50.0,
            "invested":  2000.0,
        },
    },
    "prices": {"MU": 130.0},
    "vix": 15.0,
    "latest_buy_signal": {
        "symbol":          "MU",
        "ensemble_score":  0.78,
        "xgb_prob":        0.75,
        "lstm_prob":       0.72,
        "sentiment_score": 0.10,
        "regime":          "bull_trending",
        "price":           130.0,
    },
    "trades_df":     None,
    "recent_trades": [],
    "total_trades":  10,
    "win_rate":      0.70,
}

# ── Large loss position (stop loss) ───────────
LARGE_LOSS = {
    "portfolio": "$42,350.00",
    "open_pos": {
        "APLD": {
            "shares":   500.0,
            "invested": 12000.0,
        },
    },
    "prices": {"APLD": 12.0},
    "vix": 22.0,
    "latest_buy_signal": {
        "symbol":          "APLD",
        "ensemble_score":  0.42,
        "xgb_prob":        0.40,
        "lstm_prob":       0.38,
        "sentiment_score": -0.20,
        "regime":          "bear_trending",
        "price":           12.0,
    },
    "trades_df":     None,
    "recent_trades": [],
    "total_trades":  15,
    "win_rate":      0.45,
}

# ── High concentration in one sector ──────────
HIGH_CONCENTRATION = {
    "portfolio": "$42,350.00",
    "open_pos": {
        "MU":   {"shares":  80.0, "invested": 10000.0},
        "NVDA": {"shares":  50.0, "invested": 14000.0},
        "AMD":  {"shares": 100.0, "invested": 12000.0},
        "INTC": {"shares": 200.0, "invested":  6350.0},
    },
    "prices": {
        "MU":   130.0, "NVDA": 280.0,
        "AMD":  120.0, "INTC":  32.0,
    },
    "vix": 16.0,
    "latest_buy_signal": {},
    "trades_df":         None,
    "recent_trades":     [],
    "total_trades":      30,
    "win_rate":          0.55,
}

# ── Tiny portfolio ────────────────────────────
TINY_PORTFOLIO = {
    "portfolio": "$500.00",
    "open_pos": {
        "MU": {"shares": 1.0, "invested": 130.0},
    },
    "prices": {"MU": 130.0},
    "vix": 15.0,
    "latest_buy_signal": {},
    "trades_df":         None,
    "recent_trades":     [],
    "total_trades":      1,
    "win_rate":          1.0,
}
