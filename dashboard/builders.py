"""Builder functions: fetch data, compute all display decisions, return view models."""
from __future__ import annotations

from typing import Optional

from dashboard.design_system import (
    TEXT1, TEXT2,
    ACTION_BUY, ACTION_SELL, ACTION_TRIM, ACTION_HOLD, ACTION_WATCH,
    ACTION_BUY_BG, ACTION_SELL_BG, ACTION_TRIM_BG,
    ACTION_HOLD_BG, ACTION_WATCH_BG,
    GAIN, LOSS, NEURAL,
    SURFACE2,
)
import datetime
from config import STOP_LOSS_PCT as _STOP_LOSS_PCT, ATR_TRAIL_MULTIPLIER as _ATR_TRAIL_MULT
from dashboard.data import get_data, _to_ct, safe_query
from dashboard.viewmodels import (
    PositionRow, TradeRow, HealthComponent, HealthViewModel,
    ActionRow, DecisionRow, RebalanceRow,
    CommitteeMember, CommitteeViewModel,
)
from bot.core.recommendation_engine import (
    get_portfolio_action, get_position_sizing, get_sell_analysis, get_portfolio_health,
)


# ── Color helpers ─────────────────────────────────────────────────────────────

def _action_color(action: str) -> str:
    return {
        "EXIT": LOSS, "SELL": LOSS,
        "TRIM": ACTION_TRIM,
        "BUY":  GAIN, "ADD":  GAIN,
        "WATCH": NEURAL, "HOLD": ACTION_HOLD,
    }.get(action, TEXT2)


def _action_bg(action: str) -> str:
    return {
        "EXIT": ACTION_SELL_BG, "SELL": ACTION_SELL_BG,
        "TRIM": ACTION_TRIM_BG,
        "BUY":  ACTION_BUY_BG,  "ADD":  ACTION_BUY_BG,
        "WATCH": ACTION_WATCH_BG, "HOLD": ACTION_HOLD_BG,
    }.get(action, SURFACE2)


def _score_color(score: int) -> str:
    return LOSS if score > 65 else (ACTION_TRIM if score > 35 else GAIN)


def _pnl_color(pct: float) -> str:
    return GAIN if pct >= 0 else LOSS


def _health_color(pct: int) -> str:
    return GAIN if pct >= 70 else (NEURAL if pct >= 40 else LOSS)


# ── Position / Trades builders ────────────────────────────────────────────────

def build_positions_vm() -> list[PositionRow]:
    d         = get_data()
    open_syms = d.get("open_pos", {})
    prices    = d.get("prices", {})

    # Fetch opened_at, entry_price, high_water_mark, atr_at_entry from position_state
    _opened_at:   dict[str, str]   = {}
    _entry_price: dict[str, float] = {}
    _hwm:         dict[str, float] = {}
    _atr_entry:   dict[str, float] = {}
    try:
        rows = safe_query(
            "SELECT symbol, opened_at, entry_price, high_water_mark, atr_at_entry "
            "FROM position_state",
            default=[],
        )
        for sym, ts, ep, hwm, atr in (rows or []):
            if ts:  _opened_at[sym]   = str(ts)
            if ep:  _entry_price[sym] = float(ep)
            if hwm: _hwm[sym]         = float(hwm)
            if atr: _atr_entry[sym]   = float(atr)
    except Exception:
        pass

    _today = datetime.date.today()
    _pv = 0.0
    try:
        _pv = float(d["portfolio"].replace("$", "").replace(",", "")) if d.get("portfolio", "&mdash;") != "&mdash;" else 0.0
    except Exception:
        pass

    rows: list[PositionRow] = []
    for sym, v in open_syms.items():
        cur_price = prices.get(sym, 0.0)
        cur_val   = v["shares"] * cur_price if cur_price > 0 else v["invested"]
        invested  = v["invested"]
        pnl_d     = cur_val - invested
        pnl_pct   = (pnl_d / invested * 100) if invested > 0 else 0.0
        weight    = (cur_val / _pv * 100) if _pv > 0 else 0.0

        pa     = get_portfolio_action(sym, d)
        sz     = get_position_sizing(sym, d)
        action = pa.get("action", "HOLD")
        conf   = int(pa.get("confidence", 0))
        reason = pa.get("reason", "&mdash;")
        tgt_w  = sz.get("target_weight", 0.0)

        sa         = get_sell_analysis(sym, d)
        sell_score = sa.get("sell_score", 0)

        _urgent = action in ("EXIT", "SELL")
        _medium = action in ("TRIM", "BUY", "ADD")
        action_size = (
            "large"  if _urgent or action in ("TRIM", "BUY") else
            "normal" if action in ("ADD", "WATCH") else
            "small"
        )

        days_held = 0
        opened_ts = _opened_at.get(sym)
        if opened_ts:
            try:
                opened_date = datetime.date.fromisoformat(str(opened_ts)[:10])
                days_held = (_today - opened_date).days
            except Exception:
                pass

        stop_price: Optional[float] = None
        ep  = _entry_price.get(sym)
        hwm = _hwm.get(sym)
        atr = _atr_entry.get(sym)
        if ep:
            flat_stop  = ep * (1.0 - _STOP_LOSS_PCT)
            trail_stop = (hwm - _ATR_TRAIL_MULT * atr) if (hwm and atr) else flat_stop
            stop_price = max(flat_stop, trail_stop)

        rows.append(PositionRow(
            symbol=sym,
            shares=v["shares"],
            invested=invested,
            cur_value=cur_val,
            pnl_dollar=pnl_d,
            pnl_pct=pnl_pct,
            pnl_color=_pnl_color(pnl_pct),
            weight_pct=weight,
            target_pct=tgt_w,
            action=action,
            action_size=action_size,
            confidence=conf,
            reason=reason,
            sell_score=sell_score,
            score_color=_score_color(sell_score),
            days_held=days_held,
            stop_price=stop_price,
        ))
    return rows


def build_trades_vm() -> list[TradeRow]:
    d   = get_data()
    raw = d.get("recent_trades", [])
    rows: list[TradeRow] = []
    for row in raw:
        ts, sym, action, shares, price, notional, pnl_pct_raw, regime = row
        pnl: Optional[float] = float(pnl_pct_raw) if pnl_pct_raw else None
        rows.append(TradeRow(
            timestamp=_to_ct(ts),
            symbol=sym,
            action=action,
            action_color=_action_color(action),
            shares=float(shares) if shares else 0.0,
            price=float(price) if price else 0.0,
            notional=float(notional) if notional else 0.0,
            pnl_pct=pnl,
            pnl_color=_pnl_color(pnl) if pnl is not None else TEXT2,
            regime=(regime or "&mdash;").replace("_", " ").title(),
        ))
    return rows


# ── Health builder ─────────────────────────────────────────────────────────────

def build_health_vm() -> HealthViewModel:
    d     = get_data()
    h     = get_portfolio_health(d)
    score = h.get("total", 0)
    grade = h.get("grade", "&mdash;")
    gl    = h.get("grade_label", "")
    risk_txt   = h.get("biggest_risk", "&mdash;")
    strengths  = h.get("strengths", [])[:2]
    comps = h.get("components", {})

    risk_c = LOSS if score < 60 else (NEURAL if score < 80 else GAIN)

    comp_order = ["risk", "diversification", "cash", "momentum", "quality"]
    components: list[HealthComponent] = []
    for k in comp_order:
        c      = comps.get(k, {})
        lbl    = c.get("label", k.title())
        pts    = c.get("score", 0)
        maxpts = c.get("max", 25)
        detail = c.get("detail", "")
        pct    = int(pts / maxpts * 100) if maxpts > 0 else 0
        components.append(HealthComponent(
            label=lbl, score=pts, max=maxpts,
            color=_health_color(pct), pct=pct, detail=detail,
        ))

    return HealthViewModel(
        components=components,
        total=score,
        grade=grade,
        grade_label=gl,
        biggest_risk=risk_txt,
        biggest_risk_color=risk_c,
        strengths=list(strengths),
    )


# ── Actions builder ────────────────────────────────────────────────────────────

_ACTION_ORDER = {"EXIT": 0, "SELL": 1, "TRIM": 2, "WATCH": 3, "ADD": 4, "BUY": 5, "HOLD": 6}

_URGENCY_ROW_BG = {
    "high":   ACTION_SELL_BG,
    "medium": ACTION_TRIM_BG,
    "low":    "transparent",
}
_URGENCY_BORDER = {
    "high":   ACTION_SELL,
    "medium": ACTION_TRIM,
    "low":    "transparent",
}


def build_actions_vm() -> list[ActionRow]:
    d        = get_data()
    open_pos = d.get("open_pos", {})
    recs: list[dict] = []
    for sym in open_pos:
        rec   = get_portfolio_action(sym, d)
        sz    = get_position_sizing(sym, d)
        action      = rec.get("action", "HOLD")
        dol_display = sz.get("dollar_display", "&mdash;")
        # Only show delta for actionable signals; HOLD/WATCH with a negative
        # delta reads as a loss to users &mdash; suppress it.
        if action in ("HOLD", "WATCH") and str(dol_display).startswith("-"):
            dol_display = "&mdash;"
        recs.append({
            "symbol":       sym,
            "action":       action,
            "confidence":   rec.get("confidence", 0),
            "reason":       rec.get("reason", "&mdash;"),
            "detail":       dol_display,
            "urgency":      rec.get("urgency", "low"),
        })
    recs.sort(key=lambda r: (_ACTION_ORDER.get(r["action"], 9), -r["confidence"]))

    rows: list[ActionRow] = []
    for i, r in enumerate(recs):
        action  = r["action"]
        urgency = r["urgency"]
        _urgent = action in ("EXIT", "SELL")
        _medium = action in ("TRIM", "BUY", "ADD")
        badge_size = (
            "large"  if _urgent or action in ("TRIM", "BUY") else
            "normal" if action in ("ADD", "WATCH") else
            "small"
        )
        conf   = r["confidence"]
        sym_c  = TEXT1 if (_urgent or _medium) else TEXT2
        rsn_c  = LOSS if urgency == "high" else (NEURAL if urgency == "medium" else TEXT2)
        rows.append(ActionRow(
            number=i + 1,
            symbol=r["symbol"],
            action=action,
            badge_size=badge_size,
            reason=r["reason"],
            detail=r["detail"],
            urgency=urgency,
            row_bg=_URGENCY_ROW_BG.get(urgency, "transparent"),
            row_border=_URGENCY_BORDER.get(urgency, "transparent"),
            sym_color=sym_c,
            rsn_color=rsn_c,
            confidence=conf,
        ))
    return rows


# ── Decision builder ───────────────────────────────────────────────────────────

_DECISION_ORDER = {"EXIT": 0, "SELL": 1, "TRIM": 2, "BUY": 3, "ADD": 4, "WATCH": 5, "HOLD": 6}


def build_decision_vm() -> list[DecisionRow]:
    d        = get_data()
    open_pos = d.get("open_pos", {})
    rows: list[DecisionRow] = []
    for sym in open_pos:
        pa = get_portfolio_action(sym, d)
        sa = get_sell_analysis(sym, d)
        sz = get_position_sizing(sym, d)
        action  = pa.get("action", "HOLD")
        score   = sa.get("sell_score", 0)
        cur_w   = sz.get("current_weight", 0.0)
        tgt_w   = sz.get("target_weight", 0.0)
        delta_w = sz.get("delta_weight", 0.0)
        dol     = sz.get("dollar_display", "&mdash;")

        # Reconcile target weight with portfolio action.
        # get_position_sizing() uses the stored-at-buy ensemble score which
        # can map to 0% target (exit) even when the bot's action is HOLD
        # (actual sell threshold is 40%, not 55%). Ensure the Decision Center
        # reflects what the bot is actually doing.
        if action == "HOLD":
            tgt_w   = cur_w   # no change &mdash; bot is holding the position
            delta_w = 0.0
            dol     = "&mdash;"
        elif action == "WATCH" and tgt_w < cur_w * 0.5:
            # WATCH means keep an eye on it, not exit. Cap reduction at 50%.
            tgt_w   = round(cur_w * 0.5, 1)
            delta_w = round(tgt_w - cur_w, 1)

        delta_c = GAIN if delta_w > 1 else (LOSS if delta_w < -1 else TEXT2)
        rows.append(DecisionRow(
            symbol=sym,
            action=action,
            sell_score=score,
            score_color=_score_color(score),
            cur_weight=cur_w,
            tgt_weight=tgt_w,
            delta_weight=delta_w,
            delta_color=delta_c,
            dollar_display=dol,
            reasons_sell=sa.get("reasons_to_sell", []),
            reasons_hold=sa.get("reasons_to_hold", []),
            pa_reason=pa.get("reason", ""),
        ))
    rows.sort(key=lambda r: (_DECISION_ORDER.get(r.action, 9), -r.sell_score))
    return rows


# ── Rebalance builder ──────────────────────────────────────────────────────────

def build_rebalance_vm() -> list[RebalanceRow]:
    d        = get_data()
    open_pos = d.get("open_pos", {})
    rows: list[RebalanceRow] = []
    for sym in open_pos:
        sz      = get_position_sizing(sym, d)
        cur_w   = sz.get("current_weight", 0.0)
        tgt_w   = sz.get("target_weight", 0.0)
        delta_w = sz.get("delta_weight", 0.0)
        delta_c = GAIN if delta_w > 1 else (LOSS if delta_w < -1 else TEXT2)
        sz_act  = sz.get("action", "hold").lower()
        badge   = "ADD" if sz_act == "add" else ("TRIM" if sz_act == "reduce" else "HOLD")
        rows.append(RebalanceRow(
            symbol=sym,
            cur_weight=cur_w,
            tgt_weight=tgt_w,
            delta_weight=delta_w,
            delta_color=delta_c,
            badge_action=badge,
            dollar_display=sz.get("dollar_display", "&mdash;"),
            delta_dollars=float(sz.get("delta_dollars", 0.0)),
        ))
    rows.sort(key=lambda r: -r.cur_weight)
    return rows


# ── Committee builders ─────────────────────────────────────────────────────────

def build_committee_vm(symbol: str) -> CommitteeViewModel:
    """5-member committee from latest_buy_signal (satisfies spec tests)."""
    d  = get_data()
    lb = d.get("latest_buy_signal", {})

    if not lb or lb.get("symbol") != symbol:
        return CommitteeViewModel(
            symbol=symbol, members=[], buy_votes=0, hold_votes=0, sell_votes=0,
            final_vote="HOLD", final_color=TEXT2, confidence=0, no_data=True,
        )

    xgb_p  = float(lb.get("xgb_prob",        0.0) or 0.0)
    lstm_p = float(lb.get("lstm_prob",        0.0) or 0.0)
    sent   = float(lb.get("sentiment_score",  0.0) or 0.0)
    vix    = float(d.get("vix",               0.0) or 0.0)
    r_lower = str(lb.get("regime") or "").lower()

    def _vote(v, thr=0.60):
        return "BUY" if v >= thr else ("HOLD" if v >= 0.45 else "SELL")

    sent_n = min(max((sent + 1) / 2, 0.0), 1.0)
    regime_v = "BUY" if any(x in r_lower for x in ["bull", "trending up"]) else (
               "SELL" if any(x in r_lower for x in ["bear", "trending down"]) else "HOLD")
    macro_v  = "BUY" if vix < 15 else ("HOLD" if vix < 25 else "SELL")

    members = [
        CommitteeMember("XGBoost",   _vote(xgb_p),        _action_color(_vote(xgb_p))),
        CommitteeMember("LSTM",      _vote(lstm_p),        _action_color(_vote(lstm_p))),
        CommitteeMember("Sentiment", _vote(sent_n, 0.55),  _action_color(_vote(sent_n, 0.55))),
        CommitteeMember("Regime",    regime_v,             _action_color(regime_v)),
        CommitteeMember("Macro",     macro_v,              _action_color(macro_v)),
    ]
    buy_v  = sum(1 for m in members if m.vote == "BUY")
    hold_v = sum(1 for m in members if m.vote == "HOLD")
    sell_v = sum(1 for m in members if m.vote == "SELL")
    final  = "BUY" if buy_v >= 3 else ("SELL" if sell_v >= 3 else "HOLD")
    conf   = int(float(lb.get("ensemble_score", 0.0) or 0.0) * 100)

    return CommitteeViewModel(
        symbol=symbol, members=members,
        buy_votes=buy_v, hold_votes=hold_v, sell_votes=sell_v,
        final_vote=final, final_color=_action_color(final),
        confidence=conf, no_data=False,
    )


def build_committees_vm() -> list[CommitteeViewModel]:
    """3-member committee per open position from trades_df (matches current UI)."""
    d        = get_data()
    open_pos = d.get("open_pos", {})
    df       = d.get("trades_df", None)

    _votes: dict[str, dict] = {}
    try:
        if df is not None and not df.empty:
            buys = df[df["action"] == "BUY"]
            for sym in open_pos:
                sym_buys = buys[buys["symbol"] == sym]
                if not sym_buys.empty:
                    lb = sym_buys.iloc[-1]
                    _votes[sym] = {
                        "xgb":  float(lb.get("xgb_prob",       0.0) or 0.0),
                        "lstm": float(lb.get("lstm_prob",       0.0) or 0.0),
                        "sent": float(lb.get("sentiment_score", 0.0) or 0.0),
                    }
    except Exception:
        pass

    vms: list[CommitteeViewModel] = []
    for sym in list(open_pos.keys())[:8]:
        v      = _votes.get(sym, {})
        no_data = not v
        xgb    = v.get("xgb",  0.0)
        lstm   = v.get("lstm", 0.0)
        sent   = v.get("sent", 0.0)
        sent_n = min(max((sent + 1) / 2, 0.0), 1.0)

        def _vote3(val, thr):
            return "BUY" if val >= thr else ("HOLD" if val >= 0.45 else "SELL")

        members = [] if no_data else [
            CommitteeMember("XGBoost",   _vote3(xgb, 0.60),   _action_color(_vote3(xgb, 0.60))),
            CommitteeMember("LSTM",      _vote3(lstm, 0.60),  _action_color(_vote3(lstm, 0.60))),
            CommitteeMember("Sentiment", _vote3(sent_n, 0.55), _action_color(_vote3(sent_n, 0.55))),
        ]
        buy_v = sum(1 for m in members if m.vote == "BUY")
        final_c = GAIN if buy_v >= 2 else (NEURAL if buy_v == 1 else LOSS)
        verdict = f"{buy_v}/3 BUY" if buy_v > 0 else "No BUY votes"
        vms.append(CommitteeViewModel(
            symbol=sym, members=members,
            buy_votes=buy_v, hold_votes=0, sell_votes=0,
            final_vote=verdict, final_color=final_c,
            confidence=0, no_data=no_data,
        ))
    return vms
