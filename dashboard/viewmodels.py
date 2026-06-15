"""Pure Python dataclasses for dashboard view models — zero HTML."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PositionRow:
    symbol: str
    shares: float
    invested: float
    cur_value: float
    pnl_dollar: float
    pnl_pct: float
    pnl_color: str
    weight_pct: float
    target_pct: float
    action: str
    action_size: str   # "small" | "normal" | "large"
    confidence: int
    reason: str
    sell_score: int
    score_color: str


@dataclass
class TradeRow:
    timestamp: str
    symbol: str
    action: str
    action_color: str
    shares: float
    price: float
    notional: float
    pnl_pct: Optional[float]   # None for BUY trades
    pnl_color: str
    regime: str


@dataclass
class HealthComponent:
    label: str
    score: int
    max: int
    color: str
    pct: int
    detail: str = ""


@dataclass
class HealthViewModel:
    components: list[HealthComponent]
    total: int
    grade: str
    grade_label: str
    biggest_risk: str
    biggest_risk_color: str
    strengths: list[str] = field(default_factory=list)


@dataclass
class ActionRow:
    number: int
    symbol: str
    action: str
    badge_size: str
    reason: str
    detail: str
    urgency: str        # "high" | "medium" | "low"
    row_bg: str
    row_border: str
    sym_color: str
    rsn_color: str
    confidence: int = 0


@dataclass
class DecisionRow:
    symbol: str
    action: str
    sell_score: int
    score_color: str
    cur_weight: float
    tgt_weight: float
    delta_weight: float
    delta_color: str
    dollar_display: str
    reasons_sell: list[str]
    reasons_hold: list[str]
    pa_reason: str


@dataclass
class RebalanceRow:
    symbol: str
    cur_weight: float
    tgt_weight: float
    delta_weight: float
    delta_color: str
    badge_action: str
    dollar_display: str
    delta_dollars: float   # for net_rebalance calculation


@dataclass
class CommitteeMember:
    name: str
    vote: str      # "BUY" | "HOLD" | "SELL"
    color: str


@dataclass
class CommitteeViewModel:
    symbol: str
    members: list[CommitteeMember]
    buy_votes: int
    hold_votes: int
    sell_votes: int
    final_vote: str
    final_color: str
    confidence: int
    no_data: bool = False


@dataclass
class PerfPeriod:
    label: str
    value: str
    color: str


@dataclass
class PerformanceViewModel:
    periods: list[PerfPeriod]
    win_rate: str
    total_trades: int


@dataclass
class ChangedItem:
    symbol: str
    field: str
    old_val: str
    new_val: str
    direction: str   # "up" | "down" | "neutral"
