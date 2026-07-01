"""TradeGenius dashboard package &mdash; re-exports all render functions."""
from dashboard.components.overview import (
    render_portfolio_health_hero,
)
from dashboard.components.market_mood import render_market_mood
from dashboard.components.ai_panel import (
    render_ai_recommendation,
    render_ai_committee,
)
from dashboard.components.risk import (
    render_risk_panel,
    render_market_intelligence,
)
from dashboard.components.portfolio import (
    render_positions,
    render_trades,
)
from dashboard.components.models import (
    render_validation_report,
    render_institutional_metrics,
    render_investor_view,
)
from dashboard.components.signals import (
    render_watchlist,
    render_timeline,
)
from dashboard.components.history import (
    render_whats_changed,
    render_portfolio_performance,
    _perf_choices,
)
from dashboard.components.actions import (
    render_todays_actions,
    render_portfolio_actions,
)
from dashboard.components.analysis import (
    render_sell_analysis,
    render_position_sizing_panel,
    render_position_sizing,
)
from dashboard.components.decision import render_decision_center
from dashboard.components.rebalance import render_rebalance
from dashboard.components.symbol_detail import (
    render_symbol_detail,
    _get_symbol_choices,
)
from dashboard.charts import (
    render_equity_chart,
    render_allocation_chart,
    render_pnl_chart,
    render_feature_importance_chart,
)
