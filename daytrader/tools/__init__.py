"""Agent tools for DayTrader.

This module provides tools that can be used by AI agents to interact
with the database, calculate indicators, search the web, and execute trades.
"""

from daytrader.tools.database import (
    query_candles,
    get_watchlist,
    get_positions as db_get_positions,
    get_trades,
    get_alerts,
)
from daytrader.tools.indicators import (
    calculate_indicators,
    find_support_resistance,
)
from daytrader.tools.web import (
    web_search,
    search_stock_news,
    search_company_info,
)
from daytrader.tools.broker import (
    set_broker,
    get_broker,
    place_order,
    get_positions,
    get_balance,
    get_quote,
    cancel_order,
    exit_position,
    exit_all_positions,
)

__all__ = [
    # Database tools
    "query_candles",
    "get_watchlist",
    "db_get_positions",
    "get_trades",
    "get_alerts",
    # Indicator tools
    "calculate_indicators",
    "find_support_resistance",
    # Web tools
    "web_search",
    "search_stock_news",
    "search_company_info",
    # Broker tools
    "set_broker",
    "get_broker",
    "place_order",
    "get_positions",
    "get_balance",
    "get_quote",
    "cancel_order",
    "exit_position",
    "exit_all_positions",
]
