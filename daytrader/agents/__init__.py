"""AI agents for DayTrader.

This module provides AI agents for various trading tasks:
- ResearchAgent: Web research and company analysis
- DataAnalystAgent: Technical analysis and indicators
- NewsAgent: News fetching and sentiment analysis
- TradingAgent: Trade execution with confirmation
- OrchestratorAgent: Query routing and multi-agent coordination
"""

from daytrader.agents.base import (
    create_agent,
    run_agent_sync,
    run_agent_async,
    get_model,
    get_api_key,
)
from daytrader.agents.research import ResearchAgent
from daytrader.agents.analyst import DataAnalystAgent
from daytrader.agents.news import NewsAgent, analyze_sentiment, classify_sentiment
from daytrader.agents.trader import TradingAgent
from daytrader.agents.orchestrator import OrchestratorAgent, classify_query

__all__ = [
    # Base utilities
    "create_agent",
    "run_agent_sync",
    "run_agent_async",
    "get_model",
    "get_api_key",
    # Agents
    "ResearchAgent",
    "DataAnalystAgent",
    "NewsAgent",
    "TradingAgent",
    "OrchestratorAgent",
    # Utility functions
    "analyze_sentiment",
    "classify_sentiment",
    "classify_query",
]
