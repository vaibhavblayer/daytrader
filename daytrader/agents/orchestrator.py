"""Orchestrator Agent for query routing and multi-agent coordination.

This agent routes user queries to the appropriate specialized agents
and coordinates responses from multiple agents when needed.
"""

import re
from pathlib import Path
from typing import Literal, Optional

from agents import Agent, function_tool

from daytrader.agents.base import create_agent, run_agent_sync


# Default database path
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"

# Agent types for routing
AgentType = Literal["research", "analyst", "news", "trading", "orchestrator"]


ORCHESTRATOR_INSTRUCTIONS = """You are the orchestrator agent for a day trading assistant.
Your role is to understand user queries and route them to the appropriate specialized agents.

Available agents:
1. Research Agent - For web research, company information, analyst opinions
2. Data Analyst Agent - For technical analysis, indicators, chart patterns, price data
3. News Agent - For market news, sentiment analysis, events
4. Trading Agent - For executing trades, managing positions, checking balance

Routing guidelines:
- Research questions (company info, analyst views) -> Research Agent
- Technical analysis (RSI, MACD, support/resistance) -> Data Analyst Agent
- News and sentiment questions -> News Agent
- Trade execution, positions, balance -> Trading Agent

For complex queries that need multiple agents:
1. Identify which agents are needed
2. Call each agent in sequence
3. Combine their responses coherently

IMPORTANT: For any query that would execute a trade (buy, sell, exit):
- Always route to Trading Agent
- Ensure confirmation is required before execution
- Never execute trades without explicit user confirmation

Maintain conversation context to provide coherent follow-up responses.
"""


# Keywords for routing
RESEARCH_KEYWORDS = [
    "research", "company", "analyst", "opinion", "fundamentals",
    "financials", "earnings", "revenue", "profit", "management",
    "business", "sector", "industry", "competitor", "valuation",
]

ANALYST_KEYWORDS = [
    "technical", "indicator", "rsi", "macd", "ema", "sma", "bollinger",
    "support", "resistance", "pattern", "trend", "chart", "price",
    "oversold", "overbought", "breakout", "breakdown", "level",
    "analysis", "analyze", "candle", "volume", "atr", "vwap",
]

NEWS_KEYWORDS = [
    "news", "sentiment", "event", "announcement", "headline",
    "market", "today", "recent", "latest", "update", "happening",
]

TRADING_KEYWORDS = [
    "buy", "sell", "trade", "order", "position", "exit", "close",
    "balance", "margin", "portfolio", "pnl", "profit", "loss",
    "execute", "place", "cancel", "stop", "limit", "target",
]


def classify_query(query: str) -> list[AgentType]:
    """Classify a query to determine which agents should handle it.
    
    Args:
        query: User query string.
        
    Returns:
        List of agent types that should handle this query.
    """
    query_lower = query.lower()
    agents: list[AgentType] = []
    
    # Check for trading keywords first (highest priority for safety)
    if any(kw in query_lower for kw in TRADING_KEYWORDS):
        agents.append("trading")
    
    # Check for research keywords
    if any(kw in query_lower for kw in RESEARCH_KEYWORDS):
        agents.append("research")
    
    # Check for analyst keywords
    if any(kw in query_lower for kw in ANALYST_KEYWORDS):
        agents.append("analyst")
    
    # Check for news keywords
    if any(kw in query_lower for kw in NEWS_KEYWORDS):
        agents.append("news")
    
    # Default to analyst if no specific match
    if not agents:
        agents.append("analyst")
    
    return agents


def is_trading_query(query: str) -> bool:
    """Check if a query involves trading actions.
    
    Args:
        query: User query string.
        
    Returns:
        True if the query involves trading actions.
    """
    query_lower = query.lower()
    action_keywords = ["buy", "sell", "trade", "execute", "place order", "exit", "close position"]
    return any(kw in query_lower for kw in action_keywords)


def extract_symbol(query: str) -> Optional[str]:
    """Extract a stock symbol from a query.
    
    Args:
        query: User query string.
        
    Returns:
        Extracted symbol or None.
    """
    # Look for common patterns
    # Pattern 1: Explicit symbol mention
    patterns = [
        r'\b([A-Z]{2,10})\b',  # Uppercase words 2-10 chars
        r'symbol[:\s]+([A-Za-z]+)',  # "symbol: XYZ"
        r'stock[:\s]+([A-Za-z]+)',  # "stock: XYZ"
    ]
    
    query_upper = query.upper()
    
    # Common Indian stock symbols
    known_symbols = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
        "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN", "BAJFINANCE",
        "WIPRO", "HCLTECH", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND",
    ]
    
    for symbol in known_symbols:
        if symbol in query_upper:
            return symbol
    
    # Try regex patterns
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            potential_symbol = match.group(1).upper()
            # Filter out common words
            if potential_symbol not in ["THE", "AND", "FOR", "BUY", "SELL", "RSI", "MACD", "EMA", "SMA"]:
                return potential_symbol
    
    return None


class OrchestratorAgent:
    """Agent for routing queries and coordinating multiple agents.
    
    This agent determines which specialized agents should handle
    a query and coordinates their responses.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the Orchestrator Agent.
        
        Args:
            db_path: Optional path to database.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._conversation_context: list[dict] = []
        self._agents: dict = {}
        self._agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the underlying orchestrator agent."""
        return create_agent(
            name="Orchestrator Agent",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            tools=[],  # Orchestrator uses other agents, not tools directly
        )
    
    def _get_research_agent(self):
        """Get or create the Research Agent."""
        if "research" not in self._agents:
            from daytrader.agents.research import ResearchAgent
            self._agents["research"] = ResearchAgent(self.db_path)
        return self._agents["research"]
    
    def _get_analyst_agent(self):
        """Get or create the Data Analyst Agent."""
        if "analyst" not in self._agents:
            from daytrader.agents.analyst import DataAnalystAgent
            self._agents["analyst"] = DataAnalystAgent(self.db_path)
        return self._agents["analyst"]
    
    def _get_news_agent(self):
        """Get or create the News Agent."""
        if "news" not in self._agents:
            from daytrader.agents.news import NewsAgent
            self._agents["news"] = NewsAgent(self.db_path)
        return self._agents["news"]
    
    def _get_trading_agent(self):
        """Get or create the Trading Agent."""
        if "trading" not in self._agents:
            from daytrader.agents.trader import TradingAgent
            self._agents["trading"] = TradingAgent(self.db_path)
        return self._agents["trading"]
    
    def _add_to_context(self, role: str, content: str) -> None:
        """Add a message to conversation context.
        
        Args:
            role: Message role (user/assistant).
            content: Message content.
        """
        self._conversation_context.append({
            "role": role,
            "content": content,
        })
        # Keep only last 10 messages for context
        if len(self._conversation_context) > 10:
            self._conversation_context = self._conversation_context[-10:]
    
    def get_context(self) -> list[dict]:
        """Get the current conversation context.
        
        Returns:
            List of context messages.
        """
        return self._conversation_context.copy()
    
    def clear_context(self) -> None:
        """Clear the conversation context."""
        self._conversation_context = []
    
    def get_agent_for_query(self, query: str) -> list[AgentType]:
        """Determine which agent(s) should handle a query.
        
        Args:
            query: User query string.
            
        Returns:
            List of agent types.
        """
        return classify_query(query)
    
    def route_query(self, query: str) -> str:
        """Route a query to the appropriate agent(s) and return response.
        
        Args:
            query: User query string.
            
        Returns:
            Combined response from agent(s).
        """
        # Add query to context
        self._add_to_context("user", query)
        
        # Classify the query
        agents_needed = classify_query(query)
        
        # Extract symbol if present
        symbol = extract_symbol(query)
        
        responses = []
        
        # Route to each needed agent
        for agent_type in agents_needed:
            try:
                if agent_type == "research":
                    agent = self._get_research_agent()
                    if symbol:
                        response = agent.research(symbol)
                    else:
                        response = agent.ask(query)
                    responses.append(("Research", response))
                    
                elif agent_type == "analyst":
                    agent = self._get_analyst_agent()
                    if symbol:
                        response = agent.analyze(symbol, query)
                    else:
                        response = agent.ask(query)
                    responses.append(("Analysis", response))
                    
                elif agent_type == "news":
                    agent = self._get_news_agent()
                    if symbol:
                        response = agent.get_news(symbol)
                    else:
                        response = agent.ask(query)
                    responses.append(("News", response))
                    
                elif agent_type == "trading":
                    agent = self._get_trading_agent()
                    response = agent.ask(query)
                    responses.append(("Trading", response))
                    
            except Exception as e:
                responses.append((agent_type.title(), f"Error: {str(e)}"))
        
        # Combine responses
        if len(responses) == 1:
            combined = responses[0][1]
        else:
            combined_parts = []
            for label, response in responses:
                combined_parts.append(f"## {label}\n\n{response}")
            combined = "\n\n---\n\n".join(combined_parts)
        
        # Add response to context
        self._add_to_context("assistant", combined)
        
        return combined
    
    def ask(self, query: str) -> str:
        """Ask the orchestrator a question.
        
        This is the main entry point for user queries.
        
        Args:
            query: User query string.
            
        Returns:
            Response from appropriate agent(s).
        """
        return self.route_query(query)
