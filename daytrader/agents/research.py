"""Research Agent for stock research and analysis.

This agent performs web research on stocks using search tools
to find news, analyst opinions, and company information.
"""

from pathlib import Path
from typing import Optional

from agents import Agent, function_tool

from daytrader.agents.base import create_agent, run_agent_sync
from daytrader.db.store import DataStore
from daytrader.tools.web import (
    search_company_info,
    search_stock_news,
    web_search,
)


# Default database path
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"


RESEARCH_AGENT_INSTRUCTIONS = """You are a stock research analyst specializing in Indian equity markets.
Your role is to research stocks and provide comprehensive analysis.

When researching a stock:
1. Search for recent news and developments
2. Look for analyst opinions and price targets
3. Find information about company fundamentals
4. Identify key events that may impact the stock

Always provide:
- A summary of recent news and sentiment
- Key analyst opinions if available
- Important upcoming events or catalysts
- Your overall assessment of the stock's outlook

Be factual and cite sources when possible. Focus on information relevant to trading decisions.
"""


@function_tool
def web_search_tool(
    query: str,
    max_results: int = 5,
) -> dict:
    """Search the web for information.
    
    Use this tool to search for general information about stocks,
    markets, or any trading-related topics.
    
    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        
    Returns:
        Search results with titles, URLs, and content snippets.
    """
    return web_search(
        query=query,
        max_results=max_results,
        cache_results=True,
    )


@function_tool
def get_company_info_tool(
    symbol: str,
    company_name: Optional[str] = None,
    include_financials: bool = False,
) -> dict:
    """Get company information and analysis.
    
    Use this tool to find detailed information about a company,
    including profile, analysis, and optionally financial data.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        company_name: Optional company name for better search results.
        include_financials: Whether to include financial data.
        
    Returns:
        Company information and analysis results.
    """
    return search_company_info(
        symbol=symbol,
        company_name=company_name,
        include_financials=include_financials,
    )


@function_tool
def get_stock_news_tool(
    symbol: str,
    company_name: Optional[str] = None,
    max_results: int = 10,
) -> dict:
    """Get recent news about a stock.
    
    Use this tool to find recent news articles about a specific stock.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        company_name: Optional company name for better search results.
        max_results: Maximum number of news articles to return.
        
    Returns:
        Recent news articles about the stock.
    """
    return search_stock_news(
        symbol=symbol,
        company_name=company_name,
        max_results=max_results,
    )


class ResearchAgent:
    """Agent for performing stock research using web search.
    
    This agent uses web search tools to find news, analyst opinions,
    and company information for stocks.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the Research Agent.
        
        Args:
            db_path: Optional path to database for caching.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._store = DataStore(self.db_path)
        self._agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the underlying agent."""
        return create_agent(
            name="Research Agent",
            instructions=RESEARCH_AGENT_INSTRUCTIONS,
            tools=[
                web_search_tool,
                get_company_info_tool,
                get_stock_news_tool,
            ],
        )
    
    def research(
        self,
        symbol: str,
        deep: bool = False,
        company_name: Optional[str] = None,
    ) -> str:
        """Research a stock and return analysis.
        
        Args:
            symbol: Stock symbol to research.
            deep: Whether to perform deep research including financials.
            company_name: Optional company name for better search results.
            
        Returns:
            Research analysis as a string.
        """
        # Check cache first
        cache_key = f"{symbol.upper()}_{'deep' if deep else 'basic'}"
        cached = self._store.get_cached_research(cache_key, max_age_hours=6)
        if cached:
            return cached
        
        # Build research prompt
        if deep:
            prompt = f"""Please perform comprehensive research on {symbol.upper()}.
            
Include:
1. Recent news and developments
2. Analyst opinions and price targets
3. Company fundamentals and financials
4. Key events and catalysts
5. Technical outlook if relevant

{"Company name: " + company_name if company_name else ""}

Provide a detailed analysis suitable for making trading decisions."""
        else:
            prompt = f"""Please research {symbol.upper()} and provide a summary.

Include:
1. Recent news and sentiment
2. Key analyst opinions
3. Important upcoming events

{"Company name: " + company_name if company_name else ""}

Keep the analysis concise but informative."""
        
        # Run the agent
        result = run_agent_sync(self._agent, prompt)
        
        # Cache the result
        self._store.cache_research(
            symbol=cache_key,
            content=result,
            source="research_agent",
        )
        
        return result
    
    def ask(self, query: str) -> str:
        """Ask the research agent a question.
        
        Args:
            query: Question to ask.
            
        Returns:
            Agent's response.
        """
        return run_agent_sync(self._agent, query)
