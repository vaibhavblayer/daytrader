"""News Agent for market news and sentiment analysis.

This agent fetches and analyzes news about stocks, providing
sentiment analysis and highlighting market-moving events.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

from agents import Agent, function_tool

from daytrader.agents.base import create_agent, run_agent_sync
from daytrader.db.store import DataStore
from daytrader.tools.web import search_stock_news, web_search


# Default database path
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"

# Valid sentiment values
VALID_SENTIMENTS = ("bullish", "bearish", "neutral")
SentimentType = Literal["bullish", "bearish", "neutral"]


NEWS_AGENT_INSTRUCTIONS = """You are a financial news analyst specializing in Indian equity markets.
Your role is to find, analyze, and summarize news that impacts stock prices.

When analyzing news:
1. Search for recent news about the stock or market
2. Identify key events that could impact prices
3. Analyze sentiment (bullish, bearish, or neutral)
4. Highlight corporate actions, earnings, and announcements

Always provide:
- Summary of recent news
- Sentiment classification (bullish/bearish/neutral) with confidence
- Key events and their potential impact
- Any upcoming catalysts to watch

Be objective and factual. Focus on information relevant to trading decisions.
"""


@function_tool
def search_news_tool(
    symbol: str,
    company_name: Optional[str] = None,
    max_results: int = 10,
) -> dict:
    """Search for recent news about a stock.
    
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


@function_tool
def search_market_events_tool(
    query: str,
    max_results: int = 10,
) -> dict:
    """Search for market events and announcements.
    
    Use this tool to find information about market events,
    economic data releases, and corporate announcements.
    
    Args:
        query: Search query for market events.
        max_results: Maximum number of results to return.
        
    Returns:
        Search results about market events.
    """
    # Add India market context to query
    enhanced_query = f"{query} India stock market NSE BSE"
    
    return web_search(
        query=enhanced_query,
        max_results=max_results,
        include_domains=[
            "moneycontrol.com",
            "economictimes.indiatimes.com",
            "livemint.com",
            "business-standard.com",
            "nseindia.com",
            "bseindia.com",
        ],
    )


def _analyze_sentiment_impl(text: str) -> dict:
    """Core sentiment analysis implementation.
    
    Args:
        text: News text to analyze.
        
    Returns:
        Sentiment classification with confidence score.
    """
    # Simple keyword-based sentiment analysis
    text_lower = text.lower()
    
    bullish_keywords = [
        "surge", "rally", "gain", "rise", "jump", "soar", "bullish",
        "upgrade", "buy", "outperform", "beat", "strong", "growth",
        "profit", "dividend", "expansion", "acquisition", "positive",
        "record high", "breakout", "momentum", "upside",
    ]
    
    bearish_keywords = [
        "fall", "drop", "decline", "plunge", "crash", "bearish",
        "downgrade", "sell", "underperform", "miss", "weak", "loss",
        "cut", "layoff", "debt", "negative", "concern", "risk",
        "record low", "breakdown", "downside", "warning",
    ]
    
    bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
    bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)
    
    total = bullish_count + bearish_count
    
    if total == 0:
        sentiment = "neutral"
        confidence = 0.5
    elif bullish_count > bearish_count:
        sentiment = "bullish"
        confidence = min(0.9, 0.5 + (bullish_count - bearish_count) * 0.1)
    elif bearish_count > bullish_count:
        sentiment = "bearish"
        confidence = min(0.9, 0.5 + (bearish_count - bullish_count) * 0.1)
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "bullish_signals": bullish_count,
        "bearish_signals": bearish_count,
    }


@function_tool
def analyze_sentiment_tool(text: str) -> dict:
    """Analyze sentiment of news text.
    
    Use this tool to classify the sentiment of news content
    as bullish, bearish, or neutral.
    
    Args:
        text: News text to analyze.
        
    Returns:
        Sentiment classification with confidence score.
    """
    return _analyze_sentiment_impl(text)


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text (public API).
    
    Args:
        text: Text to analyze.
        
    Returns:
        Sentiment analysis result.
    """
    return _analyze_sentiment_impl(text)


def classify_sentiment(text: str) -> tuple[SentimentType, float]:
    """Classify sentiment of text.
    
    Args:
        text: Text to analyze.
        
    Returns:
        Tuple of (sentiment, confidence).
    """
    result = _analyze_sentiment_impl(text)
    return result["sentiment"], result["confidence"]


class NewsAgent:
    """Agent for fetching and analyzing market news.
    
    This agent searches for news about stocks, analyzes sentiment,
    and identifies market-moving events.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the News Agent.
        
        Args:
            db_path: Optional path to database for caching.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._store = DataStore(self.db_path)
        self._agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the underlying agent."""
        return create_agent(
            name="News Agent",
            instructions=NEWS_AGENT_INSTRUCTIONS,
            tools=[
                search_news_tool,
                search_market_events_tool,
                analyze_sentiment_tool,
            ],
        )
    
    def get_news(self, symbol: str, company_name: Optional[str] = None) -> str:
        """Get and analyze news for a stock.
        
        Args:
            symbol: Stock symbol to get news for.
            company_name: Optional company name for better search.
            
        Returns:
            News summary with sentiment analysis.
        """
        # Check cache first
        cache_key = f"news_{symbol.upper()}"
        cached = self._store.get_cached_research(cache_key, max_age_hours=2)
        if cached:
            return cached
        
        prompt = f"""Find and analyze recent news for {symbol.upper()}.

{"Company name: " + company_name if company_name else ""}

Please:
1. Search for recent news about this stock
2. Analyze the sentiment of the news
3. Identify any key events or announcements
4. Provide an overall sentiment assessment

Include specific news items and their potential impact on the stock."""
        
        result = run_agent_sync(self._agent, prompt)
        
        # Cache the result
        self._store.cache_research(
            symbol=cache_key,
            content=result,
            source="news_agent",
        )
        
        return result
    
    def get_events(self, event_date: Optional[date] = None) -> str:
        """Get market events for a date.
        
        Args:
            event_date: Date to get events for. Defaults to today.
            
        Returns:
            Summary of market events.
        """
        target_date = event_date or date.today()
        date_str = target_date.strftime("%B %d, %Y")
        
        prompt = f"""Find market events and announcements for {date_str}.

Please search for:
1. Economic data releases
2. Corporate earnings announcements
3. IPOs and listings
4. Regulatory announcements
5. Any other market-moving events

Focus on events relevant to Indian stock markets (NSE/BSE)."""
        
        return run_agent_sync(self._agent, prompt)
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Sentiment analysis result.
        """
        return analyze_sentiment_tool(text)
    
    def ask(self, query: str) -> str:
        """Ask the news agent a question.
        
        Args:
            query: Question to ask.
            
        Returns:
            Agent's response.
        """
        return run_agent_sync(self._agent, query)
